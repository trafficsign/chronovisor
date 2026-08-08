"""Discover line-independent runtime-state access and fail-closed escapes."""

from __future__ import annotations

import ast
import gc
import sys
import threading
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any

from .access_expressions import evaluate_generic_expression
from .access_facts import AccessFactCollector
from .access_imports import (
    build_module_exports,
    resolve_import_from,
    resolve_module_attribute,
)
from .access_model import (
    FILE_BOUND_CLOSE_OBJECT_TYPE,
    FILE_BOUND_FILENO_OBJECT_TYPE,
    FILE_HANDLE_OBJECT_TYPE,
    FILE_UNKNOWN_ATTRIBUTE_OBJECT_TYPE,
    PATH_BOUND_TRANSFORM_PREFIX,
    PATH_TRANSFORMS,
    SOCKET_BOUND_METHOD_PREFIX,
    SOCKETSERVER_BOUND_METHOD_PREFIX,
    SOCKETSERVER_CLASS_DICT_OBJECT_TYPE,
    SOCKETSERVER_LOCAL_CLASS_OBJECT_PREFIX,
    SQLITE_HANDLE_OBJECT_TYPES,
    SQLITE_UNKNOWN_ATTRIBUTE_OBJECT_TYPE,
    SUPPORTED_STDLIB_MODULES,
    UNRESOLVED_RUNTIME_OBJECT_ALTERNATIVE_TYPE,
    AnalysisLimits,
    AnalysisNonConvergenceError,
    AnalysisProgress,
    DataclassInfo,
    FlowValue,
    FunctionInfo,
    _call_ordinals,
    _collect_functions,
    _collect_syntax_sites,
    _import_tables,
    _module_name,
    candidate_flow_variants,
    contaminate_runtime_objects,
    exclusive_flow_join,
    is_path_receiver,
    precise_stdlib_module_name,
    project_sqlite_attribute,
    replace_runtime_object_aliases,
    socket_handle_kind,
    socketserver_handle_state,
    stdlib_call_target_marker,
    stdlib_module_dict_reference,
    stdlib_module_mutation_attributes,
    stdlib_module_mutation_marker,
    stdlib_module_state_name,
    with_candidate_flow_variants,
)
from .access_outcomes import analyze_block_result, join_states
from .access_resolver import call_name
from .access_sinks import evaluate_call
from .access_statements import analyze_block, evaluate_control_expression

_CALL_MODULE_STATE_PREFIX = "\0runtime-module-state:"
_CALL_CLOSURE_STATE_PREFIX = "\0runtime-closure-state:"
_GC_OPTIMIZATION_LOCK = threading.RLock()

_SOCKET_ORIGIN_ALIASES = {
    "unix://$HOME/.chronovisor/runtime/semantic.sock": (
        "chronovisor.core.runtime_config",
        "SearchEmbeddingConfig.socket",
    ),
    "unix://$HOME/.chronovisor/runtime/reranker.sock": (
        "chronovisor.core.runtime_config",
        "RerankerServiceConfig.socket",
    ),
}


def _normalized_socket_locator(value: str) -> str:
    normalized = value.removeprefix("unix://")
    if normalized == "$HOME":
        return "~"
    if normalized.startswith("$HOME/"):
        return f"~/{normalized.removeprefix('$HOME/')}"
    return normalized


def _call_module_state_prefix(module: str) -> str:
    return f"{_CALL_MODULE_STATE_PREFIX}{module}:"


def _is_call_module_state_name(name: str) -> bool:
    return name.startswith(_CALL_MODULE_STATE_PREFIX)


def _call_closure_state_prefix(group_id: str) -> str:
    return f"{_CALL_CLOSURE_STATE_PREFIX}{group_id}:"


def _is_call_closure_state_name(name: str) -> bool:
    return name.startswith(_CALL_CLOSURE_STATE_PREFIX)


def _is_call_path_state_name(name: str) -> bool:
    return _is_call_module_state_name(name) or _is_call_closure_state_name(name)


def _has_nonrecursive_self_path(info: FunctionInfo) -> bool:
    """Return whether one normal/return path avoids a direct self-call."""

    def calls_self(node: ast.AST | None) -> bool:
        if node is None:
            return False

        class SelfCallVisitor(ast.NodeVisitor):
            found = False

            def visit_Call(self, item: ast.Call) -> None:
                if isinstance(item.func, ast.Name) and item.func.id == info.node.name:
                    self.found = True
                    return
                self.generic_visit(item)

            def visit_FunctionDef(self, item: ast.FunctionDef) -> None:
                return

            def visit_AsyncFunctionDef(self, item: ast.AsyncFunctionDef) -> None:
                return

            def visit_ClassDef(self, item: ast.ClassDef) -> None:
                return

            def visit_Lambda(self, item: ast.Lambda) -> None:
                return

        visitor = SelfCallVisitor()
        visitor.visit(node)
        return visitor.found

    def expression_has_nonrecursive_path(node: ast.expr | None) -> bool:
        if isinstance(node, ast.IfExp):
            return not calls_self(node.test) and (
                expression_has_nonrecursive_path(node.body)
                or expression_has_nonrecursive_path(node.orelse)
            )
        return not calls_self(node)

    def block_paths(statements: Sequence[ast.stmt]) -> tuple[bool, bool]:
        can_continue = True
        can_exit = False
        for statement in statements:
            if not can_continue:
                break
            if isinstance(statement, ast.If):
                if calls_self(statement.test):
                    can_continue = False
                    continue
                body_continue, body_exit = block_paths(statement.body)
                else_continue, else_exit = block_paths(statement.orelse)
                can_continue = body_continue or else_continue
                can_exit |= body_exit or else_exit
                continue
            if isinstance(statement, ast.Return):
                if expression_has_nonrecursive_path(statement.value):
                    can_exit = True
                can_continue = False
                continue
            if isinstance(statement, ast.Raise):
                can_continue = False
                continue
            if calls_self(statement):
                can_continue = False
        return can_continue, can_exit

    has_direct_self_call = any(calls_self(statement) for statement in info.node.body)
    can_continue, can_exit = block_paths(info.node.body)
    return has_direct_self_call and (can_continue or can_exit)


def _is_main_guard(expression: ast.expr) -> bool:
    if not isinstance(expression, ast.Compare) or len(expression.ops) != 1:
        return False
    if not isinstance(expression.ops[0], ast.Eq) or len(expression.comparators) != 1:
        return False
    left, right = expression.left, expression.comparators[0]
    pairs = ((left, right), (right, left))
    return any(
        isinstance(name, ast.Name)
        and name.id == "__name__"
        and isinstance(value, ast.Constant)
        and value.value == "__main__"
        for name, value in pairs
    )


def _main_guard_call_ids(tree: ast.Module) -> frozenset[int]:
    call_ids: set[int] = set()
    for statement in tree.body:
        if not isinstance(statement, ast.If) or not _is_main_guard(statement.test):
            continue
        for guarded_statement in statement.body:
            call_ids.update(
                id(node)
                for node in ast.walk(guarded_statement)
                if isinstance(node, ast.Call)
            )
    return frozenset(call_ids)


def _value_has_nested_origins(value: FlowValue) -> bool:
    if value.has_origins:
        return True
    if any(
        _value_has_nested_origins(item) for item in value.attribute_values.values()
    ):
        return True
    return value.structured_items is not None and any(
        _value_has_nested_origins(item) for item in value.structured_items
    )


class _AccessAnalysis:
    def __init__(
        self,
        snapshot: Mapping[str, bytes],
        resource_candidates: Iterable[Mapping[str, Any]],
        *,
        limits: AnalysisLimits,
        progress: AnalysisProgress,
    ) -> None:
        self.limits = limits
        self.progress = progress
        self.snapshot = dict(snapshot)
        self.trees: dict[str, ast.Module] = {}
        self.paths: dict[str, str] = {}
        self.imported_symbols: dict[str, dict[str, tuple[str, str]]] = {}
        self.imported_modules: dict[str, dict[str, str]] = {}
        self.functions: dict[str, FunctionInfo] = {}
        self.classes: dict[str, dict[str, str]] = {}
        package_modules: set[str] = set()
        for path, raw in sorted(self.snapshot.items()):
            if not path.startswith("src/") or not path.endswith(".py"):
                continue
            module = _module_name(path)
            tree = ast.parse(raw.decode("utf-8"), filename=path)
            self.trees[module] = tree
            self.paths[module] = path
            is_package = path.endswith("/__init__.py")
            if is_package:
                package_modules.add(module)
            symbols, modules = _import_tables(module, tree, is_package=is_package)
            self.imported_symbols[module] = symbols
            self.imported_modules[module] = modules
            functions, classes = _collect_functions(module, path, tree)
            self.functions.update(functions)
            self.classes[module] = classes
        self.package_modules = frozenset(package_modules)
        self.main_guard_call_ids = frozenset().union(
            *(_main_guard_call_ids(tree) for tree in self.trees.values())
        )
        self.future_annotations = frozenset(
            module
            for module, tree in self.trees.items()
            if _uses_future_annotations(tree)
        )
        self.dataclass_targets = frozenset(
            f"{module}:{statement.name}"
            for module, tree in self.trees.items()
            for statement in tree.body
            if isinstance(statement, ast.ClassDef)
            and any(
                _is_declared_dataclass_decorator(
                    decorator,
                    imported_symbols=self.imported_symbols[module],
                    imported_modules=self.imported_modules[module],
                )
                for decorator in statement.decorator_list
            )
        )
        known_modules = set(self.trees)
        for module in self.trees:
            parts = module.split(".")
            known_modules.update(
                ".".join(parts[:index]) for index in range(1, len(parts))
            )
        known_modules.discard("builtins")
        self.known_modules = frozenset(known_modules)
        self.origin_symbols: dict[tuple[str, str], FlowValue] = {}
        self.resource_locators: dict[str, str] = {}
        self.socket_field_origins: dict[
            tuple[str, str], tuple[str, FlowValue]
        ] = {}
        for row in resource_candidates:
            resource_id = str(row.get("id") or row.get("resource_id") or "")
            locator = row.get("locator")
            module = str(row.get("module") or "")
            symbol = str(row.get("symbol") or "")
            if (
                not resource_id
                or not module
                or not symbol
                or not isinstance(locator, Mapping)
                or not isinstance(locator.get("value"), str)
            ):
                raise ValueError("resource candidates require id/module/symbol/locator")
            locator_value = str(locator["value"])
            origin_key = (module, symbol)
            existing = self.origin_symbols.get(origin_key)
            if existing is not None and resource_id not in existing.origins:
                raise ValueError(
                    f"ambiguous runtime resource candidates for {module}:{symbol}"
                )
            previous_locator = self.resource_locators.get(resource_id)
            if previous_locator is not None and previous_locator != locator_value:
                raise ValueError(
                    f"conflicting locators for runtime resource {resource_id}"
                )
            self.resource_locators[resource_id] = locator_value
            origin_value = FlowValue(
                {resource_id: frozenset({(f"origin:{module}:{symbol}",)})}
            )
            alias = _SOCKET_ORIGIN_ALIASES.get(locator_value)
            if alias is not None:
                self.socket_field_origins[alias] = (
                    _normalized_socket_locator(locator_value),
                    origin_value.bound(
                        f"origin-alias:{alias[0]}:{alias[1]}"
                    ),
                )
                continue
            self.origin_symbols[origin_key] = origin_value
        export_table = build_module_exports(
            self.trees,
            package_modules=self.package_modules,
            known_modules=self.known_modules,
            origin_symbols=self.origin_symbols,
            limits=self.limits,
            progress=self.progress,
        )
        self.module_exports = export_table.bindings
        self.module_star_exports = export_table.star_bindings
        self.module_star_definite = export_table.star_definite
        self.module_star_policies = export_table.star_policies
        self.module_runtime_envs: dict[str, dict[str, FlowValue]] = {
            module: {} for module in sorted(self.trees)
        }
        self.function_refs_by_node = {
            id(info.node): ref for ref, info in self.functions.items()
        }
        self.function_loaded_names = {
            ref: frozenset(
                node.id
                for node in ast.walk(info.node)
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
            )
            for ref, info in self.functions.items()
        }
        self.function_parents = {
            ref: info.parent_ref for ref, info in self.functions.items()
        }
        self.recursive_base_targets = frozenset(
            ref
            for ref, info in self.functions.items()
            if _has_nonrecursive_self_path(info)
        )
        self.nested_functions = {
            (info.parent_ref, info.node.name): ref
            for ref, info in self.functions.items()
            if info.parent_ref is not None
        }
        self.closure_envs: dict[str, dict[str, FlowValue]] = {
            ref: {} for ref in self.functions
        }
        self.closure_instance_envs: dict[str, dict[str, FlowValue]] = {}
        self.closure_instance_groups: dict[str, str] = {}
        self.params: dict[str, dict[str, FlowValue]] = {
            ref: {} for ref in self.functions
        }
        self.definition_defaults: dict[str, dict[str, FlowValue]] = {
            ref: {} for ref in self.functions
        }
        self.returns: dict[str, FlowValue] = {
            ref: FlowValue() for ref in self.functions
        }
        self.called_targets: set[str] = set()
        self.locally_executed_targets: set[str] = set()
        self.summary_required_targets: set[str] = set()
        self.analyzed_summary_targets: set[str] = set()
        self._active_local_calls: set[str] = set()
        self._active_local_activations: dict[str, str] = {}
        self._local_module_states: list[
            tuple[str, dict[str, FlowValue], dict[str, set[str]]]
        ] = []
        self._local_closure_states: list[dict[str, dict[str, FlowValue]]] = []
        self.class_attrs: dict[str, dict[str, FlowValue]] = {}
        self.dataclass_infos: dict[str, DataclassInfo] = {}
        self.class_comprehension_parents: dict[
            str,
            tuple[dict[str, FlowValue], dict[str, set[str]]],
        ] = {}
        self.facts = AccessFactCollector(
            self.resource_locators,
            _collect_syntax_sites(
                self.trees,
                self.paths,
                self.function_refs_by_node,
            ),
        )
        self._persistent_changed = False

    def run(self) -> dict[str, Any]:
        outer_iteration = 0
        while True:
            outer_iteration += 1
            self.progress.record_work(
                phase="outer",
                subject="all-modules-and-functions",
                counter="outer_iterations",
                iteration=outer_iteration,
                limits=self.limits,
            )
            self._persistent_changed = False
            for module, tree in sorted(self.trees.items()):
                self.progress.record_work(
                    phase="module_analysis",
                    subject=module,
                    counter="module_analyses",
                    iteration=self.progress.module_analyses + 1,
                    limits=self.limits,
                )
                self._analyze_module(module, tree)
            module_changed = self._persistent_changed
            function_changed = False
            summary_iteration = 0
            while True:
                summary_iteration += 1
                self.progress.record_work(
                    phase="function_summary",
                    subject=f"outer:{outer_iteration}",
                    counter="function_summary_iterations",
                    iteration=summary_iteration,
                    limits=self.limits,
                )
                self._persistent_changed = False
                for ref, info in sorted(self.functions.items()):
                    if (
                        ref in self.locally_executed_targets
                        and ref not in self.summary_required_targets
                    ):
                        continue
                    if info.parent_ref is not None and ref not in self.called_targets:
                        continue
                    if not self._function_has_origin_inputs(info):
                        continue
                    self.progress.record_work(
                        phase="function_analysis",
                        subject=ref,
                        counter="function_analyses",
                        iteration=self.progress.function_analyses + 1,
                        limits=self.limits,
                    )
                    self._analyze_function(info)
                if not self._persistent_changed:
                    break
                function_changed = True
                self.progress.require_stable_or_within_limit(
                    phase="function_summary",
                    subject=f"outer:{outer_iteration}",
                    iteration=summary_iteration,
                    limit=self.limits.max_function_summary_iterations,
                )
            self._persistent_changed = module_changed or function_changed
            if not self._persistent_changed:
                break
            self.progress.require_stable_or_within_limit(
                phase="outer",
                subject="all-modules-and-functions",
                iteration=outer_iteration,
                limit=self.limits.max_outer_iterations,
            )
        return self.facts.result()

    def _function_has_origin_inputs(self, info: FunctionInfo) -> bool:
        if info.ref in self.summary_required_targets:
            return True
        if self._function_has_origin_local_import(info):
            return True
        values = [
            *self.params[info.ref].values(),
            *self.definition_defaults[info.ref].values(),
            *self.closure_envs[info.ref].values(),
        ]
        module_env = self.module_runtime_envs[info.module]
        for name in (
            info.referenced_names
            | info.global_names
            | self.function_loaded_names[info.ref]
        ):
            value = module_env.get(name)
            if value is not None:
                values.append(value)
        explicit_constructors = {
            constructor.ref: constructor
            for value in values
            for class_target in value.class_targets
            if (
                (constructor := self.functions.get(f"{class_target}.__init__"))
                is not None
                and constructor.ref != info.ref
            )
        }
        for constructor in explicit_constructors.values():
            if (
                constructor.ref in self.summary_required_targets
                or self._function_has_origin_local_import(constructor)
            ):
                return True
            values.extend(self.params[constructor.ref].values())
            values.extend(self.definition_defaults[constructor.ref].values())
            values.extend(self.closure_envs[constructor.ref].values())
            constructor_module_env = self.module_runtime_envs[constructor.module]
            for name in (
                constructor.referenced_names
                | constructor.global_names
                | self.function_loaded_names[constructor.ref]
            ):
                value = constructor_module_env.get(name)
                if value is not None:
                    values.append(value)
        if info.class_ref is not None:
            values.extend(self.class_attrs.get(info.class_ref, {}).values())
            dataclass_info = self.dataclass_infos.get(info.class_ref)
            if dataclass_info is not None:
                values.extend(
                    field.default
                    for field in dataclass_info.fields
                    if field.default is not None
                )
                values.extend(
                    field.default_factory
                    for field in dataclass_info.fields
                    if field.default_factory is not None
                )
                values.extend(
                    FlowValue(class_targets=set(field.declared_class_targets))
                    for field in dataclass_info.fields
                    if field.declared_class_targets
                )
        expanded_values = list(values)
        seen_classes: set[str] = set()
        pending_classes = set().union(*(value.class_targets for value in values))
        while pending_classes:
            class_target = pending_classes.pop()
            if class_target in seen_classes:
                continue
            seen_classes.add(class_target)
            dataclass_info = self.dataclass_infos.get(class_target)
            if dataclass_info is None:
                continue
            for field in dataclass_info.fields:
                if field.default is not None:
                    expanded_values.append(field.default)
                    pending_classes.update(field.default.class_targets)
                if field.default_factory is not None:
                    expanded_values.append(field.default_factory)
                    pending_classes.update(field.default_factory.class_targets)
                pending_classes.update(field.declared_class_targets)
        values = expanded_values
        if any(_value_has_nested_origins(value) for value in values):
            return True
        call_targets = set().union(*(value.call_targets for value in values))
        if any(
            _value_has_nested_origins(self.returns.get(target, FlowValue()))
            for target in call_targets
        ):
            return True
        module_refs = set().union(*(value.module_refs for value in values))
        return any(
            _value_has_nested_origins(exported)
            for module_ref in module_refs
            for exported in self.module_exports.get(module_ref, {}).values()
        )

    def _function_has_origin_local_import(self, info: FunctionInfo) -> bool:
        for node in ast.walk(info.node):
            candidate_modules: set[str] = set()
            candidate_values: list[FlowValue] = []
            if isinstance(node, ast.Import):
                candidate_modules.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                target_module = resolve_import_from(
                    info.module,
                    level=node.level,
                    imported_module=node.module,
                    is_package=info.module in self.package_modules,
                )
                for alias in node.names:
                    if alias.name == "*":
                        candidate_values.extend(
                            self.module_star_exports.get(target_module, {}).values()
                        )
                        continue
                    candidate_values.append(
                        self.module_exports.get(target_module, {}).get(
                            alias.name, FlowValue()
                        )
                    )
                    candidate_modules.add(
                        f"{target_module}.{alias.name}"
                        if target_module
                        else alias.name
                    )
            if any(_value_has_nested_origins(value) for value in candidate_values):
                return True
            if any(
                _value_has_nested_origins(exported)
                for candidate_module in candidate_modules
                for exported in self.module_exports.get(
                    candidate_module, {}
                ).values()
            ):
                return True
        return False

    def _analyze_module(self, module: str, tree: ast.Module) -> bool:
        env: dict[str, FlowValue] = {}
        changed = analyze_block(
            self,
            tree.body,
            module=module,
            actor=f"{module}:<module>",
            class_ref=None,
            env=env,
            object_env={},
            call_ordinals=_call_ordinals(tree),
        )[0]
        runtime_env = {
            name: value.copy()
            for name, value in env.items()
            if not _is_call_path_state_name(name)
        }
        if runtime_env != self.module_runtime_envs[module]:
            self.module_runtime_envs[module] = runtime_env
            self._persistent_changed = True
            changed = True
        return changed

    def _analyze_function(self, info: FunctionInfo) -> bool:
        module_env = self.module_runtime_envs[info.module]
        required_module_names = (
            info.referenced_names
            | info.global_names
            | self.function_loaded_names[info.ref]
        )
        env = {
            name: value.copy()
            for name, value in module_env.items()
            if name in required_module_names
            and (name not in info.local_names or name in info.global_names)
        }
        for name in info.local_names:
            env[name] = FlowValue()
        object_env: dict[str, set[str]] = {}
        for name, value in self.closure_envs[info.ref].items():
            if name in info.local_names or name in info.global_names:
                continue
            env[name] = value.copy()
            if value.object_types:
                object_env[name] = set(value.object_types)
        if info.class_ref is not None and info.parameters:
            object_env[info.parameters[0]] = {info.class_ref}
        for parameter, value in self.params[info.ref].items():
            env[parameter] = value.bound(f"param:{info.ref}:{parameter}")
        for parameter, default_value in self.definition_defaults[info.ref].items():
            if default_value.has_analysis_state:
                env[parameter] = exclusive_flow_join(
                    [
                        env.get(parameter, FlowValue()),
                        default_value.bound(f"default:{info.ref}:{parameter}"),
                    ]
                )
        changed, returned = analyze_block(
            self,
            info.node.body,
            module=info.module,
            actor=info.ref,
            class_ref=info.class_ref,
            env=env,
            object_env=object_env,
            call_ordinals=info.call_ordinals,
        )
        merged_return = exclusive_flow_join([self.returns[info.ref], returned])
        if merged_return != self.returns[info.ref]:
            self.returns[info.ref] = merged_return
            self._persistent_changed = True
            changed = True
        self.analyzed_summary_targets.add(info.ref)
        return changed

    def _merge_definition_default(
        self,
        ref: str,
        parameter: str,
        value: FlowValue,
    ) -> bool:
        previous = self.definition_defaults[ref].get(parameter, FlowValue())
        merged = exclusive_flow_join([previous, value])
        if merged == previous:
            return False
        self.definition_defaults[ref][parameter] = merged
        self._persistent_changed = True
        return True

    def _merge_closure(
        self,
        ref: str,
        *,
        actor: str,
        module: str,
        env: Mapping[str, FlowValue],
    ) -> bool:
        instance_id = self._local_closure_instance_id(ref, actor=actor)
        if instance_id is not None:
            local_instances = self._local_closure_states[-1]
            local_instances[instance_id] = self._capture_closure_env(
                ref,
                actor=actor,
                module=module,
                env=env,
                instance_id=instance_id,
            )
            return False
        changed = False
        closure = self.closure_envs[ref]
        module_env = self.module_runtime_envs[module]
        info = self.functions[ref]
        if info.parent_ref is None:
            return False
        for name, value in env.items():
            if _is_call_path_state_name(name):
                continue
            if name not in info.referenced_names and name not in info.nonlocal_names:
                continue
            if name in info.local_names or name in info.global_names:
                continue
            if name in module_env and value == module_env[name]:
                continue
            if (
                not value.has_analysis_state
            ):
                continue
            previous = closure.get(name, FlowValue())
            closure[name] = exclusive_flow_join(
                [previous, value.bound(f"closure:{actor}->{ref}:{name}")]
            )
            if closure[name] != previous:
                self._persistent_changed = True
                changed = True
        return changed

    def _function_definition_value(self, ref: str, *, actor: str) -> FlowValue:
        value = FlowValue(call_targets={ref})
        instance_id = self._local_closure_instance_id(ref, actor=actor)
        if instance_id is not None:
            value.closure_instances.add((ref, instance_id))
            activation_id = self._active_local_activations[actor]
            self.closure_instance_groups[instance_id] = activation_id
        return value

    def _local_closure_instance_id(self, ref: str, *, actor: str) -> str | None:
        activation_id = self._active_local_activations.get(actor)
        if activation_id is None or self.functions[ref].parent_ref != actor:
            return None
        return f"{activation_id}|closure:{ref}"

    def _capture_closure_env(
        self,
        ref: str,
        *,
        actor: str,
        module: str,
        env: Mapping[str, FlowValue],
        instance_id: str,
    ) -> dict[str, FlowValue]:
        captured: dict[str, FlowValue] = {}
        module_env = self.module_runtime_envs[module]
        info = self.functions[ref]
        for name, value in env.items():
            if _is_call_path_state_name(name):
                continue
            if name not in info.referenced_names and name not in info.nonlocal_names:
                continue
            if name in info.local_names or name in info.global_names:
                continue
            if name in module_env and value == module_env[name]:
                continue
            group_id = self.closure_instance_groups.get(instance_id, instance_id)
            captured[name] = value.bound(
                f"closure:{actor}:cell:{name}|group={group_id}"
            )
        return captured

    def _assignment_binding_value(
        self,
        name: str,
        value: FlowValue,
        *,
        module: str,
        actor: str,
    ) -> FlowValue:
        if actor != f"{module}:<module>":
            return value
        origin = self.origin_symbols.get((module, name))
        return origin.copy() if origin is not None else value

    def _socket_field_origin(
        self,
        *,
        module: str,
        class_ref: str,
        field_name: str,
        expression: ast.expr,
    ) -> FlowValue | None:
        if not (
            isinstance(expression, ast.Constant)
            and isinstance(expression.value, str)
        ):
            return None
        class_name = class_ref.removeprefix(f"{module}:")
        candidate = self.socket_field_origins.get(
            (module, f"{class_name}.{field_name}")
        )
        if candidate is None:
            return None
        expected, origin = candidate
        if _normalized_socket_locator(expression.value) != expected:
            return None
        return origin.copy()

    def _bind_target(
        self,
        target: ast.expr,
        value: FlowValue,
        *,
        actor: str,
        class_ref: str | None,
        env: dict[str, FlowValue],
        object_env: dict[str, set[str]],
        module: str | None = None,
        source_node: ast.AST | None = None,
        ordinal: int = 0,
    ) -> bool:
        if isinstance(target, ast.Name):
            bound = value.bound(f"alias:{actor}:{target.id}")
            env[target.id] = bound
            if value.object_types:
                object_env[target.id] = set(value.object_types)
            else:
                object_env.pop(target.id, None)
            return False
        if isinstance(target, ast.Subscript):
            if (
                isinstance(target.value, ast.Attribute)
                and target.value.attr == "__dict__"
                and isinstance(target.value.value, ast.Name)
            ):
                class_value = env.get(target.value.value.id, FlowValue())
                if self._invalidate_socketserver_subclass(
                    class_value,
                    env=env,
                    object_env=object_env,
                ):
                    return False
            base = stdlib_module_dict_reference(target.value, env)
            if base is not None:
                attribute = (
                    target.slice.value
                    if isinstance(target.slice, ast.Constant)
                    and isinstance(target.slice.value, str)
                    else "*"
                )
                module_ref = precise_stdlib_module_name(base)
                if self._taint_stdlib_module_attribute(
                    base,
                    attribute=attribute,
                    env=env,
                    object_env=object_env,
                ):
                    if (
                        attribute != "*"
                        and value.has_origins
                        and module is not None
                        and source_node is not None
                    ):
                        assert module_ref is not None
                        self.facts.record_escape(
                            value,
                            node=source_node,
                            actor=actor,
                            operation=f"assignment:{module_ref}.{attribute}",
                            sink=f"{module_ref}.{attribute}",
                            reason="registered_locator_to_stdlib_module_mutation",
                            path=self.paths[module],
                            line=int(getattr(source_node, "lineno", 0)),
                            ordinal=ordinal,
                        )
                    return False
        if isinstance(target, ast.Attribute) and isinstance(
            target.value, ast.Name
        ):
            base = env.get(target.value.id, FlowValue())
            if self._invalidate_socketserver_subclass(
                base,
                env=env,
                object_env=object_env,
            ):
                return False
            if socket_handle_kind(base) is not None:
                if base.has_origins and module is not None and source_node is not None:
                    self.facts.record_escape(
                        base,
                        node=source_node,
                        actor=actor,
                        operation=f"socket.attribute_mutation:{target.attr}",
                        sink=f"socket.socket.{target.attr}",
                        reason="unsupported_socket_handle_mutation",
                        path=self.paths[module],
                        line=int(getattr(source_node, "lineno", 0)),
                        ordinal=ordinal,
                    )
                self._contaminate_runtime_objects(env, object_env, [base])
                return False
            module_ref = precise_stdlib_module_name(base)
            if self._taint_stdlib_module_attribute(
                base,
                attribute=target.attr,
                env=env,
                object_env=object_env,
            ):
                if value.has_origins and module is not None and source_node is not None:
                    assert module_ref is not None
                    self.facts.record_escape(
                        value,
                        node=source_node,
                        actor=actor,
                        operation=f"assignment:{module_ref}.{target.attr}",
                        sink=f"{module_ref}.{target.attr}",
                        reason="registered_locator_to_stdlib_module_mutation",
                        path=self.paths[module],
                        line=int(getattr(source_node, "lineno", 0)),
                        ordinal=ordinal,
                    )
                return False
            if (
                base.instance_ids
                or base.class_targets
                or base.attribute_values
                or base.attribute_values_complete
                or base.attribute_values_ambiguous
                or base.unknown_callable
            ):
                bound_attribute = value.bound(
                    f"attribute-assignment:{actor}:{target.attr}"
                )
                precise_instance = (
                    len(base.instance_ids) == 1
                    and not base.class_targets
                    and not base.unknown_callable
                    and not base.attribute_values_ambiguous
                )
                precise_class = (
                    len(base.class_targets) == 1
                    and not base.instance_ids
                    and base.object_types == base.class_targets
                    and not base.origins
                    and not base.module_refs
                    and not base.call_targets
                    and not base.unknown_callable
                    and not base.attribute_values_ambiguous
                )
                precise_receiver = precise_instance or precise_class

                def updated_receiver(candidate: FlowValue) -> FlowValue:
                    updated = candidate.copy()
                    if precise_receiver:
                        updated.attribute_values[target.attr] = (
                            bound_attribute.copy()
                        )
                    else:
                        previous = updated.attribute_values.get(target.attr)
                        updated.attribute_values[target.attr] = (
                            bound_attribute.copy()
                            if previous is None
                            else previous.merged(bound_attribute)
                        )
                        updated.attribute_values_complete = False
                        updated.attribute_values_ambiguous = True
                    return updated

                receiver_name = target.value.id
                receiver_ids = set(base.instance_ids)
                receiver_classes = set(base.class_targets)
                for name, candidate in list(env.items()):
                    is_receiver_alias = name == receiver_name
                    if precise_instance:
                        is_receiver_alias = (
                            candidate.instance_ids == receiver_ids
                            and not candidate.class_targets
                        )
                    elif precise_class:
                        is_receiver_alias = (
                            candidate.class_targets == receiver_classes
                            and not candidate.instance_ids
                        )
                    if not is_receiver_alias:
                        continue
                    updated = updated_receiver(candidate)
                    env[name] = updated
                    if updated.object_types:
                        object_env[name] = set(updated.object_types)
                    else:
                        object_env.pop(name, None)
                return False
        if (
            class_ref is not None
            and isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "self"
        ):
            receiver = env.get("self", FlowValue())
            if receiver.instance_ids or receiver.attribute_values:
                updated = receiver.copy()
                updated.attribute_values[target.attr] = value.bound(
                    f"attr:{class_ref}:{target.attr}"
                )
                env["self"] = updated
                object_env["self"] = set(updated.object_types)
                return False
            attrs = self.class_attrs.setdefault(class_ref, {})
            previous = attrs.get(target.attr, FlowValue())
            attrs[target.attr] = previous.merged(
                value.bound(f"attr:{class_ref}:{target.attr}")
            )
            if attrs[target.attr] != previous:
                self._persistent_changed = True
                return True
            return False
        return False

    def _taint_stdlib_module_attribute(
        self,
        base: FlowValue,
        *,
        attribute: str,
        env: dict[str, FlowValue],
        object_env: dict[str, set[str]],
    ) -> bool:
        module_ref = precise_stdlib_module_name(base)
        if module_ref is None or module_ref in self.known_modules:
            return False
        marker = stdlib_module_mutation_marker(module_ref, attribute)
        state_name = stdlib_module_state_name(module_ref)
        state = env.get(state_name, FlowValue(module_refs={module_ref})).copy()
        state.object_types.add(marker)
        env[state_name] = state
        object_env[state_name] = set(state.object_types)
        for name, candidate in list(env.items()):
            if candidate.module_refs != {module_ref}:
                continue
            tainted = candidate.copy()
            tainted.object_types.add(marker)
            env[name] = tainted
            object_env[name] = set(tainted.object_types)
        return True

    def _contaminate_runtime_objects(
        self,
        env: dict[str, FlowValue],
        object_env: dict[str, set[str]],
        values: Sequence[FlowValue],
    ) -> None:
        affected = contaminate_runtime_objects(env, object_env, values)
        if not affected:
            return
        for attrs in self.class_attrs.values():
            for name, candidate in list(attrs.items()):
                if not affected.intersection(
                    candidate.runtime_object_ids
                    | candidate.runtime_descriptor_ids
                ):
                    continue
                contaminated = candidate.copy()
                contaminated.object_types.add(
                    UNRESOLVED_RUNTIME_OBJECT_ALTERNATIVE_TYPE
                )
                if contaminated == candidate:
                    continue
                attrs[name] = contaminated
                self._persistent_changed = True

    def _invalidate_socketserver_subclass(
        self,
        value: FlowValue,
        *,
        env: dict[str, FlowValue],
        object_env: dict[str, set[str]],
    ) -> bool:
        class_targets = set(value.class_targets)
        if not class_targets or not any(
            object_type.startswith(SOCKETSERVER_LOCAL_CLASS_OBJECT_PREFIX)
            for object_type in value.object_types
        ):
            return False
        changed = False
        for name, candidate in list(env.items()):
            if not class_targets.intersection(candidate.class_targets) or not any(
                object_type.startswith(SOCKETSERVER_LOCAL_CLASS_OBJECT_PREFIX)
                for object_type in candidate.object_types
            ):
                continue
            invalidated = candidate.copy()
            invalidated.attribute_values_complete = False
            invalidated.attribute_values_ambiguous = True
            if invalidated == candidate:
                continue
            env[name] = invalidated
            object_env[name] = set(invalidated.object_types)
            changed = True
        return changed

    def _replace_runtime_object_aliases(
        self,
        env: dict[str, FlowValue],
        object_env: dict[str, set[str]],
        receiver: FlowValue,
        replacement: FlowValue,
    ) -> None:
        affected = replace_runtime_object_aliases(
            env,
            object_env,
            receiver,
            replacement,
        )
        if not affected:
            return
        for attrs in self.class_attrs.values():
            for name, candidate in list(attrs.items()):
                if not affected.intersection(candidate.runtime_object_ids):
                    continue
                attrs[name] = replacement.copy()
                self._persistent_changed = True

    def _runtime_object_identity(
        self,
        node: ast.Call,
        *,
        kind: str,
        actor: str,
    ) -> str:
        site_id = self.facts.site_id(node)
        activation = self._active_local_activations.get(actor)
        if activation is None:
            return f"{kind}:{site_id}"
        return f"{kind}:{site_id}|activation={activation}"

    def _instance_class_attribute(
        self,
        instance: FlowValue,
        *,
        attribute: str,
        env: Mapping[str, FlowValue],
    ) -> FlowValue | None:
        if not instance.instance_ids or instance.class_targets:
            return None
        known_class_targets = frozenset().union(
            *(module_classes.values() for module_classes in self.classes.values())
        )
        instance_classes = sorted(
            instance.object_types.intersection(known_class_targets)
        )
        if not instance_classes:
            return None

        fallback: FlowValue | None = None
        fallback_ambiguous = instance.attribute_values_ambiguous
        resolved_classes = 0
        for class_target in instance_classes:
            class_bindings = [
                value
                for value in env.values()
                if value.class_targets == {class_target}
                and not value.instance_ids
                and attribute in value.attribute_values
            ]
            class_attribute: FlowValue | None = None
            for binding in class_bindings:
                candidate = binding.attribute_values[attribute]
                if class_attribute is None:
                    class_attribute = candidate.copy()
                    continue
                if candidate != class_attribute:
                    fallback_ambiguous = True
                class_attribute = class_attribute.merged(candidate)
            if class_attribute is None:
                info = self.dataclass_infos.get(class_target)
                field = None
                if info is not None:
                    field = next(
                        (
                            candidate
                            for candidate in info.fields
                            if candidate.name == attribute
                            and candidate.default is not None
                        ),
                        None,
                    )
                if field is not None:
                    assert field.default is not None
                    class_attribute = field.default.copy()
            if class_attribute is None:
                fallback_ambiguous = True
                continue
            resolved_classes += 1
            class_attribute = class_attribute.bound(
                f"class-attribute:{class_target}:{attribute}"
            )
            if fallback is None:
                fallback = class_attribute
            else:
                if class_attribute != fallback:
                    fallback_ambiguous = True
                fallback = fallback.merged(class_attribute)
        if fallback is None:
            return None
        if resolved_classes != len(instance_classes):
            fallback_ambiguous = True
        if fallback_ambiguous:
            fallback.attribute_values_ambiguous = True
        return fallback

    def _eval(
        self,
        node: ast.expr,
        *,
        module: str,
        actor: str,
        class_ref: str | None,
        env: dict[str, FlowValue],
        object_env: dict[str, set[str]],
        call_ordinals: Mapping[int, int],
    ) -> FlowValue:
        if isinstance(node, ast.Name):
            return env.get(node.id, FlowValue()).copy()
        if isinstance(node, ast.Attribute):
            base = self._eval(
                node.value,
                module=module,
                actor=actor,
                class_ref=class_ref,
                env=env,
                object_env=object_env,
                call_ordinals=call_ordinals,
            )
            if node.attr in base.attribute_values:
                projected_attribute = base.attribute_values[node.attr].bound(
                    f"attribute:{node.attr}"
                )
                variant_attributes = [
                    variant.attribute_values[node.attr].bound(
                        f"attribute:{node.attr}"
                    )
                    for variant in candidate_flow_variants(base)
                    if node.attr in variant.attribute_values
                ]
                projected_attribute = with_candidate_flow_variants(
                    projected_attribute, variant_attributes
                )
                if base.attribute_values_ambiguous:
                    projected_attribute.attribute_values_ambiguous = True
                return projected_attribute
            class_attribute = self._instance_class_attribute(
                base,
                attribute=node.attr,
                env=env,
            )
            if class_attribute is not None:
                return class_attribute
            if base.attribute_values or base.attribute_values_complete:
                return FlowValue(attribute_values_ambiguous=True)
            if (
                isinstance(node.value, ast.Name)
                and node.value.id == "self"
                and class_ref is not None
            ):
                value = self.class_attrs.get(class_ref, {}).get(node.attr)
                if value is not None:
                    return value.copy()
            if node.attr == "__dict__" and any(
                object_type.startswith(SOCKETSERVER_LOCAL_CLASS_OBJECT_PREFIX)
                for object_type in base.object_types
            ):
                return FlowValue(
                    object_types={SOCKETSERVER_CLASS_DICT_OBJECT_TYPE},
                    attribute_values={"\0bound_receiver": base.copy()},
                    attribute_values_complete=True,
                )
            method_targets = {
                f"{object_type}.{node.attr}"
                for object_type in base.object_types
                if f"{object_type}.{node.attr}" in self.functions
            }
            if method_targets:
                return FlowValue(
                    call_targets=method_targets,
                    attribute_values={"\0bound_receiver": base.copy()},
                    attribute_values_complete=True,
                )
            if is_path_receiver(base) and node.attr in PATH_TRANSFORMS:
                return FlowValue(
                    object_types={f"{PATH_BOUND_TRANSFORM_PREFIX}{node.attr}"},
                    attribute_values={"\0bound_receiver": base.copy()},
                    attribute_values_complete=True,
                )
            projected, projection = project_sqlite_attribute(
                base, attribute=node.attr
            )
            if projected:
                return projection
            if base.has_origins and base.object_types.intersection(
                SQLITE_HANDLE_OBJECT_TYPES
            ):
                unknown = base.bound(f"attribute:sqlite.unknown:{node.attr}")
                unknown.object_types.difference_update(SQLITE_HANDLE_OBJECT_TYPES)
                unknown.object_types.add(SQLITE_UNKNOWN_ATTRIBUTE_OBJECT_TYPE)
                return unknown
            if base.has_origins and FILE_HANDLE_OBJECT_TYPE in base.object_types:
                unknown = base.bound(f"attribute:file.unknown:{node.attr}")
                unknown.object_types.discard(FILE_HANDLE_OBJECT_TYPE)
                if node.attr == "close":
                    unknown.object_types.add(FILE_BOUND_CLOSE_OBJECT_TYPE)
                elif node.attr == "fileno":
                    unknown.object_types.add(FILE_BOUND_FILENO_OBJECT_TYPE)
                else:
                    unknown.object_types.add(FILE_UNKNOWN_ATTRIBUTE_OBJECT_TYPE)
                return unknown
            if socket_handle_kind(base) is not None:
                bound = base.bound(f"attribute:socket:{node.attr}")
                bound.object_types = {f"{SOCKET_BOUND_METHOD_PREFIX}{node.attr}"}
                bound.attribute_values = {"\0bound_receiver": base.copy()}
                bound.attribute_values_complete = True
                return bound
            if socketserver_handle_state(base) is not None:
                bound = base.bound(f"attribute:socketserver:{node.attr}")
                bound.object_types = {
                    f"{SOCKETSERVER_BOUND_METHOD_PREFIX}{node.attr}"
                }
                bound.attribute_values = {"\0bound_receiver": base.copy()}
                bound.attribute_values_complete = True
                return bound
            if is_path_receiver(base) and node.attr == "open":
                unknown = base.bound(f"attribute:path.unknown:{node.attr}")
                unknown.unknown_callable = True
                return unknown
            return resolve_module_attribute(
                base,
                node.attr,
                module_exports=self.module_exports,
                known_modules=self.known_modules,
                step=f"module-alias:{module}:{node.attr}",
            )
        if isinstance(node, ast.Call):
            return evaluate_call(
                self,
                node,
                module=module,
                actor=actor,
                class_ref=class_ref,
                env=env,
                object_env=object_env,
                call_ordinals=call_ordinals,
            )
        if isinstance(node, (ast.Yield, ast.YieldFrom)) and node.value is not None:
            evaluate_control_expression(
                self,
                node.value,
                kind="yield_from" if isinstance(node, ast.YieldFrom) else "yield",
                module=module,
                actor=actor,
                class_ref=class_ref,
                env=env,
                object_env=object_env,
                call_ordinals=call_ordinals,
            )
            return FlowValue()
        generic = evaluate_generic_expression(
            self,
            node,
            module=module,
            actor=actor,
            class_ref=class_ref,
            env=env,
            object_env=object_env,
            call_ordinals=call_ordinals,
        )
        if generic is not None:
            return generic
        return FlowValue()

    def _merge_param(self, ref: str, parameter: str, value: FlowValue) -> None:
        previous = self.params[ref].get(parameter, FlowValue())
        self.params[ref][parameter] = exclusive_flow_join([previous, value])
        if self.params[ref][parameter] != previous:
            self._persistent_changed = True

    def _mark_called_target(self, target: str) -> None:
        if target in self.called_targets:
            return
        self.called_targets.add(target)
        self._persistent_changed = True

    def _require_function_summary(self, target: str) -> None:
        if target in self.summary_required_targets:
            return
        self.summary_required_targets.add(target)
        self._persistent_changed = True

    def _execute_known_call(
        self,
        target: str,
        argument_values: Sequence[FlowValue],
        keyword_values: Mapping[str, FlowValue],
        *,
        actor: str,
        module: str,
        node: ast.Call,
        ordinal: int,
        env: dict[str, FlowValue],
        object_env: dict[str, set[str]],
        closure_instance: str | None,
        bound_receiver: FlowValue | None = None,
        receiver_out: list[FlowValue] | None = None,
        force: bool = False,
    ) -> FlowValue | None:
        info = self.functions[target]
        caller_is_local = actor in self._active_local_calls
        caller_is_module = actor == f"{module}:<module>"
        summary_only_guarded_call = (
            caller_is_module
            and id(node) in self.main_guard_call_ids
            and info.parent_ref is None
            and info.class_ref is None
            and closure_instance is None
            and bound_receiver is None
            and isinstance(node.func, ast.Name)
            and not _value_has_nested_origins(
                env.get(node.func.id, FlowValue())
            )
            and not any(
                _value_has_nested_origins(value)
                for value in [*argument_values, *keyword_values.values()]
            )
            and not self._function_has_origin_inputs(info)
        )
        if (
            not (
                caller_is_local
                or (caller_is_module and not summary_only_guarded_call)
                or force
            )
            or (
                info.parent_ref is not None
                and info.parent_ref != actor
                and closure_instance is None
            )
            or target in self._active_local_calls
        ):
            return None
        self.progress.enter_known_call(subject=target, limits=self.limits)
        try:
            return self._execute_known_call_active(
                target,
                argument_values,
                keyword_values,
                actor=actor,
                module=module,
                node=node,
                ordinal=ordinal,
                env=env,
                object_env=object_env,
                closure_instance=closure_instance,
                bound_receiver=bound_receiver,
                receiver_out=receiver_out,
            )
        finally:
            self.progress.exit_known_call()

    def _execute_known_call_active(
        self,
        target: str,
        argument_values: Sequence[FlowValue],
        keyword_values: Mapping[str, FlowValue],
        *,
        actor: str,
        module: str,
        node: ast.Call,
        ordinal: int,
        env: dict[str, FlowValue],
        object_env: dict[str, set[str]],
        closure_instance: str | None,
        bound_receiver: FlowValue | None,
        receiver_out: list[FlowValue] | None,
    ) -> FlowValue:
        info = self.functions[target]
        module_env, module_objects, external_state_prefix = self._call_module_state(
            info.module,
            actor=actor,
            module=module,
            env=env,
            object_env=object_env,
        )
        lexical_call = info.parent_ref == actor
        captured_env = (
            self._closure_instance_state(
                closure_instance,
                ref=target,
                path_env=env,
            )
            if closure_instance is not None and not lexical_call
            else None
        )
        if lexical_call:
            base_env = env
            base_objects = object_env
        elif captured_env is not None:
            base_env = captured_env
            base_objects = {
                name: set(value.object_types)
                for name, value in captured_env.items()
                if value.object_types
            }
        else:
            base_env = module_env
            base_objects = module_objects
        frame_env = {name: value.copy() for name, value in base_env.items()}
        frame_objects = {name: set(values) for name, values in base_objects.items()}
        for name, value in env.items():
            if not _is_call_closure_state_name(name):
                continue
            frame_env[name] = value.copy()
            if name in object_env:
                frame_objects[name] = set(object_env[name])
        for name in info.global_names:
            frame_env[name] = module_env.get(name, FlowValue()).copy()
            if name in module_objects:
                frame_objects[name] = set(module_objects[name])
            else:
                frame_objects.pop(name, None)
        for name in info.local_names:
            frame_env[name] = FlowValue()
            frame_objects.pop(name, None)
        values = list(argument_values)
        if info.class_ref is not None:
            receiver_value = bound_receiver or FlowValue(
                object_types={info.class_ref},
                attribute_values_ambiguous=True,
            )
            values = [receiver_value, *values]
        provided: set[str] = set()
        source_name = call_name(node.func) or "<dynamic>"
        site_id = self.facts.site_id(node)
        call_site = f"{source_name}:{ordinal}|site_id={site_id}"
        for index, value in enumerate(values):
            if index >= len(info.parameters):
                break
            parameter = info.parameters[index]
            provided.add(parameter)
            self._bind_local_call_value(
                frame_env,
                frame_objects,
                parameter,
                value,
                step=f"call:{actor}->{target}:{parameter}|site={call_site}",
            )
        for parameter, value in keyword_values.items():
            if parameter not in info.parameters:
                continue
            provided.add(parameter)
            self._bind_local_call_value(
                frame_env,
                frame_objects,
                parameter,
                value,
                step=f"call:{actor}->{target}:{parameter}|site={call_site}",
            )
        for parameter, value in self.definition_defaults[target].items():
            if parameter not in provided:
                self._bind_local_call_value(
                    frame_env,
                    frame_objects,
                    parameter,
                    value,
                    step=f"default:{target}:{parameter}",
                )
        if target not in self.locally_executed_targets:
            self.locally_executed_targets.add(target)
            self._persistent_changed = True
        activation_id = self._call_activation_id(
            target,
            actor=actor,
            module=module,
            source_name=source_name,
            ordinal=ordinal,
            site_id=site_id,
            closure_instance=closure_instance,
        )
        self._active_local_calls.add(target)
        self._active_local_activations[target] = activation_id
        self._local_module_states.append((info.module, module_env, module_objects))
        local_closures: dict[str, dict[str, FlowValue]] = {}
        self._local_closure_states.append(local_closures)
        try:
            result = analyze_block_result(
                self,
                info.node.body,
                module=info.module,
                actor=target,
                class_ref=info.class_ref,
                env=frame_env,
                object_env=frame_objects,
                call_ordinals=info.call_ordinals,
            )
        finally:
            self._local_closure_states.pop()
            self._local_module_states.pop()
            self._active_local_activations.pop(target)
            self._active_local_calls.remove(target)
        self._persist_closure_instances(local_closures)
        exit_states = [
            (outcome.env, outcome.object_env)
            for outcome in result.outcomes
            if outcome.kind in {"normal", "return"}
        ]
        if exit_states:
            exit_env, exit_objects = join_states(exit_states)
            if (
                receiver_out is not None
                and info.class_ref is not None
                and info.parameters
            ):
                receiver_out.append(
                    exit_env.get(info.parameters[0], FlowValue()).copy()
                )
            for name in exit_env:
                if _is_call_path_state_name(name):
                    self._sync_call_name(
                        name,
                        source_env=exit_env,
                        source_objects=exit_objects,
                        target_env=env,
                        target_objects=object_env,
                    )
            if lexical_call:
                for name in info.nonlocal_names:
                    self._sync_call_name(
                        name,
                        source_env=exit_env,
                        source_objects=exit_objects,
                        target_env=env,
                        target_objects=object_env,
                    )
            elif closure_instance is not None and info.nonlocal_names:
                updated_capture = {
                    name: value.copy() for name, value in (captured_env or {}).items()
                }
                updated_objects = {
                    name: set(value.object_types)
                    for name, value in updated_capture.items()
                    if value.object_types
                }
                for name in info.nonlocal_names:
                    self._sync_call_name(
                        name,
                        source_env=exit_env,
                        source_objects=exit_objects,
                        target_env=updated_capture,
                        target_objects=updated_objects,
                    )
                self._store_closure_group_state(
                    closure_instance,
                    updated_capture,
                    updated_objects,
                    env=env,
                    object_env=object_env,
                )
            for name in info.global_names:
                self._sync_call_name(
                    name,
                    source_env=exit_env,
                    source_objects=exit_objects,
                    target_env=module_env,
                    target_objects=module_objects,
                )
            self._propagate_stdlib_module_mutations(
                exit_env,
                env=env,
                object_env=object_env,
            )
        self._store_local_closure_groups(
            local_closures,
            env=env,
            object_env=object_env,
        )
        if external_state_prefix is not None:
            self._store_call_module_state(
                external_state_prefix,
                module_env,
                module_objects,
                env=env,
                object_env=object_env,
            )
        return result.returned

    def _propagate_stdlib_module_mutations(
        self,
        source_env: Mapping[str, FlowValue],
        *,
        env: dict[str, FlowValue],
        object_env: dict[str, set[str]],
    ) -> None:
        for module_ref in sorted(SUPPORTED_STDLIB_MODULES):
            attributes = frozenset().union(
                *(
                    stdlib_module_mutation_attributes(
                        value, module=module_ref
                    )
                    for value in source_env.values()
                )
            )
            if not attributes:
                continue
            base = next(
                (
                    value
                    for value in env.values()
                    if precise_stdlib_module_name(value) == module_ref
                ),
                FlowValue(module_refs={module_ref}),
            )
            for attribute in sorted(attributes):
                self._taint_stdlib_module_attribute(
                    base,
                    attribute=attribute,
                    env=env,
                    object_env=object_env,
                )

    def _call_activation_id(
        self,
        target: str,
        *,
        actor: str,
        module: str,
        source_name: str,
        ordinal: int,
        site_id: str,
        closure_instance: str | None,
    ) -> str:
        parent_activation = self._active_local_activations.get(
            actor, f"module:{module}"
        )
        callable_identity = closure_instance or target
        return (
            f"{parent_activation}|call:{actor}->{callable_identity}"
            f"|site={source_name}:{ordinal}|site_id={site_id}"
        )

    def _closure_instance_state(
        self,
        instance_id: str,
        *,
        ref: str,
        path_env: Mapping[str, FlowValue] | None = None,
    ) -> dict[str, FlowValue]:
        info = self.functions[ref]
        visible_cells = info.referenced_names | info.nonlocal_names
        group_id = self.closure_instance_groups.get(instance_id)
        if group_id is not None and path_env is not None:
            prefix = _call_closure_state_prefix(group_id)
            path_state = {
                name.removeprefix(prefix): value.copy()
                for name, value in path_env.items()
                if name.startswith(prefix)
                and name.removeprefix(prefix) in visible_cells
            }
            if path_state:
                return path_state
        for local_instances in reversed(self._local_closure_states):
            state = local_instances.get(instance_id)
            if state is not None:
                live_state = {
                    name: value.copy()
                    for name, value in state.items()
                    if name in visible_cells
                }
                if (
                    group_id is not None
                    and path_env is not None
                    and group_id in self._active_local_activations.values()
                ):
                    for name in live_state:
                        live_state[name] = path_env.get(name, FlowValue()).copy()
                return live_state
        state = self.closure_instance_envs.get(instance_id, {})
        return {
            name: value.copy() for name, value in state.items() if name in visible_cells
        }

    def _closure_capture_value(
        self,
        value: FlowValue,
        *,
        env: Mapping[str, FlowValue],
    ) -> FlowValue:
        captured = FlowValue()
        for ref, instance_id in sorted(value.closure_instances):
            for name, capture in sorted(
                self._closure_instance_state(
                    instance_id,
                    ref=ref,
                    path_env=env,
                ).items()
            ):
                if capture.has_origins:
                    captured = captured.merged(
                        capture.bound(
                            f"closure-escape:{ref}:{name}|instance={instance_id}"
                        )
                    )
        return captured

    def _store_local_closure_groups(
        self,
        instances: Mapping[str, Mapping[str, FlowValue]],
        *,
        env: dict[str, FlowValue],
        object_env: dict[str, set[str]],
    ) -> None:
        groups: dict[str, dict[str, FlowValue]] = {}
        for instance_id, state in instances.items():
            group_id = self.closure_instance_groups.get(instance_id)
            if group_id is None:
                continue
            group = groups.setdefault(group_id, {})
            for name, value in state.items():
                group[name] = group.get(name, FlowValue()).merged(value)
        for group_id, state in groups.items():
            self._store_closure_group(
                group_id,
                state,
                env=env,
                object_env=object_env,
            )

    def _store_closure_group_state(
        self,
        instance_id: str,
        state: Mapping[str, FlowValue],
        state_objects: Mapping[str, set[str]],
        *,
        env: dict[str, FlowValue],
        object_env: dict[str, set[str]],
    ) -> None:
        group_id = self.closure_instance_groups.get(instance_id)
        if group_id is None:
            return
        self._store_closure_group(
            group_id,
            state,
            state_objects=state_objects,
            env=env,
            object_env=object_env,
        )

    @staticmethod
    def _store_closure_group(
        group_id: str,
        state: Mapping[str, FlowValue],
        *,
        env: dict[str, FlowValue],
        object_env: dict[str, set[str]],
        state_objects: Mapping[str, set[str]] | None = None,
    ) -> None:
        prefix = _call_closure_state_prefix(group_id)
        for name in list(env):
            if name.startswith(prefix):
                env.pop(name, None)
                object_env.pop(name, None)
        for name, value in state.items():
            state_name = f"{prefix}{name}"
            env[state_name] = value.copy()
            if state_objects is not None and name in state_objects:
                object_env[state_name] = set(state_objects[name])

    def _persist_closure_instances(
        self,
        instances: Mapping[str, Mapping[str, FlowValue]],
    ) -> None:
        for instance_id, state in instances.items():
            previous = self.closure_instance_envs.get(instance_id, {})
            persisted: dict[str, FlowValue] = {}
            for name in previous.keys() | state.keys():
                persisted[name] = previous.get(name, FlowValue()).merged(
                    state.get(name, FlowValue())
                )
            if previous == persisted:
                continue
            self.closure_instance_envs[instance_id] = persisted
            self._persistent_changed = True

    def _call_module_state(
        self,
        target_module: str,
        *,
        actor: str,
        module: str,
        env: dict[str, FlowValue],
        object_env: dict[str, set[str]],
    ) -> tuple[dict[str, FlowValue], dict[str, set[str]], str | None]:
        for state_module, state_env, state_objects in reversed(
            self._local_module_states
        ):
            if state_module == target_module:
                self._propagate_stdlib_module_mutations(
                    env,
                    env=state_env,
                    object_env=state_objects,
                )
                return state_env, state_objects, None
        if actor == f"{module}:<module>" and target_module == module:
            return env, object_env, None
        baseline = self.module_runtime_envs[target_module]
        module_env = {name: value.copy() for name, value in baseline.items()}
        module_objects = {
            name: set(value.object_types)
            for name, value in module_env.items()
            if value.object_types
        }
        state_prefix = _call_module_state_prefix(target_module)
        for name, value in env.items():
            if not name.startswith(state_prefix):
                continue
            module_name = name.removeprefix(state_prefix)
            module_env[module_name] = value.copy()
            if name in object_env:
                module_objects[module_name] = set(object_env[name])
            else:
                module_objects.pop(module_name, None)
        self._propagate_stdlib_module_mutations(
            env,
            env=module_env,
            object_env=module_objects,
        )
        self._reconcile_stdlib_import_captures(
            target_module,
            env=module_env,
            object_env=module_objects,
        )
        return module_env, module_objects, state_prefix

    def _reconcile_stdlib_import_captures(
        self,
        module: str,
        *,
        env: dict[str, FlowValue],
        object_env: dict[str, set[str]],
    ) -> None:
        mutations = {
            module_ref: frozenset().union(
                *(
                    stdlib_module_mutation_attributes(
                        value,
                        module=module_ref,
                    )
                    for value in env.values()
                )
            )
            for module_ref in SUPPORTED_STDLIB_MODULES
        }
        for local, (module_ref, attribute) in self.imported_symbols.get(
            module,
            {},
        ).items():
            mutated = mutations.get(module_ref, frozenset())
            if not ({attribute, "*"} & set(mutated)):
                continue
            candidate = env.get(local)
            target = f"{module_ref}:{attribute}"
            if candidate is None or candidate.call_targets != {target}:
                continue
            unknown = candidate.copy()
            unknown.call_targets.clear()
            unknown.object_types.add(
                stdlib_call_target_marker(module_ref, attribute)
            )
            unknown.unknown_callable = True
            env[local] = unknown
            object_env[local] = set(unknown.object_types)

    @staticmethod
    def _store_call_module_state(
        state_prefix: str,
        module_env: Mapping[str, FlowValue],
        module_objects: Mapping[str, set[str]],
        *,
        env: dict[str, FlowValue],
        object_env: dict[str, set[str]],
    ) -> None:
        for name in list(env):
            if name.startswith(state_prefix):
                env.pop(name, None)
                object_env.pop(name, None)
        for name, value in module_env.items():
            state_name = f"{state_prefix}{name}"
            env[state_name] = value.copy()
            if name in module_objects:
                object_env[state_name] = set(module_objects[name])

    @staticmethod
    def _bind_local_call_value(
        env: dict[str, FlowValue],
        object_env: dict[str, set[str]],
        name: str,
        value: FlowValue,
        *,
        step: str,
    ) -> None:
        bound = value.bound(step)
        env[name] = bound
        if bound.object_types:
            object_env[name] = set(bound.object_types)
        else:
            object_env.pop(name, None)

    @staticmethod
    def _sync_call_name(
        name: str,
        *,
        source_env: Mapping[str, FlowValue],
        source_objects: Mapping[str, set[str]],
        target_env: dict[str, FlowValue],
        target_objects: dict[str, set[str]],
    ) -> None:
        target_env[name] = source_env.get(name, FlowValue()).copy()
        if name in source_objects:
            target_objects[name] = set(source_objects[name])
        else:
            target_objects.pop(name, None)

    def record_dynamic_star_import(
        self,
        statement: ast.ImportFrom,
        *,
        module: str,
        actor: str,
        target_module: str,
        value: FlowValue,
    ) -> None:
        if not value.has_origins:
            return
        self.facts.record_escape(
            value,
            node=statement,
            actor=actor,
            operation=f"import:*:{target_module}",
            sink=f"module:{target_module}",
            reason="dynamic_star_import",
            path=self.paths[module],
            line=int(statement.lineno),
            ordinal=(int(statement.lineno) << 20) + int(statement.col_offset) + 1,
        )

    def _bind_call_parameter(
        self,
        target: str,
        parameter: str,
        value: FlowValue,
        *,
        actor: str,
        module: str,
        node: ast.Call,
        ordinal: int,
        site_node: ast.AST | None = None,
    ) -> None:
        physical_node = site_node or node
        self._require_function_summary(target)
        safe, cyclic = value.partition_call_cycles(target=target)
        if cyclic.has_origins and target not in self.recursive_base_targets:
            self.facts.record_escape(
                cyclic,
                node=physical_node,
                actor=actor,
                operation=f"call:{target}",
                sink=target,
                reason="binding_cycle",
                path=self.paths[module],
                line=int(node.lineno),
                ordinal=ordinal,
            )
        if safe.has_analysis_state:
            source_name = call_name(node.func) or "<dynamic>"
            site_id = self.facts.site_id(physical_node)
            step = (
                f"call:{actor}->{target}:{parameter}|site={source_name}:{ordinal}"
                f"|site_id={site_id}"
            )
            self._merge_param(target, parameter, safe.bound(step))


def _uses_future_annotations(tree: ast.Module) -> bool:
    return any(
        isinstance(statement, ast.ImportFrom)
        and statement.module == "__future__"
        and any(alias.name == "annotations" for alias in statement.names)
        for statement in tree.body
    )


def _is_declared_dataclass_decorator(
    expression: ast.expr,
    *,
    imported_symbols: Mapping[str, tuple[str, str]],
    imported_modules: Mapping[str, str],
) -> bool:
    callable_expression = expression.func if isinstance(expression, ast.Call) else expression
    if isinstance(callable_expression, ast.Name):
        return imported_symbols.get(callable_expression.id) == (
            "dataclasses",
            "dataclass",
        )
    return (
        isinstance(callable_expression, ast.Attribute)
        and isinstance(callable_expression.value, ast.Name)
        and callable_expression.attr == "dataclass"
        and imported_modules.get(callable_expression.value.id) == "dataclasses"
    )


def discover_access_facts(
    snapshot: Mapping[str, bytes],
    resource_candidates: Iterable[Mapping[str, Any]],
    *,
    limits: AnalysisLimits | None = None,
    progress: AnalysisProgress | None = None,
    optimize_gc: bool | None = None,
) -> dict[str, Any]:
    """Discover concrete access and fail-closed escape facts."""

    resolved_limits = AnalysisLimits() if limits is None else limits
    resolved_progress = AnalysisProgress() if progress is None else progress
    if not isinstance(resolved_limits, AnalysisLimits):
        raise TypeError("limits must be AnalysisLimits or None")
    if not isinstance(resolved_progress, AnalysisProgress):
        raise TypeError("progress must be AnalysisProgress or None")
    if optimize_gc is not None and type(optimize_gc) is not bool:
        raise ValueError("optimize_gc must be an exact bool or None")
    resolved_progress.reset()
    analysis = _AccessAnalysis(
        snapshot,
        resource_candidates,
        limits=resolved_limits,
        progress=resolved_progress,
    )
    should_optimize_gc = (
        _default_optimize_gc() if optimize_gc is None else optimize_gc
    )
    if not should_optimize_gc:
        return analysis.run()
    with _scoped_gc_disabled():
        return analysis.run()


def _default_optimize_gc() -> bool:
    if sys.implementation.name != "cpython":
        return False
    is_gil_enabled = getattr(sys, "_is_gil_enabled", None)
    return is_gil_enabled is None or bool(is_gil_enabled())


@contextmanager
def _scoped_gc_disabled() -> Iterator[None]:
    """Disable cyclic GC during one run and restore only its prior enabled state.

    Threshold, debug, and callback changes made during the body, including by
    other threads, are intentionally preserved so this helper does not clobber
    unrelated process configuration.
    """

    with _GC_OPTIMIZATION_LOCK:
        was_enabled = gc.isenabled()
        if was_enabled:
            gc.disable()
        try:
            yield
        finally:
            if was_enabled:
                gc.enable()
            else:
                gc.disable()


__all__ = [
    "AnalysisLimits",
    "AnalysisNonConvergenceError",
    "AnalysisProgress",
    "FlowValue",
    "FunctionInfo",
    "discover_access_facts",
]
