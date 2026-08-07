"""Discover line-independent runtime-state access and fail-closed escapes."""

from __future__ import annotations

import ast
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from .access_expressions import evaluate_generic_expression
from .access_facts import AccessFactCollector
from .access_imports import build_module_exports, resolve_module_attribute
from .access_model import (
    SQLITE_HANDLE_OBJECT_TYPES,
    SQLITE_UNKNOWN_ATTRIBUTE_OBJECT_TYPE,
    SUPPORTED_STDLIB_MODULES,
    FlowValue,
    FunctionInfo,
    _call_ordinals,
    _collect_functions,
    _collect_syntax_sites,
    _import_tables,
    _module_name,
    precise_stdlib_module_name,
    project_sqlite_attribute,
    stdlib_module_mutation_attributes,
    stdlib_module_mutation_marker,
    stdlib_module_state_name,
)
from .access_outcomes import analyze_block_result, join_states
from .access_resolver import call_name
from .access_sinks import evaluate_call
from .access_statements import analyze_block, evaluate_control_expression

_CALL_MODULE_STATE_PREFIX = "\0runtime-module-state:"
_CALL_CLOSURE_STATE_PREFIX = "\0runtime-closure-state:"


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


class _AccessAnalysis:
    def __init__(
        self,
        snapshot: Mapping[str, bytes],
        resource_candidates: Iterable[Mapping[str, Any]],
    ) -> None:
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
        self.future_annotations = frozenset(
            module
            for module, tree in self.trees.items()
            if _uses_future_annotations(tree)
        )
        known_modules = set(self.trees)
        for module in self.trees:
            parts = module.split(".")
            known_modules.update(
                ".".join(parts[:index]) for index in range(1, len(parts))
            )
        self.known_modules = frozenset(known_modules)
        self.origin_symbols: dict[tuple[str, str], FlowValue] = {}
        self.resource_locators: dict[str, str] = {}
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
            self.origin_symbols[origin_key] = FlowValue(
                {resource_id: frozenset({(f"origin:{module}:{symbol}",)})}
            )
        export_table = build_module_exports(
            self.trees,
            package_modules=self.package_modules,
            known_modules=self.known_modules,
            origin_symbols=self.origin_symbols,
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
        self._active_local_calls: set[str] = set()
        self._active_local_activations: dict[str, str] = {}
        self._local_module_states: list[
            tuple[str, dict[str, FlowValue], dict[str, set[str]]]
        ] = []
        self._local_closure_states: list[dict[str, dict[str, FlowValue]]] = []
        self.class_attrs: dict[str, dict[str, FlowValue]] = {}
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
        while True:
            self._persistent_changed = False
            for module, tree in sorted(self.trees.items()):
                self._analyze_module(module, tree)
            for ref, info in sorted(self.functions.items()):
                if (
                    ref in self.locally_executed_targets
                    and ref not in self.summary_required_targets
                ):
                    continue
                if info.parent_ref is not None and ref not in self.called_targets:
                    continue
                self._analyze_function(info)
            if not self._persistent_changed:
                break
        return self.facts.result()

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
        env = {
            name: value.copy()
            for name, value in module_env.items()
            if name not in info.local_names or name in info.global_names
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
            if default_value.has_origins:
                env[parameter] = env.get(parameter, FlowValue()).merged(
                    default_value.bound(f"default:{info.ref}:{parameter}")
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
        merged_return = self.returns[info.ref].merged(returned)
        if merged_return != self.returns[info.ref]:
            self.returns[info.ref] = merged_return
            self._persistent_changed = True
            changed = True
        return changed

    def _merge_definition_default(
        self,
        ref: str,
        parameter: str,
        value: FlowValue,
    ) -> bool:
        previous = self.definition_defaults[ref].get(parameter, FlowValue())
        merged = previous.merged(value)
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
                not value.has_origins
                and not value.object_types
                and not value.module_refs
            ):
                continue
            previous = closure.get(name, FlowValue())
            closure[name] = previous.merged(
                value.bound(f"closure:{actor}->{ref}:{name}")
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
        if isinstance(target, ast.Attribute) and isinstance(
            target.value, ast.Name
        ):
            base = env.get(target.value.id, FlowValue())
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
            class_ref is not None
            and isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "self"
        ):
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
            if (
                isinstance(node.value, ast.Name)
                and node.value.id == "self"
                and class_ref is not None
            ):
                value = self.class_attrs.get(class_ref, {}).get(node.attr)
                if value is not None:
                    return value.copy()
            base = self._eval(
                node.value,
                module=module,
                actor=actor,
                class_ref=class_ref,
                env=env,
                object_env=object_env,
                call_ordinals=call_ordinals,
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
        self.params[ref][parameter] = previous.merged(value)
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
    ) -> FlowValue | None:
        info = self.functions[target]
        caller_is_local = actor in self._active_local_calls
        caller_is_module = actor == f"{module}:<module>"
        if (
            not (caller_is_local or caller_is_module)
            or (
                info.parent_ref is not None
                and info.parent_ref != actor
                and closure_instance is None
            )
            or target in self._active_local_calls
        ):
            return None
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
        if info.class_ref is not None and not target.endswith(".__init__"):
            values = [FlowValue(object_types={info.class_ref}), *values]
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
        return module_env, module_objects, state_prefix

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
        if safe.has_origins:
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


def discover_access_facts(
    snapshot: Mapping[str, bytes],
    resource_candidates: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Discover concrete access and fail-closed escape facts."""

    return _AccessAnalysis(snapshot, resource_candidates).run()


__all__ = [
    "FlowValue",
    "FunctionInfo",
    "discover_access_facts",
]
