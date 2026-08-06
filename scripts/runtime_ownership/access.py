"""Discover line-independent runtime-state access and fail-closed escapes."""

from __future__ import annotations

import ast
from collections.abc import Iterable, Mapping
from typing import Any

from .access_expressions import evaluate_generic_expression
from .access_facts import AccessFactCollector
from .access_imports import build_module_exports, resolve_module_attribute
from .access_model import (
    FlowValue,
    FunctionInfo,
    _call_ordinals,
    _collect_functions,
    _import_tables,
    _module_name,
)
from .access_resolver import call_name
from .access_sinks import evaluate_call
from .access_statements import analyze_block, evaluate_control_expression


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
        self.nested_functions = {
            (info.parent_ref, info.node.name): ref
            for ref, info in self.functions.items()
            if info.parent_ref is not None
        }
        self.closure_envs: dict[str, dict[str, FlowValue]] = {
            ref: {} for ref in self.functions
        }
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
        self.class_attrs: dict[str, dict[str, FlowValue]] = {}
        self.class_comprehension_parents: dict[
            str,
            tuple[dict[str, FlowValue], dict[str, set[str]]],
        ] = {}
        self.facts = AccessFactCollector(self.resource_locators)
        self._persistent_changed = False

    def run(self) -> dict[str, Any]:
        for _iteration in range(32):
            self._persistent_changed = False
            for module, tree in sorted(self.trees.items()):
                self._analyze_module(module, tree)
            for ref, info in sorted(self.functions.items()):
                if info.parent_ref is not None and ref not in self.called_targets:
                    continue
                self._analyze_function(info)
            if not self._persistent_changed:
                break
        else:
            raise ValueError("runtime access fixed point did not converge")
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
        runtime_env = {name: value.copy() for name, value in env.items()}
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
        changed = False
        closure = self.closure_envs[ref]
        module_env = self.module_runtime_envs[module]
        info = self.functions[ref]
        if info.parent_ref is None:
            return False
        for name, value in env.items():
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
    ) -> bool:
        if isinstance(target, ast.Name):
            bound = value.bound(f"alias:{actor}:{target.id}")
            env[target.id] = bound
            if value.object_types:
                object_env[target.id] = set(value.object_types)
            else:
                object_env.pop(target.id, None)
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
    ) -> None:
        safe, cyclic = value.partition_call_cycles(target=target)
        if cyclic.has_origins:
            self.facts.record_escape(
                cyclic,
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
            step = f"call:{actor}->{target}:{parameter}|site={source_name}:{ordinal}"
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
