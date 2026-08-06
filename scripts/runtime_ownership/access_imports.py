"""Import binding and module re-export resolution for runtime access analysis."""

from __future__ import annotations

import ast
from collections.abc import Mapping
from typing import Protocol

from .access_model import FlowValue

SymbolKey = tuple[str, str]


class ImportEngine(Protocol):
    module_exports: dict[str, dict[str, FlowValue]]
    known_modules: frozenset[str]
    package_modules: frozenset[str]


class _ExportView:
    def __init__(
        self,
        module_exports: Mapping[str, Mapping[str, FlowValue]],
        *,
        known_modules: frozenset[str],
        package_modules: frozenset[str],
    ) -> None:
        self.module_exports = {
            name: dict(values) for name, values in module_exports.items()
        }
        self.known_modules = known_modules
        self.package_modules = package_modules


def resolve_import_from(
    module: str,
    *,
    level: int,
    imported_module: str | None,
    is_package: bool,
) -> str:
    if level == 0:
        return imported_module or ""
    package = module.split(".") if is_package else module.split(".")[:-1]
    parent_hops = level - 1
    if parent_hops > len(package):
        return imported_module or ""
    prefix = package[: len(package) - parent_hops]
    if imported_module:
        prefix.extend(imported_module.split("."))
    return ".".join(prefix)


def bind_import_statement(
    engine: ImportEngine,
    statement: ast.Import | ast.ImportFrom,
    *,
    module: str,
    actor: str,
    env: dict[str, FlowValue],
    object_env: dict[str, set[str]],
) -> None:
    if isinstance(statement, ast.Import):
        for alias in statement.names:
            local = alias.asname or alias.name.split(".")[0]
            imported = alias.name if alias.asname else alias.name.split(".")[0]
            _strong_bind(
                env,
                object_env,
                local,
                FlowValue(module_refs={imported}),
                step=f"import:{actor}:{local}->{alias.name}",
            )
        return

    target_module = resolve_import_from(
        module,
        level=statement.level,
        imported_module=statement.module,
        is_package=module in engine.package_modules,
    )
    for alias in statement.names:
        if alias.name == "*":
            continue
        local = alias.asname or alias.name
        value = engine.module_exports.get(target_module, {}).get(
            alias.name, FlowValue()
        )
        child_module = f"{target_module}.{alias.name}" if target_module else alias.name
        if (
            not value.has_origins
            and not value.module_refs
            and child_module in engine.known_modules
        ):
            value = FlowValue(module_refs={child_module})
        _strong_bind(
            env,
            object_env,
            local,
            value,
            step=f"import:{actor}:{local}->{target_module}:{alias.name}",
        )


def build_module_exports(
    trees: Mapping[str, ast.Module],
    *,
    package_modules: frozenset[str],
    known_modules: frozenset[str],
    origin_symbols: Mapping[SymbolKey, FlowValue],
) -> dict[str, dict[str, FlowValue]]:
    dependencies = _export_dependencies(
        trees,
        package_modules=package_modules,
        known_modules=known_modules,
    )
    _reject_origin_cycles(dependencies, frozenset(origin_symbols))

    exports: dict[str, dict[str, FlowValue]] = {module: {} for module in trees}
    while True:
        next_exports = {
            module: _evaluate_module_exports(
                module,
                tree,
                exports=exports,
                package_modules=package_modules,
                known_modules=known_modules,
                origin_symbols=origin_symbols,
            )
            for module, tree in sorted(trees.items())
        }
        if next_exports == exports:
            return next_exports
        exports = next_exports


def resolve_module_attribute(
    value: FlowValue,
    attribute: str,
    *,
    module_exports: Mapping[str, Mapping[str, FlowValue]],
    known_modules: frozenset[str],
    step: str,
) -> FlowValue:
    resolved = FlowValue()
    for module_ref in sorted(value.module_refs):
        exported = module_exports.get(module_ref, {}).get(attribute)
        if exported is not None:
            resolved = resolved.merged(exported.bound(step))
        child_module = f"{module_ref}.{attribute}"
        if child_module in known_modules:
            resolved = resolved.merged(FlowValue(module_refs={child_module}))
    return resolved


def _evaluate_module_exports(
    module: str,
    tree: ast.Module,
    *,
    exports: Mapping[str, Mapping[str, FlowValue]],
    package_modules: frozenset[str],
    known_modules: frozenset[str],
    origin_symbols: Mapping[SymbolKey, FlowValue],
) -> dict[str, FlowValue]:
    env: dict[str, FlowValue] = {}
    object_env: dict[str, set[str]] = {}

    view = _ExportView(
        exports,
        known_modules=known_modules,
        package_modules=package_modules,
    )
    for statement in tree.body:
        if isinstance(statement, (ast.Import, ast.ImportFrom)):
            bind_import_statement(
                view,
                statement,
                module=module,
                actor=f"{module}:<module>",
                env=env,
                object_env=object_env,
            )
        elif isinstance(statement, (ast.Assign, ast.AnnAssign)):
            value_node = statement.value
            if value_node is None:
                continue
            value = _evaluate_export_expression(
                value_node,
                env=env,
                exports=exports,
                known_modules=known_modules,
                step=f"module-alias:{module}",
            )
            targets = (
                statement.targets
                if isinstance(statement, ast.Assign)
                else [statement.target]
            )
            for target in targets:
                for name in _target_names(target):
                    direct = origin_symbols.get((module, name))
                    _strong_bind(
                        env,
                        object_env,
                        name,
                        direct.copy() if direct is not None else value,
                        step=f"alias:{module}:<module>:{name}",
                    )
        elif isinstance(
            statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            env.pop(statement.name, None)
            object_env.pop(statement.name, None)
    for (origin_module, symbol), value in origin_symbols.items():
        if origin_module == module and symbol not in env:
            env[symbol] = value.copy()
    return env


def _evaluate_export_expression(
    node: ast.expr,
    *,
    env: Mapping[str, FlowValue],
    exports: Mapping[str, Mapping[str, FlowValue]],
    known_modules: frozenset[str],
    step: str,
) -> FlowValue:
    if isinstance(node, ast.Name):
        return env.get(node.id, FlowValue()).copy()
    if isinstance(node, ast.Attribute):
        base = _evaluate_export_expression(
            node.value,
            env=env,
            exports=exports,
            known_modules=known_modules,
            step=step,
        )
        return resolve_module_attribute(
            base,
            node.attr,
            module_exports=exports,
            known_modules=known_modules,
            step=f"{step}.{node.attr}",
        )
    return FlowValue()


def _strong_bind(
    env: dict[str, FlowValue],
    object_env: dict[str, set[str]],
    name: str,
    value: FlowValue,
    *,
    step: str,
) -> None:
    env[name] = value.bound(step)
    if value.object_types:
        object_env[name] = set(value.object_types)
    else:
        object_env.pop(name, None)


def _target_names(target: ast.expr) -> list[str]:
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, (ast.Tuple, ast.List)):
        return [name for item in target.elts for name in _target_names(item)]
    return []


def _export_dependencies(
    trees: Mapping[str, ast.Module],
    *,
    package_modules: frozenset[str],
    known_modules: frozenset[str],
) -> dict[SymbolKey, SymbolKey]:
    dependencies: dict[SymbolKey, SymbolKey] = {}
    for module, tree in sorted(trees.items()):
        module_aliases: dict[str, str] = {}
        symbol_aliases: dict[str, SymbolKey] = {}
        for statement in tree.body:
            if isinstance(statement, ast.Import):
                for alias in statement.names:
                    local = alias.asname or alias.name.split(".")[0]
                    module_aliases[local] = (
                        alias.name if alias.asname else alias.name.split(".")[0]
                    )
                    symbol_aliases.pop(local, None)
                    dependencies.pop((module, local), None)
            elif isinstance(statement, ast.ImportFrom):
                target_module = resolve_import_from(
                    module,
                    level=statement.level,
                    imported_module=statement.module,
                    is_package=module in package_modules,
                )
                for alias in statement.names:
                    if alias.name == "*":
                        continue
                    local = alias.asname or alias.name
                    child = (
                        f"{target_module}.{alias.name}" if target_module else alias.name
                    )
                    if child in known_modules:
                        module_aliases[local] = child
                        symbol_aliases.pop(local, None)
                        dependencies.pop((module, local), None)
                    else:
                        source = (target_module, alias.name)
                        symbol_aliases[local] = source
                        module_aliases.pop(local, None)
                        dependencies[(module, local)] = source
            elif isinstance(statement, (ast.Assign, ast.AnnAssign)):
                if statement.value is None:
                    continue
                assignment_source = _dependency_expression(
                    statement.value,
                    module=module,
                    module_aliases=module_aliases,
                    symbol_aliases=symbol_aliases,
                )
                targets = (
                    statement.targets
                    if isinstance(statement, ast.Assign)
                    else [statement.target]
                )
                for target in targets:
                    for name in _target_names(target):
                        module_aliases.pop(name, None)
                        if assignment_source is None:
                            symbol_aliases.pop(name, None)
                            dependencies.pop((module, name), None)
                        else:
                            symbol_aliases[name] = assignment_source
                            dependencies[(module, name)] = assignment_source
    return dependencies


def _dependency_expression(
    node: ast.expr,
    *,
    module: str,
    module_aliases: Mapping[str, str],
    symbol_aliases: Mapping[str, SymbolKey],
) -> SymbolKey | None:
    if isinstance(node, ast.Name):
        return symbol_aliases.get(node.id, (module, node.id))
    if not isinstance(node, ast.Attribute):
        return None
    parts: list[str] = []
    current: ast.expr = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name) or not parts:
        return None
    base_module = module_aliases.get(current.id)
    if base_module is None:
        return None
    parts.reverse()
    if len(parts) == 1:
        return base_module, parts[0]
    return ".".join([base_module, *parts[:-1]]), parts[-1]


def _reject_origin_cycles(
    dependencies: Mapping[SymbolKey, SymbolKey], origins: frozenset[SymbolKey]
) -> None:
    for origin in sorted(origins):
        path: list[SymbolKey] = []
        positions: dict[SymbolKey, int] = {}
        current = origin
        while current in dependencies:
            if current in positions:
                cycle = [*path[positions[current] :], current]
                rendered = " -> ".join(f"{module}:{symbol}" for module, symbol in cycle)
                raise ValueError(f"runtime access import cycle: {rendered}")
            positions[current] = len(path)
            path.append(current)
            current = dependencies[current]


__all__ = [
    "ImportEngine",
    "bind_import_statement",
    "build_module_exports",
    "resolve_import_from",
    "resolve_module_attribute",
]
