"""Lexical and imported call-target resolution for access discovery."""

from __future__ import annotations

import ast
from collections.abc import Mapping
from dataclasses import dataclass

from .access_model import FlowValue, FunctionInfo


@dataclass(frozen=True)
class CallResolution:
    known_targets: tuple[str, ...] = ()
    fallback_target: str = ""
    has_unknown: bool = False


def call_name(expression: ast.expr) -> str:
    if isinstance(expression, ast.Name):
        return expression.id
    if isinstance(expression, ast.Attribute):
        prefix = call_name(expression.value)
        return f"{prefix}.{expression.attr}" if prefix else expression.attr
    return ""


def resolve_call_target(
    expression: ast.expr,
    *,
    module: str,
    class_ref: str | None,
    env: Mapping[str, FlowValue],
    object_env: Mapping[str, set[str]],
    actor: str,
    functions: Mapping[str, FunctionInfo],
    nested_functions: Mapping[tuple[str, str], str],
    function_parents: Mapping[str, str | None],
    imported_symbols: Mapping[str, Mapping[str, tuple[str, str]]],
    imported_modules: Mapping[str, Mapping[str, str]],
) -> CallResolution:
    if isinstance(expression, ast.Name):
        if expression.id in env:
            value = env[expression.id]
            call_targets = value.call_targets
            if call_targets:
                known_targets = tuple(
                    sorted(target for target in call_targets if target in functions)
                )
                return CallResolution(
                    known_targets,
                    has_unknown=(
                        value.unknown_callable
                        or len(known_targets) != len(call_targets)
                    ),
                )
            return CallResolution(has_unknown=True)
        actor_info = functions.get(actor)
        if actor_info is not None and (
            expression.id in actor_info.parameters
            or expression.id in actor_info.local_names
        ):
            return CallResolution(has_unknown=True)
        scope: str | None = actor if actor in functions else None
        while scope is not None:
            nested = nested_functions.get((scope, expression.id))
            if nested is not None:
                return _call_resolution(nested, functions)
            scope = function_parents.get(scope)
        local = f"{module}:{expression.id}"
        if local in functions:
            return CallResolution((local,))
        imported = imported_symbols.get(module, {}).get(expression.id)
        if imported is not None:
            return _call_resolution(f"{imported[0]}:{imported[1]}", functions)
        return CallResolution(has_unknown=True)
    if isinstance(expression, ast.Attribute) and isinstance(expression.value, ast.Name):
        if expression.value.id == "self" and class_ref is not None:
            return _call_resolution(f"{class_ref}.{expression.attr}", functions)
        base_name = expression.value.id
        if base_name in env:
            base_value = env[base_name]
            candidates = {
                *(
                    f"{module_ref}:{expression.attr}"
                    for module_ref in base_value.module_refs
                ),
                *(
                    f"{object_type}.{expression.attr}"
                    for object_type in base_value.object_types
                ),
            }
            known_targets = tuple(
                sorted(target for target in candidates if target in functions)
            )
            unresolved_targets = tuple(
                sorted(target for target in candidates if target not in functions)
            )
            fallback_target = ""
            if not known_targets and len(unresolved_targets) == 1:
                fallback_target = unresolved_targets[0]
            return CallResolution(
                known_targets=known_targets,
                fallback_target=fallback_target,
                has_unknown=(
                    base_value.unknown_callable
                    or bool(unresolved_targets)
                    or not candidates
                ),
            )
        imported_module = imported_modules.get(module, {}).get(base_name)
        if imported_module is not None:
            return _call_resolution(f"{imported_module}:{expression.attr}", functions)
        object_types = object_env.get(base_name, set())
        if len(object_types) == 1:
            return _call_resolution(
                f"{next(iter(object_types))}.{expression.attr}", functions
            )
    return CallResolution(has_unknown=True)


def _call_resolution(
    target: str,
    functions: Mapping[str, FunctionInfo],
) -> CallResolution:
    if target in functions:
        return CallResolution((target,))
    return CallResolution(fallback_target=target)


def resolve_class_target(
    expression: ast.expr,
    *,
    module: str,
    classes: Mapping[str, Mapping[str, str]],
    imported_symbols: Mapping[str, Mapping[str, tuple[str, str]]],
    imported_modules: Mapping[str, Mapping[str, str]],
) -> str | None:
    if isinstance(expression, ast.Name):
        local = classes.get(module, {}).get(expression.id)
        if local is not None:
            return local
        imported = imported_symbols.get(module, {}).get(expression.id)
        if imported is not None:
            target = f"{imported[0]}:{imported[1]}"
            if any(target in values.values() for values in classes.values()):
                return target
    if isinstance(expression, ast.Attribute) and isinstance(expression.value, ast.Name):
        imported_module = imported_modules.get(module, {}).get(expression.value.id)
        if imported_module is not None:
            target = f"{imported_module}:{expression.attr}"
            if target in classes.get(imported_module, {}).values():
                return target
    return None


__all__ = [
    "CallResolution",
    "call_name",
    "resolve_call_target",
    "resolve_class_target",
]
