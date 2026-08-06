"""Lexical and imported call-target resolution for access discovery."""

from __future__ import annotations

import ast
from collections.abc import Mapping

from .access_model import FunctionInfo


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
    object_env: Mapping[str, set[str]],
    actor: str,
    functions: Mapping[str, FunctionInfo],
    nested_functions: Mapping[tuple[str, str], str],
    function_parents: Mapping[str, str | None],
    imported_symbols: Mapping[str, Mapping[str, tuple[str, str]]],
    imported_modules: Mapping[str, Mapping[str, str]],
) -> str:
    if isinstance(expression, ast.Name):
        scope: str | None = actor if actor in functions else None
        while scope is not None:
            nested = nested_functions.get((scope, expression.id))
            if nested is not None:
                return nested
            scope = function_parents.get(scope)
        local = f"{module}:{expression.id}"
        if local in functions:
            return local
        imported = imported_symbols.get(module, {}).get(expression.id)
        if imported is not None:
            return f"{imported[0]}:{imported[1]}"
        return ""
    if isinstance(expression, ast.Attribute) and isinstance(expression.value, ast.Name):
        if expression.value.id == "self" and class_ref is not None:
            return f"{class_ref}.{expression.attr}"
        imported_module = imported_modules.get(module, {}).get(expression.value.id)
        if imported_module is not None:
            return f"{imported_module}:{expression.attr}"
        object_types = object_env.get(expression.value.id, set())
        if len(object_types) == 1:
            return f"{next(iter(object_types))}.{expression.attr}"
    return ""


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


__all__ = ["call_name", "resolve_call_target", "resolve_class_target"]
