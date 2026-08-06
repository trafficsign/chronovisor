"""Call propagation and concrete I/O sink handling for access discovery."""

from __future__ import annotations

import ast
from collections.abc import Mapping
from typing import Protocol

from .access_facts import AccessFactCollector
from .access_model import (
    PATH_TRANSFORMS,
    READ_PATH_METHODS,
    WRITE_PATH_METHODS,
    FlowValue,
    FunctionInfo,
    _open_mode,
)
from .access_resolver import call_name, resolve_call_target, resolve_class_target


class AccessEngine(Protocol):
    paths: dict[str, str]
    functions: dict[str, FunctionInfo]
    nested_functions: dict[tuple[str, str], str]
    function_parents: dict[str, str | None]
    imported_symbols: dict[str, dict[str, tuple[str, str]]]
    imported_modules: dict[str, dict[str, str]]
    classes: dict[str, dict[str, str]]
    returns: dict[str, FlowValue]
    facts: AccessFactCollector

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
    ) -> FlowValue: ...

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
    ) -> None: ...


def evaluate_call(
    engine: AccessEngine,
    node: ast.Call,
    *,
    module: str,
    actor: str,
    class_ref: str | None,
    env: dict[str, FlowValue],
    object_env: dict[str, set[str]],
    call_ordinals: Mapping[int, int],
) -> FlowValue:
    ordinal = int(call_ordinals.get(id(node), 0))
    receiver = FlowValue()
    if isinstance(node.func, ast.Attribute):
        receiver = engine._eval(
            node.func.value,
            module=module,
            actor=actor,
            class_ref=class_ref,
            env=env,
            object_env=object_env,
            call_ordinals=call_ordinals,
        )
        method = node.func.attr
        if receiver.has_origins and method in READ_PATH_METHODS:
            engine.facts.record_access(
                receiver,
                actor=actor,
                mode="read",
                operation=f"path.{method}",
                sink=f"pathlib.Path.{method}",
                path=engine.paths[module],
                line=int(node.lineno),
                ordinal=ordinal,
            )
            return FlowValue()
        if receiver.has_origins and method in WRITE_PATH_METHODS:
            engine.facts.record_access(
                receiver,
                actor=actor,
                mode="write",
                operation=f"path.{method}",
                sink=f"pathlib.Path.{method}",
                path=engine.paths[module],
                line=int(node.lineno),
                ordinal=ordinal,
            )
            return FlowValue()
        if receiver.has_origins and method == "open":
            mode = _open_mode(node, mode_index=0)
            if isinstance(mode, str):
                engine.facts.record_escape(
                    receiver,
                    actor=actor,
                    operation="path.open",
                    sink="pathlib.Path.open",
                    reason=mode,
                    path=engine.paths[module],
                    line=int(node.lineno),
                    ordinal=ordinal,
                )
            else:
                engine.facts.record_access(
                    receiver,
                    actor=actor,
                    mode=mode[0],
                    operation=f"path.open:{mode[1]}",
                    sink="pathlib.Path.open",
                    path=engine.paths[module],
                    line=int(node.lineno),
                    ordinal=ordinal,
                )
            return FlowValue()
        if receiver.has_origins and method in PATH_TRANSFORMS:
            return receiver.bound(f"transform:{method}")
    target = resolve_call_target(
        node.func,
        module=module,
        class_ref=class_ref,
        object_env=object_env,
        actor=actor,
        functions=engine.functions,
        nested_functions=engine.nested_functions,
        function_parents=engine.function_parents,
        imported_symbols=engine.imported_symbols,
        imported_modules=engine.imported_modules,
    )
    actor_info = engine.functions.get(actor)
    if (
        isinstance(node.func, ast.Name)
        and node.func.id == "open"
        and (
            "open" in env
            or (actor_info is not None and "open" in actor_info.parameters)
        )
    ):
        target = ""
    argument_values = [
        engine._eval(
            argument,
            module=module,
            actor=actor,
            class_ref=class_ref,
            env=env,
            object_env=object_env,
            call_ordinals=call_ordinals,
        )
        for argument in node.args
    ]
    keyword_values = {
        str(keyword.arg): engine._eval(
            keyword.value,
            module=module,
            actor=actor,
            class_ref=class_ref,
            env=env,
            object_env=object_env,
            call_ordinals=call_ordinals,
        )
        for keyword in node.keywords
        if keyword.arg is not None
    }
    source_call_name = call_name(node.func)
    if target in engine.functions:
        info = engine.functions[target]
        values = list(argument_values)
        if info.class_ref is not None and not target.endswith(".__init__"):
            values = [FlowValue(), *values]
        for index, value in enumerate(values):
            if index >= len(info.parameters) or not value.has_origins:
                continue
            engine._bind_call_parameter(
                target,
                info.parameters[index],
                value,
                actor=actor,
                module=module,
                node=node,
                ordinal=ordinal,
            )
        for parameter, value in keyword_values.items():
            if parameter in info.parameters and value.has_origins:
                engine._bind_call_parameter(
                    target,
                    parameter,
                    value,
                    actor=actor,
                    module=module,
                    node=node,
                    ordinal=ordinal,
                )
        return engine.returns[target].bound(f"result:{target}")
    class_target = resolve_class_target(
        node.func,
        module=module,
        classes=engine.classes,
        imported_symbols=engine.imported_symbols,
        imported_modules=engine.imported_modules,
    )
    if class_target is not None:
        init_ref = f"{class_target}.__init__"
        if init_ref in engine.functions:
            info = engine.functions[init_ref]
            for index, value in enumerate(argument_values, start=1):
                if index < len(info.parameters) and value.has_origins:
                    engine._bind_call_parameter(
                        init_ref,
                        info.parameters[index],
                        value,
                        actor=actor,
                        module=module,
                        node=node,
                        ordinal=ordinal,
                    )
            for parameter, value in keyword_values.items():
                if parameter in info.parameters and value.has_origins:
                    engine._bind_call_parameter(
                        init_ref,
                        parameter,
                        value,
                        actor=actor,
                        module=module,
                        node=node,
                        ordinal=ordinal,
                    )
        return FlowValue(object_types={class_target})
    if _is_builtin_open(
        engine,
        node,
        target=target,
        actor=actor,
        module=module,
        env=env,
    ):
        origin = (
            argument_values[0]
            if argument_values
            else keyword_values.get("file", FlowValue())
        )
        mode = _open_mode(node, mode_index=1)
        if origin.has_origins and isinstance(mode, str):
            engine.facts.record_escape(
                origin,
                actor=actor,
                operation="builtin.open",
                sink="builtins.open",
                reason=mode,
                path=engine.paths[module],
                line=int(node.lineno),
                ordinal=ordinal,
            )
        elif origin.has_origins:
            engine.facts.record_access(
                origin,
                actor=actor,
                mode=mode[0],
                operation=f"builtin.open:{mode[1]}",
                sink="builtins.open",
                path=engine.paths[module],
                line=int(node.lineno),
                ordinal=ordinal,
            )
        return FlowValue()
    escaped = receiver.copy()
    for value in [*argument_values, *keyword_values.values()]:
        escaped = escaped.merged(value)
    if escaped.has_origins:
        engine.facts.record_escape(
            escaped,
            actor=actor,
            operation=f"call:{source_call_name or '<dynamic>'}",
            sink=target or source_call_name or "<dynamic>",
            reason="registered_locator_to_unknown_callee",
            path=engine.paths[module],
            line=int(node.lineno),
            ordinal=ordinal,
        )
    return FlowValue()


def _is_builtin_open(
    engine: AccessEngine,
    node: ast.Call,
    *,
    target: str,
    actor: str,
    module: str,
    env: dict[str, FlowValue],
) -> bool:
    if target == "builtins:open":
        return True
    if not isinstance(node.func, ast.Name) or node.func.id != "open" or target:
        return False
    if "open" in env or "open" in engine.imported_symbols.get(module, {}):
        return False
    info = engine.functions.get(actor)
    return info is None or "open" not in info.parameters


__all__ = ["AccessEngine", "evaluate_call"]
