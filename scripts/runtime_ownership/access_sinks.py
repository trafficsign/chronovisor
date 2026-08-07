"""Call propagation and concrete I/O sink handling for access discovery."""

from __future__ import annotations

import ast
from collections.abc import Mapping, Sequence
from typing import Protocol

from .access_facts import AccessFactCollector
from .access_model import (
    OS_FD_OBJECT_TYPE,
    OS_FLAG_OBJECT_PREFIX,
    OS_OPEN_ACCESS_FLAGS,
    OS_OPEN_MODIFIER_FLAGS,
    PATH_TRANSFORMS,
    READ_PATH_METHODS,
    STDLIB_OS_CALLS,
    WRITE_PATH_METHODS,
    FlowValue,
    FunctionInfo,
    _open_mode,
    is_path_receiver,
)
from .access_resolver import (
    CallResolution,
    call_name,
    resolve_call_target,
    resolve_class_target,
)


class AccessEngine(Protocol):
    paths: dict[str, str]
    known_modules: frozenset[str]
    functions: dict[str, FunctionInfo]
    nested_functions: dict[tuple[str, str], str]
    function_parents: dict[str, str | None]
    imported_symbols: dict[str, dict[str, tuple[str, str]]]
    imported_modules: dict[str, dict[str, str]]
    classes: dict[str, dict[str, str]]
    returns: dict[str, FlowValue]
    facts: AccessFactCollector

    def _mark_called_target(self, target: str) -> None: ...

    def _require_function_summary(self, target: str) -> None: ...

    def _closure_capture_value(
        self,
        value: FlowValue,
        *,
        env: Mapping[str, FlowValue],
    ) -> FlowValue: ...

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
    ) -> FlowValue | None: ...

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
        site_node: ast.AST | None = None,
    ) -> None: ...


def _eval_expression(
    engine: AccessEngine,
    expression: ast.expr,
    *,
    module: str,
    actor: str,
    class_ref: str | None,
    env: dict[str, FlowValue],
    object_env: dict[str, set[str]],
    call_ordinals: Mapping[int, int],
) -> FlowValue:
    return engine._eval(
        expression,
        module=module,
        actor=actor,
        class_ref=class_ref,
        env=env,
        object_env=object_env,
        call_ordinals=call_ordinals,
    )


def _merge_values(values: Sequence[FlowValue]) -> FlowValue:
    merged = FlowValue()
    for value in values:
        merged = merged.merged(value)
    return merged


def _evaluate_path_transform(
    engine: AccessEngine,
    node: ast.Call,
    receiver: FlowValue,
    *,
    method: str,
    module: str,
    actor: str,
    class_ref: str | None,
    env: dict[str, FlowValue],
    object_env: dict[str, set[str]],
    call_ordinals: Mapping[int, int],
    ordinal: int,
) -> FlowValue:
    argument_origins = _merge_values(
        [
            _eval_expression(
                engine,
                expression,
                module=module,
                actor=actor,
                class_ref=class_ref,
                env=env,
                object_env=object_env,
                call_ordinals=call_ordinals,
            )
            for expression in [
                *node.args,
                *(keyword.value for keyword in node.keywords),
            ]
        ]
    )
    if argument_origins.has_origins:
        engine.facts.record_escape(
            argument_origins,
            node=node,
            actor=actor,
            operation=f"path.{method}",
            sink=f"pathlib.Path.{method}",
            reason="ambiguous_registered_origin_path_transform_argument",
            path=engine.paths[module],
            line=int(node.lineno),
            ordinal=ordinal,
        )
        return FlowValue()
    return receiver.bound(f"transform:{method}")


def _evaluate_path_move(
    engine: AccessEngine,
    node: ast.Call,
    receiver: FlowValue,
    *,
    method: str,
    module: str,
    actor: str,
    class_ref: str | None,
    env: dict[str, FlowValue],
    object_env: dict[str, set[str]],
    call_ordinals: Mapping[int, int],
    ordinal: int,
) -> FlowValue:
    positional = [
        _eval_expression(
            engine,
            expression,
            module=module,
            actor=actor,
            class_ref=class_ref,
            env=env,
            object_env=object_env,
            call_ordinals=call_ordinals,
        )
        for expression in node.args
    ]
    keywords = [
        (
            keyword.arg,
            _eval_expression(
                engine,
                keyword.value,
                module=module,
                actor=actor,
                class_ref=class_ref,
                env=env,
                object_env=object_env,
                call_ordinals=call_ordinals,
            ),
        )
        for keyword in node.keywords
    ]
    precise: list[FlowValue] = []
    ambiguous: list[FlowValue] = []
    for index, (expression, value) in enumerate(zip(node.args, positional, strict=True)):
        if index == 0 and not isinstance(expression, ast.Starred):
            precise.append(value)
        else:
            ambiguous.append(value)
    for name, value in keywords:
        if name == "target":
            precise.append(value)
        else:
            ambiguous.append(value)
    if len(precise) == 1:
        destination = precise[0]
    else:
        ambiguous.extend(precise)
        destination = FlowValue()
    implicated = _merge_values(ambiguous)
    engine.facts.record_access(
        receiver,
        node=node,
        actor=actor,
        mode="write",
        operation=f"path.{method}",
        sink=f"pathlib.Path.{method}",
        path=engine.paths[module],
        line=int(node.lineno),
        ordinal=ordinal,
    )
    if destination.has_origins:
        engine.facts.record_access(
            destination,
            node=node,
            actor=actor,
            mode="write",
            operation=f"path.{method}.destination",
            sink=f"pathlib.Path.{method}",
            path=engine.paths[module],
            line=int(node.lineno),
            ordinal=ordinal,
        )
    if implicated.has_origins:
        engine.facts.record_escape(
            implicated,
            node=node,
            actor=actor,
            operation=f"path.{method}.destination",
            sink=f"pathlib.Path.{method}",
            reason="ambiguous_registered_origin_path_destination",
            path=engine.paths[module],
            line=int(node.lineno),
            ordinal=ordinal,
        )
        return FlowValue()
    return destination.bound(f"result:path.{method}.destination")


def _is_precise_os_module(value: FlowValue) -> bool:
    return (
        value.module_refs == {"os"}
        and not value.origins
        and not value.call_targets
        and not value.class_targets
        and not value.unknown_callable
        and not value.closure_instances
    )


def _stdlib_os_call_name(
    node: ast.Call,
    receiver: FlowValue,
    env: Mapping[str, FlowValue],
    *,
    known_modules: frozenset[str],
) -> str | None:
    if "os" in known_modules:
        return None
    if isinstance(node.func, ast.Attribute):
        if node.func.attr in STDLIB_OS_CALLS and _is_precise_os_module(receiver):
            return node.func.attr
        return None
    if not isinstance(node.func, ast.Name) or node.func.id not in env:
        return None
    value = env[node.func.id]
    for name in STDLIB_OS_CALLS:
        if (
            value.call_targets == {f"os:{name}"}
            and not value.origins
            and not value.module_refs
            and not value.class_targets
            and not value.unknown_callable
            and not value.closure_instances
        ):
            return name
    return None


def _precise_argument_value(
    node: ast.Call,
    argument_values: Sequence[FlowValue],
    keyword_values: Mapping[str, FlowValue],
    *,
    index: int,
    keyword: str,
) -> tuple[FlowValue, FlowValue]:
    candidates: list[FlowValue] = []
    ambiguous: list[FlowValue] = []
    if len(node.args) > index:
        value = argument_values[index]
        if isinstance(node.args[index], ast.Starred):
            ambiguous.append(value)
        else:
            candidates.append(value)
    if keyword in keyword_values:
        candidates.append(keyword_values[keyword])
    if len(candidates) == 1:
        return candidates[0], _merge_values(ambiguous)
    ambiguous.extend(candidates)
    return FlowValue(), _merge_values(ambiguous)


def _precise_argument_expression(
    node: ast.Call,
    *,
    index: int,
    keyword: str,
) -> ast.expr | None:
    candidates: list[ast.expr] = []
    if len(node.args) > index and not isinstance(node.args[index], ast.Starred):
        candidates.append(node.args[index])
    candidates.extend(
        item.value for item in node.keywords if item.arg == keyword
    )
    return candidates[0] if len(candidates) == 1 else None


def _is_precise_os_flag(value: FlowValue, flag: str) -> bool:
    return (
        value.object_types == {f"{OS_FLAG_OBJECT_PREFIX}{flag}"}
        and not value.origins
        and not value.module_refs
        and not value.call_targets
        and not value.class_targets
        and not value.unknown_callable
        and not value.closure_instances
    )


def _os_flag_names(
    expression: ast.expr,
    env: Mapping[str, FlowValue],
) -> frozenset[str] | None:
    if isinstance(expression, ast.BinOp) and isinstance(expression.op, ast.BitOr):
        left = _os_flag_names(expression.left, env)
        right = _os_flag_names(expression.right, env)
        return None if left is None or right is None else left | right
    if isinstance(expression, ast.Attribute) and isinstance(
        expression.value, ast.Name
    ):
        base = env.get(expression.value.id, FlowValue())
        if _is_precise_os_module(base):
            flag = expression.attr
            if flag in OS_OPEN_ACCESS_FLAGS | OS_OPEN_MODIFIER_FLAGS:
                return frozenset({flag})
        return None
    if isinstance(expression, ast.Name):
        value = env.get(expression.id, FlowValue())
        for flag in OS_OPEN_ACCESS_FLAGS | OS_OPEN_MODIFIER_FLAGS:
            if _is_precise_os_flag(value, flag):
                return frozenset({flag})
    return None


def _os_open_mode(
    expression: ast.expr | None,
    env: Mapping[str, FlowValue],
) -> tuple[str, str] | None:
    if isinstance(expression, ast.Constant) and isinstance(expression.value, int):
        access_bits = expression.value & 3
        mode_by_bits = {0: "read", 1: "write", 2: "read_write"}
        mode = mode_by_bits.get(access_bits)
        return None if mode is None else (mode, f"flags={expression.value}")
    if expression is None:
        return None
    names = _os_flag_names(expression, env)
    if names is None:
        return None
    access_flags = names & OS_OPEN_ACCESS_FLAGS
    if len(access_flags) != 1:
        return None
    access_flag = next(iter(access_flags))
    modes = {
        "O_RDONLY": "read",
        "O_WRONLY": "write",
        "O_RDWR": "read_write",
    }
    label = "|".join([access_flag, *sorted(names - {access_flag})])
    return modes[access_flag], label


def _unexpected_argument_origins(
    node: ast.Call,
    argument_values: Sequence[FlowValue],
    keyword_values: Mapping[str, FlowValue],
    *,
    positional_count: int,
    keyword_names: frozenset[str],
) -> FlowValue:
    values = [
        value
        for index, value in enumerate(argument_values)
        if index >= positional_count or isinstance(node.args[index], ast.Starred)
    ]
    values.extend(
        value
        for name, value in keyword_values.items()
        if name not in keyword_names
    )
    return _merge_values(values)


def _record_os_argument_escape(
    engine: AccessEngine,
    value: FlowValue,
    node: ast.Call,
    *,
    name: str,
    module: str,
    actor: str,
    ordinal: int,
) -> None:
    if not value.has_origins:
        return
    engine.facts.record_escape(
        value,
        node=node,
        actor=actor,
        operation=f"os.{name}",
        sink=f"os.{name}",
        reason=f"ambiguous_registered_origin_os_{name}_arguments",
        path=engine.paths[module],
        line=int(node.lineno),
        ordinal=ordinal,
    )


def _evaluate_os_open(
    engine: AccessEngine,
    node: ast.Call,
    argument_values: Sequence[FlowValue],
    keyword_values: Mapping[str, FlowValue],
    unknown_keyword_unpack: FlowValue,
    *,
    module: str,
    actor: str,
    env: Mapping[str, FlowValue],
    ordinal: int,
) -> FlowValue:
    path_value, path_ambiguous = _precise_argument_value(
        node,
        argument_values,
        keyword_values,
        index=0,
        keyword="path",
    )
    flags_value, flags_ambiguous = _precise_argument_value(
        node,
        argument_values,
        keyword_values,
        index=1,
        keyword="flags",
    )
    implicated = path_ambiguous.merged(flags_ambiguous)
    if flags_value.has_origins:
        implicated = implicated.merged(flags_value)
    implicated = implicated.merged(unknown_keyword_unpack).merged(
        _unexpected_argument_origins(
            node,
            argument_values,
            keyword_values,
            positional_count=2,
            keyword_names=frozenset({"path", "flags"}),
        )
    )
    if implicated.has_origins:
        _record_os_argument_escape(
            engine,
            implicated.merged(path_value),
            node,
            name="open",
            module=module,
            actor=actor,
            ordinal=ordinal,
        )
        return FlowValue()
    if not path_value.has_origins:
        return FlowValue()
    mode = _os_open_mode(
        _precise_argument_expression(node, index=1, keyword="flags"), env
    )
    if mode is None:
        engine.facts.record_escape(
            path_value,
            node=node,
            actor=actor,
            operation="os.open",
            sink="os.open",
            reason="dynamic_os_open_flags",
            path=engine.paths[module],
            line=int(node.lineno),
            ordinal=ordinal,
        )
    else:
        engine.facts.record_access(
            path_value,
            node=node,
            actor=actor,
            mode=mode[0],
            operation=f"os.open:{mode[1]}",
            sink="os.open",
            path=engine.paths[module],
            line=int(node.lineno),
            ordinal=ordinal,
        )
    descriptor = path_value.bound("result:os.open:fd")
    descriptor.object_types.add(OS_FD_OBJECT_TYPE)
    return descriptor


def _evaluate_os_move(
    engine: AccessEngine,
    node: ast.Call,
    argument_values: Sequence[FlowValue],
    keyword_values: Mapping[str, FlowValue],
    unknown_keyword_unpack: FlowValue,
    *,
    name: str,
    module: str,
    actor: str,
    ordinal: int,
) -> FlowValue:
    source, source_ambiguous = _precise_argument_value(
        node,
        argument_values,
        keyword_values,
        index=0,
        keyword="src",
    )
    destination, destination_ambiguous = _precise_argument_value(
        node,
        argument_values,
        keyword_values,
        index=1,
        keyword="dst",
    )
    implicated = source_ambiguous.merged(destination_ambiguous)
    implicated = implicated.merged(unknown_keyword_unpack).merged(
        _unexpected_argument_origins(
            node,
            argument_values,
            keyword_values,
            positional_count=2,
            keyword_names=frozenset({"src", "dst"}),
        )
    )
    if source.has_origins:
        engine.facts.record_access(
            source,
            node=node,
            actor=actor,
            mode="read_write",
            operation=f"os.{name}.source",
            sink=f"os.{name}",
            path=engine.paths[module],
            line=int(node.lineno),
            ordinal=ordinal,
        )
    if destination.has_origins:
        engine.facts.record_access(
            destination,
            node=node,
            actor=actor,
            mode="write",
            operation=f"os.{name}.destination",
            sink=f"os.{name}",
            path=engine.paths[module],
            line=int(node.lineno),
            ordinal=ordinal,
        )
    _record_os_argument_escape(
        engine,
        implicated,
        node,
        name=name,
        module=module,
        actor=actor,
        ordinal=ordinal,
    )
    return FlowValue()


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
        path_receiver = is_path_receiver(receiver)
        if path_receiver and method in {"rename", "replace"}:
            return _evaluate_path_move(
                engine,
                node,
                receiver,
                method=method,
                module=module,
                actor=actor,
                class_ref=class_ref,
                env=env,
                object_env=object_env,
                call_ordinals=call_ordinals,
                ordinal=ordinal,
            )
        if path_receiver and method in READ_PATH_METHODS:
            engine.facts.record_access(
                receiver,
                node=node,
                actor=actor,
                mode="read",
                operation=f"path.{method}",
                sink=f"pathlib.Path.{method}",
                path=engine.paths[module],
                line=int(node.lineno),
                ordinal=ordinal,
            )
            return FlowValue()
        if path_receiver and method in WRITE_PATH_METHODS:
            engine.facts.record_access(
                receiver,
                node=node,
                actor=actor,
                mode="write",
                operation=f"path.{method}",
                sink=f"pathlib.Path.{method}",
                path=engine.paths[module],
                line=int(node.lineno),
                ordinal=ordinal,
            )
            return FlowValue()
        if path_receiver and method == "open":
            mode = _open_mode(node, mode_index=0)
            if isinstance(mode, str):
                engine.facts.record_escape(
                    receiver,
                    node=node,
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
                    node=node,
                    actor=actor,
                    mode=mode[0],
                    operation=f"path.open:{mode[1]}",
                    sink="pathlib.Path.open",
                    path=engine.paths[module],
                    line=int(node.lineno),
                    ordinal=ordinal,
                )
            return FlowValue()
        if path_receiver and method in PATH_TRANSFORMS:
            return _evaluate_path_transform(
                engine,
                node,
                receiver,
                method=method,
                module=module,
                actor=actor,
                class_ref=class_ref,
                env=env,
                object_env=object_env,
                call_ordinals=call_ordinals,
                ordinal=ordinal,
            )
    resolution = resolve_call_target(
        node.func,
        module=module,
        class_ref=class_ref,
        env=env,
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
        and not resolution.known_targets
        and (
            "open" in env
            or (actor_info is not None and "open" in actor_info.parameters)
        )
    ):
        resolution = CallResolution(has_unknown=True)
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
    keyword_values: dict[str, FlowValue] = {}
    unknown_keyword_unpack = FlowValue()
    for keyword in node.keywords:
        if keyword.arg is not None:
            keyword_values[keyword.arg] = engine._eval(
                keyword.value,
                module=module,
                actor=actor,
                class_ref=class_ref,
                env=env,
                object_env=object_env,
                call_ordinals=call_ordinals,
            )
            continue
        if isinstance(keyword.value, ast.Dict):
            literal_keys = all(
                isinstance(key, ast.Constant) and isinstance(key.value, str)
                for key in keyword.value.keys
            )
            unpacked = FlowValue()
            for key, value_node in zip(
                keyword.value.keys,
                keyword.value.values,
                strict=True,
            ):
                key_value = FlowValue()
                if key is not None:
                    key_value = engine._eval(
                        key,
                        module=module,
                        actor=actor,
                        class_ref=class_ref,
                        env=env,
                        object_env=object_env,
                        call_ordinals=call_ordinals,
                    )
                value = engine._eval(
                    value_node,
                    module=module,
                    actor=actor,
                    class_ref=class_ref,
                    env=env,
                    object_env=object_env,
                    call_ordinals=call_ordinals,
                )
                if literal_keys:
                    assert isinstance(key, ast.Constant)
                    assert isinstance(key.value, str)
                    keyword_values[key.value] = value
                else:
                    unpacked = unpacked.merged(key_value).merged(value)
            if not literal_keys:
                unknown_keyword_unpack = unknown_keyword_unpack.merged(unpacked)
            continue
        unknown_keyword_unpack = unknown_keyword_unpack.merged(
            engine._eval(
                keyword.value,
                module=module,
                actor=actor,
                class_ref=class_ref,
                env=env,
                object_env=object_env,
                call_ordinals=call_ordinals,
            )
        )
    source_call_name = call_name(node.func)
    os_call = _stdlib_os_call_name(
        node,
        receiver,
        env,
        known_modules=engine.known_modules,
    )
    if os_call == "open":
        return _evaluate_os_open(
            engine,
            node,
            argument_values,
            keyword_values,
            unknown_keyword_unpack,
            module=module,
            actor=actor,
            env=env,
            ordinal=ordinal,
        )
    if os_call in {"rename", "replace"}:
        return _evaluate_os_move(
            engine,
            node,
            argument_values,
            keyword_values,
            unknown_keyword_unpack,
            name=os_call,
            module=module,
            actor=actor,
            ordinal=ordinal,
        )
    escaped = receiver.copy()
    for value in [
        *argument_values,
        *keyword_values.values(),
        unknown_keyword_unpack,
    ]:
        escaped = escaped.merged(value)
    escaped_closure_capture = engine._closure_capture_value(escaped, env=env)
    known_targets = resolution.known_targets
    authoritative_value: FlowValue | None = None
    authoritative_base_value: FlowValue | None = None
    if isinstance(node.func, ast.Name) and node.func.id in env:
        authoritative_value = env[node.func.id]
    elif (
        isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id in env
    ):
        authoritative_base_value = env[node.func.value.id]
    known_class_targets = (
        tuple(sorted(authoritative_value.class_targets))
        if authoritative_value is not None
        else ()
    )
    has_unknown = resolution.has_unknown
    if authoritative_value is not None:
        unresolved_call_targets = authoritative_value.call_targets.difference(
            known_targets
        )
        has_unknown = (
            authoritative_value.unknown_callable
            or bool(unresolved_call_targets)
            or not (
                authoritative_value.call_targets or authoritative_value.class_targets
            )
        )
    elif authoritative_base_value is not None:
        callee_value = engine._eval(
            node.func,
            module=module,
            actor=actor,
            class_ref=class_ref,
            env=env,
            object_env=object_env,
            call_ordinals=call_ordinals,
        )
        known_targets = tuple(
            sorted(
                set(known_targets)
                | {
                    target
                    for target in callee_value.call_targets
                    if target in engine.functions
                }
            )
        )
        known_class_targets = tuple(sorted(callee_value.class_targets))
        has_unknown = (
            authoritative_base_value.unknown_callable
            or callee_value.unknown_callable
            or bool(callee_value.call_targets.difference(known_targets))
            or not (known_targets or known_class_targets)
        )
    has_authoritative_binding = (
        authoritative_value is not None or authoritative_base_value is not None
    )
    if not known_targets and not known_class_targets and not has_authoritative_binding:
        class_target = resolve_class_target(
            node.func,
            module=module,
            classes=engine.classes,
            imported_symbols=engine.imported_symbols,
            imported_modules=engine.imported_modules,
        )
        if class_target is not None:
            known_class_targets = (class_target,)
            has_unknown = False
    for target in known_targets:
        engine._mark_called_target(target)
    local_returns: dict[str, FlowValue] = {}
    if len(known_targets) == 1 and not has_unknown:
        target = known_targets[0]
        closure_instances = tuple(
            sorted(
                instance_id
                for instance_target, instance_id in (
                    authoritative_value.closure_instances
                    if authoritative_value is not None
                    else set()
                )
                if instance_target == target
            )
        ) or (None,)
        merged_local_return = FlowValue()
        all_local = True
        for closure_instance in closure_instances:
            local_return = engine._execute_known_call(
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
            )
            if local_return is None:
                all_local = False
                break
            merged_local_return = merged_local_return.merged(local_return)
        if all_local:
            local_returns[target] = merged_local_return
    for target in known_targets:
        if target not in local_returns:
            engine._require_function_summary(target)
    if known_targets or known_class_targets:
        returned = FlowValue()
        for target in known_targets:
            info = engine.functions[target]
            local_return = local_returns.get(target)
            if local_return is None:
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
                local_return = engine.returns[target]
            returned = returned.merged(local_return.bound(f"result:{target}"))
        for class_target in known_class_targets:
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
            returned = returned.merged(FlowValue(object_types={class_target}))
        if has_unknown or unknown_keyword_unpack.has_origins:
            if escaped_closure_capture.has_origins:
                engine.facts.record_escape(
                    escaped_closure_capture,
                    node=node,
                    actor=actor,
                    operation=f"call:{source_call_name or '<dynamic>'}",
                    sink=source_call_name or "<dynamic>",
                    reason="closure_to_unknown_callee",
                    path=engine.paths[module],
                    line=int(node.lineno),
                    ordinal=ordinal,
                )
            if escaped.has_origins:
                engine.facts.record_escape(
                    escaped,
                    node=node,
                    actor=actor,
                    operation=f"call:{source_call_name or '<dynamic>'}",
                    sink=source_call_name or "<dynamic>",
                    reason="registered_locator_to_unknown_callee",
                    path=engine.paths[module],
                    line=int(node.lineno),
                    ordinal=ordinal,
                )
            if has_unknown:
                returned = returned.merged(FlowValue(unknown_callable=True))
        return returned
    target = resolution.fallback_target
    if _is_builtin_open(
        engine,
        node,
        target=target,
        actor=actor,
        module=module,
        env=env,
    ):
        if unknown_keyword_unpack.has_origins:
            engine.facts.record_escape(
                unknown_keyword_unpack,
                node=node,
                actor=actor,
                operation=f"call:{source_call_name or '<dynamic>'}",
                sink=target or source_call_name or "<dynamic>",
                reason="registered_locator_to_unknown_callee",
                path=engine.paths[module],
                line=int(node.lineno),
                ordinal=ordinal,
            )
        origin = (
            argument_values[0]
            if argument_values
            else keyword_values.get("file", FlowValue())
        )
        mode = _open_mode(node, mode_index=1)
        if origin.has_origins and isinstance(mode, str):
            engine.facts.record_escape(
                origin,
                node=node,
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
                node=node,
                actor=actor,
                mode=mode[0],
                operation=f"builtin.open:{mode[1]}",
                sink="builtins.open",
                path=engine.paths[module],
                line=int(node.lineno),
                ordinal=ordinal,
            )
        return FlowValue()
    if escaped.has_origins:
        engine.facts.record_escape(
            escaped,
            node=node,
            actor=actor,
            operation=f"call:{source_call_name or '<dynamic>'}",
            sink=target or source_call_name or "<dynamic>",
            reason="registered_locator_to_unknown_callee",
            path=engine.paths[module],
            line=int(node.lineno),
            ordinal=ordinal,
        )
    if escaped_closure_capture.has_origins:
        engine.facts.record_escape(
            escaped_closure_capture,
            node=node,
            actor=actor,
            operation=f"call:{source_call_name or '<dynamic>'}",
            sink=target or source_call_name or "<dynamic>",
            reason="closure_to_unknown_callee",
            path=engine.paths[module],
            line=int(node.lineno),
            ordinal=ordinal,
        )
    return FlowValue(unknown_callable=True)


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
