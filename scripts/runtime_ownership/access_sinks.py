"""Call propagation and concrete I/O sink handling for access discovery."""

from __future__ import annotations

import ast
import re
from collections.abc import Mapping, Sequence
from typing import Literal, Protocol
from urllib.parse import parse_qsl, urlsplit

from .access_facts import AccessFactCollector
from .access_model import (
    FCNTL_UNRESOLVED_LOCK_OPERATION_OBJECT_TYPE,
    OS_FLAG_OBJECT_PREFIX,
    OS_OPEN_ACCESS_FLAGS,
    OS_OPEN_MODIFIER_FLAGS,
    PATH_TRANSFORMS,
    READ_PATH_METHODS,
    SQLITE_TYPE_OBJECT_PREFIX,
    STDLIB_BUILTINS_CALLS,
    STDLIB_FCNTL_CALLS,
    STDLIB_MODULE_WILDCARD_ATTRIBUTE,
    STDLIB_OS_CALLS,
    STDLIB_SQLITE3_CALLS,
    SUPPORTED_STDLIB_MODULES,
    WRITE_PATH_METHODS,
    FlowValue,
    FunctionInfo,
    fcntl_lock_masks,
    file_handle_kind,
    is_exact_flock_descriptor,
    is_exact_os_fd,
    is_exact_path_receiver,
    is_path_receiver,
    is_precise_stdlib_module,
    open_mode_from_expression,
    precise_stdlib_module_name,
    sqlite_handle_kind,
    stdlib_call_targets,
    stdlib_module_dict_reference,
    stdlib_module_mutation_marker,
    tag_file_handle,
    tag_os_fd,
    tag_sqlite_handle,
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
    resource_locators: dict[str, str]
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

    def _taint_stdlib_module_attribute(
        self,
        base: FlowValue,
        *,
        attribute: str,
        env: dict[str, FlowValue],
        object_env: dict[str, set[str]],
    ) -> bool: ...

    def _contaminate_runtime_objects(
        self,
        env: dict[str, FlowValue],
        object_env: dict[str, set[str]],
        values: Sequence[FlowValue],
    ) -> None: ...

    def _runtime_object_identity(
        self,
        node: ast.Call,
        *,
        kind: str,
        actor: str,
    ) -> str: ...


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


def _is_unshadowed_builtin(
    engine: AccessEngine,
    node: ast.Call,
    *,
    name: str,
    actor: str,
    module: str,
    env: Mapping[str, FlowValue],
) -> bool:
    if not isinstance(node.func, ast.Name) or node.func.id != name:
        return False
    if name in env or name in engine.imported_symbols.get(module, {}):
        return False
    info = engine.functions.get(actor)
    return info is None or (
        name not in info.parameters and name not in info.local_names
    )


def _evaluate_builtin_stdlib_module_mutation(
    engine: AccessEngine,
    node: ast.Call,
    argument_values: Sequence[FlowValue],
    *,
    module: str,
    actor: str,
    env: dict[str, FlowValue],
    object_env: dict[str, set[str]],
    ordinal: int,
) -> FlowValue | None:
    name = node.func.id if isinstance(node.func, ast.Name) else ""
    arity = {"setattr": 3, "delattr": 2}.get(name)
    if arity is None or not _is_unshadowed_builtin(
        engine,
        node,
        name=name,
        actor=actor,
        module=module,
        env=env,
    ):
        return None
    if (
        len(node.args) != arity
        or any(isinstance(argument, ast.Starred) for argument in node.args)
        or node.keywords
    ):
        return FlowValue()
    attribute_node = node.args[1]
    if isinstance(attribute_node, ast.Constant):
        if not isinstance(attribute_node.value, str):
            return FlowValue()
        attribute = attribute_node.value
    else:
        attribute = STDLIB_MODULE_WILDCARD_ATTRIBUTE
    if not engine._taint_stdlib_module_attribute(
        argument_values[0],
        attribute=attribute,
        env=env,
        object_env=object_env,
    ):
        return None
    affected = _merge_values(argument_values[1:])
    if affected.has_origins:
        engine.facts.record_escape(
            affected,
            node=node,
            actor=actor,
            operation=f"builtin.{name}",
            sink=f"builtins.{name}",
            reason="registered_locator_to_stdlib_module_mutation",
            path=engine.paths[module],
            line=int(node.lineno),
            ordinal=ordinal,
        )
    return FlowValue()


def _taint_callable_stdlib_modules(
    engine: AccessEngine,
    values: Sequence[FlowValue],
    *,
    env: dict[str, FlowValue],
    object_env: dict[str, set[str]],
) -> None:
    pending = list(values)
    modules: set[str] = set()
    while pending:
        value = pending.pop()
        modules.update(
            module_ref
            for module_ref in value.module_refs
            if module_ref in SUPPORTED_STDLIB_MODULES
            and module_ref not in engine.known_modules
        )
        if value.structured_items is not None:
            pending.extend(value.structured_items)
    for module_ref in sorted(modules):
        base = next(
            (
                value
                for value in values
                if precise_stdlib_module_name(value) == module_ref
            ),
            FlowValue(module_refs={module_ref}),
        )
        engine._taint_stdlib_module_attribute(
            base,
            attribute=STDLIB_MODULE_WILDCARD_ATTRIBUTE,
            env=env,
            object_env=object_env,
        )


def _stdlib_reference_is_read_only(info: FunctionInfo, name: str) -> bool:
    parents = {
        id(child): parent
        for parent in ast.walk(info.node)
        for child in ast.iter_child_nodes(parent)
    }
    found = False
    for child in ast.walk(info.node):
        if not isinstance(child, ast.Name) or child.id != name:
            continue
        found = True
        parent = parents.get(id(child))
        if not (
            isinstance(parent, ast.Attribute)
            and parent.value is child
            and isinstance(parent.ctx, ast.Load)
        ):
            return False
    return found


def _canonical_stdlib_call_sink(
    node: ast.Call,
    receiver: FlowValue,
    env: Mapping[str, FlowValue],
    *,
    known_modules: frozenset[str],
) -> str | None:
    candidates: set[str] = set()
    if isinstance(node.func, ast.Attribute):
        module_ref = precise_stdlib_module_name(receiver)
        if (
            module_ref is not None
            and module_ref not in known_modules
            and _is_supported_stdlib_call(module_ref, node.func.attr)
        ):
            candidates.add(f"{module_ref}.{node.func.attr}")
    elif isinstance(node.func, ast.Name) and node.func.id in env:
        value = env[node.func.id]
        candidates.update(stdlib_call_targets(value))
        for target in value.call_targets:
            module_ref, separator, attribute = target.partition(":")
            if (
                separator
                and module_ref not in known_modules
                and _is_supported_stdlib_call(module_ref, attribute)
            ):
                candidates.add(f"{module_ref}.{attribute}")
    supported = {
        candidate
        for candidate in candidates
        if candidate.split(".", 1)[0] not in known_modules
    }
    return next(iter(supported)) if len(supported) == 1 else None


def _is_supported_stdlib_call(module_ref: str, attribute: str) -> bool:
    if module_ref == "builtins":
        return attribute in STDLIB_BUILTINS_CALLS
    if module_ref == "fcntl":
        return attribute in STDLIB_FCNTL_CALLS
    if module_ref == "os":
        return attribute in STDLIB_OS_CALLS
    if module_ref == "sqlite3":
        return attribute in STDLIB_SQLITE3_CALLS
    return False


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


def _is_precise_os_module(value: FlowValue, *, attribute: str) -> bool:
    return is_precise_stdlib_module(
        value,
        module="os",
        attribute=attribute,
    )


def _stdlib_builtins_call_name(
    node: ast.Call,
    receiver: FlowValue,
    env: Mapping[str, FlowValue],
    *,
    known_modules: frozenset[str],
) -> str | None:
    if "builtins" in known_modules:
        return None
    if isinstance(node.func, ast.Attribute):
        if node.func.attr in STDLIB_BUILTINS_CALLS and is_precise_stdlib_module(
            receiver,
            module="builtins",
            attribute=node.func.attr,
        ):
            return node.func.attr
        return None
    if not isinstance(node.func, ast.Name) or node.func.id not in env:
        return None
    value = env[node.func.id]
    for name in STDLIB_BUILTINS_CALLS:
        if (
            value.call_targets == {f"builtins:{name}"}
            and not value.origins
            and not value.module_refs
            and not value.class_targets
            and not value.unknown_callable
            and not value.closure_instances
        ):
            return name
    return None


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
        if node.func.attr in STDLIB_OS_CALLS and _is_precise_os_module(
            receiver, attribute=node.func.attr
        ):
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
    candidates.extend(_keyword_argument_expressions(node, keyword=keyword))
    return candidates[0] if len(candidates) == 1 else None


def _keyword_argument_expressions(
    node: ast.Call,
    *,
    keyword: str,
) -> list[ast.expr]:
    expressions = [item.value for item in node.keywords if item.arg == keyword]
    for item in node.keywords:
        if item.arg is not None or not isinstance(item.value, ast.Dict):
            continue
        for key, value in zip(
            item.value.keys,
            item.value.values,
            strict=True,
        ):
            if (
                isinstance(key, ast.Constant)
                and isinstance(key.value, str)
                and key.value == keyword
            ):
                expressions.append(value)
    return expressions


def _expanded_keyword_arguments(
    node: ast.Call,
) -> list[tuple[str, ast.expr]] | None:
    expanded: list[tuple[str, ast.expr]] = []
    for item in node.keywords:
        if item.arg is not None:
            expanded.append((item.arg, item.value))
            continue
        if not isinstance(item.value, ast.Dict):
            return None
        for key, value in zip(
            item.value.keys,
            item.value.values,
            strict=True,
        ):
            if not (
                isinstance(key, ast.Constant)
                and isinstance(key.value, str)
            ):
                return None
            expanded.append((key.value, value))
    return expanded


def _has_unknown_keyword_unpack(node: ast.Call) -> bool:
    for item in node.keywords:
        if item.arg is not None:
            continue
        if not isinstance(item.value, ast.Dict) or not all(
            isinstance(key, ast.Constant) and isinstance(key.value, str)
            for key in item.value.keys
        ):
            return True
    return False


def _valid_named_signature(
    node: ast.Call,
    *,
    positional_names: Sequence[str],
    keyword_only_names: frozenset[str],
    required_names: frozenset[str],
) -> bool:
    if any(isinstance(argument, ast.Starred) for argument in node.args):
        return False
    if len(node.args) > len(positional_names):
        return False
    keywords = _expanded_keyword_arguments(node)
    if keywords is None:
        return False
    assigned = set(positional_names[: len(node.args)])
    allowed = set(positional_names) | set(keyword_only_names)
    for name, _expression in keywords:
        if name not in allowed or name in assigned:
            return False
        assigned.add(name)
    return required_names <= assigned


_BUILTIN_OPEN_POSITIONAL_NAMES = (
    "file",
    "mode",
    "buffering",
    "encoding",
    "errors",
    "newline",
    "closefd",
    "opener",
)
_FDOPEN_POSITIONAL_NAMES = (
    "fd",
    "mode",
    "buffering",
    "encoding",
    "errors",
    "newline",
    "closefd",
    "opener",
)
_PATH_OPEN_POSITIONAL_NAMES = (
    "mode",
    "buffering",
    "encoding",
    "errors",
    "newline",
)


def _valid_file_open_signature(node: ast.Call, *, kind: str) -> bool:
    positional_names = {
        "builtin": _BUILTIN_OPEN_POSITIONAL_NAMES,
        "fdopen": _FDOPEN_POSITIONAL_NAMES,
        "path": _PATH_OPEN_POSITIONAL_NAMES,
    }[kind]
    required = frozenset() if kind == "path" else frozenset({positional_names[0]})
    return _valid_named_signature(
        node,
        positional_names=positional_names,
        keyword_only_names=frozenset(),
        required_names=required,
    )


def _open_primary_candidates(
    node: ast.Call,
    argument_values: Sequence[FlowValue],
    keyword_values: Mapping[str, FlowValue],
    unknown_keyword_unpack: FlowValue,
    *,
    primary_name: str,
) -> FlowValue:
    values: list[FlowValue] = []
    if argument_values:
        values.append(argument_values[0])
    if primary_name in keyword_values:
        values.append(keyword_values[primary_name])
    if _has_unknown_keyword_unpack(node):
        values.append(unknown_keyword_unpack)
    return _merge_values(values)


def _open_auxiliary_origins(
    argument_values: Sequence[FlowValue],
    keyword_values: Mapping[str, FlowValue],
    unknown_keyword_unpack: FlowValue,
    *,
    primary_name: str | None,
) -> FlowValue:
    values = list(argument_values if primary_name is None else argument_values[1:])
    values.extend(
        value
        for name, value in keyword_values.items()
        if name != primary_name
    )
    if primary_name is None or not unknown_keyword_unpack.has_origins:
        values.append(unknown_keyword_unpack)
    return _merge_values(values)


def _record_open_escape(
    engine: AccessEngine,
    value: FlowValue,
    node: ast.Call,
    *,
    operation: str,
    sink: str,
    reason: str,
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
        operation=operation,
        sink=sink,
        reason=reason,
        path=engine.paths[module],
        line=int(node.lineno),
        ordinal=ordinal,
    )


def _evaluate_stdlib_module_dict_mutation_call(
    engine: AccessEngine,
    node: ast.Call,
    *,
    env: dict[str, FlowValue],
    object_env: dict[str, set[str]],
) -> FlowValue | None:
    if (
        not isinstance(node.func, ast.Attribute)
        or node.func.attr != "pop"
        or not 1 <= len(node.args) <= 2
        or node.keywords
        or any(isinstance(argument, ast.Starred) for argument in node.args)
    ):
        return None
    base = stdlib_module_dict_reference(node.func.value, env)
    if base is None:
        return None
    key = node.args[0]
    attribute = (
        key.value
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
        else STDLIB_MODULE_WILDCARD_ATTRIBUTE
    )
    if not engine._taint_stdlib_module_attribute(
        base,
        attribute=attribute,
        env=env,
        object_env=object_env,
    ):
        return None
    return FlowValue(unknown_callable=True)


def _evaluate_stdlib_vars_call(
    engine: AccessEngine,
    node: ast.Call,
    argument_values: Sequence[FlowValue],
    *,
    env: Mapping[str, FlowValue],
) -> FlowValue | None:
    if (
        not isinstance(node.func, ast.Name)
        or node.func.id != "vars"
        or "vars" in env
        or len(node.args) != 1
        or node.keywords
        or isinstance(node.args[0], ast.Starred)
    ):
        return None
    value = argument_values[0]
    module_ref = precise_stdlib_module_name(value)
    if module_ref is None or module_ref in engine.known_modules:
        return None
    return value.bound("result:builtins.vars:module_dict")


def _is_static_none(expression: ast.expr | None) -> bool:
    return isinstance(expression, ast.Constant) and expression.value is None


def _static_open_option(
    node: ast.Call,
    *,
    index: int,
    keyword: str,
) -> tuple[bool, bool, object | None]:
    provided = _argument_is_provided(node, index=index, keyword=keyword)
    if not provided:
        return False, True, None
    expression = _precise_argument_expression(
        node,
        index=index,
        keyword=keyword,
    )
    if not isinstance(expression, ast.Constant):
        if (
            isinstance(expression, ast.UnaryOp)
            and isinstance(expression.op, (ast.UAdd, ast.USub))
            and isinstance(expression.operand, ast.Constant)
            and type(expression.operand.value) is int
        ):
            value = expression.operand.value
            return (
                True,
                True,
                value if isinstance(expression.op, ast.UAdd) else -value,
            )
        return True, False, None
    return True, True, expression.value


def _open_options_error(
    node: ast.Call,
    *,
    kind: str,
    mode: tuple[Literal["read", "write", "read_write"], str],
    primary_is_path: bool,
) -> str | None:
    offset = 0 if kind == "path" else 1
    binary = "b" in mode[1]
    provided, static, buffering = _static_open_option(
        node,
        index=offset + 1,
        keyword="buffering",
    )
    if provided and (
        not static
        or not isinstance(buffering, int)
        or buffering == 0
        and not binary
    ):
        return "invalid_or_ambiguous_open_options"

    for relative_index, keyword in enumerate(
        ("encoding", "errors", "newline"),
        start=2,
    ):
        provided, static, value = _static_open_option(
            node,
            index=offset + relative_index,
            keyword=keyword,
        )
        if not provided:
            continue
        if not static or value is not None and not isinstance(value, str):
            return "invalid_or_ambiguous_open_options"
        if binary and value is not None:
            return "invalid_or_ambiguous_open_options"
        if keyword == "newline" and value not in {None, "", "\n", "\r", "\r\n"}:
            return "invalid_or_ambiguous_open_options"

    if kind != "path":
        provided, static, closefd = _static_open_option(
            node,
            index=6,
            keyword="closefd",
        )
        if provided and (
            not static or primary_is_path and not bool(closefd)
        ):
            return "invalid_or_ambiguous_open_options"
    return None


def _open_closefd_enabled(node: ast.Call, *, kind: str) -> bool:
    if kind == "path":
        return True
    provided, static, value = _static_open_option(
        node,
        index=6,
        keyword="closefd",
    )
    return not provided or static and bool(value)


def _evaluate_file_open(
    engine: AccessEngine,
    node: ast.Call,
    argument_values: Sequence[FlowValue],
    keyword_values: Mapping[str, FlowValue],
    unknown_keyword_unpack: FlowValue,
    *,
    kind: str,
    receiver: FlowValue,
    module: str,
    actor: str,
    ordinal: int,
) -> FlowValue:
    operation = {
        "builtin": "builtin.open",
        "fdopen": "os.fdopen",
        "path": "path.open",
    }[kind]
    sink = {
        "builtin": "builtins.open",
        "fdopen": "os.fdopen",
        "path": "pathlib.Path.open",
    }[kind]
    primary_name = {
        "builtin": "file",
        "fdopen": "fd",
        "path": None,
    }[kind]
    auxiliary = _open_auxiliary_origins(
        argument_values,
        keyword_values,
        unknown_keyword_unpack,
        primary_name=primary_name,
    )
    if not _valid_file_open_signature(node, kind=kind):
        primary = receiver
        if primary_name is not None:
            primary = _open_primary_candidates(
                node,
                argument_values,
                keyword_values,
                unknown_keyword_unpack,
                primary_name=primary_name,
            )
        _record_open_escape(
            engine,
            primary,
            node,
            operation=operation,
            sink=sink,
            reason="invalid_or_ambiguous_open_signature",
            module=module,
            actor=actor,
            ordinal=ordinal,
        )
        _record_open_escape(
            engine,
            auxiliary,
            node,
            operation=f"{operation}.arguments",
            sink=sink,
            reason="ambiguous_registered_origin_open_arguments",
            module=module,
            actor=actor,
            ordinal=ordinal,
        )
        return FlowValue()

    primary = receiver
    if primary_name is not None:
        primary, ambiguous = _precise_argument_value(
            node,
            argument_values,
            keyword_values,
            index=0,
            keyword=primary_name,
        )
        if ambiguous.has_origins:
            _record_open_escape(
                engine,
                ambiguous,
                node,
                operation=operation,
                sink=sink,
                reason="invalid_or_ambiguous_open_signature",
                module=module,
                actor=actor,
                ordinal=ordinal,
            )
            return FlowValue()

    primary_is_fd = is_exact_os_fd(primary)
    primary_is_path = is_exact_path_receiver(primary)
    if kind == "fdopen" and not primary_is_fd:
        _record_open_escape(
            engine,
            primary,
            node,
            operation=operation,
            sink=sink,
            reason="ambiguous_registered_origin_open_arguments",
            module=module,
            actor=actor,
            ordinal=ordinal,
        )
        _record_open_escape(
            engine,
            auxiliary,
            node,
            operation=f"{operation}.arguments",
            sink=sink,
            reason="ambiguous_registered_origin_open_arguments",
            module=module,
            actor=actor,
            ordinal=ordinal,
        )
        return FlowValue()
    if kind != "fdopen" and primary.has_origins and not (
        primary_is_fd or primary_is_path
    ):
        _record_open_escape(
            engine,
            primary,
            node,
            operation=operation,
            sink=sink,
            reason="ambiguous_registered_origin_open_arguments",
            module=module,
            actor=actor,
            ordinal=ordinal,
        )
        _record_open_escape(
            engine,
            auxiliary,
            node,
            operation=f"{operation}.arguments",
            sink=sink,
            reason="ambiguous_registered_origin_open_arguments",
            module=module,
            actor=actor,
            ordinal=ordinal,
        )
        return FlowValue()

    mode_expression = _precise_argument_expression(
        node,
        index=0 if kind == "path" else 1,
        keyword="mode",
    )
    mode = open_mode_from_expression(mode_expression)
    opener_expression = (
        _precise_argument_expression(node, index=7, keyword="opener")
        if kind != "path"
        else None
    )
    opener_provided = kind != "path" and _argument_is_provided(
        node,
        index=7,
        keyword="opener",
    )
    _record_open_escape(
        engine,
        auxiliary,
        node,
        operation=f"{operation}.arguments",
        sink=sink,
        reason="ambiguous_registered_origin_open_arguments",
        module=module,
        actor=actor,
        ordinal=ordinal,
    )
    if isinstance(mode, str):
        _record_open_escape(
            engine,
            primary,
            node,
            operation=operation,
            sink=sink,
            reason=mode,
            module=module,
            actor=actor,
            ordinal=ordinal,
        )
        return FlowValue()
    if opener_provided and not _is_static_none(opener_expression):
        _record_open_escape(
            engine,
            primary,
            node,
            operation=operation,
            sink=sink,
            reason="unsupported_open_opener",
            module=module,
            actor=actor,
            ordinal=ordinal,
        )
        return FlowValue()
    options_error = _open_options_error(
        node,
        kind=kind,
        mode=mode,
        primary_is_path=primary_is_path,
    )
    if options_error is not None:
        _record_open_escape(
            engine,
            primary,
            node,
            operation=operation,
            sink=sink,
            reason=options_error,
            module=module,
            actor=actor,
            ordinal=ordinal,
        )
        return FlowValue()
    if auxiliary.has_origins:
        _record_open_escape(
            engine,
            primary,
            node,
            operation=operation,
            sink=sink,
            reason="ambiguous_registered_origin_open_arguments",
            module=module,
            actor=actor,
            ordinal=ordinal,
        )
        return FlowValue()
    if not primary.has_origins:
        return FlowValue()
    if primary_is_path:
        engine.facts.record_access(
            primary,
            node=node,
            actor=actor,
            mode=mode[0],
            operation=f"{operation}:{mode[1]}",
            sink=sink,
            path=engine.paths[module],
            line=int(node.lineno),
            ordinal=ordinal,
        )
    return tag_file_handle(
        primary.bound(f"result:{operation}:handle"),
        identity=engine._runtime_object_identity(
            node,
            kind="file",
            actor=actor,
        ),
        wraps_fd=primary_is_fd,
        closefd=_open_closefd_enabled(node, kind=kind),
    )


def _valid_zero_argument_method_signature(node: ast.Call) -> bool:
    return (
        not node.args
        and _expanded_keyword_arguments(node) == []
        and not _has_unknown_keyword_unpack(node)
    )


def _evaluate_file_handle_call(
    engine: AccessEngine,
    node: ast.Call,
    receiver: FlowValue,
    argument_values: Sequence[FlowValue],
    keyword_values: Mapping[str, FlowValue],
    unknown_keyword_unpack: FlowValue,
    *,
    method: str,
    module: str,
    actor: str,
    env: dict[str, FlowValue],
    object_env: dict[str, set[str]],
    ordinal: int,
) -> FlowValue | None:
    if method not in {"close", "fileno"}:
        return None
    if not _valid_zero_argument_method_signature(node):
        implicated = receiver.merged(
            _merge_values(
                [
                    *argument_values,
                    *keyword_values.values(),
                    unknown_keyword_unpack,
                ]
            )
        )
        reason = (
            "invalid_or_ambiguous_fileno_signature"
            if method == "fileno"
            else "registered_locator_to_unknown_callee"
        )
        _record_open_escape(
            engine,
            implicated,
            node,
            operation=f"file.{method}",
            sink=f"io.IOBase.{method}",
            reason=reason,
            module=module,
            actor=actor,
            ordinal=ordinal,
        )
        return FlowValue()
    if method == "close":
        engine._contaminate_runtime_objects(env, object_env, [receiver])
        return FlowValue()
    return tag_os_fd(receiver.bound("result:file.fileno:fd"))


def _is_precise_fcntl_module(value: FlowValue, *, attribute: str) -> bool:
    return is_precise_stdlib_module(
        value,
        module="fcntl",
        attribute=attribute,
    )


def _stdlib_fcntl_call_name(
    node: ast.Call,
    receiver: FlowValue,
    env: Mapping[str, FlowValue],
    *,
    known_modules: frozenset[str],
) -> str | None:
    if "fcntl" in known_modules:
        return None
    if isinstance(node.func, ast.Attribute):
        if node.func.attr in STDLIB_FCNTL_CALLS and _is_precise_fcntl_module(
            receiver,
            attribute=node.func.attr,
        ):
            return node.func.attr
        return None
    if not isinstance(node.func, ast.Name) or node.func.id not in env:
        return None
    value = env[node.func.id]
    for name in STDLIB_FCNTL_CALLS:
        if (
            value.call_targets == {f"fcntl:{name}"}
            and not value.origins
            and not value.module_refs
            and not value.class_targets
            and not value.unknown_callable
            and not value.closure_instances
        ):
            return name
    return None


def _flock_classification(
    masks: frozenset[int],
) -> tuple[Literal["read", "write", "read_write"], str] | None:
    if masks == {8}:
        return "read_write", "unlock"
    if not masks or any(mask not in {1, 2, 5, 6} for mask in masks):
        return None
    lock_kinds = {mask & 3 for mask in masks}
    if not lock_kinds <= {1, 2}:
        return None
    if lock_kinds == {1}:
        mode: Literal["read", "write", "read_write"] = "read"
        operation = "shared"
    elif lock_kinds == {2}:
        mode = "write"
        operation = "exclusive"
    else:
        mode = "read_write"
        operation = "shared_or_exclusive"
    blocking_kinds = {bool(mask & 4) for mask in masks}
    if blocking_kinds == {True}:
        operation = f"{operation}_nonblocking"
    elif len(blocking_kinds) == 2:
        operation = f"{operation}_maybe_nonblocking"
    return mode, operation


def _flock_operation_reason(
    expression: ast.expr | None,
    value: FlowValue,
) -> tuple[
    tuple[Literal["read", "write", "read_write"], str] | None,
    str | None,
]:
    masks = fcntl_lock_masks(value)
    if masks is not None:
        classification = _flock_classification(masks)
        if classification is not None:
            return classification, None
        return None, "invalid_flock_operation"
    if isinstance(expression, ast.Constant):
        return None, "invalid_flock_operation"
    non_fcntl_types = value.object_types.difference(
        {FCNTL_UNRESOLVED_LOCK_OPERATION_OBJECT_TYPE}
    )
    if non_fcntl_types and not all(
        object_type.startswith("stdlib-fcntl-lock-mask:")
        for object_type in non_fcntl_types
    ):
        return None, "invalid_flock_operation"
    if (
        isinstance(expression, ast.BinOp)
        and isinstance(expression.op, ast.BitOr)
        and any(isinstance(child, ast.Constant) for child in ast.walk(expression))
    ):
        return None, "invalid_flock_operation"
    return None, "dynamic_flock_operation"


def _record_flock_escape(
    engine: AccessEngine,
    value: FlowValue,
    node: ast.Call,
    *,
    operation: str,
    reason: str,
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
        operation=operation,
        sink="fcntl.flock",
        reason=reason,
        path=engine.paths[module],
        line=int(node.lineno),
        ordinal=ordinal,
    )


def _evaluate_flock(
    engine: AccessEngine,
    node: ast.Call,
    argument_values: Sequence[FlowValue],
    keyword_values: Mapping[str, FlowValue],
    unknown_keyword_unpack: FlowValue,
    *,
    module: str,
    actor: str,
    ordinal: int,
) -> FlowValue:
    valid_signature = (
        len(node.args) == 2
        and not any(isinstance(argument, ast.Starred) for argument in node.args)
        and not node.keywords
    )
    if not valid_signature:
        descriptor_values: list[FlowValue] = []
        if argument_values:
            descriptor_values.append(argument_values[0])
        if "fd" in keyword_values:
            descriptor_values.append(keyword_values["fd"])
        if _has_unknown_keyword_unpack(node):
            descriptor_values.append(unknown_keyword_unpack)
        descriptor = _merge_values(descriptor_values)
        auxiliary = _merge_values(
            [
                *argument_values[1:],
                *(
                    value
                    for name, value in keyword_values.items()
                    if name != "fd"
                ),
            ]
        )
        _record_flock_escape(
            engine,
            descriptor,
            node,
            operation="fcntl.flock",
            reason="invalid_or_ambiguous_flock_signature",
            module=module,
            actor=actor,
            ordinal=ordinal,
        )
        _record_flock_escape(
            engine,
            auxiliary,
            node,
            operation="fcntl.flock.arguments",
            reason="ambiguous_registered_origin_flock_arguments",
            module=module,
            actor=actor,
            ordinal=ordinal,
        )
        return FlowValue()

    descriptor = argument_values[0]
    operation_value = argument_values[1]
    _record_flock_escape(
        engine,
        operation_value,
        node,
        operation="fcntl.flock.arguments",
        reason="ambiguous_registered_origin_flock_arguments",
        module=module,
        actor=actor,
        ordinal=ordinal,
    )
    if not is_exact_flock_descriptor(descriptor):
        _record_flock_escape(
            engine,
            descriptor,
            node,
            operation="fcntl.flock",
            reason="ambiguous_registered_origin_flock_descriptor",
            module=module,
            actor=actor,
            ordinal=ordinal,
        )
        return FlowValue()
    operation_expression = node.args[1]
    classification, reason = _flock_operation_reason(
        operation_expression,
        operation_value,
    )
    if classification is None:
        assert reason is not None
        _record_flock_escape(
            engine,
            descriptor,
            node,
            operation="fcntl.flock",
            reason=reason,
            module=module,
            actor=actor,
            ordinal=ordinal,
        )
        return FlowValue()
    engine.facts.record_access(
        descriptor,
        node=node,
        actor=actor,
        mode=classification[0],
        operation=f"fcntl.flock:{classification[1]}",
        sink="fcntl.flock",
        path=engine.paths[module],
        line=int(node.lineno),
        ordinal=ordinal,
    )
    return FlowValue()


def _valid_sqlite_connect_signature(node: ast.Call) -> bool:
    return _valid_named_signature(
        node,
        positional_names=(
            "database",
            "timeout",
            "detect_types",
            "isolation_level",
            "check_same_thread",
            "factory",
            "cached_statements",
            "uri",
        ),
        keyword_only_names=frozenset({"autocommit"}),
        required_names=frozenset({"database"}),
    )


def _valid_sqlite_method_signature(node: ast.Call, *, method: str) -> bool:
    if any(isinstance(argument, ast.Starred) for argument in node.args):
        return False
    keywords = _expanded_keyword_arguments(node)
    if keywords is None:
        return False
    if method == "execute":
        return 1 <= len(node.args) <= 2 and not keywords
    if method == "executemany":
        return len(node.args) == 2 and not keywords
    if method == "executescript":
        return len(node.args) == 1 and not keywords
    if method in {"commit", "rollback", "close", "fetchone", "fetchall"}:
        return not node.args and not keywords
    if method == "fetchmany":
        return (
            len(node.args) <= 1
            and not keywords
            or not node.args
            and [name for name, _expression in keywords] == ["size"]
        )
    if method == "cursor":
        return (
            len(node.args) <= 1
            and not keywords
            or not node.args
            and [name for name, _expression in keywords] == ["factory"]
        )
    return False


def _argument_is_provided(
    node: ast.Call,
    *,
    index: int,
    keyword: str,
) -> bool:
    return len(node.args) > index or bool(
        _keyword_argument_expressions(node, keyword=keyword)
    )


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
        if _is_precise_os_module(base, attribute=expression.attr):
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
    return tag_os_fd(
        path_value.bound("result:os.open:fd"),
        identity=engine._runtime_object_identity(
            node,
            kind="fd",
            actor=actor,
        ),
    )


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


SQLiteAccessMode = Literal["read", "write", "read_write"]
_SQL_DYNAMIC = "\0sqlite-dynamic\0"
_SQLITE_READ_VERBS = frozenset({"SELECT"})
_SQLITE_WRITE_VERBS = frozenset(
    {
        "ALTER",
        "CREATE",
        "DELETE",
        "DROP",
        "INSERT",
        "REPLACE",
        "UPDATE",
        "VACUUM",
    }
)
_SQLITE_READ_PRAGMAS = frozenset(
    {
        "application_id",
        "collation_list",
        "compile_options",
        "data_version",
        "database_list",
        "foreign_key_check",
        "foreign_key_list",
        "freelist_count",
        "function_list",
        "index_info",
        "index_list",
        "index_xinfo",
        "integrity_check",
        "journal_mode",
        "module_list",
        "page_count",
        "page_size",
        "pragma_list",
        "quick_check",
        "schema_version",
        "table_info",
        "table_list",
        "table_xinfo",
        "user_version",
    }
)
_SQLITE_WRITE_PRAGMAS = frozenset(
    {"incremental_vacuum", "optimize", "shrink_memory", "wal_checkpoint"}
)
_SQLITE_READ_PRAGMA_ARGUMENTS = frozenset(
    {
        "foreign_key_check",
        "foreign_key_list",
        "index_info",
        "index_xinfo",
        "integrity_check",
        "quick_check",
        "table_info",
        "table_xinfo",
    }
)
_SQLITE_WRITE_PRAGMA_ARGUMENTS = frozenset(
    {"application_id", "journal_mode", "page_size", "user_version"}
)


def _tag_sqlite_handle(value: FlowValue, *, kind: str) -> FlowValue:
    return tag_sqlite_handle(value, kind=kind)


def _is_precise_sqlite3_module(
    value: FlowValue, *, attribute: str
) -> bool:
    return is_precise_stdlib_module(
        value,
        module="sqlite3",
        attribute=attribute,
    )


def _is_precise_sqlite3_type(
    expression: ast.expr | None,
    env: Mapping[str, FlowValue],
    *,
    type_name: str,
) -> bool:
    if isinstance(expression, ast.Attribute) and isinstance(
        expression.value, ast.Name
    ):
        return (
            expression.attr == type_name
            and _is_precise_sqlite3_module(
                env.get(expression.value.id, FlowValue()),
                attribute=type_name,
            )
        )
    if not isinstance(expression, ast.Name):
        return False
    value = env.get(expression.id, FlowValue())
    return (
        value.object_types == {f"{SQLITE_TYPE_OBJECT_PREFIX}{type_name}"}
        and not value.origins
        and not value.module_refs
        and not value.call_targets
        and not value.class_targets
        and not value.unknown_callable
        and not value.closure_instances
    )


def _stdlib_sqlite3_call_name(
    node: ast.Call,
    receiver: FlowValue,
    env: Mapping[str, FlowValue],
    *,
    known_modules: frozenset[str],
) -> str | None:
    if "sqlite3" in known_modules:
        return None
    if isinstance(node.func, ast.Attribute):
        if (
            node.func.attr in STDLIB_SQLITE3_CALLS
            and _is_precise_sqlite3_module(
                receiver, attribute=node.func.attr
            )
        ):
            return node.func.attr
        return None
    if not isinstance(node.func, ast.Name) or node.func.id not in env:
        return None
    value = env[node.func.id]
    for name in STDLIB_SQLITE3_CALLS:
        if (
            value.call_targets == {f"sqlite3:{name}"}
            and not value.origins
            and not value.module_refs
            and not value.class_targets
            and not value.unknown_callable
            and not value.closure_instances
        ):
            return name
    return None


def _static_text_shape(expression: ast.expr | None) -> str | None:
    if isinstance(expression, ast.Constant) and isinstance(expression.value, str):
        return expression.value
    if not isinstance(expression, ast.JoinedStr):
        return None
    parts: list[str] = []
    for item in expression.values:
        if isinstance(item, ast.Constant) and isinstance(item.value, str):
            parts.append(item.value)
        elif isinstance(item, ast.FormattedValue):
            parts.append(_SQL_DYNAMIC)
        else:
            return None
    return "".join(parts)


def _database_shape_has_read_only_mode(
    expression: ast.expr | None,
    value: FlowValue,
    resource_locators: Mapping[str, str],
) -> bool:
    shape = _static_text_shape(expression)
    locators = [
        resource_locators[resource_id]
        for resource_id in value.origins
        if resource_id in resource_locators
    ]
    if shape is not None:
        if _SQL_DYNAMIC not in shape:
            return _sqlite_uri_is_read_only(shape)
        if shape.count(_SQL_DYNAMIC) != 1 or not locators:
            return False
        return all(
            _sqlite_uri_is_read_only(shape.replace(_SQL_DYNAMIC, locator))
            for locator in locators
        )
    if not _value_preserves_locator_shape(value):
        return False
    return bool(locators) and all(_sqlite_uri_is_read_only(uri) for uri in locators)


def _sqlite_uri_is_read_only(uri: str) -> bool:
    if not uri.startswith("file:"):
        return False
    parsed = urlsplit(uri)
    if parsed.scheme != "file" or _SQL_DYNAMIC in parsed.query:
        return False
    modes = [
        value
        for name, value in parse_qsl(parsed.query, keep_blank_values=True)
        if name == "mode"
    ]
    return bool(modes) and all(mode == "ro" for mode in modes)


def _value_preserves_locator_shape(value: FlowValue) -> bool:
    return all(
        not any(
            step.startswith(("expression:", "transform:"))
            for step in chain
        )
        for chains in value.origins.values()
        for chain in chains
    )


def _is_static_true(expression: ast.expr | None) -> bool:
    return isinstance(expression, ast.Constant) and expression.value is True


def _record_sqlite_escape(
    engine: AccessEngine,
    value: FlowValue,
    node: ast.Call,
    *,
    operation: str,
    reason: str,
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
        operation=operation,
        sink="sqlite3",
        reason=reason,
        path=engine.paths[module],
        line=int(node.lineno),
        ordinal=ordinal,
    )


def _evaluate_sqlite_connect(
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
    database_arguments = []
    if argument_values:
        database_arguments.append(argument_values[0])
    if "database" in keyword_values:
        database_arguments.append(keyword_values["database"])
    database_origins = _merge_values(database_arguments)
    auxiliary_origins = unknown_keyword_unpack.merged(
        _merge_values(
            [
                *argument_values[1:],
                *(
                    value
                    for name, value in keyword_values.items()
                    if name != "database"
                ),
            ]
        )
    )
    if not _valid_sqlite_connect_signature(node):
        _record_sqlite_escape(
            engine,
            database_origins,
            node,
            operation="sqlite.connect",
            reason="invalid_or_ambiguous_sqlite_signature",
            module=module,
            actor=actor,
            ordinal=ordinal,
        )
        _record_sqlite_escape(
            engine,
            auxiliary_origins,
            node,
            operation="sqlite.connect.arguments",
            reason="ambiguous_registered_origin_sqlite_connect_arguments",
            module=module,
            actor=actor,
            ordinal=ordinal,
        )
        return FlowValue()
    database, ambiguous_database = _precise_argument_value(
        node,
        argument_values,
        keyword_values,
        index=0,
        keyword="database",
    )
    implicated = ambiguous_database.merged(auxiliary_origins)
    if database.has_origins and not is_path_receiver(database):
        implicated = implicated.merged(database)
    if implicated.has_origins:
        _record_sqlite_escape(
            engine,
            implicated.merged(database),
            node,
            operation="sqlite.connect",
            reason="ambiguous_registered_origin_sqlite_connect_arguments",
            module=module,
            actor=actor,
            ordinal=ordinal,
        )
        return FlowValue()
    if _has_unknown_keyword_unpack(node):
        _record_sqlite_escape(
            engine,
            database,
            node,
            operation="sqlite.connect",
            reason="ambiguous_registered_origin_sqlite_connect_arguments",
            module=module,
            actor=actor,
            ordinal=ordinal,
        )
        return FlowValue()
    database_expression = _precise_argument_expression(
        node, index=0, keyword="database"
    )
    if database_expression is None:
        _record_sqlite_escape(
            engine,
            database,
            node,
            operation="sqlite.connect",
            reason="ambiguous_registered_origin_sqlite_connect_arguments",
            module=module,
            actor=actor,
            ordinal=ordinal,
        )
        return FlowValue()
    factory_expression = _precise_argument_expression(
        node, index=5, keyword="factory"
    )
    if _argument_is_provided(node, index=5, keyword="factory") and not (
        _is_precise_sqlite3_type(
            factory_expression,
            env,
            type_name="Connection",
        )
    ):
        _record_sqlite_escape(
            engine,
            database,
            node,
            operation="sqlite.connect",
            reason="unsupported_sqlite_connect_factory",
            module=module,
            actor=actor,
            ordinal=ordinal,
        )
        return FlowValue()
    uri_expression = _precise_argument_expression(node, index=7, keyword="uri")
    read_only = _is_static_true(uri_expression) and (
        _database_shape_has_read_only_mode(
            database_expression,
            database,
            engine.resource_locators,
        )
    )
    if database.has_origins:
        engine.facts.record_access(
            database,
            node=node,
            actor=actor,
            mode="read" if read_only else "read_write",
            operation="sqlite.connect:ro" if read_only else "sqlite.connect:rwc",
            sink="sqlite3.connect",
            path=engine.paths[module],
            line=int(node.lineno),
            ordinal=ordinal,
        )
    return _tag_sqlite_handle(
        database.bound("result:sqlite.connect:connection"),
        kind="connection",
    )


def _strip_sql_prefix(text: str) -> str | None:
    remaining = text
    while True:
        remaining = remaining.lstrip()
        if remaining.startswith("--"):
            newline = remaining.find("\n")
            if newline < 0:
                return ""
            if _SQL_DYNAMIC in remaining[:newline]:
                return None
            remaining = remaining[newline + 1 :]
            continue
        if remaining.startswith("/*"):
            end = remaining.find("*/", 2)
            if end < 0:
                return None
            if _SQL_DYNAMIC in remaining[: end + 2]:
                return None
            remaining = remaining[end + 2 :]
            continue
        return remaining


def _classify_pragma(text: str) -> SQLiteAccessMode | None:
    if _SQL_DYNAMIC in text:
        return None
    match = re.match(
        r"PRAGMA\s+(?:(?:[A-Za-z_]\w*)\.)?([A-Za-z_]\w*)(.*)$",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if match is None:
        return None
    name = match.group(1).lower()
    remainder = match.group(2).strip()
    if "=" in remainder:
        return "write"
    if remainder.startswith("("):
        if name in _SQLITE_READ_PRAGMA_ARGUMENTS:
            return "read"
        if name in (_SQLITE_WRITE_PRAGMA_ARGUMENTS | _SQLITE_WRITE_PRAGMAS):
            return "write"
        return None
    if remainder:
        return None
    if name in _SQLITE_WRITE_PRAGMAS:
        return "write"
    if name in _SQLITE_READ_PRAGMAS:
        return "read"
    return None


def _classify_sql_statement(
    text: str,
) -> tuple[SQLiteAccessMode, str | None] | None:
    stripped = _strip_sql_prefix(text)
    if stripped is None:
        return None
    match = re.match(r"([A-Za-z]+)", stripped)
    if match is None:
        return None
    verb = match.group(1).upper()
    if verb in _SQLITE_READ_VERBS:
        return "read", None
    if verb in _SQLITE_WRITE_VERBS:
        return "write", None
    if verb == "PRAGMA":
        mode = _classify_pragma(stripped)
        return None if mode is None else (mode, None)
    if verb == "BEGIN" and re.match(
        r"BEGIN\s+IMMEDIATE(?:\s|;|$)", stripped, re.IGNORECASE
    ):
        return "write", "sqlite.transaction.begin_immediate"
    return None


def _join_sqlite_modes(modes: Sequence[SQLiteAccessMode]) -> SQLiteAccessMode:
    if not modes:
        return "read"
    if all(mode == "read" for mode in modes):
        return "read"
    if all(mode == "write" for mode in modes):
        return "write"
    return "read_write"


def _split_sql_statements(text: str) -> list[str] | None:
    statements: list[str] = []
    start = 0
    index = 0
    quote_end: str | None = None
    while index < len(text):
        character = text[index]
        if quote_end is not None:
            if character == quote_end:
                if quote_end != "]" and index + 1 < len(text) and (
                    text[index + 1] == quote_end
                ):
                    index += 2
                    continue
                quote_end = None
            index += 1
            continue
        if text.startswith("--", index):
            newline = text.find("\n", index + 2)
            index = len(text) if newline < 0 else newline + 1
            continue
        if text.startswith("/*", index):
            end = text.find("*/", index + 2)
            if end < 0:
                return None
            index = end + 2
            continue
        if character in {"'", '"', "`"}:
            quote_end = character
        elif character == "[":
            quote_end = "]"
        elif character == ";":
            statement = text[start:index].strip()
            if statement:
                statements.append(statement)
            start = index + 1
        index += 1
    if quote_end is not None:
        return None
    tail = text[start:].strip()
    if tail:
        statements.append(tail)
    return statements


def _classify_sql(
    expression: ast.expr | None,
    *,
    method: str,
) -> tuple[SQLiteAccessMode, str] | None:
    shape = _static_text_shape(expression)
    if shape is None:
        return None
    if method != "executescript":
        statements = _split_sql_statements(shape)
        if statements is None or len(statements) != 1:
            return None
        classified = _classify_sql_statement(statements[0])
        if classified is None:
            return None
        mode, special_operation = classified
        return mode, special_operation or f"sqlite.{method}:{mode}"
    if _SQL_DYNAMIC in shape:
        return None
    statements = _split_sql_statements(shape)
    if statements is None:
        return None
    classifications = [_classify_sql_statement(part) for part in statements]
    if not statements or any(item is None for item in classifications):
        return None
    modes = [item[0] for item in classifications if item is not None]
    mode = _join_sqlite_modes(modes)
    return mode, f"sqlite.executescript:{mode}"


def _evaluate_sqlite_handle_call(
    engine: AccessEngine,
    node: ast.Call,
    receiver: FlowValue,
    argument_values: Sequence[FlowValue],
    keyword_values: Mapping[str, FlowValue],
    unknown_keyword_unpack: FlowValue,
    *,
    handle_kind: str,
    method: str,
    module: str,
    actor: str,
    env: Mapping[str, FlowValue],
    ordinal: int,
) -> FlowValue | None:
    execute_methods = {"execute", "executemany", "executescript"}
    fetch_methods = {"fetchone", "fetchall", "fetchmany"}
    recognized = execute_methods | fetch_methods | {
        "close",
        "commit",
        "cursor",
        "rollback",
    }
    if method not in recognized:
        return None
    arguments = _merge_values(
        [*argument_values, *keyword_values.values(), unknown_keyword_unpack]
    )
    _record_sqlite_escape(
        engine,
        arguments,
        node,
        operation=f"sqlite.{method}.arguments",
        reason="ambiguous_registered_origin_sqlite_arguments",
        module=module,
        actor=actor,
        ordinal=ordinal,
    )
    if not _valid_sqlite_method_signature(node, method=method):
        _record_sqlite_escape(
            engine,
            receiver,
            node,
            operation=f"sqlite.{method}",
            reason="invalid_or_ambiguous_sqlite_signature",
            module=module,
            actor=actor,
            ordinal=ordinal,
        )
        return FlowValue()
    has_unknown_unpack = _has_unknown_keyword_unpack(node)
    if method == "cursor" and handle_kind == "connection":
        factory_expression = _precise_argument_expression(
            node, index=0, keyword="factory"
        )
        unsupported_factory = has_unknown_unpack or (
            _argument_is_provided(node, index=0, keyword="factory")
            and not _is_precise_sqlite3_type(
                factory_expression, env, type_name="Cursor"
            )
        )
        if unsupported_factory:
            _record_sqlite_escape(
                engine,
                receiver,
                node,
                operation="sqlite.cursor",
                reason="unsupported_sqlite_cursor_factory",
                module=module,
                actor=actor,
                ordinal=ordinal,
            )
            return FlowValue()
        return _tag_sqlite_handle(
            receiver.bound("result:sqlite.connection.cursor"), kind="cursor"
        )
    if method in execute_methods:
        keyword = "sql_script" if method == "executescript" else "sql"
        sql_expression = _precise_argument_expression(
            node,
            index=0,
            keyword=keyword,
        )
        classification = (
            None
            if has_unknown_unpack
            else _classify_sql(sql_expression, method=method)
        )
        if classification is None:
            _record_sqlite_escape(
                engine,
                receiver,
                node,
                operation=f"sqlite.{method}",
                reason="dynamic_or_unsupported_sqlite_sql",
                module=module,
                actor=actor,
                ordinal=ordinal,
            )
        elif receiver.has_origins:
            engine.facts.record_access(
                receiver,
                node=node,
                actor=actor,
                mode=classification[0],
                operation=classification[1],
                sink=f"sqlite3.{handle_kind}.{method}",
                path=engine.paths[module],
                line=int(node.lineno),
                ordinal=ordinal,
            )
        return _tag_sqlite_handle(
            receiver.bound(f"result:sqlite.{method}:cursor"), kind="cursor"
        )
    if has_unknown_unpack:
        _record_sqlite_escape(
            engine,
            receiver,
            node,
            operation=f"sqlite.{method}",
            reason="ambiguous_sqlite_lifecycle_arguments",
            module=module,
            actor=actor,
            ordinal=ordinal,
        )
        return FlowValue()
    if method in {"commit", "rollback"} and handle_kind == "connection":
        if receiver.has_origins:
            engine.facts.record_access(
                receiver,
                node=node,
                actor=actor,
                mode="write",
                operation=f"sqlite.transaction.{method}",
                sink=f"sqlite3.Connection.{method}",
                path=engine.paths[module],
                line=int(node.lineno),
                ordinal=ordinal,
            )
        return FlowValue()
    if method == "close" or (method in fetch_methods and handle_kind == "cursor"):
        return FlowValue()
    return None


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
    path_open = False
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
        exact_path_receiver = is_exact_path_receiver(receiver)
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
        if exact_path_receiver and method == "open":
            path_open = True
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
            value = engine._eval(
                keyword.value,
                module=module,
                actor=actor,
                class_ref=class_ref,
                env=env,
                object_env=object_env,
                call_ordinals=call_ordinals,
            )
            keyword_values[keyword.arg] = keyword_values.get(
                keyword.arg,
                FlowValue(),
            ).merged(value)
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
                    keyword_values[key.value] = keyword_values.get(
                        key.value,
                        FlowValue(),
                    ).merged(value)
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
    dict_mutation = _evaluate_stdlib_module_dict_mutation_call(
        engine,
        node,
        env=env,
        object_env=object_env,
    )
    if dict_mutation is not None:
        return dict_mutation
    vars_call = _evaluate_stdlib_vars_call(
        engine,
        node,
        argument_values,
        env=env,
    )
    if vars_call is not None:
        return vars_call
    builtin_mutation = _evaluate_builtin_stdlib_module_mutation(
        engine,
        node,
        argument_values,
        module=module,
        actor=actor,
        env=env,
        object_env=object_env,
        ordinal=ordinal,
    )
    if builtin_mutation is not None:
        return builtin_mutation
    if path_open:
        return _evaluate_file_open(
            engine,
            node,
            argument_values,
            keyword_values,
            unknown_keyword_unpack,
            kind="path",
            receiver=receiver,
            module=module,
            actor=actor,
            ordinal=ordinal,
        )
    if file_handle_kind(receiver) == "file" and isinstance(
        node.func,
        ast.Attribute,
    ):
        file_result = _evaluate_file_handle_call(
            engine,
            node,
            receiver,
            argument_values,
            keyword_values,
            unknown_keyword_unpack,
            method=node.func.attr,
            module=module,
            actor=actor,
            env=env,
            object_env=object_env,
            ordinal=ordinal,
        )
        if file_result is not None:
            return file_result
    source_call_name = call_name(node.func)
    canonical_call_sink = _canonical_stdlib_call_sink(
        node,
        receiver,
        env,
        known_modules=engine.known_modules,
    )
    builtins_call = _stdlib_builtins_call_name(
        node,
        receiver,
        env,
        known_modules=engine.known_modules,
    )
    bare_builtin_open = _is_bare_builtin_open_reference(
        engine,
        node,
        target=resolution.fallback_target,
        actor=actor,
        module=module,
        env=env,
    )
    if canonical_call_sink is None and bare_builtin_open:
        canonical_call_sink = "builtins.open"
    if builtins_call == "open" or (
        bare_builtin_open and _builtin_open_is_unmutated(env)
    ):
        return _evaluate_file_open(
            engine,
            node,
            argument_values,
            keyword_values,
            unknown_keyword_unpack,
            kind="builtin",
            receiver=receiver,
            module=module,
            actor=actor,
            ordinal=ordinal,
        )
    fcntl_call = _stdlib_fcntl_call_name(
        node,
        receiver,
        env,
        known_modules=engine.known_modules,
    )
    if fcntl_call == "flock":
        return _evaluate_flock(
            engine,
            node,
            argument_values,
            keyword_values,
            unknown_keyword_unpack,
            module=module,
            actor=actor,
            ordinal=ordinal,
        )
    sqlite3_call = _stdlib_sqlite3_call_name(
        node,
        receiver,
        env,
        known_modules=engine.known_modules,
    )
    if sqlite3_call == "connect":
        return _evaluate_sqlite_connect(
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
    handle_kind = sqlite_handle_kind(receiver)
    if handle_kind is not None and isinstance(node.func, ast.Attribute):
        sqlite_result = _evaluate_sqlite_handle_call(
            engine,
            node,
            receiver,
            argument_values,
            keyword_values,
            unknown_keyword_unpack,
            handle_kind=handle_kind,
            method=node.func.attr,
            module=module,
            actor=actor,
            env=env,
            ordinal=ordinal,
        )
        if sqlite_result is not None:
            return sqlite_result
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
    if os_call == "fdopen":
        return _evaluate_file_open(
            engine,
            node,
            argument_values,
            keyword_values,
            unknown_keyword_unpack,
            kind="fdopen",
            receiver=receiver,
            module=module,
            actor=actor,
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
    if authoritative_value is not None and authoritative_value.has_origins:
        escaped = escaped.merged(authoritative_value)
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
            info = engine.functions[target]
            conservative_values = [
                *argument_values,
                *keyword_values.values(),
                unknown_keyword_unpack,
                *(
                    env[name]
                    for name in info.referenced_names | info.nonlocal_names
                    if name in env
                    and not _stdlib_reference_is_read_only(info, name)
                ),
            ]
            _taint_callable_stdlib_modules(
                engine,
                conservative_values,
                env=env,
                object_env=object_env,
            )
            engine._require_function_summary(target)
    if known_targets or known_class_targets:
        if known_class_targets or has_unknown:
            _taint_callable_stdlib_modules(
                engine,
                [
                    *argument_values,
                    *keyword_values.values(),
                    unknown_keyword_unpack,
                ],
                env=env,
                object_env=object_env,
            )
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
                    sink=canonical_call_sink or source_call_name or "<dynamic>",
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
                    sink=canonical_call_sink or source_call_name or "<dynamic>",
                    reason="registered_locator_to_unknown_callee",
                    path=engine.paths[module],
                    line=int(node.lineno),
                    ordinal=ordinal,
                )
            if has_unknown:
                returned = returned.merged(FlowValue(unknown_callable=True))
            engine._contaminate_runtime_objects(
                env,
                object_env,
                [
                    receiver,
                    *argument_values,
                    *keyword_values.values(),
                    unknown_keyword_unpack,
                    *(
                        [authoritative_value]
                        if authoritative_value is not None
                        else []
                    ),
                ],
            )
        return returned
    target = resolution.fallback_target
    _taint_callable_stdlib_modules(
        engine,
        [
            *argument_values,
            *keyword_values.values(),
            unknown_keyword_unpack,
        ],
        env=env,
        object_env=object_env,
    )
    if escaped.has_origins:
        engine.facts.record_escape(
            escaped,
            node=node,
            actor=actor,
            operation=f"call:{source_call_name or '<dynamic>'}",
            sink=canonical_call_sink or target or source_call_name or "<dynamic>",
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
            sink=canonical_call_sink or target or source_call_name or "<dynamic>",
            reason="closure_to_unknown_callee",
            path=engine.paths[module],
            line=int(node.lineno),
            ordinal=ordinal,
        )
    engine._contaminate_runtime_objects(
        env,
        object_env,
        [
            receiver,
            *argument_values,
            *keyword_values.values(),
            unknown_keyword_unpack,
            *(
                [authoritative_value]
                if authoritative_value is not None
                else []
            ),
        ],
    )
    return FlowValue(unknown_callable=True)


def _is_bare_builtin_open_reference(
    engine: AccessEngine,
    node: ast.Call,
    *,
    target: str,
    actor: str,
    module: str,
    env: dict[str, FlowValue],
) -> bool:
    if not isinstance(node.func, ast.Name) or node.func.id != "open":
        return False
    if target not in {"", "builtins:open"}:
        return False
    if "open" in env or "open" in engine.imported_symbols.get(module, {}):
        return False
    info = engine.functions.get(actor)
    return info is None or not (
        "open" in info.parameters or "open" in info.local_names
    )


def _builtin_open_is_unmutated(env: Mapping[str, FlowValue]) -> bool:
    marker = stdlib_module_mutation_marker("builtins", "open")
    wildcard = stdlib_module_mutation_marker(
        "builtins",
        STDLIB_MODULE_WILDCARD_ATTRIBUTE,
    )
    return not any(
        value.module_refs == {"builtins"}
        and bool({marker, wildcard}.intersection(value.object_types))
        for value in env.values()
    )


__all__ = ["AccessEngine", "evaluate_call"]
