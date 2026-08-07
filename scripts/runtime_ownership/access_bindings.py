"""Structured target and definition-time binding semantics."""

from __future__ import annotations

import ast
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from .access_facts import AccessFactCollector
from .access_model import FlowValue, sqlite_handle_kind


class ValueEngine(Protocol):
    facts: AccessFactCollector
    paths: dict[str, str]

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
    ) -> bool: ...

    def _assignment_binding_value(
        self,
        name: str,
        value: FlowValue,
        *,
        module: str,
        actor: str,
    ) -> FlowValue: ...


class DefinitionEngine(ValueEngine, Protocol):
    function_refs_by_node: dict[int, str]

    def _merge_closure(
        self,
        ref: str,
        *,
        actor: str,
        module: str,
        env: Mapping[str, FlowValue],
    ) -> bool: ...

    def _merge_definition_default(
        self,
        ref: str,
        parameter: str,
        value: FlowValue,
    ) -> bool: ...


@dataclass
class StructuredValue:
    value: FlowValue
    items: tuple[StructuredValue, ...] | None = None


@dataclass
class IterableValue:
    item: StructuredValue
    aggregate: FlowValue
    literal: bool
    literal_length: int | None


@dataclass
class AssignmentResult:
    changed: bool
    mismatch: bool


def analyze_assignment(
    engine: ValueEngine,
    statement: ast.Assign | ast.AnnAssign,
    *,
    module: str,
    actor: str,
    class_ref: str | None,
    env: dict[str, FlowValue],
    object_env: dict[str, set[str]],
    call_ordinals: Mapping[int, int],
) -> AssignmentResult:
    value_node = statement.value
    if value_node is None:
        return AssignmentResult(False, False)
    value = evaluate_structured(
        engine,
        value_node,
        module=module,
        actor=actor,
        class_ref=class_ref,
        env=env,
        object_env=object_env,
        call_ordinals=call_ordinals,
    )
    targets = (
        statement.targets if isinstance(statement, ast.Assign) else [statement.target]
    )
    if any(structured_target_mismatch(target, value) for target in targets):
        return AssignmentResult(False, True)
    changed = False
    for target in targets:
        changed |= bind_structured_target(
            engine,
            target,
            value,
            module=module,
            actor=actor,
            class_ref=class_ref,
            env=env,
            object_env=object_env,
            call_ordinals=call_ordinals,
            inject_registered_origin=True,
        )
    return AssignmentResult(changed, False)


def analyze_augassign(
    engine: ValueEngine,
    statement: ast.AugAssign,
    *,
    module: str,
    actor: str,
    class_ref: str | None,
    env: dict[str, FlowValue],
    object_env: dict[str, set[str]],
    call_ordinals: Mapping[int, int],
) -> bool:
    """Evaluate an augmented assignment once and retain conservative provenance."""

    target_value = engine._eval(
        statement.target,
        module=module,
        actor=actor,
        class_ref=class_ref,
        env=env,
        object_env=object_env,
        call_ordinals=call_ordinals,
    )
    rhs_value = engine._eval(
        statement.value,
        module=module,
        actor=actor,
        class_ref=class_ref,
        env=env,
        object_env=object_env,
        call_ordinals=call_ordinals,
    )
    merged = target_value.merged(rhs_value)
    if merged.has_origins:
        engine.facts.record_escape(
            merged,
            node=statement,
            actor=actor,
            operation=f"augassign:{type(statement.op).__name__.lower()}",
            sink="python.AugAssign",
            reason="unsupported_registered_origin_augassign",
            path=engine.paths[module],
            line=int(statement.lineno),
            ordinal=int(call_ordinals.get(id(statement.target), 0)),
        )
    return engine._bind_target(
        statement.target,
        merged,
        actor=actor,
        class_ref=class_ref,
        env=env,
        object_env=object_env,
        module=module,
        source_node=statement,
        ordinal=int(call_ordinals.get(id(statement.target), 0)),
    )


def analyze_definition(
    engine: DefinitionEngine,
    statement: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
    *,
    module: str,
    actor: str,
    class_ref: str | None,
    env: dict[str, FlowValue],
    object_env: dict[str, set[str]],
    call_ordinals: Mapping[int, int],
) -> bool:
    changed = False
    for decorator in statement.decorator_list:
        _evaluate_definition_use(
            engine,
            decorator,
            kind="decorator",
            module=module,
            actor=actor,
            class_ref=class_ref,
            env=env,
            object_env=object_env,
            call_ordinals=call_ordinals,
        )
    nested_ref: str | None = None
    if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
        nested_ref = engine.function_refs_by_node.get(id(statement))
        for parameter, default in _function_defaults(statement):
            value = engine._eval(
                default,
                module=module,
                actor=actor,
                class_ref=class_ref,
                env=env,
                object_env=object_env,
                call_ordinals=call_ordinals,
            )
            if nested_ref is not None:
                changed |= engine._merge_definition_default(
                    nested_ref, parameter, value
                )
    else:
        for base in statement.bases:
            _evaluate_definition_use(
                engine,
                base,
                kind="class_base",
                module=module,
                actor=actor,
                class_ref=class_ref,
                env=env,
                object_env=object_env,
                call_ordinals=call_ordinals,
            )
        for keyword in statement.keywords:
            _evaluate_definition_use(
                engine,
                keyword.value,
                kind="class_keyword",
                module=module,
                actor=actor,
                class_ref=class_ref,
                env=env,
                object_env=object_env,
                call_ordinals=call_ordinals,
            )
    changed |= engine._bind_target(
        ast.Name(id=statement.name, ctx=ast.Store()),
        FlowValue(),
        actor=actor,
        class_ref=class_ref,
        env=env,
        object_env=object_env,
    )
    if nested_ref is not None:
        changed |= engine._merge_closure(
            nested_ref,
            actor=actor,
            module=module,
            env=env,
        )
    return changed


def evaluate_structured(
    engine: ValueEngine,
    node: ast.expr,
    *,
    module: str,
    actor: str,
    class_ref: str | None,
    env: dict[str, FlowValue],
    object_env: dict[str, set[str]],
    call_ordinals: Mapping[int, int],
) -> StructuredValue:
    if isinstance(node, (ast.Tuple, ast.List)) and not any(
        isinstance(item, ast.Starred) for item in node.elts
    ):
        items = tuple(
            evaluate_structured(
                engine,
                item,
                module=module,
                actor=actor,
                class_ref=class_ref,
                env=env,
                object_env=object_env,
                call_ordinals=call_ordinals,
            )
            for item in node.elts
        )
        return StructuredValue(_merge_values(items), items)
    value = engine._eval(
        node,
        module=module,
        actor=actor,
        class_ref=class_ref,
        env=env,
        object_env=object_env,
        call_ordinals=call_ordinals,
    )
    return _structured_flow_value(value)


def _structured_flow_value(value: FlowValue) -> StructuredValue:
    items = (
        tuple(_structured_flow_value(item) for item in value.structured_items)
        if value.structured_items is not None
        else None
    )
    return StructuredValue(value, items)


def evaluate_iterable(
    engine: ValueEngine,
    node: ast.expr,
    *,
    module: str,
    actor: str,
    class_ref: str | None,
    env: dict[str, FlowValue],
    object_env: dict[str, set[str]],
    call_ordinals: Mapping[int, int],
    allow_sqlite_cursor_iteration: bool = True,
) -> IterableValue:
    if isinstance(node, (ast.Tuple, ast.List)):
        structured = evaluate_structured(
            engine,
            node,
            module=module,
            actor=actor,
            class_ref=class_ref,
            env=env,
            object_env=object_env,
            call_ordinals=call_ordinals,
        )
        if structured.items is not None:
            return IterableValue(
                merge_structured(structured.items),
                structured.value,
                True,
                len(structured.items),
            )
        return IterableValue(
            StructuredValue(structured.value.copy()), structured.value, False, None
        )
    if isinstance(node, ast.Set):
        items = _evaluate_many(
            engine,
            node.elts,
            module=module,
            actor=actor,
            class_ref=class_ref,
            env=env,
            object_env=object_env,
            call_ordinals=call_ordinals,
        )
        return IterableValue(
            merge_structured(items), _merge_values(items), True, len(items)
        )
    if isinstance(node, ast.Dict):
        keys = _evaluate_many(
            engine,
            [key for key in node.keys if key is not None],
            module=module,
            actor=actor,
            class_ref=class_ref,
            env=env,
            object_env=object_env,
            call_ordinals=call_ordinals,
        )
        values = _evaluate_many(
            engine,
            node.values,
            module=module,
            actor=actor,
            class_ref=class_ref,
            env=env,
            object_env=object_env,
            call_ordinals=call_ordinals,
        )
        literal = all(key is not None for key in node.keys)
        return IterableValue(
            merge_structured(keys),
            _merge_values([*keys, *values]),
            literal,
            len(keys) if literal else None,
        )
    structured = evaluate_structured(
        engine,
        node,
        module=module,
        actor=actor,
        class_ref=class_ref,
        env=env,
        object_env=object_env,
        call_ordinals=call_ordinals,
    )
    if (
        allow_sqlite_cursor_iteration
        and sqlite_handle_kind(structured.value) == "cursor"
    ):
        return IterableValue(StructuredValue(FlowValue()), FlowValue(), False, None)
    return IterableValue(
        StructuredValue(structured.value.copy()), structured.value, False, None
    )


def merge_structured(values: Sequence[StructuredValue]) -> StructuredValue:
    aggregate = _merge_values(values)
    if not values or any(value.items is None for value in values):
        return StructuredValue(aggregate)
    item_groups = [value.items for value in values if value.items is not None]
    sizes = {len(items) for items in item_groups}
    if len(sizes) != 1:
        return StructuredValue(aggregate)
    size = sizes.pop()
    children = tuple(
        merge_structured([items[index] for items in item_groups])
        for index in range(size)
    )
    return StructuredValue(aggregate, children)


def bind_structured_target(
    engine: ValueEngine,
    target: ast.expr,
    value: StructuredValue,
    *,
    module: str,
    actor: str,
    class_ref: str | None,
    env: dict[str, FlowValue],
    object_env: dict[str, set[str]],
    call_ordinals: Mapping[int, int],
    inject_registered_origin: bool = False,
) -> bool:
    if isinstance(target, ast.Starred):
        return bind_structured_target(
            engine,
            target.value,
            value,
            module=module,
            actor=actor,
            class_ref=class_ref,
            env=env,
            object_env=object_env,
            call_ordinals=call_ordinals,
            inject_registered_origin=inject_registered_origin,
        )
    if not isinstance(target, (ast.Tuple, ast.List)):
        bound_value = value.value
        if inject_registered_origin and isinstance(target, ast.Name):
            bound_value = engine._assignment_binding_value(
                target.id,
                bound_value,
                module=module,
                actor=actor,
            )
        return engine._bind_target(
            target,
            bound_value,
            actor=actor,
            class_ref=class_ref,
            env=env,
            object_env=object_env,
            module=module,
            source_node=target,
            ordinal=int(call_ordinals.get(id(target), 0)),
        )
    assignments = _unpack_assignments(target.elts, value)
    if assignments is None:
        _record_unknown_unpack(
            engine,
            target,
            value.value,
            module=module,
            actor=actor,
            call_ordinals=call_ordinals,
        )
        assignments = [
            (leaf, StructuredValue(value.value.copy()))
            for leaf in _target_leaves(target)
        ]
    changed = False
    for child_target, child_value in assignments:
        changed |= bind_structured_target(
            engine,
            child_target,
            child_value,
            module=module,
            actor=actor,
            class_ref=class_ref,
            env=env,
            object_env=object_env,
            call_ordinals=call_ordinals,
            inject_registered_origin=inject_registered_origin,
        )
    return changed


def structured_target_mismatch(target: ast.expr, value: StructuredValue) -> bool:
    """Return whether an exact structured value cannot bind the target shape."""

    if isinstance(target, ast.Starred):
        return structured_target_mismatch(target.value, value)
    if not isinstance(target, (ast.Tuple, ast.List)):
        return False
    assignments = _unpack_assignments(target.elts, value)
    if assignments is None:
        return value.items is not None
    return any(
        structured_target_mismatch(child_target, child_value)
        for child_target, child_value in assignments
    )


def _unpack_assignments(
    targets: list[ast.expr], value: StructuredValue
) -> list[tuple[ast.expr, StructuredValue]] | None:
    if value.items is None:
        return None
    starred = [
        index for index, target in enumerate(targets) if isinstance(target, ast.Starred)
    ]
    if not starred:
        if len(targets) != len(value.items):
            return None
        return list(zip(targets, value.items, strict=True))
    if len(starred) != 1 or len(value.items) < len(targets) - 1:
        return None
    star_index = starred[0]
    suffix_count = len(targets) - star_index - 1
    assignments = list(zip(targets[:star_index], value.items[:star_index], strict=True))
    remainder_end = len(value.items) - suffix_count
    assignments.append(
        (
            targets[star_index],
            merge_structured(value.items[star_index:remainder_end]),
        )
    )
    if suffix_count:
        assignments.extend(
            zip(
                targets[star_index + 1 :],
                value.items[remainder_end:],
                strict=True,
            )
        )
    return assignments


def _target_leaves(target: ast.expr) -> list[ast.expr]:
    if isinstance(target, ast.Starred):
        return _target_leaves(target.value)
    if isinstance(target, (ast.Tuple, ast.List)):
        return [leaf for child in target.elts for leaf in _target_leaves(child)]
    return [target]


def _evaluate_many(
    engine: ValueEngine,
    nodes: Sequence[ast.expr],
    *,
    module: str,
    actor: str,
    class_ref: str | None,
    env: dict[str, FlowValue],
    object_env: dict[str, set[str]],
    call_ordinals: Mapping[int, int],
) -> tuple[StructuredValue, ...]:
    return tuple(
        evaluate_structured(
            engine,
            node,
            module=module,
            actor=actor,
            class_ref=class_ref,
            env=env,
            object_env=object_env,
            call_ordinals=call_ordinals,
        )
        for node in nodes
    )


def _merge_values(values: Sequence[StructuredValue]) -> FlowValue:
    result = FlowValue()
    for value in values:
        result = result.merged(value.value)
    return result


def _record_unknown_unpack(
    engine: ValueEngine,
    target: ast.expr,
    value: FlowValue,
    *,
    module: str,
    actor: str,
    call_ordinals: Mapping[int, int],
) -> None:
    if value.has_origins:
        engine.facts.record_escape(
            value,
            node=target,
            actor=actor,
            operation="destructure",
            sink="python.iterable-unpack",
            reason="unsupported_registered_origin_destructuring",
            path=engine.paths[module],
            line=int(target.lineno),
            ordinal=int(call_ordinals.get(id(target), 0)),
        )


def _evaluate_definition_use(
    engine: ValueEngine,
    node: ast.expr,
    *,
    kind: str,
    module: str,
    actor: str,
    class_ref: str | None,
    env: dict[str, FlowValue],
    object_env: dict[str, set[str]],
    call_ordinals: Mapping[int, int],
) -> None:
    value = engine._eval(
        node,
        module=module,
        actor=actor,
        class_ref=class_ref,
        env=env,
        object_env=object_env,
        call_ordinals=call_ordinals,
    )
    if value.has_origins:
        engine.facts.record_escape(
            value,
            node=node,
            actor=actor,
            operation=f"definition:{kind}",
            sink=f"python.{kind}",
            reason="unsupported_registered_origin_definition",
            path=engine.paths[module],
            line=int(node.lineno),
            ordinal=int(call_ordinals.get(id(node), 0)),
        )


def _function_defaults(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> list[tuple[str, ast.expr]]:
    positional = [*node.args.posonlyargs, *node.args.args]
    defaults: list[tuple[str, ast.expr]] = []
    if node.args.defaults:
        defaults.extend(
            (argument.arg, default)
            for argument, default in zip(
                positional[-len(node.args.defaults) :],
                node.args.defaults,
                strict=True,
            )
        )
    defaults.extend(
        (argument.arg, default)
        for argument, default in zip(
            node.args.kwonlyargs,
            node.args.kw_defaults,
            strict=True,
        )
        if default is not None
    )
    return defaults


__all__ = [
    "AssignmentResult",
    "DefinitionEngine",
    "IterableValue",
    "StructuredValue",
    "ValueEngine",
    "analyze_augassign",
    "analyze_assignment",
    "analyze_definition",
    "bind_structured_target",
    "evaluate_iterable",
    "evaluate_structured",
    "merge_structured",
    "structured_target_mismatch",
]
