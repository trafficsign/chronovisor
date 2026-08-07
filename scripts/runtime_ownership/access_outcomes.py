"""Explicit control outcomes for runtime-access flow analysis."""

from __future__ import annotations

import ast
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

from .access_model import (
    FCNTL_UNRESOLVED_LOCK_OPERATION_OBJECT_TYPE,
    UNRESOLVED_RUNTIME_OBJECT_ALTERNATIVE_TYPE,
    FlowValue,
    has_fcntl_lock_mask,
    has_file_descriptor_object,
    is_exact_path_receiver,
    mark_attribute_alternative_ambiguity,
)

if TYPE_CHECKING:
    from .access_statements import StatementEngine

ControlKind = Literal["normal", "return", "break", "continue", "raise"]
StateSnapshot = tuple[dict[str, FlowValue], dict[str, set[str]]]


@dataclass
class Outcome:
    kind: ControlKind
    env: dict[str, FlowValue]
    object_env: dict[str, set[str]]
    value: FlowValue

    @classmethod
    def normal(
        cls,
        env: Mapping[str, FlowValue],
        object_env: Mapping[str, set[str]],
    ) -> Outcome:
        copied_env, copied_objects = copy_state(env, object_env)
        return cls("normal", copied_env, copied_objects, FlowValue())


@dataclass
class BlockResult:
    changed: bool
    outcomes: list[Outcome]

    @property
    def returned(self) -> FlowValue:
        value = FlowValue()
        return_values: list[FlowValue] = []
        for outcome in self.outcomes:
            if outcome.kind == "return":
                return_values.append(outcome.value)
                value = value.merged(outcome.value)
        mark_attribute_alternative_ambiguity(value, return_values)
        if any(outcome.kind == "normal" for outcome in self.outcomes):
            mark_attribute_alternative_ambiguity(
                value,
                [*return_values, FlowValue()],
            )
        if has_fcntl_lock_mask(value) and (
            any(candidate == FlowValue() for candidate in return_values)
            or any(outcome.kind == "normal" for outcome in self.outcomes)
        ):
            value.object_types.add(
                FCNTL_UNRESOLVED_LOCK_OPERATION_OBJECT_TYPE
            )
        if has_file_descriptor_object(value) and (
            any(candidate == FlowValue() for candidate in return_values)
            or any(outcome.kind == "normal" for outcome in self.outcomes)
        ):
            value.object_types.add(
                UNRESOLVED_RUNTIME_OBJECT_ALTERNATIVE_TYPE
            )
        if is_exact_path_receiver(value) and (
            any(candidate == FlowValue() for candidate in return_values)
            or any(outcome.kind == "normal" for outcome in self.outcomes)
        ):
            value.object_types.add(
                UNRESOLVED_RUNTIME_OBJECT_ALTERNATIVE_TYPE
            )
        return value


def copy_state(
    env: Mapping[str, FlowValue], object_env: Mapping[str, set[str]]
) -> StateSnapshot:
    return (
        {name: value.copy() for name, value in env.items()},
        {name: set(values) for name, values in object_env.items()},
    )


def copy_outcome(outcome: Outcome, *, kind: ControlKind | None = None) -> Outcome:
    env, object_env = copy_state(outcome.env, outcome.object_env)
    return Outcome(kind or outcome.kind, env, object_env, outcome.value.copy())


def join_states(states: Iterable[StateSnapshot]) -> StateSnapshot:
    materialized = list(states)
    if not materialized:
        return {}, {}
    joined_env: dict[str, FlowValue] = {}
    joined_objects: dict[str, set[str]] = {}
    names = set().union(*(state_env for state_env, _objects in materialized))
    for name in names:
        value = FlowValue()
        candidates: list[FlowValue] = []
        for state_env, _objects in materialized:
            candidate = state_env.get(name, FlowValue())
            candidates.append(candidate)
            value = value.merged(candidate)
        mark_attribute_alternative_ambiguity(value, candidates)
        if has_fcntl_lock_mask(value) and any(
            candidate == FlowValue() for candidate in candidates
        ):
            value.object_types.add(
                FCNTL_UNRESOLVED_LOCK_OPERATION_OBJECT_TYPE
            )
        if has_file_descriptor_object(value) and any(
            candidate == FlowValue() for candidate in candidates
        ):
            value.object_types.add(
                UNRESOLVED_RUNTIME_OBJECT_ALTERNATIVE_TYPE
            )
        if is_exact_path_receiver(value) and any(
            candidate == FlowValue() for candidate in candidates
        ):
            value.object_types.add(
                UNRESOLVED_RUNTIME_OBJECT_ALTERNATIVE_TYPE
            )
        joined_env[name] = value
    object_names = set().union(
        *(state_objects for _state_env, state_objects in materialized)
    )
    for name in object_names:
        joined_objects[name] = set().union(
            *(objects.get(name, set()) for _state_env, objects in materialized)
        )
    return joined_env, joined_objects


def normalize_outcomes(outcomes: Iterable[Outcome]) -> list[Outcome]:
    grouped: dict[ControlKind, list[Outcome]] = {}
    order: list[ControlKind] = []
    for outcome in outcomes:
        if outcome.kind not in grouped:
            grouped[outcome.kind] = []
            order.append(outcome.kind)
        grouped[outcome.kind].append(outcome)
    normalized: list[Outcome] = []
    for kind in order:
        members = grouped[kind]
        env, object_env = join_states(
            (member.env, member.object_env) for member in members
        )
        value = FlowValue()
        for member in members:
            value = value.merged(member.value)
        normalized.append(Outcome(kind, env, object_env, value))
    return normalized


def sync_normal_state(
    env: dict[str, FlowValue],
    object_env: dict[str, set[str]],
    outcomes: Iterable[Outcome],
) -> None:
    normal_states = [
        (outcome.env, outcome.object_env)
        for outcome in outcomes
        if outcome.kind == "normal"
    ]
    joined_env, joined_objects = join_states(normal_states)
    env.clear()
    env.update(joined_env)
    object_env.clear()
    object_env.update(joined_objects)


def analyze_block_result(
    engine: StatementEngine,
    statements: Iterable[ast.stmt],
    *,
    module: str,
    actor: str,
    class_ref: str | None,
    env: dict[str, FlowValue],
    object_env: dict[str, set[str]],
    call_ordinals: Mapping[int, int],
    prefix_states: list[StateSnapshot] | None = None,
) -> BlockResult:
    changed = False
    outcomes = [Outcome.normal(env, object_env)]
    nested_refs: set[str] = set()
    for statement in statements:
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            nested_ref = engine.function_refs_by_node.get(id(statement))
            if nested_ref is not None:
                nested_refs.add(nested_ref)
        next_outcomes: list[Outcome] = []
        for outcome in outcomes:
            if outcome.kind != "normal":
                next_outcomes.append(outcome)
                continue
            if prefix_states is not None and _statement_may_raise(
                statement,
                evaluate_annotations=module not in engine.future_annotations,
            ):
                prefix_states.append(copy_state(outcome.env, outcome.object_env))
            result = _analyze_outcome_statement(
                engine,
                statement,
                module=module,
                actor=actor,
                class_ref=class_ref,
                env=outcome.env,
                object_env=outcome.object_env,
                call_ordinals=call_ordinals,
                prefix_states=prefix_states,
            )
            changed |= result.changed
            next_outcomes.extend(result.outcomes)
        outcomes = normalize_outcomes(next_outcomes)
    for nested_ref in nested_refs:
        if outcomes:
            capture_env, _capture_objects = join_states(
                (outcome.env, outcome.object_env) for outcome in outcomes
            )
            changed |= engine._merge_closure(
                nested_ref,
                actor=actor,
                module=module,
                env=capture_env,
            )
    return BlockResult(changed, outcomes)


def _analyze_outcome_statement(
    engine: StatementEngine,
    statement: ast.stmt,
    *,
    module: str,
    actor: str,
    class_ref: str | None,
    env: dict[str, FlowValue],
    object_env: dict[str, set[str]],
    call_ordinals: Mapping[int, int],
    prefix_states: list[StateSnapshot] | None,
) -> BlockResult:
    from .access_statements import (
        _analyze_legacy_block,
        evaluate_control_expression,
    )

    if isinstance(statement, (ast.Assign, ast.AnnAssign)):
        from .access_bindings import analyze_assignment

        assignment = analyze_assignment(
            engine,
            statement,
            module=module,
            actor=actor,
            class_ref=class_ref,
            env=env,
            object_env=object_env,
            call_ordinals=call_ordinals,
        )
        kind: ControlKind = "raise" if assignment.mismatch else "normal"
        return BlockResult(
            assignment.changed,
            [Outcome(kind, env, object_env, FlowValue())],
        )
    if isinstance(statement, ast.AugAssign):
        from .access_bindings import analyze_augassign

        changed = analyze_augassign(
            engine,
            statement,
            module=module,
            actor=actor,
            class_ref=class_ref,
            env=env,
            object_env=object_env,
            call_ordinals=call_ordinals,
        )
        return BlockResult(changed, [Outcome("normal", env, object_env, FlowValue())])
    if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
        from .access_definition_execution import (
            DefinitionExecutionEngine,
            analyze_function_definition,
        )

        changed = analyze_function_definition(
            cast(DefinitionExecutionEngine, engine),
            statement,
            module=module,
            actor=actor,
            class_ref=class_ref,
            env=env,
            object_env=object_env,
            call_ordinals=call_ordinals,
        )
        return BlockResult(changed, [Outcome("normal", env, object_env, FlowValue())])
    if isinstance(statement, ast.ClassDef):
        from .access_class_scopes import analyze_class_definition

        return analyze_class_definition(
            engine,
            statement,
            module=module,
            actor=actor,
            class_ref=class_ref,
            env=env,
            object_env=object_env,
            call_ordinals=call_ordinals,
        )
    if isinstance(statement, ast.Return):
        value = FlowValue()
        if statement.value is not None:
            value = engine._eval(
                statement.value,
                module=module,
                actor=actor,
                class_ref=class_ref,
                env=env,
                object_env=object_env,
                call_ordinals=call_ordinals,
            ).bound(f"return:{actor}")
        return BlockResult(False, [Outcome("return", env, object_env, value)])
    if isinstance(statement, ast.Raise):
        value = FlowValue()
        for kind, expression in (
            ("raise", statement.exc),
            ("raise_cause", statement.cause),
        ):
            if expression is not None:
                value = value.merged(
                    evaluate_control_expression(
                        engine,
                        expression,
                        kind=kind,
                        module=module,
                        actor=actor,
                        class_ref=class_ref,
                        env=env,
                        object_env=object_env,
                        call_ordinals=call_ordinals,
                    )
                )
        return BlockResult(False, [Outcome("raise", env, object_env, value)])
    if isinstance(statement, ast.Break):
        return BlockResult(False, [Outcome("break", env, object_env, FlowValue())])
    if isinstance(statement, ast.Continue):
        return BlockResult(False, [Outcome("continue", env, object_env, FlowValue())])
    if isinstance(statement, ast.Delete):
        for target in statement.targets:
            _bind_deleted_attribute_targets(
                engine,
                target,
                actor=actor,
                class_ref=class_ref,
                env=env,
                object_env=object_env,
            )
            _delete_name_target(target, env=env, object_env=object_env)
        return BlockResult(False, [Outcome("normal", env, object_env, FlowValue())])
    if isinstance(statement, ast.If):
        evaluate_control_expression(
            engine,
            statement.test,
            kind="if",
            module=module,
            actor=actor,
            class_ref=class_ref,
            env=env,
            object_env=object_env,
            call_ordinals=call_ordinals,
        )
        branch_outcomes: list[Outcome] = []
        changed = False
        for branch in (statement.body, statement.orelse):
            branch_env, branch_objects = copy_state(env, object_env)
            result = analyze_block_result(
                engine,
                branch,
                module=module,
                actor=actor,
                class_ref=class_ref,
                env=branch_env,
                object_env=branch_objects,
                call_ordinals=call_ordinals,
                prefix_states=prefix_states,
            )
            changed |= result.changed
            branch_outcomes.extend(result.outcomes)
        return BlockResult(changed, normalize_outcomes(branch_outcomes))
    if isinstance(statement, (ast.For, ast.AsyncFor, ast.While)):
        from .access_outcome_control import analyze_loop

        return analyze_loop(
            engine,
            statement,
            module=module,
            actor=actor,
            class_ref=class_ref,
            env=env,
            object_env=object_env,
            call_ordinals=call_ordinals,
            prefix_states=prefix_states,
        )
    if isinstance(statement, (ast.Try, ast.TryStar)):
        from .access_outcome_control import analyze_try

        return analyze_try(
            engine,
            statement,
            module=module,
            actor=actor,
            class_ref=class_ref,
            env=env,
            object_env=object_env,
            call_ordinals=call_ordinals,
            prefix_states=prefix_states,
        )
    if isinstance(statement, (ast.With, ast.AsyncWith)):
        from .access_outcome_control import analyze_with

        return analyze_with(
            engine,
            statement,
            module=module,
            actor=actor,
            class_ref=class_ref,
            env=env,
            object_env=object_env,
            call_ordinals=call_ordinals,
            prefix_states=prefix_states,
        )
    if isinstance(statement, ast.Match):
        from .access_outcome_control import analyze_match

        return analyze_match(
            engine,
            statement,
            module=module,
            actor=actor,
            class_ref=class_ref,
            env=env,
            object_env=object_env,
            call_ordinals=call_ordinals,
            prefix_states=prefix_states,
        )
    changed, returned = _analyze_legacy_block(
        engine,
        [statement],
        module=module,
        actor=actor,
        class_ref=class_ref,
        env=env,
        object_env=object_env,
        call_ordinals=call_ordinals,
        prefix_states=None,
    )
    outcomes = [Outcome("normal", env, object_env, FlowValue())]
    if returned.has_analysis_state:
        returned_env, returned_objects = copy_state(env, object_env)
        outcomes.append(Outcome("return", returned_env, returned_objects, returned))
    return BlockResult(changed, outcomes)


def _statement_may_raise(
    statement: ast.stmt,
    *,
    evaluate_annotations: bool,
) -> bool:
    if isinstance(
        statement,
        (
            ast.Import,
            ast.ImportFrom,
            ast.Assign,
            ast.AnnAssign,
            ast.AugAssign,
            ast.Expr,
            ast.Delete,
            ast.Assert,
            ast.If,
            ast.For,
            ast.AsyncFor,
            ast.While,
            ast.With,
            ast.AsyncWith,
            ast.Match,
        ),
    ):
        return True
    if isinstance(statement, ast.Return):
        return statement.value is not None
    if isinstance(statement, ast.Raise):
        return statement.exc is not None or statement.cause is not None
    if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
        arguments = statement.args
        annotated_arguments = [
            *arguments.posonlyargs,
            *arguments.args,
            *arguments.kwonlyargs,
        ]
        if arguments.vararg is not None:
            annotated_arguments.append(arguments.vararg)
        if arguments.kwarg is not None:
            annotated_arguments.append(arguments.kwarg)
        annotations_may_raise = evaluate_annotations and (
            statement.returns is not None
            or any(argument.annotation is not None for argument in annotated_arguments)
        )
        return bool(
            statement.decorator_list
            or arguments.defaults
            or any(default is not None for default in arguments.kw_defaults)
            or annotations_may_raise
        )
    return isinstance(statement, ast.ClassDef)


def _delete_name_target(
    target: ast.expr,
    *,
    env: dict[str, FlowValue],
    object_env: dict[str, set[str]],
) -> None:
    if isinstance(target, ast.Name):
        env[target.id] = FlowValue()
        object_env.pop(target.id, None)
    elif isinstance(target, (ast.Tuple, ast.List)):
        for child in target.elts:
            _delete_name_target(child, env=env, object_env=object_env)


def _bind_deleted_attribute_targets(
    engine: StatementEngine,
    target: ast.expr,
    *,
    actor: str,
    class_ref: str | None,
    env: dict[str, FlowValue],
    object_env: dict[str, set[str]],
) -> None:
    if isinstance(target, (ast.Attribute, ast.Subscript)):
        engine._bind_target(
            target,
            FlowValue(),
            actor=actor,
            class_ref=class_ref,
            env=env,
            object_env=object_env,
        )
    elif isinstance(target, (ast.Tuple, ast.List)):
        for child in target.elts:
            _bind_deleted_attribute_targets(
                engine,
                child,
                actor=actor,
                class_ref=class_ref,
                env=env,
                object_env=object_env,
            )


__all__ = [
    "BlockResult",
    "ControlKind",
    "Outcome",
    "StateSnapshot",
    "analyze_block_result",
    "copy_outcome",
    "copy_state",
    "join_states",
    "normalize_outcomes",
    "sync_normal_state",
]
