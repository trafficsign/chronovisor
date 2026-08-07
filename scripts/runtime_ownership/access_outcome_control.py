"""Outcome-aware loop and exception control-flow analysis."""

from __future__ import annotations

import ast
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Protocol, cast

from .access_bindings import (
    IterableValue,
    bind_structured_target,
    evaluate_iterable,
    structured_target_mismatch,
)
from .access_control import match_values
from .access_model import (
    FlowValue,
    file_handle_kind,
    sqlite_handle_kind,
)
from .access_outcomes import (
    BlockResult,
    Outcome,
    StateSnapshot,
    analyze_block_result,
    copy_outcome,
    copy_state,
    join_states,
    normalize_outcomes,
)

if TYPE_CHECKING:
    from .access_statements import StatementEngine


class _RuntimeContaminationEngine(Protocol):
    def _contaminate_runtime_objects(
        self,
        env: dict[str, FlowValue],
        object_env: dict[str, set[str]],
        values: Sequence[FlowValue],
    ) -> None: ...


def analyze_loop(
    engine: StatementEngine,
    statement: ast.For | ast.AsyncFor | ast.While,
    *,
    module: str,
    actor: str,
    class_ref: str | None,
    env: dict[str, FlowValue],
    object_env: dict[str, set[str]],
    call_ordinals: Mapping[int, int],
    prefix_states: list[StateSnapshot] | None,
) -> BlockResult:
    """Compute a loop fixed point without conflating exit control kinds."""

    entry_env, entry_objects = copy_state(env, object_env)
    changed = False
    iterable: IterableValue | None = None
    if isinstance(statement, (ast.For, ast.AsyncFor)):
        _record_snapshot(prefix_states, entry_env, entry_objects)
        iterable = evaluate_iterable(
            engine,
            statement.iter,
            module=module,
            actor=actor,
            class_ref=class_ref,
            env=entry_env,
            object_env=entry_objects,
            call_ordinals=call_ordinals,
            allow_sqlite_cursor_iteration=not isinstance(statement, ast.AsyncFor),
        )
        if not iterable.literal:
            _record_control_value(
                engine,
                iterable.aggregate,
                statement.iter,
                kind="async_for" if isinstance(statement, ast.AsyncFor) else "for",
                module=module,
                actor=actor,
                call_ordinals=call_ordinals,
            )
        if (
            isinstance(statement, ast.AsyncFor)
            and sqlite_handle_kind(iterable.aggregate) == "cursor"
        ):
            return BlockResult(
                changed,
                [Outcome("raise", entry_env, entry_objects, FlowValue())],
            )
        if iterable.literal_length == 0:
            exhausted = Outcome.normal(entry_env, entry_objects)
            if not statement.orelse:
                return BlockResult(changed, [exhausted])
            else_result = analyze_block_result(
                engine,
                statement.orelse,
                module=module,
                actor=actor,
                class_ref=class_ref,
                env=exhausted.env,
                object_env=exhausted.object_env,
                call_ordinals=call_ordinals,
                prefix_states=prefix_states,
            )
            return BlockResult(changed or else_result.changed, else_result.outcomes)
        if structured_target_mismatch(statement.target, iterable.item):
            return BlockResult(
                changed,
                [Outcome("raise", entry_env, entry_objects, FlowValue())],
            )

    head_env, head_objects = copy_state(entry_env, entry_objects)
    stable_result = BlockResult(False, [])
    exhaustion_env, exhaustion_objects = copy_state(head_env, head_objects)
    while True:
        body_env, body_objects = copy_state(head_env, head_objects)
        if isinstance(statement, ast.While):
            _record_snapshot(prefix_states, body_env, body_objects)
            _evaluate_control(
                engine,
                statement.test,
                kind="while",
                module=module,
                actor=actor,
                class_ref=class_ref,
                env=body_env,
                object_env=body_objects,
                call_ordinals=call_ordinals,
            )
        exhaustion_env, exhaustion_objects = copy_state(body_env, body_objects)
        if isinstance(statement, (ast.For, ast.AsyncFor)):
            if iterable is None:
                raise AssertionError("for loop iterable must be evaluated")
            changed |= bind_structured_target(
                engine,
                statement.target,
                iterable.item,
                module=module,
                actor=actor,
                class_ref=class_ref,
                env=body_env,
                object_env=body_objects,
                call_ordinals=call_ordinals,
            )
        body_result = analyze_block_result(
            engine,
            statement.body,
            module=module,
            actor=actor,
            class_ref=class_ref,
            env=body_env,
            object_env=body_objects,
            call_ordinals=call_ordinals,
            prefix_states=prefix_states,
        )
        changed |= body_result.changed
        backedges = [
            (outcome.env, outcome.object_env)
            for outcome in body_result.outcomes
            if outcome.kind in {"normal", "continue"}
        ]
        next_env, next_objects = join_states([(entry_env, entry_objects), *backedges])
        if next_env == head_env and next_objects == head_objects:
            stable_result = body_result
            break
        head_env, head_objects = next_env, next_objects

    outcomes: list[Outcome] = []
    body_can_exhaust = any(
        outcome.kind in {"normal", "continue"} for outcome in stable_result.outcomes
    )
    can_exhaust = iterable is None or not iterable.literal or body_can_exhaust
    if can_exhaust:
        exhausted = Outcome.normal(exhaustion_env, exhaustion_objects)
        if statement.orelse:
            else_result = analyze_block_result(
                engine,
                statement.orelse,
                module=module,
                actor=actor,
                class_ref=class_ref,
                env=exhausted.env,
                object_env=exhausted.object_env,
                call_ordinals=call_ordinals,
                prefix_states=prefix_states,
            )
            changed |= else_result.changed
            outcomes.extend(else_result.outcomes)
        else:
            outcomes.append(exhausted)
    for outcome in stable_result.outcomes:
        if outcome.kind == "break":
            outcomes.append(copy_outcome(outcome, kind="normal"))
        elif outcome.kind in {"return", "raise"}:
            outcomes.append(copy_outcome(outcome))
    return BlockResult(changed, normalize_outcomes(outcomes))


def analyze_try(
    engine: StatementEngine,
    statement: ast.Try | ast.TryStar,
    *,
    module: str,
    actor: str,
    class_ref: str | None,
    env: dict[str, FlowValue],
    object_env: dict[str, set[str]],
    call_ordinals: Mapping[int, int],
    prefix_states: list[StateSnapshot] | None,
) -> BlockResult:
    """Keep try outcomes distinct and run finally for every outcome kind."""

    body_env, body_objects = copy_state(env, object_env)
    exception_prefixes: list[StateSnapshot] = []
    body_result = analyze_block_result(
        engine,
        statement.body,
        module=module,
        actor=actor,
        class_ref=class_ref,
        env=body_env,
        object_env=body_objects,
        call_ordinals=call_ordinals,
        prefix_states=exception_prefixes,
    )
    if prefix_states is not None:
        prefix_states.extend(
            copy_state(prefix_env, prefix_objects)
            for prefix_env, prefix_objects in exception_prefixes
        )

    changed = body_result.changed
    outcomes: list[Outcome] = []
    raised = [outcome for outcome in body_result.outcomes if outcome.kind == "raise"]
    for outcome in body_result.outcomes:
        if outcome.kind == "normal" and statement.orelse:
            else_result = analyze_block_result(
                engine,
                statement.orelse,
                module=module,
                actor=actor,
                class_ref=class_ref,
                env=outcome.env,
                object_env=outcome.object_env,
                call_ordinals=call_ordinals,
                prefix_states=prefix_states,
            )
            changed |= else_result.changed
            outcomes.extend(else_result.outcomes)
        else:
            outcomes.append(copy_outcome(outcome))

    exception_states = [*exception_prefixes]
    exception_states.extend((outcome.env, outcome.object_env) for outcome in raised)
    if exception_states:
        handler_env, handler_objects = join_states(exception_states)
        for handler in statement.handlers:
            local_env, local_objects = copy_state(handler_env, handler_objects)
            if handler.type is not None:
                _record_snapshot(prefix_states, local_env, local_objects)
                _evaluate_control(
                    engine,
                    handler.type,
                    kind="except_type",
                    module=module,
                    actor=actor,
                    class_ref=class_ref,
                    env=local_env,
                    object_env=local_objects,
                    call_ordinals=call_ordinals,
                )
            if handler.name is not None:
                engine._bind_target(
                    ast.Name(id=handler.name, ctx=ast.Store()),
                    FlowValue(),
                    actor=actor,
                    class_ref=class_ref,
                    env=local_env,
                    object_env=local_objects,
                )
            handler_result = analyze_block_result(
                engine,
                handler.body,
                module=module,
                actor=actor,
                class_ref=class_ref,
                env=local_env,
                object_env=local_objects,
                call_ordinals=call_ordinals,
                prefix_states=prefix_states,
            )
            changed |= handler_result.changed
            outcomes.extend(handler_result.outcomes)
        if exception_prefixes:
            implicit_env, implicit_objects = join_states(exception_prefixes)
            outcomes.append(
                Outcome("raise", implicit_env, implicit_objects, FlowValue())
            )

    outcomes = normalize_outcomes(outcomes)
    if statement.finalbody:
        final_result = _apply_finally(
            engine,
            outcomes,
            statement.finalbody,
            module=module,
            actor=actor,
            class_ref=class_ref,
            call_ordinals=call_ordinals,
            prefix_states=prefix_states,
        )
        changed |= final_result.changed
        outcomes = final_result.outcomes
    return BlockResult(changed, outcomes)


def analyze_with(
    engine: StatementEngine,
    statement: ast.With | ast.AsyncWith,
    *,
    module: str,
    actor: str,
    class_ref: str | None,
    env: dict[str, FlowValue],
    object_env: dict[str, set[str]],
    call_ordinals: Mapping[int, int],
    prefix_states: list[StateSnapshot] | None,
) -> BlockResult:
    """Preserve body outcomes while modeling unknown exception suppression."""

    changed = False
    kind = "async_with" if isinstance(statement, ast.AsyncWith) else "with"
    managers: list[tuple[str | None, FlowValue]] = []
    for item in statement.items:
        _record_snapshot(prefix_states, env, object_env)
        context_value = engine._eval(
            item.context_expr,
            module=module,
            actor=actor,
            class_ref=class_ref,
            env=env,
            object_env=object_env,
            call_ordinals=call_ordinals,
        )
        context_manager_kind: str | None = None
        if sqlite_handle_kind(context_value) == "connection":
            context_manager_kind = "sqlite"
        elif file_handle_kind(context_value) == "file":
            context_manager_kind = "file"
        if isinstance(statement, ast.AsyncWith) and context_manager_kind is not None:
            _record_control_value(
                engine,
                context_value,
                item.context_expr,
                kind=kind,
                module=module,
                actor=actor,
                call_ordinals=call_ordinals,
            )
            failure_env, failure_objects = copy_state(env, object_env)
            outcomes = _exit_context_managers(
                engine,
                managers,
                [
                    Outcome(
                        "raise",
                        failure_env,
                        failure_objects,
                        context_value.copy(),
                    )
                ],
                statement=statement,
                kind=kind,
                module=module,
                actor=actor,
                prefix_states=prefix_states,
            )
            return BlockResult(changed, outcomes)
        if context_manager_kind == "sqlite":
            entered_value = context_value.bound("context:sqlite.connection.enter")
        elif context_manager_kind == "file":
            entered_value = context_value.bound("context:file.handle.enter")
        else:
            entered_value = FlowValue()
            _record_control_value(
                engine,
                context_value,
                item.context_expr,
                kind=kind,
                module=module,
                actor=actor,
                call_ordinals=call_ordinals,
            )
        managers.append(
            (
                context_manager_kind,
                entered_value
                if context_manager_kind is not None
                else context_value,
            )
        )
        if item.optional_vars is not None:
            changed |= engine._bind_target(
                item.optional_vars,
                entered_value,
                actor=actor,
                class_ref=class_ref,
                env=env,
                object_env=object_env,
            )

    body_result = analyze_block_result(
        engine,
        statement.body,
        module=module,
        actor=actor,
        class_ref=class_ref,
        env=env,
        object_env=object_env,
        call_ordinals=call_ordinals,
        prefix_states=prefix_states,
    )
    changed |= body_result.changed
    outcomes = [copy_outcome(outcome) for outcome in body_result.outcomes]
    outcomes = _exit_context_managers(
        engine,
        managers,
        outcomes,
        statement=statement,
        kind=kind,
        module=module,
        actor=actor,
        prefix_states=prefix_states,
    )
    return BlockResult(changed, outcomes)


def _exit_context_managers(
    engine: StatementEngine,
    managers: list[tuple[str | None, FlowValue]],
    outcomes: list[Outcome],
    *,
    statement: ast.With | ast.AsyncWith,
    kind: str,
    module: str,
    actor: str,
    prefix_states: list[StateSnapshot] | None,
) -> list[Outcome]:
    for manager_kind, context_value in reversed(managers):
        exited: list[Outcome] = []
        for outcome in outcomes:
            _record_snapshot(prefix_states, outcome.env, outcome.object_env)
            if manager_kind == "sqlite":
                operation = (
                    "sqlite.transaction.implicit_rollback"
                    if outcome.kind == "raise"
                    else "sqlite.transaction.implicit_commit"
                )
                _record_sqlite_context_exit(
                    engine,
                    context_value,
                    statement,
                    operation=operation,
                    module=module,
                    actor=actor,
                )
                exited.append(copy_outcome(outcome))
                continue
            if manager_kind == "file":
                closed = copy_outcome(outcome)
                cast(
                    _RuntimeContaminationEngine,
                    engine,
                )._contaminate_runtime_objects(
                    closed.env,
                    closed.object_env,
                    [context_value],
                )
                exited.append(closed)
                continue
            exited.append(copy_outcome(outcome))
            if outcome.kind != "raise":
                continue
            affected = context_value.merged(outcome.value)
            if affected.has_origins:
                engine.facts.record_escape(
                    affected,
                    node=statement,
                    actor=actor,
                    operation=f"control:{kind}_suppression",
                    sink=f"python.{kind}",
                    reason="unknown_context_manager_suppression",
                    path=engine.paths[module],
                    line=int(statement.lineno),
                    ordinal=0,
                )
            suppressed_env, suppressed_objects = copy_state(
                outcome.env, outcome.object_env
            )
            exited.append(
                Outcome("normal", suppressed_env, suppressed_objects, FlowValue())
            )
        outcomes = normalize_outcomes(exited)
    return outcomes


def _record_sqlite_context_exit(
    engine: StatementEngine,
    value: FlowValue,
    statement: ast.With | ast.AsyncWith,
    *,
    operation: str,
    module: str,
    actor: str,
) -> None:
    if not value.has_origins:
        return
    engine.facts.record_access(
        value,
        node=statement,
        actor=actor,
        mode="write",
        operation=operation,
        sink="sqlite3.Connection.__exit__",
        path=engine.paths[module],
        line=int(statement.lineno),
        ordinal=0,
    )


def analyze_match(
    engine: StatementEngine,
    statement: ast.Match,
    *,
    module: str,
    actor: str,
    class_ref: str | None,
    env: dict[str, FlowValue],
    object_env: dict[str, set[str]],
    call_ordinals: Mapping[int, int],
    prefix_states: list[StateSnapshot] | None,
) -> BlockResult:
    """Analyze match cases without normalizing terminating case outcomes."""

    _record_snapshot(prefix_states, env, object_env)
    _evaluate_control(
        engine,
        statement.subject,
        kind="match_subject",
        module=module,
        actor=actor,
        class_ref=class_ref,
        env=env,
        object_env=object_env,
        call_ordinals=call_ordinals,
    )
    subject_env, subject_objects = copy_state(env, object_env)
    changed = False
    outcomes: list[Outcome] = []
    unmatched = True
    for case in statement.cases:
        if not unmatched:
            break
        case_env, case_objects = copy_state(subject_env, subject_objects)
        _record_snapshot(prefix_states, case_env, case_objects)
        for pattern_value in match_values(case.pattern):
            _evaluate_control(
                engine,
                pattern_value,
                kind="match_pattern",
                module=module,
                actor=actor,
                class_ref=class_ref,
                env=case_env,
                object_env=case_objects,
                call_ordinals=call_ordinals,
            )
        for name in _pattern_bound_names(case.pattern):
            changed |= engine._bind_target(
                ast.Name(id=name, ctx=ast.Store()),
                FlowValue(),
                actor=actor,
                class_ref=class_ref,
                env=case_env,
                object_env=case_objects,
            )
        if case.guard is not None:
            _record_snapshot(prefix_states, case_env, case_objects)
            _evaluate_control(
                engine,
                case.guard,
                kind="match_guard",
                module=module,
                actor=actor,
                class_ref=class_ref,
                env=case_env,
                object_env=case_objects,
                call_ordinals=call_ordinals,
            )
        case_result = analyze_block_result(
            engine,
            case.body,
            module=module,
            actor=actor,
            class_ref=class_ref,
            env=case_env,
            object_env=case_objects,
            call_ordinals=call_ordinals,
            prefix_states=prefix_states,
        )
        changed |= case_result.changed
        outcomes.extend(case_result.outcomes)
        if case.guard is None and _pattern_is_irrefutable(case.pattern):
            unmatched = False
    if unmatched:
        outcomes.append(Outcome.normal(subject_env, subject_objects))
    return BlockResult(changed, normalize_outcomes(outcomes))


def _pattern_bound_names(pattern: ast.pattern) -> set[str]:
    names: set[str] = set()

    class PatternBindingVisitor(ast.NodeVisitor):
        def visit_MatchAs(self, item: ast.MatchAs) -> None:
            if item.name is not None:
                names.add(item.name)
            if item.pattern is not None:
                self.visit(item.pattern)

        def visit_MatchStar(self, item: ast.MatchStar) -> None:
            if item.name is not None:
                names.add(item.name)

        def visit_MatchMapping(self, item: ast.MatchMapping) -> None:
            if item.rest is not None:
                names.add(item.rest)
            self.generic_visit(item)

    PatternBindingVisitor().visit(pattern)
    return names


def _pattern_is_irrefutable(pattern: ast.pattern) -> bool:
    if isinstance(pattern, ast.MatchAs):
        return pattern.pattern is None or _pattern_is_irrefutable(pattern.pattern)
    if isinstance(pattern, ast.MatchOr):
        return any(_pattern_is_irrefutable(child) for child in pattern.patterns)
    return False


def _apply_finally(
    engine: StatementEngine,
    incoming: list[Outcome],
    statements: list[ast.stmt],
    *,
    module: str,
    actor: str,
    class_ref: str | None,
    call_ordinals: Mapping[int, int],
    prefix_states: list[StateSnapshot] | None,
) -> BlockResult:
    changed = False
    outcomes: list[Outcome] = []
    for prior in incoming:
        final_env, final_objects = copy_state(prior.env, prior.object_env)
        result = analyze_block_result(
            engine,
            statements,
            module=module,
            actor=actor,
            class_ref=class_ref,
            env=final_env,
            object_env=final_objects,
            call_ordinals=call_ordinals,
            prefix_states=prefix_states,
        )
        changed |= result.changed
        for final in result.outcomes:
            if final.kind == "normal":
                outcomes.append(
                    Outcome(
                        prior.kind,
                        final.env,
                        final.object_env,
                        prior.value.copy(),
                    )
                )
            else:
                outcomes.append(copy_outcome(final))
    return BlockResult(changed, normalize_outcomes(outcomes))


def _evaluate_control(
    engine: StatementEngine,
    node: ast.expr,
    *,
    kind: str,
    module: str,
    actor: str,
    class_ref: str | None,
    env: dict[str, FlowValue],
    object_env: dict[str, set[str]],
    call_ordinals: Mapping[int, int],
) -> FlowValue:
    from .access_statements import evaluate_control_expression

    return evaluate_control_expression(
        engine,
        node,
        kind=kind,
        module=module,
        actor=actor,
        class_ref=class_ref,
        env=env,
        object_env=object_env,
        call_ordinals=call_ordinals,
    )


def _record_control_value(
    engine: StatementEngine,
    value: FlowValue,
    node: ast.expr,
    *,
    kind: str,
    module: str,
    actor: str,
    call_ordinals: Mapping[int, int],
) -> None:
    if value.has_origins:
        engine.facts.record_escape(
            value,
            node=node,
            actor=actor,
            operation=f"control:{kind}",
            sink=f"python.{kind}",
            reason="unsupported_registered_origin_control_flow",
            path=engine.paths[module],
            line=int(node.lineno),
            ordinal=int(call_ordinals.get(id(node), 0)),
        )


def _record_snapshot(
    snapshots: list[StateSnapshot] | None,
    env: dict[str, FlowValue],
    object_env: dict[str, set[str]],
) -> None:
    if snapshots is not None:
        snapshots.append(copy_state(env, object_env))


__all__ = ["analyze_loop", "analyze_match", "analyze_try", "analyze_with"]
