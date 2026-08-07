"""Scope-aware statement and control-flow analysis for runtime access discovery."""

from __future__ import annotations

import ast
from collections.abc import Iterable, Mapping
from typing import Protocol

from .access_control import match_values
from .access_facts import AccessFactCollector
from .access_imports import ImportEngine, bind_import_statement
from .access_model import FlowValue, FunctionInfo
from .access_outcomes import (
    analyze_block_result,
    sync_normal_state,
)

StateSnapshot = tuple[dict[str, FlowValue], dict[str, set[str]]]


class StatementEngine(ImportEngine, Protocol):
    class_comprehension_parents: dict[
        str,
        tuple[dict[str, FlowValue], dict[str, set[str]]],
    ]
    functions: dict[str, FunctionInfo]
    function_refs_by_node: dict[int, str]
    future_annotations: frozenset[str]
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
    ) -> bool: ...

    def _assignment_binding_value(
        self,
        name: str,
        value: FlowValue,
        *,
        module: str,
        actor: str,
    ) -> FlowValue: ...

    def _merge_closure(
        self,
        ref: str,
        *,
        actor: str,
        module: str,
        env: Mapping[str, FlowValue],
    ) -> bool: ...

    def _merge_definition_default(
        self, ref: str, parameter: str, value: FlowValue
    ) -> bool: ...


def analyze_block(
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
) -> tuple[bool, FlowValue]:
    """Keep the established caller API while executing explicit outcomes."""

    result = analyze_block_result(
        engine,
        statements,
        module=module,
        actor=actor,
        class_ref=class_ref,
        env=env,
        object_env=object_env,
        call_ordinals=call_ordinals,
        prefix_states=prefix_states,
    )
    sync_normal_state(env, object_env, result.outcomes)
    return result.changed, result.returned


def _analyze_legacy_block(
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
) -> tuple[bool, FlowValue]:
    changed = False
    returned = FlowValue()
    nested_refs: set[str] = set()
    for statement in statements:
        if isinstance(statement, (ast.Import, ast.ImportFrom)):
            bind_import_statement(
                engine,
                statement,
                module=module,
                actor=actor,
                env=env,
                object_env=object_env,
            )
        elif isinstance(statement, (ast.Assign, ast.AnnAssign)):
            value_node = statement.value
            if value_node is None:
                continue
            value = engine._eval(
                value_node,
                module=module,
                actor=actor,
                class_ref=class_ref,
                env=env,
                object_env=object_env,
                call_ordinals=call_ordinals,
            )
            targets = (
                statement.targets
                if isinstance(statement, ast.Assign)
                else [statement.target]
            )
            for target in targets:
                changed |= engine._bind_target(
                    target,
                    value,
                    actor=actor,
                    class_ref=class_ref,
                    env=env,
                    object_env=object_env,
                )
        elif isinstance(statement, ast.Expr):
            engine._eval(
                statement.value,
                module=module,
                actor=actor,
                class_ref=class_ref,
                env=env,
                object_env=object_env,
                call_ordinals=call_ordinals,
            )
        elif isinstance(statement, ast.Return) and statement.value is not None:
            returned = returned.merged(
                engine._eval(
                    statement.value,
                    module=module,
                    actor=actor,
                    class_ref=class_ref,
                    env=env,
                    object_env=object_env,
                    call_ordinals=call_ordinals,
                ).bound(f"return:{actor}")
            )
        elif isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            nested_ref = engine.function_refs_by_node.get(id(statement))
            if nested_ref is not None:
                nested_refs.add(nested_ref)
                changed |= engine._merge_closure(
                    nested_ref,
                    actor=actor,
                    module=module,
                    env=env,
                )
        elif isinstance(statement, ast.If):
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
            branch_results = [
                _analyze_path(
                    engine,
                    branch,
                    module=module,
                    actor=actor,
                    class_ref=class_ref,
                    env=env,
                    object_env=object_env,
                    call_ordinals=call_ordinals,
                    prefix_states=prefix_states,
                )
                for branch in (statement.body, statement.orelse)
            ]
            branch_changed, branch_return = _join_paths(env, object_env, branch_results)
            changed |= branch_changed
            returned = returned.merged(branch_return)
        elif isinstance(statement, ast.While):
            branch_results = _loop_paths(
                engine,
                statement.body,
                statement.orelse,
                target=None,
                while_test=statement.test,
                module=module,
                actor=actor,
                class_ref=class_ref,
                env=env,
                object_env=object_env,
                call_ordinals=call_ordinals,
                prefix_states=prefix_states,
            )
            branch_changed, branch_return = _join_paths(env, object_env, branch_results)
            changed |= branch_changed
            returned = returned.merged(branch_return)
        elif isinstance(statement, (ast.For, ast.AsyncFor)):
            evaluate_control_expression(
                engine,
                statement.iter,
                kind="async_for" if isinstance(statement, ast.AsyncFor) else "for",
                module=module,
                actor=actor,
                class_ref=class_ref,
                env=env,
                object_env=object_env,
                call_ordinals=call_ordinals,
            )
            branch_results = _loop_paths(
                engine,
                statement.body,
                statement.orelse,
                target=statement.target,
                while_test=None,
                module=module,
                actor=actor,
                class_ref=class_ref,
                env=env,
                object_env=object_env,
                call_ordinals=call_ordinals,
                prefix_states=prefix_states,
            )
            branch_changed, branch_return = _join_paths(env, object_env, branch_results)
            changed |= branch_changed
            returned = returned.merged(branch_return)
        elif isinstance(statement, (ast.With, ast.AsyncWith)):
            kind = "async_with" if isinstance(statement, ast.AsyncWith) else "with"
            for item in statement.items:
                evaluate_control_expression(
                    engine,
                    item.context_expr,
                    kind=kind,
                    module=module,
                    actor=actor,
                    class_ref=class_ref,
                    env=env,
                    object_env=object_env,
                    call_ordinals=call_ordinals,
                )
                if item.optional_vars is not None:
                    changed |= engine._bind_target(
                        item.optional_vars,
                        FlowValue(),
                        actor=actor,
                        class_ref=class_ref,
                        env=env,
                        object_env=object_env,
                    )
            nested_changed, nested_return = analyze_block(
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
            changed |= nested_changed
            returned = returned.merged(nested_return)
        elif isinstance(statement, (ast.Try, ast.TryStar)):
            branch_results = _try_paths(
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
            branch_changed, branch_return = _join_paths(env, object_env, branch_results)
            changed |= branch_changed
            returned = returned.merged(branch_return)
        elif isinstance(statement, ast.Match):
            evaluate_control_expression(
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
            branch_results = _match_paths(
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
            branch_changed, branch_return = _join_paths(env, object_env, branch_results)
            changed |= branch_changed
            returned = returned.merged(branch_return)
        _record_snapshot(prefix_states, env, object_env)
    for nested_ref in nested_refs:
        changed |= engine._merge_closure(
            nested_ref,
            actor=actor,
            module=module,
            env=env,
        )
    return changed, returned


def evaluate_control_expression(
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
            operation=f"control:{kind}",
            sink=f"python.{kind}",
            reason="unsupported_registered_origin_control_flow",
            path=engine.paths[module],
            line=int(node.lineno),
            ordinal=int(call_ordinals.get(id(node), 0)),
        )
    return value


PathResult = tuple[bool, FlowValue, dict[str, FlowValue], dict[str, set[str]]]


def _copy_state(
    env: Mapping[str, FlowValue], object_env: Mapping[str, set[str]]
) -> StateSnapshot:
    return (
        {name: value.copy() for name, value in env.items()},
        {name: set(values) for name, values in object_env.items()},
    )


def _record_snapshot(
    snapshots: list[StateSnapshot] | None,
    env: Mapping[str, FlowValue],
    object_env: Mapping[str, set[str]],
) -> None:
    if snapshots is not None:
        snapshots.append(_copy_state(env, object_env))


def _join_snapshots(snapshots: list[StateSnapshot]) -> StateSnapshot:
    joined_env: dict[str, FlowValue] = {}
    joined_objects: dict[str, set[str]] = {}
    _join_paths(
        joined_env,
        joined_objects,
        [
            (False, FlowValue(), snapshot_env, snapshot_objects)
            for snapshot_env, snapshot_objects in snapshots
        ],
    )
    return joined_env, joined_objects


def _analyze_path(
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
) -> PathResult:
    branch_env, branch_objects = _copy_state(env, object_env)
    _record_snapshot(prefix_states, branch_env, branch_objects)
    changed, returned = analyze_block(
        engine,
        statements,
        module=module,
        actor=actor,
        class_ref=class_ref,
        env=branch_env,
        object_env=branch_objects,
        call_ordinals=call_ordinals,
        prefix_states=prefix_states,
    )
    return changed, returned, branch_env, branch_objects


def _loop_paths(
    engine: StatementEngine,
    body: list[ast.stmt],
    orelse: list[ast.stmt],
    *,
    target: ast.expr | None,
    while_test: ast.expr | None,
    module: str,
    actor: str,
    class_ref: str | None,
    env: dict[str, FlowValue],
    object_env: dict[str, set[str]],
    call_ordinals: Mapping[int, int],
    prefix_states: list[StateSnapshot] | None,
) -> list[PathResult]:
    entry_env, entry_objects = _copy_state(env, object_env)
    head_env, head_objects = _copy_state(env, object_env)
    changed = False
    returned = FlowValue()
    exit_env, exit_objects = _copy_state(env, object_env)
    while True:
        body_env, body_objects = _copy_state(head_env, head_objects)
        if while_test is not None:
            evaluate_control_expression(
                engine,
                while_test,
                kind="while",
                module=module,
                actor=actor,
                class_ref=class_ref,
                env=body_env,
                object_env=body_objects,
                call_ordinals=call_ordinals,
            )
            _record_snapshot(prefix_states, body_env, body_objects)
        exit_env, exit_objects = _copy_state(body_env, body_objects)
        if target is not None:
            engine._bind_target(
                target,
                FlowValue(),
                actor=actor,
                class_ref=class_ref,
                env=body_env,
                object_env=body_objects,
            )
        body_result = _analyze_path(
            engine,
            body,
            module=module,
            actor=actor,
            class_ref=class_ref,
            env=body_env,
            object_env=body_objects,
            call_ordinals=call_ordinals,
            prefix_states=prefix_states,
        )
        changed |= body_result[0]
        returned = returned.merged(body_result[1])
        next_env: dict[str, FlowValue] = {}
        next_objects: dict[str, set[str]] = {}
        _join_paths(
            next_env,
            next_objects,
            [
                (False, FlowValue(), entry_env, entry_objects),
                (False, FlowValue(), body_result[2], body_result[3]),
            ],
        )
        if next_env == head_env and next_objects == head_objects:
            break
        head_env, head_objects = next_env, next_objects

    loop_env, loop_objects = _copy_state(head_env, head_objects)
    if while_test is not None:
        _join_paths(
            loop_env,
            loop_objects,
            [
                (False, FlowValue(), head_env, head_objects),
                (False, FlowValue(), exit_env, exit_objects),
            ],
        )
    loop_result: PathResult = (changed, returned, loop_env, loop_objects)
    results = [loop_result]
    if orelse:
        results.append(
            _analyze_path(
                engine,
                orelse,
                module=module,
                actor=actor,
                class_ref=class_ref,
                env=exit_env,
                object_env=exit_objects,
                call_ordinals=call_ordinals,
                prefix_states=prefix_states,
            )
        )
    return results


def _try_paths(
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
) -> list[PathResult]:
    exception_prefixes: list[StateSnapshot] = []
    body = _analyze_path(
        engine,
        statement.body,
        module=module,
        actor=actor,
        class_ref=class_ref,
        env=env,
        object_env=object_env,
        call_ordinals=call_ordinals,
        prefix_states=exception_prefixes,
    )
    if prefix_states is not None:
        prefix_states.extend(exception_prefixes)
    normal = body
    if statement.orelse:
        normal_else = _analyze_path(
            engine,
            statement.orelse,
            module=module,
            actor=actor,
            class_ref=class_ref,
            env=body[2],
            object_env=body[3],
            call_ordinals=call_ordinals,
            prefix_states=prefix_states,
        )
        normal = (
            body[0] or normal_else[0],
            body[1].merged(normal_else[1]),
            normal_else[2],
            normal_else[3],
        )
    results = [normal]
    handler_base_env, handler_base_objects = _join_snapshots(exception_prefixes)
    for handler in statement.handlers:
        handler_env, handler_objects = _copy_state(
            handler_base_env, handler_base_objects
        )
        if handler.type is not None:
            evaluate_control_expression(
                engine,
                handler.type,
                kind="except_type",
                module=module,
                actor=actor,
                class_ref=class_ref,
                env=handler_env,
                object_env=handler_objects,
                call_ordinals=call_ordinals,
            )
            _record_snapshot(prefix_states, handler_env, handler_objects)
        if handler.name is not None:
            engine._bind_target(
                ast.Name(id=handler.name, ctx=ast.Store()),
                FlowValue(),
                actor=actor,
                class_ref=class_ref,
                env=handler_env,
                object_env=handler_objects,
            )
        results.append(
            _analyze_path(
                engine,
                handler.body,
                module=module,
                actor=actor,
                class_ref=class_ref,
                env=handler_env,
                object_env=handler_objects,
                call_ordinals=call_ordinals,
                prefix_states=prefix_states,
            )
        )
    if statement.finalbody:
        finalized: list[PathResult] = []
        for changed, returned, branch_env, branch_objects in results:
            final = _analyze_path(
                engine,
                statement.finalbody,
                module=module,
                actor=actor,
                class_ref=class_ref,
                env=branch_env,
                object_env=branch_objects,
                call_ordinals=call_ordinals,
                prefix_states=prefix_states,
            )
            finalized.append(
                (changed or final[0], returned.merged(final[1]), final[2], final[3])
            )
        unmatched_final = _analyze_path(
            engine,
            statement.finalbody,
            module=module,
            actor=actor,
            class_ref=class_ref,
            env=handler_base_env,
            object_env=handler_base_objects,
            call_ordinals=call_ordinals,
            prefix_states=prefix_states,
        )
        if finalized:
            changed, returned, branch_env, branch_objects = finalized[0]
            finalized[0] = (
                changed or unmatched_final[0],
                returned.merged(unmatched_final[1]),
                branch_env,
                branch_objects,
            )
        return finalized
    return results


def _match_paths(
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
) -> list[PathResult]:
    initial_env, initial_objects = _copy_state(env, object_env)
    results: list[PathResult] = [(False, FlowValue(), initial_env, initial_objects)]
    for case in statement.cases:
        case_env, case_objects = _copy_state(env, object_env)
        for pattern_value in match_values(case.pattern):
            evaluate_control_expression(
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
            engine._bind_target(
                ast.Name(id=name, ctx=ast.Store()),
                FlowValue(),
                actor=actor,
                class_ref=class_ref,
                env=case_env,
                object_env=case_objects,
            )
        if case.guard is not None:
            evaluate_control_expression(
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
        results.append(
            _analyze_path(
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
        )
    return results


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


def _join_paths(
    env: dict[str, FlowValue],
    object_env: dict[str, set[str]],
    results: list[PathResult],
) -> tuple[bool, FlowValue]:
    joined_env: dict[str, FlowValue] = {}
    joined_objects: dict[str, set[str]] = {}
    changed = False
    returned = FlowValue()
    names = set().union(*(branch_env for _, _, branch_env, _ in results))
    for name in names:
        value = FlowValue()
        for _changed, _returned, branch_env, _objects in results:
            value = value.merged(branch_env.get(name, FlowValue()))
        joined_env[name] = value
    object_names = set().union(*(branch_objects for _, _, _, branch_objects in results))
    for name in object_names:
        joined_objects[name] = set().union(
            *(branch_objects.get(name, set()) for _, _, _, branch_objects in results)
        )
    for branch_changed, branch_return, _branch_env, _objects in results:
        changed |= branch_changed
        returned = returned.merged(branch_return)
    env.clear()
    env.update(joined_env)
    object_env.clear()
    object_env.update(joined_objects)
    return changed, returned


__all__ = ["StatementEngine", "analyze_block", "evaluate_control_expression"]
