"""Function definition-time execution for runtime-access analysis."""

from __future__ import annotations

import ast
from collections.abc import Mapping
from typing import Protocol

from .access_facts import AccessFactCollector
from .access_model import FlowValue, FunctionInfo


class DefinitionExecutionEngine(Protocol):
    facts: AccessFactCollector
    functions: dict[str, FunctionInfo]
    paths: dict[str, str]
    function_refs_by_node: dict[int, str]
    future_annotations: frozenset[str]
    returns: dict[str, FlowValue]

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

    def _function_definition_value(
        self,
        ref: str,
        *,
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
        self,
        ref: str,
        parameter: str,
        value: FlowValue,
    ) -> bool: ...

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


def analyze_function_definition(
    engine: DefinitionExecutionEngine,
    statement: ast.FunctionDef | ast.AsyncFunctionDef,
    *,
    module: str,
    actor: str,
    class_ref: str | None,
    env: dict[str, FlowValue],
    object_env: dict[str, set[str]],
    call_ordinals: Mapping[int, int],
) -> bool:
    """Evaluate a function header in source order without executing its body."""

    changed = False
    decorators = [
        (
            decorator,
            _evaluate_header_use(
                engine,
                decorator,
                kind="decorator",
                module=module,
                actor=actor,
                class_ref=class_ref,
                env=env,
                object_env=object_env,
                call_ordinals=call_ordinals,
            ),
        )
        for decorator in statement.decorator_list
    ]

    function_ref = engine.function_refs_by_node.get(id(statement))
    for parameter, default in function_defaults(statement):
        value = engine._eval(
            default,
            module=module,
            actor=actor,
            class_ref=class_ref,
            env=env,
            object_env=object_env,
            call_ordinals=call_ordinals,
        )
        if function_ref is not None:
            changed |= engine._merge_definition_default(function_ref, parameter, value)

    if module not in engine.future_annotations:
        for annotation in function_annotations(statement):
            _evaluate_header_use(
                engine,
                annotation,
                kind="annotation",
                module=module,
                actor=actor,
                class_ref=class_ref,
                env=env,
                object_env=object_env,
                call_ordinals=call_ordinals,
            )

    definition_value = (
        engine._function_definition_value(function_ref, actor=actor)
        if function_ref is not None
        else FlowValue(unknown_callable=True)
    )
    decorated_value = apply_definition_decorators(
        engine,
        decorators,
        definition_value,
        module=module,
        actor=actor,
        call_ordinals=call_ordinals,
    )
    changed |= engine._bind_target(
        ast.Name(id=statement.name, ctx=ast.Store()),
        decorated_value,
        actor=actor,
        class_ref=class_ref,
        env=env,
        object_env=object_env,
    )
    if function_ref is not None:
        changed |= engine._merge_closure(
            function_ref,
            actor=actor,
            module=module,
            env=env,
        )
    return changed


def evaluate_class_header(
    engine: DefinitionExecutionEngine,
    statement: ast.ClassDef,
    *,
    module: str,
    actor: str,
    class_ref: str | None,
    env: dict[str, FlowValue],
    object_env: dict[str, set[str]],
    call_ordinals: Mapping[int, int],
) -> list[tuple[ast.expr, FlowValue]]:
    """Evaluate class decorators, bases, and keywords in enclosing scope."""

    decorators = [
        (
            decorator,
            _evaluate_header_use(
                engine,
                decorator,
                kind="decorator",
                module=module,
                actor=actor,
                class_ref=class_ref,
                env=env,
                object_env=object_env,
                call_ordinals=call_ordinals,
            ),
        )
        for decorator in statement.decorator_list
    ]
    expressions = [*statement.bases]
    expressions.extend(keyword.value for keyword in statement.keywords)
    for expression in expressions:
        engine._eval(
            expression,
            module=module,
            actor=actor,
            class_ref=class_ref,
            env=env,
            object_env=object_env,
            call_ordinals=call_ordinals,
        )
    return decorators


def apply_definition_decorators(
    engine: DefinitionExecutionEngine,
    decorators: list[tuple[ast.expr, FlowValue]],
    definition_value: FlowValue,
    *,
    module: str,
    actor: str,
    call_ordinals: Mapping[int, int],
) -> FlowValue:
    """Apply already-evaluated decorators from the innermost outward."""

    current = definition_value.copy()
    for expression, decorator in reversed(decorators):
        applied = FlowValue()
        known = False
        for target in sorted(decorator.call_targets):
            info = engine.functions.get(target)
            if info is None:
                continue
            known = True
            ordinal = int(call_ordinals.get(id(expression), 0))
            if info.parameters and current.has_origins:
                call = ast.copy_location(
                    ast.Call(func=expression, args=[], keywords=[]),
                    expression,
                )
                engine._bind_call_parameter(
                    target,
                    info.parameters[0],
                    current,
                    actor=actor,
                    module=module,
                    node=call,
                    ordinal=ordinal,
                    site_node=expression,
                )
            applied = applied.merged(engine.returns[target].bound(f"result:{target}"))
        for target in sorted(decorator.class_targets):
            known = True
            applied = applied.merged(FlowValue(object_types={target}))
        if decorator.unknown_callable or not known:
            applied.unknown_callable = True
            if current.has_origins:
                engine.facts.record_escape(
                    current,
                    node=expression,
                    actor=actor,
                    operation="definition:decorator_application",
                    sink="python.decorator",
                    reason="registered_locator_to_unknown_callee",
                    path=engine.paths[module],
                    line=int(expression.lineno),
                    ordinal=int(call_ordinals.get(id(expression), 0)),
                )
        current = applied
    return current


def function_defaults(
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


def function_annotations(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> list[ast.expr]:
    annotations = [
        argument.annotation
        for argument in [*node.args.posonlyargs, *node.args.args]
        if argument.annotation is not None
    ]
    if node.args.vararg is not None and node.args.vararg.annotation is not None:
        annotations.append(node.args.vararg.annotation)
    annotations.extend(
        argument.annotation
        for argument in node.args.kwonlyargs
        if argument.annotation is not None
    )
    if node.args.kwarg is not None and node.args.kwarg.annotation is not None:
        annotations.append(node.args.kwarg.annotation)
    if node.returns is not None:
        annotations.append(node.returns)
    return annotations


def _evaluate_header_use(
    engine: DefinitionExecutionEngine,
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
            operation=f"definition:{kind}",
            sink=f"python.{kind}",
            reason="unsupported_registered_origin_definition",
            path=engine.paths[module],
            line=int(node.lineno),
            ordinal=int(call_ordinals.get(id(node), 0)),
        )
    if not value.call_targets and not value.class_targets:
        value.unknown_callable = True
    return value


__all__ = [
    "DefinitionExecutionEngine",
    "analyze_function_definition",
    "apply_definition_decorators",
    "evaluate_class_header",
    "function_annotations",
    "function_defaults",
]
