"""Definition-time execution for isolated class namespaces."""

from __future__ import annotations

import ast
from collections.abc import Mapping
from typing import TYPE_CHECKING, Protocol, cast

from .access_definition_execution import (
    DefinitionExecutionEngine,
    apply_definition_decorators,
    evaluate_class_header,
)
from .access_model import FlowValue, _call_ordinals, _class_definition_ref
from .access_outcomes import (
    BlockResult,
    Outcome,
    analyze_block_result,
    copy_state,
    normalize_outcomes,
)

if TYPE_CHECKING:
    from .access_statements import StatementEngine


class ModuleScopeEngine(Protocol):
    module_runtime_envs: dict[str, dict[str, FlowValue]]


def analyze_class_definition(
    engine: StatementEngine,
    statement: ast.ClassDef,
    *,
    module: str,
    actor: str,
    class_ref: str | None,
    env: dict[str, FlowValue],
    object_env: dict[str, set[str]],
    call_ordinals: Mapping[int, int],
) -> BlockResult:
    """Execute a class header/body and expose only successful construction."""

    class_call_ordinals = _call_ordinals(
        statement,
        evaluate_annotations=module not in engine.future_annotations,
    )
    definition_engine = cast(DefinitionExecutionEngine, engine)
    decorators = evaluate_class_header(
        definition_engine,
        statement,
        module=module,
        actor=actor,
        class_ref=class_ref,
        env=env,
        object_env=object_env,
        call_ordinals=class_call_ordinals or call_ordinals,
    )
    current_ref = _class_definition_ref(
        module=module,
        actor=actor,
        enclosing_class_ref=class_ref,
        name=statement.name,
    )
    class_actor = f"{current_ref}.<classbody>"
    declared_globals, declared_nonlocals = _class_scope_declarations(statement)
    global_names = declared_globals if actor == f"{module}:<module>" else frozenset()
    nonlocal_names = _resolved_nonlocal_names(
        engine,
        actor=actor,
        names=declared_nonlocals,
    )
    lexical_parent = engine.class_comprehension_parents.get(actor)
    if lexical_parent is None:
        lexical_parent = copy_state(env, object_env)
    previous_parent = engine.class_comprehension_parents.get(class_actor)
    engine.class_comprehension_parents[class_actor] = copy_state(*lexical_parent)
    class_env, class_objects = copy_state(env, object_env)
    module_env = (
        env
        if actor == f"{module}:<module>"
        else cast(ModuleScopeEngine, engine).module_runtime_envs[module]
    )
    for name in declared_globals:
        value = module_env.get(name, FlowValue()).copy()
        class_env[name] = value
        if value.object_types:
            class_objects[name] = set(value.object_types)
        else:
            class_objects.pop(name, None)
    try:
        body_result = analyze_block_result(
            engine,
            statement.body,
            module=module,
            actor=class_actor,
            class_ref=current_ref,
            env=class_env,
            object_env=class_objects,
            call_ordinals=class_call_ordinals,
        )
    finally:
        if previous_parent is None:
            engine.class_comprehension_parents.pop(class_actor, None)
        else:
            engine.class_comprehension_parents[class_actor] = previous_parent

    changed = body_result.changed
    outcomes: list[Outcome] = []
    for body_outcome in body_result.outcomes:
        outer_env, outer_objects = copy_state(env, object_env)
        for name in global_names | nonlocal_names:
            outer_env[name] = body_outcome.env.get(name, FlowValue()).copy()
            if name in body_outcome.object_env:
                outer_objects[name] = set(body_outcome.object_env[name])
            else:
                outer_objects.pop(name, None)
        if body_outcome.kind == "normal":
            decorated_value = apply_definition_decorators(
                definition_engine,
                decorators,
                FlowValue(
                    object_types={current_ref},
                    class_targets={current_ref},
                ),
                module=module,
                actor=actor,
                call_ordinals=class_call_ordinals or call_ordinals,
            )
            changed |= engine._bind_target(
                ast.Name(id=statement.name, ctx=ast.Store()),
                decorated_value,
                actor=actor,
                class_ref=class_ref,
                env=outer_env,
                object_env=outer_objects,
            )
        outcomes.append(
            Outcome(
                body_outcome.kind,
                outer_env,
                outer_objects,
                body_outcome.value.copy(),
            )
        )
    return BlockResult(changed, normalize_outcomes(outcomes))


def _class_scope_declarations(
    statement: ast.ClassDef,
) -> tuple[frozenset[str], frozenset[str]]:
    global_names: set[str] = set()
    nonlocal_names: set[str] = set()

    class DeclarationVisitor(ast.NodeVisitor):
        def visit_Global(self, item: ast.Global) -> None:
            global_names.update(item.names)

        def visit_Nonlocal(self, item: ast.Nonlocal) -> None:
            nonlocal_names.update(item.names)

        def visit_FunctionDef(self, item: ast.FunctionDef) -> None:
            return

        def visit_AsyncFunctionDef(self, item: ast.AsyncFunctionDef) -> None:
            return

        def visit_ClassDef(self, item: ast.ClassDef) -> None:
            return

        def visit_ListComp(self, item: ast.ListComp) -> None:
            return

        def visit_SetComp(self, item: ast.SetComp) -> None:
            return

        def visit_GeneratorExp(self, item: ast.GeneratorExp) -> None:
            return

        def visit_DictComp(self, item: ast.DictComp) -> None:
            return

    visitor = DeclarationVisitor()
    for child in statement.body:
        visitor.visit(child)
    return frozenset(global_names), frozenset(nonlocal_names)


def _resolved_nonlocal_names(
    engine: StatementEngine,
    *,
    actor: str,
    names: frozenset[str],
) -> frozenset[str]:
    if not names:
        return names
    function = engine.functions.get(actor)
    unresolved = names if function is None else names - function.local_names
    if unresolved:
        rendered = ", ".join(sorted(unresolved))
        raise ValueError(f"unresolved class nonlocal in {actor}: {rendered}")
    return names


__all__ = ["analyze_class_definition"]
