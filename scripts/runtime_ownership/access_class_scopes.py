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
from .access_model import (
    DATACLASS_INIT_VAR_MARKER,
    DATACLASS_KW_ONLY_MARKER,
    SOCKETSERVER_CLASS_OBJECT_PREFIX,
    SOCKETSERVER_LOCAL_CLASS_OBJECT_PREFIX,
    STDLIB_SOCKETSERVER_CLASSES,
    DataclassFieldInfo,
    DataclassInfo,
    FlowValue,
    _call_ordinals,
    _class_definition_ref,
    is_precise_stdlib_module,
    precise_stdlib_module_name,
)
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
    module_exports: dict[str, dict[str, FlowValue]]
    dataclass_infos: dict[str, DataclassInfo]
    _persistent_changed: bool
    known_modules: frozenset[str]

    def _socket_field_origin(
        self,
        *,
        module: str,
        class_ref: str,
        field_name: str,
        expression: ast.expr,
    ) -> FlowValue | None: ...


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
            dataclass_decorators = {
                index
                for index, expression in enumerate(statement.decorator_list)
                if _is_exact_dataclass_decorator(expression, env)
            }
            dataclass_info: DataclassInfo | None = None
            if len(dataclass_decorators) == 1:
                decorator = statement.decorator_list[
                    next(iter(dataclass_decorators))
                ]
                dataclass_info = _dataclass_info(
                    cast(ModuleScopeEngine, engine),
                    statement,
                    decorator=decorator,
                    module=module,
                    current_ref=current_ref,
                    outer_env=env,
                    class_env=body_outcome.env,
                )
            definition_value = FlowValue(
                object_types={current_ref},
                class_targets={current_ref},
            )
            socketserver_kind = _safe_socketserver_subclass_kind(
                statement,
                outer_env=env,
                known_modules=cast(ModuleScopeEngine, engine).known_modules,
            )
            if socketserver_kind is not None:
                definition_value.object_types.add(
                    f"{SOCKETSERVER_LOCAL_CLASS_OBJECT_PREFIX}{socketserver_kind}"
                )
            if dataclass_info is not None:
                prior_info = cast(ModuleScopeEngine, engine).dataclass_infos.get(
                    current_ref
                )
                if prior_info != dataclass_info:
                    cast(ModuleScopeEngine, engine).dataclass_infos[current_ref] = (
                        dataclass_info
                    )
                    cast(ModuleScopeEngine, engine)._persistent_changed = True
                    changed = True
                definition_value.attribute_values = {
                    field.name: field.default.copy()
                    for field in dataclass_info.fields
                    if field.default is not None
                }
                definition_value.attribute_values_complete = (
                    not dataclass_info.shape_ambiguous
                )
                definition_value.attribute_values_ambiguous = (
                    dataclass_info.shape_ambiguous
                )
            decorated_value = apply_definition_decorators(
                definition_engine,
                [
                    decorator
                    for index, decorator in enumerate(decorators)
                    if index not in dataclass_decorators
                ],
                definition_value,
                module=module,
                actor=actor,
                call_ordinals=class_call_ordinals or call_ordinals,
            )
            if actor == f"{module}:<module>":
                exported = decorated_value.bound(
                    f"definition:{module}:<module>:{statement.name}"
                )
                previous_export = cast(ModuleScopeEngine, engine).module_exports[
                    module
                ].get(statement.name)
                if previous_export != exported:
                    cast(ModuleScopeEngine, engine).module_exports[module][
                        statement.name
                    ] = exported
                    cast(ModuleScopeEngine, engine)._persistent_changed = True
                    changed = True
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


def _safe_socketserver_subclass_kind(
    statement: ast.ClassDef,
    *,
    outer_env: Mapping[str, FlowValue],
    known_modules: frozenset[str],
) -> str | None:
    """Recognize only the inert local subclasses used by production services."""

    if statement.decorator_list or statement.keywords or len(statement.bases) != 1:
        return None
    base_expression = statement.bases[0]
    base_value: FlowValue | None = None
    if isinstance(base_expression, ast.Name):
        base_value = outer_env.get(base_expression.id)
    elif (
        isinstance(base_expression, ast.Attribute)
        and isinstance(base_expression.value, ast.Name)
    ):
        module_value = outer_env.get(base_expression.value.id)
        if (
            module_value is not None
            and "socketserver" not in known_modules
            and base_expression.attr in STDLIB_SOCKETSERVER_CLASSES
            and is_precise_stdlib_module(
                module_value,
                module="socketserver",
                attribute=base_expression.attr,
            )
        ):
            base_value = FlowValue(
                object_types={
                    f"{SOCKETSERVER_CLASS_OBJECT_PREFIX}{base_expression.attr}"
                }
            )
    if base_value is None or len(base_value.object_types) != 1:
        return None
    marker = next(iter(base_value.object_types))
    if not marker.startswith(SOCKETSERVER_CLASS_OBJECT_PREFIX):
        return None
    kind = marker.removeprefix(SOCKETSERVER_CLASS_OBJECT_PREFIX)
    if kind not in STDLIB_SOCKETSERVER_CLASSES:
        return None
    for child in statement.body:
        if isinstance(child, ast.Pass):
            continue
        if (
            isinstance(child, ast.Expr)
            and isinstance(child.value, ast.Constant)
            and isinstance(child.value.value, str)
        ):
            continue
        if not isinstance(child, (ast.Assign, ast.AnnAssign)):
            return None
        targets = child.targets if isinstance(child, ast.Assign) else [child.target]
        if len(targets) != 1 or not isinstance(targets[0], ast.Name):
            return None
        if targets[0].id not in {"allow_reuse_address", "daemon_threads"}:
            return None
        if child.value is None or not (
            isinstance(child.value, ast.Constant)
            and isinstance(child.value.value, bool)
        ):
            return None
    return kind


def _is_exact_dataclass_decorator(
    expression: ast.expr,
    env: Mapping[str, FlowValue],
) -> bool:
    callable_expression = expression.func if isinstance(expression, ast.Call) else expression
    if not _is_exact_stdlib_callable(
        callable_expression,
        env,
        target="dataclasses:dataclass",
    ):
        return False
    if not isinstance(expression, ast.Call):
        return True
    allowed = {
        "init",
        "repr",
        "eq",
        "order",
        "unsafe_hash",
        "frozen",
        "match_args",
        "kw_only",
        "slots",
        "weakref_slot",
    }
    return (
        not expression.args
        and all(keyword.arg in allowed for keyword in expression.keywords)
        and all(
            isinstance(keyword.value, ast.Constant)
            and isinstance(keyword.value.value, bool)
            for keyword in expression.keywords
        )
    )


def _is_exact_stdlib_callable(
    expression: ast.expr,
    env: Mapping[str, FlowValue],
    *,
    target: str,
) -> bool:
    if isinstance(expression, ast.Name):
        value = env.get(expression.id)
        return value is not None and _has_only_call_target(value, target)
    if isinstance(expression, ast.Attribute) and isinstance(expression.value, ast.Name):
        module = env.get(expression.value.id)
        target_module, _, target_attribute = target.partition(":")
        return (
            expression.attr == target_attribute
            and module is not None
            and precise_stdlib_module_name(module) == target_module
        )
    return False


def _has_only_call_target(value: FlowValue, target: str) -> bool:
    return (
        value.call_targets == {target}
        and not value.origins
        and not value.object_types
        and not value.module_refs
        and not value.class_targets
        and not value.unknown_callable
        and not value.attribute_values
        and not value.attribute_values_complete
        and not value.attribute_values_ambiguous
    )


def _dataclass_info(
    engine: ModuleScopeEngine,
    statement: ast.ClassDef,
    *,
    decorator: ast.expr,
    module: str,
    current_ref: str,
    outer_env: Mapping[str, FlowValue],
    class_env: Mapping[str, FlowValue],
) -> DataclassInfo | None:
    decorator_options = _boolean_keywords(decorator)
    generated_init = decorator_options.get("init", True)
    default_keyword_only = decorator_options.get("kw_only", False)
    explicit_init = any(
        isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
        and child.name == "__init__"
        for child in statement.body
    )
    inherited: dict[str, DataclassFieldInfo] = {}
    shape_ambiguous = False
    inherited_post_init: str | None = None
    for base_expression in reversed(statement.bases):
        base = _reference_value(base_expression, class_env, outer_env)
        if base is None or len(base.class_targets) != 1:
            if isinstance(base_expression, ast.Name) and base_expression.id == "object":
                continue
            shape_ambiguous = True
            continue
        base_target = next(iter(base.class_targets))
        base_info = engine.dataclass_infos.get(base_target)
        if base_info is None:
            shape_ambiguous = True
            for name, value in base.attribute_values.items():
                inherited[name] = DataclassFieldInfo(name=name, default=value.copy())
            continue
        shape_ambiguous |= base_info.shape_ambiguous
        for field in base_info.fields:
            inherited[field.name] = field
        inherited_post_init = base_info.post_init_target or inherited_post_init

    fields = dict(inherited)
    keyword_only = default_keyword_only
    for child in statement.body:
        if not (
            isinstance(child, ast.AnnAssign)
            and isinstance(child.target, ast.Name)
            and not _is_class_var_annotation(child.annotation)
        ):
            continue
        name = child.target.id
        if _is_kw_only_annotation(child.annotation, class_env, outer_env):
            if child.value is not None or keyword_only:
                return None
            keyword_only = True
            continue
        init_var = _is_init_var_annotation(
            child.annotation,
            class_env,
            outer_env,
        )
        default: FlowValue | None = None
        default_factory: FlowValue | None = None
        default_factory_expression: ast.expr | None = None
        field_init = True
        field_keyword_only = keyword_only
        if (
            child.value is not None
            and isinstance(child.value, ast.Call)
            and _is_exact_stdlib_callable(
                child.value.func,
                outer_env,
                target="dataclasses:field",
            )
            and not _is_exact_field_call(child.value, outer_env)
        ):
            return None
        if child.value is not None and _is_exact_field_call(child.value, outer_env):
            field_call = cast(ast.Call, child.value)
            options = _boolean_keywords(field_call)
            field_init = options.get("init", True)
            field_keyword_only = options.get("kw_only", keyword_only)
            default_expression = _keyword_expression(field_call, "default")
            factory_expression = _keyword_expression(
                field_call,
                "default_factory",
            )
            if default_expression is not None and factory_expression is not None:
                return None
            if default_expression is not None:
                default = class_env.get(name, FlowValue()).copy()
            elif factory_expression is not None:
                factory = _reference_value(factory_expression, class_env, outer_env)
                if factory is None and not isinstance(factory_expression, ast.Lambda):
                    return None
                if isinstance(factory_expression, ast.Lambda) and not _is_zero_arg_lambda(
                    factory_expression
                ):
                    return None
                default_factory = factory.copy() if factory is not None else FlowValue()
                default_factory_expression = factory_expression
        elif child.value is not None:
            default = class_env.get(name, FlowValue()).copy()
        default_expression = (
            _keyword_expression(cast(ast.Call, child.value), "default")
            if child.value is not None and _is_exact_field_call(child.value, outer_env)
            else child.value
        )
        if default_expression is not None:
            socket_origin = engine._socket_field_origin(
                module=module,
                class_ref=current_ref,
                field_name=name,
                expression=default_expression,
            )
            if socket_origin is not None:
                default = socket_origin
        declared = _reference_value(child.annotation, class_env, outer_env)
        fields[name] = DataclassFieldInfo(
            name=name,
            default=default,
            default_factory=default_factory,
            default_factory_expression=default_factory_expression,
            declared_class_targets=frozenset(
                declared.class_targets if declared is not None else ()
            ),
            init=field_init,
            keyword_only=field_keyword_only,
            init_var=init_var,
        )
    post_init_ref = f"{current_ref}.__post_init__"
    local_post_init = next(
        (
            child
            for child in statement.body
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            and child.name == "__post_init__"
        ),
        None,
    )
    post_init_unknown = any(
        (
            isinstance(child, (ast.Assign, ast.AnnAssign))
            and any(name == "__post_init__" for name in _assigned_names(child))
        )
        for child in statement.body
    )
    return DataclassInfo(
        fields=tuple(fields.values()),
        generated_init=generated_init and not explicit_init,
        explicit_init=explicit_init,
        shape_ambiguous=shape_ambiguous,
        post_init_target=(
            post_init_ref if local_post_init is not None else inherited_post_init
        ),
        post_init_unknown=post_init_unknown,
    )


def _is_exact_field_call(
    expression: ast.expr,
    env: Mapping[str, FlowValue],
) -> bool:
    if not isinstance(expression, ast.Call) or expression.args:
        return False
    if not _is_exact_stdlib_callable(
        expression.func,
        env,
        target="dataclasses:field",
    ):
        return False
    allowed = {
        "default",
        "default_factory",
        "init",
        "repr",
        "hash",
        "compare",
        "metadata",
        "kw_only",
    }
    names = [keyword.arg for keyword in expression.keywords]
    if any(name not in allowed for name in names) or len(names) != len(set(names)):
        return False
    boolean_options = {"init", "repr", "compare", "kw_only"}
    for keyword in expression.keywords:
        if keyword.arg in boolean_options and not (
            isinstance(keyword.value, ast.Constant)
            and isinstance(keyword.value.value, bool)
        ):
            return False
        if keyword.arg == "hash" and not (
            isinstance(keyword.value, ast.Constant)
            and (
                keyword.value.value is None
                or isinstance(keyword.value.value, bool)
            )
        ):
            return False
    return True


def _is_zero_arg_lambda(expression: ast.Lambda) -> bool:
    args = expression.args
    return not (
        args.posonlyargs
        or args.args
        or args.kwonlyargs
        or args.vararg
        or args.kwarg
    )


def _boolean_keywords(expression: ast.expr) -> dict[str, bool]:
    if not isinstance(expression, ast.Call):
        return {}
    return {
        str(keyword.arg): bool(keyword.value.value)
        for keyword in expression.keywords
        if keyword.arg is not None
        and isinstance(keyword.value, ast.Constant)
        and isinstance(keyword.value.value, bool)
    }


def _keyword_expression(expression: ast.Call, name: str) -> ast.expr | None:
    values = [keyword.value for keyword in expression.keywords if keyword.arg == name]
    return values[0] if len(values) == 1 else None


def _reference_value(
    expression: ast.expr,
    class_env: Mapping[str, FlowValue],
    outer_env: Mapping[str, FlowValue],
) -> FlowValue | None:
    if isinstance(expression, ast.Name):
        return class_env.get(expression.id) or outer_env.get(expression.id)
    if isinstance(expression, ast.Attribute) and isinstance(expression.value, ast.Name):
        base = class_env.get(expression.value.id) or outer_env.get(expression.value.id)
        if base is not None and expression.attr in base.attribute_values:
            return base.attribute_values[expression.attr]
    return None


def _is_kw_only_annotation(
    annotation: ast.expr,
    class_env: Mapping[str, FlowValue],
    outer_env: Mapping[str, FlowValue],
) -> bool:
    value = _reference_value(annotation, class_env, outer_env)
    if value is not None and value.object_types == {DATACLASS_KW_ONLY_MARKER}:
        return True
    if isinstance(annotation, ast.Attribute) and annotation.attr == "KW_ONLY":
        base = (
            class_env.get(annotation.value.id) or outer_env.get(annotation.value.id)
            if isinstance(annotation.value, ast.Name)
            else None
        )
        return base is not None and precise_stdlib_module_name(base) == "dataclasses"
    return False


def _is_init_var_annotation(
    annotation: ast.expr,
    class_env: Mapping[str, FlowValue],
    outer_env: Mapping[str, FlowValue],
) -> bool:
    base_annotation = (
        annotation.value if isinstance(annotation, ast.Subscript) else annotation
    )
    value = _reference_value(base_annotation, class_env, outer_env)
    if value is not None and value.object_types == {DATACLASS_INIT_VAR_MARKER}:
        return True
    if isinstance(base_annotation, ast.Attribute) and base_annotation.attr == "InitVar":
        base = (
            class_env.get(base_annotation.value.id)
            or outer_env.get(base_annotation.value.id)
            if isinstance(base_annotation.value, ast.Name)
            else None
        )
        return base is not None and precise_stdlib_module_name(base) == "dataclasses"
    return False


def _assigned_names(statement: ast.Assign | ast.AnnAssign) -> tuple[str, ...]:
    targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
    return tuple(
        target.id for target in targets if isinstance(target, ast.Name)
    )


def _is_class_var_annotation(annotation: ast.expr) -> bool:
    base = annotation.value if isinstance(annotation, ast.Subscript) else annotation
    if isinstance(base, ast.Name):
        return base.id == "ClassVar"
    return isinstance(base, ast.Attribute) and base.attr == "ClassVar"


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
