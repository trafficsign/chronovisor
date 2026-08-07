"""Generic expression traversal for runtime access discovery."""

from __future__ import annotations

import ast
from collections.abc import Mapping
from typing import Protocol

from .access_bindings import bind_structured_target, evaluate_iterable
from .access_facts import AccessFactCollector
from .access_model import FlowValue


class ExpressionEngine(Protocol):
    class_comprehension_parents: dict[
        str,
        tuple[dict[str, FlowValue], dict[str, set[str]]],
    ]
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


def evaluate_generic_expression(
    engine: ExpressionEngine,
    node: ast.expr,
    *,
    module: str,
    actor: str,
    class_ref: str | None,
    env: dict[str, FlowValue],
    object_env: dict[str, set[str]],
    call_ordinals: Mapping[int, int],
) -> FlowValue | None:
    if isinstance(node, ast.Await):
        value = engine._eval(
            node.value,
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
                operation="control:await",
                sink="python.await",
                reason="unsupported_registered_origin_control_flow",
                path=engine.paths[module],
                line=int(node.lineno),
                ordinal=int(call_ordinals.get(id(node), 0)),
            )
            return FlowValue()
        return value
    if isinstance(node, ast.Starred):
        return engine._eval(
            node.value,
            module=module,
            actor=actor,
            class_ref=class_ref,
            env=env,
            object_env=object_env,
            call_ordinals=call_ordinals,
        )
    if isinstance(node, ast.NamedExpr):
        value = engine._eval(
            node.value,
            module=module,
            actor=actor,
            class_ref=class_ref,
            env=env,
            object_env=object_env,
            call_ordinals=call_ordinals,
        )
        engine._bind_target(
            node.target,
            value,
            actor=actor,
            class_ref=class_ref,
            env=env,
            object_env=object_env,
        )
        return value
    if isinstance(node, ast.Lambda):
        escaped = _lambda_origins(
            engine,
            node,
            module=module,
            actor=actor,
            class_ref=class_ref,
            env=env,
            object_env=object_env,
            call_ordinals=call_ordinals,
        )
        if escaped.has_origins:
            engine.facts.record_escape(
                escaped.bound(f"lambda:{actor}"),
                node=node,
                actor=actor,
                operation="lambda",
                sink="python.lambda",
                reason="unsupported_registered_origin_lambda",
                path=engine.paths[module],
                line=int(node.lineno),
                ordinal=0,
            )
        return FlowValue()
    if isinstance(node, (ast.Tuple, ast.List)) and not any(
        isinstance(item, ast.Starred) for item in node.elts
    ):
        items = tuple(
            engine._eval(
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
        result = FlowValue(structured_items=items)
        for item in items:
            result = result.merged(item)
        result.structured_items = items
        return result.bound(f"expression:{type(node).__name__}")
    if isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp)):
        result, local_env, local_objects = _evaluate_comprehension_generators(
            engine,
            node.generators,
            module=module,
            actor=actor,
            class_ref=class_ref,
            env=env,
            object_env=object_env,
            call_ordinals=call_ordinals,
        )
        result = result.merged(
            _evaluate_comprehension_expression(
                engine,
                node.elt,
                module=module,
                actor=actor,
                class_ref=class_ref,
                env=local_env,
                object_env=local_objects,
                containing_env=env,
                containing_objects=object_env,
                call_ordinals=call_ordinals,
            )
        )
        return result.bound(f"expression:{type(node).__name__}")
    if isinstance(node, ast.DictComp):
        result, local_env, local_objects = _evaluate_comprehension_generators(
            engine,
            node.generators,
            module=module,
            actor=actor,
            class_ref=class_ref,
            env=env,
            object_env=object_env,
            call_ordinals=call_ordinals,
        )
        for child in (node.key, node.value):
            result = result.merged(
                _evaluate_comprehension_expression(
                    engine,
                    child,
                    module=module,
                    actor=actor,
                    class_ref=class_ref,
                    env=local_env,
                    object_env=local_objects,
                    containing_env=env,
                    containing_objects=object_env,
                    call_ordinals=call_ordinals,
                )
            )
        return result.bound("expression:DictComp")
    children = _generic_children(node)
    if children is None:
        return None
    result = FlowValue()
    for child in children:
        result = result.merged(
            engine._eval(
                child,
                module=module,
                actor=actor,
                class_ref=class_ref,
                env=env,
                object_env=object_env,
                call_ordinals=call_ordinals,
            )
        )
    bound = result.bound(f"expression:{type(node).__name__}")
    if isinstance(node, (ast.BinOp, ast.Subscript)) and bound.has_origins:
        engine.facts.record_escape(
            bound,
            node=node,
            actor=actor,
            operation=f"expression:{type(node).__name__.lower()}",
            sink=f"python.{type(node).__name__}",
            reason="unsupported_registered_origin_expression",
            path=engine.paths[module],
            line=int(node.lineno),
            ordinal=int(call_ordinals.get(id(node), 0)),
        )
        return FlowValue()
    return bound


def _generic_children(node: ast.expr) -> list[ast.expr] | None:
    if isinstance(node, ast.BoolOp):
        return list(node.values)
    if isinstance(node, ast.Compare):
        return [node.left, *node.comparators]
    if isinstance(node, ast.UnaryOp):
        return [node.operand]
    if isinstance(node, ast.BinOp):
        return [node.left, node.right]
    if isinstance(node, ast.IfExp):
        return [node.test, node.body, node.orelse]
    if isinstance(node, ast.Subscript):
        return [node.value, *_slice_children(node.slice)]
    if isinstance(node, ast.Dict):
        return [key for key in node.keys if key is not None] + list(node.values)
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return list(node.elts)
    if isinstance(node, ast.JoinedStr):
        return list(node.values)
    if isinstance(node, ast.FormattedValue):
        children = [node.value]
        if node.format_spec is not None:
            children.append(node.format_spec)
        return children
    return None


def _evaluate_comprehension_generators(
    engine: ExpressionEngine,
    generators: list[ast.comprehension],
    *,
    module: str,
    actor: str,
    class_ref: str | None,
    env: dict[str, FlowValue],
    object_env: dict[str, set[str]],
    call_ordinals: Mapping[int, int],
) -> tuple[FlowValue, dict[str, FlowValue], dict[str, set[str]]]:
    parent = engine.class_comprehension_parents.get(actor)
    if parent is None:
        parent = (env, object_env)
    local_env = {name: value.copy() for name, value in parent[0].items()}
    local_objects = {name: set(values) for name, values in parent[1].items()}
    result = FlowValue()
    for index, generator in enumerate(generators):
        iterable_env = env if index == 0 else local_env
        iterable_objects = object_env if index == 0 else local_objects
        iterable = evaluate_iterable(
            engine,
            generator.iter,
            module=module,
            actor=actor,
            class_ref=class_ref,
            env=iterable_env,
            object_env=iterable_objects,
            call_ordinals=call_ordinals,
        )
        result = result.merged(iterable.aggregate)
        if not iterable.literal:
            _record_comprehension_control(
                engine,
                iterable.aggregate,
                generator.iter,
                kind="comprehension_iter",
                module=module,
                actor=actor,
                call_ordinals=call_ordinals,
            )
        bind_structured_target(
            engine,
            generator.target,
            iterable.item,
            module=module,
            actor=actor,
            class_ref=class_ref,
            env=local_env,
            object_env=local_objects,
            call_ordinals=call_ordinals,
        )
        for condition in generator.ifs:
            result = result.merged(
                _evaluate_comprehension_control(
                    engine,
                    condition,
                    kind="comprehension_if",
                    module=module,
                    actor=actor,
                    class_ref=class_ref,
                    env=local_env,
                    object_env=local_objects,
                    containing_env=env,
                    containing_objects=object_env,
                    call_ordinals=call_ordinals,
                )
            )
    return result, local_env, local_objects


def _evaluate_comprehension_control(
    engine: ExpressionEngine,
    node: ast.expr,
    *,
    kind: str,
    module: str,
    actor: str,
    class_ref: str | None,
    env: dict[str, FlowValue],
    object_env: dict[str, set[str]],
    containing_env: dict[str, FlowValue],
    containing_objects: dict[str, set[str]],
    call_ordinals: Mapping[int, int],
) -> FlowValue:
    value = _evaluate_comprehension_expression(
        engine,
        node,
        module=module,
        actor=actor,
        class_ref=class_ref,
        env=env,
        object_env=object_env,
        containing_env=containing_env,
        containing_objects=containing_objects,
        call_ordinals=call_ordinals,
    )
    _record_comprehension_control(
        engine,
        value,
        node,
        kind=kind,
        module=module,
        actor=actor,
        call_ordinals=call_ordinals,
    )
    return value


def _evaluate_comprehension_expression(
    engine: ExpressionEngine,
    node: ast.expr,
    *,
    module: str,
    actor: str,
    class_ref: str | None,
    env: dict[str, FlowValue],
    object_env: dict[str, set[str]],
    containing_env: dict[str, FlowValue],
    containing_objects: dict[str, set[str]],
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
    for name in _containing_walrus_names(node):
        containing_env[name] = containing_env.get(name, FlowValue()).merged(
            env.get(name, FlowValue())
        )
        object_types = containing_objects.get(name, set()) | object_env.get(name, set())
        if object_types:
            containing_objects[name] = object_types
        else:
            containing_objects.pop(name, None)
    return value


def _containing_walrus_names(node: ast.expr) -> set[str]:
    names: set[str] = set()

    class WalrusVisitor(ast.NodeVisitor):
        def visit_NamedExpr(self, item: ast.NamedExpr) -> None:
            names.add(item.target.id)
            self.visit(item.value)

        def visit_Lambda(self, item: ast.Lambda) -> None:
            return

        def visit_ListComp(self, item: ast.ListComp) -> None:
            return

        def visit_SetComp(self, item: ast.SetComp) -> None:
            return

        def visit_GeneratorExp(self, item: ast.GeneratorExp) -> None:
            return

        def visit_DictComp(self, item: ast.DictComp) -> None:
            return

    WalrusVisitor().visit(node)
    return names


def _record_comprehension_control(
    engine: ExpressionEngine,
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


def _lambda_origins(
    engine: ExpressionEngine,
    node: ast.Lambda,
    *,
    module: str,
    actor: str,
    class_ref: str | None,
    env: dict[str, FlowValue],
    object_env: dict[str, set[str]],
    call_ordinals: Mapping[int, int],
) -> FlowValue:
    result = FlowValue()
    defaults = [*node.args.defaults]
    defaults.extend(default for default in node.args.kw_defaults if default is not None)
    for default in defaults:
        result = result.merged(
            engine._eval(
                default,
                module=module,
                actor=actor,
                class_ref=class_ref,
                env=env,
                object_env=object_env,
                call_ordinals=call_ordinals,
            )
        )
    shadowed = {
        argument.arg
        for argument in [
            *node.args.posonlyargs,
            *node.args.args,
            *node.args.kwonlyargs,
        ]
    }
    if node.args.vararg is not None:
        shadowed.add(node.args.vararg.arg)
    if node.args.kwarg is not None:
        shadowed.add(node.args.kwarg.arg)

    class LambdaBindingVisitor(ast.NodeVisitor):
        def visit_NamedExpr(self, item: ast.NamedExpr) -> None:
            shadowed.update(_assignment_target_names(item.target))
            self.visit(item.value)

        def visit_Lambda(self, item: ast.Lambda) -> None:
            return

    LambdaBindingVisitor().visit(node.body)

    class CaptureVisitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.result = FlowValue()
            self.shadowed = set(shadowed)

        def visit_Name(self, item: ast.Name) -> None:
            if isinstance(item.ctx, ast.Load) and item.id not in self.shadowed:
                self.result = self.result.merged(env.get(item.id, FlowValue()))

        def visit_Attribute(self, item: ast.Attribute) -> None:
            root = _attribute_root(item)
            if root is not None and root.id not in self.shadowed:
                self.result = self.result.merged(
                    engine._eval(
                        item,
                        module=module,
                        actor=actor,
                        class_ref=class_ref,
                        env=env,
                        object_env=object_env,
                        call_ordinals=call_ordinals,
                    )
                )
            self.generic_visit(item)

        def visit_Lambda(self, item: ast.Lambda) -> None:
            return

        def _visit_comprehension(
            self,
            generators: list[ast.comprehension],
            values: list[ast.expr],
        ) -> None:
            previous = set(self.shadowed)
            for generator in generators:
                self.visit(generator.iter)
                self.shadowed.update(_assignment_target_names(generator.target))
                for condition in generator.ifs:
                    self.visit(condition)
            for value in values:
                self.visit(value)
            self.shadowed = previous

        def visit_ListComp(self, item: ast.ListComp) -> None:
            self._visit_comprehension(item.generators, [item.elt])

        def visit_SetComp(self, item: ast.SetComp) -> None:
            self._visit_comprehension(item.generators, [item.elt])

        def visit_GeneratorExp(self, item: ast.GeneratorExp) -> None:
            self._visit_comprehension(item.generators, [item.elt])

        def visit_DictComp(self, item: ast.DictComp) -> None:
            self._visit_comprehension(item.generators, [item.key, item.value])

    visitor = CaptureVisitor()
    visitor.visit(node.body)
    return result.merged(visitor.result)


def _assignment_target_names(target: ast.expr) -> set[str]:
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.Tuple, ast.List)):
        return {name for item in target.elts for name in _assignment_target_names(item)}
    if isinstance(target, ast.Starred):
        return _assignment_target_names(target.value)
    return set()


def _attribute_root(node: ast.Attribute) -> ast.Name | None:
    current: ast.expr = node.value
    while isinstance(current, ast.Attribute):
        current = current.value
    return current if isinstance(current, ast.Name) else None


def _slice_children(node: ast.expr) -> list[ast.expr]:
    if isinstance(node, ast.Slice):
        return [
            part for part in (node.lower, node.upper, node.step) if part is not None
        ]
    if isinstance(node, ast.expr):
        return [node]
    return []


__all__ = ["ExpressionEngine", "evaluate_generic_expression"]
