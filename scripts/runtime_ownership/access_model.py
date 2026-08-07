"""Value objects and source-shape helpers for runtime access discovery."""

from __future__ import annotations

import ast
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any, Literal

MAX_BINDING_PATHS_PER_RESOURCE = 64

READ_PATH_METHODS = frozenset(
    {
        "read_text",
        "read_bytes",
        "exists",
        "stat",
        "is_file",
        "is_dir",
        "iterdir",
        "glob",
        "rglob",
    }
)
WRITE_PATH_METHODS = frozenset(
    {
        "write_text",
        "write_bytes",
        "mkdir",
        "touch",
        "unlink",
        "chmod",
        "rename",
        "replace",
        "symlink_to",
        "hardlink_to",
    }
)
PATH_TRANSFORMS = frozenset(
    {"expanduser", "resolve", "absolute", "with_suffix", "joinpath"}
)


def _stable_id(prefix: str, payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(encoded).hexdigest()}"


def _module_name(path: str) -> str:
    relative = PurePosixPath(path).relative_to("src")
    parts = list(relative.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


@dataclass
class FlowValue:
    origins: dict[str, frozenset[tuple[str, ...]]] = field(default_factory=dict)
    object_types: set[str] = field(default_factory=set)
    overflowed: frozenset[str] = field(default_factory=frozenset)
    module_refs: set[str] = field(default_factory=set)
    call_targets: set[str] = field(default_factory=set)
    class_targets: set[str] = field(default_factory=set)
    unknown_callable: bool = False
    closure_instances: set[tuple[str, str]] = field(default_factory=set)
    structured_items: tuple[FlowValue, ...] | None = None

    def copy(self) -> FlowValue:
        return FlowValue(
            dict(self.origins),
            set(self.object_types),
            frozenset(self.overflowed),
            set(self.module_refs),
            set(self.call_targets),
            set(self.class_targets),
            self.unknown_callable,
            set(self.closure_instances),
            (
                tuple(item.copy() for item in self.structured_items)
                if self.structured_items is not None
                else None
            ),
        )

    def merged(self, other: FlowValue) -> FlowValue:
        result = self.copy()
        if self.structured_items is not None and other.structured_items is not None:
            if len(self.structured_items) == len(other.structured_items):
                result.structured_items = tuple(
                    left.merged(right)
                    for left, right in zip(
                        self.structured_items,
                        other.structured_items,
                        strict=True,
                    )
                )
            else:
                result.structured_items = None
        elif self.structured_items is None and other.structured_items is not None:
            result.structured_items = (
                tuple(item.copy() for item in other.structured_items)
                if _is_flow_bottom(self)
                else None
            )
        elif self.structured_items is not None and not _is_flow_bottom(other):
            result.structured_items = None
        overflowed = set(result.overflowed | other.overflowed)
        for resource_id, paths in other.origins.items():
            bounded, truncated = _bounded_binding_paths(
                result.origins.get(resource_id, frozenset()) | paths
            )
            result.origins[resource_id] = bounded
            if truncated:
                overflowed.add(resource_id)
        result.object_types.update(other.object_types)
        result.module_refs.update(other.module_refs)
        result.call_targets.update(other.call_targets)
        result.class_targets.update(other.class_targets)
        result.closure_instances.update(other.closure_instances)
        result.unknown_callable |= other.unknown_callable
        result.overflowed = frozenset(overflowed)
        return result

    def bound(self, step: str) -> FlowValue:
        origins: dict[str, frozenset[tuple[str, ...]]] = {}
        overflowed = set(self.overflowed)
        for resource_id, paths in self.origins.items():
            bounded, truncated = _bounded_binding_paths(
                frozenset(chain if step in chain else (*chain, step) for chain in paths)
            )
            origins[resource_id] = bounded
            if truncated:
                overflowed.add(resource_id)
        return FlowValue(
            origins,
            set(self.object_types),
            frozenset(overflowed),
            set(self.module_refs),
            set(self.call_targets),
            set(self.class_targets),
            self.unknown_callable,
            set(self.closure_instances),
            (
                tuple(item.bound(step) for item in self.structured_items)
                if self.structured_items is not None
                else None
            ),
        )

    @property
    def has_origins(self) -> bool:
        return bool(self.origins)

    def partition_call_cycles(self, *, target: str) -> tuple[FlowValue, FlowValue]:
        safe: dict[str, frozenset[tuple[str, ...]]] = {}
        cyclic: dict[str, frozenset[tuple[str, ...]]] = {}
        for resource_id, chains in self.origins.items():
            safe_chains: set[tuple[str, ...]] = set()
            cyclic_chains: set[tuple[str, ...]] = set()
            for chain in chains:
                destination = (
                    cyclic_chains
                    if target in _active_call_targets(chain)
                    else safe_chains
                )
                destination.add(chain)
            if safe_chains:
                safe[resource_id] = frozenset(safe_chains)
            if cyclic_chains:
                cyclic[resource_id] = frozenset(cyclic_chains)
        object_types = set(self.object_types)
        module_refs = set(self.module_refs)
        call_targets = set(self.call_targets)
        class_targets = set(self.class_targets)
        safe_items: tuple[FlowValue, ...] | None = None
        cyclic_items: tuple[FlowValue, ...] | None = None
        if self.structured_items is not None:
            partitioned_items = tuple(
                item.partition_call_cycles(target=target)
                for item in self.structured_items
            )
            safe_items = tuple(item[0] for item in partitioned_items)
            cyclic_items = tuple(item[1] for item in partitioned_items)
        return (
            FlowValue(
                safe,
                set(object_types),
                frozenset(self.overflowed & safe.keys()),
                set(module_refs),
                set(call_targets),
                set(class_targets),
                self.unknown_callable,
                set(self.closure_instances),
                safe_items,
            ),
            FlowValue(
                cyclic,
                set(object_types),
                frozenset(self.overflowed & cyclic.keys()),
                set(module_refs),
                set(call_targets),
                set(class_targets),
                self.unknown_callable,
                set(self.closure_instances),
                cyclic_items,
            ),
        )


def _is_flow_bottom(value: FlowValue) -> bool:
    return not (
        value.origins
        or value.object_types
        or value.overflowed
        or value.module_refs
        or value.call_targets
        or value.class_targets
        or value.unknown_callable
        or value.closure_instances
        or value.structured_items is not None
    )


def _bounded_binding_paths(
    paths: frozenset[tuple[str, ...]],
) -> tuple[frozenset[tuple[str, ...]], bool]:
    """Keep provenance finite while retaining distinct shortest bindings."""

    ordered = sorted(paths, key=lambda chain: (len(chain), chain))
    return (
        frozenset(ordered[:MAX_BINDING_PATHS_PER_RESOURCE]),
        len(ordered) > MAX_BINDING_PATHS_PER_RESOURCE,
    )


def _active_call_targets(chain: tuple[str, ...]) -> frozenset[str]:
    active: list[str] = []
    for step in chain:
        if step.startswith("call:") and "->" in step:
            target_and_parameter = step.split("->", 1)[1].split("|", 1)[0]
            active.append(target_and_parameter.rsplit(":", 1)[0])
        elif step.startswith("result:"):
            completed = step.removeprefix("result:")
            for index in range(len(active) - 1, -1, -1):
                if active[index] == completed:
                    active.pop(index)
                    break
    return frozenset(active)


@dataclass(frozen=True)
class FunctionInfo:
    ref: str
    module: str
    path: str
    qualname: str
    parent_ref: str | None
    class_ref: str | None
    local_names: frozenset[str]
    global_names: frozenset[str]
    nonlocal_names: frozenset[str]
    referenced_names: frozenset[str]
    node: ast.FunctionDef | ast.AsyncFunctionDef
    parameters: tuple[str, ...]
    defaults: Mapping[str, ast.expr]
    call_ordinals: Mapping[int, int]


def _class_definition_ref(
    *,
    module: str,
    actor: str,
    enclosing_class_ref: str | None,
    name: str,
) -> str:
    if actor == f"{module}:<module>":
        return f"{module}:{name}"
    if actor.endswith(".<classbody>") and enclosing_class_ref is not None:
        return f"{enclosing_class_ref}.{name}"
    return f"{actor}.<locals>.{name}"


@dataclass(frozen=True)
class RawAccess:
    resource_id: str
    actor: str
    sink_actor: str
    mode: str
    operation: str
    sink: str
    binding_chain: tuple[str, ...]
    path: str
    line: int
    structural_ordinal: int


@dataclass(frozen=True)
class RawEscape:
    resource_id: str
    actor: str
    operation: str
    sink: str
    reason: str
    binding_chain: tuple[str, ...]
    path: str
    line: int
    structural_ordinal: int


def _call_ordinals(
    node: ast.AST,
    *,
    evaluate_annotations: bool | None = None,
) -> dict[int, int]:
    calls: list[ast.Call] = []
    if evaluate_annotations is None:
        evaluate_annotations = not (
            isinstance(node, ast.Module)
            and any(
                isinstance(statement, ast.ImportFrom)
                and statement.module == "__future__"
                and any(alias.name == "annotations" for alias in statement.names)
                for statement in node.body
            )
        )

    class ScopedCallVisitor(ast.NodeVisitor):
        def visit_Call(self, item: ast.Call) -> None:
            calls.append(item)
            self.generic_visit(item)

        def visit_FunctionDef(self, item: ast.FunctionDef) -> None:
            if item is node:
                for statement in item.body:
                    self.visit(statement)
            else:
                self._visit_function_header(item)

        def visit_AsyncFunctionDef(self, item: ast.AsyncFunctionDef) -> None:
            if item is node:
                for statement in item.body:
                    self.visit(statement)
            else:
                self._visit_function_header(item)

        def visit_ClassDef(self, item: ast.ClassDef) -> None:
            self._visit_class_header(item)
            if item is node:
                for statement in item.body:
                    self.visit(statement)

        def visit_Lambda(self, item: ast.Lambda) -> None:
            for positional_default in item.args.defaults:
                self.visit(positional_default)
            for keyword_default in item.args.kw_defaults:
                if keyword_default is not None:
                    self.visit(keyword_default)

        def _visit_function_header(
            self,
            item: ast.FunctionDef | ast.AsyncFunctionDef,
        ) -> None:
            for decorator in item.decorator_list:
                self.visit(decorator)
            for positional_default in item.args.defaults:
                self.visit(positional_default)
            for keyword_default in item.args.kw_defaults:
                if keyword_default is not None:
                    self.visit(keyword_default)
            if not evaluate_annotations:
                return
            arguments = [
                *item.args.posonlyargs,
                *item.args.args,
                *item.args.kwonlyargs,
            ]
            if item.args.vararg is not None:
                arguments.append(item.args.vararg)
            if item.args.kwarg is not None:
                arguments.append(item.args.kwarg)
            for argument in arguments:
                if argument.annotation is not None:
                    self.visit(argument.annotation)
            if item.returns is not None:
                self.visit(item.returns)

        def _visit_class_header(self, item: ast.ClassDef) -> None:
            for decorator in item.decorator_list:
                self.visit(decorator)
            for base in item.bases:
                self.visit(base)
            for keyword in item.keywords:
                self.visit(keyword.value)

    ScopedCallVisitor().visit(node)
    calls.sort(key=lambda item: (int(item.lineno), int(item.col_offset)))
    occurrences: dict[str, int] = {}
    ordinals: dict[int, int] = {}
    for call in calls:
        group = ast.dump(call.func, include_attributes=False)
        occurrences[group] = occurrences.get(group, 0) + 1
        ordinals[id(call)] = occurrences[group]
    return ordinals


def _function_parameters(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[tuple[str, ...], dict[str, ast.expr]]:
    positional = [*node.args.posonlyargs, *node.args.args]
    parameters = [argument.arg for argument in positional]
    parameters.extend(argument.arg for argument in node.args.kwonlyargs)
    if node.args.vararg is not None:
        parameters.append(node.args.vararg.arg)
    if node.args.kwarg is not None:
        parameters.append(node.args.kwarg.arg)
    defaults: dict[str, ast.expr] = {}
    if node.args.defaults:
        for argument, default in zip(
            positional[-len(node.args.defaults) :], node.args.defaults, strict=True
        ):
            defaults[argument.arg] = default
    for argument, keyword_default in zip(
        node.args.kwonlyargs, node.args.kw_defaults, strict=True
    ):
        if keyword_default is not None:
            defaults[argument.arg] = keyword_default
    return tuple(parameters), defaults


def _function_scope_bindings(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    parameters: tuple[str, ...],
) -> tuple[frozenset[str], frozenset[str], frozenset[str]]:
    local_names = set(parameters)
    global_names: set[str] = set()
    nonlocal_names: set[str] = set()

    class BindingVisitor(ast.NodeVisitor):
        def visit_Name(self, item: ast.Name) -> None:
            if isinstance(item.ctx, (ast.Store, ast.Del)):
                local_names.add(item.id)

        def visit_Global(self, item: ast.Global) -> None:
            global_names.update(item.names)

        def visit_Nonlocal(self, item: ast.Nonlocal) -> None:
            nonlocal_names.update(item.names)

        def visit_Import(self, item: ast.Import) -> None:
            for alias in item.names:
                local_names.add(alias.asname or alias.name.split(".")[0])

        def visit_ImportFrom(self, item: ast.ImportFrom) -> None:
            for alias in item.names:
                if alias.name != "*":
                    local_names.add(alias.asname or alias.name)

        def _visit_function_definition(
            self, item: ast.FunctionDef | ast.AsyncFunctionDef
        ) -> None:
            local_names.add(item.name)
            for expression in [
                *item.decorator_list,
                *item.args.defaults,
                *(value for value in item.args.kw_defaults if value is not None),
            ]:
                self.visit(expression)

        def visit_FunctionDef(self, item: ast.FunctionDef) -> None:
            self._visit_function_definition(item)

        def visit_AsyncFunctionDef(self, item: ast.AsyncFunctionDef) -> None:
            self._visit_function_definition(item)

        def visit_ClassDef(self, item: ast.ClassDef) -> None:
            local_names.add(item.name)
            for expression in [*item.decorator_list, *item.bases]:
                self.visit(expression)
            for keyword in item.keywords:
                self.visit(keyword.value)

        def visit_Lambda(self, item: ast.Lambda) -> None:
            for expression in [
                *item.args.defaults,
                *(value for value in item.args.kw_defaults if value is not None),
            ]:
                self.visit(expression)

        def visit_ExceptHandler(self, item: ast.ExceptHandler) -> None:
            if item.name is not None:
                local_names.add(item.name)
            if item.type is not None:
                self.visit(item.type)
            for statement in item.body:
                self.visit(statement)

        def visit_MatchAs(self, item: ast.MatchAs) -> None:
            if item.name is not None:
                local_names.add(item.name)
            if item.pattern is not None:
                self.visit(item.pattern)

        def visit_MatchStar(self, item: ast.MatchStar) -> None:
            if item.name is not None:
                local_names.add(item.name)

        def visit_MatchMapping(self, item: ast.MatchMapping) -> None:
            if item.rest is not None:
                local_names.add(item.rest)
            self.generic_visit(item)

        def _visit_comprehension(
            self,
            generators: list[ast.comprehension],
            values: list[ast.expr],
        ) -> None:
            for generator in generators:
                self.visit(generator.iter)
                for condition in generator.ifs:
                    self.visit(condition)
            for value in values:
                self.visit(value)

        def visit_ListComp(self, item: ast.ListComp) -> None:
            self._visit_comprehension(item.generators, [item.elt])

        def visit_SetComp(self, item: ast.SetComp) -> None:
            self._visit_comprehension(item.generators, [item.elt])

        def visit_GeneratorExp(self, item: ast.GeneratorExp) -> None:
            self._visit_comprehension(item.generators, [item.elt])

        def visit_DictComp(self, item: ast.DictComp) -> None:
            self._visit_comprehension(item.generators, [item.key, item.value])

    visitor = BindingVisitor()
    for statement in node.body:
        visitor.visit(statement)
    local_names.difference_update(global_names | nonlocal_names)
    return (
        frozenset(local_names),
        frozenset(global_names),
        frozenset(nonlocal_names),
    )


def _collect_functions(
    module: str, path: str, tree: ast.Module
) -> tuple[dict[str, FunctionInfo], dict[str, str]]:
    functions: dict[str, FunctionInfo] = {}
    classes: dict[str, str] = {}
    evaluate_annotations = not any(
        isinstance(statement, ast.ImportFrom)
        and statement.module == "__future__"
        and any(alias.name == "annotations" for alias in statement.names)
        for statement in tree.body
    )

    def nested_function_nodes(
        root: ast.AST,
    ) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
        if isinstance(root, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return [root]
        if isinstance(root, (ast.ClassDef, ast.Lambda)):
            return []
        nested: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
        for child in ast.iter_child_nodes(root):
            nested.extend(nested_function_nodes(child))
        return nested

    def add_function(
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        *,
        qualname: str,
        parent_ref: str | None,
        class_ref: str | None,
    ) -> None:
        parameters, defaults = _function_parameters(node)
        local_names, global_names, nonlocal_names = _function_scope_bindings(
            node, parameters
        )
        referenced_names = _function_referenced_names(node)
        ref = f"{module}:{qualname}"
        functions[ref] = FunctionInfo(
            ref=ref,
            module=module,
            path=path,
            qualname=qualname,
            parent_ref=parent_ref,
            class_ref=class_ref,
            local_names=local_names,
            global_names=global_names,
            nonlocal_names=nonlocal_names,
            referenced_names=referenced_names,
            node=node,
            parameters=parameters,
            defaults=defaults,
            call_ordinals=_call_ordinals(
                node,
                evaluate_annotations=evaluate_annotations,
            ),
        )
        for statement in node.body:
            for child in nested_function_nodes(statement):
                add_function(
                    child,
                    qualname=f"{qualname}.<locals>.{child.name}",
                    parent_ref=ref,
                    class_ref=None,
                )

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            add_function(
                node,
                qualname=node.name,
                parent_ref=None,
                class_ref=None,
            )
        elif isinstance(node, ast.ClassDef):
            class_ref = f"{module}:{node.name}"
            classes[node.name] = class_ref
            for item in node.body:
                if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                add_function(
                    item,
                    qualname=f"{node.name}.{item.name}",
                    parent_ref=None,
                    class_ref=class_ref,
                )
    return functions, classes


def _function_referenced_names(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> frozenset[str]:
    names: set[str] = set()

    class ReferenceVisitor(ast.NodeVisitor):
        def visit_Name(self, item: ast.Name) -> None:
            if isinstance(item.ctx, ast.Load):
                names.add(item.id)

        def visit_FunctionDef(self, item: ast.FunctionDef) -> None:
            return

        def visit_AsyncFunctionDef(self, item: ast.AsyncFunctionDef) -> None:
            return

        def visit_ClassDef(self, item: ast.ClassDef) -> None:
            return

        def visit_Lambda(self, item: ast.Lambda) -> None:
            return

    visitor = ReferenceVisitor()
    for statement in node.body:
        visitor.visit(statement)
    return frozenset(names)


def _import_tables(
    module: str, tree: ast.Module, *, is_package: bool = False
) -> tuple[dict[str, tuple[str, str]], dict[str, str]]:
    symbols: dict[str, tuple[str, str]] = {}
    modules: dict[str, str] = {}
    package = module.split(".") if is_package else module.split(".")[:-1]
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules[alias.asname or alias.name.split(".")[0]] = alias.name
        elif isinstance(node, ast.ImportFrom):
            prefix = (
                []
                if node.level == 0
                else package[: max(0, len(package) - node.level + 1)]
            )
            if node.module:
                prefix.extend(node.module.split("."))
            imported_module = ".".join(prefix)
            for alias in node.names:
                symbols[alias.asname or alias.name] = (imported_module, alias.name)
    return symbols, modules


OpenModeResult = (
    tuple[Literal["read", "write", "read_write"], str]
    | Literal["dynamic_open_mode", "invalid_open_mode"]
)


def _open_mode(node: ast.Call, *, mode_index: int) -> OpenModeResult:
    mode_node: ast.expr | None = None
    if len(node.args) > mode_index:
        mode_node = node.args[mode_index]
    for keyword in node.keywords:
        if keyword.arg == "mode":
            mode_node = keyword.value
    if mode_node is None:
        return "read", "r"
    if not isinstance(mode_node, ast.Constant) or not isinstance(mode_node.value, str):
        return "dynamic_open_mode"
    mode = mode_node.value
    if (
        not mode
        or any(character not in "rwaxbt+" for character in mode)
        or sum(mode.count(flag) for flag in "rwax") != 1
        or mode.count("+") > 1
        or mode.count("b") > 1
        or mode.count("t") > 1
        or ("b" in mode and "t" in mode)
    ):
        return "invalid_open_mode"
    if "+" in mode:
        return "read_write", mode
    if any(flag in mode for flag in "wax"):
        return "write", mode
    return "read", mode


__all__ = [
    "MAX_BINDING_PATHS_PER_RESOURCE",
    "PATH_TRANSFORMS",
    "READ_PATH_METHODS",
    "WRITE_PATH_METHODS",
    "FlowValue",
    "FunctionInfo",
    "RawAccess",
    "RawEscape",
    "_call_ordinals",
    "_collect_functions",
    "_import_tables",
    "_module_name",
    "_open_mode",
    "_stable_id",
]
