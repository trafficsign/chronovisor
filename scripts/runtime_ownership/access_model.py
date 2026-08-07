"""Value objects and source-shape helpers for runtime access discovery."""

from __future__ import annotations

import ast
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any, Literal

MAX_BINDING_PATHS_PER_RESOURCE = 64

STDLIB_BUILTINS_CALLS = frozenset({"open"})
STDLIB_DATACLASSES_CALLS = frozenset({"dataclass", "field"})
STDLIB_FCNTL_CALLS = frozenset({"flock"})
STDLIB_OS_CALLS = frozenset({"fdopen", "open", "rename", "replace"})
STDLIB_PATHLIB_CALLS = frozenset({"Path"})
STDLIB_SQLITE3_CALLS = frozenset({"connect"})
STDLIB_SQLITE3_TYPES = frozenset({"Connection", "Cursor"})
FCNTL_LOCK_FLAGS = frozenset({"LOCK_EX", "LOCK_NB", "LOCK_SH", "LOCK_UN"})
FCNTL_LOCK_FLAG_BITS = {
    "LOCK_SH": 1,
    "LOCK_EX": 2,
    "LOCK_NB": 4,
    "LOCK_UN": 8,
}
FCNTL_LOCK_MASK_OBJECT_PREFIX = "stdlib-fcntl-lock-mask:"
FCNTL_UNRESOLVED_LOCK_OPERATION_OBJECT_TYPE = (
    "stdlib-fcntl-unresolved-lock-operation"
)
OS_OPEN_ACCESS_FLAGS = frozenset({"O_RDONLY", "O_WRONLY", "O_RDWR"})
OS_OPEN_MODIFIER_FLAGS = frozenset(
    {
        "O_APPEND",
        "O_CLOEXEC",
        "O_CREAT",
        "O_DIRECTORY",
        "O_DSYNC",
        "O_EXCL",
        "O_NOFOLLOW",
        "O_NONBLOCK",
        "O_SYNC",
        "O_TMPFILE",
        "O_TRUNC",
    }
)
OS_FLAG_OBJECT_PREFIX = "stdlib-os-flag:"
OS_FD_OBJECT_TYPE = "stdlib-os-file-descriptor"
FILE_HANDLE_OBJECT_TYPE = "stdlib-file-handle"
FILE_BOUND_CLOSE_OBJECT_TYPE = "stdlib-file-bound-close"
FILE_BOUND_FILENO_OBJECT_TYPE = "stdlib-file-bound-fileno"
FILE_UNKNOWN_ATTRIBUTE_OBJECT_TYPE = "stdlib-file-unknown-attribute"
UNRESOLVED_RUNTIME_OBJECT_ALTERNATIVE_TYPE = (
    "stdlib-unresolved-runtime-object-alternative"
)
SQLITE_CONNECTION_OBJECT_TYPE = "stdlib-sqlite3-connection"
SQLITE_CURSOR_OBJECT_TYPE = "stdlib-sqlite3-cursor"
SQLITE_HANDLE_OBJECT_TYPES = frozenset(
    {SQLITE_CONNECTION_OBJECT_TYPE, SQLITE_CURSOR_OBJECT_TYPE}
)
SQLITE_UNKNOWN_ATTRIBUTE_OBJECT_TYPE = "stdlib-sqlite3-unknown-attribute"
SQLITE_TYPE_OBJECT_PREFIX = "stdlib-sqlite3-type:"
STDLIB_CALL_TARGET_PREFIX = "stdlib-call-target:"
STDLIB_MODULE_MUTATION_PREFIX = "stdlib-module-attribute-mutated:"
STDLIB_MODULE_WILDCARD_ATTRIBUTE = "*"
STDLIB_MODULE_STATE_PREFIX = "\0stdlib-module-state:"
SUPPORTED_STDLIB_MODULES = frozenset(
    {"builtins", "dataclasses", "fcntl", "os", "pathlib", "sqlite3"}
)
PATH_OBJECT_TYPE = "stdlib-pathlib-path"
PATH_STRING_OBJECT_TYPE = "stdlib-pathlib-path-representation"
PATH_BOUND_TRANSFORM_PREFIX = "stdlib-pathlib-bound-transform:"
DATACLASS_INIT_VAR_MARKER = "stdlib-dataclasses-init-var"
DATACLASS_KW_ONLY_MARKER = "stdlib-dataclasses-kw-only"
SQLITE_CURSOR_SCALAR_ATTRIBUTES = frozenset(
    {"arraysize", "description", "lastrowid", "rowcount"}
)
SQLITE_CONNECTION_SCALAR_ATTRIBUTES = frozenset(
    {
        "autocommit",
        "in_transaction",
        "isolation_level",
        "row_factory",
        "text_factory",
        "total_changes",
    }
)

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
    {
        "expanduser",
        "resolve",
        "absolute",
        "with_suffix",
        "with_name",
        "joinpath",
    }
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


def _normalized_repo_relative_path(path: str) -> str:
    raw = path.replace("\\", "/")
    if PurePosixPath(raw).is_absolute():
        raise ValueError(f"runtime syntax site path must be repo-relative: {path}")
    parts: list[str] = []
    for part in raw.split("/"):
        if part in {"", "."}:
            continue
        if part == "..":
            if not parts:
                raise ValueError(
                    f"runtime syntax site path escapes repository root: {path}"
                )
            parts.pop()
            continue
        parts.append(part)
    if not parts:
        raise ValueError("runtime syntax site path must name a repository file")
    return "/".join(parts)


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
    runtime_object_ids: set[str] = field(default_factory=set)
    runtime_close_ids: set[str] = field(default_factory=set)
    runtime_descriptor_ids: set[str] = field(default_factory=set)
    instance_ids: set[str] = field(default_factory=set)
    attribute_values: dict[str, FlowValue] = field(default_factory=dict)
    attribute_values_complete: bool = False
    attribute_values_ambiguous: bool = False

    def copy(self) -> FlowValue:
        return FlowValue(
            origins=dict(self.origins),
            object_types=set(self.object_types),
            overflowed=frozenset(self.overflowed),
            module_refs=set(self.module_refs),
            call_targets=set(self.call_targets),
            class_targets=set(self.class_targets),
            unknown_callable=self.unknown_callable,
            closure_instances=set(self.closure_instances),
            structured_items=(
                tuple(item.copy() for item in self.structured_items)
                if self.structured_items is not None
                else None
            ),
            runtime_object_ids=set(self.runtime_object_ids),
            runtime_close_ids=set(self.runtime_close_ids),
            runtime_descriptor_ids=set(self.runtime_descriptor_ids),
            instance_ids=set(self.instance_ids),
            attribute_values={
                name: value.copy() for name, value in self.attribute_values.items()
            },
            attribute_values_complete=self.attribute_values_complete,
            attribute_values_ambiguous=self.attribute_values_ambiguous,
        )

    def merged(self, other: FlowValue) -> FlowValue:
        self_bottom = _is_flow_bottom(self)
        other_bottom = _is_flow_bottom(other)
        if self_bottom:
            return other.copy()
        if other_bottom:
            return self.copy()
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
                if self_bottom
                else None
            )
        elif self.structured_items is not None and not other_bottom:
            result.structured_items = None
        self_has_attributes = bool(self.attribute_values) or (
            self.attribute_values_complete or self.attribute_values_ambiguous
        )
        other_has_attributes = bool(other.attribute_values) or (
            other.attribute_values_complete or other.attribute_values_ambiguous
        )
        if self_has_attributes and other_has_attributes:
            names = self.attribute_values.keys() | other.attribute_values.keys()
            merged_attributes: dict[str, FlowValue] = {}
            for name in names:
                left = self.attribute_values.get(name)
                right = other.attribute_values.get(name)
                if left is None:
                    assert right is not None
                    merged = right.copy()
                    merged.attribute_values_ambiguous = True
                elif right is None:
                    merged = left.copy()
                    merged.attribute_values_ambiguous = True
                else:
                    merged = left.merged(right)
                    if left.has_analysis_state != right.has_analysis_state:
                        merged.attribute_values_ambiguous = True
                merged_attributes[name] = merged
            result.attribute_values = merged_attributes
            same_shape = self.attribute_values.keys() == other.attribute_values.keys()
            result.attribute_values_complete = (
                self.attribute_values_complete
                and other.attribute_values_complete
                and same_shape
            )
            result.attribute_values_ambiguous = (
                self.attribute_values_ambiguous
                or other.attribute_values_ambiguous
                or not same_shape
                or self.attribute_values_complete != other.attribute_values_complete
            )
        elif self_has_attributes or other_has_attributes:
            source = self if self_has_attributes else other
            result.attribute_values = {
                name: value.copy() for name, value in source.attribute_values.items()
            }
            result.attribute_values_complete = False
            result.attribute_values_ambiguous = True
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
        result.runtime_object_ids.update(other.runtime_object_ids)
        result.runtime_close_ids.update(other.runtime_close_ids)
        result.runtime_descriptor_ids.update(other.runtime_descriptor_ids)
        result.instance_ids.update(other.instance_ids)
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
            origins=origins,
            object_types=set(self.object_types),
            overflowed=frozenset(overflowed),
            module_refs=set(self.module_refs),
            call_targets=set(self.call_targets),
            class_targets=set(self.class_targets),
            unknown_callable=self.unknown_callable,
            closure_instances=set(self.closure_instances),
            structured_items=(
                tuple(item.bound(step) for item in self.structured_items)
                if self.structured_items is not None
                else None
            ),
            runtime_object_ids=set(self.runtime_object_ids),
            runtime_close_ids=set(self.runtime_close_ids),
            runtime_descriptor_ids=set(self.runtime_descriptor_ids),
            instance_ids=set(self.instance_ids),
            attribute_values={
                name: value.bound(step)
                for name, value in self.attribute_values.items()
            },
            attribute_values_complete=self.attribute_values_complete,
            attribute_values_ambiguous=self.attribute_values_ambiguous,
        )

    @property
    def has_origins(self) -> bool:
        return bool(self.origins)

    @property
    def has_analysis_state(self) -> bool:
        """Return whether this value carries direct or recursively nested state."""

        return not _is_flow_bottom(self)

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
        safe_attributes: dict[str, FlowValue] = {}
        cyclic_attributes: dict[str, FlowValue] = {}
        for name, value in self.attribute_values.items():
            safe_value, cyclic_value = value.partition_call_cycles(target=target)
            safe_attributes[name] = safe_value
            cyclic_attributes[name] = cyclic_value
        return (
            FlowValue(
                origins=safe,
                object_types=set(object_types),
                overflowed=frozenset(self.overflowed & safe.keys()),
                module_refs=set(module_refs),
                call_targets=set(call_targets),
                class_targets=set(class_targets),
                unknown_callable=self.unknown_callable,
                closure_instances=set(self.closure_instances),
                structured_items=safe_items,
                runtime_object_ids=set(self.runtime_object_ids),
                runtime_close_ids=set(self.runtime_close_ids),
                runtime_descriptor_ids=set(self.runtime_descriptor_ids),
                instance_ids=set(self.instance_ids),
                attribute_values=safe_attributes,
                attribute_values_complete=self.attribute_values_complete,
                attribute_values_ambiguous=self.attribute_values_ambiguous,
            ),
            FlowValue(
                origins=cyclic,
                object_types=set(object_types),
                overflowed=frozenset(self.overflowed & cyclic.keys()),
                module_refs=set(module_refs),
                call_targets=set(call_targets),
                class_targets=set(class_targets),
                unknown_callable=self.unknown_callable,
                closure_instances=set(self.closure_instances),
                structured_items=cyclic_items,
                runtime_object_ids=set(self.runtime_object_ids),
                runtime_close_ids=set(self.runtime_close_ids),
                runtime_descriptor_ids=set(self.runtime_descriptor_ids),
                instance_ids=set(self.instance_ids),
                attribute_values=cyclic_attributes,
                attribute_values_complete=self.attribute_values_complete,
                attribute_values_ambiguous=self.attribute_values_ambiguous,
            ),
        )


@dataclass(frozen=True)
class DataclassFieldInfo:
    """Exact generated-constructor metadata for one declared dataclass field."""

    name: str
    default: FlowValue | None = None
    default_factory: FlowValue | None = None
    default_factory_expression: ast.expr | None = None
    declared_class_targets: frozenset[str] = frozenset()
    init: bool = True
    keyword_only: bool = False
    init_var: bool = False


@dataclass(frozen=True)
class DataclassInfo:
    """Analyzer-only dataclass shape; it never changes registry ownership."""

    fields: tuple[DataclassFieldInfo, ...]
    generated_init: bool
    explicit_init: bool
    shape_ambiguous: bool = False
    post_init_target: str | None = None
    post_init_unknown: bool = False


def mark_attribute_alternative_ambiguity(
    value: FlowValue,
    alternatives: Sequence[FlowValue],
) -> FlowValue:
    """Fail closed when a join can select a value without the tracked shape."""

    has_attributes = bool(value.attribute_values) or value.attribute_values_complete
    if not has_attributes:
        return value
    if any(
        not (
            alternative.attribute_values
            or alternative.attribute_values_complete
            or alternative.attribute_values_ambiguous
        )
        for alternative in alternatives
    ):
        value.attribute_values_complete = False
        value.attribute_values_ambiguous = True
    return value


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
        or value.runtime_object_ids
        or value.runtime_close_ids
        or value.runtime_descriptor_ids
        or value.instance_ids
        or value.attribute_values
        or value.attribute_values_complete
        or value.attribute_values_ambiguous
    )


def is_path_receiver(value: FlowValue) -> bool:
    """Return whether an origin-bearing value is still a Path proxy."""

    non_path_types = SQLITE_HANDLE_OBJECT_TYPES | {
        FILE_BOUND_CLOSE_OBJECT_TYPE,
        FILE_BOUND_FILENO_OBJECT_TYPE,
        FILE_HANDLE_OBJECT_TYPE,
        FILE_UNKNOWN_ATTRIBUTE_OBJECT_TYPE,
        OS_FD_OBJECT_TYPE,
        SQLITE_UNKNOWN_ATTRIBUTE_OBJECT_TYPE,
        PATH_STRING_OBJECT_TYPE,
    }
    return (
        value.has_origins
        and value.object_types.isdisjoint(non_path_types)
        and not value.attribute_values
        and not value.attribute_values_complete
        and not value.attribute_values_ambiguous
    )


def is_exact_path_receiver(value: FlowValue) -> bool:
    """Require a Path receiver with no non-Path runtime alternative."""

    return (
        value.has_origins
        and value.object_types <= {PATH_OBJECT_TYPE}
        and not value.module_refs
        and not value.call_targets
        and not value.class_targets
        and not value.unknown_callable
        and not value.closure_instances
        and value.structured_items is None
        and not value.runtime_object_ids
        and not value.runtime_close_ids
        and not value.runtime_descriptor_ids
        and not value.attribute_values
        and not value.attribute_values_complete
        and not value.attribute_values_ambiguous
    )


def is_exact_path_constructor_argument(value: FlowValue) -> bool:
    """Require an uncontaminated locator, Path, or exact Path representation."""

    return (
        value.has_origins
        and value.object_types <= {PATH_OBJECT_TYPE, PATH_STRING_OBJECT_TYPE}
        and not value.module_refs
        and not value.call_targets
        and not value.class_targets
        and not value.unknown_callable
        and not value.closure_instances
        and value.structured_items is None
        and not value.runtime_object_ids
        and not value.runtime_close_ids
        and not value.runtime_descriptor_ids
        and not value.attribute_values
        and not value.attribute_values_complete
        and not value.attribute_values_ambiguous
    )


def stdlib_module_dict_reference(
    expression: ast.expr,
    env: Mapping[str, FlowValue],
) -> FlowValue | None:
    """Resolve a precise ``module.__dict__`` or unshadowed ``vars(module)``."""

    module_expression: ast.expr | None = None
    if isinstance(expression, ast.Attribute) and expression.attr == "__dict__":
        module_expression = expression.value
    elif (
        isinstance(expression, ast.Call)
        and isinstance(expression.func, ast.Name)
        and expression.func.id == "vars"
        and "vars" not in env
        and len(expression.args) == 1
        and not expression.keywords
    ):
        module_expression = expression.args[0]
    if not isinstance(module_expression, ast.Name):
        return None
    base = env.get(module_expression.id, FlowValue())
    module_ref = precise_stdlib_module_name(base)
    if module_ref not in SUPPORTED_STDLIB_MODULES:
        return None
    return base


def is_exact_runtime_object(
    value: FlowValue,
    *,
    object_type: str,
    binding_step: str,
) -> bool:
    """Require one uncontaminated runtime object kind and its provenance tag."""

    return (
        value.has_origins
        and value.object_types == {object_type}
        and not value.module_refs
        and not value.call_targets
        and not value.class_targets
        and not value.unknown_callable
        and not value.closure_instances
        and value.structured_items is None
        and all(
            binding_step in chain
            for chains in value.origins.values()
            for chain in chains
        )
    )


def sqlite_handle_kind(value: FlowValue) -> str | None:
    """Return the precise SQLite handle kind carried by a flow value."""

    if is_exact_runtime_object(
        value,
        object_type=SQLITE_CONNECTION_OBJECT_TYPE,
        binding_step="handle:sqlite.connection",
    ):
        return "connection"
    if is_exact_runtime_object(
        value,
        object_type=SQLITE_CURSOR_OBJECT_TYPE,
        binding_step="handle:sqlite.cursor",
    ):
        return "cursor"
    return None


def file_handle_kind(value: FlowValue) -> str | None:
    """Return the precise synchronous file-handle kind carried by a value."""

    if is_exact_runtime_object(
        value,
        object_type=FILE_HANDLE_OBJECT_TYPE,
        binding_step="handle:file",
    ):
        return "file"
    return None


def is_exact_os_fd(value: FlowValue) -> bool:
    """Return whether *value* is an uncontaminated tracked OS descriptor."""

    return is_exact_runtime_object(
        value,
        object_type=OS_FD_OBJECT_TYPE,
        binding_step="handle:os.fd",
    )


def is_exact_flock_descriptor(value: FlowValue) -> bool:
    """Accept exact file handles, exact FDs, and joins of those alternatives."""

    if (
        not value.has_origins
        or not value.object_types
        or not value.object_types
        <= {FILE_HANDLE_OBJECT_TYPE, OS_FD_OBJECT_TYPE}
        or value.module_refs
        or value.call_targets
        or value.class_targets
        or value.unknown_callable
        or value.closure_instances
        or value.structured_items is not None
    ):
        return False
    return all(
        "handle:file" in chain or "handle:os.fd" in chain
        for chains in value.origins.values()
        for chain in chains
    )


def tag_file_handle(
    value: FlowValue,
    *,
    identity: str | None = None,
    wraps_fd: bool = False,
    closefd: bool = True,
) -> FlowValue:
    """Replace locator/descriptor tags with the exact file-handle tag."""

    tagged = value.bound("handle:file")
    tagged.object_types = {FILE_HANDLE_OBJECT_TYPE}
    if identity is not None:
        tagged.runtime_object_ids = {identity}
        descriptor_ids = (
            value.runtime_descriptor_ids or value.runtime_object_ids
            if wraps_fd
            else {identity}
        )
        tagged.runtime_descriptor_ids = set(descriptor_ids)
        tagged.runtime_close_ids = {identity}
        if closefd:
            tagged.runtime_close_ids.update(descriptor_ids)
    return tagged


def tag_os_fd(value: FlowValue, *, identity: str | None = None) -> FlowValue:
    """Replace file-handle/locator tags with the exact OS descriptor tag."""

    tagged = value.bound("handle:os.fd")
    tagged.object_types = {OS_FD_OBJECT_TYPE}
    if identity is not None:
        tagged.runtime_object_ids = {identity}
        tagged.runtime_close_ids = {identity}
        tagged.runtime_descriptor_ids = {identity}
    else:
        descriptor_ids = value.runtime_descriptor_ids or value.runtime_object_ids
        tagged.runtime_object_ids = set(descriptor_ids)
        tagged.runtime_close_ids = set(descriptor_ids)
        tagged.runtime_descriptor_ids = set(descriptor_ids)
    return tagged


def has_file_descriptor_object(value: FlowValue) -> bool:
    """Return whether a value carries a file-handle or OS-FD alternative."""

    return bool(
        value.object_types.intersection(
            {FILE_HANDLE_OBJECT_TYPE, OS_FD_OBJECT_TYPE}
        )
    )


def contaminate_runtime_objects(
    env: dict[str, FlowValue],
    object_env: dict[str, set[str]],
    values: Sequence[FlowValue],
) -> set[str]:
    """Invalidate exact file/FD aliases that may have been closed or mutated."""

    object_ids = set().union(
        *(
            value.runtime_close_ids or value.runtime_object_ids
            for value in values
            if has_file_descriptor_object(value)
            or value.object_types.intersection(
                {
                    FILE_BOUND_CLOSE_OBJECT_TYPE,
                    FILE_UNKNOWN_ATTRIBUTE_OBJECT_TYPE,
                }
            )
        )
    )
    if not object_ids:
        return set()

    def contaminate(candidate: FlowValue) -> FlowValue:
        contaminated = candidate.copy()
        if object_ids.intersection(
            candidate.runtime_object_ids | candidate.runtime_descriptor_ids
        ):
            contaminated.object_types.add(
                UNRESOLVED_RUNTIME_OBJECT_ALTERNATIVE_TYPE
            )
        contaminated.attribute_values = {
            attribute: contaminate(value)
            for attribute, value in candidate.attribute_values.items()
        }
        if candidate.structured_items is not None:
            contaminated.structured_items = tuple(
                contaminate(item) for item in candidate.structured_items
            )
        return contaminated

    for name, candidate in list(env.items()):
        contaminated = contaminate(candidate)
        if contaminated == candidate:
            continue
        env[name] = contaminated
        if contaminated.object_types:
            object_env[name] = set(contaminated.object_types)
        else:
            object_env.pop(name, None)
    return object_ids


def fcntl_lock_mask_value(flag: str) -> FlowValue:
    """Build the exact symbolic value for a supported ``fcntl`` lock flag."""

    return FlowValue(
        object_types={
            f"{FCNTL_LOCK_MASK_OBJECT_PREFIX}{FCNTL_LOCK_FLAG_BITS[flag]}"
        }
    )


def fcntl_lock_masks(value: FlowValue) -> frozenset[int] | None:
    """Decode exact symbolic alternatives without accepting raw integers."""

    if (
        not value.object_types
        or value.origins
        or value.module_refs
        or value.call_targets
        or value.class_targets
        or value.unknown_callable
        or value.closure_instances
        or value.structured_items is not None
    ):
        return None
    masks: set[int] = set()
    for object_type in value.object_types:
        if not object_type.startswith(FCNTL_LOCK_MASK_OBJECT_PREFIX):
            return None
        raw_mask = object_type.removeprefix(FCNTL_LOCK_MASK_OBJECT_PREFIX)
        try:
            masks.add(int(raw_mask))
        except ValueError:
            return None
    return frozenset(masks)


def has_fcntl_lock_mask(value: FlowValue) -> bool:
    """Return whether a value contains at least one symbolic lock alternative."""

    return any(
        object_type.startswith(FCNTL_LOCK_MASK_OBJECT_PREFIX)
        for object_type in value.object_types
    )


def combine_fcntl_lock_masks(
    left: FlowValue,
    right: FlowValue,
) -> FlowValue | None:
    """Apply bitwise OR to each exact alternative while preserving joins."""

    left_masks = fcntl_lock_masks(left)
    right_masks = fcntl_lock_masks(right)
    if left_masks is None or right_masks is None:
        return None
    return FlowValue(
        object_types={
            f"{FCNTL_LOCK_MASK_OBJECT_PREFIX}{left_mask | right_mask}"
            for left_mask in left_masks
            for right_mask in right_masks
        }
    ).bound("expression:fcntl.lock_bitor")


def tag_sqlite_handle(value: FlowValue, *, kind: str) -> FlowValue:
    tagged = value.bound(f"handle:sqlite.{kind}")
    tagged.object_types = {
        SQLITE_CONNECTION_OBJECT_TYPE
        if kind == "connection"
        else SQLITE_CURSOR_OBJECT_TYPE
    }
    return tagged


def project_sqlite_attribute(
    value: FlowValue,
    *,
    attribute: str,
) -> tuple[bool, FlowValue]:
    kind = sqlite_handle_kind(value)
    if kind == "cursor" and attribute == "connection":
        return (
            True,
            tag_sqlite_handle(
                value.bound("attribute:sqlite.cursor.connection"),
                kind="connection",
            ),
        )
    if kind == "cursor" and attribute in SQLITE_CURSOR_SCALAR_ATTRIBUTES:
        return True, FlowValue()
    if (
        kind == "connection"
        and attribute in SQLITE_CONNECTION_SCALAR_ATTRIBUTES
    ):
        return True, FlowValue()
    return False, FlowValue()


def stdlib_module_mutation_marker(module: str, attribute: str) -> str:
    return f"{STDLIB_MODULE_MUTATION_PREFIX}{module}:{attribute}"


def stdlib_call_target_marker(module: str, attribute: str) -> str:
    return f"{STDLIB_CALL_TARGET_PREFIX}{module}.{attribute}"


def stdlib_call_targets(value: FlowValue) -> frozenset[str]:
    return frozenset(
        object_type.removeprefix(STDLIB_CALL_TARGET_PREFIX)
        for object_type in value.object_types
        if object_type.startswith(STDLIB_CALL_TARGET_PREFIX)
    )


def stdlib_module_state_name(module: str) -> str:
    return f"{STDLIB_MODULE_STATE_PREFIX}{module}"


def stdlib_module_mutation_attributes(
    value: FlowValue,
    *,
    module: str,
) -> frozenset[str]:
    marker_prefix = f"{STDLIB_MODULE_MUTATION_PREFIX}{module}:"
    return frozenset(
        object_type.removeprefix(marker_prefix)
        for object_type in value.object_types
        if object_type.startswith(marker_prefix)
    )


def precise_stdlib_module_name(value: FlowValue) -> str | None:
    """Return the exact supported stdlib module identity, ignoring mutation tags."""

    if len(value.module_refs) != 1:
        return None
    module = next(iter(value.module_refs))
    if module not in SUPPORTED_STDLIB_MODULES:
        return None
    marker_prefix = f"{STDLIB_MODULE_MUTATION_PREFIX}{module}:"
    if (
        value.origins
        or value.call_targets
        or value.class_targets
        or value.unknown_callable
        or value.closure_instances
        or value.attribute_values
        or value.attribute_values_complete
        or value.attribute_values_ambiguous
        or any(
            not object_type.startswith(marker_prefix)
            for object_type in value.object_types
        )
    ):
        return None
    return module


def is_precise_stdlib_module(
    value: FlowValue,
    *,
    module: str,
    attribute: str,
) -> bool:
    return (
        precise_stdlib_module_name(value) == module
        and stdlib_module_mutation_marker(
            module, STDLIB_MODULE_WILDCARD_ATTRIBUTE
        )
        not in value.object_types
        and stdlib_module_mutation_marker(module, attribute)
        not in value.object_types
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
class SyntaxSite:
    """A line-independent executable syntax site with display-only evidence."""

    site_id: str
    scope: str
    kind: str
    syntax: str
    occurrence: int
    path: str
    line: int


def _collect_syntax_sites(
    trees: Mapping[str, ast.Module],
    paths: Mapping[str, str],
    function_refs_by_node: Mapping[int, str],
) -> dict[int, SyntaxSite]:
    candidates: list[tuple[ast.AST, str, str, int]] = []

    class ExecutableScopeVisitor(ast.NodeVisitor):
        def __init__(self, module: str) -> None:
            self.module = module
            self.path = _normalized_repo_relative_path(paths[module])
            self.scope = f"{module}:<module>"
            self.class_ref: str | None = None
            self.sequence = 0

        def visit(self, node: ast.AST) -> Any:
            if isinstance(node, (ast.stmt, ast.expr)):
                candidates.append((node, self.scope, self.path, self.sequence))
                self.sequence += 1
            return super().visit(node)

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self._visit_function(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self._visit_function(node)

        def _visit_function(
            self, node: ast.FunctionDef | ast.AsyncFunctionDef
        ) -> None:
            for expression in [
                *node.decorator_list,
                *node.args.defaults,
                *(value for value in node.args.kw_defaults if value is not None),
            ]:
                self.visit(expression)
            arguments = [
                *node.args.posonlyargs,
                *node.args.args,
                *node.args.kwonlyargs,
            ]
            if node.args.vararg is not None:
                arguments.append(node.args.vararg)
            if node.args.kwarg is not None:
                arguments.append(node.args.kwarg)
            for argument in arguments:
                if argument.annotation is not None:
                    self.visit(argument.annotation)
            if node.returns is not None:
                self.visit(node.returns)
            function_ref = function_refs_by_node.get(id(node))
            if function_ref is None:
                return
            previous_scope = self.scope
            previous_class_ref = self.class_ref
            self.scope = function_ref
            for statement in node.body:
                self.visit(statement)
            self.scope = previous_scope
            self.class_ref = previous_class_ref

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            for expression in [*node.decorator_list, *node.bases]:
                self.visit(expression)
            for keyword in node.keywords:
                self.visit(keyword.value)
            current_ref = _class_definition_ref(
                module=self.module,
                actor=self.scope,
                enclosing_class_ref=self.class_ref,
                name=node.name,
            )
            previous_scope = self.scope
            previous_class_ref = self.class_ref
            self.scope = f"{current_ref}.<classbody>"
            self.class_ref = current_ref
            for statement in node.body:
                self.visit(statement)
            self.scope = previous_scope
            self.class_ref = previous_class_ref

        def visit_Call(self, node: ast.Call) -> None:
            self.visit(node.func)
            for argument in node.args:
                self.visit(argument)
            for keyword in node.keywords:
                if keyword.arg == "default_factory" and isinstance(
                    keyword.value, ast.Lambda
                ):
                    self.visit(keyword.value.body)
                else:
                    self.visit(keyword.value)

        def visit_Lambda(self, node: ast.Lambda) -> None:
            for expression in [
                *node.args.defaults,
                *(value for value in node.args.kw_defaults if value is not None),
            ]:
                self.visit(expression)

    for module, tree in sorted(trees.items()):
        ExecutableScopeVisitor(module).visit(tree)

    grouped: dict[tuple[str, str, str, str], list[tuple[ast.AST, int]]] = {}
    for node, scope, path, sequence in candidates:
        kind = type(node).__name__.lower()
        syntax = ast.dump(
            node,
            annotate_fields=True,
            include_attributes=False,
        )
        grouped.setdefault((path, scope, kind, syntax), []).append((node, sequence))

    sites: dict[int, SyntaxSite] = {}
    for (path, scope, kind, syntax), nodes in sorted(grouped.items()):
        nodes.sort(
            key=lambda item: (
                int(getattr(item[0], "lineno", 0)),
                int(getattr(item[0], "col_offset", 0)),
                item[1],
            )
        )
        for occurrence, (node, _sequence) in enumerate(nodes, start=1):
            identity = {
                "path": path,
                "scope": scope,
                "kind": kind,
                "syntax": syntax,
                "occurrence": occurrence,
            }
            sites[id(node)] = SyntaxSite(
                site_id=_stable_id("runtime-site", identity),
                scope=scope,
                kind=kind,
                syntax=syntax,
                occurrence=occurrence,
                path=path,
                line=int(getattr(node, "lineno", 0)),
            )
    return sites


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
    site_id: str


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
    site_id: str


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


def open_mode_from_expression(mode_node: ast.expr | None) -> OpenModeResult:
    """Classify a statically supplied CPython text-open mode."""

    if mode_node is None:
        return "read", "r"
    if not isinstance(mode_node, ast.Constant):
        return "dynamic_open_mode"
    if not isinstance(mode_node.value, str):
        return "invalid_open_mode"
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


def _open_mode(node: ast.Call, *, mode_index: int) -> OpenModeResult:
    mode_node: ast.expr | None = None
    if len(node.args) > mode_index:
        mode_node = node.args[mode_index]
    for keyword in node.keywords:
        if keyword.arg == "mode":
            mode_node = keyword.value
    return open_mode_from_expression(mode_node)


__all__ = [
    "FCNTL_LOCK_FLAGS",
    "FCNTL_LOCK_FLAG_BITS",
    "FCNTL_LOCK_MASK_OBJECT_PREFIX",
    "FCNTL_UNRESOLVED_LOCK_OPERATION_OBJECT_TYPE",
    "FILE_BOUND_CLOSE_OBJECT_TYPE",
    "FILE_BOUND_FILENO_OBJECT_TYPE",
    "FILE_HANDLE_OBJECT_TYPE",
    "FILE_UNKNOWN_ATTRIBUTE_OBJECT_TYPE",
    "MAX_BINDING_PATHS_PER_RESOURCE",
    "OS_FD_OBJECT_TYPE",
    "OS_FLAG_OBJECT_PREFIX",
    "OS_OPEN_ACCESS_FLAGS",
    "OS_OPEN_MODIFIER_FLAGS",
    "PATH_TRANSFORMS",
    "READ_PATH_METHODS",
    "WRITE_PATH_METHODS",
    "FlowValue",
    "FunctionInfo",
    "RawAccess",
    "RawEscape",
    "SQLITE_CONNECTION_OBJECT_TYPE",
    "SQLITE_CONNECTION_SCALAR_ATTRIBUTES",
    "SQLITE_CURSOR_OBJECT_TYPE",
    "SQLITE_CURSOR_SCALAR_ATTRIBUTES",
    "SQLITE_HANDLE_OBJECT_TYPES",
    "SQLITE_TYPE_OBJECT_PREFIX",
    "SQLITE_UNKNOWN_ATTRIBUTE_OBJECT_TYPE",
    "SyntaxSite",
    "STDLIB_OS_CALLS",
    "STDLIB_BUILTINS_CALLS",
    "STDLIB_FCNTL_CALLS",
    "STDLIB_MODULE_MUTATION_PREFIX",
    "STDLIB_MODULE_STATE_PREFIX",
    "STDLIB_MODULE_WILDCARD_ATTRIBUTE",
    "STDLIB_CALL_TARGET_PREFIX",
    "STDLIB_SQLITE3_CALLS",
    "STDLIB_SQLITE3_TYPES",
    "SUPPORTED_STDLIB_MODULES",
    "UNRESOLVED_RUNTIME_OBJECT_ALTERNATIVE_TYPE",
    "_call_ordinals",
    "_collect_functions",
    "_collect_syntax_sites",
    "_import_tables",
    "_module_name",
    "_open_mode",
    "_stable_id",
    "combine_fcntl_lock_masks",
    "contaminate_runtime_objects",
    "fcntl_lock_mask_value",
    "fcntl_lock_masks",
    "has_fcntl_lock_mask",
    "has_file_descriptor_object",
    "file_handle_kind",
    "is_exact_flock_descriptor",
    "is_exact_os_fd",
    "is_exact_path_receiver",
    "is_path_receiver",
    "is_exact_runtime_object",
    "is_precise_stdlib_module",
    "project_sqlite_attribute",
    "precise_stdlib_module_name",
    "open_mode_from_expression",
    "sqlite_handle_kind",
    "stdlib_call_target_marker",
    "stdlib_call_targets",
    "stdlib_module_dict_reference",
    "stdlib_module_mutation_attributes",
    "stdlib_module_mutation_marker",
    "stdlib_module_state_name",
    "tag_file_handle",
    "tag_os_fd",
    "tag_sqlite_handle",
]
