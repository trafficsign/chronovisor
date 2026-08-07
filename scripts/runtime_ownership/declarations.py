"""Strict structural declarations derived only from explicit source snapshots.

This module deliberately does not infer ownership or operational policy.  It
turns committed source evidence into stable discovery identities and structural
resource declarations; reviewed changes after the frozen source are recorded as
explicit amendments and protocol transitions.
"""

from __future__ import annotations

import ast
import hashlib
import json
import plistlib
import re
import tomllib
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Any, TypeAlias, cast

from scripts.runtime_ownership.manifests import (
    FROZEN_SOURCE_REVISION,
    SOURCE_MANIFEST_KIND,
    CommittedFile,
    CommittedSnapshot,
)

EFFECTIVE_FEEDBACK_REVISION = "f90202f1d1b9b2ed44075f38b0668c91fc0f196f"
_FROZEN_SOURCE_FILES_SHA256 = (
    "6693cc159f8ab213a513225b73096a30e4ae629404d6b5b7906d63cb6a52e4ef"
)
_EFFECTIVE_SOURCE_FILES_SHA256 = (
    "be2ad06f687bc619a89d12ad6274d6843b26278e2094d420146105c398e73cee"
)

RESOURCE_KINDS = frozenset({"artifact", "queue", "lock", "socket", "schema", "worker"})
NAME_SUFFIXES = (
    "FILE",
    "DIR",
    "ROOT",
    "PATH",
    "QUEUE",
    "LEDGER",
    "LOCK",
    "SOCKET",
    "SOCK",
    "STATE",
    "STATUS",
    "MANIFEST",
    "ARTIFACT",
    "CACHE",
    "INDEX",
    "DB",
    "SCHEMA",
    "SCHEMA_VERSION",
)
SOURCE_OR_DEPLOYMENT_NAMES = frozenset(
    {"PROJECT_ROOT", "REPO_ROOT", "STATIC_DIR", "LAUNCH_AGENT_DIR", "WRAPPER_DIR"}
)
_OWNER_FIELDS = frozenset(
    {
        "owner",
        "owner_package",
        "owner_symbol",
        "writers",
        "readers",
        "lifecycle",
        "coordination",
    }
)
_FULL_SHA1 = re.compile(r"[0-9a-f]{40}")

Row: TypeAlias = dict[str, Any]
Resource: TypeAlias = dict[str, Any]


class DeclarationError(ValueError):
    """Raised when structural evidence is incomplete, ambiguous, or inconsistent."""


def _plain_json(value: Any, _active: set[int] | None = None) -> Any:
    """Copy supported immutable/mutable JSON containers to plain JSON values."""

    active = set() if _active is None else _active
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in active:
            raise DeclarationError("canonical JSON contains a container cycle")
        active.add(identity)
        try:
            result: dict[str, Any] = {}
            for key, item in value.items():
                if type(key) is not str:
                    raise DeclarationError("canonical JSON object keys must be strings")
                result[key] = _plain_json(item, active)
            return result
        finally:
            active.remove(identity)
    if isinstance(value, (list, tuple)):
        identity = id(value)
        if identity in active:
            raise DeclarationError("canonical JSON contains a container cycle")
        active.add(identity)
        try:
            return [_plain_json(item, active) for item in value]
        finally:
            active.remove(identity)
    if value is None or type(value) in {str, int, float, bool}:
        return value
    raise DeclarationError(f"value is not JSON-compatible: {type(value).__name__}")


def _deep_freeze(value: Any, _active: set[int] | None = None) -> Any:
    active = set() if _active is None else _active
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in active:
            raise DeclarationError("declaration JSON contains a container cycle")
        active.add(identity)
        try:
            frozen: dict[str, Any] = {}
            for key, item in value.items():
                if type(key) is not str:
                    raise DeclarationError(
                        "declaration JSON object keys must be strings"
                    )
                frozen[key] = _deep_freeze(item, active)
            return MappingProxyType(frozen)
        finally:
            active.remove(identity)
    if isinstance(value, (list, tuple)):
        identity = id(value)
        if identity in active:
            raise DeclarationError("declaration JSON contains a container cycle")
        active.add(identity)
        try:
            return tuple(_deep_freeze(item, active) for item in value)
        finally:
            active.remove(identity)
    if value is None or type(value) in {str, int, float, bool}:
        return value
    raise DeclarationError(
        f"declaration data is not JSON-compatible: {type(value).__name__}"
    )


def _frozen_rows(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    return tuple(cast(Mapping[str, Any], _deep_freeze(row)) for row in rows)


@dataclass(frozen=True)
class ConcreteDeclarations:
    """Concrete-only discovery and grouped structural resources."""

    source_head: str | None
    rows: tuple[Mapping[str, Any], ...]
    resource_candidates: tuple[Mapping[str, Any], ...]
    exclusion_candidates: tuple[Mapping[str, Any], ...]
    lock_protocol_candidates: tuple[Mapping[str, Any], ...]
    resources: tuple[Mapping[str, Any], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "rows", _frozen_rows(self.rows))
        object.__setattr__(
            self, "resource_candidates", _frozen_rows(self.resource_candidates)
        )
        object.__setattr__(
            self, "exclusion_candidates", _frozen_rows(self.exclusion_candidates)
        )
        object.__setattr__(
            self,
            "lock_protocol_candidates",
            _frozen_rows(self.lock_protocol_candidates),
        )
        object.__setattr__(self, "resources", _frozen_rows(self.resources))


@dataclass(frozen=True)
class ResourceAmendment:
    """One reviewed resource addition and the resulting effective resources."""

    amendment_id: str
    payload: Mapping[str, Any]
    resources: tuple[Mapping[str, Any], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload", _deep_freeze(self.payload))
        object.__setattr__(self, "resources", _frozen_rows(self.resources))


@dataclass(frozen=True)
class ProtocolTransition:
    """One reviewed replacement of concrete lock-protocol discoveries."""

    transition_id: str
    payload: Mapping[str, Any]
    lock_protocol_candidates: tuple[Mapping[str, Any], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload", _deep_freeze(self.payload))
        object.__setattr__(
            self,
            "lock_protocol_candidates",
            _frozen_rows(self.lock_protocol_candidates),
        )


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            _plain_json(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except DeclarationError:
        raise
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise DeclarationError("value is not canonical JSON") from exc


def _semantic_id(prefix: str, value: Any) -> str:
    return f"{prefix}:{hashlib.sha256(_canonical_bytes(value)).hexdigest()}"


def _snapshot_files_sha256(source: CommittedSnapshot) -> str:
    _snapshot_input(source)
    rows = [
        {
            "path": row.path,
            "git_mode": row.git_mode,
            "byte_count": len(row.raw_bytes),
            "blob_oid": row.blob_oid,
            "sha256": hashlib.sha256(row.raw_bytes).hexdigest(),
        }
        for row in source.files
    ]
    return hashlib.sha256(_canonical_bytes(rows)).hexdigest()


def _require_snapshot_files_sha256(
    source: CommittedSnapshot, expected: str, *, label: str
) -> None:
    actual = _snapshot_files_sha256(source)
    if actual != expected:
        raise DeclarationError(f"{label} source files manifest seal mismatch: {actual}")


def _validate_canonical_relative_path(path: object, *, label: str) -> str:
    if type(path) is not str or not path:
        raise DeclarationError(f"{label} must be a non-empty string")
    if (
        path.startswith("/")
        or path.endswith("/")
        or "//" in path
        or "\\" in path
        or "\0" in path
        or any(part in {"", ".", ".."} for part in path.split("/"))
        or PurePosixPath(path).as_posix() != path
    ):
        raise DeclarationError(f"{label} is not a canonical relative path: {path}")
    return path


def _is_selected_source_path(path: str) -> bool:
    if path == "pyproject.toml":
        return True
    if path.startswith("src/chronovisor/") and path.endswith(".py"):
        return True
    if path.startswith("launchd/") and path.endswith(".plist"):
        return "/" not in path.removeprefix("launchd/")
    if path.startswith("scripts/chronovisor-"):
        return "/" not in path.removeprefix("scripts/")
    return False


def _snapshot_input(
    source: Mapping[str, bytes] | CommittedSnapshot,
) -> tuple[dict[str, bytes], str | None]:
    if isinstance(source, CommittedSnapshot):
        if type(source) is not CommittedSnapshot:
            raise DeclarationError("CommittedSnapshot subclasses are not accepted")
        if type(source.files) is not tuple:
            raise DeclarationError("CommittedSnapshot files must be a tuple")
        if (
            type(source.manifest_kind) is not str
            or source.manifest_kind != SOURCE_MANIFEST_KIND
        ):
            raise DeclarationError(
                "CommittedSnapshot is not a source-manifest snapshot"
            )
        if (
            type(source.git_object_format) is not str
            or source.git_object_format != "sha1"
        ):
            raise DeclarationError("CommittedSnapshot must use sha1")
        if (
            type(source.revision) is not str
            or _FULL_SHA1.fullmatch(source.revision) is None
        ):
            raise DeclarationError(
                "CommittedSnapshot revision must be a full lowercase sha1"
            )
        paths: list[str] = []
        for row in source.files:
            if type(row) is not CommittedFile:
                raise DeclarationError("CommittedSnapshot files contain an invalid row")
            path = _validate_canonical_relative_path(
                row.path, label="CommittedSnapshot file path"
            )
            if not _is_selected_source_path(path):
                raise DeclarationError(
                    f"CommittedSnapshot path is outside source selection: {path}"
                )
            paths.append(path)
        if len(paths) != len(set(paths)):
            raise DeclarationError("CommittedSnapshot contains duplicate paths")
        if paths != sorted(paths):
            raise DeclarationError(
                "CommittedSnapshot paths are not canonically ordered"
            )
        for row in source.files:
            expected_mode = (
                "100755" if row.path.startswith("scripts/chronovisor-") else "100644"
            )
            if type(row.git_mode) is not str or row.git_mode != expected_mode:
                raise DeclarationError(
                    f"CommittedSnapshot git mode is invalid: {row.path}"
                )
            if type(row.git_type) is not str or row.git_type != "blob":
                raise DeclarationError(
                    f"CommittedSnapshot git type is invalid: {row.path}"
                )
            if (
                type(row.blob_oid) is not str
                or _FULL_SHA1.fullmatch(row.blob_oid) is None
            ):
                raise DeclarationError(
                    f"CommittedSnapshot blob metadata is invalid: {row.path}"
                )
            if type(row.raw_bytes) is not bytes:
                raise DeclarationError(
                    f"CommittedSnapshot blob bytes are invalid: {row.path}"
                )
            expected_oid = hashlib.sha1(
                f"blob {len(row.raw_bytes)}\0".encode("ascii") + row.raw_bytes,
                usedforsecurity=False,
            ).hexdigest()
            if row.blob_oid != expected_oid:
                raise DeclarationError(
                    f"CommittedSnapshot blob bytes are not verified: {row.path}"
                )
        snapshot = {row.path: row.raw_bytes for row in source.files}
        source_head: str | None = source.revision
    elif isinstance(source, Mapping):
        snapshot = {}
        for path, raw in source.items():
            path = _validate_canonical_relative_path(path, label="snapshot path")
            if not _is_selected_source_path(path):
                raise DeclarationError(
                    f"snapshot path is outside source selection: {path}"
                )
            if type(raw) is not bytes:
                raise DeclarationError(f"snapshot value is not bytes: {path}")
            if path in snapshot:
                raise DeclarationError(f"snapshot contains a duplicate path: {path}")
            snapshot[path] = raw
        source_head = None
    else:
        raise DeclarationError(
            "source must be a Mapping[str, bytes] or CommittedSnapshot"
        )
    if not snapshot:
        raise DeclarationError("source snapshot is empty")
    return snapshot, source_head


def _text(snapshot: Mapping[str, bytes], path: str) -> str:
    try:
        raw = snapshot[path]
    except KeyError as exc:
        raise DeclarationError(f"source evidence path is missing: {path}") from exc
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DeclarationError(f"source evidence is not UTF-8: {path}") from exc


def _module_name(path: str) -> str:
    try:
        relative = PurePosixPath(path).relative_to("src")
    except ValueError as exc:
        raise DeclarationError(f"Python source path is outside src: {path}") from exc
    parts = list(relative.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _assignment_names(node: ast.stmt) -> tuple[list[str], ast.expr | None]:
    if isinstance(node, ast.Assign):
        return [
            target.id for target in node.targets if isinstance(target, ast.Name)
        ], node.value
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return [node.target.id], node.value
    return [], None


def _resolve_import_module(current: str, module: str | None, level: int) -> str:
    if level == 0:
        return module or ""
    parts = current.split(".")[:-1]
    prefix = parts[: max(0, len(parts) - level + 1)]
    if module:
        prefix.extend(module.split("."))
    return ".".join(prefix)


class _SourceIndex:
    def __init__(self, snapshot: dict[str, bytes]) -> None:
        self.snapshot = snapshot
        self.paths: dict[str, str] = {}
        self.trees: dict[str, ast.Module] = {}
        self.definitions: dict[tuple[str, str], tuple[ast.expr | None, int]] = {}
        self.imports: dict[tuple[str, str], tuple[str, str | None]] = {}
        for path in sorted(snapshot):
            if not path.startswith("src/chronovisor/") or not path.endswith(".py"):
                continue
            module = _module_name(path)
            try:
                tree = ast.parse(_text(snapshot, path), filename=path)
            except SyntaxError as exc:
                raise DeclarationError(f"invalid Python source: {path}") from exc
            self.paths[module] = path
            self.trees[module] = tree
            for node in tree.body:
                names, value = _assignment_names(node)
                for name in names:
                    self.definitions[(module, name)] = (value, int(node.lineno))
                if isinstance(node, ast.ImportFrom):
                    target_module = _resolve_import_module(
                        module, node.module, node.level
                    )
                    for alias in node.names:
                        self.imports[(module, alias.asname or alias.name)] = (
                            target_module,
                            alias.name,
                        )
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        local = alias.asname or alias.name.split(".")[0]
                        self.imports[(module, local)] = (alias.name, None)

    def evaluate_reference(
        self,
        module: str,
        name: str,
        stack: frozenset[tuple[str, str]] = frozenset(),
    ) -> str | None:
        key = (module, name)
        if key in stack:
            return None
        if name == "CHRONOVISOR_ROOT":
            return "$CHRONOVISOR_ROOT"
        imported = self.imports.get(key)
        if imported is not None and imported[1] is not None:
            return self.evaluate_reference(imported[0], imported[1], stack | {key})
        definition = self.definitions.get(key)
        if definition is None:
            return None
        return self.evaluate_expression(module, definition[0], stack | {key})

    def evaluate_expression(
        self,
        module: str,
        expression: ast.expr | None,
        stack: frozenset[tuple[str, str]],
    ) -> str | None:
        if expression is None:
            return None
        if isinstance(expression, ast.Constant) and isinstance(expression.value, str):
            return _normalize_literal(expression.value)
        if isinstance(expression, ast.Name):
            return self.evaluate_reference(module, expression.id, stack)
        if isinstance(expression, ast.Attribute):
            if isinstance(expression.value, ast.Name):
                imported = self.imports.get((module, expression.value.id))
                if imported is not None:
                    imported_module = imported[0]
                    if imported[1] is not None:
                        candidate = f"{imported_module}.{imported[1]}"
                        imported_module = candidate if candidate in self.paths else ""
                    if imported_module:
                        resolved = self.evaluate_reference(
                            imported_module, expression.attr, stack
                        )
                        if resolved is not None:
                            return resolved
            base = self.evaluate_expression(module, expression.value, stack)
            return (
                _parent_locator(base) if expression.attr == "parent" and base else None
            )
        if isinstance(expression, ast.BinOp) and isinstance(expression.op, ast.Div):
            base = self.evaluate_expression(module, expression.left, stack)
            child = self.evaluate_expression(module, expression.right, stack)
            return _join_locator(base, child) if base and child else None
        if not isinstance(expression, ast.Call):
            return None
        call_name = _call_name(expression.func)
        if call_name in {"resolve_root", "chronovisor.core.store.resolve_root"}:
            return "$CHRONOVISOR_ROOT"
        if call_name.endswith("runtime_repo_root"):
            return "$REPO_ROOT"
        if call_name.endswith("active_config_file") and not expression.args:
            return "$CHRONOVISOR_ROOT/config.toml"
        if call_name in {"Path.home", "pathlib.Path.home"}:
            return "$HOME"
        if call_name in {"Path", "pathlib.Path"} and expression.args:
            return self.evaluate_expression(module, expression.args[0], stack)
        if isinstance(expression.func, ast.Attribute):
            base = self.evaluate_expression(module, expression.func.value, stack)
            if expression.func.attr in {"expanduser", "resolve", "absolute"}:
                return base
            if expression.func.attr == "with_suffix" and base and expression.args:
                suffix = self.evaluate_expression(module, expression.args[0], stack)
                return _with_suffix(base, suffix) if suffix is not None else None
            if expression.func.attr == "joinpath" and base:
                result = base
                for argument in expression.args:
                    child = self.evaluate_expression(module, argument, stack)
                    if child is None:
                        return None
                    result = _join_locator(result, child)
                return result
        return None


def _call_name(expression: ast.expr) -> str:
    if isinstance(expression, ast.Name):
        return expression.id
    if isinstance(expression, ast.Attribute):
        prefix = _call_name(expression.value)
        return f"{prefix}.{expression.attr}" if prefix else expression.attr
    return ""


def _normalize_literal(value: str) -> str:
    expanded = value.strip()
    if expanded.startswith("~/"):
        return f"$HOME/{expanded[2:]}"
    return "$HOME" if expanded == "~" else expanded


def _join_locator(base: str, child: str) -> str:
    return child if child.startswith("/") else f"{base.rstrip('/')}/{child.strip('/')}"


def _parent_locator(value: str) -> str:
    head, separator, _tail = value.rstrip("/").rpartition("/")
    return head if separator else value


def _with_suffix(value: str, suffix: str) -> str:
    head, separator, tail = value.rpartition("/")
    replaced = f"{tail.rsplit('.', maxsplit=1)[0]}{suffix}"
    return f"{head}/{replaced}" if separator else replaced


def _matches_suffix(name: str) -> bool:
    return any(
        name == suffix or name.endswith(f"_{suffix}") for suffix in NAME_SUFFIXES
    )


def _is_schema_name(name: str) -> bool:
    return name == "SCHEMA" or name.endswith("_SCHEMA")


def _is_schema_version_name(name: str) -> bool:
    return name == "SCHEMA_VERSION" or name.endswith("_SCHEMA_VERSION")


def _resource_kind(name: str, locator: str) -> str:
    if (
        name.endswith("_LOCK")
        or name.endswith("_LOCK_FILE")
        or locator.endswith(".lock")
    ):
        return "lock"
    if (
        name == "QUEUE_FILE"
        or name.endswith("_QUEUE")
        or name.endswith("_QUEUE_FILE")
        or name == "SEMANTIC_JOBS_DB"
    ):
        return "queue"
    return "artifact"


def _exclusion_reason(name: str, value: ast.expr | None, resolved: str | None) -> str:
    if _is_schema_version_name(name):
        return "version_only_constant"
    if name == "MIN_CASES_PER_PRODUCTION_SCHEMA":
        return "schema_suffix_quantity_false_positive"
    if _is_schema_name(name):
        return "prompt_or_in_memory_schema"
    if name in SOURCE_OR_DEPLOYMENT_NAMES or resolved == "$REPO_ROOT":
        return "source_fixture_or_deployment_path"
    if isinstance(value, ast.Constant) and value.value is None:
        return "process_local_cache_or_lock"
    if name.endswith("_STATUS") or name.endswith("_STATE"):
        return "status_enum_or_process_local_value"
    return "non_runtime_or_unresolved_origin"


def _discovery_id(row: Mapping[str, Any]) -> str:
    identity = {key: row[key] for key in ("classification", "path", "module", "symbol")}
    if row["classification"] == "resource":
        identity.update({"kind": row["kind"], "locator": row["locator"]})
    elif row["classification"] == "lock_protocol":
        identity.update(
            {
                "scope": row["scope"],
                "operation": row["operation"],
                "occurrence": row["occurrence"],
            }
        )
    else:
        identity["reason"] = row["reason"]
    return _semantic_id("runtime-site", identity)


def _ast_discovery(index: _SourceIndex) -> list[Row]:
    rows: list[Row] = []
    for (module, name), (value, line) in sorted(index.definitions.items()):
        if not _matches_suffix(name):
            continue
        resolved = index.evaluate_reference(module, name)
        row: Row = {
            "path": index.paths[module],
            "line": line,
            "module": module,
            "symbol": name,
        }
        if (
            module == "chronovisor.recall.evidence_certificate"
            and name == "CERTIFICATE_LEDGER_LOCK"
        ):
            row.update(
                {
                    "classification": "exclusion",
                    "reason": "legacy_phantom_lock_declaration",
                }
            )
        elif (
            _is_schema_name(name)
            and isinstance(value, ast.Constant)
            and isinstance(value.value, str)
            and value.value.startswith("chronovisor.")
        ):
            row.update(
                {
                    "classification": "resource",
                    "kind": "schema",
                    "locator": {"type": "schema_id", "value": value.value},
                }
            )
        elif (
            resolved
            and (
                resolved.startswith("$CHRONOVISOR_ROOT")
                or resolved.startswith("$HOME")
                or "/.chronovisor/" in resolved
            )
            and name not in SOURCE_OR_DEPLOYMENT_NAMES
        ):
            row.update(
                {
                    "classification": "resource",
                    "kind": _resource_kind(name, resolved),
                    "locator": {"type": "path", "value": resolved},
                }
            )
        else:
            row.update(
                {
                    "classification": "exclusion",
                    "reason": _exclusion_reason(name, value, resolved),
                }
            )
        row["discovery_id"] = _discovery_id(row)
        rows.append(row)
    return rows


_EXPLICIT_STATE_SPECS: tuple[tuple[str, str, str, str, str, str], ...] = (
    (
        "src/chronovisor/raw/raw_replay.py",
        'QUEUE_FILE = CHRONOVISOR_ROOT / "review" / "raw-replay-queue.jsonl"',
        "chronovisor.raw.raw_replay",
        "QUEUE_FILE.sidecar-lock",
        "lock",
        "$CHRONOVISOR_ROOT/review/raw-replay-queue.jsonl.lock",
    ),
    (
        "src/chronovisor/recall/evidence_certificate.py",
        "def append_certificates(",
        "chronovisor.recall.evidence_certificate",
        "append_certificates.sidecar-lock",
        "lock",
        "$CHRONOVISOR_ROOT/recall/evidence-certificate-ledger.jsonl.lock",
    ),
    (
        "src/chronovisor/ops/convergence.py",
        "class ConvergenceStore:",
        "chronovisor.ops.convergence",
        "ConvergenceStore.state_file",
        "artifact",
        "$CHRONOVISOR_ROOT/runtime/convergence/state.json",
    ),
    (
        "src/chronovisor/ops/convergence.py",
        "class ConvergenceStore:",
        "chronovisor.ops.convergence",
        "ConvergenceStore.events_file",
        "artifact",
        "$CHRONOVISOR_ROOT/runtime/convergence/events.jsonl",
    ),
    (
        "src/chronovisor/ops/convergence.py",
        "class ConvergenceStore:",
        "chronovisor.ops.convergence",
        "ConvergenceStore.lock_file",
        "lock",
        "$CHRONOVISOR_ROOT/runtime/convergence/state.lock",
    ),
    (
        "src/chronovisor/ingest/page_registry.py",
        "class PageRegistry:",
        "chronovisor.ingest.page_registry",
        "PageRegistry.path",
        "artifact",
        "$CHRONOVISOR_ROOT/runtime/librarian/page-registry.json",
    ),
    (
        "src/chronovisor/ingest/page_registry.py",
        "class PageRegistry:",
        "chronovisor.ingest.page_registry",
        "PageRegistry.events_path",
        "artifact",
        "$CHRONOVISOR_ROOT/runtime/librarian/page-registry-events.jsonl",
    ),
    (
        "src/chronovisor/ingest/page_registry.py",
        "class PageRegistry:",
        "chronovisor.ingest.page_registry",
        "PageRegistry.lock_path",
        "lock",
        "$CHRONOVISOR_ROOT/runtime/librarian/page-registry.lock",
    ),
    (
        "src/chronovisor/librarian/collection_authority.py",
        "class CollectionRegistry:",
        "chronovisor.librarian.collection_authority",
        "CollectionRegistry.path",
        "artifact",
        "$CHRONOVISOR_ROOT/runtime/librarian/collection-registry.json",
    ),
    (
        "src/chronovisor/librarian/collection_authority.py",
        "class CollectionRegistry:",
        "chronovisor.librarian.collection_authority",
        "CollectionRegistry.lock_path",
        "lock",
        "$CHRONOVISOR_ROOT/runtime/librarian/collection-registry.lock",
    ),
    (
        "src/chronovisor/librarian/managed_hold.py",
        "class ManagedHoldStore:",
        "chronovisor.librarian.managed_hold",
        "ManagedHoldStore.path",
        "artifact",
        "$CHRONOVISOR_ROOT/runtime/managed-holds/state.json",
    ),
    (
        "src/chronovisor/librarian/managed_hold.py",
        "class ManagedHoldStore:",
        "chronovisor.librarian.managed_hold",
        "ManagedHoldStore.lock_path",
        "lock",
        "$CHRONOVISOR_ROOT/runtime/managed-holds/state.json.lock",
    ),
    (
        "src/chronovisor/knowledge_graph/store.py",
        "class KnowledgeGraphStore:",
        "chronovisor.knowledge_graph.store",
        "KnowledgeGraphStore.events_file",
        "artifact",
        "$CHRONOVISOR_ROOT/knowledge-graph/relation-events.jsonl",
    ),
    (
        "src/chronovisor/knowledge_graph/store.py",
        "class KnowledgeGraphStore:",
        "chronovisor.knowledge_graph.store",
        "KnowledgeGraphStore.snapshot_file",
        "artifact",
        "$CHRONOVISOR_ROOT/knowledge-graph/relation-snapshot.json",
    ),
    (
        "src/chronovisor/knowledge_graph/store.py",
        "class KnowledgeGraphStore:",
        "chronovisor.knowledge_graph.store",
        "KnowledgeGraphStore.entity_snapshot_file",
        "artifact",
        "$CHRONOVISOR_ROOT/knowledge-graph/entity-snapshot.json",
    ),
    (
        "src/chronovisor/knowledge_graph/store.py",
        "class KnowledgeGraphStore:",
        "chronovisor.knowledge_graph.store",
        "KnowledgeGraphStore.community_snapshot_file",
        "artifact",
        "$CHRONOVISOR_ROOT/knowledge-graph/community-snapshot.json",
    ),
    (
        "src/chronovisor/knowledge_graph/store.py",
        "class KnowledgeGraphStore:",
        "chronovisor.knowledge_graph.store",
        "KnowledgeGraphStore.builder_state_file",
        "artifact",
        "$CHRONOVISOR_ROOT/knowledge-graph/builder-state.json",
    ),
    (
        "src/chronovisor/knowledge_graph/store.py",
        "class KnowledgeGraphStore:",
        "chronovisor.knowledge_graph.store",
        "KnowledgeGraphStore.community_summary_state_file",
        "artifact",
        "$CHRONOVISOR_ROOT/knowledge-graph/community-summary-state.json",
    ),
    (
        "src/chronovisor/knowledge_graph/store.py",
        "class KnowledgeGraphStore:",
        "chronovisor.knowledge_graph.store",
        "KnowledgeGraphStore.lock_file",
        "lock",
        "$CHRONOVISOR_ROOT/knowledge-graph/store.lock",
    ),
    (
        "src/chronovisor/decision/failure_supervisor.py",
        "def _failure_state_lock(",
        "chronovisor.decision.failure_supervisor",
        "_state_file",
        "artifact",
        "$CHRONOVISOR_ROOT/runtime/failures/state.json",
    ),
    (
        "src/chronovisor/decision/failure_supervisor.py",
        "def _failure_state_lock(",
        "chronovisor.decision.failure_supervisor",
        "_failure_state_lock",
        "lock",
        "$CHRONOVISOR_ROOT/runtime/failures/state.lock",
    ),
    (
        "src/chronovisor/ops/dashboard.py",
        'CHRONOVISOR_ROOT / "runtime" / "dashboard-access-token"',
        "chronovisor.ops.dashboard",
        "dashboard-access-token",
        "artifact",
        "$CHRONOVISOR_ROOT/runtime/dashboard-access-token",
    ),
    (
        "src/chronovisor/ops/dashboard.py",
        'CHRONOVISOR_ROOT / "runtime" / "dashboard-credentials.json"',
        "chronovisor.ops.dashboard",
        "dashboard-credentials",
        "artifact",
        "$CHRONOVISOR_ROOT/runtime/dashboard-credentials.json",
    ),
    (
        "scripts/chronovisor-searxng",
        'readonly SETTINGS="$CONFIG_ROOT/settings.yml"',
        "script:scripts/chronovisor-searxng",
        "SETTINGS",
        "artifact",
        "$HOME/.chronovisor/runtime/searxng/settings.yml",
    ),
    (
        "scripts/chronovisor-searxng",
        'readonly SECRET_FILE="$CONFIG_ROOT/secret"',
        "script:scripts/chronovisor-searxng",
        "SECRET_FILE",
        "artifact",
        "$HOME/.chronovisor/runtime/searxng/secret",
    ),
    (
        "scripts/chronovisor-searxng",
        'readonly RUNTIME_ROOT="${CHRONOVISOR_SEARXNG_ROOT:-$HOME/.local/share/chronovisor/searxng}"',
        "script:scripts/chronovisor-searxng",
        "source",
        "artifact",
        "$HOME/.local/share/chronovisor/searxng/source",
    ),
    (
        "scripts/chronovisor-searxng",
        'readonly VENV="$RUNTIME_ROOT/.venv"',
        "script:scripts/chronovisor-searxng",
        "VENV",
        "artifact",
        "$HOME/.local/share/chronovisor/searxng/.venv",
    ),
    (
        "src/chronovisor/search/search_eval.py",
        'LABEL_QUEUE_FILE = RECALL_DIR / "search-label-queue.jsonl"',
        "chronovisor.search.search_eval",
        "LABEL_QUEUE_FILE.sidecar-lock",
        "lock",
        "$CHRONOVISOR_ROOT/recall/search-label-queue.jsonl.lock",
    ),
    (
        "src/chronovisor/recall/claims.py",
        'CLAIMS_FILE = CLAIMS_DIR / "claims.jsonl"',
        "chronovisor.recall.claims",
        "CLAIMS_FILE.sidecar-lock",
        "lock",
        "$CHRONOVISOR_ROOT/claims/claims.jsonl.lock",
    ),
)


def _line_containing(text: str, needle: str) -> int:
    for line_number, line in enumerate(text.splitlines(), start=1):
        if needle in line:
            return line_number
    raise DeclarationError(f"evidence marker not found: {needle}")


def _explicit_state_discovery(snapshot: dict[str, bytes]) -> list[Row]:
    rows: list[Row] = []
    for path, needle, module, symbol, kind, locator in _EXPLICIT_STATE_SPECS:
        if symbol == "LABEL_QUEUE_FILE.sidecar-lock" and not (
            "_search_label_queue_lock(candidate_file)"
            in _text(snapshot, "src/chronovisor/ops/golden_expand.py")
            or "_search_label_queue_lock(output_file)"
            in _text(snapshot, "src/chronovisor/search/search_eval.py")
            or "_search_label_queue_lock(queue_file)"
            in _text(snapshot, "src/chronovisor/search/search_eval.py")
        ):
            continue
        if symbol == "CLAIMS_FILE.sidecar-lock" and not (
            "_claims_ledger_lock(CLAIMS_FILE)"
            in _text(snapshot, "src/chronovisor/recall/claims.py")
            or "_claims_ledger_lock(path)"
            in _text(snapshot, "src/chronovisor/recall/claims.py")
        ):
            continue
        row: Row = {
            "classification": "resource",
            "path": path,
            "line": _line_containing(_text(snapshot, path), needle),
            "module": module,
            "symbol": symbol,
            "kind": kind,
            "locator": {"type": "path", "value": locator},
        }
        row["discovery_id"] = _discovery_id(row)
        rows.append(row)
    return rows


_SOCKET_SPECS: tuple[tuple[str, str, str, str, str], ...] = (
    (
        "src/chronovisor/core/runtime_config.py",
        'socket: str = "~/.chronovisor/runtime/semantic.sock"',
        "chronovisor.search.semantic_service",
        "serve",
        "unix://$HOME/.chronovisor/runtime/semantic.sock",
    ),
    (
        "src/chronovisor/core/runtime_config.py",
        'socket: str = "~/.chronovisor/runtime/reranker.sock"',
        "chronovisor.search.reranker_service",
        "serve",
        "unix://$HOME/.chronovisor/runtime/reranker.sock",
    ),
    (
        "src/chronovisor/ops/dashboard.py",
        'parser.add_argument("--port", type=int, default=8765)',
        "chronovisor.ops.dashboard",
        "serve",
        "tcp://0.0.0.0:8765",
    ),
    (
        "src/chronovisor/core/ollama.py",
        'OLLAMA_URL = "http://localhost:11434"',
        "external",
        "ollama",
        "tcp://127.0.0.1:11434",
    ),
    (
        "scripts/chronovisor-searxng",
        'export SEARXNG_PORT="8888"',
        "script:scripts/chronovisor-searxng",
        "chronovisor-searxng",
        "tcp://127.0.0.1:8888",
    ),
    (
        "src/chronovisor/hosts/server.py",
        "def main():",
        "chronovisor.hosts.server",
        "main",
        "stdio://chronovisor-mcp",
    ),
)


def _socket_discovery(snapshot: dict[str, bytes]) -> list[Row]:
    rows: list[Row] = []
    for path, needle, module, symbol, address in _SOCKET_SPECS:
        row: Row = {
            "classification": "resource",
            "path": path,
            "line": _line_containing(_text(snapshot, path), needle),
            "module": module,
            "symbol": symbol,
            "kind": "socket",
            "locator": {"type": "socket", "value": address},
        }
        row["discovery_id"] = _discovery_id(row)
        rows.append(row)
    return rows


def _project_entrypoints(snapshot: dict[str, bytes]) -> dict[str, str]:
    try:
        project = tomllib.loads(_text(snapshot, "pyproject.toml"))
    except tomllib.TOMLDecodeError as exc:
        raise DeclarationError("pyproject.toml is invalid") from exc
    scripts = project.get("project", {}).get("scripts", {})
    if not isinstance(scripts, dict):
        raise DeclarationError("project.scripts must be a table")
    result: dict[str, str] = {}
    for name, target in sorted(scripts.items()):
        if (
            not isinstance(name, str)
            or not isinstance(target, str)
            or ":" not in target
        ):
            raise DeclarationError(
                "project.scripts entries must be string module targets"
            )
        result[name] = target
    return result


def _worker_row(
    *,
    path: str,
    line: int,
    locator_type: str,
    locator_value: str,
    module: str,
    symbol: str,
    additional_evidence: Sequence[Mapping[str, Any]] = (),
) -> Row:
    row: Row = {
        "classification": "resource",
        "path": path,
        "line": line,
        "module": module,
        "symbol": symbol,
        "kind": "worker",
        "locator": {"type": locator_type, "value": locator_value},
    }
    if additional_evidence:
        row["additional_evidence"] = [dict(item) for item in additional_evidence]
    row["discovery_id"] = _discovery_id(row)
    return row


def _launchd_invocation_lines(
    *,
    label: str,
    wrapper: str,
    arguments: list[Any],
    entrypoints: Mapping[str, str],
) -> list[int]:
    if label == "com.trafficsign.chronovisor-librarian-review":
        return [11, 20, 31]
    if label == "com.trafficsign.chronovisor-library-evidence":
        return [10]
    if label == "com.trafficsign.chronovisor-searxng":
        return [30]
    del arguments
    matches = [
        name
        for name in sorted(entrypoints, key=lambda item: (-len(item), item))
        if re.search(rf"(?<![A-Za-z0-9_-]){re.escape(name)}(?![A-Za-z0-9_-])", wrapper)
    ]
    return [] if not matches else [_line_containing(wrapper, matches[0])]


def _worker_discovery(snapshot: dict[str, bytes]) -> list[Row]:
    entrypoints = _project_entrypoints(snapshot)
    pyproject = _text(snapshot, "pyproject.toml")
    rows: list[Row] = []
    for name, target in entrypoints.items():
        module, symbol = target.split(":", maxsplit=1)
        rows.append(
            _worker_row(
                path="pyproject.toml",
                line=_line_containing(pyproject, f'{name} = "{target}"'),
                locator_type="entrypoint",
                locator_value=name,
                module=module,
                symbol=symbol,
            )
        )
    for path in sorted(snapshot):
        if not path.startswith("launchd/") or not path.endswith(".plist"):
            continue
        raw = snapshot[path]
        try:
            payload = plistlib.loads(raw)
        except Exception as exc:
            raise DeclarationError(f"invalid tracked launchd plist: {path}") from exc
        if not isinstance(payload, dict):
            raise DeclarationError(f"invalid tracked launchd plist: {path}")
        label = payload.get("Label")
        arguments = payload.get("ProgramArguments")
        if (
            not isinstance(label, str)
            or not label
            or not isinstance(arguments, list)
            or not arguments
        ):
            raise DeclarationError(f"invalid tracked launchd worker: {path}")
        wrapper_name = PurePosixPath(str(arguments[0])).name
        wrapper_path = f"scripts/{wrapper_name}"
        wrapper = _text(snapshot, wrapper_path)
        matches = [
            name
            for name in sorted(entrypoints, key=lambda item: (-len(item), item))
            if re.search(
                rf"(?<![A-Za-z0-9_-]){re.escape(name)}(?![A-Za-z0-9_-])",
                wrapper,
            )
        ]
        if matches:
            target_module, target_symbol = entrypoints[matches[0]].split(
                ":", maxsplit=1
            )
            module, symbol = target_module, target_symbol
        else:
            module, symbol = f"script:{wrapper_path}", wrapper_path
        invocation_lines = _launchd_invocation_lines(
            label=label,
            wrapper=wrapper,
            arguments=arguments,
            entrypoints=entrypoints,
        )
        rows.append(
            _worker_row(
                path=path,
                line=_line_containing(raw.decode("utf-8"), label),
                locator_type="launchd",
                locator_value=label,
                module=module,
                symbol=symbol,
                additional_evidence=[
                    {"path": wrapper_path, "line": line} for line in invocation_lines
                ],
            )
        )
    rows.extend(_lab_dispatch_workers(snapshot))
    rows.extend(_python_module_workers(snapshot))
    return rows


def _lab_dispatch_workers(snapshot: dict[str, bytes]) -> list[Row]:
    path = "src/chronovisor/lab/cli.py"
    try:
        tree = ast.parse(_text(snapshot, path), filename=path)
    except SyntaxError as exc:
        raise DeclarationError(f"invalid Python source: {path}") from exc
    rows: list[Row] = []
    for node in tree.body:
        names, value = _assignment_names(node)
        if "COMMANDS" not in names or not isinstance(value, ast.Dict):
            continue
        for key, item in zip(value.keys, value.values, strict=True):
            if (
                not isinstance(key, ast.Constant)
                or not isinstance(key.value, str)
                or not isinstance(item, ast.Tuple)
                or not item.elts
                or not isinstance(item.elts[0], ast.Constant)
                or not isinstance(item.elts[0].value, str)
            ):
                raise DeclarationError(
                    "lab COMMANDS must contain literal command/module pairs"
                )
            command = key.value
            module = item.elts[0].value
            rows.append(
                _worker_row(
                    path=path,
                    line=int(key.lineno),
                    locator_type="lab_dispatch",
                    locator_value=command,
                    module=module,
                    symbol=f"COMMANDS[{command}]",
                )
            )
    return rows


class _PythonModuleWorkerCollector(ast.NodeVisitor):
    def __init__(self, *, path: str) -> None:
        self.path = path
        self.scope: list[str] = []
        self.occurrences: Counter[tuple[str, str]] = Counter()
        self.rows: list[Row] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_List(self, node: ast.List) -> None:
        self._record_sequence(node, node.elts)
        self.generic_visit(node)

    def visit_Tuple(self, node: ast.Tuple) -> None:
        self._record_sequence(node, node.elts)
        self.generic_visit(node)

    def _record_sequence(self, node: ast.expr, values: list[ast.expr]) -> None:
        if (
            len(values) < 3
            or not isinstance(values[0], ast.Attribute)
            or not isinstance(values[0].value, ast.Name)
            or values[0].value.id != "sys"
            or values[0].attr != "executable"
            or not isinstance(values[1], ast.Constant)
            or values[1].value != "-m"
            or not isinstance(values[2], ast.Constant)
            or not isinstance(values[2].value, str)
            or not values[2].value.startswith("chronovisor.")
        ):
            return
        target = values[2].value
        scope = ".".join(self.scope) or "<module>"
        occurrence_key = (scope, target)
        self.occurrences[occurrence_key] += 1
        self.rows.append(
            _worker_row(
                path=self.path,
                line=int(node.lineno),
                locator_type="module_worker",
                locator_value=target,
                module=target,
                symbol=f"python-module:{target}:{scope}:{self.occurrences[occurrence_key]}",
            )
        )


def _python_module_workers(snapshot: dict[str, bytes]) -> list[Row]:
    rows: list[Row] = []
    for path in sorted(snapshot):
        if not path.startswith("src/chronovisor/") or not path.endswith(".py"):
            continue
        collector = _PythonModuleWorkerCollector(path=path)
        try:
            collector.visit(ast.parse(_text(snapshot, path), filename=path))
        except SyntaxError as exc:
            raise DeclarationError(f"invalid Python source: {path}") from exc
        rows.extend(collector.rows)
    return rows


class _LockProtocolCollector(ast.NodeVisitor):
    def __init__(self, *, module: str, path: str, index: _SourceIndex) -> None:
        self.module = module
        self.path = path
        self.index = index
        self.scope: list[str] = []
        self.occurrences: Counter[tuple[str, str]] = Counter()
        self.rows: list[Row] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_Call(self, node: ast.Call) -> None:
        operation = ""
        protocol = ""
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "fcntl"
            and node.func.attr == "flock"
            and len(node.args) > 1
        ):
            operation = ast.unparse(node.args[1])
            if "LOCK_UN" in operation:
                self.generic_visit(node)
                return
            protocol = _flock_protocol(operation)
        else:
            helper = self._lock_helper_name(node.func)
            if helper:
                operation, protocol = f"helper:{helper}", "helper-managed-lock"
        if protocol:
            scope = ".".join(self.scope) or "<module>"
            key = (scope, operation)
            self.occurrences[key] += 1
            row: Row = {
                "classification": "lock_protocol",
                "path": self.path,
                "line": int(node.lineno),
                "module": self.module,
                "symbol": "flock" if not operation.startswith("helper:") else operation,
                "scope": scope,
                "operation": operation,
                "protocol": protocol,
                "occurrence": self.occurrences[key],
            }
            row["discovery_id"] = _discovery_id(row)
            self.rows.append(row)
        self.generic_visit(node)

    def _lock_helper_name(self, expression: ast.expr) -> str:
        helpers = {
            "file_lock",
            "exclusive_text_file_lock",
            "sidecar_exclusive_lock",
            "_search_label_queue_lock",
            "_claims_ledger_lock",
        }
        if isinstance(expression, ast.Name):
            if expression.id in helpers:
                return expression.id
            imported = self.index.imports.get((self.module, expression.id))
            if imported is not None and imported[1] in helpers:
                return imported[1]
        if isinstance(expression, ast.Attribute) and expression.attr in helpers:
            return expression.attr
        return ""


def _flock_protocol(operation: str) -> str:
    has_exclusive = "LOCK_EX" in operation
    has_shared = "LOCK_SH" in operation
    has_nonblocking = "LOCK_NB" in operation
    if has_exclusive and has_shared:
        return "exclusive-or-shared"
    if has_exclusive and has_nonblocking:
        return "exclusive-nonblocking"
    if has_shared and has_nonblocking:
        return "shared-nonblocking"
    if has_exclusive:
        return "exclusive"
    if has_shared:
        return "shared"
    if has_nonblocking:
        return "indirect-nonblocking"
    return "indirect-operation"


def _lock_protocol_discovery(index: _SourceIndex) -> list[Row]:
    rows: list[Row] = []
    for module, tree in sorted(index.trees.items()):
        collector = _LockProtocolCollector(
            module=module, path=index.paths[module], index=index
        )
        collector.visit(tree)
        rows.extend(collector.rows)
    return rows


_LOCK_PROTECTS_LOCATORS: dict[str, tuple[tuple[str, str], ...]] = {
    "$CHRONOVISOR_ROOT/review/raw-replay-queue.jsonl.lock": (
        ("queue", "$CHRONOVISOR_ROOT/review/raw-replay-queue.jsonl"),
    ),
    "$CHRONOVISOR_ROOT/claims/claims.jsonl.lock": (
        ("artifact", "$CHRONOVISOR_ROOT/claims/claims.jsonl"),
    ),
    "$CHRONOVISOR_ROOT/recall/search-label-queue.jsonl.lock": (
        ("queue", "$CHRONOVISOR_ROOT/recall/search-label-queue.jsonl"),
    ),
    "$CHRONOVISOR_ROOT/.index/semantic/activation.lock": (
        ("artifact", "$CHRONOVISOR_ROOT/.index/semantic/active.json"),
    ),
    "$CHRONOVISOR_ROOT/recall/evidence-certificate-ledger.jsonl.lock": (
        ("artifact", "$CHRONOVISOR_ROOT/recall/evidence-certificate-ledger.jsonl"),
    ),
    "$CHRONOVISOR_ROOT/runtime/background-jobs/state.lock": (
        ("artifact", "$CHRONOVISOR_ROOT/runtime/background-jobs/state.json"),
    ),
    "$CHRONOVISOR_ROOT/runtime/model-lab/model-lab.lock": (
        ("artifact", "$CHRONOVISOR_ROOT/runtime/model-lab/active-policy.json"),
        ("artifact", "$CHRONOVISOR_ROOT/runtime/model-lab/state.json"),
        ("artifact", "$CHRONOVISOR_ROOT/runtime/model-lab/replay.jsonl"),
        ("artifact", "$CHRONOVISOR_ROOT/runtime/model-lab/history.jsonl"),
    ),
    "$CHRONOVISOR_ROOT/runtime/convergence/state.lock": (
        ("artifact", "$CHRONOVISOR_ROOT/runtime/convergence/state.json"),
        ("artifact", "$CHRONOVISOR_ROOT/runtime/convergence/events.jsonl"),
    ),
    "$CHRONOVISOR_ROOT/runtime/librarian/page-registry.lock": (
        ("artifact", "$CHRONOVISOR_ROOT/runtime/librarian/page-registry.json"),
        ("artifact", "$CHRONOVISOR_ROOT/runtime/librarian/page-registry-events.jsonl"),
    ),
    "$CHRONOVISOR_ROOT/runtime/librarian/collection-registry.lock": (
        ("artifact", "$CHRONOVISOR_ROOT/runtime/librarian/collection-registry.json"),
    ),
    "$CHRONOVISOR_ROOT/runtime/managed-holds/state.json.lock": (
        ("artifact", "$CHRONOVISOR_ROOT/runtime/managed-holds/state.json"),
    ),
    "$CHRONOVISOR_ROOT/knowledge-graph/store.lock": (
        ("artifact", "$CHRONOVISOR_ROOT/knowledge-graph/relation-events.jsonl"),
        ("artifact", "$CHRONOVISOR_ROOT/knowledge-graph/relation-snapshot.json"),
        ("artifact", "$CHRONOVISOR_ROOT/knowledge-graph/entity-snapshot.json"),
        ("artifact", "$CHRONOVISOR_ROOT/knowledge-graph/community-snapshot.json"),
        ("artifact", "$CHRONOVISOR_ROOT/knowledge-graph/builder-state.json"),
        ("artifact", "$CHRONOVISOR_ROOT/knowledge-graph/community-summary-state.json"),
    ),
    "$CHRONOVISOR_ROOT/runtime/research/consolidation.lock": (
        ("artifact", "$CHRONOVISOR_ROOT/runtime/research/consolidation-state.json"),
    ),
    "$CHRONOVISOR_ROOT/runtime/failures/state.lock": (
        ("artifact", "$CHRONOVISOR_ROOT/runtime/failures/state.json"),
    ),
    "$CHRONOVISOR_ROOT/recall/audit.lock": (
        ("artifact", "$CHRONOVISOR_ROOT/recall/audit-state.json"),
        ("artifact", "$CHRONOVISOR_ROOT/recall/pull-consumed.jsonl"),
        ("worker", "chronovisor-recall-audit"),
    ),
    "$CHRONOVISOR_ROOT/runtime/sleep-cycle.lock": (
        ("artifact", "$CHRONOVISOR_ROOT/runtime/sleep-cycle-history.jsonl"),
        ("artifact", "$CHRONOVISOR_ROOT/runtime/sleep-cycle-active-lane.json"),
        ("worker", "chronovisor-sleep"),
    ),
    "$CHRONOVISOR_ROOT/runtime/research/research-generation.lock": (
        ("artifact", "$CHRONOVISOR_ROOT/runtime/research/active-research.json"),
        ("worker", "chronovisor-research"),
    ),
    "$CHRONOVISOR_ROOT/runtime/chronovisor-mutation.lock": (
        ("artifact", "$CHRONOVISOR_ROOT/pages"),
        ("artifact", "$CHRONOVISOR_ROOT/system"),
    ),
    "$CHRONOVISOR_ROOT/runtime/decision-authority.lock": (
        ("artifact", "$CHRONOVISOR_ROOT/pages"),
        ("artifact", "$CHRONOVISOR_ROOT/system"),
    ),
    "$CHRONOVISOR_ROOT/runtime/recall-improvement/run-due.lock": (
        ("worker", "chronovisor-recall-improve"),
    ),
    "$CHRONOVISOR_ROOT/runtime/accelerator-inference.lock": (
        ("worker", "chronovisor-recall"),
        ("worker", "chronovisor-research"),
        ("worker", "chronovisor-semantic-service"),
    ),
}

_FROZEN_ID_HASHES = {
    "rows": "3a8291d534e65d4dadbdb1c1b4b3d7e9e03eb465fc883471bc0b4c08824883d0",
    "resource_candidates": "c7214ebb7fd4d84be5ac0aad4fed9eb99dc09542340981568263a072fac270f5",
    "exclusion_candidates": "5e40a4142e3a2abd1c283ca836eec8ac72cd1eb376e6a34c9dbebbd96c6ede82",
    "lock_protocol_candidates": "3bffbd1791c08c8b75cd5960617b3e3d969b1ac22a8c391a367ebf1c985a871e",
    "resources": "3a77631467721e0a261a419b95e6e6fdc23e889f5af4dd832ca9f6f742f1d326",
}


def _validate_evidence(
    snapshot: Mapping[str, bytes], evidence: Mapping[str, Any]
) -> None:
    if not isinstance(evidence, Mapping):
        raise DeclarationError("evidence must be an object")
    if set(evidence) != {"path", "line"}:
        raise DeclarationError("evidence must contain exactly path and line")
    path = _validate_canonical_relative_path(evidence["path"], label="evidence path")
    line = evidence["line"]
    if type(line) is not int or line < 1:
        raise DeclarationError(f"evidence line is invalid: {path}")
    if path not in snapshot or type(snapshot[path]) is not bytes:
        raise DeclarationError(f"source evidence path is missing or invalid: {path}")
    text = _text(snapshot, path)
    if line > len(text.splitlines()):
        raise DeclarationError(
            f"evidence line is outside the source snapshot: {path}:{line}"
        )


def _evidence_for_group(
    snapshot: Mapping[str, bytes], group: Sequence[Mapping[str, Any]]
) -> list[Row]:
    points: set[tuple[str, int]] = set()
    for row in group:
        if not isinstance(row, Mapping):
            raise DeclarationError("resource candidate must be an object")
        primary = {"path": row.get("path"), "line": row.get("line")}
        _validate_evidence(snapshot, primary)
        points.add((cast(str, primary["path"]), cast(int, primary["line"])))
        additional = row.get("additional_evidence", [])
        if not isinstance(additional, (list, tuple)):
            raise DeclarationError("additional_evidence must be a JSON sequence")
        for item in additional:
            if not isinstance(item, Mapping):
                raise DeclarationError("additional evidence must be an object")
            _validate_evidence(snapshot, item)
            points.add((cast(str, item["path"]), cast(int, item["line"])))
    result = [{"path": path, "line": line} for path, line in sorted(points)]
    for item in result:
        _validate_evidence(snapshot, item)
    return result


def _resource_id(kind: str, locator_value: str) -> str:
    """Return a kind-scoped locator identity.

    The same physical locator may intentionally have different declarations
    under different resource kinds; `{kind, locator}` is the reviewed identity.
    """

    return _semantic_id("runtime-resource", {"kind": kind, "locator": locator_value})


_LOCATOR_TYPES_BY_KIND = {
    "artifact": frozenset({"path"}),
    "queue": frozenset({"path"}),
    "lock": frozenset({"path"}),
    "socket": frozenset({"socket"}),
    "schema": frozenset({"schema_id"}),
    "worker": frozenset({"entrypoint", "launchd", "lab_dispatch", "module_worker"}),
}


def _validate_locator(kind: object, locator: object) -> tuple[str, str]:
    """Validate locator syntax without deciding whether a new locator is reviewed.

    A canonical locator such as ``$CHRONOVISOR_ROOT/unknown`` remains
    discoverable; registry/gate validation owns semantic review and rejection
    of newly surfaced resources.
    """

    if type(kind) is not str or kind not in RESOURCE_KINDS:
        raise DeclarationError("resource kind is invalid")
    if not isinstance(locator, Mapping) or set(locator) != {"type", "value"}:
        raise DeclarationError("resource locator must contain exactly type and value")
    locator_type = locator["type"]
    locator_value = locator["value"]
    if (
        type(locator_type) is not str
        or locator_type not in _LOCATOR_TYPES_BY_KIND[kind]
    ):
        raise DeclarationError(f"resource locator type is invalid for {kind}")
    if type(locator_value) is not str or not locator_value or "\0" in locator_value:
        raise DeclarationError("resource locator value is invalid")
    if locator_value.startswith("/"):
        raise DeclarationError("absolute resource locators are forbidden")
    if locator_type == "path":
        prefixes = ("$CHRONOVISOR_ROOT", "$HOME")
        prefix = next(
            (
                item
                for item in prefixes
                if locator_value == item or locator_value.startswith(f"{item}/")
            ),
            None,
        )
        if prefix is None:
            raise DeclarationError("path locator must use a reviewed symbolic root")
        tail = locator_value.removeprefix(prefix).removeprefix("/")
        if tail and (
            "//" in tail
            or "\\" in tail
            or any(character in "*?[]$" for character in tail)
            or any(
                character.isspace() or ord(character) < 32 or ord(character) == 127
                for character in tail
            )
            or any(part in {"", ".", ".."} for part in tail.split("/"))
        ):
            raise DeclarationError("path locator is not canonical")
    return kind, locator_value


def _validate_discovery_id(value: object) -> str:
    if (
        type(value) is not str
        or re.fullmatch(r"runtime-site:[0-9a-f]{64}", value) is None
    ):
        raise DeclarationError("discovery id is invalid")
    return value


def _group_resources(
    snapshot: Mapping[str, bytes], candidates: Sequence[Mapping[str, Any]]
) -> tuple[Resource, ...]:
    groups: dict[tuple[str, str], list[Row]] = {}
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise DeclarationError("resource candidate must be an object")
        kind = candidate.get("kind")
        locator = candidate.get("locator")
        valid_kind, locator_value = _validate_locator(kind, locator)
        _validate_discovery_id(candidate.get("discovery_id"))
        groups.setdefault((valid_kind, locator_value), []).append(dict(candidate))

    resources: list[Resource] = []
    for (kind, locator_value), group in sorted(groups.items()):
        locator_types = {cast(str, row["locator"]["type"]) for row in group}
        if len(locator_types) != 1:
            raise DeclarationError(f"locator type conflict for {kind}:{locator_value}")
        discovery_ids = [
            _validate_discovery_id(row.get("discovery_id")) for row in group
        ]
        if len(discovery_ids) != len(set(discovery_ids)):
            raise DeclarationError(
                f"duplicate discovery in resource group: {kind}:{locator_value}"
            )
        resource: Resource = {
            "id": _resource_id(kind, locator_value),
            "kind": kind,
            "locator": {"type": locator_types.pop(), "value": locator_value},
            "protects": [],
            "evidence": _evidence_for_group(snapshot, group),
            "discovery_ids": sorted(discovery_ids),
        }
        resources.append(resource)

    ids = [str(row["id"]) for row in resources]
    if len(ids) != len(set(ids)):
        raise DeclarationError("duplicate final resource id")
    by_key: dict[tuple[str, str], Resource] = {
        (str(row["kind"]), str(row["locator"]["value"])): row for row in resources
    }
    lock_locators = {
        str(row["locator"]["value"]) for row in resources if row["kind"] == "lock"
    }
    if lock_locators and lock_locators != set(_LOCK_PROTECTS_LOCATORS):
        raise DeclarationError(
            "reviewed lock protects mapping is incomplete or has extras"
        )
    for resource in resources:
        if resource["kind"] != "lock":
            continue
        locator_value = str(resource["locator"]["value"])
        targets = _LOCK_PROTECTS_LOCATORS.get(locator_value)
        if targets is None:
            raise DeclarationError(
                f"lock has no reviewed protects mapping: {locator_value}"
            )
        if not targets or len(targets) != len(set(targets)):
            raise DeclarationError(
                f"lock protects mapping is empty or duplicate: {locator_value}"
            )
        missing = [target for target in targets if target not in by_key]
        if missing:
            raise DeclarationError(
                f"lock protects unknown resource references: {missing}"
            )
        resource["protects"] = sorted({str(by_key[target]["id"]) for target in targets})
    _validate_resource_collection(
        resources, _LOCK_PROTECTS_LOCATORS if lock_locators else {}
    )
    return tuple(resources)


def _validate_resource_collection(
    resources: Sequence[Mapping[str, Any]],
    lock_mapping: Mapping[str, tuple[tuple[str, str], ...]],
) -> None:
    by_key: dict[tuple[str, str], Mapping[str, Any]] = {}
    by_id: dict[str, Mapping[str, Any]] = {}
    for resource in resources:
        _validate_exact_resource(resource)
        kind, locator_value = _validate_locator(
            resource.get("kind"), resource.get("locator")
        )
        resource_id = resource.get("id")
        if type(resource_id) is not str or resource_id != _resource_id(
            kind, locator_value
        ):
            raise DeclarationError(
                "resource id does not match kind-scoped locator identity"
            )
        key = (kind, locator_value)
        if key in by_key or resource_id in by_id:
            raise DeclarationError("duplicate final resource")
        protects = resource.get("protects")
        evidence = resource.get("evidence")
        discovery_ids = resource.get("discovery_ids")
        if not isinstance(protects, (list, tuple)) or any(
            type(item) is not str for item in protects
        ):
            raise DeclarationError("resource protects is invalid")
        if len(protects) != len(set(protects)):
            raise DeclarationError("resource protects contains duplicates")
        if not isinstance(evidence, (list, tuple)) or not evidence:
            raise DeclarationError("resource evidence is invalid")
        for item in evidence:
            if not isinstance(item, Mapping) or set(item) != {"path", "line"}:
                raise DeclarationError("resource evidence entry is invalid")
            _validate_canonical_relative_path(
                item["path"], label="resource evidence path"
            )
            if type(item["line"]) is not int or item["line"] < 1:
                raise DeclarationError("resource evidence line is invalid")
        if not isinstance(discovery_ids, (list, tuple)) or not discovery_ids:
            raise DeclarationError("resource discovery ids are invalid")
        validated_ids = [_validate_discovery_id(item) for item in discovery_ids]
        if validated_ids != sorted(validated_ids) or len(validated_ids) != len(
            set(validated_ids)
        ):
            raise DeclarationError(
                "resource discovery ids are not unique canonical order"
            )
        by_key[key] = resource
        by_id[resource_id] = resource

    lock_locators = {locator for (kind, locator) in by_key if kind == "lock"}
    if lock_locators != set(lock_mapping):
        raise DeclarationError(
            "lock protects mapping does not exactly cover lock resources"
        )
    for (kind, locator_value), resource in by_key.items():
        actual = tuple(resource["protects"])
        if kind != "lock":
            if actual:
                raise DeclarationError("non-lock resource cannot protect resources")
            continue
        targets = lock_mapping[locator_value]
        if not targets or len(targets) != len(set(targets)):
            raise DeclarationError("lock protects mapping is empty or duplicate")
        missing = [target for target in targets if target not in by_key]
        if missing:
            raise DeclarationError(
                f"lock protects unknown resource references: {missing}"
            )
        expected = tuple(sorted(str(by_key[target]["id"]) for target in targets))
        if actual != expected:
            raise DeclarationError(
                f"lock protects declaration drifted: {locator_value}"
            )


def _ensure_unique_discoveries(rows: Sequence[Row]) -> None:
    identifiers: list[str] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise DeclarationError("discovery row must be an object")
        identifiers.append(_validate_discovery_id(row.get("discovery_id")))
    duplicates = sorted(
        value for value, count in Counter(identifiers).items() if count > 1
    )
    if duplicates:
        raise DeclarationError(f"duplicate discovery ids: {duplicates}")


def _assert_no_owner_fields(value: Any, path: str = "root") -> None:
    if isinstance(value, Mapping):
        forbidden = sorted(_OWNER_FIELDS & set(value))
        if forbidden:
            raise DeclarationError(
                f"forbidden owner/policy fields at {path}: {forbidden}"
            )
        for key, item in value.items():
            _assert_no_owner_fields(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_no_owner_fields(item, f"{path}[{index}]")


def discover_concrete(
    source: Mapping[str, bytes] | CommittedSnapshot,
) -> ConcreteDeclarations:
    """Discover only concrete structural declarations from explicit source bytes."""

    snapshot, source_head = _snapshot_input(source)
    index = _SourceIndex(snapshot)
    rows = [
        *_ast_discovery(index),
        *_explicit_state_discovery(snapshot),
        *_socket_discovery(snapshot),
        *_worker_discovery(snapshot),
        *_lock_protocol_discovery(index),
    ]
    _ensure_unique_discoveries(rows)
    rows.sort(key=lambda row: (str(row["path"]), int(row["line"]), str(row["symbol"])))
    for row in rows:
        _validate_evidence(snapshot, {"path": row["path"], "line": row["line"]})
    resources = tuple(row for row in rows if row["classification"] == "resource")
    exclusions = tuple(row for row in rows if row["classification"] == "exclusion")
    lock_protocols = tuple(
        row for row in rows if row["classification"] == "lock_protocol"
    )
    grouped = _group_resources(snapshot, resources)
    result = ConcreteDeclarations(
        source_head=source_head,
        rows=_frozen_rows(rows),
        resource_candidates=_frozen_rows(resources),
        exclusion_candidates=_frozen_rows(exclusions),
        lock_protocol_candidates=_frozen_rows(lock_protocols),
        resources=_frozen_rows(grouped),
    )
    _assert_no_owner_fields(result.__dict__)
    return result


def _id_set_hash(rows: Sequence[Mapping[str, Any]], field: str) -> str:
    identifiers = sorted(str(row[field]) for row in rows)
    return hashlib.sha256(_canonical_bytes(identifiers)).hexdigest()


_FROZEN_DECLARATION_SHA256 = (
    "03c456e12a698dedf78f2618851b18c1916b1eb45f1b326b665af6959e8bccef"
)


def _declaration_sha256(result: ConcreteDeclarations) -> str:
    payload = {
        "source_head": result.source_head,
        "rows": result.rows,
        "resource_candidates": result.resource_candidates,
        "exclusion_candidates": result.exclusion_candidates,
        "lock_protocol_candidates": result.lock_protocol_candidates,
        "resources": result.resources,
    }
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _validate_frozen(result: ConcreteDeclarations) -> None:
    if type(result) is not ConcreteDeclarations:
        raise DeclarationError("frozen declarations have an invalid container type")
    expected_lengths = {
        "rows": 732,
        "resource_candidates": 490,
        "exclusion_candidates": 138,
        "lock_protocol_candidates": 104,
        "resources": 454,
    }
    for field, expected in expected_lengths.items():
        rows = cast(Sequence[Mapping[str, Any]], getattr(result, field))
        if len(rows) != expected:
            raise DeclarationError(
                f"frozen {field} count mismatch: {len(rows)} != {expected}"
            )
        id_field = "id" if field == "resources" else "discovery_id"
        actual_hash = _id_set_hash(rows, id_field)
        if actual_hash != _FROZEN_ID_HASHES[field]:
            raise DeclarationError(
                f"frozen {field} id-set hash mismatch: {actual_hash}"
            )
    kinds = Counter(str(row["kind"]) for row in result.resources)
    if kinds != Counter(
        {
            "artifact": 182,
            "lock": 21,
            "queue": 5,
            "schema": 154,
            "socket": 6,
            "worker": 86,
        }
    ):
        raise DeclarationError(f"frozen resource kind counts mismatch: {dict(kinds)}")
    direct = [
        row
        for row in result.lock_protocol_candidates
        if not str(row["operation"]).startswith("helper:")
    ]
    direct_shape = (
        len(direct),
        len({str(row["module"]) for row in direct}),
        len({(str(row["module"]), str(row["scope"])) for row in direct}),
    )
    if direct_shape != (52, 37, 51):
        raise DeclarationError(f"frozen direct flock counts mismatch: {direct_shape}")
    declaration_sha256 = _declaration_sha256(result)
    if declaration_sha256 != _FROZEN_DECLARATION_SHA256:
        raise DeclarationError(
            f"frozen full declaration seal mismatch: {declaration_sha256}"
        )


def load_frozen(source: CommittedSnapshot) -> ConcreteDeclarations:
    """Load and seal the exact accepted frozen source declaration set."""

    if (
        not isinstance(source, CommittedSnapshot)
        or source.revision != FROZEN_SOURCE_REVISION
    ):
        raise DeclarationError(
            f"frozen source must be exact revision {FROZEN_SOURCE_REVISION}"
        )
    _require_snapshot_files_sha256(source, _FROZEN_SOURCE_FILES_SHA256, label="frozen")
    result = discover_concrete(source)
    _validate_frozen(result)
    return result


_FEEDBACK_ARTIFACT_ID = (
    "runtime-resource:8dcd7e44878e638768b6866bb2b002ff59e5c75a9cecba7a8d54f4794897a417"
)
_FEEDBACK_LOCK_ID = (
    "runtime-resource:036b07eabe6998c8389bc2221e3074aca1b59dad202458cde1836ad40feaa512"
)
_FEEDBACK_LOCK_DISCOVERY_ID = (
    "runtime-site:f3310a069b43f8fdec587d108ff04c67a9686fea61389148ee7e192dd86528c4"
)
_EFFECTIVE_LOCK_PROTECTS_LOCATORS = {
    **_LOCK_PROTECTS_LOCATORS,
    "$CHRONOVISOR_ROOT/recall/feedback.jsonl.lock": (
        ("artifact", "$CHRONOVISOR_ROOT/recall/feedback.jsonl"),
    ),
}


def _require_line_contains(
    snapshot: Mapping[str, bytes], path: str, line: int, needle: str
) -> None:
    text = _text(snapshot, path)
    lines = text.splitlines()
    if line > len(lines) or needle not in lines[line - 1]:
        raise DeclarationError(
            f"effective source evidence does not match {path}:{line}: {needle}"
        )


def _validate_exact_resource(resource: Mapping[str, Any]) -> None:
    expected = {"id", "kind", "locator", "protects", "evidence", "discovery_ids"}
    keys = tuple(resource)
    if any(type(key) is not str for key in keys):
        raise DeclarationError("resource field names must be exact strings")
    string_keys = keys
    if set(string_keys) != expected:
        raise DeclarationError(
            f"resource fields are not structural-only: {sorted(string_keys)}"
        )


def apply_resource_amendment(
    frozen: ConcreteDeclarations,
    effective_source: CommittedSnapshot,
) -> ResourceAmendment:
    """Add the reviewed feedback sidecar lock without replacing the frozen scan."""

    _validate_frozen(frozen)
    if frozen.source_head != FROZEN_SOURCE_REVISION:
        raise DeclarationError(
            "resource amendment base is not the frozen declaration set"
        )
    if (
        not isinstance(effective_source, CommittedSnapshot)
        or effective_source.revision != EFFECTIVE_FEEDBACK_REVISION
        or effective_source.manifest_kind != SOURCE_MANIFEST_KIND
    ):
        raise DeclarationError(
            f"feedback amendment requires exact effective revision {EFFECTIVE_FEEDBACK_REVISION}"
        )
    _require_snapshot_files_sha256(
        effective_source, _EFFECTIVE_SOURCE_FILES_SHA256, label="effective feedback"
    )
    snapshot, _source_head = _snapshot_input(effective_source)
    current = discover_concrete(effective_source)
    if len(current.rows) != 731 or len(current.lock_protocol_candidates) != 103:
        raise DeclarationError(
            "effective source concrete scan does not match the reviewed delta"
        )

    path = "src/chronovisor/recall/recall_runtime.py"
    _require_line_contains(
        snapshot, path, 51, 'RECALL_FEEDBACK_FILE = RECALL_DIR / "feedback.jsonl"'
    )
    _require_line_contains(
        snapshot, path, 3073, 'lock_path = path.with_suffix(path.suffix + ".lock")'
    )
    _require_line_contains(
        snapshot, path, 3078, "fcntl.flock(descriptor, fcntl.LOCK_EX)"
    )

    if (
        _resource_id("lock", "$CHRONOVISOR_ROOT/recall/feedback.jsonl.lock")
        != _FEEDBACK_LOCK_ID
    ):
        raise DeclarationError("feedback lock resource identity drifted")
    candidate: Row = {
        "classification": "resource",
        "path": path,
        "line": 51,
        "module": "chronovisor.recall.recall_runtime",
        "symbol": "RECALL_FEEDBACK_FILE.sidecar-lock",
        "kind": "lock",
        "locator": {
            "type": "path",
            "value": "$CHRONOVISOR_ROOT/recall/feedback.jsonl.lock",
        },
    }
    if _discovery_id(candidate) != _FEEDBACK_LOCK_DISCOVERY_ID:
        raise DeclarationError("feedback lock discovery identity drifted")
    resource: Resource = {
        "id": _FEEDBACK_LOCK_ID,
        "kind": "lock",
        "locator": {
            "type": "path",
            "value": "$CHRONOVISOR_ROOT/recall/feedback.jsonl.lock",
        },
        "protects": [_FEEDBACK_ARTIFACT_ID],
        "evidence": [
            {"path": path, "line": 51},
            {"path": path, "line": 3073},
            {"path": path, "line": 3078},
        ],
        "discovery_ids": [_FEEDBACK_LOCK_DISCOVERY_ID],
    }
    _validate_exact_resource(resource)
    for evidence in cast(list[Mapping[str, Any]], resource["evidence"]):
        _validate_evidence(snapshot, evidence)

    existing_ids = {str(row["id"]) for row in frozen.resources}
    if _FEEDBACK_LOCK_ID in existing_ids:
        raise DeclarationError("resource amendment duplicates an existing resource")
    if _FEEDBACK_ARTIFACT_ID not in existing_ids:
        raise DeclarationError("resource amendment protects an unknown resource")
    resources = tuple(
        sorted(
            (*frozen.resources, resource),
            key=lambda row: (str(row["kind"]), str(row["locator"]["value"])),
        )
    )
    if len({str(row["id"]) for row in resources}) != len(resources):
        raise DeclarationError("resource amendment created duplicate final resources")
    _validate_resource_collection(resources, _EFFECTIVE_LOCK_PROTECTS_LOCATORS)
    payload = {
        "schema_version": 1,
        "kind": "resource",
        "operation": "add",
        "base_source_head": FROZEN_SOURCE_REVISION,
        "effective_source_head": EFFECTIVE_FEEDBACK_REVISION,
        "frozen_count": {"resources": 454, "by_kind": {"lock": 21}},
        "effective_count": {"resources": 455, "by_kind": {"lock": 22}},
        "resource": resource,
    }
    _assert_no_owner_fields(payload)
    return ResourceAmendment(
        amendment_id=_semantic_id("runtime-amendment", payload),
        payload=cast(Mapping[str, Any], _deep_freeze(payload)),
        resources=_frozen_rows(resources),
    )


_REMOVED_FEEDBACK_PROTOCOL_IDS = frozenset(
    {
        "runtime-site:0e7d4ca4e49982b83440b9592c1b1eb9d53f9d254335ab51ac077065f1633a6a",
        "runtime-site:58e71139ca0945c3405f3956cb7705a481fb74639f7e0e50b3333bd5dacc5a81",
    }
)
_ADDED_FEEDBACK_PROTOCOL_IDS = frozenset(
    {"runtime-site:bf488c9fa72d0998f8ff25c78053ad9f6f378a32b1e92dcfebf7bffdae807beb"}
)


def record_protocol_transition(
    frozen: ConcreteDeclarations,
    effective_source: CommittedSnapshot,
) -> ProtocolTransition:
    """Record the exact reviewed feedback-lock protocol replacement."""

    _validate_frozen(frozen)
    if frozen.source_head != FROZEN_SOURCE_REVISION:
        raise DeclarationError(
            "protocol transition base is not the frozen declaration set"
        )
    if (
        not isinstance(effective_source, CommittedSnapshot)
        or effective_source.revision != EFFECTIVE_FEEDBACK_REVISION
        or effective_source.manifest_kind != SOURCE_MANIFEST_KIND
    ):
        raise DeclarationError(
            f"protocol transition requires exact effective revision {EFFECTIVE_FEEDBACK_REVISION}"
        )
    _require_snapshot_files_sha256(
        effective_source, _EFFECTIVE_SOURCE_FILES_SHA256, label="effective protocol"
    )
    snapshot, _source_head = _snapshot_input(effective_source)
    current = discover_concrete(effective_source)
    if len(current.rows) != 731 or len(current.lock_protocol_candidates) != 103:
        raise DeclarationError(
            "effective protocol scan does not match the reviewed delta"
        )

    frozen_by_id = {
        str(row["discovery_id"]): row for row in frozen.lock_protocol_candidates
    }
    current_by_id = {
        str(row["discovery_id"]): row for row in current.lock_protocol_candidates
    }
    removed_ids = frozenset(frozen_by_id) - frozenset(current_by_id)
    added_ids = frozenset(current_by_id) - frozenset(frozen_by_id)
    if removed_ids != _REMOVED_FEEDBACK_PROTOCOL_IDS:
        raise DeclarationError(
            f"reviewed removed protocol ids drifted: {sorted(removed_ids)}"
        )
    if added_ids != _ADDED_FEEDBACK_PROTOCOL_IDS:
        raise DeclarationError(
            f"reviewed added protocol ids drifted: {sorted(added_ids)}"
        )
    removed = [dict(frozen_by_id[identifier]) for identifier in sorted(removed_ids)]
    added = [dict(current_by_id[identifier]) for identifier in sorted(added_ids)]

    recall_path = "src/chronovisor/recall/recall_runtime.py"
    correction_path = "src/chronovisor/recall/content_correction.py"
    _require_line_contains(snapshot, recall_path, 3064, "def _feedback_exclusive_lock(")
    _require_line_contains(
        snapshot,
        recall_path,
        3073,
        'lock_path = path.with_suffix(path.suffix + ".lock")',
    )
    _require_line_contains(
        snapshot, recall_path, 3078, "fcntl.flock(descriptor, fcntl.LOCK_EX)"
    )
    _require_line_contains(
        snapshot,
        correction_path,
        2145,
        "with _feedback_exclusive_lock(RECALL_FEEDBACK_FILE):",
    )
    _require_line_contains(
        snapshot,
        correction_path,
        3535,
        "with _feedback_exclusive_lock(RECALL_FEEDBACK_FILE):",
    )
    _require_line_contains(
        snapshot,
        recall_path,
        3019,
        "with _feedback_exclusive_lock(RECALL_FEEDBACK_FILE):",
    )
    shared_helper = {
        "path": recall_path,
        "module": "chronovisor.recall.recall_runtime",
        "symbol": "_feedback_exclusive_lock",
        "definition_line": 3064,
        "sidecar_derivation_line": 3073,
        "acquisition_line": 3078,
        "callsites": [
            {
                "path": correction_path,
                "line": 2145,
                "scope": "_retract_legacy_unfiltered_page_ignored_feedback",
            },
            {
                "path": correction_path,
                "line": 3535,
                "scope": "_record_wrong_retrieval",
            },
            {"path": recall_path, "line": 3019, "scope": "append_feedback"},
        ],
    }
    payload = {
        "schema_version": 1,
        "kind": "protocol_transition",
        "operation": "replace",
        "base_source_head": FROZEN_SOURCE_REVISION,
        "effective_source_head": EFFECTIVE_FEEDBACK_REVISION,
        "counts": {
            "frozen_lock_protocol_sites": 104,
            "effective_lock_protocol_sites": 103,
        },
        "removed": removed,
        "added": added,
        "shared_helper": shared_helper,
    }
    _assert_no_owner_fields(payload)
    return ProtocolTransition(
        transition_id=hashlib.sha256(_canonical_bytes(payload)).hexdigest(),
        payload=cast(Mapping[str, Any], _deep_freeze(payload)),
        lock_protocol_candidates=current.lock_protocol_candidates,
    )


__all__ = [
    "EFFECTIVE_FEEDBACK_REVISION",
    "ConcreteDeclarations",
    "DeclarationError",
    "ProtocolTransition",
    "ResourceAmendment",
    "apply_resource_amendment",
    "discover_concrete",
    "load_frozen",
    "record_protocol_transition",
]
