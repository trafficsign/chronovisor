"""Crash-safe offline cutover for a validated OKF migration workspace."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal, TypeVar

from chronovisor.core.canonical_json import canonical_json_line_bytes_strict
from chronovisor.core.durable_state import (
    StateSealError,
    atomic_write_bytes,
    file_lock,
    fsync_directory,
    okf_writer_lock,
    open_directory_nofollow,
    open_regular_nofollow,
    seal_object,
    verify_sealed_object,
)
from chronovisor.core.okf_v02 import OKF_VERSION
from chronovisor.core.okf_workspace import (
    JOURNAL_SCHEMA,
    MANIFEST_SCHEMA,
    RESTART_REFUSAL_FILENAME,
    SCHEMA_VERSION,
    SENTINEL_SCHEMA,
)

CutoverState = Literal[
    "aborted",
    "abort-in-progress",
    "committed-needs-rebuild",
    "rebuild-in-progress",
    "sealed-rebuild",
    "rollback-drill-complete",
    "recutover-in-progress",
    "finalized-v2",
    "rollback-complete",
]
FaultInjector = Callable[[str], None]
_OldActivity = tuple[int, str] | None

_RUN_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_MANIFEST_MAX_BYTES = 64 * 1024 * 1024
_MISSING = object()
_Value = TypeVar("_Value")
_MOVE_NAMES = (
    "backup-pages",
    "backup-system",
    "backup-activity",
    "publish-pages",
    "publish-system",
    "publish-activity",
)
CUTOVER_FAULT_POINTS = (
    "before-start-journal",
    "after-start-journal",
    "after-start-sentinel",
    "after-backup-directory",
    *(
        point
        for move in _MOVE_NAMES
        for point in (
            f"{move}:before-intent-journal",
            f"{move}:after-intent-journal",
            f"{move}:after-rename",
            f"{move}:after-fsync",
            f"{move}:after-completion-journal",
        )
    ),
    "before-terminal-journal",
    "after-terminal-journal",
    "after-terminal-sentinel",
)
RECEIPT_SCHEMA = "chronovisor.okf-migration-receipt.v1"
FINAL_RECEIPT_SCHEMA = "chronovisor.okf-migration-receipt.v2"
FINAL_RECEIPT_VERSION = 2
RECEIPT_FILENAME = "receipt.json"
REBUILD_PROOF_SCHEMA = "chronovisor.okf-derived-rebuild.v1"
REBUILD_PROOF_FILENAME = "rebuild-proof.json"
_FINAL_STATUS_MAPPING = {
    "missing": "stable",
    "active": "stable",
    "draft": "draft",
    "stable": "stable",
    "deprecated": "deprecated",
    "archived": "deprecated",
}
CLEANUP_FAULT_POINTS = (
    "after-cleanup-journal",
    "before-receipt-write",
    "after-receipt-write",
    *(
        point
        for name in (
            "staging",
            "rollback-backup",
            "dry-run-manifest",
            "cutover-lock",
        )
        for point in (f"before-remove-{name}", f"after-remove-{name}")
    ),
    "before-journal-remove",
    "after-journal-remove",
)
_DRILL_ASSET_NAMES = ("pages", "system", "activity")
ROLLBACK_DRILL_FAULT_POINTS = (
    "before-rollback-drill-journal",
    "after-rollback-drill-journal",
    *(
        point
        for name in _DRILL_ASSET_NAMES
        for move in (f"drill-stage-{name}", f"drill-restore-{name}")
        for point in (
            f"{move}:before-intent-journal",
            f"{move}:after-intent-journal",
            f"{move}:after-rename",
            f"{move}:after-fsync",
            f"{move}:after-completion-journal",
        )
    ),
    "after-rollback-drill-terminal-journal",
)
RECUTOVER_FAULT_POINTS = (
    "before-recutover-journal",
    "after-recutover-journal",
    *(
        point
        for name in _DRILL_ASSET_NAMES
        for move in (f"recutover-backup-{name}", f"recutover-publish-{name}")
        for point in (
            f"{move}:before-intent-journal",
            f"{move}:after-intent-journal",
            f"{move}:after-rename",
            f"{move}:after-fsync",
            f"{move}:after-completion-journal",
        )
    ),
    "after-finalized-journal",
    "after-finalized-sentinel-remove",
)
ABORT_FAULT_POINTS = (
    "before-abort-journal",
    "after-abort-journal-before-sentinel",
    "after-abort-journal",
    *(
        point
        for name in (
            "abort-stage-pages",
            "abort-stage-system",
            "abort-stage-activity",
            "abort-restore-pages",
            "abort-restore-system",
            "abort-restore-activity",
        )
        for point in (
            f"{name}:before-intent-journal",
            f"{name}:after-intent-journal",
            f"{name}:after-rename",
            f"{name}:after-fsync",
            f"{name}:after-completion-journal",
        )
    ),
    "before-abort-workspace-rename",
    "after-abort-workspace-rename",
    "before-abort-workspace-remove",
    "after-abort-workspace-remove",
    "before-abort-marker-remove",
    "after-abort-marker-remove",
)
FINALIZE_FAULT_POINTS = (
    "before-final-receipt-write",
    "after-final-receipt-write",
    *(
        point
        for name in (
            "legacy-index",
            "legacy-log",
            "legacy-schema",
            "staging",
            "rollback-backup",
            "derived-rebuild",
            "rebuild-proof",
            "dry-run-manifest",
            "restart-refusal",
            "cutover-lock",
            "journal",
        )
        for point in (f"before-final-remove-{name}", f"after-final-remove-{name}")
    ),
)


@dataclass(frozen=True, slots=True)
class _Expected:
    old_pages: dict[str, str]
    old_system: dict[str, str]
    new_pages: dict[str, str]
    new_system: dict[str, str]
    raw: dict[str, tuple[int, str]]
    reserved: dict[str, str]
    prepared_activity: tuple[int, str]
    new_activity: tuple[int, str]
    activity_prefix: tuple[int, str]
    activity_event_ids_sha256: str


@dataclass(frozen=True, slots=True)
class _Context:
    source: Path
    runtime: Path
    workspace: Path
    staging: Path
    backup: Path
    journal: Path
    sentinel: Path
    manifest_sha256: str
    expected: _Expected


@dataclass(frozen=True, slots=True)
class _Asset:
    name: str
    live: Path
    staged: Path
    backup: Path
    old: Mapping[str, str] | tuple[int, str] | None
    new: Mapping[str, str] | tuple[int, str]
    directory: bool


@dataclass(frozen=True, slots=True)
class OKFStartupDecision:
    """Content-free startup decision derived only from durable disk state."""

    allowed: bool
    layout: str
    state: str
    category: str
    run_id: str | None = None


@dataclass(frozen=True, slots=True)
class OKFRebuildGate:
    """Content-free identity needed by the offline derived rebuild."""

    source_root: Path
    workspace: Path
    manifest_sha256: str
    activity_prefix_length: int
    activity_prefix_sha256: str
    activity_event_ids_sha256: str
    derived_generation: str | None = None
    rebuild_proof_sha256: str | None = None
    stable_page_count: int | None = None


@dataclass(slots=True)
class OKFRebuildSession:
    """Exclusive offline migration session used by the derived coordinator."""

    gate: OKFRebuildGate
    _context: _Context
    _active: bool = True

    def publish_proof(self, payload: Mapping[str, object]) -> str:
        if not self._active:
            raise RuntimeError("OKF rebuild session is no longer active")
        if self.gate.derived_generation is not None:
            raise RuntimeError("OKF rebuild is already sealed")
        return _publish_okf_rebuild_proof_locked(self._context, payload)

    def seal(
        self,
        *,
        derived_generation: str,
        rebuild_proof_sha256: str,
    ) -> None:
        if not self._active:
            raise RuntimeError("OKF rebuild session is no longer active")
        if self.gate.derived_generation is not None:
            raise RuntimeError("OKF rebuild is already sealed")
        _seal_okf_rebuild_locked(
            self._context,
            derived_generation=derived_generation,
            rebuild_proof_sha256=rebuild_proof_sha256,
        )


class OKFStartupBlocked(RuntimeError):
    """Startup was refused without exposing paths or migration contents."""

    def __init__(self, decision: OKFStartupDecision) -> None:
        self.decision = decision
        super().__init__(f"OKF startup blocked: {decision.category}")


_WORKSPACE_ENTRIES = {
    "cutover.lock": "file",
    "dry-run-manifest.json": "file",
    "derived-rebuild": "directory",
    "journal.json": "file",
    RECEIPT_FILENAME: "file",
    REBUILD_PROOF_FILENAME: "file",
    RESTART_REFUSAL_FILENAME: "file",
    "rollback-backup": "directory",
    "staging": "directory",
}
_LEGACY_CLEANUP_ENTRIES = {
    name: kind
    for name, kind in _WORKSPACE_ENTRIES.items()
    if name not in {"derived-rebuild", REBUILD_PROOF_FILENAME}
}
_BOOTSTRAP_ENTRIES = {
    "config.toml": "file",
    "logs": "directory",
    "pages": "directory",
    "raw": "directory",
    "runtime": "directory",
    "system": "directory",
}
_BOOTSTRAP_RUNTIME_ENTRIES = {"okf-writer.lock": "file"}
_ROOT_RESERVED = ("index.md", "log.md", "schema.md")
_REBUILD_CORPUS_FIELDS = {
    "stable_page_count": "count",
    "stable_path_set_sha256": "sha256",
    "stable_source_set_sha256": "sha256",
    "stable_uid_set_sha256": "sha256",
}
_REBUILD_COMPONENT_FIELDS = {
    "registry": {
        "generation": "count",
        "stable_count": "count",
        "sha256": "sha256",
    },
    "uid_links": {
        "edge_count": "count",
        "unresolved_count": "count",
        "sha256": "sha256",
    },
    "portable_index": {
        "page_count": "count",
        "link_count": "count",
        "sha256": "sha256",
    },
    "index_store": {
        "page_count": "count",
        "pages_sha256": "sha256",
        "backlinks_sha256": "sha256",
    },
    "lexical": {"page_count": "count", "sha256": "sha256"},
    "semantic": {
        "page_count": "count",
        "document_count": "count",
        "corpus_sha256": "sha256",
        "generation_sha256": "sha256",
        "manifest_sha256": "sha256",
    },
    "knowledge_graph": {
        "page_count": "count",
        "relation_count": "count",
        "relation_set_sha256": "sha256",
        "snapshot_sha256": "sha256",
        "builder_sha256": "sha256",
        "external_model_calls": "count",
    },
    "cortex": {
        "node_count": "count",
        "link_count": "count",
        "typed_relation_count": "count",
        "sha256": "sha256",
        "authority_enabled": "bool",
        "runtime_state_present": "bool",
    },
    "invalidation": {
        "changed_page_count": "count",
        "source_set_sha256": "sha256",
        "target_count": "count",
    },
}


def discover_okf_startup(source_root: Path, runtime_root: Path) -> OKFStartupDecision:
    """Inspect one root without mutation and return its fail-closed startup state."""

    try:
        if _has_symlink_component(source_root) or _has_symlink_component(runtime_root):
            return _blocked("unsafe_source_root")
        source_kind = _path_kind(source_root)
        if source_kind not in {"absent", "directory"}:
            return _blocked("unsafe_source_root")

        runtime_kind = _path_kind(runtime_root)
        if runtime_kind not in {"absent", "directory"}:
            return _blocked("unsafe_runtime_root")
        migrations = runtime_root / "migrations"
        migrations_kind = _path_kind(migrations)
        if migrations_kind != "absent":
            if migrations_kind != "directory":
                return _blocked("unsafe_migration_directory")
            return _discover_migration(source_root, runtime_root, migrations)
        if source_kind == "absent":
            return OKFStartupDecision(True, "bootstrap", "uninitialized", "ok")
        return _discover_unmigrated(source_root)
    except (OSError, RuntimeError, TypeError, ValueError):
        return _blocked("startup_inspection_failed")


def require_okf_startup_allowed(
    source_root: Path, runtime_root: Path
) -> OKFStartupDecision:
    """Return the startup decision or raise a content-free refusal."""

    decision = discover_okf_startup(source_root, runtime_root)
    if not decision.allowed:
        raise OKFStartupBlocked(decision)
    return decision


def _discover_migration(
    source_root: Path, runtime_root: Path, migrations: Path
) -> OKFStartupDecision:
    entries = _directory_entries(migrations)
    if not entries:
        return _blocked("migration_residue")
    if any(
        not _RUN_ID_RE.fullmatch(name) or kind != "directory"
        for name, kind in entries.items()
    ):
        return _blocked("unsafe_migration_directory")
    if len(entries) != 1:
        return _blocked("multiple_migrations")

    run_id = next(iter(entries))
    workspace = migrations / run_id
    unsafe = _tree_has_unsafe_entry(workspace)
    if unsafe:
        return _blocked("unsafe_migration_workspace", run_id=run_id)
    workspace_entries = _directory_entries(workspace)
    if workspace_entries == {RECEIPT_FILENAME: "file"}:
        try:
            receipt = _read_any_receipt(workspace / RECEIPT_FILENAME, run_id)
            receipt_state = receipt["state"]
            if receipt["schema"] == FINAL_RECEIPT_SCHEMA:
                _require_final_receipt_layout(source_root, runtime_root, receipt)
            else:
                _require_receipt_layout(
                    source_root, runtime_root, "rollback-complete"
                )
        except (OSError, TypeError, ValueError):
            return _blocked("migration_receipt_invalid", run_id=run_id)
        layout = "okf_v0_2" if receipt["schema"] == FINAL_RECEIPT_SCHEMA else "legacy"
        return OKFStartupDecision(True, layout, str(receipt_state), "ok", run_id)
    if RECEIPT_FILENAME in workspace_entries:
        return _blocked("cleanup_incomplete", run_id=run_id)
    if any(
        _WORKSPACE_ENTRIES.get(name) != kind for name, kind in workspace_entries.items()
    ):
        return _blocked("unsafe_migration_workspace", run_id=run_id)
    if not {"dry-run-manifest.json", "journal.json"}.issubset(workspace_entries):
        return _blocked("migration_proof_invalid", run_id=run_id)
    try:
        journal = _read_canonical_object(
            workspace / "journal.json", "migration journal"
        )
    except (OSError, TypeError, ValueError):
        return _blocked("migration_proof_invalid", run_id=run_id)
    observed_state = journal.get("state")
    state = (
        observed_state
        if observed_state
        in {
            "prepared",
            "in-progress",
            "committed-needs-rebuild",
            "rebuild-in-progress",
            "abort-in-progress",
            "sealed-rebuild",
            "rollback-drill-complete",
            "recutover-in-progress",
            "finalized-v2",
            "committed",
            "rollback-complete",
        }
        else "unknown"
    )
    if state in {"prepared", "in-progress"}:
        if RESTART_REFUSAL_FILENAME in workspace_entries:
            return _blocked("restart_refusal_active", state=state, run_id=run_id)
        return _blocked("migration_nonterminal", state=state, run_id=run_id)
    if state == "committed":
        return _blocked("migration_proof_invalid", state=state, run_id=run_id)
    if state == "unknown":
        return _blocked("migration_nonterminal", state=state, run_id=run_id)
    if state == "rebuild-in-progress":
        return _blocked("rebuild_in_progress", state=state, run_id=run_id)
    if state == "abort-in-progress":
        return _blocked("abort_in_progress", state=state, run_id=run_id)
    if state == "sealed-rebuild":
        return _blocked("rollback_drill_required", state=state, run_id=run_id)
    if state == "rollback-drill-complete":
        return _blocked("recutover_required", state=state, run_id=run_id)
    if state == "recutover-in-progress":
        return _blocked("recutover_in_progress", state=state, run_id=run_id)
    if state == "finalized-v2":
        if RESTART_REFUSAL_FILENAME in workspace_entries:
            return _blocked("recutover_in_progress", state=state, run_id=run_id)
        try:
            context = _context(source_root, runtime_root, run_id)
            finalized = _read_canonical_object(
                context.journal, "migration journal"
            )
            _require_finalized_journal(context, finalized)
        except (OSError, TypeError, ValueError):
            return _blocked("migration_proof_invalid", state=state, run_id=run_id)
        return OKFStartupDecision(True, "okf_v0_2", state, "ok", run_id)
    if journal.get("cleanup_in_progress") is True:
        return _blocked("cleanup_in_progress", state=state, run_id=run_id)
    if "cleanup_in_progress" in journal:
        return _blocked("migration_proof_invalid", state=state, run_id=run_id)
    try:
        _terminal_context, _journal, terminal = _require_terminal_proof(
            source_root, runtime_root, run_id
        )
    except (OSError, TypeError, ValueError):
        return _blocked("migration_proof_invalid", state=state, run_id=run_id)
    if terminal == "committed-needs-rebuild":
        return _blocked("rebuild_required", state=terminal, run_id=run_id)
    return OKFStartupDecision(
        True,
        "legacy",
        terminal,
        "ok",
        run_id,
    )


def _discover_unmigrated(source_root: Path) -> OKFStartupDecision:
    reserved = {name: _path_kind(source_root / name) for name in _ROOT_RESERVED}
    if any(kind not in {"absent", "file"} for kind in reserved.values()):
        return _blocked("unsafe_legacy_layout")
    present = sum(kind == "file" for kind in reserved.values())
    if 0 < present < len(_ROOT_RESERVED):
        return _blocked("partial_legacy_layout")
    if present == len(_ROOT_RESERVED):
        if _unsafe_legacy_root(source_root):
            return _blocked("unsafe_legacy_layout")
        return OKFStartupDecision(True, "legacy", "unmigrated", "ok")

    from chronovisor.core.live_layout import read_live_layout_proof

    proof_path = source_root / "runtime" / "bootstrap-layout.json"
    if _path_kind(proof_path) != "absent":
        if _path_kind(proof_path) != "file":
            return _blocked("unsafe_bootstrap_layout")
        proof = read_live_layout_proof(source_root)
        if proof is None:
            return _blocked("bootstrap_proof_invalid")
        if proof["state"] == "in-progress":
            if not _is_resumable_final_bootstrap(source_root):
                return _blocked("bootstrap_proof_invalid")
            return _blocked("bootstrap_in_progress", state="in-progress")
        if _is_canonical_live_layout(source_root, proof):
            return OKFStartupDecision(True, "okf_v0_2", "ready", "ok")
        return _blocked("bootstrap_proof_invalid")
    if _path_kind(source_root / "runtime" / "bootstrap-layout.lock") == "file":
        if _is_resumable_final_bootstrap(source_root):
            return _blocked("bootstrap_in_progress", state="in-progress")
        return _blocked("bootstrap_proof_invalid")

    entries = _directory_entries(source_root)
    if any(_BOOTSTRAP_ENTRIES.get(name) != kind for name, kind in entries.items()):
        return _blocked("unsafe_bootstrap_layout")
    for name, kind in entries.items():
        if kind != "directory":
            continue
        children = _directory_entries(source_root / name)
        if name == "runtime" and children == _BOOTSTRAP_RUNTIME_ENTRIES:
            continue
        if children:
            return _blocked("content_without_migration")
    return OKFStartupDecision(True, "bootstrap", "uninitialized", "ok")


def _is_canonical_live_layout(
    source_root: Path, proof: Mapping[str, object]
) -> bool:
    """Recognize one bounded, sealed final layout without scanning the corpus."""

    from chronovisor.core.activity_log import activity_prefix_matches
    from chronovisor.core.live_layout import (
        file_sha256_nofollow,
        valid_index_shape_nofollow,
    )

    pages = source_root / "pages"
    system = source_root / "system"
    activity = source_root / "runtime" / "activity.jsonl"
    required = (
        pages / "index.md",
        pages / "log.md",
        system / "schema.md",
        activity,
    )
    if any(_path_kind(path) != "file" for path in required):
        return False
    if _unsafe_legacy_root(source_root):
        return False
    try:
        if not valid_index_shape_nofollow(pages / "index.md"):
            return False
        if file_sha256_nofollow(pages / "log.md") != proof.get("log_sha256"):
            return False
        if file_sha256_nofollow(system / "schema.md") != proof.get("schema_sha256"):
            return False
        prefix = proof.get("activity_prefix")
        if not isinstance(prefix, dict):
            return False
        prefix_length = prefix.get("length")
        if not isinstance(prefix_length, int) or isinstance(prefix_length, bool):
            return False
        prefix_sha256 = prefix.get("sha256")
        if not isinstance(prefix_sha256, str) or not activity_prefix_matches(
            activity,
            length=prefix_length,
            sha256=prefix_sha256,
        ):
            return False
    except (OSError, RuntimeError, ValueError):
        return False
    return True


def _is_resumable_final_bootstrap(source_root: Path) -> bool:
    """Allow only exact partial files emitted by fresh init itself."""

    from chronovisor.core.live_layout import LIVE_LAYOUT_LOCK, LIVE_LAYOUT_PROOF
    from chronovisor.core.reserved_documents import (
        render_pages_index,
        render_pages_log,
    )
    from chronovisor.core.store import SCHEMA_CONTENT

    entries = _directory_entries(source_root)
    if any(_BOOTSTRAP_ENTRIES.get(name) != kind for name, kind in entries.items()):
        return False
    expected_directories = {
        "pages": {
            "index.md": render_pages_index(()),
            "log.md": render_pages_log(),
        },
        "system": {"schema.md": SCHEMA_CONTENT.encode("utf-8")},
    }
    for name, expected in expected_directories.items():
        directory = source_root / name
        if _path_kind(directory) == "absent":
            continue
        if _path_kind(directory) != "directory":
            return False
        children = _directory_entries(directory)
        if any(expected.get(child) is None or kind != "file" for child, kind in children.items()):
            return False
        try:
            if any((directory / child).read_bytes() != expected[child] for child in children):
                return False
        except OSError:
            return False
    for name in ("raw", "logs"):
        directory = source_root / name
        if _path_kind(directory) == "directory" and _directory_entries(directory):
            return False
    runtime = source_root / "runtime"
    if _path_kind(runtime) != "directory":
        return False
    runtime_entries = _directory_entries(runtime)
    allowed_runtime = {
        "okf-writer.lock",
        LIVE_LAYOUT_LOCK,
        LIVE_LAYOUT_PROOF,
        "activity.jsonl",
    }
    if any(name not in allowed_runtime or kind != "file" for name, kind in runtime_entries.items()):
        return False
    activity = runtime / "activity.jsonl"
    try:
        return _path_kind(activity) == "absent" or activity.read_bytes() == b""
    except OSError:
        return False


def _unsafe_legacy_root(source_root: Path) -> bool:
    entries = _directory_entries(source_root)
    if any(kind not in {"file", "directory"} for kind in entries.values()):
        return True
    for name in ("raw", "pages", "system", "runtime", "logs"):
        if name in entries and entries[name] != "directory":
            return True
        if (
            name in {"raw", "pages", "system"}
            and name in entries
            and _tree_has_unsafe_entry(source_root / name)
        ):
            return True
    return "config.toml" in entries and entries["config.toml"] != "file"


def _tree_has_unsafe_entry(root: Path) -> bool:
    pending = [root]
    while pending:
        directory = pending.pop()
        for name, kind in _directory_entries(directory).items():
            if kind not in {"file", "directory"}:
                return True
            if kind == "directory":
                pending.append(directory / name)
    return False


def _has_symlink_component(path: Path) -> bool:
    current = path.absolute()
    while True:
        if _path_kind(current) == "symlink":
            return True
        if current == current.parent:
            return False
        current = current.parent


def _directory_entries(path: Path) -> dict[str, str]:
    for attempt in range(2):
        entries: dict[str, str] = {}
        try:
            with os.scandir(path) as iterator:
                for entry in iterator:
                    mode = entry.stat(follow_symlinks=False).st_mode
                    entries[entry.name] = _mode_kind(mode)
        except FileNotFoundError:
            if attempt:
                raise
        else:
            return entries
    raise RuntimeError("directory snapshot retry exhausted")


def _path_kind(path: Path) -> str:
    try:
        return _mode_kind(path.lstat().st_mode)
    except FileNotFoundError:
        return "absent"


def _mode_kind(mode: int) -> str:
    if stat.S_ISREG(mode):
        return "file"
    if stat.S_ISDIR(mode):
        return "directory"
    if stat.S_ISLNK(mode):
        return "symlink"
    return "unsupported"


def _blocked(
    category: str, *, state: str = "blocked", run_id: str | None = None
) -> OKFStartupDecision:
    return OKFStartupDecision(False, "blocked", state, category, run_id)


def execute_okf_cutover(
    source_root: Path,
    runtime_root: Path,
    run_id: str,
    *,
    is_quiescent: Callable[[], bool],
    fault_inject: FaultInjector | None = None,
) -> CutoverState:
    """Atomically coordinate the three direct-live OKF assets while offline."""

    with okf_writer_lock(source_root, exclusive=True, allow_create=False):
        return _execute_okf_cutover_locked(
            source_root,
            runtime_root,
            run_id,
            is_quiescent=is_quiescent,
            fault_inject=fault_inject,
        )


def _execute_okf_cutover_locked(
    source_root: Path,
    runtime_root: Path,
    run_id: str,
    *,
    is_quiescent: Callable[[], bool],
    fault_inject: FaultInjector | None,
) -> CutoverState:

    context = _context(source_root, runtime_root, run_id)
    _validate_prepared(context)
    _reject_symlink(context.workspace / "cutover.lock", "cutover lock")
    with file_lock(context.workspace / "cutover.lock"):
        if not is_quiescent():
            raise RuntimeError("OKF cutover requires a quiescent runtime")
        old_activity = _validate_prepared(context)
        _checkpoint(fault_inject, "before-start-journal")
        completed: list[str] = []
        _write_journal(
            context,
            state="in-progress",
            mode="cutover",
            phase="ready",
            step=None,
            completed=completed,
            old_activity=old_activity,
        )
        _checkpoint(fault_inject, "after-start-journal")
        _write_sentinel(context, "in-progress")
        _checkpoint(fault_inject, "after-start-sentinel")
        _mkdir_durable(context.backup)
        _checkpoint(fault_inject, "after-backup-directory")

        for asset in _assets(context, old_activity):
            _move(
                context,
                asset.name,
                asset.live,
                asset.backup,
                mode="cutover",
                completed=completed,
                old_activity=old_activity,
                fault_inject=fault_inject,
                source_may_be_absent=(
                    asset.name == "backup-activity" and old_activity is None
                ),
            )
        for asset in _assets(context, old_activity):
            _move(
                context,
                f"publish-{asset.name.removeprefix('backup-')}",
                asset.staged,
                asset.live,
                mode="cutover",
                completed=completed,
                old_activity=old_activity,
                fault_inject=fault_inject,
                source_may_be_absent=False,
            )

        states = _asset_states(context, old_activity)
        if any(state != "new" for state in states.values()):
            raise RuntimeError("OKF cutover did not publish every coordinated asset")
        _validate_static_source(context)
        _checkpoint(fault_inject, "before-terminal-journal")
        _finish(
            context,
            "committed-needs-rebuild",
            old_activity,
            completed,
            fault_inject=fault_inject,
        )
        return "committed-needs-rebuild"


def recover_okf_cutover(
    source_root: Path,
    runtime_root: Path,
    run_id: str,
    *,
    is_quiescent: Callable[[], bool],
) -> CutoverState:
    """Resolve an interrupted cutover to all-new or all-old from disk truth."""

    with okf_writer_lock(source_root, exclusive=True, allow_create=False):
        if _abort_cleanup_pending(runtime_root, run_id):
            return _resume_abort_cleanup(
                source_root,
                runtime_root,
                run_id,
                is_quiescent=is_quiescent,
                fault_inject=None,
            )
        return _recover_okf_cutover_locked(
            source_root, runtime_root, run_id, is_quiescent=is_quiescent
        )


def _recover_okf_cutover_locked(
    source_root: Path,
    runtime_root: Path,
    run_id: str,
    *,
    is_quiescent: Callable[[], bool],
) -> CutoverState:

    context = _context(source_root, runtime_root, run_id)
    preflight_journal = _read_canonical_object(
        context.journal, "migration journal"
    )
    _require_gate_identity(preflight_journal, context, JOURNAL_SCHEMA)
    if preflight_journal.get("state") == "prepared":
        _validate_prepared(context)
    _reject_symlink(context.workspace / "cutover.lock", "cutover lock")
    with file_lock(context.workspace / "cutover.lock"):
        if not is_quiescent():
            raise RuntimeError("OKF recovery requires a quiescent runtime")
        journal = _read_canonical_object(context.journal, "migration journal")
        _require_gate_identity(journal, context, JOURNAL_SCHEMA)
        state = journal.get("state")
        mode = journal.get("mode")
        if state == "abort-in-progress":
            return _abort_okf_cutover_locked(
                context,
                journal,
                fault_inject=None,
            )
        if state == "rebuild-in-progress":
            old_activity = _old_activity(journal, context, prepared=False)
            suffix = _activity_suffix_from_journal(journal)
            if any(
                value != "new"
                for value in _asset_states(
                    context,
                    old_activity,
                    allow_activity_suffix=True,
                    expected_activity_suffix=suffix,
                ).values()
            ):
                raise ValueError("rebuild assets do not match the new layout")
            _require_rebuild_sentinel_states(
                context, {"committed-needs-rebuild", "rebuild-in-progress"}
            )
            _write_sentinel(context, "rebuild-in-progress")
            return "rebuild-in-progress"
        if state == "sealed-rebuild":
            old_activity = _old_activity(journal, context, prepared=False)
            suffix = _activity_suffix_from_journal(journal)
            if any(
                value != "new"
                for value in _asset_states(
                    context,
                    old_activity,
                    allow_activity_suffix=True,
                    expected_activity_suffix=suffix,
                ).values()
            ):
                raise ValueError("sealed rebuild assets do not match the manifest")
            _derived_journal_extras(
                context,
                journal,
                rollback_outcome="pending",
                recutover_outcome="pending",
            )
            _require_rebuild_sentinel_states(
                context, {"rebuild-in-progress", "sealed-rebuild"}
            )
            _write_sentinel(context, "sealed-rebuild")
            return "sealed-rebuild"
        if state == "in-progress" and mode == "rollback-drill":
            return _rollback_drill_locked(context, journal, fault_inject=None)
        if state == "rollback-drill-complete":
            return _rollback_drill_locked(context, journal, fault_inject=None)
        if state == "recutover-in-progress":
            return _recutover_locked(context, journal, fault_inject=None)
        if state == "finalized-v2":
            _require_finalized_journal(context, journal)
            _remove_sentinel(context)
            return "finalized-v2"
        if state == "committed":
            raise ValueError("legacy committed migration state is unsupported")
        if state not in {
            "prepared",
            "in-progress",
            "committed-needs-rebuild",
            "rollback-complete",
        }:
            raise ValueError("migration journal has an unknown state")
        _validate_static_source(context)
        old_activity = _old_activity(journal, context, prepared=state == "prepared")
        if state in {"prepared", "in-progress"}:
            _require_active_sentinel(context)
        elif state == "committed-needs-rebuild":
            _require_rebuild_sentinel_states(
                context, {"in-progress", "committed-needs-rebuild"}
            )
        states = _asset_states(context, old_activity)

        if state == "committed-needs-rebuild":
            if any(value != "new" for value in states.values()):
                raise ValueError("pending-rebuild assets do not match the manifest")
            _write_sentinel(context, state)
            return "committed-needs-rebuild"
        if state == "rollback-complete":
            if any(value != "old" for value in states.values()):
                raise ValueError("rolled-back migration assets do not match the manifest")
            _remove_sentinel(context)
            return "rollback-complete"
        if all(value == "new" for value in states.values()):
            _finish(
                context,
                "committed-needs-rebuild",
                old_activity,
                [],
                fault_inject=None,
            )
            return "committed-needs-rebuild"

        _write_sentinel(context, "in-progress")
        completed: list[str] = []
        _write_journal(
            context,
            state="in-progress",
            mode="rollback",
            phase="ready",
            step=None,
            completed=completed,
            old_activity=old_activity,
        )
        for asset in _assets(context, old_activity):
            if states[asset.name] == "new":
                _move(
                    context,
                    f"rollback-new-{asset.name.removeprefix('backup-')}",
                    asset.live,
                    asset.staged,
                    mode="rollback",
                    completed=completed,
                    old_activity=old_activity,
                    fault_inject=None,
                    source_may_be_absent=False,
                )
        states = _asset_states(context, old_activity)
        for asset in _assets(context, old_activity):
            if states[asset.name] == "missing-live":
                _move(
                    context,
                    f"restore-old-{asset.name.removeprefix('backup-')}",
                    asset.backup,
                    asset.live,
                    mode="rollback",
                    completed=completed,
                    old_activity=old_activity,
                    fault_inject=None,
                    source_may_be_absent=False,
                )
        if any(
            value != "old" for value in _asset_states(context, old_activity).values()
        ):
            raise RuntimeError("OKF recovery did not restore every old asset")
        _finish(
            context,
            "rollback-complete",
            old_activity,
            completed,
            fault_inject=None,
        )
        return "rollback-complete"


def abort_okf_cutover(
    source_root: Path,
    runtime_root: Path,
    run_id: str,
    *,
    is_quiescent: Callable[[], bool],
    fault_inject: FaultInjector | None = None,
) -> CutoverState:
    """Abandon an unsealed rebuild and durably restore the legacy layout."""

    with okf_writer_lock(source_root, exclusive=True, allow_create=False):
        migrations_kind = _path_kind(runtime_root / "migrations")
        if _abort_cleanup_pending(runtime_root, run_id):
            return _resume_abort_cleanup(
                source_root,
                runtime_root,
                run_id,
                is_quiescent=is_quiescent,
                fault_inject=fault_inject,
            )
        if migrations_kind == "absent":
            decision = discover_okf_startup(source_root, runtime_root)
            if decision.allowed and decision.layout == "legacy":
                return "aborted"
            raise ValueError("migration abort has no resumable workspace")

        context = _context(source_root, runtime_root, run_id)
        preflight_journal = _read_canonical_object(
            context.journal, "migration journal"
        )
        _require_gate_identity(preflight_journal, context, JOURNAL_SCHEMA)
        if preflight_journal.get("state") not in {
            "committed-needs-rebuild",
            "rebuild-in-progress",
            "abort-in-progress",
        }:
            raise ValueError("migration is not abortable")
        if _path_kind(context.workspace / REBUILD_PROOF_FILENAME) != "absent":
            raise ValueError("migration rebuild proof is already published")
        if _path_kind(context.workspace / "derived-rebuild") != "absent":
            raise ValueError("migration derived rebuild already started")
        lock_path = context.workspace / "cutover.lock"
        _reject_symlink(lock_path, "cutover lock")
        with file_lock(lock_path):
            if not is_quiescent():
                raise RuntimeError("OKF abort requires a quiescent runtime")
            journal = _read_canonical_object(context.journal, "migration journal")
            _require_gate_identity(journal, context, JOURNAL_SCHEMA)
            return _abort_okf_cutover_locked(
                context,
                journal,
                fault_inject=fault_inject,
            )


def _abort_okf_cutover_locked(
    context: _Context,
    journal: Mapping[str, object],
    *,
    fault_inject: FaultInjector | None,
) -> CutoverState:
    state = journal.get("state")
    if state not in {
        "committed-needs-rebuild",
        "rebuild-in-progress",
        "abort-in-progress",
    }:
        raise ValueError("migration is not abortable")
    if _path_kind(context.workspace / REBUILD_PROOF_FILENAME) != "absent":
        raise ValueError("migration rebuild proof is already published")
    if _path_kind(context.workspace / "derived-rebuild") != "absent":
        raise ValueError("migration derived rebuild already started")

    old_activity = _old_activity(journal, context, prepared=False)
    if state == "abort-in-progress":
        origin = journal.get("abort_from_state")
        if origin not in {"committed-needs-rebuild", "rebuild-in-progress"}:
            raise ValueError("migration abort origin is invalid")
        suffix = (
            _activity_suffix_from_journal(journal)
            if origin == "rebuild-in-progress"
            else None
        )
        allowed_sentinels = {"abort-in-progress", "committed-needs-rebuild"}
        allowed_sentinels.add(
            "rebuild-in-progress" if origin == "rebuild-in-progress" else "in-progress"
        )
        _require_rebuild_sentinel_states(context, allowed_sentinels)
    else:
        origin = state
        suffix = (
            _activity_suffix_from_journal(journal)
            if state == "rebuild-in-progress"
            else None
        )
        _require_rebuild_sentinel_states(
            context,
            (
                {"committed-needs-rebuild", "rebuild-in-progress"}
                if state == "rebuild-in-progress"
                else {"in-progress", "committed-needs-rebuild"}
            ),
        )
        states = _asset_states(
            context,
            old_activity,
            allow_activity_suffix=suffix is not None,
            expected_activity_suffix=suffix,
        )
        if any(value != "new" for value in states.values()):
            raise ValueError("abortable migration assets do not match the new layout")

    extras: dict[str, object] = {"abort_from_state": origin}
    if suffix is not None:
        extras["activity_suffix"] = {
            "length": suffix[0],
            "sha256": suffix[1],
        }
    completed: list[str] = []

    def abort_move(name: str, source: Path, destination: Path) -> None:
        _move(
            context,
            name,
            source,
            destination,
            mode="abort",
            completed=completed,
            old_activity=old_activity,
            fault_inject=fault_inject,
            source_may_be_absent=False,
            journal_extras=extras,
            journal_state="abort-in-progress",
        )

    _checkpoint(fault_inject, "before-abort-journal")
    _write_journal(
        context,
        state="abort-in-progress",
        mode="abort",
        phase="ready",
        step=None,
        completed=[],
        old_activity=old_activity,
        extras=extras,
    )
    _checkpoint(fault_inject, "after-abort-journal-before-sentinel")
    _write_sentinel(context, "abort-in-progress")
    _checkpoint(fault_inject, "after-abort-journal")

    states = _asset_states(
        context,
        old_activity,
        allow_activity_suffix=suffix is not None,
        expected_activity_suffix=suffix,
    )
    for asset in _assets(context, old_activity):
        if states[asset.name] == "new":
            abort_move(
                f"abort-stage-{asset.name.removeprefix('backup-')}",
                asset.live,
                asset.staged,
            )
    states = _asset_states(
        context,
        old_activity,
        allow_activity_suffix=suffix is not None,
        expected_activity_suffix=suffix,
    )
    for asset in _assets(context, old_activity):
        if states[asset.name] == "missing-live":
            abort_move(
                f"abort-restore-{asset.name.removeprefix('backup-')}",
                asset.backup,
                asset.live,
            )

    if any(
        value != "old"
        for value in _asset_states(
            context,
            old_activity,
            allow_activity_suffix=suffix is not None,
            expected_activity_suffix=suffix,
        ).values()
    ):
        raise RuntimeError("OKF abort did not restore every old asset")
    _validate_static_source(context)

    _write_journal(
        context,
        state="abort-in-progress",
        mode="abort",
        phase="cleanup",
        step=None,
        completed=completed,
        old_activity=old_activity,
        extras=extras,
    )
    migrations = context.runtime / "migrations"
    if _directory_entries(migrations) != {context.workspace.name: "directory"}:
        raise ValueError("migration abort workspace set changed")
    marker_path = _abort_marker(context.runtime, context.workspace.name)
    if _path_kind(marker_path) == "absent":
        _write_object(
            marker_path,
            {
                "schema": JOURNAL_SCHEMA,
                "version": SCHEMA_VERSION,
                "run_id": context.workspace.name,
                "state": "abort-cleanup",
                "manifest_sha256": context.manifest_sha256,
            },
        )
    elif _path_kind(marker_path) != "file":
        raise ValueError("migration abort marker is unsafe")
    _require_abort_cleanup_marker(
        marker_path,
        run_id=context.workspace.name,
        manifest_sha256=context.manifest_sha256,
    )
    tombstone = _abort_tombstone(context.runtime, context.workspace.name)
    if _path_kind(tombstone) != "absent":
        raise ValueError("migration abort tombstone already exists")
    _checkpoint(fault_inject, "before-abort-workspace-rename")
    os.rename(migrations, tombstone)
    fsync_directory(context.runtime)
    _checkpoint(fault_inject, "after-abort-workspace-rename")
    _checkpoint(fault_inject, "before-abort-workspace-remove")
    _remove_tree_exact(tombstone)
    _checkpoint(fault_inject, "after-abort-workspace-remove")
    _checkpoint(fault_inject, "before-abort-marker-remove")
    _remove_file_exact(marker_path, required=True)
    _checkpoint(fault_inject, "after-abort-marker-remove")
    decision = discover_okf_startup(context.source, context.runtime)
    if not decision.allowed or decision.layout != "legacy":
        raise RuntimeError("OKF abort did not restore legacy startup")
    return "aborted"


def _abort_tombstone(runtime_root: Path, run_id: str) -> Path:
    if not _RUN_ID_RE.fullmatch(run_id):
        raise ValueError("run_id must be a safe single path component")
    return runtime_root / f".okf-abort-{run_id}"


def _abort_marker(runtime_root: Path, run_id: str) -> Path:
    if not _RUN_ID_RE.fullmatch(run_id):
        raise ValueError("run_id must be a safe single path component")
    return runtime_root / f".okf-abort-{run_id}.json"


def _abort_cleanup_pending(runtime_root: Path, run_id: str) -> bool:
    return _path_kind(_abort_tombstone(runtime_root, run_id)) != "absent" or (
        _path_kind(_abort_marker(runtime_root, run_id)) != "absent"
        and _path_kind(runtime_root / "migrations") == "absent"
    )


def _resume_abort_cleanup(
    source_root: Path,
    runtime_root: Path,
    run_id: str,
    *,
    is_quiescent: Callable[[], bool],
    fault_inject: FaultInjector | None,
) -> CutoverState:
    source = _safe_root(source_root, "source root")
    runtime = _safe_root(runtime_root, "runtime root")
    tombstone = _abort_tombstone(runtime, run_id)
    marker_path = _abort_marker(runtime, run_id)
    if _path_kind(runtime / "migrations") != "absent":
        raise ValueError("migration abort cleanup overlaps a new workspace")
    marker = _read_canonical_object(marker_path, "migration abort marker")
    manifest_sha256 = _sha(marker.get("manifest_sha256"))
    _require_abort_cleanup_marker(
        marker_path,
        run_id=run_id,
        manifest_sha256=manifest_sha256,
    )
    tombstone_kind = _path_kind(tombstone)
    if tombstone_kind not in {"absent", "directory"} or (
        tombstone_kind == "directory" and _tree_has_unsafe_entry(tombstone)
    ):
        raise ValueError("migration abort tombstone is unsafe")
    if tombstone_kind == "directory":
        entries = _directory_entries(tombstone)
        if entries and entries != {run_id: "directory"}:
            raise ValueError("migration abort tombstone identity changed")
    lock_path = tombstone / run_id / "cutover.lock"

    def remove() -> None:
        if not is_quiescent():
            raise RuntimeError("OKF abort requires a quiescent runtime")
        _checkpoint(fault_inject, "before-abort-workspace-remove")
        _remove_tree_exact(tombstone)
        _checkpoint(fault_inject, "after-abort-workspace-remove")
        _checkpoint(fault_inject, "before-abort-marker-remove")
        _remove_file_exact(marker_path, required=True)
        _checkpoint(fault_inject, "after-abort-marker-remove")

    if _path_kind(lock_path) == "file":
        with file_lock(lock_path):
            remove()
    elif _path_kind(lock_path) == "absent":
        remove()
    else:
        raise ValueError("migration abort lock is unsafe")
    decision = discover_okf_startup(source, runtime)
    if not decision.allowed or decision.layout != "legacy":
        raise RuntimeError("OKF abort did not restore legacy startup")
    return "aborted"


def _require_abort_cleanup_marker(
    path: Path,
    *,
    run_id: str,
    manifest_sha256: str,
) -> None:
    marker = _read_canonical_object(path, "migration abort marker")
    if marker != {
        "schema": JOURNAL_SCHEMA,
        "version": SCHEMA_VERSION,
        "run_id": run_id,
        "state": "abort-cleanup",
        "manifest_sha256": manifest_sha256,
    }:
        raise ValueError("migration abort marker identity changed")


def rollback_okf_rebuild(
    source_root: Path,
    runtime_root: Path,
    run_id: str,
    *,
    is_quiescent: Callable[[], bool],
    fault_inject: FaultInjector | None = None,
) -> CutoverState:
    """Run or resume the mandatory all-old rollback drill after rebuild seal."""

    with okf_writer_lock(source_root, exclusive=True, allow_create=False):
        context = _context(source_root, runtime_root, run_id)
        lock_path = context.workspace / "cutover.lock"
        _reject_symlink(lock_path, "cutover lock")
        with file_lock(lock_path):
            if not is_quiescent():
                raise RuntimeError("OKF rollback drill requires a quiescent runtime")
            journal = _read_canonical_object(context.journal, "migration journal")
            _require_gate_identity(journal, context, JOURNAL_SCHEMA)
            if journal.get("state") not in {
                "sealed-rebuild",
                "in-progress",
                "rollback-drill-complete",
            }:
                raise ValueError("migration is not ready for the rollback drill")
            if journal.get("state") == "in-progress" and journal.get("mode") != (
                "rollback-drill"
            ):
                raise ValueError("migration is in a different recovery mode")
            return _rollback_drill_locked(
                context,
                journal,
                fault_inject=fault_inject,
            )


def _rollback_drill_locked(
    context: _Context,
    journal: Mapping[str, object],
    *,
    fault_inject: FaultInjector | None,
) -> CutoverState:
    _validate_static_source(context)
    old_activity = _old_activity(journal, context, prepared=False)
    suffix = _activity_suffix_from_journal(journal)
    state = journal.get("state")
    _require_rebuild_sentinel_states(
        context, {"rebuild-in-progress", "sealed-rebuild"}
    )
    extras = _derived_journal_extras(
        context,
        journal,
        rollback_outcome=("complete" if state == "rollback-drill-complete" else "in-progress"),
        recutover_outcome="pending",
    )
    states = _asset_states(
        context,
        old_activity,
        allow_activity_suffix=True,
        expected_activity_suffix=suffix,
    )
    if state == "rollback-drill-complete":
        if any(value != "old" for value in states.values()):
            raise ValueError("rollback drill assets do not match the old layout")
        _write_sentinel(context, "sealed-rebuild")
        return "rollback-drill-complete"
    if state == "sealed-rebuild":
        if any(value != "new" for value in states.values()):
            raise ValueError("sealed rebuild assets do not match the new layout")
        _checkpoint(fault_inject, "before-rollback-drill-journal")
        _write_journal(
            context,
            state="in-progress",
            mode="rollback-drill",
            phase="ready",
            step=None,
            completed=[],
            old_activity=old_activity,
            extras=extras,
        )
        _checkpoint(fault_inject, "after-rollback-drill-journal")

    completed: list[str] = []
    for asset in _assets(context, old_activity):
        if states[asset.name] == "new":
            name = f"drill-stage-{asset.name.removeprefix('backup-')}"
            _move(
                context,
                name,
                asset.live,
                asset.staged,
                mode="rollback-drill",
                completed=completed,
                old_activity=old_activity,
                fault_inject=fault_inject,
                source_may_be_absent=False,
                journal_extras=extras,
            )
    states = _asset_states(
        context,
        old_activity,
        allow_activity_suffix=True,
        expected_activity_suffix=suffix,
    )
    for asset in _assets(context, old_activity):
        if states[asset.name] == "missing-live":
            name = f"drill-restore-{asset.name.removeprefix('backup-')}"
            _move(
                context,
                name,
                asset.backup,
                asset.live,
                mode="rollback-drill",
                completed=completed,
                old_activity=old_activity,
                fault_inject=fault_inject,
                source_may_be_absent=(
                    asset.name == "backup-activity" and old_activity is None
                ),
                journal_extras=extras,
            )
    if any(
        value != "old"
        for value in _asset_states(
            context,
            old_activity,
            allow_activity_suffix=True,
            expected_activity_suffix=suffix,
        ).values()
    ):
        raise RuntimeError("rollback drill did not restore every old asset")
    terminal_extras = {
        **extras,
        "rollback_outcome": "complete",
    }
    _write_journal(
        context,
        state="rollback-drill-complete",
        mode="rollback-drill",
        phase="complete",
        step=None,
        completed=completed,
        old_activity=old_activity,
        extras=terminal_extras,
    )
    _checkpoint(fault_inject, "after-rollback-drill-terminal-journal")
    _write_sentinel(context, "sealed-rebuild")
    return "rollback-drill-complete"


def recutover_okf_rebuild(
    source_root: Path,
    runtime_root: Path,
    run_id: str,
    *,
    is_quiescent: Callable[[], bool],
    fault_inject: FaultInjector | None = None,
) -> CutoverState:
    """Republish the sealed new layout after the successful rollback drill."""

    with okf_writer_lock(source_root, exclusive=True, allow_create=False):
        context = _context(source_root, runtime_root, run_id)
        lock_path = context.workspace / "cutover.lock"
        _reject_symlink(lock_path, "cutover lock")
        with file_lock(lock_path):
            if not is_quiescent():
                raise RuntimeError("OKF recutover requires a quiescent runtime")
            journal = _read_canonical_object(context.journal, "migration journal")
            _require_gate_identity(journal, context, JOURNAL_SCHEMA)
            if journal.get("state") not in {
                "rollback-drill-complete",
                "recutover-in-progress",
                "finalized-v2",
            }:
                raise ValueError("migration is not ready for recutover")
            return _recutover_locked(context, journal, fault_inject=fault_inject)


def _recutover_locked(
    context: _Context,
    journal: Mapping[str, object],
    *,
    fault_inject: FaultInjector | None,
) -> CutoverState:
    _validate_static_source(context)
    old_activity = _old_activity(journal, context, prepared=False)
    suffix = _activity_suffix_from_journal(journal)
    state = journal.get("state")
    extras = _derived_journal_extras(
        context,
        journal,
        rollback_outcome="complete",
        recutover_outcome=("complete" if state == "finalized-v2" else "in-progress"),
    )
    states = _asset_states(
        context,
        old_activity,
        allow_activity_suffix=True,
        expected_activity_suffix=suffix,
    )
    if state == "finalized-v2":
        _require_finalized_journal(context, journal)
        _remove_sentinel(context)
        return "finalized-v2"
    _require_rebuild_sentinel_states(
        context, {"rebuild-in-progress", "sealed-rebuild"}
    )
    if state == "rollback-drill-complete":
        if any(value != "old" for value in states.values()):
            raise ValueError("rollback drill assets do not match the old layout")
        _checkpoint(fault_inject, "before-recutover-journal")
        _write_journal(
            context,
            state="recutover-in-progress",
            mode="recutover",
            phase="ready",
            step=None,
            completed=[],
            old_activity=old_activity,
            extras=extras,
        )
        _checkpoint(fault_inject, "after-recutover-journal")

    completed: list[str] = []
    for asset in _assets(context, old_activity):
        if states[asset.name] == "old":
            name = f"recutover-backup-{asset.name.removeprefix('backup-')}"
            _move(
                context,
                name,
                asset.live,
                asset.backup,
                mode="recutover",
                completed=completed,
                old_activity=old_activity,
                fault_inject=fault_inject,
                source_may_be_absent=(
                    asset.name == "backup-activity" and old_activity is None
                ),
                journal_extras=extras,
                journal_state="recutover-in-progress",
            )
    states = _asset_states(
        context,
        old_activity,
        allow_activity_suffix=True,
        expected_activity_suffix=suffix,
    )
    for asset in _assets(context, old_activity):
        if states[asset.name] == "missing-live" or (
            asset.old is None and states[asset.name] == "old"
        ):
            name = f"recutover-publish-{asset.name.removeprefix('backup-')}"
            _move(
                context,
                name,
                asset.staged,
                asset.live,
                mode="recutover",
                completed=completed,
                old_activity=old_activity,
                fault_inject=fault_inject,
                source_may_be_absent=False,
                journal_extras=extras,
                journal_state="recutover-in-progress",
            )
    if any(
        value != "new"
        for value in _asset_states(
            context,
            old_activity,
            allow_activity_suffix=True,
            expected_activity_suffix=suffix,
        ).values()
    ):
        raise RuntimeError("recutover did not publish every sealed new asset")
    final_extras = {
        **extras,
        "recutover_outcome": "complete",
    }
    _write_journal(
        context,
        state="finalized-v2",
        mode="recutover",
        phase="complete",
        step=None,
        completed=completed,
        old_activity=old_activity,
        extras=final_extras,
    )
    _checkpoint(fault_inject, "after-finalized-journal")
    _remove_sentinel(context)
    _checkpoint(fault_inject, "after-finalized-sentinel-remove")
    return "finalized-v2"


def cleanup_okf_cutover(
    source_root: Path,
    runtime_root: Path,
    run_id: str,
    *,
    is_quiescent: Callable[[], bool],
    fault_inject: FaultInjector | None = None,
) -> CutoverState:
    """Replace one exact terminal proof workspace with a compact receipt."""

    with okf_writer_lock(source_root, exclusive=True, allow_create=False):
        return _cleanup_okf_cutover_locked(
            source_root,
            runtime_root,
            run_id,
            is_quiescent=is_quiescent,
            fault_inject=fault_inject,
        )


def _cleanup_okf_cutover_locked(
    source_root: Path,
    runtime_root: Path,
    run_id: str,
    *,
    is_quiescent: Callable[[], bool],
    fault_inject: FaultInjector | None,
) -> CutoverState:

    source, runtime, workspace = _workspace_roots(
        source_root, runtime_root, run_id
    )
    if _receipt_only(workspace):
        state, _manifest_sha256 = _read_receipt(
            workspace / RECEIPT_FILENAME, run_id
        )
        _require_receipt_layout(source, runtime, state)
        return state
    _require_cleanup_workspace(workspace)
    lock_path = workspace / "cutover.lock"
    _reject_symlink(lock_path, "cutover lock")
    with file_lock(lock_path):
        _require_cleanup_workspace(workspace)
        receipt_path = workspace / RECEIPT_FILENAME
        if not is_quiescent():
            raise RuntimeError("OKF cleanup requires a quiescent runtime")

        if _path_kind(receipt_path) == "file":
            state, manifest_sha256 = _read_receipt(receipt_path, run_id)
            journal = _read_canonical_object(
                workspace / "journal.json", "migration journal"
            )
            _require_cleanup_journal(journal, run_id, state, manifest_sha256)
            _require_receipt_layout(source, runtime, state)
        else:
            context, journal, state = _require_terminal_proof(
                source,
                runtime,
                run_id,
                allow_cleanup=True,
            )
            if journal.get("cleanup_in_progress") is not True:
                journal = {**journal, "cleanup_in_progress": True}
                _write_object(context.journal, journal)
                _checkpoint(fault_inject, "after-cleanup-journal")
            context, journal, state = _require_terminal_proof(
                source,
                runtime,
                run_id,
                allow_cleanup=True,
            )
            manifest_sha256 = context.manifest_sha256
            _checkpoint(fault_inject, "before-receipt-write")
            _write_object(
                receipt_path,
                _receipt_payload(run_id, state, manifest_sha256),
            )
            _checkpoint(fault_inject, "after-receipt-write")

        _require_cleanup_journal(journal, run_id, state, manifest_sha256)
        for name, path, directory in (
            ("staging", workspace / "staging", True),
            ("rollback-backup", workspace / "rollback-backup", True),
            ("dry-run-manifest", workspace / "dry-run-manifest.json", False),
            ("cutover-lock", lock_path, False),
        ):
            _checkpoint(fault_inject, f"before-remove-{name}")
            if directory:
                _remove_tree_exact(path)
            else:
                _remove_file_exact(path)
            _checkpoint(fault_inject, f"after-remove-{name}")

        _checkpoint(fault_inject, "before-journal-remove")
        _remove_file_exact(workspace / "journal.json", required=True)
        _checkpoint(fault_inject, "after-journal-remove")
        if not _receipt_only(workspace):
            raise RuntimeError("legacy cleanup did not leave one compact receipt")
        return state


def finalize_okf_rebuild(
    source_root: Path,
    runtime_root: Path,
    run_id: str,
    *,
    is_quiescent: Callable[[], bool],
    fault_inject: FaultInjector | None = None,
) -> CutoverState:
    """Publish the v2 final receipt, then remove only validated drill artifacts."""

    with okf_writer_lock(source_root, exclusive=True, allow_create=False):
        source, runtime, workspace = _workspace_roots(
            source_root, runtime_root, run_id
        )
        if _receipt_only(workspace):
            receipt = _read_any_receipt(workspace / RECEIPT_FILENAME, run_id)
            if receipt.get("schema") != FINAL_RECEIPT_SCHEMA:
                raise ValueError("migration receipt is not finalized v2")
            _require_final_receipt_layout(source, runtime, receipt)
            return "finalized-v2"
        _require_final_cleanup_workspace(workspace)
        lock_path = workspace / "cutover.lock"
        _reject_symlink(lock_path, "cutover lock")
        with file_lock(lock_path):
            if not is_quiescent():
                raise RuntimeError("OKF finalization requires a quiescent runtime")
            _require_final_cleanup_workspace(workspace)
            receipt_path = workspace / RECEIPT_FILENAME
            manifest_path = workspace / "dry-run-manifest.json"
            context: _Context | None = None
            if _path_kind(manifest_path) == "file":
                context = _context(source, runtime, run_id)
                journal = _read_canonical_object(
                    context.journal, "migration journal"
                )
                if _path_kind(receipt_path) == "absent":
                    _require_finalized_journal(context, journal)
                    expected_receipt = _final_receipt_payload(context, journal)
                    _checkpoint(fault_inject, "before-final-receipt-write")
                    _write_object(receipt_path, expected_receipt)
                    _checkpoint(fault_inject, "after-final-receipt-write")
                receipt = _read_any_receipt(receipt_path, run_id)
                _require_final_cleanup_journal(journal, receipt, run_id)
                if receipt.get("manifest_sha256") != context.manifest_sha256:
                    raise ValueError("final receipt manifest identity changed")
                _require_partial_final_cleanup_layout(context, receipt)
            else:
                if _path_kind(manifest_path) != "absent":
                    raise ValueError("final cleanup manifest is unsafe")
                receipt = _read_any_receipt(receipt_path, run_id)
                journal = _read_canonical_object(
                    workspace / "journal.json", "migration journal"
                )
                _require_final_cleanup_journal(journal, receipt, run_id)
                _require_final_receipt_layout(source, runtime, receipt)

            removal: tuple[tuple[str, Path, bool], ...] = (
                ("legacy-index", source / "index.md", False),
                ("legacy-log", source / "log.md", False),
                ("legacy-schema", source / "schema.md", False),
                ("staging", workspace / "staging", True),
                ("rollback-backup", workspace / "rollback-backup", True),
                ("derived-rebuild", workspace / "derived-rebuild", True),
                ("rebuild-proof", workspace / REBUILD_PROOF_FILENAME, False),
                ("dry-run-manifest", manifest_path, False),
                ("restart-refusal", workspace / RESTART_REFUSAL_FILENAME, False),
                ("cutover-lock", lock_path, False),
            )
            for name, path, directory in removal:
                _checkpoint(fault_inject, f"before-final-remove-{name}")
                if directory:
                    _remove_tree_exact(path)
                else:
                    _remove_file_exact(path)
                _checkpoint(fault_inject, f"after-final-remove-{name}")
            _require_final_receipt_layout(source, runtime, receipt)
            _checkpoint(fault_inject, "before-final-remove-journal")
            _remove_file_exact(workspace / "journal.json", required=True)
            _checkpoint(fault_inject, "after-final-remove-journal")
            if not _receipt_only(workspace):
                raise RuntimeError("final cleanup did not leave one compact receipt")
            return "finalized-v2"


def _require_final_cleanup_journal(
    journal: Mapping[str, object],
    receipt: Mapping[str, object],
    run_id: str,
) -> None:
    rebuild = receipt.get("rebuild_proof")
    if not isinstance(rebuild, Mapping):
        raise ValueError("final receipt rebuild proof is invalid")
    journal_suffix = _activity_suffix_from_journal(journal)
    receipt_suffix = _activity_suffix_from_journal(receipt)
    if (
        journal.get("schema") != JOURNAL_SCHEMA
        or journal.get("version") != SCHEMA_VERSION
        or journal.get("run_id") != run_id
        or journal.get("state") != "finalized-v2"
        or journal.get("manifest_sha256") != receipt.get("manifest_sha256")
        or journal.get("derived_generation") != rebuild.get("derived_generation")
        or journal.get("rebuild_proof_sha256") != rebuild.get("sha256")
        or journal.get("rollback_outcome") != "complete"
        or journal.get("recutover_outcome") != "complete"
        or receipt_suffix[0] < journal_suffix[0]
        or (receipt_suffix[0] == journal_suffix[0] and receipt_suffix != journal_suffix)
    ):
        raise ValueError("final cleanup journal is invalid")


def _require_final_cleanup_workspace(workspace: Path) -> None:
    if _tree_has_unsafe_entry(workspace):
        raise ValueError("final cleanup workspace is unsafe")
    entries = _directory_entries(workspace)
    if any(_WORKSPACE_ENTRIES.get(name) != kind for name, kind in entries.items()):
        raise ValueError("final cleanup workspace has an unknown artifact")
    if entries.get("journal.json") != "file":
        raise ValueError("final cleanup journal is missing")
    for name in ("staging", "rollback-backup"):
        _require_known_cleanup_tree(workspace / name)
    derived = workspace / "derived-rebuild"
    if _path_kind(derived) not in {"absent", "directory"}:
        raise ValueError("final cleanup derived tree is unsafe")
    if _path_kind(derived) == "directory" and _tree_has_unsafe_entry(derived):
        raise ValueError("final cleanup derived tree is unsafe")


def okf_startup_allowed(source_root: Path, runtime_root: Path, run_id: str) -> bool:
    """Fail closed unless a terminal journal or cleanup receipt proves startup."""

    decision = discover_okf_startup(source_root, runtime_root)
    return decision.allowed and decision.run_id == run_id


@contextmanager
def okf_rebuild_session(
    source_root: Path,
    runtime_root: Path,
    run_id: str,
    *,
    is_quiescent: Callable[[], bool],
) -> Iterator[OKFRebuildSession]:
    """Hold the root and cutover leases for one complete offline rebuild."""

    with okf_writer_lock(source_root, exclusive=True, allow_create=False):
        context = _context(source_root, runtime_root, run_id)
        lock_path = context.workspace / "cutover.lock"
        _reject_symlink(lock_path, "cutover lock")
        with file_lock(lock_path):
            if not is_quiescent():
                raise RuntimeError("OKF rebuild requires a quiescent runtime")
            gate = _begin_okf_rebuild_locked(context)
            session = OKFRebuildSession(gate, context)
            try:
                yield session
            finally:
                session._active = False


def _begin_okf_rebuild_locked(context: _Context) -> OKFRebuildGate:
    journal = _read_canonical_object(context.journal, "migration journal")
    _require_gate_identity(journal, context, JOURNAL_SCHEMA)
    state = journal.get("state")
    if state not in {
        "committed-needs-rebuild",
        "rebuild-in-progress",
        "sealed-rebuild",
    }:
        raise ValueError("migration is not ready for derived rebuild")
    old_activity = _old_activity(journal, context, prepared=False)
    _validate_static_source(context)
    activity_suffix = _require_new_live_layout(context, old_activity)
    _require_rebuild_sentinel_states(
        context,
        {
            "committed-needs-rebuild": {
                "in-progress",
                "committed-needs-rebuild",
            },
            "rebuild-in-progress": {
                "committed-needs-rebuild",
                "rebuild-in-progress",
            },
            "sealed-rebuild": {"rebuild-in-progress", "sealed-rebuild"},
        }[state],
    )
    if state in {"rebuild-in-progress", "sealed-rebuild"}:
        _require_activity_suffix_identity(journal, activity_suffix)
    if state == "sealed-rebuild":
        derived_generation = journal.get("derived_generation")
        rebuild_proof_sha256 = journal.get("rebuild_proof_sha256")
        if not isinstance(derived_generation, str):
            raise ValueError("sealed rebuild generation is missing")
        proof_sha256 = _sha(rebuild_proof_sha256)
        proof = _require_rebuild_proof(
            context,
            derived_generation=derived_generation,
            rebuild_proof_sha256=proof_sha256,
        )
        corpus = proof.get("corpus")
        stable_page_count = (
            corpus.get("stable_page_count")
            if isinstance(corpus, Mapping)
            else None
        )
        if not isinstance(stable_page_count, int) or isinstance(
            stable_page_count, bool
        ):
            raise ValueError("sealed rebuild corpus count is invalid")
        _write_sentinel(context, "sealed-rebuild")
        return OKFRebuildGate(
            source_root=context.source,
            workspace=context.workspace,
            manifest_sha256=context.manifest_sha256,
            activity_prefix_length=context.expected.activity_prefix[0],
            activity_prefix_sha256=context.expected.activity_prefix[1],
            activity_event_ids_sha256=context.expected.activity_event_ids_sha256,
            derived_generation=derived_generation,
            rebuild_proof_sha256=proof_sha256,
            stable_page_count=stable_page_count,
        )
    _write_rebuild_journal(
        context,
        state="rebuild-in-progress",
        old_activity=old_activity,
        derived_generation=None,
        activity_suffix=activity_suffix,
    )
    _write_sentinel(context, "rebuild-in-progress")
    return OKFRebuildGate(
        source_root=context.source,
        workspace=context.workspace,
        manifest_sha256=context.manifest_sha256,
        activity_prefix_length=context.expected.activity_prefix[0],
        activity_prefix_sha256=context.expected.activity_prefix[1],
        activity_event_ids_sha256=context.expected.activity_event_ids_sha256,
    )


def _seal_okf_rebuild_locked(
    context: _Context,
    *,
    derived_generation: str,
    rebuild_proof_sha256: str,
) -> None:
    """Bind one complete derived generation while keeping startup refused."""

    if not derived_generation or len(derived_generation) > 128:
        raise ValueError("derived generation identity is invalid")
    proof_sha256 = _sha(rebuild_proof_sha256)
    journal = _read_canonical_object(context.journal, "migration journal")
    _require_gate_identity(journal, context, JOURNAL_SCHEMA)
    if journal.get("state") not in {"rebuild-in-progress", "sealed-rebuild"}:
        raise ValueError("migration rebuild is not in progress")
    old_activity = _old_activity(journal, context, prepared=False)
    _validate_static_source(context)
    activity_suffix = _require_new_live_layout(context, old_activity)
    _require_activity_suffix_identity(journal, activity_suffix)
    _require_rebuild_sentinel_states(
        context,
        (
            {"committed-needs-rebuild", "rebuild-in-progress"}
            if journal.get("state") == "rebuild-in-progress"
            else {"rebuild-in-progress", "sealed-rebuild"}
        ),
    )
    _require_rebuild_proof(
        context,
        derived_generation=derived_generation,
        rebuild_proof_sha256=proof_sha256,
    )
    if journal.get("state") == "sealed-rebuild":
        if (
            journal.get("derived_generation") != derived_generation
            or journal.get("rebuild_proof_sha256") != proof_sha256
        ):
            raise ValueError("sealed rebuild identity changed")
    else:
        _write_rebuild_journal(
            context,
            state="sealed-rebuild",
            old_activity=old_activity,
            derived_generation=derived_generation,
            rebuild_proof_sha256=proof_sha256,
            activity_suffix=activity_suffix,
        )
    _write_sentinel(context, "sealed-rebuild")


def _publish_okf_rebuild_proof_locked(
    context: _Context,
    payload: Mapping[str, object],
) -> str:
    """Durably publish the compact, content-free component proof."""

    _validate_rebuild_proof_payload(payload)
    proof = seal_object(
        {
            "schema": REBUILD_PROOF_SCHEMA,
            "version": SCHEMA_VERSION,
            "run_id": context.workspace.name,
            "manifest_sha256": context.manifest_sha256,
            **dict(payload),
        }
    )
    path = context.workspace / REBUILD_PROOF_FILENAME
    _write_object(path, proof)
    with open_regular_nofollow(path) as handle:
        return hashlib.sha256(handle.read(1024 * 1024 + 1)).hexdigest()


def _require_terminal_proof(
    source_root: Path,
    runtime_root: Path,
    run_id: str,
    *,
    allow_cleanup: bool = False,
) -> tuple[_Context, dict[str, object], CutoverState]:
    context = _context(source_root, runtime_root, run_id)
    journal = _read_canonical_object(context.journal, "migration journal")
    _require_gate_identity(journal, context, JOURNAL_SCHEMA)
    state = journal.get("state")
    if state == "committed":
        raise ValueError("legacy committed migration state is unsupported")
    if state not in {"committed-needs-rebuild", "rollback-complete"}:
        raise ValueError("migration journal is not terminal")
    cleanup = journal.get("cleanup_in_progress", _MISSING)
    if state == "committed-needs-rebuild":
        if cleanup is not _MISSING:
            raise ValueError("pending-rebuild cleanup marker is invalid")
        if allow_cleanup:
            raise ValueError("migration rebuild is required before cleanup")
        _require_rebuild_sentinel_states(
            context, {"in-progress", "committed-needs-rebuild"}
        )
    else:
        if context.sentinel.exists() or context.sentinel.is_symlink():
            raise ValueError("restart refusal sentinel is active")
        if cleanup is not _MISSING and cleanup is not True:
            raise ValueError("migration cleanup marker is invalid")
        if cleanup is True and not allow_cleanup:
            raise ValueError("migration cleanup is in progress")
    _validate_static_source(context)
    old_activity = _old_activity(journal, context, prepared=False)
    expected = "new" if state == "committed-needs-rebuild" else "old"
    if any(
        value != expected for value in _asset_states(context, old_activity).values()
    ):
        raise ValueError("terminal migration assets do not match the manifest")
    return (
        context,
        journal,
        "committed-needs-rebuild"
        if state == "committed-needs-rebuild"
        else "rollback-complete",
    )


def _context(source_root: Path, runtime_root: Path, run_id: str) -> _Context:
    source, runtime, workspace = _workspace_roots(
        source_root, runtime_root, run_id
    )
    staging = workspace / "staging"
    manifest_path = workspace / "dry-run-manifest.json"
    _reject_symlink(manifest_path, "migration manifest")
    manifest = _read_manifest(manifest_path)
    manifest_raw = canonical_json_line_bytes_strict(manifest)
    if (
        manifest.get("schema") != MANIFEST_SCHEMA
        or manifest.get("version") != SCHEMA_VERSION
        or manifest.get("run_id") != run_id
        or manifest.get("state") != "validated"
    ):
        raise ValueError("migration manifest identity is invalid")
    return _Context(
        source=source,
        runtime=runtime,
        workspace=workspace,
        staging=staging,
        backup=workspace / "rollback-backup",
        journal=workspace / "journal.json",
        sentinel=workspace / RESTART_REFUSAL_FILENAME,
        manifest_sha256=hashlib.sha256(manifest_raw).hexdigest(),
        expected=_expected(manifest),
    )


def _workspace_roots(
    source_root: Path, runtime_root: Path, run_id: str
) -> tuple[Path, Path, Path]:
    if not _RUN_ID_RE.fullmatch(run_id):
        raise ValueError("run_id must be a safe single path component")
    if _has_symlink_component(source_root) or _has_symlink_component(runtime_root):
        raise ValueError("migration roots contain a symlink")
    source = _safe_root(source_root, "source root")
    runtime = _safe_root(runtime_root, "runtime root")
    workspace = runtime / "migrations" / run_id
    if (runtime / "migrations").is_symlink():
        raise ValueError("migration directory must not be a symlink")
    if workspace.is_symlink():
        raise ValueError("migration workspace must not be a symlink")
    workspace = workspace.resolve(strict=True)
    if not workspace.is_dir() or not workspace.is_relative_to(runtime):
        raise ValueError("migration workspace is outside runtime root")
    devices = {source.stat().st_dev, runtime.stat().st_dev, workspace.stat().st_dev}
    if len(devices) != 1:
        raise ValueError("source, runtime, and workspace must be on the same volume")
    return source, runtime, workspace


def _expected(manifest: Mapping[str, object]) -> _Expected:
    old_pages: dict[str, str] = {}
    old_system: dict[str, str] = {}
    new_pages: dict[str, str] = {}
    new_system: dict[str, str] = {}
    reserved: dict[str, str] = {}
    raw: dict[str, tuple[int, str]] = {}

    for item in _object_list(manifest, "documents"):
        path = _relative(item.get("relative_path"))
        _put(old_pages, path, _sha(item.get("source_sha256")))
        _put(new_pages, path, _sha(item.get("output_sha256")))
    for item in _object_list(manifest, "reserved_documents"):
        source_path = _relative(item.get("source_path"))
        staged_path = _relative(item.get("staged_path"))
        _put(reserved, source_path, _sha(item.get("source_sha256")))
        _put_scoped(new_pages, new_system, staged_path, item.get("output_sha256"))
    if set(reserved) != {"index.md", "log.md", "schema.md"}:
        raise ValueError("migration manifest reserved inventory is invalid")
    for item in _object_list(manifest, "system_documents"):
        path = _relative(item.get("relative_path"))
        scope = item.get("source_scope")
        if scope == "system":
            _put(old_system, path, _sha(item.get("source_sha256")))
        elif scope != "root":
            raise ValueError("migration manifest system source scope is invalid")
        output_sha256 = _sha(item.get("output_sha256"))
        if path in new_system:
            if new_system[path] != output_sha256:
                raise ValueError(f"conflicting migration manifest path: {path}")
        else:
            _put(new_system, path, output_sha256)
    for item in _object_list(manifest, "raw_files"):
        path = _relative(item.get("relative_path"))
        size = item.get("size")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise ValueError("migration manifest raw size is invalid")
        _put(raw, path, (size, _sha(item.get("sha256"))))
    activity = manifest.get("activity")
    if not isinstance(activity, dict):
        raise ValueError("migration manifest activity is invalid")
    count = activity.get("event_count")
    if not isinstance(count, int) or isinstance(count, bool) or count < 0:
        raise ValueError("migration manifest activity count is invalid")
    staged_activity = activity.get("sha256")
    segments = _object_list(activity, "segments")
    if [segment.get("name") for segment in segments] != [
        "legacy_root_log",
        "existing_runtime_activity",
        "archive_metadata",
    ]:
        raise ValueError("migration manifest activity segments are invalid")
    prepared_segment = segments[1]
    prepared_size = prepared_segment.get("length")
    if (
        not isinstance(prepared_size, int)
        or isinstance(prepared_size, bool)
        or prepared_size < 0
    ):
        raise ValueError("migration manifest existing activity size is invalid")
    prepared_activity = (prepared_size, _sha(prepared_segment.get("sha256")))
    prefix = activity.get("immutable_prefix")
    if not isinstance(prefix, dict):
        raise ValueError("migration manifest activity prefix is invalid")
    prefix_length = prefix.get("length")
    event_ids = prefix.get("event_ids")
    if (
        not isinstance(prefix_length, int)
        or isinstance(prefix_length, bool)
        or prefix_length < 0
        or not isinstance(event_ids, list)
        or not all(isinstance(value, str) and value for value in event_ids)
    ):
        raise ValueError("migration manifest activity prefix identity is invalid")
    event_ids_sha256 = hashlib.sha256(
        canonical_json_line_bytes_strict(event_ids)
    ).hexdigest()
    return _Expected(
        old_pages,
        old_system,
        new_pages,
        new_system,
        raw,
        reserved,
        prepared_activity,
        (-1, _sha(staged_activity)),
        (prefix_length, _sha(prefix.get("sha256"))),
        event_ids_sha256,
    )


def _validate_prepared(context: _Context) -> _OldActivity:
    expected_gate = {
        "schema": JOURNAL_SCHEMA,
        "version": SCHEMA_VERSION,
        "run_id": context.workspace.name,
        "state": "prepared",
        "manifest_sha256": context.manifest_sha256,
    }
    if _read_canonical_object(context.journal, "migration journal") != expected_gate:
        raise ValueError("migration journal is not the prepared gate")
    expected_gate["schema"] = SENTINEL_SCHEMA
    if _read_canonical_object(context.sentinel, "restart refusal sentinel") != expected_gate:
        raise ValueError("restart refusal sentinel is not the prepared gate")
    if context.backup.exists() or context.backup.is_symlink():
        raise ValueError("rollback backup already exists")
    _validate_static_source(context)
    old_activity = _optional_file_identity(context.runtime / "activity.jsonl")
    if old_activity is None:
        if context.expected.prepared_activity != (
            0,
            hashlib.sha256(b"").hexdigest(),
        ):
            raise ValueError("prepared activity snapshot is missing")
    elif old_activity != context.expected.prepared_activity:
        raise ValueError("live activity changed after workspace preparation")
    states = _asset_states(context, old_activity)
    if any(state != "old" for state in states.values()):
        raise ValueError("prepared workspace paths do not match old/live and new/staged")
    return old_activity


def _validate_static_source(context: _Context) -> None:
    _require_tree(context.source / "raw", context.expected.raw)
    for path, expected_hash in context.expected.reserved.items():
        if _file_identity(context.source / path)[1] != expected_hash:
            raise ValueError(f"reserved source changed: {path}")


def _assets(context: _Context, old_activity: _OldActivity) -> tuple[_Asset, ...]:
    return (
        _Asset(
            "backup-pages",
            context.source / "pages",
            context.staging / "pages",
            context.backup / "pages",
            context.expected.old_pages,
            context.expected.new_pages,
            True,
        ),
        _Asset(
            "backup-system",
            context.source / "system",
            context.staging / "system",
            context.backup / "system",
            context.expected.old_system,
            context.expected.new_system,
            True,
        ),
        _Asset(
            "backup-activity",
            context.runtime / "activity.jsonl",
            context.staging / "activity.jsonl",
            context.backup / "activity.jsonl",
            old_activity,
            context.expected.new_activity,
            False,
        ),
    )


def _asset_states(
    context: _Context,
    old_activity: _OldActivity,
    *,
    allow_activity_suffix: bool = False,
    expected_activity_suffix: tuple[int, str] | None = None,
) -> dict[str, Literal["old", "new", "missing-live"]]:
    result: dict[str, Literal["old", "new", "missing-live"]] = {}
    for asset in _assets(context, old_activity):
        live = asset.live.exists() or asset.live.is_symlink()
        staged = asset.staged.exists() or asset.staged.is_symlink()
        backup = asset.backup.exists() or asset.backup.is_symlink()
        for path, exists in (
            (asset.live, live),
            (asset.staged, staged),
            (asset.backup, backup),
        ):
            if exists and path.stat().st_dev != context.runtime.stat().st_dev:
                raise ValueError("all cutover assets must be on the same volume")
        if staged and backup and live:
            raise ValueError(f"duplicate cutover asset locations: {asset.name}")
        if asset.old is None:
            if backup:
                raise ValueError("absent old activity unexpectedly has a backup")
            if live:
                if staged:
                    raise ValueError("absent old activity has duplicate new copies")
                if allow_activity_suffix:
                    _require_activity_prefix_suffix(
                        context,
                        asset.live,
                        expected_suffix=expected_activity_suffix,
                    )
                else:
                    _require_identity(asset.live, asset.new, asset.directory)
                result[asset.name] = "new"
            else:
                if not staged:
                    raise ValueError("staged activity is missing")
                if expected_activity_suffix is None:
                    _require_identity(asset.staged, asset.new, asset.directory)
                else:
                    _require_activity_prefix_suffix(
                        context,
                        asset.staged,
                        expected_suffix=expected_activity_suffix,
                    )
                result[asset.name] = "old"
            continue
        if backup:
            _require_identity(asset.backup, asset.old, asset.directory)
        if staged:
            if asset.name == "backup-activity" and expected_activity_suffix is not None:
                _require_activity_prefix_suffix(
                    context,
                    asset.staged,
                    expected_suffix=expected_activity_suffix,
                )
            else:
                _require_identity(asset.staged, asset.new, asset.directory)
        if live:
            if backup:
                if staged:
                    raise ValueError(f"ambiguous live cutover asset: {asset.name}")
                if asset.name == "backup-activity" and allow_activity_suffix:
                    _require_activity_prefix_suffix(
                        context,
                        asset.live,
                        expected_suffix=expected_activity_suffix,
                    )
                else:
                    _require_identity(asset.live, asset.new, asset.directory)
                result[asset.name] = "new"
            elif staged:
                _require_identity(asset.live, asset.old, asset.directory)
                result[asset.name] = "old"
            else:
                raise ValueError(f"unaccounted live cutover asset: {asset.name}")
        else:
            if not (backup and staged):
                raise ValueError(f"missing cutover asset: {asset.name}")
            result[asset.name] = "missing-live"
    return result


def _require_activity_prefix_suffix(
    context: _Context,
    path: Path,
    *,
    expected_suffix: tuple[int, str] | None = None,
    allow_suffix_extension: bool = False,
) -> tuple[int, str]:
    """Validate the migrated immutable prefix and every mutable suffix row."""

    from chronovisor.core.activity_log import validated_activity_bytes

    raw = validated_activity_bytes(path)
    prefix_length, prefix_sha256 = context.expected.activity_prefix
    if (
        len(raw) < prefix_length
        or hashlib.sha256(raw[:prefix_length]).hexdigest() != prefix_sha256
    ):
        raise ValueError("live activity immutable migration prefix changed")
    suffix = raw[prefix_length:]
    identity = len(suffix), hashlib.sha256(suffix).hexdigest()
    if expected_suffix is not None:
        expected_length, expected_sha256 = expected_suffix
        if allow_suffix_extension:
            if (
                len(suffix) < expected_length
                or hashlib.sha256(suffix[:expected_length]).hexdigest()
                != expected_sha256
            ):
                raise ValueError("migration activity suffix identity changed")
        elif identity != expected_suffix:
            raise ValueError("migration activity suffix identity changed")
    return identity


def _require_new_live_layout(
    context: _Context,
    old_activity: _OldActivity,
) -> tuple[int, str]:
    states = _asset_states(
        context,
        old_activity,
        allow_activity_suffix=True,
    )
    if any(value != "new" for value in states.values()):
        raise ValueError("derived rebuild requires the complete new live layout")
    return _require_activity_prefix_suffix(
        context,
        context.runtime / "activity.jsonl",
    )


def _require_identity(
    path: Path,
    expected: Mapping[str, str] | tuple[int, str] | None,
    directory: bool,
) -> None:
    if directory:
        if not isinstance(expected, Mapping):
            raise TypeError("directory identity must be a mapping")
        _require_tree(path, expected)
    else:
        if expected is None or not isinstance(expected, tuple):
            raise TypeError("file identity must be a size/hash tuple")
        observed = _file_identity(path)
        if expected[0] >= 0 and observed[0] != expected[0]:
            raise ValueError(f"file size mismatch: {path}")
        if observed[1] != expected[1]:
            raise ValueError(f"file hash mismatch: {path}")


def _move(
    context: _Context,
    name: str,
    source: Path,
    destination: Path,
    *,
    mode: str,
    completed: list[str],
    old_activity: _OldActivity,
    fault_inject: FaultInjector | None,
    source_may_be_absent: bool,
    journal_extras: Mapping[str, object] | None = None,
    journal_state: str = "in-progress",
) -> None:
    _checkpoint(fault_inject, f"{name}:before-intent-journal")
    _write_journal(
        context,
        state=journal_state,
        mode=mode,
        phase="intent",
        step=name,
        completed=completed,
        old_activity=old_activity,
        extras=journal_extras,
    )
    _checkpoint(fault_inject, f"{name}:after-intent-journal")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"cutover destination already exists: {destination}")
    _reject_symlink(source, f"cutover source {name}")
    source_exists = source.exists()
    if source_exists:
        os.rename(source, destination)
    elif not source_may_be_absent:
        raise FileNotFoundError(f"cutover source is missing: {source}")
    _checkpoint(fault_inject, f"{name}:after-rename")
    if source_exists:
        _fsync_asset(destination)
    for parent in {source.parent, destination.parent}:
        fsync_directory(parent)
    _checkpoint(fault_inject, f"{name}:after-fsync")
    completed.append(name if source_exists else f"{name}:skipped")
    _write_journal(
        context,
        state=journal_state,
        mode=mode,
        phase="complete",
        step=name,
        completed=completed,
        old_activity=old_activity,
        extras=journal_extras,
    )
    _checkpoint(fault_inject, f"{name}:after-completion-journal")


def _finish(
    context: _Context,
    state: CutoverState,
    old_activity: _OldActivity,
    completed: list[str],
    *,
    fault_inject: FaultInjector | None,
) -> None:
    _write_journal(
        context,
        state=state,
        mode="cutover" if state == "committed-needs-rebuild" else "rollback",
        phase="complete",
        step=None,
        completed=completed,
        old_activity=old_activity,
    )
    _checkpoint(fault_inject, "after-terminal-journal")
    if state == "committed-needs-rebuild":
        _write_sentinel(context, state)
    else:
        _remove_sentinel(context)
    _checkpoint(fault_inject, "after-terminal-sentinel")


def _write_journal(
    context: _Context,
    *,
    state: str,
    mode: str,
    phase: str,
    step: str | None,
    completed: list[str],
    old_activity: _OldActivity,
    extras: Mapping[str, object] | None = None,
) -> None:
    payload: dict[str, object] = {
        "schema": JOURNAL_SCHEMA,
        "version": SCHEMA_VERSION,
        "run_id": context.workspace.name,
        "state": state,
        "manifest_sha256": context.manifest_sha256,
        "mode": mode,
        "phase": phase,
        "step": step,
        "completed": list(completed),
        "old_activity": (
            {"present": False}
            if old_activity is None
            else {
                "present": True,
                "size": old_activity[0],
                "sha256": old_activity[1],
            }
        ),
    }
    if extras:
        overlap = set(payload).intersection(extras)
        if overlap:
            raise ValueError("migration journal extras overlap fixed fields")
        payload.update(extras)
    _write_object(context.journal, payload)


def _write_sentinel(context: _Context, state: str) -> None:
    _write_object(
        context.sentinel,
        {
            "schema": SENTINEL_SCHEMA,
            "version": SCHEMA_VERSION,
            "run_id": context.workspace.name,
            "state": state,
            "manifest_sha256": context.manifest_sha256,
        },
    )


def _write_object(path: Path, payload: Mapping[str, object]) -> None:
    atomic_write_bytes(
        path,
        canonical_json_line_bytes_strict(payload),
        backup=False,
        min_free_bytes=0,
    )


def _receipt_payload(
    run_id: str, state: CutoverState, manifest_sha256: str
) -> dict[str, object]:
    return seal_object(
        {
            "schema": RECEIPT_SCHEMA,
            "version": SCHEMA_VERSION,
            "run_id": run_id,
            "state": state,
            "manifest_sha256": manifest_sha256,
        }
    )


def _identity_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_line_bytes_strict(value)).hexdigest()


def _final_receipt_payload(
    context: _Context,
    journal: Mapping[str, object],
) -> dict[str, object]:
    old_activity, suffix, extras = _require_finalized_journal(context, journal)
    manifest = _read_manifest(context.workspace / "dry-run-manifest.json")
    status_cohorts: list[dict[str, object]] = []
    for scope, field in (("pages", "status_cohorts"), ("system", "system_status_cohorts")):
        rows = _object_list(manifest, field)
        by_input = {row.get("input_status"): row for row in rows}
        if set(by_input) != set(_FINAL_STATUS_MAPPING):
            raise ValueError("migration status cohort inventory is invalid")
        for input_status, expected_output in _FINAL_STATUS_MAPPING.items():
            row = by_input[input_status]
            observed_input = row.get("input_status")
            output_status = row.get("output_status")
            count = row.get("count")
            if (
                observed_input != input_status
                or not isinstance(output_status, str)
                or not output_status
                or output_status != expected_output
                or not isinstance(count, int)
                or isinstance(count, bool)
                or count < 0
            ):
                raise ValueError("migration status cohort is invalid")
            if scope == "pages":
                identities = row.get("uids")
                if not isinstance(identities, list) or not all(
                    isinstance(uid, str) and uid for uid in identities
                ):
                    raise ValueError("migration status identity cohort is invalid")
                if len(identities) != count or len(identities) != len(set(identities)):
                    raise ValueError("migration status identity cohort count is invalid")
                identity_set_sha256 = _identity_sha256(sorted(identities))
            else:
                identity_set_sha256 = _sha(row.get("identity_set_sha256"))
            status_cohorts.append(
                {
                    "scope": scope,
                    "input_status": input_status,
                    "output_status": output_status,
                    "count": count,
                    "identity_set_sha256": identity_set_sha256,
                }
            )
    before = {
        "pages": context.expected.old_pages,
        "system": context.expected.old_system,
        "reserved": context.expected.reserved,
        "raw": context.expected.raw,
        "activity": old_activity,
    }
    after = {
        "pages": context.expected.new_pages,
        "system": context.expected.new_system,
        "raw": context.expected.raw,
        "activity": {
            "prefix": context.expected.activity_prefix,
            "suffix": suffix,
        },
    }
    generation = str(extras["derived_generation"])
    proof_sha256 = str(extras["rebuild_proof_sha256"])
    proof = _require_rebuild_proof(
        context,
        derived_generation=generation,
        rebuild_proof_sha256=proof_sha256,
    )
    corpus = proof.get("corpus")
    stable_count = corpus.get("stable_page_count") if isinstance(corpus, Mapping) else None
    if not isinstance(stable_count, int) or isinstance(stable_count, bool):
        raise ValueError("final rebuild corpus count is invalid")
    okf_version = manifest.get("okf_version")
    if not isinstance(okf_version, str) or not okf_version:
        raise ValueError("migration OKF version is invalid")
    return seal_object(
        {
            "schema": FINAL_RECEIPT_SCHEMA,
            "version": FINAL_RECEIPT_VERSION,
            "run_id": context.workspace.name,
            "state": "finalized-v2",
            "manifest_sha256": context.manifest_sha256,
            "before_manifest_sha256": _identity_sha256(before),
            "after_manifest_sha256": _identity_sha256(after),
            "transaction_version": SCHEMA_VERSION,
            "manifest_schema": MANIFEST_SCHEMA,
            "okf_version": okf_version,
            "status_mapping_cohorts": status_cohorts,
            "rollback_recutover": {
                "rollback": "complete",
                "recutover": "complete",
            },
            "rebuild_proof": {
                "derived_generation": generation,
                "sha256": proof_sha256,
                "stable_page_count": stable_count,
            },
            "activity_prefix": {
                "length": context.expected.activity_prefix[0],
                "sha256": context.expected.activity_prefix[1],
                "event_ids_sha256": context.expected.activity_event_ids_sha256,
            },
            "activity_suffix": {"length": suffix[0], "sha256": suffix[1]},
            "pages_log_sha256": context.expected.new_pages["log.md"],
            "system_schema_sha256": context.expected.new_system["schema.md"],
        }
    )


def _read_any_receipt(path: Path, run_id: str) -> dict[str, object]:
    try:
        receipt = verify_sealed_object(
            _read_canonical_object(path, "migration receipt", max_bytes=64 * 1024)
        )
    except StateSealError as exc:
        raise ValueError("migration receipt seal is invalid") from exc
    schema = receipt.get("schema")
    if schema == RECEIPT_SCHEMA:
        if set(receipt) != {
            "schema",
            "version",
            "run_id",
            "state",
            "manifest_sha256",
            "seal_sha256",
        }:
            raise ValueError("migration receipt fields are invalid")
        if (
            receipt.get("version") != SCHEMA_VERSION
            or receipt.get("run_id") != run_id
        ):
            raise ValueError("migration receipt identity is invalid")
        if receipt.get("state") != "rollback-complete":
            raise ValueError("migration receipt state is invalid")
        _sha(receipt.get("manifest_sha256"))
        return receipt
    if schema != FINAL_RECEIPT_SCHEMA:
        raise ValueError("migration receipt schema is invalid")
    expected = {
        "schema",
        "version",
        "run_id",
        "state",
        "manifest_sha256",
        "before_manifest_sha256",
        "after_manifest_sha256",
        "transaction_version",
        "manifest_schema",
        "okf_version",
        "status_mapping_cohorts",
        "rollback_recutover",
        "rebuild_proof",
        "activity_prefix",
        "activity_suffix",
        "pages_log_sha256",
        "system_schema_sha256",
        "seal_sha256",
    }
    if set(receipt) != expected:
        raise ValueError("final migration receipt fields are invalid")
    if (
        receipt.get("version") != FINAL_RECEIPT_VERSION
        or receipt.get("run_id") != run_id
        or receipt.get("state") != "finalized-v2"
        or receipt.get("transaction_version") != SCHEMA_VERSION
        or receipt.get("manifest_schema") != MANIFEST_SCHEMA
        or receipt.get("okf_version") != OKF_VERSION
        or receipt.get("rollback_recutover")
        != {"rollback": "complete", "recutover": "complete"}
    ):
        raise ValueError("final migration receipt identity is invalid")
    for field in (
        "manifest_sha256",
        "before_manifest_sha256",
        "after_manifest_sha256",
        "pages_log_sha256",
        "system_schema_sha256",
    ):
        _sha(receipt.get(field))
    cohorts = receipt.get("status_mapping_cohorts")
    expected_cohorts = [
        (scope, input_status, output_status)
        for scope in ("pages", "system")
        for input_status, output_status in _FINAL_STATUS_MAPPING.items()
    ]
    if not isinstance(cohorts, list) or len(cohorts) != len(expected_cohorts):
        raise ValueError("final migration status mapping is invalid")
    for cohort, expected_identity in zip(cohorts, expected_cohorts, strict=True):
        if (
            not isinstance(cohort, Mapping)
            or set(cohort)
            != {
                "scope",
                "input_status",
                "output_status",
                "count",
                "identity_set_sha256",
            }
            or (
                cohort.get("scope"),
                cohort.get("input_status"),
                cohort.get("output_status"),
            )
            != expected_identity
        ):
            raise ValueError("final migration status mapping is invalid")
        count = cohort.get("count")
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise ValueError("final migration status mapping is invalid")
        _sha(cohort.get("identity_set_sha256"))
    rebuild = receipt.get("rebuild_proof")
    if not isinstance(rebuild, Mapping) or set(rebuild) != {
        "derived_generation",
        "sha256",
        "stable_page_count",
    }:
        raise ValueError("final migration rebuild proof is invalid")
    generation = rebuild.get("derived_generation")
    if (
        not isinstance(generation, str)
        or not generation
        or len(generation) > 128
        or re.fullmatch(r"[a-z0-9-]+", generation) is None
    ):
        raise ValueError("final migration rebuild generation is invalid")
    _sha(rebuild.get("sha256"))
    stable_page_count = rebuild.get("stable_page_count")
    if (
        not isinstance(stable_page_count, int)
        or isinstance(stable_page_count, bool)
        or stable_page_count < 0
    ):
        raise ValueError("final migration rebuild count is invalid")
    for field, with_events in (("activity_prefix", True), ("activity_suffix", False)):
        identity = receipt.get(field)
        expected_fields = {"length", "sha256"}
        if with_events:
            expected_fields.add("event_ids_sha256")
        if not isinstance(identity, Mapping) or set(identity) != expected_fields:
            raise ValueError("final migration activity identity is invalid")
        length = identity.get("length")
        if not isinstance(length, int) or isinstance(length, bool) or length < 0:
            raise ValueError("final migration activity length is invalid")
        _sha(identity.get("sha256"))
        if with_events:
            _sha(identity.get("event_ids_sha256"))
    return receipt


def _read_receipt(path: Path, run_id: str) -> tuple[CutoverState, str]:
    receipt = _read_any_receipt(path, run_id)
    state = receipt.get("state")
    if receipt.get("schema") != RECEIPT_SCHEMA or state != "rollback-complete":
        raise ValueError("migration receipt state is invalid")
    return "rollback-complete", _sha(receipt.get("manifest_sha256"))


def _require_cleanup_journal(
    journal: Mapping[str, object],
    run_id: str,
    state: CutoverState,
    manifest_sha256: str,
) -> None:
    if (
        journal.get("schema") != JOURNAL_SCHEMA
        or journal.get("version") != SCHEMA_VERSION
        or journal.get("run_id") != run_id
        or journal.get("state") != state
        or journal.get("manifest_sha256") != manifest_sha256
        or journal.get("cleanup_in_progress") is not True
    ):
        raise ValueError("migration cleanup journal is invalid")


def _receipt_only(workspace: Path) -> bool:
    return _directory_entries(workspace) == {RECEIPT_FILENAME: "file"}


def _require_cleanup_workspace(workspace: Path) -> None:
    if _tree_has_unsafe_entry(workspace):
        raise ValueError("migration cleanup workspace is unsafe")
    entries = _directory_entries(workspace)
    if any(
        _LEGACY_CLEANUP_ENTRIES.get(name) != kind
        for name, kind in entries.items()
    ):
        raise ValueError("migration cleanup workspace has an unknown artifact")
    if entries.get("journal.json") != "file":
        raise ValueError("migration cleanup journal is missing")
    for name in ("staging", "rollback-backup"):
        _require_known_cleanup_tree(workspace / name)


def _require_known_cleanup_tree(path: Path) -> None:
    kind = _path_kind(path)
    if kind == "absent":
        return
    if kind != "directory":
        raise ValueError("migration cleanup tree is unsafe")
    entries = _directory_entries(path)
    allowed = {
        "activity.jsonl": "file",
        "pages": "directory",
        "system": "directory",
    }
    if any(allowed.get(name) != kind for name, kind in entries.items()):
        raise ValueError("migration cleanup tree has an unknown artifact")


def _require_receipt_layout(
    source_root: Path, runtime_root: Path, state: CutoverState
) -> None:
    source = _safe_root(source_root, "source root")
    runtime = _safe_root(runtime_root, "runtime root")
    if _unsafe_legacy_root(source):
        raise ValueError("migration receipt layout is unsafe")
    for name in ("raw", "pages", "system"):
        if _path_kind(source / name) != "directory":
            raise ValueError("migration receipt layout is incomplete")

    reserved = tuple(_path_kind(source / name) for name in _ROOT_RESERVED)
    present = sum(kind == "file" for kind in reserved)
    if any(kind not in {"absent", "file"} for kind in reserved):
        raise ValueError("migration receipt root documents are unsafe")
    activity_kind = _path_kind(runtime / "activity.jsonl")
    if present != len(_ROOT_RESERVED) or activity_kind not in {"absent", "file"}:
        raise ValueError("rolled-back receipt layout is incomplete")


def _require_final_receipt_layout(
    source_root: Path,
    runtime_root: Path,
    receipt: Mapping[str, object],
) -> None:
    _require_final_receipt_layout_files(source_root, runtime_root, receipt)
    source = _safe_root(source_root, "source root")
    if any(_path_kind(source / name) != "absent" for name in _ROOT_RESERVED):
        raise ValueError("final receipt retained legacy root documents")


def _require_partial_final_cleanup_layout(
    context: _Context,
    receipt: Mapping[str, object],
) -> None:
    _require_final_receipt_layout_files(context.source, context.runtime, receipt)
    for name, expected_hash in context.expected.reserved.items():
        kind = _path_kind(context.source / name)
        if kind == "absent":
            continue
        if kind != "file" or _file_identity(context.source / name)[1] != expected_hash:
            raise ValueError("final cleanup legacy root identity changed")


def _require_final_receipt_layout_files(
    source_root: Path,
    runtime_root: Path,
    receipt: Mapping[str, object],
) -> None:
    source = _safe_root(source_root, "source root")
    runtime = _safe_root(runtime_root, "runtime root")
    if runtime != source / "runtime":
        raise ValueError("final receipt runtime root is not canonical")
    for name in ("raw", "pages", "system", "runtime"):
        if _path_kind(source / name) != "directory":
            raise ValueError("final receipt layout is incomplete")
    prefix = receipt.get("activity_prefix")
    suffix = receipt.get("activity_suffix")
    if not isinstance(prefix, Mapping) or not isinstance(suffix, Mapping):
        raise ValueError("final receipt activity prefix is invalid")
    if not _is_canonical_live_layout(
        source,
        {
            "log_sha256": receipt.get("pages_log_sha256"),
            "schema_sha256": receipt.get("system_schema_sha256"),
            "activity_prefix": {
                "length": prefix.get("length"),
                "sha256": prefix.get("sha256"),
            },
        },
    ):
        raise ValueError("final receipt live layout is invalid")
    prefix_length = prefix.get("length")
    suffix_length = suffix.get("length")
    suffix_sha256 = suffix.get("sha256")
    if (
        not isinstance(prefix_length, int)
        or isinstance(prefix_length, bool)
        or not isinstance(suffix_length, int)
        or isinstance(suffix_length, bool)
        or not isinstance(suffix_sha256, str)
        or not _activity_segment_matches_nofollow(
            runtime / "activity.jsonl",
            offset=prefix_length,
            length=suffix_length,
            sha256=suffix_sha256,
        )
    ):
        raise ValueError("final receipt activity suffix is invalid")


def _activity_segment_matches_nofollow(
    path: Path,
    *,
    offset: int,
    length: int,
    sha256: str,
) -> bool:
    if offset < 0 or length < 0 or _SHA256_RE.fullmatch(sha256) is None:
        return False
    try:
        with open_regular_nofollow(path) as handle:
            snapshot = os.fstat(handle.fileno())
            if not stat.S_ISREG(snapshot.st_mode) or snapshot.st_size < offset + length:
                return False
            handle.seek(offset)
            digest = hashlib.sha256()
            remaining = length
            while remaining:
                chunk = handle.read(min(64 * 1024, remaining))
                if not chunk:
                    return False
                digest.update(chunk)
                remaining -= len(chunk)
            return digest.hexdigest() == sha256
    except OSError:
        return False


def _remove_file_exact(path: Path, *, required: bool = False) -> None:
    kind = _path_kind(path)
    if kind == "absent":
        if required:
            raise ValueError("required cleanup artifact is missing")
        return
    if kind != "file":
        raise ValueError("cleanup artifact is not a regular file")
    path.unlink()
    fsync_directory(path.parent)


def _remove_tree_exact(path: Path) -> None:
    try:
        with open_directory_nofollow(path.absolute().parent) as parent_fd:
            try:
                mode = os.stat(
                    path.name,
                    dir_fd=parent_fd,
                    follow_symlinks=False,
                ).st_mode
            except FileNotFoundError:
                return
            if not stat.S_ISDIR(mode):
                raise ValueError("cleanup artifact is not a directory")
            _remove_tree_at(parent_fd, path.name)
            os.fsync(parent_fd)
    except OSError as exc:
        raise ValueError("cleanup artifact parent is unsafe") from exc


def _remove_tree_at(parent_fd: int, name: str) -> None:
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    flags |= getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=parent_fd)
    except OSError as exc:
        raise ValueError("cleanup artifact is not a safe directory") from exc
    try:
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise ValueError("cleanup artifact is not a directory")
        with os.scandir(descriptor) as iterator:
            entries = sorted(
                (
                    entry.name,
                    _mode_kind(entry.stat(follow_symlinks=False).st_mode),
                )
                for entry in iterator
            )
        for child, child_kind in entries:
            if child_kind == "directory":
                _remove_tree_at(descriptor, child)
            elif child_kind == "file":
                os.unlink(child, dir_fd=descriptor)
                os.fsync(descriptor)
            else:
                raise ValueError("cleanup tree contains an unsafe artifact")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.rmdir(name, dir_fd=parent_fd)


def _read_canonical_object(
    path: Path,
    label: str,
    *,
    max_bytes: int = 16 * 1024 * 1024,
) -> dict[str, object]:
    try:
        with open_regular_nofollow(path) as handle:
            snapshot = os.fstat(handle.fileno())
            if not stat.S_ISREG(snapshot.st_mode) or snapshot.st_size > max_bytes:
                raise ValueError(f"{label} is oversized or unsafe")
            raw = handle.read(snapshot.st_size + 1)
        payload = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is unreadable") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} is not an object")
    if canonical_json_line_bytes_strict(payload) != raw:
        raise ValueError(f"{label} is not canonical JSON")
    return payload


def _read_manifest(path: Path) -> dict[str, object]:
    return _read_canonical_object(
        path, "migration manifest", max_bytes=_MANIFEST_MAX_BYTES
    )


def _require_gate_identity(
    gate: Mapping[str, object], context: _Context, schema: str
) -> None:
    if (
        gate.get("schema") != schema
        or gate.get("version") != SCHEMA_VERSION
        or gate.get("run_id") != context.workspace.name
        or gate.get("manifest_sha256") != context.manifest_sha256
    ):
        raise ValueError("migration gate identity does not match the workspace")


def _require_active_sentinel(context: _Context) -> None:
    sentinel = _read_canonical_object(context.sentinel, "restart refusal sentinel")
    _require_gate_identity(sentinel, context, SENTINEL_SCHEMA)
    if sentinel.get("state") not in {"prepared", "in-progress"}:
        raise ValueError("restart refusal sentinel state is invalid")


def _require_rebuild_sentinel_states(
    context: _Context,
    allowed: set[str],
) -> None:
    sentinel = _read_canonical_object(context.sentinel, "restart refusal sentinel")
    _require_gate_identity(sentinel, context, SENTINEL_SCHEMA)
    if sentinel.get("state") not in allowed:
        raise ValueError("restart refusal sentinel state is invalid")


def _write_rebuild_journal(
    context: _Context,
    *,
    state: str,
    old_activity: _OldActivity,
    derived_generation: str | None,
    rebuild_proof_sha256: str | None = None,
    activity_suffix: tuple[int, str] | None = None,
) -> None:
    payload: dict[str, object] = {
        "schema": JOURNAL_SCHEMA,
        "version": SCHEMA_VERSION,
        "run_id": context.workspace.name,
        "state": state,
        "manifest_sha256": context.manifest_sha256,
        "mode": "rebuild",
        "phase": "complete" if state == "sealed-rebuild" else "in-progress",
        "step": None,
        "completed": [],
        "old_activity": (
            {"present": False}
            if old_activity is None
            else {
                "present": True,
                "size": old_activity[0],
                "sha256": old_activity[1],
            }
        ),
    }
    if derived_generation is not None:
        payload["derived_generation"] = derived_generation
    if rebuild_proof_sha256 is not None:
        payload["rebuild_proof_sha256"] = rebuild_proof_sha256
    if activity_suffix is not None:
        payload["activity_suffix"] = {
            "length": activity_suffix[0],
            "sha256": activity_suffix[1],
        }
    _write_object(context.journal, payload)


def _require_activity_suffix_identity(
    journal: Mapping[str, object], expected: tuple[int, str]
) -> None:
    value = journal.get("activity_suffix")
    if not isinstance(value, Mapping) or set(value) != {"length", "sha256"}:
        raise ValueError("migration activity suffix identity is missing")
    if value.get("length") != expected[0] or value.get("sha256") != expected[1]:
        raise ValueError("migration activity suffix changed during rebuild")


def _activity_suffix_from_journal(
    journal: Mapping[str, object],
) -> tuple[int, str]:
    value = journal.get("activity_suffix")
    if not isinstance(value, Mapping) or set(value) != {"length", "sha256"}:
        raise ValueError("migration activity suffix identity is missing")
    length = value.get("length")
    if not isinstance(length, int) or isinstance(length, bool) or length < 0:
        raise ValueError("migration activity suffix length is invalid")
    return length, _sha(value.get("sha256"))


def _derived_journal_extras(
    context: _Context,
    journal: Mapping[str, object],
    *,
    rollback_outcome: str,
    recutover_outcome: str,
) -> dict[str, object]:
    if rollback_outcome not in {"pending", "in-progress", "complete"}:
        raise ValueError("rollback drill outcome is invalid")
    if recutover_outcome not in {"pending", "in-progress", "complete"}:
        raise ValueError("recutover outcome is invalid")
    generation = journal.get("derived_generation")
    if not isinstance(generation, str) or not generation or len(generation) > 128:
        raise ValueError("derived rebuild generation is missing")
    proof_sha256 = _sha(journal.get("rebuild_proof_sha256"))
    suffix = _activity_suffix_from_journal(journal)
    _require_rebuild_proof(
        context,
        derived_generation=generation,
        rebuild_proof_sha256=proof_sha256,
    )
    return {
        "derived_generation": generation,
        "rebuild_proof_sha256": proof_sha256,
        "activity_suffix": {"length": suffix[0], "sha256": suffix[1]},
        "rollback_outcome": rollback_outcome,
        "recutover_outcome": recutover_outcome,
    }


def _require_finalized_journal(
    context: _Context,
    journal: Mapping[str, object],
) -> tuple[_OldActivity, tuple[int, str], dict[str, object]]:
    _require_gate_identity(journal, context, JOURNAL_SCHEMA)
    if (
        journal.get("state") != "finalized-v2"
        or journal.get("mode") != "recutover"
        or journal.get("phase") != "complete"
        or journal.get("rollback_outcome") != "complete"
        or journal.get("recutover_outcome") != "complete"
    ):
        raise ValueError("finalized migration journal is invalid")
    extras = _derived_journal_extras(
        context,
        journal,
        rollback_outcome="complete",
        recutover_outcome="complete",
    )
    suffix = _activity_suffix_from_journal(journal)
    old_activity = _old_activity(journal, context, prepared=False)
    if any(
        value != "new"
        for value in _asset_states(
            context,
            old_activity,
            allow_activity_suffix=True,
        ).values()
    ):
        raise ValueError("finalized migration assets do not match the new layout")
    if context.sentinel.exists() or context.sentinel.is_symlink():
        _require_rebuild_sentinel_states(
            context, {"rebuild-in-progress", "sealed-rebuild"}
        )
    current_suffix = _require_activity_prefix_suffix(
        context,
        context.runtime / "activity.jsonl",
        expected_suffix=suffix,
        allow_suffix_extension=True,
    )
    return old_activity, current_suffix, extras


def _validate_rebuild_restricted_object(
    value: object,
    fields: Mapping[str, str],
    *,
    label: str,
) -> None:
    if not isinstance(value, Mapping) or set(value) != set(fields):
        raise ValueError(f"derived rebuild {label} fields are invalid")
    for name, kind in fields.items():
        item = value[name]
        if kind == "sha256":
            _sha(item)
        elif kind == "count":
            if not isinstance(item, int) or isinstance(item, bool) or item < 0:
                raise ValueError(f"derived rebuild {label} count is invalid")
        elif kind == "bool":
            if not isinstance(item, bool):
                raise ValueError(f"derived rebuild {label} flag is invalid")
        else:  # pragma: no cover - constant table invariant
            raise RuntimeError("unknown derived rebuild proof field type")


def _validate_rebuild_proof_payload(payload: Mapping[str, object]) -> None:
    if set(payload) != {"derived_generation", "corpus", "components"}:
        raise ValueError("derived rebuild proof fields are invalid")
    generation = payload.get("derived_generation")
    if (
        not isinstance(generation, str)
        or not generation
        or len(generation) > 128
        or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789-" for character in generation)
    ):
        raise ValueError("derived rebuild generation is invalid")
    _validate_rebuild_restricted_object(
        payload.get("corpus"), _REBUILD_CORPUS_FIELDS, label="corpus"
    )
    components = payload.get("components")
    if not isinstance(components, Mapping) or set(components) != set(
        _REBUILD_COMPONENT_FIELDS
    ):
        raise ValueError("derived rebuild component inventory is invalid")
    for name, fields in _REBUILD_COMPONENT_FIELDS.items():
        _validate_rebuild_restricted_object(
            components[name], fields, label=f"component {name}"
        )


def _require_rebuild_proof(
    context: _Context,
    *,
    derived_generation: str,
    rebuild_proof_sha256: str,
) -> dict[str, object]:
    path = context.workspace / REBUILD_PROOF_FILENAME
    with open_regular_nofollow(path) as handle:
        raw = handle.read(1024 * 1024 + 1)
    if len(raw) > 1024 * 1024:
        raise ValueError("derived rebuild proof is oversized")
    if hashlib.sha256(raw).hexdigest() != rebuild_proof_sha256:
        raise ValueError("derived rebuild proof hash mismatch")
    try:
        parsed = json.loads(raw)
        if not isinstance(parsed, dict) or canonical_json_line_bytes_strict(parsed) != raw:
            raise ValueError("derived rebuild proof is not canonical JSON")
        proof = verify_sealed_object(parsed)
    except (json.JSONDecodeError, StateSealError) as exc:
        raise ValueError("derived rebuild proof seal is invalid") from exc
    if (
        proof.get("schema") != REBUILD_PROOF_SCHEMA
        or proof.get("version") != SCHEMA_VERSION
        or proof.get("run_id") != context.workspace.name
        or proof.get("manifest_sha256") != context.manifest_sha256
        or proof.get("derived_generation") != derived_generation
    ):
        raise ValueError("derived rebuild proof identity is invalid")
    _validate_rebuild_proof_payload(
        {
            "derived_generation": proof.get("derived_generation"),
            "corpus": proof.get("corpus"),
            "components": proof.get("components"),
        }
    )
    return proof


def _old_activity(
    journal: Mapping[str, object], context: _Context, *, prepared: bool
) -> _OldActivity:
    if prepared:
        return _validate_prepared(context)
    item = journal.get("old_activity")
    if not isinstance(item, dict):
        raise ValueError("migration journal has no old activity identity")
    present = item.get("present")
    if present is False:
        if set(item) != {"present"}:
            raise ValueError("absent old activity identity must not contain a hash")
        return None
    if present is not True:
        raise ValueError("migration journal old activity presence is invalid")
    size = item.get("size")
    if not isinstance(size, int) or isinstance(size, bool) or size < 0:
        raise ValueError("migration journal old activity size is invalid")
    return size, _sha(item.get("sha256"))


def _remove_sentinel(context: _Context) -> None:
    if context.sentinel.is_symlink():
        raise ValueError("restart refusal sentinel must not be a symlink")
    if context.sentinel.exists():
        context.sentinel.unlink()
        fsync_directory(context.workspace)


def _require_tree(
    root: Path, expected: Mapping[str, str] | Mapping[str, tuple[int, str]]
) -> None:
    _reject_symlink(root, "managed tree")
    if not root.is_dir():
        raise ValueError(f"managed tree is missing: {root}")
    observed: dict[str, str] | dict[str, tuple[int, str]]
    if expected and isinstance(next(iter(expected.values())), tuple):
        observed = {
            path.relative_to(root).as_posix(): _file_identity(path)
            for path in _tree_files(root)
        }
    else:
        observed = {
            path.relative_to(root).as_posix(): _file_identity(path)[1]
            for path in _tree_files(root)
        }
    if observed != dict(expected):
        raise ValueError(f"managed tree hash inventory mismatch: {root}")


def _tree_files(root: Path) -> tuple[Path, ...]:
    files: list[Path] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"symlink is not allowed in managed tree: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"unsupported managed tree entry: {path}")
        files.append(path)
    return tuple(files)


def _file_identity(path: Path) -> tuple[int, str]:
    _reject_symlink(path, "managed file")
    if not path.is_file():
        raise ValueError(f"managed file is missing: {path}")
    data = path.read_bytes()
    return len(data), hashlib.sha256(data).hexdigest()


def _optional_file_identity(path: Path) -> _OldActivity:
    _reject_symlink(path, "managed file")
    return _file_identity(path) if path.exists() else None


def _fsync_asset(path: Path) -> None:
    if path.is_dir():
        for file_path in _tree_files(path):
            _fsync_file(file_path)
        for directory in sorted(
            (item for item in path.rglob("*") if item.is_dir()),
            key=lambda item: len(item.parts),
            reverse=True,
        ):
            fsync_directory(directory)
        fsync_directory(path)
    else:
        _fsync_file(path)


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _mkdir_durable(path: Path) -> None:
    if path.is_symlink():
        raise ValueError(f"cutover directory must not be a symlink: {path}")
    if path.exists():
        if not path.is_dir():
            raise ValueError(f"cutover directory is not a directory: {path}")
        return
    path.mkdir()
    fsync_directory(path.parent)


def _safe_root(path: Path, label: str) -> Path:
    _reject_symlink(path, label)
    resolved = path.resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError(f"{label} is not a directory")
    return resolved


def _reject_symlink(path: Path, label: str) -> None:
    if path.is_symlink():
        raise ValueError(f"{label} must not be a symlink")


def _relative(value: object) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\0" in value:
        raise ValueError("migration manifest contains an unsafe relative path")
    if "//" in value or value.startswith("/") or value.endswith("/"):
        raise ValueError("migration manifest contains an unsafe relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in value.split("/")):
        raise ValueError("migration manifest contains an unsafe relative path")
    return path.as_posix()


def _put(target: dict[str, _Value], key: str, value: _Value) -> None:
    if key in target:
        raise ValueError(f"duplicate migration manifest path: {key}")
    target[key] = value


def _put_scoped(
    pages: dict[str, str], system: dict[str, str], staged_path: str, digest: object
) -> None:
    scope, separator, relative = staged_path.partition("/")
    if not separator or scope not in {"pages", "system"}:
        raise ValueError("migration staged path has an invalid namespace")
    _put(pages if scope == "pages" else system, _relative(relative), _sha(digest))


def _sha(value: object) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError("migration manifest contains an invalid SHA-256")
    return value


def _object_list(
    payload: Mapping[str, object], key: str
) -> tuple[dict[str, object], ...]:
    value = payload.get(key)
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError(f"migration manifest {key} is invalid")
    return tuple(value)


def _checkpoint(fault_inject: FaultInjector | None, point: str) -> None:
    if fault_inject is not None:
        fault_inject(point)
