"""Crash-safe offline cutover for a validated OKF migration workspace."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal, TypeVar

from chronovisor.core.canonical_json import canonical_json_line_bytes_strict
from chronovisor.core.durable_state import (
    atomic_write_bytes,
    file_lock,
    fsync_directory,
)
from chronovisor.core.okf_workspace import (
    JOURNAL_SCHEMA,
    MANIFEST_SCHEMA,
    RESTART_REFUSAL_FILENAME,
    SCHEMA_VERSION,
    SENTINEL_SCHEMA,
)

CutoverState = Literal["committed", "rollback-complete"]
FaultInjector = Callable[[str], None]

_RUN_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
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
    "before-commit-journal",
    "after-commit-journal",
    "after-sentinel-remove",
)


@dataclass(frozen=True, slots=True)
class _Expected:
    old_pages: dict[str, str]
    old_system: dict[str, str]
    new_pages: dict[str, str]
    new_system: dict[str, str]
    raw: dict[str, tuple[int, str]]
    reserved: dict[str, str]
    new_activity: tuple[int, str]


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
    old: Mapping[str, str] | tuple[int, str]
    new: Mapping[str, str] | tuple[int, str]
    directory: bool


def execute_okf_cutover(
    source_root: Path,
    runtime_root: Path,
    run_id: str,
    *,
    is_quiescent: Callable[[], bool],
    fault_inject: FaultInjector | None = None,
) -> CutoverState:
    """Atomically coordinate the three direct-live OKF assets while offline."""

    context = _context(source_root, runtime_root, run_id)
    _reject_symlink(context.workspace / "cutover.lock", "cutover lock")
    with file_lock(context.workspace / "cutover.lock"):
        _validate_prepared(context)
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
            )

        states = _asset_states(context, old_activity)
        if any(state != "new" for state in states.values()):
            raise RuntimeError("OKF cutover did not publish every coordinated asset")
        _validate_static_source(context)
        _checkpoint(fault_inject, "before-commit-journal")
        _finish(
            context,
            "committed",
            old_activity,
            completed,
            fault_inject=fault_inject,
        )
        return "committed"


def recover_okf_cutover(
    source_root: Path,
    runtime_root: Path,
    run_id: str,
    *,
    is_quiescent: Callable[[], bool],
) -> CutoverState:
    """Resolve an interrupted cutover to all-new or all-old from disk truth."""

    context = _context(source_root, runtime_root, run_id)
    _reject_symlink(context.workspace / "cutover.lock", "cutover lock")
    with file_lock(context.workspace / "cutover.lock"):
        if not is_quiescent():
            raise RuntimeError("OKF recovery requires a quiescent runtime")
        journal = _read_canonical_object(context.journal, "migration journal")
        _require_gate_identity(journal, context, JOURNAL_SCHEMA)
        state = journal.get("state")
        if state not in {
            "prepared",
            "in-progress",
            "committed",
            "rollback-complete",
        }:
            raise ValueError("migration journal has an unknown state")
        _validate_static_source(context)
        old_activity = _old_activity(journal, context, prepared=state == "prepared")
        if state in {"prepared", "in-progress"}:
            _require_active_sentinel(context)
        states = _asset_states(context, old_activity)

        if state == "committed":
            if any(value != "new" for value in states.values()):
                raise ValueError("committed migration assets do not match the manifest")
            _remove_sentinel(context)
            return "committed"
        if state == "rollback-complete":
            if any(value != "old" for value in states.values()):
                raise ValueError("rolled-back migration assets do not match the manifest")
            _remove_sentinel(context)
            return "rollback-complete"
        if all(value == "new" for value in states.values()):
            _finish(context, "committed", old_activity, [], fault_inject=None)
            return "committed"

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


def okf_startup_allowed(source_root: Path, runtime_root: Path, run_id: str) -> bool:
    """Fail closed unless a terminal journal agrees with paths and hashes."""

    try:
        context = _context(source_root, runtime_root, run_id)
        if context.sentinel.exists() or context.sentinel.is_symlink():
            return False
        journal = _read_canonical_object(context.journal, "migration journal")
        _require_gate_identity(journal, context, JOURNAL_SCHEMA)
        state = journal.get("state")
        if state not in {"committed", "rollback-complete"}:
            return False
        _validate_static_source(context)
        old_activity = _old_activity(journal, context, prepared=False)
        expected = "new" if state == "committed" else "old"
        return all(
            value == expected
            for value in _asset_states(context, old_activity).values()
        )
    except (OSError, TypeError, ValueError):
        return False


def _context(source_root: Path, runtime_root: Path, run_id: str) -> _Context:
    if not _RUN_ID_RE.fullmatch(run_id):
        raise ValueError("run_id must be a safe single path component")
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
    staging = workspace / "staging"
    manifest_path = workspace / "dry-run-manifest.json"
    _reject_symlink(manifest_path, "migration manifest")
    manifest_raw = manifest_path.read_bytes()
    manifest = _read_canonical_object(manifest_path, "migration manifest")
    if (
        manifest.get("schema") != MANIFEST_SCHEMA
        or manifest.get("version") != SCHEMA_VERSION
        or manifest.get("run_id") != run_id
        or manifest.get("state") != "validated"
    ):
        raise ValueError("migration manifest identity is invalid")
    devices = {source.stat().st_dev, runtime.stat().st_dev, workspace.stat().st_dev}
    if len(devices) != 1:
        raise ValueError("source, runtime, and workspace must be on the same volume")
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
    return _Expected(
        old_pages,
        old_system,
        new_pages,
        new_system,
        raw,
        reserved,
        (-1, _sha(staged_activity)),
    )


def _validate_prepared(context: _Context) -> tuple[int, str]:
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
    old_activity = _file_identity(context.runtime / "activity.jsonl")
    states = _asset_states(context, old_activity)
    if any(state != "old" for state in states.values()):
        raise ValueError("prepared workspace paths do not match old/live and new/staged")
    return old_activity


def _validate_static_source(context: _Context) -> None:
    _require_tree(context.source / "raw", context.expected.raw)
    for path, expected_hash in context.expected.reserved.items():
        if _file_identity(context.source / path)[1] != expected_hash:
            raise ValueError(f"reserved source changed: {path}")


def _assets(context: _Context, old_activity: tuple[int, str]) -> tuple[_Asset, ...]:
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
    context: _Context, old_activity: tuple[int, str]
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
        if backup:
            _require_identity(asset.backup, asset.old, asset.directory)
        if staged:
            _require_identity(asset.staged, asset.new, asset.directory)
        if live:
            if backup:
                if staged:
                    raise ValueError(f"ambiguous live cutover asset: {asset.name}")
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


def _require_identity(
    path: Path, expected: Mapping[str, str] | tuple[int, str], directory: bool
) -> None:
    if directory:
        if not isinstance(expected, Mapping):
            raise TypeError("directory identity must be a mapping")
        _require_tree(path, expected)
    else:
        if not isinstance(expected, tuple):
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
    old_activity: tuple[int, str],
    fault_inject: FaultInjector | None,
) -> None:
    _checkpoint(fault_inject, f"{name}:before-intent-journal")
    _write_journal(
        context,
        state="in-progress",
        mode=mode,
        phase="intent",
        step=name,
        completed=completed,
        old_activity=old_activity,
    )
    _checkpoint(fault_inject, f"{name}:after-intent-journal")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"cutover destination already exists: {destination}")
    _reject_symlink(source, f"cutover source {name}")
    os.rename(source, destination)
    _checkpoint(fault_inject, f"{name}:after-rename")
    _fsync_asset(destination)
    for parent in {source.parent, destination.parent}:
        fsync_directory(parent)
    _checkpoint(fault_inject, f"{name}:after-fsync")
    completed.append(name)
    _write_journal(
        context,
        state="in-progress",
        mode=mode,
        phase="complete",
        step=name,
        completed=completed,
        old_activity=old_activity,
    )
    _checkpoint(fault_inject, f"{name}:after-completion-journal")


def _finish(
    context: _Context,
    state: CutoverState,
    old_activity: tuple[int, str],
    completed: list[str],
    *,
    fault_inject: FaultInjector | None,
) -> None:
    _write_journal(
        context,
        state=state,
        mode="cutover" if state == "committed" else "rollback",
        phase="complete",
        step=None,
        completed=completed,
        old_activity=old_activity,
    )
    _checkpoint(fault_inject, "after-commit-journal")
    _remove_sentinel(context)
    _checkpoint(fault_inject, "after-sentinel-remove")


def _write_journal(
    context: _Context,
    *,
    state: str,
    mode: str,
    phase: str,
    step: str | None,
    completed: list[str],
    old_activity: tuple[int, str],
) -> None:
    _write_object(
        context.journal,
        {
            "schema": JOURNAL_SCHEMA,
            "version": SCHEMA_VERSION,
            "run_id": context.workspace.name,
            "state": state,
            "manifest_sha256": context.manifest_sha256,
            "mode": mode,
            "phase": phase,
            "step": step,
            "completed": list(completed),
            "old_activity": {
                "size": old_activity[0],
                "sha256": old_activity[1],
            },
        },
    )


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


def _read_canonical_object(path: Path, label: str) -> dict[str, object]:
    _reject_symlink(path, label)
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is unreadable") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} is not an object")
    if canonical_json_line_bytes_strict(payload) != raw:
        raise ValueError(f"{label} is not canonical JSON")
    return payload


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


def _old_activity(
    journal: Mapping[str, object], context: _Context, *, prepared: bool
) -> tuple[int, str]:
    if prepared:
        states = _asset_states(context, _file_identity(context.runtime / "activity.jsonl"))
        if any(value != "old" for value in states.values()):
            raise ValueError("prepared workspace paths are inconsistent")
        return _file_identity(context.runtime / "activity.jsonl")
    item = journal.get("old_activity")
    if not isinstance(item, dict):
        raise ValueError("migration journal has no old activity identity")
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
