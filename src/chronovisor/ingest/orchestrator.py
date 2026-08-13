"""Orchestrator - deterministic control flow for Ingest/Lint scheduling.

NOT an LLM. Pure code logic. Local Ollama models handle content structuring;
this module handles when to trigger them.
"""

import fcntl
import hashlib
import json
import os
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any

from chronovisor.core import activity_log, runtime_status
from chronovisor.core.durable_state import fsync_directory as _fsync_directory
from chronovisor.core.link_fix import atomic_write
from chronovisor.core.store import ACTIVITY_FILE, CHRONOVISOR_ROOT, RAW_DIR

# Config
INGEST_THRESHOLD = 5  # Trigger ingest after N raw files
MAX_INGEST_BATCH_UNITS = 10  # Hard safety ceiling for semantic units per batch
LINT_INTERVAL_HOURS = 24  # Run lint every N hours

# State file
STATE_FILE = CHRONOVISOR_ROOT / ".orchestrator_state.json"

# In-process lock so concurrent calls into run_pending_ingest can't both
# spawn an ingest thread for the same batch. The state file holds the
# cross-call truth (current_job_id), this lock just serializes the
# read-modify-write around it.
_INGEST_LOCK = threading.Lock()
_INGEST_PROCESS_LEASE_ACTIVE = False


class _CompletionAckResumed(Exception):
    """Internal control signal: semantic work was already durably completed."""


def _valid_ingest_review_shard_continuation(value: object) -> bool:
    """Accept only the exact bounded-progress envelope emitted by ingest."""

    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "kind",
        "full_proposal_sha256",
        "manifest_sha256",
        "approved_shards",
        "total_shards",
        "remaining_shards",
        "review_calls_used",
        "review_call_limit",
    }:
        return False
    if value.get("schema_version") != 1 or value.get("kind") != (
        "ingest_review_shard_continuation"
    ):
        return False
    if any(
        not isinstance(value.get(field), str)
        or len(value[field]) != 64
        or any(character not in "0123456789abcdef" for character in value[field])
        for field in ("full_proposal_sha256", "manifest_sha256")
    ):
        return False
    integer_fields = (
        "approved_shards",
        "total_shards",
        "remaining_shards",
        "review_calls_used",
        "review_call_limit",
    )
    if any(
        not isinstance(value.get(field), int) or isinstance(value.get(field), bool)
        for field in integer_fields
    ):
        return False
    approved = value["approved_shards"]
    total = value["total_shards"]
    return (
        0 <= approved < total
        and value["remaining_shards"] == total - approved
        and 0 < value["review_calls_used"] <= value["review_call_limit"]
    )


@contextmanager
def _cross_process_ingest_lease():
    """Fail closed when another process already owns the ingest batch."""

    global _INGEST_PROCESS_LEASE_ACTIVE

    lock_path = CHRONOVISOR_ROOT / "runtime" / "ingest-orchestrator.lock"
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = lock_path.open("a+")
    except OSError as exc:
        yield False, f"ingest process lock unavailable: {type(exc).__name__}: {exc}"
        return
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False, "another ingest process holds the cross-process lease"
            return
        _INGEST_PROCESS_LEASE_ACTIVE = True
        try:
            yield True, None
        finally:
            _INGEST_PROCESS_LEASE_ACTIVE = False
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()


def ingest_process_lease_is_held(owner_pid: object) -> bool:
    """Return whether an ingest call still owns the OS process lease.

    PID liveness alone is insufficient for MCP processes because the process
    can outlive a failed ingest call.  The flock is the authoritative batch
    lifetime signal.  The in-process flag avoids probing our own flock through
    another descriptor, whose semantics vary across supported POSIX systems.
    """

    if owner_pid == os.getpid():
        return _INGEST_PROCESS_LEASE_ACTIVE
    lock_path = CHRONOVISOR_ROOT / "runtime" / "ingest-orchestrator.lock"
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = lock_path.open("a+")
    except OSError:
        return False
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        except OSError:
            return False
        try:
            return False
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()


def _serialize_ingest_across_processes(function):
    """Keep one batch owner across MCP, dashboard, and launchd processes."""

    @wraps(function)
    def wrapped(*args, **kwargs):
        with _cross_process_ingest_lease() as (acquired, reason):
            if not acquired:
                return {"triggered": False, "reason": reason}
            return function(*args, **kwargs)

    return wrapped


def _load_state() -> dict:
    """Load orchestrator state."""
    if STATE_FILE.exists():
        state = json.loads(STATE_FILE.read_text())
        # Legacy releases counted triage failures across unrelated raws and
        # used that batch-global counter to dead-letter the next batch.  The
        # per-raw failure supervisor now owns retry and quarantine decisions,
        # so discard the unsafe legacy field as soon as state is loaded.
        state.pop("triage_failure_count", None)
        return state
    return {
        "last_ingest": None,
        "last_lint": None,
        "processed_raw_files": [],
        "ollama_health": {"status": None, "checked_at": None},
        "current_job_id": None,
        "current_job_pid": None,
        "current_job_started_at": None,
    }


STALE_LOCK_MAX_AGE_SECONDS = 12 * 60 * 60


def _parse_iso_datetime(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _pid_is_alive(pid: object) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _lock_is_fresh_in_live_process(state: dict) -> bool:
    started_at = _parse_iso_datetime(state.get("current_job_started_at"))
    if started_at is None:
        return False
    age = (datetime.now() - started_at).total_seconds()
    if age > STALE_LOCK_MAX_AGE_SECONDS:
        return False
    pid = state.get("current_job_pid")
    return _pid_is_alive(pid) and ingest_process_lease_is_held(pid)


def _clear_current_job(state: dict) -> None:
    state["current_job_id"] = None
    state["current_job_pid"] = None
    state["current_job_started_at"] = None


def _publish_ingest_reservation(state: dict) -> None:
    """Publish the slot and clear an uncertain post-commit write failure.

    ``atomic_write`` can durably replace the state file and then report a
    directory-fsync failure.  Treat that as an uncertain commit: read the
    state back and remove only our own pending reservation before propagating
    the exception.  A later call also repairs any residue after obtaining the
    authoritative process lease.
    """

    try:
        _save_state(state)
    except BaseException:
        try:
            observed = _load_state()
            if (
                observed.get("current_job_id") == "__pending__"
                and observed.get("current_job_pid") == os.getpid()
            ):
                _clear_current_job(observed)
                _save_state(observed)
        except Exception:
            pass
        raise


def reset_stale_lock() -> None:
    """Clear ``current_job_id`` on startup if the referenced job is gone.

    ``job_store`` is in-memory: after a process restart, any job_id persisted
    in the state file refers to a job that no longer exists, so the
    orchestrator would refuse to ever trigger ingest again. Call this once
    at server startup so a crash mid-ingest doesn't permanently brick the
    queue. The reservation sentinel ``__pending__`` is also cleared.
    """
    state = _load_state()
    cur = state.get("current_job_id")
    if not cur:
        return
    if _lock_is_fresh_in_live_process(state):
        return
    if cur == "__pending__":
        _clear_current_job(state)
        _save_state(state)
        return
    from chronovisor.core.jobs import job_store  # local import to avoid cycle

    if job_store.get(cur) is None:
        _clear_current_job(state)
        _save_state(state)


def _save_state(state: dict) -> None:
    """Save orchestrator state."""
    from chronovisor.core.link_fix import atomic_write

    state.pop("triage_failure_count", None)
    atomic_write(STATE_FILE, json.dumps(state, indent=2, ensure_ascii=False) + "\n")


def get_pending_raw_files() -> list[Path]:
    """Get active raw files that haven't been processed yet.

    A raw with ``raw_status: retracted`` remains on disk as audit evidence,
    but is never offered to normal ingest.
    """
    from chronovisor.ingest.failure_supervisor import operational_deferred_raw_files
    from chronovisor.ingest.raw_replay import is_raw_retracted

    state = _load_state()
    processed = set(state.get("processed_raw_files", []))
    from chronovisor.core.raw_store import RawStore

    raw_store = RawStore(RAW_DIR)
    reference_dir = RAW_DIR.parent / "runtime" / "raw-projections" / "parents"
    raw_paths = sorted(
        (
            raw_store.materialize_ingest(unit, reference_dir)
            if unit.storage != "legacy_file"
            else unit.path
        )
        for unit in raw_store.iter_units()
        if unit.raw_id not in processed
    )
    artifact_dir = RAW_DIR.parent / "runtime" / "raw-projections" / "artifacts"
    if artifact_dir.exists():
        raw_paths.extend(sorted(artifact_dir.glob("*.md")))
        raw_paths = sorted(dict.fromkeys(raw_paths), key=lambda path: path.name)
    operational_deferred = operational_deferred_raw_files(raw_paths)
    pending = []
    for f in raw_paths:
        if (
            f.name not in processed
            and f.name not in operational_deferred
            and not is_raw_retracted(f)
        ):
            pending.append(f)
    return pending


def should_ingest() -> tuple[bool, str]:
    """Check if ingest should be triggered. Returns (should_run, reason)."""
    pending = get_pending_raw_files()
    if len(pending) >= INGEST_THRESHOLD:
        return True, f"{len(pending)} pending raw files (threshold: {INGEST_THRESHOLD})"
    return False, f"Only {len(pending)} pending (threshold: {INGEST_THRESHOLD})"


def should_lint() -> tuple[bool, str]:
    """Check if lint should be triggered. Returns (should_run, reason)."""
    state = _load_state()
    last_lint = state.get("last_lint")

    if last_lint is None:
        return True, "Lint has never been run"

    last_lint_dt = datetime.fromisoformat(last_lint)
    hours_since = (datetime.now() - last_lint_dt).total_seconds() / 3600

    if hours_since >= LINT_INTERVAL_HOURS:
        return (
            True,
            f"{hours_since:.1f} hours since last lint (threshold: {LINT_INTERVAL_HOURS}h)",
        )
    return (
        False,
        f"Only {hours_since:.1f}h since last lint (threshold: {LINT_INTERVAL_HOURS}h)",
    )


def mark_raw_processed(filenames: list[str]) -> None:
    """Mark raw files as processed.

    Public, batch-level API used by legacy full-batch ingest paths. Updates
    the processed set, refreshes ``last_ingest``, and clears
    ``current_job_id``.
    The per-raw synchronous ingest loop in :func:`run_pending_ingest`
    does NOT call this — it uses :func:`_mark_one_raw_processed` so a
    single raw's success doesn't release the batch-wide in-flight slot.
    """
    state = _load_state()
    processed = set(state.get("processed_raw_files", []))
    processed.update(filenames)
    state["processed_raw_files"] = sorted(processed)
    state["last_ingest"] = datetime.now().isoformat()
    _clear_current_job(state)
    _save_state(state)


def _mark_one_raw_processed(filename: str) -> None:
    """Per-raw success mark — does NOT touch ``current_job_id``.

    Used by :func:`run_pending_ingest`'s synchronous serial loop. Each
    raw's ``on_complete`` callback marks just that file processed, but leaves
    ``current_job_id`` intact so the batch-wide in-flight marker survives
    until the loop's outer ``finally`` clears it.
    """
    state = _load_state()
    processed = set(state.get("processed_raw_files", []))
    processed.add(filename)
    state["processed_raw_files"] = sorted(processed)
    state["last_ingest"] = datetime.now().isoformat()
    _save_state(state)


def _mark_raws_processed_preserving_lock(filenames: list[str]) -> None:
    """Mark one ingest unit's source files without releasing the batch lock."""
    state = _load_state()
    processed = set(state.get("processed_raw_files", []))
    processed.update(filenames)
    state["processed_raw_files"] = sorted(processed)
    state["last_ingest"] = datetime.now().isoformat()
    _save_state(state)


def _raws_are_durably_processed(filenames: list[str]) -> bool:
    """Confirm a processed mark after an ambiguous post-commit I/O error."""

    try:
        state = _load_state()
    except Exception:
        return False
    processed = state.get("processed_raw_files")
    return isinstance(processed, list) and set(filenames) <= set(processed)


def _orch_log(message: str) -> None:
    """Best-effort activity append; a wedged journal cannot break the loop.
    """
    try:
        activity_log.append_activity(
            message,
            source="orchestrator",
            level=runtime_status.classify_log_message(message),
            root=CHRONOVISOR_ROOT,
            path=ACTIVITY_FILE,
        )
    except Exception:
        pass
    runtime_status.safe_append_event(
        runtime_status.classify_log_message(message),
        message,
        source="orchestrator",
    )


def mark_lint_complete() -> None:
    """Mark lint as completed."""
    state = _load_state()
    state["last_lint"] = datetime.now().isoformat()
    _save_state(state)


def get_ollama_status() -> dict:
    """Report the configured ingest generation route without probing a backend."""
    from chronovisor.ingest import ingest as ingest_runtime

    try:
        route = ingest_runtime.ollama_runtime.runtime_generation_routes(
            (ingest_runtime.ollama_runtime.INGEST_GENERATION_RUNTIME_ROLE,)
        )[0]
    except ingest_runtime.ollama_runtime.RuntimeBridgeError:
        return {
            "available": False,
            "processor": "unavailable",
            "role": ingest_runtime.ollama_runtime.INGEST_GENERATION_RUNTIME_ROLE,
            "provider": None,
            "model": None,
            "location": None,
        }
    return {
        "available": True,
        "processor": route.provider,
        "role": route.role,
        "provider": route.provider,
        "model": route.model,
        "location": route.location,
    }


def ingest_authority_preflight(
    *,
    frontier_reviewer: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Prove the batch-wide semantic authority before claiming any Raw.

    Authority configuration is global to the batch. Treating an invalid
    adoption artifact as a per-Raw failure creates a false failure storm and
    can quarantine every otherwise valid Raw behind the same control-plane
    incident. This preflight keeps that global outage outside the per-Raw
    failure supervisor.
    """

    from chronovisor.ingest.ingest_review_authority import (
        current_ingest_review_authority,
        ingest_review_authority_shape_error,
    )

    authority, authority_error = current_ingest_review_authority(
        injected_reviewer=frontier_reviewer is not None
    )
    shape_error = (
        ingest_review_authority_shape_error(authority)
        if authority is not None
        else None
    )
    problem = (
        authority_error
        or shape_error
        or (None if authority is not None else "authority is missing")
    )
    if problem is not None:
        return {
            "ok": False,
            "status": "blocked",
            "blocked_by": "decision_authority",
            "retryable": True,
            "error": "local consensus authority unavailable: " + problem,
            "artifact_sha256": None,
        }
    router = authority.get("router") if isinstance(authority, dict) else None
    return {
        "ok": True,
        "status": "ready",
        "blocked_by": None,
        "retryable": False,
        "error": None,
        "artifact_sha256": (
            router.get("artifact_sha256") if isinstance(router, dict) else None
        ),
    }


@dataclass(frozen=True)
class _PendingRawUnit:
    paths: tuple[Path, ...]
    content: str | None = None
    raw_keywords: tuple[str, ...] | None = None
    fragment_record_sha256: str | None = None
    reassembled_record_bytes: bytes | None = None
    native_raw_bytes: bytes | None = None
    native_commit: object | None = None
    logical_raw_bytes: bytes | None = None

    @property
    def representative(self) -> Path:
        return self.paths[0]

    @property
    def filenames(self) -> list[str]:
        return [path.name for path in self.paths]


def _quarantine_capture_fragment_paths(
    paths: list[Path],
    *,
    record_sha256: str,
    reason: str,
    details: dict | None = None,
) -> dict:
    """Preserve unsafe fragments with a durable intent-first transaction."""

    dead_letter_dir = RAW_DIR / ".dead-letter" / "raw-capture-fragments" / record_sha256
    dead_letter_dir.mkdir(parents=True, exist_ok=True)
    source_paths = tuple(
        sorted(
            dict.fromkeys(path for path in paths if path.exists()),
            key=lambda path: path.name,
        )
    )
    planned: list[dict[str, object]] = []
    reserved_destinations: set[str] = set()
    for src in source_paths:
        source_bytes = src.read_bytes()
        dst = dead_letter_dir / src.name
        duplicate = 1
        while dst.exists() or dst.name in reserved_destinations:
            dst = dead_letter_dir / f"{src.name}.duplicate-{duplicate}"
            duplicate += 1
        reserved_destinations.add(dst.name)
        planned.append(
            {
                "source": src.name,
                "preserved_as": dst.name,
                "file_bytes": len(source_bytes),
                "file_sha256": hashlib.sha256(source_bytes).hexdigest(),
            }
        )

    manifest = {
        "schema_version": 1,
        "kind": "raw_capture_fragment_quarantine",
        "status": "prepared",
        "created_at": datetime.now().isoformat(),
        "record_sha256": record_sha256,
        "reason": reason,
        "details": details or {},
        "files": planned,
    }
    manifest_path = dead_letter_dir / "manifest.json"
    if manifest_path.exists():
        existing = _read_fragment_quarantine_manifest(manifest_path)
        if existing.get("status") == "prepared":
            return _resume_capture_fragment_quarantine(manifest_path)
        operation_id = hashlib.sha256(
            json.dumps(
                {
                    "record_sha256": record_sha256,
                    "reason": reason,
                    "files": planned,
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()[:16]
        manifest_path = dead_letter_dir / f"manifest-{operation_id}.json"
        if manifest_path.exists():
            existing = _read_fragment_quarantine_manifest(manifest_path)
            if existing.get("status") == "prepared":
                return _resume_capture_fragment_quarantine(manifest_path)
            return _fragment_quarantine_result(manifest_path, existing)

    # The atomic, fsynced manifest is the transaction intent. No source move
    # may happen before this publication succeeds.
    atomic_write(
        manifest_path,
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return _resume_capture_fragment_quarantine(manifest_path)


def _read_fragment_quarantine_manifest(manifest_path: Path) -> dict:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != 1
        or payload.get("kind") != "raw_capture_fragment_quarantine"
        or not isinstance(payload.get("record_sha256"), str)
        or not isinstance(payload.get("reason"), str)
        or not isinstance(payload.get("files"), list)
    ):
        raise RuntimeError(
            f"invalid fragment quarantine manifest: {manifest_path.name}"
        )
    # Legacy v1 manifests had no status and were already complete.
    if "status" not in payload:
        payload["status"] = "completed"
    if payload.get("status") not in {"prepared", "completed"}:
        raise RuntimeError(f"invalid fragment quarantine status: {manifest_path.name}")
    return payload


def _fragment_quarantine_result(manifest_path: Path, manifest: dict) -> dict:
    files = manifest.get("files") or []
    filenames = [
        row.get("source")
        for row in files
        if isinstance(row, dict) and isinstance(row.get("source"), str)
    ]
    return {
        "record_sha256": manifest["record_sha256"],
        "reason": manifest["reason"],
        "files": filenames,
        "manifest": str(manifest_path),
        "status": manifest.get("status", "completed"),
    }


def _resume_capture_fragment_quarantine(manifest_path: Path) -> dict:
    """Complete one prepared move transaction idempotently."""

    manifest = _read_fragment_quarantine_manifest(manifest_path)
    if manifest["status"] == "completed":
        return _fragment_quarantine_result(manifest_path, manifest)
    dead_letter_dir = manifest_path.parent
    filenames: list[str] = []
    for row in manifest["files"]:
        if not isinstance(row, dict):
            raise RuntimeError("fragment quarantine file intent is malformed")
        source = row.get("source")
        preserved_as = row.get("preserved_as")
        expected_bytes = row.get("file_bytes")
        expected_sha256 = row.get("file_sha256")
        if (
            not isinstance(source, str)
            or Path(source).name != source
            or not isinstance(preserved_as, str)
            or Path(preserved_as).name != preserved_as
            or isinstance(expected_bytes, bool)
            or not isinstance(expected_bytes, int)
            or not isinstance(expected_sha256, str)
        ):
            raise RuntimeError("fragment quarantine file intent is malformed")
        src = RAW_DIR / source
        dst = dead_letter_dir / preserved_as
        if src.exists() and dst.exists():
            raise RuntimeError(
                f"fragment quarantine source and destination both exist: {source}"
            )
        if src.exists():
            source_bytes = src.read_bytes()
            if (
                len(source_bytes) != expected_bytes
                or hashlib.sha256(source_bytes).hexdigest() != expected_sha256
            ):
                raise RuntimeError(
                    f"fragment quarantine source changed after intent: {source}"
                )
            src.rename(dst)
            _fsync_directory(RAW_DIR)
            _fsync_directory(dead_letter_dir)
        elif not dst.exists():
            raise RuntimeError(
                f"fragment quarantine evidence missing at both paths: {source}"
            )
        destination_bytes = dst.read_bytes()
        if (
            len(destination_bytes) != expected_bytes
            or hashlib.sha256(destination_bytes).hexdigest() != expected_sha256
        ):
            raise RuntimeError(
                f"fragment quarantine destination verification failed: {preserved_as}"
            )
        filenames.append(source)

    _mark_raws_processed_preserving_lock(filenames)
    completed = dict(manifest)
    completed["status"] = "completed"
    completed["completed_at"] = datetime.now().isoformat()
    atomic_write(
        manifest_path,
        json.dumps(completed, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return _fragment_quarantine_result(manifest_path, completed)


def _resume_prepared_capture_fragment_quarantines() -> list[dict]:
    root = RAW_DIR / ".dead-letter" / "raw-capture-fragments"
    if not root.is_dir():
        return []
    resumed: list[dict] = []
    for manifest_path in sorted(root.glob("*/manifest*.json")):
        manifest = _read_fragment_quarantine_manifest(manifest_path)
        if manifest["status"] == "prepared":
            resumed.append(_resume_capture_fragment_quarantine(manifest_path))
    return resumed


def _prepare_pending_raw_units(
    pending: list[Path],
) -> tuple[list[_PendingRawUnit], list[dict], list[dict]]:
    """Separate normal raws from validated, complete fragment transport sets."""

    from chronovisor.ingest.raw_capture_fragments import (
        RawCaptureFragmentError,
        group_capture_fragments,
        parse_capture_fragment,
    )

    fragments = []
    fragment_paths: set[Path] = set()
    quarantined: list[dict] = []
    deferred: list[dict] = []
    for path in pending:
        try:
            fragment = parse_capture_fragment(path)
        except (OSError, UnicodeError, RawCaptureFragmentError) as exc:
            fragment_paths.add(path)
            quarantined.append(
                _quarantine_capture_fragment_paths(
                    [path],
                    record_sha256=f"malformed-{path.stem}",
                    reason="fragment_parse_error",
                    details={"error": f"{type(exc).__name__}: {exc}"},
                )
            )
            continue
        if fragment is not None:
            fragments.append(fragment)
            fragment_paths.add(path)

    from chronovisor.core.raw_store import RawStore

    raw_store = RawStore(RAW_DIR)
    units: list[_PendingRawUnit] = []
    for path in pending:
        if path in fragment_paths:
            continue
        native = raw_store.resolve_reference(path)
        if native is None:
            units.append(_PendingRawUnit(paths=(path,)))
            continue
        if native.commit is None:
            value = raw_store.read_bytes(native)
            units.append(
                _PendingRawUnit(
                    paths=(path,),
                    content=value.decode("utf-8"),
                    logical_raw_bytes=value,
                )
            )
            continue
        host_label = "Claude Code" if native.commit.host == "claude-code" else "Codex"
        units.append(
            _PendingRawUnit(
                paths=(path,),
                raw_keywords=(host_label, "transcript-delta", "source-native"),
                native_raw_bytes=raw_store.read_bytes(native),
                native_commit=native.commit,
            )
        )
    grouped_fragments: dict[object, list] = {}
    for fragment in fragments:
        grouped_fragments.setdefault(fragment.identity, []).append(fragment)
    groups = []
    for rows in sorted(
        grouped_fragments.values(),
        key=lambda items: min(item.path.name for item in items),
    ):
        try:
            groups.extend(group_capture_fragments(rows))
        except RawCaptureFragmentError as exc:
            # Duplicate indices invalidate only their own source-record group.
            # Never let one malformed transport identity mutate unrelated raws.
            paths = sorted((row.path for row in rows), key=lambda path: path.name)
            quarantined.append(
                _quarantine_capture_fragment_paths(
                    paths,
                    record_sha256=rows[0].identity.record_sha256,
                    reason="fragment_group_invalid",
                    details={"error": str(exc)},
                )
            )

    for group in groups:
        if not group.complete:
            deferred.append(
                {
                    "record_sha256": group.identity.record_sha256,
                    "reason": "fragment_group_incomplete",
                    "missing_indices": list(group.missing_indices),
                    "files": [path.name for path in group.paths],
                }
            )
            continue
        try:
            record_bytes = group.assemble_bytes()
            ingest_content = group.ingest_content()
        except RawCaptureFragmentError as exc:
            quarantined.append(
                _quarantine_capture_fragment_paths(
                    list(group.paths),
                    record_sha256=group.identity.record_sha256,
                    reason="fragment_integrity_failure",
                    details={"error": str(exc)},
                )
            )
            continue
        host_label = "Claude Code" if group.identity.host == "claude-code" else "Codex"
        units.append(
            _PendingRawUnit(
                paths=tuple(sorted(group.paths, key=lambda path: path.name)),
                content=ingest_content,
                raw_keywords=(
                    host_label,
                    "transcript-delta",
                    "transcript-reassembled",
                ),
                fragment_record_sha256=group.identity.record_sha256,
                reassembled_record_bytes=record_bytes,
            )
        )
    units.sort(key=lambda unit: unit.representative.name)
    return units, quarantined, deferred


def _raw_unit_keywords(
    raw_keywords: tuple[str, ...] | list[str] | None,
    raw_text: str,
) -> list[str]:
    """Resolve current then legacy frontmatter keywords for one semantic unit."""

    if raw_keywords is not None:
        return list(raw_keywords)
    from chronovisor.core.legacy_frontmatter import parse as _frontmatter_parse

    meta, _body = _frontmatter_parse(raw_text)
    current = _coerce_str_list(meta.get("raw_keywords"))
    if current is not None:
        return current
    return _coerce_str_list(meta.get("keywords")) or []


def _raw_unit_event(
    *, succeeded: bool, deferred: bool, continued: bool
) -> tuple[str, str]:
    """Classify one unit's durable outcome for the operator event stream."""

    if succeeded:
        return "success", "processed"
    if continued:
        return "info", "shard review continuation pending"
    if deferred:
        return "info", "semantic deferred"
    return "warn", "not processed"


def _projection_result_summary(projection: Any) -> dict[str, Any]:
    """Return the durable, body-free projection summary exposed to operators."""

    return {
        "kind": projection.kind,
        "manifest_path": (
            str(projection.manifest_path)
            if projection.manifest_path is not None
            else None
        ),
        "projection_paths": [str(path) for path in projection.projection_paths],
        "child_paths": [str(path) for path in projection.child_paths],
        "noop_receipt_path": (
            str(projection.noop_receipt_path)
            if projection.noop_receipt_path is not None
            else None
        ),
        "parent_sha256": projection.parent_sha256,
        "projection_sha256": projection.projection_sha256,
        "record_count": projection.record_count,
        "selected_record_count": projection.selected_record_count,
        "child_count": projection.child_count,
        "role_counts": dict(projection.role_counts),
    }


def _raw_unit_result(
    *,
    filename: str,
    source_files: list[str],
    job_id: str,
    succeeded: bool,
    deferred: bool,
    continued: bool,
    continuation: object,
    fragment_record_sha256: str | None,
    projection: dict[str, Any] | None,
    supervision: dict[str, Any] | None,
    completion_ack: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build one raw unit's stable public batch-result row."""

    result: dict[str, Any] = {
        "filename": filename,
        "source_files": source_files,
        "job_id": job_id,
        "succeeded": succeeded,
        "deferred": deferred,
        "continued": continued,
    }
    if continued:
        result["continuation"] = continuation
    if fragment_record_sha256 is not None:
        result["reassembled_fragment_record_sha256"] = fragment_record_sha256
    if projection is not None:
        result["projection"] = projection
    if supervision is not None:
        result["supervision"] = supervision
    if completion_ack is not None:
        result["completion_ack"] = completion_ack
    return result


def _ingest_batch_result(
    *,
    reason: str,
    job_ids: list[str],
    filenames: list[str],
    succeeded_filenames: list[str],
    deferred_filenames: list[str],
    continued_filenames: list[str],
    failed_filenames: int,
    fragment_quarantined: list[dict],
    fragment_deferred: list[dict],
    resumed_fragment_quarantines: list[dict],
    per_raw: list[dict],
    processor: str,
    elapsed: float,
) -> dict[str, Any]:
    """Build the stable terminal envelope for a triggered ingest batch."""

    return {
        "triggered": True,
        "reason": reason,
        "job_ids": job_ids,
        "files_attempted": filenames,
        "files_processed": succeeded_filenames,
        "files_deferred": deferred_filenames,
        "files_continued": continued_filenames,
        "files_failed": failed_filenames,
        "files_quarantined": [
            name for row in fragment_quarantined for name in row.get("files", [])
        ],
        "fragment_quarantined": fragment_quarantined,
        "fragment_deferred": fragment_deferred,
        "fragment_quarantine_transactions_resumed": resumed_fragment_quarantines,
        "per_raw": per_raw,
        "processor": processor,
        "elapsed_seconds": round(elapsed, 2),
    }


def _post_ingest_lint_summary() -> dict[str, Any]:
    """Run the read-only wiki check without risking a durable ingest result."""

    try:
        from chronovisor.ingest.lint import check, summarize_issues

        return {"status": "ok", "summary": summarize_issues(check())}
    except Exception as exc:
        error_category = type(exc).__name__
        _orch_log(f"orchestrator | post-ingest lint failed: {error_category}")
        return {"status": "error", "error_category": error_category}


@_serialize_ingest_across_processes
def run_pending_ingest(
    force: bool = False,
    *,
    max_units: int = MAX_INGEST_BATCH_UNITS,
    frontier_reviewer: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> dict:
    """Run ingest on all pending raw files if threshold is met.
    Per-raw synchronous serial execution. Each pending raw is parsed for
    ``raw_keywords`` (with legacy ``keywords`` fallback), then ingested
    individually via :func:`run_ingest` with that metadata propagated as
    a side channel for downstream apply. Successful raws are marked
    individually so a single raw's failure doesn't block retry of the
    others — and so partial-batch progress survives a server crash.

    The lock and durable state slot serialize each batch until outer cleanup.

    Args:
        force: When True, bypass the ``INGEST_THRESHOLD`` check and trigger
            immediately as long as at least one raw is pending. Used by
            ``chronovisor_ingest`` to preserve its historical "ingest now" contract.
        max_units: Maximum semantic work units to process in this batch. The
            default preserves the historical batch size of 10; smaller values
            support controlled pilots without weakening process serialization.
        frontier_reviewer: Explicit test/evaluation reviewer injection. The
            production path leaves this unset and resolves adopted authority.

    Returns result dict with status and details. On a triggered batch the
    result includes per-raw entries (filename, job_id, succeeded) so
    callers can inspect partial outcomes without re-loading job state.
    """
    with _INGEST_LOCK:
        if not isinstance(max_units, int) or isinstance(max_units, bool):
            raise ValueError("max_units must be an integer between 1 and 10")
        if not 1 <= max_units <= MAX_INGEST_BATCH_UNITS:
            raise ValueError(
                f"max_units must be between 1 and {MAX_INGEST_BATCH_UNITS}"
            )
        try:
            resumed_fragment_quarantines = (
                _resume_prepared_capture_fragment_quarantines()
            )
        except Exception as exc:
            return {
                "triggered": False,
                "reason": (
                    "prepared capture fragment quarantine could not resume: "
                    f"{type(exc).__name__}: {exc}"
                ),
            }
        if force:
            pending_now = get_pending_raw_files()
            if not pending_now:
                runtime_status.safe_write_status(
                    state="idle",
                    stage="idle",
                    pending=0,
                    current_raw=None,
                    current_op=None,
                    batch=None,
                    current_job_id=None,
                    current_job_pid=None,
                    ollama=get_ollama_status(),
                    llm=None,
                )
                return {"triggered": False, "reason": "no pending raws"}
            reason = f"force=True with {len(pending_now)} pending"
        else:
            should, reason = should_ingest()
            if not should:
                runtime_status.safe_write_status(
                    state="idle",
                    stage="waiting",
                    pending=len(get_pending_raw_files()),
                    current_raw=None,
                    current_op=None,
                    batch=None,
                    current_job_id=None,
                    current_job_pid=None,
                    ollama=get_ollama_status(),
                    llm=None,
                )
                return {"triggered": False, "reason": reason}

        state = _load_state()
        if state.get("current_job_id"):
            # This function already owns the authoritative cross-process
            # flock.  Therefore no batch represented only by durable state can
            # still be active; it is residue from an interrupted prior call
            # (including the same long-lived MCP PID).  Repair it before
            # reserving the new batch instead of permanently rejecting work.
            stale_job_id = state.get("current_job_id")
            _clear_current_job(state)
            _save_state(state)
            runtime_status.safe_append_event(
                "warn",
                "orchestrator | cleared stranded ingest reservation",
                source="orchestrator",
                stale_job_id=stale_job_id,
            )

        pending = get_pending_raw_files()
        pending_before_count = len(pending)
        units, fragment_quarantined, fragment_deferred = _prepare_pending_raw_units(
            pending
        )

        # Limit semantic work units, not transport fragments. A complete
        # fragment set is reconstructed and presented to ingest exactly once.
        units = units[:max_units]
        filenames = [name for unit in units for name in unit.filenames]

        if not units:
            pending_after_preflight = len(get_pending_raw_files())
            runtime_status.safe_write_status(
                state="idle",
                stage="waiting",
                pending=pending_after_preflight,
                current_raw=None,
                current_op=None,
                current_job_id=None,
                current_job_pid=None,
                ollama=get_ollama_status(),
                llm=None,
            )
            if fragment_quarantined:
                return {
                    "triggered": True,
                    "reason": "capture fragments quarantined before semantic ingest",
                    "job_ids": [],
                    "files_attempted": [],
                    "files_processed": [],
                    "files_quarantined": [
                        name
                        for row in fragment_quarantined
                        for name in row.get("files", [])
                    ],
                    "fragment_quarantined": fragment_quarantined,
                    "fragment_deferred": fragment_deferred,
                    "processor": get_ollama_status()["processor"],
                    "elapsed_seconds": 0.0,
                }
            return {
                "triggered": False,
                "reason": "capture fragment groups are incomplete",
                "fragment_quarantined": [],
                "fragment_deferred": fragment_deferred,
            }

        # Lazy imports keep module-level cycles minimal and isolate test
        # patches that swap these out.  Resolve every fallible prerequisite
        # before publishing the durable in-flight reservation; otherwise a
        # long-lived MCP process can leave a false-live ``__pending__`` slot.
        from chronovisor.core.jobs import JobStatus, job_store
        from chronovisor.core.raw_store import raw_layout_mode
        from chronovisor.core.runtime_config import load_ingest_config
        from chronovisor.ingest.ingest import run_ingest
        from chronovisor.ingest.raw_completion_ack import (
            RawCompletionAckError,
            RawCompletionStatePending,
            load_valid_receipt,
            publish_receipt,
            receipt_path,
            receipt_summary,
        )
        from chronovisor.ingest.raw_semantic_projection import (
            ProjectionCapacityError,
            ProjectionConflictError,
            RawSemanticProjectionError,
            UnsupportedNativeTranscriptHostError,
            project_native_transcript,
            project_parent_raw,
            project_reassembled_raws,
            verify_projection_bundle,
        )

        initial_ollama_status = get_ollama_status()
        authority_preflight = ingest_authority_preflight(
            frontier_reviewer=frontier_reviewer
        )
        if not authority_preflight["ok"]:
            runtime_status.safe_write_status(
                state="blocked",
                stage="decision-authority",
                pending=pending_before_count,
                current_raw=None,
                current_op=None,
                current_job_id=None,
                current_job_pid=None,
                ollama=initial_ollama_status,
                llm=None,
                authority_preflight=authority_preflight,
            )
            return {
                "triggered": False,
                "reason": authority_preflight["error"],
                "failure_class": (
                    "ingest.runtime_local_consensus_authority_unavailable"
                ),
                "blocked_by": "decision_authority",
                "retryable": True,
                "pending_before": pending_before_count,
                "pending_after": pending_before_count,
                "files_attempted": [],
                "files_processed": [],
                "fragment_quarantined": fragment_quarantined,
                "fragment_deferred": fragment_deferred,
                "authority_preflight": authority_preflight,
            }
        per_raw: list[dict] = []
        job_ids: list[str] = []
        succeeded_filenames: list[str] = []
        deferred_filenames: list[str] = []
        continued_filenames: list[str] = []
        batch_started = time.time()

        # Reserve the batch-wide in-flight slot only after all prerequisites
        # succeeded.  From this point the loop's outer ``finally`` owns release.
        reserved_state = _load_state()
        reserved_state["current_job_id"] = "__pending__"
        reserved_state["current_job_pid"] = os.getpid()
        reserved_state["current_job_started_at"] = datetime.now().isoformat()
        _publish_ingest_reservation(reserved_state)

        runtime_status.safe_write_status(
            state="running",
            stage="batch",
            pending=pending_before_count,
            current_raw=None,
            current_op=None,
            current_job_id="__pending__",
            current_job_pid=os.getpid(),
            batch={
                "started_at": datetime.now().isoformat(),
                "index": 0,
                "total": len(filenames),
                "succeeded": 0,
                "deferred": 0,
                "continued": 0,
                "failed": 0,
                "files": filenames,
            },
            ollama=initial_ollama_status,
            llm=None,
        )

        succeeded_units = 0
        deferred_units = 0
        continued_units = 0
        try:
            for raw_index, unit in enumerate(units, start=1):
                raw_path = unit.representative
                fname = raw_path.name
                source_filenames = unit.filenames
                try:
                    raw_text = (
                        unit.content
                        if unit.content is not None
                        else raw_path.read_text()
                    )
                except Exception as e:
                    _orch_log(f"orchestrator | failed to read raw {fname}: {e}")
                    per_raw.append(
                        {
                            "filename": fname,
                            "source_files": source_filenames,
                            "succeeded": False,
                            "error": f"read error: {e}",
                        }
                    )
                    continue

                # Extract raw_keywords from frontmatter, falling back to the
                # legacy ``keywords`` field for raws written before Phase 1.
                # Anything that isn't a list of strings is normalized to [].
                raw_keywords = _raw_unit_keywords(unit.raw_keywords, raw_text)

                processor = get_ollama_status()["processor"]
                job = job_store.create(processor=processor)
                job_ids.append(job.job_id)
                runtime_status.safe_write_status(
                    state="running",
                    stage="raw",
                    pending=len(get_pending_raw_files()),
                    current_raw=fname,
                    current_op=None,
                    current_job_id=job.job_id,
                    current_job_pid=os.getpid(),
                    batch={
                        "started_at": datetime.fromtimestamp(batch_started).isoformat(),
                        "index": raw_index,
                        "total": len(units),
                        "succeeded": succeeded_units,
                        "deferred": deferred_units,
                        "continued": continued_units,
                        "failed": (
                            len(per_raw)
                            - succeeded_units
                            - deferred_units
                            - continued_units
                        ),
                        "files": filenames,
                    },
                    ollama=get_ollama_status(),
                    llm=None,
                )

                # Make the per-raw job_id observable to other processes
                # while it runs. The outer ``finally`` will clear it.
                visible_state = _load_state()
                visible_state["current_job_id"] = job.job_id
                visible_state["current_job_pid"] = os.getpid()
                if not visible_state.get("current_job_started_at"):
                    visible_state["current_job_started_at"] = datetime.now().isoformat()
                _save_state(visible_state)

                # The flag is flipped only after the processed-state write is
                # durable.  A published receipt with an interrupted state write
                # therefore remains a failed job and can resume ACK-only on the
                # next tick without repeating semantic work.
                raw_success_flag = [False]
                completion_ack_summary: list[dict | None] = [None]

                def _on_complete(
                    names=source_filenames,
                    paths=unit.paths,
                    flag=raw_success_flag,
                    summary_holder=completion_ack_summary,
                    current_job_id=job.job_id,
                ):
                    completed_job = job_store.get(current_job_id)
                    # on_complete is itself the terminal success contract.  The
                    # real ingest path has already persisted COMPLETED; keeping
                    # this normalization also supports deterministic callback
                    # implementations used by embedders and tests.
                    if completed_job is not None and completed_job.status is not (
                        JobStatus.COMPLETED
                    ):
                        job_store.update(
                            current_job_id,
                            status=JobStatus.COMPLETED,
                            completed_at=datetime.now().isoformat(),
                        )
                        completed_job = job_store.get(current_job_id)
                    receipt, payload = publish_receipt(paths, completed_job)
                    try:
                        _mark_raws_processed_preserving_lock(list(names))
                    except Exception as exc:
                        if not _raws_are_durably_processed(list(names)):
                            raise RawCompletionStatePending(
                                "raw completion ACK state pending: "
                                f"{type(exc).__name__}: {exc}"
                            ) from exc
                    flag[0] = True
                    summary_holder[0] = receipt_summary(receipt, payload)

                projection_summary: dict | None = None
                projection_failed = False
                projection_group_quarantine: dict | None = None
                try:
                    durable_receipt = load_valid_receipt(unit.paths)
                    if durable_receipt is not None:
                        durable_receipt_path = receipt_path(unit.paths)
                        try:
                            _mark_raws_processed_preserving_lock(source_filenames)
                        except Exception as exc:
                            if not _raws_are_durably_processed(source_filenames):
                                raise RawCompletionStatePending(
                                    "raw completion ACK state pending: "
                                    f"{type(exc).__name__}: {exc}"
                                ) from exc
                        raw_success_flag[0] = True
                        resumed_summary = receipt_summary(
                            durable_receipt_path, durable_receipt
                        )
                        resumed_summary["resumed"] = True
                        completion_ack_summary[0] = resumed_summary
                        job_store.update(
                            job.job_id,
                            status=JobStatus.COMPLETED,
                            processor="durable-raw-ack",
                            stage="completion-ack",
                            completed_at=datetime.now().isoformat(),
                            pages_created=[],
                            pages_updated=[],
                            result={"raw_completion_ack": resumed_summary},
                        )
                        raise _CompletionAckResumed

                    max_child_bytes = (
                        load_ingest_config().semantic_projection_max_child_bytes
                    )
                    projection_output_dir = (
                        RAW_DIR
                        if raw_layout_mode(chronovisor_root=RAW_DIR.parent) == "legacy"
                        else RAW_DIR.parent
                        / "runtime"
                        / "raw-projections"
                        / "artifacts"
                    )
                    if unit.native_raw_bytes is not None:
                        from chronovisor.core.raw_segment import RawSegmentCommit

                        if not isinstance(unit.native_commit, RawSegmentCommit):
                            raise RuntimeError("native Raw unit lost its commit")
                        projection = project_native_transcript(
                            raw_path,
                            unit.native_raw_bytes,
                            unit.native_commit,
                            output_dir=projection_output_dir,
                            max_child_bytes=max_child_bytes,
                        )
                    elif unit.reassembled_record_bytes is not None:
                        projection = project_reassembled_raws(
                            unit.paths,
                            unit.reassembled_record_bytes,
                            output_dir=projection_output_dir,
                            max_child_bytes=max_child_bytes,
                        )
                    else:
                        projection = project_parent_raw(
                            raw_path,
                            output_dir=projection_output_dir,
                            max_child_bytes=max_child_bytes,
                            raw_bytes=unit.logical_raw_bytes,
                        )
                    projection_summary = _projection_result_summary(projection)
                    if projection.kind in {"noop", "children"}:
                        if projection.manifest_path is None:
                            raise RuntimeError(
                                "semantic projection delegation has no manifest"
                            )
                        verified_manifest = verify_projection_bundle(
                            projection.manifest_path
                        )
                        expected_status = (
                            "noop" if projection.kind == "noop" else "delegated"
                        )
                        if (
                            verified_manifest.get("status") != expected_status
                            or verified_manifest.get("source_sha256")
                            != projection.parent_sha256
                            or verified_manifest.get("projection_sha256")
                            != projection.projection_sha256
                            or len(verified_manifest.get("children") or [])
                            != projection.child_count
                        ):
                            raise RuntimeError(
                                "semantic projection read-back does not match result"
                            )
                        job_store.update(
                            job.job_id,
                            status=JobStatus.COMPLETED,
                            processor="deterministic-projection",
                            stage="projection",
                            completed_at=datetime.now().isoformat(),
                            pages_created=[],
                            pages_updated=[],
                            result={"projection": projection_summary},
                        )
                        _on_complete()
                    elif projection.kind == "passthrough":
                        if projection.child_paths and (
                            "transcript-semantic-projection" not in raw_keywords
                        ):
                            raw_keywords.append("transcript-semantic-projection")
                    else:
                        raise RuntimeError(
                            f"unknown semantic projection kind: {projection.kind!r}"
                        )
                except _CompletionAckResumed:
                    pass
                except RawCompletionAckError as e:
                    projection_failed = True
                    job_store.update(
                        job.job_id,
                        status=JobStatus.FAILED,
                        stage="completion-ack",
                        completed_at=datetime.now().isoformat(),
                        error=str(e),
                    )
                    _orch_log(f"orchestrator | raw {fname} completion ACK failed: {e}")
                except Exception as e:
                    projection_failed = True
                    if isinstance(e, ProjectionConflictError):
                        projection_failure_cause = "artifact_conflict"
                    elif isinstance(e, ProjectionCapacityError):
                        projection_failure_cause = "capacity"
                    elif isinstance(e, UnsupportedNativeTranscriptHostError):
                        projection_failure_cause = "capability_unavailable"
                    elif isinstance(e, RawSemanticProjectionError):
                        projection_failure_cause = "source_invalid"
                    elif isinstance(e, OSError):
                        projection_failure_cause = "interrupted"
                    else:
                        projection_failure_cause = "internal_error"
                    projection_error = (
                        "raw semantic projection failed "
                        f"[{projection_failure_cause}]: {type(e).__name__}: {e}"
                    )
                    if (
                        projection_failure_cause == "source_invalid"
                        and unit.reassembled_record_bytes is not None
                        and unit.fragment_record_sha256 is not None
                    ):
                        projection_group_quarantine = (
                            _quarantine_capture_fragment_paths(
                                list(unit.paths),
                                record_sha256=unit.fragment_record_sha256,
                                reason="semantic_projection_source_invalid",
                                details={
                                    "error_type": type(e).__name__,
                                    "error_sha256": hashlib.sha256(
                                        str(e).encode("utf-8", errors="replace")
                                    ).hexdigest(),
                                },
                            )
                        )
                    job_store.update(
                        job.job_id,
                        status=JobStatus.FAILED,
                        stage="projection",
                        completed_at=datetime.now().isoformat(),
                        error=projection_error,
                    )
                    _orch_log(f"orchestrator | raw {fname} projection exception: {e}")

                if not projection_failed and not raw_success_flag[0]:
                    try:
                        ingest_kwargs: dict[str, Any] = {
                            "on_complete": _on_complete,
                            "metadata": {
                                "raw_keywords": raw_keywords,
                                "source_raw": fname,
                            },
                        }
                        if frontier_reviewer is not None:
                            ingest_kwargs["frontier_reviewer"] = frontier_reviewer
                        run_ingest(
                            raw_text,
                            job.job_id,
                            **ingest_kwargs,
                        )
                    except Exception as e:
                        # ``run_ingest`` already routes its own exceptions
                        # through job_store.update(FAILED); this catch is a
                        # final guard so one bad raw cannot strand the batch.
                        current_job = job_store.get(job.job_id)
                        if current_job is None or current_job.status is not (
                            JobStatus.FAILED
                        ):
                            job_store.update(
                                job.job_id,
                                status=JobStatus.FAILED,
                                completed_at=datetime.now().isoformat(),
                                error=str(e),
                            )
                        _orch_log(f"orchestrator | raw {fname} ingest exception: {e}")

                job_record = job_store.get(job.job_id)
                continuation = (
                    job_record.result.get("ingest_continuation")
                    if job_record is not None and isinstance(job_record.result, dict)
                    else None
                )
                shard_continuing = bool(
                    not raw_success_flag[0]
                    and job_record is not None
                    and job_record.status is JobStatus.COMPLETED
                    and _valid_ingest_review_shard_continuation(continuation)
                )
                supervision = None
                if raw_success_flag[0]:
                    succeeded_filenames.extend(source_filenames)
                    succeeded_units += 1
                    try:
                        from chronovisor.ingest.failure_supervisor import (
                            reset_raw_failure,
                        )

                        for source_filename in source_filenames:
                            reset_raw_failure(source_filename)
                    except Exception as e:
                        _orch_log(
                            f"orchestrator | failure supervisor reset failed "
                            f"for {fname}: {e}"
                        )
                elif shard_continuing:
                    continued_filenames.extend(source_filenames)
                    continued_units += 1
                else:
                    if projection_group_quarantine is not None:
                        supervision = {
                            "raw_file": fname,
                            "failure_class": ("raw.semantic_projection_source_invalid"),
                            "fingerprint": (
                                "raw.semantic_projection_source_invalid:"
                                + hashlib.sha256(
                                    unit.fragment_record_sha256.encode("utf-8")
                                ).hexdigest()[:16]
                            ),
                            "attempts": 1,
                            "quarantined": True,
                            "packet_path": None,
                            "quarantine_path": projection_group_quarantine["manifest"],
                            "tracked": True,
                            "transient": False,
                        }
                    else:
                        try:
                            from chronovisor.ingest.failure_supervisor import (
                                record_raw_failure,
                                result_to_dict,
                            )

                            job_record = job_store.get(job.job_id)
                            supervision = result_to_dict(
                                record_raw_failure(
                                    raw_path=raw_path,
                                    error=job_record.error if job_record else None,
                                    job_id=job.job_id,
                                    raw_text=raw_text,
                                    related_raw_paths=unit.paths,
                                )
                            )
                        except Exception as e:
                            _orch_log(
                                f"orchestrator | failure supervisor failed for "
                                f"{fname}: {e}"
                            )

                semantic_deferred = bool(
                    isinstance(supervision, dict)
                    and supervision.get("terminal_deferred") is True
                )
                if semantic_deferred:
                    deferred_filenames.extend(source_filenames)
                    deferred_units += 1

                raw_result = _raw_unit_result(
                    filename=fname,
                    source_files=source_filenames,
                    job_id=job.job_id,
                    succeeded=raw_success_flag[0],
                    deferred=semantic_deferred,
                    continued=shard_continuing,
                    continuation=continuation,
                    fragment_record_sha256=unit.fragment_record_sha256,
                    projection=projection_summary,
                    supervision=supervision,
                    completion_ack=completion_ack_summary[0],
                )
                per_raw.append(raw_result)
                event_level, event_outcome = _raw_unit_event(
                    succeeded=raw_success_flag[0],
                    deferred=semantic_deferred,
                    continued=shard_continuing,
                )
                runtime_status.safe_append_event(
                    event_level,
                    f"orchestrator | raw {fname} {event_outcome}",
                    source="orchestrator",
                    raw_file=fname,
                    job_id=job.job_id,
                    supervision=supervision,
                )
        finally:
            release_state = _load_state()
            _clear_current_job(release_state)
            _save_state(release_state)

        elapsed = time.time() - batch_started
        _orch_log(
            f"orchestrator | batch done: {len(succeeded_filenames)}/{len(filenames)} "
            f"succeeded, {len(deferred_filenames)} deferred, "
            f"{len(continued_filenames)} continued, "
            f"{elapsed:.1f}s, jobs={len(job_ids)}"
        )
        pending_after = len(get_pending_raw_files())
        failed_filenames = max(
            0,
            len(filenames)
            - len(succeeded_filenames)
            - len(deferred_filenames)
            - len(continued_filenames),
        )
        runtime_status.safe_append_metric(
            "batch",
            pending_before=pending_before_count,
            pending_after=pending_after,
            files_attempted=len(filenames),
            files_processed=len(succeeded_filenames),
            files_deferred=len(deferred_filenames),
            files_continued=len(continued_filenames),
            files_failed=failed_filenames,
            elapsed_seconds=round(elapsed, 2),
            processor=get_ollama_status()["processor"],
        )
        runtime_status.safe_write_status(
            state="running" if pending_after else "idle",
            stage="waiting" if pending_after else "idle",
            pending=pending_after,
            current_raw=None,
            current_op=None,
            current_job_id=None,
            current_job_pid=None,
            batch={
                "started_at": datetime.fromtimestamp(batch_started).isoformat(),
                "index": len(units),
                "total": len(units),
                "succeeded": succeeded_units,
                "deferred": deferred_units,
                "continued": continued_units,
                "failed": (
                    len(units) - succeeded_units - deferred_units - continued_units
                ),
                "elapsed_seconds": round(elapsed, 2),
                "files": filenames,
            },
            ollama=get_ollama_status(),
            llm=None,
        )

        result = _ingest_batch_result(
            reason=reason,
            job_ids=job_ids,
            filenames=filenames,
            succeeded_filenames=succeeded_filenames,
            deferred_filenames=deferred_filenames,
            continued_filenames=continued_filenames,
            failed_filenames=failed_filenames,
            fragment_quarantined=fragment_quarantined,
            fragment_deferred=fragment_deferred,
            resumed_fragment_quarantines=resumed_fragment_quarantines,
            per_raw=per_raw,
            processor=get_ollama_status()["processor"],
            elapsed=elapsed,
        )
        if succeeded_filenames:
            result["post_ingest_lint"] = _post_ingest_lint_summary()
        return result


def _coerce_str_list(value: object) -> list[str] | None:
    """Return ``value`` as ``list[str]`` only if every element is a str.

    Anything else (None, str, dict, list with non-str items) returns
    ``None`` so the caller can fall through to the next source. We
    deliberately don't promote scalars to singleton lists — that would
    silently rewrite intent if a raw frontmatter accidentally wrote
    ``raw_keywords: foo`` instead of ``raw_keywords: [foo]``.
    """
    if isinstance(value, list) and all(isinstance(v, str) for v in value):
        return list(value)
    return None


def run_lint_if_due(*, dry_run: bool = False) -> dict:
    """Run lint check + safe apply if due.

    Returns result dict with status and details.
    """
    should, reason = should_lint()
    if not should:
        return {"triggered": False, "reason": reason}

    from chronovisor.ingest.lint import (
        apply_safe_fixes,
        check,
        summarize_issues,
        write_repair_queue,
    )

    issues = check()
    try:
        from chronovisor.ingest.snapshot import snapshot_chronovisor

        snapshot = (
            {"status": "skipped", "reason": "dry_run"}
            if dry_run
            else snapshot_chronovisor("before scheduled lint auto-fix")
        )
    except Exception as exc:
        snapshot = {"status": "error", "error": str(exc)}
    actions = apply_safe_fixes(issues, dry_run=dry_run)
    remaining = [i for i in issues if not i.get("auto_fixable")]
    if dry_run:
        repair_queue = str(CHRONOVISOR_ROOT / "review" / "lint-repair-queue.jsonl")
    else:
        try:
            repair_queue = str(write_repair_queue(remaining))
        except Exception:
            repair_queue = None

    if not dry_run:
        mark_lint_complete()

    return {
        "triggered": True,
        "reason": reason,
        "total_issues": len(issues),
        "summary": summarize_issues(issues),
        "snapshot": snapshot,
        "actions_taken": actions,
        "remaining_issues": len(remaining),
        "repair_queue": repair_queue,
        "dry_run": dry_run,
    }


def tick() -> dict:
    """Main orchestration tick. Call this periodically.

    Checks if ingest or lint should run, and triggers them if needed.
    Returns summary of what happened.
    """
    results = {
        "timestamp": datetime.now().isoformat(),
        "ollama": get_ollama_status(),
        "ingest": run_pending_ingest(),
        "lint": run_lint_if_due(),
    }
    return results
