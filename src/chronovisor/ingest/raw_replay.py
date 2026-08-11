"""Durable, bounded retroactive raw re-ingestion.

The replay queue is a lifecycle store rather than a nightly snapshot.  A raw
has one stable key, signals are merged into that row, and terminal/retry state
survives subsequent queue refreshes. A durable pre-launch marker and whole-raw
completion journal provide at-most-once launch plus evidence-based recovery;
unprovable crash windows are never blindly replayed.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import inspect
import json
import os
import re
from collections.abc import Callable, Iterable, Mapping
from contextlib import nullcontext, suppress
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from chronovisor.core.durable_state import sidecar_exclusive_lock as _queue_lock
from chronovisor.core.hashutil import sha256_file as _sha256_path
from chronovisor.core.jobs import JobStatus, job_store
from chronovisor.core.legacy_frontmatter import parse as parse_frontmatter
from chronovisor.core.page_mutation import decision_authority_lock
from chronovisor.core.raw_store import RawStore
from chronovisor.core.store import CHRONOVISOR_ROOT, RAW_DIR, okf_runtime_operation
from chronovisor.core.timeutil import iso_seconds as _iso
from chronovisor.decision.decision_authority import (
    compare_semantic_authority,
    current_semantic_authority,
    seal_semantic_artifact,
    semantic_authority_shape_error,
    semantic_verdict_authority_error,
)
from chronovisor.decision.decision_schema_manifest import (
    RAW_REPLAY_RECONCILIATION_SCHEMA,
)
from chronovisor.decision.frontier_guard import is_human_required_result
from chronovisor.decision.semantic_hold import (
    LOCAL_SEMANTIC_NO_QUORUM,
    build_semantic_no_quorum_hold,
    canonical_sha256,
    frontier_failure_class,
    is_local_semantic_no_quorum,
    persisted_semantic_no_quorum_hold,
    semantic_no_quorum_hold_error,
)

RAW_DATE_RE = re.compile(r"(20\d{6})")
QUEUE_FILE = CHRONOVISOR_ROOT / "review" / "raw-replay-queue.jsonl"
HISTORY_FILE = CHRONOVISOR_ROOT / "runtime" / "raw-replay-history.jsonl"
COMPLETIONS_FILE = CHRONOVISOR_ROOT / "runtime" / "raw-replay-completions.jsonl"
MEMORY_INTEGRITY_FILE = CHRONOVISOR_ROOT / "eval" / "memory-integrity-latest.json"
CLAIMS_FILE = CHRONOVISOR_ROOT / "claims" / "claims.jsonl"
INGEST_FAILURE_LOG_FILE = CHRONOVISOR_ROOT / "runtime" / "ingest-read-back-failures.jsonl"
RUNTIME_STATUS_FILE = CHRONOVISOR_ROOT / "runtime" / "status.json"
FAILURE_PACKETS_DIR = CHRONOVISOR_ROOT / "runtime" / "failures" / "packets"
QUARANTINED_RAW_DIR = CHRONOVISOR_ROOT / "runtime" / "failures" / "quarantined-raw"

SCHEMA_VERSION = 2
MAX_ATTEMPTS = 3
DEFAULT_MAX_BYTES = 10 * 1024 * 1024
DEFAULT_RETRY_DELAY_SECONDS = 60 * 60
DEFAULT_QUARANTINE_RETRY_SECONDS = 6 * 60 * 60

SOURCE_PRIORITY = {
    "ingest_failure": 300,
    "memory_integrity_miss": 200,
    "explicit_migration": 100,
}
AUTO_SIGNAL_SOURCES = frozenset({"ingest_failure", "memory_integrity_miss"})
READ_BACK_REPAIR_ONLY_REASONS = frozenset({"not-in-top-results"})
FULL_REPLAY_CLAIM_TYPE = "raw_replay_completion"
TERMINAL_STATUSES = frozenset(
    {"completed", "completed_partial", "quarantined", "not_needed", "human_required"}
)
RETRYABLE_STATUSES = frozenset({"pending", "failed", "retry"})
COMPLETED_HISTORY_STATUSES = frozenset(
    {"completed", "completed_partial", "success", "already_completed"}
)
MAX_FRONTIER_ATTEMPTS = 3
RAW_REPLAY_DECISION_LANE = "raw_replay_reconciliation"
RAW_REPLAY_APPROVAL_SCHEMA_VERSION = 2
LIFECYCLE_FIELDS = (
    "status",
    "attempts",
    "next_retry_at",
    "last_attempt_at",
    "completed_at",
    "last_error",
    "job_id",
    "quarantined_at",
    "not_needed_at",
    "terminal_reason",
    "reactivated_at",
    "completion_evidence",
    "attempt_id",
    "raw_sha256",
    "started_at",
    "frontier_attempts",
    "next_frontier_retry_at",
    "frontier_decision",
    "frontier_review_artifact",
    "frontier_authorization_consumed_at",
    "frontier_authority_error",
    "semantic_hold",
    "last_failure_class",
    "semantic_hold_recheck_sha256",
    "human_required_at",
    "recovery_kind",
    "semantic_deferred_at",
    "semantic_defer_authority_sha256",
    "semantic_defer_packet",
    "semantic_defer_prior_attempts",
    "semantic_defer_error",
    "semantic_defer_job_id",
    "semantic_defer_attempt_id",
    "semantic_defer_started_at",
    "failure_reset_pending",
    "failure_reset_error",
    "failure_reset_last_attempt_at",
    "failure_reset_completed_at",
)
NONTERMINAL_FAILURE_PACKET_STATUSES = frozenset(
    {
        "pending_local_repair",
        "local_repair_failed",
        "pending_frontier",
        "frontier_retry",
        "frontier_preflight_failed",
        "pending_frontier_review",
    }
)


def _now() -> datetime:
    return datetime.now().astimezone()




def _parse_dt(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _nonnegative_int(value: object, *, default: int = 0) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError, OverflowError):
        return max(0, default)


def _quarantine_retry_seconds() -> int:
    try:
        return max(
            0,
            int(
                os.getenv(
                    "CHRONOVISOR_CONVERGENCE_QUARANTINE_RETRY_SECONDS",
                    str(DEFAULT_QUARANTINE_RETRY_SECONDS),
                )
            ),
        )
    except (TypeError, ValueError):
        return DEFAULT_QUARANTINE_RETRY_SECONDS


def _active_terminal_semantic_deferred_raw_names() -> frozenset[str]:
    """Return every raw held by the failure supervisor.

    The legacy private name is retained for compatibility with existing tests
    and callers, but operational repair packets are just as authoritative as a
    semantic no-quorum defer. Raw replay must not bypass either kind of hold.
    """

    from chronovisor.ingest.failure_supervisor import operational_deferred_raw_files

    deferred = operational_deferred_raw_files()
    return frozenset(deferred)


def _active_operational_deferred_raw_statuses(
    rows: Iterable[Mapping[str, Any]] | None = None,
) -> dict[str, str]:
    """Return operational holds relevant to the queue being processed.

    Queue runners are also used with isolated paths by tests and repair tools.
    Scanning every production Raw unit in that case both leaks global state into
    the run and needlessly materializes projection files.  Passing the scoped
    queue rows keeps the supervisor lookup bounded to the actual work set.
    """

    from chronovisor.ingest.failure_supervisor import operational_deferred_raw_files

    raw_paths: list[Path] | None = None
    if rows is not None:
        raw_paths = []
        for row in rows:
            value = row.get("path")
            if isinstance(value, str) and value:
                raw_paths.append(Path(value))
    return operational_deferred_raw_files(raw_paths)


def _scoped_operational_deferred_raw_statuses(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, str]:
    """Call the scoped provider while retaining zero-argument test adapters."""

    provider = _active_operational_deferred_raw_statuses
    if not inspect.signature(provider).parameters:
        return provider()  # type: ignore[call-arg]
    return provider(rows)


def _at_or_before(left: datetime, right: datetime) -> bool:
    compare_left = left
    compare_right = right
    if compare_left.tzinfo is None and compare_right.tzinfo is not None:
        compare_right = compare_right.replace(tzinfo=None)
    elif compare_left.tzinfo is not None and compare_right.tzinfo is None:
        compare_left = compare_left.replace(tzinfo=None)
    return compare_left <= compare_right


def _resume_due_autonomous_terminal(
    row: dict[str, Any],
    *,
    now: datetime,
    semantic_deferred_raws: frozenset[str] = frozenset(),
) -> bool:
    """Reopen non-human raw replay quarantine after a bounded cooldown.

    ``quarantine_resumed_at`` survives the legacy history projection. That
    makes the migration idempotent even when an older terminal history record
    would otherwise win again on every queue load.
    """

    if _raw_name(row) in semantic_deferred_raws:
        return False

    if _has_semantic_no_quorum_marker(row):
        return False

    status = str(row.get("status") or "")
    if status == "human_required" and is_human_required_result(row):
        return False
    legacy_human = status == "human_required"
    if status != "quarantined" and not legacy_human:
        return False
    quarantined_at = _parse_dt(
        row.get("quarantined_at")
        or row.get("human_required_at")
        or row.get("updated_at")
        or row.get("last_attempt_at")
    )
    resumed_at = _parse_dt(row.get("quarantine_resumed_at"))
    already_resumed = bool(
        resumed_at is not None
        and (quarantined_at is None or _at_or_before(quarantined_at, resumed_at))
    )
    if not legacy_human and not already_resumed and quarantined_at is not None:
        retry_at = quarantined_at + timedelta(seconds=_quarantine_retry_seconds())
        if not _at_or_before(retry_at, now):
            return False

    frontier_owned = legacy_human or bool(
        _nonnegative_int(row.get("frontier_attempts"))
        or row.get("frontier_decision")
        or "frontier" in str(row.get("terminal_reason") or "").casefold()
        or "immutable raw hash" in str(row.get("terminal_reason") or "").casefold()
    )
    row["status"] = "indeterminate" if frontier_owned else "pending"
    row["attempts"] = 0
    row["frontier_attempts"] = 0
    row["next_retry_at"] = None
    row["next_frontier_retry_at"] = _iso(now) if frontier_owned else None
    row["terminal_reason"] = None
    row["last_error"] = None
    row["human_required_at"] = None
    row["quarantine_resumed_at"] = row.get("quarantine_resumed_at") or _iso(now)
    if not already_resumed:
        row["quarantine_reopen_count"] = (
            _nonnegative_int(row.get("quarantine_reopen_count")) + 1
        )
    row["updated_at"] = _iso(now)
    return True


def is_raw_retracted(path: Path) -> bool:
    """Return whether a raw capture is explicitly excluded from ingestion.

    Missing, malformed, and unknown ``raw_status`` values deliberately fail
    open so existing raw metadata remains backwards-compatible.  Only the
    explicit scalar value ``retracted`` opts a raw out.  This function is
    read-only; the raw body remains the immutable audit record.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return False
    meta, _body = parse_frontmatter(text)
    status = meta.get("raw_status")
    return isinstance(status, str) and status.strip().casefold() == "retracted"


def raw_date(path: Path) -> str:
    match = RAW_DATE_RE.search(path.name)
    return (
        match.group(1)
        if match
        else datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y%m%d")
    )


def select_raws(
    *,
    since: str = "",
    limit: int = 0,
    store: RawStore | None = None,
) -> list[Path]:
    raw_store = store or RawStore(RAW_DIR)
    reference_dir = RAW_DIR.parent / "runtime" / "raw-projections" / "parents"
    candidates_with_dates: list[tuple[str, str, Any]] = []
    for unit in raw_store.iter_units():
        if unit.captured_at:
            date_key = unit.captured_at[:10].replace("-", "")
        else:
            match = RAW_DATE_RE.search(unit.raw_id)
            date_key = (
                match.group(1)
                if match
                else datetime.fromtimestamp(unit.path.stat().st_mtime).strftime(
                    "%Y%m%d"
                )
            )
        candidates_with_dates.append((date_key, unit.raw_id, unit))
    candidates_with_dates.sort(key=lambda row: (row[0], row[1]))
    if since:
        normalized = since.replace("-", "")
        candidates_with_dates = [
            row for row in candidates_with_dates if row[0] >= normalized
        ]

    # Retraction is content metadata, but ``limit`` is an output bound. Inspect
    # bodies only in sorted order until enough active raws have been found.
    # The previous implementation decoded every archived raw before applying
    # the bound, turning a 150-row integrity sample into a full archive scan.
    candidates: list[Path] = []
    for _date_key, _raw_id, unit in candidates_with_dates:
        if limit and len(candidates) >= limit:
            break
        try:
            text = raw_store.read_text(unit)
        except (OSError, UnicodeError):
            continue
        meta, _body = parse_frontmatter(text)
        status = meta.get("raw_status")
        if isinstance(status, str) and status.strip().casefold() == "retracted":
            continue
        path = (
            unit.path
            if unit.storage == "legacy_file"
            else raw_store.materialize_ingest(unit, reference_dir)
        )
        candidates.append(path)
    return candidates


def stable_key(raw: str | Path) -> str:
    """Return the canonical queue identity for one immutable raw capture."""
    return f"raw:{Path(raw).name}"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").split("\n")
    except OSError:
        return []
    rows: list[dict[str, Any]] = []
    for line in lines:
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _atomic_write_queue(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, default=str) + "\n"
        for row in rows
    )
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with tmp.open("w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
        except OSError:
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        with suppress(OSError):
            tmp.unlink(missing_ok=True)


def _append_history(row: dict[str, Any], path: Path | None = None) -> None:
    target = path or HISTORY_FILE
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, default=str) + "\n"
            )
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def _append_completion(row: dict[str, Any], path: Path | None = None) -> None:
    """Append the durable processed marker owned by the replay consumer.

    Ordinary page claims are per-operation and cannot prove that a whole raw
    was processed.  This journal is written from ``run_ingest.on_complete``
    before control returns to the queue runner, closing the success-to-queue
    crash window.
    """
    target = path or COMPLETIONS_FILE
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, default=str) + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)




def _raw_name(record: dict[str, Any]) -> str:
    for field in ("raw", "raw_file", "source_raw"):
        value = record.get(field)
        if not isinstance(value, str) or not value.strip():
            continue
        value = value.removeprefix("replay:")
        return Path(value).name
    path_value = record.get("path")
    return Path(path_value).name if isinstance(path_value, str) and path_value else ""


def _resolve_raw_path(raw_name: str, preferred: object = None) -> Path | None:
    candidates: list[Path] = []
    if isinstance(preferred, str) and preferred:
        candidates.append(Path(preferred).expanduser())
    candidates.extend(
        [
            RAW_DIR / raw_name,
            QUARANTINED_RAW_DIR / raw_name,
            RAW_DIR / ".dead-letter" / raw_name,
        ]
    )
    for path in candidates:
        if path.is_file():
            return path
    from chronovisor.core.raw_store import RawStore

    store = RawStore(RAW_DIR)
    unit = store.resolve(raw_name)
    if unit is not None:
        return store.materialize_ingest(
            unit,
            RAW_DIR.parent / "runtime" / "raw-projections" / "parents",
        )
    return None


def _logical_raw_bytes(raw_name: str, path: Path) -> bytes:
    """Read the immutable logical Raw behind a flat file or reference."""

    from chronovisor.core.raw_store import RawStore

    store = RawStore(RAW_DIR)
    referenced = store.resolve_reference(path)
    if referenced is not None:
        return store.read_bytes(referenced)
    if path.is_file() and path.parent in {
        RAW_DIR,
        QUARANTINED_RAW_DIR,
        RAW_DIR / ".dead-letter",
    }:
        return path.read_bytes()
    unit = store.resolve(raw_name)
    return store.read_bytes(unit) if unit is not None else path.read_bytes()


def _replay_ingest_content(
    raw_name: str, path: Path
) -> tuple[str | None, dict[str, Any]]:
    """Return semantic replay input without exposing a v2 transport trace.

    Legacy Raw files already contain the historical ingest envelope and remain
    byte-for-byte passthrough inputs.  A source-native v2 Raw is first projected
    through the same deterministic adapter used by the ordinary orchestrator.
    The verified child envelopes are then bundled into one replay operation so
    the replay lifecycle keeps its existing one-job/one-completion contract.
    """

    from chronovisor.core.raw_store import RawStore
    from chronovisor.core.runtime_config import load_ingest_config
    from chronovisor.ingest.raw_semantic_projection import project_native_transcript

    store = RawStore(RAW_DIR)
    unit = store.resolve_reference(path) or store.resolve(raw_name)
    if unit is None or unit.commit is None:
        value = _logical_raw_bytes(raw_name, path)
        return value.decode("utf-8"), {"kind": "legacy_passthrough"}

    raw_bytes = store.read_bytes(unit)
    projection = project_native_transcript(
        path,
        raw_bytes,
        unit.commit,
        output_dir=CHRONOVISOR_ROOT / "runtime" / "raw-projections" / "artifacts",
        max_child_bytes=load_ingest_config().semantic_projection_max_child_bytes,
    )
    summary: dict[str, Any] = {
        "kind": projection.kind,
        "projection_sha256": projection.projection_sha256,
        "manifest_path": (
            str(projection.manifest_path)
            if projection.manifest_path is not None
            else None
        ),
        "child_count": projection.child_count,
    }
    if projection.kind == "noop":
        return None, summary
    children = [
        json.loads(child.read_text(encoding="utf-8"))
        for child in projection.child_paths
    ]
    bundle = {
        "schema": "chronovisor.raw-replay-semantic-bundle.v1",
        "kind": "raw_replay_semantic_bundle",
        "source_raw": raw_name,
        "projection_sha256": projection.projection_sha256,
        "children": children,
    }
    return (
        json.dumps(bundle, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        summary,
    )


def _logical_raw_date(raw_name: str, path: Path) -> str:
    from chronovisor.core.raw_store import RawStore

    store = RawStore(RAW_DIR)
    unit = store.resolve_reference(path) or store.resolve(raw_name)
    if unit is not None and unit.captured_at:
        return unit.captured_at[:10].replace("-", "")
    return raw_date(path)


def _candidate(
    path: Path,
    *,
    source: str,
    reason: str,
    now: datetime,
) -> dict[str, Any] | None:
    if is_raw_retracted(path):
        return None
    raw = path.name
    value = _logical_raw_bytes(raw, path)
    return {
        "schema_version": SCHEMA_VERSION,
        "key": stable_key(raw),
        "type": "raw_replay_candidate",
        "raw": raw,
        "path": str(path),
        "date": _logical_raw_date(raw, path),
        "bytes": len(value),
        "raw_sha256": hashlib.sha256(value).hexdigest(),
        "priority": SOURCE_PRIORITY[source],
        "sources": [source],
        "reasons": [reason] if reason else [],
        "status": "pending",
        "attempts": 0,
        "next_retry_at": None,
        "created_at": _iso(now),
        "updated_at": _iso(now),
        "last_attempt_at": None,
        "completed_at": None,
        "last_error": None,
    }


def _normalize_queue_row(
    row: dict[str, Any], *, now: datetime
) -> dict[str, Any] | None:
    raw = _raw_name(row)
    if not raw:
        return None
    path = _resolve_raw_path(raw, row.get("path"))
    normalized = dict(row)
    normalized.update(
        {
            "schema_version": SCHEMA_VERSION,
            "key": stable_key(raw),
            "type": "raw_replay_candidate",
            "raw": raw,
            "path": str(path or row.get("path") or (RAW_DIR / raw)),
            "status": str(row.get("status") or "pending"),
            "attempts": _nonnegative_int(row.get("attempts")),
            "next_retry_at": row.get("next_retry_at"),
            "created_at": row.get("created_at") or _iso(now),
            "updated_at": row.get("updated_at") or _iso(now),
            "last_attempt_at": row.get("last_attempt_at"),
            "completed_at": row.get("completed_at"),
            "last_error": row.get("last_error"),
        }
    )
    sources = row.get("sources")
    if not isinstance(sources, list):
        source = row.get("source")
        sources = (
            [source] if isinstance(source, str) and source else ["explicit_migration"]
        )
    normalized["sources"] = sorted(
        {str(item) for item in sources if isinstance(item, str) and item},
        key=lambda item: (-SOURCE_PRIORITY.get(item, 0), item),
    )
    reasons = row.get("reasons")
    if not isinstance(reasons, list):
        reason = row.get("reason")
        reasons = [reason] if isinstance(reason, str) and reason else []
    normalized["reasons"] = list(
        dict.fromkeys(str(item) for item in reasons if isinstance(item, str) and item)
    )
    source_priorities = [
        SOURCE_PRIORITY[source]
        for source in normalized["sources"]
        if source in SOURCE_PRIORITY
    ]
    normalized["priority"] = (
        max(source_priorities)
        if source_priorities
        else _nonnegative_int(row.get("priority"))
    )
    if path is not None:
        value = _logical_raw_bytes(raw, path)
        normalized["bytes"] = len(value)
        normalized["raw_sha256"] = hashlib.sha256(value).hexdigest()
        normalized["date"] = _logical_raw_date(raw, path)
    else:
        normalized["bytes"] = _nonnegative_int(row.get("bytes"))
        normalized["date"] = str(row.get("date") or "")
    return normalized


def _row_is_retracted(row: dict[str, Any]) -> bool:
    raw = _raw_name(row)
    path = _resolve_raw_path(raw, row.get("path")) if raw else None
    return path is not None and is_raw_retracted(path)


def _merge_rows(
    current: dict[str, Any], incoming: dict[str, Any], *, now: datetime
) -> dict[str, Any]:
    merged = dict(current)
    merged["sources"] = sorted(
        set(current.get("sources", [])) | set(incoming.get("sources", [])),
        key=lambda item: (-SOURCE_PRIORITY.get(str(item), 0), str(item)),
    )
    merged["reasons"] = list(
        dict.fromkeys([*current.get("reasons", []), *incoming.get("reasons", [])])
    )
    merged["priority"] = max(
        _nonnegative_int(current.get("priority")),
        _nonnegative_int(incoming.get("priority")),
    )
    merged["path"] = incoming.get("path") or current.get("path")
    merged["bytes"] = incoming.get("bytes", current.get("bytes", 0))
    merged["date"] = incoming.get("date") or current.get("date", "")
    merged["updated_at"] = _iso(now)
    # Lifecycle fields intentionally remain from the durable row.  Refreshing
    # signals must never resurrect completed/quarantined work or reset backoff.
    return merged


def _lifecycle_sort_key(row: dict[str, Any]) -> tuple[int, int, int, str]:
    status = str(row.get("status") or "pending")
    terminal_rank = {
        "completed": 5,
        "completed_partial": 5,
        "human_required": 4,
        "quarantined": 3,
        "not_needed": 2,
        "indeterminate": 1,
        "running": 1,
    }.get(status, 0)
    retry_rank = 1 if status in {"failed", "retry"} else 0
    timestamp = str(
        row.get("completed_at")
        or row.get("quarantined_at")
        or row.get("not_needed_at")
        or row.get("last_attempt_at")
        or row.get("updated_at")
        or ""
    )
    return terminal_rank, _nonnegative_int(row.get("attempts")), retry_rank, timestamp


def _merge_durable_rows(
    current: dict[str, Any],
    incoming: dict[str, Any],
    *,
    now: datetime,
) -> dict[str, Any]:
    """Merge duplicate stored rows without regressing their lifecycle."""
    merged = _merge_rows(current, incoming, now=now)
    winner = max((current, incoming), key=_lifecycle_sort_key)
    for field in LIFECYCLE_FIELDS:
        if field in winner:
            merged[field] = winner.get(field)
    created = [
        str(row.get("created_at"))
        for row in (current, incoming)
        if row.get("created_at")
    ]
    if created:
        merged["created_at"] = min(created)
    return merged


def _memory_integrity_candidates(*, now: datetime) -> list[dict[str, Any]]:
    payload = _read_json(MEMORY_INTEGRITY_FILE)
    rows = payload.get("rows")
    if not isinstance(rows, list):
        return []
    out: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict) or row.get("status") != "miss":
            continue
        raw = _raw_name(row)
        path = _resolve_raw_path(raw, row.get("path")) if raw else None
        if path is None:
            continue
        reason = str(
            row.get("reason") or row.get("query") or "memory integrity search miss"
        )
        candidate = _candidate(
            path,
            source="memory_integrity_miss",
            reason=reason,
            now=now,
        )
        if candidate is not None:
            out.append(candidate)
    return out


def _explicit_migration_candidates(
    *, since: str, now: datetime
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for path in select_raws(since=since, limit=0):
        candidate = _candidate(
            path,
            source="explicit_migration",
            reason=f"explicit migration since {since or 'all'}",
            now=now,
        )
        if candidate is not None:
            out.append(candidate)
    return out


def _claim_sources_by_page() -> dict[str, list[tuple[datetime | None, int, str]]]:
    by_page: dict[str, list[tuple[datetime | None, int, str]]] = {}
    local_tz = _now().tzinfo
    for index, claim in enumerate(_read_jsonl(CLAIMS_FILE)):
        page_id = claim.get("source_page")
        raw = _raw_name(claim)
        if not isinstance(page_id, str) or not page_id or not raw:
            continue
        recorded_at = _parse_dt(claim.get("recorded_at"))
        if recorded_at is not None and recorded_at.tzinfo is None:
            recorded_at = recorded_at.replace(tzinfo=local_tz)
        by_page.setdefault(page_id, []).append((recorded_at, index, raw))
    return by_page


def _read_back_raws(
    record: dict[str, Any],
    *,
    claims_by_page: dict[str, list[tuple[datetime | None, int, str]]],
) -> list[tuple[str, str]]:
    """Resolve legacy read-back rows, which did not record their source raw."""
    failed = record.get("failed")
    if not isinstance(failed, list):
        return []
    occurred_at = _parse_dt(record.get("timestamp") or record.get("ts"))
    if occurred_at is not None and occurred_at.tzinfo is None:
        occurred_at = occurred_at.replace(tzinfo=_now().tzinfo)
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for failure in failed:
        if not isinstance(failure, dict):
            continue
        if (
            str(failure.get("reason") or "").strip().casefold()
            in READ_BACK_REPAIR_ONLY_REASONS
        ):
            continue
        page_id = failure.get("page_id")
        if not isinstance(page_id, str) or not page_id:
            continue
        events = claims_by_page.get(page_id, [])
        eligible = [
            event
            for event in events
            if occurred_at is None or event[0] is None or event[0] <= occurred_at
        ]
        if not eligible:
            continue
        _recorded_at, _index, raw = max(
            eligible,
            key=lambda event: (
                event[0] is not None,
                event[0] or datetime.min.replace(tzinfo=_now().tzinfo),
                event[1],
            ),
        )
        if raw in seen:
            continue
        seen.add(raw)
        detail = str(failure.get("reason") or "not readable through search")
        out.append((raw, f"ingest read-back failure for {page_id}: {detail}"))
    return out


def _ingest_failure_candidates(*, now: datetime) -> list[dict[str, Any]]:
    records: list[tuple[dict[str, Any], str]] = []
    for path in (
        sorted(FAILURE_PACKETS_DIR.glob("*.json"))
        if FAILURE_PACKETS_DIR.exists()
        else []
    ):
        packet = _read_json(path)
        status = str(packet.get("status") or "")
        if status in NONTERMINAL_FAILURE_PACKET_STATUSES:
            records.append(
                (
                    packet,
                    str(
                        packet.get("failure_class")
                        or packet.get("error")
                        or "ingest failure packet"
                    ),
                )
            )
    read_back_rows = _read_jsonl(INGEST_FAILURE_LOG_FILE)
    for row in read_back_rows:
        failed = row.get("failed")
        if not isinstance(failed, list) or any(
            not isinstance(item, dict)
            or str(item.get("reason") or "").strip().casefold()
            not in READ_BACK_REPAIR_ONLY_REASONS
            for item in failed
        ):
            records.append((row, str(row.get("reason") or "ingest read-back failure")))
    runtime = _read_json(RUNTIME_STATUS_FILE)
    last_success = runtime.get("last_success")
    if isinstance(last_success, dict):
        failed_ops = last_success.get("failed_ops")
        read_back = last_success.get("read_back")
        read_back_failed = (
            read_back.get("failed") if isinstance(read_back, dict) else None
        )
        actionable_read_back = isinstance(read_back_failed, list) and any(
            not isinstance(item, dict)
            or str(item.get("reason") or "").strip().casefold()
            not in READ_BACK_REPAIR_ONLY_REASONS
            for item in read_back_failed
        )
        if (isinstance(failed_ops, list) and failed_ops) or actionable_read_back:
            records.append(
                (last_success, "latest ingest had failed ops or read-back failures")
            )

    out_by_key: dict[str, dict[str, Any]] = {}
    for record, reason in records:
        raw = _raw_name(record)
        path = _resolve_raw_path(raw, record.get("path")) if raw else None
        if path is None:
            continue
        candidate = _candidate(path, source="ingest_failure", reason=reason, now=now)
        if candidate is None:
            continue
        current = out_by_key.get(candidate["key"])
        out_by_key[candidate["key"]] = (
            candidate if current is None else _merge_rows(current, candidate, now=now)
        )

    claims_by_page = _claim_sources_by_page() if read_back_rows else {}
    for record in read_back_rows:
        for raw, reason in _read_back_raws(record, claims_by_page=claims_by_page):
            path = _resolve_raw_path(raw)
            if path is None:
                continue
            candidate = _candidate(
                path, source="ingest_failure", reason=reason, now=now
            )
            if candidate is None:
                continue
            current = out_by_key.get(candidate["key"])
            out_by_key[candidate["key"]] = (
                candidate
                if current is None
                else _merge_rows(current, candidate, now=now)
            )

    # Quarantined sources are durable failure evidence even if an old packet
    # has already reached a code-fix terminal state.
    if QUARANTINED_RAW_DIR.exists():
        for path in sorted(QUARANTINED_RAW_DIR.glob("*.md")):
            candidate = _candidate(
                path,
                source="ingest_failure",
                reason="raw remains in ingest quarantine",
                now=now,
            )
            if candidate is None:
                continue
            current = out_by_key.get(candidate["key"])
            out_by_key[candidate["key"]] = (
                candidate
                if current is None
                else _merge_rows(current, candidate, now=now)
            )
    dead_letter = RAW_DIR / ".dead-letter"
    if dead_letter.exists():
        for path in sorted(dead_letter.glob("*.md")):
            candidate = _candidate(
                path,
                source="ingest_failure",
                reason="raw remains in ingest dead-letter",
                now=now,
            )
            if candidate is None:
                continue
            current = out_by_key.get(candidate["key"])
            out_by_key[candidate["key"]] = (
                candidate
                if current is None
                else _merge_rows(current, candidate, now=now)
            )
    return list(out_by_key.values())


def _history_lifecycle_states(
    history_file: Path | None = None,
) -> dict[str, dict[str, Any]]:
    states: dict[str, dict[str, Any]] = {}
    for record in _read_jsonl(history_file or HISTORY_FILE):
        raw = _raw_name(record)
        if not raw:
            continue
        key = stable_key(raw)
        status = str(record.get("status") or "").lower()
        if status in COMPLETED_HISTORY_STATUSES:
            status = (
                "completed_partial" if status == "completed_partial" else "completed"
            )
        elif status in {"error", "missing"}:
            status = "failed"
        if status not in TERMINAL_STATUSES | RETRYABLE_STATUSES:
            continue
        previous = states.get(key)
        explicit_attempts = record.get("attempts")
        attempts = (
            _nonnegative_int(explicit_attempts)
            if explicit_attempts is not None
            else _nonnegative_int(previous.get("attempts") if previous else 0) + 1
        )
        state = {
            "status": status,
            "attempts": attempts,
            "next_retry_at": record.get("next_retry_at"),
            "last_attempt_at": record.get("last_attempt_at")
            or record.get("ts")
            or record.get("timestamp"),
            "completed_at": (
                (record.get("completed_at") or record.get("ts"))
                if status in {"completed", "completed_partial"}
                else None
            ),
            "last_error": record.get("last_error") or record.get("error"),
            "job_id": record.get("job_id"),
            "quarantined_at": (
                (record.get("quarantined_at") or record.get("ts"))
                if status == "quarantined"
                else None
            ),
            "completion_evidence": (
                "replay_history"
                if status in {"completed", "completed_partial"}
                else None
            ),
            "terminal_reason": record.get("terminal_reason"),
            "frontier_decision": record.get("frontier_decision"),
            "next_frontier_retry_at": record.get("next_frontier_retry_at"),
            "frontier_review_artifact": record.get("frontier_review_artifact"),
            "frontier_authorization_consumed_at": record.get(
                "frontier_authorization_consumed_at"
            ),
            "frontier_authority_error": record.get("frontier_authority_error"),
            "human_required_at": (
                (record.get("human_required_at") or record.get("ts"))
                if status == "human_required"
                else None
            ),
            "recovery_kind": record.get("recovery_kind"),
        }
        states[key] = (
            state
            if previous is None
            else max((previous, state), key=_lifecycle_sort_key)
        )
    return states


def _merge_history_lifecycle(
    row: dict[str, Any],
    history_state: dict[str, Any] | None,
) -> dict[str, Any]:
    # A successfully published semantic defer is newer, stronger evidence than
    # the legacy failed/quarantined replay history it repairs.  Keeping the
    # durable queue row authoritative here prevents an old quarantine event
    # from resurrecting the three-attempt retry loop on the next process tick.
    if row.get("recovery_kind") == "semantic_no_quorum_terminal_defer":
        return row
    reactivated_at = _parse_dt(
        row.get("reactivated_at") or row.get("quarantine_resumed_at")
    )
    history_terminal_at = (
        _parse_dt(
            history_state.get("quarantined_at") or history_state.get("last_attempt_at")
        )
        if history_state is not None
        else None
    )
    if (
        reactivated_at is not None
        and history_terminal_at is not None
        and _at_or_before(history_terminal_at, reactivated_at)
    ):
        return row
    if history_state is None or _lifecycle_sort_key(row) >= _lifecycle_sort_key(
        history_state
    ):
        return row
    merged = dict(row)
    for field in LIFECYCLE_FIELDS:
        if field in history_state:
            merged[field] = history_state.get(field)
    return merged


def _completed_replays(
    *,
    history_file: Path | None = None,
    claims_file: Path | None = None,
    completions_file: Path | None = None,
) -> dict[str, str]:
    completed = {
        key: "replay_history"
        for key, state in _history_lifecycle_states(history_file).items()
        if state.get("status") in {"completed", "completed_partial"}
    }
    for key, record in _completion_states(completions_file).items():
        if record.get("status") in {"completed", "completed_partial"}:
            completed[key] = "replay_completion_journal"
    # Page claims are emitted before read-back and can survive a partial apply
    # or crash. Only a dedicated full-raw completion marker is exact-once
    # evidence; every ordinary replay:* claim is deliberately ignored. A
    # crash with only per-page claims enters frontier reconciliation instead
    # of being mistaken for whole-raw success or blindly replayed.
    for row in _read_jsonl(claims_file or CLAIMS_FILE):
        source = row.get("source_raw")
        if (
            not isinstance(source, str)
            or not source.startswith("replay:")
            or row.get("type") != FULL_REPLAY_CLAIM_TYPE
            or row.get("status") != "completed"
            or row.get("completion_scope") != "full_raw"
        ):
            continue
        raw = Path(source.removeprefix("replay:")).name
        if raw:
            completed[stable_key(raw)] = "replay_completion_claim"
    return completed


def _completion_states(path: Path | None = None) -> dict[str, dict[str, Any]]:
    states: dict[str, dict[str, Any]] = {}
    for record in _read_jsonl(path or COMPLETIONS_FILE):
        raw = _raw_name(record)
        status = str(record.get("status") or "")
        if not raw or status not in {"completed", "completed_partial"}:
            continue
        key = stable_key(raw)
        previous = states.get(key)
        if previous is None or str(record.get("ts") or "") >= str(
            previous.get("ts") or ""
        ):
            states[key] = record
    return states


def _queue_sort_key(row: dict[str, Any]) -> tuple[int, str, str]:
    return (
        -_nonnegative_int(row.get("priority")),
        str(row.get("date") or ""),
        str(row.get("key") or ""),
    )


def _mark_completed(
    row: dict[str, Any],
    *,
    evidence: str,
    now: datetime,
    status: str = "completed",
) -> None:
    row["status"] = status
    row["completed_at"] = row.get("completed_at") or _iso(now)
    row["completion_evidence"] = evidence
    row["next_retry_at"] = None
    row["last_error"] = None
    row["next_frontier_retry_at"] = None
    row["updated_at"] = _iso(now)


def _same_attempt(row: dict[str, Any], marker: dict[str, Any]) -> bool:
    return all(
        not row.get(field)
        or not marker.get(field)
        or str(row.get(field)) == str(marker.get(field))
        for field in ("attempt_id", "job_id", "raw_sha256")
    )


def _raw_hash_still_matches(row: dict[str, Any]) -> bool:
    expected = str(row.get("raw_sha256") or "")
    if not expected:
        return True
    path = _resolve_raw_path(str(row.get("raw") or ""), row.get("path"))
    if path is None:
        return False
    try:
        value = _logical_raw_bytes(_raw_name(row), path)
        return hashlib.sha256(value).hexdigest() == expected
    except OSError:
        return False


def _runtime_attempt_evidence(row: dict[str, Any]) -> tuple[str, str] | None:
    runtime = _read_json(RUNTIME_STATUS_FILE)
    raw = str(row.get("raw") or "")
    job_id = str(row.get("job_id") or "")
    last_success = runtime.get("last_success")
    if (
        isinstance(last_success, dict)
        and str(last_success.get("job_id") or "") == job_id
        and _raw_name(last_success) == raw
    ):
        failed_ops = last_success.get("failed_ops")
        status = (
            "completed_partial"
            if isinstance(failed_ops, list) and bool(failed_ops)
            else "completed"
        )
        return status, "runtime_last_success"
    if (
        runtime.get("state") == "error"
        and str(runtime.get("current_job_id") or "") == job_id
        and _raw_name(runtime) == raw
    ):
        error = str(runtime.get("last_error") or "runtime recorded ingest failure")
        if "partial rollback" not in error.casefold():
            return "failed", error
    return None


def _mark_failed_attempt(
    row: dict[str, Any],
    *,
    now: datetime,
    retry_delay_seconds: int,
    error: str,
) -> None:
    attempts = _nonnegative_int(row.get("attempts"))
    row["last_error"] = error
    row["updated_at"] = _iso(now)
    if attempts >= MAX_ATTEMPTS:
        row["status"] = "quarantined"
        row["quarantined_at"] = row.get("quarantined_at") or _iso(now)
        row["next_retry_at"] = None
        return
    row["status"] = "failed"
    delay = max(0, retry_delay_seconds) * (2 ** max(0, attempts - 1))
    row["next_retry_at"] = _iso(now + timedelta(seconds=delay))


def _reconcile_running_rows(
    rows: list[dict[str, Any]],
    *,
    completions_file: Path,
    history_file: Path,
    now: datetime,
    retry_delay_seconds: int,
) -> list[dict[str, Any]]:
    """Recover durable pre-launch rows without ever blindly replaying them."""
    completion_states = _completion_states(completions_file)
    events: list[dict[str, Any]] = []
    for row in rows:
        if row.get("status") != "running":
            continue
        key = str(row.get("key") or "")
        marker = completion_states.get(key)
        evidence = ""
        status = ""
        if marker is not None and _same_attempt(row, marker):
            status = str(marker.get("status") or "completed")
            evidence = "replay_completion_journal"
        elif not _raw_hash_still_matches(row):
            row["status"] = "quarantined"
            row["quarantined_at"] = _iso(now)
            row["last_error"] = "raw changed or disappeared after replay launch"
            row["terminal_reason"] = "immutable raw hash mismatch"
            row["updated_at"] = _iso(now)
        else:
            runtime_evidence = _runtime_attempt_evidence(row)
            if runtime_evidence is not None:
                status, evidence = runtime_evidence
            else:
                row["status"] = "indeterminate"
                row["last_error"] = (
                    "replay process ended without whole-raw completion evidence; "
                    "blind replay is disabled"
                )
                row["terminal_reason"] = "awaiting autonomous frontier reconciliation"
                row["frontier_attempts"] = _nonnegative_int(
                    row.get("frontier_attempts")
                )
                row["next_frontier_retry_at"] = row.get(
                    "next_frontier_retry_at"
                ) or _iso(now)
                row["updated_at"] = _iso(now)
        if status in {"completed", "completed_partial"}:
            _mark_completed(row, evidence=evidence, now=now, status=status)
            # This is evidence-based bookkeeping recovery, not reuse of a
            # semantic replay authorization. No new page mutation is launched.
            row["recovery_kind"] = "exact_already_applied"
            if status == "completed_partial":
                row["terminal_reason"] = (
                    "partial operations delegated to autonomous repair"
                )
        elif status == "failed":
            _mark_failed_attempt(
                row,
                now=now,
                retry_delay_seconds=retry_delay_seconds,
                error=evidence,
            )
        event = {
            "ts": _iso(now),
            "schema_version": SCHEMA_VERSION,
            "key": key,
            "raw": row.get("raw"),
            "status": row.get("status"),
            "attempts": row.get("attempts"),
            "job_id": row.get("job_id"),
            "attempt_id": row.get("attempt_id"),
            "completion_evidence": row.get("completion_evidence"),
            "recovery_kind": row.get("recovery_kind"),
            "error": row.get("last_error"),
            "reconciled": True,
        }
        _append_history(event, history_file)
        events.append(event)
    return events


def _frontier_due(row: dict[str, Any], *, now: datetime) -> bool:
    retry_at = _parse_dt(row.get("next_frontier_retry_at"))
    if retry_at is None:
        return True
    compare_now = now
    if retry_at.tzinfo is None and compare_now.tzinfo is not None:
        compare_now = compare_now.replace(tzinfo=None)
    elif retry_at.tzinfo is not None and compare_now.tzinfo is None:
        retry_at = retry_at.replace(tzinfo=None)
    return retry_at <= compare_now


def _raw_claim_evidence(row: dict[str, Any], claims_file: Path) -> list[dict[str, Any]]:
    raw = str(row.get("raw") or "")
    return [claim for claim in _read_jsonl(claims_file) if _raw_name(claim) == raw][
        -20:
    ]


_REVIEW_MUTABLE_ROW_FIELDS = frozenset(
    {
        "status",
        "updated_at",
        "last_error",
        "terminal_reason",
        "next_retry_at",
        "quarantined_at",
        "human_required_at",
        "frontier_attempts",
        "frontier_decision",
        "frontier_failure",
        "frontier_review_artifact",
        "frontier_authorization_consumed_at",
        "frontier_authority_error",
        "next_frontier_retry_at",
        "semantic_hold",
        "last_failure_class",
        "semantic_hold_recheck_sha256",
    }
)


def _review_queue_subject(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return the crash attempt identity unaffected by review lifecycle writes."""

    return {
        str(key): value
        for key, value in row.items()
        if key not in _REVIEW_MUTABLE_ROW_FIELDS
    }


def _raw_replay_review_evidence(
    row: dict[str, Any],
    *,
    claims_file: Path,
) -> dict[str, Any]:
    raw_path = _resolve_raw_path(str(row.get("raw") or ""), row.get("path"))
    try:
        raw_payload = _logical_raw_bytes(_raw_name(row), raw_path) if raw_path else b""
        raw_excerpt = raw_payload.decode("utf-8")[:4000] if raw_path else ""
        current_raw_sha256 = (
            hashlib.sha256(raw_payload).hexdigest() if raw_path else None
        )
    except (OSError, UnicodeDecodeError):
        raw_excerpt = ""
        current_raw_sha256 = None
    return {
        "queue_row": _review_queue_subject(row),
        "claims": _raw_claim_evidence(row, claims_file),
        "runtime_status": _read_json(RUNTIME_STATUS_FILE),
        "raw_excerpt": raw_excerpt,
        "current_raw_sha256": current_raw_sha256,
    }


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _has_semantic_no_quorum_marker(value: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    return (
        "semantic_hold" in value
        or value.get("last_failure_class") == LOCAL_SEMANTIC_NO_QUORUM
        or frontier_failure_class(value) == LOCAL_SEMANTIC_NO_QUORUM
    )


def _raw_replay_semantic_epoch(evidence: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "queue_schema_version": SCHEMA_VERSION,
        "review_schema_version": RAW_REPLAY_APPROVAL_SCHEMA_VERSION,
        "review_input_sha256": _canonical_sha256(evidence),
        "review_schema_sha256": canonical_sha256(RAW_REPLAY_RECONCILIATION_SCHEMA),
    }


def _restore_raw_replay_semantic_hold(
    row: dict[str, Any],
    *,
    hold: Mapping[str, Any] | None,
    now: datetime,
    malformed: bool = False,
) -> None:
    row["status"] = "quarantined"
    row["last_failure_class"] = LOCAL_SEMANTIC_NO_QUORUM
    row["last_error"] = (
        "malformed local semantic no-quorum hold; refusing resample"
        if malformed
        else "local semantic models did not reach a safe quorum"
    )
    row["terminal_reason"] = row["last_error"]
    row["quarantined_at"] = row.get("quarantined_at") or _iso(now)
    row["next_frontier_retry_at"] = None
    row["next_retry_at"] = None
    row["updated_at"] = _iso(now)
    if hold is not None:
        row["semantic_hold"] = dict(hold)


def _historical_raw_replay_hold(
    history_file: Path,
    *,
    key: object,
    epoch: Mapping[str, Any],
    authority: Mapping[str, Any],
) -> dict[str, Any] | None:
    for record in reversed(_read_jsonl(history_file)):
        if record.get("key") != key:
            continue
        hold = persisted_semantic_no_quorum_hold(
            record,
            lane=RAW_REPLAY_DECISION_LANE,
            epoch=epoch,
            authority=authority,
        )
        if hold is not None:
            return hold
    return None


def _current_raw_replay_authority(
    *,
    injected_reviewer: bool,
) -> tuple[dict[str, Any] | None, str | None]:
    return current_semantic_authority(
        RAW_REPLAY_DECISION_LANE,
        injected_reviewer=injected_reviewer,
    )


def _raw_replay_approval(
    row: dict[str, Any],
    *,
    evidence: dict[str, Any],
    review: dict[str, Any],
    authority: dict[str, Any],
) -> dict[str, Any]:
    return seal_semantic_artifact(
        {
            "schema_version": RAW_REPLAY_APPROVAL_SCHEMA_VERSION,
            "lane": RAW_REPLAY_DECISION_LANE,
            "key": row.get("key"),
            "raw": row.get("raw"),
            "attempt_id": row.get("attempt_id"),
            "job_id": row.get("job_id"),
            "raw_sha256": row.get("raw_sha256"),
            "observed_raw_sha256": evidence.get("current_raw_sha256"),
            "review_input_sha256": _canonical_sha256(evidence),
            "review": dict(review),
        },
        authority=authority,
        lane=RAW_REPLAY_DECISION_LANE,
    )


def _raw_replay_approval_error(
    row: dict[str, Any],
    *,
    approval: object,
    claims_file: Path,
    injected_reviewer: bool,
) -> str | None:
    if not isinstance(approval, Mapping):
        return "raw replay semantic approval is missing"
    review = approval.get("review")
    authority = approval.get("authority")
    current_evidence = _raw_replay_review_evidence(row, claims_file=claims_file)
    if (
        approval.get("schema_version") != RAW_REPLAY_APPROVAL_SCHEMA_VERSION
        or approval.get("lane") != RAW_REPLAY_DECISION_LANE
        or approval.get("key") != row.get("key")
        or approval.get("raw") != row.get("raw")
        or approval.get("attempt_id") != row.get("attempt_id")
        or approval.get("job_id") != row.get("job_id")
        or approval.get("raw_sha256") != row.get("raw_sha256")
        or approval.get("observed_raw_sha256")
        != current_evidence.get("current_raw_sha256")
        or not isinstance(review, Mapping)
    ):
        return "raw replay semantic approval identity is invalid"
    if (
        semantic_authority_shape_error(
            authority,
            lane=RAW_REPLAY_DECISION_LANE,
        )
        is not None
    ):
        return "raw replay semantic approval authority is invalid"
    if review.get("decision") != "safe_replay":
        return "raw replay semantic approval does not authorize replay"
    input_sha256 = approval.get("review_input_sha256")
    if not isinstance(input_sha256, str) or input_sha256 != _canonical_sha256(
        current_evidence
    ):
        return "raw replay semantic approval evidence changed"
    current, current_error = _current_raw_replay_authority(
        injected_reviewer=injected_reviewer
    )
    return (
        current_error
        or compare_semantic_authority(
            authority,
            current,
            lane=RAW_REPLAY_DECISION_LANE,
        )
        or semantic_verdict_authority_error(
            review,
            authority,
            lane=RAW_REPLAY_DECISION_LANE,
        )
    )


def _invalidate_safe_replay_authorization(
    row: dict[str, Any],
    *,
    now: datetime,
    reason: str,
) -> dict[str, Any]:
    updated = dict(row)
    updated["status"] = "indeterminate"
    updated["next_retry_at"] = None
    updated["next_frontier_retry_at"] = _iso(now)
    updated["last_error"] = reason
    updated["terminal_reason"] = "semantic replay authorization rejected"
    updated["frontier_authority_error"] = reason
    updated["updated_at"] = _iso(now)
    return updated


def _review_indeterminate_rows(
    rows: list[dict[str, Any]],
    *,
    claims_file: Path,
    history_file: Path,
    now: datetime,
    budget: Any | None,
    retry_delay_seconds: int,
    reviewer: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    injected_reviewer = reviewer is not None
    row: dict[str, Any] | None = None
    evidence: dict[str, Any] | None = None
    authority: dict[str, Any] | None = None
    authority_error: str | None = None
    held = 0
    for candidate in sorted(rows, key=_queue_sort_key):
        if _has_semantic_no_quorum_marker(candidate):
            candidate_evidence = _raw_replay_review_evidence(
                candidate, claims_file=claims_file
            )
            current_authority, current_error = _current_raw_replay_authority(
                injected_reviewer=injected_reviewer
            )
            persisted_hold = persisted_semantic_no_quorum_hold(
                candidate, lane=RAW_REPLAY_DECISION_LANE
            )
            if persisted_hold is None:
                _restore_raw_replay_semantic_hold(
                    candidate, hold=None, now=now, malformed=True
                )
                held += 1
                continue
            if current_error is not None or current_authority is None:
                _restore_raw_replay_semantic_hold(
                    candidate, hold=persisted_hold, now=now
                )
                held += 1
                continue
            hold_error = semantic_no_quorum_hold_error(
                persisted_hold,
                RAW_REPLAY_DECISION_LANE,
                epoch=_raw_replay_semantic_epoch(candidate_evidence),
                authority=current_authority,
            )
            if hold_error is None:
                _restore_raw_replay_semantic_hold(
                    candidate, hold=persisted_hold, now=now
                )
                held += 1
                continue
            if hold_error not in {
                "semantic hold epoch changed",
                "semantic hold authority changed",
            }:
                _restore_raw_replay_semantic_hold(
                    candidate, hold=None, now=now, malformed=True
                )
                held += 1
                continue
            historical_hold = _historical_raw_replay_hold(
                history_file,
                key=candidate.get("key"),
                epoch=_raw_replay_semantic_epoch(candidate_evidence),
                authority=current_authority,
            )
            if historical_hold is not None:
                _restore_raw_replay_semantic_hold(
                    candidate, hold=historical_hold, now=now
                )
                held += 1
                continue
            if not _frontier_due(candidate, now=now):
                continue
            row = candidate
            evidence = candidate_evidence
            authority = current_authority
            break
        if candidate.get("status") == "indeterminate" and _frontier_due(
            candidate, now=now
        ):
            row = candidate
            break
    if row is None:
        return {"reviewed": 0, "semantic_held": held, "budget_deferred": []}
    if budget is not None:
        allowed, reason = budget.consume("frontier")
        if not allowed:
            return {
                "reviewed": 0,
                "budget_deferred": [{"key": row.get("key"), "reason": reason}],
            }

    from chronovisor.decision.decision_lane_prompts import (
        build_raw_replay_reconciliation_prompt,
    )

    evidence = evidence or _raw_replay_review_evidence(row, claims_file=claims_file)
    prompt = build_raw_replay_reconciliation_prompt(evidence)
    if authority is None:
        authority, authority_error = _current_raw_replay_authority(
            injected_reviewer=injected_reviewer
        )
    if authority_error is None:
        if reviewer is not None:
            raw_review = reviewer(prompt, RAW_REPLAY_RECONCILIATION_SCHEMA)
        else:
            from chronovisor.decision.routine_review import run_structured_review

            raw_review = run_structured_review(
                prompt,
                RAW_REPLAY_RECONCILIATION_SCHEMA,
                repo_root=Path(__file__).resolve().parents[3],
                timeout=300,
                execute_patch=False,
                command_env="CHRONOVISOR_RAW_REPLAY_REVIEW_CMD",
                decision_lane=RAW_REPLAY_DECISION_LANE,
            )
        review = dict(raw_review) if isinstance(raw_review, Mapping) else {}
    else:
        review = {}
    attempts = _nonnegative_int(row.get("frontier_attempts")) + 1
    if (
        authority_error is None
        and authority is not None
        and is_local_semantic_no_quorum(review)
    ):
        hold: dict[str, Any] | None = None
        hold_error: str | None = None
        try:
            with decision_authority_lock():
                current, current_error = _current_raw_replay_authority(
                    injected_reviewer=injected_reviewer
                )
                hold_error = current_error or compare_semantic_authority(
                    authority,
                    current,
                    lane=RAW_REPLAY_DECISION_LANE,
                )
                current_evidence = _raw_replay_review_evidence(
                    row, claims_file=claims_file
                )
                if hold_error is None and _canonical_sha256(
                    current_evidence
                ) != _canonical_sha256(evidence):
                    hold_error = (
                        "raw replay review evidence changed before semantic hold"
                    )
                if hold_error is not None:
                    raise ValueError(hold_error)
                hold = build_semantic_no_quorum_hold(
                    RAW_REPLAY_DECISION_LANE,
                    _raw_replay_semantic_epoch(evidence),
                    authority,
                    review,
                )
                _restore_raw_replay_semantic_hold(row, hold=hold, now=now)
                row["frontier_attempts"] = attempts
                row["frontier_decision"] = "semantic_hold"
                row["frontier_failure"] = hold["frontier_failure"]
                row["frontier_authority_error"] = None
                row.pop("frontier_review_artifact", None)
                row.pop("semantic_hold_recheck_sha256", None)
                record = {
                    "ts": _iso(now),
                    "schema_version": SCHEMA_VERSION,
                    "key": row.get("key"),
                    "raw": row.get("raw"),
                    "status": "quarantined",
                    "frontier_attempts": attempts,
                    "frontier_decision": "semantic_hold",
                    "frontier_failure": hold["frontier_failure"],
                    "semantic_hold": hold,
                    "next_frontier_retry_at": None,
                    "terminal_reason": row.get("terminal_reason"),
                }
                _append_history(record, history_file)
        except (TypeError, ValueError) as exc:
            hold_error = str(exc)
        if hold is not None:
            return {
                "reviewed": 1,
                "semantic_held": held + 1,
                "record": record,
                "budget_deferred": [],
            }
        authority_error = hold_error or "semantic no-quorum hold provenance is invalid"
    decision = str(review.get("decision") or "needs_retry")
    confidence_raw = review.get("confidence")
    confidence_valid = (
        not isinstance(confidence_raw, bool)
        and isinstance(confidence_raw, (int, float))
        and 0.0 <= float(confidence_raw) <= 1.0
    )
    confidence = float(confidence_raw) if confidence_valid else 0.0
    reason = str(
        review.get("reason") or review.get("summary") or "invalid frontier result"
    )
    if authority_error is not None:
        decision = "needs_retry"
        reason = authority_error
    elif decision not in {
        "accept_processed",
        "safe_replay",
        "quarantine",
        "needs_retry",
    }:
        authority_error = "local consensus returned an invalid replay decision"
        decision = "needs_retry"
        reason = authority_error
    elif not confidence_valid:
        authority_error = "local consensus returned invalid confidence metadata"
        decision = "needs_retry"
        reason = authority_error

    if authority_error is None:
        assert authority is not None
        current, current_error = _current_raw_replay_authority(
            injected_reviewer=injected_reviewer
        )
        authority_error = (
            semantic_verdict_authority_error(
                review,
                authority,
                lane=RAW_REPLAY_DECISION_LANE,
            )
            or current_error
            or compare_semantic_authority(
                authority,
                current,
                lane=RAW_REPLAY_DECISION_LANE,
            )
        )
        if authority_error is not None:
            decision = "needs_retry"
            reason = authority_error

    human_required = is_human_required_result(review)
    approval: dict[str, Any] | None = None
    if authority_error is None:
        assert authority is not None
        approval = _raw_replay_approval(
            row,
            evidence=evidence,
            review=review,
            authority=authority,
        )

    # Every transition caused by a semantic verdict is serialized with an
    # adopted-authority update.  The comparison is intentionally repeated
    # here after the immediate post-review check above.
    with decision_authority_lock():
        if authority_error is None:
            assert authority is not None
            current, current_error = _current_raw_replay_authority(
                injected_reviewer=injected_reviewer
            )
            authority_error = (
                current_error
                or compare_semantic_authority(
                    authority,
                    current,
                    lane=RAW_REPLAY_DECISION_LANE,
                )
                or semantic_verdict_authority_error(
                    review,
                    authority,
                    lane=RAW_REPLAY_DECISION_LANE,
                )
            )
            if authority_error is None and _canonical_sha256(
                _raw_replay_review_evidence(row, claims_file=claims_file)
            ) != approval.get("review_input_sha256"):
                authority_error = "raw replay review evidence changed before effect"
        if authority_error is not None:
            decision = "needs_retry"
            human_required = False
            reason = authority_error
            row["status"] = "indeterminate"
            row["next_retry_at"] = None
            delay = max(0, retry_delay_seconds) * (2 ** max(0, attempts - 1))
            row["next_frontier_retry_at"] = _iso(now + timedelta(seconds=delay))
            row["last_error"] = reason
            row["frontier_authority_error"] = reason
            row.pop("frontier_review_artifact", None)
            if _has_semantic_no_quorum_marker(row) and authority is not None:
                row["semantic_hold_recheck_sha256"] = canonical_sha256(
                    {
                        "epoch": _raw_replay_semantic_epoch(evidence),
                        "authority": authority,
                    }
                )
        else:
            assert approval is not None
            row.pop("semantic_hold", None)
            row.pop("last_failure_class", None)
            row.pop("semantic_hold_recheck_sha256", None)
            row["frontier_review_artifact"] = approval
            row["frontier_authority_error"] = None
            if human_required:
                row["status"] = "human_required"
                row["human_required_at"] = _iso(now)
                row["terminal_reason"] = reason
                row["next_frontier_retry_at"] = None
            elif decision == "accept_processed":
                _mark_completed(
                    row,
                    evidence="local_consensus_indeterminate_review",
                    now=now,
                    status="completed_partial",
                )
                row["terminal_reason"] = reason
            elif decision == "safe_replay":
                row["status"] = "pending"
                row["next_retry_at"] = None
                row["next_frontier_retry_at"] = None
                row["last_error"] = None
                row["terminal_reason"] = reason
            elif decision == "quarantine" or attempts >= MAX_FRONTIER_ATTEMPTS:
                row["status"] = "quarantined"
                row["quarantined_at"] = _iso(now)
                row["next_frontier_retry_at"] = None
                row["terminal_reason"] = reason
            else:
                row["status"] = "indeterminate"
                delay = max(0, retry_delay_seconds) * (2 ** max(0, attempts - 1))
                row["next_frontier_retry_at"] = _iso(now + timedelta(seconds=delay))
                row["last_error"] = reason

        row["frontier_attempts"] = attempts
        row["frontier_decision"] = decision
        row["frontier_failure"] = review.get("frontier_failure")
        row["updated_at"] = _iso(now)
        record = {
            "ts": _iso(now),
            "schema_version": SCHEMA_VERSION,
            "key": row.get("key"),
            "raw": row.get("raw"),
            "status": row.get("status"),
            "attempts": row.get("attempts"),
            "frontier_attempts": attempts,
            "frontier_decision": decision,
            "frontier_confidence": confidence,
            "human_required": human_required,
            "frontier_failure": review.get("frontier_failure"),
            "frontier_review_artifact": row.get("frontier_review_artifact"),
            "frontier_authority_error": row.get("frontier_authority_error"),
            "next_frontier_retry_at": row.get("next_frontier_retry_at"),
            "human_required_at": row.get("human_required_at"),
            "error": row.get("last_error"),
            "terminal_reason": row.get("terminal_reason"),
        }
        _append_history(record, history_file)
    return {"reviewed": 1, "record": record, "budget_deferred": []}


def _mark_not_needed(
    row: dict[str, Any],
    *,
    now: datetime,
    reason: str = "legacy explicit migration has no current autonomous signal",
) -> None:
    row["status"] = "not_needed"
    row["next_retry_at"] = None
    row["last_error"] = None
    row["not_needed_at"] = row.get("not_needed_at") or _iso(now)
    row["terminal_reason"] = reason
    row["updated_at"] = _iso(now)


def _retire_retracted_row(row: dict[str, Any], *, now: datetime) -> None:
    status = str(row.get("status") or "pending")
    if status in TERMINAL_STATUSES or status == "running":
        return
    if _row_is_retracted(row):
        _mark_not_needed(
            row,
            now=now,
            reason="raw frontmatter marks capture as retracted",
        )


def _reactivate_not_needed(row: dict[str, Any], *, now: datetime) -> dict[str, Any]:
    reactivated = dict(row)
    reactivated["status"] = "pending"
    reactivated["next_retry_at"] = None
    reactivated["last_error"] = None
    reactivated["terminal_reason"] = None
    reactivated["reactivated_at"] = _iso(now)
    reactivated["updated_at"] = _iso(now)
    return reactivated


def _coerce_exhausted(row: dict[str, Any], *, now: datetime) -> None:
    if (
        row.get("status") not in RETRYABLE_STATUSES
        or _nonnegative_int(row.get("attempts")) < MAX_ATTEMPTS
    ):
        return
    row["status"] = "quarantined"
    row["next_retry_at"] = None
    row["quarantined_at"] = row.get("quarantined_at") or _iso(now)
    row["updated_at"] = _iso(now)


def _build_queue_unlocked(
    *,
    since: str = "",
    limit: int = 0,
    path: Path,
    dry_run: bool = False,
    include_migration: bool = True,
    include_auto_signals: bool = True,
) -> dict[str, Any]:
    now = _now()
    semantic_deferred_raws = _active_terminal_semantic_deferred_raw_names()
    deferred_statuses = _active_operational_deferred_raw_statuses()
    history_states = _history_lifecycle_states()
    existing_by_key: dict[str, dict[str, Any]] = {}
    for raw_row in _read_jsonl(path):
        row = _normalize_queue_row(raw_row, now=now)
        if row is not None:
            _retire_retracted_row(row, now=now)
            key = str(row["key"])
            current = existing_by_key.get(key)
            existing_by_key[key] = (
                row if current is None else _merge_durable_rows(current, row, now=now)
            )
    for key, row in existing_by_key.items():
        existing_by_key[key] = _merge_history_lifecycle(row, history_states.get(key))
        _coerce_exhausted(existing_by_key[key], now=now)
        _resume_due_autonomous_terminal(
            existing_by_key[key],
            now=now,
            semantic_deferred_raws=semantic_deferred_raws,
        )

    incoming: list[dict[str, Any]] = []
    if include_auto_signals:
        incoming.extend(_memory_integrity_candidates(now=now))
    if include_migration:
        incoming.extend(_explicit_migration_candidates(since=since, now=now))
    if include_auto_signals:
        incoming.extend(_ingest_failure_candidates(now=now))

    incoming_by_key: dict[str, dict[str, Any]] = {}
    for row in incoming:
        key = str(row["key"])
        row = _merge_history_lifecycle(row, history_states.get(key))
        _coerce_exhausted(row, now=now)
        _resume_due_autonomous_terminal(
            row,
            now=now,
            semantic_deferred_raws=semantic_deferred_raws,
        )
        current = incoming_by_key.get(key)
        incoming_by_key[key] = (
            row if current is None else _merge_durable_rows(current, row, now=now)
        )

    auto_signal_keys = {
        key
        for key, row in incoming_by_key.items()
        if set(row.get("sources", [])) & AUTO_SIGNAL_SOURCES
    }
    if not include_migration and include_auto_signals:
        for key, row in existing_by_key.items():
            if (
                row.get("status") == "pending"
                and set(row.get("sources", [])) == {"explicit_migration"}
                and key not in auto_signal_keys
            ) or (
                row.get("status") in RETRYABLE_STATUSES
                and "ingest_failure" in set(row.get("sources", []))
                and row.get("reasons")
                and all(
                    "not-in-top-results" in str(reason).casefold()
                    or str(reason).casefold().startswith("explicit migration")
                    for reason in row.get("reasons", [])
                )
                and any(
                    "not-in-top-results" in str(reason).casefold()
                    for reason in row.get("reasons", [])
                )
                and key not in auto_signal_keys
            ):
                _mark_not_needed(row, now=now)

    completed = _completed_replays()
    for key, row in existing_by_key.items():
        if row.get("status") == "completed":
            completed.setdefault(
                key, str(row.get("completion_evidence") or "replay_queue")
            )
    for key, evidence in completed.items():
        row = existing_by_key.get(key)
        if row is None:
            continue
        _mark_completed(row, evidence=evidence, now=now)

    actionable: list[dict[str, Any]] = []
    skipped_completed = 0
    skipped_terminal = 0
    skipped_semantic_deferred = 0
    skipped_operational_deferred = 0
    for incoming_row in sorted(incoming_by_key.values(), key=_queue_sort_key):
        key = str(incoming_row["key"])
        current = existing_by_key.get(key)
        incoming_sources = set(incoming_row.get("sources", []))
        reactivation_signal = bool(incoming_sources & AUTO_SIGNAL_SOURCES) or (
            include_migration and "explicit_migration" in incoming_sources
        )
        if incoming_row.get("status") == "not_needed" and reactivation_signal:
            incoming_row = _reactivate_not_needed(incoming_row, now=now)
        if (
            current is not None
            and current.get("status") == "not_needed"
            and reactivation_signal
        ):
            current = _reactivate_not_needed(current, now=now)
        combined = (
            incoming_row
            if current is None
            else _merge_durable_rows(current, incoming_row, now=now)
        )
        _resume_due_autonomous_terminal(
            combined,
            now=now,
            semantic_deferred_raws=semantic_deferred_raws,
        )
        if _raw_name(combined) in semantic_deferred_raws:
            if deferred_statuses.get(_raw_name(combined)) == "semantic_no_quorum":
                skipped_semantic_deferred += 1
            else:
                skipped_operational_deferred += 1
            if current is not None:
                existing_by_key[key] = combined
            continue
        if key in completed or combined.get("status") == "completed":
            skipped_completed += 1
            if current is not None:
                _mark_completed(
                    combined, evidence=completed.get(key, "replay_queue"), now=now
                )
                existing_by_key[key] = combined
            continue
        if combined.get("status") in TERMINAL_STATUSES:
            skipped_terminal += 1
            existing_by_key[key] = combined
            continue
        actionable.append(combined)

    ranked_incoming = actionable[:limit] if limit > 0 else actionable

    candidate_keys: list[str] = []
    for incoming_row in ranked_incoming:
        key = str(incoming_row["key"])
        existing_by_key[key] = incoming_row
        candidate_keys.append(key)

    rows = sorted(existing_by_key.values(), key=_queue_sort_key)
    if not dry_run:
        _atomic_write_queue(path, rows)
    status_counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status") or "pending")
        status_counts[status] = status_counts.get(status, 0) + 1
    return {
        "status": "dry_run" if dry_run else "ok",
        "dry_run": dry_run,
        "queue": str(path),
        "count": len(rows),
        "candidates": len(ranked_incoming),
        "candidate_keys": candidate_keys,
        "skipped_completed": skipped_completed,
        "skipped_terminal": skipped_terminal,
        "skipped_semantic_deferred": skipped_semantic_deferred,
        "skipped_operational_deferred": skipped_operational_deferred,
        "status_counts": dict(sorted(status_counts.items())),
    }


def build_queue(
    *,
    since: str = "",
    limit: int = 0,
    path: Path | None = None,
    dry_run: bool = False,
    include_migration: bool = True,
    include_auto_signals: bool = True,
) -> dict[str, Any]:
    """Merge current replay signals into the durable queue.

    ``include_migration`` defaults to ``True`` for compatibility with the
    historical CLI.  Autonomous callers should pass ``False`` so a nightly
    refresh consumes only integrity/failure signals rather than replaying the
    oldest raw captures forever.  Queue refresh and execution share a lock;
    dry-run deliberately avoids even creating that lock file.
    """
    target = path if path is not None else QUEUE_FILE
    kwargs = {
        "since": since,
        "limit": _nonnegative_int(limit),
        "path": target,
        "dry_run": dry_run,
        "include_migration": include_migration,
        "include_auto_signals": include_auto_signals,
    }
    if dry_run:
        return _build_queue_unlocked(**kwargs)
    with _queue_lock(target):
        return _build_queue_unlocked(**kwargs)


def _eligible(row: dict[str, Any], *, now: datetime) -> bool:
    status = str(row.get("status") or "pending")
    if status not in RETRYABLE_STATUSES:
        return False
    if _row_is_retracted(row):
        return False
    if _nonnegative_int(row.get("attempts")) >= MAX_ATTEMPTS:
        return False
    retry_at = _parse_dt(row.get("next_retry_at"))
    if retry_at is None:
        return True
    compare_now = now
    if retry_at.tzinfo is None and compare_now.tzinfo is not None:
        compare_now = compare_now.replace(tzinfo=None)
    elif retry_at.tzinfo is not None and compare_now.tzinfo is None:
        retry_at = retry_at.replace(tzinfo=None)
    return retry_at <= compare_now


def _select_bounded(
    rows: list[dict[str, Any]],
    *,
    now: datetime,
    max_runs: int,
    max_bytes: int,
    eligible_keys: set[str] | None = None,
    eligible_sources: set[str] | frozenset[str] | None = None,
    semantic_deferred_raws: frozenset[str] = frozenset(),
    initial_used_bytes: int = 0,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    used_bytes = min(max_bytes, _nonnegative_int(initial_used_bytes))
    for row in sorted(rows, key=_queue_sort_key):
        if len(selected) >= max_runs:
            break
        if _raw_name(row) in semantic_deferred_raws:
            continue
        if not _matches_eligibility_scope(
            row,
            eligible_keys=eligible_keys,
            eligible_sources=eligible_sources,
        ):
            continue
        if not _eligible(row, now=now):
            continue
        size = _nonnegative_int(row.get("bytes"))
        # An individually oversized raw cannot ever fit this budget. Select it
        # as a zero-byte preflight attempt so retry/quarantine state advances
        # instead of leaving it pending forever.
        if size > max_bytes:
            selected.append(row)
            continue
        if used_bytes + size > max_bytes:
            continue
        selected.append(row)
        used_bytes += size
    return selected


def _matches_eligibility_scope(
    row: Mapping[str, Any],
    *,
    eligible_keys: set[str] | None,
    eligible_sources: set[str] | frozenset[str] | None,
) -> bool:
    """Apply the public key/source union boundary to every queue mutation."""

    checks: list[bool] = []
    if eligible_keys is not None:
        checks.append(row.get("key") in eligible_keys)
    if eligible_sources is not None:
        sources = row.get("sources")
        checks.append(
            isinstance(sources, list) and bool(set(sources) & eligible_sources)
        )
    # Multiple selectors are a union: current signal keys plus every
    # previously queued row in the same autonomous lane.
    return not checks or any(checks)


def _job_status(job: object) -> str:
    status = getattr(job, "status", None)
    return str(getattr(status, "value", status or "missing"))


def _failure_text(job: object, status: str) -> str:
    result = getattr(job, "result", None)
    error = getattr(job, "error", None)
    if error:
        return str(error)
    if isinstance(result, dict) and result.get("failed_ops"):
        return f"replay completed with {len(result['failed_ops'])} failed operation(s)"
    return f"replay job ended with status {status}"


def _operational_deferred_candidate_result(
    row: dict[str, Any],
    *,
    now: datetime,
    job_id: str | None,
    attempt_started: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Undo a prelaunch marker when a failure hold wins the final race."""

    if job_id:
        job_store.update(
            job_id,
            status=JobStatus.FAILED,
            completed_at=_iso(now),
            error="operational failure hold published before replay launch",
        )
    updated = dict(row)
    updated["status"] = "pending"
    updated["attempts"] = max(
        0,
        _nonnegative_int(row.get("attempts")) - (1 if attempt_started else 0),
    )
    updated["job_id"] = None
    updated["next_retry_at"] = None
    updated["last_error"] = None
    updated["updated_at"] = _iso(now)
    for field in ("attempt_id", "started_at"):
        updated.pop(field, None)
    raw_bytes = _nonnegative_int(row.get("bytes"))
    return updated, {
        "ts": _iso(now),
        "schema_version": SCHEMA_VERSION,
        "key": updated.get("key"),
        "raw": updated.get("raw"),
        "path": updated.get("path"),
        "status": "pending",
        "attempts": updated["attempts"],
        "job_id": job_id,
        "job_status": "operational_deferred",
        "pages_created": [],
        "pages_updated": [],
        "failed_ops": [],
        "bytes": raw_bytes,
        "charged_bytes": 0,
        "error": None,
        "operational_deferred": True,
        "defer_reason": "operational_failure_hold",
    }


def _publish_semantic_no_quorum_defer(
    row: dict[str, Any],
    *,
    path: Path,
    error: str,
    job_id: str | None,
    raw_text: str | None = None,
) -> dict[str, Any] | None:
    """Publish an authority- and byte-bound terminal semantic defer.

    A semantic split is a valid terminal outcome for the current adopted
    authority, not an operational replay failure.  Publication is allowed only
    when the error names the *current* adopted artifact and the immutable raw
    still matches the durable attempt marker.  This also makes migration of
    legacy failed/quarantined rows safe without another model call.
    """

    from chronovisor.ingest.failure_supervisor import (
        SEMANTIC_NO_QUORUM_DEFER_REASON,
        SEMANTIC_NO_QUORUM_FAILURE_CLASS,
        classify_failure,
        current_adopted_authority_sha256,
        record_semantic_no_quorum_defer_unless_operational_hold,
    )

    failure = classify_failure(error)
    authority_sha256 = failure.authority_artifact_sha256
    if (
        failure.failure_class != SEMANTIC_NO_QUORUM_FAILURE_CLASS
        or not isinstance(authority_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", authority_sha256) is None
    ):
        return None
    expected_raw_sha256 = row.get("raw_sha256")
    if (
        not isinstance(expected_raw_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", expected_raw_sha256) is None
        or not path.is_file()
    ):
        return None
    try:
        if (
            hashlib.sha256(_logical_raw_bytes(_raw_name(row), path)).hexdigest()
            != expected_raw_sha256
        ):
            return None
        source_text = (
            raw_text
            if raw_text is not None
            else _logical_raw_bytes(_raw_name(row), path).decode("utf-8")
        )
    except (OSError, UnicodeDecodeError):
        return None

    # Router adoption and terminal packet publication form one semantic
    # authority transaction.  A concurrent artifact change must release the
    # old split instead of publishing a stale hold.
    with decision_authority_lock():
        if current_adopted_authority_sha256() != authority_sha256:
            return None
        supervision = record_semantic_no_quorum_defer_unless_operational_hold(
            raw_path=path,
            error=error,
            job_id=job_id,
            raw_text=source_text,
            related_raw_paths=(),
        )
    if supervision is None or supervision.terminal_deferred is not True:
        return None
    return {
        "reason": SEMANTIC_NO_QUORUM_DEFER_REASON,
        "authority_sha256": authority_sha256,
        "packet_path": supervision.packet_path,
        "error": error,
    }


def _preview_semantic_no_quorum_defer(
    row: Mapping[str, Any],
    *,
    path: Path,
    error: str,
) -> dict[str, Any] | None:
    """Prove that a legacy error is publishable without writing its packet."""

    from chronovisor.ingest.failure_supervisor import (
        SEMANTIC_NO_QUORUM_DEFER_REASON,
        SEMANTIC_NO_QUORUM_FAILURE_CLASS,
        classify_failure,
        current_adopted_authority_sha256,
    )

    failure = classify_failure(error)
    authority_sha256 = failure.authority_artifact_sha256
    expected_raw_sha256 = row.get("raw_sha256")
    if (
        failure.failure_class != SEMANTIC_NO_QUORUM_FAILURE_CLASS
        or not isinstance(authority_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", authority_sha256) is None
        or not isinstance(expected_raw_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", expected_raw_sha256) is None
        or not path.is_file()
    ):
        return None
    try:
        if (
            hashlib.sha256(_logical_raw_bytes(_raw_name(row), path)).hexdigest()
            != expected_raw_sha256
        ):
            return None
    except OSError:
        return None
    if current_adopted_authority_sha256() != authority_sha256:
        return None
    return {
        "reason": SEMANTIC_NO_QUORUM_DEFER_REASON,
        "authority_sha256": authority_sha256,
        "packet_path": None,
        "error": error,
        "published": False,
    }


def _cheap_semantic_no_quorum_candidate(
    row: Mapping[str, Any],
    *,
    active_reason: str | None,
) -> dict[str, Any] | None:
    """Classify a legacy semantic row without reading the immutable raw.

    Full source hashing and packet verification are deliberately deferred until
    after both the public replay bound and the shared ``CycleBudget`` approve
    the candidate.  This cheap phase may inspect queue metadata, the adopted
    authority hash, and file existence only.
    """

    from chronovisor.ingest.failure_supervisor import (
        SEMANTIC_NO_QUORUM_DEFER_REASON,
        SEMANTIC_NO_QUORUM_FAILURE_CLASS,
        classify_failure,
        current_adopted_authority_sha256,
    )

    if active_reason == SEMANTIC_NO_QUORUM_DEFER_REASON:
        return {"published": True}
    if active_reason is not None or row.get("status") not in {
        "failed",
        "quarantined",
    }:
        return None
    error = row.get("last_error")
    if not isinstance(error, str):
        return None
    failure = classify_failure(error)
    authority_sha256 = failure.authority_artifact_sha256
    expected_raw_sha256 = row.get("raw_sha256")
    raw_name = _raw_name(dict(row))
    path = _resolve_raw_path(raw_name, row.get("path"))
    if (
        failure.failure_class != SEMANTIC_NO_QUORUM_FAILURE_CLASS
        or not isinstance(authority_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", authority_sha256) is None
        or current_adopted_authority_sha256() != authority_sha256
        or not isinstance(expected_raw_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", expected_raw_sha256) is None
        or path is None
    ):
        return None
    return {
        "published": False,
        "reason": SEMANTIC_NO_QUORUM_DEFER_REASON,
        "authority_sha256": authority_sha256,
        "packet_path": None,
        "error": error,
    }


def _apply_semantic_no_quorum_defer(
    row: dict[str, Any],
    *,
    evidence: Mapping[str, Any],
    now: datetime,
) -> dict[str, Any]:
    """Convert replay failure accounting into a model-free supervisor hold."""

    updated = dict(row)
    prior_attempts = max(
        _nonnegative_int(updated.get("semantic_defer_prior_attempts")),
        _nonnegative_int(updated.get("attempts")),
    )
    updated["status"] = "pending"
    # These attempts were all observations of one authority-bound semantic
    # split, not useful replay failures.  A future adopted authority epoch gets
    # a fresh bounded attempt budget after the supervisor releases the hold.
    updated["attempts"] = 0
    updated["next_retry_at"] = None
    updated["last_error"] = str(evidence.get("error") or "semantic no quorum")
    updated["terminal_reason"] = "semantic_no_quorum"
    updated["recovery_kind"] = "semantic_no_quorum_terminal_defer"
    updated["semantic_deferred_at"] = _iso(now)
    updated["semantic_defer_authority_sha256"] = evidence.get("authority_sha256")
    updated["semantic_defer_packet"] = evidence.get("packet_path")
    updated["semantic_defer_prior_attempts"] = prior_attempts
    updated["semantic_defer_error"] = updated["last_error"]
    updated["semantic_defer_job_id"] = updated.get("job_id") or evidence.get("job_id")
    updated["semantic_defer_attempt_id"] = updated.get("attempt_id")
    updated["semantic_defer_started_at"] = updated.get("started_at")
    updated["job_id"] = None
    updated.pop("attempt_id", None)
    updated.pop("started_at", None)
    updated["frontier_attempts"] = 0
    updated["frontier_decision"] = None
    updated["next_frontier_retry_at"] = None
    updated["frontier_authority_error"] = None
    updated.pop("frontier_review_artifact", None)
    updated.pop("frontier_authorization_consumed_at", None)
    updated.pop("quarantined_at", None)
    updated.pop("human_required_at", None)
    updated["updated_at"] = _iso(now)
    return updated


def _clear_released_semantic_defer(
    row: dict[str, Any],
    *,
    now: datetime,
) -> None:
    """Retire old-authority defer metadata at the next launch boundary."""

    if row.get("recovery_kind") != "semantic_no_quorum_terminal_defer":
        return
    for field in tuple(row):
        if field.startswith("semantic_defer_") or field == "semantic_deferred_at":
            row.pop(field, None)
    row.pop("recovery_kind", None)
    row["terminal_reason"] = None
    row["reactivated_at"] = _iso(now)
    row["quarantine_resumed_at"] = row.get("quarantine_resumed_at") or _iso(now)


def _active_semantic_defer_packet_evidence(
    row: Mapping[str, Any],
    *,
    active_raws: frozenset[str],
) -> dict[str, Any] | None:
    """Recover a queue transition from an already durable supervisor packet."""

    raw_name = _raw_name(dict(row))
    if not raw_name or raw_name not in active_raws:
        return None
    from chronovisor.ingest.failure_supervisor import (
        SEMANTIC_NO_QUORUM_DEFER_REASON,
        current_adopted_authority_epoch,
        semantic_defer_packet_records,
    )

    expected_sha256 = row.get("raw_sha256")
    if (
        not isinstance(expected_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None
    ):
        return None
    current_authority_epoch = current_adopted_authority_epoch()
    for packet_path, packet, packet_raws in reversed(
        semantic_defer_packet_records(verify_sources=True)
    ):
        if raw_name not in packet_raws:
            continue
        authority_epoch = packet.get(
            "authority_epoch",
            packet.get("authority_artifact_sha256"),
        )
        if (
            not isinstance(authority_epoch, str)
            or re.fullmatch(r"[0-9a-f]{64}", authority_epoch) is None
            or (
                current_authority_epoch is not None
                and current_authority_epoch != authority_epoch
            )
        ):
            continue
        source_evidence = next(
            (
                source
                for source in packet.get("source_raws", [])
                if isinstance(source, dict) and source.get("filename") == raw_name
            ),
            None,
        )
        if (
            not isinstance(source_evidence, dict)
            or source_evidence.get("sha256") != expected_sha256
        ):
            continue
        error = packet.get("error")
        if not isinstance(error, str) or not error:
            continue
        return {
            "reason": SEMANTIC_NO_QUORUM_DEFER_REASON,
            "authority_sha256": authority_epoch,
            "packet_path": str(packet_path),
            "error": error,
            "job_id": packet.get("job_id"),
        }
    return None


def _plan_legacy_semantic_no_quorum_rows(
    rows: list[dict[str, Any]],
    *,
    active_deferred_statuses: Mapping[str, str],
    max_runs: int,
    max_bytes: int,
    eligible_keys: set[str] | None,
    eligible_sources: set[str] | frozenset[str] | None,
    completed_keys: frozenset[str],
) -> tuple[list[tuple[dict[str, Any], dict[str, Any]]], frozenset[str]]:
    """Build a bounded, metadata-only legacy migration plan.

    No immutable raw is hashed or read here.  Expensive proof happens only for
    rows admitted by ``max_runs``/``max_bytes`` and the shared cycle budget.
    """

    planned: list[tuple[dict[str, Any], dict[str, Any]]] = []
    candidate_keys: set[str] = set()
    used_bytes = 0
    for row in sorted(rows, key=_queue_sort_key):
        raw_name = _raw_name(row)
        if (
            row.get("recovery_kind") == "semantic_no_quorum_terminal_defer"
            or str(row.get("key") or "") in completed_keys
            or not _matches_eligibility_scope(
                row,
                eligible_keys=eligible_keys,
                eligible_sources=eligible_sources,
            )
        ):
            continue
        raw_bytes = _nonnegative_int(row.get("bytes"))
        active_reason = active_deferred_statuses.get(raw_name)
        if active_reason is not None and active_reason != "semantic_no_quorum":
            # A newer operational repair hold always outranks stale semantic
            # replay history. Replacing it would release an unrepaired raw on
            # the next authority change.
            continue
        candidate = _cheap_semantic_no_quorum_candidate(
            row,
            active_reason=active_reason,
        )
        if candidate is None:
            continue
        key = str(row.get("key") or "")
        candidate_keys.add(key)
        if len(planned) >= max_runs:
            continue
        if raw_bytes <= max_bytes and used_bytes + raw_bytes > max_bytes:
            continue
        planned.append((row, candidate))
        if raw_bytes <= max_bytes:
            used_bytes += raw_bytes
    return planned, frozenset(candidate_keys)


def _semantic_defer_budget_error(
    *,
    budget: Any | None,
    charge_bytes: int,
    consume: bool = True,
) -> str | None:
    """Conservatively charge one model-free state transition to CycleBudget."""

    if budget is None:
        return None
    bytes_allowed, bytes_reason = (
        budget.can_consume("raw_bytes", charge_bytes) if charge_bytes else (True, None)
    )
    mutation_allowed, mutation_reason = budget.can_consume("mutation")
    if not bytes_allowed or not mutation_allowed:
        return str(bytes_reason if not bytes_allowed else mutation_reason)
    if not consume:
        return None
    if charge_bytes:
        bytes_allowed, bytes_reason = budget.consume("raw_bytes", charge_bytes)
        if not bytes_allowed:
            return str(bytes_reason)
    mutation_allowed, mutation_reason = budget.consume("mutation")
    if not mutation_allowed:
        return str(mutation_reason)
    return None


def _reconcile_legacy_semantic_no_quorum_rows(
    planned: list[tuple[dict[str, Any], dict[str, Any]]],
    *,
    now: datetime,
    max_bytes: int,
    budget: Any | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], frozenset[str]]:
    """Apply a read-only plan only after every shared budget guard succeeds."""

    reconciled: list[dict[str, Any]] = []
    budget_deferred: list[dict[str, Any]] = []
    verification_failed: set[str] = set()
    for row, preview in planned:
        raw_name = _raw_name(row)
        raw_bytes = _nonnegative_int(row.get("bytes"))
        key = str(row.get("key") or "")
        if raw_bytes > max_bytes:
            budget_deferred.append(
                {
                    "key": row.get("key"),
                    "raw": row.get("raw"),
                    "reason": "raw_replay_byte_budget_exhausted",
                    "action": "semantic_defer_reconcile",
                }
            )
            continue
        charged_bytes = raw_bytes
        budget_error = _semantic_defer_budget_error(
            budget=budget,
            charge_bytes=charged_bytes,
        )
        if budget_error is not None:
            budget_deferred.append(
                {
                    "key": row.get("key"),
                    "raw": row.get("raw"),
                    "reason": budget_error,
                    "action": "semantic_defer_reconcile",
                }
            )
            continue
        evidence: dict[str, Any] | None = None
        if preview.get("published") is True:
            # Re-resolve the active packet while holding the authority lock.
            # A packet released by an authority change between plan and apply
            # must not leave one spurious terminal-defer cycle in the queue.
            with decision_authority_lock():
                statuses = _scoped_operational_deferred_raw_statuses([row])
                if statuses.get(raw_name) == "semantic_no_quorum":
                    evidence = _active_semantic_defer_packet_evidence(
                        row,
                        active_raws=frozenset(statuses),
                    )
        else:
            error = row.get("last_error")
            path = _resolve_raw_path(raw_name, row.get("path"))
            if not isinstance(error, str) or path is None:
                verification_failed.add(key)
                continue
            evidence = _publish_semantic_no_quorum_defer(
                row,
                path=path,
                error=error,
                job_id=str(row.get("job_id") or "") or None,
            )
        if evidence is None:
            verification_failed.add(key)
            continue
        prior_status = str(row.get("status") or "")
        prior_attempts = _nonnegative_int(row.get("attempts"))
        updated = _apply_semantic_no_quorum_defer(
            row,
            evidence=evidence,
            now=now,
        )
        row.clear()
        row.update(updated)
        reconciled.append(
            {
                "key": row.get("key"),
                "raw": row.get("raw"),
                "prior_status": prior_status,
                "prior_attempts": prior_attempts,
                "reason": evidence.get("reason"),
                "authority_sha256": evidence.get("authority_sha256"),
                "packet_path": evidence.get("packet_path"),
                "bytes": raw_bytes,
                "charged_bytes": charged_bytes,
            }
        )
    return reconciled, budget_deferred, frozenset(verification_failed)


def _preview_legacy_semantic_no_quorum_rows(
    planned: list[tuple[dict[str, Any], dict[str, Any]]],
    *,
    max_bytes: int,
    budget: Any | None,
) -> tuple[
    list[tuple[dict[str, Any], dict[str, Any]]],
    list[dict[str, Any]],
    frozenset[str],
]:
    """Verify a semantic plan without writes or budget consumption."""

    verified: list[tuple[dict[str, Any], dict[str, Any]]] = []
    budget_deferred: list[dict[str, Any]] = []
    verification_failed: set[str] = set()
    for row, preview in planned:
        raw_name = _raw_name(row)
        raw_bytes = _nonnegative_int(row.get("bytes"))
        key = str(row.get("key") or "")
        if raw_bytes > max_bytes:
            budget_deferred.append(
                {
                    "key": row.get("key"),
                    "raw": row.get("raw"),
                    "reason": "raw_replay_byte_budget_exhausted",
                    "action": "semantic_defer_reconcile",
                }
            )
            continue
        budget_error = _semantic_defer_budget_error(
            budget=budget,
            charge_bytes=raw_bytes,
            consume=False,
        )
        if budget_error is not None:
            budget_deferred.append(
                {
                    "key": row.get("key"),
                    "raw": row.get("raw"),
                    "reason": budget_error,
                    "action": "semantic_defer_reconcile",
                }
            )
            continue
        evidence: dict[str, Any] | None = None
        if preview.get("published") is True:
            with decision_authority_lock():
                statuses = _scoped_operational_deferred_raw_statuses([row])
                if statuses.get(raw_name) == "semantic_no_quorum":
                    evidence = _active_semantic_defer_packet_evidence(
                        row,
                        active_raws=frozenset(statuses),
                    )
        else:
            error = row.get("last_error")
            path = _resolve_raw_path(raw_name, row.get("path"))
            if isinstance(error, str) and path is not None:
                evidence = _preview_semantic_no_quorum_defer(
                    row,
                    path=path,
                    error=error,
                )
        if evidence is None:
            verification_failed.add(key)
            continue
        verified.append((row, evidence))
    return verified, budget_deferred, frozenset(verification_failed)


def _reconcile_pending_failure_resets(
    rows: list[dict[str, Any]],
    *,
    completion_states: Mapping[str, Mapping[str, Any]],
    completions_file: Path,
    now: datetime,
    budget: Any | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool]:
    """Finish supervisor cleanup proven pending by the completion journal."""

    from chronovisor.ingest.failure_supervisor import reset_raw_failure

    reconciled: list[dict[str, Any]] = []
    budget_deferred: list[dict[str, Any]] = []
    queue_dirty = False
    for row in rows:
        key = str(row.get("key") or "")
        completion = completion_states.get(key)
        if not isinstance(completion, Mapping):
            continue
        journal_pending = completion.get("failure_reset_pending") is True
        row_pending = row.get("failure_reset_pending") is True
        if not journal_pending and not row_pending:
            continue
        completion_sha256 = completion.get("raw_sha256")
        row_sha256 = row.get("raw_sha256")
        if (
            isinstance(completion_sha256, str)
            and isinstance(row_sha256, str)
            and completion_sha256 != row_sha256
        ):
            reconciled.append(
                {
                    "key": key,
                    "raw": row.get("raw"),
                    "status": "evidence_mismatch",
                    "reason": "completion raw sha256 no longer matches queue evidence",
                }
            )
            continue
        budget_error = _semantic_defer_budget_error(
            budget=budget,
            charge_bytes=0,
        )
        if budget_error is not None:
            budget_deferred.append(
                {
                    "key": key,
                    "raw": row.get("raw"),
                    "reason": budget_error,
                    "action": "failure_reset_reconcile",
                }
            )
            continue
        attempted_at = _iso(now)
        if journal_pending:
            try:
                reset_raw_failure(_raw_name(dict(completion)) or _raw_name(row))
            except Exception as exc:
                row["failure_reset_pending"] = True
                row["failure_reset_error"] = f"{exc.__class__.__name__}: {exc}"
                row["failure_reset_last_attempt_at"] = attempted_at
                queue_dirty = True
                reconciled.append(
                    {
                        "key": key,
                        "raw": row.get("raw"),
                        "status": "retry_failed",
                        "reason": row["failure_reset_error"],
                    }
                )
                continue
            _append_completion(
                {
                    **dict(completion),
                    "ts": attempted_at,
                    "failure_reset_pending": False,
                    "failure_reset_completed_at": attempted_at,
                },
                completions_file,
            )
        row["failure_reset_pending"] = False
        row["failure_reset_error"] = None
        row["failure_reset_last_attempt_at"] = attempted_at
        row["failure_reset_completed_at"] = attempted_at
        queue_dirty = True
        reconciled.append(
            {
                "key": key,
                "raw": row.get("raw"),
                "status": "completed",
            }
        )
    return reconciled, budget_deferred, queue_dirty


def _run_candidate(
    row: dict[str, Any],
    *,
    now: datetime,
    retry_delay_seconds: int,
    max_bytes: int,
    completions_file: Path,
    job_id: str | None = None,
    attempt_started: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from chronovisor.ingest.ingest import run_ingest

    raw = str(row.get("raw") or "")
    path = _resolve_raw_path(raw, row.get("path"))
    attempts = _nonnegative_int(row.get("attempts")) + (0 if attempt_started else 1)
    pages_created: list[str] = []
    pages_updated: list[str] = []
    active_job_id = job_id or ""
    error = ""
    job_status = "missing"
    raw_bytes = _nonnegative_int(row.get("bytes"))
    failed_ops: list[Any] = []
    completion_written = False
    completion_uncertain = False
    failure_reset_pending = False
    failure_reset_error: str | None = None
    semantic_defer: dict[str, Any] | None = None
    content: str | None = None
    projection_summary: dict[str, Any] | None = None
    if _raw_name(row) in _active_terminal_semantic_deferred_raw_names():
        return _operational_deferred_candidate_result(
            row,
            now=now,
            job_id=job_id,
            attempt_started=attempt_started,
        )
    if raw_bytes > max_bytes:
        error = f"raw exceeds replay byte budget: {raw_bytes} > {max_bytes}"
        job_status = "oversized"
    elif path is None:
        error = f"raw file not found: {raw}"
    else:
        try:
            content, projection_summary = _replay_ingest_content(raw, path)
            if not active_job_id:
                job = job_store.create(processor="ollama")
                active_job_id = job.job_id

            def record_processed() -> None:
                nonlocal completion_written, failure_reset_error
                nonlocal failure_reset_pending
                finished_job = job_store.get(active_job_id)
                result = (
                    getattr(finished_job, "result", None)
                    if finished_job is not None
                    else None
                )
                callback_failed_ops = (
                    result.get("failed_ops") if isinstance(result, dict) else None
                )
                completion_status = (
                    "completed_partial"
                    if isinstance(callback_failed_ops, list) and callback_failed_ops
                    else "completed"
                )
                completion_record = {
                    "ts": _iso(now),
                    "schema_version": SCHEMA_VERSION,
                    "type": FULL_REPLAY_CLAIM_TYPE,
                    "key": row.get("key"),
                    "raw": raw,
                    "source_raw": f"replay:{raw}",
                    "status": completion_status,
                    "completion_scope": "full_raw",
                    "attempt_id": row.get("attempt_id"),
                    "job_id": active_job_id,
                    "raw_sha256": row.get("raw_sha256"),
                    "failed_ops": callback_failed_ops or [],
                    # Success is durable before supervisor cleanup starts.  A
                    # second journal row acknowledges cleanup, making a crash
                    # or transient I/O error recoverable without replaying the
                    # raw or downgrading the completed job.
                    "failure_reset_pending": True,
                }
                _append_completion(completion_record, completions_file)
                completion_written = True
                # Raw replay bypasses orchestrator, whose success path normally
                # retires failure-supervisor state.  Release the actual source
                # only after the durable completion marker exists, so an old
                # semantic packet cannot re-hold a successfully replayed raw if
                # authority resolution later fails or rolls back.
                from chronovisor.ingest.failure_supervisor import reset_raw_failure

                try:
                    reset_raw_failure(_raw_name(row) or Path(raw).name)
                except Exception as exc:  # durable marker owns the retry
                    failure_reset_pending = True
                    failure_reset_error = f"{exc.__class__.__name__}: {exc}"
                else:
                    reset_completed_at = _iso(_now())
                    _append_completion(
                        {
                            **completion_record,
                            "ts": reset_completed_at,
                            "failure_reset_pending": False,
                            "failure_reset_completed_at": reset_completed_at,
                        },
                        completions_file,
                    )

            # Failure state can change after queue selection and even after the
            # durable running marker. This is the last instruction before
            # inference; a newly published hold wins and consumes no replay.
            if _raw_name(row) in _active_terminal_semantic_deferred_raw_names():
                return _operational_deferred_candidate_result(
                    row,
                    now=now,
                    job_id=active_job_id,
                    attempt_started=attempt_started,
                )
            if content is None:
                job_store.update(
                    active_job_id,
                    status=JobStatus.COMPLETED,
                    completed_at=_iso(now),
                    processor="deterministic-projection",
                    stage="semantic-noop",
                    result={"semantic_projection": projection_summary or {}},
                )
                record_processed()
            else:
                metadata: dict[str, Any] = {"source_raw": f"replay:{raw}"}
                if (projection_summary or {}).get("kind") != "legacy_passthrough":
                    metadata["semantic_projection"] = projection_summary
                run_ingest(
                    content,
                    active_job_id,
                    on_complete=record_processed,
                    metadata=metadata,
                )
            finished = job_store.get(active_job_id)
            job_status = _job_status(finished)
            if finished is not None:
                pages_created = list(getattr(finished, "pages_created", []) or [])
                pages_updated = list(getattr(finished, "pages_updated", []) or [])
            result = getattr(finished, "result", None) if finished is not None else None
            result_failed_ops = (
                result.get("failed_ops") if isinstance(result, dict) else None
            )
            failed_ops = (
                result_failed_ops if isinstance(result_failed_ops, list) else []
            )
            if job_status != JobStatus.COMPLETED.value:
                error = _failure_text(finished, job_status)
            elif not completion_written:
                record_processed()
        except Exception as exc:
            error = f"{exc.__class__.__name__}: {exc}"
            finished = job_store.get(active_job_id) if active_job_id else None
            observed_status = _job_status(finished)
            if observed_status == JobStatus.COMPLETED.value:
                job_status = observed_status
                result = (
                    getattr(finished, "result", None) if finished is not None else None
                )
                result_failed_ops = (
                    result.get("failed_ops") if isinstance(result, dict) else None
                )
                failed_ops = (
                    result_failed_ops if isinstance(result_failed_ops, list) else []
                )
                completion_uncertain = True
            else:
                job_status = "error"

    if error and path is not None and content is not None:
        semantic_defer = _publish_semantic_no_quorum_defer(
            row,
            path=path,
            error=error,
            job_id=active_job_id or None,
            raw_text=content,
        )

    updated = dict(row)
    updated["attempts"] = attempts
    updated["last_attempt_at"] = _iso(now)
    updated["updated_at"] = _iso(now)
    updated["job_id"] = active_job_id or None
    if semantic_defer is not None:
        updated = _apply_semantic_no_quorum_defer(
            updated,
            evidence=semantic_defer,
            now=now,
        )
    elif completion_uncertain:
        updated["status"] = "indeterminate"
        updated["next_retry_at"] = None
        updated["last_error"] = (
            "ingest completed but durable replay completion marker failed: " + error
        )
        updated["terminal_reason"] = "awaiting autonomous frontier reconciliation"
        updated["frontier_attempts"] = _nonnegative_int(
            updated.get("frontier_attempts")
        )
        updated["next_frontier_retry_at"] = _iso(now)
    elif not error:
        updated["status"] = "completed_partial" if failed_ops else "completed"
        updated["completed_at"] = _iso(now)
        updated["next_retry_at"] = None
        updated["last_error"] = None
        updated["completion_evidence"] = "replay_completion_journal"
        updated["failure_reset_pending"] = failure_reset_pending
        updated["failure_reset_error"] = failure_reset_error
        updated["failure_reset_last_attempt_at"] = _iso(now)
        if not failure_reset_pending:
            updated["failure_reset_completed_at"] = _iso(now)
        if failed_ops:
            updated["terminal_reason"] = (
                "partial operations delegated to autonomous repair"
            )
    elif attempts >= MAX_ATTEMPTS:
        updated["status"] = "quarantined"
        updated["next_retry_at"] = None
        updated["last_error"] = error
        updated["quarantined_at"] = _iso(now)
    else:
        delay = max(0, retry_delay_seconds) * (2 ** (attempts - 1))
        updated["status"] = "failed"
        updated["next_retry_at"] = _iso(now + timedelta(seconds=delay))
        updated["last_error"] = error

    record = {
        "ts": _iso(now),
        "schema_version": SCHEMA_VERSION,
        "key": updated["key"],
        "raw": raw,
        "path": updated.get("path"),
        "status": updated["status"],
        "attempts": attempts,
        "next_retry_at": updated.get("next_retry_at"),
        "job_id": active_job_id or None,
        "attempt_id": updated.get("attempt_id"),
        "job_status": job_status,
        "pages_created": pages_created,
        "pages_updated": pages_updated,
        "failed_ops": failed_ops,
        "bytes": raw_bytes,
        "charged_bytes": raw_bytes if job_status != "oversized" else 0,
        "error": error or None,
        "failure_reset_pending": failure_reset_pending,
        "failure_reset_error": failure_reset_error,
    }
    if semantic_defer is not None:
        record.update(
            {
                "status": "pending",
                "attempts": updated["attempts"],
                "next_retry_at": None,
                "operational_deferred": True,
                "semantic_deferred": True,
                "defer_reason": semantic_defer["reason"],
                "semantic_defer_authority_sha256": semantic_defer["authority_sha256"],
                "semantic_defer_packet": semantic_defer.get("packet_path"),
                "semantic_defer_prior_attempts": updated[
                    "semantic_defer_prior_attempts"
                ],
            }
        )
    return updated, record


def run_pending_queue(
    *,
    path: Path | None = None,
    history_file: Path | None = None,
    claims_file: Path | None = None,
    completions_file: Path | None = None,
    max_runs: int = 1,
    max_bytes: int = DEFAULT_MAX_BYTES,
    dry_run: bool = False,
    retry_delay_seconds: int = DEFAULT_RETRY_DELAY_SECONDS,
    now: datetime | None = None,
    eligible_keys: set[str] | None = None,
    eligible_sources: set[str] | frozenset[str] | None = None,
    budget: Any | None = None,
    frontier_reviewer: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run due queue rows within strict count and byte budgets.

    ``eligible_keys`` and ``eligible_sources`` form a union, allowing a caller
    to include both current signals and older pending work in the same lane.
    Dry-run performs no writes, creates no lock file, and never invokes ingest.
    Individually oversized rows consume a run slot but zero byte budget while
    advancing through the normal retry/quarantine lifecycle.
    """
    target = path if path is not None else QUEUE_FILE
    history_target = history_file if history_file is not None else HISTORY_FILE
    claims_target = claims_file if claims_file is not None else CLAIMS_FILE
    completions_target = (
        completions_file if completions_file is not None else COMPLETIONS_FILE
    )
    current_time = now or _now()
    run_limit = _nonnegative_int(max_runs)
    byte_limit = _nonnegative_int(max_bytes)
    semantic_deferred_raws: frozenset[str] = frozenset()
    load_maintenance_changed = False

    def in_scope(row: Mapping[str, Any]) -> bool:
        return _matches_eligibility_scope(
            row,
            eligible_keys=eligible_keys,
            eligible_sources=eligible_sources,
        )

    def load_rows(*, resume_terminals: bool) -> list[dict[str, Any]]:
        nonlocal load_maintenance_changed
        by_key: dict[str, dict[str, Any]] = {}
        history_states = _history_lifecycle_states(history_target)
        for raw_row in _read_jsonl(target):
            row = _normalize_queue_row(raw_row, now=current_time)
            if row is None:
                continue
            key = str(row["key"])
            current = by_key.get(key)
            by_key[key] = (
                row
                if current is None
                else _merge_durable_rows(current, row, now=current_time)
            )
        for key, row in by_key.items():
            if not in_scope(row):
                continue
            maintenance_before = json.dumps(
                row,
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            )
            by_key[key] = _merge_history_lifecycle(row, history_states.get(key))
            _retire_retracted_row(by_key[key], now=current_time)
            _coerce_exhausted(by_key[key], now=current_time)
            if resume_terminals:
                _resume_due_autonomous_terminal(
                    by_key[key],
                    now=current_time,
                    semantic_deferred_raws=semantic_deferred_raws,
                )
            load_maintenance_changed = load_maintenance_changed or (
                maintenance_before
                != json.dumps(
                    by_key[key],
                    ensure_ascii=False,
                    sort_keys=True,
                    default=str,
                )
            )
        return list(by_key.values())

    if dry_run:
        rows = load_rows(resume_terminals=False)
        scoped_rows = [row for row in rows if in_scope(row)]
        active_deferred_statuses = _scoped_operational_deferred_raw_statuses(
            scoped_rows
        )
        semantic_deferred_raws |= frozenset(active_deferred_statuses)
        completed = _completed_replays(
            history_file=history_target,
            claims_file=claims_target,
            completions_file=completions_target,
        )
        semantic_plan, semantic_candidate_keys = _plan_legacy_semantic_no_quorum_rows(
            rows,
            active_deferred_statuses=active_deferred_statuses,
            max_runs=run_limit,
            max_bytes=byte_limit,
            eligible_keys=eligible_keys,
            eligible_sources=eligible_sources,
            completed_keys=frozenset(completed),
        )
        semantic_verified, semantic_budget_deferred, verification_failed = (
            _preview_legacy_semantic_no_quorum_rows(
                semantic_plan,
                max_bytes=byte_limit,
                budget=budget,
            )
        )
        semantic_candidate_keys = frozenset(
            set(semantic_candidate_keys) - set(verification_failed)
        )
        semantic_reserved_rows = [row for row, _candidate in semantic_plan]
        semantic_reserved_bytes = sum(
            _nonnegative_int(row.get("bytes"))
            for row in semantic_reserved_rows
            if _nonnegative_int(row.get("bytes")) <= byte_limit
        )
        ordinary_rows = [
            row
            for row in rows
            if in_scope(row)
            and str(row.get("key") or "") not in semantic_candidate_keys
        ]
        for row in ordinary_rows:
            _resume_due_autonomous_terminal(
                row,
                now=current_time,
                semantic_deferred_raws=semantic_deferred_raws,
            )
        eligible_rows = [row for row in ordinary_rows if row["key"] not in completed]
        selected = _select_bounded(
            eligible_rows,
            now=current_time,
            max_runs=max(0, run_limit - len(semantic_plan)),
            max_bytes=byte_limit,
            eligible_keys=eligible_keys,
            eligible_sources=eligible_sources,
            semantic_deferred_raws=semantic_deferred_raws,
            initial_used_bytes=semantic_reserved_bytes,
        )
        semantic_planned = [
            {
                "key": row["key"],
                "raw": row["raw"],
                "bytes": row["bytes"],
                "charged_bytes": (
                    0
                    if _nonnegative_int(row.get("bytes")) > byte_limit
                    else row["bytes"]
                ),
                "action": "semantic_defer_reconcile",
                "attempts": row["attempts"],
                "authority_sha256": evidence.get("authority_sha256"),
                "packet_path": evidence.get("packet_path"),
            }
            for row, evidence in semantic_verified
        ]
        ordinary_planned = [
            {
                "key": row["key"],
                "raw": row["raw"],
                "bytes": row["bytes"],
                "charged_bytes": (
                    0
                    if _nonnegative_int(row.get("bytes")) > byte_limit
                    else row["bytes"]
                ),
                "action": (
                    "retry_oversized"
                    if _nonnegative_int(row.get("bytes")) > byte_limit
                    else "ingest"
                ),
                "attempts": row["attempts"],
            }
            for row in selected
        ]
        return {
            "status": "dry_run",
            "dry_run": True,
            "queue": str(target),
            "count": len(semantic_planned) + len(ordinary_planned),
            "runs": [],
            "planned": [*semantic_planned, *ordinary_planned],
            "semantic_defer_planned": semantic_planned,
            "budget_deferred": semantic_budget_deferred,
            "bytes": sum(
                _nonnegative_int(row.get("bytes"))
                for row in selected
                if _nonnegative_int(row.get("bytes")) <= byte_limit
            )
            + sum(
                _nonnegative_int(row.get("bytes"))
                for row, _evidence in semantic_verified
            ),
            "oversized": sum(
                1 for row in selected if _nonnegative_int(row.get("bytes")) > byte_limit
            )
            + sum(
                1
                for row in semantic_reserved_rows
                if _nonnegative_int(row.get("bytes")) > byte_limit
            ),
            "max_runs": run_limit,
            "max_bytes": byte_limit,
        }

    with _queue_lock(target):
        # Preserve legacy no-quorum error and attempt evidence until it has a
        # chance to become a terminal supervisor hold.  Cooldown reopening
        # clears that evidence, so it is intentionally delayed until after the
        # model-free reconciliation below.
        rows = load_rows(resume_terminals=False)
        scoped_rows = [row for row in rows if in_scope(row)]
        completion_states = _completion_states(completions_target)
        (
            failure_reset_reconciled,
            failure_reset_budget_deferred,
            failure_reset_queue_dirty,
        ) = _reconcile_pending_failure_resets(
            scoped_rows,
            completion_states=completion_states,
            completions_file=completions_target,
            now=current_time,
            budget=budget,
        )
        active_deferred_statuses = _scoped_operational_deferred_raw_statuses(
            scoped_rows
        )
        semantic_deferred_raws |= frozenset(active_deferred_statuses)
        completed = _completed_replays(
            history_file=history_target,
            claims_file=claims_target,
            completions_file=completions_target,
        )
        semantic_plan, semantic_candidate_keys = _plan_legacy_semantic_no_quorum_rows(
            rows,
            active_deferred_statuses=active_deferred_statuses,
            max_runs=run_limit,
            max_bytes=byte_limit,
            eligible_keys=eligible_keys,
            eligible_sources=eligible_sources,
            completed_keys=frozenset(completed),
        )
        (
            semantic_defer_reconciled,
            semantic_budget_deferred,
            verification_failed,
        ) = _reconcile_legacy_semantic_no_quorum_rows(
            semantic_plan,
            now=current_time,
            max_bytes=byte_limit,
            budget=budget,
        )
        semantic_candidate_keys = frozenset(
            set(semantic_candidate_keys) - set(verification_failed)
        )
        # A normal ingest may publish a semantic defer while this runner waits
        # for its queue lease.  Preserve any earlier hold and add newly active
        # ones before crash reconciliation can invoke a reviewer.
        semantic_deferred_raws = _active_terminal_semantic_deferred_raw_names()
        semantic_reserved_bytes = sum(
            _nonnegative_int(row.get("bytes"))
            for row, _candidate in semantic_plan
            if _nonnegative_int(row.get("bytes")) <= byte_limit
        )
        reconciled_bytes = sum(
            _nonnegative_int(row.get("charged_bytes"))
            for row in semantic_defer_reconciled
        )
        remaining_run_limit = max(0, run_limit - len(semantic_plan))
        ordinary_rows = [
            row
            for row in scoped_rows
            if str(row.get("key") or "") not in semantic_candidate_keys
        ]
        ordinary_state_before = {
            str(row.get("key") or ""): json.dumps(
                row,
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            )
            for row in ordinary_rows
        }
        for row in ordinary_rows:
            _resume_due_autonomous_terminal(
                row,
                now=current_time,
                semantic_deferred_raws=semantic_deferred_raws,
            )
        replayable_rows = [
            row for row in ordinary_rows if _raw_name(row) not in semantic_deferred_raws
        ]
        reconciled = _reconcile_running_rows(
            replayable_rows,
            completions_file=completions_target,
            history_file=history_target,
            now=current_time,
            retry_delay_seconds=retry_delay_seconds,
        )
        for row in ordinary_rows:
            _retire_retracted_row(row, now=current_time)
        for row in ordinary_rows:
            evidence = completed.get(str(row.get("key")))
            if not evidence or row.get("status") in {
                "completed",
                "completed_partial",
                "indeterminate",
                "running",
            }:
                continue
            _mark_completed(row, evidence=evidence, now=current_time)
            row["recovery_kind"] = "exact_already_applied"
        for row in ordinary_rows:
            _coerce_exhausted(row, now=current_time)
        semantic_deferred_raws = _active_terminal_semantic_deferred_raw_names()
        replayable_rows = [
            row
            for row in replayable_rows
            if _raw_name(row) not in semantic_deferred_raws
        ]
        frontier_reconciliation = _review_indeterminate_rows(
            replayable_rows,
            claims_file=claims_target,
            history_file=history_target,
            now=current_time,
            budget=budget,
            retry_delay_seconds=retry_delay_seconds,
            reviewer=frontier_reviewer,
        )
        selected = _select_bounded(
            ordinary_rows,
            now=current_time,
            max_runs=remaining_run_limit,
            max_bytes=byte_limit,
            eligible_keys=eligible_keys,
            eligible_sources=eligible_sources,
            semantic_deferred_raws=semantic_deferred_raws,
            initial_used_bytes=semantic_reserved_bytes,
        )
        ordinary_state_changed = any(
            ordinary_state_before.get(str(row.get("key") or ""))
            != json.dumps(
                row,
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            )
            for row in ordinary_rows
        )
        # Persist history reconciliation, duplicate collapse and schema
        # normalization before running any expensive candidate.  A call whose
        # exact semantic scope was rejected solely by budget is intentionally
        # byte-for-byte read-only; normalization must not smuggle a queue write
        # past a zero-mutation budget.
        prelaunch_write_required = bool(
            semantic_defer_reconciled
            or ordinary_state_changed
            or reconciled
            or frontier_reconciliation.get("reviewed")
            or failure_reset_queue_dirty
            or load_maintenance_changed
        )
        if prelaunch_write_required:
            _atomic_write_queue(target, sorted(rows, key=_queue_sort_key))

        by_key = {str(row["key"]): row for row in rows}
        runs: list[dict[str, Any]] = []
        budget_deferred: list[dict[str, Any]] = [
            *failure_reset_budget_deferred,
            *semantic_budget_deferred,
        ]
        operational_deferred: list[dict[str, Any]] = []
        operational_deferred_keys: set[str] = set()
        authorization_rejected: list[dict[str, Any]] = []

        def record_operational_hold(
            row: dict[str, Any],
            *,
            reason: str,
        ) -> None:
            key = str(row.get("key") or "")
            if key in operational_deferred_keys:
                return
            operational_deferred_keys.add(key)
            operational_deferred.append(
                {
                    "key": row.get("key"),
                    "raw": row.get("raw"),
                    "reason": reason,
                }
            )

        deferred_statuses = _scoped_operational_deferred_raw_statuses(scoped_rows)
        for row in scoped_rows:
            raw_name = _raw_name(row)
            if raw_name in semantic_deferred_raws:
                record_operational_hold(
                    row,
                    reason=deferred_statuses.get(raw_name) or "semantic_no_quorum",
                )

        def operational_hold_is_active(row: dict[str, Any]) -> bool:
            raw_name = _raw_name(row)
            if raw_name not in _active_terminal_semantic_deferred_raw_names():
                return False
            status = _scoped_operational_deferred_raw_statuses([row]).get(raw_name)
            reason = status or "semantic_no_quorum"
            record_operational_hold(row, reason=reason)
            return True

        for selected_row in selected:
            if operational_hold_is_active(selected_row):
                continue
            raw_bytes = _nonnegative_int(selected_row.get("bytes"))
            charge_bytes = raw_bytes if raw_bytes <= byte_limit else 0
            bytes_allowed = True
            bytes_reason: str | None = None
            mutation_allowed = True
            mutation_reason: str | None = None
            if budget is not None:
                bytes_allowed, bytes_reason = (
                    budget.can_consume("raw_bytes", charge_bytes)
                    if charge_bytes
                    else (True, None)
                )
                mutation_allowed, mutation_reason = budget.can_consume("mutation")
                # Budget preflight is user-injectable and may overlap a normal
                # ingest publishing a terminal semantic hold.  Recheck before
                # either classifying a budget defer or consuming a counter so
                # the authority-bound semantic state remains the terminal
                # reason for this attempt.
                if operational_hold_is_active(selected_row):
                    continue
                if not bytes_allowed or not mutation_allowed:
                    budget_deferred.append(
                        {
                            "key": selected_row.get("key"),
                            "raw": selected_row.get("raw"),
                            "reason": bytes_reason
                            if not bytes_allowed
                            else mutation_reason,
                        }
                    )
                    continue

            requires_semantic_authorization = (
                selected_row.get("frontier_decision") == "safe_replay"
            )
            effect_lock = (
                decision_authority_lock()
                if requires_semantic_authorization
                else nullcontext()
            )
            with effect_lock:
                if requires_semantic_authorization:
                    approval_error = _raw_replay_approval_error(
                        selected_row,
                        approval=selected_row.get("frontier_review_artifact"),
                        claims_file=claims_target,
                        injected_reviewer=frontier_reviewer is not None,
                    )
                    if operational_hold_is_active(selected_row):
                        continue
                    if approval_error is not None:
                        updated = _invalidate_safe_replay_authorization(
                            selected_row,
                            now=current_time,
                            reason=approval_error,
                        )
                        by_key[str(updated["key"])] = updated
                        record = {
                            "ts": _iso(current_time),
                            "schema_version": SCHEMA_VERSION,
                            "key": updated.get("key"),
                            "raw": updated.get("raw"),
                            "status": updated.get("status"),
                            "attempts": updated.get("attempts"),
                            "frontier_attempts": updated.get("frontier_attempts"),
                            "frontier_decision": updated.get("frontier_decision"),
                            "frontier_review_artifact": updated.get(
                                "frontier_review_artifact"
                            ),
                            "frontier_authority_error": approval_error,
                            "error": approval_error,
                            "terminal_reason": updated.get("terminal_reason"),
                            "authorization_rejected": True,
                        }
                        _append_history(record, history_target)
                        _atomic_write_queue(
                            target, sorted(by_key.values(), key=_queue_sort_key)
                        )
                        authorization_rejected.append(
                            {
                                "key": updated.get("key"),
                                "raw": updated.get("raw"),
                                "reason": approval_error,
                            }
                        )
                        continue

                # This is the last non-budget authority check before the
                # at-most-once launch boundary.  It prevents an authority
                # validation race from turning a terminal semantic split into
                # a replay attempt.
                if operational_hold_is_active(selected_row):
                    continue

                if budget is not None:
                    if charge_bytes:
                        bytes_allowed, bytes_reason = budget.consume(
                            "raw_bytes", charge_bytes
                        )
                    mutation_allowed, mutation_reason = budget.consume("mutation")
                    if operational_hold_is_active(selected_row):
                        continue
                    if not bytes_allowed or not mutation_allowed:
                        budget_deferred.append(
                            {
                                "key": selected_row.get("key"),
                                "raw": selected_row.get("raw"),
                                "reason": bytes_reason
                                if not bytes_allowed
                                else mutation_reason,
                            }
                        )
                        continue

                # CycleBudget has deliberately conservative counters and no
                # refund contract.  If a hold is published during consume(),
                # retain the charge but never create a job, write a running
                # marker, or invoke ingest.
                if operational_hold_is_active(selected_row):
                    continue

                candidate = selected_row
                candidate_path = _resolve_raw_path(
                    str(selected_row.get("raw") or ""), selected_row.get("path")
                )
                active_job_id: str | None = None
                attempt_started = False
                if candidate_path is not None and raw_bytes <= byte_limit:
                    raw_hash = _sha256_path(candidate_path)
                    job = job_store.create(processor="ollama")
                    active_job_id = job.job_id
                    candidate = dict(selected_row)
                    _clear_released_semantic_defer(candidate, now=current_time)
                    attempts = _nonnegative_int(candidate.get("attempts")) + 1
                    candidate.update(
                        {
                            "status": "running",
                            "attempts": attempts,
                            "last_attempt_at": _iso(current_time),
                            "started_at": _iso(current_time),
                            "updated_at": _iso(current_time),
                            "job_id": active_job_id,
                            "raw_sha256": raw_hash,
                            "attempt_id": f"{active_job_id}:{attempts}:{raw_hash[:16]}",
                            "next_retry_at": None,
                            "last_error": None,
                        }
                    )
                    if requires_semantic_authorization:
                        candidate["frontier_authorization_consumed_at"] = _iso(
                            current_time
                        )
                    by_key[str(candidate["key"])] = candidate
                    # This write is the at-most-once launch boundary. Ingest is
                    # never called unless the running marker is durable first.
                    _atomic_write_queue(
                        target, sorted(by_key.values(), key=_queue_sort_key)
                    )
                    attempt_started = True
                updated, record = _run_candidate(
                    candidate,
                    now=current_time,
                    retry_delay_seconds=retry_delay_seconds,
                    max_bytes=byte_limit,
                    completions_file=completions_target,
                    job_id=active_job_id,
                    attempt_started=attempt_started,
                )
                if requires_semantic_authorization:
                    record["frontier_authorization_consumed_at"] = updated.get(
                        "frontier_authorization_consumed_at"
                    )
                    record["frontier_review_artifact"] = updated.get(
                        "frontier_review_artifact"
                    )
                by_key[str(updated["key"])] = updated
                if record.get("operational_deferred") is True:
                    raw_name = _raw_name(updated)
                    record_reason = str(record.get("defer_reason") or "")
                    record_operational_hold(
                        updated,
                        reason=(
                            record_reason
                            if record_reason == "semantic_no_quorum"
                            else _scoped_operational_deferred_raw_statuses(
                                [updated]
                            ).get(raw_name)
                        )
                        or record_reason
                        or "operational_failure_hold",
                    )
                    _atomic_write_queue(
                        target, sorted(by_key.values(), key=_queue_sort_key)
                    )
                    continue
                runs.append(record)
                _append_history(record, history_target)
                # Persist every terminal/retry transition so a later candidate is
                # never replayed twice merely because a subsequent row crashed.
                _atomic_write_queue(
                    target, sorted(by_key.values(), key=_queue_sort_key)
                )

        final_rows = list(by_key.values())
        status_counts: dict[str, int] = {}
        for row in final_rows:
            status = str(row.get("status") or "pending")
            status_counts[status] = status_counts.get(status, 0) + 1
        return {
            "status": "ok",
            "dry_run": False,
            "queue": str(target),
            "runs": runs,
            "count": len(runs),
            "budget_deferred": budget_deferred,
            "semantic_deferred": [
                row
                for row in operational_deferred
                if row.get("reason") == "semantic_no_quorum"
            ],
            "operational_deferred": operational_deferred,
            "authorization_rejected": authorization_rejected,
            "semantic_defer_reconciled": semantic_defer_reconciled,
            "failure_reset_reconciled": failure_reset_reconciled,
            "reconciled": reconciled,
            "frontier_reconciliation": frontier_reconciliation,
            "bytes": sum(
                _nonnegative_int(row.get("bytes"))
                for row in selected
                if _nonnegative_int(row.get("bytes")) <= byte_limit
            )
            + reconciled_bytes,
            "oversized": sum(
                1 for row in selected if _nonnegative_int(row.get("bytes")) > byte_limit
            )
            + sum(
                1
                for row in semantic_defer_reconciled
                if _nonnegative_int(row.get("bytes")) > byte_limit
            ),
            "max_runs": run_limit,
            "max_bytes": byte_limit,
            "status_counts": dict(sorted(status_counts.items())),
        }


def run_replay(*, since: str = "", limit: int = 1) -> dict[str, Any]:
    """Compatibility entry point for explicit migration followed by replay."""
    migration_limit = _nonnegative_int(limit)
    queue = build_queue(
        since=since,
        limit=migration_limit,
        include_migration=True,
        include_auto_signals=False,
    )
    keys = {str(key) for key in queue.get("candidate_keys", [])}
    rows = [_normalize_queue_row(row, now=_now()) for row in _read_jsonl(QUEUE_FILE)]
    byte_budget = sum(
        _nonnegative_int(row.get("bytes"))
        for row in rows
        if row is not None and row.get("key") in keys
    )
    result = run_pending_queue(
        max_runs=migration_limit or len(keys),
        max_bytes=byte_budget,
        eligible_keys=keys,
    )
    return {**result, "queue_refresh": queue}


def main(argv: list[str] | None = None) -> int:
    """Run the ``chronovisor-raw-replay`` command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Plan or run retroactive raw re-ingestion."
    )
    parser.add_argument(
        "--since", default="", help="YYYYMMDD or YYYY-MM-DD lower bound."
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--run", action="store_true", help="Actually re-ingest selected raws."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect without queue/history/ingest writes.",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.dry_run:
        return _main_locked(args)
    from chronovisor.core.okf_cutover import OKFStartupBlocked

    try:
        with okf_runtime_operation(CHRONOVISOR_ROOT):
            return _main_locked(args)
    except OKFStartupBlocked:
        print(json.dumps({"status": "blocked", "category": "okf_startup_blocked"}))
        return 75


def _main_locked(args: argparse.Namespace) -> int:
    if args.run:
        if args.dry_run:
            payload = build_queue(
                since=args.since,
                limit=max(1, args.limit or 1),
                dry_run=True,
                include_migration=True,
                include_auto_signals=False,
            )
        else:
            payload = run_replay(since=args.since, limit=max(1, args.limit or 1))
    else:
        payload = build_queue(
            since=args.since, limit=max(0, args.limit), dry_run=args.dry_run
        )
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    else:
        print(
            "\t".join(
                f"{key}={value}"
                for key, value in payload.items()
                if key not in {"runs", "candidate_keys"}
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
