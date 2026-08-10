"""Failure supervision for self-healing ingest runs.

This module is intentionally deterministic.  LLMs may diagnose a packet later,
but the control loop here decides when to stop retrying a raw, how to fingerprint
the failure, and where to persist the evidence for local/frontier repair.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import re
import shutil
import threading
from collections.abc import Iterable, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from chronovisor.core import runtime_status
from chronovisor.core import store as chronovisor_store
from chronovisor.core.link_fix import atomic_write

FAILURE_THRESHOLD = 3
_FAILURE_STATE_THREAD_LOCK = threading.RLock()


@dataclass(frozen=True)
class FailureRecord:
    """Normalized failure information used by the supervisor."""

    failure_class: str
    fingerprint: str
    message: str
    requested_page_id: str | None = None
    authority_artifact_sha256: str | None = None


@dataclass(frozen=True)
class SupervisionResult:
    """Outcome of recording one failed raw."""

    raw_file: str
    failure_class: str
    fingerprint: str
    attempts: int
    quarantined: bool = False
    packet_path: str | None = None
    quarantine_path: str | None = None
    tracked: bool = True
    transient: bool = False
    terminal_deferred: bool = False


TRANSIENT_FAILURE_CLASSES = {
    "ingest.ollama_unavailable",
    "ingest.runtime_transport_error",
    "ingest.runtime_transport_timeout",
    "ingest.generation_transport_error",
    "ingest.runtime_capacity_unavailable",
    "ingest.generation_capacity_unavailable",
    "ingest.runtime_semantic_projection_interrupted",
    # The semantic mutation already has a verified durable ACK receipt.  Only
    # the processed-state write remains, so retrying performs no inference.
    "ingest.raw_completion_ack_state_pending",
}

SEMANTIC_PROJECTION_OPERATIONAL_FAILURE_CLASSES = {
    "ingest.runtime_semantic_projection_artifact_conflict",
    "ingest.runtime_semantic_projection_capacity",
    "ingest.runtime_semantic_projection_failure",
    "ingest.runtime_semantic_projection_internal_error",
}

# These failures describe a broken request contract, context calculation, or
# validator integration.  Retrying the same raw cannot fix them, but the raw is
# still valid source material and must never be quarantined as the cause.
OPERATIONAL_SELF_HEAL_FAILURE_CLASSES = {
    *SEMANTIC_PROJECTION_OPERATIONAL_FAILURE_CLASSES,
    # The immutable raw cannot repair a missing, stale, or internally
    # inconsistent local-consensus authority artifact.  Queue one
    # control-plane repair packet and keep every affected raw in place.
    "ingest.runtime_local_consensus_authority_unavailable",
    "ingest.runtime_schema_invalid",
    "ingest.runtime_input_invalid",
    "ingest.runtime_input_too_large",
    "ingest.runtime_feedback_too_large",
    "ingest.runtime_output_too_large",
    "ingest.runtime_value_validation_error",
    "ingest.runtime_value_validator_error",
    "ingest.runtime_context_truncation_suspected",
    "ingest.runtime_context_window_exceeded",
    "ingest.runtime_stream_incomplete",
    "ingest.runtime_completion_incomplete",
    "ingest.runtime_output_truncated",
    "ingest.runtime_triage_repair_exhausted",
    "ingest.runtime_triage_repeated_output",
    "ingest.runtime_triage_unknown",
    "ingest.generation_context_window_exceeded",
    "ingest.generation_context_truncation_suspected",
    "ingest.generation_feedback_too_large",
    "ingest.generation_stream_incomplete",
    "ingest.generation_completion_incomplete",
    "ingest.generation_output_truncated",
    "ingest.generation_repeated_output",
    "ingest.generation_repair_exhausted",
    "ingest.generation_validation_failed",
    # A missing/corrupt receipt after apply is a control-plane defect, never a
    # reason to blame or quarantine the immutable source raw.
    "ingest.raw_completion_receipt_publish_failed",
    "ingest.raw_completion_receipt_invalid",
}

REPAIR_SUCCESS_PACKET_STATUSES = {
    "local_repair_applied",
    "frontier_approved",
}

LOCAL_CONSENSUS_AUTHORITY_FAILURE_CLASS = (
    "ingest.runtime_local_consensus_authority_unavailable"
)
ADOPTION_ARTIFACT_INVALID_FINGERPRINT = (
    f"{LOCAL_CONSENSUS_AUTHORITY_FAILURE_CLASS}:adoption_artifact_invalid"
)

# These failures already exhausted a bounded convergence loop inside one
# ingest job. Replaying the raw through three more jobs only burns local and
# frontier tokens while reproducing the same control-path defect.
IMMEDIATE_SELF_HEAL_FAILURE_CLASSES = {
    "ingest.frontier_nonconvergent",
    "ingest.local_consensus_nonconvergent",
}

SEMANTIC_NO_QUORUM_FAILURE_CLASS = "ingest.semantic_no_quorum"
SEMANTIC_NO_QUORUM_DEFER_REASON = "semantic_no_quorum"
SEMANTIC_DEFER_RELEASE_PACKET_STATUSES = {
    "semantic_defer_released",
    "superseded_semantic_defer",
}
_CANCELLABLE_OPERATIONAL_PACKET_STATUSES = {
    "pending_local_repair",
    "local_repair_failed",
    "pending_frontier",
    "frontier_retry",
    "frontier_preflight_failed",
    "pending_frontier_review",
    "repair_deferred",
    "local_repairing",
    "frontier_running",
}
_SEMANTIC_AUTHORITY_MARKER_RE = re.compile(r"\[authority_sha256=([0-9a-f]{64})\]")


def _runtime_failures_dir() -> Path:
    return chronovisor_store.CHRONOVISOR_ROOT / "runtime" / "failures"


def _state_file() -> Path:
    return _runtime_failures_dir() / "state.json"


@contextmanager
def _failure_state_lock(*, exclusive: bool = True):
    """Serialize full state read-modify-write transactions across processes."""

    failures_dir = _runtime_failures_dir()
    failures_dir.mkdir(parents=True, exist_ok=True)
    lock_path = failures_dir / "state.lock"
    with _FAILURE_STATE_THREAD_LOCK, lock_path.open("a+") as handle:
        operation = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        fcntl.flock(handle.fileno(), operation)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _load_state() -> dict[str, Any]:
    path = _state_file()
    if not path.exists():
        return {"failures": {}}
    try:
        data = json.loads(path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {"failures": {}}
    if not isinstance(data, dict):
        return {"failures": {}}
    failures = data.get("failures")
    if not isinstance(failures, dict):
        data["failures"] = {}
    return data


def _save_state(state: dict[str, Any]) -> None:
    path = _state_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(path, json.dumps(state, indent=2, ensure_ascii=False) + "\n")


def _operational_failure_group_snapshot_unlocked(
    packet_path: Path,
) -> tuple[tuple[str, dict[str, Any]], ...]:
    """Return state rows bound to one packet from the current state snapshot."""

    target = packet_path.expanduser().resolve(strict=False)
    failures = _load_state().get("failures")
    if not isinstance(failures, dict):
        return ()
    rows: list[tuple[str, dict[str, Any]]] = []
    for raw_file, entry in failures.items():
        if not isinstance(raw_file, str) or not isinstance(entry, dict):
            continue
        value = entry.get("packet_path")
        if not isinstance(value, str) or not value.strip():
            continue
        if Path(value).expanduser().resolve(strict=False) == target:
            rows.append((raw_file, dict(entry)))
    return tuple(sorted(rows, key=lambda row: row[0]))


def operational_failure_group_snapshot(
    packet_path: Path,
) -> tuple[tuple[str, dict[str, Any]], ...]:
    """Read one packet's affected raw group without creating a lock file."""

    return _operational_failure_group_snapshot_unlocked(packet_path)


def raw_failure_snapshot(raw_files: Iterable[str]) -> dict[str, dict[str, Any]]:
    """Return durable failure ownership for an exact set of Raw filenames.

    Current state is authoritative.  A latest terminal packet is used only
    when crash recovery or legacy quarantine left no state row, so an older
    semantic hold can still observe that its retry reached a newer outcome.
    """

    requested = {name for name in raw_files if isinstance(name, str) and name}
    if not requested:
        return {}
    with _failure_state_lock(exclusive=False):
        failures = _load_state().get("failures")
        snapshot = {
            raw_file: dict(entry)
            for raw_file, entry in (failures.items() if isinstance(failures, dict) else ())
            if raw_file in requested and isinstance(entry, dict)
        }
    missing = {
        raw_file
        for raw_file in requested
        if not isinstance(snapshot.get(raw_file, {}).get("packet_path"), str)
    }
    if not missing:
        return snapshot
    packets_dir = _runtime_failures_dir() / "packets"
    try:
        packet_paths = sorted(packets_dir.glob("*.json"))
    except OSError:
        return snapshot
    fallback: dict[str, dict[str, Any]] = {}
    for packet_path in packet_paths:
        packet = _read_packet_object(packet_path)
        if packet is None or packet.get("status") != "local_quarantined":
            continue
        packet_raws = {packet.get("raw_file")}
        sources = packet.get("source_raws")
        if isinstance(sources, list):
            packet_raws.update(
                row.get("filename")
                for row in sources
                if isinstance(row, dict)
            )
        for raw_file in missing.intersection(packet_raws):
            fallback[str(raw_file)] = {
                "packet_path": str(packet_path),
                "failure_class": packet.get("failure_class"),
                "status": packet.get("status"),
                "source": "terminal_packet_fallback",
            }
    snapshot.update(fallback)
    return snapshot


@contextmanager
def lock_operational_failure_group(packet_path: Path):
    """Freeze raw attachment to one operational packet through caller commit.

    A mutating caller acquires the packet lock first, then enters this context.
    Existing packet handlers use that same packet -> state order; raw attachment
    only takes the state lock. The yielded snapshot and caller commit therefore
    form one transaction without adding a state -> packet deadlock edge.
    """

    with _failure_state_lock():
        yield _operational_failure_group_snapshot_unlocked(packet_path)


def reset_raw_failure(raw_file: str) -> None:
    """Forget tracked failures for a raw after it succeeds."""

    with _failure_state_lock():
        state = _load_state()
        failures = state.get("failures", {})
        if not isinstance(failures, dict):
            failures = {}
            state["failures"] = failures

        # Packet publication intentionally precedes the state transaction.  A
        # successful retry must therefore retire both state-backed and orphaned
        # semantic packets before deleting state, or packet reconciliation would
        # resurrect the hold on the next queue scan.
        related_raw_files = {raw_file}
        semantic_packet_paths: set[Path] = set()
        packet_records = _semantic_defer_packet_records(verify_sources=False)
        changed = True
        while changed:
            changed = False
            for tracked_raw, entry in failures.items():
                if (
                    not isinstance(tracked_raw, str)
                    or not isinstance(entry, dict)
                    or entry.get("terminal_deferred") is not True
                    or entry.get("failure_class") != SEMANTIC_NO_QUORUM_FAILURE_CLASS
                ):
                    continue
                entry_related = _safe_raw_filenames(entry.get("related_raw_files"))
                entry_related.add(tracked_raw)
                packet_value = entry.get("packet_path")
                entry_packet = (
                    Path(packet_value).expanduser()
                    if isinstance(packet_value, str) and packet_value.strip()
                    else None
                )
                if not (
                    related_raw_files.intersection(entry_related)
                    or (
                        entry_packet is not None
                        and entry_packet in semantic_packet_paths
                    )
                ):
                    continue
                before_raws = len(related_raw_files)
                before_packets = len(semantic_packet_paths)
                related_raw_files.update(entry_related)
                if entry_packet is not None:
                    semantic_packet_paths.add(entry_packet)
                changed = changed or before_raws != len(related_raw_files)
                changed = changed or before_packets != len(semantic_packet_paths)

            for packet_path, _packet, packet_raws in packet_records:
                if (
                    packet_path in semantic_packet_paths
                    or related_raw_files.intersection(packet_raws)
                ):
                    before_raws = len(related_raw_files)
                    before_packets = len(semantic_packet_paths)
                    related_raw_files.update(packet_raws)
                    semantic_packet_paths.add(packet_path)
                    changed = changed or before_raws != len(related_raw_files)
                    changed = changed or before_packets != len(semantic_packet_paths)

        released_at = datetime.now().isoformat()
        for packet_path in sorted(semantic_packet_paths):
            _release_semantic_defer_packet(packet_path, released_at=released_at)

        removed_entries: list[dict[str, Any]] = []
        for tracked_raw, entry in list(failures.items()):
            remove = tracked_raw == raw_file
            if isinstance(entry, dict) and (
                entry.get("terminal_deferred") is True
                and entry.get("failure_class") == SEMANTIC_NO_QUORUM_FAILURE_CLASS
            ):
                packet_value = entry.get("packet_path")
                entry_packet = (
                    Path(packet_value).expanduser()
                    if isinstance(packet_value, str) and packet_value.strip()
                    else None
                )
                remove = (
                    remove
                    or tracked_raw in related_raw_files
                    or (
                        entry_packet is not None
                        and entry_packet in semantic_packet_paths
                    )
                )
            if not remove:
                continue
            removed = failures.pop(tracked_raw, None)
            if isinstance(removed, dict):
                removed_entries.append(removed)

        operational_failures = state.get("operational_failures")
        if isinstance(operational_failures, dict):
            for fingerprint in {
                entry.get("fingerprint")
                for entry in removed_entries
                if isinstance(entry.get("fingerprint"), str)
            }:
                if not any(
                    isinstance(entry, dict) and entry.get("fingerprint") == fingerprint
                    for entry in failures.values()
                ):
                    operational_failures.pop(fingerprint, None)

        if removed_entries:
            _save_state(state)


def _authority_artifact_sha256_from_error(message: str) -> str | None:
    """Extract the content hash bound to one semantic authority epoch."""

    match = _SEMANTIC_AUTHORITY_MARKER_RE.search(message)
    return match.group(1) if match is not None else None


def classify_failure(message: str | None) -> FailureRecord:
    """Return a stable failure class and fingerprint for a job error."""

    msg = (message or "unknown failure").strip() or "unknown failure"

    if re.search(
        r"\blocal consensus semantic no quorum\b",
        msg,
        flags=re.IGNORECASE,
    ):
        authority_sha256 = _authority_artifact_sha256_from_error(msg)
        if authority_sha256 is None:
            failure_class = "ingest.runtime_local_consensus_authority_unavailable"
            return FailureRecord(
                failure_class=failure_class,
                fingerprint=f"{failure_class}:semantic_no_quorum_authority_invalid",
                message=msg,
            )
        return FailureRecord(
            failure_class=SEMANTIC_NO_QUORUM_FAILURE_CLASS,
            fingerprint=f"{SEMANTIC_NO_QUORUM_FAILURE_CLASS}:{authority_sha256}",
            message=msg,
            authority_artifact_sha256=authority_sha256,
        )

    raw_completion_failure = re.match(
        r"raw completion (receipt publish failed|receipt invalid|"
        r"ACK state pending):\s*(.*)",
        msg,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if raw_completion_failure:
        label = raw_completion_failure.group(1).casefold()
        failure_class = {
            "receipt publish failed": ("ingest.raw_completion_receipt_publish_failed"),
            "receipt invalid": "ingest.raw_completion_receipt_invalid",
            "ack state pending": "ingest.raw_completion_ack_state_pending",
        }[label]
        detail = raw_completion_failure.group(2).strip()
        detail_digest = hashlib.sha256(detail.encode("utf-8")).hexdigest()[:16]
        return FailureRecord(
            failure_class=failure_class,
            fingerprint=f"{failure_class}:{detail_digest}",
            message=msg,
        )

    projection_failure = re.match(
        r"raw semantic projection failed(?:\s*\[([a-z_]+)\])?:\s*([^:\s]+):",
        msg,
        flags=re.IGNORECASE,
    )
    if projection_failure:
        cause = (projection_failure.group(1) or "legacy").casefold()
        exception_type = projection_failure.group(2).casefold()
        if cause == "source_invalid":
            message_digest = hashlib.sha256(msg.encode("utf-8")).hexdigest()[:16]
            return FailureRecord(
                failure_class="raw.semantic_projection_source_invalid",
                fingerprint=(
                    "raw.semantic_projection_source_invalid:"
                    f"{exception_type}:{message_digest}"
                ),
                message=msg,
            )
        cause_classes = {
            "artifact_conflict": (
                "ingest.runtime_semantic_projection_artifact_conflict"
            ),
            "capacity": "ingest.runtime_semantic_projection_capacity",
            "internal_error": "ingest.runtime_semantic_projection_internal_error",
            "interrupted": "ingest.runtime_semantic_projection_interrupted",
            "legacy": "ingest.runtime_semantic_projection_failure",
        }
        failure_class = cause_classes.get(
            cause, "ingest.runtime_semantic_projection_failure"
        )
        message_digest = hashlib.sha256(msg.encode("utf-8")).hexdigest()[:16]
        return FailureRecord(
            failure_class=failure_class,
            fingerprint=f"{failure_class}:{exception_type}:{message_digest}",
            message=msg,
        )

    structured_failure = re.search(
        r"(?:triage|local consensus) structured failure \[([^\]]+)\]:\s*(.*)",
        msg,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if structured_failure:
        local_class = structured_failure.group(1).strip().casefold()
        if local_class in {"repair_exhausted", "repeated_output", "unknown"}:
            return FailureRecord(
                failure_class=f"ingest.runtime_triage_{local_class}",
                fingerprint=f"ingest.runtime_triage_{local_class}",
                message=msg,
            )
        return FailureRecord(
            failure_class=f"ingest.runtime_{local_class}",
            fingerprint=f"ingest.runtime_{local_class}",
            message=msg,
        )

    generation_failure = re.search(
        r"ingest generation (capacity_unavailable|context_window_exceeded|"
        r"context_truncation_suspected|feedback_too_large|stream_incomplete|"
        r"completion_incomplete|output_truncated|transport_error|"
        r"repeated_output|repair_exhausted|validation_failed):",
        msg,
        flags=re.IGNORECASE,
    )
    if generation_failure:
        local_class = generation_failure.group(1).casefold()
        return FailureRecord(
            failure_class=f"ingest.generation_{local_class}",
            fingerprint=f"ingest.generation_{local_class}",
            message=msg,
        )

    authority_unavailable = re.search(
        r"local consensus authority unavailable:\s*(.*)",
        msg,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if authority_unavailable:
        detail = authority_unavailable.group(1).strip().casefold()
        reason_match = re.match(r"([a-z][a-z0-9_.-]{0,127})\s*:", detail)
        reason_code = reason_match.group(1) if reason_match else "unknown"
        failure_class = LOCAL_CONSENSUS_AUTHORITY_FAILURE_CLASS
        return FailureRecord(
            failure_class=failure_class,
            # Bind the control-plane cause, not the raw filename or prose, so
            # simultaneous raws share one operational self-heal packet.
            fingerprint=f"{failure_class}:{reason_code}",
            message=msg,
        )

    if "frontier ingest review did not converge after" in msg.casefold():
        return FailureRecord(
            failure_class="ingest.frontier_nonconvergent",
            fingerprint="ingest.frontier_nonconvergent",
            message=msg,
        )

    if "frontier ingest review deferred:" in msg.casefold():
        return FailureRecord(
            failure_class="ingest.frontier_deferred",
            fingerprint="ingest.frontier_deferred",
            message=msg,
        )

    if "local consensus ingest review did not converge after" in msg.casefold():
        return FailureRecord(
            failure_class="ingest.local_consensus_nonconvergent",
            fingerprint="ingest.local_consensus_nonconvergent",
            message=msg,
        )

    if "local consensus ingest review deferred:" in msg.casefold():
        return FailureRecord(
            failure_class="ingest.local_consensus_deferred",
            fingerprint="ingest.local_consensus_deferred",
            message=msg,
        )

    update_target = re.search(
        r"update target not found for page_id ['\"]([^'\"]+)['\"]",
        msg,
    )
    if update_target:
        page_id = update_target.group(1)
        return FailureRecord(
            failure_class="apply.update_target_not_found",
            fingerprint=f"apply.update_target_not_found:{page_id}",
            message=msg,
            requested_page_id=page_id,
        )

    if "triage parse failed" in msg:
        return FailureRecord(
            failure_class="triage.parse_failed",
            fingerprint="triage.parse_failed",
            message=msg,
        )

    if "index_store unavailable" in msg:
        return FailureRecord(
            failure_class="apply.index_store_unavailable",
            fingerprint="apply.index_store_unavailable",
            message=msg,
        )

    msg_lower = msg.casefold()
    if (
        "sonnet fallback not yet implemented" in msg_lower
        or "ollama unavailable" in msg_lower
    ):
        return FailureRecord(
            failure_class="ingest.ollama_unavailable",
            fingerprint="ingest.ollama_unavailable",
            message=msg,
        )

    digest = hashlib.sha256(msg.encode("utf-8")).hexdigest()[:16]
    return FailureRecord(
        failure_class="unknown",
        fingerprint=f"unknown:{digest}",
        message=msg,
    )


def _safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-._")
    return cleaned[:160] or "failure"


def _similar_existing_pages(requested_page_id: str | None) -> list[str]:
    if not requested_page_id:
        return []

    def loose_key(text: str) -> str:
        return re.sub(r"[^a-z0-9]+", "-", text.casefold()).strip("-")

    target = loose_key(requested_page_id)
    matches: list[str] = []
    for path in chronovisor_store.PAGES_DIR.rglob("*.md"):
        stem = path.stem
        if loose_key(stem) == target:
            try:
                matches.append(str(path.relative_to(chronovisor_store.PAGES_DIR).with_suffix("")))
            except ValueError:
                matches.append(stem)
    return sorted(matches)


def _write_packet(
    *,
    raw_file: str,
    record: FailureRecord,
    attempts: int,
    job_id: str | None,
    raw_text: str | None,
    status: str = "pending_local_repair",
    local_decision: dict[str, Any] | None = None,
    extra_fields: dict[str, Any] | None = None,
) -> Path:
    created_at = datetime.now()
    now = created_at.isoformat()
    source_suffix = hashlib.sha256(
        f"{raw_file}\0{record.fingerprint}".encode()
    ).hexdigest()[:8]
    failure_id = (
        created_at.strftime("%Y%m%d-%H%M%S-%f")
        + "-"
        + _safe_filename(record.fingerprint)
        + "-"
        + source_suffix
    )
    packet = {
        "failure_id": failure_id,
        "created_at": now,
        "raw_file": raw_file,
        "job_id": job_id,
        "failure_class": record.failure_class,
        "fingerprint": record.fingerprint,
        "attempts": attempts,
        "error": record.message,
        "requested_page_id": record.requested_page_id,
        "similar_existing_pages": _similar_existing_pages(record.requested_page_id),
        "status": status,
        "local_model": "qwen",
        "frontier_status": "not_requested",
        "raw_preview": (raw_text or "")[:4000],
    }
    if local_decision is not None:
        packet["local_decision"] = local_decision
    if extra_fields is not None:
        packet.update(extra_fields)
    packets_dir = _runtime_failures_dir() / "packets"
    packets_dir.mkdir(parents=True, exist_ok=True)
    path = packets_dir / f"{failure_id}.json"
    atomic_write(path, json.dumps(packet, indent=2, ensure_ascii=False) + "\n")
    return path


def queue_operational_failure(
    *,
    failure_class: str,
    fingerprint: str,
    message: str,
    evidence: dict[str, Any],
    attempts: int,
    label: str,
    launch: bool = True,
) -> Path:
    """Queue a non-raw runtime failure for bounded local self-heal."""

    record = FailureRecord(
        failure_class=failure_class,
        fingerprint=fingerprint,
        message=message,
    )
    local_decision = {
        "status": "unresolved",
        "action": "none",
        "confidence": 1.0,
        "reason": "bounded operational repair attempts were exhausted",
        "requested_page_id": None,
        "target_page_id": None,
        "notes": "No raw restore is applicable to this derived-runtime failure.",
        "source": "deterministic",
    }
    packet_path = _write_packet(
        raw_file=label,
        record=record,
        attempts=max(1, attempts),
        job_id=None,
        raw_text=json.dumps(evidence, ensure_ascii=False, default=str),
        status="pending_local_repair",
        local_decision=local_decision,
    )
    runtime_status.safe_append_event(
        "warn",
        f"failure-supervisor | queued operational self-heal for {label}",
        source="failure-supervisor",
        failure_class=failure_class,
        fingerprint=fingerprint,
        packet_path=str(packet_path),
        outcome_kind="self_heal_queued",
    )
    if launch:
        _launch_self_heal(packet_path)
    return packet_path


def _launch_self_heal(packet_path: Path) -> str | None:
    """Submit a published packet and return a serializable launch error."""

    try:
        from chronovisor.core.background_jobs import start_self_heal_background

        start_self_heal_background(packet_path)
    except Exception as exc:
        error = f"{exc.__class__.__name__}: {exc}"
        runtime_status.safe_append_event(
            "warn",
            f"failure-supervisor | self-heal launch failed: {error}",
            source="failure-supervisor",
            packet_path=str(packet_path),
            outcome_kind="self_heal_launch_failed",
        )
        return error
    return None


def _quarantine_raw(raw_path: Path, packet_path: Path) -> Path | None:
    if not raw_path.exists():
        return None
    quarantine_dir = _runtime_failures_dir() / "quarantined-raw"
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    target = quarantine_dir / raw_path.name
    if target.exists():
        suffix = datetime.now().strftime("%Y%m%d-%H%M%S")
        target = quarantine_dir / f"{raw_path.stem}-{suffix}{raw_path.suffix}"
    shutil.move(str(raw_path), str(target))

    pointer = target.with_suffix(target.suffix + ".packet")
    atomic_write(pointer, str(packet_path) + "\n")
    return target


def _current_adopted_authority_sha256() -> str | None:
    """Return only the artifact hash of a currently valid adopted authority.

    A byte-different nominated file is not sufficient evidence for reopening a
    semantic hold.  The artifact may be partial, corrupt, unevaluated, or bound
    to model metadata that is no longer installed.  Reuse the router's full
    adoption resolver and fail closed whenever it cannot prove that the new
    artifact is the live authority.
    """

    try:
        from chronovisor.core import runtime_config
        from chronovisor.decision.decision_router import resolve_router_policy

        loader = runtime_config.load_decision_router_config
        config = (
            loader(chronovisor_store.CHRONOVISOR_ROOT / "config.toml")
            if getattr(loader, "__module__", "") == runtime_config.__name__
            else loader()
        )
        resolution = resolve_router_policy(config)
    except Exception:
        return None
    artifact_sha256 = resolution.artifact_sha256
    if (
        resolution.source != "adopted_artifact"
        or resolution.error is not None
        or not isinstance(artifact_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", artifact_sha256) is None
    ):
        return None
    return artifact_sha256


def _current_adopted_authority_epoch() -> str | None:
    """Fingerprint the adopted artifact together with executable policy code.

    The artifact identifies the evaluated model triplet, but a safe router
    contract can improve without changing those model weights. Semantic holds
    must be retried after such a contract change instead of remaining bound to
    the artifact digest forever.
    """

    artifact_sha256 = _current_adopted_authority_sha256()
    if artifact_sha256 is None:
        return None
    try:
        from chronovisor.decision.decision_lane_contracts import (
            LANE_CONTRACT_POLICY_VERSION,
            lane_contract_manifest_sha256,
        )
        from chronovisor.decision.decision_router import (
            DECISION_SEMANTICS_POLICY_VERSION,
            QUORUM_SAFETY_POLICY_VERSION,
        )

        payload = {
            "artifact_sha256": artifact_sha256,
            "decision_semantics_policy_version": DECISION_SEMANTICS_POLICY_VERSION,
            "lane_contract_policy_version": LANE_CONTRACT_POLICY_VERSION,
            "lane_contract_manifest_sha256": lane_contract_manifest_sha256(),
            "quorum_safety_policy_version": QUORUM_SAFETY_POLICY_VERSION,
        }
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except Exception:
        return None
    return hashlib.sha256(encoded).hexdigest()


def _semantic_unit_paths(
    raw_path: Path,
    related_raw_paths: Sequence[Path],
) -> tuple[Path, ...]:
    paths = tuple(
        sorted(
            dict.fromkeys(
                [
                    raw_path,
                    *(path for path in related_raw_paths if isinstance(path, Path)),
                ]
            ),
            key=lambda path: path.name,
        )
    )
    filenames = [path.name for path in paths]
    if not paths or len(filenames) != len(set(filenames)):
        raise ValueError("semantic defer source filenames must be unique")
    return paths


def _semantic_source_evidence(paths: Sequence[Path]) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for path in paths:
        raw = path.read_bytes()
        evidence.append(
            {
                "filename": path.name,
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    return evidence


def _raw_source_evidence(path: Path) -> dict[str, Any] | None:
    """Return an exact-byte binding for one source raw when it is readable."""

    try:
        raw = path.read_bytes()
    except OSError:
        return None
    return {
        "filename": path.name,
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _read_packet_object(packet_path: Path) -> dict[str, Any] | None:
    try:
        packet = json.loads(packet_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return packet if isinstance(packet, dict) else None


def _safe_raw_filenames(value: Any) -> set[str]:
    if not isinstance(value, list):
        return set()
    return {
        filename
        for filename in value
        if isinstance(filename, str)
        and filename
        and Path(filename).name == filename
        and filename not in {".", ".."}
    }


def _semantic_packet_source_raws(
    packet: dict[str, Any],
    *,
    verify_sources: bool,
    raw_store: Any | None = None,
) -> frozenset[str] | None:
    """Validate packet source evidence and optionally bind it to RAW_DIR bytes."""

    source_raws = packet.get("source_raws")
    if not isinstance(source_raws, list) or not source_raws:
        return None
    filenames: set[str] = set()
    for source in source_raws:
        if not isinstance(source, dict):
            return None
        filename = source.get("filename")
        byte_count = source.get("bytes")
        sha256 = source.get("sha256")
        if (
            not isinstance(filename, str)
            or not filename
            or Path(filename).name != filename
            or filename in {".", ".."}
            or filename in filenames
            or isinstance(byte_count, bool)
            or not isinstance(byte_count, int)
            or byte_count < 0
            or not isinstance(sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", sha256) is None
        ):
            return None
        if verify_sources:
            try:
                from chronovisor.core.raw_store import RawStore

                store = raw_store or RawStore(chronovisor_store.RAW_DIR)
                unit = store.resolve(filename)
                if unit is None:
                    return None
                raw = store.read_bytes(unit)
            except (OSError, ValueError):
                return None
            if len(raw) != byte_count or hashlib.sha256(raw).hexdigest() != sha256:
                return None
        filenames.add(filename)
    return frozenset(filenames)


def _semantic_defer_packet_records(
    *,
    verify_sources: bool,
) -> list[tuple[Path, dict[str, Any], frozenset[str]]]:
    """Read active semantic packets; superseded/released packets are excluded."""

    packets_dir = _runtime_failures_dir() / "packets"
    try:
        packet_paths = sorted(packets_dir.glob("*.json"))
    except OSError:
        return []
    raw_store = None
    if verify_sources:
        from chronovisor.core.raw_store import RawStore

        raw_store = RawStore(chronovisor_store.RAW_DIR)
    records: list[tuple[Path, dict[str, Any], frozenset[str]]] = []
    for packet_path in packet_paths:
        packet = _read_packet_object(packet_path)
        if (
            packet is None
            or packet.get("status") != "local_quarantined"
            or packet.get("terminal_deferred") is not True
            or packet.get("failure_class") != SEMANTIC_NO_QUORUM_FAILURE_CLASS
        ):
            continue
        authority_sha256 = packet.get("authority_artifact_sha256")
        if (
            not isinstance(authority_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", authority_sha256) is None
        ):
            continue
        source_raws = _semantic_packet_source_raws(
            packet,
            verify_sources=verify_sources,
            raw_store=raw_store,
        )
        if source_raws is None:
            continue
        records.append((packet_path, packet, source_raws))
    return records


def current_adopted_authority_sha256() -> str | None:
    return _current_adopted_authority_sha256()


def current_adopted_authority_epoch() -> str | None:
    return _current_adopted_authority_epoch()


def semantic_defer_packet_records(
    *,
    verify_sources: bool,
) -> list[tuple[Path, dict[str, Any], frozenset[str]]]:
    return _semantic_defer_packet_records(verify_sources=verify_sources)


def _release_semantic_defer_packet(
    packet_path: Path,
    *,
    released_at: str,
) -> None:
    """Atomically make one semantic packet ineligible for hold reconstruction."""

    packet = _read_packet_object(packet_path)
    if (
        packet is None
        or packet.get("status") != "local_quarantined"
        or packet.get("terminal_deferred") is not True
        or packet.get("failure_class") != SEMANTIC_NO_QUORUM_FAILURE_CLASS
    ):
        return
    packet.update(
        {
            "status": "semantic_defer_released",
            "terminal_deferred": False,
            "released_at": released_at,
            "updated_at": released_at,
        }
    )
    atomic_write(
        packet_path,
        json.dumps(packet, indent=2, ensure_ascii=False) + "\n",
    )


def _reusable_semantic_defer_packet(
    failures: dict[str, Any],
    *,
    filenames: Sequence[str],
    record: FailureRecord,
    source_evidence: Sequence[dict[str, Any]],
    authority_epoch: str,
) -> tuple[Path, dict[str, Any]] | None:
    evidence_by_name = {
        str(row.get("filename")): row
        for row in source_evidence
        if isinstance(row, dict) and isinstance(row.get("filename"), str)
    }
    for filename in filenames:
        entry = failures.get(filename)
        if (
            not isinstance(entry, dict)
            or entry.get("terminal_deferred") is not True
            or entry.get("failure_class") != SEMANTIC_NO_QUORUM_FAILURE_CLASS
            or entry.get("fingerprint") != record.fingerprint
            or entry.get("authority_artifact_sha256")
            != record.authority_artifact_sha256
            or entry.get("authority_epoch") != authority_epoch
        ):
            continue
        packet_value = entry.get("packet_path")
        if not isinstance(packet_value, str):
            continue
        packet_path = Path(packet_value).expanduser()
        try:
            packet = json.loads(packet_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        if (
            not isinstance(packet, dict)
            or packet.get("status") != "local_quarantined"
            or packet.get("terminal_deferred") is not True
            or packet.get("authority_artifact_sha256")
            != record.authority_artifact_sha256
            or packet.get("authority_epoch") != authority_epoch
        ):
            continue
        prior_evidence = packet.get("source_raws")
        if not isinstance(prior_evidence, list):
            continue
        prior_by_name = {
            str(row.get("filename")): row
            for row in prior_evidence
            if isinstance(row, dict) and isinstance(row.get("filename"), str)
        }
        if any(
            name in prior_by_name and prior_by_name[name] != evidence
            for name, evidence in evidence_by_name.items()
        ):
            continue
        return packet_path, packet
    return None


def _supersede_replaced_operational_packets(
    state: dict[str, Any],
    *,
    replaced_entries: Sequence[dict[str, Any]],
    superseded_by_packet: Path,
    superseded_at: str,
) -> None:
    """Retire unshared operational packets replaced by a semantic defer."""

    failures = state.get("failures")
    if not isinstance(failures, dict):
        return
    operational_failures = state.get("operational_failures")

    def effective_packet_path(entry: dict[str, Any]) -> tuple[str, str] | None:
        """Resolve a direct or exact legacy registry packet binding."""

        fingerprint = entry.get("fingerprint")
        if not isinstance(fingerprint, str) or not fingerprint:
            return None
        packet_path = entry.get("packet_path")
        if isinstance(packet_path, str) and packet_path:
            return fingerprint, packet_path
        if not isinstance(operational_failures, dict):
            return None
        registry_entry = operational_failures.get(fingerprint)
        if (
            not isinstance(registry_entry, dict)
            or registry_entry.get("fingerprint") != fingerprint
        ):
            return None
        registry_path = registry_entry.get("packet_path")
        if not isinstance(registry_path, str) or not registry_path:
            return None
        return fingerprint, registry_path

    candidates: dict[str, tuple[str, str]] = {}
    for entry in replaced_entries:
        if entry.get("terminal_deferred") is True:
            continue
        resolved = effective_packet_path(entry)
        if resolved is None:
            continue
        fingerprint, packet_path = resolved
        candidates[packet_path] = (fingerprint, packet_path)

    for fingerprint, packet_value in candidates.values():
        if any(
            isinstance(entry, dict)
            and (
                entry.get("packet_path") == packet_value
                or effective_packet_path(entry) == (fingerprint, packet_value)
            )
            for entry in failures.values()
        ):
            continue
        packet_path = Path(packet_value).expanduser()
        try:
            packet = json.loads(packet_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        if (
            not isinstance(packet, dict)
            or packet.get("fingerprint") != fingerprint
            or packet.get("terminal_deferred") is True
            or packet.get("status") not in _CANCELLABLE_OPERATIONAL_PACKET_STATUSES
        ):
            continue
        from chronovisor.core.self_heal_cancellation import (
            request_packet_cancellation,
        )

        cancellation = request_packet_cancellation(
            packet_path,
            reason="semantic_no_quorum_terminal_defer",
            superseded_by_packet=superseded_by_packet,
        )
        if cancellation.get("accepted") is not True:
            continue
        packet.update(
            {
                "status": "superseded_semantic_defer",
                "self_heal_queued": False,
                "next_attempt_at": None,
                "cancellation_requested_at": cancellation.get("requested_at"),
                "cancellation_path": cancellation.get("cancellation_path"),
                "superseded_at": superseded_at,
                "superseded_by_packet": str(superseded_by_packet),
                "updated_at": superseded_at,
            }
        )
        atomic_write(
            packet_path,
            json.dumps(packet, indent=2, ensure_ascii=False) + "\n",
        )
        if isinstance(operational_failures, dict):
            operational_entry = operational_failures.get(fingerprint)
            if (
                isinstance(operational_entry, dict)
                and operational_entry.get("packet_path") == packet_value
            ):
                operational_failures.pop(fingerprint, None)


def _record_semantic_no_quorum_defer_unlocked(
    *,
    raw_path: Path,
    record: FailureRecord,
    job_id: str | None,
    raw_text: str | None,
    related_raw_paths: Sequence[Path],
) -> SupervisionResult:
    """Bind one immutable semantic unit to a terminal, model-free defer."""

    authority_sha256 = record.authority_artifact_sha256
    if authority_sha256 is None:
        raise ValueError("semantic no-quorum defer requires an authority artifact hash")
    authority_epoch = _current_adopted_authority_epoch()
    if authority_epoch is None:
        raise ValueError("semantic no-quorum defer requires a valid authority epoch")
    source_paths = _semantic_unit_paths(raw_path, related_raw_paths)
    source_evidence = _semantic_source_evidence(source_paths)
    filenames = [path.name for path in source_paths]
    now = datetime.now().isoformat()
    state = _load_state()
    failures = state.setdefault("failures", {})
    if not isinstance(failures, dict):
        failures = {}
        state["failures"] = failures
    replaced_entries = [
        dict(entry)
        for filename in filenames
        if isinstance((entry := failures.get(filename)), dict)
    ]

    reusable = _reusable_semantic_defer_packet(
        failures,
        filenames=filenames,
        record=record,
        source_evidence=source_evidence,
        authority_epoch=authority_epoch,
    )
    if reusable is None:
        packet_path = _write_packet(
            raw_file=raw_path.name,
            record=record,
            attempts=1,
            job_id=job_id,
            raw_text=raw_text,
            status="local_quarantined",
            extra_fields={
                "frontier_status": "not_requested",
                "terminal_deferred": True,
                "self_heal_queued": False,
                "defer_reason": SEMANTIC_NO_QUORUM_DEFER_REASON,
                "authority_artifact_sha256": authority_sha256,
                "authority_epoch": authority_epoch,
                "related_raw_files": filenames,
                "source_raws": source_evidence,
                "quarantined_at": now,
                "next_attempt_at": None,
            },
        )
        packet = json.loads(packet_path.read_text(encoding="utf-8"))
    else:
        packet_path, packet = reusable
        prior_evidence = packet.get("source_raws")
        combined_evidence = {
            str(row["filename"]): dict(row)
            for row in prior_evidence
            if isinstance(row, dict) and isinstance(row.get("filename"), str)
        }
        combined_evidence.update(
            {
                str(row["filename"]): dict(row)
                for row in source_evidence
                if isinstance(row.get("filename"), str)
            }
        )
        combined_files = sorted(combined_evidence)
        if packet.get("related_raw_files") != combined_files or packet.get(
            "source_raws"
        ) != [combined_evidence[name] for name in combined_files]:
            packet["related_raw_files"] = combined_files
            packet["source_raws"] = [combined_evidence[name] for name in combined_files]
            packet["updated_at"] = now
            atomic_write(
                packet_path,
                json.dumps(packet, indent=2, ensure_ascii=False) + "\n",
            )
        filenames = combined_files

    first_seen_at = str(packet.get("created_at") or now)
    entry = {
        "fingerprint": record.fingerprint,
        "failure_class": record.failure_class,
        "attempts": 1,
        "first_seen_at": first_seen_at,
        "last_seen_at": now,
        "last_error": record.message,
        "job_id": job_id,
        "terminal_deferred": True,
        "self_heal_queued": False,
        "defer_reason": SEMANTIC_NO_QUORUM_DEFER_REASON,
        "authority_artifact_sha256": authority_sha256,
        "authority_epoch": authority_epoch,
        "packet_path": str(packet_path),
        "related_raw_files": list(filenames),
    }
    evidence_by_name = {
        str(row["filename"]): row
        for row in packet.get("source_raws", [])
        if isinstance(row, dict) and isinstance(row.get("filename"), str)
    }
    for filename in filenames:
        per_raw_entry = dict(entry)
        evidence = evidence_by_name.get(filename)
        if evidence is not None:
            per_raw_entry["raw_sha256"] = evidence.get("sha256")
            per_raw_entry["raw_bytes"] = evidence.get("bytes")
        failures[filename] = per_raw_entry
    _supersede_replaced_operational_packets(
        state,
        replaced_entries=replaced_entries,
        superseded_by_packet=packet_path,
        superseded_at=now,
    )
    _save_state(state)
    runtime_status.safe_append_event(
        "warn",
        f"failure-supervisor | terminal semantic defer for {raw_path.name}",
        source="failure-supervisor",
        raw_file=raw_path.name,
        related_raw_files=list(filenames),
        failure_class=record.failure_class,
        fingerprint=record.fingerprint,
        authority_artifact_sha256=authority_sha256,
        authority_epoch=authority_epoch,
        packet_path=str(packet_path),
        outcome_kind="terminal_semantic_defer",
    )
    return SupervisionResult(
        raw_file=raw_path.name,
        failure_class=record.failure_class,
        fingerprint=record.fingerprint,
        attempts=1,
        packet_path=str(packet_path),
        terminal_deferred=True,
    )


def record_semantic_no_quorum_defer(
    *,
    raw_path: Path,
    error: str | None,
    job_id: str | None = None,
    raw_text: str | None = None,
    related_raw_paths: Sequence[Path] = (),
) -> SupervisionResult:
    """Record a terminal semantic no-quorum unit without moving or retrying it."""

    record = classify_failure(error)
    if record.failure_class != SEMANTIC_NO_QUORUM_FAILURE_CLASS:
        raise ValueError("error is not an authority-bound semantic no-quorum failure")
    with _failure_state_lock():
        return _record_semantic_no_quorum_defer_unlocked(
            raw_path=raw_path,
            record=record,
            job_id=job_id,
            raw_text=raw_text,
            related_raw_paths=related_raw_paths,
        )


def record_current_semantic_no_quorum_defer(
    *,
    raw_path: Path,
    error: str | None,
    job_id: str | None = None,
    raw_text: str | None = None,
    related_raw_paths: Sequence[Path] = (),
) -> SupervisionResult | None:
    """Publish a semantic defer only while its adopted artifact is current.

    Adoption writers use the same authority lease.  The current-artifact CAS
    and failure-state publication therefore form one transaction, with a
    single lock order of authority then failure state.  A stale or invalid
    marker leaves the immutable raw pending for the next authority epoch and
    must not cancel an existing operational repair packet.
    """

    record = classify_failure(error)
    if record.failure_class != SEMANTIC_NO_QUORUM_FAILURE_CLASS:
        raise ValueError("error is not an authority-bound semantic no-quorum failure")
    authority_sha256 = record.authority_artifact_sha256
    if authority_sha256 is None:
        raise ValueError("semantic no-quorum defer requires an authority artifact hash")

    from chronovisor.core.page_mutation import decision_authority_lock

    authority_lock_path = chronovisor_store.CHRONOVISOR_ROOT / "runtime" / "decision-authority.lock"
    with decision_authority_lock(authority_lock_path):
        if _current_adopted_authority_sha256() != authority_sha256:
            return None
        with _failure_state_lock():
            return _record_semantic_no_quorum_defer_unlocked(
                raw_path=raw_path,
                record=record,
                job_id=job_id,
                raw_text=raw_text,
                related_raw_paths=related_raw_paths,
            )


def record_semantic_no_quorum_defer_unless_operational_hold(
    *,
    raw_path: Path,
    error: str | None,
    job_id: str | None = None,
    raw_text: str | None = None,
    related_raw_paths: Sequence[Path] = (),
) -> SupervisionResult | None:
    """Publish a semantic defer only if no newer operational repair owns the raw."""

    record = classify_failure(error)
    if record.failure_class != SEMANTIC_NO_QUORUM_FAILURE_CLASS:
        raise ValueError("error is not an authority-bound semantic no-quorum failure")
    source_paths = _semantic_unit_paths(raw_path, related_raw_paths)
    with _failure_state_lock():
        active = _operational_deferred_raw_files_unlocked(source_paths)
        source_names = {path.name for path in source_paths}
        if any(
            active.get(name) not in {None, SEMANTIC_NO_QUORUM_DEFER_REASON}
            for name in source_names
        ):
            return None
        return _record_semantic_no_quorum_defer_unlocked(
            raw_path=raw_path,
            record=record,
            job_id=job_id,
            raw_text=raw_text,
            related_raw_paths=related_raw_paths,
        )


def _record_operational_raw_failure(
    *,
    raw_path: Path,
    record: FailureRecord,
    job_id: str | None,
    raw_text: str | None,
    related_raw_paths: Sequence[Path] = (),
) -> tuple[SupervisionResult, Path | None]:
    """Queue one self-heal packet without blaming or moving the source raw."""

    raw_file = raw_path.name
    source_paths = tuple(
        dict.fromkeys(
            [raw_path, *(path for path in related_raw_paths if isinstance(path, Path))]
        )
    )
    raw_files = tuple(path.name for path in source_paths)
    source_evidence = {
        str(row["filename"]): row for row in _semantic_source_evidence(source_paths)
    }
    state = _load_state()
    failures = state.setdefault("failures", {})
    if not isinstance(failures, dict):
        failures = {}
        state["failures"] = failures
    operational_failures = state.setdefault("operational_failures", {})
    if not isinstance(operational_failures, dict):
        operational_failures = {}
        state["operational_failures"] = operational_failures

    def bind_raws(
        *,
        packet_path: str | None,
        launch_status: object,
        launch_error: object,
        first_seen_at: str,
    ) -> None:
        for source_file in raw_files:
            prior = failures.get(source_file)
            entry = {
                "fingerprint": record.fingerprint,
                "failure_class": record.failure_class,
                "attempts": 1,
                "first_seen_at": first_seen_at,
                "last_seen_at": datetime.now().isoformat(),
                "last_error": record.message,
                "job_id": job_id,
                "self_heal_queued": True,
                "packet_path": packet_path,
                "launch_status": launch_status,
                "launch_error": launch_error,
                "related_raw_files": list(raw_files),
            }
            # Fresh incidents bind the exact source bytes at failure time.  A
            # row already attached to this packet but lacking these fields is
            # legacy state; do not silently bless its current bytes as its
            # historical preimage.  The release CLI must supply that manifest.
            same_prior_binding = (
                isinstance(prior, dict)
                and prior.get("fingerprint") == record.fingerprint
                and prior.get("packet_path") == packet_path
            )
            if same_prior_binding:
                # Never repair historical evidence from the raw's current
                # bytes. A complete binding stays complete, a legacy row with
                # neither field stays legacy, and partial/malformed evidence
                # is preserved verbatim so the release CAS fails closed.
                assert isinstance(prior, dict)
                for field in ("raw_sha256", "raw_bytes"):
                    if field in prior:
                        entry[field] = prior[field]
            else:
                evidence = source_evidence[source_file]
                entry["raw_sha256"] = evidence["sha256"]
                entry["raw_bytes"] = evidence["bytes"]
            failures[source_file] = entry

    current = failures.get(raw_file)
    if isinstance(current, dict) and (
        _operational_entry_is_released(raw_path, current)
    ):
        released_packet = current.get("packet_path")
        released_fingerprint = current.get("fingerprint")
        if isinstance(released_packet, str) and _packet_status(current) in (
            REPAIR_SUCCESS_PACKET_STATUSES
        ):
            for source_file, value in list(failures.items()):
                if (
                    isinstance(value, dict)
                    and value.get("packet_path") == released_packet
                ):
                    failures.pop(source_file, None)
        else:
            failures.pop(raw_file, None)
        if isinstance(released_fingerprint, str) and not any(
            isinstance(value, dict) and value.get("fingerprint") == released_fingerprint
            for value in failures.values()
        ):
            operational_failures.pop(released_fingerprint, None)
        current = None
    if isinstance(current, dict) and current.get("fingerprint") != record.fingerprint:
        previous_fingerprint = current.get("fingerprint")
        failures.pop(raw_file, None)
        if isinstance(previous_fingerprint, str) and not any(
            isinstance(value, dict) and value.get("fingerprint") == previous_fingerprint
            for value in failures.values()
        ):
            operational_failures.pop(previous_fingerprint, None)
        current = None
    if (
        isinstance(current, dict)
        and current.get("fingerprint") == record.fingerprint
        and current.get("self_heal_queued") is True
    ):
        packet_path = current.get("packet_path")
        bind_raws(
            packet_path=packet_path if isinstance(packet_path, str) else None,
            launch_status=current.get("launch_status"),
            launch_error=current.get("launch_error"),
            first_seen_at=str(
                current.get("first_seen_at") or datetime.now().isoformat()
            ),
        )
        _save_state(state)
        return (
            SupervisionResult(
                raw_file=raw_file,
                failure_class=record.failure_class,
                fingerprint=record.fingerprint,
                attempts=max(1, int(current.get("attempts", 1))),
                packet_path=packet_path if isinstance(packet_path, str) else None,
            ),
            None,
        )

    now = datetime.now().isoformat()
    queued = operational_failures.get(record.fingerprint)
    if (
        isinstance(queued, dict)
        and _packet_status(queued) in REPAIR_SUCCESS_PACKET_STATUSES
    ):
        # A completed packet proves only the earlier incident was repaired.
        # Reusing it for a later recurrence would make the new raw appear
        # released immediately and would suppress a fresh repair attempt.
        completed_packet = queued.get("packet_path")
        if isinstance(completed_packet, str):
            for source_file, value in list(failures.items()):
                if (
                    isinstance(value, dict)
                    and value.get("packet_path") == completed_packet
                ):
                    failures.pop(source_file, None)
        operational_failures.pop(record.fingerprint, None)
        queued = None
    if isinstance(queued, dict) and queued.get("self_heal_queued") is True:
        packet_path = queued.get("packet_path")
        bind_raws(
            packet_path=packet_path if isinstance(packet_path, str) else None,
            launch_status=queued.get("launch_status"),
            launch_error=queued.get("launch_error"),
            first_seen_at=now,
        )
        _save_state(state)
        return (
            SupervisionResult(
                raw_file=raw_file,
                failure_class=record.failure_class,
                fingerprint=record.fingerprint,
                attempts=1,
                packet_path=packet_path if isinstance(packet_path, str) else None,
            ),
            None,
        )

    packet_path = queue_operational_failure(
        failure_class=record.failure_class,
        fingerprint=record.fingerprint,
        message=record.message,
        evidence={
            "raw_file": raw_file,
            "raw_files": list(raw_files),
            "source_raws": [source_evidence[name] for name in raw_files],
            "job_id": job_id,
            "raw_preview": (raw_text or "")[:4000],
        },
        attempts=1,
        label=raw_file,
        launch=False,
    )
    bind_raws(
        packet_path=str(packet_path),
        launch_status="submitted",
        launch_error=None,
        first_seen_at=now,
    )
    operational_failures[record.fingerprint] = {
        "fingerprint": record.fingerprint,
        "failure_class": record.failure_class,
        "attempts": 1,
        "first_seen_at": now,
        "last_seen_at": datetime.now().isoformat(),
        "self_heal_queued": True,
        "packet_path": str(packet_path),
        "source_raw_file": raw_file,
        "launch_status": "submitted",
        "launch_error": None,
    }
    _save_state(state)

    return (
        SupervisionResult(
            raw_file=raw_file,
            failure_class=record.failure_class,
            fingerprint=record.fingerprint,
            attempts=1,
            packet_path=str(packet_path),
        ),
        packet_path,
    )


def _record_operational_launch_result(
    packet_path: Path, launch_error: str | None
) -> None:
    """Persist launch diagnostics without overwriting newer state entries."""

    launch_status = "failed" if launch_error else "started"
    packet_value = str(packet_path)
    with _failure_state_lock():
        state = _load_state()
        failures = state.get("failures")
        if isinstance(failures, dict):
            for current in failures.values():
                if (
                    isinstance(current, dict)
                    and current.get("packet_path") == packet_value
                ):
                    current["launch_status"] = launch_status
                    current["launch_error"] = launch_error
        operational = state.get("operational_failures")
        if isinstance(operational, dict):
            for queued in operational.values():
                if (
                    isinstance(queued, dict)
                    and queued.get("packet_path") == packet_value
                ):
                    queued["launch_status"] = launch_status
                    queued["launch_error"] = launch_error
                    queued["launch_attempted_at"] = datetime.now().isoformat()
        _save_state(state)


def verified_projection_child_bytes(raw_file: str, *, artifact_dir: Path) -> bytes:
    """Read an exact child only after its durable projection bundle verifies."""

    match = re.fullmatch(
        r"semantic-([0-9a-f]{64})-child-[0-9]{8}-[0-9a-f]{64}\.md",
        raw_file,
    )
    if match is None:
        raise ValueError("affected_raw_missing")
    child_path = artifact_dir / raw_file
    manifest_path = artifact_dir / f"semantic-{match.group(1)}.manifest.json"
    if child_path.is_symlink() or manifest_path.is_symlink():
        raise ValueError("projection_evidence_symlink")
    resolved_dir = artifact_dir.resolve(strict=True)
    resolved_child = child_path.resolve(strict=True)
    if (
        resolved_child.parent != resolved_dir
        or manifest_path.resolve(strict=True).parent != resolved_dir
    ):
        raise ValueError("projection_evidence_outside_artifact_dir")

    from chronovisor.ingest import raw_semantic_projection

    manifest = raw_semantic_projection.verify_projection_bundle(manifest_path)
    children = manifest.get("children")
    if (
        not isinstance(children, list)
        or sum(
            isinstance(row, dict) and row.get("filename") == raw_file
            for row in children
        )
        != 1
    ):
        raise ValueError("projection_child_not_in_manifest")
    return child_path.read_bytes()


def _valid_projection_child_bundle(raw_path: Path, entry: dict[str, Any]) -> bool:
    """Return true only when a previously failed child now has a valid bundle."""

    if (
        entry.get("failure_class")
        not in SEMANTIC_PROJECTION_OPERATIONAL_FAILURE_CLASSES
    ):
        return False
    try:
        verified_projection_child_bytes(raw_path.name, artifact_dir=raw_path.parent)
    except (OSError, ValueError, TypeError):
        return False
    return True


def _projection_parent_can_retry(raw_path: Path, entry: dict[str, Any]) -> bool:
    """Allow only completed or intent-first incomplete parent bundles to retry."""

    if (
        entry.get("failure_class")
        not in SEMANTIC_PROJECTION_OPERATIONAL_FAILURE_CLASSES
        or not raw_path.is_file()
    ):
        return False
    try:
        from chronovisor.ingest.raw_semantic_projection import (
            projection_bundle_state_for_parent,
        )

        state = projection_bundle_state_for_parent(raw_path)
    except (OSError, ValueError, TypeError):
        return False
    # completed: the crash happened after durable bundle publication;
    # incomplete: manifest-first publication is explicitly resumable.
    # absent/invalid remain fail-closed and require repair evidence.
    return state in {"completed", "incomplete"}


def _packet_status(entry: dict[str, Any]) -> str:
    packet_value = entry.get("packet_path")
    if not isinstance(packet_value, str) or not packet_value.strip():
        return "packet_missing"
    packet_path = Path(packet_value).expanduser()
    if not packet_path.is_file():
        return "packet_missing"
    try:
        packet = json.loads(packet_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return "packet_invalid"
    if not isinstance(packet, dict) or not isinstance(packet.get("status"), str):
        return "packet_invalid"
    return str(packet["status"]) or "packet_invalid"


def _operational_entry_is_released(raw_path: Path, entry: dict[str, Any]) -> bool:
    return (
        _valid_projection_child_bundle(raw_path, entry)
        or _packet_status(entry) in REPAIR_SUCCESS_PACKET_STATUSES
    )


def operational_deferred_raw_files(
    raw_paths: Iterable[Path] | None = None,
) -> dict[str, str]:
    """Reconcile repair packets and return fail-closed per-raw defer statuses.

    Operational failures blame the runtime, not the immutable source raw.  A
    queued source stays on disk but is omitted from inference until its packet
    records an explicit successful repair.  Missing, malformed, active, and
    terminal-quarantine packets all remain deferred.
    """

    with _failure_state_lock(exclusive=False):
        return _operational_deferred_raw_files_unlocked(raw_paths)


def _operational_deferred_raw_files_unlocked(
    raw_paths: Iterable[Path] | None,
) -> dict[str, str]:
    state = _load_state()
    failures = state.get("failures")
    if not isinstance(failures, dict):
        failures = {}
    if raw_paths is None:
        from chronovisor.core.raw_store import RawStore

        store = RawStore(chronovisor_store.RAW_DIR)
        reference_dir = chronovisor_store.RAW_DIR.parent / "runtime" / "raw-projections" / "parents"
        available_paths = {
            unit.raw_id: (
                unit.path
                if unit.storage == "legacy_file"
                else store.materialize_ingest(unit, reference_dir)
            )
            for unit in store.iter_units()
        }
    else:
        available_paths = {path.name: path for path in raw_paths}
    deferred: dict[str, str] = {}
    authority_sha256_loaded = False
    current_authority_sha256: str | None = None
    authority_epoch_loaded = False
    current_authority_epoch: str | None = None

    for raw_file, value in list(failures.items()):
        if not isinstance(raw_file, str) or not isinstance(value, dict):
            continue
        if (
            value.get("terminal_deferred") is True
            and value.get("failure_class") == SEMANTIC_NO_QUORUM_FAILURE_CLASS
        ):
            if _packet_status(value) in SEMANTIC_DEFER_RELEASE_PACKET_STATUSES:
                # reset_raw_failure publishes the packet release before state
                # cleanup. A crash between those writes must stay released.
                continue
            stored_authority_epoch = value.get(
                "authority_epoch",
                value.get("authority_artifact_sha256"),
            )
            if not authority_epoch_loaded:
                current_authority_epoch = _current_adopted_authority_epoch()
                authority_epoch_loaded = True
            if (
                not isinstance(stored_authority_epoch, str)
                or not re.fullmatch(r"[0-9a-f]{64}", stored_authority_epoch)
                or current_authority_epoch is None
                or current_authority_epoch == stored_authority_epoch
            ):
                deferred[raw_file] = SEMANTIC_NO_QUORUM_DEFER_REASON
            # A different fully validated executable authority epoch releases
            # the immutable raw for automatic re-evaluation.
            continue
        if (
            value.get("failure_class") not in OPERATIONAL_SELF_HEAL_FAILURE_CLASSES
            and value.get("self_heal_queued") is not True
        ):
            continue
        if value.get("fingerprint") == ADOPTION_ARTIFACT_INVALID_FINGERPRINT:
            if not authority_sha256_loaded:
                current_authority_sha256 = _current_adopted_authority_sha256()
                authority_sha256_loaded = True
            if current_authority_sha256 is not None:
                # This control-plane failure is bound to the invalid
                # nomination, not to the immutable raw. A newly validated
                # adopted artifact proves that condition has cleared, so the
                # raw can automatically re-enter ingest. Successful ingest
                # then removes the historical failure-state row normally.
                continue
        raw_path = available_paths.get(raw_file)
        if raw_path is None:
            from chronovisor.core.raw_store import RawStore

            store = RawStore(chronovisor_store.RAW_DIR)
            unit = store.resolve(raw_file)
            if unit is None:
                deferred[raw_file] = _packet_status(value)
                continue
            raw_path = store.materialize_ingest(
                unit,
                chronovisor_store.RAW_DIR.parent / "runtime" / "raw-projections" / "parents",
            )
        if _operational_entry_is_released(raw_path, value):
            continue
        if _projection_parent_can_retry(raw_path, value):
            continue
        # Unknown statuses are deliberately fail closed too. A future repair
        # state must be explicitly added to the success allowlist before it
        # can re-enable inference.
        deferred[raw_file] = _packet_status(value)

    # The semantic packet is published before state.json. Rebuild the hold
    # directly from byte-bound packet evidence so a crash, missing state file,
    # malformed state JSON, or a lost per-raw entry cannot trigger blind replay.
    for _packet_path, packet, packet_raws in _semantic_defer_packet_records(
        verify_sources=True
    ):
        stored_authority_epoch = packet.get(
            "authority_epoch",
            packet["authority_artifact_sha256"],
        )
        if not authority_epoch_loaded:
            current_authority_epoch = _current_adopted_authority_epoch()
            authority_epoch_loaded = True
        if (
            current_authority_epoch is None
            or current_authority_epoch == stored_authority_epoch
        ):
            for raw_file in packet_raws:
                deferred[raw_file] = SEMANTIC_NO_QUORUM_DEFER_REASON
        # A different executable authority epoch is the automatic release
        # condition. Superseded and explicitly released packets never reach
        # this loop because their status is not local_quarantined.
    return dict(sorted(deferred.items()))


def record_raw_failure(
    *,
    raw_path: Path,
    error: str | None,
    job_id: str | None = None,
    raw_text: str | None = None,
    threshold: int = FAILURE_THRESHOLD,
    related_raw_paths: Sequence[Path] = (),
) -> SupervisionResult:
    """Record a failed raw and quarantine it after repeated same failures."""

    record = classify_failure(error)
    raw_file = raw_path.name
    if record.failure_class == SEMANTIC_NO_QUORUM_FAILURE_CLASS:
        semantic_defer = record_current_semantic_no_quorum_defer(
            raw_path=raw_path,
            error=record.message,
            job_id=job_id,
            raw_text=raw_text,
            related_raw_paths=related_raw_paths,
        )
        if semantic_defer is not None:
            return semantic_defer
        failure_class = "ingest.runtime_local_consensus_authority_unavailable"
        fingerprint = (
            f"{failure_class}:semantic_defer_authority_changed_before_publication"
        )
        runtime_status.safe_append_event(
            "warn",
            (
                "failure-supervisor | semantic defer authority changed before "
                f"publication for {raw_file}; raw left pending"
            ),
            source="failure-supervisor",
            raw_file=raw_file,
            failure_class=failure_class,
            fingerprint=fingerprint,
            outcome_kind="semantic_defer_authority_changed",
        )
        return SupervisionResult(
            raw_file=raw_file,
            failure_class=failure_class,
            fingerprint=fingerprint,
            attempts=0,
            tracked=False,
            transient=True,
        )
    if record.failure_class in TRANSIENT_FAILURE_CLASSES:
        runtime_status.safe_append_event(
            "warn",
            (
                "failure-supervisor | transient ingest failure for "
                f"{raw_file}; raw left pending"
            ),
            source="failure-supervisor",
            raw_file=raw_file,
            failure_class=record.failure_class,
            fingerprint=record.fingerprint,
        )
        return SupervisionResult(
            raw_file=raw_file,
            failure_class=record.failure_class,
            fingerprint=record.fingerprint,
            attempts=0,
            tracked=False,
            transient=True,
        )

    if record.failure_class in OPERATIONAL_SELF_HEAL_FAILURE_CLASSES:
        with _failure_state_lock():
            result, packet_to_launch = _record_operational_raw_failure(
                raw_path=raw_path,
                record=record,
                job_id=job_id,
                raw_text=raw_text,
                related_raw_paths=related_raw_paths,
            )
        if packet_to_launch is not None:
            launch_error = _launch_self_heal(packet_to_launch)
            _record_operational_launch_result(packet_to_launch, launch_error)
        return result

    with _failure_state_lock():
        state = _load_state()
        failures = state.setdefault("failures", {})
        if not isinstance(failures, dict):
            failures = {}
            state["failures"] = failures

        current = failures.get(raw_file)
        if (
            not isinstance(current, dict)
            or current.get("fingerprint") != record.fingerprint
        ):
            current = {
                "fingerprint": record.fingerprint,
                "failure_class": record.failure_class,
                "attempts": 0,
                "first_seen_at": datetime.now().isoformat(),
                "last_error": record.message,
            }
            source_evidence = _raw_source_evidence(raw_path)
            if source_evidence is not None:
                current["raw_sha256"] = source_evidence["sha256"]
                current["raw_bytes"] = source_evidence["bytes"]

        current["attempts"] = int(current.get("attempts", 0)) + 1
        current["last_seen_at"] = datetime.now().isoformat()
        current["last_error"] = record.message
        current["job_id"] = job_id
        failures[raw_file] = current
        _save_state(state)

        attempts = int(current["attempts"])
        effective_threshold = (
            1
            if record.failure_class in IMMEDIATE_SELF_HEAL_FAILURE_CLASSES
            else max(1, threshold)
        )
        if attempts < effective_threshold:
            return SupervisionResult(
                raw_file=raw_file,
                failure_class=record.failure_class,
                fingerprint=record.fingerprint,
                attempts=attempts,
            )

        packet_path = _write_packet(
            raw_file=raw_file,
            record=record,
            attempts=attempts,
            job_id=job_id,
            raw_text=raw_text,
        )
        quarantine_path = _quarantine_raw(raw_path, packet_path)

        runtime_status.safe_append_event(
            "warn",
            (
                "failure-supervisor | quarantined "
                f"{raw_file} after {attempts} repeated failures"
            ),
            source="failure-supervisor",
            raw_file=raw_file,
            failure_class=record.failure_class,
            fingerprint=record.fingerprint,
            packet_path=str(packet_path),
            quarantine_path=str(quarantine_path) if quarantine_path else None,
        )
        result = SupervisionResult(
            raw_file=raw_file,
            failure_class=record.failure_class,
            fingerprint=record.fingerprint,
            attempts=attempts,
            quarantined=True,
            packet_path=str(packet_path),
            quarantine_path=str(quarantine_path) if quarantine_path else None,
        )

    _launch_self_heal(packet_path)
    return result


def result_to_dict(result: SupervisionResult) -> dict[str, Any]:
    return asdict(result)
