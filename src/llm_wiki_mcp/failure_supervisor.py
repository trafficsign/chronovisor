"""Failure supervision for self-healing ingest runs.

This module is intentionally deterministic.  LLMs may diagnose a packet later,
but the control loop here decides when to stop retrying a raw, how to fingerprint
the failure, and where to persist the evidence for local/frontier repair.
"""

from __future__ import annotations

import hashlib
import fcntl
import json
import re
import shutil
import threading
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

from llm_wiki_mcp import runtime_status, wiki
from llm_wiki_mcp.link_fix import atomic_write


FAILURE_THRESHOLD = 3
_FAILURE_STATE_THREAD_LOCK = threading.RLock()


@dataclass(frozen=True)
class FailureRecord:
    """Normalized failure information used by the supervisor."""

    failure_class: str
    fingerprint: str
    message: str
    requested_page_id: str | None = None


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


TRANSIENT_FAILURE_CLASSES = {
    "ingest.ollama_unavailable",
    "ingest.runtime_transport_error",
    "ingest.runtime_transport_timeout",
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
    "ingest.generation_transport_error",
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

# These failures already exhausted a bounded convergence loop inside one
# ingest job. Replaying the raw through three more jobs only burns local and
# frontier tokens while reproducing the same control-path defect.
IMMEDIATE_SELF_HEAL_FAILURE_CLASSES = {
    "ingest.frontier_nonconvergent",
    "ingest.local_consensus_nonconvergent",
}


def _runtime_failures_dir() -> Path:
    return wiki.WIKI_ROOT / "runtime" / "failures"


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
    except json.JSONDecodeError:
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


def reset_raw_failure(raw_file: str) -> None:
    """Forget tracked failures for a raw after it succeeds."""

    with _failure_state_lock():
        state = _load_state()
        failures = state.get("failures", {})
        if isinstance(failures, dict) and raw_file in failures:
            removed = failures.pop(raw_file, None)
            if isinstance(removed, dict):
                fingerprint = removed.get("fingerprint")
                operational_failures = state.get("operational_failures")
                if (
                    isinstance(fingerprint, str)
                    and isinstance(operational_failures, dict)
                    and not any(
                        isinstance(entry, dict)
                        and entry.get("fingerprint") == fingerprint
                        for entry in failures.values()
                    )
                ):
                    operational_failures.pop(fingerprint, None)
            _save_state(state)


def classify_failure(message: str | None) -> FailureRecord:
    """Return a stable failure class and fingerprint for a job error."""

    msg = (message or "unknown failure").strip() or "unknown failure"

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
        r"triage structured failure \[([^\]]+)\]:\s*(.*)",
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
        failure_class = "ingest.runtime_local_consensus_authority_unavailable"
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
    for path in wiki.PAGES_DIR.rglob("*.md"):
        stem = path.stem
        if loose_key(stem) == target:
            try:
                matches.append(str(path.relative_to(wiki.PAGES_DIR).with_suffix("")))
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
) -> Path:
    created_at = datetime.now()
    now = created_at.isoformat()
    source_suffix = hashlib.sha256(
        f"{raw_file}\0{record.fingerprint}".encode("utf-8")
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
        from llm_wiki_mcp.self_heal import start_background

        start_background(packet_path)
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
            failures[source_file] = {
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


def _valid_projection_child_bundle(raw_path: Path, entry: dict[str, Any]) -> bool:
    """Return true only when a previously failed child now has a valid bundle."""

    if (
        entry.get("failure_class")
        not in SEMANTIC_PROJECTION_OPERATIONAL_FAILURE_CLASSES
    ):
        return False
    match = re.fullmatch(
        r"semantic-([0-9a-f]{64})-child-[0-9]{8}-[0-9a-f]{64}\.md",
        raw_path.name,
    )
    if match is None or not raw_path.is_file():
        return False
    manifest_path = raw_path.parent / f"semantic-{match.group(1)}.manifest.json"
    try:
        from llm_wiki_mcp.raw_semantic_projection import verify_projection_bundle

        manifest = verify_projection_bundle(manifest_path)
    except (OSError, ValueError, TypeError):
        return False
    children = manifest.get("children")
    return isinstance(children, list) and any(
        isinstance(row, dict) and row.get("filename") == raw_path.name
        for row in children
    )


def _projection_parent_can_retry(raw_path: Path, entry: dict[str, Any]) -> bool:
    """Allow only completed or intent-first incomplete parent bundles to retry."""

    if (
        entry.get("failure_class")
        not in SEMANTIC_PROJECTION_OPERATIONAL_FAILURE_CLASSES
        or not raw_path.is_file()
    ):
        return False
    try:
        from llm_wiki_mcp.raw_semantic_projection import (
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
        return {}
    available_paths = {
        path.name: path
        for path in (
            raw_paths if raw_paths is not None else sorted(wiki.RAW_DIR.glob("*.md"))
        )
    }
    deferred: dict[str, str] = {}

    for raw_file, value in list(failures.items()):
        if not isinstance(raw_file, str) or not isinstance(value, dict):
            continue
        if (
            value.get("failure_class") not in OPERATIONAL_SELF_HEAL_FAILURE_CLASSES
            and value.get("self_heal_queued") is not True
        ):
            continue
        raw_path = available_paths.get(raw_file, wiki.RAW_DIR / raw_file)
        if _operational_entry_is_released(raw_path, value):
            continue
        if _projection_parent_can_retry(raw_path, value):
            continue
        # Unknown statuses are deliberately fail closed too. A future repair
        # state must be explicitly added to the success allowlist before it
        # can re-enable inference.
        deferred[raw_file] = _packet_status(value)
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
