"""Failure supervision for self-healing ingest runs.

This module is intentionally deterministic.  LLMs may diagnose a packet later,
but the control loop here decides when to stop retrying a raw, how to fingerprint
the failure, and where to persist the evidence for local/frontier repair.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from llm_wiki_mcp import runtime_status, wiki


FAILURE_THRESHOLD = 3


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
}


def _runtime_failures_dir() -> Path:
    return wiki.WIKI_ROOT / "runtime" / "failures"


def _state_file() -> Path:
    return _runtime_failures_dir() / "state.json"


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
    path.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n")


def reset_raw_failure(raw_file: str) -> None:
    """Forget tracked failures for a raw after it succeeds."""

    state = _load_state()
    failures = state.get("failures", {})
    if isinstance(failures, dict) and raw_file in failures:
        failures.pop(raw_file, None)
        _save_state(state)


def classify_failure(message: str | None) -> FailureRecord:
    """Return a stable failure class and fingerprint for a job error."""

    msg = (message or "unknown failure").strip() or "unknown failure"

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
) -> Path:
    now = datetime.now().isoformat()
    failure_id = (
        datetime.now().strftime("%Y%m%d-%H%M%S")
        + "-"
        + _safe_filename(record.fingerprint)
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
        "status": "pending_local_repair",
        "local_model": "qwen",
        "frontier_status": "not_requested",
        "raw_preview": (raw_text or "")[:4000],
    }
    packets_dir = _runtime_failures_dir() / "packets"
    packets_dir.mkdir(parents=True, exist_ok=True)
    path = packets_dir / f"{failure_id}.json"
    path.write_text(json.dumps(packet, indent=2, ensure_ascii=False) + "\n")
    return path


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
    pointer.write_text(str(packet_path) + "\n")
    return target


def record_raw_failure(
    *,
    raw_path: Path,
    error: str | None,
    job_id: str | None = None,
    raw_text: str | None = None,
    threshold: int = FAILURE_THRESHOLD,
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
    if attempts < threshold:
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

    try:
        from llm_wiki_mcp.self_heal import start_background

        start_background(packet_path)
    except Exception as exc:
        runtime_status.safe_append_event(
            "warn",
            f"failure-supervisor | self-heal launch failed: {exc}",
            source="failure-supervisor",
            packet_path=str(packet_path),
        )

    return SupervisionResult(
        raw_file=raw_file,
        failure_class=record.failure_class,
        fingerprint=record.fingerprint,
        attempts=attempts,
        quarantined=True,
        packet_path=str(packet_path),
        quarantine_path=str(quarantine_path) if quarantine_path else None,
    )


def result_to_dict(result: SupervisionResult) -> dict[str, Any]:
    return asdict(result)
