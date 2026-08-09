"""Promote repeated recall auto-apply errors into self-heal packets."""
from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from chronovisor.core import runtime_status
from chronovisor.core import store as chronovisor_store

AUTO_APPLY_ERROR_THRESHOLD = 3
MAX_CLUSTER_SAMPLES = 8


def _failures_dir() -> Path:
    return chronovisor_store.CHRONOVISOR_ROOT / "runtime" / "failures"


def _packet_dir() -> Path:
    return _failures_dir() / "packets"


def _state_file() -> Path:
    return _failures_dir() / "auto-apply-error-state.json"


def _auto_apply_log_file(path: Path | None = None) -> Path:
    return path or (chronovisor_store.CHRONOVISOR_ROOT / "recall" / "auto-apply.jsonl")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").split("\n")
    except OSError:
        return []
    records: list[dict[str, Any]] = []
    for line in lines:
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            records.append(parsed)
    return records


def _load_state() -> dict[str, Any]:
    path = _state_file()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"clusters": {}}
    if not isinstance(data, dict):
        return {"clusters": {}}
    if not isinstance(data.get("clusters"), dict):
        data["clusters"] = {}
    return data


def _save_state(state: dict[str, Any]) -> None:
    path = _state_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def _safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-._")
    if not cleaned:
        digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
        cleaned = f"auto-apply-error-{digest}"
    return cleaned[:160]


def _error_text(record: dict[str, Any]) -> str:
    result = record.get("result")
    if isinstance(result, dict):
        error = result.get("error")
        if isinstance(error, str):
            return error
    return str(result or record.get("status") or "unknown auto-apply error")


def classify_auto_apply_error(record: dict[str, Any]) -> str:
    """Return a stable kind for a recall auto-apply error."""

    action = str(record.get("action_type") or "unknown")
    error = _error_text(record).casefold()
    if "invalid page tag" in error and "missing required prefix" in error:
        kind = "invalid_page_tag.missing_required_prefix"
    elif "invalid page tag" in error:
        kind = "invalid_page_tag"
    elif "invalid alias page_id" in error:
        kind = "invalid_alias_page_id"
    elif "query hint page_id is required" in error:
        kind = "query_hint.missing_page_id"
    elif "query hint target page does not exist" in error:
        kind = "query_hint.target_missing"
    elif "page does not exist" in error:
        kind = "target_page_missing"
    else:
        digest = hashlib.sha256(error.encode("utf-8")).hexdigest()[:12]
        kind = f"unknown.{digest}"
    return f"{action}:{kind}"


def fingerprint_for(record: dict[str, Any]) -> str:
    return f"recall.auto_apply_error:{classify_auto_apply_error(record)}"


def _sample(record: dict[str, Any]) -> dict[str, Any]:
    result = record.get("result")
    return {
        "ts": record.get("ts"),
        "apply_key": record.get("apply_key"),
        "normalize_key": record.get("normalize_key"),
        "action_type": record.get("action_type"),
        "source_ref": record.get("source_ref"),
        "status": record.get("status"),
        "error": _error_text(record),
        "result": result if isinstance(result, dict) else None,
    }


def _cluster_records(error_records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    clusters: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in error_records:
        clusters[fingerprint_for(record)].append(record)
    return dict(clusters)


def _cluster_summary(fingerprint: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    first = records[0]
    kind = fingerprint.removeprefix("recall.auto_apply_error:")
    return {
        "fingerprint": fingerprint,
        "failure_class": "recall.auto_apply_error",
        "error_kind": kind,
        "action_type": first.get("action_type"),
        "count": len(records),
        "first_seen_at": first.get("ts"),
        "last_seen_at": records[-1].get("ts"),
        "samples": [_sample(record) for record in records[-MAX_CLUSTER_SAMPLES:]],
        "suggested_scope": [
            "src/chronovisor/recall_auto_apply.py",
            "src/chronovisor/recall_auditor.py",
            "tests/test_recall_auto_apply.py",
        ],
        "remediation_goal": (
            "Diagnose why recall auto-apply keeps emitting the same error. "
            "Use local Qwen only for diagnosis; code changes must be approved by the frontier reviewer. "
            "Prefer a narrow, regression-tested fix that turns repeated unsafe auto actions into safe, "
            "observable fallback or skip behavior."
        ),
    }


def _write_packet(
    fingerprint: str,
    records: list[dict[str, Any]],
    *,
    count: int | None = None,
    first_seen_at: object | None = None,
) -> Path:
    now = datetime.now().isoformat()
    failure_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + _safe_filename(fingerprint)
    summary = _cluster_summary(fingerprint, records)
    packet_count = count if count is not None else len(records)
    if count is not None:
        summary["count"] = count
        summary["observed_count"] = len(records)
    if first_seen_at is not None:
        summary["first_seen_at"] = first_seen_at
    packet = {
        "failure_id": failure_id,
        "created_at": now,
        "raw_file": None,
        "job_id": None,
        "failure_class": "recall.auto_apply_error",
        "fingerprint": fingerprint,
        "attempts": packet_count,
        "error": f"{packet_count} repeated recall auto-apply errors: {summary['error_kind']}",
        "requested_page_id": None,
        "similar_existing_pages": [],
        "status": "pending_local_repair",
        "local_model": "qwen",
        "frontier_status": "not_requested",
        "auto_apply_error": summary,
        "raw_preview": json.dumps(summary["samples"], ensure_ascii=False, indent=2)[:4000],
    }
    path = _packet_dir() / f"{failure_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(packet, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    return path


def _packet_still_exists(path_value: object) -> bool:
    if not isinstance(path_value, str) or not path_value:
        return False
    return Path(path_value).exists()


def supervise_error_records(
    error_records: list[dict[str, Any]],
    *,
    threshold: int = AUTO_APPLY_ERROR_THRESHOLD,
    start_background: bool = True,
    dry_run: bool = False,
    accumulate: bool = True,
) -> dict[str, Any]:
    """Create self-heal packets for repeated auto-apply error clusters."""

    threshold = max(1, threshold)
    state = _load_state()
    state_clusters = state.setdefault("clusters", {})
    if not isinstance(state_clusters, dict):
        state_clusters = {}
        state["clusters"] = state_clusters

    packet_paths: list[Path] = []
    clusters_out: list[dict[str, Any]] = []
    for fingerprint, records in sorted(_cluster_records(error_records).items()):
        existing = state_clusters.get(fingerprint)
        existing_packet = existing.get("packet_path") if isinstance(existing, dict) else None
        prior_count = int(existing.get("count", 0) or 0) if isinstance(existing, dict) else 0
        total_count = prior_count + len(records) if accumulate else max(prior_count, len(records))
        summary = _cluster_summary(fingerprint, records)
        first_seen_at = (
            existing.get("first_seen_at")
            if isinstance(existing, dict) and existing.get("first_seen_at")
            else summary.get("first_seen_at")
        )
        summary["count"] = total_count
        summary["observed_count"] = len(records)
        summary["first_seen_at"] = first_seen_at
        summary["threshold"] = threshold
        summary["accumulate"] = accumulate
        summary["packet_created"] = False
        summary["would_create_packet"] = False
        if total_count >= threshold and not _packet_still_exists(existing_packet):
            summary["would_create_packet"] = True
            if not dry_run:
                packet_path = _write_packet(
                    fingerprint,
                    records,
                    count=total_count,
                    first_seen_at=first_seen_at,
                )
                packet_paths.append(packet_path)
                summary["packet_created"] = True
                summary["packet_path"] = str(packet_path)
                state_clusters[fingerprint] = {
                    "count": total_count,
                    "first_seen_at": first_seen_at,
                    "last_seen_at": summary.get("last_seen_at"),
                    "packet_path": str(packet_path),
                    "updated_at": datetime.now().isoformat(),
                }
                runtime_status.safe_append_event(
                    "warn",
                    f"auto-apply-supervisor | queued self-heal for {summary['error_kind']}",
                    source="auto-apply-supervisor",
                    fingerprint=fingerprint,
                    count=total_count,
                    observed_count=len(records),
                    packet_path=str(packet_path),
                )
                if start_background:
                    try:
                        from chronovisor.ingest.self_heal import (
                            start_background as start_self_heal,
                        )

                        start_self_heal(packet_path)
                    except Exception as exc:
                        runtime_status.safe_append_event(
                            "warn",
                            f"auto-apply-supervisor | self-heal start failed: {exc}",
                            source="auto-apply-supervisor",
                            fingerprint=fingerprint,
                        )
        else:
            if not dry_run:
                next_state = existing if isinstance(existing, dict) else {}
                next_state["count"] = total_count
                next_state["first_seen_at"] = first_seen_at
                next_state["last_seen_at"] = summary.get("last_seen_at")
                next_state["updated_at"] = datetime.now().isoformat()
                if _packet_still_exists(existing_packet):
                    next_state["packet_path"] = existing_packet
                    summary["packet_path"] = existing_packet
                state_clusters[fingerprint] = next_state
        clusters_out.append(summary)

    if not dry_run:
        _save_state(state)
    return {
        "status": "ok",
        "dry_run": dry_run,
        "accumulate": accumulate,
        "clusters": clusters_out,
        "packets_created": [str(path) for path in packet_paths],
    }


def supervise_auto_apply_log(
    *,
    log_file: Path | None = None,
    threshold: int = AUTO_APPLY_ERROR_THRESHOLD,
    start_background: bool = True,
    dry_run: bool = False,
) -> dict[str, Any]:
    records = _read_jsonl(_auto_apply_log_file(log_file))
    errors = [record for record in records if record.get("status") == "error"]
    result = supervise_error_records(
        errors,
        threshold=threshold,
        start_background=start_background,
        dry_run=dry_run,
        accumulate=False,
    )
    result["errors_seen"] = len(errors)
    result["log_file"] = str(_auto_apply_log_file(log_file))
    return result


def pending_auto_apply_error_packets() -> list[Path]:
    if not _packet_dir().exists():
        return []
    out: list[Path] = []
    for path in sorted(_packet_dir().glob("*.json")):
        try:
            packet = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if (
            packet.get("failure_class") == "recall.auto_apply_error"
            and packet.get("status") in {"pending_local_repair", "local_repair_failed", "pending_frontier", "frontier_retry"}
        ):
            out.append(path)
    return out
