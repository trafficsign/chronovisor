"""Knowledge health KPIs for dashboard and CLI."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from llm_wiki_mcp.decision_authority import semantic_authority_shape_error
from llm_wiki_mcp.index_store import get_store
from llm_wiki_mcp.jsonl import count_jsonl, read_jsonl
from llm_wiki_mcp.semantic_hold import (
    LOCAL_SEMANTIC_NO_QUORUM,
    canonical_sha256,
    persisted_semantic_no_quorum_hold,
)
from llm_wiki_mcp.wiki import RAW_DIR, WIKI_ROOT


def _jsonl_count(path: Path) -> int:
    return count_jsonl(path)


def _read_jsonl(path: Path, *, limit: int = 500) -> list[dict[str, Any]]:
    return read_jsonl(path, limit=limit)


def summary_coverage() -> dict[str, Any]:
    store = get_store()
    store.refresh()
    metas = store.all_pages_meta(include_system=False)
    knowledge = [m for m in metas if m.get("page_type") != "reference"]
    with_summary = 0
    with_questions = 0
    typed: dict[str, int] = {}
    sensitivity: dict[str, int] = {}
    for meta in knowledge:
        typed[str(meta.get("page_type") or "knowledge")] = (
            typed.get(str(meta.get("page_type") or "knowledge"), 0) + 1
        )
        tier = str(meta.get("sensitivity") or "normal")
        sensitivity[tier] = sensitivity.get(tier, 0) + 1
        full = store.meta(str(meta.get("page_id", ""))) or {}
        if isinstance(full.get("summary"), str) and full["summary"].strip():
            with_summary += 1
        questions = full.get("recall_questions")
        if isinstance(questions, list) and questions:
            with_questions += 1
    total = len(knowledge)
    return {
        "knowledge_pages": total,
        "summary_pages": with_summary,
        "recall_question_pages": with_questions,
        "summary_coverage": (with_summary / total) if total else 0.0,
        "recall_question_coverage": (with_questions / total) if total else 0.0,
        "page_types": typed,
        "sensitivity": sensitivity,
    }


def _raw_host(path: Path) -> str:
    name = path.name.lower()
    if "codex" in name:
        return "codex"
    if "claude" in name:
        return "claude"
    if "cowork" in name:
        return "cowork"
    return "unknown"


def capture_kpi() -> dict[str, Any]:
    raws = sorted(RAW_DIR.glob("*.md")) if RAW_DIR.exists() else []
    by_host: dict[str, int] = {}
    for path in raws:
        host = _raw_host(path)
        by_host[host] = by_host.get(host, 0) + 1

    claims = _read_jsonl(WIKI_ROOT / "claims" / "claims.jsonl", limit=100000)
    claimed_raws = {
        str(row.get("source_raw"))
        for row in claims
        if isinstance(row.get("source_raw"), str) and row.get("source_raw")
    }
    raw_names = {path.name for path in raws}
    covered = raw_names & claimed_raws
    return {
        "raw_files": len(raws),
        "claimed_raw_files": len(covered),
        "claim_coverage": (len(covered) / len(raws)) if raws else None,
        "raw_by_host": by_host,
        "claim_rows": len(claims),
    }


def latest_memory_integrity() -> dict[str, Any]:
    path = WIKI_ROOT / "eval" / "memory-integrity-latest.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"status": "missing", "path": str(path)}
    if not isinstance(payload, dict):
        return {"status": "invalid", "path": str(path)}
    return {
        "status": payload.get("status", "ok"),
        "path": str(path),
        "generated_at": payload.get("generated_at"),
        "total": payload.get("total", 0),
        "passed": payload.get("passed", 0),
        "missed": payload.get("missed", 0),
        "capture_rate": payload.get("capture_rate"),
        "by_host": payload.get("by_host", {}),
    }


def cofire_kpi() -> dict[str, Any]:
    path = WIKI_ROOT / "recall" / "cofire.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"status": "missing", "path": str(path), "nodes": 0, "edges": 0}
    if not isinstance(payload, dict):
        return {"status": "invalid", "path": str(path), "nodes": 0, "edges": 0}
    return {
        "status": payload.get("status", "ok"),
        "path": str(path),
        "generated_at": payload.get("generated_at"),
        "episodes": payload.get("episodes", 0),
        "nodes": payload.get("nodes", 0),
        "edges": payload.get("edges", 0),
    }


def prefetch_kpi() -> dict[str, Any]:
    path = WIKI_ROOT / "recall" / "prefetch.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"status": "missing", "path": str(path), "buckets": 0, "tokens": 0}
    if not isinstance(payload, dict):
        return {"status": "invalid", "path": str(path), "buckets": 0, "tokens": 0}
    buckets = payload.get("buckets")
    tokens = payload.get("tokens")
    return {
        "status": payload.get("status", "ok"),
        "path": str(path),
        "generated_at": payload.get("generated_at"),
        "episodes": payload.get("episodes", 0),
        "buckets": len(buckets) if isinstance(buckets, dict) else 0,
        "tokens": len(tokens) if isinstance(tokens, dict) else 0,
    }


def derived_memory_kpi() -> dict[str, Any]:
    retention_path = WIKI_ROOT / "recall" / "retention.json"
    try:
        retention = json.loads(retention_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        retention = {}
    retention_counts = retention.get("counts") if isinstance(retention, dict) else {}
    return {
        "claims": _jsonl_count(WIKI_ROOT / "claims" / "claims-index.jsonl"),
        "golden": _jsonl_count(WIKI_ROOT / "recall" / "search-golden.jsonl"),
        "distill_rows": _jsonl_count(WIKI_ROOT / "distill" / "wiki-qa.jsonl"),
        "retention_pages": int(retention_counts.get("pages") or 0)
        if isinstance(retention_counts, dict)
        else 0,
        "archive_candidates": int(retention_counts.get("archive_candidates") or 0)
        if isinstance(retention_counts, dict)
        else 0,
        "hubs": len(list((WIKI_ROOT / "pages" / "hubs").glob("*.md")))
        if (WIKI_ROOT / "pages" / "hubs").exists()
        else 0,
    }


def read_back_kpi() -> dict[str, Any]:
    run_path = WIKI_ROOT / "runtime" / "ingest-read-back-runs.jsonl"
    rows = _read_jsonl(run_path, limit=200)
    cohort = "all_ingest_runs"
    if not rows:
        rows = _read_jsonl(
            WIKI_ROOT / "runtime" / "ingest-read-back-failures.jsonl", limit=200
        )
        cohort = "legacy_failure_bearing_runs_only"
    failures = 0
    checked = 0
    latest: dict[str, Any] | None = None
    for row in rows:
        checked += int(row.get("checked") or 0)
        failed_rows = row.get("failed")
        if isinstance(failed_rows, list):
            failures += len(failed_rows)
        latest = row
    passed = max(0, checked - failures)
    ledger_path = WIKI_ROOT / "runtime" / "ingest-read-back-repair.json"
    try:
        from llm_wiki_mcp.durable_state import canonical_bytes
        from llm_wiki_mcp.read_back_integrity import verify_prior_prefix

        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        observed = ledger.get("view_sha256") if isinstance(ledger, dict) else None
        unsigned = (
            {key: value for key, value in ledger.items() if key != "view_sha256"}
            if isinstance(ledger, dict)
            else {}
        )
        expected = hashlib.sha256(canonical_bytes(unsigned)).hexdigest()
        cursor = ledger.get("source_cursor") if isinstance(ledger, dict) else None
        source_path = WIKI_ROOT / "runtime" / "ingest-read-back-failures.jsonl"
        ledger_integrity = {
            "status": (
                "ok"
                if observed == expected and verify_prior_prefix(source_path, cursor)
                else "invalid"
            ),
            "schema_version": ledger.get("schema_version")
            if isinstance(ledger, dict)
            else None,
            "source_cursor": cursor,
        }
    except FileNotFoundError:
        ledger_integrity = {"status": "missing"}
    except Exception as exc:
        ledger_integrity = {"status": "invalid", "error": str(exc)}
    return {
        "checked": checked,
        "passed": passed,
        "failures": failures,
        "pass_rate": (passed / checked) if checked else None,
        "latest": latest,
        "cohort": cohort,
        "derived_view_integrity": ledger_integrity,
    }


def autonomy_hardening_kpi() -> dict[str, Any]:
    from llm_wiki_mcp.deadman import inspect_heartbeat
    from llm_wiki_mcp.durable_state import DurableStateError, read_sealed_json
    from llm_wiki_mcp.managed_hold import ManagedHoldStore
    from llm_wiki_mcp.provisional_recall import snapshot as provisional_snapshot
    from llm_wiki_mcp.quality_guard import quality_snapshot

    runtime = WIKI_ROOT / "runtime"
    autonomy = WIKI_ROOT / "autonomy"
    managed = ManagedHoldStore(runtime / "managed-holds" / "state.json")
    try:
        managed_snapshot = managed.snapshot()
    except Exception as exc:
        managed_snapshot = {"status": "invalid", "error": str(exc)}
    try:
        provisional = provisional_snapshot(wiki_root=WIKI_ROOT)
    except Exception as exc:
        provisional = {"status": "invalid", "error": str(exc)}
    try:
        quality = quality_snapshot(runtime / "quality")
    except Exception as exc:
        quality = {"status": "invalid", "error": str(exc)}
    try:
        deadman_threshold = read_sealed_json(autonomy / "observer-threshold-state.json")
    except DurableStateError:
        deadman_threshold = {"status": "unavailable"}
    artifacts = runtime / "decision-artifacts"
    try:
        artifact_count = sum(1 for _path in artifacts.glob("[0-9a-f][0-9a-f]/*.json"))
    except OSError:
        artifact_count = 0
    return {
        "decision_artifacts": {
            "count": artifact_count,
            "replay_definition": "sealed_execution_fingerprint",
        },
        "deadman": {
            "main": inspect_heartbeat(
                autonomy / "watchdog-heartbeat.json",
                expected_role="main_watchdog",
                max_age_seconds=20 * 60,
            ),
            "observer": inspect_heartbeat(
                autonomy / "observer-heartbeat.json",
                expected_role="independent_observer",
                max_age_seconds=10 * 60,
            ),
            "threshold": deadman_threshold,
        },
        "quality": quality,
        "managed_holds": managed_snapshot,
        "provisional_recall": provisional,
        "frontier_semantic_audit_allowed": False,
    }


def recall_feedback_kpi() -> dict[str, Any]:
    rows = _read_jsonl(WIKI_ROOT / "recall" / "feedback.jsonl", limit=1000)
    counts: dict[str, int] = {}
    for row in rows:
        kind = str(row.get("kind") or "unknown")
        counts[kind] = counts.get(kind, 0) + 1
    used = counts.get("injection_used", 0)
    ignored = counts.get("injection_ignored", 0) + counts.get("false-positive", 0)
    denominator = used + ignored
    return {
        "samples": len(rows),
        "counts": counts,
        "precision_proxy": (used / denominator) if denominator else None,
        "missed_candidates": counts.get("missed_candidate", 0)
        + counts.get("missed", 0),
    }


_CONTENT_CORRECTION_SEMANTIC_HOLD_KIND = "content_correction_semantic_no_quorum"
_CONTENT_CORRECTION_DECISION_LANES = frozenset(
    {"content_correction_classification", "content_correction_review"}
)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _is_terminal_semantic_defer(item: dict[str, Any]) -> bool:
    """Separate a safe semantic non-decision from an operational quarantine."""

    common_hold = persisted_semantic_no_quorum_hold(item)
    if common_hold is not None:
        decision_lane = str(common_hold.get("lane") or "")
        result = item.get("result")
        return bool(
            str(item.get("status") or "") == "quarantined"
            and str(item.get("last_failure_class") or "") == LOCAL_SEMANTIC_NO_QUORUM
            and isinstance(result, dict)
            and result.get("terminal_reason") == "semantic_no_quorum"
            and item.get("quarantine_reason") == f"semantic_no_quorum:{decision_lane}"
        )

    result = item.get("result")
    legacy_hold = (
        result.get("legacy_semantic_hold") if isinstance(result, dict) else None
    )
    if isinstance(legacy_hold, dict):
        expected_fields = {
            "schema_version",
            "kind",
            "lane",
            "epoch",
            "epoch_sha256",
            "authority",
            "authority_sha256",
            "migrated_from",
            "hold_sha256",
        }
        unsigned = dict(legacy_hold)
        unsigned.pop("hold_sha256", None)
        try:
            legacy_valid = bool(
                set(legacy_hold) == expected_fields
                and legacy_hold.get("schema_version") == 1
                and legacy_hold.get("kind")
                == "legacy_local_semantic_no_quorum_fail_closed"
                and isinstance(legacy_hold.get("lane"), str)
                and isinstance(legacy_hold.get("epoch"), dict)
                and isinstance(legacy_hold.get("authority"), dict)
                and isinstance(legacy_hold.get("migrated_from"), dict)
                and semantic_authority_shape_error(
                    legacy_hold["authority"],
                    lane=str(legacy_hold.get("lane") or ""),
                )
                is None
                and legacy_hold.get("epoch_sha256")
                == canonical_sha256(legacy_hold["epoch"])
                and legacy_hold.get("authority_sha256")
                == canonical_sha256(legacy_hold["authority"])
                and legacy_hold.get("hold_sha256") == canonical_sha256(unsigned)
            )
        except (TypeError, ValueError):
            legacy_valid = False
        if legacy_valid:
            decision_lane = str(legacy_hold["lane"])
            return bool(
                item.get("status") == "quarantined"
                and item.get("last_failure_class") == LOCAL_SEMANTIC_NO_QUORUM
                and item.get("quarantine_reason")
                == f"semantic_no_quorum_legacy:{decision_lane}"
                and result.get("terminal_reason") == "semantic_no_quorum"
            )

    if (
        str(item.get("status") or "") != "quarantined"
        or str(item.get("lane") or "") != "content_correction"
        or str(item.get("last_failure_class") or "") != "local_semantic_no_quorum"
        or not str(item.get("quarantine_reason") or "").startswith(
            "semantic_no_quorum:"
        )
    ):
        return False
    result = item.get("result")
    if not isinstance(result, dict):
        return False
    semantic_hold = result.get("semantic_hold")
    if (
        result.get("terminal_reason") != "semantic_no_quorum"
        or not isinstance(semantic_hold, dict)
        or semantic_hold.get("kind") != _CONTENT_CORRECTION_SEMANTIC_HOLD_KIND
    ):
        return False
    decision_lane = semantic_hold.get("decision_lane")
    evidence_hashes = semantic_hold.get("page_evidence_hashes")
    return (
        decision_lane in _CONTENT_CORRECTION_DECISION_LANES
        and item.get("quarantine_reason") == f"semantic_no_quorum:{decision_lane}"
        and semantic_hold.get("input_hash") == item.get("input_hash")
        and _is_sha256(semantic_hold.get("input_hash"))
        and _is_sha256(semantic_hold.get("proposal_sha256"))
        and isinstance(evidence_hashes, dict)
        and all(
            isinstance(page_id, str) and _is_sha256(page_hash)
            for page_id, page_hash in evidence_hashes.items()
        )
        and isinstance(semantic_hold.get("authority"), dict)
        and bool(semantic_hold["authority"])
    )


def convergence_kpi() -> dict[str, Any]:
    path = WIKI_ROOT / "runtime" / "convergence" / "state.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {
            "status": "missing",
            "path": str(path),
            "items": 0,
            "by_status": {},
            "by_lane": {},
        }
    items = payload.get("items") if isinstance(payload, dict) else None
    if not isinstance(items, dict):
        return {
            "status": "invalid",
            "path": str(path),
            "items": 0,
            "by_status": {},
            "by_lane": {},
        }
    by_status: dict[str, int] = {}
    by_lane: dict[str, dict[str, int]] = {}
    now = datetime.now(timezone.utc)
    actionable_dates: list[datetime] = []
    expired_running = 0
    for item in items.values():
        if not isinstance(item, dict):
            continue
        status = (
            "semantic_deferred"
            if _is_terminal_semantic_defer(item)
            else str(item.get("status") or "unknown")
        )
        lane = str(item.get("lane") or "unknown")
        by_status[status] = by_status.get(status, 0) + 1
        lane_counts = by_lane.setdefault(lane, {})
        lane_counts[status] = lane_counts.get(status, 0) + 1
        if status in {
            "pending_local",
            "local_retry",
            "pending_frontier",
            "frontier_retry",
            "local_running",
            "frontier_running",
        }:
            try:
                actionable_dates.append(
                    datetime.fromisoformat(
                        str(item.get("created_at") or "")
                    ).astimezone(timezone.utc)
                )
            except ValueError:
                pass
        if status in {"local_running", "frontier_running"}:
            try:
                expires = datetime.fromisoformat(
                    str(item.get("lease_expires_at") or "")
                ).astimezone(timezone.utc)
                expired_running += int(expires <= now)
            except ValueError:
                expired_running += 1
    events = _read_jsonl(path.with_name("events.jsonl"), limit=100000)
    cutoff = now - timedelta(hours=24)
    recent = []
    for event in events:
        try:
            ts = datetime.fromisoformat(str(event.get("ts") or "")).astimezone(
                timezone.utc
            )
        except ValueError:
            continue
        if ts >= cutoff:
            recent.append(event)
    arrivals = sum(event.get("event") == "candidate_merged" for event in recent)
    completions = sum(
        event.get("event") in {"completed", "candidate_completed"} for event in recent
    )
    oldest = min(actionable_dates) if actionable_dates else None
    return {
        "status": "ok",
        "path": str(path),
        "items": sum(by_status.values()),
        "by_status": dict(sorted(by_status.items())),
        "by_lane": {
            lane: dict(sorted(counts.items()))
            for lane, counts in sorted(by_lane.items())
        },
        "actionable": sum(
            count
            for status, count in by_status.items()
            if status
            in {
                "pending_local",
                "local_retry",
                "pending_frontier",
                "frontier_retry",
                "local_running",
                "frontier_running",
            }
        ),
        "quarantined": by_status.get("quarantined", 0),
        "semantic_deferred": by_status.get("semantic_deferred", 0),
        "human_required": by_status.get("human_required", 0),
        "expired_running": expired_running,
        "oldest_actionable_at": oldest.isoformat(timespec="seconds")
        if oldest
        else None,
        "oldest_actionable_age_hours": round((now - oldest).total_seconds() / 3600, 2)
        if oldest
        else 0.0,
        "arrivals_24h": arrivals,
        "completions_24h": completions,
        "net_growth_24h": arrivals - completions,
    }


def capture_pipeline_kpi() -> dict[str, Any]:
    from llm_wiki_mcp.background_jobs import snapshot as job_snapshot

    sweeper_path = WIKI_ROOT / "runtime" / "session-sweeper-latest.json"
    try:
        sweeper = json.loads(sweeper_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        sweeper = {"status": "missing", "pending": None, "processed": 0}
    return {"background_jobs": job_snapshot(), "session_sweeper": sweeper}


def ingest_liveness_kpi() -> dict[str, Any]:
    path = WIKI_ROOT / "runtime" / "ingest-liveness.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {
            "status": "unknown",
            "path": str(path),
            "alert": False,
            "pending_raws": None,
        }
    if not isinstance(payload, dict):
        return {"status": "invalid", "path": str(path), "alert": False}
    waiting = payload.get("status") == "waiting_for_ollama"
    pending = int(payload.get("pending_raws") or 0)
    return {
        **payload,
        "status": "alert" if waiting and pending > 0 else "ok",
        "runtime_status": payload.get("status"),
        "alert": waiting and pending > 0,
        "path": str(path),
    }


def _queue_status_counts(path: Path, field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in _read_jsonl(path, limit=100000):
        status = str(row.get(field) or "unknown")
        counts[status] = counts.get(status, 0) + 1
    return dict(sorted(counts.items()))


def health_snapshot() -> dict[str, Any]:
    from llm_wiki_mcp.runtime_config import runtime_identity

    coverage = summary_coverage()
    duplicate_queue = WIKI_ROOT / "review" / "duplicate-candidates.jsonl"
    lint_queue = WIKI_ROOT / "review" / "lint-repair-queue.jsonl"
    golden = WIKI_ROOT / "recall" / "search-golden.jsonl"
    label_queue = WIKI_ROOT / "recall" / "search-label-queue.jsonl"
    raw_replay_queue = WIKI_ROOT / "review" / "raw-replay-queue.jsonl"
    label_statuses = _queue_status_counts(label_queue, "queue_status")
    replay_statuses = _queue_status_counts(raw_replay_queue, "status")
    ingest_liveness = ingest_liveness_kpi()
    return {
        "status": "alert" if ingest_liveness.get("alert") else "ok",
        "runtime": runtime_identity(),
        "coverage": coverage,
        "capture": capture_kpi(),
        "memory_integrity": latest_memory_integrity(),
        "cofire": cofire_kpi(),
        "prefetch": prefetch_kpi(),
        "derived": derived_memory_kpi(),
        "read_back": read_back_kpi(),
        "autonomy_hardening": autonomy_hardening_kpi(),
        "recall_feedback": recall_feedback_kpi(),
        "convergence": convergence_kpi(),
        "capture_pipeline": capture_pipeline_kpi(),
        "ingest_liveness": ingest_liveness,
        "queues": {
            "duplicate_candidates": _jsonl_count(duplicate_queue),
            "lint_repair": _jsonl_count(lint_queue),
            "search_golden": _jsonl_count(golden),
            "search_labels": _jsonl_count(label_queue),
            "search_labels_pending": sum(
                count
                for status, count in label_statuses.items()
                if status
                in {
                    "unknown",
                    "pending_review",
                    "pending_frontier_review",
                    "frontier_retry",
                    "frontier_uncertain",
                }
            ),
            "raw_replay": _jsonl_count(raw_replay_queue),
            "raw_replay_pending": sum(
                count
                for status, count in replay_statuses.items()
                if status in {"unknown", "pending", "failed", "retry", "retry_wait"}
            ),
            "search_label_statuses": label_statuses,
            "raw_replay_statuses": replay_statuses,
        },
        "paths": {
            "duplicate_queue": str(duplicate_queue),
            "lint_queue": str(lint_queue),
            "search_golden": str(golden),
            "search_label_queue": str(label_queue),
            "raw_replay_queue": str(raw_replay_queue),
        },
    }
