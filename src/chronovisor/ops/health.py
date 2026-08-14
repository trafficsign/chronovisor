"""Knowledge health KPIs for dashboard and CLI."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from chronovisor.core.index_store import canonical_document_paths, get_store
from chronovisor.core.jsonl import count_jsonl, read_jsonl
from chronovisor.core.store import CHRONOVISOR_ROOT, RAW_DIR
from chronovisor.decision.decision_authority import semantic_authority_shape_error
from chronovisor.decision.semantic_hold import (
    LOCAL_SEMANTIC_NO_QUORUM,
    canonical_sha256,
    persisted_semantic_no_quorum_hold,
)


def _jsonl_count(path: Path) -> int:
    return count_jsonl(path)


def _read_jsonl(path: Path, *, limit: int = 500) -> list[dict[str, Any]]:
    return read_jsonl(path, limit=limit)


def recall_distillation_kpi() -> dict[str, Any]:
    """Expose the distillation worker's public, privacy-safe status.

    Distillation is an optional background path.  Its absence (including the
    capture-only cold start) must never turn the foreground Recall health red.
    The distillation snapshot owns pointer validation, so this layer does not
    read private ledgers or policy artifacts itself.
    """

    try:
        from chronovisor.recall.recall_distillation_store import snapshot
    except ImportError:
        return {
            "status": "unavailable",
            "worker_status": "unavailable",
            "rollout_percent": 0.0,
            "active_policy_id": None,
            "candidate_policy_id": None,
            "lkg_policy_id": None,
            "alert": False,
        }
    try:
        value = snapshot(CHRONOVISOR_ROOT)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return {
            "status": "unavailable",
            "worker_status": "unavailable",
            "rollout_percent": 0.0,
            "active_policy_id": None,
            "candidate_policy_id": None,
            "lkg_policy_id": None,
            "alert": False,
        }
    if not isinstance(value, dict):
        return {
            "status": "invalid",
            "worker_status": "invalid",
            "rollout_percent": 0.0,
            "active_policy_id": None,
            "candidate_policy_id": None,
            "lkg_policy_id": None,
            "alert": True,
            "error": "invalid_snapshot",
        }
    status = str(value.get("status") or "invalid")
    state = value.get("worker_status", value.get("state"))
    worker_status = str(
        state.get("status") if isinstance(state, dict) else state or status
    )
    rollout_status = str(value.get("rollout_status") or "")
    rollout = value.get("rollout")
    rollout_percent = float(rollout) if isinstance(rollout, (int, float)) else 0.0
    return {
        **value,
        "worker_status": worker_status,
        "rollout_status": rollout_status,
        "rollout_percent": rollout_percent,
        "alert": status in {"stale", "tampered", "invalid", "error"},
    }


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


def semantic_index_kpi() -> dict[str, Any]:
    """Report search-index coverage without loading the model or scanning vectors."""

    from chronovisor.core.runtime_config import load_search_embedding_config

    config = load_search_embedding_config()
    if not config.enabled or config.rollout_mode == "off":
        return {
            "status": "inactive",
            "execution_mode": "disabled" if not config.enabled else "service",
            "rollout_mode": config.rollout_mode,
            "enabled": config.enabled,
        }
    from chronovisor.core.semantic_index import semantic_index_status
    from chronovisor.core.semantic_jobs import job_status

    jobs = job_status()
    dead = int((jobs.get("counts") or {}).get("dead") or 0)
    socket_ready = Path(config.socket).expanduser().is_socket()
    service_file = CHRONOVISOR_ROOT / "runtime" / "semantic-service-status.json"
    try:
        service = json.loads(service_file.read_text(encoding="utf-8"))
        if not isinstance(service, dict):
            service = {}
    except (OSError, json.JSONDecodeError):
        service = {}
    routes = service.get("routes")
    route_identity = (
        routes.get("search.semantic.foreground")
        if isinstance(routes, dict)
        else None
    )
    if not (
        isinstance(route_identity, dict)
        and set(route_identity) == {"role", "provider", "model", "location"}
        and all(isinstance(value, str) and value for value in route_identity.values())
    ):
        route_identity = None
    index = (
        semantic_index_status(expected_route=route_identity)
        if route_identity is not None
        else {
            "status": "invalid",
            "generation_id": "",
            "coverage": 0.0,
            "error": "semantic_failure",
        }
    )
    coverage = float(index.get("coverage") or 0.0)
    service_age = max(
        0.0, datetime.now(UTC).timestamp()
        - float(service.get("observed_at_epoch") or 0.0)
    )
    service_fresh = bool(service) and service_age <= 30
    service_pid = int(service.get("pid") or 0)
    service_process_alive = False
    if service_pid > 0:
        try:
            os.kill(service_pid, 0)
            service_process_alive = True
        except OSError:
            pass
    generation_matches = (
        bool(index.get("generation_id"))
        and service.get("generation_id") == index.get("generation_id")
    )
    ready = (
        index.get("status") == "ok"
        and coverage >= 0.999
        and dead == 0
        and socket_ready
        and service_fresh
        and service_process_alive
        and generation_matches
        and service.get("ready") is True
    )
    return {
        "status": "ok" if ready else "alert",
        "enabled": True,
        "execution_mode": "service",
        "rollout_mode": config.rollout_mode,
        "route": route_identity or {},
        "model": str((route_identity or {}).get("model") or ""),
        "revision": config.revision,
        "socket_ready": socket_ready,
        "service_fresh": service_fresh,
        "service_process_alive": service_process_alive,
        "generation_matches": generation_matches,
        "service_age_seconds": round(service_age, 3) if service else None,
        "service": service,
        "coverage": coverage,
        "index": index,
        "jobs": jobs,
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
    from chronovisor.core.raw_store import RawStore

    raws = tuple(RawStore(RAW_DIR).iter_units()) if RAW_DIR.exists() else ()
    artifact_dir = CHRONOVISOR_ROOT / "runtime" / "raw-projections" / "artifacts"
    artifact_paths = tuple(artifact_dir.glob("*.md")) if artifact_dir.exists() else ()
    by_host: dict[str, int] = {}
    for unit in raws:
        host = _raw_host(Path(unit.raw_id))
        by_host[host] = by_host.get(host, 0) + 1
    for path in artifact_paths:
        host = _raw_host(path)
        by_host[host] = by_host.get(host, 0) + 1

    claims = _read_jsonl(CHRONOVISOR_ROOT / "claims" / "claims.jsonl", limit=100000)
    claimed_raws = {
        str(row.get("source_raw"))
        for row in claims
        if isinstance(row.get("source_raw"), str) and row.get("source_raw")
    }
    raw_names = {unit.raw_id for unit in raws} | {path.name for path in artifact_paths}
    covered = raw_names & claimed_raws
    return {
        "raw_files": len(raws) + len(artifact_paths),
        "claimed_raw_files": len(covered),
        "claim_coverage": (
            len(covered) / (len(raws) + len(artifact_paths))
            if raws or artifact_paths
            else None
        ),
        "raw_by_host": by_host,
        "claim_rows": len(claims),
    }


def latest_memory_integrity() -> dict[str, Any]:
    path = CHRONOVISOR_ROOT / "eval" / "memory-integrity-latest.json"
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
    path = CHRONOVISOR_ROOT / "recall" / "cofire.json"
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
    path = CHRONOVISOR_ROOT / "recall" / "prefetch.json"
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
        "buckets": (
            int(payload.get("bucket_count") or 0)
            if payload.get("storage") == "sqlite"
            else len(buckets) if isinstance(buckets, dict) else 0
        ),
        "tokens": (
            int(payload.get("token_count") or 0)
            if payload.get("storage") == "sqlite"
            else len(tokens) if isinstance(tokens, dict) else 0
        ),
    }


def derived_memory_kpi() -> dict[str, Any]:
    retention_path = CHRONOVISOR_ROOT / "recall" / "retention.json"
    try:
        retention = json.loads(retention_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        retention = {}
    retention_counts = retention.get("counts") if isinstance(retention, dict) else {}
    stable_pages = canonical_document_paths(
        CHRONOVISOR_ROOT / "pages", require_stable=True
    )
    return {
        "claims": _jsonl_count(CHRONOVISOR_ROOT / "claims" / "claims-index.jsonl"),
        "golden": _jsonl_count(CHRONOVISOR_ROOT / "recall" / "search-golden.jsonl"),
        "distill_rows": _jsonl_count(CHRONOVISOR_ROOT / "distill" / "wiki-qa.jsonl"),
        "retention_pages": int(retention_counts.get("pages") or 0)
        if isinstance(retention_counts, dict)
        else 0,
        "deprecation_candidates": int(
            retention_counts.get("deprecation_candidates") or 0
        )
        if isinstance(retention_counts, dict)
        else 0,
        "hubs": sum(
            path.parent == (CHRONOVISOR_ROOT / "pages" / "hubs").resolve()
            for path in stable_pages
        ),
    }


def read_back_kpi() -> dict[str, Any]:
    run_path = CHRONOVISOR_ROOT / "runtime" / "ingest-read-back-runs.jsonl"
    rows = _read_jsonl(run_path, limit=200)
    cohort = "all_ingest_runs"
    if not rows:
        rows = _read_jsonl(
            CHRONOVISOR_ROOT / "runtime" / "ingest-read-back-failures.jsonl", limit=200
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
    ledger_path = CHRONOVISOR_ROOT / "runtime" / "ingest-read-back-repair.json"
    try:
        from chronovisor.core.durable_state import canonical_bytes
        from chronovisor.ingest.read_back_integrity import verify_prior_prefix

        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        observed = ledger.get("view_sha256") if isinstance(ledger, dict) else None
        unsigned = (
            {key: value for key, value in ledger.items() if key != "view_sha256"}
            if isinstance(ledger, dict)
            else {}
        )
        expected = hashlib.sha256(canonical_bytes(unsigned)).hexdigest()
        cursor = ledger.get("source_cursor") if isinstance(ledger, dict) else None
        source_path = CHRONOVISOR_ROOT / "runtime" / "ingest-read-back-failures.jsonl"
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
    from chronovisor.core.durable_state import DurableStateError, read_sealed_json
    from chronovisor.core.managed_hold import ManagedHoldStore
    from chronovisor.decision.quality_guard import quality_snapshot
    from chronovisor.ops.deadman import inspect_heartbeat
    from chronovisor.recall.provisional_recall import snapshot as provisional_snapshot

    runtime = CHRONOVISOR_ROOT / "runtime"
    autonomy = CHRONOVISOR_ROOT / "autonomy"
    managed = ManagedHoldStore(runtime / "managed-holds" / "state.json")
    try:
        managed_snapshot = managed.snapshot()
    except Exception as exc:
        managed_snapshot = {"status": "invalid", "error": str(exc)}
    try:
        provisional = provisional_snapshot(chronovisor_root=CHRONOVISOR_ROOT)
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
    rows = _read_jsonl(CHRONOVISOR_ROOT / "recall" / "feedback.jsonl", limit=1000)
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
    path = CHRONOVISOR_ROOT / "runtime" / "convergence" / "state.json"
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
    now = datetime.now(UTC)
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
            with contextlib.suppress(ValueError):
                actionable_dates.append(
                    datetime.fromisoformat(
                        str(item.get("created_at") or "")
                    ).astimezone(UTC)
                )
        if status in {"local_running", "frontier_running"}:
            try:
                expires = datetime.fromisoformat(
                    str(item.get("lease_expires_at") or "")
                ).astimezone(UTC)
                expired_running += int(expires <= now)
            except ValueError:
                expired_running += 1
    events = _read_jsonl(path.with_name("events.jsonl"), limit=100000)
    cutoff = now - timedelta(hours=24)
    recent = []
    for event in events:
        try:
            ts = datetime.fromisoformat(str(event.get("ts") or "")).astimezone(
                UTC
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
    from chronovisor.core.background_jobs import snapshot as job_snapshot

    sweeper_path = CHRONOVISOR_ROOT / "runtime" / "session-sweeper-latest.json"
    try:
        sweeper = json.loads(sweeper_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        sweeper = {"status": "missing", "pending": None, "processed": 0}
    return {"background_jobs": job_snapshot(), "session_sweeper": sweeper}


def ingest_liveness_kpi() -> dict[str, Any]:
    path = CHRONOVISOR_ROOT / "runtime" / "ingest-liveness.json"
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
    runtime_state = payload.get("status")
    waiting = runtime_state in {
        "waiting_for_ingest_runtime",
        "waiting_for_ollama",
    }
    authority_blocked = runtime_state == "blocked_by_decision_authority"
    pending = int(payload.get("pending_raws") or 0)
    alert = authority_blocked or (waiting and pending > 0)
    return {
        **payload,
        "status": "alert" if alert else "ok",
        "runtime_status": runtime_state,
        "alert": alert,
        "path": str(path),
    }


def research_kpi(*, limit: int = 200) -> dict[str, Any]:
    """Bounded summary of durable research traces for dashboard/alerts."""

    from chronovisor.search.research_config import load_research_config

    root = CHRONOVISOR_ROOT / "runtime" / "research"
    runs_root = root / "runs"
    try:
        summaries = sorted(
            runs_root.glob("*/summary.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )[: max(1, limit)]
    except OSError:
        summaries = []
    runs: list[dict[str, Any]] = []
    totals = {
        "runs": 0,
        "completed": 0,
        "terminal": 0,
        "first_pass_malformed": 0,
        "repair_turns": 0,
        "invalid_action_executions": 0,
        "actions": 0,
        "observations": 0,
        "supported_claims": 0,
        "contradicted_claims": 0,
        "unknown_claims": 0,
        "observation_bytes": 0,
    }
    stop_reasons: dict[str, int] = {}
    provider_counts: dict[str, int] = {}
    cache_counts: dict[str, int] = {}
    traced_actions = 0
    traced_observations = 0
    for index, path in enumerate(summaries):
        try:
            summary = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(summary, dict):
            continue
        totals["runs"] += 1
        status = str(summary.get("status") or "terminal")
        totals["completed" if status == "completed" else "terminal"] += 1
        for key in (
            "first_pass_malformed",
            "repair_turns",
            "invalid_action_executions",
            "actions",
            "observations",
        ):
            totals[key] += int(summary.get(key) or 0)
        usage = summary.get("usage") if isinstance(summary.get("usage"), dict) else {}
        totals["observation_bytes"] += int(usage.get("observation_bytes") or 0)
        reason = str(summary.get("stop_reason") or "unknown")
        stop_reasons[reason] = stop_reasons.get(reason, 0) + 1
        for claim in summary.get("claims", []):
            if not isinstance(claim, dict):
                continue
            claim_status = str(claim.get("status") or "unknown")
            key = f"{claim_status}_claims"
            if key in totals:
                totals[key] += 1
        events_path = path.with_name("events.jsonl")
        for event in _read_jsonl(events_path, limit=2_000) if index < 20 else ():
            if event.get("kind") == "action":
                traced_actions += 1
            elif event.get("kind") == "observation":
                traced_observations += 1
                metadata = (
                    event.get("metadata")
                    if isinstance(event.get("metadata"), dict)
                    else {}
                )
                provider = str(metadata.get("provider") or "")
                cache = str(metadata.get("cache") or "")
                if provider:
                    provider_counts[provider] = provider_counts.get(provider, 0) + 1
                if cache:
                    cache_counts[cache] = cache_counts.get(cache, 0) + 1
        runs.append(
            {
                "research_run_id": summary.get("research_run_id"),
                "status": status,
                "stop_reason": reason,
                "elapsed_ms": summary.get("elapsed_ms"),
                "usage": usage,
                "first_pass_malformed": summary.get("first_pass_malformed", 0),
                "repair_turns": summary.get("repair_turns", 0),
            }
        )
    config = load_research_config()
    claim_total = (
        totals["supported_claims"]
        + totals["contradicted_claims"]
        + totals["unknown_claims"]
    )
    return {
        "status": "ok",
        "enabled": config.enabled,
        "mode": config.mode,
        "kill_switches": {
            "agent": not config.enabled,
            "web": not (config.web.adapter_enabled and config.web.live_egress_enabled),
            "compaction": not config.compaction.enabled,
            "consolidation": not config.consolidation_enabled,
        },
        "totals": totals,
        "claim_coverage": (
            totals["supported_claims"] / claim_total if claim_total else 0.0
        ),
        "decision_trace_coverage": (
            min(1.0, traced_observations / traced_actions)
            if traced_actions
            else 1.0
        ),
        "stop_reasons": dict(sorted(stop_reasons.items())),
        "providers": dict(sorted(provider_counts.items())),
        "cache": dict(sorted(cache_counts.items())),
        "active": (root / "active-research.json").exists(),
        "recent": runs[:20],
    }


def _queue_status_counts(path: Path, field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in _read_jsonl(path, limit=100000):
        status = str(row.get(field) or "unknown")
        counts[status] = counts.get(status, 0) + 1
    return dict(sorted(counts.items()))


_LINT_ACTIVE_STATUSES = {
    "pending_local",
    "local_retry",
    "pending_frontier",
    "frontier_retry",
    "local_running",
    "frontier_running",
}


def _lint_queue_kpi(queue_path: Path, convergence_path: Path) -> dict[str, int]:
    """Count current unresolved lint issues instead of append-only history.

    The lint queue is a current detector snapshot, but a row remains present
    after the convergence lane has observed, routed, rejected, or applied it.
    Counting every JSONL row therefore turns completed monitor/review work into
    a permanent backlog alert.  ``issue_key`` is the detector's stable
    identity, so active convergence wins, a terminal convergence record marks
    the issue handled, and a row with no state is still genuinely unprocessed.
    """

    rows = _read_jsonl(queue_path, limit=100000)
    issue_keys: set[str] = set()
    missing_identity = 0
    for row in rows:
        issue_key = str(row.get("issue_key") or "")
        if issue_key:
            issue_keys.add(issue_key)
        else:
            missing_identity += 1

    statuses: dict[str, set[str]] = {}
    try:
        payload = json.loads(convergence_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        payload = {}
    items = payload.get("items") if isinstance(payload, dict) else None
    if isinstance(items, dict):
        for item in items.values():
            if not isinstance(item, dict) or item.get("lane") != "lint_repair":
                continue
            metadata = (
                item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
            )
            issue_key = str(metadata.get("issue_key") or "")
            if issue_key:
                statuses.setdefault(issue_key, set()).add(
                    str(item.get("status") or "unknown")
                )

    active = 0
    handled = 0
    untracked = missing_identity
    for issue_key in issue_keys:
        issue_statuses = statuses.get(issue_key, set())
        if issue_statuses & _LINT_ACTIVE_STATUSES:
            active += 1
        elif issue_statuses:
            handled += 1
        else:
            untracked += 1
    return {
        "total": len(rows),
        "unique": len(issue_keys) + missing_identity,
        "actionable": active + untracked,
        "active": active,
        "untracked": untracked,
        "handled": handled,
    }


def health_snapshot() -> dict[str, Any]:
    from chronovisor.core.runtime_config import runtime_identity
    from chronovisor.recall.librarian_status import build_librarian_status

    coverage = summary_coverage()
    duplicate_queue = CHRONOVISOR_ROOT / "review" / "duplicate-candidates.jsonl"
    lint_queue = CHRONOVISOR_ROOT / "review" / "lint-repair-queue.jsonl"
    golden = CHRONOVISOR_ROOT / "recall" / "search-golden.jsonl"
    label_queue = CHRONOVISOR_ROOT / "recall" / "search-label-queue.jsonl"
    raw_replay_queue = CHRONOVISOR_ROOT / "review" / "raw-replay-queue.jsonl"
    label_statuses = _queue_status_counts(label_queue, "queue_status")
    replay_statuses = _queue_status_counts(raw_replay_queue, "status")
    lint_queue_status = _lint_queue_kpi(
        lint_queue,
        CHRONOVISOR_ROOT / "runtime" / "convergence" / "state.json",
    )
    ingest_liveness = ingest_liveness_kpi()
    semantic_index = semantic_index_kpi()
    overall_alert = bool(ingest_liveness.get("alert")) or (
        semantic_index.get("status") == "alert"
    )
    return {
        "status": "alert" if overall_alert else "ok",
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
        "recall_distillation": recall_distillation_kpi(),
        "convergence": convergence_kpi(),
        "capture_pipeline": capture_pipeline_kpi(),
        "ingest_liveness": ingest_liveness,
        "semantic_index": semantic_index,
        "librarian": build_librarian_status(CHRONOVISOR_ROOT),
        "research": research_kpi(),
        "queues": {
            "duplicate_candidates": _jsonl_count(duplicate_queue),
            "lint_repair": lint_queue_status["actionable"],
            "lint_repair_total": lint_queue_status["total"],
            "lint_repair_active": lint_queue_status["active"],
            "lint_repair_untracked": lint_queue_status["untracked"],
            "lint_repair_handled": lint_queue_status["handled"],
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
