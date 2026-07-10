"""Knowledge health KPIs for dashboard and CLI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from llm_wiki_mcp.index_store import get_store
from llm_wiki_mcp.wiki import RAW_DIR, WIKI_ROOT


def _jsonl_count(path: Path) -> int:
    try:
        return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    except OSError:
        return 0


def _read_jsonl(path: Path, *, limit: int = 500) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()[-limit:]
    except OSError:
        return []
    rows: list[dict[str, Any]] = []
    for line in lines:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


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
        typed[str(meta.get("page_type") or "knowledge")] = typed.get(str(meta.get("page_type") or "knowledge"), 0) + 1
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
    rows = _read_jsonl(WIKI_ROOT / "runtime" / "ingest-read-back-failures.jsonl", limit=200)
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
    return {
        "checked": checked,
        "passed": passed,
        "failures": failures,
        "pass_rate": (passed / checked) if checked else None,
        "latest": latest,
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
        "missed_candidates": counts.get("missed_candidate", 0) + counts.get("missed", 0),
    }


def convergence_kpi() -> dict[str, Any]:
    path = WIKI_ROOT / "runtime" / "convergence" / "state.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"status": "missing", "path": str(path), "items": 0, "by_status": {}, "by_lane": {}}
    items = payload.get("items") if isinstance(payload, dict) else None
    if not isinstance(items, dict):
        return {"status": "invalid", "path": str(path), "items": 0, "by_status": {}, "by_lane": {}}
    by_status: dict[str, int] = {}
    by_lane: dict[str, dict[str, int]] = {}
    for item in items.values():
        if not isinstance(item, dict):
            continue
        status = str(item.get("status") or "unknown")
        lane = str(item.get("lane") or "unknown")
        by_status[status] = by_status.get(status, 0) + 1
        lane_counts = by_lane.setdefault(lane, {})
        lane_counts[status] = lane_counts.get(status, 0) + 1
    return {
        "status": "ok",
        "path": str(path),
        "items": sum(by_status.values()),
        "by_status": dict(sorted(by_status.items())),
        "by_lane": {lane: dict(sorted(counts.items())) for lane, counts in sorted(by_lane.items())},
        "actionable": sum(
            count
            for status, count in by_status.items()
            if status in {"pending_local", "local_retry", "pending_frontier", "frontier_retry"}
        ),
        "quarantined": by_status.get("quarantined", 0),
        "human_required": by_status.get("human_required", 0),
    }


def _queue_status_counts(path: Path, field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in _read_jsonl(path, limit=100000):
        status = str(row.get(field) or "unknown")
        counts[status] = counts.get(status, 0) + 1
    return dict(sorted(counts.items()))


def health_snapshot() -> dict[str, Any]:
    coverage = summary_coverage()
    duplicate_queue = WIKI_ROOT / "review" / "duplicate-candidates.jsonl"
    lint_queue = WIKI_ROOT / "review" / "lint-repair-queue.jsonl"
    golden = WIKI_ROOT / "recall" / "search-golden.jsonl"
    label_queue = WIKI_ROOT / "recall" / "search-label-queue.jsonl"
    raw_replay_queue = WIKI_ROOT / "review" / "raw-replay-queue.jsonl"
    label_statuses = _queue_status_counts(label_queue, "queue_status")
    replay_statuses = _queue_status_counts(raw_replay_queue, "status")
    return {
        "status": "ok",
        "coverage": coverage,
        "capture": capture_kpi(),
        "memory_integrity": latest_memory_integrity(),
        "cofire": cofire_kpi(),
        "prefetch": prefetch_kpi(),
        "derived": derived_memory_kpi(),
        "read_back": read_back_kpi(),
        "recall_feedback": recall_feedback_kpi(),
        "convergence": convergence_kpi(),
        "queues": {
            "duplicate_candidates": _jsonl_count(duplicate_queue),
            "lint_repair": _jsonl_count(lint_queue),
            "search_golden": _jsonl_count(golden),
            "search_labels": _jsonl_count(label_queue),
            "search_labels_pending": sum(
                count
                for status, count in label_statuses.items()
                if status in {"unknown", "pending_review", "pending_frontier_review", "frontier_retry", "frontier_uncertain"}
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
