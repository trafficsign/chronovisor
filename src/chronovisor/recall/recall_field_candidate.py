"""Shadow/canary candidate lane for Stateful Recall Field."""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from chronovisor.core.durable_state import exclusive_text_file_lock
from chronovisor.core.runtime_config import load_search_embedding_config
from chronovisor.core.store import CHRONOVISOR_ROOT
from chronovisor.recall.recall_field_schema import (
    RecallFieldConfig,
    load_recall_field_config,
)
from chronovisor.search import semantic_client

CANDIDATE_TRACE_FILE = (
    CHRONOVISOR_ROOT / "runtime" / "recall-field" / "candidate-trace.jsonl"
)
PROMOTION_ARTIFACT = CHRONOVISOR_ROOT / "runtime" / "recall-field" / "promotion.json"


def _canonical_sha256(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def selected_for_canary(session_hash: str, config: RecallFieldConfig) -> bool:
    """Select sessions deterministically; never use raw session IDs."""

    if config.mode not in {"candidate", "active"} or not session_hash:
        return False
    percent = config.canary_percent
    if config.mode == "active" and percent <= 0:
        percent = 100
    if percent <= 0:
        return False
    bucket = (
        int(
            hashlib.sha256(f"recall-field:{session_hash}".encode()).hexdigest()[:8],
            16,
        )
        % 100
    )
    return bucket < percent


def effective_rollout(config: RecallFieldConfig) -> RecallFieldConfig:
    """Resolve autonomous rollout state without mutating the user config."""

    if not config.auto_promote or config.mode in {"off", "shadow"}:
        return config
    try:
        from dataclasses import replace

        from chronovisor.recall.recall_growth import automatic_rollout

        mode, percent = automatic_rollout(enabled=True)
        return replace(config, mode=mode, canary_percent=percent)
    except Exception:
        return config


def authority_allowed(path: Path | None = None) -> bool:
    """Require a sealed, non-degrading promotion artifact for Field authority."""

    artifact = path or PROMOTION_ARTIFACT
    try:
        payload = json.loads(artifact.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict) or payload.get("status") != "passed":
        return False
    expected_sha = str(payload.get("snapshot_sha256") or "")
    unsigned = {
        key: value for key, value in payload.items() if key != "snapshot_sha256"
    }
    if expected_sha != _canonical_sha256(unsigned):
        return False
    metrics = payload.get("metrics")
    try:
        return bool(
            isinstance(metrics, dict)
            and float(metrics.get("teacher_commit_coverage") or 0.0) >= 0.99
            and float(metrics.get("precision_delta_points") or -100.0) >= -1.0
            and float(metrics.get("recall_delta_points") or -100.0) >= -1.0
            and int(metrics.get("over_4s") or 0) == 0
            and float(metrics.get("processor_used_precision_proxy") or 0.0) >= 0.90
            and expected_sha
        )
    except (TypeError, ValueError):
        return False


def _verify(
    query: str,
    page_ids: list[str],
    *,
    timeout_ms: int,
) -> tuple[list[Any], dict[str, Any]]:
    started = time.perf_counter()
    if not page_ids:
        return [], {"status": "fallback", "reason": "empty_field"}
    config = load_search_embedding_config()
    try:
        results = semantic_client.verify(
            query,
            page_ids[:30],
            config=config,
            timeout_ms=max(25, timeout_ms),
        )
    except Exception as exc:
        return [], {
            "status": "fallback",
            "reason": type(exc).__name__,
            "latency_ms": int(round((time.perf_counter() - started) * 1_000)),
        }
    return results, {
        "status": "verified",
        "candidate_count": len(page_ids[:30]),
        "verified_count": len(results),
        "latency_ms": int(round((time.perf_counter() - started) * 1_000)),
    }


def run_candidate_teacher_pair(
    *,
    query: str,
    field_turn: dict[str, Any],
    teacher_search: Callable[[], tuple[list[Any], str]],
    timeout_ms: int,
    config: RecallFieldConfig | None = None,
    certificate_boundary_enabled: bool = True,
) -> tuple[list[Any], str, dict[str, Any]]:
    """Run Field verification and the full teacher concurrently.

    The returned ranking remains teacher-owned unless a sealed promotion
    artifact exists. Candidate mode therefore cannot bypass certificates.
    """

    cfg = effective_rollout(config or load_recall_field_config())
    session = str(field_turn.get("session_hash") or "")
    page_ids = [
        str(page_id)
        for page_id in field_turn.get("candidate_page_ids", [])
        if isinstance(page_id, str) and page_id
    ][:30]
    fallback = field_turn.get("full_search_fallback") is True
    selected = selected_for_canary(session, cfg)
    if not selected or fallback or not page_ids:
        results, mode = teacher_search()
        return (
            results,
            mode,
            {
                "status": "fallback",
                "reason": (
                    "not_selected"
                    if not selected
                    else "topic_reset"
                    if fallback
                    else "empty_field"
                ),
                "teacher_count": len(results),
                "authority": "teacher",
            },
        )

    with ThreadPoolExecutor(
        max_workers=2,
        thread_name_prefix="recall-field-candidate",
    ) as executor:
        teacher_future = executor.submit(teacher_search)
        field_future = executor.submit(
            _verify,
            query,
            page_ids,
            timeout_ms=max(25, timeout_ms),
        )
        teacher_results, teacher_mode = teacher_future.result()
        verified, verify_meta = field_future.result()

    teacher_ids = [str(row.page_id) for row in teacher_results[:30]]
    field_ids = [str(row.page_id) for row in verified[:30]]
    overlap = len(set(teacher_ids) & set(field_ids))
    coverage = overlap / max(1, len(set(teacher_ids)))
    promoted = bool(
        cfg.mode == "active" and certificate_boundary_enabled and authority_allowed()
    )
    return (
        verified if promoted else teacher_results,
        "field-active" if promoted else teacher_mode,
        {
            **verify_meta,
            "status": "active" if promoted else "observed",
            "authority": "field" if promoted else "teacher",
            "field_page_ids": field_ids,
            "teacher_page_ids": teacher_ids,
            "teacher_top30_coverage": round(coverage, 6),
            "missed_page_ids": [
                page_id for page_id in teacher_ids if page_id not in set(field_ids)
            ],
            "rollback": bool(cfg.mode == "active" and not promoted),
            "rollback_reason": (
                "certificate_boundary_disabled"
                if cfg.mode == "active" and not certificate_boundary_enabled
                else "promotion_artifact_missing_or_failed"
                if cfg.mode == "active" and not promoted
                else ""
            ),
            "effective_mode": "active" if promoted else "shadow",
        },
    )


def append_candidate_trace(
    *,
    session_hash: str,
    prompt: str,
    observer: dict[str, Any],
    committed_page_ids: list[str],
    latency_ms: int,
    path: Path = CANDIDATE_TRACE_FILE,
) -> dict[str, Any]:
    """Persist privacy-safe Field/teacher disagreement evidence."""

    field_ids = [
        str(value)
        for value in observer.get("field_page_ids", [])
        if isinstance(value, str)
    ]
    committed = list(
        dict.fromkeys(page_id for page_id in committed_page_ids if page_id)
    )
    commit_coverage = (
        len(set(field_ids) & set(committed)) / len(committed) if committed else 1.0
    )
    record = {
        "schema_version": 1,
        "ts_epoch": round(time.time(), 3),
        "session_hash": session_hash,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "status": str(observer.get("status") or ""),
        "authority": str(observer.get("authority") or "teacher"),
        "fallback_reason": str(observer.get("reason") or ""),
        "teacher_top30_coverage": float(observer.get("teacher_top30_coverage") or 0.0),
        "teacher_commit_coverage": round(commit_coverage, 6),
        "field_page_ids": field_ids,
        "teacher_page_ids": list(observer.get("teacher_page_ids") or []),
        "committed_page_ids": committed,
        "missed_page_ids": list(observer.get("missed_page_ids") or []),
        "latency_ms": max(0, int(latency_ms)),
        "over_4s": latency_ms > 4_000,
        "rollback": observer.get("rollback") is True,
        "rollback_reason": str(observer.get("rollback_reason") or ""),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with exclusive_text_file_lock(lock_path):
        with path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
    os.chmod(path, 0o600)
    return record
