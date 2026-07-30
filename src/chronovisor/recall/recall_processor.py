"""Precision-first orchestration helpers for synchronous Recall."""

from __future__ import annotations

import time
from typing import Any

from chronovisor.core.runtime_config import load_reranker_config
from chronovisor.search import reranker_client


def shadow_rerank_candidates(
    query: str,
    candidates: list[Any],
    *,
    timeout_ms: int,
) -> dict[str, Any]:
    """Run the resident reranker without changing production candidate order."""

    config = load_reranker_config()
    if config.service.mode != "shadow":
        return {"status": "disabled", "reason": "not_shadow_mode"}
    if not reranker_client.selected_for_rollout(query, config):
        return {"status": "disabled", "reason": "not_selected"}
    if not candidates:
        return {"status": "skipped", "reason": "no_candidates"}
    before = [candidate.page_id for candidate in candidates[: config.top_n]]
    started = time.perf_counter()
    try:
        outcome = reranker_client.rerank(
            query,
            candidates,
            config=config,
            timeout_ms=max(25, min(timeout_ms, config.service.timeout_ms)),
        )
    except Exception as exc:
        return {
            "status": "unavailable",
            "reason": type(exc).__name__,
            "latency_ms": int(round((time.perf_counter() - started) * 1_000)),
        }
    after = [candidate.page_id for candidate in outcome.results[: config.top_n]]
    overlap = len(set(before[:5]) & set(after[:5]))
    return {
        "status": outcome.metadata.get("status", "unknown"),
        "execution": outcome.metadata.get("execution", "service"),
        "before_page_ids": before,
        "after_page_ids": after,
        "top5_overlap": overlap,
        "changed": before != after,
        "latency_ms": int(round((time.perf_counter() - started) * 1_000)),
        "service_latency_ms": outcome.metadata.get("latency_ms", 0),
        "scores": outcome.metadata.get("scores", []),
    }
