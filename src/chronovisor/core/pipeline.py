"""Rerank stage shared by the MCP search tool."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from chronovisor.core.runtime_config import RerankerConfig
from chronovisor.core.search_types import ScoredPage


@dataclass(frozen=True)
class RerankStageResult:
    results: list[ScoredPage]
    metadata: dict[str, Any]
    applied: bool


def apply_rerank_stage(
    query: str,
    results: list[ScoredPage],
    *,
    reranker_config: RerankerConfig,
    rerank_results: Callable[..., Any],
    sort_by: str = "relevance",
) -> RerankStageResult:
    if sort_by != "relevance":
        return RerankStageResult(
            results=results,
            metadata={
                "status": "skipped",
                "reason": "sort_by_not_relevance",
            },
            applied=False,
        )
    outcome = rerank_results(query, results, config=reranker_config)
    metadata = dict(outcome.metadata)
    return RerankStageResult(
        results=outcome.results,
        metadata=metadata,
        applied=metadata.get("status") == "applied",
    )
