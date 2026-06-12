"""Shared search pipeline orchestration."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from llm_wiki_mcp.runtime_config import NegativeFeedbackConfig
from llm_wiki_mcp.search_types import ScoredPage


@dataclass(frozen=True)
class PipelineConfig:
    top_n: int = 20
    folder: str | None = None
    updated_after: str | None = None
    updated_before: str | None = None
    sort_by: str = "relevance"
    semantic: bool = True
    fusion_weights: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class PipelineDependencies:
    get_bm25: Callable[[], Any]
    semantic_search: Callable[..., list[ScoredPage]]
    graph_expand_results: Callable[..., list[ScoredPage]]
    usage_prior_results: Callable[..., list[ScoredPage]]
    fuse_results: Callable[..., list[ScoredPage]]
    apply_filters: Callable[..., list[ScoredPage]]
    apply_sort: Callable[..., list[ScoredPage]]
    load_negative_feedback_config: Callable[[], NegativeFeedbackConfig]
    penalties_for_query: Callable[[str, NegativeFeedbackConfig], dict[str, float]]
    apply_penalties: Callable[[list[ScoredPage], dict[str, float]], list[ScoredPage]]


@dataclass(frozen=True)
class PipelineResult:
    results: list[ScoredPage]
    search_mode: str
    bm25_results: list[ScoredPage]
    semantic_results: list[ScoredPage]
    graph_results: list[ScoredPage]
    usage_results: list[ScoredPage]
    negative_feedback: dict[str, Any]


def run_search_pipeline(
    query: str,
    *,
    config: PipelineConfig,
    deps: PipelineDependencies,
) -> PipelineResult:
    """Run the current production search pipeline with explicit dependencies."""
    fetch_n = max(config.top_n * 5, 100)

    bm25 = deps.get_bm25()
    bm25.build()
    bm25_results = bm25.query(query, top_n=fetch_n)

    search_mode = "bm25"
    sem_results: list[ScoredPage] = []
    if config.semantic:
        sem_results = deps.semantic_search(query, top_n=fetch_n)
        if sem_results:
            search_mode = "hybrid"

    weights = dict(config.fusion_weights)
    graph_results = deps.graph_expand_results(
        bm25_results + sem_results,
        decay=float(weights.get("graph", 0.0) or 0.0),
        limit=fetch_n,
    )
    candidate_ids = {page.page_id for page in bm25_results + sem_results + graph_results}
    usage_results = (
        deps.usage_prior_results(
            candidate_ids,
            limit=fetch_n,
            decay=float(weights.get("usage_prior_decay", 0.98)),
            cap=float(weights.get("usage_prior_cap", 3.0)),
        )
        if float(weights.get("usage_prior", 0.0) or 0.0) > 0
        else []
    )
    if graph_results and search_mode == "bm25":
        search_mode = "bm25+graph"
    if config.semantic and sem_results:
        results = deps.fuse_results(
            bm25_results,
            sem_results,
            graph_results,
            usage_results,
            weights=weights,
        )
    elif graph_results or usage_results:
        results = deps.fuse_results(
            bm25_results,
            [],
            graph_results,
            usage_results,
            weights=weights,
        )
    else:
        results = bm25_results

    negative_meta: dict[str, Any] = {"status": "disabled"}
    negative_config = deps.load_negative_feedback_config()
    if negative_config.enabled:
        penalties = deps.penalties_for_query(query, negative_config)
        if penalties:
            results = deps.apply_penalties(results, penalties)
            negative_meta = {"status": "applied", "pages": sorted(penalties)}
        else:
            negative_meta = {"status": "no_match"}

    results = deps.apply_filters(
        results,
        config.folder,
        config.updated_after,
        config.updated_before,
    )
    results = deps.apply_sort(results, config.sort_by)
    return PipelineResult(
        results=results[: config.top_n],
        search_mode=search_mode,
        bm25_results=bm25_results,
        semantic_results=sem_results,
        graph_results=graph_results,
        usage_results=usage_results,
        negative_feedback=negative_meta,
    )
