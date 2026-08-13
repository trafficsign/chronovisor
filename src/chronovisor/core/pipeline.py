"""Shared core search pipeline orchestration."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from chronovisor.core.runtime_config import NegativeFeedbackConfig, RerankerConfig
from chronovisor.core.search_types import ScoredPage


@dataclass(frozen=True)
class PipelineConfig:
    top_n: int = 20
    folder: str | None = None
    updated_after: str | None = None
    updated_before: str | None = None
    sort_by: str = "relevance"
    semantic: bool = True
    fusion_weights: dict[str, float] = field(default_factory=dict)
    result_strategy: str = "production"
    graph_strategy: str = "production"
    graph_decay: float | None = None
    usage_strategy: str = "production"
    usage_include_graph: bool = True
    plain_rrf_weights: dict[str, float] = field(default_factory=dict)
    apply_negative_feedback: bool = True
    filter_results: bool = True
    sort_results: bool = True
    truncate_results: bool = True
    include_reference: bool = False
    semantic_timeout_ms: int | None = None
    anchor_seed: bool = True
    context_seed: bool = True
    verify_graph: bool = True


@dataclass(frozen=True)
class PipelineDependencies:
    get_bm25: Callable[[], Any]
    context_seed_results: Callable[..., list[ScoredPage]]
    semantic_search: Callable[..., list[ScoredPage]]
    semantic_verify: Callable[..., list[ScoredPage]]
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
    anchor_results: list[ScoredPage]
    bm25_results: list[ScoredPage]
    semantic_results: list[ScoredPage]
    graph_results: list[ScoredPage]
    context_results: list[ScoredPage]
    usage_results: list[ScoredPage]
    negative_feedback: dict[str, Any]
    stage_timings_ms: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class RerankStageResult:
    results: list[ScoredPage]
    metadata: dict[str, Any]
    applied: bool


def production_pipeline_config(
    *,
    top_n: int = 20,
    folder: str | None = None,
    updated_after: str | None = None,
    updated_before: str | None = None,
    sort_by: str = "relevance",
    semantic: bool = True,
    fusion_weights: dict[str, float] | None = None,
    include_reference: bool = False,
    semantic_timeout_ms: int | None = None,
) -> PipelineConfig:
    return PipelineConfig(
        top_n=top_n,
        folder=folder,
        updated_after=updated_after,
        updated_before=updated_before,
        sort_by=sort_by,
        semantic=semantic,
        fusion_weights=dict(fusion_weights or {}),
        result_strategy="production",
        graph_strategy="production",
        usage_strategy="production",
        include_reference=include_reference,
        semantic_timeout_ms=semantic_timeout_ms,
    )


def plain_rrf(
    channels: list[tuple[str, list[ScoredPage]]],
    *,
    k: int = 60,
    weights: dict[str, float] | None = None,
) -> list[ScoredPage]:
    weights = weights or {}
    scores: dict[str, float] = {}
    meta: dict[str, ScoredPage] = {}
    for channel, results in channels:
        weight = max(0.0, float(weights.get(channel, 1.0)))
        if weight == 0:
            continue
        for rank, page in enumerate(results):
            scores[page.page_id] = scores.get(page.page_id, 0.0) + weight / (k + rank)
            meta.setdefault(page.page_id, page)

    fused: list[ScoredPage] = []
    for page_id, score in scores.items():
        page = meta[page_id]
        fused.append(
            ScoredPage(
                page_id=page.page_id,
                title=page.title,
                folder=page.folder,
                updated=page.updated,
                score=score,
                status=page.status,
                superseded_by=page.superseded_by,
                page_type=page.page_type,
                sensitivity=page.sensitivity,
            )
        )
    return sorted(fused, key=lambda page: page.score, reverse=True)


def apply_negative_feedback_stage(
    query: str,
    results: list[ScoredPage],
    *,
    deps: PipelineDependencies,
) -> tuple[list[ScoredPage], dict[str, Any]]:
    negative_meta: dict[str, Any] = {"status": "disabled"}
    negative_config = deps.load_negative_feedback_config()
    if not negative_config.enabled:
        return results, negative_meta
    penalties = deps.penalties_for_query(query, negative_config)
    if penalties:
        return (
            deps.apply_penalties(results, penalties),
            {"status": "applied", "pages": sorted(penalties)},
        )
    return results, {"status": "no_match"}


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


def _graph_results(
    anchor_results: list[ScoredPage],
    bm25_results: list[ScoredPage],
    sem_results: list[ScoredPage],
    context_results: list[ScoredPage],
    *,
    config: PipelineConfig,
    deps: PipelineDependencies,
    fetch_n: int,
) -> list[ScoredPage]:
    if config.graph_strategy == "disabled":
        return []
    decay = (
        float(config.graph_decay)
        if config.graph_decay is not None
        else float(config.fusion_weights.get("graph", 0.0) or 0.0)
    )
    seeds = plain_rrf(
        [
            ("anchor", anchor_results),
            ("bm25", bm25_results),
            ("semantic", sem_results),
            ("context", context_results),
        ],
        weights=config.fusion_weights,
    )[:20]
    return deps.graph_expand_results(
        seeds,
        decay=decay,
        limit=fetch_n,
    )


def _usage_results(
    anchor_results: list[ScoredPage],
    bm25_results: list[ScoredPage],
    sem_results: list[ScoredPage],
    graph_results: list[ScoredPage],
    context_results: list[ScoredPage],
    *,
    config: PipelineConfig,
    deps: PipelineDependencies,
    fetch_n: int,
) -> list[ScoredPage]:
    if config.usage_strategy == "disabled":
        return []
    candidates = anchor_results + bm25_results + sem_results + context_results
    if config.usage_include_graph:
        candidates += graph_results
    candidate_ids = {page.page_id for page in candidates}
    if config.usage_strategy == "production":
        if float(config.fusion_weights.get("usage_prior", 0.0) or 0.0) <= 0:
            return []
        return deps.usage_prior_results(
            candidate_ids,
            limit=fetch_n,
            decay=float(config.fusion_weights.get("usage_prior_decay", 0.98)),
            cap=float(config.fusion_weights.get("usage_prior_cap", 3.0)),
        )
    if config.usage_strategy == "always":
        return deps.usage_prior_results(candidate_ids, limit=fetch_n)
    raise ValueError(f"unknown usage strategy: {config.usage_strategy}")


def _select_results(
    anchor_results: list[ScoredPage],
    bm25_results: list[ScoredPage],
    sem_results: list[ScoredPage],
    graph_results: list[ScoredPage],
    context_results: list[ScoredPage],
    usage_results: list[ScoredPage],
    *,
    config: PipelineConfig,
    deps: PipelineDependencies,
) -> list[ScoredPage]:
    if config.result_strategy == "bm25":
        return bm25_results
    if config.result_strategy == "semantic":
        return sem_results
    if config.result_strategy == "plain_rrf":
        return plain_rrf(
            [("bm25", bm25_results), ("semantic", sem_results)],
            weights=config.plain_rrf_weights,
        )
    if config.result_strategy == "weighted_fusion":
        return deps.fuse_results(
            bm25_results,
            sem_results,
            graph_results,
            usage_results,
            weights=config.fusion_weights,
            anchor_results=anchor_results,
            context_results=context_results,
        )
    if config.result_strategy != "production":
        raise ValueError(f"unknown result strategy: {config.result_strategy}")
    if config.semantic and sem_results:
        return deps.fuse_results(
            bm25_results,
            sem_results,
            graph_results,
            usage_results,
            weights=config.fusion_weights,
            anchor_results=anchor_results,
            context_results=context_results,
        )
    if anchor_results or graph_results or context_results or usage_results:
        return deps.fuse_results(
            bm25_results,
            [],
            graph_results,
            usage_results,
            weights=config.fusion_weights,
            anchor_results=anchor_results,
            context_results=context_results,
        )
    return bm25_results


def run_search_pipeline(
    query: str,
    *,
    config: PipelineConfig,
    deps: PipelineDependencies,
    stage_timings_ms: dict[str, int] | None = None,
) -> PipelineResult:
    """Run the current production search pipeline with explicit dependencies."""
    fetch_n = max(config.top_n * 5, 100)
    timings = stage_timings_ms if stage_timings_ms is not None else {}

    def timed(name: str, fn: Callable[[], Any]) -> Any:
        t0 = time.monotonic()
        try:
            return fn()
        finally:
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            timings[name] = timings.get(name, 0) + elapsed_ms

    bm25 = timed("bm25_load", deps.get_bm25)
    timed("bm25_build", bm25.build)
    anchor_results: list[ScoredPage] = []
    if config.anchor_seed:
        anchor_query = getattr(bm25, "anchor_query", None)
        if callable(anchor_query):
            try:
                anchor_results = timed(
                    "bm25_anchor",
                    lambda: anchor_query(
                        query,
                        top_n=min(fetch_n, 20),
                        include_reference=config.include_reference,
                    ),
                )
            except TypeError:
                anchor_results = timed(
                    "bm25_anchor",
                    lambda: anchor_query(query, top_n=min(fetch_n, 20)),
                )
    try:
        bm25_results = timed(
            "bm25_query",
            lambda: bm25.query(
                query,
                top_n=fetch_n,
                include_reference=config.include_reference,
            ),
        )
    except TypeError:
        bm25_results = timed(
            "bm25_query", lambda: bm25.query(query, top_n=fetch_n)
        )

    context_results: list[ScoredPage] = []
    if config.context_seed:
        context_results = timed(
            "context_seed", lambda: deps.context_seed_results(query, limit=4)
        )

    search_mode = "bm25"
    sem_results: list[ScoredPage] = []
    if config.semantic:
        try:
            sem_results = timed(
                "semantic",
                lambda: deps.semantic_search(
                    query,
                    top_n=fetch_n,
                    include_reference=config.include_reference,
                    timeout_ms=config.semantic_timeout_ms,
                ),
            )
        except TypeError:
            sem_results = timed(
                "semantic",
                lambda: deps.semantic_search(query, top_n=fetch_n),
            )
        if sem_results:
            search_mode = "hybrid"

    graph_results = timed(
        "graph",
        lambda: _graph_results(
            anchor_results,
            bm25_results,
            sem_results,
            context_results,
            config=config,
            deps=deps,
            fetch_n=fetch_n,
        ),
    )
    if config.verify_graph and sem_results and graph_results:
        try:
            verified = timed(
                "verify",
                lambda: deps.semantic_verify(
                    query,
                    [page.page_id for page in graph_results],
                    timeout_ms=config.semantic_timeout_ms,
                ),
            )
        except TypeError:
            verified = timed(
                "verify",
                lambda: deps.semantic_verify(
                    query, [page.page_id for page in graph_results]
                ),
            )
        semantic_by_page = {page.page_id: page for page in sem_results}
        for page in verified:
            current = semantic_by_page.get(page.page_id)
            if current is None or page.score > current.score:
                semantic_by_page[page.page_id] = page
        sem_results = sorted(
            semantic_by_page.values(),
            key=lambda page: page.score,
            reverse=True,
        )[:fetch_n]
    usage_results = timed(
        "usage",
        lambda: _usage_results(
            anchor_results,
            bm25_results,
            sem_results,
            graph_results,
            context_results,
            config=config,
            deps=deps,
            fetch_n=fetch_n,
        ),
    )
    if graph_results and search_mode == "bm25":
        search_mode = "bm25+graph"
    results = _select_results(
        anchor_results,
        bm25_results,
        sem_results,
        graph_results,
        context_results,
        usage_results,
        config=config,
        deps=deps,
    )

    negative_meta: dict[str, Any] = {"status": "disabled"}
    if config.apply_negative_feedback:
        results, negative_meta = timed(
            "negative_feedback",
            lambda: apply_negative_feedback_stage(query, results, deps=deps),
        )

    if config.filter_results:
        results = timed(
            "filter",
            lambda: deps.apply_filters(
                results,
                config.folder,
                config.updated_after,
                config.updated_before,
            ),
        )
    if config.sort_results:
        results = deps.apply_sort(results, config.sort_by)
    if config.truncate_results:
        results = results[: config.top_n]
    return PipelineResult(
        results=results,
        search_mode=search_mode,
        anchor_results=anchor_results,
        bm25_results=bm25_results,
        semantic_results=sem_results,
        graph_results=graph_results,
        context_results=context_results,
        usage_results=usage_results,
        negative_feedback=negative_meta,
        stage_timings_ms=timings,
    )
