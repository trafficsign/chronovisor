"""Core cross-encoder reranking for interactive MCP search."""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from chronovisor.core.canonical_document import (
    CanonicalDocumentError,
    parse_document,
)
from chronovisor.core.index_store import IndexStore, contained_file, get_store
from chronovisor.core.llm_runtime import (
    SAFE_FAILURE_CATEGORIES,
    LLMRuntime,
    LLMRuntimeError,
    RerankRequest,
    ResolvedRerankRoute,
    SourceDataClass,
    SourceDataClassification,
    SourceSensitivity,
)
from chronovisor.core.page_identity import normalize_page_uid
from chronovisor.core.runtime_config import RerankerConfig, load_reranker_config
from chronovisor.core.search_types import (
    ScoredPage,
    SemanticEvidence,
    parse_semantic_evidence,
)
from chronovisor.core.semantic_evidence import resolve_semantic_evidence
from chronovisor.core.store import PAGES_DIR, SYSTEM_DIR

RERANK_RUNTIME_ROLE = "search.rerank"
QUERY_SOURCE = SourceDataClassification(
    SourceDataClass.RAW,
    SourceSensitivity.NORMAL,
)
_NORMAL_PAGE_SOURCE = SourceDataClassification(
    SourceDataClass.PAGE,
    SourceSensitivity.NORMAL,
)
_HIGH_PAGE_SOURCE = SourceDataClassification(
    SourceDataClass.PAGE,
    SourceSensitivity.HIGH,
)
_SYSTEM_SOURCE = SourceDataClassification(
    SourceDataClass.SYSTEM,
    SourceSensitivity.HIGH,
)


@dataclass(frozen=True)
class RerankOutcome:
    results: list[ScoredPage]
    metadata: dict[str, Any]
    scores: tuple[RerankScore, ...] = ()


@dataclass(frozen=True)
class RerankScore:
    page_id: str
    raw_score: float
    original_rank: int
    rerank_rank: int
    margin_to_next: float


def _route_identity(route: ResolvedRerankRoute) -> dict[str, str]:
    return {
        "role": route.role,
        "provider": route.provider,
        "model": route.model,
        "location": route.location.value,
    }


def safe_reranker_error(exc: Exception) -> str:
    category = getattr(exc, "category", "")
    return (
        category
        if isinstance(exc, LLMRuntimeError) and category in SAFE_FAILURE_CATEGORIES
        else "reranker_unavailable"
    )


def resolve_rerank_candidate(
    candidate: ScoredPage | str,
    *,
    store: IndexStore | None,
    max_chars: int = 2400,
    evidence: tuple[SemanticEvidence, ...] | None = None,
) -> tuple[
    str,
    SourceDataClassification,
    tuple[str, str, int, int, str],
]:
    """Resolve passage bytes and their egress classification from one index row."""

    page_id = candidate.page_id if isinstance(candidate, ScoredPage) else candidate
    title = candidate.title if isinstance(candidate, ScoredPage) else page_id
    snippet = candidate.snippet if isinstance(candidate, ScoredPage) else ""
    candidate_evidence: tuple[SemanticEvidence, ...] = ()
    evidence_invalid = False

    def parse_candidate_evidence(
        values: object,
    ) -> tuple[SemanticEvidence, ...] | None:
        if not isinstance(values, tuple):
            return None
        if not values:
            return ()
        if any(not isinstance(item, SemanticEvidence) for item in values):
            return None
        try:
            return parse_semantic_evidence(
                [asdict(item) for item in values],
                page_id=page_id,
            )
        except TypeError, ValueError:
            return None

    if isinstance(candidate, ScoredPage):
        raw_evidence = candidate.evidence
        if raw_evidence:
            parsed = parse_candidate_evidence(raw_evidence)
            if parsed is None:
                evidence_invalid = True
            else:
                candidate_evidence = parsed
    if evidence is not None:
        parsed = parse_candidate_evidence(evidence)
        if parsed is None:
            evidence_invalid = True
        else:
            candidate_evidence = parsed
    has_evidence = bool(candidate_evidence) or evidence_invalid
    fallback = f"{title}\n\n{snippet}".strip() or page_id
    # A candidate that carried evidence must never silently fall back to a
    # stale snippet when that evidence cannot be resolved.  The page id is a
    # bounded, source-free placeholder and remains classified as high-risk.
    invalid_fallback = page_id if has_evidence else fallback
    invalid_fallback_bytes = invalid_fallback.encode("utf-8")
    invalid = (
        invalid_fallback,
        _SYSTEM_SOURCE,
        (
            "invalid",
            "",
            0,
            len(invalid_fallback_bytes),
            hashlib.sha256(invalid_fallback_bytes).hexdigest(),
        ),
    )
    if evidence_invalid:
        return invalid
    try:
        if store is None:
            return invalid
        metadata = store.meta(page_id)
        if not isinstance(metadata, dict):
            return invalid
        namespace = metadata.get("namespace")
        path_value = metadata.get("path")
        if namespace not in {"pages", "system"} or not isinstance(path_value, str):
            return invalid
        path = contained_file(
            Path(path_value),
            SYSTEM_DIR if namespace == "system" else PAGES_DIR,
        )
        if path is None:
            return invalid
        stat = path.stat()
        data = path.read_bytes()
        document = parse_document(data)
        body = document.body.decode("utf-8").strip()
    except CanonicalDocumentError, OSError, RuntimeError, UnicodeDecodeError:
        return invalid
    source_digest = hashlib.sha256(data).hexdigest()
    if (
        isinstance(candidate, ScoredPage)
        and has_evidence
        and candidate.content_sha256
        and candidate.content_sha256 != source_digest
    ):
        return invalid
    if has_evidence:
        if document.metadata.get("status") != "stable":
            return invalid
        raw_source_uid = document.metadata.get("uid")
        if raw_source_uid in (None, ""):
            source_uid = ""
        else:
            try:
                source_uid = normalize_page_uid(raw_source_uid)
            except TypeError, ValueError:
                return invalid
        if isinstance(candidate, ScoredPage) and candidate.uid:
            try:
                candidate_uid = normalize_page_uid(candidate.uid)
            except TypeError, ValueError:
                return invalid
            if candidate_uid != source_uid:
                return invalid
        for item in candidate_evidence:
            if item.source_sha256 != source_digest or item.page_uid != source_uid:
                return invalid
    source = (
        _SYSTEM_SOURCE
        if namespace == "system"
        else _NORMAL_PAGE_SOURCE
        if document.metadata.get("sensitivity") == "normal"
        else _HIGH_PAGE_SOURCE
    )
    resolved_title = metadata.get("title")
    if not isinstance(resolved_title, str) or not resolved_title.strip():
        resolved_title = title
    source_evidence = tuple(
        item for item in candidate_evidence if item.kind in {"chunk", "section-v1"}
    )
    if source_evidence:
        try:
            passages = resolve_semantic_evidence(data, page_id, source_evidence)
        except Exception:
            passages = ()
        if not passages:
            return invalid
        selected_body = "\n\n".join(passage.text for passage in passages).strip()
        if not selected_body:
            return invalid
        passage = f"{resolved_title}\n\n{selected_body[:max_chars]}".strip()
    elif candidate_evidence:
        # Page/question documents are retrieval keys. Once their identities are
        # bound to this canonical source, the canonical body is the safe passage
        # fallback; their synthetic text is never sent to the model.
        passage = f"{resolved_title}\n\n{body[:max_chars]}".strip()
    else:
        if snippet:
            body = f"{snippet}\n\n{body}"
        passage = f"{resolved_title}\n\n{body[:max_chars]}".strip()
    return (
        passage,
        source,
        (
            namespace,
            str(path),
            stat.st_mtime_ns,
            stat.st_size,
            hashlib.sha256(data).hexdigest(),
        ),
    )


def warm_reranker(
    config: RerankerConfig | None = None,
    runtime: LLMRuntime | None = None,
) -> dict[str, Any]:
    """Load and exercise the configured reranker before the first search."""

    cfg = config or load_reranker_config()
    if not cfg.enabled:
        return {"status": "disabled", "reason": "config_disabled"}
    started = time.perf_counter()
    try:
        if runtime is None:
            from chronovisor.core.llm_config import load_default_llm_runtime

            runtime = load_default_llm_runtime()
        route = runtime.resolve_rerank(RERANK_RUNTIME_ROLE)
        result = runtime.rerank(
            RERANK_RUNTIME_ROLE,
            RerankRequest(
                "reranker warmup",
                ("reranker warmup",),
                QUERY_SOURCE,
                candidate_sources=(_NORMAL_PAGE_SOURCE,),
            ),
        )
    except Exception as exc:
        return {
            "status": "unavailable",
            "reason": safe_reranker_error(exc),
        }
    return {
        "status": "ready" if len(result.items) == 1 else "unavailable",
        "route": _route_identity(route),
        "latency_ms": int(round((time.perf_counter() - started) * 1000)),
    }


def apply_reranker_scores(
    candidates: list[ScoredPage],
    raw_scores: list[float],
    *,
    config: RerankerConfig,
    metadata: dict[str, Any] | None = None,
) -> RerankOutcome:
    """Apply raw cross-encoder scores and preserve score/rank evidence."""

    rerank_n = min(max(1, config.top_n), len(candidates))
    head = candidates[:rerank_n]
    tail = candidates[rerank_n:]
    base_metadata = {
        "candidate_count": rerank_n,
        **(metadata or {}),
    }
    if len(raw_scores) != rerank_n or any(
        not math.isfinite(float(score)) for score in raw_scores
    ):
        return RerankOutcome(
            candidates,
            {
                **base_metadata,
                "status": "unavailable",
                "reason": "invalid_response",
                "score_count": len(raw_scores),
                "degraded": True,
            },
        )

    original_rank = {page.page_id: rank for rank, page in enumerate(head, start=1)}
    reranked = sorted(
        zip(head, (float(score) for score in raw_scores), strict=False),
        key=lambda item: (
            -item[1],
            original_rank[item[0].page_id],
            item[0].page_id,
        ),
    )
    rerank_rank = {
        page.page_id: rank for rank, (page, _score) in enumerate(reranked, start=1)
    }
    raw_by_page = {page.page_id: score for page, score in reranked}
    score_details: list[RerankScore] = []
    for index, (page, raw_score) in enumerate(reranked):
        next_score = reranked[index + 1][1] if index + 1 < len(reranked) else raw_score
        score_details.append(
            RerankScore(
                page_id=page.page_id,
                raw_score=raw_score,
                original_rank=original_rank[page.page_id],
                rerank_rank=rerank_rank[page.page_id],
                margin_to_next=max(0.0, raw_score - next_score),
            )
        )

    rescored = []
    for page in head:
        original_score = 1.0 / (60 + original_rank[page.page_id])
        rerank_score = 1.0 / (60 + rerank_rank[page.page_id])
        blended_score = original_score + (max(0.0, config.weight) * rerank_score)
        rescored.append(replace(page, score=blended_score))
    rescored.sort(
        key=lambda page: (
            -page.score,
            -raw_by_page[page.page_id],
            original_rank[page.page_id],
            page.page_id,
        )
    )
    moved = sum(
        original_rank[page.page_id] != rerank_rank[page.page_id] for page in head
    )
    max_rank_delta = max(
        (abs(original_rank[page.page_id] - rerank_rank[page.page_id]) for page in head),
        default=0,
    )
    serialized_scores = [
        {
            "page_id": detail.page_id,
            "raw_score": detail.raw_score,
            "original_rank": detail.original_rank,
            "rerank_rank": detail.rerank_rank,
            "margin_to_next": detail.margin_to_next,
        }
        for detail in score_details
    ]
    return RerankOutcome(
        rescored + tail,
        {
            **base_metadata,
            "status": "applied",
            "weight": config.weight,
            "moved_candidates": moved,
            "max_rank_delta": max_rank_delta,
            "scores": serialized_scores,
        },
        tuple(score_details),
    )


def rerank_results(
    query: str,
    candidates: list[ScoredPage],
    *,
    config: RerankerConfig | None = None,
) -> RerankOutcome:
    cfg = config or load_reranker_config()
    if not cfg.enabled:
        return RerankOutcome(
            candidates,
            {
                "status": "disabled",
                "reason": "config_disabled",
            },
        )
    if not candidates:
        return RerankOutcome(
            candidates, {"status": "skipped", "reason": "no_candidates"}
        )

    rerank_n = min(max(1, cfg.top_n), len(candidates))
    if cfg.service.enabled:
        from chronovisor.core import reranker_client

        try:
            return reranker_client.rerank(
                query,
                candidates,
                config=cfg,
                timeout_ms=cfg.service.timeout_ms,
            )
        except Exception as exc:
            reason = (
                exc.category
                if isinstance(exc, reranker_client.RerankerServiceUnavailable)
                else safe_reranker_error(exc)
            )
            return RerankOutcome(
                candidates,
                {
                    "status": "unavailable",
                    "reason": reason,
                    "candidate_count": rerank_n,
                    "execution": "service",
                    "degraded": True,
                },
            )

    head = candidates[:rerank_n]
    started = time.perf_counter()
    try:
        from chronovisor.core.llm_config import load_default_llm_runtime

        runtime = load_default_llm_runtime()
        route = runtime.resolve_rerank(RERANK_RUNTIME_ROLE)
        store = get_store()
        try:
            store.refresh()
            candidate_store: IndexStore | None = store
        except Exception:
            candidate_store = None
        resolved = [
            resolve_rerank_candidate(page, store=candidate_store) for page in head
        ]
        result = runtime.rerank(
            RERANK_RUNTIME_ROLE,
            RerankRequest(
                query,
                tuple(passage for passage, _source, _identity in resolved),
                QUERY_SOURCE,
                candidate_sources=tuple(
                    source for _passage, source, _identity in resolved
                ),
            ),
        )
    except Exception as exc:
        return RerankOutcome(
            candidates,
            {
                "status": "unavailable",
                "reason": safe_reranker_error(exc),
                "candidate_count": rerank_n,
                "degraded": True,
            },
        )

    elapsed_ms = int(round((time.perf_counter() - started) * 1000))
    scores_by_index = sorted(result.items, key=lambda item: item.index)
    return apply_reranker_scores(
        candidates,
        [float(item.score) for item in scores_by_index],
        config=cfg,
        metadata={
            "latency_ms": elapsed_ms,
            "execution": "in_process",
            "route": _route_identity(route),
        },
    )
