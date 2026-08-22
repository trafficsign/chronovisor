"""Core cross-encoder reranking for interactive MCP search."""

from __future__ import annotations

import hashlib
import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
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
    RerankItem,
    RerankRequest,
    RerankResult,
    ResolvedRerankRoute,
    RouteLocation,
    SourceDataClass,
    SourceDataClassification,
    SourceSensitivity,
)
from chronovisor.core.runtime_config import RerankerConfig, load_reranker_config
from chronovisor.core.search_types import ScoredPage
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
_MODEL_CACHE: dict[tuple[str, str, str, str], Any] = {}
_MODEL_LOCK = threading.RLock()


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
) -> tuple[
    str,
    SourceDataClassification,
    tuple[str, str, int, int, str],
]:
    """Resolve passage bytes and their egress classification from one index row."""

    page_id = candidate.page_id if isinstance(candidate, ScoredPage) else candidate
    title = candidate.title if isinstance(candidate, ScoredPage) else page_id
    snippet = candidate.snippet if isinstance(candidate, ScoredPage) else ""
    fallback = f"{title}\n\n{snippet}".strip() or page_id
    fallback_bytes = fallback.encode("utf-8")
    invalid = (
        fallback,
        _SYSTEM_SOURCE,
        (
            "invalid",
            "",
            0,
            len(fallback_bytes),
            hashlib.sha256(fallback_bytes).hexdigest(),
        ),
    )
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
    except (CanonicalDocumentError, OSError, RuntimeError, UnicodeDecodeError):
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


def _select_torch_device(torch_mod: Any, requested: str) -> str:
    if requested:
        return requested
    if (
        getattr(torch_mod.backends, "mps", None)
        and torch_mod.backends.mps.is_available()
    ):
        return "mps"
    if torch_mod.cuda.is_available():
        return "cuda"
    return "cpu"


def _torch_dtype(torch_mod: Any, requested: str) -> Any:
    dtype = {
        "float16": torch_mod.float16,
        "float32": torch_mod.float32,
        "bfloat16": torch_mod.bfloat16,
    }.get(requested)
    if dtype is None:
        raise RuntimeError("unsupported reranker dtype")
    return dtype


def _load_transformer_components(
    config: RerankerConfig,
    tokenizer_cls: Any,
    model_cls: Any,
) -> tuple[Any, Any]:
    """Prefer a complete local snapshot, then allow first-install download."""

    try:
        tokenizer = tokenizer_cls.from_pretrained(config.model, local_files_only=True)
        model = model_cls.from_pretrained(config.model, local_files_only=True)
    except (OSError, ValueError):
        tokenizer = tokenizer_cls.from_pretrained(config.model)
        model = model_cls.from_pretrained(config.model)
    return tokenizer, model


def _transformer_scores(
    query: str, passages: list[str], config: RerankerConfig
) -> list[float]:
    try:
        import torch  # type: ignore[import-not-found, unused-ignore]
        from transformers import (  # type: ignore[import-not-found, unused-ignore]
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )
    except Exception as exc:  # pragma: no cover - exercised via missing-dep tests
        raise RuntimeError(f"transformers backend unavailable: {exc}") from exc

    device = _select_torch_device(torch, config.device)
    key = ("transformers", config.model, device, config.dtype)
    # Hugging Face's default path performs remote metadata checks even when
    # all weights are cached. Prefer the complete local snapshot and only use
    # the network on the first installation of a model.
    with _MODEL_LOCK:
        cached = _MODEL_CACHE.get(key)
        if cached is None:
            tokenizer, model = _load_transformer_components(
                config,
                AutoTokenizer,
                AutoModelForSequenceClassification,
            )
            model.to(device=device, dtype=_torch_dtype(torch, config.dtype))
            model.eval()
            cached = (tokenizer, model)
            _MODEL_CACHE[key] = cached
        tokenizer, model = cached

        # Serialize MPS inference with startup warmup. Otherwise an immediate
        # first search can race the warmup thread on the same model instance.
        scores: list[float] = []
        batch_size = max(1, config.batch_size)
        with torch.no_grad():
            for i in range(0, len(passages), batch_size):
                batch = passages[i : i + batch_size]
                encoded = tokenizer(
                    [query] * len(batch),
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=max(1, config.max_length),
                    return_tensors="pt",
                )
                encoded = {name: value.to(device) for name, value in encoded.items()}
                logits = model(**encoded).logits
                if len(logits.shape) == 2 and logits.shape[1] > 1:
                    values = logits[:, -1]
                else:
                    values = logits.reshape(-1)
                scores.extend(
                    float(value) for value in values.detach().float().cpu().tolist()
                )
    return scores


def _flagembedding_scores(
    query: str, passages: list[str], config: RerankerConfig
) -> list[float]:
    try:
        from FlagEmbedding import FlagReranker  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - exercised via missing-dep tests
        raise RuntimeError(f"FlagEmbedding backend unavailable: {exc}") from exc

    key = ("flagembedding", config.model, config.device, "float16")
    reranker = _MODEL_CACHE.get(key)
    if reranker is None:
        kwargs: dict[str, Any] = {}
        if config.device:
            kwargs["devices"] = [config.device]
        reranker = FlagReranker(config.model, use_fp16=True, **kwargs)
        _MODEL_CACHE[key] = reranker
    pairs = [[query, passage] for passage in passages]
    try:
        raw_scores = reranker.compute_score(
            pairs,
            batch_size=max(1, config.batch_size),
            max_length=max(1, config.max_length),
        )
    except TypeError:
        raw_scores = reranker.compute_score(pairs)
    if isinstance(raw_scores, (float, int)):
        return [float(raw_scores)]
    return [float(score) for score in raw_scores]


def _score_impl(
    config: RerankerConfig,
) -> Callable[[str, list[str], RerankerConfig], list[float]]:
    backend = config.backend.lower()
    if backend == "transformers":
        return _transformer_scores
    if backend == "flagembedding":
        return _flagembedding_scores
    raise RuntimeError(f"unsupported reranker backend: {config.backend}")


class LocalRerankBackend:
    """Local Transformers/FlagEmbedding implementation of RerankBackend."""

    provider = "local-reranker"
    location = RouteLocation.LOCAL

    def __init__(self, config: RerankerConfig) -> None:
        self.config = config

    def rerank(self, request: RerankRequest, *, model: str) -> RerankResult:
        config = (
            self.config
            if model == self.config.model
            else replace(self.config, model=model)
        )
        scores = _score_impl(config)(request.query, list(request.candidates), config)
        items = sorted(
            (
                RerankItem(index=index, score=float(score))
                for index, score in enumerate(scores)
            ),
            key=lambda item: (-item.score, item.index),
        )
        return RerankResult(
            items=tuple(items),
            provider=self.provider,
            model=model,
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
        next_score = (
            reranked[index + 1][1] if index + 1 < len(reranked) else raw_score
        )
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
        (
            abs(original_rank[page.page_id] - rerank_rank[page.page_id])
            for page in head
        ),
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
