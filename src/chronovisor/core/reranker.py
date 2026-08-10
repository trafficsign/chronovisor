"""Core cross-encoder reranking for interactive MCP search."""

from __future__ import annotations

import math
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

from chronovisor.core.llm_runtime import (
    RerankItem,
    RerankRequest,
    RerankResult,
    RouteLocation,
    SourceDataClass,
    SourceDataClassification,
    SourceSensitivity,
)
from chronovisor.core.runtime_config import RerankerConfig, load_reranker_config
from chronovisor.core.search_types import ScoredPage
from chronovisor.core.store import find_page

FRONTMATTER_RE = re.compile(r"^---\n.*?\n---\n?", re.DOTALL)
_MODEL_CACHE: dict[tuple[str, str, str], Any] = {}
_MODEL_LOCK = threading.RLock()
_WARMUP_LOCK = threading.Lock()
_WARMUP_THREAD: threading.Thread | None = None


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


def _candidate_text(page: ScoredPage, *, max_chars: int = 2400) -> str:
    path = find_page(page.page_id)
    body = ""
    if path is not None:
        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            content = ""
        body = FRONTMATTER_RE.sub("", content).strip()
    if page.snippet:
        body = f"{page.snippet}\n\n{body}"
    return f"{page.title}\n\n{body[:max_chars]}".strip()


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
    key = ("transformers", config.model, device)
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
            model.to(device)
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

    key = ("flagembedding", config.model, config.device)
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
            metadata={"backend": config.backend},
        )


# The compatibility path is local-only, so no source content can cross an egress
# boundary before consumers migrate to passing their own classification.
_LOCAL_RERANK_SOURCE = SourceDataClassification(
    SourceDataClass.PAGE,
    SourceSensitivity.NORMAL,
)


def score_fn(
    config: RerankerConfig,
) -> Callable[[str, list[str], RerankerConfig], list[float]]:
    """Compatibility callable backed by the provider-neutral local component."""

    backend = LocalRerankBackend(config)

    def score(
        query: str,
        passages: list[str],
        call_config: RerankerConfig,
    ) -> list[float]:
        active_backend = (
            backend if call_config == config else LocalRerankBackend(call_config)
        )
        result = active_backend.rerank(
            RerankRequest(
                query=query,
                candidates=tuple(passages),
                source=_LOCAL_RERANK_SOURCE,
            ),
            model=call_config.model,
        )
        return [
            item.score for item in sorted(result.items, key=lambda item: item.index)
        ]

    return score


def warm_reranker(config: RerankerConfig | None = None) -> dict[str, Any]:
    """Load and exercise the configured reranker before the first search."""

    cfg = config or load_reranker_config()
    if not cfg.enabled:
        return {"status": "disabled", "reason": "config_disabled"}
    started = time.perf_counter()
    try:
        scores = score_fn(cfg)("reranker warmup", ["reranker warmup"], cfg)
    except Exception as exc:
        return {
            "status": "unavailable",
            "reason": str(exc),
            "model": cfg.model,
            "backend": cfg.backend,
        }
    return {
        "status": "ready" if len(scores) == 1 else "unavailable",
        "model": cfg.model,
        "backend": cfg.backend,
        "latency_ms": int(round((time.perf_counter() - started) * 1000)),
    }


def start_reranker_warmup(
    config: RerankerConfig | None = None,
) -> threading.Thread | None:
    """Start one daemon warmup per MCP process without delaying startup."""

    cfg = config or load_reranker_config()
    if not cfg.enabled:
        return None
    global _WARMUP_THREAD
    with _WARMUP_LOCK:
        if _WARMUP_THREAD is not None and _WARMUP_THREAD.is_alive():
            return _WARMUP_THREAD
        _WARMUP_THREAD = threading.Thread(
            target=warm_reranker,
            args=(cfg,),
            name="chronovisor-reranker-warmup",
            daemon=True,
        )
        _WARMUP_THREAD.start()
        return _WARMUP_THREAD


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
        "model": config.model,
        "backend": config.backend,
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
                "reason": (
                    "reranker returned invalid score cardinality or "
                    "non-finite scores"
                ),
                "score_count": len(raw_scores),
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
                "model": cfg.model,
                "backend": cfg.backend,
            },
        )
    if not candidates:
        return RerankOutcome(
            candidates, {"status": "skipped", "reason": "no_candidates"}
        )

    rerank_n = min(max(1, cfg.top_n), len(candidates))
    head = candidates[:rerank_n]
    passages = [_candidate_text(page) for page in head]
    started = time.perf_counter()
    try:
        scores = score_fn(cfg)(query, passages, cfg)
    except Exception as exc:
        return RerankOutcome(
            candidates,
            {
                "status": "unavailable",
                "reason": str(exc),
                "model": cfg.model,
                "backend": cfg.backend,
                "candidate_count": rerank_n,
            },
        )

    elapsed_ms = int(round((time.perf_counter() - started) * 1000))
    return apply_reranker_scores(
        candidates,
        [float(score) for score in scores],
        config=cfg,
        metadata={"latency_ms": elapsed_ms, "execution": "in_process"},
    )
