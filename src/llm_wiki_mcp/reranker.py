"""Optional cross-encoder reranking for interactive MCP search."""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, replace
from typing import Any, Callable

from llm_wiki_mcp.runtime_config import RerankerConfig, load_reranker_config
from llm_wiki_mcp.search_types import ScoredPage
from llm_wiki_mcp.wiki import find_page

_FRONTMATTER_RE = re.compile(r"^---\n.*?\n---\n?", re.DOTALL)
_MODEL_CACHE: dict[tuple[str, str, str], Any] = {}


@dataclass(frozen=True)
class RerankOutcome:
    results: list[ScoredPage]
    metadata: dict[str, Any]


def _candidate_text(page: ScoredPage, *, max_chars: int = 2400) -> str:
    path = find_page(page.page_id)
    body = ""
    if path is not None:
        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            content = ""
        body = _FRONTMATTER_RE.sub("", content).strip()
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


def _transformer_scores(
    query: str, passages: list[str], config: RerankerConfig
) -> list[float]:
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except Exception as exc:  # pragma: no cover - exercised via missing-dep tests
        raise RuntimeError(f"transformers backend unavailable: {exc}") from exc

    device = _select_torch_device(torch, config.device)
    key = ("transformers", config.model, device)
    cached = _MODEL_CACHE.get(key)
    if cached is None:
        tokenizer = AutoTokenizer.from_pretrained(config.model)
        model = AutoModelForSequenceClassification.from_pretrained(config.model)
        model.to(device)
        model.eval()
        cached = (tokenizer, model)
        _MODEL_CACHE[key] = cached
    tokenizer, model = cached

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
        from FlagEmbedding import FlagReranker
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


def _score_fn(
    config: RerankerConfig,
) -> Callable[[str, list[str], RerankerConfig], list[float]]:
    backend = config.backend.lower()
    if backend == "transformers":
        return _transformer_scores
    if backend == "flagembedding":
        return _flagembedding_scores
    raise RuntimeError(f"unsupported reranker backend: {config.backend}")


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
    tail = candidates[rerank_n:]
    passages = [_candidate_text(page) for page in head]
    started = time.perf_counter()
    try:
        scores = _score_fn(cfg)(query, passages, cfg)
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

    if len(scores) != rerank_n or any(
        not math.isfinite(float(score)) for score in scores
    ):
        return RerankOutcome(
            candidates,
            {
                "status": "unavailable",
                "reason": "reranker returned invalid score cardinality or non-finite scores",
                "model": cfg.model,
                "backend": cfg.backend,
                "candidate_count": rerank_n,
                "score_count": len(scores),
            },
        )

    original_rank = {page.page_id: rank for rank, page in enumerate(head, start=1)}
    reranked = sorted(
        zip(head, scores),
        key=lambda item: (-float(item[1]), original_rank[item[0].page_id]),
    )
    rerank_rank = {
        page.page_id: rank for rank, (page, _score) in enumerate(reranked, start=1)
    }

    rescored = []
    for page in head:
        original_score = 1.0 / (60 + original_rank[page.page_id])
        rerank_score = 1.0 / (60 + rerank_rank[page.page_id])
        blended_score = original_score + (max(0.0, cfg.weight) * rerank_score)
        rescored.append(replace(page, score=blended_score))
    rescored.sort(key=lambda page: page.score, reverse=True)
    elapsed_ms = int(round((time.perf_counter() - started) * 1000))
    moved = sum(
        original_rank[page.page_id] != rerank_rank[page.page_id] for page in head
    )
    max_rank_delta = max(
        (abs(original_rank[page.page_id] - rerank_rank[page.page_id]) for page in head),
        default=0,
    )
    return RerankOutcome(
        rescored + tail,
        {
            "status": "applied",
            "model": cfg.model,
            "backend": cfg.backend,
            "candidate_count": rerank_n,
            "weight": cfg.weight,
            "latency_ms": elapsed_ms,
            "moved_candidates": moved,
            "max_rank_delta": max_rank_delta,
        },
    )
