"""Precision-safe contextual Anti-Index and hub suppression diagnostics."""

from __future__ import annotations

import math
from typing import Any

from chronovisor.core.index_store import get_store
from chronovisor.core.runtime_config import load_negative_feedback_config
from chronovisor.core.search_types import tokenize
from chronovisor.search.negative_feedback import contextual_negative_trace


def _coverage(query_tokens: set[str], text: str) -> float:
    if not query_tokens or not text.strip():
        return 0.0
    span_tokens = set(tokenize(text))
    return len(query_tokens & span_tokens) / max(1, min(12, len(query_tokens)))


def _exact_match(query: str, candidate: Any) -> bool:
    folded = query.casefold()
    page_id = str(getattr(candidate, "page_id", "") or "")
    title = str(getattr(candidate, "title", "") or "")
    page_key = page_id.replace("-", " ").replace("_", " ").casefold()
    title_key = title.strip().casefold()
    return bool(
        (len(page_key) >= 5 and page_key in folded)
        or (len(title_key) >= 5 and title_key in folded)
    )


def ranking_components(
    query: str,
    candidates: list[Any],
    *,
    store: Any | None = None,
) -> dict[str, dict[str, float | str | bool]]:
    """Return explainable components without mutating production ranking.

    Hub suppression is contextual: high degree alone is never enough. Exact
    page/title matches are protected, while vague graph-only hits without a
    supporting span receive the largest shadow penalty.
    """

    config = load_negative_feedback_config()
    anti_trace = contextual_negative_trace(query, config)
    query_tokens = set(tokenize(query))
    specificity = min(1.0, len(query_tokens) / 10.0)
    index = store or get_store()
    refresh = getattr(index, "refresh_if_stale", None)
    if callable(refresh):
        refresh()
    output: dict[str, dict[str, float | str | bool]] = {}
    for candidate in candidates:
        page_id = str(getattr(candidate, "page_id", "") or "")
        if not page_id:
            continue
        exact = _exact_match(query, candidate)
        snippet = str(getattr(candidate, "snippet", "") or "")
        span_coverage = _coverage(query_tokens, snippet)
        try:
            out_degree = len(index.outlinks(page_id))
            in_degree = len(index.backlinks(page_id))
        except Exception:
            out_degree = 0
            in_degree = 0
        degree = out_degree + in_degree
        hubness = min(1.0, math.log1p(degree) / math.log(101.0))
        support_discount = 1.0 - min(0.85, span_coverage)
        specificity_discount = 1.0 - (0.55 * specificity)
        hub_penalty = (
            0.0
            if exact
            else min(
                0.45,
                0.45 * hubness * support_discount * specificity_discount,
            )
        )
        anti_row = anti_trace.get(page_id, {})
        output[page_id] = {
            "anti_index": round(float(anti_row.get("penalty") or 0.0), 6),
            "hub_penalty": round(hub_penalty, 6),
            "hub_degree": float(degree),
            "query_specificity": round(specificity, 6),
            "support_coverage": round(span_coverage, 6),
            "exact_match_protected": exact,
            "mode": "shadow",
        }
    return output
