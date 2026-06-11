"""Query-conditioned negative feedback penalties for search ranking.

Recall feedback already records pages that were injected for a prompt and then
ignored (``injection_ignored``) or explicitly flagged (``false-positive``).
The recall hook consumes those entries as injection suppressions, but the
search ranking itself never learned from them, so the same pages keep
surfacing in the top-20 for the same kind of vague prompts.

This module turns those feedback entries into fusion-stage penalties: when an
incoming query is lexically similar to a feedback prompt, the pages recorded
on that entry are demoted (never removed) in the fused ranking. Similarity is
Jaccard over the shared search tokenizer, so the penalty stays conditioned on
the query and does not suppress a page globally.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

from llm_wiki_mcp.runtime_config import NegativeFeedbackConfig, load_negative_feedback_config

# Test seams: when set, bypass the recall_runtime/golden default paths.
FEEDBACK_FILE_OVERRIDE: Path | None = None
GOLDEN_FILE_OVERRIDE: Path | None = None


@dataclass(frozen=True)
class _FeedbackEntry:
    tokens: frozenset[str]
    pages: tuple[str, ...]


@dataclass
class _Cache:
    key: tuple | None = None
    entries: list[_FeedbackEntry] = field(default_factory=list)


_CACHE = _Cache()
_PROTECT_CACHE = _Cache()
_CACHE_LOCK = threading.Lock()


def _feedback_file() -> Path:
    if FEEDBACK_FILE_OVERRIDE is not None:
        return FEEDBACK_FILE_OVERRIDE
    from llm_wiki_mcp.recall_runtime import RECALL_FEEDBACK_FILE

    return RECALL_FEEDBACK_FILE


def _golden_file() -> Path:
    if GOLDEN_FILE_OVERRIDE is not None:
        return GOLDEN_FILE_OVERRIDE
    from llm_wiki_mcp.recall_runtime import RECALL_DIR

    return RECALL_DIR / "search-golden.jsonl"


def _tokenize(text: str) -> frozenset[str]:
    from llm_wiki_mcp.search import tokenize

    return frozenset(tokenize(text))


def _parse_ts(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _load_entries(config: NegativeFeedbackConfig) -> list[_FeedbackEntry]:
    path = _feedback_file()
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return []
    cache_key = (str(path), mtime, config.kinds, config.max_age_days, config.max_entries)
    with _CACHE_LOCK:
        if _CACHE.key == cache_key:
            return _CACHE.entries

    cutoff = (
        datetime.now() - timedelta(days=config.max_age_days)
        if config.max_age_days > 0
        else None
    )
    entries: list[_FeedbackEntry] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    for line in lines:
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict) or row.get("kind") not in config.kinds:
            continue
        prompt = row.get("prompt")
        pages = row.get("expected_pages")
        if not isinstance(prompt, str) or not prompt.strip():
            continue
        if not isinstance(pages, list):
            continue
        page_ids = tuple(p for p in pages if isinstance(p, str) and p)
        if not page_ids:
            continue
        if cutoff is not None:
            ts = _parse_ts(row.get("ts"))
            if ts is not None and ts < cutoff:
                continue
        tokens = _tokenize(prompt)
        if not tokens:
            continue
        entries.append(_FeedbackEntry(tokens=tokens, pages=page_ids))

    # Keep the most recent entries when the file grows large.
    if len(entries) > config.max_entries:
        entries = entries[-config.max_entries:]

    with _CACHE_LOCK:
        _CACHE.key = cache_key
        _CACHE.entries = entries
    return entries


def _load_protections() -> list[_FeedbackEntry]:
    """Reviewed positive labels veto auto-penalties for similar queries.

    ``injection_ignored`` only means "the assistant did not use the page",
    not "the page is irrelevant".  When a reviewer has since confirmed the
    page as relevant for a similar query, demoting it would reintroduce the
    very recall miss the review fixed, so those pages are protected.
    """
    path = _golden_file()
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return []
    cache_key = (str(path), mtime)
    with _CACHE_LOCK:
        if _PROTECT_CACHE.key == cache_key:
            return _PROTECT_CACHE.entries

    entries: list[_FeedbackEntry] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    for line in lines:
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict) or row.get("reviewed") is not True:
            continue
        query = row.get("query")
        pages = row.get("expected_pages")
        if not isinstance(query, str) or not query.strip() or not isinstance(pages, list):
            continue
        page_ids = tuple(p for p in pages if isinstance(p, str) and p)
        if not page_ids:
            continue
        tokens = _tokenize(query)
        if not tokens:
            continue
        entries.append(_FeedbackEntry(tokens=tokens, pages=page_ids))

    with _CACHE_LOCK:
        _PROTECT_CACHE.key = cache_key
        _PROTECT_CACHE.entries = entries
    return entries


def penalties_for_query(
    query: str,
    config: NegativeFeedbackConfig | None = None,
) -> dict[str, float]:
    """Return ``{page_id: penalty}`` with penalty in [0, config.penalty]."""
    config = config or load_negative_feedback_config()
    if not config.enabled:
        return {}
    query_tokens = _tokenize(query)
    if not query_tokens:
        return {}
    penalties: dict[str, float] = {}
    for entry in _load_entries(config):
        union = len(query_tokens | entry.tokens)
        if union == 0:
            continue
        jaccard = len(query_tokens & entry.tokens) / union
        if jaccard < config.similarity_threshold:
            continue
        page_penalty = config.penalty * jaccard
        for page_id in entry.pages:
            if page_penalty > penalties.get(page_id, 0.0):
                penalties[page_id] = page_penalty
    if not penalties:
        return {}

    protected: dict[str, float] = {}
    for entry in _load_protections():
        if not set(entry.pages) & set(penalties):
            continue
        union = len(query_tokens | entry.tokens)
        if union == 0:
            continue
        jaccard = len(query_tokens & entry.tokens) / union
        if jaccard < config.similarity_threshold:
            continue
        for page_id in entry.pages:
            if jaccard > protected.get(page_id, 0.0):
                protected[page_id] = jaccard
    for page_id, pos_jaccard in protected.items():
        penalty = penalties.get(page_id)
        if penalty is not None and pos_jaccard * config.penalty >= penalty:
            del penalties[page_id]
    return penalties


def apply_penalties(results, penalties: dict[str, float]):
    """Demote penalized pages by scaling their fused score, then re-sort."""
    if not penalties:
        return results
    from llm_wiki_mcp.search import ScoredPage

    adjusted = []
    for page in results:
        penalty = penalties.get(page.page_id, 0.0)
        if penalty <= 0.0:
            adjusted.append(page)
            continue
        adjusted.append(
            ScoredPage(
                page_id=page.page_id,
                title=page.title,
                folder=page.folder,
                updated=page.updated,
                score=float(page.score) * max(0.0, 1.0 - penalty),
                status=page.status,
                superseded_by=page.superseded_by,
            )
        )
    adjusted.sort(key=lambda x: x.score, reverse=True)
    return adjusted
