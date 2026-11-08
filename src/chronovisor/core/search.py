"""Core search engine with BM25 and semantic RRF fusion."""

import contextlib
import os
import re
import threading
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

from chronovisor.core.lexical_index import LexicalIndex
from chronovisor.core.runtime_config import (
    load_search_embedding_config,
)
from chronovisor.core.search_types import ScoredPage, merge_evidence
from chronovisor.core.store import (
    CHRONOVISOR_ROOT,
    PAGES_DIR,
)


def searchable_pages() -> list[Path]:
    """Return stable canonical pages from the shared metadata snapshot."""

    from chronovisor.core.index_store import get_store

    store = get_store()
    _refresh_store_for_search(store)
    paths = []
    for page_id in store.all_page_ids(include_system=True):
        meta = store.meta(page_id)
        path = meta.get("path") if meta is not None else None
        if isinstance(path, str):
            paths.append(Path(path))
    return sorted(paths)


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------

_BM25_CACHE_FILE = CHRONOVISOR_ROOT / ".index" / "lexical.sqlite"
_LEGACY_BM25_CACHE_FILE = CHRONOVISOR_ROOT / ".index" / "bm25.json"
_STABLE_STATUS = "stable"
_VALID_LIFECYCLE_STATUSES = {"draft", _STABLE_STATUS, "deprecated"}
_REFERENCE_PAGE_TYPE = "reference"
_VALID_PAGE_TYPES = {
    "knowledge",
    _REFERENCE_PAGE_TYPE,
    "episodic",
    "semantic",
    "procedural",
    "state",
    "lesson",
    "decision",
}
_VALID_SENSITIVITY_TIERS = {"normal", "high"}


def _normalize_lifecycle_status(value: object) -> str:
    normalized = value.strip().lower() if isinstance(value, str) else ""
    if normalized in _VALID_LIFECYCLE_STATUSES:
        return normalized
    return "draft"


def _normalize_page_type(value: object, *, folder: str = "") -> str:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _VALID_PAGE_TYPES:
            return normalized
    if folder == "car-spec":
        return _REFERENCE_PAGE_TYPE
    return "knowledge"


def _normalize_sensitivity(value: object, *, folder: str = "") -> str:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _VALID_SENSITIVITY_TIERS:
            return normalized
    if folder == "career":
        return "high"
    return "normal"


def _is_stable_result(result: ScoredPage) -> bool:
    return _normalize_lifecycle_status(result.status) == _STABLE_STATUS


def _is_reference_result(result: ScoredPage) -> bool:
    return (
        _normalize_page_type(result.page_type, folder=result.folder)
        == _REFERENCE_PAGE_TYPE
    )


def _folder_from_meta(meta: dict[str, Any]) -> str:
    try:
        parent = Path(meta["path"]).parent
        if parent != PAGES_DIR:
            return parent.name
    except KeyError, TypeError:
        pass
    return ""


def _meta_page_type(meta: dict[str, Any], *, folder: str = "") -> str:
    return _normalize_page_type(meta.get("page_type"), folder=folder)


def _meta_sensitivity(meta: dict[str, Any], *, folder: str = "") -> str:
    return _normalize_sensitivity(meta.get("sensitivity"), folder=folder)


def _refresh_store_for_search(store: object) -> None:
    """Share a short read snapshot while preserving injected-store compatibility."""

    refresh_if_stale = getattr(store, "refresh_if_stale", None)
    if callable(refresh_if_stale):
        refresh_if_stale()
        return
    refresh = cast(Any, store).refresh
    refresh()


# ---------------------------------------------------------------------------
# BM25 singleton — shared across `search()` and `ingest._search_related_pages`
# ---------------------------------------------------------------------------


class BM25Index(LexicalIndex):
    """Compatibility name for the adopted SQLite inverted lexical index."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        # k1/b remain accepted so older debug/test callers do not break.
        del k1, b
        super().__init__(path=_BM25_CACHE_FILE, pages=searchable_pages)

    def build(self, *, force: bool = False) -> None:
        super().build(force=force)
        if os.environ.get("CHRONOVISOR_READ_ONLY") != "1":
            _LEGACY_BM25_CACHE_FILE.unlink(missing_ok=True)


def lexical_cache_paths() -> tuple[Path, ...]:
    return (
        _BM25_CACHE_FILE,
        Path(f"{_BM25_CACHE_FILE}-wal"),
        Path(f"{_BM25_CACHE_FILE}-shm"),
        _LEGACY_BM25_CACHE_FILE,
    )


_BM25_LOCK = threading.Lock()
_BM25_SINGLETON: BM25Index | None = None


def get_bm25() -> BM25Index:
    """Return the process-wide BM25Index instance.

    All callers go through this so that the disk-cache load + persisted
    state are paid at most once per process; subsequent `build()` calls
    are O(stat) when nothing changed.
    """
    global _BM25_SINGLETON
    if _BM25_SINGLETON is None:
        with _BM25_LOCK:
            if _BM25_SINGLETON is None:
                _BM25_SINGLETON = BM25Index()
    return _BM25_SINGLETON


# Semantic document projection shared by the immutable service index.
MAX_CHUNKS_PER_PAGE = 8
MAX_CHUNK_CHARS = 900


def _recall_questions(meta: dict[str, Any]) -> list[str]:
    questions = meta.get("recall_questions")
    if not isinstance(questions, list):
        return []
    return [q for q in questions if isinstance(q, str) and q.strip()][:8]


def _markdown_chunks(body: str, title: str, meta: dict[str, Any]) -> list[str]:
    chunks: list[str] = []
    heading = title
    buffer: list[str] = []

    summary_value = meta.get("description") or meta.get("summary")
    summary = summary_value if isinstance(summary_value, str) else ""
    entities = meta.get("entities") if isinstance(meta.get("entities"), list) else []
    page_type = meta.get("page_type") if isinstance(meta.get("page_type"), str) else ""
    updated = meta.get("updated") if isinstance(meta.get("updated"), str) else ""
    context_lines = [f"Page: {title}"]
    if summary.strip():
        context_lines.append(f"Summary: {summary.strip()}")
    if entities:
        context_lines.append(
            "Entities: " + ", ".join(str(item) for item in entities[:8])
        )
    if page_type or updated:
        context_lines.append(
            f"Metadata: type={page_type or 'knowledge'} updated={updated or 'unknown'}"
        )
    context = "\n".join(context_lines)

    def flush() -> None:
        nonlocal buffer
        text = re.sub(r"\s+", " ", "\n".join(buffer)).strip()
        buffer = []
        if not text:
            return
        prefix = (
            f"{context}\nHeading: {heading}"
            if heading and heading != title
            else context
        )
        while text:
            piece = text[:MAX_CHUNK_CHARS].strip()
            if len(text) > MAX_CHUNK_CHARS and " " in piece:
                piece = (
                    piece.rsplit(" ", 1)[0].strip() or text[:MAX_CHUNK_CHARS].strip()
                )
            chunks.append(f"{prefix}\n\n{piece}")
            text = text[len(piece) :].strip()
            if len(chunks) >= MAX_CHUNKS_PER_PAGE:
                return

    for line in body.splitlines():
        stripped = line.strip()
        heading_match = re.match(r"^(#{1,6})\s+(.+)$", stripped)
        if heading_match:
            flush()
            heading = heading_match.group(2).strip()
            if len(chunks) >= MAX_CHUNKS_PER_PAGE:
                break
            continue
        if not stripped:
            flush()
            if len(chunks) >= MAX_CHUNKS_PER_PAGE:
                break
            continue
        buffer.append(stripped)
        if sum(len(item) for item in buffer) >= MAX_CHUNK_CHARS:
            flush()
            if len(chunks) >= MAX_CHUNKS_PER_PAGE:
                break
    if len(chunks) < MAX_CHUNKS_PER_PAGE:
        flush()
    if not chunks and title.strip():
        chunks.append(title.strip())
    return chunks[:MAX_CHUNKS_PER_PAGE]


def update_embeddings(
    page_ids: list[str] | None = None, *, strict: bool = False
) -> int:
    """Refresh the configured semantic-service index.

    Nemotron writes are durable jobs.  Correctness-critical callers may use
    ``strict=True`` to wait for generation-scoped delta publication.
    """

    config = load_search_embedding_config()
    if not config.enabled:
        return 0
    from chronovisor.core.semantic_jobs import enqueue_pages, enqueue_rebuild

    if page_ids is None:
        enqueue_rebuild()
        return 0
    unique = sorted({page_id for page_id in page_ids if page_id})
    if strict:
        from chronovisor.core.semantic_client import index_pages

        response = index_pages(unique, config, wait=True)
        return int(response.get("pages_updated") or 0)

    from chronovisor.core.semantic_index import extract_page_documents
    from chronovisor.core.store import SYSTEM_DIR, find_page

    hashes: dict[str, str] = {}
    for page_id in unique:
        path = find_page(page_id)
        if path is None:
            system_path = SYSTEM_DIR / f"{page_id}.md"
            path = system_path if system_path.is_file() else None
        documents = extract_page_documents(path) if path is not None else []
        hashes[page_id] = documents[0].source_sha256 if documents else ""
    enqueue_pages(unique, source_hashes=hashes)
    return len(unique)


def semantic_search(
    query: str,
    top_n: int = 20,
    *,
    include_reference: bool = False,
    strict: bool = False,
    timeout_ms: int | None = None,
) -> list[ScoredPage]:
    """Search through the selected execution topology."""

    config = load_search_embedding_config()
    if not config.enabled:
        return []
    from chronovisor.core import semantic_client

    use_new = semantic_client.selected_for_rollout(query, config)
    if config.rollout_mode == "shadow":
        with contextlib.suppress(Exception):
            semantic_client.search(
                query,
                top_n,
                include_reference=include_reference,
                config=config,
                timeout_ms=timeout_ms,
            )
        return []
    if not use_new:
        return []
    return semantic_client.search(
        query,
        top_n,
        include_reference=include_reference,
        config=config,
        timeout_ms=timeout_ms,
    )


# ---------------------------------------------------------------------------
# RRF Fusion + Filter + Sort
# ---------------------------------------------------------------------------

DEFAULT_FUSION_WEIGHTS: dict[str, float] = {
    "bm25": 1.0,
    "semantic": 0.6,
    "bm25_score_bonus": 0.005,
    "bm25_rank_bonus": 0.006,
    "bm25_rank_decay": 0.006,
    "semantic_min_top_score": 0.45,
    "semantic_min_margin": 0.002,
    "semantic_low_confidence_weight": 0.25,
}


def _semantic_reliability_multiplier(
    semantic_results: list[ScoredPage],
    weights: dict[str, float],
) -> float:
    if not semantic_results:
        return 0.0
    min_top = max(0.0, float(weights.get("semantic_min_top_score", 0.0)))
    min_margin = max(0.0, float(weights.get("semantic_min_margin", 0.0)))
    low_weight = max(0.0, float(weights.get("semantic_low_confidence_weight", 1.0)))
    top1 = max(0.0, float(semantic_results[0].score))
    top2 = (
        max(0.0, float(semantic_results[1].score)) if len(semantic_results) > 1 else 0.0
    )
    if top1 < min_top or (top1 - top2) < min_margin:
        return low_weight
    return 1.0


def fuse_results(
    bm25_results: list[ScoredPage],
    semantic_results: list[ScoredPage],
    k: int = 60,
    weights: dict[str, float] | None = None,
) -> list[ScoredPage]:
    """Weighted Reciprocal Rank Fusion of result lists."""
    scores: dict[str, float] = {}
    meta: dict[str, ScoredPage] = {}
    weights = {**DEFAULT_FUSION_WEIGHTS, **(weights or {})}
    bm25_score_bonus = max(0.0, float(weights.get("bm25_score_bonus", 0.0)))
    bm25_rank_bonus = max(0.0, float(weights.get("bm25_rank_bonus", 0.0)))
    bm25_rank_decay = max(0.0, float(weights.get("bm25_rank_decay", 0.0)))
    semantic_multiplier = _semantic_reliability_multiplier(semantic_results, weights)

    def add_results(results: list[ScoredPage], channel: str) -> None:
        weight = max(0.0, float(weights.get(channel, 1.0)))
        if channel == "semantic":
            weight *= semantic_multiplier
        if weight == 0:
            return
        top_raw = max((float(page.score) for page in results), default=0.0)
        for rank, page in enumerate(results):
            score = weight / (k + rank)
            if channel == "bm25" and top_raw > 0:
                score += (
                    weight * bm25_score_bonus * (max(0.0, float(page.score)) / top_raw)
                )
                score += weight * max(0.0, bm25_rank_bonus - (bm25_rank_decay * rank))
            scores[page.page_id] = scores.get(page.page_id, 0) + score
            if page.page_id not in meta:
                meta[page.page_id] = page
            else:
                previous = meta[page.page_id]
                evidence = merge_evidence(previous.evidence, page.evidence)
                if previous.content_sha256:
                    evidence = tuple(
                        e
                        for e in evidence
                        if e.source_sha256 == previous.content_sha256
                    )
                if previous.uid:
                    evidence = tuple(e for e in evidence if e.page_uid == previous.uid)
                meta[page.page_id] = replace(previous, evidence=evidence)

    add_results(bm25_results, "bm25")
    add_results(semantic_results, "semantic")

    fused = []
    for pid, score in scores.items():
        p = meta[pid]
        fused.append(replace(p, score=score, snippet=""))

    fused.sort(key=lambda x: x.score, reverse=True)
    return fused


def apply_filters(
    results: list[ScoredPage],
    folder: str | None = None,
    updated_after: str | None = None,
    updated_before: str | None = None,
) -> list[ScoredPage]:
    """Filter results by folder and date range."""
    for result in results:
        result.status = _normalize_lifecycle_status(result.status)
    filtered = [r for r in results if _is_stable_result(r)]
    if not folder:
        filtered = [r for r in filtered if not _is_reference_result(r)]
    if folder:
        filtered = [r for r in filtered if r.folder.startswith(folder)]
    if updated_after:
        filtered = [
            r for r in filtered if r.updated >= updated_after and r.updated != "unknown"
        ]
    if updated_before:
        filtered = [
            r
            for r in filtered
            if r.updated <= updated_before and r.updated != "unknown"
        ]
    return filtered


def apply_sort(
    results: list[ScoredPage], sort_by: str = "relevance"
) -> list[ScoredPage]:
    """Sort results."""
    if sort_by == "date":
        return sorted(results, key=lambda x: x.updated, reverse=True)
    elif sort_by == "title":
        return sorted(results, key=lambda x: x.title)
    return results  # relevance is already sorted


# ---------------------------------------------------------------------------
# Public search API
# ---------------------------------------------------------------------------

_SEARCH_TRACE = threading.local()


def last_search_trace() -> dict[str, object]:
    return dict(getattr(_SEARCH_TRACE, "value", {}))


def search(
    query: str,
    top_n: int = 20,
    folder: str | None = None,
    updated_after: str | None = None,
    updated_before: str | None = None,
    sort_by: str = "relevance",
    semantic: bool = True,
    fusion_weights: dict[str, float] | None = None,
    semantic_timeout_ms: int | None = None,
) -> tuple[list[ScoredPage], str]:
    """Retrieve BM25 and semantic candidates with the existing weighted RRF."""
    weights = {**DEFAULT_FUSION_WEIGHTS, **(fusion_weights or {})}
    config = load_search_embedding_config()
    if fusion_weights is None and config.enabled:
        weights.update(
            semantic=config.fusion_weight,
            semantic_min_top_score=config.min_top_score,
            semantic_min_margin=config.min_margin,
            semantic_low_confidence_weight=config.low_confidence_weight,
        )
    fetch_n = max(top_n * 5, 100)
    index = get_bm25()
    index.build()
    lexical = index.query(query, top_n=fetch_n, include_reference=folder is not None)
    semantic_results = []
    if semantic:
        semantic_results = semantic_search(
            query,
            top_n=fetch_n,
            include_reference=folder is not None,
            timeout_ms=semantic_timeout_ms,
        )
    results = (
        fuse_results(lexical, semantic_results, weights=weights)
        if semantic_results
        else lexical
    )
    results = apply_filters(results, folder, updated_after, updated_before)
    _SEARCH_TRACE.value = {
        "channels": {
            "lexical": [p.page_id for p in lexical[:5]],
            "semantic": [p.page_id for p in semantic_results[:5]],
        },
    }
    return apply_sort(results, sort_by)[
        :top_n
    ], "hybrid" if semantic_results else "bm25"
