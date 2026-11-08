"""Core BM25 search engine."""

import os
import threading
from pathlib import Path
from typing import Any, cast

from chronovisor.core.lexical_index import LexicalIndex
from chronovisor.core.paths import (
    CHRONOVISOR_ROOT,
    PAGES_DIR,
)
from chronovisor.core.search_types import ScoredPage


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


# ---------------------------------------------------------------------------
# Filter + Sort
# ---------------------------------------------------------------------------

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
) -> tuple[list[ScoredPage], str]:
    """Retrieve BM25 candidates, then apply filters and sorting."""
    fetch_n = max(top_n * 5, 100)
    index = get_bm25()
    index.build()
    lexical = index.query(query, top_n=fetch_n, include_reference=folder is not None)
    results = apply_filters(lexical, folder, updated_after, updated_before)
    _SEARCH_TRACE.value = {
        "channels": {"lexical": [p.page_id for p in lexical[:5]]},
    }
    return apply_sort(results, sort_by)[:top_n], "bm25"
