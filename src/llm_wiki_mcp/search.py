"""Search engine — BM25 + semantic search with RRF fusion."""

import json
import math
import re
import threading
from dataclasses import dataclass
from pathlib import Path

from llm_wiki_mcp.wiki import WIKI_ROOT, PAGES_DIR, all_pages, page_id_from_path
from llm_wiki_mcp.link_fix import atomic_write


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class ScoredPage:
    page_id: str
    title: str
    folder: str
    updated: str
    score: float
    snippet: str = ""


# ---------------------------------------------------------------------------
# Tokenizer (no external dependency)
# ---------------------------------------------------------------------------

_CJK_RANGES = (
    ("\u3040", "\u309f"),  # Hiragana
    ("\u30a0", "\u30ff"),  # Katakana
    ("\u4e00", "\u9fff"),  # CJK Unified Ideographs
    ("\u3400", "\u4dbf"),  # CJK Extension A
    ("\uff66", "\uff9f"),  # Halfwidth Katakana
)

_FRONTMATTER_RE = re.compile(r"^---\n.*?\n---\n", re.DOTALL)


def _is_cjk(ch: str) -> bool:
    for lo, hi in _CJK_RANGES:
        if lo <= ch <= hi:
            return True
    return False


def tokenize(text: str) -> list[str]:
    """Tokenize text: ASCII words + CJK character bigrams (boundary-aware)."""
    # Strip frontmatter
    text = _FRONTMATTER_RE.sub("", text)
    text_lower = text.lower()

    tokens = []
    # ASCII words
    for m in re.finditer(r"[a-z0-9_]+", text_lower):
        word = m.group()
        if len(word) >= 2:
            tokens.append(word)

    # CJK bigrams — boundary-aware (don't cross non-CJK gaps)
    cjk_runs = re.findall(r"[" + "".join(f"{lo}-{hi}" for lo, hi in _CJK_RANGES) + r"]+", text)
    for run in cjk_runs:
        # Single CJK character: include as unigram
        if len(run) == 1:
            tokens.append(run)
        # Bigrams within each contiguous CJK run
        for i in range(len(run) - 1):
            tokens.append(run[i] + run[i + 1])

    return tokens


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------

_BM25_CACHE_FILE = WIKI_ROOT / ".index" / "bm25.json"
_BM25_CACHE_SCHEMA = 1


class BM25Index:
    """BM25 index with persistent per-page caching.

    Per-page tokenization output (`tf_map`, `doc_len`, frontmatter
    fields) is cached on disk keyed by `(mtime_ns, size)`. On subsequent
    builds, only added/changed/removed pages touch the tokenizer; the
    global `df` table is maintained incrementally and `avgdl` is
    recomputed in O(N) over cached `doc_len` values.

    Query-time scoring reads `tf_map` directly instead of recounting
    tokens from a stored token list, eliminating the per-query
    O(total_tokens) scan that the previous in-memory build incurred.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._df: dict[str, int] = {}
        self._avgdl: float = 0.0
        self._n: int = 0
        # `_cache` doubles as the on-disk persistent state and the in-memory
        # query view — query iterates `_cache.items()` directly.
        self._cache: dict[str, dict] = {}
        self._cache_loaded: bool = False

    # -- persistence ------------------------------------------------------

    def _load_cache(self) -> None:
        if not _BM25_CACHE_FILE.exists():
            return
        try:
            doc = json.loads(_BM25_CACHE_FILE.read_text())
        except (OSError, json.JSONDecodeError):
            return
        if doc.get("schema_version") != _BM25_CACHE_SCHEMA:
            return
        # Parameter change invalidates the cache (`tf` weights stay the
        # same but `df`/`avgdl` are tied to the corpus, and we want to
        # rebuild defensively if k1/b were tweaked between runs).
        if doc.get("k1") != self.k1 or doc.get("b") != self.b:
            return
        try:
            self._cache = doc.get("docs", {})
            global_state = doc.get("global", {})
            self._df = dict(global_state.get("df", {}))
            self._n = int(global_state.get("n", 0))
            self._avgdl = float(global_state.get("avgdl", 0.0))
        except (KeyError, TypeError, ValueError):
            self._cache = {}
            self._df = {}
            self._n = 0
            self._avgdl = 0.0

    def _persist_cache(self) -> None:
        _BM25_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        doc = {
            "schema_version": _BM25_CACHE_SCHEMA,
            "k1": self.k1,
            "b": self.b,
            "global": {
                "n": self._n,
                "avgdl": self._avgdl,
                "df": self._df,
            },
            "docs": self._cache,
        }
        atomic_write(_BM25_CACHE_FILE, json.dumps(doc, ensure_ascii=False))

    # -- build ------------------------------------------------------------

    def build(self) -> None:
        """Sync cache with disk and rebuild the in-memory query view.

        Cheap on warm runs: O(N) stat calls + zero parsing if nothing
        changed. Mutated pages are re-tokenized and the `df` table is
        updated incrementally (subtract old contributions, add new).
        """
        if not self._cache_loaded:
            self._load_cache()
            self._cache_loaded = True

        # Snapshot disk state.
        current: dict[str, tuple[Path, int, int]] = {}
        for path in all_pages():
            try:
                st = path.stat()
            except OSError:
                continue
            pid = page_id_from_path(path)
            current[pid] = (path, st.st_mtime_ns, st.st_size)

        old_ids = set(self._cache.keys())
        new_ids = set(current.keys())
        removed = old_ids - new_ids
        changed = False

        # Removed pages: subtract from df
        for pid in removed:
            old = self._cache.pop(pid)
            self._subtract_from_df(old.get("tf_map", {}))
            changed = True

        # New / modified pages
        for pid, (path, mtime_ns, size) in current.items():
            old = self._cache.get(pid)
            if old and old.get("mtime_ns") == mtime_ns and old.get("size") == size:
                continue  # unchanged
            if old:
                self._subtract_from_df(old.get("tf_map", {}))
            try:
                content = path.read_text()
            except (OSError, UnicodeDecodeError):
                # Drop the cache entry entirely so we don't keep stale stats.
                self._cache.pop(pid, None)
                changed = True
                continue
            fm_match = re.search(r"title:\s*(.+)", content)
            title = fm_match.group(1).strip() if fm_match else pid
            updated_match = re.search(r"updated:\s*(.+)", content)
            updated = updated_match.group(1).strip() if updated_match else ""
            folder = path.parent.name if path.parent != PAGES_DIR else ""

            title_tokens = tokenize(title) * 3
            body_tokens = tokenize(content)
            tokens = title_tokens + body_tokens
            tf_map: dict[str, int] = {}
            for tok in tokens:
                tf_map[tok] = tf_map.get(tok, 0) + 1
            doc_len = len(tokens)

            self._add_to_df(tf_map)

            self._cache[pid] = {
                "mtime_ns": mtime_ns,
                "size": size,
                "title": title,
                "folder": folder,
                "updated": updated,
                "doc_len": doc_len,
                "tf_map": tf_map,
            }
            changed = True

        # Recompute globals only if the corpus changed; otherwise the
        # n/avgdl loaded from the cache are still authoritative.
        if changed:
            self._n = len(self._cache)
            total_len = sum(d.get("doc_len", 0) for d in self._cache.values())
            self._avgdl = total_len / self._n if self._n else 1.0
            try:
                self._persist_cache()
            except OSError:
                pass

    def _subtract_from_df(self, tf_map: dict) -> None:
        for tok in tf_map.keys():
            current = self._df.get(tok, 0)
            if current <= 1:
                self._df.pop(tok, None)
            else:
                self._df[tok] = current - 1

    def _add_to_df(self, tf_map: dict) -> None:
        for tok in tf_map.keys():
            self._df[tok] = self._df.get(tok, 0) + 1

    # -- query ------------------------------------------------------------

    def query(self, query_text: str, top_n: int = 20) -> list[ScoredPage]:
        """Search the index."""
        if not self._cache:
            self.build()

        q_tokens = tokenize(query_text)
        if not q_tokens:
            return []

        # Hoist hot constants out of the inner loop.
        n = self._n
        avgdl = self._avgdl
        k1 = self.k1
        b = self.b
        df = self._df

        results = []
        for pid, doc in self._cache.items():
            tf_map = doc["tf_map"]
            dl = doc["doc_len"]
            score = 0.0
            for qt in q_tokens:
                tf = tf_map.get(qt, 0)
                if tf == 0:
                    continue
                d = df.get(qt, 0)
                idf = math.log((n - d + 0.5) / (d + 0.5) + 1)
                tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avgdl))
                score += idf * tf_norm

            if score > 0:
                results.append(ScoredPage(
                    page_id=pid,
                    title=doc["title"],
                    folder=doc["folder"],
                    updated=doc["updated"],
                    score=score,
                ))

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_n]


# ---------------------------------------------------------------------------
# BM25 singleton — shared across `search()` and `ingest._search_related_pages`
# ---------------------------------------------------------------------------

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
# Semantic search (Ollama embeddings) — SQLite-backed, incremental writes
# ---------------------------------------------------------------------------
#
# Storage model:
#   ~/.wiki/.index/embeddings.sqlite has one table:
#     embeddings(page_id PK, vector BLOB, mtime REAL, norm REAL, dim INT)
#   `vector` is a packed float32 array (4 bytes per dim → ~3KB per 768-dim
#   vector). `norm` is precomputed at write time so semantic_search never
#   recomputes per-row norms at query time.
#
# Migration: a one-shot import from the legacy ~/.wiki/.embeddings.json runs
# on first connect when the SQLite file does not yet exist.

import sqlite3
import struct

EMBED_MODEL = "nomic-embed-text"

EMBEDDINGS_DB = WIKI_ROOT / ".index" / "embeddings.sqlite"
LEGACY_EMBEDDINGS_FILE = WIKI_ROOT / ".embeddings.json"
EMBEDDINGS_FILE = LEGACY_EMBEDDINGS_FILE  # back-compat alias for any external imports

_EMBED_DB_LOCK = threading.Lock()


def _pack_vector(vec: list[float]) -> bytes:
    # Use 8-byte doubles to preserve full Python-float precision; the
    # original JSON store kept doubles, so cosine math must round-trip
    # bit-for-bit to keep semantic_search results identical.
    return struct.pack(f"<{len(vec)}d", *vec)


def _unpack_vector(blob: bytes, dim: int) -> list[float]:
    return list(struct.unpack(f"<{dim}d", blob))


def _vec_norm(vec: list[float]) -> float:
    return math.sqrt(sum(x * x for x in vec))


def _connect_embeddings() -> sqlite3.Connection:
    EMBEDDINGS_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(EMBEDDINGS_DB)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS embeddings (
            page_id TEXT PRIMARY KEY,
            vector BLOB NOT NULL,
            mtime REAL NOT NULL,
            norm REAL NOT NULL,
            dim INTEGER NOT NULL
        )
        """
    )
    return conn


_legacy_migration_done = False


def _maybe_migrate_legacy_json() -> None:
    """One-shot import of ~/.wiki/.embeddings.json into SQLite.

    Runs at most once per process. The legacy file is left in place so a
    rollback to the old code path remains possible; subsequent runs are
    no-ops because the SQLite table is already populated.
    """
    global _legacy_migration_done
    if _legacy_migration_done:
        return
    with _EMBED_DB_LOCK:
        if _legacy_migration_done:
            return
        if not LEGACY_EMBEDDINGS_FILE.exists():
            _legacy_migration_done = True
            return
        try:
            payload = json.loads(LEGACY_EMBEDDINGS_FILE.read_text())
        except (OSError, json.JSONDecodeError):
            _legacy_migration_done = True
            return
        if not isinstance(payload, dict) or not payload:
            _legacy_migration_done = True
            return
        conn = _connect_embeddings()
        try:
            existing = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
            if existing > 0:
                _legacy_migration_done = True
                return
            rows = []
            for pid, data in payload.items():
                vec = data.get("vector") if isinstance(data, dict) else None
                if not vec:
                    continue
                mtime = float(data.get("mtime", 0.0))
                norm = _vec_norm(vec)
                rows.append((pid, _pack_vector(vec), mtime, norm, len(vec)))
            if rows:
                conn.executemany(
                    "INSERT OR REPLACE INTO embeddings VALUES (?, ?, ?, ?, ?)",
                    rows,
                )
                conn.commit()
        finally:
            conn.close()
        _legacy_migration_done = True


def _load_embedding(pid: str) -> tuple[list[float], float, float] | None:
    """Return (vector, mtime, norm) for a single page, or None."""
    _maybe_migrate_legacy_json()
    conn = _connect_embeddings()
    try:
        row = conn.execute(
            "SELECT vector, mtime, norm, dim FROM embeddings WHERE page_id = ?",
            (pid,),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    blob, mtime, norm, dim = row
    return _unpack_vector(blob, int(dim)), float(mtime), float(norm)


def _store_embeddings_batch(rows: list[tuple[str, list[float], float]]) -> None:
    """Insert/replace a batch of (page_id, vector, mtime) rows."""
    if not rows:
        return
    packed = [
        (pid, _pack_vector(vec), float(mtime), _vec_norm(vec), len(vec))
        for pid, vec, mtime in rows
    ]
    with _EMBED_DB_LOCK:
        conn = _connect_embeddings()
        try:
            conn.executemany(
                "INSERT OR REPLACE INTO embeddings VALUES (?, ?, ?, ?, ?)",
                packed,
            )
            conn.commit()
        finally:
            conn.close()


def _iter_all_embeddings() -> "list[tuple[str, list[float], float, float]]":
    """Snapshot all rows as (page_id, vector, mtime, norm)."""
    _maybe_migrate_legacy_json()
    conn = _connect_embeddings()
    try:
        rows = conn.execute(
            "SELECT page_id, vector, mtime, norm, dim FROM embeddings"
        ).fetchall()
    finally:
        conn.close()
    out = []
    for pid, blob, mtime, norm, dim in rows:
        out.append((pid, _unpack_vector(blob, int(dim)), float(mtime), float(norm)))
    return out


def _embedding_count() -> int:
    _maybe_migrate_legacy_json()
    conn = _connect_embeddings()
    try:
        return conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
    finally:
        conn.close()


def _cosine_sim(a: list[float], b: list[float]) -> float:
    """Public-style helper kept for test/debug use; semantic_search uses
    a faster path with precomputed norms.
    """
    dot = sum(x * y for x, y in zip(a, b))
    na = _vec_norm(a)
    nb = _vec_norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def update_embeddings(page_ids: list[str] | None = None) -> int:
    """Update embeddings for pages. Returns count of updated pages.

    Writes are scoped to the rows that actually changed (or the rows
    explicitly requested via `page_ids`). Unchanged pages are not
    re-encoded and the SQLite table is not rewritten in full.
    """
    from llm_wiki_mcp.ollama import embed, is_available

    if not is_available():
        return 0

    _maybe_migrate_legacy_json()

    # Pull existing mtimes for the candidate page set in one query so
    # we don't pay per-row SELECTs inside the loop.
    conn = _connect_embeddings()
    try:
        existing_mtimes: dict[str, float] = {
            row[0]: float(row[1])
            for row in conn.execute("SELECT page_id, mtime FROM embeddings").fetchall()
        }
    finally:
        conn.close()

    pages_to_process: list[tuple[str, str, float]] = []
    for path in all_pages():
        pid = page_id_from_path(path)
        if page_ids and pid not in page_ids:
            continue

        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue

        existing_mtime = existing_mtimes.get(pid)
        if (
            existing_mtime is not None
            and existing_mtime >= mtime
            and not page_ids
        ):
            continue

        try:
            content = path.read_text()
        except (OSError, UnicodeDecodeError):
            continue

        fm_match = re.search(r"title:\s*(.+)", content)
        title = fm_match.group(1).strip() if fm_match else pid
        embed_text = f"{title}\n\n{_FRONTMATTER_RE.sub('', content)[:2000]}"
        pages_to_process.append((pid, embed_text, mtime))

    updated_count = 0
    if pages_to_process:
        batch_size = 32
        for i in range(0, len(pages_to_process), batch_size):
            batch = pages_to_process[i:i + batch_size]
            texts = [t[1] for t in batch]
            try:
                vectors = embed(texts)
            except Exception:
                continue
            rows: list[tuple[str, list[float], float]] = []
            for (pid, _, mtime), vec in zip(batch, vectors):
                rows.append((pid, vec, mtime))
            _store_embeddings_batch(rows)
            updated_count += len(rows)

    return updated_count


def semantic_search(query: str, top_n: int = 20) -> list[ScoredPage]:
    """Search using embedding similarity.

    Page metadata is read from the IndexStore (no per-page disk reads),
    each stored vector carries a precomputed `norm`, and the query norm
    is computed once. Inner loop is therefore one dot product per page.
    """
    from llm_wiki_mcp.ollama import embed, is_available
    from llm_wiki_mcp.index_store import get_store

    if not is_available():
        return []

    _maybe_migrate_legacy_json()

    if _embedding_count() == 0:
        return []

    try:
        q_vec = embed([query])[0]
    except Exception:
        return []

    q_norm = _vec_norm(q_vec)
    if q_norm == 0:
        return []

    store = get_store()
    store.refresh()

    results = []
    for pid, vec, _mtime, norm in _iter_all_embeddings():
        if norm == 0:
            continue
        meta = store.meta(pid)
        if meta is None:
            continue
        # Recover folder from the stored path (parent dir name relative
        # to PAGES_DIR), preserving legacy ScoredPage semantics.
        folder = ""
        try:
            parent = Path(meta["path"]).parent
            if parent != PAGES_DIR:
                folder = parent.name
        except (KeyError, TypeError):
            folder = ""

        dot = 0.0
        for x, y in zip(q_vec, vec):
            dot += x * y
        sim = dot / (q_norm * norm)
        results.append(ScoredPage(
            page_id=pid,
            title=meta["title"],
            folder=folder,
            updated=meta["updated"],
            score=sim,
        ))

    results.sort(key=lambda x: x.score, reverse=True)
    return results[:top_n]


# ---------------------------------------------------------------------------
# RRF Fusion + Filter + Sort
# ---------------------------------------------------------------------------

def fuse_results(
    bm25_results: list[ScoredPage],
    semantic_results: list[ScoredPage],
    k: int = 60,
) -> list[ScoredPage]:
    """Reciprocal Rank Fusion of two result lists."""
    scores: dict[str, float] = {}
    meta: dict[str, ScoredPage] = {}

    for rank, page in enumerate(bm25_results):
        scores[page.page_id] = scores.get(page.page_id, 0) + 1 / (k + rank)
        meta[page.page_id] = page

    for rank, page in enumerate(semantic_results):
        scores[page.page_id] = scores.get(page.page_id, 0) + 1 / (k + rank)
        if page.page_id not in meta:
            meta[page.page_id] = page

    fused = []
    for pid, score in scores.items():
        p = meta[pid]
        fused.append(ScoredPage(
            page_id=p.page_id, title=p.title, folder=p.folder,
            updated=p.updated, score=score,
        ))

    fused.sort(key=lambda x: x.score, reverse=True)
    return fused


def apply_filters(
    results: list[ScoredPage],
    folder: str | None = None,
    updated_after: str | None = None,
    updated_before: str | None = None,
) -> list[ScoredPage]:
    """Filter results by folder and date range."""
    filtered = results
    if folder:
        filtered = [r for r in filtered if r.folder.startswith(folder)]
    if updated_after:
        filtered = [r for r in filtered if r.updated >= updated_after and r.updated != "unknown"]
    if updated_before:
        filtered = [r for r in filtered if r.updated <= updated_before and r.updated != "unknown"]
    return filtered


def apply_sort(results: list[ScoredPage], sort_by: str = "relevance") -> list[ScoredPage]:
    """Sort results."""
    if sort_by == "date":
        return sorted(results, key=lambda x: x.updated, reverse=True)
    elif sort_by == "title":
        return sorted(results, key=lambda x: x.title)
    return results  # relevance is already sorted


# ---------------------------------------------------------------------------
# Public search API
# ---------------------------------------------------------------------------

def search(
    query: str,
    top_n: int = 20,
    folder: str | None = None,
    updated_after: str | None = None,
    updated_before: str | None = None,
    sort_by: str = "relevance",
    semantic: bool = True,
) -> tuple[list[ScoredPage], str]:
    """Run search and return (results, search_mode)."""
    # Fetch more results before filtering to avoid truncation-before-filter bug
    fetch_n = max(top_n * 5, 100)

    bm25 = get_bm25()
    bm25.build()
    bm25_results = bm25.query(query, top_n=fetch_n)

    search_mode = "bm25"
    if semantic:
        sem_results = semantic_search(query, top_n=fetch_n)
        if sem_results:
            results = fuse_results(bm25_results, sem_results)
            search_mode = "hybrid"
        else:
            results = bm25_results
    else:
        results = bm25_results

    # Filter THEN truncate (not the other way around)
    results = apply_filters(results, folder, updated_after, updated_before)
    results = apply_sort(results, sort_by)
    return results[:top_n], search_mode
