"""Search engine — BM25 + semantic search with RRF fusion."""

import json
import math
import os
import re
import threading
from collections import Counter, deque
from pathlib import Path
from typing import Any

from llm_wiki_mcp.frontmatter import parse as parse_frontmatter
from llm_wiki_mcp.runtime_config import (
    DEFAULT_EMBEDDING_MODEL,
    load_embedding_config,
    load_negative_feedback_config,
)
from llm_wiki_mcp.negative_feedback import apply_penalties, penalties_for_query
from llm_wiki_mcp.pipeline import PipelineDependencies, production_pipeline_config, run_search_pipeline
from llm_wiki_mcp.search_types import ScoredPage, _FRONTMATTER_RE, tokenize
from llm_wiki_mcp.wiki import WIKI_ROOT, PAGES_DIR, SYSTEM_DIR, all_pages, page_id_from_path
from llm_wiki_mcp.link_fix import atomic_write


def searchable_pages() -> list[Path]:
    """Return normal pages plus system pages that are useful recall targets."""
    return all_pages() + sorted(SYSTEM_DIR.glob("*.md"))


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------

_BM25_CACHE_FILE = WIKI_ROOT / ".index" / "bm25.json"
_BM25_CACHE_SCHEMA = 4
_ACTIVE_STATUS = "active"
_VALID_LIFECYCLE_STATUSES = {"active", "deprecated", "archived"}
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
    if not isinstance(value, str):
        return _ACTIVE_STATUS
    normalized = value.strip().lower()
    if normalized in _VALID_LIFECYCLE_STATUSES:
        return normalized
    return _ACTIVE_STATUS


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


def _is_active_result(result: ScoredPage) -> bool:
    return _normalize_lifecycle_status(result.status) == _ACTIVE_STATUS


def _is_reference_result(result: ScoredPage) -> bool:
    return _normalize_page_type(result.page_type, folder=result.folder) == _REFERENCE_PAGE_TYPE


def _folder_from_meta(meta: dict) -> str:
    try:
        parent = Path(meta["path"]).parent
        if parent != PAGES_DIR:
            return parent.name
    except (KeyError, TypeError):
        pass
    return ""


def _meta_page_type(meta: dict, *, folder: str = "") -> str:
    return _normalize_page_type(meta.get("page_type"), folder=folder)


def _meta_sensitivity(meta: dict, *, folder: str = "") -> str:
    return _normalize_sensitivity(meta.get("sensitivity"), folder=folder)


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
        self._persistence_dirty: bool = False
        # Reentrant lock so the BM25 singleton can be safely shared between
        # the FastMCP main thread and ingest's background thread. `build`
        # mutates internal state; `query` iterates `_cache.items()`. Without
        # this lock a concurrent build during a query raises
        # "dictionary changed size during iteration" and `_df` can drift
        # via interleaved subtract/add pairs.
        self._lock = threading.RLock()

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

        # Validate shape before adopting any of the loaded state — a
        # malformed-but-JSON-valid cache file would otherwise poison the
        # singleton until the next manual rebuild.
        try:
            raw_docs = doc.get("docs", {})
            if not isinstance(raw_docs, dict):
                raise ValueError("docs must be a dict")
            for pid, entry in raw_docs.items():
                if not isinstance(entry, dict):
                    raise ValueError(f"entry {pid!r} not a dict")
                tf_map = entry.get("tf_map")
                if not isinstance(tf_map, dict):
                    raise ValueError(f"entry {pid!r} missing tf_map dict")
                if not isinstance(entry.get("doc_len"), int):
                    raise ValueError(f"entry {pid!r} missing int doc_len")
                if not isinstance(entry.get("mtime_ns"), int):
                    raise ValueError(f"entry {pid!r} missing int mtime_ns")
                if not isinstance(entry.get("size"), int):
                    raise ValueError(f"entry {pid!r} missing int size")
            global_state = doc.get("global", {})
            df = dict(global_state.get("df", {}))
            n = int(global_state.get("n", 0))
            avgdl = float(global_state.get("avgdl", 0.0))
        except (KeyError, TypeError, ValueError):
            return  # Leave the singleton in its empty state and rebuild fresh.

        self._cache = raw_docs
        self._df = df
        self._n = n
        self._avgdl = avgdl

    def _persist_cache(self) -> None:
        if os.environ.get("LLM_WIKI_READ_ONLY") == "1":
            return
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
        with self._lock:
            self._build_locked()

    def _build_locked(self) -> None:
        if not self._cache_loaded:
            self._load_cache()
            self._cache_loaded = True

        # Snapshot disk state.
        current: dict[str, tuple[Path, int, int]] = {}
        for path in searchable_pages():
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
            fm, _body = parse_frontmatter(content)
            title = fm.get("title", pid) if isinstance(fm.get("title", pid), str) else pid
            updated = fm.get("updated", "") if isinstance(fm.get("updated", ""), str) else ""
            status = _normalize_lifecycle_status(fm.get("status"))
            superseded_by = (
                fm.get("superseded_by", "")
                if isinstance(fm.get("superseded_by", ""), str)
                else ""
            )
            folder = path.parent.name if path.parent != PAGES_DIR else ""
            page_type = _normalize_page_type(fm.get("type"), folder=folder)
            sensitivity = _normalize_sensitivity(fm.get("sensitivity"), folder=folder)

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
                "status": status,
                "superseded_by": superseded_by,
                "page_type": page_type,
                "sensitivity": sensitivity,
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
            self._persistence_dirty = True
        if self._persistence_dirty and os.environ.get("LLM_WIKI_READ_ONLY") != "1":
            try:
                self._persist_cache()
            except OSError:
                pass
            else:
                self._persistence_dirty = False

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

    def query(
        self,
        query_text: str,
        top_n: int = 20,
        *,
        include_reference: bool = False,
    ) -> list[ScoredPage]:
        """Search the index."""
        with self._lock:
            if not self._cache:
                self._build_locked()

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
                page_type = _normalize_page_type(doc.get("page_type"), folder=doc.get("folder", ""))
                if not include_reference and page_type == _REFERENCE_PAGE_TYPE:
                    continue
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
                        status=_normalize_lifecycle_status(doc.get("status")),
                        superseded_by=doc.get("superseded_by", "")
                        if isinstance(doc.get("superseded_by", ""), str)
                        else "",
                        page_type=page_type,
                        sensitivity=_normalize_sensitivity(doc.get("sensitivity"), folder=doc.get("folder", "")),
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
#   ~/.wiki/.index/embeddings.sqlite stores page and recall-question vectors:
#     embeddings(page_id PK, vector BLOB, mtime REAL, norm REAL, dim INT,
#                model TEXT, text_prefix TEXT)
#   `vector` is a packed float64 array (8 bytes per dim -> ~6KB per 768-dim
#   vector). `norm` is precomputed at write time so semantic_search never
#   recomputes per-row norms at query time.
#
# Migration: a one-shot import from the legacy ~/.wiki/.embeddings.json runs
# on first connect when the SQLite file does not yet exist.

import sqlite3
import struct

EMBED_MODEL = DEFAULT_EMBEDDING_MODEL

EMBEDDINGS_DB = WIKI_ROOT / ".index" / "embeddings.sqlite"
LEGACY_EMBEDDINGS_FILE = WIKI_ROOT / ".embeddings.json"
EMBEDDINGS_FILE = LEGACY_EMBEDDINGS_FILE  # back-compat alias for any external imports

_EMBED_DB_LOCK = threading.Lock()
MAX_CHUNKS_PER_PAGE = 8
MAX_CHUNK_CHARS = 900
CHUNK_SCORE_WEIGHT = 0.92
CHUNK_SEARCH_MIN_TOP_SCORE = 0.45
CHUNK_SEARCH_MIN_MARGIN = 0.002


def _embedding_profile() -> tuple[str, str, str]:
    cfg = load_embedding_config()
    return cfg.model, cfg.document_prefix, cfg.query_prefix


def _apply_prefix(prefix: str, text: str) -> str:
    return f"{prefix}{text}" if prefix else text


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, ddl: str) -> None:
    columns = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    if column not in columns:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {ddl}")


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
    if os.environ.get("LLM_WIKI_READ_ONLY") == "1":
        return sqlite3.connect(f"file:{EMBEDDINGS_DB}?mode=ro", uri=True)
    EMBEDDINGS_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(EMBEDDINGS_DB)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS embeddings (
            page_id TEXT PRIMARY KEY,
            vector BLOB NOT NULL,
            mtime REAL NOT NULL,
            norm REAL NOT NULL,
            dim INTEGER NOT NULL,
            model TEXT NOT NULL DEFAULT '',
            text_prefix TEXT NOT NULL DEFAULT ''
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS question_embeddings (
            key TEXT PRIMARY KEY,
            page_id TEXT NOT NULL,
            question_idx INTEGER NOT NULL,
            question TEXT NOT NULL,
            vector BLOB NOT NULL,
            mtime REAL NOT NULL,
            norm REAL NOT NULL,
            dim INTEGER NOT NULL,
            model TEXT NOT NULL DEFAULT '',
            text_prefix TEXT NOT NULL DEFAULT ''
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chunk_embeddings (
            key TEXT PRIMARY KEY,
            page_id TEXT NOT NULL,
            chunk_idx INTEGER NOT NULL,
            text TEXT NOT NULL,
            vector BLOB NOT NULL,
            mtime REAL NOT NULL,
            norm REAL NOT NULL,
            dim INTEGER NOT NULL,
            model TEXT NOT NULL DEFAULT '',
            text_prefix TEXT NOT NULL DEFAULT ''
        )
        """
    )
    _ensure_column(conn, "embeddings", "model", "model TEXT NOT NULL DEFAULT ''")
    _ensure_column(conn, "embeddings", "text_prefix", "text_prefix TEXT NOT NULL DEFAULT ''")
    _ensure_column(conn, "question_embeddings", "model", "model TEXT NOT NULL DEFAULT ''")
    _ensure_column(
        conn,
        "question_embeddings",
        "text_prefix",
        "text_prefix TEXT NOT NULL DEFAULT ''",
    )
    _ensure_column(conn, "chunk_embeddings", "model", "model TEXT NOT NULL DEFAULT ''")
    _ensure_column(conn, "chunk_embeddings", "text_prefix", "text_prefix TEXT NOT NULL DEFAULT ''")
    return conn


_legacy_migration_done = False


def _maybe_migrate_legacy_json() -> None:
    """One-shot import of ~/.wiki/.embeddings.json into SQLite.

    Runs at most once per process. The legacy file is left in place so a
    rollback to the old code path remains possible; subsequent runs are
    no-ops because the SQLite table is already populated.
    """
    global _legacy_migration_done
    if os.environ.get("LLM_WIKI_READ_ONLY") == "1":
        return
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
            model = EMBED_MODEL
            text_prefix = ""
            rows = []
            for pid, data in payload.items():
                vec = data.get("vector") if isinstance(data, dict) else None
                if not vec:
                    continue
                mtime = float(data.get("mtime", 0.0))
                norm = _vec_norm(vec)
                rows.append((pid, _pack_vector(vec), mtime, norm, len(vec), model, text_prefix))
            if rows:
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO embeddings
                    (page_id, vector, mtime, norm, dim, model, text_prefix)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )
                conn.commit()
        finally:
            conn.close()
        _legacy_migration_done = True


def _load_embedding(pid: str) -> tuple[list[float], float, float] | None:
    """Return (vector, mtime, norm) for a single page, or None."""
    _maybe_migrate_legacy_json()
    model, document_prefix, _query_prefix = _embedding_profile()
    conn = _connect_embeddings()
    try:
        row = conn.execute(
            """
            SELECT vector, mtime, norm, dim FROM embeddings
            WHERE page_id = ? AND model = ? AND text_prefix = ?
            """,
            (pid, model, document_prefix),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    blob, mtime, norm, dim = row
    return _unpack_vector(blob, int(dim)), float(mtime), float(norm)


def _store_embeddings_batch(
    rows: list[tuple[str, list[float], float]],
    *,
    model: str,
    text_prefix: str,
) -> None:
    """Insert/replace a batch of (page_id, vector, mtime) rows."""
    if not rows:
        return
    packed = [
        (pid, _pack_vector(vec), float(mtime), _vec_norm(vec), len(vec), model, text_prefix)
        for pid, vec, mtime in rows
    ]
    with _EMBED_DB_LOCK:
        conn = _connect_embeddings()
        try:
            conn.executemany(
                """
                INSERT OR REPLACE INTO embeddings
                (page_id, vector, mtime, norm, dim, model, text_prefix)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                packed,
            )
            conn.commit()
        finally:
            conn.close()


def _store_question_embeddings_batch(
    rows: list[tuple[str, int, str, list[float], float]],
    *,
    model: str,
    text_prefix: str,
) -> None:
    """Insert/replace recall-question embedding rows."""
    if not rows:
        return
    packed = [
        (
            f"{pid}#q{idx}",
            pid,
            idx,
            question,
            _pack_vector(vec),
            float(mtime),
            _vec_norm(vec),
            len(vec),
            model,
            text_prefix,
        )
        for pid, idx, question, vec, mtime in rows
    ]
    with _EMBED_DB_LOCK:
        conn = _connect_embeddings()
        try:
            conn.executemany(
                """
                INSERT OR REPLACE INTO question_embeddings
                (key, page_id, question_idx, question, vector, mtime, norm, dim, model, text_prefix)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                packed,
            )
            conn.commit()
        finally:
            conn.close()


def _delete_chunk_embeddings(
    page_ids: set[str],
    *,
    model: str,
    text_prefix: str,
) -> None:
    if not page_ids:
        return
    with _EMBED_DB_LOCK:
        conn = _connect_embeddings()
        try:
            for pid in page_ids:
                conn.execute(
                    """
                    DELETE FROM chunk_embeddings
                    WHERE page_id = ? AND model = ? AND text_prefix = ?
                    """,
                    (pid, model, text_prefix),
                )
            conn.commit()
        finally:
            conn.close()


def _store_chunk_embeddings_batch(
    rows: list[tuple[str, int, str, list[float], float]],
    *,
    model: str,
    text_prefix: str,
) -> None:
    """Insert/replace chunk embedding rows."""
    if not rows:
        return
    packed = [
        (
            f"{pid}#c{idx}",
            pid,
            idx,
            text,
            _pack_vector(vec),
            float(mtime),
            _vec_norm(vec),
            len(vec),
            model,
            text_prefix,
        )
        for pid, idx, text, vec, mtime in rows
    ]
    with _EMBED_DB_LOCK:
        conn = _connect_embeddings()
        try:
            conn.executemany(
                """
                INSERT OR REPLACE INTO chunk_embeddings
                (key, page_id, chunk_idx, text, vector, mtime, norm, dim, model, text_prefix)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                packed,
            )
            conn.commit()
        finally:
            conn.close()


def _iter_all_embeddings() -> "list[tuple[str, list[float], float, float]]":
    """Snapshot all rows as (page_id, vector, mtime, norm)."""
    _maybe_migrate_legacy_json()
    model, document_prefix, _query_prefix = _embedding_profile()
    conn = _connect_embeddings()
    try:
        rows = conn.execute(
            """
            SELECT page_id, vector, mtime, norm, dim FROM embeddings
            WHERE model = ? AND text_prefix = ?
            """,
            (model, document_prefix),
        ).fetchall()
    finally:
        conn.close()
    out = []
    for pid, blob, mtime, norm, dim in rows:
        out.append((pid, _unpack_vector(blob, int(dim)), float(mtime), float(norm)))
    return out


def _iter_all_chunk_embeddings() -> "list[tuple[str, str, int, str, list[float], float, float]]":
    """Snapshot chunk rows as (key, page_id, idx, text, vector, mtime, norm)."""
    _maybe_migrate_legacy_json()
    model, document_prefix, _query_prefix = _embedding_profile()
    conn = _connect_embeddings()
    try:
        rows = conn.execute(
            """
            SELECT key, page_id, chunk_idx, text, vector, mtime, norm, dim
            FROM chunk_embeddings
            WHERE model = ? AND text_prefix = ?
            """,
            (model, document_prefix),
        ).fetchall()
    finally:
        conn.close()
    out = []
    for key, pid, idx, text, blob, mtime, norm, dim in rows:
        out.append((key, pid, int(idx), text, _unpack_vector(blob, int(dim)), float(mtime), float(norm)))
    return out


def _iter_all_question_embeddings() -> "list[tuple[str, str, int, list[float], float, float]]":
    """Snapshot all recall-question rows as (key, page_id, idx, vector, mtime, norm)."""
    _maybe_migrate_legacy_json()
    model, document_prefix, _query_prefix = _embedding_profile()
    conn = _connect_embeddings()
    try:
        rows = conn.execute(
            """
            SELECT key, page_id, question_idx, vector, mtime, norm, dim
            FROM question_embeddings
            WHERE model = ? AND text_prefix = ?
            """,
            (model, document_prefix),
        ).fetchall()
    finally:
        conn.close()
    out = []
    for key, pid, idx, blob, mtime, norm, dim in rows:
        out.append((key, pid, int(idx), _unpack_vector(blob, int(dim)), float(mtime), float(norm)))
    return out


def _embedding_count() -> int:
    _maybe_migrate_legacy_json()
    model, document_prefix, _query_prefix = _embedding_profile()
    try:
        conn = _connect_embeddings()
    except sqlite3.OperationalError:
        return 0
    try:
        return conn.execute(
            "SELECT COUNT(*) FROM embeddings WHERE model = ? AND text_prefix = ?",
            (model, document_prefix),
        ).fetchone()[0]
    finally:
        conn.close()


def _chunked_page_ids() -> set[str]:
    _maybe_migrate_legacy_json()
    model, document_prefix, _query_prefix = _embedding_profile()
    conn = _connect_embeddings()
    try:
        return {
            row[0]
            for row in conn.execute(
                """
                SELECT DISTINCT page_id FROM chunk_embeddings
                WHERE model = ? AND text_prefix = ?
                """,
                (model, document_prefix),
            ).fetchall()
        }
    finally:
        conn.close()


def _recall_questions_from_content(content: str) -> list[str]:
    try:
        from llm_wiki_mcp.frontmatter import parse as parse_frontmatter

        meta, _body = parse_frontmatter(content)
    except Exception:
        return []
    questions = meta.get("recall_questions")
    if not isinstance(questions, list):
        return []
    return [q for q in questions if isinstance(q, str) and q.strip()][:8]


def _markdown_chunks(content: str, title: str) -> list[str]:
    meta, body = parse_frontmatter(content)
    chunks: list[str] = []
    heading = title
    buffer: list[str] = []

    summary = meta.get("summary") if isinstance(meta.get("summary"), str) else ""
    entities = meta.get("entities") if isinstance(meta.get("entities"), list) else []
    page_type = meta.get("type") if isinstance(meta.get("type"), str) else ""
    updated = meta.get("updated") if isinstance(meta.get("updated"), str) else ""
    context_lines = [f"Page: {title}"]
    if summary.strip():
        context_lines.append(f"Summary: {summary.strip()}")
    if entities:
        context_lines.append("Entities: " + ", ".join(str(item) for item in entities[:8]))
    if page_type or updated:
        context_lines.append(f"Metadata: type={page_type or 'knowledge'} updated={updated or 'unknown'}")
    context = "\n".join(context_lines)

    def flush() -> None:
        nonlocal buffer
        text = re.sub(r"\s+", " ", "\n".join(buffer)).strip()
        buffer = []
        if not text:
            return
        prefix = f"{context}\nHeading: {heading}" if heading and heading != title else context
        while text:
            piece = text[:MAX_CHUNK_CHARS].strip()
            if len(text) > MAX_CHUNK_CHARS and " " in piece:
                piece = piece.rsplit(" ", 1)[0].strip() or text[:MAX_CHUNK_CHARS].strip()
            chunks.append(f"{prefix}\n\n{piece}")
            text = text[len(piece):].strip()
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


def _should_scan_chunks(page_scores: list[float]) -> bool:
    if not page_scores:
        return True
    ordered = sorted(page_scores, reverse=True)
    top1 = ordered[0]
    top2 = ordered[1] if len(ordered) > 1 else 0.0
    return top1 < CHUNK_SEARCH_MIN_TOP_SCORE or (top1 - top2) < CHUNK_SEARCH_MIN_MARGIN


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
    model, document_prefix, _query_prefix = _embedding_profile()

    # Pull existing mtimes for the candidate page set in one query so
    # we don't pay per-row SELECTs inside the loop.
    conn = _connect_embeddings()
    try:
        existing_mtimes: dict[str, float] = {
            row[0]: float(row[1])
            for row in conn.execute(
                """
                SELECT page_id, mtime FROM embeddings
                WHERE model = ? AND text_prefix = ?
                """,
                (model, document_prefix),
            ).fetchall()
        }
    finally:
        conn.close()

    existing_chunk_pages = _chunked_page_ids()
    pages_to_process: list[tuple[str, str, int, str, float]] = []
    for path in searchable_pages():
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
            and pid in existing_chunk_pages
            and not page_ids
        ):
            continue

        try:
            content = path.read_text()
        except (OSError, UnicodeDecodeError):
            continue

        fm_match = re.search(r"title:\s*(.+)", content)
        title = fm_match.group(1).strip() if fm_match else pid
        questions = _recall_questions_from_content(content)
        recall_text = "\n".join(f"Q: {q}" for q in questions)
        embed_text = f"{title}\n\n{recall_text}\n\n{_FRONTMATTER_RE.sub('', content)[:2000]}"
        pages_to_process.append(("page", pid, -1, embed_text, mtime))
        for idx, question in enumerate(questions):
            pages_to_process.append(("question", pid, idx, question, mtime))
        for idx, chunk in enumerate(_markdown_chunks(content, title)):
            pages_to_process.append(("chunk", pid, idx, chunk, mtime))

    updated_count = 0
    if pages_to_process:
        _delete_chunk_embeddings(
            {pid for kind, pid, _idx, _text, _mtime in pages_to_process if kind == "page"},
            model=model,
            text_prefix=document_prefix,
        )
        batch_size = 32
        for i in range(0, len(pages_to_process), batch_size):
            batch = pages_to_process[i:i + batch_size]
            texts = [_apply_prefix(document_prefix, t[3]) for t in batch]
            try:
                vectors = embed(texts, model=model)
            except Exception:
                continue
            rows: list[tuple[str, list[float], float]] = []
            question_rows: list[tuple[str, int, str, list[float], float]] = []
            chunk_rows: list[tuple[str, int, str, list[float], float]] = []
            for (kind, pid, idx, text, mtime), vec in zip(batch, vectors):
                if kind == "question":
                    question_rows.append((pid, idx, text, vec, mtime))
                elif kind == "chunk":
                    chunk_rows.append((pid, idx, text, vec, mtime))
                else:
                    rows.append((pid, vec, mtime))
            _store_embeddings_batch(rows, model=model, text_prefix=document_prefix)
            _store_question_embeddings_batch(
                question_rows,
                model=model,
                text_prefix=document_prefix,
            )
            _store_chunk_embeddings_batch(
                chunk_rows,
                model=model,
                text_prefix=document_prefix,
            )
            updated_count += len(rows)

    return updated_count


def semantic_search(
    query: str,
    top_n: int = 20,
    *,
    include_reference: bool = False,
) -> list[ScoredPage]:
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

    model, _document_prefix, query_prefix = _embedding_profile()
    try:
        q_vec = embed([_apply_prefix(query_prefix, query)], model=model)[0]
    except Exception:
        return []

    q_norm = _vec_norm(q_vec)
    if q_norm == 0:
        return []

    store = get_store()
    store.refresh()

    by_page: dict[str, ScoredPage] = {}
    for pid, vec, _mtime, norm in _iter_all_embeddings():
        if norm == 0:
            continue
        meta = store.meta(pid)
        if meta is None:
            continue
        folder = _folder_from_meta(meta)
        page_type = _meta_page_type(meta, folder=folder)
        sensitivity = _meta_sensitivity(meta, folder=folder)
        if not include_reference and page_type == _REFERENCE_PAGE_TYPE:
            continue

        dot = 0.0
        for x, y in zip(q_vec, vec):
            dot += x * y
        sim = dot / (q_norm * norm)
        by_page[pid] = ScoredPage(
            page_id=pid,
            title=meta["title"],
            folder=folder,
            updated=meta["updated"],
            score=sim,
            status=_normalize_lifecycle_status(meta.get("status")),
            superseded_by=meta.get("superseded_by", "")
            if isinstance(meta.get("superseded_by", ""), str)
            else "",
            page_type=page_type,
            sensitivity=sensitivity,
        )

    for _key, pid, _idx, vec, _mtime, norm in _iter_all_question_embeddings():
        if norm == 0:
            continue
        meta = store.meta(pid)
        if meta is None:
            continue
        folder = _folder_from_meta(meta)
        page_type = _meta_page_type(meta, folder=folder)
        sensitivity = _meta_sensitivity(meta, folder=folder)
        if not include_reference and page_type == _REFERENCE_PAGE_TYPE:
            continue
        dot = 0.0
        for x, y in zip(q_vec, vec):
            dot += x * y
        sim = dot / (q_norm * norm)
        existing = by_page.get(pid)
        if existing is None or sim > existing.score:
            by_page[pid] = ScoredPage(
                page_id=pid,
                title=meta["title"],
                folder=folder,
                updated=meta["updated"],
                score=sim,
                status=_normalize_lifecycle_status(meta.get("status")),
                superseded_by=meta.get("superseded_by", "")
                if isinstance(meta.get("superseded_by", ""), str)
                else "",
                page_type=page_type,
                sensitivity=sensitivity,
            )

    if _should_scan_chunks([page.score for page in by_page.values()]):
        for _key, pid, _idx, text, vec, _mtime, norm in _iter_all_chunk_embeddings():
            if norm == 0:
                continue
            meta = store.meta(pid)
            if meta is None:
                continue
            folder = _folder_from_meta(meta)
            page_type = _meta_page_type(meta, folder=folder)
            sensitivity = _meta_sensitivity(meta, folder=folder)
            if not include_reference and page_type == _REFERENCE_PAGE_TYPE:
                continue
            dot = 0.0
            for x, y in zip(q_vec, vec):
                dot += x * y
            sim = (dot / (q_norm * norm)) * CHUNK_SCORE_WEIGHT
            existing = by_page.get(pid)
            if existing is None or sim > existing.score:
                by_page[pid] = ScoredPage(
                    page_id=pid,
                    title=meta["title"],
                    folder=folder,
                    updated=meta["updated"],
                    score=sim,
                    snippet=text[:240],
                    status=_normalize_lifecycle_status(meta.get("status")),
                    superseded_by=meta.get("superseded_by", "")
                    if isinstance(meta.get("superseded_by", ""), str)
                    else "",
                    page_type=page_type,
                    sensitivity=sensitivity,
                )

    results = list(by_page.values())
    results.sort(key=lambda x: x.score, reverse=True)
    return results[:top_n]


# ---------------------------------------------------------------------------
# RRF Fusion + Filter + Sort
# ---------------------------------------------------------------------------

DEFAULT_FUSION_WEIGHTS: dict[str, float] = {
    "bm25": 1.0,
    "semantic": 0.6,
    "graph": 0.0,
    "usage_prior": 0.0,
    "bm25_score_bonus": 0.005,
    "bm25_rank_bonus": 0.006,
    "bm25_rank_decay": 0.006,
    "semantic_min_top_score": 0.45,
    "semantic_min_margin": 0.002,
    "semantic_low_confidence_weight": 0.25,
    "usage_prior_decay": 0.98,
    "usage_prior_cap": 3.0,
    "retention_prior": 0.015,
}

ACTIVE_SEARCH_POLICY_FILE = WIKI_ROOT / "recall" / "search-policy.json"


def load_active_fusion_weights(path: Path | None = None) -> dict[str, float]:
    """Load the validated search policy, falling back safely to defaults.

    The artifact is deliberately separate from user configuration: evaluation
    can atomically replace it after holdout validation, while malformed or
    stale artifacts can never make production search unusable.
    """
    policy_path = path or ACTIVE_SEARCH_POLICY_FILE
    try:
        payload = json.loads(policy_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return dict(DEFAULT_FUSION_WEIGHTS)
    if (
        not isinstance(payload, dict)
        or payload.get("version") != 1
        or payload.get("source") != "search_eval.self_tune"
        or not isinstance(payload.get("holdout"), dict)
    ):
        return dict(DEFAULT_FUSION_WEIGHTS)
    raw = payload.get("weights")
    if not isinstance(raw, dict) or set(raw) != set(DEFAULT_FUSION_WEIGHTS):
        return dict(DEFAULT_FUSION_WEIGHTS)
    weights: dict[str, float] = {}
    for key, value in raw.items():
        if isinstance(value, bool):
            return dict(DEFAULT_FUSION_WEIGHTS)
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return dict(DEFAULT_FUSION_WEIGHTS)
        if not math.isfinite(numeric) or numeric < 0:
            return dict(DEFAULT_FUSION_WEIGHTS)
        weights[key] = numeric
    if not any(weights[channel] > 0 for channel in ("bm25", "semantic", "graph", "usage_prior")):
        return dict(DEFAULT_FUSION_WEIGHTS)
    for bounded in (
        "semantic_min_top_score",
        "semantic_min_margin",
        "semantic_low_confidence_weight",
        "usage_prior_decay",
    ):
        if weights[bounded] > 1:
            return dict(DEFAULT_FUSION_WEIGHTS)
    return weights


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
    top2 = max(0.0, float(semantic_results[1].score)) if len(semantic_results) > 1 else 0.0
    if top1 < min_top or (top1 - top2) < min_margin:
        return low_weight
    return 1.0


def fuse_results(
    bm25_results: list[ScoredPage],
    semantic_results: list[ScoredPage],
    graph_results: list[ScoredPage] | None = None,
    usage_results: list[ScoredPage] | None = None,
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
                score += weight * bm25_score_bonus * (max(0.0, float(page.score)) / top_raw)
                score += weight * max(0.0, bm25_rank_bonus - (bm25_rank_decay * rank))
            scores[page.page_id] = scores.get(page.page_id, 0) + score
            if page.page_id not in meta:
                meta[page.page_id] = page

    add_results(bm25_results, "bm25")
    add_results(semantic_results, "semantic")
    add_results(graph_results or [], "graph")
    add_results(usage_results or [], "usage_prior")

    retention_weight = max(0.0, float(weights.get("retention_prior", 0.0)))
    if retention_weight:
        try:
            from llm_wiki_mcp.retention import retention_score

            for page_id in list(scores):
                prior = retention_score(page_id)
                if prior > 0:
                    scores[page_id] = scores.get(page_id, 0.0) + (retention_weight * prior)
        except Exception:
            pass

    fused = []
    for pid, score in scores.items():
        p = meta[pid]
        fused.append(ScoredPage(
            page_id=p.page_id, title=p.title, folder=p.folder,
            updated=p.updated, score=score,
            status=p.status, superseded_by=p.superseded_by,
            page_type=p.page_type,
            sensitivity=p.sensitivity,
        ))

    fused.sort(key=lambda x: x.score, reverse=True)
    return fused


def graph_expand_results(results: list[ScoredPage], *, decay: float = 0.5, limit: int = 50) -> list[ScoredPage]:
    if decay <= 0 or not results:
        return []
    from llm_wiki_mcp.index_store import get_store

    store = get_store()
    store.refresh()
    seen = {result.page_id for result in results}
    expanded: dict[str, ScoredPage] = {}
    for result in results[:10]:
        linked: list[tuple[str, float]] = [(page_id, 1.0) for page_id in (store.outlinks(result.page_id) + store.backlinks(result.page_id))]
        try:
            from llm_wiki_mcp.cofire import neighbors as cofire_neighbors

            linked.extend(
                (
                    str(edge["page_id"]),
                    max(0.05, min(1.0, float(edge.get("weight") or 0.0))),
                )
                for edge in cofire_neighbors(result.page_id, limit=8)
            )
        except Exception:
            pass
        for page_id, edge_weight in linked:
            if page_id in seen:
                continue
            meta = store.meta(page_id)
            if meta is None:
                continue
            score = result.score * decay * edge_weight
            existing = expanded.get(page_id)
            if existing is not None and existing.score >= score:
                continue
            folder = _folder_from_meta(meta)
            expanded[page_id] = ScoredPage(
                page_id=page_id,
                title=meta["title"],
                folder=folder,
                updated=meta["updated"],
                score=score,
                status=_normalize_lifecycle_status(meta.get("status")),
                superseded_by=meta.get("superseded_by", "")
                if isinstance(meta.get("superseded_by", ""), str)
                else "",
                page_type=_meta_page_type(meta, folder=folder),
                sensitivity=_meta_sensitivity(meta, folder=folder),
            )
            if len(expanded) >= limit:
                break
        if len(expanded) >= limit:
            break
    return sorted(expanded.values(), key=lambda page: page.score, reverse=True)


def usage_prior_results(
    candidate_ids: set[str],
    *,
    limit: int = 50,
    decay: float = 0.98,
    cap: float = 3.0,
) -> list[ScoredPage]:
    if not candidate_ids:
        return []
    feedback_file = WIKI_ROOT / "recall" / "feedback.jsonl"
    try:
        with feedback_file.open(encoding="utf-8") as f:
            lines = deque(f, maxlen=1000)
    except OSError:
        return []
    scores: Counter[str] = Counter()
    decay = max(0.0, min(1.0, float(decay)))
    cap = max(0.0, float(cap))
    for age, line in enumerate(reversed(lines)):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(record, dict) or record.get("kind") != "injection_used":
            continue
        weight = decay ** age
        for page_id in record.get("expected_pages", []) or record.get("injected_pages", []):
            if isinstance(page_id, str) and page_id in candidate_ids:
                scores[page_id] = min(cap, scores[page_id] + weight)
    if not scores:
        return []
    from llm_wiki_mcp.index_store import get_store

    store = get_store()
    store.refresh()
    out: list[ScoredPage] = []
    for page_id, score in scores.most_common(limit):
        meta = store.meta(page_id)
        if meta is None:
            continue
        folder = _folder_from_meta(meta)
        out.append(
            ScoredPage(
                page_id=page_id,
                title=meta["title"],
                folder=folder,
                updated=meta["updated"],
                score=float(score),
                status=_normalize_lifecycle_status(meta.get("status")),
                superseded_by=meta.get("superseded_by", "")
                if isinstance(meta.get("superseded_by", ""), str)
                else "",
                page_type=_meta_page_type(meta, folder=folder),
                sensitivity=_meta_sensitivity(meta, folder=folder),
            )
        )
    return out


def apply_filters(
    results: list[ScoredPage],
    folder: str | None = None,
    updated_after: str | None = None,
    updated_before: str | None = None,
) -> list[ScoredPage]:
    """Filter results by folder and date range."""
    filtered = [r for r in results if _is_active_result(r)]
    if not folder:
        filtered = [r for r in filtered if not _is_reference_result(r)]
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

def _pipeline_dependencies() -> PipelineDependencies:
    return PipelineDependencies(
        get_bm25=get_bm25,
        semantic_search=semantic_search,
        graph_expand_results=graph_expand_results,
        usage_prior_results=usage_prior_results,
        fuse_results=fuse_results,
        apply_filters=apply_filters,
        apply_sort=apply_sort,
        load_negative_feedback_config=load_negative_feedback_config,
        penalties_for_query=penalties_for_query,
        apply_penalties=apply_penalties,
    )


def search(
    query: str,
    top_n: int = 20,
    folder: str | None = None,
    updated_after: str | None = None,
    updated_before: str | None = None,
    sort_by: str = "relevance",
    semantic: bool = True,
    fusion_weights: dict[str, float] | None = None,
) -> tuple[list[ScoredPage], str]:
    """Run search and return (results, search_mode)."""
    weights = (
        {**DEFAULT_FUSION_WEIGHTS, **fusion_weights}
        if fusion_weights is not None
        else load_active_fusion_weights()
    )
    result = run_search_pipeline(
        query,
        config=production_pipeline_config(
            top_n=top_n,
            folder=folder,
            updated_after=updated_after,
            updated_before=updated_before,
            sort_by=sort_by,
            semantic=semantic,
            fusion_weights=weights,
            include_reference=folder is not None,
        ),
        deps=_pipeline_dependencies(),
    )
    return result.results, result.search_mode
