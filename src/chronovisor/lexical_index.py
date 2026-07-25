"""Persistent inverted-BM25 and exact-anchor retrieval.

Pages remain the source of truth.  This SQLite database is a disposable,
incrementally refreshed search projection. Japanese text uses Chronovisor's
CJK bigrams and postings use integer term/page keys to avoid corpus-sized
string duplication.
"""

from __future__ import annotations

import math
import os
import sqlite3
import threading
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path

from chronovisor.frontmatter import parse as parse_frontmatter
from chronovisor.search_types import ScoredPage, tokenize
from chronovisor.store import PAGES_DIR, page_id_from_path

SCHEMA_VERSION = 5
ACTIVE_STATUS = "active"
VALID_STATUSES = {"active", "deprecated", "archived"}
VALID_PAGE_TYPES = {
    "knowledge",
    "reference",
    "episodic",
    "semantic",
    "procedural",
    "state",
    "lesson",
    "decision",
}


def _status(value: object) -> str:
    normalized = str(value or "").strip().lower()
    return normalized if normalized in VALID_STATUSES else ACTIVE_STATUS


def _page_type(value: object, folder: str) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in VALID_PAGE_TYPES:
        return normalized
    return "reference" if folder == "car-spec" else "knowledge"


def _sensitivity(value: object, folder: str) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"normal", "high"}:
        return normalized
    return "high" if folder == "career" else "normal"


def _strings(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


def _anchor_terms(
    *,
    page_id: str,
    title: str,
    folder: str,
    frontmatter: dict,
) -> dict[str, float]:
    weighted: dict[str, float] = {}

    def add(value: str, weight: float) -> None:
        for token in tokenize(value):
            weighted[token] = max(weighted.get(token, 0.0), weight)
        normalized = value.strip().lower()
        if normalized and " " not in normalized and len(normalized) >= 2:
            weighted[normalized] = max(weighted.get(normalized, 0.0), weight)

    add(page_id, 6.0)
    add(title, 5.0)
    if folder:
        add(folder, 1.0)
    for entity in _strings(frontmatter.get("entities")):
        add(entity, 4.0)
    for keyword in _strings(frontmatter.get("raw_keywords")):
        add(keyword, 3.0)
    for tag in _strings(frontmatter.get("tags")):
        add(tag, 2.0)
        if "/" in tag:
            add(tag.split("/", 1)[1], 2.5)
    uid = frontmatter.get("uid")
    if isinstance(uid, str) and uid:
        add(uid, 7.0)
    notation = frontmatter.get("classification_notation")
    if isinstance(notation, str) and notation:
        add(notation, 4.0)
    return weighted


class LexicalIndex:
    """Thread-safe persistent lexical index with bounded refresh scanning."""

    def __init__(
        self,
        *,
        path: Path,
        pages: Callable[[], list[Path]],
        refresh_interval_seconds: float = 2.0,
    ) -> None:
        self.path = path
        self._pages = pages
        self._refresh_interval = max(0.0, refresh_interval_seconds)
        self._connection: sqlite3.Connection | None = None
        self._persistent = False
        self._last_scan = 0.0
        self._lock = threading.RLock()

    def _open(self) -> sqlite3.Connection:
        read_only = os.environ.get("CHRONOVISOR_READ_ONLY") == "1"
        persistent = not read_only
        if self._connection is not None and self._persistent == persistent:
            return self._connection
        if self._connection is not None:
            self._connection.close()
            self._connection = None
        if read_only:
            if self.path.exists():
                connection = sqlite3.connect(
                    f"file:{self.path}?mode=ro",
                    uri=True,
                    check_same_thread=False,
                )
                self._connection = connection
                self._persistent = False
                return connection
            connection = sqlite3.connect(":memory:", check_same_thread=False)
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(
                self.path,
                timeout=5.0,
                check_same_thread=False,
            )
            os.chmod(self.path, 0o600)
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=NORMAL")
            connection.execute("PRAGMA busy_timeout=5000")
        self._initialize(connection)
        self._connection = connection
        self._persistent = persistent
        self._last_scan = 0.0
        return connection

    @staticmethod
    def _initialize(connection: sqlite3.Connection) -> None:
        current = int(connection.execute("PRAGMA user_version").fetchone()[0])
        legacy_projection = bool(
            connection.execute(
                """
                SELECT 1 FROM sqlite_master
                WHERE name IN ('documents', 'term_stats') LIMIT 1
                """
            ).fetchone()
        )
        if current not in (0, SCHEMA_VERSION):
            connection.executescript(
                """
                DROP TABLE IF EXISTS documents;
                DROP TABLE IF EXISTS anchors;
                DROP TABLE IF EXISTS postings;
                DROP TABLE IF EXISTS term_stats;
                DROP TABLE IF EXISTS terms;
                DROP TABLE IF EXISTS pages;
                """
            )
        else:
            connection.executescript(
                """
                DROP TABLE IF EXISTS documents;
                DROP TABLE IF EXISTS term_stats;
                """
            )
        connection.executescript(
            f"""
            CREATE TABLE IF NOT EXISTS pages (
                page_id TEXT PRIMARY KEY,
                mtime_ns INTEGER NOT NULL,
                size INTEGER NOT NULL,
                title TEXT NOT NULL,
                folder TEXT NOT NULL,
                updated TEXT NOT NULL,
                status TEXT NOT NULL,
                superseded_by TEXT NOT NULL,
                page_type TEXT NOT NULL,
                sensitivity TEXT NOT NULL,
                doc_len INTEGER NOT NULL,
                ordinal INTEGER NOT NULL UNIQUE,
                page_uid TEXT NOT NULL,
                classification_primary TEXT NOT NULL,
                classification_notation TEXT NOT NULL,
                classification_status TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS anchors (
                term TEXT NOT NULL,
                page_id TEXT NOT NULL,
                weight REAL NOT NULL,
                PRIMARY KEY (term, page_id)
            ) WITHOUT ROWID;
            CREATE INDEX IF NOT EXISTS anchors_page_idx ON anchors(page_id);
            CREATE TABLE IF NOT EXISTS terms (
                term_id INTEGER PRIMARY KEY,
                term TEXT NOT NULL UNIQUE,
                df INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS postings (
                term_id INTEGER NOT NULL,
                page_ordinal INTEGER NOT NULL,
                tf INTEGER NOT NULL,
                PRIMARY KEY (term_id, page_ordinal)
            ) WITHOUT ROWID;
            CREATE INDEX IF NOT EXISTS postings_page_idx ON postings(page_ordinal);
            PRAGMA user_version={SCHEMA_VERSION};
            """
        )
        connection.commit()
        if legacy_projection:
            connection.execute("VACUUM")

    def build(self, *, force: bool = False) -> None:
        with self._lock:
            connection = self._open()
            if not self._persistent and self.path.exists():
                return
            now = time.monotonic()
            if (
                not force
                and self._last_scan
                and now - self._last_scan < self._refresh_interval
            ):
                return
            self._last_scan = now
            current: dict[str, tuple[Path, int, int]] = {}
            for path in self._pages():
                try:
                    stat = path.stat()
                except OSError:
                    continue
                current[page_id_from_path(path)] = (
                    path,
                    stat.st_mtime_ns,
                    stat.st_size,
                )
            indexed_rows = connection.execute(
                "SELECT page_id, mtime_ns, size, ordinal FROM pages"
            ).fetchall()
            indexed = {
                str(page_id): (int(mtime_ns), int(size))
                for page_id, mtime_ns, size, _ordinal in indexed_rows
            }
            ordinals = {
                str(page_id): int(ordinal)
                for page_id, _mtime_ns, _size, ordinal in indexed_rows
            }
            next_ordinal = max(ordinals.values(), default=-1) + 1
            removed = set(indexed) - set(current)
            changed = [
                (page_id, row)
                for page_id, row in current.items()
                if indexed.get(page_id) != (row[1], row[2])
            ]
            if not removed and not changed:
                return
            prepared_pages: list[tuple[object, ...]] = []
            prepared_anchors: list[tuple[str, str, float]] = []
            prepared_postings: list[tuple[str, str, int]] = []
            prepared_ids: list[str] = []
            for page_id, (path, mtime_ns, size) in changed:
                try:
                    content = path.read_text(encoding="utf-8")
                except (OSError, UnicodeDecodeError):
                    continue
                frontmatter, _body = parse_frontmatter(content)
                title_value = frontmatter.get("title")
                title = (
                    title_value.strip()
                    if isinstance(title_value, str) and title_value.strip()
                    else page_id
                )
                updated_value = frontmatter.get("updated")
                updated = (
                    updated_value
                    if isinstance(updated_value, str)
                    else str(updated_value or "")
                )
                folder = path.parent.name if path.parent != PAGES_DIR else ""
                page_type = _page_type(frontmatter.get("type"), folder)
                superseded = frontmatter.get("superseded_by")
                superseded_by = superseded if isinstance(superseded, str) else ""
                prepared_ids.append(page_id)
                tokens = tokenize(title) * 3 + tokenize(content)
                prepared_postings.extend(
                    (term, page_id, tf) for term, tf in Counter(tokens).items()
                )
                prepared_pages.append(
                    (
                        page_id,
                        mtime_ns,
                        size,
                        title,
                        folder,
                        updated,
                        _status(frontmatter.get("status")),
                        superseded_by,
                        page_type,
                        _sensitivity(frontmatter.get("sensitivity"), folder),
                        len(tokens),
                        ordinals.get(page_id, next_ordinal),
                        (
                            frontmatter.get("uid")
                            if isinstance(frontmatter.get("uid"), str)
                            else ""
                        ),
                        (
                            frontmatter.get("classification_primary")
                            if isinstance(
                                frontmatter.get("classification_primary"), str
                            )
                            else ""
                        ),
                        (
                            frontmatter.get("classification_notation")
                            if isinstance(
                                frontmatter.get("classification_notation"), str
                            )
                            else ""
                        ),
                        (
                            frontmatter.get("classification_status")
                            if isinstance(
                                frontmatter.get("classification_status"), str
                            )
                            else "unclassified"
                        ),
                    )
                )
                if page_id not in ordinals:
                    next_ordinal += 1
                prepared_anchors.extend(
                    (term, page_id, weight)
                    for term, weight in _anchor_terms(
                        page_id=page_id,
                        title=title,
                        folder=folder,
                        frontmatter=frontmatter,
                    ).items()
                )
            with connection:
                affected_term_ids: set[int] = set()
                for page_id in (*removed, *prepared_ids):
                    page_ordinal = ordinals.get(page_id)
                    if page_ordinal is not None:
                        affected_term_ids.update(
                            int(row[0])
                            for row in connection.execute(
                                "SELECT term_id FROM postings WHERE page_ordinal = ?",
                                (page_ordinal,),
                            )
                        )
                        connection.execute(
                            "DELETE FROM postings WHERE page_ordinal = ?",
                            (page_ordinal,),
                        )
                    connection.execute(
                        "DELETE FROM anchors WHERE page_id = ?", (page_id,)
                    )
                    connection.execute(
                        "DELETE FROM pages WHERE page_id = ?", (page_id,)
                    )
                connection.executemany(
                    """
                    INSERT INTO pages
                    (page_id, mtime_ns, size, title, folder, updated, status,
                     superseded_by, page_type, sensitivity, doc_len, ordinal,
                     page_uid, classification_primary, classification_notation,
                     classification_status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    prepared_pages,
                )
                connection.executemany(
                    "INSERT INTO anchors(term, page_id, weight) VALUES (?, ?, ?)",
                    prepared_anchors,
                )
                connection.executemany(
                    "INSERT OR IGNORE INTO terms(term, df) VALUES (?, 0)",
                    ((term,) for term, _page_id, _tf in prepared_postings),
                )
                term_ids = {
                    str(term): int(term_id)
                    for term_id, term in connection.execute(
                        "SELECT term_id, term FROM terms"
                    )
                }
                page_ordinals = {
                    str(page_id): int(ordinal)
                    for page_id, ordinal in connection.execute(
                        "SELECT page_id, ordinal FROM pages"
                    )
                }
                posting_rows = [
                    (term_ids[term], page_ordinals[page_id], tf)
                    for term, page_id, tf in prepared_postings
                ]
                connection.executemany(
                    """
                    INSERT INTO postings(term_id, page_ordinal, tf)
                    VALUES (?, ?, ?)
                    """,
                    posting_rows,
                )
                affected_term_ids.update(
                    term_id for term_id, _page_ordinal, _tf in posting_rows
                )
                if len(affected_term_ids) > 1_000:
                    connection.execute("UPDATE terms SET df = 0")
                    connection.executemany(
                        "UPDATE terms SET df = ? WHERE term_id = ?",
                        (
                            (int(df), int(term_id))
                            for term_id, df in connection.execute(
                                """
                                SELECT term_id, COUNT(*)
                                FROM postings GROUP BY term_id
                                """
                            ).fetchall()
                        ),
                    )
                    connection.execute("DELETE FROM terms WHERE df = 0")
                else:
                    for term_id in affected_term_ids:
                        df = int(
                            connection.execute(
                                "SELECT COUNT(*) FROM postings WHERE term_id = ?",
                                (term_id,),
                            ).fetchone()[0]
                        )
                        if df:
                            connection.execute(
                                "UPDATE terms SET df = ? WHERE term_id = ?",
                                (df, term_id),
                            )
                        else:
                            connection.execute(
                                "DELETE FROM terms WHERE term_id = ?", (term_id,)
                            )

    @staticmethod
    def _row_to_page(row: tuple[object, ...], score: float) -> ScoredPage:
        return ScoredPage(
            page_id=str(row[0]),
            title=str(row[1]),
            folder=str(row[2]),
            updated=str(row[3]),
            score=score,
            status=str(row[4]),
            superseded_by=str(row[5]),
            page_type=str(row[6]),
            sensitivity=str(row[7]),
        )

    def query(
        self,
        query_text: str,
        top_n: int = 20,
        *,
        include_reference: bool = False,
    ) -> list[ScoredPage]:
        with self._lock:
            connection = self._open()
            query_counts = Counter(tokenize(query_text))
            if not query_counts:
                return []
            terms = list(query_counts)
            placeholders = ",".join("?" for _ in terms)
            reference_clause = (
                "" if include_reference else "AND p.page_type != 'reference'"
            )
            try:
                corpus = connection.execute(
                    "SELECT COUNT(*), COALESCE(AVG(doc_len), 1.0) FROM pages"
                ).fetchone()
                n = int(corpus[0])
                avgdl = float(corpus[1])
                rows = connection.execute(
                    f"""
                    SELECT s.term, p.page_id, p.title, p.folder, p.updated,
                           p.status, p.superseded_by, p.page_type, p.sensitivity,
                           x.tf, p.doc_len, s.df, p.ordinal
                    FROM postings x
                    JOIN terms s ON s.term_id = x.term_id
                    JOIN pages p ON p.ordinal = x.page_ordinal
                    WHERE s.term IN ({placeholders}) {reference_clause}
                    """,
                    terms,
                ).fetchall()
            except sqlite3.OperationalError:
                return []
            scores: dict[str, float] = {}
            page_rows: dict[str, tuple[object, ...]] = {}
            page_ordinals: dict[str, int] = {}
            k1 = 1.5
            b = 0.75
            for row in rows:
                term = str(row[0])
                page_id = str(row[1])
                tf = int(row[9])
                doc_len = int(row[10])
                df = int(row[11])
                idf = math.log((n - df + 0.5) / (df + 0.5) + 1)
                tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / avgdl))
                scores[page_id] = scores.get(page_id, 0.0) + (
                    query_counts[term] * idf * tf_norm
                )
                page_rows[page_id] = row[1:9]
                page_ordinals[page_id] = int(row[12])
            ranked = sorted(
                scores.items(),
                key=lambda item: (-item[1], page_ordinals[item[0]]),
            )
            return [
                self._row_to_page(page_rows[page_id], score)
                for page_id, score in ranked[: max(1, top_n)]
            ]

    def anchor_query(
        self,
        query_text: str,
        top_n: int = 20,
        *,
        include_reference: bool = False,
    ) -> list[ScoredPage]:
        terms = list(dict.fromkeys(tokenize(query_text)))
        normalized = query_text.strip().lower()
        if normalized and " " not in normalized and len(normalized) >= 2:
            terms.append(normalized)
        terms = list(dict.fromkeys(terms))
        if not terms:
            return []
        placeholders = ",".join("?" for _ in terms)
        reference_clause = "" if include_reference else "AND p.page_type != 'reference'"
        with self._lock:
            connection = self._open()
            rows = connection.execute(
                f"""
                SELECT p.page_id, p.title, p.folder, p.updated, p.status,
                       p.superseded_by, p.page_type, p.sensitivity,
                       SUM(a.weight) AS score
                FROM anchors a
                JOIN pages p ON p.page_id = a.page_id
                WHERE a.term IN ({placeholders}) {reference_clause}
                GROUP BY p.page_id
                ORDER BY score DESC, p.updated DESC
                LIMIT ?
                """,
                (*terms, max(1, top_n)),
            ).fetchall()
        return [self._row_to_page(row, float(row[8])) for row in rows]

    def stats(self) -> dict[str, object]:
        with self._lock:
            connection = self._open()
            pages = int(connection.execute("SELECT COUNT(*) FROM pages").fetchone()[0])
            anchors = int(
                connection.execute("SELECT COUNT(*) FROM anchors").fetchone()[0]
            )
            postings = int(
                connection.execute("SELECT COUNT(*) FROM postings").fetchone()[0]
            )
            terms = int(connection.execute("SELECT COUNT(*) FROM terms").fetchone()[0])
        return {
            "backend": "sqlite_inverted_bm25",
            "path": str(self.path),
            "pages": pages,
            "anchors": anchors,
            "postings": postings,
            "terms": terms,
        }
