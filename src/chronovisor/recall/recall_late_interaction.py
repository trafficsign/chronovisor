"""Separate experimental late-interaction index with incremental updates."""

from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= 0:
        return [0.0 for _value in vector]
    return [value / norm for value in vector]


def _maxsim(query: list[list[float]], document: list[list[float]]) -> float:
    if not query or not document:
        return 0.0
    normalized_doc = [_normalize(vector) for vector in document]
    return sum(
        max(
            sum(left * right for left, right in zip(qvec, dvec, strict=True))
            for dvec in normalized_doc
        )
        for qvec in (_normalize(vector) for vector in query)
    )


class LateInteractionIndex:
    """Store only challenger token vectors outside the production index."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(path) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS documents "
                "(page_id TEXT PRIMARY KEY, revision TEXT NOT NULL, vectors TEXT NOT NULL)"
            )

    def upsert(
        self,
        page_id: str,
        *,
        revision: str,
        vectors: list[list[float]],
    ) -> None:
        payload = json.dumps(vectors, separators=(",", ":"))
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                "INSERT INTO documents(page_id, revision, vectors) VALUES (?, ?, ?) "
                "ON CONFLICT(page_id) DO UPDATE SET "
                "revision=excluded.revision, vectors=excluded.vectors",
                (page_id, revision, payload),
            )

    def remove(self, page_id: str) -> None:
        with sqlite3.connect(self.path) as conn:
            conn.execute("DELETE FROM documents WHERE page_id = ?", (page_id,))

    def search(
        self,
        query_vectors: list[list[float]],
        *,
        limit: int = 20,
    ) -> list[tuple[str, float]]:
        with sqlite3.connect(self.path) as conn:
            rows = conn.execute(
                "SELECT page_id, vectors FROM documents ORDER BY page_id"
            ).fetchall()
        scored = [
            (str(page_id), _maxsim(query_vectors, json.loads(vectors)))
            for page_id, vectors in rows
        ]
        return sorted(scored, key=lambda row: (-row[1], row[0]))[: max(0, limit)]

    def stats(self) -> dict[str, int]:
        with sqlite3.connect(self.path) as conn:
            count = int(conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0])
        return {
            "documents": count,
            "bytes": self.path.stat().st_size if self.path.exists() else 0,
        }
