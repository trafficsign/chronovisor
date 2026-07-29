"""Durable semantic indexing jobs shared by ingest and the model service."""

from __future__ import annotations

from chronovisor.timeutil import utc_now as _now

import json
import os
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

from chronovisor.store import CHRONOVISOR_ROOT

SEMANTIC_JOBS_DB = CHRONOVISOR_ROOT / "runtime" / "semantic-jobs.sqlite"




def _iso(value: datetime | None = None) -> str:
    return (value or _now()).isoformat(timespec="milliseconds")


@dataclass(frozen=True)
class SemanticJob:
    job_id: str
    kind: str
    page_id: str
    source_sha256: str
    attempts: int
    created_at: str


def _connect(path: Path = SEMANTIC_JOBS_DB) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=30)
    os.chmod(path, 0o600)
    connection.row_factory = sqlite3.Row
    connection.executescript(
        """
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=FULL;
        CREATE TABLE IF NOT EXISTS jobs (
            job_id TEXT PRIMARY KEY,
            kind TEXT NOT NULL,
            page_id TEXT NOT NULL,
            source_sha256 TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL,
            attempts INTEGER NOT NULL DEFAULT 0,
            next_attempt_at TEXT NOT NULL,
            lease_until TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            completed_at TEXT,
            error TEXT NOT NULL DEFAULT ''
        );
        CREATE INDEX IF NOT EXISTS semantic_jobs_ready_idx
            ON jobs(status, next_attempt_at, created_at);
        CREATE INDEX IF NOT EXISTS semantic_jobs_page_idx
            ON jobs(kind, page_id, status);
        """
    )
    return connection


def enqueue_page(
    page_id: str,
    *,
    source_sha256: str = "",
    path: Path = SEMANTIC_JOBS_DB,
) -> str:
    """Coalesce one page update without losing a newer source hash."""

    if not page_id:
        raise ValueError("semantic job page_id is required")
    connection = _connect(path)
    try:
        connection.execute("BEGIN IMMEDIATE")
        existing = connection.execute(
            """
            SELECT job_id, source_sha256 FROM jobs
            WHERE kind = 'page' AND page_id = ? AND status = 'pending'
            ORDER BY created_at DESC LIMIT 1
            """,
            (page_id,),
        ).fetchone()
        now = _iso()
        if existing is not None:
            job_id = str(existing["job_id"])
            connection.execute(
                """
                UPDATE jobs
                SET source_sha256 = ?, status = 'pending', next_attempt_at = ?,
                    lease_until = NULL, updated_at = ?, error = '',
                    attempts = CASE WHEN source_sha256 != ? THEN 0 ELSE attempts END
                WHERE job_id = ?
                """,
                (source_sha256, now, now, source_sha256, job_id),
            )
        else:
            job_id = uuid.uuid4().hex
            connection.execute(
                """
                INSERT INTO jobs
                (job_id, kind, page_id, source_sha256, status, attempts,
                 next_attempt_at, created_at, updated_at)
                VALUES (?, 'page', ?, ?, 'pending', 0, ?, ?, ?)
                """,
                (job_id, page_id, source_sha256, now, now, now),
            )
        connection.commit()
        return job_id
    except BaseException:
        connection.rollback()
        raise
    finally:
        connection.close()


def enqueue_pages(
    page_ids: Iterable[str],
    *,
    source_hashes: dict[str, str] | None = None,
    path: Path = SEMANTIC_JOBS_DB,
) -> list[str]:
    hashes = source_hashes or {}
    return [
        enqueue_page(page_id, source_sha256=hashes.get(page_id, ""), path=path)
        for page_id in sorted(set(page_ids))
        if page_id
    ]


def enqueue_rebuild(*, path: Path = SEMANTIC_JOBS_DB) -> str:
    connection = _connect(path)
    try:
        connection.execute("BEGIN IMMEDIATE")
        existing = connection.execute(
            """
            SELECT job_id, status FROM jobs
            WHERE kind = 'rebuild' AND status IN ('pending', 'leased')
            ORDER BY created_at DESC LIMIT 1
            """
        ).fetchone()
        now = _iso()
        if existing is not None:
            job_id = str(existing["job_id"])
            if existing["status"] == "pending":
                connection.execute(
                    """
                    UPDATE jobs SET next_attempt_at = ?, updated_at = ?, error = ''
                    WHERE job_id = ?
                    """,
                    (now, now, job_id),
                )
        else:
            job_id = uuid.uuid4().hex
            connection.execute(
                """
                INSERT INTO jobs
                (job_id, kind, page_id, source_sha256, status, attempts,
                 next_attempt_at, created_at, updated_at)
                VALUES (?, 'rebuild', '', '', 'pending', 0, ?, ?, ?)
                """,
                (job_id, now, now, now),
            )
        connection.commit()
        return job_id
    except BaseException:
        connection.rollback()
        raise
    finally:
        connection.close()


def claim_next(
    *,
    kinds: tuple[str, ...] = ("page",),
    lease_seconds: int = 900,
    path: Path = SEMANTIC_JOBS_DB,
) -> SemanticJob | None:
    if not kinds:
        return None
    connection = _connect(path)
    try:
        connection.execute("BEGIN IMMEDIATE")
        now = _iso()
        connection.execute(
            """
            UPDATE jobs
            SET status = 'pending', lease_until = NULL, updated_at = ?,
                error = CASE WHEN error = '' THEN 'stale lease recovered' ELSE error END
            WHERE status = 'leased' AND lease_until IS NOT NULL AND lease_until <= ?
            """,
            (now, now),
        )
        placeholders = ",".join("?" for _ in kinds)
        row = connection.execute(
            f"""
            SELECT * FROM jobs
            WHERE status = 'pending' AND next_attempt_at <= ?
              AND kind IN ({placeholders})
            ORDER BY CASE kind WHEN 'rebuild' THEN 1 ELSE 0 END,
                     created_at, job_id
            LIMIT 1
            """,
            (now, *kinds),
        ).fetchone()
        if row is None:
            connection.commit()
            return None
        lease_until = _iso(_now() + timedelta(seconds=max(30, lease_seconds)))
        connection.execute(
            """
            UPDATE jobs SET status = 'leased', attempts = attempts + 1,
                lease_until = ?, updated_at = ? WHERE job_id = ?
            """,
            (lease_until, now, row["job_id"]),
        )
        connection.commit()
        return SemanticJob(
            job_id=str(row["job_id"]),
            kind=str(row["kind"]),
            page_id=str(row["page_id"]),
            source_sha256=str(row["source_sha256"]),
            attempts=int(row["attempts"]) + 1,
            created_at=str(row["created_at"]),
        )
    except BaseException:
        connection.rollback()
        raise
    finally:
        connection.close()


def complete(job_id: str, *, path: Path = SEMANTIC_JOBS_DB) -> None:
    connection = _connect(path)
    try:
        now = _iso()
        with connection:
            connection.execute(
                """
                UPDATE jobs SET status = 'completed', completed_at = ?,
                    lease_until = NULL, updated_at = ?, error = ''
                WHERE job_id = ? AND status = 'leased'
                """,
                (now, now, job_id),
            )
    finally:
        connection.close()


def fail(
    job_id: str,
    error: str,
    *,
    terminal: bool = False,
    path: Path = SEMANTIC_JOBS_DB,
) -> None:
    connection = _connect(path)
    try:
        row = connection.execute(
            "SELECT attempts FROM jobs WHERE job_id = ?", (job_id,)
        ).fetchone()
        attempts = int(row["attempts"]) if row is not None else 1
        terminal = terminal or attempts >= 5
        delay = min(3_600, 15 * (2 ** max(0, attempts - 1)))
        now = _now()
        with connection:
            connection.execute(
                """
                UPDATE jobs SET status = ?, next_attempt_at = ?, lease_until = NULL,
                    updated_at = ?, error = ?
                WHERE job_id = ?
                """,
                (
                    "dead" if terminal else "pending",
                    _iso(now + timedelta(seconds=delay)),
                    _iso(now),
                    error[:2_000],
                    job_id,
                ),
            )
    finally:
        connection.close()


def job_status(*, path: Path = SEMANTIC_JOBS_DB) -> dict:
    if not path.exists():
        return {
            "status": "missing",
            "path": str(path),
            "counts": {},
            "oldest_pending_at": None,
            "dead_samples": [],
        }
    connection = _connect(path)
    try:
        counts = {
            str(row[0]): int(row[1])
            for row in connection.execute(
                "SELECT status, COUNT(*) FROM jobs GROUP BY status"
            )
        }
        oldest = connection.execute(
            "SELECT MIN(created_at) FROM jobs WHERE status = 'pending'"
        ).fetchone()[0]
        dead = [
            {
                "job_id": str(row["job_id"]),
                "kind": str(row["kind"]),
                "page_id": str(row["page_id"]),
                "attempts": int(row["attempts"]),
                "error": str(row["error"]),
            }
            for row in connection.execute(
                """
                SELECT job_id, kind, page_id, attempts, error FROM jobs
                WHERE status = 'dead' ORDER BY updated_at DESC LIMIT 5
                """
            )
        ]
        return {
            "status": "ok",
            "path": str(path),
            "counts": counts,
            "oldest_pending_at": oldest,
            "dead_samples": dead,
        }
    finally:
        connection.close()


def prune_completed_jobs(
    *, keep_days: int = 7, path: Path = SEMANTIC_JOBS_DB
) -> int:
    if not path.exists():
        return 0
    cutoff = _iso(_now() - timedelta(days=max(1, keep_days)))
    connection = _connect(path)
    try:
        with connection:
            cursor = connection.execute(
                """
                DELETE FROM jobs
                WHERE status = 'completed' AND completed_at IS NOT NULL
                  AND completed_at < ?
                """,
                (cutoff,),
            )
        return max(0, int(cursor.rowcount))
    finally:
        connection.close()


def dump_status_json(*, path: Path = SEMANTIC_JOBS_DB) -> str:
    return json.dumps(job_status(path=path), ensure_ascii=False, sort_keys=True)
