"""Durable, payload-free work queue for Recall distillation backfills."""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class DistillationWorksetError(ValueError):
    """A work item, claim, or outcome violates the durable queue contract."""


@dataclass(frozen=True)
class WorkClaim:
    """An exclusive, time-bounded right to complete one work item."""

    work_id: str
    kind: str
    payload_ref: str
    payload_digest: str
    temporal_split: Mapping[str, Any]
    provenance: Mapping[str, Any]
    priority: int
    attempt: int
    owner: str
    lease_id: str
    lease_expires_at: float


_SCHEMA = """
CREATE TABLE IF NOT EXISTS work_items (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    work_id TEXT NOT NULL UNIQUE,
    kind TEXT NOT NULL,
    payload_ref TEXT NOT NULL,
    payload_digest TEXT NOT NULL,
    temporal_split_json TEXT NOT NULL,
    provenance_json TEXT NOT NULL,
    priority INTEGER NOT NULL,
    watermark_json TEXT NOT NULL,
    state TEXT NOT NULL CHECK (state IN ('ready', 'leased', 'completed', 'quarantined')),
    attempt_count INTEGER NOT NULL DEFAULT 0,
    last_error_class TEXT NOT NULL DEFAULT '',
    lease_id TEXT,
    lease_owner TEXT,
    lease_expires_at REAL,
    completion_ref TEXT NOT NULL DEFAULT '',
    completion_digest TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS work_items_claim_order
    ON work_items(kind, state, priority DESC, sequence ASC);
CREATE INDEX IF NOT EXISTS work_items_expiry
    ON work_items(state, lease_expires_at);
CREATE TABLE IF NOT EXISTS workset_state (
    key TEXT PRIMARY KEY,
    value_json TEXT NOT NULL
);
"""


def _json(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    except (TypeError, ValueError) as exc:
        raise DistillationWorksetError("metadata must be JSON serializable") from exc


def _mapping(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise DistillationWorksetError(f"{field} must be an object")
    return dict(value)


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise DistillationWorksetError(f"{field} must be a non-empty string")
    return value


def _digest(value: object, field: str) -> str:
    digest = _text(value, field).lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise DistillationWorksetError(f"{field} must be a sha256 hex digest")
    return digest


def _item(value: Mapping[str, Any], watermark_json: str) -> tuple[Any, ...]:
    allowed = {
        "work_id",
        "kind",
        "payload_ref",
        "payload_digest",
        "priority",
        "temporal_split",
        "provenance",
    }
    unexpected = set(value).difference(allowed)
    if unexpected:
        raise DistillationWorksetError(
            f"work item has unsupported fields: {sorted(unexpected)}"
        )
    priority = value.get("priority", 0)
    if not isinstance(priority, int) or isinstance(priority, bool):
        raise DistillationWorksetError("priority must be an integer")
    return (
        _text(value.get("work_id"), "work_id"),
        _text(value.get("kind"), "kind"),
        _text(value.get("payload_ref"), "payload_ref"),
        _digest(value.get("payload_digest"), "payload_digest"),
        _json(_mapping(value.get("temporal_split", {}), "temporal_split")),
        _json(_mapping(value.get("provenance", {}), "provenance")),
        priority,
        watermark_json,
    )


class DistillationWorkset:
    """A SQLite/WAL queue shared by local-triad and single-teacher backfills.

    Work items store only immutable references and SHA-256 digests; callers retain
    raw prompt bodies in their existing private ledgers.  The mutating API is
    intentionally limited to ``advance``, ``claim``, and ``commit``.
    """

    def __init__(self, path: Path | str, *, clock: Callable[[], float] = time.time) -> None:
        self.path = Path(path)
        self._clock = clock
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as connection:
            connection.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA busy_timeout=5000")
        return connection

    def advance(self, items: Iterable[Mapping[str, Any]], watermark: Any) -> dict[str, Any]:
        """Idempotently record immutable work and its source progress watermark."""

        watermark_json = _json(watermark)
        records = [_item(item, watermark_json) for item in items]
        if len({record[0] for record in records}) != len(records):
            raise DistillationWorksetError("work_id repeats within one advance")
        now = self._clock()
        inserted = 0
        existing = 0
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                for record in records:
                    prior = connection.execute(
                        """
                        SELECT kind, payload_ref, payload_digest, temporal_split_json,
                               provenance_json, priority
                        FROM work_items WHERE work_id = ?
                        """,
                        (record[0],),
                    ).fetchone()
                    immutable = record[1:7]
                    if prior is None:
                        connection.execute(
                            """
                            INSERT INTO work_items (
                                work_id, kind, payload_ref, payload_digest,
                                temporal_split_json, provenance_json, priority,
                                watermark_json, state, created_at, updated_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'ready', ?, ?)
                            """,
                            (*record, now, now),
                        )
                        inserted += 1
                    elif tuple(prior) != immutable:
                        raise DistillationWorksetError(
                            f"immutable work identity conflict: {record[0]}"
                        )
                    else:
                        existing += 1
                connection.execute(
                    """
                    INSERT INTO workset_state (key, value_json) VALUES ('watermark', ?)
                    ON CONFLICT(key) DO UPDATE SET value_json = excluded.value_json
                    """,
                    (watermark_json,),
                )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        return {"inserted": inserted, "existing": existing, "watermark": watermark}

    def claim(
        self, kind: str, limit: int, owner: str, lease_seconds: float
    ) -> tuple[WorkClaim, ...]:
        """Claim FIFO work of one kind, reclaiming expired leases atomically."""

        kind = _text(kind, "kind")
        owner = _text(owner, "owner")
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
            raise DistillationWorksetError("limit must be a positive integer")
        if (
            not isinstance(lease_seconds, (int, float))
            or isinstance(lease_seconds, bool)
            or lease_seconds <= 0
        ):
            raise DistillationWorksetError("lease_seconds must be positive")
        now = self._clock()
        expires_at = now + float(lease_seconds)
        claims: list[WorkClaim] = []
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.execute(
                    """
                    UPDATE work_items
                    SET state = 'ready', lease_id = NULL, lease_owner = NULL,
                        lease_expires_at = NULL, updated_at = ?
                    WHERE kind = ? AND state = 'leased' AND lease_expires_at <= ?
                    """,
                    (now, kind, now),
                )
                rows = connection.execute(
                    """
                    SELECT work_id, kind, payload_ref, payload_digest,
                           temporal_split_json, provenance_json, priority, attempt_count
                    FROM work_items
                    WHERE kind = ? AND state = 'ready'
                    ORDER BY priority DESC, sequence ASC
                    LIMIT ?
                    """,
                    (kind, limit),
                ).fetchall()
                for row in rows:
                    lease_id = uuid.uuid4().hex
                    result = connection.execute(
                        """
                        UPDATE work_items
                        SET state = 'leased', lease_id = ?, lease_owner = ?,
                            lease_expires_at = ?, attempt_count = attempt_count + 1,
                            updated_at = ?
                        WHERE work_id = ? AND state = 'ready'
                        """,
                        (lease_id, owner, expires_at, now, row["work_id"]),
                    )
                    if result.rowcount != 1:
                        continue
                    claims.append(
                        WorkClaim(
                            work_id=row["work_id"],
                            kind=row["kind"],
                            payload_ref=row["payload_ref"],
                            payload_digest=row["payload_digest"],
                            temporal_split=json.loads(row["temporal_split_json"]),
                            provenance=json.loads(row["provenance_json"]),
                            priority=row["priority"],
                            attempt=row["attempt_count"] + 1,
                            owner=owner,
                            lease_id=lease_id,
                            lease_expires_at=expires_at,
                        )
                    )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        return tuple(claims)

    def commit(
        self,
        claims: Sequence[WorkClaim],
        outcomes: Sequence[Mapping[str, Any]],
    ) -> dict[str, int]:
        """Commit matching leased claims with one single-writer transaction.

        Each outcome has ``status`` of ``completed``, ``retry``, or
        ``quarantined``.  ``completion_ref`` and ``completion_digest`` are only
        valid for completed work; a completion cannot carry an error class.
        Therefore ``invalid_teacher_output`` must be committed as retry or
        quarantine, never as a completed label.
        """

        if len(claims) != len(outcomes):
            raise DistillationWorksetError("claims and outcomes must have equal length")
        normalized = [
            self._outcome(claim, outcome)
            for claim, outcome in zip(claims, outcomes, strict=True)
        ]
        totals = {"completed": 0, "retry": 0, "quarantined": 0}
        now = self._clock()
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                for claim, outcome in zip(claims, normalized, strict=True):
                    row = connection.execute(
                        """
                        SELECT state, lease_id, lease_owner, lease_expires_at,
                               completion_ref, completion_digest
                        FROM work_items WHERE work_id = ?
                        """,
                        (claim.work_id,),
                    ).fetchone()
                    if row is None:
                        raise DistillationWorksetError(f"claim is no longer active: {claim.work_id}")
                    if row["state"] == "completed":
                        if (
                            outcome["status"] != "completed"
                            or row["completion_ref"] != outcome["completion_ref"]
                            or row["completion_digest"] != outcome["completion_digest"]
                        ):
                            raise DistillationWorksetError(
                                f"completion identity conflict: {claim.work_id}"
                            )
                        totals["completed"] += 1
                        continue
                    if row["state"] != "leased":
                        raise DistillationWorksetError(f"claim is no longer active: {claim.work_id}")
                    if (
                        row["lease_id"] != claim.lease_id
                        or row["lease_owner"] != claim.owner
                        or row["lease_expires_at"] is None
                        or row["lease_expires_at"] <= now
                    ):
                        raise DistillationWorksetError(f"claim ownership lost: {claim.work_id}")
                    if outcome["status"] == "completed":
                        connection.execute(
                            """
                            UPDATE work_items
                            SET state = 'completed', completion_ref = ?,
                                completion_digest = ?, last_error_class = '',
                                lease_id = NULL, lease_owner = NULL,
                                lease_expires_at = NULL, updated_at = ?
                            WHERE work_id = ?
                            """,
                            (outcome["completion_ref"], outcome["completion_digest"], now, claim.work_id),
                        )
                    else:
                        state = "ready" if outcome["status"] == "retry" else "quarantined"
                        connection.execute(
                            """
                            UPDATE work_items
                            SET state = ?, last_error_class = ?, lease_id = NULL,
                                lease_owner = NULL, lease_expires_at = NULL,
                                updated_at = ?
                            WHERE work_id = ?
                            """,
                            (state, outcome["error_class"], now, claim.work_id),
                        )
                    totals[outcome["status"]] += 1
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        return totals

    def status(self, kind: str | None = None) -> dict[str, int]:
        """Return state counters without exposing work payloads."""

        if kind is not None:
            kind = _text(kind, "kind")
        where = " WHERE kind = ?" if kind is not None else ""
        parameters: tuple[Any, ...] = (kind,) if kind is not None else ()
        counts = {
            "ready": 0,
            "leased": 0,
            "completed": 0,
            "quarantined": 0,
        }
        with closing(self._connect()) as connection:
            rows = connection.execute(
                f"SELECT state, COUNT(*) AS count FROM work_items{where} GROUP BY state",
                parameters,
            ).fetchall()
        for row in rows:
            counts[str(row["state"])] = int(row["count"])
        counts["backlog"] = counts["ready"] + counts["leased"]
        counts["total"] = sum(
            counts[state] for state in ("ready", "leased", "completed", "quarantined")
        )
        return counts

    @staticmethod
    def _outcome(claim: WorkClaim, value: Mapping[str, Any]) -> dict[str, str]:
        allowed = {"status", "error_class", "completion_ref", "completion_digest"}
        unexpected = set(value).difference(allowed)
        if unexpected:
            raise DistillationWorksetError(
                f"outcome has unsupported fields: {sorted(unexpected)}"
            )
        status = _text(value.get("status"), "status")
        if status not in {"completed", "retry", "quarantined"}:
            raise DistillationWorksetError("outcome status is invalid")
        error_class = value.get("error_class", "")
        if error_class and not isinstance(error_class, str):
            raise DistillationWorksetError("error_class must be a string")
        if status == "completed" and error_class:
            raise DistillationWorksetError("completed work cannot have an error_class")
        if status == "completed":
            return {
                "status": status,
                "error_class": "",
                "completion_ref": _text(value.get("completion_ref"), "completion_ref"),
                "completion_digest": _digest(
                    value.get("completion_digest"), "completion_digest"
                ),
            }
        if not error_class:
            raise DistillationWorksetError("non-completed work requires error_class")
        if "completion_ref" in value or "completion_digest" in value:
            raise DistillationWorksetError("failed work cannot include a completion")
        return {
            "status": status,
            "error_class": error_class,
            "completion_ref": "",
            "completion_digest": "",
        }
