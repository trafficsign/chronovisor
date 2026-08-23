"""Durable, payload-free work queue for Recall distillation backfills."""

from __future__ import annotations

import json
import math
import os
import re
import sqlite3
import stat
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


_MAX_TEXT_BYTES = 256
_MAX_JSON_BYTES = 4_096
_MAX_METADATA_DEPTH = 3
_MAX_METADATA_ITEMS = 32
_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}\Z")
_REFERENCE_RE = re.compile(
    r"(?:candidate-snapshot|candidate-ledger|label-ledger):"
    r"[A-Za-z0-9_.-]{1,128}(?::[A-Za-z0-9_.-]{1,128})?\Z"
)
_METADATA_KEY_RE = re.compile(r"[A-Za-z][A-Za-z0-9_.-]{0,63}\Z")
_METADATA_VALUE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:+,@-]{0,255}\Z")
_ROUTE_VALUE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:+,@/-]{0,255}\Z")
_ERROR_CLASS_RE = re.compile(r"[a-z][a-z0-9_.-]{0,63}\Z")
_SENSITIVE_MARKERS = (
    "secret",
    "token",
    "password",
    "credential",
    "api_key",
    "authorization",
    "bearer",
)
_PATH_RE = re.compile(
    r"(?:^[/~]|^[A-Za-z]:[\\/]|(?:^|[\\/])\.{1,2}(?:[\\/]|$)|"
    r"(?:^|:)file://|(?:^|:)https?://)",
    re.IGNORECASE,
)
_SENSITIVE_KEY_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:secret|token|password|credential|api_key|"
    r"authorization|bearer|path|file|filepath|directory|prompt|payload|body|content)"
    r"(?![A-Za-z0-9])"
)


def _json(value: Any, field: str = "metadata") -> str:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise DistillationWorksetError(f"{field} must be JSON serializable") from exc
    if len(encoded.encode("utf-8")) > _MAX_JSON_BYTES:
        raise DistillationWorksetError(f"{field} exceeds payload-free size limit")
    return encoded


def _identifier(value: object, field: str) -> str:
    text = _text(value, field)
    if (
        len(text.encode("utf-8")) > _MAX_TEXT_BYTES
        or _IDENTIFIER_RE.fullmatch(text) is None
    ):
        raise DistillationWorksetError(f"{field} must be a bounded safe identifier")
    return text


def _reference(value: object, field: str) -> str:
    text = _text(value, field)
    if (
        len(text.encode("utf-8")) > _MAX_TEXT_BYTES
        or _REFERENCE_RE.fullmatch(text) is None
        or any(marker in text.lower() for marker in _SENSITIVE_MARKERS)
    ):
        raise DistillationWorksetError(f"{field} must be a safe ledger reference")
    return text


def _metadata_key(value: object, field: str) -> str:
    if not isinstance(value, str) or _METADATA_KEY_RE.fullmatch(value) is None:
        raise DistillationWorksetError(f"{field} contains an unsafe metadata key")
    if _SENSITIVE_KEY_RE.search(value.lower()):
        raise DistillationWorksetError(f"{field} contains a sensitive metadata key")
    return value


def _metadata_value(value: object, field: str, *, key: str = "", depth: int = 0) -> Any:
    if depth > _MAX_METADATA_DEPTH:
        raise DistillationWorksetError(f"{field} metadata is too deeply nested")
    if isinstance(value, Mapping):
        if len(value) > _MAX_METADATA_ITEMS:
            raise DistillationWorksetError(f"{field} metadata has too many entries")
        return {
            _metadata_key(raw_key, field): _metadata_value(
                raw_value,
                field,
                key=str(raw_key),
                depth=depth + 1,
            )
            for raw_key, raw_value in value.items()
        }
    if isinstance(value, (list, tuple)):
        if len(value) > _MAX_METADATA_ITEMS:
            raise DistillationWorksetError(f"{field} metadata has too many entries")
        return [
            _metadata_value(item, field, key=key, depth=depth + 1) for item in value
        ]
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        if abs(value) > 1_000_000_000:
            raise DistillationWorksetError(f"{field} metadata integer is too large")
        return value
    if isinstance(value, float):
        if not math.isfinite(value) or abs(value) > 1_000_000_000_000:
            raise DistillationWorksetError(f"{field} metadata number is invalid")
        return value
    if isinstance(value, str):
        if len(value.encode("utf-8")) > _MAX_TEXT_BYTES or any(
            ord(character) < 0x20 or ord(character) == 0x7F for character in value
        ):
            raise DistillationWorksetError(f"{field} metadata string is unsafe")
        lowered = value.lower()
        if any(marker in lowered for marker in _SENSITIVE_MARKERS) or _PATH_RE.search(
            value
        ):
            raise DistillationWorksetError(
                f"{field} metadata contains secret or path data"
            )
        if value and (
            _ROUTE_VALUE_RE.fullmatch(value) is None
            if key == "route"
            else _METADATA_VALUE_RE.fullmatch(value) is None
        ):
            raise DistillationWorksetError(
                f"{field} metadata string is not a safe token"
            )
        if key == "route" and ("//" in value or "/../" in value or "/./" in value):
            raise DistillationWorksetError(f"{field} metadata route is unsafe")
        return value
    raise DistillationWorksetError(f"{field} metadata contains unsupported data")


def _metadata_json(value: object, field: str) -> str:
    return _json(_metadata_value(value, field), field)


def _finite(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DistillationWorksetError(f"{field} must be a finite number")
    try:
        parsed = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise DistillationWorksetError(f"{field} must be a finite number") from exc
    if not math.isfinite(parsed):
        raise DistillationWorksetError(f"{field} must be a finite number")
    return parsed


def _error_class(value: object) -> str:
    if not isinstance(value, str) or _ERROR_CLASS_RE.fullmatch(value) is None:
        raise DistillationWorksetError("error_class must be a bounded safe token")
    if any(marker in value.lower() for marker in _SENSITIVE_MARKERS):
        raise DistillationWorksetError("error_class must not contain secret data")
    return value


def _secure_regular_file(path: Path) -> None:
    """Reject links/non-regular files and narrow existing SQLite files to 0600."""

    try:
        initial = os.lstat(path)
    except FileNotFoundError:
        return
    if stat.S_ISLNK(initial.st_mode):
        raise DistillationWorksetError(
            f"SQLite path must not be a symlink: {path.name}"
        )
    if not stat.S_ISREG(initial.st_mode):
        raise DistillationWorksetError(
            f"SQLite path must be a regular file: {path.name}"
        )
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise DistillationWorksetError(
            f"cannot securely open SQLite path: {path.name}"
        ) from exc
    try:
        current = os.fstat(descriptor)
        if not stat.S_ISREG(current.st_mode):
            raise DistillationWorksetError(
                f"SQLite path must be a regular file: {path.name}"
            )
        os.fchmod(descriptor, 0o600)
    except OSError as exc:
        raise DistillationWorksetError(
            f"cannot secure SQLite path: {path.name}"
        ) from exc
    finally:
        os.close(descriptor)


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
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
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
        _identifier(value.get("work_id"), "work_id"),
        _identifier(value.get("kind"), "kind"),
        _reference(value.get("payload_ref"), "payload_ref"),
        _digest(value.get("payload_digest"), "payload_digest"),
        _metadata_json(
            _mapping(value.get("temporal_split", {}), "temporal_split"),
            "temporal_split",
        ),
        _metadata_json(
            _mapping(value.get("provenance", {}), "provenance"), "provenance"
        ),
        priority,
        watermark_json,
    )


def _same_temporal_split_except_plan(
    prior_json: str, current_json: str, *, allow_split_change: bool = False
) -> bool:
    try:
        prior = _metadata_value(json.loads(prior_json), "prior temporal_split")
        current = _metadata_value(json.loads(current_json), "current temporal_split")
    except (
        TypeError,
        ValueError,
        json.JSONDecodeError,
        RecursionError,
        DistillationWorksetError,
    ):
        return False
    if not isinstance(prior, dict) or not isinstance(current, dict):
        return False
    prior_plan = prior.pop("split_plan_id", None)
    current_plan = current.pop("split_plan_id", None)
    try:
        prior_digest = _digest(prior_plan, "prior split_plan_id")
        current_digest = _digest(current_plan, "current split_plan_id")
        if (
            prior_plan != prior_digest
            or current_plan != current_digest
            or prior_digest == current_digest
        ):
            return False
        if allow_split_change:
            prior_split = prior.pop("split", None)
            current_split = current.pop("split", None)
            if (
                prior_split not in {"train", "validation", "test", "embargo"}
                or current_split not in {"train", "validation", "test", "embargo"}
                or "as_of" not in prior
                or "as_of" not in current
                or "group_id" not in prior
                or "group_id" not in current
            ):
                return False
        return _metadata_json(prior, "prior temporal_split") == _metadata_json(
            current, "current temporal_split"
        )
    except DistillationWorksetError:
        return False


class DistillationWorkset:
    """A SQLite/WAL queue shared by local-triad and single-teacher backfills.

    Work items store only immutable references and SHA-256 digests; callers retain
    raw prompt bodies in their existing private ledgers.  The mutating API is
    intentionally limited to ``advance``, ``claim``, ``release_unattempted``,
    and ``commit``.
    """

    def __init__(
        self, path: Path | str, *, clock: Callable[[], float] = time.time
    ) -> None:
        self.path = Path(path)
        self._clock = clock
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as connection:
            connection.executescript(_SCHEMA)
            self._secure_sqlite_files()

    def _now(self) -> float:
        return _finite(self._clock(), "clock now")

    def _secure_sqlite_files(self) -> None:
        for path in (
            self.path,
            self.path.with_name(f"{self.path.name}-wal"),
            self.path.with_name(f"{self.path.name}-shm"),
        ):
            _secure_regular_file(path)

    def _connect(self) -> sqlite3.Connection:
        self._secure_sqlite_files()
        connection = sqlite3.connect(self.path, isolation_level=None)
        try:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute("PRAGMA busy_timeout=5000")
            self._secure_sqlite_files()
            return connection
        except Exception:
            connection.close()
            raise

    def advance(
        self, items: Iterable[Mapping[str, Any]], watermark: Any
    ) -> dict[str, Any]:
        """Idempotently record immutable work and its source progress watermark."""

        watermark_json = _metadata_json(watermark, "watermark")
        records = [_item(item, watermark_json) for item in items]
        if len({record[0] for record in records}) != len(records):
            raise DistillationWorksetError("work_id repeats within one advance")
        now = self._now()
        inserted = 0
        existing = 0
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            committed = False
            try:
                for record in records:
                    prior = connection.execute(
                        """
                        SELECT kind, payload_ref, payload_digest, temporal_split_json,
                               provenance_json, priority, state
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
                    elif tuple(prior[:6]) != immutable:
                        same_non_temporal = (
                            tuple(prior[:3]) == immutable[:3]
                            and tuple(prior[4:6]) == immutable[4:6]
                        )
                        unfinished = prior["state"] in {"ready", "quarantined"}
                        if not (
                            same_non_temporal
                            and _same_temporal_split_except_plan(
                                prior[3], immutable[3], allow_split_change=unfinished
                            )
                        ):
                            raise DistillationWorksetError(
                                f"immutable work identity conflict: {record[0]}"
                            )
                        if unfinished:
                            # Cohort plan rotations may rebind an uncompleted split.
                            connection.execute(
                                "UPDATE work_items SET temporal_split_json = ?, "
                                "updated_at = ? WHERE work_id = ?",
                                (immutable[3], now, record[0]),
                            )
                        elif prior["state"] != "completed":
                            raise DistillationWorksetError(
                                f"immutable work identity conflict: {record[0]}"
                            )
                        existing += 1
                    else:
                        existing += 1
                connection.execute(
                    """
                    INSERT INTO workset_state (key, value_json) VALUES ('watermark', ?)
                    ON CONFLICT(key) DO UPDATE SET value_json = excluded.value_json
                    """,
                    (watermark_json,),
                )
                self._secure_sqlite_files()
                connection.execute("COMMIT")
                committed = True
            except Exception:
                if not committed:
                    connection.execute("ROLLBACK")
                raise
        return {"inserted": inserted, "existing": existing, "watermark": watermark}

    def watermark(self) -> Any | None:
        """Read and validate the durable source progress watermark."""

        try:
            with closing(self._connect()) as connection:
                row = connection.execute(
                    "SELECT value_json FROM workset_state WHERE key = 'watermark'"
                ).fetchone()
                if row is None:
                    return None
                value_json = row["value_json"]
        except DistillationWorksetError:
            raise
        except (IndexError, KeyError, sqlite3.Error, TypeError) as exc:
            raise DistillationWorksetError("cannot read watermark") from exc

        if not isinstance(value_json, str):
            raise DistillationWorksetError("watermark state is corrupted")
        try:
            value = json.loads(value_json)
        except (RecursionError, json.JSONDecodeError) as exc:
            raise DistillationWorksetError("watermark state is invalid JSON") from exc
        normalized = _metadata_value(value, "watermark")
        _json(normalized, "watermark")
        return normalized

    def claim(
        self, kind: str, limit: int, owner: str, lease_seconds: float
    ) -> tuple[WorkClaim, ...]:
        """Claim FIFO work of one kind, reclaiming expired leases atomically."""

        kind = _identifier(kind, "kind")
        owner = _identifier(owner, "owner")
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
            raise DistillationWorksetError("limit must be a positive integer")
        lease_seconds_float = _finite(lease_seconds, "lease_seconds")
        if lease_seconds_float <= 0:
            raise DistillationWorksetError("lease_seconds must be positive")
        now = self._now()
        expires_at = now + lease_seconds_float
        if not math.isfinite(expires_at):
            raise DistillationWorksetError("lease expiry must be finite")
        claims: list[WorkClaim] = []
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            committed = False
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
                self._secure_sqlite_files()
                connection.execute("COMMIT")
                committed = True
            except Exception:
                if not committed:
                    connection.execute("ROLLBACK")
                raise
        return tuple(claims)

    def release_unattempted(self, claims: Sequence[WorkClaim]) -> int:
        """Return owned leases to ready without consuming an attempt."""

        work_ids = [claim.work_id for claim in claims]
        lease_ids = [claim.lease_id for claim in claims]
        if len(set(work_ids)) != len(work_ids):
            raise DistillationWorksetError("release contains duplicate work_id")
        if len(set(lease_ids)) != len(lease_ids):
            raise DistillationWorksetError("release contains duplicate lease_id")
        if not claims:
            return 0
        now = self._now()
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            committed = False
            try:
                for claim in claims:
                    row = connection.execute(
                        """
                        SELECT kind, payload_ref, payload_digest,
                               temporal_split_json, provenance_json, priority,
                               state, lease_id, lease_owner, lease_expires_at,
                               attempt_count
                        FROM work_items WHERE work_id = ?
                        """,
                        (claim.work_id,),
                    ).fetchone()
                    if row is None or row["state"] != "leased":
                        raise DistillationWorksetError(
                            f"claim is no longer active: {claim.work_id}"
                        )
                    if (
                        (
                            row["kind"],
                            row["payload_ref"],
                            row["payload_digest"],
                            row["temporal_split_json"],
                            row["provenance_json"],
                            row["priority"],
                        )
                        != (
                            claim.kind,
                            claim.payload_ref,
                            claim.payload_digest,
                            _metadata_json(claim.temporal_split, "temporal_split"),
                            _metadata_json(claim.provenance, "provenance"),
                            claim.priority,
                        )
                        or row["lease_expires_at"] != claim.lease_expires_at
                        or row["lease_id"] != claim.lease_id
                        or row["lease_owner"] != claim.owner
                        or row["lease_expires_at"] is None
                        or row["lease_expires_at"] <= now
                        or row["attempt_count"] != claim.attempt
                        or row["attempt_count"] < 1
                    ):
                        raise DistillationWorksetError(
                            f"claim ownership lost: {claim.work_id}"
                        )
                    connection.execute(
                        """
                        UPDATE work_items
                        SET state = 'ready', attempt_count = attempt_count - 1,
                            lease_id = NULL, lease_owner = NULL,
                            lease_expires_at = NULL, updated_at = ?
                        WHERE work_id = ?
                        """,
                        (now, claim.work_id),
                    )
                self._secure_sqlite_files()
                connection.execute("COMMIT")
                committed = True
            except Exception:
                if not committed:
                    connection.execute("ROLLBACK")
                raise
        return len(claims)

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
        work_ids = [claim.work_id for claim in claims]
        lease_ids = [claim.lease_id for claim in claims]
        if len(set(work_ids)) != len(work_ids):
            raise DistillationWorksetError("commit contains duplicate work_id")
        if len(set(lease_ids)) != len(lease_ids):
            raise DistillationWorksetError("commit contains duplicate lease_id")
        normalized = [
            self._outcome(claim, outcome)
            for claim, outcome in zip(claims, outcomes, strict=True)
        ]
        totals = {"completed": 0, "retry": 0, "quarantined": 0}
        now = self._now()
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            committed = False
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
                        raise DistillationWorksetError(
                            f"claim is no longer active: {claim.work_id}"
                        )
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
                        raise DistillationWorksetError(
                            f"claim is no longer active: {claim.work_id}"
                        )
                    if (
                        row["lease_id"] != claim.lease_id
                        or row["lease_owner"] != claim.owner
                        or row["lease_expires_at"] is None
                        or row["lease_expires_at"] <= now
                    ):
                        raise DistillationWorksetError(
                            f"claim ownership lost: {claim.work_id}"
                        )
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
                            (
                                outcome["completion_ref"],
                                outcome["completion_digest"],
                                now,
                                claim.work_id,
                            ),
                        )
                    else:
                        state = (
                            "ready" if outcome["status"] == "retry" else "quarantined"
                        )
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
                self._secure_sqlite_files()
                connection.execute("COMMIT")
                committed = True
            except Exception:
                if not committed:
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
        error_class_value = value.get("error_class", "")
        error_class = _error_class(error_class_value) if error_class_value != "" else ""
        if status == "completed" and error_class:
            raise DistillationWorksetError("completed work cannot have an error_class")
        if status == "completed":
            return {
                "status": status,
                "error_class": "",
                "completion_ref": _reference(
                    value.get("completion_ref"), "completion_ref"
                ),
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
