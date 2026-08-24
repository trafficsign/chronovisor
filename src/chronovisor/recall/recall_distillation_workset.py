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

from chronovisor.core.canonical_json import canonical_json_sha256_strict


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
    stage TEXT NOT NULL DEFAULT 'snapshot',
    state TEXT NOT NULL CHECK (state IN ('ready', 'leased', 'completed', 'quarantined')),
    attempt_count INTEGER NOT NULL DEFAULT 0,
    last_error_class TEXT NOT NULL DEFAULT '',
    lease_id TEXT,
    lease_owner TEXT,
    lease_expires_at REAL,
    next_attempt_at REAL,
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
CREATE TABLE IF NOT EXISTS workset_receipts (
    generation INTEGER PRIMARY KEY CHECK (generation > 0),
    previous_sha256 TEXT NOT NULL,
    operation TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    receipt_sha256 TEXT NOT NULL UNIQUE
);
"""


_MAX_TEXT_BYTES = 256
_MAX_JSON_BYTES = 4_096
_MAX_RECEIPT_JSON_BYTES = 3 * _MAX_JSON_BYTES
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
_RECEIPT_STATES = ("ready", "leased", "completed", "quarantined")
_RECEIPT_OPERATIONS = {
    "advance",
    "claim_reclaim",
    "claim",
    "release",
    "commit",
}
_STAGES = (
    "snapshot",
    "teacher",
    "counterfactual",
    "retry_wait",
    "dataset",
    "evaluation",
)
_PROGRESS_KEYS = {"cursor", "ledger_heads", "provenance", "progress_kind"}
_FAIRNESS_AGE_SECONDS = 60


def _json(
    value: Any,
    field: str = "metadata",
    *,
    max_bytes: int = _MAX_JSON_BYTES,
) -> str:
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
    if len(encoded.encode("utf-8")) > max_bytes:
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


def _stage_for_kind(kind: str) -> str:
    if "counterfactual" in kind:
        return "counterfactual"
    if "dataset" in kind:
        return "dataset"
    if "evaluation" in kind or "eval" in kind:
        return "evaluation"
    if "teacher" in kind or kind == "ox":
        return "teacher"
    return "snapshot"


def _strict_progress(value: object, field: str = "progress") -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _PROGRESS_KEYS:
        raise DistillationWorksetError(f"{field} must be a strict progress object")
    cursor = _metadata_value(value["cursor"], f"{field}.cursor")
    _validate_progress_cursor(cursor)
    heads = _mapping(value["ledger_heads"], f"{field}.ledger_heads")
    normalized_heads = {
        _metadata_key(key, f"{field}.ledger_heads"): (
            "" if head == "" else _digest(head, f"{field}.ledger_heads")
        )
        for key, head in heads.items()
    }
    provenance = _metadata_value(value["provenance"], f"{field}.provenance")
    kind = _identifier(value["progress_kind"], f"{field}.progress_kind")
    result = {
        "cursor": cursor,
        "ledger_heads": normalized_heads,
        "provenance": provenance,
        "progress_kind": kind,
    }
    _json(result, field)
    return result


def _validate_progress_cursor(value: object) -> None:
    if isinstance(value, bool):
        raise DistillationWorksetError("progress cursor is invalid")
    if isinstance(value, float) and not math.isfinite(value):
        raise DistillationWorksetError("progress cursor is invalid")
    if isinstance(value, Mapping):
        for child in value.values():
            _validate_progress_cursor(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _validate_progress_cursor(child)


def _cursor_relation(before: Any, after: Any) -> int:
    """Return -1/0/1 for a safe cursor transition; reject incomparable rewrites."""

    if _json(before, "progress cursor") == _json(after, "progress cursor"):
        return 0
    if (
        isinstance(before, (int, float))
        and not isinstance(before, bool)
        and isinstance(after, (int, float))
        and not isinstance(after, bool)
    ):
        return 1 if after > before else -1
    if isinstance(before, Mapping) and isinstance(after, Mapping):
        if set(before) != set(after):
            raise DistillationWorksetError("progress cursor identity rewrite is invalid")
        numeric: list[str] = []
        for key in before:
            left, right = before[key], after[key]
            if isinstance(left, bool) or isinstance(right, bool):
                raise DistillationWorksetError("progress cursor is invalid")
            if isinstance(left, (int, float)) and isinstance(right, (int, float)):
                if not math.isfinite(left) or not math.isfinite(right):
                    raise DistillationWorksetError("progress cursor is invalid")
                numeric.append(key)
            elif _json(left, "progress cursor") != _json(right, "progress cursor"):
                raise DistillationWorksetError("progress cursor identity rewrite is invalid")
        if numeric:
            if any(after[key] < before[key] for key in numeric):
                return -1
            if any(after[key] > before[key] for key in numeric):
                return 1
    raise DistillationWorksetError("progress cursor identity rewrite is invalid")


def _is_legacy_ox_progress_upgrade(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> bool:
    """Allow the exact OX upgrade; legacy has no split, so a digest is admissible."""

    cursor = before["cursor"]
    heads = before["ledger_heads"]
    provenance = before["provenance"]
    target_cursor = after["cursor"]
    target_heads = after["ledger_heads"]
    target_provenance = after["provenance"]
    return (
        before["progress_kind"] == "ox-label-v2"
        and isinstance(cursor, Mapping)
        and set(cursor) == {"labels"}
        and isinstance(cursor["labels"], int)
        and not isinstance(cursor["labels"], bool)
        and cursor["labels"] >= 0
        and heads.keys() == {"labels"}
        and provenance.keys() == {"profile", "profile_contract_id"}
        and provenance["profile"] == "ox-alpha-single-v1"
        and after["progress_kind"] == "ox-workset-v2"
        and isinstance(target_cursor, Mapping)
        and set(target_cursor) == {"candidate_count", "label_count", "revision_epoch"}
        and all(
            isinstance(target_cursor[key], int)
            and not isinstance(target_cursor[key], bool)
            and target_cursor[key] >= 0
            for key in target_cursor
        )
        and target_cursor["label_count"] == cursor["labels"]
        and target_cursor["revision_epoch"] == 0
        and target_heads.keys() == {"candidate", "labels"}
        and target_heads["labels"] == heads["labels"]
        and target_provenance.keys()
        == {"profile", "profile_contract_id", "probe_revision", "split_plan_id"}
        and target_provenance["profile"] == provenance["profile"]
        and target_provenance["profile_contract_id"]
        == provenance["profile_contract_id"]
        and isinstance(provenance["profile_contract_id"], str)
        and re.fullmatch(r"[0-9a-f]{64}", provenance["profile_contract_id"])
        is not None
        and target_provenance["probe_revision"] == "single-teacher-repeat-v2"
        and (
            target_provenance["split_plan_id"] == ""
            or (
                isinstance(target_provenance["split_plan_id"], str)
                and re.fullmatch(r"[0-9a-f]{64}", target_provenance["split_plan_id"])
                is not None
            )
        )
    )


def _validate_progress_transition(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> None:
    if _is_legacy_ox_progress_upgrade(before, after):
        return
    relation = _cursor_relation(before["cursor"], after["cursor"])
    if relation < 0:
        raise DistillationWorksetError("progress cursor regressed")
    if relation == 0 and _json(before, "progress") != _json(after, "progress"):
        raise DistillationWorksetError("progress cursor identity rewrite is invalid")


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


def _selection_sha256(values: Iterable[Mapping[str, Any]]) -> str:
    try:
        return canonical_json_sha256_strict(
            sorted(
                (dict(value) for value in values), key=lambda value: value["work_id"]
            )
        )
    except (KeyError, OverflowError, RecursionError, TypeError, ValueError) as exc:
        raise DistillationWorksetError("receipt selection is invalid") from exc


def _claim_selection_sha256(claims: Sequence[WorkClaim]) -> str:
    return _selection_sha256(
        {
            "work_id": claim.work_id,
            "kind": claim.kind,
            "payload_digest": claim.payload_digest,
            "temporal_split": claim.temporal_split,
            "provenance": claim.provenance,
            "attempt": claim.attempt,
            "owner": claim.owner,
            "lease_id": claim.lease_id,
        }
        for claim in claims
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
            columns = {
                str(row["name"])
                for row in connection.execute("PRAGMA table_info(work_items)")
            }
            if "next_attempt_at" not in columns:
                connection.execute(
                    "ALTER TABLE work_items ADD COLUMN next_attempt_at REAL"
                )
            if "stage" not in columns:
                connection.execute(
                    "ALTER TABLE work_items ADD COLUMN stage TEXT NOT NULL DEFAULT 'snapshot'"
                )
                connection.execute(
                    "UPDATE work_items SET stage = CASE "
                    "WHEN kind LIKE '%counterfactual%' THEN 'counterfactual' "
                    "WHEN kind LIKE '%dataset%' THEN 'dataset' "
                    "WHEN kind LIKE '%evaluation%' OR kind LIKE '%eval%' THEN 'evaluation' "
                    "WHEN kind LIKE '%teacher%' OR kind = 'ox' THEN 'teacher' "
                    "ELSE 'snapshot' END"
                )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS work_items_retry_due "
                "ON work_items(kind, state, next_attempt_at, priority DESC, sequence ASC)"
            )
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

    @staticmethod
    def _snapshot(connection: sqlite3.Connection) -> dict[str, Any]:
        counts = {state: 0 for state in _RECEIPT_STATES}
        rows = connection.execute(
            "SELECT state, COUNT(*) AS count FROM work_items GROUP BY state"
        ).fetchall()
        for row in rows:
            state = row["state"]
            if state not in counts:
                raise DistillationWorksetError("workset state is corrupted")
            counts[state] = int(row["count"])
        row = connection.execute(
            "SELECT value_json FROM workset_state WHERE key = 'watermark'"
        ).fetchone()
        if row is None:
            watermark = None
        else:
            value_json = row["value_json"]
            if not isinstance(value_json, str):
                raise DistillationWorksetError("watermark state is corrupted")
            try:
                watermark = json.loads(value_json)
            except (RecursionError, json.JSONDecodeError) as exc:
                raise DistillationWorksetError(
                    "watermark state is invalid JSON"
                ) from exc
            watermark = _metadata_value(watermark, "watermark")
            _json(watermark, "watermark")
        return {"counts": counts, "watermark": watermark}

    @staticmethod
    def _progress(connection: sqlite3.Connection) -> dict[str, Any] | None:
        row = connection.execute(
            "SELECT value_json FROM workset_state WHERE key = 'progress'"
        ).fetchone()
        if row is None:
            return None
        try:
            return _strict_progress(json.loads(row["value_json"]))
        except (KeyError, TypeError, RecursionError, json.JSONDecodeError) as exc:
            raise DistillationWorksetError("progress state is invalid JSON") from exc

    @classmethod
    def _store_progress(
        cls, connection: sqlite3.Connection, progress: Mapping[str, Any]
    ) -> dict[str, Any]:
        normalized = _strict_progress(progress)
        before = cls._progress(connection)
        if before is not None:
            _validate_progress_transition(before, normalized)
        connection.execute(
            "INSERT INTO workset_state (key, value_json) VALUES ('progress', ?) "
            "ON CONFLICT(key) DO UPDATE SET value_json = excluded.value_json",
            (_json(normalized, "progress"),),
        )
        return normalized

    @staticmethod
    def _count_delta(
        before: Mapping[str, Any], after: Mapping[str, Any]
    ) -> dict[str, int]:
        before_counts = before["counts"]
        after_counts = after["counts"]
        if set(before_counts) != set(_RECEIPT_STATES) or set(after_counts) != set(
            _RECEIPT_STATES
        ):
            raise DistillationWorksetError("receipt state counts are corrupted")
        return {
            state: int(after_counts[state]) - int(before_counts[state])
            for state in _RECEIPT_STATES
        }

    @staticmethod
    def _receipt_head(
        connection: sqlite3.Connection,
    ) -> tuple[int, str]:
        row = connection.execute(
            "SELECT generation, receipt_sha256 FROM workset_receipts "
            "ORDER BY generation DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return 0, ""
        generation = row["generation"]
        digest = row["receipt_sha256"]
        if (
            isinstance(generation, bool)
            or not isinstance(generation, int)
            or generation < 1
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise DistillationWorksetError("workset receipt head is corrupted")
        return generation, digest

    @staticmethod
    def _validate_stages(connection: sqlite3.Connection) -> None:
        if connection.execute(
            "SELECT 1 FROM work_items WHERE stage NOT IN "
            "('snapshot', 'teacher', 'counterfactual', 'retry_wait', 'dataset', 'evaluation') "
            "LIMIT 1"
        ).fetchone() is not None:
            raise DistillationWorksetError("workset stage is corrupted")

    @classmethod
    def _append_receipt(
        cls,
        connection: sqlite3.Connection,
        operation: str,
        before: Mapping[str, Any],
        after: Mapping[str, Any],
        details: Mapping[str, Any],
        *,
        before_progress: Mapping[str, Any] | None = None,
        after_progress: Mapping[str, Any] | None = None,
    ) -> None:
        if operation not in _RECEIPT_OPERATIONS:
            raise DistillationWorksetError("receipt operation is invalid")
        delta = cls._count_delta(before, after)
        payload: dict[str, Any] = {
            "before": {
                "counts": dict(before["counts"]),
                "watermark": before["watermark"],
            },
            "after": {
                "counts": dict(after["counts"]),
                "watermark": after["watermark"],
            },
            "delta": delta,
            "details": dict(details),
        }
        if before_progress is None and after_progress is None:
            before_progress = after_progress = cls._progress(connection)
        if before_progress is not None or after_progress is not None:
            bootstrap = before_progress is None
            if after_progress is None:
                raise DistillationWorksetError("receipt progress is incomplete")
            payload["version"] = 2
            if bootstrap:
                payload["bootstrap"] = True
            payload["before"]["progress"] = (
                None if bootstrap else _strict_progress(before_progress)
            )
            payload["after"]["progress"] = _strict_progress(after_progress)
        payload_json = _json(
            payload,
            "workset receipt",
            max_bytes=_MAX_RECEIPT_JSON_BYTES,
        )
        generation, previous = cls._receipt_head(connection)
        generation += 1
        envelope = {
            "generation": generation,
            "previous_sha256": previous,
            "operation": operation,
            "payload": payload,
        }
        receipt_sha256 = canonical_json_sha256_strict(envelope)
        connection.execute(
            "INSERT INTO workset_receipts "
            "(generation, previous_sha256, operation, payload_json, receipt_sha256) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                generation,
                previous,
                operation,
                payload_json,
                receipt_sha256,
            ),
        )

    @classmethod
    def _validate_receipt_payload(
        cls,
        operation: str,
        payload: object,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, int], dict[str, Any]]:
        if operation not in _RECEIPT_OPERATIONS or not isinstance(payload, Mapping):
            raise DistillationWorksetError("workset receipt payload is corrupted")
        allowed = {"before", "after", "delta", "details"}
        version = payload.get("version", 1)
        v2_allowed = {*allowed, "version"}
        if payload.get("bootstrap") is True:
            v2_allowed.add("bootstrap")
        if set(payload) != (allowed if version == 1 else v2_allowed):
            raise DistillationWorksetError("workset receipt payload is corrupted")
        if version not in (1, 2):
            raise DistillationWorksetError("workset receipt payload is corrupted")

        bootstrap = payload.get("bootstrap") is True

        def snapshot(value: object, *, before: bool) -> dict[str, Any]:
            expected = {"counts", "watermark"} if version == 1 else {"counts", "watermark", "progress"}
            if not isinstance(value, Mapping) or set(value) != expected:
                raise DistillationWorksetError("workset receipt snapshot is corrupted")
            raw_counts = value["counts"]
            if not isinstance(raw_counts, Mapping) or set(raw_counts) != set(
                _RECEIPT_STATES
            ):
                raise DistillationWorksetError("workset receipt counts are corrupted")
            counts: dict[str, int] = {}
            for state in _RECEIPT_STATES:
                count = raw_counts[state]
                if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                    raise DistillationWorksetError(
                        "workset receipt counts are corrupted"
                    )
                counts[state] = count
            watermark = _metadata_value(value["watermark"], "receipt watermark")
            result: dict[str, Any] = {"counts": counts, "watermark": watermark}
            if version == 2:
                if before and bootstrap:
                    if value["progress"] is not None:
                        raise DistillationWorksetError("workset receipt bootstrap is corrupted")
                    result["progress"] = None
                else:
                    result["progress"] = _strict_progress(
                        value["progress"], "receipt progress"
                    )
            return result

        before = snapshot(payload["before"], before=True)
        after = snapshot(payload["after"], before=False)
        raw_delta = payload["delta"]
        if not isinstance(raw_delta, Mapping) or set(raw_delta) != set(_RECEIPT_STATES):
            raise DistillationWorksetError("workset receipt delta is corrupted")
        delta: dict[str, int] = {}
        for state in _RECEIPT_STATES:
            value = raw_delta[state]
            if isinstance(value, bool) or not isinstance(value, int):
                raise DistillationWorksetError("workset receipt delta is corrupted")
            delta[state] = value
        if delta != cls._count_delta(before, after):
            raise DistillationWorksetError("workset receipt count continuity failed")

        details = payload["details"]
        if not isinstance(details, Mapping):
            raise DistillationWorksetError("workset receipt details are corrupted")
        if operation == "advance":
            base_keys = {
                "inserted",
                "rebound",
                "watermark_changed",
                "selection_sha256",
            }
            expected_keys = base_keys | ({"progress_changed"} if version == 2 else set())
            if set(details) != expected_keys:
                raise DistillationWorksetError("workset advance receipt is corrupted")
            inserted = details["inserted"]
            rebound = details["rebound"]
            changed = details["watermark_changed"]
            if (
                isinstance(inserted, bool)
                or not isinstance(inserted, int)
                or inserted < 0
                or isinstance(rebound, bool)
                or not isinstance(rebound, int)
                or rebound < 0
                or not isinstance(changed, bool)
                or changed
                != (
                    _json(before["watermark"], "receipt watermark")
                    != _json(after["watermark"], "receipt watermark")
                )
                or (
                    inserted == 0
                    and rebound == 0
                    and not changed
                    and not details.get("progress_changed", False)
                )
            ):
                raise DistillationWorksetError("workset advance receipt is corrupted")
            if version == 2 and (
                not isinstance(details["progress_changed"], bool)
                or details["progress_changed"]
                != (_json(before["progress"], "receipt progress") != _json(after["progress"], "receipt progress"))
            ):
                raise DistillationWorksetError("workset advance receipt is corrupted")
            _digest(details["selection_sha256"], "receipt selection_sha256")
            if delta != {
                "ready": inserted,
                "leased": 0,
                "completed": 0,
                "quarantined": 0,
            }:
                raise DistillationWorksetError("workset advance delta is invalid")
        elif operation in {"claim_reclaim", "claim", "release"}:
            expected_keys = {"kind", "count", "selection_sha256"}
            if set(details) != expected_keys:
                raise DistillationWorksetError("workset lease receipt is corrupted")
            kind = details["kind"]
            count = details["count"]
            if (
                not isinstance(kind, str)
                or _IDENTIFIER_RE.fullmatch(kind) is None
                or isinstance(count, bool)
                or not isinstance(count, int)
                or count < 1
            ):
                raise DistillationWorksetError("workset lease receipt is corrupted")
            _digest(details["selection_sha256"], "receipt selection_sha256")
            if operation == "claim_reclaim":
                expected = {
                    "ready": count,
                    "leased": -count,
                    "completed": 0,
                    "quarantined": 0,
                }
            elif operation == "claim":
                expected = {
                    "ready": -count,
                    "leased": count,
                    "completed": 0,
                    "quarantined": 0,
                }
            else:
                expected = {
                    "ready": count,
                    "leased": -count,
                    "completed": 0,
                    "quarantined": 0,
                }
            if delta != expected:
                raise DistillationWorksetError("workset lease delta is invalid")
        else:
            legacy_keys = {"completed", "retry", "quarantined", "selection_sha256"}
            timed_keys = {*legacy_keys, "retry_wait", "retry_schedule_sha256"}
            if set(details) not in (legacy_keys, timed_keys):
                raise DistillationWorksetError("workset commit receipt is corrupted")
            if set(details) == timed_keys and (
                isinstance(details.get("retry_wait"), bool)
                or not isinstance(details.get("retry_wait"), int)
                or details["retry_wait"] < 0
                or details["retry_wait"] > details["retry"]
            ):
                raise DistillationWorksetError("workset commit receipt is corrupted")
            if set(details) == timed_keys:
                _digest(
                    details["retry_schedule_sha256"], "receipt retry_schedule_sha256"
                )
            totals = {
                state: details[state] for state in ("completed", "retry", "quarantined")
            }
            _digest(details["selection_sha256"], "receipt selection_sha256")
            if (
                any(
                    isinstance(value, bool) or not isinstance(value, int) or value < 0
                    for value in totals.values()
                )
                or sum(totals.values()) < 1
            ):
                raise DistillationWorksetError("workset commit receipt is corrupted")
            expected = {
                "ready": totals["retry"],
                "leased": -sum(totals.values()),
                "completed": totals["completed"],
                "quarantined": totals["quarantined"],
            }
            if delta != expected:
                raise DistillationWorksetError("workset commit delta is invalid")
        return before, after, delta, dict(details)

    def audit_transition_receipts(self) -> dict[str, Any]:
        """Validate the append-only transition chain and reconcile its final state."""

        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT generation, previous_sha256, operation, payload_json, "
                "receipt_sha256 FROM workset_receipts ORDER BY generation ASC"
            ).fetchall()
            current = self._snapshot(connection)
            current_progress = self._progress(connection)
            if not rows:
                if (
                    sum(current["counts"].values()) == 0
                    and current["watermark"] is None
                ):
                    return {
                        "status": "verified-empty",
                        "receipts": 0,
                        "generation": 0,
                        "head_sha256": "",
                        "counts": current["counts"],
                        "watermark": current["watermark"],
                    }
                return {
                    "status": "legacy-unverified",
                    "receipts": 0,
                    "generation": 0,
                    "head_sha256": "",
                    "counts": current["counts"],
                    "watermark": current["watermark"],
                }

            previous_sha256 = ""
            prior_after: dict[str, Any] | None = None
            legacy_origin = False
            saw_v2 = False
            for expected_generation, row in enumerate(rows, start=1):
                generation = row["generation"]
                operation = row["operation"]
                payload_json = row["payload_json"]
                receipt_sha256 = row["receipt_sha256"]
                if (
                    isinstance(generation, bool)
                    or not isinstance(generation, int)
                    or generation != expected_generation
                    or not isinstance(operation, str)
                    or operation not in _RECEIPT_OPERATIONS
                    or not isinstance(payload_json, str)
                    or not isinstance(receipt_sha256, str)
                    or len(receipt_sha256) != 64
                    or any(
                        character not in "0123456789abcdef"
                        for character in receipt_sha256
                    )
                    or row["previous_sha256"] != previous_sha256
                ):
                    raise DistillationWorksetError("workset receipt chain is corrupted")
                try:
                    payload = json.loads(payload_json)
                except (RecursionError, json.JSONDecodeError) as exc:
                    raise DistillationWorksetError(
                        "workset receipt JSON is invalid"
                    ) from exc
                if (
                    _json(
                        payload,
                        "workset receipt",
                        max_bytes=_MAX_RECEIPT_JSON_BYTES,
                    )
                    != payload_json
                ):
                    raise DistillationWorksetError(
                        "workset receipt JSON is not canonical"
                    )
                before, after, _, _ = self._validate_receipt_payload(operation, payload)
                version = payload.get("version", 1)
                if saw_v2 and version == 1:
                    raise DistillationWorksetError("workset receipt progress downgraded")
                if version == 2:
                    if payload.get("bootstrap") is True:
                        if prior_after is not None and "progress" in prior_after:
                            raise DistillationWorksetError(
                                "workset receipt progress continuity failed"
                            )
                        legacy_origin = legacy_origin or prior_after is not None
                    elif prior_after is not None:
                        if "progress" not in prior_after:
                            if not _is_legacy_ox_progress_upgrade(
                                before["progress"], after["progress"]
                            ):
                                raise DistillationWorksetError(
                                    "workset receipt progress continuity failed"
                                )
                            legacy_origin = True
                        elif _json(
                            before["progress"], "receipt progress"
                        ) != _json(
                            prior_after["progress"], "receipt progress"
                        ):
                            raise DistillationWorksetError(
                                "workset receipt progress continuity failed"
                            )
                    saw_v2 = True
                    if before["progress"] is not None:
                        _validate_progress_transition(
                            before["progress"], after["progress"]
                        )
                if prior_after is None:
                    legacy_origin = (
                        before["counts"] != {state: 0 for state in _RECEIPT_STATES}
                        or before["watermark"] is not None
                    )
                elif _json(
                    {key: before[key] for key in ("counts", "watermark")},
                    "receipt snapshot",
                    max_bytes=_MAX_RECEIPT_JSON_BYTES,
                ) != _json(
                    {key: prior_after[key] for key in ("counts", "watermark")},
                    "receipt snapshot",
                    max_bytes=_MAX_RECEIPT_JSON_BYTES,
                ):
                    raise DistillationWorksetError("workset receipt continuity failed")
                envelope = {
                    "generation": generation,
                    "previous_sha256": previous_sha256,
                    "operation": operation,
                    "payload": payload,
                }
                if canonical_json_sha256_strict(envelope) != receipt_sha256:
                    raise DistillationWorksetError("workset receipt hash mismatch")
                previous_sha256 = receipt_sha256
                prior_after = after

            if _json(
                {key: prior_after[key] for key in ("counts", "watermark")},
                "receipt snapshot",
                max_bytes=_MAX_RECEIPT_JSON_BYTES,
            ) != _json(
                current,
                "receipt snapshot",
                max_bytes=_MAX_RECEIPT_JSON_BYTES,
            ):
                raise DistillationWorksetError("workset receipt final state mismatch")
            if "progress" in prior_after and _json(
                prior_after["progress"], "receipt progress"
            ) != _json(current_progress, "receipt progress"):
                raise DistillationWorksetError("workset receipt final progress mismatch")
            result = {
                "status": "legacy-unverified" if legacy_origin else "verified",
                "receipts": len(rows),
                "generation": len(rows),
                "head_sha256": previous_sha256,
                "counts": current["counts"],
                "watermark": current["watermark"],
            }
            if current_progress is not None:
                result["progress"] = current_progress
                result["last_durable_receipt"] = {
                    "generation": len(rows),
                    "head_sha256": previous_sha256,
                }
            return result

    def advance(
        self,
        items: Iterable[Mapping[str, Any]],
        watermark: Any,
        *,
        progress: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Idempotently record immutable work and its source progress watermark."""

        watermark_json = _metadata_json(watermark, "watermark")
        records = [_item(item, watermark_json) for item in items]
        if len({record[0] for record in records}) != len(records):
            raise DistillationWorksetError("work_id repeats within one advance")
        selection_sha256 = _selection_sha256(
            {
                "work_id": record[0],
                "kind": record[1],
                "payload_digest": record[3],
                "temporal_split": json.loads(record[4]),
                "provenance": json.loads(record[5]),
            }
            for record in records
        )
        now = self._now()
        inserted = 0
        existing = 0
        last_receipt: tuple[int, str] | None = None
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            committed = False
            try:
                before = self._snapshot(connection)
                self._validate_stages(connection)
                before_progress = self._progress(connection)
                rebound = 0
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
                                watermark_json, stage, state, created_at, updated_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'ready', ?, ?)
                            """,
                            (*record, _stage_for_kind(record[1]), now, now),
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
                            rebound += 1
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
                after_progress = (
                    self._store_progress(connection, progress)
                    if progress is not None
                    else before_progress
                )
                after = self._snapshot(connection)
                watermark_changed = _json(before["watermark"], "watermark") != _json(
                    after["watermark"], "watermark"
                )
                progress_changed = _json(before_progress, "progress") != _json(
                    after_progress, "progress"
                )
                if inserted or rebound or watermark_changed or progress_changed:
                    details = {
                        "inserted": inserted,
                        "rebound": rebound,
                        "watermark_changed": watermark_changed,
                        "selection_sha256": selection_sha256,
                    }
                    if after_progress is not None:
                        details["progress_changed"] = progress_changed
                    self._append_receipt(
                        connection,
                        "advance",
                        before,
                        after,
                        details,
                        before_progress=before_progress,
                        after_progress=after_progress,
                    )
                if progress is not None:
                    last_receipt = self._receipt_head(connection)
                self._secure_sqlite_files()
                connection.execute("COMMIT")
                committed = True
            except Exception:
                if not committed:
                    connection.execute("ROLLBACK")
                raise
        result = {"inserted": inserted, "existing": existing, "watermark": watermark}
        if progress is not None:
            result["progress"] = _strict_progress(progress)
            result["last_durable_receipt"] = {
                "generation": last_receipt[0] if last_receipt is not None else 0,
                "head_sha256": last_receipt[1] if last_receipt is not None else "",
            }
        return result

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

    def progress(self) -> dict[str, Any] | None:
        """Read the last payload-free, durable progress boundary."""

        with closing(self._connect()) as connection:
            return self._progress(connection)

    def claim(
        self, kind: str | None, limit: int, owner: str, lease_seconds: float
    ) -> tuple[WorkClaim, ...]:
        """Claim FIFO work of one kind, reclaiming expired leases atomically."""

        if kind is not None:
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
                before = self._snapshot(connection)
                self._validate_stages(connection)
                expired_rows = connection.execute(
                    """
                    SELECT work_id, kind, payload_digest, temporal_split_json,
                           provenance_json, attempt_count, lease_owner, lease_id
                    FROM work_items
                    WHERE state = 'leased' AND lease_expires_at <= ?
                      AND (? IS NULL OR kind = ?)
                    ORDER BY sequence ASC
                    """,
                    (now, kind, kind),
                ).fetchall()
                expired_selection_sha256 = _selection_sha256(
                    {
                        "work_id": row["work_id"],
                        "kind": row["kind"],
                        "payload_digest": row["payload_digest"],
                        "temporal_split": json.loads(row["temporal_split_json"]),
                        "provenance": json.loads(row["provenance_json"]),
                        "attempt": row["attempt_count"],
                        "owner": row["lease_owner"],
                        "lease_id": row["lease_id"],
                    }
                    for row in expired_rows
                )
                reclaim_result = connection.execute(
                    """
                    UPDATE work_items
                    SET state = 'ready', lease_id = NULL, lease_owner = NULL,
                        lease_expires_at = NULL, updated_at = ?
                    WHERE state = 'leased' AND lease_expires_at <= ?
                      AND (? IS NULL OR kind = ?)
                    """,
                    (now, now, kind, kind),
                )
                reclaimed = reclaim_result.rowcount
                if reclaimed != len(expired_rows):
                    raise DistillationWorksetError(
                        "expired lease reclaim is inconsistent"
                    )
                if reclaimed:
                    after_reclaim = self._snapshot(connection)
                    self._append_receipt(
                        connection,
                        "claim_reclaim",
                        before,
                        after_reclaim,
                        {
                            "kind": kind or "mixed",
                            "count": reclaimed,
                            "selection_sha256": expired_selection_sha256,
                        },
                    )
                chosen_kind = kind
                if kind is None:
                    oldest = connection.execute(
                        "SELECT kind, MIN(created_at) AS oldest FROM work_items "
                        "WHERE state = 'ready' AND (next_attempt_at IS NULL OR next_attempt_at <= ?) "
                        "GROUP BY kind ORDER BY oldest ASC LIMIT 1",
                        (now,),
                    ).fetchone()
                    if oldest is not None and now - float(oldest["oldest"]) >= _FAIRNESS_AGE_SECONDS:
                        chosen_kind = str(oldest["kind"])
                rows = connection.execute(
                    """
                    SELECT work_id, kind, payload_ref, payload_digest,
                           temporal_split_json, provenance_json, priority, attempt_count
                    FROM work_items
                    WHERE (? IS NULL OR kind = ?) AND state = 'ready'
                      AND (next_attempt_at IS NULL OR next_attempt_at <= ?)
                    ORDER BY priority DESC, sequence ASC
                    LIMIT ?
                    """,
                    (chosen_kind, chosen_kind, now, limit),
                ).fetchall()
                for row in rows:
                    lease_id = uuid.uuid4().hex
                    result = connection.execute(
                        """
                        UPDATE work_items
                        SET state = 'leased', lease_id = ?, lease_owner = ?,
                            lease_expires_at = ?, attempt_count = attempt_count + 1,
                            next_attempt_at = NULL,
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
                if claims:
                    after_claim = self._snapshot(connection)
                    self._append_receipt(
                        connection,
                        "claim",
                        after_reclaim if reclaimed else before,
                        after_claim,
                        {
                            "kind": kind or "mixed",
                            "count": len(claims),
                            "selection_sha256": _claim_selection_sha256(claims),
                        },
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
                before = self._snapshot(connection)
                self._validate_stages(connection)
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
                            lease_expires_at = NULL, next_attempt_at = NULL, updated_at = ?
                        WHERE work_id = ?
                        """,
                        (now, claim.work_id),
                    )
                after = self._snapshot(connection)
                kinds = {claim.kind for claim in claims}
                self._append_receipt(
                    connection,
                    "release",
                    before,
                    after,
                    {
                        "kind": next(iter(kinds)) if len(kinds) == 1 else "mixed",
                        "count": len(claims),
                        "selection_sha256": _claim_selection_sha256(claims),
                    },
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
        *,
        progress: Mapping[str, Any] | None = None,
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
        changed_totals = {"completed": 0, "retry": 0, "quarantined": 0}
        changed_claims: list[WorkClaim] = []
        changed_outcomes: list[dict[str, Any]] = []
        now = self._now()
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            committed = False
            try:
                before = self._snapshot(connection)
                self._validate_stages(connection)
                before_progress = self._progress(connection)
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
                                lease_expires_at = NULL, next_attempt_at = NULL, updated_at = ?
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
                                next_attempt_at = ?, updated_at = ?
                            WHERE work_id = ?
                            """,
                            (
                                state,
                                outcome["error_class"],
                                now + outcome["retry_after_seconds"]
                                if outcome["status"] == "retry"
                                else None,
                                now,
                                claim.work_id,
                            ),
                        )
                    totals[outcome["status"]] += 1
                    changed_totals[outcome["status"]] += 1
                    changed_claims.append(claim)
                    changed_outcomes.append(outcome)
                after_progress = before_progress
                if progress is not None:
                    after_progress = self._store_progress(connection, progress)
                progress_changed = _json(before_progress, "progress") != _json(
                    after_progress, "progress"
                )
                if not changed_claims and progress_changed:
                    raise DistillationWorksetError(
                        "durable progress requires an active completion"
                    )
                if sum(changed_totals.values()) > 0:
                    after = self._snapshot(connection)
                    self._append_receipt(
                        connection,
                        "commit",
                        before,
                        after,
                        {
                            **changed_totals,
                            "retry_wait": sum(
                                1
                                for outcome in normalized
                                if outcome["status"] == "retry"
                                and outcome["retry_after_seconds"] > 0
                            ),
                            "retry_schedule_sha256": canonical_json_sha256_strict(
                                [
                                    {
                                        "work_id": claim.work_id,
                                        "retry_after_seconds": outcome[
                                            "retry_after_seconds"
                                        ],
                                    }
                                    for claim, outcome in zip(
                                        changed_claims, changed_outcomes, strict=True
                                    )
                                    if outcome["status"] == "retry"
                                ]
                            ),
                            "selection_sha256": _claim_selection_sha256(changed_claims),
                        },
                        before_progress=before_progress,
                        after_progress=after_progress,
                    )
                self._secure_sqlite_files()
                connection.execute("COMMIT")
                committed = True
            except Exception:
                if not committed:
                    connection.execute("ROLLBACK")
                raise
        return totals

    def status(
        self, kind: str | None = None, *, include_timing: bool = False
    ) -> dict[str, Any]:
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
        timing_rows: Sequence[sqlite3.Row] = ()
        stage_rows: Sequence[sqlite3.Row] = ()
        with closing(self._connect()) as connection:
            rows = connection.execute(
                f"SELECT state, COUNT(*) AS count FROM work_items{where} GROUP BY state",
                parameters,
            ).fetchall()
            self._validate_stages(connection)
            progress = self._progress(connection)
            generation, head = self._receipt_head(connection)
            if include_timing:
                timing_where = " WHERE state = 'ready'" + (
                    " AND kind = ?" if kind is not None else ""
                )
                timing_rows = connection.execute(
                    "SELECT next_attempt_at, created_at FROM work_items" + timing_where,
                    parameters,
                ).fetchall()
                stage_rows = connection.execute(
                    "SELECT stage, state, next_attempt_at, created_at, COUNT(*) AS count "
                    "FROM work_items" + where + " GROUP BY stage, state, next_attempt_at, created_at",
                    parameters,
                ).fetchall()
        for row in rows:
            counts[str(row["state"])] = int(row["count"])
        counts["backlog"] = counts["ready"] + counts["leased"]
        counts["total"] = sum(
            counts[state] for state in ("ready", "leased", "completed", "quarantined")
        )
        counts["last_durable_receipt"] = {
            "generation": generation,
            "head_sha256": head,
        }
        counts["last_durable_progress"] = progress
        if include_timing:
            now = self._now()
            due = [
                row
                for row in timing_rows
                if row["next_attempt_at"] is None or row["next_attempt_at"] <= now
            ]
            waiting = [
                row
                for row in timing_rows
                if row["next_attempt_at"] is not None and row["next_attempt_at"] > now
            ]
            counts["retry_wait"] = len(waiting)
            counts["oldest_backlog_age_seconds"] = int(
                max((max(0.0, now - row["created_at"]) for row in timing_rows), default=0)
            )
            counts["oldest_ready_age_seconds"] = int(
                max((max(0.0, now - row["created_at"]) for row in due), default=0)
            )
            counts["oldest_retry_wait_age_seconds"] = int(
                max((max(0.0, now - row["created_at"]) for row in waiting), default=0)
            )
            counts["next_retry_in_seconds"] = int(
                min((row["next_attempt_at"] - now for row in waiting), default=0)
            )
            stages: dict[str, dict[str, int]] = {
                stage: {
                    state: 0
                    for state in (*_RECEIPT_STATES, "retry_wait", "backlog")
                }
                for stage in _STAGES
            }
            for row in stage_rows:
                stage = str(row["stage"])
                if stage not in stages:
                    raise DistillationWorksetError("workset stage is corrupted")
                state = str(row["state"])
                count = int(row["count"])
                stages[stage][state] += count
                if state == "ready" and row["next_attempt_at"] is not None and row["next_attempt_at"] > now:
                    stages[stage]["retry_wait"] += count
            for stage in _STAGES:
                stages[stage]["backlog"] = (
                    stages[stage]["ready"] + stages[stage]["leased"]
                )
            stages["retry_wait"]["retry_wait"] = counts["retry_wait"]
            counts["stages"] = stages
        return counts

    @staticmethod
    def _outcome(claim: WorkClaim, value: Mapping[str, Any]) -> dict[str, Any]:
        allowed = {
            "status",
            "error_class",
            "completion_ref",
            "completion_digest",
            "retry_after_seconds",
        }
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
            if "retry_after_seconds" in value:
                raise DistillationWorksetError("completed work cannot delay retry")
            return {
                "status": status,
                "error_class": "",
                "completion_ref": _reference(
                    value.get("completion_ref"), "completion_ref"
                ),
                "completion_digest": _digest(
                    value.get("completion_digest"), "completion_digest"
                ),
                "retry_after_seconds": 0.0,
            }
        if not error_class:
            raise DistillationWorksetError("non-completed work requires error_class")
        if "completion_ref" in value or "completion_digest" in value:
            raise DistillationWorksetError("failed work cannot include a completion")
        retry_after = _finite(
            value.get("retry_after_seconds", 0), "retry_after_seconds"
        )
        if retry_after < 0 or retry_after > 86_400:
            raise DistillationWorksetError("retry_after_seconds is out of range")
        if status != "retry" and retry_after:
            raise DistillationWorksetError("quarantined work cannot delay retry")
        return {
            "status": status,
            "error_class": error_class,
            "completion_ref": "",
            "completion_digest": "",
            "retry_after_seconds": retry_after,
        }
