"""Durable progress receipt for formal Recall distillation runs.

The distillation worker intentionally leaves its sealed worker state unchanged
when an attempt is deferred before a state commit.  This small receipt records
that operational fact under the same sealed-store boundary so health can tell
an active data hold from a repeatedly deferred formal run.
"""

from __future__ import annotations

import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from chronovisor.recall import recall_distillation_store as store

FORMAL_PROGRESS_FILE = "formal-progress.json"
FORMAL_PROGRESS_KIND = "formal-progress"
_SAFE_REASON_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._:-"
)
_OUTCOMES = {"running", "deferred", "data_reject", "error", "succeeded", "other"}


def _now() -> datetime:
    return datetime.now(UTC)


def _timestamp(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed.astimezone(UTC)


def _timestamp_text(value: object) -> str | None:
    parsed = _timestamp(value)
    return parsed.isoformat() if parsed is not None else None


def _safe_reason(value: object) -> str:
    if not isinstance(value, str) or not value:
        return ""
    if len(value) > 128 or any(char not in _SAFE_REASON_CHARS for char in value):
        return "unknown"
    return value


def _progress_path(root: Path) -> Path:
    return Path(store.distillation_dir(root)) / FORMAL_PROGRESS_FILE


def _progress_lock_path(root: Path) -> Path:
    path = _progress_path(root)
    return path.with_suffix(path.suffix + ".lock")


@contextmanager
def _progress_lock(root: Path) -> Iterator[None]:
    """Serialize receipt read-modify-write operations without worker lock."""

    with store._locked(_progress_lock_path(root)):
        yield


def _unsigned_progress(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: item
        for key, item in value.items()
        if key not in {"schema", "namespace", "seal_sha256"}
    }


def _read_progress(root: Path) -> dict[str, Any]:
    path = _progress_path(root)
    if not path.exists():
        return {
            "kind": FORMAL_PROGRESS_KIND,
            "first_attempt_at": "",
            "last_attempt_at": "",
            "last_run_started_at": "",
            "last_success_at": "",
            "last_outcome": "",
            "last_defer_reason": "",
            "consecutive_deferrals": 0,
            "attempt_count": 0,
            "success_count": 0,
        }
    value = store.read_sealed(path, schema=store.DISTILLATION_SCHEMA)
    if not isinstance(value, dict):
        raise store.DistillationStoreError("formal progress is not an object")
    if value.get("kind") != FORMAL_PROGRESS_KIND:
        raise store.DistillationStoreError("formal progress kind mismatch")
    for key in (
        "first_attempt_at",
        "last_attempt_at",
        "last_run_started_at",
        "last_success_at",
    ):
        timestamp = value.get(key)
        if timestamp not in {None, ""} and _timestamp(timestamp) is None:
            raise store.DistillationStoreError(
                f"formal progress timestamp is invalid: {key}"
            )
    for key in ("consecutive_deferrals", "attempt_count", "success_count"):
        number = value.get(key)
        if isinstance(number, bool) or not isinstance(number, int) or number < 0:
            raise store.DistillationStoreError(
                f"formal progress count is invalid: {key}"
            )
    outcome = value.get("last_outcome", "")
    if not isinstance(outcome, str) or outcome not in {"", *_OUTCOMES}:
        raise store.DistillationStoreError("formal progress outcome is invalid")
    reason = value.get("last_defer_reason", "")
    if not isinstance(reason, str) or (reason and _safe_reason(reason) != reason):
        raise store.DistillationStoreError("formal progress reason is invalid")
    return value


def read(root: Path) -> dict[str, Any] | None:
    """Return the sealed formal-progress receipt, or ``None`` if absent/invalid."""

    try:
        with _progress_lock(root):
            if not _progress_path(root).exists():
                return None
            return _read_progress(root)
    except (OSError, TypeError, ValueError, store.DistillationStoreError):
        return None


def begin(root: Path) -> dict[str, Any]:
    """Record the start of one formal attempt and return its private token."""

    with _progress_lock(root):
        progress = _unsigned_progress(_read_progress(root))
        started_at = _now().isoformat()
        if not progress.get("first_attempt_at"):
            progress["first_attempt_at"] = started_at
        progress["last_attempt_at"] = started_at
        progress["last_run_started_at"] = started_at
        progress["last_outcome"] = "running"
        progress["last_defer_reason"] = ""
        progress["attempt_count"] = int(progress["attempt_count"]) + 1
        store.write_sealed_state(_progress_path(root), progress)
        return {
            "started_at": started_at,
            "started_at_monotonic": time.monotonic(),
        }


def record_busy(root: Path) -> dict[str, Any]:
    """Record a deferred attempt that could not acquire the worker lock."""

    with _progress_lock(root):
        progress = _unsigned_progress(_read_progress(root))
        attempted_at = _now().isoformat()
        if not progress.get("first_attempt_at"):
            progress["first_attempt_at"] = attempted_at
        progress["last_attempt_at"] = attempted_at
        # This is not a run start.  Keep last_run_started_at untouched so a
        # concurrently finishing run can still validate its private token.
        progress["last_outcome"] = "deferred"
        progress["last_defer_reason"] = "worker_busy"
        progress["consecutive_deferrals"] = (
            int(progress["consecutive_deferrals"]) + 1
        )
        progress["attempt_count"] = int(progress["attempt_count"]) + 1
        value = store.write_sealed_state(_progress_path(root), progress)
        if not isinstance(value, dict):
            raise store.DistillationStoreError("formal progress write is invalid")
        return value


def _state_last_success(root: Path) -> datetime | None:
    try:
        state = store.read_sealed(
            store.distillation_dir(root) / store.STATE_FILE,
            schema=store.DISTILLATION_SCHEMA,
        )
    except (OSError, TypeError, ValueError, store.DistillationStoreError):
        return None
    parsed = _timestamp(state.get("last_success_at"))
    if parsed is None or parsed > _now():
        return None
    return parsed


def _current_data_reject(root: Path, started_at: datetime) -> bool:
    try:
        state = store.read_sealed(
            store.distillation_dir(root) / store.STATE_FILE,
            schema=store.DISTILLATION_SCHEMA,
        )
    except (OSError, TypeError, ValueError, store.DistillationStoreError):
        return False
    if state.get("worker_error_stage") != "counterfactual":
        return False
    if state.get("worker_error_kind") != "counterfactual_data_reject":
        return False
    failed_at = _timestamp(state.get("last_failure_at"))
    return failed_at is not None and failed_at >= started_at


def finish(
    root: Path,
    token: Mapping[str, Any],
    result: Mapping[str, Any] | None = None,
    error: BaseException | None = None,
) -> dict[str, Any]:
    """Close one formal attempt and persist its operational outcome.

    A result is successful only when the worker's sealed ``last_success_at``
    advanced after this token began.  A return value or clean process exit is
    never treated as a formal success by itself.
    """

    started_at = _timestamp(token.get("started_at"))
    if started_at is None or started_at > _now():
        raise store.DistillationStoreError("formal progress token is invalid")
    with _progress_lock(root):
        progress = _unsigned_progress(_read_progress(root))
        run_started_at = _timestamp(progress.get("last_run_started_at"))
        if run_started_at is None:
            # Receipts written before last_run_started_at was introduced used
            # last_attempt_at for the run token boundary.
            run_started_at = _timestamp(progress.get("last_attempt_at"))
        if run_started_at != started_at:
            raise store.DistillationStoreError("formal progress token is stale")

        state_success = _state_last_success(root)
        if state_success is not None and state_success > started_at:
            outcome = "succeeded"
            reason = ""
            progress["last_success_at"] = state_success.isoformat()
            progress["success_count"] = int(progress["success_count"]) + 1
            progress["consecutive_deferrals"] = 0
        elif error is not None:
            outcome = "error"
            reason = ""
            progress["consecutive_deferrals"] = 0
        elif _current_data_reject(root, started_at):
            outcome = "data_reject"
            reason = ""
            # Quality/data outcomes must remain visible in their existing
            # health class and must not be counted as an operational defer
            # streak.
            progress["consecutive_deferrals"] = 0
        elif isinstance(result, Mapping) and result.get("status") == "deferred":
            outcome = "deferred"
            reason = _safe_reason(result.get("reason"))
            progress["consecutive_deferrals"] = (
                int(progress["consecutive_deferrals"]) + 1
            )
        else:
            outcome = "other"
            reason = ""
            progress["consecutive_deferrals"] = 0

        progress["last_outcome"] = outcome
        progress["last_defer_reason"] = reason
        value = store.write_sealed_state(_progress_path(root), progress)
        if not isinstance(value, dict):
            raise store.DistillationStoreError("formal progress write is invalid")
        return value


__all__ = [
    "FORMAL_PROGRESS_FILE",
    "FORMAL_PROGRESS_KIND",
    "begin",
    "finish",
    "record_busy",
    "read",
]
