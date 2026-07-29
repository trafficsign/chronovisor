"""Small durable circuit breaker for synchronous Recall dependencies."""

from __future__ import annotations

from chronovisor.timeutil import iso_seconds as _iso

from chronovisor.timeutil import utc_now as _now

import fcntl
import json
import os
import tempfile
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

from chronovisor.recall_runtime_paths import RECALL_DIR


BREAKER_FILE = RECALL_DIR / "circuit-breaker.json"






def _parse_time(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _lock_file(path: Path) -> Path:
    return path.with_name(f"{path.name}.lock")


@contextmanager
def _locked(path: Path) -> Iterator[None]:
    lock = _lock_file(path)
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _failure_count(state: dict[str, Any]) -> int:
    try:
        return max(0, int(state.get("failures") or 0))
    except (TypeError, ValueError):
        return 0


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def snapshot(
    path: Path | None = None,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    path = path or BREAKER_FILE
    current = now or _now()
    state = _read(path)
    open_until = _parse_time(state.get("open_until"))
    is_open = open_until is not None and open_until > current
    return {
        "status": "open" if is_open else "closed",
        "failures": _failure_count(state),
        "open_until": _iso(open_until) if open_until is not None else None,
        "last_failure_at": state.get("last_failure_at"),
        "last_failure_reason": state.get("last_failure_reason", ""),
        "updated_at": state.get("updated_at"),
    }


def is_open(path: Path | None = None, *, now: datetime | None = None) -> bool:
    return snapshot(path, now=now)["status"] == "open"


def record_failure(
    reason: str,
    *,
    threshold: int,
    cooldown_seconds: int,
    path: Path | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    path = path or BREAKER_FILE
    current = now or _now()
    with _locked(path):
        state = _read(path)
        failures = _failure_count(state) + 1
        payload: dict[str, Any] = {
            "schema_version": 1,
            "failures": failures,
            "last_failure_at": _iso(current),
            "last_failure_reason": reason[:500],
            "updated_at": _iso(current),
        }
        if failures >= max(1, threshold):
            payload["open_until"] = _iso(
                current + timedelta(seconds=max(1, cooldown_seconds))
            )
        _write(path, payload)
    return snapshot(path, now=current)


def record_success(
    path: Path | None = None,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    path = path or BREAKER_FILE
    current = now or _now()
    with _locked(path):
        state = _read(path)
        if _failure_count(state) == 0 and not state.get("open_until"):
            return snapshot(path, now=current)
        _write(
            path,
            {
                "schema_version": 1,
                "failures": 0,
                "open_until": None,
                "last_success_at": _iso(current),
                "updated_at": _iso(current),
            },
        )
    return snapshot(path, now=current)
