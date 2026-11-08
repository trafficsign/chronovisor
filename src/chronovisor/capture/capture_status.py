"""Privacy-safe monitoring state for the single-process capture run."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from chronovisor.common.durable_state import atomic_write
from chronovisor.common.paths import CHRONOVISOR_ROOT
from chronovisor.common.timeutil import utc_now

STATUS_VERSION = 1
SAFE_ERROR_TOKENS = frozenset(
    {
        "capture_failed",
        "capture_status_invalid",
        "capture_status_missing",
        "capture_status_stale",
        "config_invalid",
        "failed",
        "none",
        "permanent_failure",
        "record_failed",
        "retry_failed",
        "transient_failure",
        "unknown",
        "write_failed",
    }
)


def status_path(root: Path = CHRONOVISOR_ROOT) -> Path:
    return Path(root) / "runtime" / "capture-status.json"


def timestamp(value: datetime | None = None) -> str:
    return (value or utc_now()).astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def normalize_timestamp(value: object) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def safe_error_token(value: object, default: str = "capture_status_invalid") -> str:
    return value if isinstance(value, str) and value in SAFE_ERROR_TOKENS else default


def index_error_token(value: object, default: str = "none") -> str:
    if value == "none":
        return "none"
    if isinstance(value, str) and value.isidentifier():
        return value
    return default


def _count(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return 0
    return value


def read(path: Path) -> dict[str, Any] | None:
    if path.is_symlink():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError, UnicodeError, ValueError:
        return None
    if not isinstance(data, dict) or data.get("version") != STATUS_VERSION:
        return None
    return data


def write(path: Path, status: Mapping[str, Any]) -> None:
    payload = {
        "version": STATUS_VERSION,
        "last_attempt_at": normalize_timestamp(status.get("last_attempt_at")),
        "last_success_at": normalize_timestamp(status.get("last_success_at")),
        "last_commit_at": normalize_timestamp(status.get("last_commit_at")),
        "oldest_pending_at": normalize_timestamp(status.get("oldest_pending_at")),
        "pending": _count(status.get("pending")),
        "failed": _count(status.get("failed")),
        "last_error": safe_error_token(status.get("last_error"), "capture_failed"),
        "index_refreshed_at": normalize_timestamp(status.get("index_refreshed_at")),
        "index_error": index_error_token(status.get("index_error"), "none"),
    }
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    atomic_write(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def snapshot(path: Path) -> dict[str, Any]:
    data = read(path)
    if data is None:
        missing = not path.exists()
        return {
            "available": False,
            "error": "capture_status_missing" if missing else "capture_status_invalid",
        }
    return {
        "available": True,
        "last_attempt_at": normalize_timestamp(data.get("last_attempt_at")),
        "last_success_at": normalize_timestamp(data.get("last_success_at")),
        "last_commit_at": normalize_timestamp(data.get("last_commit_at")),
        "oldest_pending_at": normalize_timestamp(data.get("oldest_pending_at")),
        "pending": _count(data.get("pending")),
        "failed": _count(data.get("failed")),
        "last_error": safe_error_token(data.get("last_error")),
        "index_refreshed_at": normalize_timestamp(data.get("index_refreshed_at")),
        "index_error": index_error_token(data.get("index_error"), "none"),
    }
