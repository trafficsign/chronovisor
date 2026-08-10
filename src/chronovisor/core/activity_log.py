"""Validated operational activity journal for the live OKF runtime."""

from __future__ import annotations

import fcntl
import json
import os
import stat
import time
from collections import deque
from collections.abc import Iterator, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from chronovisor.core.canonical_json import canonical_json_line_bytes_strict
from chronovisor.core.durable_state import open_regular_nofollow

ACTIVITY_SCHEMA = "chronovisor.activity.v1"
ACTIVITY_LEVELS = frozenset({"info", "warn", "error", "success"})
MAX_ACTIVITY_MESSAGE_BYTES = 4096
MAX_ACTIVITY_LINE_BYTES = 8192
MAX_ACTIVITY_READ_ROWS = 500
MAX_ACTIVITY_FILE_BYTES = 64 * 1024 * 1024
MAX_ACTIVITY_DELTA_BYTES = 1024 * 1024


def activity_record(
    message: str,
    *,
    source: str,
    level: str = "info",
    timestamp: str | None = None,
    event_id: str | None = None,
) -> dict[str, str]:
    """Build one fixed-schema, single-line operational activity record."""

    if not isinstance(message, str) or not message:
        raise ValueError("activity message must be non-empty text")
    if any(
        (ord(character) < 32 and character != "\n") or ord(character) == 127
        for character in message
    ):
        raise ValueError("activity message contains unsafe control text")
    if len(message.encode("utf-8")) > MAX_ACTIVITY_MESSAGE_BYTES:
        raise ValueError("activity message exceeds the bounded size")
    if (
        not isinstance(source, str)
        or not source
        or len(source) > 64
        or not source[0].isalpha()
        or any(not (character.isalnum() or character in "_.-") for character in source)
    ):
        raise ValueError("activity source is invalid")
    if level not in ACTIVITY_LEVELS:
        raise ValueError("activity level is invalid")
    observed_timestamp = timestamp or datetime.now().astimezone().isoformat(
        timespec="seconds"
    )
    try:
        datetime.fromisoformat(observed_timestamp)
    except (TypeError, ValueError) as exc:
        raise ValueError("activity timestamp is invalid") from exc
    if len(observed_timestamp) > 40:
        raise ValueError("activity timestamp is invalid")
    stable_event_id = event_id or _event_id(
        observed_timestamp,
        level,
        source,
        message,
        nonce=f"{os.getpid()}:{time.time_ns()}",
    )
    if not isinstance(stable_event_id, str) or not _valid_event_id(stable_event_id):
        raise ValueError("activity event_id is invalid")
    return {
        "schema": ACTIVITY_SCHEMA,
        "event_id": stable_event_id,
        "timestamp": observed_timestamp,
        "level": level,
        "source": source,
        "message": message,
    }


def parse_activity_line(line: str | bytes) -> dict[str, str] | None:
    """Validate one activity JSONL row, returning only operational records."""

    raw = line.encode("utf-8") if isinstance(line, str) else line
    if not raw or len(raw) > MAX_ACTIVITY_LINE_BYTES or b"\n" in raw or b"\r" in raw:
        return None
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, Mapping) or set(payload) != {
        "schema",
        "event_id",
        "timestamp",
        "level",
        "source",
        "message",
    }:
        return None
    try:
        return activity_record(
            payload["message"],
            source=payload["source"],
            level=payload["level"],
            timestamp=payload["timestamp"],
            event_id=payload["event_id"],
        )
    except (TypeError, ValueError):
        return None


def append_activity(
    message: str,
    *,
    source: str,
    level: str = "info",
    root: Path | None = None,
    path: Path | None = None,
    timestamp: str | None = None,
) -> dict[str, str]:
    """Durably append one activity row while holding the shared OKF lease."""

    from chronovisor.core import store

    if root is None:
        root = path.parent.parent if path is not None else store.CHRONOVISOR_ROOT
    target = path or root / "runtime" / "activity.jsonl"
    _require_canonical_activity_target(root, target, allow_missing=True)
    record = activity_record(
        message,
        source=source,
        level=level,
        timestamp=timestamp,
    )
    with store.okf_runtime_operation(root):
        _append_nofollow(root, canonical_json_line_bytes_strict(record))
    return record


def read_activity(path: Path, *, limit: int = 100) -> list[dict[str, str]]:
    """Return a bounded tail of valid operational activity rows."""

    bounded = max(1, min(MAX_ACTIVITY_READ_ROWS, int(limit)))
    rows: deque[dict[str, str]] = deque(maxlen=bounded)
    try:
        for line in _tail_lines_nofollow(path):
            parsed = parse_activity_line(line)
            if parsed is not None:
                rows.append(parsed)
    except (OSError, RuntimeError, ValueError):
        return []
    return list(rows)


def iter_activity(path: Path) -> Iterator[dict[str, str]]:
    """Stream every valid operational row without materializing the journal."""

    try:
        for line, oversized in _raw_lines_nofollow(path):
            if oversized:
                continue
            parsed = parse_activity_line(line.rstrip(b"\n"))
            if parsed is not None:
                yield parsed
    except (OSError, RuntimeError, ValueError):
        return


def read_activity_delta(
    path: Path,
    *,
    offset: int,
    max_bytes: int = MAX_ACTIVITY_DELTA_BYTES,
) -> tuple[list[dict[str, str]], int]:
    """Read complete validated rows after an offset through a no-follow fd."""

    if offset < 0 or max_bytes <= 0 or max_bytes > MAX_ACTIVITY_DELTA_BYTES:
        raise ValueError("activity delta bounds are invalid")
    rows: list[dict[str, str]] = []
    try:
        with open_regular_nofollow(path) as handle:
            size = os.fstat(handle.fileno()).st_size
            position = 0 if offset > size else offset
            handle.seek(position)
            start = position
            while position < size and position - start < max_bytes:
                row_start = position
                line = handle.readline(MAX_ACTIVITY_LINE_BYTES + 2)
                position = handle.tell()
                if not line.endswith(b"\n"):
                    if len(line) > MAX_ACTIVITY_LINE_BYTES:
                        return rows, size
                    return rows, row_start
                raw = line[:-1]
                if len(raw) > MAX_ACTIVITY_LINE_BYTES:
                    continue
                parsed = parse_activity_line(raw)
                if parsed is not None:
                    rows.append(parsed)
    except (OSError, RuntimeError, ValueError):
        return [], offset
    return rows, position


def valid_activity_file(path: Path) -> bool:
    """Return whether every row is a known activity or migration metadata row."""

    try:
        for line, oversized in _raw_lines_nofollow(path):
            raw = line.rstrip(b"\n")
            if oversized or not raw:
                return False
            if parse_activity_line(raw) is not None:
                continue
            payload: Any = json.loads(raw)
            if not _valid_archive_migration_row(payload):
                return False
    except (OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return True


def validated_activity_bytes(
    path: Path, *, max_bytes: int = MAX_ACTIVITY_FILE_BYTES
) -> bytes:
    """Return an exact known-schema journal or fail closed without truncation."""

    if max_bytes < 0:
        raise ValueError("activity byte bound is invalid")
    with open_regular_nofollow(path) as handle:
        raw = handle.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raise ValueError("activity journal exceeds the migration bound")
    if raw and not raw.endswith(b"\n"):
        raise ValueError("activity journal has a torn final row")
    for line in raw.splitlines():
        if len(line) > MAX_ACTIVITY_LINE_BYTES:
            raise ValueError("activity journal row exceeds the bound")
        if parse_activity_line(line) is not None:
            continue
        try:
            payload: Any = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("activity journal row is invalid") from exc
        if not _valid_archive_migration_row(payload):
            raise ValueError("activity journal row schema is invalid")
    return raw


def activity_prefix_matches(path: Path, *, length: int, sha256: str) -> bool:
    """Validate one immutable prefix without scanning the mutable suffix."""

    if length < 0 or len(sha256) != 64:
        return False
    try:
        with open_regular_nofollow(path) as handle:
            prefix = handle.read(length)
    except (OSError, RuntimeError, ValueError):
        return False
    import hashlib

    return len(prefix) == length and hashlib.sha256(prefix).hexdigest() == sha256


def _valid_archive_migration_row(payload: Any) -> bool:
    if not isinstance(payload, Mapping) or payload.get("type") != (
        "okf_archive_metadata_migrated"
    ):
        return False
    allowed = {
        "type",
        "event_id",
        "uid",
        "relative_path",
        "archive_reason",
        "archive_provenance",
    }
    required = {"type", "event_id", "uid", "relative_path"}
    if not set(payload) <= allowed or not required <= set(payload):
        return False
    return all(isinstance(value, str) and value for value in payload.values()) and (
        _valid_event_id(str(payload["event_id"]))
    )


def deterministic_event_id(*parts: str) -> str:
    """Return a stable activity identifier for deterministic migration rows."""

    return _event_id(*parts, nonce="")


def _event_id(*parts: str, nonce: str) -> str:
    import hashlib

    payload = "\x1f".join((*parts, nonce)).encode("utf-8")
    return "activity-" + hashlib.sha256(payload).hexdigest()


def _valid_event_id(value: str) -> bool:
    return (
        value.startswith(("activity-", "okf-archive-"))
        and len(value.rsplit("-", 1)[-1]) == 64
        and all(character in "0123456789abcdef" for character in value.rsplit("-", 1)[-1])
    )


def _append_nofollow(root: Path, raw: bytes) -> None:
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    directory_flags |= getattr(os, "O_CLOEXEC", 0)
    root_fd = os.open(root, directory_flags)
    runtime_fd = target_fd = -1
    try:
        runtime_fd = os.open("runtime", directory_flags, dir_fd=root_fd)
        file_flags = os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW
        file_flags |= getattr(os, "O_CLOEXEC", 0)
        target_fd = os.open(
            "activity.jsonl", file_flags, 0o600, dir_fd=runtime_fd
        )
        if not stat.S_ISREG(os.fstat(target_fd).st_mode):
            raise ValueError("activity journal is unsafe")
        fcntl.flock(target_fd, fcntl.LOCK_EX)
        try:
            size = os.lseek(target_fd, 0, os.SEEK_END)
            if size:
                os.lseek(target_fd, -1, os.SEEK_END)
                if os.read(target_fd, 1) != b"\n":
                    raise ValueError("activity journal has a torn final row")
            _write_all(target_fd, raw)
            os.fsync(target_fd)
            os.fsync(runtime_fd)
        finally:
            fcntl.flock(target_fd, fcntl.LOCK_UN)
    finally:
        if target_fd >= 0:
            os.close(target_fd)
        if runtime_fd >= 0:
            os.close(runtime_fd)
        os.close(root_fd)


def _write_all(descriptor: int, raw: bytes) -> None:
    offset = 0
    while offset < len(raw):
        written = os.write(descriptor, raw[offset:])
        if written <= 0:
            raise OSError("activity append made no progress")
        offset += written


def _require_canonical_activity_target(
    root: Path, path: Path, *, allow_missing: bool
) -> None:
    root = Path(os.path.abspath(root))
    path = Path(os.path.abspath(path))
    if path != root / "runtime" / "activity.jsonl":
        raise ValueError("activity path must be the canonical runtime journal")
    if root.is_symlink() or not root.is_dir():
        raise ValueError("activity root is unsafe")
    runtime = root / "runtime"
    if runtime.is_symlink() or not runtime.is_dir():
        raise ValueError("activity runtime directory is unsafe")
    if path.exists() or path.is_symlink():
        if not _regular_nofollow(path):
            raise ValueError("activity journal is unsafe")
    elif not allow_missing:
        raise ValueError("activity journal is missing")


def _regular_nofollow(path: Path) -> bool:
    try:
        mode = path.lstat().st_mode
    except OSError:
        return False
    return stat.S_ISREG(mode)


def _raw_lines_nofollow(path: Path) -> Iterator[tuple[bytes, bool]]:
    with open_regular_nofollow(path) as handle:
        while True:
            line = handle.readline(MAX_ACTIVITY_LINE_BYTES + 2)
            if not line:
                break
            oversized = len(line) > MAX_ACTIVITY_LINE_BYTES + 1
            if oversized and not line.endswith(b"\n"):
                while True:
                    remainder = handle.readline(MAX_ACTIVITY_LINE_BYTES + 2)
                    if not remainder or remainder.endswith(b"\n"):
                        break
            yield line, oversized


def _tail_lines_nofollow(path: Path) -> list[bytes]:
    """Read at most one bounded physical-row window from the journal tail."""

    window_bytes = (MAX_ACTIVITY_READ_ROWS + 1) * (MAX_ACTIVITY_LINE_BYTES + 1)
    with open_regular_nofollow(path) as handle:
        size = os.fstat(handle.fileno()).st_size
        offset = max(0, size - window_bytes)
        handle.seek(offset)
        raw = handle.read(window_bytes)
    if offset:
        separator = raw.find(b"\n")
        raw = b"" if separator < 0 else raw[separator + 1 :]
    if raw and not raw.endswith(b"\n"):
        separator = raw.rfind(b"\n")
        raw = b"" if separator < 0 else raw[: separator + 1]
    return raw.splitlines()[-MAX_ACTIVITY_READ_ROWS:]
