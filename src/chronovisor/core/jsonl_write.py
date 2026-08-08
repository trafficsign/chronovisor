"""Crash-tolerant durable JSONL appends."""

from __future__ import annotations

import json
import os
import tempfile
import threading
from collections.abc import Iterable, Mapping
from contextlib import suppress
from pathlib import Path
from typing import Any

from chronovisor.core.jsonl import encode_jsonl


def atomic_write_json_payload(path: Path, payload: Mapping[str, Any]) -> None:
    """Replace one sorted JSON payload without creating its parent directory."""

    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(0o600)
        os.replace(temporary, path)
    finally:
        with suppress(FileNotFoundError):
            temporary.unlink()


def atomic_replace_bytes(path: Path, data: bytes, *, mode: int = 0o600) -> None:
    """Replace a file through a same-directory, fsynced temporary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_temp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temp = Path(raw_temp)
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
        os.chmod(path, mode)
    finally:
        temp.unlink(missing_ok=True)


def atomic_replace_text(path: Path, content: str, *, mode: int = 0o600) -> None:
    """Replace UTF-8 text with the same atomic and permission contract."""
    atomic_replace_bytes(path, content.encode("utf-8"), mode=mode)


def write_jsonl_atomic(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
    *,
    sort_keys: bool = True,
    mode: int = 0o600,
) -> None:
    """Encode and atomically replace a JSONL file."""
    atomic_replace_text(
        path,
        encode_jsonl(rows, sort_keys=sort_keys),
        mode=mode,
    )


def append_jsonl_durable(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
    *,
    sort_keys: bool = False,
    default: Any = str,
) -> None:
    """Append records after delimiting any interrupted final JSON record.

    Callers own serialization (usually a lane-specific ``flock``).  This
    helper owns byte durability.  A torn tail is retained as an invalid
    historical row and terminated before valid retry rows are written, so it
    can never absorb those rows into one permanently unreadable line.
    """

    encoded = [
        (
            json.dumps(
                dict(row),
                ensure_ascii=False,
                sort_keys=sort_keys,
                default=default,
            )
            + "\n"
        ).encode("utf-8")
        for row in rows
    ]
    if not encoded:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    needs_separator = False
    try:
        with path.open("rb") as existing:
            existing.seek(0, os.SEEK_END)
            if existing.tell():
                existing.seek(-1, os.SEEK_END)
                needs_separator = existing.read(1) != b"\n"
    except FileNotFoundError:
        pass

    with path.open("ab") as handle:
        if needs_separator:
            handle.write(b"\n")
        for row in encoded:
            handle.write(row)
        handle.flush()
        os.fsync(handle.fileno())

    try:
        directory_fd = os.open(path.parent, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
