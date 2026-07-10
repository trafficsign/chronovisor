"""Crash-tolerant durable JSONL appends."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping


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
