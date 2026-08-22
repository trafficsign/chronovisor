"""Strict JSONL helpers.

JSONL records are separated by LF bytes.  ``str.splitlines()`` is unsuitable
because it also treats Unicode separators such as U+2028 as record boundaries,
silently dropping otherwise valid JSON strings.
"""

from __future__ import annotations

import json
import mmap
import os
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any


def encode_jsonl(
    rows: Iterable[Mapping[str, Any]],
    *,
    sort_keys: bool = True,
) -> str:
    """Encode mappings as LF-delimited UTF-8 JSON text."""
    return "".join(
        json.dumps(dict(row), ensure_ascii=False, sort_keys=sort_keys) + "\n"
        for row in rows
    )


def write_jsonl(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
    *,
    sort_keys: bool = True,
) -> None:
    """Replace a JSONL file without an atomicity or durability guarantee."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(encode_jsonl(rows, sort_keys=sort_keys), encoding="utf-8")


def iter_jsonl(
    path: Path, *, limit: int | None = None
) -> Iterator[dict[str, Any]]:
    """Stream valid rows, seeking directly to a bounded physical-line tail."""

    try:
        with path.open("rb") as handle:
            size = os.fstat(handle.fileno()).st_size
            if not size:
                return
            with mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as data:
                start = 0
                if limit is not None:
                    cursor = size - int(data[size - 1 : size] == b"\n")
                    for _ in range(max(1, limit)):
                        separator = data.rfind(b"\n", 0, cursor)
                        if separator < 0:
                            cursor = 0
                            break
                        cursor = separator
                    start = cursor + int(cursor > 0)
                while start < size:
                    end = data.find(b"\n", start)
                    if end < 0:
                        end = size
                    raw_line = data[start:end]
                    start = end + 1
                    if raw_line.endswith(b"\r"):
                        raw_line = raw_line[:-1]
                    if not raw_line.strip():
                        continue
                    try:
                        value = json.loads(raw_line)
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        continue
                    if isinstance(value, dict):
                        yield value
    except (OSError, ValueError):
        return


def read_jsonl(path: Path, *, limit: int | None = None) -> list[dict[str, Any]]:
    return list(iter_jsonl(path, limit=limit))


def count_jsonl(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())
    except OSError:
        return 0
