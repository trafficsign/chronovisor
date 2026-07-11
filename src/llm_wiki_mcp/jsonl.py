"""Strict JSONL helpers.

JSONL records are separated by LF bytes.  ``str.splitlines()`` is unsuitable
because it also treats Unicode separators such as U+2028 as record boundaries,
silently dropping otherwise valid JSON strings.
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Any


def read_jsonl(path: Path, *, limit: int | None = None) -> list[dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            lines = deque(handle, maxlen=max(1, limit)) if limit is not None else handle
            rows: list[dict[str, Any]] = []
            for raw_line in lines:
                line = raw_line.rstrip("\n")
                if line.endswith("\r"):
                    line = line[:-1]
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict):
                    rows.append(value)
            return rows
    except OSError:
        return []


def count_jsonl(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())
    except OSError:
        return 0
