"""Integrity scanner for append-only ingest read-back events."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class JsonlPrefix:
    records: tuple[dict[str, Any], ...]
    source_bytes: int
    complete_bytes: int
    complete_lines: int
    prefix_sha256: str
    tail_bytes: int
    invalid_lines: tuple[int, ...]

    @property
    def valid(self) -> bool:
        return self.tail_bytes == 0 and not self.invalid_lines

    def cursor(self, *, source_file: Path) -> dict[str, Any]:
        return {
            "source_file": str(source_file),
            "source_bytes": self.source_bytes,
            "complete_bytes": self.complete_bytes,
            "complete_lines": self.complete_lines,
            "records": len(self.records),
            "prefix_sha256": self.prefix_sha256,
            "tail_bytes": self.tail_bytes,
            "invalid_lines": list(self.invalid_lines),
            "integrity": "ok" if self.valid else "invalid",
        }


def scan_jsonl_prefix(path: Path) -> JsonlPrefix:
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        raw = b""
    except OSError as exc:
        raise RuntimeError(f"cannot read canonical JSONL source: {exc}") from exc
    if not raw:
        complete = b""
        tail = b""
    elif raw.endswith(b"\n"):
        complete = raw
        tail = b""
    else:
        boundary = raw.rfind(b"\n")
        complete = raw[: boundary + 1] if boundary >= 0 else b""
        tail = raw[boundary + 1 :]
    records: list[dict[str, Any]] = []
    invalid: list[int] = []
    lines = complete.splitlines()
    for line_number, encoded in enumerate(lines, start=1):
        if not encoded.strip():
            continue
        try:
            row = json.loads(encoded.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError):
            invalid.append(line_number)
            continue
        if not isinstance(row, dict):
            invalid.append(line_number)
            continue
        records.append(row)
    return JsonlPrefix(
        records=tuple(records),
        source_bytes=len(raw),
        complete_bytes=len(complete),
        complete_lines=len(lines),
        prefix_sha256=hashlib.sha256(complete).hexdigest(),
        tail_bytes=len(tail),
        invalid_lines=tuple(invalid),
    )


def cursor_is_prefix(previous: object, current: JsonlPrefix) -> bool:
    """Return whether a prior cursor still names this exact source prefix.

    The cheap current-prefix comparison is possible only when byte counts are
    equal.  A caller handling growth should hash exactly ``complete_bytes``
    from the source and compare it with the previous digest.
    """

    if not isinstance(previous, dict):
        return False
    prior_bytes = previous.get("complete_bytes")
    prior_sha = previous.get("prefix_sha256")
    return bool(
        isinstance(prior_bytes, int)
        and prior_bytes == current.complete_bytes
        and isinstance(prior_sha, str)
        and prior_sha == current.prefix_sha256
    )


def verify_prior_prefix(path: Path, previous: object) -> bool:
    if not isinstance(previous, dict):
        return False
    size = previous.get("complete_bytes")
    digest = previous.get("prefix_sha256")
    if (
        isinstance(size, bool)
        or not isinstance(size, int)
        or size < 0
        or not isinstance(digest, str)
    ):
        return False
    try:
        with path.open("rb") as stream:
            prefix = stream.read(size)
    except OSError:
        return False
    return len(prefix) == size and hashlib.sha256(prefix).hexdigest() == digest
