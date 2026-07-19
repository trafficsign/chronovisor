"""Append-only recall-feedback ledger helpers.

Feedback is operational evidence, so migrations must never rewrite or delete
historical JSONL rows.  A retraction instead names both the producer key and
the canonical digest of one exact ``page_ignored`` row.  Consumers use this
module to hide only that row; malformed or incomplete retractions fail closed
and leave the original feedback active.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


PAGE_IGNORED_RETRACTION_KIND = "page_ignored_retracted"


def read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    """Read valid object rows while tolerating interrupted JSONL tails."""

    try:
        lines = path.read_text(encoding="utf-8").split("\n")
    except OSError:
        return []
    rows: list[dict[str, Any]] = []
    for line in lines:
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def feedback_row_sha256(row: dict[str, Any]) -> str:
    """Return the stable identity used by an exact-row retraction."""

    payload = json.dumps(
        row,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def retracted_page_ignored_targets(
    rows: Iterable[dict[str, Any]],
) -> set[tuple[str, str]]:
    """Return fully bound ``(producer key, target digest)`` retractions."""

    targets: set[tuple[str, str]] = set()
    for row in rows:
        if row.get("kind") != PAGE_IGNORED_RETRACTION_KIND:
            continue
        if row.get("target_kind") != "page_ignored":
            continue
        key = row.get("content_correction_key")
        digest = row.get("target_feedback_sha256")
        if (
            isinstance(key, str)
            and key
            and isinstance(digest, str)
            and len(digest) == 64
        ):
            targets.add((key, digest))
    return targets


def active_feedback_rows(path: Path) -> list[dict[str, Any]]:
    """Return semantic feedback after exact append-only retractions."""

    rows = read_jsonl_rows(path)
    retracted = retracted_page_ignored_targets(rows)
    active: list[dict[str, Any]] = []
    for row in rows:
        if row.get("kind") == PAGE_IGNORED_RETRACTION_KIND:
            continue
        key = row.get("content_correction_key")
        if (
            row.get("kind") == "page_ignored"
            and isinstance(key, str)
            and (key, feedback_row_sha256(row)) in retracted
        ):
            continue
        active.append(row)
    return active
