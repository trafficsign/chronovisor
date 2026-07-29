"""Durably enqueue time-sensitive ingest claims without verifying in-band."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from chronovisor.core.jsonl_write import append_jsonl_durable
from chronovisor.core.store import CHRONOVISOR_ROOT

CANDIDATE_QUEUE = CHRONOVISOR_ROOT / "review" / "freshness-candidates.jsonl"
TEMPORAL = re.compile(
    r"(?:最新|現在|今日|今週|今月|今年|最近|現行|価格|発売|version|release|current|latest|today|now)",
    re.IGNORECASE,
)


def _sentences(text: str) -> Iterable[str]:
    for part in re.split(r"(?<=[。！？.!?])\s+|\n+", text):
        compact = " ".join(part.split()).strip(" -*#\t")
        if 8 <= len(compact) <= 1_000 and TEMPORAL.search(compact):
            yield compact


def enqueue_from_operations(
    planned: Iterable[Any],
    *,
    path: Path = CANDIDATE_QUEUE,
    max_candidates: int = 50,
) -> dict[str, Any]:
    existing: set[str] = set()
    try:
        for line in path.read_text(encoding="utf-8").splitlines()[-20_000:]:
            row = json.loads(line)
            if isinstance(row, dict) and row.get("candidate_id"):
                existing.add(str(row["candidate_id"]))
    except (OSError, json.JSONDecodeError):
        pass
    rows: list[dict[str, Any]] = []
    for operation in planned:
        operation_path = Path(getattr(operation, "path", "") or "")
        if path == CANDIDATE_QUEUE:
            try:
                operation_path.resolve().relative_to((CHRONOVISOR_ROOT / "pages").resolve())
            except (OSError, ValueError):
                continue
        page_id = str(getattr(operation, "page_id", "") or "")
        content = str(getattr(operation, "new_body", "") or "")
        for claim in _sentences(content):
            digest = hashlib.sha256(f"{page_id}\0{claim}".encode()).hexdigest()
            candidate_id = f"freshness:{digest}"
            if candidate_id in existing:
                continue
            rows.append(
                {
                    "schema_version": 1,
                    "candidate_id": candidate_id,
                    "queued_at": datetime.now(UTC).isoformat(timespec="seconds"),
                    "page_id": page_id,
                    "claim": claim,
                    "claim_kind": "freshness-sensitive",
                    "reported_by_user": True,
                    "externally_verified": False,
                    "status": "queued",
                    "source": "approved_ingest_postimage",
                }
            )
            existing.add(candidate_id)
            if len(rows) >= max(0, max_candidates):
                break
        if len(rows) >= max(0, max_candidates):
            break
    if rows:
        append_jsonl_durable(path, rows, sort_keys=True)
    return {"status": "ok", "enqueued": len(rows), "candidate_ids": [row["candidate_id"] for row in rows]}
