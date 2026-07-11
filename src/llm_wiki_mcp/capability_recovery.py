"""Resume external-authority queues after frontier capability is restored."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llm_wiki_mcp.jsonl import read_jsonl
from llm_wiki_mcp.link_fix import atomic_write
from llm_wiki_mcp.wiki import WIKI_ROOT


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    atomic_write(path, "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, default=str) + "\n" for row in rows))


def resume_external_queues(*, preflight: dict[str, Any], dry_run: bool = False) -> dict[str, Any]:
    if preflight.get("ok") is not True:
        return {"status": "skipped", "reason": "frontier_preflight_not_ok", "resumed": 0}
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    resumed = 0

    ledger_path = WIKI_ROOT / "runtime" / "ingest-read-back-repair.json"
    try:
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        ledger = {}
    entries = ledger.get("entries") if isinstance(ledger, dict) else None
    if isinstance(entries, dict):
        for entry in entries.values():
            if isinstance(entry, dict) and entry.get("status") == "human_required":
                entry["status"] = "pending"
                entry["next_retry_at"] = now
                entry["human_required"] = False
                entry["capability_resumed_at"] = now
                resumed += 1
        if not dry_run and resumed:
            atomic_write(ledger_path, json.dumps(ledger, ensure_ascii=False, indent=2, sort_keys=True) + "\n")

    for path, status_field, retry_status in (
        (WIKI_ROOT / "review" / "raw-replay-queue.jsonl", "status", "indeterminate"),
        (WIKI_ROOT / "recall" / "search-label-queue.jsonl", "queue_status", "pending_frontier_review"),
    ):
        rows = read_jsonl(path)
        changed = False
        for row in rows:
            if row.get(status_field) != "human_required":
                continue
            row[status_field] = retry_status
            row["human_required"] = False
            row["capability_resumed_at"] = now
            if status_field == "status":
                row["next_frontier_retry_at"] = now
            changed = True
            resumed += 1
        if changed and not dry_run:
            _write_jsonl(path, rows)
    return {"status": "ok", "resumed": resumed, "dry_run": dry_run}
