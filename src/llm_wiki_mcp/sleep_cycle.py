"""Nightly consolidation runner for LLM Wiki.

The sleep cycle is deliberately a coordinator, not a new intelligence layer:
it snapshots first, refreshes cheap eval/graph queues, and leaves destructive
merge or re-ingest work behind explicit flags.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from typing import Any

from llm_wiki_mcp.wiki import WIKI_ROOT

HISTORY_FILE = WIKI_ROOT / "runtime" / "sleep-cycle-history.jsonl"


def _append_history(row: dict[str, Any]) -> None:
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with HISTORY_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def run_sleep_cycle(
    *,
    raw_limit: int = 100,
    eval_limit: int = 100,
    duplicate_limit: int = 200,
    dry_run: bool = False,
) -> dict[str, Any]:
    from llm_wiki_mcp.cofire import build_cofire_graph
    from llm_wiki_mcp.duplicate_review import build_duplicate_review_queue, write_review_queue
    from llm_wiki_mcp.memory_integrity import run_eval
    from llm_wiki_mcp.prefetch import build_prefetch_cache
    from llm_wiki_mcp.raw_replay import QUEUE_FILE, build_queue, select_raws
    from llm_wiki_mcp.wiki_snapshot import snapshot_wiki

    started = datetime.now().isoformat(timespec="seconds")
    snapshot = (
        {"status": "skipped", "reason": "dry_run"}
        if dry_run
        else snapshot_wiki("before sleep cycle")
    )
    cofire = build_cofire_graph(write=not dry_run)
    prefetch = build_prefetch_cache(write=not dry_run)
    integrity = run_eval(limit=max(0, eval_limit), write=not dry_run)
    raw_replay = (
        {
            "status": "dry_run",
            "queue": str(QUEUE_FILE),
            "count": len(select_raws(limit=max(0, raw_limit))),
        }
        if dry_run
        else build_queue(limit=max(0, raw_limit))
    )
    duplicates = build_duplicate_review_queue(limit=max(0, duplicate_limit))
    duplicate_path = ""
    if not dry_run:
        duplicate_path = str(write_review_queue(duplicates))
    payload = {
        "status": "ok",
        "started_at": started,
        "dry_run": dry_run,
        "wiki_snapshot": snapshot,
        "cofire": {k: v for k, v in cofire.items() if k != "graph"},
        "prefetch": {
            "status": prefetch.get("status"),
            "episodes": prefetch.get("episodes", 0),
            "buckets": len(prefetch.get("buckets", {})),
            "tokens": len(prefetch.get("tokens", {})),
        },
        "memory_integrity": {k: v for k, v in integrity.items() if k != "rows"},
        "raw_replay": raw_replay,
        "duplicates": {
            "count": len(duplicates),
            "path": duplicate_path,
        },
    }
    _append_history(payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run LLM Wiki sleep consolidation.")
    parser.add_argument("--raw-limit", type=int, default=100)
    parser.add_argument("--eval-limit", type=int, default=100)
    parser.add_argument("--duplicate-limit", type=int, default=200)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    payload = run_sleep_cycle(
        raw_limit=max(0, args.raw_limit),
        eval_limit=max(0, args.eval_limit),
        duplicate_limit=max(0, args.duplicate_limit),
        dry_run=args.dry_run,
    )
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    else:
        print(f"cofire_edges\t{payload['cofire']['edges']}")
        print(f"prefetch_buckets\t{payload['prefetch']['buckets']}")
        print(f"capture_rate\t{payload['memory_integrity']['capture_rate']}")
        print(f"duplicates\t{payload['duplicates']['count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
