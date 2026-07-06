"""Retroactive raw re-ingestion planner/runner."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from llm_wiki_mcp.jobs import job_store
from llm_wiki_mcp.wiki import RAW_DIR, WIKI_ROOT

RAW_DATE_RE = re.compile(r"(20\d{6})")
QUEUE_FILE = WIKI_ROOT / "review" / "raw-replay-queue.jsonl"
HISTORY_FILE = WIKI_ROOT / "runtime" / "raw-replay-history.jsonl"


def raw_date(path: Path) -> str:
    match = RAW_DATE_RE.search(path.name)
    return match.group(1) if match else datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y%m%d")


def select_raws(*, since: str = "", limit: int = 0) -> list[Path]:
    candidates = sorted(RAW_DIR.glob("*.md"), key=lambda path: (raw_date(path), path.name))
    if since:
        normalized = since.replace("-", "")
        candidates = [path for path in candidates if raw_date(path) >= normalized]
    if limit:
        candidates = candidates[:limit]
    return candidates


def build_queue(*, since: str = "", limit: int = 0, path: Path = QUEUE_FILE) -> dict[str, Any]:
    raws = select_raws(since=since, limit=limit)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "type": "raw_replay_candidate",
            "raw": raw.name,
            "path": str(raw),
            "date": raw_date(raw),
            "bytes": raw.stat().st_size,
            "status": "pending",
        }
        for raw in raws
    ]
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return {"status": "ok", "queue": str(path), "count": len(rows)}


def _append_history(row: dict[str, Any]) -> None:
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with HISTORY_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def run_replay(*, since: str = "", limit: int = 1) -> dict[str, Any]:
    from llm_wiki_mcp.ingest import run_ingest

    raws = select_raws(since=since, limit=limit)
    runs: list[dict[str, Any]] = []
    for raw in raws:
        try:
            content = raw.read_text(encoding="utf-8")
        except OSError as exc:
            record = {"raw": raw.name, "status": "error", "error": str(exc)}
            runs.append(record)
            _append_history(record)
            continue
        job = job_store.create(processor="ollama")
        run_ingest(
            content,
            job.job_id,
            metadata={"source_raw": f"replay:{raw.name}"},
        )
        finished = job_store.get(job.job_id)
        record = {
            "raw": raw.name,
            "job_id": job.job_id,
            "status": getattr(finished.status, "value", str(finished.status)) if finished else "missing",
            "pages_created": finished.pages_created if finished else [],
            "pages_updated": finished.pages_updated if finished else [],
        }
        runs.append(record)
        _append_history(record)
    return {"status": "ok", "runs": runs, "count": len(runs)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plan or run retroactive raw re-ingestion.")
    parser.add_argument("--since", default="", help="YYYYMMDD or YYYY-MM-DD lower bound.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--run", action="store_true", help="Actually re-ingest selected raws.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.run:
        payload = run_replay(since=args.since, limit=max(1, args.limit or 1))
    else:
        payload = build_queue(since=args.since, limit=max(0, args.limit))
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    else:
        print("\t".join(f"{key}={value}" for key, value in payload.items() if key != "runs"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
