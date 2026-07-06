"""Speculative recall prefetch cache.

The first version is log-derived: after recall succeeds, the sleep cycle can
compile lightweight host/cwd/query-token buckets. Runtime recall checks these
buckets before expensive search expansion.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Any

from llm_wiki_mcp.recall_runtime_paths import RECALL_DIR

RECALL_LOG_FILE = RECALL_DIR / "recall-log.jsonl"
PREFETCH_FILE = RECALL_DIR / "prefetch.json"


def _tokens(text: str) -> set[str]:
    return {
        token.lower()
        for token in re.findall(r"[a-z0-9][a-z0-9_.+-]{2,}|[\u3040-\u30ff\u3400-\u9fff]{2,}", text.lower())
        if token not in {"codex", "claude", "wiki", "llm", "project", "memory"}
    }


def _read_recent_jsonl(path: Path, *, limit: int) -> list[dict[str, Any]]:
    try:
        with path.open(encoding="utf-8") as f:
            lines = deque(f, maxlen=max(1, limit))
    except OSError:
        return []
    rows: list[dict[str, Any]] = []
    for line in lines:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _page_ids(row: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for item in row.get("context_items", []) or []:
        if isinstance(item, dict) and isinstance(item.get("page_id"), str):
            out.append(item["page_id"])
    return list(dict.fromkeys(out))


def build_prefetch_cache(
    *,
    log_file: Path = RECALL_LOG_FILE,
    output_file: Path = PREFETCH_FILE,
    limit: int = 5000,
    write: bool = True,
) -> dict[str, Any]:
    buckets: dict[str, Counter[str]] = defaultdict(Counter)
    token_index: dict[str, Counter[str]] = defaultdict(Counter)
    episodes = 0
    for row in _read_recent_jsonl(log_file, limit=limit):
        pages = _page_ids(row)
        if not pages:
            continue
        episodes += 1
        host = str(row.get("host") or "")
        cwd = str(row.get("cwd") or "")
        bucket_key = f"{host}|{Path(cwd).name if cwd else ''}"
        for page_id in pages:
            buckets[bucket_key][page_id] += 1
        query_text = " ".join(str(item) for item in row.get("queries", []) or [])
        query_text += " " + str(row.get("prompt") or "")
        for token in _tokens(query_text):
            for page_id in pages:
                token_index[token][page_id] += 1

    payload = {
        "status": "ok",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "episodes": episodes,
        "buckets": {
            key: [{"page_id": page_id, "count": count} for page_id, count in counter.most_common(12)]
            for key, counter in buckets.items()
        },
        "tokens": {
            key: [{"page_id": page_id, "count": count} for page_id, count in counter.most_common(8)]
            for key, counter in token_index.items()
        },
    }
    if write:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def prefetch_page_ids(
    *,
    host: str,
    cwd: str,
    queries: list[str],
    prompt: str = "",
    path: Path = PREFETCH_FILE,
    limit: int = 4,
) -> list[str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    scores: Counter[str] = Counter()
    buckets = payload.get("buckets")
    if isinstance(buckets, dict):
        key = f"{host}|{Path(cwd).name if cwd else ''}"
        for row in buckets.get(key, []) if isinstance(buckets.get(key), list) else []:
            if isinstance(row, dict) and isinstance(row.get("page_id"), str):
                scores[row["page_id"]] += int(row.get("count") or 1)
    tokens = payload.get("tokens")
    if isinstance(tokens, dict):
        for token in _tokens(" ".join(queries) + " " + prompt):
            rows = tokens.get(token)
            if not isinstance(rows, list):
                continue
            for row in rows:
                if isinstance(row, dict) and isinstance(row.get("page_id"), str):
                    scores[row["page_id"]] += int(row.get("count") or 1)
    return [page_id for page_id, _count in scores.most_common(limit)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build speculative recall prefetch cache.")
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    payload = build_prefetch_cache(limit=max(1, args.limit), write=not args.no_write)
    public = {key: value for key, value in payload.items() if key not in {"buckets", "tokens"}}
    public["bucket_count"] = len(payload.get("buckets", {}))
    public["token_count"] = len(payload.get("tokens", {}))
    if args.json:
        print(json.dumps(public, ensure_ascii=False, indent=2))
    else:
        print(f"episodes\t{public['episodes']}")
        print(f"buckets\t{public['bucket_count']}")
        print(f"tokens\t{public['token_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
