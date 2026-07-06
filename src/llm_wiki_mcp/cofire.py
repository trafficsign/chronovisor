"""Recall co-fire graph.

Pages that repeatedly appear together in recall contexts become weak edges.
Search graph expansion can then use actual recall behavior, not only wiki links.
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter, deque
from datetime import datetime
from pathlib import Path
from typing import Any

from llm_wiki_mcp.recall_runtime_paths import RECALL_DIR

RECALL_LOG_FILE = RECALL_DIR / "recall-log.jsonl"
COFIRE_FILE = RECALL_DIR / "cofire.json"


def _read_recent_jsonl(path: Path, *, limit: int = 5000) -> list[dict[str, Any]]:
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


def page_ids_from_record(row: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for item in row.get("context_items", []) or []:
        if isinstance(item, dict) and isinstance(item.get("page_id"), str):
            out.append(item["page_id"])
    for key in ("injected_pages", "expected_pages"):
        for page_id in row.get(key, []) or []:
            if isinstance(page_id, str):
                out.append(page_id)
    seen: set[str] = set()
    deduped: list[str] = []
    for page_id in out:
        if page_id in seen:
            continue
        seen.add(page_id)
        deduped.append(page_id)
    return deduped


def build_cofire_graph(
    *,
    log_file: Path = RECALL_LOG_FILE,
    output_file: Path = COFIRE_FILE,
    limit: int = 5000,
    min_count: int = 2,
    write: bool = True,
) -> dict[str, Any]:
    rows = _read_recent_jsonl(log_file, limit=limit)
    pair_counts: Counter[tuple[str, str]] = Counter()
    node_counts: Counter[str] = Counter()
    episodes = 0
    for row in rows:
        page_ids = page_ids_from_record(row)
        if len(page_ids) < 2:
            continue
        episodes += 1
        for page_id in page_ids:
            node_counts[page_id] += 1
        for left, right in itertools.combinations(sorted(page_ids), 2):
            pair_counts[(left, right)] += 1

    edges: dict[str, list[dict[str, Any]]] = {}
    for (left, right), count in pair_counts.items():
        if count < min_count:
            continue
        denom = max(node_counts[left], node_counts[right], 1)
        weight = count / denom
        edges.setdefault(left, []).append({"page_id": right, "count": count, "weight": round(weight, 4)})
        edges.setdefault(right, []).append({"page_id": left, "count": count, "weight": round(weight, 4)})
    for page_id, items in edges.items():
        items.sort(key=lambda item: (float(item["weight"]), int(item["count"])), reverse=True)
        edges[page_id] = items[:20]

    payload = {
        "status": "ok",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "log_file": str(log_file),
        "episodes": episodes,
        "nodes": len(edges),
        "edges": sum(len(items) for items in edges.values()),
        "min_count": min_count,
        "graph": edges,
    }
    if write:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def neighbors(page_id: str, *, path: Path = COFIRE_FILE, limit: int = 8) -> list[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    graph = payload.get("graph")
    if not isinstance(graph, dict):
        return []
    rows = graph.get(page_id, [])
    if not isinstance(rows, list):
        return []
    return [row for row in rows[:limit] if isinstance(row, dict) and isinstance(row.get("page_id"), str)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build recall co-fire graph.")
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--min-count", type=int, default=2)
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    payload = build_cofire_graph(
        limit=max(1, args.limit),
        min_count=max(1, args.min_count),
        write=not args.no_write,
    )
    if args.json:
        print(json.dumps({k: v for k, v in payload.items() if k != "graph"}, ensure_ascii=False, indent=2))
    else:
        print(f"episodes\t{payload['episodes']}")
        print(f"nodes\t{payload['nodes']}")
        print(f"edges\t{payload['edges']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
