"""Search quality and latency eval through the MCP search tool, in process.

Eval sets are JSONL rows ``{"id", "query", "expected": [page_id, ...]}``; they hold
private queries, so they live under ``~/.chronovisor/eval/``, not in the repo.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

DEFAULT_SETS = (
    "~/.chronovisor/eval/search-manual-71.jsonl",
    "~/.chronovisor/eval/search-knownitem-200.jsonl",
)


def load_cases(paths: list[str]) -> list[dict[str, Any]]:
    cases = []
    for raw in paths:
        path = Path(raw).expanduser()
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                row["set"] = path.stem
                cases.append(row)
    return cases


def rank_of(expected: list[str], hits: list[str]) -> int | None:
    """1-based rank of the first expected page in hits, or None.

    Pages split after the set was reviewed live on as ``<id>-part-NN``; a part counts.
    """
    for i, page_id in enumerate(hits, 1):
        if any(page_id == e or page_id.startswith(f"{e}-part-") for e in expected):
            return i
    return None


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    if not n:
        return {"n": 0}
    ranks = [r["rank"] for r in rows]
    latency = sorted(r["ms"] for r in rows)
    return {
        "n": n,
        "hit@1": sum(r == 1 for r in ranks) / n,
        "hit@5": sum(r is not None and r <= 5 for r in ranks) / n,
        "hit@10": sum(r is not None for r in ranks) / n,
        "mrr": sum(1 / r for r in ranks if r) / n,
        "p50_ms": statistics.median(latency),
        "p95_ms": latency[min(n - 1, int(n * 0.95))],
        "rerank": _count(r["rerank"] for r in rows),
    }


def _count(values: Any) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        out[str(value)] = out.get(str(value), 0) + 1
    return out


def run(cases: list[dict[str, Any]], *, semantic: bool) -> dict[str, Any]:
    from chronovisor.hosts.server import chronovisor_search

    t0 = time.perf_counter()
    chronovisor_search(query="warm up", depth=0, semantic=semantic)
    cold_ms = (time.perf_counter() - t0) * 1000
    rows = []
    for case in cases:
        t = time.perf_counter()
        payload = json.loads(
            chronovisor_search(case["query"], depth=0, semantic=semantic)
        )
        ms = (time.perf_counter() - t) * 1000
        hits = [hit["page_id"] for hit in payload.get("direct_hits", [])]
        retrieval = payload.get("retrieval") or {}
        rows.append(
            {
                "id": case["id"],
                "set": case["set"],
                "rank": rank_of(case["expected"], hits),
                "ms": round(ms, 1),
                "mode": payload.get("search_mode"),
                "rerank": (payload.get("reranker") or {}).get("status"),
                "stages": retrieval.get("stage_timings_ms"),
                "hits": hits,
            }
        )
    sets = sorted({r["set"] for r in rows})
    return {
        "commit": os.environ.get("CHRONOVISOR_EVAL_LABEL", ""),
        "semantic": semantic,
        "cold_first_call_ms": round(cold_ms, 1),
        "summary": {
            "all": summarize(rows),
            **{s: summarize([r for r in rows if r["set"] == s]) for s in sets},
        },
        "rows": rows,
    }


def compare(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Paired comparison: per query, does B rank the answer better or worse than A."""
    ra = {r["id"]: r["rank"] for r in a["rows"]}
    better = worse = same = 0
    for row in b["rows"]:
        x, y = ra.get(row["id"]), row["rank"]
        sx, sy = (x or 99), (y or 99)
        if sy < sx:
            better += 1
        elif sy > sx:
            worse += 1
        else:
            same += 1
    return {"b_better": better, "b_worse": worse, "same": same}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", action="append", dest="sets")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--semantic", action="store_true")
    parser.add_argument("--out")
    parser.add_argument("--compare", nargs=2, metavar=("A", "B"))
    args = parser.parse_args(argv)
    if args.compare:
        a, b = (json.loads(Path(p).read_text()) for p in args.compare)
        print(json.dumps(compare(a, b)))
        return 0
    cases = load_cases(args.sets or list(DEFAULT_SETS))[: args.limit]
    result = run(cases, semantic=args.semantic)
    if args.out:
        Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=1))
    print(
        json.dumps(
            {"cold_first_call_ms": result["cold_first_call_ms"], **result["summary"]},
            ensure_ascii=False,
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
