"""Durable background worker for deep retrieval v1/v2."""

from __future__ import annotations

import argparse
import json
import sys

from llm_wiki_mcp.deep_retrieval import run_deep_dive, run_deep_dive_v2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--engine", choices=("v1", "v2"), default="v2")
    args = parser.parse_args(argv)
    try:
        payload = json.loads(sys.stdin.read() or "{}")
    except json.JSONDecodeError:
        payload = {}
    query = str(payload.get("query") or "") if isinstance(payload, dict) else ""
    if not query:
        print(json.dumps({"status": "failed", "error": "query is required"}))
        return 78
    options = {
        "max_iterations": int(payload.get("max_iterations") or 3),
        "fanout": int(payload.get("fanout") or 5),
        "semantic": payload.get("semantic") is not False,
        "use_llm": payload.get("use_llm") is not False,
    }
    if args.engine == "v2":
        from llm_wiki_mcp.research_config import load_research_config

        config = load_research_config()
        result = run_deep_dive_v2(query, config=config, **options)
    else:
        result = run_deep_dive(query, **options)
    result["worker_run_id"] = args.run_id
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
