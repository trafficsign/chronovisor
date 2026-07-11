"""Frequent lightweight worker for autonomous queue convergence."""

from __future__ import annotations

import argparse
import json
from typing import Any


def run_converge(*, session_limit: int = 4, job_limit: int = 8, run_sleep: bool = True) -> dict[str, Any]:
    from llm_wiki_mcp.background_jobs import retry_due
    from llm_wiki_mcp.session_sweeper import run_sweeper

    payload: dict[str, Any] = {
        "status": "ok",
        "background_jobs": retry_due(limit=job_limit),
        "session_sweeper": run_sweeper(limit=session_limit),
    }
    if run_sleep:
        from llm_wiki_mcp.sleep_cycle import run_sleep_cycle

        payload["sleep_cycle"] = run_sleep_cycle(
            raw_limit=25,
            eval_limit=25,
            duplicate_limit=100,
            dry_run=False,
        )
    if any(
        isinstance(value, dict) and value.get("status") in {"error", "attention"}
        for value in payload.values()
    ):
        payload["status"] = "attention"
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-limit", type=int, default=4)
    parser.add_argument("--job-limit", type=int, default=8)
    parser.add_argument("--no-sleep", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(run_converge(
        session_limit=max(0, args.session_limit),
        job_limit=max(0, args.job_limit),
        run_sleep=not args.no_sleep,
    ), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
