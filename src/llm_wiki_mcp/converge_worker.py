"""Frequent lightweight worker for autonomous queue convergence."""

from __future__ import annotations

import argparse
import json
from typing import Any


def run_converge(*, session_limit: int = 4, job_limit: int = 8, run_sleep: bool = False) -> dict[str, Any]:
    from llm_wiki_mcp.background_jobs import retry_due
    from llm_wiki_mcp.self_heal import enqueue_due_system_repairs
    from llm_wiki_mcp.session_sweeper import run_sweeper

    payload: dict[str, Any] = {
        "status": "ok",
        "system_repairs": enqueue_due_system_repairs(limit=min(2, job_limit)),
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
    sleep_group = parser.add_mutually_exclusive_group()
    sleep_group.add_argument(
        "--with-sleep",
        dest="run_sleep",
        action="store_true",
        help="Explicitly opt in to the full daily sleep cycle.",
    )
    sleep_group.add_argument(
        "--no-sleep",
        dest="run_sleep",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.set_defaults(run_sleep=False)
    args = parser.parse_args(argv)
    print(json.dumps(run_converge(
        session_limit=max(0, args.session_limit),
        job_limit=max(0, args.job_limit),
        run_sleep=args.run_sleep,
    ), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
