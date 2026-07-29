"""Durable background-job entry point for research runs."""

from __future__ import annotations

import argparse
import json
import sys

from chronovisor.research.research_service import run_evidence_research


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--purpose", default="explicit")
    args = parser.parse_args(argv)
    try:
        payload = json.loads(sys.stdin.read() or "{}")
    except json.JSONDecodeError:
        payload = {}
    goal = str(payload.get("goal") or "").strip() if isinstance(payload, dict) else ""
    if not goal:
        print(json.dumps({"status": "failed", "error": "goal is required"}))
        return 78
    claims = payload.get("claims") if isinstance(payload.get("claims"), list) else None
    result = run_evidence_research(
        goal,
        claims=claims,
        challenge=payload.get("challenge") is not False,
        purpose=args.purpose,
        run_id=args.run_id,
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result.get("status") in {"completed", "terminal"} else 75


if __name__ == "__main__":
    raise SystemExit(main())
