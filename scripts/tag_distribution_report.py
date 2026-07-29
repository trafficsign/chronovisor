#!/usr/bin/env python3
"""Tag distribution report (plan-3).

Read-only sweep: samples 200 pages (100 proportional + 100 minority
oversample), asks the LLM to tag each one twice (master-list-only +
free-form taxonomy gap), and writes an aggregate report. Pages on disk
are NEVER modified.

Usage:
    python3 scripts/tag_distribution_report.py
    python3 scripts/tag_distribution_report.py --sample-a 20 --sample-b 20  # smoke
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from chronovisor.core.store import SYSTEM_DIR
from chronovisor.librarian.tag_distribution import run_dry_run


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output Markdown path. Default: ~/.chronovisor/system/tag-distribution-report-{today}.md",
    )
    parser.add_argument(
        "--raw-log",
        type=Path,
        default=None,
        help="Append-only LLM raw output log. Default: ~/.chronovisor/system/tag-report-raw-llm-output-{today}.jsonl",
    )
    parser.add_argument(
        "--sample-a",
        type=int,
        default=100,
        help="Proportional sample size (default 100).",
    )
    parser.add_argument(
        "--sample-b",
        type=int,
        default=100,
        help="Minority oversample size (default 100).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling (default 42).",
    )
    args = parser.parse_args()

    today = date.today().isoformat()
    output = args.output or (SYSTEM_DIR / f"tag-distribution-report-{today}.md")
    raw_log = args.raw_log or (SYSTEM_DIR / f"tag-report-raw-llm-output-{today}.jsonl")

    stats = run_dry_run(
        output,
        raw_log,
        sample_a_n=args.sample_a,
        sample_b_n=args.sample_b,
        seed=args.seed,
    )
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
