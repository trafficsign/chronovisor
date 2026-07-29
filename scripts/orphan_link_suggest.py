#!/usr/bin/env python3
"""Dry-run orphan link suggestion (plan-2).

Generates ``~/.chronovisor/system/orphan-link-suggestions-{date}.md`` listing,
for each orphan page, the existing pages most likely to benefit from
gaining an inbound link to that orphan.

The report is diagnostic only. Pages on disk are NEVER modified; the scheduled
sleep lane sends proposals to the frontier reviewer and applies them
autonomously through the shared CAS writer.

Usage:
    python3 scripts/orphan_link_suggest.py             # full run, all orphans
    python3 scripts/orphan_link_suggest.py --limit 10  # smoke test
    python3 scripts/orphan_link_suggest.py --threshold 0.7
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

# Make the package importable when run from the repo root.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from chronovisor.ops.orphan_link import run_dry_run  # noqa: E402
from chronovisor.core.store import SYSTEM_DIR  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output Markdown path. Default: ~/.chronovisor/system/orphan-link-suggestions-{today}.md",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=5,
        help="Top-K candidate sources per orphan (default 5).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Drop suggestions below this confidence (default 0.5).",
    )
    parser.add_argument(
        "--semantic-top-n",
        type=int,
        default=20,
        help="How many semantic-search hits to consider per orphan (default 20).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Truncate the orphan list to this many entries (smoke testing).",
    )
    args = parser.parse_args()

    output = args.output or (
        SYSTEM_DIR / f"orphan-link-suggestions-{date.today().isoformat()}.md"
    )

    stats = run_dry_run(
        output,
        max_candidates=args.max_candidates,
        confidence_threshold=args.threshold,
        semantic_top_n=args.semantic_top_n,
        orphan_limit=args.limit,
    )
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
