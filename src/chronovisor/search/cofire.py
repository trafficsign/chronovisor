"""Console owner for the recall co-fire graph builder."""

from __future__ import annotations

import argparse
import json

from chronovisor.core.store import CHRONOVISOR_ROOT, okf_runtime_operation
from chronovisor.recall.cofire import build_cofire_graph


def main(argv: list[str] | None = None) -> int:
    """Run the ``chronovisor-cofire`` command-line entry point."""
    parser = argparse.ArgumentParser(description="Build recall co-fire graph.")
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--min-count", type=int, default=2)
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.no_write:
        return _main_locked(args)
    from chronovisor.core.okf_cutover import OKFStartupBlocked

    try:
        with okf_runtime_operation(CHRONOVISOR_ROOT):
            return _main_locked(args)
    except OKFStartupBlocked:
        print(json.dumps({"status": "blocked", "category": "okf_startup_blocked"}))
        return 75


def _main_locked(args: argparse.Namespace) -> int:
    payload = build_cofire_graph(
        limit=max(1, args.limit),
        min_count=max(1, args.min_count),
        write=not args.no_write,
    )
    if args.json:
        print(
            json.dumps(
                {k: v for k, v in payload.items() if k not in {"graph", "graphs"}},
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print(f"episodes\t{payload['episodes']}")
        print(f"nodes\t{payload['nodes']}")
        print(f"edges\t{payload['edges']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
