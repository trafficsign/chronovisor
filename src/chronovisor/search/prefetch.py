"""Console owner for the core speculative recall prefetch cache."""

from __future__ import annotations

import argparse
import json

from chronovisor.core.prefetch import build_prefetch_cache
from chronovisor.core.store import CHRONOVISOR_ROOT, okf_runtime_operation


def main(argv: list[str] | None = None) -> int:
    """Run the ``chronovisor-prefetch`` command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Build speculative recall prefetch cache."
    )
    parser.add_argument("--limit", type=int, default=5000)
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
    payload = build_prefetch_cache(limit=max(1, args.limit), write=not args.no_write)
    public = {
        key: value
        for key, value in payload.items()
        if key not in {"buckets", "tokens", "features"}
    }
    public["bucket_count"] = len(payload.get("buckets", {}))
    public["token_count"] = len(payload.get("tokens", {}))
    public["positive_bucket_count"] = len(
        payload.get("features", {}).get("positive_used", {}).get("buckets", {})
    )
    if args.json:
        print(json.dumps(public, ensure_ascii=False, indent=2))
    else:
        print(f"episodes\t{public['episodes']}")
        print(f"buckets\t{public['bucket_count']}")
        print(f"tokens\t{public['token_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
