#!/usr/bin/env python3.14
"""Run bounded, resumable search-label frontier review batches.

The queue and golden-set writes stay inside
``search_eval.review_label_queue_with_frontier``. This wrapper only repeats
bounded calls until the configured deadline, item cap, or queue drain, so a
scheduled invocation can make progress without an unbounded model run.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

# Make the package importable when run directly from a checkout.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from chronovisor.ingest.convergence import CycleBudget
from chronovisor.search import search_eval

DEFAULT_BATCH_SIZE = 20
DEFAULT_MAX_ELAPSED_SECONDS = 5 * 60


def _as_nonnegative_int(value: object, *, field: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an integer") from exc
    if parsed < 0:
        raise ValueError(f"{field} must be non-negative")
    return parsed


def _as_nonnegative_float(value: object, *, field: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a number") from exc
    if parsed < 0:
        raise ValueError(f"{field} must be non-negative")
    return parsed


def _batch_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    """Keep the receipt metadata-only; queue rows never leave the reviewer."""
    return {
        "status": payload.get("status"),
        "attempted": int(payload.get("attempted") or 0),
        "promoted": int(payload.get("promoted") or 0),
        "remaining": int(payload.get("remaining") or 0),
        "status_counts": payload.get("status_counts", {}),
        "budget_exhausted": bool(payload.get("budget_exhausted")),
    }


def run_drain(
    *,
    queue_file: Path = search_eval.LABEL_QUEUE_FILE,
    golden_file: Path = search_eval.GOLDEN_FILE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_items: int = 0,
    max_elapsed_seconds: float = DEFAULT_MAX_ELAPSED_SECONDS,
    poll_seconds: float = 0.0,
    min_confidence: float = 0.8,
    votes: int = 1,
    timeout: int | None = None,
    max_attempts: int = 3,
    backoff_base_seconds: int = 6 * 60 * 60,
    review_order: str = "balanced",
    dry_run: bool = False,
) -> dict[str, Any]:
    """Drain due rows in bounded review calls and return a metadata receipt."""
    batch_size = _as_nonnegative_int(batch_size, field="batch_size")
    if batch_size == 0:
        raise ValueError("batch_size must be greater than zero")
    max_items = _as_nonnegative_int(max_items, field="max_items")
    max_elapsed_seconds = _as_nonnegative_float(
        max_elapsed_seconds, field="max_elapsed_seconds"
    )
    if max_elapsed_seconds == 0:
        raise ValueError("max_elapsed_seconds must be greater than zero")
    poll_seconds = _as_nonnegative_float(poll_seconds, field="poll_seconds")
    votes = max(1, _as_nonnegative_int(votes, field="votes"))
    max_attempts = max(1, _as_nonnegative_int(max_attempts, field="max_attempts"))
    backoff_base_seconds = _as_nonnegative_int(
        backoff_base_seconds, field="backoff_base_seconds"
    )
    if review_order not in {"balanced", "fifo"}:
        raise ValueError("review_order must be 'balanced' or 'fifo'")

    started = time.monotonic()
    attempted_total = 0
    promoted_total = 0
    remaining = 0
    batches: list[dict[str, Any]] = []
    last_payload: dict[str, Any] = {}
    stop_reason = ""

    while True:
        elapsed = time.monotonic() - started
        if elapsed >= max_elapsed_seconds:
            stop_reason = "deadline"
            break
        if max_items and attempted_total >= max_items:
            stop_reason = "max_items"
            break

        limit = batch_size
        if max_items:
            limit = min(limit, max_items - attempted_total)
        remaining_seconds = max_elapsed_seconds - (time.monotonic() - started)
        if remaining_seconds <= 0:
            stop_reason = "deadline"
            break
        budget = (
            CycleBudget(
                max_frontier_calls=limit * votes,
                max_mutations=1,
                max_elapsed_seconds=remaining_seconds,
            )
            if not dry_run
            else None
        )
        review_timeout = (
            timeout if timeout is not None else max(1, int(remaining_seconds))
        )
        try:
            payload = search_eval.review_label_queue_with_frontier(
                queue_file=queue_file,
                golden_file=golden_file,
                limit=limit,
                min_confidence=min_confidence,
                votes=votes,
                timeout=review_timeout,
                max_attempts=max_attempts,
                backoff_base_seconds=backoff_base_seconds,
                dry_run=dry_run,
                budget=budget,
                review_order=review_order,
            )
        except Exception as exc:  # pragma: no cover - defensive worker boundary
            stop_reason = "exception"
            return {
                "status": "error",
                "error_class": exc.__class__.__name__,
                "attempted": attempted_total,
                "promoted": promoted_total,
                "remaining": remaining,
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "batches": batches,
                "review_order": review_order,
                "dry_run": dry_run,
                "stop_reason": stop_reason,
            }

        last_payload = payload
        batch = _batch_metadata(payload)
        batches.append(batch)
        attempted = batch["attempted"]
        attempted_total += attempted
        promoted_total += batch["promoted"]
        remaining = batch["remaining"]

        if payload.get("status") != "ok":
            stop_reason = str(payload.get("status") or "review_error")
            break
        if bool(payload.get("budget_exhausted")):
            stop_reason = "budget_exhausted"
            break
        if dry_run:
            stop_reason = "dry_run"
            break
        if remaining == 0:
            stop_reason = "drained"
            break
        if attempted == 0:
            stop_reason = "waiting"
            break
        if poll_seconds:
            remaining_sleep = max_elapsed_seconds - (time.monotonic() - started)
            if remaining_sleep > 0:
                time.sleep(min(poll_seconds, remaining_sleep))

    elapsed = round(time.monotonic() - started, 3)
    if stop_reason == "drained":
        status = "drained"
    elif stop_reason in {
        "budget_exhausted",
        "budget_deferred",
        "concurrent_update_deferred",
        "error",
    }:
        status = stop_reason
    elif last_payload.get("status") not in {None, "ok"}:
        status = stop_reason or "review_error"
    elif dry_run:
        status = "dry_run"
    elif stop_reason == "waiting":
        status = "waiting"
    elif stop_reason == "exception":
        status = "error"
    else:
        status = "bounded"

    return {
        "status": status,
        "attempted": attempted_total,
        "promoted": promoted_total,
        "remaining": remaining,
        "elapsed_seconds": elapsed,
        "batches": batches,
        "last_status_counts": last_payload.get("status_counts", {}),
        "review_order": review_order,
        "batch_size": batch_size,
        "max_items": max_items,
        "max_elapsed_seconds": max_elapsed_seconds,
        "dry_run": dry_run,
        "stop_reason": stop_reason,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-file", type=Path, default=search_eval.LABEL_QUEUE_FILE)
    parser.add_argument("--golden-file", type=Path, default=search_eval.GOLDEN_FILE)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--max-items",
        type=int,
        default=0,
        help="Maximum rows attempted in one invocation; zero means deadline bounded.",
    )
    parser.add_argument(
        "--max-elapsed-seconds",
        type=float,
        default=DEFAULT_MAX_ELAPSED_SECONDS,
    )
    parser.add_argument("--poll-seconds", type=float, default=0.0)
    parser.add_argument("--frontier-min-confidence", type=float, default=0.8)
    parser.add_argument("--frontier-votes", type=int, default=1)
    parser.add_argument("--frontier-timeout", type=int, default=None)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--backoff-base-seconds", type=int, default=6 * 60 * 60)
    parser.add_argument(
        "--review-order",
        choices=("balanced", "fifo"),
        default="balanced",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_drain(
        queue_file=args.queue_file,
        golden_file=args.golden_file,
        batch_size=args.batch_size,
        max_items=args.max_items,
        max_elapsed_seconds=args.max_elapsed_seconds,
        poll_seconds=args.poll_seconds,
        min_confidence=args.frontier_min_confidence,
        votes=args.frontier_votes,
        timeout=args.frontier_timeout,
        max_attempts=args.max_attempts,
        backoff_base_seconds=args.backoff_base_seconds,
        review_order=args.review_order,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result.get("status") not in {"error"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
