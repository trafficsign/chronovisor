"""Frequent lightweight worker for autonomous queue convergence."""

from __future__ import annotations

import argparse
import json
from typing import Any


def run_maintenance_batch(
    *,
    lint_limit: int = 50,
    orphan_limit: int = 8,
    max_elapsed_seconds: float = 15 * 60,
) -> dict[str, Any]:
    """Drain existing semantic maintenance without rebuilding sleep artifacts."""

    from chronovisor.autonomy import resolve_deferred_duplicates_with_frontier
    from chronovisor.content_correction import run_pending_corrections
    from chronovisor.convergence import ConvergenceStore, CycleBudget
    from chronovisor.duplicate_review import build_duplicate_review_queue
    from chronovisor.lint_repair import run_lint_repair
    from chronovisor.orphan_link import run_autonomous

    state = ConvergenceStore()
    budget = CycleBudget(
        max_local_calls=30,
        max_frontier_calls=28,
        max_mutations=60,
        max_elapsed_seconds=max(1.0, float(max_elapsed_seconds)),
    )

    def lane(name: str, fn) -> dict[str, Any]:
        try:
            result = fn()
        except Exception as exc:
            return {
                "status": "error",
                "lane": name,
                "error": f"{exc.__class__.__name__}: {exc}",
            }
        return (
            result
            if isinstance(result, dict)
            else {
                "status": "error",
                "lane": name,
                "error": "lane returned a non-object result",
            }
        )

    payload: dict[str, Any] = {
        "status": "ok",
        "lease_recovery": lane(
            "lease_recovery",
            lambda: state.reap_expired_leases(dry_run=False),
        ),
        "content_corrections": lane(
            "content_corrections",
            lambda: run_pending_corrections(
                max_items=2,
                store=state,
                budget=budget.slice(
                    max_local_calls=6,
                    max_frontier_calls=6,
                    max_mutations=3,
                ),
            ),
        ),
    }
    duplicate_records = lane(
        "duplicate_inventory",
        lambda: {
            "status": "ok",
            "records": build_duplicate_review_queue(limit=300),
        },
    )
    payload["duplicate_inventory"] = duplicate_records
    records = (
        duplicate_records.get("records")
        if isinstance(duplicate_records.get("records"), list)
        else []
    )
    payload["duplicates"] = lane(
        "duplicates",
        lambda: resolve_deferred_duplicates_with_frontier(
            records,
            convergence_store=state,
            budget=budget.slice(max_frontier_calls=6, max_mutations=6),
            dry_run=False,
        ),
    )
    payload["lint_repair"] = lane(
        "lint_repair",
        lambda: run_lint_repair(
            store=state,
            max_items=max(0, int(lint_limit)),
            budget=budget.slice(
                max_local_calls=12,
                max_frontier_calls=8,
                max_mutations=20,
            ),
            dry_run=False,
        ),
    )
    payload["orphan_links"] = lane(
        "orphan_links",
        lambda: run_autonomous(
            orphan_limit=max(0, int(orphan_limit)),
            max_candidates=3,
            convergence_store=state,
            budget=budget.slice(
                max_local_calls=12,
                max_frontier_calls=8,
                max_mutations=8,
            ),
            dry_run=False,
        ),
    )
    payload["budget"] = budget.snapshot()
    if any(
        isinstance(value, dict) and value.get("status") in {"error", "attention"}
        for value in payload.values()
    ):
        payload["status"] = "attention"
    return payload


def run_converge(
    *,
    session_limit: int = 4,
    job_limit: int = 8,
    run_sleep: bool = False,
    run_system_repairs: bool = True,
    run_maintenance: bool = True,
    lint_limit: int = 50,
    orphan_limit: int = 8,
    maintenance_max_elapsed_seconds: float = 15 * 60,
) -> dict[str, Any]:
    from chronovisor.background_jobs import retry_due
    from chronovisor.self_heal import enqueue_due_system_repairs
    from chronovisor.session_sweeper import run_sweeper

    payload: dict[str, Any] = {
        "status": "ok",
        "system_repairs": (
            enqueue_due_system_repairs(limit=min(2, job_limit))
            if run_system_repairs
            else {"status": "skipped", "reason": "disabled_by_cli"}
        ),
        "background_jobs": retry_due(limit=job_limit),
        "session_sweeper": run_sweeper(limit=session_limit),
        "maintenance": (
            run_maintenance_batch(
                lint_limit=lint_limit,
                orphan_limit=orphan_limit,
                max_elapsed_seconds=maintenance_max_elapsed_seconds,
            )
            if run_maintenance
            else {"status": "skipped", "reason": "disabled_by_cli"}
        ),
    }
    if run_sleep:
        from chronovisor.sleep_cycle import run_sleep_cycle

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
    parser.add_argument("--lint-limit", type=int, default=50)
    parser.add_argument("--orphan-limit", type=int, default=8)
    parser.add_argument(
        "--maintenance-max-elapsed-seconds",
        type=float,
        default=15 * 60,
    )
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
    parser.add_argument(
        "--no-system-repairs",
        dest="run_system_repairs",
        action="store_false",
        help="Do not enqueue exceptional system-code repair work.",
    )
    parser.set_defaults(run_system_repairs=True)
    parser.add_argument(
        "--no-maintenance",
        dest="run_maintenance",
        action="store_false",
        help="Do not drain bounded lint/correction/duplicate/orphan maintenance.",
    )
    parser.set_defaults(run_maintenance=True)
    args = parser.parse_args(argv)
    print(
        json.dumps(
            run_converge(
                session_limit=max(0, args.session_limit),
                job_limit=max(0, args.job_limit),
                run_sleep=args.run_sleep,
                run_system_repairs=args.run_system_repairs,
                run_maintenance=args.run_maintenance,
                lint_limit=max(0, args.lint_limit),
                orphan_limit=max(0, args.orphan_limit),
                maintenance_max_elapsed_seconds=max(
                    1.0,
                    args.maintenance_max_elapsed_seconds,
                ),
            ),
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
