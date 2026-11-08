"""The scheduled 120 s cycle: capture retries, the missed-session sweep, then the index refresh (the only scheduled index writer)."""

from __future__ import annotations

import argparse
import json
from typing import Any

from chronovisor.capture import capture_status
from chronovisor.capture.background_jobs import retry_due, snapshot
from chronovisor.capture.session_sweeper import run_sweeper
from chronovisor.common.config import RuntimeConfigError
from chronovisor.common.timeutil import utc_now

STATUS_FILE = capture_status.status_path()
RETRY_LIMIT = 8
SWEEP_LIMIT = 8
_ACTIVE_STATUSES = ("queued", "running", "retry_wait")
_FAILED_STATUSES = ("failed", "quarantined")


def _job_wrote_raw(result: dict[str, Any]) -> bool:
    if result.get("status") != "completed":
        return False
    output = result.get("output_tail")
    if not isinstance(output, str):
        return False
    for line in output.splitlines():
        try:
            payload = json.loads(line)
        except ValueError:
            continue
        if isinstance(payload, dict) and payload.get("status") == "saved":
            return True
    return '"status":"saved"' in output.replace(" ", "").replace("\n", "")


def _job_counts(state: dict[str, Any]) -> tuple[int, int, str | None, bool]:
    if state.get("status") != "ok":
        return 0, 1, None, False
    by_status = state.get("by_status")
    by_status = by_status if isinstance(by_status, dict) else {}
    pending = sum(
        int(by_status.get(name) or 0)
        for name in _ACTIVE_STATUSES
        if isinstance(by_status.get(name), int)
    )
    failed = sum(
        int(by_status.get(name) or 0)
        for name in _FAILED_STATUSES
        if isinstance(by_status.get(name), int)
    )
    oldest = state.get("oldest_pending_at")
    return pending, failed, oldest if isinstance(oldest, str) else None, True


def run_capture(*, limit: int = RETRY_LIMIT) -> dict[str, Any]:
    """Run both capture triggers and atomically publish the observed status."""
    previous = capture_status.read(STATUS_FILE) or {}
    started = utc_now()
    error: str | None = None
    jobs: dict[str, Any] = {}
    sweep: dict[str, Any] = {}
    wrote_raw = False

    try:
        jobs = retry_due(limit=limit)
        wrote_raw = any(
            _job_wrote_raw(result)
            for result in jobs.get("results", [])
            if isinstance(result, dict)
        )
    except RuntimeConfigError:
        error = "config_invalid"
    except Exception:
        error = "capture_failed"

    try:
        sweep = run_sweeper(limit=limit)
        wrote_raw = wrote_raw or any(
            isinstance(result, dict) and result.get("status") == "saved"
            for result in sweep.get("results", [])
        )
    except RuntimeConfigError:
        error = "config_invalid"
    except Exception:
        error = error or "capture_failed"

    pending, failed, oldest_pending_at, jobs_available = _job_counts(snapshot())
    if not jobs_available and error is None:
        error = "capture_failed"
    sweep_results = [
        result for result in sweep.get("results", []) if isinstance(result, dict)
    ]
    sweep_failures = sum(
        1 for result in sweep_results if result.get("status") == "error"
    )
    failed += sweep_failures

    job_results = [
        result for result in jobs.get("results", []) if isinstance(result, dict)
    ]
    retrying = bool(sweep_failures) or any(
        result.get("status") == "retry_wait" for result in job_results
    )
    if error is not None:
        last_error = error
    elif failed:
        last_error = "permanent_failure"
    elif retrying:
        last_error = "transient_failure"
    else:
        last_error = "none"

    status = {
        "last_attempt_at": capture_status.timestamp(started),
        "last_success_at": (
            capture_status.timestamp()
            if error is None
            else previous.get("last_success_at")
        ),
        "last_commit_at": (
            capture_status.timestamp() if wrote_raw else previous.get("last_commit_at")
        ),
        "oldest_pending_at": oldest_pending_at,
        "pending": pending,
        "failed": failed,
        "last_error": last_error,
    }
    capture_status.write(STATUS_FILE, status)
    return {**status, "ok": error is None}


def main(argv: list[str] | None = None) -> int:
    """Run the ``chronovisor-capture`` command-line entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=RETRY_LIMIT)
    args = parser.parse_args(argv)
    result = run_capture(limit=max(0, args.limit))
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "last_attempt_at",
                    "last_success_at",
                    "last_commit_at",
                    "pending",
                    "failed",
                    "last_error",
                )
            },
            ensure_ascii=False,
        )
    )
    return 0 if result["ok"] else 1
