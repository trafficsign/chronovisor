"""Nightly consolidation runner for Chronovisor.

The sleep cycle snapshots first, refreshes cheap eval/graph artifacts, then
hands reversible maintenance decisions to the autonomy layer.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import tempfile
import threading
import time
import uuid
from contextlib import contextmanager, suppress
from datetime import datetime
from pathlib import Path
from typing import Any

from chronovisor.core.store import CHRONOVISOR_ROOT
from chronovisor.recall.recall_growth import run_growth_cycle

HISTORY_FILE = CHRONOVISOR_ROOT / "runtime" / "sleep-cycle-history.jsonl"
LOCK_FILE = CHRONOVISOR_ROOT / "runtime" / "sleep-cycle.lock"
ACTIVE_LANE_FILE = CHRONOVISOR_ROOT / "runtime" / "sleep-cycle-active-lane.json"
HISTORY_SCHEMA_VERSION = 1
HISTORY_MAX_LINES = 1000
DEFAULT_LANE_MAX_ELAPSED_SECONDS = 5 * 60
FINALIZATION_RESERVE_SECONDS = 30


class _LaneRuntimeBudgetExceeded(BaseException):
    """Interrupt a maintenance lane without being swallowed by broad catches."""


def _cycle_remaining_seconds() -> float | None:
    raw = os.environ.get("CHRONOVISOR_CYCLE_DEADLINE_MONOTONIC")
    if not raw:
        return None
    try:
        return float(raw) - time.monotonic()
    except ValueError:
        return None


def _lane_timeout_seconds(max_elapsed_seconds: float) -> float:
    requested = max(0.0, float(max_elapsed_seconds))
    remaining = _cycle_remaining_seconds()
    if remaining is None:
        return requested
    return max(
        0.0,
        min(requested, remaining - FINALIZATION_RESERVE_SECONDS),
    )


@contextmanager
def _lane_runtime_timer(seconds: float):
    """Raise at a hard lane boundary while preserving any caller timer."""

    if (
        seconds <= 0
        or threading.current_thread() is not threading.main_thread()
        or not hasattr(signal, "setitimer")
    ):
        yield
        return

    started = time.monotonic()
    prior_handler = signal.getsignal(signal.SIGALRM)
    prior_timer = signal.getitimer(signal.ITIMER_REAL)

    def interrupt(_signum, _frame) -> None:
        raise _LaneRuntimeBudgetExceeded

    signal.signal(signal.SIGALRM, interrupt)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, prior_handler)
        if prior_timer[0] > 0:
            elapsed = time.monotonic() - started
            signal.setitimer(
                signal.ITIMER_REAL,
                max(0.000001, prior_timer[0] - elapsed),
                prior_timer[1],
            )


def _write_active_lane(payload: dict[str, Any]) -> None:
    """Keep one bounded diagnostic receipt for the currently active lane."""

    if os.environ.get("CHRONOVISOR_READ_ONLY") == "1":
        return
    ACTIVE_LANE_FILE.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{ACTIVE_LANE_FILE.name}.",
        suffix=".tmp",
        dir=ACTIVE_LANE_FILE.parent,
    )
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, ACTIVE_LANE_FILE)
    finally:
        tmp.unlink(missing_ok=True)


def _compact_scalar(value: object, *, limit: int = 200) -> object:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return str(value)[:limit]


def _compact_numeric_map(value: object, *, limit: int = 32) -> dict[str, int | float]:
    if not isinstance(value, dict):
        return {}
    return {
        str(key)[:100]: number
        for key, number in sorted(value.items(), key=lambda pair: str(pair[0]))[:limit]
        if isinstance(number, (int, float)) and not isinstance(number, bool)
    }


def _sleep_history_summary(row: dict[str, Any]) -> dict[str, Any]:
    """Project a full cycle result into bounded, non-recursive history."""

    lane_statuses = {
        str(name)[:100]: str(value.get("status") or "unknown")[:100]
        for name, value in row.items()
        if isinstance(value, dict) and "status" in value
    }
    if not lane_statuses and isinstance(row.get("lane_statuses"), dict):
        lane_statuses = {
            str(name)[:100]: str(status)[:100]
            for name, status in sorted(
                row["lane_statuses"].items(), key=lambda pair: str(pair[0])
            )[:64]
            if isinstance(status, (str, int, float, bool)) or status is None
        }
    budget = row.get("convergence_budget")
    budget = budget if isinstance(budget, dict) else {}
    raw = row.get("raw_replay") if isinstance(row.get("raw_replay"), dict) else {}
    raw_drain = raw.get("drain") if isinstance(raw.get("drain"), dict) else {}

    work: dict[str, dict[str, Any]] = {}
    count_fields = {
        "lint_repair": (
            "processed",
            "applied",
            "routed",
            "rejected",
            "quarantined",
            "human_required",
            "deferred",
        ),
        "read_back_repair": (
            "processed",
            "applied",
            "already_present",
            "retry_scheduled",
            "quarantined",
            "human_required",
            "budget_deferred",
        ),
        "search_label_review": (
            "reviewed",
            "approved",
            "rejected",
            "retry",
            "quarantined",
            "human_required",
        ),
        "self_heal": ("packets_seen",),
        "duplicate_frontier": (
            "frontier_calls",
            "applied",
            "kept_both",
        ),
        "orphan_links": ("work_items", "orphans_seen", "orphans_total"),
    }
    for lane, fields in count_fields.items():
        value = row.get(lane)
        if not isinstance(value, dict):
            continue
        counts = {
            field: _compact_scalar(value.get(field))
            for field in fields
            if field in value
        }
        if counts:
            work[lane] = counts
    if not work and isinstance(row.get("work"), dict):
        existing_work = row["work"]
        for lane, fields in count_fields.items():
            counts = existing_work.get(lane)
            if not isinstance(counts, dict):
                continue
            filtered = {
                field: _compact_scalar(counts.get(field))
                for field in fields
                if field in counts
            }
            if filtered:
                work[lane] = filtered
        existing_raw = existing_work.get("raw_replay")
        if isinstance(existing_raw, dict):
            status_counts = existing_raw.get("status_counts")
            work["raw_replay"] = {
                "runs": _compact_scalar(existing_raw.get("runs")),
                "status_counts": _compact_numeric_map(status_counts),
            }
    if raw_drain:
        work["raw_replay"] = {
            "runs": _compact_scalar(raw_drain.get("count")),
            "status_counts": _compact_numeric_map(raw_drain.get("status_counts")),
        }

    lane_errors = row.get("lane_errors")
    return {
        "schema_version": HISTORY_SCHEMA_VERSION,
        "run_id": _compact_scalar(row.get("run_id")),
        "status": _compact_scalar(row.get("status")),
        "started_at": _compact_scalar(row.get("started_at")),
        "finished_at": _compact_scalar(row.get("finished_at")),
        "dry_run": bool(row.get("dry_run")),
        "lane_errors": (
            [str(error)[:200] for error in lane_errors[:32]]
            if isinstance(lane_errors, list)
            else []
        ),
        "lane_statuses": dict(sorted(lane_statuses.items())),
        "work": work,
        "convergence_budget": {
            "used": _compact_numeric_map(budget.get("used"), limit=16),
            "limits": _compact_numeric_map(budget.get("limits"), limit=16),
        },
    }


def _atomic_write_history(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(
                    json.dumps(row, ensure_ascii=False, sort_keys=True, default=str)
                    + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        with suppress(OSError):
            tmp.unlink(missing_ok=True)


def _append_history(row: dict[str, Any], *, max_lines: int = HISTORY_MAX_LINES) -> None:
    """Store compact history and normalize legacy recursive rows on every write."""

    max_lines = max(1, int(max_lines))
    try:
        lines = [
            line
            for line in HISTORY_FILE.read_text(encoding="utf-8").split("\n")
            if line.strip()
        ]
    except OSError:
        lines = []
    previous: list[dict[str, Any]] = []
    keep_previous = max_lines - 1
    for line in lines[-keep_previous:] if keep_previous else []:
        try:
            existing = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(existing, dict):
            previous.append(_sleep_history_summary(existing))
    rows = [*previous, _sleep_history_summary(row)][-max_lines:]
    _atomic_write_history(HISTORY_FILE, rows)


def _try_acquire_lock():
    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    handle = LOCK_FILE.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return None
    return handle


def _try_acquire_read_lock():
    """Join an existing lock without creating state during dry-run."""
    try:
        handle = LOCK_FILE.open("r", encoding="utf-8")
    except FileNotFoundError:
        return False
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return None
    return handle


def _run_lane(
    name: str,
    fn,
    *,
    max_elapsed_seconds: float = DEFAULT_LANE_MAX_ELAPSED_SECONDS,
) -> dict[str, Any]:
    """Isolate one maintenance lane so the remaining queues still drain."""
    started_at = datetime.now().isoformat(timespec="seconds")
    started = time.monotonic()
    timeout_seconds = _lane_timeout_seconds(max_elapsed_seconds)
    if timeout_seconds <= 0:
        return {
            "status": "budget_deferred",
            "lane": name,
            "reason": "sleep cycle runtime budget exhausted",
            "elapsed_ms": 0,
        }
    _write_active_lane(
        {
            "schema_version": 1,
            "lane": name,
            "status": "running",
            "started_at": started_at,
            "timeout_seconds": round(timeout_seconds, 3),
        }
    )
    try:
        with _lane_runtime_timer(timeout_seconds):
            result = fn()
    except _LaneRuntimeBudgetExceeded:
        result = {
            "status": "budget_deferred",
            "lane": name,
            "reason": "lane runtime budget exhausted",
        }
    except Exception as exc:
        result = {
            "status": "error",
            "lane": name,
            "error": f"{exc.__class__.__name__}: {exc}",
        }
    if not isinstance(result, dict):
        result = {
            "status": "error",
            "lane": name,
            "error": "lane returned a non-object result",
        }
    elapsed_ms = int((time.monotonic() - started) * 1000)
    result.setdefault("elapsed_ms", elapsed_ms)
    _write_active_lane(
        {
            "schema_version": 1,
            "lane": name,
            "status": str(result.get("status") or "unknown"),
            "started_at": started_at,
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "elapsed_ms": elapsed_ms,
            "timeout_seconds": round(timeout_seconds, 3),
        }
    )
    return result


def render_summary(payload: dict[str, Any]) -> str:
    """Render a compact status report that tolerates partial/skipped cycles."""

    def field(lane: str, *keys: str, default: object = "unavailable") -> object:
        value: object = payload.get(lane, {})
        for key in keys:
            if not isinstance(value, dict) or key not in value:
                return default
            value = value[key]
        return value

    lines = [f"sleep_cycle\t{payload.get('status', 'unknown')}"]
    if payload.get("locked"):
        lines.append(
            f"reason\t{payload.get('reason', 'sleep cycle already in progress')}"
        )
        return "\n".join(lines)
    lines.extend(
        [
            f"cofire_edges\t{field('cofire', 'edges')}",
            f"prefetch_buckets\t{field('prefetch', 'buckets')}",
            f"capture_rate\t{field('memory_integrity', 'capture_rate')}",
            f"retention_pages\t{field('retention', 'counts', 'pages')}",
            f"claim_index_claims\t{field('claims', 'claims')}",
            f"golden_added\t{field('golden', 'added')}",
            f"distill_rows\t{field('distill', 'rows')}",
            f"hubs\t{field('hubs', 'hubs')}",
            f"duplicates\t{field('duplicates', 'count')}",
            f"recall_improve\t{field('recall_improve', 'status')}",
            f"autonomy\t{field('autonomy', 'status')}",
            f"raw_archive\t{field('raw_archive', 'status')}",
            f"raw_segments_sealed\t{field('raw_archive', 'eligible')}",
        ]
    )
    if payload.get("lane_errors"):
        lines.append(
            f"lane_errors\t{','.join(str(item) for item in payload['lane_errors'])}"
        )
    return "\n".join(lines)


def run_sleep_cycle(
    *,
    raw_limit: int = 100,
    eval_limit: int = 100,
    duplicate_limit: int = 200,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run one cycle, enforcing process-wide read-only cache behavior in previews."""
    lock_handle = None
    if not os.environ.get("PYTEST_CURRENT_TEST"):
        lock_handle = _try_acquire_read_lock() if dry_run else _try_acquire_lock()
        if lock_handle is None:
            return {
                "status": "skipped",
                "reason": "sleep cycle already in progress",
                "locked": True,
                "lock_file": str(LOCK_FILE),
            }
    previous_read_only = os.environ.get("CHRONOVISOR_READ_ONLY")
    previous_deadline = os.environ.get("CHRONOVISOR_CYCLE_DEADLINE_MONOTONIC")
    deadline = time.monotonic() + 30 * 60
    if previous_deadline:
        with suppress(ValueError):
            deadline = min(deadline, float(previous_deadline))
    os.environ["CHRONOVISOR_CYCLE_DEADLINE_MONOTONIC"] = str(deadline)
    if dry_run:
        os.environ["CHRONOVISOR_READ_ONLY"] = "1"
    try:
        return _run_sleep_cycle(
            raw_limit=raw_limit,
            eval_limit=eval_limit,
            duplicate_limit=duplicate_limit,
            dry_run=dry_run,
        )
    finally:
        if dry_run:
            if previous_read_only is None:
                os.environ.pop("CHRONOVISOR_READ_ONLY", None)
            else:
                os.environ["CHRONOVISOR_READ_ONLY"] = previous_read_only
        if previous_deadline is None:
            os.environ.pop("CHRONOVISOR_CYCLE_DEADLINE_MONOTONIC", None)
        else:
            os.environ["CHRONOVISOR_CYCLE_DEADLINE_MONOTONIC"] = previous_deadline
        if lock_handle not in (None, False):
            try:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
            finally:
                lock_handle.close()


def _run_sleep_cycle(
    *,
    raw_limit: int = 100,
    eval_limit: int = 100,
    duplicate_limit: int = 200,
    dry_run: bool = False,
) -> dict[str, Any]:
    from chronovisor.ops.autonomy import run_autonomy_cycle
    from chronovisor.ops.convergence import ConvergenceStore, CycleBudget
    from chronovisor.ops.distill import export_distill_dataset
    from chronovisor.ops.golden_expand import expand_golden_from_recall_questions
    from chronovisor.ops.health import health_snapshot
    from chronovisor.ops.hubs import build_hub_pages
    from chronovisor.ops.memory_integrity import run_eval
    from chronovisor.ops.reflection import write_reflection_page
    from chronovisor.ops.retention import build_retention_scores
    from chronovisor.ops.snapshot import snapshot_chronovisor
    from chronovisor.ops.state_register import refresh_state_register
    from chronovisor.raw.raw_replay import (
        AUTO_SIGNAL_SOURCES,
        build_queue,
        run_pending_queue,
    )
    from chronovisor.recall import recall_improvement
    from chronovisor.recall.claims import rebuild_claim_index
    from chronovisor.recall.duplicate_review import (
        build_duplicate_review_queue,
        write_review_queue,
    )
    from chronovisor.search.cofire import build_cofire_graph
    from chronovisor.search.prefetch import build_prefetch_cache

    try:
        per_lane_frontier = max(
            1, int(os.getenv("CHRONOVISOR_FRONTIER_CALLS_PER_LANE", "3"))
        )
    except ValueError:
        per_lane_frontier = 3
    cycle_budget = CycleBudget(
        max_local_calls=30,
        max_frontier_calls=max(24, per_lane_frontier * 8),
        max_mutations=60,
        max_raw_bytes=2_000_000,
        max_elapsed_seconds=30 * 60,
    )
    artifact_budget = cycle_budget.slice(max_mutations=16)

    def artifact_lane(name: str, fn, *, mutates: bool = True) -> dict[str, Any]:
        if dry_run or not mutates:
            allowed, reason = cycle_budget.can_consume("mutation", 0)
        else:
            allowed, reason = artifact_budget.consume("mutation")
        if not allowed:
            return {"status": "budget_deferred", "lane": name, "reason": reason}
        return _run_lane(name, fn)

    started = datetime.now().isoformat(timespec="seconds")
    run_id = uuid.uuid4().hex
    before_health_result = artifact_lane(
        "health_before", health_snapshot, mutates=False
    )
    before_health = (
        before_health_result if before_health_result.get("status") != "error" else {}
    )
    snapshot = (
        {"status": "skipped", "reason": "dry_run"}
        if dry_run
        else artifact_lane(
            "snapshot", lambda: snapshot_chronovisor("before sleep cycle")
        )
    )
    try:
        quarantine_cooldown = max(
            0,
            int(os.getenv("CHRONOVISOR_CONVERGENCE_QUARANTINE_RETRY_SECONDS", "21600")),
        )
    except ValueError:
        quarantine_cooldown = 21_600
    # This global pass reopens operational outages only. ConvergenceStore
    # deliberately leaves semantic no-quorum holds to each owning lane, which
    # can prove an exact evidence/authority epoch change before resampling.
    convergence_quarantine_recovery = _run_lane(
        "convergence_quarantine_recovery",
        lambda: ConvergenceStore().resume_due_quarantined(
            cooldown_seconds=quarantine_cooldown,
            exclude_lanes={"content_correction"},
            dry_run=dry_run,
        ),
    )
    convergence_lease_recovery = _run_lane(
        "convergence_lease_recovery",
        lambda: ConvergenceStore().reap_expired_leases(dry_run=dry_run),
    )
    # Routine sleep is a local-only data-plane job.  Frontier capability is
    # checked only when an explicit system-code incident acquires the durable
    # repair guard; polling Codex here creates process noise and can revive
    # human-boundary queues on every cycle.
    frontier_capability_preflight = {
        "ok": False,
        "status": "disabled",
        "reason": "repair_plane_only",
    }
    convergence_human_recovery = {
        "status": "skipped",
        "reason": "human_boundary_requires_explicit_recheck",
    }
    external_queue_recovery = {
        "status": "skipped",
        "reason": "repair_plane_only",
    }
    correction_budget = cycle_budget.slice(
        # One authoritative classification plus one byte-level mutation review.
        max_local_calls=6,
        max_frontier_calls=6,
        max_mutations=3,
    )
    content_corrections = _run_lane(
        "content_corrections",
        lambda: __import__(
            "chronovisor.recall.content_correction",
            fromlist=["run_pending_corrections"],
        ).run_pending_corrections(
            max_items=6,
            budget=correction_budget,
            dry_run=dry_run,
        ),
    )
    cofire = artifact_lane("cofire", lambda: build_cofire_graph(write=not dry_run))
    recall_growth = artifact_lane(
        "recall_growth", lambda: run_growth_cycle(dry_run=dry_run)
    )
    prefetch = artifact_lane(
        "prefetch", lambda: build_prefetch_cache(write=not dry_run)
    )
    retention = artifact_lane(
        "retention", lambda: build_retention_scores(write=not dry_run)
    )
    claims = artifact_lane("claims", lambda: rebuild_claim_index(write=not dry_run))
    claim_conflicts = _run_lane(
        "claim_conflicts",
        lambda: __import__(
            "chronovisor.recall.claims", fromlist=["review_claim_conflicts"]
        ).review_claim_conflicts(
            limit=per_lane_frontier,
            write=not dry_run,
        ),
    )
    golden = artifact_lane(
        "golden",
        lambda: expand_golden_from_recall_questions(limit=0, write=not dry_run),
    )
    distill = artifact_lane(
        "distill", lambda: export_distill_dataset(write=not dry_run)
    )
    hubs = artifact_lane("hubs", lambda: build_hub_pages(write=not dry_run))
    reflection = artifact_lane(
        "reflection", lambda: write_reflection_page(write=not dry_run)
    )
    state_register = artifact_lane(
        "state_register", lambda: refresh_state_register(write=not dry_run)
    )
    page_normalization = artifact_lane(
        "page_normalization",
        lambda: __import__(
            "chronovisor.ops.page_normalize", fromlist=["normalize_pages"]
        ).normalize_pages(
            write=not dry_run,
            limit=100,
            max_frontier_calls=per_lane_frontier,
        ),
    )
    metadata_backfill = _run_lane(
        "metadata_backfill",
        lambda: __import__(
            "chronovisor.ops.metadata_backfill", fromlist=["backfill_metadata"]
        ).backfill_metadata(
            limit=per_lane_frontier,
            max_frontier_calls=per_lane_frontier,
            dry_run=dry_run,
        ),
    )
    entity_backfill = _run_lane(
        "entity_backfill",
        lambda: __import__(
            "chronovisor.ops.entities", fromlist=["backfill_entities"]
        ).backfill_entities(
            limit=per_lane_frontier,
            max_frontier_calls=per_lane_frontier,
            dry_run=dry_run,
        ),
    )
    librarian_shadow = _run_lane(
        "librarian_shadow",
        lambda: __import__(
            "chronovisor.librarian.librarian", fromlist=["run_shadow"]
        ).run_shadow(
            root=CHRONOVISOR_ROOT,
            limit=100,
            full_sweep=False,
            dry_run=dry_run,
        ),
        max_elapsed_seconds=2 * 60,
    )
    librarian_cleanup = (
        {"status": "skipped", "reason": "dry_run"}
        if dry_run
        else _run_lane(
            "librarian_cleanup",
            lambda: {
                "status": "ok",
                "release": __import__(
                    "chronovisor.librarian.librarian_release",
                    fromlist=["finalize_if_ready"],
                ).finalize_if_ready(CHRONOVISOR_ROOT),
                "restore_points": __import__(
                    "chronovisor.ops.migration_snapshot",
                    fromlist=["cleanup_expired_restore_points"],
                ).cleanup_expired_restore_points(CHRONOVISOR_ROOT),
                "transaction_preimages": __import__(
                    "chronovisor.librarian.merge_transaction",
                    fromlist=["cleanup_expired_preimages"],
                ).cleanup_expired_preimages(CHRONOVISOR_ROOT),
            },
        )
    )
    integrity = (
        artifact_lane(
            "memory_integrity", lambda: run_eval(limit=eval_limit, write=not dry_run)
        )
        if eval_limit > 0
        else {"status": "skipped", "reason": "eval_limit_zero"}
    )
    lane_budgets = {
        "lint": cycle_budget.slice(
            max_local_calls=10, max_frontier_calls=per_lane_frontier, max_mutations=6
        ),
        "read_back_repair": cycle_budget.slice(
            max_frontier_calls=per_lane_frontier, max_mutations=per_lane_frontier
        ),
        "labels": cycle_budget.slice(
            max_frontier_calls=per_lane_frontier, max_mutations=per_lane_frontier
        ),
        "recall_auto_apply": cycle_budget.slice(
            max_frontier_calls=per_lane_frontier, max_mutations=per_lane_frontier
        ),
        "self_heal": cycle_budget.slice(
            max_local_calls=per_lane_frontier,
            max_frontier_calls=per_lane_frontier,
            max_mutations=per_lane_frontier,
        ),
        "recall_improve": cycle_budget.slice(
            max_frontier_calls=per_lane_frontier, max_mutations=per_lane_frontier
        ),
        "calibration": cycle_budget.slice(
            max_frontier_calls=per_lane_frontier, max_mutations=per_lane_frontier
        ),
        "self_tune": cycle_budget.slice(
            max_frontier_calls=per_lane_frontier, max_mutations=per_lane_frontier
        ),
        "duplicates": cycle_budget.slice(
            max_frontier_calls=per_lane_frontier, max_mutations=per_lane_frontier
        ),
        "orphans": cycle_budget.slice(
            max_local_calls=8,
            max_frontier_calls=per_lane_frontier,
            max_mutations=per_lane_frontier,
        ),
        "raw": cycle_budget.slice(
            max_frontier_calls=per_lane_frontier,
            max_mutations=per_lane_frontier,
            max_raw_bytes=2_000_000,
        ),
        "autonomy_duplicates": cycle_budget.slice(max_mutations=1),
        "autonomy_retention": cycle_budget.slice(
            max_frontier_calls=per_lane_frontier, max_mutations=per_lane_frontier
        ),
    }
    if dry_run:
        lint_due = _run_lane(
            "lint_due",
            lambda: __import__(
                "chronovisor.ingest.orchestrator", fromlist=["run_lint_if_due"]
            ).run_lint_if_due(dry_run=True),
        )
    else:
        lint_allowed, lint_reason = lane_budgets["lint"].consume("mutation")
        lint_due = (
            _run_lane(
                "lint_due",
                lambda: __import__(
                    "chronovisor.ingest.orchestrator", fromlist=["run_lint_if_due"]
                ).run_lint_if_due(dry_run=False),
            )
            if lint_allowed
            else {
                "status": "budget_deferred",
                "lane": "lint_due",
                "reason": lint_reason,
            }
        )
    lint_repair = _run_lane(
        "lint_repair",
        lambda: __import__(
            "chronovisor.ops.lint_repair", fromlist=["run_lint_repair"]
        ).run_lint_repair(
            max_items=5,
            budget=lane_budgets["lint"],
            dry_run=dry_run,
        ),
    )
    read_back_repair = _run_lane(
        "read_back_repair",
        lambda: __import__(
            "chronovisor.ingest.read_back_repair", fromlist=["run_read_back_repair"]
        ).run_read_back_repair(
            max_items=5,
            budget=lane_budgets["read_back_repair"],
            dry_run=dry_run,
        ),
    )
    raw_queue = artifact_lane(
        "raw_replay_queue",
        lambda: build_queue(
            limit=max(0, raw_limit),
            include_migration=False,
            include_auto_signals=True,
            dry_run=dry_run,
        ),
    )
    raw_drain = _run_lane(
        "raw_replay_drain",
        lambda: run_pending_queue(
            max_runs=1 if raw_limit > 0 else 0,
            max_bytes=2_000_000,
            dry_run=dry_run,
            eligible_keys={str(key) for key in raw_queue.get("candidate_keys", [])}
            if raw_queue.get("status") != "error"
            else set(),
            eligible_sources=(
                AUTO_SIGNAL_SOURCES
                if raw_queue.get("status") != "error"
                else frozenset()
            ),
            budget=lane_budgets["raw"],
        ),
    )
    raw_replay = {"status": "ok", "queue_refresh": raw_queue, "drain": raw_drain}
    if any(item.get("status") == "error" for item in (raw_queue, raw_drain)):
        raw_replay["status"] = "error"
    raw_archive = _run_lane(
        "raw_archive",
        lambda: __import__(
            "chronovisor.raw.raw_archive", fromlist=["seal_eligible"]
        ).seal_eligible(
            CHRONOVISOR_ROOT / "raw",
            dry_run=dry_run,
            max_segments=4,
        ),
    )

    search_labels = (
        _run_lane(
            "search_label_queue",
            lambda: __import__(
                "chronovisor.search.search_eval", fromlist=["build_label_queue"]
            ).build_label_queue(
                limit=eval_limit,
                dry_run=dry_run,
                budget=lane_budgets["labels"],
            ),
        )
        if eval_limit > 0
        else {"status": "skipped", "reason": "eval_limit_zero"}
    )
    search_label_review = (
        _run_lane(
            "search_label_review",
            lambda: __import__(
                "chronovisor.search.search_eval",
                fromlist=["review_label_queue_with_frontier"],
            ).review_label_queue_with_frontier(
                limit=min(2, eval_limit),
                max_attempts=3,
                dry_run=dry_run,
                budget=lane_budgets["labels"],
            ),
        )
        if eval_limit > 0
        else {"status": "skipped", "reason": "eval_limit_zero"}
    )
    recall_auto_apply = _run_lane(
        "recall_auto_apply",
        lambda: __import__(
            "chronovisor.recall.recall_auto_apply", fromlist=["apply_feedback_file"]
        ).apply_feedback_file(
            dry_run=dry_run,
            budget=lane_budgets["recall_auto_apply"],
        ),
    )
    self_heal = _run_lane(
        "self_heal",
        lambda: __import__(
            "chronovisor.ops.self_heal", fromlist=["run_pending"]
        ).run_pending(
            max_packets=1,
            enable_frontier=False,
            dry_run=dry_run,
            frontier_budget=lane_budgets["self_heal"],
        ),
        max_elapsed_seconds=2 * 60,
    )
    duplicate_build = _run_lane(
        "duplicates",
        lambda: {
            "status": "ok",
            "records": build_duplicate_review_queue(limit=max(0, duplicate_limit)),
        },
        max_elapsed_seconds=60,
    )
    duplicates = (
        duplicate_build.get("records", [])
        if duplicate_build.get("status") != "error"
        and isinstance(duplicate_build.get("records"), list)
        else []
    )
    duplicate_path = ""
    duplicate_status = str(duplicate_build.get("status") or "ok")
    duplicate_error = duplicate_build.get("error")
    if not dry_run and duplicate_status != "error":
        duplicate_write = artifact_lane(
            "duplicates",
            lambda: {"status": "ok", "path": str(write_review_queue(duplicates))},
        )
        duplicate_status = str(duplicate_write.get("status") or "ok")
        duplicate_path = str(duplicate_write.get("path") or "")
        duplicate_error = duplicate_write.get("error")
    recall_improve = _run_lane(
        "recall_improve",
        lambda: recall_improvement.run_due(
            apply=not dry_run,
            min_interval_hours=24.0,
            min_new_feedback=5,
            min_total_feedback=3,
            max_examples=40,
            max_elapsed_seconds=15 * 60,
            frontier_mode="auto",
            frontier_budget=lane_budgets["recall_improve"],
            dry_run=dry_run,
        ),
        max_elapsed_seconds=15 * 60,
    )
    model_lab = _run_lane(
        "model_lab",
        lambda: __import__("chronovisor.lab.model_lab", fromlist=["run_due"]).run_due(
            dry_run=dry_run,
            max_evaluations=2,
        ),
    )
    calibration = _run_lane(
        "recall_calibration",
        lambda: __import__(
            "chronovisor.recall.recall_calibration", fromlist=["run_due"]
        ).run_due(
            min_interval_hours=7 * 24,
            max_samples=2000,
            max_recomputed_features=50,
            dry_run=dry_run,
            frontier_mode="auto",
            budget=lane_budgets["calibration"],
        ),
    )
    search_self_tune = _run_lane(
        "search_self_tune",
        lambda: __import__(
            "chronovisor.search.search_eval", fromlist=["run_self_tune_due"]
        ).run_self_tune_due(
            min_interval_hours=7 * 24,
            apply=True,
            dry_run=dry_run,
            frontier_mode="auto",
            budget=lane_budgets["self_tune"],
            max_examples=40,
            max_elapsed_seconds=120,
        ),
        max_elapsed_seconds=125,
    )
    research_consolidation = _run_lane(
        "research_consolidation",
        lambda: __import__(
            "chronovisor.research.research_consolidation",
            fromlist=["run_consolidation"],
        ).run_consolidation(dry_run=dry_run),
    )
    payload = {
        "status": "ok",
        "run_id": run_id,
        "started_at": started,
        "dry_run": dry_run,
        "snapshot": snapshot,
        "convergence_quarantine_recovery": convergence_quarantine_recovery,
        "convergence_lease_recovery": convergence_lease_recovery,
        "frontier_capability_preflight": frontier_capability_preflight,
        "convergence_human_recovery": convergence_human_recovery,
        "external_queue_recovery": external_queue_recovery,
        "content_corrections": content_corrections,
        "health_before": before_health_result,
        "cofire": {k: v for k, v in cofire.items() if k != "graph"},
        "recall_growth": recall_growth,
        "prefetch": {
            "status": prefetch.get("status"),
            "episodes": prefetch.get("episodes", 0),
            "buckets": len(prefetch.get("buckets", {})),
            "tokens": len(prefetch.get("tokens", {})),
        },
        "memory_integrity": {k: v for k, v in integrity.items() if k != "rows"},
        "retention": {k: v for k, v in retention.items() if k != "pages"},
        "claims": claims,
        "claim_conflicts": claim_conflicts,
        "golden": golden,
        "distill": distill,
        "hubs": {k: v for k, v in hubs.items() if k != "paths"},
        "reflection": reflection,
        "state_register": state_register,
        "page_normalization": page_normalization,
        "metadata_backfill": metadata_backfill,
        "entity_backfill": entity_backfill,
        "librarian_shadow": librarian_shadow,
        "librarian_cleanup": librarian_cleanup,
        "raw_replay": raw_replay,
        "raw_archive": raw_archive,
        "lint_due": lint_due,
        "lint_repair": lint_repair,
        "search_labels": search_labels,
        "search_label_review": search_label_review,
        "recall_auto_apply": recall_auto_apply,
        "read_back_repair": read_back_repair,
        "self_heal": self_heal,
        "duplicates": {
            "status": duplicate_status,
            "count": len(duplicates),
            "path": duplicate_path,
            **({"error": duplicate_error} if duplicate_error else {}),
        },
        "recall_improve": recall_improve,
        "model_lab": model_lab,
        "recall_calibration": calibration,
        "search_self_tune": search_self_tune,
        "research_consolidation": research_consolidation,
    }
    payload["autonomy"] = _run_lane(
        "autonomy",
        lambda: run_autonomy_cycle(
            duplicates=duplicates,
            retention=retention,
            before_health=before_health,
            snapshot=snapshot,
            dry_run=dry_run,
            budget=lane_budgets["autonomy_duplicates"],
            retention_budget=lane_budgets["autonomy_retention"],
        ),
    )
    payload["duplicate_frontier"] = _run_lane(
        "duplicate_frontier",
        lambda: __import__(
            "chronovisor.ops.autonomy",
            fromlist=["resolve_deferred_duplicates_with_frontier"],
        ).resolve_deferred_duplicates_with_frontier(
            duplicates,
            budget=lane_budgets["duplicates"],
            dry_run=dry_run,
            inventory_complete=len(duplicates) < duplicate_limit,
        ),
    )
    payload["orphan_links"] = _run_lane(
        "orphan_links",
        lambda: __import__(
            "chronovisor.ops.orphan_link", fromlist=["run_autonomous"]
        ).run_autonomous(
            orphan_limit=2,
            max_candidates=2,
            budget=lane_budgets["orphans"],
            dry_run=dry_run,
        ),
    )
    payload["convergence_budget"] = cycle_budget.snapshot()
    if not dry_run:
        payload["snapshot_after"] = artifact_lane(
            "snapshot_after", lambda: snapshot_chronovisor("after sleep cycle")
        )
    lane_errors = [
        name
        for name, value in payload.items()
        if isinstance(value, dict) and value.get("status") == "error"
    ]
    if lane_errors:
        payload["status"] = "partial"
        payload["lane_errors"] = lane_errors
    payload["finished_at"] = datetime.now().isoformat(timespec="seconds")
    if not dry_run:
        # The history row is the durable execution receipt consumed by the
        # watchdog.  It is bounded operational state, not a semantic/page
        # mutation, so it must not compete with artifact work for the last
        # mutation token.  Otherwise a completely successful cycle can spend
        # its artifact allowance and be reported forever as
        # ``sleep_never_ran``.
        payload["convergence_budget"] = cycle_budget.snapshot()
        try:
            _append_history(payload)
            payload["history"] = {"status": "ok"}
        except OSError as exc:
            payload["history"] = {
                "status": "error",
                "lane": "sleep_history",
                "error": f"{exc.__class__.__name__}: {exc}",
            }
            payload["status"] = "partial"
            lane_errors = payload.setdefault("lane_errors", [])
            if "sleep_history" not in lane_errors:
                lane_errors.append("sleep_history")
    else:
        payload["convergence_budget"] = cycle_budget.snapshot()
    return payload


def main(argv: list[str] | None = None) -> int:
    """Run the ``chronovisor-sleep`` command-line entry point."""
    parser = argparse.ArgumentParser(description="Run Chronovisor sleep consolidation.")
    parser.add_argument("--raw-limit", type=int, default=100)
    parser.add_argument("--eval-limit", type=int, default=100)
    parser.add_argument("--duplicate-limit", type=int, default=200)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    payload = run_sleep_cycle(
        raw_limit=max(0, args.raw_limit),
        eval_limit=max(0, args.eval_limit),
        duplicate_limit=max(0, args.duplicate_limit),
        dry_run=args.dry_run,
    )
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    else:
        print(render_summary(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
