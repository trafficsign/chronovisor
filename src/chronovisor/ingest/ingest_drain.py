"""Drain pending raw Chronovisor entries through the ingest orchestrator."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from chronovisor.core import ollama, runtime_status
from chronovisor.core.link_fix import atomic_write
from chronovisor.core.store import (
    CHRONOVISOR_ROOT,
    init_chronovisor,
    okf_runtime_operation,
    okf_startup_status,
)
from chronovisor.ingest import orchestrator

DEFAULT_MAX_BATCHES = 24
DEFAULT_MAX_UNITS = orchestrator.MAX_INGEST_BATCH_UNITS
DEFAULT_SLEEP_SECONDS = 1.0
DEFAULT_IDLE_SLEEP_SECONDS = 60.0


def _liveness_file() -> Path:
    return CHRONOVISOR_ROOT / "runtime" / "ingest-liveness.json"


def _read_liveness() -> dict[str, Any]:
    try:
        payload = json.loads(_liveness_file().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _record_liveness(
    status: str,
    *,
    pending: int,
    error: str = "",
    authority_preflight: dict[str, Any] | None = None,
) -> dict[str, Any]:
    previous = _read_liveness()
    now = datetime.now().isoformat(timespec="seconds")
    previous_status = str(previous.get("status") or "")
    waiting = status == "waiting_for_ingest_runtime"
    authority_blocked = status == "blocked_by_decision_authority"
    blocked = waiting or authority_blocked
    previous_blocked = previous_status in {
        "waiting_for_ingest_runtime",
        "waiting_for_ollama",
        "blocked_by_decision_authority",
    }
    payload = {
        "schema_version": 2,
        "status": status,
        "observed_at": now,
        "pending_raws": max(0, int(pending)),
        "ingest_runtime_available": not waiting,
        "authority_available": not authority_blocked,
        "alert": authority_blocked or (waiting and pending > 0),
        "retryable": blocked,
        "consecutive_unavailable_checks": (
            int(previous.get("consecutive_unavailable_checks") or 0) + 1
            if blocked
            else 0
        ),
        "unavailable_since": (
            str(previous.get("unavailable_since") or now) if blocked else None
        ),
        "last_recovered_at": (
            now
            if not blocked and previous_blocked
            else previous.get("last_recovered_at")
        ),
        "error": error,
        "authority_preflight": authority_preflight,
        "transitioned": previous_status != status,
    }
    atomic_write(
        _liveness_file(),
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    )
    return payload


def _publish_authority_ready_status(
    authority_preflight: dict[str, Any], *, state: str, stage: str, pending: int
) -> None:
    """Clear a prior authority block only after an explicit successful proof."""

    if authority_preflight.get("ok") is not True:
        return
    runtime_status.safe_write_status(
        state=state,
        stage=stage,
        pending=max(0, int(pending)),
        current_raw=None,
        current_op=None,
        current_job_id=None,
        current_job_pid=None,
        batch=None,
        ollama=None,
        llm=None,
        authority_preflight=authority_preflight,
        mutation_authority=authority_preflight,
        mutation_ready=True,
    )


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _default_log_file() -> Path:
    stamp = datetime.now().strftime("%Y%m%d")
    return CHRONOVISOR_ROOT / "logs" / f"ingest-drain-{stamp}.jsonl"


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def _release_ingest_runner() -> dict[str, Any]:
    """Release the heavy ingest runner after a complete drain cycle."""

    try:
        route = ollama.runtime_generation_routes(
            (ollama.INGEST_GENERATION_RUNTIME_ROLE,)
        )[0]
        if route.provider != "ollama" or route.location != "local":
            return {"status": "not_applicable", "released": False}
        model = route.model
        resident = ollama.resident_model_rows()
    except Exception as exc:
        result = {
            "status": "probe_failed",
            "released": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
        runtime_status.safe_append_event(
            "warn",
            "ingest drain | runner release probe failed",
            source="ingest-drain",
            **result,
        )
        return result

    if model not in resident:
        return {
            "status": "not_resident",
            "released": False,
            "model": model,
        }

    released = ollama.unload_named_model(model)
    result = {
        "status": "released" if released else "release_failed",
        "released": released,
        "model": model,
    }
    runtime_status.safe_append_event(
        "info" if released else "warn",
        (
            "ingest drain | runner released"
            if released
            else "ingest drain | runner release failed"
        ),
        source="ingest-drain",
        **result,
    )
    return result


def _reconcile_processed_projections() -> dict[str, Any]:
    try:
        return orchestrator.reconcile_processed_projections(
            max_parents=orchestrator.PROCESSED_PROJECTION_RECONCILER_MAX_PARENTS
        )
    except Exception as exc:
        return {
            "status": "error",
            "disabled": False,
            "processed": [],
            "held": [],
            "error": f"{type(exc).__name__}: {exc}",
        }


def _drain(
    *,
    max_batches: int = DEFAULT_MAX_BATCHES,
    max_units: int = DEFAULT_MAX_UNITS,
    sleep_seconds: float = DEFAULT_SLEEP_SECONDS,
    log_file: Path | None = None,
) -> dict[str, Any]:
    """Run forced ingest batches until the queue drains or progress stops."""
    if max_batches < 0:
        raise ValueError("max_batches must be >= 0")
    if not isinstance(max_units, int) or isinstance(max_units, bool):
        raise ValueError("max_units must be an integer between 1 and 10")
    if not 1 <= max_units <= orchestrator.MAX_INGEST_BATCH_UNITS:
        raise ValueError(
            f"max_units must be between 1 and {orchestrator.MAX_INGEST_BATCH_UNITS}"
        )
    if sleep_seconds < 0:
        raise ValueError("sleep_seconds must be >= 0")

    init_chronovisor()
    orchestrator.reset_stale_lock()
    runtime_status.reset_stale_runtime_status()
    try:
        from chronovisor.ingest.managed_hold_sync import sync_ingest_semantic_holds

        managed_holds = sync_ingest_semantic_holds(chronovisor_root=CHRONOVISOR_ROOT)
    except Exception as exc:
        managed_holds = {"status": "error", "error": str(exc)}

    started = time.time()
    log_path = log_file or _default_log_file()
    pending_start = len(orchestrator.get_pending_raw_files())
    batches: list[dict[str, Any]] = []
    total_processed = 0
    stop_reason = "max_batches reached"

    if max_batches == 0:
        return {
            "status": "checked",
            "pending_start": pending_start,
            "pending_after": pending_start,
            "batches_run": 0,
            "files_processed": 0,
            "elapsed_seconds": 0.0,
            "managed_holds": managed_holds,
        }

    try:
        ingest_route = ollama.runtime_generation_routes(
            (ollama.INGEST_GENERATION_RUNTIME_ROLE,)
        )[0]
    except ollama.RuntimeBridgeError:
        ingest_route = None
    local_ollama = (
        ingest_route is not None
        and ingest_route.provider == "ollama"
        and ingest_route.location == "local"
    )
    runtime_available = ingest_route is not None and (
        not local_ollama or ollama.is_available()
    )
    if not runtime_available and pending_start > 0:
        projection_reconcile = _reconcile_processed_projections()
        pending_after_reconcile = len(orchestrator.get_pending_raw_files())
        liveness = _record_liveness(
            "waiting_for_ingest_runtime",
            pending=pending_after_reconcile,
            error=(
                "ingest runtime unavailable; raw capture remains durable and "
                "drain will retry"
            ),
        )
        if liveness["transitioned"]:
            _append_jsonl(
                log_path,
                {
                    "timestamp": datetime.now().isoformat(),
                    "kind": "ingest_liveness",
                    "status": liveness["status"],
                    "pending_before": pending_start,
                    "pending_after": pending_after_reconcile,
                    "alert": True,
                    "retryable": True,
                    "error": liveness["error"],
                },
            )
        return {
            "status": "waiting_for_ingest_runtime",
            "pending_start": pending_start,
            "pending_after": pending_after_reconcile,
            "batches_run": 0,
            "files_processed": 0,
            "stop_reason": "ingest runtime unavailable",
            "elapsed_seconds": round(time.time() - started, 2),
            "log_file": str(log_path),
            "alert": True,
            "retryable": True,
            "liveness": liveness,
            "managed_holds": managed_holds,
            "projection_reconcile": projection_reconcile,
        }
    authority_preflight = (
        orchestrator.ingest_authority_preflight() if runtime_available else None
    )
    if authority_preflight is not None and not authority_preflight["ok"]:
        projection_reconcile = _reconcile_processed_projections()
        pending_after_reconcile = len(orchestrator.get_pending_raw_files())
        liveness = _record_liveness(
            "blocked_by_decision_authority",
            pending=pending_after_reconcile,
            error=str(authority_preflight["error"]),
            authority_preflight=authority_preflight,
        )
        if liveness["transitioned"]:
            _append_jsonl(
                log_path,
                {
                    "timestamp": datetime.now().isoformat(),
                    "kind": "ingest_liveness",
                    "status": liveness["status"],
                    "pending_before": pending_start,
                    "pending_after": pending_after_reconcile,
                    "alert": True,
                    "retryable": True,
                    "error": liveness["error"],
                    "authority_preflight": authority_preflight,
                },
            )
        return {
            "status": "blocked",
            "pending_start": pending_start,
            "pending_after": pending_after_reconcile,
            "batches_run": 0,
            "files_processed": 0,
            "stop_reason": str(authority_preflight["error"]),
            "elapsed_seconds": round(time.time() - started, 2),
            "log_file": str(log_path),
            "alert": True,
            "retryable": True,
            "blocked_by": "decision_authority",
            "authority_preflight": authority_preflight,
            "liveness": liveness,
            "managed_holds": managed_holds,
            "projection_reconcile": projection_reconcile,
        }
    if isinstance(authority_preflight, dict) and authority_preflight.get("ok") is True:
        _publish_authority_ready_status(
            authority_preflight,
            state="idle" if pending_start == 0 else "ready",
            stage="idle" if pending_start == 0 else "waiting",
            pending=pending_start,
        )
    if pending_start == 0:
        liveness = _record_liveness("idle", pending=0)
    else:
        liveness = _record_liveness("ready", pending=pending_start)

    batch_authority_block: dict[str, Any] | None = None
    for batch_index in range(1, max_batches + 1):
        pending_before = len(orchestrator.get_pending_raw_files())
        if pending_before == 0:
            stop_reason = "no pending raws"
            break

        result = orchestrator.run_pending_ingest(
            force=True,
            max_units=max_units,
        )
        processed = len(result.get("files_processed", []))
        total_processed += processed
        try:
            from chronovisor.ingest.self_heal import (
                run_pending as run_pending_self_heal,
            )

            self_heal_result = run_pending_self_heal(
                max_packets=1,
                enable_frontier=False,
            )
        except Exception as exc:
            self_heal_result = {"status": "error", "error": str(exc)}
        pending_after = len(orchestrator.get_pending_raw_files())
        record = {
            "timestamp": datetime.now().isoformat(),
            "batch": batch_index,
            "pending_before": pending_before,
            "pending_after": pending_after,
            "files_processed": processed,
            "result": result,
            "self_heal": self_heal_result,
        }
        try:
            managed_holds = sync_ingest_semantic_holds(chronovisor_root=CHRONOVISOR_ROOT)
        except Exception as exc:
            managed_holds = {"status": "error", "error": str(exc)}
        record["managed_holds"] = managed_holds
        batches.append(record)
        _append_jsonl(log_path, record)

        if result.get("blocked_by") == "decision_authority":
            batch_authority_block = (
                result.get("authority_preflight")
                if isinstance(result.get("authority_preflight"), dict)
                else {
                    "ok": False,
                    "status": "blocked",
                    "blocked_by": "decision_authority",
                    "retryable": True,
                    "error": result.get("reason"),
                    "artifact_sha256": None,
                }
            )
            stop_reason = str(
                batch_authority_block.get("error")
                or "local consensus authority unavailable"
            )
            break
        if not result.get("triggered"):
            stop_reason = result.get("reason", "ingest did not trigger")
            break
        if processed == 0 and pending_after >= pending_before:
            stop_reason = "no batch progress"
            break
        if pending_after == 0:
            stop_reason = "drained"
            break
        if sleep_seconds:
            time.sleep(sleep_seconds)

    projection_reconcile = _reconcile_processed_projections()
    pending_final = len(orchestrator.get_pending_raw_files())
    if batch_authority_block is not None:
        status = "blocked"
    elif pending_final == 0:
        status = "drained"
    elif not batches and pending_start == 0 and pending_final == 0:
        status = "idle"
    elif stop_reason == "no batch progress":
        status = "stalled"
    elif batches and not batches[-1]["result"].get("triggered"):
        status = "blocked"
    else:
        status = "partial"

    if batch_authority_block is not None:
        liveness = _record_liveness(
            "blocked_by_decision_authority",
            pending=pending_final,
            error=str(batch_authority_block.get("error") or stop_reason),
            authority_preflight=batch_authority_block,
        )
    else:
        if isinstance(authority_preflight, dict) and authority_preflight.get("ok") is True:
            _publish_authority_ready_status(
                authority_preflight,
                state="idle" if pending_final == 0 else "ready",
                stage="idle" if pending_final == 0 else "waiting",
                pending=pending_final,
            )
        liveness = _record_liveness(
            "idle" if pending_final == 0 else "ready",
            pending=pending_final,
        )
    return {
        "status": status,
        "pending_start": pending_start,
        "pending_after": pending_final,
        "batches_run": len(batches),
        "files_processed": total_processed,
        "stop_reason": stop_reason,
        "elapsed_seconds": round(time.time() - started, 2),
        "log_file": str(log_path),
        "managed_holds": managed_holds,
        "liveness": liveness,
        "alert": batch_authority_block is not None,
        "retryable": batch_authority_block is not None,
        "blocked_by": (
            "decision_authority" if batch_authority_block is not None else None
        ),
        "authority_preflight": batch_authority_block,
        "projection_reconcile": projection_reconcile,
    }


def drain(
    *,
    max_batches: int = DEFAULT_MAX_BATCHES,
    max_units: int = DEFAULT_MAX_UNITS,
    sleep_seconds: float = DEFAULT_SLEEP_SECONDS,
    log_file: Path | None = None,
) -> dict[str, Any]:
    """Drain pending raws and release the ingest runner at cycle end."""

    with okf_runtime_operation(CHRONOVISOR_ROOT):
        return _drain_locked(
            max_batches=max_batches,
            max_units=max_units,
            sleep_seconds=sleep_seconds,
            log_file=log_file,
        )


def _drain_locked(
    *,
    max_batches: int,
    max_units: int,
    sleep_seconds: float,
    log_file: Path | None,
) -> dict[str, Any]:

    if not okf_startup_status(CHRONOVISOR_ROOT).allowed:
        return {"status": "blocked", "category": "okf_startup_blocked"}

    result: dict[str, Any] | None = None
    try:
        result = _drain(
            max_batches=max_batches,
            max_units=max_units,
            sleep_seconds=sleep_seconds,
            log_file=log_file,
        )
        return result
    finally:
        release = _release_ingest_runner()
        if result is not None:
            result["model_release"] = release


def watch(
    *,
    max_batches: int = DEFAULT_MAX_BATCHES,
    max_units: int = DEFAULT_MAX_UNITS,
    sleep_seconds: float = DEFAULT_SLEEP_SECONDS,
    idle_sleep_seconds: float = DEFAULT_IDLE_SLEEP_SECONDS,
    log_file: Path | None = None,
) -> None:
    """Run drain cycles forever for launchd KeepAlive mode."""
    if idle_sleep_seconds < 0:
        raise ValueError("idle_sleep_seconds must be >= 0")

    while True:
        try:
            result = drain(
                max_batches=max_batches,
                max_units=max_units,
                sleep_seconds=sleep_seconds,
                log_file=log_file,
            )
            print(json.dumps(result, ensure_ascii=False), flush=True)
        except Exception as exc:
            print(
                json.dumps(
                    {
                        "status": "error",
                        "error": str(exc),
                        "timestamp": datetime.now().isoformat(),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        if idle_sleep_seconds:
            time.sleep(idle_sleep_seconds)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Drain Chronovisor pending raw ingest backlog."
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=_env_int("CHRONOVISOR_INGEST_DRAIN_MAX_BATCHES", DEFAULT_MAX_BATCHES),
        help="Maximum orchestrator batches to run in this process.",
    )
    parser.add_argument(
        "--max-units",
        type=int,
        default=_env_int("CHRONOVISOR_INGEST_DRAIN_MAX_UNITS", DEFAULT_MAX_UNITS),
        help="Semantic work units per orchestrator batch (1-10).",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=_env_float(
            "CHRONOVISOR_INGEST_DRAIN_SLEEP_SECONDS", DEFAULT_SLEEP_SECONDS
        ),
        help="Delay between successful batches.",
    )
    parser.add_argument("--log-file", type=Path)
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Stay alive and run drain cycles repeatedly. Intended for launchd KeepAlive.",
    )
    parser.add_argument(
        "--idle-sleep-seconds",
        type=float,
        default=_env_float(
            "CHRONOVISOR_INGEST_DRAIN_IDLE_SLEEP_SECONDS", DEFAULT_IDLE_SLEEP_SECONDS
        ),
        help="Delay between drain cycles in --watch mode.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the ``chronovisor-ingest-drain`` command-line entry point."""
    from chronovisor.core.okf_cutover import OKFStartupBlocked
    try:
        with okf_runtime_operation(CHRONOVISOR_ROOT):
            return _main_locked(argv)
    except OKFStartupBlocked:
        print(json.dumps({"status": "blocked", "category": "okf_startup_blocked"}))
        return 75


def _main_locked(argv: list[str] | None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not okf_startup_status(CHRONOVISOR_ROOT).allowed:
        print(
            json.dumps({"status": "blocked", "category": "okf_startup_blocked"})
        )
        return 75
    if args.watch:
        watch(
            max_batches=args.max_batches,
            max_units=args.max_units,
            sleep_seconds=args.sleep_seconds,
            idle_sleep_seconds=args.idle_sleep_seconds,
            log_file=args.log_file,
        )
        return 0
    try:
        result = drain(
            max_batches=args.max_batches,
            max_units=args.max_units,
            sleep_seconds=args.sleep_seconds,
            log_file=args.log_file,
        )
    except Exception as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=False))
        return 1

    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
