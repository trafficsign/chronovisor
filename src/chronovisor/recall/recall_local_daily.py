"""One local-only daily maintenance cycle using the normal distillation worker."""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from chronovisor.core import ollama
from chronovisor.core.model_execution import local_reasoning_settings
from chronovisor.core.store import CHRONOVISOR_ROOT
from chronovisor.recall import recall_distillation as distill
from chronovisor.recall import recall_distillation_store as store

RECEIPT_SCHEMA = "chronovisor.recall-local-daily-cycle.v1"


def run_cycle(root: Path) -> dict[str, Any]:
    """Share the worker lease and all existing validation/promotion rules."""

    if (root / "runtime/recall-luna-bulk/supervisor-maintenance.json").exists():
        return {"status": "deferred", "reason": "maintenance"}
    config = distill.load_distillation_config(root / "config.toml")
    if not config.enabled or config.teacher_profile != distill.LOCAL_SINGLE_PROFILE:
        return {"status": "disabled", "reason": "local_teacher_profile_inactive"}
    routes = ollama.runtime_generation_routes((distill.TEACHER_ROLES[0],))
    route = routes[0]
    if route.location != "local":
        return {"status": "held", "reason": "teacher_route_not_local"}
    reasoning = local_reasoning_settings(route.role, route.model, root=root)
    if reasoning is None:
        return {"status": "held", "reason": "local_reasoning_not_configured"}
    identity = {
        "role": route.role, "provider": route.provider, "model": route.model,
        "location": route.location, "revision": route.revision,
        "reasoning": reasoning, "profile": config.teacher_profile,
    }
    directory = store.distillation_dir(root) / "local-daily"
    lock = store.acquire_nonblocking_lock(directory / "daily-worker.lock")
    if lock is None:
        return {"status": "deferred", "reason": "daily_worker_busy"}
    try:
        started = time.monotonic()
        now = datetime.now(UTC).isoformat()
        status = {
            "kind": "local-daily-cycle", "status": "running",
            "started_at": now, "day": datetime.now().astimezone().date().isoformat(),
            "teacher": identity, "execution_mode": "local_only",
        }
        store.write_sealed_state(directory / "status.json", status)
        backlog: int | None = None
        try:
            result = distill.run_distillation_chunk(
                root=root, config_path=root / "config.toml", cold_start=True,
                max_elapsed_seconds=300,
            )
            status.update(
                status=str(result.get("status") or "unknown"),
                reason=str(result.get("reason") or ""),
                processed=int(result.get("processed") or 0),
                labels_written=int(result.get("labels_written") or 0),
                counterfactuals_written=int(result.get("counterfactuals_written") or 0),
                promotion_allowed=result.get("p5_allowed") is True,
            )
            workset = result.get("local_workset")
            if isinstance(workset, dict) and type(workset.get("backlog")) is int:
                backlog = workset["backlog"]
        except Exception as exc:
            status.update(status="error", reason=type(exc).__name__)
        status.update(finished_at=datetime.now(UTC).isoformat(), elapsed_seconds=round(time.monotonic() - started, 3))
        rate = status.get("labels_written", 0) * 3600 / status["elapsed_seconds"] if status["elapsed_seconds"] else 0.0
        status["drain"] = {
            "items_per_hour": round(rate, 3), "pending": backlog,
            "eta_hours_at_observed_rate": round(backlog / rate, 3) if rate and backlog is not None else None,
            "basis": "last_full_local_cycle_including_shared_runtime_wait",
        }
        receipt_id, _, _ = store.write_immutable(directory / "cycles", status, schema=RECEIPT_SCHEMA)
        status["receipt_id"] = receipt_id
        store.write_sealed_state(directory / "status.json", status)
        return status
    finally:
        lock.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=CHRONOVISOR_ROOT)
    args = parser.parse_args()
    result = run_cycle(args.root.expanduser().resolve())
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    raise SystemExit(1 if result.get("status") == "error" else 0)


if __name__ == "__main__":
    main()
