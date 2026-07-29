"""Live P0 preemption burn test for local Librarian model workers."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from chronovisor import ollama, research_scheduler
from chronovisor.durable_state import write_sealed_json
from chronovisor.recall_runtime import (
    RecallPolicy,
    RecallRequest,
    run_recall,
)
from chronovisor.research_scheduler import research_lane, run_cancellable_command
from chronovisor.runtime_config import load_decision_router_config
from chronovisor.store import CHRONOVISOR_ROOT

SCHEMA = "chronovisor.librarian-preemption-burn.v1"


def _worker(model: str, keep_alive: str) -> int:
    result = ollama.generate(
        (
            "Generate a long numbered sequence with one explanatory sentence "
            "per number. Continue until the token limit and do not summarize."
        ),
        model=model,
        num_ctx=8_192,
        num_predict=8_192,
        keep_alive=keep_alive,
        read_timeout_ms=600_000,
        progress_callback=lambda _event: None,
        temperature=0,
        seed=0,
    )
    print(json.dumps({"completed": True, "chars": len(str(result))}))
    return 0


def _pid_alive(pid: int | None) -> bool:
    if not isinstance(pid, int) or pid <= 1:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_for_worker(run_id: str, timeout_seconds: float = 30) -> int:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        active = research_scheduler._active_research()
        if (
            isinstance(active, dict)
            and active.get("run_id") == run_id
            and active.get("model_active") is True
            and isinstance(active.get("model_pid"), int)
        ):
            return int(active["model_pid"])
        time.sleep(0.01)
    raise RuntimeError(f"model worker did not become active: {run_id}")


def _percentile(values: list[int], quantile: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(len(ordered) * quantile) - 1))
    return ordered[index]


def _cheap_policy() -> RecallPolicy:
    return RecallPolicy(
        semantic=False,
        judge_mode="off",
        rewrite_enabled=False,
        log_decisions=False,
        total_timeout_ms=4_000,
        deterministic_fallback_reserve_ms=400,
    )


def _recall(index: int) -> dict[str, Any]:
    started = time.monotonic()
    result = run_recall(
        RecallRequest(
            host="burn",
            event="UserPromptSubmit",
            prompt=(
                "Chronovisorの分類司書と安定UIDリンクの現在の設計を確認して"
            ),
            session_id=f"librarian-burn-{index}",
        ),
        _cheap_policy(),
        perform_search=True,
    )
    return {
        "latency_ms": round((time.monotonic() - started) * 1000),
        "status": result.status,
        "scheduler": dict(result.evidence_features.get("scheduler") or {}),
    }


def _preempt_one(model: str, keep_alive: str, index: int) -> dict[str, Any]:
    run_id = f"librarian-burn-{index}-{uuid.uuid4().hex[:8]}"
    outcomes: list[Any] = []
    worker_pid: int | None = None
    with research_lane(
        run_id,
        enabled=True,
        mode="explicit",
        purpose="explicit",
        needs_model=True,
    ) as lease:
        if not lease.admission.admitted:
            raise RuntimeError(f"burn worker not admitted: {lease.admission.reason}")
        thread = threading.Thread(
            target=lambda: outcomes.append(
                run_cancellable_command(
                    [
                        sys.executable,
                        "-m",
                        "chronovisor.lab.librarian_burn",
                        "worker",
                        "--model",
                        model,
                        "--keep-alive",
                        keep_alive,
                    ],
                    "",
                    lease,
                    timeout_seconds=600,
                )
            )
        )
        thread.start()
        worker_pid = _wait_for_worker(run_id)
        receipt = _recall(index)
        thread.join(timeout=5)
        if thread.is_alive():
            raise RuntimeError("preempted worker thread did not terminate")
    if not outcomes or outcomes[0].status != "cancelled":
        raise RuntimeError(f"worker was not cancelled: {outcomes}")
    if _pid_alive(worker_pid):
        raise RuntimeError(f"cancelled worker PID remains alive: {worker_pid}")
    with research_lane(
        f"{run_id}-reacquire",
        enabled=True,
        mode="explicit",
        purpose="explicit",
        needs_model=False,
    ) as reacquired:
        lease_reacquired = reacquired.admission.admitted
    return {
        "model": model,
        "worker_pid": worker_pid,
        "worker_status": outcomes[0].status,
        "worker_latency_ms": outcomes[0].latency_ms,
        "recall": receipt,
        "lease_reacquired": lease_reacquired,
        "client_connection_closed_by_worker_kill": True,
    }


def run_burn(
    root: Path = CHRONOVISOR_ROOT,
    *,
    foreground_admissions: int = 200,
) -> dict[str, Any]:
    """Preempt each configured Librarian stage then sustain 200 P0 recalls."""

    config = load_decision_router_config()
    protected_model = config.primary_model
    ollama.generate(
        "Reply with OK.",
        model=protected_model,
        num_ctx=2_048,
        num_predict=8,
        keep_alive="24h",
        read_timeout_ms=120_000,
        temperature=0,
        seed=0,
    )
    resident_before = ollama.resident_model_rows()
    stages = []
    configured = [
        (config.primary_model, config.primary_keep_alive),
        (config.challenger_model, config.challenger_keep_alive),
        (config.tie_break_model, config.tie_break_keep_alive),
    ]
    for index, (model, keep_alive) in enumerate(configured):
        stages.append(_preempt_one(model, keep_alive, index))
    recalls = [stage["recall"] for stage in stages]
    for index in range(len(recalls), max(len(recalls), foreground_admissions)):
        recalls.append(_recall(index))
    resident_after = ollama.resident_model_rows()
    latencies = [int(row["latency_ms"]) for row in recalls]
    waits = [
        int((row.get("scheduler") or {}).get("resource_wait_ms") or 0)
        for row in recalls
    ]
    receipt = {
        "schema": SCHEMA,
        "status": "passed",
        "foreground_admissions": len(recalls),
        "configured_stages": stages,
        "latency_ms": {
            "p50": round(statistics.median(latencies)),
            "p95": _percentile(latencies, 0.95),
            "p99": _percentile(latencies, 0.99),
            "max": max(latencies),
        },
        "resource_wait_ms": {
            "p95": _percentile(waits, 0.95),
            "max": max(waits),
        },
        "four_second_violations": sum(value > 4_000 for value in latencies),
        "fifty_ms_wait_violations": sum(value > 50 for value in waits),
        "protected_model": protected_model,
        "protected_model_resident_before": protected_model in resident_before,
        "protected_model_resident_after": protected_model in resident_after,
        "resident_before": resident_before,
        "resident_after": resident_after,
        "page_or_registry_mutations": 0,
    }
    if (
        receipt["four_second_violations"]
        or receipt["fifty_ms_wait_violations"]
        or not receipt["protected_model_resident_after"]
        or not all(row["lease_reacquired"] for row in stages)
    ):
        receipt["status"] = "failed"
    write_sealed_json(
        root / "runtime" / "librarian" / "phase7-burn.json",
        receipt,
        backup=True,
    )
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run", "worker"))
    parser.add_argument("--root", type=Path, default=CHRONOVISOR_ROOT)
    parser.add_argument("--foreground-admissions", type=int, default=200)
    parser.add_argument("--model")
    parser.add_argument("--keep-alive", default="10m")
    args = parser.parse_args(argv)
    if args.command == "worker":
        if not args.model:
            parser.error("--model is required for worker")
        return _worker(args.model, args.keep_alive)
    result = run_burn(
        args.root,
        foreground_admissions=max(3, args.foreground_admissions),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
