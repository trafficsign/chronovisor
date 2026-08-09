"""Forced Recall overlap cohort for classification LLM and embedding stages."""

from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from chronovisor.classification.classification import ClassificationError
from chronovisor.classification.classification_evidence_judgment import (
    current_resident_models,
)
from chronovisor.core import ollama, research_scheduler
from chronovisor.core.durable_state import write_sealed_json
from chronovisor.core.research_scheduler import (
    research_lane,
    run_cancellable_command,
)
from chronovisor.core.runtime_config import (
    load_decision_router_config,
    load_embedding_config,
)
from chronovisor.core.store import CHRONOVISOR_ROOT
from chronovisor.ops.convergence import ConvergenceStore, RetryPolicy
from chronovisor.recall.recall_runtime import RecallPolicy, RecallRequest, run_recall

SCHEMA = "chronovisor.classification-resource-overlap-burn.v1"


def _percentile(values: list[int], quantile: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(len(ordered) * quantile) - 1))
    return int(ordered[index])


def _metrics(values: list[int]) -> dict[str, int]:
    return {
        "p50": round(statistics.median(values)) if values else 0,
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
        "max": max(values, default=0),
    }


def _policy() -> RecallPolicy:
    return RecallPolicy(
        semantic=False,
        judge_mode="off",
        rewrite_enabled=False,
        log_decisions=False,
        total_timeout_ms=4_000,
        deterministic_fallback_reserve_ms=400,
    )


def _recall(index: int) -> tuple[dict[str, Any], str]:
    started = time.monotonic()
    result = run_recall(
        RecallRequest(
            host="classification-resource-burn",
            event="UserPromptSubmit",
            prompt="Chronovisorの分類司書と安定UIDリンクの現在設計を確認して",
            cwd=str(Path.cwd()),
            session_id=f"classification-resource-burn-{index}-{uuid.uuid4().hex}",
        ),
        _policy(),
        perform_search=True,
    )
    canonical = {
        "status": result.status,
        "decision": result.decision,
        "queries": result.queries,
        "items": [
            {
                "page_id": item.page_id,
                "uid": item.uid,
                "title": item.title,
                "score": item.score,
                "snippets": item.snippets,
                "sensitivity": item.sensitivity,
            }
            for item in result.context_items
        ],
        "search_mode": result.search_mode,
    }
    fingerprint = hashlib.sha256(
        json.dumps(
            canonical,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return (
        {
            "latency_ms": round((time.monotonic() - started) * 1_000),
            "status": result.status,
            "scheduler": dict(result.evidence_features.get("scheduler") or {}),
        },
        fingerprint,
    )


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


def _wait_for_probe(run_id: str, ready_file: Path, timeout_seconds: float = 30) -> int:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        active = research_scheduler._active_research()
        if (
            ready_file.is_file()
            and isinstance(active, dict)
            and active.get("run_id") == run_id
            and active.get("model_active") is True
            and isinstance(active.get("model_pid"), int)
        ):
            return int(active["model_pid"])
        time.sleep(0.01)
    raise ClassificationError(f"resource probe did not become active: {run_id}")


def _isolated_requeue_store(root: Path) -> tuple[ConvergenceStore, str]:
    base = root / "classification" / "library-evidence" / "resource-burn" / "queue"
    store = ConvergenceStore(
        base / "state.json",
        events_file=base / "events.jsonl",
        lock_file=base / "state.lock",
        policy=RetryPolicy(
            max_local_attempts=1,
            max_frontier_attempts=0,
            local_base_delay_seconds=0,
            frontier_base_delay_seconds=0,
            max_delay_seconds=0,
            lease_seconds=60,
        ),
    )
    merged = store.merge_item(
        lane="classification_resource_burn",
        source_id="forced-overlap",
        input_data={"schema": SCHEMA},
        resolver_version=SCHEMA,
    )
    return store, str(merged["item"]["key"])


def _resource_ready(
    *,
    protected_model: str,
    started: float,
) -> tuple[bool, int, list[str]]:
    deadline = started + 2
    residents: list[str] = []
    while time.monotonic() <= deadline:
        residents = current_resident_models()
        with research_lane(
            f"classification-resource-ready-{uuid.uuid4().hex}",
            enabled=True,
            mode="explicit",
            purpose="explicit",
            needs_model=False,
        ) as lease:
            admitted = lease.admission.admitted
        if admitted and protected_model in residents:
            return True, round((time.monotonic() - started) * 1_000), residents
        time.sleep(0.025)
    return False, round((time.monotonic() - started) * 1_000), residents


def _overlap_one(
    *,
    root: Path,
    stage: str,
    kind: str,
    model: str,
    protected_model: str,
    index: int,
    queue_store: ConvergenceStore,
    queue_key: str,
) -> dict[str, Any]:
    run_id = f"classification-{stage}-{index}-{uuid.uuid4().hex[:10]}"
    ready_file = (
        root
        / "classification"
        / "library-evidence"
        / "resource-burn"
        / "ready"
        / f"{run_id}.json"
    )
    outcomes: list[Any] = []
    owner = f"resource-burn:{os.getpid()}:{uuid.uuid4().hex}"
    claim = queue_store.claim_attempt(queue_key, "local", owner=owner)
    if not claim["claimed"]:
        raise ClassificationError(f"resource burn queue unavailable: {claim['reason']}")
    worker_pid: int | None = None
    recall: dict[str, Any]
    fingerprint: str
    recall_started = 0.0
    with research_lane(
        run_id,
        enabled=True,
        mode="explicit",
        purpose="explicit",
        needs_model=True,
    ) as lease:
        if not lease.admission.admitted:
            raise ClassificationError(
                f"resource burn probe not admitted: {lease.admission.reason}"
            )
        thread = threading.Thread(
            target=lambda: outcomes.append(
                run_cancellable_command(
                    [
                        sys.executable,
                        "-m",
                        "chronovisor.classification.classification_resource_probe",
                        kind,
                        "--model",
                        model,
                        "--ready-file",
                        str(ready_file),
                    ],
                    "",
                    lease,
                    timeout_seconds=600,
                )
            )
        )
        thread.start()
        worker_pid = _wait_for_probe(run_id, ready_file)
        time.sleep(0.075)
        recall_started = time.monotonic()
        recall, fingerprint = _recall(index)
        thread.join(timeout=5)
        if thread.is_alive():
            raise ClassificationError("preempted resource probe did not terminate")
    ready_file.unlink(missing_ok=True)
    cancel_ack_ms = round((time.monotonic() - recall_started) * 1_000)
    ready_started = time.monotonic()
    resource_ready, resource_ready_ms, residents = _resource_ready(
        protected_model=protected_model,
        started=ready_started,
    )
    queue = queue_store.fail_attempt(
        queue_key,
        "local",
        owner=owner,
        error="forced foreground overlap",
        failure_class="foreground_preempted",
        allow_frontier=False,
        consume_attempt=False,
    )
    scheduler = dict(recall.get("scheduler") or {})
    return {
        "stage": stage,
        "kind": kind,
        "model": model,
        "worker_pid": worker_pid,
        "worker_pid_alive": _pid_alive(worker_pid),
        "worker_status": outcomes[0].status if outcomes else "missing",
        "recall": recall,
        "recall_fingerprint": fingerprint,
        "cancel_ack_ms": cancel_ack_ms,
        "cancel_to_resource_ready_ms": resource_ready_ms,
        "resource_ready": resource_ready,
        "protected_resident": protected_model in residents,
        "resident_models": residents,
        "research_overlap": scheduler.get("research_overlap") is True,
        "research_preempted": scheduler.get("research_preempted") is True,
        "foreground_wait_ms": int(scheduler.get("resource_wait_ms") or 0),
        "lease_residual": research_scheduler._active_research() is not None,
        "attempts_consumed": int(queue["item"].get("local_attempts") or 0),
        "requeued": queue["item"].get("status") == "pending_local",
    }


def run_resource_burn(
    root: Path = CHRONOVISOR_ROOT,
    *,
    samples_per_stage: int = 50,
) -> dict[str, Any]:
    if samples_per_stage < 50:
        raise ClassificationError(
            "resource burn requires at least 50 samples per stage"
        )
    config = load_decision_router_config()
    embedding_model = load_embedding_config().model
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
    baseline_rows = [_recall(-(index + 1)) for index in range(samples_per_stage)]
    baseline = [row for row, _fingerprint in baseline_rows]
    baseline_fingerprints = {fingerprint for _row, fingerprint in baseline_rows}
    queue_store, queue_key = _isolated_requeue_store(root)
    stages = (
        ("proposal", "llm", config.primary_model),
        ("audit", "llm", config.challenger_model),
        ("tie_break", "llm", config.tie_break_model),
        ("dense_embedding", "embed", embedding_model),
    )
    stage_rows: dict[str, list[dict[str, Any]]] = {}
    for stage, kind, model in stages:
        stage_rows[stage] = [
            _overlap_one(
                root=root,
                stage=stage,
                kind=kind,
                model=model,
                protected_model=protected_model,
                index=index,
                queue_store=queue_store,
                queue_key=queue_key,
            )
            for index in range(samples_per_stage)
        ]
    baseline_latency = [int(row["latency_ms"]) for row in baseline]
    baseline_metrics = _metrics(baseline_latency)
    summaries: dict[str, Any] = {}
    all_rows = []
    for stage, rows in stage_rows.items():
        all_rows.extend(rows)
        latency = [int(row["recall"]["latency_ms"]) for row in rows]
        metrics = _metrics(latency)
        p95_delta = metrics["p95"] - baseline_metrics["p95"]
        p99_delta = metrics["p99"] - baseline_metrics["p99"]
        summaries[stage] = {
            "sample_count": len(rows),
            "recall_latency_ms": metrics,
            "p95_delta_ms": p95_delta,
            "p99_delta_ms": p99_delta,
            "p95_delta_ratio": p95_delta / max(1, baseline_metrics["p95"]),
            "p99_delta_ratio": p99_delta / max(1, baseline_metrics["p99"]),
            "foreground_wait_max_ms": max(
                (int(row["foreground_wait_ms"]) for row in rows), default=0
            ),
            "cancel_ack_max_ms": max(
                (int(row["cancel_ack_ms"]) for row in rows), default=0
            ),
            "cancel_to_resource_ready_max_ms": max(
                (int(row["cancel_to_resource_ready_ms"]) for row in rows),
                default=0,
            ),
        }
    canonical = next(iter(baseline_fingerprints), "")
    gates = {
        "baseline_stable": len(baseline_fingerprints) == 1,
        "sample_size_each_stage": all(
            summary["sample_count"] >= 50 for summary in summaries.values()
        ),
        "foreground_admission": all(
            row["foreground_wait_ms"] <= 50 for row in all_rows
        ),
        "overlap_p95": all(
            summary["p95_delta_ms"] <= 50 and summary["p95_delta_ratio"] <= 0.10
            for summary in summaries.values()
        ),
        "overlap_p99": all(
            summary["p99_delta_ms"] <= 100 and summary["p99_delta_ratio"] <= 0.15
            for summary in summaries.values()
        ),
        "recall_deadline": all(
            row["recall"]["latency_ms"] <= 4_000 for row in all_rows
        ),
        "recall_result_identical": all(
            row["recall_fingerprint"] == canonical for row in all_rows
        ),
        "research_overlap_observed": all(
            row["research_overlap"] and row["research_preempted"] for row in all_rows
        ),
        "cancel_ack": all(row["cancel_ack_ms"] <= 250 for row in all_rows),
        "resource_ready": all(
            row["resource_ready"] and row["cancel_to_resource_ready_ms"] <= 2_000
            for row in all_rows
        ),
        "protected_residency": all(row["protected_resident"] for row in all_rows),
        "worker_and_lease_released": all(
            not row["worker_pid_alive"] and not row["lease_residual"]
            for row in all_rows
        ),
        "attempt_non_consuming_requeue": all(
            row["attempts_consumed"] == 0 and row["requeued"] for row in all_rows
        ),
    }
    receipt = {
        "schema": SCHEMA,
        "status": "passed" if all(gates.values()) else "failed",
        "gates": gates,
        "samples_per_stage": samples_per_stage,
        "protected_model": protected_model,
        "baseline": {
            "sample_count": len(baseline),
            "latency_ms": baseline_metrics,
            "fingerprint": canonical,
        },
        "stages": summaries,
        "samples": stage_rows,
        "models": {
            "proposal": config.primary_model,
            "audit": config.challenger_model,
            "tie_break": config.tie_break_model,
            "dense_embedding": embedding_model,
        },
        "page_or_registry_mutations": 0,
    }
    path = (
        root
        / "classification"
        / "library-evidence"
        / "evaluation"
        / "resource-gate.json"
    )
    write_sealed_json(path, receipt, backup=True)
    return receipt
