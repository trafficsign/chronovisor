"""Locally challenge evidence bundles under the same sync-first scheduler."""

from __future__ import annotations

import json
import sys
import uuid
from collections.abc import Callable
from typing import Any

from chronovisor.core import ollama
from chronovisor.core.research_scheduler import (
    ResearchLease,
    research_lane,
    run_cancellable_command,
)
from chronovisor.research.evidence_bundle import EvidenceBundle
from chronovisor.search.research_config import ResearchConfig
from chronovisor.search.research_store import ResearchStore
from chronovisor.search.research_types import BudgetUsage

Runner = Callable[[str, str, ResearchLease, ResearchConfig], dict[str, Any]]


def _default_runner(
    operation: str,
    prompt: str,
    lease: ResearchLease,
    config: ResearchConfig,
) -> dict[str, Any]:
    runtime_role = {
        "challenge": "research.challenge",
        "tie_break": "research.tie_break",
    }.get(operation)
    if runtime_role is None:
        return {"status": "error", "error": "request_invalid"}
    try:
        routes = ollama.runtime_generation_routes((runtime_role,))
    except ollama.RuntimeBridgeError as exc:
        return {"status": "error", "error": exc.category}
    if len(routes) != 1 or routes[0].role != runtime_role:
        return {"status": "error", "error": "route_configuration_invalid"}
    route = routes[0]
    if not route.structured_output:
        return {"status": "error", "error": "capability_unavailable"}
    route_identity = {
        "role": route.role,
        "provider": route.provider,
        "model": route.model,
        "location": route.location,
    }
    request = {
        "operation": operation,
        "expected_model": route.model,
        "expected_location": route.location,
        "num_ctx": 32_768,
        "num_predict": config.budgets.max_single_generation_tokens,
        "read_timeout_ms": round(config.budgets.max_single_generation_seconds * 1000),
        "max_input_chars": 60_000,
        "max_output_chars": 5_000,
        "max_feedback_chars": 2_000,
        "prompt": prompt,
    }
    outcome = run_cancellable_command(
        [sys.executable, "-m", "chronovisor.research.research_model_worker"],
        json.dumps(request, ensure_ascii=False),
        lease,
        timeout_seconds=config.budgets.max_single_generation_seconds,
    )
    if outcome.status != "completed" or not isinstance(outcome.value, dict):
        status = (
            outcome.status
            if outcome.status in {"cancelled", "deferred", "error", "timeout"}
            else "error"
        )
        return {
            "status": status,
            "error": f"worker_{status}",
            "route": route_identity,
        }
    return {
        "status": "completed",
        **outcome.value,
        "latency_ms": outcome.latency_ms,
        "route": route_identity,
    }


def challenge_bundle(
    bundle: EvidenceBundle,
    *,
    config: ResearchConfig,
    usage: BudgetUsage | None = None,
    store: ResearchStore | None = None,
    runner: Runner | None = None,
) -> dict[str, Any]:
    usage = usage or BudgetUsage()
    store = store or ResearchStore()
    runner = runner or _default_runner
    run_id = bundle.research_run_id
    result: dict[str, Any] = {"status": "skipped", "reason": "budget"}
    with research_lane(
        f"{run_id}-challenge-{uuid.uuid4().hex[:8]}",
        enabled=config.enabled,
        mode=config.mode,
        purpose="explicit",
        needs_model=True,
    ) as lease:
        if not lease.admission.admitted:
            result = {"status": "deferred", "reason": lease.admission.reason}
        elif not usage.consume(config.budgets, "challenge_calls"):
            result = {"status": "skipped", "reason": "challenge_budget_exhausted"}
        else:
            prompt = json.dumps(bundle.to_dict(), ensure_ascii=False)[:60_000]
            challenge = runner(
                "challenge",
                prompt,
                lease,
                config,
            )
            repairs = int(challenge.get("repair_turns") or 0)
            if repairs and usage.can_consume(config.budgets, "repair_calls", repairs):
                usage.consume(config.budgets, "repair_calls", repairs)
            value = (
                challenge.get("value")
                if isinstance(challenge.get("value"), dict)
                else {}
            )
            result = {"status": challenge.get("status", "error"), "challenger": value}
            if isinstance(challenge.get("route"), dict):
                result["route"] = challenge["route"]
            disagreement = (
                challenge.get("status") == "completed"
                and challenge.get("ok") is not False
                and value.get("verdict") in {"confirm", "reject", "inconclusive"}
                and isinstance(value.get("contradictions"), list)
                and (
                    value.get("verdict") in {"reject", "inconclusive"}
                    or bool(value["contradictions"])
                )
            )
            if disagreement and usage.consume(config.budgets, "tie_break_calls"):
                tie_prompt = json.dumps(
                    {"bundle": bundle.to_dict(), "challenger": value},
                    ensure_ascii=False,
                )[:60_000]
                tie = runner(
                    "tie_break",
                    tie_prompt,
                    lease,
                    config,
                )
                result["tie_break"] = (
                    tie.get("value") if isinstance(tie.get("value"), dict) else {}
                )
                if isinstance(tie.get("route"), dict):
                    result["tie_route"] = tie["route"]
    result["usage"] = usage.to_dict()
    store.append_event(
        run_id,
        {"kind": "evidence_challenge", "result": result, "terminal": True},
    )
    return result
