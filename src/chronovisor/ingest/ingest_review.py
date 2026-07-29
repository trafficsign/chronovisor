"""Normalization and execution boundary for ingest semantic review."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable


def normalize_ingest_frontier_review(
    value: object,
    *,
    proposal: dict[str, Any],
) -> dict[str, Any]:
    """Normalize the final disposition and fail closed on silent data loss."""

    if not isinstance(value, dict):
        return {
            "decision": "retry",
            "summary": "local consensus reviewer returned a non-object payload",
            "failed_operations_disposition": "retry_required",
        }
    from chronovisor.decision.decision_schema_manifest import canonical_ingest_repair_arrays

    value = canonical_ingest_repair_arrays(value)
    raw_decision = value.get("decision")
    decision = {
        "approved": "apply_available",
        "rejected": "retry",
        "needs_retry": "retry",
    }.get(str(raw_decision), raw_decision)
    summary = value.get("summary")
    if decision not in {
        "apply_available",
        "confirmed_noop",
        "retry",
        "quarantined",
    }:
        return {
            **value,
            "decision": "retry",
            "summary": "local consensus reviewer returned an invalid decision",
            "failed_operations_disposition": "retry_required",
        }
    if not isinstance(summary, str) or not summary.strip():
        return {
            **value,
            "decision": "retry",
            "summary": "local consensus reviewer omitted its decision summary",
            "failed_operations_disposition": "retry_required",
        }
    repair_requested = any(
        isinstance(value.get(field), list) and bool(value.get(field))
        for field in ("invalid_tags", "replacement_operations")
    )
    if decision in {"apply_available", "confirmed_noop"} and repair_requested:
        decision = "retry"
        summary = (
            "repair instructions require a fresh review before terminal "
            f"disposition: {summary.strip()}"
        )
    if decision == "apply_available" and value.get("frontier_failure"):
        return {
            **value,
            "decision": "retry",
            "summary": "local consensus verdict carried a failure payload",
            "failed_operations_disposition": "retry_required",
        }

    prepared = proposal.get("prepared_operations")
    has_available_operations = isinstance(prepared, list) and bool(prepared)
    failed_specs = proposal.get("failed_operation_specs")
    has_failed_operations = isinstance(failed_specs, list) and bool(failed_specs)
    disposition = value.get("failed_operations_disposition")
    if repair_requested:
        disposition = "retry_required"
    elif not has_failed_operations and disposition is None:
        disposition = "none"

    if disposition not in {"none", "confirmed_unnecessary", "retry_required"}:
        return {
            **value,
            "decision": "retry",
            "summary": (
                "local consensus must explicitly disposition locally failed operations"
                if has_failed_operations
                else "local consensus returned an invalid failed-operation disposition"
            ),
            "failed_operations_disposition": "retry_required",
        }
    if has_failed_operations and decision in {"apply_available", "confirmed_noop"}:
        if disposition != "confirmed_unnecessary":
            return {
                **value,
                "decision": "retry",
                "summary": (
                    "partial local generation remains replayable until local consensus "
                    "explicitly confirms failed operations are unnecessary"
                ),
                "failed_operations_disposition": "retry_required",
            }
    if not has_failed_operations and not repair_requested:
        disposition = "none"
    if decision == "apply_available" and not has_available_operations:
        return {
            **value,
            "decision": "retry",
            "summary": "local consensus requested apply_available with no prepared operation",
            "failed_operations_disposition": (
                "retry_required" if has_failed_operations else "none"
            ),
        }
    return {
        **value,
        "decision": decision,
        "summary": summary.strip(),
        "failed_operations_disposition": disposition,
    }


def run_ingest_frontier_review(
    proposal: dict[str, Any],
    *,
    reviewer: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    repo_root: Path,
    decision_schema: dict[str, Any],
    prompt_builder: Callable[[dict[str, Any]], str] | None = None,
) -> dict[str, Any]:
    """Run an injected or production review and normalize its verdict."""

    if reviewer is not None:
        return normalize_ingest_frontier_review(reviewer(proposal), proposal=proposal)

    from chronovisor.decision.frontier_review import run_structured_review

    if prompt_builder is None:
        from chronovisor.decision.decision_lane_prompts import (
            build_ingest_reconciliation_prompt,
        )

        prompt_builder = build_ingest_reconciliation_prompt
    prompt = prompt_builder(proposal)
    result = run_structured_review(
        prompt,
        decision_schema,
        repo_root=repo_root,
        execute_patch=False,
        decision_lane="ingest_reconciliation",
    )
    return normalize_ingest_frontier_review(result, proposal=proposal)
