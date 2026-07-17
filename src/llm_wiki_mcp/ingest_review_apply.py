"""Authority-gated review and apply transaction for ingest proposals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from llm_wiki_mcp import decision_authority, runtime_status
from llm_wiki_mcp.ingest_review_plan import (
    IngestReviewBudgetExhausted,
    IngestReviewShardCapacityError,
    IngestReviewShardPlan as _IngestReviewShardPlan,
    IngestReviewShardPlanState as _IngestReviewShardPlanState,
)
from llm_wiki_mcp.ingest_schemas import (
    INGEST_FRONTIER_ARTIFACT_SCHEMA_VERSION,
    INGEST_REVIEW_SHARD_SCHEMA_VERSION,
)


def _runtime():
    # Imported lazily by ingest after its compatibility facade is initialized.
    from llm_wiki_mcp import ingest

    return ingest


def _runtime_call(name: str):
    def call(*args: Any, **kwargs: Any) -> Any:
        return getattr(_runtime(), name)(*args, **kwargs)

    return call


_apply_prepared_operations = _runtime_call("_apply_prepared_operations")
_build_ingest_frontier_proposal = _runtime_call("_build_ingest_frontier_proposal")
_build_ingest_review_shard_plan = _runtime_call("_build_ingest_review_shard_plan")
_canonical_json_sha256 = _runtime_call("_canonical_json_sha256")
_current_ingest_review_authority = _runtime_call("_current_ingest_review_authority")
_has_sharded_ingest_review_artifact_family = _runtime_call("_has_sharded_ingest_review_artifact_family")
_historical_ingest_sharded_review_recovery_error = _runtime_call("_historical_ingest_sharded_review_recovery_error")
_ingest_artifact_paths = _runtime_call("_ingest_artifact_paths")
_ingest_review_authority_error = _runtime_call("_ingest_review_authority_error")
_ingest_review_authority_shape_error = _runtime_call("_ingest_review_authority_shape_error")
_ingest_review_shard_failure = _runtime_call("_ingest_review_shard_failure")
_ingest_review_shard_manifest_path = _runtime_call("_ingest_review_shard_manifest_path")
_ingest_sharded_review_reuse_error = _runtime_call("_ingest_sharded_review_reuse_error")
_ingest_source_key = _runtime_call("_ingest_source_key")
_inspect_ingest_review_shard_plan_state = _runtime_call("_inspect_ingest_review_shard_plan_state")
_invalid_sharded_review_result = _runtime_call("_invalid_sharded_review_result")
_load_ingest_proposal = _runtime_call("_load_ingest_proposal")
_load_ingest_review = _runtime_call("_load_ingest_review")
_load_ingest_review_artifact = _runtime_call("_load_ingest_review_artifact")
_normalize_ingest_frontier_review = _runtime_call("_normalize_ingest_frontier_review")
_persist_ingest_review_continuation_marker = _runtime_call("_persist_ingest_review_continuation_marker")
_persist_ingest_review_shard_manifest = _runtime_call("_persist_ingest_review_shard_manifest")
_persist_ingest_review_stall = _runtime_call("_persist_ingest_review_stall")
_prepare_operations = _runtime_call("_prepare_operations")
_prepared_plan_is_fully_applied = _runtime_call("_prepared_plan_is_fully_applied")
_run_ingest_frontier_review = _runtime_call("_run_ingest_frontier_review")
_run_ingest_sharded_review = _runtime_call("_run_ingest_sharded_review")
_safe_log = _runtime_call("_safe_log")
_stored_ingest_review_shard_manifest_error = _runtime_call("_stored_ingest_review_shard_manifest_error")
_write_and_readback_ingest_review_artifact = _runtime_call("_write_and_readback_ingest_review_artifact")
_write_ingest_artifact = _runtime_call("_write_ingest_artifact")

# These types remain compatibility-owned by ingest until the final prepare/apply
# extraction. Importing here is safe because this module is loaded lazily.
from llm_wiki_mcp.ingest import (  # noqa: E402
    IngestApplyError,
    _IngestReviewShardContinuation,
)


@dataclass(frozen=True)
class IngestReviewArtifactState:
    """Pure structural projection of one optional durable review artifact."""

    review: dict[str, Any] | None
    authority: dict[str, Any] | None
    exact_postimages_already_applied: bool


def inspect_ingest_review_artifact(
    artifact: object,
    *,
    has_planned_operations: bool,
    planned_postimages_fully_applied: bool,
) -> IngestReviewArtifactState:
    """Project artifact fields without consulting live authority or the filesystem."""

    review_raw = artifact.get("review") if isinstance(artifact, dict) else None
    authority_raw = artifact.get("authority") if isinstance(artifact, dict) else None
    review = review_raw if isinstance(review_raw, dict) else None
    authority = authority_raw if isinstance(authority_raw, dict) else None
    exact = (
        review is not None
        and review.get("decision") == "apply_available"
        and has_planned_operations
        and planned_postimages_fully_applied
    )
    return IngestReviewArtifactState(
        review=review,
        authority=authority,
        exact_postimages_already_applied=exact,
    )


def review_and_apply_ingest_operations(
    operations: list[dict],
    *,
    raw_content: str,
    raw_keywords: list[str] | None = None,
    source_raw: str | None = None,
    triage_plan: list[dict] | None = None,
    failed_operation_specs: list[dict] | None = None,
    local_disposition: str = "operations_available",
    reviewer: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    force_frontier_review: bool = False,
    frontier_budget: "_FrontierCallBudget | None" = None,
    shard_continuation: _IngestReviewShardContinuation | None = None,
    allow_empty_shard_continuation: bool = False,
    continuation_reseed_from_sha256: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Authorize by risk policy, durably bind the verdict, and CAS apply."""

    source_key = _ingest_source_key(raw_content, raw_keywords)
    proposal_path, review_path = _ingest_artifact_paths(source_key)
    audit_state_path = proposal_path.parent / "audit-state.json"
    recovered = _load_ingest_proposal(
        proposal_path,
        source_key=source_key,
        raw_content=raw_content,
    )
    # Only a terminal verdict can pin a previous local proposal.  A durable
    # proposal without such a verdict represents retryable local/frontier
    # work; rebuild it from this attempt so a transient generation failure
    # cannot suppress a later complete plan forever.
    if recovered is not None:
        recovered_proposal, _recovered_planned = recovered
        recovered_sha256 = _canonical_json_sha256(recovered_proposal)
        recovered_review = _load_ingest_review(
            review_path,
            source_key=source_key,
            proposal_sha256=recovered_sha256,
        )
        if recovered_review is None:
            if shard_continuation is not None:
                if (
                    recovered_proposal != shard_continuation.proposal
                    or tuple(_recovered_planned) != shard_continuation.planned
                    or recovered_sha256 != shard_continuation.plan.full_proposal_sha256
                    or _stored_ingest_review_shard_manifest_error(
                        shard_continuation.plan,
                        source_key=source_key,
                    )
                    is not None
                ):
                    raise IngestApplyError(
                        "pre-triage shard continuation changed before review"
                    )
            else:
                if review_path.exists() and _has_sharded_ingest_review_artifact_family(
                    review_path,
                    proposal=recovered_proposal,
                    source_key=source_key,
                ):
                    return _invalid_sharded_review_result(
                        source_key=source_key,
                        proposal=recovered_proposal,
                        recovered_artifact=True,
                    )
                recovered = None
        elif shard_continuation is not None:
            raise IngestApplyError(
                "pre-triage shard continuation unexpectedly has a parent review"
            )
    elif shard_continuation is not None:
        raise IngestApplyError("pre-triage shard continuation proposal is missing")
    if recovered is None:
        planned, totals = _prepare_operations(operations, read_only=dry_run)
        proposal = _build_ingest_frontier_proposal(
            raw_content=raw_content,
            raw_keywords=raw_keywords,
            source_raw=source_raw,
            operations=operations,
            planned=planned,
            link_totals=totals,
            triage_plan=triage_plan,
            failed_operation_specs=failed_operation_specs,
            local_disposition=local_disposition,
        )
        from llm_wiki_mcp.ingest_audit import decide_ingest_audit

        audit_decision = decide_ingest_audit(
            source_key=source_key,
            raw_content=raw_content,
            operations=operations,
            failed_operation_specs=list(failed_operation_specs or []),
            local_disposition=local_disposition,
            state_path=audit_state_path,
            force=force_frontier_review,
            explicit_reviewer=reviewer is not None,
        ).to_dict()
        proposal["audit_decision"] = audit_decision
        recovered_artifact = False
    else:
        proposal, planned = recovered
        audit_raw = proposal.get("audit_decision")
        audit_decision = (
            dict(audit_raw)
            if isinstance(audit_raw, dict)
            else {
                "required": True,
                "mode": "legacy-frontier",
                "reasons": ["legacy reviewed artifact"],
            }
        )
        totals_raw = proposal.get("link_reconciliation")
        totals = (
            {
                key: int(totals_raw.get(key, 0))
                for key in ("resolved", "rewritten", "unwrapped")
            }
            if isinstance(totals_raw, dict)
            else {"resolved": 0, "rewritten": 0, "unwrapped": 0}
        )
        recovered_artifact = True

    proposal_sha256 = _canonical_json_sha256(proposal)
    if dry_run:
        return {
            "status": "dry_run",
            "source_key": source_key,
            "proposal_sha256": proposal_sha256,
            "proposal": proposal,
            "audit": audit_decision,
            "created": [],
            "updated": [],
            "artifact_written": False,
        }

    if not recovered_artifact:
        try:
            _write_ingest_artifact(
                proposal_path,
                {
                    "schema_version": INGEST_FRONTIER_ARTIFACT_SCHEMA_VERSION,
                    "kind": "ingest_frontier_proposal_artifact",
                    "source_key": source_key,
                    "proposal_sha256": proposal_sha256,
                    "proposal": proposal,
                },
            )
        except OSError as exc:
            return {
                "status": "needs_retry",
                "source_key": source_key,
                "proposal_sha256": proposal_sha256,
                "summary": f"review proposal artifact write failed: {exc}",
                "created": [],
                "updated": [],
                "audit": audit_decision,
            }

    review_artifact = _load_ingest_review_artifact(
        review_path,
        source_key=source_key,
        proposal_sha256=proposal_sha256,
    )
    if (
        review_artifact is None
        and review_path.exists()
        and _has_sharded_ingest_review_artifact_family(
            review_path,
            proposal=proposal,
            source_key=source_key,
        )
    ):
        return _invalid_sharded_review_result(
            source_key=source_key,
            proposal=proposal,
            recovered_artifact=recovered_artifact,
        )
    artifact_state = inspect_ingest_review_artifact(
        review_artifact,
        has_planned_operations=bool(planned),
        planned_postimages_fully_applied=(
            _prepared_plan_is_fully_applied(planned) if planned else False
        ),
    )
    artifact_review = artifact_state.review
    artifact_authority = artifact_state.authority
    exact_postimages_already_applied = (
        artifact_state.exact_postimages_already_applied
    )
    if (
        artifact_review is not None
        and artifact_authority is not None
        and "review_shard_proof" in artifact_review
    ):
        shard_reuse_error = (
            _historical_ingest_sharded_review_recovery_error(
                artifact_review,
                proposal,
                artifact_authority,
            )
            if exact_postimages_already_applied
            else _ingest_sharded_review_reuse_error(
                artifact_review,
                proposal,
                artifact_authority,
            )
        )
        if shard_reuse_error is not None:
            return {
                "status": "needs_retry",
                "source_key": source_key,
                "proposal_sha256": proposal_sha256,
                "review": _ingest_review_shard_failure(
                    "ingest_review_shard_reuse_invalid",
                    shard_reuse_error,
                ),
                "summary": shard_reuse_error,
                "recovered_artifact": recovered_artifact,
                "reused_review": False,
                "created": [],
                "updated": [],
                "audit": audit_decision,
            }

    # A process may die after the reviewed postimages are durable but before
    # the surrounding raw/job state is committed.  Exact postimage equality
    # proves this recovery performs no new semantic page write, so it remains
    # safe even if the adopted authority has since changed.  Partial batches,
    # confirmed-noop decisions, and legacy/malformed review artifacts never
    # receive this exception.
    if (
        artifact_review is not None
        and artifact_review.get("decision") == "apply_available"
        and artifact_authority is not None
        and _ingest_review_authority_shape_error(artifact_authority) is None
        and _ingest_review_authority_error(artifact_review, artifact_authority) is None
        and exact_postimages_already_applied
    ):
        created, updated = _apply_prepared_operations(
            planned,
            link_totals=totals,
            recovery_only=True,
        )
        return {
            "status": "apply_available",
            "source_key": source_key,
            "proposal_sha256": proposal_sha256,
            "review": artifact_review,
            "recovered_artifact": recovered_artifact,
            "reused_review": True,
            "recovery_basis": "exact_postimages_already_applied",
            "created": created,
            "updated": updated,
            "audit": audit_decision,
        }

    review_authority, authority_error = _current_ingest_review_authority(
        reviewer=reviewer
    )
    authority_shape_error = (
        _ingest_review_authority_shape_error(review_authority)
        if review_authority is not None
        else None
    )
    if (
        authority_error is not None
        or review_authority is None
        or authority_shape_error is not None
    ):
        return {
            "status": "needs_retry",
            "source_key": source_key,
            "proposal_sha256": proposal_sha256,
            "summary": authority_error
            or authority_shape_error
            or "ingest review authority is missing",
            "recovered_artifact": recovered_artifact,
            "reused_review": False,
            "created": [],
            "updated": [],
            "audit": audit_decision,
        }

    stale_review_reason: str | None = None
    if artifact_review is not None:
        if artifact_authority != review_authority:
            stale_review_reason = "ingest review authority changed before effect"
        else:
            stale_review_reason = _ingest_review_authority_error(
                artifact_review, review_authority
            )
    review = artifact_review if stale_review_reason is None else None
    reused_review = review is not None
    frontier_used = False
    if review is None:
        # Triage and generation are semantic model output.  Deterministic
        # schema/path/link validation proves that a proposal is well-formed;
        # it cannot prove that its claims are grounded in the raw.  Therefore
        # even an audit sampler's "low-risk" result must never authorize a
        # write or discard by itself.  Every semantic ingest disposition goes
        # through the lane-scoped local consensus gate, which fails closed
        # while shadowed or before an adoption artifact exists.
        frontier_used = True
        runtime_status.safe_write_status(stage="local-consensus-review")
        shard_plan: _IngestReviewShardPlan | None = None
        shard_baseline_indices: tuple[int, ...] = ()
        try:
            shard_plan = (
                shard_continuation.plan
                if shard_continuation is not None
                else _build_ingest_review_shard_plan(proposal)
            )
            if (
                shard_plan is not None
                and _ingest_review_shard_manifest_path(shard_plan).exists()
            ):
                baseline_state = _inspect_ingest_review_shard_plan_state(
                    shard_plan,
                    source_key=source_key,
                    authority=review_authority,
                )
                if baseline_state.invalid_reason is None:
                    shard_baseline_indices = baseline_state.current_approved_indices
            if shard_plan is None:
                if frontier_budget is not None and not frontier_budget.consume():
                    raise IngestReviewBudgetExhausted
                review = _run_ingest_frontier_review(proposal, reviewer=reviewer)
            else:
                review = _run_ingest_sharded_review(
                    shard_plan,
                    source_key=source_key,
                    reviewer=reviewer,
                    authority=review_authority,
                    frontier_budget=frontier_budget,
                )
        except IngestReviewBudgetExhausted:
            used = frontier_budget.used if frontier_budget is not None else 0
            limit = frontier_budget.limit if frontier_budget is not None else 0
            current_authority, current_authority_error = (
                _current_ingest_review_authority(reviewer=reviewer)
            )
            current_authority_shape_error = (
                _ingest_review_authority_shape_error(current_authority)
                if isinstance(current_authority, dict)
                else "current ingest review authority is missing"
            )
            if current_authority_error is not None or current_authority_shape_error:
                stale_summary = (
                    "ingest review authority changed before shard continuation "
                    "or became unavailable: "
                    + (current_authority_error or current_authority_shape_error)
                )
                return {
                    "status": "needs_retry",
                    "source_key": source_key,
                    "proposal_sha256": proposal_sha256,
                    "summary": stale_summary,
                    "review": _ingest_review_shard_failure(
                        "local_decision_authority_changed",
                        stale_summary,
                    ),
                    "recovered_artifact": recovered_artifact,
                    "reused_review": False,
                    "created": [],
                    "updated": [],
                    "audit": audit_decision,
                }
            assert isinstance(current_authority, dict)
            authority_epoch_changed = (
                decision_authority.compare_semantic_authority(
                    review_authority,
                    current_authority,
                    lane="ingest_reconciliation",
                )
                is not None
            )
            continuation_plan = shard_plan
            if authority_epoch_changed or (
                continuation_plan is None and allow_empty_shard_continuation
            ):
                try:
                    continuation_plan = _build_ingest_review_shard_plan(
                        proposal,
                        force_review_unit=True,
                    )
                except (IngestReviewShardCapacityError, TypeError, ValueError) as exc:
                    invalid_summary = (
                        "ingest review continuation reseed failed: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    return {
                        "status": "needs_retry",
                        "source_key": source_key,
                        "proposal_sha256": proposal_sha256,
                        "summary": invalid_summary,
                        "review": _ingest_review_shard_failure(
                            "ingest_review_shard_manifest_invalid",
                            invalid_summary,
                        ),
                        "recovered_artifact": recovered_artifact,
                        "reused_review": False,
                        "created": [],
                        "updated": [],
                        "audit": audit_decision,
                    }
            approved_shards = 0
            continuation_error: str | None = None
            shard_state: _IngestReviewShardPlanState | None = None
            if continuation_plan is not None:
                manifest_path = _ingest_review_shard_manifest_path(continuation_plan)
                if not manifest_path.exists():
                    continuation_error = _persist_ingest_review_shard_manifest(
                        continuation_plan,
                        source_key=source_key,
                    )
                if continuation_error is None:
                    shard_state = _inspect_ingest_review_shard_plan_state(
                        continuation_plan,
                        source_key=source_key,
                        authority=current_authority,
                    )
                    continuation_error = shard_state.invalid_reason
                    approved_shards = shard_state.approved_shards
            if continuation_error is not None:
                invalid_summary = (
                    "ingest review continuation proof is invalid: " + continuation_error
                )
                return {
                    "status": "needs_retry",
                    "source_key": source_key,
                    "proposal_sha256": proposal_sha256,
                    "summary": invalid_summary,
                    "review": _ingest_review_shard_failure(
                        "ingest_review_shard_artifact_invalid",
                        invalid_summary,
                    ),
                    "recovered_artifact": recovered_artifact,
                    "reused_review": False,
                    "created": [],
                    "updated": [],
                    "audit": audit_decision,
                }
            current_approved_indices = (
                shard_state.current_approved_indices if shard_state is not None else ()
            )
            made_progress = set(shard_baseline_indices) < set(current_approved_indices)
            exact_repair_reseeded = (
                allow_empty_shard_continuation
                and continuation_plan is not None
                and approved_shards == 0
            )
            continuation_allowed = (
                continuation_plan is not None
                and approved_shards < len(continuation_plan.shards)
                and (made_progress or authority_epoch_changed or exact_repair_reseeded)
            )
            if (
                continuation_plan is not None
                and shard_state is not None
                and not continuation_allowed
                and not authority_epoch_changed
                and not exact_repair_reseeded
            ):
                stall_write_error = _persist_ingest_review_stall(
                    source_key=source_key,
                    plan=continuation_plan,
                    authority=current_authority,
                    approved_indices=shard_state.current_approved_indices,
                )
                if stall_write_error is not None:
                    invalid_summary = (
                        "ingest review no-progress tombstone failed: "
                        + stall_write_error
                    )
                    return {
                        "status": "needs_retry",
                        "source_key": source_key,
                        "proposal_sha256": proposal_sha256,
                        "summary": invalid_summary,
                        "review": _ingest_review_shard_failure(
                            "ingest_review_shard_artifact_invalid",
                            invalid_summary,
                        ),
                        "recovered_artifact": recovered_artifact,
                        "reused_review": False,
                        "created": [],
                        "updated": [],
                        "audit": audit_decision,
                    }
            if (
                continuation_allowed
                and continuation_plan is not None
                and approved_shards == 0
            ):
                marker_reason = (
                    "authority_epoch_reseed"
                    if authority_epoch_changed
                    else "exact_repair_reseed"
                )
                previous_sha256 = (
                    proposal_sha256
                    if authority_epoch_changed
                    else str(continuation_reseed_from_sha256 or "")
                )
                marker_error = _persist_ingest_review_continuation_marker(
                    source_key=source_key,
                    plan=continuation_plan,
                    reason=marker_reason,
                    previous_full_proposal_sha256=previous_sha256,
                    previous_authority=review_authority,
                    current_authority=current_authority,
                )
                if marker_error is not None:
                    invalid_summary = (
                        "ingest review continuation marker failed: " + marker_error
                    )
                    return {
                        "status": "needs_retry",
                        "source_key": source_key,
                        "proposal_sha256": proposal_sha256,
                        "summary": invalid_summary,
                        "review": _ingest_review_shard_failure(
                            "ingest_review_shard_artifact_invalid",
                            invalid_summary,
                        ),
                        "recovered_artifact": recovered_artifact,
                        "reused_review": False,
                        "created": [],
                        "updated": [],
                        "audit": audit_decision,
                    }
            continuation = (
                {
                    "schema_version": INGEST_REVIEW_SHARD_SCHEMA_VERSION,
                    "kind": "ingest_review_shard_continuation",
                    "full_proposal_sha256": continuation_plan.full_proposal_sha256,
                    "manifest_sha256": continuation_plan.manifest_sha256,
                    "approved_shards": approved_shards,
                    "total_shards": len(continuation_plan.shards),
                    "remaining_shards": (
                        len(continuation_plan.shards) - approved_shards
                    ),
                    "review_calls_used": used,
                    "review_call_limit": limit,
                }
                if continuation_allowed and continuation_plan is not None
                else None
            )
            budget_status = (
                "shard_continuation_pending"
                if continuation is not None
                else "frontier_budget_exhausted"
            )
            runtime_status.safe_append_metric(
                "ingest_authorization",
                source_key=source_key,
                mode=str(audit_decision.get("mode") or "unknown"),
                frontier_used=used > 0,
                required=True,
                sample_rate=audit_decision.get("sample_rate"),
                caught_issue_rate=audit_decision.get("caught_issue_rate"),
                decision=budget_status,
            )
            result = {
                "status": budget_status,
                "source_key": source_key,
                "proposal_sha256": proposal_sha256,
                "summary": f"structured review budget exhausted ({used}/{limit})",
                "review": {
                    "decision": "retry",
                    "summary": (
                        "structured review budget exhausted; keep the raw "
                        "pending for local consensus"
                    ),
                },
                "recovered_artifact": recovered_artifact,
                "reused_review": False,
                "created": [],
                "updated": [],
                "audit": audit_decision,
            }
            if continuation is not None:
                result["shard_continuation"] = continuation
            return result
        except IngestReviewShardCapacityError as exc:
            review = _ingest_review_shard_failure(exc.failure_class, exc.reason)
        except Exception as exc:
            review = {
                "decision": "needs_retry",
                "summary": f"local consensus reviewer failed: {exc.__class__.__name__}: {exc}",
            }

    review = _normalize_ingest_frontier_review(review, proposal=proposal)
    decision = str(review.get("decision") or "retry")
    if decision in {"apply_available", "confirmed_noop"} and not reused_review:
        # A verdict is not durable until its own embedded audit and the live
        # adoption/policy epoch agree.  Re-resolving after the model call
        # catches authority replacement while consensus was in flight.
        current_authority, current_authority_error = _current_ingest_review_authority(
            reviewer=reviewer
        )
        if current_authority_error is not None or current_authority != review_authority:
            return {
                "status": "needs_retry",
                "source_key": source_key,
                "proposal_sha256": proposal_sha256,
                "review": review,
                "summary": current_authority_error
                or "ingest review authority changed during review",
                "recovered_artifact": recovered_artifact,
                "reused_review": False,
                "created": [],
                "updated": [],
                "audit": audit_decision,
            }
        if policy_error := _ingest_review_authority_error(review, review_authority):
            return {
                "status": "needs_retry",
                "source_key": source_key,
                "proposal_sha256": proposal_sha256,
                "review": review,
                "summary": policy_error,
                "recovered_artifact": recovered_artifact,
                "reused_review": False,
                "created": [],
                "updated": [],
                "audit": audit_decision,
            }
    runtime_status.safe_append_metric(
        "ingest_authorization",
        source_key=source_key,
        mode=str(audit_decision.get("mode") or "unknown"),
        frontier_used=frontier_used,
        required=audit_decision.get("required") is True,
        sample_rate=audit_decision.get("sample_rate"),
        caught_issue_rate=audit_decision.get("caught_issue_rate"),
        decision=decision,
    )
    _safe_log(
        f"ingest | authorization: {audit_decision.get('mode', 'unknown')} -> {decision}"
    )
    if frontier_used:
        try:
            from llm_wiki_mcp.ingest_audit import record_frontier_audit_outcome

            record_frontier_audit_outcome(
                state_path=audit_state_path,
                source_key=source_key,
                approved=decision in {"apply_available", "confirmed_noop"},
                mode=str(audit_decision.get("mode") or "mandatory"),
                reasons=[
                    str(reason)
                    for reason in audit_decision.get("reasons", [])
                    if isinstance(reason, str)
                ],
            )
        except Exception:
            pass
    if decision not in {"apply_available", "confirmed_noop"}:
        return {
            "status": "needs_retry" if decision == "retry" else decision,
            "source_key": source_key,
            "proposal_sha256": proposal_sha256,
            "review": review,
            "recovered_artifact": recovered_artifact,
            "reused_review": reused_review,
            "created": [],
            "updated": [],
            "audit": audit_decision,
        }

    # Adoption artifact writers hold this same lease.  Keep authority stable
    # across the final semantic effect: either the exact page CAS batch or the
    # confirmed-noop disposition that permits the caller to retire the raw.
    from llm_wiki_mcp.page_mutation import decision_authority_lock

    with decision_authority_lock():
        current_authority, current_authority_error = _current_ingest_review_authority(
            reviewer=reviewer
        )
        authority_compare_error = (
            decision_authority.compare_semantic_authority(
                review_authority,
                current_authority,
                lane="ingest_reconciliation",
            )
            if current_authority_error is None
            else current_authority_error
        )
        if authority_compare_error is not None:
            return {
                "status": "needs_retry",
                "source_key": source_key,
                "proposal_sha256": proposal_sha256,
                "review": review,
                "summary": authority_compare_error,
                "recovered_artifact": recovered_artifact,
                "reused_review": reused_review,
                "created": [],
                "updated": [],
                "audit": audit_decision,
            }
        if proof_error := _ingest_review_authority_error(review, review_authority):
            return {
                "status": "needs_retry",
                "source_key": source_key,
                "proposal_sha256": proposal_sha256,
                "review": review,
                "summary": proof_error,
                "recovered_artifact": recovered_artifact,
                "reused_review": reused_review,
                "created": [],
                "updated": [],
                "audit": audit_decision,
            }
        if reused_review:
            # Re-read inside the authority lease to close the gap between the
            # earlier reuse check and the final effect.
            durable_artifact = _load_ingest_review_artifact(
                review_path,
                source_key=source_key,
                proposal_sha256=proposal_sha256,
            )
            if (
                durable_artifact is None
                or durable_artifact.get("authority") != review_authority
                or durable_artifact.get("review") != review
            ):
                return {
                    "status": "needs_retry",
                    "source_key": source_key,
                    "proposal_sha256": proposal_sha256,
                    "review": review,
                    "summary": "frontier review artifact changed before effect",
                    "recovered_artifact": recovered_artifact,
                    "reused_review": True,
                    "created": [],
                    "updated": [],
                    "audit": audit_decision,
                }
        else:
            _readback, artifact_error = _write_and_readback_ingest_review_artifact(
                review_path,
                source_key=source_key,
                proposal_sha256=proposal_sha256,
                review=review,
                authority=review_authority,
            )
            if artifact_error is not None:
                return {
                    "status": "needs_retry",
                    "source_key": source_key,
                    "proposal_sha256": proposal_sha256,
                    "review": review,
                    "summary": artifact_error,
                    "recovered_artifact": recovered_artifact,
                    "reused_review": False,
                    "created": [],
                    "updated": [],
                    "audit": audit_decision,
                }
        if decision == "confirmed_noop":
            created, updated = [], []
        else:
            created, updated = _apply_prepared_operations(planned, link_totals=totals)
    return {
        "status": decision,
        "source_key": source_key,
        "proposal_sha256": proposal_sha256,
        "review": review,
        "authority": review_authority,
        "recovered_artifact": recovered_artifact,
        "reused_review": reused_review,
        "created": created,
        "updated": updated,
        "audit": audit_decision,
        **(
            {"stale_review_replaced": stale_review_reason}
            if stale_review_reason is not None
            else {}
        ),
    }
