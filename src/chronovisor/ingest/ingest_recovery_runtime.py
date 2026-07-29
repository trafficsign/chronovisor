"""Model-free terminal and partial-shard ingest recovery."""

from __future__ import annotations

import hashlib
import json
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable

from chronovisor.decision import decision_authority
from chronovisor.core.jobs import JobStatus


def _runtime():
    from chronovisor.ingest import ingest

    return ingest


def _runtime_call(name: str):
    def call(*args: Any, **kwargs: Any) -> Any:
        return getattr(_runtime(), name)(*args, **kwargs)

    return call


_canonical_json_sha256 = _runtime_call("_canonical_json_sha256")
_prepared_from_review_payload = _runtime_call("_prepared_from_review_payload")
_current_ingest_review_authority = _runtime_call("_current_ingest_review_authority")
_historical_ingest_sharded_review_recovery_error = _runtime_call("_historical_ingest_sharded_review_recovery_error")
_ingest_artifact_paths = _runtime_call("_ingest_artifact_paths")
_ingest_review_authority_error = _runtime_call("_ingest_review_authority_error")
_ingest_review_authority_shape_error = _runtime_call("_ingest_review_authority_shape_error")
_ingest_sharded_review_reuse_error = _runtime_call("_ingest_sharded_review_reuse_error")
_ingest_source_key = _runtime_call("_ingest_source_key")
_load_ingest_review_artifact = _runtime_call("_load_ingest_review_artifact")
_normalize_ingest_frontier_review = _runtime_call("_normalize_ingest_frontier_review")
_prepared_plan_is_fully_applied = _runtime_call("_prepared_plan_is_fully_applied")
_prepared_plan_targets_reserved_system_page = _runtime_call("_prepared_plan_targets_reserved_system_page")
_ingest_review_shard_review_identity = _runtime_call("_ingest_review_shard_review_identity")
_stored_ingest_review_shard_manifest_error = _runtime_call("_stored_ingest_review_shard_manifest_error")
_build_ingest_review_shard_plan = _runtime_call("_build_ingest_review_shard_plan")
_consume_ingest_review_continuation_marker = _runtime_call("_consume_ingest_review_continuation_marker")
_has_sharded_ingest_review_artifact_family = _runtime_call("_has_sharded_ingest_review_artifact_family")
_ingest_review_continuation_marker_path = _runtime_call("_ingest_review_continuation_marker_path")
_ingest_review_repair_transition_path = _runtime_call("_ingest_review_repair_transition_path")
_ingest_review_shard_manifest_path = _runtime_call("_ingest_review_shard_manifest_path")
_ingest_review_shard_plan_from_sealed_manifest = _runtime_call("_ingest_review_shard_plan_from_sealed_manifest")
_ingest_review_stall_path = _runtime_call("_ingest_review_stall_path")
_load_ingest_review_continuation_marker = _runtime_call("_load_ingest_review_continuation_marker")
_matching_ingest_review_stall_error = _runtime_call("_matching_ingest_review_stall_error")
_persist_ingest_review_continuation_marker = _runtime_call("_persist_ingest_review_continuation_marker")
_persist_ingest_review_shard_manifest = _runtime_call("_persist_ingest_review_shard_manifest")
_now = _runtime_call("_now")
_refresh_ingest_derived_artifacts = _runtime_call("_refresh_ingest_derived_artifacts")

from chronovisor.ingest.ingest import (  # noqa: E402
    IngestApplyError,
    PreparedIngestOperation,
    _INGEST_FRONTIER_LEGACY_ARTIFACT_SCHEMA_VERSION,
    _IngestReviewShardContinuation,
)
from chronovisor.ingest.ingest_review_plan import (
    IngestReviewShardCapacityError,
    IngestReviewShardPlan as _IngestReviewShardPlan,
    IngestReviewShardPlanState as _IngestReviewShardPlanState,
)
from chronovisor.ingest.ingest_schemas import (
    INGEST_FRONTIER_ARTIFACT_SCHEMA_VERSION,
    INGEST_FRONTIER_REVIEW_ARTIFACT_SCHEMA_VERSION,
)


def load_strict_ingest_proposal_for_recovery(
    path: Path,
    *,
    source_key: str,
    raw_content: str,
    raw_keywords: list[str] | None,
) -> tuple[dict[str, Any], list[PreparedIngestOperation]]:
    """Load one versioned terminal proposal without consulting model output.

    The ordinary retry path may replace an incomplete proposal after another
    bounded local attempt.  Pre-triage completion recovery is more privileged:
    it can retire a raw without asking a model again, so every artifact field
    that binds the raw and exact page postimages must already be intact.
    """

    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise IngestApplyError(
            "pre-triage terminal proposal artifact is unreadable: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    expected_top_level = {
        "schema_version",
        "kind",
        "source_key",
        "proposal_sha256",
        "proposal",
    }
    if not isinstance(artifact, dict) or set(artifact) != expected_top_level:
        raise IngestApplyError(
            "pre-triage terminal proposal artifact schema is invalid"
        )
    proposal = artifact.get("proposal")
    proposal_sha256 = (
        _canonical_json_sha256(proposal) if isinstance(proposal, dict) else None
    )
    artifact_version = artifact.get("schema_version")
    expected_raw_sha256 = hashlib.sha256(raw_content.encode("utf-8")).hexdigest()
    if (
        not isinstance(artifact_version, int)
        or isinstance(artifact_version, bool)
        or artifact_version
        not in {
            _INGEST_FRONTIER_LEGACY_ARTIFACT_SCHEMA_VERSION,
            INGEST_FRONTIER_ARTIFACT_SCHEMA_VERSION,
        }
        or artifact.get("kind") != "ingest_frontier_proposal_artifact"
        or artifact.get("source_key") != source_key
        or artifact.get("proposal_sha256") != proposal_sha256
        or not isinstance(proposal, dict)
        or proposal.get("schema_version") != artifact_version
        or proposal.get("kind") != "ingest_semantic_mutation_proposal"
        or proposal.get("source_key") != source_key
        or proposal.get("raw_content") != raw_content
        or proposal.get("raw_sha256") != expected_raw_sha256
        or proposal.get("raw_keywords") != list(raw_keywords or [])
    ):
        raise IngestApplyError(
            "pre-triage terminal proposal artifact binding is invalid"
        )
    required_proposal_fields = {
        "schema_version",
        "kind",
        "source_key",
        "source_raw",
        "raw_content",
        "raw_sha256",
        "raw_keywords",
        "local_disposition",
        "triage_plan",
        "failed_operation_specs",
        "local_generated_operations",
        "prepared_operations",
        "link_reconciliation",
    }
    if not required_proposal_fields <= set(proposal) or set(proposal) - (
        required_proposal_fields | {"audit_decision"}
    ):
        raise IngestApplyError("pre-triage terminal proposal payload schema is invalid")
    if (
        proposal.get("source_raw") is not None
        and not isinstance(proposal.get("source_raw"), str)
    ) or not isinstance(proposal.get("local_disposition"), str):
        raise IngestApplyError("pre-triage terminal proposal metadata is invalid")
    for field in (
        "triage_plan",
        "failed_operation_specs",
        "local_generated_operations",
        "prepared_operations",
    ):
        if not isinstance(proposal.get(field), list):
            raise IngestApplyError(f"pre-triage terminal proposal {field} is invalid")
    if not isinstance(proposal.get("link_reconciliation"), dict) or (
        "audit_decision" in proposal
        and not isinstance(proposal.get("audit_decision"), dict)
    ):
        raise IngestApplyError("pre-triage terminal proposal audit metadata is invalid")
    planned = _prepared_from_review_payload(
        proposal.get("prepared_operations"),
        require_source_provenance=(
            artifact_version >= INGEST_FRONTIER_ARTIFACT_SCHEMA_VERSION
        ),
    )
    if planned is None:
        raise IngestApplyError(
            "pre-triage terminal proposal page postimages are invalid"
        )
    target_paths = [str(item.path.resolve(strict=False)) for item in planned]
    page_ids = [item.page_id for item in planned]
    if len(target_paths) != len(set(target_paths)) or len(page_ids) != len(
        set(page_ids)
    ):
        raise IngestApplyError(
            "pre-triage terminal proposal has duplicate page targets"
        )
    return proposal, planned


def load_pretriage_terminal_recovery(
    raw_content: str,
    raw_keywords: list[str] | None,
    *,
    reviewer: Callable[[dict[str, Any]], dict[str, Any]] | None,
) -> dict[str, Any] | None:
    """Return a model-free terminal recovery only when its proof is complete.

    A proposal without a current terminal review remains ordinary retry work.
    Legacy review artifacts are likewise left to the normal local-consensus
    path.  A malformed *current* artifact fails closed instead of being
    overwritten and silently treated as new work.
    """

    source_key = _ingest_source_key(raw_content, raw_keywords)
    proposal_path, review_path = _ingest_artifact_paths(source_key)
    if not proposal_path.exists():
        if review_path.exists():
            raise IngestApplyError(
                "pre-triage terminal review exists without its proposal artifact"
            )
        return None

    try:
        proposal_candidate = json.loads(proposal_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise IngestApplyError(
            "pre-triage terminal proposal artifact is unreadable: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(proposal_candidate, dict):
        raise IngestApplyError(
            "pre-triage terminal proposal artifact schema is invalid"
        )
    proposal_version = proposal_candidate.get("schema_version")
    if not review_path.exists():
        if proposal_version == _INGEST_FRONTIER_LEGACY_ARTIFACT_SCHEMA_VERSION:
            # A v1 proposal without a durable authority seal never authorized
            # an effect.  Its rows predate source-operation provenance, so the
            # ordinary path may safely replace it with a complete v2 proposal.
            return None
        _load_strict_ingest_proposal_for_recovery(
            proposal_path,
            source_key=source_key,
            raw_content=raw_content,
            raw_keywords=raw_keywords,
        )
        return None
    try:
        review_candidate = json.loads(review_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise IngestApplyError(
            "pre-triage terminal review artifact is unreadable: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(review_candidate, dict):
        raise IngestApplyError("pre-triage terminal review artifact is not an object")
    review_version = review_candidate.get("schema_version")
    if (
        isinstance(review_version, int)
        and not isinstance(review_version, bool)
        and review_version < INGEST_FRONTIER_REVIEW_ARTIFACT_SCHEMA_VERSION
    ):
        # Historical frontier-shaped verdicts have no local authority seal.
        # They are neither trusted nor treated as corruption; the normal local
        # path will replace them after a fresh adopted-consensus decision.
        if proposal_version == _INGEST_FRONTIER_LEGACY_ARTIFACT_SCHEMA_VERSION:
            return None
        _load_strict_ingest_proposal_for_recovery(
            proposal_path,
            source_key=source_key,
            raw_content=raw_content,
            raw_keywords=raw_keywords,
        )
        return None
    proposal, planned = _load_strict_ingest_proposal_for_recovery(
        proposal_path,
        source_key=source_key,
        raw_content=raw_content,
        raw_keywords=raw_keywords,
    )
    if _prepared_plan_targets_reserved_system_page(planned):
        # This proposal was created before normal ingest learned to repair
        # reserved targets in-session.  It is never a valid terminal proof;
        # leave its raw pending and let the ordinary path overwrite it after a
        # fresh, bounded triage repair.
        return None
    expected_review_fields = {
        "schema_version",
        "kind",
        "source_key",
        "proposal_sha256",
        "review",
        "authority",
    }
    if set(review_candidate) != expected_review_fields:
        raise IngestApplyError("pre-triage terminal review artifact schema is invalid")
    proposal_sha256 = _canonical_json_sha256(proposal)
    review_artifact = _load_ingest_review_artifact(
        review_path,
        source_key=source_key,
        proposal_sha256=proposal_sha256,
    )
    if review_artifact is None or review_artifact != review_candidate:
        raise IngestApplyError("pre-triage terminal review artifact binding is invalid")
    review = review_artifact.get("review")
    authority = review_artifact.get("authority")
    if not isinstance(review, dict) or not isinstance(authority, dict):
        raise IngestApplyError("pre-triage terminal review proof is missing")
    normalized_review = _normalize_ingest_frontier_review(review, proposal=proposal)
    if normalized_review != review:
        raise IngestApplyError(
            "pre-triage terminal review is not canonical for its proposal"
        )
    authority_shape_error = _ingest_review_authority_shape_error(authority)
    authority_proof_error = _ingest_review_authority_error(review, authority)
    if authority_shape_error is not None or authority_proof_error is not None:
        raise IngestApplyError(
            "pre-triage terminal review authority is invalid: "
            + (authority_shape_error or authority_proof_error or "unknown error")
        )
    decision = review.get("decision")
    fully_applied = (
        decision == "apply_available"
        and bool(planned)
        and _prepared_plan_is_fully_applied(planned)
    )
    if "review_shard_proof" in review:
        shard_reuse_error = (
            _historical_ingest_sharded_review_recovery_error(
                review,
                proposal,
                authority,
            )
            if fully_applied
            else _ingest_sharded_review_reuse_error(
                review,
                proposal,
                authority,
            )
        )
        if shard_reuse_error is not None:
            raise IngestApplyError(
                "pre-triage terminal shard review binding is invalid: "
                + shard_reuse_error
            )
    if decision not in {"apply_available", "confirmed_noop"}:
        return None
    if decision == "apply_available":
        if not fully_applied:
            return None
    else:
        current_authority, current_authority_error = _current_ingest_review_authority(
            reviewer=reviewer
        )
        if current_authority_error is not None or (
            decision_authority.compare_semantic_authority(
                authority,
                current_authority,
                lane="ingest_reconciliation",
            )
            is not None
        ):
            return None

    created = [item.page_id for item in planned if item.op_type == "create"]
    updated = [item.page_id for item in planned if item.op_type == "update"]
    audit = proposal.get("audit_decision")
    failed_specs = proposal.get("failed_operation_specs")
    return {
        "status": decision,
        "source_key": source_key,
        "proposal_sha256": proposal_sha256,
        "review": review,
        "authority": authority,
        "created": created,
        "updated": updated,
        "audit": dict(audit) if isinstance(audit, dict) else {},
        "failed_operation_specs": (
            list(failed_specs) if isinstance(failed_specs, list) else []
        ),
        "recovered_artifact": True,
        "reused_review": True,
        "recovery_basis": (
            "exact_postimages_already_applied"
            if decision == "apply_available"
            else "durable_confirmed_noop"
        ),
    }


def inspect_ingest_review_shard_plan_state(
    plan: _IngestReviewShardPlan,
    *,
    source_key: str,
    authority: dict[str, Any],
) -> _IngestReviewShardPlanState:
    """Classify every durable shard against one exact current authority.

    A well-formed verdict from an older authority epoch is retained as audit
    evidence but contributes no approval to the current epoch.  Invalid,
    non-terminal, or integrity-mismatched artifacts remain fail-closed.
    """

    manifest_error = _stored_ingest_review_shard_manifest_error(
        plan,
        source_key=source_key,
    )
    if manifest_error is not None:
        return _IngestReviewShardPlanState(
            statuses=(),
            reviews=(),
            authorities=(),
            invalid_reason=manifest_error,
        )
    statuses: list[str] = []
    reviews: list[dict[str, Any] | None] = []
    authorities: list[dict[str, Any] | None] = []
    for shard_index, shard in enumerate(plan.shards):
        shard_source_key, shard_review_path = _ingest_review_shard_review_identity(
            plan,
            shard_index=shard_index,
            shard=shard,
        )
        if not shard_review_path.exists():
            statuses.append("missing")
            reviews.append(None)
            authorities.append(None)
            continue
        shard_artifact = _load_ingest_review_artifact(
            shard_review_path,
            source_key=shard_source_key,
            proposal_sha256=shard.proposal_sha256,
            require_integrity=True,
        )
        shard_review = (
            shard_artifact.get("review") if isinstance(shard_artifact, dict) else None
        )
        artifact_authority = (
            shard_artifact.get("authority")
            if isinstance(shard_artifact, dict)
            else None
        )
        if (
            shard_artifact is None
            or not isinstance(artifact_authority, dict)
            or not isinstance(shard_review, dict)
            or _normalize_ingest_frontier_review(
                shard_review,
                proposal=shard.proposal,
            )
            != shard_review
            or shard_review.get("decision") != "apply_available"
            or _ingest_review_authority_error(
                shard_review,
                artifact_authority,
            )
            is not None
        ):
            return _IngestReviewShardPlanState(
                statuses=tuple(statuses + ["invalid"]),
                reviews=tuple(reviews + [None]),
                authorities=tuple(authorities + [None]),
                invalid_reason=(
                    f"review shard {shard_index} durable verdict is invalid"
                ),
            )
        if artifact_authority == authority:
            statuses.append("current_approved")
            reviews.append(shard_review)
            authorities.append(artifact_authority)
        else:
            statuses.append("stale_valid")
            reviews.append(None)
            authorities.append(artifact_authority)
    return _IngestReviewShardPlanState(
        statuses=tuple(statuses),
        reviews=tuple(reviews),
        authorities=tuple(authorities),
    )


def load_pretriage_ingest_shard_continuation(
    raw_content: str,
    raw_keywords: list[str] | None,
    *,
    reviewer: Callable[[dict[str, Any]], dict[str, Any]] | None,
) -> _IngestReviewShardContinuation | None:
    """Load durable partial shard progress without rerunning semantic generation.

    A continuation is intentionally narrower than terminal recovery: the full
    proposal is still unapplied and no parent verdict exists yet.  Durable
    approvals from the current authority are reused; self-consistent approvals
    from an older epoch are treated as unapproved and reviewed again.
    """

    source_key = _ingest_source_key(raw_content, raw_keywords)
    proposal_path, review_path = _ingest_artifact_paths(source_key)
    if not proposal_path.exists():
        return None
    proposal, planned = _load_strict_ingest_proposal_for_recovery(
        proposal_path,
        source_key=source_key,
        raw_content=raw_content,
        raw_keywords=raw_keywords,
    )
    if (
        proposal.get("schema_version")
        == _INGEST_FRONTIER_LEGACY_ARTIFACT_SCHEMA_VERSION
    ):
        # Schema-v1 proposals predate source-operation provenance and therefore
        # cannot be projected into a current review shard.  Once their complete
        # envelope has passed the strict raw/source/digest checks above, an
        # *unapproved* v1 proposal (including one with only a legacy unsealed
        # review) has authorized no effect and may be replaced by the ordinary
        # v2 generation path.  The stable source-key path makes this a one-way
        # transition: after the v2 artifact is durably written, this legacy
        # branch cannot be selected again.
        #
        # Do not use that migration escape hatch when any continuation-family
        # evidence exists.  Such state would mean the artifact is not merely an
        # old unreviewed proposal; overwriting it could discard a partial proof.
        direct_continuation_paths = tuple(
            path
            for path in (
                _ingest_review_continuation_marker_path(source_key),
                _ingest_review_repair_transition_path(source_key),
                _ingest_review_stall_path(source_key),
            )
            if path.exists()
        )
        if direct_continuation_paths or _has_sharded_ingest_review_artifact_family(
            review_path,
            proposal=proposal,
            source_key=source_key,
        ):
            raise IngestApplyError(
                "pre-triage legacy proposal has partial continuation evidence; "
                "refusing regeneration"
            )
        return None
    if review_path.exists():
        return None
    if _prepared_plan_targets_reserved_system_page(planned):
        return None
    if proposal.get("failed_operation_specs"):
        # Partial generation is intentionally regenerated from the saved
        # triage plan; it cannot be converted into an exact shard continuation.
        return None
    authority, authority_error = _current_ingest_review_authority(reviewer=reviewer)
    authority_shape_error = (
        _ingest_review_authority_shape_error(authority)
        if authority is not None
        else None
    )
    if authority_error is not None or authority is None or authority_shape_error:
        raise IngestApplyError(
            "pre-triage shard continuation authority is unavailable: "
            + (authority_error or authority_shape_error or "authority is missing")
        )

    try:
        plan = _build_ingest_review_shard_plan(
            proposal,
            force_review_unit=True,
        )
    except (IngestReviewShardCapacityError, TypeError, ValueError) as exc:
        raise IngestApplyError(
            f"pre-triage shard continuation recomputation failed: {exc}"
        ) from exc
    assert plan is not None
    manifest_path = _ingest_review_shard_manifest_path(plan)
    prior_authority: dict[str, Any] | None = None
    prior_verified_progress = False
    if not manifest_path.exists():
        prior_plan_found = False
        proposal_sha256 = _canonical_json_sha256(proposal)
        for prior_path in sorted(
            proposal_path.parent.glob("review-shard-manifest-*.json")
        ):
            try:
                prior_artifact = json.loads(prior_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                continue
            if (
                not isinstance(prior_artifact, dict)
                or prior_artifact.get("source_key") != source_key
                or prior_artifact.get("full_proposal_sha256") != proposal_sha256
            ):
                continue
            prior_manifest = prior_artifact.get("manifest")
            try:
                prior_plan = _ingest_review_shard_plan_from_sealed_manifest(
                    proposal,
                    prior_manifest,
                )
            except (IngestReviewShardCapacityError, TypeError, ValueError) as exc:
                raise IngestApplyError(
                    "pre-triage shard continuation prior manifest is invalid: "
                    + str(exc)
                ) from exc
            if prior_path != _ingest_review_shard_manifest_path(prior_plan):
                raise IngestApplyError(
                    "pre-triage shard continuation prior manifest filename is invalid"
                )
            prior_state = _inspect_ingest_review_shard_plan_state(
                prior_plan,
                source_key=source_key,
                authority=authority,
            )
            if prior_state.invalid_reason is not None:
                raise IngestApplyError(
                    "pre-triage shard continuation prior proof is invalid: "
                    + prior_state.invalid_reason
                )
            prior_authority = next(
                (
                    candidate
                    for candidate in prior_state.authorities
                    if isinstance(candidate, dict) and candidate != authority
                ),
                prior_authority,
            )
            prior_verified_progress = prior_verified_progress or bool(
                prior_state.current_approved_indices
                or "stale_valid" in prior_state.statuses
            )
            prior_plan_found = True
        if not prior_plan_found:
            # A standard proposal with only a semantic retry is not a bounded
            # continuation.  Let the ordinary convergence loop regenerate it.
            return None
        manifest_error = _persist_ingest_review_shard_manifest(
            plan,
            source_key=source_key,
        )
        if manifest_error is not None:
            raise IngestApplyError(
                "pre-triage shard continuation manifest reseed failed: "
                + manifest_error
            )
    shard_state = _inspect_ingest_review_shard_plan_state(
        plan,
        source_key=source_key,
        authority=authority,
    )
    if shard_state.invalid_reason is not None:
        raise IngestApplyError(
            "pre-triage shard continuation durable approval is invalid: "
            + shard_state.invalid_reason
        )
    prior_authority = next(
        (
            candidate
            for candidate in shard_state.authorities
            if isinstance(candidate, dict) and candidate != authority
        ),
        prior_authority,
    )
    if stall_error := _matching_ingest_review_stall_error(
        source_key=source_key,
        plan=plan,
        authority=authority,
        approved_indices=shard_state.current_approved_indices,
    ):
        raise IngestApplyError(
            "pre-triage shard continuation is stalled: " + stall_error
        )
    if shard_state.approved_shards == 0:
        marker, marker_error = _load_ingest_review_continuation_marker(
            source_key=source_key,
            plan=plan,
            authority=authority,
        )
        if marker_error is not None:
            raise IngestApplyError(
                "pre-triage zero-approval continuation marker is invalid: "
                + marker_error
            )
        stale_marker: dict[str, Any] | None = None
        if marker is None:
            stale_marker, stale_marker_error = _load_ingest_review_continuation_marker(
                source_key=source_key,
                plan=plan,
                authority=authority,
                allow_stale_identity=True,
            )
            if stale_marker_error is not None:
                raise IngestApplyError(
                    "pre-triage stale continuation marker is invalid: "
                    + stale_marker_error
                )
        if marker is None and (
            prior_authority is not None
            or prior_verified_progress
            or stale_marker is not None
        ):
            stale_authority_sha256 = (
                str(stale_marker.get("current_authority_sha256"))
                if isinstance(stale_marker, dict)
                else None
            )
            marker_reason = (
                "authority_epoch_reseed"
                if prior_authority is not None
                or (
                    stale_authority_sha256 is not None
                    and stale_authority_sha256 != _canonical_json_sha256(authority)
                )
                else "router_config_reseed"
            )
            marker_write_error = _persist_ingest_review_continuation_marker(
                source_key=source_key,
                plan=plan,
                reason=marker_reason,
                previous_full_proposal_sha256=plan.full_proposal_sha256,
                previous_authority=(
                    prior_authority or stale_authority_sha256 or authority
                ),
                current_authority=authority,
            )
            if marker_write_error is not None:
                raise IngestApplyError(
                    "pre-triage authority reseed marker failed: " + marker_write_error
                )
            marker, marker_error = _load_ingest_review_continuation_marker(
                source_key=source_key,
                plan=plan,
                authority=authority,
            )
            if marker_error is not None or marker is None:
                raise IngestApplyError(
                    "pre-triage authority reseed marker readback failed: "
                    + (marker_error or "marker is missing")
                )
        if marker is None:
            return None
        if marker.get("state") == "available":
            if consume_error := _consume_ingest_review_continuation_marker(
                source_key,
                marker,
            ):
                raise IngestApplyError(
                    "pre-triage zero-approval continuation marker was not claimed: "
                    + consume_error
                )
    return _IngestReviewShardContinuation(
        proposal=proposal,
        planned=tuple(planned),
        plan=plan,
        approved_shards=shard_state.approved_shards,
    )


# Preserve the original intra-module spellings in the mechanically moved code.
_load_strict_ingest_proposal_for_recovery = load_strict_ingest_proposal_for_recovery
_inspect_ingest_review_shard_plan_state = inspect_ingest_review_shard_plan_state


def complete_pretriage_terminal_recovery(
    recovery: dict[str, Any],
    *,
    raw_content: str,
    raw_keywords: list[str] | None,
    source_raw: str | None,
    job_id: str,
    on_complete: Callable[[], Any] | None,
    reviewer: Callable[[dict[str, Any]], dict[str, Any]] | None,
) -> dict[str, Any]:
    """Finish a proven prior effect and publish its raw ACK under locks."""

    changed_pages = list(recovery.get("created") or []) + list(
        recovery.get("updated") or []
    )

    from chronovisor.ingest.page_mutation import (
        decision_authority_lock,
        chronovisor_mutation_lock,
    )

    # Match the normal effect lock order.  The authority lease prevents a
    # terminal review artifact from being replaced while the page lease keeps
    # exact postimages stable through job completion and ACK publication.
    with decision_authority_lock():
        page_lease = (
            chronovisor_mutation_lock()
            if recovery.get("status") == "apply_available"
            else nullcontext()
        )
        with page_lease:
            verified = _runtime()._load_pretriage_terminal_recovery(
                raw_content,
                raw_keywords,
                reviewer=reviewer,
            )
            if verified is None or verified != recovery:
                raise IngestApplyError(
                    "pre-triage terminal recovery proof changed before raw retirement"
                )
            # Derived refresh is intentionally inside the same proof/effect
            # locks. A stale or concurrently replaced terminal artifact must
            # fail before even rebuildable claims/index side effects occur.
            read_back_result = _refresh_ingest_derived_artifacts(
                changed_pages,
                source_raw=source_raw,
            )
            final_verified = _runtime()._load_pretriage_terminal_recovery(
                raw_content,
                raw_keywords,
                reviewer=reviewer,
            )
            if final_verified is None or final_verified != recovery:
                raise IngestApplyError(
                    "pre-triage terminal recovery proof changed during derived refresh"
                )
            job_result: dict[str, Any] = {
                "frontier": {
                    "status": recovery.get("status"),
                    "proposal_sha256": recovery.get("proposal_sha256"),
                    "source_key": recovery.get("source_key"),
                    "review": recovery.get("review"),
                    "recovered_artifact": True,
                    "reused_review": True,
                },
                "audit": recovery.get("audit"),
                "pretriage_recovery": {
                    "basis": recovery.get("recovery_basis"),
                    "model_calls": 0,
                },
            }
            failed_specs = list(recovery.get("failed_operation_specs") or [])
            if failed_specs:
                job_result.update({"partial": True, "failed_ops": failed_specs})
            if read_back_result.get("failed"):
                job_result["read_back"] = read_back_result
            _runtime().job_store.update(
                job_id,
                status=JobStatus.COMPLETED,
                processor="durable-ingest-recovery",
                stage="pre-triage-recovery",
                completed_at=_now(),
                pages_created=list(recovery.get("created") or []),
                pages_updated=list(recovery.get("updated") or []),
                result=job_result,
            )
            if on_complete:
                # For orchestrated raws this publishes the content-bound ACK
                # before the processed-state transition.  Any callback error
                # escapes and the outer job boundary records FAILED.
                on_complete()
    return job_result
