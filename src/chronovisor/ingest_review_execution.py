"""Standard and sharded ingest-review execution without page mutation."""

from __future__ import annotations

import json
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from chronovisor import decision_authority
from chronovisor.canonical_json import canonical_json_sha256_stringifying_strict
from chronovisor.ingest_review_plan import (
    IngestReviewBudgetExhausted,
    IngestReviewShard,
    IngestReviewShardPlan,
    IngestReviewShardPlanState,
)
from chronovisor.ingest_review_store import write_ingest_artifact
from chronovisor.ingest_schemas import (
    INGEST_REVIEW_SHARD_POLICY_VERSION,
    INGEST_REVIEW_SHARD_SCHEMA_VERSION,
)


def ingest_review_shard_manifest_path(
    artifact_root: Path, plan: IngestReviewShardPlan
) -> Path:
    identity = canonical_json_sha256_stringifying_strict(
        {
            "kind": "ingest_review_shard_manifest_artifact",
            "full_proposal_sha256": plan.full_proposal_sha256,
            "manifest_sha256": plan.manifest_sha256,
        }
    )
    return artifact_root / f"review-shard-manifest-{identity}.json"


def ingest_review_shard_review_identity(
    artifact_root: Path,
    plan: IngestReviewShardPlan,
    *,
    shard_index: int,
    shard: IngestReviewShard,
) -> tuple[str, Path]:
    identity = canonical_json_sha256_stringifying_strict(
        {
            "kind": "ingest_review_shard_verdict",
            "full_proposal_sha256": plan.full_proposal_sha256,
            "manifest_sha256": plan.manifest_sha256,
            "shard_index": shard_index,
            "shard_proposal_sha256": shard.proposal_sha256,
        }
    )
    return identity, artifact_root / f"review-shard-{identity}.review.json"


def ingest_review_shard_failure(failure_class: str, summary: str) -> dict[str, Any]:
    return {
        "decision": "retry",
        "summary": summary,
        "failed_operations_disposition": "retry_required",
        "tests_run": [],
        "risk": "No page mutation was authorized.",
        "notes": None,
        "frontier_failure": {
            "failure_class": failure_class,
            "rescue_status": "local_quarantined",
            "summary": summary,
            "human_required": False,
            "notify_user": False,
        },
    }


def ingest_review_shard_manifest_artifact_payload(
    plan: IngestReviewShardPlan,
    *,
    source_key: str,
) -> dict[str, Any]:
    return {
        "schema_version": INGEST_REVIEW_SHARD_SCHEMA_VERSION,
        "kind": "ingest_review_shard_manifest_artifact",
        "source_key": source_key,
        "full_proposal_sha256": plan.full_proposal_sha256,
        "manifest_sha256": plan.manifest_sha256,
        "manifest": plan.manifest,
    }


def stored_ingest_review_shard_manifest_error(
    path: Path,
    plan: IngestReviewShardPlan,
    *,
    source_key: str,
) -> str | None:
    if not path.exists():
        return "review shard manifest is missing"
    try:
        current = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return f"review shard manifest is unreadable: {type(exc).__name__}: {exc}"
    expected = ingest_review_shard_manifest_artifact_payload(
        plan, source_key=source_key
    )
    if current != expected:
        return "review shard manifest changed or failed exact recomputation"
    return None


def persist_ingest_review_shard_manifest(
    path: Path,
    plan: IngestReviewShardPlan,
    *,
    source_key: str,
) -> str | None:
    payload = ingest_review_shard_manifest_artifact_payload(
        plan, source_key=source_key
    )
    if path.exists():
        return stored_ingest_review_shard_manifest_error(
            path, plan, source_key=source_key
        )
    try:
        write_ingest_artifact(path, payload)
        readback = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return f"review shard manifest write/readback failed: {type(exc).__name__}: {exc}"
    if readback != payload:
        return "review shard manifest readback verification failed"
    return None


def ingest_review_shard_aggregate(
    plan: IngestReviewShardPlan,
    shard_reviews: list[dict[str, Any]],
) -> dict[str, Any]:
    tests_run = list(
        dict.fromkeys(
            test
            for review in shard_reviews
            for test in review.get("tests_run", [])
            if isinstance(test, str)
        )
    )
    return {
        "decision": "apply_available",
        "summary": (
            f"All {len(shard_reviews)} exact review shards received independent "
            "local-consensus approval."
        ),
        "failed_operations_disposition": "none",
        "tests_run": tests_run,
        "risk": None,
        "notes": None,
        "review_shard_proof": {
            "schema_version": INGEST_REVIEW_SHARD_SCHEMA_VERSION,
            "policy_version": INGEST_REVIEW_SHARD_POLICY_VERSION,
            "full_proposal_sha256": plan.full_proposal_sha256,
            "manifest": plan.manifest,
            "manifest_sha256": plan.manifest_sha256,
            "shard_reviews": [
                {
                    "shard_index": shard_index,
                    "proposal_sha256": shard.proposal_sha256,
                    "review": review,
                }
                for shard_index, (shard, review) in enumerate(
                    zip(plan.shards, shard_reviews, strict=True)
                )
            ],
        },
    }


@dataclass(frozen=True)
class IngestShardedReviewDeps:
    persist_manifest: Callable[..., str | None]
    inspect_plan_state: Callable[..., IngestReviewShardPlanState]
    review_identity: Callable[..., tuple[str, Path]]
    run_frontier_review: Callable[..., dict[str, Any]]
    normalize_review: Callable[..., dict[str, Any]]
    current_authority: Callable[..., tuple[dict[str, Any] | None, str | None]]
    authority_error: Callable[[dict[str, Any], dict[str, Any]], str | None]
    write_and_readback: Callable[..., tuple[dict[str, Any] | None, str | None]]
    authority_lock: Callable[[], AbstractContextManager[Any]]


def run_ingest_sharded_review(
    plan: IngestReviewShardPlan,
    *,
    source_key: str,
    reviewer: Callable[[dict[str, Any]], dict[str, Any]] | None,
    authority: dict[str, Any],
    deps: IngestShardedReviewDeps,
    frontier_budget: Any | None = None,
) -> dict[str, Any]:
    """Review exact shards and aggregate only a complete all-approval proof."""

    manifest_error = deps.persist_manifest(plan, source_key=source_key)
    if manifest_error is not None:
        return ingest_review_shard_failure(
            "ingest_review_shard_manifest_invalid", manifest_error
        )
    shard_state = deps.inspect_plan_state(
        plan, source_key=source_key, authority=authority
    )
    if shard_state.invalid_reason is not None:
        return ingest_review_shard_failure(
            "ingest_review_shard_artifact_invalid", shard_state.invalid_reason
        )

    shard_reviews: list[dict[str, Any]] = []
    for shard_index, shard in enumerate(plan.shards):
        shard_source_key, shard_review_path = deps.review_identity(
            plan, shard_index=shard_index, shard=shard
        )
        current_review = shard_state.reviews[shard_index]
        if isinstance(current_review, dict):
            review = current_review
        else:
            if frontier_budget is not None and not frontier_budget.consume():
                raise IngestReviewBudgetExhausted
            try:
                review = deps.run_frontier_review(shard.proposal, reviewer=reviewer)
            except Exception as exc:
                return ingest_review_shard_failure(
                    "local_consensus_failed",
                    "local consensus reviewer failed for ingest review shard "
                    f"{shard_index}: {type(exc).__name__}: {exc}",
                )
            review = deps.normalize_review(review, proposal=shard.proposal)
            if review.get("decision") == "apply_available":
                with deps.authority_lock():
                    current_authority, current_error = deps.current_authority(
                        reviewer=reviewer
                    )
                    compare_error = current_error or decision_authority.compare_semantic_authority(
                        authority,
                        current_authority,
                        lane="ingest_reconciliation",
                    )
                    if compare_error is not None:
                        return ingest_review_shard_failure(
                            "local_decision_authority_changed", compare_error
                        )
                    if proof_error := deps.authority_error(review, authority):
                        return ingest_review_shard_failure(
                            "local_consensus_proof_invalid", proof_error
                        )
                    _readback, artifact_error = deps.write_and_readback(
                        shard_review_path,
                        source_key=shard_source_key,
                        proposal_sha256=shard.proposal_sha256,
                        review=review,
                        authority=authority,
                        integrity=True,
                    )
                if artifact_error is not None:
                    return ingest_review_shard_failure(
                        "ingest_review_shard_artifact_invalid", artifact_error
                    )
        shard_reviews.append(review)

    repair_reviews = [
        review
        for review in shard_reviews
        if any(
            isinstance(review.get(field), list) and bool(review.get(field))
            for field in ("invalid_tags", "replacement_operations")
        )
    ]
    non_approvals = [
        review for review in shard_reviews if review.get("decision") != "apply_available"
    ]
    if not non_approvals:
        return ingest_review_shard_aggregate(plan, shard_reviews)
    if len(repair_reviews) == 1 and non_approvals == repair_reviews:
        return repair_reviews[0]
    if repair_reviews:
        return ingest_review_shard_failure(
            (
                "ingest_review_multiple_repairs_unsupported"
                if len(repair_reviews) > 1
                else "ingest_review_mixed_repair_unsupported"
            ),
            (
                "multiple ingest review shards requested repairs; no mutation was authorized"
                if len(repair_reviews) > 1
                else "one ingest review shard requested repair while another did not approve; "
                "no mutation was authorized"
            ),
        )
    if any(review.get("decision") == "confirmed_noop" for review in non_approvals):
        return ingest_review_shard_failure(
            "ingest_review_shard_nonapproval",
            "confirmed_noop on one operation shard cannot discard the complete raw",
        )
    return non_approvals[0]
