"""Authority-envelope and shard-proof validation for ingest review."""

from __future__ import annotations

import re
from typing import Any

from chronovisor.core.canonical_json import canonical_json_sha256_stringifying_strict
from chronovisor.decision import decision_authority
from chronovisor.ingest.ingest_schemas import (
    INGEST_REVIEW_LIMIT_FIELDS,
    INGEST_REVIEW_SHARD_POLICY_VERSION,
    INGEST_REVIEW_SHARD_ROW_FIELDS,
    INGEST_REVIEW_SHARD_SCHEMA_VERSION,
)


def current_ingest_review_authority(
    *, injected_reviewer: bool
) -> tuple[dict[str, Any] | None, str | None]:
    """Resolve the exact enabled local authority allowed to affect ingest."""

    return decision_authority.current_semantic_authority(
        "ingest_reconciliation",
        injected_reviewer=injected_reviewer,
    )


def ingest_review_authority_error(
    review: dict[str, Any],
    authority: dict[str, Any],
) -> str | None:
    """Cross-check the verdict and trusted local quorum with its authority."""

    if review.get("review_shard_proof") is not None:
        return ingest_review_shard_proof_error(review, authority)
    if authority.get("source") == "injected_reviewer_boundary":
        return None
    return decision_authority.semantic_verdict_authority_error(
        review,
        authority,
        lane="ingest_reconciliation",
    )


def ingest_review_shard_proof_error(
    review: dict[str, Any],
    authority: dict[str, Any],
) -> str | None:
    """Validate every quorum proof in one host-aggregated shard verdict."""

    if set(review) != {
        "decision",
        "summary",
        "failed_operations_disposition",
        "tests_run",
        "risk",
        "notes",
        "review_shard_proof",
    }:
        return "ingest shard aggregate review schema is invalid"
    if (
        review.get("decision") != "apply_available"
        or review.get("failed_operations_disposition") != "none"
        or not isinstance(review.get("summary"), str)
        or not str(review.get("summary")).strip()
        or not isinstance(review.get("tests_run"), list)
    ):
        return "ingest shard aggregate disposition is invalid"
    proof = review.get("review_shard_proof")
    if not isinstance(proof, dict) or set(proof) != {
        "schema_version",
        "policy_version",
        "full_proposal_sha256",
        "manifest",
        "manifest_sha256",
        "shard_reviews",
    }:
        return "ingest shard aggregate proof schema is invalid"
    manifest = proof.get("manifest")
    manifest_sha256 = proof.get("manifest_sha256")
    if (
        proof.get("schema_version") != INGEST_REVIEW_SHARD_SCHEMA_VERSION
        or proof.get("policy_version") != INGEST_REVIEW_SHARD_POLICY_VERSION
        or not isinstance(manifest, dict)
        or not isinstance(manifest_sha256, str)
        or canonical_json_sha256_stringifying_strict(manifest) != manifest_sha256
        or proof.get("full_proposal_sha256") != manifest.get("full_proposal_sha256")
    ):
        return "ingest shard aggregate manifest binding is invalid"
    manifest_rows = manifest.get("shards")
    shard_reviews = proof.get("shard_reviews")
    operation_count = manifest.get("full_operation_count")
    review_limits = manifest.get("review_limits")
    if (
        set(manifest)
        != {
            "schema_version",
            "policy_version",
            "kind",
            "source_key",
            "full_proposal_sha256",
            "full_operation_count",
            "review_limits",
            "shards",
        }
        or manifest.get("schema_version") != INGEST_REVIEW_SHARD_SCHEMA_VERSION
        or manifest.get("policy_version") != INGEST_REVIEW_SHARD_POLICY_VERSION
        or manifest.get("kind") != "ingest_review_shard_manifest"
        or not isinstance(manifest.get("source_key"), str)
        or not re.fullmatch(r"[0-9a-f]{64}", str(manifest.get("source_key")))
        or not isinstance(manifest.get("full_proposal_sha256"), str)
        or not re.fullmatch(
            r"[0-9a-f]{64}", str(manifest.get("full_proposal_sha256"))
        )
        or not isinstance(operation_count, int)
        or isinstance(operation_count, bool)
        or operation_count < 1
        or not isinstance(review_limits, dict)
        or set(review_limits) != INGEST_REVIEW_LIMIT_FIELDS
        or not all(
            isinstance(value, int) and not isinstance(value, bool) and value > 0
            for value in review_limits.values()
        )
        or review_limits.get("min_num_ctx", 0) > review_limits.get("num_ctx", 0)
        or not isinstance(manifest_rows, list)
        or not manifest_rows
        or not isinstance(shard_reviews, list)
        or len(shard_reviews) != len(manifest_rows)
    ):
        return "ingest shard aggregate manifest is invalid"
    flattened: list[int] = []
    for expected_index, (manifest_row, review_row) in enumerate(
        zip(manifest_rows, shard_reviews, strict=True)
    ):
        if (
            not isinstance(manifest_row, dict)
            or set(manifest_row) != INGEST_REVIEW_SHARD_ROW_FIELDS
            or manifest_row.get("shard_index") != expected_index
            or not isinstance(manifest_row.get("proposal_sha256"), str)
            or not re.fullmatch(
                r"[0-9a-f]{64}", str(manifest_row.get("proposal_sha256"))
            )
            or not isinstance(manifest_row.get("effective_request_sha256"), str)
            or not re.fullmatch(
                r"[0-9a-f]{64}",
                str(manifest_row.get("effective_request_sha256")),
            )
            or not isinstance(manifest_row.get("original_operation_indices"), list)
            or not manifest_row.get("original_operation_indices")
            or not all(
                isinstance(manifest_row.get(field), int)
                and not isinstance(manifest_row.get(field), bool)
                and manifest_row.get(field) > 0
                for field in (
                    "effective_input_chars",
                    "effective_input_bytes",
                    "required_num_ctx",
                    "selected_num_ctx",
                )
            )
            or manifest_row.get("effective_input_bytes")
            > review_limits.get("max_input_chars", 0)
            or manifest_row.get("required_num_ctx")
            > manifest_row.get("selected_num_ctx")
            or manifest_row.get("selected_num_ctx")
            > review_limits.get("num_ctx", 0)
            or manifest_row.get("selected_num_ctx")
            < review_limits.get("min_num_ctx", 0)
            or not isinstance(review_row, dict)
            or set(review_row) != {"shard_index", "proposal_sha256", "review"}
            or review_row.get("shard_index") != expected_index
            or review_row.get("proposal_sha256")
            != manifest_row.get("proposal_sha256")
            or not isinstance(review_row.get("review"), dict)
        ):
            return "ingest shard aggregate review binding is invalid"
        indices = manifest_row["original_operation_indices"]
        if not all(
            isinstance(index, int) and not isinstance(index, bool) and index >= 0
            for index in indices
        ):
            return "ingest shard aggregate operation indices are invalid"
        flattened.extend(indices)
        shard_review = review_row["review"]
        if (
            shard_review.get("decision") != "apply_available"
            or shard_review.get("failed_operations_disposition") != "none"
            or any(
                shard_review.get(field)
                for field in ("invalid_tags", "replacement_operations")
            )
        ):
            return "ingest shard aggregate contains a non-approval"
        if authority.get("source") != "injected_reviewer_boundary":
            proof_error = decision_authority.semantic_verdict_authority_error(
                shard_review,
                authority,
                lane="ingest_reconciliation",
            )
            if proof_error is not None:
                return proof_error
    if flattened != list(range(operation_count)) or len(flattened) != len(
        set(flattened)
    ):
        return "ingest shard aggregate coverage is not exact and non-overlapping"
    return None


def ingest_review_authority_shape_error(authority: dict[str, Any]) -> str | None:
    """Reject authority envelopes that cannot identify a review epoch."""

    return decision_authority.semantic_authority_shape_error(
        authority,
        lane="ingest_reconciliation",
    )
