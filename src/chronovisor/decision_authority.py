"""Durable authority seals for local semantic verdict artifacts.

A semantic verdict is reusable only while the exact authority that produced it
is still current: the lane must remain enabled, the lane/case contracts must be
unchanged, and the same evaluated model triplet must still be adopted.  This
module keeps that epoch identity uniform across mutation-capable lanes.
"""

from __future__ import annotations

import copy
import hashlib
import re
from collections.abc import Mapping
from typing import Any


AUTHORITY_VERSION = 1
INJECTED_REVIEWER_SOURCE = "injected_reviewer_boundary"
ADOPTED_LOCAL_SOURCE = "adopted_local_consensus"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def current_semantic_authority(
    lane: str,
    *,
    injected_reviewer: bool = False,
) -> tuple[dict[str, Any] | None, str | None]:
    """Return the exact authority currently allowed to produce lane effects.

    ``injected_reviewer`` is an explicit test/integration boundary.  It must be
    selected by the caller from dependency injection, never inferred from a
    production model response.
    """

    if not isinstance(lane, str) or not lane.strip():
        return None, "decision_authority_lane_required"
    lane = lane.strip()
    if injected_reviewer:
        return {
            "source": INJECTED_REVIEWER_SOURCE,
            "authority_version": AUTHORITY_VERSION,
            "lane": lane,
        }, None

    try:
        from chronovisor.decision_lane_contract_cases import (
            decision_lane_contract_case_manifest_sha256,
        )
        from chronovisor.decision_lane_contracts import (
            lane_contract_manifest_sha256,
            lane_contract_sha256,
        )
        from chronovisor.decision_policy import resolve_decision_policy
        from chronovisor.decision_router import resolve_router_policy
        from chronovisor.runtime_config import load_decision_router_config

        policy, mode, policy_error = resolve_decision_policy(lane)
        if policy is None or policy_error is not None or mode != "enabled":
            return None, policy_error or f"decision_lane_not_enabled:{lane}:{mode}"
        resolution = resolve_router_policy(load_decision_router_config())
        authority = {
            "source": ADOPTED_LOCAL_SOURCE,
            "authority_version": AUTHORITY_VERSION,
            "lane": lane,
            "lane_contract_sha256": lane_contract_sha256(lane),
            "lane_contract_manifest_sha256": lane_contract_manifest_sha256(),
            "lane_contract_case_manifest_sha256": (
                decision_lane_contract_case_manifest_sha256()
            ),
            "policy": {
                "kind": policy.kind if policy is not None else None,
                "schema_name": policy.schema_name if policy is not None else None,
                "mode": mode,
                "error": policy_error,
            },
            # The router audit binds the evaluated artifact digest and exact
            # primary/challenger/tie-break model triplet.
            "router": resolution.audit_record(),
        }
    except Exception as exc:
        return None, f"decision_authority_unavailable:{exc.__class__.__name__}:{exc}"

    if (
        resolution.source != "adopted_artifact"
        or resolution.error is not None
        or not resolution.artifact_sha256
    ):
        return None, resolution.error or f"decision_adoption_not_valid:{lane}"
    return authority, None


def semantic_authority_shape_error(
    authority: object,
    *,
    lane: str,
) -> str | None:
    """Validate that an authority seal fully identifies one decision epoch."""

    if not isinstance(authority, Mapping):
        return "decision authority seal is missing"
    if (
        authority.get("authority_version") != AUTHORITY_VERSION
        or authority.get("lane") != lane
    ):
        return "decision authority identity is invalid"
    source = authority.get("source")
    if source == INJECTED_REVIEWER_SOURCE:
        if set(authority) != {"source", "authority_version", "lane"}:
            return "injected decision authority identity is invalid"
        return None
    if source != ADOPTED_LOCAL_SOURCE:
        return "decision authority source is invalid"
    for field in (
        "lane_contract_sha256",
        "lane_contract_manifest_sha256",
        "lane_contract_case_manifest_sha256",
    ):
        value = authority.get(field)
        if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
            return f"decision authority {field} is invalid"
    policy = authority.get("policy")
    if (
        not isinstance(policy, Mapping)
        or set(policy) != {"kind", "schema_name", "mode", "error"}
        or policy.get("mode") != "enabled"
        or policy.get("error") is not None
        or not isinstance(policy.get("kind"), str)
        or not isinstance(policy.get("schema_name"), str)
        or not policy.get("schema_name")
    ):
        return "decision authority policy is invalid"
    router = authority.get("router")
    models = router.get("models") if isinstance(router, Mapping) else None
    if (
        not isinstance(router, Mapping)
        or set(router) != {"source", "artifact_sha256", "error", "models"}
        or router.get("source") != "adopted_artifact"
        or not isinstance(router.get("artifact_sha256"), str)
        or _SHA256_RE.fullmatch(router["artifact_sha256"]) is None
        or router.get("error") is not None
        or not isinstance(models, list)
        or len(models) != 3
        or not all(isinstance(model, str) and model for model in models)
    ):
        return "decision authority router identity is invalid"
    return None


def semantic_verdict_authority_provenance_error(
    review: object,
    authority: object,
    *,
    lane: str,
) -> str | None:
    """Cross-check only a verdict's embedded lane/router epoch provenance."""

    shape_error = semantic_authority_shape_error(authority, lane=lane)
    if shape_error is not None:
        return shape_error
    assert isinstance(authority, Mapping)
    if authority.get("source") == INJECTED_REVIEWER_SOURCE:
        return None
    if not isinstance(review, Mapping):
        return "decision verdict is missing"
    policy_audit = review.get("decision_policy")
    if not isinstance(policy_audit, Mapping):
        return "decision verdict authority audit is missing"
    observed_policy = {
        "kind": policy_audit.get("kind"),
        "schema_name": policy_audit.get("schema_name"),
        "mode": policy_audit.get("mode"),
        "error": policy_audit.get("error"),
    }
    if observed_policy != authority.get("policy"):
        return "decision verdict lane authority changed"
    if policy_audit.get("router_policy") != authority.get("router"):
        return "decision verdict router authority changed"
    return None


def semantic_verdict_authority_error(
    review: object,
    authority: object,
    *,
    lane: str,
) -> str | None:
    """Cross-check a verdict's provenance and successful quorum proof."""

    provenance_error = semantic_verdict_authority_provenance_error(
        review,
        authority,
        lane=lane,
    )
    if provenance_error is not None:
        return provenance_error
    assert isinstance(authority, Mapping)
    if authority.get("source") == INJECTED_REVIEWER_SOURCE:
        return None
    assert isinstance(review, Mapping)
    router = authority.get("router")
    router_models = router.get("models") if isinstance(router, Mapping) else None
    consensus_error = _local_consensus_proof_error(
        review.get("local_consensus"),
        router_models=router_models,
    )
    if consensus_error is not None:
        return consensus_error
    action_proof_error = _canonical_action_proof_error(
        review,
        authority=authority,
    )
    if action_proof_error is not None:
        return action_proof_error
    return None


def _canonical_action_proof_error(
    review: Mapping[str, Any],
    *,
    authority: Mapping[str, Any],
) -> str | None:
    """Bind the returned action payload to the router's quorum signature.

    The redacted consensus envelope proves that two adopted models agreed on a
    digest, but that proof is useful only if the durable verdict still hashes
    to the same digest.  Recompute it from the canonical production schema
    named by the sealed lane policy so copying a valid quorum envelope onto a
    different action can never authorize an effect.
    """

    policy = authority.get("policy")
    schema_name = policy.get("schema_name") if isinstance(policy, Mapping) else None
    if not isinstance(schema_name, str) or not schema_name:
        return "decision verdict canonical schema is missing"
    try:
        from chronovisor.decision_router import canonical_agreement_signature
        from chronovisor.decision_schema_manifest import production_decision_schemas

        schema = production_decision_schemas().get(schema_name)
    except Exception as exc:
        return f"decision verdict canonical schema unavailable:{exc.__class__.__name__}"
    if schema is None:
        return "decision verdict canonical schema is unknown"
    try:
        signature = canonical_agreement_signature(review, schema=schema)
    except Exception as exc:
        return f"decision verdict canonical action is invalid:{exc.__class__.__name__}"
    actual_sha256 = hashlib.sha256(signature.encode("utf-8")).hexdigest()
    consensus = review.get("local_consensus")
    agreement_sha256 = (
        consensus.get("agreement_sha256") if isinstance(consensus, Mapping) else None
    )
    if actual_sha256 != agreement_sha256:
        return "decision verdict action does not match local consensus agreement"
    return None


def _local_consensus_proof_error(
    consensus: object,
    *,
    router_models: object,
) -> str | None:
    """Validate the redacted :class:`DecisionRouterResult` quorum proof.

    The decision payload is model-authored, while ``local_consensus`` is added
    by the trusted router boundary.  Production verdict authority therefore
    requires that audit envelope to prove the returned value actually reached
    a two-model quorum; merely matching the adopted router policy is not
    sufficient.
    """

    if not isinstance(consensus, Mapping):
        return "decision verdict local consensus proof is missing"
    if (
        not isinstance(router_models, list)
        or len(router_models) != 3
        or not all(isinstance(model, str) and model for model in router_models)
    ):
        return "decision verdict router model triplet is invalid"
    agreement_sha256 = consensus.get("agreement_sha256")
    votes = consensus.get("votes")
    if (
        consensus.get("status") != "agreed"
        or consensus.get("ok") is not True
        or not isinstance(agreement_sha256, str)
        or _SHA256_RE.fullmatch(agreement_sha256) is None
        or consensus.get("failure_class") is not None
        or consensus.get("quarantine_reason") is not None
        or not isinstance(votes, list)
        or not 2 <= len(votes) <= 3
    ):
        return "decision verdict local consensus proof is invalid"

    expected_roles = ("primary", "challenger", "tie_break")
    agreeing_votes = 0
    for index, vote in enumerate(votes):
        if not isinstance(vote, Mapping):
            return "decision verdict local consensus vote is invalid"
        valid = vote.get("valid")
        signature_sha256 = vote.get("signature_sha256")
        model = vote.get("model")
        role = vote.get("role")
        if (
            not isinstance(valid, bool)
            or role != expected_roles[index]
            or model != router_models[index]
        ):
            return "decision verdict local consensus vote authority is invalid"
        if valid:
            if (
                not isinstance(signature_sha256, str)
                or _SHA256_RE.fullmatch(signature_sha256) is None
                or vote.get("invalid_reason") is not None
            ):
                return "decision verdict local consensus vote is invalid"
            if signature_sha256 == agreement_sha256:
                agreeing_votes += 1
        elif signature_sha256 is not None:
            return "decision verdict local consensus vote is invalid"

    if agreeing_votes < 2:
        return "decision verdict local consensus quorum is not proven"
    return None


def seal_semantic_artifact(
    payload: Mapping[str, Any],
    *,
    authority: Mapping[str, Any],
    lane: str,
) -> dict[str, Any]:
    """Return a detached artifact payload carrying a validated authority seal."""

    shape_error = semantic_authority_shape_error(authority, lane=lane)
    if shape_error is not None:
        raise ValueError(shape_error)
    if "authority" in payload:
        raise ValueError("semantic artifact payload already contains authority")
    sealed = copy.deepcopy(dict(payload))
    sealed["authority"] = copy.deepcopy(dict(authority))
    return sealed


def compare_semantic_authority(
    expected: object,
    current: object,
    *,
    lane: str,
) -> str | None:
    """Fail closed unless two complete authority seals name the same epoch."""

    expected_error = semantic_authority_shape_error(expected, lane=lane)
    if expected_error is not None:
        return expected_error
    current_error = semantic_authority_shape_error(current, lane=lane)
    if current_error is not None:
        return current_error
    if expected != current:
        return "decision authority changed before effect"
    return None


__all__ = [
    "ADOPTED_LOCAL_SOURCE",
    "AUTHORITY_VERSION",
    "INJECTED_REVIEWER_SOURCE",
    "compare_semantic_authority",
    "current_semantic_authority",
    "seal_semantic_artifact",
    "semantic_authority_shape_error",
    "semantic_verdict_authority_error",
    "semantic_verdict_authority_provenance_error",
]
