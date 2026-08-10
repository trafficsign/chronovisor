"""Durable authority seals for configured semantic verdict artifacts.

A semantic verdict is reusable only while the exact authority that produced it
is still current: the lane must remain enabled, the lane/case contracts must be
unchanged, and the same configured route triplet must still be authoritative.
This module keeps that epoch identity uniform across mutation-capable lanes.
"""

from __future__ import annotations

import copy
import hashlib
import math
import re
from collections.abc import Mapping
from typing import Any

from chronovisor.core import llm_runtime, provider_profiles, runtime_config

AUTHORITY_VERSION = 2
INJECTED_REVIEWER_SOURCE = "injected_reviewer_boundary"
RUNTIME_ROUTE_SOURCE = "configured_runtime_consensus"
# Compatibility name for callers that compare against the exported constant.
ADOPTED_LOCAL_SOURCE = RUNTIME_ROUTE_SOURCE
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_SAFE_DIAGNOSTIC_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/+\-]{0,127}\Z")
_CONSENSUS_AUDIT_FIELDS = {
    "status",
    "ok",
    "conservative_veto_fired",
    "conservative_veto_bypassed_by_lane_policy",
    "dissent_effect_class",
    "quorum_safety_policy_version",
    "agreement_sha256",
    "failure_class",
    "quarantine_reason",
    "num_ctx",
    "residency",
    "votes",
}
_VOTE_AUDIT_FIELDS = {
    "role",
    "provider",
    "model",
    "route_provenance",
    "returned_model",
    "requested_num_ctx",
    "valid",
    "signature_sha256",
    "invalid_reason",
    "decision_label",
    "effect_class",
    "runtime_observation",
    "session",
}
_SESSION_AUDIT_FIELDS = {
    "ok",
    "model",
    "failure_class",
    "returned_model",
    "first_pass_valid",
    "repair_turns",
    "attempts",
}
_ATTEMPT_AUDIT_FIELDS = {
    "index",
    "valid",
    "output_sha256",
    "output_chars",
    "normalized",
    "error_fingerprint",
    "issues",
}
_ISSUE_AUDIT_FIELDS = {
    "pointer_sha256",
    "keyword",
    "expected_sha256",
    "received",
    "message_sha256",
    "line",
    "column",
    "byte_offset",
    "snippet_sha256",
}
_RECEIVED_AUDIT_FIELDS = {"type", "chars", "length", "sha256", "value_sha256"}


def returned_model_evidence_is_safe(value: object) -> bool:
    """Accept only absent or content-free provider model identifiers."""

    return value is None or llm_runtime.safe_metadata_identifier(value) == value


def _audit_issue_is_safe(issue: object) -> bool:
    if not isinstance(issue, Mapping) or not set(issue).issubset(_ISSUE_AUDIT_FIELDS):
        return False
    received = issue.get("received")
    required = {
        "pointer_sha256",
        "keyword",
        "expected_sha256",
        "received",
        "message_sha256",
    }
    return bool(
        required.issubset(issue)
        and llm_runtime.safe_metadata_identifier(issue.get("keyword"))
        == issue.get("keyword")
        and _SHA256_RE.fullmatch(str(issue.get("pointer_sha256") or ""))
        and _SHA256_RE.fullmatch(str(issue.get("expected_sha256") or ""))
        and _SHA256_RE.fullmatch(str(issue.get("message_sha256") or ""))
        and isinstance(received, Mapping)
        and set(received).issubset(_RECEIVED_AUDIT_FIELDS)
        and llm_runtime.safe_metadata_identifier(received.get("type"))
        == received.get("type")
        and all(
            name not in received
            or (
                not isinstance(received.get(name), bool)
                and isinstance(received.get(name), int)
                and received[name] >= 0
            )
            for name in ("chars", "length")
        )
        and (
            "sha256" not in received
            or _SHA256_RE.fullmatch(str(received.get("sha256") or ""))
        )
        and (
            "value_sha256" not in received
            or _SHA256_RE.fullmatch(str(received.get("value_sha256") or ""))
        )
        and (
            "snippet_sha256" not in issue
            or _SHA256_RE.fullmatch(str(issue.get("snippet_sha256") or ""))
        )
        and all(
            name not in issue
            or (
                not isinstance(issue.get(name), bool)
                and isinstance(issue.get(name), int)
                and issue[name] >= 0
            )
            for name in ("line", "column", "byte_offset")
        )
    )


def _audit_attempt_is_safe(attempt: object) -> bool:
    if not isinstance(attempt, Mapping) or set(attempt) != _ATTEMPT_AUDIT_FIELDS:
        return False
    issues = attempt.get("issues")
    valid = attempt.get("valid")
    error_fingerprint = attempt.get("error_fingerprint")
    base = (
        not isinstance(attempt.get("index"), bool)
        and isinstance(attempt.get("index"), int)
        and attempt["index"] >= 0
        and isinstance(valid, bool)
        and _SHA256_RE.fullmatch(str(attempt.get("output_sha256") or ""))
        is not None
        and not isinstance(attempt.get("output_chars"), bool)
        and isinstance(attempt.get("output_chars"), int)
        and attempt["output_chars"] >= 0
        and isinstance(attempt.get("normalized"), bool)
        and isinstance(issues, list)
        and all(_audit_issue_is_safe(issue) for issue in issues)
    )
    if not base:
        return False
    if valid:
        return error_fingerprint is None and not issues
    return bool(
        _SHA256_RE.fullmatch(str(error_fingerprint or "")) is not None and issues
    )


def _safe_diagnostic(value: object) -> bool:
    if value is None or isinstance(value, bool):
        return True
    if isinstance(value, int):
        return value >= 0
    if isinstance(value, float):
        return math.isfinite(value) and value >= 0
    if isinstance(value, str):
        return _SAFE_DIAGNOSTIC_IDENTIFIER.fullmatch(value) is not None
    if isinstance(value, Mapping):
        return all(
            llm_runtime.safe_metadata_identifier(key) == key
            and _safe_diagnostic(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return all(_safe_diagnostic(item) for item in value)
    return False


def _runtime_audit_is_safe(runtime: object) -> bool:
    if not isinstance(runtime, Mapping) or set(runtime) != {
        "status",
        "model_size_bytes",
        "num_ctx",
    }:
        return False
    return bool(
        llm_runtime.safe_metadata_identifier(runtime.get("status"))
        == runtime.get("status")
        and all(
            runtime.get(name) is None
            or (
                not isinstance(runtime.get(name), bool)
                and isinstance(runtime.get(name), int)
                and runtime[name] >= 0
            )
            for name in ("model_size_bytes", "num_ctx")
        )
    )


def _session_audit_is_safe(
    session: object,
    *,
    model: object,
    returned_model: object,
) -> bool:
    if not isinstance(session, Mapping) or set(session) != _SESSION_AUDIT_FIELDS:
        return False
    attempts = session.get("attempts")
    failure_class = session.get("failure_class")
    return bool(
        isinstance(session.get("ok"), bool)
        and session.get("model") == model
        and session.get("returned_model") == returned_model
        and (
            failure_class is None
            or llm_runtime.safe_metadata_identifier(failure_class) == failure_class
        )
        and isinstance(session.get("first_pass_valid"), bool)
        and not isinstance(session.get("repair_turns"), bool)
        and isinstance(session.get("repair_turns"), int)
        and session["repair_turns"] >= 0
        and isinstance(attempts, list)
        and all(_audit_attempt_is_safe(attempt) for attempt in attempts)
    )


def base_semantic_authority(
    lane: str,
    *,
    injected_reviewer: bool = False,
    router_config: runtime_config.DecisionRouterConfig | None = None,
    router: Any | None = None,
    refresh_router: bool = False,
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
        from chronovisor.decision.decision_lane_contract_cases import (
            decision_lane_contract_case_manifest_sha256,
        )
        from chronovisor.decision.decision_lane_contracts import (
            lane_contract_manifest_sha256,
            lane_contract_sha256,
        )
        from chronovisor.decision.decision_policy import resolve_decision_policy
        from chronovisor.decision.decision_router import (
            QUORUM_SAFETY_POLICY_VERSION,
            DecisionRouter,
        )

        policy, mode, policy_error = resolve_decision_policy(lane)
        if policy is None or policy_error is not None or mode != "enabled":
            return None, policy_error or f"decision_lane_not_enabled:{lane}:{mode}"
        resolved_router = router or DecisionRouter(
            config=router_config,
            audit_role=lane,
            decision_lane=lane,
            require_adopted=True,
        )
        resolution = resolved_router.authority_router(refresh=refresh_router)
        authority = {
            "source": RUNTIME_ROUTE_SOURCE,
            "authority_version": AUTHORITY_VERSION,
            "lane": lane,
            "lane_contract_sha256": lane_contract_sha256(lane),
            "lane_contract_manifest_sha256": lane_contract_manifest_sha256(),
            "lane_contract_case_manifest_sha256": (
                decision_lane_contract_case_manifest_sha256()
            ),
            "quorum_safety_policy_version": QUORUM_SAFETY_POLICY_VERSION,
            "policy": {
                "kind": policy.kind if policy is not None else None,
                "schema_name": policy.schema_name if policy is not None else None,
                "mode": mode,
                "error": policy_error,
            },
            "router": resolution,
        }
    except Exception as exc:
        return None, f"decision_authority_unavailable:{exc.__class__.__name__}"

    if (
        resolution.get("source") != "runtime_role_mapping"
        or resolution.get("error") is not None
    ):
        return None, str(
            resolution.get("error") or f"decision_runtime_routes_not_valid:{lane}"
        )
    return authority, None


def current_semantic_authority(
    lane: str,
    *,
    injected_reviewer: bool = False,
    router_config: runtime_config.DecisionRouterConfig | None = None,
    router: Any | None = None,
    refresh_router: bool = False,
) -> tuple[dict[str, Any] | None, str | None]:
    """Return effect authority without making evaluation adoption a use condition."""

    return base_semantic_authority(
        lane,
        injected_reviewer=injected_reviewer,
        router_config=router_config,
        router=router,
        refresh_router=refresh_router,
    )


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
    if source != RUNTIME_ROUTE_SOURCE:
        return "decision authority source is invalid"
    expected_fields = {
        "source",
        "authority_version",
        "lane",
        "lane_contract_sha256",
        "lane_contract_manifest_sha256",
        "lane_contract_case_manifest_sha256",
        "quorum_safety_policy_version",
        "policy",
        "router",
    }
    if set(authority) != expected_fields:
        return "decision authority fields are invalid"
    quorum_safety_policy_version = authority.get("quorum_safety_policy_version")
    if (
        isinstance(quorum_safety_policy_version, bool)
        or not isinstance(quorum_safety_policy_version, int)
        or quorum_safety_policy_version < 1
    ):
        return "decision authority quorum safety policy is invalid"
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
    routes = router.get("routes") if isinstance(router, Mapping) else None
    if (
        not isinstance(router, Mapping)
        or set(router) != {"source", "error", "routes"}
        or router.get("source") != "runtime_role_mapping"
        or router.get("error") is not None
        or route_provenance_error(routes) is not None
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
    router_routes = router.get("routes") if isinstance(router, Mapping) else None
    consensus_error = _local_consensus_proof_error(
        review.get("local_consensus"),
        router_routes=router_routes,
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

    The redacted consensus envelope proves that two configured voters agreed on a
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
        from chronovisor.decision.decision_router import canonical_agreement_signature
        from chronovisor.decision.decision_schema_manifest import (
            production_decision_schemas,
        )

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
    router_routes: object,
) -> str | None:
    """Validate the redacted :class:`DecisionRouterResult` quorum proof.

    The decision payload is model-authored, while ``local_consensus`` is added
    by the trusted router boundary.  Production verdict authority therefore
    requires that audit envelope to prove the returned value actually reached
    a two-voter quorum; merely matching the configured router policy is not
    sufficient.
    """

    if not isinstance(consensus, Mapping):
        return "decision verdict local consensus proof is missing"
    if (
        route_provenance_error(router_routes) is not None
    ):
        return "decision verdict router route triplet is invalid"
    assert isinstance(router_routes, list)
    agreement_sha256 = consensus.get("agreement_sha256")
    votes = consensus.get("votes")
    if (
        consensus.get("status") != "agreed"
        or set(consensus) != _CONSENSUS_AUDIT_FIELDS
        or consensus.get("ok") is not True
        or not isinstance(agreement_sha256, str)
        or _SHA256_RE.fullmatch(agreement_sha256) is None
        or consensus.get("failure_class") is not None
        or consensus.get("quarantine_reason") is not None
        or not isinstance(consensus.get("conservative_veto_fired"), bool)
        or not isinstance(
            consensus.get("conservative_veto_bypassed_by_lane_policy"), bool
        )
        or consensus.get("dissent_effect_class")
        not in {None, "mutating", "conservative", "unclassifiable"}
        or isinstance(consensus.get("quorum_safety_policy_version"), bool)
        or not isinstance(consensus.get("quorum_safety_policy_version"), int)
        or consensus["quorum_safety_policy_version"] < 1
        or isinstance(consensus.get("num_ctx"), bool)
        or not isinstance(consensus.get("num_ctx"), int)
        or consensus["num_ctx"] < 1
        or not _safe_diagnostic(consensus.get("residency"))
        or not isinstance(votes, list)
        or not 2 <= len(votes) <= 3
    ):
        return "decision verdict local consensus proof is invalid"

    expected_roles = ("primary", "challenger", "tie_break")
    agreeing_votes = 0
    for index, vote in enumerate(votes):
        if not isinstance(vote, Mapping) or set(vote) != _VOTE_AUDIT_FIELDS:
            return "decision verdict local consensus vote is invalid"
        valid = vote.get("valid")
        signature_sha256 = vote.get("signature_sha256")
        route = router_routes[index]
        assert isinstance(route, Mapping)
        model = vote.get("model")
        provider = vote.get("provider")
        provenance = vote.get("route_provenance")
        returned_model = vote.get("returned_model")
        runtime_observation = vote.get("runtime_observation")
        session = vote.get("session")
        role = vote.get("role")
        if (
            not isinstance(valid, bool)
            or role != expected_roles[index]
            or model != route.get("model")
            or provider != route.get("provider")
            or provenance != route
            or not returned_model_evidence_is_safe(returned_model)
            or isinstance(vote.get("requested_num_ctx"), bool)
            or not isinstance(vote.get("requested_num_ctx"), int)
            or vote["requested_num_ctx"] < 1
            or (
                vote.get("decision_label") is not None
                and llm_runtime.safe_metadata_identifier(
                    vote.get("decision_label")
                )
                != vote.get("decision_label")
            )
            or vote.get("effect_class")
            not in {None, "mutating", "conservative", "unclassifiable"}
            or not _runtime_audit_is_safe(runtime_observation)
            or not _session_audit_is_safe(
                session,
                model=model,
                returned_model=returned_model,
            )
        ):
            return "decision verdict local consensus vote authority is invalid"
        if valid:
            if (
                (
                    route.get("location") == "remote"
                    and returned_model != route.get("model")
                )
                or not isinstance(signature_sha256, str)
                or _SHA256_RE.fullmatch(signature_sha256) is None
                or vote.get("invalid_reason") is not None
            ):
                return "decision verdict local consensus vote is invalid"
            if signature_sha256 == agreement_sha256:
                agreeing_votes += 1
        elif (
            signature_sha256 is not None
            or not isinstance(vote.get("invalid_reason"), str)
            or not vote["invalid_reason"]
            or llm_runtime.safe_metadata_identifier(vote.get("invalid_reason"))
            != vote.get("invalid_reason")
        ):
            return "decision verdict local consensus vote is invalid"

    if agreeing_votes < 2:
        return "decision verdict local consensus quorum is not proven"
    return None


def route_provenance_error(routes: object) -> str | None:
    if not isinstance(routes, list) or len(routes) != 3:
        return "route triplet is invalid"
    expected_roles = (
        "classification.primary",
        "classification.challenger",
        "classification.tie_break",
    )
    identities: set[tuple[str, str, str]] = set()
    for expected_role, route in zip(expected_roles, routes, strict=True):
        if not isinstance(route, Mapping) or set(route) != {
            "role",
            "provider",
            "model",
            "location",
            "protocol",
            "endpoint_sha256",
            "revision",
            "ollama",
        }:
            return "route fields are invalid"
        provider = route.get("provider")
        model = route.get("model")
        location = route.get("location")
        protocol = route.get("protocol")
        endpoint_sha256 = route.get("endpoint_sha256")
        revision = route.get("revision")
        ollama_identity = route.get("ollama")
        if (
            route.get("role") != expected_role
            or not isinstance(provider, str)
            or llm_runtime.safe_metadata_identifier(provider) != provider
            or not isinstance(model, str)
            or llm_runtime.safe_metadata_identifier(model) != model
            or location not in {"local", "remote"}
            or not isinstance(protocol, str)
            or llm_runtime.safe_metadata_identifier(protocol) != protocol
            or protocol == "unknown"
            or endpoint_sha256 is not None
            and (
                not isinstance(endpoint_sha256, str)
                or _SHA256_RE.fullmatch(endpoint_sha256) is None
            )
            or revision is not None
            and llm_runtime.safe_metadata_identifier(revision) != revision
            or location == "remote"
            and (
                revision is None
                or not isinstance(endpoint_sha256, str)
                or _SHA256_RE.fullmatch(endpoint_sha256) is None
                or protocol
                not in {
                    item.value for item in provider_profiles.ProviderProtocol
                }
            )
        ):
            return "route identity is invalid"
        if provider == "ollama" and location == "local":
            engine = (
                ollama_identity.get("engine")
                if isinstance(ollama_identity, Mapping)
                else None
            )
            if (
                not isinstance(ollama_identity, Mapping)
                or set(ollama_identity)
                != {"engine", "digest", "quantization_level"}
                or not isinstance(engine, Mapping)
                or set(engine) != {"name", "version"}
                or engine.get("name") != "ollama"
                or not isinstance(engine.get("version"), str)
                or not engine.get("version")
                or not isinstance(ollama_identity.get("digest"), str)
                or not ollama_identity.get("digest")
                or not isinstance(ollama_identity.get("quantization_level"), str)
                or not ollama_identity.get("quantization_level")
                or protocol != "ollama-native"
                or not isinstance(endpoint_sha256, str)
                or _SHA256_RE.fullmatch(endpoint_sha256) is None
            ):
                return "Ollama route identity is invalid"
        elif ollama_identity is not None:
            return "non-Ollama route has Ollama identity"
        identity = (protocol, endpoint_sha256 or provider, model)
        if identity in identities:
            return "route identities are not independent"
        identities.add(identity)
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
    "RUNTIME_ROUTE_SOURCE",
    "base_semantic_authority",
    "compare_semantic_authority",
    "current_semantic_authority",
    "route_provenance_error",
    "returned_model_evidence_is_safe",
    "seal_semantic_artifact",
    "semantic_authority_shape_error",
    "semantic_verdict_authority_error",
    "semantic_verdict_authority_provenance_error",
]
