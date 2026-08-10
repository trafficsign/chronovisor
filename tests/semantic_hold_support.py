"""Shared strict semantic-hold fixtures for lane regression tests."""

from __future__ import annotations

from chronovisor.decision import semantic_hold
from chronovisor.decision.decision_authority import AUTHORITY_VERSION
from chronovisor.decision.decision_router import QUORUM_SAFETY_POLICY_VERSION


def semantic_authority(
    lane: str = "recall_auto_apply",
    *,
    artifact_sha256: str = "a" * 64,
    kind: str = "consensus",
    schema_name: str = "generic_decision",
    quorum_safety_policy_version: int = QUORUM_SAFETY_POLICY_VERSION,
) -> dict[str, object]:
    models = ("ornith:test", "gpt-oss:test", "gemma:test")
    routes = [
        {
            "role": f"classification.{role}",
            "provider": "ollama",
            "model": model,
            "location": "local",
            "protocol": "ollama-native",
            "endpoint_sha256": "f" * 64,
            "revision": None,
            "ollama": {
                "engine": {"name": "ollama", "version": "test"},
                "digest": artifact_sha256 if index == 1 else str(index) * 64,
                "quantization_level": "Q4_K_M",
            },
        }
        for index, (role, model) in enumerate(
            zip(("primary", "challenger", "tie_break"), models, strict=True),
            start=1,
        )
    ]
    return {
        "source": "configured_runtime_consensus",
        "authority_version": AUTHORITY_VERSION,
        "lane": lane,
        "lane_contract_sha256": "b" * 64,
        "lane_contract_manifest_sha256": "c" * 64,
        "lane_contract_case_manifest_sha256": "d" * 64,
        "quorum_safety_policy_version": quorum_safety_policy_version,
        "policy": {
            "kind": kind,
            "schema_name": schema_name,
            "mode": "enabled",
            "error": None,
        },
        "router": {
            "source": "runtime_role_mapping",
            "error": None,
            "routes": routes,
        },
    }


def _vote(role: str, route: dict[str, object], signature: str) -> dict[str, object]:
    model = str(route["model"])
    return {
        "role": role,
        "provider": route["provider"],
        "model": model,
        "route_provenance": route,
        "returned_model": None,
        "requested_num_ctx": 32_768,
        "valid": True,
        "signature_sha256": signature,
        "invalid_reason": None,
        "decision_label": "needs_retry",
        "effect_class": "conservative",
        "runtime_observation": {
            "status": "unavailable",
            "model_size_bytes": None,
            "num_ctx": None,
        },
        "session": {
            "ok": True,
            "model": model,
            "failure_class": None,
            "returned_model": None,
            "first_pass_valid": True,
            "repair_turns": 0,
            "attempts": [
                {
                    "index": 0,
                    "valid": True,
                    "output_sha256": signature,
                    "output_chars": 120,
                    "normalized": False,
                    "error_fingerprint": None,
                    "issues": [],
                }
            ],
        },
    }


def semantic_review(
    authority: dict[str, object] | None = None,
    *,
    lane: str = "recall_auto_apply",
    signatures: tuple[str, str, str] = ("1" * 64, "2" * 64, "3" * 64),
) -> dict[str, object]:
    authority = authority or semantic_authority(lane)
    router = authority["router"]
    assert isinstance(router, dict)
    routes = router["routes"]
    assert isinstance(routes, list)
    reason = "local_models_did_not_reach_two_vote_quorum"
    policy = authority["policy"]
    assert isinstance(policy, dict)
    return {
        "decision": "needs_retry",
        "summary": reason,
        "tests_run": [],
        "commit": None,
        "committed": False,
        "pushed": False,
        "risk": None,
        "notes": None,
        "frontier_failure": {
            "failure_class": semantic_hold.LOCAL_SEMANTIC_NO_QUORUM,
            "rescue_status": "local_quarantined",
            "summary": reason,
            "human_required": False,
            "notify_user": False,
        },
        "human_required": False,
        "reviewer": "local_consensus",
        "local_consensus": {
            "status": "quarantined",
            "ok": False,
            "conservative_veto_fired": False,
            "conservative_veto_bypassed_by_lane_policy": False,
            "dissent_effect_class": None,
            "quorum_safety_policy_version": QUORUM_SAFETY_POLICY_VERSION,
            "agreement_sha256": None,
            "failure_class": "local_consensus_failed",
            "quarantine_reason": reason,
            "num_ctx": 32_768,
            "residency": {},
            "votes": [
                _vote("primary", routes[0], signatures[0]),
                _vote("challenger", routes[1], signatures[1]),
                _vote("tie_break", routes[2], signatures[2]),
            ],
        },
        "decision_policy": {
            "lane": lane,
            "kind": policy["kind"],
            "schema_name": policy["schema_name"],
            "mode": policy["mode"],
            "error": policy["error"],
            "expected_schema_sha256": "e" * 64,
            "actual_schema_sha256": "e" * 64,
            "router_policy": router,
        },
    }


def structured_review_epoch(
    authority: dict[str, object],
    *,
    lane: str = "recall_auto_apply",
    schema_sha256: str = "e" * 64,
    prompt: str = "private prompt",
    system: str | None = "private system",
    effective_request_sha256: str = "f" * 64,
    resolver_sha256: str = semantic_hold.STRUCTURED_REVIEW_HOLD_RESOLVER_SHA256,
) -> dict[str, object]:
    return semantic_hold.build_structured_review_hold_epoch(
        lane=lane,
        authority=authority,
        schema_sha256=schema_sha256,
        prompt=prompt,
        system=system,
        effective_request_sha256=effective_request_sha256,
        resolver_sha256=resolver_sha256,
    )
