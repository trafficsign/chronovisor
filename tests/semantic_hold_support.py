"""Shared strict semantic-hold fixtures for lane regression tests."""

from __future__ import annotations

from chronovisor.search import semantic_hold


def semantic_authority(
    lane: str = "recall_auto_apply",
    *,
    artifact_sha256: str = "a" * 64,
    kind: str = "consensus",
    schema_name: str = "generic_decision",
) -> dict[str, object]:
    return {
        "source": "adopted_local_consensus",
        "authority_version": 1,
        "lane": lane,
        "lane_contract_sha256": "b" * 64,
        "lane_contract_manifest_sha256": "c" * 64,
        "lane_contract_case_manifest_sha256": "d" * 64,
        "policy": {
            "kind": kind,
            "schema_name": schema_name,
            "mode": "enabled",
            "error": None,
        },
        "router": {
            "source": "adopted_artifact",
            "artifact_sha256": artifact_sha256,
            "error": None,
            "models": ["ornith:test", "gpt-oss:test", "gemma:test"],
        },
    }


def _vote(role: str, model: str, signature: str) -> dict[str, object]:
    return {
        "role": role,
        "model": model,
        "requested_num_ctx": 32_768,
        "valid": True,
        "signature_sha256": signature,
        "invalid_reason": None,
        "runtime_observation": {
            "status": "unavailable",
            "model_size_bytes": None,
            "num_ctx": None,
        },
        "session": {
            "ok": True,
            "model": model,
            "failure_class": None,
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
    models = router["models"]
    assert isinstance(models, list)
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
            "quorum_safety_policy_version": 1,
            "agreement_sha256": None,
            "failure_class": "local_consensus_failed",
            "quarantine_reason": reason,
            "num_ctx": 32_768,
            "residency": {},
            "votes": [
                _vote("primary", str(models[0]), signatures[0]),
                _vote("challenger", str(models[1]), signatures[1]),
                _vote("tie_break", str(models[2]), signatures[2]),
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
