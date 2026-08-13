from __future__ import annotations

import hashlib

import pytest

from chronovisor.decision.decision_authority import (
    AUTHORITY_VERSION,
    compare_semantic_authority,
    seal_semantic_artifact,
    semantic_authority_shape_error,
    semantic_verdict_authority_error,
)
from chronovisor.decision.decision_router import (
    QUORUM_SAFETY_POLICY_VERSION,
    DecisionVote,
)
from chronovisor.decision.local_structured import (
    LocalStructuredResult,
    StructuredAttempt,
)


def _production_authority(*, artifact_sha256: str = "a" * 64) -> dict:
    models = ("primary", "challenger", "tie-break")
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
                "digest": artifact_sha256 if index == 0 else str(index) * 64,
                "quantization_level": "Q4_K_M",
            },
        }
        for index, (role, model) in enumerate(
            zip(("primary", "challenger", "tie_break"), models, strict=True)
        )
    ]
    return {
        "source": "configured_runtime_consensus",
        "authority_version": AUTHORITY_VERSION,
        "lane": "example_lane",
        "lane_contract_sha256": "b" * 64,
        "lane_contract_manifest_sha256": "c" * 64,
        "lane_contract_case_manifest_sha256": "d" * 64,
        "quorum_safety_policy_version": QUORUM_SAFETY_POLICY_VERSION,
        "policy": {
            "kind": "consensus",
            "schema_name": "generic_decision",
            "mode": "enabled",
            "error": None,
        },
        "router": {
            "source": "runtime_role_mapping",
            "error": None,
            "routes": routes,
        },
    }


def _vote_audit(role: str, route: dict, agreement: str) -> dict:
    model = route["model"]
    returned_model = model if route["location"] == "remote" else None
    result = LocalStructuredResult(
        ok=True,
        model=model,
        attempts=(StructuredAttempt(0, True, agreement, 80, False, None, ()),),
        returned_model=returned_model,
        think="medium",
        ollama_think="medium",
        num_predict=256,
        think_selection_reason="medium_default",
        required_num_ctx=8_000,
        requested_num_ctx=16_384,
        effective_num_ctx=16_384,
    )
    return DecisionVote(
        role=role,
        model=model,
        provider=route["provider"],
        result=result,
        requested_num_ctx=16_384,
        route_provenance=route,
        signature="canonical action",
        signature_sha256=agreement,
        decision_label="approved",
        effect_class="mutating",
    ).audit_record()


def _review(authority: dict) -> dict:
    from chronovisor.decision.decision_router import canonical_agreement_signature
    from chronovisor.decision.decision_schema_manifest import (
        production_decision_schemas,
    )

    review = {
        "decision": "approved",
        "summary": "canonical action proof",
        "tests_run": [],
        "commit": None,
        "committed": False,
        "pushed": False,
        "risk": None,
        "notes": None,
        "decision_policy": {
            **authority["policy"],
            "router_policy": authority["router"],
        },
    }
    signature = canonical_agreement_signature(
        review,
        schema=production_decision_schemas()[authority["policy"]["schema_name"]],
    )
    agreement = hashlib.sha256(signature.encode("utf-8")).hexdigest()
    routes = authority["router"]["routes"]
    review["local_consensus"] = {
        "status": "agreed",
        "ok": True,
        "conservative_veto_fired": False,
        "conservative_veto_bypassed_by_lane_policy": False,
        "dissent_effect_class": None,
        "quorum_safety_policy_version": QUORUM_SAFETY_POLICY_VERSION,
        "agreement_sha256": agreement,
        "failure_class": None,
        "quarantine_reason": None,
        "num_ctx": 16_384,
        "residency": {},
        "votes": [
            _vote_audit("primary", routes[0], agreement),
            _vote_audit("challenger", routes[1], agreement),
        ],
    }
    return review


def _remote_authority() -> dict:
    authority = _production_authority()
    for index, route in enumerate(authority["router"]["routes"]):
        route.update(
            provider=f"remote-{index}",
            location="remote",
            protocol="openai-compatible",
            endpoint_sha256=str(index + 1) * 64,
            revision=f"deployment-{index}",
            ollama=None,
        )
    return authority


def test_semantic_artifact_seal_binds_complete_authority_epoch() -> None:
    authority = _production_authority()

    artifact = seal_semantic_artifact(
        {"schema_version": 1, "review": _review(authority)},
        authority=authority,
        lane="example_lane",
    )
    authority["router"]["routes"][0]["ollama"]["digest"] = "e" * 64

    assert (
        artifact["authority"]["router"]["routes"][0]["ollama"]["digest"]
        == "a" * 64
    )
    assert (
        semantic_authority_shape_error(
            artifact["authority"],
            lane="example_lane",
        )
        is None
    )
    assert (
        semantic_verdict_authority_error(
            artifact["review"],
            artifact["authority"],
            lane="example_lane",
        )
        is None
    )


def test_semantic_authority_compare_rejects_adoption_epoch_change() -> None:
    expected = _production_authority()
    current = _production_authority(artifact_sha256="e" * 64)

    assert (
        compare_semantic_authority(expected, current, lane="example_lane")
        == "decision authority changed before effect"
    )


def test_semantic_authority_rejects_incomplete_or_mismatched_verdict() -> None:
    authority = _production_authority()
    incomplete = dict(authority)
    incomplete.pop("lane_contract_case_manifest_sha256")

    assert (
        semantic_authority_shape_error(incomplete, lane="example_lane")
        == "decision authority fields are invalid"
    )
    assert (
        semantic_verdict_authority_error(
            {"decision_policy": {**authority["policy"], "router_policy": {}}},
            authority,
            lane="example_lane",
        )
        == "decision verdict router authority changed"
    )


def test_semantic_authority_requires_a_real_local_quorum_proof() -> None:
    authority = _production_authority()
    missing = _review(authority)
    missing.pop("local_consensus")
    split_vote = _review(authority)
    split_vote["local_consensus"]["votes"][1]["signature_sha256"] = "f" * 64

    assert (
        semantic_verdict_authority_error(
            missing,
            authority,
            lane="example_lane",
        )
        == "decision verdict local consensus proof is missing"
    )
    assert (
        semantic_verdict_authority_error(
            split_vote,
            authority,
            lane="example_lane",
        )
        == "decision verdict local consensus quorum is not proven"
    )


def test_semantic_authority_rejects_one_vote_conservative_synthetic_agreement() -> None:
    from chronovisor.decision.decision_router import canonical_agreement_signature
    from chronovisor.decision.decision_schema_manifest import (
        production_decision_schemas,
    )

    authority = _production_authority()
    review = _review(authority)
    mutating_agreement = review["local_consensus"]["agreement_sha256"]
    review["decision"] = "rejected"
    conservative_signature = canonical_agreement_signature(
        review,
        schema=production_decision_schemas()[authority["policy"]["schema_name"]],
    )
    conservative_agreement = hashlib.sha256(
        conservative_signature.encode("utf-8")
    ).hexdigest()
    review["local_consensus"]["agreement_sha256"] = conservative_agreement
    review["local_consensus"]["votes"].append(
        _vote_audit(
            "tie_break",
            authority["router"]["routes"][2],
            conservative_agreement,
        )
    )
    assert all(
        vote["signature_sha256"] == mutating_agreement
        for vote in review["local_consensus"]["votes"][:2]
    )

    assert (
        semantic_verdict_authority_error(
            review,
            authority,
            lane="example_lane",
        )
        == "decision verdict local consensus quorum is not proven"
    )


def test_semantic_authority_binds_actual_action_to_consensus_signature() -> None:
    authority = _production_authority()
    tampered = _review(authority)
    tampered["decision"] = "rejected"

    assert (
        semantic_verdict_authority_error(
            tampered,
            authority,
            lane="example_lane",
        )
        == "decision verdict action does not match local consensus agreement"
    )


def test_semantic_authority_rejects_unknown_canonical_schema() -> None:
    authority = _production_authority()
    review = _review(authority)
    authority["policy"]["schema_name"] = "not-a-production-schema"
    review["decision_policy"]["schema_name"] = "not-a-production-schema"

    assert (
        semantic_verdict_authority_error(
            review,
            authority,
            lane="example_lane",
        )
        == "decision verdict canonical schema is unknown"
    )


def test_semantic_authority_binds_vote_roles_and_models_to_router_triplet() -> None:
    authority = _production_authority()
    wrong_model = _review(authority)
    wrong_model["local_consensus"]["votes"][1]["model"] = "arbitrary-model"
    wrong_role = _review(authority)
    wrong_role["local_consensus"]["votes"][0]["role"] = "tie_break"

    assert (
        semantic_verdict_authority_error(
            wrong_model,
            authority,
            lane="example_lane",
        )
        == "decision verdict local consensus vote authority is invalid"
    )
    assert (
        semantic_verdict_authority_error(
            wrong_role,
            authority,
            lane="example_lane",
        )
        == "decision verdict local consensus vote authority is invalid"
    )


def test_semantic_authority_accepts_current_producer_and_legacy_session_audits() -> (
    None
):
    authority = _production_authority()
    review = _review(authority)

    assert (
        semantic_verdict_authority_error(review, authority, lane="example_lane") is None
    )

    for vote in review["local_consensus"]["votes"]:
        vote["session"] = LocalStructuredResult(
            ok=True,
            model=vote["model"],
            returned_model=vote["returned_model"],
        ).audit_record()
    assert (
        semantic_verdict_authority_error(review, authority, lane="example_lane") is None
    )

    legacy_fields = {
        "ok",
        "model",
        "failure_class",
        "returned_model",
        "first_pass_valid",
        "repair_turns",
        "attempts",
    }
    for vote in review["local_consensus"]["votes"]:
        vote["session"] = {
            key: value for key, value in vote["session"].items() if key in legacy_fields
        }
    assert (
        semantic_verdict_authority_error(review, authority, lane="example_lane") is None
    )


@pytest.mark.parametrize(
    ("field", "invalid", "remove"),
    [
        ("structured_generation_policy_version", True, False),
        ("structured_generation_policy_sha256", "A" * 64, False),
        ("think", "private reasoning", False),
        ("ollama_think", 1, False),
        ("num_predict", 0, False),
        ("think_selection_reason", "private reason", False),
        ("required_num_ctx", True, False),
        ("requested_num_ctx", 0, False),
        ("effective_num_ctx", -1, False),
        ("context_tokens", 32_768, False),
        ("private_payload", "must not persist", False),
        ("think", None, True),
    ],
)
def test_semantic_authority_rejects_invalid_current_session_audit(
    field: str,
    invalid: object,
    remove: bool,
) -> None:
    authority = _production_authority()
    review = _review(authority)
    session = review["local_consensus"]["votes"][0]["session"]
    if remove:
        session.pop(field)
    else:
        session[field] = invalid

    assert (
        semantic_verdict_authority_error(review, authority, lane="example_lane")
        == "decision verdict local consensus vote authority is invalid"
    )


def test_semantic_authority_accepts_exact_tie_break_quorum_after_invalid_primary() -> (
    None
):
    authority = _production_authority()
    review = _review(authority)
    agreement = review["local_consensus"]["agreement_sha256"]
    review["local_consensus"]["votes"][0].update(
        valid=False,
        signature_sha256=None,
        invalid_reason="transport_error",
    )
    review["local_consensus"]["votes"].append(
        _vote_audit("tie_break", authority["router"]["routes"][2], agreement)
    )

    assert (
        semantic_verdict_authority_error(
            review,
            authority,
            lane="example_lane",
        )
        is None
    )


def test_injected_reviewer_explicitly_bypasses_local_quorum_proof() -> None:
    authority = {
        "source": "injected_reviewer_boundary",
        "authority_version": AUTHORITY_VERSION,
        "lane": "example_lane",
    }

    assert (
        semantic_verdict_authority_error(
            {"decision": "approved"},
            authority,
            lane="example_lane",
        )
        is None
    )


def test_remote_authority_requires_configured_revision_and_exact_returned_model() -> None:
    authority = _remote_authority()
    review = _review(authority)
    for vote in review["local_consensus"]["votes"]:
        vote["returned_model"] = vote["model"]

    assert semantic_authority_shape_error(authority, lane="example_lane") is None
    assert (
        semantic_verdict_authority_error(review, authority, lane="example_lane")
        is None
    )

    authority["router"]["routes"][0]["revision"] = None
    assert "router identity" in str(
        semantic_authority_shape_error(authority, lane="example_lane")
    )

    authority = _remote_authority()
    review = _review(authority)
    for vote in review["local_consensus"]["votes"]:
        vote["returned_model"] = vote["model"]
    review["local_consensus"]["votes"][0]["returned_model"] = "other-model"
    assert (
        semantic_verdict_authority_error(review, authority, lane="example_lane")
        == "decision verdict local consensus vote authority is invalid"
    )

    for field, invalid in (("protocol", "unknown"), ("endpoint_sha256", None)):
        authority = _remote_authority()
        authority["router"]["routes"][0][field] = invalid
        assert "router identity" in str(
            semantic_authority_shape_error(authority, lane="example_lane")
        )


def test_remote_invalid_vote_is_excluded_from_authoritative_quorum() -> None:
    authority = _remote_authority()
    review = _review(authority)
    agreement = review["local_consensus"]["agreement_sha256"]
    review["local_consensus"]["votes"][0].update(
        valid=False,
        returned_model="different-primary",
        signature_sha256=None,
        invalid_reason="returned_model_mismatch",
    )
    review["local_consensus"]["votes"][0]["session"]["returned_model"] = (
        "different-primary"
    )
    review["local_consensus"]["votes"].append(
        _vote_audit("tie_break", authority["router"]["routes"][2], agreement)
    )

    assert (
        semantic_verdict_authority_error(review, authority, lane="example_lane")
        is None
    )

    review["local_consensus"]["votes"][2].update(
        valid=False,
        returned_model=None,
        signature_sha256=None,
        invalid_reason="timeout",
    )
    review["local_consensus"]["votes"][2]["session"]["returned_model"] = None
    assert (
        semantic_verdict_authority_error(review, authority, lane="example_lane")
        == "decision verdict local consensus quorum is not proven"
    )


def test_route_authority_rejects_aliased_voters_and_invalid_ollama_transport() -> None:
    authority = _remote_authority()
    for route in authority["router"]["routes"]:
        route["model"] = "shared-model"
        route["endpoint_sha256"] = "9" * 64
    assert "router identity" in str(
        semantic_authority_shape_error(authority, lane="example_lane")
    )

    for field, invalid in (("protocol", "custom-transport"), ("endpoint_sha256", None)):
        authority = _production_authority()
        authority["router"]["routes"][0][field] = invalid
        assert "router identity" in str(
            semantic_authority_shape_error(authority, lane="example_lane")
        )


def test_old_authority_schema_fails_closed() -> None:
    authority = _production_authority()
    authority["authority_version"] = AUTHORITY_VERSION - 1

    assert (
        semantic_authority_shape_error(authority, lane="example_lane")
        == "decision authority identity is invalid"
    )


def test_authority_rejects_extra_durable_fields() -> None:
    authority = _production_authority()
    authority["prompt"] = "CANARY private prompt"

    assert (
        semantic_authority_shape_error(authority, lane="example_lane")
        == "decision authority fields are invalid"
    )

    review = _review(_production_authority())
    review["local_consensus"]["votes"][0]["session"]["attempts"][0]["issues"] = [
        {
            "pointer_sha256": "1" * 64,
            "keyword": "type",
            "expected_sha256": "2" * 64,
            "received": {"value": "CANARY private response"},
            "message_sha256": "3" * 64,
        }
    ]
    assert (
        semantic_verdict_authority_error(
            review,
            _production_authority(),
            lane="example_lane",
        )
        == "decision verdict local consensus vote authority is invalid"
    )


def test_authority_accepts_content_free_local_residency_audit() -> None:
    from chronovisor.core.ollama_calibration import ModelResidencyPlan

    authority = _production_authority()
    review = _review(authority)
    review["local_consensus"]["residency"] = ModelResidencyPlan(
        num_ctx=16_384,
        max_resident_models=2,
        capacity_bytes=100,
        reserve_bytes=10,
        available_bytes=90,
        total_bytes=100,
        estimated_model_bytes=(("primary", 10),),
        role_contexts=(("primary", 16_384),),
        resident_models=("primary",),
        calibrated_models=("primary",),
        source="macos_vm_stat+ollama+identity_unavailable",
    ).audit_record()

    assert (
        semantic_verdict_authority_error(review, authority, lane="example_lane")
        is None
    )

    review["local_consensus"]["residency"]["source"] = "CANARY private source"
    assert (
        semantic_verdict_authority_error(review, authority, lane="example_lane")
        == "decision verdict local consensus proof is invalid"
    )


def test_qualification_lane_uses_runtime_routes_without_legacy_adoption() -> None:
    from chronovisor.decision.decision_authority import current_semantic_authority

    class FakeRouter:
        def authority_router(self, *, refresh: bool = False) -> dict:
            assert refresh is False
            return _production_authority()["router"]

    authority, error = current_semantic_authority(
        "recall_answer_adjudication",
        router=FakeRouter(),
    )

    assert error is None
    assert authority is not None
    assert "capability_sha256" not in authority
    assert (
        semantic_authority_shape_error(
            authority,
            lane="recall_answer_adjudication",
        )
        is None
    )
