from __future__ import annotations

import hashlib

from llm_wiki_mcp.decision_authority import (
    compare_semantic_authority,
    seal_semantic_artifact,
    semantic_authority_shape_error,
    semantic_verdict_authority_error,
)


def _production_authority(*, artifact_sha256: str = "a" * 64) -> dict:
    return {
        "source": "adopted_local_consensus",
        "authority_version": 1,
        "lane": "example_lane",
        "lane_contract_sha256": "b" * 64,
        "lane_contract_manifest_sha256": "c" * 64,
        "lane_contract_case_manifest_sha256": "d" * 64,
        "policy": {
            "kind": "consensus",
            "schema_name": "generic_decision",
            "mode": "enabled",
            "error": None,
        },
        "router": {
            "source": "adopted_artifact",
            "artifact_sha256": artifact_sha256,
            "error": None,
            "models": ["primary", "challenger", "tie-break"],
        },
    }


def _review(authority: dict) -> dict:
    from llm_wiki_mcp.decision_router import canonical_agreement_signature
    from llm_wiki_mcp.decision_schema_manifest import production_decision_schemas

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
    review["local_consensus"] = {
        "status": "agreed",
        "ok": True,
        "agreement_sha256": agreement,
        "failure_class": None,
        "quarantine_reason": None,
        "votes": [
            {
                "role": "primary",
                "model": "primary",
                "valid": True,
                "signature_sha256": agreement,
                "invalid_reason": None,
            },
            {
                "role": "challenger",
                "model": "challenger",
                "valid": True,
                "signature_sha256": agreement,
                "invalid_reason": None,
            },
        ],
    }
    return review


def test_semantic_artifact_seal_binds_complete_authority_epoch() -> None:
    authority = _production_authority()

    artifact = seal_semantic_artifact(
        {"schema_version": 1, "review": _review(authority)},
        authority=authority,
        lane="example_lane",
    )
    authority["router"]["artifact_sha256"] = "e" * 64

    assert artifact["authority"]["router"]["artifact_sha256"] == "a" * 64
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

    assert "case_manifest" in str(
        semantic_authority_shape_error(incomplete, lane="example_lane")
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
    from llm_wiki_mcp.decision_router import canonical_agreement_signature
    from llm_wiki_mcp.decision_schema_manifest import production_decision_schemas

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
        {
            "role": "tie_break",
            "model": "tie-break",
            "valid": True,
            "signature_sha256": conservative_agreement,
            "invalid_reason": None,
        }
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
        {
            "role": "tie_break",
            "model": "tie-break",
            "valid": True,
            "signature_sha256": agreement,
            "invalid_reason": None,
        }
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
        "authority_version": 1,
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
