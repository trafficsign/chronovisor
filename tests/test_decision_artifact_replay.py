"""Replay must retain the same bounded, redacted authority evidence as generation."""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest

from chronovisor.core import store
from chronovisor.decision import decision_authority, decision_router
from chronovisor.decision.decision_artifact import DecisionArtifactError
from chronovisor.decision.decision_router import DecisionRouter
from chronovisor.decision.local_structured import ChatRequest
from tests.test_full_autonomy import SCHEMA, _config


@pytest.mark.parametrize("repairs", [0, 1, 2])
def test_single_model_replay_preserves_authority_repair_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, repairs: int
) -> None:
    monkeypatch.setattr(store, "CHRONOVISOR_ROOT", tmp_path / "wiki")
    monkeypatch.setattr(
        decision_router, "bind_lane_contract_request",
        lambda _lane, prompt, _schema, system: (prompt, system),
    )
    monkeypatch.setattr(
        decision_authority, "current_semantic_authority",
        lambda lane: ({"source": "test", "lane": lane, "epoch": "one"}, None),
    )
    calls: list[ChatRequest] = []

    def transport(request: ChatRequest) -> str:
        calls.append(request)
        if len(calls) <= repairs:
            return "{}" if len(calls) == 1 else '{"decision":"approved"}'
        return json.dumps({"decision": "approved", "reason": "synthetic fixture"})

    router = DecisionRouter(
        config=replace(_config(), authority_kind="single_model_v1"),
        transport=transport, resolve_adoption=False, record_replay=False,
        live_resource_control=False, artifact_replay=True,
        decision_artifact_root=tmp_path / "artifacts",
    )
    first = router.decide("synthetic prompt", SCHEMA, decision_lane="test_lane")
    second = router.decide("synthetic prompt", SCHEMA, decision_lane="test_lane")

    assert first.ok and second.ok
    assert len(calls) == repairs + 1
    assert second.residency is not None
    assert second.residency["source"] == "single_model_artifact_replay"
    assert second.residency["model_invocations"] == 0
    original = first.votes[0].result.audit_record()
    restored = second.votes[0].result.audit_record()
    assert restored["attempts"] == original["attempts"]
    assert restored["repair_turns"] == repairs
    assert restored["first_pass_valid"] == (repairs == 0)
    assert len(second.votes[0].result.attempts) == repairs + 1
    assert second.votes[0].result.attempts[-1].valid
    assert decision_authority._single_model_proof_error(
        second.audit_record(), router_routes=router.authority_router()["routes"]
    ) is None

    fingerprint = str(second.residency["execution_fingerprint"])
    artifact = router.decision_artifact_store.load(fingerprint)
    assert artifact is not None
    # Even a caller bypassing store.load must not replay missing or unsafe proof.
    for mutation in ("missing_attempt", "extra_field", "invalid_issue"):
        broken = copy.deepcopy(artifact)
        attempts = broken["single_model_proof"]["attempts"]
        if mutation == "missing_attempt":
            attempts.pop()
        elif mutation == "extra_field":
            attempts[-1]["untrusted_text"] = "synthetic canary"
        else:
            attempts[-1]["issues"] = [{"message": "synthetic canary"}]
        with pytest.raises(DecisionArtifactError, match="repair history"):
            router._replay_single_artifact(broken, schema=SCHEMA, context_tier=32_768)
    assert router.decision_artifact_store.load(fingerprint) == artifact
