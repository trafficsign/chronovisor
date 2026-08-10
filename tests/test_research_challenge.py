from __future__ import annotations

from pathlib import Path

from chronovisor.research.evidence_bundle import build_bundle, simple_assess_claims
from chronovisor.research.research_challenge import challenge_bundle
from chronovisor.search.research_config import ResearchConfig
from chronovisor.search.research_store import ResearchStore


def test_disagreement_calls_tie_break_with_role_budgets(
    tmp_path: Path, monkeypatch
) -> None:
    from chronovisor.core import research_scheduler
    from chronovisor.search import research_store

    monkeypatch.setattr(research_store, "CHRONOVISOR_ROOT", tmp_path / "wiki")
    monkeypatch.setattr(research_scheduler, "RUNTIME_DIR", tmp_path / "scheduler")
    store = ResearchStore(root=tmp_path / "research")
    artifact = store.put_artifact(
        "claim text", source_type="chronovisor_read", source_uri="wiki:x", durable=True
    )
    bundle = build_bundle(
        run_id="run",
        claims=simple_assess_claims([("claim text", False)], [artifact]),
        artifacts=[artifact],
        store=store,
    )
    roles: list[str] = []

    def runner(operation, _prompt, _lease, _config):
        roles.append(operation)
        route = {
            "role": f"research.{operation}",
            "provider": "provider-test",
            "model": f"model-{operation}",
            "location": "local",
        }
        if operation == "challenge":
            return {
                "status": "completed",
                "route": route,
                "value": {
                    "verdict": "inconclusive",
                    "unsupported_claims": ["claim text"],
                    "contradictions": [],
                    "injection_detected": False,
                    "rationale": "weak evidence",
                },
            }
        return {
            "status": "completed",
            "route": route,
            "value": {"choice": "unknown", "rationale": "tie"},
        }

    result = challenge_bundle(
        bundle,
        config=ResearchConfig(enabled=True, mode="explicit"),
        store=store,
        runner=runner,
    )
    assert roles == ["challenge", "tie_break"]
    assert result["usage"]["challenge_calls"] == 1
    assert result["usage"]["tie_break_calls"] == 1
    assert result["route"]["role"] == "research.challenge"
    assert result["tie_route"]["role"] == "research.tie_break"
    event = store.events("run")[-1]
    assert event["kind"] == "evidence_challenge"
    assert event["result"]["route"] == result["route"]
    assert event["result"]["tie_route"] == result["tie_route"]

    malformed_calls: list[str] = []

    def malformed_runner(operation, _prompt, _lease, _config):
        malformed_calls.append(operation)
        return {
            "status": "completed",
            "value": {"verdict": "invalid", "contradictions": ["x"]},
        }

    malformed = challenge_bundle(
        bundle,
        config=ResearchConfig(enabled=True, mode="explicit"),
        store=store,
        runner=malformed_runner,
    )
    assert malformed_calls == ["challenge"]
    assert malformed["usage"]["tie_break_calls"] == 0
