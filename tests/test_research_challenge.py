from __future__ import annotations

from pathlib import Path

from chronovisor.evidence_bundle import build_bundle, simple_assess_claims
from chronovisor.research_challenge import challenge_bundle
from chronovisor.research_config import ResearchConfig
from chronovisor.research_store import ResearchStore


def test_disagreement_calls_tie_break_with_role_budgets(tmp_path: Path, monkeypatch) -> None:
    from chronovisor import research_scheduler, research_store

    monkeypatch.setattr(research_store, "CHRONOVISOR_ROOT", tmp_path / "wiki")
    monkeypatch.setattr(research_scheduler, "RUNTIME_DIR", tmp_path / "scheduler")
    store = ResearchStore(root=tmp_path / "research")
    artifact = store.put_artifact("claim text", source_type="chronovisor_read", source_uri="wiki:x", durable=True)
    bundle = build_bundle(
        run_id="run",
        claims=simple_assess_claims([("claim text", False)], [artifact]),
        artifacts=[artifact],
        store=store,
    )
    roles: list[str] = []

    def runner(role, _model, _prompt, _schema, _lease, _config):
        roles.append(role)
        if role == "research_challenge":
            return {
                "status": "completed",
                "value": {
                    "verdict": "inconclusive",
                    "unsupported_claims": ["claim text"],
                    "contradictions": [],
                    "injection_detected": False,
                    "rationale": "weak evidence",
                },
            }
        return {"status": "completed", "value": {"choice": "unknown", "rationale": "tie"}}

    result = challenge_bundle(
        bundle,
        config=ResearchConfig(enabled=True, mode="explicit"),
        store=store,
        runner=runner,
    )
    assert roles == ["research_challenge", "research_tie_break"]
    assert result["usage"]["challenge_calls"] == 1
    assert result["usage"]["tie_break_calls"] == 1
