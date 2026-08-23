from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from chronovisor.research import research_challenge
from chronovisor.research.evidence_bundle import build_bundle, simple_assess_claims
from chronovisor.research.research_challenge import challenge_bundle
from chronovisor.search.research_config import ResearchConfig
from chronovisor.search.research_store import ResearchStore


def test_default_runner_budgets_repairs_inside_total_wall_time(monkeypatch) -> None:
    captured: dict[str, object] = {}
    route = SimpleNamespace(
        role="research.challenge",
        provider="omlx",
        model="Muse-Glimmer-30B-4bit",
        location="local",
        structured_output=True,
    )
    monkeypatch.setattr(
        research_challenge.ollama,
        "runtime_generation_routes",
        lambda _roles: (route,),
    )

    def run_command(_command, payload, _lease, *, timeout_seconds):
        captured["request"] = json.loads(payload)
        captured["timeout_seconds"] = timeout_seconds
        return SimpleNamespace(status="completed", value={"ok": True}, latency_ms=1)

    monkeypatch.setattr(research_challenge, "run_cancellable_command", run_command)

    config = ResearchConfig(enabled=True, mode="explicit")
    config = replace(
        config,
        budgets=replace(config.budgets, max_repair_calls=0),
    )
    result = research_challenge._default_runner(
        "challenge",
        "evidence",
        object(),
        config,
    )

    assert result["status"] == "completed"
    assert captured["request"]["num_predict"] == 512
    assert captured["timeout_seconds"] == 90.0


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
    wall_seconds: list[float] = []
    clock = iter((100.0, 110.0, 150.0, 200.0, 210.0, 300.0, 310.0))
    monkeypatch.setattr(
        research_challenge,
        "time",
        SimpleNamespace(monotonic=lambda: next(clock)),
        raising=False,
    )

    def runner(operation, _prompt, _lease, runner_config):
        roles.append(operation)
        wall_seconds.append(runner_config.budgets.max_total_wall_seconds)
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
            "repair_turns": 1,
            "value": {"choice": "unknown", "rationale": "tie"},
        }

    result = challenge_bundle(
        bundle,
        config=ResearchConfig(enabled=True, mode="explicit"),
        store=store,
        runner=runner,
    )
    assert roles == ["challenge", "tie_break"]
    assert wall_seconds == [80.0, 40.0]
    assert result["usage"]["challenge_calls"] == 1
    assert result["usage"]["tie_break_calls"] == 1
    assert result["usage"]["repair_calls"] == 1
    assert result["route"]["role"] == "research.challenge"
    assert result["tie_route"]["role"] == "research.tie_break"
    event = store.events("run")[-1]
    assert event["kind"] == "evidence_challenge"
    assert event["result"]["route"] == result["route"]
    assert event["result"]["tie_route"] == result["tie_route"]

    def over_budget_runner(_operation, _prompt, _lease, _config):
        return {
            "status": "completed",
            "repair_turns": 1,
            "value": {
                "verdict": "confirm",
                "unsupported_claims": [],
                "contradictions": [],
                "injection_detected": False,
                "rationale": "repaired",
            },
        }

    no_repairs = ResearchConfig(enabled=True, mode="explicit")
    no_repairs = replace(
        no_repairs,
        budgets=replace(no_repairs.budgets, max_repair_calls=0),
    )
    over_budget = challenge_bundle(
        bundle,
        config=no_repairs,
        store=store,
        runner=over_budget_runner,
    )
    assert over_budget["status"] == "skipped"
    assert over_budget["reason"] == "repair_budget_exhausted"

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
