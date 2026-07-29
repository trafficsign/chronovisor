from __future__ import annotations

import json
from pathlib import Path

from chronovisor.ops import health


def test_research_kpi_surfaces_trace_claims_and_kill_switches(tmp_path: Path, monkeypatch) -> None:
    wiki = tmp_path / "wiki"
    run = wiki / "runtime" / "research" / "runs" / "run-1"
    run.mkdir(parents=True)
    (run / "summary.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "research_run_id": "run-1",
                "stop_reason": "completed",
                "actions": 1,
                "observations": 1,
                "first_pass_malformed": 1,
                "repair_turns": 1,
                "invalid_action_executions": 0,
                "usage": {"observation_bytes": 100},
                "claims": [
                    {"claim": "a", "status": "supported"},
                    {"claim": "b", "status": "unknown"},
                ],
            }
        ),
        encoding="utf-8",
    )
    (run / "events.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"kind": "action", "iteration": 1}),
                json.dumps({"kind": "observation", "iteration": 1, "metadata": {"provider": "fixture", "cache": "hit"}}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(health, "CHRONOVISOR_ROOT", wiki)
    monkeypatch.setenv("CHRONOVISOR_RESEARCH_ENABLED", "0")
    monkeypatch.setenv("CHRONOVISOR_RESEARCH_MODE", "off")
    result = health.research_kpi()
    assert result["totals"]["runs"] == 1
    assert result["totals"]["supported_claims"] == 1
    assert result["totals"]["unknown_claims"] == 1
    assert result["decision_trace_coverage"] == 1.0
    assert result["providers"] == {"fixture": 1}
    assert result["kill_switches"]["agent"] is True
