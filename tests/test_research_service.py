from __future__ import annotations

from pathlib import Path

import pytest

from chronovisor.research import research_service
from chronovisor.research.research_orchestrator import PlannerResponse
from chronovisor.research.research_service import run_evidence_research
from chronovisor.search.research_config import ResearchConfig
from chronovisor.search.research_store import ResearchStore


@pytest.fixture(autouse=True)
def _valid_okf_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "okf-root"
    root.mkdir()
    for name in ("index.md", "log.md", "schema.md"):
        (root / name).write_text("legacy\n", encoding="utf-8")
    monkeypatch.setattr(research_service, "CHRONOVISOR_ROOT", root)


def test_service_writes_bundle_audit_and_receipt(tmp_path: Path, monkeypatch) -> None:
    from chronovisor.core import research_scheduler
    from chronovisor.research import research_auditor
    from chronovisor.search import research_store

    monkeypatch.setattr(research_store, "CHRONOVISOR_ROOT", tmp_path / "wiki")
    monkeypatch.setattr(research_scheduler, "RUNTIME_DIR", tmp_path / "scheduler")
    monkeypatch.setattr(research_auditor, "AUDIT_LOG", tmp_path / "audit.jsonl")
    store = ResearchStore(root=tmp_path / "research")

    class FinishPlanner:
        needs_model = False

        def plan(self, *_args, **_kwargs):
            return PlannerResponse(
                {
                    "type": "finish",
                    "arguments": {"answer": "unknown"},
                    "rationale": "test",
                }
            )

    result = run_evidence_research(
        "no local evidence",
        config=ResearchConfig(enabled=True, mode="explicit"),
        planner=FinishPlanner(),
        challenge=False,
        run_id="service-run",
        store=store,
    )
    assert result["evidence_bundle_id"].startswith("bundle:")
    assert result["claims"][0]["status"] == "unknown"
    assert any(
        row.get("kind") == "durable_receipt" for row in store.events("service-run")
    )
    assert (tmp_path / "wiki" / "research" / "bundles" / "service-run.json").exists()


def test_service_never_returns_empty_answer_after_planner_terminal(
    tmp_path: Path, monkeypatch
) -> None:
    from chronovisor.research import research_auditor

    monkeypatch.setattr(research_auditor, "AUDIT_LOG", tmp_path / "audit.jsonl")
    store = ResearchStore(root=tmp_path / "research")
    artifact = store.put_artifact(
        '{"body":"chronovisor_research is published by fresh MCP"}',
        source_type="chronovisor_read",
        source_uri="wiki:mcp-publication",
        title="MCP Publication",
        citation="wiki:mcp-publication",
        trust="local",
        durable=True,
    )
    monkeypatch.setattr(
        research_service,
        "run_research",
        lambda *_args, **_kwargs: {
            "schema_version": 1,
            "status": "terminal",
            "research_run_id": "terminal-run",
            "goal": "check publication",
            "answer": "",
            "stop_reason": "duplicate_action",
            "usage": {},
            "artifact_ids": [artifact.artifact_id],
        },
    )

    result = run_evidence_research(
        "check publication",
        claims=["chronovisor_research is published by fresh MCP"],
        config=ResearchConfig(enabled=True, mode="explicit"),
        challenge=False,
        run_id="terminal-run",
        store=store,
    )

    assert result["answer"]
    assert result["answer_mode"] == "deterministic_claim_assessment"
    assert "[supported]" in result["answer"]


def test_cli_defaults_to_durable_background_queue(monkeypatch, capsys) -> None:
    seen: dict[str, object] = {}

    def fake_enqueue(goal, *, claims, challenge, purpose):
        seen.update(
            goal=goal,
            claims=claims,
            challenge=challenge,
            purpose=purpose,
        )
        return {"status": "queued", "job_id": "job-1"}

    monkeypatch.setattr(research_service, "enqueue_evidence_research", fake_enqueue)

    assert (
        research_service.main(
            ["latest claim", "--claim", "claim one", "--no-challenge", "--json"]
        )
        == 0
    )

    assert seen == {
        "goal": "latest claim",
        "claims": ["claim one"],
        "challenge": False,
        "purpose": "explicit",
    }
    assert '"status": "queued"' in capsys.readouterr().out


def test_cli_sync_is_explicit(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        research_service,
        "run_evidence_research",
        lambda goal, **kwargs: {
            "status": "completed",
            "goal": goal,
            "challenge": kwargs["challenge"],
        },
    )

    assert research_service.main(["local evidence", "--sync", "--json"]) == 0
    payload = capsys.readouterr().out
    assert '"status": "completed"' in payload
    assert '"goal": "local evidence"' in payload


def test_server_publishes_chronovisor_research() -> None:
    from chronovisor.hosts import server

    names = {tool.name for tool in server.mcp._tool_manager.list_tools()}

    assert "chronovisor_research" in names
