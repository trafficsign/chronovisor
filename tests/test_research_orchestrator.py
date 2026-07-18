from __future__ import annotations

from llm_wiki_mcp import research_orchestrator, research_scheduler
from llm_wiki_mcp.research_config import ResearchConfig
from llm_wiki_mcp.research_orchestrator import DeterministicPlanner, PlannerResponse
from llm_wiki_mcp.research_store import ResearchStore


def _isolate_scheduler(tmp_path, monkeypatch) -> None:
    root = tmp_path / "scheduler"
    monkeypatch.setattr(research_scheduler, "RUNTIME_DIR", root)
    monkeypatch.setattr(research_scheduler, "SYNC_DIR", root / "sync")
    monkeypatch.setattr(research_scheduler, "RESEARCH_LOCK", root / "lock")
    monkeypatch.setattr(research_scheduler, "ACTIVE_FILE", root / "active.json")
    monkeypatch.setattr(research_scheduler, "SCHEDULER_LOG", root / "log.jsonl")


def test_deterministic_kernel_terminalizes_every_action(tmp_path, monkeypatch) -> None:
    _isolate_scheduler(tmp_path, monkeypatch)

    def tool(action, _context):
        if action.type.value == "wiki_search":
            return {"results": [{"page_id": "page-a"}]}
        return {"page_id": "page-a", "body": "evidence"}

    monkeypatch.setattr(research_orchestrator, "execute_tool", tool)
    store = ResearchStore(tmp_path / "store")
    result = research_orchestrator.run_research(
        "goal",
        config=ResearchConfig(enabled=True, mode="trace"),
        planner=DeterministicPlanner(),
        store=store,
    )
    events = store.events(result["research_run_id"])
    actions = [row for row in events if row.get("kind") == "action"]
    observations = [row for row in events if row.get("kind") == "observation"]

    assert result["stop_reason"] == "completed"
    assert len(actions) == len(observations)
    assert result["invalid_action_executions"] == 0


def test_malformed_action_is_terminal_and_never_executed(tmp_path, monkeypatch) -> None:
    _isolate_scheduler(tmp_path, monkeypatch)
    monkeypatch.setattr(
        research_orchestrator,
        "execute_tool",
        lambda *_args: (_ for _ in ()).throw(AssertionError("invalid action executed")),
    )

    class Malformed:
        needs_model = False

        def plan(self, *args, **kwargs):
            return PlannerResponse({"type": "shell", "arguments": {}})

    store = ResearchStore(tmp_path / "store")
    result = research_orchestrator.run_research(
        "goal",
        config=ResearchConfig(enabled=True, mode="trace"),
        planner=Malformed(),
        store=store,
    )

    assert result["stop_reason"] == "malformed_action"
    assert any(row.get("kind") == "malformed_action" and row.get("terminal") is True for row in store.events(result["research_run_id"]))
