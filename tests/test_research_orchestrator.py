from __future__ import annotations

from llm_wiki_mcp import research_orchestrator, research_scheduler
from llm_wiki_mcp.research_config import CompactionConfig, ResearchConfig
from llm_wiki_mcp.research_orchestrator import (
    DeterministicPlanner,
    LocalPlanner,
    PlannerResponse,
    ResearchState,
)
from llm_wiki_mcp.research_scheduler import (
    CancellableResult,
    ResearchAdmission,
    ResearchLease,
)
from llm_wiki_mcp.research_types import ResearchBudget
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
    assert any(
        row.get("kind") == "malformed_action" and row.get("terminal") is True
        for row in store.events(result["research_run_id"])
    )


def test_action_contract_rejects_wrong_arguments_before_execution(
    tmp_path, monkeypatch
) -> None:
    _isolate_scheduler(tmp_path, monkeypatch)
    monkeypatch.setattr(
        research_orchestrator,
        "execute_tool",
        lambda *_args: (_ for _ in ()).throw(AssertionError("invalid action executed")),
    )

    class WrongArguments:
        needs_model = False

        def plan(self, *args, **kwargs):
            return PlannerResponse(
                {
                    "type": "wiki_read",
                    "arguments": {"query": "topic", "url": "https://example.test"},
                    "rationale": "read",
                }
            )

    result = research_orchestrator.run_research(
        "goal",
        config=ResearchConfig(enabled=True, mode="trace"),
        planner=WrongArguments(),
        store=ResearchStore(tmp_path / "store"),
    )

    assert result["stop_reason"] == "malformed_action"
    assert result["actions"] == 0
    assert result["invalid_action_executions"] == 0


def test_local_planner_preserves_transport_failure_class(monkeypatch) -> None:
    calls = []

    def run_worker(*_args, **kwargs):
        calls.append(kwargs)
        return CancellableResult(
            "completed",
            value={
                "ok": False,
                "failure_class": "transport_error",
                "failure_reason": "connection reset",
                "repair_turns": 0,
            },
            latency_ms=123,
        )

    monkeypatch.setattr(
        research_orchestrator,
        "run_cancellable_command",
        run_worker,
    )
    lease = ResearchLease(
        ResearchAdmission(True, "admitted", "run-a"),
        0.0,
    )

    response = LocalPlanner("planner-model").plan(
        ResearchState("run-a", "goal"),
        lease=lease,
        budget=ResearchBudget(),
        events=[],
    )

    assert response.status == "error"
    assert response.error == "transport_error: connection reset"
    assert response.latency_ms == 123
    assert calls[0]["timeout_seconds"] == 90.0


def test_local_planner_session_timeout_respects_run_deadline(monkeypatch) -> None:
    calls = []

    def run_worker(*_args, **kwargs):
        calls.append(kwargs)
        return CancellableResult("timeout", error="deadline")

    monkeypatch.setattr(
        research_orchestrator,
        "run_cancellable_command",
        run_worker,
    )
    state = ResearchState(
        "run-deadline",
        "goal",
        deadline_monotonic=research_orchestrator.time.monotonic() + 5,
    )
    lease = ResearchLease(
        ResearchAdmission(True, "admitted", "run-deadline"),
        0.0,
    )

    response = LocalPlanner("planner-model").plan(
        state,
        lease=lease,
        budget=ResearchBudget(),
        events=[],
    )

    assert response.status == "timeout"
    assert 0 < calls[0]["timeout_seconds"] <= 5


def test_transport_error_retries_without_counting_malformed(
    tmp_path, monkeypatch
) -> None:
    _isolate_scheduler(tmp_path, monkeypatch)

    class TransientPlanner:
        needs_model = False

        def plan(self, state, **_kwargs):
            if state.iterations == 1:
                return PlannerResponse(
                    None,
                    status="error",
                    first_pass_valid=False,
                    error="transport_error: connection reset",
                )
            return PlannerResponse(
                {
                    "type": "finish",
                    "arguments": {"answer": "recovered"},
                }
            )

    store = ResearchStore(tmp_path / "store")
    result = research_orchestrator.run_research(
        "goal",
        config=ResearchConfig(enabled=True, mode="trace"),
        planner=TransientPlanner(),
        store=store,
    )
    events = store.events(result["research_run_id"])

    assert result["stop_reason"] == "completed"
    assert result["answer"] == "recovered"
    assert result["first_pass_malformed"] == 0
    assert result["usage"]["planner_calls"] == 2
    assert any(
        row.get("kind") == "planner_error" and row.get("terminal") is True
        for row in events
    )


def test_duplicate_search_recovers_by_reading_best_unseen_page(
    tmp_path, monkeypatch
) -> None:
    _isolate_scheduler(tmp_path, monkeypatch)

    class RepeatingPlanner:
        needs_model = False

        def plan(self, state, **_kwargs):
            if not state.actions:
                return PlannerResponse(
                    {"type": "wiki_search", "arguments": {"query": "topic"}}
                )
            if not any(action.type.value == "wiki_read" for action in state.actions):
                return PlannerResponse(
                    {"type": "wiki_search", "arguments": {"query": "topic"}}
                )
            return PlannerResponse(
                {"type": "finish", "arguments": {"answer": "recovered"}}
            )

    def tool(action, _context):
        if action.type.value == "wiki_search":
            return {"results": [{"page_id": "best-page"}]}
        return {
            "page_id": "best-page",
            "title": "Best Page",
            "body": "evidence",
            "citation": "wiki:best-page",
        }

    monkeypatch.setattr(research_orchestrator, "execute_tool", tool)
    store = ResearchStore(tmp_path / "store")
    result = research_orchestrator.run_research(
        "goal",
        config=ResearchConfig(enabled=True, mode="trace"),
        planner=RepeatingPlanner(),
        store=store,
    )
    events = store.events(result["research_run_id"])

    assert result["stop_reason"] == "completed"
    assert result["answer"] == "recovered"
    assert [
        (row.get("action") or {}).get("type")
        for row in events
        if row.get("kind") == "action"
    ] == [
        "wiki_search",
        "wiki_read",
        "finish",
    ]
    assert any(row.get("kind") == "duplicate_action_recovered" for row in events)


def test_restart_terminalizes_orphan_action_and_advances_epoch(
    tmp_path, monkeypatch
) -> None:
    _isolate_scheduler(tmp_path, monkeypatch)
    store = ResearchStore(tmp_path / "store")
    store.append_event(
        "resumed-run",
        {
            "kind": "action",
            "epoch": 0,
            "iteration": 1,
            "action": {"type": "wiki_search", "arguments": {"query": "old"}},
        },
    )
    monkeypatch.setattr(
        research_orchestrator,
        "execute_tool",
        lambda action, _context: (
            {"results": []} if action.type.value == "wiki_search" else {}
        ),
    )

    result = research_orchestrator.run_research(
        "goal",
        run_id="resumed-run",
        config=ResearchConfig(enabled=True, mode="trace"),
        planner=DeterministicPlanner(),
        store=store,
    )
    events = store.events("resumed-run")

    assert result["stop_reason"] == "completed"
    assert any(row.get("status") == "orphan_terminalized" for row in events)
    assert any(row.get("kind") == "action" and row.get("epoch") == 1 for row in events)


def test_large_observation_is_externalized_and_checkpoint_receipted(
    tmp_path, monkeypatch
) -> None:
    _isolate_scheduler(tmp_path, monkeypatch)
    store = ResearchStore(tmp_path / "store")
    store.checkpoints = tmp_path / "checkpoints"
    monkeypatch.setattr(
        research_orchestrator,
        "execute_tool",
        lambda action, _context: (
            {"results": [{"page_id": "x", "blob": "z" * 5000}]}
            if action.type.value == "wiki_search"
            else {}
        ),
    )
    config = ResearchConfig(
        enabled=True,
        mode="trace",
        compaction=CompactionConfig(
            enabled=True,
            checkpoint_enabled=True,
            checkpoint_ttl_seconds=60,
            checkpoint_max_total_bytes=1_000_000,
            gc_on_durable_receipt=True,
        ),
    )

    result = research_orchestrator.run_research(
        "goal", config=config, planner=DeterministicPlanner(), store=store
    )
    observations = [
        row
        for row in store.events(result["research_run_id"])
        if row.get("kind") == "observation"
    ]
    import json

    checkpoint = json.loads(next(store.checkpoints.glob("*.json")).read_text())

    assert observations[0]["artifact_id"].startswith("sha256:")
    assert store.read_artifact(observations[0]["artifact_id"])
    assert checkpoint["active"] is False
    assert checkpoint["durable_receipt"] is True
