from __future__ import annotations

import json
from pathlib import Path

from llm_wiki_mcp import recall_runtime
from llm_wiki_mcp.recall_runtime import RecallPolicy, RecallRequest, render_output, run_recall
from llm_wiki_mcp import state_register


def test_state_register_context_is_injected_for_codex(monkeypatch) -> None:
    monkeypatch.setattr(recall_runtime, "should_inject_state", lambda host: host == "codex")
    monkeypatch.setattr(
        recall_runtime,
        "format_state_context",
        lambda *, host, cwd: "[WORKING_MEMORY]\ncurrent project state\n[/WORKING_MEMORY]",
    )
    request = RecallRequest(host="codex", event="UserPromptSubmit", prompt="うん", cwd="/tmp")
    result = run_recall(request, RecallPolicy(judge_mode="off", log_decisions=False), perform_search=False)

    assert result.decision == "none"
    assert "current project state" in result.context
    assert "state register injected" in result.reasons

    output = json.loads(render_output(result, "codex"))
    assert output["hookSpecificOutput"]["additionalContext"] == result.context


def test_format_state_context_marks_stale_state(tmp_path: Path) -> None:
    path = tmp_path / "current-state.md"
    path.write_text(
        "---\ntitle: Current State\nupdated: 2026-04-17\n---\n# Current State\n\nold body",
        encoding="utf-8",
    )

    context = state_register.format_state_context(host="codex", cwd="/tmp", path=path)

    assert "updated=2026-04-17" in context
    assert "stale=true" in context
    assert "old body" in context


def test_refresh_state_register_writes_recent_pages(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "current-state.md"

    class FakeStore:
        def refresh(self) -> None:
            pass

        def meta(self, page_id: str):
            return {
                "page_id": page_id,
                "title": "Recent Page",
                "summary": "Summary",
                "updated": "2026-07-06",
                "page_type": "knowledge",
            }

        def all_pages_meta(self, include_system: bool = False):
            return []

    monkeypatch.setattr("llm_wiki_mcp.index_store.get_store", lambda: FakeStore())

    payload = state_register.refresh_state_register(["recent-page"], path=path)

    assert payload["pages"] == ["recent-page"]
    assert "[[recent-page]]" in path.read_text(encoding="utf-8")
