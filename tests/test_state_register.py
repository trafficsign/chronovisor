from __future__ import annotations

import json

from llm_wiki_mcp import recall_runtime
from llm_wiki_mcp.recall_runtime import RecallPolicy, RecallRequest, render_output, run_recall


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
