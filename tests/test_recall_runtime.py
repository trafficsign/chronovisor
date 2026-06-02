from __future__ import annotations

import json

from llm_wiki_mcp.recall_runtime import (
    RecallPolicy,
    RecallRequest,
    append_feedback,
    evaluate_heuristic,
    render_output,
    request_from_hook_payload,
    run_recall,
)


def test_explicit_past_project_prompt_crosses_read_threshold() -> None:
    policy = RecallPolicy(judge_mode="off")
    request = RecallRequest(
        host="test",
        event="UserPromptSubmit",
        prompt="昨日LLM Wikiのフック直したやつ、Claude Codeにも入れられる?",
        cwd="/Users/trafficsign/projects/personal/llm-wiki-mcp",
    )

    score, reasons, matched = evaluate_heuristic(request, policy)

    assert score >= policy.read_threshold
    assert "昨日" in matched["past_reference"]
    assert any("known recurring" in reason for reason in reasons)


def test_simple_chitchat_stays_below_search_threshold() -> None:
    policy = RecallPolicy(judge_mode="off")
    request = RecallRequest(
        host="test",
        event="UserPromptSubmit",
        prompt="今日暑いな",
        cwd="/tmp",
    )

    score, reasons, _matched = evaluate_heuristic(request, policy)

    assert score < policy.search_threshold
    assert "simple chitchat" in reasons


def test_run_recall_without_search_returns_queries_for_gate_hit() -> None:
    policy = RecallPolicy(judge_mode="off", log_decisions=False)
    request = RecallRequest(
        host="test",
        event="UserPromptSubmit",
        prompt="前回のCodex hook設定の続き、どう実装する?",
        cwd="/Users/trafficsign/projects/personal/llm-wiki-mcp",
    )

    result = run_recall(request, policy, perform_search=False)

    assert result.decision == "read"
    assert result.queries
    assert result.confidence >= policy.read_threshold


def test_system_task_notification_skips_recall() -> None:
    policy = RecallPolicy(judge_mode="off", log_decisions=False)
    request = RecallRequest(
        host="test",
        event="UserPromptSubmit",
        prompt="<task-notification>Codex hook task completed for yesterday's LLM Wiki work.</task-notification>",
        cwd="/Users/trafficsign/projects/personal/llm-wiki-mcp",
    )

    result = run_recall(request, policy, perform_search=False)

    assert result.status == "skipped"
    assert result.decision == "none"
    assert result.queries == []
    assert "system notification prompt" in result.reasons


def test_recall_context_injection_skips_recall() -> None:
    policy = RecallPolicy(judge_mode="off", log_decisions=False)
    request = RecallRequest(
        host="test",
        event="UserPromptSubmit",
        prompt="[RECALL_CONTEXT]\npages:\n- systemheadertemplate\n[/RECALL_CONTEXT]",
        cwd="/Users/trafficsign/projects/personal/llm-wiki-mcp",
    )

    result = run_recall(request, policy, perform_search=False)

    assert result.status == "skipped"
    assert result.decision == "none"
    assert result.queries == []
    assert "recall context injection" in result.reasons


def test_codex_internal_suggestion_prompt_skips_recall() -> None:
    policy = RecallPolicy(judge_mode="off", log_decisions=False)
    request = RecallRequest(
        host="codex",
        event="UserPromptSubmit",
        prompt=(
            "# Overview\n\n"
            "Generate 0 to 3 hyperpersonalized suggestions for what this user can do "
            "with Codex in this local project."
        ),
        cwd="/Users/trafficsign/Documents/Codex/2026-05-02/new-chat",
    )

    result = run_recall(request, policy, perform_search=False)

    assert result.status == "skipped"
    assert result.decision == "none"
    assert result.queries == []
    assert "codex internal suggestion prompt" in result.reasons


def test_user_discussing_task_notification_is_not_filtered() -> None:
    policy = RecallPolicy(judge_mode="off", log_decisions=False)
    request = RecallRequest(
        host="test",
        event="UserPromptSubmit",
        prompt="さっきの <task-notification> の recall 誤発火をどう直す?",
        cwd="/Users/trafficsign/projects/personal/llm-wiki-mcp",
    )

    result = run_recall(request, policy, perform_search=False)

    assert result.decision != "none"
    assert result.queries


def test_trailing_system_block_is_stripped_before_recall_gate() -> None:
    policy = RecallPolicy(judge_mode="off", log_decisions=False)
    request = RecallRequest(
        host="test",
        event="UserPromptSubmit",
        prompt=(
            "昨日LLM Wikiの recall hook の続き、どう直す?\n"
            "<task-notification>\n"
            "<summary>sdksintro systemheadertemplate internal-model-paradox</summary>\n"
            "</task-notification>"
        ),
        cwd="/Users/trafficsign/projects/personal/llm-wiki-mcp",
    )

    result = run_recall(request, policy, perform_search=False)

    assert result.decision == "read"
    assert "stripped system notification block" in result.reasons
    assert result.queries
    assert all("task-notification" not in query for query in result.queries)
    assert all("systemheadertemplate" not in query for query in result.queries)


def test_middle_recall_context_block_is_stripped_before_recall_gate() -> None:
    policy = RecallPolicy(judge_mode="off", log_decisions=False)
    request = RecallRequest(
        host="test",
        event="UserPromptSubmit",
        prompt=(
            "昨日LLM Wikiの recall hook の続き。\n"
            "[RECALL_CONTEXT]\n"
            "pages:\n"
            "- systemheadertemplate\n"
            "[/RECALL_CONTEXT]\n"
            "この誤発火をどう直す?"
        ),
        cwd="/Users/trafficsign/projects/personal/llm-wiki-mcp",
    )

    result = run_recall(request, policy, perform_search=False)

    assert result.decision == "read"
    assert "stripped recall context block" in result.reasons
    assert result.queries
    assert all("RECALL_CONTEXT" not in query for query in result.queries)
    assert all("systemheadertemplate" not in query for query in result.queries)


def test_feedback_exact_false_positive_suppresses_repeat(tmp_path, monkeypatch) -> None:
    from llm_wiki_mcp import recall_runtime

    feedback_file = tmp_path / "feedback.jsonl"
    monkeypatch.setattr(recall_runtime, "RECALL_FEEDBACK_FILE", feedback_file)
    prompt = "昨日LLM Wikiの recall hook が誤発火した内部通知"
    append_feedback("false-positive", "exact repeat should not recall", prompt=prompt, host="test")

    policy = RecallPolicy(judge_mode="off", log_decisions=False)
    result = run_recall(
        RecallRequest(host="test", event="UserPromptSubmit", prompt=prompt),
        policy,
        perform_search=False,
    )

    assert result.status == "skipped"
    assert result.decision == "none"
    assert "feedback false-positive prompt" in result.reasons


def test_feedback_machine_tag_suppresses_same_tag(tmp_path, monkeypatch) -> None:
    from llm_wiki_mcp import recall_runtime

    feedback_file = tmp_path / "feedback.jsonl"
    monkeypatch.setattr(recall_runtime, "RECALL_FEEDBACK_FILE", feedback_file)
    append_feedback(
        "false-positive",
        "machine tag should become suppressible",
        prompt="<tool-result>old noisy result</tool-result>",
        host="test",
    )

    policy = RecallPolicy(judge_mode="off", log_decisions=False)
    result = run_recall(
        RecallRequest(
            host="test",
            event="UserPromptSubmit",
            prompt="<tool-result>new noisy result about yesterday LLM Wiki</tool-result>",
        ),
        policy,
        perform_search=False,
    )

    assert result.status == "skipped"
    assert result.decision == "none"
    assert "feedback false-positive tag <tool-result>" in result.reasons


def test_hook_payload_accepts_claude_and_codex_prompt_keys() -> None:
    claude = request_from_hook_payload(
        {"user_prompt": "昨日のあれ", "cwd": "/a", "session_id": "s1"},
        host="claude-code",
        event="UserPromptSubmit",
    )
    codex = request_from_hook_payload(
        {"prompt": "前回の続き", "working_directory": "/b", "thread_id": "t1"},
        host="codex",
        event="UserPromptSubmit",
    )

    assert claude.prompt == "昨日のあれ"
    assert claude.cwd == "/a"
    assert claude.session_id == "s1"
    assert codex.prompt == "前回の続き"
    assert codex.cwd == "/b"
    assert codex.session_id == "t1"


def test_codex_output_uses_hook_json_when_context_exists() -> None:
    policy = RecallPolicy(judge_mode="off", log_decisions=False)
    request = RecallRequest(
        host="test",
        event="UserPromptSubmit",
        prompt="前回のLLM Wiki設計の続き",
    )
    result = run_recall(request, policy, perform_search=False)
    result.context = "[RECALL_CONTEXT]\nhello\n[/RECALL_CONTEXT]"

    rendered = render_output(result, "codex")
    parsed = json.loads(rendered)

    assert parsed["systemMessage"]
    assert parsed["hookSpecificOutput"]["additionalContext"] == result.context


def test_feedback_writer_uses_configurable_path(tmp_path, monkeypatch) -> None:
    from llm_wiki_mcp import recall_runtime

    feedback_file = tmp_path / "feedback.jsonl"
    monkeypatch.setattr(recall_runtime, "RECALL_FEEDBACK_FILE", feedback_file)

    record = append_feedback("missed", "前回の話を検索しなかった", prompt="昨日のあれ")

    assert record["kind"] == "missed"
    assert feedback_file.exists()
    assert "前回の話" in feedback_file.read_text()
