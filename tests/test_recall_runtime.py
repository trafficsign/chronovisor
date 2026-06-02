from __future__ import annotations

import json

from llm_wiki_mcp.recall_runtime import (
    RecallPolicy,
    RecallRequest,
    append_feedback,
    evaluate_heuristic,
    load_policy,
    render_output,
    request_from_hook_payload,
    run_local_judge,
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


def test_judge_timeout_fails_silent_in_search_zone(monkeypatch) -> None:
    from llm_wiki_mcp import recall_runtime

    def timeout_judge(*_args: object, **_kwargs: object) -> tuple[None, list[str], str]:
        return None, [], "judge unavailable: ReadTimeout"

    monkeypatch.setattr(recall_runtime, "run_local_judge", timeout_judge)
    policy = RecallPolicy(judge_mode="auto", log_decisions=False)
    request = RecallRequest(
        host="test",
        event="UserPromptSubmit",
        prompt="LLM Wiki の運用どうする?",
        cwd="/Users/trafficsign/projects/personal/llm-wiki-mcp",
    )

    result = recall_runtime.run_recall(request, policy, perform_search=False)

    assert result.status == "skipped"
    assert result.decision == "none"
    assert result.queries == []
    assert result.confidence >= policy.search_threshold
    assert "judge unavailable: ReadTimeout" in result.reasons
    assert "judge unavailable; fail-silent" in result.reasons


def test_judge_timeout_can_fall_back_when_fail_silent_disabled(monkeypatch) -> None:
    from llm_wiki_mcp import recall_runtime

    def timeout_judge(*_args: object, **_kwargs: object) -> tuple[None, list[str], str]:
        return None, [], "judge unavailable: ReadTimeout"

    monkeypatch.setattr(recall_runtime, "run_local_judge", timeout_judge)
    policy = RecallPolicy(
        judge_mode="auto",
        log_decisions=False,
        fail_silent_on_judge_unavailable=False,
    )
    request = RecallRequest(
        host="test",
        event="UserPromptSubmit",
        prompt="LLM Wiki の運用どうする?",
        cwd="/Users/trafficsign/projects/personal/llm-wiki-mcp",
    )

    result = recall_runtime.run_recall(request, policy, perform_search=False)

    assert result.decision == "search"
    assert result.queries
    assert "judge unavailable: ReadTimeout" in result.reasons


def test_judge_can_lower_search_zone_to_none(monkeypatch) -> None:
    from llm_wiki_mcp import recall_runtime

    def no_recall_judge(*_args: object, **_kwargs: object) -> tuple[float, list[str], str]:
        return 0.2, [], "不要"

    monkeypatch.setattr(recall_runtime, "run_local_judge", no_recall_judge)
    policy = RecallPolicy(judge_mode="auto", log_decisions=False)
    request = RecallRequest(
        host="test",
        event="UserPromptSubmit",
        prompt="LLM Wiki の運用どうする?",
        cwd="/Users/trafficsign/projects/personal/llm-wiki-mcp",
    )

    result = recall_runtime.run_recall(request, policy, perform_search=False)

    assert result.decision == "none"
    assert result.confidence == 0.2
    assert result.queries == []
    assert "judge: 不要" in result.reasons


def test_obvious_read_does_not_wait_for_auto_judge(monkeypatch) -> None:
    from llm_wiki_mcp import recall_runtime

    def unexpected_judge(*_args: object, **_kwargs: object) -> tuple[None, list[str], str]:
        raise AssertionError("auto judge should not run for obvious read prompts")

    monkeypatch.setattr(recall_runtime, "run_local_judge", unexpected_judge)
    policy = RecallPolicy(judge_mode="auto", log_decisions=False)
    request = RecallRequest(
        host="test",
        event="UserPromptSubmit",
        prompt="昨日LLM Wikiのフック直したやつ、Claude Codeにも入れられる?",
        cwd="/Users/trafficsign/projects/personal/llm-wiki-mcp",
    )

    result = recall_runtime.run_recall(request, policy, perform_search=False)

    assert result.decision == "read"
    assert result.confidence >= policy.read_threshold


def test_gate_config_overrides_legacy_model_and_budget(tmp_path) -> None:
    config = tmp_path / "recall.toml"
    config.write_text(
        """
enabled = true
model = "qwen3.6:35b-a3b-q8_0"

[budgets]
judge_timeout_ms = 4000

[gate]
model = "qwen3.5:4b"
think = false
timeout_ms = 1200
num_ctx = 2048
num_predict = 128
"""
    )

    policy = load_policy(config)

    assert policy.judge_model == "qwen3.5:4b"
    assert policy.judge_think is False
    assert policy.judge_timeout_ms == 1200
    assert policy.judge_num_ctx == 2048
    assert policy.judge_num_predict == 128


def test_local_judge_uses_gate_generation_options(monkeypatch) -> None:
    from llm_wiki_mcp import recall_runtime

    captured: dict[str, object] = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, str]:
            return {
                "response": json.dumps(
                    {"decision": "none", "confidence": 0.2, "reason": "不要", "queries": []},
                    ensure_ascii=False,
                )
            }

    class FakeClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            captured["client_args"] = args
            captured["client_kwargs"] = kwargs

        def __enter__(self) -> "FakeClient":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def post(self, path: str, *, json: dict[str, object]) -> FakeResponse:
            captured["path"] = path
            captured["payload"] = json
            return FakeResponse()

    monkeypatch.setattr(recall_runtime.httpx, "Client", FakeClient)
    policy = RecallPolicy(
        judge_model="qwen3.5:4b",
        judge_think=False,
        judge_timeout_ms=1200,
        judge_num_ctx=2048,
        judge_num_predict=128,
    )

    score, queries, reason = run_local_judge(
        RecallRequest(host="test", event="UserPromptSubmit", prompt="LLM Wiki の運用どうする?"),
        0.5,
        policy,
    )

    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert score == 0.2
    assert queries == []
    assert reason == "不要"
    assert captured["path"] == "/api/generate"
    assert payload["model"] == "qwen3.5:4b"
    assert payload["think"] is False
    assert payload["options"] == {
        "temperature": 0,
        "num_ctx": 2048,
        "num_predict": 128,
    }


def test_local_judge_decision_bounds_confidence(monkeypatch) -> None:
    from llm_wiki_mcp import recall_runtime

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, str]:
            return {
                "response": json.dumps(
                    {"decision": "none", "confidence": 0.9, "reason": "不要"},
                    ensure_ascii=False,
                )
            }

    class FakeClient:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            return None

        def __enter__(self) -> "FakeClient":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def post(self, _path: str, *, json: dict[str, object]) -> FakeResponse:
            return FakeResponse()

    monkeypatch.setattr(recall_runtime.httpx, "Client", FakeClient)
    policy = RecallPolicy(judge_model="qwen3.5:4b", judge_timeout_ms=2000)

    score, _queries, reason = run_local_judge(
        RecallRequest(host="test", event="UserPromptSubmit", prompt="LLM Wiki の運用どうする?"),
        0.5,
        policy,
    )

    assert score < policy.search_threshold
    assert reason == "不要"


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
