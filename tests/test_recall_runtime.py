from __future__ import annotations

import json

import pytest

from llm_wiki_mcp.recall_runtime import (
    ContextItem,
    RecallPolicy,
    RecallRequest,
    RecallResult,
    append_feedback,
    best_excerpt_index,
    build_queries,
    evaluate_heuristic,
    excerpt_terms,
    format_recall_context,
    load_policy,
    main,
    render_output,
    request_from_hook_payload,
    run_local_judge,
    run_query_rewriter,
    run_recall,
    search_candidates,
    warm_recall_model,
)
from llm_wiki_mcp.search import ScoredPage


@pytest.fixture(autouse=True)
def disable_live_recall_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_WIKI_RECALL_IMPROVEMENT_POLICY", "0")


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


def test_ascii_term_matching_uses_word_boundaries() -> None:
    policy = RecallPolicy(judge_mode="off")
    request = RecallRequest(
        host="test",
        event="UserPromptSubmit",
        prompt="LLM Wiki runtime と uvx cache の注意点",
    )

    _score, _reasons, matched = evaluate_heuristic(request, policy)

    assert "me " not in matched["ownership"]
    assert "llm wiki" in matched["project"]
    assert "uvx" in matched["project"]


def test_build_queries_does_not_add_single_generic_decision_term() -> None:
    policy = RecallPolicy(max_queries=3)
    request = RecallRequest(
        host="test",
        event="UserPromptSubmit",
        prompt="マツダでCADグループから設計グループへ移った話",
    )
    matched = {
        "project": [],
        "decision": ["設計"],
        "past_reference": [],
        "ownership": [],
        "ambiguity": [],
    }

    queries = build_queries(request, matched, [], policy)

    assert queries == ["マツダでCADグループから設計グループへ移った話"]


def test_search_candidates_prefers_specific_earlier_query(monkeypatch) -> None:
    from llm_wiki_mcp import recall_runtime

    def fake_search(
        *,
        query: str,
        top_n: int,
        semantic: bool,
        fusion_weights: dict[str, float],
    ) -> tuple[list[ScoredPage], str]:
        if query == "specific":
            return [ScoredPage("target", "target", "", "", 0.08)], "hybrid"
        return [ScoredPage("generic", "generic", "", "", 0.10)], "hybrid"

    monkeypatch.setattr(recall_runtime, "run_search", fake_search)

    results, _mode = search_candidates(["specific", "generic"], RecallPolicy())

    assert [result.page_id for result in results[:2]] == ["target", "generic"]


def test_search_candidates_filters_sensitive_pages_in_work_context(monkeypatch) -> None:
    from llm_wiki_mcp import recall_runtime

    def fake_search(
        *,
        query: str,
        top_n: int,
        semantic: bool,
        fusion_weights: dict[str, float],
    ) -> tuple[list[ScoredPage], str]:
        del query, top_n, semantic, fusion_weights
        return [
            ScoredPage("career-note", "Career Note", "career", "", 1.0, sensitivity="high"),
            ScoredPage("work-note", "Work Note", "ai", "", 0.5),
        ], "hybrid"

    monkeypatch.setattr(recall_runtime, "run_search", fake_search)

    results, _mode = search_candidates(
        ["query"],
        RecallPolicy(),
        request=RecallRequest(
            host="codex",
            event="UserPromptSubmit",
            prompt="project status",
            cwd="/Users/trafficsign/projects/work/client",
        ),
    )

    assert [result.page_id for result in results] == ["work-note"]


def test_search_candidates_allows_sensitive_pages_when_prompt_requests_it(monkeypatch) -> None:
    from llm_wiki_mcp import recall_runtime

    def fake_search(
        *,
        query: str,
        top_n: int,
        semantic: bool,
        fusion_weights: dict[str, float],
    ) -> tuple[list[ScoredPage], str]:
        del query, top_n, semantic, fusion_weights
        return [ScoredPage("career-note", "Career Note", "career", "", 1.0, sensitivity="high")], "hybrid"

    monkeypatch.setattr(recall_runtime, "run_search", fake_search)

    results, _mode = search_candidates(
        ["query"],
        RecallPolicy(),
        request=RecallRequest(
            host="codex",
            event="UserPromptSubmit",
            prompt="面接の話を思い出して",
            cwd="/Users/trafficsign/projects/work/client",
        ),
    )

    assert [result.page_id for result in results] == ["career-note"]


def test_best_excerpt_index_prefers_dense_query_terms() -> None:
    body = (
        "LLM Wiki was mentioned near the top.\n"
        "Some unrelated changelog text follows.\n"
        "Deployment note: git push plus uvx cache refresh is required before restart.\n"
    ).lower()
    terms = excerpt_terms(["LLM Wiki GitHub runtime uvx cache push"])

    idx = best_excerpt_index(body, terms, max_chars=80)

    assert "uvx cache refresh" in body[max(0, idx - 50) : idx + 80]


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
model = "legacy-local-model"

[budgets]
judge_timeout_ms = 4000

[gate]
model = "qwen3.5:4b-mlx"
think = false
timeout_ms = 1200
num_ctx = 2048
num_predict = 128
keep_alive = "1h"
warmup_timeout_ms = 9000

[rewrite]
timeout_ms = 1400
"""
    )

    policy = load_policy(config)

    assert policy.judge_model == "qwen3.5:4b-mlx"
    assert policy.judge_think is False
    assert policy.judge_timeout_ms == 1200
    assert policy.judge_num_ctx == 2048
    assert policy.judge_num_predict == 128
    assert policy.judge_keep_alive == "1h"
    assert policy.warmup_timeout_ms == 9000
    assert policy.rewrite_timeout_ms == 1400


def test_gate_defaults_keep_model_resident_and_rewrite_timeout_longer(tmp_path) -> None:
    config = tmp_path / "recall.toml"
    config.write_text("enabled = true\n")

    policy = load_policy(config)

    assert policy.judge_keep_alive == "24h"
    assert policy.warmup_timeout_ms == 15000
    assert policy.rewrite_timeout_ms == 3000


def test_fusion_config_reads_channel_weights_and_bm25_bonus(tmp_path) -> None:
    config = tmp_path / "recall.toml"
    config.write_text(
        """
[fusion]
bm25 = 1.0
semantic = 0.6
graph = 0.0
usage_prior = 0.0
bm25_score_bonus = 0.005
bm25_rank_bonus = 0.006
bm25_rank_decay = 0.006
semantic_min_top_score = 0.45
semantic_min_margin = 0.002
semantic_low_confidence_weight = 0.25
""",
        encoding="utf-8",
    )

    policy = load_policy(config)

    assert policy.fusion_bm25 == 1.0
    assert policy.fusion_semantic == 0.6
    assert policy.fusion_graph == 0.0
    assert policy.fusion_usage_prior == 0.0
    assert policy.fusion_bm25_score_bonus == 0.005
    assert policy.fusion_bm25_rank_bonus == 0.006
    assert policy.fusion_bm25_rank_decay == 0.006
    assert policy.fusion_semantic_min_top_score == 0.45
    assert policy.fusion_semantic_min_margin == 0.002
    assert policy.fusion_semantic_low_confidence_weight == 0.25


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
        judge_model="qwen3.5:4b-mlx",
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
    assert payload["model"] == "qwen3.5:4b-mlx"
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
    policy = RecallPolicy(judge_model="qwen3.5:4b-mlx", judge_timeout_ms=2000)

    score, _queries, reason = run_local_judge(
        RecallRequest(host="test", event="UserPromptSubmit", prompt="LLM Wiki の運用どうする?"),
        0.5,
        policy,
    )

    assert score < policy.search_threshold
    assert reason == "不要"


def test_query_rewriter_timeout_falls_back_with_reason(monkeypatch) -> None:
    from llm_wiki_mcp import recall_runtime

    class FakeClient:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            return None

        def __enter__(self) -> "FakeClient":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def post(self, _path: str, *, json: dict[str, object]) -> object:
            raise recall_runtime.httpx.ReadTimeout("cold model")

    monkeypatch.setattr(recall_runtime.httpx, "Client", FakeClient)

    queries, confidence, reason = run_query_rewriter(
        RecallRequest(host="test", event="UserPromptSubmit", prompt="前のあれ"),
        {"past_reference": ["前の"], "ambiguity": ["あれ"]},
        RecallPolicy(rewrite_timeout_ms=3000),
        "",
    )

    assert queries == []
    assert confidence == 0.0
    assert reason == "rewrite fallback: ReadTimeout"


def test_warm_recall_model_uses_configured_keep_alive(monkeypatch) -> None:
    from llm_wiki_mcp import recall_runtime

    captured: list[dict[str, object]] = []

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

    class FakeClient:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            return None

        def __enter__(self) -> "FakeClient":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def post(self, _path: str, *, json: dict[str, object]) -> FakeResponse:
            captured.append(json)
            return FakeResponse()

    monkeypatch.setattr(recall_runtime.httpx, "Client", FakeClient)

    result = warm_recall_model(
        RecallPolicy(
            judge_model="judge-model",
            rewrite_model="rewrite-model",
            judge_keep_alive="1h",
        )
    )

    assert result["ok"] is True
    assert result["models"] == ["judge-model", "rewrite-model"]
    assert [payload["keep_alive"] for payload in captured] == ["1h", "1h"]


def test_run_recall_records_rewrite_fallback_metrics(monkeypatch) -> None:
    from llm_wiki_mcp import recall_runtime

    def fake_search(**_kwargs: object) -> tuple[list[ScoredPage], str]:
        return [], "bm25"

    def timeout_rewriter(*_args: object, **_kwargs: object) -> tuple[list[str], float, str]:
        return [], 0.0, "rewrite fallback: ReadTimeout"

    monkeypatch.setattr(recall_runtime, "run_search", fake_search)
    monkeypatch.setattr(recall_runtime, "run_query_rewriter", timeout_rewriter)

    result = run_recall(
        RecallRequest(
            host="test",
            event="UserPromptSubmit",
            prompt="前のあれ、LLM Wikiでどう直すんだっけ?",
            cwd="/Users/trafficsign/projects/personal/llm-wiki-mcp",
        ),
        RecallPolicy(judge_mode="off", log_decisions=False),
        perform_search=True,
    )

    assert "rewrite fallback: ReadTimeout" in result.reasons
    assert result.evidence_features["rewrite_attempted"] is True
    assert result.evidence_features["rewrite_status"] == "fallback"
    assert result.evidence_features["rewrite_reason"] == "rewrite fallback: ReadTimeout"
    assert isinstance(result.evidence_features["rewrite_latency_ms"], int)


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


def test_recall_context_includes_decision_id() -> None:
    result = RecallResult(
        status="ok",
        decision="read",
        confidence=0.8,
        queries=["昨日のLLM Wiki"],
        reasons=["past reference term"],
        matched_terms={},
        decision_id="20260602T120000-deadbeef",
        context_items=[
            ContextItem(
                page_id="llm-wiki-recall-configuration",
                title="LLM Wiki Recall Configuration",
                updated="2026-06-02",
                score=1.0,
                sensitivity="high",
            )
        ],
    )

    context = format_recall_context(result, RecallPolicy())

    assert "decision_id=20260602T120000-deadbeef" in context
    assert "updated: 2026-06-02" in context
    assert "sensitivity: high" in context


def test_run_recall_log_records_decision_snapshot(tmp_path, monkeypatch) -> None:
    from llm_wiki_mcp import recall_runtime

    log_file = tmp_path / "recall-log.jsonl"
    monkeypatch.setattr(recall_runtime, "RECALL_LOG_FILE", log_file)

    result = run_recall(
        RecallRequest(
            host="test",
            event="UserPromptSubmit",
            prompt="昨日LLM Wikiのフック直したやつ、Claude Codeにも入れられる?",
            cwd="/Users/trafficsign/projects/personal/llm-wiki-mcp",
        ),
        RecallPolicy(judge_mode="off", log_decisions=True),
        perform_search=False,
    )
    record = json.loads(log_file.read_text().splitlines()[-1])

    assert record["decision_id"] == result.decision_id
    assert record["decision"] == "read"
    assert record["confidence"] == result.confidence
    assert record["queries"] == result.queries
    assert record["prompt_hash"]
    assert record["judge_confidence"] is None
    assert "past reference term" in record["reasons"]


def test_recall_log_also_writes_live_episode_snapshot(tmp_path, monkeypatch) -> None:
    from llm_wiki_mcp import recall_runtime

    wiki_root = tmp_path / "wiki"
    log_file = wiki_root / "recall" / "recall-log.jsonl"
    monkeypatch.setattr(recall_runtime, "RECALL_LOG_FILE", log_file)
    result = RecallResult(
        status="ok",
        decision="read",
        confidence=0.8,
        queries=["llm wiki recall"],
        reasons=["test"],
        matched_terms={},
        decision_id="d1",
        context_items=[
            ContextItem(
                page_id="page-a",
                title="Page A",
                updated="2026-07-05",
                score=1.0,
            )
        ],
    )

    recall_runtime.append_recall_log(
        RecallRequest(
            host="codex",
            event="UserPromptSubmit",
            prompt="LLM Wiki recall",
            cwd="/repo",
            session_id="s1",
        ),
        result,
    )

    live_file = wiki_root / "runtime" / "recall-improvement" / "live-episodes.jsonl"
    live = json.loads(live_file.read_text(encoding="utf-8"))
    assert live["decision_id"] == "d1"
    assert live["quality"]["usefulness"] == "unknown"
    assert live["pages"] == ["page-a"]


def test_feedback_writer_uses_configurable_path(tmp_path, monkeypatch) -> None:
    from llm_wiki_mcp import recall_runtime

    feedback_file = tmp_path / "feedback.jsonl"
    monkeypatch.setattr(recall_runtime, "RECALL_FEEDBACK_FILE", feedback_file)

    record = append_feedback("missed", "前回の話を検索しなかった", prompt="昨日のあれ")

    assert record["kind"] == "missed"
    assert feedback_file.exists()
    assert "前回の話" in feedback_file.read_text()


def test_missed_feedback_prompt_only_records_without_expected(tmp_path, monkeypatch) -> None:
    from llm_wiki_mcp import recall_runtime

    feedback_file = tmp_path / "feedback.jsonl"
    monkeypatch.setattr(recall_runtime, "RECALL_FEEDBACK_FILE", feedback_file)

    record = append_feedback("missed", prompt="本当は前回の設定を引くべきだった")

    assert record["kind"] == "missed"
    assert record["prompt"] == "本当は前回の設定を引くべきだった"
    assert record["note"] == ""
    assert record["expected_pages"] == []
    assert record["expected_queries"] == []
    assert record["ref"] == ""
    assert record["snapshot"] is None


def test_recent_cli_lists_latest_recall_decisions(tmp_path, monkeypatch, capsys) -> None:
    from llm_wiki_mcp import recall_runtime

    log_file = tmp_path / "recall-log.jsonl"
    monkeypatch.setattr(recall_runtime, "RECALL_LOG_FILE", log_file)
    log_file.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "ts": "2026-06-02T10:00:00",
                        "decision_id": "old",
                        "host": "codex",
                        "decision": "none",
                        "confidence": 0.1,
                        "prompt_preview": "old prompt",
                        "queries": [],
                        "pages": [],
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "ts": "2026-06-02T10:01:00",
                        "decision_id": "new",
                        "host": "claude-code",
                        "decision": "read",
                        "confidence": 0.8,
                        "prompt_preview": "new prompt",
                        "queries": ["new query"],
                        "pages": ["claude-code-recall-hook-implementation"],
                    },
                    ensure_ascii=False,
                ),
            ]
        )
        + "\n"
    )

    assert main(["--recent", "1"]) == 0
    output = json.loads(capsys.readouterr().out)

    assert output["status"] == "ok"
    assert len(output["items"]) == 1
    assert output["items"][0]["decision_id"] == "new"
    assert output["items"][0]["pages"] == ["claude-code-recall-hook-implementation"]


def test_missed_feedback_ref_embeds_snapshot(tmp_path, monkeypatch, capsys) -> None:
    from llm_wiki_mcp import recall_runtime

    log_file = tmp_path / "recall-log.jsonl"
    feedback_file = tmp_path / "feedback.jsonl"
    monkeypatch.setattr(recall_runtime, "RECALL_LOG_FILE", log_file)
    monkeypatch.setattr(recall_runtime, "RECALL_FEEDBACK_FILE", feedback_file)
    log_file.write_text(
        json.dumps(
            {
                "ts": "2026-06-02T10:00:00",
                "decision_id": "20260602T100000-deadbeef",
                "host": "codex",
                "event": "UserPromptSubmit",
                "cwd": "/repo",
                "session_id": "s1",
                "prompt_preview": "前に話した recall gate",
                "decision": "none",
                "confidence": 0.34,
                "queries": [],
                "pages": [],
                "reasons": ["judge: 不要"],
                "used_judge": True,
                "judge_confidence": 0.34,
                "judge_reason": "不要",
                "latency_ms": 900,
                "status": "ok",
                "error": "",
            },
            ensure_ascii=False,
        )
        + "\n"
    )

    assert main(
        [
            "--feedback",
            "missed",
            "--prompt",
            "前に話した recall gate",
            "--expected-page",
            "llm-wiki-recall-configuration",
            "--expected-query",
            "recall gate model",
            "--ref",
            "20260602T100000-deadbeef",
        ]
    ) == 0
    output = json.loads(capsys.readouterr().out)
    record = json.loads(feedback_file.read_text().splitlines()[-1])

    assert output["status"] == "recorded"
    assert record["kind"] == "missed"
    assert record["expected_pages"] == ["llm-wiki-recall-configuration"]
    assert record["expected_queries"] == ["recall gate model"]
    assert record["snapshot"]["decision_id"] == "20260602T100000-deadbeef"
    assert record["snapshot"]["score"] == 0.34
    assert record["snapshot"]["judge_reason"] == "不要"


def test_missed_candidate_feedback_does_not_suppress_runtime(tmp_path, monkeypatch) -> None:
    from llm_wiki_mcp import recall_runtime

    feedback_file = tmp_path / "feedback.jsonl"
    monkeypatch.setattr(recall_runtime, "RECALL_FEEDBACK_FILE", feedback_file)
    prompt = "昨日LLM Wikiのフック直したやつ、Claude Codeにも入れられる?"
    append_feedback("missed_candidate", prompt=prompt, extra={"source": "auditor"})

    result = run_recall(
        RecallRequest(
            host="test",
            event="UserPromptSubmit",
            prompt=prompt,
            cwd="/Users/trafficsign/projects/personal/llm-wiki-mcp",
        ),
        RecallPolicy(judge_mode="off", log_decisions=False),
        perform_search=False,
    )

    assert result.decision == "read"
    assert "feedback false-positive prompt" not in result.reasons
