from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from chronovisor.core import llm_config, reranker_client, runtime_config
from chronovisor.core.runtime_config import RerankerConfig
from chronovisor.recall import recall_runtime, recall_session
from chronovisor.recall.recall_runtime import (
    ContextItem,
    RecallPolicy,
    RecallRequest,
    RecallResult,
)

THRESHOLD = math.log(0.011624461214494347 / (1 - 0.011624461214494347))


def _result() -> RecallResult:
    return RecallResult(
        status="ok",
        decision="search",
        confidence=0.9,
        queries=["q"],
        reasons=[],
        matched_terms={},
        session_id="s1",
        context_items=[
            ContextItem("keep", "Keep title", "", 1.0, snippets=["useful evidence"]),
            ContextItem("drop", "Drop title", "", 0.9, snippets=["unrelated evidence"]),
        ],
    )


@pytest.fixture
def filter_env(tmp_path, monkeypatch):
    calls: list[dict] = []
    env = SimpleNamespace(mode="on", provider="local-decision", calls=calls, fail=None)

    monkeypatch.setattr(recall_session, "SESSIONS_DIR", tmp_path / "sessions")
    monkeypatch.setattr(
        runtime_config,
        "load_reranker_config",
        lambda *_a, **_k: RerankerConfig(recall_filter=env.mode),
    )
    monkeypatch.setattr(
        llm_config,
        "load_default_llm_runtime",
        lambda: SimpleNamespace(
            resolve_rerank=lambda _role: SimpleNamespace(provider=env.provider)
        ),
    )

    def recall_filter(query, context, texts, *, config, timeout_ms):
        calls.append(
            {"query": query, "context": context, "texts": texts, "timeout": timeout_ms}
        )
        if env.fail:
            raise env.fail
        return [THRESHOLD + 1 if "useful" in text else THRESHOLD - 1 for text in texts]

    monkeypatch.setattr(reranker_client, "recall_filter", recall_filter)
    return env


def _render(result: RecallResult, prompt: str = "how do I sort?") -> str:
    _state, recall_context, _context = recall_runtime._render_final_context(
        result,
        active_request=RecallRequest(
            host="claude", event="UserPromptSubmit", prompt=prompt, session_id="s1"
        ),
        policy=RecallPolicy(max_context_chars=3000, max_total_context_chars=3000),
    )
    return recall_context


def test_on_drops_items_under_the_frozen_threshold(filter_env) -> None:
    result = _result()

    context = _render(result)

    assert [item.page_id for item in result.context_items] == ["keep"]
    assert "Keep title" in context and "Drop title" not in context
    (call,) = filter_env.calls
    assert call["texts"] == [
        "Keep title\nuseful evidence",
        "Drop title\nunrelated evidence",
    ]
    record = result.evidence_features["recall_filter"]
    assert record["status"] == "applied"
    assert record["dropped_page_ids"] == ["drop"]


def test_shadow_scores_and_logs_without_dropping(filter_env) -> None:
    filter_env.mode = "shadow"
    result = _result()

    context = _render(result)

    assert [item.page_id for item in result.context_items] == ["keep", "drop"]
    assert "Drop title" in context
    record = result.evidence_features["recall_filter"]
    assert record["status"] == "observed" and record["dropped_page_ids"] == ["drop"]
    assert [item["page_id"] for item in record["items"]] == ["keep", "drop"]


def test_on_fails_closed_when_the_filter_cannot_run(filter_env) -> None:
    filter_env.fail = reranker_client.RerankerServiceUnavailable("transport_error")
    result = _result()

    assert _render(result) == ""
    assert result.context_items == []
    assert result.evidence_features["recall_filter"]["reason"] == "transport_error"


def test_off_and_relevance_routes_leave_items_alone(filter_env) -> None:
    for mode, provider in (("off", "local-decision"), ("on", "local-reranker")):
        filter_env.mode, filter_env.provider = mode, provider
        result = _result()
        _render(result)
        assert len(result.context_items) == 2
    assert filter_env.calls == []


def test_an_item_stays_when_any_of_its_passages_passes(filter_env) -> None:
    passage = lambda text: SimpleNamespace(  # noqa: E731
        text=text,
        byte_start=0,
        byte_end=len(text),
        evidence=SimpleNamespace(
            doc_id="d", page_uid="u", source_sha256="h", generation_id="g"
        ),
    )
    result = _result()
    result.context_items[1].source_passages = (
        passage("unrelated"),
        passage("useful part"),
    )

    _render(result)

    assert [item.page_id for item in result.context_items] == ["keep", "drop"]
    assert filter_env.calls[0]["texts"][1:] == [
        "Drop title\nunrelated",
        "Drop title\nuseful part",
    ]


def test_the_preceding_turn_is_the_last_distinct_logged_user_turn(filter_env) -> None:
    recall_runtime._remember_user_turn(
        RecallRequest(
            host="claude",
            event="UserPromptSubmit",
            prompt="sort a dict in python",
            session_id="s1",
        )
    )
    _render(_result(), prompt="and by value?")
    _render(_result(), prompt="sort a dict in python")

    assert [call["context"] for call in filter_env.calls] == [
        "sort a dict in python",
        "",
    ]
    state = recall_session.load_session_state("s1")
    assert recall_session.RecallSessionState.from_dict(
        "s1", state.to_dict()
    ).last_prompt == ("sort a dict in python")


def test_the_filter_gets_only_the_remaining_budget(filter_env) -> None:
    result = _result()
    recall_runtime._render_final_context(
        result,
        active_request=RecallRequest(
            host="claude", event="UserPromptSubmit", prompt="q", session_id="s1"
        ),
        policy=RecallPolicy(max_context_chars=3000, max_total_context_chars=3000),
        filter_deadline_at=recall_runtime.time.monotonic() + 0.1,
    )

    assert filter_env.calls == []
    assert result.context_items == []
    assert result.evidence_features["recall_filter"]["status"] == "unavailable"


@pytest.mark.parametrize(
    ("table", "mode", "probability"),
    [
        ({}, "off", 0.011624461214494347),
        ({"recall_filter": "SHADOW"}, "shadow", 0.011624461214494347),
        (
            {"recall_filter": "always", "recall_filter_min_probability": 2},
            "off",
            0.011624461214494347,
        ),
        ({"recall_filter": "on", "recall_filter_min_probability": 0.05}, "on", 0.05),
    ],
)
def test_config_parses_the_recall_filter(tmp_path, table, mode, probability) -> None:
    lines = ["[search.reranker]", "enabled = true"] + [
        f"{key} = {value!r}".replace("'", '"') for key, value in table.items()
    ]
    path = tmp_path / "config.toml"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    config = runtime_config.load_reranker_config(path)

    assert (config.recall_filter, config.recall_filter_min_probability) == (
        mode,
        probability,
    )
