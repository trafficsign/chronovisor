from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from chronovisor.ingest.triage_plan import (
    canonical_triage_target,
    collapse_exact_duplicate_operations,
    distinct_target_collisions,
)


def _operation(
    *,
    filename: str = "ai/claude-code-wiki-save-hook.md",
    summary: str = "fact 0",
    keywords: list[str] | None = None,
    op_type: str = "create",
) -> dict:
    return {
        "type": op_type,
        "filename": filename,
        "title": "Claude Code Wiki Save Hook",
        "keywords": keywords or ["save-hook"],
        "summary": summary,
    }


class _QueueTransport:
    def __init__(self, *responses: str) -> None:
        self.responses = list(responses)
        self.requests: list[object] = []

    def __call__(self, request):
        self.requests.append(request)
        return self.responses.pop(0)


def test_exact_duplicate_operation_is_the_only_safe_host_collapse() -> None:
    first = _operation(filename="AI/Legacy_Hook.md", op_type="update")
    duplicate = _operation(filename="ai/legacy_hook", op_type="update")

    collapsed = collapse_exact_duplicate_operations([first, duplicate])

    assert collapsed == [first]


def test_distinct_meaning_for_one_normalized_page_id_is_not_dropped() -> None:
    first = _operation(
        filename="ai/Café.md",
        summary="preserve first fact",
        op_type="update",
    )
    second = _operation(
        filename="other/cafe\u0301.md",
        summary="preserve second fact",
        op_type="update",
    )

    collapsed, collisions = distinct_target_collisions([first, second])

    assert collapsed == [first, second]
    assert canonical_triage_target(first["filename"]) == canonical_triage_target(
        second["filename"]
    )
    assert collisions == [
        {
            "target_page_id": "café",
            "operation_indices": [0, 1],
            "operation_count": 2,
        }
    ]


def test_collision_feedback_indices_refer_to_original_plan() -> None:
    first = _operation(summary="first fact", op_type="update")
    exact_duplicate = dict(first)
    distinct = _operation(summary="second fact", op_type="update")

    collapsed, collisions = distinct_target_collisions(
        [first, exact_duplicate, distinct]
    )

    assert collapsed == [first, distinct]
    assert collisions[0]["operation_indices"] == [0, 2]


def test_ingest_validator_requests_semantic_merge_in_same_session(
    monkeypatch,
) -> None:
    from chronovisor.core import search
    from chronovisor.ingest import ingest

    invalid = [
        _operation(summary=f"fact {index}", keywords=[f"keyword-{index}"])
        for index in range(7)
    ]
    valid = _operation(
        summary="; ".join(f"fact {index}" for index in range(7)),
        keywords=[f"keyword-{index}" for index in range(7)],
    )
    transport = _QueueTransport(json.dumps(invalid), json.dumps([valid]))

    class EmptyIndex:
        def ensure_loaded(self) -> None:
            pass

        def all_canonical_page_keys(self) -> set[str]:
            return set()

        def page_count(self) -> int:
            return 0

    monkeypatch.setattr(ingest, "get_store", EmptyIndex)
    monkeypatch.setattr(search, "search", lambda *_args, **_kwargs: ([], "bm25"))
    monkeypatch.setattr(ingest, "_find_existing_create_target", lambda _op: None)
    monkeypatch.setattr(
        ingest,
        "load_ingest_config",
        lambda: SimpleNamespace(
            model="triage:test",
            num_ctx=32_768,
            max_num_ctx=262_144,
            num_predict=4_096,
            keep_alive="0",
            read_timeout_ms=5_000,
        ),
    )

    result = ingest._triage("raw content", transport=transport)

    assert result == [valid]
    assert len(transport.requests) == 2
    prior_response = transport.requests[1].messages[-2]
    assert prior_response["role"] == "assistant"
    assert "invalid plain-text record omitted" in prior_response["content"]
    assert "fact 0" not in prior_response["content"]
    assert "fact 6" not in prior_response["content"]
    feedback = transport.requests[1].messages[-1]["content"]
    assert '"keyword":"uniqueTarget"' in feedback
    assert '"operation_indices":[0,1,2,3,4,5,6]' in feedback
    assert "complete previous plan" in feedback
    assert "merge every distinct fact" in feedback
    assert "fact 0" not in feedback
    assert "fact 6" not in feedback
    assert len(feedback.encode("utf-8")) < 4_000


def test_effective_search_before_create_collision_is_repaired_before_generate(
    monkeypatch,
) -> None:
    from chronovisor.ingest import ingest

    plan = [
        _operation(filename="ai/new-save-hook.md", summary="first fact"),
        _operation(filename="ai/save-hook-notes.md", summary="second fact"),
    ]
    existing = Path("/wiki/pages/ai/claude-code-wiki-save-hook.md")
    monkeypatch.setattr(
        ingest,
        "_find_existing_create_target",
        lambda _op: (existing, "same-title", 1.0),
    )
    monkeypatch.setattr(
        ingest,
        "_relative_page_filename",
        lambda path: f"ai/{path.name}",
    )

    issues = ingest._triage_plan_validation_issues(
        plan,
        resolve_effective_targets=True,
    )

    assert len(issues) == 1
    assert issues[0].keyword == "uniqueTarget"
    assert issues[0].received["target_page_id"] == "claude-code-wiki-save-hook"
    assert issues[0].received["operation_indices"] == [0, 1]
    assert issues[0].received["operation_count"] == 2


def test_prompt_forbids_multiple_operations_for_one_target() -> None:
    from chronovisor.core.ollama import TRIAGE_SYSTEM_PROMPT

    assert "exactly one operation per case/Unicode-insensitive target page ID" in (
        TRIAGE_SYSTEM_PROMPT
    )
    assert "preserve all of them" in TRIAGE_SYSTEM_PROMPT


def test_c2_reasoning_disabled_is_explicit_and_default_false_is_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chronovisor.ingest import ingest_triage

    config = SimpleNamespace(
        model="triage:test",
        num_ctx=32_768,
        max_num_ctx=65_536,
        num_predict=256,
        keep_alive="0",
        read_timeout_ms=5_000,
    )
    monkeypatch.setattr(
        ingest_triage,
        "_build_triage_catalog",
        lambda *_args, **_kwargs: (object(), "bounded catalog"),
    )
    monkeypatch.setattr(
        ingest_triage,
        "load_ingest_config",
        lambda: config,
    )
    monkeypatch.setattr(
        ingest_triage,
        "required_structured_context_tokens",
        lambda *_args, **_kwargs: 512,
    )
    monkeypatch.setattr(
        ingest_triage,
        "_select_ingest_context",
        lambda *_args, **_kwargs: 32_768,
    )
    monkeypatch.setattr(
        ingest_triage,
        "_validate_effective_triage_plan",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        ingest_triage,
        "_validate_triage_plan",
        lambda value, **_kwargs: list(value) if isinstance(value, list) else None,
    )

    captured: list[dict[str, object]] = []
    real_session = ingest_triage.LocalStructuredSession

    def capture_session(**kwargs: object):
        captured.append(dict(kwargs))
        return real_session(**kwargs)

    monkeypatch.setattr(ingest_triage, "LocalStructuredSession", capture_session)
    records = [{"record_id": "turn-1", "text": "明示された決定"}]
    response = json.dumps({"operations": [], "semantic_evidence": []})

    default_transport = _QueueTransport(response)
    default_result = ingest_triage.triage_c2(
        records,
        transport=default_transport,
    )
    disabled_transport = _QueueTransport(response)
    disabled_result = ingest_triage.triage_c2(
        records,
        transport=disabled_transport,
        reasoning_disabled=True,
    )

    assert default_result is not None
    assert disabled_result is not None
    assert default_result == ingest_triage.triage_c2(
        records,
        transport=_QueueTransport(response),
        reasoning_disabled=False,
    )
    assert len(captured) == 3
    assert captured[0]["reasoning_disabled"] is False
    assert captured[1]["reasoning_disabled"] is True
    assert captured[2]["reasoning_disabled"] is False
    assert default_transport.requests[0].think is not False
    assert disabled_transport.requests[0].think is False
    assert disabled_transport.requests[0].think_selection_reason == (
        "caller_disabled_reasoning"
    )
    assert disabled_result["audit"]["local_structured"]["think"] is False
    assert disabled_result["audit"]["local_structured"]["think_selection_reason"] == (
        "caller_disabled_reasoning"
    )
    assert default_result["audit"]["request_sha256"] == disabled_result["audit"]["request_sha256"]

    invalid_transport = _QueueTransport(response)
    with pytest.raises(ValueError, match="reasoning_disabled"):
        ingest_triage.triage_c2(
            records,
            transport=invalid_transport,
            reasoning_disabled=1,  # type: ignore[arg-type]
        )
    assert invalid_transport.requests == []
