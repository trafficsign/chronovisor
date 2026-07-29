from __future__ import annotations

import hashlib
import json
import threading
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from chronovisor.core import ollama
from chronovisor.decision.local_structured import (
    STRUCTURED_GENERATION_POLICY_VERSION,
    ChatRequest,
    LocalConsensusAuditStore,
    LocalStructuredSession,
    ValidationIssue,
    normalize_json_output,
    structured_generation_policy,
    structured_generation_policy_sha256,
    structured_think_mode,
    validate_json,
)

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["decision", "summary"],
    "properties": {
        "decision": {"type": "string", "enum": ["apply", "defer"]},
        "summary": {"type": "string", "minLength": 1},
    },
}


class QueueTransport:
    def __init__(
        self,
        *responses: str | ollama.ChatResponse | ollama.GenerateResponse | Exception,
    ) -> None:
        self.responses = deque(responses)
        self.requests: list[ChatRequest] = []

    def __call__(
        self, request: ChatRequest
    ) -> str | ollama.ChatResponse | ollama.GenerateResponse:
        self.requests.append(request)
        response = self.responses.popleft()
        if isinstance(response, Exception):
            raise response
        return response


def test_structured_generation_policy_seals_the_fixed_sampler() -> None:
    assert STRUCTURED_GENERATION_POLICY_VERSION == 3
    assert structured_generation_policy() == {
        "version": 3,
        "temperature": 0,
        "seed": 0,
        "think": {
            "default": False,
            "model_family_overrides": {
                "gpt-oss": {
                    "default": "medium",
                    "num_ctx_at_least": {"65536": "low"},
                }
            },
        },
        "stream": False,
        "format": "json_schema",
    }
    assert structured_think_mode("gpt-oss:20b", num_ctx=32_768) == "medium"
    assert (
        structured_think_mode("registry/local/gpt-oss:20b", num_ctx=65_536)
        == "low"
    )
    assert structured_think_mode("ornith:35b", num_ctx=114_688) is False
    assert len(structured_generation_policy_sha256()) == 64


def test_transport_format_schema_does_not_weaken_client_validation() -> None:
    validation_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["summary"],
        "properties": {"summary": {"type": "string", "maxLength": 2}},
    }
    format_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["summary"],
        "properties": {"summary": {"type": "string"}},
    }
    transport = QueueTransport('{"summary":"too long"}', '{"summary":"ok"}')

    result = _session(transport).run(
        "summarize",
        validation_schema,
        format_schema=format_schema,
    )

    assert result.ok is True
    assert result.value == {"summary": "ok"}
    assert result.first_pass_valid is False
    assert transport.requests[0].schema == format_schema
    assert transport.requests[0].messages[0]["content"].find('"maxLength": 2') > 0


@pytest.fixture(autouse=True)
def _isolate_default_audit_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chronovisor.core import store

    monkeypatch.setattr(store, "CHRONOVISOR_ROOT", tmp_path / "wiki")


def _session(transport: QueueTransport, **overrides: Any) -> LocalStructuredSession:
    options: dict[str, Any] = {
        "num_ctx": 16_384,
        "num_predict": 256,
        "max_input_chars": 20_000,
        "max_output_chars": 1_000,
        "max_feedback_chars": 2_000,
    }
    options.update(overrides)
    return LocalStructuredSession(model="local:test", transport=transport, **options)


def test_activity_marker_tracks_redacted_structured_phase(tmp_path: Path) -> None:
    store = LocalConsensusAuditStore(tmp_path / "audit")

    with store.activity(
        request_sha256="a" * 64,
        role="ingest_review:challenger",
        model="ornith:test",
    ) as update:
        marker_path = next(store.active_dir.glob("*.json"))
        initial = json.loads(marker_path.read_text(encoding="utf-8"))
        update("repair", 1)
        repaired = json.loads(marker_path.read_text(encoding="utf-8"))

        assert initial["phase"] == "trigger"
        assert initial["attempt"] == 0
        assert repaired["phase"] == "repair"
        assert repaired["attempt"] == 1
        assert repaired["request_sha256"] == "a" * 64
        assert "prompt" not in repaired
        assert "raw_output" not in repaired

    assert list(store.active_dir.glob("*.json")) == []


def test_first_pass_valid_uses_fixed_non_thinking_request() -> None:
    transport = QueueTransport('{"decision":"apply","summary":"ok"}')

    result = _session(transport).run(
        "decide", SCHEMA, system="Follow the decision rule."
    )

    assert result.ok is True
    assert result.value == {"decision": "apply", "summary": "ok"}
    assert result.first_pass_valid is True
    assert result.repair_turns == 0
    request = transport.requests[0]
    assert request.model == "local:test"
    assert request.num_ctx == 16_384
    assert request.num_predict == 256
    assert request.temperature == 0
    assert request.seed == 0
    assert request.think is False
    assert request.schema == SCHEMA
    assert request.messages[0]["role"] == "system"
    assert "untrusted data" in request.messages[0]["content"]
    assert '"decision"' in request.messages[0]["content"]


def test_schema_valid_semantic_error_repairs_in_the_same_session() -> None:
    transport = QueueTransport(
        '{"decision":"apply","summary":"wrong"}',
        '{"decision":"apply","summary":"exact"}',
    )

    def validate(value: Any) -> list[ValidationIssue]:
        if value.get("summary") == "exact":
            return []
        return [
            ValidationIssue(
                pointer="/summary",
                keyword="const",
                expected="exact",
                received="wrong",
                message="copy the exact bound value",
            )
        ]

    result = _session(transport).run(
        "decide",
        SCHEMA,
        value_validator=validate,
    )

    assert result.ok is True
    assert result.value["summary"] == "exact"
    assert result.first_pass_valid is False
    assert result.repair_turns == 1
    assert "copy the exact bound value" in transport.requests[1].messages[-1]["content"]


def test_active_marker_is_atomic_redacted_and_removed_after_session(
    tmp_path: Path,
) -> None:
    audit_root = tmp_path / "local-consensus"
    secret = "private user correction that must never be persisted"
    observed: dict[str, object] = {}

    def inspect_while_active(request: ChatRequest) -> str:
        paths = list((audit_root / "active").glob("*.json"))
        assert len(paths) == 1
        marker = json.loads(paths[0].read_text(encoding="utf-8"))
        observed.update(marker)
        serialized = paths[0].read_text(encoding="utf-8")
        assert secret not in serialized
        assert set(marker) == {
            "request_sha256",
            "role",
            "model",
            "phase",
            "attempt",
            "started_at",
            "updated_at",
            "pid",
        }
        assert marker["phase"] == "generate"
        assert marker["attempt"] == 0
        return '{"decision":"apply","summary":"ok"}'

    result = LocalStructuredSession(
        model="local:test",
        role="primary",
        transport=inspect_while_active,
        audit_root=audit_root,
        max_input_chars=20_000,
        max_output_chars=1_000,
        max_feedback_chars=2_000,
    ).run(secret, SCHEMA)

    assert result.ok is True
    assert observed["role"] == "primary"
    assert observed["model"] == "local:test"
    assert list((audit_root / "active").glob("*.json")) == []
    audit_text = (audit_root / "audit.jsonl").read_text(encoding="utf-8")
    assert secret not in audit_text
    summary = json.loads((audit_root / "summary.json").read_text(encoding="utf-8"))
    assert summary["sessions"]["first_pass_valid"] == 1
    assert summary["sessions"]["repaired"] == 0


def test_transport_failure_clears_activity_and_records_failure(tmp_path: Path) -> None:
    audit_root = tmp_path / "local-consensus"
    transport = QueueTransport(RuntimeError("offline"))

    result = _session(
        transport,
        role="challenger",
        audit_root=audit_root,
    ).run("decide", SCHEMA)

    assert result.failure_class == "transport_error"
    assert list((audit_root / "active").glob("*.json")) == []
    summary = json.loads((audit_root / "summary.json").read_text(encoding="utf-8"))
    assert summary["sessions"]["failures"] == {"transport_error": 1}


def test_observability_write_failure_does_not_change_valid_result(
    tmp_path: Path,
) -> None:
    blocked_root = tmp_path / "not-a-directory"
    blocked_root.write_text("file", encoding="utf-8")
    transport = QueueTransport('{"decision":"apply","summary":"ok"}')

    result = _session(transport, audit_root=blocked_root).run("decide", SCHEMA)

    assert result.ok is True
    assert result.value["decision"] == "apply"


def test_activity_role_rejects_payload_like_values() -> None:
    with pytest.raises(ValueError, match="role"):
        LocalStructuredSession(
            model="local:test",
            role="private user prompt with spaces",
            transport=QueueTransport('{"decision":"apply","summary":"unused"}'),
        )


def test_audit_store_keeps_a_bounded_tail_and_refreshes_summary(tmp_path: Path) -> None:
    store = LocalConsensusAuditStore(tmp_path / "audit", max_records=2)

    for index in range(3):
        store.append(
            {
                "kind": "decision",
                "request_sha256": str(index),
                "status": "quarantined" if index == 2 else "agreed",
                "pair_agreement": index == 1,
                "tie_break_used": False,
                "unresolved_quarantine": index == 2,
            }
        )

    rows = [
        json.loads(line)
        for line in store.audit_file.read_text(encoding="utf-8").splitlines()
    ]
    summary = json.loads(store.summary_file.read_text(encoding="utf-8"))
    assert [row["request_sha256"] for row in rows] == ["1", "2"]
    assert summary["retained_records"] == 2
    assert summary["decisions"] == {
        "agreed": 1,
        "pair_agreement": 1,
        "tie_break_used": 0,
        "total": 2,
        "unresolved_quarantine": 1,
    }


def test_trace_store_keeps_ordered_redacted_bounded_transitions(tmp_path: Path) -> None:
    store = LocalConsensusAuditStore(
        tmp_path / "audit",
        max_records=2,
        max_trace_records=3,
    )
    secret = "never-copy-this-prompt"

    for index, phase in enumerate(("trigger", "context", "generate", "validate")):
        store.record_transition(
            request_sha256="a" * 64,
            role="ingest_review:primary",
            model="local:test",
            phase=phase,
            attempt=index,
        )

    rows = [
        json.loads(line)
        for line in store.trace_file.read_text(encoding="utf-8").splitlines()
    ]
    assert [row["phase"] for row in rows] == ["context", "generate", "validate"]
    assert len({row["event_id"] for row in rows}) == 3
    assert all(row["kind"] == "phase" for row in rows)
    assert secret not in store.trace_file.read_text(encoding="utf-8")
    assert not any("prompt" in row or "raw_output" in row for row in rows)


def test_session_trace_records_real_phases_and_terminal_result(tmp_path: Path) -> None:
    audit_root = tmp_path / "local-consensus"
    result = _session(
        QueueTransport('{"decision":"apply","summary":"ok"}'),
        audit_root=audit_root,
    ).run("private prompt", SCHEMA)

    rows = [
        json.loads(line)
        for line in (audit_root / "trace-events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert result.ok is True
    assert [row["phase"] for row in rows] == [
        "trigger",
        "context",
        "generate",
        "validate",
        "vote",
        "vote",
    ]
    assert rows[-1]["kind"] == "session"
    assert rows[-1]["status"] == "done"
    assert "private prompt" not in json.dumps(rows)


def test_audit_quarantine_is_compare_and_swap_guarded(tmp_path: Path) -> None:
    store = LocalConsensusAuditStore(tmp_path / "local-consensus")
    store.append({"kind": "session", "role": "test", "model": "fake"})
    store.append({"kind": "session", "role": "test", "model": "fake"})
    original = store.audit_file.read_bytes()
    digest = hashlib.sha256(original).hexdigest()

    result = store.quarantine_records(
        expected_sha256=digest,
        reason="test audit isolation bug",
    )

    assert result["status"] == "quarantined"
    assert result["records"] == 2
    assert Path(result["archive"]).read_bytes() == original
    assert store.audit_file.read_bytes() == b""
    trace_archive = Path(result["archive"]).with_name(
        f"{Path(result['archive']).stem}-trace.jsonl"
    )
    assert trace_archive.exists()
    assert store.trace_file.read_bytes() == b""
    summary = json.loads(store.summary_file.read_text(encoding="utf-8"))
    assert summary["retained_records"] == 0
    with pytest.raises(RuntimeError, match="changed before quarantine"):
        store.quarantine_records(
            expected_sha256=digest,
            reason="stale cleanup",
        )


def test_parse_error_is_repaired_in_same_client_side_session() -> None:
    transport = QueueTransport(
        '{"decision":"apply",',
        '{"decision":"apply","summary":"fixed"}',
    )

    result = _session(transport).run("decide", SCHEMA)

    assert result.ok is True
    assert result.repair_turns == 1
    assert len(transport.requests) == 2
    second = transport.requests[1]
    assert [message["role"] for message in second.messages] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert second.messages[2]["content"] == '{"decision":"apply",'
    feedback = second.messages[3]["content"]
    assert '"keyword":"parse"' in feedback
    assert '"pointer":""' in feedback
    assert '"line":1' in feedback
    assert '"column":21' in feedback
    assert '"byte_offset":20' in feedback
    assert "preserve unrelated fields only when they remain" in feedback


def test_schema_errors_use_escaped_rfc6901_pointers() -> None:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["a/b~c"],
        "properties": {"a/b~c": {"type": "integer"}},
    }

    issues = validate_json({}, schema)

    assert len(issues) == 1
    assert issues[0].pointer == "/a~1b~0c"
    assert issues[0].keyword == "required"
    assert issues[0].expected == "property is present"
    assert issues[0].received == {"type": "missing"}


def test_schema_repair_prompt_contains_exact_pointer_expected_and_received() -> None:
    transport = QueueTransport(
        '{"decision":7,"summary":"wrong type"}',
        '{"decision":"apply","summary":"fixed"}',
    )

    result = _session(transport).run("decide", SCHEMA)

    assert result.ok is True
    feedback = transport.requests[1].messages[-1]["content"]
    assert '"pointer":"/decision"' in feedback
    assert '"keyword":"type"' in feedback
    assert '"expected":["string"]' in feedback
    assert '"received":{"type":"integer","value":7}' in feedback
    assert "Never change a truthful failed factual" in feedback
    assert "re-evaluate that root action or decision" in feedback


def test_validator_handles_existing_schema_subset_strictly() -> None:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["count", "names", "slug"],
        "properties": {
            "count": {"type": "integer", "minimum": 1, "maximum": 3},
            "names": {
                "type": "array",
                "items": {"type": "string", "minLength": 2, "maxLength": 4},
                "minItems": 1,
                "maxItems": 2,
                "uniqueItems": True,
            },
            "slug": {"type": "string", "pattern": "^[a-z-]+$"},
        },
    }

    issues = validate_json(
        {"count": True, "names": ["x", "x", "longer"], "slug": "BAD", "extra": 1},
        schema,
    )

    observed = {(issue.pointer, issue.keyword) for issue in issues}
    assert ("/count", "type") in observed
    assert ("/names", "maxItems") in observed
    assert ("/names/1", "uniqueItems") in observed
    assert ("/names/0", "minLength") in observed
    assert ("/names/2", "maxLength") in observed
    assert ("/slug", "pattern") in observed
    assert ("/extra", "additionalProperties") in observed


def test_same_invalid_output_stops_before_second_repair() -> None:
    invalid = '{"summary":"missing decision"}'
    transport = QueueTransport(
        invalid, invalid, '{"decision":"apply","summary":"unused"}'
    )

    result = _session(transport).run("decide", SCHEMA)

    assert result.ok is False
    assert result.failure_class == "repeated_output"
    assert len(result.attempts) == 2
    assert len(transport.requests) == 2


def test_same_validation_fingerprint_allows_second_repair_when_output_changes() -> None:
    transport = QueueTransport(
        '{"summary":"first"}',
        '{"summary":"second"}',
        '{"decision":"apply","summary":"fixed"}',
    )

    result = _session(transport).run("decide", SCHEMA)

    assert result.ok is True
    assert result.value == {"decision": "apply", "summary": "fixed"}
    assert result.repair_turns == 2
    assert result.attempts[0].output_sha256 != result.attempts[1].output_sha256
    assert result.attempts[0].error_fingerprint == result.attempts[1].error_fingerprint
    assert len(transport.requests) == 3


def test_same_validation_fingerprint_still_obeys_fixed_repair_limit() -> None:
    transport = QueueTransport(
        '{"summary":"first"}',
        '{"summary":"second"}',
        '{"summary":"third"}',
    )

    result = _session(transport).run("decide", SCHEMA)

    assert result.ok is False
    assert result.failure_class == "repair_exhausted"
    assert result.repair_turns == 2
    assert len({attempt.output_sha256 for attempt in result.attempts}) == 3
    assert len({attempt.error_fingerprint for attempt in result.attempts}) == 1
    assert len(transport.requests) == 3


def test_caller_can_disable_repair_turns_for_a_hard_synchronous_budget() -> None:
    transport = QueueTransport(
        '{"summary":"missing decision"}',
        '{"decision":"apply","summary":"unused repair"}',
    )

    result = _session(transport, max_responses=1).run("decide", SCHEMA)

    assert result.ok is False
    assert result.failure_class == "repair_exhausted"
    assert result.repair_turns == 0
    assert len(transport.requests) == 1


def test_max_responses_cannot_exceed_global_repair_safety_cap() -> None:
    with pytest.raises(ValueError, match="must not exceed the safety cap"):
        _session(QueueTransport(), max_responses=4)


def test_session_stops_after_initial_plus_two_repairs() -> None:
    transport = QueueTransport(
        '{"summary":"missing"}',
        '{"decision":"other","summary":"wrong enum"}',
        '{"decision":"apply","summary":"ok","extra":true}',
    )

    result = _session(transport).run("decide", SCHEMA)

    assert result.ok is False
    assert result.failure_class == "repair_exhausted"
    assert len(result.attempts) == 3
    assert len(transport.requests) == 3


def test_output_cap_repairs_without_putting_oversize_text_in_history() -> None:
    oversized = "x" * 101
    transport = QueueTransport(
        oversized,
        '{"decision":"apply","summary":"compact"}',
    )

    result = _session(transport, max_output_chars=100).run("decide", SCHEMA)

    assert result.ok is True
    assert result.first_pass_valid is False
    assert result.repair_turns == 1
    assert result.attempts[0].issues[0].keyword == "maxOutputBytes"
    assert len(transport.requests) == 2
    repair_messages = transport.requests[1].messages
    assert oversized not in json.dumps(repair_messages)
    assert "exceeded the fixed output limit" in repair_messages[-1]["content"]


def test_output_cap_fails_closed_after_two_oversize_repairs() -> None:
    transport = QueueTransport("x" * 101, "y" * 101, "z" * 101)

    result = _session(transport, max_output_chars=100).run("decide", SCHEMA)

    assert result.ok is False
    assert result.failure_class == "repair_exhausted"
    assert result.repair_turns == 2
    assert len(result.attempts) == 3
    assert all(
        attempt.issues[0].keyword == "maxOutputBytes" for attempt in result.attempts
    )


def test_initial_input_byte_cap_fails_before_call() -> None:
    transport = QueueTransport('{"decision":"apply","summary":"unused"}')

    result = _session(
        transport,
        max_input_chars=500,
        max_output_chars=500,
        max_feedback_chars=500,
    ).run("x" * 200, SCHEMA)

    assert result.ok is False
    assert result.failure_class == "input_too_large"
    assert transport.requests == []


def test_context_preflight_reserves_two_maximum_repair_histories() -> None:
    transport = QueueTransport('{"decision":"apply","summary":"unused"}')

    result = _session(
        transport,
        num_ctx=4_096,
        num_predict=256,
        max_input_chars=20_000,
        max_output_chars=1_000,
        max_feedback_chars=1_000,
    ).run("short", SCHEMA)

    assert result.ok is False
    assert result.failure_class == "context_window_exceeded"
    assert "two fixed UTF-8 byte-bounded repair histories" in result.failure_reason
    assert transport.requests == []


def test_production_112k_policy_retains_largest_bounded_prompt_and_two_repairs() -> (
    None
):
    transport = QueueTransport('{"decision":"apply","summary":"ok"}')

    result = _session(
        transport,
        num_ctx=114_688,
        num_predict=3_072,
        max_input_chars=93_000,
        max_output_chars=4_000,
        max_feedback_chars=2_000,
    ).run("x" * 92_000, SCHEMA)

    assert result.ok is True
    assert len(transport.requests) == 1


def test_context_window_guard_fails_before_ollama_can_truncate() -> None:
    transport = QueueTransport('{"decision":"apply","summary":"unused"}')

    result = _session(
        transport,
        num_ctx=4_096,
        num_predict=512,
        max_input_chars=20_000,
        max_output_chars=500,
        max_feedback_chars=500,
    ).run("記憶" * 1_200, SCHEMA)

    assert result.ok is False
    assert result.failure_class == "context_window_exceeded"
    assert transport.requests == []


def test_ascii_incompressible_context_is_rejected_before_transport() -> None:
    transport = QueueTransport('{"decision":"apply","summary":"unused"}')
    payload = "".join(f"id_{index:08x}_" for index in range(320))

    result = _session(
        transport,
        num_ctx=4_096,
        num_predict=512,
        max_input_chars=20_000,
        max_output_chars=500,
        max_feedback_chars=500,
    ).run(payload, SCHEMA)

    assert result.ok is False
    assert result.failure_class == "context_window_exceeded"
    assert transport.requests == []


def test_ollama_context_accounting_fails_closed_after_unexpected_shift() -> None:
    transport = QueueTransport(
        ollama.ChatResponse(
            content='{"decision":"apply","summary":"unsafe"}',
            prompt_eval_count=4_000,
            eval_count=200,
        )
    )

    result = _session(
        transport,
        num_ctx=4_096,
        num_predict=256,
        max_output_chars=200,
        max_feedback_chars=200,
    ).run("decide", SCHEMA)

    assert result.ok is False
    assert result.failure_class == "context_truncation_suspected"


def test_chat_incomplete_completion_rejects_valid_json_without_repair() -> None:
    transport = QueueTransport(
        ollama.ChatResponse(
            content='{"decision":"apply","summary":"unsafe"}',
            done=False,
        )
    )

    result = _session(transport).run("decide", SCHEMA)

    assert result.ok is False
    assert result.failure_class == "completion_incomplete"
    assert result.attempts == ()
    assert len(transport.requests) == 1


def test_generate_incomplete_stream_rejects_valid_json_without_repair() -> None:
    transport = QueueTransport(
        ollama.GenerateResponse(
            content='{"decision":"apply","summary":"unsafe"}',
            done=False,
            streamed=True,
        )
    )

    result = _session(transport).run("decide", SCHEMA)

    assert result.ok is False
    assert result.failure_class == "stream_incomplete"
    assert result.attempts == ()
    assert len(transport.requests) == 1


@pytest.mark.parametrize(
    "response",
    [
        ollama.ChatResponse(
            content='{"decision":"apply","summary":"unsafe"}',
            done=True,
            done_reason="length",
        ),
        ollama.GenerateResponse(
            content='{"decision":"apply","summary":"unsafe"}',
            done=True,
            done_reason="max_tokens",
        ),
    ],
)
def test_output_limit_reason_repairs_without_parsing_or_replaying_partial(
    response: ollama.ChatResponse | ollama.GenerateResponse,
) -> None:
    partial = response.content
    transport = QueueTransport(
        response,
        '{"decision":"apply","summary":"complete"}',
    )

    result = _session(transport).run("decide", SCHEMA)

    assert result.ok is True
    assert result.value == {"decision": "apply", "summary": "complete"}
    assert len(result.attempts) == 2
    assert result.attempts[0].issues[0].keyword == "completionMetadata"
    assert len(transport.requests) == 2
    repair_messages = transport.requests[1].messages
    assert all(message["content"] != partial for message in repair_messages)
    assert "Previous response omitted" in repair_messages[-2]["content"]
    assert "output limit" in repair_messages[-1]["content"]


def test_output_limit_on_every_turn_fails_operationally_after_bounded_repairs() -> None:
    responses = [
        ollama.GenerateResponse(
            content=f'{{"decision":"apply","summary":"partial-{index}"}}',
            done=True,
            done_reason="length",
        )
        for index in range(3)
    ]
    transport = QueueTransport(*responses)

    result = _session(transport).run("decide", SCHEMA)

    assert result.ok is False
    assert result.failure_class == "output_truncated"
    assert len(result.attempts) == 3
    assert len(transport.requests) == 3
    for request, previous in zip(transport.requests[1:], responses[:2], strict=True):
        assert all(
            message["content"] != previous.content for message in request.messages
        )


def test_default_transport_reuses_larger_resident_context_without_eviction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CHRONOVISOR_OLLAMA_RESOURCE_LOCK", str(tmp_path / "resource.lock"))
    planner_calls: list[dict[str, Any]] = []
    chat_contexts: list[int] = []

    def planner(_models: list[str], **kwargs: Any) -> SimpleNamespace:
        planner_calls.append(kwargs)
        return SimpleNamespace(
            max_resident_models=1,
            initial_eviction_models=(),
            context_for=lambda _model: 114_688,
        )

    def chat(_messages: list[dict[str, str]], **kwargs: Any) -> ollama.ChatResponse:
        chat_contexts.append(kwargs["num_ctx"])
        return ollama.ChatResponse(
            content='{"decision":"apply","summary":"ok"}',
            done=True,
            done_reason="stop",
        )

    monkeypatch.setattr(ollama, "plan_model_residency", planner)
    monkeypatch.setattr(ollama, "chat", chat)
    monkeypatch.setattr(
        ollama,
        "unload_named_model",
        lambda _model: pytest.fail("compatible larger runner must not be unloaded"),
    )

    result = LocalStructuredSession(
        model="local:test",
        audit_root=tmp_path / "audit",
        num_ctx=32_768,
        num_predict=256,
        max_input_chars=20_000,
        max_output_chars=1_000,
        max_feedback_chars=2_000,
        resource_min_num_ctx=32_768,
        resource_max_num_ctx=114_688,
        resource_memory_reserve_gib=8,
    ).run("decide", SCHEMA)

    assert result.ok is True
    assert chat_contexts == [114_688]
    assert planner_calls == [
        {
            "num_ctx": 32_768,
            "max_num_ctx": 114_688,
            "reserve_bytes": 8 * ollama.GIB,
            "configured_max_resident": 1,
            "reuse_larger_context": True,
        }
    ]


def test_default_transport_oversize_input_has_no_runner_side_effects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CHRONOVISOR_OLLAMA_RESOURCE_LOCK", str(tmp_path / "resource.lock"))
    planner_calls: list[object] = []
    unload_calls: list[object] = []
    chat_calls: list[object] = []

    def planner(*args: object, **kwargs: object) -> None:
        planner_calls.append((args, kwargs))
        pytest.fail("oversize input must fail before residency planning")

    def unload(*args: object, **kwargs: object) -> bool:
        unload_calls.append((args, kwargs))
        pytest.fail("oversize input must not evict a runner")

    def chat(*args: object, **kwargs: object) -> None:
        chat_calls.append((args, kwargs))
        pytest.fail("oversize input must not reach Ollama")

    monkeypatch.setattr(ollama, "plan_model_residency", planner)
    monkeypatch.setattr(ollama, "unload_named_model", unload)
    monkeypatch.setattr(ollama, "chat", chat)

    result = LocalStructuredSession(
        model="local:test",
        audit_root=tmp_path / "audit",
        num_ctx=32_768,
        num_predict=256,
        max_input_chars=256,
        max_output_chars=1_000,
        max_feedback_chars=2_000,
        resource_min_num_ctx=32_768,
        resource_max_num_ctx=114_688,
    ).run("oversized-user-input" * 100, SCHEMA)

    assert result.ok is False
    assert result.failure_class == "input_too_large"
    assert planner_calls == []
    assert unload_calls == []
    assert chat_calls == []
    assert ollama.model_resource_lease_mode() is None


def test_default_transport_maps_resource_lease_timeout_to_capacity_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    @contextmanager
    def busy_lease(**_kwargs: object) -> Iterator[None]:
        raise TimeoutError("busy")
        yield  # pragma: no cover

    monkeypatch.setattr(ollama, "model_resource_lease", busy_lease)
    monkeypatch.setattr(
        ollama,
        "plan_model_residency",
        lambda *_args, **_kwargs: pytest.fail(
            "busy resource must fail before residency planning"
        ),
    )

    result = LocalStructuredSession(
        model="local:test",
        audit_root=tmp_path / "audit",
        num_ctx=32_768,
        num_predict=256,
        max_input_chars=20_000,
        max_output_chars=1_000,
        max_feedback_chars=2_000,
        resource_lease_timeout_ms=25,
    ).run("decide", SCHEMA)

    assert result.ok is False
    assert result.failure_class == "capacity_unavailable"
    assert result.failure_reason == "structured model resource is busy"


def test_default_transport_holds_exclusive_lease_across_all_repair_turns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CHRONOVISOR_OLLAMA_RESOURCE_LOCK", str(tmp_path / "resource.lock"))
    large_entered = threading.Event()
    release_large = threading.Event()
    small_entered = threading.Event()
    call_order: list[int] = []
    failures: list[BaseException] = []
    results: dict[str, Any] = {}
    large_calls = 0

    def planner(_models: list[str], **kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(
            max_resident_models=1,
            initial_eviction_models=(),
            context_for=lambda _model: kwargs["num_ctx"],
        )

    def chat(_messages: list[dict[str, str]], **kwargs: Any) -> ollama.ChatResponse:
        nonlocal large_calls
        context = kwargs["num_ctx"]
        call_order.append(context)
        if context == 114_688:
            large_calls += 1
            if large_calls == 1:
                large_entered.set()
                assert release_large.wait(timeout=5)
                return ollama.ChatResponse(content="{}")
            return ollama.ChatResponse(
                content='{"decision":"apply","summary":"repaired"}'
            )
        small_entered.set()
        return ollama.ChatResponse(content='{"decision":"defer","summary":"small"}')

    monkeypatch.setattr(ollama, "plan_model_residency", planner)
    monkeypatch.setattr(ollama, "chat", chat)

    def run(name: str, context: int) -> None:
        try:
            results[name] = LocalStructuredSession(
                model="local:test",
                audit_root=tmp_path / name,
                num_ctx=context,
                num_predict=256,
                max_input_chars=20_000,
                max_output_chars=1_000,
                max_feedback_chars=2_000,
                resource_min_num_ctx=context,
                resource_max_num_ctx=context,
            ).run("decide", SCHEMA)
        except BaseException as exc:  # pragma: no cover - surfaced below
            failures.append(exc)

    large = threading.Thread(target=run, args=("large", 114_688))
    small = threading.Thread(target=run, args=("small", 32_768))
    large.start()
    assert large_entered.wait(timeout=5)
    small.start()
    assert not small_entered.wait(timeout=0.1)
    release_large.set()
    large.join(timeout=5)
    small.join(timeout=5)

    assert not large.is_alive()
    assert not small.is_alive()
    assert failures == []
    assert results["large"].ok is True
    assert results["large"].repair_turns == 1
    assert results["small"].ok is True
    assert call_order == [114_688, 114_688, 32_768]


def test_unsupported_schema_keyword_fails_before_transport() -> None:
    transport = QueueTransport('{"decision":"apply","summary":"unused"}')

    result = _session(transport).run("decide", {"type": "string", "oneOf": []})

    assert result.ok is False
    assert result.failure_class == "schema_invalid"
    assert transport.requests == []


def test_feedback_cap_fails_closed_instead_of_truncating_errors() -> None:
    transport = QueueTransport('{"summary":"missing decision"}')

    result = _session(transport, max_feedback_chars=80).run("decide", SCHEMA)

    assert result.ok is False
    assert result.failure_class == "feedback_too_large"
    assert len(transport.requests) == 1


def test_transport_timeout_is_not_retried_as_a_json_repair() -> None:
    transport = QueueTransport(httpx.ReadTimeout("slow"))

    result = _session(transport).run("decide", SCHEMA)

    assert result.ok is False
    assert result.failure_class == "transport_timeout"
    assert len(transport.requests) == 1


def test_duplicate_json_object_keys_are_rejected_not_silently_overwritten() -> None:
    transport = QueueTransport(
        '{"decision":"defer","decision":"apply","summary":"ambiguous"}',
        '{"decision":"apply","summary":"fixed"}',
    )

    result = _session(transport).run("decide", SCHEMA)

    assert result.ok is True
    assert result.repair_turns == 1
    assert result.attempts[0].issues[0].keyword == "parse"
    assert "duplicate object key" in result.attempts[0].issues[0].message


def test_only_whole_document_known_wrappers_are_normalized() -> None:
    fenced, fenced_changed = normalize_json_output('```json\n{"ok":true}\n```')
    prose, prose_changed = normalize_json_output('answer: {"ok":true}')
    channel, channel_changed = normalize_json_output(
        '<|channel|>final<|message|>{"ok":true}<|return|>'
    )

    assert (fenced, fenced_changed) == ('{"ok":true}', True)
    assert (channel, channel_changed) == ('{"ok":true}', True)
    assert (prose, prose_changed) == ('answer: {"ok":true}', False)


def test_audit_record_never_contains_raw_model_output_or_payload() -> None:
    secret = "secret-user-payload"
    transport = QueueTransport(json.dumps({"decision": "apply", "summary": secret}))

    result = _session(transport).run(secret, SCHEMA)
    serialized = json.dumps(result.audit_record(), ensure_ascii=False)

    assert result.ok is True
    assert secret not in serialized
    assert result.attempts[0].output_sha256 in serialized


def test_invalid_attempt_audit_hashes_snippets_and_received_values() -> None:
    secret = "secret-invalid-decision"
    transport = QueueTransport(
        json.dumps({"decision": secret, "summary": "bad"}),
        json.dumps({"decision": "apply", "summary": "fixed"}),
    )

    result = _session(transport).run("prompt", SCHEMA)
    serialized = json.dumps(result.audit_record(), ensure_ascii=False)

    assert result.ok is True
    assert secret not in serialized
    assert "value_sha256" in serialized
