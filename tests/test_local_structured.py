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

from chronovisor.core import llm_config, ollama
from chronovisor.core.llm_runtime import (
    MAX_OUTPUT_TOKENS,
    GenerationResult,
    GenerationRoute,
    LLMRuntime,
    MessageGenerationRequest,
    RouteLocation,
    SourceDataClass,
    SourceDataClassification,
    SourceSensitivity,
    TokenUsage,
)
from chronovisor.core.ollama_adapter import OllamaAdapter
from chronovisor.decision import local_structured
from chronovisor.decision.local_structured import (
    STRUCTURED_GENERATION_POLICY_VERSION,
    ChatRequest,
    LocalConsensusAuditStore,
    LocalStructuredSession,
    ValidationIssue,
    normalize_json_output,
    production_reasoning_authority_matches,
    required_structured_context_tokens,
    structured_generation_policy,
    structured_generation_policy_sha256,
    structured_reasoning_output_reservation,
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
QWEN_STRUCTURED_MODEL = "qwen3.8:27b-axq4"
MUSE_STRUCTURED_MODEL = "muse-glimmer:30b-q4k-dynamic"

REASONING_ROUTES = {
    "classification.primary": (
        "qwen3.8:27b-axq4",
        "09eba04b154c7a50fdc04104d0e5f0d9d535df5ffef7105529a4b001711d3bac",
    ),
    "classification.challenger": (
        "muse-glimmer:30b-q4k-dynamic",
        "76f57f0eaee1605340e376f868f01410a0677e9d318dfcec1d73fedfa165dcfa",
    ),
    "classification.tie_break": (
        "gemma4:26b-optiq4",
        "45ae882f90f8a0fd0b59a49cec62b0b3368545a7a60342af3e867bdcc46ae271",
    ),
}


def _reasoning_authority(runtime_role: str) -> dict[str, Any]:
    model, digest = REASONING_ROUTES[runtime_role]
    return {
        "role": runtime_role,
        "provider": "ollama",
        "model": model,
        "location": "local",
        "ollama": {
            "engine": {
                "name": "ollama",
                "version": "0.32.12-mixedmlx.1eeb7aad",
            },
            "digest": digest,
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


def test_structured_generation_policy_seals_adaptive_reasoning_authority() -> None:
    assert STRUCTURED_GENERATION_POLICY_VERSION == 16
    assert structured_generation_policy() == {
        "version": 16,
        "temperature": 0,
        "seed": 0,
        "think": {
            "default": "low",
            "fallback": "low",
            "levels": ["low", "medium", "high"],
            "bounded_low_lanes": ["local_repair", "read_back_repair"],
            "disabled_lanes": [
                "content_correction_review",
                "ingest_reconciliation",
            ],
            "adaptive_canary_adopted": False,
            "adaptive_authority": [
                {
                    "runtime_role": "classification.primary",
                    "model": "qwen3.8:27b-axq4",
                    "model_digest": (
                        "09eba04b154c7a50fdc04104d0e5f0d9d535df5ffef7105529a4b001711d3bac"
                    ),
                    "renderer": "boolean",
                    "levels": ["low", "medium", "high"],
                    "provider": "ollama",
                    "location": "local",
                    "engine": {
                        "name": "ollama",
                        "version": "0.32.12-mixedmlx.1eeb7aad",
                    },
                },
                {
                    "runtime_role": "classification.challenger",
                    "model": "muse-glimmer:30b-q4k-dynamic",
                    "model_digest": (
                        "76f57f0eaee1605340e376f868f01410a0677e9d318dfcec1d73fedfa165dcfa"
                    ),
                    "renderer": "native_levels",
                    "levels": ["low", "medium", "high"],
                    "provider": "ollama",
                    "location": "local",
                    "engine": {
                        "name": "ollama",
                        "version": "0.32.12-mixedmlx.1eeb7aad",
                    },
                },
                {
                    "runtime_role": "classification.tie_break",
                    "model": "gemma4:26b-optiq4",
                    "model_digest": (
                        "45ae882f90f8a0fd0b59a49cec62b0b3368545a7a60342af3e867bdcc46ae271"
                    ),
                    "renderer": "boolean",
                    "levels": ["low", "medium", "high"],
                    "provider": "ollama",
                    "location": "local",
                    "engine": {
                        "name": "ollama",
                        "version": "0.32.12-mixedmlx.1eeb7aad",
                    },
                },
            ],
            "output_budget": {
                "basis": "configured_num_predict",
                "reservation": "high",
                "multipliers": {
                    "low": {"numerator": 2, "denominator": 3},
                    "medium": {"numerator": 1, "denominator": 1},
                    "high": {"numerator": 4, "denominator": 3},
                },
            },
        },
        "compatibility": {
            QWEN_STRUCTURED_MODEL: {
                "initial": {"think": "selected_boolean", "format": None},
            },
            MUSE_STRUCTURED_MODEL: {
                "initial": {"think": "selected", "format": None},
            },
        },
        "repair": {
            "scope": "all_models",
            "think": False,
            "format": "json_schema",
        },
        "stream": False,
        "format": "json_schema",
    }
    assert structured_think_mode("gpt-oss:20b", num_ctx=32_768) == "medium"
    assert structured_think_mode("muse-glimmer:30b", num_ctx=65_536) == "medium"
    assert structured_think_mode("ornith:35b", num_ctx=114_688) == "medium"
    assert structured_reasoning_output_reservation(3_072) == 4_096
    with pytest.raises(ValueError, match="exceeds runtime limit"):
        structured_reasoning_output_reservation(98_305)
    assert len(structured_generation_policy_sha256()) == 64


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        (
            {"decision_lane": "local_repair", "runtime_role": "classification.primary"},
            "low",
        ),
        ({"runtime_role": "classification.primary"}, "low"),
        ({"runtime_role": "classification.tie_break"}, "low"),
        (
            {"runtime_role": "classification.primary", "task_impact": "high"},
            "high",
        ),
    ],
)
def test_structured_think_mode_selects_verified_candidate_levels(
    overrides: dict[str, object], expected: str
) -> None:
    assert (
        structured_think_mode(
            "local:test",
            num_ctx=16_384,
            required_num_ctx=8_000,
            num_predict=2_048,
            supported_reasoning_levels=("low", "medium", "high"),
            adaptive_reasoning_adopted=True,
            **{"task_impact": "normal", **overrides},
        )
        == expected
    )


def test_structured_think_mode_falls_back_for_capability_and_headroom() -> None:
    common = {
        "model": "local:test",
        "num_ctx": 16_384,
        "required_num_ctx": 8_000,
        "num_predict": 2_048,
        "runtime_role": "classification.tie_break",
        "task_impact": "normal",
        "adaptive_reasoning_adopted": True,
    }
    assert structured_think_mode(**common) == "low"
    assert (
        structured_think_mode(**common, supported_reasoning_levels=("low", "medium"))
        == "low"
    )
    assert (
        structured_think_mode(
            **{**common, "num_ctx": 9_000},
            supported_reasoning_levels=("medium", "high"),
        )
        == "low"
    )
    assert (
        structured_think_mode(
            **{**common, "adaptive_reasoning_adopted": False},
            supported_reasoning_levels=("low", "medium", "high"),
        )
        == "low"
    )


@pytest.mark.parametrize(
    "decision_lane", ["content_correction_review", "ingest_reconciliation"]
)
def test_quality_equivalent_lane_uses_no_reasoning(decision_lane: str) -> None:
    assert (
        structured_think_mode(
            "local:test",
            num_ctx=16_384,
            required_num_ctx=8_000,
            num_predict=2_048,
            runtime_role="classification.primary",
            decision_lane=decision_lane,
            task_impact="normal",
            supported_reasoning_levels=("low", "medium", "high"),
            adaptive_reasoning_adopted=True,
        )
        is False
    )


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


def test_plain_choice_accepts_only_one_exact_id_and_repairs_once() -> None:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["selection_id"],
        "properties": {
            "selection_id": {
                "type": "string",
                "enum": ["apply_available", "confirmed_noop"],
            }
        },
    }
    invalid = '{"selection_id":"apply_available"}'
    transport = QueueTransport(invalid, "apply_available")

    result = _session(transport).run(
        "decide",
        schema,
        plain_choice_field="selection_id",
    )

    assert result.ok is True
    assert result.value == {"selection_id": "apply_available"}
    assert result.repair_turns == 1
    assert [request.schema for request in transport.requests] == [None, None]
    assert (
        "Return exactly one allowed ID as plain text"
        in transport.requests[0].messages[0]["content"]
    )
    assert invalid not in json.dumps(transport.requests[1].messages)
    assert "not exactly one allowed ID" in transport.requests[1].messages[-1]["content"]


def test_plain_choice_fails_closed_after_one_repair() -> None:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["selection_id"],
        "properties": {
            "selection_id": {
                "type": "string",
                "enum": ["apply_available", "confirmed_noop"],
            }
        },
    }
    transport = QueueTransport("wrong", "still_wrong", "apply_available")

    result = _session(transport).run(
        "decide",
        schema,
        plain_choice_field="selection_id",
    )

    assert result.ok is False
    assert result.failure_class == "repair_exhausted"
    assert len(transport.requests) == 2


def test_single_enum_object_automatically_uses_plain_choice() -> None:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["selection_id"],
        "properties": {
            "selection_id": {
                "type": "string",
                "enum": ["apply_available", "confirmed_noop"],
            }
        },
    }
    transport = QueueTransport("confirmed_noop")

    result = _session(transport).run("decide", schema)

    assert result.ok is True
    assert result.value == {"selection_id": "confirmed_noop"}
    assert transport.requests[0].schema is None
    assert "Return exactly one allowed ID" in transport.requests[0].messages[0][
        "content"
    ]


def test_plain_text_contract_is_host_materialized_and_schema_validated() -> None:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["summary"],
        "properties": {"summary": {"type": "string", "maxLength": 2}},
    }

    def decode(text: str) -> dict[str, str]:
        prefix, separator, value = text.partition("|")
        if not separator or prefix.strip() != "SUMMARY":
            raise ValueError("expected SUMMARY | value")
        return {"summary": value.strip()}

    transport = QueueTransport("SUMMARY | too long", "SUMMARY | ok")
    result = _session(transport).run(
        "summarize",
        schema,
        plain_text_contract="SUMMARY | value",
        plain_text_decoder=decode,
    )

    assert result.ok is True
    assert result.value == {"summary": "ok"}
    assert result.repair_turns == 1
    assert [request.schema for request in transport.requests] == [None, None]
    assert "Do not return JSON" in transport.requests[0].messages[0]["content"]
    assert "JSON Schema" not in transport.requests[0].messages[0]["content"]
    assert "plain-text record" in transport.requests[1].messages[-1]["content"]


def test_plain_text_contract_requires_decoder_pair() -> None:
    result = _session(QueueTransport()).run(
        "summarize",
        {
            "type": "object",
            "additionalProperties": False,
            "required": ["summary"],
            "properties": {"summary": {"type": "string"}},
        },
        plain_text_contract="SUMMARY | value",
    )

    assert result.ok is False
    assert result.failure_class == "schema_invalid"


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
    model = options.pop("model", "local:test")
    return LocalStructuredSession(model=model, transport=transport, **options)


def _install_default_local_runtime(
    monkeypatch: pytest.MonkeyPatch,
    *,
    model: str = "local:test",
) -> LLMRuntime:
    adapter = OllamaAdapter()
    runtime = LLMRuntime(
        generation={"librarian.review": GenerationRoute(adapter, model)},
        local_controls={"librarian.review": adapter},
    )
    monkeypatch.setattr(llm_config, "load_default_llm_runtime", lambda: runtime)
    return runtime


def _trace_phases(audit_root: Path) -> list[str]:
    return [
        json.loads(line)["phase"]
        for line in (audit_root / "trace-events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]


def _assert_single_load_before_context(audit_root: Path) -> None:
    phases = _trace_phases(audit_root)
    assert phases.count("load") == 1
    assert phases.index("load") < phases.index("context")


class _RemoteGenerationBackend:
    provider = "remote-test"
    location = RouteLocation.REMOTE

    def __init__(self) -> None:
        self.requests: list[MessageGenerationRequest] = []

    def generate(self, request: object, *, model: str) -> GenerationResult:
        assert isinstance(request, MessageGenerationRequest)
        self.requests.append(request)
        return GenerationResult(
            content='{"decision":"apply","summary":"remote"}',
            provider=self.provider,
            model=model,
            completed=True,
            finish_reason="stop",
            usage=TokenUsage(input_tokens=12, output_tokens=8),
        )


def _reject_ollama_control(monkeypatch: pytest.MonkeyPatch) -> None:
    def rejected(*_args: object, **_kwargs: object) -> None:
        pytest.fail("remote structured generation must not touch Ollama control")

    for name in (
        "model_resource_lease_mode",
        "model_resource_lease",
        "plan_model_residency",
        "unload_named_model",
        "chat",
    ):
        monkeypatch.setattr(ollama, name, rejected)


def test_default_transport_routes_remote_without_ollama_control(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audit_root = tmp_path / "audit"
    backend = _RemoteGenerationBackend()
    runtime = LLMRuntime(
        generation={
            "review.remote": GenerationRoute(backend, "configured-remote-model")
        },
        remote_egress_opt_ins={("review.remote", SourceDataClass.PAGE)},
    )
    monkeypatch.setattr(llm_config, "load_default_llm_runtime", lambda: runtime)
    _reject_ollama_control(monkeypatch)

    result = LocalStructuredSession(
        runtime_role="review.remote",
        source_data_class="page",
        source_sensitivity="high",
        audit_root=audit_root,
    ).run("decide", SCHEMA)

    assert result.ok is True
    assert result.model == "configured-remote-model"
    assert len(backend.requests) == 1
    assert backend.requests[0].source == SourceDataClassification(
        SourceDataClass.PAGE,
        SourceSensitivity.HIGH,
    )
    _assert_single_load_before_context(audit_root)


def test_default_transport_routes_non_ollama_local_without_ollama_control(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audit_root = tmp_path / "audit"
    backend = _RemoteGenerationBackend()
    backend.location = RouteLocation.LOCAL
    runtime = LLMRuntime(
        generation={"review.local": GenerationRoute(backend, "native-local-model")}
    )
    monkeypatch.setattr(llm_config, "load_default_llm_runtime", lambda: runtime)
    _reject_ollama_control(monkeypatch)

    result = LocalStructuredSession(
        runtime_role="review.local",
        source_data_class="page",
        source_sensitivity="high",
        audit_root=audit_root,
    ).run("decide", SCHEMA)

    assert result.ok is True
    assert result.model == "native-local-model"
    assert len(backend.requests) == 1
    assert callable(backend.requests[0].progress_callback)
    _assert_single_load_before_context(audit_root)


def test_stream_progress_persists_only_bounded_counters(tmp_path: Path) -> None:
    audit_root = tmp_path / "audit"
    observed_active: list[dict[str, Any]] = []
    calls = 0

    def transport(request: ChatRequest) -> str:
        nonlocal calls
        calls += 1
        assert request.progress_callback is not None
        if calls == 1:
            request.progress_callback(
                {
                    "output_tokens": 24,
                    "max_output_tokens": 256,
                    "generation_seconds": 0.5,
                    "tokens_per_second": 48.0,
                    "token_count_exact": False,
                    "content": "must never be persisted",
                }
            )
            observed_active.append(
                json.loads(next((audit_root / "active").glob("*.json")).read_text())
            )
            return "not json"
        observed_active.append(
            json.loads(next((audit_root / "active").glob("*.json")).read_text())
        )
        request.progress_callback(
            {
                "output_tokens": 32,
                "max_output_tokens": 256,
                "generation_seconds": 0.6,
                "tokens_per_second": 53.333,
                "token_count_exact": True,
                "reasoning_content": "must never be persisted",
            }
        )
        return '{"decision":"apply","summary":"ok"}'

    result = _session(
        transport,
        audit_root=audit_root,
        role="review:primary",
    ).run("decide", SCHEMA)

    assert result.ok is True
    assert observed_active[0]["generation"] == {
        "output_tokens": 24,
        "max_output_tokens": 256,
        "generation_seconds": 0.5,
        "tokens_per_second": 48.0,
        "token_count_exact": False,
    }
    assert "generation" not in observed_active[1]
    session = json.loads((audit_root / "audit.jsonl").read_text())
    assert session["generation"] == {
        "output_tokens": 32,
        "max_output_tokens": 256,
        "generation_seconds": 0.6,
        "tokens_per_second": 53.333,
        "token_count_exact": True,
    }
    assert "content" not in json.dumps(session)
    trace = [
        json.loads(line)
        for line in (audit_root / "trace-events.jsonl").read_text().splitlines()
    ]
    assert [row["phase"] for row in trace if row["kind"] == "phase"].count(
        "generate"
    ) == 1
    assert [row["phase"] for row in trace if row["kind"] == "phase"].count(
        "repair"
    ) == 1
    assert trace[-1]["generation"] == session["generation"]


def test_resource_managed_default_local_emits_one_load_before_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audit_root = tmp_path / "audit"
    _install_default_local_runtime(monkeypatch)
    monkeypatch.setattr(ollama, "model_resource_lease_mode", lambda: "exclusive")
    monkeypatch.setattr(
        ollama,
        "chat",
        lambda *_args, **_kwargs: ollama.ChatResponse(
            content='{"decision":"apply","summary":"managed"}'
        ),
    )

    result = LocalStructuredSession(
        model="local:test",
        audit_root=audit_root,
        resource_managed=True,
    ).run("decide", SCHEMA)

    assert result.ok is True
    _assert_single_load_before_context(audit_root)


def test_default_transport_egress_denial_is_safe_and_call_free(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = _RemoteGenerationBackend()
    runtime = LLMRuntime(
        generation={"review.remote": GenerationRoute(backend, "remote-model")}
    )
    monkeypatch.setattr(llm_config, "load_default_llm_runtime", lambda: runtime)
    _reject_ollama_control(monkeypatch)

    result = LocalStructuredSession(
        runtime_role="review.remote",
        audit_root=tmp_path / "audit",
    ).run("decide", SCHEMA)

    assert result.ok is False
    assert result.failure_class == "egress_denied"
    assert backend.requests == []


def test_explicit_runtime_location_mismatch_fails_before_backend_or_control(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = _RemoteGenerationBackend()
    runtime = LLMRuntime(
        generation={"review.remote": GenerationRoute(backend, "remote-model")}
    )
    monkeypatch.setattr(llm_config, "load_default_llm_runtime", lambda: runtime)
    _reject_ollama_control(monkeypatch)

    result = LocalStructuredSession(
        model="remote-model",
        runtime_role="review.remote",
        runtime_location="local",
        audit_root=tmp_path / "audit",
    ).run("decide", SCHEMA)

    assert result.ok is False
    assert result.failure_class == "route_configuration_invalid"
    assert backend.requests == []
    assert "load" not in _trace_phases(tmp_path / "audit")


def test_explicit_runtime_role_model_mismatch_fails_before_backend_or_control(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = _RemoteGenerationBackend()
    runtime = LLMRuntime(
        generation={"review.remote": GenerationRoute(backend, "remote-model")}
    )
    monkeypatch.setattr(llm_config, "load_default_llm_runtime", lambda: runtime)
    _reject_ollama_control(monkeypatch)

    result = LocalStructuredSession(
        model="wrong-model",
        runtime_role="review.remote",
        source_data_class="page",
        source_sensitivity="normal",
        audit_root=tmp_path / "audit",
    ).run("decide", SCHEMA)

    assert result.ok is False
    assert result.failure_class == "route_configuration_invalid"
    assert backend.requests == []


def test_custom_transport_does_not_load_default_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        llm_config,
        "load_default_llm_runtime",
        lambda: pytest.fail("custom transport must not load the default runtime"),
    )
    transport = QueueTransport('{"decision":"apply","summary":"custom"}')

    result = _session(transport).run("decide", SCHEMA)

    assert result.ok is True
    assert len(transport.requests) == 1


def test_custom_transport_requires_explicit_synthetic_model() -> None:
    with pytest.raises(ValueError, match="injected transport requires"):
        LocalStructuredSession(
            runtime_role="review.test",
            transport=QueueTransport("{}"),
        )


def test_runtime_role_type_error_is_normalized_to_value_error() -> None:
    with pytest.raises(ValueError, match="runtime_role"):
        LocalStructuredSession(model="local:test", runtime_role=1)  # type: ignore[arg-type]


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


def test_first_pass_valid_uses_fixed_medium_thinking_request() -> None:
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
    assert request.num_predict == 170
    assert request.temperature == 0
    assert request.seed == 0
    assert request.think == "low"
    assert request.schema == SCHEMA
    assert request.messages[0]["role"] == "system"
    assert "untrusted data" in request.messages[0]["content"]
    assert '"decision"' in request.messages[0]["content"]


def test_content_correction_review_skips_initial_reasoning() -> None:
    transport = QueueTransport('{"decision":"apply","summary":"ok"}')

    result = _session(
        transport,
        model=QWEN_STRUCTURED_MODEL,
        runtime_role="classification.primary",
        decision_lane="content_correction_review",
    ).run("decide", SCHEMA)

    assert result.ok is True
    assert transport.requests[0].think is False
    assert transport.requests[0].ollama_think is False
    assert transport.requests[0].think_selection_reason == (
        "quality_equivalent_no_reasoning"
    )


@pytest.mark.parametrize(
    ("model", "initial_think", "initial_ollama_think", "initial_schema"),
    [
        (QWEN_STRUCTURED_MODEL, "low", True, None),
        (MUSE_STRUCTURED_MODEL, "low", "low", None),
        ("Qwen3.8-27B-4bit", "low", "low", None),
        ("Muse-Glimmer-30B-4bit", "low", "low", None),
        ("gemma-4-26b-a4b-it-4bit", "low", "low", None),
        ("gemma4:26b", "low", "low", SCHEMA),
    ],
)
def test_all_model_repairs_use_strict_schema_without_thinking(
    model: str,
    initial_think: bool | str,
    initial_ollama_think: bool | str,
    initial_schema: dict[str, Any] | None,
) -> None:
    transport = QueueTransport(
        '{"decision":"invalid","summary":"bad"}',
        '{"decision":"apply","summary":"ok"}',
    )
    result = _session(transport, model=model).run("decide", SCHEMA)

    assert result.ok is True
    assert [request.schema for request in transport.requests] == [
        initial_schema,
        SCHEMA,
    ]
    assert [request.think for request in transport.requests] == [
        initial_think,
        False,
    ]
    assert [request.ollama_think for request in transport.requests] == [
        initial_ollama_think,
        False,
    ]
    assert result.think is False
    assert result.think_selection_reason == "structured_repair"


def test_qwen_structured_compatibility_keeps_client_validation_fail_closed() -> None:
    transport = QueueTransport('{"decision":"invalid","summary":"bad"}')

    result = _session(
        transport,
        model=QWEN_STRUCTURED_MODEL,
        max_responses=1,
    ).run("decide", SCHEMA)

    assert result.ok is False
    assert result.failure_class == "repair_exhausted"
    assert transport.requests[0].schema is None
    assert transport.requests[0].think == "low"
    assert transport.requests[0].ollama_think is True


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
            "thread_id",
            "think",
            "think_selection_reason",
            "structured_generation_policy_version",
            "structured_generation_policy_sha256",
            "required_num_ctx",
            "requested_num_ctx",
            "context_tokens",
        }
        assert marker["phase"] == "generate"
        assert marker["attempt"] == 0
        assert marker["think"] == "low"
        assert marker["think_selection_reason"] == "adaptive_canary_not_adopted"
        assert marker["context_tokens"] == 32_768
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
    audit = json.loads(audit_text)
    assert audit["think"] == "low"
    assert audit["ollama_think"] == "low"
    assert audit["num_predict"] == 1_365
    assert audit["think_selection_reason"] == "adaptive_canary_not_adopted"
    assert audit["context_tokens"] == 32_768
    assert audit["requested_num_ctx"] == 32_768
    assert isinstance(audit["required_num_ctx"], int)
    assert audit["structured_generation_policy_version"] == 16
    assert audit["structured_generation_policy_sha256"] == (
        structured_generation_policy_sha256()
    )
    assert result.ollama_think == "low"
    assert result.num_predict == 1_365
    assert result.audit_record()["ollama_think"] == "low"
    assert result.audit_record()["num_predict"] == 1_365
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
    audit = json.loads((audit_root / "audit.jsonl").read_text(encoding="utf-8"))
    assert audit["think"] == "low"
    summary = json.loads((audit_root / "summary.json").read_text(encoding="utf-8"))
    assert summary["sessions"]["failures"] == {"transport_error": 1}
    trace = [
        json.loads(line)
        for line in (audit_root / "trace-events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [row["phase"] for row in trace if row["kind"] == "phase"][-1] == ("generate")
    assert not any(row["kind"] == "phase" and row["phase"] == "vote" for row in trace)
    assert trace[-1]["kind"] == "session"
    assert trace[-1]["status"] == "error"


def test_direct_session_repairs_without_thinking() -> None:
    transport = QueueTransport(
        '{"decision":"apply"}',
        '{"decision":"apply","summary":"ok"}',
    )

    result = _session(
        transport,
        model="muse-glimmer:30b-q4k-dynamic",
        runtime_role="classification.challenger",
        decision_lane="local_repair",
    ).run("repair", SCHEMA)

    assert result.ok is True
    assert [request.think for request in transport.requests] == ["low", False]
    assert [request.think_selection_reason for request in transport.requests] == [
        "adaptive_canary_not_adopted",
        "structured_repair",
    ]
    assert result.think is False
    assert result.think_selection_reason == "structured_repair"
    assert result.required_num_ctx == required_structured_context_tokens(
        "repair",
        SCHEMA,
        system=None,
        num_predict=256,
        max_output_chars=1_000,
        max_feedback_chars=2_000,
    )


@pytest.mark.parametrize("runtime_role", REASONING_ROUTES)
def test_production_roles_use_medium_until_canary_then_strict_repair(
    tmp_path: Path,
    runtime_role: str,
) -> None:
    model, _digest = REASONING_ROUTES[runtime_role]
    transport = QueueTransport(
        '{"decision":"apply"}',
        '{"decision":"apply","summary":"ok"}',
    )

    result = _session(
        transport,
        model=model,
        runtime_role=runtime_role,
        decision_lane="local_repair",
        task_impact="high",
        reasoning_authority=_reasoning_authority(runtime_role),
        audit_root=tmp_path / "audit",
        num_ctx=32_768,
        num_predict=3_072,
    ).run("decide", SCHEMA)

    assert result.ok is True
    assert [request.think for request in transport.requests] == ["low", False]
    assert [request.num_predict for request in transport.requests] == [2_048, 2_048]
    assert [request.ollama_think for request in transport.requests] == [
        "low" if runtime_role == "classification.challenger" else True,
        False,
    ]
    assert transport.requests[0].think_selection_reason == (
        "adaptive_canary_not_adopted"
    )
    assert result.ollama_think is False
    assert result.num_predict == 2_048
    audit = json.loads((tmp_path / "audit" / "audit.jsonl").read_text(encoding="utf-8"))
    assert audit["ollama_think"] == result.ollama_think
    assert audit["num_predict"] == 2_048


@pytest.mark.parametrize(("ollama_think", "expected"), [(None, "medium"), (True, True)])
def test_default_transport_forwards_effective_reasoning_to_ollama(
    monkeypatch: pytest.MonkeyPatch,
    ollama_think: bool | None,
    expected: bool | str,
) -> None:
    observed: dict[str, Any] = {}

    def chat(_messages: object, **kwargs: Any) -> ollama.ChatResponse:
        observed.update(kwargs)
        return ollama.ChatResponse(content='{"decision":"apply","summary":"ok"}')

    monkeypatch.setattr(ollama, "runtime_structured_chat", chat)
    request = ChatRequest(
        model="local:test",
        messages=(),
        schema=SCHEMA,
        num_ctx=16_384,
        num_predict=2_048,
        keep_alive="0",
        read_timeout_ms=1_000,
        max_output_chars=1_000,
        think="medium",
        ollama_think=ollama_think,
    )

    local_structured._default_transport(
        request,
        runtime_role="classification.primary",
        expected_model="local:test",
        expected_location="local",
        source_data_class="system",
        source_sensitivity="high",
    )

    assert observed["think"] == expected
    assert observed["num_predict"] == 2_048


@pytest.mark.parametrize(
    ("runtime_role", "engine_version", "model_digest"),
    [
        (
            "classification.primary",
            "0.32.13",
            "09eba04b154c7a50fdc04104d0e5f0d9d535df5ffef7105529a4b001711d3bac",
        ),
        (
            "classification.primary",
            "0.32.12-mixedmlx.1eeb7aad",
            "0" * 64,
        ),
        (
            "classification.challenger",
            "0.32.12-mixedmlx.1eeb7aad",
            "0" * 64,
        ),
        (
            "classification.tie_break",
            "0.32.12-mixedmlx.1eeb7aad",
            "0" * 64,
        ),
    ],
)
def test_reasoning_authority_fails_closed_on_runtime_identity_drift(
    runtime_role: str,
    engine_version: str,
    model_digest: str,
) -> None:
    transport = QueueTransport('{"decision":"apply","summary":"ok"}')
    model, _digest = REASONING_ROUTES[runtime_role]
    result = _session(
        transport,
        model=model,
        runtime_role=runtime_role,
        decision_lane="local_repair",
        reasoning_authority={
            "role": runtime_role,
            "provider": "ollama",
            "model": model,
            "location": "local",
            "ollama": {
                "engine": {"name": "ollama", "version": engine_version},
                "digest": model_digest,
            },
        },
    ).run("repair", SCHEMA)

    assert result.ok is True
    assert transport.requests[0].think == "low"
    assert result.think_selection_reason == "adaptive_canary_not_adopted"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("role", "classification.challenger"),
        ("model", "wrong:model"),
        ("provider", "remote"),
        ("location", "remote"),
    ],
)
def test_reasoning_authority_fails_closed_on_route_identity_drift(
    field: str,
    value: str,
) -> None:
    runtime_role = "classification.primary"
    authority = _reasoning_authority(runtime_role)
    authority[field] = value
    transport = QueueTransport('{"decision":"apply","summary":"ok"}')
    assert not production_reasoning_authority_matches(
        REASONING_ROUTES[runtime_role][0], runtime_role, authority
    )

    result = _session(
        transport,
        model=REASONING_ROUTES[runtime_role][0],
        runtime_role=runtime_role,
        decision_lane="local_repair",
        reasoning_authority=authority,
    ).run("repair", SCHEMA)

    assert result.ok is True
    assert transport.requests[0].think == "low"
    assert transport.requests[0].num_predict == 170
    assert result.think_selection_reason == "adaptive_canary_not_adopted"


def test_exact_reasoning_authority_rejects_oversize_high_budget_at_construction() -> (
    None
):
    runtime_role = "classification.primary"
    num_predict = MAX_OUTPUT_TOKENS * 3 // 4 + 1
    authority = _reasoning_authority(runtime_role)
    assert production_reasoning_authority_matches(
        REASONING_ROUTES[runtime_role][0], runtime_role, authority
    )

    with pytest.raises(ValueError, match="exceeds runtime limit"):
        _session(
            QueueTransport(),
            model=REASONING_ROUTES[runtime_role][0],
            runtime_role=runtime_role,
            reasoning_authority=authority,
            num_predict=num_predict,
        )

    session = _session(
        QueueTransport(),
        model=REASONING_ROUTES[runtime_role][0],
        runtime_role=runtime_role,
        num_predict=num_predict,
    )
    assert session.num_predict == num_predict


def test_reasoning_authority_is_detached_from_external_mutation() -> None:
    authority = _reasoning_authority("classification.challenger")
    transport = QueueTransport('{"decision":"apply","summary":"ok"}')
    session = _session(
        transport,
        model="muse-glimmer:30b-q4k-dynamic",
        runtime_role="classification.challenger",
        decision_lane="local_repair",
        reasoning_authority=authority,
    )

    authority["ollama"]["engine"]["version"] = "0.32.9"
    result = session.run("repair", SCHEMA)

    assert result.ok is True
    assert transport.requests[0].think == "low"
    assert result.think_selection_reason == "adaptive_canary_not_adopted"


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


@pytest.mark.parametrize(
    "field,value",
    (("source_data_class", "private"), ("source_sensitivity", "secret")),
)
def test_source_classification_rejects_unknown_values(field: str, value: str) -> None:
    with pytest.raises(ValueError, match=field):
        LocalStructuredSession(
            model="local:test",
            transport=QueueTransport("{}"),
            **{field: value},
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
        "conservative_veto_bypassed_by_lane_policy": 0,
        "conservative_veto_fired": 0,
        "dissent_effect_classes": {},
        "model_conservative_vote_rates": {},
        "pair_agreement": 1,
        "tie_break_used": 0,
        "total": 2,
        "unresolved_quarantine": 1,
    }


def test_decision_summary_aggregates_veto_dissent_and_model_rates(
    tmp_path: Path,
) -> None:
    store = LocalConsensusAuditStore(tmp_path / "audit")
    store.append(
        {
            "kind": "decision",
            "status": "agreed",
            "conservative_veto_fired": True,
            "conservative_veto_bypassed_by_lane_policy": True,
            "dissent_effect_class": "conservative",
            "votes": [
                {"model": "model-a", "valid": True, "effect_class": "mutating"},
                {
                    "model": "model-b",
                    "valid": True,
                    "effect_class": "conservative",
                },
                {"model": "model-a", "valid": True, "effect_class": "mutating"},
            ],
        }
    )
    store.append(
        {
            "kind": "decision",
            "status": "quarantined",
            "conservative_veto_fired": True,
            "conservative_veto_bypassed_by_lane_policy": False,
            "dissent_effect_class": "unclassifiable",
            "votes": [
                {
                    "model": "model-b",
                    "valid": True,
                    "effect_class": "conservative",
                },
                {
                    "model": "model-c",
                    "valid": False,
                    "effect_class": "conservative",
                },
            ],
        }
    )

    summary = json.loads(store.summary_file.read_text(encoding="utf-8"))
    decisions = summary["decisions"]

    assert summary["schema_version"] == 3
    assert decisions["conservative_veto_fired"] == 2
    assert decisions["conservative_veto_bypassed_by_lane_policy"] == 1
    assert decisions["dissent_effect_classes"] == {
        "conservative": 1,
        "unclassifiable": 1,
    }
    assert decisions["model_conservative_vote_rates"] == {
        "model-a": {
            "valid_votes": 2,
            "conservative_votes": 0,
            "conservative_rate": 0.0,
        },
        "model-b": {
            "valid_votes": 2,
            "conservative_votes": 2,
            "conservative_rate": 1.0,
        },
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
        "load",
        "context",
        "generate",
        "validate",
        "vote",
        "vote",
    ]
    assert all(row["think"] == "low" for row in rows)
    assert all(row["context_tokens"] == 16_384 for row in rows)
    assert all(row["requested_num_ctx"] == 16_384 for row in rows)
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
    invalid = '{"decision":"apply",'
    transport = QueueTransport(
        invalid,
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
    assert "Previous invalid JSON omitted" in second.messages[2]["content"]
    assert invalid not in json.dumps(second.messages)
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


def test_output_cap_fails_closed_after_two_oversize_repairs(tmp_path: Path) -> None:
    transport = QueueTransport("x" * 101, "y" * 101, "z" * 101)
    audit_root = tmp_path / "audit"

    result = _session(transport, audit_root=audit_root, max_output_chars=100).run(
        "decide", SCHEMA
    )

    assert result.ok is False
    assert result.failure_class == "repair_exhausted"
    assert result.repair_turns == 2
    assert len(result.attempts) == 3
    assert all(
        attempt.issues[0].keyword == "maxOutputBytes" for attempt in result.attempts
    )
    trace = [
        json.loads(line)
        for line in (audit_root / "trace-events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [row["phase"] for row in trace if row["kind"] == "phase"][-1] == ("validate")
    assert not any(row["kind"] == "phase" and row["phase"] == "vote" for row in trace)


def test_initial_input_byte_cap_fails_before_call(tmp_path: Path) -> None:
    transport = QueueTransport('{"decision":"apply","summary":"unused"}')
    audit_root = tmp_path / "local-consensus"

    result = _session(
        transport,
        audit_root=audit_root,
        max_input_chars=500,
        max_output_chars=500,
        max_feedback_chars=500,
    ).run("x" * 200, SCHEMA)

    assert result.ok is False
    assert result.failure_class == "input_too_large"
    assert transport.requests == []
    assert "load" not in _trace_phases(audit_root)
    audit = json.loads((audit_root / "audit.jsonl").read_text(encoding="utf-8"))
    assert "think" not in audit
    assert "ollama_think" not in audit
    assert "num_predict" not in audit
    assert result.ollama_think is None
    assert result.num_predict is None
    assert result.audit_record()["ollama_think"] is None
    assert result.audit_record()["num_predict"] is None


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
    audit_root = tmp_path / "audit"
    _install_default_local_runtime(monkeypatch, model="configured:test")
    monkeypatch.setenv(
        "CHRONOVISOR_OLLAMA_RESOURCE_LOCK", str(tmp_path / "resource.lock")
    )
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
        assert kwargs["model"] == "configured:test"
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
        model="legacy:test",
        audit_root=audit_root,
        num_ctx=114_688,
        num_predict=256,
        max_input_chars=20_000,
        max_output_chars=1_000,
        max_feedback_chars=2_000,
        resource_min_num_ctx=32_768,
        resource_max_num_ctx=114_688,
        resource_memory_reserve_gib=8,
    ).run("decide", SCHEMA)

    assert result.ok is True
    assert result.requested_num_ctx == 32_768
    assert result.effective_num_ctx == 114_688
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
    audit = json.loads((audit_root / "audit.jsonl").read_text(encoding="utf-8"))
    assert audit["requested_num_ctx"] == 32_768
    assert audit["context_tokens"] == 114_688
    _assert_single_load_before_context(audit_root)


def test_default_transport_oversize_input_has_no_runner_side_effects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(
        "CHRONOVISOR_OLLAMA_RESOURCE_LOCK", str(tmp_path / "resource.lock")
    )
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

    monkeypatch.setattr(
        llm_config,
        "load_default_llm_runtime",
        lambda: pytest.fail("oversize input must fail before runtime loading"),
    )
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
    assert "load" not in _trace_phases(tmp_path / "audit")


def test_default_transport_maps_resource_lease_timeout_to_capacity_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_default_local_runtime(monkeypatch)

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
    assert "load" not in _trace_phases(tmp_path / "audit")


def test_default_transport_holds_exclusive_lease_across_all_repair_turns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_default_local_runtime(monkeypatch)
    monkeypatch.setenv(
        "CHRONOVISOR_OLLAMA_RESOURCE_LOCK", str(tmp_path / "resource.lock")
    )
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


def test_unsupported_schema_keyword_fails_before_transport(tmp_path: Path) -> None:
    transport = QueueTransport('{"decision":"apply","summary":"unused"}')
    audit_root = tmp_path / "audit"

    result = _session(transport, audit_root=audit_root).run(
        "decide", {"type": "string", "oneOf": []}
    )

    assert result.ok is False
    assert result.failure_class == "schema_invalid"
    assert transport.requests == []
    assert "load" not in _trace_phases(audit_root)


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
    thinking, thinking_changed = normalize_json_output(
        '{"ok":true}\n</think>\n\n{"ok":true}'
    )
    literal, literal_changed = normalize_json_output(
        '{"summary":"literal </think> marker"}'
    )

    assert (fenced, fenced_changed) == ('{"ok":true}', True)
    assert (channel, channel_changed) == ('{"ok":true}', True)
    assert (thinking, thinking_changed) == ('{"ok":true}', True)
    assert (literal, literal_changed) == (
        '{"summary":"literal </think> marker"}',
        False,
    )
    assert (prose, prose_changed) == ('answer: {"ok":true}', False)


def test_reasoning_protocol_prefix_is_normalized_for_any_model() -> None:
    normalized = '{"decision":"apply","summary":"reasoned"}'
    raw = f"status to=user<|message|>{normalized}"
    transport = QueueTransport(raw)
    result = LocalStructuredSession(
        model="local:test",
        transport=transport,
        num_ctx=65_536,
        num_predict=256,
    ).run("decide", SCHEMA)

    assert result.ok is True
    assert result.value == {"decision": "apply", "summary": "reasoned"}
    assert result.attempts[0].normalized is True
    assert (
        result.attempts[0].output_sha256
        == hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    )
    assert transport.requests[0].think == "low"


def test_fixed_reasoning_protocol_prefix_is_normalized_for_any_model() -> None:
    normalized = '{"decision":"apply","summary":"reasoned"}'
    transport = QueueTransport(f" to=user<|message|>{normalized}")
    result = LocalStructuredSession(
        model="muse-glimmer:30b-q4k-dynamic",
        transport=transport,
        num_ctx=65_536,
        num_predict=256,
    ).run("decide", SCHEMA)

    assert result.ok is True
    assert result.value == {"decision": "apply", "summary": "reasoned"}
    assert result.attempts[0].normalized is True
    assert (
        result.attempts[0].output_sha256
        == hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    )
    assert transport.requests[0].think == "low"


@pytest.mark.parametrize(
    ("raw", "keyword"),
    [
        (
            'answer: status to=user<|message|>{"decision":"apply","summary":"unsafe"}',
            "parse",
        ),
        (
            '<|start|>status to=user<|message|>{"decision":"apply","summary":"unsafe"}',
            "parse",
        ),
        (
            'status to=user<|message|x>{"decision":"apply","summary":"unsafe"}',
            "parse",
        ),
        (
            'status to=user<|message|>answer: {"decision":"apply","summary":"unsafe"}',
            "parse",
        ),
        (
            "status to=user<|message|>status to=user<|message|>"
            '{"decision":"apply","summary":"unsafe"}',
            "parse",
        ),
        (
            'answer: to=user<|message|>{"decision":"apply","summary":"unsafe"}',
            "parse",
        ),
        (
            'to=user<|message|x>{"decision":"apply","summary":"unsafe"}',
            "parse",
        ),
        (
            "to=user<|message|>to=user<|message|>"
            '{"decision":"apply","summary":"unsafe"}',
            "parse",
        ),
        (
            'status to=user<|message|>{"decision":"unsafe","summary":"unsafe"}',
            "enum",
        ),
    ],
)
def test_reasoning_protocol_normalization_remains_fail_closed(
    raw: str, keyword: str
) -> None:
    result = LocalStructuredSession(
        model="local:test",
        transport=QueueTransport(raw),
        max_responses=1,
        num_ctx=65_536,
        num_predict=256,
    ).run("decide", SCHEMA)

    assert result.ok is False
    assert result.failure_class == "repair_exhausted"
    assert result.attempts[0].issues[0].keyword == keyword


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
