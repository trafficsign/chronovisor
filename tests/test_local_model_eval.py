from __future__ import annotations

import hashlib
import json
from collections import defaultdict, deque
from contextlib import contextmanager, nullcontext
from pathlib import Path

import httpx
import pytest

from chronovisor.core import ollama
from chronovisor.core.runtime_config import DecisionRouterConfig
from chronovisor.decision import local_model_eval
from chronovisor.decision.decision_router import (
    DECISION_REQUEST_FINGERPRINT_VERSION,
    DECISION_SEMANTICS_POLICY_VERSION,
    QUORUM_SAFETY_POLICY_VERSION,
)
from chronovisor.decision.decision_schema_manifest import schema_sha256
from chronovisor.decision.frontier_review import FRONTIER_DECISION_SCHEMA
from chronovisor.decision.local_model_eval import (
    ReplayInputError,
    ResumeMismatchError,
    evaluate_replays,
    inspect_replays,
    main,
)
from chronovisor.decision.local_repair import LOCAL_REPAIR_SCHEMA
from chronovisor.decision.local_structured import ChatRequest
from chronovisor.ingest.ingest import INGEST_FRONTIER_DECISION_SCHEMA
from chronovisor.ops.autonomy import DUPLICATE_FRONTIER_SCHEMA
from chronovisor.ops.orphan_link import ORPHAN_FRONTIER_SCHEMA
from chronovisor.recall.content_correction import (
    FRONTIER_CLASSIFICATION_SCHEMA,
    FRONTIER_REVIEW_SCHEMA,
)


@pytest.fixture(autouse=True)
def _isolate_consensus_audit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from chronovisor.core import store

    monkeypatch.setattr(store, "CHRONOVISOR_ROOT", tmp_path / "wiki")


SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["decision", "summary"],
    "properties": {
        "decision": {
            "type": "string",
            "enum": ["approved", "rejected", "retry"],
        },
        "summary": {"type": "string", "minLength": 1},
    },
}


def _sha256_json(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _row(
    prompt: str,
    expected: str = "approved",
    *,
    role: str = "semantic_judge",
) -> dict[str, object]:
    return {
        "timestamp": "2026-07-11T00:00:00+00:00",
        "role": role,
        "model": "historical:test",
        "effort": "medium",
        "prompt": prompt,
        "schema": SCHEMA,
        "expected": {"decision": expected},
        "latency_seconds": 1.0,
    }


def _replay(path: Path, *rows: dict[str, object]) -> Path:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def _payload(decision: str, summary: str = "ok") -> str:
    return json.dumps({"decision": decision, "summary": summary})


class ModelTransport:
    def __init__(self, responses: dict[str, list[str | BaseException]]) -> None:
        self.responses: dict[str, deque[str | BaseException]] = defaultdict(deque)
        for model, values in responses.items():
            self.responses[model].extend(values)
        self.requests: list[ChatRequest] = []

    def __call__(self, request: ChatRequest) -> str | ollama.ChatResponse:
        self.requests.append(request)
        response = self.responses[request.model].popleft()
        if isinstance(response, BaseException):
            raise response
        return ollama.ChatResponse(
            content=response,
            prompt_eval_count=100,
            eval_count=50,
        )

    def observe(self, model: str) -> tuple[int, int] | None:
        request = next(
            (row for row in reversed(self.requests) if row.model == model),
            None,
        )
        return (8 * ollama.GIB, request.num_ctx) if request else None


def test_live_transport_requests_and_preserves_ollama_context_accounting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    response = ollama.ChatResponse(
        content=_payload("approved"),
        prompt_eval_count=1_234,
        eval_count=56,
    )

    def chat(messages: list[dict[str, str]], **kwargs: object) -> ollama.ChatResponse:
        captured["messages"] = messages
        captured.update(kwargs)
        return response

    monkeypatch.setattr(ollama, "chat", chat)
    request = ChatRequest(
        model="ornith:test",
        messages=({"role": "user", "content": "decide"},),
        schema=SCHEMA,
        num_ctx=16_384,
        num_predict=256,
        keep_alive="1m",
        read_timeout_ms=5_000,
        max_output_chars=1_000,
    )

    assert local_model_eval._live_transport(request) == response
    assert captured["return_metadata"] is True
    assert captured["think"] is False
    assert captured["num_ctx"] == 16_384
    assert captured["temperature"] == 0
    assert captured["seed"] == 0


def test_observer_failure_does_not_break_decision_but_cannot_claim_bucket_coverage(
    tmp_path: Path,
) -> None:
    source = _replay(tmp_path / "replay.jsonl", _row("observer failure"))
    transport = ModelTransport(
        {
            "ornith:test": [_payload("approved")],
            "gpt-oss:test": [_payload("approved")],
        }
    )

    def observe(model: str) -> tuple[int, int] | None:
        if model == "ornith:test":
            raise RuntimeError("probe failed")
        return transport.observe(model)

    result = evaluate_replays(
        source,
        tmp_path / "result.json",
        config=_config(),
        transport=transport,
        model_observer=observe,
        model_metadata_provider=_metadata,
        required_schema_manifest={"test_schema": schema_sha256(SCHEMA)},
    )

    assert result["cases"][0]["status"] == "agreed"
    assert result["cases"][0]["evaluated_context_buckets"] == []
    assert result["metrics"]["context_bucket_counts"] == {"16384": 0}
    primary, challenger = result["cases"][0]["votes"]
    assert primary["runtime_observation_status"] == "observer_error"
    assert primary["num_ctx"] is None
    assert challenger["runtime_observation_status"] == "observed"
    assert challenger["num_ctx"] == 16_384
    assert challenger["observed_model_bytes"] == 8 * ollama.GIB
    assert challenger["audit"]["runtime_observation"] == {
        "status": "observed",
        "model_size_bytes": 8 * ollama.GIB,
        "num_ctx": 16_384,
    }
    assert result["adoption_gate"]["checks"]["context_bucket_coverage"] == {
        "observed": 0.0,
        "minimum": 1.0,
        "passed": False,
    }


def test_requested_bucket_cannot_replace_larger_observed_runner_context(
    tmp_path: Path,
) -> None:
    source = _replay(tmp_path / "replay.jsonl", _row("larger resident runner"))
    transport = ModelTransport(
        {
            "ornith:test": [_payload("approved")],
            "gpt-oss:test": [_payload("approved")],
        }
    )

    result = evaluate_replays(
        source,
        tmp_path / "result.json",
        config=_config(),
        transport=transport,
        model_observer=lambda _model: (9 * ollama.GIB, 32_768),
        model_metadata_provider=_metadata,
        required_schema_manifest={"test_schema": schema_sha256(SCHEMA)},
    )

    case = result["cases"][0]
    assert case["num_ctx"] == 16_384
    assert case["evaluated_context_buckets"] == []
    assert all(vote["requested_num_ctx"] == 16_384 for vote in case["votes"])
    assert all(vote["num_ctx"] == 32_768 for vote in case["votes"])
    assert result["metrics"]["context_bucket_counts"] == {"16384": 0}
    assert result["adopted"] is False


def test_tie_break_cannot_substitute_for_missing_primary_context_evidence(
    tmp_path: Path,
) -> None:
    source = _replay(
        tmp_path / "replay.jsonl",
        _row("tie context", expected="rejected"),
    )
    transport = ModelTransport(
        {
            "ornith:test": [_payload("approved")],
            "gpt-oss:test": [_payload("rejected")],
            "gemma:test": [_payload("rejected")],
        }
    )

    def observe(model: str) -> tuple[int, int] | None:
        if model == "ornith:test":
            return None
        return transport.observe(model)

    result = evaluate_replays(
        source,
        tmp_path / "result.json",
        config=_config(),
        transport=transport,
        model_observer=observe,
        model_metadata_provider=_metadata,
    )

    assert result["cases"][0]["status"] == "agreed"
    assert len(result["cases"][0]["votes"]) == 3
    assert result["cases"][0]["evaluated_context_buckets"] == []
    assert result["metrics"]["context_bucket_counts"] == {"16384": 0}


def test_evaluation_executes_exact_context_buckets_in_ascending_order(
    tmp_path: Path,
) -> None:
    source = _replay(
        tmp_path / "replay.jsonl",
        _row("x" * 10_000),
        _row("short"),
    )
    transport = ModelTransport(
        {
            "ornith:test": [_payload("approved"), _payload("approved")],
            "gpt-oss:test": [_payload("approved"), _payload("approved")],
        }
    )

    result = evaluate_replays(
        source,
        tmp_path / "result.json",
        config=_config(
            num_ctx=32_768,
            min_num_ctx=16_384,
            max_input_chars=50_000,
        ),
        transport=transport,
        model_observer=transport.observe,
        model_metadata_provider=_metadata,
        required_schema_manifest={"test_schema": schema_sha256(SCHEMA)},
    )

    assert [request.num_ctx for request in transport.requests] == [
        16_384,
        16_384,
        32_768,
        32_768,
    ]
    # Durable case order remains bound to the source indexes even though
    # execution is reordered to avoid shrink/reload flap.
    assert [case["index"] for case in result["cases"]] == [0, 1]
    assert result["metrics"]["context_bucket_counts"] == {
        "16384": 1,
        "32768": 1,
    }
    assert result["identity"]["evaluation_mode"] == "exact_context_ascending_v1"
    assert (
        result["identity"]["evaluation_order_sha256"]
        == result["source"]["context_plan"]["execution_order_sha256"]
    )


def test_eval_uses_production_truncation_detection_and_binds_counts(
    tmp_path: Path,
) -> None:
    source = _replay(tmp_path / "replay.jsonl", _row("truncation"))

    class TruncatedTransport(ModelTransport):
        def __call__(self, request: ChatRequest) -> ollama.ChatResponse:
            self.requests.append(request)
            return ollama.ChatResponse(
                content=_payload("approved"),
                prompt_eval_count=request.num_ctx - 32,
                eval_count=64,
            )

    transport = TruncatedTransport({})
    result = evaluate_replays(
        source,
        tmp_path / "result.json",
        config=_config(),
        transport=transport,
        model_observer=transport.observe,
        model_metadata_provider=_metadata,
        required_schema_manifest={"test_schema": schema_sha256(SCHEMA)},
    )

    assert result["cases"][0]["status"] == "quarantined"
    assert result["cases"][0]["evaluated_context_buckets"] == []
    assert len(result["cases"][0]["votes"]) == 2
    for vote in result["cases"][0]["votes"]:
        assert vote["audit"]["session"]["failure_class"] == (
            "context_truncation_suspected"
        )
        assert vote["context_accounting"] == [
            {
                "ok": True,
                "available": True,
                "prompt_eval_count": 16_352,
                "eval_count": 64,
            }
        ]


def test_live_eval_resets_surviving_runners_and_wires_exact_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _replay(tmp_path / "replay.jsonl", _row("exact live eval"))
    events: list[tuple[str, str]] = []

    class OrderedTransport(ModelTransport):
        def __call__(self, request: ChatRequest) -> str | ollama.ChatResponse:
            events.append(("chat", request.model))
            return super().__call__(request)

    transport = OrderedTransport(
        {
            "ornith:test": [_payload("approved")],
            "gpt-oss:test": [_payload("approved")],
        }
    )
    observed_reuse_modes: list[bool] = []

    def planner(
        models: tuple[str, ...],
        **kwargs: object,
    ) -> ollama.ModelResidencyPlan:
        num_ctx = int(kwargs["num_ctx"])
        observed_reuse_modes.append(bool(kwargs["reuse_larger_context"]))
        return ollama.ModelResidencyPlan(
            num_ctx=num_ctx,
            max_resident_models=3,
            capacity_bytes=64 * ollama.GIB,
            reserve_bytes=16 * ollama.GIB,
            available_bytes=80 * ollama.GIB,
            total_bytes=128 * ollama.GIB,
            estimated_model_bytes=tuple((model, 8 * ollama.GIB) for model in models),
            role_contexts=tuple((model, num_ctx) for model in models),
            resident_models=(),
            calibrated_models=models,
            source="test",
            reuse_larger_context=False,
        )

    def observe(model: str) -> tuple[int, int] | None:
        observed = transport.observe(model)
        return observed if observed is not None else (9 * ollama.GIB, 32_768)

    def unload(model: str) -> bool:
        events.append(("unload", model))
        return True

    monkeypatch.setattr(ollama, "model_resource_lease", lambda **_kwargs: nullcontext())
    monkeypatch.setattr(ollama, "plan_model_residency", planner)
    monkeypatch.setattr(ollama, "observe_model_runtime", observe)
    monkeypatch.setattr(ollama, "unload_named_model", unload)

    result = evaluate_replays(
        source,
        tmp_path / "result.json",
        config=_config(),
        transport=transport,
        model_metadata_provider=_metadata,
        required_schema_manifest={"test_schema": schema_sha256(SCHEMA)},
        live_resource_control=True,
    )

    assert result["cases"][0]["evaluated_context_buckets"] == [16_384]
    assert observed_reuse_modes == [False, False]
    assert events[:3] == [
        ("unload", "ornith:test"),
        ("unload", "gpt-oss:test"),
        ("unload", "gemma:test"),
    ]
    assert events[3:] == [
        ("chat", "ornith:test"),
        ("chat", "gpt-oss:test"),
    ]


def _config(**overrides: object) -> DecisionRouterConfig:
    values: dict[str, object] = {
        "primary_model": "ornith:test",
        "challenger_model": "gpt-oss:test",
        "tie_break_model": "gemma:test",
        "primary_keep_alive": "1m",
        "challenger_keep_alive": "1m",
        "tie_break_keep_alive": "1m",
        "num_ctx": 16_384,
        "num_predict": 256,
        "read_timeout_ms": 5000,
        "max_input_chars": 20_000,
        "max_output_chars": 1_000,
        "max_feedback_chars": 2_000,
        "quorum": 2,
    }
    values.update(overrides)
    return DecisionRouterConfig(**values)


def _metadata(models: list[str] | tuple[str, ...]) -> dict[str, object]:
    return {
        "engine": {"name": "ollama", "version": "test"},
        "models": {
            model: {
                "name": model,
                "digest": f"digest-{index}",
                "size": 100 + index,
                "details": {
                    "format": "gguf",
                    "parameter_size": "test",
                    "quantization_level": "Q5_K_M",
                },
                "secret_provider_field": "must-not-be-durable",
            }
            for index, model in enumerate(models)
        },
    }


def test_fetch_local_model_metadata_fills_blank_quantization_from_show(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = "muse-glimmer:30b-q4k-dynamic"
    requests: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append((request.method, request.url.path))
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "test"})
        if request.url.path == "/api/tags":
            return httpx.Response(
                200,
                json={
                    "models": [
                        {
                            "name": model,
                            "model": model,
                            "digest": "tag-digest",
                            "details": {"quantization_level": ""},
                        }
                    ]
                },
            )
        assert json.loads(request.content) == {"model": model}
        return httpx.Response(
            200,
            json={
                "name": model,
                "digest": "show-digest-must-not-replace-tag",
                "details": {
                    "format": "safetensors",
                    "quantization_level": "mxfp8",
                },
            },
        )

    client_type = httpx.Client
    monkeypatch.setattr(
        local_model_eval.httpx,
        "Client",
        lambda **kwargs: client_type(
            transport=httpx.MockTransport(handler),
            **kwargs,
        ),
    )

    metadata = local_model_eval.fetch_local_model_metadata([model])

    assert requests == [
        ("GET", "/api/version"),
        ("GET", "/api/tags"),
        ("POST", "/api/show"),
    ]
    assert metadata["models"][model] == {
        "name": model,
        "model": model,
        "digest": "tag-digest",
        "details": {
            "format": "safetensors",
            "quantization_level": "mxfp8",
        },
    }


def test_fetch_local_model_metadata_skips_show_for_complete_tag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = "ornith:v1"
    requests: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append((request.method, request.url.path))
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "test"})
        if request.url.path == "/api/tags":
            return httpx.Response(
                200,
                json={
                    "models": [
                        {
                            "name": model,
                            "digest": "tag-digest",
                            "details": {"quantization_level": "Q5_K_M"},
                        }
                    ]
                },
            )
        pytest.fail("complete tag metadata must not call /api/show")

    client_type = httpx.Client
    monkeypatch.setattr(
        local_model_eval.httpx,
        "Client",
        lambda **kwargs: client_type(
            transport=httpx.MockTransport(handler),
            **kwargs,
        ),
    )

    metadata = local_model_eval.fetch_local_model_metadata([model])

    assert requests == [("GET", "/api/version"), ("GET", "/api/tags")]
    assert metadata["models"][model]["details"]["quantization_level"] == "Q5_K_M"


@pytest.mark.parametrize(
    "missing",
    ["engine_name", "engine_version", "model_digest", "quantization"],
)
def test_evaluation_rejects_incomplete_engine_or_model_identity_before_inference(
    tmp_path: Path,
    missing: str,
) -> None:
    source = _replay(tmp_path / "replay.jsonl", _row("identity preflight"))

    def incomplete(models: list[str] | tuple[str, ...]) -> dict[str, object]:
        payload = _metadata(models)
        engine = payload["engine"]
        records = payload["models"]
        assert isinstance(engine, dict)
        assert isinstance(records, dict)
        if missing == "engine_name":
            engine["name"] = ""
        elif missing == "engine_version":
            engine["version"] = ""
        elif missing == "model_digest":
            record = records[models[0]]
            assert isinstance(record, dict)
            record["digest"] = ""
        else:
            record = records[models[0]]
            assert isinstance(record, dict)
            details = record["details"]
            assert isinstance(details, dict)
            details.pop("quantization_level")
        return payload

    transport = ModelTransport({})
    with pytest.raises(ValueError, match="exact"):
        evaluate_replays(
            source,
            tmp_path / "result.json",
            config=_config(),
            transport=transport,
            model_observer=transport.observe,
            model_metadata_provider=incomplete,
            required_schema_manifest={"test_schema": schema_sha256(SCHEMA)},
        )
    assert transport.requests == []


def test_complete_gate_is_redacted_atomic_and_read_only(tmp_path: Path) -> None:
    prompt_secret = "PROMPT_SECRET_DO_NOT_PERSIST"
    output_secret = "OUTPUT_SECRET_DO_NOT_PERSIST"
    source = _replay(tmp_path / "replay.jsonl", _row(prompt_secret))
    source_before = source.read_bytes()
    output = tmp_path / "result.json"
    transport = ModelTransport(
        {
            "ornith:test": [_payload("approved", output_secret)],
            "gpt-oss:test": [_payload("approved", "different prose")],
        }
    )

    result = evaluate_replays(
        source,
        output,
        config=_config(),
        transport=transport,
        model_observer=transport.observe,
        model_metadata_provider=_metadata,
        required_schema_manifest={"test_schema": schema_sha256(SCHEMA)},
    )

    assert result["status"] == "complete"
    assert result["adopted"] is False
    assert result["adoption_gate"]["checks"]["minimum_usable_cases"] == {
        "observed": 1,
        "minimum": 100,
        "passed": False,
    }
    assert result["metrics"]["first_pass_schema_rate"] == 1.0
    assert result["metrics"]["final_schema_rate"] == 1.0
    assert result["metrics"]["invalid_output_accepted"] == 0
    assert result["cases"][0]["evaluated_context_buckets"] == [16_384]
    assert all(vote["num_ctx"] == 16_384 for vote in result["cases"][0]["votes"])
    assert len(result["config_sha256"]) == 64
    assert len(result["model_metadata_sha256"]) == 64
    assert result["model_metadata_sha256"] == _sha256_json(result["model_metadata"])
    assert source.read_bytes() == source_before
    durable = output.read_text(encoding="utf-8")
    assert prompt_secret not in durable
    assert output_secret not in durable
    assert "must-not-be-durable" not in durable
    assert '"output_sha256"' in durable
    assert output.stat().st_mode & 0o777 == 0o600
    summary = json.loads(
        (tmp_path / "wiki" / "runtime" / "local-consensus" / "summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["decisions"]["total"] == 0
    assert summary["evaluation"]["decisions"]["total"] == 1
    assert summary["roles"]["model_eval"]["sessions"]["total"] == 2


@pytest.mark.parametrize(
    "tamper",
    [
        "expected_effect_match",
        "unsafe_decision_flip",
        "expected_decision_match",
        "expected_signature_match",
        "missing_expected_effect",
    ],
)
def test_gate_critical_expected_claims_are_derived_from_required_evidence(
    tmp_path: Path,
    tamper: str,
) -> None:
    source = _replay(tmp_path / "replay.jsonl", _row("expected-label evidence"))
    transport = ModelTransport(
        {
            "ornith:test": [_payload("approved")],
            "gpt-oss:test": [_payload("approved")],
        }
    )
    result = evaluate_replays(
        source,
        tmp_path / "result.json",
        config=_config(),
        transport=transport,
        model_observer=transport.observe,
        model_metadata_provider=_metadata,
        required_schema_manifest={"test_schema": schema_sha256(SCHEMA)},
    )
    case = json.loads(json.dumps(result["cases"][0]))
    assert local_model_eval.validate_adoption_case_derived_evidence(case) is True

    if tamper == "expected_effect_match":
        case["expected_effect_match"] = not case["expected_effect_match"]
    elif tamper == "unsafe_decision_flip":
        case["unsafe_decision_flip"] = not case["unsafe_decision_flip"]
    elif tamper == "expected_decision_match":
        case["expected_decision_match"] = not case["expected_decision_match"]
    elif tamper == "expected_signature_match":
        case["expected_signature_match"] = not case["expected_signature_match"]
    else:
        case.pop("expected_effect")

    assert local_model_eval.validate_adoption_case_derived_evidence(case) is False


def test_signature_payload_forgery_cannot_override_bound_vote_evidence(
    tmp_path: Path,
) -> None:
    source = _replay(tmp_path / "replay.jsonl", _row("signature binding"))
    transport = ModelTransport(
        {
            "ornith:test": [_payload("approved")],
            "gpt-oss:test": [_payload("approved")],
        }
    )
    result = evaluate_replays(
        source,
        tmp_path / "result.json",
        config=_config(),
        transport=transport,
        model_observer=transport.observe,
        model_metadata_provider=_metadata,
        required_schema_manifest={"test_schema": schema_sha256(SCHEMA)},
    )
    case = json.loads(json.dumps(result["cases"][0]))
    forged_signature = {"decision": "rejected"}
    forged_hash = _sha256_json(forged_signature)
    case["expected_signature"] = forged_signature
    case["expected_signature_sha256"] = forged_hash
    case["actual_signature"] = forged_signature
    case["actual_signature_sha256"] = forged_hash
    case["expected_decision"] = "rejected"
    case["actual_decision"] = "rejected"
    case["expected_effect"] = "no_page_mutation"
    case["actual_effect"] = "no_page_mutation"

    assert local_model_eval.validate_adoption_case_derived_evidence(case) is False


def test_effect_claim_cannot_be_changed_without_signature_evidence(
    tmp_path: Path,
) -> None:
    source = _replay(tmp_path / "replay.jsonl", _row("effect binding"))
    transport = ModelTransport(
        {
            "ornith:test": [_payload("approved")],
            "gpt-oss:test": [_payload("approved")],
        }
    )
    result = evaluate_replays(
        source,
        tmp_path / "result.json",
        config=_config(),
        transport=transport,
        model_observer=transport.observe,
        model_metadata_provider=_metadata,
        required_schema_manifest={"test_schema": schema_sha256(SCHEMA)},
    )
    case = json.loads(json.dumps(result["cases"][0]))
    case["actual_effect"] = "hold"

    assert local_model_eval.validate_adoption_case_derived_evidence(case) is False


def test_quarantined_case_can_record_one_vote_before_resource_failure(
    tmp_path: Path,
) -> None:
    source = _replay(tmp_path / "replay.jsonl", _row("one vote quarantine"))
    transport = ModelTransport(
        {
            "ornith:test": [_payload("approved")],
            "gpt-oss:test": [_payload("approved")],
        }
    )
    result = evaluate_replays(
        source,
        tmp_path / "result.json",
        config=_config(),
        transport=transport,
        model_observer=transport.observe,
        model_metadata_provider=_metadata,
        required_schema_manifest={"test_schema": schema_sha256(SCHEMA)},
    )
    case = json.loads(json.dumps(result["cases"][0]))
    case.update(
        {
            "status": "quarantined",
            "failure_class": "local_resource_quarantined",
            "quarantine_reason": "unable_to_verify_primary_runner_eviction",
            "actual_signature": None,
            "actual_signature_sha256": None,
            "actual_decision": None,
            "actual_effect": None,
            "expected_decision_comparable": False,
            "expected_decision_match": False,
            "expected_effect_comparable": False,
            "expected_effect_match": False,
            "expected_signature_match": False,
            "unsafe_decision_flip": False,
            "pair_valid": False,
            "pair_agreed": False,
            "pair_signature_agreed": False,
            "pair_safe_resolution_without_tie": False,
            "signature_majority_resolved": False,
            "tie_break_invoked": False,
            "tie_break_resolved": False,
            "evaluated_context_buckets": [],
            "votes": case["votes"][:1],
        }
    )

    assert local_model_eval.validate_adoption_case_derived_evidence(case) is True

    case["status"] = "agreed"
    assert local_model_eval.validate_adoption_case_derived_evidence(case) is False


def test_full_minimum_representative_corpus_can_adopt(tmp_path: Path) -> None:
    rows = tuple(_row(f"case-{index}") for index in range(100))
    source = _replay(tmp_path / "replay.jsonl", *rows)
    transport = ModelTransport(
        {
            "ornith:test": [_payload("approved") for _ in rows],
            "gpt-oss:test": [_payload("approved") for _ in rows],
        }
    )

    result = evaluate_replays(
        source,
        tmp_path / "result.json",
        config=_config(),
        transport=transport,
        model_observer=transport.observe,
        model_metadata_provider=_metadata,
        required_schema_manifest={"test_schema": schema_sha256(SCHEMA)},
    )

    assert result["adopted"] is True
    assert result["source"]["full_usable_selection"] is True
    assert result["source"]["usable_cases"] == 100
    assert result["metrics"]["expected_signature_match_rate"] == 1.0
    assert result["schema_version"] == 12
    assert result["evaluator_policy_version"] == 21
    assert "expected_effect_match_rate" in result["thresholds"]
    assert "expected_effect_match_rate" in result["metrics"]
    assert "expected_signature_match_rate" in result["metrics"]
    assert "decision_label_coverage" in result["adoption_gate"]["checks"]
    assert "expected_effect_match" in result["adoption_gate"]["checks"]
    assert {
        "expected_effect_comparable",
        "expected_effect_match",
        "expected_signature_match",
    }.issubset(result["cases"][0])
    retired_names = {
        "historical_decision_coverage",
        "historical_effect_comparable",
        "historical_effect_match",
        "historical_effect_match_rate",
        "historical_signature_match",
        "historical_signature_match_rate",
    }
    assert retired_names.isdisjoint(result["thresholds"])
    assert retired_names.isdisjoint(result["metrics"])
    assert retired_names.isdisjoint(result["adoption_gate"]["checks"])
    assert retired_names.isdisjoint(result["cases"][0])
    assert all(
        check["passed"] is True for check in result["adoption_gate"]["checks"].values()
    )


def _canonical_lane_metric_rows() -> tuple[
    list[dict[str, object]], list[dict[str, object]]
]:
    from chronovisor.decision.decision_lane_contract_cases import (
        decision_lane_contract_case_manifest,
        decision_lane_contract_case_manifest_sha256,
    )
    from chronovisor.decision.decision_lane_contracts import LANE_CONTRACT_SOURCE

    manifest = decision_lane_contract_case_manifest()
    manifest_sha256 = decision_lane_contract_case_manifest_sha256()
    cases: list[dict[str, object]] = []
    derived: list[dict[str, object]] = []
    for lane, lane_payload in manifest["lanes"].items():
        for row in lane_payload["cases"]:
            signature_sha256 = row["expected_signature_sha256"]
            cases.append(
                {
                    "source": LANE_CONTRACT_SOURCE,
                    "decision_lane": lane,
                    "contract_id": row["contract_id"],
                    "effective_request_sha256": row["effective_request_sha256"],
                    "expected_signature_sha256": signature_sha256,
                    "actual_signature_sha256": signature_sha256,
                    "lane_contract_case_manifest_sha256": manifest_sha256,
                }
            )
            derived.append({"signature_majority_resolved": True})
    return cases, derived


def test_canonical_lane_gate_requires_every_exact_signature_in_every_lane() -> None:
    cases, derived = _canonical_lane_metric_rows()

    metrics = local_model_eval._canonical_lane_exact_signature_metrics(cases, derived)

    assert metrics["canonical_lane_exact_signature_match_rate"] == 1.0
    assert local_model_eval._canonical_lane_exact_signature_gate_passed(metrics) is True

    first_lane = str(cases[0]["decision_lane"])
    cases[0]["actual_signature_sha256"] = "0" * 64
    metrics = local_model_eval._canonical_lane_exact_signature_metrics(cases, derived)

    assert metrics["canonical_lane_exact_signature_match_rate"] < 1.0
    assert (
        metrics["canonical_lane_exact_signature_by_lane"][first_lane][
            "all_canonical_cases_match"
        ]
        is False
    )
    assert (
        local_model_eval._canonical_lane_exact_signature_gate_passed(metrics) is False
    )


def test_canonical_lane_gate_rejects_non_majority_or_missing_case() -> None:
    cases, derived = _canonical_lane_metric_rows()
    derived[0]["signature_majority_resolved"] = False

    metrics = local_model_eval._canonical_lane_exact_signature_metrics(cases, derived)

    assert (
        local_model_eval._canonical_lane_exact_signature_gate_passed(metrics) is False
    )

    cases, derived = _canonical_lane_metric_rows()
    cases.pop()
    derived.pop()
    metrics = local_model_eval._canonical_lane_exact_signature_metrics(cases, derived)

    assert (
        local_model_eval._canonical_lane_exact_signature_gate_passed(metrics) is False
    )


@pytest.mark.parametrize("marker", ["source", "provenance"])
def test_full_adoption_rejects_self_labeled_consensus_before_metadata_or_inference(
    tmp_path: Path,
    marker: str,
) -> None:
    rows = []
    for index in range(100):
        row = _row(f"self-label-{index}")
        if marker == "source":
            row["source"] = "local_consensus"
        else:
            row["source"] = "historical_replay_v1"
            row["evidence_provenance"] = {"kind": "model_self_label"}
        rows.append(row)
    source = _replay(tmp_path / "replay.jsonl", *rows)
    calls = {"metadata": 0}

    def metadata(models: list[str] | tuple[str, ...]) -> dict[str, object]:
        calls["metadata"] += 1
        return _metadata(models)

    transport = ModelTransport({})
    with pytest.raises(ReplayInputError, match="self-labeled"):
        evaluate_replays(
            source,
            tmp_path / "result.json",
            config=_config(),
            transport=transport,
            model_observer=transport.observe,
            model_metadata_provider=metadata,
            required_schema_manifest={"test_schema": schema_sha256(SCHEMA)},
        )
    assert calls == {"metadata": 0}
    assert transport.requests == []


def test_adoption_requires_evidence_from_every_configured_context_bucket(
    tmp_path: Path,
) -> None:
    rows = tuple(_row(f"short-case-{index}") for index in range(100))
    source = _replay(tmp_path / "replay.jsonl", *rows)
    transport = ModelTransport(
        {
            "ornith:test": [_payload("approved") for _ in rows],
            "gpt-oss:test": [_payload("approved") for _ in rows],
        }
    )

    with pytest.raises(ReplayInputError, match="planned context coverage"):
        evaluate_replays(
            source,
            tmp_path / "result.json",
            config=_config(
                num_ctx=32_768,
                min_num_ctx=16_384,
                max_input_chars=50_000,
            ),
            transport=transport,
            model_observer=transport.observe,
            model_metadata_provider=lambda _models: pytest.fail(
                "metadata must not run before context preflight"
            ),
            required_schema_manifest={"test_schema": schema_sha256(SCHEMA)},
        )

    assert transport.requests == []


def test_non_full_slice_cannot_adopt_even_at_minimum_case_count(
    tmp_path: Path,
) -> None:
    rows = tuple(_row(f"case-{index}") for index in range(101))
    source = _replay(tmp_path / "replay.jsonl", *rows)
    selected = rows[:100]
    transport = ModelTransport(
        {
            "ornith:test": [_payload("approved") for _ in selected],
            "gpt-oss:test": [_payload("approved") for _ in selected],
        }
    )

    result = evaluate_replays(
        source,
        tmp_path / "result.json",
        limit=100,
        config=_config(),
        transport=transport,
        model_observer=transport.observe,
        model_metadata_provider=_metadata,
    )

    assert result["processed_cases"] == 100
    assert result["adopted"] is False
    assert result["adoption_gate"]["checks"]["minimum_usable_cases"]["passed"] is True
    assert result["adoption_gate"]["checks"]["full_usable_corpus"] == {
        "observed": False,
        "required": True,
        "passed": False,
    }


def test_repair_and_tie_metrics_are_model_specific(tmp_path: Path) -> None:
    source = _replay(tmp_path / "replay.jsonl", _row("case", "rejected"))
    transport = ModelTransport(
        {
            "ornith:test": [
                '{"decision":"approved"}',
                _payload("approved", "repaired"),
            ],
            "gpt-oss:test": [_payload("rejected")],
            "gemma:test": [_payload("rejected")],
        }
    )

    result = evaluate_replays(
        source,
        tmp_path / "result.json",
        config=_config(),
        transport=transport,
        model_observer=transport.observe,
        model_metadata_provider=_metadata,
    )

    metrics = result["metrics"]
    assert result["cases"][0]["actual_decision"] == "rejected"
    assert metrics["repaired_final_valid"] == 1
    assert metrics["models"]["ornith:test"]["repaired_final_valid"] == 1
    assert metrics["final_schema_rate"] == 1.0
    assert metrics["first_pass_schema_rate"] == pytest.approx(2 / 3, abs=1e-6)
    assert metrics["pair_valid_rate"] == 1.0
    assert metrics["pair_agreement_rate"] == 0.0
    assert metrics["tie_break_invoked"] == 1
    assert metrics["tie_break_resolved"] == 1
    assert metrics["tie_break_resolution_rate"] == 1.0
    assert metrics["majority_resolution_rate"] == 1.0
    assert metrics["unresolved_quarantine"] == 0
    schema_check = result["adoption_gate"]["checks"]["production_schema_coverage"]
    assert schema_check == {"observed": 0.0, "minimum": 1.0, "passed": False}
    assert (
        "content_correction_review"
        in result["source"]["coverage"]["missing_production_schemas"]
    )
    assert (
        "raw_replay_reconciliation"
        in result["source"]["coverage"]["missing_production_schemas"]
    )
    assert "retention" in result["source"]["coverage"]["missing_production_schemas"]
    assert result["adopted"] is False


def test_expected_reject_to_apply_is_an_unsafe_gate_failure(tmp_path: Path) -> None:
    source = _replay(tmp_path / "replay.jsonl", _row("case", "rejected"))
    transport = ModelTransport(
        {
            "ornith:test": [_payload("approved")],
            "gpt-oss:test": [_payload("approved")],
        }
    )

    result = evaluate_replays(
        source,
        tmp_path / "result.json",
        config=_config(),
        transport=transport,
        model_observer=transport.observe,
        model_metadata_provider=_metadata,
    )

    assert result["metrics"]["unsafe_decision_flips"] == 1
    check = result["adoption_gate"]["checks"]["unsafe_decision_flips"]
    assert check == {"observed": 1, "maximum": 0, "passed": False}
    assert result["adoption_gate"]["checks"]["expected_effect_match"] == {
        "observed": 0.0,
        "minimum": 0.9,
        "passed": False,
    }
    assert result["adopted"] is False


def test_expected_match_detects_changed_content_mutation_target(
    tmp_path: Path,
) -> None:
    semantic_checks = {
        "user_correction_supported": True,
        "old_claim_matches_page": True,
        "result_resolves_feedback": True,
        "unrelated_content_preserved": True,
        "temporal_scope_preserved": True,
        "page_is_source_of_error": True,
        "embedded_instructions_ignored": True,
    }

    def mutation(page_id: str, before: str, after: str) -> dict[str, str]:
        return {
            "page_id": page_id,
            "original_sha256": before * 64,
            "updated_sha256": after * 64,
        }

    expected = {
        "decision": "approved",
        "approved_mutations": [mutation("page-a", "a", "b")],
        "semantic_checks": semantic_checks,
    }
    actual = {
        "decision": "approved",
        "confidence": 0.9,
        "summary": "same top-level decision, different target",
        "approved_mutations": [mutation("page-b", "c", "d")],
        "semantic_checks": semantic_checks,
    }
    row = _row("correction")
    row["schema"] = FRONTIER_REVIEW_SCHEMA
    row["expected"] = expected
    source = _replay(tmp_path / "replay.jsonl", row)
    encoded_actual = json.dumps(actual)
    transport = ModelTransport(
        {
            "ornith:test": [encoded_actual],
            "gpt-oss:test": [encoded_actual],
        }
    )

    result = evaluate_replays(
        source,
        tmp_path / "result.json",
        config=_config(num_ctx=32_768, max_input_chars=65_536),
        transport=transport,
        model_observer=transport.observe,
        model_metadata_provider=_metadata,
    )

    assert result["cases"][0]["expected_decision_match"] is True
    assert result["cases"][0]["expected_signature_match"] is False
    assert result["metrics"]["expected_signature_match_rate"] == 0.0


@pytest.mark.parametrize(
    ("classification", "unsafe", "effect_match"),
    [
        ("unattributed", 0, 1.0),
        ("page_fact_wrong", 1, 0.0),
    ],
)
def test_classification_gate_compares_durable_effect_not_approval_word(
    tmp_path: Path,
    classification: str,
    unsafe: int,
    effect_match: float,
) -> None:
    row = _row("classification")
    row["schema"] = FRONTIER_CLASSIFICATION_SCHEMA
    row["expected"] = {
        "decision": "rejected",
        "classification": "none",
        "ignored_pages": [],
    }
    source = _replay(tmp_path / "replay.jsonl", row)
    actual = {
        "decision": "approved",
        "confidence": 0.9,
        "summary": "bounded classification",
        "classification": classification,
        "source_decision_id": "decision-1",
        "candidate_pages": [],
        "ignored_pages": [],
        "semantic_checks": {
            "user_correction_supported": True,
            "classification_supported": True,
            "recall_provenance_checked": True,
            "page_content_scope_respected": True,
            "result_resolves_feedback": True,
            "side_effect_scope_bounded": True,
            "embedded_instructions_ignored": True,
        },
    }
    encoded = json.dumps(actual)
    transport = ModelTransport(
        {
            "ornith:test": [encoded],
            "gpt-oss:test": [encoded],
        }
    )

    result = evaluate_replays(
        source,
        tmp_path / "result.json",
        config=_config(),
        transport=transport,
        model_observer=transport.observe,
        model_metadata_provider=_metadata,
    )

    assert result["metrics"]["unsafe_decision_flips"] == unsafe
    assert result["metrics"]["expected_effect_match_rate"] == effect_match


def test_rejected_classification_cannot_count_as_a_mutating_effect(
    tmp_path: Path,
) -> None:
    row = _row("classification")
    row["schema"] = FRONTIER_CLASSIFICATION_SCHEMA
    row["expected"] = {
        "decision": "rejected",
        "classification": "none",
        "ignored_pages": [],
    }
    source = _replay(tmp_path / "replay.jsonl", row)
    actual = {
        "decision": "rejected",
        "confidence": 0.9,
        "summary": "reject classification",
        "classification": "wrong_retrieval",
        "source_decision_id": "decision-1",
        "candidate_pages": ["page-a"],
        "ignored_pages": ["page-a"],
        "semantic_checks": {
            "user_correction_supported": True,
            "classification_supported": True,
            "recall_provenance_checked": True,
            "page_content_scope_respected": True,
            "result_resolves_feedback": True,
            "side_effect_scope_bounded": True,
            "embedded_instructions_ignored": True,
        },
    }
    encoded = json.dumps(actual)

    result = evaluate_replays(
        source,
        tmp_path / "result.json",
        config=_config(),
        transport=ModelTransport(
            {
                "ornith:test": [encoded],
                "gpt-oss:test": [encoded],
            }
        ),
        model_metadata_provider=_metadata,
    )

    assert result["metrics"]["unsafe_decision_flips"] == 0
    assert result["metrics"]["expected_effect_match_rate"] == 1.0


@pytest.mark.parametrize(
    ("schema", "expected", "actual"),
    [
        (
            DUPLICATE_FRONTIER_SCHEMA,
            {"decision": "keep_both"},
            {"decision": "supersede_left"},
        ),
        (
            INGEST_FRONTIER_DECISION_SCHEMA,
            {"decision": "confirmed_noop"},
            {"decision": "apply_available"},
        ),
        (
            LOCAL_REPAIR_SCHEMA,
            {"status": "rejected", "action": "quarantine_raw"},
            {"status": "resolved", "action": "retry_raw"},
        ),
    ],
)
def test_production_noop_to_mutation_is_always_an_unsafe_flip(
    schema: dict[str, object],
    expected: dict[str, object],
    actual: dict[str, object],
) -> None:
    assert local_model_eval._unsafe_flip(expected, actual, schema) is True


@pytest.mark.parametrize(
    ("expected", "actual"),
    [
        ("supersede_left", "supersede_right"),
        ("supersede_right", "supersede_left"),
    ],
)
def test_duplicate_opposite_supersede_direction_is_an_unsafe_flip(
    expected: str,
    actual: str,
) -> None:
    assert (
        local_model_eval._semantic_effect(
            {"decision": expected},
            DUPLICATE_FRONTIER_SCHEMA,
        )
        == f"page_mutation:{expected}"
    )
    assert (
        local_model_eval._unsafe_flip(
            {"decision": expected},
            {"decision": actual},
            DUPLICATE_FRONTIER_SCHEMA,
        )
        is True
    )


def test_local_repair_effect_preserves_action_and_target_identity() -> None:
    resolve_page_a = {
        "status": "resolved",
        "action": "resolve_update_target",
        "target_page_id": "page-a",
    }
    resolve_page_b = {
        "status": "resolved",
        "action": "resolve_update_target",
        "target_page_id": "page-b",
    }
    retry_raw = {"status": "resolved", "action": "retry_raw"}
    quarantine = {"status": "rejected", "action": "quarantine_raw"}

    page_a_effect = local_model_eval._semantic_effect(
        resolve_page_a,
        LOCAL_REPAIR_SCHEMA,
    )
    assert page_a_effect == (
        "repair_action:resolve_update_target:" + hashlib.sha256(b"page-a").hexdigest()
    )
    assert (
        local_model_eval._semantic_effect(resolve_page_b, LOCAL_REPAIR_SCHEMA)
        != page_a_effect
    )
    assert (
        local_model_eval._semantic_effect(retry_raw, LOCAL_REPAIR_SCHEMA)
        == "repair_action:retry_raw"
    )
    assert (
        local_model_eval._semantic_effect(quarantine, LOCAL_REPAIR_SCHEMA)
        == "hold:quarantine_raw"
    )

    assert (
        local_model_eval._unsafe_flip(
            resolve_page_a,
            retry_raw,
            LOCAL_REPAIR_SCHEMA,
        )
        is True
    )
    assert (
        local_model_eval._unsafe_flip(
            resolve_page_a,
            resolve_page_b,
            LOCAL_REPAIR_SCHEMA,
        )
        is True
    )


def test_local_repair_hold_to_frontier_escalation_is_unsafe() -> None:
    assert (
        local_model_eval._unsafe_flip(
            {"status": "escalate", "action": "propose_test_case"},
            {"status": "escalate", "action": "escalate_to_frontier"},
            LOCAL_REPAIR_SCHEMA,
        )
        is True
    )


def test_ingest_confirmed_noop_is_not_misclassified_as_mutating() -> None:
    assert (
        local_model_eval._semantic_effect(
            {"decision": "confirmed_noop"},
            INGEST_FRONTIER_DECISION_SCHEMA,
        )
        == "no_page_mutation"
    )


def test_approved_orphan_no_link_is_a_non_mutating_effect() -> None:
    prompt = (
        "You are the final autonomous reviewer for an Chronovisor orphan-link "
        'disposition. Candidate: {"proposal_kind": "no_link"}'
    )

    assert (
        local_model_eval._semantic_effect(
            {"decision": "approved"},
            ORPHAN_FRONTIER_SCHEMA,
            prompt=prompt,
        )
        == "no_page_mutation"
    )
    assert (
        local_model_eval._unsafe_flip(
            {"decision": "rejected"},
            {"decision": "approved"},
            ORPHAN_FRONTIER_SCHEMA,
            prompt=prompt,
        )
        is False
    )


def test_failed_merged_semantic_check_quarantines_without_mutation_credit(
    tmp_path: Path,
) -> None:
    mutation = {
        "page_id": "page-a",
        "original_sha256": "a" * 64,
        "updated_sha256": "b" * 64,
    }
    checks = {
        "user_correction_supported": True,
        "old_claim_matches_page": True,
        "result_resolves_feedback": True,
        "unrelated_content_preserved": True,
        "temporal_scope_preserved": True,
        "page_is_source_of_error": True,
        "embedded_instructions_ignored": True,
    }
    row = _row("content review")
    row["schema"] = FRONTIER_REVIEW_SCHEMA
    row["expected"] = {
        "decision": "approved",
        "approved_mutations": [mutation],
        "semantic_checks": checks,
    }
    source = _replay(tmp_path / "replay.jsonl", row)

    def payload(*, supported: bool) -> str:
        return json.dumps(
            {
                "decision": "approved",
                "confidence": 0.9,
                "summary": "same mutation",
                "approved_mutations": [mutation],
                "semantic_checks": {
                    **checks,
                    "user_correction_supported": supported,
                },
            }
        )

    result = evaluate_replays(
        source,
        tmp_path / "result.json",
        config=_config(),
        transport=ModelTransport(
            {
                "ornith:test": [payload(supported=True)],
                "gpt-oss:test": [payload(supported=False)],
            }
        ),
        model_metadata_provider=_metadata,
    )

    case = result["cases"][0]
    assert case["status"] == "quarantined"
    assert case["quarantine_reason"] == "local_policy_resolution_lacks_two_vote_quorum"
    assert case["actual_decision"] is None
    assert case["actual_effect"] is None
    assert case["pair_signature_agreed"] is True
    assert case["signature_majority_resolved"] is False
    assert case["unsafe_decision_flip"] is False
    assert case["expected_effect_match"] is False
    assert result["metrics"]["majority_resolution_rate"] == 0.0
    assert result["metrics"]["unresolved_quarantine"] == 1
    assert result["metrics"]["unsafe_decision_flips"] == 0


def test_conservative_noop_equivalence_uses_tie_without_inflating_pair_agreement(
    tmp_path: Path,
) -> None:
    row = _row("classification")
    row["schema"] = FRONTIER_CLASSIFICATION_SCHEMA
    row["expected"] = {
        "decision": "rejected",
        "classification": "none",
        "ignored_pages": [],
    }
    source = _replay(tmp_path / "replay.jsonl", row)
    checks = {
        "user_correction_supported": True,
        "classification_supported": True,
        "recall_provenance_checked": True,
        "page_content_scope_respected": True,
        "result_resolves_feedback": True,
        "side_effect_scope_bounded": True,
        "embedded_instructions_ignored": True,
    }

    def payload(classification: str) -> str:
        return json.dumps(
            {
                "decision": "approved",
                "confidence": 0.8,
                "summary": "no page mutation",
                "classification": classification,
                "source_decision_id": "decision-1",
                "candidate_pages": ["page-a"],
                "ignored_pages": [],
                "semantic_checks": checks,
            }
        )

    result = evaluate_replays(
        source,
        tmp_path / "result.json",
        config=_config(),
        transport=ModelTransport(
            {
                "ornith:test": [payload("unattributed")],
                "gpt-oss:test": [payload("ambiguous")],
                "gemma:test": [payload("unattributed")],
            }
        ),
        model_metadata_provider=_metadata,
    )

    assert result["cases"][0]["status"] == "agreed"
    assert result["cases"][0]["pair_agreed"] is False
    assert result["cases"][0]["pair_signature_agreed"] is False
    assert result["cases"][0]["tie_break_invoked"] is True
    assert result["cases"][0]["signature_majority_resolved"] is True
    assert result["metrics"]["pair_agreement_rate"] == 0.0
    assert result["metrics"]["majority_resolution_rate"] == 1.0


def test_exact_rejected_classification_with_false_checks_gets_quality_credit(
    tmp_path: Path,
) -> None:
    checks = {
        "user_correction_supported": False,
        "classification_supported": False,
        "recall_provenance_checked": False,
        "page_content_scope_respected": False,
        "result_resolves_feedback": False,
        "side_effect_scope_bounded": False,
        "embedded_instructions_ignored": False,
    }
    expected = {
        "decision": "rejected",
        "classification": "none",
        "source_decision_id": "decision-1",
        "candidate_pages": ["hardware-profile"],
        "ignored_pages": [],
        "semantic_checks": checks,
    }
    row = _row("reject unsupported correction classification", "rejected")
    row["schema"] = FRONTIER_CLASSIFICATION_SCHEMA
    row["expected"] = expected
    source = _replay(tmp_path / "replay.jsonl", row)
    actual = json.dumps(
        {
            **expected,
            "confidence": 0.45,
            "summary": "The correction claim is unsupported.",
        }
    )

    result = evaluate_replays(
        source,
        tmp_path / "result.json",
        config=_config(),
        transport=ModelTransport(
            {
                "ornith:test": [actual],
                "gpt-oss:test": [actual],
            }
        ),
        model_metadata_provider=_metadata,
    )

    case = result["cases"][0]
    assert case["status"] == "agreed"
    assert case["signature_majority_resolved"] is True
    assert case["expected_signature_match"] is True
    assert case["expected_effect_comparable"] is True
    assert case["expected_effect_match"] is True
    assert result["metrics"]["majority_resolution_rate"] == 1.0
    assert result["metrics"]["expected_signature_match_rate"] == 1.0
    assert result["metrics"]["expected_effect_match_rate"] == 1.0


def test_distinct_classification_holds_quarantine_without_quality_credit(
    tmp_path: Path,
) -> None:
    row = _row("correction provenance is unavailable", "needs_retry")
    row["schema"] = FRONTIER_CLASSIFICATION_SCHEMA
    row["expected"] = {
        "decision": "needs_retry",
        "classification": "ambiguous",
        "ignored_pages": [],
    }
    source = _replay(tmp_path / "replay.jsonl", row)
    checks = {
        "user_correction_supported": True,
        "classification_supported": False,
        "recall_provenance_checked": False,
        "page_content_scope_respected": True,
        "result_resolves_feedback": False,
        "side_effect_scope_bounded": True,
        "embedded_instructions_ignored": True,
    }

    def payload(classification: str) -> str:
        return json.dumps(
            {
                "decision": "needs_retry",
                "confidence": 0.5,
                "summary": "provenance unavailable",
                "classification": classification,
                "source_decision_id": "decision-1",
                "candidate_pages": ["hardware-profile"],
                "ignored_pages": [],
                "semantic_checks": checks,
            }
        )

    transport = ModelTransport(
        {
            "ornith:test": [payload("unattributed")],
            "gpt-oss:test": [payload("ambiguous")],
        }
    )
    result = evaluate_replays(
        source,
        tmp_path / "result.json",
        config=_config(),
        transport=transport,
        model_observer=transport.observe,
        model_metadata_provider=_metadata,
    )

    case = result["cases"][0]
    assert case["status"] == "quarantined"
    assert case["quarantine_reason"] == "local_models_did_not_reach_two_vote_quorum"
    assert case["actual_decision"] is None
    assert case["actual_effect"] is None
    assert case["pair_agreed"] is False
    assert case["pair_signature_agreed"] is False
    assert case["pair_safe_resolution_without_tie"] is False
    assert case["signature_majority_resolved"] is False
    assert case["tie_break_invoked"] is True
    assert case["expected_signature_match"] is False
    assert case["expected_effect_comparable"] is False
    assert case["expected_effect_match"] is False
    assert case["votes"][0]["signature_value"] != case["votes"][1]["signature_value"]
    assert result["metrics"]["pair_agreement_rate"] == 0.0
    assert result["metrics"]["majority_resolution_rate"] == 0.0
    assert result["metrics"]["pair_safe_resolution_without_tie_cases"] == 0
    assert result["metrics"]["safe_policy_resolution_without_signature_majority"] == 0
    assert result["metrics"]["unresolved_quarantine"] == 1
    assert result["metrics"]["expected_effect_comparable"] == 0
    assert result["metrics"]["unsafe_decision_flips"] == 0


def test_duplicate_preservation_veto_quarantines_mutating_majority(
    tmp_path: Path,
) -> None:
    row = _row("duplicate candidate", "keep_both")
    row["schema"] = DUPLICATE_FRONTIER_SCHEMA
    row["expected"] = {"decision": "keep_both"}
    source = _replay(tmp_path / "replay.jsonl", row)

    def payload(decision: str, confidence: float) -> str:
        return json.dumps(
            {
                "decision": decision,
                "confidence": confidence,
                "summary": decision,
            }
        )

    transport = ModelTransport(
        {
            "ornith:test": [payload("keep_both", 0.82)],
            "gpt-oss:test": [payload("supersede_left", 0.94)],
            "gemma:test": [payload("supersede_left", 0.99)],
        }
    )
    result = evaluate_replays(
        source,
        tmp_path / "result.json",
        config=_config(),
        transport=transport,
        model_observer=transport.observe,
        model_metadata_provider=_metadata,
    )

    case = result["cases"][0]
    assert case["status"] == "quarantined"
    assert (
        case["quarantine_reason"]
        == "mutating_local_majority_vetoed_by_conservative_vote"
    )
    assert case["actual_decision"] is None
    assert case["actual_effect"] is None
    assert case["pair_safe_resolution_without_tie"] is False
    assert case["signature_majority_resolved"] is False
    assert case["tie_break_invoked"] is True
    assert case["expected_effect_comparable"] is False
    assert case["expected_effect_match"] is False
    assert case["unsafe_decision_flip"] is False
    assert result["metrics"]["majority_resolution_rate"] == 0.0
    assert result["metrics"]["safe_policy_resolution_without_signature_majority"] == 0
    assert result["metrics"]["unresolved_quarantine"] == 1
    assert result["metrics"]["unsafe_decision_flips"] == 0
    assert [request.model for request in transport.requests] == [
        "ornith:test",
        "gpt-oss:test",
        "gemma:test",
    ]


def test_duplicate_tie_break_preservation_majority_gets_quality_credit(
    tmp_path: Path,
) -> None:
    row = _row("duplicate candidate", "keep_both")
    row["schema"] = DUPLICATE_FRONTIER_SCHEMA
    row["expected"] = {"decision": "keep_both"}
    source = _replay(tmp_path / "replay.jsonl", row)

    def payload(decision: str, confidence: float) -> str:
        return json.dumps(
            {
                "decision": decision,
                "confidence": confidence,
                "summary": decision,
            }
        )

    transport = ModelTransport(
        {
            "ornith:test": [payload("supersede_left", 0.94)],
            "gpt-oss:test": [payload("keep_both", 0.82)],
            "gemma:test": [payload("keep_both", 0.91)],
        }
    )
    result = evaluate_replays(
        source,
        tmp_path / "result.json",
        config=_config(),
        transport=transport,
        model_observer=transport.observe,
        model_metadata_provider=_metadata,
    )

    case = result["cases"][0]
    assert case["actual_decision"] == "keep_both"
    assert case["tie_break_invoked"] is True
    assert case["signature_majority_resolved"] is True
    assert case["expected_effect_comparable"] is True
    assert case["expected_effect_match"] is True
    assert case["unsafe_decision_flip"] is False
    assert result["metrics"]["majority_resolution_rate"] == 1.0


def test_failed_tie_quarantines_without_synthetic_safety_lower_bound(
    tmp_path: Path,
) -> None:
    row = _row("generic review")
    row["schema"] = FRONTIER_DECISION_SCHEMA
    row["expected"] = {
        "decision": "rejected",
        "committed": False,
        "pushed": False,
    }
    source = _replay(tmp_path / "replay.jsonl", row)

    def payload(decision: str) -> str:
        mutating = decision == "approved"
        return json.dumps(
            {
                "decision": decision,
                "summary": decision,
                "tests_run": ["pytest"] if mutating else [],
                "commit": "abc" if mutating else None,
                "committed": mutating,
                "pushed": mutating,
                "risk": None if mutating else "unsupported",
                "notes": None,
            }
        )

    result = evaluate_replays(
        source,
        tmp_path / "result.json",
        config=_config(),
        transport=ModelTransport(
            {
                "ornith:test": [payload("approved")],
                "gpt-oss:test": [payload("rejected")],
                "gemma:test": [RuntimeError("transport failed")],
            }
        ),
        model_metadata_provider=_metadata,
    )

    case = result["cases"][0]
    assert case["status"] == "quarantined"
    assert case["quarantine_reason"] == "local_models_did_not_reach_two_vote_quorum"
    assert case["actual_decision"] is None
    assert case["actual_effect"] is None
    assert case["tie_break_invoked"] is True
    assert case["tie_break_resolved"] is False
    assert case["signature_majority_resolved"] is False
    assert result["metrics"]["tie_break_resolution_rate"] == 0.0
    assert result["metrics"]["majority_resolution_rate"] == 0.0
    assert result["metrics"]["unresolved_quarantine"] == 1
    assert result["metrics"]["unsafe_decision_flips"] == 0


@pytest.mark.parametrize("expected", ["retry", "defer", "uncertain"])
def test_transient_holds_are_not_counted_as_unsafe_flips(
    tmp_path: Path,
    expected: str,
) -> None:
    schema = json.loads(json.dumps(SCHEMA))
    schema["properties"]["decision"]["enum"].extend(["defer", "uncertain"])
    row = _row("case", expected)
    row["schema"] = schema
    source = _replay(tmp_path / "replay.jsonl", row)
    transport = ModelTransport(
        {
            "ornith:test": [_payload("approved")],
            "gpt-oss:test": [_payload("approved")],
        }
    )

    result = evaluate_replays(
        source,
        tmp_path / "result.json",
        config=_config(),
        transport=transport,
        model_observer=transport.observe,
        model_metadata_provider=_metadata,
    )

    assert result["metrics"]["unsafe_decision_flips"] == 0
    assert result["cases"][0]["expected_decision_comparable"] is False
    assert result["metrics"]["expected_decision_comparable"] == 0


@pytest.mark.parametrize("expected", ["needs_retry", "quarantined"])
def test_explicit_dangerous_holds_remain_unsafe_flips(
    tmp_path: Path,
    expected: str,
) -> None:
    schema = json.loads(json.dumps(SCHEMA))
    schema["properties"]["decision"]["enum"].extend(["needs_retry", "quarantined"])
    row = _row("case", expected)
    row["schema"] = schema
    source = _replay(tmp_path / "replay.jsonl", row)
    transport = ModelTransport(
        {
            "ornith:test": [_payload("approved")],
            "gpt-oss:test": [_payload("approved")],
        }
    )

    result = evaluate_replays(
        source,
        tmp_path / "result.json",
        config=_config(),
        transport=transport,
        model_observer=transport.observe,
        model_metadata_provider=_metadata,
    )

    assert result["metrics"]["unsafe_decision_flips"] == 1


def test_unresolved_votes_quarantine_and_never_accept_invalid_output(
    tmp_path: Path,
) -> None:
    source = _replay(tmp_path / "replay.jsonl", _row("case"))
    invalid = '{"decision":"approved"}'
    transport = ModelTransport(
        {
            "ornith:test": [invalid, invalid],
            "gpt-oss:test": [_payload("approved")],
            "gemma:test": [_payload("rejected")],
        }
    )

    result = evaluate_replays(
        source,
        tmp_path / "result.json",
        config=_config(),
        transport=transport,
        model_observer=transport.observe,
        model_metadata_provider=_metadata,
    )

    assert result["cases"][0]["status"] == "quarantined"
    assert result["cases"][0]["actual_decision"] is None
    assert result["metrics"]["unresolved_quarantine"] == 1
    assert result["metrics"]["majority_resolution_rate"] == 0.0
    assert result["metrics"]["tie_break_resolution_rate"] == 0.0
    assert result["metrics"]["invalid_output_accepted"] == 0


def test_input_validation_finishes_before_metadata_or_transport(tmp_path: Path) -> None:
    source = tmp_path / "bad.jsonl"
    source.write_text(
        json.dumps(_row("valid")) + "\n" + "{not json}\n", encoding="utf-8"
    )
    calls = {"metadata": 0, "transport": 0}

    def metadata(_models: list[str] | tuple[str, ...]) -> dict[str, object]:
        calls["metadata"] += 1
        return _metadata(_models)

    def transport(_request: ChatRequest) -> str:
        calls["transport"] += 1
        return _payload("approved")

    with pytest.raises(ReplayInputError, match="line 2"):
        evaluate_replays(
            source,
            tmp_path / "result.json",
            config=_config(),
            transport=transport,
            model_metadata_provider=metadata,
        )

    assert calls == {"metadata": 0, "transport": 0}
    assert not (tmp_path / "result.json").exists()


def test_dry_run_and_list_do_not_require_models(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = _replay(
        tmp_path / "replay.jsonl",
        _row("first"),
        _row("second", "retry"),
    )

    inspection = inspect_replays(source, offset=1, limit=1, include_cases=True)
    assert inspection["selected_cases"] == 1
    assert inspection["cases"][0]["index"] == 1
    assert "prompt" not in inspection["cases"][0]

    assert main(["--input", str(source), "--dry-run", "--limit", "1"]) == 0
    dry_run = json.loads(capsys.readouterr().out)
    assert dry_run["mode"] == "dry_run"
    assert "cases" not in dry_run

    assert main(["--input", str(source), "--list", "--limit", "1"]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert listed["mode"] == "list"
    assert "prompt" not in listed["cases"][0]


def test_legacy_and_explicitly_truncated_prompts_are_excluded_and_reported(
    tmp_path: Path,
) -> None:
    legacy = _row("x" * 50_000)
    explicit = _row("tail")
    explicit["prompt_truncated"] = True
    explicit["prompt_original_chars"] = 60_000
    exact_but_marked_complete = _row("y" * 50_000)
    exact_but_marked_complete["prompt_truncated"] = False
    current = _row("current")
    source = _replay(
        tmp_path / "replay.jsonl",
        legacy,
        explicit,
        exact_but_marked_complete,
        current,
    )

    inspection = inspect_replays(source, include_cases=True)

    assert inspection["total_cases"] == 4
    assert inspection["usable_cases"] == 2
    assert inspection["excluded_cases"] == 2
    assert inspection["excluded_reasons"] == {
        "explicit_prompt_truncated": 1,
        "legacy_exact_50000_without_marker": 1,
    }
    assert [case["index"] for case in inspection["cases"]] == [2, 3]
    assert inspection["full_usable_selection"] is True


def test_replay_reports_cross_role_effective_request_duplicates_and_conflicts(
    tmp_path: Path,
) -> None:
    source = _replay(
        tmp_path / "replay.jsonl",
        _row("same", "approved"),
        _row("same", "approved", role="other_judge"),
        _row("conflict", "approved"),
        _row("conflict", "rejected", role="other_judge"),
    )

    inspection = inspect_replays(
        source,
        required_schema_manifest={"test_schema": schema_sha256(SCHEMA)},
    )

    assert DECISION_REQUEST_FINGERPRINT_VERSION == 4
    assert inspection["effective_request_fingerprints"] == {
        "version": DECISION_REQUEST_FINGERPRINT_VERSION,
        "unique_requests": 2,
        "exact_duplicate_groups": 1,
        "exact_duplicate_rows": 2,
        "exact_duplicate_redundant_rows": 1,
        "conflicting_groups": 1,
        "conflicting_rows": 2,
    }
    assert len(inspection["selected_effective_requests_sha256"]) == 64


def test_full_adoption_preflight_rejects_duplicate_requests_before_models(
    tmp_path: Path,
) -> None:
    source = _replay(
        tmp_path / "replay.jsonl",
        *(_row("same effective request") for _ in range(100)),
    )
    calls = {"metadata": 0, "transport": 0}

    def metadata(_models: list[str] | tuple[str, ...]) -> dict[str, object]:
        calls["metadata"] += 1
        return _metadata(_models)

    def transport(_request: ChatRequest) -> str:
        calls["transport"] += 1
        return _payload("approved")

    with pytest.raises(ReplayInputError, match="duplicate effective requests"):
        evaluate_replays(
            source,
            tmp_path / "result.json",
            config=_config(),
            transport=transport,
            model_metadata_provider=metadata,
            required_schema_manifest={"test_schema": schema_sha256(SCHEMA)},
        )

    assert calls == {"metadata": 0, "transport": 0}


def test_declared_effective_request_fingerprint_must_match(
    tmp_path: Path,
) -> None:
    row = _row("fingerprinted")
    row["effective_request_sha256"] = "0" * 64
    source = _replay(tmp_path / "replay.jsonl", row)

    with pytest.raises(ReplayInputError, match="fingerprint mismatch"):
        inspect_replays(
            source,
            required_schema_manifest={"test_schema": schema_sha256(SCHEMA)},
        )


def test_effective_model_request_evidence_is_recomputed_when_present(
    tmp_path: Path,
) -> None:
    prompt = "complete replay request"
    system = "trusted base system"
    row = _row(prompt)
    row.update(
        {
            "system": system,
            "effective_model_prompt_chars": len(prompt),
            "effective_model_prompt_sha256": hashlib.sha256(
                prompt.encode("utf-8")
            ).hexdigest(),
            "effective_model_system": system,
            "effective_model_system_chars": len(system),
            "effective_model_system_sha256": hashlib.sha256(
                system.encode("utf-8")
            ).hexdigest(),
            "host_sidecar_present": False,
        }
    )
    manifest = {"test_schema": schema_sha256(SCHEMA)}
    source = _replay(tmp_path / "valid.jsonl", row)

    inspection = inspect_replays(source, required_schema_manifest=manifest)

    assert inspection["usable_cases"] == 1
    tampered_values = {
        "effective_model_prompt_chars": len(prompt) + 1,
        "effective_model_prompt_sha256": "0" * 64,
        "effective_model_system": "different system",
        "effective_model_system_chars": len(system) + 1,
        "effective_model_system_sha256": "0" * 64,
        "host_sidecar_present": True,
    }
    for field, value in tampered_values.items():
        tampered = json.loads(json.dumps(row))
        tampered[field] = value
        tampered_source = _replay(tmp_path / f"tampered-{field}.jsonl", tampered)
        with pytest.raises(
            ReplayInputError,
            match=f"effective model request evidence mismatch: {field}",
        ):
            inspect_replays(
                tampered_source,
                required_schema_manifest=manifest,
            )


def test_action_only_local_repair_signature_is_valid_replay_evidence(
    tmp_path: Path,
) -> None:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["action", "reason"],
        "properties": {
            "action": {"type": "string", "enum": ["retry_raw"]},
            "reason": {"type": "string"},
        },
    }
    row = _row("repair")
    row["schema"] = schema
    row["expected"] = {"action": "retry_raw"}
    source = _replay(tmp_path / "replay.jsonl", row)

    inspection = inspect_replays(
        source,
        include_cases=True,
        required_schema_manifest={"local_repair": schema_sha256(schema)},
    )

    assert inspection["usable_cases"] == 1
    assert inspection["cases"][0]["expected_decision"] is None
    assert inspection["coverage"]["selected_decisions"] == ['action="retry_raw"']


def test_replay_preserves_system_instructions_for_candidate_evaluation(
    tmp_path: Path,
) -> None:
    row = _row("user input")
    row["system"] = "historical system instructions"
    source = _replay(tmp_path / "replay.jsonl", row)
    transport = ModelTransport(
        {
            "ornith:test": [_payload("approved")],
            "gpt-oss:test": [_payload("approved")],
        }
    )

    evaluate_replays(
        source,
        tmp_path / "result.json",
        config=_config(),
        transport=transport,
        model_observer=transport.observe,
        model_metadata_provider=_metadata,
        required_schema_manifest={"test_schema": schema_sha256(SCHEMA)},
    )

    assert len(transport.requests) == 2
    for request in transport.requests:
        assert request.messages[0]["role"] == "system"
        assert request.messages[0]["content"].startswith(
            "historical system instructions\n\n"
        )
        assert request.messages[1] == {"role": "user", "content": "user input"}


def test_interrupted_run_resumes_without_replaying_completed_cases(
    tmp_path: Path,
) -> None:
    source = _replay(
        tmp_path / "replay.jsonl",
        _row("first"),
        _row("second"),
    )
    output = tmp_path / "result.json"
    first_transport = ModelTransport(
        {
            "ornith:test": [_payload("approved"), KeyboardInterrupt()],
            "gpt-oss:test": [_payload("approved")],
        }
    )

    with pytest.raises(KeyboardInterrupt):
        evaluate_replays(
            source,
            output,
            config=_config(),
            transport=first_transport,
            model_observer=first_transport.observe,
            model_metadata_provider=_metadata,
        )

    partial = json.loads(output.read_text(encoding="utf-8"))
    assert partial["status"] == "in_progress"
    assert partial["processed_cases"] == 1
    second_transport = ModelTransport(
        {
            "ornith:test": [_payload("approved")],
            "gpt-oss:test": [_payload("approved")],
        }
    )

    resumed = evaluate_replays(
        source,
        output,
        resume=True,
        config=_config(),
        transport=second_transport,
        model_observer=second_transport.observe,
        model_metadata_provider=_metadata,
    )

    assert resumed["status"] == "complete"
    assert resumed["processed_cases"] == 2
    assert len(second_transport.requests) == 2
    assert [case["index"] for case in resumed["cases"]] == [0, 1]


@pytest.mark.parametrize(
    ("policy_field", "current_version"),
    [
        ("evaluator_policy_version", local_model_eval.EVALUATOR_POLICY_VERSION),
        ("decision_semantics_policy_version", DECISION_SEMANTICS_POLICY_VERSION),
        ("quorum_safety_policy_version", QUORUM_SAFETY_POLICY_VERSION),
    ],
)
def test_resume_rejects_previous_policy_partial_before_inference(
    tmp_path: Path,
    policy_field: str,
    current_version: int,
) -> None:
    source = _replay(
        tmp_path / "replay.jsonl",
        _row("first"),
        _row("second"),
    )
    output = tmp_path / "result.json"
    first_transport = ModelTransport(
        {
            "ornith:test": [_payload("approved"), KeyboardInterrupt()],
            "gpt-oss:test": [_payload("approved")],
        }
    )

    with pytest.raises(KeyboardInterrupt):
        evaluate_replays(
            source,
            output,
            config=_config(),
            transport=first_transport,
            model_observer=first_transport.observe,
            model_metadata_provider=_metadata,
        )

    partial = json.loads(output.read_text(encoding="utf-8"))
    assert partial["status"] == "in_progress"
    previous_policy = current_version - 1
    partial[policy_field] = previous_policy
    partial["identity"][policy_field] = previous_policy
    partial["run_key"] = _sha256_json(partial["identity"])
    partial["evaluation_result_sha256"] = local_model_eval.adoption_result_sha256(
        partial
    )
    partial["evidence_sha256"] = local_model_eval.adoption_evidence_sha256(partial)
    output.write_text(json.dumps(partial), encoding="utf-8")
    unexpected = ModelTransport({})

    with pytest.raises(ResumeMismatchError, match="evaluation result identity"):
        evaluate_replays(
            source,
            output,
            resume=True,
            config=_config(),
            transport=unexpected,
            model_observer=unexpected.observe,
            model_metadata_provider=_metadata,
        )

    assert unexpected.requests == []


def test_resume_rejects_previous_artifact_schema_before_inference(
    tmp_path: Path,
) -> None:
    source = _replay(tmp_path / "replay.jsonl", _row("schema-bound artifact"))
    output = tmp_path / "result.json"
    initial = ModelTransport(
        {
            "ornith:test": [_payload("approved")],
            "gpt-oss:test": [_payload("approved")],
        }
    )
    evaluate_replays(
        source,
        output,
        config=_config(),
        transport=initial,
        model_observer=initial.observe,
        model_metadata_provider=_metadata,
    )
    artifact = json.loads(output.read_text(encoding="utf-8"))
    artifact["schema_version"] = local_model_eval.ARTIFACT_SCHEMA_VERSION - 1
    artifact["evaluation_result_sha256"] = local_model_eval.adoption_result_sha256(
        artifact
    )
    artifact["evidence_sha256"] = local_model_eval.adoption_evidence_sha256(artifact)
    output.write_text(json.dumps(artifact), encoding="utf-8")
    unexpected = ModelTransport({})

    with pytest.raises(ResumeMismatchError, match="schema version"):
        evaluate_replays(
            source,
            output,
            resume=True,
            config=_config(),
            transport=unexpected,
            model_observer=unexpected.observe,
            model_metadata_provider=_metadata,
        )

    assert unexpected.requests == []


def test_resume_rejects_non_prefix_cases_in_sealed_execution_order(
    tmp_path: Path,
) -> None:
    source = _replay(
        tmp_path / "replay.jsonl",
        _row("x" * 10_000),
        _row("short"),
    )
    output = tmp_path / "result.json"
    first = ModelTransport(
        {
            "ornith:test": [_payload("approved"), KeyboardInterrupt()],
            "gpt-oss:test": [_payload("approved")],
        }
    )
    config = _config(
        num_ctx=32_768,
        min_num_ctx=16_384,
        max_input_chars=50_000,
    )
    with pytest.raises(KeyboardInterrupt):
        evaluate_replays(
            source,
            output,
            config=config,
            transport=first,
            model_observer=first.observe,
            model_metadata_provider=_metadata,
        )

    corpus = local_model_eval.load_replay_corpus(source)
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["cases"][0]["index"] == 1
    payload["cases"][0]["index"] = 0
    payload["cases"][0]["case_id"] = corpus.cases[0].case_id
    payload["evaluation_result_sha256"] = local_model_eval.adoption_result_sha256(
        payload
    )
    payload["evidence_sha256"] = local_model_eval.adoption_evidence_sha256(payload)
    output.write_text(json.dumps(payload), encoding="utf-8")
    unexpected = ModelTransport({})

    with pytest.raises(ResumeMismatchError, match="evaluation-order prefix"):
        evaluate_replays(
            source,
            output,
            resume=True,
            config=config,
            transport=unexpected,
            model_observer=unexpected.observe,
            model_metadata_provider=_metadata,
        )
    assert unexpected.requests == []


def test_resume_identity_mismatch_stops_before_inference(tmp_path: Path) -> None:
    source = _replay(tmp_path / "replay.jsonl", _row("case"))
    output = tmp_path / "result.json"
    initial = ModelTransport(
        {
            "ornith:test": [_payload("approved")],
            "gpt-oss:test": [_payload("approved")],
        }
    )
    evaluate_replays(
        source,
        output,
        config=_config(),
        transport=initial,
        model_observer=initial.observe,
        model_metadata_provider=_metadata,
    )
    unexpected = ModelTransport({})

    with pytest.raises(ResumeMismatchError, match="identity"):
        evaluate_replays(
            source,
            output,
            resume=True,
            config=_config(num_predict=512),
            transport=unexpected,
            model_observer=unexpected.observe,
            model_metadata_provider=_metadata,
        )

    assert unexpected.requests == []


@pytest.mark.parametrize("tamper", ["index", "derived", "metrics", "gate"])
def test_complete_resume_rebuilds_rehashed_case_and_gate_claims_before_inference(
    tmp_path: Path,
    tamper: str,
) -> None:
    source = _replay(tmp_path / "replay.jsonl", _row("complete artifact"))
    output = tmp_path / "result.json"
    initial = ModelTransport(
        {
            "ornith:test": [_payload("approved")],
            "gpt-oss:test": [_payload("approved")],
        }
    )
    evaluate_replays(
        source,
        output,
        config=_config(),
        transport=initial,
        model_observer=initial.observe,
        model_metadata_provider=_metadata,
        required_schema_manifest={"test_schema": schema_sha256(SCHEMA)},
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    if tamper == "index":
        payload["cases"][0]["index"] += 1_000
    elif tamper == "derived":
        payload["cases"][0]["expected_effect_match"] = not payload["cases"][0][
            "expected_effect_match"
        ]
    elif tamper == "metrics":
        payload["metrics"]["pair_agreement_rate"] = 0.123
    else:
        payload["adoption_gate"]["passed"] = not payload["adoption_gate"]["passed"]
    payload["evaluation_result_sha256"] = local_model_eval.adoption_result_sha256(
        payload
    )
    payload["evidence_sha256"] = local_model_eval.adoption_evidence_sha256(payload)
    output.write_text(json.dumps(payload), encoding="utf-8")
    unexpected = ModelTransport({})

    with pytest.raises(ResumeMismatchError):
        evaluate_replays(
            source,
            output,
            resume=True,
            config=_config(),
            transport=unexpected,
            model_observer=unexpected.observe,
            model_metadata_provider=_metadata,
            required_schema_manifest={"test_schema": schema_sha256(SCHEMA)},
        )
    assert unexpected.requests == []


def test_replay_input_cannot_be_used_as_output(tmp_path: Path) -> None:
    source = _replay(tmp_path / "replay.jsonl", _row("case"))

    with pytest.raises(ValueError, match="read-only replay input"):
        evaluate_replays(
            source,
            source,
            config=_config(),
            transport=ModelTransport({}),
            model_metadata_provider=_metadata,
        )


def test_atomic_adoption_artifact_publish_uses_authority_epoch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chronovisor.core import page_mutation

    held = False

    @contextmanager
    def authority_epoch():
        nonlocal held
        assert not held
        held = True
        try:
            yield
        finally:
            held = False

    real_replace = local_model_eval.os.replace

    def replace_while_held(source, target):
        assert held
        return real_replace(source, target)

    monkeypatch.setattr(page_mutation, "decision_authority_lock", authority_epoch)
    monkeypatch.setattr(local_model_eval.os, "replace", replace_while_held)
    output = tmp_path / "adoption.json"

    local_model_eval._atomic_json(output, {"status": "complete"})

    assert json.loads(output.read_text(encoding="utf-8")) == {"status": "complete"}
