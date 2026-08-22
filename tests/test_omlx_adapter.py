"""oMLX adapter: parameter mapping, response normalization, and config wiring."""

from __future__ import annotations

import json
import tomllib
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
import pytest

from chronovisor.core import omlx_adapter
from chronovisor.core.llm_config import (
    LLMConfigError,
    build_llm_runtime,
    parse_llm_config,
)
from chronovisor.core.llm_runtime import (
    EmbeddingRequest,
    GenerationRequest,
    MessageGenerationRequest,
    RouteLocation,
    SafeBackendError,
    SourceDataClass,
    SourceDataClassification,
    SourceSensitivity,
)
from chronovisor.core.omlx_adapter import OMLXAdapter

NORMAL_PAGE = SourceDataClassification(SourceDataClass.PAGE, SourceSensitivity.NORMAL)


@pytest.fixture(autouse=True)
def _mock_models_do_not_use_dflash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings_path = tmp_path / "model_settings.json"
    settings_path.write_text(
        json.dumps(
            {
                "models": {
                    "m": {"dflash_enabled": False},
                    "Ornith-1.5-9B-MLX-4bit": {"dflash_enabled": False},
                }
            }
        )
    )
    monkeypatch.setattr(omlx_adapter, "OMLX_MODEL_SETTINGS_PATH", settings_path)


def _chat_json(content: str = "ok", *, reasoning: str | None = None) -> dict[str, object]:
    message: dict[str, object] = {"role": "assistant", "content": content}
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": "m",
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": 12,
            "completion_tokens": 4,
            "total_tokens": 16,
            "total_time": 1.5,
            "model_load_duration": 0.2,
        },
    }


def _json(request: httpx.Request) -> dict[str, Any]:
    return json.loads(request.content)


def _minimal_message_request(*, think: bool | str = False) -> MessageGenerationRequest:
    return MessageGenerationRequest(
        messages=({"role": "user", "content": "x"},),
        format=None,
        source=NORMAL_PAGE,
        num_ctx=1024,
        max_output_tokens=8,
        keep_alive="0",
        timeout_ms=1000,
        max_output_chars=200,
        temperature=0,
        seed=0,
        think=think,
    )


def test_generate_maps_parameters_and_drops_unsupported() -> None:
    captured: list[dict[str, Any]] = []

    def handle(request: httpx.Request) -> httpx.Response:
        captured.append({"json": _json(request), "url": str(request.url)})
        return httpx.Response(
            200, json=_chat_json("42"), request=request,
            headers={"Content-Type": "application/json"},
        )

    adapter = OMLXAdapter(transport=httpx.MockTransport(handle))
    request = MessageGenerationRequest(
        messages=({"role": "user", "content": "2+2=?"},),
        format=None,
        source=NORMAL_PAGE,
        num_ctx=4096,
        max_output_tokens=16,
        keep_alive="24h",
        timeout_ms=30000,
        max_output_chars=100,
        temperature=0.3,
        seed=7,
        think=False,
    )
    adapter.generate(request, model="Ornith-1.5-9B-MLX-4bit")
    body = captured[0]["json"]
    assert body["model"] == "Ornith-1.5-9B-MLX-4bit"
    assert body["max_tokens"] == 16
    assert body["temperature"] == 0.3
    assert body["seed"] == 7
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert "num_ctx" not in body
    assert "keep_alive" not in body
    assert urlparse(str(captured[0]["url"])).path.startswith("/v1/chat/completions")


def test_adapter_uses_configured_local_server() -> None:
    urls: list[str] = []

    def handle(request: httpx.Request) -> httpx.Response:
        urls.append(str(request.url))
        return httpx.Response(200, json=_chat_json(), request=request)

    OMLXAdapter(
        base_url="http://127.0.0.1:8001/v1",
        transport=httpx.MockTransport(handle),
    ).generate(_minimal_message_request(), model="m")

    assert urlparse(urls[0]).port == 8001


def test_generate_maps_reasoning_level_to_omlx_controls() -> None:
    captured: list[dict[str, Any]] = []

    def handle(request: httpx.Request) -> httpx.Response:
        captured.append(_json(request))
        return httpx.Response(200, json=_chat_json(), request=request)

    OMLXAdapter(transport=httpx.MockTransport(handle)).generate(
        _minimal_message_request(think="low"), model="m"
    )

    assert captured[0]["reasoning_effort"] == "low"
    assert captured[0]["chat_template_kwargs"] == {"enable_thinking": True}


def test_generate_sends_x_api_key_header() -> None:
    captured: list[dict[str, Any]] = []

    def handle(request: httpx.Request) -> httpx.Response:
        captured.append({"headers": dict(request.headers)})
        return httpx.Response(200, json=_chat_json(), request=request)

    OMLXAdapter(transport=httpx.MockTransport(handle)).generate(
        _minimal_message_request(), model="m"
    )
    assert captured[0]["headers"]["x-api-key"]


def test_generate_maps_json_mapping_to_json_schema() -> None:
    captured: list[dict[str, Any]] = []

    def handle(request: httpx.Request) -> httpx.Response:
        captured.append({"json": _json(request)})
        return httpx.Response(200, json=_chat_json('{"ok":true}'), request=request)

    request = MessageGenerationRequest(
        messages=({"role": "user", "content": "j"},),
        format={"type": "object", "properties": {"ok": {"type": "boolean"}}},
        source=NORMAL_PAGE,
        num_ctx=1024,
        max_output_tokens=8,
        keep_alive="0",
        timeout_ms=1000,
        max_output_chars=200,
        temperature=0,
        seed=0,
        think=False,
    )
    OMLXAdapter(transport=httpx.MockTransport(handle)).generate(request, model="m")
    response_format = captured[0]["json"]["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["name"] == "response"


def test_generate_maps_json_string_to_json_object() -> None:
    captured: list[dict[str, Any]] = []

    def handle(request: httpx.Request) -> httpx.Response:
        captured.append({"json": _json(request)})
        return httpx.Response(200, json=_chat_json('{"ok":true}'), request=request)

    request = GenerationRequest(
        prompt="json here",
        format="json",
        source=NORMAL_PAGE,
        max_output_tokens=8,
        temperature=0,
    )
    OMLXAdapter(transport=httpx.MockTransport(handle)).generate(request, model="m")
    assert captured[0]["json"]["response_format"] == {"type": "json_object"}
    assert captured[0]["json"]["chat_template_kwargs"] == {
        "enable_thinking": False
    }


def test_generate_normalizes_response() -> None:
    adapter = OMLXAdapter(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json=_chat_json("42"), request=r))
    )
    result = adapter.generate(_minimal_message_request(), model="m")
    assert result.provider == "omlx"
    assert result.content == "42"
    assert result.finish_reason == "stop"
    assert (result.usage.input_tokens, result.usage.output_tokens) == (12, 4)
    assert result.metadata["total_time"] == 1.5


def test_generate_streams_redacted_progress_and_reassembles_response() -> None:
    captured: list[dict[str, Any]] = []
    updates: list[dict[str, Any]] = []
    chunks = [
        {"model": "m", "choices": [{"delta": {"role": "assistant"}}]},
        {"model": "m", "choices": [{"delta": {"reasoning_content": "plan"}}]},
        {"model": "m", "choices": [{"delta": {"content": '{"ok":true}'}}]},
        {"model": "m", "choices": [{"delta": {}, "finish_reason": "stop"}]},
        {
            "model": "m",
            "choices": [],
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 4,
                "total_tokens": 16,
                "generation_duration": 0.5,
                "generation_tokens_per_second": 8.0,
            },
        },
    ]
    body = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
    body += "data: [DONE]\n\n"

    def handle(request: httpx.Request) -> httpx.Response:
        captured.append(_json(request))
        return httpx.Response(
            200,
            content=body,
            request=request,
            headers={"Content-Type": "text/event-stream"},
        )

    request = replace(
        _minimal_message_request(),
        progress_callback=updates.append,
    )
    result = OMLXAdapter(transport=httpx.MockTransport(handle)).generate(
        request, model="m"
    )

    assert captured[0]["stream"] is True
    assert captured[0]["stream_options"] == {"include_usage": True}
    assert result.content == '{"ok":true}'
    assert result.completed is True
    assert result.metadata["streamed"] is True
    assert result.metadata["reasoning_content"] == "plan"
    assert (result.usage.input_tokens, result.usage.output_tokens) == (12, 4)
    assert updates[-1] == {
        "output_tokens": 4,
        "generation_seconds": 0.5,
        "tokens_per_second": 8.0,
        "token_count_exact": True,
        "max_output_tokens": 8,
    }
    assert not any(
        "content" in update or "reasoning_content" in update for update in updates
    )


def test_generate_marks_stream_without_done_as_incomplete() -> None:
    body = 'data: {"model":"m","choices":[{"delta":{"content":"x"}}]}\n\n'
    adapter = OMLXAdapter(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, content=body, request=request)
        )
    )

    result = adapter.generate(
        replace(_minimal_message_request(), progress_callback=lambda _event: None),
        model="m",
    )

    assert result.content == "x"
    assert result.completed is False
    assert result.metadata["streamed"] is True


def test_generate_falls_back_to_reasoning_content() -> None:
    adapter = OMLXAdapter(
        transport=httpx.MockTransport(
            lambda r: httpx.Response(200, json=_chat_json("", reasoning="think"), request=r)
        )
    )
    result = adapter.generate(_minimal_message_request(), model="m")
    assert result.content == "think"
    assert result.metadata["reasoning_content"] == "think"


def test_generate_maps_http_errors_into_safe_categories() -> None:
    def adapter_for(status: int) -> OMLXAdapter:
        return OMLXAdapter(
            transport=httpx.MockTransport(
                lambda r: httpx.Response(status, json={"error": {}}, request=r)
            )
        )

    with pytest.raises(SafeBackendError) as exc5:
        adapter_for(503).generate(_minimal_message_request(), model="m")
    assert exc5.value.safe_category == "http_5xx"
    assert exc5.value.transient is True

    with pytest.raises(SafeBackendError) as exc429:
        adapter_for(429).generate(_minimal_message_request(), model="m")
    assert exc429.value.safe_category == "http_429"
    assert exc429.value.transient is True

    with pytest.raises(SafeBackendError) as exc400:
        adapter_for(400).generate(_minimal_message_request(), model="m")
    assert exc400.value.safe_category == "invalid_request"
    assert exc400.value.transient is False


def test_generate_maps_transport_to_transient_error() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    adapter = OMLXAdapter(transport=httpx.MockTransport(handle))
    with pytest.raises(SafeBackendError) as exc:
        adapter.generate(_minimal_message_request(), model="m")
    assert exc.value.safe_category == "transport_error"
    assert exc.value.transient is True


def test_embed_returns_index_ordered_vectors() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        body = _json(request)
        assert body["model"] == "bge-m3-mlx-fp16"
        assert body["input"] == ["a", "bb"]
        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": [
                    {"index": 1, "embedding": [0.0, 1.0]},
                    {"index": 0, "embedding": [1.0, 0.0]},
                ],
            },
            request=request,
        )

    adapter = OMLXAdapter(transport=httpx.MockTransport(handle))
    request = EmbeddingRequest(texts=("a", "bb"), source=NORMAL_PAGE, timeout_ms=1000)
    result = adapter.embed(request, model="bge-m3-mlx-fp16")
    assert result.provider == "omlx"
    assert result.vectors == ((1.0, 0.0), (0.0, 1.0))


def test_generate_rejects_unknown_string_format() -> None:
    adapter = OMLXAdapter(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json=_chat_json(), request=r))
    )
    request = GenerationRequest(
        prompt="x", format="notjson", source=NORMAL_PAGE, max_output_tokens=8, temperature=0
    )
    with pytest.raises(SafeBackendError):
        adapter.generate(request, model="m")


def test_adapter_exposes_local_location_and_lease() -> None:
    adapter = OMLXAdapter()
    assert adapter.location is RouteLocation.LOCAL
    with adapter.resource_lease(exclusive=False):
        pass


def test_dflash_generation_uses_exclusive_lease_only_when_enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings_path = tmp_path / "model_settings.json"
    settings_path.write_text(
        json.dumps(
            {
                "models": {
                    "teacher": {"dflash_enabled": True},
                    "gate": {"dflash_enabled": False},
                }
            }
        )
    )
    leases: list[tuple[bool, int | None]] = []
    monkeypatch.setattr(omlx_adapter, "OMLX_MODEL_SETTINGS_PATH", settings_path)
    monkeypatch.setattr(
        omlx_adapter,
        "model_resource_lease",
        lambda *, exclusive, timeout_ms=None: (
            leases.append((exclusive, timeout_ms)) or nullcontext()
        ),
    )
    adapter = OMLXAdapter(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json=_chat_json(), request=request)
        )
    )

    adapter.generate(_minimal_message_request(), model="teacher")
    adapter.generate(_minimal_message_request(), model="gate")

    assert leases == [(True, 1000)]


def test_config_parses_omlx_provider_and_builds_runtime() -> None:
    payload = tomllib.loads(
        """
[llm.providers.omlx]
kind = "omlx"

[llm.providers.ollama]
kind = "ollama"

[llm.roles."recall.gate"]
capability = "generation"
provider = "omlx"
model = "Ornith-1.5-9B-MLX-4bit"

[llm.roles."classification.embedding"]
capability = "embedding"
provider = "omlx"
model = "bge-m3-mlx-fp16"
"""
    )
    config = parse_llm_config(payload)
    assert config.providers["omlx"].kind == "omlx"
    assert config.providers["omlx"].capabilities_for("x").generation is True
    assert config.providers["omlx"].capabilities_for("x").embedding is True
    assert config.providers["omlx"].capabilities_for("x").streaming is True
    runtime = build_llm_runtime(config)
    assert runtime.resolve_generation("recall.gate").provider == "omlx"
    assert runtime.resolve_embedding("classification.embedding").provider == "omlx"


def test_config_routes_distinct_local_omlx_servers() -> None:
    payload = tomllib.loads(
        """
[llm.providers.omlx_qwen]
kind = "omlx"
endpoint = "http://127.0.0.1:8000/v1"

[llm.providers.omlx_gemma]
kind = "omlx"
endpoint = "http://127.0.0.1:8001/v1"

[llm.roles."classification.primary"]
capability = "generation"
provider = "omlx_qwen"
model = "Qwen3.8-27B-4bit"

[llm.roles."classification.challenger"]
capability = "generation"
provider = "omlx_gemma"
model = "gemma-4-26b-a4b-it-4bit"
"""
    )

    runtime = build_llm_runtime(parse_llm_config(payload))
    primary = runtime.resolve_generation("classification.primary")
    challenger = runtime.resolve_generation("classification.challenger")

    assert primary.endpoint_sha256 != challenger.endpoint_sha256
    assert primary.protocol == challenger.protocol == "omlx-native"


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://127.0.0.1:8000/v1",
        "http://192.168.100.3:8000/v1",
        "http://127.0.0.1:8000/admin",
    ],
)
def test_config_rejects_non_loopback_omlx_endpoint(endpoint: str) -> None:
    payload = tomllib.loads(
        f'''\n[llm.providers.omlx]\nkind = "omlx"\nendpoint = "{endpoint}"\n'''
    )

    with pytest.raises(LLMConfigError):
        parse_llm_config(payload)


def test_config_rejects_unknown_keys_on_omlx_provider() -> None:
    payload = tomllib.loads(
        """
[llm.providers.omlx]
kind = "omlx"
api_key = "secret"
"""
    )
    with pytest.raises(LLMConfigError):
        parse_llm_config(payload)
