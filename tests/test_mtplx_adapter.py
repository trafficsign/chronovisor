"""MTPLX keeps Gate prompts intact while closing the transport schema."""

import hashlib
import json

import httpx
import pytest

from chronovisor.core.llm_config import (
    LLMConfigError,
    build_llm_runtime,
    parse_llm_config,
)
from chronovisor.core.llm_runtime import (
    MessageGenerationRequest,
    SourceDataClass,
    SourceDataClassification,
    SourceSensitivity,
)
from chronovisor.core.mtplx_adapter import MTPLX_BASE_URL, MTPLXAdapter


def test_structured_transport_preserves_prompt_and_input_schema() -> None:
    captured = []

    def handle(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(
            200,
            json={
                "model": "ornith",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": '{"decision":"skip"}',
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            },
        )

    schema = {
        "type": "object",
        "properties": {"decision": {"type": "string"}},
        "required": ["decision"],
    }
    system = "Original schema: " + json.dumps(schema)
    messages = (
        {"role": "system", "content": system},
        {"role": "user", "content": "hello"},
    )
    request = MessageGenerationRequest(
        messages=messages,
        format=schema,
        source=SourceDataClassification(SourceDataClass.PAGE, SourceSensitivity.NORMAL),
        num_ctx=4096,
        max_output_tokens=64,
        keep_alive="24h",
        timeout_ms=1500,
        max_output_chars=384,
        temperature=0,
        seed=0,
        think=False,
    )
    adapter = MTPLXAdapter(transport=httpx.MockTransport(handle))
    result = adapter.generate(request, model="ornith")
    body = json.loads(captured[0].content)
    assert str(captured[0].url) == MTPLX_BASE_URL + "/chat/completions"
    assert body["messages"] == list(messages)
    assert (
        body["response_format"]["json_schema"]["schema"]["additionalProperties"]
        is False
    )
    assert "additionalProperties" not in schema
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert result.provider == "mtplx"
    assert not adapter._uses_dflash("ornith")


def test_mtplx_route_is_local_generation_only() -> None:
    payload = {
        "llm": {
            "providers": {"mtplx": {"kind": "mtplx"}},
            "roles": {
                "recall.gate": {
                    "capability": "generation",
                    "provider": "mtplx",
                    "model": "ornith",
                }
            },
        }
    }
    config = parse_llm_config(payload)
    assert not config.providers["mtplx"].capabilities_for("ornith").embedding
    route = build_llm_runtime(config).resolve_generation("recall.gate")
    assert route.provider == "mtplx"
    assert route.protocol == "mtplx-native"
    assert route.endpoint_sha256 == hashlib.sha256(MTPLX_BASE_URL.encode()).hexdigest()
    payload["llm"]["roles"]["recall.gate"]["capability"] = "embedding"
    with pytest.raises(LLMConfigError):
        parse_llm_config(payload)


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://example.com:18145/v1",
        "http://127.0.0.1:18145/admin",
        "http://user:pass@127.0.0.1:18145/v1",
        "https://127.0.0.1:18145/v1",
    ],
)
def test_mtplx_endpoint_rejects_remote_or_ambiguous_routes(endpoint: str) -> None:
    with pytest.raises(LLMConfigError):
        parse_llm_config(
            {"llm": {"providers": {"mtplx": {"kind": "mtplx", "endpoint": endpoint}}}}
        )
