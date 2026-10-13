"""MTPLX keeps Gate prompts intact while closing the transport schema."""

import hashlib
import json
from copy import deepcopy

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
from chronovisor.decision.local_structured import validate_json
from chronovisor.recall.recall_distillation_worker import _schema


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


def test_teacher_wire_schema_preserves_strict_response_validation() -> None:
    schema = _schema("teacher", candidate_ids=("candidate-a",))
    original = deepcopy(schema)
    wire = MTPLXAdapter._response_format(schema)["json_schema"]["schema"]
    fields = wire["properties"]["labels"]["items"]["properties"]
    assert "uniqueItems" not in fields["minimal_atom_ids"]
    assert "uniqueItems" not in fields["missing_slots"]
    assert fields["minimal_atom_ids"]["maxItems"] == 8
    assert schema == original
    label = {
        "candidate_id": "candidate-a", "verdict": "relevant", "confidence": 0.9,
        "rationale": "Exact matching evidence", "minimal_atom_ids": ["atom-a"],
        "missing_slots": [], "changing_claim": "",
    }
    assert validate_json({"labels": [label]}, schema) == []
    label["minimal_atom_ids"] = ["atom-a", "atom-a"]
    assert [issue.keyword for issue in validate_json({"labels": [label]}, schema)] == [
        "uniqueItems"
    ]


def test_wire_schema_keeps_property_names_and_literal_values() -> None:
    literal = {"type": "array", "uniqueItems": True}
    schema = {
        "type": "object",
        "properties": {"uniqueItems": {"const": literal}},
        "additionalProperties": {"type": "array", "items": {"type": "string"}, "uniqueItems": True},
    }
    wire = MTPLXAdapter._response_format(schema)["json_schema"]["schema"]
    assert wire["properties"]["uniqueItems"]["const"] == literal
    assert schema["additionalProperties"]["uniqueItems"] is True


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
