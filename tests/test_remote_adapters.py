from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any, cast
from urllib.request import Request

import httpx
import pytest

from chronovisor.core.anthropic_adapter import compose_anthropic_adapter
from chronovisor.core.llm_runtime import (
    EmbeddingRequest,
    GenerationRequest,
    SourceDataClass,
    SourceDataClassification,
    SourceSensitivity,
)
from chronovisor.core.llm_security import AuthScheme, CredentialRef, CredentialResolver
from chronovisor.core.openai_compatible_adapter import (
    compose_openai_compatible_adapter,
)
from chronovisor.core.provider_profiles import (
    CURATED_PROFILE_IDS,
    HTTPXSender,
    ProviderAdapterError,
    ProviderFailureCategory,
    ProviderProfile,
    ProviderProtocol,
    curated_profile,
    generic_openai_profile,
)

CANARY = "sk-CANARY-REMOTE-ADAPTER"
CREDENTIAL_REF = CredentialRef.parse("env:REMOTE_ADAPTER_API_KEY")
NORMAL_PAGE = SourceDataClassification(SourceDataClass.PAGE, SourceSensitivity.NORMAL)
QWEN_ENDPOINT = "https://dashscope.example.test/compatible-mode/v1"


class FakeSender:
    def __init__(self, *outcomes: object) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[tuple[Request, bool]] = []

    def __call__(self, request: Request, *, follow_redirects: bool) -> object:
        self.calls.append((request, follow_redirects))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _resolver(tmp_path: Path) -> CredentialResolver:
    return CredentialResolver(
        environ={CREDENTIAL_REF.target: CANARY},
        repo_root=tmp_path / "repo",
        home_root=tmp_path / "home",
    )


def _response(
    payload: object | None = None,
    *,
    status: int = 200,
    body: bytes | None = None,
    request_id: str = "req_fixture_1",
) -> httpx.Response:
    if body is not None:
        return httpx.Response(
            status, content=body, headers={"x-request-id": request_id}
        )
    return httpx.Response(status, json=payload, headers={"x-request-id": request_id})


def _openai_success(
    *,
    content: str = "answer",
    returned_model: str = "provider-returned-model",
) -> httpx.Response:
    return _response(
        {
            "model": returned_model,
            "choices": [
                {
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2},
        }
    )


def _curated(profile_id: str) -> ProviderProfile:
    return curated_profile(
        profile_id,
        CREDENTIAL_REF,
        endpoint_override=(
            QWEN_ENDPOINT if profile_id in {"qwen", "dashscope"} else None
        ),
    )


def test_curated_profiles_are_immutable_conservative_capability_truth() -> None:
    embedding_ids = {"openai", "qwen", "dashscope", "gemini", "mistral", "openrouter"}
    non_embedding_ids = {"deepseek", "kimi", "zai", "glm", "anthropic"}

    assert set(CURATED_PROFILE_IDS) == embedding_ids | non_embedding_ids
    for profile_id in CURATED_PROFILE_IDS:
        profile = _curated(profile_id)
        capabilities = profile.capabilities_for("arbitrary-model")
        assert profile.profile_id == profile_id
        assert capabilities.generation
        assert capabilities.embedding is (profile_id in embedding_ids)
        assert not capabilities.structured_output
        assert not capabilities.streaming
        assert not capabilities.tools
        assert not capabilities.rerank
        assert isinstance(profile.credential_ref, CredentialRef)
        with pytest.raises(FrozenInstanceError):
            profile.endpoint = "https://other.example.com"  # type: ignore[misc]


def test_curated_endpoint_and_auth_matrix_is_fixed() -> None:
    expected = {
        "openai": ("https://api.openai.com/v1", AuthScheme.BEARER),
        "qwen": (QWEN_ENDPOINT, AuthScheme.BEARER),
        "dashscope": (QWEN_ENDPOINT, AuthScheme.BEARER),
        "gemini": (
            "https://generativelanguage.googleapis.com/v1beta/openai",
            AuthScheme.BEARER,
        ),
        "deepseek": ("https://api.deepseek.com", AuthScheme.BEARER),
        "kimi": ("https://api.moonshot.ai/v1", AuthScheme.BEARER),
        "zai": ("https://api.z.ai/api/paas/v4", AuthScheme.BEARER),
        "glm": ("https://api.z.ai/api/paas/v4", AuthScheme.BEARER),
        "mistral": ("https://api.mistral.ai/v1", AuthScheme.BEARER),
        "openrouter": ("https://openrouter.ai/api/v1", AuthScheme.BEARER),
        "anthropic": ("https://api.anthropic.com/v1", AuthScheme.X_API_KEY),
    }

    assert {
        profile_id: (_curated(profile_id).endpoint, _curated(profile_id).auth_scheme)
        for profile_id in CURATED_PROFILE_IDS
    } == expected


def test_qwen_requires_explicit_region_or_workspace_endpoint() -> None:
    for profile_id in ("qwen", "dashscope"):
        with pytest.raises(ProviderAdapterError) as exc:
            curated_profile(profile_id, CREDENTIAL_REF)
        assert exc.value.category is ProviderFailureCategory.PROFILE_INVALID


def test_generic_profile_uses_same_data_contract_and_model_scoped_structured() -> None:
    profile = generic_openai_profile(
        "private-gateway",
        "https://gateway.example.com/openai/v1",
        CREDENTIAL_REF,
        embedding=True,
        structured_output_models={"json-model"},
    )

    assert profile.protocol is ProviderProtocol.OPENAI_COMPATIBLE
    assert profile.capabilities_for("json-model").structured_output
    assert not profile.capabilities_for("other-model").structured_output
    assert profile.capabilities.embedding
    with pytest.raises(ProviderAdapterError):
        cast(Any, ProviderProfile)(
            profile_id="bad/profile",
            protocol=ProviderProtocol.OPENAI_COMPATIBLE,
            endpoint="https://gateway.example.com/v1",
            credential_ref=CANARY,
            auth_scheme=profile.auth_scheme,
            capabilities=profile.capabilities,
        )


@pytest.mark.parametrize(
    "profile_id",
    [
        "openai",
        "qwen",
        "dashscope",
        "gemini",
        "deepseek",
        "kimi",
        "zai",
        "glm",
        "mistral",
        "openrouter",
    ],
)
def test_all_openai_compatible_curated_profiles_share_generation_contract(
    profile_id: str,
    tmp_path: Path,
) -> None:
    sender = FakeSender(_openai_success())
    adapter = compose_openai_compatible_adapter(
        _curated(profile_id), _resolver(tmp_path), sender=sender
    )

    result = adapter.generate(
        GenerationRequest("prompt", NORMAL_PAGE, max_output_tokens=64),
        model="route-model",
    )

    assert result.provider == profile_id
    assert result.model == "route-model"
    assert result.metadata == {
        "returned_model": "provider-returned-model",
        "request_id": "req_fixture_1",
    }
    request, follow_redirects = sender.calls[0]
    assert follow_redirects is False
    assert request.full_url.endswith("/chat/completions")
    assert CANARY not in repr(request)


def test_openai_generation_normalizes_body_and_sends_structured_only_when_scoped(
    tmp_path: Path,
) -> None:
    profile = generic_openai_profile(
        "generic-json",
        "https://gateway.example.com/v1",
        CREDENTIAL_REF,
        structured_output_models={"json-model"},
    )
    sender = FakeSender(_openai_success(content='{"ok":true}'))
    adapter = compose_openai_compatible_adapter(
        profile, _resolver(tmp_path), sender=sender
    )
    schema = {"type": "object", "required": ["ok"]}

    result = adapter.generate(
        GenerationRequest(
            "prompt",
            NORMAL_PAGE,
            system="system",
            format=schema,
            temperature=0,
        ),
        model="json-model",
    )

    assert result.content == '{"ok":true}'
    assert result.usage.input_tokens == 3
    request_body = json.loads(cast(bytes, sender.calls[0][0].data))
    assert request_body["messages"] == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "prompt"},
    ]
    assert request_body["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "response", "strict": True, "schema": schema},
    }

    denied_sender = FakeSender(_openai_success())
    denied = compose_openai_compatible_adapter(
        profile, _resolver(tmp_path), sender=denied_sender
    )
    with pytest.raises(ProviderAdapterError) as exc:
        denied.generate(
            GenerationRequest("prompt", NORMAL_PAGE, format=schema),
            model="other-model",
        )
    assert exc.value.category is ProviderFailureCategory.CAPABILITY_UNAVAILABLE
    assert denied_sender.calls == []


def test_embedding_response_is_reordered_and_count_validated(tmp_path: Path) -> None:
    sender = FakeSender(
        _response(
            {
                "model": "returned-embedder",
                "data": [
                    {"index": 1, "embedding": [3, 4]},
                    {"index": 0, "embedding": [1, 2]},
                ],
            }
        ),
        _response({"data": [{"index": 0, "embedding": [1, 2]}]}),
    )
    adapter = compose_openai_compatible_adapter(
        _curated("openai"), _resolver(tmp_path), sender=sender
    )
    request = EmbeddingRequest(("first", "second"), NORMAL_PAGE)

    result = adapter.embed(request, model="route-embedder")

    assert result.vectors == ((1.0, 2.0), (3.0, 4.0))
    assert result.model == "route-embedder"
    with pytest.raises(ProviderAdapterError) as exc:
        adapter.embed(request, model="route-embedder")
    assert exc.value.category is ProviderFailureCategory.INVALID_RESPONSE


@pytest.mark.parametrize("profile_id", ["deepseek", "kimi", "zai", "glm"])
def test_unconfirmed_embedding_capability_fails_before_network(
    profile_id: str,
    tmp_path: Path,
) -> None:
    sender = FakeSender(_response({"data": []}))
    adapter = compose_openai_compatible_adapter(
        _curated(profile_id), _resolver(tmp_path), sender=sender
    )

    with pytest.raises(ProviderAdapterError) as exc:
        adapter.embed(EmbeddingRequest(("text",), NORMAL_PAGE), model="embedder")

    assert exc.value.category is ProviderFailureCategory.CAPABILITY_UNAVAILABLE
    assert sender.calls == []


@pytest.mark.parametrize(
    "response, category",
    [
        (
            _response({"error": CANARY}, status=401),
            ProviderFailureCategory.UNAUTHORIZED,
        ),
        (
            _response({"error": CANARY}, status=429),
            ProviderFailureCategory.RATE_LIMITED,
        ),
        (
            _response({"error": CANARY}, status=503),
            ProviderFailureCategory.SERVER_ERROR,
        ),
        (
            _response({"redirect": CANARY}, status=302),
            ProviderFailureCategory.REDIRECT_REJECTED,
        ),
        (
            _response(body=CANARY.encode("utf-8")),
            ProviderFailureCategory.INVALID_RESPONSE,
        ),
        (
            _response({"choices": [{"message": CANARY}]}),
            ProviderFailureCategory.INVALID_RESPONSE,
        ),
    ],
)
def test_openai_failures_are_safe_and_body_free(
    response: httpx.Response,
    category: ProviderFailureCategory,
    tmp_path: Path,
) -> None:
    sender = FakeSender(response)
    adapter = compose_openai_compatible_adapter(
        _curated("openai"), _resolver(tmp_path), sender=sender
    )

    with pytest.raises(ProviderAdapterError) as exc:
        adapter.generate(
            GenerationRequest(CANARY, NORMAL_PAGE, system=CANARY), model="model"
        )

    assert exc.value.category is category
    assert exc.value.request_id == "req_fixture_1"
    assert exc.value.__cause__ is None
    assert exc.value.__context__ is None
    assert CANARY not in str(exc.value)
    assert CANARY not in repr(exc.value)
    with pytest.raises(TypeError) as json_exc:
        json.dumps(exc.value)
    assert CANARY not in str(json_exc.value)


def test_timeout_and_transport_failure_are_distinct_and_redacted(
    tmp_path: Path,
) -> None:
    for failure, category in (
        (httpx.ReadTimeout(CANARY), ProviderFailureCategory.TIMEOUT),
        (OSError(CANARY), ProviderFailureCategory.TRANSPORT_ERROR),
    ):
        sender = FakeSender(failure)
        adapter = compose_openai_compatible_adapter(
            _curated("openai"), _resolver(tmp_path), sender=sender
        )
        with pytest.raises(ProviderAdapterError) as exc:
            adapter.generate(GenerationRequest(CANARY, NORMAL_PAGE), model="model")
        assert exc.value.category is category
        assert CANARY not in repr(exc.value)


def test_httpx_sender_pins_tls_defaults_and_redirect_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_request(method: str, url: str, **kwargs: object) -> httpx.Response:
        captured.update({"method": method, "url": url, **kwargs})
        return _openai_success()

    monkeypatch.setattr(httpx, "request", fake_request)
    request = Request(
        "https://api.example.com/v1/chat/completions",
        data=b"{}",
        method="POST",
    )

    HTTPXSender()(request, follow_redirects=False)

    assert captured["follow_redirects"] is False
    assert "verify" not in captured


def test_anthropic_native_messages_shape_and_generation_only(tmp_path: Path) -> None:
    sender = FakeSender(
        _response(
            {
                "model": "claude-returned",
                "content": [
                    {"type": "text", "text": "first"},
                    {"type": "text", "text": " second"},
                ],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 4, "output_tokens": 2},
            }
        )
    )
    adapter = compose_anthropic_adapter(
        _curated("anthropic"), _resolver(tmp_path), sender=sender
    )

    result = adapter.generate(
        GenerationRequest("prompt", NORMAL_PAGE, system="system", max_output_tokens=32),
        model="claude-route",
    )

    assert result.content == "first second"
    assert result.model == "claude-route"
    assert result.metadata["returned_model"] == "claude-returned"
    request = sender.calls[0][0]
    body = json.loads(cast(bytes, request.data))
    assert body == {
        "model": "claude-route",
        "messages": [{"role": "user", "content": "prompt"}],
        "max_tokens": 32,
        "system": "system",
    }
    assert request.get_header("X-api-key") == CANARY
    assert request.get_header("Anthropic-version") == "2023-06-01"
    assert not hasattr(adapter, "embed")
    assert not hasattr(adapter, "rerank")


def test_anthropic_rejects_structured_and_non_text_shape_before_leak(
    tmp_path: Path,
) -> None:
    profile = _curated("anthropic")
    sender = FakeSender(
        _response(
            {
                "content": [{"type": "tool_use", "input": CANARY}],
                "stop_reason": "tool_use",
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }
        )
    )
    adapter = compose_anthropic_adapter(profile, _resolver(tmp_path), sender=sender)

    with pytest.raises(ProviderAdapterError) as structured:
        adapter.generate(
            GenerationRequest("prompt", NORMAL_PAGE, format={"type": "object"}),
            model="claude",
        )
    assert structured.value.category is ProviderFailureCategory.CAPABILITY_UNAVAILABLE
    assert sender.calls == []

    with pytest.raises(ProviderAdapterError) as malformed:
        adapter.generate(GenerationRequest(CANARY, NORMAL_PAGE), model="claude")
    assert malformed.value.category is ProviderFailureCategory.INVALID_RESPONSE
    assert CANARY not in repr(malformed.value)
