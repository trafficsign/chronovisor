"""OpenAI-compatible non-stream generation and embedding adapter."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import cast

from chronovisor.core.llm_runtime import (
    BackendCapabilities,
    EmbeddingRequest,
    EmbeddingResult,
    GenerationInput,
    GenerationRequest,
    GenerationResult,
    RouteLocation,
    TokenUsage,
)
from chronovisor.core.llm_security import (
    AuthenticatedTransport,
    CredentialResolver,
    RequestSender,
)
from chronovisor.core.provider_profiles import (
    ProviderAdapterError,
    ProviderFailureCategory,
    ProviderJSONResponse,
    ProviderProfile,
    ProviderProtocol,
    authenticated_transport,
    post_json,
    response_metadata,
    safe_finish_reason,
)


def _invalid_request() -> ProviderAdapterError:
    return ProviderAdapterError(ProviderFailureCategory.INVALID_REQUEST)


def _invalid_response(
    response: ProviderJSONResponse, stage: str | None = None
) -> ProviderAdapterError:
    return ProviderAdapterError(
        ProviderFailureCategory.INVALID_RESPONSE,
        request_id=response.request_id,
        stage=stage,
    )


def _messages(request: GenerationInput) -> list[dict[str, str]]:
    if isinstance(request, GenerationRequest):
        if not isinstance(request.prompt, str) or (
            request.system is not None and not isinstance(request.system, str)
        ):
            raise _invalid_request()
        messages = []
        if request.system is not None:
            messages.append({"role": "system", "content": request.system})
        messages.append({"role": "user", "content": request.prompt})
        return messages
    messages = []
    for item in request.messages:
        if not isinstance(item, Mapping):
            raise _invalid_request()
        role = item.get("role")
        content = item.get("content")
        if role not in {"system", "user", "assistant"} or not isinstance(content, str):
            raise _invalid_request()
        messages.append({"role": role, "content": content})
    if not messages:
        raise _invalid_request()
    return messages


def _request_format(request: GenerationInput) -> Mapping[str, object] | str | None:
    value = request.format
    if value is None or value == {}:
        return None
    if isinstance(value, str):
        if value != "json":
            raise _invalid_request()
        return value
    if not isinstance(value, Mapping):
        raise _invalid_request()
    return cast(Mapping[str, object], value)


def _response_format(value: Mapping[str, object] | str) -> Mapping[str, object]:
    if value == "json":
        return {"type": "json_object"}
    return {
        "type": "json_schema",
        "json_schema": {"name": "response", "strict": True, "schema": value},
    }


def _optional_generation_parameters(
    request: GenerationInput,
) -> dict[str, object]:
    result: dict[str, object] = {}
    max_tokens = request.max_output_tokens
    if max_tokens is not None:
        if (
            isinstance(max_tokens, bool)
            or not isinstance(max_tokens, int)
            or max_tokens <= 0
        ):
            raise _invalid_request()
        result["max_tokens"] = max_tokens
    temperature = request.temperature
    if temperature is not None:
        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not math.isfinite(temperature)
        ):
            raise _invalid_request()
        result["temperature"] = temperature
    return result


def _token_count(value: object) -> int | None:
    return (
        value
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0
        else None
    )


class OpenAICompatibleAdapter:
    location = RouteLocation.REMOTE

    def __init__(
        self,
        profile: ProviderProfile,
        transport: AuthenticatedTransport,
    ) -> None:
        if profile.protocol is not ProviderProtocol.OPENAI_COMPATIBLE:
            raise ProviderAdapterError(ProviderFailureCategory.PROFILE_INVALID)
        self._profile = profile
        self._transport = transport
        self.provider = profile.profile_id

    @property
    def capabilities(self) -> BackendCapabilities:
        return self._profile.capabilities

    def capabilities_for(self, model: str) -> BackendCapabilities:
        return self._profile.capabilities_for(model)

    def generate(self, request: GenerationInput, *, model: str) -> GenerationResult:
        if not isinstance(model, str) or not model:
            raise _invalid_request()
        format_value = _request_format(request)
        capabilities = self.capabilities_for(model)
        if format_value is not None and not capabilities.structured_output:
            raise ProviderAdapterError(ProviderFailureCategory.CAPABILITY_UNAVAILABLE)
        payload: dict[str, object] = {
            "model": model,
            "messages": _messages(request),
            **_optional_generation_parameters(request),
        }
        if format_value is not None:
            payload["response_format"] = _response_format(format_value)
        response = post_json(
            self._transport,
            self._profile.url("/chat/completions"),
            payload,
            timeout_ms=request.timeout_ms,
        )
        return self._generation_result(response, model)

    def embed(self, request: EmbeddingRequest, *, model: str) -> EmbeddingResult:
        if not self._profile.capabilities.embedding:
            raise ProviderAdapterError(ProviderFailureCategory.CAPABILITY_UNAVAILABLE)
        if (
            not isinstance(model, str)
            or not model
            or not isinstance(request.texts, tuple)
            or not request.texts
            or not all(isinstance(text, str) for text in request.texts)
        ):
            raise _invalid_request()
        response = post_json(
            self._transport,
            self._profile.url("/embeddings"),
            {"model": model, "input": list(request.texts)},
            timeout_ms=request.timeout_ms,
        )
        return self._embedding_result(response, model, len(request.texts))

    def _generation_result(
        self,
        response: ProviderJSONResponse,
        model: str,
    ) -> GenerationResult:
        choices = response.payload.get("choices")
        if not isinstance(choices, list) or len(choices) != 1:
            raise _invalid_response(response, "choices_shape")
        choice = choices[0]
        if not isinstance(choice, Mapping):
            raise _invalid_response(response, "choice_shape")
        message = choice.get("message")
        if not isinstance(message, Mapping):
            raise _invalid_response(response, "message_shape")
        content = message.get("content")
        if not isinstance(content, str):
            raise _invalid_response(response, "content_shape")
        raw_finish_reason = choice.get("finish_reason")
        finish_reason = safe_finish_reason(
            raw_finish_reason,
            allowed=frozenset({"stop", "length", "content_filter"}),
        )
        if finish_reason is None:
            raise _invalid_response(response, "finish_reason")
        usage = response.payload.get("usage")
        token_usage = TokenUsage()
        if usage is not None:
            if not isinstance(usage, Mapping):
                raise _invalid_response(response, "usage_shape")
            input_tokens = _token_count(usage.get("prompt_tokens"))
            output_tokens = _token_count(usage.get("completion_tokens"))
            if input_tokens is None or output_tokens is None:
                raise _invalid_response(response, "usage_tokens")
            token_usage = TokenUsage(input_tokens, output_tokens)
        return GenerationResult(
            content=content,
            provider=self.provider,
            model=model,
            finish_reason=finish_reason,
            usage=token_usage,
            metadata=response_metadata(response.payload, response.request_id),
        )

    def _embedding_result(
        self,
        response: ProviderJSONResponse,
        model: str,
        expected_count: int,
    ) -> EmbeddingResult:
        rows = response.payload.get("data")
        if not isinstance(rows, list) or len(rows) != expected_count:
            raise _invalid_response(response)
        vectors: dict[int, tuple[float, ...]] = {}
        for row in rows:
            if not isinstance(row, Mapping):
                raise _invalid_response(response)
            index = row.get("index")
            embedding = row.get("embedding")
            if (
                isinstance(index, bool)
                or not isinstance(index, int)
                or index in vectors
                or not isinstance(embedding, list)
                or not embedding
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    for value in embedding
                )
            ):
                raise _invalid_response(response)
            vectors[index] = tuple(float(value) for value in embedding)
        if (
            set(vectors) != set(range(expected_count))
            or len({len(vector) for vector in vectors.values()}) != 1
        ):
            raise _invalid_response(response)
        return EmbeddingResult(
            tuple(vectors[index] for index in range(expected_count)),
            self.provider,
            model,
        )


def compose_openai_compatible_adapter(
    profile: ProviderProfile,
    resolver: CredentialResolver,
    *,
    sender: RequestSender | None = None,
) -> OpenAICompatibleAdapter:
    if profile.protocol is not ProviderProtocol.OPENAI_COMPATIBLE:
        raise ProviderAdapterError(ProviderFailureCategory.PROFILE_INVALID)
    return OpenAICompatibleAdapter(
        profile,
        authenticated_transport(profile, resolver, sender=sender),
    )
