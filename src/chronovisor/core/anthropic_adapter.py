"""Anthropic native Messages non-stream generation adapter."""

from __future__ import annotations

import math
from collections.abc import Mapping

from chronovisor.core.llm_runtime import (
    BackendCapabilities,
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


def _invalid_response(response: ProviderJSONResponse) -> ProviderAdapterError:
    return ProviderAdapterError(
        ProviderFailureCategory.INVALID_RESPONSE,
        request_id=response.request_id,
    )


def _messages_and_system(
    request: GenerationInput,
) -> tuple[list[dict[str, str]], str | None]:
    if isinstance(request, GenerationRequest):
        if not isinstance(request.prompt, str) or (
            request.system is not None and not isinstance(request.system, str)
        ):
            raise _invalid_request()
        return [{"role": "user", "content": request.prompt}], request.system
    messages: list[dict[str, str]] = []
    system_parts: list[str] = []
    for item in request.messages:
        if not isinstance(item, Mapping):
            raise _invalid_request()
        role = item.get("role")
        content = item.get("content")
        if role not in {"system", "user", "assistant"} or not isinstance(content, str):
            raise _invalid_request()
        if role == "system":
            system_parts.append(content)
        else:
            messages.append({"role": str(role), "content": content})
    if not messages:
        raise _invalid_request()
    return messages, "\n\n".join(system_parts) if system_parts else None


def _max_tokens(request: GenerationInput) -> int:
    value = request.max_output_tokens
    if value is None:
        return 1024
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise _invalid_request()
    return value


def _token_count(value: object) -> int | None:
    return (
        value
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0
        else None
    )


class AnthropicMessagesAdapter:
    location = RouteLocation.REMOTE

    def __init__(
        self,
        profile: ProviderProfile,
        transport: AuthenticatedTransport,
    ) -> None:
        if profile.protocol is not ProviderProtocol.ANTHROPIC_MESSAGES:
            raise ProviderAdapterError(ProviderFailureCategory.PROFILE_INVALID)
        self._profile = profile
        self._transport = transport
        self.provider = profile.profile_id

    @property
    def capabilities(self) -> BackendCapabilities:
        return self._profile.capabilities

    def generate(self, request: GenerationInput, *, model: str) -> GenerationResult:
        if not isinstance(model, str) or not model:
            raise _invalid_request()
        if request.format is not None and request.format != {}:
            raise ProviderAdapterError(ProviderFailureCategory.CAPABILITY_UNAVAILABLE)
        messages, system = _messages_and_system(request)
        payload: dict[str, object] = {
            "model": model,
            "messages": messages,
            "max_tokens": _max_tokens(request),
        }
        if system is not None:
            payload["system"] = system
        temperature = request.temperature
        if temperature is not None:
            if (
                isinstance(temperature, bool)
                or not isinstance(temperature, (int, float))
                or not math.isfinite(temperature)
            ):
                raise _invalid_request()
            payload["temperature"] = temperature
        response = post_json(
            self._transport,
            self._profile.url("/messages"),
            payload,
            headers={"anthropic-version": "2023-06-01"},
        )
        return self._generation_result(response, model)

    def _generation_result(
        self,
        response: ProviderJSONResponse,
        model: str,
    ) -> GenerationResult:
        blocks = response.payload.get("content")
        if not isinstance(blocks, list) or not blocks:
            raise _invalid_response(response)
        texts: list[str] = []
        for block in blocks:
            if (
                not isinstance(block, Mapping)
                or block.get("type") != "text"
                or not isinstance(block.get("text"), str)
            ):
                raise _invalid_response(response)
            texts.append(block["text"])
        usage = response.payload.get("usage")
        if not isinstance(usage, Mapping):
            raise _invalid_response(response)
        input_tokens = _token_count(usage.get("input_tokens"))
        output_tokens = _token_count(usage.get("output_tokens"))
        raw_finish_reason = response.payload.get("stop_reason")
        finish_reason = safe_finish_reason(
            raw_finish_reason,
            allowed=frozenset(
                {
                    "end_turn",
                    "max_tokens",
                    "model_context_window_exceeded",
                    "pause_turn",
                    "refusal",
                    "stop_sequence",
                }
            ),
        )
        if input_tokens is None or output_tokens is None or finish_reason is None:
            raise _invalid_response(response)
        return GenerationResult(
            content="".join(texts),
            provider=self.provider,
            model=model,
            finish_reason=finish_reason,
            usage=TokenUsage(input_tokens, output_tokens),
            metadata=response_metadata(response.payload, response.request_id),
        )


def compose_anthropic_adapter(
    profile: ProviderProfile,
    resolver: CredentialResolver,
    *,
    sender: RequestSender | None = None,
) -> AnthropicMessagesAdapter:
    if profile.protocol is not ProviderProtocol.ANTHROPIC_MESSAGES:
        raise ProviderAdapterError(ProviderFailureCategory.PROFILE_INVALID)
    return AnthropicMessagesAdapter(
        profile,
        authenticated_transport(profile, resolver, sender=sender),
    )
