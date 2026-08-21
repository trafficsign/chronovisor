"""oMLX components for the provider-neutral LLM runtime.

oMLX (z-lab DFlash fork) exposes an OpenAI-compatible HTTP API on
`http://127.0.0.1:8000/v1` with `x-api-key` header auth.  This adapter
mirrors `OllamaAdapter` but speaks that schema, with measured
parameter mapping:

- ``max_tokens``           <- ``max_output_tokens``
- ``temperature``          <- ``temperature``
- ``seed``                 <- ``seed`` (when present)
- ``chat_template_kwargs.enable_thinking`` <- boolean ``think`` and
  ``reasoning_effort`` <- string reasoning levels for message requests.
  Raw prompt requests explicitly disable thinking because their runtime type
  has no reasoning contract.
- ``response_format``      <- ``format`` (``"json"`` -> ``json_object``,
  mapping -> best-effort ``json_schema``)
- NOT sent: ``num_ctx``, ``keep_alive``.  oMLX silently ignores unknown
  parameters, so dropping them is the honest mapping (context/residency
  are oMLX-managed; see the migration handoff brief).
- embeddings via ``POST /v1/embeddings`` (normalized vectors).

DFlash engines are singleton per server (models swap; pinned DFlash
models block other DFlash loads), so cross-process serialization via
``ollama_lease.model_resource_lease`` is intentionally reused for oMLX.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from typing import Any

import httpx

from chronovisor.core.llm_runtime import (
    BackendCapabilities,
    EmbeddingRequest,
    EmbeddingResult,
    EmbeddingRoute,
    GenerationInput,
    GenerationResult,
    GenerationRoute,
    LLMRuntime,
    MessageGenerationRequest,
    RouteLocation,
    SafeBackendError,
    TokenUsage,
)
from chronovisor.core.ollama_lease import model_resource_lease

OMLX_BASE_URL = os.environ.get("OMLX_BASE_URL", "http://127.0.0.1:8000/v1")
OMLX_API_KEY = os.environ.get("OMLX_API_KEY", "omlx-local")
OMLX_MODEL_SETTINGS_PATH = Path(
    os.environ.get("OMLX_MODEL_SETTINGS_PATH", "~/.omlx/model_settings.json")
).expanduser()
_OMLX_AUTH_HEADER = "x-api-key"


class OMLXAdapter:
    """Compose generation, embedding, and lease control for oMLX."""

    provider = "omlx"
    location = RouteLocation.LOCAL

    def __init__(self, *, transport: httpx.BaseTransport | None = None) -> None:
        # `transport` is a test seam for httpx.MockTransport; production
        # callers construct OMLXAdapter() without arguments.
        self._transport = transport

    def generate(self, request: GenerationInput, *, model: str) -> GenerationResult:
        payload = self._request_payload(request, model=model)
        try:
            # ponytail: one DFlash engine per oMLX server; replace this global
            # lease only if oMLX gains native per-engine request queueing.
            lease = (
                model_resource_lease(exclusive=True, timeout_ms=request.timeout_ms)
                if self._uses_dflash(model)
                else nullcontext()
            )
            with lease:
                response = self._post(
                    "/chat/completions",
                    payload,
                    timeout_ms=request.timeout_ms,
                )
        except (TimeoutError, httpx.TimeoutException):
            raise SafeBackendError("timeout", transient=True) from None
        except httpx.TransportError:
            raise SafeBackendError("transport_error", transient=True) from None
        return self._normalize_chat(response, model=model, request=request)

    def embed(self, request: EmbeddingRequest, *, model: str) -> EmbeddingResult:
        try:
            response = self._post(
                "/embeddings",
                {"model": model, "input": list(request.texts)},
                timeout_ms=request.timeout_ms,
            )
        except httpx.TimeoutException:
            raise SafeBackendError("timeout", transient=True) from None
        except httpx.TransportError:
            raise SafeBackendError("transport_error", transient=True) from None
        payload = self._json_body(response)
        records = payload.get("data")
        if not isinstance(records, list):
            raise SafeBackendError("invalid_response", transient=False) from None
        vectors: list[tuple[float, ...]] = []
        for record in sorted(
            (item for item in records if isinstance(item, dict)),
            key=lambda item: item.get("index", 0) if isinstance(item.get("index"), int) else 0,
        ):
            embedding = record.get("embedding")
            if not isinstance(embedding, list):
                raise SafeBackendError("invalid_response", transient=False) from None
            vectors.append(tuple(float(value) for value in embedding))
        return EmbeddingResult(
            vectors=tuple(vectors),
            provider=self.provider,
            model=model,
        )

    def resource_lease(
        self, *, exclusive: bool, timeout_ms: int | None = None
    ) -> AbstractContextManager[None]:
        return model_resource_lease(exclusive=exclusive, timeout_ms=timeout_ms)

    def resident_models(self) -> Mapping[str, tuple[int, int]]:
        # oMLX manages residency internally; no per-model residency info
        # is exposed over the HTTP API.  Adaptive residency that depends
        # on this must be reconsidered for the omlx provider.
        return {}

    def unload(self, model: str, *, verify_timeout: float = 30.0) -> bool:
        # Not exposed by oMLX; no-op that reports failure honestly.
        return False

    def _request_payload(self, request: GenerationInput, *, model: str) -> dict[str, object]:
        if isinstance(request, MessageGenerationRequest):
            messages = [dict(message) for message in request.messages]
            think: object = request.think
            temperature: object | None = request.temperature
            seed: object | None = request.seed
        else:
            messages = []
            if request.system is not None:
                messages.append({"role": "system", "content": request.system})
            messages.append({"role": "user", "content": request.prompt})
            # GenerationRequest has no reasoning contract.  oMLX chat models
            # otherwise use their template default, which can be unbounded.
            think = False
            temperature = request.temperature
            seed = request.seed
        payload: dict[str, object] = {"model": model, "messages": messages}
        if request.max_output_tokens is not None:
            payload["max_tokens"] = request.max_output_tokens
        if temperature is not None:
            payload["temperature"] = temperature
        if seed is not None:
            payload["seed"] = seed
        format_value: Mapping[str, Any] | str | None = (
            dict(request.format) if isinstance(request.format, Mapping) else request.format
        )
        if format_value is not None:
            payload["response_format"] = self._response_format(format_value)
        if isinstance(think, (bool, str)):
            payload["chat_template_kwargs"] = {"enable_thinking": think is not False}
            if isinstance(think, str):
                payload["reasoning_effort"] = think
        return payload

    def _post(
        self,
        path: str,
        payload: Mapping[str, object],
        *,
        timeout_ms: int | None,
    ) -> httpx.Response:
        with httpx.Client(base_url=OMLX_BASE_URL, transport=self._transport) as client:
            return client.post(
                path,
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    _OMLX_AUTH_HEADER: OMLX_API_KEY,
                },
                timeout=None if timeout_ms is None else timeout_ms / 1000,
            )

    @staticmethod
    def _uses_dflash(model: str) -> bool:
        """Conservatively serialize models unless settings explicitly disable DFlash."""

        try:
            settings = json.loads(OMLX_MODEL_SETTINGS_PATH.read_text())
            model_settings = settings["models"][model]
        except (OSError, ValueError, KeyError, TypeError):
            return True
        return not (
            isinstance(model_settings, dict)
            and model_settings.get("dflash_enabled") is False
        )

    @staticmethod
    def _response_format(value: Mapping[str, Any] | str) -> dict[str, object]:
        if isinstance(value, str):
            if value != "json":
                raise SafeBackendError("invalid_request", transient=False)
            return {"type": "json_object"}
        return {
            "type": "json_schema",
            "json_schema": {"name": "response", "strict": True, "schema": dict(value)},
        }

    @staticmethod
    def _json_body(response: httpx.Response) -> dict[str, Any]:
        if response.status_code >= 300:
            if response.status_code == 401:
                category, transient = "http_401", False
            elif response.status_code == 429:
                category, transient = "http_429", True
            elif response.status_code >= 500:
                category, transient = "http_5xx", True
            else:
                category, transient = "invalid_request", False
            raise SafeBackendError(category, transient=transient)
        try:
            payload = response.json()
        except ValueError:
            raise SafeBackendError("invalid_response", transient=False) from None
        if not isinstance(payload, dict):
            raise SafeBackendError("invalid_response", transient=False) from None
        return payload

    def _normalize_chat(
        self,
        response: httpx.Response,
        *,
        model: str,
        request: GenerationInput,
    ) -> GenerationResult:
        payload = self._json_body(response)
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise SafeBackendError("invalid_response", transient=False) from None
        message = choices[0].get("message")
        if not isinstance(message, dict):
            raise SafeBackendError("invalid_response", transient=False) from None
        content = message.get("content")
        reasoning = message.get("reasoning_content")
        if not isinstance(content, str) or content == "":
            # Thinking-only turns (e.g. small max_tokens) return an empty
            # content with a reasoning_content payload; best-effort extract.
            content = reasoning if isinstance(reasoning, str) else ""
        finish_reason = choices[0].get("finish_reason")
        metadata: dict[str, Any] = {}
        if isinstance(reasoning, str) and reasoning:
            metadata["reasoning_content"] = reasoning
        metadata["total_time"] = payload.get("usage", {}).get("total_time")
        metadata["model_load_duration"] = payload.get("usage", {}).get("model_load_duration")
        usage = payload.get("usage")
        input_tokens: int | None = None
        output_tokens: int | None = None
        if isinstance(usage, dict):
            input_tokens = usage.get("prompt_tokens") or usage.get("input_tokens")
            output_tokens = usage.get("completion_tokens") or usage.get("output_tokens")
        max_chars: int | None = getattr(request, "max_output_chars", None)
        if isinstance(max_chars, int) and max_chars > 0 and content:
            content = content[:max_chars]
        return GenerationResult(
            content=content,
            provider=self.provider,
            model=model,
            completed=bool(finish_reason is not None) or bool(content),
            finish_reason=finish_reason if isinstance(finish_reason, str) else None,
            usage=TokenUsage(
                input_tokens=input_tokens or 0,
                output_tokens=output_tokens or 0,
            ),
            metadata=metadata,
        )


def compose_omlx_runtime(
    *,
    generation_roles: Mapping[str, str] | None = None,
    embedding_roles: Mapping[str, str] | None = None,
) -> LLMRuntime:
    """Build role routes over one shared oMLX adapter (legacy helper).

    Production wiring goes through ``llm_config.build_llm_runtime`` with
    provider kind ``"omlx"``; this helper mirrors ``compose_ollama_runtime``
    for parity with existing legacy callers/tests.
    """

    adapter = OMLXAdapter()
    generation_roles = generation_roles or {}
    embedding_roles = embedding_roles or {}
    local_roles = generation_roles.keys() | embedding_roles.keys()
    return LLMRuntime(
        generation={
            role: GenerationRoute(
                adapter,
                model,
                BackendCapabilities(
                    generation=True, embedding=True, structured_output=True
                ),
                "omlx-native",
                hashlib.sha256(OMLX_BASE_URL.encode("utf-8")).hexdigest(),
            )
            for role, model in generation_roles.items()
        },
        embedding={
            role: EmbeddingRoute(adapter, model)
            for role, model in embedding_roles.items()
        },
        local_controls={role: adapter for role in local_roles},
    )
