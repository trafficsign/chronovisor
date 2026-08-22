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
models block other DFlash loads), so separate local servers can keep one
DFlash model resident each while cross-process serialization via
``ollama_lease.model_resource_lease`` keeps inference exclusive.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable, Mapping
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

    def __init__(
        self,
        *,
        base_url: str = OMLX_BASE_URL,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        # `transport` is a test seam for httpx.MockTransport; production
        # callers construct OMLXAdapter() without arguments.
        self.base_url = base_url.rstrip("/")
        self._transport = transport

    def generate(self, request: GenerationInput, *, model: str) -> GenerationResult:
        payload = self._request_payload(request, model=model)
        progress_callback = getattr(request, "progress_callback", None)
        streamed = callable(progress_callback)
        stream_complete: bool | None = None
        try:
            # ponytail: one DFlash engine per oMLX server; replace this global
            # lease only if oMLX gains native per-engine request queueing.
            lease = (
                model_resource_lease(exclusive=True, timeout_ms=request.timeout_ms)
                if self._uses_dflash(model)
                else nullcontext()
            )
            with lease:
                if streamed:
                    response, stream_complete = self._post_stream(
                        "/chat/completions",
                        payload,
                        timeout_ms=request.timeout_ms,
                        progress_callback=progress_callback,
                        max_output_tokens=request.max_output_tokens,
                    )
                else:
                    response = self._post(
                        "/chat/completions",
                        payload,
                        timeout_ms=request.timeout_ms,
                    )
        except (TimeoutError, httpx.TimeoutException):
            raise SafeBackendError("timeout", transient=True) from None
        except httpx.TransportError:
            raise SafeBackendError("transport_error", transient=True) from None
        return self._normalize_chat(
            response,
            model=model,
            request=request,
            streamed=streamed,
            stream_complete=stream_complete,
        )

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
        with httpx.Client(base_url=self.base_url, transport=self._transport) as client:
            return client.post(
                path,
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    _OMLX_AUTH_HEADER: OMLX_API_KEY,
                },
                timeout=None if timeout_ms is None else timeout_ms / 1000,
            )

    def _post_stream(
        self,
        path: str,
        payload: Mapping[str, object],
        *,
        timeout_ms: int | None,
        progress_callback: Callable[[dict[str, Any]], None],
        max_output_tokens: int | None,
    ) -> tuple[httpx.Response, bool]:
        request_payload = {
            **payload,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        content: list[str] = []
        reasoning: list[str] = []
        usage: dict[str, Any] = {}
        finish_reason: str | None = None
        output_chunks = 0
        saw_done = False
        started = time.monotonic()
        last_emitted = started

        def emit(*, exact: bool = False) -> None:
            nonlocal last_emitted
            now = time.monotonic()
            output_tokens = (
                usage.get("completion_tokens") if exact else output_chunks
            )
            if not isinstance(output_tokens, int) or isinstance(output_tokens, bool):
                output_tokens = output_chunks
                exact = False
            generation_seconds = (
                usage.get("generation_duration") if exact else None
            )
            if not isinstance(generation_seconds, (int, float)):
                generation_seconds = max(0.0, now - started)
            tokens_per_second = (
                usage.get("generation_tokens_per_second") if exact else None
            )
            if not isinstance(tokens_per_second, (int, float)):
                tokens_per_second = (
                    output_tokens / generation_seconds if generation_seconds > 0 else 0.0
                )
            event: dict[str, Any] = {
                "output_tokens": output_tokens,
                "generation_seconds": round(float(generation_seconds), 3),
                "tokens_per_second": round(float(tokens_per_second), 3),
                "token_count_exact": exact,
            }
            if isinstance(max_output_tokens, int) and not isinstance(
                max_output_tokens, bool
            ):
                event["max_output_tokens"] = max_output_tokens
            try:
                progress_callback(event)
            except Exception:
                pass
            last_emitted = now

        with httpx.Client(base_url=self.base_url, transport=self._transport) as client:
            with client.stream(
                "POST",
                path,
                json=request_payload,
                headers={
                    "Content-Type": "application/json",
                    _OMLX_AUTH_HEADER: OMLX_API_KEY,
                },
                timeout=None if timeout_ms is None else timeout_ms / 1000,
            ) as response:
                if response.status_code >= 300:
                    response.read()
                    self._json_body(response)
                for line in response.iter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        saw_done = True
                        break
                    try:
                        event = json.loads(data)
                    except (TypeError, json.JSONDecodeError):
                        raise SafeBackendError(
                            "invalid_response", transient=False
                        ) from None
                    if not isinstance(event, dict):
                        raise SafeBackendError(
                            "invalid_response", transient=False
                        ) from None
                    if "error" in event:
                        raise SafeBackendError("http_5xx", transient=True)
                    event_usage = event.get("usage")
                    if isinstance(event_usage, dict):
                        usage.update(event_usage)
                    choices = event.get("choices")
                    choice = (
                        choices[0] if isinstance(choices, list) and choices else None
                    )
                    if isinstance(choice, dict):
                        delta = choice.get("delta")
                        emitted_text = False
                        if isinstance(delta, dict):
                            value = delta.get("content")
                            if isinstance(value, str) and value:
                                content.append(value)
                                emitted_text = True
                            value = delta.get("reasoning_content")
                            if isinstance(value, str) and value:
                                reasoning.append(value)
                                emitted_text = True
                        if emitted_text:
                            output_chunks += 1
                        value = choice.get("finish_reason")
                        if isinstance(value, str) and value:
                            finish_reason = value
                    now = time.monotonic()
                    if output_chunks and now - last_emitted >= 0.5:
                        emit()

        exact = isinstance(usage.get("completion_tokens"), int)
        if output_chunks or exact:
            emit(exact=exact)
        message: dict[str, Any] = {
            "role": "assistant",
            "content": "".join(content),
        }
        if reasoning:
            message["reasoning_content"] = "".join(reasoning)
        normalized = httpx.Response(
            200,
            json={
                "model": str(payload.get("model") or ""),
                "choices": [
                    {"index": 0, "message": message, "finish_reason": finish_reason}
                ],
                "usage": usage,
            },
        )
        return normalized, bool(saw_done and finish_reason)

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
        streamed: bool = False,
        stream_complete: bool | None = None,
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
        usage = payload.get("usage")
        if isinstance(usage, dict):
            metadata["total_time"] = usage.get("total_time")
            metadata["model_load_duration"] = usage.get("model_load_duration")
        if streamed:
            metadata["streamed"] = True
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
            completed=(
                bool(stream_complete)
                if streamed
                else bool(finish_reason is not None) or bool(content)
            ),
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
                    generation=True,
                    embedding=True,
                    structured_output=True,
                    streaming=True,
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
