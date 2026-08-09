"""Ollama components for the provider-neutral LLM runtime."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import AbstractContextManager
from typing import Any

from chronovisor.core import ollama
from chronovisor.core.llm_runtime import (
    EmbeddingRequest,
    EmbeddingResult,
    EmbeddingRoute,
    GenerationInput,
    GenerationResult,
    GenerationRoute,
    LLMRuntime,
    MessageGenerationRequest,
    TokenUsage,
)


class OllamaAdapter:
    """Compose generation, embedding, and local control on the shared facade."""

    provider = "ollama"

    def generate(self, request: GenerationInput, *, model: str) -> GenerationResult:
        if isinstance(request, MessageGenerationRequest):
            return self._chat(request, model=model)
        format_value: dict[str, Any] | str | None = (
            dict(request.format)
            if isinstance(request.format, Mapping)
            else request.format
        )
        output = ollama.generate(
            request.prompt,
            request.system,
            format=format_value,
            progress_callback=request.progress_callback,
            model=model,
            num_ctx=request.num_ctx,
            num_predict=request.max_output_tokens,
            keep_alive=request.keep_alive,
            read_timeout_ms=request.timeout_ms,
            temperature=request.temperature,
            seed=request.seed,
            return_metadata=True,
        )
        return self._normalize_generation(output, model=model)

    def _chat(
        self, request: MessageGenerationRequest, *, model: str
    ) -> GenerationResult:
        output = ollama.chat(
            [dict(message) for message in request.messages],
            model=model,
            format=dict(request.format),
            num_ctx=request.num_ctx,
            num_predict=request.max_output_tokens,
            keep_alive=request.keep_alive,
            read_timeout_ms=request.timeout_ms,
            max_output_chars=request.max_output_chars,
            temperature=request.temperature,
            seed=request.seed,
            think=request.think,
            return_metadata=True,
        )
        return self._normalize_generation(output, model=model)

    def embed(self, request: EmbeddingRequest, *, model: str) -> EmbeddingResult:
        vectors = ollama.embed(
            list(request.texts), model=model, read_timeout_ms=request.timeout_ms
        )
        return EmbeddingResult(
            vectors=tuple(tuple(vector) for vector in vectors),
            provider=self.provider,
            model=model,
        )

    def resource_lease(
        self, *, exclusive: bool, timeout_ms: int | None = None
    ) -> AbstractContextManager[None]:
        return ollama.model_resource_lease(exclusive=exclusive, timeout_ms=timeout_ms)

    def resident_models(self) -> Mapping[str, tuple[int, int]]:
        return ollama.resident_model_rows()

    def unload(self, model: str, *, verify_timeout: float = 30.0) -> bool:
        return ollama.unload_named_model(model, verify_timeout=verify_timeout)

    def _normalize_generation(
        self,
        output: str | ollama.ChatResponse | ollama.GenerateResponse,
        *,
        model: str,
    ) -> GenerationResult:
        if isinstance(output, str):
            return GenerationResult(
                content=output,
                provider=self.provider,
                model=model,
            )
        metadata: dict[str, Any] = {}
        if isinstance(output, ollama.GenerateResponse):
            metadata["streamed"] = output.streamed
        return GenerationResult(
            content=output.content,
            provider=self.provider,
            model=model,
            completed=output.done,
            finish_reason=output.done_reason,
            usage=TokenUsage(
                input_tokens=output.prompt_eval_count,
                output_tokens=output.eval_count,
            ),
            metadata=metadata,
        )


def compose_ollama_runtime(
    *,
    generation_roles: Mapping[str, str] | None = None,
    embedding_roles: Mapping[str, str] | None = None,
) -> LLMRuntime:
    """Build exact local role routes over one shared Ollama adapter."""

    adapter = OllamaAdapter()
    generation_roles = generation_roles or {}
    embedding_roles = embedding_roles or {}
    local_roles = generation_roles.keys() | embedding_roles.keys()
    return LLMRuntime(
        generation={
            role: GenerationRoute(adapter, model)
            for role, model in generation_roles.items()
        },
        embedding={
            role: EmbeddingRoute(adapter, model)
            for role, model in embedding_roles.items()
        },
        local_controls={role: adapter for role in local_roles},
    )
