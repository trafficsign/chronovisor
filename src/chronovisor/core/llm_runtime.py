"""Provider-neutral model runtime contracts and capability routing."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar


class LLMRuntimeError(RuntimeError):
    """Base class for safe, provider-neutral runtime failures."""


class CapabilityUnavailableError(LLMRuntimeError):
    def __init__(self, role: str, capability: str) -> None:
        self.role = role
        self.capability = capability
        super().__init__(f"{capability} is not configured for role {role!r}")


class BackendExecutionError(LLMRuntimeError):
    def __init__(self, role: str, capability: str, provider: str) -> None:
        self.role = role
        self.capability = capability
        self.provider = provider
        super().__init__(f"{provider} {capability} backend failed for role {role!r}")


class BackendContractError(LLMRuntimeError):
    def __init__(self, role: str, capability: str, reason: str) -> None:
        self.role = role
        self.capability = capability
        self.reason = reason
        super().__init__(
            f"{capability} backend contract failed for role {role!r}: {reason}"
        )


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int | None = None
    output_tokens: int | None = None


@dataclass(frozen=True)
class GenerationRequest:
    prompt: str
    system: str | None = None
    format: Mapping[str, Any] | str | None = None
    progress_callback: Callable[[dict[str, Any]], None] | None = None
    num_ctx: int | None = None
    max_output_tokens: int | None = None
    keep_alive: str | None = None
    timeout_ms: int | None = None
    temperature: int | float | None = None
    seed: int | None = None


@dataclass(frozen=True)
class MessageGenerationRequest:
    messages: tuple[Mapping[str, str], ...]
    format: Mapping[str, Any]
    num_ctx: int
    max_output_tokens: int
    keep_alive: str
    timeout_ms: int
    max_output_chars: int
    temperature: int | float = 0
    seed: int = 0
    think: bool | str = False


GenerationInput = GenerationRequest | MessageGenerationRequest


@dataclass(frozen=True)
class GenerationResult:
    content: str
    provider: str
    model: str
    completed: bool = True
    finish_reason: str | None = None
    usage: TokenUsage = field(default_factory=TokenUsage)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EmbeddingRequest:
    texts: tuple[str, ...]
    timeout_ms: int | None = None


@dataclass(frozen=True)
class EmbeddingResult:
    vectors: tuple[tuple[float, ...], ...]
    provider: str
    model: str


@dataclass(frozen=True)
class RerankRequest:
    query: str
    candidates: tuple[str, ...]
    timeout_ms: int | None = None


@dataclass(frozen=True)
class RerankItem:
    index: int
    score: float


@dataclass(frozen=True)
class RerankResult:
    items: tuple[RerankItem, ...]
    provider: str
    model: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


class GenerationBackend(Protocol):
    provider: str

    def generate(self, request: GenerationInput, *, model: str) -> GenerationResult: ...


class EmbeddingBackend(Protocol):
    provider: str

    def embed(self, request: EmbeddingRequest, *, model: str) -> EmbeddingResult: ...


class RerankBackend(Protocol):
    provider: str

    def rerank(self, request: RerankRequest, *, model: str) -> RerankResult: ...


class LocalRuntimeControl(Protocol):
    def resource_lease(
        self, *, exclusive: bool, timeout_ms: int | None = None
    ) -> AbstractContextManager[None]: ...

    def resident_models(self) -> Mapping[str, tuple[int, int]]: ...

    def unload(self, model: str, *, verify_timeout: float = 30.0) -> bool: ...


@dataclass(frozen=True)
class GenerationRoute:
    backend: GenerationBackend
    model: str


@dataclass(frozen=True)
class EmbeddingRoute:
    backend: EmbeddingBackend
    model: str


@dataclass(frozen=True)
class RerankRoute:
    backend: RerankBackend
    model: str


Route = TypeVar("Route")
Result = TypeVar("Result")


def _resolve(routes: Mapping[str, Route], role: str, capability: str) -> Route:
    route = routes.get(role)
    if route is None:
        raise CapabilityUnavailableError(role, capability)
    return route


def _invoke(
    operation: Callable[[], Result],
    *,
    role: str,
    capability: str,
    provider: str,
) -> Result:
    try:
        return operation()
    except LLMRuntimeError:
        raise
    except Exception as exc:
        raise BackendExecutionError(role, capability, provider) from exc


def _valid_token_count(value: int | None) -> bool:
    return value is None or (
        isinstance(value, int) and not isinstance(value, bool) and value >= 0
    )


class LLMRuntime:
    """Resolve one exact role route per capability without fallback."""

    def __init__(
        self,
        *,
        generation: Mapping[str, GenerationRoute] | None = None,
        embedding: Mapping[str, EmbeddingRoute] | None = None,
        rerank: Mapping[str, RerankRoute] | None = None,
        local_controls: Mapping[str, LocalRuntimeControl] | None = None,
    ) -> None:
        self._generation = dict(generation or {})
        self._embedding = dict(embedding or {})
        self._rerank = dict(rerank or {})
        self._local_controls = dict(local_controls or {})

    def generate(self, role: str, request: GenerationInput) -> GenerationResult:
        route = _resolve(self._generation, role, "generation")
        result = _invoke(
            lambda: route.backend.generate(request, model=route.model),
            role=role,
            capability="generation",
            provider=route.backend.provider,
        )
        return self._validate_generation(role, route, result)

    def embed(self, role: str, request: EmbeddingRequest) -> EmbeddingResult:
        route = _resolve(self._embedding, role, "embedding")
        result = _invoke(
            lambda: route.backend.embed(request, model=route.model),
            role=role,
            capability="embedding",
            provider=route.backend.provider,
        )
        if result.provider != route.backend.provider or result.model != route.model:
            raise BackendContractError(role, "embedding", "route identity mismatch")
        if len(result.vectors) != len(request.texts):
            raise BackendContractError(role, "embedding", "vector count mismatch")
        dimensions = {len(vector) for vector in result.vectors}
        if result.vectors and (dimensions == {0} or len(dimensions) != 1):
            raise BackendContractError(role, "embedding", "invalid vector dimensions")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for vector in result.vectors
            for value in vector
        ):
            raise BackendContractError(role, "embedding", "invalid vector value")
        return result

    def rerank(self, role: str, request: RerankRequest) -> RerankResult:
        route = _resolve(self._rerank, role, "rerank")
        result = _invoke(
            lambda: route.backend.rerank(request, model=route.model),
            role=role,
            capability="rerank",
            provider=route.backend.provider,
        )
        if result.provider != route.backend.provider or result.model != route.model:
            raise BackendContractError(role, "rerank", "route identity mismatch")
        if len(result.items) != len(request.candidates) or {
            item.index for item in result.items
        } != set(range(len(request.candidates))):
            raise BackendContractError(role, "rerank", "invalid ranking indices")
        if any(
            isinstance(item.score, bool)
            or not isinstance(item.score, (int, float))
            or not math.isfinite(item.score)
            for item in result.items
        ):
            raise BackendContractError(role, "rerank", "invalid ranking score")
        return result

    def _local_control_for(self, role: str) -> LocalRuntimeControl | None:
        """Internal operational hook; application model calls never receive it."""

        return self._local_controls.get(role)

    @staticmethod
    def _validate_generation(
        role: str,
        route: GenerationRoute,
        result: GenerationResult,
    ) -> GenerationResult:
        if result.provider != route.backend.provider or result.model != route.model:
            raise BackendContractError(role, "generation", "route identity mismatch")
        if not isinstance(result.content, str) or not isinstance(
            result.completed, bool
        ):
            raise BackendContractError(role, "generation", "invalid completion")
        if result.finish_reason is not None and not isinstance(
            result.finish_reason, str
        ):
            raise BackendContractError(role, "generation", "invalid finish reason")
        if not _valid_token_count(result.usage.input_tokens) or not _valid_token_count(
            result.usage.output_tokens
        ):
            raise BackendContractError(role, "generation", "invalid token usage")
        return result
