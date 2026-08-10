"""Provider-neutral model runtime contracts and capability routing."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, ClassVar, Protocol, TypeVar


class LLMRuntimeError(RuntimeError):
    """Base class for safe, provider-neutral runtime failures."""

    category: ClassVar[str] = "runtime_error"


class CapabilityUnavailableError(LLMRuntimeError):
    category = "capability_unavailable"

    def __init__(self, role: str, capability: str) -> None:
        self.role = role
        self.capability = capability
        super().__init__(f"{capability} is not configured for role {role!r}")


class BackendExecutionError(LLMRuntimeError):
    category = "backend_error"

    def __init__(self, role: str, capability: str, provider: str) -> None:
        self.role = role
        self.capability = capability
        self.provider = provider
        super().__init__(f"{provider} {capability} backend failed for role {role!r}")


class BackendContractError(LLMRuntimeError):
    category = "backend_contract_error"

    def __init__(self, role: str, capability: str, reason: str) -> None:
        self.role = role
        self.capability = capability
        self.reason = reason
        super().__init__(
            f"{capability} backend contract failed for role {role!r}: {reason}"
        )


class SourceClassificationError(LLMRuntimeError):
    category = "source_classification_required"

    def __init__(self, role: str, capability: str) -> None:
        self.role = role
        self.capability = capability
        super().__init__(f"valid source classification is required for {role!r}")


class EgressDeniedError(LLMRuntimeError):
    category = "egress_denied"

    def __init__(self, role: str, capability: str) -> None:
        self.role = role
        self.capability = capability
        super().__init__(f"remote egress denied for role {role!r}")


class RouteConfigurationError(LLMRuntimeError):
    category = "route_configuration_invalid"

    def __init__(self, role: str, capability: str) -> None:
        self.role = role
        self.capability = capability
        super().__init__(f"route location is invalid for role {role!r}")


class RouteLocation(StrEnum):
    LOCAL = "local"
    REMOTE = "remote"


class SourceDataClass(StrEnum):
    PAGE = "page"
    DERIVED_SNIPPET = "derived_snippet"
    RAW = "raw"
    SYSTEM = "system"


class SourceSensitivity(StrEnum):
    NORMAL = "normal"
    HIGH = "high"


@dataclass(frozen=True)
class SourceDataClassification:
    data_class: SourceDataClass
    sensitivity: SourceSensitivity


@dataclass(frozen=True)
class RuntimeFailureTelemetry:
    category: str
    role: str
    capability: str
    provider: str
    location: str


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int | None = None
    output_tokens: int | None = None


@dataclass(frozen=True)
class BackendCapabilities:
    generation: bool
    embedding: bool
    structured_output: bool = False
    streaming: bool = False
    tools: bool = False
    rerank: bool = False


@dataclass(frozen=True)
class GenerationRequest:
    prompt: str = field(repr=False)
    source: SourceDataClassification
    system: str | None = field(default=None, repr=False)
    format: Mapping[str, Any] | str | None = field(default=None, repr=False)
    progress_callback: Callable[[dict[str, Any]], None] | None = field(
        default=None, repr=False
    )
    num_ctx: int | None = None
    max_output_tokens: int | None = None
    keep_alive: str | None = None
    timeout_ms: int | None = None
    temperature: int | float | None = None
    seed: int | None = None


@dataclass(frozen=True)
class MessageGenerationRequest:
    messages: tuple[Mapping[str, str], ...] = field(repr=False)
    format: Mapping[str, Any] = field(repr=False)
    source: SourceDataClassification
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
    content: str = field(repr=False)
    provider: str
    model: str
    completed: bool = True
    finish_reason: str | None = None
    usage: TokenUsage = field(default_factory=TokenUsage)
    metadata: Mapping[str, Any] = field(default_factory=dict, repr=False)


@dataclass(frozen=True)
class EmbeddingRequest:
    texts: tuple[str, ...] = field(repr=False)
    source: SourceDataClassification
    timeout_ms: int | None = None


@dataclass(frozen=True)
class EmbeddingResult:
    vectors: tuple[tuple[float, ...], ...]
    provider: str
    model: str


@dataclass(frozen=True)
class RerankRequest:
    query: str = field(repr=False)
    candidates: tuple[str, ...] = field(repr=False)
    source: SourceDataClassification
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
    metadata: Mapping[str, Any] = field(default_factory=dict, repr=False)


class GenerationBackend(Protocol):
    provider: str
    location: RouteLocation

    def generate(self, request: GenerationInput, *, model: str) -> GenerationResult: ...


class EmbeddingBackend(Protocol):
    provider: str
    location: RouteLocation

    def embed(self, request: EmbeddingRequest, *, model: str) -> EmbeddingResult: ...


class RerankBackend(Protocol):
    provider: str
    location: RouteLocation

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
        remote_egress_opt_ins: Iterable[tuple[str, SourceDataClass]] = (),
        telemetry: Callable[[RuntimeFailureTelemetry], None] | None = None,
    ) -> None:
        self._generation = dict(generation or {})
        self._embedding = dict(embedding or {})
        self._rerank = dict(rerank or {})
        self._local_controls = dict(local_controls or {})
        self._remote_egress_opt_ins = frozenset(remote_egress_opt_ins)
        self._telemetry = telemetry

    def generate(self, role: str, request: GenerationInput) -> GenerationResult:
        route = _resolve(self._generation, role, "generation")
        location = self._preflight(
            role=role,
            capability="generation",
            provider=route.backend.provider,
            location=route.backend.location,
            source=request.source,
        )
        result = self._invoke(
            lambda: route.backend.generate(request, model=route.model),
            role=role,
            capability="generation",
            provider=route.backend.provider,
            location=location,
        )
        return self._validate_generation(role, route, result)

    def embed(self, role: str, request: EmbeddingRequest) -> EmbeddingResult:
        route = _resolve(self._embedding, role, "embedding")
        location = self._preflight(
            role=role,
            capability="embedding",
            provider=route.backend.provider,
            location=route.backend.location,
            source=request.source,
        )
        result = self._invoke(
            lambda: route.backend.embed(request, model=route.model),
            role=role,
            capability="embedding",
            provider=route.backend.provider,
            location=location,
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
        location = self._preflight(
            role=role,
            capability="rerank",
            provider=route.backend.provider,
            location=route.backend.location,
            source=request.source,
        )
        result = self._invoke(
            lambda: route.backend.rerank(request, model=route.model),
            role=role,
            capability="rerank",
            provider=route.backend.provider,
            location=location,
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

    def _preflight(
        self,
        *,
        role: str,
        capability: str,
        provider: str,
        location: object,
        source: object,
    ) -> RouteLocation:
        if (
            not isinstance(source, SourceDataClassification)
            or not isinstance(source.data_class, SourceDataClass)
            or not isinstance(source.sensitivity, SourceSensitivity)
        ):
            self._emit_failure(
                SourceClassificationError.category,
                role=role,
                capability=capability,
                provider=provider,
                location=location if isinstance(location, RouteLocation) else None,
            )
            raise SourceClassificationError(role, capability)
        if not isinstance(location, RouteLocation):
            self._emit_failure(
                RouteConfigurationError.category,
                role=role,
                capability=capability,
                provider=provider,
                location=None,
            )
            raise RouteConfigurationError(role, capability)
        if location is RouteLocation.LOCAL:
            return location
        default_allowed = (
            source.sensitivity is SourceSensitivity.NORMAL
            and source.data_class
            in {SourceDataClass.PAGE, SourceDataClass.DERIVED_SNIPPET}
        )
        opted_in = (role, source.data_class) in self._remote_egress_opt_ins
        if not default_allowed and not opted_in:
            self._emit_failure(
                EgressDeniedError.category,
                role=role,
                capability=capability,
                provider=provider,
                location=location,
            )
            raise EgressDeniedError(role, capability)
        return location

    def _invoke(
        self,
        operation: Callable[[], Result],
        *,
        role: str,
        capability: str,
        provider: str,
        location: RouteLocation,
    ) -> Result:
        try:
            return operation()
        except Exception:
            pass
        self._emit_failure(
            BackendExecutionError.category,
            role=role,
            capability=capability,
            provider=provider,
            location=location,
        )
        raise BackendExecutionError(role, capability, provider)

    def _emit_failure(
        self,
        category: str,
        *,
        role: str,
        capability: str,
        provider: str,
        location: RouteLocation | None,
    ) -> None:
        if self._telemetry is None:
            return
        event = RuntimeFailureTelemetry(
            category=category,
            role=role,
            capability=capability,
            provider=provider,
            location=location.value if location is not None else "invalid",
        )
        try:
            self._telemetry(event)
        except Exception:
            pass

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
