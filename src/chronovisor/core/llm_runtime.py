"""Provider-neutral model runtime contracts and capability routing."""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Iterable, Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, NoReturn, Protocol, TypeVar

from chronovisor.core.llm_security import MAX_REQUEST_TIMEOUT_MS

MAX_CONTEXT_TOKENS = 1_048_576
MAX_OUTPUT_TOKENS = 131_072
MAX_OUTPUT_CHARS = 16_000_000
MAX_RETRIES = 5
SAFE_FAILURE_CATEGORIES = frozenset(
    {
        "backend_error",
        "backend_contract_error",
        "request_invalid",
        "capability_unavailable",
        "source_classification_required",
        "egress_denied",
        "vote_invalid",
        "route_configuration_invalid",
        "profile_invalid",
        "invalid_request",
        "http_401",
        "http_429",
        "http_5xx",
        "http_error",
        "redirect_rejected",
        "timeout",
        "transport_error",
        "invalid_response",
        "credential_ref_invalid",
        "credential_missing",
        "backend_rejected",
        "store_locked",
        "store_unavailable",
        "mounted_file_rejected",
        "endpoint_rejected",
        "origin_mismatch",
    }
)
TRANSIENT_FAILURE_CATEGORIES = frozenset(
    {"http_429", "http_5xx", "timeout", "transport_error"}
)
_SAFE_METADATA_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}\Z")


def safe_metadata_identifier(value: object) -> str | None:
    return (
        value
        if isinstance(value, str)
        and _SAFE_METADATA_IDENTIFIER.fullmatch(value) is not None
        else None
    )


class LLMRuntimeError(RuntimeError):
    """Base class for safe, provider-neutral runtime failures."""

    category = "runtime_error"


class SafeBackendError(RuntimeError):
    """Adapter failure marker containing only allowlisted runtime metadata."""

    def __init__(
        self,
        category: str,
        *,
        request_id: str | None = None,
        transient: bool = False,
    ) -> None:
        self.safe_category = (
            category if category in SAFE_FAILURE_CATEGORIES else "backend_error"
        )
        self.request_id = safe_metadata_identifier(request_id)
        self.transient = (
            transient and self.safe_category in TRANSIENT_FAILURE_CATEGORIES
        )
        super().__init__(self.safe_category)


class CapabilityUnavailableError(LLMRuntimeError):
    category = "capability_unavailable"

    def __init__(self, role: str, capability: str) -> None:
        self.role = role
        self.capability = capability
        super().__init__(f"{capability} is not configured for role {role!r}")


class BackendExecutionError(LLMRuntimeError):
    category = "backend_error"

    def __init__(
        self,
        role: str,
        capability: str,
        provider: str,
        *,
        category: str = "backend_error",
        retry_count: int = 0,
        request_id: str | None = None,
    ) -> None:
        self.role = role
        self.capability = capability
        self.provider = provider
        self.category = (
            category if category in SAFE_FAILURE_CATEGORIES else "backend_error"
        )
        self.retry_count = retry_count
        self.request_id = safe_metadata_identifier(request_id)
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


class RequestValidationError(LLMRuntimeError):
    category = "request_invalid"

    def __init__(self, role: str, capability: str, field_name: str) -> None:
        self.role = role
        self.capability = capability
        self.field_name = field_name
        super().__init__(f"invalid {field_name} for {capability} role {role!r}")


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


class EmbeddingPurpose(StrEnum):
    DOCUMENT = "document"
    QUERY = "query"


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
    retry_count: int = 0
    request_id: str | None = None

    def __post_init__(self) -> None:
        if not (
            self.category in SAFE_FAILURE_CATEGORIES
            and safe_metadata_identifier(self.role) is not None
            and self.capability in {"generation", "embedding", "rerank"}
            and safe_metadata_identifier(self.provider) is not None
            and self.location in {"local", "remote", "invalid"}
            and isinstance(self.retry_count, int)
            and not isinstance(self.retry_count, bool)
            and 0 <= self.retry_count <= MAX_RETRIES
            and (
                self.request_id is None
                or safe_metadata_identifier(self.request_id) == self.request_id
            )
        ):
            raise ValueError("unsafe runtime failure telemetry")


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
    format: Mapping[str, Any] | None = field(repr=False)
    source: SourceDataClassification
    num_ctx: int
    max_output_tokens: int
    keep_alive: str
    timeout_ms: int
    max_output_chars: int
    temperature: int | float = 0
    seed: int = 0
    think: bool | str = False
    progress_callback: Callable[[dict[str, Any]], None] | None = field(
        default=None, repr=False
    )


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
    purpose: EmbeddingPurpose = EmbeddingPurpose.DOCUMENT


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
    candidate_sources: tuple[SourceDataClassification, ...] | None = None


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
    capabilities: BackendCapabilities = field(
        default_factory=lambda: BackendCapabilities(
            generation=True,
            embedding=False,
            structured_output=False,
        )
    )
    protocol: str = "unknown"
    endpoint_sha256: str | None = None
    revision: str | None = None


@dataclass(frozen=True)
class ResolvedGenerationRoute:
    role: str
    provider: str
    model: str
    location: RouteLocation
    capabilities: BackendCapabilities
    protocol: str
    endpoint_sha256: str | None
    revision: str | None


@dataclass(frozen=True)
class ResolvedEmbeddingRoute:
    role: str
    provider: str
    model: str
    location: RouteLocation


@dataclass(frozen=True)
class ResolvedRerankRoute:
    role: str
    provider: str
    model: str
    location: RouteLocation


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


def _valid_budget(value: object, maximum: int, *, optional: bool) -> bool:
    return (optional and value is None) or (
        isinstance(value, int) and not isinstance(value, bool) and 0 < value <= maximum
    )


def _result_request_id(result: object) -> str | None:
    metadata = getattr(result, "metadata", None)
    return (
        safe_metadata_identifier(metadata.get("request_id"))
        if isinstance(metadata, Mapping)
        else None
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
        max_retries: int = 0,
        telemetry: Callable[[RuntimeFailureTelemetry], None] | None = None,
    ) -> None:
        if (
            isinstance(max_retries, bool)
            or not isinstance(max_retries, int)
            or not 0 <= max_retries <= MAX_RETRIES
        ):
            raise ValueError(f"max_retries must be between 0 and {MAX_RETRIES}")
        self._generation = dict(generation or {})
        self._embedding = dict(embedding or {})
        self._rerank = dict(rerank or {})
        self._local_controls = dict(local_controls or {})
        self._remote_egress_opt_ins = frozenset(remote_egress_opt_ins)
        self._max_retries = max_retries
        self._telemetry = telemetry

    def generate(self, role: str, request: GenerationInput) -> GenerationResult:
        route = _resolve(self._generation, role, "generation")
        self._validate_request(
            role, "generation", route.backend.provider, route.backend.location, request
        )
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
        return self._validate_contract(
            lambda: self._validate_generation(role, route, result),
            role=role,
            capability="generation",
            provider=route.backend.provider,
            location=location,
            request_id=_result_request_id(result),
        )

    def generation_location(self, role: str) -> RouteLocation:
        """Return the configured generation location without invoking a backend."""

        return self.resolve_generation(role).location

    def resolve_generation(self, role: str) -> ResolvedGenerationRoute:
        """Return one immutable configured route without invoking its backend."""

        route = _resolve(self._generation, role, "generation")
        if not isinstance(route.backend.location, RouteLocation):
            raise RouteConfigurationError(role, "generation")
        return ResolvedGenerationRoute(
            role=role,
            provider=route.backend.provider,
            model=route.model,
            location=route.backend.location,
            capabilities=route.capabilities,
            protocol=route.protocol,
            endpoint_sha256=route.endpoint_sha256,
            revision=route.revision,
        )

    def embed(self, role: str, request: EmbeddingRequest) -> EmbeddingResult:
        route = _resolve(self._embedding, role, "embedding")
        self._validate_request(
            role, "embedding", route.backend.provider, route.backend.location, request
        )
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
        return self._validate_contract(
            lambda: self._validate_embedding(role, route, request, result),
            role=role,
            capability="embedding",
            provider=route.backend.provider,
            location=location,
        )

    def resolve_embedding(self, role: str) -> ResolvedEmbeddingRoute:
        """Return one immutable configured embedding route without invoking it."""

        route = _resolve(self._embedding, role, "embedding")
        if not isinstance(route.backend.location, RouteLocation):
            raise RouteConfigurationError(role, "embedding")
        return ResolvedEmbeddingRoute(
            role=role,
            provider=route.backend.provider,
            model=route.model,
            location=route.backend.location,
        )

    def release_embedding(self, role: str) -> None:
        """Release an optional local embedding model without exposing its backend."""

        route = _resolve(self._embedding, role, "embedding")
        if route.backend.location is not RouteLocation.LOCAL:
            return
        close = getattr(route.backend, "close", None)
        if callable(close):
            close()

    def rerank(self, role: str, request: RerankRequest) -> RerankResult:
        route = _resolve(self._rerank, role, "rerank")
        self._validate_request(
            role, "rerank", route.backend.provider, route.backend.location, request
        )
        location = self._preflight(
            role=role,
            capability="rerank",
            provider=route.backend.provider,
            location=route.backend.location,
            source=request.source,
        )
        for source in request.candidate_sources or ():
            self._preflight(
                role=role,
                capability="rerank",
                provider=route.backend.provider,
                location=route.backend.location,
                source=source,
            )
        result = self._invoke(
            lambda: route.backend.rerank(request, model=route.model),
            role=role,
            capability="rerank",
            provider=route.backend.provider,
            location=location,
        )
        return self._validate_contract(
            lambda: self._validate_rerank(role, route, request, result),
            role=role,
            capability="rerank",
            provider=route.backend.provider,
            location=location,
            request_id=_result_request_id(result),
        )

    def resolve_rerank(self, role: str) -> ResolvedRerankRoute:
        """Return one immutable configured rerank route without invoking it."""

        route = _resolve(self._rerank, role, "rerank")
        if not isinstance(route.backend.location, RouteLocation):
            raise RouteConfigurationError(role, "rerank")
        return ResolvedRerankRoute(
            role=role,
            provider=route.backend.provider,
            model=route.model,
            location=route.backend.location,
        )

    def _local_control_for(self, role: str) -> LocalRuntimeControl | None:
        """Internal operational hook; application model calls never receive it."""

        return self._local_controls.get(role)

    def _validate_request(
        self,
        role: str,
        capability: str,
        provider: str,
        location: object,
        request: object,
    ) -> None:
        budgets: tuple[tuple[str, object, int, bool], ...]
        if isinstance(request, GenerationRequest):
            budgets = (
                ("num_ctx", request.num_ctx, MAX_CONTEXT_TOKENS, True),
                (
                    "max_output_tokens",
                    request.max_output_tokens,
                    MAX_OUTPUT_TOKENS,
                    True,
                ),
                ("timeout_ms", request.timeout_ms, MAX_REQUEST_TIMEOUT_MS, True),
            )
        elif isinstance(request, MessageGenerationRequest):
            budgets = (
                ("num_ctx", request.num_ctx, MAX_CONTEXT_TOKENS, False),
                (
                    "max_output_tokens",
                    request.max_output_tokens,
                    MAX_OUTPUT_TOKENS,
                    False,
                ),
                ("timeout_ms", request.timeout_ms, MAX_REQUEST_TIMEOUT_MS, False),
                (
                    "max_output_chars",
                    request.max_output_chars,
                    MAX_OUTPUT_CHARS,
                    False,
                ),
            )
        elif isinstance(request, (EmbeddingRequest, RerankRequest)):
            budgets = (
                ("timeout_ms", request.timeout_ms, MAX_REQUEST_TIMEOUT_MS, True),
            )
        else:
            budgets = (("request", None, 0, False),)
        for field_name, value, maximum, optional in budgets:
            if not _valid_budget(value, maximum, optional=optional):
                self._reject_request(
                    role, capability, provider, location, field_name
                )
        if isinstance(request, RerankRequest):
            if not isinstance(request.query, str) or not request.query.strip():
                self._reject_request(role, capability, provider, location, "query")
            if (
                not isinstance(request.candidates, tuple)
                or not request.candidates
                or not all(
                    isinstance(candidate, str) and candidate.strip()
                    for candidate in request.candidates
                )
            ):
                self._reject_request(
                    role, capability, provider, location, "candidates"
                )
            sources = request.candidate_sources
            if sources is None:
                if location is not RouteLocation.LOCAL:
                    self._reject_request(
                        role,
                        capability,
                        provider,
                        location,
                        "candidate_sources",
                    )
            elif (
                not isinstance(sources, tuple)
                or len(sources) != len(request.candidates)
                or not all(
                    isinstance(source, SourceDataClassification)
                    and isinstance(source.data_class, SourceDataClass)
                    and isinstance(source.sensitivity, SourceSensitivity)
                    for source in sources
                )
            ):
                self._reject_request(
                    role,
                    capability,
                    provider,
                    location,
                    "candidate_sources",
                )
        if isinstance(request, EmbeddingRequest) and not isinstance(
            request.purpose, EmbeddingPurpose
        ):
            self._reject_request(role, capability, provider, location, "purpose")

    def _reject_request(
        self,
        role: str,
        capability: str,
        provider: str,
        location: object,
        field_name: str,
    ) -> NoReturn:
        self._emit_failure(
            RequestValidationError.category,
            role=role,
            capability=capability,
            provider=provider,
            location=location if isinstance(location, RouteLocation) else None,
        )
        raise RequestValidationError(role, capability, field_name)

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
        retry_count = 0
        while True:
            try:
                return operation()
            except SafeBackendError as exc:
                category = exc.safe_category
                request_id = exc.request_id
                transient = exc.transient
            except Exception:
                category = "backend_error"
                request_id = None
                transient = False
            if transient and retry_count < self._max_retries:
                retry_count += 1
                continue
            self._emit_failure(
                category,
                role=role,
                capability=capability,
                provider=provider,
                location=location,
                retry_count=retry_count,
                request_id=request_id,
            )
            raise BackendExecutionError(
                role,
                capability,
                provider,
                category=category,
                retry_count=retry_count,
                request_id=request_id,
            )

    def _validate_contract(
        self,
        validation: Callable[[], Result],
        *,
        role: str,
        capability: str,
        provider: str,
        location: RouteLocation,
        request_id: str | None = None,
    ) -> Result:
        try:
            return validation()
        except BackendContractError as exc:
            reason = exc.reason
        except Exception:
            reason = "invalid result"
        self._emit_failure(
            BackendContractError.category,
            role=role,
            capability=capability,
            provider=provider,
            location=location,
            request_id=request_id,
        )
        raise BackendContractError(role, capability, reason)

    def _emit_failure(
        self,
        category: str,
        *,
        role: str,
        capability: str,
        provider: str,
        location: RouteLocation | None,
        retry_count: int = 0,
        request_id: str | None = None,
    ) -> None:
        if self._telemetry is None:
            return
        try:
            event = RuntimeFailureTelemetry(
                category=category,
                role=role,
                capability=capability,
                provider=provider,
                location=location.value if location is not None else "invalid",
                retry_count=retry_count,
                request_id=request_id,
            )
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

    @staticmethod
    def _validate_embedding(
        role: str,
        route: EmbeddingRoute,
        request: EmbeddingRequest,
        result: EmbeddingResult,
    ) -> EmbeddingResult:
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

    @staticmethod
    def _validate_rerank(
        role: str,
        route: RerankRoute,
        request: RerankRequest,
        result: RerankResult,
    ) -> RerankResult:
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
