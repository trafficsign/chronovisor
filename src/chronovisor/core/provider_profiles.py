"""Immutable remote-provider profiles and shared safe HTTP normalization."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from enum import StrEnum
from types import MappingProxyType
from typing import Literal, cast
from urllib.parse import urlsplit
from urllib.request import Request

import httpx

from chronovisor.core.llm_runtime import (
    TRANSIENT_FAILURE_CATEGORIES,
    BackendCapabilities,
    SafeBackendError,
    safe_metadata_identifier,
)
from chronovisor.core.llm_security import (
    DEFAULT_REQUEST_TIMEOUT_MS,
    AuthenticatedTransport,
    AuthScheme,
    CredentialBinding,
    CredentialRef,
    CredentialResolver,
    CredentialSecurityError,
    RequestSender,
    canonical_endpoint,
)


class ProviderProtocol(StrEnum):
    OPENAI_COMPATIBLE = "openai-compatible"
    ANTHROPIC_MESSAGES = "anthropic-messages"


class ProviderFailureCategory(StrEnum):
    PROFILE_INVALID = "profile_invalid"
    CAPABILITY_UNAVAILABLE = "capability_unavailable"
    INVALID_REQUEST = "invalid_request"
    UNAUTHORIZED = "http_401"
    RATE_LIMITED = "http_429"
    SERVER_ERROR = "http_5xx"
    HTTP_ERROR = "http_error"
    REDIRECT_REJECTED = "redirect_rejected"
    TIMEOUT = "timeout"
    TRANSPORT_ERROR = "transport_error"
    INVALID_RESPONSE = "invalid_response"
    CREDENTIAL_REF_INVALID = "credential_ref_invalid"
    CREDENTIAL_MISSING = "credential_missing"
    BACKEND_REJECTED = "backend_rejected"
    STORE_LOCKED = "store_locked"
    STORE_UNAVAILABLE = "store_unavailable"
    MOUNT_REJECTED = "mounted_file_rejected"
    ENDPOINT_REJECTED = "endpoint_rejected"
    ORIGIN_MISMATCH = "origin_mismatch"


class ProviderAdapterError(SafeBackendError):
    """A body-, prompt-, and credential-free remote provider failure."""

    def __init__(
        self,
        category: ProviderFailureCategory,
        *,
        request_id: str | None = None,
    ) -> None:
        self.category = category
        super().__init__(
            category.value,
            request_id=request_id,
            transient=category.value in TRANSIENT_FAILURE_CATEGORIES,
        )


@dataclass(frozen=True)
class ProviderProfile:
    profile_id: str
    protocol: ProviderProtocol
    endpoint: str
    credential_ref: CredentialRef
    auth_scheme: AuthScheme
    capabilities: BackendCapabilities
    structured_output_models: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        valid = (
            isinstance(self.protocol, ProviderProtocol)
            and isinstance(self.credential_ref, CredentialRef)
            and isinstance(self.auth_scheme, AuthScheme)
            and isinstance(self.capabilities, BackendCapabilities)
            and isinstance(self.structured_output_models, frozenset)
            and self.capabilities.generation
            and not self.capabilities.structured_output
            and not self.capabilities.streaming
            and not self.capabilities.tools
            and not self.capabilities.rerank
            and all(
                isinstance(value, bool)
                for value in (
                    self.capabilities.generation,
                    self.capabilities.embedding,
                    self.capabilities.structured_output,
                    self.capabilities.streaming,
                    self.capabilities.tools,
                    self.capabilities.rerank,
                )
            )
            and all(
                isinstance(model, str)
                and bool(model)
                and model == model.strip()
                and not any(ord(character) < 32 for character in model)
                for model in self.structured_output_models
            )
        )
        try:
            canonical = canonical_endpoint(self.endpoint, cloud_secret=True)
            binding = CredentialBinding.bind(
                self.profile_id, canonical.origin, self.auth_scheme
            )
        except (CredentialSecurityError, TypeError):
            pass
        else:
            valid = (
                valid
                and not urlsplit(canonical.url).query
                and binding.profile_id == self.profile_id
                and (
                    self.protocol is ProviderProtocol.OPENAI_COMPATIBLE
                    or not self.capabilities.embedding
                )
            )
            if valid:
                object.__setattr__(self, "endpoint", canonical.url.rstrip("/"))
                object.__setattr__(
                    self,
                    "structured_output_models",
                    frozenset(self.structured_output_models),
                )
                return
        raise ProviderAdapterError(ProviderFailureCategory.PROFILE_INVALID)

    def binding(self) -> CredentialBinding:
        return CredentialBinding.bind(self.profile_id, self.endpoint, self.auth_scheme)

    def url(self, path: str) -> str:
        if not isinstance(path, str) or not path.startswith("/"):
            raise ProviderAdapterError(ProviderFailureCategory.PROFILE_INVALID)
        return f"{self.endpoint}{path}"

    def capabilities_for(self, model: str) -> BackendCapabilities:
        return replace(
            self.capabilities,
            structured_output=model in self.structured_output_models,
        )


_CURATED_SPECS = MappingProxyType(
    {
        "openai": (
            ProviderProtocol.OPENAI_COMPATIBLE,
            "https://api.openai.com/v1",
            AuthScheme.BEARER,
            True,
        ),
        "qwen": (
            ProviderProtocol.OPENAI_COMPATIBLE,
            None,
            AuthScheme.BEARER,
            True,
        ),
        "dashscope": (
            ProviderProtocol.OPENAI_COMPATIBLE,
            None,
            AuthScheme.BEARER,
            True,
        ),
        "gemini": (
            ProviderProtocol.OPENAI_COMPATIBLE,
            "https://generativelanguage.googleapis.com/v1beta/openai",
            AuthScheme.BEARER,
            True,
        ),
        "deepseek": (
            ProviderProtocol.OPENAI_COMPATIBLE,
            "https://api.deepseek.com",
            AuthScheme.BEARER,
            False,
        ),
        "kimi": (
            ProviderProtocol.OPENAI_COMPATIBLE,
            "https://api.moonshot.ai/v1",
            AuthScheme.BEARER,
            False,
        ),
        "zai": (
            ProviderProtocol.OPENAI_COMPATIBLE,
            "https://api.z.ai/api/paas/v4",
            AuthScheme.BEARER,
            False,
        ),
        "glm": (
            ProviderProtocol.OPENAI_COMPATIBLE,
            "https://api.z.ai/api/paas/v4",
            AuthScheme.BEARER,
            False,
        ),
        "mistral": (
            ProviderProtocol.OPENAI_COMPATIBLE,
            "https://api.mistral.ai/v1",
            AuthScheme.BEARER,
            True,
        ),
        "openrouter": (
            ProviderProtocol.OPENAI_COMPATIBLE,
            "https://openrouter.ai/api/v1",
            AuthScheme.BEARER,
            True,
        ),
        "anthropic": (
            ProviderProtocol.ANTHROPIC_MESSAGES,
            "https://api.anthropic.com/v1",
            AuthScheme.X_API_KEY,
            False,
        ),
    }
)
_CURATED_STRUCTURED_OUTPUT_MODELS = MappingProxyType(
    {"openai": frozenset({"gpt-5", "gpt-5-mini", "gpt-5-nano"})}
)
CURATED_PROFILE_IDS = tuple(_CURATED_SPECS)


def curated_profile(
    profile_id: str,
    credential_ref: CredentialRef,
    *,
    endpoint_override: str | None = None,
) -> ProviderProfile:
    spec = _CURATED_SPECS.get(profile_id)
    if spec is None:
        raise ProviderAdapterError(ProviderFailureCategory.PROFILE_INVALID)
    protocol, default_endpoint, auth_scheme, embedding = spec
    endpoint = endpoint_override or default_endpoint
    if endpoint is None:
        raise ProviderAdapterError(ProviderFailureCategory.PROFILE_INVALID)
    return ProviderProfile(
        profile_id=profile_id,
        protocol=protocol,
        endpoint=endpoint,
        credential_ref=credential_ref,
        auth_scheme=auth_scheme,
        capabilities=BackendCapabilities(generation=True, embedding=embedding),
        structured_output_models=_CURATED_STRUCTURED_OUTPUT_MODELS.get(
            profile_id, frozenset()
        ),
    )


def generic_openai_profile(
    profile_id: str,
    endpoint: str,
    credential_ref: CredentialRef,
    *,
    auth_scheme: AuthScheme = AuthScheme.BEARER,
    embedding: bool = False,
    structured_output_models: Iterable[str] = (),
) -> ProviderProfile:
    return ProviderProfile(
        profile_id=profile_id,
        protocol=ProviderProtocol.OPENAI_COMPATIBLE,
        endpoint=endpoint,
        credential_ref=credential_ref,
        auth_scheme=auth_scheme,
        capabilities=BackendCapabilities(generation=True, embedding=embedding),
        structured_output_models=frozenset(structured_output_models),
    )


class HTTPXSender:
    """Real HTTPS sender with certificate verification on and redirects off."""

    def __call__(
        self,
        request: Request,
        *,
        follow_redirects: Literal[False],
        timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_MS / 1000,
    ) -> object:
        if follow_redirects is not False:
            raise RuntimeError("redirect policy violation")
        content = request.data
        if content is not None and not isinstance(content, bytes):
            raise RuntimeError("invalid request body")
        return httpx.request(
            request.get_method(),
            request.full_url,
            content=content,
            headers=dict(request.header_items()),
            timeout=timeout_seconds,
            follow_redirects=False,
        )


@dataclass(frozen=True)
class _TransportFailure:
    category: ProviderFailureCategory


class _TimeoutAwareSender:
    def __init__(self, sender: RequestSender) -> None:
        self._sender = sender

    def __call__(
        self,
        request: Request,
        *,
        follow_redirects: Literal[False],
        timeout_seconds: float,
    ) -> object:
        try:
            return self._sender(
                request,
                follow_redirects=follow_redirects,
                timeout_seconds=timeout_seconds,
            )
        except httpx.TimeoutException:
            return _TransportFailure(ProviderFailureCategory.TIMEOUT)


def authenticated_transport(
    profile: ProviderProfile,
    resolver: CredentialResolver,
    *,
    sender: RequestSender | None = None,
) -> AuthenticatedTransport:
    secret = resolver.resolve(profile.credential_ref)
    return AuthenticatedTransport(
        profile_id=profile.profile_id,
        endpoint=profile.endpoint,
        secret=secret,
        binding=profile.binding(),
        auth_scheme=profile.auth_scheme,
        sender=_TimeoutAwareSender(sender if sender is not None else HTTPXSender()),
    )


@dataclass(frozen=True)
class ProviderJSONResponse:
    payload: Mapping[str, object] = field(repr=False)
    request_id: str | None = None


def response_metadata(
    payload: Mapping[str, object], request_id: str | None
) -> Mapping[str, str]:
    metadata: dict[str, str] = {}
    returned_model = safe_metadata_identifier(payload.get("model"))
    if returned_model is not None:
        metadata["returned_model"] = returned_model
    if request_id is not None:
        metadata["request_id"] = request_id
    return metadata


def safe_finish_reason(
    value: object,
    *,
    allowed: frozenset[str],
) -> str | None:
    if value is None:
        return None
    return value if isinstance(value, str) and value in allowed else None


def post_json(
    transport: AuthenticatedTransport,
    url: str,
    payload: Mapping[str, object],
    *,
    headers: Mapping[str, str] | None = None,
    timeout_ms: int | None = None,
) -> ProviderJSONResponse:
    try:
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
    except Exception:
        pass
    else:
        try:
            response = transport.send(
                url,
                data=encoded,
                headers={"Content-Type": "application/json", **(headers or {})},
                timeout_ms=(
                    DEFAULT_REQUEST_TIMEOUT_MS if timeout_ms is None else timeout_ms
                ),
            )
        except CredentialSecurityError as exc:
            security_category = exc.category.value
        else:
            if isinstance(response, _TransportFailure):
                raise ProviderAdapterError(response.category)
            if not isinstance(response, httpx.Response):
                raise ProviderAdapterError(ProviderFailureCategory.INVALID_RESPONSE)
            request_id = safe_metadata_identifier(
                response.headers.get("x-request-id")
                or response.headers.get("request-id")
            )
            status = response.status_code
            if status == 401:
                raise ProviderAdapterError(
                    ProviderFailureCategory.UNAUTHORIZED, request_id=request_id
                )
            if status == 429:
                raise ProviderAdapterError(
                    ProviderFailureCategory.RATE_LIMITED, request_id=request_id
                )
            if status >= 500:
                raise ProviderAdapterError(
                    ProviderFailureCategory.SERVER_ERROR, request_id=request_id
                )
            if 300 <= status < 400:
                raise ProviderAdapterError(
                    ProviderFailureCategory.REDIRECT_REJECTED, request_id=request_id
                )
            if status != 200:
                raise ProviderAdapterError(
                    ProviderFailureCategory.HTTP_ERROR, request_id=request_id
                )
            try:
                decoded = response.json()
            except Exception:
                pass
            else:
                if isinstance(decoded, dict) and all(
                    isinstance(key, str) for key in decoded
                ):
                    return ProviderJSONResponse(
                        cast(Mapping[str, object], decoded), request_id
                    )
            raise ProviderAdapterError(
                ProviderFailureCategory.INVALID_RESPONSE, request_id=request_id
            )
        try:
            category = ProviderFailureCategory(security_category)
        except ValueError:
            category = ProviderFailureCategory.TRANSPORT_ERROR
        raise ProviderAdapterError(category)
    raise ProviderAdapterError(ProviderFailureCategory.INVALID_REQUEST)
