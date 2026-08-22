"""Strict Campaign W configuration and one-shot LLMRuntime composition."""

from __future__ import annotations

import hashlib
import os
import re
import stat
import tomllib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from enum import StrEnum
from functools import cache
from pathlib import Path
from types import MappingProxyType
from typing import cast
from urllib.parse import urlsplit

from chronovisor.core import runtime_status
from chronovisor.core.anthropic_adapter import compose_anthropic_adapter
from chronovisor.core.llm_runtime import (
    BackendCapabilities,
    EmbeddingBackend,
    EmbeddingRoute,
    GenerationBackend,
    GenerationRoute,
    LLMRuntime,
    RerankBackend,
    RerankRoute,
    RuntimeFailureTelemetry,
    SourceDataClass,
)
from chronovisor.core.llm_runtime import (
    safe_metadata_identifier as safe_metadata_identifier,
)
from chronovisor.core.llm_security import (
    AuthScheme,
    CredentialBackend,
    CredentialRef,
    CredentialResolver,
    CredentialSecurityError,
    RequestSender,
)
from chronovisor.core.nemotron_adapter import NemotronEmbeddingBackend
from chronovisor.core.ollama_adapter import OllamaAdapter
from chronovisor.core.ollama_transport import OLLAMA_URL
from chronovisor.core.omlx_adapter import OMLX_BASE_URL, OMLXAdapter
from chronovisor.core.openai_compatible_adapter import compose_openai_compatible_adapter
from chronovisor.core.provider_profiles import (
    CURATED_PROFILE_IDS,
    ProviderAdapterError,
    ProviderProfile,
    ProviderProtocol,
    curated_profile,
    generic_openai_profile,
)
from chronovisor.core.reranker import LocalRerankBackend
from chronovisor.core.runtime_config import (
    CONFIG_FILE,
    RerankerConfig,
    SearchEmbeddingConfig,
    load_search_embedding_config,
)

_PROFILE_ID = re.compile(r"[a-z][a-z0-9._-]{0,63}\Z")
_ROLE_ID = re.compile(r"[a-z][a-z0-9._-]{0,127}\Z")
_CAPABILITY_FIELDS = frozenset(
    {"generation", "embedding", "rerank", "structured_output", "streaming", "tools"}
)
_RAW_SECRET_FIELDS = frozenset(
    {
        "api_key",
        "apikey",
        "access_token",
        "authorization",
        "client_secret",
        "credential",
        "password",
        "secret",
        "token",
    }
)
_INSECURE_TRANSPORT_FIELDS = frozenset(
    {
        "allow_insecure",
        "follow_redirects",
        "insecure",
        "ssl_verify",
        "tls_verify",
        "verify",
        "verify_ssl",
        "verify_tls",
    }
)


class LLMConfigFailureCategory(StrEnum):
    UNAVAILABLE = "llm_config_unavailable"
    PARSE_ERROR = "llm_config_parse_error"
    SCHEMA_INVALID = "llm_config_schema_invalid"
    CAPABILITY_UNAVAILABLE = "llm_capability_unavailable"


class LLMConfigError(RuntimeError):
    def __init__(self, category: LLMConfigFailureCategory) -> None:
        self.category = category
        super().__init__(category.value)


class RoleCapability(StrEnum):
    GENERATION = "generation"
    EMBEDDING = "embedding"
    RERANK = "rerank"


@dataclass(frozen=True)
class ProviderDefinition:
    provider_id: str
    kind: str
    capabilities: BackendCapabilities
    profile: ProviderProfile | None = None
    reranker_config: RerankerConfig | None = None
    embedding_device: str | None = None
    endpoint: str | None = None

    def capabilities_for(self, model: str) -> BackendCapabilities:
        return (
            self.profile.capabilities_for(model)
            if self.profile is not None
            else self.capabilities
        )


@dataclass(frozen=True)
class RoleDefinition:
    role: str
    capability: RoleCapability
    provider_id: str
    model: str
    required_capabilities: tuple[str, ...] = ()
    revision: str | None = None


@dataclass(frozen=True)
class LLMConfig:
    providers: Mapping[str, ProviderDefinition]
    roles: Mapping[str, RoleDefinition]
    egress_opt_ins: frozenset[tuple[str, SourceDataClass]] = frozenset()


SenderFactory = Callable[[ProviderProfile], RequestSender]


def _fail(
    category: LLMConfigFailureCategory = LLMConfigFailureCategory.SCHEMA_INVALID,
) -> LLMConfigError:
    return LLMConfigError(category)


def _exact_keys(value: Mapping[str, object], allowed: set[str]) -> None:
    if set(value) - allowed:
        raise _fail()


def _reject_credential_fields(value: object) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise _fail()
            normalized = key.lower().replace("-", "_")
            raw_secret = normalized in _RAW_SECRET_FIELDS or normalized.endswith(
                (
                    "_api_key",
                    "_access_token",
                    "_auth_token",
                    "_authorization",
                    "_client_secret",
                    "_password",
                    "_secret",
                )
            )
            if raw_secret or normalized in _INSECURE_TRANSPORT_FIELDS:
                raise _fail()
            _reject_credential_fields(child)
    elif isinstance(value, list):
        for child in value:
            _reject_credential_fields(child)


def _string(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(ord(character) < 32 for character in value)
    ):
        raise _fail()
    return value


def _positive_int(value: object, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise _fail()
    return value


def _string_list(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise _fail()
    result = tuple(_string(item) for item in value)
    if len(set(result)) != len(result):
        raise _fail()
    return result


def _credential_ref(value: object, provider_id: str) -> CredentialRef:
    try:
        ref = CredentialRef.parse(_string(value))
    except (CredentialSecurityError, LLMConfigError):
        pass
    else:
        if (
            ref.backend is not CredentialBackend.OS_KEYRING
            or ref.target.partition("/")[0] == provider_id
        ):
            return ref
    raise _fail()


def _remote_profile(
    provider_id: str,
    kind: str,
    table: Mapping[str, object],
) -> ProviderDefinition:
    if kind == ProviderProtocol.OPENAI_COMPATIBLE.value:
        _exact_keys(
            table,
            {
                "kind",
                "endpoint",
                "credential_ref",
                "auth_scheme",
            },
        )
        try:
            auth_scheme = AuthScheme.parse(_string(table.get("auth_scheme", "bearer")))
            profile = generic_openai_profile(
                provider_id,
                _string(table.get("endpoint")),
                _credential_ref(table.get("credential_ref"), provider_id),
                auth_scheme=auth_scheme,
            )
        except (CredentialSecurityError, ProviderAdapterError, LLMConfigError):
            pass
        else:
            return ProviderDefinition(
                provider_id, kind, profile.capabilities, profile=profile
            )
        raise _fail()
    _exact_keys(
        table,
        {"kind", "endpoint", "credential_ref"},
    )
    endpoint = table.get("endpoint")
    if endpoint is not None:
        endpoint = _string(endpoint)
    try:
        base = curated_profile(
            kind,
            _credential_ref(table.get("credential_ref"), provider_id),
            endpoint_override=endpoint,
        )
        profile = replace(base, profile_id=provider_id)
    except (CredentialSecurityError, ProviderAdapterError, LLMConfigError):
        pass
    else:
        return ProviderDefinition(
            provider_id, kind, profile.capabilities, profile=profile
        )
    raise _fail()


def _provider(provider_id: str, value: object) -> ProviderDefinition:
    if _PROFILE_ID.fullmatch(provider_id) is None or not isinstance(value, Mapping):
        raise _fail()
    table = dict(value)
    kind = _string(table.get("kind"))
    if kind == "ollama":
        _exact_keys(table, {"kind"})
        return ProviderDefinition(
            provider_id,
            kind,
            BackendCapabilities(
                generation=True, embedding=True, structured_output=True
            ),
        )
    if kind == "omlx":
        _exact_keys(table, {"kind", "endpoint"})
        endpoint = _string(table.get("endpoint", OMLX_BASE_URL)).rstrip("/")
        try:
            parsed = urlsplit(endpoint)
            port = parsed.port
        except ValueError:
            raise _fail() from None
        if (
            parsed.scheme != "http"
            or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
            or parsed.username is not None
            or parsed.password is not None
            or port is None
            or parsed.path != "/v1"
            or parsed.query
            or parsed.fragment
        ):
            raise _fail()
        return ProviderDefinition(
            provider_id,
            kind,
            BackendCapabilities(
                generation=True,
                embedding=True,
                structured_output=True,
                streaming=True,
            ),
            endpoint=endpoint,
        )
    if kind == "local-transformers":
        _exact_keys(table, {"kind", "backend", "device", "max_length", "batch_size"})
        backend_value = table.get("backend", "transformers")
        device_value = table.get("device", "")
        if (
            not isinstance(backend_value, str)
            or backend_value not in {"transformers", "flagembedding"}
            or not isinstance(device_value, str)
            or any(ord(character) < 32 for character in device_value)
        ):
            raise _fail()
        config = RerankerConfig(
            enabled=True,
            backend=backend_value,
            device=device_value,
            max_length=_positive_int(
                table.get("max_length"), RerankerConfig.max_length
            ),
            batch_size=_positive_int(
                table.get("batch_size"), RerankerConfig.batch_size
            ),
        )
        return ProviderDefinition(
            provider_id,
            kind,
            BackendCapabilities(generation=False, embedding=False, rerank=True),
            reranker_config=config,
        )
    if kind == "nemotron":
        _exact_keys(table, {"kind", "device"})
        device = _string(table.get("device"))
        if device not in {"mps", "cpu"}:
            raise _fail()
        return ProviderDefinition(
            provider_id,
            kind,
            BackendCapabilities(generation=False, embedding=True),
            embedding_device=device,
        )
    if kind in CURATED_PROFILE_IDS or kind == ProviderProtocol.OPENAI_COMPATIBLE.value:
        return _remote_profile(provider_id, kind, table)
    raise _fail()


def _role(
    role_name: str,
    value: object,
    providers: Mapping[str, ProviderDefinition],
) -> RoleDefinition:
    if _ROLE_ID.fullmatch(role_name) is None or not isinstance(value, Mapping):
        raise _fail()
    table = dict(value)
    _exact_keys(
        table,
        {"capability", "provider", "model", "required_capabilities", "revision"},
    )
    try:
        capability = RoleCapability(_string(table.get("capability")))
    except (ValueError, LLMConfigError):
        raise _fail() from None
    provider_id = _string(table.get("provider"))
    provider = providers.get(provider_id)
    if provider is None:
        raise _fail()
    model = _string(table.get("model"))
    required = _string_list(table.get("required_capabilities"))
    revision = _string(table["revision"]) if "revision" in table else None
    if revision is not None and safe_metadata_identifier(revision) is None:
        raise _fail()
    if any(item not in _CAPABILITY_FIELDS for item in required):
        raise _fail()
    capabilities = provider.capabilities_for(model)
    if not getattr(capabilities, capability.value) or any(
        not getattr(capabilities, item) for item in required
    ):
        raise _fail(LLMConfigFailureCategory.CAPABILITY_UNAVAILABLE)
    return RoleDefinition(role_name, capability, provider_id, model, required, revision)


def _egress_opt_ins(
    value: object,
    roles: Mapping[str, RoleDefinition],
    providers: Mapping[str, ProviderDefinition],
) -> frozenset[tuple[str, SourceDataClass]]:
    if value is None:
        return frozenset()
    if not isinstance(value, list):
        raise _fail()
    result: set[tuple[str, SourceDataClass]] = set()
    for item in value:
        if not isinstance(item, Mapping):
            raise _fail()
        table = dict(item)
        _exact_keys(table, {"role", "data_class"})
        role_name = _string(table.get("role"))
        role = roles.get(role_name)
        if role is None or providers[role.provider_id].profile is None:
            raise _fail()
        try:
            data_class = SourceDataClass(_string(table.get("data_class")))
        except (ValueError, LLMConfigError):
            raise _fail() from None
        pair = (role_name, data_class)
        if pair in result:
            raise _fail()
        result.add(pair)
    return frozenset(result)


def parse_llm_config(payload: Mapping[str, object]) -> LLMConfig:
    _reject_credential_fields(payload)
    llm = payload.get("llm")
    if not isinstance(llm, Mapping):
        raise _fail()
    table = dict(llm)
    _exact_keys(table, {"providers", "roles", "egress_opt_in"})
    raw_providers = table.get("providers")
    raw_roles = table.get("roles")
    if (
        not isinstance(raw_providers, Mapping)
        or not raw_providers
        or not isinstance(raw_roles, Mapping)
        or not raw_roles
    ):
        raise _fail()
    providers = {
        provider_id: _provider(provider_id, value)
        for provider_id, value in raw_providers.items()
        if isinstance(provider_id, str)
    }
    if len(providers) != len(raw_providers):
        raise _fail()
    roles = {
        role_name: _role(role_name, value, providers)
        for role_name, value in raw_roles.items()
        if isinstance(role_name, str)
    }
    if len(roles) != len(raw_roles):
        raise _fail()
    egress = _egress_opt_ins(table.get("egress_opt_in"), roles, providers)
    return LLMConfig(MappingProxyType(providers), MappingProxyType(roles), egress)


def load_llm_config(path: Path | str | None = None) -> LLMConfig:
    resolved = Path(path).expanduser() if path is not None else CONFIG_FILE
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        path_metadata = resolved.lstat()
        if not stat.S_ISREG(path_metadata.st_mode):
            raise _fail()
        descriptor = os.open(resolved, flags)
    except OSError:
        raise _fail(LLMConfigFailureCategory.UNAVAILABLE) from None
    try:
        metadata = os.fstat(descriptor)
        unsafe_owner = hasattr(os, "getuid") and metadata.st_uid != os.getuid()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or unsafe_owner
            or metadata.st_mode & 0o022
            or (metadata.st_dev, metadata.st_ino)
            != (path_metadata.st_dev, path_metadata.st_ino)
        ):
            raise _fail()
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = -1
            snapshot = stream.read()
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    try:
        payload = tomllib.loads(snapshot.decode("utf-8"))
    except (UnicodeError, tomllib.TOMLDecodeError):
        pass
    else:
        if isinstance(payload, dict):
            return parse_llm_config(payload)
    raise _fail(LLMConfigFailureCategory.PARSE_ERROR)


def build_llm_runtime(
    config: LLMConfig,
    *,
    resolver: CredentialResolver | None = None,
    sender_factory: SenderFactory | None = None,
    search_embedding_config: SearchEmbeddingConfig | None = None,
    telemetry: Callable[[RuntimeFailureTelemetry], None] | None = None,
) -> LLMRuntime:
    if resolver is None:
        resolver = CredentialResolver()
    used_provider_ids = {role.provider_id for role in config.roles.values()}
    nemotron_provider_ids = {
        provider_id
        for provider_id in used_provider_ids
        if config.providers[provider_id].kind == "nemotron"
    }
    if nemotron_provider_ids and search_embedding_config is None:
        search_embedding_config = load_search_embedding_config()
    incremental_provider_id: str | None = None
    if search_embedding_config is not None:
        for role_name, expected_device in (
            ("search.semantic.foreground", search_embedding_config.query_device),
            (
                "search.semantic.incremental",
                search_embedding_config.incremental_device,
            ),
        ):
            role = config.roles.get(role_name)
            if role is None:
                continue
            provider = config.providers[role.provider_id]
            if provider.kind == "nemotron":
                if provider.embedding_device != expected_device:
                    raise _fail()
                if role_name == "search.semantic.incremental":
                    incremental_provider_id = role.provider_id
        foreground = config.roles.get("search.semantic.foreground")
        incremental = config.roles.get("search.semantic.incremental")
        if (
            foreground is not None
            and incremental is not None
            and config.providers[foreground.provider_id].kind == "nemotron"
            and config.providers[incremental.provider_id].kind == "nemotron"
            and foreground.provider_id == incremental.provider_id
        ):
            raise _fail()
    backends: dict[str, object] = {}
    for provider_id in used_provider_ids:
        provider = config.providers[provider_id]
        if provider.kind == "ollama":
            backends[provider_id] = OllamaAdapter()
        elif provider.kind == "omlx":
            backends[provider_id] = OMLXAdapter(
                base_url=provider.endpoint or OMLX_BASE_URL
            )
        elif provider.kind == "local-transformers":
            if provider.reranker_config is None:
                raise _fail()
            backends[provider_id] = LocalRerankBackend(provider.reranker_config)
        elif provider.kind == "nemotron":
            if provider.embedding_device is None:
                raise _fail()
            if search_embedding_config is None:
                raise _fail()
            provider_models = {
                role.model
                for role in config.roles.values()
                if role.provider_id == provider_id
                and role.capability is RoleCapability.EMBEDDING
            }
            if len(provider_models) != 1:
                raise _fail()
            backends[provider_id] = NemotronEmbeddingBackend(
                search_embedding_config,
                model=next(iter(provider_models)),
                device=provider.embedding_device,
                incremental=provider_id == incremental_provider_id,
            )
        elif provider.profile is not None:
            sender = (
                sender_factory(provider.profile) if sender_factory is not None else None
            )
            backends[provider_id] = (
                compose_anthropic_adapter(provider.profile, resolver, sender=sender)
                if provider.profile.protocol is ProviderProtocol.ANTHROPIC_MESSAGES
                else compose_openai_compatible_adapter(
                    provider.profile, resolver, sender=sender
                )
            )
        else:
            raise _fail()
    generation: dict[str, GenerationRoute] = {}
    embedding: dict[str, EmbeddingRoute] = {}
    rerank: dict[str, RerankRoute] = {}
    local_controls: dict[str, OllamaAdapter] = {}
    for role_name, role in config.roles.items():
        backend = backends[role.provider_id]
        if role.capability is RoleCapability.GENERATION:
            provider = config.providers[role.provider_id]
            if provider.profile is not None:
                protocol = provider.profile.protocol.value
                endpoint_sha256 = hashlib.sha256(
                    provider.profile.endpoint.encode("utf-8")
                ).hexdigest()
            elif provider.kind == "ollama":
                protocol = "ollama-native"
                endpoint_sha256 = hashlib.sha256(OLLAMA_URL.encode("utf-8")).hexdigest()
            elif provider.kind == "omlx":
                protocol = "omlx-native"
                endpoint_sha256 = hashlib.sha256(
                    (provider.endpoint or OMLX_BASE_URL).encode("utf-8")
                ).hexdigest()
            else:
                protocol = provider.kind
                endpoint_sha256 = None
            generation[role_name] = GenerationRoute(
                cast(GenerationBackend, backend),
                role.model,
                provider.capabilities_for(role.model),
                protocol,
                endpoint_sha256,
                role.revision,
            )
        elif role.capability is RoleCapability.EMBEDDING:
            embedding[role_name] = EmbeddingRoute(
                cast(EmbeddingBackend, backend), role.model
            )
        else:
            rerank[role_name] = RerankRoute(cast(RerankBackend, backend), role.model)
        if isinstance(backend, OllamaAdapter):
            local_controls[role_name] = backend
    return LLMRuntime(
        generation=generation,
        embedding=embedding,
        rerank=rerank,
        local_controls=local_controls,
        remote_egress_opt_ins=config.egress_opt_ins,
        telemetry=telemetry,
    )


def load_llm_runtime(
    path: Path | str | None = None,
    *,
    resolver: CredentialResolver | None = None,
    sender_factory: SenderFactory | None = None,
    telemetry: Callable[[RuntimeFailureTelemetry], None] | None = None,
) -> LLMRuntime:
    return build_llm_runtime(
        load_llm_config(path),
        resolver=resolver,
        sender_factory=sender_factory,
        telemetry=telemetry,
    )


@cache
def load_default_llm_runtime() -> LLMRuntime:
    """Load the process-wide runtime from the canonical configuration."""

    return load_llm_runtime(telemetry=runtime_status.append_runtime_failure)
