from __future__ import annotations

import hashlib
import json
import os
import tomllib
from pathlib import Path
from urllib.request import Request

import httpx
import pytest

from chronovisor.core import llm_config, runtime_status
from chronovisor.core.anthropic_adapter import AnthropicMessagesAdapter
from chronovisor.core.llm_config import (
    LLMConfigError,
    LLMConfigFailureCategory,
    build_llm_runtime,
    load_default_llm_runtime,
    load_llm_config,
    parse_llm_config,
)
from chronovisor.core.llm_runtime import (
    CapabilityUnavailableError,
    EmbeddingRequest,
    GenerationRequest,
    RouteLocation,
    SourceDataClass,
    SourceDataClassification,
    SourceSensitivity,
)
from chronovisor.core.llm_security import (
    CredentialRef,
    CredentialResolver,
    CredentialSecurityError,
    SecretValue,
)
from chronovisor.core.nemotron_adapter import NemotronEmbeddingBackend
from chronovisor.core.ollama_adapter import OllamaAdapter
from chronovisor.core.openai_compatible_adapter import OpenAICompatibleAdapter
from chronovisor.core.provider_profiles import CURATED_PROFILE_IDS, ProviderProfile
from chronovisor.core.reranker import LocalRerankBackend
from chronovisor.core.runtime_config import SearchEmbeddingConfig


def test_default_runtime_loader_caches_one_process_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = object()
    calls = 0
    telemetry: list[object] = []

    def load(**kwargs: object) -> object:
        nonlocal calls
        calls += 1
        telemetry.append(kwargs.get("telemetry"))
        return expected

    load_default_llm_runtime.cache_clear()
    monkeypatch.setattr(llm_config, "load_llm_runtime", load)
    try:
        assert load_default_llm_runtime() is expected
        assert load_default_llm_runtime() is expected
        assert calls == 1
        assert telemetry == [runtime_status.append_runtime_failure]
    finally:
        load_default_llm_runtime.cache_clear()


CANARY = "sk-CANARY-LLM-CONFIG"
NORMAL_PAGE = SourceDataClassification(SourceDataClass.PAGE, SourceSensitivity.NORMAL)


class CountingResolver(CredentialResolver):
    def __init__(self) -> None:
        self.calls: list[CredentialRef] = []

    def resolve(self, ref: CredentialRef) -> SecretValue:
        self.calls.append(ref)
        return SecretValue(CANARY)


class FakeSender:
    def __init__(self, *responses: httpx.Response) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[Request, bool, float]] = []

    def __call__(
        self, request: Request, *, follow_redirects: bool, timeout_seconds: float
    ) -> object:
        self.calls.append((request, follow_redirects, timeout_seconds))
        return self.responses.pop(0)


def _response(payload: object) -> httpx.Response:
    return httpx.Response(200, json=payload)


def _provider(kind: str = "openai") -> dict[str, object]:
    result: dict[str, object] = {
        "kind": kind,
        "credential_ref": "env:CHRONOVISOR_TEST_API_KEY",
    }
    if kind in {"qwen", "dashscope"}:
        result["endpoint"] = "https://dashscope.example.test/compatible-mode/v1"
    return result


def _remote_payload(
    *,
    kind: str = "openai",
    capability: str = "generation",
) -> dict[str, object]:
    return {
        "llm": {
            "providers": {"remote": _provider(kind)},
            "roles": {
                "answer": {
                    "capability": capability,
                    "provider": "remote",
                    "model": "route-model",
                }
            },
        }
    }


def test_local_only_config_composes_ollama_and_transformer_reranker() -> None:
    config = parse_llm_config(
        {
            "llm": {
                "providers": {
                    "ollama": {"kind": "ollama"},
                    "reranker": {
                        "kind": "local-transformers",
                        "backend": "transformers",
                        "batch_size": 4,
                    },
                },
                "roles": {
                    "answer": {
                        "capability": "generation",
                        "provider": "ollama",
                        "model": "local-generation",
                    },
                    "embed": {
                        "capability": "embedding",
                        "provider": "ollama",
                        "model": "local-embedding",
                    },
                    "rerank": {
                        "capability": "rerank",
                        "provider": "reranker",
                        "model": "local-reranker",
                    },
                },
            }
        }
    )
    resolver = CountingResolver()

    runtime = build_llm_runtime(config, resolver=resolver)

    assert isinstance(runtime._generation["answer"].backend, OllamaAdapter)
    assert isinstance(runtime._embedding["embed"].backend, OllamaAdapter)
    assert isinstance(runtime._rerank["rerank"].backend, LocalRerankBackend)
    assert runtime.resolve_rerank("rerank").model == "local-reranker"
    assert runtime._rerank["rerank"].backend.config.batch_size == 4
    assert resolver.calls == []


def test_legacy_ingest_model_cannot_override_generation_role() -> None:
    config = parse_llm_config(
        {
            "ingest": {"model": "stale-selector"},
            "llm": {
                "providers": {"local": {"kind": "ollama"}},
                "roles": {
                    "ingest.generation": {
                        "capability": "generation",
                        "provider": "local",
                        "model": "route-selected",
                    }
                },
            },
        }
    )

    assert config.roles["ingest.generation"].model == "route-selected"


def test_local_nemotron_roles_compose_lazy_device_bound_backends() -> None:
    config = parse_llm_config(
        {
            "llm": {
                "providers": {
                    "foreground": {"kind": "nemotron", "device": "mps"},
                    "incremental": {"kind": "nemotron", "device": "cpu"},
                },
                "roles": {
                    "search.semantic.foreground": {
                        "capability": "embedding",
                        "provider": "foreground",
                        "model": "nemotron-test",
                    },
                    "search.semantic.incremental": {
                        "capability": "embedding",
                        "provider": "incremental",
                        "model": "nemotron-test",
                    },
                },
            }
        }
    )
    search_config = SearchEmbeddingConfig(dimensions=2)

    runtime = build_llm_runtime(config, search_embedding_config=search_config)

    foreground = runtime._embedding["search.semantic.foreground"].backend
    incremental = runtime._embedding["search.semantic.incremental"].backend
    assert isinstance(foreground, NemotronEmbeddingBackend)
    assert isinstance(incremental, NemotronEmbeddingBackend)
    assert foreground.device == "mps"
    assert incremental.device == "cpu"
    assert foreground.model == "nemotron-test"
    assert incremental.model == "nemotron-test"
    assert foreground.incremental is False
    assert incremental.incremental is True
    assert foreground._model is None
    assert incremental._model is None


def test_local_nemotron_role_device_mismatch_fails_before_model_loading() -> None:
    config = parse_llm_config(
        {
            "llm": {
                "providers": {"wrong": {"kind": "nemotron", "device": "cpu"}},
                "roles": {
                    "search.semantic.foreground": {
                        "capability": "embedding",
                        "provider": "wrong",
                        "model": "nemotron-test",
                    }
                },
            }
        }
    )

    with pytest.raises(LLMConfigError):
        build_llm_runtime(
            config,
            search_embedding_config=SearchEmbeddingConfig(),
        )


def test_local_nemotron_provider_rejects_multiple_route_models() -> None:
    config = parse_llm_config(
        {
            "llm": {
                "providers": {"foreground": {"kind": "nemotron", "device": "mps"}},
                "roles": {
                    "search.semantic.foreground": {
                        "capability": "embedding",
                        "provider": "foreground",
                        "model": "model-a",
                    },
                    "other.embedding": {
                        "capability": "embedding",
                        "provider": "foreground",
                        "model": "model-b",
                    },
                },
            }
        }
    )

    with pytest.raises(LLMConfigError):
        build_llm_runtime(config, search_embedding_config=SearchEmbeddingConfig())


def test_one_remote_profile_resolves_once_and_routes_generation_and_embedding() -> None:
    payload = _remote_payload()
    roles = payload["llm"]["roles"]  # type: ignore[index]
    roles["embed"] = {  # type: ignore[index]
        "capability": "embedding",
        "provider": "remote",
        "model": "route-embedder",
    }
    config = parse_llm_config(payload)
    resolver = CountingResolver()
    sender = FakeSender(
        _response(
            {
                "choices": [
                    {
                        "message": {"content": "answer"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            }
        ),
        _response({"data": [{"index": 0, "embedding": [1.0, 2.0]}]}),
    )
    profiles: list[ProviderProfile] = []

    def sender_factory(profile: ProviderProfile) -> FakeSender:
        profiles.append(profile)
        return sender

    runtime = build_llm_runtime(
        config, resolver=resolver, sender_factory=sender_factory
    )
    generated = runtime.generate("answer", GenerationRequest("prompt", NORMAL_PAGE))
    embedded = runtime.embed("embed", EmbeddingRequest(("text",), NORMAL_PAGE))

    assert generated.content == "answer"
    assert embedded.vectors == ((1.0, 2.0),)
    assert len(resolver.calls) == 1
    assert len(profiles) == 1
    assert len(sender.calls) == 2
    assert all(not follow_redirects for _, follow_redirects, _ in sender.calls)
    assert CANARY not in repr(sender.calls)


def test_hybrid_config_routes_local_and_remote_without_domain_provider_inputs() -> None:
    config = parse_llm_config(
        {
            "llm": {
                "providers": {
                    "cloud": _provider("anthropic"),
                    "ollama": {"kind": "ollama"},
                    "reranker": {"kind": "local-transformers"},
                },
                "roles": {
                    "answer": {
                        "capability": "generation",
                        "provider": "cloud",
                        "model": "claude-route-model",
                    },
                    "embed": {
                        "capability": "embedding",
                        "provider": "ollama",
                        "model": "embed-route-model",
                    },
                    "rerank": {
                        "capability": "rerank",
                        "provider": "reranker",
                        "model": "rerank-route-model",
                    },
                },
                "egress_opt_in": [{"role": "answer", "data_class": "system"}],
            }
        }
    )
    resolver = CountingResolver()
    sender = FakeSender()

    runtime = build_llm_runtime(
        config, resolver=resolver, sender_factory=lambda _profile: sender
    )

    assert isinstance(runtime._generation["answer"].backend, AnthropicMessagesAdapter)
    assert isinstance(runtime._embedding["embed"].backend, OllamaAdapter)
    assert isinstance(runtime._rerank["rerank"].backend, LocalRerankBackend)
    assert len(resolver.calls) == 1
    assert config.egress_opt_ins == {("answer", SourceDataClass.SYSTEM)}


def test_cloud_only_representative_roles_use_one_openai_backend() -> None:
    decision_specs = {
        "classification.primary": ("gpt-5", "deployment-primary"),
        "classification.challenger": ("gpt-5-mini", "deployment-challenger"),
        "classification.tie_break": ("gpt-5-nano", "deployment-tie-break"),
    }
    decision_roles = {
        role: {
            "capability": "generation",
            "provider": "remote",
            "model": model,
            "required_capabilities": ["structured_output"],
            "revision": revision,
        }
        for role, (model, revision) in decision_specs.items()
    }
    embedding_roles = {
        role: {
            "capability": "embedding",
            "provider": "remote",
            "model": "text-embedding-3-large",
        }
        for role in (
            "search.semantic.foreground",
            "search.semantic.incremental",
            "knowledge.embedding",
            "classification.embedding",
        )
    }
    config = parse_llm_config(
        {
            "llm": {
                "providers": {"remote": _provider("openai")},
                "roles": {
                    "librarian.review": {
                        "capability": "generation",
                        "provider": "remote",
                        "model": "gpt-5-mini",
                    },
                    **decision_roles,
                    **embedding_roles,
                },
            }
        }
    )
    resolver = CountingResolver()

    runtime = build_llm_runtime(
        config, resolver=resolver, sender_factory=lambda _profile: FakeSender()
    )

    generation_routes = {
        role: runtime.resolve_generation(role) for role in runtime._generation
    }
    assert {
        role: (route.provider, route.model, route.location, route.revision)
        for role, route in generation_routes.items()
    } == {
        "librarian.review": ("remote", "gpt-5-mini", RouteLocation.REMOTE, None),
        **{
            role: ("remote", model, RouteLocation.REMOTE, revision)
            for role, (model, revision) in decision_specs.items()
        },
    }
    embedding_routes = {
        role: runtime.resolve_embedding(role) for role in runtime._embedding
    }
    assert {
        role: (route.provider, route.model, route.location)
        for role, route in embedding_routes.items()
    } == {
        role: ("remote", "text-embedding-3-large", RouteLocation.REMOTE)
        for role in embedding_roles
    }
    assert {type(route.backend) for route in runtime._generation.values()} == {
        OpenAICompatibleAdapter
    }
    assert {type(route.backend) for route in runtime._embedding.values()} == {
        OpenAICompatibleAdapter
    }
    assert runtime._local_controls == {}
    assert all(
        generation_routes[role].capabilities.structured_output
        for role in decision_roles
    )
    assert "search.rerank" not in config.roles
    with pytest.raises(CapabilityUnavailableError):
        runtime.resolve_rerank("search.rerank")
    assert len(resolver.calls) == 1


def test_hybrid_representative_roles_keep_only_tie_break_and_rerank_local() -> None:
    remote_decision_specs = {
        "classification.primary": ("gpt-5-mini", "deployment-primary"),
        "classification.challenger": ("gpt-5-nano", "deployment-challenger"),
    }
    remote_decision_roles = {
        role: {
            "capability": "generation",
            "provider": "remote",
            "model": model,
            "required_capabilities": ["structured_output"],
            "revision": revision,
        }
        for role, (model, revision) in remote_decision_specs.items()
    }
    remote_generation_roles = {
        "librarian.review": {
            "capability": "generation",
            "provider": "remote",
            "model": "gpt-5",
        },
        **remote_decision_roles,
    }
    remote_embedding_roles = {
        role: {
            "capability": "embedding",
            "provider": "remote",
            "model": "text-embedding-3-large",
        }
        for role in (
            "search.semantic.foreground",
            "search.semantic.incremental",
            "knowledge.embedding",
            "classification.embedding",
        )
    }
    config = parse_llm_config(
        {
            "llm": {
                "providers": {
                    "remote": _provider("openai"),
                    "local": {"kind": "ollama"},
                    "reranker": {"kind": "local-transformers"},
                },
                "roles": {
                    **remote_generation_roles,
                    "classification.tie_break": {
                        "capability": "generation",
                        "provider": "local",
                        "model": "local-tie-break",
                        "required_capabilities": ["structured_output"],
                    },
                    **remote_embedding_roles,
                    "search.rerank": {
                        "capability": "rerank",
                        "provider": "reranker",
                        "model": "local-reranker",
                    },
                },
            }
        }
    )
    resolver = CountingResolver()

    runtime = build_llm_runtime(
        config, resolver=resolver, sender_factory=lambda _profile: FakeSender()
    )

    generation_routes = {
        role: runtime.resolve_generation(role) for role in runtime._generation
    }
    assert {
        role: (route.provider, route.model, route.location, route.revision)
        for role, route in generation_routes.items()
    } == {
        "librarian.review": ("remote", "gpt-5", RouteLocation.REMOTE, None),
        **{
            role: ("remote", model, RouteLocation.REMOTE, revision)
            for role, (model, revision) in remote_decision_specs.items()
        },
        "classification.tie_break": (
            "ollama",
            "local-tie-break",
            RouteLocation.LOCAL,
            None,
        ),
    }
    embedding_routes = {
        role: runtime.resolve_embedding(role) for role in runtime._embedding
    }
    assert {
        role: (route.provider, route.model, route.location)
        for role, route in embedding_routes.items()
    } == {
        role: ("remote", "text-embedding-3-large", RouteLocation.REMOTE)
        for role in remote_embedding_roles
    }
    assert {
        role: type(route.backend) for role, route in runtime._generation.items()
    } == {
        **{role: OpenAICompatibleAdapter for role in remote_generation_roles},
        "classification.tie_break": OllamaAdapter,
    }
    assert {type(route.backend) for route in runtime._embedding.values()} == {
        OpenAICompatibleAdapter
    }
    assert type(runtime._rerank["search.rerank"].backend) is LocalRerankBackend
    assert set(runtime._local_controls) == {"classification.tie_break"}
    rerank = runtime.resolve_rerank("search.rerank")
    assert (rerank.role, rerank.provider, rerank.model, rerank.location) == (
        "search.rerank",
        "local-reranker",
        "local-reranker",
        RouteLocation.LOCAL,
    )
    assert len(resolver.calls) == 1


@pytest.mark.parametrize("kind", CURATED_PROFILE_IDS)
def test_every_curated_provider_parses_with_conservative_generation(kind: str) -> None:
    config = parse_llm_config(_remote_payload(kind=kind))

    provider = config.providers["remote"]
    assert provider.profile is not None
    assert provider.profile.capabilities.generation
    assert not provider.profile.capabilities.structured_output
    assert not provider.profile.capabilities.tools


@pytest.mark.parametrize("model", ["gpt-5", "gpt-5-mini", "gpt-5-nano"])
def test_verified_curated_openai_models_resolve_structured_output(model: str) -> None:
    payload = _remote_payload(kind="openai")
    role = payload["llm"]["roles"]["answer"]  # type: ignore[index]
    role["model"] = model  # type: ignore[index]
    role["required_capabilities"] = ["structured_output"]  # type: ignore[index]

    config = parse_llm_config(payload)
    runtime = build_llm_runtime(
        config,
        resolver=CountingResolver(),
        sender_factory=lambda _profile: FakeSender(),
    )

    assert config.providers["remote"].capabilities_for(model).structured_output
    assert runtime.resolve_generation("answer").model == model


@pytest.mark.parametrize(
    ("kind", "model"),
    [
        ("openai", "unverified-openai-model"),
        ("openai-compatible", "gpt-5"),
        ("openai-compatible", "gpt-5-mini"),
        ("openai-compatible", "gpt-5-nano"),
    ],
)
def test_unverified_remote_structured_output_fails_closed(
    kind: str, model: str
) -> None:
    payload = _remote_payload(kind=kind)
    provider = payload["llm"]["providers"]["remote"]  # type: ignore[index]
    if kind == "openai-compatible":
        provider["endpoint"] = "https://gateway.example.test/v1"  # type: ignore[index]
    role = payload["llm"]["roles"]["answer"]  # type: ignore[index]
    role["model"] = model  # type: ignore[index]
    role["required_capabilities"] = ["structured_output"]  # type: ignore[index]

    with pytest.raises(LLMConfigError) as exc:
        parse_llm_config(payload)

    assert exc.value.category is LLMConfigFailureCategory.CAPABILITY_UNAVAILABLE


def test_generic_openai_compatible_uses_shared_profile_constructor() -> None:
    payload = _remote_payload(kind="openai-compatible")
    provider = payload["llm"]["providers"]["remote"]  # type: ignore[index]
    provider["endpoint"] = "https://gateway.example.test/openai/v1"  # type: ignore[index]
    provider["auth_scheme"] = "x-api-key"  # type: ignore[index]

    config = parse_llm_config(payload)
    resolver = CountingResolver()
    runtime = build_llm_runtime(
        config, resolver=resolver, sender_factory=lambda _profile: FakeSender()
    )

    assert isinstance(runtime._generation["answer"].backend, OpenAICompatibleAdapter)
    assert len(resolver.calls) == 1


def test_role_revision_is_safe_and_propagates_to_runtime_route() -> None:
    payload = _remote_payload(kind="openai-compatible")
    provider = payload["llm"]["providers"]["remote"]  # type: ignore[index]
    provider["endpoint"] = "https://gateway.example.test/v1"  # type: ignore[index]
    role = payload["llm"]["roles"]["answer"]  # type: ignore[index]
    role["revision"] = "deployment-2026.08.10"  # type: ignore[index]

    config = parse_llm_config(payload)
    runtime = build_llm_runtime(
        config,
        resolver=CountingResolver(),
        sender_factory=lambda _profile: FakeSender(),
    )
    route = runtime.resolve_generation("answer")

    assert config.roles["answer"].revision == "deployment-2026.08.10"
    assert route.revision == "deployment-2026.08.10"
    assert route.protocol == "openai-compatible"
    assert route.endpoint_sha256 == hashlib.sha256(
        b"https://gateway.example.test/v1"
    ).hexdigest()


@pytest.mark.parametrize(
    "revision",
    ["line\nbreak", "/private/model/path", "x" * 129, "unsafe@revision"],
)
def test_role_revision_rejects_unsafe_identity(revision: str) -> None:
    payload = _remote_payload()
    role = payload["llm"]["roles"]["answer"]  # type: ignore[index]
    role["revision"] = revision  # type: ignore[index]

    with pytest.raises(LLMConfigError) as exc:
        parse_llm_config(payload)

    assert exc.value.category is LLMConfigFailureCategory.SCHEMA_INVALID
    assert revision not in repr(exc.value)


@pytest.mark.parametrize(
    "payload,category",
    [
        (
            _remote_payload(kind="deepseek", capability="embedding"),
            LLMConfigFailureCategory.CAPABILITY_UNAVAILABLE,
        ),
        (
            {
                "llm": {
                    "providers": {"remote": _provider()},
                    "roles": {
                        "answer": {
                            "capability": "generation",
                            "provider": "remote",
                            "model": "model",
                            "required_capabilities": ["tools"],
                        }
                    },
                }
            },
            LLMConfigFailureCategory.CAPABILITY_UNAVAILABLE,
        ),
        (
            _remote_payload(capability="unknown"),
            LLMConfigFailureCategory.SCHEMA_INVALID,
        ),
    ],
)
def test_unsupported_capabilities_fail_before_composition(
    payload: dict[str, object],
    category: LLMConfigFailureCategory,
) -> None:
    with pytest.raises(LLMConfigError) as exc:
        parse_llm_config(payload)

    assert exc.value.category is category


@pytest.mark.parametrize(
    "mutate",
    [
        lambda payload: payload.update({"api_key": CANARY}),
        lambda payload: payload.update({"provider_api_key": CANARY}),
        lambda payload: payload["llm"].update({"unknown": True}),  # type: ignore[union-attr]
        lambda payload: payload["llm"]["providers"]["remote"].update(  # type: ignore[index,union-attr]
            {"token": CANARY}
        ),
        lambda payload: payload["llm"]["providers"]["remote"].update(  # type: ignore[index,union-attr]
            {"tls_verify": False}
        ),
        lambda payload: payload["llm"]["providers"]["remote"].update(  # type: ignore[index,union-attr]
            {"endpoint": f"https://user:{CANARY}@example.test/v1"}
        ),
        lambda payload: payload["llm"]["providers"]["remote"].update(  # type: ignore[index,union-attr]
            {"endpoint": f"https://example.test/v1?api_key={CANARY}"}
        ),
        lambda payload: payload["llm"]["providers"]["remote"].update(  # type: ignore[index,union-attr]
            {"structured_output_models": ["unverified-model"]}
        ),
        lambda payload: payload["llm"]["providers"]["remote"].update(  # type: ignore[index,union-attr]
            {"embedding": True}
        ),
        lambda payload: payload["llm"]["providers"]["remote"].update(  # type: ignore[index,union-attr]
            {"kind": 7}
        ),
        lambda payload: payload["llm"]["providers"]["remote"].update(  # type: ignore[index,union-attr]
            {"credential_ref": "oskeyring:other/default"}
        ),
    ],
)
def test_invalid_and_secret_bearing_config_is_rejected_without_canary_leak(
    mutate: object,
) -> None:
    payload = _remote_payload(kind="openai-compatible")
    provider = payload["llm"]["providers"]["remote"]  # type: ignore[index]
    provider["endpoint"] = "https://gateway.example.test/v1"  # type: ignore[index]
    assert callable(mutate)
    mutate(payload)

    with pytest.raises(LLMConfigError) as exc:
        parse_llm_config(payload)

    assert exc.value.category is LLMConfigFailureCategory.SCHEMA_INVALID
    assert CANARY not in repr(exc.value)


def test_unused_remote_provider_is_not_resolved() -> None:
    config = parse_llm_config(
        {
            "llm": {
                "providers": {
                    "local": {"kind": "ollama"},
                    "unused": _provider(),
                },
                "roles": {
                    "answer": {
                        "capability": "generation",
                        "provider": "local",
                        "model": "local-model",
                    }
                },
            }
        }
    )
    resolver = CountingResolver()

    build_llm_runtime(config, resolver=resolver)

    assert resolver.calls == []


def test_missing_credential_fails_before_sender_call(tmp_path: Path) -> None:
    config = parse_llm_config(_remote_payload())
    sender = FakeSender()
    resolver = CredentialResolver(
        environ={}, repo_root=tmp_path / "repo", home_root=tmp_path / "home"
    )

    with pytest.raises(CredentialSecurityError):
        build_llm_runtime(
            config, resolver=resolver, sender_factory=lambda _profile: sender
        )

    assert sender.calls == []


def test_file_loader_reports_missing_parse_and_unsafe_mode_safely(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing.toml"
    with pytest.raises(LLMConfigError) as unavailable:
        load_llm_config(missing)
    assert unavailable.value.category is LLMConfigFailureCategory.UNAVAILABLE

    malformed = tmp_path / "malformed.toml"
    malformed.write_text(f"api_key = {json.dumps(CANARY)}\n[llm", encoding="utf-8")
    with pytest.raises(LLMConfigError) as parse_error:
        load_llm_config(malformed)
    assert parse_error.value.category is LLMConfigFailureCategory.PARSE_ERROR
    assert CANARY not in repr(parse_error.value)

    unsafe = tmp_path / "unsafe.toml"
    unsafe.write_text("[llm]\n", encoding="utf-8")
    unsafe.chmod(0o622)
    with pytest.raises(LLMConfigError) as unsafe_error:
        load_llm_config(unsafe)
    assert unsafe_error.value.category is LLMConfigFailureCategory.SCHEMA_INVALID

    if hasattr(os, "symlink"):
        symlink = tmp_path / "config-link.toml"
        symlink.symlink_to(malformed)
        with pytest.raises(LLMConfigError) as symlink_error:
            load_llm_config(symlink)
        assert symlink_error.value.category is LLMConfigFailureCategory.SCHEMA_INVALID


def test_repository_example_has_representative_local_role_map() -> None:
    example = Path(__file__).parents[1] / "config.toml.example"
    config = load_llm_config(example)

    assert set(config.roles) >= {
        "knowledge.embedding",
        "search.semantic.foreground",
        "search.semantic.incremental",
        "search.rerank",
        "librarian.review",
        "librarian.review.challenger",
        "recall.answer.runner",
        "recall.answer.scorer",
        "recall.certificate_judge.primary",
        "recall.certificate_judge.escalation",
        "recall.auditor",
        "recall.gate",
        "recall.query_rewriter",
        "recall.policy_proposer.primary",
        "recall.policy_proposer.challenger",
        "recall.rubric.variant",
        "recall.distill.teacher.a",
        "recall.distill.teacher.b",
        "recall.distill.teacher.c",
        "recall.distill.answer_generator",
        "recall.distill.utility_judge",
        "research.planner",
        "research.challenge",
        "research.tie_break",
        "research.deep_retrieval_requery",
        "knowledge.relation_extraction",
        "knowledge.community_summary",
        "ingest.generation",
        "lint.tag_repair",
        "lint.orphan_link",
        "recall.content_correction.proposer",
        "classification.primary",
        "classification.challenger",
        "classification.tie_break",
        "classification.anchor_set",
        "classification.decision",
        "classification.direct_decision",
        "classification.hierarchy",
        "classification.query",
        "classification.query_v2",
        "classification.anchor",
        "classification.anchor.primary",
        "classification.anchor.challenger",
        "classification.embedding",
    }
    knowledge_embedding = config.roles["knowledge.embedding"]
    assert knowledge_embedding.provider_id == "local"
    assert knowledge_embedding.model == "bge-m3"
    classification_embedding = config.roles["classification.embedding"]
    assert classification_embedding.provider_id == "local"
    assert classification_embedding.model == "bge-m3"
    assert [
        config.roles[role].model
        for role in (
            "classification.primary",
            "classification.challenger",
            "classification.tie_break",
        )
    ] == [
        "maxwell1500/ornith-35b:Q5_K_M",
        "muse-glimmer:30b-mxfp8-dflash",
        "gemma4:26b",
    ]
    text = example.read_text(encoding="utf-8")
    for role, data_class in (
        ("search.semantic.foreground", "raw"),
        ("search.semantic.foreground", "system"),
        ("search.semantic.incremental", "page"),
        ("search.semantic.incremental", "system"),
    ):
        assert f'# role = "{role}"\n# data_class = "{data_class}"' in text
    librarian = config.roles["librarian.review"]
    challenger = config.roles["librarian.review.challenger"]
    assert librarian.model != challenger.model
    answer_runner = config.roles["recall.answer.runner"]
    answer_scorer = config.roles["recall.answer.scorer"]
    assert answer_runner.model != answer_scorer.model
    certificate_primary = config.roles["recall.certificate_judge.primary"]
    certificate_escalation = config.roles["recall.certificate_judge.escalation"]
    assert certificate_primary.model != certificate_escalation.model
    auditor = config.roles["recall.auditor"]
    assert auditor.provider_id == "local"
    assert auditor.model == "maxwell1500/ornith-35b:Q5_K_M"
    recall_gate = config.roles["recall.gate"]
    recall_rewriter = config.roles["recall.query_rewriter"]
    assert recall_gate.provider_id == recall_rewriter.provider_id == "local"
    assert recall_gate.model == recall_rewriter.model == "ornith:9b-q4_K_M"
    proposer_roles = (
        "recall.policy_proposer.primary",
        "recall.policy_proposer.challenger",
    )
    proposer_routes = [config.roles[role] for role in proposer_roles]
    assert [route.model for route in proposer_routes] == [
        "maxwell1500/ornith-35b:Q5_K_M",
        "gemma4:26b",
    ]
    assert all(
        route.required_capabilities == ("structured_output",)
        for route in proposer_routes
    )
    for role in proposer_roles:
        assert f'# role = "{role}"\n# data_class = "raw"' in text
    rubric_variant = config.roles["recall.rubric.variant"]
    assert rubric_variant.provider_id == "local"
    assert rubric_variant.model == "gemma4:26b"
    assert rubric_variant.required_capabilities == ("structured_output",)
    assert (
        '# role = "recall.rubric.variant"\n# data_class = "raw"' in text
    )
    distill_roles = (
        "recall.distill.teacher.a",
        "recall.distill.teacher.b",
        "recall.distill.teacher.c",
        "recall.distill.answer_generator",
        "recall.distill.utility_judge",
    )
    distill_routes = [config.roles[role] for role in distill_roles]
    assert [route.provider_id for route in distill_routes] == ["local"] * 5
    assert all(
        route.required_capabilities == ("structured_output",)
        for route in distill_routes
    )
    assert (
        config.roles["recall.distill.answer_generator"].model
        != config.roles["recall.distill.utility_judge"].model
    )
    assert 'raw/high inputs must never\n# use a remote route' in text
    distillation = tomllib.loads(text)["recall"]["distillation"]
    assert distillation == {
        "enabled": False,
        "chunk_size": 25,
        "max_input_bytes": 24000,
        "max_candidates": 200,
        "hard_floor_rallies": 1000,
        "hard_floor_days": 30,
        "hard_floor_windows": 3,
        "hard_floor_verified_labels": 500,
        "hard_floor_per_class": 100,
        "rollout_stages": [5, 25, 100],
        "canary_min_days": 7,
    }
    assert [
        config.roles[role].model
        for role in (
            "research.planner",
            "research.challenge",
            "research.tie_break",
        )
    ] == [
        "maxwell1500/ornith-35b:Q5_K_M",
        "gpt-oss:20b",
        "gemma4:26b",
    ]
    deep_retrieval_requery = config.roles["research.deep_retrieval_requery"]
    assert deep_retrieval_requery.provider_id == "local"
    assert deep_retrieval_requery.model == "maxwell1500/ornith-35b:Q5_K_M"
    assert deep_retrieval_requery.required_capabilities == ("structured_output",)
    assert (
        '# role = "research.deep_retrieval_requery"\n# data_class = "raw"'
        in text
    )
    ingest_generation = config.roles["ingest.generation"]
    assert ingest_generation.provider_id == "local"
    assert ingest_generation.model == "maxwell1500/ornith-35b:Q5_K_M"
    assert config.roles["lint.tag_repair"].model == ingest_generation.model
    assert config.roles["lint.orphan_link"].model == ingest_generation.model
    assert config.roles["knowledge.relation_extraction"].model == "gemma4:26b"
    assert config.roles["knowledge.community_summary"].model == "gemma4:26b"
    assert (
        config.roles["recall.content_correction.proposer"].model
        == ingest_generation.model
    )
