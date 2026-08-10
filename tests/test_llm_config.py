from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.request import Request

import httpx
import pytest

from chronovisor.core.anthropic_adapter import AnthropicMessagesAdapter
from chronovisor.core.llm_config import (
    LLMConfigError,
    LLMConfigFailureCategory,
    build_llm_runtime,
    load_llm_config,
    parse_llm_config,
)
from chronovisor.core.llm_runtime import (
    EmbeddingRequest,
    GenerationRequest,
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
from chronovisor.core.ollama_adapter import OllamaAdapter
from chronovisor.core.openai_compatible_adapter import OpenAICompatibleAdapter
from chronovisor.core.provider_profiles import CURATED_PROFILE_IDS, ProviderProfile
from chronovisor.core.reranker import LocalRerankBackend

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
        self.calls: list[tuple[Request, bool]] = []

    def __call__(self, request: Request, *, follow_redirects: bool) -> object:
        self.calls.append((request, follow_redirects))
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
    assert resolver.calls == []


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
    assert all(not follow_redirects for _, follow_redirects in sender.calls)
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


@pytest.mark.parametrize("kind", CURATED_PROFILE_IDS)
def test_every_curated_provider_parses_with_conservative_generation(kind: str) -> None:
    config = parse_llm_config(_remote_payload(kind=kind))

    provider = config.providers["remote"]
    assert provider.profile is not None
    assert provider.profile.capabilities.generation
    assert not provider.profile.capabilities.structured_output
    assert not provider.profile.capabilities.tools


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
    config = load_llm_config(Path(__file__).parents[1] / "config.toml.example")

    assert set(config.roles) >= {
        "search.semantic",
        "search.rerank",
        "librarian.review",
        "classification.primary",
        "classification.challenger",
        "classification.tie_break",
    }
