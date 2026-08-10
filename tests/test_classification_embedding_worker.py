from __future__ import annotations

from typing import Any

import pytest

from chronovisor.core.llm_runtime import (
    BackendContractError,
    EgressDeniedError,
    EmbeddingRequest,
    EmbeddingResult,
    EmbeddingRoute,
    LLMRuntime,
    RouteLocation,
    SourceDataClass,
    SourceSensitivity,
)
from chronovisor.recall import classification_embedding_worker as worker
from chronovisor.recall.classification import ClassificationError


class Backend:
    def __init__(self, provider: str, location: RouteLocation) -> None:
        self.provider = provider
        self.location = location
        self.calls: list[EmbeddingRequest] = []

    def embed(self, request: EmbeddingRequest, *, model: str) -> EmbeddingResult:
        self.calls.append(request)
        return EmbeddingResult(
            tuple((float(index), 1.0) for index, _text in enumerate(request.texts)),
            self.provider,
            model,
        )


def payload(**overrides: Any) -> dict[str, Any]:
    return {
        "schema": worker.SCHEMA,
        "texts": ["one", "two"],
        "source_data_class": "derived_snippet",
        "source_sensitivity": "normal",
        "embedding_purpose": "document",
        "read_timeout_ms": 30_000,
        **overrides,
    }


def install_runtime(
    monkeypatch: pytest.MonkeyPatch,
    backend: Backend,
    *,
    model: str = "embedding-model",
) -> LLMRuntime:
    runtime = LLMRuntime(
        embedding={worker.RUNTIME_ROLE: EmbeddingRoute(backend, model)}
    )
    monkeypatch.setattr(worker.llm_config, "load_default_llm_runtime", lambda: runtime)
    return runtime


def forbid_ollama(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    calls = dict.fromkeys(
        (
            "embed",
            "model_digests",
            "model_resource_lease",
            "plan_model_residency",
            "resident_model_rows",
            "unload_named_model",
        ),
        0,
    )

    def forbidden(name: str):
        def call(*_args: object, **_kwargs: object) -> None:
            calls[name] += 1
            raise AssertionError(f"worker touched ollama.{name}")

        return call

    for name in calls:
        monkeypatch.setattr(worker.ollama, name, forbidden(name))
    return calls


def test_remote_normal_uses_fixed_route_and_captures_request_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = Backend("remote-test", RouteLocation.REMOTE)
    install_runtime(monkeypatch, backend)
    ollama_calls = forbid_ollama(monkeypatch)

    result = worker.run(
        payload(
            source_data_class="page",
            embedding_purpose="query",
            read_timeout_ms=12_345,
        )
    )

    assert result["route_identity"] == {
        "role": worker.RUNTIME_ROLE,
        "provider": "remote-test",
        "model": "embedding-model",
        "location": "remote",
        "model_digest": None,
    }
    request = backend.calls[0]
    assert request.texts == ("one", "two")
    assert request.source.data_class is SourceDataClass.PAGE
    assert request.source.sensitivity is SourceSensitivity.NORMAL
    assert request.purpose.value == "query"
    assert request.timeout_ms == 12_345
    assert ollama_calls == dict.fromkeys(ollama_calls, 0)


def test_remote_high_is_denied_before_backend_without_ollama(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = Backend("remote-test", RouteLocation.REMOTE)
    install_runtime(monkeypatch, backend)
    ollama_calls = forbid_ollama(monkeypatch)

    with pytest.raises(EgressDeniedError):
        worker.run(payload(source_sensitivity="high"))

    assert backend.calls == []
    assert ollama_calls == dict.fromkeys(ollama_calls, 0)


def test_local_non_ollama_has_null_digest_without_ollama_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = Backend("nemotron", RouteLocation.LOCAL)
    install_runtime(monkeypatch, backend, model="nemotron-embed")
    ollama_calls = forbid_ollama(monkeypatch)

    result = worker.run(payload())

    assert result["route_identity"]["model_digest"] is None
    assert len(backend.calls) == 1
    assert ollama_calls == dict.fromkeys(ollama_calls, 0)


def test_local_ollama_digest_is_resolved_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = Backend("ollama", RouteLocation.LOCAL)
    install_runtime(monkeypatch, backend, model="bge-m3")
    digest_calls = 0

    def digests(models: list[str]) -> dict[str, str]:
        nonlocal digest_calls
        digest_calls += 1
        assert models == ["bge-m3"]
        return {"bge-m3": "sha256:exact"}

    monkeypatch.setattr(worker.ollama, "model_digests", digests)

    result = worker.run(payload())

    assert result["route_identity"]["model_digest"] == "sha256:exact"
    assert digest_calls == 1
    assert len(backend.calls) == 1


def test_missing_local_ollama_digest_fails_before_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = Backend("ollama", RouteLocation.LOCAL)
    install_runtime(monkeypatch, backend, model="bge-m3")
    monkeypatch.setattr(worker.ollama, "model_digests", lambda _models: {})

    with pytest.raises(BackendContractError) as error:
        worker.run(payload())

    assert error.value.reason == "model_digest_missing"
    assert backend.calls == []


@pytest.mark.parametrize(
    "override",
    [
        {"model": "override"},
        {"provider": "override"},
        {"runtime_role": "knowledge.embedding"},
        {"model_digest": "override"},
        {"source_data_class": "private"},
        {"source_sensitivity": "personal"},
        {"embedding_purpose": "classification"},
        {"read_timeout_ms": True},
        {"read_timeout_ms": 0},
    ],
)
def test_invalid_payload_fails_before_runtime_or_ollama(
    monkeypatch: pytest.MonkeyPatch,
    override: dict[str, Any],
) -> None:
    resolves = 0

    def runtime() -> None:
        nonlocal resolves
        resolves += 1
        raise AssertionError("invalid payload resolved runtime")

    monkeypatch.setattr(worker.llm_config, "load_default_llm_runtime", runtime)
    ollama_calls = forbid_ollama(monkeypatch)

    with pytest.raises(ClassificationError):
        worker.run(payload(**override))

    assert resolves == 0
    assert ollama_calls == dict.fromkeys(ollama_calls, 0)
