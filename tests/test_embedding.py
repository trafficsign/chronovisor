"""Tests for provider-neutral knowledge embeddings and route-bound caching."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from chronovisor.core import embedding as emb_mod
from chronovisor.core.embedding import (
    EMBEDDING_ROLE,
    cosine,
    embed_text,
    embed_texts,
    most_similar,
)
from chronovisor.core.llm_runtime import (
    BackendContractError,
    EgressDeniedError,
    EmbeddingPurpose,
    EmbeddingRequest,
    EmbeddingResult,
    EmbeddingRoute,
    LLMRuntime,
    RouteLocation,
    SourceDataClass,
    SourceSensitivity,
)


class FakeEmbeddingBackend:
    def __init__(
        self,
        vectors: dict[str, list[float]],
        *,
        provider: str = "ollama",
        location: RouteLocation = RouteLocation.LOCAL,
    ) -> None:
        self.vectors = vectors
        self.provider = provider
        self.location = location
        self.calls: list[EmbeddingRequest] = []

    def embed(self, request: EmbeddingRequest, *, model: str) -> EmbeddingResult:
        self.calls.append(request)
        return EmbeddingResult(
            tuple(tuple(self.vectors[text]) for text in request.texts),
            self.provider,
            model,
        )


@pytest.fixture()
def isolated_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    cache_dir = tmp_path / ".embeddings"
    monkeypatch.setattr(emb_mod, "_CACHE_DIR", cache_dir)
    return cache_dir


def install_runtime(
    monkeypatch: pytest.MonkeyPatch,
    backend: FakeEmbeddingBackend,
    *,
    model: str = "bge-m3",
    allow_egress: bool = False,
) -> LLMRuntime:
    runtime = LLMRuntime(
        embedding={EMBEDDING_ROLE: EmbeddingRoute(backend, model)},
        remote_egress_opt_ins=(
            {(EMBEDDING_ROLE, SourceDataClass.DERIVED_SNIPPET)} if allow_egress else ()
        ),
    )
    monkeypatch.setattr(emb_mod, "load_default_llm_runtime", lambda: runtime)
    return runtime


def install_local_ollama(
    monkeypatch: pytest.MonkeyPatch,
    vectors: dict[str, list[float]],
    *,
    digest: str = "digest-v1",
) -> FakeEmbeddingBackend:
    backend = FakeEmbeddingBackend(vectors)
    install_runtime(monkeypatch, backend)
    monkeypatch.setattr(
        emb_mod.ollama,
        "model_digests",
        lambda models: {model: digest for model in models},
    )
    return backend


def forbid_ollama(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    calls = {
        name: 0
        for name in (
            "embed",
            "model_digests",
            "model_resource_lease",
            "plan_model_residency",
            "resident_model_rows",
            "unload_named_model",
        )
    }

    def forbidden(name: str):
        def call(*_args: object, **_kwargs: object) -> None:
            calls[name] += 1
            raise AssertionError(f"remote embedding touched ollama.{name}")

        return call

    for name in calls:
        monkeypatch.setattr(emb_mod.ollama, name, forbidden(name))
    return calls


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        ([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], 1.0),
        ([1.0, 0.0], [-1.0, 0.0], -1.0),
        ([1.0, 0.0], [0.0, 1.0], 0.0),
        ([0.0, 0.0], [1.0, 1.0], 0.0),
        ([], [1.0], 0.0),
    ],
)
def test_cosine(left: list[float], right: list[float], expected: float) -> None:
    assert cosine(left, right) == pytest.approx(expected, abs=1e-9)


def test_local_ollama_route_digest_and_cache_are_exact(
    isolated_cache: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = install_local_ollama(monkeypatch, {"hello": [0.1, 0.2, 0.3]})

    result = embed_texts(["hello"], return_route=True)
    assert isinstance(result, tuple)
    first, route = result
    second = embed_text("hello")

    assert first == [second] == [[0.1, 0.2, 0.3]]
    assert route == {
        "role": EMBEDDING_ROLE,
        "provider": "ollama",
        "model": "bge-m3",
        "location": "local",
        "model_digest": "digest-v1",
    }
    assert len(backend.calls) == 1
    assert json.loads(emb_mod._cache_path("hello", route).read_text()) == first[0]


def test_cache_v2_separates_route_digest_purpose_and_legacy_key(
    isolated_cache: Path,
) -> None:
    route = {
        "role": EMBEDDING_ROLE,
        "provider": "ollama",
        "model": "bge-m3",
        "location": "local",
        "model_digest": "digest-v1",
    }
    paths = {
        emb_mod._cache_path("text", route),
        emb_mod._cache_path("text", {**route, "model_digest": "digest-v2"}),
        emb_mod._cache_path("text", route, EmbeddingPurpose.QUERY),
        emb_mod._cache_path("text", {**route, "provider": "remote"}),
    }
    legacy = hashlib.sha256(b"bge-m3|text").hexdigest()

    assert len(paths) == 4
    assert all(path.stem != legacy for path in paths)


def test_corrupt_cache_falls_back_to_runtime(
    isolated_cache: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = install_local_ollama(monkeypatch, {"hello": [1.0, 2.0]})
    route = emb_mod._route_identity(
        emb_mod.load_default_llm_runtime().resolve_embedding(EMBEDDING_ROLE)
    )
    path = emb_mod._cache_path("hello", route)
    path.parent.mkdir(parents=True)
    path.write_text("not-json{{{")

    assert embed_text("hello") == [1.0, 2.0]
    assert len(backend.calls) == 1


def test_batch_requests_only_uncached_documents_with_fixed_source(
    isolated_cache: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = install_local_ollama(
        monkeypatch,
        {"a": [1.0], "bb": [2.0], "ccc": [3.0]},
    )
    route = emb_mod._route_identity(
        emb_mod.load_default_llm_runtime().resolve_embedding(EMBEDDING_ROLE)
    )
    cached = emb_mod._cache_path("a", route)
    cached.parent.mkdir(parents=True)
    cached.write_text(json.dumps([9.0]))

    assert embed_texts(["a", "bb", "ccc"]) == [[9.0], [2.0], [3.0]]
    request = backend.calls[0]
    assert request.texts == ("bb", "ccc")
    assert request.source.data_class is SourceDataClass.DERIVED_SNIPPET
    assert request.source.sensitivity is SourceSensitivity.HIGH
    assert request.purpose is EmbeddingPurpose.DOCUMENT


def test_empty_batch_does_not_resolve_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        emb_mod,
        "load_default_llm_runtime",
        lambda: (_ for _ in ()).throw(AssertionError("runtime must not load")),
    )

    assert embed_texts([]) == []


def test_most_similar_uses_query_and_document_purposes(
    isolated_cache: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = install_local_ollama(
        monkeypatch,
        {
            "query": [1.0, 0.0],
            "near": [0.9, 0.1],
            "far": [0.0, 1.0],
        },
    )
    digest_calls = 0

    def model_digests(models: list[str]) -> dict[str, str]:
        nonlocal digest_calls
        digest_calls += 1
        return {model: "digest-v1" for model in models}

    monkeypatch.setattr(emb_mod.ollama, "model_digests", model_digests)

    result = most_similar("query", ["near", "far"], threshold=0.8)

    assert result is not None and result[0] == "near" and result[1] > 0.99
    assert [request.purpose for request in backend.calls] == [
        EmbeddingPurpose.QUERY,
        EmbeddingPurpose.DOCUMENT,
    ]
    assert digest_calls == 1


def test_most_similar_returns_none_below_threshold_or_without_candidates(
    isolated_cache: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = install_local_ollama(
        monkeypatch,
        {"query": [1.0, 0.0], "far": [0.0, 1.0]},
    )

    assert most_similar("query", ["far"], threshold=0.5) is None
    assert most_similar("query", [], threshold=0.5) is None
    assert len(backend.calls) == 2


def test_missing_local_ollama_digest_fails_before_embedding(
    isolated_cache: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = install_local_ollama(monkeypatch, {"text": [1.0]}, digest="")

    with pytest.raises(BackendContractError) as error:
        embed_text("text")

    assert error.value.reason == "model_digest_missing"
    assert backend.calls == []


def test_local_nemotron_route_has_no_ollama_digest(
    isolated_cache: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = FakeEmbeddingBackend(
        {"text": [1.0]}, provider="nemotron", location=RouteLocation.LOCAL
    )
    install_runtime(monkeypatch, backend, model="nemotron-model")
    ollama_calls = forbid_ollama(monkeypatch)

    result = embed_texts(["text"], return_route=True)
    assert isinstance(result, tuple)
    vectors, route = result

    assert vectors == [[1.0]]
    assert route["model_digest"] is None
    assert route["provider"] == "nemotron"
    assert ollama_calls == dict.fromkeys(ollama_calls, 0)


def test_remote_high_source_requires_explicit_egress_without_ollama(
    isolated_cache: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = FakeEmbeddingBackend(
        {"text": [1.0]}, provider="remote-test", location=RouteLocation.REMOTE
    )
    install_runtime(monkeypatch, backend)
    ollama_calls = forbid_ollama(monkeypatch)

    with pytest.raises(EgressDeniedError):
        embed_text("text")

    assert backend.calls == []
    assert ollama_calls == dict.fromkeys(ollama_calls, 0)


def test_remote_opt_in_embeds_without_ollama_controls(
    isolated_cache: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = FakeEmbeddingBackend(
        {"text": [1.0]}, provider="remote-test", location=RouteLocation.REMOTE
    )
    install_runtime(monkeypatch, backend, allow_egress=True)
    ollama_calls = forbid_ollama(monkeypatch)

    result = embed_texts(["text"], return_route=True)
    assert isinstance(result, tuple)
    vectors, route = result

    assert vectors == [[1.0]]
    assert route == {
        "role": EMBEDDING_ROLE,
        "provider": "remote-test",
        "model": "bge-m3",
        "location": "remote",
        "model_digest": None,
    }
    assert len(backend.calls) == 1
    assert backend.calls[0].source.sensitivity is SourceSensitivity.HIGH
    assert ollama_calls == dict.fromkeys(ollama_calls, 0)
