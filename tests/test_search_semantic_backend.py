from __future__ import annotations

import pytest

from chronovisor.core import ollama, search, semantic_client, semantic_jobs
from chronovisor.core.llm_runtime import (
    EgressDeniedError,
    EmbeddingRequest,
    EmbeddingResult,
    EmbeddingRoute,
    LLMRuntime,
    RouteLocation,
    SourceDataClass,
    SourceSensitivity,
)
from chronovisor.core.runtime_config import SearchEmbeddingConfig
from chronovisor.core.search_types import ScoredPage


def _config(
    *, backend: str = "nemotron_service", mode: str = "on", enabled: bool = True
) -> SearchEmbeddingConfig:
    return SearchEmbeddingConfig(
        enabled=enabled,
        backend=backend,
        rollout_mode=mode,
        canary_percent=100,
    )


def test_nemotron_search_keeps_domain_service_topology(monkeypatch) -> None:
    expected = [
        ScoredPage(
            page_id="p",
            title="P",
            folder="ai",
            updated="2026-07-24",
            score=0.9,
            status="stable",
        )
    ]
    monkeypatch.setattr(search, "load_search_embedding_config", lambda: _config())
    monkeypatch.setattr(semantic_client, "search", lambda *args, **kwargs: expected)

    assert search.semantic_search("query") == expected


def test_nemotron_provider_failure_does_not_fall_back_to_bm25(monkeypatch) -> None:
    monkeypatch.setattr(search, "load_search_embedding_config", lambda: _config())

    def broken(*args, **kwargs):
        raise OSError("service down")

    monkeypatch.setattr(semantic_client, "search", broken)

    with pytest.raises(OSError, match="service down"):
        search.semantic_search("query")


def test_nemotron_verify_failure_is_not_hidden(monkeypatch) -> None:
    monkeypatch.setattr(search, "load_search_embedding_config", lambda: _config())
    monkeypatch.setattr(semantic_client, "verify", lambda *args, **kwargs: 1 / 0)

    with pytest.raises(ZeroDivisionError):
        search.semantic_verify("query", ["p"])


def test_nemotron_updates_are_durable_jobs(monkeypatch) -> None:
    monkeypatch.setattr(search, "load_search_embedding_config", lambda: _config())
    monkeypatch.setattr("chronovisor.core.store.find_page", lambda _page_id: None)
    captured: dict[str, object] = {}

    def fake_enqueue(page_ids, *, source_hashes):
        captured["page_ids"] = page_ids
        captured["source_hashes"] = source_hashes
        return ["job"]

    monkeypatch.setattr(semantic_jobs, "enqueue_pages", fake_enqueue)

    assert search.update_embeddings(["b", "a", "a"]) == 2
    assert captured == {
        "page_ids": ["a", "b"],
        "source_hashes": {"a": "", "b": ""},
    }


class _EmbeddingBackend:
    provider = "test"
    location = RouteLocation.LOCAL

    def __init__(self) -> None:
        self.requests: list[EmbeddingRequest] = []

    def embed(self, request: EmbeddingRequest, *, model: str) -> EmbeddingResult:
        self.requests.append(request)
        return EmbeddingResult(((1.0, 0.0),), self.provider, model)


class _Store:
    def refresh(self) -> None:
        pass

    def meta(self, page_id: str) -> dict[str, object]:
        return {
            "title": page_id,
            "updated": "2026-08-10",
            "path": f"/wiki/pages/{page_id}.md",
            "status": "stable",
            "superseded_by": "",
        }


def _runtime(backend: _EmbeddingBackend) -> LLMRuntime:
    return LLMRuntime(
        embedding={"search.semantic": EmbeddingRoute(backend, "embed-model")}
    )


def _prepare_runtime_search(monkeypatch, runtime: LLMRuntime) -> None:
    monkeypatch.setattr(
        search,
        "load_search_embedding_config",
        lambda: _config(backend="legacy_ollama"),
    )
    monkeypatch.setattr(search, "load_default_llm_runtime", lambda: runtime)
    monkeypatch.setattr(
        search, "_search_embedding_profile", lambda: ("embed-model", "", "")
    )
    monkeypatch.setattr(search, "_embedding_count", lambda **_kwargs: 1)
    monkeypatch.setattr(
        search,
        "_iter_all_embeddings",
        lambda **_kwargs: [("p", [1.0, 0.0], 0.0, 1.0)],
    )
    monkeypatch.setattr(search, "_iter_all_question_embeddings", lambda **_kwargs: [])
    monkeypatch.setattr("chronovisor.core.index_store.get_store", lambda: _Store())


def test_in_process_search_uses_runtime_role_and_fail_closed_query_source(
    monkeypatch,
) -> None:
    backend = _EmbeddingBackend()
    _prepare_runtime_search(monkeypatch, _runtime(backend))

    assert [page.page_id for page in search.semantic_search("query")] == ["p"]
    request = backend.requests[0]
    assert request.source.data_class is SourceDataClass.RAW
    assert request.source.sensitivity is SourceSensitivity.HIGH


def test_remote_query_is_denied_before_backend_or_local_runtime_touch(
    monkeypatch,
) -> None:
    backend = _EmbeddingBackend()
    backend.location = RouteLocation.REMOTE
    _prepare_runtime_search(monkeypatch, _runtime(backend))

    def local_touch(*args, **kwargs):
        raise AssertionError("local runtime must not be touched")

    monkeypatch.setattr(ollama, "embed", local_touch)
    monkeypatch.setattr(ollama, "model_resource_lease", local_touch)
    monkeypatch.setattr(ollama, "resident_model_rows", local_touch)

    with pytest.raises(EgressDeniedError):
        search.semantic_search("unclassified query")
    assert backend.requests == []


def test_disabled_semantic_search_is_the_explicit_bm25_only_mode(monkeypatch) -> None:
    monkeypatch.setattr(
        search, "load_search_embedding_config", lambda: _config(enabled=False)
    )
    monkeypatch.setattr(
        search,
        "load_default_llm_runtime",
        lambda: (_ for _ in ()).throw(AssertionError("runtime must stay disabled")),
    )

    assert search.semantic_search("query") == []
    assert search.update_embeddings() == 0
