from __future__ import annotations

import pytest

from chronovisor.core import search, semantic_client, semantic_jobs
from chronovisor.core.runtime_config import SearchEmbeddingConfig
from chronovisor.core.search_types import ScoredPage


def _config(*, mode: str = "on", enabled: bool = True) -> SearchEmbeddingConfig:
    return SearchEmbeddingConfig(
        enabled=enabled,
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


def test_disabled_semantic_search_is_the_explicit_bm25_only_mode(monkeypatch) -> None:
    monkeypatch.setattr(
        search, "load_search_embedding_config", lambda: _config(enabled=False)
    )
    assert search.semantic_search("query") == []
    assert search.update_embeddings() == 0
