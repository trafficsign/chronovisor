from chronovisor.core.runtime_config import SearchEmbeddingConfig
from chronovisor.core.search_types import ScoredPage
from chronovisor.search import search, semantic_client, semantic_jobs


def _config(*, mode: str = "on") -> SearchEmbeddingConfig:
    return SearchEmbeddingConfig(
        enabled=True,
        backend="nemotron_service",
        rollout_mode=mode,
        canary_percent=100,
    )


def test_nemotron_search_uses_service(monkeypatch) -> None:
    expected = [
        ScoredPage(
            page_id="p",
            title="P",
            folder="ai",
            updated="2026-07-24",
            score=0.9,
        )
    ]
    monkeypatch.setattr(search, "load_search_embedding_config", lambda: _config())
    monkeypatch.setattr(semantic_client, "search", lambda *args, **kwargs: expected)
    assert search.semantic_search("query") == expected


def test_nemotron_failure_returns_empty_for_bm25_fallback(monkeypatch) -> None:
    monkeypatch.setattr(search, "load_search_embedding_config", lambda: _config())

    def broken(*args, **kwargs):
        raise OSError("service down")

    monkeypatch.setattr(semantic_client, "search", broken)
    assert search.semantic_search("query") == []


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
