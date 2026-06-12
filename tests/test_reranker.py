from __future__ import annotations

import json

from llm_wiki_mcp import reranker, server
from llm_wiki_mcp.reranker import RerankOutcome, rerank_results
from llm_wiki_mcp.runtime_config import RerankerConfig
from llm_wiki_mcp.search import ScoredPage


def page(page_id: str, score: float = 1.0) -> ScoredPage:
    return ScoredPage(
        page_id=page_id,
        title=page_id,
        folder="",
        updated="2026-06-11",
        score=score,
    )


def test_rerank_results_disabled_preserves_order() -> None:
    candidates = [page("a"), page("b")]

    outcome = rerank_results("query", candidates, config=RerankerConfig(enabled=False))

    assert outcome.results == candidates
    assert outcome.metadata["status"] == "disabled"


def test_rerank_results_applies_scores_without_touching_tail(monkeypatch) -> None:
    candidates = [page("a"), page("b"), page("c")]

    def fake_score_fn(config):
        def score(query, passages, cfg):
            assert query == "query"
            assert len(passages) == 2
            return [0.1, 0.9]

        return score

    monkeypatch.setattr(reranker, "_score_fn", fake_score_fn)
    monkeypatch.setattr(reranker, "find_page", lambda _page_id: None)

    outcome = rerank_results(
        "query",
        candidates,
        config=RerankerConfig(enabled=True, top_n=2, weight=2.0),
    )

    assert [result.page_id for result in outcome.results] == ["b", "a", "c"]
    assert outcome.metadata["status"] == "applied"
    assert outcome.metadata["candidate_count"] == 2


def test_rerank_results_unavailable_preserves_order() -> None:
    candidates = [page("a"), page("b")]

    outcome = rerank_results(
        "query",
        candidates,
        config=RerankerConfig(enabled=True, backend="missing"),
    )

    assert outcome.results == candidates
    assert outcome.metadata["status"] == "unavailable"


def test_wiki_search_uses_reranker_only_when_enabled(monkeypatch) -> None:
    class FakeStore:
        def refresh(self) -> None:
            pass

        def tags(self, page_id: str) -> list[str]:
            return []

        def outlinks(self, page_id: str) -> list[str]:
            return []

    def fake_search(**kwargs):
        assert kwargs["top_n"] == 20
        return [page("a"), page("b")], "hybrid"

    def fake_rerank(query, candidates, *, config):
        assert query == "needle"
        assert config.enabled is True
        return RerankOutcome(
            [candidates[1], candidates[0]],
            {
                "status": "applied",
                "model": config.model,
                "backend": config.backend,
                "candidate_count": 2,
                "weight": config.weight,
                "latency_ms": 3,
            },
        )

    from llm_wiki_mcp import search as search_mod
    from llm_wiki_mcp import runtime_config

    monkeypatch.setattr(search_mod, "search", fake_search)
    monkeypatch.setattr(runtime_config, "load_reranker_config", lambda: RerankerConfig(enabled=True))
    monkeypatch.setattr(reranker, "rerank_results", fake_rerank)
    monkeypatch.setattr(server, "get_store", lambda: FakeStore())
    monkeypatch.setattr(server, "find_page", lambda _page_id: None)

    tool_fn = server.wiki_search.fn if hasattr(server.wiki_search, "fn") else server.wiki_search
    payload = json.loads(tool_fn("needle", depth=0))

    assert payload["search_mode"] == "hybrid+rerank"
    assert payload["reranker"]["status"] == "applied"
    assert [hit["page_id"] for hit in payload["direct_hits"]] == ["b", "a"]


def test_wiki_search_reranks_after_tag_filter(monkeypatch) -> None:
    class FakeStore:
        def refresh(self) -> None:
            pass

        def tags(self, page_id: str) -> list[str]:
            return ["d/keep"] if page_id == "keep" else []

        def outlinks(self, page_id: str) -> list[str]:
            return []

    seen_candidates: list[list[str]] = []

    def fake_search(**kwargs):
        assert kwargs["top_n"] == 20
        return [page("keep"), page("drop")], "hybrid"

    def fake_rerank(query, candidates, *, config):
        seen_candidates.append([candidate.page_id for candidate in candidates])
        return RerankOutcome(
            candidates,
            {
                "status": "applied",
                "model": config.model,
                "backend": config.backend,
                "candidate_count": len(candidates),
                "weight": config.weight,
                "latency_ms": 2,
            },
        )

    from llm_wiki_mcp import search as search_mod
    from llm_wiki_mcp import runtime_config

    monkeypatch.setattr(search_mod, "search", fake_search)
    monkeypatch.setattr(runtime_config, "load_reranker_config", lambda: RerankerConfig(enabled=True))
    monkeypatch.setattr(reranker, "rerank_results", fake_rerank)
    monkeypatch.setattr(server, "get_store", lambda: FakeStore())
    monkeypatch.setattr(server, "find_page", lambda _page_id: None)

    tool_fn = server.wiki_search.fn if hasattr(server.wiki_search, "fn") else server.wiki_search
    payload = json.loads(tool_fn("needle", depth=0, tags=["d/keep"]))

    assert seen_candidates == [["keep"]]
    assert payload["reranker"]["status"] == "applied"
    assert [hit["page_id"] for hit in payload["direct_hits"]] == ["keep"]
