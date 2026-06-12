from __future__ import annotations

import pytest

from llm_wiki_mcp import search, search_eval
from llm_wiki_mcp.reranker import RerankOutcome
from llm_wiki_mcp.runtime_config import NegativeFeedbackConfig, RerankerConfig
from llm_wiki_mcp.search import ScoredPage


def page(page_id: str, score: float, *, status: str = "active") -> ScoredPage:
    return ScoredPage(
        page_id=page_id,
        title=page_id,
        folder="",
        updated="2026-06-12",
        score=score,
        status=status,
    )


class FakeBM25:
    def __init__(self, results: list[ScoredPage]) -> None:
        self.results = results
        self.built = False
        self.queries: list[tuple[str, int]] = []

    def build(self) -> None:
        self.built = True

    def query(self, query: str, top_n: int = 20) -> list[ScoredPage]:
        self.queries.append((query, top_n))
        return self.results[:top_n]


def disabled_negative_feedback() -> NegativeFeedbackConfig:
    return NegativeFeedbackConfig(enabled=False)


def test_production_search_calls_zero_weight_graph_and_skips_usage_prior(monkeypatch) -> None:
    bm25 = FakeBM25([page("bm25-a", 10.0), page("bm25-b", 9.0)])
    graph_calls: list[dict[str, object]] = []

    def fake_graph(results: list[ScoredPage], *, decay: float = 0.5, limit: int = 50):
        graph_calls.append(
            {
                "page_ids": [result.page_id for result in results],
                "decay": decay,
                "limit": limit,
            }
        )
        return []

    def fail_usage(*args, **kwargs):
        raise AssertionError("usage prior must stay gated while its weight is zero")

    monkeypatch.setattr(search, "get_bm25", lambda: bm25)
    monkeypatch.setattr(search, "semantic_search", lambda query, top_n=20: [page("sem-a", 0.8)])
    monkeypatch.setattr(search, "graph_expand_results", fake_graph)
    monkeypatch.setattr(search, "usage_prior_results", fail_usage)
    monkeypatch.setattr(search, "load_negative_feedback_config", disabled_negative_feedback)

    results, search_mode = search.search("query", top_n=3, semantic=True)

    assert bm25.built is True
    assert bm25.queries == [("query", 100)]
    assert search_mode == "hybrid"
    assert graph_calls == [
        {"page_ids": ["bm25-a", "bm25-b", "sem-a"], "decay": 0.0, "limit": 100}
    ]
    assert [(result.page_id, result.score) for result in results] == pytest.approx(
        [
            ("bm25-a", 0.027666666666666666),
            ("bm25-b", 0.02089344262295082),
            ("sem-a", 0.01),
        ]
    )


def test_production_search_builds_usage_prior_only_when_weight_is_positive(monkeypatch) -> None:
    bm25 = FakeBM25([page("bm25-a", 10.0)])
    usage_calls: list[dict[str, object]] = []

    def fake_usage(candidate_ids, *, limit: int = 50, decay: float = 0.98, cap: float = 3.0):
        usage_calls.append(
            {
                "candidate_ids": set(candidate_ids),
                "limit": limit,
                "decay": decay,
                "cap": cap,
            }
        )
        return [page("graph-a", 3.0)]

    monkeypatch.setattr(search, "get_bm25", lambda: bm25)
    monkeypatch.setattr(search, "semantic_search", lambda query, top_n=20: [])
    monkeypatch.setattr(search, "graph_expand_results", lambda results, **kwargs: [page("graph-a", 5.0)])
    monkeypatch.setattr(search, "usage_prior_results", fake_usage)
    monkeypatch.setattr(search, "load_negative_feedback_config", disabled_negative_feedback)

    results, search_mode = search.search(
        "query",
        top_n=2,
        semantic=False,
        fusion_weights={"usage_prior": 0.2},
    )

    assert search_mode == "bm25+graph"
    assert usage_calls == [
        {
            "candidate_ids": {"bm25-a", "graph-a"},
            "limit": 100,
            "decay": 0.98,
            "cap": 3.0,
        }
    ]
    assert [result.page_id for result in results] == ["bm25-a", "graph-a"]


def test_eval_hybrid_current_omits_graph_and_usage_channels(monkeypatch) -> None:
    bm25 = FakeBM25([page("bm25-a", 10.0), page("bm25-b", 9.0)])

    def fail_graph(*args, **kwargs):
        raise AssertionError("hybrid-current currently does not construct graph candidates")

    def fail_usage(*args, **kwargs):
        raise AssertionError("hybrid-current currently does not construct usage-prior candidates")

    monkeypatch.setattr(search_eval, "get_bm25", lambda: bm25)
    monkeypatch.setattr(search_eval, "semantic_search", lambda query, top_n=20: [page("sem-a", 0.8)])
    monkeypatch.setattr(search_eval, "graph_expand_results", fail_graph)
    monkeypatch.setattr(search_eval, "usage_prior_results", fail_usage)
    monkeypatch.setattr(search_eval, "load_negative_feedback_config", disabled_negative_feedback)

    payload = search_eval.run_variant("query", "hybrid-current", top_n=3)

    assert [result.page_id for result in payload["results"]] == ["bm25-a", "bm25-b", "sem-a"]
    assert payload["channels"]["graph"] == []
    assert payload["channels"]["usage_prior"] == []


def test_eval_hybrid_rerank_applies_negative_feedback_after_rerank(monkeypatch) -> None:
    bm25 = FakeBM25([page("a", 3.0), page("b", 2.0)])
    penalty_inputs: list[list[str]] = []

    def fake_rerank(query, candidates, *, config):
        assert query == "query"
        assert config.enabled is True
        return RerankOutcome(
            [candidates[1], candidates[0]],
            {"status": "applied", "candidate_count": len(candidates)},
        )

    def fake_apply_penalties(results, penalties):
        penalty_inputs.append([result.page_id for result in results])
        return results

    monkeypatch.setattr(search_eval, "get_bm25", lambda: bm25)
    monkeypatch.setattr(search_eval, "semantic_search", lambda query, top_n=20: [])
    monkeypatch.setattr(search_eval, "load_reranker_config", lambda: RerankerConfig(enabled=True))
    monkeypatch.setattr(search_eval, "rerank_results", fake_rerank)
    monkeypatch.setattr(
        search_eval,
        "load_negative_feedback_config",
        lambda: NegativeFeedbackConfig(enabled=True),
    )
    monkeypatch.setattr(search_eval, "penalties_for_query", lambda query, config: {"a": 0.5})
    monkeypatch.setattr(search_eval, "apply_penalties", fake_apply_penalties)

    payload = search_eval.run_variant("query", "hybrid-rerank", top_n=2)

    assert penalty_inputs == [["b", "a"]]
    assert [result.page_id for result in payload["results"]] == ["b", "a"]
    assert payload["channels"]["reranker"]["status"] == "applied"
    assert payload["channels"]["negative_feedback"] == {"status": "applied", "pages": ["a"]}


def test_run_weighted_hybrid_uses_bm25_and_semantic_only(monkeypatch) -> None:
    bm25 = FakeBM25([page("bm25-a", 10.0)])

    def fail_graph(*args, **kwargs):
        raise AssertionError("run_weighted_hybrid currently has no graph channel")

    def fail_usage(*args, **kwargs):
        raise AssertionError("run_weighted_hybrid currently has no usage-prior channel")

    monkeypatch.setattr(search_eval, "get_bm25", lambda: bm25)
    monkeypatch.setattr(search_eval, "semantic_search", lambda query, top_n=20: [page("sem-a", 0.8)])
    monkeypatch.setattr(search_eval, "graph_expand_results", fail_graph)
    monkeypatch.setattr(search_eval, "usage_prior_results", fail_usage)

    payload = search_eval.run_weighted_hybrid(
        "query",
        {**search_eval.DEFAULT_FUSION_WEIGHTS, "bm25_score_bonus": 0.0},
        top_n=2,
    )

    assert [result.page_id for result in payload["results"]] == ["bm25-a", "sem-a"]
