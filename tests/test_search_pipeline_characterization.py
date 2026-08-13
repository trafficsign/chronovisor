from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from chronovisor.core import pipeline as pipeline_mod
from chronovisor.core import search
from chronovisor.core.reranker import RerankOutcome
from chronovisor.core.runtime_config import NegativeFeedbackConfig, RerankerConfig
from chronovisor.core.search import ScoredPage
from chronovisor.search import search_eval


def page(page_id: str, score: float, *, status: str = "stable") -> ScoredPage:
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


@pytest.fixture()
def isolated_selection_index(monkeypatch) -> None:
    from chronovisor.recall import contextual_suppression

    monkeypatch.setattr(contextual_suppression, "ranking_components", lambda *_args: {})


def test_pipeline_module_does_not_import_upper_layers() -> None:
    source = Path(pipeline_mod.__file__).read_text(encoding="utf-8")

    assert "chronovisor.hosts.server" not in source
    assert "chronovisor.search.search_eval" not in source


def test_production_search_calls_bounded_graph_and_skips_usage_prior(
    monkeypatch,
) -> None:
    from chronovisor.core import retention

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
    monkeypatch.setattr(
        search, "semantic_search", lambda query, top_n=20: [page("sem-a", 0.8)]
    )
    monkeypatch.setattr(search, "graph_expand_results", fake_graph)
    monkeypatch.setattr(search, "usage_prior_results", fail_usage)
    monkeypatch.setattr(
        search, "load_negative_feedback_config", disabled_negative_feedback
    )
    monkeypatch.setattr(
        search,
        "load_active_fusion_weights",
        lambda: dict(search.DEFAULT_FUSION_WEIGHTS),
    )
    monkeypatch.setattr(retention, "retention_score", lambda _page_id: 0.0)

    results, search_mode = search.search("query", top_n=3, semantic=True)

    assert bm25.built is True
    assert bm25.queries == [("query", 100)]
    assert search_mode == "hybrid"
    assert graph_calls == [
        {"page_ids": ["bm25-a", "bm25-b", "sem-a"], "decay": 0.3, "limit": 100}
    ]
    assert [result.page_id for result in results] == [
        "bm25-a",
        "bm25-b",
        "sem-a",
    ]
    assert [result.score for result in results] == pytest.approx(
        [
            0.027666666666666666,
            0.02089344262295082,
            0.01,
        ]
    )


def test_production_search_reports_anonymous_stage_timings(monkeypatch) -> None:
    from chronovisor.core import retention

    bm25 = FakeBM25([page("bm25-a", 10.0)])

    monkeypatch.setattr(search, "get_bm25", lambda: bm25)
    monkeypatch.setattr(
        search, "semantic_search", lambda query, top_n=20: [page("sem-a", 0.8)]
    )
    monkeypatch.setattr(search, "graph_expand_results", lambda results, **kwargs: [])
    monkeypatch.setattr(
        search, "load_negative_feedback_config", disabled_negative_feedback
    )
    monkeypatch.setattr(
        search,
        "load_active_fusion_weights",
        lambda: dict(search.DEFAULT_FUSION_WEIGHTS),
    )
    monkeypatch.setattr(retention, "retention_score", lambda _page_id: 0.0)

    search.search("query", top_n=3, semantic=True)
    trace = search.last_search_trace()
    timings = trace.get("stage_timings_ms")
    assert isinstance(timings, dict)
    assert "bm25_build" in timings
    assert "bm25_query" in timings
    assert "semantic" in timings
    assert all(isinstance(ms, int) and ms >= 0 for ms in timings.values())


def test_pipeline_stage_timing_accumulates_compatibility_retry(monkeypatch) -> None:
    class RetryingBM25(FakeBM25):
        def query(self, query: str, top_n: int = 20, **kwargs) -> list[ScoredPage]:
            if "include_reference" in kwargs:
                self.queries.append((query, top_n))
                raise TypeError("legacy query signature")
            return super().query(query, top_n)

    bm25 = RetryingBM25([page("bm25-a", 10.0)])
    ticks = iter(range(100))
    monkeypatch.setattr(
        pipeline_mod,
        "time",
        SimpleNamespace(monotonic=lambda: next(ticks)),
    )
    monkeypatch.setattr(search, "get_bm25", lambda: bm25)
    monkeypatch.setattr(search, "context_seed_results", lambda *_a, **_k: [])
    monkeypatch.setattr(search, "graph_expand_results", lambda *_a, **_k: [])
    monkeypatch.setattr(
        search, "load_negative_feedback_config", disabled_negative_feedback
    )
    monkeypatch.setattr(
        search,
        "load_active_fusion_weights",
        lambda: dict(search.DEFAULT_FUSION_WEIGHTS),
    )

    search.search("query", top_n=1, semantic=False)

    assert bm25.queries == [("query", 100), ("query", 100)]
    assert search.last_search_trace()["stage_timings_ms"]["bm25_query"] == 2000


def test_production_search_preserves_partial_timings_on_base_exception(
    monkeypatch,
) -> None:
    class PipelineInterrupted(BaseException):
        pass

    ticks = iter(range(100))
    monkeypatch.setattr(
        pipeline_mod,
        "time",
        SimpleNamespace(monotonic=lambda: next(ticks)),
    )
    monkeypatch.setattr(search, "get_bm25", lambda: FakeBM25([page("page", 1.0)]))
    monkeypatch.setattr(
        search,
        "semantic_search",
        lambda *_a, **_k: (_ for _ in ()).throw(PipelineInterrupted()),
    )
    monkeypatch.setattr(
        search,
        "load_active_fusion_weights",
        lambda: dict(search.DEFAULT_FUSION_WEIGHTS),
    )

    with pytest.raises(PipelineInterrupted):
        search.search("query", top_n=1, semantic=True)

    timings = search.last_search_trace()["stage_timings_ms"]
    assert timings == {
        "bm25_load": 1000,
        "bm25_build": 1000,
        "bm25_query": 2000,
        "context_seed": 1000,
        "semantic": 1000,
    }


def test_production_search_builds_usage_prior_only_when_weight_is_positive(
    monkeypatch,
) -> None:
    bm25 = FakeBM25([page("bm25-a", 10.0)])
    usage_calls: list[dict[str, object]] = []

    def fake_usage(
        candidate_ids, *, limit: int = 50, decay: float = 0.98, cap: float = 3.0
    ):
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
    monkeypatch.setattr(
        search, "graph_expand_results", lambda results, **kwargs: [page("graph-a", 5.0)]
    )
    monkeypatch.setattr(search, "usage_prior_results", fake_usage)
    monkeypatch.setattr(
        search, "load_negative_feedback_config", disabled_negative_feedback
    )

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


def test_eval_hybrid_current_uses_production_graph_call_and_usage_gate(
    monkeypatch,
    isolated_selection_index,
) -> None:
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
        raise AssertionError(
            "hybrid-current must keep usage-prior gated while weight is zero"
        )

    monkeypatch.setattr(search_eval, "get_bm25", lambda: bm25)
    monkeypatch.setattr(
        search_eval, "semantic_search", lambda query, top_n=20: [page("sem-a", 0.8)]
    )
    monkeypatch.setattr(search_eval, "graph_expand_results", fake_graph)
    monkeypatch.setattr(search_eval, "usage_prior_results", fail_usage)
    monkeypatch.setattr(
        search_eval, "load_negative_feedback_config", disabled_negative_feedback
    )

    payload = search_eval.run_variant("query", "hybrid-current", top_n=3)

    assert graph_calls == [
        {"page_ids": ["bm25-a", "bm25-b", "sem-a"], "decay": 0.3, "limit": 100}
    ]
    assert [result.page_id for result in payload["results"]] == [
        "bm25-a",
        "bm25-b",
        "sem-a",
    ]
    assert payload["channels"]["graph"] == []
    assert payload["channels"]["usage_prior"] == []


def test_hybrid_current_tracks_production_when_default_graph_weight_changes(
    monkeypatch,
    isolated_selection_index,
) -> None:
    def fake_graph(results: list[ScoredPage], *, decay: float = 0.5, limit: int = 50):
        assert decay == 0.5
        return [page("graph-a", 5.0)]

    for module in (search, search_eval):
        monkeypatch.setattr(
            module, "get_bm25", lambda: FakeBM25([page("bm25-a", 10.0)])
        )
        monkeypatch.setattr(module, "semantic_search", lambda query, top_n=20: [])
        monkeypatch.setattr(module, "graph_expand_results", fake_graph)
        monkeypatch.setattr(module, "usage_prior_results", lambda *args, **kwargs: [])
        monkeypatch.setattr(
            module, "load_negative_feedback_config", disabled_negative_feedback
        )
        monkeypatch.setitem(module.DEFAULT_FUSION_WEIGHTS, "graph", 0.5)
    monkeypatch.setattr(
        search,
        "load_active_fusion_weights",
        lambda: dict(search.DEFAULT_FUSION_WEIGHTS),
    )
    monkeypatch.setattr(
        search_eval,
        "load_active_fusion_weights",
        lambda: dict(search_eval.DEFAULT_FUSION_WEIGHTS),
    )

    production_results, production_mode = search.search("query", top_n=2)
    eval_payload = search_eval.run_variant("query", "hybrid-current", top_n=2)

    assert production_mode == "bm25+graph"
    assert [result.page_id for result in production_results] == ["bm25-a", "graph-a"]
    assert [result.page_id for result in eval_payload["results"]] == [
        "bm25-a",
        "graph-a",
    ]
    assert eval_payload["channels"]["graph"] == ["graph-a"]


def test_eval_hybrid_rerank_applies_negative_feedback_after_rerank(
    monkeypatch,
    isolated_selection_index,
) -> None:
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
    monkeypatch.setattr(
        search_eval, "graph_expand_results", lambda *_args, **_kwargs: []
    )
    monkeypatch.setattr(
        search_eval, "load_reranker_config", lambda: RerankerConfig(enabled=True)
    )
    monkeypatch.setattr(search_eval, "rerank_results", fake_rerank)
    monkeypatch.setattr(
        search_eval,
        "load_negative_feedback_config",
        lambda: NegativeFeedbackConfig(enabled=True),
    )
    monkeypatch.setattr(
        search_eval, "penalties_for_query", lambda query, config: {"a": 0.5}
    )
    monkeypatch.setattr(search_eval, "apply_penalties", fake_apply_penalties)

    payload = search_eval.run_variant("query", "hybrid-rerank", top_n=2)

    assert penalty_inputs == [["b", "a"]]
    assert [result.page_id for result in payload["results"]] == ["b", "a"]
    assert payload["channels"]["reranker"]["status"] == "applied"
    assert payload["channels"]["negative_feedback"] == {
        "status": "applied",
        "pages": ["a"],
    }


def test_run_weighted_hybrid_uses_production_graph_call_and_usage_gate(
    monkeypatch,
) -> None:
    bm25 = FakeBM25([page("bm25-a", 10.0)])
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

    monkeypatch.setattr(search_eval, "get_bm25", lambda: bm25)
    monkeypatch.setattr(
        search_eval, "semantic_search", lambda query, top_n=20: [page("sem-a", 0.8)]
    )
    monkeypatch.setattr(search_eval, "graph_expand_results", fake_graph)
    monkeypatch.setattr(search_eval, "usage_prior_results", fail_usage)
    monkeypatch.setattr(
        search_eval, "load_negative_feedback_config", disabled_negative_feedback
    )

    payload = search_eval.run_weighted_hybrid(
        "query",
        {**search_eval.DEFAULT_FUSION_WEIGHTS, "bm25_score_bonus": 0.0},
        top_n=2,
    )

    assert graph_calls == [
        {"page_ids": ["bm25-a", "sem-a"], "decay": 0.3, "limit": 100}
    ]
    assert [result.page_id for result in payload["results"]] == ["bm25-a", "sem-a"]
