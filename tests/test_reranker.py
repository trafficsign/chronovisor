from __future__ import annotations

import json
import threading

from chronovisor.core import reranker
from chronovisor.core.reranker import RerankOutcome, rerank_results
from chronovisor.core.runtime_config import RerankerConfig
from chronovisor.core.search import ScoredPage
from chronovisor.hosts import server


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

    monkeypatch.setattr(reranker, "score_fn", fake_score_fn)
    monkeypatch.setattr(reranker, "find_page", lambda _page_id: None)

    outcome = rerank_results(
        "query",
        candidates,
        config=RerankerConfig(enabled=True, top_n=2, weight=2.0),
    )

    assert [result.page_id for result in outcome.results] == ["b", "a", "c"]
    assert outcome.metadata["status"] == "applied"
    assert outcome.metadata["candidate_count"] == 2
    assert outcome.metadata["execution"] == "in_process"
    assert [row["page_id"] for row in outcome.metadata["scores"]] == ["b", "a"]
    assert [detail.raw_score for detail in outcome.scores] == [0.9, 0.1]
    assert round(outcome.scores[0].margin_to_next, 6) == 0.8


def test_rerank_results_unavailable_preserves_order() -> None:
    candidates = [page("a"), page("b")]

    outcome = rerank_results(
        "query",
        candidates,
        config=RerankerConfig(enabled=True, backend="missing"),
    )

    assert outcome.results == candidates
    assert outcome.metadata["status"] == "unavailable"


def test_rerank_results_rejects_partial_score_vectors(monkeypatch) -> None:
    candidates = [page("a"), page("b")]
    monkeypatch.setattr(reranker, "find_page", lambda _page_id: None)
    monkeypatch.setattr(
        reranker,
        "score_fn",
        lambda _config: lambda _query, _passages, _cfg: [0.9],
    )

    outcome = rerank_results(
        "query", candidates, config=RerankerConfig(enabled=True, top_n=2)
    )

    assert outcome.results == candidates
    assert outcome.metadata["status"] == "unavailable"
    assert outcome.metadata["score_count"] == 1


def test_transformer_loader_prefers_complete_local_snapshot() -> None:
    calls: list[tuple[str, bool]] = []

    class FakeTokenizer:
        @classmethod
        def from_pretrained(cls, _model, **kwargs):
            calls.append(("tokenizer", kwargs.get("local_files_only", False)))
            return "tokenizer"

    class FakeModel:
        @classmethod
        def from_pretrained(cls, _model, **kwargs):
            calls.append(("model", kwargs.get("local_files_only", False)))
            return "model"

    tokenizer, model = reranker._load_transformer_components(
        RerankerConfig(enabled=True), FakeTokenizer, FakeModel
    )

    assert (tokenizer, model) == ("tokenizer", "model")
    assert calls == [("tokenizer", True), ("model", True)]


def test_transformer_loader_allows_first_install_fallback() -> None:
    calls: list[tuple[str, bool]] = []

    class FakeTokenizer:
        @classmethod
        def from_pretrained(cls, _model, **kwargs):
            local = kwargs.get("local_files_only", False)
            calls.append(("tokenizer", local))
            if local:
                raise OSError("not cached")
            return "tokenizer"

    class FakeModel:
        @classmethod
        def from_pretrained(cls, _model, **kwargs):
            calls.append(("model", kwargs.get("local_files_only", False)))
            return "model"

    tokenizer, model = reranker._load_transformer_components(
        RerankerConfig(enabled=True), FakeTokenizer, FakeModel
    )

    assert (tokenizer, model) == ("tokenizer", "model")
    assert calls[:3] == [
        ("tokenizer", True),
        ("tokenizer", False),
        ("model", False),
    ]


def test_start_reranker_warmup_is_single_daemon(monkeypatch) -> None:
    started: list[str] = []
    release = threading.Event()

    def fake_warm(config):
        started.append(config.model)
        release.wait(1)
        return {"status": "ready"}

    monkeypatch.setattr(reranker, "warm_reranker", fake_warm)
    monkeypatch.setattr(reranker, "_WARMUP_THREAD", None)
    config = RerankerConfig(enabled=True)

    first = reranker.start_reranker_warmup(config)
    second = reranker.start_reranker_warmup(config)

    assert first is not None
    assert first is second
    assert first.daemon is True
    release.set()
    first.join(1)
    assert started == [config.model]


def test_chronovisor_search_uses_reranker_only_when_enabled(monkeypatch) -> None:
    class FakeStore:
        def refresh(self) -> None:
            pass

        def tags(self, page_id: str) -> list[str]:
            return []

        def outlinks(self, page_id: str) -> list[str]:
            return []

    def fake_search(**kwargs):
        assert kwargs["top_n"] == 10
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

    from chronovisor.core import runtime_config
    from chronovisor.core import search as search_mod

    monkeypatch.setattr(search_mod, "search", fake_search)
    monkeypatch.setattr(
        runtime_config, "load_reranker_config", lambda: RerankerConfig(enabled=True)
    )
    monkeypatch.setattr(reranker, "rerank_results", fake_rerank)
    monkeypatch.setattr(server, "get_store", lambda: FakeStore())
    monkeypatch.setattr(server, "find_page", lambda _page_id: None)

    tool_fn = (
        server.chronovisor_search.fn
        if hasattr(server.chronovisor_search, "fn")
        else server.chronovisor_search
    )
    payload = json.loads(tool_fn("needle", depth=0))

    assert payload["search_mode"] == "hybrid+rerank"
    assert payload["reranker"]["status"] == "applied"
    assert [hit["page_id"] for hit in payload["direct_hits"]] == ["b", "a"]


def test_chronovisor_search_reranks_after_tag_filter(monkeypatch) -> None:
    class FakeStore:
        def refresh(self) -> None:
            pass

        def tags(self, page_id: str) -> list[str]:
            return ["d/keep"] if page_id == "keep" else []

        def outlinks(self, page_id: str) -> list[str]:
            return []

    seen_candidates: list[list[str]] = []

    def fake_search(**kwargs):
        assert kwargs["top_n"] == 10
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

    from chronovisor.core import runtime_config
    from chronovisor.core import search as search_mod

    monkeypatch.setattr(search_mod, "search", fake_search)
    monkeypatch.setattr(
        runtime_config, "load_reranker_config", lambda: RerankerConfig(enabled=True)
    )
    monkeypatch.setattr(reranker, "rerank_results", fake_rerank)
    monkeypatch.setattr(server, "get_store", lambda: FakeStore())
    monkeypatch.setattr(server, "find_page", lambda _page_id: None)

    tool_fn = (
        server.chronovisor_search.fn
        if hasattr(server.chronovisor_search, "fn")
        else server.chronovisor_search
    )
    payload = json.loads(tool_fn("needle", depth=0, tags=["d/keep"]))

    assert seen_candidates == [["keep"]]
    assert payload["reranker"]["status"] == "applied"
    assert [hit["page_id"] for hit in payload["direct_hits"]] == ["keep"]
