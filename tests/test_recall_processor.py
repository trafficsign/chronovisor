from __future__ import annotations

from chronovisor.core.runtime_config import RerankerConfig, RerankerServiceConfig
from chronovisor.recall import recall_processor
from chronovisor.search.reranker import RerankOutcome
from chronovisor.search.search_types import ScoredPage


def page(page_id: str) -> ScoredPage:
    return ScoredPage(
        page_id=page_id,
        title=page_id,
        folder="",
        updated="2026-07-30",
        score=1.0,
    )


def shadow_config() -> RerankerConfig:
    return RerankerConfig(
        enabled=True,
        top_n=2,
        service=RerankerServiceConfig(
            enabled=True,
            mode="shadow",
            timeout_ms=500,
        ),
    )


def test_shadow_reranker_records_before_after_without_mutating(monkeypatch) -> None:
    candidates = [page("a"), page("b")]
    monkeypatch.setattr(
        recall_processor, "load_reranker_config", shadow_config
    )
    monkeypatch.setattr(
        recall_processor.reranker_client,
        "selected_for_rollout",
        lambda _query, _config: True,
    )
    monkeypatch.setattr(
        recall_processor.reranker_client,
        "rerank",
        lambda _query, values, **_kwargs: RerankOutcome(
            [values[1], values[0]],
            {
                "status": "applied",
                "execution": "service",
                "latency_ms": 7,
                "scores": [],
            },
        ),
    )

    payload = recall_processor.shadow_rerank_candidates(
        "query", candidates, timeout_ms=600
    )

    assert [item.page_id for item in candidates] == ["a", "b"]
    assert payload["before_page_ids"] == ["a", "b"]
    assert payload["after_page_ids"] == ["b", "a"]
    assert payload["changed"] is True


def test_shadow_reranker_fails_open(monkeypatch) -> None:
    monkeypatch.setattr(
        recall_processor, "load_reranker_config", shadow_config
    )
    monkeypatch.setattr(
        recall_processor.reranker_client,
        "selected_for_rollout",
        lambda _query, _config: True,
    )
    monkeypatch.setattr(
        recall_processor.reranker_client,
        "rerank",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("service stopped")
        ),
    )

    payload = recall_processor.shadow_rerank_candidates(
        "query", [page("a")], timeout_ms=500
    )

    assert payload["status"] == "unavailable"
    assert payload["reason"] == "RuntimeError"
