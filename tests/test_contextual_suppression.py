from __future__ import annotations

from chronovisor.recall.contextual_suppression import ranking_components
from chronovisor.search.search_types import ScoredPage


class Graph:
    def refresh_if_stale(self) -> None:
        pass

    def outlinks(self, page_id: str) -> list[str]:
        return [f"out-{index}" for index in range(40)] if page_id == "hub" else []

    def backlinks(self, page_id: str) -> list[str]:
        return [f"in-{index}" for index in range(60)] if page_id == "hub" else []


def candidate(page_id: str, title: str, snippet: str = "") -> ScoredPage:
    return ScoredPage(
        page_id=page_id,
        title=title,
        folder="",
        updated="2026-07-30",
        score=0.1,
        snippet=snippet,
    )


def test_hub_penalty_is_contextual_and_exact_match_is_protected(monkeypatch) -> None:
    monkeypatch.setattr(
        "chronovisor.recall.contextual_suppression.contextual_negative_trace",
        lambda *_args, **_kwargs: {},
    )
    rows = [
        candidate("hub", "General memory hub"),
        candidate("exact-page", "Chronovisor Recall Processor"),
    ]

    components = ranking_components(
        "Chronovisor Recall Processor の設定",
        rows,
        store=Graph(),
    )

    assert components["hub"]["hub_penalty"] > 0
    assert components["exact-page"]["hub_penalty"] == 0
    assert components["exact-page"]["exact_match_protected"] is True


def test_supporting_span_reduces_contextual_hub_penalty(monkeypatch) -> None:
    monkeypatch.setattr(
        "chronovisor.recall.contextual_suppression.contextual_negative_trace",
        lambda *_args, **_kwargs: {},
    )
    without_span = ranking_components(
        "stateful recall field activation",
        [candidate("hub", "Hub")],
        store=Graph(),
    )
    with_span = ranking_components(
        "stateful recall field activation",
        [
            candidate(
                "hub",
                "Hub",
                "stateful recall field activation is persisted per session",
            )
        ],
        store=Graph(),
    )

    assert (
        with_span["hub"]["hub_penalty"]
        < without_span["hub"]["hub_penalty"]
    )
