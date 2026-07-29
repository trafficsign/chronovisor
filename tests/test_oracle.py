from __future__ import annotations

from chronovisor.research import oracle
from chronovisor.search.search_types import ScoredPage


def test_oracle_bundle_returns_pages_and_claims(monkeypatch) -> None:
    monkeypatch.setattr(
        oracle,
        "search",
        lambda query, top_n=8: (
            [ScoredPage(page_id="p", title="P", folder="", updated="2026-07-06", score=1.0)],
            "hybrid",
        ),
    )
    monkeypatch.setattr(
        oracle,
        "search_claims",
        lambda query, limit=12: [{"claim_id": "c", "source_page": "p", "predicate": "page.summary", "value": "P"}],
    )

    payload = oracle.oracle_bundle("query", ensure_claim_index=False)

    assert payload["answer_mode"] == "cite-only"
    assert payload["pages"][0]["page_id"] == "p"
    assert payload["claims"][0]["claim_id"] == "c"

