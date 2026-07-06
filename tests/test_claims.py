from __future__ import annotations

import json
from pathlib import Path

from llm_wiki_mcp import claims


def test_page_claims_extracts_summary_entities_and_lead(tmp_path: Path, monkeypatch) -> None:
    page = tmp_path / "alpha.md"
    page.write_text(
        "---\n"
        "title: Alpha\n"
        "updated: 2026-07-06\n"
        "summary: Alpha summary\n"
        "entities: [MHI, Codex]\n"
        "---\n"
        "# Alpha Body\n"
        "Important body lead.\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(claims, "find_page", lambda page_id: page if page_id == "alpha" else None)

    rows = claims.page_claims("alpha")

    predicates = {row["predicate"] for row in rows}
    assert {"page.title", "page.summary", "page.entity", "body.lead"} <= predicates
    assert any(row["value"] == "Alpha summary" for row in rows)


def test_search_claims_scores_token_overlap(tmp_path: Path) -> None:
    path = tmp_path / "claims.jsonl"
    path.write_text(
        json.dumps({"claim_id": "c1", "subject": "alpha", "predicate": "page.summary", "value": "MHI Codex memory"})
        + "\n"
        + json.dumps({"claim_id": "c2", "subject": "beta", "predicate": "page.summary", "value": "unrelated"})
        + "\n",
        encoding="utf-8",
    )

    rows = claims.search_claims("Codex memory", path=path)

    assert [row["claim_id"] for row in rows] == ["c1"]

