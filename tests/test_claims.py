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


def test_append_page_claims_requires_source_raw(tmp_path: Path, monkeypatch) -> None:
    page = tmp_path / "alpha.md"
    page.write_text("---\ntitle: Alpha\nupdated: 2026-07-06\n---\nbody", encoding="utf-8")
    ledger = tmp_path / "claims.jsonl"
    monkeypatch.setattr(claims, "find_page", lambda page_id: page if page_id == "alpha" else None)
    monkeypatch.setattr(claims, "CLAIMS_FILE", ledger)

    payload = claims.append_page_claims(["alpha"], source_raw="")

    assert payload["status"] == "skipped"
    assert not ledger.exists()


def test_sanitize_claim_ledger_drops_placeholders(tmp_path: Path, monkeypatch) -> None:
    real_page = tmp_path / "real.md"
    real_page.write_text("real", encoding="utf-8")
    path = tmp_path / "claims.jsonl"
    path.write_text(
        json.dumps({"source_raw": "", "source_page": "p0", "value": "body"}) + "\n"
        + json.dumps({"source_raw": "raw.md", "source_page": "real", "value": "useful"}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(claims, "find_page", lambda page_id: real_page if page_id == "real" else None)

    payload = claims.sanitize_claim_ledger(path=path)

    assert payload["kept"] == 1
    assert payload["dropped"] == 1
    assert "useful" in path.read_text(encoding="utf-8")
