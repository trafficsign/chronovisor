from __future__ import annotations

from chronovisor import ingest
from chronovisor.frontmatter import parse, patch


def test_ensure_recall_metadata_frontmatter_adds_summary_and_questions(monkeypatch) -> None:
    monkeypatch.setattr(
        ingest,
        "_generate_recall_metadata",
        lambda title, body, page_id: {
            "summary": "short summary",
            "recall_questions": ["What was decided?", "How to continue?"],
        },
    )
    text = "---\ntitle: Sample\nupdated: 2026-06-11\n---\nBody\n"

    out = ingest._ensure_recall_metadata_frontmatter(text, "sample", parse, patch)

    assert "summary: short summary" in out
    assert "recall_questions: [What was decided?, How to continue?]" in out


def test_recall_question_sanitizer_removes_inline_list_breakers() -> None:
    assert ingest._safe_recall_field("a, b [c]: d") == "a b c d"
