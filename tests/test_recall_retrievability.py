from __future__ import annotations

from chronovisor.core.frontmatter import parse, patch
from chronovisor.ingest import ingest


def test_ensure_recall_metadata_frontmatter_adds_description_and_questions(
    monkeypatch,
) -> None:
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

    meta, body = parse(out)
    assert meta["description"] == "short summary"
    assert "summary" not in meta
    assert meta["recall_questions"] == ["What was decided?", "How to continue?"]
    assert body == "Body\n"


def test_existing_description_is_preserved_when_questions_are_missing(
    monkeypatch,
) -> None:
    generated = {"summary": "replacement summary", "recall_questions": ["New question?"]}
    calls = 0

    def generate(_title: str, _body: str, _page_id: str) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return generated

    monkeypatch.setattr(ingest, "_generate_recall_metadata", generate)
    text = (
        "---\n"
        "title: Sample\n"
        "description: Canonical description\n"
        "---\n"
        "Body\n"
    )

    out = ingest._ensure_recall_metadata_frontmatter(text, "sample", parse, patch)

    meta, body = parse(out)
    assert meta["description"] == "Canonical description"
    assert meta["recall_questions"] == ["New question?"]
    assert calls == 1
    assert body == "Body\n"


def test_legacy_summary_is_promoted_without_generation_or_metadata_loss(
    monkeypatch,
) -> None:
    def fail_generation(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("legacy summary promotion must not call the model")

    monkeypatch.setattr(ingest, "_generate_recall_metadata", fail_generation)
    text = (
        "---\n"
        "title: Legacy page\n"
        "updated: 2026-06-11\n"
        "status: stable\n"
        "type: knowledge\n"
        "tags: [d/example, t/analysis]\n"
        "summary: Legacy wording\n"
        "recall_questions: ['What was decided?']\n"
        "---\n"
        "Body line.\n\nSecond paragraph.\n"
    )
    before_meta, before_body = parse(text)

    out = ingest._ensure_recall_metadata_frontmatter(text, "legacy-page", parse, patch)

    after_meta, after_body = parse(out)
    assert after_meta["description"] == before_meta["summary"] == "Legacy wording"
    assert after_meta["summary"] == before_meta["summary"]
    assert after_meta["updated"] == before_meta["updated"]
    assert after_meta["tags"] == before_meta["tags"]
    assert after_body == before_body


def test_forced_rebuild_regenerates_description_and_stale_summary() -> None:
    text = (
        "---\n"
        "title: Corrected page\n"
        "description: Stale canonical description\n"
        "summary: Stale legacy summary\n"
        "recall_questions: ['Stale question?']\n"
        "---\n"
        "Corrected body.\n"
    )

    out = ingest._ensure_recall_metadata_frontmatter(
        text,
        "corrected-page",
        parse,
        patch,
        force_deterministic_rebuild=True,
    )

    expected = ingest._fallback_recall_metadata(
        "Corrected page", "Corrected body.\n", "corrected-page"
    )
    meta, body = parse(out)
    assert meta["description"] == expected["summary"]
    assert meta["summary"] == expected["summary"]
    assert meta["recall_questions"] == expected["recall_questions"]
    assert body == "Corrected body.\n"


def test_forced_rebuild_does_not_add_legacy_summary_to_fresh_page() -> None:
    text = "---\ntitle: Fresh page\n---\nFresh body.\n"

    out = ingest._ensure_recall_metadata_frontmatter(
        text,
        "fresh-page",
        parse,
        patch,
        force_deterministic_rebuild=True,
    )

    meta, _body = parse(out)
    assert meta["description"] == "Fresh body."
    assert "summary" not in meta


def test_recall_question_sanitizer_removes_inline_list_breakers() -> None:
    assert ingest._safe_recall_field("a, b [c]: d") == "a b c d"
