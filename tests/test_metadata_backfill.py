from __future__ import annotations

from pathlib import Path

from chronovisor import metadata_backfill


def _decision(value: str) -> dict:
    return {
        "decision": value,
        "summary": value,
        "tests_run": [],
        "commit": None,
        "committed": False,
        "pushed": False,
        "risk": None,
        "notes": None,
    }


def test_cached_rejection_does_not_pin_later_metadata_candidates(tmp_path: Path, monkeypatch) -> None:
    first = tmp_path / "first.md"
    second = tmp_path / "second.md"
    first.write_text("---\ntitle: First\n---\nbody\n", encoding="utf-8")
    second.write_text("---\ntitle: Second\n---\nbody\n", encoding="utf-8")
    monkeypatch.setattr(metadata_backfill, "all_pages", lambda: [first, second])
    monkeypatch.setattr(metadata_backfill, "REVIEW_DIR", tmp_path / "reviews")

    def propose(text, _page_id, parse, patch):
        meta, _body = parse(text)
        return patch(text, {"summary": f"{meta['title']} summary", "recall_questions": [f"What is {meta['title']}?"]})

    monkeypatch.setattr(metadata_backfill, "_ensure_recall_metadata_frontmatter", propose)
    calls: list[str] = []

    def reviewer(prompt, _schema):
        page = "first" if '"page_id": "first"' in prompt else "second"
        calls.append(page)
        return _decision("rejected" if page == "first" else "approved")

    first_run = metadata_backfill.backfill_metadata(limit=1, max_frontier_calls=1, reviewer=reviewer)
    second_run = metadata_backfill.backfill_metadata(limit=1, max_frontier_calls=1, reviewer=reviewer)

    assert first_run["rejected"] == 1
    assert second_run["updated"] == 1
    assert calls == ["first", "second"]
    assert "Second summary" in second.read_text(encoding="utf-8")


def test_local_metadata_proposal_is_stable_for_exact_preimage(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(metadata_backfill, "REVIEW_DIR", tmp_path / "reviews")
    generated = iter(["proposal one", "proposal two"])
    monkeypatch.setattr(
        metadata_backfill,
        "_ensure_recall_metadata_frontmatter",
        lambda *_args: next(generated),
    )

    first = metadata_backfill._stable_local_proposal("original", "page")
    second = metadata_backfill._stable_local_proposal("original", "page")

    assert first == second == "proposal one"
