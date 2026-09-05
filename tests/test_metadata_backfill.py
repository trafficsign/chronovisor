from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from chronovisor.ops import metadata_backfill


@pytest.fixture(autouse=True)
def _valid_okf_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("index.md", "log.md", "schema.md"):
        (tmp_path / name).write_text("legacy\n", encoding="utf-8")
    from chronovisor.core import page_mutation

    monkeypatch.setattr(page_mutation, "CHRONOVISOR_ROOT", tmp_path)
    monkeypatch.setattr(metadata_backfill, "CHRONOVISOR_ROOT", tmp_path)


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
        return patch(text, {"description": f"{meta['title']} summary", "recall_questions": [f"What is {meta['title']}?"]})

    monkeypatch.setattr(metadata_backfill, "ensure_recall_metadata_frontmatter", propose)
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
        "ensure_recall_metadata_frontmatter",
        lambda *_args: next(generated),
    )

    first = metadata_backfill._stable_local_proposal("original", "page")
    second = metadata_backfill._stable_local_proposal("original", "page")

    assert first == second == "proposal one"


def test_backfill_selects_missing_canonical_description(tmp_path: Path, monkeypatch) -> None:
    legacy = tmp_path / "legacy.md"
    canonical = tmp_path / "canonical.md"
    legacy.write_text(
        "---\ntitle: Legacy\ntype: Concept\nstatus: stable\n"
        "summary: Existing summary\nrecall_questions: ['What is this?']\n---\nBody.\n",
        encoding="utf-8",
    )
    canonical.write_text(
        "---\ntitle: Canonical\ntype: Concept\nstatus: stable\n"
        "description: Existing description\nrecall_questions: ['What is this?']\n---\nBody.\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(metadata_backfill, "all_pages", lambda: [legacy, canonical])

    result = metadata_backfill.backfill_metadata(limit=0, dry_run=True)

    assert result["candidates"] == 1
    assert result["pages"] == ["legacy"]
    assert "description:" not in legacy.read_text(encoding="utf-8")


def test_backfill_promotes_existing_summary_through_review(tmp_path: Path, monkeypatch) -> None:
    from chronovisor.core import frontmatter
    from chronovisor.ingest import ingest

    page = tmp_path / "legacy.md"
    original = (
        "---\ntitle: Legacy\ntype: Concept\nstatus: stable\n"
        "summary: 'Existing: exact summary'\n"
        "recall_questions: ['What is this?']\ncustom: {keep: true}\n---\nBody.\n"
    )
    page.write_text(original, encoding="utf-8")
    monkeypatch.setattr(metadata_backfill, "all_pages", lambda: [page])
    monkeypatch.setattr(metadata_backfill, "REVIEW_DIR", tmp_path / "reviews")
    monkeypatch.setattr(
        ingest, "_generate_recall_metadata",
        lambda *_args: pytest.fail("Existing summary must not be regenerated"),
    )
    calls = []

    def reviewer(prompt, _schema):
        calls.append(prompt)
        return _decision("approved")

    result = metadata_backfill.backfill_metadata(reviewer=reviewer)

    assert result["updated"] == 1
    assert result["frontier_calls"] == len(calls) == 1
    meta, body = frontmatter.parse(page.read_text(encoding="utf-8"))
    before, original_body = frontmatter.parse(original)
    assert meta == {**before, "description": before["summary"]}
    assert body == original_body


def test_typed_yaml_generated_frontmatter_has_stable_proposal_and_prompt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chronovisor.decision.lint_mutation_contract import (
        build_safe_fix_prompt,
        canonical_hash,
    )

    page = tmp_path / "typed.md"
    original = (
        "---\n"
        "title: Typed metadata\n"
        "updated: 2026-08-11\n"
        "features: !!set\n"
        "  ? gamma\n"
        "  ? alpha\n"
        "  ? beta\n"
        "---\n"
        "Body.\n"
    )
    page.write_text(original, encoding="utf-8")
    monkeypatch.setattr(metadata_backfill, "all_pages", lambda: [page])
    monkeypatch.setattr(metadata_backfill, "REVIEW_DIR", tmp_path / "reviews")

    def propose(text, _page_id, _parse, patch):
        return patch(
            text,
            {
                "description": "Typed metadata summary.",
                "recall_questions": ["What is typed metadata?"],
            },
        )

    monkeypatch.setattr(
        metadata_backfill,
        "ensure_recall_metadata_frontmatter",
        propose,
    )
    captured: dict[str, object] = {}

    def review(proposal, **kwargs):
        captured["proposal"] = proposal
        captured["updated_text"] = kwargs["updated_text"]
        return {"decision": "needs_retry", "summary": "captured", "valid": True}

    monkeypatch.setattr(metadata_backfill, "review_semantic_mutation", review)

    result = metadata_backfill.backfill_metadata(
        limit=1,
        max_frontier_calls=1,
        reviewer=lambda *_args: _decision("approved"),
    )

    proposal = captured["proposal"]
    assert isinstance(proposal, dict)
    assert proposal["details"]["generated_frontmatter"] == {
        "kind": "canonical_yaml",
        "utf8_bytes": 213,
        "sha256": "c93e0cbef75d2beac16a5e99cd2449517565bbb7f542408773c2d6bd3bc9f700",
    }
    prompt = build_safe_fix_prompt(proposal, expected_text=original)
    assert canonical_hash(proposal) == (
        "302459718ae246736b3b51b2079dd9e97c2c6a03bc9ed9084db5714e9a184fe8"
    )
    assert hashlib.sha256(prompt.encode("utf-8")).hexdigest() == (
        "dff2f4da9bfd2ef12833a769434ba37df9568dffe991ff54cda8efc26175279d"
    )
    json.dumps(proposal, ensure_ascii=False, allow_nan=False)
    assert result["retry"] == 1
