"""Unit tests for the tag-backfill skip-marker logic.

The full sweep is gated by Ollama, so these tests stub out the LLM and
focus on the file-level contracts: ``_mark_unfit`` writes a stable
frontmatter field, ``_is_marked_unfit`` reads it back, and a successful
re-run clears the stale marker.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import tag_backfill_apply as tba  # noqa: E402
from llm_wiki_mcp.frontmatter import parse as fm_parse  # noqa: E402


@pytest.fixture
def page_with_fm(tmp_path: Path, monkeypatch) -> Path:
    """A page in a stub PAGES_DIR with a minimal frontmatter."""
    pages_dir = tmp_path / "pages"
    pages_dir.mkdir()
    page = pages_dir / "example.md"
    page.write_text(
        "---\n"
        "title: Example\n"
        "updated: 2026-05-09\n"
        "---\n\n"
        "Body of the example page.\n"
    )
    # Both tag_backfill_apply and the underlying wiki helper resolve
    # pages relative to PAGES_DIR. Patch both so find_page() looks here.
    import llm_wiki_mcp.wiki as wiki_mod

    monkeypatch.setattr(wiki_mod, "PAGES_DIR", pages_dir)
    return page


def test_mark_unfit_adds_tag_status(page_with_fm: Path) -> None:
    tba._mark_unfit(page_with_fm, tba.TAG_STATUS_NO_FIT)
    meta, _ = fm_parse(page_with_fm.read_text())
    assert meta.get("tag_status") == tba.TAG_STATUS_NO_FIT


def test_mark_unfit_overwrites_existing_status(page_with_fm: Path) -> None:
    tba._mark_unfit(page_with_fm, tba.TAG_STATUS_NO_FIT)
    tba._mark_unfit(page_with_fm, tba.TAG_STATUS_FORMAT_FAIL)
    meta, _ = fm_parse(page_with_fm.read_text())
    assert meta.get("tag_status") == tba.TAG_STATUS_FORMAT_FAIL


def test_is_marked_unfit_detects_status(page_with_fm: Path) -> None:
    assert tba._is_marked_unfit("example") is False
    tba._mark_unfit(page_with_fm, tba.TAG_STATUS_NO_FIT)
    assert tba._is_marked_unfit("example") is True


def test_is_marked_unfit_handles_missing_page() -> None:
    # Page that doesn't exist anywhere should report False, not raise.
    assert tba._is_marked_unfit("definitely-not-a-real-page-id") is False


def test_mark_unfit_preserves_body(page_with_fm: Path) -> None:
    _, body_before = fm_parse(page_with_fm.read_text())
    tba._mark_unfit(page_with_fm, tba.TAG_STATUS_FORMAT_FAIL)
    _, body_after = fm_parse(page_with_fm.read_text())
    assert body_after == body_before


def test_successful_apply_clears_tag_status(page_with_fm: Path) -> None:
    """A page that was previously marked but later succeeds must have
    the marker removed — otherwise we'd carry forward stale state."""
    from llm_wiki_mcp.frontmatter import patch as fm_patch

    # Simulate the prior failed run leaving a marker behind.
    tba._mark_unfit(page_with_fm, tba.TAG_STATUS_FORMAT_FAIL)
    assert "tag_status" in fm_parse(page_with_fm.read_text())[0]

    # Simulate the success path the way _process_one does it.
    original = page_with_fm.read_text()
    patched = fm_patch(original, {"tags": ["d/ai-industry"]}, deletes=["tag_status"])
    page_with_fm.write_text(patched)

    meta, _ = fm_parse(page_with_fm.read_text())
    assert "tag_status" not in meta
    assert meta["tags"] == ["d/ai-industry"]
