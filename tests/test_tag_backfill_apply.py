"""Fail-closed tests for the obsolete local-model tag backfill."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import tag_backfill_apply as tba  # noqa: E402
from chronovisor.raw.legacy_semantic_write import (  # noqa: E402
    LegacySemanticMutationDisabled,
)


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
    import chronovisor.core.store as wiki_mod

    monkeypatch.setattr(wiki_mod, "PAGES_DIR", pages_dir)
    return page


@pytest.mark.parametrize(
    "status",
    [tba.TAG_STATUS_NO_FIT, tba.TAG_STATUS_FORMAT_FAIL],
)
def test_mark_unfit_is_disabled_and_preserves_page(
    page_with_fm: Path,
    status: str,
) -> None:
    before = page_with_fm.read_bytes()

    with pytest.raises(LegacySemanticMutationDisabled):
        tba._mark_unfit(page_with_fm, status)

    assert page_with_fm.read_bytes() == before


def test_is_marked_unfit_detects_status(page_with_fm: Path) -> None:
    assert tba._is_marked_unfit("example") is False
    from chronovisor.core.frontmatter import patch as fm_patch

    page_with_fm.write_text(
        fm_patch(page_with_fm.read_text(), {"tag_status": tba.TAG_STATUS_NO_FIT})
    )
    assert tba._is_marked_unfit("example") is True


def test_is_marked_unfit_handles_missing_page() -> None:
    # Page that doesn't exist anywhere should report False, not raise.
    assert tba._is_marked_unfit("definitely-not-a-real-page-id") is False


def test_main_fails_before_store_or_local_model(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["tag_backfill_apply.py"])

    with pytest.raises(LegacySemanticMutationDisabled):
        tba.main()
