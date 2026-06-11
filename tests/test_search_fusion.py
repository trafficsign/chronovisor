from __future__ import annotations

from llm_wiki_mcp import search
from llm_wiki_mcp.search import ScoredPage, fuse_results


def page(page_id: str, score: float) -> ScoredPage:
    return ScoredPage(
        page_id=page_id,
        title=page_id,
        folder="",
        updated="2026-06-11",
        score=score,
    )


def test_fusion_keeps_strong_bm25_match_ahead_of_semantic_only_neighbor() -> None:
    results = fuse_results(
        bm25_results=[page("exact", 100.0), page("neighbor", 10.0)],
        semantic_results=[page("neighbor", 0.9), page("other", 0.8)],
        weights={"bm25": 1.0, "semantic": 1.0},
    )

    assert [result.page_id for result in results[:2]] == ["exact", "neighbor"]


def test_searchable_pages_includes_system_pages(tmp_path, monkeypatch) -> None:
    pages_dir = tmp_path / "pages"
    system_dir = tmp_path / "system"
    pages_dir.mkdir()
    system_dir.mkdir()
    page = pages_dir / "normal.md"
    system = system_dir / "claude-code.md"
    page.write_text("# Normal\n", encoding="utf-8")
    system.write_text("# System\n", encoding="utf-8")
    monkeypatch.setattr(search, "SYSTEM_DIR", system_dir)
    monkeypatch.setattr(search, "all_pages", lambda: [page])

    assert search.searchable_pages() == [page, system]
