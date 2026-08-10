from __future__ import annotations

from pathlib import Path

from chronovisor.core.lexical_index import LexicalIndex


def test_inverted_bm25_and_anchor_channels_find_japanese_and_metadata(
    tmp_path: Path,
) -> None:
    pages = tmp_path / "pages"
    pages.mkdir()
    semantic = pages / "semantic"
    semantic.mkdir()
    target = semantic / "nemotron-search.md"
    target.write_text(
        """---
title: Nemotron 検索設計
status: stable
type: knowledge
updated: 2026-07-24
tags: [d/ai-tools, t/retrieval]
entities: [NVIDIA, Nemotron]
raw_keywords: [agentic retrieval]
---
日本語の意味検索と関連ページ探索を高速化する。
""",
        encoding="utf-8",
    )
    other = semantic / "unrelated.md"
    other.write_text(
        """---
title: unrelated
status: stable
type: knowledge
updated: 2026-07-24
---
別の記録。
""",
        encoding="utf-8",
    )
    index = LexicalIndex(
        path=tmp_path / "lexical.sqlite",
        pages=lambda: [target, other],
        refresh_interval_seconds=0,
    )

    index.build()

    assert index.query("意味検索", top_n=2)[0].page_id == "nemotron-search"
    assert index.anchor_query("NVIDIA", top_n=2)[0].page_id == "nemotron-search"
    assert index.anchor_query("retrieval", top_n=2)[0].page_id == "nemotron-search"
    assert index.stats()["backend"] == "sqlite_inverted_bm25"


def test_inverted_bm25_refresh_removes_deleted_pages(tmp_path: Path) -> None:
    page = tmp_path / "obsolete.md"
    page.write_text(
        "---\ntitle: Obsolete\nstatus: stable\ntype: knowledge\nupdated: 2026-07-24\n---\nretired token\n",
        encoding="utf-8",
    )
    index = LexicalIndex(
        path=tmp_path / "lexical.sqlite",
        pages=lambda: [page] if page.exists() else [],
        refresh_interval_seconds=0,
    )
    index.build()
    assert index.query("retired")

    page.unlink()
    index.build(force=True)

    assert index.query("retired") == []


def test_lexical_index_includes_only_canonical_stable_pages_and_system(
    tmp_path: Path,
) -> None:
    pages = tmp_path / "pages"
    system = tmp_path / "system"
    pages.mkdir()
    system.mkdir()
    documents = {
        pages / "stable.md": (
            "---\ntitle: Stable\nstatus: stable\ntype: knowledge\n---\n"
            "stabletoken\n"
        ),
        pages / "draft.md": (
            "---\ntitle: Draft\nstatus: draft\ntype: knowledge\n---\ndrafttoken\n"
        ),
        pages / "deprecated.md": (
            "---\ntitle: Deprecated\nstatus: deprecated\ntype: knowledge\n---\n"
            "deprecatedtoken\n"
        ),
        pages / "missing-type.md": (
            "---\ntitle: Missing\nstatus: stable\n---\nmissingtypetoken\n"
        ),
        pages / "legacy-link.md": (
            "---\ntitle: Legacy\nstatus: stable\ntype: knowledge\n---\n"
            "[[stable]] legacytoken\n"
        ),
        system / "current-state.md": (
            "---\ntitle: Current\nstatus: stable\n---\n"
            "[Stable](</pages/stable.md>) systemtoken\n"
        ),
    }
    for path, content in documents.items():
        path.write_text(content, encoding="utf-8")
    index = LexicalIndex(
        path=tmp_path / "lexical.sqlite",
        pages=lambda: list(documents),
        refresh_interval_seconds=0,
    )

    index.build()

    assert [row.page_id for row in index.query("stabletoken")] == ["stable"]
    assert [row.page_id for row in index.query("systemtoken")] == ["current-state"]
    for excluded in ("drafttoken", "deprecatedtoken", "missingtypetoken", "legacytoken"):
        assert index.query(excluded) == []
