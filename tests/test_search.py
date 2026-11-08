from __future__ import annotations

from chronovisor.index import index_store as index_store_mod
from chronovisor.search import search
from chronovisor.search.search import ScoredPage, apply_filters


def test_bm25_read_only_refresh_persists_dirty_cache_on_next_normal_build(
    tmp_path, monkeypatch
) -> None:
    page_path = tmp_path / "page.md"
    page_path.write_text(
        "---\ntitle: Page\nupdated: 2026-01-01\n---\nsearchable body\n"
    )
    cache_path = tmp_path / ".index" / "bm25.json"
    monkeypatch.setattr(search, "_BM25_CACHE_FILE", cache_path)
    monkeypatch.setattr(search, "searchable_pages", lambda: [page_path])
    monkeypatch.setenv("CHRONOVISOR_READ_ONLY", "1")
    index = search.BM25Index()

    index.build()

    assert not cache_path.exists()
    monkeypatch.delenv("CHRONOVISOR_READ_ONLY")
    index.build()
    assert cache_path.exists()


def page(
    page_id: str,
    score: float,
    *,
    status: str = "stable",
    folder: str = "",
    page_type: str = "knowledge",
) -> ScoredPage:
    return ScoredPage(
        page_id=page_id,
        title=page_id,
        folder=folder,
        updated="2026-06-11",
        score=score,
        status=status,
        page_type=page_type,
    )


def test_searchable_pages_includes_system_pages(tmp_path, monkeypatch) -> None:
    pages_dir = tmp_path / "pages"
    system_dir = tmp_path / "system"
    pages_dir.mkdir()
    system_dir.mkdir()
    page = pages_dir / "normal.md"
    system = system_dir / "claude-code.md"
    page.write_text("# Normal\n", encoding="utf-8")
    system.write_text("# System\n", encoding="utf-8")

    class FakeStore:
        def refresh(self) -> None:
            pass

        def all_page_ids(self, *, include_system: bool = False):
            assert include_system is True
            return {"normal", "claude-code"}

        def meta(self, page_id: str):
            return {"path": str(page if page_id == "normal" else system)}

    monkeypatch.setattr(index_store_mod, "get_store", lambda: FakeStore())

    assert search.searchable_pages() == [page, system]


def test_apply_filters_exposes_only_canonical_stable_pages() -> None:
    results = apply_filters(
        [
            page("stable", 1.0, status="stable"),
            page("draft", 2.0, status="draft"),
            page("old", 2.0, status="deprecated"),
            page("draft", 3.0, status="draft"),
        ]
    )

    assert [(result.page_id, result.status) for result in results] == [
        ("stable", "stable")
    ]


def test_apply_filters_excludes_reference_pages_by_default() -> None:
    results = apply_filters(
        [
            page("knowledge", 1.0),
            page("car", 2.0, folder="car-spec", page_type="reference"),
        ]
    )

    assert [result.page_id for result in results] == ["knowledge"]


def test_apply_filters_allows_reference_pages_when_folder_requested() -> None:
    results = apply_filters(
        [
            page("knowledge", 1.0),
            page("car", 2.0, folder="car-spec", page_type="reference"),
        ],
        folder="car-spec",
    )

    assert [result.page_id for result in results] == ["car"]
