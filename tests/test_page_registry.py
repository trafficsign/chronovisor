from __future__ import annotations

from pathlib import Path

from chronovisor.ingest.page_registry import PageRegistry


def _page(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"---\ntitle: {path.stem}\n---\n", encoding="utf-8")


def test_page_paths_exclude_reserved_documents_and_symlinks(tmp_path: Path) -> None:
    for relative in (
        "pages/index.md",
        "pages/log.md",
        "pages/schema.md",
        "pages/nested/index.md",
        "pages/nested/log.md",
        "pages/nested/schema.md",
        "system/index.md",
        "system/log.md",
    ):
        _page(tmp_path / relative)
    _page(tmp_path / "pages" / "concept-index.md")
    _page(tmp_path / "system" / "schema.md")
    outside = tmp_path / "outside.md"
    _page(outside)
    (tmp_path / "pages" / "outside-link.md").symlink_to(outside)
    (tmp_path / "pages" / "inside-link.md").symlink_to(
        tmp_path / "pages" / "concept-index.md"
    )

    paths = PageRegistry._page_paths(tmp_path, include_system=True)

    assert [path.relative_to(tmp_path).as_posix() for path in paths] == [
        "pages/concept-index.md",
        "system/schema.md",
    ]
