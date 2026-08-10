from __future__ import annotations

import json
from pathlib import Path

import pytest

from chronovisor.core import store
from chronovisor.hosts import server


def test_find_page_rejects_paths_and_symlink_escape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pages = tmp_path / "wiki" / "pages"
    nested = pages / "nested"
    nested.mkdir(parents=True)
    normal = nested / "normal.md"
    normal.write_text("normal", encoding="utf-8")
    outside = tmp_path / "outside.md"
    outside.write_text("secret", encoding="utf-8")
    (pages / "page-link.md").symlink_to(outside)
    monkeypatch.setattr(store, "PAGES_DIR", pages)

    assert store.find_page("normal") == normal.resolve()
    for page_id in (
        "",
        ".",
        "..",
        str(outside.with_suffix("")),
        "../outside",
        "nested/normal",
        "nested\\normal",
        "norm*",
        "norm?l",
        "[n]ormal",
        "page-link",
    ):
        assert store.find_page(page_id) is None


def test_chronovisor_read_confines_pages_and_system(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "wiki"
    pages = root / "pages"
    system = root / "system"
    nested = pages / "nested"
    nested.mkdir(parents=True)
    system.mkdir()
    normal = nested / "normal.md"
    normal.write_text("# Normal\n", encoding="utf-8")
    system_normal = system / "system-normal.md"
    system_normal.write_text("# System normal\n", encoding="utf-8")
    outside = tmp_path / "outside.md"
    outside.write_text("SECRET", encoding="utf-8")
    (pages / "page-link.md").symlink_to(outside)
    (system / "system-link.md").symlink_to(outside)

    class FakeStore:
        def refresh(self) -> None:
            return None

        def meta(self, _page_id: str) -> None:
            return None

        def outlinks(self, _page_id: str) -> list[str]:
            return []

        def backlinks(self, _page_id: str) -> list[str]:
            return []

    monkeypatch.setattr(store, "CHRONOVISOR_ROOT", root)
    monkeypatch.setattr(store, "PAGES_DIR", pages)
    monkeypatch.setattr(server, "CHRONOVISOR_ROOT", root)
    monkeypatch.setattr(server, "SYSTEM_DIR", system)
    monkeypatch.setattr(server, "get_store", FakeStore)
    monkeypatch.setattr(server, "_append_pull_log", lambda _row: None)
    read = getattr(server.chronovisor_read, "fn", server.chronovisor_read)

    for page_id, expected in (
        ("normal", "# Normal\n"),
        ("system-normal", "# System normal\n"),
    ):
        payload = json.loads(read(page_id))
        assert payload["page_id"] == page_id
        assert payload["content"] == expected

    for page_id in (
        str(outside.with_suffix("")),
        "../outside",
        "nested/normal",
        "nested\\normal",
        "norm*",
        "norm?l",
        "[n]ormal",
        "page-link",
        "system-link",
    ):
        assert json.loads(read(page_id)) == {
            "error": f"Page '{page_id}' not found"
        }
