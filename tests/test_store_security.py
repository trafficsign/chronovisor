from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

from chronovisor.core import store
from chronovisor.core.canonical_document import validate_canonical_document
from chronovisor.hosts import server


def test_all_pages_returns_only_stable_nonreserved_canonical_documents(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pages = tmp_path / "pages"
    nested = pages / "nested"
    nested.mkdir(parents=True)
    template = "---\ntitle: {title}\nstatus: {status}\ntype: knowledge\n---\nbody\n"
    stable = nested / "stable.md"
    stable.write_text(template.format(title="Stable", status="stable"))
    for name, status in (("draft", "draft"), ("deprecated", "deprecated")):
        (pages / f"{name}.md").write_text(
            template.format(title=name, status=status), encoding="utf-8"
        )
    (pages / "index.md").write_text(
        template.format(title="Reserved", status="stable"), encoding="utf-8"
    )
    (pages / "invalid.md").write_text(
        "---\ntitle: Invalid\nstatus: stable\n---\nbody\n", encoding="utf-8"
    )
    monkeypatch.setattr(store, "PAGES_DIR", pages)

    assert store.all_pages() == [stable.resolve()]


def test_init_chronovisor_corrects_private_directory_modes(tmp_path: Path) -> None:
    root = tmp_path / "wiki"
    context = store.RuntimeContext(root)
    for directory in (root, context.raw_dir, context.pages_dir, context.system_dir):
        directory.mkdir(parents=True, exist_ok=True)
        directory.chmod(0o755)

    store.init_chronovisor(context)

    assert all(
        stat.S_IMODE(directory.stat().st_mode) == 0o700
        for directory in (root, context.raw_dir, context.pages_dir, context.system_dir)
    )


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
    normal_content = (
        "---\ntitle: Normal\nstatus: stable\ntype: knowledge\n---\n# Normal\n"
    )
    normal.write_text(normal_content, encoding="utf-8")
    system_normal = system / "system-normal.md"
    system_content = "---\ntitle: System normal\nstatus: stable\n---\n# System normal\n"
    system_normal.write_text(system_content, encoding="utf-8")
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
    monkeypatch.setattr(server, "get_store", FakeStore)
    monkeypatch.setattr(server, "_append_pull_log", lambda _row: None)
    read = getattr(server.chronovisor_read, "fn", server.chronovisor_read)

    for page_id, expected in (
        ("normal", normal_content),
        ("system-normal", system_content),
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
def test_generated_schema_teaches_the_canonical_page_contract() -> None:
    document = validate_canonical_document(
        store.SCHEMA_CONTENT.encode("utf-8"),
        namespace="pages",
        path="schema.md",
        require_stable=True,
    )

    assert document.metadata["type"] == "knowledge"
    assert "[[" not in store.SCHEMA_CONTENT
    assert "(<jt-v10-probability-contexts.md>)" in store.SCHEMA_CONTENT
