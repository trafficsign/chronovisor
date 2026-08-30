from __future__ import annotations

from pathlib import Path

import pytest

from chronovisor.core import page_mutation, reserved_documents, store
from chronovisor.core.canonical_document import CanonicalDocument, serialize_document
from chronovisor.core.link_fix import atomic_write
from chronovisor.core.okf_v02 import validate_pages_bundle


def _page(
    title: str,
    *,
    status: str = "stable",
    body: str = "Body.\n",
) -> bytes:
    return serialize_document(
        CanonicalDocument(
            metadata={"title": title, "status": status, "type": "knowledge"},
            body=body.encode(),
        )
    )


def _fresh_root(tmp_path: Path, name: str = "wiki") -> Path:
    root = tmp_path / name
    store.init_chronovisor(store.RuntimeContext(root))
    return root


def test_reserved_index_projects_only_stable_nonreserved_pages(tmp_path: Path) -> None:
    root = _fresh_root(tmp_path)
    pages = root / "pages"
    (pages / "stable.md").write_bytes(_page("Stable"))
    (pages / "draft.md").write_bytes(_page("Draft", status="draft"))
    (pages / "deprecated.md").write_bytes(
        _page("Deprecated", status="deprecated")
    )

    rendered = reserved_documents.rebuild_pages_index(pages)

    assert b"[Stable](stable.md)" in rendered
    assert b"draft.md" not in rendered
    assert b"deprecated.md" not in rendered
    assert rendered.count(b"index.md") == 0
    assert rendered.count(b"log.md") == 0
    errors = [
        issue for issue in validate_pages_bundle(pages) if issue.severity == "error"
    ]
    assert not errors


def test_outer_mutation_refreshes_once_and_custom_root_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = _fresh_root(tmp_path, "target")
    untouched = _fresh_root(tmp_path, "untouched")
    untouched_index = (untouched / "pages" / "index.md").read_bytes()
    calls = 0
    original = reserved_documents.rebuild_pages_index

    def counted(pages_dir: Path) -> bytes:
        nonlocal calls
        calls += 1
        return original(pages_dir)

    monkeypatch.setattr(reserved_documents, "rebuild_pages_index", counted)
    with page_mutation.chronovisor_mutation_lock(pages_dir=target / "pages"):
        with page_mutation.chronovisor_mutation_lock(pages_dir=target / "pages"):
            (target / "pages" / "created.md").write_bytes(_page("Created"))

    assert calls == 1
    assert b"[Created](created.md)" in (target / "pages" / "index.md").read_bytes()
    assert (untouched / "pages" / "index.md").read_bytes() == untouched_index


def test_legacy_mutation_does_not_create_or_validate_final_index(
    tmp_path: Path,
) -> None:
    root = tmp_path / "legacy"
    pages = root / "pages"
    pages.mkdir(parents=True)
    for name in ("index.md", "log.md", "schema.md"):
        (root / name).write_text("legacy\n", encoding="utf-8")
    (pages / "legacy.md").write_text("legacy page without canonical metadata\n")

    with page_mutation.chronovisor_mutation_lock(pages_dir=pages):
        (pages / "mutation-canary").write_text("committed\n", encoding="utf-8")

    assert not (pages / "index.md").exists()
    assert (pages / "mutation-canary").read_text(encoding="utf-8") == "committed\n"


def test_mutation_projection_tracks_update_deprecate_restore_and_delete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _fresh_root(tmp_path)
    pages = root / "pages"
    path = pages / "alpha.md"
    untouched = pages / "untouched.md"

    with page_mutation.chronovisor_mutation_lock(pages_dir=pages):
        path.write_bytes(_page("Alpha"))
        untouched.write_bytes(_page("Untouched"))
    assert b"[Alpha](alpha.md)" in (pages / "index.md").read_bytes()
    assert b"[Untouched](untouched.md)" in (pages / "index.md").read_bytes()

    def unexpected_full_rebuild(_pages_dir: Path) -> bytes:
        raise AssertionError("exact mutation receipts must not rescan all pages")

    monkeypatch.setattr(
        reserved_documents,
        "rebuild_pages_index",
        unexpected_full_rebuild,
    )

    with page_mutation.chronovisor_mutation_lock(
        pages_dir=pages,
        changed_paths=[path],
    ):
        atomic_write(path, _page("Alpha updated").decode())
    assert b"[Alpha updated](alpha.md)" in (pages / "index.md").read_bytes()
    assert b"[Untouched](untouched.md)" in (pages / "index.md").read_bytes()

    with page_mutation.chronovisor_mutation_lock(
        pages_dir=pages,
        changed_paths=[path],
    ):
        atomic_write(path, _page("Alpha updated", status="deprecated").decode())
    assert b"alpha.md" not in (pages / "index.md").read_bytes()
    assert b"[Untouched](untouched.md)" in (pages / "index.md").read_bytes()

    with page_mutation.chronovisor_mutation_lock(
        pages_dir=pages,
        changed_paths=[path],
    ):
        atomic_write(path, _page("Alpha restored").decode())
    assert b"[Alpha restored](alpha.md)" in (pages / "index.md").read_bytes()

    with page_mutation.chronovisor_mutation_lock(
        pages_dir=pages,
        changed_paths=[path],
    ):
        path.unlink()
    assert b"alpha.md" not in (pages / "index.md").read_bytes()
    assert b"[Untouched](untouched.md)" in (pages / "index.md").read_bytes()


def test_nonpage_receipt_does_not_rebuild_pages_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _fresh_root(tmp_path)
    pages = root / "pages"
    index_before = (pages / "index.md").read_bytes()
    system_path = root / "system" / "current-state.md"

    def unexpected_full_rebuild(_pages_dir: Path) -> bytes:
        raise AssertionError("a system-only mutation must not scan pages")

    monkeypatch.setattr(
        reserved_documents,
        "rebuild_pages_index",
        unexpected_full_rebuild,
    )
    with page_mutation.chronovisor_mutation_lock(
        pages_dir=pages,
        changed_paths=[system_path],
    ):
        system_path.write_bytes(_page("Current State"))

    assert (pages / "index.md").read_bytes() == index_before


def test_refresh_failure_does_not_mask_original_mutation_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _fresh_root(tmp_path)

    def broken(_pages_dir: Path) -> bytes:
        raise RuntimeError("refresh failed")

    monkeypatch.setattr(reserved_documents, "rebuild_pages_index", broken)
    with pytest.raises(ValueError, match="original failure"):
        with page_mutation.chronovisor_mutation_lock(pages_dir=root / "pages"):
            raise ValueError("original failure")
