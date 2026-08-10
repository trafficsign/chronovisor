from __future__ import annotations

import threading
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


def test_mutation_projection_tracks_update_deprecate_restore_and_delete(
    tmp_path: Path,
) -> None:
    root = _fresh_root(tmp_path)
    pages = root / "pages"
    path = pages / "alpha.md"

    with page_mutation.chronovisor_mutation_lock(pages_dir=pages):
        path.write_bytes(_page("Alpha"))
    assert b"[Alpha](alpha.md)" in (pages / "index.md").read_bytes()

    with page_mutation.chronovisor_mutation_lock(pages_dir=pages):
        atomic_write(path, _page("Alpha updated").decode())
    assert b"[Alpha updated](alpha.md)" in (pages / "index.md").read_bytes()

    with page_mutation.chronovisor_mutation_lock(pages_dir=pages):
        atomic_write(path, _page("Alpha updated", status="deprecated").decode())
    assert b"alpha.md" not in (pages / "index.md").read_bytes()

    with page_mutation.chronovisor_mutation_lock(pages_dir=pages):
        atomic_write(path, _page("Alpha restored").decode())
    assert b"[Alpha restored](alpha.md)" in (pages / "index.md").read_bytes()

    with page_mutation.chronovisor_mutation_lock(pages_dir=pages):
        path.unlink()
    assert b"alpha.md" not in (pages / "index.md").read_bytes()


def test_startup_repairs_stale_index_without_changing_immutable_proof(
    tmp_path: Path,
) -> None:
    root = _fresh_root(tmp_path)
    proof = root / "runtime" / "bootstrap-layout.json"
    proof_before = proof.read_bytes()
    (root / "pages" / "late.md").write_bytes(_page("Late"))
    assert b"late.md" not in (root / "pages" / "index.md").read_bytes()

    store.init_chronovisor(store.RuntimeContext(root))

    assert b"[Late](late.md)" in (root / "pages" / "index.md").read_bytes()
    assert proof.read_bytes() == proof_before


def test_startup_index_repair_serializes_with_concurrent_page_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _fresh_root(tmp_path)
    pages = root / "pages"
    alpha = pages / "alpha.md"
    with page_mutation.chronovisor_mutation_lock(pages_dir=pages):
        alpha.write_bytes(_page("Alpha"))

    entered = threading.Event()
    release = threading.Event()
    mutation_done = threading.Event()
    failures: list[BaseException] = []
    original = reserved_documents.rebuild_pages_index
    calls = 0

    def delayed(pages_dir: Path) -> bytes:
        nonlocal calls
        calls += 1
        if calls == 1:
            entered.set()
            assert release.wait(timeout=5)
        return original(pages_dir)

    def repair() -> None:
        try:
            store.init_chronovisor(store.RuntimeContext(root))
        except BaseException as exc:  # pragma: no cover - asserted below
            failures.append(exc)

    def mutate() -> None:
        try:
            with page_mutation.chronovisor_mutation_lock(pages_dir=pages):
                alpha.write_bytes(_page("Alpha", status="deprecated"))
                (pages / "beta.md").write_bytes(_page("Beta"))
        except BaseException as exc:  # pragma: no cover - asserted below
            failures.append(exc)
        finally:
            mutation_done.set()

    monkeypatch.setattr(reserved_documents, "rebuild_pages_index", delayed)
    repair_thread = threading.Thread(target=repair)
    repair_thread.start()
    assert entered.wait(timeout=5)
    mutation_thread = threading.Thread(target=mutate)
    mutation_thread.start()
    assert not mutation_done.wait(timeout=0.1)
    release.set()
    repair_thread.join(timeout=5)
    mutation_thread.join(timeout=5)

    assert not repair_thread.is_alive()
    assert not mutation_thread.is_alive()
    assert not failures
    assert (pages / "index.md").read_bytes() == reserved_documents.render_pages_index(
        [reserved_documents.PageIndexEntry("beta.md", "Beta")]
    )


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
