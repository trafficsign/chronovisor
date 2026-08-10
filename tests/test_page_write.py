from __future__ import annotations

from pathlib import Path

import pytest

from chronovisor.core import page_mutation
from chronovisor.ingest import page_write


@pytest.fixture(autouse=True)
def _valid_okf_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "wiki-root"
    root.mkdir()
    for name in ("index.md", "log.md", "schema.md"):
        (root / name).write_text("legacy\n", encoding="utf-8")
    monkeypatch.setattr(page_mutation, "CHRONOVISOR_ROOT", root)


def _doc(body: str, *, status: str = "stable") -> str:
    return f"---\ntitle: Page\nstatus: {status}\ntype: knowledge\n---\n{body}"


def _prepare(path: Path, content: str) -> page_write.PreparedWikiWrite:
    return page_write.prepare_page_write(
        path,
        content,
        namespace="pages",
        source_path=path.name,
    )


def test_stale_preimage_never_overwrites_foreign_change(tmp_path: Path) -> None:
    path = tmp_path / "page.md"
    path.write_text(_doc("original\n"), encoding="utf-8")
    plan = _prepare(path, _doc("generated\n"))
    path.write_text("foreign\n", encoding="utf-8")

    result = page_write.apply_page_writes([plan])

    assert result["status"] == "retry"
    assert "changed before apply" in result["reason"]
    assert path.read_text(encoding="utf-8") == "foreign\n"


def test_concurrent_identical_output_is_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "page.md"
    path.write_text(_doc("original\n"), encoding="utf-8")
    plan = _prepare(path, _doc("generated\n"))
    path.write_text(_doc("generated\n"), encoding="utf-8")

    result = page_write.apply_page_writes([plan])

    assert result["status"] == "unchanged"
    assert path.read_text(encoding="utf-8") == _doc("generated\n")


def test_batch_failure_rolls_back_only_successful_owned_writes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    first = tmp_path / "first.md"
    second = tmp_path / "second.md"
    first.write_text(_doc("first-old\n"), encoding="utf-8")
    second.write_text(_doc("second-old\n"), encoding="utf-8")
    plans = [
        _prepare(first, _doc("first-new\n")),
        _prepare(second, _doc("second-new\n")),
    ]
    real_atomic_write = page_write.atomic_write
    calls = 0

    def fail_second(path: Path, content: str) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("disk full")
        real_atomic_write(path, content)

    monkeypatch.setattr(page_write, "atomic_write", fail_second)

    result = page_write.apply_page_writes(plans)

    assert result["status"] == "retry"
    assert result["rolled_back"][str(first)] is True
    assert first.read_text(encoding="utf-8") == _doc("first-old\n")
    assert second.read_text(encoding="utf-8") == _doc("second-old\n")


def test_batch_failure_removes_an_owned_create(tmp_path: Path, monkeypatch) -> None:
    created = tmp_path / "created.md"
    second = tmp_path / "second.md"
    second.write_text(_doc("second-old\n"), encoding="utf-8")
    plans = [
        _prepare(created, _doc("created-new\n")),
        _prepare(second, _doc("second-new\n")),
    ]
    real_atomic_write = page_write.atomic_write
    calls = 0

    def fail_second(path: Path, content: str) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("disk full")
        real_atomic_write(path, content)

    monkeypatch.setattr(page_write, "atomic_write", fail_second)

    result = page_write.apply_page_writes(plans)

    assert result["status"] == "retry"
    assert result["rolled_back"][str(created)] is True
    assert not created.exists()
    assert second.read_text(encoding="utf-8") == _doc("second-old\n")


def test_failure_after_replace_is_detected_and_owned_write_is_rolled_back(
    tmp_path: Path,
    monkeypatch,
) -> None:
    path = tmp_path / "page.md"
    path.write_text(_doc("old\n"), encoding="utf-8")
    plan = _prepare(path, _doc("new\n"))
    real_atomic_write = page_write.atomic_write
    raised = False

    def replace_then_fail(target: Path, content: str) -> None:
        nonlocal raised
        real_atomic_write(target, content)
        if not raised:
            raised = True
            raise OSError("late failure")

    monkeypatch.setattr(page_write, "atomic_write", replace_then_fail)

    result = page_write.apply_page_writes([plan])

    assert result["status"] == "retry"
    assert result["rolled_back"][str(path)] is True
    assert path.read_text(encoding="utf-8") == _doc("old\n")


def test_rollback_does_not_clobber_foreign_bytes(tmp_path: Path, monkeypatch) -> None:
    first = tmp_path / "first.md"
    second = tmp_path / "second.md"
    first.write_text(_doc("first-old\n"), encoding="utf-8")
    second.write_text(_doc("second-old\n"), encoding="utf-8")
    plans = [
        _prepare(first, _doc("first-new\n")),
        _prepare(second, _doc("second-new\n")),
    ]
    real_atomic_write = page_write.atomic_write
    calls = 0

    def foreign_then_fail(path: Path, content: str) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            real_atomic_write(path, content)
            path.write_text("foreign\n", encoding="utf-8")
            return
        raise OSError("second failed")

    monkeypatch.setattr(page_write, "atomic_write", foreign_then_fail)

    result = page_write.apply_page_writes(plans)

    assert result["status"] == "retry"
    assert result["rolled_back"][str(first)] is False
    assert first.read_text(encoding="utf-8") == "foreign\n"
    assert second.read_text(encoding="utf-8") == _doc("second-old\n")


def test_batch_rejects_duplicate_page_ids_before_writing(tmp_path: Path) -> None:
    first = tmp_path / "a" / "same.md"
    second = tmp_path / "b" / "same.md"
    plans = [
        _prepare(first, _doc("first\n")),
        _prepare(second, _doc("second\n")),
    ]

    result = page_write.apply_page_writes(plans)

    assert result["status"] == "retry"
    assert result["reason"] == "duplicate_page_id"
    assert not first.exists()
    assert not second.exists()


def test_wiki_create_rejects_existing_page_id_in_other_folder(
    tmp_path: Path,
    monkeypatch,
) -> None:
    pages_dir = tmp_path / "wiki" / "pages"
    system_dir = tmp_path / "wiki" / "system"
    existing = pages_dir / "organized" / "generated.md"
    target = pages_dir / "preferred" / "generated.md"
    existing.parent.mkdir(parents=True)
    system_dir.mkdir(parents=True)
    existing.write_text(_doc("existing\n"), encoding="utf-8")
    monkeypatch.setattr(page_write, "PAGES_DIR", pages_dir)
    monkeypatch.setattr(page_write, "SYSTEM_DIR", system_dir)

    result = page_write.apply_page_writes(
        [page_write.prepare_page_write(target, _doc("duplicate\n"))]
    )

    assert result["status"] == "retry"
    assert "already exists" in result["reason"]
    assert existing.read_text(encoding="utf-8") == _doc("existing\n")
    assert not target.exists()


@pytest.mark.parametrize("status", ["", "active", "draft", "deprecated"])
def test_writer_requires_explicit_stable_status(tmp_path: Path, status: str) -> None:
    content = (
        "---\ntitle: Page\n---\nbody\n" if not status else _doc("body\n", status=status)
    )

    with pytest.raises(ValueError, match="status|stable"):
        _prepare(tmp_path / "page.md", content)


def test_writer_preserves_nested_unknown_yaml_and_body_bytes(tmp_path: Path) -> None:
    path = tmp_path / "page.md"
    original = (
        "---\ntitle: Page\nstatus: stable\ntype: knowledge\nextension:\n  nested:\n    - one\n"
        "---\nbody with  two spaces\r\n"
    )
    path.write_bytes(original.encode("utf-8"))
    candidate = original.replace("one", "two")

    result = page_write.apply_page_writes([_prepare(path, candidate)])

    assert result["status"] == "applied"
    assert path.read_bytes().endswith(b"body with  two spaces\r\n")


def test_writer_preserves_existing_unknown_yaml_absent_from_generated_candidate(
    tmp_path: Path,
) -> None:
    path = tmp_path / "page.md"
    path.write_text(
        "---\ntitle: Existing\nstatus: stable\ntype: knowledge\n"
        "extension:\n  nested: [one, two]\n---\nold body\n",
        encoding="utf-8",
    )

    result = page_write.apply_page_writes([_prepare(path, _doc("generated body\n"))])

    assert result["status"] == "applied"
    written = path.read_text(encoding="utf-8")
    assert "extension:\n  nested:\n  - one\n  - two\n" in written
    assert written.endswith("generated body\n")


def test_writer_requires_complete_explicit_location_outside_roots(
    tmp_path: Path,
) -> None:
    path = tmp_path / "page.md"

    with pytest.raises(page_write.WikiWriteError, match="required outside"):
        page_write.prepare_page_write(path, _doc("body\n"))
    with pytest.raises(page_write.WikiWriteError, match="provided together"):
        page_write.prepare_page_write(
            path,
            _doc("body\n"),
            namespace="pages",
        )


def test_writer_rejects_canonical_parent_symlink_escape(
    tmp_path: Path,
    monkeypatch,
) -> None:
    pages = tmp_path / "wiki" / "pages"
    system = tmp_path / "wiki" / "system"
    outside = tmp_path / "outside"
    pages.mkdir(parents=True)
    system.mkdir(parents=True)
    outside.mkdir()
    (pages / "hubs").symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(page_write, "PAGES_DIR", pages)
    monkeypatch.setattr(page_write, "SYSTEM_DIR", system)

    with pytest.raises(page_write.WikiWriteError, match="symlink"):
        page_write.prepare_page_write(
            pages / "hubs" / "escaped.md",
            _doc("body\n"),
        )

    assert not (outside / "escaped.md").exists()


def test_writer_rechecks_parent_symlink_under_apply_lock(
    tmp_path: Path,
    monkeypatch,
) -> None:
    pages = tmp_path / "wiki" / "pages"
    system = tmp_path / "wiki" / "system"
    outside = tmp_path / "outside"
    parent = pages / "hubs"
    parent.mkdir(parents=True)
    system.mkdir(parents=True)
    outside.mkdir()
    monkeypatch.setattr(page_write, "PAGES_DIR", pages)
    monkeypatch.setattr(page_write, "SYSTEM_DIR", system)
    target = parent / "escaped.md"
    plan = page_write.prepare_page_write(target, _doc("body\n"))

    parent.rename(pages / "hubs-before-swap")
    parent.symlink_to(outside, target_is_directory=True)
    result = page_write.apply_page_writes([plan])

    assert result["status"] == "retry"
    assert "symlink" in result["reason"]
    assert not (outside / "escaped.md").exists()
