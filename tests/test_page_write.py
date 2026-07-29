from __future__ import annotations

from pathlib import Path

from chronovisor.ingest import page_write


def test_stale_preimage_never_overwrites_foreign_change(tmp_path: Path) -> None:
    path = tmp_path / "page.md"
    path.write_text("original\n", encoding="utf-8")
    plan = page_write.prepare_page_write(path, "generated\n")
    path.write_text("foreign\n", encoding="utf-8")

    result = page_write.apply_page_writes([plan])

    assert result["status"] == "retry"
    assert "changed before apply" in result["reason"]
    assert path.read_text(encoding="utf-8") == "foreign\n"


def test_concurrent_identical_output_is_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "page.md"
    path.write_text("original\n", encoding="utf-8")
    plan = page_write.prepare_page_write(path, "generated\n")
    path.write_text("generated\n", encoding="utf-8")

    result = page_write.apply_page_writes([plan])

    assert result["status"] == "unchanged"
    assert path.read_text(encoding="utf-8") == "generated\n"


def test_batch_failure_rolls_back_only_successful_owned_writes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    first = tmp_path / "first.md"
    second = tmp_path / "second.md"
    first.write_text("first-old\n", encoding="utf-8")
    second.write_text("second-old\n", encoding="utf-8")
    plans = [
        page_write.prepare_page_write(first, "first-new\n"),
        page_write.prepare_page_write(second, "second-new\n"),
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
    assert first.read_text(encoding="utf-8") == "first-old\n"
    assert second.read_text(encoding="utf-8") == "second-old\n"


def test_batch_failure_removes_an_owned_create(tmp_path: Path, monkeypatch) -> None:
    created = tmp_path / "created.md"
    second = tmp_path / "second.md"
    second.write_text("second-old\n", encoding="utf-8")
    plans = [
        page_write.prepare_page_write(created, "created-new\n"),
        page_write.prepare_page_write(second, "second-new\n"),
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
    assert second.read_text(encoding="utf-8") == "second-old\n"


def test_failure_after_replace_is_detected_and_owned_write_is_rolled_back(
    tmp_path: Path,
    monkeypatch,
) -> None:
    path = tmp_path / "page.md"
    path.write_text("old\n", encoding="utf-8")
    plan = page_write.prepare_page_write(path, "new\n")
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
    assert path.read_text(encoding="utf-8") == "old\n"


def test_rollback_does_not_clobber_foreign_bytes(tmp_path: Path, monkeypatch) -> None:
    first = tmp_path / "first.md"
    second = tmp_path / "second.md"
    first.write_text("first-old\n", encoding="utf-8")
    second.write_text("second-old\n", encoding="utf-8")
    plans = [
        page_write.prepare_page_write(first, "first-new\n"),
        page_write.prepare_page_write(second, "second-new\n"),
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
    assert second.read_text(encoding="utf-8") == "second-old\n"


def test_batch_rejects_duplicate_page_ids_before_writing(tmp_path: Path) -> None:
    first = tmp_path / "a" / "same.md"
    second = tmp_path / "b" / "same.md"
    plans = [
        page_write.prepare_page_write(first, "first\n"),
        page_write.prepare_page_write(second, "second\n"),
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
    existing.write_text("existing\n", encoding="utf-8")
    monkeypatch.setattr(page_write, "PAGES_DIR", pages_dir)
    monkeypatch.setattr(page_write, "SYSTEM_DIR", system_dir)

    result = page_write.apply_page_writes(
        [page_write.prepare_page_write(target, "duplicate\n")]
    )

    assert result["status"] == "retry"
    assert "already exists" in result["reason"]
    assert existing.read_text(encoding="utf-8") == "existing\n"
    assert not target.exists()
