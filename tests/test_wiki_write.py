from __future__ import annotations

from pathlib import Path

from llm_wiki_mcp import wiki_write


def test_stale_preimage_never_overwrites_foreign_change(tmp_path: Path) -> None:
    path = tmp_path / "page.md"
    path.write_text("original\n", encoding="utf-8")
    plan = wiki_write.prepare_wiki_write(path, "generated\n")
    path.write_text("foreign\n", encoding="utf-8")

    result = wiki_write.apply_wiki_writes([plan])

    assert result["status"] == "retry"
    assert "changed before apply" in result["reason"]
    assert path.read_text(encoding="utf-8") == "foreign\n"


def test_concurrent_identical_output_is_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "page.md"
    path.write_text("original\n", encoding="utf-8")
    plan = wiki_write.prepare_wiki_write(path, "generated\n")
    path.write_text("generated\n", encoding="utf-8")

    result = wiki_write.apply_wiki_writes([plan])

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
        wiki_write.prepare_wiki_write(first, "first-new\n"),
        wiki_write.prepare_wiki_write(second, "second-new\n"),
    ]
    real_atomic_write = wiki_write.atomic_write
    calls = 0

    def fail_second(path: Path, content: str) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("disk full")
        real_atomic_write(path, content)

    monkeypatch.setattr(wiki_write, "atomic_write", fail_second)

    result = wiki_write.apply_wiki_writes(plans)

    assert result["status"] == "retry"
    assert result["rolled_back"][str(first)] is True
    assert first.read_text(encoding="utf-8") == "first-old\n"
    assert second.read_text(encoding="utf-8") == "second-old\n"


def test_batch_failure_removes_an_owned_create(tmp_path: Path, monkeypatch) -> None:
    created = tmp_path / "created.md"
    second = tmp_path / "second.md"
    second.write_text("second-old\n", encoding="utf-8")
    plans = [
        wiki_write.prepare_wiki_write(created, "created-new\n"),
        wiki_write.prepare_wiki_write(second, "second-new\n"),
    ]
    real_atomic_write = wiki_write.atomic_write
    calls = 0

    def fail_second(path: Path, content: str) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("disk full")
        real_atomic_write(path, content)

    monkeypatch.setattr(wiki_write, "atomic_write", fail_second)

    result = wiki_write.apply_wiki_writes(plans)

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
    plan = wiki_write.prepare_wiki_write(path, "new\n")
    real_atomic_write = wiki_write.atomic_write
    raised = False

    def replace_then_fail(target: Path, content: str) -> None:
        nonlocal raised
        real_atomic_write(target, content)
        if not raised:
            raised = True
            raise OSError("late failure")

    monkeypatch.setattr(wiki_write, "atomic_write", replace_then_fail)

    result = wiki_write.apply_wiki_writes([plan])

    assert result["status"] == "retry"
    assert result["rolled_back"][str(path)] is True
    assert path.read_text(encoding="utf-8") == "old\n"


def test_rollback_does_not_clobber_foreign_bytes(tmp_path: Path, monkeypatch) -> None:
    first = tmp_path / "first.md"
    second = tmp_path / "second.md"
    first.write_text("first-old\n", encoding="utf-8")
    second.write_text("second-old\n", encoding="utf-8")
    plans = [
        wiki_write.prepare_wiki_write(first, "first-new\n"),
        wiki_write.prepare_wiki_write(second, "second-new\n"),
    ]
    real_atomic_write = wiki_write.atomic_write
    calls = 0

    def foreign_then_fail(path: Path, content: str) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            real_atomic_write(path, content)
            path.write_text("foreign\n", encoding="utf-8")
            return
        raise OSError("second failed")

    monkeypatch.setattr(wiki_write, "atomic_write", foreign_then_fail)

    result = wiki_write.apply_wiki_writes(plans)

    assert result["status"] == "retry"
    assert result["rolled_back"][str(first)] is False
    assert first.read_text(encoding="utf-8") == "foreign\n"
    assert second.read_text(encoding="utf-8") == "second-old\n"
