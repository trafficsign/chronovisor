from __future__ import annotations

import os
from pathlib import Path

import pytest

from chronovisor.recall import recall_distillation_store as store


def test_pinned_immutable_write_never_follows_a_directory_swap(tmp_path: Path) -> None:
    directory, external, displaced = (
        tmp_path / "trusted",
        tmp_path / "external",
        tmp_path / "displaced",
    )
    directory.mkdir()
    external.mkdir()
    external_json = external / "unchanged.json"
    external_lock = external / ".immutable.lock"
    external_json.write_text("external-json")
    external_lock.write_text("external-lock")

    def swap() -> None:
        os.rename(directory, displaced)
        directory.symlink_to(external, target_is_directory=True)

    artifact_id, _, artifact = store.write_immutable_pinned(
        directory,
        {"kind": "pinned-race"},
        schema="test.r4.v1",
        before_persist=swap,
    )

    assert (
        store.read_immutable_pinned(displaced, artifact_id, schema="test.r4.v1")
        == artifact
    )
    assert external_json.read_text() == "external-json"
    assert external_lock.read_text() == "external-lock"
    assert {path.name for path in external.iterdir()} == {
        "unchanged.json",
        ".immutable.lock",
    }


def test_pinned_immutable_write_rejects_a_preopen_real_directory_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory, replacement, displaced = (
        tmp_path / "trusted",
        tmp_path / "replacement",
        tmp_path / "displaced",
    )
    directory.mkdir()
    replacement.mkdir()
    (replacement / "sentinel").write_text("must-survive")
    real_open = store._open_directory_nofollow

    def swap_before_open(
        path: Path,
        *,
        create: bool,
        snapshot: tuple[Path, tuple[tuple[int, int] | None, ...]],
    ) -> int:
        os.rename(directory, displaced)
        os.rename(replacement, directory)
        return real_open(path, create=create, snapshot=snapshot)

    monkeypatch.setattr(store, "_open_directory_nofollow", swap_before_open)

    with pytest.raises(store.DistillationStoreError, match="changed"):
        store.write_immutable_pinned(
            directory, {"kind": "preopen-write"}, schema="test.r4.v1"
        )

    assert list(displaced.iterdir()) == []
    assert {path.name for path in directory.iterdir()} == {"sentinel"}


def test_pinned_immutable_unlink_never_deletes_a_swapped_external_file(
    tmp_path: Path,
) -> None:
    directory, external, displaced = (
        tmp_path / "trusted",
        tmp_path / "external",
        tmp_path / "displaced",
    )
    directory.mkdir()
    external.mkdir()
    artifact_id, _, artifact = store.write_immutable_pinned(
        directory, {"kind": "pinned-unlink"}, schema="test.r4.v1"
    )
    external_file = external / f"{artifact_id}.json"
    external_file.write_text("must-survive")

    def swap() -> None:
        os.rename(directory, displaced)
        directory.symlink_to(external, target_is_directory=True)

    store.unlink_immutable_pinned(
        directory,
        artifact_id,
        expected=artifact,
        schema="test.r4.v1",
        before_unlink=swap,
    )

    assert not (displaced / f"{artifact_id}.json").exists()
    assert external_file.read_text() == "must-survive"


def test_pinned_immutable_unlink_rejects_a_preopen_real_directory_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory, replacement, displaced = (
        tmp_path / "trusted",
        tmp_path / "replacement",
        tmp_path / "displaced",
    )
    artifact_id, _, artifact = store.write_immutable_pinned(
        directory, {"kind": "preopen-unlink"}, schema="test.r4.v1"
    )
    store.write_immutable_pinned(
        replacement,
        {"kind": "preopen-unlink"},
        schema="test.r4.v1",
        artifact_id=artifact_id,
    )
    real_open = store._open_directory_nofollow

    def swap_before_open(
        path: Path,
        *,
        create: bool,
        snapshot: tuple[Path, tuple[tuple[int, int] | None, ...]],
    ) -> int:
        os.rename(directory, displaced)
        os.rename(replacement, directory)
        return real_open(path, create=create, snapshot=snapshot)

    monkeypatch.setattr(store, "_open_directory_nofollow", swap_before_open)

    with pytest.raises(store.DistillationStoreError, match="changed"):
        store.unlink_immutable_pinned(
            directory,
            artifact_id,
            expected=artifact,
            schema="test.r4.v1",
        )

    assert (displaced / f"{artifact_id}.json").exists()
    assert (directory / f"{artifact_id}.json").exists()


def test_pinned_immutable_read_rejects_a_preopen_real_directory_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory, replacement, displaced = (
        tmp_path / "trusted",
        tmp_path / "replacement",
        tmp_path / "displaced",
    )
    artifact_id, _, _ = store.write_immutable_pinned(
        directory, {"kind": "preopen-read"}, schema="test.r4.v1"
    )
    store.write_immutable_pinned(
        replacement,
        {"kind": "preopen-read"},
        schema="test.r4.v1",
        artifact_id=artifact_id,
    )
    real_open = store._open_directory_nofollow

    def swap_before_open(
        path: Path,
        *,
        create: bool,
        snapshot: tuple[Path, tuple[tuple[int, int] | None, ...]],
    ) -> int:
        directory.rename(displaced)
        replacement.rename(directory)
        return real_open(path, create=create, snapshot=snapshot)

    monkeypatch.setattr(store, "_open_directory_nofollow", swap_before_open)

    with pytest.raises(store.DistillationStoreError, match="changed"):
        store.read_immutable_pinned(directory, artifact_id, schema="test.r4.v1")

    assert (displaced / f"{artifact_id}.json").exists()
    assert (directory / f"{artifact_id}.json").exists()


def test_pinned_immutable_write_rejects_a_preopen_parent_swap_before_create(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent, replacement, displaced = (
        tmp_path / "trusted-parent",
        tmp_path / "replacement-parent",
        tmp_path / "displaced-parent",
    )
    parent.mkdir()
    replacement.mkdir()
    (replacement / "sentinel").write_text("must-survive")
    directory = parent / "missing"
    real_open = store._open_directory_nofollow

    def swap_before_open(
        path: Path,
        *,
        create: bool,
        snapshot: tuple[Path, tuple[tuple[int, int] | None, ...]],
    ) -> int:
        parent.rename(displaced)
        replacement.rename(parent)
        return real_open(path, create=create, snapshot=snapshot)

    monkeypatch.setattr(store, "_open_directory_nofollow", swap_before_open)

    with pytest.raises(store.DistillationStoreError, match="changed"):
        store.write_immutable_pinned(
            directory, {"kind": "preopen-create"}, schema="test.r4.v1"
        )

    assert list(displaced.iterdir()) == []
    assert {path.name for path in parent.iterdir()} == {"sentinel"}


def test_pinned_immutable_write_rejects_an_intercall_directory_replacement(
    tmp_path: Path,
) -> None:
    directory, replacement, displaced = (
        tmp_path / "trusted",
        tmp_path / "replacement",
        tmp_path / "displaced",
    )
    directory.mkdir()
    replacement.mkdir()
    (replacement / "sentinel").write_text("must-survive")
    identity = store.pinned_directory_identity(directory, create=False)
    directory.rename(displaced)
    replacement.rename(directory)

    with pytest.raises(store.DistillationStoreError, match="changed"):
        store.write_immutable_pinned(
            directory,
            {"kind": "intercall-write"},
            schema="test.r4.v1",
            expected_directory_identity=identity,
        )

    assert list(displaced.iterdir()) == []
    assert {path.name for path in directory.iterdir()} == {"sentinel"}


def test_pinned_immutable_write_rejects_an_unsafe_existing_artifact(tmp_path: Path) -> None:
    directory, external = tmp_path / "trusted", tmp_path / "external"
    directory.mkdir()
    external.write_text("must-survive")
    artifact_id, _, _ = store.write_immutable_pinned(
        tmp_path / "seed", {"kind": "unsafe-existing"}, schema="test.r4.v1"
    )
    (directory / f"{artifact_id}.json").symlink_to(external)

    with pytest.raises(store.DistillationStoreError, match="unreadable"):
        store.write_immutable_pinned(
            directory,
            {"kind": "unsafe-existing"},
            schema="test.r4.v1",
            artifact_id=artifact_id,
        )

    assert external.read_text() == "must-survive"


def test_pinned_immutable_write_completes_short_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_write = store.os.write

    def short_write(descriptor: int, data: bytes) -> int:
        return real_write(descriptor, data[:1])

    monkeypatch.setattr(store.os, "write", short_write)
    artifact_id, _, artifact = store.write_immutable_pinned(
        tmp_path / "trusted", {"kind": "short-write"}, schema="test.r4.v1"
    )

    assert store.read_immutable_pinned(
        tmp_path / "trusted", artifact_id, schema="test.r4.v1"
    ) == artifact


def test_pinned_immutable_write_runs_after_persist_while_artifact_exists(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "trusted"
    observed: list[bool] = []

    artifact_id, _, artifact = store.write_immutable_pinned(
        directory,
        {"kind": "after-persist"},
        schema="test.r4.v1",
        after_persist=lambda: observed.append(
            any(directory.glob("*.json"))
        ),
    )

    assert observed == [True]
    assert store.read_immutable_pinned(directory, artifact_id, schema="test.r4.v1") == artifact


def test_pinned_immutable_write_cleans_new_artifact_through_displaced_fd(
    tmp_path: Path,
) -> None:
    directory, displaced = tmp_path / "trusted", tmp_path / "displaced"
    directory.mkdir()

    def rename_then_fail() -> None:
        directory.rename(displaced)
        raise RuntimeError("post-persist failure")

    with pytest.raises(RuntimeError, match="post-persist failure"):
        store.write_immutable_pinned(
            directory,
            {"kind": "displaced-cleanup"},
            schema="test.r4.v1",
            after_persist=rename_then_fail,
        )

    assert not list(displaced.glob("*.json"))


def test_pinned_immutable_write_retains_preexisting_artifact_on_after_persist_failure(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "trusted"
    artifact_id, _, artifact = store.write_immutable_pinned(
        directory, {"kind": "preexisting"}, schema="test.r4.v1"
    )

    with pytest.raises(RuntimeError, match="post-persist failure"):
        store.write_immutable_pinned(
            directory,
            {"kind": "preexisting"},
            schema="test.r4.v1",
            after_persist=lambda: (_ for _ in ()).throw(
                RuntimeError("post-persist failure")
            ),
        )

    assert store.read_immutable_pinned(directory, artifact_id, schema="test.r4.v1") == artifact
