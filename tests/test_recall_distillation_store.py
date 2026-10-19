from __future__ import annotations

import os
import threading
from pathlib import Path

import pytest

from chronovisor.recall import recall_distillation_store as store


def _unique_payload(identity: str, binding: str) -> dict[str, str]:
    return {"decision_id": identity, "idempotency_sha256": binding}


def test_unique_append_retries_when_preparation_races_with_append(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A preparation CAS miss is part of the public append retry boundary."""

    path = tmp_path / "receipts.jsonl"
    store.append_chain(path, _unique_payload("seed", "seed-binding"))
    original_read = store._read_pinned_ledger
    calls = 0
    calls_lock = threading.Lock()
    competing_done = threading.Event()
    competing_errors: list[BaseException] = []
    competing_rows: list[dict[str, object]] = []
    competing_thread: threading.Thread | None = None

    def compete() -> None:
        try:
            competing_rows.append(
                store.append_chain_unique(
                    path,
                    _unique_payload("competing", "competing-binding"),
                    unique_field="decision_id",
                    binding_field="idempotency_sha256",
                )
            )
        except BaseException as exc:  # pragma: no cover - asserted below
            competing_errors.append(exc)
        finally:
            competing_done.set()

    def read_then_append(descriptor: int, source: Path, state: object, **kwargs: object) -> bytes:
        nonlocal calls, competing_thread
        with calls_lock:
            calls += 1
            trigger = calls == 2
        data = original_read(descriptor, source, state, **kwargs)
        if trigger:
            competing_thread = threading.Thread(target=compete)
            competing_thread.start()
            assert competing_done.wait(10), "competing append did not finish"
            competing_thread.join(10)
            assert not competing_thread.is_alive(), "competing append is still running"
            if competing_errors:
                raise competing_errors[0]
        return data

    monkeypatch.setattr(store, "_read_pinned_ledger", read_then_append)
    try:
        appended = store.append_chain_unique(
            path,
            _unique_payload("main", "main-binding"),
            unique_field="decision_id",
            binding_field="idempotency_sha256",
        )
    finally:
        if competing_thread is not None:
            competing_thread.join(10)

    assert appended["decision_id"] == "main"
    assert len(competing_rows) == 1
    assert (
        store.append_chain_unique(
            path,
            _unique_payload("competing", "competing-binding"),
            unique_field="decision_id",
            binding_field="idempotency_sha256",
        )
        == competing_rows[0]
    )
    assert calls >= 3
    assert store.verify_chain(path)["records"] == 3
    assert {
        row["decision_id"] for row in store.read_chain(path)
    } == {"seed", "competing", "main"}


def test_unique_append_retries_three_times_for_persistent_preparation_busy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "receipts.jsonl"
    store.append_chain(path, _unique_payload("seed", "seed-binding"))
    calls = 0

    def always_busy(*_args: object, **_kwargs: object) -> None:
        nonlocal calls
        calls += 1
        raise store.DistillationStoreBusy("synthetic preparation race")

    monkeypatch.setattr(store, "prepare_unique_index", always_busy)
    with pytest.raises(store.DistillationStoreBusy, match="synthetic preparation race"):
        store.append_chain_unique(
            path,
            _unique_payload("main", "main-binding"),
            unique_field="decision_id",
            binding_field="idempotency_sha256",
        )
    assert calls == 3


def test_unique_append_does_not_retry_nonbusy_preparation_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "receipts.jsonl"
    store.append_chain(path, _unique_payload("seed", "seed-binding"))
    prepare_calls = 0

    def invalid_prepare(*_args: object, **_kwargs: object) -> None:
        nonlocal prepare_calls
        prepare_calls += 1
        raise store.DistillationStoreError("synthetic invalid identity")

    monkeypatch.setattr(store, "prepare_unique_index", invalid_prepare)
    with pytest.raises(store.DistillationStoreError, match="synthetic invalid identity"):
        store.append_chain_unique(
            path,
            _unique_payload("main", "main-binding"),
            unique_field="decision_id",
            binding_field="idempotency_sha256",
        )
    assert prepare_calls == 1


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


def test_chain_checkpoint_and_cursor_survive_device_renumbering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chronovisor.recall import recall_file_identity as file_identity

    path = tmp_path / "rows.jsonl"
    store.append_chain(path, {"kind": "fixture", "value": 1})
    _, cursor, _ = store.read_chain_tail(path)
    checkpoint_path = store._chain_checkpoint_path(path)
    checkpoint = store.read_sealed(checkpoint_path)
    checkpoint["file_state"]["st_dev"] += 2
    cursor["file_state"]["st_dev"] += 2
    store.write_sealed_state(checkpoint_path, {k: v for k, v in checkpoint.items() if k not in {"schema", "namespace", "seal_sha256"}})
    original = checkpoint_path.read_bytes()
    monkeypatch.setattr(file_identity.sys, "platform", "darwin")
    monkeypatch.setattr(file_identity, "birthtime_ns", lambda _value: 0)
    rows, current, reset = store.read_chain_tail(path, cursor)
    assert rows == [] and reset is False
    assert current["head_sha256"] == cursor["head_sha256"]
    assert checkpoint_path.read_bytes() == original
    # A subsequent append reuses the verified prefix through the normal writer.
    store.append_chain(path, {"kind": "fixture", "value": 2})
    rows, _, reset = store.read_chain_tail(path, cursor)
    assert len(rows) == 1 and rows[0]["value"] == 2 and reset is False


def test_rebound_cursor_still_rejects_truncation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chronovisor.recall import recall_file_identity as file_identity

    path = tmp_path / "rows.jsonl"
    store.append_chain(path, {"kind": "fixture", "value": 1})
    _, cursor, _ = store.read_chain_tail(path)
    cursor["file_state"]["st_dev"] += 2
    monkeypatch.setattr(file_identity.sys, "platform", "darwin")
    monkeypatch.setattr(file_identity, "birthtime_ns", lambda _value: 0)
    with path.open("r+b") as handle:
        handle.truncate(0)
    with pytest.raises(store.DistillationStoreError, match="truncated"):
        store.read_chain_tail(path, cursor)
