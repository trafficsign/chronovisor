from __future__ import annotations

from pathlib import Path

import pytest

from chronovisor.recall import recall_distillation_store as store


def test_append_reuses_matching_checkpoint_without_full_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "ledger.jsonl"
    store.append_chain(path, {"index": 0})
    verify = store.verify_chain
    read_bytes = Path.read_bytes

    def unexpected_verify(_path: Path) -> dict[str, object]:
        raise AssertionError("steady-state append must not scan the ledger")

    def reject_ledger_read(self: Path) -> bytes:
        if self == path:
            raise AssertionError("steady-state append must not read the ledger")
        return read_bytes(self)

    monkeypatch.setattr(store, "verify_chain", unexpected_verify)
    monkeypatch.setattr(Path, "read_bytes", reject_ledger_read)
    store.append_chain_batch(path, ({"index": 1}, {"index": 2}))
    monkeypatch.setattr(store, "verify_chain", verify)
    monkeypatch.setattr(Path, "read_bytes", read_bytes)

    assert [row["index"] for row in store.read_chain(path)] == [0, 1, 2]


def test_append_recovers_an_interrupted_final_record(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    store.append_chain(path, {"index": 0})
    with path.open("ab") as handle:
        handle.write(b'{"index":')
        handle.flush()

    store.append_chain(path, {"index": 1})

    assert [row["index"] for row in store.read_chain(path)] == [0, 1]


def test_chain_head_reuses_matching_checkpoint_without_ledger_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "ledger.jsonl"
    row = store.append_chain(path, {"index": 0})
    read_bytes = Path.read_bytes

    def unexpected_verify(_path: Path) -> dict[str, object]:
        raise AssertionError("matching checkpoint must not scan the ledger")

    def reject_ledger_read(self: Path) -> bytes:
        if self == path:
            raise AssertionError("matching checkpoint must not read the ledger")
        return read_bytes(self)

    monkeypatch.setattr(store, "verify_chain", unexpected_verify)
    monkeypatch.setattr(Path, "read_bytes", reject_ledger_read)
    assert store.chain_head(path) == {
        "records": 1,
        "head_sha256": row["record_sha256"],
    }


def test_chain_head_recovers_checkpoint_mismatch_and_torn_tail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "ledger.jsonl"
    row = store.append_chain(path, {"index": 0})
    recover = store._recover_chain_tail
    calls = 0

    def tracked_recovery(recovery_path: Path) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return recover(recovery_path)

    with path.open("ab") as handle:
        handle.write(b'{"index":')
        handle.flush()
    monkeypatch.setattr(store, "_recover_chain_tail", tracked_recovery)

    assert store.chain_head(path) == {
        "records": 1,
        "head_sha256": row["record_sha256"],
    }
    assert calls == 1
    assert store.verify_chain(path) == {
        "records": 1,
        "head_sha256": row["record_sha256"],
    }


def test_chain_head_recovers_an_invalid_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "ledger.jsonl"
    row = store.append_chain(path, {"index": 0})
    store._chain_checkpoint_path(path).write_text("{}", encoding="utf-8")
    recover = store._recover_chain_tail
    calls = 0

    def tracked_recovery(recovery_path: Path) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return recover(recovery_path)

    monkeypatch.setattr(store, "_recover_chain_tail", tracked_recovery)

    assert store.chain_head(path) == {
        "records": 1,
        "head_sha256": row["record_sha256"],
    }
    assert calls == 1
