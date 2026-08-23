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
