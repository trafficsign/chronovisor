from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from stat import S_IMODE
from typing import Any, cast

import pytest

from chronovisor.core.canonical_json import (
    canonical_json_bytes_strict,
    canonical_json_sha256_strict,
)
from chronovisor.recall import recall_distillation_store as store


def test_read_sealed_wraps_json_value_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "state.json"
    path.write_bytes(b"{}")

    def invalid_integer(_payload: object) -> object:
        raise ValueError("integer string conversion limit exceeded")

    monkeypatch.setattr(store.json, "loads", invalid_integer)
    with pytest.raises(store.DistillationStoreError, match="cannot read sealed"):
        store.read_sealed(path)


@pytest.mark.parametrize(
    "artifact_id",
    [
        "../" + "a" * 64,
        "/tmp/" + "b" * 64,
        "nested/" + "c" * 64,
        "nested\\" + "d" * 64,
        "g" * 63,
        "G" * 64,
    ],
)
def test_write_immutable_rejects_path_traversal_before_write(
    tmp_path: Path, artifact_id: str
) -> None:
    directory = tmp_path / "artifacts"

    with pytest.raises(store.DistillationStoreError, match="invalid artifact id"):
        store.write_immutable(
            directory,
            {"kind": "test"},
            schema="chronovisor.test.v1",
            artifact_id=artifact_id,
        )

    assert not directory.exists()
    assert not (tmp_path.parent / f"{artifact_id}.json").exists()


def test_write_immutable_accepts_only_explicit_digest_basename(tmp_path: Path) -> None:
    artifact_id = "a" * 64
    identity, path, _ = store.write_immutable(
        tmp_path / "artifacts",
        {"kind": "test"},
        schema="chronovisor.test.v1",
        artifact_id=artifact_id,
    )

    assert identity == artifact_id
    assert path == tmp_path / "artifacts" / f"{artifact_id}.json"


@pytest.mark.parametrize("field", ["schema", "namespace"])
@pytest.mark.parametrize(
    "checker", [store.verify_chain, store._recover_chain_tail, store.read_chain]
)
def test_chain_rejects_forged_schema_and_namespace(
    tmp_path: Path, field: str, checker: Callable[[Path], object]
) -> None:
    path = tmp_path / f"{field}.jsonl"
    row = store.append_chain(path, {"index": 0})
    forged = dict(row)
    forged[field] = "forged"
    unsigned = {key: value for key, value in forged.items() if key != "record_sha256"}
    forged["record_sha256"] = canonical_json_sha256_strict(unsigned)
    path.write_bytes(canonical_json_bytes_strict(forged) + b"\n")

    with pytest.raises(store.DistillationStoreError, match=f"{field} mismatch"):
        checker(path)


def test_batch_rejects_an_infinite_iterable_after_501_items(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    consumed = 0

    def payloads() -> Iterator[dict[str, int]]:
        nonlocal consumed
        while True:
            consumed += 1
            yield {"index": consumed}

    with pytest.raises(store.DistillationStoreError, match="batch is too large"):
        store.append_chain_batch(path, payloads())

    assert consumed == 501
    assert not path.exists()


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


def test_read_chain_uses_checkpoint_without_rehashing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "ledger.jsonl"
    store.append_chain_batch(path, ({"index": 0}, {"index": 1}))

    def unexpected_recovery(_path: Path) -> dict[str, object]:
        raise AssertionError("matching checkpoint must not rehash the ledger")

    monkeypatch.setattr(store, "_recover_chain_tail", unexpected_recovery)

    assert [row["index"] for row in store.read_chain(path)] == [0, 1]


def test_sealed_checkpoint_rejects_non_digest_head(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    row = store.append_chain(path, {"index": 0})
    store.write_sealed_state(
        store._chain_checkpoint_path(path),
        {
            "kind": "ledger-chain-checkpoint",
            "ledger_name": path.name,
            "records": 1,
            "head_sha256": "not-a-digest",
            "file_state": store._ledger_file_state(path),
        },
    )

    assert store.chain_head(path) == {
        "records": 1,
        "head_sha256": row["record_sha256"],
    }


def _historical_atom() -> dict[str, object]:
    return {
        "atom_id": "atom-1",
        "host": "codex",
        "session_cluster_id": "session-1",
        "source_index": 0,
        "timestamp_us": 1,
        "text_sha256": "a" * 64,
        "ref": {"raw_id": "raw-1", "line": 0},
        "text": "hello world",
    }


def test_historical_index_temp_and_final_db_are_private(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "historical.sqlite"
    observed_modes: list[int] = []
    connect = sqlite3.connect

    def tracking_connect(
        database: Any, *args: Any, **kwargs: Any
    ) -> sqlite3.Connection:
        database_path = Path(database)
        if database_path != path and database_path.name.endswith(".tmp"):
            observed_modes.append(S_IMODE(database_path.stat().st_mode))
        return cast(sqlite3.Connection, connect(database, *args, **kwargs))

    monkeypatch.setattr(sqlite3, "connect", tracking_connect)
    store.create_historical_index(path, [_historical_atom()])

    assert observed_modes == [0o600]
    assert S_IMODE(path.stat().st_mode) == 0o600


def test_historical_index_serializes_concurrent_builders(tmp_path: Path) -> None:
    path = tmp_path / "historical.sqlite"
    atoms = [_historical_atom()]

    with ThreadPoolExecutor(max_workers=4) as executor:
        digests = list(
            executor.map(
                lambda _index: store.create_historical_index(path, atoms), range(4)
            )
        )

    assert len(set(digests)) == 1
    assert S_IMODE(path.stat().st_mode) == 0o600


def _unique_payload(identity: str, binding: str) -> dict[str, str]:
    return {"decision_id": identity, "idempotency_sha256": binding}


def test_unique_append_uses_index_without_full_ledger_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "receipts.jsonl"
    first = store.append_chain_unique(
        path,
        _unique_payload("decision-1", "binding-1"),
        unique_field="decision_id",
        binding_field="idempotency_sha256",
    )
    read_bytes = Path.read_bytes

    def reject_ledger_read(self: Path) -> bytes:
        if self == path:
            raise AssertionError("steady-state unique append must not read the ledger")
        return read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", reject_ledger_read)
    monkeypatch.setattr(
        store,
        "_recover_chain_tail",
        lambda _path: (_ for _ in ()).throw(AssertionError("must not rebuild")),
    )

    assert (
        store.append_chain_unique(
            path,
            _unique_payload("decision-1", "binding-1"),
            unique_field="decision_id",
            binding_field="idempotency_sha256",
        )
        == first
    )
    store.append_chain_unique(
        path,
        _unique_payload("decision-2", "binding-2"),
        unique_field="decision_id",
        binding_field="idempotency_sha256",
    )


def test_unique_index_preserves_duplicate_and_conflict_behavior(tmp_path: Path) -> None:
    path = tmp_path / "receipts.jsonl"
    first = store.append_chain_unique(
        path,
        _unique_payload("decision-1", "binding-1"),
        unique_field="decision_id",
        binding_field="idempotency_sha256",
    )

    assert (
        store.append_chain_unique(
            path,
            _unique_payload("decision-1", "binding-1"),
            unique_field="decision_id",
            binding_field="idempotency_sha256",
        )
        == first
    )
    with pytest.raises(store.DistillationStoreError, match="identity conflict"):
        store.append_chain_unique(
            path,
            _unique_payload("decision-1", "binding-2"),
            unique_field="decision_id",
            binding_field="idempotency_sha256",
        )


@pytest.mark.parametrize(
    "failure",
    [
        "legacy",
        "missing",
        "corrupt",
        "torn",
        "separator_only",
        "valid_no_newline",
        "batch_stale",
    ],
)
def test_unique_index_recovers_and_rebuilds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    path = tmp_path / "receipts.jsonl"
    if failure == "legacy":
        store.append_chain(path, {"legacy": True})
    else:
        store.append_chain_unique(
            path,
            _unique_payload("decision-1", "binding-1"),
            unique_field="decision_id",
            binding_field="idempotency_sha256",
        )
    if failure == "missing":
        store._unique_index_path(path).unlink()
    elif failure == "corrupt":
        store._unique_index_path(path).write_bytes(b"not sqlite")
    elif failure == "torn":
        with path.open("ab") as handle:
            handle.write(b'{"broken":')
            handle.flush()
    elif failure == "separator_only":
        with path.open("ab") as handle:
            handle.write(b"\n")
            handle.flush()
    elif failure == "valid_no_newline":
        path.write_bytes(path.read_bytes().removesuffix(b"\n"))
    elif failure == "batch_stale":
        store.append_chain_batch(path, [{"index": 2}])
    recover = store._recover_chain_tail
    calls = 0

    def tracked_recovery(recovery_path: Path) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return recover(recovery_path)

    monkeypatch.setattr(store, "_recover_chain_tail", tracked_recovery)
    store.append_chain_unique(
        path,
        _unique_payload("decision-2", "binding-2"),
        unique_field="decision_id",
        binding_field="idempotency_sha256",
    )

    assert calls == 1
    assert store.verify_chain(path)["records"] == (3 if failure == "batch_stale" else 2)


def test_unique_index_rebuilds_out_of_bounds_offset(tmp_path: Path) -> None:
    path = tmp_path / "receipts.jsonl"
    first = store.append_chain_unique(
        path,
        _unique_payload("decision-1", "binding-1"),
        unique_field="decision_id",
        binding_field="idempotency_sha256",
    )
    with sqlite3.connect(store._unique_index_path(path)) as connection:
        connection.execute("PRAGMA foreign_keys = OFF")
        connection.execute("UPDATE entries SET offset = ?", (2**63 - 1,))
        connection.execute("UPDATE rows SET offset = ?", (2**63 - 1,))

    assert (
        store.append_chain_unique(
            path,
            _unique_payload("decision-1", "binding-1"),
            unique_field="decision_id",
            binding_field="idempotency_sha256",
        )
        == first
    )


def test_unique_index_rebuilds_after_logical_row_deletion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "receipts.jsonl"
    first = store.append_chain_unique(
        path,
        _unique_payload("decision-1", "binding-1"),
        unique_field="decision_id",
        binding_field="idempotency_sha256",
    )
    with sqlite3.connect(store._unique_index_path(path)) as connection:
        connection.execute("DELETE FROM entries")
    recover = store._recover_chain_tail
    calls = 0

    def tracked_recovery(recovery_path: Path) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return recover(recovery_path)

    monkeypatch.setattr(store, "_recover_chain_tail", tracked_recovery)

    assert (
        store.append_chain_unique(
            path,
            _unique_payload("decision-1", "binding-1"),
            unique_field="decision_id",
            binding_field="idempotency_sha256",
        )
        == first
    )
    assert calls == 1
    assert store.verify_chain(path)["records"] == 1


def test_snapshot_uses_label_projection_without_reading_ledger_body(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = store.distillation_dir(tmp_path)
    label_path = root / store.LABEL_LEDGER_FILE
    store.append_chain_batch(
        label_path,
        [
            {"authority": "teacher-only", "assignment": {"probe": True}},
            {"authority": "verified"},
        ],
    )
    store.write_sealed_state(
        root / store.STATE_FILE,
        {"status": "ready"},
    )
    read_bytes = Path.read_bytes

    def reject_label_body(self: Path) -> bytes:
        if self == label_path:
            raise AssertionError("projected snapshot must not decode labels")
        return read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", reject_label_body)
    result = store.snapshot(tmp_path)

    assert (result["teacher_only"], result["verified_truth"], result["probe_not_truth"]) == (
        1,
        1,
        1,
    )


def test_snapshot_ignores_forged_baseline_counts(tmp_path: Path) -> None:
    root = store.distillation_dir(tmp_path)
    label_path = root / store.LABEL_LEDGER_FILE
    store.append_chain(label_path, {"authority": "teacher-only"})
    label_head = store.chain_head(label_path)
    baseline_id, _, _ = store.write_immutable(
        root / "baselines",
        {
            "kind": "baseline",
            "label_chain_head": label_head["head_sha256"],
            "label_records": label_head["records"],
            "counts": {
                "teacher_only_labels": 7,
                "verified_truth_labels": 3,
                "locked_test_probe_pairs": 2,
            },
        },
        schema="chronovisor.recall-distill-baseline.v1",
    )
    store.write_sealed_state(
        root / store.STATE_FILE,
        {"status": "ready", "baseline_artifact_id": baseline_id},
    )

    result = store.snapshot(tmp_path)

    assert (result["teacher_only"], result["verified_truth"], result["probe_not_truth"]) == (
        1,
        0,
        0,
    )


def test_label_projection_recovers_missing_and_interrupted_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = store.distillation_dir(tmp_path) / store.LABEL_LEDGER_FILE
    store.append_chain(path, {"authority": "teacher-only"})
    store._label_health_projection_path(path).unlink()

    projection = store.label_health_projection(path, repair=True)

    assert projection["counts"] == {
        "teacher_only": 1,
        "verified_truth": 0,
        "probe_not_truth": 0,
    }
    assert store._label_health_projection_path(path).exists()

    def interrupted(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise store.DistillationStoreError("interrupted projection")

    monkeypatch.setattr(store, "_write_label_health_projection", interrupted)
    store.append_chain(path, {"authority": "verified"})

    assert store.label_health_projection(path)["counts"] == {
        "teacher_only": 1,
        "verified_truth": 1,
        "probe_not_truth": 0,
    }
    monkeypatch.undo()
    assert store.label_health_projection(path, repair=True)["counts"] == {
        "teacher_only": 1,
        "verified_truth": 1,
        "probe_not_truth": 0,
    }
