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

    with path.open("ab") as handle:
        handle.write(b'{"index":')
        handle.flush()
    decode_calls = _install_decode_lock_probe(path, monkeypatch)

    assert store.chain_head(path) == {
        "records": 1,
        "head_sha256": row["record_sha256"],
    }
    assert store.verify_chain(path) == {
        "records": 1,
        "head_sha256": row["record_sha256"],
    }
    assert decode_calls() > 0


def test_chain_head_recovers_an_invalid_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "ledger.jsonl"
    row = store.append_chain(path, {"index": 0})
    store._chain_checkpoint_path(path).write_text("{}", encoding="utf-8")
    decode_calls = _install_decode_lock_probe(path, monkeypatch)

    assert store.chain_head(path) == {
        "records": 1,
        "head_sha256": row["record_sha256"],
    }
    assert store.verify_chain(path) == {
        "records": 1,
        "head_sha256": row["record_sha256"],
    }
    assert decode_calls() > 0


def test_append_label_batch_repairs_missing_projection_without_locked_decode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / store.LABEL_LEDGER_FILE
    store.append_chain(path, {"authority": "teacher-only"})
    store._label_health_projection_path(path).unlink()
    decode_calls = _install_decode_lock_probe(path, monkeypatch)

    appended = store.append_chain(path, {"authority": "verified"})

    assert appended["authority"] == "verified"
    assert store.verify_chain(path)["records"] == 2
    assert store.label_health_projection(path)["counts"] == {
        "teacher_only": 1,
        "verified_truth": 1,
        "probe_not_truth": 0,
    }
    assert decode_calls() > 0


def test_append_label_batch_fails_closed_on_source_rewrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / store.LABEL_LEDGER_FILE
    store.append_chain(path, {"authority": "teacher-only"})
    store._label_health_projection_path(path).unlink()
    rewritten = path.read_bytes().replace(b'"teacher-only"', b'"teacher_only"', 1)
    original_decode = store._decode_chain_bytes
    mutated = False

    def rewrite_once(data: bytes, **kwargs: object) -> object:
        nonlocal mutated
        if not mutated:
            path.write_bytes(rewritten)
            mutated = True
        return original_decode(data, **kwargs)

    monkeypatch.setattr(store, "_decode_chain_bytes", rewrite_once)
    with pytest.raises(store.DistillationStoreBusy, match="changed during snapshot"):
        store.append_chain(path, {"authority": "verified"})

    assert mutated
    assert b'"verified"' not in path.read_bytes()


def test_read_chain_uses_checkpoint_without_rehashing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "ledger.jsonl"
    store.append_chain_batch(path, ({"index": 0}, {"index": 1}))

    def unexpected_recovery(_path: Path) -> dict[str, object]:
        raise AssertionError("matching checkpoint must not rehash the ledger")

    monkeypatch.setattr(store, "_recover_chain_tail", unexpected_recovery)

    assert [row["index"] for row in store.read_chain(path)] == [0, 1]


def _install_decode_lock_probe(
    path: Path, monkeypatch: pytest.MonkeyPatch
) -> Callable[[], int]:
    """Assert every strict ledger decode runs after releasing its lock."""

    lock_path = path.with_suffix(path.suffix + ".lock")
    original_decode = store._decode_chain_bytes
    calls = 0

    def probe(data: bytes, **kwargs: object) -> object:
        nonlocal calls
        handle = store.acquire_nonblocking_lock(lock_path)
        assert handle is not None, "ledger lock was held during decode"
        store.release_lock(handle)
        calls += 1
        return original_decode(data, **kwargs)

    monkeypatch.setattr(store, "_decode_chain_bytes", probe)
    return lambda: calls


def test_read_chain_repairs_stale_checkpoint_after_unlocked_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "stale.jsonl"
    expected = store.append_chain_batch(path, ({"index": 0}, {"index": 1}))
    store._chain_checkpoint_path(path).write_bytes(b"invalid checkpoint")
    original_decode = store._decode_chain_bytes
    lock_path = path.with_suffix(path.suffix + ".lock")

    def probe_lock(data: bytes, **kwargs: object) -> object:
        handle = store.acquire_nonblocking_lock(lock_path)
        assert handle is not None
        store.release_lock(handle)
        return original_decode(data, **kwargs)

    monkeypatch.setattr(store, "_decode_chain_bytes", probe_lock)
    assert store.read_chain(path) == expected
    assert store._read_chain_checkpoint(path) == store.chain_head(path)


def test_read_chain_repairs_a_torn_final_row_after_unlocked_attempt(
    tmp_path: Path,
) -> None:
    path = tmp_path / "torn.jsonl"
    expected = store.append_chain(path, {"index": 0})
    with path.open("ab") as handle:
        handle.write(b'{"index":')
        handle.flush()

    assert store.read_chain(path) == [expected]
    assert store.verify_chain(path) == {
        "records": 1,
        "head_sha256": expected["record_sha256"],
    }


def test_read_chain_accepts_a_valid_final_row_without_newline(
    tmp_path: Path,
) -> None:
    path = tmp_path / "legacy-no-newline.jsonl"
    expected = store.append_chain(path, {"index": 0})
    path.write_bytes(path.read_bytes().removesuffix(b"\n"))
    before = path.read_bytes()

    assert store.read_chain(path) == [expected]
    assert path.read_bytes() == before
    assert store.chain_head(path) == {
        "records": 1,
        "head_sha256": expected["record_sha256"],
    }


def test_read_chain_fails_closed_on_middle_corruption_without_locked_rescan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "middle-corruption.jsonl"
    rows = store.append_chain_batch(path, ({"index": 0}, {"index": 1}))
    first = dict(rows[0])
    first["index"] = 9
    unsigned = {key: value for key, value in first.items() if key != "record_sha256"}
    first["record_sha256"] = canonical_json_sha256_strict(unsigned)
    raw = path.read_bytes().splitlines(keepends=True)
    path.write_bytes(canonical_json_bytes_strict(first) + b"\n" + raw[1])

    monkeypatch.setattr(
        store,
        "_recover_chain_tail",
        lambda _path: (_ for _ in ()).throw(
            AssertionError("middle corruption must not rescan under lock")
        ),
    )
    with pytest.raises(store.DistillationStoreError, match="chain"):
        store.read_chain(path)


def _tail_cursor(path: Path) -> dict[str, object]:
    rows, cursor, reset = store.read_chain_tail(path)
    assert reset is True
    assert len(rows) == cursor["records"]
    return cursor


def test_read_chain_tail_returns_cold_suffix_and_unchanged_snapshot(
    tmp_path: Path,
) -> None:
    path = tmp_path / "tail.jsonl"
    store.append_chain_batch(path, ({"index": 0}, {"index": 1}))

    rows, head, reset = store.read_chain_tail(path)
    assert [row["index"] for row in rows] == [0, 1]
    assert set(head) == {
        "ledger_path",
        "file_state",
        "offset",
        "records",
        "head_sha256",
        "generation",
    }
    assert head["records"] == store.chain_head(path)["records"]
    assert isinstance(head["generation"], str)
    assert reset is True

    cursor = _tail_cursor(path)
    assert store.read_chain_tail(path, cursor) == ([], head, False)
    store.append_chain(path, {"index": 2})
    rows, head, reset = store.read_chain_tail(path, cursor)
    assert [row["index"] for row in rows] == [2]
    assert head["records"] == store.chain_head(path)["records"]
    assert head["generation"] == cursor["generation"]
    assert reset is False


def test_read_chain_tail_resets_after_rewrite_and_new_writer_generation(
    tmp_path: Path,
) -> None:
    path = tmp_path / "tail.jsonl"
    original = store.append_chain_batch(path, ({"index": 0}, {"index": 1}))
    cursor = _tail_cursor(path)

    # Rebuild a valid chain after changing the already-cursor-covered prefix,
    # then append a valid suffix without using the normal writer checkpoint path.
    rewritten: list[dict[str, Any]] = []
    previous = ""
    for index, source in enumerate(original):
        unsigned = {
            key: value for key, value in source.items() if key != "record_sha256"
        }
        if index == 0:
            unsigned["index"] = 9
        unsigned["previous_sha256"] = previous
        row = {
            **unsigned,
            "record_sha256": canonical_json_sha256_strict(unsigned),
        }
        rewritten.append(row)
        previous = row["record_sha256"]
    path.write_bytes(
        b"".join(canonical_json_bytes_strict(row) + b"\n" for row in rewritten)
    )
    payload = {
        "schema": store.DISTILLATION_SCHEMA,
        "namespace": "recall-distillation",
        "previous_sha256": previous,
        "index": 2,
    }
    suffix = {
        **payload,
        "record_sha256": canonical_json_sha256_strict(payload),
    }
    store._append_fsynced(path, canonical_json_bytes_strict(suffix) + b"\n")
    resealed_head = store._recover_chain_tail(path)
    store._write_chain_checkpoint(path, resealed_head)

    rows, current, reset = store.read_chain_tail(path, cursor)
    assert [row["index"] for row in rows] == [9, 1, 2]
    assert reset is True
    assert current["generation"] != cursor["generation"]


def test_legacy_checkpoint_generation_forces_one_tail_reset(
    tmp_path: Path,
) -> None:
    path = tmp_path / "legacy.jsonl"
    store.append_chain(path, {"index": 0})
    cursor = _tail_cursor(path)
    checkpoint = store._chain_checkpoint_path(path)
    payload = store.read_sealed(checkpoint, schema=store.DISTILLATION_SCHEMA)
    payload.pop("generation", None)
    payload = store._sealed(payload)
    checkpoint.write_bytes(store.canonical_json_bytes_strict(payload))

    store.append_chain(path, {"index": 1})
    rows, migrated, reset = store.read_chain_tail(path, cursor)
    assert [row["index"] for row in rows] == [0, 1]
    assert reset is True
    assert isinstance(migrated["generation"], str)

    store.append_chain(path, {"index": 2})
    rows, current, reset = store.read_chain_tail(path, migrated)
    assert [row["index"] for row in rows] == [2]
    assert reset is False
    assert current["generation"] == migrated["generation"]


def test_read_chain_tail_reads_only_the_verified_suffix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "tail.jsonl"
    store.append_chain_batch(path, ({"index": index} for index in range(25)))
    cursor = _tail_cursor(path)
    store.append_chain(path, {"index": 25})
    calls: list[tuple[int, int]] = []
    original_pread = store.os.pread

    def tracking_pread(fd: int, size: int, offset: int) -> bytes:
        calls.append((size, offset))
        return original_pread(fd, size, offset)

    monkeypatch.setattr(store.os, "pread", tracking_pread)
    rows, _current, reset = store.read_chain_tail(path, cursor)

    assert [row["index"] for row in rows] == [25]
    assert reset is False
    assert calls[0] == (1, cursor["offset"] - 1)
    assert all(offset >= cursor["offset"] - 1 for _size, offset in calls)
    assert sum(size for size, _offset in calls) < cursor["offset"]


def test_read_chain_tail_decodes_without_holding_ledger_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "unlocked.jsonl"
    store.append_chain(path, {"index": 0})
    original_decode = store._decode_chain_bytes
    lock_path = path.with_suffix(path.suffix + ".lock")

    def probe_lock(data: bytes, **kwargs: object) -> object:
        handle = store.acquire_nonblocking_lock(lock_path)
        assert handle is not None
        store.release_lock(handle)
        return original_decode(data, **kwargs)

    monkeypatch.setattr(store, "_decode_chain_bytes", probe_lock)
    rows, _cursor, reset = store.read_chain_tail(path)
    assert [row["index"] for row in rows] == [0]
    assert reset is True


def test_read_chain_tail_fails_closed_on_clean_truncation(tmp_path: Path) -> None:
    path = tmp_path / "tail.jsonl"
    store.append_chain_batch(path, ({"index": 0}, {"index": 1}))
    cursor = _tail_cursor(path)
    first_end = path.read_bytes().find(b"\n") + 1
    path.write_bytes(path.read_bytes()[:first_end])

    with pytest.raises(store.DistillationStoreError, match="truncated"):
        store.read_chain_tail(path, cursor)


def test_read_chain_tail_treats_absent_nonzero_checkpoint_as_reset(
    tmp_path: Path,
) -> None:
    path = tmp_path / "missing.jsonl"
    store.write_sealed_state(
        store._chain_checkpoint_path(path),
        {
            "kind": "ledger-chain-checkpoint",
            "ledger_name": path.name,
            "records": 1,
            "head_sha256": "a" * 64,
            "file_state": None,
        },
    )

    rows, cursor, reset = store.read_chain_tail(path)
    assert rows == []
    assert cursor["records"] == 0
    assert cursor["head_sha256"] == ""
    assert reset is True


def test_read_chain_tail_resets_when_absent_cursor_ledger_is_created(
    tmp_path: Path,
) -> None:
    path = tmp_path / "created.jsonl"
    rows, cursor, reset = store.read_chain_tail(path)
    assert rows == []
    assert cursor["file_state"] is None
    assert reset is True

    appended = store.append_chain(path, {"index": 0})
    rows, current, reset = store.read_chain_tail(path, cursor)
    assert rows == [appended]
    assert reset is True
    assert isinstance(current["generation"], str)


def test_read_chain_tail_resets_on_corrupt_checkpoint_without_resealing(
    tmp_path: Path,
) -> None:
    path = tmp_path / "tail.jsonl"
    store.append_chain(path, {"index": 0})
    cursor = _tail_cursor(path)
    checkpoint = store._chain_checkpoint_path(path)
    checkpoint.write_bytes(b"corrupt")

    rows, head, reset = store.read_chain_tail(path, cursor)
    assert [row["index"] for row in rows] == [0]
    assert head["records"] == 1
    assert reset is True
    assert checkpoint.read_bytes() == b"corrupt"


@pytest.mark.parametrize("mutation", ["rewrite", "append"])
def test_read_chain_tail_rejects_mutation_during_decode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    path = tmp_path / "tail.jsonl"
    store.append_chain_batch(path, ({"index": 0}, {"index": 1}))
    original_pread = store.os.pread
    mutated = False

    def mutating_pread(fd: int, size: int, offset: int) -> bytes:
        nonlocal mutated
        data = original_pread(fd, size, offset)
        if not mutated:
            mutated = True
            if mutation == "rewrite":
                raw = path.read_bytes()
                path.write_bytes(raw.replace(b'"index":1', b'"index":2'))
            else:
                store.append_chain(path, {"index": 2})
        return data

    monkeypatch.setattr(store.os, "pread", mutating_pread)
    with pytest.raises(store.DistillationStoreBusy, match="changed"):
        store.read_chain_tail(path)


def test_read_chain_tail_rejects_mutation_after_decode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "tail.jsonl"
    store.append_chain_batch(path, ({"index": 0}, {"index": 1}))
    original_hash = store.canonical_json_sha256_strict
    calls = 0
    mutated = False

    def mutating_hash(value: object) -> str:
        nonlocal calls, mutated
        digest = original_hash(value)
        calls += 1
        if calls >= 3 and not mutated:
            mutated = True
            raw = path.read_bytes()
            path.write_bytes(raw.replace(b'"index":1', b'"index":2'))
        return digest

    monkeypatch.setattr(store, "canonical_json_sha256_strict", mutating_hash)
    with pytest.raises(store.DistillationStoreBusy, match="during snapshot decode"):
        store.read_chain_tail(path)


def test_snapshot_closes_descriptor_when_pin_races(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "race.jsonl"
    store.append_chain(path, {"index": 0})
    opened: list[int] = []
    closed: list[int] = []
    original_open = store.os.open
    original_close = store.os.close
    original_state = store._ledger_file_state_from_stat

    def tracking_open(*args: object, **kwargs: object) -> int:
        descriptor = original_open(*args, **kwargs)
        opened.append(descriptor)
        return descriptor

    def tracking_close(descriptor: int) -> None:
        closed.append(descriptor)
        original_close(descriptor)

    def fail_state(_value: object) -> dict[str, int]:
        raise store.DistillationStoreError("injected pin race")

    monkeypatch.setattr(store.os, "open", tracking_open)
    monkeypatch.setattr(store.os, "close", tracking_close)
    monkeypatch.setattr(store, "_ledger_file_state_from_stat", fail_state)
    with pytest.raises(store.DistillationStoreBusy, match="cannot be pinned"):
        store.read_chain_tail(path)

    assert opened
    assert opened[-1] in closed
    monkeypatch.setattr(store, "_ledger_file_state_from_stat", original_state)


def test_unique_label_append_updates_existing_health_projection(tmp_path: Path) -> None:
    path = tmp_path / store.LABEL_LEDGER_FILE
    store.append_chain_unique(
        path,
        {"decision_id": "one", "idempotency_sha256": "binding-one", "authority": "teacher-only"},
        unique_field="decision_id",
        binding_field="idempotency_sha256",
    )
    assert store.label_health_projection(path, repair=True)["counts"] == {
        "teacher_only": 1,
        "verified_truth": 0,
        "probe_not_truth": 0,
    }
    store.append_chain_unique(
        path,
        {"decision_id": "two", "idempotency_sha256": "binding-two", "authority": "verified"},
        unique_field="decision_id",
        binding_field="idempotency_sha256",
    )
    assert store.label_health_projection(path)["counts"] == {
        "teacher_only": 1,
        "verified_truth": 1,
        "probe_not_truth": 0,
    }


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


@pytest.mark.parametrize("state", ["missing", "stale"])
def test_unique_index_preparation_decodes_without_holding_ledger_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, state: str
) -> None:
    path = tmp_path / "receipts.jsonl"
    if state == "missing":
        store.append_chain(
            path,
            {"decision_id": "decision-1", "idempotency_sha256": "binding-1"},
        )
    else:
        store.append_chain_unique(
            path,
            _unique_payload("decision-1", "binding-1"),
            unique_field="decision_id",
            binding_field="idempotency_sha256",
        )
        store.append_chain(
            path,
            {"decision_id": "decision-2", "idempotency_sha256": "binding-2"},
        )
    decode_calls = _install_decode_lock_probe(path, monkeypatch)

    appended = store.append_chain_unique(
        path,
        _unique_payload("decision-3", "binding-3"),
        unique_field="decision_id",
        binding_field="idempotency_sha256",
    )

    assert appended["decision_id"] == "decision-3"
    assert store.verify_chain(path)["records"] == (2 if state == "missing" else 3)
    assert decode_calls() > 0


def test_append_chain_unique_locked_requires_prepared_index(tmp_path: Path) -> None:
    path = tmp_path / "receipts.jsonl"
    store.append_chain(
        path,
        {"decision_id": "decision-1", "idempotency_sha256": "binding-1"},
    )

    with store._locked(path.with_suffix(path.suffix + ".lock")):
        with pytest.raises(store.DistillationStoreBusy, match="requires preparation"):
            store.append_chain_unique_locked(
                path,
                _unique_payload("decision-2", "binding-2"),
                unique_field="decision_id",
                binding_field="idempotency_sha256",
            )


def test_unique_index_preparation_fails_closed_on_source_rewrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "receipts.jsonl"
    store.append_chain(
        path,
        {"decision_id": "decision-1", "idempotency_sha256": "binding-1"},
    )
    before = path.read_bytes()
    rewritten = before.replace(b'"decision-1"', b'"decision-2"', 1)
    assert rewritten != before and len(rewritten) == len(before)
    original_decode = store._decode_chain_bytes
    mutated = False

    def rewrite_once(data: bytes, **kwargs: object) -> object:
        nonlocal mutated
        if not mutated:
            path.write_bytes(rewritten)
            mutated = True
        return original_decode(data, **kwargs)

    monkeypatch.setattr(store, "_decode_chain_bytes", rewrite_once)
    with pytest.raises(store.DistillationStoreBusy, match="changed during snapshot"):
        store.append_chain_unique(
            path,
            _unique_payload("decision-3", "binding-3"),
            unique_field="decision_id",
            binding_field="idempotency_sha256",
        )

    assert mutated
    assert b'"decision-3"' not in path.read_bytes()


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
    decode_calls = _install_decode_lock_probe(path, monkeypatch)
    store.append_chain_unique(
        path,
        _unique_payload("decision-2", "binding-2"),
        unique_field="decision_id",
        binding_field="idempotency_sha256",
    )

    assert decode_calls() > 0
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
    decode_calls = _install_decode_lock_probe(path, monkeypatch)

    assert (
        store.append_chain_unique(
            path,
            _unique_payload("decision-1", "binding-1"),
            unique_field="decision_id",
            binding_field="idempotency_sha256",
        )
        == first
    )
    assert decode_calls() > 0
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
