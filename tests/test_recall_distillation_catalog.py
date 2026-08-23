from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Iterable, Iterator
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from chronovisor.core.canonical_json import canonical_json_sha256_strict
from chronovisor.core.raw_segment import append_capture
from chronovisor.core.store import RuntimeContext, init_chronovisor
from chronovisor.recall import recall_distillation_catalog as catalog
from chronovisor.recall import recall_distillation_store as store


def _message(role: str, text: str, timestamp: str) -> dict[str, object]:
    return {
        "type": "response_item",
        "timestamp": timestamp,
        "payload": {
            "type": "message",
            "role": role,
            "content": [
                {
                    "type": "input_text" if role == "user" else "output_text",
                    "text": text,
                }
            ],
        },
    }


def _capture(
    root: Path,
    raw_id: str,
    session_key: str,
    events: list[dict[str, object]],
    *,
    after_line: int = 0,
) -> Path:
    raw_dir = root / "raw"
    source = root / f"{raw_id}.jsonl"
    payload = b"".join(
        json.dumps(event, separators=(",", ":")).encode() + b"\n" for event in events
    )
    source.write_bytes(payload)
    append_capture(
        raw_dir=raw_dir,
        raw_id=raw_id,
        idempotency_key=raw_id.removeprefix("save-").removesuffix(".md"),
        host="codex",
        session_key=session_key,
        session_id=session_key,
        source_file=source,
        after_line=after_line,
        until_line=after_line + len(events),
        source_bytes=payload,
        record_count=len(events),
        now=datetime(2026, 8, 23, tzinfo=ZoneInfo("Asia/Tokyo")),
    )
    return raw_dir


def test_bootstrap_catalog_keeps_text_out_of_sqlite_and_resolves_refs(
    tmp_path: Path,
) -> None:
    init_chronovisor(RuntimeContext(tmp_path))
    raw_dir = _capture(
        tmp_path,
        "save-codex-one.md",
        "a" * 24,
        [
            _message("user", "alpha question", "2026-08-01T00:00:00Z"),
            _message("assistant", "alpha answer", "2026-08-01T00:00:01Z"),
        ],
    )

    result = catalog.advance(raw_dir, tmp_path, 4096)
    rows = catalog.rallies(tmp_path)

    assert result.status == "bootstrap"
    assert len(rows) == 1
    assert "alpha" not in json.dumps(rows)
    answer_ref = rows[0]["actual_answer_refs"][0]
    assert catalog.texts(raw_dir, tmp_path, refs=[answer_ref]) == {
        answer_ref["semantic_sha256"]: "alpha answer"
    }
    with sqlite3.connect(catalog.catalog_path(tmp_path)) as connection:
        event_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(events)")
        }
    assert "text" not in event_columns


def test_text_cache_reads_only_requested_raw_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    init_chronovisor(RuntimeContext(tmp_path))
    raw_dir = _capture(
        tmp_path,
        "save-codex-one.md",
        "a" * 24,
        [
            _message("user", "first query", "2026-08-01T00:00:00Z"),
            _message("assistant", "first answer", "2026-08-01T00:00:01Z"),
        ],
    )
    _capture(
        tmp_path,
        "save-codex-two.md",
        "b" * 24,
        [_message("assistant", "second answer", "2026-08-02T00:00:01Z")],
    )
    catalog.advance(raw_dir, tmp_path, 4096)
    with sqlite3.connect(catalog.catalog_path(tmp_path)) as connection:
        first_hash = connection.execute(
            "SELECT semantic_sha256 FROM events WHERE raw_id='save-codex-one.md' "
            "AND role='assistant'"
        ).fetchone()[0]

    from chronovisor.core.raw_store import RawStore, RawUnit

    original_read = RawStore.iter_segment_bytes
    reads: list[str] = []

    def record_read(
        self: RawStore, raw_ids: Iterable[str] | None = None
    ) -> Iterator[tuple[RawUnit, bytes]]:
        for unit, raw in original_read(self, raw_ids):
            reads.append(unit.raw_id)
            yield unit, raw

    monkeypatch.setattr(RawStore, "iter_segment_bytes", record_read)
    cache = catalog.CatalogTextCache(raw_dir, tmp_path)

    assert cache.get(first_hash) == "first answer"
    assert cache.get(first_hash) == "first answer"
    assert reads == ["save-codex-one.md"]


def test_steady_state_same_watermark_reads_no_raw_and_new_session_is_delta(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    init_chronovisor(RuntimeContext(tmp_path))
    raw_dir = _capture(
        tmp_path,
        "save-codex-one.md",
        "a" * 24,
        [_message("user", "first", "2026-08-01T00:00:00Z")],
    )
    catalog.advance(raw_dir, tmp_path, 4096)

    from chronovisor.core.raw_store import RawStore

    original_read = RawStore.read_bytes

    def unexpected_read(self: RawStore, raw: object) -> bytes:
        raise AssertionError(f"no-op read Raw: {raw}")

    monkeypatch.setattr(RawStore, "read_bytes", unexpected_read)
    assert catalog.advance(raw_dir, tmp_path, 4096).status == "noop"
    monkeypatch.setattr(RawStore, "read_bytes", original_read)

    _capture(
        tmp_path,
        "save-codex-two.md",
        "b" * 24,
        [
            _message("user", "second", "2026-08-02T00:00:00Z"),
            _message("assistant", "second answer", "2026-08-02T00:00:01Z"),
        ],
    )
    reads: list[str] = []

    def record_read(self: RawStore, raw: object) -> bytes:
        reads.append(getattr(raw, "raw_id", str(raw)))
        return original_read(self, raw)

    monkeypatch.setattr(RawStore, "read_bytes", record_read)
    result = catalog.advance(raw_dir, tmp_path, 4096)

    assert result.status == "advanced"
    assert result.indexed_raw_ids == ("save-codex-two.md",)
    assert len(result.rally_ids) == 1
    assert reads == ["save-codex-two.md"]


def test_post_commit_crash_retries_as_noop_without_duplicate_or_raw_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    init_chronovisor(RuntimeContext(tmp_path))
    raw_dir = _capture(
        tmp_path,
        "save-codex-one.md",
        "a" * 24,
        [_message("user", "first", "2026-08-01T00:00:00Z")],
    )
    catalog.advance(raw_dir, tmp_path, 4096)
    _capture(
        tmp_path,
        "save-codex-two.md",
        "b" * 24,
        [
            _message("user", "second", "2026-08-02T00:00:00Z"),
            _message("assistant", "second answer", "2026-08-02T00:00:01Z"),
        ],
    )

    real_connect = catalog._connect
    crashed = {"value": False}

    class CommitThenCrash:
        def __init__(self, connection: sqlite3.Connection) -> None:
            self.connection = connection

        def __getattr__(self, name: str) -> object:
            return getattr(self.connection, name)

        def commit(self) -> None:
            self.connection.commit()
            if not crashed["value"]:
                crashed["value"] = True
                raise RuntimeError("crash after catalog commit")

    def crashing_connect(root: Path) -> CommitThenCrash:
        return CommitThenCrash(real_connect(root))

    monkeypatch.setattr(catalog, "_connect", crashing_connect)
    with pytest.raises(RuntimeError, match="after catalog commit"):
        catalog.advance(raw_dir, tmp_path, 4096)

    monkeypatch.setattr(catalog, "_connect", real_connect)

    from chronovisor.core.raw_store import RawStore

    def unexpected_read(self: RawStore, raw: object) -> bytes:
        raise AssertionError(f"retry reread Raw: {raw}")

    monkeypatch.setattr(RawStore, "read_bytes", unexpected_read)
    result = catalog.advance(raw_dir, tmp_path, 4096)

    assert result.status == "noop"
    assert result.indexed_raw_ids == ()
    with sqlite3.connect(catalog.catalog_path(tmp_path)) as connection:
        assert connection.execute("SELECT COUNT(*) FROM raw_units").fetchone()[0] == 2
        assert connection.execute("SELECT COUNT(*) FROM rallies").fetchone()[0] == 2


def test_digest_conflict_fails_closed(tmp_path: Path) -> None:
    init_chronovisor(RuntimeContext(tmp_path))
    raw_dir = _capture(
        tmp_path,
        "save-codex-one.md",
        "a" * 24,
        [_message("user", "first", "2026-08-01T00:00:00Z")],
    )
    catalog.advance(raw_dir, tmp_path, 4096)
    _capture(
        tmp_path,
        "save-codex-two.md",
        "b" * 24,
        [_message("user", "second", "2026-08-02T00:00:00Z")],
    )
    with sqlite3.connect(catalog.catalog_path(tmp_path)) as connection:
        connection.execute("UPDATE raw_units SET raw_sha256='conflict'")

    with pytest.raises(catalog.CatalogError, match="digest"):
        catalog.advance(raw_dir, tmp_path, 4096)
    with sqlite3.connect(catalog.catalog_path(tmp_path)) as connection:
        assert connection.execute("SELECT COUNT(*) FROM raw_units").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 1


def test_existing_session_delta_is_deferred_without_false_rally_boundary(
    tmp_path: Path,
) -> None:
    init_chronovisor(RuntimeContext(tmp_path))
    raw_dir = _capture(
        tmp_path,
        "save-codex-one.md",
        "a" * 24,
        [_message("user", "first", "2026-08-01T00:00:00Z")],
    )
    catalog.advance(raw_dir, tmp_path, 4096)
    initial = catalog.rallies(tmp_path)
    _capture(
        tmp_path,
        "save-codex-two.md",
        "a" * 24,
        [_message("assistant", "late answer", "2026-08-01T00:00:01Z")],
        after_line=1,
    )

    result = catalog.advance(raw_dir, tmp_path, 4096)

    assert result.rally_ids == (initial[0]["rally_id"],)
    assert result.deferred_session_keys == (("codex", "a" * 24),)
    rows = catalog.rallies(tmp_path)
    assert len(rows) == 1
    assert rows[0]["rally_id"] == initial[0]["rally_id"]
    assert [ref["raw_id"] for ref in rows[0]["actual_answer_refs"]] == [
        "save-codex-two.md"
    ]


def test_overlapping_session_source_position_fails_closed(tmp_path: Path) -> None:
    init_chronovisor(RuntimeContext(tmp_path))
    raw_dir = _capture(
        tmp_path,
        "save-codex-one.md",
        "a" * 24,
        [_message("user", "first", "2026-08-01T00:00:00Z")],
    )
    catalog.advance(raw_dir, tmp_path, 4096)
    _capture(
        tmp_path,
        "save-codex-two.md",
        "a" * 24,
        [_message("assistant", "overlap", "2026-08-01T00:00:01Z")],
    )

    with pytest.raises(sqlite3.IntegrityError):
        catalog.advance(raw_dir, tmp_path, 4096)

    with sqlite3.connect(catalog.catalog_path(tmp_path)) as connection:
        assert connection.execute("SELECT COUNT(*) FROM raw_units").fetchone()[0] == 1


def test_historical_index_adopts_exact_catalog_parity_without_raw_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    init_chronovisor(RuntimeContext(tmp_path))
    raw_dir = _capture(
        tmp_path,
        "save-codex-one.md",
        "a" * 24,
        [
            _message("user", "query", "2026-08-01T00:00:00Z"),
            _message("assistant", "adoptable answer", "2026-08-01T00:00:01Z"),
        ],
    )
    catalog.advance(raw_dir, tmp_path, 4096)
    first = catalog.sync_historical_index(raw_dir, tmp_path)

    from chronovisor.core.raw_store import RawStore, RawUnit

    def unexpected_read(
        self: RawStore, raw_ids: Iterable[str] | None = None
    ) -> Iterator[tuple[RawUnit, bytes]]:
        raise AssertionError(f"exact parity reread Raw: {raw_ids}")
        yield from ()

    original_connect = catalog.sqlite3.connect

    class NoIndexScan(sqlite3.Connection):
        def execute(
            self, statement: str, *args: object, **kwargs: object
        ) -> sqlite3.Cursor:
            if "from atoms" in statement.lower():
                raise AssertionError("exact parity scanned historical index")
            return super().execute(statement, *args, **kwargs)

    def guarded_connect(*args: object, **kwargs: object) -> sqlite3.Connection:
        kwargs["factory"] = NoIndexScan
        return original_connect(*args, **kwargs)

    monkeypatch.setattr(RawStore, "iter_segment_bytes", unexpected_read)
    monkeypatch.setattr(catalog.sqlite3, "connect", guarded_connect)
    monkeypatch.setattr(
        store,
        "_search_terms",
        lambda _text: (_ for _ in ()).throw(AssertionError("exact parity tokenized")),
    )
    assert catalog.sync_historical_index(raw_dir, tmp_path) == first


def test_historical_index_reads_only_delta_raw_and_is_searchable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    init_chronovisor(RuntimeContext(tmp_path))
    raw_dir = _capture(
        tmp_path,
        "save-codex-one.md",
        "a" * 24,
        [
            _message("user", "first", "2026-08-01T00:00:00Z"),
            _message("assistant", "first answer", "2026-08-01T00:00:01Z"),
        ],
    )
    catalog.advance(raw_dir, tmp_path, 4096)
    catalog.sync_historical_index(raw_dir, tmp_path)
    _capture(
        tmp_path,
        "save-codex-two.md",
        "b" * 24,
        [
            _message("user", "second", "2026-08-02T00:00:00Z"),
            _message("assistant", "delta search answer", "2026-08-02T00:00:01Z"),
        ],
    )
    catalog.advance(raw_dir, tmp_path, 4096)

    from chronovisor.core.raw_store import RawStore, RawUnit

    original_read = RawStore.iter_segment_bytes
    reads: list[str] = []

    def record_read(
        self: RawStore, raw_ids: Iterable[str] | None = None
    ) -> Iterator[tuple[RawUnit, bytes]]:
        for unit, raw in original_read(self, raw_ids):
            reads.append(unit.raw_id)
            yield unit, raw

    monkeypatch.setattr(RawStore, "iter_segment_bytes", record_read)
    digest = catalog.sync_historical_index(raw_dir, tmp_path)

    assert reads == ["save-codex-two.md"]
    assert len(digest) == 64
    found = store.search_historical_index(
        catalog.historical_index_path(tmp_path),
        query="delta search",
        as_of_us=9_999_999_999_999_999,
        host="other",
        session_cluster_id="other",
        source_index=0,
        limit=10,
    )
    assert any(row["text_sha256"] for row in found)


def test_historical_index_conflicting_atom_fails_closed(tmp_path: Path) -> None:
    init_chronovisor(RuntimeContext(tmp_path))
    raw_dir = _capture(
        tmp_path,
        "save-codex-one.md",
        "a" * 24,
        [
            _message("user", "query", "2026-08-01T00:00:00Z"),
            _message("assistant", "answer", "2026-08-01T00:00:01Z"),
        ],
    )
    catalog.advance(raw_dir, tmp_path, 4096)
    catalog.sync_historical_index(raw_dir, tmp_path)
    with sqlite3.connect(catalog.historical_index_path(tmp_path)) as connection:
        connection.execute("UPDATE atoms SET host='conflict'")

    with pytest.raises(catalog.CatalogError, match="assistant atom"):
        catalog.sync_historical_index(raw_dir, tmp_path)


def test_historical_index_checkpoint_mismatch_revalidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    init_chronovisor(RuntimeContext(tmp_path))
    raw_dir = _capture(
        tmp_path,
        "save-codex-one.md",
        "a" * 24,
        [
            _message("user", "query", "2026-08-01T00:00:00Z"),
            _message("assistant", "answer", "2026-08-01T00:00:01Z"),
        ],
    )
    catalog.advance(raw_dir, tmp_path, 4096)
    digest = catalog.sync_historical_index(raw_dir, tmp_path)
    index_path = catalog.historical_index_path(tmp_path)
    catalog._index_checkpoint_path(index_path).write_text("{}", encoding="utf-8")
    inspect = catalog._index_atoms
    calls = 0

    def tracked_inspect(
        connection: sqlite3.Connection, expected: object
    ) -> tuple[dict[str, dict[str, object]], str]:
        nonlocal calls
        calls += 1
        return inspect(connection, expected)

    monkeypatch.setattr(catalog, "_index_atoms", tracked_inspect)
    assert catalog.sync_historical_index(raw_dir, tmp_path) == digest
    assert calls == 1

    state = index_path.stat()
    os.utime(index_path, ns=(state.st_atime_ns, state.st_mtime_ns + 1_000_000))
    assert catalog.sync_historical_index(raw_dir, tmp_path) == digest
    assert calls == 2
    assert catalog.sync_historical_index(raw_dir, tmp_path) == digest


def _candidate_snapshot(rally_id: str, candidate_ids: list[str]) -> dict[str, object]:
    snapshot: dict[str, object] = {
        "schema": catalog.CANDIDATE_SNAPSHOT_SCHEMA,
        "rally_id": rally_id,
        "as_of": "2026-08-23T00:00:00Z",
        "retriever_revision": "historical-fts-v1",
        "feature_revision": "recall-distill-text-v2",
        "query_feature_text_sha256": "a" * 64,
        "candidates": [
            {
                "candidate_id": candidate_id,
                "rank": index + 1,
                "text_sha256": (str(index + 1) * 64)[:64],
                "candidate_feature_text_sha256": "b" * 64,
            }
            for index, candidate_id in enumerate(candidate_ids)
        ],
    }
    snapshot["snapshot_sha256"] = canonical_json_sha256_strict(snapshot)
    return snapshot


def test_candidate_offset_index_bootstrap_tail_noop_and_random_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = store.distillation_dir(tmp_path) / "candidate-ledger.jsonl"
    first = {
        "kind": "candidate-snapshot",
        "rally_id": "rally-1",
        "snapshot": _candidate_snapshot("rally-1", ["c1", "c2", "c3", "c4"]),
    }
    store.append_chain(ledger, first)
    bootstrap = catalog.sync_candidate_index(tmp_path, ledger)
    assert bootstrap["count"] == 1
    assert bootstrap["head_sha256"]
    assert catalog.catalog_path(tmp_path).stat().st_mode & 0o777 == 0o600
    with sqlite3.connect(catalog.catalog_path(tmp_path)) as connection:
        assert connection.execute("SELECT COUNT(*) FROM candidate_records").fetchone()[0] == 1
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(candidate_records)")
        }
        assert "snapshot" not in columns
        assert "features" not in columns

    original_loads = catalog.json.loads
    monkeypatch.setattr(
        catalog.json,
        "loads",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("no-change sync decoded the ledger")
        ),
    )
    assert catalog.sync_candidate_index(tmp_path, ledger)["status"] == "noop"
    monkeypatch.setattr(catalog.json, "loads", original_loads)

    second = {
        "kind": "candidate-snapshot",
        "rally_id": "rally-2",
        "snapshot": _candidate_snapshot("rally-2", ["c5"]),
    }
    store.append_chain(ledger, second)
    loads = 0

    def count_loads(value: object, *args: object, **kwargs: object) -> object:
        nonlocal loads
        loads += 1
        return original_loads(value, *args, **kwargs)

    monkeypatch.setattr(catalog.json, "loads", count_loads)
    advanced = catalog.sync_candidate_index(tmp_path, ledger)
    assert advanced["status"] == "advanced"
    assert advanced["count"] == 2
    assert loads <= 2
    claim = next(
        row
        for row in store.read_chain(ledger)
        if row["rally_id"] == "rally-2"
    )
    assert catalog.read_candidate_snapshots(
        tmp_path,
        ledger,
        [{"rally_id": "rally-2", "record_sha256": claim["record_sha256"]}],
    )["rally-2"]["rally_id"] == "rally-2"
    assert catalog.read_candidate_snapshots(
        tmp_path,
        ledger,
        [
            {"payload_ref": "candidate-snapshot:rally-2:c5"},
            {"payload_ref": "candidate-snapshot:rally-2:c5"},
        ],
    )["rally-2"]["rally_id"] == "rally-2"
    with pytest.raises(catalog.CatalogError, match="absent from snapshot"):
        catalog.read_candidate_snapshots(
            tmp_path,
            ledger,
            [{"payload_ref": "candidate-snapshot:rally-2:wrong"}],
        )
    with pytest.raises(catalog.CatalogError, match="reference"):
        catalog.read_candidate_snapshots(
            tmp_path,
            ledger,
            [{"rally_id": "rally-2", "payload_ref": "garbage"}],
        )
    with pytest.raises(catalog.CatalogError, match="conflicts"):
        catalog.read_candidate_snapshots(
            tmp_path,
            ledger,
            [
                {
                    "rally_id": "rally-1",
                    "payload_ref": "candidate-snapshot:rally-2:c5",
                }
            ],
        )
    assert catalog.candidate_rally_ids(tmp_path, after_seq=1) == {"rally-2"}
    with sqlite3.connect(catalog.catalog_path(tmp_path)) as connection:
        connection.execute(
            "UPDATE candidate_records SET rally_id='tampered' WHERE rally_id='rally-1'"
        )
    with pytest.raises(catalog.CatalogError, match="index row"):
        catalog.candidate_rally_ids(tmp_path)


def test_candidate_index_bootstrap_uses_sealed_head_without_rehashing_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = store.distillation_dir(tmp_path) / "candidate-ledger.jsonl"
    snapshot = _candidate_snapshot("rally-large", ["c1"])
    candidate = snapshot["candidates"][0]
    assert isinstance(candidate, dict)
    candidate["text"] = "x" * 200_000
    snapshot["snapshot_sha256"] = canonical_json_sha256_strict(
        {key: value for key, value in snapshot.items() if key != "snapshot_sha256"}
    )
    store.append_chain(
        ledger,
        {
            "kind": "candidate-snapshot",
            "rally_id": "rally-large",
            "snapshot": snapshot,
        },
    )
    assert store.chain_head(ledger)["records"] == 1

    original_bytes = catalog.canonical_json_bytes_strict
    original_hash = catalog.canonical_json_sha256_strict

    def reject_payload_bytes(value: object, *args: object, **kwargs: object) -> bytes:
        if isinstance(value, dict) and ("snapshot" in value or "candidates" in value):
            raise AssertionError("bootstrap reserialized a large candidate payload")
        return original_bytes(value, *args, **kwargs)

    def reject_payload_hash(value: object, *args: object, **kwargs: object) -> str:
        if isinstance(value, dict) and ("snapshot" in value or "candidates" in value):
            raise AssertionError("bootstrap rehashed a large candidate payload")
        return original_hash(value, *args, **kwargs)

    monkeypatch.setattr(catalog, "canonical_json_bytes_strict", reject_payload_bytes)
    monkeypatch.setattr(catalog, "canonical_json_sha256_strict", reject_payload_hash)
    result = catalog.sync_candidate_index(tmp_path, ledger)
    assert result["status"] == "bootstrap"
    assert result["count"] == 1


def test_candidate_index_bootstrap_fails_closed_on_head_tamper(tmp_path: Path) -> None:
    ledger = store.distillation_dir(tmp_path) / "candidate-ledger.jsonl"
    store.append_chain(
        ledger,
        {
            "kind": "candidate-snapshot",
            "rally_id": "rally-1",
            "snapshot": _candidate_snapshot("rally-1", ["c1"]),
        },
    )
    tampered = ledger.read_bytes().replace(b'"rally-1"', b'"rally-x"', 1)
    ledger.write_bytes(tampered)
    with pytest.raises(catalog.CatalogError, match="candidate ledger head is invalid"):
        catalog.sync_candidate_index(tmp_path, ledger)


def test_candidate_offset_index_fails_closed_and_explicit_rebuild(
    tmp_path: Path,
) -> None:
    ledger = store.distillation_dir(tmp_path) / "candidate-ledger.jsonl"
    store.append_chain(
        ledger,
        {
            "kind": "candidate-snapshot",
            "rally_id": "rally-1",
            "snapshot": _candidate_snapshot("rally-1", ["c1"]),
        },
    )
    catalog.sync_candidate_index(tmp_path, ledger)
    with sqlite3.connect(catalog.catalog_path(tmp_path)) as connection:
        offset = connection.execute(
            "SELECT offset FROM candidate_records WHERE rally_id='rally-1'"
        ).fetchone()[0]
        connection.execute(
            "UPDATE candidate_records SET offset=? WHERE rally_id='rally-1'",
            (offset + 1,),
        )
    with pytest.raises(catalog.CatalogError, match="offset index"):
        catalog.read_candidate_snapshots(tmp_path, ledger, ["rally-1"])

    original_ledger = ledger.read_bytes()
    ledger.write_bytes(original_ledger[:-3])
    with pytest.raises(
        catalog.CatalogError, match="rollback|changed|truncated|offset index"
    ):
        catalog.sync_candidate_index(tmp_path, ledger)

    ledger.write_bytes(original_ledger)
    with sqlite3.connect(catalog.catalog_path(tmp_path)) as connection:
        connection.execute("DELETE FROM candidate_records")
    repaired = catalog.sync_candidate_index(tmp_path, ledger, rebuild=True)
    assert repaired["status"] == "bootstrap"


def test_candidate_offset_index_rejects_symlinked_catalog(tmp_path: Path) -> None:
    path = catalog.catalog_path(tmp_path)
    path.parent.mkdir(parents=True)
    target = tmp_path / "outside.sqlite"
    target.touch()
    path.symlink_to(target)

    with pytest.raises(catalog.CatalogError, match="unsafe"):
        catalog.candidate_index_state(tmp_path)
