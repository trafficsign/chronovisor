from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

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

    from chronovisor.core.raw_store import RawStore

    original_read = RawStore.read_bytes
    reads: list[str] = []

    def record_read(self: RawStore, raw: object) -> bytes:
        reads.append(getattr(raw, "raw_id", str(raw)))
        return original_read(self, raw)

    monkeypatch.setattr(RawStore, "read_bytes", record_read)
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

    from chronovisor.core.raw_store import RawStore

    def unexpected_read(self: RawStore, raw: object) -> bytes:
        raise AssertionError(f"exact parity reread Raw: {raw}")

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

    monkeypatch.setattr(RawStore, "read_bytes", unexpected_read)
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

    from chronovisor.core.raw_store import RawStore

    original_read = RawStore.read_bytes
    reads: list[str] = []

    def record_read(self: RawStore, raw: object) -> bytes:
        reads.append(getattr(raw, "raw_id", str(raw)))
        return original_read(self, raw)

    monkeypatch.setattr(RawStore, "read_bytes", record_read)
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
    assert calls == 2
