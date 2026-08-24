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
from chronovisor.core.raw_store import RawStore
from chronovisor.core.store import RuntimeContext, init_chronovisor
from chronovisor.recall import recall_distillation as distill
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


def _rewrite_checkpoint(path: Path, checkpoint: dict[str, object]) -> None:
    store.write_sealed_state(
        path,
        {
            key: value
            for key, value in checkpoint.items()
            if key not in {"schema", "namespace", "seal_sha256"}
        },
    )


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


def test_warm_catalog_and_fts_skip_raw_inventory_identity_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    init_chronovisor(RuntimeContext(tmp_path))
    raw_dir = _capture(
        tmp_path,
        "save-codex-one.md",
        "a" * 24,
        [_message("assistant", "answer", "2026-08-01T00:00:00Z")],
    )
    catalog.advance(raw_dir, tmp_path, 4096)
    digest = catalog.sync_historical_index(raw_dir, tmp_path)

    def unexpected_inventory(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("warm path iterated the Raw inventory")

    monkeypatch.setattr(RawStore, "iter_segment_units", unexpected_inventory)
    monkeypatch.setattr(catalog, "_unit_identity", unexpected_inventory)

    assert catalog.advance(raw_dir, tmp_path, 4096).status == "noop"
    assert catalog.sync_historical_index(raw_dir, tmp_path) == digest


def test_post_commit_crash_repairs_catalog_on_retry(
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

    result = catalog.advance(raw_dir, tmp_path, 4096)
    assert result.status == "repaired"
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

    assert catalog.advance(raw_dir, tmp_path, 4096).status == "repaired"
    with sqlite3.connect(catalog.catalog_path(tmp_path)) as connection:
        assert connection.execute("SELECT COUNT(*) FROM raw_units").fetchone()[0] == 2
        assert connection.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 2


def test_catalog_event_metadata_tamper_requires_repair(tmp_path: Path) -> None:
    init_chronovisor(RuntimeContext(tmp_path))
    raw_dir = _capture(
        tmp_path,
        "save-codex-one.md",
        "a" * 24,
        [_message("user", "first", "2026-08-01T00:00:00Z")],
    )
    catalog.advance(raw_dir, tmp_path, 4096)
    with sqlite3.connect(catalog.catalog_path(tmp_path)) as connection:
        connection.execute("UPDATE events SET source_index=999")

    assert catalog.advance(raw_dir, tmp_path, 4096).status == "repaired"


def test_catalog_direct_reads_fail_closed_after_metadata_tamper(tmp_path: Path) -> None:
    init_chronovisor(RuntimeContext(tmp_path))
    raw_dir = _capture(
        tmp_path,
        "save-codex-one.md",
        "a" * 24,
        [
            _message("user", "question", "2026-08-01T00:00:00Z"),
            _message("assistant", "answer", "2026-08-01T00:00:01Z"),
        ],
    )
    catalog.advance(raw_dir, tmp_path, 4096)
    answer = catalog.rallies(tmp_path)[0]["actual_answer_refs"][0]
    with sqlite3.connect(catalog.catalog_path(tmp_path)) as connection:
        connection.execute("UPDATE events SET source_index=999 WHERE role='assistant'")

    with pytest.raises(catalog.CatalogError, match="checkpoint requires repair"):
        catalog.rallies(tmp_path)
    with pytest.raises(catalog.CatalogError, match="checkpoint requires repair"):
        catalog.texts(raw_dir, tmp_path, refs=[answer])
    with pytest.raises(catalog.CatalogError, match="checkpoint requires repair"):
        catalog.CatalogTextCache(raw_dir, tmp_path).prefetch([answer["semantic_sha256"]])


def test_catalog_text_cache_hit_fails_closed_after_tamper(tmp_path: Path) -> None:
    init_chronovisor(RuntimeContext(tmp_path))
    raw_dir = _capture(
        tmp_path,
        "save-codex-one.md",
        "a" * 24,
        [
            _message("user", "question", "2026-08-01T00:00:00Z"),
            _message("assistant", "answer", "2026-08-01T00:00:01Z"),
        ],
    )
    catalog.advance(raw_dir, tmp_path, 4096)
    answer = catalog.rallies(tmp_path)[0]["actual_answer_refs"][0]
    cache = catalog.CatalogTextCache(raw_dir, tmp_path)
    assert cache[answer["semantic_sha256"]] == "answer"
    with sqlite3.connect(catalog.catalog_path(tmp_path)) as connection:
        connection.execute("UPDATE events SET source_index=999 WHERE role='assistant'")

    with pytest.raises(catalog.CatalogError, match="checkpoint requires repair"):
        cache[answer["semantic_sha256"]]


@pytest.mark.parametrize("sidecar_action", ["unlink", "rewrite"])
def test_catalog_text_cache_hit_checks_checkpoint_sidecar(
    tmp_path: Path, sidecar_action: str
) -> None:
    init_chronovisor(RuntimeContext(tmp_path))
    raw_dir = _capture(
        tmp_path,
        "save-codex-one.md",
        "a" * 24,
        [
            _message("user", "question", "2026-08-01T00:00:00Z"),
            _message("assistant", "answer", "2026-08-01T00:00:01Z"),
        ],
    )
    catalog.advance(raw_dir, tmp_path, 4096)
    answer = catalog.rallies(tmp_path)[0]["actual_answer_refs"][0]
    cache = catalog.CatalogTextCache(raw_dir, tmp_path)
    assert cache[answer["semantic_sha256"]] == "answer"
    sidecar = catalog._catalog_checkpoint_path(tmp_path)
    if sidecar_action == "unlink":
        sidecar.unlink()
    else:
        sidecar.write_bytes(sidecar.read_bytes())

    with pytest.raises(catalog.CatalogError, match="checkpoint requires repair"):
        cache[answer["semantic_sha256"]]
    with pytest.raises(catalog.CatalogError, match="checkpoint requires repair"):
        len(cache)
    with pytest.raises(catalog.CatalogError, match="checkpoint requires repair"):
        iter(cache)


def test_legacy_catalog_and_checkpoint_loss_rebuild_once(tmp_path: Path) -> None:
    init_chronovisor(RuntimeContext(tmp_path))
    raw_dir = _capture(
        tmp_path,
        "save-codex-one.md",
        "a" * 24,
        [
            _message("user", "question", "2026-08-01T00:00:00Z"),
            _message("assistant", "answer", "2026-08-01T00:00:01Z"),
        ],
    )
    catalog.advance(raw_dir, tmp_path, 4096)
    answer = catalog.rallies(tmp_path)[0]["actual_answer_refs"][0]
    cache = catalog.CatalogTextCache(raw_dir, tmp_path)
    assert cache[answer["semantic_sha256"]] == "answer"
    path = catalog.catalog_path(tmp_path)
    catalog._catalog_checkpoint_path(tmp_path).unlink()
    assert catalog.advance(raw_dir, tmp_path, 4096).status == "repaired"

    path.unlink()
    with sqlite3.connect(path) as connection:
        connection.execute(
            """CREATE TABLE raw_units(
                raw_id TEXT PRIMARY KEY,raw_sha256 TEXT,receipt_sha256 TEXT,
                host TEXT,session_key TEXT,captured_at TEXT,record_count INTEGER,status TEXT
            )"""
        )
        connection.execute(
            "INSERT INTO raw_units VALUES(?,?,?,?,?,?,?,?)",
            ("old", "x", "y", "codex", "legacy", "now", 1, "indexed"),
        )
        connection.execute("CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL)")
        connection.execute(
            "INSERT INTO metadata VALUES('schema',?)", (catalog.CATALOG_SCHEMA,)
        )
        connection.execute("INSERT INTO metadata VALUES('watermark','legacy')")
    assert catalog.advance(raw_dir, tmp_path, 4096).status == "repaired"


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


def test_warm_existing_session_uses_metadata_tail_and_matches_full_replay(
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
        "a" * 24,
        [
            _message("tool", "tool output", "2026-08-01T00:00:02Z"),
            _message("user", "second", "2026-08-01T00:00:03Z"),
            _message("assistant", "second answer", "2026-08-01T00:00:04Z"),
        ],
        after_line=2,
    )
    expected = distill.extract_rallies(raw_dir, root=tmp_path, max_context_bytes=4096)

    from chronovisor.core.raw_store import RawStore

    original_read = RawStore.read_bytes
    reads: list[str] = []

    def only_new_raw(self: RawStore, raw: object) -> bytes:
        raw_id = getattr(raw, "raw_id", str(raw))
        reads.append(raw_id)
        if raw_id != "save-codex-two.md":
            raise AssertionError(f"warm path reread old Raw: {raw_id}")
        return original_read(self, raw)

    monkeypatch.setattr(RawStore, "read_bytes", only_new_raw)
    monkeypatch.setattr(
        catalog,
        "_session_events",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("warm path rebuilt the full session")
        ),
    )
    monkeypatch.setattr(
        distill,
        "extract_rallies",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("warm path replayed all rallies")
        ),
    )
    monkeypatch.setattr(
        store,
        "create_historical_index",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("warm path rebuilt the full FTS index")
        ),
    )

    result = catalog.advance(raw_dir, tmp_path, 4096)
    catalog.sync_historical_index(raw_dir, tmp_path)

    assert result.status == "advanced"
    assert result.deferred_session_keys == (("codex", "a" * 24),)
    assert catalog.rallies(tmp_path) == expected
    assert reads == ["save-codex-two.md"]


def test_warm_existing_session_rejects_out_of_order_source_gap(tmp_path: Path) -> None:
    init_chronovisor(RuntimeContext(tmp_path))
    raw_dir = _capture(
        tmp_path,
        "save-codex-one.md",
        "a" * 24,
        [
            _message("user", "first", "2026-08-01T00:00:00Z"),
            {"type": "unknown", "timestamp": "2026-08-01T00:00:01Z"},
            _message("assistant", "first answer", "2026-08-01T00:00:02Z"),
        ],
    )
    catalog.advance(raw_dir, tmp_path, 4096)
    _capture(
        tmp_path,
        "save-codex-two.md",
        "a" * 24,
        [_message("assistant", "late", "2026-08-01T00:00:03Z")],
        after_line=1,
    )

    with pytest.raises(catalog.CatalogError, match="source interval"):
        catalog.advance(raw_dir, tmp_path, 4096)
    with sqlite3.connect(catalog.catalog_path(tmp_path)) as connection:
        assert connection.execute("SELECT COUNT(*) FROM raw_units").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 2


def test_historical_gaps_and_unknown_only_overlap_are_distinct(tmp_path: Path) -> None:
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
        [_message("assistant", "late", "2026-08-01T00:00:01Z")],
        after_line=180,
    )
    catalog.advance(raw_dir, tmp_path, 4096)
    _capture(
        tmp_path,
        "save-codex-three.md",
        "a" * 24,
        [{"type": "unknown", "timestamp": "2026-08-01T00:00:02Z"}],
        after_line=180,
    )

    with pytest.raises(catalog.CatalogError, match="source interval"):
        catalog.advance(raw_dir, tmp_path, 4096)


def test_warm_old_raw_tamper_is_deferred_to_text_resolution(
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
    first_answer = catalog.rallies(tmp_path)[0]["actual_answer_refs"][0]
    _capture(
        tmp_path,
        "save-codex-two.md",
        "a" * 24,
        [_message("assistant", "second", "2026-08-01T00:00:02Z")],
        after_line=2,
    )
    first_raw = next(
        unit
        for unit in RawStore(raw_dir, mode="v2").iter_segment_units()
        if unit.raw_id == "save-codex-one.md"
    )
    assert first_raw.path is not None
    first_raw.path.write_bytes(
        first_raw.path.read_bytes().replace(b"first answer", b"wrong answer")
    )

    original_read = RawStore.read_bytes

    def only_new_raw(self: RawStore, raw: object) -> bytes:
        if getattr(raw, "raw_id", str(raw)) != "save-codex-two.md":
            raise AssertionError("warm path reread old Raw")
        return original_read(self, raw)

    monkeypatch.setattr(RawStore, "read_bytes", only_new_raw)
    catalog.advance(raw_dir, tmp_path, 4096)
    monkeypatch.setattr(RawStore, "read_bytes", original_read)

    with pytest.raises(
        catalog.CatalogError, match="conflict|cannot be decoded|cannot be read"
    ):
        catalog.texts(raw_dir, tmp_path, refs=[first_answer])


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

    with pytest.raises(catalog.CatalogError, match="source interval"):
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
    monkeypatch.setattr(
        catalog,
        "_index_atoms",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("delta sync scanned existing FTS atoms")
        ),
    )
    original_atoms = catalog._catalog_assistant_atoms

    def only_delta_atoms(
        root: Path, *, after_rowid: int | None = None
    ) -> dict[str, dict[str, object]]:
        if after_rowid is None:
            raise AssertionError("delta sync scanned all catalog assistants")
        return original_atoms(root, after_rowid=after_rowid)

    monkeypatch.setattr(catalog, "_catalog_assistant_atoms", only_delta_atoms)
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


def test_catalog_and_historical_index_rebuild_have_the_same_digest(
    tmp_path: Path,
) -> None:
    init_chronovisor(RuntimeContext(tmp_path))
    raw_dir = _capture(
        tmp_path,
        "save-codex-z.md",
        "a" * 24,
        [_message("assistant", "first", "2026-08-01T00:00:00Z")],
    )
    catalog.advance(raw_dir, tmp_path, 4096)
    first = catalog.sync_historical_index(raw_dir, tmp_path)
    _capture(
        tmp_path,
        "save-codex-a.md",
        "b" * 24,
        [_message("assistant", "second", "2026-08-02T00:00:00Z")],
    )
    catalog.advance(raw_dir, tmp_path, 4096)
    advanced = catalog.sync_historical_index(raw_dir, tmp_path)
    index_path = catalog.historical_index_path(tmp_path)
    index_path.unlink()
    catalog._index_checkpoint_path(index_path).unlink()
    catalog.catalog_path(tmp_path).unlink()
    catalog._catalog_checkpoint_path(tmp_path).unlink()
    catalog.advance(raw_dir, tmp_path, 4096)

    assert first != advanced
    assert catalog.sync_historical_index(raw_dir, tmp_path) == advanced


def test_historical_index_digest_is_order_independent_and_incremental() -> None:
    first = {"atom_id": "a" * 64, "catalog_rowid": 2}
    second = {"atom_id": "b" * 64, "catalog_rowid": 1}

    expected = catalog._index_digest({"first": first, "second": second})

    assert catalog._index_digest({"second": second, "first": first}) == expected
    assert (
        catalog._advance_index_digest(catalog._index_digest({"first": first}), [second])
        == expected
    )


def test_catalog_lineage_stays_stable_for_normal_delta_and_candidate_sync(
    tmp_path: Path,
) -> None:
    init_chronovisor(RuntimeContext(tmp_path))
    raw_dir = _capture(
        tmp_path,
        "save-codex-c.md",
        "c" * 24,
        [_message("assistant", "c", "2026-08-01T00:00:00Z")],
    )
    catalog.advance(raw_dir, tmp_path, 4096)
    catalog.sync_historical_index(raw_dir, tmp_path)
    before = catalog._read_catalog_checkpoint(tmp_path)
    assert before is not None
    lineage = before["catalog_lineage"]
    _capture(
        tmp_path,
        "save-codex-d.md",
        "d" * 24,
        [_message("assistant", "d", "2026-08-02T00:00:00Z")],
    )

    assert catalog.advance(raw_dir, tmp_path, 4096).status == "advanced"
    catalog.sync_historical_index(raw_dir, tmp_path)
    current = catalog._read_catalog_checkpoint(tmp_path)
    index = catalog._read_index_checkpoint(catalog.historical_index_path(tmp_path))
    assert current is not None and index is not None
    assert current["catalog_lineage"] == lineage == index["catalog_lineage"]

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
    assert catalog._read_catalog_checkpoint(tmp_path)["catalog_lineage"] == lineage


def test_catalog_checkpoint_loss_rebuilds_lineage_before_fts_delta(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A post-commit checkpoint loss must not reuse the old FTS rowid cursor."""

    init_chronovisor(RuntimeContext(tmp_path))
    raw_dir = tmp_path / "raw"
    for raw_id in (
        "save-codex-c.md",
        "save-codex-d.md",
        "save-codex-e.md",
        "save-codex-f.md",
    ):
        _capture(
            tmp_path,
            raw_id,
            raw_id.removeprefix("save-codex-").removesuffix(".md") * 24,
            [_message("assistant", raw_id, "2026-08-01T00:00:00Z")],
        )
    catalog.advance(raw_dir, tmp_path, 4096)
    catalog.sync_historical_index(raw_dir, tmp_path)
    old_checkpoint = catalog._read_catalog_checkpoint(tmp_path)
    assert old_checkpoint is not None

    _capture(
        tmp_path,
        "save-codex-a.md",
        "a" * 24,
        [_message("assistant", "post-commit tail", "2026-08-02T00:00:00Z")],
    )
    assert catalog.advance(raw_dir, tmp_path, 4096).status == "advanced"
    catalog._catalog_checkpoint_path(tmp_path).unlink()

    assert catalog.advance(raw_dir, tmp_path, 4096).status == "repaired"
    repaired = catalog._read_catalog_checkpoint(tmp_path)
    assert repaired is not None
    assert repaired["catalog_lineage"] != old_checkpoint["catalog_lineage"]
    calls = 0
    inspect = catalog._index_atoms

    def counted(
        *args: object, **kwargs: object
    ) -> tuple[dict[str, dict[str, object]], str]:
        nonlocal calls
        calls += 1
        return inspect(*args, **kwargs)

    monkeypatch.setattr(catalog, "_index_atoms", counted)
    catalog.sync_historical_index(raw_dir, tmp_path)
    assert calls == 1
    with sqlite3.connect(catalog.historical_index_path(tmp_path)) as connection:
        assert connection.execute("SELECT COUNT(*) FROM atoms").fetchone()[0] == 5
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM (SELECT atom_id FROM atoms GROUP BY atom_id HAVING COUNT(*) > 1)"
            ).fetchone()[0]
            == 0
        )
    index = catalog._read_index_checkpoint(catalog.historical_index_path(tmp_path))
    assert index is not None and index["catalog_lineage"] == repaired["catalog_lineage"]


@pytest.mark.parametrize("lineage", ["missing", None, "A" * 64])
def test_catalog_lineage_legacy_migrates_and_malformed_fails_closed(
    tmp_path: Path, lineage: str | None
) -> None:
    init_chronovisor(RuntimeContext(tmp_path))
    raw_dir = _capture(
        tmp_path,
        "save-codex-one.md",
        "a" * 24,
        [_message("assistant", "answer", "2026-08-01T00:00:00Z")],
    )
    catalog.advance(raw_dir, tmp_path, 4096)
    catalog.sync_historical_index(raw_dir, tmp_path)
    checkpoint_path = catalog._catalog_checkpoint_path(tmp_path)
    checkpoint = store.read_sealed(checkpoint_path, schema=store.DISTILLATION_SCHEMA)
    if lineage == "missing":
        checkpoint.pop("catalog_lineage")
    else:
        checkpoint["catalog_lineage"] = lineage
    _rewrite_checkpoint(checkpoint_path, checkpoint)

    if lineage != "missing":
        assert catalog._read_catalog_checkpoint(tmp_path) is None
        with pytest.raises(catalog.CatalogError, match="requires repair"):
            catalog.sync_historical_index(raw_dir, tmp_path)
    else:
        legacy = catalog._read_catalog_checkpoint(tmp_path)
        assert legacy is not None and "catalog_lineage" not in legacy
    assert catalog.advance(raw_dir, tmp_path, 4096).status == "repaired"
    migrated = catalog._read_catalog_checkpoint(tmp_path)
    assert migrated is not None and migrated["catalog_lineage"] != lineage


@pytest.mark.parametrize("lineage", ["missing", None])
def test_missing_or_null_fts_lineage_forces_full_validation_then_migrates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, lineage: str | None
) -> None:
    init_chronovisor(RuntimeContext(tmp_path))
    raw_dir = _capture(
        tmp_path,
        "save-codex-one.md",
        "a" * 24,
        [_message("assistant", "answer", "2026-08-01T00:00:00Z")],
    )
    catalog.advance(raw_dir, tmp_path, 4096)
    catalog.sync_historical_index(raw_dir, tmp_path)
    path = catalog.historical_index_path(tmp_path)
    checkpoint_path = catalog._index_checkpoint_path(path)
    checkpoint = store.read_sealed(checkpoint_path, schema=store.DISTILLATION_SCHEMA)
    if lineage == "missing":
        checkpoint.pop("catalog_lineage")
    else:
        checkpoint["catalog_lineage"] = lineage
    _rewrite_checkpoint(checkpoint_path, checkpoint)
    if lineage is None:
        assert catalog._read_index_checkpoint(path) is None
    calls = 0
    inspect = catalog._index_atoms

    def counted(
        *args: object, **kwargs: object
    ) -> tuple[dict[str, dict[str, object]], str]:
        nonlocal calls
        calls += 1
        return inspect(*args, **kwargs)

    monkeypatch.setattr(catalog, "_index_atoms", counted)
    catalog.sync_historical_index(raw_dir, tmp_path)
    assert calls == 1
    assert (
        catalog._read_index_checkpoint(path)["catalog_lineage"]
        == catalog._read_catalog_checkpoint(tmp_path)["catalog_lineage"]
    )


def test_historical_index_migrates_v1_after_catalog_rowids_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    init_chronovisor(RuntimeContext(tmp_path))
    raw_dir = _capture(
        tmp_path,
        "save-codex-z.md",
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
        "save-codex-m.md",
        "b" * 24,
        [
            _message("user", "second", "2026-08-02T00:00:00Z"),
            _message("assistant", "second answer", "2026-08-02T00:00:01Z"),
        ],
    )
    catalog.advance(raw_dir, tmp_path, 4096)
    catalog.sync_historical_index(raw_dir, tmp_path)
    index_path = catalog.historical_index_path(tmp_path)
    old_atoms = catalog._catalog_assistant_atoms(tmp_path)
    legacy_digest = catalog._legacy_index_digest(old_atoms)
    with sqlite3.connect(index_path) as connection:
        connection.execute(
            "UPDATE metadata SET value=? WHERE key='content_sha256'",
            (legacy_digest,),
        )
    catalog_checkpoint = catalog._read_catalog_checkpoint(tmp_path)
    assert catalog_checkpoint is not None
    catalog._write_index_checkpoint(
        index_path,
        catalog_checkpoint,
        legacy_digest,
        len(old_atoms),
        content_digest_schema=catalog.LEGACY_HISTORICAL_INDEX_DIGEST_SCHEMA,
    )

    _capture(
        tmp_path,
        "save-codex-a.md",
        "a" * 24,
        [_message("assistant", "tail answer", "2026-08-01T00:00:02Z")],
        after_line=2,
    )
    catalog.catalog_path(tmp_path).unlink()
    catalog._catalog_checkpoint_path(tmp_path).unlink()
    catalog.advance(raw_dir, tmp_path, 4096)
    rebuilt_atoms = catalog._catalog_assistant_atoms(tmp_path)
    assert legacy_digest != catalog._legacy_index_digest(
        {atom_id: rebuilt_atoms[atom_id] for atom_id in old_atoms}
    )

    original_read = RawStore.iter_segment_bytes
    reads: list[str] = []

    def record_read(
        self: RawStore, raw_ids: Iterable[str] | None = None
    ) -> Iterator[tuple[object, bytes]]:
        for unit, raw in original_read(self, raw_ids):
            reads.append(unit.raw_id)
            yield unit, raw

    monkeypatch.setattr(RawStore, "iter_segment_bytes", record_read)
    digest = catalog.sync_historical_index(raw_dir, tmp_path)

    assert reads == ["save-codex-a.md"]
    with sqlite3.connect(index_path) as connection:
        assert (
            dict(connection.execute("SELECT key,value FROM metadata"))["content_sha256"]
            == digest
        )
    checkpoint = catalog._read_index_checkpoint(index_path)
    assert checkpoint is not None
    assert checkpoint["content_digest_schema"] == catalog.HISTORICAL_INDEX_DIGEST_SCHEMA
    assert checkpoint["atom_count"] == len(rebuilt_atoms)


def test_index_checkpoint_loss_inserts_pending_atoms(tmp_path: Path) -> None:
    init_chronovisor(RuntimeContext(tmp_path))
    raw_dir = _capture(
        tmp_path,
        "save-codex-one.md",
        "a" * 24,
        [_message("assistant", "first", "2026-08-01T00:00:00Z")],
    )
    catalog.advance(raw_dir, tmp_path, 4096)
    catalog.sync_historical_index(raw_dir, tmp_path)
    _capture(
        tmp_path,
        "save-codex-two.md",
        "b" * 24,
        [_message("assistant", "checkpoint recovery", "2026-08-02T00:00:00Z")],
    )
    catalog.advance(raw_dir, tmp_path, 4096)
    index_path = catalog.historical_index_path(tmp_path)
    catalog._index_checkpoint_path(index_path).unlink()

    catalog.sync_historical_index(raw_dir, tmp_path)
    found = store.search_historical_index(
        index_path,
        query="checkpoint recovery",
        as_of_us=9_999_999_999_999_999,
        host="other",
        session_cluster_id="other",
        source_index=0,
        limit=10,
    )
    assert found


def test_legacy_index_checkpoint_rejects_non_hex_digest(tmp_path: Path) -> None:
    init_chronovisor(RuntimeContext(tmp_path))
    raw_dir = _capture(
        tmp_path,
        "save-codex-one.md",
        "a" * 24,
        [_message("assistant", "answer", "2026-08-01T00:00:00Z")],
    )
    catalog.advance(raw_dir, tmp_path, 4096)
    catalog.sync_historical_index(raw_dir, tmp_path)
    index_path = catalog.historical_index_path(tmp_path)
    invalid_digest = "g" * 64
    with sqlite3.connect(index_path) as connection:
        connection.execute(
            "UPDATE metadata SET value=? WHERE key='content_sha256'",
            (invalid_digest,),
        )
    catalog_checkpoint = catalog._read_catalog_checkpoint(tmp_path)
    assert catalog_checkpoint is not None
    catalog._write_index_checkpoint(
        index_path,
        catalog_checkpoint,
        invalid_digest,
        1,
        content_digest_schema=catalog.LEGACY_HISTORICAL_INDEX_DIGEST_SCHEMA,
    )

    assert (
        catalog._read_index_checkpoint(
            index_path,
            content_digest_schema=catalog.LEGACY_HISTORICAL_INDEX_DIGEST_SCHEMA,
        )
        is None
    )
    with pytest.raises(catalog.CatalogError, match="content digest conflicts"):
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
    expected = catalog._catalog_assistant_atoms(tmp_path)
    with sqlite3.connect(index_path) as connection:
        indexed, _verified = catalog._index_atoms(connection, expected)
        bootstrap_digest = canonical_json_sha256_strict(
            sorted(indexed.values(), key=lambda atom: str(atom["atom_id"]))
        )
        connection.execute(
            "UPDATE metadata SET value=? WHERE key='content_sha256'",
            (bootstrap_digest,),
        )
    catalog._index_checkpoint_path(index_path).write_text("{}", encoding="utf-8")
    inspect = catalog._index_atoms
    calls = 0

    def tracked_inspect(
        connection: sqlite3.Connection, expected: object, **kwargs: object
    ) -> tuple[dict[str, dict[str, object]], str]:
        nonlocal calls
        calls += 1
        return inspect(connection, expected, **kwargs)

    monkeypatch.setattr(catalog, "_index_atoms", tracked_inspect)
    assert catalog.sync_historical_index(raw_dir, tmp_path) == digest
    assert calls == 1
    with sqlite3.connect(index_path) as connection:
        assert (
            dict(connection.execute("SELECT key,value FROM metadata"))["content_sha256"]
            == digest
        )

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


def test_candidate_sync_keeps_catalog_checkpoint_warm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    init_chronovisor(RuntimeContext(tmp_path))
    raw_dir = _capture(
        tmp_path,
        "save-codex-one.md",
        "a" * 24,
        [
            _message("user", "question", "2026-08-01T00:00:00Z"),
            _message("assistant", "answer", "2026-08-01T00:00:01Z"),
        ],
    )
    catalog.advance(raw_dir, tmp_path, 4096)
    answer = catalog.rallies(tmp_path)[0]["actual_answer_refs"][0]
    cache = catalog.CatalogTextCache(raw_dir, tmp_path)
    assert cache[answer["semantic_sha256"]] == "answer"
    ledger = store.distillation_dir(tmp_path) / "candidate-ledger.jsonl"
    store.append_chain(
        ledger,
        {
            "kind": "candidate-snapshot",
            "rally_id": "rally-1",
            "snapshot": _candidate_snapshot("rally-1", ["c1"]),
        },
    )
    assert catalog.sync_candidate_index(tmp_path, ledger)["status"] == "bootstrap"
    assert cache[answer["semantic_sha256"]] == "answer"
    checkpoint = catalog._read_catalog_checkpoint(tmp_path)
    assert checkpoint is not None

    from chronovisor.core.raw_store import RawStore

    monkeypatch.setattr(
        RawStore,
        "read_bytes",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("candidate sync invalidated warm catalog")
        ),
    )
    assert catalog.advance(raw_dir, tmp_path, 4096).status == "noop"
    assert catalog._read_catalog_checkpoint(tmp_path) is not None
    store.append_chain(
        ledger,
        {
            "kind": "candidate-snapshot",
            "rally_id": "rally-2",
            "snapshot": _candidate_snapshot("rally-2", ["c2"]),
        },
    )
    assert catalog.sync_candidate_index(tmp_path, ledger)["status"] == "advanced"
    assert cache[answer["semantic_sha256"]] == "answer"
    assert catalog.advance(raw_dir, tmp_path, 4096).status == "noop"


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
