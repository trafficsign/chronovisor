"""Incremental, metadata-only catalog for Recall distillation Raw v2."""

from __future__ import annotations

import hashlib
import json
import secrets
import sqlite3
import stat
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from chronovisor.core.canonical_json import (
    canonical_json_bytes_strict,
    canonical_json_sha256_strict,
)
from chronovisor.core.raw_segment import RawSegmentCorrupt
from chronovisor.core.raw_store import (
    RawStore,
    RawUnit,
    committed_event_spans,
    committed_raw_watermark,
)
from chronovisor.recall import recall_distillation as distill
from chronovisor.recall import recall_distillation_store as store

CATALOG_SCHEMA = "chronovisor.recall-distillation-catalog.v1"
HISTORICAL_INDEX_SCHEMA = "chronovisor.recall-historical-fts.v1"
HISTORICAL_INDEX_CHECKPOINT_KIND = "historical-index-checkpoint"
LEGACY_HISTORICAL_INDEX_DIGEST_SCHEMA = "historical-fts-chain-v1"
HISTORICAL_INDEX_DIGEST_SCHEMA = "historical-fts-multiset-v2"
CANDIDATE_SNAPSHOT_SCHEMA = "chronovisor.recall-candidate-snapshot.v1"
CANDIDATE_INDEX_SCHEMA = "chronovisor.recall-candidate-offset-index.v1"


class CatalogError(ValueError):
    """The derived catalog cannot safely represent the committed Raw inventory."""


class _CatalogRepairRequired(CatalogError):
    """The text-backed session rebuild is needed to repair a missing tail."""


@dataclass(frozen=True)
class CatalogAdvance:
    status: str
    watermark: str
    indexed_raw_ids: tuple[str, ...]
    archived_raw_ids: tuple[str, ...]
    rally_ids: tuple[str, ...]
    deferred_session_keys: tuple[tuple[str, str], ...]


def catalog_path(root: Path) -> Path:
    """Keep the additive catalog separate from the replace-on-build FTS cache."""

    return store.distillation_dir(root) / "historical-catalog.sqlite"


def historical_index_path(root: Path) -> Path:
    return store.distillation_dir(root) / "historical-index.sqlite"


def _index_checkpoint_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".checkpoint.json")


def _catalog_checkpoint_path(root: Path) -> Path:
    return _index_checkpoint_path(catalog_path(root))


def _candidate_catalog_state_path(root: Path) -> Path:
    return catalog_path(root).with_suffix(".sqlite.candidate-state.json")


def _ensure_private_file(path: Path) -> None:
    """Set private permissions only when needed; chmod mutates ctime."""

    try:
        if stat.S_IMODE(path.stat().st_mode) != 0o600:
            path.chmod(0o600)
    except OSError as exc:
        raise CatalogError("historical catalog is unavailable") from exc


def _index_file_state(path: Path) -> dict[str, int] | None:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise CatalogError("historical index is unreadable") from exc
    return {
        "size_bytes": stat.st_size,
        "st_dev": stat.st_dev,
        "st_ino": stat.st_ino,
        "st_mtime_ns": stat.st_mtime_ns,
        "st_ctime_ns": stat.st_ctime_ns,
    }


def _new_catalog_lineage() -> str:
    return secrets.token_hex(32)


def _catalog_lineage(checkpoint: Mapping[str, Any]) -> str | None:
    """Return a valid lineage, keeping sealed pre-lineage checkpoints readable."""

    lineage = checkpoint.get("catalog_lineage")
    if lineage is None:
        return None
    if (
        not isinstance(lineage, str)
        or len(lineage) != 64
        or any(character not in "0123456789abcdef" for character in lineage)
    ):
        return None
    return lineage


def _write_catalog_checkpoint(
    root: Path, watermark: str, event_rowid: int, *, catalog_lineage: str
) -> None:
    if _catalog_lineage({"catalog_lineage": catalog_lineage}) is None:
        raise CatalogError("catalog lineage is invalid")
    store.write_sealed_state(
        _catalog_checkpoint_path(root),
        {
            "kind": "historical-catalog-checkpoint",
            "catalog_name": catalog_path(root).name,
            "catalog_schema": CATALOG_SCHEMA,
            "catalog_watermark": watermark,
            "event_rowid": event_rowid,
            "catalog_lineage": catalog_lineage,
            "file_state": _index_file_state(catalog_path(root)),
        },
    )


def _read_catalog_checkpoint(root: Path) -> dict[str, Any] | None:
    try:
        checkpoint = store.read_sealed(
            _catalog_checkpoint_path(root), schema=store.DISTILLATION_SCHEMA
        )
    except store.DistillationStoreError:
        return None
    if (
        checkpoint.get("kind") != "historical-catalog-checkpoint"
        or checkpoint.get("catalog_name") != catalog_path(root).name
        or checkpoint.get("catalog_schema") != CATALOG_SCHEMA
        or not isinstance(checkpoint.get("catalog_watermark"), str)
        or not isinstance(checkpoint.get("event_rowid"), int)
        or isinstance(checkpoint.get("event_rowid"), bool)
        or checkpoint["event_rowid"] < 0
        or checkpoint.get("file_state") != _index_file_state(catalog_path(root))
        or (
            "catalog_lineage" in checkpoint and _catalog_lineage(checkpoint) is None
        )
    ):
        return None
    return checkpoint


def _write_candidate_catalog_state(
    root: Path, checkpoint: Mapping[str, Any]
) -> None:
    store.write_sealed_state(
        _candidate_catalog_state_path(root),
        {
            "kind": "candidate-catalog-state",
            "catalog_watermark": checkpoint["catalog_watermark"],
            "event_rowid": checkpoint["event_rowid"],
            "catalog_file_state": _index_file_state(catalog_path(root)),
        },
    )


def _read_candidate_catalog_state(root: Path) -> dict[str, Any] | None:
    try:
        state = store.read_sealed(
            _candidate_catalog_state_path(root), schema=store.DISTILLATION_SCHEMA
        )
    except store.DistillationStoreError:
        return None
    if (
        state.get("kind") != "candidate-catalog-state"
        or not isinstance(state.get("catalog_watermark"), str)
        or not isinstance(state.get("event_rowid"), int)
        or state.get("catalog_file_state") != _index_file_state(catalog_path(root))
    ):
        return None
    return state


def _write_index_checkpoint(
    path: Path,
    catalog_checkpoint: Mapping[str, Any],
    digest: str,
    count: int,
    *,
    content_digest_schema: str = HISTORICAL_INDEX_DIGEST_SCHEMA,
) -> None:
    store.write_sealed_state(
        _index_checkpoint_path(path),
        {
            "kind": HISTORICAL_INDEX_CHECKPOINT_KIND,
            "index_name": path.name,
            "historical_index_schema": HISTORICAL_INDEX_SCHEMA,
            "content_digest_schema": content_digest_schema,
            "catalog_watermark": catalog_checkpoint["catalog_watermark"],
            "catalog_event_rowid": catalog_checkpoint["event_rowid"],
            "catalog_lineage": catalog_checkpoint["catalog_lineage"],
            "catalog_file_state": catalog_checkpoint["file_state"],
            "content_sha256": digest,
            "atom_count": count,
            "file_state": _index_file_state(path),
        },
    )


def _read_index_checkpoint(
    path: Path,
    *,
    content_digest_schema: str = HISTORICAL_INDEX_DIGEST_SCHEMA,
) -> dict[str, Any] | None:
    try:
        checkpoint = store.read_sealed(
            _index_checkpoint_path(path), schema=store.DISTILLATION_SCHEMA
        )
    except store.DistillationStoreError:
        return None
    if (
        checkpoint.get("kind") != HISTORICAL_INDEX_CHECKPOINT_KIND
        or checkpoint.get("index_name") != path.name
        or checkpoint.get("historical_index_schema") != HISTORICAL_INDEX_SCHEMA
        or checkpoint.get("content_digest_schema") != content_digest_schema
        or not isinstance(checkpoint.get("catalog_watermark"), str)
        or not isinstance(checkpoint.get("catalog_event_rowid"), int)
        or isinstance(checkpoint.get("catalog_event_rowid"), bool)
        or checkpoint["catalog_event_rowid"] < 0
        or not isinstance(checkpoint.get("content_sha256"), str)
        or len(checkpoint["content_sha256"]) != 64
        or any(
            character not in "0123456789abcdef"
            for character in checkpoint["content_sha256"]
        )
        or not isinstance(checkpoint.get("atom_count"), int)
        or isinstance(checkpoint.get("atom_count"), bool)
        or checkpoint["atom_count"] < 0
        or not isinstance(checkpoint.get("catalog_file_state"), dict)
        or (
            "catalog_lineage" in checkpoint and _catalog_lineage(checkpoint) is None
        )
        or checkpoint.get("file_state") != _index_file_state(path)
    ):
        return None
    return checkpoint


def _connect(root: Path) -> sqlite3.Connection:
    path = catalog_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise CatalogError("historical catalog path is unsafe")
    try:
        connection = sqlite3.connect(path)
    except sqlite3.Error as exc:
        raise CatalogError("historical catalog is unavailable") from exc
    try:
        _ensure_private_file(path)
    except CatalogError:
        connection.close()
        raise
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA journal_mode=DELETE")
    connection.execute("PRAGMA synchronous=FULL")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS metadata(
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS raw_units(
            raw_id TEXT PRIMARY KEY,
            raw_sha256 TEXT NOT NULL,
            receipt_sha256 TEXT NOT NULL,
            host TEXT NOT NULL,
            session_key TEXT NOT NULL,
            captured_at TEXT NOT NULL,
            record_count INTEGER NOT NULL,
            status TEXT NOT NULL,
            source_after_line INTEGER NOT NULL,
            source_until_line INTEGER NOT NULL,
            CHECK(source_after_line >= 0),
            CHECK(source_until_line >= source_after_line)
        );
        CREATE TABLE IF NOT EXISTS events(
            raw_id TEXT NOT NULL,
            event_index INTEGER NOT NULL,
            raw_sha256 TEXT NOT NULL,
            receipt_sha256 TEXT NOT NULL,
            host TEXT NOT NULL,
            session_key TEXT NOT NULL,
            session_cluster_id TEXT NOT NULL,
            session_id_sha256 TEXT NOT NULL,
            source_index INTEGER NOT NULL,
            byte_start INTEGER NOT NULL,
            byte_end INTEGER NOT NULL,
            timestamp TEXT NOT NULL,
            timestamp_us INTEGER NOT NULL,
            role TEXT NOT NULL,
            semantic_sha256 TEXT NOT NULL,
            text_bytes INTEGER NOT NULL,
            nonempty INTEGER NOT NULL,
            prompt_hash TEXT,
            structural_json TEXT NOT NULL,
            PRIMARY KEY(raw_id, event_index),
            FOREIGN KEY(raw_id) REFERENCES raw_units(raw_id)
        );
        CREATE INDEX IF NOT EXISTS events_text_sha256 ON events(semantic_sha256);
        CREATE INDEX IF NOT EXISTS events_session ON events(host, session_key, source_index);
        CREATE UNIQUE INDEX IF NOT EXISTS events_source_position
            ON events(host, session_key, source_index);
        CREATE TABLE IF NOT EXISTS rallies(
            rally_id TEXT PRIMARY KEY,
            as_of_us INTEGER NOT NULL,
            row_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS candidate_records(
            record_index INTEGER PRIMARY KEY,
            rally_id TEXT NOT NULL UNIQUE,
            previous_sha256 TEXT NOT NULL,
            record_sha256 TEXT NOT NULL UNIQUE,
            snapshot_sha256 TEXT NOT NULL,
            offset INTEGER NOT NULL CHECK(offset >= 0),
            length INTEGER NOT NULL CHECK(length > 0),
            index_sha256 TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS candidate_records_rally
            ON candidate_records(rally_id);
        CREATE TABLE IF NOT EXISTS candidate_index_state(
            singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
            index_schema TEXT NOT NULL,
            ledger_path TEXT NOT NULL,
            ledger_exists INTEGER NOT NULL CHECK(ledger_exists IN (0,1)),
            ledger_size INTEGER NOT NULL CHECK(ledger_size >= 0),
            ledger_dev INTEGER NOT NULL,
            ledger_ino INTEGER NOT NULL,
            ledger_mtime_ns INTEGER NOT NULL,
            ledger_ctime_ns INTEGER NOT NULL,
            record_count INTEGER NOT NULL CHECK(record_count >= 0),
            head_sha256 TEXT NOT NULL
        );
        """
    )
    return connection


def _unit_identity(unit: RawUnit) -> tuple[str, str]:
    commit = unit.commit
    if commit is None or unit.sha256 is None or unit.captured_at is None:
        raise CatalogError("Raw v2 unit has no committed receipt")
    return unit.sha256, canonical_json_sha256_strict(commit.to_dict())


def _event_row(
    unit: RawUnit, event: Mapping[str, Any], *, start: int, end: int, index: int
) -> dict[str, Any] | None:
    commit = unit.commit
    raw_sha256, receipt_sha256 = _unit_identity(unit)
    assert commit is not None
    role, text = distill._event_semantics(commit.host, event)
    if role not in {"user", "assistant", "tool"}:
        return None
    timestamp, timestamp_us = distill._timestamp(
        event.get("timestamp"), commit.captured_at
    )
    return {
        "raw_id": unit.raw_id,
        "event_index": index,
        "raw_sha256": raw_sha256,
        "receipt_sha256": receipt_sha256,
        "host": commit.host,
        "session_key": commit.session_key,
        "session_cluster_id": hashlib.sha256(
            f"{commit.host}\0{commit.session_key}".encode()
        ).hexdigest(),
        "session_id_sha256": hashlib.sha256(
            str(commit.session_id or commit.session_key).encode()
        ).hexdigest(),
        "source_index": commit.after_line + index + 1,
        "byte_start": start,
        "byte_end": end,
        "timestamp": timestamp,
        "timestamp_us": timestamp_us,
        "role": role,
        "text": text,
        "semantic_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "text_bytes": len(text.encode("utf-8")),
        "nonempty": bool(text.strip()),
        "prompt_hash": distill._prompt_hash(text) if role == "user" else None,
        "structural": distill._structural_tokens(event),
    }


def _read_unit_events(
    raw_store: RawStore, unit: RawUnit
) -> tuple[str, list[dict[str, Any]]]:
    try:
        raw = raw_store.read_bytes(unit)
        if raw_store.is_archived_legacy_markdown(unit, raw):
            return "archived", []
        commit = unit.commit
        if commit is None:
            raise CatalogError("Raw v2 unit has no committed receipt")
        spans = committed_event_spans(raw, commit.record_count)
    except (RawSegmentCorrupt, UnicodeError, OSError, distill.DistillationError) as exc:
        raise CatalogError("committed Raw event stream is invalid") from exc
    rows: list[dict[str, Any]] = []
    for index, (start, encoded) in enumerate(spans):
        try:
            event = json.loads(encoded)
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise CatalogError("committed Raw event is invalid") from exc
        if not isinstance(event, dict):
            raise CatalogError("committed Raw event is not an object")
        row = _event_row(
            unit, event, start=start, end=start + len(encoded), index=index
        )
        if row is not None:
            rows.append(row)
    return "indexed", rows


def _store_unit(connection: sqlite3.Connection, unit: RawUnit, *, status: str) -> None:
    raw_sha256, receipt_sha256 = _unit_identity(unit)
    assert unit.commit is not None and unit.captured_at is not None
    connection.execute(
        """INSERT INTO raw_units VALUES(?,?,?,?,?,?,?,?,?,?)""",
        (
            unit.raw_id,
            raw_sha256,
            receipt_sha256,
            unit.commit.host,
            unit.commit.session_key,
            unit.captured_at,
            unit.commit.record_count,
            status,
            unit.commit.after_line,
            unit.commit.until_line,
        ),
    )


def _validate_new_unit_intervals(
    connection: sqlite3.Connection, units: Iterable[RawUnit]
) -> None:
    """Require only newly committed intervals to append within each session.

    Historical committed inventories may intentionally omit source ranges.  A
    new interval may follow that history, but it may never overlap or precede
    an already cataloged Raw interval, including a Raw containing only unknown
    events.
    """

    try:
        columns = {
            str(row[1]) for row in connection.execute("PRAGMA table_info(raw_units)")
        }
    except sqlite3.DatabaseError as exc:
        raise CatalogError("historical catalog is unreadable") from exc
    if {"source_after_line", "source_until_line"} - columns:
        raise CatalogError("catalog source range migration requires repair")

    prior_until: dict[tuple[str, str], int] = {}
    for unit in sorted(
        units,
        key=lambda value: (
            value.commit.host if value.commit is not None else "",
            value.commit.session_key if value.commit is not None else "",
            value.commit.after_line if value.commit is not None else -1,
            value.raw_id,
        ),
    ):
        assert unit.commit is not None
        commit = unit.commit
        after, until = int(commit.after_line), int(commit.until_line)
        if after < 0 or until < after:
            raise CatalogError("committed Raw source interval is invalid")
        key = (commit.host, commit.session_key)
        if key not in prior_until:
            row = connection.execute(
                """SELECT MAX(source_until_line) FROM raw_units
                   WHERE host=? AND session_key=?""",
                key,
            ).fetchone()
            if row is not None and row[0] is not None:
                prior_until[key] = int(row[0])
        if after < prior_until.get(key, after):
            raise CatalogError("catalog session source interval is not append-only")
        prior_until[key] = max(prior_until.get(key, after), until)


def _reset_catalog_for_repair(connection: sqlite3.Connection) -> None:
    """Make an old or unsealed derived catalog safe for one full Raw rebuild."""

    columns = {str(row[1]) for row in connection.execute("PRAGMA table_info(raw_units)")}
    for name in ("source_after_line", "source_until_line"):
        if name not in columns:
            connection.execute(f"ALTER TABLE raw_units ADD COLUMN {name} INTEGER")
    for table in (
        "candidate_index_state",
        "candidate_records",
        "rallies",
        "events",
        "raw_units",
        "metadata",
    ):
        connection.execute(f"DELETE FROM {table}")


def _store_events(
    connection: sqlite3.Connection, rows: Iterable[Mapping[str, Any]]
) -> None:
    connection.executemany(
        """INSERT INTO events VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            (
                row["raw_id"],
                row["event_index"],
                row["raw_sha256"],
                row["receipt_sha256"],
                row["host"],
                row["session_key"],
                row["session_cluster_id"],
                row["session_id_sha256"],
                row["source_index"],
                row["byte_start"],
                row["byte_end"],
                row["timestamp"],
                row["timestamp_us"],
                row["role"],
                row["semantic_sha256"],
                row["text_bytes"],
                int(bool(row["nonempty"])),
                row["prompt_hash"],
                json.dumps(row["structural"], sort_keys=True, separators=(",", ":")),
            )
            for row in rows
        ),
    )


def _store_rallies(
    connection: sqlite3.Connection, rows: Iterable[Mapping[str, Any]]
) -> tuple[str, ...]:
    values = sorted(rows, key=lambda row: str(row["rally_id"]))
    encoded: dict[str, tuple[int, str]] = {}
    for row in values:
        rally_id = str(row["rally_id"])
        value = (
            int(row["as_of_us"]),
            json.dumps(row, sort_keys=True, separators=(",", ":")),
        )
        prior = encoded.get(rally_id)
        if prior is not None and prior != value:
            raise CatalogError("rally duplicate conflicts within delta")
        encoded[rally_id] = value
    for rally_id, (as_of_us, row_json) in encoded.items():
        connection.execute(
            """
            INSERT INTO rallies(rally_id,as_of_us,row_json) VALUES(?,?,?)
            ON CONFLICT(rally_id) DO UPDATE SET
                as_of_us=excluded.as_of_us,
                row_json=excluded.row_json
            """,
            (rally_id, as_of_us, row_json),
        )
    return tuple(encoded)


def _bootstrap(
    raw_dir: Path, root: Path, max_context_bytes: int, units: list[RawUnit]
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]], set[str]]:
    try:
        event_rows = distill._events(raw_dir)
        rallies = distill.extract_rallies(
            raw_dir,
            root=root,
            max_context_bytes=max_context_bytes,
            _event_rows=event_rows,
        )
    except (distill.DistillationError, RawSegmentCorrupt) as exc:
        raise CatalogError("cannot bootstrap committed Raw catalog") from exc
    by_raw: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in event_rows:
        by_raw[str(row["raw_id"])].append(_catalog_event(row))
    archived: set[str] = set()
    raw_store = RawStore(raw_dir, mode="v2")
    for unit in units:
        if unit.raw_id in by_raw:
            continue
        status, _ = _read_unit_events(raw_store, unit)
        if status == "archived":
            archived.add(unit.raw_id)
    return by_raw, rallies, archived


def _catalog_event(row: Mapping[str, Any]) -> dict[str, Any]:
    text = str(row["text"])
    return {
        **{
            key: row[key]
            for key in (
                "raw_id",
                "raw_sha256",
                "receipt_sha256",
                "event_index",
                "host",
                "session_key",
                "session_cluster_id",
                "session_id_sha256",
                "source_index",
                "byte_start",
                "byte_end",
                "timestamp",
                "timestamp_us",
                "role",
                "semantic_sha256",
                "structural",
            )
        },
        "text_bytes": len(text.encode("utf-8")),
        "nonempty": bool(text.strip()),
        "prompt_hash": distill._prompt_hash(text) if row["role"] == "user" else None,
    }


def _resolve_rows(raw_dir: Path, rows: Iterable[Mapping[str, Any]]) -> dict[str, str]:
    """Resolve only the verified Raw units referenced by ``rows``."""

    raw_store = RawStore(raw_dir, mode="v2")
    resolved: dict[str, str] = {}
    by_raw: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_raw[str(row["raw_id"])].append(row)
    seen: set[str] = set()
    try:
        for unit, raw in raw_store.iter_segment_bytes(by_raw):
            raw_id = unit.raw_id
            seen.add(raw_id)
            event_rows = by_raw[raw_id]
            raw_sha256, receipt_sha256 = _unit_identity(unit)
            if any(
                row["raw_sha256"] != raw_sha256
                or row["receipt_sha256"] != receipt_sha256
                for row in event_rows
            ):
                raise CatalogError("catalog Raw digest conflicts with source")
            assert unit.commit is not None
            spans = committed_event_spans(raw, unit.commit.record_count)
            for row in event_rows:
                index = int(row["event_index"])
                if index >= len(spans):
                    raise CatalogError("catalog event index is invalid")
                start, encoded = spans[index]
                if (
                    start != row["byte_start"]
                    or start + len(encoded) != row["byte_end"]
                ):
                    raise CatalogError("catalog event range conflicts with source")
                try:
                    event = json.loads(encoded)
                    role, text = distill._event_semantics(unit.commit.host, event)
                except (
                    UnicodeError,
                    json.JSONDecodeError,
                    distill.DistillationError,
                ) as exc:
                    raise CatalogError("catalog event cannot be decoded") from exc
                if (
                    role != row["role"]
                    or hashlib.sha256(text.encode()).hexdigest()
                    != row["semantic_sha256"]
                ):
                    raise CatalogError("catalog event semantics conflict with source")
                resolved[str(row["semantic_sha256"])] = text
    except (RawSegmentCorrupt, OSError) as exc:
        raise CatalogError("catalog Raw cannot be read") from exc
    if seen != set(by_raw):
        raise CatalogError("catalog Raw is unavailable")
    return resolved


def _session_events(
    connection: sqlite3.Connection,
    raw_dir: Path,
    host: str,
    session_key: str,
) -> list[dict[str, Any]]:
    rows = list(
        connection.execute(
            """
            SELECT * FROM events
            WHERE host=? AND session_key=?
            ORDER BY source_index,raw_id,event_index
            """,
            (host, session_key),
        )
    )
    text_by_hash = _resolve_rows(raw_dir, rows)
    events: list[dict[str, Any]] = []
    for row in rows:
        event = dict(row)
        event["text"] = text_by_hash.get(str(row["semantic_sha256"]), "")
        try:
            event["structural"] = json.loads(str(row["structural_json"]))
        except json.JSONDecodeError as exc:
            raise CatalogError("catalog structural tokens are invalid") from exc
        events.append(event)
    return events


def _catalog_session_events(
    connection: sqlite3.Connection,
    host: str,
    session_key: str,
) -> list[dict[str, Any]]:
    """Read one session's event metadata without resolving its Raw text."""

    try:
        rows = list(
            connection.execute(
                """
                SELECT * FROM events
                WHERE host=? AND session_key=?
                ORDER BY source_index,raw_id,event_index
                """,
                (host, session_key),
            )
        )
    except sqlite3.DatabaseError as exc:
        raise CatalogError("historical catalog is unreadable") from exc
    events: list[dict[str, Any]] = []
    for row in rows:
        event = dict(row)
        try:
            event["structural"] = json.loads(str(event.pop("structural_json")))
        except (TypeError, json.JSONDecodeError) as exc:
            raise CatalogError("catalog structural tokens are invalid") from exc
        try:
            for key in (
                "event_index",
                "source_index",
                "byte_start",
                "byte_end",
                "timestamp_us",
                "text_bytes",
            ):
                event[key] = int(event[key])
        except (KeyError, TypeError, ValueError) as exc:
            raise CatalogError("catalog event metadata is invalid") from exc
        if (
            event["event_index"] < 0
            or event["source_index"] < 0
            or event["byte_start"] < 0
            or event["byte_end"] < event["byte_start"]
            or event["text_bytes"] < 0
        ):
            raise CatalogError("catalog event metadata is invalid")
        event["nonempty"] = bool(event["nonempty"])
        events.append(event)
    return events


def _catalog_event_ref(row: Mapping[str, Any]) -> dict[str, Any]:
    """Build a text-free event reference from catalog metadata."""

    return distill._event_ref(row)


def _rally_id_for_query(row: Mapping[str, Any]) -> str:
    return canonical_json_sha256_strict(
        {
            "schema": distill.RALLY_SCHEMA,
            "host": row["host"],
            "session_key": row["session_key"],
            "raw_id": row["raw_id"],
            "event_index": row["event_index"],
            "raw_sha256": row["raw_sha256"],
        }
    )


def _materialize_catalog_rally(
    query: Mapping[str, Any],
    prefix: list[Mapping[str, Any]],
    answer_rows: list[Mapping[str, Any]],
    tool_rows: list[Mapping[str, Any]],
    *,
    max_context_bytes: int,
    exposure: Mapping[tuple[str, str, str], list[dict[str, Any]]],
) -> dict[str, Any]:
    """Materialize one rally from metadata and the changed tail only."""

    full_refs = [_catalog_event_ref(row) for row in prefix]
    selected: list[dict[str, Any]] = []
    context_suffix_bytes = 0
    for row in reversed(prefix):
        text_bytes = int(row["text_bytes"])
        if context_suffix_bytes + text_bytes > max_context_bytes:
            break
        selected.append(_catalog_event_ref(row))
        context_suffix_bytes += text_bytes
    selected.reverse()

    answer_refs = [_catalog_event_ref(row) for row in answer_rows]
    tool_refs = [_catalog_event_ref(row) for row in tool_rows]
    as_of_us = int(query["timestamp_us"])
    query_sha256 = str(query["semantic_sha256"])
    possible_receipts = exposure.get(
        (
            str(query["host"]),
            str(query["session_id_sha256"]),
            query_sha256,
        ),
        [],
    )
    answer_end_us = max(
        (int(ref["timestamp_us"]) for ref in answer_refs), default=-1
    )
    receipt_rows = [
        row
        for row in possible_receipts
        if as_of_us
        <= distill._timestamp(row["observed_at"], row["observed_at"])[1]
        <= answer_end_us
    ]
    exposure_ambiguous = len(receipt_rows) > 1
    if len(receipt_rows) != 1:
        receipt_rows = []
    has_answer = bool(answer_refs)
    has_exposure = bool(receipt_rows)
    rally_id = _rally_id_for_query(query)
    return {
        "host": query["host"],
        "session_cluster_id": query["session_cluster_id"],
        "session_id_sha256": query["session_id_sha256"],
        "as_of": query["timestamp"],
        "as_of_us": as_of_us,
        "source_index": query["source_index"],
        "query_sha256": query_sha256,
        "prompt_hash": query["prompt_hash"],
        "context_suffix_bytes": context_suffix_bytes,
        "full_context": {
            "event_count": len(full_refs),
            "refs_sha256": canonical_json_sha256_strict(full_refs),
            "first_ref": full_refs[0] if full_refs else None,
            "last_ref": full_refs[-1] if full_refs else None,
        },
        "schema": distill.RALLY_SCHEMA,
        "boundary_revision": "rally-v1",
        "rally_id": rally_id,
        "query_ref": _catalog_event_ref(query),
        "context_refs": selected,
        "actual_answer_refs": answer_refs,
        "tool_refs": tool_refs,
        "exposure_receipts": receipt_rows,
        "eligibility": {
            "relevance": True,
            "answer_utility": has_answer and has_exposure,
            "reason": (
                "eligible"
                if has_answer and has_exposure
                else "missing_answer"
                if not has_answer
                else "ambiguous_exact_exposure"
                if exposure_ambiguous
                else "missing_exact_exposure"
            ),
        },
    }


def _incremental_session_rallies(
    connection: sqlite3.Connection,
    root: Path,
    host: str,
    session_key: str,
    new_rows: list[Mapping[str, Any]],
    max_context_bytes: int,
) -> list[dict[str, Any]]:
    """Project only the changed rally tail using catalog metadata.

    Existing event text is deliberately not resolved here.  The catalog keeps
    enough byte/role/reference metadata to append answers and form the next
    query context without rereading committed Raw prefixes.
    """

    all_rows = _catalog_session_events(connection, host, session_key)
    new_keys = {
        (str(row["raw_id"]), int(row["event_index"])) for row in new_rows
    }
    by_key = {
        (str(row["raw_id"]), int(row["event_index"])): row for row in all_rows
    }
    if not new_keys <= set(by_key):
        raise CatalogError("catalog session delta is incomplete")
    existing_rows = [
        row
        for row in all_rows
        if (str(row["raw_id"]), int(row["event_index"])) not in new_keys
    ]
    if existing_rows and new_rows:
        old_max = max(int(row["source_index"]) for row in existing_rows)
        new_min = min(int(row["source_index"]) for row in new_rows)
        if new_min <= old_max:
            raise CatalogError("catalog session source interval is not append-only")

    ordered = sorted(
        all_rows,
        key=lambda row: (
            int(row["source_index"]),
            str(row["raw_id"]),
            int(row["event_index"]),
        ),
    )
    new_source = {
        int(row["source_index"])
        for row in new_rows
    }
    queries = [
        row
        for row in ordered
        if row["role"] == "user"
        and bool(row["nonempty"])
        and int(row["source_index"]) in new_source
    ]
    old_queries = [
        row
        for row in existing_rows
        if row["role"] == "user" and bool(row["nonempty"])
    ]
    old_query = old_queries[-1] if old_queries else None
    first_new_query = queries[0] if queries else None
    changed_old = False
    if old_query is not None:
        old_source = int(old_query["source_index"])
        changed_old = any(
            int(row["source_index"]) > old_source
            and (
                first_new_query is None
                or int(row["source_index"])
                < int(first_new_query["source_index"])
            )
            for row in new_rows
        )
    affected_queries: list[Mapping[str, Any]] = []
    if changed_old:
        affected_queries.append(old_query)
    affected_queries.extend(queries)
    if not affected_queries:
        return []

    exposure = distill._exposure_map(root)
    materialized: list[dict[str, Any]] = []
    for index, query in enumerate(affected_queries):
        query_source = int(query["source_index"])
        next_query_source = (
            int(affected_queries[index + 1]["source_index"])
            if index + 1 < len(affected_queries)
            else None
        )
        prefix = [
            row for row in ordered if int(row["source_index"]) < query_source
        ]
        answer_rows: list[Mapping[str, Any]] = []
        tool_rows: list[Mapping[str, Any]] = []
        for row in ordered:
            source = int(row["source_index"])
            if source <= query_source:
                continue
            if next_query_source is not None and source >= next_query_source:
                break
            if row["role"] == "assistant" and bool(row["nonempty"]):
                answer_rows.append(row)
            elif row["role"] == "tool":
                tool_rows.append(row)
        if query is old_query:
            rally_id = _rally_id_for_query(query)
            stored = connection.execute(
                "SELECT row_json FROM rallies WHERE rally_id=?", (rally_id,)
            ).fetchone()
            if stored is None:
                raise _CatalogRepairRequired("existing rally is absent from catalog")
            try:
                prior = json.loads(str(stored[0]))
            except (TypeError, json.JSONDecodeError) as exc:
                raise _CatalogRepairRequired("existing rally is invalid") from exc
            if not isinstance(prior, dict) or prior.get("rally_id") != rally_id:
                raise _CatalogRepairRequired("existing rally conflicts with query")
        materialized.append(
            _materialize_catalog_rally(
                query,
                prefix,
                answer_rows,
                tool_rows,
                max_context_bytes=max_context_bytes,
                exposure=exposure,
            )
        )
    return materialized


def advance(raw_dir: Path, root: Path, max_context_bytes: int) -> CatalogAdvance:
    """Add newly committed Raw v2 units without duplicating transcript text."""

    if max_context_bytes <= 0:
        raise CatalogError("max_context_bytes must be positive")
    try:
        raw_store = RawStore(raw_dir, mode="v2")
        units = list(raw_store.iter_segment_units())
        watermark = committed_raw_watermark(raw_dir)
    except RawSegmentCorrupt as exc:
        raise CatalogError("committed Raw inventory is invalid") from exc
    current = {unit.raw_id: unit for unit in units}
    prior_checkpoint = (
        _read_catalog_checkpoint(root) if catalog_path(root).exists() else None
    )
    connection = _connect(root)
    try:
        connection.execute("BEGIN IMMEDIATE")
        metadata = dict(connection.execute("SELECT key,value FROM metadata"))
        schema = metadata.get("schema")
        stored = {
            str(row["raw_id"]): row
            for row in connection.execute(
                "SELECT raw_id,raw_sha256,receipt_sha256 FROM raw_units"
            )
        }
        columns = {str(row[1]) for row in connection.execute("PRAGMA table_info(raw_units)")}
        repairing = (
            schema not in {None, CATALOG_SCHEMA}
            or (schema is None and bool(stored))
            or (
                schema is not None
                and (
                    prior_checkpoint is None
                    or prior_checkpoint["catalog_watermark"] != metadata.get("watermark")
                    or _catalog_lineage(prior_checkpoint) is None
                )
            )
            or bool({"source_after_line", "source_until_line"} - columns)
        )
        if repairing:
            _reset_catalog_for_repair(connection)
            schema = None
            stored = {}
        if schema is not None and set(stored) - set(current):
            raise CatalogError("committed Raw inventory removed from catalog")
        new_units: list[RawUnit] = []
        for raw_id, unit in current.items():
            existing = stored.get(raw_id)
            raw_sha256, receipt_sha256 = _unit_identity(unit)
            if existing is None:
                new_units.append(unit)
            elif (
                existing["raw_sha256"] != raw_sha256
                or existing["receipt_sha256"] != receipt_sha256
            ):
                raise CatalogError("committed Raw digest conflicts with catalog")
        if schema is not None and not new_units:
            if metadata.get("watermark") != watermark:
                raise CatalogError("catalog watermark conflicts with committed Raw")
            connection.rollback()
            return CatalogAdvance("noop", watermark, (), (), (), ())

        indexed: list[str] = []
        archived: list[str] = []
        if schema is None:
            by_raw, rally_rows, archived_ids = _bootstrap(
                raw_dir, root, max_context_bytes, units
            )
            for unit in units:
                status = "archived" if unit.raw_id in archived_ids else "indexed"
                _store_unit(connection, unit, status=status)
                _store_events(connection, by_raw.get(unit.raw_id, ()))
                (archived if status == "archived" else indexed).append(unit.raw_id)
            rally_ids = _store_rallies(connection, rally_rows)
            deferred: tuple[tuple[str, str], ...] = ()
            status = "repaired" if repairing else "bootstrap"
        else:
            _validate_new_unit_intervals(connection, new_units)
            existing_sessions = {
                (str(row[0]), str(row[1]))
                for row in connection.execute(
                    "SELECT DISTINCT host,session_key FROM raw_units"
                )
            }
            delta: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
            session_new_rows: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(
                list
            )
            deferred_set: set[tuple[str, str]] = set()
            for unit in new_units:
                unit_status, rows = _read_unit_events(raw_store, unit)
                _store_unit(connection, unit, status=unit_status)
                _store_events(connection, rows)
                (archived if unit_status == "archived" else indexed).append(unit.raw_id)
                for row in rows:
                    key = (str(row["host"]), str(row["session_key"]))
                    session_new_rows[key].append(row)
                    if key in existing_sessions:
                        deferred_set.add(key)
                    else:
                        delta[key].append(row)
            rally_rows = []
            for key, rows in sorted(delta.items()):
                try:
                    rally_rows.extend(
                        _incremental_session_rallies(
                            connection,
                            root,
                            key[0],
                            key[1],
                            rows,
                            max_context_bytes,
                        )
                    )
                except CatalogError as exc:
                    raise CatalogError("cannot derive delta rallies") from exc
            for host, session_key in sorted(deferred_set):
                try:
                    rally_rows.extend(
                        _incremental_session_rallies(
                            connection,
                            root,
                            host,
                            session_key,
                            session_new_rows[(host, session_key)],
                            max_context_bytes,
                        )
                    )
                except _CatalogRepairRequired:
                    try:
                        session_rows = _session_events(
                            connection, raw_dir, host, session_key
                        )
                        rally_rows.extend(
                            distill.extract_rallies(
                                raw_dir,
                                root=root,
                                max_context_bytes=max_context_bytes,
                                _event_rows=session_rows,
                            )
                        )
                    except (CatalogError, distill.DistillationError) as exc:
                        raise CatalogError(
                            "cannot rebuild existing session tail"
                        ) from exc
                except CatalogError as exc:
                    raise CatalogError(str(exc)) from exc
            rally_ids = _store_rallies(connection, rally_rows)
            deferred = tuple(sorted(deferred_set))
            status = "advanced"
        connection.executemany(
            "INSERT OR REPLACE INTO metadata VALUES(?,?)",
            (("schema", CATALOG_SCHEMA), ("watermark", watermark)),
        )
        event_rowid = int(
            connection.execute("SELECT COALESCE(MAX(rowid),0) FROM events").fetchone()[0]
        )
        catalog_lineage = (
            _new_catalog_lineage()
            if schema is None or repairing
            else cast(str, _catalog_lineage(prior_checkpoint))
        )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
    _ensure_private_file(catalog_path(root))
    _write_catalog_checkpoint(
        root, watermark, event_rowid, catalog_lineage=catalog_lineage
    )
    return CatalogAdvance(
        status,
        watermark,
        tuple(sorted(indexed)),
        tuple(sorted(archived)),
        rally_ids,
        deferred,
    )


def _require_catalog_checkpoint(root: Path) -> None:
    if _read_catalog_checkpoint(root) is None:
        raise CatalogError("catalog checkpoint requires repair")


def rallies(root: Path, ids: Iterable[str] | None = None) -> list[dict[str, Any]]:
    """Read text-free Rally manifests already derived by :func:`advance`."""

    _require_catalog_checkpoint(root)
    selected = None if ids is None else tuple(ids)
    with _connect(root) as connection:
        if selected is not None:
            if not selected:
                return []
            placeholders = ",".join("?" for _ in selected)
            rows = connection.execute(
                f"SELECT row_json FROM rallies WHERE rally_id IN ({placeholders}) "
                "ORDER BY as_of_us,rally_id",
                selected,
            )
        else:
            rows = connection.execute(
                "SELECT row_json FROM rallies ORDER BY as_of_us,rally_id"
            )
        return [json.loads(row[0]) for row in rows]


def _candidate_file_state(path: Path) -> dict[str, int]:
    """Return a cheap identity/size tuple for the append-only candidate ledger."""

    try:
        stat = path.stat()
    except FileNotFoundError:
        return {
            "ledger_exists": 0,
            "ledger_size": 0,
            "ledger_dev": -1,
            "ledger_ino": -1,
            "ledger_mtime_ns": -1,
            "ledger_ctime_ns": -1,
        }
    except OSError as exc:
        raise CatalogError("candidate ledger is unreadable") from exc
    if not path.is_file():
        raise CatalogError("candidate ledger is not a regular file")
    return {
        "ledger_exists": 1,
        "ledger_size": stat.st_size,
        "ledger_dev": stat.st_dev,
        "ledger_ino": stat.st_ino,
        "ledger_mtime_ns": stat.st_mtime_ns,
        "ledger_ctime_ns": stat.st_ctime_ns,
    }


def _candidate_sha(value: object, *, allow_empty: bool = False) -> str:
    if allow_empty and value == "":
        return ""
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise CatalogError("candidate ledger digest is invalid")
    return value


def _candidate_snapshot(
    row: Mapping[str, Any],
    *,
    offset: int,
    length: int,
    expected_previous: str,
    verify_digests: bool = True,
) -> dict[str, Any]:
    """Validate one canonical candidate row and retain only index metadata."""

    if (
        row.get("schema") != store.DISTILLATION_SCHEMA
        or row.get("namespace") != "recall-distillation"
        or row.get("kind") != "candidate-snapshot"
    ):
        raise CatalogError("candidate ledger row metadata is invalid")
    previous = _candidate_sha(row.get("previous_sha256"), allow_empty=True)
    if previous != expected_previous:
        raise CatalogError("candidate ledger chain mismatch")
    record_sha256 = _candidate_sha(row.get("record_sha256"))
    if verify_digests:
        unsigned = {
            key: value for key, value in row.items() if key != "record_sha256"
        }
        try:
            record_digest = canonical_json_sha256_strict(unsigned)
        except (TypeError, ValueError, OverflowError) as exc:
            raise CatalogError("candidate ledger record is not canonical") from exc
        if record_sha256 != record_digest:
            raise CatalogError("candidate ledger record digest mismatch")
    rally_id = row.get("rally_id")
    if not isinstance(rally_id, str) or not rally_id:
        raise CatalogError("candidate rally id is invalid")
    snapshot = row.get("snapshot")
    if not isinstance(snapshot, Mapping):
        raise CatalogError("candidate snapshot is not an object")
    if (
        snapshot.get("schema") != CANDIDATE_SNAPSHOT_SCHEMA
        or snapshot.get("rally_id") != rally_id
        or not isinstance(snapshot.get("as_of"), str)
        or not isinstance(snapshot.get("retriever_revision"), str)
        or not isinstance(snapshot.get("feature_revision"), str)
        or not isinstance(snapshot.get("query_feature_text_sha256"), str)
        or not isinstance(snapshot.get("candidates"), list)
    ):
        raise CatalogError("candidate snapshot shape is invalid")
    _candidate_sha(snapshot["query_feature_text_sha256"])
    snapshot_sha256 = _candidate_sha(snapshot.get("snapshot_sha256"))
    if verify_digests:
        snapshot_unsigned = {
            key: value for key, value in snapshot.items() if key != "snapshot_sha256"
        }
        try:
            snapshot_digest = canonical_json_sha256_strict(snapshot_unsigned)
        except (TypeError, ValueError, OverflowError) as exc:
            raise CatalogError("candidate snapshot is not canonical") from exc
        if snapshot_sha256 != snapshot_digest:
            raise CatalogError("candidate snapshot digest mismatch")
    seen_candidates: set[str] = set()
    candidates = snapshot["candidates"]
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, Mapping):
            raise CatalogError("candidate item is not an object")
        candidate_id = candidate.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise CatalogError("candidate id is invalid")
        if candidate_id in seen_candidates:
            raise CatalogError("candidate id is duplicated within rally")
        seen_candidates.add(candidate_id)
        rank = candidate.get("rank", index + 1)
        if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
            raise CatalogError("candidate rank is invalid")
        text_sha256 = candidate.get("text_sha256")
        if text_sha256 is not None:
            _candidate_sha(text_sha256)
        feature_text_sha256 = candidate.get("candidate_feature_text_sha256")
        if feature_text_sha256 is not None:
            _candidate_sha(feature_text_sha256)
        source_index = candidate.get("source_index")
        if source_index is not None and (
            isinstance(source_index, bool) or not isinstance(source_index, int)
        ):
            raise CatalogError("candidate source index is invalid")
    if offset < 0 or length <= 0:
        raise CatalogError("candidate ledger offset is invalid")
    return {
        "rally_id": rally_id,
        "previous_sha256": previous,
        "record_sha256": record_sha256,
        "snapshot_sha256": snapshot_sha256,
        "offset": offset,
        "length": length,
    }


def _scan_candidate_ledger(
    path: Path,
    *,
    start: int,
    expected_previous: str,
    expected_count: int,
    verified_head: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Decode a full ledger for rebuild or only its append tail.

    ``store.chain_head`` already verified the complete append-only ledger when
    ``verified_head`` is supplied.  Keep the structural/offset checks here,
    but do not hash each large JSON row a second time.
    """

    state = _candidate_file_state(path)
    if not state["ledger_exists"]:
        if start or expected_count or expected_previous:
            raise CatalogError("candidate ledger was truncated")
        return []
    if start < 0 or start > state["ledger_size"]:
        raise CatalogError("candidate ledger offset is invalid")
    parsed: list[dict[str, Any]] = []
    try:
        with path.open("rb") as handle:
            if start:
                handle.seek(start)
            previous = expected_previous
            record_index = expected_count
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if not line.endswith(b"\n"):
                    raise CatalogError("candidate ledger tail is truncated")
                try:
                    decoded = json.loads(line)
                except (UnicodeError, json.JSONDecodeError) as exc:
                    raise CatalogError("candidate ledger row is invalid") from exc
                if not isinstance(decoded, dict):
                    raise CatalogError("candidate ledger row is not an object")
                if verified_head is None:
                    try:
                        canonical_line = canonical_json_bytes_strict(decoded) + b"\n"
                    except (TypeError, ValueError, OverflowError) as exc:
                        raise CatalogError(
                            "candidate ledger row is not canonical"
                        ) from exc
                    if canonical_line != line:
                        raise CatalogError("candidate ledger row is not canonical")
                metadata = _candidate_snapshot(
                    decoded,
                    offset=offset,
                    length=len(line),
                    expected_previous=previous,
                    verify_digests=verified_head is None,
                )
                metadata["record_index"] = record_index
                parsed.append(metadata)
                previous = str(metadata["record_sha256"])
                record_index += 1
    except CatalogError:
        raise
    except (OSError, UnicodeError) as exc:
        raise CatalogError("candidate ledger is unreadable") from exc
    if verified_head is not None and (
        verified_head.get("records") != expected_count + len(parsed)
        or verified_head.get("head_sha256")
        != (expected_previous if not parsed else parsed[-1]["record_sha256"])
    ):
        raise CatalogError("candidate ledger head conflicts with scan")
    return parsed


def _candidate_index_digest(row: Mapping[str, Any]) -> str:
    return canonical_json_sha256_strict(
        {
            key: row[key]
            for key in (
                "record_index",
                "rally_id",
                "previous_sha256",
                "record_sha256",
                "snapshot_sha256",
                "offset",
                "length",
            )
        }
    )


def _verify_candidate_index_row(row: sqlite3.Row) -> dict[str, Any]:
    value = dict(row)
    if _candidate_sha(value.get("index_sha256")) != _candidate_index_digest(value):
        raise CatalogError("candidate offset index row is invalid")
    return value


def _candidate_index_state_row(
    connection: sqlite3.Connection,
) -> dict[str, Any] | None:
    try:
        rows = list(connection.execute("SELECT * FROM candidate_index_state"))
    except sqlite3.DatabaseError as exc:
        raise CatalogError("candidate index state is unreadable") from exc
    if not rows:
        return None
    if len(rows) != 1:
        raise CatalogError("candidate index state is duplicated")
    row = dict(rows[0])
    if row.get("singleton") != 1 or row.get("index_schema") != CANDIDATE_INDEX_SCHEMA:
        raise CatalogError("candidate index schema conflicts")
    if not isinstance(row.get("ledger_path"), str) or not row["ledger_path"]:
        raise CatalogError("candidate index ledger path is invalid")
    if row.get("ledger_exists") not in (0, 1):
        raise CatalogError("candidate index file state is invalid")
    for key in (
        "ledger_size",
        "ledger_dev",
        "ledger_ino",
        "ledger_mtime_ns",
        "ledger_ctime_ns",
        "record_count",
    ):
        if isinstance(row.get(key), bool) or not isinstance(row.get(key), int):
            raise CatalogError("candidate index file state is invalid")
    _candidate_sha(row.get("head_sha256"), allow_empty=True)
    if (row["record_count"] == 0) != (row["head_sha256"] == ""):
        raise CatalogError("candidate index count/head mismatch")
    return row


def _candidate_db_summary(
    connection: sqlite3.Connection, state: Mapping[str, Any]
) -> tuple[int, str]:
    try:
        count = int(
            connection.execute("SELECT COUNT(*) FROM candidate_records").fetchone()[0]
        )
        raw_row = connection.execute(
            "SELECT * FROM candidate_records ORDER BY record_index DESC LIMIT 1"
        ).fetchone()
    except sqlite3.DatabaseError as exc:
        raise CatalogError("candidate index is unreadable") from exc
    row = None if raw_row is None else _verify_candidate_index_row(raw_row)
    head = "" if row is None else str(row["record_sha256"])
    if count != int(state["record_count"]) or head != state["head_sha256"]:
        raise CatalogError("candidate index state conflicts with records")
    if row is not None and int(row["record_index"]) != count - 1:
        raise CatalogError("candidate index sequence is invalid")
    if row is not None and (
        int(row["offset"]) < 0
        or int(row["length"]) <= 0
        or int(row["offset"]) + int(row["length"]) != int(state["ledger_size"])
    ):
        raise CatalogError("candidate offset index size conflicts with ledger")
    return count, head


def _candidate_state_values(
    path: Path,
    file_state: Mapping[str, int],
    *,
    record_count: int,
    head_sha256: str,
) -> tuple[Any, ...]:
    return (
        1,
        CANDIDATE_INDEX_SCHEMA,
        str(path),
        file_state["ledger_exists"],
        file_state["ledger_size"],
        file_state["ledger_dev"],
        file_state["ledger_ino"],
        file_state["ledger_mtime_ns"],
        file_state["ledger_ctime_ns"],
        record_count,
        head_sha256,
    )


def _upsert_candidate_index_state(
    connection: sqlite3.Connection, values: tuple[Any, ...]
) -> None:
    connection.execute(
        """
        INSERT INTO candidate_index_state(
            singleton,index_schema,ledger_path,ledger_exists,ledger_size,
            ledger_dev,ledger_ino,ledger_mtime_ns,ledger_ctime_ns,
            record_count,head_sha256
        ) VALUES(?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(singleton) DO UPDATE SET
            index_schema=excluded.index_schema,
            ledger_path=excluded.ledger_path,
            ledger_exists=excluded.ledger_exists,
            ledger_size=excluded.ledger_size,
            ledger_dev=excluded.ledger_dev,
            ledger_ino=excluded.ledger_ino,
            ledger_mtime_ns=excluded.ledger_mtime_ns,
            ledger_ctime_ns=excluded.ledger_ctime_ns,
            record_count=excluded.record_count,
            head_sha256=excluded.head_sha256
        """,
        values,
    )


def _insert_candidate_rows(
    connection: sqlite3.Connection,
    rows: Iterable[dict[str, Any]],
) -> None:
    for metadata in rows:
        try:
            connection.execute(
                """
                INSERT INTO candidate_records(
                    record_index,rally_id,previous_sha256,record_sha256,
                    snapshot_sha256,offset,length,index_sha256
                ) VALUES(?,?,?,?,?,?,?,?)
                """,
                (
                    metadata["record_index"],
                    metadata["rally_id"],
                    metadata["previous_sha256"],
                    metadata["record_sha256"],
                    metadata["snapshot_sha256"],
                    metadata["offset"],
                    metadata["length"],
                    _candidate_index_digest(metadata),
                ),
            )
        except sqlite3.IntegrityError as exc:
            raise CatalogError("candidate rally is duplicated") from exc


def sync_candidate_index(
    root: Path, ledger_path: Path, *, rebuild: bool = False
) -> dict[str, Any]:
    """Project candidate ledger offsets without copying snapshot bodies."""

    path = ledger_path.expanduser().resolve(strict=False)
    catalog_checkpoint = _read_catalog_checkpoint(root)
    connection = _connect(root)
    status = "noop"
    try:
        connection.execute("BEGIN IMMEDIATE")
        state = _candidate_index_state_row(connection)
        current = _candidate_file_state(path)
        if rebuild:
            connection.execute("DELETE FROM candidate_records")
            connection.execute("DELETE FROM candidate_index_state")
            state = None
            mode = "bootstrap"
        elif state is None:
            if connection.execute(
                "SELECT COUNT(*) FROM candidate_records"
            ).fetchone()[0]:
                raise CatalogError("candidate index state is missing")
            mode = "bootstrap"
        else:
            if state["ledger_path"] != str(path):
                raise CatalogError("candidate index ledger path conflicts")
            count, head = _candidate_db_summary(connection, state)
            stored = {
                key: state[key]
                for key in (
                    "ledger_exists",
                    "ledger_size",
                    "ledger_dev",
                    "ledger_ino",
                    "ledger_mtime_ns",
                    "ledger_ctime_ns",
                )
            }
            if current == stored:
                connection.rollback()
                return {
                    "status": "noop",
                    "count": count,
                    "record_count": count,
                    "indexed_seq": count,
                    "indexed_offset": int(state["ledger_size"]),
                    "head_sha256": head,
                }
            if (
                current["ledger_exists"] == 0
                or current["ledger_size"] < state["ledger_size"]
                or (
                    current["ledger_dev"],
                    current["ledger_ino"],
                )
                != (state["ledger_dev"], state["ledger_ino"])
            ):
                if count == 0 and state["ledger_size"] == 0:
                    mode = "bootstrap"
                    connection.execute("DELETE FROM candidate_records")
                else:
                    raise CatalogError("candidate ledger rollback or replacement")
            elif current["ledger_size"] == state["ledger_size"]:
                try:
                    verified_head = store.chain_head(path)
                except store.DistillationStoreError as exc:
                    raise CatalogError("candidate ledger head is invalid") from exc
                if (
                    verified_head["records"] != count
                    or verified_head["head_sha256"] != head
                ):
                    raise CatalogError("candidate ledger head conflicts with index")
                _upsert_candidate_index_state(
                    connection,
                    _candidate_state_values(
                        path,
                        current,
                        record_count=count,
                        head_sha256=head,
                    ),
                )
                connection.commit()
                _ensure_private_file(catalog_path(root))
                lineage = (
                    _catalog_lineage(catalog_checkpoint)
                    if catalog_checkpoint is not None
                    else None
                )
                if catalog_checkpoint is not None and lineage is not None:
                    _write_candidate_catalog_state(root, catalog_checkpoint)
                    _write_catalog_checkpoint(
                        root,
                        str(catalog_checkpoint["catalog_watermark"]),
                        int(catalog_checkpoint["event_rowid"]),
                        catalog_lineage=lineage,
                    )
                return {
                    "status": "noop",
                    "count": count,
                    "record_count": count,
                    "indexed_seq": count,
                    "indexed_offset": int(current["ledger_size"]),
                    "head_sha256": head,
                }
            else:
                mode = "tail"
        if mode == "bootstrap":
            start = 0
            expected_previous = ""
            expected_count = 0
        else:
            # ``state`` is present for the only remaining path: an append tail.
            assert state is not None
            start = int(state["ledger_size"])
            expected_previous = str(state["head_sha256"])
            expected_count = int(state["record_count"])
        verified_head: Mapping[str, Any] | None = None
        if mode in {"bootstrap", "tail"}:
            try:
                verified_head = store.chain_head(path)
            except store.DistillationStoreError as exc:
                raise CatalogError("candidate ledger head is invalid") from exc
            # ``chain_head`` may recover an interrupted final line while
            # rebuilding its checkpoint; use the post-recovery file state for
            # the immutable scan below.
            current = _candidate_file_state(path)
        parsed = _scan_candidate_ledger(
            path,
            start=start,
            expected_previous=expected_previous,
            expected_count=expected_count,
            verified_head=verified_head,
        )
        after = _candidate_file_state(path)
        if after != current:
            raise CatalogError("candidate ledger changed during indexing")
        if mode == "bootstrap":
            _insert_candidate_rows(connection, parsed)
            count = len(parsed)
        else:
            _insert_candidate_rows(connection, parsed)
            count = expected_count + len(parsed)
        head = expected_previous if not parsed else str(parsed[-1]["record_sha256"])
        if verified_head is not None and (
            verified_head["records"] != count
            or verified_head["head_sha256"] != head
        ):
            raise CatalogError("candidate ledger head conflicts with index")
        _upsert_candidate_index_state(
            connection,
            _candidate_state_values(
                path,
                current,
                record_count=count,
                head_sha256=head,
            ),
        )
        connection.commit()
        status = "bootstrap" if mode == "bootstrap" else "advanced"
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
    _ensure_private_file(catalog_path(root))
    lineage = (
        _catalog_lineage(catalog_checkpoint)
        if catalog_checkpoint is not None
        else None
    )
    if catalog_checkpoint is not None and lineage is not None:
        _write_candidate_catalog_state(root, catalog_checkpoint)
        _write_catalog_checkpoint(
            root,
            str(catalog_checkpoint["catalog_watermark"]),
            int(catalog_checkpoint["event_rowid"]),
            catalog_lineage=lineage,
        )
    return {
        "status": status,
        "count": count,
        "record_count": count,
        "indexed_seq": count,
        "indexed_offset": int(current["ledger_size"]),
        "head_sha256": head,
    }


def candidate_index_state(root: Path) -> dict[str, Any]:
    """Return indexed count/head metadata without touching the ledger body."""

    connection = _connect(root)
    try:
        state = _candidate_index_state_row(connection)
        if state is None:
            return {
                "count": 0,
                "record_count": 0,
                "indexed_seq": 0,
                "indexed_offset": 0,
                "ledger_size": 0,
                "ledger_ino": -1,
                "head_sha256": "",
            }
        count, head = _candidate_db_summary(connection, state)
        return {
            **state,
            "count": count,
            "record_count": count,
            "indexed_seq": count,
            "indexed_offset": state["ledger_size"],
            "head_sha256": head,
        }
    finally:
        connection.close()


def candidate_rally_ids(root: Path, *, after_seq: int = 0) -> set[str]:
    """Return rally ids already projected into the candidate offset index."""

    if isinstance(after_seq, bool) or not isinstance(after_seq, int) or after_seq < 0:
        raise CatalogError("candidate index sequence is invalid")
    connection = _connect(root)
    try:
        state = _candidate_index_state_row(connection)
        if state is None:
            return set()
        _candidate_db_summary(connection, state)
        rows = connection.execute(
            "SELECT * FROM candidate_records WHERE record_index >= ? "
            "ORDER BY record_index",
            (after_seq,),
        )
        return {
            str(_verify_candidate_index_row(row)["rally_id"])
            for row in rows
        }
    except sqlite3.DatabaseError as exc:
        raise CatalogError("candidate index is unreadable") from exc
    finally:
        connection.close()


def read_candidate_snapshots(
    root: Path,
    ledger_path: Path,
    rally_ids: Iterable[str | Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Read exact candidate snapshot rows by indexed offset/length."""

    path = ledger_path.expanduser().resolve(strict=False)
    requested: dict[str, list[Mapping[str, Any]]] = {}
    expected_candidates: dict[str, set[str]] = defaultdict(set)
    if isinstance(rally_ids, (str, Mapping)):
        values: Iterable[str | Mapping[str, Any]] = (rally_ids,)
    elif hasattr(rally_ids, "payload_ref"):
        values = ({"payload_ref": rally_ids.payload_ref},)
    else:
        values = rally_ids
    for value in values:
        if isinstance(value, Mapping):
            rally_id = value.get("rally_id")
            payload_ref = value.get("payload_ref")
            candidate_id = value.get("candidate_id")
            if payload_ref is not None:
                if not isinstance(payload_ref, str):
                    raise CatalogError("candidate claim reference is invalid")
                parts = payload_ref.split(":")
                if (
                    len(parts) not in (2, 3)
                    or parts[0] != "candidate-snapshot"
                    or not parts[1]
                    or (len(parts) == 3 and not parts[2])
                ):
                    raise CatalogError("candidate claim reference is invalid")
                if isinstance(rally_id, str) and rally_id != parts[1]:
                    raise CatalogError("candidate claim reference conflicts")
                rally_id = parts[1]
                if len(parts) == 3:
                    if isinstance(candidate_id, str) and candidate_id != parts[2]:
                        raise CatalogError("candidate claim reference conflicts")
                    candidate_id = parts[2]
            if not isinstance(rally_id, str) or not rally_id:
                raise CatalogError("candidate claim reference is invalid")
            if candidate_id is not None:
                if not isinstance(candidate_id, str) or not candidate_id:
                    raise CatalogError("candidate claim reference is invalid")
                expected_candidates[rally_id].add(candidate_id)
            requested.setdefault(rally_id, []).append(value)
        elif isinstance(value, str) and value:
            requested.setdefault(value, [])
        else:
            raise CatalogError("candidate rally id is invalid")
    if not requested:
        return {}
    connection = _connect(root)
    try:
        state = _candidate_index_state_row(connection)
        if state is None or state["ledger_path"] != str(path):
            raise CatalogError("candidate index is not synchronized")
        _candidate_db_summary(connection, state)
        current = _candidate_file_state(path)
        stored = {
            key: state[key]
            for key in (
                "ledger_exists",
                "ledger_size",
                "ledger_dev",
                "ledger_ino",
                "ledger_mtime_ns",
                "ledger_ctime_ns",
            )
        }
        if current != stored:
            raise CatalogError("candidate ledger changed after indexing")
        placeholders = ",".join("?" for _ in requested)
        rows = [
            _verify_candidate_index_row(row)
            for row in connection.execute(
                "SELECT * FROM candidate_records WHERE rally_id IN ("
                + placeholders
                + ")",
                tuple(requested),
            )
        ]
        if len(rows) != len(requested):
            raise CatalogError("candidate rally is absent from index")
    except sqlite3.DatabaseError as exc:
        raise CatalogError("candidate index is unreadable") from exc
    finally:
        connection.close()
    result: dict[str, dict[str, Any]] = {}
    try:
        with path.open("rb") as handle:
            for row in rows:
                offset = int(row["offset"])
                length = int(row["length"])
                if (
                    offset < 0
                    or length <= 0
                    or offset + length > int(state["ledger_size"])
                ):
                    raise CatalogError("candidate offset index is invalid")
                handle.seek(offset)
                encoded = handle.read(length)
                if len(encoded) != length or not encoded.endswith(b"\n"):
                    raise CatalogError("candidate ledger row is truncated")
                try:
                    decoded = json.loads(encoded)
                except (UnicodeError, json.JSONDecodeError) as exc:
                    raise CatalogError("candidate ledger row is invalid") from exc
                if not isinstance(decoded, dict):
                    raise CatalogError("candidate ledger row is not an object")
                try:
                    canonical_line = canonical_json_bytes_strict(decoded) + b"\n"
                except (TypeError, ValueError, OverflowError) as exc:
                    raise CatalogError("candidate ledger row is not canonical") from exc
                if canonical_line != encoded:
                    raise CatalogError("candidate ledger row is not canonical")
                metadata = _candidate_snapshot(
                    decoded,
                    offset=offset,
                    length=length,
                    expected_previous=str(row["previous_sha256"]),
                )
                if (
                    metadata["rally_id"] != row["rally_id"]
                    or metadata["record_sha256"] != row["record_sha256"]
                    or metadata["snapshot_sha256"] != row["snapshot_sha256"]
                ):
                    raise CatalogError("candidate index row conflicts with ledger")
                rally_id = str(row["rally_id"])
                for claim in requested[rally_id]:
                    for key in ("record_sha256", "snapshot_sha256"):
                        if key in claim and claim[key] != row[key]:
                            raise CatalogError("candidate claim conflicts with index")
                    claim_offset = claim.get("offset")
                    claim_length = claim.get("length")
                    if claim_offset is not None and claim_offset != offset:
                        raise CatalogError("candidate claim offset conflicts")
                    if claim_length is not None and claim_length != length:
                        raise CatalogError("candidate claim length conflicts")
                snapshot = cast(dict[str, Any], decoded["snapshot"])
                actual_candidates = {
                    str(candidate.get("candidate_id"))
                    for candidate in snapshot.get("candidates", [])
                    if isinstance(candidate, Mapping)
                }
                if not expected_candidates[rally_id] <= actual_candidates:
                    raise CatalogError("candidate claim is absent from snapshot")
                result[rally_id] = snapshot
    except CatalogError:
        raise
    except OSError as exc:
        raise CatalogError("candidate ledger is unreadable") from exc
    return result


def texts(
    raw_dir: Path,
    root: Path,
    hashes: Iterable[str] | None = None,
    refs: Iterable[Mapping[str, Any]] | None = None,
) -> dict[str, str]:
    """Resolve requested catalog text from verified Raw bytes on demand."""

    _require_catalog_checkpoint(root)
    requested_hashes = set(hashes or ())
    requested_refs = list(refs or ())
    with _connect(root) as connection:
        rows: dict[tuple[str, int], sqlite3.Row] = {}
        if requested_hashes:
            placeholders = ",".join("?" for _ in requested_hashes)
            for row in connection.execute(
                f"SELECT * FROM events WHERE semantic_sha256 IN ({placeholders})",
                tuple(sorted(requested_hashes)),
            ):
                rows[(str(row["raw_id"]), int(row["event_index"]))] = row
        for ref in requested_refs:
            raw_id = ref.get("raw_id")
            event_index = ref.get("event_index")
            if not isinstance(raw_id, str) or not isinstance(event_index, int):
                raise CatalogError("text reference is invalid")
            row = connection.execute(
                "SELECT * FROM events WHERE raw_id=? AND event_index=?",
                (raw_id, event_index),
            ).fetchone()
            if row is None:
                raise CatalogError("text reference is absent from catalog")
            for key in ("raw_sha256", "receipt_sha256", "semantic_sha256"):
                if ref.get(key) != row[key]:
                    raise CatalogError("text reference conflicts with catalog")
            rows[(raw_id, event_index)] = row
    return _resolve_rows(raw_dir, cast(Iterable[Mapping[str, Any]], rows.values()))


class CatalogTextCache(Mapping[str, str]):
    """Resolve only requested catalog text and retain it for the current run."""

    def __init__(self, raw_dir: Path, root: Path) -> None:
        checkpoint = _read_catalog_checkpoint(root)
        if checkpoint is None:
            raise CatalogError("catalog checkpoint requires repair")
        self._raw_dir = raw_dir
        self._root = root
        self._catalog_file_state = checkpoint["file_state"]
        self._checkpoint_file_state = _index_file_state(_catalog_checkpoint_path(root))
        self._candidate_state_file_state = _index_file_state(
            _candidate_catalog_state_path(root)
        )
        self._watermark = checkpoint["catalog_watermark"]
        self._event_rowid = checkpoint["event_rowid"]
        self._cache: dict[str, str] = {}

    def _require_current_catalog(self) -> None:
        catalog_state = _index_file_state(catalog_path(self._root))
        checkpoint_state = _index_file_state(_catalog_checkpoint_path(self._root))
        if (
            catalog_state == self._catalog_file_state
            and checkpoint_state == self._checkpoint_file_state
        ):
            return
        candidate_state = _index_file_state(_candidate_catalog_state_path(self._root))
        if candidate_state == self._candidate_state_file_state:
            raise CatalogError("catalog checkpoint requires repair")
        bridge = _read_candidate_catalog_state(self._root)
        checkpoint = _read_catalog_checkpoint(self._root)
        if (
            bridge is None
            or checkpoint is None
            or bridge["catalog_watermark"] != self._watermark
            or bridge["event_rowid"] != self._event_rowid
            or checkpoint["catalog_watermark"] != self._watermark
            or checkpoint["event_rowid"] != self._event_rowid
        ):
            raise CatalogError("catalog checkpoint requires repair")
        self._catalog_file_state = checkpoint["file_state"]
        self._checkpoint_file_state = checkpoint_state
        self._candidate_state_file_state = candidate_state

    def __getitem__(self, key: str) -> str:
        # Raw is immutable after resolution; rereading it per cache hit would
        # break the warm-path contract. A new cache resolution verifies Raw.
        self._require_current_catalog()
        if key not in self._cache:
            self.prefetch((key,))
        return self._cache[key]

    def __iter__(self) -> Iterator[str]:
        self._require_current_catalog()
        return iter(self._cache)

    def __len__(self) -> int:
        self._require_current_catalog()
        return len(self._cache)

    def prefetch(self, hashes: Iterable[str]) -> None:
        self._require_current_catalog()
        missing = sorted(
            {
                value
                for value in hashes
                if isinstance(value, str) and value and value not in self._cache
            }
        )
        if not missing:
            return
        selected: dict[str, sqlite3.Row] = {}
        path = catalog_path(self._root)
        try:
            with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
                connection.row_factory = sqlite3.Row
                for offset in range(0, len(missing), 500):
                    chunk = missing[offset : offset + 500]
                    placeholders = ",".join("?" for _ in chunk)
                    for row in connection.execute(
                        f"SELECT * FROM events WHERE semantic_sha256 IN ({placeholders})",
                        chunk,
                    ):
                        selected.setdefault(str(row["semantic_sha256"]), row)
        except sqlite3.DatabaseError as exc:
            raise CatalogError("historical catalog is unreadable") from exc
        self._cache.update(
            _resolve_rows(
                self._raw_dir,
                cast(Iterable[Mapping[str, Any]], selected.values()),
            )
        )


def _catalog_watermark(raw_dir: Path, root: Path) -> str:
    """Read only the catalog watermark before touching derived atom tables."""

    try:
        watermark = committed_raw_watermark(raw_dir)
    except RawSegmentCorrupt as exc:
        raise CatalogError("committed Raw inventory is invalid") from exc
    path = catalog_path(root)
    if not path.exists():
        raise CatalogError("historical catalog is absent")
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
            connection.row_factory = sqlite3.Row
            metadata = dict(connection.execute("SELECT key,value FROM metadata"))
            if (
                metadata.get("schema") != CATALOG_SCHEMA
                or metadata.get("watermark") != watermark
            ):
                raise CatalogError("historical catalog is not current")
    except sqlite3.DatabaseError as exc:
        raise CatalogError("historical catalog is unreadable") from exc
    return watermark


def _catalog_assistant_atoms_from_rows(
    rows: Iterable[sqlite3.Row],
) -> dict[str, dict[str, Any]]:
    """Convert selected catalog assistant rows into text-free atom metadata."""

    atoms: dict[str, dict[str, Any]] = {}
    for row in rows:
        try:
            structural = json.loads(str(row["structural_json"]))
        except json.JSONDecodeError as exc:
            raise CatalogError("catalog structural tokens are invalid") from exc
        ref = {
            "raw_id": row["raw_id"],
            "byte_range": [row["byte_start"], row["byte_end"]],
            "raw_sha256": row["raw_sha256"],
            "receipt_sha256": row["receipt_sha256"],
            "event_index": row["event_index"],
            "source_index": row["source_index"],
            "timestamp": row["timestamp"],
            "timestamp_us": row["timestamp_us"],
            "role": row["role"],
            "semantic_sha256": row["semantic_sha256"],
            "structural": structural,
        }
        atom_id = canonical_json_sha256_strict(
            {"kind": "assistant-atom-v1", "ref": ref}
        )
        atom = {
            "atom_id": atom_id,
            "host": row["host"],
            "session_cluster_id": row["session_cluster_id"],
            "source_index": row["source_index"],
            "timestamp_us": row["timestamp_us"],
            "text_sha256": row["semantic_sha256"],
            "ref": ref,
            "catalog_row": row,
            "catalog_rowid": int(row["catalog_rowid"]),
        }
        if atom_id in atoms:
            raise CatalogError("catalog assistant atom conflicts")
        atoms[atom_id] = atom
    return atoms


def _catalog_assistant_atoms(
    root: Path, *, after_rowid: int | None = None
) -> dict[str, dict[str, Any]]:
    """Read all assistants for rebuild, or only catalog rows after a cursor."""

    path = catalog_path(root)
    query = "SELECT rowid AS catalog_rowid,* FROM events WHERE role='assistant' AND nonempty=1"
    parameters: tuple[int, ...] = ()
    if after_rowid is not None:
        query += " AND rowid>?"
        parameters = (after_rowid,)
    query += " ORDER BY rowid"
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
            connection.row_factory = sqlite3.Row
            rows = list(connection.execute(query, parameters))
    except sqlite3.DatabaseError as exc:
        raise CatalogError("historical catalog is unreadable") from exc
    return _catalog_assistant_atoms_from_rows(rows)


def _index_atoms(
    connection: sqlite3.Connection,
    expected: Mapping[str, Mapping[str, Any]],
    *,
    legacy_checkpoint: Mapping[str, Any] | None = None,
) -> tuple[dict[str, dict[str, Any]], str]:
    try:
        metadata = dict(connection.execute("SELECT key,value FROM metadata"))
        if set(metadata) != {"schema", "content_sha256"}:
            raise CatalogError("historical index metadata conflicts")
        if metadata.get("schema") != HISTORICAL_INDEX_SCHEMA:
            raise CatalogError("historical index schema conflicts")
        records = list(
            connection.execute(
                """SELECT rowid,atom_id,host,session_cluster_id,source_index,
                          timestamp_us,text_sha256,ref_json,text FROM atoms"""
            )
        )
        fts = dict(connection.execute("SELECT rowid,search_text FROM atoms_fts"))
    except sqlite3.DatabaseError as exc:
        raise CatalogError("historical index is unreadable") from exc
    found_ids = {str(record[1]) for record in records}
    if found_ids - set(expected):
        raise CatalogError("historical index has extra assistant atom")
    if len(records) != len(found_ids) or set(fts) != {record[0] for record in records}:
        raise CatalogError("historical index FTS rows conflict")
    atoms: dict[str, dict[str, Any]] = {}
    for record in records:
        (
            rowid,
            atom_id,
            host,
            cluster,
            source_index,
            timestamp_us,
            text_sha256,
            ref_json,
            text,
        ) = record
        try:
            ref = json.loads(ref_json)
        except (TypeError, json.JSONDecodeError) as exc:
            raise CatalogError("historical index reference is invalid") from exc
        expected_atom = expected.get(str(atom_id))
        if expected_atom is None or (
            host,
            cluster,
            source_index,
            timestamp_us,
            text_sha256,
            ref,
        ) != (
            expected_atom["host"],
            expected_atom["session_cluster_id"],
            expected_atom["source_index"],
            expected_atom["timestamp_us"],
            expected_atom["text_sha256"],
            expected_atom["ref"],
        ):
            raise CatalogError("historical index assistant atom conflicts")
        if (
            not isinstance(text, str)
            or hashlib.sha256(text.encode()).hexdigest() != text_sha256
        ):
            raise CatalogError("historical index text conflicts")
        if fts[rowid] != store._search_terms(text):
            raise CatalogError("historical index FTS text conflicts")
        atoms[str(atom_id)] = {
            "atom_id": atom_id,
            "host": host,
            "session_cluster_id": cluster,
            "source_index": source_index,
            "timestamp_us": timestamp_us,
            "text_sha256": text_sha256,
            "ref": ref,
            "text": text,
        }
    digest = _index_digest({atom_id: expected[atom_id] for atom_id in atoms})
    legacy_chain_digest = _legacy_index_digest(
        {atom_id: expected[atom_id] for atom_id in atoms}
    )
    legacy_canonical_digest = canonical_json_sha256_strict(
        sorted(atoms.values(), key=lambda atom: str(atom["atom_id"]))
    )
    accepted_digests = {digest, legacy_chain_digest, legacy_canonical_digest}
    if (
        legacy_checkpoint is not None
        and legacy_checkpoint.get("content_sha256") == metadata.get("content_sha256")
        and legacy_checkpoint.get("atom_count") == len(records)
    ):
        accepted_digests.add(str(metadata["content_sha256"]))
    if metadata.get("content_sha256") not in accepted_digests:
        raise CatalogError("historical index content digest conflicts")
    return atoms, digest


def _resolved_atoms(
    raw_dir: Path, pending: Mapping[str, Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    texts_by_hash = _resolve_rows(
        raw_dir, (atom["catalog_row"] for atom in pending.values())
    )
    atoms: dict[str, dict[str, Any]] = {}
    for atom_id, atom in pending.items():
        text = texts_by_hash.get(str(atom["text_sha256"]))
        if text is None:
            raise CatalogError("catalog assistant text is unavailable")
        atoms[atom_id] = {
            key: value for key, value in atom.items() if key != "catalog_row"
        } | {"text": text}
    return atoms


def _index_digest(atoms: Mapping[str, Mapping[str, Any]]) -> str:
    """Build an order-independent checksum for the unique assistant atom set."""

    return _advance_index_digest(
        hashlib.sha256(f"{HISTORICAL_INDEX_DIGEST_SCHEMA}\0seed".encode()).hexdigest(),
        atoms.values(),
    )


def _advance_index_digest(previous: str, added: Iterable[Mapping[str, Any]]) -> str:
    if len(previous) != 64:
        raise CatalogError("historical index digest is invalid")
    try:
        state = int(previous, 16)
    except ValueError as exc:
        raise CatalogError("historical index digest is invalid") from exc
    for atom in added:
        term = hashlib.sha256(
            f"{HISTORICAL_INDEX_DIGEST_SCHEMA}\0atom\0{atom['atom_id']}".encode()
        ).digest()
        state = (state + int.from_bytes(term, "big")) % (1 << 256)
    return f"{state:064x}"


def _legacy_index_digest(atoms: Mapping[str, Mapping[str, Any]]) -> str:
    state = bytes.fromhex(
        hashlib.sha256(LEGACY_HISTORICAL_INDEX_DIGEST_SCHEMA.encode()).hexdigest()
    )
    for atom in sorted(atoms.values(), key=lambda value: int(value["catalog_rowid"])):
        state = hashlib.sha256(state + str(atom["atom_id"]).encode() + b"\0").digest()
    return state.hex()


def sync_historical_index(raw_dir: Path, root: Path) -> str:
    """Incrementally synchronize the existing assistant FTS from the catalog."""

    watermark = _catalog_watermark(raw_dir, root)
    catalog_checkpoint = _read_catalog_checkpoint(root)
    if (
        catalog_checkpoint is None
        or catalog_checkpoint["catalog_watermark"] != watermark
        or _catalog_lineage(catalog_checkpoint) is None
    ):
        raise CatalogError("catalog checkpoint requires repair")
    path = historical_index_path(root)
    checkpoint = _read_index_checkpoint(path)
    legacy_checkpoint = (
        _read_index_checkpoint(
            path,
            content_digest_schema=LEGACY_HISTORICAL_INDEX_DIGEST_SCHEMA,
        )
        if checkpoint is None and path.exists()
        else None
    )
    checkpoint_for_validation = checkpoint or legacy_checkpoint
    if (
        checkpoint is not None
        and checkpoint["catalog_watermark"] == watermark
        and checkpoint["catalog_file_state"] == catalog_checkpoint["file_state"]
        and checkpoint["catalog_event_rowid"] == catalog_checkpoint["event_rowid"]
        and _catalog_lineage(checkpoint) == _catalog_lineage(catalog_checkpoint)
    ):
        return str(checkpoint["content_sha256"])
    if not path.exists():
        expected = _catalog_assistant_atoms(root)
        atoms = _resolved_atoms(raw_dir, expected)
        store.create_historical_index(path, atoms.values())
        digest = _index_digest(atoms)
        try:
            with sqlite3.connect(path) as writable:
                writable.execute(
                    "UPDATE metadata SET value=? WHERE key='content_sha256'", (digest,)
                )
        except sqlite3.DatabaseError as exc:
            raise CatalogError("historical index bootstrap is unavailable") from exc
        try:
            with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as readonly:
                _atoms, verified = _index_atoms(readonly, expected)
        except sqlite3.DatabaseError as exc:
            raise CatalogError("historical index bootstrap is unreadable") from exc
        if digest != verified:
            raise CatalogError("historical index bootstrap digest conflicts")
        _ensure_private_file(path)
        _write_index_checkpoint(path, catalog_checkpoint, verified, len(atoms))
        return verified

    if (
        checkpoint is not None
        and checkpoint["catalog_event_rowid"] <= catalog_checkpoint["event_rowid"]
        and _catalog_lineage(checkpoint) == _catalog_lineage(catalog_checkpoint)
    ):
        pending = _catalog_assistant_atoms(
            root, after_rowid=int(checkpoint["catalog_event_rowid"])
        )
        if not pending:
            _write_index_checkpoint(
                path,
                catalog_checkpoint,
                str(checkpoint["content_sha256"]),
                int(checkpoint["atom_count"]),
            )
            return str(checkpoint["content_sha256"])
        connection: sqlite3.Connection | None = None
        try:
            connection = sqlite3.connect(path)
            connection.execute("PRAGMA journal_mode=DELETE")
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute("BEGIN IMMEDIATE")
            added = _resolved_atoms(raw_dir, pending)
            digest = _advance_index_digest(
                str(checkpoint["content_sha256"]), added.values()
            )
            for atom in added.values():
                cursor = connection.execute(
                    """INSERT INTO atoms(
                        atom_id,host,session_cluster_id,source_index,timestamp_us,
                        text_sha256,ref_json,text
                    ) VALUES(?,?,?,?,?,?,?,?)""",
                    (
                        atom["atom_id"],
                        atom["host"],
                        atom["session_cluster_id"],
                        atom["source_index"],
                        atom["timestamp_us"],
                        atom["text_sha256"],
                        json.dumps(atom["ref"], sort_keys=True, separators=(",", ":")),
                        atom["text"],
                    ),
                )
                connection.execute(
                    "INSERT INTO atoms_fts(rowid,search_text) VALUES(?,?)",
                    (cursor.lastrowid, store._search_terms(str(atom["text"]))),
                )
            connection.execute(
                "UPDATE metadata SET value=? WHERE key='content_sha256'", (digest,)
            )
            connection.commit()
        except sqlite3.DatabaseError as exc:
            if connection is not None:
                connection.rollback()
            raise CatalogError("historical index update failed") from exc
        finally:
            if connection is not None:
                connection.close()
        _ensure_private_file(path)
        _write_index_checkpoint(
            path,
            catalog_checkpoint,
            digest,
            int(checkpoint["atom_count"]) + len(added),
        )
        return digest

    # Missing or altered index state cannot be safely incremented.  This is
    # the only existing-index path allowed to inspect every atom.
    expected = _catalog_assistant_atoms(root)
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(path)
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("BEGIN IMMEDIATE")
        indexed, digest = _index_atoms(
            connection, expected, legacy_checkpoint=checkpoint_for_validation
        )
        pending = {
            atom_id: atom
            for atom_id, atom in expected.items()
            if atom_id not in indexed
        }
        if not pending:
            connection.execute(
                "UPDATE metadata SET value=? WHERE key='content_sha256'", (digest,)
            )
            connection.commit()
            connection.close()
            connection = None
            _ensure_private_file(path)
            _write_index_checkpoint(path, catalog_checkpoint, digest, len(expected))
            return digest
        added = _resolved_atoms(raw_dir, pending)
        digest = _index_digest(expected)
        for atom in added.values():
            cursor = connection.execute(
                """INSERT INTO atoms(
                    atom_id,host,session_cluster_id,source_index,timestamp_us,
                    text_sha256,ref_json,text
                ) VALUES(?,?,?,?,?,?,?,?)""",
                (
                    atom["atom_id"],
                    atom["host"],
                    atom["session_cluster_id"],
                    atom["source_index"],
                    atom["timestamp_us"],
                    atom["text_sha256"],
                    json.dumps(atom["ref"], sort_keys=True, separators=(",", ":")),
                    atom["text"],
                ),
            )
            connection.execute(
                "INSERT INTO atoms_fts(rowid,search_text) VALUES(?,?)",
                (cursor.lastrowid, store._search_terms(str(atom["text"]))),
            )
        connection.execute(
            "UPDATE metadata SET value=? WHERE key='content_sha256'", (digest,)
        )
        connection.commit()
    except sqlite3.DatabaseError as exc:
        if connection is not None:
            connection.rollback()
        raise CatalogError("historical index update failed") from exc
    finally:
        if connection is not None:
            connection.close()
    _ensure_private_file(path)
    _write_index_checkpoint(path, catalog_checkpoint, digest, len(expected))
    return digest
