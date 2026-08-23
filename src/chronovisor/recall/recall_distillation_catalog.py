"""Incremental, metadata-only catalog for Recall distillation Raw v2."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from chronovisor.core.canonical_json import canonical_json_sha256_strict
from chronovisor.core.raw_store import (
    RawSegmentCorrupt,
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


class CatalogError(ValueError):
    """The derived catalog cannot safely represent the committed Raw inventory."""


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


def _write_index_checkpoint(path: Path, watermark: str, digest: str, count: int) -> None:
    store.write_sealed_state(
        _index_checkpoint_path(path),
        {
            "kind": HISTORICAL_INDEX_CHECKPOINT_KIND,
            "index_name": path.name,
            "historical_index_schema": HISTORICAL_INDEX_SCHEMA,
            "catalog_watermark": watermark,
            "content_sha256": digest,
            "atom_count": count,
            "file_state": _index_file_state(path),
        },
    )


def _read_index_checkpoint(path: Path, watermark: str) -> dict[str, Any] | None:
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
        or checkpoint.get("catalog_watermark") != watermark
        or not isinstance(checkpoint.get("content_sha256"), str)
        or not isinstance(checkpoint.get("atom_count"), int)
        or isinstance(checkpoint.get("atom_count"), bool)
        or checkpoint["atom_count"] < 0
        or checkpoint.get("file_state") != _index_file_state(path)
    ):
        return None
    return checkpoint


def _connect(root: Path) -> sqlite3.Connection:
    path = catalog_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
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
            status TEXT NOT NULL
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
        """
    )
    return connection


def _unit_identity(unit: RawUnit) -> tuple[str, str]:
    commit = unit.commit
    if commit is None or unit.sha256 is None or unit.captured_at is None:
        raise CatalogError("Raw v2 unit has no committed receipt")
    return unit.sha256, canonical_json_sha256_strict(commit.to_dict())


def _event_row(unit: RawUnit, event: Mapping[str, Any], *, start: int, end: int, index: int) -> dict[str, Any] | None:
    commit = unit.commit
    raw_sha256, receipt_sha256 = _unit_identity(unit)
    assert commit is not None
    role, text = distill._event_semantics(commit.host, event)
    if role not in {"user", "assistant", "tool"}:
        return None
    timestamp, timestamp_us = distill._timestamp(event.get("timestamp"), commit.captured_at)
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


def _read_unit_events(raw_store: RawStore, unit: RawUnit) -> tuple[str, list[dict[str, Any]]]:
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
        row = _event_row(unit, event, start=start, end=start + len(encoded), index=index)
        if row is not None:
            rows.append(row)
    return "indexed", rows


def _store_unit(
    connection: sqlite3.Connection, unit: RawUnit, *, status: str
) -> None:
    raw_sha256, receipt_sha256 = _unit_identity(unit)
    assert unit.commit is not None and unit.captured_at is not None
    connection.execute(
        """INSERT INTO raw_units VALUES(?,?,?,?,?,?,?,?)""",
        (
            unit.raw_id,
            raw_sha256,
            receipt_sha256,
            unit.commit.host,
            unit.commit.session_key,
            unit.captured_at,
            unit.commit.record_count,
            status,
        ),
    )


def _store_events(connection: sqlite3.Connection, rows: Iterable[Mapping[str, Any]]) -> None:
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


def _store_rallies(connection: sqlite3.Connection, rows: Iterable[Mapping[str, Any]]) -> tuple[str, ...]:
    values = sorted(rows, key=lambda row: str(row["rally_id"]))
    encoded: dict[str, tuple[int, str]] = {}
    for row in values:
        rally_id = str(row["rally_id"])
        value = (int(row["as_of_us"]), json.dumps(row, sort_keys=True, separators=(",", ":")))
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
        **{key: row[key] for key in (
            "raw_id", "raw_sha256", "receipt_sha256", "event_index", "host",
            "session_key", "session_cluster_id", "session_id_sha256", "source_index",
            "byte_start", "byte_end", "timestamp", "timestamp_us", "role",
            "semantic_sha256", "structural",
        )},
        "text_bytes": len(text.encode("utf-8")),
        "nonempty": bool(text.strip()),
        "prompt_hash": distill._prompt_hash(text) if row["role"] == "user" else None,
    }


def _resolve_rows(
    raw_dir: Path, rows: Iterable[Mapping[str, Any]]
) -> dict[str, str]:
    """Resolve only the verified Raw units referenced by ``rows``."""

    raw_store = RawStore(raw_dir, mode="v2")
    resolved: dict[str, str] = {}
    by_raw: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_raw[str(row["raw_id"])].append(row)
    for raw_id, event_rows in by_raw.items():
        unit = raw_store.resolve_segment(raw_id)
        if unit is None:
            raise CatalogError("catalog Raw is unavailable")
        raw_sha256, receipt_sha256 = _unit_identity(unit)
        if any(
            row["raw_sha256"] != raw_sha256 or row["receipt_sha256"] != receipt_sha256
            for row in event_rows
        ):
            raise CatalogError("catalog Raw digest conflicts with source")
        try:
            raw = raw_store.read_bytes(unit)
            assert unit.commit is not None
            spans = committed_event_spans(raw, unit.commit.record_count)
        except (RawSegmentCorrupt, OSError) as exc:
            raise CatalogError("catalog Raw cannot be read") from exc
        for row in event_rows:
            index = int(row["event_index"])
            if index >= len(spans):
                raise CatalogError("catalog event index is invalid")
            start, encoded = spans[index]
            if start != row["byte_start"] or start + len(encoded) != row["byte_end"]:
                raise CatalogError("catalog event range conflicts with source")
            try:
                event = json.loads(encoded)
                role, text = distill._event_semantics(unit.commit.host, event)
            except (UnicodeError, json.JSONDecodeError, distill.DistillationError) as exc:
                raise CatalogError("catalog event cannot be decoded") from exc
            if role != row["role"] or hashlib.sha256(text.encode()).hexdigest() != row["semantic_sha256"]:
                raise CatalogError("catalog event semantics conflict with source")
            resolved[str(row["semantic_sha256"])] = text
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
    connection = _connect(root)
    try:
        connection.execute("BEGIN IMMEDIATE")
        metadata = dict(connection.execute("SELECT key,value FROM metadata"))
        schema = metadata.get("schema")
        if schema not in {None, CATALOG_SCHEMA}:
            raise CatalogError("catalog schema mismatch")
        stored = {
            str(row["raw_id"]): row
            for row in connection.execute(
                "SELECT raw_id,raw_sha256,receipt_sha256 FROM raw_units"
            )
        }
        if schema is None and stored:
            raise CatalogError("catalog metadata is missing")
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
            status = "bootstrap"
        else:
            existing_sessions = {
                (str(row[0]), str(row[1]))
                for row in connection.execute("SELECT DISTINCT host,session_key FROM events")
            }
            delta: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
            deferred_set: set[tuple[str, str]] = set()
            for unit in new_units:
                unit_status, rows = _read_unit_events(raw_store, unit)
                _store_unit(connection, unit, status=unit_status)
                _store_events(connection, rows)
                (archived if unit_status == "archived" else indexed).append(unit.raw_id)
                for row in rows:
                    key = (str(row["host"]), str(row["session_key"]))
                    if key in existing_sessions:
                        deferred_set.add(key)
                    else:
                        delta[key].append(row)
            rally_rows = []
            for key, rows in sorted(delta.items()):
                if key in deferred_set:
                    continue
                try:
                    rally_rows.extend(
                        distill.extract_rallies(
                            raw_dir,
                            root=root,
                            max_context_bytes=max_context_bytes,
                            _event_rows=sorted(rows, key=lambda row: int(row["source_index"])),
                        )
                    )
                except distill.DistillationError as exc:
                    raise CatalogError("cannot derive delta rallies") from exc
            for host, session_key in sorted(deferred_set):
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
                    raise CatalogError("cannot rebuild existing session tail") from exc
            rally_ids = _store_rallies(connection, rally_rows)
            deferred = tuple(sorted(deferred_set))
            status = "advanced"
        connection.executemany(
            "INSERT OR REPLACE INTO metadata VALUES(?,?)",
            (("schema", CATALOG_SCHEMA), ("watermark", watermark)),
        )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
    catalog_path(root).chmod(0o600)
    return CatalogAdvance(
        status,
        watermark,
        tuple(sorted(indexed)),
        tuple(sorted(archived)),
        rally_ids,
        deferred,
    )


def rallies(root: Path, ids: Iterable[str] | None = None) -> list[dict[str, Any]]:
    """Read text-free Rally manifests already derived by :func:`advance`."""

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
            rows = connection.execute("SELECT row_json FROM rallies ORDER BY as_of_us,rally_id")
        return [json.loads(row[0]) for row in rows]


def texts(
    raw_dir: Path,
    root: Path,
    hashes: Iterable[str] | None = None,
    refs: Iterable[Mapping[str, Any]] | None = None,
) -> dict[str, str]:
    """Resolve requested catalog text from verified Raw bytes on demand."""

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
    return _resolve_rows(raw_dir, rows.values())


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


def _catalog_assistant_atoms(root: Path) -> dict[str, dict[str, Any]]:
    """Read catalog assistant metadata after the small watermark preflight."""

    path = catalog_path(root)
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
            connection.row_factory = sqlite3.Row
            rows = list(
                connection.execute(
                    "SELECT * FROM events WHERE role='assistant' AND nonempty=1 "
                    "ORDER BY raw_id,event_index"
                )
            )
    except sqlite3.DatabaseError as exc:
        raise CatalogError("historical catalog is unreadable") from exc
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
        }
        if atom_id in atoms:
            raise CatalogError("catalog assistant atom conflicts")
        atoms[atom_id] = atom
    return atoms


def _index_atoms(
    connection: sqlite3.Connection, expected: Mapping[str, Mapping[str, Any]]
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
        rowid, atom_id, host, cluster, source_index, timestamp_us, text_sha256, ref_json, text = record
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
        if not isinstance(text, str) or hashlib.sha256(text.encode()).hexdigest() != text_sha256:
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
    digest = canonical_json_sha256_strict(
        sorted(atoms.values(), key=lambda atom: str(atom["atom_id"]))
    )
    if metadata.get("content_sha256") != digest:
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
        atoms[atom_id] = {key: value for key, value in atom.items() if key != "catalog_row"} | {
            "text": text
        }
    return atoms


def _index_digest(atoms: Mapping[str, Mapping[str, Any]]) -> str:
    return canonical_json_sha256_strict(
        sorted(atoms.values(), key=lambda atom: str(atom["atom_id"]))
    )


def sync_historical_index(raw_dir: Path, root: Path) -> str:
    """Incrementally synchronize the existing assistant FTS from the catalog."""

    watermark = _catalog_watermark(raw_dir, root)
    path = historical_index_path(root)
    checkpoint = _read_index_checkpoint(path, watermark)
    if checkpoint is not None:
        return str(checkpoint["content_sha256"])
    expected = _catalog_assistant_atoms(root)
    if not path.exists():
        atoms = _resolved_atoms(raw_dir, expected)
        digest = store.create_historical_index(path, atoms.values())
        try:
            with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
                _atoms, verified = _index_atoms(connection, expected)
        except sqlite3.DatabaseError as exc:
            raise CatalogError("historical index bootstrap is unreadable") from exc
        if digest != verified:
            raise CatalogError("historical index bootstrap digest conflicts")
        _write_index_checkpoint(path, watermark, verified, len(atoms))
        return verified
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(path)
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("BEGIN IMMEDIATE")
        indexed, digest = _index_atoms(connection, expected)
        pending = {atom_id: atom for atom_id, atom in expected.items() if atom_id not in indexed}
        if not pending:
            connection.rollback()
            connection.close()
            connection = None
            _write_index_checkpoint(path, watermark, digest, len(expected))
            return digest
        added = _resolved_atoms(raw_dir, pending)
        all_atoms = indexed | added
        digest = _index_digest(all_atoms)
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
    path.chmod(0o600)
    _write_index_checkpoint(path, watermark, digest, len(expected))
    return digest
