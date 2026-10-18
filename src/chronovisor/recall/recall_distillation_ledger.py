"""Verified append-only ledger projections for bounded operational readers."""

from __future__ import annotations

import json
import sqlite3
import stat
import threading
from collections.abc import Iterator, Mapping
from contextlib import closing
from pathlib import Path
from typing import Any

from chronovisor.recall import recall_distillation_store as store

_VERSION = "recall-ledger-projection-v2"
_FIELDS = frozenset({
    "record_sha256", "kind", "rally_id", "candidate_id", "receipt_sha256",
    "work_id", "profile", "profile_contract_id", "authority", "probe",
    "dimension", "verdict", "closed_predicate",
})
_CACHE: dict[Path, LedgerIndex] = {}
_CACHE_LOCK = threading.RLock()


def file_state(path: Path) -> dict[str, int] | None:
    """Reject aliases and bind a derived file to its complete physical state."""
    try:
        value = path.lstat()
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(value.st_mode):
        raise store.DistillationStoreError("ledger projection source is not regular")
    return {
        "st_dev": value.st_dev, "st_ino": value.st_ino,
        "size_bytes": value.st_size, "st_mtime_ns": value.st_mtime_ns,
        "st_ctime_ns": value.st_ctime_ns,
    }


def configure_writer(connection: sqlite3.Connection) -> None:
    """Keep rolled-back work from invalidating a sealed physical checkpoint."""

    connection.execute("PRAGMA journal_mode=DELETE")
    connection.execute("PRAGMA synchronous=FULL")
    # Default spill writes uncommitted pages to the main file. Rollback then
    # restores bytes but changes mtime/ctime, discarding verified prior work.
    connection.execute("PRAGMA cache_spill=OFF")


def connect(path: Path, *, readonly: bool = False) -> sqlite3.Connection:
    if readonly:
        connection = sqlite3.connect(
            path.absolute().as_uri() + "?mode=ro", uri=True, check_same_thread=False,
        )
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            file_state(path)
        connection = sqlite3.connect(path)
        if stat.S_IMODE(path.stat().st_mode) != 0o600:
            path.chmod(0o600)
        configure_writer(connection)
    connection.row_factory = sqlite3.Row
    return connection


def checkpoint(path: Path) -> dict[str, Any] | None:
    try:
        value = store.read_sealed(
            path.with_suffix(path.suffix + ".checkpoint.json"),
            schema=store.DISTILLATION_SCHEMA,
        )
        current = file_state(path)
        if current is None or value.get("file_state") != current:
            return None
        return value
    except (OSError, store.DistillationStoreError):
        return None


def seal_checkpoint(path: Path, **payload: Any) -> None:
    store.write_sealed_state(
        path.with_suffix(path.suffix + ".checkpoint.json"),
        {"file_state": file_state(path), **payload},
    )


def _row_values(sequence: int, row: Mapping[str, Any]) -> tuple[Any, ...]:
    manifest = row.get("manifest")
    rally_id = (
        manifest.get("rally_id") if isinstance(manifest, Mapping)
        else row.get("rally_id")
    )
    assignment = row.get("assignment")
    return (
        sequence, str(row["record_sha256"]), str(row.get("kind") or ""),
        str(rally_id or ""), str(row.get("candidate_id") or ""),
        str(row.get("receipt_sha256") or row.get("provider_receipt_sha256") or ""),
        str(row.get("work_id") or row.get("task_id") or ""),
        str(row.get("profile") or ""), str(row.get("profile_contract_id") or ""),
        str(row.get("authority") or ""),
        str(row.get("dimension") or ""), str(row.get("verdict") or ""),
        str(row.get("closed_predicate") or ""),
        int(isinstance(assignment, Mapping) and assignment.get("probe") is True),
        store.projection_json(dict(row)),
    )


def _schema(connection: sqlite3.Connection) -> None:
    connection.executescript("""
        CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS rows(
            sequence INTEGER PRIMARY KEY, record_sha256 TEXT NOT NULL UNIQUE,
            kind TEXT NOT NULL, rally_id TEXT NOT NULL, candidate_id TEXT NOT NULL,
            receipt_sha256 TEXT NOT NULL, work_id TEXT NOT NULL,
            profile TEXT NOT NULL, profile_contract_id TEXT NOT NULL,
            authority TEXT NOT NULL, dimension TEXT NOT NULL, verdict TEXT NOT NULL,
            closed_predicate TEXT NOT NULL, probe INTEGER NOT NULL, row_json TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS rows_rally ON rows(rally_id,sequence);
        CREATE INDEX IF NOT EXISTS rows_kind ON rows(kind,sequence);
        CREATE INDEX IF NOT EXISTS rows_receipt ON rows(receipt_sha256,sequence);
        CREATE INDEX IF NOT EXISTS rows_work ON rows(work_id,sequence);
        CREATE INDEX IF NOT EXISTS rows_profile ON rows(profile,profile_contract_id);
        CREATE INDEX IF NOT EXISTS rows_authority_verdict ON rows(authority,verdict);
        CREATE INDEX IF NOT EXISTS rows_structural ON rows(authority,dimension,closed_predicate);
    """)


def _sync(path: Path, index_path: Path) -> dict[str, Any]:
    with store._locked(index_path.with_suffix(index_path.suffix + ".lock")):
        saved = checkpoint(index_path)
        cursor = None
        if saved is not None and saved.get("kind") == _VERSION:
            try:
                with closing(connect(index_path, readonly=True)) as connection:
                    metadata = dict(connection.execute("SELECT key,value FROM metadata"))
                    cursor = json.loads(metadata["cursor"])
                    if cursor != saved.get("cursor"):
                        cursor = None
            except (sqlite3.Error, ValueError, KeyError):
                cursor = None
        additions, current, reset = store.read_chain_tail(path, cursor)
        if not reset and cursor == current:
            return current
        with closing(connect(index_path)) as connection:
            if reset:
                connection.executescript("DROP TABLE IF EXISTS rows; DROP TABLE IF EXISTS metadata;")
            _schema(connection)
            with connection:
                start = 0 if reset or cursor is None else int(cursor["records"])
                connection.executemany(
                    "INSERT INTO rows VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (_row_values(start + index + 1, row) for index, row in enumerate(additions)),
                )
                connection.execute(
                    "INSERT OR REPLACE INTO metadata VALUES ('cursor',?)",
                    (store.projection_json(current),),
                )
        seal_checkpoint(index_path, kind=_VERSION, cursor=current)
        return current


class LedgerIndex:
    """A pinned, verified source view; a change invalidates the entire view."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.index_path = path.with_suffix(path.suffix + ".projection.sqlite3")
        self.cursor = _sync(path, self.index_path)
        self._state = self.signature()
        self._connection = connect(self.index_path, readonly=True)
        self._lock = threading.RLock()
        saved = checkpoint(self.index_path)
        if (
            self._state[0] != self.cursor["file_state"]
            or saved is None
            or saved.get("cursor") != self.cursor
            or self.signature() != self._state
        ):
            self.close()
            raise store.DistillationStoreError("ledger projection changed while opening")

    def signature(self) -> tuple[Any, ...]:
        return (
            file_state(self.path), file_state(store._chain_checkpoint_path(self.path)),
            file_state(self.index_path),
            file_state(self.index_path.with_suffix(self.index_path.suffix + ".checkpoint.json")),
        )

    def current(self) -> bool:
        return self.signature() == self._state

    def close(self) -> None:
        self._connection.close()

    def query(self, sql: str, parameters: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
        with self._lock:
            if not self.current():
                raise store.DistillationStoreError("ledger projection source changed")
            rows = self._connection.execute(sql, parameters).fetchall()
            if not self.current():
                raise store.DistillationStoreError("ledger projection changed during read")
            return rows

    def rows(self, *, after: int = 0, kind: str | None = None) -> list[dict[str, Any]]:
        sql = "SELECT row_json FROM rows WHERE sequence>?"
        parameters: tuple[Any, ...] = (after,)
        if kind is not None:
            sql += " AND kind=?"
            parameters += (kind,)
        return [json.loads(row[0]) for row in self.query(sql + " ORDER BY sequence", parameters)]

    def distinct(self, field: str) -> set[str]:
        if field not in _FIELDS:
            raise ValueError("unknown ledger projection field")
        return {str(row[0]) for row in self.query(f"SELECT DISTINCT {field} FROM rows")}

    def lookup(self, field: str, key: str) -> dict[str, Any] | None:
        if field not in _FIELDS:
            raise ValueError("unknown ledger projection field")
        rows = self.query(
            f"SELECT row_json FROM rows WHERE {field}=? ORDER BY sequence DESC LIMIT 1",
            (key,),
        )
        return json.loads(rows[0][0]) if rows else None


class LedgerMapping(Mapping[str, dict[str, Any]]):
    def __init__(self, index: LedgerIndex, field: str) -> None:
        if field not in _FIELDS:
            raise ValueError("unknown ledger projection field")
        self.index = index
        self.field = field

    def __getitem__(self, key: str) -> dict[str, Any]:
        result = self.index.lookup(self.field, key)
        if result is None:
            raise KeyError(key)
        return result

    def __iter__(self) -> Iterator[str]:
        return iter(self.index.distinct(self.field))

    def __len__(self) -> int:
        return int(self.index.query(f"SELECT COUNT(DISTINCT {self.field}) FROM rows")[0][0])


def ledger_index(path: Path) -> LedgerIndex:
    path = path.expanduser().absolute()
    with _CACHE_LOCK:
        cached = _CACHE.get(path)
        if cached is not None:
            if cached.current():
                return cached
            del _CACHE[path]
        value = LedgerIndex(path)
        if len(_CACHE) >= 16:
            # Existing Mapping views retain their own connection and source
            # guard until their callers release them.
            _CACHE.pop(next(iter(_CACHE)))
        _CACHE[path] = value
        return value


def clear_cache() -> None:
    with _CACHE_LOCK:
        for value in _CACHE.values():
            value.close()
        _CACHE.clear()
