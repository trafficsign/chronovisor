"""Private durable storage for autonomous Recall distillation."""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import sqlite3
import tempfile
import unicodedata
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any, BinaryIO

from chronovisor.core.canonical_json import (
    canonical_json_bytes_strict,
    canonical_json_sha256_strict,
)
from chronovisor.core.store import CHRONOVISOR_ROOT

DISTILLATION_SCHEMA = "chronovisor.recall-distillation.v1"
DISTILLATION_DIR = CHRONOVISOR_ROOT / "runtime" / "recall-distillation"
STATE_FILE = "state.json"
POINTER_FILES = {
    "active": "active-policy.json",
    "candidate": "candidate-policy.json",
    "lkg": "lkg-policy.json",
}


class DistillationStoreError(ValueError):
    """A distillation artifact is malformed, unsealed, or conflicting."""


class DistillationStoreBusy(RuntimeError):
    """A nonblocking private-store write could not acquire its lock."""


def _validate_artifact_id(value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or Path(value).name != value
        or "/" in value
        or "\\" in value
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise DistillationStoreError("invalid artifact id")
    return value


def _reject_reserved(payload: Mapping[str, Any], reserved: set[str]) -> None:
    conflict = reserved.intersection(payload)
    if conflict:
        raise DistillationStoreError(
            f"reserved artifact fields are caller-controlled: {sorted(conflict)}"
        )


def _require_chain_metadata(row: Mapping[str, Any], index: int) -> None:
    if row.get("schema") != DISTILLATION_SCHEMA:
        raise DistillationStoreError(f"ledger row {index} schema mismatch")
    if row.get("namespace") != "recall-distillation":
        raise DistillationStoreError(f"ledger row {index} namespace mismatch")


def distillation_dir(root: Path | None = None) -> Path:
    return (root or CHRONOVISOR_ROOT) / "runtime" / "recall-distillation"


def _sealed(payload: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = dict(payload)
    unsigned.pop("seal_sha256", None)
    return {**unsigned, "seal_sha256": canonical_json_sha256_strict(unsigned)}


def verify_seal(payload: object, *, schema: str | None = None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise DistillationStoreError("artifact is not an object")
    unsigned = {key: value for key, value in payload.items() if key != "seal_sha256"}
    if payload.get("seal_sha256") != canonical_json_sha256_strict(unsigned):
        raise DistillationStoreError("artifact seal mismatch")
    if schema is not None and payload.get("schema") != schema:
        raise DistillationStoreError("artifact schema mismatch")
    if payload.get("namespace") != "recall-distillation":
        raise DistillationStoreError("artifact namespace mismatch")
    return payload


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            temporary = Path(handle.name)
            os.fchmod(handle.fileno(), 0o600)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_sealed_state(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    _reject_reserved(payload, {"schema", "namespace", "seal_sha256"})
    artifact = _sealed(
        {
            "schema": DISTILLATION_SCHEMA,
            "namespace": "recall-distillation",
            **payload,
        }
    )
    _atomic_write(path, canonical_json_bytes_strict(artifact) + b"\n")
    return verify_seal(json.loads(path.read_bytes()), schema=DISTILLATION_SCHEMA)


def read_sealed(path: Path, *, schema: str | None = None) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError, UnicodeError) as exc:
        raise DistillationStoreError(
            f"cannot read sealed artifact: {path.name}"
        ) from exc
    return verify_seal(payload, schema=schema)


def write_immutable(
    directory: Path,
    payload: Mapping[str, Any],
    *,
    schema: str,
    artifact_id: str | None = None,
    nonblocking: bool = False,
) -> tuple[str, Path, dict[str, Any]]:
    _reject_reserved(payload, {"schema", "namespace", "artifact_id", "seal_sha256"})
    unsigned = {
        "schema": schema,
        "namespace": "recall-distillation",
        **payload,
    }
    identity = _validate_artifact_id(
        artifact_id or canonical_json_sha256_strict(unsigned)
    )
    artifact = _sealed({"artifact_id": identity, **unsigned})
    encoded = canonical_json_bytes_strict(artifact) + b"\n"
    path = directory / f"{identity}.json"
    directory.mkdir(parents=True, exist_ok=True)

    def persist() -> None:
        if path.exists():
            if path.read_bytes() != encoded:
                raise DistillationStoreError("immutable artifact conflict")
        else:
            _atomic_write(path, encoded)

    if nonblocking:
        lock = acquire_nonblocking_lock(directory / ".immutable.lock")
        if lock is None:
            raise DistillationStoreBusy("immutable artifact store is busy")
        try:
            persist()
        finally:
            release_lock(lock)
    else:
        with _locked(directory / ".immutable.lock"):
            persist()
    return identity, path, verify_seal(artifact, schema=schema)


@contextlib.contextmanager
def _locked(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def verify_chain(path: Path) -> dict[str, Any]:
    previous = ""
    count = 0
    try:
        lines = path.read_bytes().splitlines()
    except FileNotFoundError:
        lines = []
    except OSError as exc:
        raise DistillationStoreError("ledger is unreadable") from exc
    for index, line in enumerate(lines):
        try:
            row = json.loads(line)
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise DistillationStoreError(f"ledger row {index} is invalid") from exc
        if not isinstance(row, dict):
            raise DistillationStoreError(f"ledger row {index} is not an object")
        _require_chain_metadata(row, index)
        digest = row.get("record_sha256")
        unsigned = {key: value for key, value in row.items() if key != "record_sha256"}
        if row.get(
            "previous_sha256"
        ) != previous or digest != canonical_json_sha256_strict(unsigned):
            raise DistillationStoreError(f"ledger chain mismatch at row {index}")
        previous = str(digest)
        count += 1
    return {"records": count, "head_sha256": previous}


def _chain_checkpoint_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".head.json")


def _ledger_file_state(path: Path) -> dict[str, int] | None:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise DistillationStoreError("ledger is unreadable") from exc
    return {
        "size_bytes": stat.st_size,
        "st_dev": stat.st_dev,
        "st_ino": stat.st_ino,
        "st_mtime_ns": stat.st_mtime_ns,
        "st_ctime_ns": stat.st_ctime_ns,
    }


def _write_chain_checkpoint(path: Path, head: Mapping[str, Any]) -> None:
    write_sealed_state(
        _chain_checkpoint_path(path),
        {
            "kind": "ledger-chain-checkpoint",
            "ledger_name": path.name,
            "records": head["records"],
            "head_sha256": head["head_sha256"],
            "file_state": _ledger_file_state(path),
        },
    )


def _read_chain_checkpoint(path: Path) -> dict[str, Any] | None:
    try:
        checkpoint = read_sealed(
            _chain_checkpoint_path(path), schema=DISTILLATION_SCHEMA
        )
    except DistillationStoreError:
        return None
    if (
        checkpoint.get("kind") != "ledger-chain-checkpoint"
        or checkpoint.get("ledger_name") != path.name
        or not isinstance(checkpoint.get("records"), int)
        or isinstance(checkpoint.get("records"), bool)
        or checkpoint["records"] < 0
        or not isinstance(checkpoint.get("head_sha256"), str)
        or (
            checkpoint["head_sha256"] != ""
            and (
                len(checkpoint["head_sha256"]) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in checkpoint["head_sha256"]
                )
            )
        )
        or (checkpoint["records"] == 0) != (checkpoint["head_sha256"] == "")
        or checkpoint.get("file_state") != _ledger_file_state(path)
    ):
        return None
    return {
        "records": checkpoint["records"],
        "head_sha256": checkpoint["head_sha256"],
    }


def _fsync_directory(path: Path) -> None:
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _recover_chain_tail(path: Path) -> dict[str, Any]:
    """Verify a ledger and discard only an interrupted final record."""

    previous = ""
    count = 0
    last_good_offset = 0
    try:
        with path.open("rb") as handle:
            for index, line in enumerate(iter(handle.readline, b"")):
                end_offset = handle.tell()
                try:
                    row = json.loads(line)
                except (UnicodeError, json.JSONDecodeError) as exc:
                    if not line.endswith(b"\n"):
                        with path.open("r+b") as writable:
                            writable.truncate(last_good_offset)
                            writable.flush()
                            os.fsync(writable.fileno())
                        _fsync_directory(path)
                        return {"records": count, "head_sha256": previous}
                    raise DistillationStoreError(
                        f"ledger row {index} is invalid"
                    ) from exc
                if not isinstance(row, dict):
                    raise DistillationStoreError(f"ledger row {index} is not an object")
                _require_chain_metadata(row, index)
                digest = row.get("record_sha256")
                unsigned = {
                    key: value for key, value in row.items() if key != "record_sha256"
                }
                if row.get(
                    "previous_sha256"
                ) != previous or digest != canonical_json_sha256_strict(unsigned):
                    if not line.endswith(b"\n"):
                        with path.open("r+b") as writable:
                            writable.truncate(last_good_offset)
                            writable.flush()
                            os.fsync(writable.fileno())
                        _fsync_directory(path)
                        return {"records": count, "head_sha256": previous}
                    raise DistillationStoreError(
                        f"ledger chain mismatch at row {index}"
                    )
                previous = str(digest)
                count += 1
                last_good_offset = end_offset
    except FileNotFoundError:
        return {"records": 0, "head_sha256": ""}
    except OSError as exc:
        raise DistillationStoreError("ledger is unreadable") from exc
    return {"records": count, "head_sha256": previous}


def _append_fsynced(path: Path, data: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    with os.fdopen(descriptor, "ab") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _chain_separator(path: Path) -> bytes:
    state = _ledger_file_state(path)
    if state is None or state["size_bytes"] == 0:
        return b""
    with path.open("rb") as handle:
        handle.seek(-1, os.SEEK_END)
        return b"" if handle.read(1) == b"\n" else b"\n"


def chain_head(path: Path) -> dict[str, Any]:
    """Return a verified ledger head without scanning a matching checkpoint."""

    lock_path = path.with_suffix(path.suffix + ".lock")
    with _locked(lock_path):
        head = _read_chain_checkpoint(path)
        if head is not None:
            return head
        head = _recover_chain_tail(path)
        _write_chain_checkpoint(path, head)
        return head


def append_chain(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    return append_chain_batch(path, (payload,))[0]


def append_chain_batch(
    path: Path, payloads: Iterable[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Append a bounded batch with a checkpoint and one data fsync."""

    values: list[dict[str, Any]] = []
    for index, payload in enumerate(payloads):
        if index >= 500:
            raise DistillationStoreError("ledger batch is too large")
        values.append(dict(payload))
    if not values:
        return []
    reserved = {
        "schema",
        "namespace",
        "previous_sha256",
        "record_sha256",
        "seal_sha256",
    }
    for payload in values:
        _reject_reserved(payload, reserved)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with _locked(lock_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        head = _read_chain_checkpoint(path) or _recover_chain_tail(path)
        # The checkpoint is a write-ahead recovery point for an interrupted append.
        _write_chain_checkpoint(path, head)
        previous = str(head["head_sha256"])
        rows: list[dict[str, Any]] = []
        encoded = bytearray()
        for payload in values:
            unsigned = {
                "schema": DISTILLATION_SCHEMA,
                "namespace": "recall-distillation",
                "previous_sha256": previous,
                **payload,
            }
            row = {
                **unsigned,
                "record_sha256": canonical_json_sha256_strict(unsigned),
            }
            rows.append(row)
            previous = row["record_sha256"]
            encoded.extend(canonical_json_bytes_strict(row) + b"\n")
        _append_fsynced(path, _chain_separator(path) + encoded)
        _write_chain_checkpoint(
            path,
            {"records": head["records"] + len(rows), "head_sha256": previous},
        )
        return rows


def acquire_nonblocking_lock(path: Path) -> BinaryIO | None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return None
    return handle


def release_lock(handle: BinaryIO) -> None:
    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    handle.close()


def append_chain_unique(
    path: Path,
    payload: Mapping[str, Any],
    *,
    unique_field: str,
    binding_field: str,
) -> dict[str, Any]:
    """Append once by a stable identity while holding the ledger lock."""

    lock_path = path.with_suffix(path.suffix + ".lock")
    with _locked(lock_path):
        return append_chain_unique_locked(
            path,
            payload,
            unique_field=unique_field,
            binding_field=binding_field,
        )


def append_chain_unique_locked(
    path: Path,
    payload: Mapping[str, Any],
    *,
    unique_field: str,
    binding_field: str,
) -> dict[str, Any]:
    """Append one unique row while the caller holds this chain's lock."""

    _reject_reserved(
        payload,
        {
            "schema",
            "namespace",
            "previous_sha256",
            "record_sha256",
            "seal_sha256",
        },
    )
    identity = payload.get(unique_field)
    binding = payload.get(binding_field)
    if not isinstance(identity, str) or not identity or not isinstance(binding, str):
        raise DistillationStoreError("unique ledger identity is invalid")
    path.parent.mkdir(parents=True, exist_ok=True)
    head = verify_chain(path)
    existing = path.read_bytes() if path.exists() else b""
    for line in existing.splitlines():
        row = json.loads(line)
        if not isinstance(row, dict):
            raise DistillationStoreError("ledger row is not an object")
        if row.get(unique_field) != identity:
            continue
        if row.get(binding_field) == binding:
            return row
        raise DistillationStoreError("unique ledger identity conflict")
    unsigned = {
        "schema": DISTILLATION_SCHEMA,
        "namespace": "recall-distillation",
        "previous_sha256": head["head_sha256"],
        **payload,
    }
    row = {**unsigned, "record_sha256": canonical_json_sha256_strict(unsigned)}
    _atomic_write(path, existing + canonical_json_bytes_strict(row) + b"\n")
    return row


def read_chain(path: Path) -> list[dict[str, Any]]:
    """Decode a stable ledger snapshot after checkpoint-backed verification."""

    lock_path = path.with_suffix(path.suffix + ".lock")
    with _locked(lock_path):
        head = _read_chain_checkpoint(path)
        if head is None:
            head = _recover_chain_tail(path)
            _write_chain_checkpoint(path, head)
        try:
            lines = path.read_bytes().splitlines()
        except FileNotFoundError:
            lines = []
        except OSError as exc:
            raise DistillationStoreError("ledger is unreadable") from exc
        rows: list[dict[str, Any]] = []
        for index, line in enumerate(lines):
            try:
                row = json.loads(line)
            except (UnicodeError, json.JSONDecodeError) as exc:
                raise DistillationStoreError(f"ledger row {index} is invalid") from exc
            if not isinstance(row, dict):
                raise DistillationStoreError(f"ledger row {index} is not an object")
            _require_chain_metadata(row, index)
            rows.append(row)
        if (
            len(rows) != head["records"]
            or (rows[-1].get("record_sha256") if rows else "") != head["head_sha256"]
        ):
            raise DistillationStoreError(
                "ledger checkpoint does not match decoded rows"
            )
        return rows


def write_pointer(
    root: Path, kind: str, policy_id: str, **metadata: Any
) -> dict[str, Any]:
    if kind not in POINTER_FILES:
        raise DistillationStoreError("unknown policy pointer kind")
    if (
        len(policy_id) != 64
        or any(character not in "0123456789abcdef" for character in policy_id)
        or Path(policy_id).name != policy_id
    ):
        raise DistillationStoreError("invalid policy id")
    return write_sealed_state(
        distillation_dir(root) / POINTER_FILES[kind],
        {"kind": f"{kind}-policy-pointer", "policy_id": policy_id, **metadata},
    )


def read_pointer(root: Path, kind: str) -> dict[str, Any]:
    if kind not in POINTER_FILES:
        raise DistillationStoreError("unknown policy pointer kind")
    return read_sealed(
        distillation_dir(root) / POINTER_FILES[kind], schema=DISTILLATION_SCHEMA
    )


def clear_pointer(root: Path, kind: str) -> None:
    if kind not in POINTER_FILES:
        raise DistillationStoreError("unknown policy pointer kind")
    path = distillation_dir(root) / POINTER_FILES[kind]
    path.unlink(missing_ok=True)


def create_historical_index(path: Path, atoms: Iterable[Mapping[str, Any]]) -> str:
    with _locked(path.with_suffix(path.suffix + ".lock")):
        return _create_historical_index_locked(path, atoms)


def _create_historical_index_locked(
    path: Path, atoms: Iterable[Mapping[str, Any]]
) -> str:
    """Build one deterministic assistant-only point-in-time FTS5 cache."""

    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted((dict(atom) for atom in atoms), key=lambda row: str(row["atom_id"]))
    rows_digest = canonical_json_sha256_strict(rows)
    if path.exists():
        try:
            with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
                metadata = dict(conn.execute("SELECT key,value FROM metadata"))
            if (
                metadata.get("schema") == "chronovisor.recall-historical-fts.v1"
                and metadata.get("content_sha256") == rows_digest
            ):
                path.chmod(0o600)
                return rows_digest
        except (OSError, sqlite3.DatabaseError):
            pass
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    os.fchmod(descriptor, 0o600)
    os.close(descriptor)
    try:
        with sqlite3.connect(temporary) as conn:
            conn.executescript(
                """
                PRAGMA journal_mode=DELETE;
                PRAGMA synchronous=FULL;
                CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE atoms(
                    atom_id TEXT PRIMARY KEY,
                    host TEXT NOT NULL,
                    session_cluster_id TEXT NOT NULL,
                    source_index INTEGER NOT NULL,
                    timestamp_us INTEGER NOT NULL,
                    text_sha256 TEXT NOT NULL,
                    ref_json TEXT NOT NULL,
                    text TEXT NOT NULL
                );
                CREATE VIRTUAL TABLE atoms_fts USING fts5(
                    search_text, tokenize='unicode61'
                );
                """
            )
            conn.execute(
                "INSERT INTO metadata VALUES (?, ?)",
                ("schema", "chronovisor.recall-historical-fts.v1"),
            )
            for row in rows:
                cursor = conn.execute(
                    """INSERT INTO atoms(
                        atom_id,host,session_cluster_id,source_index,timestamp_us,
                        text_sha256,ref_json,text
                    ) VALUES(?,?,?,?,?,?,?,?)""",
                    (
                        row["atom_id"],
                        row["host"],
                        row["session_cluster_id"],
                        int(row["source_index"]),
                        int(row["timestamp_us"]),
                        row["text_sha256"],
                        json.dumps(row["ref"], sort_keys=True, separators=(",", ":")),
                        row["text"],
                    ),
                )
                conn.execute(
                    "INSERT INTO atoms_fts(rowid,search_text) VALUES(?,?)",
                    (cursor.lastrowid, _search_terms(str(row["text"]))),
                )
            conn.execute(
                "INSERT INTO metadata VALUES (?, ?)",
                ("content_sha256", rows_digest),
            )
            conn.commit()
        temporary.chmod(0o600)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        path.chmod(0o600)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    return rows_digest


def search_historical_index(
    path: Path,
    *,
    query: str,
    as_of_us: int,
    host: str,
    session_cluster_id: str,
    source_index: int,
    limit: int,
) -> list[dict[str, Any]]:
    terms = _search_terms(query).split()[:128]
    if not terms or limit <= 0:
        return []
    expression = " OR ".join(f'"{term}"' for term in terms)
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
        rows = conn.execute(
            """SELECT a.atom_id,a.host,a.session_cluster_id,a.source_index,
                      a.timestamp_us,a.text_sha256,a.ref_json,bm25(atoms_fts) AS rank
               FROM atoms_fts JOIN atoms a ON a.rowid=atoms_fts.rowid
               WHERE atoms_fts MATCH ?
                 AND (a.timestamp_us < ? OR
                      (a.timestamp_us = ? AND a.host = ? AND
                       a.session_cluster_id = ? AND a.source_index < ?))
               ORDER BY rank,a.atom_id LIMIT ?""",
            (
                expression,
                as_of_us,
                as_of_us,
                host,
                session_cluster_id,
                source_index,
                limit,
            ),
        ).fetchall()
    return [
        {
            "candidate_id": row[0],
            "host": row[1],
            "session_cluster_id": row[2],
            "source_index": row[3],
            "timestamp_us": row[4],
            "text_sha256": row[5],
            "ref": json.loads(row[6]),
            "channel": "historical-fts-v1",
            "rank": rank,
        }
        for rank, row in enumerate(rows, start=1)
    ]


def _search_terms(text: str) -> str:
    """Use deterministic trigrams so whitespace-free Japanese remains searchable."""

    normalized = unicodedata.normalize("NFKC", text).casefold()
    tokens = [token for token in normalized.replace('"', " ").split() if token]
    terms: list[str] = []
    for token in tokens:
        compact = "".join(character for character in token if character.isalnum())
        if not compact:
            continue
        if len(compact) < 3:
            terms.append(compact)
        else:
            terms.extend(
                compact[index : index + 3] for index in range(len(compact) - 2)
            )
    return " ".join(terms)


def snapshot(chronovisor_root: Path) -> dict[str, Any]:
    """Return a privacy-safe, seal-verified health snapshot."""

    root = distillation_dir(chronovisor_root)
    try:
        state = read_sealed(root / STATE_FILE, schema=DISTILLATION_SCHEMA)
        pointers: dict[str, str] = {}
        for kind in POINTER_FILES:
            pointer_path = root / POINTER_FILES[kind]
            try:
                policy_id = str(read_pointer(chronovisor_root, kind)["policy_id"])
                if len(policy_id) != 64 or any(
                    character not in "0123456789abcdef" for character in policy_id
                ):
                    raise DistillationStoreError("pointer policy id is invalid")
                policy = read_sealed(
                    root / "policies" / f"{policy_id}.json",
                    schema="chronovisor.recall-distill-policy.v2",
                )
                if policy.get("artifact_id") != policy_id:
                    raise DistillationStoreError("pointer policy identity mismatch")
                pointers[kind] = policy_id[:12]
            except FileNotFoundError:
                pointers[kind] = ""
            except (DistillationStoreError, KeyError):
                if pointer_path.exists():
                    raise
                pointers[kind] = ""
        rollout_status = str(state.get("status") or "unknown")
        if rollout_status not in {
            "capture_only",
            "ready",
            "replay",
            "shadow",
            "canary",
            "active",
            "rolled_back",
            "quarantined",
        }:
            rollout_status = "unknown"
        worker_status = str(state.get("worker_status") or rollout_status)
        if worker_status not in {
            "disabled",
            "idle",
            "capture_only",
            "ready",
            "deferred",
        }:
            worker_status = rollout_status
        rollout = state.get("rollout_percent", 0)
        rollout = int(rollout) if isinstance(rollout, (int, float)) else 0
        error_code = str(state.get("error_code") or "")
        if error_code and not all(
            character.islower() or character.isdigit() or character in "_-"
            for character in error_code
        ):
            error_code = "invalid_error_code"
        labels = read_chain(root / "label-ledger.jsonl")
        teacher_only = sum(row.get("authority") == "teacher-only" for row in labels)
        verified_truth = sum(row.get("authority") == "verified" for row in labels)
        probe_not_truth = 0
        baseline_id = state.get("baseline_artifact_id")
        if isinstance(baseline_id, str) and len(baseline_id) == 64:
            baseline = read_sealed(
                root / "baselines" / f"{baseline_id}.json",
                schema="chronovisor.recall-distill-baseline.v1",
            )
            counts = baseline.get("counts")
            if isinstance(counts, Mapping):
                teacher_only = int(counts.get("teacher_only_labels") or 0)
                verified_truth = int(counts.get("verified_truth_labels") or 0)
                probe_not_truth = int(counts.get("locked_test_probe_pairs") or 0)
        paired_denominator = sum(
            row.get("kind") == "shadow-policy-observation"
            and row.get("paired_eligible") is True
            and row.get("stage") == rollout_status
            and row.get("stage_started_at") == state.get("stage_started_at")
            and row.get("qualified_run_id") == state.get("stage_run_id")
            for row in read_chain(root / "shadow-observation-receipts.jsonl")
        )
        manifest_backlog = state.get("manifest_backlog", 0)
        candidate_backlog = state.get("candidate_backlog", 0)
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (manifest_backlog, candidate_backlog)
        ):
            raise DistillationStoreError("cold-start backlog is invalid")
        return {
            "schema": "chronovisor.recall-distillation-health.v1",
            "status": worker_status,
            "state": worker_status,
            "worker_status": worker_status,
            "rollout_status": rollout_status,
            "rollout": rollout if rollout in {0, 5, 25, 100} else 0,
            "state_sha256": str(state.get("seal_sha256") or "")[:12],
            "active_policy_id": pointers["active"],
            "candidate_policy_id": pointers["candidate"],
            "lkg_policy_id": pointers["lkg"],
            "last_success_at": str(state.get("last_success_at") or ""),
            "error_code": error_code[:80],
            "feature_revision": "recall-distill-text-v2",
            "teacher_only": teacher_only,
            "verified_truth": verified_truth,
            "probe_not_truth": probe_not_truth,
            "paired_denominator": paired_denominator,
            "hold_reason": str(state.get("hold_reason") or "")[:80],
            "cold_start_pending": bool(state.get("cold_start_pending", False)),
            "split_plan_id": str(state.get("split_plan_id") or "")[:12],
            "manifest_backlog": manifest_backlog,
            "candidate_backlog": candidate_backlog,
        }
    except DistillationStoreError:
        state_path = root / STATE_FILE
        missing = not state_path.exists()
        return {
            "schema": "chronovisor.recall-distillation-health.v1",
            "status": "unavailable" if missing else "tampered",
            "state": "unavailable" if missing else "tampered",
            "worker_status": "unavailable" if missing else "tampered",
            "rollout_status": "unavailable" if missing else "tampered",
            "rollout": 0,
            "state_sha256": "",
            "active_policy_id": "",
            "candidate_policy_id": "",
            "lkg_policy_id": "",
            "last_success_at": "",
            "error_code": "missing_state" if missing else "invalid_state",
            "feature_revision": "recall-distill-text-v2",
            "teacher_only": 0,
            "verified_truth": 0,
            "probe_not_truth": 0,
            "paired_denominator": 0,
            "hold_reason": "",
            "cold_start_pending": False,
            "split_plan_id": "",
            "manifest_backlog": 0,
            "candidate_backlog": 0,
        }
