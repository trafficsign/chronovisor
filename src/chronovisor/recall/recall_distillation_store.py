"""Private durable storage for autonomous Recall distillation."""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import sqlite3
import tempfile
import unicodedata
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

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


def _reject_reserved(payload: Mapping[str, Any], reserved: set[str]) -> None:
    conflict = reserved.intersection(payload)
    if conflict:
        raise DistillationStoreError(
            f"reserved artifact fields are caller-controlled: {sorted(conflict)}"
        )


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
) -> tuple[str, Path, dict[str, Any]]:
    _reject_reserved(payload, {"schema", "namespace", "artifact_id", "seal_sha256"})
    unsigned = {
        "schema": schema,
        "namespace": "recall-distillation",
        **payload,
    }
    identity = artifact_id or canonical_json_sha256_strict(unsigned)
    artifact = _sealed({"artifact_id": identity, **unsigned})
    encoded = canonical_json_bytes_strict(artifact) + b"\n"
    path = directory / f"{identity}.json"
    directory.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        if path.read_bytes() != encoded:
            raise DistillationStoreError("immutable artifact conflict") from exc
    else:
        with os.fdopen(fd, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        directory_fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    return identity, path, verify_seal(artifact, schema=schema)


@contextlib.contextmanager
def _locked(path: Path):
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
        digest = row.get("record_sha256")
        unsigned = {key: value for key, value in row.items() if key != "record_sha256"}
        if row.get(
            "previous_sha256"
        ) != previous or digest != canonical_json_sha256_strict(unsigned):
            raise DistillationStoreError(f"ledger chain mismatch at row {index}")
        previous = str(digest)
        count += 1
    return {"records": count, "head_sha256": previous}


def append_chain(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
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
    lock_path = path.with_suffix(path.suffix + ".lock")
    with _locked(lock_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        head = verify_chain(path)
        unsigned = {
            "schema": DISTILLATION_SCHEMA,
            "namespace": "recall-distillation",
            "previous_sha256": head["head_sha256"],
            **payload,
        }
        row = {**unsigned, "record_sha256": canonical_json_sha256_strict(unsigned)}
        encoded = canonical_json_bytes_strict(row) + b"\n"
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        with os.fdopen(fd, "ab") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        return row


def append_chain_unique(
    path: Path,
    payload: Mapping[str, Any],
    *,
    unique_field: str,
    binding_field: str,
) -> dict[str, Any]:
    """Append once by a stable identity while holding the ledger lock."""

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
    lock_path = path.with_suffix(path.suffix + ".lock")
    with _locked(lock_path):
        head = verify_chain(path)
        if path.exists():
            for line in path.read_text().splitlines():
                row = json.loads(line)
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
        fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        with os.fdopen(fd, "ab") as handle:
            handle.write(canonical_json_bytes_strict(row) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return row


def read_chain(path: Path) -> list[dict[str, Any]]:
    verify_chain(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


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
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.unlink(missing_ok=True)
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
                    schema="chronovisor.recall-distill-policy.v1",
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
        }
