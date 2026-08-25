#!/usr/bin/env python3
"""One-shot, fail-closed R4 Workset archive/cutover utility.

The normal distillation worker continues to use ``ox-workset.sqlite3``.  This
tool is intentionally an operator action: it archives the legacy queue and
atomically replaces only that file with a verified-empty v2 queue.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import re
import sqlite3
import stat
import sys
import tempfile
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from chronovisor.core import canonical_json
from chronovisor.core.durable_state import okf_writer_lock
from chronovisor.recall import recall_distillation as distill
from chronovisor.recall import recall_distillation_store as store
from chronovisor.recall.recall_distillation_workset import (
    DistillationWorkset,
    DistillationWorksetError,
)

ARCHIVE_SCHEMA = "chronovisor.recall-r4-workset-archive.v1"
CUTOVER_SCHEMA = "chronovisor.recall-r4-workset-cutover.v1"
COPYFILE_ALL = (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3)
COPYFILE_NOFOLLOW = (1 << 18) | (1 << 19)
COPYFILE_CLONE_FORCE = 1 << 25
_SIDECARS = ("-wal", "-shm", "-journal")
_LEGACY_COLUMNS = {
    "sequence",
    "work_id",
    "kind",
    "payload_ref",
    "payload_digest",
    "temporal_split_json",
    "provenance_json",
    "priority",
    "watermark_json",
    "state",
    "attempt_count",
    "last_error_class",
    "lease_id",
    "lease_owner",
    "lease_expires_at",
    "completion_ref",
    "completion_digest",
    "created_at",
    "updated_at",
}
_V2_COLUMNS = _LEGACY_COLUMNS | {"stage", "next_attempt_at"}
_BASE_TABLES = {"work_items", "workset_state", "sqlite_sequence"}
_BASE_INDEXES = {"work_items_claim_order", "work_items_expiry"}


class CutoverError(RuntimeError):
    """The cutover cannot prove a safe state."""


def _sha256(path: Path) -> str:
    expected = _regular(path)
    assert expected is not None
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        expected_identity = (
            expected.st_dev,
            expected.st_ino,
            expected.st_size,
            expected.st_mtime_ns,
            expected.st_ctime_ns,
        )
        observed_identity = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        if not stat.S_ISREG(before.st_mode) or observed_identity != expected_identity:
            raise CutoverError("unsafe file")
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
        after = os.fstat(descriptor)
        if (
            observed_identity
            != (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            )
            or (post := _regular(path)) is None
            or (
                post.st_dev,
                post.st_ino,
                post.st_size,
                post.st_mtime_ns,
                post.st_ctime_ns,
            )
            != observed_identity
        ):
            raise CutoverError("file changed during digest")
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def _read_regular_bytes(path: Path, *, limit: int = 128 * 1024 * 1024) -> bytes:
    expected = _regular(path)
    assert expected is not None
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        expected_identity = (
            expected.st_dev,
            expected.st_ino,
            expected.st_size,
            expected.st_mtime_ns,
            expected.st_ctime_ns,
        )
        observed_identity = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        if (
            not stat.S_ISREG(before.st_mode)
            or observed_identity != expected_identity
            or before.st_size > limit
        ):
            raise CutoverError("unsafe artifact")
        parts: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise CutoverError("artifact short read")
            parts.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(descriptor)
        if (
            observed_identity
            != (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            )
            or (post := _regular(path)) is None
            or (
                post.st_dev,
                post.st_ino,
                post.st_size,
                post.st_mtime_ns,
                post.st_ctime_ns,
            )
            != observed_identity
        ):
            raise CutoverError("artifact changed during read")
        return b"".join(parts)
    finally:
        os.close(descriptor)


def _fsync_regular(path: Path) -> None:
    expected = _regular(path)
    assert expected is not None
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        observed = os.fstat(descriptor)
        identity = (
            expected.st_dev,
            expected.st_ino,
            expected.st_size,
            expected.st_mtime_ns,
            expected.st_ctime_ns,
        )
        if (
            not stat.S_ISREG(observed.st_mode)
            or (
                observed.st_dev,
                observed.st_ino,
                observed.st_size,
                observed.st_mtime_ns,
                observed.st_ctime_ns,
            )
            != identity
        ):
            raise CutoverError("unsafe file")
        os.fsync(descriptor)
        post = _regular(path)
        if (
            post is None
            or (
                post.st_dev,
                post.st_ino,
                post.st_size,
                post.st_mtime_ns,
                post.st_ctime_ns,
            )
            != identity
        ):
            raise CutoverError("file changed during fsync")
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _directory_identity(path: Path) -> tuple[int, int]:
    _safe_directory(path)
    observed = path.lstat()
    if stat.S_ISLNK(observed.st_mode) or not stat.S_ISDIR(observed.st_mode):
        raise CutoverError("unsafe directory")
    return observed.st_dev, observed.st_ino


def _regular(path: Path, *, required: bool = True) -> os.stat_result | None:
    try:
        observed = path.lstat()
    except FileNotFoundError:
        if required:
            raise CutoverError(f"missing required file: {path.name}") from None
        return None
    if stat.S_ISLNK(observed.st_mode) or not stat.S_ISREG(observed.st_mode):
        raise CutoverError(f"unsafe file: {path.name}")
    return observed


def _safe_directory(path: Path) -> None:
    current = path.absolute()
    while current != current.parent:
        try:
            observed = current.lstat()
        except FileNotFoundError:
            current = current.parent
            continue
        if stat.S_ISLNK(observed.st_mode) or not stat.S_ISDIR(observed.st_mode):
            raise CutoverError("unsafe directory component")
        current = current.parent


def _copyfile_clone(source: Path, destination: Path, flags: int) -> None:
    if sys.platform != "darwin":
        raise CutoverError("copyfile(3) clone requires Darwin/APFS")
    library = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
    function = library.copyfile
    function.argtypes = [
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_void_p,
        ctypes.c_uint32,
    ]
    function.restype = ctypes.c_int
    if function(os.fsencode(source), os.fsencode(destination), None, int(flags)) != 0:
        raise CutoverError(f"forced APFS clone failed: errno={ctypes.get_errno()}")


def _sealed(payload: Mapping[str, Any], *, schema: str) -> dict[str, Any]:
    unsigned = {"schema": schema, "namespace": "recall-distillation", **payload}
    return {
        **unsigned,
        "artifact_id": canonical_json.canonical_json_sha256_strict(unsigned),
        "seal_sha256": canonical_json.canonical_json_sha256_strict(
            {
                "artifact_id": canonical_json.canonical_json_sha256_strict(unsigned),
                **unsigned,
            }
        ),
    }


def _canonical_artifact_id(payload: Mapping[str, Any]) -> bool:
    artifact_id = payload.get("artifact_id")
    return (
        isinstance(artifact_id, str)
        and re.fullmatch(r"[0-9a-f]{64}", artifact_id) is not None
        and artifact_id
        == canonical_json.canonical_json_sha256_strict(
            {
                key: value
                for key, value in payload.items()
                if key not in {"artifact_id", "seal_sha256"}
            }
        )
    )


def _validate_r0_evidence(path: Path) -> dict[str, Any]:
    try:
        payload = store.verify_seal(
            json.loads(_read_regular_bytes(path)), schema="chronovisor.recall-r0.v1"
        )
    except (OSError, ValueError, store.DistillationStoreError) as exc:
        raise CutoverError("R0 evidence is invalid") from exc
    if payload.get(
        "artifact_id"
    ) != distill.R4_R0_EVIDENCE_ID or not _canonical_artifact_id(payload):
        raise CutoverError("R0 evidence is invalid")
    return payload


def _write_once(
    path: Path, payload: Mapping[str, Any], *, schema: str
) -> dict[str, Any]:
    _safe_directory(path.parent)
    artifact = _sealed(payload, schema=schema)
    encoded = canonical_json.canonical_json_bytes_strict(artifact) + b"\n"
    existing = _regular(path, required=False)
    if existing is not None:
        try:
            decoded = json.loads(_read_regular_bytes(path))
            store.verify_seal(decoded, schema=schema)
            if not _canonical_artifact_id(decoded):
                raise ValueError("artifact id")
        except (OSError, ValueError, store.DistillationStoreError) as exc:
            raise CutoverError("sealed artifact is invalid") from exc
        if canonical_json.canonical_json_bytes_strict(decoded) + b"\n" != encoded:
            raise CutoverError("sealed artifact collision")
        return decoded
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o400)
        offset = 0
        while offset < len(encoded):
            written = os.write(descriptor, encoded[offset:])
            if written <= 0:
                raise OSError("sealed artifact short write")
            offset += written
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError:
            current = _regular(path)
            if current is None or _read_regular_bytes(path) != encoded:
                raise CutoverError("sealed artifact collision") from None
    finally:
        if descriptor != -1:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
    _fsync_directory(path.parent)
    if _regular(path) is None or _read_regular_bytes(path) != encoded:
        raise CutoverError("sealed artifact readback failed")
    return artifact


def _atomic_output(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    return _write_once(path, payload, schema=CUTOVER_SCHEMA)


def _reject_root_output(root: Path, output: Path | None) -> None:
    if output is None:
        return
    resolved = output.absolute()
    _safe_directory(resolved.parent)
    if resolved == root or resolved.is_relative_to(root):
        raise CutoverError("output must be outside the production root")


def _reject_output_overlap(output: Path | None, *inputs: Path) -> None:
    if output is None:
        return
    output_path = output.absolute()
    output_stat = _regular(output_path, required=False)
    for input_path in inputs:
        absolute = input_path.absolute()
        if output_path == absolute:
            raise CutoverError("output must not overlap an input artifact")
        input_stat = _regular(absolute, required=False)
        if (
            output_stat is not None
            and input_stat is not None
            and (output_stat.st_dev, output_stat.st_ino)
            == (input_stat.st_dev, input_stat.st_ino)
        ):
            raise CutoverError("output must not overlap an input artifact")


def _sqlite_identity(path: Path) -> dict[str, Any]:
    _regular(path)
    allowed = {path.name + suffix for suffix in _SIDECARS}
    unknown = [
        item.name
        for item in path.parent.glob(f"{path.name}-*")
        if item.name not in allowed
    ]
    if unknown:
        raise CutoverError("unknown workset sidecar exists")
    for suffix in _SIDECARS:
        sidecar = path.with_name(path.name + suffix)
        observed = _regular(sidecar, required=False)
        if suffix == "-wal" and (observed is None or observed.st_size != 0):
            raise CutoverError("workset WAL is not empty")
        if suffix == "-journal" and observed is not None:
            raise CutoverError("workset journal exists")
    try:
        uri = f"{path.absolute().as_uri()}?mode=ro&immutable=1"
        connection = sqlite3.connect(uri, uri=True)
        try:
            tables = {
                str(name)
                for (name,) in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            columns = {
                str(name)
                for (_, name, *_rest) in connection.execute(
                    "PRAGMA table_info(work_items)"
                )
            }
            indexes = {
                str(name)
                for (name,) in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='index' AND sql IS NOT NULL"
                )
            }
            if (
                tables != _BASE_TABLES
                and tables != _BASE_TABLES | {"workset_receipts"}
                or (columns != _LEGACY_COLUMNS and columns != _V2_COLUMNS)
                or not _BASE_INDEXES.issubset(indexes)
                or indexes - (_BASE_INDEXES | {"work_items_retry_due"})
            ):
                raise CutoverError("legacy workset schema is unknown")
            counts = {
                str(state): int(count)
                for state, count in connection.execute(
                    "SELECT state, COUNT(*) FROM work_items GROUP BY state"
                )
            }
            if set(counts) - {"ready", "leased", "completed", "quarantined"}:
                raise CutoverError("legacy workset state is unknown")
            lease_count = int(
                connection.execute(
                    "SELECT COUNT(*) FROM work_items WHERE state='leased' OR lease_id IS NOT NULL "
                    "OR lease_owner IS NOT NULL OR lease_expires_at IS NOT NULL"
                ).fetchone()[0]
            )
            select = [
                "work_id",
                "kind",
                "payload_ref",
                "payload_digest",
                "temporal_split_json",
                "provenance_json",
                "priority",
                "watermark_json",
            ]
            if "stage" in columns:
                select.append("stage")
            select.extend(
                [
                    "state",
                    "attempt_count",
                    "last_error_class",
                    "lease_id",
                    "lease_owner",
                    "lease_expires_at",
                ]
            )
            if "next_attempt_at" in columns:
                select.append("next_attempt_at")
            select.extend(["completion_ref", "completion_digest"])
            logical_digest = hashlib.sha256()
            row_count = 0
            for row in connection.execute(
                f"SELECT {', '.join(select)} FROM work_items ORDER BY sequence"
            ):
                logical_digest.update(
                    canonical_json.canonical_json_bytes_strict(list(row))
                )
                logical_digest.update(b"\n")
                row_count += 1
            logical = logical_digest.hexdigest()
            receipt_count = (
                int(
                    connection.execute(
                        "SELECT COUNT(*) FROM workset_receipts"
                    ).fetchone()[0]
                )
                if "workset_receipts" in tables
                else 0
            )
            integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
        finally:
            connection.close()
    except (OSError, sqlite3.Error, DistillationWorksetError) as exc:
        raise CutoverError("workset is invalid") from exc
    if integrity != "ok" or lease_count != 0 or receipt_count != 0 or row_count == 0:
        raise CutoverError("workset is not an idle legacy queue")
    return {
        "main_sha256": _sha256(path),
        "rows": row_count,
        "states": {
            state: counts.get(state, 0)
            for state in ("ready", "leased", "completed", "quarantined")
        },
        "logical_sha256": logical,
        "audit_status": "legacy-unverified",
    }


def _fresh_identity(path: Path) -> dict[str, Any]:
    _regular(path)
    allowed = {path.name + suffix for suffix in _SIDECARS}
    if any(item.name not in allowed for item in path.parent.glob(f"{path.name}-*")):
        raise CutoverError("fresh workset sidecar is unknown")
    if _regular(path.with_name(path.name + "-journal"), required=False) is not None:
        raise CutoverError("fresh workset journal exists")
    try:
        queue = DistillationWorkset(path, migrate=False)
        receipt = queue.audit_transition_receipts()
        connection = sqlite3.connect(f"file:{path.absolute()}?mode=ro", uri=True)
        try:
            integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
            tables = {
                str(name)
                for (name,) in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            columns = {
                str(name)
                for (_, name, *_rest) in connection.execute(
                    "PRAGMA table_info(work_items)"
                )
            }
            indexes = {
                str(name)
                for (name,) in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='index' AND sql IS NOT NULL"
                )
            }
        finally:
            connection.close()
    except (OSError, sqlite3.Error, DistillationWorksetError) as exc:
        raise CutoverError("fresh workset is invalid") from exc
    if (
        receipt.get("status") != "verified-empty"
        or integrity != "ok"
        or tables != _BASE_TABLES | {"workset_receipts"}
        or columns != _V2_COLUMNS
        or indexes != _BASE_INDEXES | {"work_items_retry_due"}
    ):
        raise CutoverError("fresh workset is not verified-empty")
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    wal = _regular(path.with_name(path.name + "-wal"), required=False)
    if wal is not None and wal.st_size != 0:
        raise CutoverError("fresh workset WAL is not empty")
    _fsync_regular(path)
    return {"main_sha256": _sha256(path), "audit": receipt, "integrity": integrity}


def _unlink_sidecars(path: Path) -> None:
    for suffix in _SIDECARS:
        target = path.with_name(path.name + suffix)
        if _regular(target, required=False) is not None:
            target.unlink()


def _clone_snapshot(
    old: Path, snapshot: Path, identity: Mapping[str, Any]
) -> dict[str, Any]:
    snapshot.mkdir(mode=0o700, exist_ok=True)
    if snapshot.is_symlink() or not snapshot.is_dir() or any(snapshot.iterdir()):
        raise CutoverError("archive snapshot is unsafe")
    flags = COPYFILE_ALL | COPYFILE_NOFOLLOW | COPYFILE_CLONE_FORCE
    copied: dict[str, Any] = {}
    for suffix in ("", "-wal", "-shm"):
        source = old.with_name(old.name + suffix)
        if _regular(source, required=False) is None:
            if suffix == "":
                raise CutoverError("legacy main disappeared")
            copied[suffix or "main"] = None
            continue
        target = snapshot / source.name
        _copyfile_clone(source, target, flags)
        _regular(target)
        copied[suffix or "main"] = {
            "sha256": _sha256(target),
            "size_bytes": target.stat().st_size,
        }
    if copied["main"]["sha256"] != identity["main_sha256"]:
        raise CutoverError("archive main identity changed")
    if _sqlite_identity(snapshot / old.name) != dict(identity):
        raise CutoverError("archive logical identity changed")
    return copied


def _snapshot_file_state(path: Path) -> dict[str, Any]:
    copied: dict[str, Any] = {}
    for suffix in ("", "-wal", "-shm"):
        item = path.with_name(path.name + suffix)
        observed = _regular(item, required=False)
        copied[suffix or "main"] = (
            None
            if observed is None
            else {"sha256": _sha256(item), "size_bytes": observed.st_size}
        )
    return copied


def _snapshot_state(
    snapshot: Path, old: Path, identity: Mapping[str, Any]
) -> dict[str, Any]:
    if snapshot.is_symlink() or not snapshot.is_dir():
        raise CutoverError("partial archive snapshot is unsafe")
    allowed = {old.name, old.name + "-wal", old.name + "-shm"}
    if {item.name for item in snapshot.iterdir()} - allowed:
        raise CutoverError("partial archive snapshot is unsafe")
    copied = _snapshot_file_state(snapshot / old.name)
    for suffix in ("", "-wal", "-shm"):
        item = snapshot / f"{old.name}{suffix}"
        if _regular(item, required=False) is not None:
            _fsync_regular(item)
    if copied["main"] is None or copied["main"]["sha256"] != identity["main_sha256"]:
        raise CutoverError("partial archive identity changed")
    if _sqlite_identity(snapshot / old.name) != dict(identity):
        raise CutoverError("partial archive logical identity changed")
    _fsync_directory(snapshot)
    return copied


def _resume_partial_snapshot(
    old: Path, snapshot: Path, identity: Mapping[str, Any]
) -> dict[str, Any]:
    """Complete only an unsealed snapshot whose present files still match source."""

    if snapshot.is_symlink() or not snapshot.is_dir():
        raise CutoverError("partial archive snapshot is unsafe")
    allowed = {old.name, old.name + "-wal", old.name + "-shm"}
    if {item.name for item in snapshot.iterdir()} - allowed:
        raise CutoverError("partial archive snapshot is unsafe")
    actual = _snapshot_file_state(snapshot / old.name)
    source = _snapshot_file_state(old)
    if source["main"] is None or source["main"]["sha256"] != identity["main_sha256"]:
        raise CutoverError("legacy source changed while archiving")
    if actual["main"] != source["main"]:
        raise CutoverError("partial archive snapshot is tampered")
    if actual == source:
        return _snapshot_state(snapshot, old, identity)
    # A pre-manifest crash may have copied the main before volatile sidecars.
    # Any differing present file is tampering, not an incomplete clone.
    if any(actual[key] is not None and actual[key] != source[key] for key in actual):
        raise CutoverError("partial archive snapshot is tampered")
    for item in snapshot.iterdir():
        _regular(item)
        item.unlink()
    _fsync_directory(snapshot)
    return _clone_snapshot(old, snapshot, identity)


def _validate_snapshot_manifest(
    snapshot: Path,
    old: Path,
    identity: Mapping[str, Any],
    expected: object,
) -> dict[str, Any]:
    """Bind a sealed manifest to every archived file, including volatile SHM."""

    if not isinstance(expected, Mapping) or set(expected) != {"main", "-wal", "-shm"}:
        raise CutoverError("archive snapshot manifest is invalid")
    for value in expected.values():
        if value is None:
            continue
        if (
            not isinstance(value, Mapping)
            or set(value) != {"sha256", "size_bytes"}
            or not isinstance(value["sha256"], str)
            or re.fullmatch(r"[0-9a-f]{64}", value["sha256"]) is None
            or not isinstance(value["size_bytes"], int)
            or isinstance(value["size_bytes"], bool)
            or value["size_bytes"] < 0
        ):
            raise CutoverError("archive snapshot manifest is invalid")
    actual = _snapshot_state(snapshot, old, identity)
    if actual != dict(expected):
        raise CutoverError("archive snapshot does not match manifest")
    return actual


def _validate_archive_shape(archive_root: Path) -> None:
    _safe_directory(archive_root)
    if archive_root.is_symlink() or not archive_root.is_dir():
        raise CutoverError("archive directory is unsafe")
    entries = {item.name for item in archive_root.iterdir()}
    required = {"snapshot", "archive-manifest.json"}
    allowed = required | {"cutover-completion.json"}
    if not required.issubset(entries) or entries - allowed:
        raise CutoverError("archive directory shape is invalid")
    for name in entries:
        item = archive_root / name
        if name == "snapshot":
            _safe_directory(item)
        else:
            _regular(item)


def _validate_completion(
    archive_root: Path, manifest: Mapping[str, Any]
) -> dict[str, Any] | None:
    completion_path = archive_root / "cutover-completion.json"
    if _regular(completion_path, required=False) is None:
        return None
    payload = _read_sealed_regular(completion_path, schema=CUTOVER_SCHEMA)
    legacy = manifest.get("legacy")
    fresh = payload.get("fresh")
    if (
        not _canonical_artifact_id(payload)
        or payload.get("kind") != "r4-workset-cutover-completion"
        or payload.get("archive_artifact_id") != manifest.get("artifact_id")
        or not isinstance(legacy, Mapping)
        or payload.get("legacy_main_sha256") != legacy.get("main_sha256")
        or not isinstance(fresh, Mapping)
        or fresh.get("main_sha256") != manifest.get("fresh_main_sha256")
        or payload.get("provider_calls") != 0
        or payload.get("ox_enabled") is not False
        or payload.get("production_certification") is not False
    ):
        raise CutoverError("archive completion is invalid")
    return payload


@contextmanager
def _cutover_locks(root: Path):
    with okf_writer_lock(root, exclusive=True, allow_create=False):
        directory = store.distillation_dir(root)
        directory_identity = _directory_identity(directory)
        worker_path = directory / "distillation-worker.lock"
        expected = _regular(worker_path)
        assert expected is not None
        worker = store.acquire_nonblocking_lock(worker_path)
        if worker is None:
            raise CutoverError("distillation worker is busy")
        try:
            observed = os.fstat(worker.fileno())
            expected_identity = (expected.st_dev, expected.st_ino)
            if (
                not stat.S_ISREG(observed.st_mode)
                or (observed.st_dev, observed.st_ino) != expected_identity
                or _directory_identity(directory) != directory_identity
                or (current := _regular(worker_path)) is None
                or (current.st_dev, current.st_ino) != expected_identity
            ):
                raise CutoverError("distillation worker lock changed")
            yield
        finally:
            store.release_lock(worker)


def _read_sealed_regular(path: Path, *, schema: str | None = None) -> dict[str, Any]:
    """Read a small sealed state through one stable no-follow descriptor."""

    try:
        return store.verify_seal(json.loads(_read_regular_bytes(path)), schema=schema)
    except (OSError, ValueError, store.DistillationStoreError) as exc:
        raise CutoverError("sealed state is invalid") from exc


def _verify_or_anchor(root: Path, r0: Path, source: Mapping[str, str]) -> None:
    r0_payload = _validate_r0_evidence(r0)
    anchor = store.distillation_dir(root) / distill.R4_CANDIDATE_ANCHOR_FILE
    if _regular(anchor, required=False) is None:
        _read_sealed_regular(anchor.with_name("candidate-ledger.jsonl.head.json"))
        distill.bootstrap_r4_candidate_anchor(
            root=root, tracked_r0_evidence=r0, source_binding=source
        )
        return
    try:
        r0_sha = _sha256(r0)
        current = _read_sealed_regular(
            anchor, schema=distill.R4_CANDIDATE_ANCHOR_SCHEMA
        )
        checkpoint = _read_sealed_regular(
            anchor.with_name("candidate-ledger.jsonl.head.json")
        )
    except (OSError, store.DistillationStoreError) as exc:
        raise CutoverError("R0 anchor is invalid") from exc
    expected = {
        "r0_artifact_id": r0_payload.get("artifact_id"),
        "r0_file_sha256": r0_sha,
        "bootstrap_source_commit": source.get("source_commit"),
    }
    candidate = current.get("candidate_checkpoint")
    critical = distill._r4_critical_module_sha256()
    candidate_file = anchor.parent / "candidate-ledger.jsonl"
    candidate_observed = _regular(candidate_file)
    assert candidate_observed is not None
    try:
        r0_candidate = r0_payload["production"]["ledgers"]["candidate-ledger.jsonl"]
    except (KeyError, TypeError) as exc:
        raise CutoverError("R0 evidence is invalid") from exc
    expected_candidate = {
        "head_sha256": checkpoint.get("head_sha256"),
        "records": checkpoint.get("records"),
        "bytes": candidate_observed.st_size,
        "file_state": checkpoint.get("file_state"),
    }
    expected_keys = {
        "schema",
        "namespace",
        "artifact_id",
        "seal_sha256",
        "kind",
        "r0_artifact_id",
        "r0_file_sha256",
        "bootstrap_source_commit",
        "candidate_checkpoint",
        "critical_module_sha256",
    }
    if (
        set(current) != expected_keys
        or current.get("kind") != "r4-candidate-anchor"
        or not _canonical_artifact_id(current)
        or r0_payload.get("artifact_id") != distill.R4_R0_EVIDENCE_ID
        or any(current.get(key) != value for key, value in expected.items())
        or not isinstance(candidate, Mapping)
        or dict(candidate) != expected_candidate
        or current.get("critical_module_sha256") != critical
        or not isinstance(expected["r0_artifact_id"], str)
        or not isinstance(expected["r0_file_sha256"], str)
        or not isinstance(expected["bootstrap_source_commit"], str)
        or not isinstance(critical, Mapping)
        or not critical
        or set(candidate) != set(expected_candidate)
        or not isinstance(expected_candidate["head_sha256"], str)
        or not isinstance(expected_candidate["records"], int)
        or isinstance(expected_candidate["records"], bool)
        or not isinstance(expected_candidate["bytes"], int)
        or not isinstance(expected_candidate["file_state"], Mapping)
        or not isinstance(r0_candidate, Mapping)
        or r0_candidate.get("head_sha256") != expected_candidate["head_sha256"]
        or r0_candidate.get("records") != expected_candidate["records"]
        or r0_candidate.get("bytes") != expected_candidate["bytes"]
        or re.fullmatch(r"[0-9a-f]{64}", expected_candidate["head_sha256"]) is None
        or expected_candidate["records"] < 0
        or expected_candidate["bytes"] < 0
        or isinstance(expected_candidate["bytes"], bool)
        or not isinstance(expected_candidate["file_state"], Mapping)
        or expected_candidate["file_state"].get("size_bytes")
        != expected_candidate["bytes"]
        or r0_candidate.get("file_state") != expected_candidate["file_state"]
        or _sha256(r0) != r0_sha
    ):
        raise CutoverError("R0 anchor does not match current root")


def _prestate(root: Path) -> dict[str, str | None]:
    directory = store.distillation_dir(root)
    paths = {
        "candidate": directory / "candidate-ledger.jsonl",
        "checkpoint": directory / "candidate-ledger.jsonl.head.json",
        "config": root / "config.toml",
        "lock": directory / "distillation-worker.lock",
    }
    return {
        name: _sha256(path) if _regular(path, required=False) else None
        for name, path in paths.items()
    }


def _candidate_binding(
    root: Path, prestate: Mapping[str, str | None] | None = None
) -> dict[str, Any]:
    directory = store.distillation_dir(root)
    candidate = directory / "candidate-ledger.jsonl"
    checkpoint = directory / "candidate-ledger.jsonl.head.json"
    observed = _regular(candidate)
    _regular(checkpoint)
    assert observed is not None
    file_state = {
        "size_bytes": observed.st_size,
        "st_dev": observed.st_dev,
        "st_ino": observed.st_ino,
        "st_mtime_ns": observed.st_mtime_ns,
        "st_ctime_ns": observed.st_ctime_ns,
    }
    return {
        "sha256": (prestate or {}).get("candidate") or _sha256(candidate),
        "size_bytes": observed.st_size,
        "file_state": file_state,
        "checkpoint_sha256": (prestate or {}).get("checkpoint") or _sha256(checkpoint),
    }


def _validate_candidate_binding(
    root: Path, expected: object, prestate: Mapping[str, str | None] | None = None
) -> dict[str, Any]:
    if (
        not isinstance(expected, Mapping)
        or set(expected) != {"sha256", "size_bytes", "file_state", "checkpoint_sha256"}
        or not isinstance(expected["sha256"], str)
        or re.fullmatch(r"[0-9a-f]{64}", expected["sha256"]) is None
        or not isinstance(expected["checkpoint_sha256"], str)
        or re.fullmatch(r"[0-9a-f]{64}", expected["checkpoint_sha256"]) is None
        or not isinstance(expected["size_bytes"], int)
        or isinstance(expected["size_bytes"], bool)
        or expected["size_bytes"] < 0
        or not isinstance(expected["file_state"], Mapping)
        or set(expected["file_state"])
        != {"size_bytes", "st_dev", "st_ino", "st_mtime_ns", "st_ctime_ns"}
        or any(
            not isinstance(expected["file_state"][key], int)
            or isinstance(expected["file_state"][key], bool)
            for key in expected["file_state"]
        )
        or expected["file_state"].get("size_bytes") != expected["size_bytes"]
    ):
        raise CutoverError("candidate manifest binding is invalid")
    current = _candidate_binding(root, prestate)
    if current != dict(expected):
        raise CutoverError("candidate changed since archive manifest")
    return current


def _existing_archive(directory: Path, current_sha: str) -> Path | None:
    archives = directory / "workset-archives"
    if not archives.exists():
        return None
    _safe_directory(archives)
    matches: list[Path] = []
    for candidate in archives.iterdir():
        if (
            not candidate.name.startswith("legacy-v1-")
            or candidate.is_symlink()
            or not candidate.is_dir()
        ):
            raise CutoverError("unsafe archive directory")
        manifest_path = candidate / "archive-manifest.json"
        if _regular(manifest_path, required=False) is None:
            continue
        try:
            manifest = _read_manifest(manifest_path)
        except (OSError, ValueError, store.DistillationStoreError) as exc:
            raise CutoverError("archive manifest is invalid") from exc
        legacy = manifest.get("legacy")
        if not isinstance(legacy, Mapping):
            raise CutoverError("archive manifest is invalid")
        if candidate.name != f"legacy-v1-{legacy.get('main_sha256')}":
            raise CutoverError("archive manifest directory is invalid")
        if current_sha in {
            manifest.get("fresh_main_sha256"),
            legacy.get("main_sha256"),
        }:
            matches.append(candidate)
    if len(matches) > 1:
        raise CutoverError("multiple archives match canonical workset")
    return matches[0] if matches else None


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = store.verify_seal(
            json.loads(_read_regular_bytes(path)), schema=ARCHIVE_SCHEMA
        )
        if (
            not isinstance(payload.get("legacy"), Mapping)
            or not _canonical_artifact_id(payload)
            or payload.get("kind") != "r4-legacy-workset-archive"
            or payload.get("provider_calls") != 0
            or payload.get("ox_enabled") is not False
            or payload.get("production_certification") is not False
            or not isinstance(payload.get("fresh_main_sha256"), str)
            or not isinstance(payload.get("source"), Mapping)
            or not isinstance(payload.get("authority"), Mapping)
            or not isinstance(payload.get("snapshot"), Mapping)
            or not isinstance(payload.get("candidate"), Mapping)
            or not isinstance(payload["legacy"].get("main_sha256"), str)
            or re.fullmatch(r"[0-9a-f]{64}", payload["fresh_main_sha256"]) is None
            or re.fullmatch(r"[0-9a-f]{64}", payload["legacy"]["main_sha256"]) is None
        ):
            raise ValueError("legacy")
        return payload
    except (OSError, ValueError, store.DistillationStoreError) as exc:
        raise CutoverError("archive manifest is invalid") from exc


def cutover(
    *,
    root: Path,
    offline_evidence: Path,
    r0_evidence: Path,
    source_commit: str,
    output: Path | None,
    execute: bool = False,
    fault: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Archive a legacy queue and atomically install a verified-empty queue."""

    root = root.absolute()
    _safe_directory(root)
    _reject_root_output(root, output)
    _regular(offline_evidence)
    _regular(r0_evidence)
    _reject_output_overlap(output, offline_evidence, r0_evidence)
    source = distill._validate_ox_source_binding(distill.ox_alpha_source_binding())
    if source.get("source_commit") != source_commit:
        raise CutoverError("requested source commit is not the active OX binding")
    config = distill.load_distillation_config(root / "config.toml")
    if config.ox_enabled:
        raise CutoverError("OX must remain disabled during Workset cutover")
    old = store.distillation_dir(root) / "ox-workset.sqlite3"
    current_sha = _sha256(old)
    existing = _existing_archive(store.distillation_dir(root), current_sha)
    if existing is not None:
        _validate_archive_shape(existing)
    existing_manifest = (
        _read_manifest(existing / "archive-manifest.json") if existing else None
    )
    if existing is not None and existing_manifest is not None:
        _validate_completion(existing, existing_manifest)
    manifest_resume = bool(
        existing_manifest
        and current_sha
        in {
            existing_manifest.get("fresh_main_sha256"),
            existing_manifest.get("legacy", {}).get("main_sha256"),
        }
    )
    fresh_resume = bool(
        existing_manifest and current_sha == existing_manifest.get("fresh_main_sha256")
    )
    if not execute:
        if not manifest_resume:
            # Read-only evidence validation is the stop-before-mutate gate.
            distill._r4_legacy_offline_bootstrap(
                offline_evidence, root=root, expected_source_binding=source
            )
        try:
            authority = list(
                distill._r4_distillation_root_authority(root, register=False)
            )
        except distill.DistillationError as exc:
            if str(exc) != "local R4 root authority is missing":
                raise
            authority = None
        result = {
            "verdict": "resume-preflight" if manifest_resume else "preflight",
            "root": str(root),
            "authority": authority,
            "workset": {
                "main_sha256": current_sha,
                "state": "fresh-pending" if fresh_resume else "legacy-pending",
            }
            if manifest_resume
            else _sqlite_identity(old),
            "provider_calls": 0,
            "ox_enabled": False,
        }
        if output is not None:
            result["output"] = _atomic_output(output, result)
        return result
    # This explicit migration is deliberately before all queue mutations; normal
    # bootstrap must never adopt a populated legacy distillation directory.
    try:
        if manifest_resume:
            authority = distill._r4_distillation_root_authority(root, register=False)
        else:
            distill._r4_legacy_offline_bootstrap(
                offline_evidence, root=root, expected_source_binding=source
            )
            try:
                authority = distill._r4_distillation_root_authority(
                    root, register=False
                )
            except distill.DistillationError as exc:
                if str(exc) != "local R4 root authority is missing":
                    raise
                authority = distill.migrate_r4_legacy_distillation_root_authority(
                    root=root,
                    offline_bootstrap_evidence=offline_evidence,
                    expected_source_binding=source,
                )
    except distill.DistillationError as exc:
        raise CutoverError("R4 authority migration is invalid") from exc
    with _cutover_locks(root):
        locked_source = distill._validate_ox_source_binding(
            distill.ox_alpha_source_binding()
        )
        if locked_source != source:
            raise CutoverError("OX source binding changed before cutover lock")
        source = locked_source
        if distill.load_distillation_config(root / "config.toml").ox_enabled:
            raise CutoverError("OX must remain disabled during Workset cutover")
        if distill._r4_distillation_root_authority(root, register=False) != authority:
            raise CutoverError("root authority changed before cutover lock")
        locked_sha = _sha256(old)
        locked_archive = _existing_archive(store.distillation_dir(root), locked_sha)
        locked_manifest = (
            _read_manifest(locked_archive / "archive-manifest.json")
            if locked_archive
            else None
        )
        locked_resume = bool(
            locked_manifest
            and locked_sha
            in {
                locked_manifest.get("fresh_main_sha256"),
                locked_manifest.get("legacy", {}).get("main_sha256"),
            }
        )
        if locked_sha != current_sha or locked_resume != manifest_resume:
            raise CutoverError("canonical workset changed before cutover lock")
        if not locked_resume:
            distill._r4_legacy_offline_bootstrap(
                offline_evidence, root=root, expected_source_binding=source
            )
        before = _prestate(root)
        current_sha = _sha256(old)
        archive_root = _existing_archive(store.distillation_dir(root), current_sha) or (
            store.distillation_dir(root)
            / "workset-archives"
            / f"legacy-v1-{current_sha}"
        )
        snapshot = archive_root / "snapshot"
        manifest_path = archive_root / "archive-manifest.json"
        completion_path = archive_root / "cutover-completion.json"
        _safe_directory(archive_root.parent)
        identity: dict[str, Any]
        if _regular(manifest_path, required=False) is not None:
            _validate_archive_shape(archive_root)
            manifest = _read_manifest(manifest_path)
            if locked_manifest is not None and manifest != locked_manifest:
                raise CutoverError("archive manifest changed before cutover")
            _validate_completion(archive_root, manifest)
            identity = dict(manifest["legacy"])
            fresh_sha = manifest.get("fresh_main_sha256")
            if archive_root.name != f"legacy-v1-{identity.get('main_sha256')}":
                raise CutoverError("archive manifest directory is invalid")
            _validate_snapshot_manifest(
                snapshot, old, identity, manifest.get("snapshot")
            )
            _validate_candidate_binding(root, manifest.get("candidate"), before)
            if (
                manifest.get("source") != source
                or manifest.get("authority")
                != {"device": authority[0], "inode": authority[1]}
                or _sqlite_identity(snapshot / old.name) != identity
                or current_sha not in {identity.get("main_sha256"), fresh_sha}
            ):
                raise CutoverError("canonical workset is unknown")
        else:
            identity = _sqlite_identity(old)
            candidate_binding = _candidate_binding(root, before)
            archive_root.parent.mkdir(mode=0o700, exist_ok=True)
            _safe_directory(archive_root.parent)
            if archive_root.exists():
                if archive_root.is_symlink() or not archive_root.is_dir():
                    raise CutoverError("partial archive is unsafe")
                entries = {item.name for item in archive_root.iterdir()}
                if entries - {"snapshot"}:
                    raise CutoverError("partial archive is unsafe")
                copied = (
                    _clone_snapshot(old, snapshot, identity)
                    if not entries
                    or (
                        entries == {"snapshot"}
                        and snapshot.is_dir()
                        and not any(snapshot.iterdir())
                    )
                    else _resume_partial_snapshot(old, snapshot, identity)
                )
            else:
                archive_root.mkdir(mode=0o700)
                _fsync_directory(archive_root.parent)
                snapshot.mkdir(mode=0o700)
                if fault is not None:
                    fault("after-snapshot-mkdir")
                copied = _clone_snapshot(old, snapshot, identity)
            for suffix in ("", "-wal", "-shm"):
                item = snapshot / f"{old.name}{suffix}"
                if _regular(item, required=False) is not None:
                    _fsync_regular(item)
            _fsync_directory(snapshot)
            _fsync_directory(archive_root)
            temp = old.with_name(f".{old.name}.r4-{identity['main_sha256']}.tmp")
            if _regular(temp, required=False) is None:
                if any(
                    _regular(temp.with_name(temp.name + suffix), required=False)
                    is not None
                    for suffix in _SIDECARS
                ):
                    raise CutoverError("prepared fresh workset sidecar is orphaned")
                DistillationWorkset(temp, migrate=True)
                fresh = _fresh_identity(temp)
                _unlink_sidecars(temp)
            else:
                fresh = _fresh_identity(temp)
                _unlink_sidecars(temp)
            manifest = _write_once(
                manifest_path,
                {
                    "kind": "r4-legacy-workset-archive",
                    "legacy": identity,
                    "snapshot": copied,
                    "candidate": candidate_binding,
                    "fresh_main_sha256": fresh["main_sha256"],
                    "source": source,
                    "authority": {"device": authority[0], "inode": authority[1]},
                    "provider_calls": 0,
                    "ox_enabled": False,
                    "production_certification": False,
                },
                schema=ARCHIVE_SCHEMA,
            )
            _fsync_directory(archive_root.parent)
            # Keep the prepared fresh main until after the sealed intent exists.
        _validate_candidate_binding(root, manifest.get("candidate"), before)
        # Validate/create the independent R0 anchor only after the legacy queue
        # has passed schema/lease/WAL validation, but before canonical swap.
        _verify_or_anchor(root, r0_evidence, source)
        if current_sha == identity["main_sha256"]:
            temp = old.with_name(f".{old.name}.r4-{identity['main_sha256']}.tmp")
            if _regular(temp, required=False) is None:
                DistillationWorkset(temp, migrate=True)
                fresh = _fresh_identity(temp)
                _unlink_sidecars(temp)
                if fresh["main_sha256"] != manifest["fresh_main_sha256"]:
                    raise CutoverError("fresh workset is not reproducible")
            else:
                fresh = _fresh_identity(temp)
                _unlink_sidecars(temp)
                if fresh["main_sha256"] != manifest["fresh_main_sha256"]:
                    raise CutoverError("prepared fresh workset is invalid")
            if fault is not None:
                fault("before-swap")
            if distill.load_distillation_config(root / "config.toml").ox_enabled:
                raise CutoverError("OX must remain disabled during Workset cutover")
            _unlink_sidecars(old)
            os.replace(temp, old)
            directory_fd = os.open(
                old.parent, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
            )
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
            if fault is not None:
                fault("after-swap")
        if _sha256(old) != manifest["fresh_main_sha256"]:
            raise CutoverError("canonical fresh workset mismatch")
        fresh_identity = _fresh_identity(old)
        if distill.load_distillation_config(root / "config.toml").ox_enabled:
            raise CutoverError("OX must remain disabled during Workset cutover")
        _unlink_sidecars(old)
        _verify_or_anchor(root, r0_evidence, source)
        if before != _prestate(root):
            raise CutoverError("protected R4 prestate changed")
        completion = _write_once(
            completion_path,
            {
                "kind": "r4-workset-cutover-completion",
                "archive_artifact_id": manifest["artifact_id"],
                "legacy_main_sha256": identity["main_sha256"],
                "fresh": fresh_identity,
                "provider_calls": 0,
                "ox_enabled": False,
                "production_certification": False,
            },
            schema=CUTOVER_SCHEMA,
        )
    result = {
        "verdict": "completed",
        "archive": str(archive_root),
        "manifest": manifest,
        "completion": completion,
        "provider_calls": 0,
        "ox_enabled": False,
    }
    if output is not None:
        result["output"] = _atomic_output(output, result)
    return result


def rollback(
    *,
    root: Path,
    operation_id: str,
    output: Path | None,
    execute: bool = False,
    fault: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    root = root.absolute()
    _safe_directory(root)
    _reject_root_output(root, output)
    if re.fullmatch(r"legacy-v1-[0-9a-f]{64}", operation_id) is None:
        raise CutoverError("operation id is invalid")
    config = distill.load_distillation_config(root / "config.toml")
    if config.ox_enabled:
        raise CutoverError("OX must remain disabled during Workset rollback")
    archive_root = store.distillation_dir(root) / "workset-archives" / operation_id
    old = store.distillation_dir(root) / "ox-workset.sqlite3"
    _reject_output_overlap(
        output,
        old,
        archive_root / "archive-manifest.json",
        archive_root / "cutover-completion.json",
        archive_root / "snapshot" / old.name,
        *(
            archive_root / "snapshot" / f"{old.name}{suffix}"
            for suffix in ("-wal", "-shm")
        ),
    )
    archive_identity = _directory_identity(archive_root)
    manifest_path = archive_root / "archive-manifest.json"
    try:
        manifest = _read_manifest(manifest_path)
        legacy = dict(manifest["legacy"])
        if operation_id != f"legacy-v1-{legacy['main_sha256']}":
            raise ValueError("operation")
    except (
        KeyError,
        OSError,
        ValueError,
        store.DistillationStoreError,
        CutoverError,
    ) as exc:
        raise CutoverError("rollback archive is invalid") from exc
    _validate_archive_shape(archive_root)
    _validate_completion(archive_root, manifest)
    if not execute:
        result = {
            "verdict": "rollback-preflight",
            "operation_id": operation_id,
            "current_sha256": _sha256(old),
            "legacy": legacy,
        }
        if output is not None:
            result["output"] = _atomic_output(output, result)
        return result
    with _cutover_locks(root):
        if _directory_identity(archive_root) != archive_identity:
            raise CutoverError("rollback archive changed")
        _validate_archive_shape(archive_root)
        locked_manifest = _read_manifest(manifest_path)
        if locked_manifest != manifest:
            raise CutoverError("rollback manifest changed")
        manifest = locked_manifest
        legacy = dict(manifest["legacy"])
        if operation_id != f"legacy-v1-{legacy['main_sha256']}":
            raise CutoverError("rollback archive changed")
        _validate_completion(archive_root, manifest)
        distill._r4_distillation_root_authority(root, register=False)
        if distill.load_distillation_config(root / "config.toml").ox_enabled:
            raise CutoverError("OX must remain disabled during Workset rollback")
        snapshot_main = archive_root / "snapshot" / "ox-workset.sqlite3"
        _validate_snapshot_manifest(
            archive_root / "snapshot", old, legacy, manifest.get("snapshot")
        )
        if _sqlite_identity(snapshot_main) != legacy:
            raise CutoverError("rollback archive identity changed")
        rollback_temp = old.with_name(
            f".{old.name}.rollback-{legacy['main_sha256']}.tmp"
        )
        if _sha256(old) == legacy["main_sha256"]:
            if _regular(rollback_temp, required=False) is not None:
                raise CutoverError("rollback temp main is unexpected")
            try:
                if _sqlite_identity(old) != legacy:
                    raise CutoverError("rollback legacy identity changed")
            except CutoverError as exc:
                allowed = {old.name + suffix for suffix in _SIDECARS}
                if any(
                    item.name not in allowed
                    for item in old.parent.glob(f"{old.name}-*")
                ):
                    raise
                _unlink_sidecars(old)
                for suffix in ("-wal", "-shm"):
                    source = archive_root / "snapshot" / f"{old.name}{suffix}"
                    if _regular(source, required=False) is not None:
                        _copyfile_clone(
                            source,
                            old.with_name(old.name + suffix),
                            COPYFILE_ALL | COPYFILE_NOFOLLOW | COPYFILE_CLONE_FORCE,
                        )
                _fsync_directory(old.parent)
                if _sqlite_identity(old) != legacy:
                    raise CutoverError("rollback legacy recovery failed") from exc
            _unlink_sidecars(rollback_temp)
            _fsync_directory(old.parent)
            return {
                "verdict": "rollback-noop",
                "operation_id": operation_id,
                "provider_calls": 0,
                "ox_enabled": False,
                "production_certification": False,
            }
        if _sha256(old) != manifest.get("fresh_main_sha256"):
            raise CutoverError("canonical workset is unknown")
        temp = rollback_temp
        if _regular(temp, required=False) is not None:
            _unlink_sidecars(temp)
            temp.unlink()
        elif any(
            _regular(temp.with_name(temp.name + suffix), required=False) is not None
            for suffix in _SIDECARS
        ):
            raise CutoverError("rollback temp sidecar is orphaned")
        for suffix in ("", "-wal", "-shm"):
            source = archive_root / "snapshot" / f"{old.name}{suffix}"
            if _regular(source, required=False) is not None:
                _copyfile_clone(
                    source,
                    temp.with_name(temp.name + suffix),
                    COPYFILE_ALL | COPYFILE_NOFOLLOW | COPYFILE_CLONE_FORCE,
                )
        if _sqlite_identity(temp) != legacy:
            raise CutoverError("rollback archive identity changed")
        _unlink_sidecars(old)
        os.replace(temp, old)
        if fault is not None:
            fault("after-rollback-main-swap")
        for suffix in ("-wal", "-shm"):
            sidecar = temp.with_name(temp.name + suffix)
            if _regular(sidecar, required=False) is not None:
                os.replace(sidecar, old.with_name(old.name + suffix))
        directory_fd = os.open(
            old.parent, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        if _sqlite_identity(old) != legacy:
            raise CutoverError("rollback verification failed")
    result = {
        "verdict": "rolled-back",
        "operation_id": operation_id,
        "provider_calls": 0,
        "ox_enabled": False,
        "production_certification": False,
    }
    if output is not None:
        result["output"] = _atomic_output(output, result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--offline-evidence", type=Path)
    parser.add_argument("--r0-evidence", type=Path)
    parser.add_argument("--source-commit")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--rollback")
    args = parser.parse_args(argv)
    try:
        if args.rollback:
            result = rollback(
                root=args.root,
                operation_id=args.rollback,
                output=args.output,
                execute=args.execute,
            )
        else:
            if (
                not args.offline_evidence
                or not args.r0_evidence
                or not args.source_commit
            ):
                parser.error(
                    "--offline-evidence, --r0-evidence, and --source-commit are required"
                )
            result = cutover(
                root=args.root,
                offline_evidence=args.offline_evidence,
                r0_evidence=args.r0_evidence,
                source_commit=args.source_commit,
                output=args.output,
                execute=args.execute,
            )
    except (
        CutoverError,
        distill.DistillationError,
        store.DistillationStoreError,
        DistillationWorksetError,
        OSError,
        sqlite3.Error,
        ValueError,
    ) as exc:
        print(
            json.dumps({"verdict": "rejected", "reason": str(exc)}, sort_keys=True),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
