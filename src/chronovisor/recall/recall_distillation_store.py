"""Private durable storage for autonomous Recall distillation."""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import sqlite3
import stat
import tempfile
import unicodedata
import uuid
from collections.abc import Callable, Iterable, Iterator, Mapping
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
LABEL_LEDGER_FILE = "label-ledger.jsonl"
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
    except (OSError, ValueError, UnicodeError) as exc:
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


def _snapshot_directory_path(
    path: Path, *, expected_identity: tuple[int, int] | None = None
) -> tuple[Path, tuple[tuple[int, int] | None, ...]]:
    """Capture every existing directory inode before the pinned traversal."""

    absolute = path.expanduser().absolute()
    current = Path(absolute.anchor)
    identities: list[tuple[int, int] | None] = []
    missing = False
    for component in (None, *absolute.parts[1:]):
        if component is not None:
            current /= component
        if missing:
            identities.append(None)
            continue
        try:
            state = current.lstat()
        except FileNotFoundError:
            missing = True
            identities.append(None)
            continue
        except OSError as exc:
            raise DistillationStoreError(
                "immutable artifact directory is unsafe"
            ) from exc
        if not stat.S_ISDIR(state.st_mode):
            raise DistillationStoreError("immutable artifact directory is unsafe")
        identities.append((state.st_dev, state.st_ino))
    if expected_identity is not None and identities[-1] != expected_identity:
        raise DistillationStoreError("immutable artifact directory changed")
    return absolute, tuple(identities)


def _open_directory_nofollow(
    path: Path,
    *,
    create: bool,
    snapshot: tuple[Path, tuple[tuple[int, int] | None, ...]],
) -> int:
    """Open a directory through no-follow components and retain its inode fd."""

    absolute = path.expanduser().absolute()
    snapshot_path, expected_identities = snapshot
    if snapshot_path != absolute or len(expected_identities) != len(absolute.parts):
        raise DistillationStoreError("immutable artifact directory is unsafe")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    descriptor = os.open("/", flags)
    try:
        root_state = os.fstat(descriptor)
        if expected_identities[0] != (root_state.st_dev, root_state.st_ino):
            raise DistillationStoreError("immutable artifact directory changed")
        for index, component in enumerate(absolute.parts[1:], start=1):
            expected = expected_identities[index]
            if expected is None:
                if not create:
                    raise DistillationStoreError(
                        "immutable artifact directory is missing"
                    )
                try:
                    os.mkdir(component, 0o700, dir_fd=descriptor)
                except FileExistsError as exc:
                    raise DistillationStoreError(
                        "immutable artifact directory changed"
                    ) from exc
                created_state = os.stat(
                    component, dir_fd=descriptor, follow_symlinks=False
                )
                expected = (created_state.st_dev, created_state.st_ino)
            try:
                child = os.open(component, flags, dir_fd=descriptor)
            except FileNotFoundError as exc:
                message = (
                    "immutable artifact directory is missing"
                    if expected is None
                    else "immutable artifact directory changed"
                )
                raise DistillationStoreError(message) from exc
            child_state = os.fstat(child)
            if expected is not None and expected != (
                child_state.st_dev,
                child_state.st_ino,
            ):
                os.close(child)
                raise DistillationStoreError("immutable artifact directory changed")
            os.close(descriptor)
            descriptor = child
        return descriptor
    except DistillationStoreError:
        os.close(descriptor)
        raise
    except (OSError, ValueError) as exc:
        os.close(descriptor)
        raise DistillationStoreError("immutable artifact directory is unsafe") from exc


def _read_pinned_regular(directory_fd: int, name: str) -> bytes:
    try:
        descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory_fd)
    except FileNotFoundError:
        raise
    except OSError as exc:
        raise DistillationStoreError("immutable artifact is unreadable") from exc
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise DistillationStoreError("immutable artifact is unsafe")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 65_536):
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def pinned_directory_identity(
    directory: Path, *, create: bool, require_missing: bool = False
) -> tuple[int, int]:
    """Return the inode identity of one safely traversed directory."""

    snapshot = _snapshot_directory_path(directory)
    if require_missing and snapshot[1][-1] is not None:
        raise DistillationStoreError("immutable artifact directory changed")
    descriptor = _open_directory_nofollow(
        directory, create=create, snapshot=snapshot
    )
    try:
        state = os.fstat(descriptor)
        return state.st_dev, state.st_ino
    finally:
        os.close(descriptor)


def write_immutable_pinned(
    directory: Path,
    payload: Mapping[str, Any],
    *,
    schema: str,
    artifact_id: str | None = None,
    before_persist: Callable[[], None] | None = None,
    expected_directory_identity: tuple[int, int] | None = None,
) -> tuple[str, Path, dict[str, Any]]:
    """Immutable artifact write relative to a no-follow, inode-pinned directory."""

    _reject_reserved(payload, {"schema", "namespace", "artifact_id", "seal_sha256"})
    unsigned = {"schema": schema, "namespace": "recall-distillation", **payload}
    identity = _validate_artifact_id(artifact_id or canonical_json_sha256_strict(unsigned))
    artifact = _sealed({"artifact_id": identity, **unsigned})
    encoded = canonical_json_bytes_strict(artifact) + b"\n"
    name = f"{identity}.json"
    snapshot = _snapshot_directory_path(
        directory, expected_identity=expected_directory_identity
    )
    directory_fd = _open_directory_nofollow(
        directory, create=True, snapshot=snapshot
    )
    try:
        lock_fd = os.open(
            ".immutable.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600, dir_fd=directory_fd
        )
        try:
            if not stat.S_ISREG(os.fstat(lock_fd).st_mode):
                raise DistillationStoreError("immutable artifact lock is unsafe")
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            if before_persist is not None:
                before_persist()
            try:
                current = _read_pinned_regular(directory_fd, name)
            except FileNotFoundError:
                current = None
            if current is not None:
                if current != encoded:
                    raise DistillationStoreError("immutable artifact conflict")
            else:
                temporary = f".{name}.{uuid.uuid4().hex}.tmp"
                temporary_fd = os.open(
                    temporary,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW,
                    0o600,
                    dir_fd=directory_fd,
                )
                try:
                    offset = 0
                    while offset < len(encoded):
                        written = os.write(temporary_fd, encoded[offset:])
                        if written <= 0:
                            raise OSError("immutable artifact short write")
                        offset += written
                    os.fsync(temporary_fd)
                    os.replace(temporary, name, src_dir_fd=directory_fd, dst_dir_fd=directory_fd)
                    os.fsync(directory_fd)
                finally:
                    os.close(temporary_fd)
                    try:
                        os.unlink(temporary, dir_fd=directory_fd)
                    except FileNotFoundError:
                        pass
            if _read_pinned_regular(directory_fd, name) != encoded:
                raise DistillationStoreError("immutable artifact read-back failed")
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)
    finally:
        os.close(directory_fd)
    return identity, directory / name, verify_seal(artifact, schema=schema)


def read_immutable_pinned(
    directory: Path,
    artifact_id: str,
    *,
    schema: str | None = None,
    expected_directory_identity: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Read one named immutable artifact relative to a no-follow directory fd."""

    identity = _validate_artifact_id(artifact_id)
    snapshot = _snapshot_directory_path(
        directory, expected_identity=expected_directory_identity
    )
    descriptor = _open_directory_nofollow(
        directory, create=False, snapshot=snapshot
    )
    try:
        try:
            payload = json.loads(_read_pinned_regular(descriptor, f"{identity}.json"))
        except (ValueError, UnicodeError) as exc:
            raise DistillationStoreError("immutable artifact is invalid") from exc
    finally:
        os.close(descriptor)
    return verify_seal(payload, schema=schema)


def unlink_immutable_pinned(
    directory: Path,
    artifact_id: str,
    *,
    expected: Mapping[str, Any],
    schema: str,
    before_unlink: Callable[[], None] | None = None,
    expected_directory_identity: tuple[int, int] | None = None,
) -> None:
    """Unlink only a sealed expected artifact through its pinned directory fd."""

    identity = _validate_artifact_id(artifact_id)
    snapshot = _snapshot_directory_path(
        directory, expected_identity=expected_directory_identity
    )
    descriptor = _open_directory_nofollow(
        directory, create=False, snapshot=snapshot
    )
    try:
        try:
            payload = json.loads(
                _read_pinned_regular(descriptor, f"{identity}.json")
            )
        except (ValueError, UnicodeError) as exc:
            raise DistillationStoreError("immutable artifact is invalid") from exc
        if verify_seal(payload, schema=schema) != expected:
            raise DistillationStoreError("immutable artifact changed")
        if before_unlink is not None:
            before_unlink()
        os.unlink(f"{identity}.json", dir_fd=descriptor)
        os.fsync(descriptor)
    except OSError as exc:
        raise DistillationStoreError("immutable artifact cleanup failed") from exc
    finally:
        os.close(descriptor)


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


def _label_health_projection_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".health.json")


def _label_health_counts(rows: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    counts = {"teacher_only": 0, "verified_truth": 0, "probe_not_truth": 0}
    for row in rows:
        if row.get("authority") == "teacher-only":
            counts["teacher_only"] += 1
        elif row.get("authority") == "verified":
            counts["verified_truth"] += 1
        assignment = row.get("assignment")
        if isinstance(assignment, Mapping) and assignment.get("probe") is True:
            counts["probe_not_truth"] += 1
    return counts


def _read_label_health_projection(
    path: Path, head: Mapping[str, Any]
) -> dict[str, Any] | None:
    try:
        projection = read_sealed(
            _label_health_projection_path(path), schema=DISTILLATION_SCHEMA
        )
    except DistillationStoreError:
        return None
    counts = projection.get("counts")
    records = head.get("records")
    if (
        projection.get("kind") != "label-health-projection"
        or projection.get("ledger_name") != path.name
        or projection.get("label_chain_head") != head.get("head_sha256")
        or projection.get("label_records") != records
        or not isinstance(counts, Mapping)
        or set(counts) != {"teacher_only", "verified_truth", "probe_not_truth"}
        or not isinstance(records, int)
        or isinstance(records, bool)
        or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in counts.values()
        )
        or counts["teacher_only"] + counts["verified_truth"] > records
        or counts["probe_not_truth"] > records
    ):
        return None
    return {
        "label_chain_head": head["head_sha256"],
        "label_records": records,
        "counts": {key: int(value) for key, value in counts.items()},
    }


def _write_label_health_projection(
    path: Path, head: Mapping[str, Any], counts: Mapping[str, int]
) -> dict[str, Any]:
    return write_sealed_state(
        _label_health_projection_path(path),
        {
            "kind": "label-health-projection",
            "ledger_name": path.name,
            "label_chain_head": head["head_sha256"],
            "label_records": head["records"],
            "counts": dict(counts),
        },
    )


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
                    if not line.endswith(b"\n") or (
                        line == b"\n"
                        and end_offset == os.fstat(handle.fileno()).st_size
                    ):
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
        label_counts: dict[str, int] | None = None
        if path.name == LABEL_LEDGER_FILE:
            projection = _read_label_health_projection(path, head)
            label_counts = (
                dict(projection["counts"])
                if projection is not None
                else _label_health_counts(_read_chain_locked(path, head))
            )
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
        updated_head = {
            "records": head["records"] + len(rows),
            "head_sha256": previous,
        }
        _write_chain_checkpoint(path, updated_head)
        if label_counts is not None:
            additions = _label_health_counts(rows)
            try:
                _write_label_health_projection(
                    path,
                    updated_head,
                    {
                        key: label_counts[key] + additions[key]
                        for key in label_counts
                    },
                )
            except (OSError, DistillationStoreError):
                # The ledger is already durable; a stale projection fails closed
                # to the exact ledger scan on the next health read.
                pass
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


def _unique_index_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".unique.sqlite3")


def _unique_index_checkpoint_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".checkpoint.json")


def _write_unique_index_checkpoint(path: Path) -> None:
    write_sealed_state(
        _unique_index_checkpoint_path(path),
        {
            "kind": "unique-ledger-index-checkpoint",
            "index_name": path.name,
            "file_state": _ledger_file_state(path),
        },
    )


def _unique_index_checkpoint_matches(path: Path) -> bool:
    try:
        checkpoint = read_sealed(
            _unique_index_checkpoint_path(path), schema=DISTILLATION_SCHEMA
        )
    except DistillationStoreError:
        return False
    return (
        checkpoint.get("kind") == "unique-ledger-index-checkpoint"
        and checkpoint.get("index_name") == path.name
        and checkpoint.get("file_state") == _ledger_file_state(path)
    )


def _unique_index_metadata(
    path: Path,
    head: Mapping[str, Any],
    *,
    unique_field: str,
    binding_field: str,
) -> dict[str, str]:
    return {
        "schema": "chronovisor.unique-ledger-index.v1",
        "ledger_name": path.name,
        "unique_field": unique_field,
        "binding_field": binding_field,
        "records": str(head["records"]),
        "head_sha256": str(head["head_sha256"]),
        "file_state": canonical_json_bytes_strict(_ledger_file_state(path)).decode(),
    }


def _open_unique_index(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.close(descriptor)
    if stat.S_IMODE(path.stat().st_mode) != 0o600:
        os.chmod(path, 0o600)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = DELETE")
    connection.execute("PRAGMA synchronous = FULL")
    return connection


def _rebuild_unique_index(
    path: Path,
    *,
    unique_field: str,
    binding_field: str,
) -> tuple[dict[str, Any], sqlite3.Connection]:
    """Recover the ledger once, then atomically replace its derived index."""

    head = _recover_chain_tail(path)
    separator = _chain_separator(path)
    if separator:
        _append_fsynced(path, separator)
    _write_chain_checkpoint(path, head)
    index_path = _unique_index_path(path)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{index_path.name}.", suffix=".tmp", dir=index_path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        connection = _open_unique_index(temporary)
        try:
            connection.executescript(
                """
                CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE rows (
                    offset INTEGER PRIMARY KEY,
                    length INTEGER NOT NULL,
                    record_sha256 TEXT NOT NULL
                );
                CREATE TABLE entries (
                    unique_field TEXT NOT NULL,
                    binding_field TEXT NOT NULL,
                    identity BLOB NOT NULL,
                    binding BLOB NOT NULL,
                    offset INTEGER NOT NULL REFERENCES rows(offset),
                    PRIMARY KEY (unique_field, binding_field, identity, offset)
                );
                """
            )
            with path.open("rb") if path.exists() else contextlib.nullcontext() as handle:
                if handle is not None:
                    while line := handle.readline():
                        offset = handle.tell() - len(line)
                        row = json.loads(line)
                        if not isinstance(row, dict):
                            raise DistillationStoreError("ledger row is not an object")
                        connection.execute(
                            "INSERT INTO rows VALUES (?, ?, ?)",
                            (offset, len(line), str(row.get("record_sha256") or "")),
                        )
                        connection.execute(
                            "INSERT INTO entries VALUES (?, ?, ?, ?, ?)",
                            (
                                unique_field,
                                binding_field,
                                sqlite3.Binary(
                                    canonical_json_bytes_strict(row.get(unique_field))
                                ),
                                sqlite3.Binary(
                                    canonical_json_bytes_strict(row.get(binding_field))
                                ),
                                offset,
                            ),
                        )
            connection.executemany(
                "INSERT INTO metadata VALUES (?, ?)",
                _unique_index_metadata(
                    path,
                    head,
                    unique_field=unique_field,
                    binding_field=binding_field,
                ).items(),
            )
            connection.commit()
        finally:
            connection.close()
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, index_path)
        _fsync_directory(index_path)
        _write_unique_index_checkpoint(index_path)
        return head, _open_unique_index(index_path)
    except (OSError, sqlite3.Error, ValueError, UnicodeError) as exc:
        temporary.unlink(missing_ok=True)
        raise DistillationStoreError("unique ledger index rebuild failed") from exc


def _unique_index_matches(
    connection: sqlite3.Connection,
    path: Path,
    head: Mapping[str, Any],
    *,
    unique_field: str,
    binding_field: str,
) -> bool:
    try:
        metadata = dict(connection.execute("SELECT key, value FROM metadata"))
        return metadata == _unique_index_metadata(
            path,
            head,
            unique_field=unique_field,
            binding_field=binding_field,
        )
    except (sqlite3.Error, TypeError, ValueError):
        return False


def _indexed_unique_rows(
    connection: sqlite3.Connection,
    path: Path,
    identity: str,
    *,
    unique_field: str,
    binding_field: str,
) -> list[dict[str, Any]] | None:
    try:
        state = _ledger_file_state(path)
        file_size = int(state["size_bytes"]) if state is not None else 0
        entries = connection.execute(
            """
            SELECT rows.offset, rows.length, rows.record_sha256
            FROM entries JOIN rows USING (offset)
            WHERE entries.unique_field = ? AND entries.binding_field = ?
              AND entries.identity = ? ORDER BY rows.offset
            """,
            (
                unique_field,
                binding_field,
                sqlite3.Binary(canonical_json_bytes_strict(identity)),
            ),
        ).fetchall()
        if not entries:
            return []
        with path.open("rb") as handle:
            rows: list[dict[str, Any]] = []
            for offset, length, digest in entries:
                if (
                    not isinstance(offset, int)
                    or not isinstance(length, int)
                    or offset < 0
                    or length < 1
                    or offset > file_size
                    or length > file_size - offset
                ):
                    return None
                handle.seek(offset)
                line = handle.read(length)
                if len(line) != length or not line.endswith(b"\n"):
                    return None
                row = json.loads(line)
                if not isinstance(row, dict):
                    return None
                _require_chain_metadata(row, 0)
                unsigned = {
                    key: value for key, value in row.items() if key != "record_sha256"
                }
                if (
                    row.get("record_sha256") != digest
                    or row.get("record_sha256")
                    != canonical_json_sha256_strict(unsigned)
                    or row.get(unique_field) != identity
                ):
                    return None
                # The entry binding is deliberately re-derived from the exact JSONL row.
                if connection.execute(
                    """
                    SELECT binding FROM entries
                    WHERE unique_field = ? AND binding_field = ?
                      AND identity = ? AND offset = ?
                    """,
                    (
                        unique_field,
                        binding_field,
                        sqlite3.Binary(canonical_json_bytes_strict(identity)),
                        offset,
                    ),
                ).fetchone() != (
                    canonical_json_bytes_strict(row.get(binding_field)),
                ):
                    return None
                rows.append(row)
            return rows
    except (OSError, OverflowError, sqlite3.Error, ValueError, UnicodeError):
        return None


def _update_unique_index(
    connection: sqlite3.Connection,
    path: Path,
    head: Mapping[str, Any],
    row: Mapping[str, Any],
    *,
    offset: int,
    length: int,
    unique_field: str,
    binding_field: str,
) -> None:
    try:
        connection.execute(
            "INSERT INTO rows VALUES (?, ?, ?)",
            (offset, length, str(row["record_sha256"])),
        )
        connection.execute(
            "INSERT INTO entries VALUES (?, ?, ?, ?, ?)",
            (
                unique_field,
                binding_field,
                sqlite3.Binary(canonical_json_bytes_strict(row.get(unique_field))),
                sqlite3.Binary(canonical_json_bytes_strict(row.get(binding_field))),
                offset,
            ),
        )
        connection.execute("DELETE FROM metadata")
        connection.executemany(
            "INSERT INTO metadata VALUES (?, ?)",
            _unique_index_metadata(
                path,
                head,
                unique_field=unique_field,
                binding_field=binding_field,
            ).items(),
        )
        connection.commit()
        _write_unique_index_checkpoint(_unique_index_path(path))
    except sqlite3.Error as exc:
        connection.rollback()
        raise DistillationStoreError("unique ledger index update failed") from exc


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
    head = _read_chain_checkpoint(path)
    connection: sqlite3.Connection | None = None
    try:
        if head is not None:
            try:
                index_path = _unique_index_path(path)
                if _unique_index_checkpoint_matches(index_path):
                    connection = _open_unique_index(index_path)
            except (OSError, sqlite3.Error):
                connection = None
        if connection is None or head is None or not _unique_index_matches(
            connection,
            path,
            head,
            unique_field=unique_field,
            binding_field=binding_field,
        ):
            if connection is not None:
                connection.close()
                connection = None
            head, connection = _rebuild_unique_index(
                path, unique_field=unique_field, binding_field=binding_field
            )
        existing_rows = _indexed_unique_rows(
            connection,
            path,
            identity,
            unique_field=unique_field,
            binding_field=binding_field,
        )
        if existing_rows is None:
            connection.close()
            connection = None
            head, connection = _rebuild_unique_index(
                path, unique_field=unique_field, binding_field=binding_field
            )
            existing_rows = _indexed_unique_rows(
                connection,
                path,
                identity,
                unique_field=unique_field,
                binding_field=binding_field,
            )
            if existing_rows is None:
                raise DistillationStoreError("unique ledger index does not match ledger")
        for existing in existing_rows:
            if existing.get(binding_field) == binding:
                return existing
            raise DistillationStoreError("unique ledger identity conflict")
        unsigned = {
            "schema": DISTILLATION_SCHEMA,
            "namespace": "recall-distillation",
            "previous_sha256": head["head_sha256"],
            **payload,
        }
        row = {**unsigned, "record_sha256": canonical_json_sha256_strict(unsigned)}
        encoded = canonical_json_bytes_strict(row) + b"\n"
        separator = _chain_separator(path)
        offset = (_ledger_file_state(path) or {"size_bytes": 0})["size_bytes"] + len(
            separator
        )
        _append_fsynced(path, separator + encoded)
        updated_head = {
            "records": head["records"] + 1,
            "head_sha256": row["record_sha256"],
        }
        _write_chain_checkpoint(path, updated_head)
        _update_unique_index(
            connection,
            path,
            updated_head,
            row,
            offset=offset,
            length=len(encoded),
            unique_field=unique_field,
            binding_field=binding_field,
        )
        return row
    finally:
        if connection is not None:
            connection.close()


def _read_chain_locked(
    path: Path, head: Mapping[str, Any] | None = None
) -> list[dict[str, Any]]:
    if head is None:
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
        or (rows[-1].get("record_sha256") if rows else "")
        != head["head_sha256"]
    ):
        raise DistillationStoreError("ledger checkpoint does not match decoded rows")
    return rows


def read_chain(path: Path) -> list[dict[str, Any]]:
    """Decode a stable ledger snapshot after checkpoint-backed verification."""

    with _locked(path.with_suffix(path.suffix + ".lock")):
        return _read_chain_locked(path)


def label_health_projection(
    path: Path, *, repair: bool = False
) -> dict[str, Any]:
    """Read exact label health counts under the ledger lock."""

    with _locked(path.with_suffix(path.suffix + ".lock")):
        head = _read_chain_checkpoint(path)
        if head is None:
            head = _recover_chain_tail(path)
            _write_chain_checkpoint(path, head)
        projection = _read_label_health_projection(path, head)
        if projection is not None:
            return projection
        counts = _label_health_counts(_read_chain_locked(path, head))
        projection = {
            "label_chain_head": head["head_sha256"],
            "label_records": head["records"],
            "counts": counts,
        }
        if repair:
            try:
                _write_label_health_projection(path, head, counts)
            except (OSError, DistillationStoreError):
                # ponytail: the projection is an optimization; retry repair on
                # the next read instead of failing exact health reporting.
                pass
        return projection


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
        label_counts = label_health_projection(
            root / LABEL_LEDGER_FILE, repair=True
        )["counts"]
        teacher_only = label_counts["teacher_only"]
        verified_truth = label_counts["verified_truth"]
        probe_not_truth = label_counts["probe_not_truth"]
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
