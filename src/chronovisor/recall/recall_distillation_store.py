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
from chronovisor.recall.recall_file_identity import persisted_file_state_matches

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


def projection_json(value: Any) -> str:
    """Encode projection metadata through the store's canonical JSON boundary."""

    return canonical_json_bytes_strict(value).decode("utf-8")


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
    after_persist: Callable[[], None] | None = None,
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
            created = False
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
                    created = True
                finally:
                    os.close(temporary_fd)
                    try:
                        os.unlink(temporary, dir_fd=directory_fd)
                    except FileNotFoundError:
                        pass
            if _read_pinned_regular(directory_fd, name) != encoded:
                raise DistillationStoreError("immutable artifact read-back failed")
            if after_persist is not None:
                try:
                    after_persist()
                except BaseException as exc:
                    if created:
                        if _read_pinned_regular(directory_fd, name) != encoded:
                            raise DistillationStoreError(
                                "immutable artifact changed during cleanup"
                            ) from exc
                        os.unlink(name, dir_fd=directory_fd)
                        os.fsync(directory_fd)
                    raise
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


_CHAIN_GENERATION_LENGTH = 32


def _new_chain_generation() -> str:
    return uuid.uuid4().hex


def _valid_chain_generation(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _CHAIN_GENERATION_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _checkpoint_head(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "records": checkpoint["records"],
        "head_sha256": checkpoint["head_sha256"],
    }


def _write_chain_checkpoint(
    path: Path, head: Mapping[str, Any], *, generation: str | None = None
) -> None:
    if generation is None:
        generation = _new_chain_generation()
    if not _valid_chain_generation(generation):
        raise DistillationStoreError("invalid ledger chain generation")
    write_sealed_state(
        _chain_checkpoint_path(path),
        {
            "kind": "ledger-chain-checkpoint",
            "ledger_name": path.name,
            "records": head["records"],
            "head_sha256": head["head_sha256"],
            "file_state": _ledger_file_state(path),
            "generation": generation,
        },
    )


def _read_chain_checkpoint_details(path: Path) -> dict[str, Any] | None:
    try:
        checkpoint = read_sealed(
            _chain_checkpoint_path(path), schema=DISTILLATION_SCHEMA
        )
    except DistillationStoreError:
        return None
    current_state = _ledger_file_state(path)
    generation = checkpoint.get("generation")
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
        or not persisted_file_state_matches(checkpoint.get("file_state"), current_state, path=path)
        or (current_state is None and checkpoint["records"] != 0)
        or ("generation" in checkpoint and not _valid_chain_generation(generation))
    ):
        return None
    return {
        "records": checkpoint["records"],
        "head_sha256": checkpoint["head_sha256"],
        "generation": generation,
    }


def _read_chain_checkpoint(path: Path) -> dict[str, Any] | None:
    checkpoint = _read_chain_checkpoint_details(path)
    return _checkpoint_head(checkpoint) if checkpoint is not None else None


_LEDGER_STATE_KEYS = (
    "size_bytes",
    "st_dev",
    "st_ino",
    "st_mtime_ns",
    "st_ctime_ns",
)


def _ledger_file_state_from_stat(value: os.stat_result) -> dict[str, int]:
    return {
        "size_bytes": value.st_size,
        "st_dev": value.st_dev,
        "st_ino": value.st_ino,
        "st_mtime_ns": value.st_mtime_ns,
        "st_ctime_ns": value.st_ctime_ns,
    }


def _validate_chain_tail_cursor(
    path: Path, cursor: Mapping[str, Any] | None
) -> tuple[dict[str, int] | None, int, int, str, str | None] | None:
    if not isinstance(cursor, Mapping):
        return None
    if cursor.get("ledger_path") != str(path.expanduser().absolute()):
        return None
    generation = cursor.get("generation")
    if generation is not None and not _valid_chain_generation(generation):
        return None
    raw_state = cursor.get("file_state")
    if raw_state is None:
        offset = cursor.get("offset")
        records = cursor.get("records")
        head = cursor.get("head_sha256")
        if (
            offset == 0
            and records == 0
            and not isinstance(offset, bool)
            and not isinstance(records, bool)
            and head == ""
        ):
            return None, 0, 0, "", generation
        return None
    if not isinstance(raw_state, Mapping):
        return None
    if set(raw_state) != set(_LEDGER_STATE_KEYS):
        return None
    state: dict[str, int] = {}
    for key in _LEDGER_STATE_KEYS:
        value = raw_state.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            return None
        state[key] = value
    offset = cursor.get("offset")
    records = cursor.get("records")
    if (
        not isinstance(offset, int)
        or isinstance(offset, bool)
        or offset < 0
        or not isinstance(records, int)
        or isinstance(records, bool)
        or records < 0
        or state["size_bytes"] != offset
    ):
        return None
    head = cursor.get("head_sha256")
    if not isinstance(head, str):
        return None
    if head and (
        len(head) != 64 or any(character not in "0123456789abcdef" for character in head)
    ):
        return None
    if (records == 0) != (head == ""):
        return None
    return state, offset, records, head, generation


def _chain_tail_cursor(
    path: Path,
    state: Mapping[str, int] | None,
    head: Mapping[str, Any],
    *,
    generation: str | None = None,
) -> dict[str, Any]:
    size = int(state["size_bytes"]) if state is not None else 0
    return {
        "ledger_path": str(path.expanduser().absolute()),
        "file_state": dict(state) if state is not None else None,
        "offset": size,
        "records": int(head["records"]),
        "head_sha256": str(head["head_sha256"]),
        "generation": generation,
    }


@contextlib.contextmanager
def _snapshot_lock(
    path: Path, *, readonly: bool = False, nonblocking: bool = False,
) -> Iterator[None]:
    lock_path = path.with_suffix(path.suffix + ".lock")
    if not readonly and not nonblocking:
        with _locked(lock_path):
            yield
        return
    if readonly:
        descriptor = os.open(lock_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            fcntl.flock(descriptor, fcntl.LOCK_SH | (fcntl.LOCK_NB if nonblocking else 0))
            yield
        finally:
            os.close(descriptor)
    else:
        handle = acquire_nonblocking_lock(lock_path)
        if handle is None:
            raise DistillationStoreBusy("ledger snapshot is busy")
        try:
            yield
        finally:
            release_lock(handle)


def _capture_ledger_snapshot(
    path: Path, *, readonly: bool = False, nonblocking: bool = False,
) -> tuple[int | None, dict[str, int] | None, dict[str, Any] | None]:
    """Pin a ledger descriptor and its verified writer checkpoint briefly."""

    with _snapshot_lock(path, readonly=readonly, nonblocking=nonblocking):
        state = _ledger_file_state(path)
        head = _read_chain_checkpoint_details(path)
        if readonly and head is None:
            raise DistillationStoreError("ledger checkpoint is unavailable")
        if state is None:
            return None, None, head
        try:
            descriptor = os.open(path, os.O_RDONLY)
        except OSError as exc:
            raise DistillationStoreBusy("ledger cannot be pinned") from exc
        try:
            descriptor_state = _ledger_file_state_from_stat(os.fstat(descriptor))
            if descriptor_state != state or _ledger_file_state(path) != state:
                raise DistillationStoreBusy("ledger changed while pinning snapshot")
        except Exception as exc:
            os.close(descriptor)
            if isinstance(exc, DistillationStoreBusy):
                raise
            raise DistillationStoreBusy("ledger cannot be pinned") from exc
        return descriptor, state, head


def _read_pinned_ledger(
    descriptor: int,
    path: Path,
    state: Mapping[str, int],
    *,
    start: int = 0,
) -> bytes:
    """Read one pinned file range, outside the ledger lock."""

    try:
        before = _ledger_file_state_from_stat(os.fstat(descriptor))
        if before != dict(state) or _ledger_file_state(path) != dict(state):
            raise DistillationStoreBusy("ledger changed before snapshot read")
        size = int(state["size_bytes"])
        if start < 0 or start > size:
            raise DistillationStoreError("ledger cursor offset is out of bounds")
        if start and os.pread(descriptor, 1, start - 1) != b"\n":
            raise DistillationStoreError("ledger cursor offset is not a row boundary")
        chunks: list[bytes] = []
        offset = start
        while offset < size:
            chunk = os.pread(descriptor, min(1024 * 1024, size - offset), offset)
            if not chunk:
                raise DistillationStoreBusy("ledger shortened during snapshot read")
            chunks.append(chunk)
            offset += len(chunk)
        after = _ledger_file_state_from_stat(os.fstat(descriptor))
        if after != dict(state) or _ledger_file_state(path) != dict(state):
            raise DistillationStoreBusy("ledger changed during snapshot read")
        return b"".join(chunks)
    except OSError as exc:
        raise DistillationStoreBusy("ledger changed during snapshot read") from exc
    finally:
        os.close(descriptor)


def _decode_chain_bytes(
    data: bytes,
    *,
    records: int = 0,
    previous: str = "",
    strict: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(data.splitlines(keepends=True), start=records):
        if not line.endswith(b"\n"):
            raise DistillationStoreError(f"ledger row {index} is incomplete")
        try:
            row = json.loads(line)
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise DistillationStoreError(f"ledger row {index} is invalid") from exc
        if not isinstance(row, dict):
            raise DistillationStoreError(f"ledger row {index} is not an object")
        _require_chain_metadata(row, index)
        if strict:
            unsigned = {
                key: value for key, value in row.items() if key != "record_sha256"
            }
            if row.get("previous_sha256") != previous or row.get(
                "record_sha256"
            ) != canonical_json_sha256_strict(unsigned):
                raise DistillationStoreError(f"ledger chain mismatch at row {index}")
            previous = str(row["record_sha256"])
        else:
            previous = str(row.get("record_sha256") or "")
        rows.append(row)
    return rows, {"records": records + len(rows), "head_sha256": previous}


def _decode_verified_snapshot(
    path: Path,
    state: Mapping[str, int],
    data: bytes,
    *,
    records: int = 0,
    previous: str = "",
    strict: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows, head = _decode_chain_bytes(
        data,
        records=records,
        previous=previous,
        strict=strict,
    )
    if _ledger_file_state(path) != dict(state):
        raise DistillationStoreBusy("ledger changed during snapshot decode")
    return rows, head


def _read_chain_snapshot_with_state(
    path: Path, *, strict: bool = False, readonly: bool = False, nonblocking: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, int] | None]:
    descriptor, state, checkpoint = _capture_ledger_snapshot(
        path, readonly=readonly, nonblocking=nonblocking,
    )
    if descriptor is None or state is None:
        if checkpoint is None:
            return [], {"records": 0, "head_sha256": ""}, state
        return [], _checkpoint_head(checkpoint), state
    data = _read_pinned_ledger(descriptor, path, state)
    rows, decoded = _decode_verified_snapshot(
        path,
        state,
        data,
        strict=strict or checkpoint is None,
    )
    checkpoint_head = _checkpoint_head(checkpoint) if checkpoint is not None else None
    if checkpoint_head is not None and decoded != checkpoint_head:
        raise DistillationStoreError("ledger checkpoint does not match decoded rows")
    return rows, checkpoint_head or decoded, state


def _read_chain_snapshot(
    path: Path, *, strict: bool = False
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows, head, _state = _read_chain_snapshot_with_state(path, strict=strict)
    return rows, head


def _read_chain_snapshot_recovering_final_fragment(
    path: Path, *, strict: bool = False, nonblocking: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, int] | None]:
    """Verify outside the lock and repair only an invalid final fragment."""

    try:
        return _read_chain_snapshot_with_state(path, strict=strict, nonblocking=nonblocking)
    except DistillationStoreError as exc:
        snapshot_error = exc

    descriptor, state, checkpoint = _capture_ledger_snapshot(path, nonblocking=nonblocking)
    if descriptor is None or state is None:
        raise snapshot_error
    data = _read_pinned_ledger(descriptor, path, state)
    lines = data.splitlines(keepends=True)
    if not lines:
        raise snapshot_error
    final_line = lines[-1]
    if final_line.endswith(b"\n") and final_line != b"\n":
        raise snapshot_error
    prefix_data = b"".join(lines[:-1])
    prefix_rows, prefix_head = _decode_chain_bytes(prefix_data, strict=True)
    checkpoint_head = _checkpoint_head(checkpoint) if checkpoint is not None else None

    if final_line != b"\n" and not final_line.endswith(b"\n"):
        try:
            final_row = json.loads(final_line)
        except (UnicodeError, json.JSONDecodeError):
            final_row = None
        if final_row is not None:
            if not isinstance(final_row, dict):
                raise snapshot_error
            _require_chain_metadata(final_row, len(prefix_rows))
            unsigned = {
                key: value
                for key, value in final_row.items()
                if key != "record_sha256"
            }
            if final_row.get("previous_sha256") == prefix_head["head_sha256"] and (
                final_row.get("record_sha256")
                == canonical_json_sha256_strict(unsigned)
            ):
                rows, head = _decode_chain_bytes(data + b"\n", strict=True)
                if checkpoint_head is not None and checkpoint_head != head:
                    raise DistillationStoreError(
                        "ledger checkpoint does not match decoded rows"
                    )
                return rows, head, state

    if checkpoint_head is not None and checkpoint_head != prefix_head:
        raise DistillationStoreError("ledger checkpoint does not match decoded rows")

    with _snapshot_lock(path, nonblocking=nonblocking):
        if _ledger_file_state(path) != state:
            raise DistillationStoreBusy("ledger changed during final-fragment repair")
        with path.open("r+b") as writable:
            writable.truncate(len(prefix_data))
            writable.flush()
            os.fsync(writable.fileno())
        _fsync_directory(path)
        _write_chain_checkpoint(path, prefix_head)
        repaired_state = _ledger_file_state(path)
    return prefix_rows, prefix_head, repaired_state


def _fsync_directory(path: Path) -> None:
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _recover_chain_tail(path: Path) -> dict[str, Any]:
    """Verify/recover through the shared snapshot path outside the ledger lock."""
    _rows, head, _state = _read_chain_snapshot_recovering_final_fragment(path, strict=True)
    return head


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
    rows = read_chain(path)
    return {
        "records": len(rows),
        "head_sha256": rows[-1]["record_sha256"] if rows else "",
    }


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
    chain_head(path)
    if path.name == LABEL_LEDGER_FILE:
        label_health_projection(path, repair=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with _locked(lock_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = _read_chain_checkpoint_details(path)
        if checkpoint is not None:
            head = _checkpoint_head(checkpoint)
            generation = checkpoint.get("generation") or _new_chain_generation()
        else:
            raise DistillationStoreBusy("ledger changed after append preparation")
        label_counts: dict[str, int] | None = None
        if path.name == LABEL_LEDGER_FILE:
            projection = _read_label_health_projection(path, head)
            if projection is None:
                raise DistillationStoreBusy("label health changed after append preparation")
            label_counts = dict(projection["counts"])
        # The checkpoint is a write-ahead recovery point for an interrupted append.
        _write_chain_checkpoint(path, head, generation=generation)
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
        _write_chain_checkpoint(path, updated_head, generation=generation)
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

    for attempt in range(3):
        try:
            prepare_unique_index(path, unique_field=unique_field, binding_field=binding_field)
            with _locked(path.with_suffix(path.suffix + ".lock")):
                return append_chain_unique_locked(
                    path, payload, unique_field=unique_field, binding_field=binding_field,
                )
        except DistillationStoreBusy:
            if attempt == 2:
                raise
    raise DistillationStoreBusy("unique ledger changed during append")


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
        and persisted_file_state_matches(checkpoint.get("file_state"), _ledger_file_state(path), path=path)
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


def prepare_unique_index(
    path: Path, *, unique_field: str, binding_field: str, nonblocking: bool = False,
) -> None:
    """Verify/rebuild a derived index before acquiring a ledger writer lock."""

    with _snapshot_lock(path, nonblocking=nonblocking):
        head = _read_chain_checkpoint(path)
        index_path = _unique_index_path(path)
        if head is not None and _unique_index_checkpoint_matches(index_path):
            try:
                with contextlib.closing(_open_unique_index(index_path)) as connection:
                    if _unique_index_matches(
                        connection, path, head,
                        unique_field=unique_field, binding_field=binding_field,
                    ):
                        return
            except (OSError, sqlite3.Error):
                pass
    _head, connection = _rebuild_unique_index(
        path, unique_field=unique_field, binding_field=binding_field,
        nonblocking=nonblocking,
    )
    connection.close()


def _rebuild_unique_index(
    path: Path,
    *,
    unique_field: str,
    binding_field: str,
    nonblocking: bool = False,
) -> tuple[dict[str, Any], sqlite3.Connection]:
    """Recover the ledger once, then atomically replace its derived index."""

    rows, head, expected_state = _read_chain_snapshot_recovering_final_fragment(
        path, strict=True, nonblocking=nonblocking,
    )
    with _snapshot_lock(path, nonblocking=nonblocking):
        if _ledger_file_state(path) != expected_state:
            raise DistillationStoreBusy("unique ledger changed during verification")
        separator = _chain_separator(path)
        if separator:
            _append_fsynced(path, separator)
        if _read_chain_checkpoint(path) != head:
            _write_chain_checkpoint(path, head)
        expected_state = _ledger_file_state(path)
    descriptor, captured_state, _checkpoint = _capture_ledger_snapshot(
        path, nonblocking=nonblocking,
    )
    if captured_state != expected_state:
        if descriptor is not None:
            os.close(descriptor)
        raise DistillationStoreBusy("unique ledger changed before indexing")
    data = (
        _read_pinned_ledger(descriptor, path, captured_state)
        if descriptor is not None and captured_state is not None else b""
    )
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
            offset = 0
            for line, row in zip(data.splitlines(keepends=True), rows, strict=True):
                connection.execute(
                    "INSERT INTO rows VALUES (?, ?, ?)",
                    (offset, len(line), str(row["record_sha256"])),
                )
                connection.execute(
                    "INSERT INTO entries VALUES (?, ?, ?, ?, ?)",
                    (
                        unique_field, binding_field,
                        sqlite3.Binary(canonical_json_bytes_strict(row.get(unique_field))),
                        sqlite3.Binary(canonical_json_bytes_strict(row.get(binding_field))),
                        offset,
                    ),
                )
                offset += len(line)
            connection.executemany(
                "INSERT INTO metadata VALUES (?, ?)",
                {
                    **_unique_index_metadata(
                        path, head, unique_field=unique_field, binding_field=binding_field,
                    ),
                    "file_state": projection_json(expected_state),
                }.items(),
            )
            connection.commit()
        finally:
            connection.close()
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        with _snapshot_lock(path, nonblocking=nonblocking):
            if _ledger_file_state(path) != expected_state or _read_chain_checkpoint(path) != head:
                raise DistillationStoreBusy("unique ledger changed during index build")
            os.replace(temporary, index_path)
            _fsync_directory(index_path)
            _write_unique_index_checkpoint(index_path)
            return head, _open_unique_index(index_path)
    except (OSError, sqlite3.Error, ValueError, UnicodeError) as exc:
        temporary.unlink(missing_ok=True)
        raise DistillationStoreError("unique ledger index rebuild failed") from exc
    finally:
        temporary.unlink(missing_ok=True)


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
    checkpoint = _read_chain_checkpoint_details(path)
    head = _checkpoint_head(checkpoint) if checkpoint is not None else None
    generation = checkpoint.get("generation") if checkpoint is not None else None
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
            raise DistillationStoreBusy("unique ledger index requires preparation")
        existing_rows = _indexed_unique_rows(
            connection,
            path,
            identity,
            unique_field=unique_field,
            binding_field=binding_field,
        )
        if existing_rows is None:
            raise DistillationStoreBusy("unique ledger index requires verification")
        label_projection = (
            _read_label_health_projection(path, head)
            if path.name == LABEL_LEDGER_FILE
            else None
        )
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
        _write_chain_checkpoint(
            path,
            updated_head,
            generation=generation or _new_chain_generation(),
        )
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
        if label_projection is not None:
            additions = _label_health_counts((row,))
            try:
                _write_label_health_projection(
                    path,
                    updated_head,
                    {
                        key: int(label_projection["counts"][key]) + additions[key]
                        for key in additions
                    },
                )
            except (OSError, DistillationStoreError):
                # The ledger and unique index are durable; health is rebuilt
                # from the exact chain if this optimization is interrupted.
                pass
        return row
    finally:
        if connection is not None:
            connection.close()


def read_chain(path: Path) -> list[dict[str, Any]]:
    """Decode a stable ledger snapshot without holding the ledger lock."""

    rows, head, expected_state = _read_chain_snapshot_recovering_final_fragment(path)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with _locked(lock_path):
        if _ledger_file_state(path) != expected_state:
            raise DistillationStoreBusy("ledger changed during checkpoint repair")
        checkpoint = _read_chain_checkpoint(path)
        if checkpoint != head:
            _write_chain_checkpoint(path, head)
    return rows


def read_chain_tail(
    path: Path, cursor: Mapping[str, Any] | None = None
) -> tuple[list[dict[str, Any]], dict[str, Any], bool]:
    """Read a verified complete ledger or only rows appended after ``cursor``.

    The cursor is a sealed projection's exact byte boundary.  A state mismatch
    never reuses its prefix: the full ledger is verified and ``reset`` is true.
    A valid writer checkpoint lets an append-only suffix be verified without
    decoding the old rows.
    """

    descriptor, state, checkpoint = _capture_ledger_snapshot(path)
    parsed = _validate_chain_tail_cursor(path, cursor)
    if descriptor is None or state is None:
        if parsed is not None and parsed[1] > 0:
            raise DistillationStoreError("ledger was truncated after the cursor")
        rows, head = _decode_chain_bytes(b"", strict=True)
        checkpoint_head = _checkpoint_head(checkpoint) if checkpoint is not None else None
        if checkpoint_head is not None and checkpoint_head != head:
            raise DistillationStoreError("ledger checkpoint does not match decoded rows")
        current = _chain_tail_cursor(
            path,
            state,
            checkpoint_head or head,
            generation=checkpoint.get("generation") if checkpoint is not None else None,
        )
        unchanged = (
            parsed is not None
            and parsed[0] is None
            and parsed[2:4] == (0, "")
            and parsed[4] == current.get("generation")
        )
        return rows, current, not unchanged
    if (
        parsed is not None
        and parsed[0] is not None
        and persisted_file_state_matches(parsed[0], state, path=path, identity_only=True)
        and state["size_bytes"] < parsed[1]
    ):
        os.close(descriptor)
        raise DistillationStoreError("ledger was truncated after the cursor")
    if checkpoint is None or parsed is None:
        data = _read_pinned_ledger(descriptor, path, state)
        rows, decoded = _decode_verified_snapshot(path, state, data, strict=True)
        checkpoint_head = _checkpoint_head(checkpoint) if checkpoint is not None else None
        if checkpoint_head is not None and checkpoint_head != decoded:
            raise DistillationStoreError("ledger checkpoint does not match decoded rows")
        return (
            rows,
            _chain_tail_cursor(
                path,
                state,
                checkpoint_head or decoded,
                generation=checkpoint.get("generation") if checkpoint is not None else None,
            ),
            True,
        )

    checkpoint_head = _checkpoint_head(checkpoint)
    current_generation = checkpoint.get("generation")
    cursor_state, offset, records, previous, cursor_generation = parsed
    same_identity = (
        cursor_state is not None
        and persisted_file_state_matches(cursor_state, state, path=path, identity_only=True)
    )
    if (
        not same_identity
        or state["size_bytes"] < offset
        or checkpoint["records"] < records
        or (
            checkpoint["records"] == records
            and checkpoint_head["head_sha256"] != previous
        )
        or cursor_generation != current_generation
    ):
        data = _read_pinned_ledger(descriptor, path, state)
        rows, decoded = _decode_verified_snapshot(path, state, data, strict=True)
        if decoded != checkpoint_head:
            raise DistillationStoreError("ledger checkpoint does not match decoded rows")
        return (
            rows,
            _chain_tail_cursor(
                path, state, checkpoint_head, generation=current_generation
            ),
            True,
        )

    if checkpoint["records"] == records:
        if not persisted_file_state_matches(cursor_state, state, path=path) or state["size_bytes"] != offset:
            data = _read_pinned_ledger(descriptor, path, state)
            rows, decoded = _decode_verified_snapshot(path, state, data, strict=True)
            if decoded != checkpoint_head:
                raise DistillationStoreError(
                    "ledger checkpoint does not match decoded rows"
                )
            return (
                rows,
                _chain_tail_cursor(
                    path, state, checkpoint_head, generation=current_generation
                ),
                True,
            )
        os.close(descriptor)
        return (
            [],
            _chain_tail_cursor(
                path, state, checkpoint_head, generation=current_generation
            ),
            False,
        )

    if state["size_bytes"] <= offset:
        data = _read_pinned_ledger(descriptor, path, state)
        rows, decoded = _decode_verified_snapshot(path, state, data, strict=True)
        if decoded != checkpoint_head:
            raise DistillationStoreError("ledger checkpoint does not match decoded rows")
        return (
            rows,
            _chain_tail_cursor(
                path, state, checkpoint_head, generation=current_generation
            ),
            True,
        )
    if current_generation is None or cursor_generation is None:
        data = _read_pinned_ledger(descriptor, path, state)
        rows, decoded = _decode_verified_snapshot(path, state, data, strict=True)
        if decoded != checkpoint_head:
            raise DistillationStoreError("ledger checkpoint does not match decoded rows")
        return (
            rows,
            _chain_tail_cursor(
                path, state, checkpoint_head, generation=current_generation
            ),
            True,
        )
    suffix = _read_pinned_ledger(descriptor, path, state, start=offset)
    tail, decoded = _decode_verified_snapshot(
        path,
        state,
        suffix,
        records=records,
        previous=previous,
        strict=True,
    )
    if decoded != checkpoint_head:
        raise DistillationStoreError("ledger checkpoint does not match decoded rows")
    return (
        tail,
        _chain_tail_cursor(path, state, checkpoint_head, generation=current_generation),
        False,
    )


def label_health_projection(
    path: Path, *, repair: bool = False
) -> dict[str, Any]:
    """Read exact label health counts without decoding under the ledger lock."""

    with _locked(path.with_suffix(path.suffix + ".lock")):
        head = _read_chain_checkpoint(path)
        projection = (
            _read_label_health_projection(path, head) if head is not None else None
        )
    if projection is not None:
        return projection

    rows, decoded_head, expected_state = _read_chain_snapshot_recovering_final_fragment(
        path, strict=True
    )
    with _locked(path.with_suffix(path.suffix + ".lock")):
        if _ledger_file_state(path) != expected_state:
            raise DistillationStoreBusy("ledger changed during health read")
        current = _read_chain_checkpoint(path)
        if current is None:
            _write_chain_checkpoint(path, decoded_head)
        elif current != decoded_head:
            raise DistillationStoreBusy("label checkpoint changed during health read")
    counts = _label_health_counts(rows)
    projection = {
        "label_chain_head": decoded_head["head_sha256"],
        "label_records": decoded_head["records"],
        "counts": counts,
    }
    if repair:
        with _locked(path.with_suffix(path.suffix + ".lock")):
            current = _read_chain_checkpoint(path)
            if current == decoded_head and _read_label_health_projection(path, current) is None:
                try:
                    _write_label_health_projection(path, decoded_head, counts)
                except (OSError, DistillationStoreError):
                    # ponytail: the projection is an optimization; retry repair
                    # on the next read instead of failing exact health reporting.
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
        last_label_at = state.get("last_label_at")
        teacher_profile = state.get("teacher_profile")
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
            "last_label_at": (
                last_label_at[:80] if isinstance(last_label_at, str) else ""
            ),
            "teacher_profile": (
                teacher_profile[:80] if isinstance(teacher_profile, str) else ""
            ),
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
            "last_label_at": "",
            "teacher_profile": "",
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
