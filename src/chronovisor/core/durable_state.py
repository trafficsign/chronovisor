"""Small, deterministic durability primitives for autonomy state.

The semantic data plane has stricter stores of its own.  This module is for
rebuildable runtime projections, leases, heartbeats, and content-addressed
decision artifacts.  Writes are sealed, fsynced, lock-serialized, backed up,
and rejected before publication when the destination filesystem is critically
low on space.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping

from chronovisor.core.canonical_json import canonical_json_line_bytes_strict


SEAL_FIELD = "seal_sha256"
DEFAULT_MIN_FREE_BYTES = 16 * 1024 * 1024


class DurableStateError(RuntimeError):
    """A state object could not be verified or durably published."""


class DiskPressureError(DurableStateError):
    """The destination filesystem has too little free space to write safely."""


class StateSealError(DurableStateError):
    """A JSON object is missing or does not match its content seal."""


def canonical_bytes(value: Any) -> bytes:
    return canonical_json_line_bytes_strict(value)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def seal_object(payload: Mapping[str, Any]) -> dict[str, Any]:
    sealed = {str(key): value for key, value in payload.items() if key != SEAL_FIELD}
    sealed[SEAL_FIELD] = canonical_sha256(sealed)
    return sealed


def verify_sealed_object(payload: object) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise StateSealError("sealed state must be a JSON object")
    observed = payload.get(SEAL_FIELD)
    if not isinstance(observed, str) or len(observed) != 64:
        raise StateSealError("sealed state is missing seal_sha256")
    unsigned = {key: value for key, value in payload.items() if key != SEAL_FIELD}
    expected = canonical_sha256(unsigned)
    if observed != expected:
        raise StateSealError("sealed state digest mismatch")
    return dict(payload)


@contextmanager
def file_lock(path: Path, *, exclusive: bool = True) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


@contextmanager
def exclusive_text_file_lock(path: Path) -> Iterator[None]:
    """Lock an appendable UTF-8 sidecar using blocking exclusive ``flock``."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def sidecar_exclusive_lock(path: Path):
    """Lock ``<path>.lock`` with :func:`exclusive_text_file_lock`."""

    return exclusive_text_file_lock(path.with_suffix(path.suffix + ".lock"))


def fsync_directory(path: Path) -> None:
    """Durably publish directory-entry changes under ``path``."""

    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_bytes(
    path: Path,
    raw: bytes,
    *,
    backup: bool = True,
    min_free_bytes: int = DEFAULT_MIN_FREE_BYTES,
) -> None:
    if not isinstance(raw, bytes):
        raise TypeError("raw must be bytes")
    path.parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(path.parent).free
    required = max(0, int(min_free_bytes)) + len(raw) * 2
    if free < required:
        raise DiskPressureError(
            f"insufficient free space for durable write ({free}<{required})"
        )
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        if backup and path.exists():
            backup_path = path.with_name(f"{path.name}.bak")
            prior = path.read_bytes()
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=path.parent,
                prefix=f".{backup_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as backup_stream:
                backup_tmp = Path(backup_stream.name)
                backup_stream.write(prior)
                backup_stream.flush()
                os.fsync(backup_stream.fileno())
            os.replace(backup_tmp, backup_path)
        os.replace(temporary, path)
        temporary = None
        fsync_directory(path.parent)
        if path.read_bytes() != raw:
            raise DurableStateError(f"durable read-back mismatch: {path}")
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def write_sealed_json(
    path: Path,
    payload: Mapping[str, Any],
    *,
    backup: bool = True,
    min_free_bytes: int = DEFAULT_MIN_FREE_BYTES,
) -> dict[str, Any]:
    sealed = seal_object(payload)
    atomic_write_bytes(
        path,
        canonical_bytes(sealed),
        backup=backup,
        min_free_bytes=min_free_bytes,
    )
    return sealed


def read_sealed_json(
    path: Path,
    *,
    recover_backup: bool = False,
) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return verify_sealed_object(payload)
    except (OSError, UnicodeError, json.JSONDecodeError, StateSealError) as exc:
        if not recover_backup:
            if isinstance(exc, StateSealError):
                raise
            raise DurableStateError(f"cannot read durable state: {path}: {exc}") from exc
    backup_path = path.with_name(f"{path.name}.bak")
    try:
        payload = json.loads(backup_path.read_text(encoding="utf-8"))
        recovered = verify_sealed_object(payload)
    except (OSError, UnicodeError, json.JSONDecodeError, StateSealError) as exc:
        raise DurableStateError(
            f"primary and backup durable state are invalid: {path}"
        ) from exc
    atomic_write_bytes(path, canonical_bytes(recovered), backup=False)
    return recovered
