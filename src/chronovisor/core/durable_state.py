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
import stat
import tempfile
import threading
from collections.abc import Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager, suppress
from pathlib import Path
from typing import IO, Any

from chronovisor.core.canonical_json import canonical_json_line_bytes_strict

SEAL_FIELD = "seal_sha256"
DEFAULT_MIN_FREE_BYTES = 16 * 1024 * 1024
_LOCK_STATE = threading.local()


class DurableStateError(RuntimeError):
    """A state object could not be verified or durably published."""


class DiskPressureError(DurableStateError):
    """The destination filesystem has too little free space to write safely."""


class StateSealError(DurableStateError):
    """A JSON object is missing or does not match its content seal."""


@contextmanager
def open_directory_nofollow(path: Path) -> Iterator[int]:
    """Pin one directory while refusing symlinks in every path component."""

    absolute = path.absolute()
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    directory_flags |= getattr(os, "O_CLOEXEC", 0)
    directory_fd = os.open(absolute.anchor, directory_flags)
    try:
        for part in absolute.parts[1:]:
            next_fd = os.open(part, directory_flags, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = next_fd
        yield directory_fd
    finally:
        os.close(directory_fd)


@contextmanager
def open_regular_nofollow(path: Path) -> Iterator[IO[bytes]]:
    """Open one regular file while refusing symlinks in every path component."""

    absolute = path.absolute()
    file_fd = -1
    with open_directory_nofollow(absolute.parent) as directory_fd:
        try:
            flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
            file_fd = os.open(absolute.name, flags, dir_fd=directory_fd)
            if not stat.S_ISREG(os.fstat(file_fd).st_mode):
                raise ValueError("path is not a regular file")
            with os.fdopen(file_fd, "rb", closefd=False) as handle:
                yield handle
        finally:
            if file_fd >= 0:
                os.close(file_fd)


def atomic_write_bytes_at(directory_fd: int, name: str, raw: bytes) -> None:
    """Atomically publish one regular file relative to a pinned directory fd."""

    if not isinstance(raw, bytes):
        raise TypeError("raw must be bytes")
    if not name or name != Path(name).name or name in {".", ".."}:
        raise ValueError("atomic filename must be one safe path component")
    try:
        existing = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        existing = None
    if existing is not None and not stat.S_ISREG(existing.st_mode):
        raise ValueError("atomic destination is not a regular file")
    filesystem = os.fstatvfs(directory_fd)
    free = filesystem.f_bavail * filesystem.f_frsize
    if free < len(raw) * 2:
        raise DiskPressureError("insufficient free space for durable write")
    temporary = f".{name}.{os.getpid()}.{os.urandom(8).hex()}.tmp"
    descriptor = -1
    published = False
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
        flags |= getattr(os, "O_CLOEXEC", 0)
        descriptor = os.open(temporary, flags, 0o600, dir_fd=directory_fd)
        offset = 0
        while offset < len(raw):
            written = os.write(descriptor, raw[offset:])
            if written <= 0:
                raise OSError("atomic write made no progress")
            offset += written
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.replace(
            temporary,
            name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
        )
        published = True
        os.fsync(directory_fd)
        flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
        descriptor = os.open(name, flags, dir_fd=directory_fd)
        snapshot = os.fstat(descriptor)
        if not stat.S_ISREG(snapshot.st_mode) or snapshot.st_size != len(raw):
            raise DurableStateError("durable read-back identity mismatch")
        observed = bytearray()
        while len(observed) < len(raw):
            chunk = os.read(descriptor, len(raw) - len(observed))
            if not chunk:
                break
            observed.extend(chunk)
        if bytes(observed) != raw:
            raise DurableStateError("durable read-back mismatch")
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if not published:
            with suppress(FileNotFoundError):
                os.unlink(temporary, dir_fd=directory_fd)


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
def file_lock(
    path: Path,
    *,
    exclusive: bool = True,
    blocking: bool = True,
    fsync_on_open: bool = False,
    dir_fd: int | None = None,
) -> Iterator[int]:
    if dir_fd is None:
        path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_RDWR
        | os.O_CREAT
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
        dir_fd=dir_fd,
    )
    owner_pid = os.getpid()
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise OSError("lock path is not a regular file")
        os.fchmod(descriptor, 0o600)
        if fsync_on_open:
            os.fsync(descriptor)
        operation = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        if not blocking:
            operation |= fcntl.LOCK_NB
        fcntl.flock(descriptor, operation)
        yield descriptor
    finally:
        if os.getpid() == owner_pid:
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


def sidecar_exclusive_lock(path: Path) -> AbstractContextManager[None]:
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
            with suppress(FileNotFoundError):
                temporary.unlink()


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
            raise DurableStateError(
                f"cannot read durable state: {path}: {exc}"
            ) from exc
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


@contextmanager
def reentrant_exclusive_lock(lock_path: Path) -> Iterator[bool]:
    """Hold one process lock, allowing only same-thread nested entry."""

    lock_path = lock_path.resolve(strict=False)
    process_id = os.getpid()
    state_pid = getattr(_LOCK_STATE, "process_id", None)
    if state_pid != process_id:
        # A fork must never inherit the parent's in-memory ownership claim.
        _LOCK_STATE.process_id = process_id
        _LOCK_STATE.depths = {}
    depths: dict[str, int] = _LOCK_STATE.depths
    identity = os.fspath(lock_path)
    depth = depths.get(identity, 0)
    if depth:
        depths[identity] = depth + 1
        try:
            yield False
        finally:
            remaining = depths[identity] - 1
            if remaining:
                depths[identity] = remaining
            else:
                depths.pop(identity, None)
        return

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+b")
    locked = False
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        locked = True
        depths[identity] = 1
        yield True
    finally:
        try:
            depths.pop(identity, None)
            if locked:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def atomic_write(path: Path, content: str) -> None:
    """tempfile + os.replace による atomic write。

    同一ディレクトリに dot-prefix の一時ファイルを作り、fsync 後に rename で置換。
    書き込み失敗時は tempfile をクリーンアップする。
    """
    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    )
    tmp_path = tmp.name
    try:
        tmp.write(content)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp.close()
        os.replace(tmp_path, path)
        dir_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except Exception:
        with suppress(OSError):
            os.unlink(tmp_path)
        raise
