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
OKF_WRITER_LOCK_FILENAME = "okf-writer.lock"
_OKF_LOCK_STATE = threading.local()


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


def _reject_symlink_components(path: Path) -> None:
    current = path.absolute()
    while True:
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            pass
        else:
            if stat.S_ISLNK(mode):
                raise ValueError("OKF writer lock root contains a symlink")
        if current == current.parent:
            return
        current = current.parent


def _runtime_entry_kind(directory_fd: int, name: str) -> str | None:
    try:
        mode = os.stat(name, dir_fd=directory_fd, follow_symlinks=False).st_mode
    except FileNotFoundError:
        return None
    return "file" if stat.S_ISREG(mode) else "unsafe"


def okf_writer_lease_is_exclusive(root: Path) -> bool:
    """Return whether this thread currently holds the root's exclusive lease."""

    if getattr(_OKF_LOCK_STATE, "process_id", None) != os.getpid():
        return False
    expanded = root.expanduser()
    if not expanded.is_absolute():
        return False
    leases = getattr(_OKF_LOCK_STATE, "leases", None)
    if not isinstance(leases, dict):
        return False
    active = leases.get(os.fspath(expanded.absolute()))
    return bool(active is not None and active[0])


@contextmanager
def okf_writer_lock(
    root: Path,
    *,
    exclusive: bool = False,
    allow_create: bool = True,
    blocking: bool = True,
) -> Iterator[None]:
    """Hold the root-scoped OKF writer lease in shared or exclusive mode."""

    root = root.expanduser()
    if not root.is_absolute():
        raise ValueError("OKF writer lock root must be absolute")
    root = root.absolute()
    if root == root.parent:
        raise ValueError("OKF writer lock root must not be the filesystem root")
    _reject_symlink_components(root.parent)
    if exclusive and not root.parent.is_dir():
        raise ValueError("exclusive OKF writer lock root must exist")
    bootstrap = not (root / "runtime" / OKF_WRITER_LOCK_FILENAME).exists()

    process_id = os.getpid()
    if getattr(_OKF_LOCK_STATE, "process_id", None) != process_id:
        _OKF_LOCK_STATE.process_id = process_id
        _OKF_LOCK_STATE.leases = {}
    leases: dict[str, tuple[bool, int]] = _OKF_LOCK_STATE.leases
    identity = os.fspath(root)
    active = leases.get(identity)
    if active is not None:
        active_exclusive, depth = active
        if exclusive and not active_exclusive:
            raise RuntimeError("cannot upgrade a shared OKF writer lease")
        leases[identity] = (active_exclusive, depth + 1)
        try:
            yield
        finally:
            remaining = leases[identity][1] - 1
            if remaining:
                leases[identity] = (active_exclusive, remaining)
            else:
                leases.pop(identity, None)
        return

    if not exclusive:
        root.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    parent_fd = os.open(
        root.parent,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    owner_pid = os.getpid()
    parent_locked = False
    root_fd: int | None = None
    root_locked = False
    runtime_fd: int | None = None
    try:
        hierarchy_exclusive = exclusive or bootstrap
        while True:
            parent_operation = (
                fcntl.LOCK_EX if hierarchy_exclusive else fcntl.LOCK_SH
            )
            if exclusive or not blocking:
                parent_operation |= fcntl.LOCK_NB
            fcntl.flock(parent_fd, parent_operation)
            parent_locked = True
            root_created = False
            try:
                root_fd = os.open(
                    root.name,
                    os.O_RDONLY
                    | getattr(os, "O_CLOEXEC", 0)
                    | getattr(os, "O_DIRECTORY", 0)
                    | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=parent_fd,
                )
            except FileNotFoundError as exc:
                if not hierarchy_exclusive:
                    fcntl.flock(parent_fd, fcntl.LOCK_UN)
                    parent_locked = False
                    hierarchy_exclusive = True
                    continue
                if exclusive:
                    raise ValueError(
                        "exclusive OKF writer lock root must exist"
                    ) from exc
                os.mkdir(root.name, mode=0o700, dir_fd=parent_fd)
                root_created = True
                root_fd = os.open(
                    root.name,
                    os.O_RDONLY
                    | getattr(os, "O_CLOEXEC", 0)
                    | getattr(os, "O_DIRECTORY", 0)
                    | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=parent_fd,
                )
            root_operation = (
                fcntl.LOCK_EX if hierarchy_exclusive else fcntl.LOCK_SH
            )
            if exclusive or not blocking:
                root_operation |= fcntl.LOCK_NB
            fcntl.flock(root_fd, root_operation)
            root_locked = True
            runtime_created = False
            try:
                runtime_fd = os.open(
                    "runtime",
                    os.O_RDONLY
                    | getattr(os, "O_CLOEXEC", 0)
                    | getattr(os, "O_DIRECTORY", 0)
                    | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=root_fd,
                )
            except FileNotFoundError as exc:
                if not hierarchy_exclusive:
                    fcntl.flock(root_fd, fcntl.LOCK_UN)
                    root_locked = False
                    os.close(root_fd)
                    root_fd = None
                    fcntl.flock(parent_fd, fcntl.LOCK_UN)
                    parent_locked = False
                    hierarchy_exclusive = True
                    continue
                if not allow_create:
                    raise RuntimeError(
                        "OKF writer lock is missing from migration runtime"
                    ) from exc
                try:
                    os.mkdir("runtime", mode=0o700, dir_fd=root_fd)
                    runtime_created = True
                except FileExistsError:
                    pass
                runtime_fd = os.open(
                    "runtime",
                    os.O_RDONLY
                    | getattr(os, "O_CLOEXEC", 0)
                    | getattr(os, "O_DIRECTORY", 0)
                    | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=root_fd,
                )
            lock_kind = _runtime_entry_kind(runtime_fd, OKF_WRITER_LOCK_FILENAME)
            if lock_kind is None and not hierarchy_exclusive:
                os.close(runtime_fd)
                runtime_fd = None
                fcntl.flock(root_fd, fcntl.LOCK_UN)
                root_locked = False
                os.close(root_fd)
                root_fd = None
                fcntl.flock(parent_fd, fcntl.LOCK_UN)
                parent_locked = False
                hierarchy_exclusive = True
                continue
            break
        try:
            if lock_kind == "unsafe":
                raise RuntimeError("OKF writer lock entry is unsafe")
            created = lock_kind is None
            migrations_kind = _runtime_entry_kind(runtime_fd, "migrations")
            if created and (not allow_create or migrations_kind is not None):
                raise RuntimeError("OKF writer lock is missing from migration runtime")
            with file_lock(
                Path(OKF_WRITER_LOCK_FILENAME),
                exclusive=exclusive,
                blocking=blocking and not exclusive,
                fsync_on_open=created,
                dir_fd=runtime_fd,
            ):
                if created:
                    os.fsync(runtime_fd)
                    if runtime_created:
                        os.fsync(root_fd)
                    if root_created:
                        os.fsync(parent_fd)
                if hierarchy_exclusive and not exclusive:
                    fcntl.flock(root_fd, fcntl.LOCK_SH)
                    fcntl.flock(parent_fd, fcntl.LOCK_SH)
                leases[identity] = (exclusive, 1)
                try:
                    yield
                finally:
                    leases.pop(identity, None)
        finally:
            os.close(runtime_fd)
            runtime_fd = None
    except BlockingIOError as exc:
        raise RuntimeError("OKF writer lease is busy") from exc
    finally:
        if root_fd is not None:
            if root_locked and os.getpid() == owner_pid:
                fcntl.flock(root_fd, fcntl.LOCK_UN)
            os.close(root_fd)
        if parent_locked and os.getpid() == owner_pid:
            fcntl.flock(parent_fd, fcntl.LOCK_UN)
        os.close(parent_fd)


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
