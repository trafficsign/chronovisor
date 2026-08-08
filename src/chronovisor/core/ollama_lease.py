"""Cross-process resource leases for local Ollama operations."""

from __future__ import annotations

import errno
import fcntl
import os
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from chronovisor.core.store import CHRONOVISOR_ROOT

_RESOURCE_LEASE_STATE = threading.local()


class _ProcessResourceLock:
    """A writer-preferring reader/writer lock for this Python process."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._readers = 0
        self._writer_thread: int | None = None
        self._waiting_writers = 0

    def acquire(self, *, exclusive: bool, timeout_s: float | None = None) -> bool:
        thread_id = threading.get_ident()
        with self._condition:
            if exclusive:
                self._waiting_writers += 1
                try:
                    acquired = self._condition.wait_for(
                        lambda: self._writer_thread is None and self._readers == 0,
                        timeout=timeout_s,
                    )
                    if acquired:
                        self._writer_thread = thread_id
                    return acquired
                finally:
                    self._waiting_writers -= 1
            acquired = self._condition.wait_for(
                lambda: self._writer_thread is None and self._waiting_writers == 0,
                timeout=timeout_s,
            )
            if acquired:
                self._readers += 1
            return acquired

    def release(self, *, exclusive: bool) -> None:
        with self._condition:
            if exclusive:
                if self._writer_thread != threading.get_ident():
                    raise RuntimeError(
                        "resource lease writer released by another thread"
                    )
                self._writer_thread = None
            else:
                if self._readers < 1:
                    raise RuntimeError("resource lease reader count underflow")
                self._readers -= 1
            self._condition.notify_all()


_PROCESS_RESOURCE_LOCK = _ProcessResourceLock()


def _acquire_file_lease(
    handle: Any,
    *,
    exclusive: bool,
    deadline_at: float | None,
) -> None:
    operation = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
    if deadline_at is None:
        fcntl.flock(handle.fileno(), operation)
        return
    while True:
        try:
            fcntl.flock(handle.fileno(), operation | fcntl.LOCK_NB)
            return
        except OSError as exc:
            if exc.errno not in {errno.EACCES, errno.EAGAIN}:
                raise
        remaining_s = deadline_at - time.monotonic()
        if remaining_s <= 0:
            raise TimeoutError("Ollama resource lease acquisition timed out")
        time.sleep(min(0.005, remaining_s))


@contextmanager
def model_resource_lease(
    *,
    exclusive: bool,
    timeout_ms: int | None = None,
    root: Path = CHRONOVISOR_ROOT,
) -> Iterator[None]:
    """Coordinate inference and runner eviction across threads and processes.

    A thread holding an exclusive lease may safely enter shared or exclusive
    code without weakening its lease. Upgrading a shared lease to exclusive
    is rejected instead of risking an upgrade deadlock.
    """

    if timeout_ms is not None and (
        isinstance(timeout_ms, bool)
        or not isinstance(timeout_ms, int)
        or timeout_ms < 0
    ):
        raise ValueError("timeout_ms must be a non-negative integer")
    depth = int(getattr(_RESOURCE_LEASE_STATE, "depth", 0))
    if depth > 0:
        held_exclusive = bool(getattr(_RESOURCE_LEASE_STATE, "exclusive", False))
        if exclusive and not held_exclusive:
            raise RuntimeError(
                "cannot upgrade a shared Ollama resource lease to exclusive"
            )
        _RESOURCE_LEASE_STATE.depth = depth + 1
        try:
            yield
        finally:
            _RESOURCE_LEASE_STATE.depth -= 1
        return
    lock_path = Path(
        os.environ.get(
            "CHRONOVISOR_OLLAMA_RESOURCE_LOCK",
            str(root / "runtime/ollama-resource.lock"),
        )
    )
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    deadline_at = None if timeout_ms is None else time.monotonic() + (timeout_ms / 1000)
    acquired_process_lease = _PROCESS_RESOURCE_LOCK.acquire(
        exclusive=exclusive,
        timeout_s=(
            None if deadline_at is None else max(0.0, deadline_at - time.monotonic())
        ),
    )
    if not acquired_process_lease:
        raise TimeoutError("Ollama resource lease acquisition timed out")
    try:
        with lock_path.open("a+") as handle:
            _acquire_file_lease(
                handle,
                exclusive=exclusive,
                deadline_at=deadline_at,
            )
            _RESOURCE_LEASE_STATE.depth = 1
            _RESOURCE_LEASE_STATE.exclusive = exclusive
            try:
                yield
            finally:
                _RESOURCE_LEASE_STATE.depth = 0
                _RESOURCE_LEASE_STATE.exclusive = False
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        _PROCESS_RESOURCE_LOCK.release(exclusive=exclusive)


def model_resource_lease_mode() -> str | None:
    """Return the current thread's nested resource-lease mode, if any."""

    if int(getattr(_RESOURCE_LEASE_STATE, "depth", 0)) < 1:
        return None
    return (
        "exclusive"
        if bool(getattr(_RESOURCE_LEASE_STATE, "exclusive", False))
        else "shared"
    )
