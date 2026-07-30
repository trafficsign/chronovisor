"""Cross-process coordination for foreground MPS inference."""

from __future__ import annotations

import fcntl
import time
from collections.abc import Iterator
from contextlib import contextmanager

from chronovisor.core.store import CHRONOVISOR_ROOT

ACCELERATOR_LOCK = CHRONOVISOR_ROOT / "runtime" / "accelerator-inference.lock"


class AcceleratorLeaseTimeout(TimeoutError):
    pass


@contextmanager
def accelerator_lease(*, timeout_ms: int) -> Iterator[float]:
    """Serialize short foreground MPS jobs across resident model services."""

    ACCELERATOR_LOCK.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    deadline = started + max(1, timeout_ms) / 1_000
    with ACCELERATOR_LOCK.open("a+") as handle:
        while True:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError as exc:
                if time.monotonic() >= deadline:
                    raise AcceleratorLeaseTimeout(
                        "foreground accelerator lease timed out"
                    ) from exc
                time.sleep(0.005)
        try:
            yield (time.monotonic() - started) * 1_000
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
