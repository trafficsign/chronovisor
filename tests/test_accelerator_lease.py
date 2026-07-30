from __future__ import annotations

import threading
import time

import pytest

from chronovisor.search import accelerator_lease


def test_accelerator_lease_serializes_threads(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        accelerator_lease,
        "ACCELERATOR_LOCK",
        tmp_path / "accelerator.lock",
    )
    entered = threading.Event()
    release = threading.Event()

    def hold_lease() -> None:
        with accelerator_lease.accelerator_lease(timeout_ms=500):
            entered.set()
            release.wait(timeout=1)

    thread = threading.Thread(target=hold_lease)
    thread.start()
    assert entered.wait(timeout=1)
    with pytest.raises(accelerator_lease.AcceleratorLeaseTimeout):
        with accelerator_lease.accelerator_lease(timeout_ms=25):
            pass
    release.set()
    thread.join(timeout=1)


def test_accelerator_lease_reports_wait_time(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        accelerator_lease,
        "ACCELERATOR_LOCK",
        tmp_path / "accelerator.lock",
    )
    started = time.monotonic()
    with accelerator_lease.accelerator_lease(timeout_ms=100) as waited_ms:
        assert waited_ms >= 0
    assert time.monotonic() - started < 0.1
