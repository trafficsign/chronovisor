from __future__ import annotations

import json
import os
import sys
import time

from chronovisor.core import research_scheduler


def _paths(tmp_path, monkeypatch) -> None:
    root = tmp_path / "research"
    monkeypatch.setattr(research_scheduler, "RUNTIME_DIR", root)
    monkeypatch.setattr(research_scheduler, "SYNC_DIR", root / "sync-pending")
    monkeypatch.setattr(research_scheduler, "RESEARCH_LOCK", root / "research.lock")
    monkeypatch.setattr(research_scheduler, "ACTIVE_FILE", root / "active.json")
    monkeypatch.setattr(research_scheduler, "SCHEDULER_LOG", root / "scheduler.jsonl")


def test_auto_model_research_is_rejected_without_protected_capacity(tmp_path, monkeypatch) -> None:
    _paths(tmp_path, monkeypatch)
    monkeypatch.delenv("CHRONOVISOR_RESEARCH_CAPACITY_PROVEN", raising=False)

    with research_scheduler.research_lane(
        "run", enabled=True, mode="auto", purpose="auto", needs_model=True
    ) as lease:
        assert lease.admission.admitted is False
        assert lease.admission.reason == "protected_capacity_unproven"


def test_200_foreground_admissions_stay_under_wait_limit_when_research_is_denied(
    tmp_path, monkeypatch
) -> None:
    _paths(tmp_path, monkeypatch)
    monkeypatch.delenv("CHRONOVISOR_RESEARCH_CAPACITY_PROVEN", raising=False)
    waits = []
    for index in range(200):
        with research_scheduler.research_lane(
            f"auto-{index}",
            enabled=True,
            mode="auto",
            purpose="auto",
            needs_model=True,
        ) as lease:
            assert lease.admission.reason == "protected_capacity_unproven"
        with research_scheduler.foreground_lane(preempt_grace_ms=250) as receipt:
            waits.append(receipt.resource_wait_ms)

    p95 = sorted(waits)[int(len(waits) * 0.95) - 1]
    assert p95 <= 50


def test_foreground_diagnostics_do_not_use_durable_append(tmp_path, monkeypatch) -> None:
    _paths(tmp_path, monkeypatch)
    durable_calls: list[bool] = []
    monkeypatch.setattr(
        research_scheduler,
        "append_jsonl_durable",
        lambda *_args, **_kwargs: durable_calls.append(True),
    )

    with research_scheduler.foreground_lane(preempt_grace_ms=0):
        pass

    assert durable_calls == []
    deadline = time.monotonic() + 1
    while not research_scheduler.SCHEDULER_LOG.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    rows = [json.loads(line) for line in research_scheduler.SCHEDULER_LOG.read_text().splitlines()]
    while len(rows) < 2 and time.monotonic() < deadline:
        time.sleep(0.01)
        rows = [json.loads(line) for line in research_scheduler.SCHEDULER_LOG.read_text().splitlines()]
    assert [row["kind"] for row in rows] == ["sync_enter", "sync_exit"]


def test_foreground_diagnostic_failure_is_best_effort(tmp_path, monkeypatch) -> None:
    _paths(tmp_path, monkeypatch)
    monkeypatch.setattr(
        research_scheduler.os,
        "open",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("offline")),
    )

    with research_scheduler.foreground_lane(preempt_grace_ms=0) as receipt:
        assert receipt.resource_wait_ms <= 50


def test_foreground_diagnostic_thread_start_failure_is_best_effort(
    tmp_path, monkeypatch
) -> None:
    _paths(tmp_path, monkeypatch)

    class UnavailableThread:
        def __init__(self, **_kwargs) -> None:
            pass

        def start(self) -> None:
            raise RuntimeError("thread resources exhausted")

    monkeypatch.setattr(research_scheduler.threading, "Thread", UnavailableThread)

    with research_scheduler.foreground_lane(preempt_grace_ms=0) as receipt:
        assert receipt.resource_wait_ms <= 50


def test_foreground_slow_diagnostic_append_does_not_block(tmp_path, monkeypatch) -> None:
    _paths(tmp_path, monkeypatch)
    original_open = research_scheduler.os.open

    def slow_open(*args, **kwargs):
        time.sleep(0.2)
        return original_open(*args, **kwargs)

    monkeypatch.setattr(research_scheduler.os, "open", slow_open)
    started = time.monotonic()

    with research_scheduler.foreground_lane(preempt_grace_ms=0):
        pass

    assert time.monotonic() - started < 0.1


def test_foreground_marker_cancels_running_research_child(tmp_path, monkeypatch) -> None:
    _paths(tmp_path, monkeypatch)
    with research_scheduler.research_lane(
        "run", enabled=True, mode="explicit", purpose="explicit", needs_model=True
    ) as lease:
        import threading

        result = []
        thread = threading.Thread(
            target=lambda: result.append(
                research_scheduler.run_cancellable_command(
                    [sys.executable, "-c", "import time; time.sleep(5)"],
                    "",
                    lease,
                    timeout_seconds=10,
                )
            )
        )
        thread.start()
        time.sleep(0.05)
        with research_scheduler.foreground_lane(preempt_grace_ms=250) as receipt:
            thread.join(timeout=1)
            assert receipt.research_overlap is True
            assert receipt.preempted is True
            assert receipt.resource_wait_ms <= 50
        assert result[0].status == "cancelled"
        assert result[0].latency_ms < 1000


def test_closed_foreground_marker_still_classifies_sigkill_as_cancelled(
    tmp_path,
    monkeypatch,
) -> None:
    _paths(tmp_path, monkeypatch)
    with research_scheduler.research_lane(
        "run", enabled=True, mode="explicit", purpose="explicit", needs_model=True
    ) as lease:
        import threading

        result = []
        thread = threading.Thread(
            target=lambda: result.append(
                research_scheduler.run_cancellable_command(
                    [sys.executable, "-c", "import time; time.sleep(5)"],
                    "",
                    lease,
                    timeout_seconds=10,
                    poll_seconds=0.2,
                )
            )
        )
        thread.start()
        time.sleep(0.05)
        with research_scheduler.foreground_lane(preempt_grace_ms=0) as receipt:
            assert receipt.preempted is True
        thread.join(timeout=1)

        assert result[0].status == "cancelled"
        assert result[0].error == "cancelled for foreground sync"


def test_foreground_does_not_wait_for_non_model_research_phase(
    tmp_path, monkeypatch
) -> None:
    _paths(tmp_path, monkeypatch)
    with research_scheduler.research_lane(
        "run", enabled=True, mode="explicit", purpose="explicit", needs_model=True
    ):
        active = research_scheduler._active_research()
        assert active is not None
        assert active["model_active"] is False

        with research_scheduler.foreground_lane(preempt_grace_ms=250) as receipt:
            assert receipt.research_overlap is True
            assert receipt.preempted is False
            assert receipt.resource_wait_ms <= 50


def test_sync_pending_removes_dead_process_marker(tmp_path, monkeypatch) -> None:
    _paths(tmp_path, monkeypatch)
    marker_id = "0" * 32
    marker = research_scheduler.SYNC_DIR / f"12345-{marker_id}.json"
    marker.parent.mkdir(parents=True)
    marker.write_text(
        json.dumps({"pid": 12345, "marker_id": marker_id, "ts": "now"})
    )

    def dead_process(_pid: int, _signal: int) -> None:
        raise ProcessLookupError

    monkeypatch.setattr(research_scheduler.os, "kill", dead_process)

    assert research_scheduler.sync_pending() is False
    assert marker.exists() is False


def test_sync_pending_keeps_live_process_marker(tmp_path, monkeypatch) -> None:
    _paths(tmp_path, monkeypatch)
    marker_id = "0" * 32
    marker = research_scheduler.SYNC_DIR / f"{os.getpid()}-{marker_id}.json"
    marker.parent.mkdir(parents=True)
    marker.write_text(
        json.dumps({"pid": os.getpid(), "marker_id": marker_id, "ts": "now"})
    )

    assert research_scheduler.sync_pending() is True
    assert marker.exists() is True


def test_sync_pending_prevents_model_child_start(tmp_path, monkeypatch) -> None:
    _paths(tmp_path, monkeypatch)
    with research_scheduler.research_lane(
        "run", enabled=True, mode="explicit", purpose="explicit", needs_model=True
    ) as lease:
        research_scheduler.SYNC_DIR.mkdir(parents=True)
        (research_scheduler.SYNC_DIR / "pending.json").write_text("{}")
        result = research_scheduler.run_cancellable_command(
            [sys.executable, "-c", "raise SystemExit(99)"],
            "",
            lease,
            timeout_seconds=10,
        )

    assert result.status == "cancelled"
    assert result.error == "cancelled for foreground sync"


def test_cancellable_child_receives_stdin(tmp_path, monkeypatch) -> None:
    _paths(tmp_path, monkeypatch)
    with research_scheduler.research_lane(
        "run", enabled=True, mode="explicit", purpose="explicit", needs_model=True
    ) as lease:
        result = research_scheduler.run_cancellable_command(
            [
                sys.executable,
                "-c",
                "import json,sys; print(json.dumps({'value': sys.stdin.read()}))",
            ],
            "payload",
            lease,
            timeout_seconds=10,
        )

    assert result.status == "completed"
    assert result.value == {"value": "payload"}


def test_cancellable_child_drains_output_larger_than_pipe_buffer(
    tmp_path,
    monkeypatch,
) -> None:
    _paths(tmp_path, monkeypatch)
    with research_scheduler.research_lane(
        "run", enabled=True, mode="explicit", purpose="explicit", needs_model=True
    ) as lease:
        result = research_scheduler.run_cancellable_command(
            [
                sys.executable,
                "-c",
                "import json; print(json.dumps({'value': 'x' * 1_000_000}))",
            ],
            "",
            lease,
            timeout_seconds=10,
        )

    assert result.status == "completed"
    assert result.value == {"value": "x" * 1_000_000}
