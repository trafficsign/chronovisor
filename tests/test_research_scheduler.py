from __future__ import annotations

import sys
import time

from chronovisor import research_scheduler


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
