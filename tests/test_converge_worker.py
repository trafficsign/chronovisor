from __future__ import annotations

from chronovisor.ops import (
    background_jobs,
    converge_worker,
    self_heal,
    session_sweeper,
    sleep_cycle,
)
from chronovisor.recall import recall_distillation


def _stub_lightweight_lanes(monkeypatch) -> None:
    monkeypatch.setattr(
        background_jobs,
        "retry_due",
        lambda *, limit: {"status": "ok", "limit": limit},
    )
    monkeypatch.setattr(
        session_sweeper,
        "run_sweeper",
        lambda *, limit: {"status": "ok", "limit": limit},
    )
    monkeypatch.setattr(
        self_heal,
        "enqueue_due_system_repairs",
        lambda *, limit: {"status": "ok", "limit": limit},
    )
    monkeypatch.setattr(
        converge_worker,
        "run_maintenance_batch",
        lambda **_kwargs: {"status": "ok"},
    )


def test_converge_default_is_lightweight_and_never_runs_sleep(monkeypatch) -> None:
    _stub_lightweight_lanes(monkeypatch)
    monkeypatch.setattr(
        sleep_cycle,
        "run_sleep_cycle",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("default converge must not run the full sleep cycle")
        ),
    )
    monkeypatch.setattr(
        converge_worker,
        "run_maintenance_batch",
        lambda **kwargs: {"status": "ok", **kwargs},
    )
    monkeypatch.setattr(recall_distillation, "distillation_enabled", lambda: False)

    result = converge_worker.run_converge(session_limit=2, job_limit=3)

    assert result["status"] == "ok"
    assert result["background_jobs"]["limit"] == 3
    assert result["system_repairs"]["limit"] == 2
    assert result["session_sweeper"]["limit"] == 2
    assert result["maintenance"]["lint_limit"] == 50
    assert result["maintenance"]["orphan_limit"] == 8
    assert "sleep_cycle" not in result
    assert result["recall_distillation"] == {
        "status": "skipped",
        "reason": "cold_start_not_due",
    }


def test_converge_full_sleep_requires_explicit_opt_in(monkeypatch) -> None:
    _stub_lightweight_lanes(monkeypatch)
    monkeypatch.setattr(recall_distillation, "distillation_enabled", lambda: False)
    calls: list[dict[str, object]] = []

    def fake_sleep_cycle(**kwargs):
        calls.append(kwargs)
        return {"status": "ok"}

    monkeypatch.setattr(sleep_cycle, "run_sleep_cycle", fake_sleep_cycle)

    result = converge_worker.run_converge(run_sleep=True)

    assert result["sleep_cycle"]["status"] == "ok"
    assert calls == [
        {
            "raw_limit": 25,
            "eval_limit": 25,
            "duplicate_limit": 100,
            "dry_run": False,
        }
    ]


def test_converge_can_disable_system_repair_enqueue(monkeypatch) -> None:
    _stub_lightweight_lanes(monkeypatch)
    monkeypatch.setattr(
        self_heal,
        "enqueue_due_system_repairs",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("system repair enqueue must be disabled")
        ),
    )
    monkeypatch.setattr(recall_distillation, "distillation_enabled", lambda: False)

    result = converge_worker.run_converge(
        session_limit=0,
        job_limit=8,
        run_system_repairs=False,
    )

    assert result["status"] == "ok"
    assert result["system_repairs"] == {
        "status": "skipped",
        "reason": "disabled_by_cli",
    }


def test_converge_can_disable_maintenance(monkeypatch) -> None:
    _stub_lightweight_lanes(monkeypatch)
    monkeypatch.setattr(
        converge_worker,
        "run_maintenance_batch",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("maintenance must be disabled")
        ),
    )
    monkeypatch.setattr(recall_distillation, "distillation_enabled", lambda: False)

    result = converge_worker.run_converge(run_maintenance=False)

    assert result["maintenance"] == {
        "status": "skipped",
        "reason": "disabled_by_cli",
    }


def test_converge_runs_one_bounded_cold_start_chunk_when_due(monkeypatch) -> None:
    _stub_lightweight_lanes(monkeypatch)
    monkeypatch.setattr(recall_distillation, "distillation_enabled", lambda: True)
    monkeypatch.setattr(recall_distillation, "cold_start_due", lambda: True)
    calls: list[dict[str, object]] = []

    def fake_chunk(**kwargs):
        calls.append(kwargs)
        return {"status": "deferred", "reason": "worker_busy"}

    monkeypatch.setattr(recall_distillation, "run_distillation_chunk", fake_chunk)

    result = converge_worker.run_converge()

    assert calls == [{"cold_start": True, "max_elapsed_seconds": 300}]
    assert result["recall_distillation"] == {
        "status": "deferred",
        "reason": "worker_busy",
    }
    assert result["status"] == "ok"


def test_converge_starts_cold_start_before_maintenance(monkeypatch) -> None:
    events: list[str] = []
    monkeypatch.setattr(
        background_jobs,
        "retry_due",
        lambda *, limit: events.append("background_jobs") or {"status": "ok"},
    )
    monkeypatch.setattr(
        session_sweeper,
        "run_sweeper",
        lambda *, limit: events.append("session_sweeper") or {"status": "ok"},
    )
    monkeypatch.setattr(
        self_heal,
        "enqueue_due_system_repairs",
        lambda *, limit: events.append("system_repairs") or {"status": "ok"},
    )
    monkeypatch.setattr(
        converge_worker,
        "run_maintenance_batch",
        lambda **_kwargs: events.append("maintenance") or {"status": "ok"},
    )
    monkeypatch.setattr(recall_distillation, "distillation_enabled", lambda: True)
    monkeypatch.setattr(recall_distillation, "cold_start_due", lambda: True)
    monkeypatch.setattr(
        recall_distillation,
        "run_distillation_chunk",
        lambda **_kwargs: events.append("distillation") or {"status": "deferred"},
    )

    converge_worker.run_converge()

    assert events[0] == "distillation"
    assert events.index("distillation") < events.index("maintenance")


def test_converge_skips_cold_start_after_completion(monkeypatch) -> None:
    _stub_lightweight_lanes(monkeypatch)
    monkeypatch.setattr(recall_distillation, "distillation_enabled", lambda: True)
    monkeypatch.setattr(recall_distillation, "cold_start_due", lambda: False)
    monkeypatch.setattr(
        recall_distillation,
        "run_distillation_chunk",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("must not run")),
    )

    result = converge_worker.run_converge()

    assert result["recall_distillation"] == {
        "status": "skipped",
        "reason": "cold_start_not_due",
    }


def test_converge_isolates_cold_start_error(monkeypatch) -> None:
    _stub_lightweight_lanes(monkeypatch)
    monkeypatch.setattr(recall_distillation, "distillation_enabled", lambda: True)
    monkeypatch.setattr(recall_distillation, "cold_start_due", lambda: True)
    monkeypatch.setattr(
        recall_distillation,
        "run_distillation_chunk",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    result = converge_worker.run_converge()

    assert result["background_jobs"] == {"status": "ok", "limit": 8}
    assert result["recall_distillation"] == {
        "status": "error",
        "error": "RuntimeError: boom",
    }
    assert result["status"] == "attention"
