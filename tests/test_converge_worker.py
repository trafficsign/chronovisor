from __future__ import annotations

from chronovisor import (
    background_jobs,
    converge_worker,
    self_heal,
    session_sweeper,
    sleep_cycle,
)


def test_converge_default_is_lightweight_and_never_runs_sleep(monkeypatch) -> None:
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

    result = converge_worker.run_converge(session_limit=2, job_limit=3)

    assert result["status"] == "ok"
    assert result["background_jobs"]["limit"] == 3
    assert result["system_repairs"]["limit"] == 2
    assert result["session_sweeper"]["limit"] == 2
    assert result["maintenance"]["lint_limit"] == 50
    assert result["maintenance"]["orphan_limit"] == 8
    assert "sleep_cycle" not in result


def test_converge_full_sleep_requires_explicit_opt_in(monkeypatch) -> None:
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
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("system repair enqueue must be disabled")
        ),
    )
    monkeypatch.setattr(
        converge_worker,
        "run_maintenance_batch",
        lambda **_kwargs: {"status": "ok"},
    )

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
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("maintenance must be disabled")
        ),
    )

    result = converge_worker.run_converge(run_maintenance=False)

    assert result["maintenance"] == {
        "status": "skipped",
        "reason": "disabled_by_cli",
    }
