from __future__ import annotations

from llm_wiki_mcp import (
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

    result = converge_worker.run_converge(session_limit=2, job_limit=3)

    assert result["status"] == "ok"
    assert result["background_jobs"]["limit"] == 3
    assert result["system_repairs"]["limit"] == 2
    assert result["session_sweeper"]["limit"] == 2
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
