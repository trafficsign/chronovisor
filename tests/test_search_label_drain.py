from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

from chronovisor.core.okf_cutover import OKFStartupDecision
from scripts import search_label_drain


def test_run_drain_repeats_bounded_batches_until_queue_is_empty(monkeypatch) -> None:
    responses = iter(
        [
            {
                "status": "ok",
                "attempted": 2,
                "promoted": 1,
                "remaining": 3,
                "status_counts": {"pending_review": 2},
            },
            {
                "status": "ok",
                "attempted": 2,
                "promoted": 1,
                "remaining": 0,
                "status_counts": {"frontier_approved": 2},
            },
        ]
    )
    calls: list[dict] = []

    def review(**kwargs):
        calls.append(kwargs)
        return next(responses)

    monkeypatch.setattr(
        search_label_drain.search_eval,
        "review_label_queue_with_frontier",
        review,
    )

    result = search_label_drain.run_drain(
        queue_file=Path("queue.jsonl"),
        golden_file=Path("golden.jsonl"),
        batch_size=2,
        max_elapsed_seconds=5,
    )

    assert result["status"] == "drained"
    assert result["attempted"] == 4
    assert result["promoted"] == 2
    assert result["remaining"] == 0
    assert len(calls) == 2
    assert all(call["review_order"] == "balanced" for call in calls)
    assert all(call["dry_run"] is False for call in calls)


def test_run_drain_dry_run_is_one_bounded_call(monkeypatch) -> None:
    calls: list[dict] = []

    def review(**kwargs):
        calls.append(kwargs)
        return {
            "status": "ok",
            "attempted": kwargs["limit"],
            "promoted": 0,
            "remaining": 10,
            "status_counts": {"pending_review": kwargs["limit"]},
        }

    monkeypatch.setattr(
        search_label_drain.search_eval,
        "review_label_queue_with_frontier",
        review,
    )

    result = search_label_drain.run_drain(
        batch_size=3,
        max_elapsed_seconds=5,
        dry_run=True,
    )

    assert result["status"] == "dry_run"
    assert result["attempted"] == 3
    assert len(calls) == 1


def test_run_drain_hides_reviewer_exception_details(monkeypatch) -> None:
    def review(**_kwargs):
        raise RuntimeError("candidate payload must not be printed")

    monkeypatch.setattr(
        search_label_drain.search_eval,
        "review_label_queue_with_frontier",
        review,
    )

    result = search_label_drain.run_drain(max_elapsed_seconds=5)

    assert result["status"] == "error"
    assert result["error_class"] == "RuntimeError"
    assert "candidate payload" not in str(result)


def test_scheduled_drain_defers_maintenance_and_busy_writer(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(search_label_drain.store, "CHRONOVISOR_ROOT", tmp_path)
    marker = tmp_path / "runtime/recall-luna-bulk/supervisor-maintenance.json"
    marker.parent.mkdir(parents=True)
    marker.write_text("{}")
    calls = []
    monkeypatch.setattr(search_label_drain, "run_drain", lambda **kw: calls.append(kw))
    assert search_label_drain.main([]) == 0
    assert json.loads(capsys.readouterr().out)["reason"] == "maintenance"
    marker.unlink()

    @contextmanager
    def busy(root, *, blocking):
        assert root == tmp_path and blocking is False
        raise search_label_drain.OKFStartupBlocked(
            OKFStartupDecision(False, "blocked", "blocked", "writer_lease_unavailable")
        )
        yield  # pragma: no cover

    monkeypatch.setattr(search_label_drain.store, "okf_runtime_operation", busy)
    assert search_label_drain.main([]) == 0
    assert json.loads(capsys.readouterr().out)["reason"] == "okf_startup_blocked"
    assert calls == []


def test_scheduled_drain_holds_writer_lease_during_review(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(search_label_drain.store, "CHRONOVISOR_ROOT", tmp_path)
    held = []

    @contextmanager
    def operation(root, *, blocking):
        assert root == tmp_path and blocking is False
        held.append(True)
        try:
            yield
        finally:
            held.pop()

    def review(**kwargs):
        assert held and kwargs["batch_size"] == 20
        return {"status": "bounded", "attempted": 20}

    monkeypatch.setattr(search_label_drain.store, "okf_runtime_operation", operation)
    monkeypatch.setattr(search_label_drain, "run_drain", review)
    assert search_label_drain.main([]) == 0
    assert json.loads(capsys.readouterr().out)["attempted"] == 20
    assert held == []
