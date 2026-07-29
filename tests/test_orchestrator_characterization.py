from __future__ import annotations

import json
from pathlib import Path

from chronovisor.ingest import orchestrator


def _state(**overrides) -> dict:
    state = {
        "last_ingest": None,
        "last_lint": None,
        "processed_raw_files": [],
        "ollama_health": {"status": None, "checked_at": None},
        "current_job_id": None,
        "current_job_pid": None,
        "current_job_started_at": None,
    }
    state.update(overrides)
    return state


def test_load_state_drops_unsafe_legacy_batch_failure_counter(
    tmp_path: Path, monkeypatch
) -> None:
    path = tmp_path / "state.json"
    path.write_text(
        json.dumps(_state(triage_failure_count=9, current_job_id="job-a")),
        encoding="utf-8",
    )
    monkeypatch.setattr(orchestrator, "STATE_FILE", path)

    loaded = orchestrator._load_state()

    assert "triage_failure_count" not in loaded
    assert loaded["current_job_id"] == "job-a"
    assert "triage_failure_count" in json.loads(path.read_text(encoding="utf-8"))


def test_reset_stale_pending_reservation_clears_all_owner_fields(
    tmp_path: Path, monkeypatch
) -> None:
    path = tmp_path / "state.json"
    path.write_text(
        json.dumps(
            _state(
                current_job_id="__pending__",
                current_job_pid=123,
                current_job_started_at="2026-07-17T12:00:00",
            )
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(orchestrator, "STATE_FILE", path)
    monkeypatch.setattr(orchestrator, "_lock_is_fresh_in_live_process", lambda _state: False)

    orchestrator.reset_stale_lock()

    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert persisted["current_job_id"] is None
    assert persisted["current_job_pid"] is None
    assert persisted["current_job_started_at"] is None
    assert path.read_text(encoding="utf-8").endswith("\n")


def test_reset_stale_lock_preserves_live_cross_process_owner(
    tmp_path: Path, monkeypatch
) -> None:
    path = tmp_path / "state.json"
    original = _state(
        current_job_id="job-a",
        current_job_pid=456,
        current_job_started_at="2026-07-17T12:00:00",
    )
    path.write_text(json.dumps(original), encoding="utf-8")
    monkeypatch.setattr(orchestrator, "STATE_FILE", path)
    monkeypatch.setattr(orchestrator, "_lock_is_fresh_in_live_process", lambda _state: True)

    orchestrator.reset_stale_lock()

    assert json.loads(path.read_text(encoding="utf-8")) == original


def test_mark_one_raw_processed_preserves_batch_reservation(
    tmp_path: Path, monkeypatch
) -> None:
    path = tmp_path / "state.json"
    path.write_text(
        json.dumps(_state(current_job_id="job-a", current_job_pid=456)),
        encoding="utf-8",
    )
    monkeypatch.setattr(orchestrator, "STATE_FILE", path)

    orchestrator._mark_one_raw_processed("raw-a.md")

    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert persisted["processed_raw_files"] == ["raw-a.md"]
    assert persisted["current_job_id"] == "job-a"
    assert persisted["current_job_pid"] == 456
