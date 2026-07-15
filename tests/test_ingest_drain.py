"""Tests for the launchd-friendly ingest drain CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_wiki_mcp import ingest_drain


@pytest.fixture(autouse=True)
def _disable_runtime_status_reset(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        ingest_drain.runtime_status,
        "reset_stale_runtime_status",
        lambda: False,
    )
    monkeypatch.setattr(ingest_drain, "WIKI_ROOT", tmp_path / "wiki")


def test_drain_runs_batches_until_empty(tmp_path: Path, monkeypatch) -> None:
    state = {"pending": 25, "init": 0, "reset": 0}

    monkeypatch.setattr(
        ingest_drain, "init_wiki", lambda: state.__setitem__("init", state["init"] + 1)
    )
    monkeypatch.setattr(
        ingest_drain.orchestrator,
        "reset_stale_lock",
        lambda: state.__setitem__("reset", state["reset"] + 1),
    )
    monkeypatch.setattr(
        ingest_drain.orchestrator,
        "get_pending_raw_files",
        lambda: [object()] * state["pending"],
    )

    def fake_run_pending_ingest(*, force: bool = False, max_units: int = 10) -> dict:
        assert force is True
        assert max_units == 10
        processed = min(max_units, state["pending"])
        state["pending"] -= processed
        return {
            "triggered": True,
            "files_processed": [f"raw-{i}.md" for i in range(processed)],
        }

    monkeypatch.setattr(
        ingest_drain.orchestrator, "run_pending_ingest", fake_run_pending_ingest
    )

    log_file = tmp_path / "drain.jsonl"
    result = ingest_drain.drain(max_batches=3, sleep_seconds=0, log_file=log_file)

    assert result["status"] == "drained"
    assert result["pending_start"] == 25
    assert result["pending_after"] == 0
    assert result["batches_run"] == 3
    assert result["files_processed"] == 25
    assert state["init"] == 1
    assert state["reset"] == 1

    records = [json.loads(line) for line in log_file.read_text().splitlines()]
    assert [record["files_processed"] for record in records] == [10, 10, 5]


def test_drain_stops_when_batch_makes_no_progress(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(ingest_drain, "init_wiki", lambda: None)
    monkeypatch.setattr(ingest_drain.orchestrator, "reset_stale_lock", lambda: None)
    monkeypatch.setattr(
        ingest_drain.orchestrator, "get_pending_raw_files", lambda: [object()] * 3
    )
    monkeypatch.setattr(
        ingest_drain.orchestrator,
        "run_pending_ingest",
        lambda *, force=False, max_units=10: {
            "triggered": True,
            "files_processed": [],
        },
    )

    result = ingest_drain.drain(
        max_batches=5, sleep_seconds=0, log_file=tmp_path / "drain.jsonl"
    )

    assert result["status"] == "stalled"
    assert result["pending_after"] == 3
    assert result["batches_run"] == 1
    assert result["stop_reason"] == "no batch progress"


def test_drain_check_mode_does_not_run_ingest(monkeypatch) -> None:
    monkeypatch.setattr(ingest_drain, "init_wiki", lambda: None)
    monkeypatch.setattr(ingest_drain.orchestrator, "reset_stale_lock", lambda: None)
    monkeypatch.setattr(
        ingest_drain.orchestrator, "get_pending_raw_files", lambda: [object()] * 2
    )
    monkeypatch.setattr(
        ingest_drain.orchestrator,
        "run_pending_ingest",
        lambda *, force=False, max_units=10: (_ for _ in ()).throw(
            AssertionError("should not run")
        ),
    )

    result = ingest_drain.drain(max_batches=0)

    assert result["status"] == "checked"
    assert result["pending_after"] == 2
    assert result["batches_run"] == 0


def test_drain_forwards_single_unit_pilot_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = {"pending": 2}
    calls: list[tuple[bool, int]] = []

    monkeypatch.setattr(ingest_drain, "init_wiki", lambda: None)
    monkeypatch.setattr(ingest_drain.orchestrator, "reset_stale_lock", lambda: None)
    monkeypatch.setattr(
        ingest_drain.orchestrator,
        "get_pending_raw_files",
        lambda: [object()] * state["pending"],
    )

    def fake_run_pending_ingest(*, force: bool, max_units: int) -> dict:
        calls.append((force, max_units))
        state["pending"] -= 1
        return {"triggered": True, "files_processed": ["raw-0.md"]}

    monkeypatch.setattr(
        ingest_drain.orchestrator,
        "run_pending_ingest",
        fake_run_pending_ingest,
    )

    result = ingest_drain.drain(
        max_batches=1,
        max_units=1,
        sleep_seconds=0,
        log_file=tmp_path / "pilot.jsonl",
    )

    assert calls == [(True, 1)]
    assert result["status"] == "partial"
    assert result["files_processed"] == 1
    assert result["pending_after"] == 1


@pytest.mark.parametrize("max_units", [0, 11])
def test_drain_rejects_out_of_range_max_units(max_units: int) -> None:
    with pytest.raises(ValueError, match="max_units must be between 1 and 10"):
        ingest_drain.drain(max_units=max_units)


def test_parser_reads_max_units_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_WIKI_INGEST_DRAIN_MAX_UNITS", "1")

    args = ingest_drain.build_parser().parse_args([])

    assert args.max_units == 1
