"""Tests for runtime observability files."""

from __future__ import annotations

from pathlib import Path

from chronovisor.core import runtime_status


def patch_runtime(tmp_path: Path, monkeypatch) -> Path:
    runtime_dir = tmp_path / "runtime"
    monkeypatch.setattr(runtime_status, "RUNTIME_DIR", runtime_dir)
    monkeypatch.setattr(runtime_status, "STATUS_FILE", runtime_dir / "status.json")
    monkeypatch.setattr(runtime_status, "EVENTS_FILE", runtime_dir / "events.jsonl")
    monkeypatch.setattr(runtime_status, "METRICS_FILE", runtime_dir / "metrics.jsonl")
    return runtime_dir


def test_status_event_and_metric_roundtrip(tmp_path: Path, monkeypatch) -> None:
    patch_runtime(tmp_path, monkeypatch)

    runtime_status.write_status({"state": "running", "pending": 7})
    runtime_status.append_event("success", "ingest | completed: 1 created")
    runtime_status.append_metric("batch", pending_before=7, pending_after=6, files_processed=1)

    status = runtime_status.read_status()
    assert status["state"] == "running"
    assert status["pending"] == 7
    assert status["last_event"]["message"] == "ingest | completed: 1 created"
    assert runtime_status.read_events()[0]["level"] == "success"
    assert runtime_status.read_metrics()[0]["pending_after"] == 6


def test_safe_helpers_swallow_io_errors(monkeypatch) -> None:
    monkeypatch.setattr(runtime_status, "write_status", lambda _fields: (_ for _ in ()).throw(OSError("nope")))
    monkeypatch.setattr(runtime_status, "append_event", lambda *_a, **_kw: (_ for _ in ()).throw(OSError("nope")))
    monkeypatch.setattr(runtime_status, "append_metric", lambda *_a, **_kw: (_ for _ in ()).throw(OSError("nope")))

    runtime_status.safe_write_status(state="running")
    runtime_status.safe_append_event("info", "hello")
    runtime_status.safe_append_metric("batch")


def test_classify_log_message() -> None:
    assert runtime_status.classify_log_message("ingest | completed: 1 created") == "success"
    assert runtime_status.classify_log_message("ingest | partial: 1 dead-lettered") == "warn"
    assert runtime_status.classify_log_message("ingest | generate parse failed") == "error"
    assert runtime_status.classify_log_message("ingest | stage 1: triage started") == "info"


def test_reset_stale_runtime_status_clears_dead_live_pid(tmp_path: Path, monkeypatch) -> None:
    patch_runtime(tmp_path, monkeypatch)
    monkeypatch.setattr(runtime_status, "_pid_is_alive", lambda _pid: False)

    runtime_status.write_status({
        "state": "running",
        "stage": "generate",
        "current_job_pid": 12345,
        "current_job_id": "job-1",
        "current_raw": "raw.md",
        "current_op": "page.md",
        "llm": {"active": True, "updated_at": runtime_status.now_iso()},
    })

    assert runtime_status.reset_stale_runtime_status() is True
    status = runtime_status.read_status()
    assert status["state"] == "idle"
    assert status["stage"] == "waiting"
    assert status["current_job_pid"] is None
    assert status["llm"] is None
    assert runtime_status.read_events()[-1]["source"] == "runtime"


def test_reset_stale_runtime_status_keeps_live_pid(tmp_path: Path, monkeypatch) -> None:
    patch_runtime(tmp_path, monkeypatch)
    monkeypatch.setattr(runtime_status, "_pid_is_alive", lambda _pid: True)

    runtime_status.write_status({
        "state": "running",
        "stage": "generate",
        "current_job_pid": 12345,
        "llm": {"active": True, "updated_at": runtime_status.now_iso()},
    })

    assert runtime_status.reset_stale_runtime_status() is False
    status = runtime_status.read_status()
    assert status["state"] == "running"
    assert status["llm"]["active"] is True
