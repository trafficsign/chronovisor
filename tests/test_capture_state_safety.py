from __future__ import annotations

import json
from pathlib import Path

import pytest

from chronovisor.core import background_jobs as jobs


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, "JOB_DIR", tmp_path)
    monkeypatch.setattr(jobs, "STATE_FILE", tmp_path / "state.json")
    monkeypatch.setattr(jobs, "LOCK_FILE", tmp_path / "state.lock")
    return jobs.STATE_FILE


def enqueue():
    return jobs.enqueue_job(
        name="codex-save",
        module="chronovisor.hosts.codex_record",
        args=[],
        env={},
        stdin_text="{}",
    )


def test_missing_capture_ledger_starts_empty(ledger):
    assert enqueue()["status"] == "queued"


@pytest.mark.parametrize("payload", [b'{"private transcript', b"\xff", b"[]", b'{"jobs": []}'])
def test_corrupt_ledger_is_preserved_and_blocks_reinitialization(ledger, payload):
    ledger.write_bytes(payload)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="capture job state") as failure:
            enqueue()
        assert "private transcript" not in str(failure.value)
    preserved = list(ledger.parent.glob("state.json.corrupt-*"))
    assert len(preserved) == 1
    assert preserved[0].read_bytes() == payload
    assert not ledger.exists()
    status = jobs.snapshot()
    assert status["status"] == "unavailable"
    assert status["error"] == "capture_job_state_corrupt"


def test_unreadable_ledger_is_preserved(ledger, monkeypatch):
    payload = b'{"schema_version":1,"jobs":{}}'
    ledger.write_bytes(payload)
    read = Path.read_text

    def unreadable(path, *args, **kwargs):
        if path == ledger:
            raise PermissionError("private transcript")
        return read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", unreadable)
    with pytest.raises(RuntimeError, match="capture job state"):
        enqueue()
    assert next(ledger.parent.glob("state.json.corrupt-*")).read_bytes() == payload


def test_snapshot_reports_corruption_without_mutating_ledger(ledger):
    ledger.write_bytes(b"{private transcript")
    before = ledger.read_bytes(), ledger.stat().st_mtime_ns
    status = jobs.snapshot()
    assert status["status"] == "unavailable"
    assert "private transcript" not in json.dumps(status)
    assert (ledger.read_bytes(), ledger.stat().st_mtime_ns) == before
