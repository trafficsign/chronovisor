from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from chronovisor.capture import background_jobs as jobs


@pytest.mark.parametrize("command", ["run", "retry", "status"])
def test_job_cli_prints_only_operational_metadata(tmp_path, monkeypatch, capsys, command):
    monkeypatch.setattr(jobs, "JOB_DIR", tmp_path)
    monkeypatch.setattr(jobs, "STATE_FILE", tmp_path / "state.json")
    monkeypatch.setattr(jobs, "LOCK_FILE", tmp_path / "state.lock")
    monkeypatch.setattr(
        jobs.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=1, stdout="private output", stderr="private transcript"
        ),
    )
    job = jobs.enqueue_job(
        name="codex-save",
        module="chronovisor.hosts.codex_record",
        args=[],
        env={},
        stdin_text="private hook payload",
    )
    jobs.main([command, job["job_id"]] if command == "run" else [command])
    output = capsys.readouterr().out
    assert "private" not in output
    assert "stdin" not in output
    assert "output_tail" not in output
    for line in output.splitlines():
        assert isinstance(json.loads(line), dict)
