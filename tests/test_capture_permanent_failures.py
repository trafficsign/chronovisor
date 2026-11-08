from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from chronovisor.capture import background_jobs, claude_code_record, codex_record
from chronovisor.capture.transcript import ClaudeCodeSaveError, CodexSaveError


def _configure_jobs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(background_jobs, "JOB_DIR", tmp_path)
    monkeypatch.setattr(background_jobs, "STATE_FILE", tmp_path / "state.json")
    monkeypatch.setattr(background_jobs, "LOCK_FILE", tmp_path / "state.lock")


@pytest.mark.parametrize(
    ("module", "error"),
    (
        (codex_record, CodexSaveError),
        (claude_code_record, ClaudeCodeSaveError),
    ),
)
def test_permanent_record_error_exits_once(
    module: object,
    error: type[Exception],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise error("bad input")

    monkeypatch.setattr(module, "run", fail)

    code = module.main(["--session-file", "missing.jsonl", "--save"])

    assert code == background_jobs.PERMANENT_EXIT_CODE
    assert json.loads(capsys.readouterr().out)["status"] == "error"


@pytest.mark.parametrize("module", (codex_record, claude_code_record))
def test_transient_record_error_keeps_retrying(
    module: object,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise OSError("disk unavailable")

    monkeypatch.setattr(module, "run", fail)

    assert module.main(["--session-file", "session.jsonl", "--save"]) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "error"


@pytest.mark.parametrize("module", (codex_record, claude_code_record))
def test_missing_transcript_is_a_permanent_failure(
    module: object,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    missing = str(tmp_path / "missing.jsonl")

    code = module.main(["--session-file", missing, "--save"])

    assert code == background_jobs.PERMANENT_EXIT_CODE
    assert json.loads(capsys.readouterr().out)["status"] == "error"


@pytest.mark.parametrize(
    ("exit_code", "expected_status", "retried"),
    (
        (background_jobs.PERMANENT_EXIT_CODE, "failed", False),
        (1, "retry_wait", True),
    ),
)
def test_job_outcome_for_permanent_and_transient_exits(
    exit_code: int,
    expected_status: str,
    retried: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_jobs(tmp_path, monkeypatch)
    monkeypatch.setattr(
        background_jobs.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=exit_code,
            stdout='{"status":"error"}',
            stderr="",
        ),
    )
    job = background_jobs.enqueue_job(
        name="codex-save",
        module="chronovisor.hosts.codex_record",
        args=["--hook", "--save"],
        env={},
        stdin_text='{"session_id":"session-1"}',
    )

    result = background_jobs.run_job(job["job_id"])

    assert result["status"] == expected_status
    assert result["attempts"] == 1
    assert (result["next_retry_at"] is not None) is retried
