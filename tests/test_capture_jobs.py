from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from chronovisor.core import background_jobs as core_background_jobs
from chronovisor.hosts import codex_record
from chronovisor.ops import background_jobs, session_sweeper
from chronovisor.raw.agent_save_base import should_process


def test_ops_background_jobs_is_core_module_alias() -> None:
    assert background_jobs is core_background_jobs


def test_codex_detects_unified_exec_apply_patch(tmp_path: Path) -> None:
    session = tmp_path / "session.jsonl"
    rows = [
        {
            "type": "session_meta",
            "payload": {"id": "019f0000-0000-7000-8000-000000000000"},
        },
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "fix"}],
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "custom_tool_call",
                "name": "exec",
                "input": "text(await tools.apply_patch('*** Begin Patch'))",
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "done"}],
            },
        },
    ]
    session.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    result = codex_record.extract_transcript_slice(session)

    assert result.has_file_changes is True
    assert should_process(result, {}) == (True, "file_changes")


def test_session_sweeper_excludes_internal_codex_sessions(tmp_path: Path) -> None:
    user = tmp_path / "user.jsonl"
    subagent = tmp_path / "subagent.jsonl"
    automated = tmp_path / "automated.jsonl"
    user.write_text(
        json.dumps(
            {
                "type": "session_meta",
                "payload": {
                    "originator": "Codex Desktop",
                    "thread_source": "user",
                    "source": "vscode",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    subagent.write_text(
        json.dumps(
            {
                "type": "session_meta",
                "payload": {
                    "originator": "Codex Desktop",
                    "thread_source": "subagent",
                    "source": {"subagent": {}},
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    automated.write_text(
        json.dumps(
            {
                "type": "session_meta",
                "payload": {"originator": "codex_exec", "source": "exec"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert session_sweeper._is_user_codex_session(user) is True
    assert session_sweeper._is_user_codex_session(subagent) is False
    assert session_sweeper._is_user_codex_session(automated) is False


def test_session_sweeper_backoff_does_not_pin_queue(
    tmp_path: Path, monkeypatch
) -> None:
    failed = tmp_path / "failed.jsonl"
    later = tmp_path / "later.jsonl"
    monkeypatch.setattr(session_sweeper, "STATE_FILE", tmp_path / "state.json")
    monkeypatch.setattr(session_sweeper, "STATUS_FILE", tmp_path / "latest.json")
    monkeypatch.setattr(
        session_sweeper,
        "discover_pending",
        lambda **_kwargs: [("codex", failed), ("codex", later)],
    )
    seen: list[Path] = []

    def run_one(_host, path):
        seen.append(path)
        if path == failed:
            raise RuntimeError("temporary")
        return {"status": "saved"}

    monkeypatch.setattr(session_sweeper, "_run_one", run_one)
    first = session_sweeper.run_sweeper(limit=1)
    second = session_sweeper.run_sweeper(limit=1)

    assert first["results"][0]["status"] == "error"
    assert second["results"][0]["session_file"] == str(later)
    assert seen == [failed, later]


def test_background_job_failure_is_durable_and_retryable(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(background_jobs, "JOB_DIR", tmp_path)
    monkeypatch.setattr(background_jobs, "STATE_FILE", tmp_path / "state.json")
    monkeypatch.setattr(background_jobs, "LOCK_FILE", tmp_path / "state.lock")
    commands: list[list[str]] = []

    def fail(command: list[str], **_kwargs) -> SimpleNamespace:
        commands.append(command)
        return SimpleNamespace(returncode=1, stdout="", stderr="temporary failure")

    monkeypatch.setattr(background_jobs.subprocess, "run", fail)
    job = background_jobs.enqueue_job(
        name="save",
        module="chronovisor.hosts.codex_record",
        args=[],
        env={},
        stdin_text="{}",
    )

    result = background_jobs.run_job(job["job_id"])
    stored = json.loads((tmp_path / "state.json").read_text())["jobs"][job["job_id"]]

    assert result["status"] == "retry_wait"
    assert commands == [
        [background_jobs.sys.executable, "-m", "chronovisor.hosts.codex_record"]
    ]
    assert stored["exit_code"] == 1
    assert "temporary failure" in stored["output_tail"]


def test_background_job_child_env_keeps_only_runtime_and_fixed_job_flags(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(background_jobs, "JOB_DIR", tmp_path)
    monkeypatch.setattr(background_jobs, "STATE_FILE", tmp_path / "state.json")
    monkeypatch.setattr(background_jobs, "LOCK_FILE", tmp_path / "state.lock")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("LANG", "C")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-parent-secret-canary")
    monkeypatch.setenv("ARBITRARY_PARENT_CANARY", "plain-parent-canary")
    seen: dict[str, str] = {}

    def succeed(_command: list[str], **kwargs) -> SimpleNamespace:
        seen.update(kwargs["env"])
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(background_jobs.subprocess, "run", succeed)
    job = background_jobs.enqueue_job(
        name="codex-save",
        module="chronovisor.hosts.codex_record",
        args=[],
        env={
            "CODEX_CHRONOVISOR_RECORD_ENABLED": "1",
            "ARBITRARY_JOB_CANARY": "plain-job-canary",
            "OPENAI_API_KEY": "sk-job-secret-canary",
        },
        stdin_text="{}",
    )

    result = background_jobs.run_job(job["job_id"])

    assert result["status"] == "completed"
    assert seen["HOME"] == str(tmp_path / "home")
    assert seen["LANG"] == "C"
    assert seen["CODEX_CHRONOVISOR_RECORD_ENABLED"] == "1"
    assert (
        not {
            "ARBITRARY_JOB_CANARY",
            "ARBITRARY_PARENT_CANARY",
            "OPENAI_API_KEY",
        }
        & seen.keys()
    )
    assert job["env"] == {"CODEX_CHRONOVISOR_RECORD_ENABLED": "1"}


def test_background_job_subprocess_error_does_not_leak_detail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    canary = "sk-subprocess-error-canary"
    monkeypatch.setattr(background_jobs, "JOB_DIR", tmp_path)
    monkeypatch.setattr(background_jobs, "STATE_FILE", tmp_path / "state.json")
    monkeypatch.setattr(background_jobs, "LOCK_FILE", tmp_path / "state.lock")

    def fail(*_args, **_kwargs):
        raise OSError(canary)

    monkeypatch.setattr(background_jobs.subprocess, "run", fail)
    job = background_jobs.enqueue_job(
        name="save",
        module="chronovisor.hosts.codex_record",
        args=[],
        env={},
        stdin_text="{}",
    )

    result = background_jobs.run_job(job["job_id"])
    output = capsys.readouterr().out
    stored = (tmp_path / "state.json").read_text(encoding="utf-8")

    assert result["output_tail"] == "OSError"
    assert canary not in output
    assert canary not in stored
    assert canary not in repr(result)


def test_background_job_explicit_quarantine_exit_is_terminal(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(background_jobs, "JOB_DIR", tmp_path)
    monkeypatch.setattr(background_jobs, "STATE_FILE", tmp_path / "state.json")
    monkeypatch.setattr(background_jobs, "LOCK_FILE", tmp_path / "state.lock")
    monkeypatch.setattr(
        background_jobs.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=background_jobs.QUARANTINE_EXIT_CODE,
            stdout='{"status":"human_required"}',
            stderr="",
        ),
    )
    job = background_jobs.enqueue_job(
        name="self-heal",
        module="chronovisor.hosts.codex_record",
        args=[],
        env={},
        stdin_text="",
    )

    result = background_jobs.run_job(job["job_id"])

    assert result["status"] == "quarantined"
    assert result["next_retry_at"] is None


def test_capture_jobs_coalesce_by_session_and_keep_latest_payload(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(background_jobs, "JOB_DIR", tmp_path)
    monkeypatch.setattr(background_jobs, "STATE_FILE", tmp_path / "state.json")
    monkeypatch.setattr(background_jobs, "LOCK_FILE", tmp_path / "state.lock")
    first_payload = json.dumps({"session_id": "session-1", "turn": 1})
    latest_payload = json.dumps({"session_id": "session-1", "turn": 2})

    first = background_jobs.enqueue_job(
        name="codex-save",
        module="chronovisor.hosts.codex_record",
        args=["--hook", "--save"],
        env={},
        stdin_text=first_payload,
    )
    second = background_jobs.enqueue_job(
        name="codex-save",
        module="chronovisor.hosts.codex_record",
        args=["--hook", "--save"],
        env={},
        stdin_text=latest_payload,
    )

    stored = json.loads((tmp_path / "state.json").read_text())
    assert second["job_id"] == first["job_id"]
    assert second["coalesced"] is True
    assert second["enqueued"] is False
    assert len(stored["jobs"]) == 1
    assert stored["jobs"][first["job_id"]]["stdin"] == latest_payload


def test_capture_jobs_without_session_identity_do_not_cross_coalesce(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(background_jobs, "JOB_DIR", tmp_path)
    monkeypatch.setattr(background_jobs, "STATE_FILE", tmp_path / "state.json")
    monkeypatch.setattr(background_jobs, "LOCK_FILE", tmp_path / "state.lock")

    first = background_jobs.enqueue_job(
        name="codex-save",
        module="chronovisor.hosts.codex_record",
        args=["--hook", "--save"],
        env={},
        stdin_text="{}",
    )
    second = background_jobs.enqueue_job(
        name="codex-save",
        module="chronovisor.hosts.codex_record",
        args=["--hook", "--save"],
        env={},
        stdin_text="{}",
    )

    assert first["job_id"] != second["job_id"]
    assert first["coalesced"] is False
    assert second["coalesced"] is False


def test_running_capture_coalesce_requests_one_tail_rerun(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(background_jobs, "JOB_DIR", tmp_path)
    monkeypatch.setattr(background_jobs, "STATE_FILE", tmp_path / "state.json")
    monkeypatch.setattr(background_jobs, "LOCK_FILE", tmp_path / "state.lock")
    first = background_jobs.enqueue_job(
        name="codex-save",
        module="chronovisor.hosts.codex_record",
        args=["--hook", "--save"],
        env={},
        stdin_text=json.dumps({"session_id": "session-1", "turn": 1}),
    )
    assert background_jobs._claim(first["job_id"])["status"] == "running"
    latest_payload = json.dumps({"session_id": "session-1", "turn": 2})

    coalesced = background_jobs.enqueue_job(
        name="codex-save",
        module="chronovisor.hosts.codex_record",
        args=["--hook", "--save"],
        env={},
        stdin_text=latest_payload,
    )
    rerun = background_jobs._finish(first["job_id"], exit_code=0, output="ok")

    assert coalesced["rerun_requested"] is True
    assert rerun["status"] == "queued"
    assert rerun["stdin"] == latest_payload
    claimed_again = background_jobs._claim(first["job_id"])
    assert claimed_again is not None
    completed = background_jobs._finish(first["job_id"], exit_code=0, output="ok")
    assert completed["status"] == "completed"
    assert completed["stdin"] == ""


def test_background_job_lane_is_single_flight(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(background_jobs, "JOB_DIR", tmp_path)
    monkeypatch.setattr(background_jobs, "STATE_FILE", tmp_path / "state.json")
    monkeypatch.setattr(background_jobs, "LOCK_FILE", tmp_path / "state.lock")
    first = background_jobs.enqueue_job(
        name="codex-save",
        module="chronovisor.hosts.codex_record",
        args=["--hook", "--save"],
        env={},
        stdin_text=json.dumps({"session_id": "session-1"}),
    )
    second = background_jobs.enqueue_job(
        name="codex-save",
        module="chronovisor.hosts.codex_record",
        args=["--hook", "--save"],
        env={},
        stdin_text=json.dumps({"session_id": "session-2"}),
    )

    assert background_jobs._claim(first["job_id"])["owner_pid"] == os.getpid()
    assert background_jobs._claim(second["job_id"]) is None
    background_jobs._finish(first["job_id"], exit_code=0, output="ok")
    assert background_jobs._claim(second["job_id"])["owner_pid"] == os.getpid()


def test_background_job_terminal_history_is_pruned(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(background_jobs, "JOB_DIR", tmp_path)
    monkeypatch.setattr(background_jobs, "STATE_FILE", tmp_path / "state.json")
    monkeypatch.setattr(background_jobs, "LOCK_FILE", tmp_path / "state.lock")
    monkeypatch.setattr(background_jobs, "MAX_TERMINAL_JOBS", 2)

    for index in range(4):
        job = background_jobs.enqueue_job(
            name="demo",
            module="chronovisor.hosts.codex_record",
            args=[str(index)],
            env={},
            stdin_text=str(index),
        )
        background_jobs._claim(job["job_id"])
        background_jobs._finish(job["job_id"], exit_code=0, output="ok")

    stored = json.loads((tmp_path / "state.json").read_text())
    assert len(stored["jobs"]) == 2
    assert stored["pruned_terminal_total"] == 2


def test_background_job_child_env_keeps_record_path_settings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(background_jobs, "JOB_DIR", tmp_path)
    monkeypatch.setattr(background_jobs, "STATE_FILE", tmp_path / "state.json")
    monkeypatch.setattr(background_jobs, "LOCK_FILE", tmp_path / "state.lock")
    codex_home = str(tmp_path / "codex-home")
    monkeypatch.setenv("CODEX_HOME", codex_home)
    monkeypatch.setenv("CHRONOVISOR_RAW_LAYOUT", "v2")
    monkeypatch.setenv("PWD", "/tmp/record-project")
    seen: dict[str, str] = {}

    def succeed(_command: list[str], **kwargs) -> SimpleNamespace:
        seen.update(kwargs["env"])
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(background_jobs.subprocess, "run", succeed)
    job = background_jobs.enqueue_job(
        name="codex-save",
        module="chronovisor.hosts.codex_record",
        args=["--hook", "--save"],
        env={},
        stdin_text="{}",
    )

    background_jobs.run_job(job["job_id"])

    assert seen["CODEX_HOME"] == codex_home
    assert seen["CHRONOVISOR_RAW_LAYOUT"] == "v2"
    assert seen["PWD"] == "/tmp/record-project"
