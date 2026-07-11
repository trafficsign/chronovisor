from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from llm_wiki_mcp import background_jobs, hook_dispatcher, recall_runtime


def _isolate_background_jobs(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(background_jobs, "JOB_DIR", tmp_path / "jobs")
    monkeypatch.setattr(background_jobs, "STATE_FILE", tmp_path / "jobs" / "state.json")
    monkeypatch.setattr(background_jobs, "LOCK_FILE", tmp_path / "jobs" / "state.lock")
    monkeypatch.setattr(hook_dispatcher, "init_wiki", lambda: None)


def test_user_prompt_dispatches_to_recall_runtime(monkeypatch, capsys) -> None:
    seen: dict[str, object] = {}

    def fake_run_recall(request, policy, *, perform_search: bool):
        seen["request"] = request
        seen["perform_search"] = perform_search
        return recall_runtime.RecallResult(
            status="ok",
            decision="none",
            confidence=0.1,
            queries=[],
            reasons=["test"],
            matched_terms={},
        )

    monkeypatch.setattr(recall_runtime, "run_recall", fake_run_recall)
    monkeypatch.setattr(recall_runtime, "load_policy", lambda _path: recall_runtime.RecallPolicy(log_decisions=False))
    monkeypatch.setattr(
        "sys.stdin",
        io.StringIO(json.dumps({"prompt": "今日暑いな", "cwd": "/repo"}, ensure_ascii=False)),
    )

    assert hook_dispatcher.main(["--host", "codex", "--event", "UserPromptSubmit", "--hook"]) == 0

    output = capsys.readouterr().out.strip()
    assert output == "{}"
    request = seen["request"]
    assert request.host == "codex"
    assert request.prompt == "今日暑いな"
    assert seen["perform_search"] is True


def test_stop_dispatch_only_save_for_legacy_wrapper(monkeypatch, tmp_path, capsys) -> None:
    config = tmp_path / "config.toml"
    config.write_text("[hooks.stop]\nsave = true\naudit = true\n", encoding="utf-8")
    monkeypatch.setenv("CODEX_WIKI_SAVE_ENABLED", "1")
    monkeypatch.setenv("LLM_WIKI_RECALL_AUDIT_ENABLED", "1")
    monkeypatch.setattr("sys.stdin", io.StringIO("{}"))

    assert hook_dispatcher.main(
        [
            "--host",
            "codex",
            "--event",
            "Stop",
            "--hook",
            "--config",
            str(config),
            "--only",
            "save",
            "--dry-run",
            "--format",
            "json",
        ]
    ) == 0
    output = json.loads(capsys.readouterr().out)

    assert [task["name"] for task in output["tasks"]] == ["codex-save"]
    save = output["tasks"][0]
    assert save["module"] == "llm_wiki_mcp.codex_save"
    assert save["args"] == ["--hook", "--save"]
    assert "--trigger-ingest" not in save["args"]


def test_stop_dispatch_full_entrypoint_enqueues_only_capture_work(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    config = tmp_path / "config.toml"
    config.write_text(
        "[hooks.stop]\n"
        "save = true\n"
        "audit = true\n"
        "content_correction = true\n"
        "recall_improve = true\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("CLAUDE_CODE_WIKI_SAVE_ENABLED", "1")
    monkeypatch.setenv("LLM_WIKI_RECALL_AUDIT_ENABLED", "1")
    monkeypatch.setenv("LLM_WIKI_CONTENT_CORRECTION_ENABLED", "1")
    monkeypatch.setenv("LLM_WIKI_RECALL_IMPROVE_ENABLED", "1")
    monkeypatch.setattr("sys.stdin", io.StringIO("{}"))

    assert hook_dispatcher.main(
        [
            "--host",
            "claude-code",
            "--event",
            "Stop",
            "--hook",
            "--config",
            str(config),
            "--dry-run",
            "--format",
            "json",
        ]
    ) == 0
    output = json.loads(capsys.readouterr().out)

    assert [task["name"] for task in output["tasks"]] == [
        "claude-code-save",
        "content-correction-capture",
    ]
    assert output["tasks"][0]["args"] == ["--hook", "--save"]
    correction = output["tasks"][1]
    assert correction["module"] == "llm_wiki_mcp.content_correction"
    assert correction["args"] == [
        "--host",
        "claude-code",
        "--hook",
        "--capture-only",
    ]


def test_stop_dispatch_only_improve_is_disabled(monkeypatch, tmp_path, capsys) -> None:
    config = tmp_path / "config.toml"
    config.write_text("[hooks.stop]\nsave = true\naudit = true\nrecall_improve = true\n", encoding="utf-8")
    monkeypatch.setenv("LLM_WIKI_RECALL_IMPROVE_ENABLED", "1")
    monkeypatch.setattr("sys.stdin", io.StringIO("{}"))

    assert hook_dispatcher.main(
        [
            "--host",
            "codex",
            "--event",
            "Stop",
            "--hook",
            "--config",
            str(config),
            "--only",
            "improve",
            "--dry-run",
            "--format",
            "json",
        ]
    ) == 0
    output = json.loads(capsys.readouterr().out)

    assert output["tasks"] == []


def test_stop_dispatch_only_content_correction_uses_capture_only_worker(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    config = tmp_path / "config.toml"
    config.write_text("[hooks.stop]\ncontent_correction = true\n", encoding="utf-8")
    monkeypatch.setenv("LLM_WIKI_CONTENT_CORRECTION_ENABLED", "1")
    monkeypatch.setattr("sys.stdin", io.StringIO("{}"))

    assert hook_dispatcher.main(
        [
            "--host",
            "codex",
            "--event",
            "Stop",
            "--hook",
            "--config",
            str(config),
            "--only",
            "correction",
            "--dry-run",
            "--format",
            "json",
        ]
    ) == 0
    output = json.loads(capsys.readouterr().out)

    assert output["tasks"] == [
        {
            "name": "content-correction-capture",
            "module": "llm_wiki_mcp.content_correction",
            "args": ["--host", "codex", "--hook", "--capture-only"],
            "dry_run": True,
        }
    ]


def test_stop_content_correction_false_enqueues_nothing(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    config = tmp_path / "config.toml"
    config.write_text(
        "[hooks.stop]\nsave = false\ncontent_correction = false\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("LLM_WIKI_CONTENT_CORRECTION_ENABLED", "1")
    monkeypatch.setattr("sys.stdin", io.StringIO('{"session_id":"session-1"}'))

    assert hook_dispatcher.main(
        [
            "--host",
            "codex",
            "--event",
            "Stop",
            "--hook",
            "--config",
            str(config),
            "--format",
            "json",
        ]
    ) == 0

    assert json.loads(capsys.readouterr().out)["tasks"] == []


def test_stop_capture_enqueue_coalesces_and_never_starts_a_subprocess(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    _isolate_background_jobs(monkeypatch, tmp_path)
    config = tmp_path / "config.toml"
    config.write_text(
        "[hooks.stop]\nsave = false\ncontent_correction = true\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("LLM_WIKI_CONTENT_CORRECTION_ENABLED", "1")
    monkeypatch.setattr(
        background_jobs.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("Stop must not run a subprocess"),
    )
    monkeypatch.setattr(
        background_jobs.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("Stop must not spawn a subprocess"),
    )
    session_id = "11111111-2222-3333-4444-555555555555"
    session_file = tmp_path / "session.jsonl"
    argv = [
        "--host",
        "codex",
        "--event",
        "Stop",
        "--hook",
        "--config",
        str(config),
        "--format",
        "json",
    ]

    monkeypatch.setattr(
        "sys.stdin",
        io.StringIO(
            json.dumps(
                {"session_id": session_id, "session_file": str(session_file), "turn": 1}
            )
        ),
    )
    assert hook_dispatcher.main(argv) == 0
    first = json.loads(capsys.readouterr().out)["tasks"][0]

    monkeypatch.setattr(
        "sys.stdin",
        io.StringIO(
            json.dumps(
                {"session_id": session_id, "session_file": str(session_file), "turn": 2}
            )
        ),
    )
    assert hook_dispatcher.main(argv) == 0
    second = json.loads(capsys.readouterr().out)["tasks"][0]

    assert first["name"] == "content-correction-capture"
    assert first["enqueued"] is True
    assert second["job_id"] == first["job_id"]
    assert second["enqueued"] is False
    assert second["coalesced"] is True
    state = json.loads(background_jobs.STATE_FILE.read_text(encoding="utf-8"))
    assert len(state["jobs"]) == 1
    stored = state["jobs"][first["job_id"]]
    assert stored["lane_key"] == "content-correction-capture"
    assert stored["module"] == "llm_wiki_mcp.content_correction"
    assert stored["args"] == ["--host", "codex", "--hook", "--capture-only"]
    assert json.loads(stored["stdin"])["turn"] == 2


def test_spawn_task_only_enqueues_without_process(monkeypatch) -> None:
    seen: dict[str, object] = {}

    def fake_enqueue_job(**kwargs):
        seen.update(kwargs)
        return {
            "job_id": "job-1",
            "status": "queued",
            "enqueued": True,
            "coalesced": False,
        }

    monkeypatch.setattr(background_jobs, "enqueue_job", fake_enqueue_job)
    task = hook_dispatcher.BackgroundTask(
        name="codex-save",
        module="llm_wiki_mcp.codex_save",
        args=["--hook", "--save"],
        env={"CODEX_WIKI_SAVE_ENABLED": "1"},
    )

    result = hook_dispatcher.spawn_task(task, '{"session_id":"session-1"}')

    assert result == {
        "job_id": "job-1",
        "status": "queued",
        "enqueued": True,
        "coalesced": False,
    }
    assert seen["stdin_text"] == '{"session_id":"session-1"}'


def test_stop_dispatch_requires_env_without_unified_config(monkeypatch, tmp_path, capsys) -> None:
    legacy = tmp_path / "recall.toml"
    legacy.write_text("enabled = true\n", encoding="utf-8")
    monkeypatch.delenv("CODEX_WIKI_SAVE_ENABLED", raising=False)
    monkeypatch.delenv("LLM_WIKI_RECALL_AUDIT_ENABLED", raising=False)
    monkeypatch.delenv("LLM_WIKI_RECALL_IMPROVE_ENABLED", raising=False)
    monkeypatch.delenv("LLM_WIKI_CONTENT_CORRECTION_ENABLED", raising=False)
    monkeypatch.setattr("sys.stdin", io.StringIO("{}"))

    assert hook_dispatcher.main(
        [
            "--host",
            "codex",
            "--event",
            "Stop",
            "--hook",
            "--config",
            str(legacy),
            "--dry-run",
            "--format",
            "json",
        ]
    ) == 0
    output = json.loads(capsys.readouterr().out)

    assert output["tasks"] == []


def test_internal_frontier_stop_never_spawns_tasks(monkeypatch, capsys) -> None:
    monkeypatch.setenv("LLM_WIKI_INTERNAL_FRONTIER", "1")
    monkeypatch.setattr("sys.stdin", io.StringIO("{}"))

    assert hook_dispatcher.main([
        "--host", "codex", "--event", "Stop", "--hook", "--dry-run", "--format", "json",
    ]) == 0

    output = json.loads(capsys.readouterr().out)
    assert output == {"status": "suppressed", "reason": "internal_frontier", "tasks": []}
