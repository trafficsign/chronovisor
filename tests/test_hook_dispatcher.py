from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from llm_wiki_mcp import (
    background_jobs,
    hook_dispatcher,
    recall_breaker,
    recall_runtime,
)


@pytest.fixture(autouse=True)
def isolate_recall_breaker(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(recall_breaker, "BREAKER_FILE", tmp_path / "breaker.json")


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
    monkeypatch.setattr(
        recall_runtime,
        "load_policy",
        lambda _path: recall_runtime.RecallPolicy(log_decisions=False),
    )
    monkeypatch.setattr(
        "sys.stdin",
        io.StringIO(
            json.dumps({"prompt": "今日暑いな", "cwd": "/repo"}, ensure_ascii=False)
        ),
    )

    assert (
        hook_dispatcher.main(
            ["--host", "codex", "--event", "UserPromptSubmit", "--hook"]
        )
        == 0
    )

    output = capsys.readouterr().out.strip()
    assert output == "{}"
    request = seen["request"]
    assert request.host == "codex"
    assert request.prompt == "今日暑いな"
    assert seen["perform_search"] is True


def test_user_prompt_unexpected_failure_is_exit_zero_fail_open(
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(hook_dispatcher, "init_wiki", lambda: None)
    monkeypatch.setattr(
        recall_runtime,
        "load_policy",
        lambda _path: recall_runtime.RecallPolicy(log_decisions=False),
    )
    monkeypatch.setattr(
        recall_runtime,
        "run_recall",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("offline")),
    )
    monkeypatch.setattr(
        "sys.stdin", io.StringIO('{"prompt":"remember","thread_id":"t1"}')
    )

    assert (
        hook_dispatcher.main(
            ["--host", "codex", "--event", "UserPromptSubmit", "--hook"]
        )
        == 0
    )
    assert capsys.readouterr().out.strip() == "{}"
    assert recall_breaker.snapshot()["failures"] == 1


def test_user_prompt_hard_timeout_injects_model_free_fallback(
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(hook_dispatcher, "init_wiki", lambda: None)
    monkeypatch.setattr(
        recall_runtime,
        "load_policy",
        lambda _path: recall_runtime.RecallPolicy(log_decisions=False),
    )
    monkeypatch.setattr(
        recall_runtime,
        "run_recall",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            hook_dispatcher.RecallWallClockTimeout("primary timeout")
        ),
    )
    monkeypatch.setattr(
        recall_runtime,
        "run_deterministic_fallback",
        lambda *_args, **_kwargs: recall_runtime.RecallResult(
            status="degraded",
            decision="read",
            confidence=0.7,
            queries=["fallback"],
            reasons=["deterministic fallback"],
            matched_terms={},
            context="[WORKING_MEMORY]\ncore\n[/WORKING_MEMORY]",
        ),
    )
    monkeypatch.setattr("sys.stdin", io.StringIO('{"prompt":"前回の続き"}'))

    assert (
        hook_dispatcher.main(
            ["--host", "codex", "--event", "UserPromptSubmit", "--hook"]
        )
        == 0
    )
    output = json.loads(capsys.readouterr().out)
    assert "core" in output["hookSpecificOutput"]["additionalContext"]
    assert recall_breaker.snapshot()["failures"] == 1


def test_user_prompt_open_breaker_uses_bm25_only_policy(monkeypatch, capsys) -> None:
    seen: dict[str, object] = {}

    def fake_run(_request, policy, *, perform_search: bool):
        seen.update(
            semantic=policy.semantic,
            judge_mode=policy.judge_mode,
            rewrite_enabled=policy.rewrite_enabled,
            perform_search=perform_search,
        )
        return recall_runtime.RecallResult(
            status="ok",
            decision="none",
            confidence=0.0,
            queries=[],
            reasons=[],
            matched_terms={},
        )

    monkeypatch.setattr(hook_dispatcher, "init_wiki", lambda: None)
    monkeypatch.setattr(recall_breaker, "is_open", lambda: True)
    monkeypatch.setattr(recall_runtime, "run_recall", fake_run)
    monkeypatch.setattr(
        recall_runtime,
        "load_policy",
        lambda _path: recall_runtime.RecallPolicy(log_decisions=False),
    )
    monkeypatch.setattr("sys.stdin", io.StringIO('{"prompt":"remember"}'))

    assert (
        hook_dispatcher.main(
            ["--host", "codex", "--event", "UserPromptSubmit", "--hook"]
        )
        == 0
    )
    assert capsys.readouterr().out.strip() == "{}"
    assert seen == {
        "semantic": False,
        "judge_mode": "off",
        "rewrite_enabled": False,
        "perform_search": True,
    }


def test_user_prompt_policy_failure_is_still_exit_zero_fail_open(
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(hook_dispatcher, "init_wiki", lambda: None)
    monkeypatch.setattr(
        recall_runtime,
        "load_policy",
        lambda _path: (_ for _ in ()).throw(PermissionError("config unavailable")),
    )
    monkeypatch.setattr("sys.stdin", io.StringIO('{"prompt":"remember"}'))

    assert (
        hook_dispatcher.main(
            ["--host", "codex", "--event", "UserPromptSubmit", "--hook"]
        )
        == 0
    )
    assert capsys.readouterr().out.strip() == "{}"


def test_user_prompt_init_failure_is_still_exit_zero_fail_open(
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(
        hook_dispatcher,
        "init_wiki",
        lambda: (_ for _ in ()).throw(PermissionError("wiki unavailable")),
    )
    monkeypatch.setattr("sys.stdin", io.StringIO('{"prompt":"remember"}'))

    assert (
        hook_dispatcher.main(
            ["--host", "codex", "--event", "UserPromptSubmit", "--hook"]
        )
        == 0
    )
    assert capsys.readouterr().out.strip() == "{}"


def test_stop_dispatch_only_save_for_legacy_wrapper(
    monkeypatch, tmp_path, capsys
) -> None:
    config = tmp_path / "config.toml"
    config.write_text("[hooks.stop]\nsave = true\naudit = true\n", encoding="utf-8")
    monkeypatch.setenv("CODEX_WIKI_SAVE_ENABLED", "1")
    monkeypatch.setenv("LLM_WIKI_RECALL_AUDIT_ENABLED", "1")
    monkeypatch.setattr("sys.stdin", io.StringIO("{}"))

    assert (
        hook_dispatcher.main(
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
        )
        == 0
    )
    output = json.loads(capsys.readouterr().out)

    assert [task["name"] for task in output["tasks"]] == ["codex-save"]
    save = output["tasks"][0]
    assert save["module"] == "llm_wiki_mcp.codex_save"
    assert save["args"] == ["--hook", "--save"]
    assert "--trigger-ingest" not in save["args"]
    assert save["on_success"] == [
        {
            "name": "recall-audit-candidate",
            "module": "llm_wiki_mcp.recall_auditor",
            "args": ["--host", "codex", "--hook"],
            "env": {},
            "when_output_status": "saved",
        }
    ]


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

    assert (
        hook_dispatcher.main(
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
        )
        == 0
    )
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
    config.write_text(
        "[hooks.stop]\nsave = true\naudit = true\nrecall_improve = true\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("LLM_WIKI_RECALL_IMPROVE_ENABLED", "1")
    monkeypatch.setattr("sys.stdin", io.StringIO("{}"))

    assert (
        hook_dispatcher.main(
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
        )
        == 0
    )
    output = json.loads(capsys.readouterr().out)

    assert output["tasks"] == []
    assert output["status"] == "compatibility_noop"
    assert output["deprecated"] is True


def test_stop_dispatch_only_content_correction_uses_capture_only_worker(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    config = tmp_path / "config.toml"
    config.write_text("[hooks.stop]\ncontent_correction = true\n", encoding="utf-8")
    monkeypatch.setenv("LLM_WIKI_CONTENT_CORRECTION_ENABLED", "1")
    monkeypatch.setattr("sys.stdin", io.StringIO("{}"))

    assert (
        hook_dispatcher.main(
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
        )
        == 0
    )
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

    assert (
        hook_dispatcher.main(
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
        )
        == 0
    )

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


def test_stop_dispatch_requires_env_without_unified_config(
    monkeypatch, tmp_path, capsys
) -> None:
    legacy = tmp_path / "recall.toml"
    legacy.write_text("enabled = true\n", encoding="utf-8")
    monkeypatch.delenv("CODEX_WIKI_SAVE_ENABLED", raising=False)
    monkeypatch.delenv("LLM_WIKI_RECALL_AUDIT_ENABLED", raising=False)
    monkeypatch.delenv("LLM_WIKI_RECALL_IMPROVE_ENABLED", raising=False)
    monkeypatch.delenv("LLM_WIKI_CONTENT_CORRECTION_ENABLED", raising=False)
    monkeypatch.setattr("sys.stdin", io.StringIO("{}"))

    assert (
        hook_dispatcher.main(
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
        )
        == 0
    )
    output = json.loads(capsys.readouterr().out)

    assert output["tasks"] == []


def test_internal_frontier_stop_never_spawns_tasks(monkeypatch, capsys) -> None:
    monkeypatch.setenv("LLM_WIKI_INTERNAL_FRONTIER", "1")
    monkeypatch.setattr("sys.stdin", io.StringIO("{}"))

    assert (
        hook_dispatcher.main(
            [
                "--host",
                "codex",
                "--event",
                "Stop",
                "--hook",
                "--dry-run",
                "--format",
                "json",
            ]
        )
        == 0
    )

    output = json.loads(capsys.readouterr().out)
    assert output == {
        "status": "suppressed",
        "reason": "internal_frontier",
        "tasks": [],
    }
