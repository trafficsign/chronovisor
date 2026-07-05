from __future__ import annotations

import io
import json
from pathlib import Path

from llm_wiki_mcp import hook_dispatcher, recall_runtime


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


def test_stop_dispatch_full_entrypoint_runs_save_and_audit(monkeypatch, tmp_path, capsys) -> None:
    config = tmp_path / "config.toml"
    config.write_text("[hooks.stop]\nsave = true\naudit = true\nrecall_improve = true\n", encoding="utf-8")
    monkeypatch.setenv("CLAUDE_CODE_WIKI_SAVE_ENABLED", "1")
    monkeypatch.setenv("LLM_WIKI_RECALL_AUDIT_ENABLED", "1")
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
        "claude-code-recall-audit",
        "claude-code-recall-improve",
    ]
    improve = output["tasks"][-1]
    assert improve["module"] == "llm_wiki_mcp.recall_improvement"
    assert improve["args"][:2] == ["run-due", "--config"]


def test_stop_dispatch_only_improve_for_scheduler(monkeypatch, tmp_path, capsys) -> None:
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

    assert [task["name"] for task in output["tasks"]] == ["codex-recall-improve"]


def test_stop_dispatch_requires_env_without_unified_config(monkeypatch, tmp_path, capsys) -> None:
    legacy = tmp_path / "recall.toml"
    legacy.write_text("enabled = true\n", encoding="utf-8")
    monkeypatch.delenv("CODEX_WIKI_SAVE_ENABLED", raising=False)
    monkeypatch.delenv("LLM_WIKI_RECALL_AUDIT_ENABLED", raising=False)
    monkeypatch.delenv("LLM_WIKI_RECALL_IMPROVE_ENABLED", raising=False)
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
