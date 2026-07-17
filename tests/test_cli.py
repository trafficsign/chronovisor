from __future__ import annotations

import json
from pathlib import Path

from llm_wiki_mcp import cli, runtime_config, runtime_status, wiki
from llm_wiki_mcp import recall_runtime


def patch_wiki(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "wiki"
    raw = root / "raw"
    pages = root / "pages"
    system = root / "system"
    recall = root / "recall"
    runtime = root / "runtime"
    for path in (raw, pages, system, recall, runtime):
        path.mkdir(parents=True, exist_ok=True)
    (raw / "r.md").write_text("raw", encoding="utf-8")
    (pages / "p.md").write_text("---\ntitle: P\n---\n", encoding="utf-8")
    (system / "user-profile.md").write_text("# User\n", encoding="utf-8")
    (recall / "recall-log.jsonl").write_text(
        json.dumps({"decision": "read"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (recall / "feedback.jsonl").write_text(
        json.dumps({"kind": "missed_candidate"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    config = root / "config.toml"
    config.write_text("[hooks.stop]\nsave = true\naudit = true\n", encoding="utf-8")

    monkeypatch.setattr(wiki, "WIKI_ROOT", root)
    monkeypatch.setattr(wiki, "RAW_DIR", raw)
    monkeypatch.setattr(wiki, "PAGES_DIR", pages)
    monkeypatch.setattr(wiki, "SYSTEM_DIR", system)
    monkeypatch.setattr(runtime_config, "CONFIG_FILE", config)
    monkeypatch.setattr(runtime_config, "LEGACY_RECALL_CONFIG_FILE", root / "recall.toml")
    monkeypatch.setattr(recall_runtime, "RECALL_DIR", recall)
    monkeypatch.setattr(recall_runtime, "RECALL_LOG_FILE", recall / "recall-log.jsonl")
    monkeypatch.setattr(recall_runtime, "RECALL_FEEDBACK_FILE", recall / "feedback.jsonl")
    monkeypatch.setattr(cli, "RECALL_DIR", recall)
    monkeypatch.setattr(cli, "RECALL_LOG_FILE", recall / "recall-log.jsonl")
    monkeypatch.setattr(cli, "RECALL_FEEDBACK_FILE", recall / "feedback.jsonl")
    monkeypatch.setattr(runtime_status, "RUNTIME_DIR", runtime)
    monkeypatch.setattr(runtime_status, "STATUS_FILE", runtime / "status.json")
    monkeypatch.setattr(runtime_status, "EVENTS_FILE", runtime / "events.jsonl")
    monkeypatch.setattr(runtime_status, "METRICS_FILE", runtime / "metrics.jsonl")


def test_status_json_reports_wiki_and_recall_counts(tmp_path, monkeypatch, capsys) -> None:
    patch_wiki(tmp_path, monkeypatch)

    assert cli.main(["status", "--json"]) == 0
    output = json.loads(capsys.readouterr().out)

    assert output["wiki"]["raw_files"] == 1
    assert output["wiki"]["pages"] == 1
    assert output["config"]["mode"] == "unified"
    assert output["recall"]["decisions"] == {"read": 1}
    assert output["recall"]["feedback"] == {"missed_candidate": 1}


def test_hooks_inspect_json_handles_missing_host_files(tmp_path, monkeypatch, capsys) -> None:
    patch_wiki(tmp_path, monkeypatch)
    monkeypatch.setattr(cli, "CODEX_HOOKS_FILE", tmp_path / "missing-hooks.json")
    monkeypatch.setattr(cli, "CLAUDE_SETTINGS_FILE", tmp_path / "missing-settings.json")

    assert cli.main(["hooks", "inspect", "--json"]) == 0
    output = json.loads(capsys.readouterr().out)

    assert output["codex"]["entries"] == []
    assert output["claude_code"]["entries"] == []
    assert output["hook_policy"]["stop_audit"] is True
    assert output["warnings"] == []


def test_hooks_inspect_marks_legacy_audit_wrapper_as_noop(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    patch_wiki(tmp_path, monkeypatch)
    hooks = tmp_path / "hooks.json"
    hooks.write_text(
        json.dumps(
            {
                "hooks": {
                    "Stop": [
                        {
                            "hooks": [
                                {
                                    "type": "command",
                                    "command": "scripts/codex_recall_audit_hook.sh",
                                    "timeout": 5000,
                                }
                            ]
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(cli, "CODEX_HOOKS_FILE", hooks)
    monkeypatch.setattr(cli, "CLAUDE_SETTINGS_FILE", tmp_path / "missing.json")

    assert cli.main(["hooks", "inspect", "--json"]) == 0
    output = json.loads(capsys.readouterr().out)

    entry = output["codex"]["entries"][0]
    assert entry["compatibility"] == "deprecated_noop"
    assert entry["deprecated"] is True
    assert entry["removal_after"] == "2026-10-01"
    assert output["warnings"][0]["replacement"].startswith("llm-wiki-hook")


def test_recall_improve_run_due_cli_forwards_scheduler_args(tmp_path, monkeypatch, capsys) -> None:
    patch_wiki(tmp_path, monkeypatch)
    from llm_wiki_mcp import recall_improvement

    seen: dict[str, object] = {}

    def fake_run_due(**kwargs):
        seen.update(kwargs)
        return {"status": "due", "dry_run": kwargs["dry_run"]}

    monkeypatch.setattr(recall_improvement, "run_due", fake_run_due)

    assert cli.main(["recall-improve", "run-due", "--dry-run", "--json"]) == 0
    output = json.loads(capsys.readouterr().out)

    assert output == {"status": "due", "dry_run": True}
    assert seen["dry_run"] is True
    assert str(seen["log_file"]).endswith("recall-log.jsonl")


def test_sleep_cli_non_json_handles_partial_cycle(monkeypatch, capsys) -> None:
    from llm_wiki_mcp import sleep_cycle

    monkeypatch.setattr(
        sleep_cycle,
        "run_sleep_cycle",
        lambda **_kwargs: {
            "status": "partial",
            "lane_errors": ["cofire"],
            "cofire": {"status": "error", "error": "boom"},
            "autonomy": {"status": "ok"},
        },
    )

    assert cli.main(["sleep"]) == 0
    output = capsys.readouterr().out
    assert "sleep_cycle\tpartial" in output
    assert "cofire_edges\tunavailable" in output
    assert "lane_errors\tcofire" in output


def test_sleep_cli_non_json_handles_locked_cycle(monkeypatch, capsys) -> None:
    from llm_wiki_mcp import sleep_cycle

    monkeypatch.setattr(
        sleep_cycle,
        "run_sleep_cycle",
        lambda **_kwargs: {
            "status": "skipped",
            "locked": True,
            "reason": "sleep cycle already in progress",
        },
    )

    assert cli.main(["sleep"]) == 0
    output = capsys.readouterr().out
    assert "sleep_cycle\tskipped" in output
    assert "reason\tsleep cycle already in progress" in output


def test_install_codex_hooks_replaces_legacy_entries_and_trust(tmp_path, monkeypatch) -> None:
    patch_wiki(tmp_path, monkeypatch)
    hooks_file = tmp_path / "codex/hooks.json"
    config_file = tmp_path / "codex/config.toml"
    hooks_file.parent.mkdir(parents=True)
    hooks_file.write_text(
        json.dumps(
            {
                "hooks": {
                    "UserPromptSubmit": [
                        {
                            "hooks": [
                                {"type": "command", "command": "cmux notify", "timeout": 1000},
                                {
                                    "type": "command",
                                    "command": "bash /repo/scripts/codex_recall_hook.sh",
                                    "timeout": 5000,
                                },
                            ]
                        }
                    ],
                    "Stop": [
                        {
                            "hooks": [
                                {"type": "command", "command": "cmux stop", "timeout": 1000},
                                {
                                    "type": "command",
                                    "command": "bash /repo/scripts/codex_wiki_save_hook.sh",
                                    "timeout": 5000,
                                },
                                {
                                    "type": "command",
                                    "command": "bash /repo/scripts/codex_recall_audit_hook.sh",
                                    "timeout": 5000,
                                },
                            ]
                        }
                    ],
                }
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    config_file.write_text(
        "\n".join(
            [
                "[hooks.state]",
                "",
                '[hooks.state."/Users/trafficsign/.config/codex/hooks.json:user_prompt_submit:0:1"]',
                "enabled = true",
                'trusted_hash = "old-user"',
                "",
                '[hooks.state."/Users/trafficsign/.config/codex/hooks.json:stop:0:1"]',
                "enabled = true",
                'trusted_hash = "old-stop-save"',
                "",
                '[hooks.state."/Users/trafficsign/.config/codex/hooks.json:stop:0:2"]',
                "enabled = true",
                'trusted_hash = "old-stop-audit"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(cli, "CODEX_HOOKS_FILE", hooks_file)
    monkeypatch.setattr(cli, "CODEX_CONFIG_FILE", config_file)

    result = cli.install_codex_hooks("llm-wiki-hook")

    installed = json.loads(hooks_file.read_text(encoding="utf-8"))
    user_hooks = installed["hooks"]["UserPromptSubmit"][0]["hooks"]
    stop_hooks = installed["hooks"]["Stop"][0]["hooks"]
    assert [hook["command"] for hook in user_hooks] == [
        "cmux notify",
        (
            "CODEX_HOME=/Users/trafficsign/.config/codex "
            "llm-wiki-hook --host codex --event UserPromptSubmit --hook"
        ),
    ]
    assert user_hooks[-1]["timeout"] == 7000
    assert [hook["command"] for hook in stop_hooks] == [
        "cmux stop",
        (
            "CODEX_HOME=/Users/trafficsign/.config/codex "
            "llm-wiki-hook --host codex --event Stop --hook"
        ),
    ]
    assert stop_hooks[-1]["timeout"] == 5000
    assert set(result["trusted_hashes"]) == {"user_prompt_submit:0:1", "stop:0:1"}
    config_text = config_file.read_text(encoding="utf-8")
    assert "old-user" not in config_text
    assert "old-stop-save" not in config_text
    assert "old-stop-audit" not in config_text
    assert "stop:0:2" not in config_text
    assert result["trusted_hashes"]["user_prompt_submit:0:1"] in config_text
    assert result["trusted_hashes"]["stop:0:1"] in config_text


def test_install_claude_code_hooks_preserves_non_wiki_entries(tmp_path, monkeypatch) -> None:
    patch_wiki(tmp_path, monkeypatch)
    settings_file = tmp_path / "claude/settings.json"
    settings_file.parent.mkdir(parents=True)
    settings_file.write_text(
        json.dumps(
            {
                "hooks": {
                    "UserPromptSubmit": [
                        {
                            "hooks": [
                                {
                                    "type": "command",
                                    "command": "bash /repo/scripts/claude_code_recall_hook.sh",
                                    "timeout": 5000,
                                },
                                {"type": "command", "command": "agent-router", "timeout": 1000},
                            ]
                        }
                    ],
                    "Stop": [
                        {
                            "hooks": [
                                {"type": "command", "command": "afplay done.aiff"},
                                {"type": "command", "command": "lazy-detect", "timeout": 1000},
                                {
                                    "type": "command",
                                    "command": "bash /repo/scripts/claude_code_wiki_save_hook.sh",
                                    "timeout": 5000,
                                },
                                {
                                    "type": "command",
                                    "command": "bash /repo/scripts/claude_code_recall_audit_hook.sh",
                                    "timeout": 5000,
                                },
                            ]
                        }
                    ],
                }
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(cli, "CLAUDE_SETTINGS_FILE", settings_file)

    cli.install_claude_code_hooks("llm-wiki-hook")

    installed = json.loads(settings_file.read_text(encoding="utf-8"))
    assert [hook["command"] for hook in installed["hooks"]["UserPromptSubmit"][0]["hooks"]] == [
        "llm-wiki-hook --host claude-code --event UserPromptSubmit --hook",
        "agent-router",
    ]
    assert installed["hooks"]["UserPromptSubmit"][0]["hooks"][0]["timeout"] == 7000
    assert [hook["command"] for hook in installed["hooks"]["Stop"][0]["hooks"]] == [
        "afplay done.aiff",
        "lazy-detect",
        "llm-wiki-hook --host claude-code --event Stop --hook",
    ]
    assert installed["hooks"]["Stop"][0]["hooks"][-1]["timeout"] == 5000


def test_hooks_install_cli_dry_run_json(tmp_path, monkeypatch, capsys) -> None:
    patch_wiki(tmp_path, monkeypatch)
    monkeypatch.setattr(cli, "CODEX_HOOKS_FILE", tmp_path / "codex/hooks.json")
    monkeypatch.setattr(cli, "CODEX_CONFIG_FILE", tmp_path / "codex/config.toml")
    monkeypatch.setattr(cli, "CLAUDE_SETTINGS_FILE", tmp_path / "claude/settings.json")

    assert cli.main([
        "hooks",
        "install",
        "--host",
        "all",
        "--command-prefix",
        "llm-wiki-hook",
        "--dry-run",
        "--json",
    ]) == 0
    output = json.loads(capsys.readouterr().out)

    assert output["host"] == "all"
    assert [result["host"] for result in output["results"]] == ["codex", "claude-code"]
    assert all(result["dry_run"] for result in output["results"])
    assert not (tmp_path / "codex/hooks.json").exists()
    assert not (tmp_path / "claude/settings.json").exists()


def test_default_hook_prefix_uses_pushed_github_runtime(monkeypatch) -> None:
    monkeypatch.delenv("LLM_WIKI_RUNTIME_SOURCE", raising=False)

    prefix = cli.default_hook_command_prefix()

    assert prefix.startswith("uvx --from ")
    assert "git+ssh://git@github.com/trafficsign/llm-wiki-mcp" in prefix
    assert "uv run --project" not in prefix
