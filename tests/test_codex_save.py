"""Tests for Codex-to-LLM-Wiki save harness."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from llm_wiki_mcp import codex_save


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def sample_session(path: Path) -> None:
    write_jsonl(
        path,
        [
            {
                "timestamp": "2026-05-25T00:00:00Z",
                "type": "session_meta",
                "payload": {
                    "id": "019e5ec3-42fe-7f70-9402-7ff20da6be69",
                    "cwd": "/tmp/project",
                    "originator": "Codex Desktop",
                    "cli_version": "0.133.0-alpha.1",
                    "source": "vscode",
                    "model_provider": "openai",
                },
            },
            {
                "timestamp": "2026-05-25T00:00:01Z",
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "# AGENTS.md instructions for /tmp/project\nignore me",
                        },
                        {"type": "input_text", "text": "<environment_context>\nignore me"},
                    ],
                },
            },
            {
                "timestamp": "2026-05-25T00:00:02Z",
                "type": "response_item",
                "payload": {"type": "function_call_output", "output": "large command output"},
            },
            {
                "timestamp": "2026-05-25T00:00:03Z",
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Codex 側の保存ハーネスを実装したい"}],
                },
            },
            {
                "timestamp": "2026-05-25T00:00:04Z",
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "phase": "final",
                    "content": [{"type": "output_text", "text": "gpt-5.4-mini で要約します"}],
                },
            },
        ],
    )


def args_for(session_file: Path, state_file: Path) -> SimpleNamespace:
    return SimpleNamespace(
        session_id=None,
        cwd=None,
        session_file=str(session_file),
        sessions_root=None,
        state_file=str(state_file),
        model=codex_save.DEFAULT_MEMORY_MODEL,
        max_chars=codex_save.DEFAULT_MAX_CHARS,
        timeout=1,
        dry_run=False,
        save=True,
        extract_only=False,
        ignore_state=False,
        hook=False,
        trigger_ingest=False,
    )


def test_extract_transcript_slice_filters_injected_context_and_tool_output(tmp_path: Path) -> None:
    session = tmp_path / "session.jsonl"
    sample_session(session)

    result = codex_save.extract_transcript_slice(session)

    assert result.session_id == "019e5ec3-42fe-7f70-9402-7ff20da6be69"
    assert result.cwd == "/tmp/project"
    assert result.scanned_until_line == 5
    assert [(record.role, record.line) for record in result.records] == [
        ("user", 4),
        ("assistant", 5),
    ]
    assert "AGENTS.md" not in codex_save.format_transcript(result.records)
    assert "large command output" not in codex_save.format_transcript(result.records)


def test_extract_transcript_slice_honors_after_line(tmp_path: Path) -> None:
    session = tmp_path / "session.jsonl"
    sample_session(session)

    result = codex_save.extract_transcript_slice(session, after_line=4)

    assert [(record.role, record.line) for record in result.records] == [("assistant", 5)]


def test_parse_writer_output_sanitizes_keywords() -> None:
    output = json.dumps(
        {
            "should_save": True,
            "content": "Body",
            "keywords": ["Codex", "Codex", "bad,keyword", 123, "LLM Wiki"],
            "reason": "durable",
        }
    )

    result = codex_save.parse_writer_output(output)

    assert result.should_save is True
    assert result.keywords == ["Codex", "LLM Wiki"]
    assert result.rejected_keywords == ["bad,keyword", "123"]


def test_run_memory_writer_uses_mini_schema_and_disables_hooks(monkeypatch) -> None:
    seen: dict[str, object] = {}

    def fake_run(cmd, **kwargs):
        output_path = Path(cmd[cmd.index("-o") + 1])
        output_path.write_text(
            json.dumps(
                {
                    "should_save": True,
                    "content": "Body",
                    "keywords": ["Codex", "LLM Wiki"],
                    "reason": "ok",
                }
            )
        )
        seen["cmd"] = cmd
        seen["env"] = kwargs["env"]
        seen["input"] = kwargs["input"]
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(codex_save.subprocess, "run", fake_run)

    result = codex_save.run_memory_writer("prompt")

    cmd = seen["cmd"]
    assert cmd[cmd.index("-m") + 1] == codex_save.DEFAULT_MEMORY_MODEL
    assert "--output-schema" in cmd
    assert "--disable" in cmd
    assert "hooks" in cmd
    assert "--ephemeral" in cmd
    assert seen["env"]["CODEX_HOME"].endswith("/.config/codex")
    assert seen["env"][codex_save.HOOK_ENABLE_ENV] == "0"
    assert seen["input"] == "prompt"
    assert result.content == "Body"


def test_save_mode_updates_state_and_prevents_duplicate_save(tmp_path: Path, monkeypatch) -> None:
    session = tmp_path / "session.jsonl"
    state = tmp_path / "state.json"
    sample_session(session)
    calls: list[dict] = []

    monkeypatch.setattr(codex_save, "init_wiki", lambda: None)
    monkeypatch.setattr(
        codex_save,
        "run_memory_writer",
        lambda *args, **kwargs: codex_save.WriterResult(
            should_save=True,
            content="Durable memory",
            keywords=["Codex", "LLM Wiki"],
            reason="useful",
            rejected_keywords=[],
        ),
    )

    def fake_save_raw(
        content: str,
        *,
        session_id: str,
        keywords: list[str],
        trigger_ingest: bool,
    ) -> dict:
        calls.append(
            {
                "content": content,
                "session_id": session_id,
                "keywords": keywords,
                "trigger_ingest": trigger_ingest,
            }
        )
        return {"saved": "raw.md"}

    monkeypatch.setattr(codex_save, "save_raw", fake_save_raw)

    first = codex_save.run(args_for(session, state))
    second = codex_save.run(args_for(session, state))

    assert first["status"] == "saved"
    assert second["status"] == "skipped"
    assert second["record_count"] == 0
    assert len(calls) == 1
    assert calls[0]["session_id"] == "codex-019e5ec3-42fe-7f70-9402-7ff20da6be69"
    assert calls[0]["trigger_ingest"] is False
    saved_state = json.loads(state.read_text())
    assert saved_state["files"][str(session)]["last_saved_line"] == 5


def test_hook_mode_is_disabled_without_env(tmp_path: Path, monkeypatch) -> None:
    session = tmp_path / "session.jsonl"
    state = tmp_path / "state.json"
    sample_session(session)
    args = args_for(session, state)
    args.hook = True
    monkeypatch.delenv(codex_save.HOOK_ENABLE_ENV, raising=False)
    monkeypatch.setattr(codex_save, "init_wiki", lambda: None)

    result = codex_save.run(args, stdin_text="{}")

    assert result["status"] == "disabled"
