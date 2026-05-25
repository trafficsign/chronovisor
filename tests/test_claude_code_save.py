"""Tests for Claude Code session saver."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from llm_wiki_mcp import claude_code_save


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def sample_session(path: Path) -> None:
    write_jsonl(
        path,
        [
            {
                "type": "last-prompt",
                "sessionId": "abc-1234-def",
                "cwd": "/tmp/project",
                "timestamp": "2026-05-25T00:00:00Z",
                "message": {"content": ""},
            },
            {
                "type": "system",
                "sessionId": "abc-1234-def",
                "cwd": "/tmp/project",
                "timestamp": "2026-05-25T00:00:01Z",
                "message": {"content": "system prompt here"},
            },
            {
                "type": "user",
                "sessionId": "abc-1234-def",
                "cwd": "/tmp/project",
                "timestamp": "2026-05-25T00:00:02Z",
                "message": {"content": "<system-reminder>hook output</system-reminder>"},
            },
            {
                "type": "user",
                "sessionId": "abc-1234-def",
                "cwd": "/tmp/project",
                "timestamp": "2026-05-25T00:00:03Z",
                "message": {"content": "Wiki の保存フローを自動化したい"},
            },
            {
                "type": "assistant",
                "sessionId": "abc-1234-def",
                "cwd": "/tmp/project",
                "timestamp": "2026-05-25T00:00:04Z",
                "message": {
                    "content": [
                        {"type": "thinking", "thinking": "internal reasoning"},
                        {"type": "text", "text": "了解、実装します。"},
                        {"type": "tool_use", "name": "Read", "input": {"file_path": "/tmp/foo"}},
                    ]
                },
            },
            {
                "type": "user",
                "sessionId": "abc-1234-def",
                "cwd": "/tmp/project",
                "timestamp": "2026-05-25T00:00:05Z",
                "message": {
                    "content": [
                        {"type": "tool_result", "content": "file contents here", "tool_use_id": "x"},
                    ]
                },
            },
            {
                "type": "user",
                "sessionId": "abc-1234-def",
                "cwd": "/tmp/project",
                "timestamp": "2026-05-25T00:00:06Z",
                "message": {"content": "いいね、続けて"},
            },
            {
                "type": "assistant",
                "sessionId": "abc-1234-def",
                "cwd": "/tmp/project",
                "timestamp": "2026-05-25T00:00:07Z",
                "message": {
                    "content": [
                        {"type": "text", "text": "テストを書きます。"},
                    ]
                },
            },
        ],
    )


def args_for(session_file: Path, state_file: Path, *, ignore_state: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        session_id=None,
        session_file=str(session_file),
        state_file=str(state_file),
        max_chars=claude_code_save.DEFAULT_MAX_CHARS,
        dry_run=False,
        save=True,
        extract_only=False,
        ignore_state=ignore_state,
        hook=False,
        trigger_ingest=False,
    )


def test_extract_filters_system_and_injected(tmp_path: Path) -> None:
    session = tmp_path / "session.jsonl"
    sample_session(session)

    result = claude_code_save.extract_transcript_slice(session)

    assert result.session_id == "abc-1234-def"
    assert result.cwd == "/tmp/project"
    roles_lines = [(r.role, r.line) for r in result.records]
    assert ("user", 3) not in roles_lines, "system-reminder should be filtered"
    assert ("user", 4) in roles_lines
    assert ("assistant", 5) in roles_lines
    assert ("user", 6) not in roles_lines, "tool_result should be filtered"
    assert ("user", 7) in roles_lines
    assert ("assistant", 8) in roles_lines

    transcript = claude_code_save.format_transcript(result.records)
    assert "system-reminder" not in transcript
    assert "internal reasoning" not in transcript
    assert "tool_use" not in transcript
    assert "Wiki の保存フローを自動化したい" in transcript
    assert "了解、実装します。" in transcript


def test_extract_honors_after_line(tmp_path: Path) -> None:
    session = tmp_path / "session.jsonl"
    sample_session(session)

    result = claude_code_save.extract_transcript_slice(session, after_line=5)

    roles_lines = [(r.role, r.line) for r in result.records]
    assert ("user", 4) not in roles_lines
    assert ("user", 7) in roles_lines
    assert ("assistant", 8) in roles_lines


def test_is_injected_context() -> None:
    assert claude_code_save.is_injected_context("<system-reminder>foo</system-reminder>")
    assert claude_code_save.is_injected_context("<command-name>/compact</command-name>")
    assert claude_code_save.is_injected_context("<local-command-caveat>Caveat: ...")
    assert claude_code_save.is_injected_context("<local-command-stdout>output</local-command-stdout>")
    assert claude_code_save.is_injected_context("# AGENTS.md instructions for /tmp")
    assert claude_code_save.is_injected_context("<environment_context>\ndata")
    assert claude_code_save.is_injected_context(
        "Note: /tmp/foo was read before the last conversation was summarized, but ..."
    )
    assert not claude_code_save.is_injected_context("hello world")
    assert not claude_code_save.is_injected_context("Note: this is a normal note")


def test_parse_writer_output_sanitizes_keywords() -> None:
    output = json.dumps(
        {
            "should_save": True,
            "content": "Body",
            "keywords": ["Claude Code", "Claude Code", "bad,keyword", 123, "LLM Wiki"],
            "reason": "durable",
        }
    )

    result = claude_code_save.parse_writer_output(output)

    assert result.should_save is True
    assert result.keywords == ["Claude Code", "LLM Wiki"]
    assert result.rejected_keywords == ["bad,keyword", "123"]


def test_trim_middle() -> None:
    short = "hello"
    assert claude_code_save.trim_middle(short, 1000) == short

    long_text = "A" * 5000 + "B" * 5000
    trimmed = claude_code_save.trim_middle(long_text, 2000)
    assert "[... transcript trimmed" in trimmed
    assert trimmed.startswith("A")
    assert trimmed.endswith("B")


def test_state_roundtrip(tmp_path: Path) -> None:
    state_file = tmp_path / "state.json"
    session_file = tmp_path / "session.jsonl"

    state = claude_code_save.load_state(state_file)
    assert state == {"version": 1, "files": {}}

    ts = claude_code_save.TranscriptSlice(
        session_file=session_file,
        scanned_until_line=42,
        records=[],
        session_id="test-id",
        cwd="/tmp",
    )
    claude_code_save.update_state(state, session_file=session_file, transcript_slice=ts, status="saved")
    claude_code_save.write_state(state_file, state)

    loaded = claude_code_save.load_state(state_file)
    assert claude_code_save.saved_line_for(loaded, session_file) == 42


def test_save_mode_updates_state_and_prevents_duplicate(tmp_path: Path, monkeypatch) -> None:
    session = tmp_path / "session.jsonl"
    state = tmp_path / "state.json"
    sample_session(session)
    calls: list[dict] = []

    monkeypatch.setattr(claude_code_save, "init_wiki", lambda: None)
    monkeypatch.setattr(
        claude_code_save,
        "run_memory_writer",
        lambda *args, **kwargs: claude_code_save.WriterResult(
            should_save=True,
            content="Durable memory",
            keywords=["Claude Code", "LLM Wiki"],
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

    monkeypatch.setattr(claude_code_save, "save_raw", fake_save_raw)

    first = claude_code_save.run(args_for(session, state, ignore_state=True))
    second = claude_code_save.run(args_for(session, state))

    assert first["status"] == "saved"
    assert second["status"] == "skipped"
    assert second["record_count"] == 0
    assert len(calls) == 1
    assert calls[0]["session_id"] == "claude-code-abc-1234-def"
    assert calls[0]["trigger_ingest"] is False
    saved_state = json.loads(state.read_text())
    assert saved_state["files"][str(session)]["last_saved_line"] == 8


def test_hook_mode_disabled_without_env(tmp_path: Path, monkeypatch) -> None:
    session = tmp_path / "session.jsonl"
    state = tmp_path / "state.json"
    sample_session(session)
    args = args_for(session, state)
    args.hook = True
    monkeypatch.delenv(claude_code_save.HOOK_ENABLE_ENV, raising=False)
    monkeypatch.setattr(claude_code_save, "init_wiki", lambda: None)

    result = claude_code_save.run(args, stdin_text="{}")

    assert result["status"] == "disabled"


def test_hook_hints_from_payload() -> None:
    payload = {
        "session_id": "my-session-id",
        "transcript_path": "/tmp/test.jsonl",
        "cwd": "/tmp/project",
    }
    hints = claude_code_save.hook_hints(payload)
    assert hints["session_id"] == "my-session-id"
    assert hints["transcript_path"] == "/tmp/test.jsonl"
    assert hints["cwd"] == "/tmp/project"


def test_message_content_text_string() -> None:
    assert claude_code_save.message_content_text("hello") == "hello"
    assert claude_code_save.message_content_text("<system-reminder>x</system-reminder>") == ""


def test_message_content_text_list() -> None:
    content = [
        {"type": "thinking", "thinking": "ignore"},
        {"type": "text", "text": "visible"},
        {"type": "tool_use", "name": "Read"},
    ]
    assert claude_code_save.message_content_text(content) == "visible"


def test_message_content_text_list_with_tool_result() -> None:
    content = [
        {"type": "tool_result", "content": "file output", "tool_use_id": "abc"},
    ]
    assert claude_code_save.message_content_text(content) == ""


# ---------------------------------------------------------------------------
# Timing / trigger logic
# ---------------------------------------------------------------------------


def session_with_edits(path: Path, user_turns: int = 3, include_edit: bool = True) -> None:
    rows = [
        {
            "type": "last-prompt",
            "sessionId": "timing-test",
            "cwd": "/tmp/project",
            "timestamp": "2026-05-25T00:00:00Z",
            "message": {"content": ""},
        },
    ]
    for i in range(user_turns):
        rows.append(
            {
                "type": "user",
                "sessionId": "timing-test",
                "cwd": "/tmp/project",
                "timestamp": f"2026-05-25T00:0{i}:01Z",
                "message": {"content": f"User message {i}"},
            }
        )
        content: list[dict] = [{"type": "text", "text": f"Response {i}"}]
        if include_edit and i == user_turns - 1:
            content.append({"type": "tool_use", "name": "Edit", "input": {"file_path": "/tmp/foo"}})
        rows.append(
            {
                "type": "assistant",
                "sessionId": "timing-test",
                "cwd": "/tmp/project",
                "timestamp": f"2026-05-25T00:0{i}:02Z",
                "message": {"content": content},
            }
        )
    write_jsonl(path, rows)


def test_timing_skip_few_turns_no_edits(tmp_path: Path, monkeypatch) -> None:
    session = tmp_path / "session.jsonl"
    state_file = tmp_path / "state.json"
    session_with_edits(session, user_turns=3, include_edit=False)

    monkeypatch.setattr(claude_code_save, "init_wiki", lambda: None)
    args = args_for(session, state_file)
    result = claude_code_save.run(args)

    assert result["status"] == "skipped"
    assert "waiting" in result["reason"]
    assert not state_file.exists(), "state should not be written when trigger doesn't fire"


def test_timing_triggers_on_edit(tmp_path: Path, monkeypatch) -> None:
    session = tmp_path / "session.jsonl"
    state_file = tmp_path / "state.json"
    session_with_edits(session, user_turns=2, include_edit=True)

    monkeypatch.setattr(claude_code_save, "init_wiki", lambda: None)
    monkeypatch.setattr(
        claude_code_save,
        "run_memory_writer",
        lambda *a, **kw: claude_code_save.WriterResult(
            should_save=True, content="Memory", keywords=["test"], reason="edit", rejected_keywords=[],
        ),
    )
    monkeypatch.setattr(
        claude_code_save,
        "save_raw",
        lambda *a, **kw: {"saved": "raw.md"},
    )
    args = args_for(session, state_file)
    result = claude_code_save.run(args)

    assert result["status"] == "saved"
    assert result["trigger"] == "file_changes"


def test_timing_triggers_on_turn_interval(tmp_path: Path, monkeypatch) -> None:
    session = tmp_path / "session.jsonl"
    state_file = tmp_path / "state.json"
    session_with_edits(session, user_turns=10, include_edit=False)

    monkeypatch.setattr(claude_code_save, "init_wiki", lambda: None)
    monkeypatch.setattr(
        claude_code_save,
        "run_memory_writer",
        lambda *a, **kw: claude_code_save.WriterResult(
            should_save=True, content="Memory", keywords=["test"], reason="turns", rejected_keywords=[],
        ),
    )
    monkeypatch.setattr(
        claude_code_save,
        "save_raw",
        lambda *a, **kw: {"saved": "raw.md"},
    )
    args = args_for(session, state_file)
    result = claude_code_save.run(args)

    assert result["status"] == "saved"
    assert "turn_interval" in result["trigger"]


def test_timing_cooldown_blocks_save(tmp_path: Path, monkeypatch) -> None:
    session = tmp_path / "session.jsonl"
    state_file = tmp_path / "state.json"
    session_with_edits(session, user_turns=2, include_edit=True)

    from datetime import timedelta
    recent = (datetime.now(timezone.utc) - timedelta(seconds=60)).isoformat()
    state = {
        "version": 1,
        "files": {
            str(session): {
                "last_saved_line": 0,
                "session_id": "timing-test",
                "cwd": "/tmp/project",
                "status": "saved",
                "updated_at": recent,
                "last_saved_at": recent,
            }
        },
    }
    state_file.write_text(json.dumps(state))

    monkeypatch.setattr(claude_code_save, "init_wiki", lambda: None)
    args = args_for(session, state_file)
    result = claude_code_save.run(args)

    assert result["status"] == "skipped"
    assert "cooldown" in result["reason"]


def test_timing_bypass_with_ignore_state(tmp_path: Path, monkeypatch) -> None:
    session = tmp_path / "session.jsonl"
    state_file = tmp_path / "state.json"
    session_with_edits(session, user_turns=1, include_edit=False)

    monkeypatch.setattr(claude_code_save, "init_wiki", lambda: None)
    monkeypatch.setattr(
        claude_code_save,
        "run_memory_writer",
        lambda *a, **kw: claude_code_save.WriterResult(
            should_save=True, content="Memory", keywords=["test"], reason="bypass", rejected_keywords=[],
        ),
    )
    monkeypatch.setattr(
        claude_code_save,
        "save_raw",
        lambda *a, **kw: {"saved": "raw.md"},
    )
    args = args_for(session, state_file)
    args.ignore_state = True
    result = claude_code_save.run(args)

    assert result["status"] == "saved"


def test_content_has_file_changes() -> None:
    assert claude_code_save._content_has_file_changes([
        {"type": "text", "text": "hello"},
        {"type": "tool_use", "name": "Edit", "input": {}},
    ])
    assert claude_code_save._content_has_file_changes([
        {"type": "tool_use", "name": "Write", "input": {}},
    ])
    assert not claude_code_save._content_has_file_changes([
        {"type": "tool_use", "name": "Read", "input": {}},
    ])
    assert not claude_code_save._content_has_file_changes("just a string")


def test_extract_detects_file_changes_and_user_turns(tmp_path: Path) -> None:
    session = tmp_path / "session.jsonl"
    session_with_edits(session, user_turns=5, include_edit=True)
    result = claude_code_save.extract_transcript_slice(session)
    assert result.has_file_changes is True
    assert result.user_turn_count == 5


def test_extract_no_file_changes(tmp_path: Path) -> None:
    session = tmp_path / "session.jsonl"
    session_with_edits(session, user_turns=3, include_edit=False)
    result = claude_code_save.extract_transcript_slice(session)
    assert result.has_file_changes is False
    assert result.user_turn_count == 3
