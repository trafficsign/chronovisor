"""Tests for Claude Code session saver."""

from __future__ import annotations

import base64
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from chronovisor.hosts import claude_code_record
from chronovisor.raw.raw_semantic_projection import project_parent_raw
from chronovisor.raw.save_transaction import make_save_transaction


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
        max_chars=claude_code_record.DEFAULT_MAX_CHARS,
        dry_run=False,
        save=True,
        extract_only=False,
        ignore_state=ignore_state,
        hook=False,
        trigger_ingest=False,
    )


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def test_extract_preserves_every_event_but_semantic_view_filters_transport(
    tmp_path: Path,
) -> None:
    session = tmp_path / "session.jsonl"
    sample_session(session)

    result = claude_code_record.extract_transcript_slice(session)

    assert result.session_id == "abc-1234-def"
    assert result.cwd == "/tmp/project"
    assert [(r.role, r.line) for r in result.records] == [
        ("event", 1),
        ("event", 2),
        ("event", 3),
        ("user", 4),
        ("assistant", 5),
        ("tool", 6),
        ("user", 7),
        ("assistant", 8),
    ]

    transcript = claude_code_record.format_transcript(result.records)
    assert "system-reminder" not in transcript
    assert "internal reasoning" not in transcript
    assert "tool_use" not in transcript
    assert "Wiki の保存フローを自動化したい" in transcript
    assert "了解、実装します。" in transcript
    serialized = claude_code_record.serialize_transcript_records(result.records)
    source_events = [json.loads(line) for line in session.read_text().splitlines()]
    serialized_rows = json.loads(serialized)
    assert [row["event"] for row in serialized_rows] == source_events
    assert [_canonical_json_bytes(row["event"]) for row in serialized_rows] == [
        _canonical_json_bytes(event) for event in source_events
    ]
    assert "system prompt here" in serialized
    assert "system-reminder" in serialized
    assert "internal reasoning" in serialized
    assert '"type": "tool_use"' in serialized
    assert '"type": "tool_result"' in serialized
    assert "file contents here" in serialized


def test_projection_omits_reasoning_tools_and_system_from_complete_raw(
    tmp_path: Path,
) -> None:
    session = tmp_path / "complete-session.jsonl"
    rows = [
        {
            "type": "system",
            "sessionId": "complete",
            "message": {"content": "privileged system prompt"},
        },
        {
            "type": "assistant",
            "sessionId": "complete",
            "message": {
                "content": [
                    {"type": "thinking", "thinking": "private reasoning"},
                    {"type": "redacted_thinking", "data": "opaque reasoning bytes"},
                    {"type": "text", "text": "visible assistant body"},
                ]
            },
        },
        {
            "type": "user",
            "sessionId": "complete",
            "message": {"content": "<system-reminder>injected bytes</system-reminder>"},
        },
        {
            "type": "user",
            "sessionId": "complete",
            "message": {
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": "tool-1",
                    "content": "private tool output",
                }]
            },
        },
        {
            "type": "user",
            "sessionId": "complete",
            "message": {"content": "visible user body"},
        },
    ]
    write_jsonl(session, rows)

    extracted = claude_code_record.extract_transcript_slice(session)
    serialized = claude_code_record.serialize_transcript_records(extracted.records)
    serialized_events = [row["event"] for row in json.loads(serialized)]
    assert [_canonical_json_bytes(event) for event in serialized_events] == [
        _canonical_json_bytes(event) for event in rows
    ]

    transaction = make_save_transaction(
        host="claude-code",
        session_file=session,
        session_id=extracted.session_id,
        after_line=0,
        until_line=extracted.scanned_until_line,
    )
    raw_path = tmp_path / f"save-{transaction.idempotency_key}.md"
    raw_path.write_text(claude_code_record.build_raw_content(extracted, transaction=transaction))
    projection = project_parent_raw(
        raw_path, output_dir=tmp_path / "projection", max_child_bytes=4_000
    )
    projected = "\n".join(path.read_text() for path in projection.child_paths)

    assert projection.record_count == len(rows)
    assert projection.selected_record_count == 2
    assert "visible assistant body" in projected
    assert "visible user body" in projected
    for transport_only in (
        "privileged system prompt",
        "private reasoning",
        "opaque reasoning bytes",
        "injected bytes",
        "private tool output",
    ):
        assert transport_only in serialized
        assert transport_only not in projected


def test_extract_preserves_image_file_and_tool_payloads(tmp_path: Path) -> None:
    session = tmp_path / "structured.jsonl"
    write_jsonl(
        session,
        [
            {
                "type": "user",
                "sessionId": "structured",
                "message": {
                    "content": [
                        {"type": "image", "source": {"type": "base64", "data": "AA=="}},
                        {"type": "document", "source": {"type": "base64", "data": "AQ=="}},
                    ]
                },
            },
            {
                "type": "assistant",
                "sessionId": "structured",
                "message": {
                    "content": [
                        {"type": "tool_use", "name": "Read", "input": {"file_path": "/tmp/a"}}
                    ]
                },
            },
        ],
    )

    result = claude_code_record.extract_transcript_slice(session)
    serialized = claude_code_record.serialize_transcript_records(result.records)

    assert [(record.role, record.line) for record in result.records] == [
        ("user", 1),
        ("tool", 2),
    ]
    assert '"type": "image"' in serialized
    assert '"type": "document"' in serialized
    assert '"name": "Read"' in serialized


def test_extract_honors_after_line(tmp_path: Path) -> None:
    session = tmp_path / "session.jsonl"
    sample_session(session)

    result = claude_code_record.extract_transcript_slice(session, after_line=5)

    roles_lines = [(r.role, r.line) for r in result.records]
    assert ("user", 4) not in roles_lines
    assert ("user", 7) in roles_lines
    assert ("assistant", 8) in roles_lines


def test_is_injected_context() -> None:
    assert claude_code_record.is_injected_context("<system-reminder>foo</system-reminder>")
    assert claude_code_record.is_injected_context("<command-name>/compact</command-name>")
    assert claude_code_record.is_injected_context("<local-command-caveat>Caveat: ...")
    assert claude_code_record.is_injected_context("<local-command-stdout>output</local-command-stdout>")
    assert claude_code_record.is_injected_context("# AGENTS.md instructions for /tmp")
    assert claude_code_record.is_injected_context("<environment_context>\ndata")
    assert claude_code_record.is_injected_context(
        "Note: /tmp/foo was read before the last conversation was summarized, but ..."
    )
    assert not claude_code_record.is_injected_context("hello world")
    assert not claude_code_record.is_injected_context("Note: this is a normal note")


def test_parse_writer_output_sanitizes_keywords() -> None:
    output = json.dumps(
        {
            "should_save": True,
            "content": "Body",
            "keywords": ["Claude Code", "Claude Code", "bad,keyword", 123, "Chronovisor"],
            "reason": "durable",
            "evidence_quotes": ["Wiki の保存フローを自動化したい"],
        }
    )

    result = claude_code_record.parse_writer_output(output)

    assert result.should_save is True
    assert result.keywords == ["Claude Code", "Chronovisor"]
    assert result.rejected_keywords == ["bad,keyword", "123"]
    assert result.evidence_quotes == ["Wiki の保存フローを自動化したい"]


def test_parse_writer_output_requires_evidence_quotes_for_save() -> None:
    output = json.dumps(
        {
            "should_save": True,
            "content": "Body",
            "keywords": ["Claude Code"],
            "reason": "durable",
            "evidence_quotes": [],
        }
    )

    with pytest.raises(
        claude_code_record.ClaudeCodeSaveError,
        match="must provide user evidence_quotes",
    ):
        claude_code_record.parse_writer_output(output)


def test_run_memory_writer_is_retired_without_frontier_delegation() -> None:
    with pytest.raises(
        claude_code_record.ClaudeCodeSaveError,
        match="deterministic-lossless",
    ):
        claude_code_record.run_memory_writer("prompt")


def test_writer_prompt_forbids_assistant_model_name_substitution(tmp_path: Path) -> None:
    session = tmp_path / "session.jsonl"
    sample_session(session)

    prompt = claude_code_record.build_writer_prompt(
        claude_code_record.extract_transcript_slice(session),
        max_chars=claude_code_record.DEFAULT_MAX_CHARS,
    )

    assert "evidence_quotes" in claude_code_record.MEMORY_WRITER_SCHEMA["required"]
    assert "exact substrings from USER messages only" in prompt
    assert "do not replace Q-KUN with Qwen" in prompt
    assert "appears only in ASSISTANT text, omit it" in prompt


def test_user_evidence_validation_rejects_assistant_only_quote(tmp_path: Path) -> None:
    session = tmp_path / "session.jsonl"
    sample_session(session)
    transcript_slice = claude_code_record.extract_transcript_slice(session)
    valid = claude_code_record.WriterResult(
        should_save=True,
        content="Body",
        keywords=["Claude Code"],
        reason="durable",
        rejected_keywords=[],
        evidence_quotes=["Wiki の保存フローを自動化したい"],
    )
    invalid = claude_code_record.WriterResult(
        should_save=True,
        content="Body",
        keywords=["Claude Code"],
        reason="durable",
        rejected_keywords=[],
        evidence_quotes=["了解、実装します。"],
    )

    claude_code_record.validate_user_evidence_quotes(valid, transcript_slice)
    with pytest.raises(
        claude_code_record.ClaudeCodeSaveError,
        match="not found verbatim in USER",
    ):
        claude_code_record.validate_user_evidence_quotes(invalid, transcript_slice)


def test_user_evidence_validation_rejects_normalized_model_and_capacity(
    tmp_path: Path,
) -> None:
    transcript_slice = claude_code_record.TranscriptSlice(
        session_file=tmp_path / "session.jsonl",
        scanned_until_line=2,
        records=[
            claude_code_record.TranscriptRecord(1, "user", "Q-KUNの32GPレビューを見た"),
            claude_code_record.TranscriptRecord(2, "assistant", "Qwenの32GBレビューですね"),
        ],
    )
    invalid = claude_code_record.WriterResult(
        should_save=True,
        content="ユーザーはQwenの32GBレビューを見た。",
        keywords=["Qwen", "32GB"],
        reason="durable",
        rejected_keywords=[],
        evidence_quotes=["Q-KUNの32GPレビューを見た"],
    )
    valid = claude_code_record.WriterResult(
        should_save=True,
        content="ユーザーはQ-KUNの32GPレビューを見た。",
        keywords=["Q-KUN", "32GP"],
        reason="durable",
        rejected_keywords=[],
        evidence_quotes=["Q-KUNの32GPレビューを見た"],
    )
    invalid_keyword = claude_code_record.WriterResult(
        should_save=True,
        content="ユーザーはQ-KUNの32GPレビューを見た。",
        keywords=["Qwen"],
        reason="durable",
        rejected_keywords=[],
        evidence_quotes=["Q-KUNの32GPレビューを見た"],
    )

    with pytest.raises(
        claude_code_record.ClaudeCodeSaveError,
        match="ungrounded protected literal",
    ):
        claude_code_record.validate_user_evidence_quotes(invalid, transcript_slice)
    with pytest.raises(claude_code_record.ClaudeCodeSaveError, match="keywords"):
        claude_code_record.validate_user_evidence_quotes(invalid_keyword, transcript_slice)
    claude_code_record.validate_user_evidence_quotes(valid, transcript_slice)


def test_trim_middle() -> None:
    short = "hello"
    assert claude_code_record.trim_middle(short, 1000) == short

    long_text = "A" * 5000 + "B" * 5000
    trimmed = claude_code_record.trim_middle(long_text, 2000)
    assert "[... transcript trimmed" in trimmed
    assert trimmed.startswith("A")
    assert trimmed.endswith("B")


def test_state_roundtrip(tmp_path: Path) -> None:
    state_file = tmp_path / "state.json"
    session_file = tmp_path / "session.jsonl"

    state = claude_code_record.load_state(state_file)
    assert state == {"version": 1, "files": {}}

    ts = claude_code_record.TranscriptSlice(
        session_file=session_file,
        scanned_until_line=42,
        records=[],
        session_id="test-id",
        cwd="/tmp",
    )
    claude_code_record.update_state(state, session_file=session_file, transcript_slice=ts, status="saved")
    claude_code_record.write_state(state_file, state)

    loaded = claude_code_record.load_state(state_file)
    assert claude_code_record.saved_line_for(loaded, session_file) == 42


def test_save_mode_updates_state_and_prevents_duplicate(tmp_path: Path, monkeypatch) -> None:
    session = tmp_path / "session.jsonl"
    state = tmp_path / "state.json"
    sample_session(session)
    calls: list[dict] = []

    monkeypatch.setattr(claude_code_record, "init_chronovisor", lambda: None)
    monkeypatch.setattr(
        claude_code_record,
        "run_memory_writer",
        lambda *args, **kwargs: pytest.fail("normal capture must not start a writer"),
    )

    def fake_save_raw(
        content: str,
        *,
        session_id: str,
        keywords: list[str],
        trigger_ingest: bool,
        idempotency_key: str,
    ) -> dict:
        calls.append(
            {
                "content": content,
                "session_id": session_id,
                "keywords": keywords,
                "trigger_ingest": trigger_ingest,
                "idempotency_key": idempotency_key,
            }
        )
        return {"saved": "raw.md"}

    monkeypatch.setattr(claude_code_record, "save_raw", fake_save_raw)
    monkeypatch.setattr(
        claude_code_record, "validate_published_save_receipt", lambda **_kwargs: None
    )

    first_args = args_for(session, state, ignore_state=True)
    first_args.trigger_ingest = True
    first = claude_code_record.run(first_args)
    second = claude_code_record.run(args_for(session, state))

    assert first["status"] == "saved"
    assert second["status"] == "skipped"
    assert second["record_count"] == 0
    assert len(calls) == 1
    assert calls[0]["session_id"] == "claude-code-abc-1234-def"
    assert calls[0]["trigger_ingest"] is False
    assert calls[0]["idempotency_key"].startswith("claude-code-")
    assert '"text": "Wiki の保存フローを自動化したい"' in calls[0]["content"]
    assert "Capture mode: deterministic-lossless" in calls[0]["content"]
    assert first["capture_mode"] == "deterministic-lossless"
    assert first["keywords"] == ["Claude Code", "transcript-delta"]
    saved_state = json.loads(state.read_text())
    assert saved_state["files"][str(session)]["last_saved_line"] == 8


def test_v2_save_preserves_source_lines_without_legacy_markdown(
    tmp_path: Path, monkeypatch
) -> None:
    session = tmp_path / "session.jsonl"
    state = tmp_path / "state.json"
    raw_dir = tmp_path / "raw"
    sample_session(session)
    source_bytes = session.read_bytes()
    monkeypatch.setenv("CHRONOVISOR_RAW_LAYOUT", "v2")
    monkeypatch.setattr(claude_code_record, "RAW_DIR", raw_dir)
    monkeypatch.setattr(claude_code_record, "init_chronovisor", lambda: None)
    monkeypatch.setattr(
        claude_code_record,
        "save_raw",
        lambda *_args, **_kwargs: pytest.fail("v2 must not publish legacy Markdown"),
    )

    result = claude_code_record.run(args_for(session, state, ignore_state=True))

    assert result["status"] == "saved"
    assert result["save_result"]["storage"] == "segment_open"
    segment = Path(result["save_result"]["path"])
    assert segment.read_bytes() == source_bytes
    commit = json.loads(
        Path(result["save_result"]["commit_path"])
        .read_text(encoding="utf-8")
        .splitlines()[-1]
    )
    captured = datetime.fromisoformat(commit["captured_at"])
    assert segment.relative_to(raw_dir).parts[:3] == (
        captured.strftime("%Y"),
        captured.strftime("%m"),
        captured.strftime("%d"),
    )
    assert json.loads(state.read_text())["files"][str(session)]["last_saved_line"] == 8


def test_v2_save_accepts_one_oversized_native_record(
    tmp_path: Path, monkeypatch
) -> None:
    session = tmp_path / "oversized.jsonl"
    state = tmp_path / "state.json"
    raw_dir = tmp_path / "raw"
    write_jsonl(
        session,
        [
            {
                "type": "user",
                "sessionId": "oversized-v2",
                "message": {"content": "X" * 2_000},
            }
        ],
    )
    monkeypatch.setenv("CHRONOVISOR_RAW_LAYOUT", "v2")
    monkeypatch.setattr(claude_code_record, "RAW_DIR", raw_dir)
    monkeypatch.setattr(claude_code_record, "init_chronovisor", lambda: None)
    args = args_for(session, state, ignore_state=True)
    args.max_chars = 128

    result = claude_code_record.run(args)

    assert result["status"] == "saved"
    assert Path(result["save_result"]["path"]).read_bytes() == session.read_bytes()
    assert result["after_line"] == 0
    assert result["scanned_until_line"] == 1
    assert json.loads(state.read_text())["files"][str(session)]["last_saved_line"] == 1


def test_one_stop_drains_every_bounded_transcript_chunk(
    tmp_path: Path, monkeypatch
) -> None:
    session = tmp_path / "session.jsonl"
    state = tmp_path / "state.json"
    sample_session(session)
    calls: list[str] = []
    monkeypatch.setattr(claude_code_record, "init_chronovisor", lambda: None)
    monkeypatch.setattr(
        claude_code_record,
        "save_raw",
        lambda content, **_kwargs: calls.append(content)
        or {"saved": f"raw-{len(calls)}.md"},
    )
    monkeypatch.setattr(
        claude_code_record,
        "validate_published_save_receipt",
        lambda **_kwargs: None,
    )
    extracted = claude_code_record.extract_transcript_slice(session)
    args = args_for(session, state, ignore_state=True)
    args.max_chars = max(
        len(claude_code_record._serialized_records_bytes([record]))
        for record in extracted.records
    )

    result = claude_code_record.run(args)

    assert result["status"] == "saved"
    assert result["chunk_count"] >= 2
    assert len(calls) == result["chunk_count"]
    assert "Wiki の保存フローを自動化したい" in "\n".join(calls)
    assert "file contents here" in "\n".join(calls)
    assert "テストを書きます。" in "\n".join(calls)
    assert json.loads(state.read_text())["files"][str(session)]["last_saved_line"] == 8


def _fragment_payload(content: str) -> dict:
    encoded = content.split("```json\n", 1)[1].split("\n```", 1)[0]
    return json.loads(encoded)


def test_oversized_first_record_is_reassemblable_and_commits_after_all_fragments(
    tmp_path: Path, monkeypatch
) -> None:
    session = tmp_path / "oversized.jsonl"
    state = tmp_path / "state.json"
    write_jsonl(
        session,
        [
            {
                "type": "user",
                "sessionId": "oversized",
                "message": {"content": "記憶" * 300},
            }
        ],
    )
    calls: list[dict] = []
    monkeypatch.setattr(claude_code_record, "init_chronovisor", lambda: None)
    monkeypatch.setattr(
        claude_code_record,
        "save_raw",
        lambda content, **kwargs: calls.append({"content": content, **kwargs})
        or {"saved": f"raw-{len(calls)}.md"},
    )
    monkeypatch.setattr(
        claude_code_record, "validate_published_save_receipt", lambda **_kwargs: None
    )
    args = args_for(session, state, ignore_state=True)
    args.max_chars = 128

    result = claude_code_record.run(args)
    payloads = [_fragment_payload(call["content"]) for call in calls]
    reconstructed = b"".join(base64.b64decode(row["data"]) for row in payloads)
    expected = claude_code_record._serialized_records_bytes(
        [claude_code_record.extract_transcript_slice(session).records[0]]
    )

    assert result["status"] == "saved"
    assert result["oversized_record"] is True
    assert result["fragment_count"] == len(calls) > 1
    assert all(row["fragment_bytes"] <= args.max_chars for row in payloads)
    assert reconstructed == expected
    assert len({call["idempotency_key"] for call in calls}) == len(calls)
    assert json.loads(state.read_text())["files"][str(session)]["last_saved_line"] == 1


def test_oversized_fragment_failure_never_advances_cursor(
    tmp_path: Path, monkeypatch
) -> None:
    session = tmp_path / "oversized.jsonl"
    state = tmp_path / "state.json"
    write_jsonl(
        session,
        [
            {
                "type": "user",
                "sessionId": "oversized",
                "message": {"content": "X" * 1000},
            }
        ],
    )
    monkeypatch.setattr(claude_code_record, "init_chronovisor", lambda: None)
    attempts = 0

    def fail_second_fragment(_content: str, **_kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 2:
            raise RuntimeError("injected fragment failure")
        return {"saved": f"raw-{attempts}.md"}

    monkeypatch.setattr(claude_code_record, "save_raw", fail_second_fragment)
    monkeypatch.setattr(
        claude_code_record, "validate_published_save_receipt", lambda **_kwargs: None
    )
    args = args_for(session, state, ignore_state=True)
    args.max_chars = 128

    with pytest.raises(RuntimeError, match="injected fragment failure"):
        claude_code_record.run(args)

    assert not state.exists()


@pytest.mark.parametrize("mode", ["off", "invalid-mode"])
def test_raw_capture_policy_fail_closed_without_cursor_or_model(
    tmp_path: Path, monkeypatch, mode: str
) -> None:
    session = tmp_path / "session.jsonl"
    state = tmp_path / "state.json"
    sample_session(session)
    state.write_text('{"version": 1, "files": {}}\n')
    original_state = state.read_bytes()
    monkeypatch.setenv("CHRONOVISOR_DECISION_POLICY_RAW_CAPTURE", mode)
    monkeypatch.setattr(
        claude_code_record, "init_chronovisor", lambda: pytest.fail("policy gate must precede init")
    )
    monkeypatch.setattr(
        claude_code_record,
        "save_raw",
        lambda *_args, **_kwargs: pytest.fail("must not publish"),
    )
    monkeypatch.setattr(
        claude_code_record,
        "run_memory_writer",
        lambda *_args, **_kwargs: pytest.fail("must not start a model"),
    )

    result = claude_code_record.run(args_for(session, state))

    assert result["status"] == "deferred"
    assert result["model_calls"] == 0
    assert result["decision_policy"]["mode"] == "off"
    assert state.read_bytes() == original_state


def test_retry_recovers_raw_published_before_state_commit(
    tmp_path: Path,
    monkeypatch,
) -> None:
    session = tmp_path / "session.jsonl"
    state = tmp_path / "state.json"
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    sample_session(session)
    with session.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "type": "assistant",
                    "sessionId": "abc-1234-def",
                    "cwd": "/tmp/project",
                    "timestamp": "2026-05-25T00:00:08Z",
                    "message": {
                        "content": [
                            {
                                "type": "tool_use",
                                "name": "Edit",
                                "input": {"file_path": "/tmp/foo"},
                            }
                        ]
                    },
                }
            )
            + "\n"
        )
    state.write_text(
        json.dumps(
            {
                "version": 1,
                "files": {
                    str(session): {
                        "last_saved_line": 3,
                        "session_id": "abc-1234-def",
                        "status": "declined",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    save_calls: list[str] = []

    monkeypatch.setattr(claude_code_record, "init_chronovisor", lambda: None)
    monkeypatch.setattr(claude_code_record, "RAW_DIR", raw_dir)

    def durable_fake_save(content: str, *, idempotency_key: str, **_kwargs):
        save_calls.append(idempotency_key)
        path = raw_dir / f"save-{idempotency_key}.md"
        path.write_text(content, encoding="utf-8")
        return {"saved": path.name, "path": str(path)}

    real_write_state = claude_code_record.write_state
    state_writes = 0

    def fail_first_state_commit(path: Path, payload: dict) -> None:
        nonlocal state_writes
        state_writes += 1
        if state_writes == 1:
            raise OSError("injected crash after raw publish")
        real_write_state(path, payload)

    monkeypatch.setattr(
        claude_code_record,
        "run_memory_writer",
        lambda *args, **kwargs: pytest.fail("recovery must not start a writer"),
    )
    monkeypatch.setattr(claude_code_record, "save_raw", durable_fake_save)
    monkeypatch.setattr(claude_code_record, "write_state", fail_first_state_commit)

    with pytest.raises(OSError, match="injected crash"):
        claude_code_record.run(args_for(session, state))

    retry = claude_code_record.run(args_for(session, state))

    assert retry["status"] == "recovered"
    assert retry["recovered_save"]["idempotency_key"] == save_calls[0]
    assert len(save_calls) == 1
    assert len(list(raw_dir.glob("*.md"))) == 1
    saved_state = json.loads(state.read_text())
    assert saved_state["files"][str(session)]["last_saved_line"] == 9


def test_corrupt_publisher_receipt_does_not_advance_cursor(
    tmp_path: Path,
    monkeypatch,
) -> None:
    session = tmp_path / "session.jsonl"
    state = tmp_path / "state.json"
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    sample_session(session)
    monkeypatch.setattr(claude_code_record, "init_chronovisor", lambda: None)
    monkeypatch.setattr(claude_code_record, "RAW_DIR", raw_dir)
    monkeypatch.setattr(
        claude_code_record,
        "run_memory_writer",
        lambda *args, **kwargs: pytest.fail("normal capture must not start a writer"),
    )

    def corrupt_save(content: str, *, idempotency_key: str, **_kwargs):
        path = raw_dir / f"save-{idempotency_key}.md"
        path.write_text(
            content.replace("Wiki の保存フローを自動化したい", "tampered memory"),
            encoding="utf-8",
        )
        return {"saved": path.name, "path": str(path)}

    monkeypatch.setattr(claude_code_record, "save_raw", corrupt_save)

    with pytest.raises(
        claude_code_record.ClaudeCodeSaveError,
        match="receipt validation failed",
    ):
        claude_code_record.run(args_for(session, state, ignore_state=True))

    assert not state.exists()


def test_save_captures_user_and_assistant_text_without_writer(
    tmp_path: Path,
    monkeypatch,
) -> None:
    session = tmp_path / "session.jsonl"
    state = tmp_path / "state.json"
    sample_session(session)
    monkeypatch.setattr(claude_code_record, "init_chronovisor", lambda: None)
    monkeypatch.setattr(
        claude_code_record,
        "run_memory_writer",
        lambda *args, **kwargs: pytest.fail("normal capture must not start a writer"),
    )
    saved: list[str] = []
    monkeypatch.setattr(
        claude_code_record,
        "save_raw",
        lambda content, **kwargs: saved.append(content) or {"saved": "raw.md"},
    )
    monkeypatch.setattr(
        claude_code_record, "validate_published_save_receipt", lambda **_kwargs: None
    )

    result = claude_code_record.run(args_for(session, state, ignore_state=True))

    assert result["status"] == "saved"
    assert len(saved) == 1
    assert '"role": "user"' in saved[0]
    assert '"role": "assistant"' in saved[0]
    assert "了解、実装します。" in saved[0]


def test_hook_mode_disabled_without_env(tmp_path: Path, monkeypatch) -> None:
    session = tmp_path / "session.jsonl"
    state = tmp_path / "state.json"
    sample_session(session)
    args = args_for(session, state)
    args.hook = True
    monkeypatch.delenv(claude_code_record.HOOK_ENABLE_ENV, raising=False)
    monkeypatch.setattr(claude_code_record, "init_chronovisor", lambda: None)

    result = claude_code_record.run(args, stdin_text="{}")

    assert result["status"] == "disabled"


def test_hook_hints_from_payload() -> None:
    payload = {
        "session_id": "my-session-id",
        "transcript_path": "/tmp/test.jsonl",
        "cwd": "/tmp/project",
    }
    hints = claude_code_record.hook_hints(payload)
    assert hints["session_id"] == "my-session-id"
    assert hints["transcript_path"] == "/tmp/test.jsonl"
    assert hints["cwd"] == "/tmp/project"


def test_message_content_text_string() -> None:
    assert claude_code_record.message_content_text("hello") == "hello"
    assert claude_code_record.message_content_text("<system-reminder>x</system-reminder>") == ""


def test_message_content_text_list() -> None:
    content = [
        {"type": "thinking", "thinking": "ignore"},
        {"type": "text", "text": "visible"},
        {"type": "tool_use", "name": "Read"},
    ]
    assert claude_code_record.message_content_text(content) == "visible"


def test_message_content_text_list_with_tool_result() -> None:
    content = [
        {"type": "tool_result", "content": "file output", "tool_use_id": "abc"},
    ]
    assert claude_code_record.message_content_text(content) == ""


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


def test_short_tail_is_captured_without_frontier_disposition(tmp_path: Path, monkeypatch) -> None:
    session = tmp_path / "session.jsonl"
    state_file = tmp_path / "state.json"
    session_with_edits(session, user_turns=3, include_edit=False)

    monkeypatch.setattr(claude_code_record, "init_chronovisor", lambda: None)
    monkeypatch.setattr(
        claude_code_record,
        "run_memory_writer",
        lambda *args, **kwargs: pytest.fail("normal capture must not start a writer"),
    )
    monkeypatch.setattr(
        claude_code_record,
        "save_raw",
        lambda *args, **kwargs: {"saved": "raw.md"},
    )
    monkeypatch.setattr(
        claude_code_record, "validate_published_save_receipt", lambda **_kwargs: None
    )
    args = args_for(session, state_file)
    result = claude_code_record.run(args)

    assert result["status"] == "saved"
    assert result["trigger"] == "session_tail"
    assert result["capture_mode"] == "deterministic-lossless"
    assert json.loads(state_file.read_text())["files"][str(session)]["status"] == "saved"


def test_timing_triggers_on_edit(tmp_path: Path, monkeypatch) -> None:
    session = tmp_path / "session.jsonl"
    state_file = tmp_path / "state.json"
    session_with_edits(session, user_turns=2, include_edit=True)

    monkeypatch.setattr(claude_code_record, "init_chronovisor", lambda: None)
    monkeypatch.setattr(
        claude_code_record,
        "run_memory_writer",
        lambda *a, **kw: claude_code_record.WriterResult(
            should_save=True, content="Memory", keywords=["test"], reason="edit", rejected_keywords=[],
            evidence_quotes=["User message 0"],
        ),
    )
    monkeypatch.setattr(
        claude_code_record,
        "save_raw",
        lambda *a, **kw: {"saved": "raw.md"},
    )
    monkeypatch.setattr(
        claude_code_record, "validate_published_save_receipt", lambda **_kwargs: None
    )
    args = args_for(session, state_file)
    result = claude_code_record.run(args)

    assert result["status"] == "saved"
    assert result["trigger"] == "file_changes"


def test_concurrent_stop_workers_publish_delta_exactly_once(
    tmp_path: Path,
    monkeypatch,
) -> None:
    session = tmp_path / "session.jsonl"
    state_file = tmp_path / "state.json"
    session_with_edits(session, user_turns=2, include_edit=True)

    monkeypatch.setattr(claude_code_record, "init_chronovisor", lambda: None)
    first_save_entered = threading.Event()
    release_first_save = threading.Event()
    duplicate_save_entered = threading.Event()
    second_lock_attempted = threading.Event()
    calls_guard = threading.Lock()
    lock_attempts: list[int] = []
    saves: list[str] = []

    from contextlib import contextmanager

    real_transaction_lock = claude_code_record.save_transaction_lock

    @contextmanager
    def instrumented_transaction_lock(**kwargs):
        with calls_guard:
            lock_attempts.append(1)
            if len(lock_attempts) == 2:
                second_lock_attempted.set()
        with real_transaction_lock(**kwargs) as lock_path:
            yield lock_path

    def fake_save(content: str, **_kwargs):
        with calls_guard:
            saves.append(content)
            first = len(saves) == 1
        if first:
            first_save_entered.set()
            assert release_first_save.wait(5)
        else:
            duplicate_save_entered.set()
        return {"saved": "raw.md"}

    monkeypatch.setattr(
        claude_code_record,
        "run_memory_writer",
        lambda *args, **kwargs: pytest.fail("normal capture must not start a writer"),
    )
    monkeypatch.setattr(claude_code_record, "save_raw", fake_save)
    monkeypatch.setattr(
        claude_code_record, "validate_published_save_receipt", lambda **_kwargs: None
    )
    monkeypatch.setattr(
        claude_code_record,
        "save_transaction_lock",
        instrumented_transaction_lock,
    )
    second_started = threading.Event()

    def second_worker():
        second_started.set()
        return claude_code_record.run(args_for(session, state_file))

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(claude_code_record.run, args_for(session, state_file))
        assert first_save_entered.wait(5)
        second = pool.submit(second_worker)
        assert second_started.wait(5)
        assert second_lock_attempted.wait(5)
        assert not duplicate_save_entered.is_set()
        release_first_save.set()
        results = [first.result(timeout=5), second.result(timeout=5)]

    assert not duplicate_save_entered.is_set()
    assert len(lock_attempts) == 2
    assert len(saves) == 1
    assert sorted(result["status"] for result in results) == ["saved", "skipped"]
    saved_state = json.loads(state_file.read_text())
    assert saved_state["files"][str(session)]["last_saved_line"] == 5


def test_timing_triggers_on_turn_interval(tmp_path: Path, monkeypatch) -> None:
    session = tmp_path / "session.jsonl"
    state_file = tmp_path / "state.json"
    session_with_edits(session, user_turns=10, include_edit=False)

    monkeypatch.setattr(claude_code_record, "init_chronovisor", lambda: None)
    monkeypatch.setattr(
        claude_code_record,
        "run_memory_writer",
        lambda *a, **kw: claude_code_record.WriterResult(
            should_save=True, content="Memory", keywords=["test"], reason="turns", rejected_keywords=[],
            evidence_quotes=["User message 0"],
        ),
    )
    monkeypatch.setattr(
        claude_code_record,
        "save_raw",
        lambda *a, **kw: {"saved": "raw.md"},
    )
    monkeypatch.setattr(
        claude_code_record, "validate_published_save_receipt", lambda **_kwargs: None
    )
    args = args_for(session, state_file)
    result = claude_code_record.run(args)

    assert result["status"] == "saved"
    assert result["trigger"] == "session_tail"


def test_recent_save_does_not_strand_new_tail(tmp_path: Path, monkeypatch) -> None:
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

    monkeypatch.setattr(claude_code_record, "init_chronovisor", lambda: None)
    monkeypatch.setattr(
        claude_code_record,
        "run_memory_writer",
        lambda *a, **kw: claude_code_record.WriterResult(
            should_save=True, content="Memory", keywords=["test"], reason="tail",
            rejected_keywords=[], evidence_quotes=["User message 0"],
        ),
    )
    monkeypatch.setattr(claude_code_record, "save_raw", lambda *a, **kw: {"saved": "raw.md"})
    monkeypatch.setattr(claude_code_record, "validate_published_save_receipt", lambda **_kwargs: None)
    args = args_for(session, state_file)
    result = claude_code_record.run(args)

    assert result["status"] == "saved"
    assert result["trigger"] == "file_changes"


def test_timing_bypass_with_ignore_state(tmp_path: Path, monkeypatch) -> None:
    session = tmp_path / "session.jsonl"
    state_file = tmp_path / "state.json"
    session_with_edits(session, user_turns=1, include_edit=False)

    monkeypatch.setattr(claude_code_record, "init_chronovisor", lambda: None)
    monkeypatch.setattr(
        claude_code_record,
        "run_memory_writer",
        lambda *a, **kw: claude_code_record.WriterResult(
            should_save=True, content="Memory", keywords=["test"], reason="bypass", rejected_keywords=[],
            evidence_quotes=["User message 0"],
        ),
    )
    monkeypatch.setattr(
        claude_code_record,
        "save_raw",
        lambda *a, **kw: {"saved": "raw.md"},
    )
    monkeypatch.setattr(
        claude_code_record, "validate_published_save_receipt", lambda **_kwargs: None
    )
    args = args_for(session, state_file)
    args.ignore_state = True
    result = claude_code_record.run(args)

    assert result["status"] == "saved"


def test_content_has_file_changes() -> None:
    assert claude_code_record._content_has_file_changes([
        {"type": "text", "text": "hello"},
        {"type": "tool_use", "name": "Edit", "input": {}},
    ])
    assert claude_code_record._content_has_file_changes([
        {"type": "tool_use", "name": "Write", "input": {}},
    ])
    assert not claude_code_record._content_has_file_changes([
        {"type": "tool_use", "name": "Read", "input": {}},
    ])
    assert not claude_code_record._content_has_file_changes("just a string")


def test_extract_detects_file_changes_and_user_turns(tmp_path: Path) -> None:
    session = tmp_path / "session.jsonl"
    session_with_edits(session, user_turns=5, include_edit=True)
    result = claude_code_record.extract_transcript_slice(session)
    assert result.has_file_changes is True
    assert result.user_turn_count == 5


def test_extract_no_file_changes(tmp_path: Path) -> None:
    session = tmp_path / "session.jsonl"
    session_with_edits(session, user_turns=3, include_edit=False)
    result = claude_code_record.extract_transcript_slice(session)
    assert result.has_file_changes is False
    assert result.user_turn_count == 3
