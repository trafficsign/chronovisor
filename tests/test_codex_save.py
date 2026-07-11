"""Tests for Codex-to-LLM-Wiki save harness."""

from __future__ import annotations

import base64
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

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


def args_for(session_file: Path, state_file: Path, *, ignore_state: bool = False) -> SimpleNamespace:
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
        ignore_state=ignore_state,
        hook=False,
        trigger_ingest=False,
    )


def test_extract_transcript_slice_filters_injected_context_and_preserves_tool_output(
    tmp_path: Path,
) -> None:
    session = tmp_path / "session.jsonl"
    sample_session(session)

    result = codex_save.extract_transcript_slice(session)

    assert result.session_id == "019e5ec3-42fe-7f70-9402-7ff20da6be69"
    assert result.cwd == "/tmp/project"
    assert result.scanned_until_line == 5
    assert [(record.role, record.line) for record in result.records] == [
        ("tool", 3),
        ("user", 4),
        ("assistant", 5),
    ]
    assert "AGENTS.md" not in codex_save.format_transcript(result.records)
    assert "large command output" not in codex_save.format_transcript(result.records)
    serialized = codex_save.serialize_transcript_records(result.records)
    assert "large command output" in serialized
    assert result.records[0].event_type == "function_call_output"


def test_extract_preserves_image_file_and_tool_payloads(tmp_path: Path) -> None:
    session = tmp_path / "structured.jsonl"
    write_jsonl(
        session,
        [
            {
                "type": "session_meta",
                "payload": {"id": "structured", "cwd": "/tmp/project"},
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_image", "image_url": "data:image/png;base64,AA=="},
                        {"type": "input_file", "filename": "notes.pdf", "file_data": "AQ=="},
                    ],
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "custom_tool_call",
                    "name": "view_image",
                    "input": {"path": "/tmp/image.png"},
                },
            },
        ],
    )

    result = codex_save.extract_transcript_slice(session)
    serialized = codex_save.serialize_transcript_records(result.records)

    assert [(record.role, record.line) for record in result.records] == [
        ("user", 2),
        ("tool", 3),
    ]
    assert "data:image/png;base64,AA==" in serialized
    assert '"filename": "notes.pdf"' in serialized
    assert '"name": "view_image"' in serialized


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
            "evidence_quotes": ["Codex 側の保存ハーネスを実装したい"],
        }
    )

    result = codex_save.parse_writer_output(output)

    assert result.should_save is True
    assert result.keywords == ["Codex", "LLM Wiki"]
    assert result.rejected_keywords == ["bad,keyword", "123"]
    assert result.evidence_quotes == ["Codex 側の保存ハーネスを実装したい"]


def test_parse_writer_output_requires_evidence_quotes_for_save() -> None:
    output = json.dumps(
        {
            "should_save": True,
            "content": "Body",
            "keywords": ["Codex"],
            "reason": "durable",
            "evidence_quotes": [],
        }
    )

    with pytest.raises(codex_save.CodexSaveError, match="must provide user evidence_quotes"):
        codex_save.parse_writer_output(output)


def test_writer_prompt_forbids_assistant_model_name_substitution(tmp_path: Path) -> None:
    session = tmp_path / "session.jsonl"
    sample_session(session)

    prompt = codex_save.build_writer_prompt(
        codex_save.extract_transcript_slice(session),
        max_chars=codex_save.DEFAULT_MAX_CHARS,
    )

    assert "evidence_quotes" in codex_save.MEMORY_WRITER_SCHEMA["required"]
    assert "exact substrings from USER messages only" in prompt
    assert "do not replace Q-KUN with Qwen" in prompt
    assert "appears only in ASSISTANT text, omit it" in prompt


def test_user_evidence_validation_rejects_assistant_only_quote(tmp_path: Path) -> None:
    session = tmp_path / "session.jsonl"
    sample_session(session)
    transcript_slice = codex_save.extract_transcript_slice(session)
    valid = codex_save.WriterResult(
        should_save=True,
        content="Body",
        keywords=["Codex"],
        reason="durable",
        rejected_keywords=[],
        evidence_quotes=["Codex 側の保存ハーネスを実装したい"],
    )
    invalid = codex_save.WriterResult(
        should_save=True,
        content="Body",
        keywords=["Codex"],
        reason="durable",
        rejected_keywords=[],
        evidence_quotes=["gpt-5.4-mini で要約します"],
    )

    codex_save.validate_user_evidence_quotes(valid, transcript_slice)
    with pytest.raises(codex_save.CodexSaveError, match="not found verbatim in USER"):
        codex_save.validate_user_evidence_quotes(invalid, transcript_slice)


def test_user_evidence_validation_rejects_normalized_model_and_capacity(
    tmp_path: Path,
) -> None:
    transcript_slice = codex_save.TranscriptSlice(
        session_file=tmp_path / "session.jsonl",
        scanned_until_line=2,
        records=[
            codex_save.TranscriptRecord(1, "user", "Q-KUNの32GPレビューを見た"),
            codex_save.TranscriptRecord(2, "assistant", "Qwenの32GBレビューですね"),
        ],
    )
    invalid = codex_save.WriterResult(
        should_save=True,
        content="ユーザーはQwenの32GBレビューを見た。",
        keywords=["Qwen", "32GB"],
        reason="durable",
        rejected_keywords=[],
        evidence_quotes=["Q-KUNの32GPレビューを見た"],
    )
    valid = codex_save.WriterResult(
        should_save=True,
        content="ユーザーはQ-KUNの32GPレビューを見た。",
        keywords=["Q-KUN", "32GP"],
        reason="durable",
        rejected_keywords=[],
        evidence_quotes=["Q-KUNの32GPレビューを見た"],
    )
    invalid_keyword = codex_save.WriterResult(
        should_save=True,
        content="ユーザーはQ-KUNの32GPレビューを見た。",
        keywords=["Qwen"],
        reason="durable",
        rejected_keywords=[],
        evidence_quotes=["Q-KUNの32GPレビューを見た"],
    )

    with pytest.raises(codex_save.CodexSaveError, match="ungrounded protected literal"):
        codex_save.validate_user_evidence_quotes(invalid, transcript_slice)
    with pytest.raises(codex_save.CodexSaveError, match="keywords"):
        codex_save.validate_user_evidence_quotes(invalid_keyword, transcript_slice)
    codex_save.validate_user_evidence_quotes(valid, transcript_slice)


def test_run_memory_writer_is_retired_without_a_subprocess_surface() -> None:
    assert not hasattr(codex_save, "subprocess")

    with pytest.raises(codex_save.CodexSaveError, match="deterministic-lossless"):
        codex_save.run_memory_writer("prompt")


def test_save_mode_updates_state_and_prevents_duplicate_save(tmp_path: Path, monkeypatch) -> None:
    session = tmp_path / "session.jsonl"
    state = tmp_path / "state.json"
    sample_session(session)
    calls: list[dict] = []

    monkeypatch.setattr(codex_save, "init_wiki", lambda: None)
    monkeypatch.setattr(
        codex_save,
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

    monkeypatch.setattr(codex_save, "save_raw", fake_save_raw)
    monkeypatch.setattr(
        codex_save, "validate_published_save_receipt", lambda **_kwargs: None
    )

    first_args = args_for(session, state, ignore_state=True)
    first_args.trigger_ingest = True
    first = codex_save.run(first_args)
    second = codex_save.run(args_for(session, state))

    assert first["status"] == "saved"
    assert second["status"] == "skipped"
    assert second["record_count"] == 0
    assert len(calls) == 1
    assert calls[0]["session_id"] == "codex-019e5ec3-42fe-7f70-9402-7ff20da6be69"
    assert calls[0]["trigger_ingest"] is False
    assert calls[0]["idempotency_key"].startswith("codex-")
    assert '"text": "Codex 側の保存ハーネスを実装したい"' in calls[0]["content"]
    assert "Capture mode: deterministic-lossless" in calls[0]["content"]
    assert first["capture_mode"] == "deterministic-lossless"
    assert first["keywords"] == ["Codex", "transcript-delta"]
    saved_state = json.loads(state.read_text())
    assert saved_state["files"][str(session)]["last_saved_line"] == 5


def test_one_stop_drains_every_bounded_transcript_chunk(
    tmp_path: Path, monkeypatch
) -> None:
    session = tmp_path / "session.jsonl"
    state = tmp_path / "state.json"
    sample_session(session)
    calls: list[str] = []
    monkeypatch.setattr(codex_save, "init_wiki", lambda: None)
    monkeypatch.setattr(
        codex_save,
        "save_raw",
        lambda content, **_kwargs: calls.append(content)
        or {"saved": f"raw-{len(calls)}.md"},
    )
    monkeypatch.setattr(
        codex_save, "validate_published_save_receipt", lambda **_kwargs: None
    )
    extracted = codex_save.extract_transcript_slice(session)
    args = args_for(session, state, ignore_state=True)
    args.max_chars = max(
        len(codex_save._serialized_records_bytes([record]))
        for record in extracted.records
    )

    result = codex_save.run(args)

    assert result["status"] == "saved"
    assert result["chunk_count"] >= 2
    assert len(calls) == result["chunk_count"]
    assert "large command output" in "\n".join(calls)
    assert "Codex 側の保存ハーネスを実装したい" in "\n".join(calls)
    assert "gpt-5.4-mini で要約します" in "\n".join(calls)
    assert json.loads(state.read_text())["files"][str(session)]["last_saved_line"] == 5


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
                "type": "session_meta",
                "payload": {"id": "oversized", "cwd": "/tmp/project"},
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "記憶" * 300}],
                },
            },
        ],
    )
    calls: list[dict] = []
    monkeypatch.setattr(codex_save, "init_wiki", lambda: None)
    monkeypatch.setattr(
        codex_save,
        "save_raw",
        lambda content, **kwargs: calls.append({"content": content, **kwargs})
        or {"saved": f"raw-{len(calls)}.md"},
    )
    monkeypatch.setattr(
        codex_save, "validate_published_save_receipt", lambda **_kwargs: None
    )
    args = args_for(session, state, ignore_state=True)
    args.max_chars = 128

    result = codex_save.run(args)
    payloads = [_fragment_payload(call["content"]) for call in calls]
    reconstructed = b"".join(base64.b64decode(row["data"]) for row in payloads)
    expected = codex_save._serialized_records_bytes(
        [codex_save.extract_transcript_slice(session).records[0]]
    )

    assert result["status"] == "saved"
    assert result["oversized_record"] is True
    assert result["fragment_count"] == len(calls) > 1
    assert all(row["fragment_bytes"] <= args.max_chars for row in payloads)
    assert reconstructed == expected
    assert len({call["idempotency_key"] for call in calls}) == len(calls)
    assert json.loads(state.read_text())["files"][str(session)]["last_saved_line"] == 2


def test_oversized_fragment_failure_never_advances_cursor(
    tmp_path: Path, monkeypatch
) -> None:
    session = tmp_path / "oversized.jsonl"
    state = tmp_path / "state.json"
    write_jsonl(
        session,
        [
            {"type": "session_meta", "payload": {"id": "oversized"}},
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "X" * 1000}],
                },
            },
        ],
    )
    monkeypatch.setattr(codex_save, "init_wiki", lambda: None)
    attempts = 0

    def fail_second_fragment(_content: str, **_kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 2:
            raise RuntimeError("injected fragment failure")
        return {"saved": f"raw-{attempts}.md"}

    monkeypatch.setattr(codex_save, "save_raw", fail_second_fragment)
    monkeypatch.setattr(
        codex_save, "validate_published_save_receipt", lambda **_kwargs: None
    )
    args = args_for(session, state, ignore_state=True)
    args.max_chars = 128

    with pytest.raises(RuntimeError, match="injected fragment failure"):
        codex_save.run(args)

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
    monkeypatch.setenv("LLM_WIKI_DECISION_POLICY_RAW_CAPTURE", mode)
    monkeypatch.setattr(
        codex_save, "init_wiki", lambda: pytest.fail("policy gate must precede init")
    )
    monkeypatch.setattr(
        codex_save, "save_raw", lambda *_args, **_kwargs: pytest.fail("must not publish")
    )
    monkeypatch.setattr(
        codex_save,
        "run_memory_writer",
        lambda *_args, **_kwargs: pytest.fail("must not start a model"),
    )

    result = codex_save.run(args_for(session, state))

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
                    "timestamp": "2026-05-25T00:00:05Z",
                    "type": "response_item",
                    "payload": {"type": "function_call", "name": "apply_patch"},
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
                        "session_id": "019e5ec3-42fe-7f70-9402-7ff20da6be69",
                        "status": "declined",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    save_calls: list[str] = []

    monkeypatch.setattr(codex_save, "init_wiki", lambda: None)
    monkeypatch.setattr(codex_save, "RAW_DIR", raw_dir)

    def durable_fake_save(content: str, *, idempotency_key: str, **_kwargs):
        save_calls.append(idempotency_key)
        path = raw_dir / f"save-{idempotency_key}.md"
        path.write_text(content, encoding="utf-8")
        return {"saved": path.name, "path": str(path)}

    real_write_state = codex_save.write_state
    state_writes = 0

    def fail_first_state_commit(path: Path, payload: dict) -> None:
        nonlocal state_writes
        state_writes += 1
        if state_writes == 1:
            raise OSError("injected crash after raw publish")
        real_write_state(path, payload)

    monkeypatch.setattr(
        codex_save,
        "run_memory_writer",
        lambda *args, **kwargs: pytest.fail("recovery must not start a writer"),
    )
    monkeypatch.setattr(codex_save, "save_raw", durable_fake_save)
    monkeypatch.setattr(codex_save, "write_state", fail_first_state_commit)

    with pytest.raises(OSError, match="injected crash"):
        codex_save.run(args_for(session, state))

    retry = codex_save.run(args_for(session, state))

    assert retry["status"] == "recovered"
    assert retry["recovered_save"]["idempotency_key"] == save_calls[0]
    assert len(save_calls) == 1
    assert len(list(raw_dir.glob("*.md"))) == 1
    saved_state = json.loads(state.read_text())
    assert saved_state["files"][str(session)]["last_saved_line"] == 6


def test_corrupt_publisher_receipt_does_not_advance_cursor(
    tmp_path: Path,
    monkeypatch,
) -> None:
    session = tmp_path / "session.jsonl"
    state = tmp_path / "state.json"
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    sample_session(session)
    monkeypatch.setattr(codex_save, "init_wiki", lambda: None)
    monkeypatch.setattr(codex_save, "RAW_DIR", raw_dir)
    monkeypatch.setattr(
        codex_save,
        "run_memory_writer",
        lambda *args, **kwargs: pytest.fail("normal capture must not start a writer"),
    )

    def corrupt_save(content: str, *, idempotency_key: str, **_kwargs):
        path = raw_dir / f"save-{idempotency_key}.md"
        path.write_text(
            content.replace("Codex 側の保存ハーネスを実装したい", "tampered memory"),
            encoding="utf-8",
        )
        return {"saved": path.name, "path": str(path)}

    monkeypatch.setattr(codex_save, "save_raw", corrupt_save)

    with pytest.raises(codex_save.CodexSaveError, match="receipt validation failed"):
        codex_save.run(args_for(session, state, ignore_state=True))

    assert not state.exists()


def test_concurrent_stop_workers_publish_delta_exactly_once(
    tmp_path: Path,
    monkeypatch,
) -> None:
    session = tmp_path / "session.jsonl"
    state = tmp_path / "state.json"
    sample_session(session)
    with session.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "timestamp": "2026-05-25T00:00:05Z",
                    "type": "response_item",
                    "payload": {"type": "function_call", "name": "apply_patch"},
                }
            )
            + "\n"
        )

    monkeypatch.setattr(codex_save, "init_wiki", lambda: None)
    first_save_entered = threading.Event()
    release_first_save = threading.Event()
    duplicate_save_entered = threading.Event()
    second_lock_attempted = threading.Event()
    calls_guard = threading.Lock()
    lock_attempts: list[int] = []
    saves: list[str] = []

    from contextlib import contextmanager

    real_transaction_lock = codex_save.save_transaction_lock

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
        codex_save,
        "run_memory_writer",
        lambda *args, **kwargs: pytest.fail("normal capture must not start a writer"),
    )
    monkeypatch.setattr(codex_save, "save_raw", fake_save)
    monkeypatch.setattr(
        codex_save, "validate_published_save_receipt", lambda **_kwargs: None
    )
    monkeypatch.setattr(codex_save, "save_transaction_lock", instrumented_transaction_lock)
    second_started = threading.Event()

    def second_worker():
        second_started.set()
        return codex_save.run(args_for(session, state))

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(codex_save.run, args_for(session, state))
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
    saved_state = json.loads(state.read_text())
    assert saved_state["files"][str(session)]["last_saved_line"] == 6


def test_save_captures_user_and_assistant_text_without_writer(
    tmp_path: Path,
    monkeypatch,
) -> None:
    session = tmp_path / "session.jsonl"
    state = tmp_path / "state.json"
    sample_session(session)
    monkeypatch.setattr(codex_save, "init_wiki", lambda: None)
    monkeypatch.setattr(
        codex_save,
        "run_memory_writer",
        lambda *args, **kwargs: pytest.fail("normal capture must not start a writer"),
    )
    saved: list[str] = []
    monkeypatch.setattr(
        codex_save,
        "save_raw",
        lambda content, **kwargs: saved.append(content) or {"saved": "raw.md"},
    )
    monkeypatch.setattr(
        codex_save, "validate_published_save_receipt", lambda **_kwargs: None
    )

    result = codex_save.run(args_for(session, state, ignore_state=True))

    assert result["status"] == "saved"
    assert len(saved) == 1
    assert '"role": "user"' in saved[0]
    assert '"role": "assistant"' in saved[0]
    assert "gpt-5.4-mini で要約します" in saved[0]


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
