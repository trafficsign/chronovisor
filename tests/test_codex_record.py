"""Tests for Codex-to-LLM-Wiki save harness."""

from __future__ import annotations

import base64
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from chronovisor.core import codex_transcript
from chronovisor.core.save_transaction import make_save_transaction
from chronovisor.core.store import RuntimeContext, init_chronovisor
from chronovisor.core.transcript import CodexSaveError
from chronovisor.hosts import codex_record
from chronovisor.ingest.raw_semantic_projection import project_parent_raw
from chronovisor.raw import codex_capture_delta, record_transaction


@pytest.fixture(autouse=True)
def _valid_okf_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "wiki"
    root.mkdir()
    for name in ("index.md", "log.md", "schema.md"):
        (root / name).write_text("legacy\n", encoding="utf-8")
    monkeypatch.setattr(codex_record, "CHRONOVISOR_ROOT", root)


def test_transcript_api_is_reexported_from_raw_modules() -> None:
    assert codex_record.TranscriptRecord is codex_transcript.TranscriptRecord
    assert codex_record.TranscriptSlice is codex_transcript.TranscriptSlice
    assert codex_record.codex_home is codex_transcript.codex_home
    assert codex_record.default_sessions_root is codex_transcript.default_sessions_root
    assert codex_record.find_session_file is codex_transcript.find_session_file
    assert codex_record.hook_hints is codex_transcript.hook_hints
    assert codex_record.CodexSaveError is CodexSaveError
    assert codex_capture_delta.CodexSaveError is CodexSaveError


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
        model=codex_record.DEFAULT_MEMORY_MODEL,
        max_chars=codex_record.DEFAULT_MAX_CHARS,
        timeout=1,
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

    result = codex_record.extract_transcript_slice(session)

    assert result.session_id == "019e5ec3-42fe-7f70-9402-7ff20da6be69"
    assert result.cwd == "/tmp/project"
    assert result.scanned_until_line == 5
    assert [(record.role, record.line) for record in result.records] == [
        ("event", 1),
        ("event", 2),
        ("tool", 3),
        ("user", 4),
        ("assistant", 5),
    ]
    assert "AGENTS.md" not in codex_record.format_transcript(result.records)
    assert "large command output" not in codex_record.format_transcript(result.records)
    serialized = codex_record.serialize_transcript_records(result.records)
    source_events = [json.loads(line) for line in session.read_text().splitlines()]
    serialized_rows = json.loads(serialized)
    assert [row["event"] for row in serialized_rows] == source_events
    assert [_canonical_json_bytes(row["event"]) for row in serialized_rows] == [
        _canonical_json_bytes(event) for event in source_events
    ]
    assert "AGENTS.md instructions" in serialized
    assert "large command output" in serialized
    assert result.records[2].event_type == "function_call_output"


def test_projection_omits_privileged_reasoning_and_tools_from_complete_raw(
    tmp_path: Path,
) -> None:
    session = tmp_path / "complete-session.jsonl"
    rows = [
        {"type": "session_meta", "payload": {"id": "complete", "cwd": "/tmp/project"}},
        {"type": "turn_context", "payload": {"private": "transport context"}},
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "developer",
                "content": [{"type": "input_text", "text": "privileged developer prompt"}],
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "private reasoning"}],
                "encrypted_content": "opaque reasoning bytes",
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "user",
                "content": [{
                    "type": "input_text",
                    "text": "# AGENTS.md instructions\ninjected bytes",
                }],
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "function_call_output",
                "call_id": "tool-1",
                "output": "private tool output",
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "visible user body"}],
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "visible assistant body"}],
            },
        },
    ]
    write_jsonl(session, rows)

    extracted = codex_record.extract_transcript_slice(session)
    serialized = codex_record.serialize_transcript_records(extracted.records)
    serialized_events = [row["event"] for row in json.loads(serialized)]
    assert [_canonical_json_bytes(event) for event in serialized_events] == [
        _canonical_json_bytes(event) for event in rows
    ]

    transaction = make_save_transaction(
        host="codex",
        session_file=session,
        session_id=extracted.session_id,
        after_line=0,
        until_line=extracted.scanned_until_line,
    )
    raw_path = tmp_path / f"save-{transaction.idempotency_key}.md"
    raw_path.write_text(codex_record.build_raw_content(extracted, transaction=transaction))
    projection = project_parent_raw(
        raw_path, output_dir=tmp_path / "projection", max_child_bytes=4_000
    )
    projected = "\n".join(path.read_text() for path in projection.child_paths)

    assert projection.record_count == len(rows)
    assert projection.selected_record_count == 2
    assert "visible user body" in projected
    assert "visible assistant body" in projected
    for transport_only in (
        "transport context",
        "privileged developer prompt",
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

    result = codex_record.extract_transcript_slice(session)
    serialized = codex_record.serialize_transcript_records(result.records)

    assert [(record.role, record.line) for record in result.records] == [
        ("event", 1),
        ("user", 2),
        ("tool", 3),
    ]
    assert "data:image/png;base64,AA==" in serialized
    assert '"filename": "notes.pdf"' in serialized
    assert '"name": "view_image"' in serialized


def test_extract_transcript_slice_honors_after_line(tmp_path: Path) -> None:
    session = tmp_path / "session.jsonl"
    sample_session(session)

    result = codex_record.extract_transcript_slice(session, after_line=4)

    assert [(record.role, record.line) for record in result.records] == [("assistant", 5)]


def test_parse_writer_output_sanitizes_keywords() -> None:
    output = json.dumps(
        {
            "should_save": True,
            "content": "Body",
            "keywords": ["Codex", "Codex", "bad,keyword", 123, "Chronovisor"],
            "reason": "durable",
            "evidence_quotes": ["Codex 側の保存ハーネスを実装したい"],
        }
    )

    result = codex_record.parse_writer_output(output)

    assert result.should_save is True
    assert result.keywords == ["Codex", "Chronovisor"]
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

    with pytest.raises(codex_record.CodexSaveError, match="must provide user evidence_quotes"):
        codex_record.parse_writer_output(output)


def test_writer_prompt_forbids_assistant_model_name_substitution(tmp_path: Path) -> None:
    session = tmp_path / "session.jsonl"
    sample_session(session)

    prompt = codex_record.build_writer_prompt(
        codex_record.extract_transcript_slice(session),
        max_chars=codex_record.DEFAULT_MAX_CHARS,
    )

    assert "evidence_quotes" in codex_record.MEMORY_WRITER_SCHEMA["required"]
    assert "exact substrings from USER messages only" in prompt
    assert "do not replace Q-KUN with Qwen" in prompt
    assert "appears only in ASSISTANT text, omit it" in prompt


def test_user_evidence_validation_rejects_assistant_only_quote(tmp_path: Path) -> None:
    session = tmp_path / "session.jsonl"
    sample_session(session)
    transcript_slice = codex_record.extract_transcript_slice(session)
    valid = codex_record.WriterResult(
        should_save=True,
        content="Body",
        keywords=["Codex"],
        reason="durable",
        rejected_keywords=[],
        evidence_quotes=["Codex 側の保存ハーネスを実装したい"],
    )
    invalid = codex_record.WriterResult(
        should_save=True,
        content="Body",
        keywords=["Codex"],
        reason="durable",
        rejected_keywords=[],
        evidence_quotes=["gpt-5.4-mini で要約します"],
    )

    codex_record.validate_user_evidence_quotes(valid, transcript_slice)
    with pytest.raises(codex_record.CodexSaveError, match="not found verbatim in USER"):
        codex_record.validate_user_evidence_quotes(invalid, transcript_slice)


def test_user_evidence_validation_rejects_normalized_model_and_capacity(
    tmp_path: Path,
) -> None:
    transcript_slice = codex_record.TranscriptSlice(
        session_file=tmp_path / "session.jsonl",
        scanned_until_line=2,
        records=[
            codex_record.TranscriptRecord(1, "user", "Q-KUNの32GPレビューを見た"),
            codex_record.TranscriptRecord(2, "assistant", "Qwenの32GBレビューですね"),
        ],
    )
    invalid = codex_record.WriterResult(
        should_save=True,
        content="ユーザーはQwenの32GBレビューを見た。",
        keywords=["Qwen", "32GB"],
        reason="durable",
        rejected_keywords=[],
        evidence_quotes=["Q-KUNの32GPレビューを見た"],
    )
    valid = codex_record.WriterResult(
        should_save=True,
        content="ユーザーはQ-KUNの32GPレビューを見た。",
        keywords=["Q-KUN", "32GP"],
        reason="durable",
        rejected_keywords=[],
        evidence_quotes=["Q-KUNの32GPレビューを見た"],
    )
    invalid_keyword = codex_record.WriterResult(
        should_save=True,
        content="ユーザーはQ-KUNの32GPレビューを見た。",
        keywords=["Qwen"],
        reason="durable",
        rejected_keywords=[],
        evidence_quotes=["Q-KUNの32GPレビューを見た"],
    )

    with pytest.raises(codex_record.CodexSaveError, match="ungrounded protected literal"):
        codex_record.validate_user_evidence_quotes(invalid, transcript_slice)
    with pytest.raises(codex_record.CodexSaveError, match="keywords"):
        codex_record.validate_user_evidence_quotes(invalid_keyword, transcript_slice)
    codex_record.validate_user_evidence_quotes(valid, transcript_slice)


def test_run_memory_writer_is_retired_without_a_subprocess_surface() -> None:
    assert not hasattr(codex_record, "subprocess")

    with pytest.raises(codex_record.CodexSaveError, match="deterministic-lossless"):
        codex_record.run_memory_writer("prompt")


def test_save_mode_updates_state_and_prevents_duplicate_save(tmp_path: Path, monkeypatch) -> None:
    session = tmp_path / "session.jsonl"
    state = tmp_path / "state.json"
    sample_session(session)
    calls: list[dict] = []

    monkeypatch.setattr(record_transaction, "init_chronovisor", lambda: None)
    monkeypatch.setattr(
        codex_record,
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

    monkeypatch.setattr(record_transaction, "save_raw", fake_save_raw)
    monkeypatch.setattr(
        record_transaction, "validate_published_save_receipt", lambda **_kwargs: None
    )

    first_args = args_for(session, state, ignore_state=True)
    first_args.trigger_ingest = True
    first = codex_record.run(first_args)
    second = codex_record.run(args_for(session, state))

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


def test_v2_save_preserves_source_lines_without_legacy_markdown(
    tmp_path: Path, monkeypatch
) -> None:
    session = tmp_path / "session.jsonl"
    state = tmp_path / "state.json"
    raw_dir = tmp_path / "wiki" / "raw"
    sample_session(session)
    source_bytes = session.read_bytes()
    monkeypatch.setenv("CHRONOVISOR_RAW_LAYOUT", "v2")
    monkeypatch.setattr(record_transaction, "RAW_DIR", raw_dir)
    monkeypatch.setattr(record_transaction, "init_chronovisor", lambda: None)
    monkeypatch.setattr(
        record_transaction,
        "save_raw",
        lambda *_args, **_kwargs: pytest.fail("v2 must not publish legacy Markdown"),
    )

    result = codex_record.run(args_for(session, state, ignore_state=True))

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
    assert json.loads(state.read_text())["files"][str(session)]["last_saved_line"] == 5


def test_v2_save_accepts_oversized_native_record_after_metadata(
    tmp_path: Path, monkeypatch
) -> None:
    session = tmp_path / "oversized.jsonl"
    state = tmp_path / "state.json"
    raw_dir = tmp_path / "wiki" / "raw"
    write_jsonl(
        session,
        [
            {"type": "session_meta", "payload": {"id": "oversized-v2"}},
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "X" * 2_000}],
                },
            },
        ],
    )
    monkeypatch.setenv("CHRONOVISOR_RAW_LAYOUT", "v2")
    monkeypatch.setattr(record_transaction, "RAW_DIR", raw_dir)
    monkeypatch.setattr(record_transaction, "init_chronovisor", lambda: None)
    args = args_for(session, state, ignore_state=True)
    args.max_chars = 128

    result = codex_record.run(args)

    assert result["status"] == "saved"
    assert result["chunk_count"] == 2
    assert Path(result["save_result"]["path"]).read_bytes() == session.read_bytes()
    assert result["scanned_until_line"] == 2
    assert json.loads(state.read_text())["files"][str(session)]["last_saved_line"] == 2


@pytest.mark.parametrize(
    "max_chars",
    [codex_record.DEFAULT_MAX_CHARS, 128],
    ids=["normal", "multi-chunk"],
)
def test_runtime_context_keeps_v2_stop_capture_under_supplied_root(
    tmp_path: Path, monkeypatch, max_chars: int
) -> None:
    session = tmp_path / "session.jsonl"
    runtime = RuntimeContext(tmp_path / "runtime-context")
    init_chronovisor(runtime)
    forbidden_default = tmp_path / "global-default"
    sample_session(session)
    monkeypatch.setenv("CHRONOVISOR_RAW_LAYOUT", "v2")
    monkeypatch.setattr(record_transaction, "RAW_DIR", forbidden_default / "raw")
    monkeypatch.setattr(
        codex_record,
        "DEFAULT_STATE_FILE",
        forbidden_default / "codex-save-state.json",
    )
    args = args_for(session, codex_record.DEFAULT_STATE_FILE, ignore_state=True)
    args.state_file = None
    args.max_chars = max_chars

    result = codex_record.run(args, context=runtime)
    save_results = result.get("save_results", [result["save_result"]])

    assert result["status"] == "saved"
    if max_chars == codex_record.DEFAULT_MAX_CHARS:
        assert result["chunk_count"] == 1
    else:
        assert result["chunk_count"] > 1
    assert result["scanned_until_line"] == 5
    assert all(
        Path(save_result["path"]).is_relative_to(runtime.raw_dir)
        for save_result in save_results
    )
    assert json.loads(runtime.codex_state_file.read_text())["files"][str(session)][
        "last_saved_line"
    ] == 5
    assert runtime.pages_dir.is_dir()
    assert runtime.system_dir.is_dir()
    assert not forbidden_default.exists()


def test_explicit_default_state_file_remains_an_override(tmp_path: Path) -> None:
    parser = codex_record.build_parser()
    runtime = RuntimeContext(tmp_path / "runtime-context")

    assert parser.parse_args([]).state_file is None
    explicit = parser.parse_args(
        ["--state-file", str(codex_record.DEFAULT_STATE_FILE)]
    )
    assert codex_record._resolve_state_file(explicit, context=runtime) == (
        codex_record.DEFAULT_STATE_FILE
    )


def test_runtime_context_rejects_legacy_layout_before_global_publish(
    tmp_path: Path, monkeypatch
) -> None:
    session = tmp_path / "session.jsonl"
    runtime = RuntimeContext(tmp_path / "runtime-context")
    forbidden_default = tmp_path / "global-default"
    sample_session(session)
    runtime.root.mkdir()
    runtime.config_file.write_text(
        '[decision_policies]\nraw_capture = "enabled"\n[raw]\nlayout = "legacy"\n'
    )
    monkeypatch.setattr(
        record_transaction,
        "init_chronovisor",
        lambda *, context=None: None,
    )
    monkeypatch.delenv("CHRONOVISOR_RAW_LAYOUT", raising=False)
    monkeypatch.delenv("CHRONOVISOR_DECISION_POLICY_RAW_CAPTURE", raising=False)
    monkeypatch.setattr(record_transaction, "RAW_DIR", forbidden_default / "raw")
    monkeypatch.setattr(
        codex_record,
        "DEFAULT_STATE_FILE",
        forbidden_default / "codex-save-state.json",
    )
    calls: list[None] = []

    def fail_global_publish(*_args, **_kwargs):
        calls.append(None)
        pytest.fail("RuntimeContext must not call the global legacy writer")

    monkeypatch.setattr(record_transaction, "save_raw", fail_global_publish)
    args = args_for(session, codex_record.DEFAULT_STATE_FILE, ignore_state=True)
    args.state_file = None

    with pytest.raises(codex_record.CodexSaveError, match="RuntimeContext.*v2"):
        codex_record.run(args, context=runtime)

    assert calls == []
    assert not runtime.raw_dir.exists()
    assert not forbidden_default.exists()


def test_shadow_save_compares_legacy_records_with_native_source(
    tmp_path: Path, monkeypatch
) -> None:
    session = tmp_path / "session.jsonl"
    state = tmp_path / "state.json"
    raw_dir = tmp_path / "wiki" / "raw"
    sample_session(session)
    monkeypatch.setenv("CHRONOVISOR_RAW_LAYOUT", "shadow")
    monkeypatch.setattr(record_transaction, "RAW_DIR", raw_dir)
    monkeypatch.setattr(record_transaction, "init_chronovisor", lambda: None)
    monkeypatch.setattr(
        record_transaction,
        "save_raw",
        lambda _content, **_kwargs: {"saved": "legacy-authority.md"},
    )
    monkeypatch.setattr(
        record_transaction, "validate_published_save_receipt", lambda **_kwargs: None
    )

    result = codex_record.run(args_for(session, state, ignore_state=True))

    shadow = result["save_result"]
    assert shadow["layout"] == "shadow"
    assert shadow["shadow_comparison"]["status"] == "match"
    assert shadow["shadow_comparison"]["missing"] == 0
    assert Path(shadow["shadow_result"]["path"]).read_bytes() == session.read_bytes()


def test_one_stop_drains_every_bounded_transcript_chunk(
    tmp_path: Path, monkeypatch
) -> None:
    session = tmp_path / "session.jsonl"
    state = tmp_path / "state.json"
    sample_session(session)
    calls: list[str] = []
    monkeypatch.setattr(record_transaction, "init_chronovisor", lambda: None)
    monkeypatch.setattr(
        record_transaction,
        "save_raw",
        lambda content, **_kwargs: calls.append(content)
        or {"saved": f"raw-{len(calls)}.md"},
    )
    monkeypatch.setattr(
        record_transaction, "validate_published_save_receipt", lambda **_kwargs: None
    )
    extracted = codex_record.extract_transcript_slice(session)
    args = args_for(session, state, ignore_state=True)
    args.max_chars = max(
        len(codex_record._serialized_records_bytes([record]))
        for record in extracted.records
    )

    result = codex_record.run(args)

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
    monkeypatch.setattr(record_transaction, "init_chronovisor", lambda: None)
    monkeypatch.setattr(
        codex_record,
        "save_raw",
        lambda content, **kwargs: calls.append({"content": content, **kwargs})
        or {"saved": f"raw-{len(calls)}.md"},
    )
    monkeypatch.setattr(
        codex_record, "validate_published_save_receipt", lambda **_kwargs: None
    )
    args = args_for(session, state, ignore_state=True)
    args.max_chars = 128

    result = codex_record.run(args)
    payloads = [_fragment_payload(call["content"]) for call in calls]
    reconstructed = b"".join(base64.b64decode(row["data"]) for row in payloads)
    expected = codex_record._serialized_records_bytes(
        [codex_record.extract_transcript_slice(session).records[0]]
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
    monkeypatch.setattr(record_transaction, "init_chronovisor", lambda: None)
    attempts = 0

    def fail_second_fragment(_content: str, **_kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 2:
            raise RuntimeError("injected fragment failure")
        return {"saved": f"raw-{attempts}.md"}

    monkeypatch.setattr(codex_record, "save_raw", fail_second_fragment)
    monkeypatch.setattr(
        codex_record, "validate_published_save_receipt", lambda **_kwargs: None
    )
    args = args_for(session, state, ignore_state=True)
    args.max_chars = 128

    with pytest.raises(RuntimeError, match="injected fragment failure"):
        codex_record.run(args)

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
        record_transaction, "init_chronovisor", lambda: pytest.fail("policy gate must precede init")
    )
    monkeypatch.setattr(
        record_transaction, "save_raw", lambda *_args, **_kwargs: pytest.fail("must not publish")
    )
    monkeypatch.setattr(
        codex_record,
        "run_memory_writer",
        lambda *_args, **_kwargs: pytest.fail("must not start a model"),
    )

    result = codex_record.run(args_for(session, state))

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
    raw_dir = tmp_path / "wiki" / "raw"
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

    monkeypatch.setattr(record_transaction, "init_chronovisor", lambda: None)
    monkeypatch.setattr(record_transaction, "RAW_DIR", raw_dir)

    def durable_fake_save(content: str, *, idempotency_key: str, **_kwargs):
        save_calls.append(idempotency_key)
        path = raw_dir / f"save-{idempotency_key}.md"
        path.write_text(content, encoding="utf-8")
        return {"saved": path.name, "path": str(path)}

    real_write_state = codex_record.write_state
    state_writes = 0

    def fail_first_state_commit(path: Path, payload: dict) -> None:
        nonlocal state_writes
        state_writes += 1
        if state_writes == 1:
            raise OSError("injected crash after raw publish")
        real_write_state(path, payload)

    monkeypatch.setattr(
        codex_record,
        "run_memory_writer",
        lambda *args, **kwargs: pytest.fail("recovery must not start a writer"),
    )
    monkeypatch.setattr(record_transaction, "save_raw", durable_fake_save)
    monkeypatch.setattr(record_transaction, "write_state", fail_first_state_commit)

    with pytest.raises(OSError, match="injected crash"):
        codex_record.run(args_for(session, state))

    retry = codex_record.run(args_for(session, state))

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
    raw_dir = tmp_path / "wiki" / "raw"
    raw_dir.mkdir()
    sample_session(session)
    monkeypatch.setattr(record_transaction, "init_chronovisor", lambda: None)
    monkeypatch.setattr(record_transaction, "RAW_DIR", raw_dir)
    monkeypatch.setattr(
        codex_record,
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

    monkeypatch.setattr(record_transaction, "save_raw", corrupt_save)

    with pytest.raises(codex_record.CodexSaveError, match="receipt validation failed"):
        codex_record.run(args_for(session, state, ignore_state=True))

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

    monkeypatch.setattr(record_transaction, "init_chronovisor", lambda: None)
    first_save_entered = threading.Event()
    release_first_save = threading.Event()
    duplicate_save_entered = threading.Event()
    second_lock_attempted = threading.Event()
    calls_guard = threading.Lock()
    lock_attempts: list[int] = []
    saves: list[str] = []

    from contextlib import contextmanager

    real_transaction_lock = codex_record.save_transaction_lock

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
        codex_record,
        "run_memory_writer",
        lambda *args, **kwargs: pytest.fail("normal capture must not start a writer"),
    )
    monkeypatch.setattr(record_transaction, "save_raw", fake_save)
    monkeypatch.setattr(
        record_transaction, "validate_published_save_receipt", lambda **_kwargs: None
    )
    monkeypatch.setattr(codex_record, "save_transaction_lock", instrumented_transaction_lock)
    second_started = threading.Event()

    def second_worker():
        second_started.set()
        return codex_record.run(args_for(session, state))

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(codex_record.run, args_for(session, state))
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
    monkeypatch.setattr(record_transaction, "init_chronovisor", lambda: None)
    monkeypatch.setattr(
        codex_record,
        "run_memory_writer",
        lambda *args, **kwargs: pytest.fail("normal capture must not start a writer"),
    )
    saved: list[str] = []
    monkeypatch.setattr(
        record_transaction,
        "save_raw",
        lambda content, **kwargs: saved.append(content) or {"saved": "raw.md"},
    )
    monkeypatch.setattr(
        record_transaction, "validate_published_save_receipt", lambda **_kwargs: None
    )

    result = codex_record.run(args_for(session, state, ignore_state=True))

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
    monkeypatch.delenv(codex_record.HOOK_ENABLE_ENV, raising=False)
    monkeypatch.setattr(record_transaction, "init_chronovisor", lambda: None)

    result = codex_record.run(args, stdin_text="{}")

    assert result["status"] == "disabled"
