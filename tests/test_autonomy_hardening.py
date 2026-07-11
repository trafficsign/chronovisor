from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from llm_wiki_mcp import background_jobs, codex_save, recall_hints, session_sweeper
from llm_wiki_mcp.frontmatter import normalize_nested, parse
from llm_wiki_mcp.jsonl import read_jsonl


def test_jsonl_reader_preserves_unicode_line_separator(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    path.write_text(json.dumps({"text": "before\u2028after"}, ensure_ascii=False) + "\n", encoding="utf-8")

    assert read_jsonl(path) == [{"text": "before\u2028after"}]


def test_nested_frontmatter_merges_disjoint_metadata() -> None:
    text = "---\npermalink: x\n---\n---\ntitle: T\ntags: [a]\n---\nbody\n"

    updated, result = normalize_nested(text)
    meta, body = parse(updated)

    assert result["changed"] is True
    assert meta == {"permalink": "x", "title": "T", "tags": ["a"]}
    assert body == "body\n"


def test_nested_frontmatter_conflict_is_not_silently_resolved() -> None:
    text = "---\ntitle: Outer\n---\n---\ntitle: Inner\n---\nbody\n"

    updated, result = normalize_nested(text)

    assert updated == text
    assert result["reason"] == "conflicting_nested_frontmatter"


def test_codex_detects_unified_exec_apply_patch(tmp_path: Path) -> None:
    session = tmp_path / "session.jsonl"
    rows = [
        {"type": "session_meta", "payload": {"id": "019f0000-0000-7000-8000-000000000000"}},
        {"type": "response_item", "payload": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "fix"}]}},
        {"type": "response_item", "payload": {"type": "custom_tool_call", "name": "exec", "input": "text(await tools.apply_patch('*** Begin Patch'))"}},
        {"type": "response_item", "payload": {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "done"}]}},
    ]
    session.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    result = codex_save.extract_transcript_slice(session)

    assert result.has_file_changes is True
    assert codex_save.should_process(result, {}) == (True, "file_changes")


def test_session_sweeper_excludes_internal_codex_sessions(tmp_path: Path) -> None:
    user = tmp_path / "user.jsonl"
    subagent = tmp_path / "subagent.jsonl"
    automated = tmp_path / "automated.jsonl"
    user.write_text(json.dumps({"type": "session_meta", "payload": {
        "originator": "Codex Desktop", "thread_source": "user", "source": "vscode"
    }}) + "\n", encoding="utf-8")
    subagent.write_text(json.dumps({"type": "session_meta", "payload": {
        "originator": "Codex Desktop", "thread_source": "subagent", "source": {"subagent": {}}
    }}) + "\n", encoding="utf-8")
    automated.write_text(json.dumps({"type": "session_meta", "payload": {
        "originator": "codex_exec", "source": "exec"
    }}) + "\n", encoding="utf-8")

    assert session_sweeper._is_user_codex_session(user) is True
    assert session_sweeper._is_user_codex_session(subagent) is False
    assert session_sweeper._is_user_codex_session(automated) is False


def test_memory_writer_repairs_grounding_once(tmp_path: Path, monkeypatch) -> None:
    transcript = codex_save.TranscriptSlice(
        session_file=tmp_path / "session.jsonl",
        scanned_until_line=1,
        records=[codex_save.TranscriptRecord(line=1, role="user", text="Q-KUN 32GP")],
    )
    outputs = iter([
        codex_save.WriterResult(True, "Qwen 32GB", ["Qwen"], "first", [], ["Q-KUN 32GB"]),
        codex_save.WriterResult(False, "", [], "ungrounded", [], []),
    ])
    prompts: list[str] = []

    def writer(prompt, **_kwargs):
        prompts.append(prompt)
        return next(outputs)

    monkeypatch.setattr(codex_save, "run_memory_writer", writer)
    result = codex_save.run_grounded_memory_writer(
        "original prompt", transcript, model="test", reasoning_effort="medium", timeout=1
    )

    assert result.should_save is False
    assert len(prompts) == 2
    assert "deterministic grounding gate" in prompts[1]


def test_session_sweeper_backoff_does_not_pin_queue(tmp_path: Path, monkeypatch) -> None:
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


def test_legacy_auto_hints_are_quarantined(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "query-hints.json"
    path.write_text(json.dumps({"version": 1, "hints": [
        {"page_id": "bad", "query": "q", "source": "recall-auto-apply"},
        {"page_id": "manual", "query": "q", "source": "manual"},
    ]}), encoding="utf-8")

    result = recall_hints.quarantine_legacy_query_hints(path=path, write=True)

    assert result["quarantined"] == 1
    assert [row["page_id"] for row in recall_hints.load_query_hints(path)] == ["manual"]


def test_background_job_failure_is_durable_and_retryable(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(background_jobs, "JOB_DIR", tmp_path)
    monkeypatch.setattr(background_jobs, "STATE_FILE", tmp_path / "state.json")
    monkeypatch.setattr(background_jobs, "LOCK_FILE", tmp_path / "state.lock")
    monkeypatch.setattr(
        background_jobs.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout="", stderr="temporary failure"),
    )
    job = background_jobs.enqueue_job(
        name="save", module="example", args=[], env={}, stdin_text="{}"
    )

    result = background_jobs.run_job(job["job_id"])
    stored = json.loads((tmp_path / "state.json").read_text())["jobs"][job["job_id"]]

    assert result["status"] == "retry_wait"
    assert stored["exit_code"] == 1
    assert "temporary failure" in stored["output_tail"]
