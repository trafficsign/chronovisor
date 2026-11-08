"""Golden replay tests for the current Claude Code/Codex capture paths.

The source input is deliberately written in two passes: the first pass ends
with an incomplete JSONL record, then the same path receives the completion
and one continuation record.  Expected raw bytes below are literal fixtures;
they are not generated from the parser under test.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import pytest

from chronovisor.capture import background_jobs


@dataclass(frozen=True)
class GoldenReplay:
    name: str
    host: str
    module: str
    session_id: str
    first_raw: bytes
    partial_tail: bytes
    appended_bytes: bytes
    second_raw: bytes
    first_until_line: int


CODEX_TOOLS_FIRST = (
    b'{"timestamp":"2026-11-08T01:00:00Z","type":"session_meta","payload":{"id":"00000000-0000-7000-8000-000000000001","cwd":"/work/codex-tools","originator":"Codex Desktop","cli_version":"1.2.3","source":"vscode","model_provider":"openai"}}\n'
    b'{"timestamp":"2026-11-08T01:00:01Z","type":"response_item","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"Inspect the durable capture receipt."}]}}\n'
    b'{"timestamp":"2026-11-08T01:00:02Z","type":"response_item","payload":{"type":"custom_tool_call","name":"exec","input":{"cmd":"git status --short"}}}\n'
)
CODEX_TOOLS_TAIL = b'{"timestamp":"2026-11-08T01:00:03Z","type":"response_item","payload":{"type":"message","role":"assistant","content":[{"type":"output_text","text":"Capture is'
CODEX_TOOLS_APPEND = (
    b' ready to replay."}]}}\n'
    b'{"timestamp":"2026-11-08T01:00:04Z","type":"response_item","payload":{"type":"function_call_output","output":"ok"}}\n'
)
CODEX_TOOLS_SECOND = (
    b'{"timestamp":"2026-11-08T01:00:03Z","type":"response_item","payload":{"type":"message","role":"assistant","content":[{"type":"output_text","text":"Capture is ready to replay."}]}}\n'
    b'{"timestamp":"2026-11-08T01:00:04Z","type":"response_item","payload":{"type":"function_call_output","output":"ok"}}\n'
)

CODEX_STRUCTURED_FIRST = (
    b'{"timestamp":"2026-11-08T02:00:00Z","type":"session_meta","payload":{"id":"00000000-0000-7000-8000-000000000002","cwd":"/work/codex-structured","source":"cli","model_provider":"openai"}}\n'
    b'{"timestamp":"2026-11-08T02:00:01Z","type":"response_item","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"Summarize the notes."},{"type":"input_file","filename":"notes.md","file_data":"bm90ZXM="}]}}\n'
    b'{"timestamp":"2026-11-08T02:00:02Z","type":"response_item","payload":{"type":"function_call","name":"write_file","arguments":"{\\"path\\":\\"notes.md\\"}"}}\n'
)
CODEX_STRUCTURED_TAIL = b'{"timestamp":"2026-11-08T02:00:03Z","type":"response_item","payload":{"type":"message","role":"assistant","phase":"commentary","content":[{"type":"output_text","text":"Notes'
CODEX_STRUCTURED_APPEND = (
    b' restored."}]}}\n'
    b'{"timestamp":"2026-11-08T02:00:04Z","type":"response_item","payload":{"type":"message","role":"assistant","content":[{"type":"output_text","text":"Done."}]}}\n'
)
CODEX_STRUCTURED_SECOND = (
    b'{"timestamp":"2026-11-08T02:00:03Z","type":"response_item","payload":{"type":"message","role":"assistant","phase":"commentary","content":[{"type":"output_text","text":"Notes restored."}]}}\n'
    b'{"timestamp":"2026-11-08T02:00:04Z","type":"response_item","payload":{"type":"message","role":"assistant","content":[{"type":"output_text","text":"Done."}]}}\n'
)

CLAUDE_TOOLS_FIRST = (
    b'{"type":"user","sessionId":"claude-golden-1","cwd":"/work/claude-tools","timestamp":"2026-11-08T03:00:00Z","message":{"content":"Review the journal."}}\n'
    b'{"type":"assistant","sessionId":"claude-golden-1","cwd":"/work/claude-tools","timestamp":"2026-11-08T03:00:01Z","message":{"content":[{"type":"text","text":"I will inspect it."},{"type":"tool_use","name":"Read","input":{"file_path":"capture.md"}}]}}\n'
    b'{"type":"user","sessionId":"claude-golden-1","cwd":"/work/claude-tools","timestamp":"2026-11-08T03:00:02Z","message":{"content":[{"type":"tool_result","tool_use_id":"read-1","content":"existing bytes"}]}}\n'
)
CLAUDE_TOOLS_TAIL = b'{"type":"assistant","sessionId":"claude-golden-1","cwd":"/work/claude-tools","timestamp":"2026-11-08T03:00:03Z","message":{"content":[{"type":"text","text":"The'
CLAUDE_TOOLS_APPEND = (
    b' journal is durable."}]}}\n'
    b'{"type":"user","sessionId":"claude-golden-1","cwd":"/work/claude-tools","timestamp":"2026-11-08T03:00:04Z","message":{"content":"Continue with replay."}}\n'
)
CLAUDE_TOOLS_SECOND = (
    b'{"type":"assistant","sessionId":"claude-golden-1","cwd":"/work/claude-tools","timestamp":"2026-11-08T03:00:03Z","message":{"content":[{"type":"text","text":"The journal is durable."}]}}\n'
    b'{"type":"user","sessionId":"claude-golden-1","cwd":"/work/claude-tools","timestamp":"2026-11-08T03:00:04Z","message":{"content":"Continue with replay."}}\n'
)

CLAUDE_EDIT_FIRST = (
    b'{"type":"user","sessionId":"claude-golden-2","cwd":"/work/claude-edit","timestamp":"2026-11-08T04:00:00Z","message":{"content":"Add a fixture."}}\n'
    b'{"type":"assistant","sessionId":"claude-golden-2","cwd":"/work/claude-edit","timestamp":"2026-11-08T04:00:01Z","message":{"content":[{"type":"text","text":"I can add it."}]}}\n'
    b'{"type":"assistant","sessionId":"claude-golden-2","cwd":"/work/claude-edit","timestamp":"2026-11-08T04:00:02Z","message":{"content":[{"type":"tool_use","name":"Write","input":{"file_path":"fixture.txt","content":"golden"}}]}}\n'
)
CLAUDE_EDIT_TAIL = b'{"type":"assistant","sessionId":"claude-golden-2","cwd":"/work/claude-edit","timestamp":"2026-11-08T04:00:03Z","message":{"content":[{"type":"text","text":"Fixture'
CLAUDE_EDIT_APPEND = (
    b' complete."}]}}\n'
    b'{"type":"user","sessionId":"claude-golden-2","cwd":"/work/claude-edit","timestamp":"2026-11-08T04:00:04Z","message":{"content":"Thanks."}}\n'
)
CLAUDE_EDIT_SECOND = (
    b'{"type":"assistant","sessionId":"claude-golden-2","cwd":"/work/claude-edit","timestamp":"2026-11-08T04:00:03Z","message":{"content":[{"type":"text","text":"Fixture complete."}]}}\n'
    b'{"type":"user","sessionId":"claude-golden-2","cwd":"/work/claude-edit","timestamp":"2026-11-08T04:00:04Z","message":{"content":"Thanks."}}\n'
)


GOLDEN_REPLAYS = (
    GoldenReplay(
        name="codex-tools",
        host="codex",
        module="chronovisor.hosts.codex_record",
        session_id="00000000-0000-7000-8000-000000000001",
        first_raw=CODEX_TOOLS_FIRST,
        partial_tail=CODEX_TOOLS_TAIL,
        appended_bytes=CODEX_TOOLS_APPEND,
        second_raw=CODEX_TOOLS_SECOND,
        first_until_line=3,
    ),
    GoldenReplay(
        name="codex-structured",
        host="codex",
        module="chronovisor.hosts.codex_record",
        session_id="00000000-0000-7000-8000-000000000002",
        first_raw=CODEX_STRUCTURED_FIRST,
        partial_tail=CODEX_STRUCTURED_TAIL,
        appended_bytes=CODEX_STRUCTURED_APPEND,
        second_raw=CODEX_STRUCTURED_SECOND,
        first_until_line=3,
    ),
    GoldenReplay(
        name="claude-tools",
        host="claude-code",
        module="chronovisor.hosts.claude_code_record",
        session_id="claude-golden-1",
        first_raw=CLAUDE_TOOLS_FIRST,
        partial_tail=CLAUDE_TOOLS_TAIL,
        appended_bytes=CLAUDE_TOOLS_APPEND,
        second_raw=CLAUDE_TOOLS_SECOND,
        first_until_line=3,
    ),
    GoldenReplay(
        name="claude-edit",
        host="claude-code",
        module="chronovisor.hosts.claude_code_record",
        session_id="claude-golden-2",
        first_raw=CLAUDE_EDIT_FIRST,
        partial_tail=CLAUDE_EDIT_TAIL,
        appended_bytes=CLAUDE_EDIT_APPEND,
        second_raw=CLAUDE_EDIT_SECOND,
        first_until_line=3,
    ),
)


@pytest.mark.parametrize("replay", GOLDEN_REPLAYS, ids=lambda value: value.name)
def test_capture_golden_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    replay: GoldenReplay,
) -> None:
    """Capture a partial transcript, append it, and preserve exact source bytes."""
    capture_root = tmp_path / "capture-root"
    input_path = tmp_path / "input" / "session.jsonl"
    state_file = capture_root / f"{replay.host}-save-state.json"
    job_dir = capture_root / "runtime" / "background-jobs"
    input_path.parent.mkdir(parents=True)
    capture_root.mkdir(parents=True)
    # The replay is config-driven; the suite-wide isolation override would
    # otherwise reach the capture child and select the legacy layout.
    monkeypatch.delenv("CHRONOVISOR_RAW_LAYOUT", raising=False)
    (capture_root / "config.toml").write_text(
        '[raw]\nlayout = "v2"\n', encoding="utf-8"
    )

    # One fixed source path is used for both replays; only the isolated root changes per test.
    input_path.write_bytes(replay.first_raw + replay.partial_tail)
    monkeypatch.setattr(background_jobs, "CHRONOVISOR_ROOT", capture_root)
    monkeypatch.setattr(background_jobs, "JOB_DIR", job_dir)
    monkeypatch.setattr(background_jobs, "STATE_FILE", job_dir / "state.json")
    monkeypatch.setattr(background_jobs, "LOCK_FILE", job_dir / "state.lock")

    first_job = _enqueue(replay, input_path, state_file)
    first_result = background_jobs.run_job(first_job["job_id"])
    assert first_result["status"] == "completed"
    assert first_result["exit_code"] == 0
    _assert_job_state(first_job["job_id"], attempts=1)

    first_commit, first_data = _capture_receipt(capture_root)
    _assert_commit(
        first_commit,
        replay=replay,
        input_path=input_path,
        after_line=0,
        until_line=replay.first_until_line,
        source_bytes=replay.first_raw,
        offset=0,
    )
    assert first_data == replay.first_raw
    _assert_cursor(state_file, input_path, replay.first_until_line)

    with input_path.open("ab") as handle:
        handle.write(replay.appended_bytes)
    second_job = _enqueue(replay, input_path, state_file)
    second_result = background_jobs.run_job(second_job["job_id"])
    assert second_result["status"] == "completed"
    assert second_result["exit_code"] == 0
    _assert_job_state(second_job["job_id"], attempts=1)

    commits, data = _capture_receipt(capture_root, all_commits=True)
    assert len(commits) == 2
    assert data == replay.first_raw + replay.second_raw
    _assert_commit(
        commits[0],
        replay=replay,
        input_path=input_path,
        after_line=0,
        until_line=replay.first_until_line,
        source_bytes=replay.first_raw,
        offset=0,
    )
    _assert_commit(
        commits[1],
        replay=replay,
        input_path=input_path,
        after_line=replay.first_until_line,
        until_line=replay.first_until_line + 2,
        source_bytes=replay.second_raw,
        offset=len(replay.first_raw),
    )
    assert commits[0]["idempotency_key"] != commits[1]["idempotency_key"]
    assert (
        commits[0]["idempotency_key"].split("-from", 1)[0]
        == commits[1]["idempotency_key"].split("-from", 1)[0]
    )
    _assert_cursor(state_file, input_path, replay.first_until_line + 2)


def _enqueue(
    replay: GoldenReplay, input_path: Path, state_file: Path
) -> dict[str, object]:
    return background_jobs.enqueue_job(
        name=f"{replay.host}-save",
        module=replay.module,
        args=[
            "--session-file",
            str(input_path),
            "--state-file",
            str(state_file),
            "--save",
        ],
        env={},
        stdin_text=json.dumps(
            {"session_id": replay.session_id, "session_file": str(input_path)},
            ensure_ascii=False,
        ),
    )


def _capture_receipt(
    capture_root: Path, *, all_commits: bool = False
) -> tuple[dict[str, object] | list[dict[str, object]], bytes]:
    raw_root = capture_root / "raw"
    data_paths = sorted(raw_root.rglob("*.jsonl.open"))
    commit_paths = sorted(raw_root.rglob("*.commits.jsonl"))
    assert len(data_paths) == 1
    assert len(commit_paths) == 1
    commits = [json.loads(line) for line in commit_paths[0].read_text().splitlines()]
    assert commits
    assert all(isinstance(commit, dict) for commit in commits)
    return (commits if all_commits else commits[0]), data_paths[0].read_bytes()


def _assert_commit(
    commit: dict[str, object],
    *,
    replay: GoldenReplay,
    input_path: Path,
    after_line: int,
    until_line: int,
    source_bytes: bytes,
    offset: int,
) -> None:
    idempotency_key = commit["idempotency_key"]
    assert isinstance(idempotency_key, str)
    assert re.fullmatch(rf"{replay.host}-[0-9a-f]{{24}}-from\d+-to\d+", idempotency_key)
    assert commit["raw_id"] == f"save-{idempotency_key}.md"
    assert commit["schema"] == "chronovisor.raw-segment-commit.v1"
    assert commit["host"] == replay.host
    assert isinstance(commit["session_key"], str)
    assert re.fullmatch(r"[0-9a-f]{24}", commit["session_key"])
    assert commit["session_id"] == replay.session_id
    assert commit["source_file"] == str(input_path.resolve())
    assert commit["after_line"] == after_line
    assert commit["until_line"] == until_line
    assert commit["offset"] == offset
    assert commit["length"] == len(source_bytes)
    assert commit["record_count"] == until_line - after_line
    assert commit["part"] == 1
    assert commit["sha256"] == hashlib.sha256(source_bytes).hexdigest()


def _assert_cursor(state_file: Path, input_path: Path, last_saved_line: int) -> None:
    state = json.loads(state_file.read_text(encoding="utf-8"))
    entry = state["files"][str(input_path)]
    assert entry["last_saved_line"] == last_saved_line
    assert entry["status"] == "saved"
    assert isinstance(entry["last_saved_at"], str)


def _assert_job_state(job_id: object, *, attempts: int) -> None:
    assert isinstance(job_id, str)
    state = json.loads(background_jobs.STATE_FILE.read_text(encoding="utf-8"))
    job = state["jobs"][job_id]
    assert job["status"] == "completed"
    assert job["attempts"] == attempts
    assert job["exit_code"] == 0
    assert job["next_retry_at"] is None
    assert job["stdin"] == ""
