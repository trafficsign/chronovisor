from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from chronovisor.core import background_jobs as core_background_jobs
from chronovisor.core.frontmatter import normalize_nested, parse
from chronovisor.core.jsonl import read_jsonl
from chronovisor.hosts import codex_record
from chronovisor.ingest import recall_hints
from chronovisor.ops import background_jobs, session_sweeper


@pytest.fixture(autouse=True)
def _valid_background_job_okf_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "wiki"
    root.mkdir()
    for name in ("index.md", "log.md", "schema.md"):
        (root / name).write_text("legacy\n", encoding="utf-8")
    monkeypatch.setattr(core_background_jobs, "CHRONOVISOR_ROOT", root)
    monkeypatch.setattr(session_sweeper, "CHRONOVISOR_ROOT", root)


def test_ops_background_jobs_is_core_module_alias() -> None:
    assert background_jobs is core_background_jobs


def test_session_sweeper_write_false_still_requires_writer_lease(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chronovisor.core.okf_cutover import OKFStartupBlocked

    root = tmp_path / "blocked"
    root.mkdir()
    (root / "private.txt").write_text("canary", encoding="utf-8")
    monkeypatch.setattr(session_sweeper, "CHRONOVISOR_ROOT", root)
    monkeypatch.setattr(
        session_sweeper,
        "discover_pending",
        lambda *_args, **_kwargs: pytest.fail("sweeper ran before startup gate"),
    )

    with pytest.raises(OKFStartupBlocked):
        session_sweeper.run_sweeper(write=False)

    assert [path.name for path in root.iterdir()] == ["private.txt"]


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

    result = codex_record.extract_transcript_slice(session)

    assert result.has_file_changes is True
    assert codex_record.should_process(result, {}) == (True, "file_changes")


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
    transcript = codex_record.TranscriptSlice(
        session_file=tmp_path / "session.jsonl",
        scanned_until_line=1,
        records=[codex_record.TranscriptRecord(line=1, role="user", text="Q-KUN 32GP")],
    )
    outputs = iter([
        codex_record.WriterResult(True, "Qwen 32GB", ["Qwen"], "first", [], ["Q-KUN 32GB"]),
        codex_record.WriterResult(False, "", [], "ungrounded", [], []),
    ])
    prompts: list[str] = []

    def writer(prompt, **_kwargs):
        prompts.append(prompt)
        return next(outputs)

    monkeypatch.setattr(codex_record, "run_memory_writer", writer)
    result = codex_record.run_grounded_memory_writer(
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


def test_canonicalize_query_hint_targets_rewrites_only_page_identity(
    tmp_path: Path,
) -> None:
    path = tmp_path / "query-hints.json"
    original = {
        "version": 1,
        "hints": [
            {
                "page_id": "former-page",
                "query": "historical product name stays searchable",
                "normalize_key": "immutable-provenance-key",
            }
        ],
    }
    path.write_text(json.dumps(original), encoding="utf-8")

    result = recall_hints.canonicalize_query_hint_targets(
        path=path,
        aliases={"former-page": "chronovisor/current-page"},
        write=True,
    )

    rewritten = json.loads(path.read_text(encoding="utf-8"))["hints"][0]
    assert result["changed"] == 1
    assert rewritten["page_id"] == "current-page"
    assert rewritten["query"] == original["hints"][0]["query"]
    assert rewritten["normalize_key"] == "immutable-provenance-key"


def test_background_job_failure_is_durable_and_retryable(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(background_jobs, "JOB_DIR", tmp_path)
    monkeypatch.setattr(background_jobs, "STATE_FILE", tmp_path / "state.json")
    monkeypatch.setattr(background_jobs, "LOCK_FILE", tmp_path / "state.lock")
    commands: list[list[str]] = []

    def fail(command: list[str], **_kwargs) -> SimpleNamespace:
        commands.append(command)
        return SimpleNamespace(returncode=1, stdout="", stderr="temporary failure")

    monkeypatch.setattr(background_jobs.subprocess, "run", fail)
    job = background_jobs.enqueue_job(
        name="save", module="example", args=[], env={}, stdin_text="{}"
    )

    result = background_jobs.run_job(job["job_id"])
    stored = json.loads((tmp_path / "state.json").read_text())["jobs"][job["job_id"]]

    assert result["status"] == "retry_wait"
    assert commands == [[background_jobs.sys.executable, "-m", "example"]]
    assert stored["exit_code"] == 1
    assert "temporary failure" in stored["output_tail"]


def test_background_job_child_env_keeps_only_runtime_and_fixed_job_flags(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(background_jobs, "JOB_DIR", tmp_path)
    monkeypatch.setattr(background_jobs, "STATE_FILE", tmp_path / "state.json")
    monkeypatch.setattr(background_jobs, "LOCK_FILE", tmp_path / "state.lock")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("LANG", "C")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-parent-secret-canary")
    monkeypatch.setenv("ARBITRARY_PARENT_CANARY", "plain-parent-canary")
    seen: dict[str, str] = {}

    def succeed(_command: list[str], **kwargs) -> SimpleNamespace:
        seen.update(kwargs["env"])
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(background_jobs.subprocess, "run", succeed)
    job = background_jobs.enqueue_job(
        name="content-correction-capture",
        module="example",
        args=[],
        env={
            "CHRONOVISOR_CONTENT_CORRECTION_ENABLED": "1",
            "ARBITRARY_JOB_CANARY": "plain-job-canary",
            "OPENAI_API_KEY": "sk-job-secret-canary",
        },
        stdin_text="{}",
    )

    result = background_jobs.run_job(job["job_id"])

    assert result["status"] == "completed"
    assert seen["HOME"] == str(tmp_path / "home")
    assert seen["LANG"] == "C"
    assert seen["CHRONOVISOR_CONTENT_CORRECTION_ENABLED"] == "1"
    assert (
        not {
            "ARBITRARY_JOB_CANARY",
            "ARBITRARY_PARENT_CANARY",
            "OPENAI_API_KEY",
        }
        & seen.keys()
    )
    assert job["env"] == {"CHRONOVISOR_CONTENT_CORRECTION_ENABLED": "1"}


def test_background_job_fixed_env_allowlist_matches_current_callers() -> None:
    assert frozenset(
        {
            "CHRONOVISOR_CONTENT_CORRECTION_ENABLED",
            "CHRONOVISOR_RECALL_ANSWER_CAPTURE_ENABLED",
            "CHRONOVISOR_RESEARCH_RUN_ID",
            "CLAUDE_CODE_CHRONOVISOR_RECORD_ENABLED",
            "CODEX_CHRONOVISOR_RECORD_ENABLED",
            "OLLAMA_CALIBRATION_FILE",
            "OLLAMA_HOST",
            "OLLAMA_RESOURCE_LOCK",
            "OLLAMA_URL",
        }
    ) == background_jobs._JOB_ENV_NAMES


def test_background_job_subprocess_error_does_not_leak_detail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    canary = "sk-subprocess-error-canary"
    monkeypatch.setattr(background_jobs, "JOB_DIR", tmp_path)
    monkeypatch.setattr(background_jobs, "STATE_FILE", tmp_path / "state.json")
    monkeypatch.setattr(background_jobs, "LOCK_FILE", tmp_path / "state.lock")

    def fail(*_args, **_kwargs):
        raise OSError(canary)

    monkeypatch.setattr(background_jobs.subprocess, "run", fail)
    job = background_jobs.enqueue_job(
        name="save", module="example", args=[], env={}, stdin_text="{}"
    )

    result = background_jobs.run_job(job["job_id"])
    output = capsys.readouterr().out
    stored = (tmp_path / "state.json").read_text(encoding="utf-8")

    assert result["output_tail"] == "OSError"
    assert canary not in output
    assert canary not in stored
    assert canary not in repr(result)


def test_background_job_load_preserves_durable_module_fields(
    tmp_path: Path, monkeypatch
) -> None:
    state_file = tmp_path / "state.json"
    monkeypatch.setattr(background_jobs, "STATE_FILE", state_file)
    state_file.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "jobs": {
                    "job-1": {
                        "job_id": "job-1",
                        "name": "self-heal",
                        "module": "chronovisor.self_heal",
                        "args": ["--packet", "/tmp/packet.json"],
                        "stdin": "",
                        "dedupe_key": "legacy",
                        "status": "queued",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    loaded = background_jobs._load()

    job = loaded["jobs"]["job-1"]
    assert job["module"] == "chronovisor.self_heal"
    assert job["dedupe_key"] == "legacy"


def test_successful_save_enqueues_audit_after_receipt_in_same_commit(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(background_jobs, "JOB_DIR", tmp_path)
    monkeypatch.setattr(background_jobs, "STATE_FILE", tmp_path / "state.json")
    monkeypatch.setattr(background_jobs, "LOCK_FILE", tmp_path / "state.lock")
    stdin_text = '{"session_id":"session-1","turn":2}'
    save = background_jobs.enqueue_job(
        name="codex-save",
        module="chronovisor.hosts.codex_record",
        args=["--hook", "--save"],
        env={},
        stdin_text=stdin_text,
        on_success=[
            {
                "name": "recall-audit-candidate",
                "module": "chronovisor.recall.recall_auditor",
                "args": ["--host", "codex", "--hook"],
                "env": {},
                "when_output_status": "saved",
            }
        ],
    )
    assert background_jobs._claim(save["job_id"])["status"] == "running"

    completed = background_jobs._finish(
        save["job_id"], exit_code=0, output='log line\n{"status":"saved"}'
    )
    state = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))

    assert completed["status"] == "completed"
    assert len(completed["followup_job_ids"]) == 1
    followup = state["jobs"][completed["followup_job_ids"][0]]
    assert followup["status"] == "queued"
    assert followup["name"] == "recall-audit-candidate"
    assert followup["stdin"] == stdin_text
    assert followup["parent_job_id"] == save["job_id"]

    background_jobs._finish(
        save["job_id"], exit_code=0, output='{"status":"saved"}'
    )
    state = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert len(state["jobs"]) == 2


@pytest.mark.parametrize("status", ["saved", "recovered"])
def test_save_receipt_output_is_forwarded_exactly_to_answer_capture(
    tmp_path: Path, monkeypatch, status: str
) -> None:
    monkeypatch.setattr(background_jobs, "JOB_DIR", tmp_path)
    monkeypatch.setattr(background_jobs, "STATE_FILE", tmp_path / "state.json")
    monkeypatch.setattr(background_jobs, "LOCK_FILE", tmp_path / "state.lock")
    save = background_jobs.enqueue_job(
        name="codex-save",
        module="chronovisor.hosts.codex_record",
        args=["--hook", "--save"],
        env={},
        stdin_text='{"stale":"hook-payload"}',
        on_success=[
            {
                "name": "recall-answer-capture",
                "module": "chronovisor.recall.recall_answer_eval",
                "args": ["--host", "codex", "--hook", "--capture-only"],
                "env": {},
                "when_output_statuses": ["saved", "recovered"],
                "stdin_from_output": True,
            }
        ],
    )
    background_jobs._claim(save["job_id"])
    receipt = json.dumps(
        {"status": status, "session_id": "session", "session_file": "/exact"},
        separators=(",", ":"),
    )

    completed = background_jobs._finish(
        save["job_id"], exit_code=0, output=f"log\n{receipt}\n"
    )
    state = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    followup = state["jobs"][completed["followup_job_ids"][0]]

    assert followup["stdin"] == receipt


def test_failed_save_does_not_enqueue_audit_candidate(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(background_jobs, "JOB_DIR", tmp_path)
    monkeypatch.setattr(background_jobs, "STATE_FILE", tmp_path / "state.json")
    monkeypatch.setattr(background_jobs, "LOCK_FILE", tmp_path / "state.lock")
    save = background_jobs.enqueue_job(
        name="codex-save",
        module="chronovisor.hosts.codex_record",
        args=["--hook", "--save"],
        env={},
        stdin_text='{"session_id":"session-1"}',
        on_success=[
            {
                "name": "recall-audit-candidate",
                "module": "chronovisor.recall.recall_auditor",
                "args": ["--host", "codex", "--hook"],
                "env": {},
                "when_output_status": "saved",
            }
        ],
    )
    background_jobs._claim(save["job_id"])

    result = background_jobs._finish(save["job_id"], exit_code=1, output="failed")
    state = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))

    assert result["status"] == "retry_wait"
    assert len(state["jobs"]) == 1


def test_successful_save_without_new_receipt_does_not_enqueue_audit(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(background_jobs, "JOB_DIR", tmp_path)
    monkeypatch.setattr(background_jobs, "STATE_FILE", tmp_path / "state.json")
    monkeypatch.setattr(background_jobs, "LOCK_FILE", tmp_path / "state.lock")
    save = background_jobs.enqueue_job(
        name="codex-save",
        module="chronovisor.hosts.codex_record",
        args=["--hook", "--save"],
        env={},
        stdin_text='{"session_id":"session-1"}',
        on_success=[
            {
                "name": "recall-audit-candidate",
                "module": "chronovisor.recall.recall_auditor",
                "args": ["--host", "codex", "--hook"],
                "env": {},
                "when_output_status": "saved",
            }
        ],
    )
    background_jobs._claim(save["job_id"])

    result = background_jobs._finish(
        save["job_id"],
        exit_code=0,
        output='{"status":"skipped","reason":"no new transcript records"}',
    )
    state = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))

    assert result["status"] == "completed"
    assert "followup_job_ids" not in result
    assert len(state["jobs"]) == 1


def test_coalesced_save_carries_receipt_until_latest_pass_finishes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(background_jobs, "JOB_DIR", tmp_path)
    monkeypatch.setattr(background_jobs, "STATE_FILE", tmp_path / "state.json")
    monkeypatch.setattr(background_jobs, "LOCK_FILE", tmp_path / "state.lock")
    first_payload = '{"session_id":"session-1","turn":1}'
    latest_payload = '{"session_id":"session-1","turn":2}'
    followup = [
        {
            "name": "recall-audit-candidate",
            "module": "chronovisor.recall.recall_auditor",
            "args": ["--host", "codex", "--hook"],
            "env": {},
            "when_output_status": "saved",
        }
    ]
    save = background_jobs.enqueue_job(
        name="codex-save",
        module="chronovisor.hosts.codex_record",
        args=["--hook", "--save"],
        env={},
        stdin_text=first_payload,
        on_success=followup,
    )
    background_jobs._claim(save["job_id"])
    coalesced = background_jobs.enqueue_job(
        name="codex-save",
        module="chronovisor.hosts.codex_record",
        args=["--hook", "--save"],
        env={},
        stdin_text=latest_payload,
        on_success=followup,
    )
    assert coalesced["rerun_requested"] is True

    rerun = background_jobs._finish(
        save["job_id"], exit_code=0, output='{"status":"saved"}'
    )
    assert rerun["status"] == "queued"
    assert "followup_job_ids" not in rerun
    background_jobs._claim(save["job_id"])

    completed = background_jobs._finish(
        save["job_id"],
        exit_code=0,
        output='{"status":"skipped","reason":"no new transcript records"}',
    )
    state = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))

    assert completed["status"] == "completed"
    assert len(completed["followup_job_ids"]) == 1
    audit = state["jobs"][completed["followup_job_ids"][0]]
    assert audit["stdin"] == latest_payload


def test_background_job_explicit_quarantine_exit_is_terminal(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(background_jobs, "JOB_DIR", tmp_path)
    monkeypatch.setattr(background_jobs, "STATE_FILE", tmp_path / "state.json")
    monkeypatch.setattr(background_jobs, "LOCK_FILE", tmp_path / "state.lock")
    monkeypatch.setattr(
        background_jobs.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=background_jobs.QUARANTINE_EXIT_CODE,
            stdout='{"status":"human_required"}',
            stderr="",
        ),
    )
    job = background_jobs.enqueue_job(
        name="self-heal", module="example", args=[], env={}, stdin_text=""
    )

    result = background_jobs.run_job(job["job_id"])

    assert result["status"] == "quarantined"
    assert result["next_retry_at"] is None


@pytest.mark.parametrize("claimed", [False, True], ids=("queued", "running"))
def test_background_job_exact_cancellation_is_terminal(
    tmp_path: Path,
    monkeypatch,
    claimed: bool,
) -> None:
    monkeypatch.setattr(background_jobs, "JOB_DIR", tmp_path)
    monkeypatch.setattr(background_jobs, "STATE_FILE", tmp_path / "state.json")
    monkeypatch.setattr(background_jobs, "LOCK_FILE", tmp_path / "state.lock")
    args = ["--packet", "/tmp/system-operational-one.json", "--enable-frontier-repair"]
    job = background_jobs.enqueue_job(
        name="system-code-repair",
        module="chronovisor.ops.self_heal",
        args=args,
        env={},
        stdin_text="sensitive payload",
    )
    if claimed:
        assert background_jobs._claim(job["job_id"])["status"] == "running"

    cancelled = background_jobs.cancel_matching_jobs(
        name="system-code-repair",
        module="chronovisor.ops.self_heal",
        args=args,
        reason="verified repair",
        stdin_text="sensitive payload",
    )

    assert cancelled["cancelled_job_ids"] == [job["job_id"]]
    assert background_jobs._claim(job["job_id"]) is None
    stored = json.loads((tmp_path / "state.json").read_text())["jobs"][job["job_id"]]
    assert stored["status"] == "cancelled"
    assert stored["cancellation_reason"] == "verified repair"
    assert stored["stdin"] == ""
    if claimed:
        finished = background_jobs._finish(job["job_id"], exit_code=0, output="cached")
        assert finished["status"] == "cancelled"
    stale_enqueue = background_jobs.enqueue_job(
        name="system-code-repair",
        module="chronovisor.ops.self_heal",
        args=args,
        env={},
        stdin_text="sensitive payload",
    )
    assert stale_enqueue["status"] == "cancelled"
    assert stale_enqueue["enqueued"] is False
    assert stale_enqueue["cancelled"] is True


def test_background_job_cancellation_tombstone_blocks_cancel_before_enqueue(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(background_jobs, "JOB_DIR", tmp_path)
    monkeypatch.setattr(background_jobs, "STATE_FILE", tmp_path / "state.json")
    monkeypatch.setattr(background_jobs, "LOCK_FILE", tmp_path / "state.lock")
    args = ["--packet", "/tmp/system-operational-one.json", "--enable-frontier-repair"]

    cancelled = background_jobs.cancel_matching_jobs(
        name="system-code-repair",
        module="chronovisor.ops.self_heal",
        args=args,
        reason="verified repair",
    )
    stale_enqueue = background_jobs.enqueue_job(
        name="system-code-repair",
        module="chronovisor.ops.self_heal",
        args=args,
        env={},
        stdin_text="",
    )

    assert cancelled["matched"] == 0
    assert cancelled["cancelled"] == 0
    assert cancelled["tombstoned"] is True
    assert stale_enqueue["status"] == "cancelled"
    assert stale_enqueue["enqueued"] is False
    assert stale_enqueue["cancelled"] is True
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["jobs"] == {}
    assert list(state["cancellation_tombstones"]) == [cancelled["dedupe_key"]]


def test_capture_jobs_coalesce_by_session_and_keep_latest_payload(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(background_jobs, "JOB_DIR", tmp_path)
    monkeypatch.setattr(background_jobs, "STATE_FILE", tmp_path / "state.json")
    monkeypatch.setattr(background_jobs, "LOCK_FILE", tmp_path / "state.lock")
    first_payload = json.dumps({"session_id": "session-1", "turn": 1})
    latest_payload = json.dumps({"session_id": "session-1", "turn": 2})

    first = background_jobs.enqueue_job(
        name="codex-save",
        module="chronovisor.hosts.codex_record",
        args=["--hook", "--save"],
        env={},
        stdin_text=first_payload,
    )
    second = background_jobs.enqueue_job(
        name="codex-save",
        module="chronovisor.hosts.codex_record",
        args=["--hook", "--save"],
        env={},
        stdin_text=latest_payload,
    )

    stored = json.loads((tmp_path / "state.json").read_text())
    assert second["job_id"] == first["job_id"]
    assert second["coalesced"] is True
    assert second["enqueued"] is False
    assert len(stored["jobs"]) == 1
    assert stored["jobs"][first["job_id"]]["stdin"] == latest_payload


def test_capture_jobs_without_session_identity_do_not_cross_coalesce(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(background_jobs, "JOB_DIR", tmp_path)
    monkeypatch.setattr(background_jobs, "STATE_FILE", tmp_path / "state.json")
    monkeypatch.setattr(background_jobs, "LOCK_FILE", tmp_path / "state.lock")

    first = background_jobs.enqueue_job(
        name="codex-save",
        module="chronovisor.hosts.codex_record",
        args=["--hook", "--save"],
        env={},
        stdin_text="{}",
    )
    second = background_jobs.enqueue_job(
        name="codex-save",
        module="chronovisor.hosts.codex_record",
        args=["--hook", "--save"],
        env={},
        stdin_text="{}",
    )

    assert first["job_id"] != second["job_id"]
    assert first["coalesced"] is False
    assert second["coalesced"] is False


def test_running_capture_coalesce_requests_one_tail_rerun(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(background_jobs, "JOB_DIR", tmp_path)
    monkeypatch.setattr(background_jobs, "STATE_FILE", tmp_path / "state.json")
    monkeypatch.setattr(background_jobs, "LOCK_FILE", tmp_path / "state.lock")
    first = background_jobs.enqueue_job(
        name="codex-save",
        module="chronovisor.hosts.codex_record",
        args=["--hook", "--save"],
        env={},
        stdin_text=json.dumps({"session_id": "session-1", "turn": 1}),
    )
    assert background_jobs._claim(first["job_id"])["status"] == "running"
    latest_payload = json.dumps({"session_id": "session-1", "turn": 2})

    coalesced = background_jobs.enqueue_job(
        name="codex-save",
        module="chronovisor.hosts.codex_record",
        args=["--hook", "--save"],
        env={},
        stdin_text=latest_payload,
    )
    rerun = background_jobs._finish(first["job_id"], exit_code=0, output="ok")

    assert coalesced["rerun_requested"] is True
    assert rerun["status"] == "queued"
    assert rerun["stdin"] == latest_payload
    claimed_again = background_jobs._claim(first["job_id"])
    assert claimed_again is not None
    completed = background_jobs._finish(first["job_id"], exit_code=0, output="ok")
    assert completed["status"] == "completed"
    assert completed["stdin"] == ""


def test_background_job_lane_is_single_flight(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(background_jobs, "JOB_DIR", tmp_path)
    monkeypatch.setattr(background_jobs, "STATE_FILE", tmp_path / "state.json")
    monkeypatch.setattr(background_jobs, "LOCK_FILE", tmp_path / "state.lock")
    first = background_jobs.enqueue_job(
        name="codex-save",
        module="chronovisor.hosts.codex_record",
        args=["--hook", "--save"],
        env={},
        stdin_text=json.dumps({"session_id": "session-1"}),
    )
    second = background_jobs.enqueue_job(
        name="codex-save",
        module="chronovisor.hosts.codex_record",
        args=["--hook", "--save"],
        env={},
        stdin_text=json.dumps({"session_id": "session-2"}),
    )

    assert background_jobs._claim(first["job_id"])["owner_pid"] == os.getpid()
    assert background_jobs._claim(second["job_id"]) is None
    background_jobs._finish(first["job_id"], exit_code=0, output="ok")
    assert background_jobs._claim(second["job_id"])["owner_pid"] == os.getpid()


def test_background_job_terminal_history_is_pruned(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(background_jobs, "JOB_DIR", tmp_path)
    monkeypatch.setattr(background_jobs, "STATE_FILE", tmp_path / "state.json")
    monkeypatch.setattr(background_jobs, "LOCK_FILE", tmp_path / "state.lock")
    monkeypatch.setattr(background_jobs, "MAX_TERMINAL_JOBS", 2)

    for index in range(4):
        job = background_jobs.enqueue_job(
            name="demo",
            module="example",
            args=[str(index)],
            env={},
            stdin_text=str(index),
        )
        background_jobs._claim(job["job_id"])
        background_jobs._finish(job["job_id"], exit_code=0, output="ok")

    stored = json.loads((tmp_path / "state.json").read_text())
    assert len(stored["jobs"]) == 2
    assert stored["pruned_terminal_total"] == 2


def test_background_job_snapshot_separates_recent_quarantines(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(background_jobs, "JOB_DIR", tmp_path)
    monkeypatch.setattr(background_jobs, "STATE_FILE", tmp_path / "state.json")
    monkeypatch.setattr(background_jobs, "LOCK_FILE", tmp_path / "state.lock")
    now = datetime.now(UTC)
    recent = (now - timedelta(hours=2)).isoformat(timespec="seconds")
    old = (now - timedelta(days=5)).isoformat(timespec="seconds")
    state = {
        "schema_version": 1,
        "jobs": {
            "old": {"status": "quarantined", "updated_at": old},
            "recent": {"status": "quarantined", "updated_at": recent},
            "done": {"status": "completed", "updated_at": recent},
            "queued": {"status": "queued", "created_at": recent},
        },
    }
    (tmp_path / "state.json").write_text(json.dumps(state), encoding="utf-8")

    result = background_jobs.snapshot()

    assert result["by_status"] == {
        "quarantined": 2,
        "completed": 1,
        "queued": 1,
    }
    assert result["quarantined_24h"] == 1
    assert result["latest_quarantined_at"] == recent
    assert result["oldest_pending_at"] == recent
