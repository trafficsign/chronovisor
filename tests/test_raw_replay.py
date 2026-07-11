from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from llm_wiki_mcp import raw_replay
from llm_wiki_mcp.convergence import CycleBudget
from llm_wiki_mcp.jobs import JobStatus


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _isolate_paths(tmp_path: Path, monkeypatch) -> dict[str, Path]:
    paths = {
        "raw": tmp_path / "raw",
        "queue": tmp_path / "review" / "raw-replay-queue.jsonl",
        "history": tmp_path / "runtime" / "raw-replay-history.jsonl",
        "completions": tmp_path / "runtime" / "raw-replay-completions.jsonl",
        "memory": tmp_path / "eval" / "memory-integrity-latest.json",
        "claims": tmp_path / "claims" / "claims.jsonl",
        "failure_log": tmp_path / "runtime" / "ingest-read-back-failures.jsonl",
        "runtime_status": tmp_path / "runtime" / "status.json",
        "packets": tmp_path / "runtime" / "failures" / "packets",
        "quarantine": tmp_path / "runtime" / "failures" / "quarantined-raw",
    }
    paths["raw"].mkdir(parents=True)
    monkeypatch.setattr(raw_replay, "RAW_DIR", paths["raw"])
    monkeypatch.setattr(raw_replay, "QUEUE_FILE", paths["queue"])
    monkeypatch.setattr(raw_replay, "HISTORY_FILE", paths["history"])
    monkeypatch.setattr(raw_replay, "COMPLETIONS_FILE", paths["completions"])
    monkeypatch.setattr(raw_replay, "MEMORY_INTEGRITY_FILE", paths["memory"])
    monkeypatch.setattr(raw_replay, "CLAIMS_FILE", paths["claims"])
    monkeypatch.setattr(raw_replay, "INGEST_FAILURE_LOG_FILE", paths["failure_log"])
    monkeypatch.setattr(raw_replay, "RUNTIME_STATUS_FILE", paths["runtime_status"])
    monkeypatch.setattr(raw_replay, "FAILURE_PACKETS_DIR", paths["packets"])
    monkeypatch.setattr(raw_replay, "QUARANTINED_RAW_DIR", paths["quarantine"])
    return paths


class _FakeJobStore:
    def __init__(self, *, result_status: JobStatus, error: str | None = None) -> None:
        self.result_status = result_status
        self.error = error
        self.jobs: dict[str, SimpleNamespace] = {}
        self.created = 0

    def create(self, processor: str) -> SimpleNamespace:
        self.created += 1
        job = SimpleNamespace(
            job_id=f"job-{self.created}",
            status=JobStatus.PENDING,
            processor=processor,
            pages_created=[],
            pages_updated=[],
            result=None,
            error=None,
        )
        self.jobs[job.job_id] = job
        return job

    def get(self, job_id: str) -> SimpleNamespace | None:
        return self.jobs.get(job_id)

    def finish(self, job_id: str) -> None:
        job = self.jobs[job_id]
        job.status = self.result_status
        job.error = self.error
        if self.result_status == JobStatus.COMPLETED:
            job.pages_created = ["created-page"]


def test_build_queue_selects_raws_by_date(tmp_path: Path, monkeypatch) -> None:
    paths = _isolate_paths(tmp_path, monkeypatch)
    (paths["raw"] / "20260701-codex-a.md").write_text("a", encoding="utf-8")
    (paths["raw"] / "20260706-codex-b.md").write_text("bb", encoding="utf-8")

    payload = raw_replay.build_queue(
        since="2026-07-05",
        path=paths["queue"],
        include_auto_signals=False,
    )

    assert payload["count"] == 1
    text = paths["queue"].read_text(encoding="utf-8")
    assert "20260706-codex-b.md" in text
    assert "20260701-codex-a.md" not in text


def test_select_raws_skips_retracted_before_limit_and_preserves_body(
    tmp_path: Path, monkeypatch
) -> None:
    paths = _isolate_paths(tmp_path, monkeypatch)
    retracted = paths["raw"] / "20260701-codex-retracted.md"
    retracted_text = (
        "---\n"
        "raw_status: retracted\n"
        "retraction_reason: incorrect_entity_merge\n"
        "---\n"
        "Original incorrect capture remains available for audit.\n"
    )
    retracted.write_text(retracted_text, encoding="utf-8")
    active = paths["raw"] / "20260702-codex-active.md"
    active.write_text("---\nraw_status: active\n---\nactive\n", encoding="utf-8")

    selected = raw_replay.select_raws(limit=1)

    assert selected == [active]
    assert retracted.read_text(encoding="utf-8") == retracted_text


def test_auto_signal_does_not_create_candidate_for_retracted_raw(
    tmp_path: Path, monkeypatch
) -> None:
    paths = _isolate_paths(tmp_path, monkeypatch)
    retracted = paths["raw"] / "20260701-codex-retracted.md"
    retracted.write_text(
        "---\nraw_status: retracted\n---\nincorrect capture\n",
        encoding="utf-8",
    )
    paths["memory"].parent.mkdir(parents=True)
    paths["memory"].write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "raw": retracted.name,
                        "path": str(retracted),
                        "status": "miss",
                        "query": "kuycon monitor",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = raw_replay.build_queue(
        path=paths["queue"],
        include_migration=False,
        include_auto_signals=True,
    )

    assert result["candidates"] == 0
    assert result["candidate_keys"] == []
    assert _read_jsonl(paths["queue"]) == []


def test_pending_queue_retires_raw_retracted_after_it_was_queued(
    tmp_path: Path, monkeypatch
) -> None:
    paths = _isolate_paths(tmp_path, monkeypatch)
    retracted = paths["raw"] / "20260701-codex-retracted.md"
    retracted_text = "---\nraw_status: retracted\n---\noriginal body\n"
    retracted.write_text(retracted_text, encoding="utf-8")
    _write_jsonl(
        paths["queue"],
        [
            {
                "key": raw_replay.stable_key(retracted),
                "raw": retracted.name,
                "path": str(retracted),
                "status": "pending",
                "attempts": 0,
                "sources": ["memory_integrity_miss"],
                "priority": 200,
                "bytes": retracted.stat().st_size,
                "date": "20260701",
            }
        ],
    )

    result = raw_replay.run_pending_queue(
        path=paths["queue"],
        history_file=paths["history"],
        claims_file=paths["claims"],
        completions_file=paths["completions"],
        max_runs=1,
    )

    assert result["count"] == 0
    [row] = _read_jsonl(paths["queue"])
    assert row["status"] == "not_needed"
    assert row["terminal_reason"] == "raw frontmatter marks capture as retracted"
    assert retracted.read_text(encoding="utf-8") == retracted_text


def test_build_queue_merges_signals_without_resetting_retry_state(tmp_path: Path, monkeypatch) -> None:
    paths = _isolate_paths(tmp_path, monkeypatch)
    raw = paths["raw"] / "20260706-codex-a.md"
    raw.write_text("body", encoding="utf-8")
    retry_at = "2026-07-11T12:00:00+00:00"
    _write_jsonl(
        paths["queue"],
        [
            {
                "key": raw_replay.stable_key(raw),
                "raw": raw.name,
                "path": str(raw),
                "status": "failed",
                "attempts": 2,
                "next_retry_at": retry_at,
                "sources": ["explicit_migration"],
                "priority": 200,
                "bytes": 4,
                "date": "20260706",
            }
        ],
    )
    paths["memory"].parent.mkdir(parents=True)
    paths["memory"].write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "raw": raw.name,
                        "path": str(raw),
                        "status": "miss",
                        "query": "missing memory",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    paths["packets"].mkdir(parents=True)
    (paths["packets"] / "failure.json").write_text(
        json.dumps(
            {
                "raw_file": raw.name,
                "status": "pending_local_repair",
                "failure_class": "triage.parse_failed",
            }
        ),
        encoding="utf-8",
    )

    raw_replay.build_queue(since="2026-07-01", path=paths["queue"])

    [row] = _read_jsonl(paths["queue"])
    assert row["key"] == f"raw:{raw.name}"
    assert row["sources"] == [
        "ingest_failure",
        "memory_integrity_miss",
        "explicit_migration",
    ]
    assert row["priority"] == 300
    assert row["status"] == "failed"
    assert row["attempts"] == 2
    assert row["next_retry_at"] == retry_at


def test_legacy_read_back_failure_resolves_raw_from_claim_time(tmp_path: Path, monkeypatch) -> None:
    paths = _isolate_paths(tmp_path, monkeypatch)
    earlier = paths["raw"] / "20260701-earlier.md"
    later = paths["raw"] / "20260701-later.md"
    earlier.write_text("earlier", encoding="utf-8")
    later.write_text("later", encoding="utf-8")
    _write_jsonl(
        paths["claims"],
        [
            {
                "source_page": "missing-page",
                "source_raw": earlier.name,
                "recorded_at": "2026-07-10T11:59:00+09:00",
            },
            {
                "source_page": "missing-page",
                "source_raw": later.name,
                "recorded_at": "2026-07-10T12:01:00+09:00",
            },
        ],
    )
    _write_jsonl(
        paths["failure_log"],
        [
            {
                "timestamp": "2026-07-10T12:00:00+0900",
                "failed": [{"page_id": "missing-page", "reason": "missing-metadata"}],
            }
        ],
    )

    result = raw_replay.build_queue(
        path=paths["queue"],
        include_migration=False,
        include_auto_signals=True,
    )

    assert result["candidate_keys"] == [raw_replay.stable_key(earlier)]
    [row] = _read_jsonl(paths["queue"])
    assert row["sources"] == ["ingest_failure"]
    assert row["priority"] == 300


def test_not_in_top_read_back_failure_is_owned_by_hint_repair_not_replay(
    tmp_path: Path, monkeypatch
) -> None:
    paths = _isolate_paths(tmp_path, monkeypatch)
    raw = paths["raw"] / "20260701-read-back.md"
    raw.write_text("body", encoding="utf-8")
    _write_jsonl(
        paths["claims"],
        [
            {
                "source_page": "page-a",
                "source_raw": raw.name,
                "recorded_at": "2026-07-10T11:59:00+09:00",
            }
        ],
    )
    _write_jsonl(
        paths["failure_log"],
        [
            {
                "timestamp": "2026-07-10T12:00:00+09:00",
                "failed": [{"page_id": "page-a", "reason": "not-in-top-results"}],
            }
        ],
    )

    result = raw_replay.build_queue(
        path=paths["queue"], include_migration=False, include_auto_signals=True
    )

    assert result["candidate_keys"] == []
    assert _read_jsonl(paths["queue"]) == []


def test_autonomous_refresh_retires_legacy_migration_until_explicitly_requested(
    tmp_path: Path,
    monkeypatch,
) -> None:
    paths = _isolate_paths(tmp_path, monkeypatch)
    raw = paths["raw"] / "20260701-legacy.md"
    raw.write_text("body", encoding="utf-8")
    _write_jsonl(
        paths["queue"],
        [
            {
                "raw": raw.name,
                "path": str(raw),
                "status": "pending",
                "attempts": 0,
                "sources": ["explicit_migration"],
                "priority": 200,
            }
        ],
    )

    autonomous = raw_replay.build_queue(
        path=paths["queue"],
        include_migration=False,
        include_auto_signals=True,
    )

    [retired] = _read_jsonl(paths["queue"])
    assert retired["status"] == "not_needed"
    assert retired["priority"] == 100
    assert autonomous["status_counts"] == {"not_needed": 1}

    explicit = raw_replay.build_queue(
        path=paths["queue"],
        limit=1,
        include_migration=True,
        include_auto_signals=False,
    )

    [reactivated] = _read_jsonl(paths["queue"])
    assert explicit["candidate_keys"] == [raw_replay.stable_key(raw)]
    assert reactivated["status"] == "pending"
    assert reactivated["terminal_reason"] is None


def test_autonomous_signal_keeps_matching_legacy_row_pending(tmp_path: Path, monkeypatch) -> None:
    paths = _isolate_paths(tmp_path, monkeypatch)
    legacy = paths["raw"] / "20260701-legacy.md"
    needed = paths["raw"] / "20260702-needed.md"
    legacy.write_text("legacy", encoding="utf-8")
    needed.write_text("needed", encoding="utf-8")
    _write_jsonl(
        paths["queue"],
        [
            {
                "raw": raw.name,
                "path": str(raw),
                "status": "pending",
                "sources": ["explicit_migration"],
                "priority": 200,
            }
            for raw in (legacy, needed)
        ],
    )
    paths["memory"].parent.mkdir(parents=True)
    paths["memory"].write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "raw": needed.name,
                        "path": str(needed),
                        "status": "miss",
                        "query": "still missing",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    raw_replay.build_queue(
        path=paths["queue"],
        include_migration=False,
        include_auto_signals=True,
    )

    rows = {row["raw"]: row for row in _read_jsonl(paths["queue"])}
    assert rows[legacy.name]["status"] == "not_needed"
    assert rows[needed.name]["status"] == "pending"
    assert rows[needed.name]["sources"] == ["memory_integrity_miss", "explicit_migration"]
    assert rows[needed.name]["priority"] == 200


def test_terminal_history_does_not_consume_migration_limit(tmp_path: Path, monkeypatch) -> None:
    paths = _isolate_paths(tmp_path, monkeypatch)
    completed = paths["raw"] / "20260701-completed.md"
    quarantined = paths["raw"] / "20260702-quarantined.md"
    pending = paths["raw"] / "20260703-pending.md"
    for raw in (completed, quarantined, pending):
        raw.write_text(raw.stem, encoding="utf-8")
    _write_jsonl(
        paths["history"],
        [
            {"raw": completed.name, "status": "completed", "attempts": 1},
            {"raw": quarantined.name, "status": "quarantined", "attempts": 3},
        ],
    )

    result = raw_replay.build_queue(
        path=paths["queue"],
        limit=1,
        include_auto_signals=False,
    )

    assert result["candidate_keys"] == [raw_replay.stable_key(pending)]
    rows = {row["raw"]: row for row in _read_jsonl(paths["queue"])}
    assert completed.name not in rows
    assert rows[quarantined.name]["status"] == "quarantined"
    assert rows[quarantined.name]["attempts"] == 3
    assert rows[pending.name]["status"] == "pending"


def test_nonhuman_quarantine_reopens_after_cooldown(tmp_path: Path, monkeypatch) -> None:
    paths = _isolate_paths(tmp_path, monkeypatch)
    raw = paths["raw"] / "20260702-quarantined.md"
    raw.write_text("body", encoding="utf-8")
    _write_jsonl(
        paths["queue"],
        [
            {
                "raw": raw.name,
                "path": str(raw),
                "status": "quarantined",
                "attempts": 3,
                "quarantined_at": "2000-01-01T00:00:00+00:00",
                "sources": ["explicit_migration"],
            }
        ],
    )
    monkeypatch.setenv("LLM_WIKI_CONVERGENCE_QUARANTINE_RETRY_SECONDS", "1")

    raw_replay.build_queue(
        path=paths["queue"],
        include_migration=False,
        include_auto_signals=False,
    )

    [row] = _read_jsonl(paths["queue"])
    assert row["status"] == "pending"
    assert row["attempts"] == 0
    assert row["quarantine_reopen_count"] == 1
    assert row["quarantine_resumed_at"]


def test_legacy_nonexternal_human_required_is_reclassified_for_frontier_retry(
    tmp_path: Path, monkeypatch
) -> None:
    paths = _isolate_paths(tmp_path, monkeypatch)
    raw = paths["raw"] / "20260702-legacy-human.md"
    raw.write_text("body", encoding="utf-8")
    _write_jsonl(
        paths["queue"],
        [
            {
                "raw": raw.name,
                "path": str(raw),
                "status": "human_required",
                "frontier_attempts": 3,
                "frontier_failure": {"failure_class": "frontier_tool_unavailable"},
                "sources": ["ingest_failure"],
            }
        ],
    )

    raw_replay.build_queue(
        path=paths["queue"],
        include_migration=False,
        include_auto_signals=False,
    )

    [row] = _read_jsonl(paths["queue"])
    assert row["status"] == "indeterminate"
    assert row["frontier_attempts"] == 0
    assert row["quarantine_reopen_count"] == 1


def test_external_authority_human_required_remains_terminal(
    tmp_path: Path, monkeypatch
) -> None:
    paths = _isolate_paths(tmp_path, monkeypatch)
    raw = paths["raw"] / "20260702-auth.md"
    raw.write_text("body", encoding="utf-8")
    _write_jsonl(
        paths["queue"],
        [
            {
                "raw": raw.name,
                "path": str(raw),
                "status": "human_required",
                "frontier_failure": {"failure_class": "oauth_required"},
                "sources": ["ingest_failure"],
            }
        ],
    )

    raw_replay.build_queue(
        path=paths["queue"],
        include_migration=False,
        include_auto_signals=False,
    )

    [row] = _read_jsonl(paths["queue"])
    assert row["status"] == "human_required"
    assert "quarantine_resumed_at" not in row


def test_failed_history_restores_attempts_and_backoff_after_queue_loss(tmp_path: Path, monkeypatch) -> None:
    paths = _isolate_paths(tmp_path, monkeypatch)
    raw = paths["raw"] / "20260701-failed.md"
    raw.write_text("body", encoding="utf-8")
    retry_at = "2026-07-10T12:00:00+00:00"
    _write_jsonl(
        paths["history"],
        [
            {
                "raw": raw.name,
                "status": "failed",
                "attempts": 2,
                "next_retry_at": retry_at,
                "error": "transient failure",
            }
        ],
    )

    raw_replay.build_queue(path=paths["queue"], include_auto_signals=False)

    [row] = _read_jsonl(paths["queue"])
    assert row["status"] == "failed"
    assert row["attempts"] == 2
    assert row["next_retry_at"] == retry_at
    preview = raw_replay.run_pending_queue(
        path=paths["queue"],
        history_file=paths["history"],
        claims_file=paths["claims"],
        max_runs=1,
        max_bytes=100,
        now=datetime(2026, 7, 10, 11, 59, tzinfo=timezone.utc),
        dry_run=True,
    )
    assert preview["count"] == 0


def test_duplicate_stable_keys_preserve_terminal_lifecycle(tmp_path: Path, monkeypatch) -> None:
    paths = _isolate_paths(tmp_path, monkeypatch)
    raw = paths["raw"] / "20260701-duplicate.md"
    raw.write_text("body", encoding="utf-8")
    base = {
        "raw": raw.name,
        "path": str(raw),
        "sources": ["explicit_migration"],
        "priority": 200,
    }
    _write_jsonl(
        paths["queue"],
        [
            {**base, "status": "completed", "attempts": 1, "completed_at": "2026-07-10T10:00:00+00:00"},
            {**base, "status": "pending", "attempts": 0},
        ],
    )

    result = raw_replay.build_queue(
        path=paths["queue"],
        include_migration=False,
        include_auto_signals=False,
    )

    assert result["count"] == 1
    [row] = _read_jsonl(paths["queue"])
    assert row["key"] == raw_replay.stable_key(raw)
    assert row["status"] == "completed"
    assert row["attempts"] == 1


def test_history_and_full_replay_claim_make_pending_rows_exact_once(tmp_path: Path, monkeypatch) -> None:
    paths = _isolate_paths(tmp_path, monkeypatch)
    a = paths["raw"] / "20260701-a.md"
    b = paths["raw"] / "20260702-b.md"
    a.write_text("a", encoding="utf-8")
    b.write_text("b", encoding="utf-8")
    raw_replay.build_queue(path=paths["queue"], include_auto_signals=False)
    _write_jsonl(paths["history"], [{"raw": a.name, "status": "completed"}])
    _write_jsonl(
        paths["claims"],
        [
            {
                "type": raw_replay.FULL_REPLAY_CLAIM_TYPE,
                "source_raw": f"replay:{b.name}",
                "status": "completed",
                "completion_scope": "full_raw",
            }
        ],
    )
    store = _FakeJobStore(result_status=JobStatus.COMPLETED)
    monkeypatch.setattr(raw_replay, "job_store", store)

    result = raw_replay.run_pending_queue(
        path=paths["queue"],
        history_file=paths["history"],
        claims_file=paths["claims"],
        max_runs=10,
        max_bytes=100,
    )

    assert result["count"] == 0
    assert store.created == 0
    rows = _read_jsonl(paths["queue"])
    assert {row["status"] for row in rows} == {"completed"}
    assert {row["completion_evidence"] for row in rows} == {
        "replay_history",
        "replay_completion_claim",
    }


def test_ordinary_replay_claim_after_crash_does_not_suppress_retry(tmp_path: Path, monkeypatch) -> None:
    paths = _isolate_paths(tmp_path, monkeypatch)
    raw = paths["raw"] / "20260701-crashed.md"
    raw.write_text("body", encoding="utf-8")
    raw_replay.build_queue(path=paths["queue"], include_auto_signals=False)
    _write_jsonl(
        paths["claims"],
        [
            {
                "source_raw": f"replay:{raw.name}",
                "source_page": "page-written-before-crash",
                "op": "ingest",
            }
        ],
    )
    store = _FakeJobStore(result_status=JobStatus.COMPLETED)
    monkeypatch.setattr(raw_replay, "job_store", store)
    monkeypatch.setattr(
        "llm_wiki_mcp.ingest.run_ingest",
        lambda _content, job_id, on_complete=None, metadata=None: store.finish(job_id),
    )

    result = raw_replay.run_pending_queue(
        path=paths["queue"],
        history_file=paths["history"],
        claims_file=paths["claims"],
        max_runs=1,
        max_bytes=100,
    )

    assert result["count"] == 1
    assert store.created == 1
    [row] = _read_jsonl(paths["queue"])
    assert row["status"] == "completed"


def test_partial_replay_is_terminal_and_does_not_repeat_successful_operations(tmp_path: Path, monkeypatch) -> None:
    paths = _isolate_paths(tmp_path, monkeypatch)
    raw = paths["raw"] / "20260701-partial.md"
    raw.write_text("body", encoding="utf-8")
    raw_replay.build_queue(path=paths["queue"], include_auto_signals=False)
    store = _FakeJobStore(result_status=JobStatus.COMPLETED)
    monkeypatch.setattr(raw_replay, "job_store", store)
    calls = 0

    def fake_ingest(_content, job_id, on_complete=None, metadata=None):
        nonlocal calls
        calls += 1
        store.finish(job_id)
        job = store.get(job_id)
        assert job is not None
        if calls == 1:
            job.result = {"failed_ops": [{"filename": "missing-page.md"}]}
            _write_jsonl(
                paths["claims"],
                [
                    {
                        "source_raw": metadata["source_raw"],
                        "source_page": "created-page",
                        "op": "ingest",
                    }
                ],
            )

    monkeypatch.setattr("llm_wiki_mcp.ingest.run_ingest", fake_ingest)
    started = datetime(2026, 7, 10, 10, 0, tzinfo=timezone.utc)

    partial = raw_replay.run_pending_queue(
        path=paths["queue"],
        history_file=paths["history"],
        claims_file=paths["claims"],
        max_runs=1,
        max_bytes=100,
        now=started,
    )
    next_cycle = raw_replay.run_pending_queue(
        path=paths["queue"],
        history_file=paths["history"],
        claims_file=paths["claims"],
        max_runs=1,
        max_bytes=100,
        now=started + timedelta(hours=1),
    )

    assert partial["runs"][0]["status"] == "completed_partial"
    assert next_cycle["runs"] == []
    assert calls == 1
    [row] = _read_jsonl(paths["queue"])
    assert row["status"] == "completed_partial"
    assert row["attempts"] == 1


def test_replay_persists_running_marker_before_ingest(tmp_path: Path, monkeypatch) -> None:
    paths = _isolate_paths(tmp_path, monkeypatch)
    raw = paths["raw"] / "20260701-running.md"
    raw.write_text("body", encoding="utf-8")
    raw_replay.build_queue(path=paths["queue"], include_auto_signals=False)
    store = _FakeJobStore(result_status=JobStatus.COMPLETED)
    monkeypatch.setattr(raw_replay, "job_store", store)

    def inspect_then_finish(_content, job_id, on_complete=None, metadata=None):
        [running] = _read_jsonl(paths["queue"])
        assert running["status"] == "running"
        assert running["job_id"] == job_id
        assert running["attempt_id"]
        assert len(running["raw_sha256"]) == 64
        store.finish(job_id)

    monkeypatch.setattr("llm_wiki_mcp.ingest.run_ingest", inspect_then_finish)

    result = raw_replay.run_pending_queue(
        path=paths["queue"],
        history_file=paths["history"],
        claims_file=paths["claims"],
        completions_file=paths["completions"],
        max_runs=1,
        max_bytes=100,
    )

    assert result["runs"][0]["status"] == "completed"
    assert _read_jsonl(paths["completions"])[0]["status"] == "completed"


def test_completion_callback_recovers_crash_before_queue_finalize(tmp_path: Path, monkeypatch) -> None:
    paths = _isolate_paths(tmp_path, monkeypatch)
    raw = paths["raw"] / "20260701-callback-crash.md"
    raw.write_text("body", encoding="utf-8")
    raw_replay.build_queue(path=paths["queue"], include_auto_signals=False)
    store = _FakeJobStore(result_status=JobStatus.COMPLETED)
    monkeypatch.setattr(raw_replay, "job_store", store)

    def crash_after_completion(_content, job_id, on_complete=None, metadata=None):
        store.finish(job_id)
        assert on_complete is not None
        on_complete()
        raise SystemExit("simulated process loss")

    monkeypatch.setattr("llm_wiki_mcp.ingest.run_ingest", crash_after_completion)
    with pytest.raises(SystemExit):
        raw_replay.run_pending_queue(
            path=paths["queue"],
            history_file=paths["history"],
            claims_file=paths["claims"],
            completions_file=paths["completions"],
            max_runs=1,
            max_bytes=100,
        )
    [running] = _read_jsonl(paths["queue"])
    assert running["status"] == "running"

    monkeypatch.setattr(
        "llm_wiki_mcp.ingest.run_ingest",
        lambda *_args, **_kwargs: pytest.fail("completed replay must not launch again"),
    )
    recovered = raw_replay.run_pending_queue(
        path=paths["queue"],
        history_file=paths["history"],
        claims_file=paths["claims"],
        completions_file=paths["completions"],
        max_runs=1,
        max_bytes=100,
    )

    assert recovered["runs"] == []
    [completed] = _read_jsonl(paths["queue"])
    assert completed["status"] == "completed"
    assert completed["completion_evidence"] == "replay_completion_journal"


def test_unknown_crashed_replay_is_frontier_quarantined_not_blindly_retried(
    tmp_path: Path, monkeypatch
) -> None:
    paths = _isolate_paths(tmp_path, monkeypatch)
    raw = paths["raw"] / "20260701-unknown-crash.md"
    raw.write_text("body", encoding="utf-8")
    raw_replay.build_queue(path=paths["queue"], include_auto_signals=False)
    store = _FakeJobStore(result_status=JobStatus.COMPLETED)
    monkeypatch.setattr(raw_replay, "job_store", store)
    monkeypatch.setattr(
        "llm_wiki_mcp.ingest.run_ingest",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(SystemExit("crash before apply proof")),
    )
    with pytest.raises(SystemExit):
        raw_replay.run_pending_queue(
            path=paths["queue"],
            history_file=paths["history"],
            claims_file=paths["claims"],
            completions_file=paths["completions"],
            max_runs=1,
            max_bytes=100,
        )
    monkeypatch.setattr(
        "llm_wiki_mcp.frontier_review.run_structured_review",
        lambda *_args, **_kwargs: {
            "decision": "quarantine",
            "confidence": 0.99,
            "reason": "cannot prove that replay is duplicate-safe",
        },
    )
    budget = CycleBudget(max_frontier_calls=1, max_mutations=1, max_raw_bytes=100)

    recovered = raw_replay.run_pending_queue(
        path=paths["queue"],
        history_file=paths["history"],
        claims_file=paths["claims"],
        completions_file=paths["completions"],
        max_runs=1,
        max_bytes=100,
        budget=budget,
    )

    assert recovered["runs"] == []
    assert recovered["frontier_reconciliation"]["reviewed"] == 1
    [quarantined] = _read_jsonl(paths["queue"])
    assert quarantined["status"] == "quarantined"
    assert store.created == 1


def test_safe_replay_decision_is_not_blocked_by_confidence_metadata(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from llm_wiki_mcp import frontier_review

    row = {
        "key": "raw-replay:test",
        "raw": "missing.md",
        "status": "indeterminate",
        "frontier_attempts": 0,
    }
    monkeypatch.setattr(
        frontier_review,
        "run_structured_review",
        lambda *_args, **_kwargs: {
            "decision": "safe_replay",
            "confidence": 0.01,
            "reason": "no mutation evidence exists",
        },
    )

    result = raw_replay._review_indeterminate_rows(
        [row],
        claims_file=tmp_path / "claims.jsonl",
        history_file=tmp_path / "history.jsonl",
        now=datetime(2026, 7, 11, tzinfo=timezone.utc),
        budget=None,
        retry_delay_seconds=60,
    )

    assert result["reviewed"] == 1
    assert row["status"] == "pending"
    assert row["frontier_decision"] == "safe_replay"


def test_completed_ingest_with_broken_completion_journal_never_becomes_retryable(
    tmp_path: Path, monkeypatch
) -> None:
    paths = _isolate_paths(tmp_path, monkeypatch)
    raw = paths["raw"] / "20260701-journal-error.md"
    raw.write_text("body", encoding="utf-8")
    raw_replay.build_queue(path=paths["queue"], include_auto_signals=False)
    store = _FakeJobStore(result_status=JobStatus.COMPLETED)
    monkeypatch.setattr(raw_replay, "job_store", store)
    monkeypatch.setattr(
        "llm_wiki_mcp.ingest.run_ingest",
        lambda _content, job_id, on_complete=None, metadata=None: store.finish(job_id),
    )
    monkeypatch.setattr(
        raw_replay,
        "_append_completion",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("disk unavailable")),
    )

    result = raw_replay.run_pending_queue(
        path=paths["queue"],
        history_file=paths["history"],
        claims_file=paths["claims"],
        completions_file=paths["completions"],
        max_runs=1,
        max_bytes=100,
    )

    assert result["runs"][0]["status"] == "indeterminate"
    [row] = _read_jsonl(paths["queue"])
    assert row["status"] == "indeterminate"
    assert row["next_retry_at"] is None


def test_run_replay_limit_zero_keeps_legacy_all_raws_behavior(tmp_path: Path, monkeypatch) -> None:
    paths = _isolate_paths(tmp_path, monkeypatch)
    for name in ("20260701-a.md", "20260702-b.md"):
        (paths["raw"] / name).write_text(name, encoding="utf-8")
    store = _FakeJobStore(result_status=JobStatus.COMPLETED)
    monkeypatch.setattr(raw_replay, "job_store", store)
    monkeypatch.setattr(
        "llm_wiki_mcp.ingest.run_ingest",
        lambda _content, job_id, on_complete=None, metadata=None: store.finish(job_id),
    )

    result = raw_replay.run_replay(limit=0)

    assert result["count"] == 2
    assert store.created == 2
    assert len(_read_jsonl(paths["history"])) == 2


def test_dry_run_is_fully_read_only(tmp_path: Path, monkeypatch) -> None:
    paths = _isolate_paths(tmp_path, monkeypatch)
    raw = paths["raw"] / "20260701-a.md"
    raw.write_text("body", encoding="utf-8")
    raw_replay.build_queue(path=paths["queue"], include_auto_signals=False)
    before = paths["queue"].read_bytes()
    lock_path = paths["queue"].with_suffix(".jsonl.lock")
    lock_before = lock_path.read_bytes()
    store = _FakeJobStore(result_status=JobStatus.COMPLETED)
    monkeypatch.setattr(raw_replay, "job_store", store)

    result = raw_replay.run_pending_queue(
        path=paths["queue"],
        history_file=paths["history"],
        claims_file=paths["claims"],
        max_runs=1,
        max_bytes=100,
        dry_run=True,
    )

    assert result["status"] == "dry_run"
    assert result["planned"][0]["raw"] == raw.name
    assert paths["queue"].read_bytes() == before
    assert not paths["history"].exists()
    assert lock_path.read_bytes() == lock_before
    assert store.created == 0

    preview_path = tmp_path / "preview" / "queue.jsonl"
    raw_replay.build_queue(
        path=preview_path,
        include_auto_signals=False,
        dry_run=True,
    )
    assert not preview_path.exists()
    assert not preview_path.with_suffix(".jsonl.lock").exists()


def test_run_pending_queue_respects_priority_run_and_byte_bounds(tmp_path: Path, monkeypatch) -> None:
    paths = _isolate_paths(tmp_path, monkeypatch)
    rows = []
    sources = ("ingest_failure", "memory_integrity_miss", "explicit_migration")
    for index, (priority, source) in enumerate(zip((300, 200, 100), sources), start=1):
        raw = paths["raw"] / f"2026070{index}-{priority}.md"
        raw.write_text("xx", encoding="utf-8")
        rows.append(
            {
                "key": raw_replay.stable_key(raw),
                "raw": raw.name,
                "path": str(raw),
                "date": f"2026070{index}",
                "bytes": 2,
                "priority": priority,
                "sources": [source],
                "status": "pending",
                "attempts": 0,
            }
        )
    _write_jsonl(paths["queue"], rows)
    store = _FakeJobStore(result_status=JobStatus.COMPLETED)
    monkeypatch.setattr(raw_replay, "job_store", store)
    monkeypatch.setattr(
        "llm_wiki_mcp.ingest.run_ingest",
        lambda _content, job_id, on_complete=None, metadata=None: store.finish(job_id),
    )

    result = raw_replay.run_pending_queue(
        path=paths["queue"],
        history_file=paths["history"],
        claims_file=paths["claims"],
        max_runs=2,
        max_bytes=4,
    )

    assert result["count"] == 2
    assert result["bytes"] == 4
    assert [run["raw"] for run in result["runs"]] == ["20260701-300.md", "20260702-200.md"]
    status_by_raw = {row["raw"]: row["status"] for row in _read_jsonl(paths["queue"])}
    assert status_by_raw["20260701-300.md"] == "completed"
    assert status_by_raw["20260702-200.md"] == "completed"
    assert status_by_raw["20260703-100.md"] == "pending"


def test_replay_ingest_cannot_restore_claim_removed_by_applied_correction(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from llm_wiki_mcp import index_store, ingest, page_mutation, wiki

    paths = _isolate_paths(tmp_path, monkeypatch)
    pages = tmp_path / "pages"
    pages.mkdir()
    page = pages / "display.md"
    page.write_text(
        "---\ntitle: Display\nsummary: Canonical display count\n"
        "recall_questions: [How many displays?]\nupdated: 2026-07-10\n---\n"
        "The setup has two G32P displays.\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(wiki, "PAGES_DIR", pages)
    monkeypatch.setattr(ingest, "PAGES_DIR", pages)
    monkeypatch.setattr(ingest, "INDEX_FILE", tmp_path / "index.md")
    monkeypatch.setattr(ingest, "LOG_FILE", tmp_path / "log.md")
    monkeypatch.setattr(page_mutation, "PAGES_DIR", pages)
    monkeypatch.setattr(
        page_mutation,
        "WIKI_MUTATION_LOCK",
        tmp_path / "runtime" / "wiki-mutation.lock",
    )

    class FakeIndex:
        def refresh(self) -> None:
            pass

        def all_pages_meta(self, include_system=True):
            return [{"page_id": "display"}]

        def all_page_ids(self, include_system=True):
            return {"display"}

        def all_tags(self, include_system=False):
            return set()

    monkeypatch.setattr(index_store, "get_store", lambda: FakeIndex())
    prepared = page_mutation.prepare_page_mutation(
        "display",
        [
            {
                "old_text": "The setup has two G32P displays.",
                "new_text": "The setup has one G32P display.",
            }
        ],
        correction_id="corr-replay-display",
    )
    assert page_mutation.apply_prepared_mutations([prepared])["status"] == "applied"

    raw = paths["raw"] / "20260701-stale-display.md"
    raw.write_text("The setup has two G32P displays.", encoding="utf-8")
    _write_jsonl(
        paths["queue"],
        [
            {
                "key": raw_replay.stable_key(raw),
                "raw": raw.name,
                "path": str(raw),
                "date": "20260701",
                "bytes": raw.stat().st_size,
                "priority": 100,
                "sources": ["explicit_migration"],
                "status": "pending",
                "attempts": 0,
            }
        ],
    )
    store = _FakeJobStore(result_status=JobStatus.COMPLETED)
    monkeypatch.setattr(raw_replay, "job_store", store)

    def replay_via_ingest(content, job_id, on_complete=None, metadata=None):
        _created, updated = ingest._apply_operations(
            [
                {
                    "type": "update",
                    "filename": "display.md",
                    "content": content,
                }
            ]
        )
        job = store.get(job_id)
        assert job is not None
        job.status = JobStatus.COMPLETED
        job.pages_updated = updated
        if on_complete is not None:
            on_complete()

    monkeypatch.setattr("llm_wiki_mcp.ingest.run_ingest", replay_via_ingest)

    result = raw_replay.run_pending_queue(
        path=paths["queue"],
        history_file=paths["history"],
        claims_file=paths["claims"],
        max_runs=1,
        max_bytes=1_000,
    )

    written = page.read_text(encoding="utf-8")
    assert result["runs"][0]["status"] == "completed"
    assert "two G32P displays" not in written
    assert written.count("one G32P display") >= 2
    assert "applied_corrections: [corr-replay-display]" in written


def test_eligibility_union_drains_current_keys_and_prior_auto_lane(
    tmp_path: Path,
    monkeypatch,
) -> None:
    paths = _isolate_paths(tmp_path, monkeypatch)
    auto = paths["raw"] / "20260701-auto.md"
    current = paths["raw"] / "20260702-current.md"
    unrelated = paths["raw"] / "20260703-unrelated.md"
    auto.write_text("auto", encoding="utf-8")
    current.write_text("current", encoding="utf-8")
    unrelated.write_text("unrelated", encoding="utf-8")
    _write_jsonl(
        paths["queue"],
        [
            {
                "raw": auto.name,
                "path": str(auto),
                "status": "pending",
                "sources": ["memory_integrity_miss"],
                "priority": 200,
            },
            {
                "raw": current.name,
                "path": str(current),
                "status": "pending",
                "sources": ["explicit_migration"],
                "priority": 100,
            },
            {
                "raw": unrelated.name,
                "path": str(unrelated),
                "status": "pending",
                "sources": ["explicit_migration"],
                "priority": 100,
            },
        ],
    )
    store = _FakeJobStore(result_status=JobStatus.COMPLETED)
    monkeypatch.setattr(raw_replay, "job_store", store)
    monkeypatch.setattr(
        "llm_wiki_mcp.ingest.run_ingest",
        lambda _content, job_id, on_complete=None, metadata=None: store.finish(job_id),
    )

    result = raw_replay.run_pending_queue(
        path=paths["queue"],
        history_file=paths["history"],
        claims_file=paths["claims"],
        max_runs=3,
        max_bytes=100,
        eligible_keys={raw_replay.stable_key(current)},
        eligible_sources=raw_replay.AUTO_SIGNAL_SOURCES,
    )

    assert [run["raw"] for run in result["runs"]] == [auto.name, current.name]
    rows = {row["raw"]: row for row in _read_jsonl(paths["queue"])}
    assert rows[auto.name]["status"] == "completed"
    assert rows[current.name]["status"] == "completed"
    assert rows[unrelated.name]["status"] == "pending"


def test_raw_larger_than_byte_budget_retries_then_quarantines(tmp_path: Path, monkeypatch) -> None:
    paths = _isolate_paths(tmp_path, monkeypatch)
    raw = paths["raw"] / "20260701-oversized.md"
    raw.write_bytes(b"x" * 2_000_001)
    raw_replay.build_queue(path=paths["queue"], include_auto_signals=False)
    store = _FakeJobStore(result_status=JobStatus.COMPLETED)
    monkeypatch.setattr(raw_replay, "job_store", store)
    started = datetime(2026, 7, 10, 10, 0, tzinfo=timezone.utc)

    preview = raw_replay.run_pending_queue(
        path=paths["queue"],
        history_file=paths["history"],
        claims_file=paths["claims"],
        max_runs=1,
        max_bytes=2_000_000,
        now=started,
        dry_run=True,
    )
    first = raw_replay.run_pending_queue(
        path=paths["queue"],
        history_file=paths["history"],
        claims_file=paths["claims"],
        max_runs=1,
        max_bytes=2_000_000,
        now=started,
    )
    second = raw_replay.run_pending_queue(
        path=paths["queue"],
        history_file=paths["history"],
        claims_file=paths["claims"],
        max_runs=1,
        max_bytes=2_000_000,
        now=started + timedelta(hours=1),
    )
    third = raw_replay.run_pending_queue(
        path=paths["queue"],
        history_file=paths["history"],
        claims_file=paths["claims"],
        max_runs=1,
        max_bytes=2_000_000,
        now=started + timedelta(hours=3),
    )

    assert preview["planned"][0]["action"] == "retry_oversized"
    assert preview["planned"][0]["charged_bytes"] == 0
    assert first["runs"][0]["job_status"] == "oversized"
    assert first["bytes"] == 0
    assert first["oversized"] == 1
    assert second["runs"][0]["status"] == "failed"
    assert third["runs"][0]["status"] == "quarantined"
    assert store.created == 0
    [row] = _read_jsonl(paths["queue"])
    assert row["attempts"] == 3
    assert row["status"] == "quarantined"
    assert len(_read_jsonl(paths["history"])) == 3


def test_three_failed_attempts_end_in_quarantine(tmp_path: Path, monkeypatch) -> None:
    paths = _isolate_paths(tmp_path, monkeypatch)
    raw = paths["raw"] / "20260701-a.md"
    raw.write_text("body", encoding="utf-8")
    raw_replay.build_queue(path=paths["queue"], include_auto_signals=False)
    store = _FakeJobStore(result_status=JobStatus.FAILED, error="model output invalid")
    monkeypatch.setattr(raw_replay, "job_store", store)
    monkeypatch.setattr(
        "llm_wiki_mcp.ingest.run_ingest",
        lambda _content, job_id, on_complete=None, metadata=None: store.finish(job_id),
    )
    started = datetime(2026, 7, 10, 10, 0, tzinfo=timezone.utc)

    first = raw_replay.run_pending_queue(
        path=paths["queue"],
        history_file=paths["history"],
        claims_file=paths["claims"],
        max_runs=1,
        max_bytes=100,
        now=started,
    )
    too_early = raw_replay.run_pending_queue(
        path=paths["queue"],
        history_file=paths["history"],
        claims_file=paths["claims"],
        max_runs=1,
        max_bytes=100,
        now=started + timedelta(minutes=59),
    )
    second = raw_replay.run_pending_queue(
        path=paths["queue"],
        history_file=paths["history"],
        claims_file=paths["claims"],
        max_runs=1,
        max_bytes=100,
        now=started + timedelta(hours=1),
    )
    third = raw_replay.run_pending_queue(
        path=paths["queue"],
        history_file=paths["history"],
        claims_file=paths["claims"],
        max_runs=1,
        max_bytes=100,
        now=started + timedelta(hours=3),
    )

    assert first["runs"][0]["status"] == "failed"
    assert too_early["count"] == 0
    assert second["runs"][0]["status"] == "failed"
    assert third["runs"][0]["status"] == "quarantined"
    [row] = _read_jsonl(paths["queue"])
    assert row["attempts"] == 3
    assert row["status"] == "quarantined"
    assert row["next_retry_at"] is None
    assert len(_read_jsonl(paths["history"])) == 3


def test_budget_defer_does_not_consume_replay_attempt(tmp_path: Path, monkeypatch) -> None:
    paths = _isolate_paths(tmp_path, monkeypatch)
    raw = paths["raw"] / "20260701-budget.md"
    raw.write_text("body", encoding="utf-8")
    raw_replay.build_queue(
        path=paths["queue"], include_migration=True, include_auto_signals=False
    )
    budget = CycleBudget(max_mutations=0, max_raw_bytes=100)

    result = raw_replay.run_pending_queue(
        path=paths["queue"],
        history_file=paths["history"],
        claims_file=paths["claims"],
        max_runs=1,
        max_bytes=100,
        budget=budget,
    )

    assert result["runs"] == []
    assert result["budget_deferred"][0]["reason"] == "mutation_budget_exhausted"
    [row] = _read_jsonl(paths["queue"])
    assert row["status"] == "pending"
    assert row["attempts"] == 0
    assert not paths["history"].exists()


def test_zero_byte_replay_still_requires_mutation_budget(tmp_path: Path, monkeypatch) -> None:
    paths = _isolate_paths(tmp_path, monkeypatch)
    raw = paths["raw"] / "20260701-empty.md"
    raw.write_bytes(b"")
    raw_replay.build_queue(path=paths["queue"], include_auto_signals=False)
    store = _FakeJobStore(result_status=JobStatus.COMPLETED)
    monkeypatch.setattr(raw_replay, "job_store", store)
    budget = CycleBudget(max_mutations=0, max_raw_bytes=0)

    result = raw_replay.run_pending_queue(
        path=paths["queue"],
        history_file=paths["history"],
        claims_file=paths["claims"],
        completions_file=paths["completions"],
        max_runs=1,
        max_bytes=0,
        budget=budget,
    )

    assert result["runs"] == []
    assert result["budget_deferred"][0]["reason"] == "mutation_budget_exhausted"
    assert store.created == 0
    [row] = _read_jsonl(paths["queue"])
    assert row["status"] == "pending"
    assert row["attempts"] == 0


def test_mixed_legacy_readback_only_sources_retire_without_raw_replay(
    tmp_path: Path, monkeypatch
) -> None:
    paths = _isolate_paths(tmp_path, monkeypatch)
    raw = paths["raw"] / "20260701-readback-only.md"
    raw.write_text("body", encoding="utf-8")
    _write_jsonl(
        paths["queue"],
        [
            {
                "key": raw_replay.stable_key(raw),
                "raw": raw.name,
                "path": str(raw),
                "status": "pending",
                "attempts": 0,
                "sources": ["ingest_failure", "explicit_migration"],
                "reasons": [
                    "ingest read-back failure: not-in-top-results",
                    "explicit migration since all",
                ],
                "priority": 300,
                "bytes": raw.stat().st_size,
                "date": "20260701",
            }
        ],
    )

    result = raw_replay.build_queue(
        path=paths["queue"],
        include_migration=False,
        include_auto_signals=True,
    )

    assert result["candidate_keys"] == []
    [row] = _read_jsonl(paths["queue"])
    assert row["status"] == "not_needed"
