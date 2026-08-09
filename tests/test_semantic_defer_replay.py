from __future__ import annotations

import hashlib
import json
from contextlib import nullcontext
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from chronovisor.core.jobs import JobStatus
from chronovisor.ingest import raw_replay
from chronovisor.ingest.convergence import CycleBudget


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _isolate_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    from chronovisor.ingest import failure_supervisor

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
    monkeypatch.setattr(failure_supervisor, "reset_raw_failure", lambda _raw: None)
    return paths


def test_auto_signals_skip_semantic_defer_until_authority_artifact_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chronovisor.core import runtime_config, store
    from chronovisor.decision import decision_router
    from chronovisor.ingest import failure_supervisor
    from chronovisor.ops import runtime_status

    paths = _isolate_paths(tmp_path, monkeypatch)
    monkeypatch.setattr(store, "CHRONOVISOR_ROOT", tmp_path)
    monkeypatch.setattr(store, "RAW_DIR", paths["raw"])
    monkeypatch.setattr(runtime_status, "RUNTIME_DIR", tmp_path / "runtime")
    monkeypatch.setattr(
        runtime_status,
        "STATUS_FILE",
        tmp_path / "runtime" / "status.json",
    )
    monkeypatch.setattr(
        runtime_status,
        "EVENTS_FILE",
        tmp_path / "runtime" / "events.jsonl",
    )
    monkeypatch.setattr(
        runtime_status,
        "METRICS_FILE",
        tmp_path / "runtime" / "metrics.jsonl",
    )
    raw = paths["raw"] / "20260714-semantic-split.md"
    raw.write_text("immutable source", encoding="utf-8")

    paths["memory"].parent.mkdir(parents=True)
    paths["memory"].write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "raw": raw.name,
                        "path": str(raw),
                        "status": "miss",
                        "query": "semantic split",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    paths["runtime_status"].parent.mkdir(parents=True, exist_ok=True)
    paths["runtime_status"].write_text(
        json.dumps(
            {
                "last_success": {
                    "source_raw": raw.name,
                    "failed_ops": [{"filename": "memory/split.md"}],
                }
            }
        ),
        encoding="utf-8",
    )

    authority_artifact = tmp_path / "adopted-router.json"
    authority_artifact.write_bytes(b"authority epoch one")
    monkeypatch.setattr(
        runtime_config,
        "load_decision_router_config",
        lambda: SimpleNamespace(adoption_artifact=str(authority_artifact)),
    )
    monkeypatch.setattr(
        decision_router,
        "resolve_router_policy",
        lambda config: SimpleNamespace(
            source="adopted_artifact",
            error=None,
            artifact_sha256=hashlib.sha256(
                Path(config.adoption_artifact).read_bytes()
            ).hexdigest(),
        ),
    )
    deferred_authority_sha256 = hashlib.sha256(
        authority_artifact.read_bytes()
    ).hexdigest()
    supervision = failure_supervisor.record_raw_failure(
        raw_path=raw,
        error=(
            "local consensus semantic no quorum "
            f"[authority_sha256={deferred_authority_sha256}]: "
            "local_models_did_not_reach_two_vote_quorum"
        ),
    )
    assert supervision.terminal_deferred is True

    held = raw_replay.build_queue(
        path=paths["queue"],
        include_migration=True,
        include_auto_signals=True,
    )

    assert held["candidates"] == 0
    assert held["candidate_keys"] == []
    assert held["skipped_semantic_deferred"] == 1
    assert paths["queue"].read_text(encoding="utf-8") == ""

    authority_artifact.write_bytes(b"authority epoch two")
    released = raw_replay.build_queue(
        path=paths["queue"],
        include_migration=True,
        include_auto_signals=True,
    )

    assert released["candidates"] == 1
    assert released["candidate_keys"] == [raw_replay.stable_key(raw)]
    assert released["skipped_semantic_deferred"] == 0


@pytest.mark.parametrize(
    ("queue_status", "attempts", "extra_fields"),
    [
        ("pending", 0, {}),
        (
            "quarantined",
            3,
            {"quarantined_at": "2000-01-01T00:00:00+00:00"},
        ),
    ],
    ids=("pending", "cooldown-quarantined"),
)
def test_existing_queue_row_does_not_cooldown_reopen_or_launch_while_deferred(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    queue_status: str,
    attempts: int,
    extra_fields: dict[str, str],
) -> None:
    paths = _isolate_paths(tmp_path, monkeypatch)
    raw = paths["raw"] / "20260714-semantic-split.md"
    raw.write_text("immutable source", encoding="utf-8")
    _write_jsonl(
        paths["queue"],
        [
            {
                "key": raw_replay.stable_key(raw),
                "raw": raw.name,
                "path": str(raw),
                "status": queue_status,
                "attempts": attempts,
                "sources": ["ingest_failure"],
                "priority": 300,
                "bytes": raw.stat().st_size,
                "date": "20260714",
                **extra_fields,
            }
        ],
    )
    deferred = {raw.name}
    monkeypatch.setattr(
        raw_replay,
        "_active_terminal_semantic_deferred_raw_names",
        lambda: frozenset(deferred),
    )
    monkeypatch.setenv("CHRONOVISOR_CONVERGENCE_QUARANTINE_RETRY_SECONDS", "0")
    monkeypatch.setattr(
        'chronovisor.ingest.ingest.run_ingest',
        lambda *_args, **_kwargs: pytest.fail("semantic defer must not launch ingest"),
    )

    held = raw_replay.run_pending_queue(
        path=paths["queue"],
        history_file=paths["history"],
        claims_file=paths["claims"],
        completions_file=paths["completions"],
        max_runs=1,
        max_bytes=100,
        now=datetime(2026, 7, 14, tzinfo=UTC),
    )

    assert held["count"] == 0
    [held_row] = [
        json.loads(line)
        for line in paths["queue"].read_text(encoding="utf-8").splitlines()
    ]
    assert held_row["status"] == queue_status
    assert "quarantine_resumed_at" not in held_row

    deferred.clear()
    released = raw_replay.run_pending_queue(
        path=paths["queue"],
        history_file=paths["history"],
        claims_file=paths["claims"],
        completions_file=paths["completions"],
        max_runs=1,
        max_bytes=100,
        dry_run=True,
        now=datetime(2026, 7, 14, tzinfo=UTC),
    )

    assert released["count"] == 1
    assert released["planned"][0]["raw"] == raw.name


def test_semantic_defer_skips_indeterminate_frontier_reconciliation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _isolate_paths(tmp_path, monkeypatch)
    raw = paths["raw"] / "20260714-indeterminate-split.md"
    raw.write_text("immutable source", encoding="utf-8")
    _write_jsonl(
        paths["queue"],
        [
            {
                "key": raw_replay.stable_key(raw),
                "raw": raw.name,
                "path": str(raw),
                "status": "indeterminate",
                "attempts": 1,
                "sources": ["ingest_failure"],
                "priority": 300,
                "bytes": raw.stat().st_size,
                "date": "20260714",
            }
        ],
    )
    monkeypatch.setattr(
        raw_replay,
        "_active_terminal_semantic_deferred_raw_names",
        lambda: frozenset({raw.name}),
    )

    result = raw_replay.run_pending_queue(
        path=paths["queue"],
        history_file=paths["history"],
        claims_file=paths["claims"],
        completions_file=paths["completions"],
        max_runs=1,
        max_bytes=100,
        frontier_reviewer=lambda *_args, **_kwargs: pytest.fail(
            "semantic defer must not invoke frontier reconciliation"
        ),
    )

    assert result["count"] == 0
    assert result["frontier_reconciliation"]["reviewed"] == 0


def test_semantic_defer_published_during_budget_preflight_blocks_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _isolate_paths(tmp_path, monkeypatch)
    raw = paths["raw"] / "20260714-budget-race.md"
    raw.write_text("immutable source", encoding="utf-8")
    key = raw_replay.stable_key(raw)
    _write_jsonl(
        paths["queue"],
        [
            {
                "key": key,
                "raw": raw.name,
                "path": str(raw),
                "status": "pending",
                "attempts": 0,
                "sources": ["ingest_failure"],
                "priority": 300,
                "bytes": raw.stat().st_size,
                "date": "20260714",
            }
        ],
    )
    deferred: set[str] = set()
    monkeypatch.setattr(
        raw_replay,
        "_active_terminal_semantic_deferred_raw_names",
        lambda: frozenset(deferred),
    )
    consume_calls: list[tuple[str, int]] = []

    class PublishingBudget:
        def can_consume(self, kind: str, amount: int = 1) -> tuple[bool, str]:
            if kind == "raw_bytes":
                deferred.add(raw.name)
            return True, "ok"

        def consume(self, kind: str, amount: int = 1) -> tuple[bool, str]:
            consume_calls.append((kind, amount))
            return True, "ok"

    monkeypatch.setattr(
        raw_replay,
        "job_store",
        SimpleNamespace(
            create=lambda **_kwargs: pytest.fail(
                "semantic defer must not create a replay job"
            )
        ),
    )
    monkeypatch.setattr(
        raw_replay,
        "_run_candidate",
        lambda *_args, **_kwargs: pytest.fail("semantic defer must not invoke replay"),
    )

    result = raw_replay.run_pending_queue(
        path=paths["queue"],
        history_file=paths["history"],
        claims_file=paths["claims"],
        completions_file=paths["completions"],
        max_runs=1,
        max_bytes=100,
        budget=PublishingBudget(),
        now=datetime(2026, 7, 14, tzinfo=UTC),
    )

    assert result["count"] == 0
    assert result["budget_deferred"] == []
    assert result["semantic_deferred"] == [
        {"key": key, "raw": raw.name, "reason": "semantic_no_quorum"}
    ]
    assert consume_calls == []
    [queue_row] = [
        json.loads(line)
        for line in paths["queue"].read_text(encoding="utf-8").splitlines()
    ]
    assert queue_row["status"] == "pending"
    assert "job_id" not in queue_row


def test_no_quorum_from_replay_publishes_one_terminal_defer_without_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chronovisor.ingest import failure_supervisor

    paths = _isolate_paths(tmp_path, monkeypatch)
    raw = paths["raw"] / "20260714-replay-semantic-split.md"
    content = "immutable replay source"
    raw.write_text(content, encoding="utf-8")
    key = raw_replay.stable_key(raw)
    _write_jsonl(
        paths["queue"],
        [
            {
                "key": key,
                "raw": raw.name,
                "path": str(raw),
                "status": "pending",
                "attempts": 0,
                "sources": ["ingest_failure"],
                "priority": 300,
                "bytes": raw.stat().st_size,
                "date": "20260714",
            }
        ],
    )
    authority_sha256 = "a" * 64
    error = (
        "local consensus semantic no quorum "
        f"[authority_sha256={authority_sha256}]: conservative vote vetoed"
    )
    deferred: dict[str, str] = {}
    monkeypatch.setattr(
        raw_replay,
        "_active_terminal_semantic_deferred_raw_names",
        lambda: frozenset(deferred),
    )
    monkeypatch.setattr(
        raw_replay,
        "_active_operational_deferred_raw_statuses",
        lambda: dict(deferred),
    )
    monkeypatch.setattr(raw_replay, "decision_authority_lock", nullcontext)
    monkeypatch.setenv("CHRONOVISOR_CONVERGENCE_QUARANTINE_RETRY_SECONDS", "0")
    current_authority_sha256 = [authority_sha256]
    monkeypatch.setattr(
        failure_supervisor,
        "_current_adopted_authority_sha256",
        lambda: current_authority_sha256[0],
    )
    publications: list[dict[str, object]] = []

    def publish(**kwargs: object) -> SimpleNamespace:
        publications.append(dict(kwargs))
        deferred[raw.name] = "semantic_no_quorum"
        return SimpleNamespace(
            terminal_deferred=True,
            packet_path=str(paths["packets"] / "semantic.json"),
        )

    monkeypatch.setattr(
        failure_supervisor,
        "record_semantic_no_quorum_defer_unless_operational_hold",
        publish,
    )

    class Store:
        def __init__(self) -> None:
            self.created = 0
            self.jobs: dict[str, SimpleNamespace] = {}

        def create(self, processor: str) -> SimpleNamespace:
            self.created += 1
            job = SimpleNamespace(
                job_id=f"job-{self.created}",
                processor=processor,
                status=JobStatus.PENDING,
                error=None,
                result=None,
                pages_created=[],
                pages_updated=[],
            )
            self.jobs[job.job_id] = job
            return job

        def get(self, job_id: str) -> SimpleNamespace | None:
            return self.jobs.get(job_id)

        def update(self, job_id: str, **kwargs: object) -> None:
            for field, value in kwargs.items():
                setattr(self.jobs[job_id], field, value)

    store = Store()
    monkeypatch.setattr(raw_replay, "job_store", store)
    ingest_calls: list[str] = []
    ingest_error = [error]

    def fail_with_semantic_split(
        raw_text: str,
        job_id: str,
        *,
        on_complete,
        metadata: dict[str, str],
    ) -> None:
        del on_complete
        ingest_calls.append(str(metadata["source_raw"]))
        assert raw_text == content
        store.update(job_id, status=JobStatus.FAILED, error=ingest_error[0])

    monkeypatch.setattr(
        'chronovisor.ingest.ingest.run_ingest',
        fail_with_semantic_split,
    )

    def forbid_frontier(*_args, **_kwargs):
        pytest.fail("terminal semantic defer must not invoke frontier")

    first = raw_replay.run_pending_queue(
        path=paths["queue"],
        history_file=paths["history"],
        claims_file=paths["claims"],
        completions_file=paths["completions"],
        max_runs=1,
        max_bytes=100,
        retry_delay_seconds=0,
        frontier_reviewer=forbid_frontier,
        now=datetime(2026, 7, 14, tzinfo=UTC),
    )
    second = raw_replay.run_pending_queue(
        path=paths["queue"],
        history_file=paths["history"],
        claims_file=paths["claims"],
        completions_file=paths["completions"],
        max_runs=1,
        max_bytes=100,
        retry_delay_seconds=0,
        frontier_reviewer=forbid_frontier,
        now=datetime(2026, 7, 14, tzinfo=UTC),
    )

    assert ingest_calls == [f"replay:{raw.name}"]
    assert len(publications) == 1
    assert publications[0]["raw_path"] == raw
    assert publications[0]["raw_text"] == content
    assert publications[0]["error"] == error
    assert publications[0]["job_id"] == "job-1"
    assert publications[0]["related_raw_paths"] == ()
    assert first["count"] == 0
    assert first["runs"] == []
    assert first["frontier_reconciliation"]["reviewed"] == 0
    assert first["semantic_deferred"] == [
        {"key": key, "raw": raw.name, "reason": "semantic_no_quorum"}
    ]
    assert second["count"] == 0
    assert second["frontier_reconciliation"]["reviewed"] == 0
    [row] = [json.loads(line) for line in paths["queue"].read_text().splitlines()]
    assert row["status"] == "pending"
    assert row["attempts"] == 0
    assert row["last_error"] == error
    assert row["terminal_reason"] == "semantic_no_quorum"
    assert row["recovery_kind"] == "semantic_no_quorum_terminal_defer"
    assert row["semantic_defer_authority_sha256"] == authority_sha256
    assert row["semantic_defer_prior_attempts"] == 1
    assert row["semantic_defer_job_id"] == "job-1"
    assert row["semantic_defer_attempt_id"].startswith("job-1:1:")
    assert row["semantic_defer_started_at"]
    assert row["job_id"] is None
    assert "attempt_id" not in row
    assert "started_at" not in row
    assert row["frontier_attempts"] == 0

    # A different validated authority releases the supervisor hold. The next
    # attempt must not carry old semantic-defer lifecycle metadata into its
    # independent failure result.
    deferred.clear()
    current_authority_sha256[0] = "d" * 64
    ingest_error[0] = "ordinary transient replay failure"
    third = raw_replay.run_pending_queue(
        path=paths["queue"],
        history_file=paths["history"],
        claims_file=paths["claims"],
        completions_file=paths["completions"],
        max_runs=1,
        max_bytes=100,
        retry_delay_seconds=0,
        frontier_reviewer=forbid_frontier,
        now=datetime(2026, 7, 14, 0, 0, 1, tzinfo=UTC),
    )

    assert third["count"] == 1
    assert third["runs"][0]["status"] == "failed"
    assert ingest_calls == [f"replay:{raw.name}", f"replay:{raw.name}"]
    [released_row] = _read_jsonl(paths["queue"])
    assert released_row["status"] == "failed"
    assert released_row["attempts"] == 1
    assert released_row["reactivated_at"]
    assert "recovery_kind" not in released_row
    assert not any(key.startswith("semantic_defer_") for key in released_row)
    assert "semantic_deferred_at" not in released_row


@pytest.mark.parametrize("limit_axis", ["runs", "bytes"])
def test_legacy_no_quorum_reconcile_is_model_free_bounded_and_scope_exact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    limit_axis: str,
) -> None:
    from chronovisor.ingest import failure_supervisor

    paths = _isolate_paths(tmp_path, monkeypatch)
    authority_sha256 = "b" * 64
    raws = [
        paths["raw"] / "20260714-in-scope.md",
        paths["raw"] / "20260714-out-of-scope.md",
    ]
    rows: list[dict[str, object]] = []
    for index, raw in enumerate(raws, start=1):
        raw.write_text(f"immutable source {index}", encoding="utf-8")
        rows.append(
            {
                "key": raw_replay.stable_key(raw),
                "raw": raw.name,
                "path": str(raw),
                "status": "quarantined",
                "attempts": 3,
                "sources": ["ingest_failure"],
                "priority": 300,
                "bytes": raw.stat().st_size,
                "date": "20260714",
                "job_id": f"legacy-job-{index}",
                "attempt_id": f"legacy-attempt-{index}",
                "started_at": "2026-07-14T00:00:00+00:00",
                "raw_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
                "last_error": (
                    "local consensus semantic no quorum "
                    f"[authority_sha256={authority_sha256}]: split"
                ),
                "quarantined_at": "2026-07-14T00:01:00+00:00",
            }
        )
    out_indeterminate = paths["raw"] / "20260714-out-indeterminate.md"
    out_indeterminate.write_text("out of scope indeterminate", encoding="utf-8")
    rows.append(
        {
            "key": raw_replay.stable_key(out_indeterminate),
            "raw": out_indeterminate.name,
            "path": str(out_indeterminate),
            "status": "indeterminate",
            "attempts": 1,
            "frontier_attempts": 1,
            "next_frontier_retry_at": "2000-01-01T00:00:00+00:00",
            "sources": ["ingest_failure"],
            "priority": 300,
            "bytes": out_indeterminate.stat().st_size,
            "date": "20260714",
            "job_id": "out-indeterminate-job",
        }
    )
    out_running = paths["raw"] / "20260714-out-running.md"
    out_running.write_text("out of scope running", encoding="utf-8")
    out_running_sha256 = hashlib.sha256(out_running.read_bytes()).hexdigest()
    rows.append(
        {
            "key": raw_replay.stable_key(out_running),
            "raw": out_running.name,
            "path": str(out_running),
            "status": "running",
            "attempts": 1,
            "sources": ["ingest_failure"],
            "priority": 300,
            "bytes": out_running.stat().st_size,
            "date": "20260714",
            "job_id": "out-running-job",
            "attempt_id": f"out-running-job:1:{out_running_sha256[:16]}",
            "started_at": "2026-07-14T00:00:00+00:00",
            "raw_sha256": out_running_sha256,
        }
    )
    in_scope_pending = paths["raw"] / "20260714-z-in-scope-pending.md"
    in_scope_pending.write_text("p", encoding="utf-8")
    rows.append(
        {
            "key": raw_replay.stable_key(in_scope_pending),
            "raw": in_scope_pending.name,
            "path": str(in_scope_pending),
            "status": "pending",
            "attempts": 0,
            "sources": ["ingest_failure"],
            "priority": 300,
            "bytes": in_scope_pending.stat().st_size,
            "date": "20260714",
        }
    )
    _write_jsonl(paths["queue"], rows)
    _write_jsonl(paths["history"], rows)
    deferred: dict[str, str] = {}
    monkeypatch.setattr(
        raw_replay,
        "_active_terminal_semantic_deferred_raw_names",
        lambda: frozenset(deferred),
    )
    monkeypatch.setattr(
        raw_replay,
        "_active_operational_deferred_raw_statuses",
        lambda: dict(deferred),
    )
    monkeypatch.setattr(raw_replay, "decision_authority_lock", nullcontext)
    monkeypatch.setenv("CHRONOVISOR_CONVERGENCE_QUARANTINE_RETRY_SECONDS", "0")
    monkeypatch.setattr(
        failure_supervisor,
        "_current_adopted_authority_sha256",
        lambda: authority_sha256,
    )
    publications: list[str] = []

    def publish(**kwargs: object) -> SimpleNamespace:
        raw_path = kwargs["raw_path"]
        assert isinstance(raw_path, Path)
        publications.append(raw_path.name)
        deferred[raw_path.name] = "semantic_no_quorum"
        return SimpleNamespace(
            terminal_deferred=True,
            packet_path=str(paths["packets"] / f"{raw_path.stem}.json"),
        )

    monkeypatch.setattr(
        failure_supervisor,
        "record_semantic_no_quorum_defer_unless_operational_hold",
        publish,
    )
    monkeypatch.setattr(
        raw_replay,
        "job_store",
        SimpleNamespace(
            create=lambda **_kwargs: pytest.fail("reconcile must be model-free")
        ),
    )
    monkeypatch.setattr(
        'chronovisor.ingest.ingest.run_ingest',
        lambda *_args, **_kwargs: pytest.fail("reconcile must not ingest"),
    )

    result = raw_replay.run_pending_queue(
        path=paths["queue"],
        history_file=paths["history"],
        claims_file=paths["claims"],
        completions_file=paths["completions"],
        max_runs=1 if limit_axis == "runs" else 2,
        max_bytes=100 if limit_axis == "runs" else raws[0].stat().st_size,
        eligible_keys={
            raw_replay.stable_key(raws[0]),
            raw_replay.stable_key(in_scope_pending),
        },
        eligible_sources=None,
        frontier_reviewer=lambda *_args, **_kwargs: pytest.fail(
            "reconcile must not invoke frontier"
        ),
        now=datetime(2026, 7, 14, tzinfo=UTC),
    )

    assert publications == [raws[0].name]
    assert result["count"] == 0
    assert result["frontier_reconciliation"]["reviewed"] == 0
    assert result["semantic_defer_reconciled"] == [
        {
            "key": raw_replay.stable_key(raws[0]),
            "raw": raws[0].name,
            "prior_status": "quarantined",
            "prior_attempts": 3,
            "reason": "semantic_no_quorum",
            "authority_sha256": authority_sha256,
            "packet_path": str(paths["packets"] / f"{raws[0].stem}.json"),
            "bytes": raws[0].stat().st_size,
            "charged_bytes": raws[0].stat().st_size,
        }
    ]
    by_raw = {row["raw"]: row for row in _read_jsonl(paths["queue"])}
    migrated = by_raw[raws[0].name]
    untouched = by_raw[raws[1].name]
    assert migrated["status"] == "pending"
    assert migrated["attempts"] == 0
    assert migrated["semantic_defer_prior_attempts"] == 3
    assert migrated["semantic_defer_job_id"] == "legacy-job-1"
    assert migrated["semantic_defer_attempt_id"] == "legacy-attempt-1"
    assert migrated["job_id"] is None
    assert "attempt_id" not in migrated
    assert "started_at" not in migrated
    assert untouched["status"] == "quarantined"
    assert untouched["attempts"] == 3
    assert untouched["job_id"] == "legacy-job-2"
    assert "semantic_defer_authority_sha256" not in untouched
    assert by_raw[in_scope_pending.name]["status"] == "pending"
    assert by_raw[in_scope_pending.name]["attempts"] == 0
    untouched_indeterminate = by_raw[out_indeterminate.name]
    assert untouched_indeterminate["status"] == "indeterminate"
    assert untouched_indeterminate["frontier_attempts"] == 1
    assert untouched_indeterminate["job_id"] == "out-indeterminate-job"
    untouched_running = by_raw[out_running.name]
    assert untouched_running["status"] == "running"
    assert untouched_running["attempts"] == 1
    assert untouched_running["job_id"] == "out-running-job"
    assert untouched_running["attempt_id"].startswith("out-running-job:1:")


def test_active_semantic_packet_recovers_running_marker_after_publish_crash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _isolate_paths(tmp_path, monkeypatch)
    raw = paths["raw"] / "20260714-packet-published-before-queue-commit.md"
    raw.write_text("immutable crash source", encoding="utf-8")
    key = raw_replay.stable_key(raw)
    raw_sha256 = hashlib.sha256(raw.read_bytes()).hexdigest()
    _write_jsonl(
        paths["queue"],
        [
            {
                "key": key,
                "raw": raw.name,
                "path": str(raw),
                "status": "running",
                "attempts": 1,
                "sources": ["ingest_failure"],
                "priority": 300,
                "bytes": raw.stat().st_size,
                "date": "20260714",
                "job_id": "crashed-job",
                "attempt_id": f"crashed-job:1:{raw_sha256[:16]}",
                "started_at": "2026-07-14T00:00:00+00:00",
                "raw_sha256": raw_sha256,
            }
        ],
    )
    authority_sha256 = "c" * 64
    packet_path = paths["packets"] / "durable-semantic-packet.json"
    deferred = {raw.name: "semantic_no_quorum"}
    monkeypatch.setattr(
        raw_replay,
        "_active_terminal_semantic_deferred_raw_names",
        lambda: frozenset(deferred),
    )
    monkeypatch.setattr(
        raw_replay,
        "_active_operational_deferred_raw_statuses",
        lambda: dict(deferred),
    )

    def packet_evidence(
        row: dict[str, object], *, active_raws: frozenset[str]
    ) -> dict[str, object] | None:
        assert row["raw"] == raw.name
        assert active_raws == frozenset({raw.name})
        return {
            "reason": "semantic_no_quorum",
            "authority_sha256": authority_sha256,
            "packet_path": str(packet_path),
            "error": (
                "local consensus semantic no quorum "
                f"[authority_sha256={authority_sha256}]: split"
            ),
            "job_id": "crashed-job",
        }

    monkeypatch.setattr(
        raw_replay,
        "_active_semantic_defer_packet_evidence",
        packet_evidence,
    )
    monkeypatch.setattr(
        raw_replay,
        "job_store",
        SimpleNamespace(
            create=lambda **_kwargs: pytest.fail("crash recovery must not ingest")
        ),
    )

    result = raw_replay.run_pending_queue(
        path=paths["queue"],
        history_file=paths["history"],
        claims_file=paths["claims"],
        completions_file=paths["completions"],
        max_runs=1,
        max_bytes=100,
        eligible_keys={key},
        frontier_reviewer=lambda *_args, **_kwargs: pytest.fail(
            "crash recovery must not invoke frontier"
        ),
        now=datetime(2026, 7, 14, tzinfo=UTC),
    )

    assert result["count"] == 0
    assert result["reconciled"] == []
    assert result["frontier_reconciliation"]["reviewed"] == 0
    assert result["semantic_defer_reconciled"][0]["prior_status"] == "running"
    [row] = _read_jsonl(paths["queue"])
    assert row["status"] == "pending"
    assert row["attempts"] == 0
    assert row["recovery_kind"] == "semantic_no_quorum_terminal_defer"
    assert row["semantic_defer_job_id"] == "crashed-job"
    assert row["semantic_defer_attempt_id"].startswith("crashed-job:1:")
    assert row["job_id"] is None
    assert "attempt_id" not in row
    assert "started_at" not in row


def test_newer_operational_hold_outranks_legacy_no_quorum_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chronovisor.ingest import failure_supervisor

    paths = _isolate_paths(tmp_path, monkeypatch)
    raw = paths["raw"] / "20260714-operational-hold-wins.md"
    raw.write_text("immutable operational source", encoding="utf-8")
    authority_sha256 = "e" * 64
    _write_jsonl(
        paths["queue"],
        [
            {
                "key": raw_replay.stable_key(raw),
                "raw": raw.name,
                "path": str(raw),
                "status": "quarantined",
                "attempts": 3,
                "sources": ["ingest_failure"],
                "priority": 300,
                "bytes": raw.stat().st_size,
                "date": "20260714",
                "raw_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
                "last_error": (
                    "local consensus semantic no quorum "
                    f"[authority_sha256={authority_sha256}]: stale split"
                ),
                "quarantined_at": "2000-01-01T00:00:00+00:00",
            }
        ],
    )
    deferred = {raw.name: "pending_local_repair"}
    monkeypatch.setattr(
        raw_replay,
        "_active_terminal_semantic_deferred_raw_names",
        lambda: frozenset(deferred),
    )
    monkeypatch.setattr(
        raw_replay,
        "_active_operational_deferred_raw_statuses",
        lambda: dict(deferred),
    )
    monkeypatch.setattr(
        failure_supervisor,
        "record_semantic_no_quorum_defer_unless_operational_hold",
        lambda **_kwargs: pytest.fail(
            "operational hold must not be replaced by semantic history"
        ),
    )
    monkeypatch.setattr(
        'chronovisor.ingest.ingest.run_ingest',
        lambda *_args, **_kwargs: pytest.fail("operational hold must not ingest"),
    )

    result = raw_replay.run_pending_queue(
        path=paths["queue"],
        history_file=paths["history"],
        claims_file=paths["claims"],
        completions_file=paths["completions"],
        max_runs=1,
        max_bytes=100,
        frontier_reviewer=lambda *_args, **_kwargs: pytest.fail(
            "operational hold must not invoke frontier"
        ),
        now=datetime(2026, 7, 14, tzinfo=UTC),
    )

    assert result["count"] == 0
    assert result["semantic_defer_reconciled"] == []
    assert result["semantic_deferred"] == []
    assert result["operational_deferred"] == [
        {
            "key": raw_replay.stable_key(raw),
            "raw": raw.name,
            "reason": "pending_local_repair",
        }
    ]
    [row] = _read_jsonl(paths["queue"])
    assert row["status"] == "quarantined"
    assert row["attempts"] == 3
    assert "semantic_defer_authority_sha256" not in row


@pytest.mark.parametrize("exhausted", ["raw_bytes", "mutation", "elapsed"])
def test_legacy_semantic_reconcile_obeys_shared_cycle_budget_before_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    exhausted: str,
) -> None:
    from chronovisor.ingest import failure_supervisor

    paths = _isolate_paths(tmp_path, monkeypatch)
    raw = paths["raw"] / f"20260714-budget-{exhausted}.md"
    raw.write_text("immutable budget-bound semantic source", encoding="utf-8")
    authority_sha256 = "f" * 64
    key = raw_replay.stable_key(raw)
    _write_jsonl(
        paths["queue"],
        [
            {
                "schema_version": 2,
                "key": key,
                "type": "raw_replay_candidate",
                "raw": raw.name,
                "path": str(raw),
                "status": "quarantined",
                "attempts": 3,
                "sources": ["ingest_failure"],
                "reasons": ["legacy semantic split"],
                "priority": 300,
                "bytes": raw.stat().st_size,
                "date": "20260714",
                "created_at": "2026-07-14T00:00:00+00:00",
                "updated_at": "2026-07-14T00:01:00+00:00",
                "last_attempt_at": "2026-07-14T00:01:00+00:00",
                "completed_at": None,
                "next_retry_at": None,
                "raw_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
                "last_error": (
                    "local consensus semantic no quorum "
                    f"[authority_sha256={authority_sha256}]: budget split"
                ),
                "quarantined_at": "2026-07-14T00:01:00+00:00",
            }
        ],
    )
    queue_before_bytes = paths["queue"].read_bytes()
    queue_before = _read_jsonl(paths["queue"])[0]
    monkeypatch.setattr(
        raw_replay,
        "_active_terminal_semantic_deferred_raw_names",
        lambda: frozenset(),
    )
    monkeypatch.setattr(
        raw_replay,
        "_active_operational_deferred_raw_statuses",
        lambda: {},
    )
    monkeypatch.setattr(
        failure_supervisor,
        "_current_adopted_authority_sha256",
        lambda: authority_sha256,
    )
    monkeypatch.setattr(
        failure_supervisor,
        "record_semantic_no_quorum_defer_unless_operational_hold",
        lambda **_kwargs: pytest.fail("budget denial must precede packet publication"),
    )
    monkeypatch.setattr(
        raw_replay,
        "job_store",
        SimpleNamespace(
            create=lambda **_kwargs: pytest.fail("budget denial must not create a job")
        ),
    )
    monkeypatch.setattr(
        'chronovisor.ingest.ingest.run_ingest',
        lambda *_args, **_kwargs: pytest.fail("budget denial must not ingest"),
    )
    budget = CycleBudget(
        max_local_calls=1,
        max_frontier_calls=0,
        max_mutations=0 if exhausted == "mutation" else 1,
        max_raw_bytes=0 if exhausted == "raw_bytes" else raw.stat().st_size,
        max_elapsed_seconds=0 if exhausted == "elapsed" else 60,
    )

    result = raw_replay.run_pending_queue(
        path=paths["queue"],
        history_file=paths["history"],
        claims_file=paths["claims"],
        completions_file=paths["completions"],
        max_runs=1,
        max_bytes=100,
        eligible_keys={key},
        eligible_sources=None,
        budget=budget,
        frontier_reviewer=lambda *_args, **_kwargs: pytest.fail(
            "budget denial must not invoke frontier"
        ),
        now=datetime(2026, 7, 14, tzinfo=UTC),
    )

    assert result["count"] == 0
    assert result["semantic_defer_reconciled"] == []
    assert result["budget_deferred"] == [
        {
            "key": key,
            "raw": raw.name,
            "reason": f"{exhausted}_budget_exhausted",
            "action": "semantic_defer_reconcile",
        }
    ]
    assert budget.snapshot()["used"] == {
        "local": 0,
        "frontier": 0,
        "mutation": 0,
        "raw_bytes": 0,
    }
    queue_after = _read_jsonl(paths["queue"])[0]
    for field in (
        "status",
        "attempts",
        "last_error",
        "quarantined_at",
        "raw_sha256",
    ):
        assert queue_after[field] == queue_before[field]
    assert "recovery_kind" not in queue_after
    assert not any(field.startswith("semantic_defer_") for field in queue_after)
    assert paths["queue"].read_bytes() == queue_before_bytes
    assert not paths["packets"].exists()


def test_dry_run_previews_future_and_cooldown_legacy_semantic_reconcile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chronovisor.ingest import failure_supervisor

    paths = _isolate_paths(tmp_path, monkeypatch)
    authority_sha256 = "9" * 64
    rows: list[dict[str, object]] = []
    for index, status in enumerate(("failed", "quarantined"), start=1):
        raw = paths["raw"] / f"20260714-dry-semantic-{index}.md"
        raw.write_text(f"immutable dry source {index}", encoding="utf-8")
        rows.append(
            {
                "key": raw_replay.stable_key(raw),
                "raw": raw.name,
                "path": str(raw),
                "status": status,
                "attempts": index,
                "sources": ["ingest_failure"],
                "priority": 300,
                "bytes": raw.stat().st_size,
                "date": "20260714",
                "raw_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
                "last_error": (
                    "local consensus semantic no quorum "
                    f"[authority_sha256={authority_sha256}]: dry split {index}"
                ),
                "next_retry_at": (
                    "2099-01-01T00:00:00+00:00" if status == "failed" else None
                ),
                "quarantined_at": (
                    "2026-07-14T00:00:00+00:00" if status == "quarantined" else None
                ),
            }
        )
    _write_jsonl(paths["queue"], rows)
    queue_before = paths["queue"].read_bytes()
    monkeypatch.setenv("CHRONOVISOR_CONVERGENCE_QUARANTINE_RETRY_SECONDS", "86400")
    monkeypatch.setattr(
        raw_replay,
        "_active_terminal_semantic_deferred_raw_names",
        lambda: frozenset(),
    )
    monkeypatch.setattr(
        raw_replay,
        "_active_operational_deferred_raw_statuses",
        lambda: {},
    )
    monkeypatch.setattr(
        failure_supervisor,
        "_current_adopted_authority_sha256",
        lambda: authority_sha256,
    )
    monkeypatch.setattr(
        failure_supervisor,
        "record_semantic_no_quorum_defer_unless_operational_hold",
        lambda **_kwargs: pytest.fail("dry-run must not publish a packet"),
    )

    result = raw_replay.run_pending_queue(
        path=paths["queue"],
        history_file=paths["history"],
        claims_file=paths["claims"],
        completions_file=paths["completions"],
        max_runs=2,
        max_bytes=100,
        dry_run=True,
        eligible_keys={str(row["key"]) for row in rows},
        eligible_sources=None,
        now=datetime(2026, 7, 14, tzinfo=UTC),
    )

    assert result["count"] == 2
    assert [row["action"] for row in result["planned"]] == [
        "semantic_defer_reconcile",
        "semantic_defer_reconcile",
    ]
    assert result["semantic_defer_planned"] == result["planned"]
    assert [row["attempts"] for row in result["planned"]] == [1, 2]
    assert paths["queue"].read_bytes() == queue_before
    assert not paths["queue"].with_suffix(".jsonl.lock").exists()
    assert not paths["packets"].exists()


def test_oversized_legacy_semantic_reconcile_is_read_only_and_fully_uncharged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chronovisor.ingest import failure_supervisor

    paths = _isolate_paths(tmp_path, monkeypatch)
    raw = paths["raw"] / "20260714-oversized-semantic.md"
    raw.write_bytes(b"x" * 101)
    authority_sha256 = "8" * 64
    key = raw_replay.stable_key(raw)
    _write_jsonl(
        paths["queue"],
        [
            {
                "schema_version": 2,
                "key": key,
                "type": "raw_replay_candidate",
                "raw": raw.name,
                "path": str(raw),
                "status": "quarantined",
                "attempts": 3,
                "sources": ["ingest_failure"],
                "reasons": ["semantic split"],
                "priority": 300,
                "bytes": raw.stat().st_size,
                "date": "20260714",
                "created_at": "2026-07-14T00:00:00+00:00",
                "updated_at": "2026-07-14T00:01:00+00:00",
                "last_attempt_at": "2026-07-14T00:01:00+00:00",
                "next_retry_at": None,
                "raw_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
                "last_error": (
                    "local consensus semantic no quorum "
                    f"[authority_sha256={authority_sha256}]: oversized split"
                ),
                "quarantined_at": "2026-07-14T00:01:00+00:00",
            }
        ],
    )
    queue_before = paths["queue"].read_bytes()
    monkeypatch.setattr(
        raw_replay, "_active_terminal_semantic_deferred_raw_names", lambda: frozenset()
    )
    monkeypatch.setattr(
        raw_replay, "_active_operational_deferred_raw_statuses", lambda: {}
    )
    monkeypatch.setattr(
        failure_supervisor,
        "_current_adopted_authority_sha256",
        lambda: authority_sha256,
    )
    monkeypatch.setattr(
        raw_replay,
        "_sha256_path",
        lambda _path: pytest.fail("public byte denial must precede raw hashing"),
    )
    monkeypatch.setattr(
        failure_supervisor,
        "record_semantic_no_quorum_defer_unless_operational_hold",
        lambda **_kwargs: pytest.fail("public byte denial must not publish"),
    )
    budget = CycleBudget(
        max_local_calls=1,
        max_frontier_calls=0,
        max_mutations=1,
        max_raw_bytes=1_000,
        max_elapsed_seconds=60,
    )

    result = raw_replay.run_pending_queue(
        path=paths["queue"],
        history_file=paths["history"],
        claims_file=paths["claims"],
        completions_file=paths["completions"],
        max_runs=1,
        max_bytes=100,
        eligible_keys={key},
        eligible_sources=None,
        budget=budget,
        now=datetime(2026, 7, 14, tzinfo=UTC),
    )

    assert result["count"] == 0
    assert result["semantic_defer_reconciled"] == []
    assert result["budget_deferred"] == [
        {
            "key": key,
            "raw": raw.name,
            "reason": "raw_replay_byte_budget_exhausted",
            "action": "semantic_defer_reconcile",
        }
    ]
    assert budget.snapshot()["used"] == {
        "local": 0,
        "frontier": 0,
        "mutation": 0,
        "raw_bytes": 0,
    }
    assert paths["queue"].read_bytes() == queue_before
    assert not paths["packets"].exists()


def test_dry_run_verifies_only_bounded_semantic_candidates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chronovisor.ingest import failure_supervisor

    paths = _isolate_paths(tmp_path, monkeypatch)
    authority_sha256 = "7" * 64
    rows: list[dict[str, object]] = []
    for index in range(2):
        raw = paths["raw"] / f"20260714-bounded-preview-{index}.md"
        raw.write_text(f"bounded source {index}", encoding="utf-8")
        rows.append(
            {
                "key": raw_replay.stable_key(raw),
                "raw": raw.name,
                "path": str(raw),
                "status": "quarantined",
                "attempts": 3,
                "sources": ["ingest_failure"],
                "priority": 300 - index,
                "bytes": raw.stat().st_size,
                "date": "20260714",
                "raw_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
                "last_error": (
                    "local consensus semantic no quorum "
                    f"[authority_sha256={authority_sha256}]: split {index}"
                ),
                "quarantined_at": "2026-07-14T00:00:00+00:00",
            }
        )
    _write_jsonl(paths["queue"], rows)
    monkeypatch.setattr(
        raw_replay, "_active_terminal_semantic_deferred_raw_names", lambda: frozenset()
    )
    monkeypatch.setattr(
        raw_replay, "_active_operational_deferred_raw_statuses", lambda: {}
    )
    monkeypatch.setattr(
        failure_supervisor,
        "_current_adopted_authority_sha256",
        lambda: authority_sha256,
    )
    verified: list[str] = []

    def preview(row, *, path: Path, error: str):
        verified.append(path.name)
        return {
            "reason": "semantic_no_quorum",
            "authority_sha256": authority_sha256,
            "packet_path": None,
            "error": error,
            "published": False,
        }

    monkeypatch.setattr(raw_replay, "_preview_semantic_no_quorum_defer", preview)

    result = raw_replay.run_pending_queue(
        path=paths["queue"],
        history_file=paths["history"],
        claims_file=paths["claims"],
        completions_file=paths["completions"],
        max_runs=1,
        max_bytes=100,
        dry_run=True,
        eligible_keys={str(row["key"]) for row in rows},
        eligible_sources=None,
        now=datetime(2026, 7, 14, tzinfo=UTC),
    )

    assert result["count"] == 1
    assert verified == [str(rows[0]["raw"])]
    assert result["planned"][0]["key"] == rows[0]["key"]


def test_active_packet_authority_change_between_plan_and_apply_is_not_deferred(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _isolate_paths(tmp_path, monkeypatch)
    raw = paths["raw"] / "20260714-authority-race.md"
    raw.write_text("authority race source", encoding="utf-8")
    key = raw_replay.stable_key(raw)
    _write_jsonl(
        paths["queue"],
        [
            {
                "key": key,
                "raw": raw.name,
                "path": str(raw),
                "status": "quarantined",
                "attempts": 3,
                "sources": ["ingest_failure"],
                "priority": 300,
                "bytes": raw.stat().st_size,
                "date": "20260714",
                "raw_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
                "quarantined_at": "2026-07-14T00:00:00+00:00",
            }
        ],
    )
    status_calls = 0

    def statuses() -> dict[str, str]:
        nonlocal status_calls
        status_calls += 1
        return {raw.name: "semantic_no_quorum"} if status_calls == 1 else {}

    monkeypatch.setattr(
        raw_replay, "_active_operational_deferred_raw_statuses", statuses
    )
    monkeypatch.setattr(
        raw_replay, "_active_terminal_semantic_deferred_raw_names", lambda: frozenset()
    )
    monkeypatch.setattr(
        raw_replay,
        "_active_semantic_defer_packet_evidence",
        lambda *_args, **_kwargs: pytest.fail(
            "released packet must not be verified or applied"
        ),
    )
    monkeypatch.setenv("CHRONOVISOR_CONVERGENCE_QUARANTINE_RETRY_SECONDS", "86400")
    monkeypatch.setattr(
        'chronovisor.ingest.ingest.run_ingest',
        lambda *_args, **_kwargs: pytest.fail("cooldown row must not ingest"),
    )

    result = raw_replay.run_pending_queue(
        path=paths["queue"],
        history_file=paths["history"],
        claims_file=paths["claims"],
        completions_file=paths["completions"],
        max_runs=1,
        max_bytes=100,
        eligible_keys={key},
        eligible_sources=None,
        now=datetime(2026, 7, 14, tzinfo=UTC),
    )

    assert result["semantic_defer_reconciled"] == []
    [queue_row] = _read_jsonl(paths["queue"])
    assert queue_row["status"] == "quarantined"
    assert "recovery_kind" not in queue_row


def test_dry_run_reports_shared_budget_denial_without_source_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chronovisor.ingest import failure_supervisor

    paths = _isolate_paths(tmp_path, monkeypatch)
    raw = paths["raw"] / "20260714-dry-budget-denied.md"
    raw.write_text("dry budget source", encoding="utf-8")
    authority_sha256 = "6" * 64
    key = raw_replay.stable_key(raw)
    _write_jsonl(
        paths["queue"],
        [
            {
                "key": key,
                "raw": raw.name,
                "path": str(raw),
                "status": "failed",
                "attempts": 1,
                "sources": ["ingest_failure"],
                "priority": 300,
                "bytes": raw.stat().st_size,
                "date": "20260714",
                "raw_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
                "last_error": (
                    "local consensus semantic no quorum "
                    f"[authority_sha256={authority_sha256}]: dry budget split"
                ),
                "next_retry_at": "2099-01-01T00:00:00+00:00",
            }
        ],
    )
    monkeypatch.setattr(
        raw_replay, "_active_terminal_semantic_deferred_raw_names", lambda: frozenset()
    )
    monkeypatch.setattr(
        raw_replay, "_active_operational_deferred_raw_statuses", lambda: {}
    )
    monkeypatch.setattr(
        failure_supervisor,
        "_current_adopted_authority_sha256",
        lambda: authority_sha256,
    )
    monkeypatch.setattr(
        raw_replay,
        "_preview_semantic_no_quorum_defer",
        lambda *_args, **_kwargs: pytest.fail(
            "dry budget denial must precede source verification"
        ),
    )
    budget = CycleBudget(
        max_local_calls=1,
        max_frontier_calls=0,
        max_mutations=0,
        max_raw_bytes=100,
        max_elapsed_seconds=60,
    )

    result = raw_replay.run_pending_queue(
        path=paths["queue"],
        history_file=paths["history"],
        claims_file=paths["claims"],
        completions_file=paths["completions"],
        max_runs=1,
        max_bytes=100,
        dry_run=True,
        eligible_keys={key},
        eligible_sources=None,
        budget=budget,
        now=datetime(2026, 7, 14, tzinfo=UTC),
    )

    assert result["count"] == 0
    assert result["planned"] == []
    assert result["budget_deferred"] == [
        {
            "key": key,
            "raw": raw.name,
            "reason": "mutation_budget_exhausted",
            "action": "semantic_defer_reconcile",
        }
    ]
    assert budget.snapshot()["used"]["mutation"] == 0


def test_successful_replay_releases_failure_supervisor_for_actual_raw(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chronovisor.ingest import failure_supervisor

    paths = _isolate_paths(tmp_path, monkeypatch)
    raw = paths["raw"] / "20260714-authority-released-success.md"
    raw.write_text("successful authority-epoch replay", encoding="utf-8")
    key = raw_replay.stable_key(raw)
    _write_jsonl(
        paths["queue"],
        [
            {
                "key": key,
                "raw": raw.name,
                "path": str(raw),
                "status": "pending",
                "attempts": 0,
                "sources": ["ingest_failure"],
                "priority": 300,
                "bytes": raw.stat().st_size,
                "date": "20260714",
            }
        ],
    )
    monkeypatch.setattr(
        raw_replay,
        "_active_terminal_semantic_deferred_raw_names",
        lambda: frozenset(),
    )
    monkeypatch.setattr(
        raw_replay,
        "_active_operational_deferred_raw_statuses",
        lambda: {},
    )

    class Store:
        def __init__(self) -> None:
            self.jobs: dict[str, SimpleNamespace] = {}

        def create(self, processor: str) -> SimpleNamespace:
            job = SimpleNamespace(
                job_id="successful-replay-job",
                processor=processor,
                status=JobStatus.PENDING,
                error=None,
                result=None,
                pages_created=[],
                pages_updated=[],
            )
            self.jobs[job.job_id] = job
            return job

        def get(self, job_id: str) -> SimpleNamespace | None:
            return self.jobs.get(job_id)

        def update(self, job_id: str, **kwargs: object) -> None:
            for field, value in kwargs.items():
                setattr(self.jobs[job_id], field, value)

    store = Store()
    monkeypatch.setattr(raw_replay, "job_store", store)
    released: list[str] = []
    monkeypatch.setattr(failure_supervisor, "reset_raw_failure", released.append)

    def succeed(raw_text: str, job_id: str, *, on_complete, metadata) -> None:
        assert raw_text == raw.read_text()
        assert metadata == {"source_raw": f"replay:{raw.name}"}
        store.update(job_id, status=JobStatus.COMPLETED, result={})
        on_complete()

    monkeypatch.setattr('chronovisor.ingest.ingest.run_ingest', succeed)

    result = raw_replay.run_pending_queue(
        path=paths["queue"],
        history_file=paths["history"],
        claims_file=paths["claims"],
        completions_file=paths["completions"],
        max_runs=1,
        max_bytes=100,
        eligible_keys={key},
        eligible_sources=None,
        frontier_reviewer=lambda *_args, **_kwargs: pytest.fail(
            "successful replay must not invoke frontier"
        ),
        now=datetime(2026, 7, 14, tzinfo=UTC),
    )

    assert result["count"] == 1
    assert result["runs"][0]["status"] == "completed"
    assert released == [raw.name]
    pending_cleanup, completion = _read_jsonl(paths["completions"])
    assert pending_cleanup["failure_reset_pending"] is True
    assert completion["raw"] == raw.name
    assert completion["status"] == "completed"
    assert completion["failure_reset_pending"] is False
    [queue_row] = _read_jsonl(paths["queue"])
    assert queue_row["failure_reset_pending"] is False


def test_completion_survives_reset_failure_and_retries_cleanup_without_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chronovisor.ingest import failure_supervisor

    paths = _isolate_paths(tmp_path, monkeypatch)
    raw = paths["raw"] / "20260714-reset-retry.md"
    raw.write_text("durable completion before cleanup", encoding="utf-8")
    key = raw_replay.stable_key(raw)
    _write_jsonl(
        paths["queue"],
        [
            {
                "key": key,
                "raw": raw.name,
                "path": str(raw),
                "status": "pending",
                "attempts": 0,
                "sources": ["ingest_failure"],
                "priority": 300,
                "bytes": raw.stat().st_size,
                "date": "20260714",
            }
        ],
    )
    monkeypatch.setattr(
        raw_replay, "_active_terminal_semantic_deferred_raw_names", lambda: frozenset()
    )
    monkeypatch.setattr(
        raw_replay, "_active_operational_deferred_raw_statuses", lambda: {}
    )

    class Store:
        def __init__(self) -> None:
            self.jobs: dict[str, SimpleNamespace] = {}
            self.created = 0

        def create(self, processor: str) -> SimpleNamespace:
            self.created += 1
            job = SimpleNamespace(
                job_id=f"reset-retry-{self.created}",
                processor=processor,
                status=JobStatus.PENDING,
                error=None,
                result=None,
                pages_created=[],
                pages_updated=[],
            )
            self.jobs[job.job_id] = job
            return job

        def get(self, job_id: str) -> SimpleNamespace | None:
            return self.jobs.get(job_id)

        def update(self, job_id: str, **kwargs: object) -> None:
            for field, value in kwargs.items():
                setattr(self.jobs[job_id], field, value)

    store = Store()
    monkeypatch.setattr(raw_replay, "job_store", store)
    reset_calls: list[str] = []

    def flaky_reset(raw_name: str) -> None:
        reset_calls.append(raw_name)
        if len(reset_calls) == 1:
            raise OSError("transient supervisor state write failure")

    monkeypatch.setattr(failure_supervisor, "reset_raw_failure", flaky_reset)

    def succeed(_raw_text: str, job_id: str, *, on_complete, metadata) -> None:
        assert metadata == {"source_raw": f"replay:{raw.name}"}
        store.update(job_id, status=JobStatus.COMPLETED, result={})
        on_complete()

    monkeypatch.setattr('chronovisor.ingest.ingest.run_ingest', succeed)
    now = datetime(2026, 7, 14, tzinfo=UTC)
    first = raw_replay.run_pending_queue(
        path=paths["queue"],
        history_file=paths["history"],
        claims_file=paths["claims"],
        completions_file=paths["completions"],
        max_runs=1,
        max_bytes=100,
        eligible_keys={key},
        eligible_sources=None,
        now=now,
    )

    assert first["runs"][0]["status"] == "completed"
    assert first["runs"][0]["failure_reset_pending"] is True
    [after_first] = _read_jsonl(paths["queue"])
    assert after_first["status"] == "completed"
    assert after_first["failure_reset_pending"] is True

    second = raw_replay.run_pending_queue(
        path=paths["queue"],
        history_file=paths["history"],
        claims_file=paths["claims"],
        completions_file=paths["completions"],
        max_runs=1,
        max_bytes=100,
        eligible_keys={key},
        eligible_sources=None,
        now=now,
    )

    assert second["count"] == 0
    assert second["failure_reset_reconciled"] == [
        {"key": key, "raw": raw.name, "status": "completed"}
    ]
    assert reset_calls == [raw.name, raw.name]
    assert store.created == 1
    [after_second] = _read_jsonl(paths["queue"])
    assert after_second["status"] == "completed"
    assert after_second["failure_reset_pending"] is False
    assert _read_jsonl(paths["completions"])[-1]["failure_reset_pending"] is False


def test_pending_failure_reset_obeys_mutation_budget_without_queue_rewrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chronovisor.ingest import failure_supervisor

    paths = _isolate_paths(tmp_path, monkeypatch)
    raw = paths["raw"] / "20260714-reset-budget.md"
    raw.write_text("completed raw awaiting cleanup", encoding="utf-8")
    key = raw_replay.stable_key(raw)
    raw_sha256 = hashlib.sha256(raw.read_bytes()).hexdigest()
    queue_row = {
        "schema_version": 2,
        "key": key,
        "type": "raw_replay_candidate",
        "raw": raw.name,
        "path": str(raw),
        "status": "completed",
        "attempts": 1,
        "sources": ["ingest_failure"],
        "reasons": [],
        "priority": 300,
        "bytes": raw.stat().st_size,
        "date": "20260714",
        "created_at": "2026-07-14T00:00:00+00:00",
        "updated_at": "2026-07-14T00:01:00+00:00",
        "last_attempt_at": "2026-07-14T00:01:00+00:00",
        "completed_at": "2026-07-14T00:01:00+00:00",
        "next_retry_at": None,
        "last_error": None,
        "raw_sha256": raw_sha256,
        "failure_reset_pending": True,
    }
    _write_jsonl(paths["queue"], [queue_row])
    _write_jsonl(
        paths["completions"],
        [
            {
                "ts": "2026-07-14T00:01:00+00:00",
                "schema_version": 2,
                "type": "raw_replay_completion",
                "key": key,
                "raw": raw.name,
                "source_raw": f"replay:{raw.name}",
                "status": "completed",
                "completion_scope": "full_raw",
                "raw_sha256": raw_sha256,
                "failure_reset_pending": True,
            }
        ],
    )
    queue_before = paths["queue"].read_bytes()
    monkeypatch.setattr(
        raw_replay, "_active_terminal_semantic_deferred_raw_names", lambda: frozenset()
    )
    monkeypatch.setattr(
        raw_replay, "_active_operational_deferred_raw_statuses", lambda: {}
    )
    monkeypatch.setattr(
        failure_supervisor,
        "reset_raw_failure",
        lambda _raw: pytest.fail("zero mutation budget must defer cleanup"),
    )
    budget = CycleBudget(
        max_local_calls=0,
        max_frontier_calls=0,
        max_mutations=0,
        max_raw_bytes=0,
        max_elapsed_seconds=60,
    )

    result = raw_replay.run_pending_queue(
        path=paths["queue"],
        history_file=paths["history"],
        claims_file=paths["claims"],
        completions_file=paths["completions"],
        max_runs=1,
        max_bytes=100,
        eligible_keys={key},
        eligible_sources=None,
        budget=budget,
        now=datetime(2026, 7, 14, tzinfo=UTC),
    )

    assert result["count"] == 0
    assert result["budget_deferred"] == [
        {
            "key": key,
            "raw": raw.name,
            "reason": "mutation_budget_exhausted",
            "action": "failure_reset_reconcile",
        }
    ]
    assert paths["queue"].read_bytes() == queue_before
    assert _read_jsonl(paths["completions"])[-1]["failure_reset_pending"] is True
