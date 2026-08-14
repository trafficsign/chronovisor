"""Tests for dashboard data assembly."""

from __future__ import annotations

import base64
import json
import os
import socket
import ssl
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from chronovisor.core import ollama, runtime_status
from chronovisor.core.activity_log import activity_record
from chronovisor.core.canonical_json import canonical_json_line_bytes_strict
from chronovisor.core.durable_state import write_sealed_json
from chronovisor.core.runtime_config import SearchEmbeddingConfig
from chronovisor.ingest import orchestrator
from chronovisor.ops import dashboard


@pytest.fixture(autouse=True)
def isolate_live_runtime_routes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep dashboard unit tests off the operator's live runtime config."""

    monkeypatch.setattr(
        dashboard,
        "runtime_generation_routes",
        lambda roles: tuple(
            ollama.RuntimeGenerationRoute(
                role=role,
                provider="ollama",
                model=f"{role}:test",
                location="local",
                structured_output=True,
            )
            for role in roles
        ),
    )


def _reset_processing_activity_cache() -> None:
    with dashboard._PROCESSING_ACTIVITY_CACHE_CONDITION:
        dashboard._PROCESSING_ACTIVITY_CACHE.update(
            {
                "source": None,
                "snapshot": None,
                "audited_at": 0.0,
                "refreshing": False,
                "build_count": 0,
                "cache_hits": 0,
                "coalesced": 0,
                "error_count": 0,
                "last_build_duration_ms": 0.0,
                "last_error": None,
            }
        )
        dashboard._PROCESSING_ACTIVITY_CACHE_CONDITION.notify_all()


def _reset_snapshot_fingerprint_cache() -> None:
    with dashboard._SNAPSHOT_FINGERPRINT_CONDITION:
        dashboard._SNAPSHOT_FINGERPRINT_CACHE.update(
            {
                "source": None,
                "fingerprint": None,
                "audited_at": 0.0,
                "probing": False,
                "generation": 0,
                "probe_count": 0,
                "cache_hits": 0,
                "coalesced": 0,
                "error_count": 0,
            }
        )
        dashboard._SNAPSHOT_FINGERPRINT_CONDITION.notify_all()


def _run_node_scenario(source: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["node", "-"],
        input=source,
        check=True,
        capture_output=True,
        encoding="utf-8",
    )


def _activity_bytes(*rows: tuple[str, str]) -> bytes:
    return b"".join(
        canonical_json_line_bytes_strict(
            activity_record(message, source="ingest", timestamp=timestamp)
        )
        for timestamp, message in rows
    )


def test_typed_graph_dashboard_snapshot_separates_engineering_and_authority(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(dashboard, "CHRONOVISOR_ROOT", tmp_path)
    write_sealed_json(
        tmp_path / "runtime" / "typed-graph" / "status.json",
        {
            "mode": "candidate",
            "engineering_complete": True,
            "authority_mature": False,
            "relation_counts": {"verified": 7, "authoritative": 0},
            "builder": {"status": "ok", "remaining_pages": 12},
            "consensus": {"status": "ok", "verified": 3, "held": 1},
            "external_model_calls": 0,
        },
    )
    write_sealed_json(
        tmp_path / "runtime" / "typed-graph" / "promotion.json",
        {"mode": "shadow", "canary_percent": 0, "reason": "collecting"},
    )
    write_sealed_json(
        tmp_path / "runtime" / "recall-rubric" / "status.json",
        {"status": "builtin", "samples": 4},
    )

    value = dashboard._typed_graph_dashboard_snapshot()

    assert value["engineering_complete"] is True
    assert value["authority_mature"] is False
    assert value["builder"]["remaining_pages"] == 12
    assert value["rollout"] == {
        "mode": "shadow",
        "canary_percent": 0,
        "reason": "collecting",
        "gates": {},
        "sample_count": 0,
        "sample_unit": "",
    }
    assert "collecting authority" in dashboard._typed_graph_lane_detail(value)


def test_mark_batch_activity_requires_a_running_batch_job() -> None:
    completed = {
        "state": "idle",
        "stage": "idle",
        "current_job_id": None,
        "batch": {"index": 2, "total": 2, "succeeded": 2, "failed": 0},
    }
    dashboard._mark_batch_activity(completed)
    assert completed["batch"]["active"] is False

    running = {
        "state": "running",
        "stage": "raw",
        "current_job_id": "job-1",
        "batch": {"index": 1, "total": 2, "succeeded": 0, "failed": 0},
    }
    dashboard._mark_batch_activity(running)
    assert running["batch"]["active"] is True

    waiting = {
        "state": "running",
        "stage": "waiting",
        "current_job_id": None,
        "batch": {"index": 2, "total": 2, "succeeded": 2, "failed": 0},
    }
    dashboard._mark_batch_activity(waiting)
    assert waiting["batch"]["active"] is False


def test_mark_batch_activity_accepts_local_consensus_stages_and_legacy_cache() -> None:
    for stage in (
        "local-consensus-review",
        "local-regenerate",
        "frontier-regenerate",
    ):
        status = {
            "state": "running",
            "stage": stage,
            "current_job_id": "job-1",
            "batch": {"index": 1, "total": 2, "succeeded": 0, "failed": 0},
        }

        dashboard._mark_batch_activity(status)

        assert status["batch"]["active"] is True


def test_dead_orchestrator_pid_clears_stale_live_status(monkeypatch) -> None:
    monkeypatch.setattr(runtime_status, "_pid_is_alive", lambda _pid: False)
    cached = {
        "state": "running",
        "stage": "triage",
        "current_raw": "pending.md",
        "current_op": "create",
        "current_job_id": "job-1",
        "current_job_pid": 999999,
        "llm": {"active": True, "model": "local"},
        "batch": {"index": 2, "total": 2, "succeeded": 2, "failed": 0},
    }
    orchestrator_state = {"current_job_id": "job-1", "current_job_pid": 999999}

    status = dashboard._canonicalize_runtime_status(
        cached,
        orchestrator_state,
        pending=1,
    )
    dashboard._mark_batch_activity(status)

    assert status["state"] == "idle"
    assert status["stage"] == "waiting"
    assert status["current_job_id"] is None
    assert status["current_job_pid"] is None
    assert status["current_raw"] is None
    assert status["llm"]["active"] is False
    assert status["batch"]["active"] is False


def test_reused_orchestrator_pid_clears_stale_live_status(monkeypatch) -> None:
    monkeypatch.setattr(runtime_status, "_pid_is_alive", lambda _pid: True)
    monkeypatch.setattr(
        dashboard,
        "_process_started_at",
        lambda _pid: datetime.fromisoformat("2026-07-11T18:00:00"),
    )
    cached = {
        "state": "running",
        "stage": "generate",
        "current_job_id": "job-1",
        "current_job_pid": 4242,
        "llm": {"active": True},
        "batch": {"index": 2, "total": 2, "succeeded": 2, "failed": 0},
    }
    orchestrator_state = {
        "current_job_id": "job-1",
        "current_job_pid": 4242,
        "current_job_started_at": "2026-07-11T17:00:00",
    }

    status = dashboard._canonicalize_runtime_status(
        cached,
        orchestrator_state,
        pending=1,
    )
    dashboard._mark_batch_activity(status)

    assert status["state"] == "idle"
    assert status["current_job_id"] is None
    assert status["batch"]["active"] is False


def test_original_orchestrator_process_keeps_live_status(monkeypatch) -> None:
    from chronovisor.ingest import orchestrator

    monkeypatch.setattr(runtime_status, "_pid_is_alive", lambda _pid: True)
    monkeypatch.setattr(
        dashboard,
        "_process_started_at",
        lambda _pid: datetime.fromisoformat("2026-07-11T16:59:50"),
    )
    monkeypatch.setattr(orchestrator, "ingest_process_lease_is_held", lambda _pid: True)
    status = dashboard._canonicalize_runtime_status(
        {"state": "idle", "stage": "waiting"},
        {
            "current_job_id": "job-1",
            "current_job_pid": 4242,
            "current_job_started_at": "2026-07-11T17:00:00",
        },
        pending=1,
    )

    assert status["state"] == "running"
    assert status["current_job_id"] == "job-1"


def test_live_long_lived_process_without_ingest_lease_is_idle(monkeypatch) -> None:
    from chronovisor.ingest import orchestrator

    monkeypatch.setattr(runtime_status, "_pid_is_alive", lambda _pid: True)
    monkeypatch.setattr(
        dashboard,
        "_process_started_at",
        lambda _pid: datetime.fromisoformat("2026-07-11T16:59:50"),
    )
    monkeypatch.setattr(
        orchestrator,
        "ingest_process_lease_is_held",
        lambda _pid: False,
    )

    status = dashboard._canonicalize_runtime_status(
        {"state": "running", "stage": "raw", "current_job_id": "job-1"},
        {
            "current_job_id": "job-1",
            "current_job_pid": 4242,
            "current_job_started_at": "2026-07-11T17:00:00",
        },
        pending=1,
    )

    assert status["state"] == "idle"
    assert status["stage"] == "waiting"
    assert status["current_job_id"] is None


def test_snapshot_component_error_boundary_returns_structured_error() -> None:
    def broken_component():
        raise RuntimeError("duplicate page id")

    result = dashboard._safe_snapshot_component(
        "health",
        broken_component,
        {"summary": {}},
    )

    assert result == {
        "summary": {},
        "status": "error",
        "component": "health",
        "error_class": "RuntimeError",
        "error": "duplicate page id",
    }


def test_local_consensus_snapshot_removes_dead_markers_and_exposes_redacted_metrics(
    tmp_path: Path,
    monkeypatch,
) -> None:
    chronovisor_root = tmp_path / "wiki"
    root = chronovisor_root / "runtime" / "local-consensus"
    active_dir = root / "active"
    active_dir.mkdir(parents=True)
    alive_pid = os.getpid()
    dead_pid = 999_999
    marker = {
        "request_sha256": "a" * 64,
        "role": "primary",
        "model": "ornith:test",
        "think": "medium",
        "required_num_ctx": 12_000,
        "requested_num_ctx": 16_384,
        "context_tokens": 16_384,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "pid": alive_pid,
    }
    (active_dir / "alive.json").write_text(json.dumps(marker), encoding="utf-8")
    (active_dir / "dead.json").write_text(
        json.dumps({**marker, "pid": dead_pid}),
        encoding="utf-8",
    )
    (active_dir / "stale.json").write_text(
        json.dumps({**marker, "started_at": "2020-01-01T00:00:00Z"}),
        encoding="utf-8",
    )
    summary = dashboard._empty_local_consensus_summary()
    summary["sessions"].update(
        {"first_pass_valid": 7, "repaired": 2, "repair_turns": 3}
    )
    summary["decisions"].update(
        {
            "pair_agreement": 4,
            "tie_break_used": 2,
            "unresolved_quarantine": 1,
            "conservative_veto_fired": 3,
            "conservative_veto_bypassed_by_lane_policy": 2,
            "dissent_effect_classes": {
                "conservative": 2,
                "unclassifiable": 1,
            },
            "model_conservative_vote_rates": {
                "ornith:test": {
                    "valid_votes": 4,
                    "conservative_votes": 1,
                    "conservative_rate": 0.25,
                }
            },
        }
    )
    root.mkdir(parents=True, exist_ok=True)
    (root / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (root / "audit.jsonl").write_text(
        json.dumps(
            {
                "kind": "session",
                "timestamp": "2026-07-11T11:59:00Z",
                "request_sha256": "a" * 64,
                "role": "primary",
                "model": "ornith:test",
                "think": "medium",
                "required_num_ctx": 12_000,
                "requested_num_ctx": 16_384,
                "context_tokens": 16_384,
                "ok": True,
            }
        )
        + "\n"
        + json.dumps(
            {
                "kind": "decision",
                "timestamp": "2026-07-11T12:00:00Z",
                "request_sha256": "a" * 64,
                "status": "agreed",
                "pair_agreement": True,
                "tie_break_used": False,
                "unresolved_quarantine": False,
                "conservative_veto_fired": True,
                "conservative_veto_bypassed_by_lane_policy": True,
                "dissent_effect_class": "conservative",
                "prompt": "secret prompt",
                "raw_output": "secret output",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(dashboard, "CHRONOVISOR_ROOT", chronovisor_root)
    monkeypatch.setattr(
        runtime_status,
        "_pid_is_alive",
        lambda pid: pid == alive_pid,
    )

    snapshot = dashboard._local_consensus_snapshot()

    assert snapshot["active"] is True
    assert snapshot["count"] == 1
    assert snapshot["activities"][0]["model"] == "ornith:test"
    assert snapshot["activities"][0]["think"] == "medium"
    assert snapshot["activities"][0]["required_context_tokens"] == 12_000
    assert snapshot["activities"][0]["requested_context_tokens"] == 16_384
    assert snapshot["activities"][0]["context_tokens"] == 16_384
    assert snapshot["summary"]["sessions"]["first_pass_valid"] == 7
    assert snapshot["summary"]["sessions"]["repaired"] == 2
    assert snapshot["summary"]["sessions"]["repair_turns"] == 3
    assert snapshot["summary"]["decisions"]["pair_agreement"] == 4
    assert snapshot["summary"]["decisions"]["tie_break_used"] == 2
    assert snapshot["summary"]["decisions"]["unresolved_quarantine"] == 1
    assert snapshot["summary"]["decisions"]["conservative_veto_fired"] == 3
    assert (
        snapshot["summary"]["decisions"][
            "conservative_veto_bypassed_by_lane_policy"
        ]
        == 2
    )
    assert snapshot["summary"]["decisions"]["dissent_effect_classes"] == {
        "conservative": 2,
        "unclassifiable": 1,
    }
    assert snapshot["summary"]["decisions"]["model_conservative_vote_rates"] == {
        "ornith:test": {
            "valid_votes": 4,
            "conservative_votes": 1,
            "conservative_rate": 0.25,
        }
    }
    session = next(row for row in snapshot["history"] if row["kind"] == "session")
    decision = next(row for row in snapshot["history"] if row["kind"] == "decision")
    assert session["think"] == "medium"
    assert session["required_context_tokens"] == 12_000
    assert session["requested_context_tokens"] == 16_384
    assert session["context_tokens"] == 16_384
    assert decision["conservative_veto_fired"] is True
    assert (
        decision["conservative_veto_bypassed_by_lane_policy"] is True
    )
    assert decision["dissent_effect_class"] == "conservative"
    assert all("prompt" not in row for row in snapshot["history"])
    assert all("raw_output" not in row for row in snapshot["history"])
    assert not (active_dir / "dead.json").exists()
    assert not (active_dir / "stale.json").exists()


def test_local_consensus_reads_trace_before_audit_history(
    tmp_path: Path, monkeypatch
) -> None:
    reads: list[str] = []

    def read_jsonl(path: Path, *, limit: int) -> list[dict[str, Any]]:
        reads.append(path.name)
        return []

    monkeypatch.setattr(dashboard, "CHRONOVISOR_ROOT", tmp_path)
    monkeypatch.setattr(dashboard, "_local_consensus_activities", lambda: [])
    monkeypatch.setattr(dashboard, "_read_json_file", lambda _path: {})
    monkeypatch.setattr(dashboard, "_read_jsonl_file", read_jsonl)
    monkeypatch.setattr(
        dashboard,
        "_decision_trace_models",
        lambda: {role: "not configured" for role in dashboard._DECISION_TRACE_ROLES},
    )

    dashboard._local_consensus_snapshot()

    assert reads == ["trace-events.jsonl", "audit.jsonl"]


def test_processing_activity_projects_simultaneous_llm_workflows(monkeypatch) -> None:
    _reset_processing_activity_cache()
    status = {
        "state": "running",
        "stage": "apply",
        "current_job_id": "ingest-job",
        "current_raw": "raw.md",
        "batch": {"active": True},
        "llm": {
            "active": True,
            "model": "ingest:test",
            "started_at": "2026-08-01T00:00:00+09:00",
            "updated_at": "2026-08-01T00:00:01+09:00",
        },
    }
    monkeypatch.setattr(runtime_status, "read_status", lambda: dict(status))
    monkeypatch.setattr(orchestrator, "_load_state", lambda: {})
    monkeypatch.setattr(
        dashboard,
        "_canonicalize_runtime_status",
        lambda value, _orch, *, pending: dict(value),
    )
    monkeypatch.setattr(
        dashboard,
        "_local_consensus_activities",
        lambda: [
            {
                "request_sha256": "a" * 64,
                "role": "recall_auto_apply:challenger",
                "model": "recall:test",
                "phase": "generate",
                "started_at": "2026-08-01T00:00:02+09:00",
                "updated_at": "2026-08-01T00:00:03+09:00",
                "pid": 42,
                "thread_id": 84,
            },
            {
                "request_sha256": "b" * 64,
                "role": "model_eval:primary",
                "model": "eval:test",
                "phase": "validate",
                "started_at": "2026-08-01T00:00:04+09:00",
                "updated_at": "2026-08-01T00:00:05+09:00",
            },
        ],
    )
    monkeypatch.setattr(
        dashboard,
        "_model_activities",
        lambda: [
            {
                "activity_id": "same-call",
                "pipeline": "audit",
                "component": "chronovisor.decision.local_structured",
                "caller": "default_transport",
                "operation": "chat",
                "model": "recall:test",
                "started_at": "2026-08-01T00:00:02+09:00",
                "updated_at": "2026-08-01T00:00:03+09:00",
                "pid": 42,
                "thread_id": 84,
            },
            {
                "activity_id": "recent-ingest-call",
                "pipeline": "ingest",
                "component": "chronovisor.ingest.ingest_generation",
                "caller": "generate_page",
                "operation": "generate",
                "model": "stale-ingest:test",
                "started_at": "2026-08-01T00:00:01+09:00",
                "updated_at": "2026-08-01T00:00:02+09:00",
                "recent": True,
            },
        ],
    )
    monkeypatch.setattr(
        dashboard,
        "_frontier_activity_snapshot",
        lambda: {"active": False, "count": 0, "latest": None},
    )
    monkeypatch.setattr(
        dashboard,
        "_frontier_repair_snapshot",
        lambda limit=40: {"active": False, "active_incident": None},
    )

    first = dashboard._processing_activity_snapshot()
    second = dashboard._processing_activity_snapshot()
    by_key = {lane["key"]: lane for lane in first["lanes"]}

    assert first["active_count"] == 3
    assert first["revision"] == second["revision"]
    assert by_key["ingest"]["current_step"] == "apply"
    assert by_key["ingest"]["model"] == "ingest:test"
    assert by_key["recall"]["current_step"] == "challenger"
    assert by_key["recall"]["role"] == "recall_auto_apply:challenger"
    assert by_key["improve"]["current_step"] == "verify"
    assert next(
        step for step in by_key["recall"]["steps"] if step["key"] == "challenger"
    )["status"] == "active"
    assert by_key["audit"]["state"] == "idle"
    assert by_key["repair"]["state"] == "idle"


def test_processing_activity_projects_direct_ollama_calls(monkeypatch) -> None:
    _reset_processing_activity_cache()
    monkeypatch.setattr(runtime_status, "read_status", lambda: {})
    monkeypatch.setattr(orchestrator, "_load_state", lambda: {})
    monkeypatch.setattr(
        dashboard,
        "_canonicalize_runtime_status",
        lambda value, _orch, *, pending: dict(value),
    )
    monkeypatch.setattr(dashboard, "_local_consensus_activities", lambda: [])
    monkeypatch.setattr(
        dashboard,
        "_model_activities",
        lambda: [
            {
                "activity_id": "activity-1",
                "pipeline": "recall",
                "component": "chronovisor.recall.recall_processor",
                "caller": "judge_candidates",
                "operation": "chat",
                "model": "recall:test",
                "started_at": "2026-08-01T00:00:00+09:00",
                "updated_at": "2026-08-01T00:00:00+09:00",
                "pid": 42,
            }
        ],
    )
    monkeypatch.setattr(
        dashboard,
        "_frontier_activity_snapshot",
        lambda: {"active": False, "count": 0, "latest": None},
    )
    monkeypatch.setattr(
        dashboard,
        "_frontier_repair_snapshot",
        lambda limit=40: {"active": False, "active_incident": None},
    )

    snapshot = dashboard._processing_activity_snapshot()
    by_key = {lane["key"]: lane for lane in snapshot["lanes"]}

    assert snapshot["active_count"] == 1
    assert by_key["recall"]["current_step"] == "primary"
    assert by_key["recall"]["role"] == "Recall Processor"
    assert by_key["recall"]["phase"] == "judge_candidates"
    assert by_key["recall"]["model"] == "recall:test"


def test_processing_activity_cache_single_flights_concurrent_callers(
    monkeypatch,
) -> None:
    _reset_processing_activity_cache()
    entered = threading.Event()
    release = threading.Event()
    calls = 0

    def build() -> dict:
        nonlocal calls
        calls += 1
        entered.set()
        assert release.wait(timeout=2)
        return {"revision": "one", "active_count": 0, "lanes": []}

    monkeypatch.setattr(
        dashboard, "_processing_activity_source_fingerprint", lambda: ("same",)
    )
    monkeypatch.setattr(dashboard, "_build_processing_activity_snapshot", build)

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(dashboard._processing_activity_snapshot) for _ in range(8)]
        assert entered.wait(timeout=2)
        release.set()
        snapshots = [future.result(timeout=2) for future in futures]

    assert calls == 1
    assert {snapshot["revision"] for snapshot in snapshots} == {"one"}
    metrics = snapshots[-1]["_dashboard"]["activity_cache"]
    assert metrics["build_count"] == 1
    assert metrics["coalesced"] >= 1


def test_processing_activity_cache_reuses_recent_source_and_refreshes_on_change(
    monkeypatch,
) -> None:
    _reset_processing_activity_cache()
    clock = [100.0]
    source = [("source", 1)]
    calls = 0

    def build() -> dict:
        nonlocal calls
        calls += 1
        return {
            "revision": str(calls),
            "active_count": 1,
            "lanes": [{"state": "active", "recent": True}],
        }

    class ImmediateThread:
        def __init__(self, *, target, args, **_kwargs) -> None:
            self.target = target
            self.args = args

        def start(self) -> None:
            self.target(*self.args)

    monkeypatch.setattr(dashboard.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        dashboard, "_processing_activity_source_fingerprint", lambda: source[0]
    )
    monkeypatch.setattr(dashboard, "_build_processing_activity_snapshot", build)
    monkeypatch.setattr(dashboard.threading, "Thread", ImmediateThread)

    assert dashboard._processing_activity_snapshot()["revision"] == "1"
    assert dashboard._processing_activity_snapshot()["revision"] == "1"
    assert calls == 1

    source[0] = ("source", 2)
    # A real source change refreshes synchronously on the next poll even
    # though no recent-lane audit time has elapsed.
    # The changed-source leader publishes fresh data; concurrent callers can
    # still receive the last-good snapshot while this one build is in flight.
    assert dashboard._processing_activity_snapshot()["revision"] == "2"
    assert dashboard._processing_activity_snapshot()["revision"] == "2"
    assert calls == 2


def test_processing_activity_source_fingerprint_observes_atomic_marker_rename(
    tmp_path: Path, monkeypatch
) -> None:
    chronovisor_root = tmp_path / "wiki"
    active_dir = chronovisor_root / "runtime" / "model-activity" / "active"
    active_dir.mkdir(parents=True)
    monkeypatch.setattr(dashboard, "CHRONOVISOR_ROOT", chronovisor_root)
    monkeypatch.setattr(
        runtime_status, "STATUS_FILE", chronovisor_root / "runtime" / "status.json"
    )
    monkeypatch.setattr(
        orchestrator, "STATE_FILE", chronovisor_root / ".orchestrator_state.json"
    )

    before = dashboard._processing_activity_source_fingerprint()
    temporary = active_dir / ".activity.tmp"
    temporary.write_text('{"schema_version": 1}\n', encoding="utf-8")
    os.replace(temporary, active_dir / "activity.json")
    after = dashboard._processing_activity_source_fingerprint()

    assert before != after


def test_processing_activity_source_tracks_local_consensus_marker_lifecycle(
    tmp_path: Path, monkeypatch
) -> None:
    chronovisor_root = tmp_path / "wiki"
    active_dir = chronovisor_root / "runtime" / "local-consensus" / "active"
    active_dir.mkdir(parents=True)
    monkeypatch.setattr(dashboard, "CHRONOVISOR_ROOT", chronovisor_root)
    monkeypatch.setattr(
        runtime_status, "STATUS_FILE", chronovisor_root / "runtime" / "status.json"
    )
    monkeypatch.setattr(
        orchestrator, "STATE_FILE", chronovisor_root / ".orchestrator_state.json"
    )

    before = dashboard._processing_activity_source_fingerprint()
    temporary = active_dir / ".review.tmp"
    marker = active_dir / "review.json"
    temporary.write_text('{"schema_version": 1}\n', encoding="utf-8")
    os.replace(temporary, marker)
    after_create = dashboard._processing_activity_source_fingerprint()
    marker.unlink()
    after_remove = dashboard._processing_activity_source_fingerprint()

    assert after_create != before
    assert after_remove != after_create


def test_processing_activity_cache_rejects_a_build_crossing_source_epochs(
    monkeypatch,
) -> None:
    _reset_processing_activity_cache()
    source = [("source", 1)]
    calls = 0

    def build() -> dict:
        nonlocal calls
        calls += 1
        if calls == 2:
            source[0] = ("source", 3)
        return {"revision": str(calls), "active_count": 0, "lanes": []}

    monkeypatch.setattr(
        dashboard, "_processing_activity_source_fingerprint", lambda: source[0]
    )
    monkeypatch.setattr(dashboard, "_build_processing_activity_snapshot", build)

    assert dashboard._processing_activity_snapshot()["revision"] == "1"
    source[0] = ("source", 2)
    # Revision 2 spans source epochs 2 and 3, so it is never published over
    # the last known-good revision.
    assert dashboard._processing_activity_snapshot()["revision"] == "1"
    assert dashboard._PROCESSING_ACTIVITY_CACHE["source"] is None
    assert dashboard._processing_activity_snapshot()["revision"] == "3"
    assert calls == 3


def test_processing_activity_cold_cache_retries_cross_epoch_build(
    monkeypatch,
) -> None:
    _reset_processing_activity_cache()
    source = [("source", 1)]
    calls = 0

    def build() -> dict:
        nonlocal calls
        calls += 1
        if calls == 1:
            source[0] = ("source", 2)
        return {"revision": str(calls), "active_count": 0, "lanes": []}

    monkeypatch.setattr(
        dashboard, "_processing_activity_source_fingerprint", lambda: source[0]
    )
    monkeypatch.setattr(dashboard, "_build_processing_activity_snapshot", build)

    snapshot = dashboard._processing_activity_snapshot()

    assert snapshot["revision"] == "2"
    assert calls == 2
    assert dashboard._PROCESSING_ACTIVITY_CACHE["source"] == ("source", 2)
    assert snapshot["_dashboard"]["activity_cache"]["build_count"] == 1


def test_processing_activity_cold_cache_fails_closed_after_two_epoch_changes(
    monkeypatch,
) -> None:
    _reset_processing_activity_cache()
    source = [0]

    def build() -> dict:
        source[0] += 1
        return {"revision": str(source[0]), "active_count": 0, "lanes": []}

    monkeypatch.setattr(
        dashboard,
        "_processing_activity_source_fingerprint",
        lambda: ("source", source[0]),
    )
    monkeypatch.setattr(dashboard, "_build_processing_activity_snapshot", build)

    with pytest.raises(RuntimeError, match="source changed during two builds"):
        dashboard._processing_activity_snapshot()

    assert dashboard._PROCESSING_ACTIVITY_CACHE["snapshot"] is None
    assert dashboard._PROCESSING_ACTIVITY_CACHE["refreshing"] is False
    assert dashboard._PROCESSING_ACTIVITY_CACHE["error_count"] == 1


@pytest.mark.parametrize(
    ("active_count", "lanes"),
    [
        (0, []),
        (1, [{"state": "active", "recent": False}]),
    ],
    ids=["idle", "ongoing"],
)
def test_processing_activity_cache_audits_idle_and_ongoing_at_one_second(
    monkeypatch, active_count: int, lanes: list[dict]
) -> None:
    _reset_processing_activity_cache()
    clock = [200.0]
    calls = 0

    def build() -> dict:
        nonlocal calls
        calls += 1
        return {
            "revision": str(calls),
            "active_count": active_count,
            "lanes": lanes,
        }

    class ImmediateThread:
        def __init__(self, *, target, args, **_kwargs) -> None:
            self.target = target
            self.args = args

        def start(self) -> None:
            self.target(*self.args)

    monkeypatch.setattr(dashboard.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        dashboard, "_processing_activity_source_fingerprint", lambda: ("same",)
    )
    monkeypatch.setattr(dashboard, "_build_processing_activity_snapshot", build)
    monkeypatch.setattr(dashboard.threading, "Thread", ImmediateThread)

    assert dashboard._processing_activity_snapshot()["revision"] == "1"
    clock[0] += dashboard.PROCESSING_ACTIVITY_AUDIT_SECONDS - 0.001
    assert dashboard._processing_activity_snapshot()["revision"] == "1"
    clock[0] += 0.001
    assert dashboard._processing_activity_snapshot()["revision"] == "1"
    refreshed = dashboard._processing_activity_snapshot()

    assert refreshed["revision"] == "2"
    assert calls == 2
    metrics = refreshed["_dashboard"]["activity_cache"]
    assert metrics["audit_seconds"] == 1.0
    assert metrics["recent_audit_seconds"] == 0.25
    assert "active_audit_seconds" not in metrics


def test_processing_activity_cache_audits_recent_lane_at_poll_cadence(
    monkeypatch,
) -> None:
    _reset_processing_activity_cache()
    clock = [400.0]
    calls = 0

    def build() -> dict:
        nonlocal calls
        calls += 1
        return {
            "revision": str(calls),
            "active_count": 1,
            "lanes": [{"state": "active", "recent": True}],
        }

    class ImmediateThread:
        def __init__(self, *, target, args, **_kwargs) -> None:
            self.target = target
            self.args = args

        def start(self) -> None:
            self.target(*self.args)

    monkeypatch.setattr(dashboard.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        dashboard, "_processing_activity_source_fingerprint", lambda: ("same",)
    )
    monkeypatch.setattr(dashboard, "_build_processing_activity_snapshot", build)
    monkeypatch.setattr(dashboard.threading, "Thread", ImmediateThread)

    assert dashboard._processing_activity_snapshot()["revision"] == "1"
    clock[0] += dashboard.PROCESSING_ACTIVITY_RECENT_AUDIT_SECONDS - 0.001
    assert dashboard._processing_activity_snapshot()["revision"] == "1"
    clock[0] += 0.001
    assert dashboard._processing_activity_snapshot()["revision"] == "1"
    refreshed = dashboard._processing_activity_snapshot()

    assert refreshed["revision"] == "2"
    assert calls == 2
    assert (
        dashboard.PROCESSING_ACTIVITY_RECENT_AUDIT_SECONDS
        == dashboard.PROCESSING_ACTIVITY_POLL_SECONDS
    )
    assert (
        dashboard.PROCESSING_ACTIVITY_RECENT_AUDIT_SECONDS
        < dashboard.MODEL_ACTIVITY_VISIBLE_SECONDS
    )


def test_processing_activity_cache_keeps_stale_success_and_retries_on_audit(
    monkeypatch,
) -> None:
    _reset_processing_activity_cache()
    clock = [300.0]
    source = [("source", 1)]
    attempts = 0

    def build() -> dict:
        nonlocal attempts
        attempts += 1
        if attempts == 2:
            raise RuntimeError("transient")
        return {"revision": str(attempts), "active_count": 0, "lanes": []}

    class ImmediateThread:
        def __init__(self, *, target, args, **_kwargs) -> None:
            self.target = target
            self.args = args

        def start(self) -> None:
            self.target(*self.args)

    monkeypatch.setattr(dashboard.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        dashboard, "_processing_activity_source_fingerprint", lambda: source[0]
    )
    monkeypatch.setattr(dashboard, "_build_processing_activity_snapshot", build)
    monkeypatch.setattr(dashboard.threading, "Thread", ImmediateThread)

    assert dashboard._processing_activity_snapshot()["revision"] == "1"
    source[0] = ("source", 2)
    failed_refresh = dashboard._processing_activity_snapshot()
    assert failed_refresh["revision"] == "1"
    assert failed_refresh["_dashboard"]["activity_cache"]["error_count"] == 1
    assert dashboard._processing_activity_snapshot()["revision"] == "1"
    assert attempts == 2

    clock[0] += dashboard.PROCESSING_ACTIVITY_AUDIT_SECONDS
    assert dashboard._processing_activity_snapshot()["revision"] == "1"
    assert dashboard._processing_activity_snapshot()["revision"] == "3"
    assert attempts == 3


def test_processing_role_projection_covers_dashboard_lanes() -> None:
    assert dashboard._processing_pipeline_for_role("recall_judge") == "recall"
    assert dashboard._processing_pipeline_for_role("ingest_reconciliation") == "ingest"
    assert dashboard._processing_pipeline_for_role("model_eval:primary") == "improve"
    assert dashboard._processing_pipeline_for_role("orphan_link:challenger") == "improve"
    assert dashboard._processing_pipeline_for_role("content_correction_classification") == "audit"
    assert dashboard._processing_pipeline_for_role("local_repair") == "repair"
    assert dashboard._processing_model_step("recall", "search") == "search"
    assert dashboard._processing_model_step("recall", "rerank") == "rerank"
    assert dashboard._processing_model_step("recall", "chat") == "primary"


def test_decision_trace_projects_live_phase_and_completed_vote(monkeypatch) -> None:
    monkeypatch.setattr(
        dashboard,
        "_decision_trace_models",
        lambda: {
            "primary": "primary:model",
            "challenger": "challenger:model",
            "tie_break": "tie:model",
        },
    )
    request = "a" * 64
    trace = dashboard._decision_trace_snapshot(
        [
            {
                "request_sha256": request,
                "role": "ingest_review:challenger",
                "model": "challenger:model",
                "phase": "validate",
                "attempt": 1,
                "think": "high",
                "required_context_tokens": 24_000,
                "requested_context_tokens": 32_768,
                "context_tokens": 32_768,
                "elapsed_seconds": 42,
                "updated_at": "2026-07-15T12:00:00Z",
            }
        ],
        [
            {
                "kind": "session",
                "timestamp": "2026-07-15T11:59:00Z",
                "request_sha256": request,
                "role": "ingest_review:primary",
                "model": "primary:model",
                "ok": True,
                "first_pass_valid": True,
                "repair_turns": 2,
                "think": False,
                "required_context_tokens": 12_000,
                "requested_context_tokens": 16_384,
                "context_tokens": 16_384,
            }
        ],
        None,
    )

    assert trace["state"] == "active"
    assert trace["task_role"] == "ingest_review"
    assert trace["lanes"][0]["state"] == "done"
    assert trace["lanes"][0]["think"] == "off"
    assert trace["lanes"][0]["required_context_tokens"] == 12_000
    assert trace["lanes"][0]["requested_context_tokens"] == 16_384
    assert trace["lanes"][0]["context_tokens"] == 16_384
    assert trace["lanes"][0]["repair_turns"] == 2
    assert trace["lanes"][1]["state"] == "active"
    assert trace["lanes"][1]["think"] == "high"
    assert trace["lanes"][1]["required_context_tokens"] == 24_000
    assert trace["lanes"][1]["requested_context_tokens"] == 32_768
    assert trace["lanes"][1]["context_tokens"] == 32_768
    assert trace["context_tokens"] == 32_768
    assert trace["lanes"][1]["repair_turns"] == 0
    assert trace["lanes"][1]["steps"][4]["status"] == "active"
    assert trace["lanes"][2]["state"] == "pending"
    assert trace["lanes"][2]["think"] == "—"
    assert trace["lanes"][2]["required_context_tokens"] is None
    assert trace["lanes"][2]["requested_context_tokens"] is None
    assert trace["lanes"][2]["context_tokens"] is None
    assert trace["lanes"][2]["repair_turns"] == 0
    assert [step["label"] for step in trace["overall"]] == [
        "Packet",
        "Dispatch",
        "Generate",
        "Validate",
        "Quorum",
        "Artifact",
        "Decision",
    ]
    assert trace["overall"][2]["status"] == "done"
    assert trace["overall"][3]["status"] == "active"
    repair_steps = dashboard._decision_trace_steps("active", phase="repair")
    assert repair_steps[3]["status"] == "done"
    assert repair_steps[4]["status"] == "active"
    assert trace["overall"][4]["status"] == "pending"


def test_decision_trace_active_validation_marks_generation_done(monkeypatch) -> None:
    monkeypatch.setattr(
        dashboard,
        "_decision_trace_models",
        lambda: {
            "primary": "primary:model",
            "challenger": "challenger:model",
            "tie_break": "tie:model",
        },
    )
    request = "9" * 64

    trace = dashboard._decision_trace_snapshot(
        [
            {
                "request_sha256": request,
                "role": "wiki_generation",
                "model": "primary:model",
                "phase": "validate",
                "attempt": 0,
            }
        ],
        [],
        None,
    )

    assert trace["overall"][2]["status"] == "done"
    assert trace["overall"][3]["status"] == "active"


def test_decision_trace_exposes_only_ordered_events_for_current_request(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        dashboard,
        "_decision_trace_models",
        lambda: {
            "primary": "primary:model",
            "challenger": "challenger:model",
            "tie_break": "tie:model",
        },
    )
    request = "c" * 64
    other_request = "d" * 64
    rows = [
        {
            "event_id": "event-other",
            "kind": "phase",
            "timestamp": "2026-07-15T11:59:59Z",
            "request_sha256": other_request,
            "role": "ingest_review:primary",
            "model": "other:model",
            "phase": "generate",
            "attempt": 0,
            "status": "active",
            "prompt": "must not escape",
        },
        {
            "event_id": "event-1",
            "kind": "phase",
            "timestamp": "2026-07-15T12:00:00Z",
            "request_sha256": request,
            "role": "ingest_review:primary",
            "model": "primary:model",
            "phase": "generate",
            "attempt": 0,
            "status": "active",
            "think": False,
            "think_selection_reason": "bounded_lane",
            "required_num_ctx": 24_000,
            "requested_num_ctx": 32_768,
            "context_tokens": 32_768,
            "raw_output": "must not escape",
        },
        {
            "event_id": "event-repair",
            "kind": "phase",
            "timestamp": "2026-07-15T12:00:00.5Z",
            "request_sha256": request,
            "role": "ingest_review:primary",
            "model": "primary:model",
            "phase": "repair",
            "attempt": 1,
            "status": "active",
        },
        {
            "event_id": "event-2",
            "kind": "session",
            "timestamp": "2026-07-15T12:00:01Z",
            "request_sha256": request,
            "role": "ingest_review:primary",
            "model": "primary:model",
            "phase": "vote",
            "attempt": 0,
            "status": "done",
        },
    ]
    trace = dashboard._decision_trace_snapshot(
        [
            {
                "request_sha256": request,
                "role": "ingest_review:challenger",
                "model": "challenger:model",
                "phase": "context",
                "attempt": 0,
            }
        ],
        [],
        None,
        rows,
    )

    assert [event["event_id"] for event in trace["events"]] == [
        "event-1",
        "event-repair",
        "event-2",
    ]
    assert trace["event_count"] == 3
    assert trace["events"][0]["lane"] == "primary"
    assert trace["events"][0]["overall_key"] == "generate"
    assert trace["events"][0]["think"] == "off"
    assert trace["events"][0]["think_selection_reason"] == "bounded_lane"
    assert trace["events"][0]["required_context_tokens"] == 24_000
    assert trace["events"][0]["requested_context_tokens"] == 32_768
    assert trace["events"][0]["context_tokens"] == 32_768
    assert trace["events"][1]["overall_key"] == "validate"
    assert trace["events"][2]["label"] == "Vote accepted"
    assert "prompt" not in trace["events"][0]
    assert "raw_output" not in trace["events"][0]


@pytest.mark.parametrize(
    "prior_phases,expected_phase,expected_overall",
    (
        ((), "trigger", "dispatch"),
        (("trigger",), "trigger", "dispatch"),
        (("load",), "load", "dispatch"),
        (("generate",), "generate", "generate"),
        (("generate", "vote"), "generate", "generate"),
    ),
)
def test_decision_trace_failed_session_stays_at_last_observed_phase(
    prior_phases: tuple[str, ...],
    expected_phase: str,
    expected_overall: str,
) -> None:
    request = "8" * 64
    rows = []
    for index, phase in enumerate(prior_phases):
        rows.append(
            {
                "event_id": f"phase-{index}-{phase}",
                "kind": "phase",
                "timestamp": "2026-07-15T12:00:00Z",
                "request_sha256": request,
                "role": "ingest_review:primary",
                "phase": phase,
                "status": "active",
            }
        )
    rows.append(
        {
            "event_id": "failed-session",
            "kind": "session",
            "timestamp": "2026-07-15T12:00:01Z",
            "request_sha256": request,
            "role": "ingest_review:primary",
            "phase": "vote",
            "status": "error",
        }
    )

    failed = dashboard._decision_trace_events(rows, request_sha256=request)[-1]

    assert failed["phase"] == expected_phase
    assert failed["overall_key"] == expected_overall
    assert failed["label"] == "Session failed"


def test_local_consensus_snapshot_keeps_redacted_failed_vote_role(
    tmp_path: Path, monkeypatch
) -> None:
    request = "9" * 64
    audit_rows = [
        {
            "kind": "session",
            "timestamp": "2026-07-15T12:00:03Z",
            "request_sha256": request,
            "role": "local_repair:tie_break",
            "ok": False,
            "failure_class": "output_truncated",
        },
        {
            "kind": "decision",
            "timestamp": "2026-07-15T12:00:04Z",
            "request_sha256": request,
            "role": "local_repair",
            "status": "quarantined",
            "quarantine_reason": "local_models_did_not_reach_two_vote_quorum",
            "failure_class": "local_consensus_failed",
            "vote_count": 3,
            "valid_votes": 2,
            "pair_agreement": False,
            "tie_break_used": True,
            "votes": [
                {"role": "primary", "raw_output": "secret"},
                {"role": "challenger"},
                {"role": "tie_break"},
            ],
        },
    ]
    trace_rows = [
        {
            "event_id": "tie-validate",
            "kind": "phase",
            "timestamp": "2026-07-15T12:00:02Z",
            "request_sha256": request,
            "role": "local_repair:tie_break",
            "phase": "validate",
            "status": "active",
        },
        {
            "event_id": "tie-failed",
            "kind": "session",
            "timestamp": "2026-07-15T12:00:03Z",
            "request_sha256": request,
            "role": "local_repair:tie_break",
            "phase": "vote",
            "status": "error",
        },
    ]
    monkeypatch.setattr(dashboard, "CHRONOVISOR_ROOT", tmp_path)
    monkeypatch.setattr(dashboard, "_local_consensus_activities", lambda: [])
    monkeypatch.setattr(dashboard, "_read_json_file", lambda _path: {})
    monkeypatch.setattr(
        dashboard,
        "_read_jsonl_file",
        lambda path, *, limit: audit_rows if path.name == "audit.jsonl" else trace_rows,
    )

    snapshot = dashboard._local_consensus_snapshot()
    decision = snapshot["latest_decision"]
    tie_break = snapshot["decision_trace"]["lanes"][2]

    assert decision["vote_roles"] == ["primary", "challenger", "tie_break"]
    assert "votes" not in decision
    assert tie_break["phase"] == "vote"
    assert [step["status"] for step in tie_break["steps"]] == [
        "done",
        "done",
        "done",
        "done",
        "done",
        "error",
    ]


def test_decision_trace_failed_standalone_session_is_phase_aware_not_ready() -> None:
    request = "4" * 64
    session = {
        "kind": "session",
        "timestamp": "2026-07-15T12:00:01Z",
        "request_sha256": request,
        "role": "wiki_generation",
        "model": "primary:model",
        "ok": False,
        "failure_class": "repair_exhausted",
        "context_tokens": 32_768,
    }
    events = [
        {
            "event_id": phase,
            "kind": "phase",
            "timestamp": "2026-07-15T12:00:00Z",
            "request_sha256": request,
            "role": "wiki_generation",
            "phase": phase,
            "status": "active",
        }
        for phase in ("trigger", "load", "context", "generate")
    ]
    events.append(
        {
            "event_id": "failed",
            "kind": "session",
            "timestamp": session["timestamp"],
            "request_sha256": request,
            "role": "wiki_generation",
            "phase": "vote",
            "status": "error",
        }
    )

    trace = dashboard._decision_trace_snapshot([], [session], None, events)

    assert trace["state"] == "quarantined"
    assert trace["summary"] == "Structured result invalid"
    assert trace["context_tokens"] == 32_768
    assert trace["lanes"][0]["state"] == "error"
    assert trace["lanes"][0]["phase"] == "generate"
    assert [step["status"] for step in trace["lanes"][0]["steps"]] == [
        "done",
        "done",
        "done",
        "error",
        "skipped",
        "skipped",
    ]


def test_decision_trace_successful_standalone_session_completes_validation() -> None:
    trace = dashboard._decision_trace_snapshot(
        [],
        [
            {
                "kind": "session",
                "timestamp": "2026-07-15T12:00:01Z",
                "request_sha256": "5" * 64,
                "role": "wiki_generation",
                "model": "primary:model",
                "ok": True,
                "context_tokens": 16_384,
            }
        ],
        None,
    )

    assert trace["state"] == "ready"
    assert trace["context_tokens"] == 16_384
    assert trace["overall"][2]["status"] == "done"
    assert trace["overall"][3]["status"] == "done"
    assert trace["overall"][-2]["status"] == "done"
    assert trace["overall"][-1]["status"] == "done"


def test_decision_trace_excludes_previous_execution_with_same_request_hash(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        dashboard,
        "_decision_trace_models",
        lambda: {
            "primary": "primary:model",
            "challenger": "challenger:model",
            "tie_break": "tie:model",
        },
    )
    request = "e" * 64
    trace = dashboard._decision_trace_snapshot(
        [
            {
                "request_sha256": request,
                "role": "ingest_review:challenger",
                "model": "challenger:model",
                "phase": "generate",
                "attempt": 0,
                "started_at": "2026-07-15T12:10:02Z",
            }
        ],
        [
            {
                "kind": "session",
                "timestamp": "2026-07-15T12:00:02Z",
                "request_sha256": request,
                "role": "ingest_review:tie_break",
                "model": "tie:model",
                "ok": True,
            },
            {
                "kind": "session",
                "timestamp": "2026-07-15T12:10:01Z",
                "request_sha256": request,
                "role": "ingest_review:primary",
                "model": "primary:model",
                "ok": True,
                "first_pass_valid": True,
                "repair_turns": 0,
            },
        ],
        None,
        [
            {
                "event_id": "old-tie",
                "kind": "session",
                "timestamp": "2026-07-15T12:00:02Z",
                "request_sha256": request,
                "role": "ingest_review:tie_break",
                "phase": "vote",
                "status": "done",
            },
            {
                "event_id": "old-decision",
                "kind": "decision",
                "timestamp": "2026-07-15T12:00:03Z",
                "request_sha256": request,
                "role": "ingest_review",
                "phase": "decision",
                "status": "done",
            },
            {
                "event_id": "new-primary",
                "kind": "session",
                "timestamp": "2026-07-15T12:10:01Z",
                "request_sha256": request,
                "role": "ingest_review:primary",
                "phase": "vote",
                "status": "done",
            },
            {
                "event_id": "new-challenger",
                "kind": "phase",
                "timestamp": "2026-07-15T12:10:02Z",
                "request_sha256": request,
                "role": "ingest_review:challenger",
                "phase": "generate",
                "status": "active",
            },
        ],
    )

    assert trace["state"] == "active"
    assert [lane["state"] for lane in trace["lanes"]] == [
        "done",
        "active",
        "pending",
    ]
    assert [event["event_id"] for event in trace["events"]] == [
        "new-primary",
        "new-challenger",
    ]


def test_decision_trace_keeps_newer_terminal_request_until_active_updates() -> None:
    active_request = "1" * 64
    completed_request = "2" * 64
    activities = [
        {
            "request_sha256": active_request,
            "role": "ingest_review:primary",
            "model": "primary:model",
            "phase": "generate",
            "started_at": "2026-07-15T12:00:00Z",
            "updated_at": "2026-07-15T12:00:05Z",
        }
    ]
    session = {
        "kind": "session",
        "timestamp": "2026-07-15T12:00:09Z",
        "request_sha256": completed_request,
        "role": "ingest_review:primary",
        "model": "primary:model",
        "ok": True,
        "repair_turns": 0,
    }
    decision = {
        "kind": "decision",
        "timestamp": "2026-07-15T12:00:10Z",
        "request_sha256": completed_request,
        "role": "ingest_review",
        "status": "agreed",
        "pair_agreement": True,
        "vote_count": 2,
        "valid_votes": 2,
    }

    completed_session = dashboard._decision_trace_snapshot(
        activities, [session], None
    )
    activities[0]["updated_at"] = "2026-07-15T12:00:10Z"
    completed_decision = dashboard._decision_trace_snapshot(
        activities, [session, decision], decision
    )

    assert completed_session["request_sha256"] == completed_request
    assert completed_session["state"] == "idle"
    assert completed_session["summary"] == "Local quorum incomplete"
    assert completed_session["overall"][2]["status"] == "pending"
    assert completed_session["overall"][3]["status"] == "pending"
    assert completed_session["active"] is False
    assert completed_decision["request_sha256"] == completed_request
    assert completed_decision["state"] == "agreed"
    assert completed_decision["active"] is False

    activities[0]["updated_at"] = "2026-07-15T12:00:11Z"
    newer_active = dashboard._decision_trace_snapshot(
        activities, [session, decision], decision
    )

    assert newer_active["request_sha256"] == active_request
    assert newer_active["state"] == "active"
    assert newer_active["active"] is True


def test_decision_trace_handoff_skips_completed_requests(
    tmp_path: Path, monkeypatch
) -> None:
    pinned_request = "1" * 64
    completed_request = "2" * 64
    next_request = "3" * 64
    activities = [
        {
            "request_sha256": pinned_request,
            "role": "ingest_review:primary",
            "phase": "generate",
            "started_at": "2026-08-12T00:00:01Z",
            "updated_at": "2026-08-12T00:00:02Z",
        }
    ]
    history = [
        {
            "kind": "decision",
            "timestamp": "2026-08-12T00:00:03Z",
            "request_sha256": completed_request,
            "status": "agreed",
            "pair_agreement": True,
        }
    ]
    monkeypatch.setattr(dashboard, "CHRONOVISOR_ROOT", tmp_path)
    monkeypatch.setattr(dashboard, "_local_consensus_activities", lambda: activities)
    monkeypatch.setattr(dashboard, "_read_json_file", lambda _path: {})
    monkeypatch.setattr(
        dashboard,
        "_read_jsonl_file",
        lambda path, *, limit: history if path.name == "audit.jsonl" else [],
    )

    initial = dashboard._local_consensus_snapshot(next_active=True)
    pinned = dashboard._local_consensus_snapshot(
        preferred_request_sha256=pinned_request,
        next_active=True,
    )

    assert initial["decision_trace"]["request_sha256"] == pinned_request
    assert pinned["decision_trace"]["request_sha256"] == pinned_request

    activities.clear()
    history.append(
        {
            "kind": "decision",
            "timestamp": "2026-08-12T00:00:04Z",
            "request_sha256": pinned_request,
            "status": "agreed",
            "pair_agreement": True,
        }
    )
    terminal = dashboard._local_consensus_snapshot(
        preferred_request_sha256=pinned_request,
        next_active=True,
    )

    assert terminal["decision_trace"]["request_sha256"] == pinned_request
    assert terminal["decision_trace"]["state"] == "agreed"

    history.clear()
    missing = dashboard._local_consensus_snapshot(
        preferred_request_sha256=pinned_request,
        next_active=True,
    )
    assert "decision_trace" not in missing

    activities.append(
        {
            "request_sha256": next_request,
            "role": "model_eval:primary",
            "phase": "generate",
            "started_at": "2026-08-12T00:00:05Z",
            "updated_at": "2026-08-12T00:00:06Z",
        }
    )
    handoff = dashboard._local_consensus_snapshot(
        preferred_request_sha256=pinned_request,
        next_active=True,
    )

    assert handoff["decision_trace"]["request_sha256"] == next_request
    assert handoff["decision_trace"]["active"] is True


def test_decision_trace_pin_survives_the_gap_between_lane_markers() -> None:
    pinned_request = "4" * 64
    other_request = "5" * 64
    phase_event = {
        "event_id": "primary-vote",
        "kind": "phase",
        "timestamp": "2026-08-12T00:00:01Z",
        "request_sha256": pinned_request,
        "role": "ingest_review:primary",
        "phase": "vote",
        "status": "done",
    }
    other_activity = {
        "request_sha256": other_request,
        "role": "model_eval:primary",
        "phase": "generate",
        "started_at": "2026-08-12T00:00:02Z",
        "updated_at": "2026-08-12T00:00:03Z",
    }

    trace = dashboard._decision_trace_snapshot(
        [other_activity],
        [],
        None,
        [phase_event],
        preferred_request_sha256=pinned_request,
    )

    assert trace["request_sha256"] == pinned_request
    assert trace["active"] is False
    assert [event["event_id"] for event in trace["events"]] == ["primary-vote"]


def test_decision_trace_latest_execution_preserves_reused_hash_boundary() -> None:
    request = "3" * 64
    decision = {
        "kind": "decision",
        "timestamp": "2026-07-15T12:00:10Z",
        "request_sha256": request,
        "role": "ingest_review",
        "status": "agreed",
        "pair_agreement": True,
        "vote_count": 2,
        "valid_votes": 2,
    }
    activity = {
        "request_sha256": request,
        "role": "ingest_review:primary",
        "model": "primary:model",
        "phase": "generate",
        "started_at": "2026-07-15T12:00:00Z",
        "updated_at": "2026-07-15T12:00:05Z",
    }
    old_decision_event = {
        "event_id": "old-decision",
        "kind": "decision",
        "timestamp": "2026-07-15T12:00:10Z",
        "request_sha256": request,
        "role": "ingest_review",
        "phase": "decision",
        "status": "done",
    }

    completed = dashboard._decision_trace_snapshot(
        [activity], [decision], decision, [old_decision_event]
    )

    assert completed["state"] == "agreed"
    assert completed["active"] is False

    activity["started_at"] = "2026-07-15T12:00:11Z"
    activity["updated_at"] = "2026-07-15T12:00:12Z"
    new_phase_event = {
        "event_id": "new-generate",
        "kind": "phase",
        "timestamp": "2026-07-15T12:00:12Z",
        "request_sha256": request,
        "role": "ingest_review:primary",
        "phase": "generate",
        "status": "active",
    }
    restarted = dashboard._decision_trace_snapshot(
        [activity],
        [decision],
        decision,
        [old_decision_event, new_phase_event],
    )

    assert restarted["state"] == "active"
    assert restarted["active"] is True
    assert [event["event_id"] for event in restarted["events"]] == [
        "new-generate"
    ]


def test_decision_trace_bounds_repeated_standalone_sessions_with_same_hash() -> None:
    request = "6" * 64
    old = {
        "kind": "session",
        "timestamp": "2026-07-15T12:00:01Z",
        "request_sha256": request,
        "role": "wiki_generation",
        "model": "old:model",
        "ok": True,
    }
    new = {
        **old,
        "timestamp": "2026-07-15T12:01:01Z",
        "model": "new:model",
        "ok": False,
        "failure_class": "transport_error",
    }
    events = [
        {
            "event_id": "old-session",
            "kind": "session",
            "timestamp": old["timestamp"],
            "request_sha256": request,
            "role": "wiki_generation",
            "phase": "vote",
            "status": "done",
        },
        {
            "event_id": "new-generate",
            "kind": "phase",
            "timestamp": "2026-07-15T12:01:00Z",
            "request_sha256": request,
            "role": "wiki_generation",
            "phase": "generate",
            "status": "active",
        },
        {
            "event_id": "new-session",
            "kind": "session",
            "timestamp": new["timestamp"],
            "request_sha256": request,
            "role": "wiki_generation",
            "phase": "vote",
            "status": "error",
        },
    ]

    trace = dashboard._decision_trace_snapshot([], [old, new], None, events)

    assert trace["state"] == "quarantined"
    assert trace["lanes"][0]["model"] == "new:model"
    assert trace["lanes"][0]["detail"] == "transport_error"
    assert [event["event_id"] for event in trace["events"]] == [
        "new-generate",
        "new-session",
    ]


def test_decision_trace_marks_pair_quorum_and_unused_tie_break(monkeypatch) -> None:
    monkeypatch.setattr(
        dashboard,
        "_decision_trace_models",
        lambda: {
            "primary": "primary:model",
            "challenger": "challenger:model",
            "tie_break": "tie:model",
        },
    )
    request = "b" * 64
    decision = {
        "kind": "decision",
        "timestamp": "2026-07-15T12:00:02Z",
        "request_sha256": request,
        "role": "ingest_review",
        "status": "agreed",
        "pair_agreement": True,
        "tie_break_used": False,
        "vote_count": 2,
        "valid_votes": 2,
        "models": ["primary:model", "challenger:model"],
    }
    sessions = [
        {
            "kind": "session",
            "timestamp": f"2026-07-15T12:00:0{index}Z",
            "request_sha256": request,
            "role": f"ingest_review:{role}",
            "model": f"{role}:model",
            "ok": True,
            "first_pass_valid": True,
            "repair_turns": 0,
            "think": "medium",
        }
        for index, role in enumerate(("primary", "challenger"))
    ]
    trace = dashboard._decision_trace_snapshot([], [*sessions, decision], decision)

    assert trace["state"] == "agreed"
    assert trace["summary"] == "2/2 pair agreement"
    assert trace["quorum_attempted"] is True
    assert trace["vote_count"] == 2
    assert trace["valid_votes"] == 2
    assert trace["pair_agreement"] is True
    assert trace["tie_break_used"] is False
    assert trace["lanes"][0]["state"] == "done"
    assert trace["lanes"][1]["state"] == "done"
    assert trace["lanes"][2]["state"] == "skipped"
    assert trace["lanes"][2]["detail"] == "Primary pair agreed"
    assert [lane["think"] for lane in trace["lanes"]] == [
        "medium",
        "medium",
        "—",
    ]
    assert trace["overall"][4]["status"] == "done"
    assert trace["overall"][5]["status"] == "done"
    assert trace["overall"][6]["status"] == "done"

    held = {
        **decision,
        "status": "quarantined",
        "pair_agreement": False,
        "vote_count": 2,
        "valid_votes": 1,
        "quarantine_reason": "fewer_than_two_valid_local_votes",
    }
    held_trace = dashboard._decision_trace_snapshot([], [held], held)
    assert held_trace["lanes"][2]["state"] == "skipped"
    assert held_trace["lanes"][2]["detail"] == "Decision held without tie-break"


def test_decision_trace_hides_legacy_and_artifact_replay_reasoning(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        dashboard,
        "_decision_trace_models",
        lambda: {
            "primary": "primary:model",
            "challenger": "challenger:model",
            "tie_break": "tie:model",
        },
    )
    request = "f" * 64
    legacy_session = {
        "kind": "session",
        "request_sha256": request,
        "role": "primary",
        "model": "primary:model",
        "ok": True,
    }
    legacy = dashboard._decision_trace_snapshot([], [legacy_session], None)

    assert [lane["think"] for lane in legacy["lanes"]] == ["—", "—", "—"]

    observed_session = {
        **legacy_session,
        "think": "medium",
    }
    replay = {
        "kind": "decision_artifact_replay",
        "request_sha256": request,
        "role": "routine",
        "status": "agreed",
        "models": ["primary:model", "challenger:model"],
    }
    artifact = dashboard._decision_trace_snapshot(
        [], [observed_session, replay], replay
    )

    assert [lane["think"] for lane in artifact["lanes"]] == ["—", "—", "—"]
    assert artifact["quorum_attempted"] is True
    assert artifact["vote_count"] is None
    assert artifact["valid_votes"] is None
    assert artifact["pair_agreement"] is None


@pytest.mark.parametrize(
    "failure_class,quarantine_reason",
    (
        ("input_invalid", "structured_request_preflight_failed:input_invalid"),
        ("route_configuration_invalid", "router_config_invalid:route"),
        ("schema_invalid", "structured_request_preflight_failed:schema_invalid"),
        ("local_resource_quarantined", "decision_runner_does_not_fit_reserved_memory"),
        ("transport_error", "structured_transport_failed"),
    ),
)
def test_decision_trace_keeps_early_terminal_failures_before_quorum(
    failure_class: str,
    quarantine_reason: str,
) -> None:
    decision = {
        "kind": "decision",
        "request_sha256": "6" * 64,
        "role": "ingest_reconciliation",
        "status": "quarantined",
        "failure_class": failure_class,
        "quarantine_reason": quarantine_reason,
        "vote_count": 0,
        "valid_votes": 0,
        "pair_agreement": False,
        "tie_break_used": False,
    }

    trace = dashboard._decision_trace_snapshot([], [decision], decision)

    assert trace["quorum_attempted"] is False
    assert trace["vote_count"] == 0
    assert trace["valid_votes"] == 0
    assert trace["pair_agreement"] is False
    assert trace["tie_break_used"] is False
    assert [lane["state"] for lane in trace["lanes"]] == [
        "pending",
        "pending",
        "pending",
    ]
    assert [step["status"] for step in trace["overall"]] == [
        "done",
        "done",
        "pending",
        "pending",
        "pending",
        "skipped",
        "skipped",
    ]
    assert trace["context_tokens"] is None


def test_decision_trace_explains_semantic_quality_and_resource_holds() -> None:
    semantic = dashboard._decision_trace_outcome(
        {
            "quarantine_reason": "local_models_did_not_reach_two_vote_quorum",
            "failure_class": "local_consensus_failed",
        },
        trace_state="quarantined",
        task_role="ingest_reconciliation",
    )
    quality = dashboard._decision_trace_outcome(
        {"quarantine_reason": "fewer_than_two_valid_local_votes"},
        trace_state="quarantined",
        task_role="ingest_reconciliation",
    )
    resource = dashboard._decision_trace_outcome(
        {
            "quarantine_reason": "decision_runner_does_not_fit_reserved_memory",
            "failure_class": "local_resource_quarantined",
        },
        trace_state="quarantined",
        task_role="ingest_reconciliation",
    )

    assert semantic == {
        "kind": "semantic_hold",
        "reason": "Valid models disagreed",
        "data": "Raw retained",
        "next": "Recheck after model or policy change",
        "code": "local_models_did_not_reach_two_vote_quorum",
    }
    assert quality["kind"] == "quality_hold"
    assert quality["reason"] == "Too few valid model votes"
    assert quality["data"] == "Raw retained"
    assert resource["kind"] == "operational_hold"
    assert resource["reason"] == "Model memory could not be verified"
    assert resource["next"] == "Retry when capacity recovers"


def test_decision_trace_marks_only_artifact_publish_failure_after_artifact() -> None:
    decision = {
        "kind": "decision",
        "request_sha256": "7" * 64,
        "role": "ingest_reconciliation",
        "status": "quarantined",
        "quarantine_reason": "canonical_decision_artifact_publish_failed",
        "failure_class": "decision_artifact_invalid",
        "valid_votes": 2,
    }

    trace = dashboard._decision_trace_snapshot([], [decision], decision)

    assert trace["overall"][4]["status"] == "done"
    assert trace["overall"][-2]["status"] == "error"
    assert trace["overall"][-1]["status"] == "skipped"
    assert trace["outcome"]["reason"] == "Decision artifact seal failed"
    assert trace["outcome"]["code"] == "canonical_decision_artifact_publish_failed"
    assert all(lane["context_tokens"] is None for lane in trace["lanes"])

    no_quorum_decision = {
        **decision,
        "quarantine_reason": "local_models_did_not_reach_two_vote_quorum",
        "failure_class": "local_consensus_failed",
    }
    no_quorum = dashboard._decision_trace_snapshot(
        [], [no_quorum_decision], no_quorum_decision
    )
    assert no_quorum["state"] == "quarantined"
    assert no_quorum["outcome"]["kind"] == "semantic_hold"
    assert no_quorum["outcome"]["code"] == (
        "local_models_did_not_reach_two_vote_quorum"
    )
    assert no_quorum["overall"][-2]["status"] == "skipped"
    assert no_quorum["overall"][-1]["status"] == "skipped"


def test_decision_trace_explains_lane_policy_veto_bypass() -> None:
    decision = {
        "kind": "decision",
        "request_sha256": "e" * 64,
        "role": "recall_auto_apply",
        "status": "agreed",
        "tie_break_used": True,
        "conservative_veto_fired": True,
        "conservative_veto_bypassed_by_lane_policy": True,
        "dissent_effect_class": "unclassifiable",
    }

    outcome = dashboard._decision_trace_outcome(
        decision,
        trace_state="agreed",
        task_role="recall_auto_apply",
    )
    trace = dashboard._decision_trace_snapshot([], [decision], decision)

    assert outcome == {
        "kind": "approved",
        "reason": "Lane policy bypassed conservative veto",
        "data": "Dissent effect: unclassifiable",
        "next": "Mutation may proceed",
        "code": "conservative_veto_bypassed_by_lane_policy",
    }
    assert trace["summary"] == (
        "2/3 quorum · conservative veto bypassed by lane policy"
    )
    assert trace["tie_break_used"] is True
    assert [lane["state"] for lane in trace["lanes"]] == [
        "pending",
        "pending",
        "pending",
    ]


def test_local_consensus_snapshot_removes_reused_pid_marker(
    tmp_path: Path,
    monkeypatch,
) -> None:
    chronovisor_root = tmp_path / "wiki"
    active_dir = chronovisor_root / "runtime" / "local-consensus" / "active"
    active_dir.mkdir(parents=True)
    marker_path = active_dir / "reused.json"
    marker_path.write_text(
        json.dumps(
            {
                "request_sha256": "a" * 64,
                "role": "primary",
                "model": "ornith:test",
                "started_at": "2026-07-11T17:00:00",
                "pid": 4242,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(dashboard, "CHRONOVISOR_ROOT", chronovisor_root)
    monkeypatch.setattr(runtime_status, "_pid_is_alive", lambda _pid: True)
    monkeypatch.setattr(
        dashboard,
        "_process_started_at",
        lambda _pid: datetime.fromisoformat("2026-07-11T18:00:00"),
    )

    snapshot = dashboard._local_consensus_snapshot()

    assert snapshot["active"] is False
    assert not marker_path.exists()


def test_frontier_activity_snapshot_removes_reused_pid_marker(
    tmp_path: Path,
    monkeypatch,
) -> None:
    chronovisor_root = tmp_path / "wiki"
    active_dir = chronovisor_root / "runtime" / "frontier-reviews" / "active"
    active_dir.mkdir(parents=True)
    marker_path = active_dir / "reused.json"
    marker_path.write_text(
        json.dumps(
            {
                "review_id": "review-1",
                "started_at": "2026-07-11T17:00:00",
                "pid": 4242,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(dashboard, "CHRONOVISOR_ROOT", chronovisor_root)
    monkeypatch.setattr(runtime_status, "_pid_is_alive", lambda _pid: True)
    monkeypatch.setattr(
        dashboard,
        "_process_started_at",
        lambda _pid: datetime.fromisoformat("2026-07-11T18:00:00"),
    )

    snapshot = dashboard._frontier_activity_snapshot()

    assert snapshot["active"] is False
    assert not marker_path.exists()


def test_frontier_repair_snapshot_uses_guard_ledger_and_dead_owner_is_inactive(
    tmp_path: Path,
    monkeypatch,
) -> None:
    chronovisor_root = tmp_path / "wiki"
    root = chronovisor_root / "runtime" / "frontier-repair"
    root.mkdir(parents=True)
    incident_id = "incident-1"
    owner_process_started_at = datetime.fromisoformat("2026-07-11T11:00:00")
    state = {
        "schema_version": 1,
        "active_incident_id": incident_id,
        "incidents": {
            incident_id: {
                "incident_id": incident_id,
                "status": "started",
                "reserved_at": "2026-07-11T11:59:00Z",
                "started_at": datetime.now().astimezone().isoformat(),
                "finished_at": None,
                "lease_expires_at": (
                    datetime.now().astimezone() + timedelta(hours=1)
                ).isoformat(),
                "owner_pid": os.getpid(),
                "owner_process_started_at": owner_process_started_at.isoformat(),
                "pid": os.getpid(),
                "fingerprint_key": "b" * 64,
                "evidence": {
                    "component": "ingest",
                    "failure_class": "adapter_crash",
                    "notes": {"raw_payload": "must stay private"},
                },
            }
        },
    }
    (root / "state.json").write_text(json.dumps(state), encoding="utf-8")
    (root / "events.jsonl").write_text(
        json.dumps(
            {
                "sequence": 1,
                "timestamp": "2026-07-11T12:00:00Z",
                "event": "incident_started",
                "incident_id": incident_id,
                "private_details": "must stay private",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(dashboard, "CHRONOVISOR_ROOT", chronovisor_root)
    monkeypatch.setattr(runtime_status, "_pid_is_alive", lambda _pid: True)
    monkeypatch.setattr(
        dashboard,
        "_process_started_at",
        lambda _pid: owner_process_started_at,
    )

    active = dashboard._frontier_repair_snapshot()

    assert active["active"] is True
    assert active["summary"]["starts_24h"] == 1
    assert active["summary"]["counts"] == {"started": 1}
    serialized = json.dumps(active, ensure_ascii=False)
    assert "must stay private" not in serialized

    monkeypatch.setattr(runtime_status, "_pid_is_alive", lambda _pid: False)
    dead = dashboard._frontier_repair_snapshot()
    assert dead["active"] is False
    assert dead["stale_active_incident"] is True


def test_frontier_repair_snapshot_rejects_reused_pid_and_legacy_identity(
    tmp_path: Path,
    monkeypatch,
) -> None:
    chronovisor_root = tmp_path / "wiki"
    root = chronovisor_root / "runtime" / "frontier-repair"
    root.mkdir(parents=True)
    incident_id = "incident-1"
    incident = {
        "incident_id": incident_id,
        "status": "started",
        "reserved_at": "2026-07-11T11:59:00Z",
        "started_at": "2026-07-11T12:00:00Z",
        "finished_at": None,
        "lease_expires_at": (
            datetime.now().astimezone() + timedelta(hours=1)
        ).isoformat(),
        "owner_pid": 4242,
        "owner_process_started_at": "2026-07-11T11:00:00",
        "pid": 4243,
        "fingerprint_key": "b" * 64,
        "evidence": {},
    }
    state = {
        "schema_version": 1,
        "active_incident_id": incident_id,
        "incidents": {incident_id: incident},
    }
    state_path = root / "state.json"
    state_path.write_text(json.dumps(state), encoding="utf-8")
    monkeypatch.setattr(dashboard, "CHRONOVISOR_ROOT", chronovisor_root)
    monkeypatch.setattr(runtime_status, "_pid_is_alive", lambda _pid: True)
    monkeypatch.setattr(
        dashboard,
        "_process_started_at",
        lambda _pid: datetime.fromisoformat("2026-07-11T18:00:00"),
    )

    reused = dashboard._frontier_repair_snapshot()

    assert reused["active"] is False
    assert reused["stale_active_incident"] is True

    del incident["owner_process_started_at"]
    state_path.write_text(json.dumps(state), encoding="utf-8")
    monkeypatch.setattr(
        dashboard,
        "_process_started_at",
        lambda _pid: datetime.fromisoformat("2026-07-11T11:00:00"),
    )

    legacy = dashboard._frontier_repair_snapshot()

    assert legacy["active"] is False
    assert legacy["stale_active_incident"] is True


def test_frontier_repair_snapshot_preserves_unavailable_identity_until_lease(
    tmp_path: Path,
    monkeypatch,
) -> None:
    chronovisor_root = tmp_path / "wiki"
    root = chronovisor_root / "runtime" / "frontier-repair"
    root.mkdir(parents=True)
    incident_id = "incident-unavailable"
    now = datetime.now().astimezone()
    incident = {
        "incident_id": incident_id,
        "status": "started",
        "reserved_at": (now - timedelta(minutes=2)).isoformat(),
        "started_at": (now - timedelta(minutes=1)).isoformat(),
        "finished_at": None,
        "lease_expires_at": (now + timedelta(hours=1)).isoformat(),
        "owner_pid": 4242,
        "owner_process_started_at": (now - timedelta(hours=1)).isoformat(),
        "pid": 4243,
        "fingerprint_key": "c" * 64,
        "evidence": {},
    }
    state = {
        "schema_version": 1,
        "active_incident_id": incident_id,
        "incidents": {incident_id: incident},
    }
    state_path = root / "state.json"
    state_path.write_text(json.dumps(state), encoding="utf-8")
    monkeypatch.setattr(dashboard, "CHRONOVISOR_ROOT", chronovisor_root)
    monkeypatch.setattr(runtime_status, "_pid_is_alive", lambda _pid: True)
    monkeypatch.setattr(dashboard, "_process_started_at", lambda _pid: None)

    unavailable = dashboard._frontier_repair_snapshot()

    assert unavailable["active"] is True
    assert unavailable["stale_active_incident"] is False
    assert unavailable["active_incident"]["owner_alive"] is True
    assert unavailable["active_incident"]["owner_identity_status"] == "unavailable"
    assert unavailable["active_incident"]["lease_expired"] is False

    incident["lease_expires_at"] = (now - timedelta(seconds=1)).isoformat()
    state_path.write_text(json.dumps(state), encoding="utf-8")
    expired = dashboard._frontier_repair_snapshot()

    assert expired["active"] is False
    assert expired["stale_active_incident"] is True
    assert expired["active_incident"]["owner_identity_status"] == "unavailable"
    assert expired["active_incident"]["lease_expired"] is True


def test_dashboard_static_labels_routine_review_as_local_consensus() -> None:
    app = "".join(
        (dashboard.STATIC_DIR / name).read_text(encoding="utf-8")
        for name in ("app.js", "app-renderer.js", "app-client.js")
    )
    page = (dashboard.STATIC_DIR / "index.html").read_text(encoding="utf-8")
    style = (dashboard.STATIC_DIR / "style.css").read_text(encoding="utf-8")

    assert "Frontier reviewing" not in app
    assert "Local consensus reviewing" in app
    assert "Local model evaluation" in app
    assert '"local-consensus-review": "Local review"' in app
    assert "function stageMetricLabel(value)" in app
    assert "els.stage.textContent = stageMetricLabel(stageValue);" in app
    assert "els.stage.title = stageValue;" in app
    assert "correction uncertainty" in app
    assert "batch.active === true ? 1 : 0" in app
    assert "batch.total ? 1 : 0" not in app
    assert "conservative_veto_fired" in app
    assert "conservative_veto_bypassed_by_lane_policy" in app
    assert "dissent_effect_classes" in app
    assert "model_conservative_vote_rates" in app
    assert "conservative votes" in app
    assert 'id="local-consensus"' in page
    assert 'id="frontier-repair"' in page
    assert "Frontier Repair" in page
    assert 'id="decision-trace-panel"' in page
    assert 'data-decision-lane="primary"' in page
    assert 'data-decision-lane="challenger"' in page
    assert 'data-decision-lane="tie_break"' in page
    assert 'id="decision-outcome-reason"' in page
    assert 'id="decision-outcome-data"' in page
    assert 'id="decision-outcome-next"' in page
    assert 'id="decision-transition-state"' in page
    assert 'id="decision-transition-feed"' in page
    assert 'id="lan-share-button"' in page
    assert "Runnable Work" in page
    assert 'id="held-value"' in page
    assert 'id="held-sub"' in page
    assert 'id="processing-panel"' in page
    assert 'id="processing-lanes"' in page
    assert 'id="processing-connection"' in page
    assert "<span>Page changes</span>" in page
    assert "${pageChanges} changes" in app
    assert "${pages} pages" not in app
    assert 'class="decision-trace-scroll" id="decision-trace-scroll"' in page
    assert 'id="decision-trace-harness" viewBox="0 0 1500 650"' in page
    assert ".decision-trace-scroll {\n  width: 100%;\n  overflow-x: auto;" in style
    assert ".decision-trace-harness {\n  display: block;\n  width: 1400px;" in style
    assert "height: var(--panel-height);" in style
    assert "#model-lab-panel" in style
    assert "#model-panel {\n  height: auto;\n  min-height: 500px;" in style
    assert (
        "#model-panel .model-grid {\n"
        "  flex: 0 0 auto;\n"
        "  overflow: visible;"
    ) in style
    assert "min-height: 764px;" in style
    assert "min-height: 1084px;" in style
    assert "#stage-value" in style
    assert "text-overflow: ellipsis;" in style
    assert "white-space: nowrap;" in style
    assert ".decision-outcome-facts" in style
    assert ".decision-transition-event.current" in style
    assert "function updateDecisionSvgHarness" in app
    assert "const decisionTracePlayback" in app
    assert "const ACTIVE_DECISION_REFRESH_DELAY_MS = 800" in app
    assert 'fetch("/api/local-consensus"' in app
    assert 'new EventSource("/api/activity-stream")' in app
    assert 'fetch("/api/activity"' in app
    assert "function renderProcessingActivity" in app
    assert "els.pending.textContent = fmt(ready);" in app
    assert "els.held.textContent = fmt(held);" in app
    assert "${semanticDeferred} semantic · ${operationalDeferred} operational" in app
    assert "grid-template-columns: repeat(5, minmax(0, 1fr));" in style
    assert "No synthetic progress" in app
    assert "lane.think = fmt(event.think, \"—\").toLowerCase();" in app
    assert ".decision-trace-panel" in style
    assert ".processing-lane.active" in style
    assert "processing-electric-pulse" in style
    assert "grid-template-columns: repeat(6, minmax(0, 1fr));" in style
    assert "--processing-step-count" not in app
    assert "place-items: start;" in style
    assert 'lane.recent ? "PULSE" : "ACTIVE"' in app
    assert 'details.push("just completed")' in app
    assert (
        page.index('/static/app.js')
        < page.index('/static/app-renderer.js')
        < page.index('/static/app-client.js')
    )


def test_processing_activity_rejects_out_of_order_poll_after_newer_stream() -> None:
    renderer = (dashboard.STATIC_DIR / "app-renderer.js").read_text(encoding="utf-8")
    scenario = f"""
const vm = require("node:vm");
class FakeNode {{
  constructor(tag = "div") {{
    this.tag = tag;
    this.dataset = {{}};
    this.children = [];
    this.className = "";
    this.attributes = {{}};
    this.textContent = "";
    this.title = "";
    this.tabIndex = -1;
    this.parent = null;
  }}
  appendChild(child) {{
    if (child.parent) child.parent.children = child.parent.children.filter((item) => item !== child);
    child.parent = this;
    this.children.push(child);
    return child;
  }}
  append(...children) {{ children.forEach((child) => this.appendChild(child)); }}
  addEventListener() {{}}
  setAttribute(key, value) {{ this.attributes[key] = String(value); }}
  remove() {{
    if (this.parent) this.parent.children = this.parent.children.filter((item) => item !== this);
    this.parent = null;
  }}
  matches(selector) {{
    if (selector.startsWith(".")) return this.className.split(/\\s+/).includes(selector.slice(1));
    return this.tag === selector;
  }}
  querySelectorAll(selector) {{
    const matches = [];
    const visit = (node) => node.children.forEach((child) => {{
      if (child.matches(selector)) matches.push(child);
      visit(child);
    }});
    visit(this);
    return matches;
  }}
  querySelector(selector) {{ return this.querySelectorAll(selector)[0] || null; }}
}}
const processingLanes = new FakeNode("main");
const document = {{
  body: {{ dataset: {{}} }},
  createElement: (tag) => new FakeNode(tag),
  visibilityState: "visible",
}};
const els = {{ processingLanes, processingPanel: {{ dataset: {{}} }} }};
const window = {{ matchMedia: () => ({{ matches: false }}) }};
const sandbox = {{ window, document, els, STAGE_METRIC_LABELS: {{}} }};
vm.createContext(sandbox);
vm.runInContext({json.dumps(renderer)}, sandbox);
const renderProcessingActivity = window.__chronovisorDashboardTest.renderProcessingActivity;
const keys = ["ingest", "recall", "audit", "improve", "repair", "typed_graph"];
const payload = (generatedAt, revision, state) => ({{
  generated_at: generatedAt,
  revision,
  active_count: state === "active" ? keys.length : 0,
  lanes: keys.map((key) => ({{
    key,
    label: key,
    state,
    current_step: "work",
    steps: [{{ key: "work", label: "Work", status: state }}],
  }})),
}});
const acceptedStream = renderProcessingActivity(
  payload("2026-08-12T12:00:10.000Z", "stream-new", "active")
);
const acceptedOldPoll = renderProcessingActivity(
  payload("2026-08-12T12:00:09.000Z", "poll-old", "idle")
);
const acceptedDuplicate = renderProcessingActivity(
  payload("2026-08-12T12:00:11.000Z", "stream-new", "idle")
);
const acceptedBetween = renderProcessingActivity(
  payload("2026-08-12T12:00:10.500Z", "poll-between", "idle")
);
process.stdout.write(JSON.stringify({{
  acceptedStream,
  acceptedOldPoll,
  acceptedDuplicate,
  acceptedBetween,
  revision: document.body.dataset.processingRevision,
  lanes: processingLanes.querySelectorAll(".processing-lane").map((row) => ({{
    key: row.dataset.processingLane,
    state: row.className,
  }})),
}}));
"""

    completed = _run_node_scenario(scenario)
    result = json.loads(completed.stdout)

    assert result == {
        "acceptedStream": True,
        "acceptedOldPoll": False,
        "acceptedDuplicate": False,
        "acceptedBetween": False,
        "revision": "stream-new",
        "lanes": [
            {"key": key, "state": "processing-lane active"}
            for key in ("ingest", "recall", "audit", "improve", "repair", "typed_graph")
        ],
    }


def test_decision_trace_lane_rails_start_at_source_node_edges() -> None:
    page = (dashboard.STATIC_DIR / "index.html").read_text(encoding="utf-8")
    for key in (
        "pair-artifact-join",
        "tie_break-quorum",
        "quorum-artifact-join",
        "quorum-hold",
    ):
        assert f'data-path-key="{key}"' in page
    for key in (
        "pair-artifact",
        "tie_break-artifact",
        "pair-hold",
        "tie_break-hold",
        "pair-quorum",
        "quorum-artifact",
        "quorum-artifact-trunk",
        "artifact-input",
    ):
        assert f'data-path-key="{key}"' not in page
    assert 'data-trace-key="quorum"' in page
    common_rails = {
        "trigger-load": "M240 0 H374",
        "load-context": "M394 0 H555",
        "context-generate": "M575 0 H733",
        "generate-validate": "M753 0 H900",
    }

    for lane in ("primary", "challenger", "tie_break"):
        rails = {
            **common_rails,
            "validate-vote": (
                "M920 0 H1086 Q1096 0 1096 10 V25"
                if lane == "primary"
                else "M920 0 H1086"
            ),
        }
        lane_markup = page.split(f'data-decision-lane="{lane}"', 1)[1].split(
            '<g class="decision-lane-steps"', 1
        )[0]
        assert lane_markup.count("trace-lane-rail") == len(rails)
        for key, geometry in rails.items():
            assert f'data-lane-path="{key}" d="{geometry}"' in lane_markup

    primary_markup = page.split('data-decision-lane="primary"', 1)[1].split(
        'data-decision-lane="challenger"', 1
    )[0]
    assert (
        'data-decision-lane-step="vote" transform="translate(1096 35)"><circle r="10"></circle><text x="-25" y="-17">Vote</text>'
        in primary_markup
    )


def test_dashboard_reuses_decision_trace_poll_for_live_consensus_status() -> None:
    app = "".join(
        (dashboard.STATIC_DIR / name).read_text(encoding="utf-8")
        for name in ("app.js", "app-renderer.js", "app-client.js")
    )
    summary_helper = app.split(
        "function renderLocalConsensusSummary(status)", 1
    )[1].split("function render(snapshot)", 1)[0]
    render_block = app.split("function render(snapshot)", 1)[1].split(
        "let refreshInFlight", 1
    )[0]
    live_helper = app.split("function renderLiveConsensus(consensus)", 1)[1].split(
        "async function refreshDecisionTrace", 1
    )[0]
    refresh_block = app.split("async function refreshDecisionTrace()", 1)[1].split(
        "async function decisionTraceRefreshLoop", 1
    )[0]

    assert "let latestRenderedStatus = null;" in app
    assert "latestRenderedStatus = status;" in render_block
    assert "renderLocalConsensusSummary(status);" in render_block
    assert "status.decision_policies" in summary_helper
    assert "...latestRenderedStatus" in live_helper
    assert "...currentConsensus" in live_helper
    assert "...liveConsensus" in live_helper
    assert "...(currentConsensus.summary || {})" in live_helper
    assert "...(liveConsensus.summary || {})" in live_helper
    assert "local_consensus: mergedConsensus" in live_helper
    assert "renderDecisionTrace(mergedConsensus);" in live_helper
    assert (
        'const underlyingState = String(latestRenderedStatus.state || "").toLowerCase();'
        in live_helper
    )
    assert '["error", "blocked"].includes(underlyingState)' in live_helper
    assert 'mergedConsensus.active ? "running" : latestRenderedStatus.state' in live_helper
    assert "setState(displayState);" in live_helper
    assert "renderLocalConsensusSummary(latestRenderedStatus);" in live_helper
    assert "renderWorkStatus(latestRenderedStatus);" in live_helper
    assert 'fetch("/api/local-consensus"' in refresh_block
    assert "renderLiveConsensus(consensus);" in refresh_block
    assert "void refreshLiveModelStatus(consensus.activities || []);" in refresh_block
    assert refresh_block.count("fetch(") == 1
    assert 'fetch("/api/model-status"' in app
    assert "renderLiveModelStatus(await response.json(), activities);" in app
    assert "MODEL_ACTIVITY_LABELS[activity.phase]" in app


def test_decision_trace_poll_pins_until_terminal_render() -> None:
    client = (dashboard.STATIC_DIR / "app-client.js").read_text(encoding="utf-8")
    refresh = "async function refreshDecisionTrace()" + client.split(
        "async function refreshDecisionTrace()", 1
    )[1].split("async function decisionTraceRefreshLoop", 1)[0]
    pinned_request = "1" * 64
    newer_request = "2" * 64
    scenario = f"""
const vm = require("node:vm");
const urls = [];
const rendered = [];
const responses = [
  {{ decision_trace: {{ request_sha256: "{pinned_request}", state: "active" }} }},
  {{ decision_trace: {{ request_sha256: "{pinned_request}", state: "agreed" }} }},
  {{ decision_trace: {{ request_sha256: "{newer_request}", state: "active" }} }},
];
const sandbox = {{
  AbortController,
  encodeURIComponent,
  rendered,
  window: {{ setTimeout: () => 1, clearTimeout: () => {{}} }},
  fetch: async (url) => {{
    urls.push(url);
    return {{ ok: true, json: async () => ({{ local_consensus: responses.shift() }}) }};
  }},
}};
vm.createContext(sandbox);
vm.runInContext(
  `let decisionRefreshInFlight = false;
let decisionTracePinnedRequest = "";
let nextDecisionRefreshDelayMs = 0;
const DECISION_REFRESH_TIMEOUT_MS = 2500;
const ACTIVE_DECISION_REFRESH_DELAY_MS = 800;
const IDLE_DECISION_REFRESH_DELAY_MS = 2500;
const renderLiveConsensus = (consensus) => rendered.push(consensus.decision_trace.state);
const refreshLiveModelStatus = () => Promise.resolve();\n`
  + {json.dumps(refresh)}
  + `\nthis.__test = {{ refreshDecisionTrace, pin: () => decisionTracePinnedRequest }};`,
  sandbox,
);
(async () => {{
  await sandbox.__test.refreshDecisionTrace();
  const activePin = sandbox.__test.pin();
  await sandbox.__test.refreshDecisionTrace();
  const terminalPin = sandbox.__test.pin();
  await sandbox.__test.refreshDecisionTrace();
  process.stdout.write(JSON.stringify({{
    urls,
    rendered,
    pins: [activePin, terminalPin, sandbox.__test.pin()],
  }}));
}})();
"""

    result = json.loads(_run_node_scenario(scenario).stdout)

    assert result == {
        "urls": [
            "/api/local-consensus?next=active",
            f"/api/local-consensus?next=active&request_sha256={pinned_request}",
            f"/api/local-consensus?next=active&request_sha256={pinned_request}",
        ],
        "rendered": ["active", "agreed", "active"],
        "pins": [pinned_request, pinned_request, newer_request],
    }


def test_live_model_status_survives_later_stale_full_render() -> None:
    renderer = (dashboard.STATIC_DIR / "app-renderer.js").read_text(encoding="utf-8")
    hooks = """
const seen = [];
const noop = () => {};
setState = noop;
renderWorkStatus = noop;
renderDecisionTrace = noop;
renderLlm = noop;
renderLocalConsensusSummary = noop;
renderSelfHeal = noop;
renderRecall = noop;
renderRecallImprovement = noop;
renderSaveHistory = noop;
renderKnowledgeMix = noop;
renderLibrarian = noop;
renderHealth = noop;
renderModelLab = noop;
renderEvents = noop;
drawLineChart = noop;
drawBatchChart = noop;
renderModelStatus = (status, _failures, activities) => seen.push({
  status: status.models[0].status,
  activity: activities[0]?.phase || null,
});
this.__test = { renderLiveModelStatus, render, seen };
"""
    scenario = f"""
const vm = require("node:vm");
const element = () => ({{ textContent: "", title: "" }});
const els = new Proxy({{}}, {{
  get(target, key) {{
    if (!(key in target)) target[key] = element();
    return target[key];
  }},
}});
const sandbox = {{
  window: {{ matchMedia: () => ({{ matches: false }}) }},
  document: {{ visibilityState: "visible" }},
  els,
  latestRenderedStatus: null,
  STAGE_METRIC_LABELS: {{}},
}};
vm.createContext(sandbox);
vm.runInContext({json.dumps(renderer + hooks)}, sandbox);
sandbox.__test.renderLiveModelStatus(
  {{ model_status: {{ models: [{{ status: "loaded" }}], summary: {{}} }} }},
  [{{ phase: "generate" }}],
);
sandbox.__test.render({{
  status: {{}},
  model_status: {{ models: [{{ status: "ready" }}], summary: {{}} }},
}});
process.stdout.write(JSON.stringify(sandbox.__test.seen));
"""

    completed = _run_node_scenario(scenario)

    assert json.loads(completed.stdout) == [
        {"status": "loaded", "activity": "generate"},
        {"status": "loaded", "activity": "generate"},
    ]


def test_live_consensus_survives_later_stale_full_render() -> None:
    renderer = (dashboard.STATIC_DIR / "app-renderer.js").read_text(encoding="utf-8")
    client = (dashboard.STATIC_DIR / "app-client.js").read_text(encoding="utf-8")
    live_helper = "function renderLiveConsensus(consensus)" + client.split(
        "function renderLiveConsensus(consensus)", 1
    )[1].split("async function refreshLiveModelStatus", 1)[0]
    hooks = """
const seen = [];
const noop = () => {};
setState = noop;
renderWorkStatus = noop;
renderLlm = noop;
renderLocalConsensusSummary = noop;
renderSelfHeal = noop;
renderRecall = noop;
renderRecallImprovement = noop;
renderSaveHistory = noop;
renderKnowledgeMix = noop;
renderLibrarian = noop;
renderHealth = noop;
renderModelStatus = noop;
renderModelLab = noop;
renderEvents = noop;
drawLineChart = noop;
drawBatchChart = noop;
renderDecisionTrace = (consensus) => {
  const trace = consensus.decision_trace || {};
  const steps = decisionTimelineSteps(trace).map((step) => step.label);
  decisionTracePlayback.request = String(trace.request_sha256 || "");
  decisionTracePlayback.current = { request_sha256: trace.request_sha256, steps };
  seen.push({ request: trace.request_sha256, steps });
};
"""
    scenario = f"""
const vm = require("node:vm");
const element = () => ({{ textContent: "", title: "" }});
const els = new Proxy({{}}, {{
  get(target, key) {{
    if (!(key in target)) target[key] = element();
    return target[key];
  }},
}});
const sandbox = {{
  window: {{ matchMedia: () => ({{ matches: false }}) }},
  document: {{ visibilityState: "visible" }},
  els,
  latestRenderedStatus: {{ state: "running", local_consensus: {{}} }},
  STAGE_METRIC_LABELS: {{}},
}};
vm.createContext(sandbox);
vm.runInContext(
  {json.dumps(renderer + hooks + live_helper)}
    + "\\nthis.__test = {{ render, renderLiveConsensus, seen, playback: () => decisionTracePlayback }};",
  sandbox,
);
const live = {{
  active: true,
  decision_trace: {{
    request_sha256: "f749",
    active: true,
    events: [{{
      event_id: "live-generate",
      kind: "phase",
      lane: "challenger",
      phase: "generate",
      status: "active",
    }}],
  }},
}};
const stale = {{
  active: true,
  decision_trace: {{
    request_sha256: "8103",
    active: true,
    events: [{{
      event_id: "stale-validate",
      kind: "phase",
      lane: "primary",
      phase: "validate",
      status: "active",
    }}],
  }},
}};
sandbox.__test.renderLiveConsensus(live);
sandbox.__test.render({{
  status: {{ state: "running", local_consensus: stale }},
  local_consensus: stale,
}});
process.stdout.write(JSON.stringify({{
  seen: sandbox.__test.seen,
  playback: sandbox.__test.playback(),
}}));
"""

    completed = _run_node_scenario(scenario)
    result = json.loads(completed.stdout)

    assert [row["request"] for row in result["seen"]] == ["f749", "f749"]
    assert result["seen"][-1]["steps"] == ["Packet", "Challenger Generate #1"]
    assert result["playback"]["request"] == "f749"
    assert result["playback"]["current"]["steps"] == [
        "Packet",
        "Challenger Generate #1",
    ]


def test_processing_lane_trace_selection_maps_all_workflows_and_fallbacks() -> None:
    renderer = (dashboard.STATIC_DIR / "app-renderer.js").read_text(encoding="utf-8")
    scenario = f"""
const vm = require("node:vm");
const sandbox = {{ window: {{ matchMedia: () => ({{ matches: false }}) }} }};
vm.createContext(sandbox);
vm.runInContext(
  {json.dumps(renderer)}
    + "\\nthis.__test = {{ processingLaneForTrace, lanes: latestProcessingLanes }};",
  sandbox,
);
const laneKeys = ["ingest", "recall", "audit", "improve", "repair", "typed_graph"];
laneKeys.forEach((key) => sandbox.__test.lanes.set(key, {{ state: "idle" }}));
const roles = {{
  ingest: "ingest_review",
  recall: "recall_auto_apply",
  audit: "content_correction_classification",
  improve: "model_eval",
  repair: "local_repair",
  typed_graph: "relation_extract",
}};
const mapped = Object.fromEntries(Object.entries(roles).map(([key, task_role]) => [
  key,
  sandbox.__test.processingLaneForTrace({{ request_sha256: "request", task_role }}),
]));
sandbox.__test.lanes.get("recall").state = "active";
sandbox.__test.lanes.get("recall").current_step = "consensus";
const consensusFallback = sandbox.__test.processingLaneForTrace({{}});
sandbox.__test.lanes.get("recall").current_step = "search";
sandbox.__test.lanes.get("ingest").state = "active";
const activeFallback = sandbox.__test.processingLaneForTrace({{}});
laneKeys.forEach((key) => sandbox.__test.lanes.set(key, {{ state: "idle" }}));
const idleFallback = sandbox.__test.processingLaneForTrace({{}});
process.stdout.write(JSON.stringify({{
  mapped,
  consensusFallback,
  activeFallback,
  idleFallback,
}}));
"""

    completed = _run_node_scenario(scenario)
    result = json.loads(completed.stdout)

    assert result["mapped"] == {
        "ingest": "ingest",
        "recall": "recall",
        "audit": "audit",
        "improve": "improve",
        "repair": "repair",
        "typed_graph": "typed_graph",
    }
    assert result["consensusFallback"] == "recall"
    assert result["activeFallback"] == "ingest"
    assert result["idleFallback"] == ""


def test_decision_trace_same_request_polls_merge_monotonically() -> None:
    renderer = (dashboard.STATIC_DIR / "app-renderer.js").read_text(encoding="utf-8")
    scenario = f"""
const vm = require("node:vm");
const clearedTimers = [];
const sandbox = {{
  window: {{
    matchMedia: () => ({{ matches: true }}),
    clearTimeout: (timer) => clearedTimers.push(timer),
  }},
  document: {{ visibilityState: "visible" }},
}};
vm.createContext(sandbox);
vm.runInContext(
  {json.dumps(renderer)}
    + `
renderDecisionTraceFrame = () => {{}};
renderDecisionTransitionFeed = () => {{}};
setDecisionTransitionState = () => {{}};
this.__test = {{ renderDecisionTrace, playback: decisionTracePlayback }};`,
  sandbox,
);
const stepKeys = ["trigger", "load", "context", "generate", "validate", "vote"];
const lane = (key, state = "active") => ({{
  key,
  label: key,
  model: `${{key}}:model`,
  state,
  steps: stepKeys.map((step) => ({{ key: step, status: "pending" }})),
}});
const event = (id, laneKey, phase, milliseconds, extra = {{}}) => ({{
  event_id: id,
  lane: laneKey,
  phase,
  kind: "phase",
  status: "active",
  overall_key: ["trigger", "load"].includes(phase) ? "dispatch" : "generate",
  timestamp: `2026-08-12T00:00:0${{milliseconds}}.000Z`,
  ...extra,
}});
const overall = () => ["packet", "dispatch", "generate", "validate", "quorum"]
  .map((key) => ({{ key, status: "pending" }}));
const request = "shared-request";
const render = (decision_trace) => sandbox.__test.renderDecisionTrace({{ decision_trace }});
const snapshot = () => ({{
  request: sandbox.__test.playback.request,
  state: sandbox.__test.playback.current.state,
  startedAt: sandbox.__test.playback.current.started_at || null,
  seen: [...sandbox.__test.playback.seen].sort(),
  queued: sandbox.__test.playback.queue.map((item) => item.event_id),
  timer: sandbox.__test.playback.timer,
  clearedTimers: [...clearedTimers],
  eventIds: sandbox.__test.playback.target.events.map((item) => item.event_id),
  eventStatuses: Object.fromEntries(sandbox.__test.playback.target.events.map((item) => [
    item.event_id,
    item.status,
  ])),
  eventThink: Object.fromEntries(sandbox.__test.playback.target.events.map((item) => [
    item.event_id,
    item.think || null,
  ])),
  lanes: Object.fromEntries(sandbox.__test.playback.current.lanes.map((item) => [
    item.key,
    item.steps.map((step) => step.status),
  ])),
}});

render({{
  request_sha256: request,
  state: "active",
  active: true,
  overall: overall(),
  lanes: [lane("primary")],
  events: [
    event("p-trigger", "primary", "trigger", 1),
    event("p-load", "primary", "load", 2),
    event("p-context", "primary", "context", 3),
    event("p-generate", "primary", "generate", 4),
  ],
}});
const primary = snapshot();

render({{
  request_sha256: request,
  state: "active",
  active: true,
  overall: overall(),
  lanes: [lane("challenger")],
  events: [
    event("c-trigger", "challenger", "trigger", 1),
    event("c-load", "challenger", "load", 2),
    event("c-context", "challenger", "context", 3),
    event("c-generate", "challenger", "generate", 5),
  ],
}});
const interleaved = snapshot();

render({{
  request_sha256: request,
  state: "active",
  active: true,
  overall: overall(),
  lanes: [lane("tie_break")],
  events: [
    event("t-trigger", "tie_break", "trigger", 6),
    event("t-load", "tie_break", "load", 7),
  ],
}});
const allLanes = snapshot();

render({{
  request_sha256: request,
  state: "active",
  active: true,
  overall: overall(),
  lanes: [lane("primary", "pending")],
  events: [event(
    "p-trigger",
    "primary",
    "trigger",
    1,
    {{ think: "medium", context_tokens: 65536 }},
  )],
}});
const stale = snapshot();

render({{
  request_sha256: request,
  state: "agreed",
  active: false,
  summary: "sealed",
  started_at: "2026-08-12T00:00:00.000Z",
  updated_at: "2026-08-12T00:00:09.000Z",
  overall: overall(),
  lanes: [lane("primary", "done"), lane("challenger", "done")],
  events: [{{
    event_id: "decision",
    lane: null,
    phase: "decision",
    kind: "decision",
    status: "done",
    overall_key: "decision",
    timestamp: "2026-08-12T00:00:09.000Z",
  }}],
}});
const terminal = snapshot();

render({{
  request_sha256: request,
  state: "active",
  active: true,
  overall: overall(),
  lanes: [lane("primary", "pending")],
  events: [
    event("p-trigger", "primary", "trigger", 1),
    {{
      event_id: "decision",
      lane: null,
      phase: "decision",
      kind: "decision",
      status: "active",
      overall_key: "decision",
      timestamp: "2026-08-12T00:00:09.000Z",
    }},
  ],
}});
const lateActive = snapshot();

render({{
  request_sha256: "other-request",
  state: "active",
  active: true,
  overall: overall(),
  lanes: [lane("primary")],
  events: [event("other-trigger", "primary", "trigger", 1)],
}});
const other = snapshot();
render({{
  request_sha256: request,
  state: "active",
  active: true,
  overall: overall(),
  lanes: [lane("primary", "pending")],
  events: [event("p-trigger", "primary", "trigger", 1)],
}});
const returned = snapshot();
sandbox.__test.playback.timer = 99;
render({{
  request_sha256: request,
  state: "active",
  active: true,
  started_at: "2026-08-12T00:00:10.000Z",
  updated_at: "2026-08-12T00:00:10.000Z",
  overall: overall(),
  lanes: [lane("primary")],
  events: [event(
    "next-trigger",
    "primary",
    "trigger",
    9,
    {{ timestamp: "2026-08-12T00:00:11.000Z" }},
  )],
}});
const repeatedHash = snapshot();
process.stdout.write(JSON.stringify({{
  primary,
  interleaved,
  allLanes,
  stale,
  terminal,
  lateActive,
  other,
  returned,
  repeatedHash,
}}));
"""

    completed = _run_node_scenario(scenario)
    result = json.loads(completed.stdout)

    assert result["primary"]["lanes"]["primary"] == [
        "done", "done", "done", "active", "pending", "pending"
    ]
    assert result["interleaved"]["lanes"]["primary"] == result["primary"]["lanes"]["primary"]
    assert result["interleaved"]["lanes"]["challenger"] == [
        "done", "done", "done", "active", "pending", "pending"
    ]
    assert set(result["allLanes"]["lanes"]) == {"primary", "challenger", "tie_break"}
    assert result["allLanes"]["lanes"]["tie_break"] == [
        "done", "active", "pending", "pending", "pending", "pending"
    ]
    assert result["stale"]["lanes"] == result["allLanes"]["lanes"]
    assert result["stale"]["eventThink"]["p-trigger"] == "medium"
    assert set(result["stale"]["eventIds"]) == {
        "p-trigger", "p-load", "p-context", "p-generate",
        "c-trigger", "c-load", "c-context", "c-generate",
        "t-trigger", "t-load",
    }
    assert result["terminal"]["state"] == "agreed"
    assert result["lateActive"]["state"] == "agreed"
    assert result["lateActive"]["eventIds"] == result["terminal"]["eventIds"]
    assert result["lateActive"]["eventStatuses"]["decision"] == "done"
    assert result["other"]["request"] == "other-request"
    assert result["other"]["eventIds"] == ["other-trigger"]
    assert set(result["other"]["lanes"]) == {"primary"}
    assert result["returned"]["request"] == "shared-request"
    assert result["returned"]["state"] == "agreed"
    assert result["returned"]["eventIds"] == result["terminal"]["eventIds"]
    assert result["repeatedHash"]["state"] == "active"
    assert result["repeatedHash"]["eventIds"] == ["next-trigger"]
    assert result["repeatedHash"]["startedAt"] == "2026-08-12T00:00:10.000Z"
    assert result["repeatedHash"]["seen"] == ["next-trigger"]
    assert result["repeatedHash"]["queued"] == []
    assert result["repeatedHash"]["timer"] is None
    assert result["repeatedHash"]["clearedTimers"] == [99]


def test_decision_trace_keeps_an_observed_load_across_partial_polls() -> None:
    renderer = (dashboard.STATIC_DIR / "app-renderer.js").read_text(encoding="utf-8")
    scenario = f"""
const vm = require("node:vm");
const sandbox = {{
  window: {{ matchMedia: () => ({{ matches: true }}) }},
  document: {{ visibilityState: "visible" }},
}};
vm.createContext(sandbox);
vm.runInContext(
  {json.dumps(renderer)}
    + `
renderDecisionTraceFrame = () => {{}};
renderDecisionTransitionFeed = () => {{}};
setDecisionTransitionState = () => {{}};
this.__test = {{ renderDecisionTrace, playback: decisionTracePlayback }};`,
  sandbox,
);
const stepKeys = ["trigger", "load", "context", "generate", "validate", "vote"];
const event = (event_id, phase) => ({{
  event_id,
  lane: "primary",
  phase,
  kind: "phase",
  status: "active",
  overall_key: ["trigger", "load"].includes(phase) ? "dispatch" : "generate",
}});
const trace = (phase, events) => ({{
  request_sha256: "partial-poll-request",
  state: "active",
  active: true,
  events,
  overall: ["packet", "dispatch", "generate", "validate", "quorum"]
    .map((key) => ({{ key, status: "pending" }})),
  lanes: [{{
    key: "primary",
    label: "Primary",
    model: "primary:model",
    state: "active",
    phase,
    steps: stepKeys.map((key) => ({{
      key,
      status: key === phase ? "active"
        : stepKeys.indexOf(key) < stepKeys.indexOf(phase) ? "done" : "pending",
    }})),
  }}],
}});
const render = (value) => sandbox.__test.renderDecisionTrace({{ decision_trace: value }});
const steps = () => sandbox.__test.playback.current.lanes[0].steps.map((step) => step.status);

render(trace("load", [event("trigger", "trigger")]));
const duringLoad = steps();
render(trace("generate", [
  event("trigger", "trigger"),
  event("context", "context"),
  event("generate", "generate"),
]));
const duringGenerate = steps();
process.stdout.write(JSON.stringify({{ duringLoad, duringGenerate }}));
"""

    completed = _run_node_scenario(scenario)

    assert json.loads(completed.stdout) == {
        "duringLoad": ["done", "active", "pending", "pending", "pending", "pending"],
        "duringGenerate": ["done", "done", "done", "active", "pending", "pending"],
    }


def test_decision_trace_same_hash_reset_uses_terminal_and_execution_boundaries() -> None:
    renderer = (dashboard.STATIC_DIR / "app-renderer.js").read_text(encoding="utf-8")
    scenario = f"""
const vm = require("node:vm");
const sandbox = {{ window: {{ matchMedia: () => ({{ matches: true }}) }} }};
vm.createContext(sandbox);
vm.runInContext(
  {json.dumps(renderer)}
    + "\\nthis.__test = {{ decisionTraceIsTerminal, decisionTraceStartsNewExecution }};",
  sandbox,
);
const event = (event_id, timestamp) => ({{ event_id, timestamp }});
const terminal = {{
  state: "agreed",
  active: false,
  started_at: "2026-08-12T00:00:00.000Z",
  updated_at: "2026-08-12T00:00:09.000Z",
  events: [event("decision", "2026-08-12T00:00:09.000Z")],
}};
process.stdout.write(JSON.stringify({{
  terminalStates: {{
    agreed: sandbox.__test.decisionTraceIsTerminal({{ state: "agreed" }}),
    quarantined: sandbox.__test.decisionTraceIsTerminal({{ state: "quarantined" }}),
    readySingle: sandbox.__test.decisionTraceIsTerminal({{
      state: "ready", quorum_flow: false,
    }}),
    readyQuorum: sandbox.__test.decisionTraceIsTerminal({{
      state: "ready", quorum_flow: true,
    }}),
    idle: sandbox.__test.decisionTraceIsTerminal({{ state: "idle", active: false }}),
  }},
  newerStart: sandbox.__test.decisionTraceStartsNewExecution(terminal, {{
    state: "active",
    active: true,
    started_at: "2026-08-12T00:00:10.000Z",
    events: [event("decision", "2026-08-12T00:00:09.000Z")],
  }}),
  newerEvent: sandbox.__test.decisionTraceStartsNewExecution(terminal, {{
    state: "active",
    active: true,
    started_at: terminal.started_at,
    events: [event("next-trigger", "2026-08-12T00:00:10.000Z")],
  }}),
  staleEvent: sandbox.__test.decisionTraceStartsNewExecution(terminal, {{
    state: "active",
    active: true,
    started_at: terminal.started_at,
    events: [event("late-old-poll", "2026-08-12T00:00:08.000Z")],
  }}),
}}));
"""

    completed = _run_node_scenario(scenario)

    assert json.loads(completed.stdout) == {
        "terminalStates": {
            "agreed": True,
            "quarantined": True,
            "readySingle": True,
            "readyQuorum": False,
            "idle": False,
        },
        "newerStart": True,
        "newerEvent": True,
        "staleEvent": False,
    }


def test_decision_trace_frame_prefers_terminal_vote_facts() -> None:
    renderer = (dashboard.STATIC_DIR / "app-renderer.js").read_text(encoding="utf-8")
    scenario = f"""
const vm = require("node:vm");
const element = () => ({{ textContent: "", dataset: {{}}, title: "" }});
const els = new Proxy({{}}, {{
  get(target, key) {{
    if (!(key in target)) target[key] = element();
    return target[key];
  }},
}});
const sandbox = {{
  window: {{ matchMedia: () => ({{ matches: false }}) }},
  document: {{ visibilityState: "visible" }},
  els,
}};
vm.createContext(sandbox);
vm.runInContext(
  {json.dumps(renderer)}
    + `
updateDecisionSvgHarness = () => {{}};
updateProcessingTraceSelection = () => {{}};
setWorkState = () => {{}};
this.__test = {{ renderDecisionTraceFrame, els }};`,
  sandbox,
);
const capture = (trace) => {{
  sandbox.__test.renderDecisionTraceFrame({{
    request_sha256: "terminal-request",
    overall: [{{ key: "decision", label: "Decision", status: "done" }}],
    outcome: {{}},
    ...trace,
  }});
  return {{
    badge: sandbox.__test.els.decisionBadge.textContent,
    modelCalls: sandbox.__test.els.decisionModelCalls.textContent,
    quorum: sandbox.__test.els.decisionQuorum.textContent,
  }};
}};
process.stdout.write(JSON.stringify({{
  terminal: capture({{
    state: "agreed",
    active: false,
    vote_count: 3,
    valid_votes: 2,
    tie_break_used: true,
    lanes: [],
  }}),
  activeTie: capture({{
    state: "active",
    active: true,
    vote_count: 3,
    valid_votes: 2,
    tie_break_used: false,
    lanes: [{{ key: "tie_break", state: "active" }}],
  }}),
  singleModel: capture({{
    state: "active",
    active: true,
    quorum_flow: false,
    vote_count: 1,
    valid_votes: 1,
    lanes: [{{ key: "primary", state: "done" }}],
  }}),
}}));
"""

    completed = _run_node_scenario(scenario)

    assert json.loads(completed.stdout) == {
        "terminal": {"badge": "APPROVED", "modelCalls": "3", "quorum": "2 / 3"},
        "activeTie": {"badge": "RESOLVING", "modelCalls": "3", "quorum": "2 / 3"},
        "singleModel": {"badge": "WAITING", "modelCalls": "1", "quorum": "1 / 1"},
    }
    assert 'hold: safeNoQuorum || sealFailure ? "error" : "pending"' in renderer


def test_decision_trace_reasoning_high_resolves_and_unknown_fails_closed() -> None:
    renderer = (dashboard.STATIC_DIR / "app-renderer.js").read_text(encoding="utf-8")
    helper = "function decisionReasoningPlanState" + renderer.split(
        "function decisionReasoningPlanState", 1
    )[1].split("function updateDecisionSvgHarness", 1)[0]
    scenario = f"""
const vm = require("node:vm");
const sandbox = {{}};
vm.createContext(sandbox);
vm.runInContext(
  {json.dumps(helper)} + "\\nthis.reasoningState = decisionReasoningPlanState;",
  sandbox,
);
process.stdout.write(JSON.stringify({{
  high: sandbox.reasoningState({{ think: "HIGH" }}, "done"),
  unknown: sandbox.reasoningState({{ think: "adaptive" }}, "done"),
  absent: sandbox.reasoningState(null, "active"),
}}));
"""

    completed = _run_node_scenario(scenario)

    assert json.loads(completed.stdout) == {
        "high": {"mode": "high", "fit": "done"},
        "unknown": {"mode": "adaptive", "fit": "pending"},
        "absent": {"mode": "—", "fit": "pending"},
    }
    assert '["plan-dispatch", fitState]' in renderer


def test_decision_trace_pair_branches_require_observed_vote_truth() -> None:
    renderer = (dashboard.STATIC_DIR / "app-renderer.js").read_text(encoding="utf-8")
    helper = (
        "function fmt"
        + renderer.split("function fmt", 1)[1].split("function shortName", 1)[0]
        + "function decisionPairBranchStates"
        + renderer.split("function decisionPairBranchStates", 1)[1].split(
            "function updateDecisionSvgHarness", 1
        )[0]
    )
    scenario = f"""
const vm = require("node:vm");
const sandbox = {{}};
vm.createContext(sandbox);
vm.runInContext(
  {json.dumps(helper)} + "\\nthis.pairStates = decisionPairBranchStates;",
  sandbox,
);
const pair = (trace, state = "pending") => sandbox.pairStates(trace, {{ state }});
process.stdout.write(JSON.stringify({{
  artifactOnly: pair({{ pair_agreement: null, quorum_attempted: true }}),
  explicitYes: pair({{ pair_agreement: true, quorum_attempted: true }}),
  explicitNo: pair({{ pair_agreement: false, quorum_attempted: true }}),
  unattemptedNo: pair({{ pair_agreement: false, quorum_attempted: false }}),
  activeTie: pair({{ pair_agreement: null, tie_break_used: false }}, "active"),
  finishedTie: pair({{ pair_agreement: null, tie_break_used: true }}, "done"),
}}));
"""

    completed = _run_node_scenario(scenario)

    assert json.loads(completed.stdout) == {
        "artifactOnly": {"tieObserved": False, "yes": "pending", "no": "pending"},
        "explicitYes": {"tieObserved": False, "yes": "done", "no": "pending"},
        "explicitNo": {"tieObserved": False, "yes": "pending", "no": "done"},
        "unattemptedNo": {"tieObserved": False, "yes": "pending", "no": "pending"},
        "activeTie": {"tieObserved": True, "yes": "pending", "no": "active"},
        "finishedTie": {"tieObserved": True, "yes": "pending", "no": "done"},
    }


def test_dashboard_static_layout_aligns_peer_panels_and_contains_event_badges() -> None:
    style = (dashboard.STATIC_DIR / "style.css").read_text(encoding="utf-8")

    assert ".knowledge-panel {\n  height: var(--panel-height);\n}" in style
    assert "grid-template-columns: 84px 160px minmax(0, 1fr);" in style
    assert (
        ".event-level {\n"
        "  display: inline-flex;\n"
        "  justify-content: center;\n"
        "  min-width: 0;\n"
        "  overflow: hidden;"
    ) in style
    assert "text-overflow: ellipsis;\n  text-transform: uppercase;\n  white-space: nowrap;" in style
    assert ".decision-trace-harness .decision-role {" in style
    assert "fill: #e0e5e9;\n  font-size: 12px;" in style
    assert ".decision-trace-harness .decision-model," in style
    assert ".decision-trace-harness .decision-think," in style
    assert "fill: #687784;\n  font-size: 8px;" in style


def test_decision_trace_replay_uses_only_observed_context_metadata() -> None:
    renderer = (dashboard.STATIC_DIR / "app-renderer.js").read_text(encoding="utf-8")
    scenario = f"""
const vm = require("node:vm");
const sandbox = {{ window: {{ matchMedia: () => ({{ matches: false }}) }} }};
vm.createContext(sandbox);
vm.runInContext(
  {json.dumps(renderer)}
    + "\\nthis.__test = {{ decisionTraceBlank, applyDecisionTransition }};",
  sandbox,
);
const target = {{
  request_sha256: "request",
  task_role: "ingest_review",
  overall: [
    {{ key: "dispatch", status: "done" }},
    {{ key: "generate", status: "done" }},
    {{ key: "validate", status: "done" }},
    {{ key: "quorum", status: "done" }},
  ],
  context_tokens: 131072,
  events: [
    {{ event_id: "trigger", lane: "primary", phase: "trigger", kind: "phase", status: "active", overall_key: "dispatch" }},
    {{ event_id: "load", lane: "primary", phase: "load", kind: "phase", status: "active", overall_key: "dispatch" }},
    {{ event_id: "context", lane: "primary", phase: "context", kind: "phase", status: "active", overall_key: "generate", think: "medium", required_context_tokens: 12000, requested_context_tokens: 32768, context_tokens: 32768 }},
    {{ event_id: "generate", lane: "primary", phase: "generate", kind: "phase", status: "active", overall_key: "generate", think: "medium", required_context_tokens: 12000, requested_context_tokens: 32768, context_tokens: 32768 }},
  ],
  lanes: [
    {{
      key: "primary",
      label: "Primary",
      model: "primary:model",
      think: "high",
      required_context_tokens: 64000,
      requested_context_tokens: 131072,
      context_tokens: 131072,
      steps: ["trigger", "load", "context", "generate", "validate", "vote"].map(
        (key) => ({{ key, status: "done" }}),
      ),
    }},
    {{ key: "challenger", think: "low", steps: [] }},
    {{ key: "tie_break", think: "off", steps: [] }},
  ],
}};
const blank = sandbox.__test.decisionTraceBlank(target);
let frame = blank;
const observed = {{}};
for (const event of target.events) {{
  frame = sandbox.__test.applyDecisionTransition(frame, target, event);
  observed[event.phase] = {{
    think: frame.lanes[0].think,
    required: frame.lanes[0].required_context_tokens,
    requested: frame.lanes[0].requested_context_tokens,
    effective: frame.lanes[0].context_tokens,
    traceContext: frame.context_tokens,
  }};
}}
const laneModes = [["primary", "medium"], ["challenger", "low"], ["tie_break", "off"]]
  .map(([lane, think]) => sandbox.__test.applyDecisionTransition(
    blank,
    target,
    {{ lane, phase: "context", kind: "phase", status: "active", overall_key: "generate", think, context_tokens: 32768 }},
  ).lanes.find((item) => item.key === lane).think);
process.stdout.write(JSON.stringify({{
  blank: blank.lanes.map((lane) => lane.think),
  blankContext: blank.context_tokens,
  blankLaneContext: blank.lanes[0].context_tokens,
  observed,
  laneModes,
}}));
"""

    completed = _run_node_scenario(scenario)
    result = json.loads(completed.stdout)

    assert result["blank"] == ["—", "—", "—"]
    assert result["blankContext"] is None
    assert result["blankLaneContext"] is None
    assert result["observed"]["trigger"] == {
        "think": "—",
        "required": None,
        "requested": None,
        "effective": None,
        "traceContext": None,
    }
    assert result["observed"]["load"] == result["observed"]["trigger"]
    assert result["observed"]["context"] == {
        "think": "medium",
        "required": 12_000,
        "requested": 32_768,
        "effective": 32_768,
        "traceContext": 32_768,
    }
    assert result["observed"]["generate"] == result["observed"]["context"]
    assert result["laneModes"] == ["medium", "low", "off"]


def test_decision_trace_timeline_is_granular_forward_only_and_bounded() -> None:
    renderer = (dashboard.STATIC_DIR / "app-renderer.js").read_text(encoding="utf-8")
    scenario = f"""
const vm = require("node:vm");
const sandbox = {{ window: {{ matchMedia: () => ({{ matches: false }}) }} }};
vm.createContext(sandbox);
vm.runInContext(
  {json.dumps(renderer)}
    + "\\nthis.__test = {{ decisionTimelineSteps, decisionTimelineCurrent, decisionTraceBlank, applyDecisionTransition }};",
  sandbox,
);
const phases = ["generate", "validate", "repair", "validate"];
const events = phases.map((phase, index) => ({{
  event_id: `event-${{index + 1}}`,
  kind: "phase",
  lane: "primary",
  phase,
  status: "active",
  attempt: index < 2 ? 0 : 1,
  label: phase,
}}));
const target = {{
  request_sha256: "request",
  active: true,
  events,
  overall: [],
  lanes: [{{
    key: "primary",
    label: "Primary",
    model: "primary:model",
    think: "medium",
    steps: ["trigger", "load", "context", "generate", "validate", "vote"].map(
      (key) => ({{ key, status: "pending" }}),
    ),
  }}],
}};
let frame = sandbox.__test.decisionTraceBlank(target);
const frames = events.map((event) => {{
  frame = sandbox.__test.applyDecisionTransition(frame, target, event);
  return {{
    eventIds: frame.events.map((item) => item.event_id),
    labels: sandbox.__test.decisionTimelineSteps(frame).map((step) => step.label),
    statuses: sandbox.__test.decisionTimelineSteps(frame).map((step) => step.status),
    laneStatuses: frame.lanes[0].steps.map((step) => step.status),
  }};
}});
const errorTimeline = sandbox.__test.decisionTimelineSteps({{
  request_sha256: "error-request",
  active: false,
  events: [
    {{ event_id: "vote", kind: "phase", lane: "primary", phase: "vote", status: "active" }},
    {{ event_id: "error", kind: "session", lane: "primary", phase: "vote", status: "error", label: "Vote rejected" }},
  ],
}});
const fallback = sandbox.__test.decisionTimelineSteps({{
  events: [],
  overall: [
    {{ key: "packet", label: "Packet", status: "done" }},
    {{ key: "generate", label: "Generate", status: "active" }},
  ],
}});
process.stdout.write(JSON.stringify({{
  frames,
  errorTimeline,
  errorCurrent: sandbox.__test.decisionTimelineCurrent(errorTimeline),
  fallback,
}}));
"""

    completed = _run_node_scenario(scenario)
    result = json.loads(completed.stdout)
    frames = result["frames"]

    assert frames[1]["eventIds"] == ["event-1", "event-2"]
    assert frames[1]["labels"] == [
        "Packet",
        "Primary Generate #1",
        "Primary Validate #1",
    ]
    assert frames[-1]["labels"] == [
        "Packet",
        "Primary Generate #1",
        "Primary Validate #1",
        "Primary Repair #1",
        "Primary Validate #2",
    ]
    assert frames[-1]["statuses"] == ["done", "done", "done", "done", "active"]
    assert frames[2]["laneStatuses"] == [
        "pending",
        "pending",
        "pending",
        "done",
        "active",
        "pending",
    ]
    assert [step["label"] for step in result["errorTimeline"]] == [
        "Packet",
        "Primary Vote #1",
        "Primary Vote rejected",
    ]
    assert result["errorTimeline"][-1]["status"] == "error"
    assert result["errorCurrent"] == {
        "position": 3,
        "label": "Primary Vote rejected",
    }
    assert [step["label"] for step in result["fallback"]] == [
        "Packet",
        "Generate",
    ]


def test_decision_trace_replay_colors_only_observed_phases_for_every_lane() -> None:
    renderer = (dashboard.STATIC_DIR / "app-renderer.js").read_text(encoding="utf-8")
    scenario = f"""
const vm = require("node:vm");
const sandbox = {{ window: {{ matchMedia: () => ({{ matches: false }}) }} }};
vm.createContext(sandbox);
vm.runInContext(
  {json.dumps(renderer)}
    + "\\nthis.__test = {{ decisionTraceBlank, applyDecisionTransition }};",
  sandbox,
);
const laneKeys = ["primary", "challenger", "tie_break"];
const stepKeys = ["trigger", "load", "context", "generate", "validate", "vote"];
const overallKey = (phase) =>
  ["trigger", "load"].includes(phase) ? "dispatch"
    : ["context", "generate"].includes(phase) ? "generate"
      : ["repair", "validate"].includes(phase) ? "validate" : "quorum";
const replay = (lane, phases, terminalStatus) => {{
  const phaseEvents = phases.map((phase, index) => ({{
    event_id: `${{lane}}-${{index}}-${{phase}}`,
    lane,
    phase,
    kind: "phase",
    status: "active",
    overall_key: overallKey(phase),
  }}));
  const events = [
    ...phaseEvents,
    {{
      event_id: `${{lane}}-session`,
      lane,
      phase: "vote",
      kind: "session",
      status: terminalStatus,
      overall_key: "quorum",
    }},
  ];
  const target = {{
    request_sha256: `${{lane}}-request`,
    active: terminalStatus !== "done",
    events,
    overall: ["packet", "dispatch", "generate", "validate", "quorum"]
      .map((key) => ({{ key, status: "done" }})),
    lanes: laneKeys.map((key) => ({{
      key,
      label: key,
      model: `${{key}}:model`,
      think: "medium",
      steps: stepKeys.map((step) => ({{ key: step, status: "done" }})),
    }})),
  }};
  let frame = sandbox.__test.decisionTraceBlank(target);
  const snapshots = [];
  const overallSnapshots = [];
  for (const event of events) {{
    frame = sandbox.__test.applyDecisionTransition(frame, target, event);
    snapshots.push(frame.lanes.find((item) => item.key === lane).steps.map(
      (step) => step.status,
    ));
    overallSnapshots.push(Object.fromEntries(
      frame.overall.map((step) => [step.key, step.status]),
    ));
  }}
  return {{ snapshots, overallSnapshots, final: snapshots.at(-1) }};
}};
process.stdout.write(JSON.stringify({{
  normal: Object.fromEntries(laneKeys.map((lane) => [lane, replay(
    lane,
    ["trigger", "load", "context", "generate", "validate", "vote"],
    "done",
  )])),
  noLoad: replay("primary", ["trigger", "context", "generate", "validate", "vote"], "done"),
  preflightFailure: replay("primary", ["trigger", "vote"], "error"),
  capacityFailure: replay("primary", ["trigger", "load", "vote"], "error"),
  transportFailure: replay(
    "primary",
    ["trigger", "load", "context", "generate", "vote"],
    "error",
  ),
  repair: replay(
    "primary",
    ["trigger", "load", "context", "generate", "validate", "repair", "validate", "vote"],
    "done",
  ),
}}));
"""

    completed = _run_node_scenario(scenario)
    result = json.loads(completed.stdout)

    expected_done = ["done"] * 6
    for lane in ("primary", "challenger", "tie_break"):
        assert result["normal"][lane]["snapshots"][:3] == [
            ["active", "pending", "pending", "pending", "pending", "pending"],
            ["done", "active", "pending", "pending", "pending", "pending"],
            ["done", "done", "active", "pending", "pending", "pending"],
        ]
        assert result["normal"][lane]["final"] == expected_done
        assert result["normal"][lane]["overallSnapshots"][0] == {
            "packet": "done",
            "dispatch": "active",
            "generate": "pending",
            "validate": "pending",
            "quorum": "pending",
        }
        assert result["normal"][lane]["overallSnapshots"][2] == {
            "packet": "done",
            "dispatch": "done",
            "generate": "active",
            "validate": "pending",
            "quorum": "pending",
        }
        assert result["normal"][lane]["overallSnapshots"][-1]["quorum"] == "active"
    assert result["noLoad"]["final"] == [
        "done", "pending", "done", "done", "done", "done"
    ]
    assert result["preflightFailure"]["final"] == [
        "done", "pending", "pending", "pending", "pending", "error"
    ]
    assert result["capacityFailure"]["final"] == [
        "done", "done", "pending", "pending", "pending", "error"
    ]
    assert result["transportFailure"]["final"] == [
        "done", "done", "done", "done", "pending", "error"
    ]
    assert result["repair"]["snapshots"][5] == [
        "done", "done", "done", "done", "active", "pending"
    ]
    assert result["repair"]["final"] == expected_done


def test_decision_trace_initial_active_render_uses_only_observed_events() -> None:
    renderer = (dashboard.STATIC_DIR / "app-renderer.js").read_text(encoding="utf-8")
    scenario = f"""
const vm = require("node:vm");
const sandbox = {{ window: {{ matchMedia: () => ({{ matches: false }}) }} }};
vm.createContext(sandbox);
vm.runInContext(
  {json.dumps(renderer)}
    + `
const capturedFrames = [];
renderDecisionTraceFrame = (trace) => capturedFrames.push(
  JSON.parse(JSON.stringify(trace)),
);
renderDecisionTransitionFeed = () => {{}};
setDecisionTransitionState = () => {{}};
this.__test = {{ renderDecisionTrace, capturedFrames, decisionTracePlayback }};`,
  sandbox,
);
const stepKeys = ["trigger", "load", "context", "generate", "validate", "vote"];
const target = {{
  request_sha256: "initial-active-request",
  active: true,
  events: [
    {{ event_id: "trigger", lane: "primary", phase: "trigger", kind: "phase", status: "active", overall_key: "dispatch" }},
    {{ event_id: "context", lane: "primary", phase: "context", kind: "phase", status: "active", overall_key: "generate" }},
    {{ event_id: "generate", lane: "primary", phase: "generate", kind: "phase", status: "active", overall_key: "generate" }},
  ],
  overall: ["packet", "dispatch", "generate", "validate", "quorum"]
    .map((key) => ({{ key, status: "done" }})),
  lanes: [{{
    key: "primary",
    label: "Primary",
    model: "primary:model",
    steps: stepKeys.map((key) => ({{ key, status: "done" }})),
  }}],
}};
sandbox.__test.renderDecisionTrace({{ decision_trace: target }});
const frame = sandbox.__test.capturedFrames[0];
process.stdout.write(JSON.stringify({{
  lane: frame.lanes[0].steps.map((step) => step.status),
  overall: Object.fromEntries(frame.overall.map((step) => [step.key, step.status])),
  events: frame.events.map((event) => event.event_id),
  current: sandbox.__test.decisionTracePlayback.current.lanes[0].steps.map(
    (step) => step.status,
  ),
}}));
"""

    completed = _run_node_scenario(scenario)
    result = json.loads(completed.stdout)

    assert result["lane"] == [
        "done", "pending", "done", "active", "pending", "pending"
    ]
    assert result["overall"] == {
        "packet": "done",
        "dispatch": "done",
        "generate": "active",
        "validate": "pending",
        "quorum": "pending",
    }
    assert result["events"] == ["trigger", "context", "generate"]
    assert result["current"] == result["lane"]


def test_decision_trace_batches_new_events_into_300ms_frames() -> None:
    renderer = (dashboard.STATIC_DIR / "app-renderer.js").read_text(encoding="utf-8")
    scenario = f"""
const vm = require("node:vm");
const timers = [];
const window = {{
  matchMedia: () => ({{ matches: false }}),
  setTimeout: (callback, delay) => (timers.push({{ callback, delay }}), timers.length),
  clearTimeout: () => {{}},
}};
const sandbox = {{ window, document: {{ visibilityState: "visible" }} }};
vm.createContext(sandbox);
vm.runInContext(
  {json.dumps(renderer)}
    + `
const capturedFrames = [];
renderDecisionTraceFrame = (trace) => capturedFrames.push(
  JSON.parse(JSON.stringify(trace)),
);
renderDecisionTransitionFeed = () => {{}};
setDecisionTransitionState = () => {{}};
this.__test = {{ renderDecisionTrace, capturedFrames, decisionTracePlayback }};`,
  sandbox,
);
const steps = ["trigger", "load", "context", "generate", "validate", "vote"];
const lanes = ["primary", "challenger", "tie_break"].map((key) => ({{
  key,
  label: key,
  model: `${{key}}:model`,
  state: "active",
  steps: steps.map((step) => ({{ key: step, status: "pending" }})),
}}));
const initial = {{
  request_sha256: "request",
  active: true,
  state: "active",
  events: [
    {{ event_id: "primary-trigger", lane: "primary", phase: "trigger", kind: "phase", status: "active", overall_key: "dispatch" }},
  ],
  overall: ["packet", "dispatch", "generate", "validate", "quorum"]
    .map((key) => ({{ key, status: "pending" }})),
  lanes,
}};
const updated = JSON.parse(JSON.stringify(initial));
updated.events.push(
  {{ event_id: "challenger-context", lane: "challenger", phase: "context", kind: "phase", status: "active", overall_key: "generate" }},
  {{ event_id: "tie-validate", lane: "tie_break", phase: "validate", kind: "phase", status: "active", overall_key: "validate" }},
);
sandbox.__test.renderDecisionTrace({{ decision_trace: initial }});
sandbox.__test.renderDecisionTrace({{ decision_trace: updated }});
const afterUpdate = {{
  target: sandbox.__test.decisionTracePlayback.target.events.map((event) => event.event_id),
  current: sandbox.__test.decisionTracePlayback.current.events.map((event) => event.event_id),
  queue: sandbox.__test.decisionTracePlayback.queue.map((event) => event.event_id),
  delays: timers.map((timer) => timer.delay),
}};
timers[0].callback();
const afterTimer = {{
  current: sandbox.__test.decisionTracePlayback.current.events.map((event) => event.event_id),
  queue: sandbox.__test.decisionTracePlayback.queue.map((event) => event.event_id),
  delays: timers.map((timer) => timer.delay),
  frames: sandbox.__test.capturedFrames.map((trace) => trace.events.map((event) => event.event_id)),
}};
process.stdout.write(JSON.stringify({{ afterUpdate, afterTimer }}));
"""

    result = json.loads(_run_node_scenario(scenario).stdout)

    assert result["afterUpdate"] == {
        "target": ["primary-trigger", "challenger-context", "tie-validate"],
        "current": ["primary-trigger", "challenger-context"],
        "queue": ["tie-validate"],
        "delays": [300],
    }
    assert result["afterTimer"] == {
        "current": ["primary-trigger", "challenger-context", "tie-validate"],
        "queue": [],
        "delays": [300, 300],
        "frames": [
            ["primary-trigger"],
            ["primary-trigger", "challenger-context"],
            ["primary-trigger", "challenger-context", "tie-validate"],
        ],
    }


def test_build_snapshot_combines_runtime_and_queue(tmp_path: Path, monkeypatch) -> None:
    chronovisor_root = tmp_path / "wiki"
    raw_dir = chronovisor_root / "raw"
    runtime_dir = chronovisor_root / "runtime"
    raw_dir.mkdir(parents=True)
    (chronovisor_root / "pages").mkdir()
    (chronovisor_root / "system").mkdir()
    (raw_dir / "r1.md").write_text("raw")

    monkeypatch.setattr(dashboard, "CHRONOVISOR_ROOT", chronovisor_root)
    monkeypatch.setattr(dashboard, "init_chronovisor", lambda: None)
    monkeypatch.setattr(
        dashboard,
        "ACTIVITY_FILE",
        chronovisor_root / "runtime" / "activity.jsonl",
    )
    monkeypatch.setattr(runtime_status, "RUNTIME_DIR", runtime_dir)
    monkeypatch.setattr(runtime_status, "STATUS_FILE", runtime_dir / "status.json")
    monkeypatch.setattr(runtime_status, "EVENTS_FILE", runtime_dir / "events.jsonl")
    monkeypatch.setattr(runtime_status, "METRICS_FILE", runtime_dir / "metrics.jsonl")

    from chronovisor.core import store
    from chronovisor.ingest import orchestrator

    monkeypatch.setattr(store, "CHRONOVISOR_ROOT", chronovisor_root)
    monkeypatch.setattr(store, "RAW_DIR", raw_dir)
    monkeypatch.setattr(store, "PAGES_DIR", chronovisor_root / "pages")
    monkeypatch.setattr(store, "SYSTEM_DIR", chronovisor_root / "system")
    monkeypatch.setattr(store, "INDEX_FILE", chronovisor_root / "pages" / "index.md")
    monkeypatch.setattr(store, "LOG_FILE", chronovisor_root / "pages" / "log.md")
    monkeypatch.setattr(
        store, "ACTIVITY_FILE", runtime_dir / "activity.jsonl"
    )
    monkeypatch.setattr(orchestrator, "RAW_DIR", raw_dir)
    monkeypatch.setattr(
        orchestrator, "STATE_FILE", chronovisor_root / ".orchestrator_state.json"
    )
    monkeypatch.setattr(
        dashboard, "_ollama_snapshot", lambda: {"available": False, "models": []}
    )
    monkeypatch.setattr(
        dashboard,
        "_model_status_snapshot",
        lambda ollama=None: {
            "available": False,
            "models": [],
            "summary": {"installed": 0, "loaded": 0},
        },
    )
    monkeypatch.setattr(
        dashboard,
        "_frontier_preflight_snapshot",
        lambda: {"ok": True, "checked_at": "2026-06-01T12:00:00"},
    )
    monkeypatch.setattr(
        dashboard,
        "health_snapshot",
        lambda: (_ for _ in ()).throw(RuntimeError("duplicate page id")),
    )

    runtime_status.write_status({"state": "running", "stage": "generate"})
    runtime_status.append_event("info", "ingest | stage 1: triage started")
    runtime_status.append_metric(
        "batch", pending_before=2, pending_after=1, files_processed=1
    )
    frontier_activity_dir = runtime_dir / "frontier-reviews" / "active"
    frontier_activity_dir.mkdir(parents=True)
    (frontier_activity_dir / "review-1.json").write_text(
        json.dumps(
            {
                "review_id": "review-1",
                "active": True,
                "kind": "semantic_judge",
                "reviewer": "codex",
                "model": "gpt-5.5",
                "pid": os.getpid(),
                "started_at": datetime.now().isoformat(timespec="seconds"),
            }
        ),
        encoding="utf-8",
    )
    consensus_dir = runtime_dir / "local-consensus"
    consensus_active = consensus_dir / "active"
    consensus_active.mkdir(parents=True)
    (consensus_active / "vote-1.json").write_text(
        json.dumps(
            {
                "request_sha256": "c" * 64,
                "role": "primary",
                "model": "ornith:test",
                "pid": os.getpid(),
                "started_at": datetime.now().isoformat(timespec="seconds"),
            }
        ),
        encoding="utf-8",
    )
    (consensus_dir / "summary.json").write_text(
        json.dumps(dashboard._empty_local_consensus_summary()),
        encoding="utf-8",
    )
    failures_dir = runtime_dir / "failures"
    packets_dir = failures_dir / "packets"
    packets_dir.mkdir(parents=True)
    packet = {
        "failure_id": "f1",
        "created_at": "2026-06-01T12:00:00",
        "raw_file": "broken.md",
        "failure_class": "apply.update_target_not_found",
        "status": "local_repair_applied",
    }
    (packets_dir / "f1.json").write_text(json.dumps(packet), encoding="utf-8")
    registry_record = {
        "timestamp": "2026-06-01T12:01:00",
        "failure_id": "f1",
        "raw_file": "broken.md",
        "failure_class": "apply.update_target_not_found",
        "resolution": "local",
        "decision": {
            "action": "resolve_update_target",
            "requested_page_id": "missing",
            "target_page_id": "ai/target",
            "source": "qwen",
        },
        "action": {
            "alias": {"requested": "missing", "target": "ai/target"},
            "retry": {"files_processed": ["broken.md"]},
        },
    }
    (failures_dir / "failure-registry.jsonl").write_text(
        json.dumps(registry_record) + "\n",
        encoding="utf-8",
    )

    snapshot = dashboard.build_snapshot()

    assert snapshot["status"]["pending"] == 1
    assert snapshot["status"]["state"] == "running"
    assert snapshot["events"]
    assert snapshot["metrics"][0]["pending_after"] == 1
    assert snapshot["self_heal"]["status"] == "resolved"
    assert snapshot["self_heal"]["counts"]["resolved"] == 1
    assert snapshot["self_heal"]["latest"]["raw_file"] == "broken.md"
    assert "alias missing -> ai/target" in snapshot["self_heal"]["latest"]["detail"]
    assert (
        snapshot["self_heal"]["latest"]["details"]["failure"]["packet_status"]
        == "local_repair_applied"
    )
    assert snapshot["self_heal"]["latest"]["details"]["decision"]["source"] == "qwen"
    assert snapshot["self_heal"]["latest"]["details"]["action"]["retry"][
        "files_processed"
    ] == ["broken.md"]
    assert snapshot["self_heal"]["watch"]["packets"]["total"] == 1
    assert snapshot["self_heal"]["watch"]["frontier_preflight"]["ok"] is True
    assert snapshot["frontier_review"]["active"] is True
    assert snapshot["frontier_review"]["count"] == 1
    assert snapshot["status"]["frontier_review"]["latest"]["model"] == "gpt-5.5"
    assert snapshot["local_consensus"]["active"] is True
    assert snapshot["local_consensus"]["latest"]["model"] == "ornith:test"
    assert snapshot["status"]["review_kind"] == "local_consensus"
    assert snapshot["frontier_repair"]["process_activity"]["active"] is True
    assert snapshot["health"]["status"] == "error"
    assert snapshot["health"]["error_class"] == "RuntimeError"
    assert snapshot["health"]["error"] == "duplicate page id"


def test_snapshot_handler_returns_non_disclosing_json_error(
    monkeypatch,
) -> None:
    canary = "dashboard-secret-canary"
    monkeypatch.setattr(
        dashboard,
        "_cached_snapshot",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError(canary)),
    )
    server = dashboard.ThreadingHTTPServer(("127.0.0.1", 0), dashboard.DashboardHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        response = dashboard.httpx.get(f"http://{host}:{port}/api/snapshot", timeout=2)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert response.status_code == 500
    assert response.json() == {
        "status": "error",
        "category": "internal_error",
        "error": "Dashboard request failed.",
    }
    assert canary not in response.text


def test_model_status_handler_reads_live_ollama_without_snapshot_cache(
    monkeypatch,
) -> None:
    model = "muse-glimmer:30b-nvfp4-dflash"
    monkeypatch.setattr(
        dashboard,
        "_cached_snapshot",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("cache used")),
    )
    monkeypatch.setattr(
        dashboard,
        "_ollama_snapshot",
        lambda: {
            "available": True,
            "models": [{"name": model, "size_vram": 35_000}],
        },
    )
    monkeypatch.setattr(
        dashboard,
        "_ollama_tags_snapshot",
        lambda: {
            "available": True,
            "models": [{"name": model, "size": 35_000}],
        },
    )
    monkeypatch.setattr(
        dashboard,
        "_configured_model_roles",
        lambda: {model: {"decision-challenger"}},
    )
    monkeypatch.setattr(runtime_status, "read_events", lambda **_kwargs: [])
    server = dashboard.ThreadingHTTPServer(("127.0.0.1", 0), dashboard.DashboardHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        response = dashboard.httpx.get(
            f"http://{host}:{port}/api/model-status", timeout=2
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    payload = response.json()
    muse = next(row for row in payload["model_status"]["models"] if row["name"] == model)
    assert response.status_code == 200
    assert muse["status"] == "loaded"
    assert muse["running"] is True


def test_dashboard_private_client_scope_rejects_public_addresses() -> None:
    assert dashboard._private_client_scope("127.0.0.1") == "loopback"
    assert dashboard._private_client_scope("192.168.1.22") == "private"
    assert dashboard._private_client_scope("10.1.2.3") == "private"
    assert dashboard._private_client_scope("169.254.1.2") == "private"
    assert dashboard._private_client_scope("192.0.0.1") == "public"
    assert dashboard._private_client_scope("100.64.0.1") == "public"
    assert dashboard._private_client_scope("8.8.8.8") == "public"
    assert dashboard._private_client_scope("not-an-ip") == "invalid"


def test_dashboard_static_path_resolver_rejects_directory_and_symlink_escapes(
    tmp_path: Path,
) -> None:
    static_dir = tmp_path / "static"
    static_dir.mkdir()
    inside = static_dir / "app.js"
    inside.write_text("inside", encoding="utf-8")
    outside = tmp_path / "outside.js"
    outside.write_text("outside", encoding="utf-8")
    (static_dir / "linked.js").symlink_to(outside)

    assert dashboard._resolve_static_path(static_dir, "/static/app.js") == inside
    assert dashboard._resolve_static_path(static_dir, "/static/../outside.js") is None
    assert dashboard._resolve_static_path(static_dir, "/static/linked.js") is None


def test_dashboard_host_allowlist_uses_loopback_names_and_actual_port() -> None:
    server = dashboard.ThreadingHTTPServer(("127.0.0.1", 0), dashboard.DashboardHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        responses = [
            dashboard.httpx.get(
                f"http://{host}:{port}/api/lan-access",
                headers={"Host": allowed},
                timeout=2,
            )
            for allowed in (
                f"localhost:{port}",
                f"127.0.0.1:{port}",
                f"[::1]:{port}",
            )
        ]
        wrong_name = dashboard.httpx.get(
            f"http://{host}:{port}/api/lan-access",
            headers={"Host": f"attacker.example:{port}"},
            timeout=2,
        )
        wrong_port = dashboard.httpx.get(
            f"http://{host}:{port}/api/lan-access",
            headers={"Host": "127.0.0.1:1"},
            timeout=2,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert [response.status_code for response in responses] == [200, 200, 200]
    assert wrong_name.status_code == 421
    assert wrong_port.status_code == 421
    assert wrong_name.headers["x-frame-options"] == "DENY"


def test_dashboard_http_origin_is_same_origin_or_absent() -> None:
    server = dashboard.ThreadingHTTPServer(("127.0.0.1", 0), dashboard.DashboardHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    request_host = f"127.0.0.1:{port}"
    url = f"http://{host}:{port}/api/lan-access"
    try:
        missing = dashboard.httpx.get(
            url, headers={"Host": request_host}, timeout=2
        )
        same = dashboard.httpx.get(
            url,
            headers={"Host": request_host, "Origin": f"http://{request_host}"},
            timeout=2,
        )
        cross = dashboard.httpx.get(
            url,
            headers={"Host": request_host, "Origin": "http://attacker.example"},
            timeout=2,
        )
        other_loopback_origin = dashboard.httpx.get(
            url,
            headers={
                "Host": request_host,
                "Origin": f"http://localhost:{port}",
            },
            timeout=2,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert missing.status_code == 200
    assert same.status_code == 200
    assert cross.status_code == 403
    assert other_loopback_origin.status_code == 403


def test_dashboard_websocket_requires_same_origin() -> None:
    server = dashboard.ThreadingHTTPServer(("127.0.0.1", 0), dashboard.DashboardHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address

    def websocket_status(origin: str | None) -> int:
        origin_header = f"Origin: {origin}\r\n" if origin else ""
        request = (
            "GET /api/cortex/events HTTP/1.1\r\n"
            f"Host: 127.0.0.1:{port}\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            "Sec-WebSocket-Version: 13\r\n"
            "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n"
            f"{origin_header}\r\n"
        ).encode("ascii")
        with socket.create_connection((host, port), timeout=2) as connection:
            connection.sendall(request)
            status_line = connection.recv(4096).split(b"\r\n", 1)[0]
        return int(status_line.split()[1])

    try:
        missing = websocket_status(None)
        cross = websocket_status("http://attacker.example")
        same = websocket_status(f"http://127.0.0.1:{port}")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert missing == 403
    assert cross == 403
    assert same == 101


def test_dashboard_disables_lan_share_cookie_and_non_loopback_bind() -> None:
    server = dashboard.ThreadingHTTPServer(("127.0.0.1", 0), dashboard.DashboardHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        response = dashboard.httpx.get(
            f"http://{host}:{port}/api/lan-access?access_token={'a' * 43}",
            timeout=2,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert response.json() == {
        "enabled": False,
        "urls": [],
        "trusted_lan_only": False,
    }
    assert "set-cookie" not in response.headers
    with pytest.raises(ValueError, match="explicit private IPv4"):
        dashboard.serve("127.0.0.1", 0, lan=True)
    with pytest.raises(ValueError, match="explicit private IPv4"):
        dashboard.serve("0.0.0.0", 0, lan=True)
    with pytest.raises(ValueError, match="dashboard host must be"):
        dashboard.serve("0.0.0.0", 0)
    with pytest.raises(ValueError, match="dashboard host must be"):
        dashboard.serve("::1", 0)


def test_dashboard_lan_requires_basic_then_uses_secure_bounded_session(
    tmp_path: Path, monkeypatch
) -> None:
    credentials_path = tmp_path / "dashboard-credentials.json"
    dashboard._write_dashboard_credentials(
        credentials_path, "admin", "correct horse battery staple"
    )
    with pytest.raises(ValueError, match="safe ASCII"):
        dashboard._write_dashboard_credentials(
            credentials_path, "管理者", "correct horse battery staple"
        )
    credentials = dashboard._load_dashboard_credentials(credentials_path)
    assert not dashboard._dashboard_credentials_match(
        credentials, "管理者", "correct horse battery staple"
    )
    monkeypatch.setattr(dashboard, "_private_client_scope", lambda _value: "private")
    server = dashboard.ThreadingHTTPServer(("127.0.0.1", 0), dashboard.DashboardHandler)
    server.lan_access_enabled = True
    server.dashboard_public_host = "192.168.50.20"
    server.dashboard_credentials = credentials
    server.dashboard_auth_lock = threading.Lock()
    server.dashboard_auth_slots = threading.BoundedSemaphore(1)
    server.dashboard_stream_slots = threading.BoundedSemaphore(
        dashboard.DASHBOARD_STREAM_LIMIT
    )
    server.dashboard_login_attempts = {}
    server.dashboard_sessions = {}
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    headers = {
        "Host": f"192.168.50.20:{port}",
        "Origin": f"https://192.168.50.20:{port}",
    }
    try:
        with dashboard.httpx.Client(follow_redirects=False, timeout=2) as client:
            challenge = client.get(f"http://{host}:{port}/", headers=headers)
            rejected = client.get(
                f"http://{host}:{port}/", headers=headers, auth=("admin", "wrong")
            )
            assert server.dashboard_auth_slots.acquire(blocking=False)
            saturated = client.get(
                f"http://{host}:{port}/",
                headers=headers,
                auth=("admin", "correct horse battery staple"),
            )
            server.dashboard_auth_slots.release()
            accepted = client.get(
                f"http://{host}:{port}/",
                headers=headers,
                auth=("admin", "correct horse battery staple"),
            )
            session_cookie = accepted.headers["set-cookie"].split(";", 1)[0]
            page = client.get(
                f"http://{host}:{port}/",
                headers={**headers, "Cookie": session_cookie},
            )
            duplicate_cookie = client.get(
                f"http://{host}:{port}/",
                headers=[
                    *headers.items(),
                    ("Cookie", session_cookie),
                    ("Cookie", session_cookie),
                ],
            )
            server.dashboard_sessions.clear()
            server.dashboard_login_attempts[host] = [
                time.monotonic()
            ] * dashboard.DASHBOARD_LOGIN_ATTEMPT_LIMIT
            rate_limited = client.get(
                f"http://{host}:{port}/",
                headers=headers,
                auth=("admin", "correct horse battery staple"),
            )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert challenge.status_code == 401
    assert rejected.status_code == 401
    assert saturated.status_code == 429
    assert accepted.status_code == 303
    assert accepted.headers["location"] == "/"
    assert "Secure" in accepted.headers["set-cookie"]
    assert "HttpOnly" in accepted.headers["set-cookie"]
    assert "SameSite=Strict" in accepted.headers["set-cookie"]
    assert page.status_code == 200
    assert duplicate_cookie.status_code == 401
    assert rate_limited.status_code == 429
    assert len(server.dashboard_sessions) == 0
    assert credentials_path.stat().st_mode & 0o777 == 0o600
    assert "correct horse battery staple" not in credentials_path.read_text()


def test_dashboard_lan_tls_server_and_private_key_permissions(
    tmp_path: Path, monkeypatch
) -> None:
    key = tmp_path / "dashboard.key"
    cert = tmp_path / "dashboard.crt"
    credentials_path = tmp_path / "dashboard-credentials.json"
    subprocess.run(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-days",
            "1",
            "-keyout",
            str(key),
            "-out",
            str(cert),
            "-subj",
            "/CN=192.168.50.20",
        ],
        check=True,
        capture_output=True,
    )
    key.chmod(0o600)
    dashboard._write_dashboard_credentials(
        credentials_path, "admin", "correct horse battery staple"
    )
    monkeypatch.setattr(dashboard, "_private_client_scope", lambda _value: "private")
    server = dashboard._ThreadingHTTPSServer(
        ("127.0.0.1", 0), dashboard.DashboardHandler, certfile=cert, keyfile=key
    )
    server.lan_access_enabled = True
    server.dashboard_public_host = "192.168.50.20"
    server.dashboard_credentials = dashboard._load_dashboard_credentials(
        credentials_path
    )
    server.dashboard_auth_lock = threading.Lock()
    server.dashboard_auth_slots = threading.BoundedSemaphore(
        dashboard.DASHBOARD_AUTH_CONCURRENCY_LIMIT
    )
    server.dashboard_stream_slots = threading.BoundedSemaphore(
        dashboard.DASHBOARD_STREAM_LIMIT
    )
    server.dashboard_login_attempts = {}
    server.dashboard_sessions = {}
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        response = dashboard.httpx.get(
            f"https://{host}:{port}/",
            headers={
                "Host": f"192.168.50.20:{port}",
                "Origin": f"https://192.168.50.20:{port}",
            },
            auth=("admin", "correct horse battery staple"),
            verify=False,
            follow_redirects=False,
            timeout=2,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert response.status_code == 303

    key.chmod(0o644)
    with pytest.raises(RuntimeError, match="0600"):
        dashboard._read_private_file(key, "dashboard TLS private key")
    link = tmp_path / "linked.key"
    link.symlink_to(key)
    with pytest.raises(RuntimeError, match="unavailable"):
        dashboard._read_private_file(link, "dashboard TLS private key")


def test_dashboard_tls_slow_handshake_does_not_block_and_handler_slots_recover(
    tmp_path: Path, monkeypatch
) -> None:
    key = tmp_path / "dashboard.key"
    cert = tmp_path / "dashboard.crt"
    credentials_path = tmp_path / "dashboard-credentials.json"
    subprocess.run(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-days",
            "1",
            "-keyout",
            str(key),
            "-out",
            str(cert),
            "-subj",
            "/CN=192.168.50.20",
        ],
        check=True,
        capture_output=True,
    )
    key.chmod(0o600)
    dashboard._write_dashboard_credentials(
        credentials_path, "admin", "correct horse battery staple"
    )
    monkeypatch.setattr(dashboard, "_private_client_scope", lambda _value: "private")
    server = dashboard._ThreadingHTTPSServer(
        ("127.0.0.1", 0),
        dashboard.DashboardHandler,
        certfile=cert,
        keyfile=key,
        handshake_timeout=2,
        io_timeout=0.2,
        handler_limit=2,
    )
    server.lan_access_enabled = True
    server.dashboard_public_host = "192.168.50.20"
    server.dashboard_credentials = dashboard._load_dashboard_credentials(
        credentials_path
    )
    server.dashboard_auth_lock = threading.Lock()
    server.dashboard_auth_slots = threading.BoundedSemaphore(1)
    server.dashboard_stream_slots = threading.BoundedSemaphore(2)
    server.dashboard_login_attempts = {}
    server.dashboard_sessions = {}
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    slow = socket.create_connection((host, port), timeout=2)
    try:
        response = dashboard.httpx.get(
            f"https://{host}:{port}/",
            headers={
                "Host": f"192.168.50.20:{port}",
                "Origin": f"https://192.168.50.20:{port}",
            },
            auth=("admin", "correct horse battery staple"),
            verify=False,
            follow_redirects=False,
            timeout=1,
        )
        assert response.status_code == 303
        assert server.dashboard_handler_slots.acquire(blocking=False)
        assert not server.dashboard_handler_slots.acquire(blocking=False)
        rejected = socket.create_connection((host, port), timeout=2)
        rejected.settimeout(2)
        assert rejected.recv(1) == b""
        rejected.close()
        server.dashboard_handler_slots.release()
    finally:
        slow.close()

    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        acquired = [
            server.dashboard_handler_slots.acquire(blocking=False) for _ in range(2)
        ]
        for value in acquired:
            if value:
                server.dashboard_handler_slots.release()
        if all(acquired):
            break
        time.sleep(0.01)
    client_context = ssl._create_unverified_context()
    idle = client_context.wrap_socket(
        socket.create_connection((host, port), timeout=2),
        server_hostname="192.168.50.20",
    )
    time.sleep(0.3)
    idle.close()
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        post_handshake = [
            server.dashboard_handler_slots.acquire(blocking=False) for _ in range(2)
        ]
        for value in post_handshake:
            if value:
                server.dashboard_handler_slots.release()
        if all(post_handshake):
            break
        time.sleep(0.01)
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)
    assert all(acquired)
    assert all(post_handshake)


def test_dashboard_lan_websocket_accepts_basic_only_on_exact_https_origin(
    tmp_path: Path, monkeypatch
) -> None:
    class ClosingCursor:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def poll_payload(self) -> dict[str, object]:
            raise BrokenPipeError

    credentials_path = tmp_path / "dashboard-credentials.json"
    dashboard._write_dashboard_credentials(
        credentials_path, "admin", "correct horse battery staple"
    )
    monkeypatch.setattr(dashboard, "_private_client_scope", lambda _value: "private")
    monkeypatch.setattr(dashboard, "CortexEventCursor", ClosingCursor)
    server = dashboard.ThreadingHTTPServer(("127.0.0.1", 0), dashboard.DashboardHandler)
    server.lan_access_enabled = True
    server.dashboard_public_host = "192.168.50.20"
    server.dashboard_credentials = dashboard._load_dashboard_credentials(
        credentials_path
    )
    server.dashboard_auth_lock = threading.Lock()
    server.dashboard_auth_slots = threading.BoundedSemaphore(
        dashboard.DASHBOARD_AUTH_CONCURRENCY_LIMIT
    )
    server.dashboard_stream_slots = threading.BoundedSemaphore(2)
    server.dashboard_login_attempts = {}
    server.dashboard_sessions = {}
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    authorization = base64.b64encode(b"admin:correct horse battery staple").decode(
        "ascii"
    )
    session_token = "browser-session-token"
    server.dashboard_sessions[session_token] = time.monotonic() + 60

    def websocket_status(
        *,
        origin: str,
        auth: str | None,
        cookie: str | None = None,
        key: str = "dGhlIHNhbXBsZSBub25jZQ==",
    ) -> int:
        auth_header = f"Authorization: Basic {auth}\r\n" if auth else ""
        cookie_header = f"Cookie: {cookie}\r\n" if cookie else ""
        request = (
            "GET /api/cortex/events HTTP/1.1\r\n"
            f"Host: 192.168.50.20:{port}\r\n"
            f"Origin: {origin}\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            "Sec-WebSocket-Version: 13\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            f"{auth_header}{cookie_header}\r\n"
        ).encode("ascii")
        with socket.create_connection((host, port), timeout=2) as connection:
            connection.sendall(request)
            return int(connection.recv(4096).split(b"\r\n", 1)[0].split()[1])

    try:
        missing = websocket_status(origin=f"https://192.168.50.20:{port}", auth=None)
        wrong_origin = websocket_status(
            origin="https://attacker.example", auth=authorization
        )
        invalid_key = websocket_status(
            origin=f"https://192.168.50.20:{port}",
            auth=authorization,
            key="invalid",
        )
        assert server.dashboard_stream_slots.acquire(blocking=False)
        assert server.dashboard_stream_slots.acquire(blocking=False)
        server.dashboard_stream_slots.release()
        server.dashboard_stream_slots.release()
        accepted = websocket_status(
            origin=f"https://192.168.50.20:{port}", auth=authorization
        )
        assert server.dashboard_stream_slots.acquire(blocking=False)
        assert server.dashboard_stream_slots.acquire(blocking=False)
        stream_saturated = websocket_status(
            origin=f"https://192.168.50.20:{port}", auth=authorization
        )
        sse_saturated = dashboard.httpx.get(
            f"http://{host}:{port}/api/activity-stream",
            headers={
                "Host": f"192.168.50.20:{port}",
                "Origin": f"https://192.168.50.20:{port}",
                "Cookie": (f"{dashboard.DASHBOARD_SESSION_COOKIE}={session_token}"),
            },
            timeout=2,
        )
        server.dashboard_stream_slots.release()
        server.dashboard_stream_slots.release()
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            acquired = [
                server.dashboard_stream_slots.acquire(blocking=False)
                for _ in range(2)
            ]
            for value in acquired:
                if value:
                    server.dashboard_stream_slots.release()
            if all(acquired):
                break
            time.sleep(0.01)
        else:
            pytest.fail("stream slots were not released")
        session_accepted = websocket_status(
            origin=f"https://192.168.50.20:{port}",
            auth=None,
            cookie=f"{dashboard.DASHBOARD_SESSION_COOKIE}={session_token}",
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert missing == 401
    assert wrong_origin == 403
    assert invalid_key == 400
    assert accepted == 101
    assert stream_saturated == 429
    assert sse_saturated.status_code == 429
    assert session_accepted == 101


def test_dashboard_cli_rejects_unconfigured_lan_and_non_loopback_host(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        dashboard,
        "runtime_identity",
        lambda *, config_only=False: {
            "dashboard": {"host": "127.0.0.1", "port": 8765},
            "dashboard_lan": {
                "host": None,
                "port": 8766,
                "tls_cert_file": None,
                "tls_key_file": None,
                "credentials_file": None,
            },
        },
    )
    with pytest.raises(SystemExit) as lan:
        dashboard.main(["--lan"])
    with pytest.raises(SystemExit) as public_host:
        dashboard.main(["--host", "0.0.0.0"])
    with pytest.raises(SystemExit) as ipv6_host:
        dashboard.main(["--host", "::1"])

    assert lan.value.code == 2
    assert public_host.value.code == 2
    assert ipv6_host.value.code == 2


def test_dashboard_cli_reports_short_lan_password_without_traceback_or_secret(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    credentials_file = tmp_path / "dashboard-lan-credentials.json"
    prompts: list[str] = []
    secret = "short7"
    monkeypatch.setattr(
        dashboard,
        "runtime_identity",
        lambda *, config_only=False: {
            "dashboard": {"host": "127.0.0.1", "port": 8765},
            "dashboard_lan": {
                "host": None,
                "port": 8766,
                "tls_cert_file": None,
                "tls_key_file": None,
                "credentials_file": str(credentials_file),
            },
        },
    )
    monkeypatch.setattr(
        dashboard.getpass,
        "getpass",
        lambda prompt: prompts.append(prompt) or secret,
    )

    with pytest.raises(SystemExit) as result:
        dashboard.main(["--set-lan-credentials"])

    captured = capsys.readouterr()
    assert result.value.code == 2
    assert "minimum 8 characters" in prompts[0]
    assert "at least 8 characters" in captured.err
    assert "Traceback" not in captured.err
    assert secret not in captured.err
    assert not credentials_file.exists()


def test_dashboard_parser_uses_configured_loopback_bind(monkeypatch) -> None:
    monkeypatch.setattr(
        dashboard,
        "runtime_identity",
        lambda *, config_only=False: {
            "dashboard": {"host": "localhost", "port": 9876}
        },
    )

    args = dashboard.build_parser().parse_args([])

    assert (args.host, args.port) == ("localhost", 9876)


def test_snapshot_fingerprint_probe_single_flights_concurrent_callers(
    monkeypatch,
) -> None:
    _reset_snapshot_fingerprint_cache()
    entered = threading.Event()
    release = threading.Event()
    calls = 0

    def build_fingerprint() -> tuple[str, int]:
        nonlocal calls
        calls += 1
        entered.set()
        assert release.wait(timeout=2)
        return ("fingerprint", calls)

    monkeypatch.setattr(
        dashboard, "_snapshot_source_probe_identity", lambda: ("source",)
    )
    monkeypatch.setattr(
        dashboard, "_build_snapshot_source_fingerprint", build_fingerprint
    )

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [
            pool.submit(dashboard._snapshot_source_fingerprint) for _ in range(8)
        ]
        assert entered.wait(timeout=2)
        release.set()
        fingerprints = [future.result(timeout=2) for future in futures]

    assert calls == 1
    assert fingerprints == [("fingerprint", 1)] * 8
    assert dashboard._SNAPSHOT_FINGERPRINT_CACHE["probe_count"] == 1
    assert dashboard._SNAPSHOT_FINGERPRINT_CACHE["coalesced"] >= 1


def test_snapshot_fingerprint_probe_reuses_audit_and_invalidates_source(
    monkeypatch,
) -> None:
    _reset_snapshot_fingerprint_cache()
    clock = [400.0]
    source = [("source", 1)]
    calls = 0

    def build_fingerprint() -> tuple[str, int]:
        nonlocal calls
        calls += 1
        return ("fingerprint", calls)

    monkeypatch.setattr(dashboard.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        dashboard, "_snapshot_source_probe_identity", lambda: source[0]
    )
    monkeypatch.setattr(
        dashboard, "_build_snapshot_source_fingerprint", build_fingerprint
    )

    assert dashboard._snapshot_source_fingerprint() == ("fingerprint", 1)
    assert dashboard._snapshot_source_fingerprint() == ("fingerprint", 1)
    clock[0] += dashboard.SNAPSHOT_FINGERPRINT_AUDIT_SECONDS - 0.001
    assert dashboard._snapshot_source_fingerprint() == ("fingerprint", 1)

    source[0] = ("source", 2)
    assert dashboard._snapshot_source_fingerprint() == ("fingerprint", 2)
    clock[0] += dashboard.SNAPSHOT_FINGERPRINT_AUDIT_SECONDS
    assert dashboard._snapshot_source_fingerprint() == ("fingerprint", 3)
    assert calls == 3


def test_snapshot_fingerprint_probe_releases_waiters_after_error(
    monkeypatch,
) -> None:
    _reset_snapshot_fingerprint_cache()
    calls = 0

    def build_fingerprint() -> tuple[str, int]:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("probe failed")
        return ("fingerprint", calls)

    monkeypatch.setattr(
        dashboard, "_snapshot_source_probe_identity", lambda: ("source",)
    )
    monkeypatch.setattr(
        dashboard, "_build_snapshot_source_fingerprint", build_fingerprint
    )

    with pytest.raises(RuntimeError, match="probe failed"):
        dashboard._snapshot_source_fingerprint()
    assert dashboard._SNAPSHOT_FINGERPRINT_CACHE["probing"] is False
    assert dashboard._SNAPSHOT_FINGERPRINT_CACHE["error_count"] == 1
    assert dashboard._snapshot_source_fingerprint() == ("fingerprint", 2)


def test_snapshot_probe_identity_isolates_root_and_builder(
    tmp_path: Path, monkeypatch
) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    monkeypatch.setattr(dashboard, "CHRONOVISOR_ROOT", first_root)
    first = dashboard._snapshot_source_probe_identity()

    monkeypatch.setattr(dashboard, "CHRONOVISOR_ROOT", second_root)
    second = dashboard._snapshot_source_probe_identity()
    assert first != second

    original_builder = dashboard.build_snapshot
    monkeypatch.setattr(dashboard, "build_snapshot", lambda: {})
    replaced_builder = dashboard._snapshot_source_probe_identity()
    assert replaced_builder != second
    monkeypatch.setattr(dashboard, "build_snapshot", original_builder)


def test_snapshot_fingerprint_tracks_explicit_runtime_cold_sources(
    tmp_path: Path, monkeypatch
) -> None:
    chronovisor_root = tmp_path / "wiki"
    runtime_dir = chronovisor_root / "runtime"
    artifact_dir = runtime_dir / "raw-projections" / "artifacts"
    artifact_dir.mkdir(parents=True)
    monkeypatch.setattr(dashboard, "CHRONOVISOR_ROOT", chronovisor_root)
    monkeypatch.setattr(
        dashboard,
        "ACTIVITY_FILE",
        chronovisor_root / "runtime" / "activity.jsonl",
    )
    monkeypatch.setattr(
        orchestrator, "STATE_FILE", chronovisor_root / ".orchestrator_state.json"
    )
    monkeypatch.setattr(runtime_status, "STATUS_FILE", runtime_dir / "status.json")
    monkeypatch.setattr(runtime_status, "EVENTS_FILE", runtime_dir / "events.jsonl")
    monkeypatch.setattr(runtime_status, "METRICS_FILE", runtime_dir / "metrics.jsonl")

    before = dashboard._build_snapshot_source_fingerprint()
    temporary = artifact_dir / ".raw.tmp"
    temporary.write_text("projection\n", encoding="utf-8")
    os.replace(temporary, artifact_dir / "raw.md")
    after_projection = dashboard._build_snapshot_source_fingerprint()
    assert after_projection != before

    (runtime_dir / "ingest-liveness.json").write_text("{}\n", encoding="utf-8")
    after_liveness = dashboard._build_snapshot_source_fingerprint()
    assert after_liveness != after_projection


def test_cached_snapshot_and_fingerprint_probe_have_no_lock_cycle(
    monkeypatch,
) -> None:
    _reset_snapshot_fingerprint_cache()
    dashboard._SNAPSHOT_CACHE.update(
        {
            "built_at": 0.0,
            "fingerprint": None,
            "snapshot": None,
            "refreshing": False,
        }
    )
    builds = 0

    def build_snapshot() -> dict:
        nonlocal builds
        builds += 1
        return {
            "serial": builds,
            "status": {"state": "idle", "batch": {"active": False}},
        }

    monkeypatch.setattr(
        dashboard, "_snapshot_source_probe_identity", lambda: ("source",)
    )
    monkeypatch.setattr(
        dashboard,
        "_build_snapshot_source_fingerprint",
        lambda: ("fingerprint",),
    )
    monkeypatch.setattr(dashboard, "build_snapshot", build_snapshot)

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(dashboard._cached_snapshot) for _ in range(8)]
        snapshots = [future.result(timeout=3) for future in futures]

    assert {snapshot["serial"] for snapshot in snapshots} == {1}
    assert builds == 1


def test_cached_snapshot_ignores_live_status_churn_but_refreshes_cold_sources(
    tmp_path: Path, monkeypatch
) -> None:
    _reset_snapshot_fingerprint_cache()
    chronovisor_root = tmp_path / "wiki"
    runtime_dir = chronovisor_root / "runtime"
    runtime_dir.mkdir(parents=True)
    monkeypatch.setattr(dashboard, "CHRONOVISOR_ROOT", chronovisor_root)
    monkeypatch.setattr(
        dashboard,
        "ACTIVITY_FILE",
        chronovisor_root / "runtime" / "activity.jsonl",
    )
    monkeypatch.setattr(
        orchestrator, "STATE_FILE", chronovisor_root / ".orchestrator_state.json"
    )
    monkeypatch.setattr(runtime_status, "STATUS_FILE", runtime_dir / "status.json")
    monkeypatch.setattr(runtime_status, "EVENTS_FILE", runtime_dir / "events.jsonl")
    monkeypatch.setattr(runtime_status, "METRICS_FILE", runtime_dir / "metrics.jsonl")
    dashboard._SNAPSHOT_CACHE.update(
        {
            "built_at": 0.0,
            "fingerprint": None,
            "snapshot": None,
            "refreshing": False,
        }
    )
    calls = 0

    def fake_build() -> dict:
        nonlocal calls
        calls += 1
        return {
            "serial": calls,
            "status": {"state": "idle", "batch": {"active": False}},
        }

    monkeypatch.setattr(dashboard, "build_snapshot", fake_build)
    runtime_status.STATUS_FILE.write_text(
        json.dumps(
            {
                "state": "running",
                "stage": "raw",
                "current_job_id": "job-1",
                "current_raw": "raw.md",
                "updated_at": "2026-08-05T10:00:00+09:00",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert dashboard._cached_snapshot()["serial"] == 1
    assert dashboard._cached_snapshot()["serial"] == 1
    temporary = runtime_dir / ".status.tmp"
    temporary.write_text(
        json.dumps(
            {
                "state": "running",
                "stage": "generate",
                "current_job_id": "job-1",
                "current_raw": "raw.md",
                "updated_at": "2026-08-05T10:00:01+09:00",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, runtime_status.STATUS_FILE)
    # Force a full probe to prove neither status.json nor the runtime root
    # directory identity participates in the cold fingerprint.
    dashboard._invalidate_snapshot_fingerprint_probe()
    assert dashboard._cached_snapshot()["serial"] == 1

    runtime_status.EVENTS_FILE.write_text("{}\n", encoding="utf-8")
    assert dashboard._cached_snapshot()["serial"] == 2
    assert calls == 2


def test_cached_snapshot_refreshes_on_semantic_runtime_epoch_change(
    tmp_path: Path, monkeypatch
) -> None:
    _reset_snapshot_fingerprint_cache()
    chronovisor_root = tmp_path / "wiki"
    runtime_dir = chronovisor_root / "runtime"
    runtime_dir.mkdir(parents=True)
    monkeypatch.setattr(dashboard, "CHRONOVISOR_ROOT", chronovisor_root)
    monkeypatch.setattr(
        dashboard,
        "ACTIVITY_FILE",
        chronovisor_root / "runtime" / "activity.jsonl",
    )
    monkeypatch.setattr(
        orchestrator, "STATE_FILE", chronovisor_root / ".orchestrator_state.json"
    )
    monkeypatch.setattr(runtime_status, "STATUS_FILE", runtime_dir / "status.json")
    monkeypatch.setattr(runtime_status, "EVENTS_FILE", runtime_dir / "events.jsonl")
    monkeypatch.setattr(runtime_status, "METRICS_FILE", runtime_dir / "metrics.jsonl")
    dashboard._SNAPSHOT_CACHE.update(
        {
            "built_at": 0.0,
            "fingerprint": None,
            "snapshot": None,
            "refreshing": False,
        }
    )
    runtime_status.STATUS_FILE.write_text(
        '{"state": "idle", "stage": "idle"}\n', encoding="utf-8"
    )
    calls = 0

    def fake_build() -> dict:
        nonlocal calls
        calls += 1
        live = runtime_status.read_status()
        active = live.get("state") == "running"
        return {
            "serial": calls,
            "status": {
                "state": "running" if active else "idle",
                "batch": {"active": active},
            },
        }

    monkeypatch.setattr(dashboard, "build_snapshot", fake_build)

    assert dashboard._cached_snapshot()["serial"] == 1
    temporary = runtime_dir / ".status.tmp"
    temporary.write_text(
        json.dumps(
            {
                "state": "running",
                "stage": "raw",
                "current_job_id": "job-2",
                "current_raw": "next.md",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, runtime_status.STATUS_FILE)

    refreshed = dashboard._cached_snapshot()
    assert refreshed["serial"] == 2
    assert refreshed["status"]["state"] == "running"
    assert calls == 2


def test_cached_snapshot_uses_30_second_cold_ttl_while_active(
    tmp_path: Path, monkeypatch
) -> None:
    _reset_snapshot_fingerprint_cache()
    chronovisor_root = tmp_path / "wiki"
    chronovisor_root.mkdir()
    monkeypatch.setattr(dashboard, "CHRONOVISOR_ROOT", chronovisor_root)
    monkeypatch.setattr(
        dashboard,
        "ACTIVITY_FILE",
        chronovisor_root / "runtime" / "activity.jsonl",
    )
    monkeypatch.setattr(
        orchestrator, "STATE_FILE", chronovisor_root / ".orchestrator_state.json"
    )
    dashboard._SNAPSHOT_CACHE.update(
        {
            "built_at": 0.0,
            "fingerprint": None,
            "snapshot": None,
            "refreshing": False,
        }
    )
    clock = [100.0]
    calls = 0

    def fake_build() -> dict:
        nonlocal calls
        calls += 1
        return {
            "serial": calls,
            "status": {"state": "running", "batch": {"active": True}},
        }

    monkeypatch.setattr(dashboard, "build_snapshot", fake_build)
    monkeypatch.setattr(dashboard.time, "monotonic", lambda: clock[0])

    assert dashboard._cached_snapshot()["serial"] == 1
    assert dashboard.SNAPSHOT_ACTIVE_CACHE_SECONDS == 30.0
    clock[0] += dashboard.SNAPSHOT_ACTIVE_CACHE_SECONDS - 0.001
    assert dashboard._cached_snapshot()["serial"] == 1
    clock[0] += 0.001
    assert dashboard._cached_snapshot()["serial"] == 2
    assert calls == 2


def test_snapshot_live_status_overlay_is_non_mutating_and_preserves_cold_fields(
    monkeypatch,
) -> None:
    cached = {
        "status": {
            "state": "idle",
            "stage": "idle",
            "pending": 7,
            "batch": {"index": 4, "total": 4, "active": False},
            "local_consensus": {"active": False, "count": 11},
            "frontier_repair": {"active": False, "summary": {"total": 3}},
            "frontier_review": {"active": False, "count": 2},
            "decision_policies": {"lanes": {"ingest": "enabled"}},
            "semantic_deferred": {"count": 2, "samples": ["semantic.md"]},
            "operational_deferred": {"count": 1, "samples": ["blocked.md"]},
            "raw_outstanding": 10,
        },
        "events": [{"kind": "cold"}],
        "local_consensus": {"active": False, "count": 11},
        "_dashboard": {"detail_state": "ready"},
    }
    original = json.loads(json.dumps(cached))
    monkeypatch.setattr(
        runtime_status,
        "read_status",
        lambda: {
            "state": "running",
            "stage": "generate",
            "current_job_id": "job-live",
            "current_job_pid": 42,
            "current_raw": "new.md",
            "current_op": "create",
            "batch": {"index": 1, "total": 3},
            "llm": {"active": True, "model": "live:model"},
            "semantic_deferred": {"count": 999},
        },
    )
    monkeypatch.setattr(
        orchestrator,
        "_load_state",
        lambda: {
            "current_job_id": "job-live",
            "current_job_pid": 42,
            "current_job_started_at": "2026-08-05T10:00:00+09:00",
        },
    )
    monkeypatch.setattr(
        orchestrator, "ingest_process_lease_is_held", lambda _pid: True
    )
    monkeypatch.setattr(
        dashboard, "_job_process_identity_matches", lambda _pid, _started: True
    )

    overlaid = dashboard._snapshot_with_live_status(cached)

    assert overlaid["status"]["state"] == "running"
    assert overlaid["status"]["stage"] == "generate"
    assert overlaid["status"]["current_job_id"] == "job-live"
    assert overlaid["status"]["current_raw"] == "new.md"
    assert overlaid["status"]["pending"] == 7
    assert overlaid["status"]["batch"]["active"] is True
    for key in dashboard._COLD_STATUS_DERIVED_KEYS:
        assert overlaid["status"][key] == cached["status"][key]
    assert overlaid["events"] == [{"kind": "cold"}]
    assert overlaid["_dashboard"] == {
        "detail_state": "ready",
        "live_overlay": True,
    }
    assert cached == original


def test_snapshot_live_status_overlay_reflects_idle_to_active_next_response(
    monkeypatch,
) -> None:
    cached = {
        "status": {
            "state": "idle",
            "stage": "idle",
            "pending": 1,
            "batch": {"active": False},
        }
    }
    live = [{"state": "idle", "stage": "idle", "batch": {"active": False}}]
    orch = [{}]
    monkeypatch.setattr(runtime_status, "read_status", lambda: dict(live[0]))
    monkeypatch.setattr(orchestrator, "_load_state", lambda: dict(orch[0]))
    monkeypatch.setattr(
        orchestrator, "ingest_process_lease_is_held", lambda _pid: True
    )
    monkeypatch.setattr(
        dashboard, "_job_process_identity_matches", lambda _pid, _started: True
    )

    first = dashboard._snapshot_with_live_status(cached)
    assert first["status"]["state"] == "idle"

    live[0] = {
        "state": "running",
        "stage": "raw",
        "current_job_id": "job-2",
        "current_job_pid": 84,
        "current_raw": "raw.md",
        "batch": {"index": 1, "total": 2},
    }
    orch[0] = {
        "current_job_id": "job-2",
        "current_job_pid": 84,
        "current_job_started_at": "2026-08-05T11:00:00+09:00",
    }
    second = dashboard._snapshot_with_live_status(cached)

    assert second["status"]["state"] == "running"
    assert second["status"]["stage"] == "raw"
    assert second["status"]["current_job_id"] == "job-2"
    assert second["status"]["batch"]["active"] is True
    assert cached["status"]["state"] == "idle"


def test_snapshot_live_status_overlay_failure_keeps_cached_snapshot(
    monkeypatch,
) -> None:
    cached = {
        "status": {"state": "idle", "stage": "idle", "pending": 2},
        "events": [{"kind": "cold"}],
        "_dashboard": {"detail_state": "refreshing", "stale": True},
    }
    original = json.loads(json.dumps(cached))
    monkeypatch.setattr(
        runtime_status,
        "read_status",
        lambda: (_ for _ in ()).throw(RuntimeError("status unavailable")),
    )

    overlaid = dashboard._snapshot_with_live_status(cached)

    assert overlaid["status"] == cached["status"]
    assert overlaid["events"] == cached["events"]
    assert overlaid["_dashboard"] == {
        "detail_state": "refreshing",
        "stale": True,
        "live_overlay": False,
        "live_overlay_error": "RuntimeError",
    }
    assert cached == original


def test_snapshot_routes_apply_live_overlay() -> None:
    source = Path(dashboard.__file__).read_text(encoding="utf-8")
    snapshot_route = source.split('elif path == "/api/snapshot":', 1)[1].split(
        'elif path == "/api/status":', 1
    )[0]
    status_route = source.split('elif path == "/api/status":', 1)[1].split(
        'elif path == "/api/local-consensus":', 1
    )[0]

    assert "_snapshot_with_live_status(" in snapshot_route
    assert "_cached_snapshot(allow_stale=True)" in snapshot_route
    assert "_snapshot_with_live_status(" in status_route
    assert '"_dashboard": snapshot.get("_dashboard") or {}' in status_route


def test_local_consensus_route_stays_on_direct_live_path() -> None:
    source = Path(dashboard.__file__).read_text(encoding="utf-8")
    local_consensus_route = source.split(
        'elif path == "/api/local-consensus":', 1
    )[1].split('elif path == "/api/activity":', 1)[0]
    processing_source = source.split(
        "def _processing_activity_source_fingerprint", 1
    )[1].split("def _processing_activity_cache_metrics_locked", 1)[0]

    assert "_local_consensus_snapshot(" in local_consensus_route
    assert "preferred_request_sha256=preferred_request_sha256" in local_consensus_route
    assert 'next_active=next_trace == "active"' in local_consensus_route
    assert "_cached_snapshot" not in local_consensus_route
    assert (
        'CHRONOVISOR_ROOT / "runtime" / "local-consensus" / "active"'
        in processing_source
    )


def test_cached_snapshot_serves_stale_while_refreshing_in_background(
    monkeypatch,
) -> None:
    _reset_snapshot_fingerprint_cache()
    dashboard._SNAPSHOT_CACHE.update(
        {
            "built_at": 0.0,
            "fingerprint": ("old",),
            "snapshot": {"serial": 1, "status": {"state": "idle"}},
            "refreshing": False,
        }
    )
    calls = 0

    def fake_build() -> dict:
        nonlocal calls
        calls += 1
        return {"serial": 2, "status": {"state": "idle"}}

    class ImmediateThread:
        def __init__(self, *, target, args, **_kwargs) -> None:
            self.target = target
            self.args = args

        def start(self) -> None:
            self.target(*self.args)

    monkeypatch.setattr(dashboard, "build_snapshot", fake_build)
    monkeypatch.setattr(
        dashboard, "_snapshot_source_fingerprint", lambda: ("new",)
    )
    monkeypatch.setattr(dashboard.threading, "Thread", ImmediateThread)

    stale = dashboard._cached_snapshot(allow_stale=True)

    assert stale["serial"] == 1
    assert stale["_dashboard"] == {
        "detail_state": "refreshing",
        "stale": True,
    }
    assert dashboard._SNAPSHOT_CACHE["snapshot"]["serial"] == 2
    assert dashboard._SNAPSHOT_CACHE["refreshing"] is False
    assert calls == 1


def test_fast_snapshot_reads_status_without_building_archive_components(
    monkeypatch,
) -> None:
    from chronovisor.recall import librarian_status

    monkeypatch.setattr(
        dashboard,
        "init_chronovisor",
        lambda: pytest.fail("fast snapshot must not initialize the store"),
    )
    monkeypatch.setattr(
        runtime_status,
        "read_status",
        lambda: {"state": "running", "stage": "generate", "pending": 3},
    )
    monkeypatch.setattr(
        runtime_status, "read_events", lambda limit: [{"kind": "event"}]
    )
    monkeypatch.setattr(
        runtime_status, "read_metrics", lambda limit: [{"kind": "metric"}]
    )
    monkeypatch.setattr(
        dashboard,
        "_save_history_snapshot",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("fast snapshot must not scan save history")
        ),
    )
    monkeypatch.setattr(
        librarian_status,
        "build_librarian_status",
        lambda *_args, **_kwargs: pytest.fail(
            "fast snapshot must not scan librarian state"
        ),
    )

    snapshot = dashboard.build_fast_snapshot()

    assert snapshot["status"]["pending"] == 3
    assert snapshot["events"] == [{"kind": "event"}]
    assert snapshot["metrics"] == [{"kind": "metric"}]
    assert snapshot["save_history"] == {}
    assert snapshot["librarian"] == {}
    assert snapshot["_dashboard"] == {"detail_state": "loading"}


def test_materialized_component_survives_process_memory_reset_and_rejects_tamper(
    tmp_path: Path, monkeypatch
) -> None:
    chronovisor_root = tmp_path / "wiki"
    monkeypatch.setattr(dashboard, "CHRONOVISOR_ROOT", chronovisor_root)
    monkeypatch.setattr(dashboard.time, "time", lambda: 100.0)
    calls = 0

    def build() -> dict:
        nonlocal calls
        calls += 1
        return {"serial": calls}

    assert dashboard._materialized_component(
        "test-view",
        fingerprint="a" * 64,
        builder=build,
        audit_seconds=60,
    ) == {"serial": 1}
    cache_key = (str(chronovisor_root), "test-view")
    dashboard._MATERIALIZED_COMPONENTS.pop(cache_key, None)

    assert dashboard._materialized_component(
        "test-view",
        fingerprint="a" * 64,
        builder=build,
        audit_seconds=60,
    ) == {"serial": 1}
    assert calls == 1

    cache_path = dashboard._materialized_component_path("test-view")
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    payload["value"]["serial"] = 999
    cache_path.write_text(json.dumps(payload), encoding="utf-8")
    dashboard._MATERIALIZED_COMPONENTS.pop(cache_key, None)

    assert dashboard._materialized_component(
        "test-view",
        fingerprint="a" * 64,
        builder=build,
        audit_seconds=60,
    ) == {"serial": 2}
    assert calls == 2


def test_health_materialization_fingerprint_changes_with_runtime(
    tmp_path: Path,
    monkeypatch,
) -> None:
    chronovisor_root = tmp_path / "wiki"
    chronovisor_root.mkdir()
    monkeypatch.setattr(dashboard, "CHRONOVISOR_ROOT", chronovisor_root)
    runtime = {
        "commit_id": "a" * 40,
        "module_path": "/archive/a/chronovisor/runtime_config.py",
        "package_version": "0.1.1",
    }
    monkeypatch.setattr(dashboard, "runtime_identity", lambda: dict(runtime))

    before = dashboard._health_materialization_fingerprint([])
    runtime["commit_id"] = "b" * 40
    runtime["module_path"] = "/archive/b/chronovisor/runtime_config.py"
    after = dashboard._health_materialization_fingerprint([])

    assert before != after


def test_health_materialization_fingerprint_tracks_ingest_liveness(
    tmp_path: Path,
    monkeypatch,
) -> None:
    chronovisor_root = tmp_path / "wiki"
    runtime_dir = chronovisor_root / "runtime"
    runtime_dir.mkdir(parents=True)
    monkeypatch.setattr(dashboard, "CHRONOVISOR_ROOT", chronovisor_root)
    monkeypatch.setattr(
        dashboard,
        "runtime_identity",
        lambda: {
            "commit_id": "a" * 40,
            "module_path": "/archive/a/chronovisor/runtime_config.py",
            "package_version": "0.1.1",
        },
    )

    before = dashboard._health_materialization_fingerprint([])
    (runtime_dir / "ingest-liveness.json").write_text(
        json.dumps(
            {
                "status": "blocked_by_decision_authority",
                "alert": True,
            }
        ),
        encoding="utf-8",
    )
    after = dashboard._health_materialization_fingerprint([])

    assert before != after


def test_health_materialization_fingerprint_tracks_distillation_status_and_pointers(
    tmp_path: Path,
    monkeypatch,
) -> None:
    chronovisor_root = tmp_path / "wiki"
    distillation = chronovisor_root / "runtime" / "recall-distillation"
    distillation.mkdir(parents=True)
    monkeypatch.setattr(dashboard, "CHRONOVISOR_ROOT", chronovisor_root)
    monkeypatch.setattr(
        dashboard,
        "runtime_identity",
        lambda: {
            "commit_id": "a" * 40,
            "module_path": "/archive/a/chronovisor/runtime_config.py",
            "package_version": "0.1.1",
        },
    )

    before = dashboard._health_materialization_fingerprint([])
    for name in (
        "state.json",
        "active-policy.json",
        "candidate-policy.json",
        "lkg-policy.json",
    ):
        (distillation / name).write_text(json.dumps({"name": name}), encoding="utf-8")
    after = dashboard._health_materialization_fingerprint([])

    assert before != after


def test_dashboard_static_renders_compact_distillation_health_status() -> None:
    page = (dashboard.STATIC_DIR / "index.html").read_text(encoding="utf-8")
    app = (dashboard.STATIC_DIR / "app.js").read_text(encoding="utf-8")
    renderer = (dashboard.STATIC_DIR / "app-renderer.js").read_text(encoding="utf-8")

    assert 'id="health-recall-distillation"' in page
    assert 'id="health-recall-distillation-detail"' in page
    assert "healthRecallDistillation" in app
    assert "healthRecallDistillationDetail" in app
    assert "recall_distillation" in renderer
    assert "rollout_status" in renderer
    assert "active_policy_id" in renderer
    assert "candidate_policy_id" in renderer
    assert "lkg_policy_id" in renderer
    assert "teacher-only" in renderer
    assert "verified truth" in renderer
    assert "not truth" in renderer
    assert "paired" in renderer
    assert "feature_revision" in renderer
    assert "hold_reason" in renderer


def test_materialized_component_returns_stale_while_audit_refreshes(
    tmp_path: Path, monkeypatch
) -> None:
    chronovisor_root = tmp_path / "wiki"
    monkeypatch.setattr(dashboard, "CHRONOVISOR_ROOT", chronovisor_root)
    clock = [100.0]
    monkeypatch.setattr(dashboard.time, "time", lambda: clock[0])
    calls = 0

    def build() -> dict:
        nonlocal calls
        calls += 1
        return {"serial": calls}

    class ImmediateThread:
        def __init__(self, *, target, args=(), kwargs=None, **_options) -> None:
            self.target = target
            self.args = args
            self.kwargs = kwargs or {}

        def start(self) -> None:
            self.target(*self.args, **self.kwargs)

    monkeypatch.setattr(dashboard.threading, "Thread", ImmediateThread)
    assert dashboard._materialized_component(
        "audit-view",
        fingerprint="a" * 64,
        builder=build,
        audit_seconds=60,
    ) == {"serial": 1}

    clock[0] += 61
    stale = dashboard._materialized_component(
        "audit-view",
        fingerprint="a" * 64,
        builder=build,
        audit_seconds=60,
    )

    assert stale == {"serial": 1}
    assert dashboard._materialized_component(
        "audit-view",
        fingerprint="a" * 64,
        builder=build,
        audit_seconds=60,
    ) == {"serial": 2}
    assert calls == 2


def test_cached_snapshot_rebuilds_after_a_source_changes_during_build(
    tmp_path: Path, monkeypatch
) -> None:
    _reset_snapshot_fingerprint_cache()
    chronovisor_root = tmp_path / "wiki"
    runtime_dir = chronovisor_root / "runtime"
    runtime_dir.mkdir(parents=True)
    monkeypatch.setattr(dashboard, "CHRONOVISOR_ROOT", chronovisor_root)
    monkeypatch.setattr(
        dashboard,
        "ACTIVITY_FILE",
        chronovisor_root / "runtime" / "activity.jsonl",
    )
    monkeypatch.setattr(
        orchestrator, "STATE_FILE", chronovisor_root / ".orchestrator_state.json"
    )
    monkeypatch.setattr(runtime_status, "STATUS_FILE", runtime_dir / "status.json")
    monkeypatch.setattr(runtime_status, "EVENTS_FILE", runtime_dir / "events.jsonl")
    monkeypatch.setattr(runtime_status, "METRICS_FILE", runtime_dir / "metrics.jsonl")
    dashboard._SNAPSHOT_CACHE.update(
        {
            "built_at": 0.0,
            "fingerprint": None,
            "snapshot": None,
            "refreshing": False,
        }
    )
    calls = 0

    def fake_build() -> dict:
        nonlocal calls
        calls += 1
        if calls == 1:
            runtime_status.EVENTS_FILE.write_text("{}\n", encoding="utf-8")
        return {
            "serial": calls,
            "status": {"state": "idle", "batch": {"active": False}},
        }

    monkeypatch.setattr(dashboard, "build_snapshot", fake_build)

    assert dashboard._cached_snapshot()["serial"] == 1
    assert dashboard._SNAPSHOT_CACHE["fingerprint"] is None
    assert dashboard._cached_snapshot()["serial"] == 2
    assert dashboard._cached_snapshot()["serial"] == 2
    assert calls == 2


def test_cached_snapshot_ignores_consensus_live_churn_but_tracks_audit(
    tmp_path: Path, monkeypatch
) -> None:
    _reset_snapshot_fingerprint_cache()
    chronovisor_root = tmp_path / "wiki"
    runtime_dir = chronovisor_root / "runtime"
    consensus_dir = runtime_dir / "local-consensus"
    active_dir = consensus_dir / "active"
    active_dir.mkdir(parents=True)
    summary = consensus_dir / "summary.json"
    audit = consensus_dir / "audit.jsonl"
    trace = consensus_dir / "trace-events.jsonl"
    summary.write_text('{"revision": 1}\n', encoding="utf-8")
    audit.write_text('{"revision": 1}\n', encoding="utf-8")
    trace.write_text('{"revision": 1}\n', encoding="utf-8")
    monkeypatch.setattr(dashboard, "CHRONOVISOR_ROOT", chronovisor_root)
    monkeypatch.setattr(
        dashboard,
        "ACTIVITY_FILE",
        chronovisor_root / "runtime" / "activity.jsonl",
    )
    monkeypatch.setattr(
        orchestrator, "STATE_FILE", chronovisor_root / ".orchestrator_state.json"
    )
    monkeypatch.setattr(runtime_status, "STATUS_FILE", runtime_dir / "status.json")
    monkeypatch.setattr(runtime_status, "EVENTS_FILE", runtime_dir / "events.jsonl")
    monkeypatch.setattr(runtime_status, "METRICS_FILE", runtime_dir / "metrics.jsonl")
    dashboard._SNAPSHOT_CACHE.update(
        {
            "built_at": 0.0,
            "fingerprint": None,
            "snapshot": None,
            "refreshing": False,
        }
    )
    calls = 0

    def fake_build() -> dict:
        nonlocal calls
        calls += 1
        return {
            "serial": calls,
            "status": {"state": "idle", "batch": {"active": False}},
        }

    monkeypatch.setattr(dashboard, "build_snapshot", fake_build)

    assert dashboard._cached_snapshot()["serial"] == 1
    before = dashboard._build_snapshot_source_fingerprint()

    marker_temporary = active_dir / ".request.tmp"
    marker = active_dir / "request.json"
    marker_temporary.write_text('{"revision": 2}\n', encoding="utf-8")
    os.replace(marker_temporary, marker)
    summary_temporary = consensus_dir / ".summary.tmp"
    summary_temporary.write_text('{"revision": 2}\n', encoding="utf-8")
    os.replace(summary_temporary, summary)
    with trace.open("a", encoding="utf-8") as handle:
        handle.write('{"revision": 2}\n')

    dashboard._invalidate_snapshot_fingerprint_probe()
    after_create = dashboard._build_snapshot_source_fingerprint()
    assert after_create == before
    assert dashboard._cached_snapshot()["serial"] == 1

    with audit.open("a", encoding="utf-8") as handle:
        handle.write('{"revision": 2}\n')

    dashboard._invalidate_snapshot_fingerprint_probe()
    after_audit = dashboard._build_snapshot_source_fingerprint()
    assert after_audit != before
    assert dashboard._cached_snapshot()["serial"] == 2
    assert calls == 2


def test_build_snapshot_surfaces_frontier_human_required(
    tmp_path: Path, monkeypatch
) -> None:
    chronovisor_root = tmp_path / "wiki"
    raw_dir = chronovisor_root / "raw"
    runtime_dir = chronovisor_root / "runtime"
    raw_dir.mkdir(parents=True)
    (chronovisor_root / "pages").mkdir()
    (chronovisor_root / "system").mkdir()

    monkeypatch.setattr(dashboard, "CHRONOVISOR_ROOT", chronovisor_root)
    monkeypatch.setattr(dashboard, "init_chronovisor", lambda: None)
    monkeypatch.setattr(
        dashboard,
        "ACTIVITY_FILE",
        chronovisor_root / "runtime" / "activity.jsonl",
    )
    monkeypatch.setattr(runtime_status, "RUNTIME_DIR", runtime_dir)
    monkeypatch.setattr(runtime_status, "STATUS_FILE", runtime_dir / "status.json")
    monkeypatch.setattr(runtime_status, "EVENTS_FILE", runtime_dir / "events.jsonl")
    monkeypatch.setattr(runtime_status, "METRICS_FILE", runtime_dir / "metrics.jsonl")

    from chronovisor.core import store
    from chronovisor.ingest import orchestrator

    monkeypatch.setattr(store, "CHRONOVISOR_ROOT", chronovisor_root)
    monkeypatch.setattr(store, "RAW_DIR", raw_dir)
    monkeypatch.setattr(store, "PAGES_DIR", chronovisor_root / "pages")
    monkeypatch.setattr(store, "SYSTEM_DIR", chronovisor_root / "system")
    monkeypatch.setattr(store, "INDEX_FILE", chronovisor_root / "pages" / "index.md")
    monkeypatch.setattr(store, "LOG_FILE", chronovisor_root / "pages" / "log.md")
    monkeypatch.setattr(
        store, "ACTIVITY_FILE", runtime_dir / "activity.jsonl"
    )
    monkeypatch.setattr(orchestrator, "RAW_DIR", raw_dir)
    monkeypatch.setattr(
        orchestrator, "STATE_FILE", chronovisor_root / ".orchestrator_state.json"
    )
    monkeypatch.setattr(
        dashboard, "_ollama_snapshot", lambda: {"available": False, "models": []}
    )
    monkeypatch.setattr(
        dashboard,
        "_model_status_snapshot",
        lambda ollama=None: {
            "available": False,
            "models": [],
            "summary": {"installed": 0, "loaded": 0},
        },
    )
    monkeypatch.setattr(
        dashboard,
        "_frontier_preflight_snapshot",
        lambda: {"ok": False, "checked_at": "2026-06-04T22:00:00"},
    )

    runtime_status.write_status({"state": "idle"})
    failures_dir = runtime_dir / "failures"
    packets_dir = failures_dir / "packets"
    packets_dir.mkdir(parents=True)
    packet = {
        "failure_id": "auth1",
        "created_at": "2026-06-04T22:00:00",
        "updated_at": "2026-06-04T22:01:00",
        "raw_file": "auth-broken.md",
        "failure_class": "triage.parse_failed",
        "fingerprint": "triage.parse_failed",
        "status": "human_required",
        "frontier_result": {
            "decision": "needs_retry",
            "summary": "codex auth missing",
            "rescue_status": "human_required",
            "human_required": True,
            "frontier_failure": {"failure_class": "auth_required"},
            "access_repair": {
                "applied": True,
                "repairs": [{"type": "cli_option_adapted"}],
            },
        },
        "human_notification": {
            "body": "Codex の認証が切れている可能性があります。ログイン確認が必要です。",
            "delivery": {"sent": True},
        },
        "pending_frontier_review_path": str(
            failures_dir / "pending-frontier-review" / "auth1.json"
        ),
    }
    (packets_dir / "auth1.json").write_text(json.dumps(packet), encoding="utf-8")
    registry_record = {
        "timestamp": "2026-06-04T22:02:00",
        "failure_id": "auth1",
        "raw_file": "auth-broken.md",
        "failure_class": "triage.parse_failed",
        "fingerprint": "triage.parse_failed",
        "resolution": "frontier",
        "decision": {"status": "escalate", "action": "escalate_to_frontier"},
        "frontier": packet["frontier_result"],
        "human_notification": packet["human_notification"],
        "pending_frontier_review_path": packet["pending_frontier_review_path"],
    }
    (failures_dir / "failure-registry.jsonl").write_text(
        json.dumps(registry_record, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    snapshot = dashboard.build_snapshot()

    self_heal = snapshot["self_heal"]
    assert self_heal["status"] == "failed"
    assert self_heal["counts"]["human_required"] == 1
    assert self_heal["counts"]["failed"] == 1
    assert self_heal["latest"]["title"] == "Human required"
    assert self_heal["latest"]["details"]["frontier"]["human_required"] is True
    assert (
        self_heal["latest"]["details"]["frontier"]["access_repair"]["applied"] is True
    )
    assert (
        self_heal["latest"]["details"]["human_notification"]["delivery"]["sent"] is True
    )
    assert (
        self_heal["latest"]["details"]["pending_frontier_review_path"]
        == packet["pending_frontier_review_path"]
    )
    assert self_heal["watch"]["packets"]["failed"] == 1
    assert self_heal["watch"]["frontier_preflight"]["ok"] is False


def test_runtime_failure_snapshot_is_safe_exact_deterministic_and_bounded(
    monkeypatch,
) -> None:
    canary = "W7 CANARY secret?credential"
    configured = SimpleNamespace(
        providers={
            "fake-config": SimpleNamespace(
                kind="openai", profile=SimpleNamespace(profile_id="fake")
            )
        },
        roles={
            "review": SimpleNamespace(
                provider_id="fake-config",
                model="writer-a",
                capability="generation",
            )
        }
    )
    monkeypatch.setattr(dashboard.llm_config, "load_llm_config", lambda: configured)
    events = [
        {"kind": "other", "error": canary},
        {
            "kind": "runtime_failure",
            "category": "http_401",
            "role": "review",
            "capability": "generation",
            "provider": "fake",
            "location": "remote",
            "retry_count": 0,
            "timestamp": "2026-08-10T10:00:00+09:00",
            "configured_model": canary,
            "request_id": canary,
            "error": canary,
            "prompt": canary,
            "content": canary,
            "credential": canary,
        },
        {
            "kind": "runtime_failure",
            "category": "http_429",
            "role": "review",
            "capability": "generation",
            "provider": "fake",
            "location": "remote",
            "retry_count": 2,
            "timestamp": "2026-08-10T10:01:00+09:00",
            "request_id": "req_safe_2",
        },
        {
            "kind": "runtime_failure",
            "category": "timeout",
            "role": "review",
            "capability": "generation",
            "provider": "fake",
            "location": "remote",
            "retry_count": 1,
            "timestamp": "2026-08-10T10:02:00+09:00",
            "configured_model": "writer-b",
        },
        {
            "kind": "runtime_failure",
            "category": "backend_error",
            "role": "invalid.timestamp",
            "capability": "generation",
            "provider": "fake",
            "location": "local",
            "retry_count": 0,
            "timestamp": "not-a-timestamp",
        },
        {
            "kind": "runtime_failure",
            "category": "backend_error",
            "role": "naive.timestamp",
            "capability": "generation",
            "provider": "fake",
            "location": "local",
            "retry_count": 0,
            "timestamp": "2026-08-10T20:00:00",
        },
        *[
            {
                "kind": "runtime_failure",
                "category": "backend_error",
                "role": f"role.{index:03d}",
                "capability": "generation",
                "provider": "fake",
                "location": "local",
                "retry_count": 0,
            }
            for index in range(70)
        ],
    ]

    snapshot = dashboard._runtime_failure_snapshot(events, decision_rows=[])
    serialized = json.dumps(snapshot, ensure_ascii=False)

    assert len(snapshot["runtime_failures"]) == 64
    assert snapshot["runtime_failures"][0]["role"] == "review"
    assert snapshot["last_failure"]["role"] == "review"
    assert snapshot["last_failure"]["category"] == "timeout"
    assert canary not in serialized

    focused = dashboard._runtime_failure_snapshot(events[:4], decision_rows=[])
    assert focused["runtime_failures"] == [focused["last_failure"]]
    assert focused["last_failure"]["configured_model"] == "writer-a"
    assert focused["last_failure"]["category"] == "timeout"
    assert (
        dashboard._runtime_failure_snapshot(events[:3], decision_rows=[])[
            "last_failure"
        ]["request_id"]
        == "req_safe_2"
    )


def test_runtime_failure_projects_invalid_votes_safely_by_timestamp_and_bounds(
    tmp_path: Path,
    monkeypatch,
) -> None:
    chronovisor_root = tmp_path / "wiki"
    audit_file = chronovisor_root / "runtime" / "local-consensus" / "audit.jsonl"
    audit_file.parent.mkdir(parents=True)
    monkeypatch.setattr(dashboard, "CHRONOVISOR_ROOT", chronovisor_root)
    roles = {
        role: SimpleNamespace(
            provider_id="configured-provider",
            model=f"configured-{role.rsplit('.', 1)[-1]}",
            capability="generation",
        )
        for role in (
            "classification.primary",
            "classification.challenger",
            "classification.tie_break",
            "runtime.role",
        )
    }
    monkeypatch.setattr(
        dashboard.llm_config,
        "load_llm_config",
        lambda: SimpleNamespace(
            providers={
                "configured-provider": SimpleNamespace(
                    kind="openai",
                    profile=SimpleNamespace(profile_id="fake"),
                )
            },
            roles=roles,
        ),
    )
    canary = "W7-CANARY?credential=secret prompt session"

    def decision(
        timestamp: str,
        role: str,
        *,
        provider: str = "fake",
        reason: str = "returned_model_mismatch",
    ) -> dict[str, object]:
        return {
            "kind": "decision",
            "timestamp": timestamp,
            "prompt": canary,
            "error": canary,
            "content": canary,
            "votes": [
                {
                    "valid": False,
                    "role": role.rsplit(".", 1)[-1],
                    "invalid_reason": reason,
                    "model": canary,
                    "returned_model": canary,
                    "credential": canary,
                    "session": {"error": canary, "request_id": canary},
                    "route_provenance": {
                        "role": role,
                        "provider": provider,
                        "location": "remote",
                        "model": canary,
                        "revision": canary,
                    },
                }
            ],
        }

    audit_file.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                decision(
                    "2026-08-10T10:05:00+09:00",
                    "classification.challenger",
                    reason="egress_denied",
                ),
                decision(
                    "2026-08-10T10:00:00+09:00",
                    "classification.primary",
                ),
                decision(
                    "2026-08-10T10:02:00+09:00",
                    "classification.tie_break",
                    provider="drifted-provider",
                ),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    snapshot = dashboard._runtime_failure_snapshot(
        [
            {
                "kind": "runtime_failure",
                "category": "http_429",
                "role": "runtime.role",
                "capability": "generation",
                "provider": "fake",
                "location": "remote",
                "retry_count": 1,
                "timestamp": "2026-08-10T10:03:00+09:00",
                "request_id": "req-safe",
            }
        ]
    )
    failures = snapshot["runtime_failures"]
    by_role = {row["role"]: row for row in failures}

    assert [row["role"] for row in failures] == [
        "classification.challenger",
        "runtime.role",
        "classification.tie_break",
        "classification.primary",
    ]
    assert snapshot["last_failure"] == failures[0]
    assert by_role["classification.challenger"]["category"] == "egress_denied"
    assert by_role["classification.primary"]["category"] == "vote_invalid"
    assert by_role["classification.primary"]["configured_model"] == (
        "configured-primary"
    )
    assert "configured_model" not in by_role["classification.tie_break"]
    for role in (
        "classification.primary",
        "classification.challenger",
        "classification.tie_break",
    ):
        assert by_role[role]["capability"] == "generation"
        assert by_role[role]["retry_count"] == 0
        assert "request_id" not in by_role[role]
    assert canary not in json.dumps(snapshot, ensure_ascii=False)

    valid = decision(
        "2026-08-10T11:00:00+09:00",
        "classification.primary",
    )
    valid["votes"][0]["valid"] = True
    unknown = decision(
        "2026-08-10T11:01:00+09:00",
        "classification.unknown",
    )
    mismatched = decision(
        "2026-08-10T11:02:00+09:00",
        "classification.primary",
    )
    mismatched["votes"][0]["role"] = "challenger"
    assert dashboard._decision_vote_failure_events([valid, unknown, mismatched]) == []


@pytest.mark.parametrize(
    ("kind", "profile_id", "event_provider", "capability"),
    [
        ("ollama", None, "ollama", "generation"),
        ("openai", "remote-profile", "remote-profile", "generation"),
        ("local-transformers", None, "local-reranker", "rerank"),
    ],
)
def test_runtime_failure_joins_exact_configured_backend_provider(
    monkeypatch,
    kind: str,
    profile_id: str | None,
    event_provider: str,
    capability: str,
) -> None:
    provider = SimpleNamespace(
        kind=kind,
        profile=(SimpleNamespace(profile_id=profile_id) if profile_id else None),
    )
    config = SimpleNamespace(
        providers={"configured-provider": provider},
        roles={
            "exact.role": SimpleNamespace(
                provider_id="configured-provider",
                model="configured-model",
                capability=capability,
            )
        },
    )
    monkeypatch.setattr(dashboard.llm_config, "load_llm_config", lambda: config)
    base = {
        "kind": "runtime_failure",
        "category": "backend_error",
        "role": "exact.role",
        "capability": capability,
        "location": "local" if profile_id is None else "remote",
        "retry_count": 0,
    }

    joined = dashboard._runtime_failure_snapshot(
        [{**base, "provider": event_provider}], decision_rows=[]
    )["last_failure"]
    drifted = dashboard._runtime_failure_snapshot(
        [{**base, "provider": "other-provider"}], decision_rows=[]
    )["last_failure"]
    wrong_location = dashboard._runtime_failure_snapshot(
        [
            {
                **base,
                "provider": event_provider,
                "location": "remote" if base["location"] == "local" else "local",
            }
        ],
        decision_rows=[],
    )["last_failure"]

    assert joined["configured_model"] == "configured-model"
    assert "configured_model" not in drifted
    assert "configured_model" not in wrong_location


def test_runtime_failure_invalidates_model_cache_and_overlays_stale_snapshot(
    tmp_path: Path, monkeypatch
) -> None:
    chronovisor_root = tmp_path / "wiki"
    runtime_dir = chronovisor_root / "runtime"
    runtime_dir.mkdir(parents=True)
    monkeypatch.setattr(dashboard, "CHRONOVISOR_ROOT", chronovisor_root)
    monkeypatch.setattr(
        dashboard,
        "ACTIVITY_FILE",
        chronovisor_root / "runtime" / "activity.jsonl",
    )
    monkeypatch.setattr(runtime_status, "RUNTIME_DIR", runtime_dir)
    monkeypatch.setattr(runtime_status, "STATUS_FILE", runtime_dir / "status.json")
    monkeypatch.setattr(runtime_status, "EVENTS_FILE", runtime_dir / "events.jsonl")
    monkeypatch.setattr(runtime_status, "METRICS_FILE", runtime_dir / "metrics.jsonl")
    monkeypatch.setattr(
        dashboard.llm_config,
        "load_llm_config",
        lambda: SimpleNamespace(
            providers={
                "fake-config": SimpleNamespace(
                    kind="openai", profile=SimpleNamespace(profile_id="fake")
                )
            },
            roles={
                "review": SimpleNamespace(
                    provider_id="fake-config",
                    model="writer",
                    capability="generation",
                )
            }
        ),
    )
    monkeypatch.setattr(orchestrator, "_load_state", lambda: {})
    monkeypatch.setattr(
        dashboard,
        "_canonicalize_runtime_status",
        lambda status, _state, *, pending: status,
    )
    runtime_status.write_status({"state": "idle"})
    model_before = dashboard._model_status_materialization_fingerprint({"models": []})
    snapshot_before = dashboard._build_snapshot_source_fingerprint()

    runtime_status.append_runtime_failure(
        dashboard.llm_config.RuntimeFailureTelemetry(
            "http_429",
            "review",
            "generation",
            "fake",
            "remote",
            retry_count=2,
            request_id="req_safe_3",
        )
    )

    model_after = dashboard._model_status_materialization_fingerprint({"models": []})
    snapshot_after = dashboard._build_snapshot_source_fingerprint()
    overlaid = dashboard._snapshot_with_live_status(
        {
            "status": {"state": "idle"},
            "model_status": {"models": []},
            "runtime_failures": [],
            "last_failure": None,
        }
    )

    assert model_after != model_before
    assert snapshot_after != snapshot_before
    assert overlaid["last_failure"]["category"] == "http_429"
    assert overlaid["last_failure"]["configured_model"] == "writer"
    assert overlaid["runtime_failures"] == [overlaid["last_failure"]]


def test_decision_audit_invalidates_model_and_snapshot_fingerprints(
    tmp_path: Path,
    monkeypatch,
) -> None:
    chronovisor_root = tmp_path / "wiki"
    runtime_dir = chronovisor_root / "runtime"
    audit_file = runtime_dir / "local-consensus" / "audit.jsonl"
    audit_file.parent.mkdir(parents=True)
    monkeypatch.setattr(dashboard, "CHRONOVISOR_ROOT", chronovisor_root)
    monkeypatch.setattr(
        dashboard,
        "ACTIVITY_FILE",
        chronovisor_root / "runtime" / "activity.jsonl",
    )
    monkeypatch.setattr(
        orchestrator, "STATE_FILE", chronovisor_root / ".orchestrator_state.json"
    )
    monkeypatch.setattr(runtime_status, "STATUS_FILE", runtime_dir / "status.json")
    monkeypatch.setattr(runtime_status, "EVENTS_FILE", runtime_dir / "events.jsonl")
    monkeypatch.setattr(runtime_status, "METRICS_FILE", runtime_dir / "metrics.jsonl")

    model_before = dashboard._model_status_materialization_fingerprint({"models": []})
    snapshot_before = dashboard._build_snapshot_source_fingerprint()
    audit_file.write_text('{"kind":"decision"}\n', encoding="utf-8")

    assert (
        dashboard._model_status_materialization_fingerprint({"models": []})
        != model_before
    )
    assert dashboard._build_snapshot_source_fingerprint() != snapshot_before


def test_model_fleet_runtime_failure_renderer_uses_text_only() -> None:
    page = (dashboard.STATIC_DIR / "index.html").read_text(encoding="utf-8")
    app = (dashboard.STATIC_DIR / "app.js").read_text(encoding="utf-8")
    renderer = (dashboard.STATIC_DIR / "app-renderer.js").read_text(encoding="utf-8")
    source = Path(dashboard.__file__).read_text(encoding="utf-8")
    block = renderer.split("function renderRuntimeFailures(failures)", 1)[1].split(
        "function renderModelStatus", 1
    )[0]
    route = source.split('elif path == "/api/model-status":', 1)[1].split(
        'elif path.startswith("/static/"):', 1
    )[0]

    assert 'id="model-failure-feed"' in page
    assert "modelFailureFeed: document.getElementById" in app
    assert "textContent" in block
    assert "innerHTML" not in block
    assert "configured_model" in block
    assert "request_id" in block
    assert "snapshot.runtime_failures" in renderer
    assert "failures = _runtime_failure_snapshot(" in route
    assert "**failures" in route
    assert "ollama = _ollama_snapshot()" in route
    assert '"model_status": _model_status_snapshot(ollama)' in route
    assert "_cached_snapshot" not in route
    assert "W7 CANARY" not in page + app + renderer


def test_model_status_snapshot_combines_ollama_and_config(monkeypatch) -> None:
    monkeypatch.setattr(
        dashboard,
        "load_search_embedding_config",
        lambda: SearchEmbeddingConfig(enabled=False),
    )
    monkeypatch.setattr(dashboard, "ingest_model", lambda: "qwen3.6:35b-a3b-mxfp8")
    monkeypatch.setattr(
        dashboard,
        "_ollama_snapshot",
        lambda: {
            "available": True,
            "models": [
                {
                    "name": "qwen3.6:35b-a3b-mxfp8",
                    "model": "qwen3.6:35b-a3b-mxfp8",
                    "size": 39_000,
                    "size_vram": 39_000,
                    "context_length": 32768,
                    "expires_at": "2026-07-05T20:11:28+09:00",
                    "details": {"format": "safetensors", "quantization_level": "mxfp8"},
                },
                {
                    "name": "bge-m3:latest",
                    "model": "bge-m3:latest",
                    "size": 664,
                    "size_vram": 664,
                    "context_length": 8192,
                    "details": {"format": "gguf", "quantization_level": "F16"},
                },
            ],
        },
    )
    monkeypatch.setattr(
        dashboard,
        "_ollama_tags_snapshot",
        lambda: {
            "available": True,
            "models": [
                {
                    "name": "qwen3.6:35b-a3b-mxfp8",
                    "model": "qwen3.6:35b-a3b-mxfp8",
                    "size": 37_000,
                    "modified_at": "2026-07-05T18:04:01+09:00",
                    "details": {"format": "safetensors", "quantization_level": "mxfp8"},
                    "capabilities": ["completion", "tools"],
                },
                {
                    "name": "gemma4:26b-mxfp8",
                    "model": "gemma4:26b-mxfp8",
                    "size": 27_000,
                    "details": {"format": "safetensors", "quantization_level": "mxfp8"},
                    "capabilities": ["completion"],
                },
                {
                    "name": "bge-m3:latest",
                    "model": "bge-m3:latest",
                    "size": 1_157,
                    "details": {"format": "gguf", "quantization_level": "F16"},
                    "capabilities": ["embedding"],
                },
                {
                    "name": "muse-glimmer:30b-nvfp4-dflash",
                    "model": "muse-glimmer:30b-nvfp4-dflash",
                    "size": 13_000,
                    "details": {"format": "safetensors", "quantization_level": "NVFP4"},
                    "capabilities": ["completion"],
                },
            ],
        },
    )
    monkeypatch.setattr(
        dashboard,
        "load_audit_policy",
        lambda: SimpleNamespace(enabled=True),
    )
    role_models = {
        dashboard.AUDITOR_RUNTIME_ROLE: "qwen3.6:35b-a3b-mxfp8",
        dashboard.RECALL_GATE_RUNTIME_ROLE: "qwen3.5:4b-mlx",
        dashboard.RECALL_QUERY_REWRITER_RUNTIME_ROLE: "qwen3.5:4b-mlx",
        dashboard.PROPOSER_RUNTIME_ROLES[0]: "qwen3.6:35b-a3b-mxfp8",
        dashboard.PROPOSER_RUNTIME_ROLES[1]: "gemma4:26b-mxfp8",
        dashboard.DECISION_RUNTIME_ROLES[0]: "qwen3.6:35b-a3b-mxfp8",
        dashboard.DECISION_RUNTIME_ROLES[1]: "muse-glimmer:30b-nvfp4-dflash",
        dashboard.DECISION_RUNTIME_ROLES[2]: "gemma4:26b-mxfp8",
    }
    monkeypatch.setattr(
        dashboard,
        "runtime_generation_routes",
        lambda roles: tuple(
            ollama.RuntimeGenerationRoute(
                role=role,
                provider="ollama",
                model=role_models[role],
                location="local",
                structured_output=True,
            )
            for role in roles
        ),
    )
    monkeypatch.setattr(
        dashboard.recall_runtime,
        "load_policy",
        lambda: SimpleNamespace(
            judge_mode="auto",
            rewrite_enabled=True,
        ),
    )
    monkeypatch.setattr(
        dashboard.llm_config,
        "load_default_llm_runtime",
        lambda: SimpleNamespace(
            resolve_embedding=lambda role: SimpleNamespace(
                role=role,
                provider="ollama",
                model="bge-m3",
                location=SimpleNamespace(value="local"),
            )
        ),
    )
    monkeypatch.setattr(
        dashboard,
        "load_reranker_config",
        lambda: SimpleNamespace(enabled=False, model="BAAI/bge-reranker-v2-m3"),
    )

    snapshot = dashboard._model_status_snapshot()
    by_name = {row["name"]: row for row in snapshot["models"]}

    assert snapshot["summary"]["installed"] == 4
    assert snapshot["summary"]["all_installed"] == 4
    assert snapshot["summary"]["unused_installed"] == 0
    assert snapshot["summary"]["loaded"] == 2
    assert snapshot["summary"]["configured"] == 5
    assert by_name["qwen3.6:35b-a3b-mxfp8"]["status"] == "loaded"
    assert by_name["qwen3.6:35b-a3b-mxfp8"]["roles"] == [
        "ingest",
        "audit",
        "improve",
        "decision-primary",
    ]
    assert by_name["gemma4:26b-mxfp8"]["status"] == "ready"
    assert by_name["gemma4:26b-mxfp8"]["roles"] == [
        "improve",
        "decision-tie-break",
    ]
    assert by_name["bge-m3:latest"]["roles"] == ["embed"]
    assert "bge-m3" not in by_name
    assert by_name["muse-glimmer:30b-nvfp4-dflash"]["status"] == "ready"
    assert by_name["muse-glimmer:30b-nvfp4-dflash"]["roles"] == [
        "decision-challenger"
    ]
    assert by_name["qwen3.5:4b-mlx"]["status"] == "missing"
    assert by_name["qwen3.5:4b-mlx"]["roles"] == ["gate", "rewrite"]


def test_configured_model_roles_use_runtime_router_triplet(monkeypatch) -> None:
    monkeypatch.setattr(
        dashboard,
        "load_search_embedding_config",
        lambda: SearchEmbeddingConfig(enabled=False),
    )
    monkeypatch.setattr(dashboard, "ingest_model", lambda: "")
    monkeypatch.setattr(
        dashboard, "load_audit_policy", lambda: SimpleNamespace(enabled=False)
    )
    monkeypatch.setattr(
        dashboard.recall_runtime,
        "load_policy",
        lambda: SimpleNamespace(judge_mode="off", rewrite_enabled=False),
    )

    decision_models = ("route-primary", "route-challenger", "route-tie")

    def configured_routes(roles):
        if tuple(roles) == dashboard.PROPOSER_RUNTIME_ROLES:
            raise ollama.RuntimeBridgeError("capability_unavailable")
        assert tuple(roles) == dashboard.DECISION_RUNTIME_ROLES
        return tuple(
            ollama.RuntimeGenerationRoute(
                role=role,
                provider="ollama",
                model=model,
                location="local",
                structured_output=True,
            )
            for role, model in zip(roles, decision_models, strict=True)
        )

    monkeypatch.setattr(dashboard, "runtime_generation_routes", configured_routes)
    monkeypatch.setattr(
        ollama,
        "embedding_model",
        lambda: (_ for _ in ()).throw(AssertionError("legacy selector used")),
    )
    resolved: list[str] = []
    monkeypatch.setattr(
        dashboard.llm_config,
        "load_default_llm_runtime",
        lambda: SimpleNamespace(
            resolve_embedding=lambda role: (
                resolved.append(role)
                or SimpleNamespace(
                    provider="ollama",
                    model="route-selected-knowledge",
                    location=SimpleNamespace(value="local"),
                )
            )
        ),
    )
    monkeypatch.setattr(
        dashboard,
        "load_reranker_config",
        lambda: SimpleNamespace(enabled=False),
    )

    roles = dashboard._configured_model_roles()

    assert set(roles) == {
        "route-primary",
        "route-challenger",
        "route-tie",
        "route-selected-knowledge",
    }
    assert roles["route-selected-knowledge"] == {"embed"}
    assert resolved == ["knowledge.embedding"]

    resolved.clear()
    monkeypatch.setattr(
        dashboard,
        "load_search_embedding_config",
        lambda: SearchEmbeddingConfig(enabled=True),
    )
    monkeypatch.setattr(
        dashboard.llm_config,
        "load_default_llm_runtime",
        lambda: SimpleNamespace(
            resolve_embedding=lambda role: (
                resolved.append(role)
                or SimpleNamespace(
                    provider="ollama",
                    model=(
                        "route-selected-knowledge"
                        if role == "knowledge.embedding"
                        else "route-selected-embedding"
                    ),
                    location=SimpleNamespace(value="local"),
                )
            )
        ),
    )

    roles = dashboard._configured_model_roles()

    assert resolved == [
        "knowledge.embedding",
        "search.semantic.foreground",
        "search.semantic.incremental",
    ]
    assert roles["route-selected-embedding"] == {"search-embed"}

    monkeypatch.setattr(
        dashboard,
        "load_reranker_config",
        lambda: SimpleNamespace(enabled=True, model="legacy-selector"),
    )
    monkeypatch.setattr(
        dashboard.llm_config,
        "load_default_llm_runtime",
        lambda: SimpleNamespace(
            resolve_embedding=lambda role: (
                resolved.append(role)
                or SimpleNamespace(
                    provider="ollama",
                    model=(
                        "route-selected-knowledge"
                        if role == "knowledge.embedding"
                        else "route-selected-embedding"
                    ),
                    location=SimpleNamespace(value="local"),
                )
            ),
            resolve_rerank=lambda role: (
                resolved.append(role)
                or SimpleNamespace(
                    provider="ollama",
                    model="route-selected-reranker",
                    location=SimpleNamespace(value="local"),
                )
            ),
        ),
    )

    roles = dashboard._configured_model_roles()

    assert resolved[-1] == "search.rerank"
    assert roles["route-selected-reranker"] == {"rerank"}
    assert "legacy-selector" not in roles


def test_dashboard_omits_remote_and_non_ollama_generation_routes(
    monkeypatch,
) -> None:
    monkeypatch.setattr(dashboard, "ingest_model", lambda: "")
    monkeypatch.setattr(
        dashboard, "load_audit_policy", lambda: SimpleNamespace(enabled=False)
    )
    monkeypatch.setattr(
        dashboard.recall_runtime,
        "load_policy",
        lambda: SimpleNamespace(judge_mode="off", rewrite_enabled=False),
    )
    monkeypatch.setattr(
        dashboard,
        "runtime_generation_routes",
        lambda roles: tuple(
            ollama.RuntimeGenerationRoute(
                role,
                "openai" if index == 0 else "native-local",
                f"hidden-{index}",
                "remote" if index == 0 else "local",
                True,
            )
            for index, role in enumerate(roles)
        ),
    )
    monkeypatch.setattr(
        dashboard,
        "load_search_embedding_config",
        lambda: SearchEmbeddingConfig(enabled=True),
    )
    remote_route = SimpleNamespace(
        provider="openai",
        model="hidden-remote",
        location=SimpleNamespace(value="remote"),
    )
    monkeypatch.setattr(
        dashboard.llm_config,
        "load_default_llm_runtime",
        lambda: SimpleNamespace(
            resolve_embedding=lambda _role: remote_route,
            resolve_rerank=lambda _role: remote_route,
        ),
    )
    monkeypatch.setattr(
        dashboard, "load_reranker_config", lambda: SimpleNamespace(enabled=True)
    )

    assert dashboard._configured_model_roles() == {}


def test_decision_trace_models_use_ordered_runtime_routes(monkeypatch) -> None:
    calls: list[tuple[str, ...]] = []

    def routes(roles):
        calls.append(tuple(roles))
        return tuple(
            ollama.RuntimeGenerationRoute(
                role=role,
                provider="openai",
                model=f"remote-{index}",
                location="remote",
                structured_output=True,
                protocol="openai-compatible",
                endpoint_sha256="e" * 64,
                revision="2026-08",
            )
            for index, role in enumerate(roles)
        )

    monkeypatch.setattr(dashboard, "runtime_generation_routes", routes)

    assert dashboard._decision_trace_models() == {
        "primary": "remote-0",
        "challenger": "remote-1",
        "tie_break": "remote-2",
    }
    assert calls == [dashboard.DECISION_RUNTIME_ROLES]


def test_dashboard_has_no_legacy_decision_selector_path() -> None:
    source = Path(dashboard.__file__).read_text(encoding="utf-8")

    assert "resolve_router_policy" not in source
    assert "_resolved_decision_router_config" not in source
    assert "adoption_artifact" not in source


def test_self_heal_snapshot_surfaces_watch_status(tmp_path: Path, monkeypatch) -> None:
    chronovisor_root = tmp_path / "wiki"
    failures_dir = chronovisor_root / "runtime" / "failures"
    packets_dir = failures_dir / "packets"
    logs_dir = chronovisor_root / "logs"
    packets_dir.mkdir(parents=True)
    logs_dir.mkdir(parents=True)

    monkeypatch.setattr(dashboard, "CHRONOVISOR_ROOT", chronovisor_root)
    monkeypatch.setattr(
        dashboard,
        "_frontier_preflight_snapshot",
        lambda: {
            "ok": True,
            "checked_at": "2026-07-04T19:30:00",
            "codex_version_ok": True,
            "exec_help_ok": True,
        },
    )

    pending_packet = {
        "failure_id": "queued1",
        "created_at": "2026-07-04T19:00:00",
        "raw_file": "queued.md",
        "failure_class": "triage.parse_failed",
        "status": "pending_frontier",
    }
    approved_packet = {
        "failure_id": "ok1",
        "created_at": "2026-07-04T18:00:00",
        "raw_file": "ok.md",
        "failure_class": "apply.update_target_not_found",
        "status": "frontier_approved",
    }
    (packets_dir / "queued1.json").write_text(
        json.dumps(pending_packet), encoding="utf-8"
    )
    (packets_dir / "ok1.json").write_text(json.dumps(approved_packet), encoding="utf-8")
    (logs_dir / "ingest-drain-20260704.jsonl").write_text(
        json.dumps(
            {
                "timestamp": "2026-07-04T19:31:00",
                "self_heal": {
                    "status": "ok",
                    "packets_seen": 1,
                    "results": [{"status": "pending_frontier"}],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    self_heal = dashboard._self_heal_snapshot()

    watch = self_heal["watch"]
    assert watch["last_checked"]["timestamp"] == "2026-07-04T19:31:00"
    assert watch["last_checked"]["packets_seen"] == 1
    assert watch["last_checked"]["results"] == 1
    assert watch["packets"]["total"] == 2
    assert watch["packets"]["pending"] == 1
    assert watch["packets"]["failed"] == 0
    assert watch["packets"]["status_counts"]["pending_frontier"] == 1
    assert watch["frontier_preflight"]["ok"] is True


def test_repair_deferred_packet_stays_pending_and_warns(
    tmp_path: Path,
    monkeypatch,
) -> None:
    chronovisor_root = tmp_path / "wiki"
    packets_dir = chronovisor_root / "runtime" / "failures" / "packets"
    packets_dir.mkdir(parents=True)
    packet = {
        "failure_id": "system-repair-1",
        "created_at": "2026-07-13T20:00:00",
        "updated_at": "2026-07-13T20:01:00",
        "failure_class": "system_health_snapshot_exception",
        "incident_kind": "system_code_repair",
        "status": "repair_deferred",
        "error": "frontier repair cooldown is active",
    }
    (packets_dir / "system-repair-1.json").write_text(
        json.dumps(packet),
        encoding="utf-8",
    )
    monkeypatch.setattr(dashboard, "CHRONOVISOR_ROOT", chronovisor_root)
    monkeypatch.setattr(
        dashboard,
        "_frontier_preflight_snapshot",
        lambda: {"ok": True},
    )

    watch = dashboard._self_heal_watch_snapshot({"system-repair-1": packet})
    summary = dashboard._packet_summary(packet)
    self_heal = dashboard._self_heal_snapshot()

    assert watch["packets"]["pending"] == 1
    assert watch["packets"]["failed"] == 0
    assert watch["packets"]["status_counts"] == {"repair_deferred": 1}
    assert watch["packets"]["pending_samples"][0]["status"] == "repair_deferred"
    assert summary["state"] == "pending"
    assert summary["level"] == "warn"
    assert summary["title"] == "Frontier repair deferred"
    assert self_heal["status"] == "pending"
    assert self_heal["counts"]["pending"] == 1
    assert self_heal["counts"]["resolved"] == 0
    assert self_heal["latest"]["state"] == "pending"
    assert self_heal["latest"]["level"] == "warn"
    assert self_heal["latest"]["title"] == "Frontier repair deferred"


def test_frontier_dashboard_snapshot_never_runs_codex_preflight(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from chronovisor.decision import frontier_review

    monkeypatch.setattr(dashboard, "CHRONOVISOR_ROOT", tmp_path / "wiki")
    monkeypatch.setattr(
        frontier_review,
        "run_frontier_preflight",
        lambda: (_ for _ in ()).throw(
            AssertionError("dashboard must not start a Codex preflight")
        ),
    )

    snapshot = dashboard._frontier_preflight_snapshot()

    assert snapshot["ok"] is True
    assert snapshot["mode"] == "on_demand_only"
    assert snapshot["state"] == "standby"
    assert snapshot["subprocess_checked"] is False


def test_recall_snapshot_reads_logs_and_eval(tmp_path: Path, monkeypatch) -> None:
    chronovisor_root = tmp_path / "wiki"
    recall_dir = chronovisor_root / "recall"
    eval_dir = chronovisor_root / "runtime" / "eval"
    recall_dir.mkdir(parents=True)
    eval_dir.mkdir(parents=True)

    monkeypatch.setattr(dashboard, "CHRONOVISOR_ROOT", chronovisor_root)

    records = [
        {
            "ts": "2026-06-11T10:00:00",
            "decision": "search",
            "latency_ms": 120,
            "used_judge": False,
            "evidence_features": {"rewrite_confidence": 0.8},
            "pages": ["a", "b"],
            "prompt_preview": "あれどうなった?",
            "host": "codex",
        },
        {
            "ts": "2026-06-11T10:01:00",
            "decision": "none",
            "latency_ms": 40,
            "used_judge": True,
            "evidence_features": {},
            "pages": [],
            "prompt_preview": "こんにちは",
            "host": "claude-code",
        },
    ]
    (recall_dir / "recall-log.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8"
    )
    (recall_dir / "pull-log.jsonl").write_text(
        json.dumps({"type": "read", "page_id": "a"}) + "\n", encoding="utf-8"
    )
    (recall_dir / "calibration-history.jsonl").write_text(
        json.dumps({"ts": "2026-06-10T03:00:00", "reason": "improved"}) + "\n",
        encoding="utf-8",
    )
    (eval_dir / "baseline-20260611.json").write_text(
        json.dumps(
            {
                "metrics": {
                    "examples": 816,
                    "recall_at_1": 0.71,
                    "recall_at_3": 0.83,
                    "waste_injection_rate": 0.33,
                    "latency_ms": {"p50": 95, "p95": 137},
                },
                "policy": {"gate_mode": "evidence"},
            }
        ),
        encoding="utf-8",
    )

    recall = dashboard._recall_snapshot()

    assert recall["samples"] == 2
    assert recall["decisions"] == {"none": 1, "search": 1, "read": 0}
    assert recall["judge_used"] == 1
    assert recall["rewrite_used"] == 1
    assert recall["latency_ms"]["p50"] == 40
    assert recall["pulls"]["total"] == 1
    assert recall["pulls"]["counts"]["read"] == 1
    assert recall["latest_eval"]["recall_at_3"] == 0.83
    assert recall["calibration"]["last_applied"]["reason"] == "improved"
    assert recall["recent"][-1]["decision"] == "none"


def test_recall_snapshot_empty_wiki(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(dashboard, "CHRONOVISOR_ROOT", tmp_path / "wiki")

    recall = dashboard._recall_snapshot()

    assert recall["samples"] == 0
    assert recall["decisions"] == {"none": 0, "search": 0, "read": 0}
    assert recall["latency_ms"]["p50"] is None
    assert recall["latest_eval"] is None
    assert recall["calibration"]["last_applied"] is None


def test_save_history_snapshot_combines_raw_drain_and_log(
    tmp_path: Path, monkeypatch
) -> None:
    chronovisor_root = tmp_path / "wiki"
    raw_dir = chronovisor_root / "raw"
    logs_dir = chronovisor_root / "logs"
    raw_dir.mkdir(parents=True)
    logs_dir.mkdir(parents=True)
    activity_file = chronovisor_root / "runtime" / "activity.jsonl"
    activity_file.parent.mkdir(parents=True)

    monkeypatch.setattr(dashboard, "CHRONOVISOR_ROOT", chronovisor_root)
    monkeypatch.setattr(dashboard, "ACTIVITY_FILE", activity_file)

    raw_names = [
        "20260701-120000-codex-dashboard-save-history-aaaaaaaa.md",
        "20260701-121000-claude-code-dashboard-save-history-bbbbbbbb.md",
        "vestige-20260702-event-cccccccc.md",
    ]
    for name in raw_names:
        (raw_dir / name).write_text("raw", encoding="utf-8")

    drain_record = {
        "timestamp": "2026-07-01T12:30:00",
        "files_processed": 1,
        "result": {
            "files_attempted": [raw_names[0], "20260701-122000-failed-dddddddd.md"],
            "files_processed": [raw_names[0]],
            "per_raw": [
                {"filename": raw_names[0], "succeeded": True},
                {"filename": "20260701-122000-failed-dddddddd.md", "succeeded": False},
            ],
        },
    }
    (logs_dir / "ingest-drain-20260701.jsonl").write_text(
        json.dumps(drain_record) + "\n",
        encoding="utf-8",
    )
    activity_file.write_bytes(
        _activity_bytes(
            (
                "2026-07-01T12:31:00+00:00",
                "ingest | created save-history-dashboard",
            ),
            (
                "2026-07-01T12:32:00+00:00",
                "ingest | updated chronovisor-dashboard",
            ),
            (
                "2026-07-02T12:32:00+00:00",
                "ingest | updated save-history-dashboard",
            ),
        )
    )

    history = dashboard._save_history_snapshot(days=4, today=date(2026, 7, 4))
    by_date = {row["date"]: row for row in history["days"]}

    assert history["totals"]["raw_saved"] == 3
    assert history["totals"]["raw_bytes"] == 9
    assert history["totals"]["processed_bytes"] == 3
    assert history["totals"]["pending_bytes"] == 6
    assert history["totals"]["failed_bytes"] == 0
    assert history["totals"]["processed"] == 1
    assert history["totals"]["attempted"] == 2
    assert history["totals"]["succeeded"] == 1
    assert history["totals"]["failed"] == 1
    assert history["totals"]["pages_created"] == 1
    assert history["totals"]["pages_updated"] == 2
    assert by_date["2026-07-01"]["raw_saved"] == 2
    assert by_date["2026-07-01"]["raw_bytes"] == 6
    assert by_date["2026-07-01"]["processed_bytes"] == 3
    assert by_date["2026-07-01"]["pending_bytes"] == 3
    assert by_date["2026-07-01"]["processed"] == 1
    assert by_date["2026-07-01"]["failed"] == 1
    assert by_date["2026-07-01"]["raw_segments"] == [
        {
            "name": raw_names[0],
            "bytes": 3,
            "status": "processed",
            "source": "codex",
        },
        {
            "name": raw_names[1],
            "bytes": 3,
            "status": "pending",
            "source": "claude-code",
        },
    ]
    assert by_date["2026-07-02"]["raw_saved"] == 1
    assert by_date["2026-07-02"]["pending_bytes"] == 3
    assert by_date["2026-07-02"]["raw_segments"][0]["status"] == "pending"
    assert by_date["2026-07-02"]["pages_updated"] == 1
    assert by_date["2026-07-01"]["sources"] == [
        {"name": "claude-code", "count": 1},
        {"name": "codex", "count": 1},
    ]
    assert history["sources"] == [
        {"name": "claude-code", "count": 1},
        {"name": "codex", "count": 1},
        {"name": "vestige", "count": 1},
    ]
    assert history["recent"][-1]["date"] == "2026-07-02"


def test_save_history_snapshot_reconciles_processed_orchestrator_state(
    tmp_path: Path,
    monkeypatch,
) -> None:
    chronovisor_root = tmp_path / "wiki"
    raw_dir = chronovisor_root / "raw"
    raw_dir.mkdir(parents=True)
    raw_name = "20260704-120000-codex-processed-without-drain-log-aaaaaaaa.md"
    failed_name = "20260704-121000-codex-explicit-failure-bbbbbbbb.md"
    (raw_dir / raw_name).write_text("raw", encoding="utf-8")
    (raw_dir / failed_name).write_text("bad", encoding="utf-8")
    (chronovisor_root / ".orchestrator_state.json").write_text(
        json.dumps({"processed_raw_files": [raw_name, failed_name]}),
        encoding="utf-8",
    )
    logs_dir = chronovisor_root / "logs"
    logs_dir.mkdir()
    (logs_dir / "ingest-drain-20260704.jsonl").write_text(
        json.dumps(
            {
                "timestamp": "2026-07-04T12:30:00",
                "result": {
                    "files_attempted": [failed_name],
                    "per_raw": [{"filename": failed_name, "succeeded": False}],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(dashboard, "CHRONOVISOR_ROOT", chronovisor_root)
    monkeypatch.setattr(
        dashboard,
        "ACTIVITY_FILE",
        chronovisor_root / "runtime" / "activity.jsonl",
    )

    history = dashboard._save_history_snapshot(days=1, today=date(2026, 7, 4))
    day = history["days"][0]

    assert history["totals"]["processed_bytes"] == 6
    assert history["totals"]["pending_bytes"] == 0
    assert history["totals"]["failed_bytes"] == 0
    assert day["raw_segments"] == [
        {
            "name": raw_name,
            "bytes": 3,
            "status": "processed",
            "source": "codex",
        },
        {
            "name": failed_name,
            "bytes": 3,
            "status": "processed",
            "source": "codex",
        },
    ]


def test_save_history_excludes_generated_semantic_projection_children(
    tmp_path: Path,
    monkeypatch,
) -> None:
    chronovisor_root = tmp_path / "wiki"
    raw_dir = chronovisor_root / "raw"
    raw_dir.mkdir(parents=True)
    parent_name = "20260704-120000-codex-parent-aaaaaaaa.md"
    child_name = f"semantic-{'a' * 64}-child-00000001-{'b' * 64}.md"
    (raw_dir / parent_name).write_text("parent", encoding="utf-8")
    (raw_dir / child_name).write_text("derived child", encoding="utf-8")

    monkeypatch.setattr(dashboard, "CHRONOVISOR_ROOT", chronovisor_root)
    monkeypatch.setattr(
        dashboard,
        "ACTIVITY_FILE",
        chronovisor_root / "runtime" / "activity.jsonl",
    )

    history = dashboard._save_history_snapshot(days=1, today=date(2026, 7, 4))

    assert dashboard._raw_source_label(child_name) == "projection"
    assert history["totals"]["raw_saved"] == 1
    assert history["totals"]["raw_bytes"] == len("parent")
    assert history["sources"] == [{"name": "codex", "count": 1}]
    assert [segment["name"] for segment in history["days"][0]["raw_segments"]] == [
        parent_name
    ]


def test_save_history_expands_fragment_group_status_and_processed_wins(
    tmp_path: Path,
    monkeypatch,
) -> None:
    chronovisor_root = tmp_path / "wiki"
    raw_dir = chronovisor_root / "raw"
    logs_dir = chronovisor_root / "logs"
    raw_dir.mkdir(parents=True)
    logs_dir.mkdir()
    fragment_names = [
        "20260704-120000-codex-fragment-one-aaaaaaaa.md",
        "20260704-120001-codex-fragment-two-bbbbbbbb.md",
    ]
    for name in fragment_names:
        (raw_dir / name).write_text("raw", encoding="utf-8")
    failed = {
        "timestamp": "2026-07-04T12:30:00",
        "result": {
            "per_raw": [
                {
                    "filename": fragment_names[0],
                    "source_files": fragment_names,
                    "succeeded": False,
                }
            ]
        },
    }
    drain_log = logs_dir / "ingest-drain-20260704.jsonl"
    drain_log.write_text(json.dumps(failed) + "\n", encoding="utf-8")

    monkeypatch.setattr(dashboard, "CHRONOVISOR_ROOT", chronovisor_root)
    monkeypatch.setattr(
        dashboard,
        "ACTIVITY_FILE",
        chronovisor_root / "runtime" / "activity.jsonl",
    )

    failed_history = dashboard._save_history_snapshot(days=1, today=date(2026, 7, 4))
    assert failed_history["totals"]["failed_bytes"] == 6
    assert failed_history["totals"]["pending_bytes"] == 0

    succeeded = {
        "timestamp": "2026-07-04T12:31:00",
        "result": {
            "per_raw": [
                {
                    "filename": fragment_names[0],
                    "source_files": fragment_names,
                    "succeeded": True,
                }
            ]
        },
    }
    drain_log.write_text(
        json.dumps(failed) + "\n" + json.dumps(succeeded) + "\n",
        encoding="utf-8",
    )
    processed_history = dashboard._save_history_snapshot(days=1, today=date(2026, 7, 4))
    assert processed_history["totals"]["processed_bytes"] == 6
    assert processed_history["totals"]["failed_bytes"] == 0
    assert processed_history["totals"]["pending_bytes"] == 0


def test_save_history_snapshot_empty_wiki(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(dashboard, "CHRONOVISOR_ROOT", tmp_path / "wiki")
    monkeypatch.setattr(
        dashboard,
        "ACTIVITY_FILE",
        tmp_path / "wiki" / "runtime" / "activity.jsonl",
    )

    history = dashboard._save_history_snapshot(days=2, today=date(2026, 7, 4))

    assert [row["date"] for row in history["days"]] == ["2026-07-03", "2026-07-04"]
    assert history["recent"] == []
    assert history["totals"]["raw_saved"] == 0
    assert history["totals"]["raw_bytes"] == 0
    assert history["totals"]["pending_bytes"] == 0
    assert history["days"][0]["raw_segments"] == []


def test_save_history_only_includes_segment_detail_for_recent_chart_window(
    tmp_path: Path, monkeypatch
) -> None:
    chronovisor_root = tmp_path / "wiki"
    raw_dir = chronovisor_root / "raw"
    raw_dir.mkdir(parents=True)
    old_name = "20260701-120000-codex-old-detail-aaaaaaaa.md"
    recent_name = "20260702-120000-codex-recent-detail-bbbbbbbb.md"
    (raw_dir / old_name).write_text("old", encoding="utf-8")
    (raw_dir / recent_name).write_text("recent", encoding="utf-8")

    monkeypatch.setattr(dashboard, "CHRONOVISOR_ROOT", chronovisor_root)
    monkeypatch.setattr(
        dashboard,
        "ACTIVITY_FILE",
        chronovisor_root / "runtime" / "activity.jsonl",
    )

    history = dashboard._save_history_snapshot(days=31, today=date(2026, 7, 31))
    by_date = {row["date"]: row for row in history["days"]}

    assert by_date["2026-07-01"]["raw_saved"] == 1
    assert by_date["2026-07-01"]["raw_bytes"] == 3
    assert by_date["2026-07-01"]["raw_segments"] == []
    assert by_date["2026-07-02"]["raw_segments"] == [
        {
            "name": recent_name,
            "bytes": 6,
            "status": "pending",
            "source": "codex",
        }
    ]


def test_save_history_compacts_large_days_without_losing_status_bytes() -> None:
    segments = [
        {
            "name": f"raw-{index:03d}.md",
            "bytes": index + 1,
            "status": "processed" if index % 2 == 0 else "pending",
            "source": "codex",
        }
        for index in range(dashboard.SAVE_HISTORY_MAX_SEGMENTS_PER_DAY + 1)
    ]

    compacted = dashboard._compact_raw_segments(segments)

    assert [segment["status"] for segment in compacted] == [
        "processed",
        "pending",
    ]
    assert sum(segment["bytes"] for segment in compacted) == sum(
        segment["bytes"] for segment in segments
    )
    assert compacted[0]["source"] == "aggregate"
    assert compacted[0]["name"].endswith("processed Raw saves")


def test_knowledge_mix_snapshot_groups_pages_by_category(
    tmp_path: Path, monkeypatch
) -> None:
    chronovisor_root = tmp_path / "wiki"
    pages_dir = chronovisor_root / "pages"
    (pages_dir / "ai").mkdir(parents=True)
    (pages_dir / "macos").mkdir()
    stable_texts = {
        pages_dir / "ai" / "agent-memory.md": (
            "---\ntitle: Agent memory\nstatus: stable\ntype: knowledge\n---\n" + "a" * 20
        ),
        pages_dir / "ai" / "evals.md": (
            "---\ntitle: Evals\nstatus: stable\ntype: knowledge\n---\n" + "b" * 10
        ),
        pages_dir / "macos" / "display.md": (
            "---\ntitle: Display\nstatus: stable\ntype: knowledge\n---\n" + "c" * 15
        ),
    }
    for path, content in stable_texts.items():
        path.write_text(content, encoding="utf-8")
    (pages_dir / "ai" / "draft.md").write_text(
        "---\ntitle: Draft\nstatus: draft\ntype: knowledge\n---\nexcluded",
        encoding="utf-8",
    )
    (pages_dir / "macos" / "invalid.md").write_text(
        "---\ntitle: Invalid\nstatus: stable\n---\nexcluded",
        encoding="utf-8",
    )
    for relative in (
        "index.md",
        "log.md",
        "schema.md",
        "ai/index.md",
        "ai/log.md",
        "ai/schema.md",
    ):
        (pages_dir / relative).write_text("reserved", encoding="utf-8")
    outside = chronovisor_root / "outside.md"
    outside.write_text("outside", encoding="utf-8")
    (pages_dir / "outside-link.md").symlink_to(outside)

    monkeypatch.setattr(dashboard, "CHRONOVISOR_ROOT", chronovisor_root)

    mix = dashboard._knowledge_mix_snapshot()

    assert mix["total_pages"] == 3
    assert mix["total_bytes"] == sum(
        len(content.encode("utf-8")) for content in stable_texts.values()
    )
    assert [row["id"] for row in mix["categories"]] == ["ai", "macos"]
    assert mix["categories"][0]["label"] == "AI"
    assert mix["categories"][0]["pages"] == 2
    assert mix["categories"][0]["bytes"] == sum(
        len(stable_texts[path].encode("utf-8"))
        for path in stable_texts
        if path.parent.name == "ai"
    )
    assert "ai/agent-memory.md" in mix["categories"][0]["samples"]
