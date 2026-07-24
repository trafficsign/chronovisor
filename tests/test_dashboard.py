"""Tests for dashboard data assembly."""

from __future__ import annotations

import json
import os
import threading
from datetime import date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

from chronovisor import dashboard, orchestrator, runtime_status
from chronovisor.runtime_config import SearchEmbeddingConfig


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
    from chronovisor import orchestrator

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
    from chronovisor import orchestrator

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
        {"pair_agreement": 4, "tie_break_used": 2, "unresolved_quarantine": 1}
    )
    root.mkdir(parents=True, exist_ok=True)
    (root / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (root / "audit.jsonl").write_text(
        json.dumps(
            {
                "kind": "decision",
                "timestamp": "2026-07-11T12:00:00Z",
                "request_sha256": "a" * 64,
                "status": "agreed",
                "pair_agreement": True,
                "tie_break_used": False,
                "unresolved_quarantine": False,
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
    assert snapshot["summary"]["sessions"]["first_pass_valid"] == 7
    assert snapshot["summary"]["sessions"]["repaired"] == 2
    assert snapshot["summary"]["sessions"]["repair_turns"] == 3
    assert snapshot["summary"]["decisions"]["pair_agreement"] == 4
    assert snapshot["summary"]["decisions"]["tie_break_used"] == 2
    assert snapshot["summary"]["decisions"]["unresolved_quarantine"] == 1
    assert "prompt" not in snapshot["history"][0]
    assert "raw_output" not in snapshot["history"][0]
    assert not (active_dir / "dead.json").exists()
    assert not (active_dir / "stale.json").exists()


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
                "repair_turns": 0,
            }
        ],
        None,
    )

    assert trace["state"] == "active"
    assert trace["task_role"] == "ingest_review"
    assert trace["lanes"][0]["state"] == "done"
    assert trace["lanes"][1]["state"] == "active"
    assert trace["lanes"][1]["steps"][4]["status"] == "active"
    assert trace["lanes"][2]["state"] == "pending"
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
    assert trace["overall"][4]["status"] == "pending"


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
            "raw_output": "must not escape",
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
        "event-2",
    ]
    assert trace["event_count"] == 2
    assert trace["events"][0]["lane"] == "primary"
    assert trace["events"][0]["overall_key"] == "generate"
    assert trace["events"][1]["label"] == "Vote accepted"
    assert "prompt" not in trace["events"][0]
    assert "raw_output" not in trace["events"][0]


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
        }
        for index, role in enumerate(("primary", "challenger"))
    ]
    trace = dashboard._decision_trace_snapshot([], [*sessions, decision], decision)

    assert trace["state"] == "agreed"
    assert trace["summary"] == "2/2 pair agreement"
    assert trace["lanes"][0]["state"] == "done"
    assert trace["lanes"][1]["state"] == "done"
    assert trace["lanes"][2]["state"] == "skipped"
    assert trace["overall"][4]["status"] == "done"
    assert trace["overall"][5]["status"] == "done"


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
    app = (dashboard.STATIC_DIR / "app.js").read_text(encoding="utf-8")
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
    assert "<span>Page changes</span>" in page
    assert "${pageChanges} changes" in app
    assert "${pages} pages" not in app
    assert "grid-template-columns: repeat(7, minmax(0, 1fr));" in style
    assert "grid-template-columns: repeat(6, minmax(50px, 1fr));" in style
    assert "height: var(--panel-height);" in style
    assert "#model-lab-panel" in style
    assert "#model-panel .model-grid" in style
    assert "height: 500px;" in style
    assert "height: 764px;" in style
    assert "height: 1084px;" in style
    assert "#stage-value" in style
    assert "text-overflow: ellipsis;" in style
    assert "white-space: nowrap;" in style
    assert ".decision-outcome-facts" in style
    assert ".decision-transition-event.current" in style
    assert "function reconcileDecisionSteps" in app
    assert "const decisionTracePlayback" in app
    assert "const ACTIVE_DECISION_REFRESH_DELAY_MS = 800" in app
    assert 'fetch("/api/local-consensus"' in app
    assert "No synthetic progress" in app
    assert ".decision-trace-panel" in style


def test_build_snapshot_combines_runtime_and_queue(tmp_path: Path, monkeypatch) -> None:
    chronovisor_root = tmp_path / "wiki"
    raw_dir = chronovisor_root / "raw"
    runtime_dir = chronovisor_root / "runtime"
    raw_dir.mkdir(parents=True)
    (chronovisor_root / "pages").mkdir()
    (chronovisor_root / "system").mkdir()
    (raw_dir / "r1.md").write_text("raw")

    monkeypatch.setattr(dashboard, "CHRONOVISOR_ROOT", chronovisor_root)
    monkeypatch.setattr(dashboard, "LOG_FILE", chronovisor_root / "log.md")
    monkeypatch.setattr(runtime_status, "RUNTIME_DIR", runtime_dir)
    monkeypatch.setattr(runtime_status, "STATUS_FILE", runtime_dir / "status.json")
    monkeypatch.setattr(runtime_status, "EVENTS_FILE", runtime_dir / "events.jsonl")
    monkeypatch.setattr(runtime_status, "METRICS_FILE", runtime_dir / "metrics.jsonl")

    from chronovisor import orchestrator, store

    monkeypatch.setattr(store, "CHRONOVISOR_ROOT", chronovisor_root)
    monkeypatch.setattr(store, "RAW_DIR", raw_dir)
    monkeypatch.setattr(store, "PAGES_DIR", chronovisor_root / "pages")
    monkeypatch.setattr(store, "SYSTEM_DIR", chronovisor_root / "system")
    monkeypatch.setattr(store, "INDEX_FILE", chronovisor_root / "index.md")
    monkeypatch.setattr(store, "LOG_FILE", chronovisor_root / "log.md")
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


def test_snapshot_handler_returns_json_error_instead_of_empty_socket(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        dashboard,
        "build_snapshot",
        lambda: (_ for _ in ()).throw(RuntimeError("snapshot exploded")),
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
        "error_class": "RuntimeError",
        "error": "snapshot exploded",
    }


def test_dashboard_lan_token_is_private_and_reused(tmp_path: Path) -> None:
    token_path = tmp_path / "runtime" / "dashboard-access-token"

    first = dashboard._load_or_create_dashboard_token(token_path)
    second = dashboard._load_or_create_dashboard_token(token_path)

    assert first == second
    assert dashboard.DASHBOARD_TOKEN_RE.fullmatch(first)
    assert token_path.stat().st_mode & 0o777 == 0o600

    rotated = dashboard._rotate_dashboard_token(token_path)

    assert rotated != first
    assert dashboard._load_or_create_dashboard_token(token_path) == rotated
    assert token_path.stat().st_mode & 0o777 == 0o600


def test_dashboard_credentials_are_hashed_private_and_verifiable(
    tmp_path: Path,
) -> None:
    credentials_path = tmp_path / "runtime" / "dashboard-credentials.json"
    password = "test-password-never-store-verbatim"

    dashboard._write_dashboard_credentials(credentials_path, "admin", password)
    credentials = dashboard._load_dashboard_credentials(credentials_path)

    assert password not in credentials_path.read_text(encoding="utf-8")
    assert credentials_path.stat().st_mode & 0o777 == 0o600
    assert dashboard._dashboard_credentials_match(credentials, "admin", password)
    assert not dashboard._dashboard_credentials_match(
        credentials, "admin", "wrong-password"
    )
    assert not dashboard._dashboard_credentials_match(credentials, "other", password)


def test_dashboard_private_client_scope_rejects_public_addresses() -> None:
    assert dashboard._private_client_scope("127.0.0.1") == "loopback"
    assert dashboard._private_client_scope("192.168.1.22") == "private"
    assert dashboard._private_client_scope("10.1.2.3") == "private"
    assert dashboard._private_client_scope("8.8.8.8") == "public"
    assert dashboard._private_client_scope("not-an-ip") == "invalid"


def test_dashboard_lan_hosts_prefers_routable_ip_over_mdns_and_link_local(
    monkeypatch,
) -> None:
    monkeypatch.setattr(dashboard.socket, "gethostname", lambda: "MacStudio.local")

    def fake_run(args, **_kwargs):
        addresses = {"en0": "169.254.240.103\n", "en1": "192.168.100.5\n"}
        return SimpleNamespace(stdout=addresses[str(args[-1])])

    monkeypatch.setattr(dashboard.subprocess, "run", fake_run)

    assert dashboard._dashboard_lan_hosts() == [
        "192.168.100.5",
        "MacStudio.local",
        "169.254.240.103",
    ]


def test_dashboard_lan_link_bootstraps_cookie_and_removes_query_token(
    monkeypatch,
) -> None:
    token = "a" * 43
    monkeypatch.setattr(dashboard, "_dashboard_lan_hosts", lambda: ["store.local"])
    server = dashboard.ThreadingHTTPServer(("127.0.0.1", 0), dashboard.DashboardHandler)
    server.lan_access_enabled = True
    server.lan_access_token = token
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        with dashboard.httpx.Client(follow_redirects=False, timeout=2) as client:
            bootstrap = client.get(
                f"http://{host}:{port}/?access_token={token}&view=trace"
            )
            page = client.get(f"http://{host}:{port}/")
            access = client.get(f"http://{host}:{port}/api/lan-access")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert bootstrap.status_code == 303
    assert bootstrap.headers["location"] == "/?view=trace"
    assert "HttpOnly" in bootstrap.headers["set-cookie"]
    assert "Max-Age=31536000" in bootstrap.headers["set-cookie"]
    assert page.status_code == 200
    assert access.json() == {
        "enabled": True,
        "urls": [f"http://store.local:{port}/?access_token={token}"],
        "trusted_lan_only": True,
    }


def test_dashboard_private_lan_uses_basic_auth_and_keeps_recovery_token(
    tmp_path: Path,
    monkeypatch,
) -> None:
    token = "b" * 43
    credentials_path = tmp_path / "dashboard-credentials.json"
    dashboard._write_dashboard_credentials(credentials_path, "admin", "correct-pass")
    credentials = dashboard._load_dashboard_credentials(credentials_path)
    monkeypatch.setattr(dashboard, "_private_client_scope", lambda _value: "private")
    server = dashboard.ThreadingHTTPServer(("127.0.0.1", 0), dashboard.DashboardHandler)
    server.lan_access_enabled = True
    server.lan_access_token = token
    server.dashboard_credentials = credentials
    server.login_attempt_lock = threading.Lock()
    server.login_attempts = {}
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        with dashboard.httpx.Client(follow_redirects=False, timeout=2) as client:
            challenge = client.get(f"http://{host}:{port}/?view=trace")
            rejected = client.get(
                f"http://{host}:{port}/",
                auth=("admin", "wrong-pass"),
            )
            accepted = client.get(
                f"http://{host}:{port}/",
                auth=("admin", "correct-pass"),
            )
        with dashboard.httpx.Client(follow_redirects=False, timeout=2) as recovery:
            bootstrap = recovery.get(
                f"http://{host}:{port}/?access_token={token}&view=trace"
            )
            recovered = recovery.get(f"http://{host}:{port}/")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert challenge.status_code == 401
    assert challenge.headers["www-authenticate"] == (
        'Basic realm="Chronovisor Dashboard", charset="UTF-8"'
    )
    assert rejected.status_code == 401
    assert accepted.status_code == 200
    assert bootstrap.status_code == 303
    assert bootstrap.headers["location"] == "/?view=trace"
    assert recovered.status_code == 200


def test_dashboard_basic_auth_rate_limits_repeated_failures(
    tmp_path: Path,
    monkeypatch,
) -> None:
    credentials_path = tmp_path / "dashboard-credentials.json"
    dashboard._write_dashboard_credentials(credentials_path, "admin", "correct-pass")
    monkeypatch.setattr(dashboard, "_private_client_scope", lambda _value: "private")
    server = dashboard.ThreadingHTTPServer(("127.0.0.1", 0), dashboard.DashboardHandler)
    server.lan_access_enabled = True
    server.lan_access_token = "c" * 43
    server.dashboard_credentials = dashboard._load_dashboard_credentials(
        credentials_path
    )
    server.login_attempt_lock = threading.Lock()
    server.login_attempts = {}
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        with dashboard.httpx.Client(follow_redirects=False, timeout=2) as client:
            challenge = client.get(f"http://{host}:{port}/")
            responses = [
                client.get(
                    f"http://{host}:{port}/",
                    auth=("admin", "wrong"),
                )
                for _ in range(dashboard.DASHBOARD_LOGIN_ATTEMPT_LIMIT)
            ]
            blocked = client.get(
                f"http://{host}:{port}/",
                auth=("admin", "correct-pass"),
            )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert challenge.status_code == 401
    assert all(response.status_code == 401 for response in responses[:-1])
    assert responses[-1].status_code == 429
    assert blocked.status_code == 429


def test_cached_snapshot_reuses_idle_result_until_a_source_changes(
    tmp_path: Path, monkeypatch
) -> None:
    chronovisor_root = tmp_path / "wiki"
    runtime_dir = chronovisor_root / "runtime"
    runtime_dir.mkdir(parents=True)
    monkeypatch.setattr(dashboard, "CHRONOVISOR_ROOT", chronovisor_root)
    monkeypatch.setattr(dashboard, "LOG_FILE", chronovisor_root / "log.md")
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
    assert dashboard._cached_snapshot()["serial"] == 1
    runtime_status.STATUS_FILE.write_text("{}\n", encoding="utf-8")
    assert dashboard._cached_snapshot()["serial"] == 2
    assert calls == 2


def test_cached_snapshot_uses_short_ttl_while_active(
    tmp_path: Path, monkeypatch
) -> None:
    chronovisor_root = tmp_path / "wiki"
    chronovisor_root.mkdir()
    monkeypatch.setattr(dashboard, "CHRONOVISOR_ROOT", chronovisor_root)
    monkeypatch.setattr(dashboard, "LOG_FILE", chronovisor_root / "log.md")
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
    clock[0] += dashboard.SNAPSHOT_ACTIVE_CACHE_SECONDS / 2
    assert dashboard._cached_snapshot()["serial"] == 1
    clock[0] += dashboard.SNAPSHOT_ACTIVE_CACHE_SECONDS
    assert dashboard._cached_snapshot()["serial"] == 2
    assert calls == 2


def test_cached_snapshot_serves_stale_while_refreshing_in_background(
    monkeypatch,
) -> None:
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
    monkeypatch.setattr(dashboard, "init_chronovisor", lambda: None)
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

    snapshot = dashboard.build_fast_snapshot()

    assert snapshot["status"]["pending"] == 3
    assert snapshot["events"] == [{"kind": "event"}]
    assert snapshot["metrics"] == [{"kind": "metric"}]
    assert snapshot["save_history"] == {}
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
    chronovisor_root = tmp_path / "wiki"
    runtime_dir = chronovisor_root / "runtime"
    runtime_dir.mkdir(parents=True)
    monkeypatch.setattr(dashboard, "CHRONOVISOR_ROOT", chronovisor_root)
    monkeypatch.setattr(dashboard, "LOG_FILE", chronovisor_root / "log.md")
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
            runtime_status.STATUS_FILE.write_text("{}\n", encoding="utf-8")
        return {
            "serial": calls,
            "status": {"state": "idle", "batch": {"active": False}},
        }

    monkeypatch.setattr(dashboard, "build_snapshot", fake_build)

    assert dashboard._cached_snapshot()["serial"] == 1
    assert dashboard._cached_snapshot()["serial"] == 2
    assert dashboard._cached_snapshot()["serial"] == 2
    assert calls == 2


def test_cached_idle_snapshot_invalidates_on_standalone_consensus_activity(
    tmp_path: Path, monkeypatch
) -> None:
    chronovisor_root = tmp_path / "wiki"
    chronovisor_root.mkdir()
    monkeypatch.setattr(dashboard, "CHRONOVISOR_ROOT", chronovisor_root)
    monkeypatch.setattr(dashboard, "LOG_FILE", chronovisor_root / "log.md")
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
    active_dir = chronovisor_root / "runtime" / "local-consensus" / "active"
    active_dir.mkdir(parents=True)
    (active_dir / "request.json").write_text("{}\n", encoding="utf-8")
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
    monkeypatch.setattr(dashboard, "LOG_FILE", chronovisor_root / "log.md")
    monkeypatch.setattr(runtime_status, "RUNTIME_DIR", runtime_dir)
    monkeypatch.setattr(runtime_status, "STATUS_FILE", runtime_dir / "status.json")
    monkeypatch.setattr(runtime_status, "EVENTS_FILE", runtime_dir / "events.jsonl")
    monkeypatch.setattr(runtime_status, "METRICS_FILE", runtime_dir / "metrics.jsonl")

    from chronovisor import orchestrator, store

    monkeypatch.setattr(store, "CHRONOVISOR_ROOT", chronovisor_root)
    monkeypatch.setattr(store, "RAW_DIR", raw_dir)
    monkeypatch.setattr(store, "PAGES_DIR", chronovisor_root / "pages")
    monkeypatch.setattr(store, "SYSTEM_DIR", chronovisor_root / "system")
    monkeypatch.setattr(store, "INDEX_FILE", chronovisor_root / "index.md")
    monkeypatch.setattr(store, "LOG_FILE", chronovisor_root / "log.md")
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
                    "name": "gpt-oss:20b",
                    "model": "gpt-oss:20b",
                    "size": 13_000,
                    "details": {"format": "gguf", "quantization_level": "MXFP4"},
                    "capabilities": ["completion"],
                },
            ],
        },
    )
    monkeypatch.setattr(
        dashboard,
        "load_audit_policy",
        lambda: SimpleNamespace(enabled=True, model="qwen3.6:35b-a3b-mxfp8"),
    )
    monkeypatch.setattr(
        dashboard.recall_runtime,
        "load_policy",
        lambda: SimpleNamespace(
            judge_mode="auto",
            judge_model="qwen3.5:4b-mlx",
            rewrite_enabled=True,
            rewrite_model="qwen3.5:4b-mlx",
        ),
    )
    monkeypatch.setattr(
        dashboard,
        "configured_models",
        lambda models=None: ("qwen3.6:35b-a3b-mxfp8", "gemma4:26b-mxfp8"),
    )
    decision_config = SimpleNamespace(
        primary_model="qwen3.6:35b-a3b-mxfp8",
        challenger_model="gpt-oss:20b",
        tie_break_model="gemma4:26b-mxfp8",
    )
    monkeypatch.setattr(
        dashboard, "load_decision_router_config", lambda: decision_config
    )
    monkeypatch.setattr(
        dashboard,
        "resolve_router_policy",
        lambda config: SimpleNamespace(config=config),
    )
    monkeypatch.setattr(dashboard, "embedding_model", lambda: "bge-m3")
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
    assert by_name["gpt-oss:20b"]["status"] == "ready"
    assert by_name["gpt-oss:20b"]["roles"] == ["decision-challenger"]
    assert by_name["qwen3.5:4b-mlx"]["status"] == "missing"
    assert by_name["qwen3.5:4b-mlx"]["roles"] == ["gate", "rewrite"]


def test_configured_model_roles_use_adopted_router_triplet(monkeypatch) -> None:
    monkeypatch.setattr(
        dashboard,
        "load_search_embedding_config",
        lambda: SearchEmbeddingConfig(enabled=False),
    )
    bootstrap = SimpleNamespace(
        primary_model="bootstrap-primary",
        challenger_model="bootstrap-challenger",
        tie_break_model="bootstrap-tie",
    )
    adopted = SimpleNamespace(
        primary_model="adopted-primary",
        challenger_model="adopted-challenger",
        tie_break_model="adopted-tie",
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
    monkeypatch.setattr(dashboard, "configured_models", lambda _models=None: ())
    monkeypatch.setattr(dashboard, "load_decision_router_config", lambda: bootstrap)
    monkeypatch.setattr(
        dashboard,
        "resolve_router_policy",
        lambda _config: SimpleNamespace(config=adopted),
    )
    monkeypatch.setattr(dashboard, "embedding_model", lambda: "")
    monkeypatch.setattr(
        dashboard,
        "load_reranker_config",
        lambda: SimpleNamespace(enabled=False),
    )

    roles = dashboard._configured_model_roles()

    assert set(roles) == {
        "adopted-primary",
        "adopted-challenger",
        "adopted-tie",
    }
    assert not any(name.startswith("bootstrap-") for name in roles)


def test_decision_router_dashboard_resolution_is_cached(monkeypatch) -> None:
    configured = SimpleNamespace(
        adoption_artifact="",
        primary_model="primary",
        challenger_model="challenger",
        tie_break_model="tie",
    )
    calls = 0
    monkeypatch.setattr(dashboard, "load_decision_router_config", lambda: configured)
    monkeypatch.setattr(
        dashboard,
        "_DECISION_ROUTER_CACHE",
        {"key": None, "expires_at": 0.0, "config": None},
    )

    def resolve(config):
        nonlocal calls
        calls += 1
        return SimpleNamespace(config=config)

    monkeypatch.setattr(dashboard, "resolve_router_policy", resolve)

    assert dashboard._resolved_decision_router_config() is configured
    assert dashboard._resolved_decision_router_config() is configured
    assert calls == 1


def test_decision_router_dashboard_cache_ttl_starts_after_resolution(
    monkeypatch,
) -> None:
    configured = SimpleNamespace(
        adoption_artifact="",
        primary_model="primary",
        challenger_model="challenger",
        tie_break_model="tie",
    )
    calls = 0
    clock = iter((100.0, 120.0, 134.9))
    monkeypatch.setattr(dashboard, "load_decision_router_config", lambda: configured)
    monkeypatch.setattr(dashboard.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(
        dashboard,
        "_DECISION_ROUTER_CACHE",
        {"key": None, "expires_at": 0.0, "config": None},
    )

    def resolve(config):
        nonlocal calls
        calls += 1
        return SimpleNamespace(config=config)

    monkeypatch.setattr(dashboard, "resolve_router_policy", resolve)

    assert dashboard._resolved_decision_router_config() is configured
    assert dashboard._resolved_decision_router_config() is configured
    assert calls == 1
    assert dashboard._DECISION_ROUTER_CACHE["expires_at"] == 135.0


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
    from chronovisor import frontier_review

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
    log_file = chronovisor_root / "log.md"

    monkeypatch.setattr(dashboard, "CHRONOVISOR_ROOT", chronovisor_root)
    monkeypatch.setattr(dashboard, "LOG_FILE", log_file)

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
    log_file.write_text(
        "\n".join(
            [
                "# Change Log",
                "- [2026-07-01 12:31] ingest | created save-history-dashboard",
                "- [2026-07-01 12:32] ingest | updated chronovisor-dashboard",
                "- [2026-07-02 12:32] ingest | updated save-history-dashboard",
            ]
        ),
        encoding="utf-8",
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
    monkeypatch.setattr(dashboard, "LOG_FILE", chronovisor_root / "log.md")

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
    monkeypatch.setattr(dashboard, "LOG_FILE", chronovisor_root / "log.md")

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
    monkeypatch.setattr(dashboard, "LOG_FILE", chronovisor_root / "log.md")

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
    monkeypatch.setattr(dashboard, "LOG_FILE", tmp_path / "wiki" / "log.md")

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
    monkeypatch.setattr(dashboard, "LOG_FILE", chronovisor_root / "log.md")

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
    (pages_dir / "ai" / "agent-memory.md").write_text("a" * 20, encoding="utf-8")
    (pages_dir / "ai" / "evals.md").write_text("b" * 10, encoding="utf-8")
    (pages_dir / "macos" / "display.md").write_text("c" * 15, encoding="utf-8")

    monkeypatch.setattr(dashboard, "CHRONOVISOR_ROOT", chronovisor_root)

    mix = dashboard._knowledge_mix_snapshot()

    assert mix["total_pages"] == 3
    assert mix["total_bytes"] == 45
    assert [row["id"] for row in mix["categories"]] == ["ai", "macos"]
    assert mix["categories"][0]["label"] == "AI"
    assert mix["categories"][0]["pages"] == 2
    assert mix["categories"][0]["bytes"] == 30
    assert round(mix["categories"][0]["share"], 3) == 0.667
    assert "ai/agent-memory.md" in mix["categories"][0]["samples"]
