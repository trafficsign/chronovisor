from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from llm_wiki_mcp.frontier_guard import (
    EvidenceValidationError,
    FrontierGuard,
    PermitDenied,
    RepairIncidentEvidence,
    repair_fingerprint,
)


BASE = datetime(2026, 7, 11, 8, 0, tzinfo=timezone.utc)


def evidence(
    name: str = "json-contract",
    **overrides: object,
) -> RepairIncidentEvidence:
    values: dict[str, object] = {
        "component": "watchdog.health_snapshot",
        "fingerprint": repair_fingerprint("watchdog.health_snapshot", name),
        "failure_class": "system_health_snapshot_exception",
        "occurrence_count": 3,
        "distinct_inputs": (
            repair_fingerprint("input", name, 1),
            repair_fingerprint("input", name, 2),
        ),
        "local_repair_attempts": 2,
        "local_repair_evidence": ("a" * 64, "b" * 64),
        "reproduction_command": ("pytest", "tests/test_ingest.py", "-k", name),
        "notes": {"producer": "trusted_watchdog", "incident_key": f"incident-{name}"},
    }
    values.update(overrides)
    return RepairIncidentEvidence(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"role": "semantic_review"}, "role must be code_repair"),
        ({"incident_kind": "content_review"}, "system_code_repair"),
        ({"fingerprint": ""}, "fingerprint"),
        ({"occurrence_count": 2}, "at least 3"),
        ({"distinct_inputs": ("same", "same")}, "2 distinct"),
        ({"local_repair_attempts": 1}, "2 local repair attempts"),
        ({"local_repair_evidence": ("a" * 64,)}, "evidence digest"),
        (
            {
                "reproduction_command": (),
                "failing_test": None,
                "reproduction_artifact": None,
            },
            "reproduction command",
        ),
        ({"failure_class": "auth_required"}, "human boundaries"),
        ({"failure_class": "quota_or_billing_required"}, "human boundaries"),
        ({"failure_class": "keychain_permission_required"}, "human boundaries"),
    ],
)
def test_repair_evidence_is_strict(
    overrides: dict[str, object], message: str
) -> None:
    with pytest.raises(EvidenceValidationError, match=message):
        evidence(**overrides)


def test_all_local_unavailable_is_the_only_occurrence_threshold_bypass() -> None:
    admitted = evidence(
        occurrence_count=0,
        all_local_models_unavailable=True,
        local_unavailability_artifact="runtime/local-model-health.json",
    )

    assert admitted.occurrence_count == 0
    with pytest.raises(EvidenceValidationError, match="health-check artifact"):
        evidence(occurrence_count=0, all_local_models_unavailable=True)


def test_reserve_does_not_spend_budget_and_start_records_pid_owner(
    tmp_path: Path,
) -> None:
    guard = FrontierGuard(tmp_path / "guard")
    permit = guard.reserve(evidence(), owner="self-heal:test", now=BASE)

    reserved = guard.inspect(now=BASE).state
    incident = reserved["incidents"][permit.incident_id]
    assert incident["status"] == "reserved"
    assert incident["started_at"] is None
    assert reserved["fingerprints"] == {}

    started = permit.start(pid=os.getpid(), now=BASE + timedelta(seconds=1))
    assert started["status"] == "started"
    assert started["pid"] == os.getpid()
    assert started["owner"] == "self-heal:test"
    assert guard.inspect(now=BASE + timedelta(seconds=1)).state["fingerprints"]

    finished = permit.finish("failed", now=BASE + timedelta(seconds=2))
    assert finished["status"] == "failed"
    assert guard.inspect(now=BASE + timedelta(seconds=2)).state["active_incident_id"] is None


def test_unstarted_release_does_not_consume_global_budget(tmp_path: Path) -> None:
    guard = FrontierGuard(tmp_path / "guard")
    first = guard.reserve(evidence("first"), now=BASE)
    first.release(now=BASE + timedelta(seconds=1))

    second = guard.reserve(evidence("second"), now=BASE + timedelta(seconds=2))
    assert second.status == "reserved"
    second.release(now=BASE + timedelta(seconds=3))


def test_different_fingerprint_is_still_global_single_flight(tmp_path: Path) -> None:
    guard_root = tmp_path / "guard"
    first_guard = FrontierGuard(guard_root)
    second_guard = FrontierGuard(guard_root)
    current = datetime.now(timezone.utc)

    with first_guard.permit(evidence("first"), now=current):
        with pytest.raises(PermitDenied) as denied:
            with second_guard.permit(evidence("second"), now=current):
                pytest.fail("second frontier permit must never be yielded")

    assert denied.value.reason == "global_single_flight"


def test_started_incident_cannot_execute_twice_and_state_survives_reopen(
    tmp_path: Path,
) -> None:
    root = tmp_path / "guard"
    first_guard = FrontierGuard(root)
    permit = first_guard.reserve(evidence(), owner="worker-a", now=BASE)
    permit.start(pid=os.getpid(), now=BASE)

    with pytest.raises(PermitDenied) as denied:
        permit.start(pid=os.getpid(), now=BASE + timedelta(seconds=1))
    assert denied.value.reason == "incident_already_started"

    permit.finish("succeeded", details={"commit": "abc123"}, now=BASE + timedelta(seconds=2))
    reopened = FrontierGuard(root)
    state = reopened.inspect(now=BASE + timedelta(seconds=3)).state
    assert state["incidents"][permit.incident_id]["status"] == "succeeded"
    assert state["incidents"][permit.incident_id]["result"] == {"commit": "abc123"}

    events = [json.loads(line) for line in reopened.events_file.read_text().splitlines()]
    assert [event["event"] for event in events] == [
        "incident_reserved",
        "incident_started",
        "incident_finished",
    ]
    assert [event["sequence"] for event in events] == [1, 2, 3]


def test_fingerprint_cooldown_and_global_daily_budget_are_durable(
    tmp_path: Path,
) -> None:
    root = tmp_path / "guard"
    guard = FrontierGuard(root)
    first = guard.reserve(evidence("same"), now=BASE)
    first.start(pid=os.getpid(), now=BASE)
    first.finish("failed", now=BASE + timedelta(seconds=1))

    reopened = FrontierGuard(root)
    with pytest.raises(PermitDenied) as same_denied:
        reopened.reserve(evidence("same"), now=BASE + timedelta(hours=23))
    assert same_denied.value.reason == "incident_already_started"

    with pytest.raises(PermitDenied) as other_denied:
        reopened.reserve(evidence("other"), now=BASE + timedelta(hours=23))
    assert other_denied.value.reason == "global_24h_budget"

    next_day = reopened.reserve(evidence("other"), now=BASE + timedelta(hours=24, seconds=1))
    assert next_day.status == "reserved"
    next_day.release(now=BASE + timedelta(hours=24, seconds=2))

    with pytest.raises(PermitDenied) as same_packet_denied:
        reopened.reserve(evidence("same"), now=BASE + timedelta(hours=25))
    assert same_packet_denied.value.reason == "incident_already_started"


def test_crashed_started_incident_becomes_terminal_abandoned(tmp_path: Path) -> None:
    root = tmp_path / "guard"
    guard = FrontierGuard(root, default_lease=timedelta(seconds=5))
    permit = guard.reserve(evidence(), now=BASE)
    permit.start(pid=999_999_999, now=BASE)

    reopened = FrontierGuard(root, default_lease=timedelta(seconds=5))
    before_expiry = reopened.inspect(dry_run=True, now=BASE + timedelta(seconds=1))
    assert before_expiry.would_abandon == ()
    assert before_expiry.state["incidents"][permit.incident_id]["status"] == "started"

    recovered = reopened.inspect(dry_run=False, now=BASE + timedelta(seconds=6))
    incident = recovered.state["incidents"][permit.incident_id]
    assert recovered.would_abandon == (permit.incident_id,)
    assert incident["status"] == "abandoned"
    assert incident["abandon_reason"] == "lease_expired"
    assert recovered.state["active_incident_id"] is None

    with pytest.raises(PermitDenied, match="incident_already_started"):
        permit.start(pid=os.getpid(), now=BASE + timedelta(hours=25))


def test_dry_inspect_projects_recovery_without_any_filesystem_write(
    tmp_path: Path,
) -> None:
    empty_root = tmp_path / "empty"
    empty = FrontierGuard(empty_root).inspect(dry_run=True, now=BASE)
    assert empty.state["incidents"] == {}
    assert not empty_root.exists()

    root = tmp_path / "guard"
    guard = FrontierGuard(root, default_lease=timedelta(seconds=1))
    permit = guard.reserve(evidence(), now=BASE)
    permit.start(pid=os.getpid(), now=BASE, lease=timedelta(seconds=1))
    before = {
        path.name: path.read_bytes()
        for path in root.iterdir()
        if path.is_file()
    }

    projected = FrontierGuard(root).inspect(
        dry_run=True,
        now=BASE + timedelta(seconds=2),
    )
    after = {
        path.name: path.read_bytes()
        for path in root.iterdir()
        if path.is_file()
    }
    assert projected.would_abandon == (permit.incident_id,)
    assert projected.state["incidents"][permit.incident_id]["status"] == "abandoned"
    assert before == after
    assert json.loads(guard.state_file.read_text())["incidents"][permit.incident_id][
        "status"
    ] == "started"


def test_context_manager_abandons_unfinished_started_incident(tmp_path: Path) -> None:
    guard = FrontierGuard(tmp_path / "guard")
    current = datetime.now(timezone.utc)
    with guard.permit(evidence(), now=current) as permit:
        permit.start(pid=os.getpid(), now=current)

    state = guard.inspect(now=current + timedelta(seconds=1)).state
    assert state["incidents"][permit.incident_id]["status"] == "abandoned"
    assert (
        state["incidents"][permit.incident_id]["abandon_reason"]
        == "permit_context_exited_without_finish"
    )
