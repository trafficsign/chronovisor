from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from chronovisor.ingest.system_incident_supervisor import SystemIncidentSupervisor

BASE = datetime(2026, 7, 11, 12, 0, tzinfo=UTC)


class FakeClock:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self) -> datetime:
        value = BASE + timedelta(seconds=self.calls)
        self.calls += 1
        return value


def _tree_bytes(root: Path) -> dict[str, bytes]:
    if not root.exists():
        return {}
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _failing_runner(_attempt: int) -> None:
    raise RuntimeError("index invariant 9182 at /Users/alice/private-page.md")


def _repairer(attempt: int, *, dry_run: bool) -> dict[str, object]:
    return {
        "action_id": f"test-local-repair-{attempt}",
        "performed": not dry_run,
        "projected": dry_run,
    }


@pytest.mark.parametrize(
    "error",
    [
        json.JSONDecodeError("model json output", "{", 0),
        RuntimeError("semantic disagreement in structured output"),
        RuntimeError("keychain credential permission required"),
        RuntimeError("billing quota exhausted"),
    ],
)
def test_routine_and_human_boundary_errors_are_never_observed(
    tmp_path: Path,
    error: BaseException,
) -> None:
    runner_calls: list[int] = []
    supervisor = SystemIncidentSupervisor(
        tmp_path / "incidents",
        packet_dir=tmp_path / "packets",
        enqueue=lambda _path: pytest.fail("excluded incident must not enqueue"),
    )

    result = supervisor.observe_health_snapshot_exception(
        error,
        run_id="run-a",
        runner=lambda attempt: runner_calls.append(attempt),
        repairer=_repairer,
    )

    assert result["status"] == "excluded"
    assert result["reason"] in {"routine_model_or_data_error", "human_boundary"}
    assert runner_calls == []
    assert _tree_bytes(tmp_path) == {}


def test_local_recheck_recovery_does_not_create_durable_incident(tmp_path: Path) -> None:
    supervisor = SystemIncidentSupervisor(
        tmp_path / "incidents",
        packet_dir=tmp_path / "packets",
        enqueue=lambda _path: pytest.fail("recovered incident must not enqueue"),
    )
    calls: list[int] = []

    result = supervisor.observe_health_snapshot_exception(
        RuntimeError("index invariant failed"),
        run_id="run-a",
        runner=lambda attempt: calls.append(attempt) or {"status": "ok"},
        repairer=_repairer,
    )

    assert result["status"] == "recovered_locally"
    assert calls == [1]
    assert _tree_bytes(tmp_path) == {}


def test_different_recheck_exception_is_not_reproducible(tmp_path: Path) -> None:
    supervisor = SystemIncidentSupervisor(
        tmp_path / "incidents",
        packet_dir=tmp_path / "packets",
        enqueue=lambda _path: pytest.fail("different failure must not enqueue"),
    )

    def different_failure(_attempt: int) -> None:
        raise OSError("different subsystem failed")

    result = supervisor.observe_health_snapshot_exception(
        RuntimeError("index invariant failed"),
        run_id="run-a",
        runner=different_failure,
        repairer=_repairer,
    )

    assert result["status"] == "not_reproduced_locally"
    assert result["local_repairs"][0]["status"] == "different_failure"
    assert _tree_bytes(tmp_path) == {}


def test_three_reproductions_two_identities_create_and_enqueue_one_packet(
    tmp_path: Path,
) -> None:
    enqueued: list[Path] = []

    def fake_enqueue(path: Path) -> dict[str, object]:
        enqueued.append(path)
        return {"job_id": "job-1", "enqueued": True, "coalesced": False}

    supervisor = SystemIncidentSupervisor(
        tmp_path / "incidents",
        packet_dir=tmp_path / "packets",
        clock=FakeClock(),
        enqueue=fake_enqueue,
    )
    error = RuntimeError("index invariant 9182 at /Users/alice/private-page.md")

    first = supervisor.observe_health_snapshot_exception(
        error,
        run_id="run-a",
        input_id="input-a",
        runner=_failing_runner,
        repairer=_repairer,
    )
    second = supervisor.observe_health_snapshot_exception(
        error,
        run_id="run-a",
        input_id="input-a",
        runner=_failing_runner,
        repairer=_repairer,
    )
    third = supervisor.observe_health_snapshot_exception(
        error,
        run_id="run-b",
        input_id="input-b",
        runner=_failing_runner,
        repairer=_repairer,
    )

    assert first["status"] == "observed"
    assert second["status"] == "observed"
    assert third["status"] == "packet_created"
    assert third["occurrence_count"] == 3
    assert third["distinct_input_count"] == 2
    assert len(enqueued) == 1

    packet_path = Path(third["packet_path"])
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    evidence = packet["repair_evidence"]
    assert packet["incident_kind"] == "system_code_repair"
    assert packet["status"] == "pending_frontier"
    assert packet["local_repair_attempts"] == 2
    assert evidence["role"] == "code_repair"
    assert evidence["incident_kind"] == "system_code_repair"
    assert evidence["occurrence_count"] == 3
    assert evidence["distinct_input_count"] == 2
    assert evidence["local_repair_attempts"] == 2
    assert evidence["notes"]["incident_key"] == packet_path.stem
    assert evidence["reproduction"]["command"]
    assert evidence["reproduction"]["failing_test"]
    assert evidence["reproduction"]["artifact"]

    persisted_text = json.dumps(_tree_bytes(tmp_path), default=lambda value: value.decode("utf-8"))
    assert "/Users/alice" not in persisted_text
    assert "private-page" not in persisted_text
    assert "9182" not in persisted_text

    packet_before = packet_path.read_bytes()
    duplicate = supervisor.observe_health_snapshot_exception(
        error,
        run_id="run-c",
        input_id="input-c",
        runner=_failing_runner,
        repairer=_repairer,
    )
    assert duplicate["status"] == "packet_exists"
    assert enqueued == [packet_path]
    assert packet_path.read_bytes() == packet_before


def test_existing_packet_retries_enqueue_after_failure(tmp_path: Path) -> None:
    calls: list[Path] = []

    def flaky_enqueue(path: Path) -> dict[str, object]:
        calls.append(path)
        if len(calls) == 1:
            raise RuntimeError("queue temporarily unavailable")
        return {"job_id": "job-retry", "enqueued": True, "coalesced": False}

    supervisor = SystemIncidentSupervisor(
        tmp_path / "incidents",
        packet_dir=tmp_path / "packets",
        clock=FakeClock(),
        enqueue=flaky_enqueue,
    )
    error = RuntimeError("index invariant 9182 at /Users/alice/private-page.md")
    for run_id, input_id in (
        ("run-a", "input-a"),
        ("run-a", "input-a"),
        ("run-b", "input-b"),
    ):
        failed = supervisor.observe_health_snapshot_exception(
            error,
            run_id=run_id,
            input_id=input_id,
            runner=_failing_runner,
            repairer=_repairer,
        )

    assert failed["status"] == "packet_enqueue_failed"
    assert failed["enqueue_error_type"] == "RuntimeError"

    retried = supervisor.observe_health_snapshot_exception(
        error,
        run_id="run-c",
        input_id="input-c",
        runner=_failing_runner,
        repairer=_repairer,
    )
    duplicate = supervisor.observe_health_snapshot_exception(
        error,
        run_id="run-d",
        input_id="input-d",
        runner=_failing_runner,
        repairer=_repairer,
    )

    assert retried["status"] == "packet_exists_enqueue_pending"
    assert retried["enqueue"]["job_id"] == "job-retry"
    assert duplicate["status"] == "packet_exists"
    assert len(calls) == 2


def test_dry_run_projects_threshold_without_changing_any_byte(tmp_path: Path) -> None:
    enqueued: list[Path] = []
    supervisor = SystemIncidentSupervisor(
        tmp_path / "incidents",
        packet_dir=tmp_path / "packets",
        clock=FakeClock(),
        enqueue=lambda path: enqueued.append(path) or {"job_id": "unexpected"},
    )
    error = RuntimeError("index invariant 9182 at /Users/alice/private-page.md")
    for run_id in ("run-a", "run-a"):
        supervisor.observe_health_snapshot_exception(
            error,
            run_id=run_id,
            runner=_failing_runner,
            repairer=_repairer,
        )
    before = _tree_bytes(tmp_path)

    result = supervisor.observe_health_snapshot_exception(
        error,
        run_id="run-b",
        runner=_failing_runner,
        repairer=_repairer,
        dry_run=True,
    )

    assert result["status"] == "dry_run"
    assert result["projected_status"] == "would_create_packet"
    assert result["occurrence_count"] == 3
    assert result["distinct_input_count"] == 2
    assert enqueued == []
    assert _tree_bytes(tmp_path) == before


def test_self_heal_enqueue_helper_uses_durable_ledger_without_popen(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chronovisor.ingest import self_heal
    from chronovisor.ops import background_jobs

    captured: list[dict[str, object]] = []
    monkeypatch.setattr(
        background_jobs,
        "enqueue_job",
        lambda **kwargs: captured.append(kwargs) or {"job_id": "job-1", "enqueued": True},
    )
    monkeypatch.setattr(
        self_heal.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("trusted enqueue must not spawn"),
    )
    packet = tmp_path / "packet.json"
    fingerprint = "trusted-system-fingerprint"
    packet.write_text(
        json.dumps(
            {
                "job_id": "trusted-watchdog",
                "failure_class": "system_health_snapshot_exception",
                "incident_kind": "system_code_repair",
                "fingerprint": fingerprint,
                "local_repair_attempts": 2,
                "local_repair_evidence": ["a" * 64, "b" * 64],
                "repair_evidence": {
                    "component": "watchdog.health_snapshot",
                    "fingerprint": fingerprint,
                    "failure_class": "system_health_snapshot_exception",
                    "occurrence_count": 3,
                    "distinct_inputs": ["input-a", "input-b"],
                    "local_repair_attempts": 2,
                    "local_repair_evidence": ["a" * 64, "b" * 64],
                    "reproduction_command": ["uv", "run", "health-check"],
                    "notes": {
                        "producer": "trusted_watchdog",
                        "incident_key": packet.stem,
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    result = self_heal.enqueue_system_code_repair(packet)

    assert result["job_id"] == "job-1"
    assert captured == [
        {
            "name": "system-code-repair",
            "module": "chronovisor.ops.self_heal",
            "args": [
                "--packet",
                str(packet.resolve()),
                "--enable-frontier-repair",
            ],
            "env": {},
            "stdin_text": "",
        }
    ]


def test_watchdog_captures_health_exception_without_raw_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chronovisor.ingest import system_incident_supervisor as supervisor_module
    from chronovisor.ops import autonomy

    secret = "private user text at /Users/alice/secret.md"
    health_calls = 0

    def broken_health() -> dict[str, object]:
        nonlocal health_calls
        health_calls += 1
        raise RuntimeError(secret)

    supervised: list[dict[str, object]] = []

    def fake_supervise(error: BaseException, **kwargs: object) -> dict[str, object]:
        supervised.append({"error_type": error.__class__.__name__, **kwargs})
        runner = kwargs["runner"]
        assert callable(runner)
        for attempt in (1, 2):
            with pytest.raises(RuntimeError):
                runner(attempt)
        return {
            "status": "observed",
            "fingerprint": "safe-fingerprint",
            "occurrence_count": 1,
            "distinct_input_count": 1,
        }

    monkeypatch.setattr("chronovisor.ops.health.health_snapshot", broken_health)
    monkeypatch.setattr(supervisor_module, "supervise_health_snapshot_exception", fake_supervise)
    monkeypatch.setattr(autonomy, "_latest_jsonl", lambda _path: {})

    payload = autonomy.watchdog_snapshot(write=False)

    assert health_calls == 3
    assert supervised and supervised[0]["dry_run"] is True
    assert payload["health"] == {
        "status": "error",
        "component": "watchdog.health_snapshot",
    }
    component = payload["alerts"][0]
    assert component["type"] == "component_error"
    assert component["incident_status"] == "observed"
    assert component["fingerprint"] == "safe-fingerprint"
    assert secret not in json.dumps(payload)


def test_normal_watchdog_alert_never_calls_incident_supervisor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chronovisor.ingest import system_incident_supervisor as supervisor_module
    from chronovisor.ops import autonomy

    monkeypatch.setattr(
        "chronovisor.ops.health.health_snapshot",
        lambda: {
            "memory_integrity": {"capture_rate": 0.95},
            "queues": {"duplicate_candidates": 0, "lint_repair": 0},
        },
    )
    monkeypatch.setattr(autonomy, "_latest_jsonl", lambda _path: {})
    monkeypatch.setattr(
        supervisor_module,
        "supervise_health_snapshot_exception",
        lambda *_args, **_kwargs: pytest.fail("normal alert must not create an incident"),
    )

    payload = autonomy.watchdog_snapshot(write=False)

    assert payload["status"] == "alert"
    assert payload["alerts"][0]["type"] == "sleep_never_ran"


def test_disabled_derived_repair_lane_performs_no_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chronovisor.ingest.system_incident_supervisor import _default_health_repair

    monkeypatch.setenv("CHRONOVISOR_DECISION_POLICY_DERIVED_INDEX_REBUILD", "off")

    with pytest.raises(RuntimeError, match="disabled"):
        _default_health_repair(1, dry_run=False)
