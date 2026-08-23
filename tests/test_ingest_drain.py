"""Tests for the launchd-friendly ingest drain CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chronovisor.core.okf_cutover import OKFStartupDecision
from chronovisor.ingest import ingest_drain, managed_hold_sync, self_heal


@pytest.fixture(autouse=True)
def _disable_runtime_status_reset(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime_root = tmp_path / "wiki"
    (runtime_root / "runtime").mkdir(parents=True)
    for name in ("index.md", "log.md", "schema.md"):
        (runtime_root / name).write_text("legacy\n", encoding="utf-8")
    monkeypatch.setattr(
        ingest_drain.runtime_status,
        "reset_stale_runtime_status",
        lambda: False,
    )
    monkeypatch.setattr(ingest_drain, "init_chronovisor", lambda: runtime_root)
    monkeypatch.setattr(ingest_drain.orchestrator, "reset_stale_lock", lambda: False)
    monkeypatch.setattr(ingest_drain, "CHRONOVISOR_ROOT", runtime_root)
    monkeypatch.setattr(
        ingest_drain,
        "okf_startup_status",
        lambda _root: OKFStartupDecision(
            True, "bootstrap", "uninitialized", "ok"
        ),
    )
    monkeypatch.setattr(
        managed_hold_sync,
        "sync_ingest_semantic_holds",
        lambda **_kwargs: {"status": "ok"},
    )
    monkeypatch.setattr(
        self_heal,
        "run_pending",
        lambda **_kwargs: {"status": "ok"},
    )
    monkeypatch.setattr(ingest_drain.ollama, "ingest_model", lambda: "ornith:test")
    monkeypatch.setattr(
        ingest_drain.ollama,
        "runtime_generation_routes",
        lambda roles: tuple(
            ingest_drain.ollama.RuntimeGenerationRoute(
                role=role,
                provider="ollama",
                model="ornith:test",
                location="local",
                structured_output=True,
            )
            for role in roles
        ),
    )
    monkeypatch.setattr(ingest_drain.ollama, "resident_model_rows", lambda: {})
    monkeypatch.setattr(ingest_drain.ollama, "is_available", lambda: True)
    monkeypatch.setattr(
        ingest_drain.orchestrator,
        "ingest_authority_preflight",
        lambda: {
            "ok": True,
            "status": "ready",
            "blocked_by": None,
            "retryable": False,
            "error": None,
            "artifact_sha256": "a" * 64,
        },
    )
    monkeypatch.setattr(
        ingest_drain.orchestrator,
        "reconcile_processed_projections",
        lambda **_kwargs: {
            "status": "disabled",
            "disabled": True,
            "reason": "test",
            "processed": [],
            "held": [],
        },
    )


def test_release_ingest_runner_unloads_only_when_resident(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, str, dict]] = []
    unloaded: list[str] = []
    monkeypatch.setattr(ingest_drain.ollama, "ingest_model", lambda: "ornith:test")
    monkeypatch.setattr(
        ingest_drain.ollama,
        "resident_model_rows",
        lambda: {"ornith:test": (20, 32768), "recall:test": (4, 4096)},
    )
    monkeypatch.setattr(
        ingest_drain.ollama,
        "unload_named_model",
        lambda model: unloaded.append(model) or True,
    )
    monkeypatch.setattr(
        ingest_drain.runtime_status,
        "safe_append_event",
        lambda level, message, **fields: events.append((level, message, fields)),
    )

    result = ingest_drain._release_ingest_runner()

    assert result == {
        "status": "released",
        "released": True,
        "model": "ornith:test",
    }
    assert unloaded == ["ornith:test"]
    assert events[0][0:2] == ("info", "ingest drain | runner released")


def test_release_ingest_runner_does_not_wake_absent_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ingest_drain.ollama, "ingest_model", lambda: "ornith:test")
    monkeypatch.setattr(
        ingest_drain.ollama,
        "resident_model_rows",
        lambda: {"recall:test": (4, 4096)},
    )
    monkeypatch.setattr(
        ingest_drain.ollama,
        "unload_named_model",
        lambda model: (_ for _ in ()).throw(AssertionError(model)),
    )

    result = ingest_drain._release_ingest_runner()

    assert result == {
        "status": "not_resident",
        "released": False,
        "model": "ornith:test",
    }


def test_remote_drain_skips_all_ollama_probes_and_release_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ingest_drain.ollama,
        "runtime_generation_routes",
        lambda roles: tuple(
            ingest_drain.ollama.RuntimeGenerationRoute(
                role=role,
                provider="remote-test",
                model="remote-model",
                location="remote",
                structured_output=True,
            )
            for role in roles
        ),
    )

    def forbidden(*_args, **_kwargs):
        pytest.fail("remote ingest drain touched Ollama")

    for name in (
        "is_available",
        "resident_model_rows",
        "unload_named_model",
        "model_resource_lease",
        "plan_model_residency",
    ):
        monkeypatch.setattr(ingest_drain.ollama, name, forbidden)
    monkeypatch.setattr(
        ingest_drain.orchestrator,
        "get_pending_raw_files",
        lambda: [],
    )
    monkeypatch.setattr(ingest_drain, "init_chronovisor", lambda: None)
    monkeypatch.setattr(ingest_drain.orchestrator, "reset_stale_lock", lambda: None)

    result = ingest_drain._drain(max_batches=1, sleep_seconds=0)

    assert result["status"] == "drained"
    assert ingest_drain._release_ingest_runner() == {
        "status": "not_applicable",
        "released": False,
    }


def test_drain_attaches_release_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ingest_drain,
        "_drain",
        lambda **kwargs: {"status": "drained", "kwargs": kwargs},
    )
    monkeypatch.setattr(
        ingest_drain,
        "_release_ingest_runner",
        lambda: {"status": "released", "released": True, "model": "ornith:test"},
    )

    result = ingest_drain.drain(max_batches=2, max_units=3, sleep_seconds=0)

    assert result["kwargs"]["max_batches"] == 2
    assert result["kwargs"]["max_units"] == 3
    assert result["model_release"] == {
        "status": "released",
        "released": True,
        "model": "ornith:test",
    }


def test_drain_releases_runner_when_cycle_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    releases: list[bool] = []

    def fail(**kwargs) -> dict:
        raise RuntimeError("cycle failed")

    monkeypatch.setattr(ingest_drain, "_drain", fail)
    monkeypatch.setattr(
        ingest_drain,
        "_release_ingest_runner",
        lambda: releases.append(True) or {"status": "released"},
    )

    with pytest.raises(RuntimeError, match="cycle failed"):
        ingest_drain.drain()

    assert releases == [True]


def test_drain_runs_batches_until_empty(tmp_path: Path, monkeypatch) -> None:
    state = {"pending": 25, "init": 0, "reset": 0}
    events: list[str] = []

    monkeypatch.setattr(
        ingest_drain, "init_chronovisor", lambda: state.__setitem__("init", state["init"] + 1)
    )
    monkeypatch.setattr(
        ingest_drain.orchestrator,
        "reset_stale_lock",
        lambda: state.__setitem__("reset", state["reset"] + 1),
    )
    monkeypatch.setattr(
        ingest_drain.orchestrator,
        "get_pending_raw_files",
        lambda: [object()] * state["pending"],
    )

    def fake_run_pending_ingest(*, force: bool = False, max_units: int = 10) -> dict:
        events.append("ingest")
        assert force is True
        assert max_units == 10
        processed = min(max_units, state["pending"])
        state["pending"] -= processed
        return {
            "triggered": True,
            "files_processed": [f"raw-{i}.md" for i in range(processed)],
        }

    monkeypatch.setattr(
        ingest_drain.orchestrator, "run_pending_ingest", fake_run_pending_ingest
    )
    monkeypatch.setattr(
        ingest_drain.orchestrator,
        "reconcile_processed_projections",
        lambda **kwargs: events.append(f"reconcile:{kwargs['max_parents']}")
        or {"status": "ok", "disabled": False, "processed": [], "held": []},
    )

    log_file = tmp_path / "drain.jsonl"
    result = ingest_drain.drain(max_batches=3, sleep_seconds=0, log_file=log_file)

    assert result["status"] == "drained"
    assert result["pending_start"] == 25
    assert result["pending_after"] == 0
    assert result["batches_run"] == 3
    assert result["files_processed"] == 25
    assert state["init"] == 1
    assert state["reset"] == 1
    assert events[-1] == "reconcile:128"

    records = [json.loads(line) for line in log_file.read_text().splitlines()]
    assert [record["files_processed"] for record in records] == [10, 10, 5]


def test_drain_stops_when_batch_makes_no_progress(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(ingest_drain, "init_chronovisor", lambda: None)
    monkeypatch.setattr(ingest_drain.orchestrator, "reset_stale_lock", lambda: None)
    monkeypatch.setattr(
        ingest_drain.orchestrator, "get_pending_raw_files", lambda: [object()] * 3
    )
    monkeypatch.setattr(
        ingest_drain.orchestrator,
        "run_pending_ingest",
        lambda *, force=False, max_units=10: {
            "triggered": True,
            "files_processed": [],
        },
    )

    result = ingest_drain.drain(
        max_batches=5, sleep_seconds=0, log_file=tmp_path / "drain.jsonl"
    )

    assert result["status"] == "stalled"
    assert result["pending_after"] == 3
    assert result["batches_run"] == 1
    assert result["stop_reason"] == "no batch progress"


def test_drain_check_mode_does_not_run_ingest(monkeypatch) -> None:
    monkeypatch.setattr(ingest_drain, "init_chronovisor", lambda: None)
    monkeypatch.setattr(ingest_drain.orchestrator, "reset_stale_lock", lambda: None)
    monkeypatch.setattr(
        ingest_drain.orchestrator, "get_pending_raw_files", lambda: [object()] * 2
    )
    monkeypatch.setattr(
        ingest_drain.orchestrator,
        "run_pending_ingest",
        lambda *, force=False, max_units=10: (_ for _ in ()).throw(
            AssertionError("should not run")
        ),
    )
    monkeypatch.setattr(
        ingest_drain.orchestrator,
        "reconcile_processed_projections",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("check mode must remain read-only")
        ),
    )

    result = ingest_drain.drain(max_batches=0)

    assert result["status"] == "checked"
    assert result["pending_after"] == 2
    assert result["batches_run"] == 0


def test_drain_forwards_single_unit_pilot_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = {"pending": 2}
    calls: list[tuple[bool, int]] = []

    monkeypatch.setattr(ingest_drain, "init_chronovisor", lambda: None)
    monkeypatch.setattr(ingest_drain.orchestrator, "reset_stale_lock", lambda: None)
    monkeypatch.setattr(
        ingest_drain.orchestrator,
        "get_pending_raw_files",
        lambda: [object()] * state["pending"],
    )

    def fake_run_pending_ingest(*, force: bool, max_units: int) -> dict:
        calls.append((force, max_units))
        state["pending"] -= 1
        return {"triggered": True, "files_processed": ["raw-0.md"]}

    monkeypatch.setattr(
        ingest_drain.orchestrator,
        "run_pending_ingest",
        fake_run_pending_ingest,
    )

    result = ingest_drain.drain(
        max_batches=1,
        max_units=1,
        sleep_seconds=0,
        log_file=tmp_path / "pilot.jsonl",
    )

    assert calls == [(True, 1)]
    assert result["status"] == "partial"
    assert result["files_processed"] == 1
    assert result["pending_after"] == 1


@pytest.mark.parametrize("max_units", [0, 11])
def test_drain_rejects_out_of_range_max_units(max_units: int) -> None:
    with pytest.raises(ValueError, match="max_units must be between 1 and 10"):
        ingest_drain.drain(max_units=max_units)


def test_parser_reads_max_units_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CHRONOVISOR_INGEST_DRAIN_MAX_UNITS", "1")

    args = ingest_drain.build_parser().parse_args([])

    assert args.max_units == 1


def test_drain_waits_for_ingest_runtime_and_recovers_without_losing_pending_raws(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = {"pending": 2, "available": False, "runs": 0}
    monkeypatch.setattr(ingest_drain, "init_chronovisor", lambda: None)
    monkeypatch.setattr(ingest_drain.orchestrator, "reset_stale_lock", lambda: None)
    monkeypatch.setattr(
        ingest_drain.orchestrator,
        "get_pending_raw_files",
        lambda: [object()] * state["pending"],
    )
    monkeypatch.setattr(ingest_drain.ollama, "is_available", lambda: state["available"])

    def run_pending(*, force: bool, max_units: int) -> dict:
        state["runs"] += 1
        state["pending"] = 0
        return {"triggered": True, "files_processed": ["a.md", "b.md"]}

    monkeypatch.setattr(ingest_drain.orchestrator, "run_pending_ingest", run_pending)
    log_file = tmp_path / "drain.jsonl"

    waiting = ingest_drain.drain(max_batches=1, sleep_seconds=0, log_file=log_file)

    assert waiting["status"] == "waiting_for_ingest_runtime"
    assert waiting["stop_reason"] == "ingest runtime unavailable"
    assert waiting["liveness"]["ingest_runtime_available"] is False
    assert waiting["pending_after"] == 2
    assert waiting["alert"] is True
    assert state["runs"] == 0

    state["available"] = True
    recovered = ingest_drain.drain(max_batches=1, sleep_seconds=0, log_file=log_file)

    assert recovered["status"] == "drained"
    assert recovered["pending_after"] == 0
    assert state["runs"] == 1
    assert recovered["liveness"]["last_recovered_at"]


def test_route_failure_reports_generic_runtime_wait_without_ollama_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ingest_drain, "init_chronovisor", lambda: None)
    monkeypatch.setattr(ingest_drain.orchestrator, "reset_stale_lock", lambda: None)
    monkeypatch.setattr(
        ingest_drain.orchestrator,
        "get_pending_raw_files",
        lambda: [object()],
    )
    monkeypatch.setattr(
        ingest_drain.ollama,
        "runtime_generation_routes",
        lambda _roles: (_ for _ in ()).throw(
            ingest_drain.ollama.RuntimeBridgeError("route_configuration_invalid")
        ),
    )
    monkeypatch.setattr(
        ingest_drain.ollama,
        "is_available",
        lambda: pytest.fail("unresolved route probed Ollama"),
    )

    result = ingest_drain._drain(max_batches=1, sleep_seconds=0)

    assert result["status"] == "waiting_for_ingest_runtime"
    assert result["stop_reason"] == "ingest runtime unavailable"
    assert "ollama" not in result["liveness"]["error"].casefold()


def test_drain_blocks_globally_before_raw_processing_when_authority_is_invalid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = {"pending": 3, "runs": 0}
    monkeypatch.setattr(ingest_drain, "init_chronovisor", lambda: None)
    monkeypatch.setattr(ingest_drain.orchestrator, "reset_stale_lock", lambda: None)
    monkeypatch.setattr(
        ingest_drain.orchestrator,
        "get_pending_raw_files",
        lambda: [object()] * state["pending"],
    )
    monkeypatch.setattr(
        ingest_drain.orchestrator,
        "ingest_authority_preflight",
        lambda: {
            "ok": False,
            "status": "blocked",
            "blocked_by": "decision_authority",
            "retryable": True,
            "error": (
                "local consensus authority unavailable: "
                "adoption_artifact_invalid:policy version mismatch"
            ),
            "artifact_sha256": None,
        },
    )
    monkeypatch.setattr(
        ingest_drain.orchestrator,
        "run_pending_ingest",
        lambda **_kwargs: state.__setitem__("runs", state["runs"] + 1),
    )

    result = ingest_drain.drain(
        max_batches=3,
        sleep_seconds=0,
        log_file=tmp_path / "authority-block.jsonl",
    )

    assert result["status"] == "blocked"
    assert result["blocked_by"] == "decision_authority"
    assert result["pending_after"] == 3
    assert result["files_processed"] == 0
    assert result["alert"] is True
    assert result["liveness"]["status"] == "blocked_by_decision_authority"
    assert state["runs"] == 0


def test_drain_reports_authority_outage_even_when_failed_raws_are_deferred(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ingest_drain, "init_chronovisor", lambda: None)
    monkeypatch.setattr(ingest_drain.orchestrator, "reset_stale_lock", lambda: None)
    monkeypatch.setattr(
        ingest_drain.orchestrator,
        "get_pending_raw_files",
        lambda: [],
    )
    monkeypatch.setattr(
        ingest_drain.orchestrator,
        "ingest_authority_preflight",
        lambda: {
            "ok": False,
            "status": "blocked",
            "blocked_by": "decision_authority",
            "retryable": True,
            "error": (
                "local consensus authority unavailable: "
                "adoption_artifact_invalid:policy version mismatch"
            ),
            "artifact_sha256": None,
        },
    )

    result = ingest_drain.drain(
        max_batches=1,
        sleep_seconds=0,
        log_file=tmp_path / "authority-empty.jsonl",
    )

    assert result["status"] == "blocked"
    assert result["pending_after"] == 0
    assert result["alert"] is True
    assert result["liveness"]["alert"] is True


def test_successful_authority_preflight_clears_sticky_runtime_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority = {
        "ok": False,
        "status": "blocked",
        "blocked_by": "decision_authority",
        "retryable": True,
        "error": "authority unavailable",
        "artifact_sha256": None,
    }
    monkeypatch.setattr(
        ingest_drain.orchestrator,
        "ingest_authority_preflight",
        lambda: dict(authority),
    )
    monkeypatch.setattr(
        ingest_drain.orchestrator,
        "get_pending_raw_files",
        lambda: [],
    )
    writes: list[dict] = []
    monkeypatch.setattr(
        ingest_drain.runtime_status,
        "safe_write_status",
        lambda **fields: writes.append(fields),
    )

    blocked = ingest_drain._drain(max_batches=1, sleep_seconds=0)
    assert blocked["status"] == "blocked"

    authority.update(
        {
            "ok": True,
            "status": "ready",
            "blocked_by": None,
            "retryable": False,
            "error": None,
            "artifact_sha256": "a" * 64,
        }
    )
    recovered = ingest_drain._drain(max_batches=1, sleep_seconds=0)

    assert recovered["status"] == "drained"
    assert writes[-1]["state"] == "idle"
    assert writes[-1]["stage"] == "idle"
    assert writes[-1]["pending"] == 0
    assert writes[-1]["mutation_ready"] is True
    assert writes[-1]["mutation_authority"]["ok"] is True
