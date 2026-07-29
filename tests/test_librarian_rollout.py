from __future__ import annotations

import json
from pathlib import Path

from chronovisor.librarian import collection_authority
from chronovisor.librarian import librarian_rollout
from chronovisor.core.durable_state import read_sealed_json, write_sealed_json


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_rollout_uses_collection_phase4_and_skips_legacy_classifier(
    tmp_path: Path,
    monkeypatch,
) -> None:
    write_sealed_json(
        tmp_path
        / "runtime"
        / "librarian"
        / "phase4-collection-authority.json",
        {"status": "adopted"},
        backup=False,
    )
    monkeypatch.setattr(
        collection_authority,
        "run_collection_librarian",
        lambda *_args, **_kwargs: {
            "status": "ok",
            "sync": {
                "assignment_count": 12,
                "collection_count": 3,
                "page_registry_mirror": {"generation": 7},
            },
            "quality": {
                "status": "passed",
                "warnings": ["top_collection_share"],
                "hard_failures": [],
            },
            "queue": {"open": 2},
        },
    )
    monkeypatch.setattr(
        collection_authority,
        "collection_authority_status",
        lambda _root: {"active": True, "mode": "collection-first"},
    )
    monkeypatch.setattr(
        librarian_rollout,
        "adjudicate",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("legacy classification must not run")
        ),
    )

    result = librarian_rollout.run_rollout(tmp_path)

    assert result["status"] == "observing"
    assert result["stage"] == "collection_review_queue"
    assert (
        read_sealed_json(
            tmp_path / "runtime" / "librarian" / "phase5-receipt.json"
        )["method"]
        == "collection-first-authority"
    )
    assert (
        read_sealed_json(
            tmp_path / "runtime" / "librarian" / "phase6-receipt.json"
        )["page_mutations"]
        == 0
    )


def test_collection_rollout_advances_when_review_queue_is_empty(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: list[str] = []
    write_sealed_json(
        tmp_path
        / "runtime"
        / "librarian"
        / "phase4-collection-authority.json",
        {"status": "adopted"},
        backup=False,
    )
    monkeypatch.setattr(
        collection_authority,
        "run_collection_librarian",
        lambda *_args, **_kwargs: (
            calls.append("collection_sync")
            or {
                "status": "ok",
                "sync": {
                    "assignment_count": 12,
                    "collection_count": 3,
                    "page_registry_mirror": {"generation": 7},
                },
                "quality": {
                    "status": "passed",
                    "warnings": [],
                    "hard_failures": [],
                },
                "queue": {"open": 0},
            }
        ),
    )
    monkeypatch.setattr(
        collection_authority,
        "collection_authority_status",
        lambda _root: {"active": True, "mode": "collection-first"},
    )
    _write(
        tmp_path / "runtime" / "librarian" / "phase7-burn.json",
        {"status": "passed"},
    )
    _write(
        tmp_path / "runtime" / "librarian" / "phase10-pilot.json",
        {"status": "ok"},
    )
    _write(
        tmp_path / "runtime" / "librarian" / "phase11-receipt.json",
        {"status": "ok"},
    )
    monkeypatch.setattr(
        librarian_rollout,
        "start_soak",
        lambda _root: calls.append("observation") or {"status": "running"},
    )
    monkeypatch.setattr(
        librarian_rollout,
        "advance_migration_observation",
        lambda _root, *, stage: calls.append(f"observe:{stage}") or {},
    )
    monkeypatch.setattr(
        librarian_rollout,
        "reconcile_librarian_state",
        lambda *_args, **_kwargs: calls.append("reconcile") or {"status": "ok"},
    )
    monkeypatch.setattr(
        librarian_rollout,
        "finalize_if_ready",
        lambda _root: calls.append("release") or {"status": "released"},
    )

    result = librarian_rollout.run_rollout(tmp_path)

    assert result["status"] == "released"
    assert result["stage"] == "complete"
    assert calls == [
        "collection_sync",
        "observation",
        "observe:phase5_collection_shadow_complete",
        "observe:phase6_collection_authority_complete",
        "observe:phase7_preemption_burn_complete",
        "observe:phase10_pilot_complete",
        "observe:phase11_full_migration_complete",
        "collection_sync",
        "reconcile",
        "release",
    ]


def test_released_collection_rollout_stays_complete(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _write(
        tmp_path
        / "runtime"
        / "librarian"
        / "phase4-collection-authority.json",
        {"status": "adopted"},
    )
    _write(
        tmp_path / "runtime" / "librarian" / "phase12-release.json",
        {"status": "released", "released_at": "2026-07-27T00:00:00+00:00"},
    )
    monkeypatch.setattr(
        collection_authority,
        "run_collection_librarian",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("released rollout must not restart collection sync")
        ),
    )

    result = librarian_rollout.run_rollout(tmp_path)

    assert result["status"] == "released"
    assert result["stage"] == "complete"
    assert result["detail"]["released_at"] == "2026-07-27T00:00:00+00:00"


def test_rollout_observes_migration_and_releases_after_evidence(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: list[str] = []
    fixture_root = tmp_path / "classification" / "fixtures"

    def adjudicate(_root, *, batch_size):
        calls.append(f"adjudicate:{batch_size}")
        path = fixture_root / "classification-adjudication-300.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")
        return {"status": "adjudicated"}

    def lock(_root):
        calls.append("lock")
        _write(fixture_root / "manifest.json", {"status": "locked"})
        return {"status": "locked"}

    def phase0(_root):
        calls.append("phase0")
        _write(
            tmp_path / "runtime" / "librarian" / "phase0-receipt.json",
            {"status": "ok"},
        )
        return {"status": "ok"}

    def distribution(_root):
        calls.append("distribution")
        _write(
            tmp_path / "classification" / "distribution-analysis.json",
            {"status": "ok"},
        )
        return {"status": "ok"}

    def calibrate(_root):
        calls.append("calibrate")
        result = {"status": "adopted"}
        _write(tmp_path / "classification" / "calibration.json", result)
        return result

    def shadow(_root, **kwargs):
        calls.append("shadow")
        result = {"status": "ok", "remaining": 0}
        _write(
            tmp_path / "runtime" / "librarian" / "phase5-receipt.json",
            result,
        )
        return result

    def migration(_root, **kwargs):
        calls.append("migration")
        result = {"status": "ok"}
        _write(
            tmp_path / "runtime" / "librarian" / "phase6-receipt.json",
            result,
        )
        return result

    def burn(_root, **kwargs):
        calls.append("burn")
        result = {"status": "passed"}
        _write(
            tmp_path / "runtime" / "librarian" / "phase7-burn.json",
            result,
        )
        return result

    def merge(_root, *, pilot_limit):
        calls.append("pilot" if pilot_limit is not None else "full")
        result = {"status": "ok"}
        filename = (
            "phase10-pilot.json"
            if pilot_limit is not None
            else "phase11-receipt.json"
        )
        _write(tmp_path / "runtime" / "librarian" / filename, result)
        return result

    def soak(_root, **kwargs):
        calls.append("observation")
        result = {
            "status": "running",
            "observation_mode": "concurrent_migration",
        }
        _write(tmp_path / "runtime" / "librarian" / "soak.json", result)
        return result

    def finalize(_root, **kwargs):
        calls.append("release")
        return {"status": "released"}

    def advance(_root, *, stage, **kwargs):
        calls.append(f"observe:{stage}")
        return {"status": "running", "observed_through": stage}

    monkeypatch.setattr(librarian_rollout, "adjudicate", adjudicate)
    monkeypatch.setattr(
        librarian_rollout,
        "adjudication_path",
        lambda _root: fixture_root / "classification-adjudication-300.jsonl",
    )
    monkeypatch.setattr(
        librarian_rollout,
        "fixture_paths",
        lambda _root: (
            fixture_root / "dev.jsonl",
            fixture_root / "holdout.jsonl",
            fixture_root / "manifest.json",
        ),
    )
    monkeypatch.setattr(librarian_rollout, "lock", lock)
    monkeypatch.setattr(librarian_rollout, "capture_phase0_artifacts", phase0)
    monkeypatch.setattr(librarian_rollout, "distribution", distribution)
    monkeypatch.setattr(librarian_rollout, "calibrate", calibrate)
    monkeypatch.setattr(librarian_rollout, "run_full_model_shadow", shadow)
    monkeypatch.setattr(librarian_rollout, "migrate_active_metadata", migration)
    monkeypatch.setattr(librarian_rollout, "run_burn", burn)
    monkeypatch.setattr(librarian_rollout, "run_merge_migration", merge)
    monkeypatch.setattr(
        librarian_rollout,
        "reconcile_librarian_state",
        lambda *args, **kwargs: {"status": "ok"},
    )
    monkeypatch.setattr(librarian_rollout, "start_soak", soak)
    monkeypatch.setattr(
        librarian_rollout,
        "advance_migration_observation",
        advance,
    )
    monkeypatch.setattr(librarian_rollout, "finalize_if_ready", finalize)

    result = librarian_rollout.run_rollout(tmp_path)

    assert result["status"] == "released"
    assert calls == [
        "adjudicate:20",
        "lock",
        "phase0",
        "distribution",
        "calibrate",
        "observation",
        "shadow",
        "observe:phase5_full_shadow_complete",
        "migration",
        "observe:phase6_active_metadata_complete",
        "burn",
        "observe:phase7_preemption_burn_complete",
        "pilot",
        "observe:phase10_pilot_complete",
        "full",
        "observe:phase11_full_migration_complete",
        "release",
    ]


def test_rollout_stops_at_rejected_calibration(
    tmp_path: Path,
    monkeypatch,
) -> None:
    fixture_root = tmp_path / "classification" / "fixtures"
    fixture_root.mkdir(parents=True)
    (fixture_root / "classification-adjudication-300.jsonl").write_text(
        "{}\n",
        encoding="utf-8",
    )
    _write(fixture_root / "manifest.json", {"status": "locked"})
    _write(
        tmp_path / "runtime" / "librarian" / "phase0-receipt.json",
        {"status": "ok"},
    )
    _write(
        tmp_path / "classification" / "distribution-analysis.json",
        {"status": "ok"},
    )
    _write(
        tmp_path / "classification" / "calibration.json",
        {
            "status": "rejected",
            "input_fingerprint": {"dev_fixture_sha256": "sha256:current"},
        },
    )
    monkeypatch.setattr(
        librarian_rollout,
        "calibration_input_fingerprint",
        lambda _root: {"dev_fixture_sha256": "sha256:current"},
    )
    monkeypatch.setattr(
        librarian_rollout,
        "calibrate",
        lambda _root: (_ for _ in ()).throw(
            AssertionError("unchanged rejected calibration must not rerun")
        ),
    )
    monkeypatch.setattr(
        librarian_rollout,
        "adjudication_path",
        lambda _root: fixture_root / "classification-adjudication-300.jsonl",
    )
    monkeypatch.setattr(
        librarian_rollout,
        "fixture_paths",
        lambda _root: (
            fixture_root / "dev.jsonl",
            fixture_root / "holdout.jsonl",
            fixture_root / "manifest.json",
        ),
    )

    result = librarian_rollout.run_rollout(tmp_path)

    assert result["status"] == "blocked"
    assert result["stage"] == "quality_gate"


def test_rollout_retries_rejected_calibration_after_fixture_audit(
    tmp_path: Path,
    monkeypatch,
) -> None:
    fixture_root = tmp_path / "classification" / "fixtures"
    fixture_root.mkdir(parents=True)
    (fixture_root / "classification-adjudication-300.jsonl").write_text(
        "{}\n",
        encoding="utf-8",
    )
    _write(
        fixture_root / "manifest.json",
        {"status": "locked", "holdout": {"opened_at": None}},
    )
    _write(
        tmp_path / "runtime" / "librarian" / "phase0-receipt.json",
        {"status": "ok"},
    )
    _write(
        tmp_path / "classification" / "distribution-analysis.json",
        {"status": "ok"},
    )
    _write(
        tmp_path / "classification" / "calibration.json",
        {
            "status": "rejected",
            "input_fingerprint": {"dev_fixture_sha256": "sha256:old"},
        },
    )
    monkeypatch.setattr(
        librarian_rollout,
        "adjudication_path",
        lambda _root: fixture_root / "classification-adjudication-300.jsonl",
    )
    monkeypatch.setattr(
        librarian_rollout,
        "fixture_paths",
        lambda _root: (
            fixture_root / "dev.jsonl",
            fixture_root / "holdout.jsonl",
            fixture_root / "manifest.json",
        ),
    )
    current = {"dev_fixture_sha256": "sha256:audited"}
    monkeypatch.setattr(
        librarian_rollout,
        "calibration_input_fingerprint",
        lambda _root: current,
    )
    calls: list[str] = []

    def recalibrate(_root):
        calls.append("calibrate")
        return {
            "status": "rejected",
            "input_fingerprint": current,
        }

    monkeypatch.setattr(librarian_rollout, "calibrate", recalibrate)

    result = librarian_rollout.run_rollout(tmp_path)

    assert calls == ["calibrate"]
    assert result["status"] == "blocked"
    assert result["stage"] == "quality_gate"


def test_rollout_resumes_incomplete_opened_holdout(
    tmp_path: Path,
    monkeypatch,
) -> None:
    fixture_root = tmp_path / "classification" / "fixtures"
    fixture_root.mkdir(parents=True)
    (fixture_root / "classification-adjudication-300.jsonl").write_text(
        "{}\n",
        encoding="utf-8",
    )
    _write(
        fixture_root / "manifest.json",
        {
            "status": "locked",
            "holdout": {"opened_at": "2026-07-26T07:11:35+00:00"},
        },
    )
    _write(
        tmp_path / "runtime" / "librarian" / "phase0-receipt.json",
        {"status": "ok"},
    )
    _write(
        tmp_path / "classification" / "distribution-analysis.json",
        {"status": "ok"},
    )
    _write(
        tmp_path / "classification" / "calibration.json",
        {"status": "rejected"},
    )
    monkeypatch.setattr(
        librarian_rollout,
        "adjudication_path",
        lambda _root: fixture_root / "classification-adjudication-300.jsonl",
    )
    monkeypatch.setattr(
        librarian_rollout,
        "fixture_paths",
        lambda _root: (
            fixture_root / "dev.jsonl",
            fixture_root / "holdout.jsonl",
            fixture_root / "manifest.json",
        ),
    )
    calls: list[str] = []

    def resume(_root):
        calls.append("calibrate")
        return {"status": "rejected"}

    monkeypatch.setattr(librarian_rollout, "calibrate", resume)

    result = librarian_rollout.run_rollout(tmp_path)

    assert calls == ["calibrate"]
    assert result["status"] == "blocked"
    assert result["stage"] == "quality_gate"
