from __future__ import annotations

import json
from pathlib import Path

from chronovisor import librarian_rollout


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_rollout_resumes_all_phases_and_starts_soak(
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
        calls.append("soak")
        result = {"status": "running", "required_days": 7}
        _write(tmp_path / "runtime" / "librarian" / "soak.json", result)
        return result

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

    result = librarian_rollout.run_rollout(tmp_path)

    assert result["status"] == "soaking"
    assert calls == [
        "adjudicate:20",
        "lock",
        "phase0",
        "distribution",
        "calibrate",
        "shadow",
        "migration",
        "burn",
        "pilot",
        "full",
        "soak",
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

    result = librarian_rollout.run_rollout(tmp_path)

    assert result["status"] == "blocked"
    assert result["stage"] == "quality_gate"
