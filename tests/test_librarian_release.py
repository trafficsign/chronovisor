import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from chronovisor import librarian_release

NOW = datetime(2026, 7, 25, 9, 0, tzinfo=UTC)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _release_state() -> dict:
    return {
        "authority": {"active": True},
        "progress": {
            "full_sweep": {"current": True},
            "classification_terminal": {"numerator": 1, "denominator": 1},
            "migration_batch": {"numerator": 1, "denominator": 1},
        },
        "initial_organization_complete_at": None,
    }


def test_soak_requires_every_release_prerequisite(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="release prerequisites"):
        librarian_release.start_soak(tmp_path, now=NOW)


def test_seven_day_soak_is_not_bypassable_and_timewarp_cleanup_is_tested(
    tmp_path: Path,
    monkeypatch,
) -> None:
    statuses = {
        "phase0-receipt.json": "ok",
        "phase1-receipt.json": "verified",
        "phase3-receipt.json": "verified",
        "phase5-receipt.json": "ok",
        "phase6-receipt.json": "ok",
        "phase7-burn.json": "passed",
        "phase10-pilot.json": "ok",
        "phase11-receipt.json": "ok",
    }
    for filename, status in statuses.items():
        _write_json(
            tmp_path / "runtime" / "librarian" / filename,
            {"status": status},
        )
    _write_json(
        tmp_path / "classification" / "calibration.json",
        {"status": "adopted"},
    )
    monkeypatch.setattr(
        librarian_release,
        "reconcile_librarian_state",
        lambda *args, **kwargs: _release_state(),
    )

    soak = librarian_release.start_soak(tmp_path, days=7, now=NOW)
    assert soak["status"] == "running"
    assert librarian_release.finalize_if_ready(
        tmp_path,
        now=NOW + timedelta(days=1),
    )["status"] == "running"
    with pytest.raises(RuntimeError, match="still running"):
        librarian_release.finalize_release(
            tmp_path,
            now=NOW + timedelta(days=6, hours=23),
        )

    monkeypatch.setattr(
        librarian_release,
        "_restore_all",
        lambda _root: [{"status": "verified", "restore_id": "test"}],
    )
    monkeypatch.setattr(
        librarian_release,
        "cleanup_expired_restore_points",
        lambda *args, **kwargs: {"deleted": ["restore"], "retained": []},
    )
    monkeypatch.setattr(
        librarian_release,
        "cleanup_expired_preimages",
        lambda *args, **kwargs: {"deleted": ["preimage"], "retained": []},
    )

    released = librarian_release.finalize_if_ready(
        tmp_path,
        now=NOW + timedelta(days=7),
    )

    assert released["status"] == "released"
    assert released["cleanup"]["restore_points"]["deleted"] == ["restore"]
    assert released["cleanup"]["transaction_preimages"]["deleted"] == ["preimage"]
