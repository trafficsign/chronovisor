import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from chronovisor.recall import librarian_release

NOW = datetime(2026, 7, 25, 9, 0, tzinfo=UTC)


@pytest.fixture(autouse=True)
def _valid_okf_root(tmp_path: Path) -> None:
    for name in ("index.md", "log.md", "schema.md"):
        (tmp_path / name).write_text("legacy\n", encoding="utf-8")


def test_phase0_main_blocks_before_mutation_on_unsafe_root(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "blocked"
    root.mkdir()
    (root / "private.txt").write_text("canary", encoding="utf-8")

    assert librarian_release.main(["phase0", "--root", str(root)]) == 75
    assert json.loads(capsys.readouterr().out) == {
        "status": "blocked",
        "category": "okf_startup_blocked",
    }
    assert [path.name for path in root.iterdir()] == ["private.txt"]


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


def test_migration_observation_starts_before_release_prerequisites(
    tmp_path: Path,
) -> None:
    observation = librarian_release.start_soak(tmp_path, now=NOW)

    assert observation["status"] == "running"
    assert observation["observation_mode"] == "concurrent_migration"
    assert observation["wall_clock_required_seconds"] == 0
    advanced = librarian_release.advance_migration_observation(
        tmp_path,
        stage="phase5_full_shadow_complete",
        now=NOW,
    )
    repeated = librarian_release.advance_migration_observation(
        tmp_path,
        stage="phase5_full_shadow_complete",
        now=NOW,
    )
    assert advanced["observed_through"] == "phase5_full_shadow_complete"
    assert len(repeated["checkpoints"]) == 2
    waiting = librarian_release.finalize_if_ready(tmp_path, now=NOW)
    assert waiting["status"] == "observing"


def test_release_uses_migration_evidence_without_post_migration_delay(
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

    observation = librarian_release.start_soak(tmp_path, now=NOW)
    assert observation["status"] == "running"

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
        now=NOW,
    )

    assert released["status"] == "released"
    assert released["observation"]["observation_mode"] == "concurrent_migration"
    assert released["observation"]["observed_through"] == "phase12_postflight"
    assert released["cleanup"]["restore_points"]["deleted"] == ["restore"]
    assert released["cleanup"]["transaction_preimages"]["deleted"] == ["preimage"]


def test_release_recovers_completed_observation_without_receipt(
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
        tmp_path / "runtime" / "librarian" / "soak.json",
        {
            "schema": librarian_release.SOAK_SCHEMA,
            "status": "complete",
            "observation_mode": "concurrent_migration",
            "starts_at": NOW.isoformat(),
        },
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
    monkeypatch.setattr(
        librarian_release,
        "_restore_all",
        lambda _root: [{"status": "verified", "restore_id": "test"}],
    )
    monkeypatch.setattr(
        librarian_release,
        "cleanup_expired_restore_points",
        lambda *args, **kwargs: {"deleted": [], "retained": []},
    )
    monkeypatch.setattr(
        librarian_release,
        "cleanup_expired_preimages",
        lambda *args, **kwargs: {"deleted": [], "retained": []},
    )

    released = librarian_release.finalize_if_ready(tmp_path, now=NOW)

    assert released["status"] == "released"
    assert (
        tmp_path / "runtime" / "librarian" / "phase12-release.json"
    ).is_file()
