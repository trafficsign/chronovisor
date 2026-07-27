from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from chronovisor.migration_snapshot import (
    cleanup_expired_restore_points,
    create_restore_point,
    restore_drill,
    verify_restore_point,
)


def test_restore_point_isolated_verified_and_expires(tmp_path: Path) -> None:
    root = tmp_path / "wiki"
    page = root / "pages" / "alpha.md"
    page.parent.mkdir(parents=True)
    page.write_text("---\ntitle: Alpha\n---\n\nBody\n", encoding="utf-8")
    lock = root / ".index" / "semantic" / "activation.lock"
    lock.parent.mkdir(parents=True)
    lock.touch()
    now = datetime(2026, 7, 25, tzinfo=UTC)

    created = create_restore_point(root, reason="phase-6", ttl_days=7, now=now)
    restore_path = Path(created["path"])
    assert (
        root / "runtime" / "librarian" / "migration-restore-points"
        in restore_path.parents
    )
    assert verify_restore_point(restore_path)["status"] == "verified"

    destination = tmp_path / "drill"
    assert restore_drill(restore_path, destination)["status"] == "verified"
    assert (destination / "pages" / "alpha.md").read_bytes() == page.read_bytes()

    cleanup = cleanup_expired_restore_points(root, now=now + timedelta(days=8))
    assert created["restore_id"] in cleanup["deleted"]
    assert not restore_path.exists()


def test_phase6_restore_insurance_survives_ttl_until_release(
    tmp_path: Path,
) -> None:
    root = tmp_path / "wiki"
    page = root / "pages" / "alpha.md"
    page.parent.mkdir(parents=True)
    page.write_text("body\n", encoding="utf-8")
    now = datetime(2026, 7, 25, tzinfo=UTC)
    created = create_restore_point(root, reason="phase-6", ttl_days=7, now=now)
    receipt = root / "runtime" / "librarian" / "phase6-receipt.json"
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text('{"status":"ok"}', encoding="utf-8")

    retained = cleanup_expired_restore_points(
        root,
        now=now + timedelta(days=8),
    )
    forced = cleanup_expired_restore_points(
        root,
        now=now + timedelta(days=8),
        force=True,
    )

    assert retained["reason"] == "migration_release_insurance_active"
    assert created["restore_id"] in retained["retained"]
    assert created["restore_id"] in forced["deleted"]


def test_verified_release_force_removes_unexpired_restore_point(
    tmp_path: Path,
) -> None:
    root = tmp_path / "wiki"
    page = root / "pages" / "alpha.md"
    page.parent.mkdir(parents=True)
    page.write_text("body\n", encoding="utf-8")
    now = datetime(2026, 7, 25, tzinfo=UTC)
    created = create_restore_point(root, reason="phase-6", ttl_days=7, now=now)

    cleanup = cleanup_expired_restore_points(root, now=now, force=True)

    assert created["restore_id"] in cleanup["deleted"]
    assert cleanup["retained"] == []
