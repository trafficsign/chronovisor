from __future__ import annotations

from datetime import datetime, timedelta, timezone
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
    now = datetime(2026, 7, 25, tzinfo=timezone.utc)

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
