from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from chronovisor.core.legacy_archive import (
    iter_legacy_members,
    migrate_processed_legacy,
    read_legacy_member,
    verify_legacy_manifest,
)
from chronovisor.core.raw_segment import RawSegmentCorrupt
from chronovisor.core.raw_store import RawStore
from chronovisor.raw.raw_archive import archive_status


def _legacy_files(raw_dir: Path) -> tuple[Path, Path]:
    raw_dir.mkdir(parents=True)
    first = raw_dir / "save-old-one.md"
    second = raw_dir / "save-old-日本語-two.md"
    first.write_bytes(b"first byte-exact raw\n")
    second.write_bytes("second 日本語 raw\n".encode())
    old = datetime(2026, 7, 16, 12, tzinfo=ZoneInfo("Asia/Tokyo")).timestamp()
    os.utime(first, (old, old))
    os.utime(second, (old, old))
    return first, second


def test_legacy_shadow_archive_round_trips_and_flat_file_keeps_precedence(
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "raw"
    first, second = _legacy_files(raw_dir)
    original = {first.name: first.read_bytes(), second.name: second.read_bytes()}
    processed = set(original)

    preview = migrate_processed_legacy(
        raw_dir,
        processed_raw_ids=processed,
        before="2026/07/18",
        dry_run=True,
        max_archive_bytes=1024,
    )
    assert preview["members"] == 2
    assert list(raw_dir.rglob("*.tar.zst")) == []

    result = migrate_processed_legacy(
        raw_dir,
        processed_raw_ids=processed,
        before="2026/07/18",
        dry_run=False,
        max_archive_bytes=1024,
    )
    assert result["members"] == 2
    manifest = next(raw_dir.rglob("legacy-part-*.manifest.json"))
    assert verify_legacy_manifest(manifest, full=True)["logical_bytes"] == sum(
        map(len, original.values())
    )
    members = {member.raw_id: member for member in iter_legacy_members(raw_dir)}
    assert {
        raw_id: read_legacy_member(member) for raw_id, member in members.items()
    } == original
    assert (
        RawStore(raw_dir, mode="legacy").read_bytes(first.name) == original[first.name]
    )
    status = archive_status(raw_dir)
    assert status["logical_bytes"] == sum(map(len, original.values()))
    assert status["stored_bytes"] > status["logical_bytes"]


def test_verified_shadow_can_remove_flat_sources_on_later_run(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    first, second = _legacy_files(raw_dir)
    processed = {first.name, second.name}
    migrate_processed_legacy(
        raw_dir,
        processed_raw_ids=processed,
        before="2026/07/18",
        dry_run=False,
    )

    relocation_commits: list[Path] = []
    result = migrate_processed_legacy(
        raw_dir,
        processed_raw_ids=processed,
        before="2026/07/18",
        dry_run=False,
        remove_source=True,
        before_source_delete=lambda path, _manifest: relocation_commits.append(path),
    )

    assert result["members"] == 2
    assert not first.exists() and not second.exists()
    store = RawStore(raw_dir)
    assert store.read_bytes(first.name) == b"first byte-exact raw\n"
    unit = store.resolve(first.name)
    assert unit is not None
    reference = store.materialize_ingest(unit, tmp_path / "runtime" / "parents")
    assert store.resolve_reference(reference) == unit
    assert len(relocation_commits) == 1


def test_legacy_archive_corruption_fails_closed(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    first, _second = _legacy_files(raw_dir)
    migrate_processed_legacy(
        raw_dir,
        processed_raw_ids={first.name},
        before="2026/07/18",
        dry_run=False,
    )
    archive = next(raw_dir.rglob("*.tar.zst"))
    damaged = bytearray(archive.read_bytes())
    damaged[len(damaged) // 2] ^= 1
    archive.write_bytes(damaged)

    with pytest.raises(RawSegmentCorrupt, match="digest"):
        verify_legacy_manifest(next(raw_dir.rglob("*.manifest.json")), full=True)


def test_legacy_migration_leaves_oversized_member_unmodified(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    source = raw_dir / "save-too-large.md"
    source.write_bytes(b"0123456789")
    old = datetime(2026, 7, 16, 12, tzinfo=ZoneInfo("Asia/Tokyo")).timestamp()
    os.utime(source, (old, old))

    result = migrate_processed_legacy(
        raw_dir,
        processed_raw_ids={source.name},
        before="2026/07/18",
        dry_run=False,
        max_archive_bytes=5,
    )

    assert result["members"] == 0
    assert result["skipped"]["oversized"] == 1
    assert source.read_bytes() == b"0123456789"
    assert not list(raw_dir.rglob("*.tar.zst"))
