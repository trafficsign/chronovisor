"""Keep historical archive readers after removing archive migration writers."""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest
import zstandard

from chronovisor.raw.legacy_archive import (
    iter_legacy_members,
    read_legacy_member,
    verify_legacy_manifest,
)
from chronovisor.raw.raw_segment import RawSegmentCorrupt
from chronovisor.raw.raw_store import RawStore


def archive_fixture(raw_dir: Path) -> tuple[Path, Path, dict[str, bytes]]:
    values = {
        "save-old-one.md": b"first byte-exact raw\n",
        "save-old-日本語-two.md": "second 日本語 raw\n".encode(),
    }
    folder = raw_dir / "2026" / "07" / "16"
    folder.mkdir(parents=True)
    tar = io.BytesIO()
    with tarfile.open(fileobj=tar, mode="w") as handle:
        for name, content in values.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            handle.addfile(info, io.BytesIO(content))
    compressed = zstandard.ZstdCompressor().compress(tar.getvalue())
    archive = folder / "legacy-part-000001.tar.zst"
    archive.write_bytes(compressed)
    manifest = folder / "legacy-part-000001.manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "chronovisor.raw-legacy-archive.v1",
                "archive": archive.name,
                "captured_date": "2026/07/16",
                "logical_bytes": sum(map(len, values.values())),
                "compressed_bytes": len(compressed),
                "compressed_sha256": hashlib.sha256(compressed).hexdigest(),
                "members": [
                    {
                        "raw_id": name,
                        "bytes": len(content),
                        "sha256": hashlib.sha256(content).hexdigest(),
                    }
                    for name, content in values.items()
                ],
            }
        )
    )
    return archive, manifest, values


def test_legacy_archive_byte_exact_reads_without_flat_sources(tmp_path):
    _, manifest, values = archive_fixture(tmp_path)
    assert verify_legacy_manifest(manifest, full=True)["logical_bytes"] == sum(
        map(len, values.values())
    )
    assert {
        member.raw_id: read_legacy_member(member)
        for member in iter_legacy_members(tmp_path)
    } == values
    store = RawStore(tmp_path)
    for name, value in values.items():
        assert store.read_bytes(name) == value
        assert store.resolve(name).storage == "legacy_archive"


def test_flat_file_keeps_precedence_over_legacy_archive(tmp_path):
    _, _, values = archive_fixture(tmp_path)
    name = next(iter(values))
    (tmp_path / name).write_bytes(b"new flat bytes")
    assert RawStore(tmp_path, mode="legacy").read_bytes(name) == b"new flat bytes"


def test_legacy_archive_corruption_fails_closed(tmp_path):
    archive, manifest, _ = archive_fixture(tmp_path)
    damaged = bytearray(archive.read_bytes())
    damaged[len(damaged) // 2] ^= 1
    archive.write_bytes(damaged)
    with pytest.raises(RawSegmentCorrupt, match="digest"):
        verify_legacy_manifest(manifest, full=True)
