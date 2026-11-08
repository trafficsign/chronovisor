"""Byte-exact zstd/tar archives for already-published flat Raw files."""

from __future__ import annotations

import json
import re
import tarfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeGuard

import zstandard as zstd

from chronovisor.core.hashutil import sha256_bytes as _sha256
from chronovisor.core.hashutil import sha256_file as _sha256_path
from chronovisor.core.raw_segment import RawSegmentCorrupt
from chronovisor.core.sealed_artifact_decoder import schema_matches

LEGACY_ARCHIVE_SCHEMA = "chronovisor.raw-legacy-archive.v1"
DEFAULT_ARCHIVE_BYTES = 128 * 1024 * 1024
MAX_RAW_ID_CHARS = 240


@dataclass(frozen=True)
class LegacyArchiveMember:
    raw_id: str
    archive_path: Path
    manifest_path: Path
    length: int
    sha256: str
    captured_date: str


def _safe_raw_id(value: object) -> TypeGuard[str]:
    """Accept historical Unicode basenames without weakening path safety."""

    return (
        isinstance(value, str)
        and 0 < len(value) <= MAX_RAW_ID_CHARS
        and value not in {".", ".."}
        and not value.startswith(".")
        and "/" not in value
        and "\\" not in value
        and Path(value).name == value
        and all(ord(char) >= 32 and ord(char) != 127 for char in value)
    )


def load_legacy_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RawSegmentCorrupt(
            f"legacy archive manifest is unreadable: {path}"
        ) from exc
    if not isinstance(payload, dict) or not schema_matches(
        payload.get("schema"), LEGACY_ARCHIVE_SCHEMA
    ):
        raise RawSegmentCorrupt("legacy archive manifest schema mismatch")
    archive_name = payload.get("archive")
    members = payload.get("members")
    if (
        not isinstance(archive_name, str)
        or Path(archive_name).name != archive_name
        or not isinstance(members, list)
        or not members
        or isinstance(payload.get("logical_bytes"), bool)
        or not isinstance(payload.get("logical_bytes"), int)
        or not 0 < int(payload["logical_bytes"]) <= DEFAULT_ARCHIVE_BYTES
    ):
        raise RawSegmentCorrupt("legacy archive manifest shape is invalid")
    return payload


def iter_legacy_members(raw_dir: Path) -> Iterator[LegacyArchiveMember]:
    pattern = "[0-9][0-9][0-9][0-9]/[0-9][0-9]/[0-9][0-9]/legacy-part-*.manifest.json"
    seen: set[str] = set()
    for manifest_path in sorted(raw_dir.glob(pattern)):
        payload = load_legacy_manifest(manifest_path)
        archive_path = manifest_path.with_name(str(payload["archive"]))
        captured_date = str(payload.get("captured_date") or "")
        for row in payload["members"]:
            if not isinstance(row, dict):
                raise RawSegmentCorrupt("legacy archive member is not an object")
            raw_id = row.get("raw_id")
            length = row.get("bytes")
            sha256 = row.get("sha256")
            if (
                not _safe_raw_id(raw_id)
                or isinstance(length, bool)
                or not isinstance(length, int)
                or not 0 <= length <= DEFAULT_ARCHIVE_BYTES
                or not isinstance(sha256, str)
                or re.fullmatch(r"[0-9a-f]{64}", sha256) is None
            ):
                raise RawSegmentCorrupt("legacy archive member evidence is invalid")
            if raw_id in seen:
                raise RawSegmentCorrupt(f"duplicate archived Raw ID: {raw_id}")
            seen.add(raw_id)
            yield LegacyArchiveMember(
                raw_id=raw_id,
                archive_path=archive_path,
                manifest_path=manifest_path,
                length=length,
                sha256=sha256,
                captured_date=captured_date,
            )


def _iter_tar_members(
    archive_path: Path,
) -> Iterator[tuple[tarfile.TarFile, tarfile.TarInfo]]:
    with archive_path.open("rb") as source:
        with zstd.ZstdDecompressor(max_window_size=256 * 1024 * 1024).stream_reader(
            source
        ) as reader:
            with tarfile.open(fileobj=reader, mode="r|") as archive:
                for info in archive:
                    yield archive, info


def read_legacy_member(member: LegacyArchiveMember) -> bytes:
    if not member.archive_path.is_file() or member.archive_path.is_symlink():
        raise RawSegmentCorrupt("legacy archive object is missing or unsafe")
    manifest = verify_legacy_manifest(member.manifest_path, full=False)
    if member.archive_path.name != manifest["archive"]:
        raise RawSegmentCorrupt("legacy archive member locator is inconsistent")
    observed_logical_bytes = 0
    for archive, info in _iter_tar_members(member.archive_path):
        if not info.isfile() or Path(info.name).name != info.name:
            raise RawSegmentCorrupt("legacy archive contains an unsafe member")
        observed_logical_bytes += info.size
        if observed_logical_bytes > manifest["logical_bytes"]:
            raise RawSegmentCorrupt("legacy archive exceeds declared logical bytes")
        if info.name != member.raw_id:
            continue
        if info.size != member.length:
            raise RawSegmentCorrupt("legacy archive member size mismatch")
        extracted = archive.extractfile(info)
        if extracted is None:
            raise RawSegmentCorrupt("legacy archive member cannot be read")
        value = extracted.read(member.length + 1)
        if len(value) != member.length or _sha256(value) != member.sha256:
            raise RawSegmentCorrupt("legacy archive member digest mismatch")
        return value
    raise RawSegmentCorrupt(f"legacy archive member is missing: {member.raw_id}")


def verify_legacy_manifest(path: Path, *, full: bool = False) -> dict[str, Any]:
    payload = load_legacy_manifest(path)
    archive_path = path.with_name(str(payload["archive"]))
    if not archive_path.is_file() or archive_path.is_symlink():
        raise RawSegmentCorrupt("legacy archive is missing or unsafe")
    if archive_path.stat().st_size != payload.get("compressed_bytes"):
        raise RawSegmentCorrupt("legacy archive compressed size mismatch")
    if _sha256_path(archive_path) != payload.get("compressed_sha256"):
        raise RawSegmentCorrupt("legacy archive compressed digest mismatch")
    expected = {
        str(row["raw_id"]): (int(row["bytes"]), str(row["sha256"]))
        for row in payload["members"]
    }
    if len(expected) != len(payload["members"]):
        raise RawSegmentCorrupt("legacy archive manifest has duplicate members")
    if sum(length for length, _digest in expected.values()) != payload.get(
        "logical_bytes"
    ):
        raise RawSegmentCorrupt("legacy archive logical byte total mismatch")
    if full:
        observed: set[str] = set()
        observed_logical_bytes = 0
        for archive, info in _iter_tar_members(archive_path):
            if not info.isfile() or Path(info.name).name != info.name:
                raise RawSegmentCorrupt("legacy archive contains an unsafe member")
            observed_logical_bytes += info.size
            if observed_logical_bytes > payload["logical_bytes"]:
                raise RawSegmentCorrupt("legacy archive exceeds declared logical bytes")
            evidence = expected.get(info.name)
            if evidence is None or info.name in observed:
                raise RawSegmentCorrupt("legacy archive has an unexpected member")
            length, digest = evidence
            if info.size != length:
                raise RawSegmentCorrupt("legacy archive member size mismatch")
            extracted = archive.extractfile(info)
            if extracted is None:
                raise RawSegmentCorrupt("legacy archive member cannot be read")
            value = extracted.read(length + 1)
            if len(value) != length or _sha256(value) != digest:
                raise RawSegmentCorrupt("legacy archive member digest mismatch")
            observed.add(info.name)
        if observed != set(expected):
            raise RawSegmentCorrupt("legacy archive member set is incomplete")
        if observed_logical_bytes != payload["logical_bytes"]:
            raise RawSegmentCorrupt("legacy archive logical byte total mismatch")
    return payload
