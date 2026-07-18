"""Byte-exact zstd/tar archives for already-published flat Raw files."""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import tarfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterator

import zstandard as zstd

from llm_wiki_mcp.link_fix import atomic_write
from llm_wiki_mcp.raw_segment import CAPTURE_TIMEZONE, RawSegmentCorrupt


LEGACY_ARCHIVE_SCHEMA = "llm-wiki.raw-legacy-archive.v1"
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


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_raw_id(value: object) -> bool:
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


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def load_legacy_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RawSegmentCorrupt(
            f"legacy archive manifest is unreadable: {path}"
        ) from exc
    if not isinstance(payload, dict) or payload.get("schema") != LEGACY_ARCHIVE_SCHEMA:
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


def _iter_tar_members(archive_path: Path):
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


def write_legacy_archive(
    paths: list[Path],
    *,
    archive_path: Path,
    captured_date: str,
    compression_level: int = 9,
) -> dict[str, Any]:
    if not paths:
        raise ValueError("legacy archive needs at least one source")
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = archive_path.with_name(
        archive_path.name.removesuffix(".tar.zst") + ".manifest.json"
    )
    if manifest_path.exists():
        return verify_legacy_manifest(manifest_path, full=True)
    source_rows: list[tuple[Path, bytes, dict[str, Any]]] = []
    for path in sorted(paths, key=lambda item: item.name):
        if not path.is_file() or path.is_symlink() or Path(path.name).name != path.name:
            raise RawSegmentCorrupt(f"legacy source is missing or unsafe: {path}")
        value = path.read_bytes()
        source_rows.append(
            (
                path,
                value,
                {"raw_id": path.name, "bytes": len(value), "sha256": _sha256(value)},
            )
        )
    temporary = archive_path.with_name(f".{archive_path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as target:
            with zstd.ZstdCompressor(level=compression_level).stream_writer(
                target, closefd=False
            ) as compressor:
                with tarfile.open(fileobj=compressor, mode="w|") as archive:
                    for path, value, _row in source_rows:
                        info = tarfile.TarInfo(path.name)
                        info.size = len(value)
                        info.mode = 0o600
                        info.mtime = 0
                        archive.addfile(info, io.BytesIO(value))
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, archive_path)
        _fsync_directory(archive_path.parent)
        manifest = {
            "schema": LEGACY_ARCHIVE_SCHEMA,
            "archive": archive_path.name,
            "captured_date": captured_date,
            "logical_bytes": sum(row[2]["bytes"] for row in source_rows),
            "compressed_bytes": archive_path.stat().st_size,
            "compressed_sha256": _sha256_path(archive_path),
            "compression": {
                "container": "tar",
                "codec": "zstd",
                "level": compression_level,
            },
            "members": [row[2] for row in source_rows],
            "created_at": datetime.now(CAPTURE_TIMEZONE).isoformat(),
        }
        atomic_write(
            manifest_path,
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )
        _fsync_directory(archive_path.parent)
        verify_legacy_manifest(manifest_path, full=True)
        return manifest
    finally:
        temporary.unlink(missing_ok=True)


def migrate_processed_legacy(
    raw_dir: Path,
    *,
    processed_raw_ids: set[str],
    before: str | None = None,
    dry_run: bool = True,
    remove_source: bool = False,
    max_archive_bytes: int = DEFAULT_ARCHIVE_BYTES,
    compression_level: int = 9,
    before_source_delete: Callable[[Path, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Archive completed flat Raw units; shadow mode keeps every source."""

    raw_dir = raw_dir.expanduser().resolve(strict=False)
    if (
        isinstance(max_archive_bytes, bool)
        or not isinstance(max_archive_bytes, int)
        or not 0 < max_archive_bytes <= DEFAULT_ARCHIVE_BYTES
    ):
        raise ValueError(f"max_archive_bytes must be within 1..{DEFAULT_ARCHIVE_BYTES}")
    cutoff = before or datetime.now(CAPTURE_TIMEZONE).strftime("%Y/%m/%d")
    datetime.strptime(cutoff, "%Y/%m/%d")
    archived_members = {
        member.raw_id: member for member in iter_legacy_members(raw_dir)
    }
    by_date: dict[str, list[Path]] = {}
    preexisting_removals: list[Path] = []
    skipped = {
        "not_processed": 0,
        "today_or_newer": 0,
        "already_archived": 0,
        "oversized": 0,
    }
    candidates = tuple(raw_dir.glob("*.md")) + tuple(raw_dir.glob("semantic-*.json"))
    for path in sorted(candidates, key=lambda item: item.name):
        if not path.is_file() or path.is_symlink():
            continue
        if path.name not in processed_raw_ids:
            skipped["not_processed"] += 1
            continue
        captured = datetime.fromtimestamp(
            path.stat().st_mtime, tz=CAPTURE_TIMEZONE
        ).strftime("%Y/%m/%d")
        if captured >= cutoff:
            skipped["today_or_newer"] += 1
            continue
        if path.stat().st_size > max_archive_bytes:
            skipped["oversized"] += 1
            continue
        archived = archived_members.get(path.name)
        if archived is not None:
            if remove_source:
                source = path.read_bytes()
                if len(source) != archived.length or _sha256(source) != archived.sha256:
                    raise RawSegmentCorrupt(
                        f"legacy pre-delete restore mismatch: {path.name}"
                    )
                preexisting_removals.append(path)
            else:
                skipped["already_archived"] += 1
            continue
        by_date.setdefault(captured, []).append(path)

    planned: list[tuple[str, list[Path], int]] = []
    for captured, paths in sorted(by_date.items()):
        day_dir = raw_dir / captured
        existing_parts = [
            int(match.group(1))
            for path in day_dir.glob("legacy-part-*.manifest.json")
            if (
                match := re.fullmatch(r"legacy-part-(\d{3})\.manifest\.json", path.name)
            )
        ]
        part = max(existing_parts, default=0) + 1
        batch: list[Path] = []
        batch_bytes = 0
        for path in paths:
            size = path.stat().st_size
            if batch and batch_bytes + size > max_archive_bytes:
                planned.append((captured, batch, part))
                part += 1
                batch = []
                batch_bytes = 0
            batch.append(path)
            batch_bytes += size
        if batch:
            planned.append((captured, batch, part))

    results: list[dict[str, Any]] = []
    if preexisting_removals:
        results.append(
            {
                "archive": "existing_verified_archives",
                "members": len(preexisting_removals),
                "logical_bytes": sum(
                    path.stat().st_size for path in preexisting_removals
                ),
                "action": "would_remove_verified_sources"
                if dry_run
                else "removed_verified_sources",
            }
        )
        if not dry_run:
            verified_manifests: dict[Path, dict[str, Any]] = {}
            for path in preexisting_removals:
                manifest_path = archived_members[path.name].manifest_path
                if manifest_path not in verified_manifests:
                    verified_manifests[manifest_path] = verify_legacy_manifest(
                        manifest_path, full=True
                    )
            for manifest_path, manifest in sorted(verified_manifests.items()):
                if before_source_delete is not None:
                    before_source_delete(manifest_path, manifest)
            for path in preexisting_removals:
                path.unlink()
            _fsync_directory(raw_dir)
    for captured, paths, part in planned:
        archive_path = raw_dir / captured / f"legacy-part-{part:03d}.tar.zst"
        if dry_run:
            results.append(
                {
                    "archive": str(archive_path),
                    "members": len(paths),
                    "logical_bytes": sum(path.stat().st_size for path in paths),
                    "action": "would_archive",
                }
            )
            continue
        manifest = write_legacy_archive(
            paths,
            archive_path=archive_path,
            captured_date=captured,
            compression_level=compression_level,
        )
        if remove_source:
            members = {
                str(row["raw_id"]): (int(row["bytes"]), str(row["sha256"]))
                for row in manifest["members"]
            }
            for path in paths:
                source = path.read_bytes()
                evidence = members.get(path.name)
                if (
                    evidence is None
                    or len(source) != evidence[0]
                    or _sha256(source) != evidence[1]
                ):
                    raise RawSegmentCorrupt(
                        f"legacy pre-delete restore mismatch: {path.name}"
                    )
            if before_source_delete is not None:
                before_source_delete(
                    archive_path.with_name(
                        archive_path.name.removesuffix(".tar.zst") + ".manifest.json"
                    ),
                    manifest,
                )
            for path in paths:
                path.unlink()
            _fsync_directory(raw_dir)
        results.append(
            {
                "archive": str(archive_path),
                "members": len(paths),
                "logical_bytes": manifest["logical_bytes"],
                "compressed_bytes": manifest["compressed_bytes"],
                "action": "archived_and_removed"
                if remove_source
                else "shadow_archived",
            }
        )
    return {
        "status": "dry_run" if dry_run else "ok",
        "before": cutoff,
        "archives": len(planned),
        "members": sum(len(paths) for _captured, paths, _part in planned)
        + len(preexisting_removals),
        "remove_source": remove_source,
        "skipped": skipped,
        "results": results,
    }
