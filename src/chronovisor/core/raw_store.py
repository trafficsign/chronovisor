"""Logical access to legacy Raw files and Raw Archive v2 segments.

Raw IDs are stable public identities.  Physical paths are implementation
details, so readers can move from flat ``*.md`` files to date-partitioned,
compressed transcript segments without changing queues, ledgers, or replay
markers.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import threading
import tomllib
from collections import OrderedDict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

from chronovisor.core.canonical_json import canonical_json_sha256_strict
from chronovisor.core.raw_segment import (
    RawSegmentCommit,
    RawSegmentCorrupt,
    manifest_commits,
    read_commits,
    read_open_range,
    read_sealed_range,
)
from chronovisor.core.sealed_artifact_decoder import schema_matches

RawLayoutMode = Literal["legacy", "shadow", "v2"]
RawStorageKind = Literal[
    "legacy_file", "legacy_archive", "segment_open", "segment_sealed"
]
RAW_REFERENCE_SCHEMA = "chronovisor.raw-reference.v1"
_LEGACY_ARCHIVE_BASENAME_RE = re.compile(r"^legacy-part-[0-9]{3}\.tar\.zst$")
_SEGMENT_SNAPSHOT_CACHE_SIZE = 8


def committed_event_spans(
    raw: bytes, record_count: int
) -> tuple[tuple[int, bytes], ...]:
    """Return byte-exact JSON event spans from one committed Raw v2 unit."""

    if not raw.endswith(b"\n"):
        raise RawSegmentCorrupt("committed Raw record count is invalid")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RawSegmentCorrupt("committed Raw event stream is invalid UTF-8") from exc
    decoder = json.JSONDecoder()
    spans: list[tuple[int, bytes]] = []
    cursor = 0
    byte_cursor = 0
    while cursor < len(text):
        try:
            event, end = decoder.raw_decode(text, cursor)
        except json.JSONDecodeError as exc:
            raise RawSegmentCorrupt(
                "committed Raw event stream is invalid JSON"
            ) from exc
        if not isinstance(event, dict):
            raise RawSegmentCorrupt("committed Raw event must be an object")
        next_cursor = end
        while next_cursor < len(text) and text[next_cursor] in " \t\r\n":
            next_cursor += 1
        if "\n" not in text[end:next_cursor]:
            raise RawSegmentCorrupt(
                "committed Raw event stream has no record separator"
            )
        encoded = text[cursor:next_cursor].encode("utf-8")
        spans.append((byte_cursor, encoded))
        byte_cursor += len(encoded)
        cursor = next_cursor
    if len(spans) != record_count or byte_cursor != len(raw):
        raise RawSegmentCorrupt("committed Raw record count is invalid")
    return tuple(spans)


def raw_layout_mode(
    value: str | None = None, *, chronovisor_root: Path | None = None
) -> RawLayoutMode:
    """Resolve one durable storage mode for every Wiki process.

    An explicit argument is useful for offline tools and tests.  The
    environment variable remains the emergency/operator override.  Normal
    production processes converge on ``[raw].layout`` in the Wiki root so a
    Stop hook, MCP server, ingest worker, and dashboard cannot silently use
    different layouts.
    """

    configured: object = None
    if (
        value is None
        and os.environ.get("CHRONOVISOR_RAW_LAYOUT") is None
        and chronovisor_root
    ):
        config_path = chronovisor_root.expanduser() / "config.toml"
        try:
            payload = tomllib.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError):
            payload = {}
        raw_config = payload.get("raw") if isinstance(payload, dict) else None
        if isinstance(raw_config, dict):
            configured = raw_config.get("layout")
    selected_value = value or os.environ.get("CHRONOVISOR_RAW_LAYOUT") or configured
    selected = str(selected_value or "v2").strip().lower()
    if selected not in {"legacy", "shadow", "v2"}:
        raise ValueError("CHRONOVISOR_RAW_LAYOUT must be legacy, shadow, or v2")
    return selected  # type: ignore[return-value]


@dataclass(frozen=True)
class RawUnit:
    raw_id: str
    storage: RawStorageKind
    path: Path
    offset: int
    length: int
    sha256: str | None
    captured_at: str | None
    commit: RawSegmentCommit | None = None
    archive_member: object | None = None
    device: int | None = None
    inode: int | None = None

    @property
    def is_segment(self) -> bool:
        return self.storage in {"segment_open", "segment_sealed"}


class RawStore:
    """Dual-read store with deterministic identity precedence.

    ``legacy`` and ``shadow`` prefer a legacy file when both layouts contain
    the same logical Raw ID.  ``v2`` prefers a segment commit.  This makes the
    feature flag reversible while shadow comparison is running.
    """

    # Segment commits are immutable after their journal/manifest is published.
    # Reuse that parsed snapshot across short-lived stores, but only while the
    # authoritative index files retain the same lstat identities.
    _segment_snapshots: OrderedDict[
        Path, tuple[tuple[tuple[str, int, int, int, int], ...], tuple[RawUnit, ...]]
    ] = OrderedDict()
    _segment_snapshots_lock = threading.Lock()

    def __init__(self, raw_dir: Path, *, mode: RawLayoutMode | str | None = None):
        self.raw_dir = raw_dir.expanduser().resolve(strict=False)
        self.mode = raw_layout_mode(mode, chronovisor_root=self.raw_dir.parent)
        self._units_cache: tuple[RawUnit, ...] | None = None
        self._units_by_id: dict[str, RawUnit] | None = None
        self._segment_units_cache: tuple[RawUnit, ...] | None = None
        self._legacy_archive_units_by_id: dict[str, RawUnit] | None = None
        self._verified_legacy_manifests: dict[Path, tuple[int, ...]] = {}

    def _legacy_units(self) -> Iterator[RawUnit]:
        candidates = list(self.raw_dir.glob("*.md"))
        candidates.extend(
            self.raw_dir.glob("[0-9][0-9][0-9][0-9]/[0-9][0-9]/[0-9][0-9]/*.md")
        )
        for path in sorted(candidates):
            if not path.is_file() or path.is_symlink():
                continue
            stat = path.stat()
            yield RawUnit(
                raw_id=path.name,
                storage="legacy_file",
                path=path,
                offset=0,
                length=stat.st_size,
                sha256=None,
                captured_at=None,
                device=stat.st_dev,
                inode=stat.st_ino,
            )

    def _sealed_units(self) -> Iterator[RawUnit]:
        pattern = "[0-9][0-9][0-9][0-9]/[0-9][0-9]/[0-9][0-9]/*.manifest.json"
        for manifest_path in sorted(self.raw_dir.glob(pattern)):
            if manifest_path.name.startswith("legacy-part-"):
                continue
            try:
                commits = manifest_commits(manifest_path)
            except RawSegmentCorrupt:
                raise
            segment_name = manifest_path.name.removesuffix(".manifest.json")
            segment_path = manifest_path.with_name(f"{segment_name}.jsonl.zst")
            for commit in commits:
                yield RawUnit(
                    raw_id=commit.raw_id,
                    storage="segment_sealed",
                    path=segment_path,
                    offset=commit.offset,
                    length=commit.length,
                    sha256=commit.sha256,
                    captured_at=commit.captured_at,
                    commit=commit,
                )

    def _legacy_archive_units(self) -> Iterator[RawUnit]:
        from chronovisor.core.legacy_archive import iter_legacy_members

        for member in iter_legacy_members(self.raw_dir):
            # Projection manifests/receipts/noop records may share a legacy
            # archive with their completed semantic child. They are durable
            # bundle evidence, not logical Raw queue units.
            if not member.raw_id.endswith(".md"):
                continue
            yield RawUnit(
                raw_id=member.raw_id,
                storage="legacy_archive",
                path=member.archive_path,
                offset=0,
                length=member.length,
                sha256=member.sha256,
                captured_at=f"{member.captured_date.replace('/', '-')}T00:00:00+09:00",
                archive_member=member,
            )

    def _legacy_archive_index(self) -> dict[str, RawUnit]:
        """Index archive manifest rows without constructing segment/flat units."""

        if self._legacy_archive_units_by_id is None:
            from chronovisor.core.legacy_archive import iter_legacy_members

            self._legacy_archive_units_by_id = {
                member.raw_id: RawUnit(
                    raw_id=member.raw_id,
                    storage="legacy_archive",
                    path=member.archive_path,
                    offset=0,
                    length=member.length,
                    sha256=member.sha256,
                    captured_at=(
                        f"{member.captured_date.replace('/', '-')}T00:00:00+09:00"
                    ),
                    archive_member=member,
                )
                for member in iter_legacy_members(self.raw_dir)
                if member.raw_id.endswith(".md")
            }
        return self._legacy_archive_units_by_id

    def resolve_legacy_archive(self, raw_id: str) -> RawUnit | None:
        """Resolve one archived Markdown Raw without scanning every storage kind."""

        if Path(raw_id).name != raw_id or not raw_id:
            raise ValueError("raw_id must be a basename")
        return self._legacy_archive_index().get(raw_id)

    def _open_units(self) -> Iterator[RawUnit]:
        pattern = "[0-9][0-9][0-9][0-9]/[0-9][0-9]/[0-9][0-9]/*.commits.jsonl"
        for commit_path in sorted(self.raw_dir.glob(pattern)):
            base = commit_path.name.removesuffix(".commits.jsonl")
            data_path = commit_path.with_name(f"{base}.jsonl.open")
            if not data_path.is_file() or data_path.is_symlink():
                raise RawSegmentCorrupt(
                    f"open segment journal has no safe data file: {commit_path.name}"
                )
            for commit in read_commits(commit_path):
                yield RawUnit(
                    raw_id=commit.raw_id,
                    storage="segment_open",
                    path=data_path,
                    offset=commit.offset,
                    length=commit.length,
                    sha256=commit.sha256,
                    captured_at=commit.captured_at,
                    commit=commit,
                )

    def _segment_snapshot_signature(self) -> tuple[tuple[str, int, int, int, int], ...]:
        """Identify the two durable indexes that define segment membership."""

        journal_pattern = (
            "[0-9][0-9][0-9][0-9]/[0-9][0-9]/[0-9][0-9]/*.commits.jsonl"
        )
        manifest_pattern = (
            "[0-9][0-9][0-9][0-9]/[0-9][0-9]/[0-9][0-9]/*.manifest.json"
        )
        paths = [*self.raw_dir.glob(journal_pattern)]
        paths.extend(
            path
            for path in self.raw_dir.glob(manifest_pattern)
            if not path.name.startswith("legacy-part-")
        )
        signature: list[tuple[str, int, int, int, int]] = []
        for path in sorted(paths):
            try:
                identity = path.lstat()
            except FileNotFoundError:
                continue
            signature.append(
                (
                    path.relative_to(self.raw_dir).as_posix(),
                    identity.st_dev,
                    identity.st_ino,
                    identity.st_size,
                    identity.st_mtime_ns,
                )
            )
        return tuple(signature)

    def _load_segment_units(self) -> tuple[RawUnit, ...]:
        selected: dict[str, RawUnit] = {}
        for unit in tuple(self._open_units()) + tuple(self._sealed_units()):
            selected[unit.raw_id] = unit
        return tuple(selected[raw_id] for raw_id in sorted(selected))

    def iter_units(self) -> Iterator[RawUnit]:
        if self._units_cache is None:
            legacy = tuple(self._legacy_archive_units()) + tuple(self._legacy_units())
            # A crash after publishing a sealed manifest but before deleting
            # its open files may expose both copies. The sealed manifest is
            # the newer durable evidence, so load open first and sealed second.
            segments = tuple(self._open_units()) + tuple(self._sealed_units())
            ordered = (
                segments + legacy
                if self.mode in {"legacy", "shadow"}
                else legacy + segments
            )
            selected: dict[str, RawUnit] = {}
            for unit in ordered:
                previous = selected.get(unit.raw_id)
                if (
                    previous is not None
                    and previous.storage == unit.storage
                    and previous.path != unit.path
                ):
                    raise RawSegmentCorrupt(f"duplicate physical Raw ID: {unit.raw_id}")
                selected[unit.raw_id] = unit
            self._units_by_id = selected
            self._units_cache = tuple(selected[raw_id] for raw_id in sorted(selected))
        yield from self._units_cache

    def iter_segment_units(self) -> Iterator[RawUnit]:
        """List v2 logical units without touching the legacy flat directory."""

        if self._segment_units_cache is None:
            signature = self._segment_snapshot_signature()
            cached_units: tuple[RawUnit, ...] | None = None
            with self._segment_snapshots_lock:
                cached = self._segment_snapshots.get(self.raw_dir)
                if cached is not None and cached[0] == signature:
                    cached_units = cached[1]
                    self._segment_snapshots.move_to_end(self.raw_dir)
            if cached_units is not None:
                self._segment_units_cache = cached_units
            else:
                units = self._load_segment_units()
                # Never retain a snapshot whose authoritative indexes changed
                # while it was parsed; a following store will parse afresh.
                if self._segment_snapshot_signature() == signature:
                    with self._segment_snapshots_lock:
                        self._segment_snapshots[self.raw_dir] = (signature, units)
                        self._segment_snapshots.move_to_end(self.raw_dir)
                        while len(self._segment_snapshots) > _SEGMENT_SNAPSHOT_CACHE_SIZE:
                            self._segment_snapshots.popitem(last=False)
                self._segment_units_cache = units
        yield from self._segment_units_cache

    def iter_segment_bytes(
        self, raw_ids: Iterable[str] | None = None
    ) -> Iterator[tuple[RawUnit, bytes]]:
        """Read selected physical v2 segments once and verify their logical units."""

        requested = set(raw_ids) if raw_ids is not None else None
        groups: dict[tuple[RawStorageKind, Path], list[RawUnit]] = {}
        for unit in self.iter_segment_units():
            if requested is not None and unit.raw_id not in requested:
                continue
            groups.setdefault((unit.storage, unit.path), []).append(unit)
        for (storage, path), units in sorted(
            groups.items(), key=lambda item: (str(item[0][1]), item[0][0])
        ):
            logical_end = max(unit.offset + unit.length for unit in units)
            segment = (
                read_open_range(path, 0, logical_end)
                if storage == "segment_open"
                else read_sealed_range(path, 0, logical_end)
            )
            for unit in sorted(units, key=lambda item: item.raw_id):
                value = segment[unit.offset : unit.offset + unit.length]
                if len(value) != unit.length:
                    raise RawSegmentCorrupt(
                        f"segment Raw range is truncated: {unit.raw_id}"
                    )
                if unit.sha256 is None:
                    raise RawSegmentCorrupt(f"segment Raw has no digest: {unit.raw_id}")
                if hashlib.sha256(value).hexdigest() != unit.sha256:
                    raise RawSegmentCorrupt(
                        f"segment Raw digest mismatch: {unit.raw_id}"
                    )
                yield unit, value

    def resolve(self, raw_id: str) -> RawUnit | None:
        if Path(raw_id).name != raw_id or not raw_id:
            raise ValueError("raw_id must be a basename")
        if self._units_by_id is None:
            tuple(self.iter_units())
        return self._units_by_id.get(raw_id) if self._units_by_id is not None else None

    def list_all(self) -> tuple[RawUnit, ...]:
        return tuple(self.iter_units())

    def list_active(self) -> tuple[RawUnit, ...]:
        """Return addressable units; queue state is owned by the orchestrator."""

        return self.list_all()

    def resolve_segment(self, raw_id: str) -> RawUnit | None:
        if Path(raw_id).name != raw_id or not raw_id:
            raise ValueError("raw_id must be a basename")
        return next(
            (unit for unit in self.iter_segment_units() if unit.raw_id == raw_id),
            None,
        )

    def read_bytes(self, raw: str | RawUnit) -> bytes:
        unit = self.resolve(raw) if isinstance(raw, str) else raw
        if unit is None:
            raise FileNotFoundError(raw)
        if unit.storage == "legacy_file":
            return self._read_legacy_bytes(unit)
        if unit.storage == "legacy_archive":
            from chronovisor.core.legacy_archive import (
                LegacyArchiveMember,
                read_legacy_member,
            )

            if not isinstance(unit.archive_member, LegacyArchiveMember):
                raise RawSegmentCorrupt(
                    f"archived Raw has no member locator: {unit.raw_id}"
                )
            return read_legacy_member(unit.archive_member)
        if unit.storage == "segment_open":
            value = read_open_range(unit.path, unit.offset, unit.length)
        else:
            value = read_sealed_range(unit.path, unit.offset, unit.length)
        if unit.sha256 is None:
            raise RawSegmentCorrupt(f"segment Raw has no digest: {unit.raw_id}")
        if hashlib.sha256(value).hexdigest() != unit.sha256:
            raise RawSegmentCorrupt(f"segment Raw digest mismatch: {unit.raw_id}")
        return value

    def _read_legacy_bytes(self, unit: RawUnit) -> bytes:
        """Read one enumerated legacy Raw through a no-follow descriptor chain."""

        if unit.device is None or unit.inode is None:
            raise RawSegmentCorrupt("legacy Raw has no inode binding")
        try:
            relative = unit.path.relative_to(self.raw_dir)
        except ValueError as exc:
            raise RawSegmentCorrupt("legacy Raw path is outside Raw root") from exc
        if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
            raise RawSegmentCorrupt("legacy Raw relative path is invalid")
        nofollow = getattr(os, "O_NOFOLLOW", None)
        directory = getattr(os, "O_DIRECTORY", None)
        if nofollow is None or directory is None:
            raise RawSegmentCorrupt("safe legacy Raw descriptor open is unavailable")

        directory_fd: int | None = None
        file_fd: int | None = None
        try:
            directory_fd = os.open(self.raw_dir, os.O_RDONLY | directory | nofollow)
            for part in relative.parts[:-1]:
                next_fd = os.open(
                    part,
                    os.O_RDONLY | directory | nofollow,
                    dir_fd=directory_fd,
                )
                os.close(directory_fd)
                directory_fd = next_fd
            file_fd = os.open(
                relative.parts[-1], os.O_RDONLY | nofollow, dir_fd=directory_fd
            )
            observed = os.fstat(file_fd)
            if (
                not stat.S_ISREG(observed.st_mode)
                or observed.st_dev != unit.device
                or observed.st_ino != unit.inode
                or observed.st_size != unit.length
            ):
                raise RawSegmentCorrupt(
                    f"legacy Raw changed while reading: {unit.raw_id}"
                )
            with os.fdopen(file_fd, "rb") as stream:
                file_fd = None
                value = stream.read()
        except OSError as exc:
            raise RawSegmentCorrupt(
                f"legacy Raw is missing or unsafe: {unit.raw_id}"
            ) from exc
        finally:
            if file_fd is not None:
                os.close(file_fd)
            if directory_fd is not None:
                os.close(directory_fd)
        if len(value) != unit.length:
            raise RawSegmentCorrupt(f"legacy Raw changed while reading: {unit.raw_id}")
        return value

    def read_exact(self, raw: str | RawUnit) -> bytes:
        return self.read_bytes(raw)

    def logical_stat(self, raw: str | RawUnit) -> dict[str, object]:
        unit = self.resolve(raw) if isinstance(raw, str) else raw
        if unit is None:
            raise FileNotFoundError(raw)
        return {
            "raw_id": unit.raw_id,
            "storage": unit.storage,
            "logical_bytes": unit.length,
            "sha256": unit.sha256,
            "captured_at": unit.captured_at,
            "physical_path": str(unit.path),
        }

    def restore(self, raw: str | RawUnit, output: Path) -> Path:
        value = self.read_bytes(raw)
        output = output.expanduser().resolve(strict=False)
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            raise FileExistsError(output)
        from chronovisor.core.link_fix import atomic_write

        try:
            text = value.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise RawSegmentCorrupt("Raw restore target is not UTF-8") from exc
        atomic_write(output, text)
        if output.read_bytes() != value:
            output.unlink(missing_ok=True)
            raise RawSegmentCorrupt("restored Raw failed byte-exact readback")
        return output

    def quarantine(self, raw_id: str, output_dir: Path) -> Path:
        """Move only a standalone legacy source; shared archives never move."""

        unit = self.resolve(raw_id)
        if unit is None:
            raise FileNotFoundError(raw_id)
        if unit.storage != "legacy_file":
            raise RawSegmentCorrupt(
                "shared/archive Raw must be quarantined by logical queue state"
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        target = output_dir / unit.raw_id
        if target.exists():
            raise FileExistsError(target)
        os.replace(unit.path, target)
        for directory in {unit.path.parent, target.parent}:
            descriptor = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        return target

    def read_text(self, raw: str | RawUnit, *, encoding: str = "utf-8") -> str:
        return self.read_bytes(raw).decode(encoding)

    def is_archived_legacy_markdown(self, unit: RawUnit, value: bytes) -> bool:
        """Recognize one historical Markdown unit copied into a v2 segment."""

        commit = unit.commit
        if (
            commit is None
            or not unit.is_segment
            or unit.sha256 is None
            or commit.sha256 != unit.sha256
            or len(value) != unit.length
            or hashlib.sha256(value).hexdigest() != unit.sha256
        ):
            return False
        archive_name = Path(commit.source_file).name
        if (
            _LEGACY_ARCHIVE_BASENAME_RE.fullmatch(archive_name) is None
            or commit.record_count != 1
        ):
            return False
        manifest_path = unit.path.with_name(
            archive_name.removesuffix(".tar.zst") + ".manifest.json"
        )
        from chronovisor.core.canonical_document import (
            CanonicalDocumentError,
            parse_document,
        )
        from chronovisor.core.legacy_archive import verify_legacy_manifest
        from chronovisor.core.legacy_frontmatter import parse as parse_frontmatter

        try:
            commit.validate()
            text = value.decode("utf-8")
            document = parse_document(value)
        except (UnicodeDecodeError, CanonicalDocumentError, RawSegmentCorrupt):
            return False
        try:
            manifest_stat = manifest_path.lstat()
        except FileNotFoundError:
            manifest_stat = None
        except OSError:
            return False
        if manifest_stat is not None:
            if not stat.S_ISREG(manifest_stat.st_mode):
                return False
            archive_path = unit.path.with_name(archive_name)
            try:
                archive_stat = archive_path.lstat()
            except OSError:
                return False
            if not stat.S_ISREG(archive_stat.st_mode):
                return False
            artifact_identity = (
                manifest_stat.st_dev,
                manifest_stat.st_ino,
                manifest_stat.st_mtime_ns,
                manifest_stat.st_size,
                archive_stat.st_dev,
                archive_stat.st_ino,
                archive_stat.st_mtime_ns,
                archive_stat.st_size,
            )
            if self._verified_legacy_manifests.get(manifest_path) != artifact_identity:
                try:
                    manifest = verify_legacy_manifest(manifest_path, full=False)
                    manifest_after = manifest_path.lstat()
                    archive_after = archive_path.lstat()
                except (OSError, RawSegmentCorrupt):
                    return False
                after_identity = (
                    manifest_after.st_dev,
                    manifest_after.st_ino,
                    manifest_after.st_mtime_ns,
                    manifest_after.st_size,
                    archive_after.st_dev,
                    archive_after.st_ino,
                    archive_after.st_mtime_ns,
                    archive_after.st_size,
                )
                if (
                    not stat.S_ISREG(manifest_after.st_mode)
                    or not stat.S_ISREG(archive_after.st_mode)
                    or after_identity != artifact_identity
                    or manifest["archive"] != archive_name
                ):
                    return False
                self._verified_legacy_manifests[manifest_path] = artifact_identity
        keywords = document.metadata.get("raw_keywords")
        legacy_metadata, _body = parse_frontmatter(text)
        legacy_keywords = legacy_metadata.get("raw_keywords")
        return (
            set(document.metadata) == {"raw_keywords"}
            and isinstance(keywords, list)
            and not any(isinstance(keyword, (dict, list, set)) for keyword in keywords)
            and set(legacy_metadata) == {"raw_keywords"}
            and isinstance(legacy_keywords, list)
            and len(legacy_keywords) == len(keywords)
            and all(
                isinstance(keyword, str) and keyword.strip()
                for keyword in legacy_keywords
            )
        )

    def reference_payload(self, unit: RawUnit) -> dict[str, object]:
        if unit.storage == "legacy_file":
            raise ValueError("flat legacy Raw files do not need an ingest reference")
        payload: dict[str, object] = {
            "schema": RAW_REFERENCE_SCHEMA,
            "raw_id": unit.raw_id,
            "length": unit.length,
            "sha256": unit.sha256,
        }
        if unit.commit is not None:
            payload["commit"] = unit.commit.to_dict()
        elif unit.storage == "legacy_archive":
            from chronovisor.core.legacy_archive import LegacyArchiveMember

            if not isinstance(unit.archive_member, LegacyArchiveMember):
                raise RawSegmentCorrupt("archived Raw has no member evidence")
            payload["legacy_byte_passthrough"] = True
        return payload

    def materialize_ingest(self, unit: RawUnit, output_dir: Path) -> Path:
        """Publish a small, deterministic logical-reference file for queues.

        The file is intentionally not a copy of transcript bytes.  Existing
        Path-oriented completion/defer ledgers can bind to it while the
        semantic adapter resolves and verifies the authoritative segment.
        """

        if unit.storage == "legacy_file":
            return unit.path
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / unit.raw_id
        payload = self.reference_payload(unit)
        encoded = (
            json.dumps(
                payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
            + "\n"
        ).encode("utf-8")
        if path.exists():
            if path.is_symlink():
                raise RawSegmentCorrupt(
                    f"logical Raw reference conflicts with segment: {unit.raw_id}"
                )
            if path.read_bytes() == encoded:
                return path
            existing = self.resolve_reference(path)
            if existing is None or self.reference_payload(
                existing
            ) != self.reference_payload(unit):
                raise RawSegmentCorrupt(
                    f"logical Raw reference conflicts with segment: {unit.raw_id}"
                )
            return path
        from chronovisor.core.link_fix import atomic_write

        atomic_write(path, encoded.decode("utf-8"))
        return path

    def resolve_reference(self, path: Path) -> RawUnit | None:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict) or not schema_matches(
            payload.get("schema"), RAW_REFERENCE_SCHEMA
        ):
            return None
        raw_id = payload.get("raw_id")
        if not isinstance(raw_id, str) or raw_id != path.name:
            raise RawSegmentCorrupt("logical Raw reference ID is invalid")
        unit: RawUnit | None
        if isinstance(payload.get("commit"), dict):
            commit = RawSegmentCommit.from_dict(payload["commit"])
            if commit.raw_id != raw_id:
                raise RawSegmentCorrupt("logical Raw reference commit ID is invalid")
            try:
                day = datetime.fromisoformat(commit.captured_at).date()
            except (TypeError, ValueError) as exc:
                raise RawSegmentCorrupt(
                    "logical Raw reference commit timestamp is invalid"
                ) from exc
            prefix = f"{commit.host}-{commit.session_key}"
            base = f"{prefix}-part-{commit.part:03d}"
            day_dir = self.raw_dir / day.strftime("%Y") / day.strftime("%m") / day.strftime("%d")
            sealed_path = day_dir / f"{base}.jsonl.zst"
            sealed_manifest = day_dir / f"{base}.manifest.json"
            open_path = day_dir / f"{base}.jsonl.open"
            open_journal = day_dir / f"{base}.commits.jsonl"
            if (
                sealed_path.is_file()
                and not sealed_path.is_symlink()
                and sealed_manifest.is_file()
                and not sealed_manifest.is_symlink()
            ):
                observed = next(
                    (item for item in manifest_commits(sealed_manifest) if item.raw_id == raw_id),
                    None,
                )
                unit = RawUnit(
                    raw_id=raw_id,
                    storage="segment_sealed",
                    path=sealed_path,
                    offset=commit.offset,
                    length=commit.length,
                    sha256=commit.sha256,
                    captured_at=commit.captured_at,
                    commit=commit,
                )
            elif (
                open_path.is_file()
                and not open_path.is_symlink()
                and open_journal.is_file()
                and not open_journal.is_symlink()
            ):
                observed = next(
                    (item for item in read_commits(open_journal) if item.raw_id == raw_id),
                    None,
                )
                unit = RawUnit(
                    raw_id=raw_id,
                    storage="segment_open",
                    path=open_path,
                    offset=commit.offset,
                    length=commit.length,
                    sha256=commit.sha256,
                    captured_at=commit.captured_at,
                    commit=commit,
                )
            else:
                unit = None
                observed = None
            if observed != commit:
                raise RawSegmentCorrupt("logical Raw reference does not match its segment")
        else:
            # Legacy archive references historically carried only the logical
            # ID, byte length, and digest.  Resolving them through ``resolve``
            # builds the complete 20k-unit inventory for every reconciler
            # process.  Read the small archive manifests directly instead;
            # archive bytes remain verified by ``read_bytes`` before use.
            if payload.get("legacy_byte_passthrough") is True:
                expected_length = payload.get("length")
                expected_sha256 = payload.get("sha256")
                unit = self._legacy_archive_index().get(raw_id)
                if unit is not None and (
                    unit.length != expected_length or unit.sha256 != expected_sha256
                ):
                    raise RawSegmentCorrupt(
                        "logical Raw reference archive evidence mismatch"
                    )
                if unit is None:
                    # Preserve the historical fallback for a moved/deleted
                    # archive; this branch is exceptional and will be held by
                    # the caller rather than turning an invalid reference into
                    # a flat-file projection.
                    unit = self.resolve(raw_id)
            else:
                unit = self.resolve(raw_id)
        normalized_payload = dict(payload)
        normalized_payload["schema"] = RAW_REFERENCE_SCHEMA
        if unit is None or normalized_payload != self.reference_payload(unit):
            raise RawSegmentCorrupt("logical Raw reference does not match its segment")
        return unit

    def __contains__(self, raw_id: object) -> bool:
        return isinstance(raw_id, str) and self.resolve(raw_id) is not None

    def __iter__(self) -> Iterator[RawUnit]:
        return self.iter_units()


def committed_raw_watermark(raw_dir: Path) -> str:
    """Return the committed-receipt inventory identity without Raw content."""

    rows: list[dict[str, object]] = []
    for unit in RawStore(raw_dir, mode="v2").iter_segment_units():
        commit = unit.commit
        if commit is None or unit.sha256 is None or unit.captured_at is None:
            raise RawSegmentCorrupt("Raw unit has no committed receipt")
        rows.append(
            {
                "raw_id": unit.raw_id,
                "byte_range": [0, unit.length],
                "byte_coordinate_space": "logical_raw",
                "raw_sha256": unit.sha256,
                "receipt_sha256": canonical_json_sha256_strict(commit.to_dict()),
                "captured_at": unit.captured_at,
                "host": commit.host,
                "session_key": commit.session_key,
                "source_line_range": [commit.after_line, commit.until_line],
            }
        )
    return canonical_json_sha256_strict(rows)
