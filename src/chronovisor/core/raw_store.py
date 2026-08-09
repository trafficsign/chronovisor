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
import tomllib
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

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
    if value is None and os.environ.get("CHRONOVISOR_RAW_LAYOUT") is None and chronovisor_root:
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

    @property
    def is_segment(self) -> bool:
        return self.storage in {"segment_open", "segment_sealed"}


class RawStore:
    """Dual-read store with deterministic identity precedence.

    ``legacy`` and ``shadow`` prefer a legacy file when both layouts contain
    the same logical Raw ID.  ``v2`` prefers a segment commit.  This makes the
    feature flag reversible while shadow comparison is running.
    """

    def __init__(self, raw_dir: Path, *, mode: RawLayoutMode | str | None = None):
        self.raw_dir = raw_dir.expanduser().resolve(strict=False)
        self.mode = raw_layout_mode(mode, chronovisor_root=self.raw_dir.parent)
        self._units_cache: tuple[RawUnit, ...] | None = None
        self._units_by_id: dict[str, RawUnit] | None = None
        self._segment_units_cache: tuple[RawUnit, ...] | None = None

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
                    raise RawSegmentCorrupt(
                        f"duplicate physical Raw ID: {unit.raw_id}"
                    )
                selected[unit.raw_id] = unit
            self._units_by_id = selected
            self._units_cache = tuple(selected[raw_id] for raw_id in sorted(selected))
        yield from self._units_cache

    def iter_segment_units(self) -> Iterator[RawUnit]:
        """List v2 logical units without touching the legacy flat directory."""

        if self._segment_units_cache is None:
            selected: dict[str, RawUnit] = {}
            for unit in tuple(self._open_units()) + tuple(self._sealed_units()):
                selected[unit.raw_id] = unit
            self._segment_units_cache = tuple(
                selected[raw_id] for raw_id in sorted(selected)
            )
        yield from self._segment_units_cache

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
            value = unit.path.read_bytes()
            if len(value) != unit.length:
                raise RawSegmentCorrupt(
                    f"legacy Raw changed while reading: {unit.raw_id}"
                )
            return value
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
            if (
                existing is None
                or self.reference_payload(existing) != self.reference_payload(unit)
            ):
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
        if (
            not isinstance(payload, dict)
            or not schema_matches(payload.get("schema"), RAW_REFERENCE_SCHEMA)
        ):
            return None
        raw_id = payload.get("raw_id")
        if not isinstance(raw_id, str) or raw_id != path.name:
            raise RawSegmentCorrupt("logical Raw reference ID is invalid")
        unit = (
            self.resolve_segment(raw_id)
            if isinstance(payload.get("commit"), dict)
            else self.resolve(raw_id)
        )
        normalized_payload = dict(payload)
        normalized_payload["schema"] = RAW_REFERENCE_SCHEMA
        if unit is None or normalized_payload != self.reference_payload(unit):
            raise RawSegmentCorrupt("logical Raw reference does not match its segment")
        return unit

    def __contains__(self, raw_id: object) -> bool:
        return isinstance(raw_id, str) and self.resolve(raw_id) is not None

    def __iter__(self) -> Iterator[RawUnit]:
        return self.iter_units()
