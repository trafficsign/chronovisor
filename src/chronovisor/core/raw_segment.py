"""Crash-safe source-native transcript segments for Raw Archive v2.

Open segments live directly below ``raw/YYYY/MM/DD``.  The date is a physical
partition, while ``.open``/``.zst`` and the manifest express lifecycle state.
Every save transaction remains a logical Raw unit with its historical
``save-<idempotency-key>.md`` identity; callers never use the segment filename
as the durable identity.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, TypeGuard
from zoneinfo import ZoneInfo

import zstandard as zstd

from chronovisor.core.hashutil import sha256_bytes as _sha256
from chronovisor.core.hashutil import sha256_file as _sha256_path
from chronovisor.core.sealed_artifact_decoder import schema_matches

COMMIT_SCHEMA = "chronovisor.raw-segment-commit.v1"
MANIFEST_SCHEMA = "chronovisor.raw-segment-manifest.v1"
DEFAULT_PART_BYTES = 128 * 1024 * 1024
DEFAULT_COMPRESSION_LEVEL = 9
MAX_ZSTD_WINDOW_BYTES = 256 * 1024 * 1024
CAPTURE_TIMEZONE = ZoneInfo("Asia/Tokyo")

_HOST_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,31}$")
_SESSION_KEY_RE = re.compile(r"^[0-9a-f]{24}$")
_RAW_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,239}$")
_PART_RE = re.compile(r"-part-(\d{3})(?:\.jsonl\.(?:open|zst))$")
_SAVE_RAW_ID_RE = re.compile(
    r"^save-(?P<host>codex|claude-code)-(?P<session>[0-9a-f]{24})-from\d+-to\d+\.md$"
)


class RawSegmentError(RuntimeError):
    """A segment cannot be published, verified, or restored safely."""


class RawSegmentCorrupt(RawSegmentError):
    """Durable segment bytes disagree with their commit or manifest."""


@dataclass(frozen=True)
class RawSegmentCommit:
    schema: str
    raw_id: str
    idempotency_key: str
    host: str
    session_key: str
    session_id: str | None
    source_file: str
    after_line: int
    until_line: int
    offset: int
    length: int
    sha256: str
    record_count: int
    captured_at: str
    part: int

    @classmethod
    def from_dict(cls, value: object) -> RawSegmentCommit:
        if not isinstance(value, dict):
            raise RawSegmentCorrupt("segment commit is not an object")
        try:
            commit = cls(**value)
        except TypeError as exc:
            raise RawSegmentCorrupt("segment commit schema mismatch") from exc
        commit.validate()
        return commit

    def validate(self) -> None:
        if not schema_matches(self.schema, COMMIT_SCHEMA):
            raise RawSegmentCorrupt("segment commit schema version mismatch")
        if (
            _RAW_ID_RE.fullmatch(self.raw_id) is None
            or Path(self.raw_id).name != self.raw_id
        ):
            raise RawSegmentCorrupt("segment raw_id is invalid")
        if not self.idempotency_key or self.raw_id != f"save-{self.idempotency_key}.md":
            raise RawSegmentCorrupt("segment raw_id/idempotency identity mismatch")
        if _HOST_RE.fullmatch(self.host) is None:
            raise RawSegmentCorrupt("segment host is invalid")
        if _SESSION_KEY_RE.fullmatch(self.session_key) is None:
            raise RawSegmentCorrupt("segment session key is invalid")
        if self.after_line < 0 or self.until_line <= self.after_line:
            raise RawSegmentCorrupt("segment source interval is invalid")
        if self.offset < 0 or self.length <= 0 or self.record_count <= 0:
            raise RawSegmentCorrupt("segment range is invalid")
        if self.part < 1 or self.part > 999:
            raise RawSegmentCorrupt("segment part is invalid")
        if not _is_sha256(self.sha256):
            raise RawSegmentCorrupt("segment range sha256 is invalid")
        try:
            datetime.fromisoformat(self.captured_at)
        except (TypeError, ValueError) as exc:
            raise RawSegmentCorrupt("segment capture timestamp is invalid") from exc

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RawSegmentReceipt:
    commit: RawSegmentCommit
    data_path: Path
    commit_path: Path
    deduplicated: bool

    @property
    def sealed(self) -> bool:
        return self.data_path.name.endswith(".jsonl.zst")

    def to_result(self) -> dict[str, object]:
        return {
            "saved": self.commit.raw_id,
            "raw_id": self.commit.raw_id,
            "path": str(self.data_path),
            "commit_path": str(self.commit_path),
            "storage": "segment_sealed" if self.sealed else "segment_open",
            "deduplicated": self.deduplicated,
        }


def _is_sha256(value: object) -> TypeGuard[str]:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )






def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    encoded = (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
    try:
        descriptor = os.open(
            temporary,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o600,
        )
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def capture_date(now: datetime | None = None) -> str:
    current = now or datetime.now(CAPTURE_TIMEZONE)
    if current.tzinfo is None:
        current = current.replace(tzinfo=CAPTURE_TIMEZONE)
    return current.astimezone(CAPTURE_TIMEZONE).strftime("%Y/%m/%d")


def _captured_at(now: datetime | None = None) -> str:
    current = now or datetime.now(CAPTURE_TIMEZONE)
    if current.tzinfo is None:
        current = current.replace(tzinfo=CAPTURE_TIMEZONE)
    return current.astimezone(CAPTURE_TIMEZONE).isoformat()


def _segment_prefix(host: str, session_key: str) -> str:
    if _HOST_RE.fullmatch(host) is None:
        raise ValueError(f"invalid segment host: {host!r}")
    if _SESSION_KEY_RE.fullmatch(session_key) is None:
        raise ValueError(f"invalid segment session key: {session_key!r}")
    return f"{host}-{session_key}"


def segment_paths(day_dir: Path, prefix: str, part: int) -> tuple[Path, Path]:
    base = f"{prefix}-part-{part:03d}"
    return day_dir / f"{base}.jsonl.open", day_dir / f"{base}.commits.jsonl"


def sealed_paths(data_path: Path) -> tuple[Path, Path]:
    suffix = ".jsonl.open"
    if not data_path.name.endswith(suffix):
        raise ValueError(f"not an open raw segment: {data_path}")
    base = data_path.name[: -len(suffix)]
    return (
        data_path.with_name(f"{base}.jsonl.zst"),
        data_path.with_name(f"{base}.manifest.json"),
    )


def journal_path_for(data_path: Path) -> Path:
    suffix = ".jsonl.open"
    if not data_path.name.endswith(suffix):
        raise ValueError(f"not an open raw segment: {data_path}")
    base = data_path.name[: -len(suffix)]
    return data_path.with_name(f"{base}.commits.jsonl")


def _complete_journal_bytes(path: Path) -> bytes:
    if not path.exists():
        return b""
    raw = path.read_bytes()
    if not raw:
        return b""
    end = raw.rfind(b"\n")
    return raw[: end + 1] if end >= 0 else b""


def read_commits(path: Path) -> tuple[RawSegmentCommit, ...]:
    commits: list[RawSegmentCommit] = []
    for line in _complete_journal_bytes(path).splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RawSegmentCorrupt(f"invalid commit journal row: {path.name}") from exc
        commits.append(RawSegmentCommit.from_dict(payload))
    previous_end = 0
    seen: set[str] = set()
    for commit in commits:
        if commit.raw_id in seen:
            raise RawSegmentCorrupt(
                f"duplicate raw_id in commit journal: {commit.raw_id}"
            )
        if commit.offset != previous_end:
            raise RawSegmentCorrupt("segment commit ranges are not contiguous")
        previous_end = commit.offset + commit.length
        seen.add(commit.raw_id)
    return tuple(commits)


def repair_open_segment(
    data_path: Path, commit_path: Path
) -> tuple[RawSegmentCommit, ...]:
    """Discard only bytes that have no durable commit receipt."""

    commit_path.parent.mkdir(parents=True, exist_ok=True)
    complete = _complete_journal_bytes(commit_path)
    if commit_path.exists() and commit_path.read_bytes() != complete:
        with commit_path.open("r+b") as handle:
            handle.truncate(len(complete))
            handle.flush()
            os.fsync(handle.fileno())
    commits = read_commits(commit_path)
    committed_end = commits[-1].offset + commits[-1].length if commits else 0
    data_path.parent.mkdir(parents=True, exist_ok=True)
    if not data_path.exists():
        if committed_end:
            raise RawSegmentCorrupt("commit journal exists but segment data is missing")
        data_path.touch(mode=0o600)
    actual_size = data_path.stat().st_size
    if actual_size < committed_end:
        raise RawSegmentCorrupt("segment data is shorter than its committed range")
    if actual_size > committed_end:
        with data_path.open("r+b") as handle:
            handle.truncate(committed_end)
            handle.flush()
            os.fsync(handle.fileno())
    for commit in commits:
        if (
            _sha256(read_open_range(data_path, commit.offset, commit.length))
            != commit.sha256
        ):
            raise RawSegmentCorrupt(f"committed range hash mismatch: {commit.raw_id}")
    return commits


def read_open_range(path: Path, offset: int, length: int) -> bytes:
    with path.open("rb") as handle:
        handle.seek(offset)
        raw = handle.read(length)
    if len(raw) != length:
        raise RawSegmentCorrupt("open segment range is truncated")
    return raw


def read_sealed_range(path: Path, offset: int, length: int) -> bytes:
    if offset < 0 or length <= 0:
        raise ValueError("sealed segment range is invalid")
    remaining_skip = offset
    result = bytearray()
    with path.open("rb") as source:
        with zstd.ZstdDecompressor(max_window_size=MAX_ZSTD_WINDOW_BYTES).stream_reader(
            source
        ) as reader:
            while remaining_skip:
                chunk = reader.read(min(1024 * 1024, remaining_skip))
                if not chunk:
                    raise RawSegmentCorrupt(
                        "sealed segment ends before requested range"
                    )
                remaining_skip -= len(chunk)
            remaining = length
            while remaining:
                chunk = reader.read(min(1024 * 1024, remaining))
                if not chunk:
                    raise RawSegmentCorrupt("sealed segment range is truncated")
                result.extend(chunk)
                remaining -= len(chunk)
    return bytes(result)


def _raw_id_prefix(raw_id: str) -> str | None:
    match = _SAVE_RAW_ID_RE.fullmatch(raw_id)
    if match is None:
        return None
    return f"{match.group('host')}-{match.group('session')}"


def _journal_candidates(raw_dir: Path, *, prefix: str | None = None) -> Iterable[Path]:
    filename = f"{prefix}-part-*.commits.jsonl" if prefix else "*.commits.jsonl"
    yield from raw_dir.glob(f"[0-9][0-9][0-9][0-9]/[0-9][0-9]/[0-9][0-9]/{filename}")


def _manifest_candidates(raw_dir: Path, *, prefix: str | None = None) -> Iterable[Path]:
    filename = f"{prefix}-part-*.manifest.json" if prefix else "*.manifest.json"
    yield from raw_dir.glob(f"[0-9][0-9][0-9][0-9]/[0-9][0-9]/[0-9][0-9]/{filename}")


def find_commit(raw_dir: Path, raw_id: str) -> RawSegmentReceipt | None:
    prefix = _raw_id_prefix(raw_id)
    for path in sorted(_journal_candidates(raw_dir, prefix=prefix)):
        for commit in read_commits(path):
            if commit.raw_id == raw_id:
                base = path.name.removesuffix(".commits.jsonl")
                return RawSegmentReceipt(
                    commit=commit,
                    data_path=path.with_name(f"{base}.jsonl.open"),
                    commit_path=path,
                    deduplicated=True,
                )
    for path in sorted(_manifest_candidates(raw_dir, prefix=prefix)):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict) or not schema_matches(
            payload.get("schema"), MANIFEST_SCHEMA
        ):
            continue
        segment = path.with_name(str(payload.get("segment") or ""))
        for row in payload.get("commits", []):
            commit = RawSegmentCommit.from_dict(row)
            if commit.raw_id == raw_id:
                return RawSegmentReceipt(
                    commit=commit,
                    data_path=segment,
                    commit_path=path,
                    deduplicated=True,
                )
    return None


def append_capture(
    *,
    raw_dir: Path,
    raw_id: str,
    idempotency_key: str,
    host: str,
    session_key: str,
    session_id: str | None,
    source_file: Path,
    after_line: int,
    until_line: int,
    source_bytes: bytes,
    record_count: int,
    now: datetime | None = None,
    max_part_bytes: int = DEFAULT_PART_BYTES,
) -> RawSegmentReceipt:
    """Append one exact source interval and publish its durable commit receipt."""

    if raw_id != f"save-{idempotency_key}.md":
        raise ValueError("raw_id must preserve the legacy save transaction identity")
    if not source_bytes or not source_bytes.endswith(b"\n"):
        raise ValueError(
            "source_bytes must contain complete LF-terminated JSONL records"
        )
    if record_count <= 0 or max_part_bytes <= 0:
        raise ValueError("record_count and max_part_bytes must be positive")
    prefix = _segment_prefix(host, session_key)
    identity_lock_dir = raw_dir / ".locks"
    identity_lock_dir.mkdir(parents=True, exist_ok=True)
    # A bounded lock shard prevents the idempotency race across capture-day
    # boundaries without creating one permanent filesystem entry per Raw.
    identity_lock_path = (
        identity_lock_dir / f"idempotency-{_sha256(raw_id.encode())[:2]}.lock"
    )
    with identity_lock_path.open("a+b") as identity_lock:
        fcntl.flock(identity_lock.fileno(), fcntl.LOCK_EX)
        try:
            return _append_capture_locked(
                raw_dir=raw_dir,
                raw_id=raw_id,
                idempotency_key=idempotency_key,
                host=host,
                session_key=session_key,
                session_id=session_id,
                source_file=source_file,
                after_line=after_line,
                until_line=until_line,
                source_bytes=source_bytes,
                record_count=record_count,
                now=now,
                max_part_bytes=max_part_bytes,
                prefix=prefix,
            )
        finally:
            fcntl.flock(identity_lock.fileno(), fcntl.LOCK_UN)


def _append_capture_locked(
    *,
    raw_dir: Path,
    raw_id: str,
    idempotency_key: str,
    host: str,
    session_key: str,
    session_id: str | None,
    source_file: Path,
    after_line: int,
    until_line: int,
    source_bytes: bytes,
    record_count: int,
    now: datetime | None,
    max_part_bytes: int,
    prefix: str,
) -> RawSegmentReceipt:
    """Publish while the cross-date logical Raw identity lock is held."""

    existing = find_commit(raw_dir, raw_id)
    if existing is not None:
        if (
            existing.commit.sha256 != _sha256(source_bytes)
            or existing.commit.after_line != after_line
            or existing.commit.until_line != until_line
            or existing.commit.host != host
            or existing.commit.session_key != session_key
        ):
            raise RawSegmentCorrupt(
                "idempotency key collision with different source bytes"
            )
        return existing

    day_dir = raw_dir / capture_date(now)
    day_dir.mkdir(parents=True, exist_ok=True)
    lock_path = day_dir / f".{prefix}.lock"
    with lock_path.open("a+b") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            existing = find_commit(raw_dir, raw_id)
            if existing is not None:
                if existing.commit.sha256 != _sha256(source_bytes):
                    raise RawSegmentCorrupt(
                        "idempotency key collision with different source bytes"
                    )
                return existing

            segment_files = tuple(day_dir.glob(f"{prefix}-part-*.jsonl.open")) + tuple(
                day_dir.glob(f"{prefix}-part-*.jsonl.zst")
            )
            parts = {
                int(match.group(1))
                for path in segment_files
                if (match := _PART_RE.search(path.name)) is not None
            }
            open_parts = {
                int(match.group(1))
                for path in day_dir.glob(f"{prefix}-part-*.jsonl.open")
                if (match := _PART_RE.search(path.name)) is not None
                and not path.with_name(
                    path.name.removesuffix(".jsonl.open") + ".manifest.json"
                ).exists()
            }
            part = max(open_parts) if open_parts else max(parts, default=0) + 1
            data_path, commit_path = segment_paths(day_dir, prefix, part)
            commits = repair_open_segment(data_path, commit_path)
            committed_end = commits[-1].offset + commits[-1].length if commits else 0
            if committed_end and committed_end + len(source_bytes) > max_part_bytes:
                part += 1
                if part > 999:
                    raise RawSegmentError("daily segment part limit exceeded")
                data_path, commit_path = segment_paths(day_dir, prefix, part)
                commits = repair_open_segment(data_path, commit_path)
                committed_end = (
                    commits[-1].offset + commits[-1].length if commits else 0
                )

            with data_path.open("ab") as data_handle:
                data_handle.write(source_bytes)
                data_handle.flush()
                os.fsync(data_handle.fileno())

            commit = RawSegmentCommit(
                schema=COMMIT_SCHEMA,
                raw_id=raw_id,
                idempotency_key=idempotency_key,
                host=host,
                session_key=session_key,
                session_id=session_id,
                source_file=str(source_file.expanduser().resolve(strict=False)),
                after_line=after_line,
                until_line=until_line,
                offset=committed_end,
                length=len(source_bytes),
                sha256=_sha256(source_bytes),
                record_count=record_count,
                captured_at=_captured_at(now),
                part=part,
            )
            commit.validate()
            encoded = (
                json.dumps(
                    commit.to_dict(),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            ).encode("utf-8")
            with commit_path.open("ab") as commit_handle:
                commit_handle.write(encoded)
                commit_handle.flush()
                os.fsync(commit_handle.fileno())
            _fsync_directory(day_dir)
            if (
                _sha256(read_open_range(data_path, commit.offset, commit.length))
                != commit.sha256
            ):
                raise RawSegmentCorrupt("published segment receipt failed readback")
            return RawSegmentReceipt(
                commit=commit,
                data_path=data_path,
                commit_path=commit_path,
                deduplicated=False,
            )
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def _verify_compressed(
    path: Path, *, expected_bytes: int, expected_sha256: str
) -> None:
    digest = hashlib.sha256()
    observed = 0
    with path.open("rb") as source:
        with zstd.ZstdDecompressor(max_window_size=MAX_ZSTD_WINDOW_BYTES).stream_reader(
            source
        ) as reader:
            while chunk := reader.read(1024 * 1024):
                digest.update(chunk)
                observed += len(chunk)
                if observed > expected_bytes:
                    raise RawSegmentCorrupt(
                        "compressed segment exceeds declared logical bytes"
                    )
    if observed != expected_bytes or digest.hexdigest() != expected_sha256:
        raise RawSegmentCorrupt("compressed segment restore verification failed")


def seal_segment(
    data_path: Path,
    *,
    compression_level: int = DEFAULT_COMPRESSION_LEVEL,
    remove_open: bool = False,
) -> dict[str, Any]:
    """Seal one open segment without changing any logical Raw identity."""

    commit_path = journal_path_for(data_path)
    compressed_path, manifest_path = sealed_paths(data_path)
    if manifest_path.exists():
        manifest = verify_manifest(manifest_path, full=True)
        if remove_open:
            data_path.unlink(missing_ok=True)
            commit_path.unlink(missing_ok=True)
            _fsync_directory(data_path.parent)
        return manifest

    base = data_path.name.removesuffix(".jsonl.open")
    prefix = re.sub(r"-part-\d{3}$", "", base)
    lock_path = data_path.parent / f".{prefix}.lock"
    with lock_path.open("a+b") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        temporary = compressed_path.with_name(
            f".{compressed_path.name}.{os.getpid()}.tmp"
        )
        try:
            if manifest_path.exists():
                manifest = verify_manifest(manifest_path, full=True)
            else:
                commits = repair_open_segment(data_path, commit_path)
                if not commits:
                    raise RawSegmentError("cannot seal an empty segment")
                logical_bytes = data_path.stat().st_size
                logical_sha256 = _sha256_path(data_path)
                with data_path.open("rb") as source, temporary.open("wb") as target:
                    zstd.ZstdCompressor(level=compression_level).copy_stream(
                        source, target
                    )
                    target.flush()
                    os.fsync(target.fileno())
                _verify_compressed(
                    temporary,
                    expected_bytes=logical_bytes,
                    expected_sha256=logical_sha256,
                )
                os.replace(temporary, compressed_path)
                _fsync_directory(compressed_path.parent)
                manifest = {
                    "schema": MANIFEST_SCHEMA,
                    "segment": compressed_path.name,
                    "logical_bytes": logical_bytes,
                    "logical_sha256": logical_sha256,
                    "compressed_bytes": compressed_path.stat().st_size,
                    "compressed_sha256": _sha256_path(compressed_path),
                    "compression": {"codec": "zstd", "level": compression_level},
                    "sealed_at": _captured_at(),
                    "commits": [commit.to_dict() for commit in commits],
                }
                _atomic_write_json(manifest_path, manifest)
                verify_manifest(manifest_path, full=True)
            if remove_open:
                data_path.unlink(missing_ok=True)
                commit_path.unlink(missing_ok=True)
                _fsync_directory(data_path.parent)
            return manifest
        finally:
            temporary.unlink(missing_ok=True)
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def load_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RawSegmentCorrupt(f"segment manifest is unreadable: {path.name}") from exc
    if not isinstance(payload, dict) or not schema_matches(
        payload.get("schema"), MANIFEST_SCHEMA
    ):
        raise RawSegmentCorrupt("segment manifest schema mismatch")
    return payload


def manifest_commits(path: Path) -> tuple[RawSegmentCommit, ...]:
    payload = load_manifest(path)
    commits = tuple(
        RawSegmentCommit.from_dict(row) for row in payload.get("commits", [])
    )
    previous_end = 0
    for commit in commits:
        if commit.offset != previous_end:
            raise RawSegmentCorrupt("sealed commit ranges are not contiguous")
        previous_end = commit.offset + commit.length
    if previous_end != payload.get("logical_bytes"):
        raise RawSegmentCorrupt("sealed commit ranges do not cover the segment")
    return commits


def verify_manifest(path: Path, *, full: bool = False) -> dict[str, Any]:
    payload = load_manifest(path)
    segment_name = payload.get("segment")
    if not isinstance(segment_name, str) or Path(segment_name).name != segment_name:
        raise RawSegmentCorrupt("manifest segment path is invalid")
    segment = path.with_name(segment_name)
    if not segment.is_file() or segment.is_symlink():
        raise RawSegmentCorrupt("manifest segment is missing or unsafe")
    compressed_bytes = payload.get("compressed_bytes")
    compressed_sha256 = payload.get("compressed_sha256")
    logical_bytes = payload.get("logical_bytes")
    logical_sha256 = payload.get("logical_sha256")
    if (
        not isinstance(compressed_bytes, int)
        or compressed_bytes <= 0
        or not _is_sha256(compressed_sha256)
        or not isinstance(logical_bytes, int)
        or logical_bytes <= 0
        or not _is_sha256(logical_sha256)
    ):
        raise RawSegmentCorrupt("manifest size or digest fields are invalid")
    if (
        segment.stat().st_size != compressed_bytes
        or _sha256_path(segment) != compressed_sha256
    ):
        raise RawSegmentCorrupt("compressed segment evidence mismatch")
    commits = manifest_commits(path)
    if len({commit.raw_id for commit in commits}) != len(commits):
        raise RawSegmentCorrupt("sealed segment contains duplicate raw IDs")
    if full:
        _verify_compressed(
            segment,
            expected_bytes=logical_bytes,
            expected_sha256=logical_sha256,
        )
        for commit in commits:
            restored = read_sealed_range(segment, commit.offset, commit.length)
            if _sha256(restored) != commit.sha256:
                raise RawSegmentCorrupt(
                    f"sealed raw range hash mismatch: {commit.raw_id}"
                )
    return payload


def copy_source_interval(path: Path, *, after_line: int, until_line: int) -> bytes:
    """Return exact complete source lines for one cursor interval.

    A non-newline-terminated tail is deliberately excluded from ``until_line``
    by the caller.  Encountering one inside the requested interval is an error.
    """

    if after_line < 0 or until_line <= after_line:
        raise ValueError("source interval is invalid")
    selected: list[bytes] = []
    with path.open("rb") as handle:
        for line_no, line in enumerate(handle, start=1):
            if line_no <= after_line:
                continue
            if line_no > until_line:
                break
            if not line.endswith(b"\n"):
                raise RawSegmentError(
                    "source interval contains an incomplete trailing line"
                )
            selected.append(line)
    if len(selected) != until_line - after_line:
        raise RawSegmentError(
            "source interval is shorter than the durable cursor range"
        )
    return b"".join(selected)


def restored_segment_bytes(path: Path) -> bytes:
    """Test/export helper for one bounded sealed segment."""

    with path.open("rb") as source:
        with zstd.ZstdDecompressor(max_window_size=MAX_ZSTD_WINDOW_BYTES).stream_reader(
            source
        ) as reader:
            return reader.read()
