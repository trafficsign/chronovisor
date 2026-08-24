"""Append-only, read-only evidence collection for a real Recall R7 rollout.

This module deliberately does not advance a rollout or call a teacher.  It is
the narrow boundary between live, independently captured observations and the
receipt validator: a poll is accepted only after the current sealed policy
state and local process/dashboard facts have been re-read by this process.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import csv
import ctypes
import fcntl
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from importlib import metadata
from pathlib import Path
from statistics import NormalDist
from typing import Any
from urllib.parse import unquote, urlsplit, urlunsplit

from chronovisor.core.canonical_json import (
    canonical_json_line_bytes_strict,
    canonical_json_sha256_strict,
)
from chronovisor.core.runtime_config import (
    DEFAULT_DASHBOARD_HOST,
    DEFAULT_DASHBOARD_PORT,
    DEFAULT_LAUNCHD_LABEL_PREFIX,
)
from chronovisor.recall import recall_distillation as distillation
from chronovisor.recall import recall_distillation_rollout as rollout
from chronovisor.recall import recall_distillation_store as store

EVIDENCE_SCHEMA = "chronovisor.recall-r7-evidence.v1"
POLL_SCHEMA = "chronovisor.recall-r7-poll.v2"
LEDGER_SCHEMA = "chronovisor.recall-r7-poll-ledger.v1"
LIVE_ATTESTATION_SCHEMA = "chronovisor.recall-r7-live-attestation.v1"
TEST_LIVE_ATTESTATION_SCHEMA = "chronovisor.recall-r7-live-attestation-test.v1"
DOM_CAPTURE_SCHEMA = "chronovisor.recall-r7-dom-capture.v1"
TEST_DOM_CAPTURE_SCHEMA = "chronovisor.recall-r7-dom-capture-test.v1"
TEST_FAILURE_SCHEMA = "chronovisor.recall-r7-failure-test.v1"
STAGES = ("shadow", "5", "25", "100")
MIN_DAYS = 7
MIN_PAIRED = 500
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_STARTED_AT = re.compile(r"[1-9][0-9]*\.[0-9]{6}\Z")
MAX_BYTES = 12 * 1024 * 1024
# Launchd may point at the installed ``uv``/``uvx`` binary; keep process
# identity bounded without rejecting the current 44 MiB deployment image.
MAX_PROCESS_BYTES = 64 * 1024 * 1024
MAX_OBSERVATION_SKEW_SECONDS = 300
_STAT_KEYS = frozenset({"dev", "ino", "mode", "size", "mtime_ns", "ctime_ns"})
_TRUSTED_GIT = Path("/usr/bin/git")
_FIXED_PRODUCTION_ROOT = (Path.home() / ".chronovisor").resolve(strict=False)
_FIXED_EVIDENCE_ROOT = (
    _FIXED_PRODUCTION_ROOT / "runtime" / "recall-distillation" / "r7-live-evidence"
)
_FIXED_DOM_CAPTURE_ROOT = _FIXED_EVIDENCE_ROOT / "dom-captures"
_FIXED_DASHBOARD_ORIGIN = (
    f"http://{DEFAULT_DASHBOARD_HOST}:{DEFAULT_DASHBOARD_PORT}"
)
_ARCHIVE_KEYS = (
    "archive_commit",
    "module_path",
    "module_bytes_sha256",
    "distribution_name",
    "distribution_version",
    "record_path",
    "record_file_sha256",
    "record_module_sha256",
    "record_module_size",
    "tracked_path",
    "tracked_mode",
    "tracked_blob_sha1",
)
_TEST_RUNTIME_KEYS = frozenset(
    {
        "archive_commit",
        "direct_url_sha256",
        "direct_url_raw_sha256",
        "direct_url_payload_sha256",
        "module_path",
        "module_bytes_sha256",
        "module_lstat",
        "distribution_name",
        "distribution_version",
    }
)
_RUNTIME_KEYS = frozenset(
    {
        *_TEST_RUNTIME_KEYS,
        "record_path",
        "record_file_sha256",
        "record_module_sha256",
        "record_module_size",
        "tracked_path",
        "tracked_mode",
        "tracked_blob_sha1",
    }
)
_SERVICE_LABELS = {
    # Do not derive formal labels from ambient environment/configuration.  A
    # launchd prefix is a deployment constant, not caller evidence.
    "dashboard": f"{DEFAULT_LAUNCHD_LABEL_PREFIX}dashboard.managed",
    "ingest": f"{DEFAULT_LAUNCHD_LABEL_PREFIX}ingest-drain.managed",
    "lan-dashboard": f"{DEFAULT_LAUNCHD_LABEL_PREFIX}lan-dashboard.managed",
    "library-evidence": f"{DEFAULT_LAUNCHD_LABEL_PREFIX}library-evidence.managed",
}
_LAUNCHCTL_FIELDS = frozenset(
    {
        # Identity fields used by the closed receipt contract.
        "label",
        "service",
        "role",
        "pid",
        "parent pid",
        "child pid",
        "state",
        # Fields emitted by `launchctl print` for a submitted LaunchAgent.
        "active count",
        "service count",
        "active service count",
        "maximum allowed shutdown time",
        "service stats",
        "trial factors",
        "reload count",
        "memory",
        "active config locked",
        "creator",
        "creator euid",
        "auxiliary bootstrapper",
        "security context",
        "bringup time",
        "death port",
        "subdomains",
        "ID",
        "name",
        "path",
        "type",
        "managed_by",
        "program identifier",
        "parent bundle identifier",
        "parent bundle version",
        "BTM uuid",
        "program",
        "arguments",
        "working directory",
        "stdout path",
        "stderr path",
        "inherited environment",
        "default environment",
        "environment",
        "domain",
        "asid",
        "minimum runtime",
        "exit timeout",
        "runs",
        "immediate reason",
        "forks",
        "execs",
        "initialized",
        "trampolined",
        "started suspended",
        "proxy started suspended",
        "checked allocations",
        "checked allocations reason",
        "checked allocations flags",
        "last exit code",
        "last terminating signal",
        "resource coalition",
        "jetsam coalition",
        "jetsam priority",
        "jetsam memory limit (active)",
        "jetsam memory limit (inactive)",
        "jetsamproperties category",
        "jetsam thread limit",
        "cpumon",
        "spawn type",
        "job state",
        "properties",
        "submitted job. ignore execute allowed",
    }
)
_LAUNCHCTL_BARE_BLOCKS = frozenset(
    {"arguments", "service stats", "subdomains", "security context"}
)
_LAUNCHCTL_ENVIRONMENT_BLOCKS = frozenset(
    {"environment", "default environment", "inherited environment"}
)
_LAUNCHCTL_BARE_LINES = frozenset({"submitted job. ignore execute allowed"})
_LAUNCHCTL_ARGUMENTS_LIST_SCOPE = "__arguments_list__"
_LAUNCHCTL_ARGV_SCALAR = re.compile(r"^[^\x00-\x1f\x7f\\=(){}<>]+$")


class EvidenceError(ValueError):
    """A live evidence input cannot be safely certified."""


def _fixed_direct_url_path() -> Path:
    """Resolve the installed distribution's direct_url receipt.

    The caller is deliberately not allowed to select this path.  If the
    installed archive has no direct_url receipt, formal evidence is held.
    """

    try:
        distribution = metadata.distribution("chronovisor")
        distribution_path = getattr(distribution, "_path", None)
        if not isinstance(distribution_path, Path):
            raise EvidenceError("installed archive metadata is unavailable")
        candidate = distribution_path / "direct_url.json"
        if _has_symlink_component(candidate):
            raise EvidenceError("installed archive metadata is symlinked")
        path = candidate.resolve(strict=True)
    except (OSError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise EvidenceError("installed archive metadata is unavailable") from exc
    return path


def _fixed_source_root() -> Path:
    """Resolve the installed archive's trusted editable checkout, if any."""

    path = _fixed_direct_url_path()
    raw, _ = _read_stable_file(path, "installed archive metadata")
    try:
        value = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise EvidenceError("installed archive metadata is invalid") from exc
    if not isinstance(value, Mapping) or not isinstance(value.get("url"), str):
        raise EvidenceError("installed archive source is unavailable")
    parsed = urlsplit(value["url"])
    if (
        parsed.scheme != "file"
        or parsed.netloc not in {"", "localhost"}
        or parsed.query
        or parsed.fragment
    ):
        raise EvidenceError("installed archive source is not a local checkout")
    raw_candidate = Path(unquote(parsed.path)).expanduser()
    if _has_symlink_component(raw_candidate):
        raise EvidenceError("installed archive source is unsafe")
    try:
        candidate = raw_candidate.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise EvidenceError("installed archive source is unsafe") from exc
    if not candidate.is_dir():
        raise EvidenceError("installed archive source is unsafe")
    return candidate


def _fixed_dom_capture_path(stage: str, run_id: str) -> Path:
    """Find exactly one trusted external DOM receipt for a stage/run pair."""

    if stage not in STAGES or _HEX.fullmatch(run_id) is None:
        raise EvidenceError("DOM capture binding is invalid")
    if _has_symlink_component(_FIXED_DOM_CAPTURE_ROOT):
        raise EvidenceError("DOM capture root is symlinked")
    try:
        candidates = sorted(_FIXED_DOM_CAPTURE_ROOT.glob("*.json"), key=str)
    except OSError as exc:
        raise EvidenceError("DOM capture root is unavailable") from exc
    matches: list[Path] = []
    for candidate in candidates:
        if candidate.is_symlink() or not candidate.is_file():
            raise EvidenceError("DOM capture path is unsafe")
        try:
            artifact, _, _ = _read_sealed_artifact(
                candidate, DOM_CAPTURE_SCHEMA, "DOM capture"
            )
        except EvidenceError:
            continue
        if artifact.get("stage") == stage and artifact.get("run_id") == run_id:
            matches.append(candidate)
    if len(matches) != 1:
        raise EvidenceError("trusted DOM capture is unavailable")
    return matches[0]


def _digest(value: object) -> str:
    return canonical_json_sha256_strict(value)


def _archive_projection(runtime: Mapping[str, Any]) -> dict[str, Any]:
    return {key: runtime[key] for key in _ARCHIVE_KEYS if key in runtime}


def _id(value: object, label: str) -> str:
    if not isinstance(value, str) or _HEX.fullmatch(value) is None:
        raise EvidenceError(f"{label} is not sha256")
    return value


def _utc(value: object, label: str) -> datetime:
    if not isinstance(value, str):
        raise EvidenceError(f"{label} is not UTC")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise EvidenceError(f"{label} is not UTC") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise EvidenceError(f"{label} is not UTC")
    return parsed.astimezone(UTC)


def _sealed(payload: Mapping[str, Any], schema: str, label: str) -> dict[str, Any]:
    if (
        payload.get("schema") != schema
        or payload.get("namespace") != "recall-distillation"
    ):
        raise EvidenceError(f"{label} schema mismatch")
    unsigned = {key: value for key, value in payload.items() if key != "seal_sha256"}
    if payload.get("seal_sha256") != _digest(unsigned):
        raise EvidenceError(f"{label} seal mismatch")
    return dict(payload)


def _safe_json(path: Path, label: str) -> dict[str, Any]:
    try:
        raw, _ = _read_stable_file(path, label)
    except OSError as exc:
        raise EvidenceError(f"{label} unreadable") from exc
    try:
        value = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"{label} invalid JSON") from exc
    if not isinstance(value, dict):
        raise EvidenceError(f"{label} is not object")
    return value


def _safe_json_raw(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    raw, _ = _read_stable_file(path, label)
    try:
        value = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"{label} invalid JSON") from exc
    if not isinstance(value, dict):
        raise EvidenceError(f"{label} is not object")
    return value, raw


def _validate_artifact_identity(
    artifact: Mapping[str, Any],
    path: Path | None,
    label: str,
    *,
    raw: bytes | None = None,
) -> dict[str, Any]:
    """Recompute an immutable artifact identity from its canonical content.

    ``store.read_sealed`` intentionally verifies only the seal.  A caller can
    therefore manufacture a correctly sealed object with an arbitrary
    ``artifact_id`` unless this second, content-addressed check is performed.
    The identity excludes both self-referential fields, exactly as
    ``write_immutable`` computes it; the seal includes ``artifact_id``.
    """

    artifact_id = artifact.get("artifact_id")
    if not isinstance(artifact_id, str) or _HEX.fullmatch(artifact_id) is None:
        raise EvidenceError(f"{label} artifact id is invalid")
    unsigned_identity = {
        key: value
        for key, value in artifact.items()
        if key not in {"artifact_id", "seal_sha256"}
    }
    if artifact_id != _digest(unsigned_identity):
        raise EvidenceError(f"{label} artifact id/content mismatch")
    unsigned_seal = {
        key: value for key, value in artifact.items() if key != "seal_sha256"
    }
    seal = artifact.get("seal_sha256")
    if not isinstance(seal, str) or _HEX.fullmatch(seal) is None:
        raise EvidenceError(f"{label} seal is invalid")
    if seal != _digest(unsigned_seal):
        raise EvidenceError(f"{label} seal mismatch")
    if path is not None and path.stem != artifact_id:
        raise EvidenceError(f"{label} path/id mismatch")
    if raw is not None and raw != canonical_json_line_bytes_strict(dict(artifact)):
        raise EvidenceError(f"{label} bytes are not canonical")
    return dict(artifact)


def _read_sealed_artifact(
    path: Path, schema: str, label: str
) -> tuple[dict[str, Any], bytes, dict[str, int]]:
    """Read, decode, seal-check, and content-address one immutable file.

    This is deliberately independent of ``store.read_sealed``.  The raw bytes
    used for the file digest, canonical identity, and decoded object all come
    from one stable descriptor read, avoiding a seal/id from one read paired
    with bytes from another.
    """

    if _has_symlink_component(path):
        raise EvidenceError(f"{label} path unsafe")
    raw, metadata = _read_stable_file(path, label)
    try:
        value = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"{label} invalid JSON") from exc
    if not isinstance(value, dict):
        raise EvidenceError(f"{label} is not object")
    if value.get("schema") != schema or value.get("namespace") != "recall-distillation":
        raise EvidenceError(f"{label} schema mismatch")
    _validate_artifact_identity(value, path, label, raw=raw)
    return value, raw, metadata


def _read_sealed_state(path: Path, schema: str, label: str) -> dict[str, Any]:
    """Read a mutable sealed state file from one stable descriptor snapshot."""

    if _has_symlink_component(path):
        raise EvidenceError(f"{label} path unsafe")
    raw, _ = _read_stable_file(path, label)
    try:
        value = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"{label} invalid JSON") from exc
    if not isinstance(value, dict):
        raise EvidenceError(f"{label} is not object")
    try:
        return dict(store.verify_seal(value, schema=schema))
    except store.DistillationStoreError as exc:
        raise EvidenceError(f"{label} seal is invalid") from exc


def _artifact_ref_values(artifact: Mapping[str, Any], body: bytes) -> dict[str, str]:
    return {
        "artifact_id": str(artifact["artifact_id"]),
        "file_sha256": hashlib.sha256(body).hexdigest(),
        "seal_sha256": str(artifact["seal_sha256"]),
    }


def _read_dom_capture(
    path: Path, *, stage: str, run_id: str, test_only: bool = False
) -> tuple[dict[str, Any], bytes, dict[str, str]]:
    """Read a trusted browser capture receipt and return its bound projection."""

    if test_only:
        dom, raw = _safe_json_raw(path, "test DOM capture")
        producer = dom.get("producer")
        if (
            dom.get("kind") != "browser-dom-capture"
            or dom.get("synthetic_fixture") is not False
            or not isinstance(producer, Mapping)
            or set(producer) != {"name", "version"}
            or producer.get("name") != "chronovisor-browser"
            or isinstance(producer.get("version"), bool)
            or not isinstance(producer.get("version"), int)
            or not isinstance(dom.get("html_sha256"), str)
            or _HEX.fullmatch(dom["html_sha256"]) is None
        ):
            raise EvidenceError("test DOM capture is invalid")
        return dom, raw, {
            "capture_artifact_id": hashlib.sha256(raw).hexdigest(),
            "capture_file_sha256": hashlib.sha256(raw).hexdigest(),
            "capture_seal_sha256": "0" * 64,
        }
    if path.parent.resolve() != _FIXED_DOM_CAPTURE_ROOT.resolve():
        raise EvidenceError("DOM capture is outside the managed authority")
    artifact, raw, _ = _read_sealed_artifact(path, DOM_CAPTURE_SCHEMA, "DOM capture")
    producer = artifact.get("producer")
    producer_version = producer.get("version") if isinstance(producer, Mapping) else None
    if (
        set(artifact)
        != {
            "schema",
            "namespace",
            "artifact_id",
            "seal_sha256",
            "kind",
            "stage",
            "run_id",
            "captured_at",
            "synthetic_fixture",
            "producer",
            "html_sha256",
        }
        or artifact.get("kind") != "browser-dom-capture"
        or artifact.get("synthetic_fixture") is not False
        or artifact.get("stage") != stage
        or artifact.get("run_id") != run_id
        or not isinstance(artifact.get("captured_at"), str)
        or not isinstance(producer, Mapping)
        or set(producer) != {"name", "version"}
        or producer.get("name") != "chronovisor-browser"
        or isinstance(producer.get("version"), bool)
        or not isinstance(producer.get("version"), int)
        or not isinstance(producer_version, int)
        or producer_version < 1
        or not isinstance(artifact.get("html_sha256"), str)
        or _HEX.fullmatch(artifact["html_sha256"]) is None
    ):
        raise EvidenceError("trusted DOM capture schema is invalid")
    _utc(artifact["captured_at"], "DOM capture time")
    return artifact, raw, {
        "capture_artifact_id": str(artifact["artifact_id"]),
        "capture_file_sha256": hashlib.sha256(raw).hexdigest(),
        "capture_seal_sha256": str(artifact["seal_sha256"]),
    }


def _stat_identity(value: os.stat_result) -> dict[str, int]:
    """Return only stable, non-sensitive file identity fields."""

    return {
        "dev": int(value.st_dev),
        "ino": int(value.st_ino),
        "mode": int(value.st_mode),
        "size": int(value.st_size),
        "mtime_ns": int(value.st_mtime_ns),
        "ctime_ns": int(value.st_ctime_ns),
    }


def _same_stat(left: Mapping[str, int], right: Mapping[str, int]) -> bool:
    return left == right


def _read_stable_file(
    path: Path, label: str, *, max_bytes: int = MAX_BYTES
) -> tuple[bytes, dict[str, int]]:
    """Read a regular file through a descriptor-pinned, double-read boundary."""

    raw_candidate = Path(os.path.abspath(os.fspath(Path(path).expanduser())))
    symlink_components = [
        part
        for part in (raw_candidate, *raw_candidate.parents)
        if part.is_symlink()
    ]
    if any(part not in {Path("/var"), Path("/tmp")} for part in symlink_components):
        raise EvidenceError(f"{label} path unsafe")
    try:
        candidate = raw_candidate.resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise EvidenceError(f"{label} path unsafe") from exc
    if not candidate.is_absolute() or len(candidate.parts) < 2:
        raise EvidenceError(f"{label} path unsafe")
    components = candidate.parts
    file_name = components[-1]
    if file_name in {"", ".", ".."}:
        raise EvidenceError(f"{label} path unsafe")

    def read_all(descriptor: int) -> bytes:
        try:
            os.lseek(descriptor, 0, os.SEEK_SET)
        except OSError as exc:
            raise EvidenceError(f"{label} is not seekable") from exc
        chunks: list[bytes] = []
        remaining = max_bytes + 1
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        body = b"".join(chunks)
        if len(body) > max_bytes:
            raise EvidenceError(f"{label} too large")
        return body

    directory_flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        directory_flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        directory_flags |= os.O_NOFOLLOW
    file_flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        file_flags |= os.O_NOFOLLOW
    directory_fds: list[tuple[Path, int, dict[str, int]]] = []
    descriptor: int | None = None
    final_descriptor: int | None = None
    try:
        current_fd = os.open(os.sep, directory_flags)
        directory_fds.append((Path(os.sep), current_fd, _stat_identity(os.fstat(current_fd))))
        prefix = Path(os.sep)
        for component in components[1:-1]:
            next_fd = os.open(component, directory_flags, dir_fd=current_fd)
            prefix /= component
            directory_fds.append((prefix, next_fd, _stat_identity(os.fstat(next_fd))))
            current_fd = next_fd
        parent_fd = current_fd
        before_raw = os.stat(file_name, dir_fd=parent_fd, follow_symlinks=False)
        if stat.S_ISLNK(before_raw.st_mode) or not stat.S_ISREG(before_raw.st_mode):
            raise EvidenceError(f"{label} path unsafe")
        before = _stat_identity(before_raw)
        if before["size"] > max_bytes:
            raise EvidenceError(f"{label} too large")
        descriptor = os.open(file_name, file_flags, dir_fd=parent_fd)
        opened = _stat_identity(os.fstat(descriptor))
        if not _same_stat(before, opened):
            raise EvidenceError(f"{label} changed before read")
        first_body = read_all(descriptor)
        mid = _stat_identity(os.fstat(descriptor))
        second_body = read_all(descriptor)
        after = _stat_identity(os.fstat(descriptor))
        final_raw = os.stat(file_name, dir_fd=parent_fd, follow_symlinks=False)
        final = _stat_identity(final_raw)
        final_descriptor = os.open(file_name, file_flags, dir_fd=parent_fd)
        final_opened = _stat_identity(os.fstat(final_descriptor))
        final_body = read_all(final_descriptor)
        final_closed = _stat_identity(os.fstat(final_descriptor))
        parent_final = _stat_identity(os.fstat(parent_fd))
        prefix_changed = any(
            _stat_identity(os.fstat(directory_fd)) != identity
            or _stat_identity(os.stat(prefix_path, follow_symlinks=False)) != identity
            for prefix_path, directory_fd, identity in directory_fds
        )
    except EvidenceError:
        raise
    except OSError as exc:
        raise EvidenceError(f"{label} unreadable") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if final_descriptor is not None:
            os.close(final_descriptor)
        for _prefix, directory_fd, _identity in reversed(directory_fds):
            os.close(directory_fd)
    if (
        first_body != second_body
        or second_body != final_body
        or not _same_stat(before, mid)
        or not _same_stat(before, after)
        or not _same_stat(before, final)
        or not _same_stat(before, opened)
        or not _same_stat(before, final_opened)
        or not _same_stat(before, final_closed)
        or parent_final != directory_fds[-1][2]
        or prefix_changed
    ):
        raise EvidenceError(f"{label} changed during read")
    return first_body, before


def _read_stable_sha256(
    path: Path, label: str, *, max_bytes: int = MAX_PROCESS_BYTES
) -> tuple[str, dict[str, int]]:
    """Hash a regular file through pinned descriptors without buffering it."""

    raw_candidate = Path(os.path.abspath(os.fspath(Path(path).expanduser())))
    symlink_components = [
        part
        for part in (raw_candidate, *raw_candidate.parents)
        if part.is_symlink()
    ]
    if any(part not in {Path("/var"), Path("/tmp")} for part in symlink_components):
        raise EvidenceError(f"{label} path unsafe")
    try:
        candidate = raw_candidate.resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise EvidenceError(f"{label} path unsafe") from exc
    if not candidate.is_absolute() or len(candidate.parts) < 2:
        raise EvidenceError(f"{label} path unsafe")
    components = candidate.parts
    file_name = components[-1]
    if file_name in {"", ".", ".."}:
        raise EvidenceError(f"{label} path unsafe")

    def digest_descriptor(descriptor: int) -> tuple[str, int]:
        try:
            os.lseek(descriptor, 0, os.SEEK_SET)
        except OSError as exc:
            raise EvidenceError(f"{label} is not seekable") from exc
        digest = hashlib.sha256()
        total = 0
        while True:
            try:
                chunk = os.read(descriptor, 1024 * 1024)
            except OSError as exc:
                raise EvidenceError(f"{label} unreadable") from exc
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise EvidenceError(f"{label} too large")
            digest.update(chunk)
        return digest.hexdigest(), total

    directory_flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        directory_flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        directory_flags |= os.O_NOFOLLOW
    file_flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        file_flags |= os.O_NOFOLLOW
    directory_fds: list[tuple[Path, int, dict[str, int]]] = []
    descriptor: int | None = None
    final_descriptor: int | None = None
    try:
        current_fd = os.open(os.sep, directory_flags)
        directory_fds.append((Path(os.sep), current_fd, _stat_identity(os.fstat(current_fd))))
        prefix = Path(os.sep)
        for component in components[1:-1]:
            next_fd = os.open(component, directory_flags, dir_fd=current_fd)
            prefix /= component
            directory_fds.append((prefix, next_fd, _stat_identity(os.fstat(next_fd))))
            current_fd = next_fd
        parent_fd = current_fd
        before_raw = os.stat(file_name, dir_fd=parent_fd, follow_symlinks=False)
        if stat.S_ISLNK(before_raw.st_mode) or not stat.S_ISREG(before_raw.st_mode):
            raise EvidenceError(f"{label} path unsafe")
        before = _stat_identity(before_raw)
        if before["size"] > max_bytes:
            raise EvidenceError(f"{label} too large")
        descriptor = os.open(file_name, file_flags, dir_fd=parent_fd)
        opened = _stat_identity(os.fstat(descriptor))
        if not _same_stat(before, opened):
            raise EvidenceError(f"{label} changed before read")
        first_digest, first_size = digest_descriptor(descriptor)
        mid = _stat_identity(os.fstat(descriptor))
        second_digest, second_size = digest_descriptor(descriptor)
        after = _stat_identity(os.fstat(descriptor))
        final_raw = os.stat(file_name, dir_fd=parent_fd, follow_symlinks=False)
        if stat.S_ISLNK(final_raw.st_mode) or not stat.S_ISREG(final_raw.st_mode):
            raise EvidenceError(f"{label} path unsafe")
        final = _stat_identity(final_raw)
        final_descriptor = os.open(file_name, file_flags, dir_fd=parent_fd)
        final_opened = _stat_identity(os.fstat(final_descriptor))
        final_digest, final_size = digest_descriptor(final_descriptor)
        final_closed = _stat_identity(os.fstat(final_descriptor))
        parent_final = _stat_identity(os.fstat(parent_fd))
        prefix_changed = any(
            _stat_identity(os.fstat(directory_fd)) != identity
            or _stat_identity(os.stat(prefix_path, follow_symlinks=False)) != identity
            for prefix_path, directory_fd, identity in directory_fds
        )
    except EvidenceError:
        raise
    except OSError as exc:
        raise EvidenceError(f"{label} unreadable") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if final_descriptor is not None:
            os.close(final_descriptor)
        for _prefix, directory_fd, _identity in reversed(directory_fds):
            os.close(directory_fd)
    if (
        first_digest != second_digest
        or second_digest != final_digest
        or first_size != second_size
        or second_size != final_size
        or first_size != before["size"]
        or not _same_stat(before, mid)
        or not _same_stat(before, after)
        or not _same_stat(before, final)
        or not _same_stat(before, opened)
        or not _same_stat(before, final_opened)
        or not _same_stat(before, final_closed)
        or parent_final != directory_fds[-1][2]
        or prefix_changed
    ):
        raise EvidenceError(f"{label} changed during read")
    return first_digest, before


def _path_metadata(path: Path, label: str) -> dict[str, Any]:
    """Capture path identity without following a symlink at the path itself."""

    try:
        before = path.lstat()
        if stat.S_ISLNK(before.st_mode):
            raise EvidenceError(f"{label} path unsafe")
        after = path.lstat()
    except EvidenceError:
        raise
    except OSError as exc:
        raise EvidenceError(f"{label} unavailable") from exc
    first = _stat_identity(before)
    final = _stat_identity(after)
    if first != final:
        raise EvidenceError(f"{label} changed during identity read")
    if stat.S_ISREG(before.st_mode):
        body, _ = _read_stable_file(path, label)
        payload_sha = hashlib.sha256(body).hexdigest()
    elif stat.S_ISDIR(before.st_mode):
        entries: list[list[Any]] = []
        try:
            children = sorted(path.iterdir(), key=lambda item: item.name)
        except OSError as exc:
            raise EvidenceError(f"{label} unreadable") from exc
        for child in children:
            child_stat = child.lstat()
            if stat.S_ISLNK(child_stat.st_mode):
                raise EvidenceError(f"{label} contains symlink")
            child_entry: list[Any] = [child.name, _stat_identity(child_stat)]
            if stat.S_ISREG(child_stat.st_mode):
                body, _ = _read_stable_file(child, f"{label} child")
                child_entry.append(hashlib.sha256(body).hexdigest())
            entries.append(child_entry)
        payload_sha = _digest(entries)
    else:
        raise EvidenceError(f"{label} path unsafe")
    return {"path": str(path), "lstat": first, "bytes_sha256": payload_sha}


def _has_symlink_component(path: Path) -> bool:
    candidate = path.expanduser()
    try:
        return any(part.is_symlink() for part in (candidate, *candidate.parents))
    except (OSError, RuntimeError):
        return True


def _readonly_chain_snapshot(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read one checkpoint-bound chain snapshot without creating its lock file."""
    lock_path = path.with_suffix(path.suffix + ".lock")
    if _has_symlink_component(path) or _has_symlink_component(lock_path):
        raise EvidenceError("runtime chain lock is symlinked")
    try:
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(lock_path, flags)
    except OSError as exc:
        raise EvidenceError("runtime chain lock is unavailable") from exc
    try:
        fcntl.flock(descriptor, fcntl.LOCK_SH | fcntl.LOCK_NB)
        head = store._read_chain_checkpoint(path)
        if head is None:
            raise EvidenceError("runtime chain checkpoint is absent")
        rows = store._read_chain_locked(path, head)
        previous = ""
        for index, row in enumerate(rows):
            unsigned = {
                key: value for key, value in row.items() if key != "record_sha256"
            }
            if row.get("previous_sha256") != previous or row.get(
                "record_sha256"
            ) != _digest(unsigned):
                raise EvidenceError(f"runtime chain mismatch at row {index}")
            previous = str(row["record_sha256"])
        if len(rows) != head["records"] or previous != head["head_sha256"]:
            raise EvidenceError("runtime chain checkpoint mismatch")
        return rows, dict(head)
    except store.DistillationStoreError as exc:
        raise EvidenceError("runtime chain snapshot is invalid") from exc
    finally:
        os.close(descriptor)


def _protected_file_state(root: Path) -> str:
    if _has_symlink_component(root):
        raise EvidenceError("protected runtime root is symlinked")
    directory = store.distillation_dir(root)
    paths = [
        directory / store.STATE_FILE,
        *(directory / filename for filename in store.POINTER_FILES.values()),
        directory / "shadow-observation-receipts.jsonl",
        store._chain_checkpoint_path(directory / "shadow-observation-receipts.jsonl"),
    ]
    state: list[tuple[str, int, int, int, int, int] | tuple[str, None]] = []
    for path in paths:
        try:
            stat = path.lstat()
        except FileNotFoundError:
            state.append((path.name, None))
            continue
        if path.is_symlink():
            raise EvidenceError("protected runtime path is symlinked")
        state.append(
            (
                path.name,
                stat.st_dev,
                stat.st_ino,
                stat.st_size,
                stat.st_mtime_ns,
                stat.st_ctime_ns,
            )
        )
    return _digest(state)


def _ignored_source_code(relative: str) -> bool:
    """Whether an ignored path could alter the protected R7 import surface."""
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        return True
    protected = (len(path.parts) >= 2 and path.parts[:2] == ("src", "chronovisor")) or (
        len(path.parts) >= 2
        and path.parts[0] == "scripts"
        and path.parts[1].startswith("recall_r7")
    )
    if not protected:
        return False
    # Bytecode/type/test caches cannot change the R7 import surface. Every
    # other ignored path under a protected namespace is treated as source.
    return not any(
        component in {"__pycache__", ".pytest_cache", ".mypy_cache"}
        for component in path.parts
    )


def _git_run(source: Path, arguments: Sequence[str]) -> bytes:
    """Run git with all ambient ``GIT_*`` variables removed."""

    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    env.update(
        {
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_SYSTEM": os.devnull,
            "GIT_TERMINAL_PROMPT": "0",
        }
    )
    try:
        result = subprocess.run(
            [str(_TRUSTED_GIT), "--no-optional-locks", *arguments],
            cwd=source,
            check=True,
            capture_output=True,
            env=env,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise EvidenceError("source identity unavailable") from exc
    return result.stdout


def _git_path(source: Path, argument: str) -> Path:
    query = (
        ["--git-path", argument.partition("=")[2]]
        if argument.startswith("--git-path=")
        else [argument]
    )
    raw = _git_run(source, ["rev-parse", "--path-format=absolute", *query])
    value = raw.rstrip(b"\r\n")
    if not value or b"\0" in value:
        raise EvidenceError("source git path is invalid")
    path = Path(os.fsdecode(value))
    if not path.is_absolute() or _has_symlink_component(path):
        raise EvidenceError("source git path is unsafe")
    return path


def _git_object(source: Path, object_name: str, object_type: str) -> bytes:
    """Read one referenced Git object with ambient Git configuration removed."""

    if not _HEX40(object_name) or object_type not in {"commit", "tree", "blob"}:
        raise EvidenceError("source Git object is invalid")
    return _git_run(source, ["cat-file", object_type, object_name])


def _source_identity_once(source: Path) -> dict[str, Any]:
    try:
        if _has_symlink_component(source):
            raise EvidenceError("source path is symlinked")
        source = source.resolve(strict=True)
        if not source.is_dir():
            raise EvidenceError("source path is not a directory")
        commit = os.fsdecode(_git_run(source, ["rev-parse", "HEAD"]).strip())
        if _COMMIT.fullmatch(commit) is None:
            raise EvidenceError("source commit is invalid")
        status = _git_run(
            source, ["status", "--porcelain=v1", "--untracked-files=all", "-z"]
        )
        ignored = _git_run(
            source, ["ls-files", "-o", "-i", "--exclude-standard", "-z"]
        )
        indexed = _git_run(source, ["ls-files", "-s", "-z"])
        git_dir = _git_path(source, "--git-dir")
        worktree = _git_path(source, "--show-toplevel")
        index_path = _git_path(source, "--git-path=index")
        head_path = _git_path(source, "--git-path=HEAD")
    except EvidenceError:
        raise
    except (OSError, UnicodeError) as exc:
        raise EvidenceError("source identity unavailable") from exc
    if worktree != source:
        raise EvidenceError("git worktree mismatch")
    if not git_dir.is_dir() or not worktree.is_dir():
        raise EvidenceError("source git path is not a directory")
    if status:
        raise EvidenceError("source commit drift or dirty checkout")
    ignored_paths = tuple(os.fsdecode(path) for path in ignored.split(b"\0") if path)
    if any(_ignored_source_code(path) for path in ignored_paths):
        raise EvidenceError("ignored protected source drift")

    # The four git inputs are captured as bytes and metadata.  Their values are
    # retained only as digests; the receipt never contains repository contents.
    try:
        head_bytes, head_stat = _read_stable_file(head_path, "git HEAD")
        index_bytes, index_stat = _read_stable_file(index_path, "git index")
    except EvidenceError:
        raise
    # Bind the referenced commit/tree objects as well as the index/worktree.
    # This catches replacement of a reachable object under .git/objects even
    # when the checkout bytes and index still look unchanged.
    head_object = _git_object(source, commit, "commit")
    tree_id = os.fsdecode(_git_run(source, ["rev-parse", "HEAD^{tree}"]).strip())
    if not _HEX40(tree_id):
        raise EvidenceError("source tree object is invalid")
    tree_object = _git_object(source, tree_id, "tree")
    metadata = {
        "git_dir": _path_metadata(git_dir, "git dir"),
        "worktree": _path_metadata(worktree, "git worktree"),
        "index": {
            "path": str(index_path),
            "lstat": index_stat,
            "bytes_sha256": hashlib.sha256(index_bytes).hexdigest(),
        },
        "head": {
            "path": str(head_path),
            "lstat": head_stat,
            "bytes_sha256": hashlib.sha256(head_bytes).hexdigest(),
        },
        "status_sha256": hashlib.sha256(status).hexdigest(),
        "tree_sha256": hashlib.sha256(tree_object).hexdigest(),
        "head_object_sha256": hashlib.sha256(head_object).hexdigest(),
        "tree_object_sha256": hashlib.sha256(tree_object).hexdigest(),
    }
    tree: list[tuple[str, str, str]] = []
    bytes_tree: list[list[str]] = []
    blob_objects: list[tuple[str, bytes]] = []
    for record in indexed.split(b"\0"):
        if not record:
            continue
        header, separator, path = record.partition(b"\t")
        fields = header.split()
        if separator != b"\t" or len(fields) != 3:
            raise EvidenceError("source index is malformed")
        mode, blob, stage = (os.fsdecode(field) for field in fields)
        relative = os.fsdecode(path)
        relative_path = Path(relative)
        candidate = source / relative_path
        if (
            mode not in {"100644", "100755"}
            or stage != "0"
            or not _HEX40(blob)
            or relative_path.is_absolute()
            or ".." in relative_path.parts
            or _has_symlink_component(candidate)
        ):
            raise EvidenceError("source tracked path is unsafe")
        body, file_stat = _read_stable_file(candidate, "source tracked bytes")
        # A tracked blob is SHA-1 over the Git blob header and bytes.  Checking
        # it independently catches a status/index race and byte substitution.
        blob_digest = hashlib.sha1(
            f"blob {len(body)}\0".encode() + body
        ).hexdigest()
        if blob_digest != blob:
            raise EvidenceError("source tracked bytes do not match index blob")
        if _git_object(source, blob, "blob") != body:
            raise EvidenceError("source Git blob does not match worktree bytes")
        blob_objects.append((blob, body))
        tree.append((mode, blob, relative))
        bytes_tree.append(
            [mode, relative, hashlib.sha256(body).hexdigest(), _digest(file_stat)]
        )
    if not tree:
        raise EvidenceError("source index is empty")
    try:
        head_final, head_final_stat = _read_stable_file(head_path, "git HEAD")
        index_final, index_final_stat = _read_stable_file(index_path, "git index")
        status_final = _git_run(
            source, ["status", "--porcelain=v1", "--untracked-files=all", "-z"]
        )
        indexed_final = _git_run(source, ["ls-files", "-s", "-z"])
        head_object_final = _git_object(source, commit, "commit")
        tree_id_final = os.fsdecode(
            _git_run(source, ["rev-parse", "HEAD^{tree}"]).strip()
        )
        tree_object_final = _git_object(source, tree_id_final, "tree")
        blob_objects_final = [
            (blob, _git_object(source, blob, "blob")) for blob, _ in blob_objects
        ]
    except EvidenceError:
        raise
    if (
        head_final != head_bytes
        or index_final != index_bytes
        or head_final_stat != head_stat
        or index_final_stat != index_stat
        or status_final != status
        or indexed_final != indexed
        or head_object_final != head_object
        or tree_id_final != tree_id
        or tree_object_final != tree_object
        or any(body != expected for (_, body), (_, expected) in zip(blob_objects, blob_objects_final, strict=True))
    ):
        raise EvidenceError("source checkout changed during identity read")
    return {
        "source_commit": commit,
        "source_clean": "true",
        "source_tree_sha256": _digest(tree),
        "source_bytes_sha256": _digest(
            [[mode, relative, digest] for mode, relative, digest, _ in bytes_tree]
        ),
        "git": metadata,
        "tracked": bytes_tree,
    }


def _HEX40(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{40}", value) is not None


def source_identity(source: Path) -> dict[str, Any]:
    """Seal a clean tracked tree after before/after/final byte rechecks."""

    before = _source_identity_once(source)
    after = _source_identity_once(source)
    final = _source_identity_once(source)
    if before != after or after != final:
        raise EvidenceError("source identity changed during capture")
    return before


_source_identity = source_identity


class _ProcBsdInfo(ctypes.Structure):
    _fields_ = [
        ("flags", ctypes.c_uint32),
        ("status", ctypes.c_uint32),
        ("xstatus", ctypes.c_uint32),
        ("process_id", ctypes.c_uint32),
        ("parent_process_id", ctypes.c_uint32),
        ("uid", ctypes.c_uint32),
        ("gid", ctypes.c_uint32),
        ("ruid", ctypes.c_uint32),
        ("rgid", ctypes.c_uint32),
        ("svuid", ctypes.c_uint32),
        ("svgid", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("command", ctypes.c_char * 16),
        ("name", ctypes.c_char * 32),
        ("nfiles", ctypes.c_uint32),
        ("pgid", ctypes.c_uint32),
        ("pjobc", ctypes.c_uint32),
        ("e_tdev", ctypes.c_uint32),
        ("e_tpgid", ctypes.c_uint32),
        ("nice", ctypes.c_int32),
        ("start_seconds", ctypes.c_uint64),
        ("start_microseconds", ctypes.c_uint64),
    ]


def _darwin_process_probe(pid: int) -> tuple[Path, int, int, int]:
    """Read executable path and start tuple from macOS libproc."""

    if sys.platform != "darwin":
        raise EvidenceError("process executable identity mismatch: macOS required")
    try:
        libproc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
        path_buffer = ctypes.create_string_buffer(4096)
        proc_pidpath = libproc.proc_pidpath
        proc_pidpath.argtypes = (ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32)
        proc_pidpath.restype = ctypes.c_int
        path_length = proc_pidpath(pid, path_buffer, len(path_buffer))
        if path_length <= 0 or path_length >= len(path_buffer):
            raise EvidenceError("native process image is unavailable")
        info = _ProcBsdInfo()
        proc_pidinfo = libproc.proc_pidinfo
        proc_pidinfo.argtypes = (
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_uint64,
            ctypes.c_void_p,
            ctypes.c_int,
        )
        proc_pidinfo.restype = ctypes.c_int
        # PROC_PIDTBSDINFO is the documented BSD process identity flavor.
        if proc_pidinfo(
            pid, 3, 0, ctypes.byref(info), ctypes.sizeof(info)
        ) != ctypes.sizeof(info):
            raise EvidenceError("native process start identity is unavailable")
        reported_raw = bytes(path_buffer.value)
        reported_path = Path(os.fsdecode(reported_raw))
        if _has_symlink_component(reported_path):
            raise EvidenceError("native process executable path is unsafe")
        reported = reported_path.resolve(strict=True)
    except EvidenceError:
        raise
    except (OSError, AttributeError, UnicodeError) as exc:
        raise EvidenceError("process identity unavailable") from exc
    process_id = int(info.process_id)
    seconds = int(info.start_seconds)
    micros = int(info.start_microseconds)
    if process_id <= 0 or seconds < 1 or not 0 <= micros < 1_000_000:
        raise EvidenceError("native process start identity is invalid")
    return reported, process_id, seconds, micros


def _validate_launchctl_output_fields(
    text: str, label: str
) -> dict[str, Any]:
    """Parse launchctl's closed grammar and retain root identity fields."""

    header = re.compile(r"^gui/([0-9]+)/([^\s=]+)\s*=\s*\{$")
    assignment = re.compile(r"^([^=]+?)\s*=\s*(.*)$")
    arrow = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]*\s*=>\s*.*$")
    header_label: str | None = None
    scopes: list[str] = []
    seen: dict[tuple[str, ...], set[str]] = {}
    top_level: dict[str, str] = {}

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line == "}":
            if not scopes or scopes[-1] == _LAUNCHCTL_ARGUMENTS_LIST_SCOPE:
                raise EvidenceError("launchd service output has an unmatched brace")
            scopes.pop()
            continue
        if line == ")":
            if not scopes or scopes[-1] != _LAUNCHCTL_ARGUMENTS_LIST_SCOPE:
                raise EvidenceError("launchd service output has an unmatched parenthesis")
            scopes.pop()
            continue
        header_match = header.fullmatch(line)
        if header_match is not None:
            if header_label is not None or scopes:
                raise EvidenceError("launchd service output has duplicate header")
            header_label = header_match.group(2)
            if header_label != label:
                raise EvidenceError("launchd service header label mismatch")
            scopes.append("__root__")
            continue
        if not scopes:
            raise EvidenceError("launchd service output has an unknown line")
        if scopes[-1] == _LAUNCHCTL_ARGUMENTS_LIST_SCOPE:
            if _LAUNCHCTL_ARGV_SCALAR.fullmatch(line) is None:
                raise EvidenceError("launchd service output has an invalid argv scalar")
            continue
        if arrow.fullmatch(line) is not None:
            if scopes[-1] not in _LAUNCHCTL_ENVIRONMENT_BLOCKS:
                raise EvidenceError("launchd service output has an unknown arrow")
            continue
        match = assignment.fullmatch(line)
        if match is not None:
            field = match.group(1).strip()
            value = match.group(2).strip()
            if field not in _LAUNCHCTL_FIELDS:
                raise EvidenceError("launchd service output contains an unknown field")
            if field == "program" and _LAUNCHCTL_ARGV_SCALAR.fullmatch(value) is None:
                raise EvidenceError("launchd service output has an invalid program")
            if field == "arguments" and value not in {"{", "("}:
                raise EvidenceError("launchd service output has an invalid arguments list")
            scope = tuple(scopes)
            fields = seen.setdefault(scope, set())
            if field in fields:
                raise EvidenceError("launchd service output contains a duplicate field")
            fields.add(field)
            if len(scopes) == 1:
                top_level[field] = value
            if value == "{":
                scopes.append(field)
            elif value == "(":
                scopes.append(_LAUNCHCTL_ARGUMENTS_LIST_SCOPE)
            continue
        if line in _LAUNCHCTL_BARE_LINES and scopes == ["__root__"]:
            continue
        if scopes[-1] in _LAUNCHCTL_BARE_BLOCKS:
            if scopes[-1] == "subdomains" and re.fullmatch(r"pid/[0-9]+", line):
                continue
            if scopes[-1] == "service stats" and re.fullmatch(
                r"[^=]+\([0-9]+ records\)", line
            ):
                continue
            if scopes[-1] == "security context" and re.fullmatch(
                r"(?:uid|euid|asid) (?:unset|[0-9]+)", line
            ):
                continue
            if scopes[-1] == "arguments":
                continue
        raise EvidenceError("launchd service output has an unknown line")
    if header_label is None or scopes:
        raise EvidenceError("launchd service output has an incomplete structure")
    return {"header_label": header_label, "top_level": top_level}


def _launchctl_probe(role: str, pid: int | None = None) -> dict[str, Any]:
    """Read one fixed launchd service and bind its canonical process PID.

    Formal producers pass no PID: launchctl is the authority and the process
    identity is derived from its output.  Parent/child fields are optional in
    ``launchctl print`` and are cross-checked against trusted ``ps`` lineage
    when a full process receipt is assembled.  The optional argument is
    retained solely for the owned test seam and is never used to choose a
    production process.
    """

    label = _SERVICE_LABELS.get(role)
    if label is None:
        raise EvidenceError("launchd service role is not approved")
    if sys.platform != "darwin":
        raise EvidenceError("launchd service identity mismatch: macOS required")
    target = f"gui/{os.getuid()}/{label}"
    try:
        completed = subprocess.run(
            ["/bin/launchctl", "print", target],
            check=False,
            capture_output=True,
            timeout=5,
            env={
                key: value
                for key, value in os.environ.items()
                if not key.startswith("GIT_")
            },
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise EvidenceError("launchd service identity unavailable") from exc
    if completed.returncode != 0:
        raise EvidenceError("launchd service is not loaded")
    raw = completed.stdout
    if isinstance(raw, str):
        raw = raw.encode()
    if not isinstance(raw, bytes) or len(raw) == 0 or len(raw) > MAX_PROCESS_BYTES:
        raise EvidenceError("launchd service output is invalid")
    try:
        text = raw.decode()
    except UnicodeDecodeError as exc:
        raise EvidenceError("launchd service output is invalid") from exc
    parsed = _validate_launchctl_output_fields(text, label)
    header_label = parsed.get("header_label")
    top_level = parsed.get("top_level")
    if not isinstance(header_label, str) or not isinstance(top_level, Mapping):
        raise EvidenceError("launchd service identity parse is invalid")

    def parse_pid_field(field: str, *, required: bool) -> int | None:
        value = top_level.get(field)
        if value is None:
            if required:
                raise EvidenceError("launchd service/PID binding mismatch")
            return None
        if not isinstance(value, str) or re.fullmatch(r"[0-9]+", value) is None:
            raise EvidenceError("launchd service/PID binding mismatch")
        parsed_pid = int(value)
        if parsed_pid <= 0:
            raise EvidenceError("launchd service/PID binding mismatch")
        return parsed_pid

    process_pid = parse_pid_field("pid", required=True)
    parent_pid = parse_pid_field("parent pid", required=False)
    child_pid = parse_pid_field("child pid", required=False)
    state = top_level.get("state")
    identity_label_matches = [
        value
        for field in ("label", "service")
        if isinstance(value := top_level.get(field), str)
    ]
    role_value = top_level.get("role")
    if (
        len(identity_label_matches) > 1
        or any(value != header_label for value in identity_label_matches)
        or (role_value is not None and role_value != role)
        or isinstance(pid, bool)
        or (pid is not None and (not isinstance(pid, int) or process_pid != pid))
        or (child_pid is not None and child_pid != process_pid)
        or state != "running"
    ):
        raise EvidenceError("launchd service/PID binding mismatch")
    return {
        "role": role,
        "domain": f"gui/{os.getuid()}",
        "label": label,
        "state": state,
        "pid": process_pid,
        "parent_pid": parent_pid,
        "child_pid": child_pid,
        "captured_at": datetime.now(UTC).isoformat(),
        "raw_output_sha256": hashlib.sha256(raw).hexdigest(),
    }


def _ps_process_lineage(pid: int) -> tuple[int, int]:
    """Read one process's current PID/PPID tuple from trusted ``/bin/ps``."""

    if sys.platform != "darwin" or isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        raise EvidenceError("process lineage is unavailable")
    try:
        completed = subprocess.run(
            ["/bin/ps", "-p", str(pid), "-o", "pid=,ppid="],
            check=False,
            capture_output=True,
            timeout=5,
            env={
                key: value
                for key, value in os.environ.items()
                if not key.startswith("GIT_")
            },
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise EvidenceError("process lineage is unavailable") from exc
    if completed.returncode != 0:
        raise EvidenceError("process lineage is unavailable")
    raw = completed.stdout
    if isinstance(raw, str):
        raw = raw.encode()
    if not isinstance(raw, bytes) or len(raw) == 0 or len(raw) > MAX_PROCESS_BYTES:
        raise EvidenceError("process lineage is invalid")
    try:
        lines = [line.strip() for line in raw.decode().splitlines() if line.strip()]
    except UnicodeDecodeError as exc:
        raise EvidenceError("process lineage is invalid") from exc
    if len(lines) != 1:
        raise EvidenceError("process lineage is invalid")
    match = re.fullmatch(r"([0-9]+)\s+([0-9]+)", lines[0])
    if match is None:
        raise EvidenceError("process lineage is invalid")
    reported_pid, parent_pid = (int(match.group(index)) for index in (1, 2))
    if reported_pid != pid or parent_pid <= 0:
        raise EvidenceError("process lineage mismatch")
    return reported_pid, parent_pid


def _service_process_identity(
    service_role: str, expected_started_at: object | None = None
) -> dict[str, Any]:
    """Bind launchd's PID to libproc bytes and a separately read ps lineage."""

    service = _launchctl_probe(service_role)
    process_pid = service.get("pid")
    if isinstance(process_pid, bool) or not isinstance(process_pid, int) or process_pid <= 0:
        raise EvidenceError("launchd process PID is invalid")
    child_pid = service.get("child_pid")
    parent_pid = service.get("parent_pid")
    if child_pid is not None and (
        isinstance(child_pid, bool) or not isinstance(child_pid, int) or child_pid != process_pid
    ):
        raise EvidenceError("launchd child PID is invalid")
    if parent_pid is not None and (
        isinstance(parent_pid, bool) or not isinstance(parent_pid, int) or parent_pid <= 0
    ):
        raise EvidenceError("launchd parent PID is invalid")
    lineage_before = _ps_process_lineage(process_pid)
    executable, native_pid, _seconds, _micros = _darwin_process_probe(process_pid)
    if native_pid != process_pid:
        raise EvidenceError("launchd process identity mismatch")
    identity = _process_identity(
        executable,
        process_pid,
        expected_started_at,
        service_role,
    )
    # ``_process_identity`` rereads launchctl itself.  Compare the first and
    # final service snapshots so a stop/start between reads cannot be hidden.
    if not _same_process_identity(
        {**identity, "service": service},
        identity,
    ):
        raise EvidenceError("launchd process identity changed during read")
    lineage_after = _ps_process_lineage(process_pid)
    if lineage_after != lineage_before:
        raise EvidenceError("process lineage changed during identity read")
    final_service = identity.get("service")
    if not isinstance(final_service, Mapping):
        raise EvidenceError("launchd service identity is invalid")
    for snapshot in (service, final_service):
        snapshot_child = snapshot.get("child_pid")
        snapshot_parent = snapshot.get("parent_pid")
        if snapshot_child is not None and snapshot_child != process_pid:
            raise EvidenceError("launchd child PID mismatch")
        if snapshot_parent is not None and snapshot_parent != lineage_before[1]:
            raise EvidenceError("launchd parent PID mismatch")
    normalized_service = dict(final_service)
    normalized_service.pop("pid", None)
    normalized_service["parent_pid"] = lineage_before[1]
    normalized_service["child_pid"] = process_pid
    identity["service"] = normalized_service
    return identity


def _same_process_identity(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    """Compare process rereads while allowing their capture timestamp to move."""

    if set(left) != set(right):
        return False
    for key in left:
        if key != "service":
            if left.get(key) != right.get(key):
                return False
            continue
        left_service = left.get(key)
        right_service = right.get(key)
        if not isinstance(left_service, Mapping) or not isinstance(right_service, Mapping):
            return False
        if {
            service_key: left_service.get(service_key)
            for service_key in left_service
            if service_key != "captured_at"
        } != {
            service_key: right_service.get(service_key)
            for service_key in right_service
            if service_key != "captured_at"
        }:
            return False
        try:
            left_captured = _utc(left_service.get("captured_at"), "launchd capture time")
            right_captured = _utc(right_service.get("captured_at"), "launchd capture time")
        except EvidenceError:
            return False
        if abs((left_captured - right_captured).total_seconds()) > MAX_OBSERVATION_SKEW_SECONDS:
            return False
    return True


def _process_identity(
    executable: Path,
    pid: int,
    expected_started_at: object | None = None,
    service_role: str | None = None,
) -> dict[str, Any]:
    """Bind a positive PID to its native macOS image, start epoch and bytes."""

    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        raise EvidenceError("executable/PID is unsafe")
    if _has_symlink_component(executable):
        raise EvidenceError("executable/PID is unsafe")
    try:
        resolved = executable.resolve(strict=True)
    except OSError as exc:
        raise EvidenceError("executable/PID is unsafe") from exc
    first_sha256, first_stat = _read_stable_sha256(
        resolved, "executable", max_bytes=MAX_PROCESS_BYTES
    )
    reported, process_id, seconds, micros = _darwin_process_probe(pid)
    second_sha256, second_stat = _read_stable_sha256(
        resolved, "executable", max_bytes=MAX_PROCESS_BYTES
    )
    final_sha256, final_stat = _read_stable_sha256(
        resolved, "executable", max_bytes=MAX_PROCESS_BYTES
    )
    reported_final, process_id_final, seconds_final, micros_final = _darwin_process_probe(pid)
    if (
        first_sha256 != second_sha256
        or second_sha256 != final_sha256
        or first_stat != second_stat
        or second_stat != final_stat
        or reported_final != reported
        or process_id_final != process_id
        or seconds_final != seconds
        or micros_final != micros
    ):
        raise EvidenceError("executable changed during identity read")
    started_at = f"{seconds}.{micros:06d}"
    if expected_started_at is not None and (
        not isinstance(expected_started_at, str) or expected_started_at != started_at
    ):
        raise EvidenceError("process start identity mismatch")
    if process_id != pid or reported != resolved:
        raise EvidenceError("process executable identity mismatch")
    identity: dict[str, Any] = {
        "pid": pid,
        "started_at": started_at,
        "executable_path": str(resolved),
        "executable_lstat": first_stat,
        "executable_sha256": first_sha256,
    }
    if service_role is not None:
        service = _launchctl_probe(service_role, pid)
        if service.get("pid") != pid:
            raise EvidenceError("launchd process PID mismatch")
        child_pid = service.get("child_pid")
        if child_pid is not None and child_pid != pid:
            raise EvidenceError("launchd child PID mismatch")
        identity["service"] = service
    return identity


def _direct_url(path: Path, *, formal: bool = False) -> dict[str, Any]:
    """Derive archive identity, optionally requiring installed RECORD/Git binding."""

    raw, _ = _read_stable_file(path, "runtime direct_url")
    try:
        value = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise EvidenceError("runtime direct_url invalid JSON") from exc
    if not isinstance(value, Mapping):
        raise EvidenceError("runtime direct_url is not object")
    commit = (
        value.get("vcs_info", {}).get("commit_id")
        if isinstance(value.get("vcs_info"), Mapping)
        else None
    )
    if not isinstance(commit, str) or _COMMIT.fullmatch(commit) is None:
        raise EvidenceError("runtime archive commit unavailable")
    raw_sha = hashlib.sha256(raw).hexdigest()
    payload_sha = _digest(value)
    try:
        module_path = Path(__file__).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise EvidenceError("runtime module path is unsafe") from exc
    if _has_symlink_component(Path(__file__)):
        raise EvidenceError("runtime module path is unsafe")
    module_body, module_stat = _read_stable_file(module_path, "runtime module")
    try:
        distribution_version = metadata.version("chronovisor")
    except metadata.PackageNotFoundError as exc:
        raise EvidenceError("runtime distribution identity unavailable") from exc
    runtime: dict[str, Any] = {
        "archive_commit": commit,
        "direct_url_sha256": payload_sha,
        "direct_url_raw_sha256": raw_sha,
        "direct_url_payload_sha256": payload_sha,
        "module_path": str(module_path),
        "module_bytes_sha256": hashlib.sha256(module_body).hexdigest(),
        "module_lstat": module_stat,
        "distribution_name": "chronovisor",
        "distribution_version": distribution_version,
    }
    if not formal:
        return runtime
    fixed_direct_url = _fixed_direct_url_path()
    if path != fixed_direct_url:
        raise EvidenceError("runtime direct_url authority is not fixed")
    source_root = _fixed_source_root()
    try:
        distribution = metadata.distribution("chronovisor")
        distribution_metadata_path = getattr(distribution, "_path", None)
        if not isinstance(distribution_metadata_path, Path):
            raise EvidenceError("runtime distribution metadata is unavailable")
        distribution_root = distribution_metadata_path.parent.resolve(strict=True)
        record_path = distribution_metadata_path / "RECORD"
        if _has_symlink_component(record_path):
            raise EvidenceError("runtime distribution RECORD is symlinked")
        record_raw, _ = _read_stable_file(record_path, "runtime distribution RECORD")
        rows = list(csv.reader(record_raw.decode("utf-8").splitlines()))
    except (
        OSError,
        UnicodeError,
        csv.Error,
        TypeError,
        ValueError,
        metadata.PackageNotFoundError,
    ) as exc:
        raise EvidenceError("runtime distribution metadata is unavailable") from exc
    tracked_relative = "src/chronovisor/recall/recall_r7_evidence.py"
    try:
        source_relative = module_path.relative_to(source_root).as_posix()
    except ValueError:
        try:
            installed_relative = module_path.relative_to(distribution_root).as_posix()
        except ValueError as exc:
            raise EvidenceError("runtime module is outside trusted roots") from exc
        if installed_relative != "chronovisor/recall/recall_r7_evidence.py":
            raise EvidenceError("runtime module path is not the expected module") from None
        source_relative = tracked_relative
    else:
        if source_relative not in {
            tracked_relative,
            "chronovisor/recall/recall_r7_evidence.py",
        }:
            raise EvidenceError("runtime module path is not the expected module")
        tracked_relative = source_relative
    direct_url_target = path.resolve(strict=True)
    try:
        direct_url_target.relative_to(distribution_root)
    except ValueError as exc:
        raise EvidenceError("runtime direct_url is outside distribution metadata") from exc
    record_match: list[tuple[list[str], Path]] = []
    direct_url_record_match: list[tuple[list[str], Path]] = []
    for row in rows:
        if len(row) != 3 or not row[0] or Path(row[0]).is_absolute():
            continue
        candidate = distribution_root / Path(row[0])
        if _has_symlink_component(candidate):
            raise EvidenceError("runtime distribution RECORD path is unsafe")
        try:
            candidate = candidate.resolve(strict=True)
        except OSError as exc:
            raise EvidenceError("runtime distribution RECORD path is unavailable") from exc
        try:
            candidate.relative_to(distribution_root)
        except ValueError as exc:
            raise EvidenceError("runtime distribution RECORD path escapes metadata") from exc
        if candidate == module_path:
            record_match.append((row, candidate))
        if candidate == direct_url_target:
            direct_url_record_match.append((row, candidate))
    if len(record_match) != 1:
        raise EvidenceError("runtime module is absent from distribution RECORD")
    record_row, record_target = record_match[0]
    record_hash, separator, encoded_hash = record_row[1].partition("=")
    if record_hash != "sha256" or not separator or not encoded_hash:
        raise EvidenceError("runtime module RECORD hash is invalid")
    if re.fullmatch(r"[A-Za-z0-9_-]+", encoded_hash) is None:
        raise EvidenceError("runtime module RECORD hash is invalid")
    try:
        expected_record_digest = base64.b64decode(
            encoded_hash + "=" * (-len(encoded_hash) % 4),
            altchars=b"-_",
            validate=True,
        )
    except (ValueError, binascii.Error) as exc:
        raise EvidenceError("runtime module RECORD hash is invalid") from exc
    if expected_record_digest != hashlib.sha256(module_body).digest():
        raise EvidenceError("runtime module RECORD bytes mismatch")
    try:
        record_size = int(record_row[2])
    except ValueError as exc:
        raise EvidenceError("runtime module RECORD size is invalid") from exc
    if record_size != len(module_body):
        raise EvidenceError("runtime module RECORD size mismatch")
    if len(direct_url_record_match) != 1:
        raise EvidenceError("runtime direct_url is absent from distribution RECORD")
    direct_row, _direct_target = direct_url_record_match[0]
    direct_hash, direct_separator, direct_encoded_hash = direct_row[1].partition("=")
    if (
        direct_hash != "sha256"
        or not direct_separator
        or not direct_encoded_hash
        or re.fullmatch(r"[A-Za-z0-9_-]+", direct_encoded_hash) is None
    ):
        raise EvidenceError("runtime direct_url RECORD hash is invalid")
    try:
        expected_direct_digest = base64.b64decode(
            direct_encoded_hash + "=" * (-len(direct_encoded_hash) % 4),
            altchars=b"-_",
            validate=True,
        )
    except (ValueError, binascii.Error) as exc:
        raise EvidenceError("runtime direct_url RECORD hash is invalid") from exc
    if expected_direct_digest != hashlib.sha256(raw).digest():
        raise EvidenceError("runtime direct_url RECORD bytes mismatch")
    try:
        direct_size = int(direct_row[2])
    except ValueError as exc:
        raise EvidenceError("runtime direct_url RECORD size is invalid") from exc
    if direct_size != len(raw):
        raise EvidenceError("runtime direct_url RECORD size mismatch")
    try:
        git_head = os.fsdecode(_git_run(source_root, ["rev-parse", "HEAD"]).strip())
        indexed = _git_run(source_root, ["ls-files", "-s", "--", tracked_relative])
    except EvidenceError:
        raise
    if git_head != commit:
        raise EvidenceError("runtime archive commit is not the source HEAD")
    indexed_rows = [line for line in indexed.decode("utf-8").splitlines() if line]
    if len(indexed_rows) != 1:
        raise EvidenceError("runtime module is not uniquely tracked")
    header, separator, indexed_path = indexed_rows[0].partition("\t")
    fields = header.split()
    if separator != "\t" or len(fields) != 3 or indexed_path != tracked_relative:
        raise EvidenceError("runtime tracked module record is invalid")
    tracked_mode, tracked_blob, tracked_stage = fields
    if tracked_mode not in {"100644", "100755"} or tracked_stage != "0" or not _HEX40(tracked_blob):
        raise EvidenceError("runtime tracked module mode is invalid")
    if _git_object(source_root, tracked_blob, "blob") != module_body:
        raise EvidenceError("runtime tracked module bytes mismatch")
    runtime.update(
        {
            "record_path": str(record_path),
            "record_file_sha256": hashlib.sha256(record_raw).hexdigest(),
            "record_module_sha256": hashlib.sha256(module_body).hexdigest(),
            "record_module_size": record_size,
            "tracked_path": tracked_relative,
            "tracked_mode": tracked_mode,
            "tracked_blob_sha1": tracked_blob,
        }
    )
    # ``record_target`` is compared above to the same descriptor-derived path;
    # retaining the local name documents that the RECORD row was dereferenced.
    if record_target != module_path:
        raise EvidenceError("runtime module RECORD target drift")
    return runtime


def _fetch(url: str, label: str, *, fixed_endpoint: bool = False) -> dict[str, Any]:
    parsed = urlsplit(url)
    try:
        parsed_port = parsed.port
    except ValueError as exc:
        raise EvidenceError(f"{label} port is invalid") from exc
    if (
        parsed.scheme not in {"http", "https"}
        or parsed.hostname not in {"127.0.0.1", "::1", "localhost"}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise EvidenceError(f"{label} must be loopback")
    if fixed_endpoint:
        expected_path = "/api/health" if "health" in label.casefold() else "/api/fast-snapshot"
        if (
            parsed.scheme != "http"
            or parsed.hostname != DEFAULT_DASHBOARD_HOST
            or parsed_port != DEFAULT_DASHBOARD_PORT
            or parsed.path != expected_path
        ):
            raise EvidenceError(f"{label} is not the fixed dashboard endpoint")
    requested = urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))

    class _NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *_args: Any, **_kwargs: Any) -> None:
            raise EvidenceError(f"{label} redirect rejected")

    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}), _NoRedirect()
    )
    try:
        with opener.open(
            urllib.request.Request(
                requested, headers={"Accept": "application/json"}, method="GET"
            ),
            timeout=5,
        ) as response:
            body = response.read(MAX_BYTES + 1)
            status = int(response.status)
            final_url = response.geturl()
    except EvidenceError:
        raise
    except urllib.error.HTTPError as exc:
        raise EvidenceError(f"{label} status is not successful") from exc
    except (OSError, urllib.error.URLError) as exc:
        raise EvidenceError(f"{label} unavailable") from exc
    if len(body) > MAX_BYTES:
        raise EvidenceError(f"{label} too large")
    if final_url != requested or status != 200:
        raise EvidenceError(f"{label} response endpoint/status mismatch")
    try:
        payload = json.loads(body)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"{label} is not JSON") from exc
    if not isinstance(payload, Mapping):
        raise EvidenceError(f"{label} is not object")
    safe_url = requested
    return {
        "url": safe_url,
        "status": status,
        "body_sha256": hashlib.sha256(body).hexdigest(),
        "payload_sha256": _digest(payload),
        "payload": payload,
    }


def _valid_local_endpoint(url: object, path_suffix: str) -> bool:
    """Validate the fixed loopback endpoint shape stored in a receipt."""

    if not isinstance(url, str):
        return False
    parsed = urlsplit(url)
    return (
        parsed.scheme in {"http", "https"}
        and parsed.hostname in {"127.0.0.1", "::1", "localhost"}
        and parsed.username is None
        and parsed.password is None
        and not parsed.query
        and not parsed.fragment
        and parsed.path.rstrip("/") == path_suffix
    )


def _valid_fixed_endpoint(url: object, path_suffix: str) -> bool:
    return (
        isinstance(url, str)
        and url == f"{_FIXED_DASHBOARD_ORIGIN}{path_suffix}"
    )


def _validate_dashboard_payload(
    payload: Mapping[str, Any], endpoint: str, *, source_commit: str | None = None
) -> None:
    """Validate the small semantic surface needed by formal R7 evidence."""

    if endpoint == "health":
        if set(payload) != {"health"} or not isinstance(payload.get("health"), Mapping):
            raise EvidenceError("dashboard health schema is not closed")
        health = payload["health"]
        runtime = health.get("runtime")
        if (
            not isinstance(runtime, Mapping)
            or not isinstance(runtime.get("commit_id"), str)
            or _COMMIT.fullmatch(runtime["commit_id"]) is None
            or runtime.get("drift") is not False
        ):
            raise EvidenceError("dashboard health runtime identity is invalid")
        if source_commit is not None and runtime["commit_id"] != source_commit:
            raise EvidenceError("dashboard health commit drift")
        return
    if endpoint != "api":
        raise EvidenceError("unknown dashboard endpoint")
    # Fast-snapshot is intentionally broad, but its semantic shell is closed:
    # the route must expose the status/health/dashboard projections and no
    # caller-supplied identity fields are accepted as authority.
    expected = {
        "status",
        "events",
        "metrics",
        "runtime_failures",
        "last_failure",
        "local_consensus",
        "frontier_repair",
        "local_runtime",
        "ollama",
        "model_status",
        "self_heal",
        "recall",
        "recall_improvement",
        "model_lab",
        "typed_graph",
        "save_history",
        "knowledge_mix",
        "librarian",
        "health",
        "_dashboard",
    }
    if set(payload) != expected:
        raise EvidenceError("dashboard API schema is not closed")
    if not isinstance(payload.get("_dashboard"), Mapping):
        raise EvidenceError("dashboard API dashboard projection is invalid")


_ATTESTATION_KEYS = {
    "schema",
    "namespace",
    "artifact_id",
    "seal_sha256",
    "kind",
    "stage",
    "run_id",
    "captured_at",
    "collector",
    "source",
    "runtime",
    "process",
    "archive",
    "direct_url",
    "health",
    "api",
    "dom",
    "rollback",
}


def _validate_live_attestation_current(artifact: Mapping[str, Any]) -> None:
    process = artifact.get("process")
    service = process.get("service") if isinstance(process, Mapping) else None
    role = service.get("role") if isinstance(service, Mapping) else None
    if not isinstance(role, str):
        raise EvidenceError("live attestation service authority is invalid")
    current = _current_formal_inputs(
        str(artifact["stage"]), str(artifact["run_id"]), role
    )
    if (
        artifact.get("source")
        != {
            key: current["source"][key]
            for key in ("source_commit", "source_tree_sha256", "source_bytes_sha256")
        }
        or artifact.get("runtime") != current["runtime"]
        or not isinstance(process, Mapping)
        or not _same_process_identity(process, current["process"])
        or artifact.get("health") != current["health"]
        or artifact.get("api") != current["api"]
        or artifact.get("dom") != current["dom"]
    ):
        raise EvidenceError("live attestation current authority drift")


def _attestation_ref(path: Path) -> dict[str, str]:
    """Return the three immutable identifiers used to cross-bind a live receipt."""

    expected_parent = _FIXED_EVIDENCE_ROOT / "r7-live-attestations"
    if (
        not path.is_absolute()
        or _has_symlink_component(path)
        or _HEX.fullmatch(path.stem) is None
        or path.parent != expected_parent
    ):
        raise EvidenceError("live attestation is outside managed evidence")
    path = expected_parent / f"{path.stem}.json"
    try:
        artifact, body, _ = _read_sealed_artifact(path, LIVE_ATTESTATION_SCHEMA, "live attestation")
        _validate_attestation_payload(artifact)
    except (OSError, store.DistillationStoreError) as exc:
        raise EvidenceError("live attestation is unavailable") from exc
    _validate_live_attestation_current(artifact)
    return _artifact_ref_values(artifact, body)


def _test_attestation_ref(path: Path) -> dict[str, str]:
    """Read the explicitly test-only attestation namespace.

    Test rollback drills may need an attestation-shaped input, but that input
    must never be accepted by the production collector/validator.  Keeping a
    separate schema makes that boundary machine-checkable.
    """

    try:
        artifact, body, _ = _read_sealed_artifact(path, TEST_LIVE_ATTESTATION_SCHEMA, "test live attestation")
    except (OSError, store.DistillationStoreError) as exc:
        raise EvidenceError("test live attestation is unavailable") from exc
    return _artifact_ref_values(artifact, body)


def _validate_attestation_payload(
    artifact: Mapping[str, Any], *, expected_stage: str | None = None,
    expected_run_id: str | None = None,
    allow_test_service: bool = False,
    check_identity: bool = True,
) -> dict[str, Any]:
    """Validate the closed external live-attestation contract."""

    if check_identity:
        _validate_artifact_identity(artifact, None, "live attestation")

    if (
        set(artifact) != _ATTESTATION_KEYS
        or artifact.get("schema") != LIVE_ATTESTATION_SCHEMA
        or artifact.get("namespace") != "recall-distillation"
        or artifact.get("kind") != "r7-live-attestation"
        or not isinstance(artifact.get("artifact_id"), str)
        or _HEX.fullmatch(artifact["artifact_id"]) is None
        or not isinstance(artifact.get("stage"), str)
        or artifact["stage"] not in STAGES
        or (expected_stage is not None and artifact["stage"] != expected_stage)
        or not isinstance(artifact.get("run_id"), str)
        or _HEX.fullmatch(artifact["run_id"]) is None
        or (expected_run_id is not None and artifact["run_id"] != expected_run_id)
        or not isinstance(artifact.get("captured_at"), str)
    ):
        raise EvidenceError("live attestation schema is invalid")
    _utc(artifact["captured_at"], "live attestation capture time")
    attestation_time = _utc(artifact["captured_at"], "live attestation capture time")
    collector = artifact.get("collector")
    if (
        not isinstance(collector, Mapping)
        or set(collector) != {"name", "version", "synthetic_fixture"}
        or collector.get("name") != "chronovisor-r7-attestation"
        or isinstance(collector.get("version"), bool)
        or not isinstance(collector.get("version"), int)
        or collector.get("version") != 1
        or collector.get("synthetic_fixture") is not False
    ):
        raise EvidenceError("live attestation collector is invalid")
    source = artifact.get("source")
    if (
        not isinstance(source, Mapping)
        or set(source) != {"source_commit", "source_tree_sha256", "source_bytes_sha256"}
        or not isinstance(source.get("source_commit"), str)
        or _COMMIT.fullmatch(source["source_commit"]) is None
        or not all(
            isinstance(source.get(key), str) and _HEX.fullmatch(source[key]) is not None
            for key in ("source_tree_sha256", "source_bytes_sha256")
        )
    ):
        raise EvidenceError("live attestation source is invalid")
    runtime = artifact.get("runtime")
    runtime_keys = _TEST_RUNTIME_KEYS if allow_test_service else _RUNTIME_KEYS
    if not isinstance(runtime, Mapping) or set(runtime) != runtime_keys:
        raise EvidenceError("live attestation runtime is invalid")
    if (
        not isinstance(runtime.get("archive_commit"), str)
        or _COMMIT.fullmatch(runtime["archive_commit"]) is None
        or any(
            not isinstance(runtime.get(key), str) or _HEX.fullmatch(runtime[key]) is None
            for key in (
                "direct_url_sha256",
                "direct_url_raw_sha256",
                "direct_url_payload_sha256",
                "module_bytes_sha256",
            )
        )
        or not isinstance(runtime.get("module_path"), str)
        or not Path(runtime["module_path"]).is_absolute()
        or _has_symlink_component(Path(runtime["module_path"]))
        or not isinstance(runtime.get("module_lstat"), Mapping)
        or set(runtime["module_lstat"]) != _STAT_KEYS
        or any(
            isinstance(runtime["module_lstat"].get(key), bool)
            or not isinstance(runtime["module_lstat"].get(key), int)
            for key in _STAT_KEYS
        )
        or runtime.get("distribution_name") != "chronovisor"
        or not isinstance(runtime.get("distribution_version"), str)
        or not runtime["distribution_version"]
        or runtime["direct_url_sha256"] != runtime["direct_url_payload_sha256"]
        or runtime["archive_commit"] != source["source_commit"]
    ):
        raise EvidenceError("live attestation runtime is invalid")
    if not allow_test_service and (
            not isinstance(runtime.get("record_path"), str)
            or not Path(runtime["record_path"]).is_absolute()
            or _has_symlink_component(Path(runtime["record_path"]))
            or Path(runtime["record_path"]).name != "RECORD"
            or any(
                not isinstance(runtime.get(key), str)
                or _HEX.fullmatch(runtime[key]) is None
                for key in (
                    "record_file_sha256",
                    "record_module_sha256",
                    "tracked_blob_sha1",
                )
            )
            or not isinstance(runtime.get("record_module_size"), int)
            or isinstance(runtime.get("record_module_size"), bool)
            or runtime["record_module_size"] < 0
            or not isinstance(runtime.get("tracked_path"), str)
            or Path(runtime["tracked_path"]).is_absolute()
            or ".." in Path(runtime["tracked_path"]).parts
            or runtime["tracked_path"]
            not in {
                "src/chronovisor/recall/recall_r7_evidence.py",
                "chronovisor/recall/recall_r7_evidence.py",
            }
            or runtime.get("tracked_mode") not in {"100644", "100755"}
    ):
        raise EvidenceError("live attestation archive binding is invalid")
    expected_archive = _archive_projection(runtime)
    if artifact.get("archive") != expected_archive:
        raise EvidenceError("live attestation archive is invalid")
    if artifact.get("direct_url") != dict(runtime):
        raise EvidenceError("live attestation direct_url is invalid")
    process = artifact.get("process")
    process_keys = {
        "pid",
        "started_at",
        "executable_path",
        "executable_lstat",
        "executable_sha256",
    }
    if not allow_test_service:
        process_keys.add("service")
    if (
        not isinstance(process, Mapping)
        or set(process) != process_keys
        or isinstance(process.get("pid"), bool)
        or not isinstance(process.get("pid"), int)
        or process["pid"] <= 0
        or not isinstance(process.get("started_at"), str)
        or _STARTED_AT.fullmatch(process["started_at"]) is None
        or not isinstance(process.get("executable_path"), str)
        or not Path(process["executable_path"]).is_absolute()
        or _has_symlink_component(Path(process["executable_path"]))
        or not isinstance(process.get("executable_lstat"), Mapping)
        or set(process["executable_lstat"])
        != _STAT_KEYS
        or any(
            isinstance(process["executable_lstat"].get(key), bool)
            or not isinstance(process["executable_lstat"].get(key), int)
            for key in _STAT_KEYS
        )
        or not isinstance(process.get("executable_sha256"), str)
        or _HEX.fullmatch(process["executable_sha256"]) is None
    ):
        raise EvidenceError("live attestation process is invalid")
    if not allow_test_service:
        service = process.get("service")
        service_parent_pid = service.get("parent_pid") if isinstance(service, Mapping) else None
        if (
            not isinstance(service, Mapping)
            or set(service)
            != {
                "role",
                "domain",
                "label",
                "state",
                "parent_pid",
                "child_pid",
                "captured_at",
                "raw_output_sha256",
            }
            or service.get("role") not in _SERVICE_LABELS
            or service.get("label") != _SERVICE_LABELS[service["role"]]
            or service.get("domain") != f"gui/{os.getuid()}"
            or service.get("state") != "running"
            or isinstance(service.get("parent_pid"), bool)
            or not isinstance(service_parent_pid, int)
            or service_parent_pid <= 0
            or service.get("child_pid") != process.get("pid")
            or not isinstance(service.get("captured_at"), str)
            or _utc(service.get("captured_at"), "launchd capture time") is None
            or not isinstance(service.get("raw_output_sha256"), str)
            or _HEX.fullmatch(service["raw_output_sha256"]) is None
        ):
            raise EvidenceError("live attestation launchd service is invalid")
        if abs(
            (_utc(service["captured_at"], "launchd capture time") - attestation_time).total_seconds()
        ) > MAX_OBSERVATION_SKEW_SECONDS:
            raise EvidenceError("live attestation launchd clock skew is excessive")
    for endpoint_name in ("health", "api"):
        endpoint = artifact.get(endpoint_name)
        if (
            not isinstance(endpoint, Mapping)
            or set(endpoint) != {"url", "status", "body_sha256", "payload_sha256"}
            or not (
                _valid_local_endpoint(
                    endpoint.get("url"),
                    "/api/health"
                    if endpoint_name == "health"
                    else "/api/fast-snapshot",
                )
                if allow_test_service
                else _valid_fixed_endpoint(
                    endpoint.get("url"),
                    "/api/health"
                    if endpoint_name == "health"
                    else "/api/fast-snapshot",
                )
            )
            or endpoint.get("status") != 200
            or not isinstance(endpoint.get("body_sha256"), str)
            or _HEX.fullmatch(endpoint["body_sha256"]) is None
            or not isinstance(endpoint.get("payload_sha256"), str)
            or _HEX.fullmatch(endpoint["payload_sha256"]) is None
        ):
            raise EvidenceError(f"live attestation {endpoint_name} is invalid")
    dom = artifact.get("dom")
    dom_keys = (
        {
            "kind",
            "synthetic_fixture",
            "producer_name",
            "producer_version",
            "html_sha256",
            "capture_sha256",
        }
        if allow_test_service
        else {
            "kind",
            "synthetic_fixture",
            "producer_name",
            "producer_version",
            "html_sha256",
            "capture_sha256",
            "capture_artifact_id",
            "capture_file_sha256",
            "capture_seal_sha256",
        }
    )
    if (
        not isinstance(dom, Mapping)
        or set(dom) != dom_keys
        or dom.get("kind") != "browser-dom-capture"
        or dom.get("synthetic_fixture") is not False
        or dom.get("producer_name") != "chronovisor-browser"
        or isinstance(dom.get("producer_version"), bool)
        or not isinstance(dom.get("producer_version"), int)
        or not isinstance(dom.get("html_sha256"), str)
        or _HEX.fullmatch(dom["html_sha256"]) is None
        or not isinstance(dom.get("capture_sha256"), str)
        or _HEX.fullmatch(dom["capture_sha256"]) is None
        or (
            not allow_test_service
            and any(
                not isinstance(dom.get(key), str)
                or _HEX.fullmatch(dom[key]) is None
                for key in (
                    "capture_artifact_id",
                    "capture_file_sha256",
                    "capture_seal_sha256",
                )
            )
        )
    ):
        raise EvidenceError("live attestation DOM is invalid")
    if not allow_test_service:
        capture_path = (
            _FIXED_DOM_CAPTURE_ROOT / f"{dom['capture_artifact_id']}.json"
        )
        try:
            capture, capture_raw, capture_ref = _read_dom_capture(
                capture_path,
                stage=str(artifact["stage"]),
                run_id=str(artifact["run_id"]),
            )
        except EvidenceError as exc:
            raise EvidenceError("live attestation DOM authority is unavailable") from exc
        if (
            capture_ref["capture_artifact_id"] != dom["capture_artifact_id"]
            or capture_ref["capture_file_sha256"] != dom["capture_file_sha256"]
            or capture_ref["capture_seal_sha256"] != dom["capture_seal_sha256"]
            or hashlib.sha256(capture_raw).hexdigest() != dom["capture_sha256"]
            or capture.get("html_sha256") != dom["html_sha256"]
            or not isinstance(capture.get("producer"), Mapping)
            or capture["producer"].get("name") != dom["producer_name"]
            or capture["producer"].get("version") != dom["producer_version"]
        ):
            raise EvidenceError("live attestation DOM capture binding mismatch")
        if abs(
            (_utc(capture["captured_at"], "DOM capture time") - attestation_time).total_seconds()
        ) > MAX_OBSERVATION_SKEW_SECONDS:
            raise EvidenceError("live attestation DOM clock skew is excessive")
    rollback = artifact.get("rollback")
    if (
        not isinstance(rollback, Mapping)
        or set(rollback) != {"status", "artifact_id", "receipt_sha256"}
        or rollback.get("status") not in {"not_triggered", "triggered", "rolled_back"}
        or (rollback.get("artifact_id") is not None and _HEX.fullmatch(str(rollback["artifact_id"])) is None)
        or (rollback.get("receipt_sha256") is not None and _HEX.fullmatch(str(rollback["receipt_sha256"])) is None)
    ):
        raise EvidenceError("live attestation rollback is invalid")
    return dict(artifact)


def validate_live_attestation(
    path: Path, *, expected_stage: str | None = None, expected_run_id: str | None = None
) -> dict[str, str]:
    """Public validator for an external, content-addressed live attestation."""

    expected_parent = _FIXED_EVIDENCE_ROOT / "r7-live-attestations"
    if (
        not path.is_absolute()
        or _has_symlink_component(path)
        or _HEX.fullmatch(path.stem) is None
        or path.parent != expected_parent
    ):
        raise EvidenceError("live attestation is outside managed evidence")
    path = expected_parent / f"{path.stem}.json"
    try:
        artifact, body, _ = _read_sealed_artifact(
            path, LIVE_ATTESTATION_SCHEMA, "live attestation"
        )
        _validate_attestation_payload(
            artifact, expected_stage=expected_stage, expected_run_id=expected_run_id
        )
        _validate_live_attestation_current(artifact)
    except (OSError, store.DistillationStoreError) as exc:
        raise EvidenceError("live attestation is unavailable") from exc
    return _artifact_ref_values(artifact, body)


def _write_live_attestation_payload(
    evidence_root: Path, payload: Mapping[str, Any]
) -> dict[str, str]:
    """Persist one already-derived attestation payload (internal/test path)."""

    _validate_attestation_payload(
        {
            "schema": LIVE_ATTESTATION_SCHEMA,
            "namespace": "recall-distillation",
            "artifact_id": "0" * 64,
            "seal_sha256": "0" * 64,
            **payload,
        },
        check_identity=False,
    )
    artifact_id, path, artifact = store.write_immutable(
        evidence_root / "r7-live-attestations", payload, schema=LIVE_ATTESTATION_SCHEMA
    )
    readback, body, _ = _read_sealed_artifact(
        path, LIVE_ATTESTATION_SCHEMA, "live attestation"
    )
    if readback != artifact:
        raise EvidenceError("live attestation immutable readback mismatch")
    return {
        "artifact_id": artifact_id,
        "seal_sha256": artifact["seal_sha256"],
        "file_sha256": hashlib.sha256(body).hexdigest(),
        "path": str(path.resolve()),
    }


def _write_test_live_attestation_payload(
    evidence_root: Path, payload: Mapping[str, Any]
) -> dict[str, str]:
    """Write a test-only attestation that formal validators cannot consume."""

    normalized = dict(payload)
    normalized["test_only"] = True
    # Validate the payload's nested shape with the production contract before
    # changing only the outer schema.  The schema itself remains the rejection
    # boundary for production readers.
    _validate_attestation_payload(
        {
            "schema": LIVE_ATTESTATION_SCHEMA,
            "namespace": "recall-distillation",
            "artifact_id": "0" * 64,
            "seal_sha256": "0" * 64,
            **payload,
        },
        allow_test_service=True,
        check_identity=False,
    )
    artifact_id, path, artifact = store.write_immutable(
        evidence_root / "r7-live-attestations-test",
        normalized,
        schema=TEST_LIVE_ATTESTATION_SCHEMA,
    )
    readback, body, _ = _read_sealed_artifact(
        path, TEST_LIVE_ATTESTATION_SCHEMA, "test live attestation"
    )
    if readback != artifact:
        raise EvidenceError("test live attestation immutable readback mismatch")
    return {
        "artifact_id": artifact_id,
        "seal_sha256": artifact["seal_sha256"],
        "file_sha256": hashlib.sha256(body).hexdigest(),
        "path": str(path.resolve()),
    }


def write_live_attestation(
    evidence_root: Path,
    *,
    root: Path,
    source_root: Path,
    direct_url_path: Path,
    executable: Path,
    pid: int,
    stage: str,
    run_id: str,
    dashboard_url: str,
    dom_capture_path: Path,
    expected_started_at: object | None = None,
    service_role: str = "dashboard",
    allow_test_root: bool = False,
) -> dict[str, str]:
    """Collect and seal a live attestation from independent local inputs."""

    _id(run_id, "live attestation run id")
    if stage not in STAGES:
        raise EvidenceError("live attestation stage is invalid")
    if service_role not in _SERVICE_LABELS:
        raise EvidenceError("live attestation service role is not approved")
    if allow_test_root and (
        root.resolve() == _FIXED_PRODUCTION_ROOT
        or evidence_root.resolve() == _FIXED_EVIDENCE_ROOT
    ):
        raise EvidenceError("test live attestation cannot use production authority")
    if not allow_test_root:
        if _has_symlink_component(root) or _has_symlink_component(evidence_root):
            raise EvidenceError("live attestation root/evidence is symlinked")
        if root.resolve() != _FIXED_PRODUCTION_ROOT:
            raise EvidenceError("live attestation root is not the production runtime")
        if evidence_root.resolve() != _FIXED_EVIDENCE_ROOT.resolve():
            raise EvidenceError("live attestation evidence root is not managed")
        # Never operate through a caller-provided alias, even after a resolved
        # equality check.  A parent directory can be swapped between that
        # check and the first open; production paths are the constants below.
        root = _FIXED_PRODUCTION_ROOT
        evidence_root = _FIXED_EVIDENCE_ROOT
        if _has_symlink_component(evidence_root):
            raise EvidenceError("live attestation evidence root is unsafe")
        # These arguments remain in the compatibility signature for runtime
        # callers, but formal collection never trusts them.
        source_root = _fixed_source_root()
        direct_url_path = _fixed_direct_url_path()
        dom_capture_path = _fixed_dom_capture_path(stage, run_id)
        process = _service_process_identity(service_role)
        effective_service_role = service_role
        dashboard_origin = _FIXED_DASHBOARD_ORIGIN
    else:
        if _has_symlink_component(root) or any(
            _has_symlink_component(path)
            for path in (source_root, evidence_root, direct_url_path, executable, dom_capture_path)
        ):
            raise EvidenceError("live attestation input is symlinked")
        effective_service_role = None
        dashboard_origin = dashboard_url.rstrip("/")
        process = None
    evidence_root.mkdir(parents=True, exist_ok=True)
    dom, dom_raw, dom_ref = _read_dom_capture(
        dom_capture_path,
        stage=stage,
        run_id=run_id,
        test_only=allow_test_root,
    )
    health = _fetch(
        f"{dashboard_origin}/api/health", "dashboard health", fixed_endpoint=not allow_test_root
    )
    api = _fetch(
        f"{dashboard_origin}/api/fast-snapshot", "dashboard API", fixed_endpoint=not allow_test_root
    )
    if health["status"] != 200 or api["status"] != 200:
        raise EvidenceError("live attestation dashboard status is not successful")
    runtime = _direct_url(direct_url_path, formal=not allow_test_root)
    if process is None:
        process = (
            _process_identity(executable, pid, expected_started_at, effective_service_role)
            if expected_started_at is not None
            else _process_identity(executable, pid, service_role=effective_service_role)
        )
    source = _source_identity(source_root)
    health_payload = health["payload"]
    _validate_dashboard_payload(health_payload, "health", source_commit=source["source_commit"])
    _validate_dashboard_payload(api["payload"], "api")
    health_runtime = (
        health_payload.get("health", {}).get("runtime", {})
        if isinstance(health_payload.get("health"), Mapping)
        else {}
    )
    if (
        runtime["archive_commit"] != source["source_commit"]
        or not isinstance(health_runtime, Mapping)
        or health_runtime.get("commit_id") != source["source_commit"]
        or health_runtime.get("drift") is not False
    ):
        raise EvidenceError("live attestation runtime commit drift")
    endpoint_health = {key: health[key] for key in ("url", "status", "body_sha256", "payload_sha256")}
    endpoint_api = {key: api[key] for key in ("url", "status", "body_sha256", "payload_sha256")}
    payload = {
        "kind": "r7-live-attestation",
        "stage": stage,
        "run_id": run_id,
        "captured_at": datetime.now(UTC).isoformat(),
        "collector": {
            "name": "chronovisor-r7-attestation",
            "version": 1,
            "synthetic_fixture": False,
        },
        "source": {
            key: source[key]
            for key in ("source_commit", "source_tree_sha256", "source_bytes_sha256")
        },
        "runtime": runtime,
        "process": process,
        "archive": _archive_projection(runtime),
        "direct_url": runtime,
        "health": endpoint_health,
        "api": endpoint_api,
        "dom": {
            "kind": "browser-dom-capture",
            "synthetic_fixture": False,
            "producer_name": dom["producer"]["name"],
            "producer_version": dom["producer"]["version"],
            "html_sha256": dom["html_sha256"],
            "capture_sha256": hashlib.sha256(dom_raw).hexdigest(),
            **dom_ref,
        },
        "rollback": {"status": "not_triggered", "artifact_id": None, "receipt_sha256": None},
    }
    artifact_path: Path | None = None
    artifact_bytes: bytes | None = None
    try:
        produced = (
            _write_test_live_attestation_payload(evidence_root, payload)
            if allow_test_root
            else _write_live_attestation_payload(evidence_root, payload)
        )
        artifact_path = Path(produced["path"])
        artifact_bytes, _ = _read_stable_file(artifact_path, "live attestation artifact")
        dom_after, dom_after_raw, dom_ref_after = _read_dom_capture(
            dom_capture_path,
            stage=stage,
            run_id=run_id,
            test_only=allow_test_root,
        )
        process_after = (
            _service_process_identity(service_role, process["started_at"])
            if not allow_test_root
            else _process_identity(
                executable, pid, process["started_at"], effective_service_role
            )
        )
        health_after = _fetch(
            f"{dashboard_origin}/api/health",
            "dashboard health",
            fixed_endpoint=not allow_test_root,
        )
        api_after = _fetch(
            f"{dashboard_origin}/api/fast-snapshot",
            "dashboard API",
            fixed_endpoint=not allow_test_root,
        )
        if (
            dom_after_raw != dom_raw
            or dom_after != dom
            or dom_ref_after != dom_ref
            or _direct_url(direct_url_path, formal=not allow_test_root) != runtime
            or not _same_process_identity(process_after, process)
            or {
                key: health_after[key]
                for key in ("url", "status", "body_sha256", "payload_sha256")
            }
            != endpoint_health
            or {
                key: api_after[key]
                for key in ("url", "status", "body_sha256", "payload_sha256")
            }
            != endpoint_api
            or _source_identity(source_root) != source
        ):
            raise EvidenceError("live attestation input drift")
        attestation_reader = _test_attestation_ref if allow_test_root else _attestation_ref
        if attestation_reader(artifact_path) != {
            "artifact_id": produced["artifact_id"],
            "file_sha256": produced["file_sha256"],
            "seal_sha256": produced["seal_sha256"],
        }:
            raise EvidenceError("live attestation publication mismatch")
        return produced
    except Exception:
        if artifact_path is not None and artifact_bytes is not None:
            _remove_own_artifact(artifact_path, artifact_bytes)
        raise


record_live_attestation = write_live_attestation


read_live_attestation = validate_live_attestation


def write_external_failure_event(
    evidence_root: Path,
    *,
    stage: str,
    run_id: str,
    poll_id: str,
    poll_sha256: str | None = None,
    source_commit: str | None = None,
    archive_commit: str | None = None,
    process: Mapping[str, Any] | None = None,
    live_attestation: Mapping[str, str] | None = None,
    supervisor_failure_path: Path | None = None,
    allow_test_root: bool = False,
) -> dict[str, str]:
    """Import a supervisor failure in production; write only test fixtures.

    A production caller cannot manufacture a failure receipt from identity
    mappings or a token.  The supervisor owns a sealed event at the fixed
    ``failures`` path; this function only reads and returns its identifiers.
    The generic writer remains available exclusively in the explicit
    test-only namespace.
    """

    production_evidence = evidence_root.resolve() == _FIXED_EVIDENCE_ROOT.resolve()
    if allow_test_root and production_evidence:
        raise EvidenceError("test failure writer cannot use production authority")
    if not production_evidence and not allow_test_root:
        raise EvidenceError("external failure writer requires managed evidence root")
    if production_evidence:
        if _has_symlink_component(evidence_root):
            raise EvidenceError("external failure evidence root is symlinked")
        evidence_root = _FIXED_EVIDENCE_ROOT
        if _has_symlink_component(evidence_root):
            raise EvidenceError("external failure evidence root is unsafe")
        if supervisor_failure_path is None:
            raise EvidenceError("external supervisor failure event is required")
        if live_attestation is not None:
            raise EvidenceError("production failure identity is caller-controlled")
        if (
            not supervisor_failure_path.is_absolute()
            or _has_symlink_component(supervisor_failure_path)
            or _HEX.fullmatch(supervisor_failure_path.stem) is None
        ):
            raise EvidenceError("external supervisor failure event path is unsafe")
        poll_path = evidence_root / "polls" / f"{poll_id}.json"
        poll, poll_raw, _ = _read_sealed_artifact(poll_path, POLL_SCHEMA, "failure poll")
        if poll.get("stage") != stage or poll.get("run_id") != run_id:
            raise EvidenceError("external failure poll binding is invalid")
        if supervisor_failure_path.parent.resolve() != (evidence_root / "failures").resolve():
            raise EvidenceError("external supervisor failure event is outside managed evidence")
        supervisor_failure_path = evidence_root / "failures" / f"{supervisor_failure_path.stem}.json"
        actual_poll_file_sha256 = hashlib.sha256(poll_raw).hexdigest()
        actual_poll_seal = str(poll["seal_sha256"])
        actual_poll_id = str(poll["artifact_id"])
        if poll_sha256 is not None and poll_sha256 not in {
            actual_poll_id,
            actual_poll_file_sha256,
            actual_poll_seal,
        }:
            raise EvidenceError("external failure poll hash mismatch")
        poll_sha256 = actual_poll_seal
        poll_process = poll.get("process")
        role = (
            poll_process.get("service", {}).get("role")
            if isinstance(poll_process, Mapping)
            and isinstance(poll_process.get("service"), Mapping)
            else None
        )
        if not isinstance(role, str):
            raise EvidenceError("external failure process role is invalid")
        if not _verify_live_attestation_poll(
            evidence_root, poll, root=_FIXED_PRODUCTION_ROOT
        ):
            raise EvidenceError("external failure live attestation is invalid")
        current = _current_formal_inputs(stage, run_id, role)
        event = _read_external_failure(
            supervisor_failure_path,
            poll=poll,
            stage=stage,
            run_id=run_id,
            source=current["source"],
            runtime=current["runtime"],
            process=current["process"],
        )
        return {
            "artifact_id": event["artifact_id"],
            "seal_sha256": event["receipt_sha256"],
            "file_sha256": event["bytes_sha256"],
            "path": str(supervisor_failure_path.resolve()),
        }
    if _has_symlink_component(evidence_root):
        raise EvidenceError("external failure evidence root is unsafe")
    if supervisor_failure_path is not None:
        raise EvidenceError("supervisor failure import is production-only")
    if stage != "100" or _HEX.fullmatch(run_id) is None or _HEX.fullmatch(poll_id) is None:
        raise EvidenceError("external failure stage/run binding is invalid")
    if not isinstance(poll_sha256, str):
        raise EvidenceError("external failure poll binding is invalid")
    _id(poll_sha256, "external failure poll")
    if (
        not isinstance(source_commit, str)
        or not isinstance(archive_commit, str)
        or _COMMIT.fullmatch(source_commit) is None
        or _COMMIT.fullmatch(archive_commit) is None
    ):
        raise EvidenceError("external failure commit binding is invalid")
    if (
        not isinstance(process, Mapping)
        or not isinstance(process.get("pid"), int)
        or isinstance(process.get("pid"), bool)
        or process["pid"] <= 0
        or not isinstance(process.get("started_at"), str)
    ):
        raise EvidenceError("external failure process binding is invalid")
    payload: dict[str, Any] = {
        "kind": "r7-external-failure",
        "captured_at": datetime.now(UTC).isoformat(),
        "stage": stage,
        "run_id": run_id,
        "poll_id": poll_id,
        "poll_sha256": poll_sha256,
        "source_commit": source_commit,
        "archive_commit": archive_commit,
        "process_pid": process["pid"],
        "process_started_at": process["started_at"],
    }
    if live_attestation is not None:
        if set(live_attestation) != {"artifact_id", "file_sha256", "seal_sha256"} or any(
            not isinstance(live_attestation.get(key), str)
            or _HEX.fullmatch(live_attestation[key]) is None
            for key in live_attestation
        ):
            raise EvidenceError("external failure attestation reference is invalid")
        payload.update(
            {
                "live_attestation_artifact_id": live_attestation["artifact_id"],
                "live_attestation_file_sha256": live_attestation["file_sha256"],
                "live_attestation_seal_sha256": live_attestation["seal_sha256"],
            }
        )
    output_schema = "chronovisor.recall-r7-failure.v1"
    if allow_test_root:
        payload["test_only"] = True
        output_schema = TEST_FAILURE_SCHEMA
    artifact_id, path, artifact = store.write_immutable(
        evidence_root / "failures", payload, schema=output_schema
    )
    readback, body, _ = _read_sealed_artifact(
        path, output_schema, "external failure event"
    )
    if readback != artifact:
        raise EvidenceError("external failure event readback mismatch")
    return {
        "artifact_id": artifact_id,
        "seal_sha256": artifact["seal_sha256"],
        "file_sha256": hashlib.sha256(body).hexdigest(),
        "path": str(path.resolve()),
    }


def _validate_live_attestation_binding(
    path: Path,
    *,
    stage: str,
    run_id: str,
    source: Mapping[str, Any],
    runtime: Mapping[str, Any],
    process: Mapping[str, Any],
    health: Mapping[str, Any],
    api: Mapping[str, Any],
    dom: Mapping[str, Any],
) -> dict[str, str]:
    artifact, body, _ = _read_sealed_artifact(
        path, LIVE_ATTESTATION_SCHEMA, "live attestation"
    )
    _validate_attestation_payload(
        artifact, expected_stage=stage, expected_run_id=run_id
    )
    reference = _artifact_ref_values(artifact, body)
    if (
        artifact.get("source")
        != {
            key: source[key]
            for key in ("source_commit", "source_tree_sha256", "source_bytes_sha256")
        }
        or artifact.get("runtime") != dict(runtime)
        or artifact.get("archive") != _archive_projection(runtime)
        or artifact.get("direct_url") != dict(runtime)
        or not isinstance(artifact.get("process"), Mapping)
        or not _same_process_identity(artifact["process"], process)
        or artifact.get("health")
        != {key: health[key] for key in ("url", "status", "body_sha256", "payload_sha256")}
        or artifact.get("api")
        != {key: api[key] for key in ("url", "status", "body_sha256", "payload_sha256")}
        or artifact.get("dom") != dict(dom)
    ):
        raise EvidenceError("live attestation binding mismatch")
    return reference


def _rollback_attestation(
    *,
    root: Path,
    evidence_root: Path,
    poll: Mapping[str, Any],
    stage: str,
    run_id: str,
    source: Mapping[str, Any],
    runtime: Mapping[str, Any],
    process: Mapping[str, Any],
    failure_token: str | None,
) -> tuple[dict[str, str], Path]:
    """Resolve the poll's external live attestation for a rollback."""

    ref = poll.get("live_attestation")
    if isinstance(ref, Mapping) and isinstance(ref.get("artifact_id"), str):
        artifact_id = ref["artifact_id"]
        if _HEX.fullmatch(artifact_id) is None:
            raise EvidenceError("live attestation artifact id is invalid")
        production = root.resolve() == _FIXED_PRODUCTION_ROOT
        directory_name = "r7-live-attestations" if production else "r7-live-attestations-test"
        schema = LIVE_ATTESTATION_SCHEMA if production else TEST_LIVE_ATTESTATION_SCHEMA
        path = evidence_root / directory_name / f"{artifact_id}.json"
        artifact, body, _ = _read_sealed_artifact(path, schema, "live attestation")
        if _artifact_ref_values(artifact, body) != dict(ref):
            raise EvidenceError("live attestation poll reference mismatch")
        if production:
            _validate_attestation_payload(
                artifact, expected_stage=stage, expected_run_id=run_id
            )
        else:
            test_payload = {
                key: value
                for key, value in artifact.items()
                if key not in {"test_only", "schema", "namespace", "artifact_id", "seal_sha256"}
            }
            _validate_attestation_payload(
                {
                    "schema": LIVE_ATTESTATION_SCHEMA,
                    "namespace": "recall-distillation",
                    "artifact_id": artifact["artifact_id"],
                    "seal_sha256": artifact["seal_sha256"],
                    **test_payload,
                },
                expected_stage=stage,
                expected_run_id=run_id,
                allow_test_service=True,
                check_identity=False,
            )
        if (
            artifact.get("source", {}).get("source_commit") != source.get("source_commit")
            or artifact.get("runtime") != runtime
            or not isinstance(artifact.get("process"), Mapping)
            or not _same_process_identity(artifact["process"], process)
        ):
            raise EvidenceError("live attestation rollback binding mismatch")
        poll_dom = poll.get("dom")
        if isinstance(poll_dom, Mapping) and "producer_name" in poll_dom:
            poll_dom = dict(poll_dom)
        if (
            isinstance(poll.get("health"), Mapping)
            and artifact.get("health") != poll.get("health")
        ) or (
            isinstance(poll.get("api"), Mapping)
            and artifact.get("api") != poll.get("api")
        ) or (
            isinstance(poll_dom, Mapping)
            and artifact.get("dom") != poll_dom
        ):
            raise EvidenceError("live attestation rollback surface mismatch")
        return dict(ref), path
    if root.resolve() == _FIXED_PRODUCTION_ROOT or failure_token != "deterministic-test-failure":
        raise EvidenceError("external live attestation is required")
    empty_health = {
        "url": "http://127.0.0.1/api/health",
        "status": 200,
        "body_sha256": "0" * 64,
        "payload_sha256": "0" * 64,
    }
    empty_api = {
        "url": "http://127.0.0.1/api/fast-snapshot",
        "status": 200,
        "body_sha256": "0" * 64,
        "payload_sha256": "0" * 64,
    }
    empty_dom = {
        "kind": "browser-dom-capture",
        "synthetic_fixture": False,
        "producer_name": "chronovisor-browser",
        "producer_version": 1,
        "html_sha256": "0" * 64,
        "capture_sha256": "0" * 64,
    }
    payload = {
        "kind": "r7-live-attestation",
        "stage": stage,
        "run_id": run_id,
        "captured_at": datetime.now(UTC).isoformat(),
        "collector": {
            "name": "chronovisor-r7-attestation",
            "version": 1,
            "synthetic_fixture": False,
        },
        "source": {
            key: source[key]
            for key in ("source_commit", "source_tree_sha256", "source_bytes_sha256")
        },
        "runtime": dict(runtime),
        "process": dict(process),
        "archive": _archive_projection(runtime),
        "direct_url": dict(runtime),
        "health": empty_health,
        "api": empty_api,
        "dom": empty_dom,
        "rollback": {"status": "not_triggered", "artifact_id": None, "receipt_sha256": None},
    }
    produced = _write_test_live_attestation_payload(evidence_root, payload)
    return (
        {
            key: produced[key]
            for key in ("artifact_id", "file_sha256", "seal_sha256")
        },
        Path(produced["path"]),
    )


def _stage_state(root: Path, stage: str) -> dict[str, str | None]:
    if stage not in STAGES:
        raise EvidenceError("unknown rollout stage")
    if _has_symlink_component(root):
        raise EvidenceError("sealed rollout root is symlinked")
    try:
        state = _read_sealed_state(
            store.distillation_dir(root) / store.STATE_FILE,
            store.DISTILLATION_SCHEMA,
            "rollout state",
        )
        baseline_id = _id(state.get("baseline_artifact_id"), "state baseline")
        baseline, _, _ = _read_sealed_artifact(
            store.distillation_dir(root) / "baselines" / f"{baseline_id}.json",
            "chronovisor.recall-distill-baseline.v1",
            "rollout baseline",
        )
        candidate = _read_sealed_state(
            store.distillation_dir(root) / store.POINTER_FILES["candidate"],
            store.DISTILLATION_SCHEMA,
            "candidate policy pointer",
        )
        lkg = _read_sealed_state(
            store.distillation_dir(root) / store.POINTER_FILES["lkg"],
            store.DISTILLATION_SCHEMA,
            "LKG policy pointer",
        )
        active = _read_sealed_state(
            store.distillation_dir(root) / store.POINTER_FILES["active"],
            store.DISTILLATION_SCHEMA,
            "active policy pointer",
        )
        candidate_id, lkg_id, active_id = (
            _id(candidate.get("policy_id"), "candidate policy"),
            _id(lkg.get("policy_id"), "LKG policy"),
            _id(active.get("policy_id"), "active policy"),
        )
        if baseline.get("artifact_id") != baseline_id:
            raise EvidenceError("baseline state/artifact mismatch")
        policies: dict[str, Mapping[str, Any]] = {}
        for policy_id in (candidate_id, lkg_id, active_id):
            policies[policy_id], _, _ = _read_sealed_artifact(
                store.distillation_dir(root) / "policies" / f"{policy_id}.json",
                rollout.POLICY_SCHEMA,
                "rollout policy",
            )
    except (OSError, EvidenceError, store.DistillationStoreError) as exc:
        raise EvidenceError("sealed rollout state unavailable") from exc
    percent = {"shadow": 0, "5": 5, "25": 25, "100": 100}[stage]
    if stage == "shadow":
        expected_active = lkg_id
    elif stage in {"5", "25"}:
        expected_active = lkg_id
        if state.get("status") != "canary" or state.get("rollout_percent") != percent:
            raise EvidenceError("canary state does not match stage")
    else:
        if state.get("status") != "canary" or state.get("rollout_percent") != 100:
            raise EvidenceError("100 stage is not gate-authorized")
        expected_active = candidate_id
    if active_id != expected_active:
        raise EvidenceError("active policy violates stage semantics")
    candidate_policy = policies[candidate_id]
    if (
        candidate_policy.get("feature_keys") != list(distillation.FAST_FEATURE_KEYS)
        or candidate_policy.get("feature_revision")
        != distillation.TEXT_FEATURE_REVISION
    ):
        raise EvidenceError("candidate policy feature contract mismatch")
    feature = _digest(
        {
            "feature_keys": candidate_policy["feature_keys"],
            "feature_revision": candidate_policy["feature_revision"],
            "weights": candidate_policy.get("weights"),
            "bias": candidate_policy.get("bias"),
            "threshold": candidate_policy.get("threshold"),
            "abstain_margin": candidate_policy.get("abstain_margin"),
        }
    )
    return {
        "baseline_id": baseline_id,
        "candidate_id": candidate_id,
        "lkg_id": lkg_id,
        "active_id": active_id,
        "candidate_feature_contract_sha256": feature,
    }


def _runtime_observations(
    root: Path, stage: str, identities: Mapping[str, str | None]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Re-derive pairs only from the runtime's checkpoint-backed receipt chain."""
    if _has_symlink_component(root):
        raise EvidenceError("runtime observation root is symlinked")
    ledger_path = store.distillation_dir(root) / "shadow-observation-receipts.jsonl"
    receipts, head = _readonly_chain_snapshot(ledger_path)
    state = _read_sealed_state(
        store.distillation_dir(root) / store.STATE_FILE,
        store.DISTILLATION_SCHEMA,
        "rollout state",
    )
    expected_runtime_stage = "shadow" if stage == "shadow" else "canary"
    expected_percent = {"shadow": 0, "5": 5, "25": 25, "100": 100}[stage]
    if (
        expected_runtime_stage == "canary"
        and state.get("rollout_percent") != expected_percent
    ):
        raise EvidenceError("runtime canary percent does not match collector stage")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for receipt in receipts:
        keys = (
            "decision_id",
            "host",
            "session_id_sha256",
            "query_semantic_sha256",
            "policy_id",
            "incumbent_policy_id",
            "served_policy_id",
            "stage",
            "stage_started_at",
            "qualified_run_id",
            "selected_candidate_ids",
            "incumbent_selected_candidate_ids",
            "paired_eligible",
            "candidate_pool_sha256",
            "candidate_feature_snapshot_sha256",
            "runtime_observation_sha256",
            "operational_evidence_sha256",
            "observed_at",
        )
        binding = {key: receipt.get(key) for key in keys}
        artifact_id = receipt.get("shadow_observation_artifact_id")
        if (
            receipt.get("kind") != "shadow-policy-observation"
            or receipt.get("binding_sha256") != _digest(binding)
            or not isinstance(artifact_id, str)
            or _HEX.fullmatch(artifact_id) is None
        ):
            continue
        try:
            artifact, _, _ = _read_sealed_artifact(
                store.distillation_dir(root)
                / "shadow-observations"
                / f"{artifact_id}.json",
                distillation.SHADOW_OBSERVATION_SCHEMA,
                "shadow observation",
            )
        except (EvidenceError, store.DistillationStoreError):
            continue
        if artifact.get("artifact_id") != artifact_id or any(
            artifact.get(key) != value for key, value in binding.items()
        ):
            continue
        if (
            artifact.get("policy_id") != identities["candidate_id"]
            or artifact.get("incumbent_policy_id") != identities["lkg_id"]
            or artifact.get("stage") != expected_runtime_stage
            or artifact.get("stage_started_at") != state.get("stage_started_at")
            or artifact.get("qualified_run_id") != state.get("stage_run_id")
            or artifact.get("paired_eligible") is not True
        ):
            continue
        observation = artifact.get("runtime_observation")
        evidence = artifact.get("operational_evidence")
        selected = artifact.get("selected_candidate_ids")
        incumbent_selected = artifact.get("incumbent_selected_candidate_ids")
        if (
            not isinstance(observation, Mapping)
            or set(observation)
            != {"decision", "selected_count", "evaluated_count", "latency_ms", "timed_out", "error_code"}
            or not isinstance(evidence, Mapping)
            or set(evidence)
            != {
                "candidate_quality",
                "baseline_quality",
                "candidate_anchor_retained",
                "baseline_anchor_retained",
                "resource_ok",
                "integrity_ok",
                "negative_veto",
                "deadline_ms",
                "feature_snapshot_sha256",
                "candidate_feature_bytes_sha256",
                "baseline_feature_bytes_sha256",
                "feature_parity",
            }
            or not isinstance(selected, list)
            or not isinstance(incumbent_selected, list)
            or not all(isinstance(value, str) for value in (*selected, *incumbent_selected))
            or _digest(evidence) != artifact.get("operational_evidence_sha256")
            or _digest(observation) != artifact.get("runtime_observation_sha256")
            or artifact.get("candidate_feature_snapshot_sha256")
            != evidence.get("feature_snapshot_sha256")
        ):
            continue
        try:
            latency = float(observation["latency_ms"])
            timed_out = observation["timed_out"]
        except (KeyError, TypeError, ValueError):
            continue
        if not 0 <= latency <= 60_000 or not isinstance(timed_out, bool):
            continue
        if any(
            not isinstance(evidence.get(key), bool)
            for key in (
                "candidate_quality",
                "baseline_quality",
                "candidate_anchor_retained",
                "baseline_anchor_retained",
                "resource_ok",
                "integrity_ok",
                "negative_veto",
                "feature_parity",
            )
        ) or (
            isinstance(evidence.get("deadline_ms"), bool)
            or not isinstance(evidence.get("deadline_ms"), int)
            or not 1 <= evidence["deadline_ms"] <= 1_200
        ):
            continue
        observation_id = _id(artifact_id, "shadow observation")
        if observation_id in seen:
            raise EvidenceError("duplicate runtime shadow observation")
        seen.add(observation_id)
        rows.append(
            {
                "observation_id": observation_id,
                "decision_sha256": _digest(artifact["decision_id"]),
                "session_sha256": artifact["session_id_sha256"],
                "query_sha256": artifact["query_semantic_sha256"],
                "candidate_pool_sha256": artifact["candidate_pool_sha256"],
                "feature_snapshot_sha256": artifact[
                    "candidate_feature_snapshot_sha256"
                ],
                "feature_bytes_sha256": artifact[
                    "candidate_feature_snapshot_sha256"
                ],
                "feature_contract_sha256": identities[
                    "candidate_feature_contract_sha256"
                ],
                "host": artifact["host"],
                "cohort": artifact.get("cohort", artifact["host"]),
                "worker_id": artifact.get("worker_id", "recall-runtime"),
                "candidate_covered": bool(selected),
                "baseline_covered": bool(
                    incumbent_selected
                ),
                "candidate_abstained": not bool(selected),
                "baseline_abstained": not bool(incumbent_selected),
                "candidate_score_ms": int(latency),
                "live_latency_ms": int(latency),
                "timed_out": timed_out,
                "operational_evidence": dict(evidence),
                "observed_at": artifact["observed_at"],
                "run_id": artifact["qualified_run_id"],
                "stage": artifact["stage"],
            }
        )
    return rows, {
        "records": head["records"],
        "head_sha256": head["head_sha256"],
        "stage_run_id": state.get("stage_run_id"),
    }


def _wilson_lower(successes: int, total: int) -> float:
    if total < 1 or not 0 <= successes <= total:
        raise EvidenceError("invalid Wilson denominator")
    z = NormalDist().inv_cdf(0.975)
    point = successes / total
    denominator = 1 + z * z / total
    return float(max(
        0.0,
        (
            point
            + z * z / (2 * total)
            - z * (point * (1 - point) / total + z * z / (4 * total * total)) ** 0.5
        )
        / denominator,
    ))


def _p95(values: Sequence[int]) -> int:
    if not values:
        raise EvidenceError("latency observations absent")
    ordered = sorted(values)
    return ordered[(len(ordered) * 0.95).__ceil__() - 1]


def _ledger_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    raw_bytes, _ = _read_stable_file(path, "poll ledger")
    rows: list[dict[str, Any]] = []
    previous = ""
    raw_lines = raw_bytes.splitlines()
    for index, raw in enumerate(raw_lines):
        try:
            row = json.loads(raw)
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise EvidenceError("ledger is corrupt") from exc
        if not isinstance(row, dict) or raw != canonical_json_line_bytes_strict(row).rstrip(b"\n"):
            raise EvidenceError("ledger is not canonical")
        if (
            not isinstance(row, dict)
            or row.get("schema") != LEDGER_SCHEMA
            or row.get("previous_sha256") != previous
            or set(row)
            != {
                "schema",
                "namespace",
                "poll_id",
                "poll_sha256",
                "stage",
                "observed_at",
                "monotonic_ns",
                "previous_sha256",
                "entry_sha256",
            }
        ):
            raise EvidenceError("ledger chain mismatch")
        _id(row.get("poll_sha256"), "ledger poll artifact")
        unsigned = {key: value for key, value in row.items() if key != "entry_sha256"}
        if row.get("entry_sha256") != _digest(unsigned):
            raise EvidenceError("ledger hash mismatch")
        _id(row.get("poll_id"), "ledger poll")
        _utc(row.get("observed_at"), "ledger observed_at")
        if isinstance(row.get("monotonic_ns"), bool) or not isinstance(
            row.get("monotonic_ns"), int
        ):
            raise EvidenceError("ledger monotonic clock invalid")
        if index and (
            row["observed_at"] <= rows[-1]["observed_at"]
            or row["monotonic_ns"] <= rows[-1]["monotonic_ns"]
        ):
            raise EvidenceError("system clock moved backwards")
        previous = row["entry_sha256"]
        rows.append(row)
    return rows


def _ledger_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"count": 0, "head_sha256": ""}
    try:
        raw, _ = _read_stable_file(path, "ledger state")
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise EvidenceError("ledger state is not object")
        if raw != canonical_json_line_bytes_strict(value):
            raise EvidenceError("ledger state is not canonical")
        value = _sealed(value, store.DISTILLATION_SCHEMA, "ledger state")
    except (OSError, UnicodeError, json.JSONDecodeError, store.DistillationStoreError) as exc:
        raise EvidenceError("ledger state is invalid") from exc
    if (
        value.get("kind") != "r7-poll-ledger-state"
        or isinstance(value.get("count"), bool)
        or not isinstance(value.get("count"), int)
        or value["count"] < 0
        or not isinstance(value.get("head_sha256"), str)
    ):
        raise EvidenceError("ledger state schema mismatch")
    if value["head_sha256"] and _HEX.fullmatch(value["head_sha256"]) is None:
        raise EvidenceError("ledger state head is invalid")
    return {"count": value["count"], "head_sha256": value["head_sha256"]}


def _check_ledger_state(root: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    state = _ledger_state(root / "poll-ledger-state.json")
    head = rows[-1]["entry_sha256"] if rows else ""
    if state != {"count": len(rows), "head_sha256": head}:
        raise EvidenceError("ledger head/count mismatch")


def _append_ledger(
    path: Path,
    poll_id: str,
    stage: str,
    observed_at: datetime,
    monotonic_ns: int,
    *,
    poll_sha256: str | None = None,
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(".lock")
    if _has_symlink_component(path) or _has_symlink_component(lock_path):
        raise EvidenceError("poll ledger path is symlinked")
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        lock_descriptor = os.open(lock_path, flags, 0o600)
    except OSError as exc:
        raise EvidenceError("poll ledger lock unavailable") from exc
    with os.fdopen(lock_descriptor, "a", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        prior_ledger = (
            _read_stable_file(path, "poll ledger")[0] if path.exists() else None
        )
        state_path = path.parent / "poll-ledger-state.json"
        prior_state = (
            _read_stable_file(state_path, "poll ledger state")[0]
            if state_path.exists()
            else None
        )
        rows = _ledger_rows(path)
        _check_ledger_state(path.parent, rows)
        if poll_id in {row["poll_id"] for row in rows}:
            raise EvidenceError("duplicate poll id")
        prior = rows[-1]["entry_sha256"] if rows else ""
        entry = {
            "schema": LEDGER_SCHEMA,
            "namespace": "recall-distillation",
            "poll_id": poll_id,
            "stage": stage,
            "observed_at": observed_at.isoformat(),
            "monotonic_ns": monotonic_ns,
            "previous_sha256": prior,
        }
        if poll_sha256 is None:
            raise EvidenceError("ledger poll artifact hash is required")
        _id(poll_sha256, "ledger poll artifact")
        entry["poll_sha256"] = poll_sha256
        entry["entry_sha256"] = _digest(entry)
        try:
            ledger_flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
            if hasattr(os, "O_NOFOLLOW"):
                ledger_flags |= os.O_NOFOLLOW
            ledger_descriptor = os.open(path, ledger_flags, 0o600)
            with os.fdopen(ledger_descriptor, "a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(entry, sort_keys=True, separators=(",", ":")) + "\n"
                )
                handle.flush()
                os.fsync(handle.fileno())
            store.write_sealed_state(
                state_path,
                {
                    "kind": "r7-poll-ledger-state",
                    "count": len(rows) + 1,
                    "head_sha256": entry["entry_sha256"],
                },
            )
            readback = _ledger_rows(path)
            _check_ledger_state(path.parent, readback)
            if not readback or readback[-1] != entry:
                raise EvidenceError("ledger append readback mismatch")
        except Exception:
            if prior_ledger is None:
                path.unlink(missing_ok=True)
            else:
                path.write_bytes(prior_ledger)
            if prior_state is None:
                state_path.unlink(missing_ok=True)
            else:
                state_path.write_bytes(prior_state)
            raise
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    return str(entry["entry_sha256"])


def _evidence_files(evidence_root: Path) -> dict[str, tuple[int, int, int, str]]:
    """Inventory regular evidence files, excluding lock implementation details."""

    if not evidence_root.exists():
        return {}
    if _has_symlink_component(evidence_root) or not evidence_root.is_dir():
        raise EvidenceError("evidence root is unsafe")
    result: dict[str, tuple[int, int, int, str]] = {}
    for path in evidence_root.rglob("*"):
        if path.name.endswith(".lock") or path.name == ".immutable.lock":
            continue
        if path.is_dir() and not path.is_symlink():
            continue
        if path.is_symlink() or not path.is_file():
            raise EvidenceError("evidence tree contains unsafe entry")
        relative = path.relative_to(evidence_root).as_posix()
        body, metadata = _read_stable_file(path, f"evidence file {relative}")
        result[relative] = (
            metadata["dev"],
            metadata["ino"],
            metadata["mode"],
            hashlib.sha256(body).hexdigest(),
        )
    return result


def _remove_own_artifact(path: Path, expected: bytes) -> None:
    """Remove only an immutable artifact still byte-identical to this writer."""

    try:
        if _has_symlink_component(path) or path.is_symlink():
            raise EvidenceError("collector orphan artifact path is symlinked")
        if not path.exists():
            return
        current, _ = _read_stable_file(path, "collector orphan artifact")
        if current != expected:
            raise EvidenceError("collector orphan artifact was modified")
        path.unlink()
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except EvidenceError:
        raise
    except OSError as exc:
        raise EvidenceError("collector orphan cleanup failed") from exc


def _collect_poll_locked(
    *,
    root: Path,
    source_root: Path,
    evidence_root: Path,
    stage: str,
    run_id: str,
    dashboard_url: str,
    dom_capture_path: Path,
    direct_url_path: Path,
    executable: Path,
    pid: int,
    expected_started_at: object | None = None,
    service_role: str = "dashboard",
    live_attestation_path: Path | None = None,
    live_attestation_artifact_id: str | None = None,
) -> dict[str, Any]:
    """Read one independent poll and append it immutably.  Time is OS supplied."""
    _id(run_id, "run id")
    if service_role not in _SERVICE_LABELS:
        raise EvidenceError("collector service role is not approved")
    if _has_symlink_component(root) or _has_symlink_component(evidence_root):
        raise EvidenceError("collector root/evidence is symlinked")
    if root.resolve() != _FIXED_PRODUCTION_ROOT:
        raise EvidenceError("collector root is not the production runtime")
    if evidence_root.resolve() != _FIXED_EVIDENCE_ROOT.resolve():
        raise EvidenceError("collector evidence root is not managed")
    root = _FIXED_PRODUCTION_ROOT
    evidence_root = _FIXED_EVIDENCE_ROOT
    source_root = _fixed_source_root()
    direct_url_path = _fixed_direct_url_path()
    dom_capture_path = _fixed_dom_capture_path(stage, run_id)
    if evidence_root.resolve(strict=False).is_relative_to(source_root.resolve()):
        raise EvidenceError("evidence output overlaps source checkout")
    evidence_root.mkdir(parents=True, exist_ok=True)
    evidence_before = _evidence_files(evidence_root)
    now = datetime.now(UTC)
    monotonic_ns = time.monotonic_ns()
    identities = _stage_state(root, stage)
    observations, observation_chain = _runtime_observations(root, stage, identities)
    if observation_chain["stage_run_id"] != run_id:
        raise EvidenceError("poll run does not match runtime stage")
    for observation in observations:
        observed_at = _utc(observation.get("observed_at"), "observation time")
        if abs((now - observed_at).total_seconds()) > MAX_OBSERVATION_SKEW_SECONDS:
            raise EvidenceError("observation/poll clock skew is excessive")
    dom, dom_raw, dom_ref = _read_dom_capture(
        dom_capture_path, stage=stage, run_id=run_id
    )
    health = _fetch(
        f"{_FIXED_DASHBOARD_ORIGIN}/api/health",
        "dashboard health",
        fixed_endpoint=True,
    )
    api = _fetch(
        f"{_FIXED_DASHBOARD_ORIGIN}/api/fast-snapshot",
        "dashboard API",
        fixed_endpoint=True,
    )
    if health["status"] != 200 or api["status"] != 200:
        raise EvidenceError("dashboard health/API status is not successful")
    runtime = _direct_url(direct_url_path, formal=True)
    process = _service_process_identity(service_role)
    source = _source_identity(source_root)
    _validate_dashboard_payload(health["payload"], "health", source_commit=source["source_commit"])
    _validate_dashboard_payload(api["payload"], "api")
    health_payload = health["payload"]
    health_runtime = (
        health_payload.get("health", {}).get("runtime", {})
        if isinstance(health_payload.get("health"), Mapping)
        else {}
    )
    if (
        runtime["archive_commit"] != source["source_commit"]
        or not isinstance(health_runtime, Mapping)
        or health_runtime.get("commit_id") != source["source_commit"]
        or health_runtime.get("drift") is not False
    ):
        raise EvidenceError("runtime/archive/dashboard commit drift")
    if live_attestation_path is None:
        if live_attestation_artifact_id is None or _HEX.fullmatch(live_attestation_artifact_id) is None:
            raise EvidenceError("external live attestation is required")
        live_attestation_path = (
            evidence_root
            / "r7-live-attestations"
            / f"{live_attestation_artifact_id}.json"
        )
    elif live_attestation_artifact_id is not None and live_attestation_path.stem != live_attestation_artifact_id:
        raise EvidenceError("live attestation path/id mismatch")
    if live_attestation_path.parent.resolve() != (
        evidence_root / "r7-live-attestations"
    ).resolve():
        raise EvidenceError("live attestation path is outside artifact parent")
    if _has_symlink_component(live_attestation_path):
        raise EvidenceError("live attestation path is unsafe")
    live_attestation_path = (
        evidence_root
        / "r7-live-attestations"
        / f"{live_attestation_path.stem}.json"
    )
    dom_projection = {
        "kind": "browser-dom-capture",
        "synthetic_fixture": False,
        "producer_name": dom["producer"]["name"],
        "producer_version": dom["producer"]["version"],
        "html_sha256": dom["html_sha256"],
        "capture_sha256": hashlib.sha256(dom_raw).hexdigest(),
        **dom_ref,
    }
    live_attestation = _validate_live_attestation_binding(
        live_attestation_path,
        stage=stage,
        run_id=run_id,
        source=source,
        runtime=runtime,
        process=process,
        health=health,
        api=api,
        dom=dom_projection,
    )
    payload = {
        "kind": "r7-live-poll",
        "stage": stage,
        "run_id": run_id,
        "captured_at": now.isoformat(),
        "monotonic_ns": monotonic_ns,
        "identities": identities,
        "source": source,
        "runtime": runtime,
        "process": process,
        "health": {key: value for key, value in health.items() if key != "payload"},
        "api": {key: value for key, value in api.items() if key != "payload"},
        "dom": dom_projection,
        "dom_sha256": _digest(dom_projection),
        "live_attestation": live_attestation,
        "observation_chain": observation_chain,
        "observations_sha256": _digest(observations),
        "observations": observations,
        "producer": {
            "name": "chronovisor-r7-evidence",
            "version": 1,
            "synthetic_fixture": False,
        },
    }
    poll_path: Path | None = None
    poll_bytes: bytes | None = None
    ledger_path = evidence_root / "poll-ledger.jsonl"
    ledger_state_path = evidence_root / "poll-ledger-state.json"
    if ledger_path.is_symlink() or ledger_state_path.is_symlink():
        raise EvidenceError("poll ledger path is symlinked")
    ledger_before = (
        _read_stable_file(ledger_path, "poll ledger")[0]
        if ledger_path.exists()
        else None
    )
    ledger_state_before = (
        _read_stable_file(ledger_state_path, "poll ledger state")[0]
        if ledger_state_path.exists()
        else None
    )
    ledger_written = False
    try:
        poll_id, poll_path, artifact = store.write_immutable(
            evidence_root / "polls", payload, schema=POLL_SCHEMA
        )
        poll_bytes, _ = _read_stable_file(poll_path, "poll artifact")
        # Re-stat every mutable input and source/state after writing.  A changed
        # input is not evidence for this immutable poll.
        dom_after, dom_after_raw, dom_ref_after = _read_dom_capture(
            dom_capture_path, stage=stage, run_id=run_id
        )
        process_after = _service_process_identity(service_role, process["started_at"])
        health_after = _fetch(
            f"{_FIXED_DASHBOARD_ORIGIN}/api/health",
            "dashboard health",
            fixed_endpoint=True,
        )
        api_after = _fetch(
            f"{_FIXED_DASHBOARD_ORIGIN}/api/fast-snapshot",
            "dashboard API",
            fixed_endpoint=True,
        )
        _validate_dashboard_payload(
            health_after["payload"], "health", source_commit=source["source_commit"]
        )
        _validate_dashboard_payload(api_after["payload"], "api")
        observations_after, observation_chain_after = _runtime_observations(
            root, stage, identities
        )
        live_attestation_after = _attestation_ref(live_attestation_path)
        if (
            dom_after_raw != dom_raw
            or _direct_url(direct_url_path, formal=True) != runtime
            or not _same_process_identity(process_after, process)
            or {
                key: health_after[key]
                for key in ("url", "status", "body_sha256", "payload_sha256")
            }
            != {key: health[key] for key in ("url", "status", "body_sha256", "payload_sha256")}
            or {
                key: api_after[key]
                for key in ("url", "status", "body_sha256", "payload_sha256")
            }
            != {key: api[key] for key in ("url", "status", "body_sha256", "payload_sha256")}
            or live_attestation_after != live_attestation
            or _source_identity(source_root) != source
            or _stage_state(root, stage) != identities
            or observation_chain_after != observation_chain
            or observations_after != observations
            or dom_after != dom
            or dom_ref_after != dom_ref
        ):
            raise EvidenceError("TOCTOU input/state drift")
        evidence_after = _evidence_files(evidence_root)
        own_name = poll_path.relative_to(evidence_root).as_posix()
        evidence_after.pop(own_name, None)
        if evidence_after != evidence_before:
            raise EvidenceError("evidence root changed during poll")
        ledger = _append_ledger(
            ledger_path,
            poll_id,
            stage,
            now,
            monotonic_ns,
            poll_sha256=hashlib.sha256(poll_bytes).hexdigest(),
        )
        ledger_written = True
        # The immutable poll and its ledger row must be readable as one
        # publication boundary before the caller can observe success.
        readback, readback_bytes, _ = _read_sealed_artifact(
            poll_path, POLL_SCHEMA, "poll artifact"
        )
        if readback != artifact or readback_bytes != poll_bytes:
            raise EvidenceError("poll immutable readback mismatch")
        rows = _ledger_rows(evidence_root / "poll-ledger.jsonl")
        if not rows or rows[-1].get("entry_sha256") != ledger:
            raise EvidenceError("poll ledger readback mismatch")
        evidence_final = _evidence_files(evidence_root)
        evidence_final.pop(own_name, None)
        for ledger_name in ("poll-ledger.jsonl", "poll-ledger-state.json"):
            evidence_final.pop(ledger_name, None)
        evidence_expected = dict(evidence_before)
        evidence_expected.pop("poll-ledger.jsonl", None)
        evidence_expected.pop("poll-ledger-state.json", None)
        if evidence_final != evidence_expected:
            raise EvidenceError("evidence root changed after poll publication")
    except Exception:
        if ledger_written:
            if ledger_before is None:
                ledger_path.unlink(missing_ok=True)
            else:
                ledger_path.write_bytes(ledger_before)
            if ledger_state_before is None:
                ledger_state_path.unlink(missing_ok=True)
            else:
                ledger_state_path.write_bytes(ledger_state_before)
        if poll_path is not None and poll_bytes is not None:
            _remove_own_artifact(poll_path, poll_bytes)
        raise
    return {
        "poll_id": poll_id,
        "poll_sha256": hashlib.sha256(poll_bytes).hexdigest(),
        "ledger_entry_sha256": ledger,
        "captured_at": now.isoformat(),
    }


def collect_poll(
    *,
    root: Path,
    source_root: Path,
    evidence_root: Path,
    stage: str,
    run_id: str,
    dashboard_url: str,
    dom_capture_path: Path,
    direct_url_path: Path,
    executable: Path,
    pid: int,
    expected_started_at: object | None = None,
    service_role: str = "dashboard",
    live_attestation_path: Path | None = None,
    live_attestation_artifact_id: str | None = None,
) -> dict[str, Any]:
    """Serialize collectors so publication cannot race another poll writer."""

    if _has_symlink_component(root) or _has_symlink_component(evidence_root):
        raise EvidenceError("collector root/evidence is symlinked")
    if root.resolve() != _FIXED_PRODUCTION_ROOT:
        raise EvidenceError("collector root is not the production runtime")
    if evidence_root.resolve() != _FIXED_EVIDENCE_ROOT.resolve():
        raise EvidenceError("collector evidence root is not managed")
    # The lock and all subsequent operations must use the fixed production
    # paths themselves; a caller alias is never an operational authority.
    root = _FIXED_PRODUCTION_ROOT
    evidence_root = _FIXED_EVIDENCE_ROOT
    if _has_symlink_component(evidence_root):
        raise EvidenceError("symlinked collector root/input")
    lock_path = evidence_root / ".collector.lock"
    if _has_symlink_component(lock_path):
        raise EvidenceError("collector lock is symlinked")
    lock_flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        lock_flags |= os.O_NOFOLLOW
    try:
        lock_descriptor = os.open(lock_path, lock_flags, 0o600)
        with os.fdopen(lock_descriptor, "a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            return _collect_poll_locked(
                root=root,
                source_root=source_root,
                evidence_root=evidence_root,
                stage=stage,
                run_id=run_id,
                dashboard_url=dashboard_url,
                dom_capture_path=dom_capture_path,
                direct_url_path=direct_url_path,
                executable=executable,
                pid=pid,
                expected_started_at=expected_started_at,
                service_role=service_role,
                live_attestation_path=live_attestation_path,
                live_attestation_artifact_id=live_attestation_artifact_id,
            )
    except OSError as exc:
        raise EvidenceError("collector lock unavailable") from exc


def _poll_files(evidence_root: Path) -> set[str]:
    directory = evidence_root / "polls"
    if not directory.exists():
        return set()
    if directory.is_symlink() or not directory.is_dir():
        raise EvidenceError("poll artifact directory is unsafe")
    result: set[str] = set()
    for path in directory.iterdir():
        if path.name == ".immutable.lock":
            continue
        if path.is_symlink() or not path.is_file() or path.suffix != ".json":
            raise EvidenceError("poll artifact path is unsafe")
        poll_id = path.stem
        if _HEX.fullmatch(poll_id) is None:
            raise EvidenceError("poll artifact name is invalid")
        result.add(poll_id)
    return result


def _poll_dimensions(poll: Mapping[str, Any]) -> bool:
    """Reject hand-shaped polls before they can contribute to a certificate."""
    top = {
        "schema",
        "namespace",
        "artifact_id",
        "seal_sha256",
        "kind",
        "stage",
        "run_id",
        "captured_at",
        "monotonic_ns",
        "identities",
        "source",
        "runtime",
        "process",
        "health",
        "api",
        "dom",
        "dom_sha256",
        "live_attestation",
        "observation_chain",
        "observations_sha256",
        "observations",
        "producer",
    }
    if set(poll) != top or poll.get("kind") != "r7-live-poll":
        return False
    stage = poll.get("stage")
    run_id = poll.get("run_id")
    if (
        stage not in STAGES
        or not isinstance(run_id, str)
        or _HEX.fullmatch(run_id) is None
        or not isinstance(poll.get("captured_at"), str)
        or isinstance(poll.get("monotonic_ns"), bool)
        or not isinstance(poll.get("monotonic_ns"), int)
        or poll["monotonic_ns"] < 1
        or not isinstance(poll.get("artifact_id"), str)
        or _HEX.fullmatch(poll["artifact_id"]) is None
    ):
        return False
    producer = poll.get("producer")
    if (
        not isinstance(producer, Mapping)
        or set(producer) != {"name", "version", "synthetic_fixture"}
        or producer.get("name") != "chronovisor-r7-evidence"
        or isinstance(producer.get("version"), bool)
        or not isinstance(producer.get("version"), int)
        or producer.get("version") != 1
        or producer.get("synthetic_fixture") is not False
    ):
        return False
    identities = poll.get("identities")
    if not isinstance(identities, Mapping) or set(identities) != {
        "baseline_id",
        "candidate_id",
        "lkg_id",
        "active_id",
        "candidate_feature_contract_sha256",
    }:
        return False
    if not all(_HEX.fullmatch(str(identities.get(key))) for key in identities):
        return False
    source = poll.get("source")
    runtime = poll.get("runtime")
    process = poll.get("process")
    health = poll.get("health")
    api = poll.get("api")
    if not isinstance(source, Mapping) or not isinstance(runtime, Mapping):
        return False
    source_keys = {
        "source_commit",
        "source_clean",
        "source_tree_sha256",
        "source_bytes_sha256",
        "git",
        "tracked",
    }
    if (
        set(source) != source_keys
        or source.get("source_clean") != "true"
        or not isinstance(source.get("source_commit"), str)
        or _COMMIT.fullmatch(source["source_commit"]) is None
        or not all(
            isinstance(source.get(key), str)
            and _HEX.fullmatch(source[key]) is not None
            for key in ("source_tree_sha256", "source_bytes_sha256")
        )
        or not isinstance(source.get("git"), Mapping)
        or set(source["git"])
        != {
            "git_dir",
            "worktree",
            "index",
            "head",
            "status_sha256",
            "tree_sha256",
            "head_object_sha256",
            "tree_object_sha256",
        }
        or not isinstance(source.get("tracked"), list)
    ):
        return False
    if not all(
        isinstance(value, Mapping)
        and set(value) == {"path", "lstat", "bytes_sha256"}
        and isinstance(value.get("path"), str)
        and Path(value["path"]).is_absolute()
        and not _has_symlink_component(Path(value["path"]))
        and isinstance(value.get("bytes_sha256"), str)
        and _HEX.fullmatch(value["bytes_sha256"]) is not None
        and isinstance(value.get("lstat"), Mapping)
        and set(value["lstat"]) == _STAT_KEYS
        and all(
            isinstance(value["lstat"].get(key), int)
            and not isinstance(value["lstat"].get(key), bool)
            for key in _STAT_KEYS
        )
        for value in (source["git"]["git_dir"], source["git"]["worktree"])
    ):
        return False
    for name in ("index", "head"):
        value = source["git"].get(name)
        if (
            not isinstance(value, Mapping)
            or set(value) != {"path", "lstat", "bytes_sha256"}
            or not isinstance(value.get("path"), str)
            or not Path(value["path"]).is_absolute()
            or _has_symlink_component(Path(value["path"]))
            or not isinstance(value.get("lstat"), Mapping)
            or set(value["lstat"]) != _STAT_KEYS
            or any(
                isinstance(value["lstat"].get(key), bool)
                or not isinstance(value["lstat"].get(key), int)
                for key in _STAT_KEYS
            )
            or not isinstance(value.get("bytes_sha256"), str)
            or _HEX.fullmatch(value["bytes_sha256"]) is None
        ):
            return False
    if (
        not all(
            _HEX.fullmatch(str(source["git"].get(key)))
            for key in (
                "status_sha256",
                "tree_sha256",
                "head_object_sha256",
                "tree_object_sha256",
            )
        )
        or not isinstance(source["tracked"], list)
        or any(
            not isinstance(row, list)
            or len(row) != 4
            or row[0] not in {"100644", "100755"}
            or not isinstance(row[1], str)
            or Path(row[1]).is_absolute()
            or ".." in Path(row[1]).parts
            or not isinstance(row[2], str)
            or _HEX.fullmatch(row[2]) is None
            or not isinstance(row[3], str)
            or _HEX.fullmatch(row[3]) is None
            for row in source["tracked"]
        )
    ):
        return False
    if set(runtime) != _RUNTIME_KEYS:
        return False
    if (
        not isinstance(runtime.get("archive_commit"), str)
        or _COMMIT.fullmatch(runtime["archive_commit"]) is None
        or any(
            not isinstance(runtime.get(key), str)
            or _HEX.fullmatch(runtime[key]) is None
            for key in (
                "direct_url_sha256",
                "direct_url_raw_sha256",
                "direct_url_payload_sha256",
                "module_bytes_sha256",
            )
        )
        or runtime["direct_url_sha256"] != runtime["direct_url_payload_sha256"]
        or runtime["archive_commit"] != source["source_commit"]
        or not isinstance(runtime.get("module_path"), str)
        or not Path(runtime["module_path"]).is_absolute()
        or _has_symlink_component(Path(runtime["module_path"]))
        or not isinstance(runtime.get("module_lstat"), Mapping)
        or set(runtime["module_lstat"]) != _STAT_KEYS
        or any(
            isinstance(runtime["module_lstat"].get(key), bool)
            or not isinstance(runtime["module_lstat"].get(key), int)
            for key in _STAT_KEYS
        )
        or runtime.get("distribution_name") != "chronovisor"
        or not isinstance(runtime.get("distribution_version"), str)
        or not runtime["distribution_version"]
    ):
        return False
    if (
        not isinstance(runtime.get("record_path"), str)
        or not Path(runtime["record_path"]).is_absolute()
        or _has_symlink_component(Path(runtime["record_path"]))
        or Path(runtime["record_path"]).name != "RECORD"
        or any(
            not isinstance(runtime.get(key), str)
            or _HEX.fullmatch(runtime[key]) is None
            for key in (
                "record_file_sha256",
                "record_module_sha256",
                "tracked_blob_sha1",
            )
        )
        or isinstance(runtime.get("record_module_size"), bool)
        or not isinstance(runtime.get("record_module_size"), int)
        or runtime["record_module_size"] < 0
        or not isinstance(runtime.get("tracked_path"), str)
        or Path(runtime["tracked_path"]).is_absolute()
        or ".." in Path(runtime["tracked_path"]).parts
        or runtime["tracked_path"]
        not in {
            "src/chronovisor/recall/recall_r7_evidence.py",
            "chronovisor/recall/recall_r7_evidence.py",
        }
        or runtime.get("tracked_mode") not in {"100644", "100755"}
    ):
        return False
    if (
        not isinstance(process, Mapping)
        or set(process)
        != {
            "pid",
            "started_at",
            "executable_path",
            "executable_lstat",
            "executable_sha256",
            "service",
        }
        or isinstance(process.get("pid"), bool)
        or not isinstance(process.get("pid"), int)
        or process["pid"] <= 0
        or not isinstance(process.get("started_at"), str)
        or _STARTED_AT.fullmatch(process["started_at"]) is None
        or not isinstance(process.get("executable_path"), str)
        or not Path(process["executable_path"]).is_absolute()
        or _has_symlink_component(Path(process["executable_path"]))
        or not isinstance(process.get("executable_lstat"), Mapping)
        or set(process["executable_lstat"])
        != _STAT_KEYS
        or any(
            isinstance(process["executable_lstat"].get(key), bool)
            or not isinstance(process["executable_lstat"].get(key), int)
            for key in _STAT_KEYS
        )
        or not isinstance(process.get("executable_sha256"), str)
        or _HEX.fullmatch(process["executable_sha256"]) is None
    ):
        return False
    service = process.get("service")
    service_parent_pid = service.get("parent_pid") if isinstance(service, Mapping) else None
    if (
        not isinstance(service, Mapping)
        or set(service)
        != {
            "role",
            "domain",
            "label",
            "state",
            "parent_pid",
            "child_pid",
            "captured_at",
            "raw_output_sha256",
        }
        or service.get("role") not in _SERVICE_LABELS
        or service.get("label") != _SERVICE_LABELS[service["role"]]
            or service.get("domain") != f"gui/{os.getuid()}"
            or service.get("state") != "running"
            or isinstance(service.get("parent_pid"), bool)
            or not isinstance(service_parent_pid, int)
        or service_parent_pid <= 0
        or service.get("child_pid") != process.get("pid")
        or not isinstance(service.get("captured_at"), str)
        or _utc(service.get("captured_at"), "launchd capture time") is None
        or not isinstance(service.get("raw_output_sha256"), str)
        or _HEX.fullmatch(service["raw_output_sha256"]) is None
    ):
        return False
    for endpoint_name, endpoint in (("health", health), ("api", api)):
        if (
            not isinstance(endpoint, Mapping)
            or set(endpoint) != {"url", "status", "body_sha256", "payload_sha256"}
            or not _valid_local_endpoint(
                endpoint.get("url"),
                "/api/health" if endpoint_name == "health" else "/api/fast-snapshot",
            )
            or endpoint.get("status") != 200
            or not isinstance(endpoint.get("body_sha256"), str)
            or _HEX.fullmatch(endpoint["body_sha256"]) is None
            or not isinstance(endpoint.get("payload_sha256"), str)
            or _HEX.fullmatch(endpoint["payload_sha256"]) is None
        ):
            return False
    dom = poll.get("dom")
    if (
        not isinstance(dom, Mapping)
        or set(dom)
        != {
            "kind",
            "synthetic_fixture",
            "producer_name",
            "producer_version",
            "html_sha256",
            "capture_sha256",
            "capture_artifact_id",
            "capture_file_sha256",
            "capture_seal_sha256",
        }
        or dom.get("kind") != "browser-dom-capture"
        or dom.get("synthetic_fixture") is not False
        or dom.get("producer_name") != "chronovisor-browser"
        or isinstance(dom.get("producer_version"), bool)
        or not isinstance(dom.get("producer_version"), int)
        or not isinstance(dom.get("html_sha256"), str)
        or _HEX.fullmatch(dom["html_sha256"]) is None
        or not isinstance(dom.get("capture_sha256"), str)
        or _HEX.fullmatch(dom["capture_sha256"]) is None
        or any(
            not isinstance(dom.get(key), str) or _HEX.fullmatch(dom[key]) is None
            for key in (
                "capture_artifact_id",
                "capture_file_sha256",
                "capture_seal_sha256",
            )
        )
        or not isinstance(poll.get("dom_sha256"), str)
        or _HEX.fullmatch(poll["dom_sha256"]) is None
        or poll.get("dom_sha256") != _digest(dom)
    ):
        return False
    attestation = poll.get("live_attestation")
    if (
        not isinstance(attestation, Mapping)
        or set(attestation) != {"artifact_id", "file_sha256", "seal_sha256"}
        or not all(isinstance(attestation.get(key), str) for key in attestation)
        or _HEX.fullmatch(attestation["artifact_id"]) is None
        or _HEX.fullmatch(attestation["file_sha256"]) is None
        or _HEX.fullmatch(attestation["seal_sha256"]) is None
    ):
        return False
    chain = poll.get("observation_chain")
    if (
        not isinstance(chain, Mapping)
        or set(chain) != {"records", "head_sha256", "stage_run_id"}
        or isinstance(chain.get("records"), bool)
        or not isinstance(chain.get("records"), int)
        or chain["records"] < 0
        or not isinstance(chain.get("head_sha256"), str)
        or (chain["head_sha256"] and _HEX.fullmatch(chain["head_sha256"]) is None)
        or chain.get("stage_run_id") != run_id
    ):
        return False
    observations = poll.get("observations")
    if not isinstance(observations, list) or poll.get("observations_sha256") != _digest(observations):
        return False
    row_keys = {
        "observation_id", "decision_sha256", "session_sha256", "query_sha256",
        "candidate_pool_sha256", "feature_snapshot_sha256", "feature_bytes_sha256",
        "feature_contract_sha256", "host", "cohort", "worker_id", "candidate_covered",
        "baseline_covered", "candidate_abstained", "baseline_abstained", "candidate_score_ms",
        "live_latency_ms", "timed_out", "operational_evidence", "observed_at", "run_id", "stage",
    }
    for row in observations:
        if (
            not isinstance(row, Mapping)
            or set(row) != row_keys
            or row.get("run_id") != run_id
            or row.get("stage") != stage
            or any(
                not isinstance(row.get(key), str) or _HEX.fullmatch(row[key]) is None
                for key in (
                    "observation_id", "decision_sha256", "session_sha256", "query_sha256",
                    "candidate_pool_sha256", "feature_snapshot_sha256", "feature_bytes_sha256",
                    "feature_contract_sha256",
                )
            )
            or any(not isinstance(row.get(key), str) or not row[key] for key in ("host", "cohort", "worker_id", "observed_at"))
            or any(not isinstance(row.get(key), bool) for key in ("candidate_covered", "baseline_covered", "candidate_abstained", "baseline_abstained", "timed_out"))
            or any(isinstance(row.get(key), bool) or not isinstance(row.get(key), int) or row[key] < 0 for key in ("candidate_score_ms", "live_latency_ms"))
            or not isinstance(row.get("operational_evidence"), Mapping)
        ):
            return False
        try:
            poll_time = _utc(poll["captured_at"], "poll time")
            observation_time = _utc(row["observed_at"], "observation time")
        except EvidenceError:
            return False
        if abs((poll_time - observation_time).total_seconds()) > MAX_OBSERVATION_SKEW_SECONDS:
            return False
    return True


def _verify_runtime_poll(root: Path, poll: Mapping[str, Any]) -> bool:
    """Require every collector row to remain derivable from protected runtime state."""
    try:
        ledger_path = store.distillation_dir(root) / "shadow-observation-receipts.jsonl"
        chain, source_head = _readonly_chain_snapshot(ledger_path)
        state = _read_sealed_state(
            store.distillation_dir(root) / store.STATE_FILE,
            store.DISTILLATION_SCHEMA,
            "rollout state",
        )
    except (OSError, EvidenceError):
        return False
    if (
        poll.get("run_id") != state.get("stage_run_id")
        or poll.get("run_id") != state.get("last_run_id")
    ):
        return False
    expected_head = poll.get("observation_chain")
    if not isinstance(expected_head, Mapping) or source_head.get("records") != expected_head.get(
        "records"
    ) or source_head.get("head_sha256") != expected_head.get("head_sha256"):
        return False
    if expected_head.get("head_sha256") not in {"", chain[-1].get("record_sha256") if chain else ""}:
        return False
    receipts = {
        row.get("shadow_observation_artifact_id"): row
        for row in chain
        if row.get("kind") == "shadow-policy-observation"
    }
    for row in poll.get("observations", []):
        if not isinstance(row, Mapping):
            return False
        artifact_id = row.get("observation_id")
        receipt = receipts.get(artifact_id)
        if (
            not isinstance(artifact_id, str)
            or _HEX.fullmatch(artifact_id) is None
            or Path(artifact_id).name != artifact_id
            or not isinstance(receipt, Mapping)
        ):
            return False
        try:
            artifact, _, _ = _read_sealed_artifact(
                store.distillation_dir(root)
                / "shadow-observations"
                / f"{artifact_id}.json",
                distillation.SHADOW_OBSERVATION_SCHEMA,
                "shadow observation",
            )
        except (EvidenceError, store.DistillationStoreError):
            return False
        if (
            artifact.get("artifact_id") != artifact_id
            or artifact.get("decision_id") != receipt.get("decision_id")
            or row.get("session_sha256") != artifact.get("session_id_sha256")
            or row.get("query_sha256") != artifact.get("query_semantic_sha256")
            or row.get("candidate_pool_sha256") != artifact.get("candidate_pool_sha256")
            or row.get("feature_snapshot_sha256")
            != artifact.get("candidate_feature_snapshot_sha256")
            or row.get("feature_bytes_sha256")
            != artifact.get("candidate_feature_snapshot_sha256")
            or row.get("feature_contract_sha256")
            != poll.get("identities", {}).get("candidate_feature_contract_sha256")
            or row.get("operational_evidence")
            != artifact.get("operational_evidence")
            or _digest(artifact.get("operational_evidence"))
            != artifact.get("operational_evidence_sha256")
            or _digest(artifact.get("runtime_observation"))
            != artifact.get("runtime_observation_sha256")
            or row.get("stage") != artifact.get("stage")
            or row.get("run_id") != artifact.get("qualified_run_id")
            or row.get("observed_at") != artifact.get("observed_at")
        ):
            return False
    return True


def _current_formal_inputs(
    stage: str, run_id: str, service_role: str
) -> dict[str, Any]:
    """Re-acquire every production authority for final poll validation."""

    source_root = _fixed_source_root()
    direct_url_path = _fixed_direct_url_path()
    dom_path = _fixed_dom_capture_path(stage, run_id)
    source = _source_identity(source_root)
    runtime = _direct_url(direct_url_path, formal=True)
    process = _service_process_identity(service_role)
    health = _fetch(
        f"{_FIXED_DASHBOARD_ORIGIN}/api/health",
        "dashboard health",
        fixed_endpoint=True,
    )
    api = _fetch(
        f"{_FIXED_DASHBOARD_ORIGIN}/api/fast-snapshot",
        "dashboard API",
        fixed_endpoint=True,
    )
    _validate_dashboard_payload(health["payload"], "health", source_commit=source["source_commit"])
    _validate_dashboard_payload(api["payload"], "api")
    dom, dom_raw, dom_ref = _read_dom_capture(
        dom_path, stage=stage, run_id=run_id
    )
    dom_projection = {
        "kind": "browser-dom-capture",
        "synthetic_fixture": False,
        "producer_name": dom["producer"]["name"],
        "producer_version": dom["producer"]["version"],
        "html_sha256": dom["html_sha256"],
        "capture_sha256": hashlib.sha256(dom_raw).hexdigest(),
        **dom_ref,
    }
    return {
        "source": source,
        "runtime": runtime,
        "process": process,
        "health": {key: health[key] for key in ("url", "status", "body_sha256", "payload_sha256")},
        "api": {key: api[key] for key in ("url", "status", "body_sha256", "payload_sha256")},
        "dom": dom_projection,
    }


def _verify_live_attestation_poll(
    evidence_root: Path, poll: Mapping[str, Any], *, root: Path | None = None
) -> bool:
    try:
        reference = poll.get("live_attestation")
        if not isinstance(reference, Mapping):
            return False
        artifact_id = reference.get("artifact_id")
        if not isinstance(artifact_id, str) or _HEX.fullmatch(artifact_id) is None:
            return False
        path = evidence_root / "r7-live-attestations" / f"{artifact_id}.json"
        artifact, body, _ = _read_sealed_artifact(
            path, LIVE_ATTESTATION_SCHEMA, "live attestation"
        )
        if _artifact_ref_values(artifact, body) != dict(reference):
            return False
        _validate_attestation_payload(
            artifact,
            expected_stage=str(poll.get("stage")),
            expected_run_id=str(poll.get("run_id")),
        )
        if abs(
            (
                _utc(artifact.get("captured_at"), "live attestation capture time")
                - _utc(poll.get("captured_at"), "poll time")
            ).total_seconds()
        ) > MAX_OBSERVATION_SKEW_SECONDS:
            return False
        valid = not (
            artifact.get("source")
            != {
                key: poll["source"][key]
                for key in ("source_commit", "source_tree_sha256", "source_bytes_sha256")
            }
            or artifact.get("runtime") != poll.get("runtime")
            or not isinstance(artifact.get("process"), Mapping)
            or not isinstance(poll.get("process"), Mapping)
            or not _same_process_identity(artifact["process"], poll["process"])
            or artifact.get("health") != poll.get("health")
            or artifact.get("api") != poll.get("api")
            or artifact.get("dom") != poll.get("dom")
        )
        if not valid:
            return False
        if root is not None and root.resolve() == _FIXED_PRODUCTION_ROOT:
            process_value = poll.get("process")
            if (
                not isinstance(process_value, Mapping)
                or not isinstance(process_value.get("service"), Mapping)
                or not isinstance(process_value["service"].get("role"), str)
            ):
                return False
            current = _current_formal_inputs(
                str(poll.get("stage")),
                str(poll.get("run_id")),
                process_value["service"]["role"],
            )
            return all(
                poll.get(key) == value
                and artifact.get(key) == value
                for key, value in current.items()
            )
        return True
    except (
        EvidenceError,
        OSError,
        store.DistillationStoreError,
        TypeError,
        KeyError,
        ValueError,
    ):
        return False


def _held_collector(reason: str, polls: int = 0) -> dict[str, Any]:
    return {
        "schema": EVIDENCE_SCHEMA,
        "certification": False,
        "certification_reason": reason,
        "stages": {stage: {"certified": False, "reason": reason} for stage in STAGES},
        "polls": polls,
        "protected_state_unchanged": False,
        "identity": {},
        "source": {},
    }


def _fail_closed_collector(function: Any) -> Any:
    def wrapped(evidence_root: Path, *, root: Path | None = None) -> dict[str, Any]:
        try:
            result = function(evidence_root, root=root)
            if not isinstance(result, Mapping):
                raise EvidenceError("collector result is not an object")
            return dict(result)
        except (
            EvidenceError,
            OSError,
            KeyError,
            TypeError,
            ValueError,
            store.DistillationStoreError,
        ):
            return _held_collector("collector_bundle_invalid")

    return wrapped


@_fail_closed_collector
def validate_collector(
    evidence_root: Path, *, root: Path | None = None
) -> dict[str, Any]:
    """Recompute the only certifiable facts from immutable collector output."""
    if root is not None and not _has_symlink_component(root):
        try:
            is_production_root = root.resolve() == _FIXED_PRODUCTION_ROOT
        except OSError:
            is_production_root = False
        if is_production_root and (
            _has_symlink_component(evidence_root)
            or evidence_root.resolve() != _FIXED_EVIDENCE_ROOT
        ):
            return _held_collector("collector_evidence_authority_invalid")
    try:
        ledger = _ledger_rows(evidence_root / "poll-ledger.jsonl")
        _check_ledger_state(evidence_root, ledger)
        if _poll_files(evidence_root) != {str(row["poll_id"]) for row in ledger}:
            return _held_collector("collector_orphan_poll")
    except EvidenceError:
        return _held_collector("collector_ledger_invalid")
    polls: list[dict[str, Any]] = []
    for entry in ledger:
        path = evidence_root / "polls" / f"{entry['poll_id']}.json"
        try:
            poll, poll_raw, _ = _read_sealed_artifact(path, POLL_SCHEMA, "poll artifact")
        except store.DistillationStoreError:
            return _held_collector("collector_poll_invalid", len(polls))
        if entry.get("poll_sha256") != hashlib.sha256(poll_raw).hexdigest():
            raise EvidenceError("poll ledger/artifact bytes mismatch")
        if (
            poll.get("artifact_id") != entry["poll_id"]
            or poll.get("stage") != entry["stage"]
        ):
            raise EvidenceError("poll ledger/artifact binding mismatch")
        producer = poll.get("producer")
        if (
            not isinstance(producer, Mapping)
            or producer.get("synthetic_fixture") is not False
        ):
            raise EvidenceError("synthetic poll cannot certify")
        chain = poll.get("observation_chain")
        if (
            not isinstance(chain, Mapping)
            or isinstance(chain.get("records"), bool)
            or not isinstance(chain.get("records"), int)
            or not isinstance(chain.get("head_sha256"), str)
            or (chain["head_sha256"] and _HEX.fullmatch(chain["head_sha256"]) is None)
            or not isinstance(poll.get("identities"), Mapping)
            or not isinstance(poll.get("source"), Mapping)
            or not _poll_dimensions(poll)
        ):
            return _held_collector("collector_poll_provenance_invalid", len(polls))
        try:
            _utc(poll.get("captured_at"), "poll time")
        except EvidenceError:
            return _held_collector("collector_poll_timestamp_invalid", len(polls))
        polls.append(poll)
    stage_order = {stage: index for index, stage in enumerate(STAGES)}
    if any(entry.get("stage") not in stage_order for entry in ledger) or any(
        stage_order[str(left["stage"])] > stage_order[str(right["stage"])]
        for left, right in zip(ledger, ledger[1:], strict=False)
    ):
        raise EvidenceError("rollout stage sequence is invalid")
    stages: dict[str, dict[str, Any]] = {}
    reused: set[str] = set()
    reused_decisions: set[str] = set()
    stage_runs: set[str] = set()
    identities = {
        _digest(
            {
                key: value
                for key, value in poll["identities"].items()
                if key != "active_id"
            }
        )
        for poll in polls
    }
    sources = {_digest(poll["source"]) for poll in polls}
    if len(identities) > 1 or len(sources) > 1:
        raise EvidenceError("poll identity drift")
    for stage in STAGES:
        stage_polls = [poll for poll in polls if poll.get("stage") == stage]
        if not stage_polls:
            stages[stage] = {"certified": False, "reason": "no_real_polls"}
            continue
        times = [_utc(poll["captured_at"], "poll time") for poll in stage_polls]
        observations = [
            row for poll in stage_polls for row in poll.get("observations", [])
        ]
        run_ids = {poll.get("run_id") for poll in stage_polls}
        if (
            len(run_ids) != 1
            or not all(
                isinstance(run_id, str) and _HEX.fullmatch(run_id) for run_id in run_ids
            )
            or stage_runs.intersection(run_ids)
            or not all(isinstance(row, Mapping) for row in observations)
        ):
            raise EvidenceError("stage/run binding mismatch")
        stage_runs.update(run_id for run_id in run_ids if isinstance(run_id, str))
        ids = [row.get("observation_id") for row in observations]
        if (
            any(not isinstance(item, str) for item in ids)
            or len(ids) != len(set(ids))
            or reused.intersection(ids)
        ):
            raise EvidenceError("duplicate cross-stage observations")
        reused.update(ids)
        decision_ids = {
            _digest(
                (
                    row.get("decision_sha256"),
                    row.get("session_sha256"),
                    row.get("query_sha256"),
                    row.get("candidate_pool_sha256"),
                    row.get("feature_bytes_sha256"),
                )
            )
            for row in observations
        }
        if len(decision_ids) != len(observations) or reused_decisions.intersection(
            decision_ids
        ):
            raise EvidenceError("same-decision cross-stage reuse")
        reused_decisions.update(decision_ids)
        host = Counter(str(row["host"]) for row in observations)
        cohort = Counter(str(row["cohort"]) for row in observations)
        total = len(observations)
        evidence = [row.get("operational_evidence") for row in observations]
        evidence_keys = {
            "candidate_quality",
            "baseline_quality",
            "candidate_anchor_retained",
            "baseline_anchor_retained",
            "resource_ok",
            "integrity_ok",
            "negative_veto",
            "deadline_ms",
            "feature_snapshot_sha256",
            "candidate_feature_bytes_sha256",
            "baseline_feature_bytes_sha256",
            "feature_parity",
        }
        evidence_ok = all(
            isinstance(value, Mapping)
            and set(value) == evidence_keys
            and all(
                isinstance(value[key], bool)
                for key in evidence_keys
                - {
                    "deadline_ms",
                    "feature_snapshot_sha256",
                    "candidate_feature_bytes_sha256",
                    "baseline_feature_bytes_sha256",
                }
            )
            and isinstance(value.get("deadline_ms"), int)
            and not isinstance(value.get("deadline_ms"), bool)
            and 1 <= int(value["deadline_ms"]) <= 1_200
            and value.get("feature_snapshot_sha256")
            == row.get("feature_snapshot_sha256")
            and value.get("candidate_feature_bytes_sha256")
            == row.get("feature_bytes_sha256")
            and value.get("baseline_feature_bytes_sha256")
            == row.get("feature_bytes_sha256")
            for row, value in zip(observations, evidence, strict=True)
        )
        quality = sum(
            bool(value.get("candidate_quality"))
            for value in evidence
            if isinstance(value, Mapping)
        )
        baseline_quality = sum(
            bool(value.get("baseline_quality"))
            for value in evidence
            if isinstance(value, Mapping)
        )
        coverage = sum(bool(row.get("candidate_covered")) for row in observations)
        anchor = sum(
            bool(value.get("candidate_anchor_retained"))
            for value in evidence
            if isinstance(value, Mapping)
        )
        baseline_anchor = sum(
            bool(value.get("baseline_anchor_retained"))
            for value in evidence
            if isinstance(value, Mapping)
        )
        abstain_delta = (
            (
                sum(bool(row.get("candidate_abstained")) for row in observations)
                - sum(bool(row.get("baseline_abstained")) for row in observations)
            )
            / total
            if total
            else 1.0
        )
        timeouts = sum(bool(row.get("timed_out")) for row in observations)
        score = [row.get("candidate_score_ms") for row in observations]
        live = [row.get("live_latency_ms") for row in observations]
        metrics_ok = (
            bool(total)
            and all(
                isinstance(value, int) and not isinstance(value, bool)
                for value in (*score, *live)
            )
            and quality >= baseline_quality
            and coverage / total >= 0.95
            and anchor >= baseline_anchor
            and evidence_ok
            and all(
                bool(value["resource_ok"])
                and bool(value["integrity_ok"])
                and not bool(value["negative_veto"])
                for value in evidence
                if isinstance(value, Mapping)
            )
            and abstain_delta <= 0.02
            and _p95(score) <= 180
            and sorted(live)[(len(live) - 1) // 2] <= 400
            and _p95(live) < 900
            and timeouts / total <= 0.01
            and _wilson_lower(total - timeouts, total) >= 0.97
        )
        stages[stage] = {
            "certified": times[-1] - times[0] >= timedelta(days=MIN_DAYS)
            and total >= MIN_PAIRED
            and metrics_ok,
            "days": (times[-1] - times[0]).total_seconds() / 86_400,
            "paired": total,
            "host_counts": dict(host),
            "cohort_counts": dict(cohort),
            "metrics_ok": metrics_ok,
        }
    production_root = _FIXED_PRODUCTION_ROOT
    source_bound = False
    if (
        root is not None
        and not _has_symlink_component(root)
        and root.resolve() == production_root
    ):
        try:
            before = _protected_file_state(production_root)
            source_bound = all(
                _poll_dimensions(poll)
                and _verify_runtime_poll(production_root, poll)
                and _verify_live_attestation_poll(
                    evidence_root, poll, root=production_root
                )
                for poll in polls
            )
            source_bound = source_bound and before == _protected_file_state(
                production_root
            )
        except (OSError, EvidenceError):
            source_bound = False
    certified = source_bound and all(
        stage.get("certified") is True for stage in stages.values()
    )
    hold_reason = (
        "complete_authoritative_collector_bundle"
        if certified
        else "full_authoritative_production_snapshot_unavailable"
        if source_bound
        else "authoritative_runtime_observation_chain_unavailable"
    )
    for stage_data in stages.values():
        if not certified:
            stage_data["certified"] = False
        stage_data["reason"] = hold_reason
    return {
        "schema": EVIDENCE_SCHEMA,
        "certification": certified,
        "certification_reason": hold_reason,
        "stages": stages,
        "polls": len(polls),
        "protected_state_unchanged": source_bound,
        "identity": (
            {
                key: value
                for key, value in polls[0].get("identities", {}).items()
                if key != "active_id"
            }
            if polls
            else {}
        ),
        "source": polls[0].get("source", {}) if polls else {},
        "active_host_cohort_roster": {
            "hosts": sorted(
                {str(row["host"]) for poll in polls for row in poll["observations"]}
            ),
            "cohorts": sorted(
                {str(row["cohort"]) for poll in polls for row in poll["observations"]}
            ),
        }
        if certified
        else {},
    }


def _rollback_state(root: Path) -> dict[str, Any]:
    directory = store.distillation_dir(root)
    state_path = store.distillation_dir(root) / store.STATE_FILE
    if _has_symlink_component(state_path):
        raise EvidenceError("rollback state path is symlinked")
    state = _read_sealed_state(state_path, store.DISTILLATION_SCHEMA, "rollback state")
    lkg = _read_sealed_state(
        directory / store.POINTER_FILES["lkg"],
        store.DISTILLATION_SCHEMA,
        "rollback LKG pointer",
    )["policy_id"]
    active = _read_sealed_state(
        directory / store.POINTER_FILES["active"],
        store.DISTILLATION_SCHEMA,
        "rollback active pointer",
    )["policy_id"]
    try:
        candidate = _read_sealed_state(
            directory / store.POINTER_FILES["candidate"],
            store.DISTILLATION_SCHEMA,
            "rollback candidate pointer",
        )["policy_id"]
    except (EvidenceError, store.DistillationStoreError):
        candidate = None
    return {
        "state_sha256": _digest(state),
        "status": state.get("status"),
        "rollout_percent": state.get("rollout_percent"),
        "learning_halted": state.get("learning_halted"),
        "last_run_id": state.get("last_run_id"),
        "active_policy_id": active,
        "candidate_policy_id": candidate,
        "lkg_policy_id": lkg,
        "quarantine_id": state.get("quarantine_id"),
    }


def _sealed_file_ref(path: Path, label: str) -> dict[str, Any]:
    """Hash a sealed artifact without copying its contents into a receipt."""

    if _has_symlink_component(path):
        raise EvidenceError(f"{label} path is symlinked")
    try:
        body, metadata = _read_stable_file(path, label)
    except EvidenceError:
        raise
    return {
        "path": str(path.resolve()),
        "bytes_sha256": hashlib.sha256(body).hexdigest(),
        "lstat": metadata,
    }


def _pointer_state(root: Path) -> dict[str, Any]:
    directory = store.distillation_dir(root)
    result: dict[str, Any] = {}
    for kind, filename in store.POINTER_FILES.items():
        path = directory / filename
        if not path.exists():
            result[kind] = None
            continue
        if path.is_symlink():
            raise EvidenceError(f"{kind} pointer is symlinked")
        pointer = _read_sealed_state(
            path, store.DISTILLATION_SCHEMA, f"{kind} pointer"
        )
        result[kind] = {
            "policy_id": pointer.get("policy_id"),
            "sha256": _sealed_file_ref(path, f"{kind} pointer")["bytes_sha256"],
        }
    return result


def _rollback_ledgers(root: Path) -> dict[str, Any]:
    directory = store.distillation_dir(root)
    result: dict[str, Any] = {}
    ledger_name = "shadow-observation-receipts.jsonl"
    for path in (
        directory / ledger_name,
        store._chain_checkpoint_path(directory / ledger_name),
    ):
        result[str(path.name)] = (
            _sealed_file_ref(path, f"rollback ledger {path.name}")
            if path.exists()
            else None
        )
    return result


def _rollback_image(root: Path) -> dict[str, Any]:
    directory = store.distillation_dir(root)
    state_path = directory / store.STATE_FILE
    state_ref = _sealed_file_ref(state_path, "rollback state")
    return {
        "state": _rollback_state(root),
        "state_file": state_ref,
        "pointers": _pointer_state(root),
        "ledgers": _rollback_ledgers(root),
    }


def _validate_rollback_image(
    image: object, root: Path, label: str
) -> dict[str, Any]:
    """Validate the sealed pre/post image shape independently of receipt refs."""

    test_image = root.resolve() != _FIXED_PRODUCTION_ROOT
    if not isinstance(image, Mapping) or set(image) != {
        "state",
        "state_file",
        "pointers",
        "ledgers",
    }:
        raise EvidenceError(f"{label} schema is invalid")
    state = image.get("state")
    if (
        not isinstance(state, Mapping)
        or set(state)
        != {
            "state_sha256",
            "status",
            "rollout_percent",
            "learning_halted",
            "last_run_id",
            "active_policy_id",
            "candidate_policy_id",
            "lkg_policy_id",
            "quarantine_id",
        }
        or not isinstance(state.get("state_sha256"), str)
        or _HEX.fullmatch(state["state_sha256"]) is None
        or not isinstance(state.get("status"), str)
        or isinstance(state.get("rollout_percent"), bool)
        or not isinstance(state.get("rollout_percent"), int)
        or (
            not isinstance(state.get("learning_halted"), bool)
            and not (test_image and state.get("learning_halted") is None)
        )
    ):
        raise EvidenceError(f"{label} state is invalid")
    for key in ("active_policy_id", "lkg_policy_id"):
        if not isinstance(state.get(key), str) or _HEX.fullmatch(state[key]) is None:
            raise EvidenceError(f"{label} state policy identity is invalid")
    for key in ("candidate_policy_id", "quarantine_id", "last_run_id"):
        value = state.get(key)
        if value is not None and (
            not isinstance(value, str) or _HEX.fullmatch(value) is None
        ):
            raise EvidenceError(f"{label} state optional identity is invalid")

    def validate_ref(value: object, expected_path: Path, ref_label: str) -> None:
        if (
            not isinstance(value, Mapping)
            or set(value) != {"path", "bytes_sha256", "lstat"}
            or value.get("path") != str(expected_path.resolve())
            or not isinstance(value.get("bytes_sha256"), str)
            or _HEX.fullmatch(value["bytes_sha256"]) is None
            or not isinstance(value.get("lstat"), Mapping)
            or set(value["lstat"]) != _STAT_KEYS
            or any(
                isinstance(value["lstat"].get(key), bool)
                or not isinstance(value["lstat"].get(key), int)
                for key in _STAT_KEYS
            )
        ):
            raise EvidenceError(f"{ref_label} is invalid")

    directory = store.distillation_dir(root)
    validate_ref(image.get("state_file"), directory / store.STATE_FILE, f"{label} state file")
    pointers = image.get("pointers")
    if not isinstance(pointers, Mapping) or set(pointers) != set(store.POINTER_FILES):
        raise EvidenceError(f"{label} pointers are invalid")
    for kind, _filename in store.POINTER_FILES.items():
        pointer = pointers.get(kind)
        if pointer is None:
            continue
        if (
            not isinstance(pointer, Mapping)
            or set(pointer) != {"policy_id", "sha256"}
            or not isinstance(pointer.get("policy_id"), str)
            or _HEX.fullmatch(pointer["policy_id"]) is None
            or not isinstance(pointer.get("sha256"), str)
            or _HEX.fullmatch(pointer["sha256"]) is None
        ):
            raise EvidenceError(f"{label} {kind} pointer is invalid")
    ledgers = image.get("ledgers")
    expected_ledgers = {
        "shadow-observation-receipts.jsonl",
        "shadow-observation-receipts.jsonl.head.json",
    }
    if not isinstance(ledgers, Mapping) or set(ledgers) != expected_ledgers:
        raise EvidenceError(f"{label} ledgers are invalid")
    for name, value in ledgers.items():
        if value is not None:
            validate_ref(value, directory / name, f"{label} ledger {name}")
    return dict(image)


def _quarantine_ref(root: Path, quarantine_id: object) -> dict[str, Any] | None:
    if not isinstance(quarantine_id, str) or _HEX.fullmatch(quarantine_id) is None:
        return None
    path = store.distillation_dir(root) / "quarantine" / f"{quarantine_id}.json"
    if not path.exists():
        return None
    artifact, _, _ = _read_sealed_artifact(path, rollout.QUARANTINE_SCHEMA, "rollback quarantine")
    return {
        "artifact_id": quarantine_id,
        "receipt_sha256": artifact["seal_sha256"],
        **_sealed_file_ref(path, "rollback quarantine"),
    }


def _read_external_failure(
    path: Path,
    *,
    poll: Mapping[str, Any],
    stage: str,
    run_id: str,
    source: Mapping[str, Any],
    runtime: Mapping[str, Any],
    process: Mapping[str, Any],
) -> dict[str, Any]:
    """Accept only a sealed, independently persisted failure event."""

    try:
        event, body, _ = _read_sealed_artifact(
            path, "chronovisor.recall-r7-failure.v1", "external failure event"
        )
    except (OSError, store.DistillationStoreError) as exc:
        raise EvidenceError("external failure event is unavailable") from exc
    required = {
        "schema",
        "namespace",
        "artifact_id",
        "seal_sha256",
        "kind",
        "captured_at",
        "producer",
        "stage",
        "run_id",
        "poll_id",
        "poll_sha256",
        "poll_file_sha256",
        "source_commit",
        "archive_commit",
        "process_pid",
        "process_started_at",
        "source",
        "runtime",
        "archive",
        "process",
    }
    extended = required | {
        "live_attestation_artifact_id",
        "live_attestation_file_sha256",
        "live_attestation_seal_sha256",
    }
    if (
        set(event) != extended
        or event.get("schema") != "chronovisor.recall-r7-failure.v1"
        or event.get("namespace") != "recall-distillation"
        or event.get("kind") != "r7-external-failure"
    ):
        raise EvidenceError("external failure event schema is invalid")
    event_time = _utc(event.get("captured_at"), "external failure capture time")
    poll_time = _utc(poll.get("captured_at"), "failure poll capture time")
    if abs((event_time - poll_time).total_seconds()) > MAX_OBSERVATION_SKEW_SECONDS:
        raise EvidenceError("external failure clock skew is excessive")
    producer = event.get("producer")
    if (
        not isinstance(producer, Mapping)
        or set(producer) != {"name", "version", "synthetic_fixture"}
        or producer.get("name") != "chronovisor-failure-supervisor"
        or producer.get("version") != 1
        or producer.get("synthetic_fixture") is not False
    ):
        raise EvidenceError("external failure producer is invalid")
    if (
        event.get("artifact_id") != path.stem
        or event.get("stage") != stage
        or event.get("stage") != "100"
        or event.get("run_id") != run_id
        or event.get("poll_id") != poll.get("artifact_id")
        or event.get("poll_sha256") != poll.get("seal_sha256")
        or event.get("poll_file_sha256")
        != hashlib.sha256(
            canonical_json_line_bytes_strict(dict(poll))
        ).hexdigest()
        or event.get("source_commit") != source.get("source_commit")
        or event.get("archive_commit") != runtime.get("archive_commit")
        or event.get("process_pid") != process.get("pid")
        or event.get("process_started_at") != process.get("started_at")
        or event.get("source") != dict(source)
        or event.get("runtime") != dict(runtime)
        or event.get("archive") != _archive_projection(runtime)
        or event.get("process") != dict(process)
        or not isinstance(event.get("process_pid"), int)
        or isinstance(event.get("process_pid"), bool)
        or event["process_pid"] <= 0
        or not isinstance(event.get("process_started_at"), str)
    ):
        raise EvidenceError("external failure event binding mismatch")
    if any(
        not isinstance(event.get(key), str) or _HEX.fullmatch(event[key]) is None
        for key in (
            "live_attestation_artifact_id",
            "live_attestation_file_sha256",
            "live_attestation_seal_sha256",
        )
    ):
        raise EvidenceError("external failure event attestation binding is invalid")
    poll_attestation = poll.get("live_attestation")
    if (
        not isinstance(poll_attestation, Mapping)
        or event["live_attestation_artifact_id"] != poll_attestation.get("artifact_id")
        or event["live_attestation_file_sha256"] != poll_attestation.get("file_sha256")
        or event["live_attestation_seal_sha256"] != poll_attestation.get("seal_sha256")
    ):
        raise EvidenceError("external failure event attestation mismatch")
    return {
        "artifact_id": str(event["artifact_id"]),
        "receipt_sha256": str(event["seal_sha256"]),
        "path": str(path.resolve()),
        "bytes_sha256": hashlib.sha256(body).hexdigest(),
    }


def _record_forced_rollback_locked(
    *,
    root: Path,
    evidence_root: Path,
    source_root: Path,
    direct_url_path: Path,
    executable: Path,
    pid: int,
    stage: str,
    run_id: str,
    poll_id: str,
    failure_token: str | None = None,
    failure_event_path: Path | None = None,
    failure_event: Mapping[str, str] | None = None,
    service_role: str = "dashboard",
    allow_test_root: bool = False,
) -> dict[str, Any]:
    """Record an externally sealed failure and a state-first LKG rollback."""
    _id(run_id, "rollback run id")
    _id(poll_id, "rollback poll id")
    if stage != "100":
        raise EvidenceError("forced rollback requires stage 100")
    production_root = root.resolve() == _FIXED_PRODUCTION_ROOT
    if allow_test_root and production_root:
        raise EvidenceError("test rollback cannot use production authority")
    # The sole compatibility seam is an explicitly named owned temp-root
    # drill; it writes TEST_FAILURE_SCHEMA and cannot enter production paths.
    allow_test_root = allow_test_root or (
        not production_root and failure_token == "deterministic-test-failure"
    )
    if (
        not production_root
        and not allow_test_root
        and failure_token != "deterministic-test-failure"
    ):
        raise EvidenceError("rollback root requires explicit test-only mode")
    if production_root and service_role not in _SERVICE_LABELS:
        raise EvidenceError("rollback service role is not approved")
    if _has_symlink_component(root) or _has_symlink_component(evidence_root):
        raise EvidenceError("forced rollback root/evidence is unsafe")
    if production_root:
        if evidence_root.resolve() != _FIXED_EVIDENCE_ROOT.resolve():
            raise EvidenceError("rollback evidence root is not managed")
        root = _FIXED_PRODUCTION_ROOT
        evidence_root = _FIXED_EVIDENCE_ROOT
        source_root = _fixed_source_root()
        direct_url_path = _fixed_direct_url_path()
        process = None
        executable = Path("/")
    elif not allow_test_root:
        raise EvidenceError("rollback test root is not authorized")
    rollback_dir = evidence_root / "rollbacks"
    if rollback_dir.exists():
        if rollback_dir.is_symlink() or not rollback_dir.is_dir():
            raise EvidenceError("rollback receipt directory is unsafe")
        for existing_path in sorted(rollback_dir.glob("*.json"), key=str):
            try:
                existing, _, _ = _read_sealed_artifact(
                    existing_path,
                    "chronovisor.recall-r7-rollback.v1",
                    "rollback receipt",
                )
            except EvidenceError:
                continue
            existing_pre = existing.get("pre")
            if (
                isinstance(existing_pre, Mapping)
                and existing_pre.get("run_id") == run_id
                and existing_pre.get("poll_id") == poll_id
            ):
                result = validate_rollback(
                    root,
                    existing_path,
                    allow_test_root=allow_test_root,
                )
                result["changed"] = False
                return result
    if failure_event is not None:
        if failure_event_path is not None or not isinstance(failure_event.get("path"), str):
            raise EvidenceError("external failure event path is invalid")
        failure_event_path = Path(failure_event["path"])
    poll_path = evidence_root / "polls" / f"{poll_id}.json"
    poll, poll_raw, _ = _read_sealed_artifact(poll_path, POLL_SCHEMA, "rollback poll")
    if poll.get("stage") != stage or poll.get("run_id") != run_id:
        raise EvidenceError("rollback poll/stage binding mismatch")
    runtime = _direct_url(direct_url_path, formal=production_root)
    process = (
        _service_process_identity(service_role)
        if production_root
        else _process_identity(executable, pid, service_role=None)
    )
    if production_root:
        executable = Path(str(process["executable_path"]))
    source = _source_identity(source_root)
    if production_root:
        if (
            not _poll_dimensions(poll)
            or not _verify_runtime_poll(root, poll)
            or not _verify_live_attestation_poll(evidence_root, poll, root=root)
            or poll.get("runtime") != runtime
            or poll.get("source") != source
            or not isinstance(poll.get("process"), Mapping)
            or not _same_process_identity(poll["process"], process)
        ):
            raise EvidenceError("rollback poll authority is not current")
        ledger_rows = _ledger_rows(evidence_root / "poll-ledger.jsonl")
        ledger_row = next(
            (row for row in ledger_rows if row.get("poll_id") == poll_id), None
        )
        if (
            not isinstance(ledger_row, Mapping)
            or ledger_row.get("poll_sha256") != hashlib.sha256(poll_raw).hexdigest()
        ):
            raise EvidenceError("rollback poll ledger bytes mismatch")
    live_attestation, live_attestation_path = _rollback_attestation(
        root=root,
        evidence_root=evidence_root,
        poll=poll,
        stage=stage,
        run_id=run_id,
        source=source,
        runtime=runtime,
        process=process,
        failure_token=failure_token,
    )
    if failure_event_path is not None:
        if production_root:
            if (
                not failure_event_path.is_absolute()
                or _has_symlink_component(failure_event_path)
                or _HEX.fullmatch(failure_event_path.stem) is None
                or failure_event_path.parent.resolve()
                != (evidence_root / "failures").resolve()
            ):
                raise EvidenceError("external failure event is outside managed evidence")
            failure_event_path = evidence_root / "failures" / f"{failure_event_path.stem}.json"
        failure_event = _read_external_failure(
            failure_event_path,
            poll=poll,
            stage=stage,
            run_id=run_id,
            source=source,
            runtime=runtime,
            process=process,
        )
        failure_kind = "r7_external_failure"
        failure_payload: dict[str, Any] = {
            "kind": failure_kind,
            **failure_event,
        }
    else:
        # Compatibility for the historical owned-temp-root drill only.  A
        # production call must provide an independently sealed event; arbitrary
        # caller tokens are never accepted as failure evidence.
        if production_root or not allow_test_root or failure_token != "deterministic-test-failure":
            raise EvidenceError("external sealed failure event is required")
        token_sha = hashlib.sha256(failure_token.encode()).hexdigest()
        failure_id, failure_path, failure_artifact = store.write_immutable(
            evidence_root / "failures",
            {
                "kind": "r7-external-failure",
                "captured_at": datetime.now(UTC).isoformat(),
                "stage": stage,
                "run_id": run_id,
                "poll_id": poll_id,
                "poll_sha256": poll["seal_sha256"],
                "poll_file_sha256": hashlib.sha256(poll_raw).hexdigest(),
                "source_commit": source["source_commit"],
                "archive_commit": runtime["archive_commit"],
                "process_pid": process["pid"],
                "process_started_at": process["started_at"],
                "token_sha256": token_sha,
                "test_only": True,
            },
            schema=TEST_FAILURE_SCHEMA,
        )
        # ``_read_external_failure`` intentionally requires the exact producer
        # shape; the one-time compatibility artifact is sealed and then read
        # through the same path before proceeding.
        failure_event_path = failure_path
        failure_event = {
            "artifact_id": failure_id,
            "receipt_sha256": failure_artifact["seal_sha256"],
            "path": str(failure_path.resolve()),
            "bytes_sha256": hashlib.sha256(
                _read_stable_file(failure_path, "test failure event")[0]
            ).hexdigest(),
        }
        failure_kind = "r7_forced_failure"
        failure_payload = {
            "kind": failure_kind,
            "token_sha256": token_sha,
            "artifact_id": failure_id,
            "receipt_sha256": failure_artifact["seal_sha256"],
        }
    pre = {
        "stage": stage,
        "run_id": run_id,
        "poll_id": poll_id,
        "poll_sha256": poll["seal_sha256"],
        "stage_identity": _stage_state(root, stage),
        "runtime": runtime,
        "process": process,
        "source": source,
        "live_attestation": live_attestation,
        "verification": {
            "source_root": str(source_root.resolve()),
            "direct_url_path": str(direct_url_path.resolve()),
            "root": str(root.resolve()),
            "evidence_root": str(evidence_root.resolve()),
            "executable_path": str(executable.resolve()),
        },
        "failure_event": failure_event,
        "image": _rollback_image(root),
    }
    _validate_rollback_image(pre["image"], root, "rollback preimage")
    if poll.get("identities") != pre["stage_identity"]:
        raise EvidenceError("rollback poll identity drift")
    intent_id, intent_path, intent_artifact = store.write_immutable(
        evidence_root / "rollback-intents",
        {
            "kind": "r7-rollback-intent",
            "captured_at": datetime.now(UTC).isoformat(),
            "stage": stage,
            "run_id": run_id,
            "poll_id": poll_id,
            "poll_sha256": poll["seal_sha256"],
            "failure_event": failure_event,
            "pre": pre,
        },
        schema="chronovisor.recall-r7-rollback-intent.v1",
    )
    intent_bytes = canonical_json_line_bytes_strict(intent_artifact)
    if (
        _rollback_image(root) != pre["image"]
        or _stage_state(root, stage) != pre["stage_identity"]
    ):
        _remove_own_artifact(intent_path, intent_bytes)
        raise EvidenceError("rollback preimage changed before mutation")
    poll_before, _, _ = _read_sealed_artifact(
        poll_path, POLL_SCHEMA, "rollback poll"
    )
    if poll_before != poll:
        _remove_own_artifact(intent_path, intent_bytes)
        raise EvidenceError("rollback poll changed before mutation")
    process_before_mutation = (
        _service_process_identity(service_role, process["started_at"])
        if production_root
        else _process_identity(executable, pid, process["started_at"], None)
    )
    if not _same_process_identity(process_before_mutation, process):
        _remove_own_artifact(intent_path, intent_bytes)
        raise EvidenceError("rollback process changed before mutation")
    if (
        _source_identity(source_root) != source
        or _direct_url(direct_url_path, formal=production_root) != runtime
    ):
        _remove_own_artifact(intent_path, intent_bytes)
        raise EvidenceError("rollback source/runtime changed before mutation")
    try:
        if pre["image"]["state"]["status"] == "rolled_back" and pre["image"]["state"]["last_run_id"] == run_id:
            result = {
                "status": "rolled_back",
                "rollout_percent": 0,
                "learning_halted": True,
                "last_run_id": run_id,
                "changed": False,
            }
        else:
            # The public rollout helper takes the same lock; invoke its
            # lock-held primitive to keep this transaction atomic.
            rollout_state = rollout._state(root)
            result = rollout._rollback_locked(
                root, rollout_state, run_id, "r7_forced_failure"
            )
    except Exception:
        _remove_own_artifact(intent_path, intent_bytes)
        raise
    post_process = (
        _service_process_identity(service_role, process["started_at"])
        if production_root
        else _process_identity(executable, pid, process["started_at"], None)
    )
    if not _same_process_identity(post_process, process):
        raise EvidenceError("rollback process PID was reused")
    post = {
        "rollback_result": result,
        "poll": {"artifact_id": poll_id, "seal_sha256": poll["seal_sha256"]},
        "state": _rollback_state(root),
        "runtime": _direct_url(direct_url_path, formal=production_root),
        "process": post_process,
        "source": _source_identity(source_root),
        "image": _rollback_image(root),
        "live_attestation": (
            _test_attestation_ref(live_attestation_path)
            if live_attestation_path.parent.name == "r7-live-attestations-test"
            else _attestation_ref(live_attestation_path)
        ),
        "quarantine": _quarantine_ref(root, _rollback_state(root)["quarantine_id"]),
        "intent": {
            "artifact_id": intent_id,
            "receipt_sha256": intent_artifact["seal_sha256"],
            "path": str(intent_path.resolve()),
            "bytes_sha256": hashlib.sha256(intent_bytes).hexdigest(),
        },
    }
    _validate_rollback_image(post["image"], root, "rollback postimage")
    payload = {
        "kind": "r7-authoritative-forced-rollback",
        "captured_at": datetime.now(UTC).isoformat(),
        "pre": pre,
        "injected_failure": failure_payload,
        "post": post,
    }
    artifact_id, receipt_path, artifact = store.write_immutable(
        evidence_root / "rollbacks", payload, schema="chronovisor.recall-r7-rollback.v1"
    )
    readback, receipt_bytes, _ = _read_sealed_artifact(
        receipt_path, "chronovisor.recall-r7-rollback.v1", "rollback receipt"
    )
    if readback != artifact:
        raise EvidenceError("rollback receipt immutable readback mismatch")
    return {
        "artifact_id": artifact_id,
        "receipt_sha256": artifact["seal_sha256"],
        "run_id": run_id,
        "stage": stage,
        "source_commit": source["source_commit"],
        "archive_commit": runtime["archive_commit"],
        "paths": {
            "root": str(root.resolve()),
            "evidence_root": str(evidence_root.resolve()),
            "poll": str(poll_path.resolve()),
            "failure_event": str(failure_event_path.resolve()),
            "receipt": str(receipt_path.resolve()),
        },
    }


def record_forced_rollback(
    *,
    root: Path,
    evidence_root: Path,
    source_root: Path,
    direct_url_path: Path,
    executable: Path,
    pid: int,
    stage: str,
    run_id: str,
    poll_id: str,
    failure_token: str | None = None,
    failure_event_path: Path | None = None,
    failure_event: Mapping[str, str] | None = None,
    service_role: str = "dashboard",
    allow_test_root: bool = False,
) -> dict[str, Any]:
    """Serialize forced rollback with the rollout transaction lock held."""

    production_root = root.resolve() == _FIXED_PRODUCTION_ROOT
    if allow_test_root and production_root:
        raise EvidenceError("test rollback cannot use production authority")
    if (
        not production_root
        and not allow_test_root
        and failure_token != "deterministic-test-failure"
    ):
        raise EvidenceError("rollback root requires explicit test-only mode")
    if production_root:
        if _has_symlink_component(root) or _has_symlink_component(evidence_root):
            raise EvidenceError("rollback root/evidence is symlinked")
        root = _FIXED_PRODUCTION_ROOT
        evidence_root = _FIXED_EVIDENCE_ROOT
    lock_path = store.distillation_dir(root) / "rollout.lock"
    if _has_symlink_component(lock_path):
        raise EvidenceError("rollback transaction lock is symlinked")
    lock_flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        lock_flags |= os.O_NOFOLLOW
    try:
        lock_descriptor = os.open(lock_path, lock_flags, 0o600)
        try:
            fcntl.flock(lock_descriptor, fcntl.LOCK_EX)
            return _record_forced_rollback_locked(
                root=root,
                evidence_root=evidence_root,
                source_root=source_root,
                direct_url_path=direct_url_path,
                executable=executable,
                pid=pid,
                stage=stage,
                run_id=run_id,
                poll_id=poll_id,
                failure_token=failure_token,
                failure_event_path=failure_event_path,
                failure_event=failure_event,
                service_role=service_role,
                allow_test_root=allow_test_root,
            )
        finally:
            try:
                fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
            finally:
                os.close(lock_descriptor)
    except OSError as exc:
        raise EvidenceError("rollback transaction lock unavailable") from exc


def validate_rollback(
    root: Path, receipt_path: Path, *, allow_test_root: bool = False
) -> dict[str, Any]:
    """Re-read the rollback's poll, pointers, archive, and executable identity."""
    if allow_test_root and root.resolve() == _FIXED_PRODUCTION_ROOT:
        raise EvidenceError("test rollback cannot use production authority")
    if _has_symlink_component(receipt_path):
        raise EvidenceError("rollback receipt path is symlinked")
    if _has_symlink_component(root) or (
        not allow_test_root and root.resolve() != _FIXED_PRODUCTION_ROOT
    ):
        raise EvidenceError("rollback root is not the production runtime")
    if not allow_test_root:
        root = _FIXED_PRODUCTION_ROOT
        expected_parent = _FIXED_EVIDENCE_ROOT / "rollbacks"
        if (
            not receipt_path.is_absolute()
            or _HEX.fullmatch(receipt_path.stem) is None
            or receipt_path.parent != expected_parent
        ):
            raise EvidenceError("rollback receipt is outside managed evidence")
        receipt_path = expected_parent / f"{receipt_path.stem}.json"
    try:
        receipt, _, _ = _read_sealed_artifact(
            receipt_path,
            "chronovisor.recall-r7-rollback.v1",
            "rollback receipt",
        )
    except (EvidenceError, store.DistillationStoreError) as exc:
        raise EvidenceError("rollback authoritative R7 binding is unavailable") from exc
    _utc(receipt.get("captured_at"), "rollback capture time")
    required = {
        "schema",
        "namespace",
        "artifact_id",
        "seal_sha256",
        "kind",
        "captured_at",
        "pre",
        "injected_failure",
        "post",
    }
    if set(receipt) != required or receipt.get("kind") != "r7-authoritative-forced-rollback":
        raise EvidenceError("rollback receipt schema is invalid")
    pre = receipt.get("pre")
    post = receipt.get("post")
    failure = receipt.get("injected_failure")
    if (
        not isinstance(pre, Mapping)
        or not isinstance(post, Mapping)
        or not isinstance(failure, Mapping)
        or set(pre)
        != {
            "stage",
            "run_id",
            "poll_id",
            "poll_sha256",
            "stage_identity",
            "runtime",
            "process",
            "source",
            "live_attestation",
            "verification",
            "failure_event",
            "image",
        }
        or set(post)
        != {
            "rollback_result",
            "poll",
            "state",
            "runtime",
            "process",
            "source",
            "image",
            "live_attestation",
            "quarantine",
            "intent",
        }
    ):
        raise EvidenceError("rollback receipt binding is invalid")
    _validate_rollback_image(pre.get("image"), root, "rollback preimage")
    _validate_rollback_image(post.get("image"), root, "rollback postimage")
    poll_id = pre.get("poll_id")
    if not isinstance(poll_id, str) or _HEX.fullmatch(poll_id) is None:
        raise EvidenceError("rollback poll identity is invalid")
    verification = pre.get("verification")
    if (
        not isinstance(verification, Mapping)
        or set(verification)
        != {
            "source_root",
            "direct_url_path",
            "root",
            "evidence_root",
            "executable_path",
        }
        or not all(isinstance(verification.get(key), str) for key in verification)
        or Path(str(verification["root"])).resolve() != root.resolve()
    ):
        raise EvidenceError("rollback verification paths are invalid")
    if not allow_test_root and (
        Path(str(verification["evidence_root"])) != _FIXED_EVIDENCE_ROOT
        or Path(str(verification["source_root"])) != _fixed_source_root()
        or Path(str(verification["direct_url_path"])) != _fixed_direct_url_path()
        or Path(str(verification["root"])) != _FIXED_PRODUCTION_ROOT
    ):
        raise EvidenceError("rollback verification authority is not fixed")
    evidence_path = (
        _FIXED_EVIDENCE_ROOT
        if not allow_test_root
        else Path(str(verification["evidence_root"])).resolve()
    )
    if _has_symlink_component(evidence_path):
        raise EvidenceError("rollback evidence root is symlinked")
    poll_dir = evidence_path / "polls"
    if poll_dir.is_symlink() or not poll_dir.is_dir():
        raise EvidenceError("rollback poll directory is unsafe")
    poll_path = poll_dir / f"{poll_id}.json"
    poll, poll_raw, _ = _read_sealed_artifact(poll_path, POLL_SCHEMA, "rollback poll")
    if (
        poll.get("artifact_id") != poll_id
        or poll.get("seal_sha256") != pre.get("poll_sha256")
        or poll.get("stage") != pre.get("stage")
        or poll.get("run_id") != pre.get("run_id")
        or poll.get("identities") != pre.get("stage_identity")
        or pre.get("stage") != "100"
    ):
        raise EvidenceError("rollback pre-state/poll binding mismatch")
    if not allow_test_root:
        ledger_rows = _ledger_rows(evidence_path / "poll-ledger.jsonl")
        ledger_row = next(
            (row for row in ledger_rows if row.get("poll_id") == poll_id), None
        )
        if (
            not isinstance(ledger_row, Mapping)
            or ledger_row.get("poll_sha256") != hashlib.sha256(poll_raw).hexdigest()
        ):
            raise EvidenceError("rollback poll ledger bytes mismatch")
    live_ref = pre.get("live_attestation")
    if (
        not isinstance(live_ref, Mapping)
        or set(live_ref) != {"artifact_id", "file_sha256", "seal_sha256"}
        or not all(isinstance(live_ref.get(key), str) for key in live_ref)
        or _HEX.fullmatch(str(live_ref.get("artifact_id"))) is None
        or _HEX.fullmatch(str(live_ref.get("file_sha256"))) is None
        or _HEX.fullmatch(str(live_ref.get("seal_sha256"))) is None
    ):
        raise EvidenceError("rollback live attestation reference is invalid")
    production_live_path = (
        Path(str(verification["evidence_root"]))
        / "r7-live-attestations"
        / f"{live_ref['artifact_id']}.json"
    )
    test_live_path = (
        Path(str(verification["evidence_root"]))
        / "r7-live-attestations-test"
        / f"{live_ref['artifact_id']}.json"
    )
    if not allow_test_root or (
        production_live_path.is_file() and not production_live_path.is_symlink()
    ):
        live_path = production_live_path
        live_schema = LIVE_ATTESTATION_SCHEMA
    elif test_live_path.is_file() and not test_live_path.is_symlink():
        live_path = test_live_path
        live_schema = TEST_LIVE_ATTESTATION_SCHEMA
    else:
        raise EvidenceError("rollback live attestation readback mismatch")
    live_artifact, live_body, _ = _read_sealed_artifact(
        live_path, live_schema, "rollback live attestation"
    )
    if _artifact_ref_values(live_artifact, live_body) != dict(live_ref):
        raise EvidenceError("rollback live attestation readback mismatch")
    if live_schema == LIVE_ATTESTATION_SCHEMA:
        _validate_attestation_payload(
            live_artifact,
            expected_stage=str(pre["stage"]),
            expected_run_id=str(pre["run_id"]),
        )
    else:
        test_payload = {
            key: value
            for key, value in live_artifact.items()
            if key not in {"test_only", "schema", "namespace", "artifact_id", "seal_sha256"}
        }
        _validate_attestation_payload(
            {
                "schema": LIVE_ATTESTATION_SCHEMA,
                "namespace": "recall-distillation",
                "artifact_id": live_artifact["artifact_id"],
                "seal_sha256": live_artifact["seal_sha256"],
                **test_payload,
            },
            expected_stage=str(pre["stage"]),
            expected_run_id=str(pre["run_id"]),
            allow_test_service=True,
            check_identity=False,
        )
    if (
        live_artifact.get("source", {}).get("source_commit") != pre.get("source", {}).get("source_commit")
        or live_artifact.get("runtime") != pre.get("runtime")
        or not isinstance(live_artifact.get("process"), Mapping)
        or not isinstance(pre.get("process"), Mapping)
        or not _same_process_identity(live_artifact["process"], pre["process"])
    ):
        raise EvidenceError("rollback live attestation binding mismatch")
    failure_event = pre.get("failure_event")
    if (
        not isinstance(failure_event, Mapping)
        or set(failure_event)
        != {"artifact_id", "receipt_sha256", "path", "bytes_sha256"}
        or not all(isinstance(failure_event.get(key), str) for key in failure_event)
        or _HEX.fullmatch(str(failure_event.get("artifact_id"))) is None
        or _HEX.fullmatch(str(failure_event.get("receipt_sha256"))) is None
        or _HEX.fullmatch(str(failure_event.get("bytes_sha256"))) is None
    ):
        raise EvidenceError("rollback failure event reference is invalid")
    failure_path = Path(str(failure_event["path"]))
    if (
        not failure_path.is_absolute()
        or _has_symlink_component(failure_path)
        or failure_path.parent.resolve() != (evidence_path / "failures").resolve()
    ):
        raise EvidenceError("rollback failure event readback mismatch")
    failure_path = evidence_path / "failures" / f"{failure_path.stem}.json"
    failure_raw, _ = _read_stable_file(failure_path, "rollback failure event")
    if hashlib.sha256(failure_raw).hexdigest() != failure_event["bytes_sha256"]:
        raise EvidenceError("rollback failure event readback mismatch")
    if failure.get("kind") == "r7_external_failure":
        _read_external_failure(
            failure_path,
            poll=poll,
            stage=str(pre["stage"]),
            run_id=str(pre["run_id"]),
            source=pre["source"],
            runtime=pre["runtime"],
            process=pre["process"],
        )
        if failure.get("artifact_id") != failure_event["artifact_id"]:
            raise EvidenceError("rollback failure event artifact mismatch")
    elif failure.get("kind") == "r7_forced_failure":
        if not allow_test_root:
            raise EvidenceError("test failure event cannot certify")
        failure_artifact, _, _ = _read_sealed_artifact(
            failure_path, TEST_FAILURE_SCHEMA, "test failure event"
        )
        if (
            _HEX.fullmatch(str(failure.get("token_sha256"))) is None
            or failure.get("artifact_id") != failure_event["artifact_id"]
            or failure.get("receipt_sha256") != failure_event["receipt_sha256"]
            or failure_artifact.get("artifact_id") != failure_event["artifact_id"]
            or failure_artifact.get("seal_sha256") != failure_event["receipt_sha256"]
        ):
            raise EvidenceError("rollback failure event token mismatch")
    else:
        raise EvidenceError("rollback failure event kind is invalid")
    expected = _rollback_state(root)
    if post.get("poll") != {
        "artifact_id": poll_id,
        "seal_sha256": pre.get("poll_sha256"),
    }:
        raise EvidenceError("rollback post-poll binding mismatch")
    if (
        post.get("state") != expected
        or expected["status"] != "rolled_back"
        or expected["rollout_percent"] != 0
        or expected["learning_halted"] is not True
        or expected["active_policy_id"] != expected["lkg_policy_id"]
        or expected["candidate_policy_id"] is not None
    ):
        raise EvidenceError("rollback post-state/pointer mismatch")
    result = post.get("rollback_result")
    if (
        not isinstance(result, Mapping)
        or result.get("status") != "rolled_back"
        or result.get("rollout_percent") != 0
        or result.get("learning_halted") is not True
        or result.get("last_run_id") != pre.get("run_id")
    ):
        raise EvidenceError("rollback mutation result mismatch")
    if (
        post.get("runtime") != pre.get("runtime")
        or post.get("source") != pre.get("source")
        or not isinstance(post.get("process"), Mapping)
        or not isinstance(pre.get("process"), Mapping)
        or not _same_process_identity(post["process"], pre["process"])
    ):
        raise EvidenceError("rollback runtime identity changed")
    if post.get("live_attestation") != live_ref:
        raise EvidenceError("rollback live attestation changed")
    process = pre.get("process")
    process_service_role = (
        process.get("service", {}).get("role")
        if isinstance(process, Mapping)
        and isinstance(process.get("service"), Mapping)
        else None
    )
    live_process = (
        _service_process_identity(
            process_service_role, process.get("started_at") if isinstance(process, Mapping) else None
        )
        if not allow_test_root
        and isinstance(process_service_role, str)
        and isinstance(process, Mapping)
        else (
            _process_identity(
                Path(process["executable_path"]),
                process["pid"],
                process.get("started_at"),
                process_service_role,
            )
            if isinstance(process, Mapping)
            and isinstance(process.get("executable_path"), str)
            and isinstance(process.get("pid"), int)
            and not isinstance(process.get("pid"), bool)
            and process.get("pid", 0) > 0
            else None
        )
    )
    if (
        not isinstance(process, Mapping)
        or not isinstance(process.get("executable_path"), str)
        or isinstance(process.get("pid"), bool)
        or not isinstance(process.get("pid"), int)
        or _direct_url(
            _fixed_direct_url_path()
            if not allow_test_root
            else Path(verification["direct_url_path"]),
            formal=not allow_test_root,
        )
        != post["runtime"]
        or not isinstance(live_process, Mapping)
        or not _same_process_identity(live_process, post["process"])
        or live_process.get("started_at") != process.get("started_at")
        or _source_identity(
            _fixed_source_root()
            if not allow_test_root
            else Path(verification["source_root"])
        )
        != post["source"]
        or _rollback_image(root) != post["image"]
        or post.get("quarantine")
        != _quarantine_ref(root, expected.get("quarantine_id"))
    ):
        raise EvidenceError("rollback live identity mismatch")
    intent = post.get("intent")
    intent_path = (
        Path(str(intent["path"])) if isinstance(intent, Mapping) and isinstance(intent.get("path"), str) else None
    )
    if (
        not isinstance(intent, Mapping)
        or set(intent) != {"artifact_id", "receipt_sha256", "path", "bytes_sha256"}
        or not all(isinstance(intent.get(key), str) for key in intent)
        or _HEX.fullmatch(str(intent.get("artifact_id"))) is None
        or _HEX.fullmatch(str(intent.get("receipt_sha256"))) is None
        or _HEX.fullmatch(str(intent.get("bytes_sha256"))) is None
        or intent_path is None
        or _has_symlink_component(intent_path)
        or intent_path.parent.resolve() != (evidence_path / "rollback-intents").resolve()
    ):
        raise EvidenceError("rollback intent readback mismatch")
    intent_path = evidence_path / "rollback-intents" / f"{intent_path.stem}.json"
    intent_artifact, intent_raw, _ = _read_sealed_artifact(
        intent_path,
        "chronovisor.recall-r7-rollback-intent.v1",
        "rollback intent",
    )
    intent_time = _utc(intent_artifact.get("captured_at"), "rollback intent capture time")
    receipt_time = _utc(receipt.get("captured_at"), "rollback capture time")
    if abs((receipt_time - intent_time).total_seconds()) > MAX_OBSERVATION_SKEW_SECONDS:
        raise EvidenceError("rollback intent clock skew is excessive")
    if (
        set(intent_artifact)
        != {
            "schema",
            "namespace",
            "artifact_id",
            "seal_sha256",
            "kind",
            "captured_at",
            "stage",
            "run_id",
            "poll_id",
            "poll_sha256",
            "failure_event",
            "pre",
        }
        or intent_artifact.get("kind") != "r7-rollback-intent"
        or intent_artifact.get("stage") != pre.get("stage")
        or intent_artifact.get("run_id") != pre.get("run_id")
        or intent_artifact.get("poll_id") != pre.get("poll_id")
        or intent_artifact.get("poll_sha256") != pre.get("poll_sha256")
        or intent_artifact.get("failure_event") != pre.get("failure_event")
    ):
        raise EvidenceError("rollback intent binding is invalid")
    if (
        intent_artifact.get("artifact_id") != intent["artifact_id"]
        or intent_artifact.get("seal_sha256") != intent["receipt_sha256"]
        or hashlib.sha256(intent_raw).hexdigest() != intent["bytes_sha256"]
        or intent_artifact.get("pre") != pre
    ):
        raise EvidenceError("rollback intent readback mismatch")
    paths = {
        "root": str(root.resolve()),
        "evidence_root": str(Path(verification["evidence_root"]).resolve()),
        "poll": str(poll_path.resolve()),
        "failure_event": str(failure_path.resolve()),
        "receipt": str(receipt_path.resolve()),
    }
    return {
        "artifact_id": str(receipt["artifact_id"]),
        "receipt_sha256": str(receipt["seal_sha256"]),
        "run_id": str(pre["run_id"]),
        "stage": str(pre["stage"]),
        "source_commit": str(pre["source"]["source_commit"]),
        "archive_commit": str(pre["runtime"]["archive_commit"]),
        "paths": paths,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    record = commands.add_parser("record-poll", help="read and seal one live poll")
    for name in (
        "root",
        "source-root",
        "evidence-root",
        "dom-capture",
        "direct-url",
        "executable",
        "live-attestation",
    ):
        record.add_argument(f"--{name}", type=Path, required=True)
    record.add_argument("--stage", choices=STAGES, required=True)
    record.add_argument("--run-id", required=True)
    record.add_argument("--dashboard-url", required=True)
    record.add_argument("--pid", type=int, required=True)
    record.add_argument("--service-role", choices=tuple(_SERVICE_LABELS), default="dashboard")
    record.add_argument("--expected-started-at")
    record.add_argument("--live-attestation-artifact-id")
    validate = commands.add_parser("validate", help="recompute sealed poll evidence")
    validate.add_argument("--evidence-root", type=Path, required=True)
    validate.add_argument("--root", type=Path)
    validate.add_argument("--forced-failure-receipt", type=Path)
    rollback = commands.add_parser(
        "record-forced-rollback", help="inject and seal a state-first LKG rollback"
    )
    for name in (
        "root",
        "source-root",
        "evidence-root",
        "direct-url",
        "executable",
    ):
        rollback.add_argument(f"--{name}", type=Path, required=True)
    rollback.add_argument("--stage", choices=STAGES, required=True)
    rollback.add_argument("--run-id", required=True)
    rollback.add_argument("--poll-id", required=True)
    rollback.add_argument("--failure-token")
    rollback.add_argument("--failure-event", type=Path)
    rollback.add_argument("--pid", type=int, required=True)
    rollback.add_argument("--service-role", choices=tuple(_SERVICE_LABELS), default="dashboard")
    args = parser.parse_args(argv)
    try:
        if args.command == "record-poll":
            result = collect_poll(
                root=args.root,
                source_root=args.source_root,
                evidence_root=args.evidence_root,
                stage=args.stage,
                run_id=args.run_id,
                dashboard_url=args.dashboard_url,
                dom_capture_path=args.dom_capture,
                direct_url_path=args.direct_url,
                executable=args.executable,
                pid=args.pid,
                expected_started_at=args.expected_started_at,
                service_role=args.service_role,
                live_attestation_path=args.live_attestation,
                live_attestation_artifact_id=args.live_attestation_artifact_id,
            )
        elif args.command == "record-forced-rollback":
            result = record_forced_rollback(
                root=args.root,
                source_root=args.source_root,
                evidence_root=args.evidence_root,
                direct_url_path=args.direct_url,
                executable=args.executable,
                pid=args.pid,
                stage=args.stage,
                run_id=args.run_id,
                poll_id=args.poll_id,
                failure_token=args.failure_token,
                failure_event_path=args.failure_event,
                service_role=args.service_role,
            )
        else:
            result = validate_collector(args.evidence_root, root=args.root)
            if args.forced_failure_receipt is not None and args.root is None:
                raise EvidenceError("rollback validation needs root")
            if args.forced_failure_receipt is not None:
                result["rollback"] = validate_rollback(
                    args.root, args.forced_failure_receipt
                )
        print(json.dumps(result, sort_keys=True))
        return 0 if args.command != "validate" or result.get("certification") is True else 1
    except (EvidenceError, OSError, store.DistillationStoreError) as exc:
        print(f"r7 evidence failed: {str(exc).split(':', 1)[0]}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
