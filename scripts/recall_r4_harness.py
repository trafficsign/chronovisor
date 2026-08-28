#!/usr/bin/env python3
"""Fail-closed, source-bound evidence harness for Recall R4.

The harness validates receipts; it does not run a teacher or make a provider
request.  A source-contract verdict is intentionally independent from the
production verdict: synthetic receipts can exercise the contract, while
production certification requires the fixed managed root's sealed runtime,
workset, ledger-checkpoint, and quality evidence chain.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import json
import math
import os
import pwd
import re
import sqlite3
import stat
import subprocess
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback is fail-closed.
    tomllib = None  # type: ignore[assignment]

R4_SCHEMA = "chronovisor.recall-r4.v1"
R4_AUTHORITY_RECEIPT_SCHEMA = "chronovisor.recall-r4-authority-receipt.v1"
_AUTHORITY_EMBEDDED_NAME = re.compile(r"[0-9]{4,}\.jsonl?")
RECEIPT_SCHEMA = "chronovisor.recall-r4-receipt.v1"
SOURCE_SCHEMA = "chronovisor.recall-r4-source-contract.v1"
LOCAL_PROFILE = "local-triad-v1"
OX_PROFILE = "deepseek-v4-flash-single-v1"
OX_ROUTE = "opencode-go/deepseek-v4-flash"
OX_MODEL = "deepseek-v4-flash"
OX_ENDPOINT = "https://opencode.ai/zen/go/v1/chat/completions"
OX_SCHEMA = "chronovisor.recall-distill-teacher-batch.v1"
OX_PROFILE_SCHEMA = "chronovisor.recall-distill-remote-profile.v2"
OX_LEGACY_PROFILE_SCHEMA = "chronovisor.recall-distill-ox-profile.v1"
OX_COHORT = "deepseek-v4-flash-backfill-v1"
OX_IDENTITY_REVISION = "deepseek-v4-flash-fixed-identity-v1"
OX_REQUEST_REVISION = "json-schema-core-label-abstain-16k-240s-v7"
OX_PROMPT_SHA256 = "f6a61adb72cafa813a7df9afd6d143c7636069358be17508ac7ad1c0a540bf5a"
OX_SCHEMA_SHA256 = "325a07d3a80d1aa38e9e95569af722b39de962c63994476f57d3baa3444786d7"
OX_ROUTE_SHA256 = "f9efd571e56d404593011ef2107c2ec56a5bc756193212caec9c0c05c4df576c"
OX_MODEL_SHA256 = "0e1ac8e00052dc78415580486fb8b4b65ae25525ad7683a4cfee574a6cd35185"
OX_KILL_CATEGORIES = (
    "402",
    "payment_required",
    "model_unavailable",
    "route_model_drift",
    "privacy_gate",
)
OX_FIXED_IDENTITY = {
    "revision": OX_IDENTITY_REVISION,
    "route_identity": {
        "provider": "opencode-go",
        "model": OX_ROUTE,
        "location": "remote",
    },
    "model_digest": OX_MODEL_SHA256,
    "route_digest": OX_ROUTE_SHA256,
    "prompt_template_sha256": OX_PROMPT_SHA256,
    "schema_revision_sha256": OX_SCHEMA_SHA256,
}
OX_STAGES = (1, 2, 5, 10)
OX_PROBE_REVISION = "deepseek-single-teacher-repeat-v1"
OX_MIN_BLIND_REPEAT_PAIRS = 20
LOCAL_ROLES = (
    "recall.distill.teacher.a",
    "recall.distill.teacher.b",
    "recall.distill.teacher.c",
)
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_OUTCOME_CLASSES = {"valid", "deferred", "invalid"}
_DEFERRED_REASONS = {"capacity", "timeout", "preemption"}
_INVALID_REASONS = {"schema", "coverage", "route_model_mismatch"}
_MAX_RECEIPT_BYTES = 512 * 1024
LOCAL_ASSIGNMENT_REVISION = "assignment-v2"
LOCAL_PROBE_REVISION = "probe-v2"
LOCAL_PROBE_RATE = 0.15
LOCAL_MIN_INITIAL_RECEIPTS = 20
LOCAL_MIN_VALID_RATE = 0.95
LOCAL_MAX_LOAD_SKEW = 0.10
LOCAL_PROBE_TOLERANCE = 0.02
_RECEIPT_ENVELOPE_KEYS = {
    "schema",
    "namespace",
    "artifact_id",
    "receipt_id",
    "receipt_identity",
    "receipt_sha256",
    "seal_sha256",
}
_LOCAL_RECEIPT_COMMON_KEYS = _RECEIPT_ENVELOPE_KEYS | {
    "profile",
    "captured_at",
    "source_commit",
    "source_tree_sha256",
    "work_id",
    "attempt",
    "rally_id",
    "candidate_id",
    "primary_owner",
    "probe",
    "assignment_revision",
    "probe_assignment_revision",
    "route_identity",
    "lane",
    "live_recall",
    "configured_max_inflight",
    "failure_injection",
    "outcome",
    "workset_receipt",
}
_LOCAL_VALID_RECEIPT_KEYS = _LOCAL_RECEIPT_COMMON_KEYS | {
    "label_record_sha256",
}
_LOCAL_FAILURE_RECEIPT_KEYS = _LOCAL_RECEIPT_COMMON_KEYS | {
    "attempt_record_sha256",
    "claim_receipt",
    "diagnostic",
}


def _account_identity() -> tuple[int, Path]:
    """Resolve the OS account identity without consulting HOME."""

    try:
        uid = os.getuid()
        home = Path(pwd.getpwuid(uid).pw_dir).resolve(strict=False)
    except (KeyError, OSError, RuntimeError, TypeError, ValueError):
        return -1, Path("/nonexistent/chronovisor-account")
    if uid < 0 or not home.is_absolute():
        return -1, Path("/nonexistent/chronovisor-account")
    return uid, home


ACCOUNT_UID, ACCOUNT_HOME = _account_identity()

# Production evidence is intentionally rooted at the one managed Chronovisor
# data directory.  The name is a module constant (rather than an environment
# variable or a CLI value) so an arbitrary fixture cannot become authoritative.
# Tests may replace this constant with an isolated root before calling the
# private collector; the normal CLI never accepts a root override.
PRODUCTION_ROOT = ACCOUNT_HOME / ".chronovisor"
PRODUCTION_DISTILLATION_RELATIVE = Path("runtime") / "recall-distillation"
PRODUCTION_WORKSET_RELATIVE = PRODUCTION_DISTILLATION_RELATIVE / "ox-workset.sqlite3"
PRODUCTION_CANDIDATE_RELATIVE = (
    PRODUCTION_DISTILLATION_RELATIVE / "candidate-ledger.jsonl"
)
PRODUCTION_CANDIDATE_CHECKPOINT_RELATIVE = (
    PRODUCTION_DISTILLATION_RELATIVE / "candidate-ledger.jsonl.head.json"
)
PRODUCTION_LABEL_RELATIVE = PRODUCTION_DISTILLATION_RELATIVE / "label-ledger.jsonl"
PRODUCTION_LABEL_CHECKPOINT_RELATIVE = (
    PRODUCTION_DISTILLATION_RELATIVE / "label-ledger.jsonl.head.json"
)
PRODUCTION_STATE_RELATIVE = PRODUCTION_DISTILLATION_RELATIVE / "state.json"
PRODUCTION_CONFIG_RELATIVE = Path("config.toml")
PRODUCTION_CONTRACT_DIR_RELATIVE = (
    PRODUCTION_DISTILLATION_RELATIVE / "ox-profile-contracts"
)
PRODUCTION_CANDIDATE_ANCHOR_RELATIVE = (
    PRODUCTION_DISTILLATION_RELATIVE / "r4-candidate-anchor.json"
)
PRODUCTION_EVENT_ANCHOR_RELATIVE = PRODUCTION_DISTILLATION_RELATIVE / "ox-event-anchors"
R4_CANDIDATE_ANCHOR_SCHEMA = "chronovisor.recall-r4-candidate-anchor.v1"
R4_FAULT_SCENARIO_SCHEMA = "chronovisor.recall-r4-fault-scenario.v1"
PRODUCTION_MAX_SQLITE_BYTES = 256 * 1024 * 1024
PRODUCTION_MAX_LEDGER_BYTES = 4 * 1024 * 1024 * 1024
PRODUCTION_MAX_FULL_LEDGER_BYTES = 8 * 1024 * 1024
PRODUCTION_MAX_LEDGER_TAIL_BYTES = 512 * 1024
PRODUCTION_MAX_ROWS = 100_000
PRODUCTION_WORKSET_STATES = ("ready", "leased", "completed", "quarantined")
PRODUCTION_WORKSET_OPERATIONS = {
    "advance",
    "claim_reclaim",
    "claim",
    "release",
    "commit",
}
PRODUCTION_RAMP_CAPS = (1, 2, 5, 10)
PRODUCTION_FAULT_SCENARIOS = (
    "http_429",
    "http_5xx",
    "timeout",
    "http_402_paid",
    "model_drift",
    "invalid_output_quarantine",
    "lease_expiry_reclaim",
    "resource_pressure_preemption",
    "disable_rollback",
)

# The candidate ledger is too large to re-hash on every hot-path collection.
# Its sealed R0 artifact is tracked by the clean source tree and therefore is
# the immutable baseline; changes require a fresh offline snapshot.
R0_SCHEMA = "chronovisor.recall-r0.v1"
R0_EVIDENCE_ID = "4de2cfe3f33e5c9c5153b264ebee8fae24d814856e0ac339e53c3077dc7efb33"
R0_EVIDENCE_RELATIVE = Path(
    "_handoff/evidence/2026-08-23-recall-distillation-recovery/"
    "r0-measured-baseline-4de2cfe3.json"
)


class R4Error(ValueError):
    """An R4 contract violation."""


def _json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise R4Error("value is not canonical JSON") from exc


def _sha256(value: object) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _expected_ox_request_sha256(
    *, profile_contract_id: str, payload_digest: str
) -> str:
    return _sha256(
        {
            "identity_revision": OX_IDENTITY_REVISION,
            "payload_digest": payload_digest,
            "profile_contract_id": profile_contract_id,
            "request_revision": OX_REQUEST_REVISION,
            "route_digest": OX_ROUTE_SHA256,
        }
    )


def _expected_ox_provider_request_sha256(
    *, profile_contract_id: str, payload_digest: str, work_id: str, expires_at: str
) -> str:
    return _sha256(
        {
            "contract": profile_contract_id,
            "expires_at": expires_at,
            "identity_revision": OX_IDENTITY_REVISION,
            "model_digest": OX_MODEL_SHA256,
            "payload_digest": payload_digest,
            "prompt_sha256": OX_PROMPT_SHA256,
            "route_digest": OX_ROUTE_SHA256,
            "schema_sha256": OX_SCHEMA_SHA256,
            "work_id": work_id,
        }
    )


def _publish_owned_artifact(
    directory: Path,
    name: str,
    encoded: bytes,
    *,
    before_publish: Callable[[], None] | None = None,
) -> Path:
    """Publish and re-read a sealed artifact through one stable directory FD."""

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    directory_fd = os.open(directory, flags)
    temporary = f".{name}.tmp"
    try:
        opened_directory = os.fstat(directory_fd)

        def verify_directory() -> None:
            try:
                named_directory = os.stat(directory, follow_symlinks=False)
            except OSError as exc:
                raise R4Error("artifact directory changed during publication") from exc
            if (
                stat.S_ISLNK(named_directory.st_mode)
                or named_directory.st_dev != opened_directory.st_dev
                or named_directory.st_ino != opened_directory.st_ino
            ):
                raise R4Error("artifact directory changed during publication")

        try:
            existing_fd = os.open(
                name,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=directory_fd,
            )
        except FileNotFoundError:
            existing_fd = -1
        except OSError as exc:
            raise R4Error("immutable artifact path is unsafe") from exc
        if existing_fd >= 0:
            with os.fdopen(existing_fd, "rb") as handle:
                if handle.read() != encoded:
                    raise R4Error("owned fault scenario artifact conflict")
            verify_directory()
            return directory / name
        try:
            fd = os.open(
                temporary,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=directory_fd,
            )
        except OSError as exc:
            raise R4Error("immutable artifact temporary path is unsafe") from exc
        with os.fdopen(fd, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        if before_publish is not None:
            before_publish()
        os.replace(temporary, name, src_dir_fd=directory_fd, dst_dir_fd=directory_fd)
        os.fsync(directory_fd)
        fd = os.open(name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=directory_fd)
        with os.fdopen(fd, "rb") as handle:
            if handle.read() != encoded:
                raise R4Error("owned fault scenario artifact readback failed")
        # The name used by callers must still resolve to the directory we
        # opened.  The dirfd keeps publication safe during a rename race; this
        # check prevents returning a pathname redirected to an attacker's
        # replacement directory while the callback was running.
        verify_directory()
        return directory / name
    finally:
        try:
            os.unlink(temporary, dir_fd=directory_fd)
        except FileNotFoundError:
            pass
        os.close(directory_fd)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    except OSError as exc:
        raise R4Error(f"file read failed: {path.name}") from exc
    return digest.hexdigest()


def _stat(path: Path) -> dict[str, int]:
    try:
        value = path.lstat()
    except OSError as exc:
        raise R4Error(f"path stat failed: {path}") from exc
    if stat.S_ISLNK(value.st_mode) or not stat.S_ISREG(value.st_mode):
        raise R4Error(f"path is not a regular file: {path.name}")
    return {
        "st_dev": int(value.st_dev),
        "st_ino": int(value.st_ino),
        "st_mode": int(value.st_mode & 0o7777),
        "st_size": int(value.st_size),
        "st_mtime_ns": int(value.st_mtime_ns),
    }


def _has_symlink_component(path: Path) -> bool:
    current = path.expanduser()
    while True:
        if current.is_symlink():
            return True
        parent = current.parent
        if parent == current:
            return False
        current = parent


def _reject_original_symlinks(paths: Iterable[tuple[str, Path]]) -> None:
    """Reject symlink components before resolve() can hide the boundary."""

    for name, original in paths:
        if _has_symlink_component(original.expanduser().absolute()):
            raise R4Error(f"{name} path contains a symlink")


def _overlap(left: Path, right: Path) -> bool:
    a = left.expanduser().resolve(strict=False)
    b = right.expanduser().resolve(strict=False)
    return a == b or a.is_relative_to(b) or b.is_relative_to(a)


def assert_root_matrix(
    source_root: Path,
    output: Path,
    production_root: Path | None = None,
    input_roots: Iterable[Path] = (),
) -> None:
    """Reject symlink entry points and output/input overlap before reading."""

    roots = {"source": source_root, "output": output}
    if production_root is not None:
        roots["production"] = production_root
    roots.update({f"input[{i}]": value for i, value in enumerate(input_roots)})
    for name, path in roots.items():
        if _has_symlink_component(path):
            raise R4Error(f"{name} path contains a symlink")
    entries = tuple(roots.items())
    for index, (left_name, left) in enumerate(entries):
        for right_name, right in entries[index + 1 :]:
            if _overlap(left, right):
                raise R4Error(f"{left_name}/{right_name} paths overlap")
    if not source_root.is_dir():
        raise R4Error("source root is not a directory")
    if output.exists() and not output.is_dir():
        raise R4Error("output root is not a directory")
    if output.is_dir() and any(item.is_symlink() for item in output.rglob("*")):
        raise R4Error("output tree contains a symlink")


def _tracked_paths(root: Path) -> list[str]:
    try:
        result = subprocess.run(
            ["git", "ls-files", "-co", "--exclude-standard", "-z"],
            cwd=root,
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise R4Error("source inventory failed") from exc
    values = [os.fsdecode(value) for value in result.stdout.split(b"\0") if value]
    return sorted(values)


def _git_mode_blob_paths(
    root: Path, command: list[str]
) -> tuple[tuple[bytes, bytes, bytes], ...]:
    try:
        output = subprocess.run(
            command, cwd=root, check=True, capture_output=True
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise R4Error("source index snapshot failed") from exc
    entries: list[tuple[bytes, bytes, bytes]] = []
    for entry in output.split(b"\0"):
        if not entry:
            continue
        try:
            metadata, path = entry.split(b"\t", 1)
            fields = metadata.split(b" ")
            if command[2] == "-s":
                mode, blob, stage = fields
                if stage != b"0":
                    raise ValueError
            else:
                mode, _kind, blob = fields
        except ValueError as exc:
            raise R4Error("source index snapshot is malformed") from exc
        entries.append((mode, blob, path))
    return tuple(sorted(entries))


def _source_tree_digest(root: Path) -> dict[str, Any]:
    """Hash the source tree with lstat-before/after TOCTOU checks."""

    index_before = _git_mode_blob_paths(root, ["git", "ls-files", "-s", "-z"])
    head_before = _git_mode_blob_paths(root, ["git", "ls-tree", "-r", "-z", "HEAD"])
    if index_before != head_before:
        raise R4Error("source index differs from HEAD tree")
    try:
        initial_commit = subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise R4Error("source initial HEAD lookup failed") from exc
    expected_blobs = {
        os.fsdecode(path): (mode, blob) for mode, blob, path in head_before
    }
    digest = hashlib.sha256()
    files = 0
    symlinks = 0
    ox_identity_sha256 = ""
    for relative in _tracked_paths(root):
        path = root / relative
        try:
            before = path.lstat()
        except OSError as exc:
            raise R4Error("source path disappeared during capture") from exc
        if stat.S_ISLNK(before.st_mode):
            raise R4Error(f"source tree contains tracked symlink: {relative}")
        elif stat.S_ISREG(before.st_mode):
            if before.st_size > 512 * 1024 * 1024:
                raise R4Error(f"source file is too large: {relative}")
            content = _file_sha256(path)
            try:
                raw = path.read_bytes()
                expected_mode, expected_blob = expected_blobs[relative]
            except (KeyError, OSError) as exc:
                raise R4Error("source HEAD tree entry is unavailable") from exc
            actual_blob = (
                hashlib.sha1(f"blob {len(raw)}\0".encode("ascii") + raw)
                .hexdigest()
                .encode("ascii")
            )
            if (
                actual_blob != expected_blob
                or f"{before.st_mode & 0o777777:o}".encode("ascii") != expected_mode
            ):
                raise R4Error("source dirty: file differs from HEAD tree")
            try:
                after = path.lstat()
            except OSError as exc:
                raise R4Error("source path disappeared during capture") from exc
            if (
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            ) != (
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            ):
                raise R4Error("source changed during capture")
            record = {
                "kind": "file",
                "path": relative,
                "size": int(before.st_size),
                "sha256": content,
            }
            if (
                relative
                == "src/chronovisor/recall/recall_distillation_remote_teacher.py"
            ):
                ox_identity_sha256 = content
            files += 1
        else:
            raise R4Error(f"source path is not a file or symlink: {relative}")
        digest.update(_json_bytes(record))
        digest.update(b"\n")
    try:
        status = subprocess.run(
            ["git", "status", "--short", "--untracked-files=all"],
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout.decode("utf-8", "replace")
        commit = subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise R4Error("source git state lookup failed") from exc
    for relative, (expected_mode, expected_blob) in expected_blobs.items():
        path = root / relative
        try:
            final_stat = path.lstat()
            final_bytes = path.read_bytes()
        except OSError as exc:
            raise R4Error("source file disappeared during final sweep") from exc
        final_blob = (
            hashlib.sha1(f"blob {len(final_bytes)}\0".encode("ascii") + final_bytes)
            .hexdigest()
            .encode("ascii")
        )
        if (
            not stat.S_ISREG(final_stat.st_mode)
            or f"{final_stat.st_mode & 0o777777:o}".encode("ascii") != expected_mode
            or final_blob != expected_blob
        ):
            raise R4Error("source file changed during final sweep")
    index_after = _git_mode_blob_paths(root, ["git", "ls-files", "-s", "-z"])
    head_after = _git_mode_blob_paths(root, ["git", "ls-tree", "-r", "-z", "HEAD"])
    if (
        index_after != index_before
        or head_after != head_before
        or index_after != head_after
        or commit != initial_commit
    ):
        raise R4Error("source index changed during capture")
    return {
        "commit": commit,
        "clean": not status,
        "status_sha256": _sha256(status),
        "status_count": len(status.splitlines()),
        "tree_sha256": digest.hexdigest(),
        "file_count": files,
        "symlink_count": symlinks,
        "ox_identity_sha256": ox_identity_sha256,
        "account_uid": ACCOUNT_UID,
        "account_home": str(ACCOUNT_HOME),
    }


def _assert_source(source_root: Path, expected_commit: str) -> dict[str, Any]:
    if _COMMIT.fullmatch(expected_commit) is None:
        raise R4Error("expected source commit is not a full SHA-1")
    snapshot = _source_tree_digest(source_root)
    if snapshot["commit"] != expected_commit:
        raise R4Error("source HEAD does not match expected commit")
    if snapshot["clean"] is not True:
        raise R4Error("source checkout is dirty")
    return snapshot


def _sealed(payload: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = {key: value for key, value in payload.items() if key != "seal_sha256"}
    return {**unsigned, "seal_sha256": _sha256(unsigned)}


def _producer_receipt_digest(value: Mapping[str, Any]) -> str:
    """Digest the unsigned producer receipt, excluding its seal and digest."""

    return _sha256(
        {
            key: item
            for key, item in value.items()
            if key not in {"seal_sha256", "receipt_sha256"}
        }
    )


def _verify_seal(value: object, *, schema: str = RECEIPT_SCHEMA) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise R4Error("receipt is not an object")
    if value.get("schema") != schema:
        raise R4Error("receipt schema mismatch")
    if value.get("namespace") != "recall-distillation":
        raise R4Error("receipt namespace mismatch")
    if value.get("seal_sha256") != _sha256(
        {key: item for key, item in value.items() if key != "seal_sha256"}
    ):
        raise R4Error("receipt seal mismatch")
    receipt_id = (
        value.get("receipt_id")
        if schema == RECEIPT_SCHEMA
        else value.get("artifact_id")
    )
    if not isinstance(receipt_id, str) or _SHA.fullmatch(receipt_id) is None:
        raise R4Error("receipt id is invalid")
    if schema == RECEIPT_SCHEMA:
        identity = value.get("receipt_identity")
        if (
            value.get("artifact_id") != receipt_id
            or not isinstance(identity, Mapping)
            or not identity
            or receipt_id != _sha256(identity)
            or value.get("receipt_sha256") != _producer_receipt_digest(value)
        ):
            raise R4Error("receipt identity binding is invalid")
    return value


def _receipt_file(path: Path) -> list[dict[str, Any]]:
    before = _stat(path)
    before_digest = _file_sha256(path)
    if before["st_size"] > _MAX_RECEIPT_BYTES:
        raise R4Error("receipt file exceeds bounded size")
    try:
        raw = path.read_bytes()
        if path.suffix == ".jsonl":
            payload = [json.loads(line) for line in raw.splitlines() if line.strip()]
        else:
            payload = json.loads(raw)
    except (OSError, ValueError, UnicodeError) as exc:
        raise R4Error(f"receipt JSON is invalid: {path.name}") from exc
    after = _stat(path)
    if before != after:
        raise R4Error("receipt changed during capture")
    values = payload if isinstance(payload, list) else [payload]
    receipts = [_verify_seal(value) for value in values]
    final = _stat(path)
    if final != before or _file_sha256(path) != before_digest:
        raise R4Error("receipt final re-stat failed")
    return receipts


def load_receipts(path: Path | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load sealed JSON/JSONL receipts and return a final file inventory."""

    if path is None:
        return [], {"files": [], "count": 0}
    if _has_symlink_component(path):
        raise R4Error("receipt input path contains a symlink")
    paths = (
        sorted((*path.glob("*.json"), *path.glob("*.jsonl")))
        if path.is_dir()
        else [path]
    )
    if not paths:
        raise R4Error("receipt input is empty")
    receipts: list[dict[str, Any]] = []
    files: list[dict[str, Any]] = []
    for item in paths:
        if item.is_symlink() or not item.is_file():
            raise R4Error("receipt input contains an unsafe entry")
        before = _stat(item)
        values = _receipt_file(item)
        digest = _file_sha256(item)
        if _stat(item) != before:
            raise R4Error("receipt changed after validation")
        files.append({"path": item.name, "sha256": digest, "file_state": before})
        receipts.extend(values)
    ids = [str(value["receipt_id"]) for value in receipts]
    if len(ids) != len(set(ids)):
        raise R4Error("duplicate receipt id")
    return receipts, {"files": files, "count": len(receipts)}


def _text(value: object) -> str:
    return value if isinstance(value, str) else ""


def _int(value: object, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise R4Error("receipt integer is invalid")
    return value


def _expected_owner(rally_id: str, candidate_id: str) -> str:
    key = f"{LOCAL_ASSIGNMENT_REVISION}\0{rally_id}\0{candidate_id}"
    index = int(hashlib.sha256(key.encode()).hexdigest()[:16], 16)
    return LOCAL_ROLES[index % len(LOCAL_ROLES)]


def _expected_probe(rally_id: str, candidate_id: str) -> bool:
    key = f"{LOCAL_PROBE_REVISION}\0{rally_id}\0{candidate_id}"
    digest = hashlib.sha256(key.encode()).digest()
    return int.from_bytes(digest[:8], "big") % 10_000 < 1_500


def _validate_local(
    receipts: Sequence[Mapping[str, Any]], source: Mapping[str, Any]
) -> dict[str, Any]:
    reasons: set[str] = set()
    rows = [row for row in receipts if row.get("profile") == LOCAL_PROFILE]
    if len(rows) != len(receipts):
        reasons.add("profile_mixing")
    identities: dict[str, tuple[str, str]] = {}
    owners: Counter[str] = Counter()
    categories: set[str] = set()
    quality_rows: list[Mapping[str, Any]] = []
    valid = deferred = invalid = 0
    probe_revision = ""
    work_attempts: set[tuple[str, int]] = set()
    failure_worksets: dict[str, tuple[str, str, str, list[str]]] = {}
    for row in rows:
        failure_injection = row.get("failure_injection")
        expected_keys = (
            _LOCAL_FAILURE_RECEIPT_KEYS
            if failure_injection is True
            else _LOCAL_VALID_RECEIPT_KEYS
        )
        if set(row) != expected_keys:
            reasons.add("local_receipt_shape_invalid")
            continue
        work_id = _text(row.get("work_id"))
        attempt = row.get("attempt")
        binding_key = (
            "attempt_record_sha256"
            if failure_injection is True
            else "label_record_sha256"
        )
        binding_sha256 = _text(row.get(binding_key))
        workset_receipt = row.get("workset_receipt")
        expected_identity = {
            "profile": LOCAL_PROFILE,
            "work_id": work_id,
            "attempt": attempt,
            binding_key: binding_sha256,
        }
        claim_receipt = row.get("claim_receipt")
        if failure_injection is True and isinstance(claim_receipt, Mapping):
            expected_identity.update(
                {
                    "captured_at": row.get("captured_at"),
                    "claim_receipt_sha256": claim_receipt.get("head_sha256"),
                }
            )
        workset_binding_valid = isinstance(workset_receipt, Mapping)
        if failure_injection is True:
            workset_keys = {
                "generation",
                "head_sha256",
                "operation",
                "selection_sha256",
                "work_ids_sha256",
            }
            workset_binding_valid = (
                workset_binding_valid
                and set(workset_receipt)
                in (workset_keys, {*workset_keys, "context_sha256"})
                and isinstance(claim_receipt, Mapping)
                and set(claim_receipt)
                == {
                    "generation",
                    "head_sha256",
                    "selection_sha256",
                    "work_ids_sha256",
                }
                and not isinstance(claim_receipt.get("generation"), bool)
                and isinstance(claim_receipt.get("generation"), int)
                and claim_receipt["generation"] >= 1
                and not isinstance(workset_receipt.get("generation"), bool)
                and isinstance(workset_receipt.get("generation"), int)
                and workset_receipt["generation"]
                == claim_receipt["generation"] + 1
                and workset_receipt.get("operation") in {"release", "commit"}
                and workset_receipt.get("selection_sha256")
                == claim_receipt.get("selection_sha256")
                and workset_receipt.get("work_ids_sha256")
                == claim_receipt.get("work_ids_sha256")
                and all(
                    _SHA.fullmatch(_text(receipt.get(key))) is not None
                    for receipt, keys in (
                        (
                            claim_receipt,
                            ("head_sha256", "selection_sha256", "work_ids_sha256"),
                        ),
                        (
                            workset_receipt,
                            ("head_sha256", "selection_sha256", "work_ids_sha256"),
                        ),
                    )
                    for key in keys
                )
                and (
                    "context_sha256" not in workset_receipt
                    or _SHA.fullmatch(_text(workset_receipt.get("context_sha256")))
                    is not None
                )
            )
        elif workset_binding_valid:
            workset_binding_valid = (
                set(workset_receipt) == {"generation", "head_sha256"}
                and not isinstance(workset_receipt.get("generation"), bool)
                and isinstance(workset_receipt.get("generation"), int)
                and workset_receipt["generation"] >= 1
                and _SHA.fullmatch(_text(workset_receipt.get("head_sha256")))
                is not None
            )
        if (
            _parse_expiry(row.get("captured_at")) is None
            or re.fullmatch(r"local-teacher-[0-9a-f]{64}", work_id) is None
            or isinstance(attempt, bool)
            or not isinstance(attempt, int)
            or attempt < 1
            or _SHA.fullmatch(binding_sha256) is None
            or not workset_binding_valid
            or row.get("receipt_identity") != expected_identity
            or row.get("receipt_id") != _sha256(expected_identity)
        ):
            reasons.add("local_durable_binding_invalid")
        elif (work_id, attempt) in work_attempts:
            reasons.add("local_work_attempt_duplicate")
        else:
            work_attempts.add((work_id, attempt))
            if failure_injection is True:
                assert isinstance(claim_receipt, Mapping)
                assert isinstance(workset_receipt, Mapping)
                head_sha256 = _text(workset_receipt.get("head_sha256"))
                binding = (
                    _text(claim_receipt.get("head_sha256")),
                    _text(workset_receipt.get("selection_sha256")),
                    _text(workset_receipt.get("work_ids_sha256")),
                )
                group = failure_worksets.setdefault(head_sha256, (*binding, []))
                if group[:3] != binding:
                    reasons.add("local_failure_workset_binding_invalid")
                else:
                    group[3].append(work_id)
        identity = row.get("route_identity")
        if not isinstance(identity, Mapping):
            reasons.add("route_identity_missing")
            continue
        if set(identity) != {"role", "provider", "model", "location"}:
            reasons.add("route_identity_shape_invalid")
        role = _text(identity.get("role"))
        provider = _text(identity.get("provider"))
        model = _text(identity.get("model"))
        location = _text(identity.get("location"))
        if (
            role not in LOCAL_ROLES
            or not provider
            or location != "local"
            or not model
        ):
            reasons.add("route_model_mismatch")
        else:
            identity_value = (provider, model)
            if role in identities and identities[role] != identity_value:
                reasons.add("local_route_model_mixing")
            identities[role] = identity_value
        if (
            row.get("source_commit") != source["commit"]
            or row.get("source_tree_sha256") != source["tree_sha256"]
        ):
            reasons.add("source_binding_mismatch")
        candidate = _text(row.get("candidate_id"))
        rally_id = _text(row.get("rally_id"))
        if not rally_id:
            reasons.add("rally_identity_missing")
        owner = _text(row.get("primary_owner"))
        owners[owner] += 1
        if not candidate or owner != _expected_owner(rally_id, candidate):
            reasons.add("primary_ownership_nondeterministic")
        probe = row.get("probe")
        if not isinstance(probe, bool) or probe != _expected_probe(rally_id, candidate):
            reasons.add("probe_assignment_nondeterministic")
        if row.get("assignment_revision") != LOCAL_ASSIGNMENT_REVISION:
            reasons.add("assignment_revision_mismatch")
        revision = _text(row.get("probe_assignment_revision"))
        if revision != LOCAL_PROBE_REVISION:
            reasons.add("probe_assignment_missing")
        elif not probe_revision:
            probe_revision = revision
        elif revision != probe_revision:
            reasons.add("probe_assignment_mixing")
        lane = row.get("lane")
        if (
            not isinstance(lane, Mapping)
            or set(lane) != {"mode", "purpose", "admitted", "inflight"}
            or lane.get("mode") != "sleep"
            or lane.get("purpose") != "sleep"
            or lane.get("admitted") is not True
            or isinstance(lane.get("inflight"), bool)
            or lane.get("inflight") != 1
        ):
            reasons.add("scheduler_lane_invalid")
        live = row.get("live_recall")
        if (
            not isinstance(live, Mapping)
            or set(live) != {"model_calls", "remote_egress"}
            or isinstance(live.get("model_calls"), bool)
            or isinstance(live.get("remote_egress"), bool)
            or live.get("model_calls") != 0
            or live.get("remote_egress") != 0
        ):
            reasons.add("live_recall_egress")
        if failure_injection is True:
            diagnostic = row.get("diagnostic")
            if (
                not isinstance(diagnostic, Mapping)
                or set(diagnostic) != {"provider_calls", "network_egress"}
                or isinstance(diagnostic.get("provider_calls"), bool)
                or isinstance(diagnostic.get("network_egress"), bool)
                or diagnostic.get("provider_calls") != 0
                or diagnostic.get("network_egress") != 0
            ):
                reasons.add("failure_diagnostic_egress")
        outcome = row.get("outcome")
        if not isinstance(outcome, Mapping):
            reasons.add("outcome_missing")
            continue
        outcome_class = _text(outcome.get("class"))
        reason = _text(outcome.get("reason"))
        expected_outcome_keys = (
            {"class", "reason", "schema_valid", "coverage_valid"}
            if outcome_class == "valid"
            else {"class", "reason"}
        )
        if set(outcome) != expected_outcome_keys:
            reasons.add("outcome_shape_invalid")
        if outcome_class not in _OUTCOME_CLASSES:
            reasons.add("outcome_class_invalid")
        elif outcome_class == "valid":
            valid += 1
            if (
                reason != "ok"
                or outcome.get("schema_valid") is not True
                or outcome.get("coverage_valid") is not True
            ):
                reasons.add("valid_outcome_unverified")
        elif outcome_class == "deferred":
            deferred += 1
            if reason not in _DEFERRED_REASONS:
                reasons.add("deferred_reason_invalid")
        else:
            invalid += 1
            if reason not in _INVALID_REASONS:
                reasons.add("invalid_reason_invalid")
        if (outcome_class == "valid") != (failure_injection is False):
            reasons.add("failure_injection_outcome_mismatch")
        if reason and failure_injection is True:
            categories.add(reason)
        configured_max_inflight = row.get("configured_max_inflight")
        if (
            isinstance(configured_max_inflight, bool)
            or not isinstance(configured_max_inflight, int)
            or not 1 <= configured_max_inflight <= 10
        ):
            reasons.add("local_configured_inflight_invalid")
        if not isinstance(failure_injection, bool):
            reasons.add("failure_injection_invalid")
        elif failure_injection is False and outcome_class == "valid":
            quality_rows.append(row)
    if set(identities) != set(LOCAL_ROLES):
        reasons.add("local_identity_count_not_three")
    if len(set(identities.values())) != len(identities):
        reasons.add("local_route_model_not_distinct")
    if set(owners) - set(LOCAL_ROLES):
        reasons.add("unknown_primary_owner")
    quality_owners = Counter(_text(row.get("primary_owner")) for row in quality_rows)
    counts = [quality_owners[role] for role in LOCAL_ROLES]
    if quality_rows and max(counts) > 0:
        skew = (max(counts) - min(counts)) / max(counts)
    else:
        skew = 0.0
    if not math.isfinite(skew):
        reasons.add("load_skew_invalid")
    elif skew > LOCAL_MAX_LOAD_SKEW:
        reasons.add("load_skew_above_p0")
    valid_quality = sum(
        row.get("outcome", {}).get("class") == "valid"
        for row in quality_rows
        if isinstance(row.get("outcome"), Mapping)
    )
    valid_rate = valid_quality / len(quality_rows) if quality_rows else 0.0
    if len(quality_rows) < LOCAL_MIN_INITIAL_RECEIPTS:
        reasons.add("local_initial_receipts_below_floor")
    if valid_rate < LOCAL_MIN_VALID_RATE:
        reasons.add("local_valid_rate_below_floor")
    probe_rate = (
        sum(bool(row.get("probe")) for row in quality_rows) / len(quality_rows)
        if quality_rows
        else 0.0
    )
    if abs(probe_rate - LOCAL_PROBE_RATE) > LOCAL_PROBE_TOLERANCE:
        reasons.add("probe_rate_not_fifteen_percent")
    required_categories = _DEFERRED_REASONS | _INVALID_REASONS
    if any(
        _sha256(sorted(work_ids)) != expected
        for _, _, expected, work_ids in failure_worksets.values()
    ):
        reasons.add("local_failure_workset_binding_invalid")
    if not required_categories.issubset(categories):
        reasons.add("failure_class_coverage_incomplete")
    if not valid:
        reasons.add("valid_outcome_missing")
    return {
        "profile": LOCAL_PROFILE,
        "passed": not reasons,
        "reasons": sorted(reasons),
        "rows": len(rows),
        "identities": {
            role: {"provider": p, "model": m} for role, (p, m) in identities.items()
        },
        "load": {
            "per_role": dict(quality_owners),
            "skew": round(skew, 8),
            "max_skew": LOCAL_MAX_LOAD_SKEW,
        },
        "outcomes": {
            "valid": valid,
            "deferred": deferred,
            "invalid": invalid,
            "categories": sorted(categories),
        },
        "probe_assignment_revision": probe_revision,
        "probe_rate": round(probe_rate, 8),
        "valid_rate": round(valid_rate, 8),
        "initial_receipts": len(quality_rows),
    }


_OX_EXPIRY_RE = re.compile(
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:Z|[+-]\d{2}:\d{2})\Z"
)
_OX_MAX_EXPIRY = datetime(2100, 1, 1, tzinfo=UTC)


def _parse_expiry(value: object) -> float | None:
    """Parse only timezone-aware RFC3339 strings; timestamps are not accepted."""

    if not isinstance(value, str) or _OX_EXPIRY_RE.fullmatch(value) is None:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError, OverflowError):
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    try:
        return parsed.timestamp()
    except (OverflowError, OSError, ValueError):
        return None


def _canonical_future_expiry(value: object) -> str | None:
    parsed_timestamp = _parse_expiry(value)
    if parsed_timestamp is None:
        return None
    try:
        parsed = datetime.fromtimestamp(parsed_timestamp, tz=UTC)
    except (OverflowError, OSError, ValueError):
        return None
    if parsed <= datetime.now(UTC) or parsed >= _OX_MAX_EXPIRY:
        return None
    return parsed.strftime("%Y-%m-%dT%H:%M:%SZ")


def _runtime_workset_projection_is_valid(
    workset: object, stages: object, *, label_count: object
) -> bool:
    """Validate the bounded Workset summary carried by the runtime collector."""

    if not isinstance(workset, Mapping) or set(workset) != {
        "rows",
        "counts",
        "sha256",
        "receipts",
    }:
        return False
    rows = workset.get("rows")
    counts = workset.get("counts")
    receipts = workset.get("receipts")
    if (
        isinstance(label_count, bool)
        or not isinstance(label_count, int)
        or label_count < 1
        or isinstance(rows, bool)
        or not isinstance(rows, int)
        or rows < 1
        or not isinstance(counts, Mapping)
        or set(counts) != set(PRODUCTION_WORKSET_STATES)
        or _SHA.fullmatch(_text(workset.get("sha256"))) is None
        or not isinstance(receipts, Mapping)
        or set(receipts)
        != {
            "count",
            "generation",
            "head_sha256",
            "by_generation",
            "verified",
            "status",
            "legacy_unverified_excluded",
        }
    ):
        return False
    if any(
        isinstance(counts.get(state), bool)
        or not isinstance(counts.get(state), int)
        or counts[state] < 0
        for state in PRODUCTION_WORKSET_STATES
    ) or rows != sum(counts.values()) or counts.get("leased") != 0:
        return False
    receipt_count = receipts.get("count")
    generation = receipts.get("generation")
    by_generation = receipts.get("by_generation")
    if (
        isinstance(receipt_count, bool)
        or not isinstance(receipt_count, int)
        or receipt_count < 1
        or generation != receipt_count
        or _SHA.fullmatch(_text(receipts.get("head_sha256"))) is None
        or receipts.get("verified") is not True
        or receipts.get("status") != "verified"
        or receipts.get("legacy_unverified_excluded") is not False
        or not isinstance(by_generation, Mapping)
        or set(by_generation) != {str(value) for value in range(1, receipt_count + 1)}
    ):
        return False
    for receipt in by_generation.values():
        if (
            not isinstance(receipt, Mapping)
            or set(receipt) != {"receipt_sha256", "operation", "details"}
            or _SHA.fullmatch(_text(receipt.get("receipt_sha256"))) is None
            or receipt.get("operation") not in PRODUCTION_WORKSET_OPERATIONS
            or not isinstance(receipt.get("details"), Mapping)
        ):
            return False
    if not isinstance(stages, Mapping):
        return False
    seen_work_ids: set[str] = set()
    valid_receipts_total = 0
    for cap in OX_STAGES:
        stage = stages.get(str(cap))
        if not isinstance(stage, Mapping):
            return False
        valid = stage.get("valid_receipts")
        attempts = stage.get("attempts")
        rate = stage.get("valid_rate")
        work_ids = stage.get("work_ids")
        if (
            isinstance(valid, bool)
            or not isinstance(valid, int)
            or valid < 20
            or isinstance(attempts, bool)
            or not isinstance(attempts, int)
            or attempts < valid
            or type(rate) is not float
            or not math.isfinite(rate)
            or rate != round(valid / attempts, 8)
            or rate < 0.95
            or not isinstance(work_ids, list)
            or len(work_ids) != valid
            or any(
                not isinstance(work_id, str)
                or _SHA.fullmatch(work_id) is None
                for work_id in work_ids
            )
        ):
            return False
        stage_work_ids = set(work_ids)
        if len(stage_work_ids) != len(work_ids) or seen_work_ids.intersection(
            stage_work_ids
        ):
            return False
        seen_work_ids.update(stage_work_ids)
        valid_receipts_total += valid
    return (
        counts.get("completed") == label_count
        and valid_receipts_total <= label_count
    )


def _validate_runtime_ox_projection(
    projection: Mapping[str, Any], source: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate the producer's sealed OX contract from fixed-root evidence."""

    reasons: set[str] = set()
    profile = projection.get("profile_contract")
    contract = profile.get("sealed") if isinstance(profile, Mapping) else None
    expected_source = {
        "source_commit": source.get("commit"),
        "source_tree_sha256": source.get("tree_sha256"),
        "source_ox_identity_sha256": source.get("ox_identity_sha256"),
    }
    expected_keys = {
        "schema",
        "namespace",
        "artifact_id",
        "seal_sha256",
        "kind",
        "profile",
        "cohort",
        "route",
        "endpoint",
        "request_model",
        "required_returned_model",
        "request_revision",
        "fixed_identity",
        "free_only",
        "no_paid_fallback",
        "official_status",
        "expires_at",
        "docs_url",
        "kill_categories",
        "max_inflight",
        "teacher_claim_limit",
        "live_recall_model_calls",
        "source_commit",
        "source_tree_sha256",
        "source_ox_identity_sha256",
        "relevant_config_sha256",
    }
    if not isinstance(contract, Mapping) or set(contract) != expected_keys:
        reasons.add("runtime_contract_invalid")
        contract = {}
    else:
        unsigned = {
            key: value
            for key, value in contract.items()
            if key not in {"artifact_id", "seal_sha256"}
        }
        expected = {
            "schema": OX_PROFILE_SCHEMA,
            "namespace": "recall-distillation",
            "kind": "opencode-go-subscription-profile",
            "profile": OX_PROFILE,
            "cohort": OX_COHORT,
            "route": OX_ROUTE,
            "endpoint": OX_ENDPOINT,
            "request_model": OX_MODEL,
            "required_returned_model": OX_MODEL,
            "request_revision": OX_REQUEST_REVISION,
            "fixed_identity": OX_FIXED_IDENTITY,
            "free_only": False,
            "no_paid_fallback": True,
            "official_status": "subscription",
            "docs_url": "https://dev.opencode.ai/docs/go/",
            "kill_categories": list(OX_KILL_CATEGORIES),
            "max_inflight": 10,
            "teacher_claim_limit": 1,
            "live_recall_model_calls": 0,
            **expected_source,
        }
        if (
            contract.get("artifact_id") != _sha256(unsigned)
            or contract.get("seal_sha256")
            != _sha256({"artifact_id": contract.get("artifact_id"), **unsigned})
            or any(contract.get(key) != value for key, value in expected.items())
            or _canonical_future_expiry(contract.get("expires_at"))
            != contract.get("expires_at")
            or _SHA.fullmatch(_text(contract.get("relevant_config_sha256"))) is None
        ):
            reasons.add("runtime_contract_invalid")
        expected_id = contract.get("artifact_id")
        if (
            not isinstance(profile, Mapping)
            or profile.get("artifact_id") != expected_id
            or profile.get("sha256")
            != hashlib.sha256(_json_bytes(contract) + b"\n").hexdigest()
        ):
            reasons.add("runtime_contract_binding_invalid")
    labels = projection.get("labels")
    events = projection.get("events")
    quality = projection.get("quality")

    def event_head_valid(kind: str, *, required: bool) -> bool:
        event = events.get(kind) if isinstance(events, Mapping) else None
        count = event.get("count") if isinstance(event, Mapping) else None
        head = event.get("head_sha256") if isinstance(event, Mapping) else None
        return bool(
            isinstance(event, Mapping)
            and set(event) == {"count", "head_sha256"}
            and isinstance(count, int)
            and not isinstance(count, bool)
            and count >= int(required)
            and (
                (count == 0 and head == "")
                or (count > 0 and _SHA.fullmatch(_text(head)))
            )
        )

    stage_bindings_valid = (
        isinstance(quality, Mapping)
        and isinstance(quality.get("stages"), Mapping)
        and _runtime_workset_projection_is_valid(
            projection.get("workset"),
            quality["stages"],
            label_count=labels.get("count") if isinstance(labels, Mapping) else None,
        )
    )
    if (
        not isinstance(labels, Mapping)
        or not isinstance(labels.get("count"), int)
        or labels.get("count", 0) < 80
        or _SHA.fullmatch(_text(labels.get("head_sha256"))) is None
        or _SHA.fullmatch(_text(labels.get("sha256"))) is None
        or not isinstance(events, Mapping)
        # The runtime projection keeps a zero-count legacy ledger slot for
        # deterministic backward-compatible readback.  It is part of the
        # fixed producer shape, not an alternate source of OX authority.
        or set(events) != {"ramp", "failure", "lease", "legacy"}
        or not event_head_valid("ramp", required=True)
        or not event_head_valid("failure", required=False)
        or not event_head_valid("lease", required=False)
        or events.get("legacy") != {"count": 0, "head_sha256": ""}
        or not isinstance(quality, Mapping)
        or quality.get("receipt_authority") != "adapter_observed_not_provider_signed"
        or not isinstance(quality.get("stages"), Mapping)
        or set(quality["stages"]) != {str(cap) for cap in OX_STAGES}
        or not stage_bindings_valid
    ):
        reasons.add("runtime_label_or_event_binding_invalid")
    if not stage_bindings_valid:
        reasons.add("runtime_workset_binding_invalid")
    if projection.get("passed") is not True or projection.get("provider_calls") != 0:
        reasons.add("runtime_projection_unavailable")
    return {
        "profile": OX_PROFILE,
        "passed": not reasons,
        "reasons": sorted(reasons),
        "rows": 1,
        "contract": dict(contract),
        "stages": dict(quality.get("stages", {}))
        if isinstance(quality, Mapping)
        else {},
        "failure_receipts": ["runtime-sealed-fixed-root-v1"],
    }


def _runtime_ox_authority(projection: Mapping[str, Any]) -> dict[str, Any]:
    """Name the only OX authority allowed for a production R4 verdict."""

    return {
        "kind": "fixed_root_runtime_sealed_projection",
        "projection_sha256": _sha256(projection),
        "receipt_inventory": {"files": [], "count": 0},
    }


def _validate_ox(
    receipts: Sequence[Mapping[str, Any]],
    source: Mapping[str, Any],
    *,
    production_projection: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not receipts and production_projection is not None:
        return _validate_runtime_ox_projection(production_projection, source)
    reasons: set[str] = set()
    rows = [row for row in receipts if row.get("profile") == OX_PROFILE]
    if len(rows) != len(receipts):
        reasons.add("profile_mixing")
    identity: dict[str, Any] | None = None
    stages: dict[int, dict[str, Any]] = {}
    all_label_ids: list[str] = []
    all_commit_ids: list[str] = []
    all_work_ids: list[str] = []
    all_provider_receipts: set[str] = set()
    recovery_seen = False
    failure_flags: set[str] = set()
    stage_order: list[int] = []
    previous_captured_at: float | None = None
    for row in rows:
        captured_at = _parse_expiry(row.get("captured_at"))
        if captured_at is None:
            reasons.add("ox_captured_at_invalid")
        elif previous_captured_at is not None and captured_at < previous_captured_at:
            reasons.add("ox_captured_at_not_monotonic")
        if captured_at is not None:
            previous_captured_at = captured_at
        contract = row.get("contract")
        contract_mapping = contract if isinstance(contract, Mapping) else {}
        if not isinstance(contract, Mapping):
            reasons.add("contract_missing")
        else:
            required = {
                "route",
                "model",
                "request_model",
                "required_returned_model",
                "request_revision",
                "prompt_sha256",
                "schema",
                "contract_id",
                "expires_at",
                "schema_sha256",
                "route_sha256",
                "model_sha256",
                "cohort",
                "identity_revision",
                "fixed_identity",
                "free_only",
                "no_paid_fallback",
                "kill_categories",
                "live_recall_model_calls",
                "source_commit",
                "source_tree_sha256",
                "source_ox_identity_sha256",
            }
            if not required.issubset(contract):
                reasons.add("contract_identity_incomplete")
            if (
                contract.get("route") != OX_ROUTE
                or contract.get("model") != OX_MODEL
                or contract.get("request_model") != OX_MODEL
                or contract.get("required_returned_model") != OX_MODEL
                or contract.get("request_revision") != OX_REQUEST_REVISION
                or contract.get("schema") != OX_SCHEMA
                or contract.get("prompt_sha256") != OX_PROMPT_SHA256
                or contract.get("schema_sha256") != OX_SCHEMA_SHA256
                or contract.get("route_sha256") != OX_ROUTE_SHA256
                or contract.get("model_sha256") != OX_MODEL_SHA256
                or contract.get("cohort") != OX_COHORT
                or contract.get("identity_revision") != OX_IDENTITY_REVISION
                or contract.get("fixed_identity") != OX_FIXED_IDENTITY
                or contract.get("free_only") is not False
                or contract.get("no_paid_fallback") is not True
                or contract.get("kill_categories") != list(OX_KILL_CATEGORIES)
                or isinstance(contract.get("live_recall_model_calls"), bool)
                or contract.get("live_recall_model_calls") != 0
                or contract.get("source_commit") != source.get("commit")
                or contract.get("source_tree_sha256") != source.get("tree_sha256")
                or contract.get("source_ox_identity_sha256")
                != source.get("ox_identity_sha256")
            ):
                reasons.add("ox_identity_mismatch")
            if not _SHA.fullmatch(
                _text(contract.get("prompt_sha256"))
            ) or not _SHA.fullmatch(_text(contract.get("contract_id"))):
                reasons.add("ox_digest_invalid")
            contract_identity = {
                key: contract.get(key)
                for key in (
                    "route",
                    "model",
                    "prompt_sha256",
                    "schema",
                    "schema_sha256",
                    "route_sha256",
                    "model_sha256",
                    "cohort",
                    "identity_revision",
                    "request_revision",
                    "request_model",
                    "required_returned_model",
                    "fixed_identity",
                    "free_only",
                    "no_paid_fallback",
                    "kill_categories",
                    "live_recall_model_calls",
                    "source_commit",
                    "source_tree_sha256",
                    "source_ox_identity_sha256",
                    "expires_at",
                )
            }
            if contract.get("contract_id") != _sha256(contract_identity):
                reasons.add("ox_contract_digest_unbound")
            source_identity = _text(contract.get("source_ox_identity_sha256"))
            expected_source_identity = _text(source.get("ox_identity_sha256"))
            if expected_source_identity and source_identity != expected_source_identity:
                reasons.add("ox_source_identity_mismatch")
            canonical_expiry = _canonical_future_expiry(contract.get("expires_at"))
            if canonical_expiry is None or contract.get("expires_at") != canonical_expiry:
                reasons.add("ox_contract_expired_or_missing")
        control = row.get("control")
        if (
            not isinstance(control, Mapping)
            or control.get("ox_enabled") is not True
            or control.get("free_only") is not False
            or control.get("no_paid_fallback") is not True
            or control.get("kill_switch_supported") is not True
            or control.get("kill_switch_tripped") is not False
        ):
            reasons.add("ox_control_gate_invalid")
        if (
            row.get("source_commit") != source["commit"]
            or row.get("source_tree_sha256") != source["tree_sha256"]
        ):
            reasons.add("source_binding_mismatch")
        negative_veto = row.get("negative_veto")
        if (
            not isinstance(negative_veto, Mapping)
            or negative_veto.get("authenticated") is not True
            or negative_veto.get("exact_binding") is not True
            or negative_veto.get("conflicts") != 0
        ):
            reasons.add("negative_veto_gate_invalid")
        blind_repeat = row.get("blind_repeat")
        blind_pairs = (
            blind_repeat.get("pairs") if isinstance(blind_repeat, Mapping) else None
        )
        blind_pairs_valid = (
            isinstance(blind_pairs, int)
            and not isinstance(blind_pairs, bool)
            and blind_pairs >= OX_MIN_BLIND_REPEAT_PAIRS
        )
        if (
            not isinstance(blind_repeat, Mapping)
            or blind_repeat.get("revision") != OX_PROBE_REVISION
            or blind_repeat.get("complete") is not True
            or blind_repeat.get("stability_passed") is not True
            or not blind_pairs_valid
        ):
            reasons.add("blind_repeat_gate_invalid")
        order_swap = row.get("order_swap")
        swap_pairs = (
            order_swap.get("pairs") if isinstance(order_swap, Mapping) else None
        )
        swap_pairs_valid = (
            isinstance(swap_pairs, int)
            and not isinstance(swap_pairs, bool)
            and swap_pairs >= OX_MIN_BLIND_REPEAT_PAIRS
        )
        if (
            not isinstance(order_swap, Mapping)
            or order_swap.get("complete") is not True
            or not swap_pairs_valid
        ):
            reasons.add("order_swap_gate_invalid")
        rollback = row.get("rollback")
        if (
            not isinstance(rollback, Mapping)
            or rollback.get("verified") is not True
            or rollback.get("active_unchanged") is not True
            or rollback.get("status") not in {"not_rolled_back", "rolled_back"}
        ):
            reasons.add("rollback_gate_invalid")
        stage = row.get("stage")
        if isinstance(stage, Mapping):
            cap = stage.get("cap")
            if isinstance(cap, int) and not isinstance(cap, bool) and cap in OX_STAGES:
                if cap in stages:
                    reasons.add("ox_duplicate_stage")
                else:
                    expected_cap = (
                        OX_STAGES[len(stage_order)]
                        if len(stage_order) < len(OX_STAGES)
                        else None
                    )
                    if cap != expected_cap:
                        reasons.add("ox_ramp_order_invalid")
                    if stage_order:
                        prior = stages[stage_order[-1]]
                        prior_count = prior.get("valid_receipts")
                        prior_attempts = prior.get("attempts")
                        if (
                            isinstance(prior_count, bool)
                            or not isinstance(prior_count, int)
                            or isinstance(prior_attempts, bool)
                            or not isinstance(prior_attempts, int)
                            or prior_attempts < 1
                            or prior_count < 20
                            or prior_count / prior_attempts < 0.95
                        ):
                            reasons.add("ox_prior_stage_ineligible")
                    stage_order.append(cap)
                stages[cap] = dict(stage)
                count = _int(stage.get("valid_receipts"))
                attempts = _int(stage.get("attempts"), minimum=1)
                if count < 20 or count > attempts or count / attempts < 0.95:
                    reasons.add("ox_quality_floor_failed")
                labels = stage.get("labels")
                if not isinstance(labels, list) or len(labels) < 20:
                    reasons.add("ox_label_receipts_missing")
                else:
                    stage_label_ids = [
                        _text(item.get("label_id"))
                        for item in labels
                        if isinstance(item, Mapping)
                    ]
                    stage_commit_ids = [
                        _text(item.get("commit_id"))
                        for item in labels
                        if isinstance(item, Mapping)
                    ]
                    work_ids = [
                        _text(item.get("work_id"))
                        for item in labels
                        if isinstance(item, Mapping)
                    ]
                    provider_receipts = [
                        _text(item.get("provider_receipt_sha256"))
                        for item in labels
                        if isinstance(item, Mapping)
                    ]
                    if (
                        len(stage_label_ids) != len(labels)
                        or len(stage_commit_ids) != len(labels)
                        or len(work_ids) != len(labels)
                        or not all(stage_label_ids)
                        or not all(stage_commit_ids)
                        or not all(work_ids)
                        or len(set(stage_label_ids)) != len(stage_label_ids)
                        or len(set(stage_commit_ids)) != len(stage_commit_ids)
                        or len(set(work_ids)) != len(work_ids)
                        or len(provider_receipts) != len(labels)
                    ):
                        reasons.add("ox_label_identity_invalid")
                    if any(
                        not isinstance(item, Mapping)
                        or item.get("request_revision")
                        != contract_mapping.get("request_revision")
                        for item in labels
                    ):
                        reasons.add("ox_label_request_revision_invalid")
                    if any(
                        not isinstance(item, Mapping)
                        or item.get("expires_at") != contract_mapping.get("expires_at")
                        or _canonical_future_expiry(item.get("expires_at"))
                        != item.get("expires_at")
                        for item in labels
                    ):
                        reasons.add("ox_label_expiry_invalid")
                    if any(
                        not isinstance(item, Mapping)
                        or item.get("profile_contract_id")
                        != contract_mapping.get("contract_id")
                        or item.get("source_commit")
                        != contract_mapping.get("source_commit")
                        or item.get("source_tree_sha256")
                        != contract_mapping.get("source_tree_sha256")
                        or item.get("source_ox_identity_sha256")
                        != contract_mapping.get("source_ox_identity_sha256")
                        or _SHA.fullmatch(_text(item.get("payload_digest"))) is None
                        or not isinstance(item.get("payload_source"), Mapping)
                        or _sha256(item.get("payload_source"))
                        != item.get("payload_digest")
                        or item.get("work_id")
                        != _sha256(
                            {
                                "kind": "ox-teacher-label-v1",
                                "profile": OX_PROFILE,
                                "cohort": OX_COHORT,
                                "route": OX_ROUTE,
                                "profile_contract_id": contract_mapping.get(
                                    "contract_id"
                                ),
                                "payload_digest": item.get("payload_digest"),
                            }
                        )
                        or item.get("request_sha256")
                        != _expected_ox_request_sha256(
                            profile_contract_id=_text(
                                contract_mapping.get("contract_id")
                            ),
                            payload_digest=_text(item.get("payload_digest")),
                        )
                        or item.get("provider_request_sha256")
                        != _expected_ox_provider_request_sha256(
                            profile_contract_id=_text(
                                contract_mapping.get("contract_id")
                            ),
                            payload_digest=_text(item.get("payload_digest")),
                            work_id=_text(item.get("work_id")),
                            expires_at=_text(contract_mapping.get("expires_at")),
                        )
                        or _SHA.fullmatch(
                            _text(item.get("provider_receipt_sha256"))
                        )
                        is None
                        or item.get("provider_receipt_sha256")
                        == item.get("provider_request_sha256")
                        or "provider_response_request_sha256" in item
                        for item in labels
                    ):
                        reasons.add("ox_label_binding_invalid")
                    stage_provider_groups = set(provider_receipts)
                    if (
                        len(stage_provider_groups) != count
                        or bool(stage_provider_groups & all_provider_receipts)
                    ):
                        reasons.add("ox_label_count_mismatch")
                    else:
                        all_label_ids.extend(stage_label_ids)
                        all_commit_ids.extend(stage_commit_ids)
                        all_work_ids.extend(work_ids)
                        all_provider_receipts.update(stage_provider_groups)
            else:
                reasons.add("ox_stage_invalid")
        transitions = row.get("transition_receipts")
        if not isinstance(transitions, list):
            reasons.add("ox_transition_receipts_missing")
            transitions = []
        for transition in transitions:
            if not isinstance(transition, Mapping):
                reasons.add("ox_transition_receipt_invalid")
                continue
            category = _text(transition.get("category"))
            if category == "429":
                before_cap = transition.get("before_cap")
                after_cap = transition.get("after_cap")
                if (
                    isinstance(before_cap, bool)
                    or not isinstance(before_cap, int)
                    or before_cap < 1
                    or before_cap not in OX_STAGES
                    or isinstance(after_cap, bool)
                    or not isinstance(after_cap, int)
                    or after_cap < 1
                    or after_cap not in OX_STAGES
                    or after_cap != max(1, before_cap // 2)
                ):
                    reasons.add("429_halving_invalid")
                else:
                    failure_flags.add("429_halved")
            elif category in {"5xx", "timeout"}:
                retry_attempts = transition.get("attempts")
                attempts_valid = (
                    isinstance(retry_attempts, int)
                    and not isinstance(retry_attempts, bool)
                    and 1 <= retry_attempts <= 3
                )
                if (
                    not attempts_valid
                    or transition.get("bounded") is not True
                    or transition.get("status") != "deferred"
                ):
                    reasons.add("bounded_retry_invalid")
                else:
                    failure_flags.add(f"{category}_bounded_retry")
            elif category in {"402", "paid", "model_drift"}:
                if transition.get("status") != "hard_stop":
                    reasons.add("hard_stop_invalid")
                else:
                    failure_flags.add(f"{category}_hard_stop")
            else:
                reasons.add("unknown_failure_category")
        if identity is None:
            identity = dict(contract) if isinstance(contract, Mapping) else None
        elif identity != contract:
            reasons.add("ox_contract_mixing")
        for key in (
            "sensitive",
            "raw",
            "billable",
            "paid",
            "paid_calls",
            "unexpected_route",
            "unexpected_routes",
            "drift",
            "route_model_drift",
            "model_drift",
            "dup",
            "duplicate",
            "duplicate_label",
            "duplicate_commit",
        ):
            if row.get(key, 0) != 0:
                reasons.add(f"ox_{key}_nonzero")
        recovery = row.get("lease_recovery")
        if isinstance(recovery, Mapping):
            recovery_seen = True
            if recovery.get("leased_after") != 0:
                reasons.add("leased_work_not_recovered")
        label_id = _text(row.get("label_id"))
        commit_id = _text(row.get("commit_id"))
        if label_id:
            all_label_ids.append(label_id)
        if commit_id:
            all_commit_ids.append(commit_id)
    if len(all_label_ids) != len(set(all_label_ids)):
        reasons.add("ox_duplicate_label")
    if len(all_commit_ids) != len(set(all_commit_ids)):
        reasons.add("ox_duplicate_commit")
    if len(all_work_ids) != len(set(all_work_ids)):
        reasons.add("ox_duplicate_work")
    if not recovery_seen:
        reasons.add("lease_recovery_missing")
    if tuple(sorted(stages)) != OX_STAGES:
        reasons.add("ox_ramp_stages_incomplete")
    if stage_order != list(OX_STAGES):
        reasons.add("ox_ramp_order_invalid")
    if "429_halved" not in failure_flags:
        reasons.add("429_halving_missing")
    if not {"5xx_bounded_retry", "timeout_bounded_retry"}.issubset(failure_flags):
        reasons.add("bounded_retry_missing")
    if not {"402_hard_stop", "paid_hard_stop", "model_drift_hard_stop"}.issubset(
        failure_flags
    ):
        reasons.add("hard_stop_coverage_missing")
    return {
        "profile": OX_PROFILE,
        "passed": not reasons,
        "reasons": sorted(reasons),
        "rows": len(rows),
        "contract": identity,
        "stages": {str(cap): stages[cap] for cap in sorted(stages)},
        "failure_receipts": sorted(failure_flags),
    }


def _production_stat(path: Path, *, label: str) -> dict[str, int]:
    """Capture one production file without following symlinks."""

    if _has_symlink_component(path):
        raise R4Error(f"{label} path contains a symlink")
    try:
        value = path.lstat()
    except OSError as exc:
        raise R4Error(f"{label} path is unavailable") from exc
    if stat.S_ISLNK(value.st_mode) or not stat.S_ISREG(value.st_mode):
        raise R4Error(f"{label} path is not a regular file")
    return {
        "st_dev": int(value.st_dev),
        "st_ino": int(value.st_ino),
        "st_mode": int(value.st_mode & 0o7777),
        "st_size": int(value.st_size),
        "st_mtime_ns": int(value.st_mtime_ns),
        "st_ctime_ns": int(value.st_ctime_ns),
    }


def _production_directory_identity(path: Path, *, label: str) -> dict[str, int]:
    """Capture a production directory's device/inode without following links."""

    if _has_symlink_component(path):
        raise R4Error(f"{label} path contains a symlink")
    try:
        value = path.lstat()
    except OSError as exc:
        raise R4Error(f"{label} path is unavailable") from exc
    if stat.S_ISLNK(value.st_mode) or not stat.S_ISDIR(value.st_mode):
        raise R4Error(f"{label} path is not a directory")
    return {
        "st_dev": int(value.st_dev),
        "st_ino": int(value.st_ino),
    }


def _production_directory_fd_identity(fd: int, *, label: str) -> dict[str, int]:
    try:
        value = os.fstat(fd)
    except OSError as exc:
        raise R4Error(f"{label} descriptor is unavailable") from exc
    if not stat.S_ISDIR(value.st_mode):
        raise R4Error(f"{label} descriptor is not a directory")
    return {"st_dev": int(value.st_dev), "st_ino": int(value.st_ino)}


def _production_file_bytes(
    path: Path, *, label: str, max_bytes: int
) -> tuple[bytes, dict[str, int], str]:
    """Read a bounded production file and re-stat it before returning."""

    before = _production_stat(path, label=label)
    if before["st_size"] > max_bytes:
        raise R4Error(f"{label} file exceeds bounded size")
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise R4Error(f"{label} file read failed") from exc
    after = _production_stat(path, label=label)
    if before != after or len(data) != before["st_size"]:
        raise R4Error(f"{label} changed during capture")
    return data, after, hashlib.sha256(data).hexdigest()


def _production_json(
    path: Path,
    *,
    label: str,
    schema: str | None = None,
    max_bytes: int = _MAX_RECEIPT_BYTES,
) -> tuple[dict[str, Any], dict[str, int], str]:
    data, state, digest = _production_file_bytes(path, label=label, max_bytes=max_bytes)
    try:
        payload = json.loads(data)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise R4Error(f"{label} JSON is invalid") from exc
    if not isinstance(payload, dict):
        raise R4Error(f"{label} JSON is not an object")
    if payload.get("namespace") != "recall-distillation":
        raise R4Error(f"{label} namespace is invalid")
    if schema is not None and payload.get("schema") != schema:
        raise R4Error(f"{label} schema is invalid")
    unsigned = {key: value for key, value in payload.items() if key != "seal_sha256"}
    if payload.get("seal_sha256") != _sha256(unsigned):
        raise R4Error(f"{label} seal mismatch")
    canonical = _json_bytes(payload)
    # Managed immutable JSON files are written with one record-delimiting
    # newline.  Accept that exact suffix, but reject every other whitespace
    # or re-serialization variant so a resealed/truncated file cannot pass.
    if data not in {canonical, canonical + b"\n"}:
        raise R4Error(f"{label} JSON is not canonical")
    final = _production_stat(path, label=label)
    if final != state:
        raise R4Error(f"{label} changed after validation")
    return payload, state, digest


def _production_historical_profile_contract(
    path: Path, *, contract_id: str
) -> tuple[dict[str, Any], dict[str, int], str]:
    contract, state, digest = _production_json(
        path, label="historical production profile contract"
    )
    if contract.get("schema") not in {OX_PROFILE_SCHEMA, OX_LEGACY_PROFILE_SCHEMA}:
        raise R4Error("historical production profile contract schema is invalid")
    unsigned = {
        key: value
        for key, value in contract.items()
        if key not in {"artifact_id", "seal_sha256"}
    }
    if contract.get("artifact_id") != contract_id or contract_id != _sha256(unsigned):
        raise R4Error("historical production profile contract identity mismatch")
    return contract, state, digest


def _production_pre_profile_local_label(row: Mapping[str, Any]) -> bool:
    """Recognize the exact local label shape written before profiles existed."""

    route_identity = row.get("route_identity")
    local_identity = (
        isinstance(route_identity, Mapping)
        and set(route_identity) == {"location", "model", "provider", "role"}
        and route_identity.get("location") == "local"
        and route_identity.get("role") == row.get("route")
        and all(route_identity.get(key) for key in ("model", "provider"))
    )
    return (
        row.get("profile_contract_id") is None
        and row.get("kind") in {"teacher-label", "counterfactual-label"}
        and row.get("route")
        in {
            "recall.distill.teacher.a",
            "recall.distill.teacher.b",
            "recall.distill.teacher.c",
            "counterfactual",
        }
        and all(
            row.get(key) is None
            for key in (
                "profile",
                "cohort",
                "teacher_role",
                "source_commit",
            )
        )
        and (route_identity in (None, {}) or local_identity)
    )


def _production_companion_state(path: Path, *, label: str) -> dict[str, int] | None:
    try:
        value = path.lstat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise R4Error(f"{label} path is unavailable") from exc
    if stat.S_ISLNK(value.st_mode):
        raise R4Error(f"{label} path contains a symlink")
    if not stat.S_ISREG(value.st_mode):
        raise R4Error(f"{label} path is not a regular file")
    return _production_stat(path, label=label)


def _production_sqlite_state(path: Path, *, label: str) -> dict[str, Any]:
    """Capture sidecar identities for a read-only SQLite observation."""

    return {
        "main": _production_stat(path, label=label),
        "wal": _production_companion_state(
            path.with_name(f"{path.name}-wal"), label=f"{label}-wal"
        ),
        "shm": _production_companion_state(
            path.with_name(f"{path.name}-shm"), label=f"{label}-shm"
        ),
    }


def _production_sqlite_unchanged(
    path: Path, before: Mapping[str, Any], *, label: str
) -> bool:
    return _production_sqlite_state(path, label=label) == dict(before)


def _counts(value: object, *, label: str) -> dict[str, int]:
    if not isinstance(value, Mapping) or set(value) != set(PRODUCTION_WORKSET_STATES):
        raise R4Error(f"{label} counts are invalid")
    result: dict[str, int] = {}
    for state in PRODUCTION_WORKSET_STATES:
        count = value[state]
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise R4Error(f"{label} counts are invalid")
        result[state] = count
    return result


def _production_workset_receipts(
    connection: sqlite3.Connection,
    current_counts: Mapping[str, int],
    current_watermark: object,
    current_progress: object,
) -> dict[str, Any]:
    try:
        rows = connection.execute(
            "SELECT generation, previous_sha256, operation, payload_json, "
            "receipt_sha256 FROM workset_receipts ORDER BY generation ASC"
        ).fetchall()
    except sqlite3.Error as exc:
        raise R4Error("production workset receipt read failed") from exc
    if not rows:
        raise R4Error("production workset receipts are empty")
    if len(rows) > PRODUCTION_MAX_ROWS:
        raise R4Error("production workset receipt count is unbounded")
    previous = ""
    prior_after: dict[str, Any] | None = None
    saw_v2 = False
    legacy_unverified = False
    receipt_by_generation: dict[str, dict[str, Any]] = {}
    for expected_generation, row in enumerate(rows, start=1):
        generation, previous_sha, operation, payload_json, receipt_sha = row
        if (
            isinstance(generation, bool)
            or not isinstance(generation, int)
            or generation != expected_generation
            or previous_sha != previous
            or operation not in PRODUCTION_WORKSET_OPERATIONS
            or not isinstance(payload_json, str)
            or not isinstance(receipt_sha, str)
            or _SHA.fullmatch(receipt_sha) is None
        ):
            raise R4Error("production workset receipt chain is corrupted")
        try:
            payload = json.loads(payload_json)
        except (ValueError, UnicodeError, RecursionError) as exc:
            raise R4Error("production workset receipt JSON is invalid") from exc
        if (
            not isinstance(payload, Mapping)
            or _json_bytes(payload).decode() != payload_json
        ):
            raise R4Error("production workset receipt JSON is not canonical")
        before = payload.get("before")
        after = payload.get("after")
        delta = payload.get("delta")
        version = payload.get("version", 1)
        allowed_payload = {"before", "after", "delta", "details"}
        if version == 2:
            allowed_payload.add("version")
            if payload.get("bootstrap") is True:
                allowed_payload.add("bootstrap")
        if version not in {1, 2} or set(payload) != allowed_payload:
            raise R4Error("production workset receipt schema is invalid")
        if version == 1:
            # v1 remains readable for historical inspection, but it has no
            # durable progress/work-id boundary and cannot certify R4.
            legacy_unverified = True
        if not isinstance(before, Mapping) or not isinstance(after, Mapping):
            raise R4Error("production workset receipt snapshot is invalid")
        expected_snapshot_keys = {"counts", "watermark"}
        if version == 2:
            expected_snapshot_keys.add("progress")
        if (
            set(before) != expected_snapshot_keys
            or set(after) != expected_snapshot_keys
            or (
                version == 2
                and payload.get("bootstrap") is True
                and before["progress"] is not None
            )
        ):
            raise R4Error("production workset receipt snapshot is invalid")
        if saw_v2 and version == 1:
            raise R4Error("production workset receipt progress downgraded")
        if version == 2:
            if (
                not isinstance(before.get("progress"), Mapping)
                and before.get("progress") is not None
            ) or (
                not isinstance(after.get("progress"), Mapping)
                and after.get("progress") is not None
            ):
                raise R4Error("production workset receipt progress is invalid")
            if prior_after is not None and _json_bytes(
                before.get("progress")
            ) != _json_bytes(prior_after.get("progress")):
                raise R4Error("production workset receipt progress continuity failed")
            saw_v2 = True
        before_counts = _counts(before.get("counts"), label="receipt before")
        after_counts = _counts(after.get("counts"), label="receipt after")
        if not isinstance(delta, Mapping) or set(delta) != set(
            PRODUCTION_WORKSET_STATES
        ):
            raise R4Error("production workset receipt delta is invalid")
        for state in PRODUCTION_WORKSET_STATES:
            value = delta[state]
            if isinstance(value, bool) or not isinstance(value, int):
                raise R4Error("production workset receipt delta is invalid")
            if value != after_counts[state] - before_counts[state]:
                raise R4Error("production workset receipt continuity failed")
        details = payload.get("details")
        if not isinstance(details, Mapping):
            raise R4Error("production workset receipt details are invalid")
        if operation == "advance":
            expected_details = {
                "inserted",
                "rebound",
                "watermark_changed",
                "selection_sha256",
            }
            if version == 2:
                expected_details.add("progress_changed")
            if set(details) != expected_details:
                raise R4Error("production workset advance receipt is invalid")
            if (
                isinstance(details["inserted"], bool)
                or not isinstance(details["inserted"], int)
                or details["inserted"] < 0
                or isinstance(details["rebound"], bool)
                or not isinstance(details["rebound"], int)
                or details["rebound"] < 0
                or not isinstance(details["watermark_changed"], bool)
                or not isinstance(details.get("progress_changed", False), bool)
                or _SHA.fullmatch(str(details["selection_sha256"])) is None
            ):
                raise R4Error("production workset advance receipt is invalid")
            expected_delta = {
                "ready": details["inserted"],
                "leased": 0,
                "completed": 0,
                "quarantined": 0,
            }
            if dict(delta) != expected_delta:
                raise R4Error("production workset advance delta is invalid")
        elif operation in {"claim_reclaim", "claim", "release"}:
            expected_details = {"kind", "count", "selection_sha256"}
            work_bound_details = {*expected_details, "work_ids_sha256"}
            allowed_details = (
                (work_bound_details,)
                if version == 2
                else (expected_details, work_bound_details)
            )
            if set(details) not in allowed_details:
                raise R4Error("production workset lease receipt is invalid")
            kind = details["kind"]
            count = details["count"]
            if (
                not isinstance(kind, str)
                or not kind
                or isinstance(count, bool)
                or not isinstance(count, int)
                or count < 1
                or _SHA.fullmatch(str(details["selection_sha256"])) is None
                or (
                    "work_ids_sha256" in details
                    and _SHA.fullmatch(str(details["work_ids_sha256"])) is None
                )
            ):
                raise R4Error("production workset lease receipt is invalid")
            expected_delta = {
                "ready": count if operation in {"claim_reclaim", "release"} else -count,
                "leased": -count
                if operation in {"claim_reclaim", "release"}
                else count,
                "completed": 0,
                "quarantined": 0,
            }
            if dict(delta) != expected_delta:
                raise R4Error("production workset lease delta is invalid")
        else:
            expected_details = {"completed", "retry", "quarantined", "selection_sha256"}
            timed_details = expected_details | {"retry_wait", "retry_schedule_sha256"}
            work_bound_details = expected_details | {"work_ids_sha256"}
            timed_work_bound_details = timed_details | {"work_ids_sha256"}
            if set(details) not in (
                expected_details,
                timed_details,
                work_bound_details,
                timed_work_bound_details,
            ):
                raise R4Error("production workset commit receipt is invalid")
            totals = {
                key: details[key] for key in ("completed", "retry", "quarantined")
            }
            if (
                any(
                    isinstance(value, bool) or not isinstance(value, int) or value < 0
                    for value in totals.values()
                )
                or sum(totals.values()) < 1
                or _SHA.fullmatch(str(details["selection_sha256"])) is None
                or (
                    "work_ids_sha256" in details
                    and _SHA.fullmatch(str(details["work_ids_sha256"])) is None
                )
            ):
                raise R4Error("production workset commit receipt is invalid")
            if set(details) in (timed_details, timed_work_bound_details) and (
                isinstance(details["retry_wait"], bool)
                or not isinstance(details["retry_wait"], int)
                or details["retry_wait"] < 0
                or details["retry_wait"] > details["retry"]
                or _SHA.fullmatch(str(details["retry_schedule_sha256"])) is None
            ):
                raise R4Error("production workset commit receipt is invalid")
            expected_delta = {
                "ready": totals["retry"],
                "leased": -sum(totals.values()),
                "completed": totals["completed"],
                "quarantined": totals["quarantined"],
            }
            if dict(delta) != expected_delta:
                raise R4Error("production workset commit delta is invalid")
        if prior_after is not None and (
            before_counts != prior_after["counts"]
            or before.get("watermark") != prior_after["watermark"]
        ):
            raise R4Error("production workset receipt continuity failed")
        envelope = {
            "generation": generation,
            "previous_sha256": previous,
            "operation": operation,
            "payload": payload,
        }
        if _sha256(envelope) != receipt_sha:
            raise R4Error("production workset receipt hash mismatch")
        receipt_by_generation[str(generation)] = {
            "receipt_sha256": receipt_sha,
            "operation": operation,
            "details": dict(details),
        }
        previous = receipt_sha
        prior_after = {
            "counts": after_counts,
            "watermark": after.get("watermark"),
            "progress": after.get("progress") if version == 2 else None,
        }
    if prior_after is None or prior_after["counts"] != dict(current_counts):
        raise R4Error("production workset receipt final state mismatch")
    if prior_after["watermark"] != current_watermark:
        raise R4Error("production workset receipt final watermark mismatch")
    if saw_v2 and _json_bytes(prior_after["progress"]) != _json_bytes(current_progress):
        raise R4Error("production workset receipt final progress mismatch")
    return {
        "count": len(rows),
        "generation": len(rows),
        "head_sha256": previous,
        "by_generation": receipt_by_generation,
        "verified": True,
        "status": "legacy-unverified" if legacy_unverified else "verified",
        "legacy_unverified_excluded": legacy_unverified,
    }


def _valid_ox_workset_provenance(
    provenance: Mapping[str, Any], contract_id: str
) -> bool:
    base = {
        "profile",
        "cohort",
        "route",
        "teacher_role",
        "profile_contract_id",
        "probe",
    }
    probe = {
        "probe_revision",
        "repeat_pair_id",
        "fixed_repeat",
        "order_swap",
        "blind_order",
        "probe_batch_id",
        "order_variant",
        "candidate_position",
    }
    if (
        provenance.get("profile") != OX_PROFILE
        or provenance.get("cohort") != OX_COHORT
        or provenance.get("route") != OX_ROUTE
        or provenance.get("teacher_role") != "recall.distill.teacher.deepseek-v4-flash"
        or provenance.get("profile_contract_id") != contract_id
        or type(provenance.get("probe")) is not bool
    ):
        return False
    if provenance["probe"] is False:
        return set(provenance) == base
    return (
        set(provenance) == base | probe
        and provenance.get("probe_revision") == OX_PROBE_REVISION
        and _SHA.fullmatch(str(provenance.get("repeat_pair_id"))) is not None
        and provenance.get("fixed_repeat") is True
        and provenance.get("order_swap") is True
        and provenance.get("blind_order") in {"a_first", "b_first"}
        and _SHA.fullmatch(str(provenance.get("probe_batch_id"))) is not None
        and type(provenance.get("order_variant")) is int
        and provenance["order_variant"] in {1, 2}
        and type(provenance.get("candidate_position")) is int
        and provenance["candidate_position"] in {0, 1}
    )


def _production_workset(path: Path) -> dict[str, Any]:
    """Read and verify the managed SQLite workset without opening a writer."""

    before_files = _production_sqlite_state(path, label="production workset")
    if before_files["main"]["st_size"] > PRODUCTION_MAX_SQLITE_BYTES:
        raise R4Error("production workset exceeds bounded size")
    try:
        # SQLite's mode=ro still updates WAL shared-memory read marks.  A
        # checkpointed database is safe to open immutable; a non-empty WAL is
        # not a frozen production snapshot and must fail closed.
        wal = before_files["wal"]
        if wal is not None and wal["st_size"] > 0:
            raise R4Error("production workset has uncheckpointed WAL")
        source_connection = sqlite3.connect(f"file:{path}?immutable=1", uri=True)
        try:
            connection = sqlite3.connect(":memory:")
            source_connection.backup(connection)
        finally:
            source_connection.close()
    except sqlite3.Error as exc:
        raise R4Error("production workset cannot be opened read-only") from exc
    try:
        connection.execute("PRAGMA query_only = ON")
        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        if not {"work_items", "workset_state", "workset_receipts"}.issubset(tables):
            raise R4Error("production workset schema is incomplete")
        columns = {
            str(row[1]) for row in connection.execute("PRAGMA table_info(work_items)")
        }
        required_columns = {
            "sequence",
            "work_id",
            "kind",
            "payload_ref",
            "payload_digest",
            "lease_id",
            "lease_owner",
            "lease_expires_at",
            "temporal_split_json",
            "provenance_json",
            "state",
            "attempt_count",
            "completion_ref",
            "completion_digest",
        }
        if not required_columns.issubset(columns):
            raise R4Error("production workset columns are incomplete")
        row_count = int(
            connection.execute("SELECT COUNT(*) FROM work_items").fetchone()[0]
        )
        if row_count < 1 or row_count > PRODUCTION_MAX_ROWS:
            raise R4Error("production workset row count is outside bounds")
        rows = connection.execute(
            "SELECT sequence, work_id, kind, payload_ref, payload_digest, "
            "temporal_split_json, provenance_json, stage, state, attempt_count, "
            "completion_ref, completion_digest, lease_id, lease_owner, lease_expires_at "
            "FROM work_items ORDER BY sequence"
        ).fetchall()
        if len(rows) != row_count:
            raise R4Error("production workset row count changed during read")
        work_ids: set[str] = set()
        completed: dict[str, dict[str, Any]] = {}
        items: dict[str, dict[str, Any]] = {}
        counts = {state: 0 for state in PRODUCTION_WORKSET_STATES}
        provenance_identity: dict[str, Any] | None = None
        for expected_sequence, row in enumerate(rows, start=1):
            (
                sequence,
                work_id,
                kind,
                payload_ref,
                payload_digest,
                temporal_json,
                provenance_json,
                stage,
                state,
                attempt_count,
                completion_ref,
                completion_digest,
                lease_id,
                lease_owner,
                lease_expires_at,
            ) = row
            if (
                sequence != expected_sequence
                or not isinstance(work_id, str)
                or _SHA.fullmatch(work_id) is None
                or work_id in work_ids
                or not isinstance(kind, str)
                or kind != "ox"
                or not isinstance(payload_ref, str)
                or not payload_ref.startswith("candidate-")
                or _SHA.fullmatch(str(payload_digest)) is None
                or stage != "teacher"
                or state not in PRODUCTION_WORKSET_STATES
                or isinstance(attempt_count, bool)
                or not isinstance(attempt_count, int)
                or attempt_count < 0
                or (
                    state != "leased"
                    and (lease_id is not None or lease_owner is not None or lease_expires_at is not None)
                )
                or (
                    state == "leased"
                    and (
                        not isinstance(lease_id, str)
                        or not lease_id
                        or not isinstance(lease_owner, str)
                        or not lease_owner
                        or not isinstance(lease_expires_at, (int, float))
                        or lease_expires_at <= 0
                    )
                )
            ):
                raise R4Error("production workset item identity is invalid")
            try:
                temporal = json.loads(temporal_json)
                provenance = json.loads(provenance_json)
            except (ValueError, TypeError, UnicodeError, RecursionError) as exc:
                raise R4Error("production workset metadata is invalid") from exc
            if (
                not isinstance(temporal, Mapping)
                or not isinstance(provenance, Mapping)
                or _json_bytes(temporal).decode() != temporal_json
                or _json_bytes(provenance).decode() != provenance_json
                or (state == "completed" and attempt_count < 1)
            ):
                raise R4Error("production workset metadata is invalid")
            if (
                set(temporal)
                != {"as_of", "group_id", "split", "split_plan_id"}
                or any(
                    not isinstance(temporal.get(key), str) or not temporal[key]
                    for key in ("as_of", "group_id")
                )
                or not isinstance(temporal.get("split_plan_id"), str)
                or temporal.get("split") not in {"train", "validation", "test", "embargo"}
            ):
                raise R4Error("production workset temporal split is invalid")
            expected_provenance = {
                key: provenance.get(key)
                for key in ("cohort", "profile", "profile_contract_id", "route")
            }
            if (
                not _valid_ox_workset_provenance(
                    provenance, str(expected_provenance["profile_contract_id"])
                )
                or _SHA.fullmatch(str(expected_provenance["profile_contract_id"]))
                is None
            ):
                raise R4Error("production workset provenance is invalid")
            if provenance_identity is None:
                provenance_identity = dict(expected_provenance)
            elif provenance_identity != expected_provenance:
                raise R4Error("production workset provenance is mixed")
            expected_work_id = _sha256(
                {
                    "kind": "ox-teacher-label-v1",
                    "profile": OX_PROFILE,
                    "cohort": OX_COHORT,
                    "route": OX_ROUTE,
                    "profile_contract_id": expected_provenance["profile_contract_id"],
                    "payload_digest": str(payload_digest),
                }
            )
            if work_id != expected_work_id:
                raise R4Error("production workset payload digest is unbound")
            parts = payload_ref.split(":")
            if (
                len(parts) != 3
                or parts[0] != "candidate-snapshot"
                or not parts[1]
                or not parts[2]
            ):
                raise R4Error("production workset payload reference is invalid")
            work_ids.add(work_id)
            items[work_id] = {
                "payload_ref": payload_ref,
                "payload_digest": str(payload_digest),
                "temporal_split": dict(temporal),
                "provenance": dict(expected_provenance),
            }
            counts[state] += 1
            if state == "completed":
                if (
                    not isinstance(completion_ref, str)
                    or not completion_ref.startswith("label-ledger:")
                    or _SHA.fullmatch(str(completion_digest)) is None
                    or completion_ref.removeprefix("label-ledger:") != completion_digest
                ):
                    raise R4Error("production completed work lacks a sealed label ref")
                completed[work_id] = {
                    "work_id": work_id,
                    "completion_ref": completion_ref,
                    "completion_digest": str(completion_digest),
                    "attempt_count": attempt_count,
                    "provenance": dict(provenance),
                }
        state_rows = connection.execute(
            "SELECT key, value_json FROM workset_state"
        ).fetchall()
        state_values: dict[str, Any] = {}
        for key, value_json in state_rows:
            if not isinstance(key, str) or key in state_values:
                raise R4Error("production workset state is invalid")
            try:
                value = json.loads(value_json)
            except (ValueError, TypeError, UnicodeError, RecursionError) as exc:
                raise R4Error("production workset state JSON is invalid") from exc
            if (
                not isinstance(value_json, str)
                or _json_bytes(value).decode() != value_json
            ):
                raise R4Error("production workset state JSON is not canonical")
            state_values[key] = value
        if "watermark" not in state_values:
            raise R4Error("production workset watermark is missing")
        watermark = state_values.get("watermark")
        if (
            not isinstance(watermark, Mapping)
            or set(watermark)
            != {"candidate_records", "candidate_head", "split_plan_id", "probe_revision"}
            or isinstance(watermark.get("candidate_records"), bool)
            or not isinstance(watermark.get("candidate_records"), int)
            or watermark.get("candidate_records", 0) < 1
            or _SHA.fullmatch(str(watermark.get("candidate_head"))) is None
            or not isinstance(watermark.get("split_plan_id"), str)
            or watermark.get("probe_revision") != OX_PROBE_REVISION
        ):
            raise R4Error("production workset watermark is invalid")
        receipt = _production_workset_receipts(
            connection, counts, watermark, state_values.get("progress")
        )
    except sqlite3.Error as exc:
        raise R4Error("production workset read failed") from exc
    finally:
        connection.close()
    if not _production_sqlite_unchanged(path, before_files, label="production workset"):
        raise R4Error("production workset changed during validation")
    digest = _file_sha256(path)
    if not _production_sqlite_unchanged(path, before_files, label="production workset"):
        raise R4Error("production workset changed during hashing")
    return {
        "rows": row_count,
        "counts": counts,
        "completed": completed,
        "items": items,
        "provenance": provenance_identity or {},
        "watermark": watermark,
        "receipts": receipt,
        "file_state": before_files,
        "sha256": digest,
    }


def _production_ledger_checkpoint(
    path: Path, checkpoint_path: Path, *, ledger_name: str
) -> dict[str, Any]:
    checkpoint, checkpoint_state, checkpoint_sha = _production_json(
        checkpoint_path,
        label=f"production {ledger_name} checkpoint",
        schema="chronovisor.recall-distillation.v1",
    )
    if (
        checkpoint.get("kind") != "ledger-chain-checkpoint"
        or checkpoint.get("ledger_name") != ledger_name
        or isinstance(checkpoint.get("records"), bool)
        or not isinstance(checkpoint.get("records"), int)
        or checkpoint.get("records", 0) < 1
        or _SHA.fullmatch(str(checkpoint.get("head_sha256"))) is None
    ):
        raise R4Error(f"production {ledger_name} checkpoint identity is invalid")
    file_state = checkpoint.get("file_state")
    if not isinstance(file_state, Mapping):
        raise R4Error(f"production {ledger_name} checkpoint file state is missing")
    current = _production_stat(path, label=f"production {ledger_name}")
    if current["st_size"] > PRODUCTION_MAX_LEDGER_BYTES:
        raise R4Error(f"production {ledger_name} exceeds bounded size")
    expected = {
        "size_bytes": current["st_size"],
        "st_dev": current["st_dev"],
        "st_ino": current["st_ino"],
        "st_mtime_ns": current["st_mtime_ns"],
        "st_ctime_ns": current["st_ctime_ns"],
    }
    if dict(file_state) != expected:
        raise R4Error(f"production {ledger_name} checkpoint file state mismatch")
    if (
        _production_stat(checkpoint_path, label=f"production {ledger_name} checkpoint")
        != checkpoint_state
    ):
        raise R4Error(f"production {ledger_name} checkpoint changed during validation")
    return {
        "records": int(checkpoint["records"]),
        "head_sha256": str(checkpoint["head_sha256"]),
        "file_state": dict(file_state),
        "ledger_state": current,
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_state": checkpoint_state,
    }


def _production_workset_candidate_binding(
    workset: Mapping[str, Any],
    candidate: Mapping[str, Any],
    candidate_path: Path,
) -> list[dict[str, Any]]:
    """Stream every workset reference from the sealed candidate generation."""

    watermark = workset.get("watermark")
    items = workset.get("items")
    completed = workset.get("completed")
    if (
        not isinstance(watermark, Mapping)
        or not isinstance(items, Mapping)
        or not isinstance(completed, Mapping)
        or watermark.get("candidate_records") != candidate.get("records")
        or watermark.get("candidate_head") != candidate.get("head_sha256")
    ):
        raise R4Error("production workset candidate watermark is unbound")
    references: dict[str, set[str]] = {}
    completed_rallies: set[str] = set()
    for work_id, item in items.items():
        if _SHA.fullmatch(str(work_id)) is None or not isinstance(item, Mapping):
            raise R4Error("production workset item inventory is invalid")
        payload_ref = item.get("payload_ref")
        if not isinstance(payload_ref, str):
            raise R4Error("production workset payload reference is invalid")
        parts = payload_ref.split(":")
        if (
            len(parts) != 3
            or parts[0] != "candidate-snapshot"
            or not parts[1]
            or not parts[2]
        ):
            raise R4Error("production workset payload reference is invalid")
        references.setdefault(parts[1], set()).add(parts[2])
        if work_id in completed:
            completed_rallies.add(parts[1])
    if not references:
        raise R4Error("production workset has no candidate references")

    before = _production_stat(candidate_path, label="production candidate ledger")
    if candidate.get("ledger_state") != before:
        raise R4Error("production candidate ledger changed before workset binding")
    resolved: set[str] = set()
    completed_rows: list[dict[str, Any]] = []
    records = 0
    head = ""
    try:
        with candidate_path.open("rb") as handle:
            for records, raw in enumerate(handle, start=1):
                if records > PRODUCTION_MAX_ROWS:
                    raise R4Error("production candidate ledger is unbounded")
                if not raw.endswith(b"\n") or raw == b"\n":
                    raise R4Error("production candidate ledger is truncated")
                try:
                    record = json.loads(raw[:-1])
                except (ValueError, UnicodeError, RecursionError) as exc:
                    raise R4Error("production candidate ledger JSON is invalid") from exc
                if not isinstance(record, dict):
                    raise R4Error("production candidate ledger row is invalid")
                head = str(record.get("record_sha256") or "")
                rally_id = record.get("rally_id")
                if not isinstance(rally_id, str) or not rally_id:
                    raise R4Error("production candidate ledger snapshot is invalid")
                if rally_id not in references:
                    continue
                if rally_id in resolved:
                    raise R4Error("production candidate ledger snapshot is invalid")
                snapshot = record.get("snapshot")
                candidates = (
                    snapshot.get("candidates")
                    if isinstance(snapshot, Mapping)
                    else None
                )
                if (
                    not isinstance(snapshot, Mapping)
                    or snapshot.get("rally_id") != rally_id
                    or not isinstance(candidates, list)
                    or any(
                        sum(
                            isinstance(value, Mapping)
                            and value.get("candidate_id") == candidate_id
                            for value in candidates
                        )
                        != 1
                        for candidate_id in references[rally_id]
                    )
                ):
                    raise R4Error(
                        "production workset payload does not resolve in candidate ledger"
                    )
                resolved.add(rally_id)
                if rally_id in completed_rallies:
                    completed_rows.append(record)
    except OSError as exc:
        raise R4Error("production candidate ledger read failed") from exc
    if (
        records != candidate.get("records")
        or head != candidate.get("head_sha256")
        or resolved != set(references)
        or len(completed_rows) != len(completed_rallies)
    ):
        raise R4Error("production workset payload does not resolve in candidate ledger")
    if _production_stat(candidate_path, label="production candidate ledger") != before:
        raise R4Error("production candidate ledger changed during workset binding")
    return completed_rows


def _production_critical_module_sha256(source_root: Path) -> dict[str, str]:
    """Compare installed module bytes with the exact audited checkout."""

    modules = {
        "recall_distillation": "chronovisor.recall.recall_distillation",
        "remote_teacher": "chronovisor.recall.recall_distillation_remote_teacher",
        "workset": "chronovisor.recall.recall_distillation_workset",
        "runtime_config": "chronovisor.core.runtime_config",
    }
    result: dict[str, str] = {}
    for label, name in modules.items():
        source = source_root / "src" / Path(*name.split(".")).with_suffix(".py")
        try:
            spec = importlib.util.find_spec(name)
            installed = Path(str(spec.origin)).resolve(strict=True) if spec else None
            if installed is None or _has_symlink_component(installed):
                return {}
            source_bytes = source.read_bytes()
            installed_bytes = installed.read_bytes()
        except (ImportError, OSError, TypeError, ValueError):
            return {}
        if source_bytes != installed_bytes:
            return {}
        result[label] = hashlib.sha256(source_bytes).hexdigest()
    return result


def _load_production_anchor(
    production_root: Path, *, source: Mapping[str, Any]
) -> dict[str, Any]:
    """Read the explicitly bootstrapped managed R4 candidate anchor.

    This intentionally never falls back to ``_handoff`` or creates an anchor:
    a state reseal cannot turn checkout data into production authority.
    """

    path = production_root / PRODUCTION_CANDIDATE_ANCHOR_RELATIVE
    payload, state, file_sha256 = _production_json(
        path, label="production R4 candidate anchor", schema=R4_CANDIDATE_ANCHOR_SCHEMA
    )
    allowed = {
        "schema",
        "namespace",
        "kind",
        "artifact_id",
        "seal_sha256",
        "r0_artifact_id",
        "r0_file_sha256",
        "bootstrap_source_commit",
        "candidate_checkpoint",
        "critical_module_sha256",
    }
    if set(payload) != allowed or payload.get("kind") != "r4-candidate-anchor":
        raise R4Error("production R4 candidate anchor schema is invalid")
    artifact_id = payload.get("artifact_id")
    unsigned = {
        key: value
        for key, value in payload.items()
        if key in allowed - {"artifact_id", "seal_sha256"}
    }
    candidate = payload.get("candidate_checkpoint")
    if (
        not isinstance(artifact_id, str)
        or _SHA.fullmatch(artifact_id) is None
        or artifact_id
        != _sha256(
            {
                "schema": R4_CANDIDATE_ANCHOR_SCHEMA,
                "namespace": "recall-distillation",
                **unsigned,
            }
        )
        or payload.get("r0_artifact_id") != R0_EVIDENCE_ID
        or not isinstance(payload.get("r0_file_sha256"), str)
        or _SHA.fullmatch(str(payload.get("r0_file_sha256"))) is None
        or payload.get("bootstrap_source_commit") != source.get("commit")
        or not isinstance(candidate, Mapping)
        or not isinstance(payload.get("critical_module_sha256"), Mapping)
        or set(payload["critical_module_sha256"])
        != {"recall_distillation", "remote_teacher", "workset", "runtime_config"}
        or any(
            _SHA.fullmatch(str(value)) is None
            for value in payload["critical_module_sha256"].values()
        )
    ):
        raise R4Error("production R4 candidate anchor identity is invalid")
    head, records, bytes_value, file_state = (
        candidate.get("head_sha256"),
        candidate.get("records"),
        candidate.get("bytes"),
        candidate.get("file_state"),
    )
    if (
        not isinstance(head, str)
        or _SHA.fullmatch(head) is None
        or isinstance(records, bool)
        or not isinstance(records, int)
        or records < 1
        or isinstance(bytes_value, bool)
        or not isinstance(bytes_value, int)
        or bytes_value < 1
        or not isinstance(file_state, Mapping)
        or file_state.get("size_bytes") != bytes_value
    ):
        raise R4Error("production R4 candidate anchor checkpoint is invalid")
    return {
        "artifact_id": artifact_id,
        "file_sha256": file_sha256,
        "seal_sha256": str(payload["seal_sha256"]),
        "file_state": state,
        "r0_artifact_id": payload["r0_artifact_id"],
        "r0_file_sha256": payload["r0_file_sha256"],
        "bootstrap_source_commit": payload["bootstrap_source_commit"],
        "critical_module_sha256": dict(payload["critical_module_sha256"]),
        "candidate": {
            "head_sha256": head,
            "records": records,
            "bytes": bytes_value,
            "file_state": dict(file_state),
        },
    }


def _production_candidate_tail(
    path: Path, checkpoint: Mapping[str, Any], anchor: Mapping[str, Any]
) -> dict[str, Any]:
    """Require the current candidate checkpoint to equal the sealed R0 base.

    Candidate ledgers are production-sized.  A future append must first create
    a new offline R0 snapshot; accepting a self-resealed checkpoint plus a
    bounded tail would make the checkpoint itself the authority again.
    """

    expected = anchor.get("candidate")
    if not isinstance(expected, Mapping):
        raise R4Error("R0 candidate ledger anchor is invalid")
    anchor_head = expected.get("head_sha256")
    anchor_records = expected.get("records")
    anchor_bytes = expected.get("bytes")
    if (
        not isinstance(anchor_head, str)
        or _SHA.fullmatch(anchor_head) is None
        or isinstance(anchor_records, bool)
        or not isinstance(anchor_records, int)
        or anchor_records < 1
        or isinstance(anchor_bytes, bool)
        or not isinstance(anchor_bytes, int)
        or anchor_bytes < 1
    ):
        raise R4Error("R0 candidate ledger anchor is invalid")
    records = checkpoint.get("records")
    head = checkpoint.get("head_sha256")
    ledger_state = checkpoint.get("ledger_state")
    if (
        isinstance(records, bool)
        or not isinstance(records, int)
        or records < anchor_records
        or not isinstance(head, str)
        or _SHA.fullmatch(head) is None
        or not isinstance(ledger_state, Mapping)
    ):
        raise R4Error("production candidate checkpoint precedes R0 anchor")
    current_bytes = ledger_state.get("st_size")
    if (
        isinstance(current_bytes, bool)
        or not isinstance(current_bytes, int)
        or current_bytes < anchor_bytes
    ):
        raise R4Error("production candidate checkpoint precedes R0 bytes")
    anchor_file_state = _stable_candidate_file_state(expected.get("file_state"))
    checkpoint_file_state = _stable_candidate_file_state(checkpoint.get("file_state"))
    if (
        not anchor_file_state
        or checkpoint_file_state != anchor_file_state
        or records != anchor_records
        or current_bytes != anchor_bytes
        or head != anchor_head
    ):
        # A new append requires a fresh offline R0 snapshot.  We deliberately
        # do not trust a current checkpoint to authenticate an unanchored tail.
        raise R4Error("production candidate ledger differs from sealed R0 anchor")
    if _production_stat(path, label="production candidate ledger") != dict(
        ledger_state
    ):
        raise R4Error("production candidate ledger changed during anchor validation")
    return {
        "anchor_artifact_id": anchor.get("artifact_id"),
        "anchor_file_sha256": anchor.get("file_sha256"),
        "anchor_seal_sha256": anchor.get("seal_sha256"),
        "anchor_file_state": anchor_file_state,
        "anchor_r0_artifact_id": anchor.get("r0_artifact_id"),
        "anchor_r0_file_sha256": anchor.get("r0_file_sha256"),
        "anchor_bootstrap_source_commit": anchor.get("bootstrap_source_commit"),
        "anchor_critical_module_sha256": anchor.get("critical_module_sha256"),
        "anchor_head_sha256": anchor_head,
        "anchor_records": anchor_records,
        "anchor_bytes": anchor_bytes,
        "tail_records": 0,
        "tail_bytes": 0,
        "tail_verified": True,
    }


def _stable_candidate_file_state(value: Any) -> dict[str, int]:
    """Normalize durable candidate identity across APFS mount-id changes."""

    stable = {"size_bytes", "st_ino", "st_mtime_ns", "st_ctime_ns"}
    if not isinstance(value, Mapping) or set(value) not in (
        stable,
        stable | {"st_dev"},
    ):
        return {}
    result = {key: value[key] for key in stable}
    return (
        result
        if all(type(item) is int and item >= 0 for item in result.values())
        else {}
    )


def _production_chain(
    path: Path, checkpoint_path: Path, *, ledger_name: str = "label-ledger.jsonl"
) -> dict[str, Any]:
    """Verify a bounded label-ledger view against its sealed head checkpoint.

    A small ledger is checked in full.  Once it exceeds the hot-path bound,
    only the bounded tail is parsed; the sealed checkpoint supplies the
    historical record count, head digest, and immutable file identity.
    """

    checkpoint = _production_ledger_checkpoint(
        path, checkpoint_path, ledger_name=ledger_name
    )
    before = _production_stat(path, label="production label ledger")
    if before["st_size"] > PRODUCTION_MAX_LEDGER_BYTES:
        raise R4Error("production label ledger exceeds bounded size")
    checkpoint_file_state = checkpoint["file_state"]

    rows: list[dict[str, Any]] = []
    previous = ""
    full_scan = before["st_size"] <= PRODUCTION_MAX_FULL_LEDGER_BYTES
    try:
        with path.open("rb") as handle:
            if not full_scan:
                offset = max(0, before["st_size"] - PRODUCTION_MAX_LEDGER_TAIL_BYTES)
                handle.seek(offset)
                if offset:
                    handle.readline()
                raw_tail = handle.read(PRODUCTION_MAX_LEDGER_TAIL_BYTES + 1)
                if len(raw_tail) > PRODUCTION_MAX_LEDGER_TAIL_BYTES:
                    raise R4Error("production label ledger tail exceeds bound")
                lines = raw_tail.splitlines(keepends=True)
                if not lines or not lines[-1].endswith(b"\n"):
                    raise R4Error("production label ledger is truncated")
                iterator = enumerate(lines, start=1)
            else:
                iterator = enumerate(handle, start=1)
            for index, raw in iterator:
                if index > PRODUCTION_MAX_ROWS:
                    raise R4Error("production label ledger is unbounded")
                if not raw.endswith(b"\n") or raw == b"\n":
                    raise R4Error("production label ledger is truncated")
                try:
                    payload = json.loads(raw[:-1])
                except (ValueError, UnicodeError, RecursionError) as exc:
                    raise R4Error("production label ledger JSON is invalid") from exc
                if not isinstance(payload, dict) or _json_bytes(payload) != raw[:-1]:
                    raise R4Error("production label ledger JSON is not canonical")
                if (
                    payload.get("schema") != "chronovisor.recall-distillation.v1"
                    or payload.get("namespace") != "recall-distillation"
                    or not isinstance(payload.get("previous_sha256"), str)
                    or _SHA.fullmatch(str(payload.get("record_sha256"))) is None
                ):
                    raise R4Error("production label ledger chain is corrupted")
                if full_scan and payload.get("previous_sha256") != previous:
                    raise R4Error("production label ledger chain is corrupted")
                if (
                    not full_scan
                    and rows
                    and payload.get("previous_sha256") != previous
                ):
                    raise R4Error("production label ledger tail chain is corrupted")
                unsigned = {
                    key: value
                    for key, value in payload.items()
                    if key != "record_sha256"
                }
                if _sha256(unsigned) != payload["record_sha256"]:
                    raise R4Error("production label ledger hash mismatch")
                previous = str(payload["record_sha256"])
                rows.append(payload)
    except OSError as exc:
        raise R4Error("production label ledger read failed") from exc
    after = _production_stat(path, label="production label ledger")
    if before != after:
        raise R4Error("production label ledger changed during validation")
    if previous != checkpoint["head_sha256"]:
        raise R4Error("production label checkpoint head mismatch")
    if full_scan and len(rows) != checkpoint["records"]:
        raise R4Error("production label checkpoint count mismatch")
    if len(rows) > checkpoint["records"]:
        raise R4Error("production label checkpoint tail count mismatch")
    digest: str | None = _file_sha256(path) if full_scan else None
    if full_scan and _production_stat(path, label="production label ledger") != after:
        raise R4Error("production label ledger changed during hashing")
    if (
        _production_stat(checkpoint_path, label="production label checkpoint")
        != checkpoint["checkpoint_state"]
    ):
        raise R4Error("production label checkpoint changed during validation")
    return {
        "rows": rows,
        "count": int(checkpoint["records"]),
        "head_sha256": str(checkpoint["head_sha256"]),
        "file_state": after,
        "sha256": digest,
        "checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "checkpoint_file_state": dict(checkpoint_file_state),
    }


def _production_ox_events(
    root: Path,
    *,
    source: Mapping[str, Any],
    contract_id: str,
    workset: Mapping[str, Any] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Re-derive OX ramp/failure/lease evidence from immutable event chains."""

    if workset is None:
        workset = _production_workset(root / PRODUCTION_WORKSET_RELATIVE)

    contract, _, _ = _production_json(
        root / PRODUCTION_CONTRACT_DIR_RELATIVE / f"{contract_id}.json",
        label="production profile contract",
        schema="chronovisor.recall-distill-remote-profile.v2",
    )
    if (
        contract.get("artifact_id") != contract_id
        or contract.get("request_revision") != OX_REQUEST_REVISION
    ):
        raise R4Error("production OX contract request revision is invalid")
    contract_revision = contract["request_revision"]
    contract_expiry = _canonical_future_expiry(contract.get("expires_at"))
    if contract_expiry is None or contract.get("expires_at") != contract_expiry:
        raise R4Error("production OX contract expiry is invalid")
    contracts: dict[str, Mapping[str, Any]] = {contract_id: contract}

    names = {
        "ramp": "ox-ramp-receipts.jsonl",
        "failure": "ox-failure-receipts.jsonl",
        "lease": "ox-lease-recovery-receipts.jsonl",
    }
    group_kinds = {
        "ramp": "ox-ramp-stage",
        "failure": "ox-provider-failure",
        "lease": "ox-lease-reclaim",
    }
    result: dict[str, list[dict[str, Any]]] = {
        **{key: [] for key in names},
        # Read legacy chains only to make their non-certifying status explicit.
        # They must never supply formal ramp/failure/lease evidence.
        "legacy": [],
    }
    common = {
        "profile_contract_id": contract_id,
        "source_commit": source.get("commit"),
        "source_tree_sha256": source.get("tree_sha256"),
        "source_ox_identity_sha256": source.get("ox_identity_sha256"),
    }
    legacy_allowed = {
        "ox-ramp-stage": common.keys()
        | {
            "kind",
            "request_revision",
            "expires_at",
                "cap",
                "valid_receipts",
            "attempts",
            "work_ids",
            "label_count",
            "label_head_sha256",
            "captured_at",
        },
        "ox-provider-failure": common.keys()
        | {
            "kind",
            "request_revision",
            "expires_at",
            "category",
            "status",
            "attempts",
            "bounded",
            "before_cap",
            "after_cap",
            "work_ids",
            "attempts_by_work",
            "provider_receipts",
            "captured_at",
        },
        "ox-lease-reclaim": common.keys()
        | {
            "kind",
            "request_revision",
            "expires_at",
            "workset_receipt_generation",
            "workset_receipt_sha256",
            "reclaimed",
            "leased_after",
            "captured_at",
        },
    }
    for group, name in names.items():
        seen_event_keys: set[str] = set()
        path = root / PRODUCTION_DISTILLATION_RELATIVE / name
        if not path.exists():
            continue
        checkpoint = path.with_suffix(path.suffix + ".head.json")
        view = _production_chain(path, checkpoint, ledger_name=name)
        if view["sha256"] is None:
            raise R4Error("production OX event ledger must fit the exact collector")
        for record_index, record in enumerate(view["rows"], start=1):
            payload = {
                key: value
                for key, value in record.items()
                if key
                not in {
                    "schema",
                    "namespace",
                    "previous_sha256",
                    "record_sha256",
                    "event_key",
                    "event_binding_sha256",
                }
            }
            kind = payload.get("kind")
            event_version = payload.get("event_version")
            if not isinstance(kind, str) or kind not in legacy_allowed:
                raise R4Error("production OX event schema is invalid")
            event_contract_id = payload.get("profile_contract_id")
            if (
                not isinstance(event_contract_id, str)
                or _SHA.fullmatch(event_contract_id) is None
            ):
                raise R4Error("production OX event contract id is invalid")
            event_contract = contracts.get(event_contract_id)
            if event_contract is None:
                event_contract, _, _ = _production_historical_profile_contract(
                    root
                    / PRODUCTION_CONTRACT_DIR_RELATIVE
                    / f"{event_contract_id}.json",
                    contract_id=event_contract_id,
                )
                contracts[event_contract_id] = event_contract
            historical = event_contract_id != contract_id
            if historical:
                event_revision = event_contract.get("request_revision")
                event_expiry = event_contract.get("expires_at")
                event_common = {
                    "profile_contract_id": event_contract_id,
                    "source_commit": event_contract.get("source_commit"),
                    "source_tree_sha256": event_contract.get("source_tree_sha256"),
                    "source_ox_identity_sha256": event_contract.get(
                        "source_ox_identity_sha256"
                    ),
                }
                if (
                    event_contract.get("artifact_id") != event_contract_id
                    or not isinstance(event_revision, str)
                    or not event_revision
                    or _parse_expiry(event_expiry) is None
                    or _COMMIT.fullmatch(str(event_common["source_commit"])) is None
                    or _SHA.fullmatch(str(event_common["source_tree_sha256"])) is None
                    or _SHA.fullmatch(
                        str(event_common["source_ox_identity_sha256"])
                    )
                    is None
                ):
                    raise R4Error("historical production OX contract is invalid")
            else:
                event_revision = contract_revision
                event_expiry = contract_expiry
                event_common = common
            if (
                kind != group_kinds[group]
                or any(
                    payload.get(key) != value for key, value in event_common.items()
                )
                or payload.get("request_revision") != event_revision
                or payload.get("expires_at") != event_expiry
                or _parse_expiry(payload.get("expires_at")) is None
                or not isinstance(payload.get("captured_at"), str)
                or not isinstance(record.get("event_key"), str)
                or not isinstance(record.get("event_binding_sha256"), str)
                or record["event_binding_sha256"] != _sha256(payload)
            ):
                raise R4Error("production OX event schema is invalid")

            if event_version is not None and (
                isinstance(event_version, bool) or not isinstance(event_version, int)
            ):
                raise R4Error("production OX event version is invalid")
            legacy = event_version is None or event_version == 1
            if legacy:
                legacy_keys = set(legacy_allowed[kind]) | (
                    {"event_version"} if event_version == 1 else set()
                )
                legacy_required = set(legacy_allowed[kind])
                if kind == "ox-provider-failure":
                    legacy_required -= {"bounded", "before_cap", "after_cap"}
                if not legacy_required.issubset(payload) or not set(payload).issubset(
                    legacy_keys
                ):
                    raise R4Error("production OX legacy event schema is invalid")
            elif event_version != 2:
                raise R4Error("production OX event version is invalid")
            else:
                v2_common = {
                    "event_version",
                    "kind",
                    "profile_contract_id",
                    "source_commit",
                    "source_tree_sha256",
                    "source_ox_identity_sha256",
                    "request_revision",
                    "expires_at",
                    "captured_at",
                }
                expected_keys = {
                    "ox-ramp-stage": v2_common
                        | {
                            "cap",
                            "next_cap",
                            "valid_receipts",
                        "attempts",
                        "work_ids",
                        "label_count",
                        "label_head_sha256",
                        "failure_record_count",
                        "failure_head_sha256",
                    },
                    "ox-provider-failure": v2_common
                    | {
                        "category",
                        "status",
                        "cap",
                        "attempts",
                        "work_ids",
                        "attempts_by_work",
                        "provider_requests",
                        "provider_receipts",
                    },
                    "ox-lease-reclaim": v2_common
                    | {
                        "workset_receipt_generation",
                        "workset_receipt_sha256",
                        "work_ids_sha256",
                        "reclaimed",
                        "leased_after",
                    },
                }[kind]
                category = payload.get("category")
                if kind == "ox-provider-failure":
                    if category == "429":
                        expected_keys |= {"before_cap", "after_cap"}
                    elif category in {"5xx", "timeout", "402", "paid", "model_drift"}:
                        if category in {"5xx", "timeout"}:
                            expected_keys |= {"bounded"}
                    else:
                        raise R4Error("production OX failure category is invalid")
                if set(payload) != expected_keys:
                    raise R4Error("production OX event schema is invalid")

            if kind == "ox-provider-failure":
                attempts = payload.get("attempts")
                work_ids = payload.get("work_ids")
                attempts_by_work = payload.get("attempts_by_work")
                provider_requests = payload.get("provider_requests")
                provider_receipts = payload.get("provider_receipts")
                if (
                    isinstance(attempts, bool)
                    or not isinstance(attempts, int)
                    or attempts < 1
                    or not isinstance(work_ids, list)
                    or not work_ids
                    or not all(
                        isinstance(work_id, str) and _SHA.fullmatch(work_id)
                        for work_id in work_ids
                    )
                    or len(set(work_ids)) != len(work_ids)
                    or not isinstance(attempts_by_work, Mapping)
                    or set(attempts_by_work) != set(work_ids)
                    or not all(
                        isinstance(value, int)
                        and not isinstance(value, bool)
                        and value > 0
                        for value in attempts_by_work.values()
                    )
                    or not isinstance(provider_receipts, Mapping)
                    or set(provider_receipts) != set(work_ids)
                    or (
                        not legacy
                        and (
                            not isinstance(provider_requests, Mapping)
                            or set(provider_requests) != set(work_ids)
                        )
                    )
                    or not all(isinstance(value, str) and _SHA.fullmatch(value) for value in provider_receipts.values())
                    or len(set(provider_receipts.values())) != 1
                    or attempts != len(set(provider_receipts.values()))
                    or not str(payload.get("captured_at"))
                ):
                    raise R4Error("production OX failure event is incomplete")
                category = payload.get("category")
                if not legacy and (
                    (isinstance(payload.get("cap"), bool)
                        or payload.get("cap") not in PRODUCTION_RAMP_CAPS)
                    or (category == "429" and (
                        payload.get("status") != "deferred"
                        or isinstance(payload.get("before_cap"), bool)
                        or not isinstance(payload.get("before_cap"), int)
                        or payload.get("before_cap") not in PRODUCTION_RAMP_CAPS
                        or isinstance(payload.get("after_cap"), bool)
                        or not isinstance(payload.get("after_cap"), int)
                        or payload.get("after_cap") not in PRODUCTION_RAMP_CAPS
                    ))
                    or (category in {"5xx", "timeout"} and (
                        payload.get("status") != "deferred"
                        or payload.get("bounded") is not True
                    ))
                    or (category in {"402", "paid", "model_drift"}
                        and payload.get("status") != "hard_stop")
                ):
                    raise R4Error("production OX failure category is invalid")
                work_items = workset.get("items")
                if not isinstance(work_items, Mapping):
                    raise R4Error("production OX workset payload inventory is missing")
                for work_id in work_ids:
                    item = work_items.get(work_id)
                    if not isinstance(item, Mapping):
                        raise R4Error("production OX failure work is unbound")
                    payload_digest = item.get("payload_digest")
                    if not isinstance(payload_digest, str) or _SHA.fullmatch(payload_digest) is None:
                        raise R4Error("production OX workset payload digest is invalid")
                    if (
                        not legacy
                        and (
                            not isinstance(provider_requests, Mapping)
                            or provider_requests.get(work_id)
                        != _expected_ox_provider_request_sha256(
                            profile_contract_id=event_contract_id,
                            payload_digest=payload_digest,
                            work_id=work_id,
                            expires_at=str(event_expiry),
                        )
                        )
                    ):
                        raise R4Error("production OX provider request is unbound")
                    if legacy and provider_receipts.get(work_id) != _expected_ox_provider_request_sha256(
                        profile_contract_id=event_contract_id,
                        payload_digest=payload_digest,
                        work_id=work_id,
                        expires_at=str(event_expiry),
                    ):
                        raise R4Error("production OX legacy provider receipt is unbound")
                    if (
                        not legacy
                        and provider_receipts.get(work_id)
                        == _expected_ox_provider_request_sha256(
                            profile_contract_id=event_contract_id,
                            payload_digest=payload_digest,
                            work_id=work_id,
                            expires_at=str(event_expiry),
                        )
                    ):
                        raise R4Error("production OX actual provider receipt is synthetic")
            elif kind == "ox-ramp-stage":
                ramp_cap = payload.get("cap")
                expected_next_cap = (
                    {1: 2, 2: 5, 5: 10, 10: 10}.get(ramp_cap)
                    if isinstance(ramp_cap, int)
                    else None
                )
                if (
                        isinstance(ramp_cap, bool)
                        or ramp_cap not in PRODUCTION_RAMP_CAPS
                        or (
                            not legacy
                            and (
                                isinstance(payload.get("next_cap"), bool)
                                or payload.get("next_cap") not in PRODUCTION_RAMP_CAPS
                                or payload.get("next_cap")
                                != expected_next_cap
                            )
                        )
                    or isinstance(payload.get("valid_receipts"), bool)
                    or not isinstance(payload.get("valid_receipts"), int)
                    or payload.get("valid_receipts", -1) < 0
                    or isinstance(payload.get("attempts"), bool)
                    or not isinstance(payload.get("attempts"), int)
                    or payload.get("attempts", -1) < 0
                    or not isinstance(payload.get("work_ids"), list)
                    or not all(
                        isinstance(work_id, str) and _SHA.fullmatch(work_id)
                        for work_id in payload["work_ids"]
                    )
                    or len(set(payload["work_ids"])) != len(payload["work_ids"])
                    or isinstance(payload.get("label_count"), bool)
                    or not isinstance(payload.get("label_count"), int)
                    or payload.get("label_count", 0) < 1
                    or _SHA.fullmatch(str(payload.get("label_head_sha256"))) is None
                        or (
                            not legacy
                            and (
                                isinstance(payload.get("failure_record_count"), bool)
                                or not isinstance(payload.get("failure_record_count"), int)
                                or payload.get("failure_record_count", -1) < 0
                                or (
                                    payload.get("failure_record_count") == 0
                                    and payload.get("failure_head_sha256") != ""
                                )
                                or (
                                    payload.get("failure_record_count", 0) > 0
                                    and _SHA.fullmatch(str(payload.get("failure_head_sha256")))
                                    is None
                                )
                            )
                        )
                ):
                    raise R4Error("production OX ramp event is incomplete")
            else:
                if (
                    isinstance(payload.get("workset_receipt_generation"), bool)
                    or not isinstance(payload.get("workset_receipt_generation"), int)
                    or payload.get("workset_receipt_generation", 0) < 1
                    or _SHA.fullmatch(str(payload.get("workset_receipt_sha256")))
                    is None
                    or _SHA.fullmatch(str(payload.get("work_ids_sha256"))) is None
                    or isinstance(payload.get("reclaimed"), bool)
                    or not isinstance(payload.get("reclaimed"), int)
                    or payload.get("reclaimed", -1) < 0
                    or isinstance(payload.get("leased_after"), bool)
                    or not isinstance(payload.get("leased_after"), int)
                    or payload.get("leased_after", -1) < 0
                ):
                    raise R4Error("production OX lease event is incomplete")
            if legacy:
                identity = {
                    key: payload.get(key)
                    for key in (
                        "profile_contract_id",
                        "source_commit",
                        "source_tree_sha256",
                        "source_ox_identity_sha256",
                        "request_revision",
                    )
                }
                if kind == "ox-ramp-stage":
                    identity = {"kind": kind, **identity, "cap": payload.get("cap")}
                elif kind == "ox-provider-failure":
                    identity = {
                        "kind": kind,
                        **identity,
                        "category": payload.get("category"),
                        "work_ids": payload.get("work_ids"),
                        "attempts_by_work": payload.get("attempts_by_work"),
                        "provider_receipts": payload.get("provider_receipts"),
                    }
                else:
                    identity = {
                        "kind": kind,
                        **identity,
                        "receipt": payload.get("workset_receipt_sha256"),
                    }
            else:
                identity = {
                    key: value for key, value in payload.items() if key != "captured_at"
                }
            if record["event_key"] != _sha256(identity):
                raise R4Error("production OX event identity is invalid")
            if record["event_key"] in seen_event_keys:
                raise R4Error("production OX event identity is duplicated")
            seen_event_keys.add(str(record["event_key"]))
            anchor_id = _sha256(
                {
                    "schema": "chronovisor.recall-distill-ox-event-anchor.v1",
                    "namespace": "recall-distillation",
                    "kind": "ox-event-anchor",
                    "ledger_name": name,
                    "event_key": record["event_key"],
                    "event_binding_sha256": record["event_binding_sha256"],
                    "record_sha256": record["record_sha256"],
                }
            )
            anchor, _, _ = _production_json(
                root / PRODUCTION_EVENT_ANCHOR_RELATIVE / f"{anchor_id}.json",
                label="production OX event anchor",
                schema="chronovisor.recall-distill-ox-event-anchor.v1",
            )
            expected_anchor = {
                "schema": "chronovisor.recall-distill-ox-event-anchor.v1",
                "namespace": "recall-distillation",
                "kind": "ox-event-anchor",
                "artifact_id": anchor_id,
                "ledger_name": name,
                "event_key": record["event_key"],
                "event_binding_sha256": record["event_binding_sha256"],
                "record_sha256": record["record_sha256"],
            }
            if (
                set(anchor) != {*expected_anchor, "seal_sha256"}
                or any(anchor.get(key) != value for key, value in expected_anchor.items())
                or anchor.get("seal_sha256")
                != _sha256(expected_anchor)
            ):
                raise R4Error("production OX event anchor is invalid")
            if historical:
                continue
            event = {
                **payload,
                "record_sha256": record["record_sha256"],
                "record_index": record_index,
            }
            if legacy:
                result["legacy"].append({"group": group, **event})
            else:
                result[group].append(event)
    return result


def _load_owned_fault_scenarios(
    path: Path | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load canonical owned-clone artifacts, never from the managed root."""

    if path is None:
        return [], {"files": [], "count": 0}
    if _has_symlink_component(path) or not path.is_dir():
        raise R4Error("owned fault input path is unsafe")
    try:
        paths = sorted(path.iterdir())
    except OSError as exc:
        raise R4Error("owned fault input is unavailable") from exc
    if not paths:
        return [], {"files": [], "count": 0}
    artifacts: list[dict[str, Any]] = []
    files: list[dict[str, Any]] = []
    for item in paths:
        if item.is_symlink() or not item.is_file() or item.suffix != ".json":
            raise R4Error("owned fault input contains an unsafe entry")
        before = _stat(item)
        digest = _file_sha256(item)
        if before["st_size"] > _MAX_RECEIPT_BYTES:
            raise R4Error("owned fault artifact exceeds bounded size")
        try:
            raw = item.read_bytes()
            artifact = json.loads(raw)
        except (OSError, ValueError, UnicodeError) as exc:
            raise R4Error("owned fault artifact JSON is invalid") from exc
        verified = _verify_seal(artifact, schema=R4_FAULT_SCENARIO_SCHEMA)
        artifact_id = verified["artifact_id"]
        if (
            item.name != f"{artifact_id}.json"
            or raw != _json_bytes(verified) + b"\n"
            or _stat(item) != before
            or _file_sha256(item) != digest
        ):
            raise R4Error("owned fault artifact is not canonical")
        artifacts.append(verified)
        files.append({"path": item.name, "sha256": digest, "file_state": before})
    if len({str(item["artifact_id"]) for item in artifacts}) != len(artifacts):
        raise R4Error("duplicate owned fault artifact id")
    return artifacts, {"files": files, "count": len(artifacts)}


def _validate_owned_fault_scenarios(
    artifacts: Sequence[Mapping[str, Any]], source: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate the nine provider-free public-worker faults as source evidence."""

    reasons: set[str] = set()
    found: set[str] = set()
    contract_ids: set[str] = set()
    expected_source = {
        "source_commit": source.get("commit"),
        "source_tree_sha256": source.get("tree_sha256"),
        "source_ox_identity_sha256": source.get("ox_identity_sha256"),
    }
    expected_keys = {
        "artifact_id", "schema", "namespace", "seal_sha256", "scenario",
        "writer_path", "test_only", "source", "profile_contract_id", "outcome",
        "workset_receipt", "event_heads", "owned_root",
    }

    def valid_file_state(value: object) -> bool:
        keys = {
            "st_dev",
            "st_ino",
            "st_mode",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        }
        return (
            isinstance(value, Mapping)
            and set(value) == keys
            and all(
                isinstance(value.get(key), int)
                and not isinstance(value.get(key), bool)
                and value[key] >= 0
                for key in keys
            )
            and value["st_dev"] > 0
            and value["st_ino"] > 0
            and value["st_mode"] == 0o600
        )

    def valid_sqlite_state(value: object) -> bool:
        return (
            isinstance(value, Mapping)
            and set(value) == {"main", "wal", "shm"}
            and valid_file_state(value.get("main"))
            and all(
                value.get(name) is None or valid_file_state(value.get(name))
                for name in ("wal", "shm")
            )
        )

    for artifact in artifacts:
        artifact_id = artifact.get("artifact_id")
        unsigned = {
            key: value
            for key, value in artifact.items()
            if key not in {"artifact_id", "seal_sha256"}
        }
        scenario = artifact.get("scenario")
        outcome = artifact.get("outcome")
        receipt = artifact.get("workset_receipt")
        heads = artifact.get("event_heads")
        owned_root = artifact.get("owned_root")
        invalid = (
            set(artifact) != expected_keys
            or not isinstance(artifact_id, str)
            or artifact_id != _sha256(unsigned)
            or artifact.get("writer_path") != "public-run-distillation-chunk-v1"
            or artifact.get("test_only") is not True
            or artifact.get("source") != expected_source
            or not isinstance(scenario, str)
            or scenario not in PRODUCTION_FAULT_SCENARIOS
            or scenario in found
            or _SHA.fullmatch(str(artifact.get("profile_contract_id"))) is None
            or not isinstance(outcome, Mapping)
            or set(outcome) != {
                "profile_stopped", "backoff_bounded", "quarantined", "ready",
                "leased", "duplicate_labels", "adapter_calls", "provider_calls",
            }
            or not isinstance(outcome.get("profile_stopped"), bool)
            or not isinstance(outcome.get("backoff_bounded"), bool)
            or any(
                isinstance(outcome.get(name), bool)
                or not isinstance(outcome.get(name), int)
                or outcome[name] < 0
                for name in (
                    "quarantined", "ready", "leased", "duplicate_labels",
                    "adapter_calls", "provider_calls",
                )
            )
            or outcome.get("leased") != 0
            or outcome.get("duplicate_labels") != 0
            or outcome.get("provider_calls") != 0
            or not isinstance(receipt, Mapping)
            or set(receipt) != {"generation", "head_sha256"}
            or isinstance(receipt.get("generation"), bool)
            or not isinstance(receipt.get("generation"), int)
            or receipt["generation"] < 1
            or _SHA.fullmatch(str(receipt.get("head_sha256"))) is None
            or not isinstance(heads, Mapping)
            or set(heads) != {"ramp", "failure", "lease"}
            or any(
                not isinstance(value, str) or (value and _SHA.fullmatch(value) is None)
                for value in heads.values()
            )
            or not isinstance(owned_root, Mapping)
            or set(owned_root) != {"before", "after", "run_status"}
            or not valid_sqlite_state(owned_root.get("before"))
            or not valid_sqlite_state(owned_root.get("after"))
            or owned_root.get("run_status") != "deferred"
        )
        if invalid:
            reasons.add("owned_fault_artifact_invalid")
            continue
        # ``invalid`` has already rejected these shapes.  Keep the explicit
        # guard so static analysis retains that fail-closed fact.
        if not isinstance(scenario, str) or not isinstance(outcome, Mapping):
            raise R4Error("owned fault validation shape changed")
        before_main = owned_root["before"]["main"]
        after_main = owned_root["after"]["main"]
        if (
            before_main["st_dev"] != after_main["st_dev"]
            or before_main["st_ino"] != after_main["st_ino"]
            or before_main["st_mode"] != after_main["st_mode"]
        ):
            reasons.add("owned_fault_safe_outcome_invalid")
            continue
        if scenario in {"http_429", "http_5xx", "timeout"} and not outcome[
            "backoff_bounded"
        ]:
            reasons.add("owned_fault_safe_outcome_invalid")
            continue
        if scenario in {"http_402_paid", "model_drift", "disable_rollback"} and not outcome[
            "profile_stopped"
        ]:
            reasons.add("owned_fault_safe_outcome_invalid")
            continue
        if scenario == "invalid_output_quarantine" and (
            outcome["quarantined"] < 1 or outcome["adapter_calls"] < 1
        ):
            reasons.add("owned_fault_safe_outcome_invalid")
            continue
        if scenario == "lease_expiry_reclaim" and not heads["lease"]:
            reasons.add("owned_fault_safe_outcome_invalid")
            continue
        if scenario == "resource_pressure_preemption" and outcome["adapter_calls"] < 1:
            reasons.add("owned_fault_safe_outcome_invalid")
            continue
        provider_faults = {
            "http_429",
            "http_5xx",
            "timeout",
            "http_402_paid",
            "model_drift",
            "lease_expiry_reclaim",
            "resource_pressure_preemption",
        }
        if scenario in provider_faults and (
            outcome["adapter_calls"] < 1 or not heads["failure"]
        ):
            reasons.add("owned_fault_safe_outcome_invalid")
            continue
        if scenario == "disable_rollback" and outcome["adapter_calls"] != 0:
            reasons.add("owned_fault_safe_outcome_invalid")
            continue
        contract_ids.add(str(artifact["profile_contract_id"]))
        found.add(scenario)
    if found != set(PRODUCTION_FAULT_SCENARIOS):
        reasons.add("owned_fault_scenarios_incomplete")
    if len(contract_ids) != 1:
        reasons.add("owned_fault_contract_mixing")
    return {
        "passed": not reasons,
        "reasons": sorted(reasons),
        "count": len(artifacts),
        "scenarios": sorted(found),
    }


def run_owned_fault_scenarios(
    *, source_root: Path, source_commit: str, output: Path
) -> list[Path]:
    """Run the nine provider-free OX faults through the public worker.

    This is intentionally an owned-clone-only contract test.  The adapter has
    no transport and records attempted adapter calls separately from network
    calls (which are always zero).  Its sealed results are source-bound test
    evidence and can never replace the fixed-root production collector.
    """

    _reject_original_symlinks((("source", source_root), ("fault output", output)))
    source_root = source_root.expanduser().resolve(strict=True)
    output = output.expanduser().resolve(strict=False)
    assert_root_matrix(source_root, output)
    if _overlap(output, PRODUCTION_ROOT.expanduser().absolute()):
        raise R4Error("fault output/production paths overlap")
    source = _assert_source(source_root, source_commit)
    output.mkdir(parents=True, exist_ok=True)
    from chronovisor.core.llm_runtime import GenerationResult, RouteLocation
    from chronovisor.core.provider_profiles import (
        ProviderAdapterError,
        ProviderFailureCategory,
    )
    from chronovisor.core.raw_segment import append_capture
    from chronovisor.core.store import RuntimeContext, init_chronovisor
    from chronovisor.recall import recall_distillation as distillation
    from chronovisor.recall.recall_distillation_remote_teacher import (
        OX_ALPHA_ENDPOINT,
        OpenCodeOxAlphaTeacher,
    )
    from chronovisor.recall.recall_distillation_workset import DistillationWorkset

    categories: dict[str, ProviderFailureCategory | None] = {
        "http_429": ProviderFailureCategory.RATE_LIMITED,
        "http_5xx": ProviderFailureCategory.SERVER_ERROR,
        "timeout": ProviderFailureCategory.TIMEOUT,
        "http_402_paid": ProviderFailureCategory.PAYMENT_REQUIRED,
        "model_drift": None,
        "invalid_output_quarantine": None,
            "lease_expiry_reclaim": ProviderFailureCategory.SERVER_ERROR,
        "resource_pressure_preemption": ProviderFailureCategory.RATE_LIMITED,
        "disable_rollback": None,
    }

    class _OwnedBackend:
        provider = "opencode-go"
        location = RouteLocation.REMOTE
        _profile = type("Profile", (), {"endpoint": OX_ALPHA_ENDPOINT})()

        def __init__(self, scenario: str) -> None:
            self.scenario = scenario
            self.adapter_calls = 0
            self.network_calls = 0

        def capabilities_for(self, _model: str) -> object:
            return type("Capabilities", (), {"structured_output": True})()

        def generate(self, _request: object, *, model: str) -> GenerationResult:
            self.adapter_calls += 1
            category = categories[self.scenario]
            if category is not None:
                raise ProviderAdapterError(
                    category,
                    request_id=f"owned-{self.scenario}-{self.adapter_calls}",
                )
            if self.scenario == "model_drift":
                    return GenerationResult(
                        content='{"labels":[]}',
                        provider="opencode-go",
                        model=model,
                        finish_reason="stop",
                        metadata={
                            "returned_model": "unexpected-model",
                            "request_id": f"owned-{self.scenario}-{self.adapter_calls}",
                        },
                    )
            return GenerationResult(
                content="not-json",
                provider="opencode-go",
                model=model,
                finish_reason="stop",
                metadata={"returned_model": "deepseek-v4-flash"},
            )

    published: list[Path] = []
    with tempfile.TemporaryDirectory(prefix="chronovisor-r4-fault-") as temporary:
        # macOS commonly spells this path through /var, which is a symlink to
        # /private/var.  Pass only the canonical path into the same OKF guard
        # used by production; do not weaken that guard for the owned clone.
        base = Path(temporary).resolve(strict=True)
        for scenario in PRODUCTION_FAULT_SCENARIOS:
            root = base / scenario
            root.mkdir()
            init_chronovisor(RuntimeContext(root))
            raw = root / "raw"
            source_file = root / "source.jsonl"
            rows = [
                {
                    "type": "response_item",
                    "timestamp": f"2026-01-{(index % 27) + 1:02d}T00:00:00Z",
                    "payload": {
                        "type": "message",
                        "role": "user" if index % 2 == 0 else "assistant",
                        "content": [
                            {
                                "type": "input_text"
                                if index % 2 == 0
                                else "output_text",
                                "text": f"owned fault evidence {index}",
                            }
                        ],
                    },
                }
                for index in range(48)
            ]
            data = b"".join(_json_bytes(row) + b"\n" for row in rows)
            source_file.write_bytes(data)
            append_capture(
                raw_dir=raw,
                raw_id="save-codex-test.md",
                idempotency_key="codex-test",
                host="codex",
                session_key="a" * 24,
                session_id=f"owned-{scenario}",
                source_file=source_file,
                after_line=0,
                until_line=len(rows),
                source_bytes=data,
                record_count=len(rows),
                now=datetime(2026, 8, 25, tzinfo=UTC),
            )
            config = root / "config.toml"
            config.write_text(
                "[recall.distillation]\n"
                "enabled = true\n"
                'teacher_profile = "deepseek-v4-flash-single-v1"\n'
                "ox_enabled = true\nox_free_only = false\n"
                'ox_expires_at = "2099-01-01T00:00:00Z"\n'
                "teacher_claim_limit = 1\nteacher_max_inflight = 1\n",
                encoding="utf-8",
            )
            workset_path = root / PRODUCTION_WORKSET_RELATIVE
            distillation.bootstrap_r4_distillation_root_authority(root)
            DistillationWorkset(
                workset_path
            )  # Explicit bootstrap; worker cannot migrate.
            attestation = (
                root
                / PRODUCTION_DISTILLATION_RELATIVE
                / "r4-simulation-attestation.json"
            )
            root_identity = root.stat()
            attestation_expiry = (
                datetime.now(UTC) + timedelta(minutes=5)
            ).isoformat().replace("+00:00", "Z")
            attestation.write_bytes(
                _json_bytes(
                    _sealed(
                        {
                            "schema": "chronovisor.recall-r4-simulation-attestation.v1",
                            "namespace": "recall-distillation",
                            "expires_at": attestation_expiry,
                            "owned_root": {
                                "st_dev": root_identity.st_dev,
                                "st_ino": root_identity.st_ino,
                            },
                            "source_binding": {
                                "source_commit": source["commit"],
                                "source_tree_sha256": source["tree_sha256"],
                                "source_ox_identity_sha256": source[
                                    "ox_identity_sha256"
                                ],
                            },
                        }
                    )
                )
            )
            backend = _OwnedBackend(scenario)
            teacher = OpenCodeOxAlphaTeacher(
                backend,
                test_only=True,
                enabled=scenario != "disable_rollback",
                simulation_attestation=attestation,
                owned_root=root,
            )
            before = _production_sqlite_state(workset_path, label="owned fault workset")
            result = distillation.run_distillation_chunk(
                root=root,
                raw_dir=raw,
                config_path=config,
                teachers={distillation.OX_TEACHER_ROLE: teacher},
                max_elapsed_seconds=60,
            )
            after = _production_sqlite_state(workset_path, label="owned fault workset")
            if scenario == "lease_expiry_reclaim":
                # Exercise the durable ownership boundary independently of the
                # provider fault: a reclaimed old lease must reject its former
                # owner before the new owner is released.
                # The preceding provider fault leaves the one-item fixture on
                # its bounded 60s retry schedule.  Advance only this owned
                # Workset clock before exercising lease expiry itself.
                fault_clock = [time.time() + 61]
                fault_workset = DistillationWorkset(
                    workset_path, clock=lambda clock=fault_clock: clock[0], migrate=False
                )
                old_claim = fault_workset.claim("ox", 1, "owned-old", 0.01)
                if len(old_claim) != 1:
                    raise R4Error("owned lease scenario has no claimable work")
                fault_clock[0] += 1
                new_claim = fault_workset.claim("ox", 1, "owned-new", 60)
                if len(new_claim) != 1:
                    raise R4Error("owned lease was not reclaimed")
                try:
                    fault_workset.commit(old_claim, [{"status": "retry"}])
                except Exception:
                    pass
                else:
                    raise R4Error("reclaimed old owner was accepted")
                if fault_workset.release_unattempted(new_claim) != 1:
                    raise R4Error("reclaimed lease release failed")
                lease_receipt = fault_workset.audit_transition_receipts()
                distillation._append_ox_event(
                    root,
                    "ox-lease-recovery-receipts.jsonl",
                    {
                        "kind": "ox-lease-reclaim",
                        "profile_contract_id": str(
                            distillation.store.read_sealed(
                                distillation.store.distillation_dir(root) / "state.json"
                            )["profile_contract_id"]
                        ),
                        "source_commit": source["commit"],
                        "source_tree_sha256": source["tree_sha256"],
                        "source_ox_identity_sha256": source["ox_identity_sha256"],
                        "request_revision": distillation.OX_RAMP_REQUEST_REVISION,
                        "workset_receipt_generation": lease_receipt["generation"],
                        "workset_receipt_sha256": lease_receipt["head_sha256"],
                        "reclaimed": 1,
                        "leased_after": 0,
                        "captured_at": datetime.now(UTC).isoformat().replace(
                            "+00:00", "Z"
                        ),
                    },
                    unique_key="workset_receipt_sha256",
                )
            if scenario == "resource_pressure_preemption":
                # The preemption path is a real durable release, not merely a
                # provider error label: no lease may survive resource pressure.
                pressure_workset = DistillationWorkset(workset_path, migrate=False)
                pressure_claim = pressure_workset.claim("ox", 1, "owned-preempt", 60)
                if len(pressure_claim) != 1 or pressure_workset.release_unattempted(
                    pressure_claim
                ) != 1:
                    raise R4Error("owned resource preemption did not release work")
            receipt = DistillationWorkset(workset_path, migrate=False).audit_transition_receipts()
            status = DistillationWorkset(workset_path, migrate=False).status("ox")
            label_rows = distillation._read_chain(
                distillation.store.distillation_dir(root) / "label-ledger.jsonl"
            )
            label_work_ids = [
                str(row.get("work_id") or "")
                for row in label_rows
                if row.get("kind") == "teacher-label"
            ]
            duplicate_labels = sum(
                count - 1
                for count in Counter(label_work_ids).values()
                if count > 1
            )
            failure_rows = distillation._read_chain(
                distillation.store.distillation_dir(root) / "ox-failure-receipts.jsonl"
            )
            backoff_bounded = any(
                (
                    row.get("category") == "429"
                    and row.get("status") == "deferred"
                    and isinstance(row.get("before_cap"), int)
                    and not isinstance(row.get("before_cap"), bool)
                    and isinstance(row.get("after_cap"), int)
                    and not isinstance(row.get("after_cap"), bool)
                    and row["after_cap"] <= row["before_cap"]
                )
                or (
                    row.get("category") in {"5xx", "timeout"}
                    and row.get("bounded") is True
                    and row.get("status") == "deferred"
                )
                for row in failure_rows
            )
            event_heads = {
                "ramp": str(
                    distillation._ox_event_head(root, "ox-ramp-receipts.jsonl")[
                        "head_sha256"
                    ]
                ),
                "failure": str(
                    distillation._ox_event_head(root, "ox-failure-receipts.jsonl")[
                        "head_sha256"
                    ]
                ),
                "lease": str(
                    distillation._ox_event_head(
                        root, "ox-lease-recovery-receipts.jsonl"
                    )["head_sha256"]
                ),
            }
            state = distillation.store.read_sealed(
                distillation.store.distillation_dir(root) / "state.json"
            )
            contract_id = str(state.get("profile_contract_id") or "")
            if (
                _SHA.fullmatch(contract_id) is None
                or backend.network_calls != 0
                or (
                    categories[scenario] is not None
                    and _SHA.fullmatch(event_heads["failure"]) is None
                )
                or (
                    scenario == "lease_expiry_reclaim"
                    and _SHA.fullmatch(event_heads["lease"]) is None
                )
            ):
                raise R4Error("owned fault scenario did not reach a safe durable state")
            unsigned = {
                "schema": R4_FAULT_SCENARIO_SCHEMA,
                "namespace": "recall-distillation",
                "scenario": scenario,
                "writer_path": "public-run-distillation-chunk-v1",
                "test_only": True,
                "source": {
                    "source_commit": source["commit"],
                    "source_tree_sha256": source["tree_sha256"],
                    "source_ox_identity_sha256": source["ox_identity_sha256"],
                },
                "profile_contract_id": contract_id,
                "outcome": {
                    "profile_stopped": state.get("ox_profile_stopped") is True,
                    "backoff_bounded": backoff_bounded,
                    "quarantined": int(status["quarantined"]),
                    "ready": int(status["ready"]),
                    "leased": int(status["leased"]),
                    "duplicate_labels": duplicate_labels,
                    "adapter_calls": backend.adapter_calls,
                    "provider_calls": backend.network_calls,
                },
                "workset_receipt": {
                    "generation": receipt["generation"],
                    "head_sha256": receipt["head_sha256"],
                },
                "event_heads": event_heads,
                "owned_root": {
                    "before": before,
                    "after": after,
                    "run_status": result.get("status"),
                },
            }
            artifact_id = _sha256(unsigned)
            artifact = _sealed({"artifact_id": artifact_id, **unsigned})
            encoded = _json_bytes(artifact) + b"\n"
            path = _publish_owned_artifact(output, f"{artifact_id}.json", encoded)
            published.append(path)
    artifacts, inventory = _load_owned_fault_scenarios(output)
    result = _validate_owned_fault_scenarios(artifacts, source)
    if (
        result.get("passed") is not True
        or inventory.get("count") != len(PRODUCTION_FAULT_SCENARIOS)
        or len(published) != len(PRODUCTION_FAULT_SCENARIOS)
        or _assert_source(source_root, source_commit) != source
    ):
        raise R4Error("owned fault diagnostic readback failed")
    return published


@lru_cache(maxsize=32)
def _fresh_owned_fault_contract(
    source_root: Path, source_commit: str
) -> dict[str, Any]:
    """Re-run every provider-free fault through the public worker."""

    with tempfile.TemporaryDirectory(
        prefix="chronovisor-r4-owned-faults-",
        dir=Path(tempfile.gettempdir()).resolve(),
    ) as temp:
        output = Path(temp) / "artifacts"
        paths = run_owned_fault_scenarios(
            source_root=source_root, source_commit=source_commit, output=output
        )
        source = _assert_source(source_root, source_commit)
        artifacts, inventory = _load_owned_fault_scenarios(output)
        result = _validate_owned_fault_scenarios(artifacts, source)
        if (
            result.get("passed") is not True
            or inventory.get("count") != len(PRODUCTION_FAULT_SCENARIOS)
            or len(paths) != len(PRODUCTION_FAULT_SCENARIOS)
        ):
            raise R4Error("owned fault contract failed")
        return result


def _production_identity(
    *,
    source: Mapping[str, Any],
    state: Mapping[str, Any],
    config: Mapping[str, Any],
    config_sha256: str,
    contract: Mapping[str, Any],
    contract_sha256: str,
    workset: Mapping[str, Any],
    candidate: Mapping[str, Any],
    labels: Mapping[str, Any],
    root: Path,
    critical_modules: Mapping[str, str],
) -> set[str]:
    reasons: set[str] = set()
    if state.get("schema") != "chronovisor.recall-distillation.v1":
        reasons.add("production_state_schema_invalid")
    if state.get("kind") != "worker-state":
        reasons.add("production_state_kind_invalid")
    distillation = (
        config.get("recall", {}).get("distillation")
        if isinstance(config.get("recall"), Mapping)
        else None
    )
    if not isinstance(distillation, Mapping):
        reasons.add("production_config_distillation_missing")
        distillation = {}
    required_config = {
        "enabled": True,
        "teacher_profile": OX_PROFILE,
        "teacher_max_inflight": 10,
        "teacher_claim_limit": 1,
        "ox_enabled": True,
        "ox_free_only": False,
    }
    for key, expected in required_config.items():
        if distillation.get(key) != expected:
            reasons.add(f"production_config_{key}_invalid")
    relevant_config = {
        key: distillation.get(key)
        for key in (
            "teacher_profile",
            "teacher_max_inflight",
            "teacher_claim_limit",
            "ox_enabled",
            "ox_free_only",
            "ox_expires_at",
            "max_input_bytes",
            "max_candidates",
        )
    }
    if not isinstance(contract.get("relevant_config_sha256"), str) or contract.get(
        "relevant_config_sha256"
    ) != _sha256(relevant_config):
        reasons.add("production_contract_config_mismatch")
    if state.get("source_commit") != source.get("commit") or state.get(
        "source_tree_sha256"
    ) != source.get("tree_sha256"):
        reasons.add("production_state_source_mismatch")
    if source.get("account_uid") != ACCOUNT_UID or source.get("account_home") != str(
        ACCOUNT_HOME
    ):
        reasons.add("production_account_identity_invalid")
    runtime = state.get("runtime_identity")
    if not isinstance(runtime, Mapping):
        reasons.add("production_runtime_identity_missing")
        runtime = {}
    expected_runtime = {
        "root": str(root.absolute()),
        "account_uid": source.get("account_uid"),
        "account_home": source.get("account_home"),
        "source_commit": source.get("commit"),
        "source_tree_sha256": source.get("tree_sha256"),
        "source_ox_identity_sha256": source.get("ox_identity_sha256"),
        "config_sha256": config_sha256,
        "workset_sha256": workset.get("sha256"),
        "profile_contract_sha256": contract_sha256,
        "workset_receipt_head": workset.get("receipts", {}).get("head_sha256"),
        "candidate_checkpoint_head": candidate.get("head_sha256"),
        "candidate_checkpoint_records": candidate.get("records"),
        "candidate_checkpoint_file_state": _stable_candidate_file_state(
            candidate.get("file_state")
        ),
        "candidate_anchor_artifact_id": candidate.get("anchor_artifact_id"),
        "candidate_anchor_file_state": candidate.get("anchor_file_state"),
        "candidate_anchor_r0_artifact_id": candidate.get("anchor_r0_artifact_id"),
        "candidate_anchor_r0_file_sha256": candidate.get("anchor_r0_file_sha256"),
        "candidate_anchor_bootstrap_source_commit": candidate.get(
            "anchor_bootstrap_source_commit"
        ),
        "candidate_anchor_critical_module_sha256": candidate.get(
            "anchor_critical_module_sha256"
        ),
        "candidate_anchor_head_sha256": candidate.get("anchor_head_sha256"),
        "candidate_anchor_records": candidate.get("anchor_records"),
        "candidate_anchor_bytes": candidate.get("anchor_bytes"),
        "candidate_tail_records": candidate.get("tail_records"),
        "candidate_tail_bytes": candidate.get("tail_bytes"),
        "label_receipt_head": labels.get("head_sha256"),
        "label_checkpoint_records": labels.get("count"),
        "label_checkpoint_file_state": labels.get("checkpoint_file_state"),
    }
    if critical_modules:
        expected_runtime["critical_module_sha256"] = dict(critical_modules)
        expected_runtime["candidate_anchor_critical_module_sha256"] = dict(
            critical_modules
        )
    for key, expected in expected_runtime.items():
        if runtime.get(key) != expected:
            reasons.add(f"production_runtime_{key}_mismatch")
    archive_commit = runtime.get("archive_commit")
    expected_commit = runtime.get("expected_commit")
    direct_url = runtime.get("direct_url")
    module_path = runtime.get("module_path")
    archive_path = runtime.get("archive_path")
    direct_vcs = direct_url.get("vcs_info") if isinstance(direct_url, Mapping) else None
    if (
        archive_commit != source.get("commit")
        or expected_commit != source.get("commit")
        or runtime.get("drift") is not False
        or not isinstance(direct_url, Mapping)
        or runtime.get("direct_url_sha256") != _sha256(direct_url)
        or not isinstance(direct_vcs, Mapping)
        or direct_vcs.get("commit_id") != source.get("commit")
        or not isinstance(archive_path, str)
        or not isinstance(module_path, str)
        or Path(module_path).name != "runtime_config.py"
        or not Path(module_path).is_relative_to(Path(archive_path))
    ):
        reasons.add("production_runtime_archive_binding_invalid")
    os_identity = runtime.get("os_identity")
    if not isinstance(os_identity, Mapping):
        reasons.add("production_os_identity_missing")
    else:
        # This is observational, not a caller-provided attestation: the value
        # is compared to the current interpreter's OS view at collection time.
        observed = {
            "system": os.uname().sysname,
            "release": os.uname().release,
            "machine": os.uname().machine,
        }
        if dict(os_identity) != observed:
            reasons.add("production_os_identity_mismatch")
    profile_contract = {
        "profile": OX_PROFILE,
        "cohort": OX_COHORT,
        "route": OX_ROUTE,
        "endpoint": OX_ENDPOINT,
        "request_model": OX_MODEL,
        "required_returned_model": OX_MODEL,
        "request_revision": OX_REQUEST_REVISION,
        "fixed_identity": OX_FIXED_IDENTITY,
        "free_only": False,
        "no_paid_fallback": True,
        "kill_categories": list(OX_KILL_CATEGORIES),
        "max_inflight": 10,
        "teacher_claim_limit": 1,
        "live_recall_model_calls": 0,
        "source_commit": source.get("commit"),
        "source_tree_sha256": source.get("tree_sha256"),
        "source_ox_identity_sha256": source.get("ox_identity_sha256"),
    }
    expected_contract_keys = {
        "schema",
        "namespace",
        "artifact_id",
        "kind",
        "profile",
        "cohort",
        "route",
        "endpoint",
        "request_model",
        "required_returned_model",
        "request_revision",
        "fixed_identity",
        "free_only",
        "no_paid_fallback",
        "official_status",
        "expires_at",
        "docs_url",
        "kill_categories",
        "max_inflight",
        "teacher_claim_limit",
        "live_recall_model_calls",
        "source_commit",
        "source_tree_sha256",
        "source_ox_identity_sha256",
        "relevant_config_sha256",
        "seal_sha256",
    }
    if set(contract) != expected_contract_keys:
        reasons.add("production_contract_schema_invalid")
    for key, expected in profile_contract.items():
        if contract.get(key) != expected:
            reasons.add(f"production_contract_{key}_invalid")
    contract_unsigned = {
        key: value
        for key, value in contract.items()
        if key not in {"artifact_id", "seal_sha256"}
    }
    if contract.get("artifact_id") != _sha256(contract_unsigned):
        reasons.add("production_contract_digest_invalid")
    if contract.get("max_inflight") != 10:
        reasons.add("production_contract_max_inflight_invalid")
    canonical_expiry = _canonical_future_expiry(contract.get("expires_at"))
    if canonical_expiry is None or contract.get("expires_at") != canonical_expiry:
        reasons.add("production_contract_expired_or_missing")
    contract_id = _text(state.get("profile_contract_id"))
    if (
        _SHA.fullmatch(contract_id) is None
        or contract.get("artifact_id") != contract_id
        or workset.get("provenance", {}).get("profile_contract_id") != contract_id
    ):
        reasons.add("production_profile_contract_binding_invalid")
    return reasons


def _production_payload_source_matches(
    row: Mapping[str, Any],
    *,
    rallies: Mapping[str, Mapping[str, Any]],
    snapshots: Mapping[str, Mapping[str, Any]],
) -> bool:
    """Reuse the materializer's source projection for production labels."""

    rally_id = row.get("rally_id")
    candidate_id = row.get("candidate_id")
    if (
        not isinstance(rally_id, str)
        or not rally_id
        or not isinstance(candidate_id, str)
        or not candidate_id
    ):
        return False
    rally = rallies.get(rally_id)
    snapshot = snapshots.get(rally_id)
    if not isinstance(rally, Mapping) or not isinstance(snapshot, Mapping):
        return False
    candidates = snapshot.get("candidates")
    if not isinstance(candidates, list):
        return False
    matches = [
        candidate
        for candidate in candidates
        if isinstance(candidate, Mapping)
        and candidate.get("candidate_id") == candidate_id
    ]
    if len(matches) != 1:
        return False
    candidate = matches[0]
    candidate_text_sha256 = candidate.get("text_sha256")
    if not isinstance(candidate_text_sha256, str) or _SHA.fullmatch(candidate_text_sha256) is None:
        return False
    try:
        from chronovisor.recall.recall_distillation import (
            _materialization_payload_source_matches,
        )
    except (ImportError, AttributeError):
        return False
    assignment = row.get("assignment")
    return _materialization_payload_source_matches(
        row.get("payload_source"),
        rally_id=rally_id,
        candidate_id=candidate_id,
        # The remote producer deliberately excludes conversation context;
        # verify that closed payload shape instead of the richer local rally.
        rally={**rally, "context_refs": []},
        snapshot_sha256=str(snapshot.get("snapshot_sha256") or ""),
        candidate_text_sha256=candidate_text_sha256,
        assignment=assignment if isinstance(assignment, Mapping) else None,
    )


def _production_quality(
    *,
    state: Mapping[str, Any],
    workset: Mapping[str, Any],
    labels: Mapping[str, Any],
    events: Mapping[str, Sequence[Mapping[str, Any]]],
    source: Mapping[str, Any],
    contract: Mapping[str, Any],
    contract_id: str,
    candidate_rows: Sequence[Mapping[str, Any]] | None = None,
    rallies: Mapping[str, Mapping[str, Any]] | None = None,
    historical_contract_ids: frozenset[str] = frozenset(),
) -> tuple[set[str], dict[str, Any]]:
    reasons: set[str] = set()
    contract_identity = {
        "profile": OX_PROFILE,
        "cohort": OX_COHORT,
        "route": OX_ROUTE,
        "endpoint": OX_ENDPOINT,
        "request_model": OX_MODEL,
        "required_returned_model": OX_MODEL,
        "request_revision": OX_REQUEST_REVISION,
        "fixed_identity": OX_FIXED_IDENTITY,
        "free_only": False,
        "no_paid_fallback": True,
        "kill_categories": list(OX_KILL_CATEGORIES),
        "max_inflight": 10,
        "teacher_claim_limit": 1,
        "live_recall_model_calls": 0,
        "source_commit": source.get("commit"),
        "source_tree_sha256": source.get("tree_sha256"),
        "source_ox_identity_sha256": source.get("ox_identity_sha256"),
    }
    for key, expected in contract_identity.items():
        if contract.get(key) != expected:
            reasons.add(f"production_contract_{key}_invalid")
    contract_unsigned = {
        key: value
        for key, value in contract.items()
        if key not in {"artifact_id", "seal_sha256"}
    }
    if (
        contract.get("artifact_id") != contract_id
        or contract.get("artifact_id") != _sha256(contract_unsigned)
    ):
        reasons.add("production_profile_contract_binding_invalid")
    canonical_contract_expiry = _canonical_future_expiry(contract.get("expires_at"))
    if (
        canonical_contract_expiry is None
        or contract.get("expires_at") != canonical_contract_expiry
    ):
        reasons.add("production_contract_expired_or_missing")
    label_rows = labels.get("rows")
    if not isinstance(label_rows, list) or not label_rows:
        return {"production_labels_missing"}, {"stages": {}}
    current_label_rows: list[tuple[int, Mapping[str, Any]]] = []
    for label_index, row in enumerate(label_rows):
        if not isinstance(row, Mapping):
            reasons.add("production_label_invalid")
        elif row.get("profile_contract_id") == contract_id:
            current_label_rows.append((label_index, row))
        elif (
            row.get("profile_contract_id") not in historical_contract_ids
            and not _production_pre_profile_local_label(row)
        ):
            reasons.add("production_label_identity_invalid")
    if not current_label_rows:
        return reasons | {"production_labels_missing"}, {"stages": {}}
    completed = workset.get("completed")
    if not isinstance(completed, Mapping):
        return {"production_completed_inventory_missing"}, {"stages": {}}
    work_items = workset.get("items")
    if not isinstance(work_items, Mapping):
        return {"production_workset_items_missing"}, {"stages": {}}
    candidate_snapshots: dict[str, Mapping[str, Any]] = {}
    if candidate_rows is not None:
        for candidate_row in candidate_rows:
            snapshot = candidate_row.get("snapshot")
            rally_id = candidate_row.get("rally_id")
            if (
                not isinstance(rally_id, str)
                or not isinstance(snapshot, Mapping)
                or snapshot.get("rally_id") != rally_id
                or rally_id in candidate_snapshots
            ):
                reasons.add("production_candidate_source_invalid")
                continue
            candidate_snapshots[rally_id] = snapshot
    label_by_digest: dict[str, Mapping[str, Any]] = {}
    label_by_work: dict[str, Mapping[str, Any]] = {}
    label_ids: set[str] = set()
    commit_ids: set[str] = set()
    label_index_by_digest: dict[str, int] = {}
    stage_work_units: dict[int, list[dict[str, Any]]] = {
        cap: [] for cap in PRODUCTION_RAMP_CAPS
    }
    for label_index, row in current_label_rows:
        digest = _text(row.get("record_sha256"))
        work_id = _text(row.get("work_id"))
        # The production label writer's durable identity is the chained
        # record digest.  Older synthetic fixtures carried redundant
        # label_id/commit_id fields; derive both aliases from that immutable
        # digest when absent instead of requiring a field the writer never
        # publishes.
        label_id = _text(row.get("label_id")) or digest
        commit_id = _text(row.get("commit_id")) or digest
        identity = row.get("route_identity")
        if (
            row.get("kind") != "teacher-label"
            or row.get("status") != "completed"
            or row.get("profile") != OX_PROFILE
            or row.get("cohort") != OX_COHORT
            or row.get("profile_contract_id") != contract_id
            or row.get("source_commit") != source.get("commit")
            or row.get("source_tree_sha256") != source.get("tree_sha256")
            or row.get("route") != OX_ROUTE
            or row.get("teacher_role") != "recall.distill.teacher.deepseek-v4-flash"
            or row.get("identity_revision") != OX_IDENTITY_REVISION
            or row.get("request_revision") != OX_REQUEST_REVISION
            or row.get("request_revision") != contract.get("request_revision")
            or row.get("expires_at") != contract.get("expires_at")
            or _canonical_future_expiry(row.get("expires_at")) is None
            or row.get("expires_at")
            != _canonical_future_expiry(row.get("expires_at"))
            or row.get("route_digest") != OX_ROUTE_SHA256
            or row.get("model_digest") != OX_MODEL_SHA256
            or row.get("prompt_sha256") != OX_PROMPT_SHA256
            or row.get("schema_sha256") != OX_SCHEMA_SHA256
            or row.get("test_only") is not False
            or row.get("source_ox_identity_sha256") != source.get("ox_identity_sha256")
            or not isinstance(identity, Mapping)
            or dict(identity)
            != {"provider": "opencode-go", "model": OX_ROUTE, "location": "remote"}
            or _SHA.fullmatch(digest) is None
            or _SHA.fullmatch(work_id) is None
            or _SHA.fullmatch(label_id) is None
            or _SHA.fullmatch(commit_id) is None
            or not isinstance(row.get("payload_source"), Mapping)
            or _SHA.fullmatch(_text(row.get("payload_digest"))) is None
            or _sha256(row.get("payload_source")) != row.get("payload_digest")
            or (
                candidate_rows is not None
                and (
                    rallies is None
                    or not _production_payload_source_matches(
                        row,
                        rallies=rallies,
                        snapshots=candidate_snapshots,
                    )
                )
            )
            or work_id
            != _sha256(
                {
                    "kind": "ox-teacher-label-v1",
                    "profile": OX_PROFILE,
                    "cohort": OX_COHORT,
                    "route": OX_ROUTE,
                    "profile_contract_id": contract_id,
                    "payload_digest": row.get("payload_digest"),
                }
            )
            or row.get("request_sha256")
            != _expected_ox_request_sha256(
                profile_contract_id=contract_id,
                payload_digest=_text(row.get("payload_digest")),
            )
            or row.get("provider_request_sha256")
            != _expected_ox_provider_request_sha256(
                profile_contract_id=contract_id,
                payload_digest=_text(row.get("payload_digest")),
                work_id=work_id,
                expires_at=_text(contract.get("expires_at")),
            )
            or _SHA.fullmatch(_text(row.get("provider_receipt_sha256"))) is None
            or row.get("provider_receipt_sha256")
            == _expected_ox_provider_request_sha256(
                profile_contract_id=contract_id,
                payload_digest=_text(row.get("payload_digest")),
                work_id=work_id,
                expires_at=_text(contract.get("expires_at")),
            )
            or "provider_response_request_sha256" in row
            or digest in label_by_digest
            or work_id in label_by_work
            or label_id in label_ids
            or commit_id in commit_ids
        ):
            reasons.add("production_label_identity_invalid")
        if (
            _SHA.fullmatch(digest) is None
            or _SHA.fullmatch(work_id) is None
            or _SHA.fullmatch(label_id) is None
            or _SHA.fullmatch(commit_id) is None
        ):
            continue
        label_by_digest[digest] = row
        label_by_work[work_id] = row
        label_ids.add(label_id)
        commit_ids.add(commit_id)
        label_index_by_digest[digest] = label_index
        work = completed.get(work_id)
        work_item = work_items.get(work_id)
        if (
            not isinstance(work, Mapping)
            or work.get("completion_ref") != f"label-ledger:{digest}"
            or work.get("completion_digest") != digest
            or isinstance(work.get("attempt_count"), bool)
            or not isinstance(work.get("attempt_count"), int)
            or work.get("attempt_count", 0) < 1
            or row.get("attempt_count") != work.get("attempt_count")
            or not isinstance(work_item, Mapping)
            or work_item.get("payload_digest") != row.get("payload_digest")
            or work_item.get("provenance")
            != {
                "cohort": OX_COHORT,
                "profile": OX_PROFILE,
                "profile_contract_id": contract_id,
                "route": OX_ROUTE,
            }
        ):
            reasons.add("production_workset_label_binding_invalid")
        cap = row.get("ramp_cap")
        if (
            isinstance(cap, bool)
            or not isinstance(cap, int)
            or cap not in PRODUCTION_RAMP_CAPS
        ):
            reasons.add("production_label_ramp_cap_missing")
        else:
            if (
                isinstance(work, Mapping)
                and work.get("completion_ref") == f"label-ledger:{digest}"
                and work.get("completion_digest") == digest
                and isinstance(work.get("attempt_count"), int)
                and not isinstance(work.get("attempt_count"), bool)
                and work["attempt_count"] >= 1
            ):
                stage_work_units[cap].append(
                    {
                        "work_id": work_id,
                        "attempt_count": work["attempt_count"],
                        "label_record_sha256": digest,
                        "label_index": label_index,
                        "provider_receipt_sha256": _text(
                            row.get("provider_receipt_sha256")
                        ),
                    }
                )
    if set(label_by_work) != set(completed):
        reasons.add("production_completed_label_set_mismatch")
    success_attempts: dict[tuple[str, int], str] = {}
    provider_receipts_by_work: set[tuple[str, str]] = set()
    for units in stage_work_units.values():
        for unit in units:
            attempt_key = (str(unit["work_id"]), int(unit["attempt_count"]))
            receipt = str(unit["provider_receipt_sha256"])
            if (
                attempt_key in success_attempts
                or (attempt_key[0], receipt) in provider_receipts_by_work
            ):
                reasons.add("production_provider_attempt_duplicate")
                continue
            success_attempts[attempt_key] = receipt
            provider_receipts_by_work.add((attempt_key[0], receipt))
    failure_units_by_cap: dict[int, list[dict[str, Any]]] = {
        cap: [] for cap in PRODUCTION_RAMP_CAPS
    }
    transitions = events.get("failure")
    if not isinstance(transitions, Sequence):
        reasons.add("production_failure_receipts_missing")
        transitions = []
    failure_receipt_records: dict[str, int] = {}
    for transition in transitions:
        if not isinstance(transition, Mapping):
            reasons.add("production_failure_receipt_invalid")
            continue
        cap = transition.get("cap")
        work_ids = transition.get("work_ids")
        attempts_by_work = transition.get("attempts_by_work")
        provider_receipts = transition.get("provider_receipts")
        transition_attempts = transition.get("attempts")
        if (
            isinstance(cap, bool)
            or cap not in PRODUCTION_RAMP_CAPS
            or not isinstance(work_ids, list)
            or isinstance(transition.get("record_index"), bool)
            or not isinstance(transition.get("record_index"), int)
            or transition.get("record_index", 0) < 1
            or _SHA.fullmatch(_text(transition.get("record_sha256"))) is None
            or not isinstance(provider_receipts, Mapping)
            or isinstance(transition_attempts, bool)
            or not isinstance(transition_attempts, int)
            or not isinstance(attempts_by_work, Mapping)
            or set(attempts_by_work) != set(work_ids)
            or set(provider_receipts) != set(work_ids)
            or not all(
                isinstance(value, int) and not isinstance(value, bool) and value > 0
                for value in attempts_by_work.values()
            )
            or not all(
                isinstance(value, str) and _SHA.fullmatch(value)
                for value in provider_receipts.values()
            )
            or len(set(provider_receipts.values())) != 1
            or transition_attempts != len(set(provider_receipts.values()))
        ):
            reasons.add("production_failure_receipt_invalid")
            continue
        failure_receipt = next(iter(provider_receipts.values()))
        record_index = transition["record_index"]
        previous_record = failure_receipt_records.setdefault(
            failure_receipt, record_index
        )
        if previous_record != record_index:
            reasons.add("production_provider_receipt_reused")
            continue
        for work_id in work_ids:
            attempt = attempts_by_work.get(work_id)
            failure_receipt = provider_receipts.get(work_id)
            if (
                not isinstance(work_id, str)
                or _SHA.fullmatch(work_id) is None
                or isinstance(attempt, bool)
                or not isinstance(attempt, int)
                or attempt < 1
                or not isinstance(failure_receipt, str)
                or _SHA.fullmatch(failure_receipt) is None
            ):
                reasons.add("production_failure_receipt_invalid")
                continue
            attempt_key = (work_id, attempt)
            if (
                attempt_key in success_attempts
                or (work_id, failure_receipt) in provider_receipts_by_work
            ):
                reasons.add("production_provider_attempt_duplicate")
                continue
            completed_label = label_by_work.get(work_id)
            if (
                completed_label is not None
                and completed_label.get("attempt_count") == attempt
                and completed_label.get("provider_receipt_sha256") != failure_receipt
            ):
                reasons.add("production_provider_receipt_binding_invalid")
                continue
            success_attempts[attempt_key] = failure_receipt
            provider_receipts_by_work.add((work_id, failure_receipt))
            failure_units_by_cap[int(cap)].append(
                {
                    "work_id": work_id,
                    "attempt_count": attempt,
                    "provider_receipt_sha256": failure_receipt,
                    "record_index": transition.get("record_index"),
                }
            )
    legacy_events = events.get("legacy")
    if isinstance(legacy_events, Sequence) and legacy_events:
        reasons.add("production_legacy_ox_events_noncertifying")
    ramp = events.get("ramp")
    if not isinstance(ramp, Sequence) or len(ramp) < len(PRODUCTION_RAMP_CAPS):
        reasons.add("production_ramp_receipts_missing")
        ramp = []
    accepted_ramp: list[tuple[int, Mapping[str, Any]]] = []
    if ramp:
        expected_path = ((1, 2), (2, 5), (5, 10), (10, 10))
        cursor = len(ramp)
        selected_reverse: list[tuple[int, Mapping[str, Any]]] = []
        for expected_cap, expected_next in reversed(expected_path):
            found: tuple[int, Mapping[str, Any]] | None = None
            for index in range(cursor - 1, -1, -1):
                candidate = ramp[index]
                if (
                    isinstance(candidate, Mapping)
                    and candidate.get("cap") == expected_cap
                    and candidate.get("next_cap") == expected_next
                ):
                    found = (index, candidate)
                    break
            if found is None:
                break
            selected_reverse.append(found)
            cursor = found[0]
        if len(selected_reverse) != len(expected_path):
            reasons.add("production_ramp_requalification_path_invalid")
        else:
            accepted_ramp = list(reversed(selected_reverse))
    stages: dict[str, Any] = {}
    seen_caps: set[int] = set()
    ramp_caps: list[int] = []
    response_cap_by_receipt: dict[str, int] = {}
    assigned_failure_record_indices: set[int] = set()
    accepted_failure_record_indices: set[int] = set()
    for ramp_index, row in accepted_ramp:
        previous_label_count = 0
        previous_failure_count = 0
        if ramp_index:
            prior = ramp[ramp_index - 1]
            if not isinstance(prior, Mapping):
                reasons.add("production_ramp_requalification_path_invalid")
                continue
            prior_label_count = prior.get("label_count")
            prior_failure_count = prior.get("failure_record_count")
            if (
                isinstance(prior_label_count, bool)
                or not isinstance(prior_label_count, int)
                or prior_label_count < 1
                or isinstance(prior_failure_count, bool)
                or not isinstance(prior_failure_count, int)
                or prior_failure_count < 0
            ):
                reasons.add("production_ramp_requalification_path_invalid")
                continue
            previous_label_count = prior_label_count
            previous_failure_count = prior_failure_count
        if not isinstance(row, Mapping):
            reasons.add("production_ramp_receipt_invalid")
            continue
        cap = row.get("cap")
        if isinstance(cap, int) and not isinstance(cap, bool):
            ramp_caps.append(cap)
        if (
            isinstance(cap, bool)
            or not isinstance(cap, int)
            or cap not in PRODUCTION_RAMP_CAPS
            or cap in seen_caps
            or row.get("source_commit") != source.get("commit")
            or row.get("profile_contract_id") != contract_id
        ):
            reasons.add("production_ramp_quality_invalid")
            continue
        label_head = _text(row.get("label_head_sha256"))
        head_index = label_index_by_digest.get(label_head)
        label_count = row.get("label_count")
        if (
            head_index is None
            or isinstance(label_count, bool)
            or not isinstance(label_count, int)
            or label_count != head_index + 1
            or label_count <= previous_label_count
        ):
            reasons.add("production_ramp_label_checkpoint_invalid")
            continue
        head_row = label_by_digest[label_head]
        if head_row.get("profile_contract_id") != contract_id or head_row.get("ramp_cap") != cap:
            reasons.add("production_ramp_quality_invalid")
            continue
        failure_count = row.get("failure_record_count")
        failure_head = row.get("failure_head_sha256")
        if (
            isinstance(failure_count, bool)
            or not isinstance(failure_count, int)
            or failure_count < 0
            or failure_count < previous_failure_count
            or (failure_count == 0 and failure_head != "")
            or (failure_count > 0 and (
                not isinstance(failure_head, str)
                or _SHA.fullmatch(failure_head) is None
                or failure_count > len(transitions)
                or not isinstance(transitions[failure_count - 1], Mapping)
                or transitions[failure_count - 1].get("record_sha256") != failure_head
                or transitions[failure_count - 1].get("record_index") != failure_count
            ))
        ):
            reasons.add("production_ramp_failure_checkpoint_invalid")
            continue
        if cap == 10 and (
            label_count != len(label_rows) or failure_count != len(transitions)
        ):
            reasons.add("production_ramp_terminal_checkpoint_incomplete")
            continue
        units = [
            unit
            for unit in stage_work_units[cap]
            if previous_label_count <= unit["label_index"] <= head_index
        ]
        success_by_receipt: dict[str, dict[str, Any]] = {}
        for unit in units:
            receipt = str(unit["provider_receipt_sha256"])
            prior_cap = response_cap_by_receipt.get(receipt)
            if prior_cap is not None and prior_cap != cap:
                reasons.add("production_provider_receipt_cross_cap")
                continue
            response_cap_by_receipt[receipt] = cap
            success_by_receipt.setdefault(receipt, unit)
        failure_by_receipt: dict[str, dict[str, Any]] = {}
        for unit in failure_units_by_cap[cap]:
            if not previous_failure_count < unit["record_index"] <= failure_count:
                continue
            accepted_failure_record_indices.add(int(unit["record_index"]))
            assigned_failure_record_indices.add(int(unit["record_index"]))
            receipt = str(unit["provider_receipt_sha256"])
            prior_cap = response_cap_by_receipt.get(receipt)
            if prior_cap is not None and prior_cap != cap:
                reasons.add("production_provider_receipt_cross_cap")
                continue
            response_cap_by_receipt[receipt] = cap
            if receipt in success_by_receipt:
                reasons.add("production_provider_receipt_binding_invalid")
                continue
            failure_by_receipt.setdefault(receipt, unit)
        valid = len(success_by_receipt)
        attempts = valid + len(failure_by_receipt)
        work_ids = [str(unit["work_id"]) for unit in success_by_receipt.values()]
        if (
            valid < 20
            or attempts < valid
            or valid / attempts < 0.95
            or len(set(work_ids)) != len(work_ids)
        ):
            reasons.add("production_ramp_quality_invalid")
            continue
        # Event counters are attested audit values only.  Certification is
        # derived from sealed labels and their completed Workset units above.
        if (
            row.get("valid_receipts") != valid
            or row.get("attempts") != attempts
            or row.get("work_ids") != work_ids
        ):
            reasons.add("production_ramp_event_audit_mismatch")
        seen_caps.add(cap)
        stages[str(cap)] = {
            "valid_receipts": valid,
            "attempts": attempts,
            "valid_rate": round(valid / attempts, 8),
            "work_ids": list(work_ids),
        }
    if seen_caps != set(PRODUCTION_RAMP_CAPS):
        reasons.add("production_ramp_stages_incomplete")
    if ramp_caps != list(PRODUCTION_RAMP_CAPS):
        reasons.add("production_ramp_order_invalid")
    if assigned_failure_record_indices != accepted_failure_record_indices:
        reasons.add("production_failure_stage_assignment_invalid")
    if workset.get("counts", {}).get("leased") != 0:
        reasons.add("production_leased_work_present")
    receipt_state = workset.get("receipts")
    if isinstance(receipt_state, Mapping) and (
        receipt_state.get("status") != "verified"
        or receipt_state.get("legacy_unverified_excluded") is True
    ):
        reasons.add("production_workset_receipts_noncertifying")
    for key in (
        "sensitive",
        "raw",
        "billable",
        "unexpected_route",
        "duplicate_label",
        "duplicate_commit",
    ):
        value = state.get(key)
        if value is None:
            # The production projection does not materialize these legacy
            # counter aliases.  Their zero value is derived from the strict
            # row identity/route/duplicate checks above, never from a
            # caller-provided boolean.
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value != 0:
            reasons.add(f"production_{key}_veto_invalid")
    lease_events = events.get("lease")
    receipt_index = workset.get("receipts", {}).get("by_generation")
    seen_lease_receipts: set[str] = set()
    lease_invalid = not isinstance(lease_events, Sequence) or (
        bool(lease_events) and not isinstance(receipt_index, Mapping)
    )
    if not lease_invalid and isinstance(lease_events, Sequence):
        for row in lease_events:
            if not isinstance(row, Mapping):
                lease_invalid = True
                break
            generation = row.get("workset_receipt_generation")
            receipt_sha = row.get("workset_receipt_sha256")
            work_ids_sha = row.get("work_ids_sha256")
            receipt = receipt_index.get(str(generation))
            details = receipt.get("details") if isinstance(receipt, Mapping) else None
            if (
                isinstance(row.get("reclaimed"), bool)
                or not isinstance(row.get("reclaimed"), int)
                or row.get("reclaimed", -1) < 1
                or row.get("leased_after") != 0
                or isinstance(generation, bool)
                or not isinstance(generation, int)
                or not isinstance(receipt_sha, str)
                or _SHA.fullmatch(receipt_sha) is None
                or receipt_sha in seen_lease_receipts
                or not isinstance(receipt, Mapping)
                or receipt.get("receipt_sha256") != receipt_sha
                or receipt.get("operation") != "claim_reclaim"
                or not isinstance(details, Mapping)
                or details.get("count") != row.get("reclaimed")
                or not isinstance(work_ids_sha, str)
                or _SHA.fullmatch(work_ids_sha) is None
                or details.get("work_ids_sha256") != work_ids_sha
            ):
                lease_invalid = True
                break
            seen_lease_receipts.add(receipt_sha)
    if lease_invalid:
        reasons.add("production_lease_recovery_invalid")
    quality_gates = state.get("quality_gates")
    if not isinstance(quality_gates, Mapping):
        reasons.add("production_quality_gates_missing")
    else:
        negative_veto = quality_gates.get("negative_veto")
        if (
            not isinstance(negative_veto, Mapping)
            or negative_veto.get("authenticated") is not True
            or negative_veto.get("exact_binding") is not True
            or negative_veto.get("conflicts") != 0
        ):
            reasons.add("production_negative_veto_gate_invalid")
        blind_repeat = quality_gates.get("blind_repeat")
        if (
            not isinstance(blind_repeat, Mapping)
            or blind_repeat.get("revision") != OX_PROBE_REVISION
            or blind_repeat.get("complete") is not True
            or blind_repeat.get("stability_passed") is not True
            or not isinstance(blind_repeat.get("pairs"), int)
            or isinstance(blind_repeat.get("pairs"), bool)
            or blind_repeat.get("pairs", 0) < OX_MIN_BLIND_REPEAT_PAIRS
        ):
            reasons.add("production_blind_repeat_gate_invalid")
        order_swap = quality_gates.get("order_swap")
        if (
            not isinstance(order_swap, Mapping)
            or order_swap.get("complete") is not True
            or not isinstance(order_swap.get("pairs"), int)
            or isinstance(order_swap.get("pairs"), bool)
            or order_swap.get("pairs", 0) < OX_MIN_BLIND_REPEAT_PAIRS
        ):
            reasons.add("production_order_swap_gate_invalid")
        rollback = quality_gates.get("rollback")
        if (
            not isinstance(rollback, Mapping)
            or rollback.get("verified") is not True
            or rollback.get("active_unchanged") is not True
            or rollback.get("status") not in {"not_rolled_back", "rolled_back"}
        ):
            reasons.add("production_rollback_gate_invalid")
    transitions = events.get("failure")
    if not isinstance(transitions, Sequence):
        reasons.add("production_failure_receipts_missing")
        transitions = []
    for transition in transitions:
        if not isinstance(transition, Mapping):
            reasons.add("production_failure_receipt_invalid")
            continue
        category = _text(transition.get("category"))
        if category == "429":
            before_cap = transition.get("before_cap")
            after_cap = transition.get("after_cap")
            valid = (
                isinstance(before_cap, int)
                and not isinstance(before_cap, bool)
                and before_cap in PRODUCTION_RAMP_CAPS
                and before_cap > 1
                and isinstance(after_cap, int)
                and not isinstance(after_cap, bool)
                and after_cap in PRODUCTION_RAMP_CAPS
                and after_cap == max(1, before_cap // 2)
                and transition.get("status") == "deferred"
            )
        elif category in {"5xx", "timeout"}:
            valid = (
                isinstance(transition.get("attempts"), int)
                and not isinstance(transition.get("attempts"), bool)
                and 1 <= transition["attempts"] <= 3
                and transition.get("bounded") is True
                and transition.get("status") == "deferred"
            )
        elif category in {"402", "paid", "model_drift"}:
            valid = transition.get("status") == "hard_stop"
        else:
            reasons.add("production_failure_category_invalid")
            continue
        if not valid:
            reasons.add("production_failure_receipt_invalid")
    return reasons, {
        "stages": stages,
        "labels": len(current_label_rows),
        # OX exposes no provider-signed receipt.  These hashes are durable
        # adapter observations, bound by the event chain, immutable anchor,
        # exact runtime source, and single-writer Workset/label evidence.
        "receipt_authority": "adapter_observed_not_provider_signed",
    }


def _source_ox_identity(source_root: Path) -> dict[str, str]:
    """Recompute the fixed OX identity from the exact source checkout."""

    module_path = (
        source_root
        / "src"
        / "chronovisor"
        / "recall"
        / "recall_distillation_remote_teacher.py"
    )
    if _has_symlink_component(module_path) or not module_path.is_file():
        raise R4Error("source OX identity module is unavailable")
    source_path = str(source_root / "src")
    previous_path = list(sys.path)
    if source_path in sys.path:
        sys.path.remove(source_path)
    sys.path.insert(0, source_path)
    previous_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec = importlib.util.spec_from_file_location(
            "chronovisor_r4_source_remote_teacher", module_path
        )
        if spec is None or spec.loader is None:
            raise R4Error("source OX identity module cannot be loaded")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        identity = getattr(module, "OX_ALPHA_FIXED_IDENTITY", None)
        route_identity = (
            identity.get("route_identity") if isinstance(identity, Mapping) else None
        )
        values = {
            "identity_revision": identity.get("revision")
            if isinstance(identity, Mapping)
            else None,
            "provider": route_identity.get("provider")
            if isinstance(route_identity, Mapping)
            else None,
            "model": route_identity.get("model")
            if isinstance(route_identity, Mapping)
            else None,
            "location": route_identity.get("location")
            if isinstance(route_identity, Mapping)
            else None,
            "model_sha256": identity.get("model_digest")
            if isinstance(identity, Mapping)
            else None,
            "route_sha256": identity.get("route_digest")
            if isinstance(identity, Mapping)
            else None,
            "prompt_sha256": identity.get("prompt_template_sha256")
            if isinstance(identity, Mapping)
            else None,
            "schema_sha256": identity.get("schema_revision_sha256")
            if isinstance(identity, Mapping)
            else None,
        }
        if any(not isinstance(value, str) or not value for value in values.values()):
            raise R4Error("source OX identity is incomplete")
        return {key: str(value) for key, value in values.items()}
    except (ImportError, OSError, AttributeError, TypeError, ValueError) as exc:
        if isinstance(exc, R4Error):
            raise
        raise R4Error("source OX identity import failed") from exc
    finally:
        sys.dont_write_bytecode = previous_bytecode
        sys.path[:] = previous_path
        sys.modules.pop("chronovisor_r4_source_remote_teacher", None)


def _collect_authoritative_production(
    *,
    source_root: Path,
    source: Mapping[str, Any],
    production_root: Path,
) -> dict[str, Any]:
    """Collect production evidence only from the fixed managed root.

    This routine intentionally has no writer, provider, subprocess, or network
    path.  It reads the sealed state, profile contract, label chain, and SQLite
    workset directly, re-statting every input before returning.  A caller-supplied
    receipt bundle is not involved, so a copied JSON fixture cannot certify.
    """

    unavailable = {
        "passed": False,
        "reasons": [],
        "collector": "fixed-production-root-workset-v1",
        "provider_calls": 0,
        "root": str(PRODUCTION_ROOT.absolute()),
    }
    root_fd: int | None = None
    cwd_fd: int | None = None
    restore_error: OSError | None = None
    try:
        expected_original = PRODUCTION_ROOT.expanduser().absolute()
        original_root = production_root.expanduser().absolute()
        if _has_symlink_component(expected_original) or _has_symlink_component(
            original_root
        ):
            unavailable["reasons"] = ["production_root_unavailable"]
            return unavailable
        expected_root = expected_original.resolve(strict=False)
        if original_root.resolve(strict=False) != expected_root:
            unavailable["reasons"] = ["production_root_not_authoritative"]
            return unavailable
        if not production_root.is_dir():
            unavailable["reasons"] = ["production_root_unavailable"]
            return unavailable
        root_identity = _production_directory_identity(
            original_root, label="production root"
        )
        directory_flags = (
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        )
        cwd_fd = os.open(".", directory_flags)
        root_fd = os.open(str(original_root), directory_flags)
        # ponytail: this one-shot CLI uses process-global cwd; concurrent
        # embedding must move the same reads to openat(dir_fd=...) instead.
        os.fchdir(root_fd)
        collection_root = Path(".")
        if (
            _production_directory_fd_identity(root_fd, label="production root")
            != root_identity
        ):
            raise R4Error("production root changed while opening")
        if _has_symlink_component(source_root):
            unavailable["reasons"] = ["source_root_contains_symlink"]
            return unavailable
        supplied_commit = source.get("commit") if isinstance(source, Mapping) else None
        if not isinstance(supplied_commit, str) or _COMMIT.fullmatch(supplied_commit) is None:
            unavailable["reasons"] = ["source_identity_unavailable"]
            return unavailable
        audited_source = _assert_source(source_root, supplied_commit)
        if dict(source) != audited_source:
            unavailable["reasons"] = ["source_identity_mismatch"]
            return unavailable
        source = audited_source
        source_identity = _source_ox_identity(source_root)
        critical_modules = _production_critical_module_sha256(source_root)
        expected_identity = {
            "identity_revision": OX_IDENTITY_REVISION,
            "provider": "opencode-go",
            # The fixed source identity binds the fully-qualified route/model;
            # the profile contract's request_model remains the short model id.
            "model": OX_ROUTE,
            "location": "remote",
            "model_sha256": OX_MODEL_SHA256,
            "route_sha256": OX_ROUTE_SHA256,
            "prompt_sha256": OX_PROMPT_SHA256,
            "schema_sha256": OX_SCHEMA_SHA256,
        }
        if source_identity != expected_identity:
            raise R4Error("source OX identity does not match the fixed contract")
        distill = collection_root / PRODUCTION_DISTILLATION_RELATIVE
        state_path = collection_root / PRODUCTION_STATE_RELATIVE
        workset_path = collection_root / PRODUCTION_WORKSET_RELATIVE
        candidate_path = collection_root / PRODUCTION_CANDIDATE_RELATIVE
        candidate_checkpoint_path = (
            collection_root / PRODUCTION_CANDIDATE_CHECKPOINT_RELATIVE
        )
        label_path = collection_root / PRODUCTION_LABEL_RELATIVE
        label_checkpoint_path = collection_root / PRODUCTION_LABEL_CHECKPOINT_RELATIVE
        config_path = collection_root / PRODUCTION_CONFIG_RELATIVE
        state, state_file_state, state_sha256 = _production_json(
            state_path,
            label="production state",
            schema="chronovisor.recall-distillation.v1",
        )
        if _has_symlink_component(distill) or not distill.is_dir():
            raise R4Error("production distillation directory is unavailable")
        workset = _production_workset(workset_path)
        candidate = _production_ledger_checkpoint(
            candidate_path,
            candidate_checkpoint_path,
            ledger_name="candidate-ledger.jsonl",
        )
        candidate_anchor = _load_production_anchor(original_root, source=source)
        candidate_tail = _production_candidate_tail(
            candidate_path, candidate, candidate_anchor
        )
        candidate = {**candidate, **candidate_tail}
        candidate_view = _production_chain(
            candidate_path,
            candidate_checkpoint_path,
            ledger_name="candidate-ledger.jsonl",
        )
        if (
            candidate_view["count"] != candidate["records"]
            or candidate_view["head_sha256"] != candidate["head_sha256"]
            or candidate_view["file_state"] != candidate["ledger_state"]
        ):
            raise R4Error("production candidate ledger checkpoint is inconsistent")
        completed_candidate_rows = _production_workset_candidate_binding(
            workset, candidate, candidate_path
        )
        labels = _production_chain(label_path, label_checkpoint_path)
        try:
            from chronovisor.recall.recall_distillation import _materialization_rallies

            production_rallies = _materialization_rallies(original_root, None)
        except (ImportError, AttributeError, OSError, R4Error) as exc:
            raise R4Error("production rally source is unavailable") from exc
        config_bytes, config_state, config_sha256 = _production_file_bytes(
            config_path, label="production config", max_bytes=_MAX_RECEIPT_BYTES
        )
        if tomllib is None:
            raise R4Error("production config parser is unavailable")
        try:
            config = tomllib.loads(config_bytes.decode("utf-8"))
        except (ValueError, UnicodeError) as exc:
            raise R4Error("production config TOML is invalid") from exc
        if not isinstance(config, Mapping):
            raise R4Error("production config is not an object")
        contract_id = _text(state.get("profile_contract_id"))
        if _SHA.fullmatch(contract_id) is None:
            raise R4Error("production profile contract id is invalid")
        contract_path = (
            collection_root / PRODUCTION_CONTRACT_DIR_RELATIVE / f"{contract_id}.json"
        )
        contract, contract_state, contract_sha256 = _production_json(
            contract_path,
            label="production profile contract",
            schema="chronovisor.recall-distill-remote-profile.v2",
        )
        contract_unsigned = {
            key: value
            for key, value in contract.items()
            if key not in {"artifact_id", "seal_sha256"}
        }
        if contract.get("artifact_id") != contract_id or contract_id != _sha256(
            contract_unsigned
        ):
            raise R4Error("production profile contract identity mismatch")
        historical_contract_ids: set[str] = set()
        for row in labels["rows"]:
            if not isinstance(row, Mapping):
                continue
            historical_id = _text(row.get("profile_contract_id"))
            if historical_id == contract_id or _SHA.fullmatch(historical_id) is None:
                continue
            _production_historical_profile_contract(
                collection_root
                / PRODUCTION_CONTRACT_DIR_RELATIVE
                / f"{historical_id}.json",
                contract_id=historical_id,
            )
            historical_contract_ids.add(historical_id)
        events = _production_ox_events(
            original_root,
            source=source,
            contract_id=contract_id,
            workset=workset,
        )
        identity_reasons = _production_identity(
            source=source,
            state=state,
            config=config,
            config_sha256=config_sha256,
            contract=contract,
            contract_sha256=contract_sha256,
            workset=workset,
            candidate=candidate,
            labels=labels,
            root=original_root,
            critical_modules=critical_modules,
        )
        quality_reasons, quality = _production_quality(
            state=state,
            workset=workset,
            labels=labels,
            events=events,
            source=source,
            contract=contract,
            contract_id=contract_id,
            candidate_rows=completed_candidate_rows,
            rallies=production_rallies,
            historical_contract_ids=frozenset(historical_contract_ids),
        )
        # Fault injection belongs to the explicit, provider-free owned-clone
        # source contract.  Requiring its test-only artifacts under the live
        # root made this read-only collector impossible to certify safely.
        reasons = identity_reasons | quality_reasons
        if _production_stat(state_path, label="production state") != state_file_state:
            raise R4Error("production state changed during validation")
        if _production_stat(config_path, label="production config") != config_state:
            raise R4Error("production config changed during validation")
        if (
            _production_stat(contract_path, label="production profile contract")
            != contract_state
        ):
            raise R4Error("production profile contract changed during validation")
        if (
            _production_stat(candidate_path, label="production candidate-ledger")
            != candidate["ledger_state"]
            or _production_stat(
                candidate_checkpoint_path,
                label="production candidate-ledger checkpoint",
            )
            != candidate["checkpoint_state"]
        ):
            raise R4Error("production candidate ledger changed during validation")
        if candidate_anchor.get("file_state") is not None and (
            _production_stat(
                collection_root / PRODUCTION_CANDIDATE_ANCHOR_RELATIVE,
                label="production R4 candidate anchor",
            )
            != candidate_anchor["file_state"]
        ):
            raise R4Error("production R4 candidate anchor changed during validation")
        if not _production_sqlite_unchanged(
            workset_path, workset["file_state"], label="production workset"
        ):
            raise R4Error("production workset changed during validation")
        if (
            _production_directory_fd_identity(root_fd, label="production root")
            != root_identity
            or _production_directory_identity(original_root, label="production root")
            != root_identity
        ):
            raise R4Error("production root changed during validation")
        output = {
            "passed": not reasons,
            "reasons": sorted(reasons),
            "collector": "fixed-production-root-workset-v1",
            "provider_calls": 0,
            "root": str(original_root),
            "state": {
                "sha256": state_sha256,
                "file_state": _production_stat(state_path, label="production state"),
            },
            "config": {"sha256": config_sha256},
            "profile_contract": {
                "artifact_id": contract_id,
                "sha256": contract_sha256,
                "sealed": contract,
            },
            "workset": {
                "rows": workset["rows"],
                "counts": workset["counts"],
                "sha256": workset["sha256"],
                "receipts": workset["receipts"],
            },
            "candidate_checkpoint": candidate,
            "candidate_anchor": candidate_anchor,
            "labels": {
                "count": labels["count"],
                "head_sha256": labels["head_sha256"],
                "sha256": labels["sha256"],
            },
            "events": {
                key: {
                    "count": len(rows),
                    "head_sha256": rows[-1]["record_sha256"] if rows else "",
                }
                for key, rows in events.items()
            },
            "scenarios": {"scope": "non-authoritative-owned-fault-diagnostic"},
            "quality": quality,
        }
        return output
    except (
        R4Error,
        OSError,
        sqlite3.Error,
        ValueError,
        UnicodeError,
        RecursionError,
        OverflowError,
    ) as exc:
        unavailable["reasons"] = [str(exc) or "production_evidence_invalid"]
        return unavailable
    finally:
        if cwd_fd is not None:
            try:
                os.fchdir(cwd_fd)
            except OSError as exc:
                restore_error = exc
        if root_fd is not None:
            try:
                os.close(root_fd)
            except OSError:
                pass
        if cwd_fd is not None:
            try:
                os.close(cwd_fd)
            except OSError:
                pass
        if restore_error is not None:
            raise R4Error("production cwd restore failed") from restore_error


def _validate_production_attestations(
    receipts: Sequence[Mapping[str, Any]], source: Mapping[str, Any]
) -> dict[str, Any]:
    """Keep production certification disabled until a trusted chain exists.

    JSON seals and caller-provided producer bundles are replayable claims, not
    authoritative runtime evidence.  The external-bundle path therefore never
    promotes them to production certification; only the fixed-root collector
    can produce that verdict.  Source-contract validation remains usable via
    ``--source-contract-only``.
    """

    del source
    kinds = sorted(
        {
            _text(row.get("kind"))
            for row in receipts
            if isinstance(row, Mapping) and _text(row.get("kind"))
        }
    )
    producers = sorted(
        {
            _text(producer.get("id"))
            for row in receipts
            if isinstance(row, Mapping)
            for producer in [row.get("producer")]
            if isinstance(producer, Mapping) and _text(producer.get("id"))
        }
    )
    return {
        "passed": False,
        "reasons": ["independent_live_provider_attestation_unavailable"],
        "kinds": kinds,
        "producers": producers,
    }


def _write_immutable(
    output: Path,
    payload: Mapping[str, Any],
    *,
    before_publish: Callable[[], None] | None = None,
) -> tuple[str, Path, dict[str, Any]]:
    output.mkdir(parents=True, exist_ok=True)
    unsigned = {"schema": R4_SCHEMA, "namespace": "recall-distillation", **payload}
    artifact_id = _sha256(unsigned)
    artifact = _sealed({"artifact_id": artifact_id, **unsigned})
    path = output / f"{artifact_id}.json"
    encoded = _json_bytes(artifact) + b"\n"
    published = _publish_owned_artifact(
        output, path.name, encoded, before_publish=before_publish
    )
    return artifact_id, published, artifact


def _write_immutable_pinned(
    directory_fd: int,
    payload: Mapping[str, Any],
    *,
    verify_directory: Callable[[], None],
    before_publish: Callable[[], None] | None = None,
) -> tuple[str, str, dict[str, Any], bool]:
    """Publish an R4 artifact without reopening its output pathname."""

    unsigned = {"schema": R4_SCHEMA, "namespace": "recall-distillation", **payload}
    artifact_id = _sha256(unsigned)
    artifact = _sealed({"artifact_id": artifact_id, **unsigned})
    name = f"{artifact_id}.json"
    encoded = _json_bytes(artifact) + b"\n"
    verify_directory()
    try:
        existing, _state = _authority_read_fd(directory_fd, name, label="R4 artifact")
    except R4Error:
        existing = None
    if existing is not None:
        if existing != encoded:
            raise R4Error("owned R4 artifact conflict")
        verify_directory()
        return artifact_id, name, artifact, True
    if before_publish is not None:
        before_publish()
    verify_directory()
    _authority_publish_fd(directory_fd, name, encoded, label="R4 artifact")
    try:
        verify_directory()
        observed, _state = _authority_read_fd(
            directory_fd, name, label="R4 artifact"
        )
        if observed != encoded:
            raise R4Error("R4 artifact changed after publication")
    except BaseException:
        try:
            current, _state = _authority_read_fd(
                directory_fd, name, label="R4 artifact"
            )
            if current == encoded:
                os.unlink(name, dir_fd=directory_fd)
                os.fsync(directory_fd)
        except (OSError, R4Error):
            pass
        raise
    return artifact_id, name, artifact, False


_AUTHORITY_DIRECTORY_FLAGS = (
    os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
)


def _authority_directory_identity(value: os.stat_result) -> tuple[int, int]:
    if not stat.S_ISDIR(value.st_mode):
        raise R4Error("authority staging directory is unsafe")
    return int(value.st_dev), int(value.st_ino)


def _open_authority_output_root(output: Path) -> tuple[int, int, str, tuple[int, int]]:
    """Create and pin ``output`` through parent dirfds, without path reopen."""

    original = output.expanduser().absolute()
    if not original.is_absolute() or len(original.parts) < 2:
        raise R4Error("authority receipt output path is invalid")
    current = os.open("/", _AUTHORITY_DIRECTORY_FLAGS)
    root_fd = -1
    try:
        for component in original.parts[1:-1]:
            try:
                os.mkdir(component, 0o700, dir_fd=current)
            except FileExistsError:
                pass
            opened = os.open(component, _AUTHORITY_DIRECTORY_FLAGS, dir_fd=current)
            os.close(current)
            current = opened
        name = original.name
        try:
            os.mkdir(name, 0o700, dir_fd=current)
        except FileExistsError:
            pass
        root_fd = os.open(name, _AUTHORITY_DIRECTORY_FLAGS, dir_fd=current)
    except OSError as exc:
        if root_fd >= 0:
            os.close(root_fd)
        os.close(current)
        raise R4Error("authority receipt output path is unsafe") from exc
    try:
        root_identity = _authority_directory_identity(os.fstat(root_fd))
        named_identity = _authority_directory_identity(
            os.stat(name, dir_fd=current, follow_symlinks=False)
        )
    except (OSError, R4Error) as exc:
        os.close(root_fd)
        os.close(current)
        raise R4Error("authority receipt output path is unsafe") from exc
    if named_identity != root_identity:
        os.close(root_fd)
        os.close(current)
        raise R4Error("authority receipt output changed during staging")
    return current, root_fd, name, root_identity


def _open_existing_authority_root(output: Path) -> tuple[int, int, str, tuple[int, int]]:
    """Pin an existing output root for authority validation without creating it."""

    original = output.expanduser().absolute()
    if not original.is_absolute() or len(original.parts) < 2:
        raise R4Error("authority receipt output path is invalid")
    current = os.open("/", _AUTHORITY_DIRECTORY_FLAGS)
    root_fd = -1
    try:
        for component in original.parts[1:-1]:
            opened = os.open(component, _AUTHORITY_DIRECTORY_FLAGS, dir_fd=current)
            os.close(current)
            current = opened
        name = original.name
        root_fd = os.open(name, _AUTHORITY_DIRECTORY_FLAGS, dir_fd=current)
        identity = _authority_directory_identity(os.fstat(root_fd))
        if _authority_directory_identity(
            os.stat(name, dir_fd=current, follow_symlinks=False)
        ) != identity:
            raise R4Error("authority receipt output changed during validation")
        return current, root_fd, name, identity
    except (OSError, R4Error) as exc:
        if root_fd >= 0:
            os.close(root_fd)
        os.close(current)
        raise R4Error("authority receipt output path is unsafe") from exc


def _assert_authority_root(
    parent_fd: int, root_fd: int, name: str, expected: tuple[int, int]
) -> None:
    try:
        named = _authority_directory_identity(
            os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        )
    except OSError as exc:
        raise R4Error("authority receipt output changed during staging") from exc
    if _authority_directory_identity(os.fstat(root_fd)) != expected or named != expected:
        raise R4Error("authority receipt output changed during staging")


def _authority_read_fd(directory_fd: int, name: str, *, label: str) -> tuple[bytes, dict[str, int]]:
    if Path(name).name != name:
        raise R4Error(f"{label} name is unsafe")
    try:
        fd = os.open(name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=directory_fd)
    except OSError as exc:
        raise R4Error(f"{label} path is unsafe") from exc
    try:
        before_result = os.fstat(fd)
        if not stat.S_ISREG(before_result.st_mode) or before_result.st_nlink != 1:
            raise R4Error(f"{label} file is unsafe")
        before = _authority_fd_state(before_result)
        raw = b"".join(iter(lambda: os.read(fd, 1024 * 1024), b""))
        if _authority_fd_state(os.fstat(fd)) != before or len(raw) != before["st_size"]:
            raise R4Error(f"{label} changed during read")
        return raw, before
    finally:
        os.close(fd)


def _authority_publish_fd(directory_fd: int, name: str, raw: bytes, *, label: str) -> dict[str, int]:
    """Atomically create one immutable staging file below a pinned dirfd."""

    if Path(name).name != name:
        raise R4Error(f"{label} name is unsafe")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(name, flags, 0o600, dir_fd=directory_fd)
    except OSError as exc:
        raise R4Error(f"{label} immutable file already exists") from exc
    try:
        try:
            offset = 0
            while offset < len(raw):
                written_count = os.write(fd, raw[offset:])
                if written_count <= 0:
                    raise R4Error(f"{label} temporary write failed")
                offset += written_count
            os.fsync(fd)
            written = os.fstat(fd)
            if (
                not stat.S_ISREG(written.st_mode)
                or written.st_nlink != 1
                or written.st_size != len(raw)
            ):
                raise R4Error(f"{label} staging file is unsafe")
        finally:
            os.close(fd)
        os.fsync(directory_fd)
        observed, state = _authority_read_fd(directory_fd, name, label=label)
        if observed != raw:
            raise R4Error(f"{label} changed after publication")
        return state
    except BaseException:
        try:
            os.unlink(name, dir_fd=directory_fd)
        except FileNotFoundError:
            pass
        try:
            os.fsync(directory_fd)
        except OSError:
            pass
        raise


def _authority_receipt_values(
    raw: bytes, *, name: str, label: str
) -> list[dict[str, Any]]:
    if len(raw) > _MAX_RECEIPT_BYTES:
        raise R4Error("receipt file exceeds bounded size")
    try:
        payload = (
            [json.loads(line) for line in raw.splitlines() if line.strip()]
            if name.endswith(".jsonl")
            else json.loads(raw)
        )
    except (ValueError, UnicodeError) as exc:
        raise R4Error(f"{label} JSON is invalid") from exc
    values = payload if isinstance(payload, list) else [payload]
    return [_verify_seal(value) for value in values]


def _authority_receipt_ids(
    receipts: Sequence[Mapping[str, Any]], *, label: str
) -> set[str]:
    """Require receipt IDs to be canonical and unique within one authority set."""

    ids: set[str] = set()
    for receipt in receipts:
        receipt_id = receipt.get("receipt_id")
        if not isinstance(receipt_id, str) or _SHA.fullmatch(receipt_id) is None:
            raise R4Error(f"{label} receipt id is invalid")
        if receipt_id in ids:
            raise R4Error(f"duplicate {label} receipt id")
        ids.add(receipt_id)
    return ids


def _capture_authority_input_payloads(
    source_path: Path | None, *, kind: str
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, str]]]:
    """Capture receipt bytes once; the sealed authority file owns the result."""

    if source_path is None:
        return [], {"files": [], "count": 0}, []
    paths = sorted((*source_path.glob("*.json"), *source_path.glob("*.jsonl")))
    receipts: list[dict[str, Any]] = []
    files: list[dict[str, Any]] = []
    payloads: list[dict[str, str]] = []
    for index, path in enumerate(paths):
        raw, state, _parent = _read_authority_regular(path, label=f"{kind} receipt")
        name = f"{index:04d}{path.suffix}"
        digest = hashlib.sha256(raw).hexdigest()
        receipts.extend(
            _authority_receipt_values(raw, name=path.name, label=f"{kind} receipt")
        )
        files.append(
            {
                "path": name,
                "sha256": digest,
                "file_state": _authority_inventory_file_state(state),
            }
        )
        payloads.append(
            {
                "path": name,
                "sha256": digest,
                "payload_b64": base64.b64encode(raw).decode("ascii"),
            }
        )
    _authority_receipt_ids(receipts, label=kind)
    return receipts, {"files": files, "count": len(receipts)}, payloads


def _read_embedded_authority_inputs(
    value: object, *, inventory: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not isinstance(value, Mapping) or set(value) != {"local", "ox"}:
        raise R4Error("authority receipt embedded inputs are invalid")
    result: dict[str, list[dict[str, Any]]] = {}
    all_ids: set[str] = set()
    for kind in ("local", "ox"):
        rows = value.get(kind)
        listed = inventory.get(kind) if isinstance(inventory, Mapping) else None
        if (
            not isinstance(rows, list)
            or not isinstance(listed, Mapping)
            or set(listed) != {"files", "count"}
            or isinstance(listed.get("count"), bool)
            or not isinstance(listed.get("count"), int)
        ):
            raise R4Error("authority receipt embedded inputs are invalid")
        files = listed.get("files")
        if not isinstance(files, list) or len(rows) != len(files):
            raise R4Error("authority receipt embedded inputs are invalid")
        receipts: list[dict[str, Any]] = []
        paths: set[str] = set()
        for row, file in zip(rows, files, strict=True):
            path = row.get("path") if isinstance(row, Mapping) else None
            payload_b64 = row.get("payload_b64") if isinstance(row, Mapping) else None
            if (
                not isinstance(row, Mapping)
                or set(row) != {"path", "sha256", "payload_b64"}
                or not isinstance(file, Mapping)
                or set(file) != {"path", "sha256", "file_state"}
                or row.get("path") != file.get("path")
                or row.get("sha256") != file.get("sha256")
                or not isinstance(path, str)
                or _AUTHORITY_EMBEDDED_NAME.fullmatch(path) is None
                or Path(path).name != path
                or "/" in path
                or "\x00" in path
                or _SHA.fullmatch(str(row.get("sha256"))) is None
                or not isinstance(payload_b64, str)
                or not _authority_inventory_file_state_valid(file.get("file_state"))
                or str(row["path"]) in paths
            ):
                raise R4Error("authority receipt embedded inputs are invalid")
            try:
                raw = base64.b64decode(payload_b64, validate=True)
            except ValueError as exc:
                raise R4Error("authority receipt embedded inputs are invalid") from exc
            if (
                base64.b64encode(raw).decode("ascii") != payload_b64
                or hashlib.sha256(raw).hexdigest() != row["sha256"]
                or file["file_state"].get("st_size") != len(raw)
            ):
                raise R4Error("authority receipt embedded inputs are invalid")
            paths.add(str(row["path"]))
            receipts.extend(
                _authority_receipt_values(
                    raw, name=str(row["path"]), label=f"{kind} receipt"
                )
            )
        if listed.get("count") != len(receipts):
            raise R4Error("authority receipt embedded inputs are invalid")
        ids = _authority_receipt_ids(receipts, label=kind)
        if all_ids.intersection(ids):
            raise R4Error("duplicate authority receipt id")
        all_ids.update(ids)
        result[kind] = receipts
    return result["local"], result["ox"]


def produce_source_bound_authority_receipt(
    output: Path,
    *,
    source_root: Path,
    source_commit: str,
    local_receipts: Path | None,
    ox_receipts: Path | None,
    before_stage: Callable[[str], None] | None = None,
    _output_context: tuple[int, int, str, tuple[int, int]] | None = None,
    _expected_projection: Mapping[str, Any] | None = None,
    _expected_inventory: Mapping[str, Any] | None = None,
    _expected_results: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Publish a formal-only authority receipt before its R4 artifact exists."""

    _reject_original_symlinks((("authority receipt output", output),))
    owns_output_context = _output_context is None
    parent_fd, output_fd, output_name, output_identity = (
        _open_authority_output_root(output)
        if _output_context is None
        else _output_context
    )
    published_authority_name: str | None = None
    succeeded = False
    try:
        if before_stage is not None:
            before_stage("output-opened")
        _assert_authority_root(parent_fd, output_fd, output_name, output_identity)
        source = _assert_source(source_root, source_commit)
        projection = _collect_authoritative_production(
            source_root=source_root, source=source, production_root=PRODUCTION_ROOT
        )
        if _expected_projection is not None and projection != _expected_projection:
            raise R4Error("production projection changed before authority publication")
        if projection.get("passed") is not True or projection.get("provider_calls") != 0:
            return {"available": False, "reason": "formal_production_authority_unavailable"}, {"local": {"files": [], "count": 0}, "ox": {"files": [], "count": 0}, "production": {"files": [], "count": 0}}
        owned_fault_result = _fresh_owned_fault_contract(source_root, source_commit)
        local, local_inventory, local_payloads = _capture_authority_input_payloads(
            local_receipts, kind="local"
        )
        ox, ox_inventory, ox_payloads = _capture_authority_input_payloads(
            ox_receipts, kind="ox"
        )
        if ox:
            raise R4Error("formal OX authority must be fixed-root runtime evidence")
        if _authority_receipt_ids(local, label="local").intersection(
            _authority_receipt_ids(ox, label="ox")
        ):
            raise R4Error("duplicate authority receipt id")
        local_result = _validate_local(local, source)
        ox_result = _validate_ox((), source, production_projection=projection)
        if (
            not local_result["passed"]
            or not ox_result["passed"]
            or owned_fault_result.get("passed") is not True
        ):
            raise R4Error("authority receipt inputs no longer satisfy source contract")
        input_inventory = {"local": local_inventory, "ox": ox_inventory}
        input_results = {
            "local": local_result,
            "ox": ox_result,
            "owned_faults": owned_fault_result,
        }
        if _expected_inventory is not None and input_inventory != _expected_inventory:
            raise R4Error("authority receipt inputs changed before publication")
        if _expected_results is not None and input_results != _expected_results:
            raise R4Error("authority receipt results changed before publication")
        inventory = {**input_inventory, "production": {"files": [], "count": 0}}
        ox_authority = _runtime_ox_authority(projection)
        payload = {
            "schema": R4_AUTHORITY_RECEIPT_SCHEMA,
            "namespace": "recall-distillation",
            "captured_at": datetime.now(UTC).isoformat(),
            "source": {"source_commit": source["commit"], "source_tree_sha256": source["tree_sha256"], "source_ox_identity_sha256": source["ox_identity_sha256"]},
            "production_projection_sha256": _sha256(projection),
            "ox_authority": ox_authority,
            "receipt_inventory": dict(inventory),
            "input_payloads": {"local": local_payloads, "ox": ox_payloads},
        }
        receipt_id = _sha256(payload)
        receipt = _sealed({"artifact_id": receipt_id, **payload})
        name = f"{receipt_id}.authority.json"
        encoded = _json_bytes(receipt) + b"\n"
        _authority_publish_fd(output_fd, name, encoded, label="authority receipt")
        published_authority_name = name
        raw, _state = _authority_read_fd(output_fd, name, label="authority receipt")
        _assert_authority_root(parent_fd, output_fd, output_name, output_identity)
        if raw != encoded:
            raise R4Error("authority receipt changed after publication")
        succeeded = True
        return {"available": True, "artifact_id": receipt_id, "seal_sha256": receipt["seal_sha256"], "relative_path": name, "file_sha256": hashlib.sha256(raw).hexdigest(), "parent_dev": output_identity[0], "parent_ino": output_identity[1]}, inventory
    finally:
        # A failed staged publication must not leave partial files in the
        # pinned original directory, and never follows an attacker replacement.
        if not succeeded and published_authority_name is not None:
            try:
                os.unlink(published_authority_name, dir_fd=output_fd)
            except FileNotFoundError:
                pass
        if owns_output_context:
            close_errors: list[OSError] = []
            for fd in (output_fd, parent_fd):
                try:
                    os.close(fd)
                except OSError as exc:
                    close_errors.append(exc)
            if close_errors:
                primary = sys.exc_info()[1]
                if primary is None:
                    raise R4Error("authority staging descriptor cleanup failed") from close_errors[0]
                primary.add_note(
                    "authority staging descriptor cleanup also failed: "
                    + "; ".join(str(error) for error in close_errors)
                )


def _authority_fd_state(value: os.stat_result) -> dict[str, int]:
    return {
        "st_dev": int(value.st_dev),
        "st_ino": int(value.st_ino),
        "st_mode": int(value.st_mode & 0o7777),
        "st_size": int(value.st_size),
        "st_mtime_ns": int(value.st_mtime_ns),
        "st_ctime_ns": int(value.st_ctime_ns),
    }


def _authority_directory_state(value: os.stat_result) -> dict[str, int]:
    return {
        "st_dev": int(value.st_dev),
        "st_ino": int(value.st_ino),
        "st_mode": int(value.st_mode & 0o7777),
    }


def _authority_inventory_file_state(value: Mapping[str, int]) -> dict[str, int]:
    """Keep the public receipt inventory compatible with the R4 file schema."""

    result = {
        key: int(value[key])
        for key in ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns")
    }
    if not _authority_inventory_file_state_valid(result):
        raise R4Error("authority file state is invalid")
    return result


def _authority_inventory_file_state_valid(value: object, *, size: int | None = None) -> bool:
    """Validate the closed, nonnegative metadata projection used in receipts."""

    if not isinstance(value, Mapping) or set(value) != {
        "st_dev",
        "st_ino",
        "st_mode",
        "st_size",
        "st_mtime_ns",
    }:
        return False
    if any(
        isinstance(item, bool) or not isinstance(item, int) or item < 0
        for item in value.values()
    ):
        return False
    if value["st_mode"] > 0o7777:
        return False
    return size is None or value["st_size"] == size


def _authority_path_states(path: Path, *, label: str) -> list[dict[str, int]]:
    """Open every absolute path component without following links."""

    original = path.expanduser().absolute()
    if not original.is_absolute() or len(original.parts) < 2:
        raise R4Error(f"{label} path is invalid")
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    current = os.open("/", directory_flags)
    states: list[dict[str, int]] = []
    try:
        for index, component in enumerate(original.parts[1:]):
            final = index == len(original.parts) - 2
            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            if not final:
                flags |= getattr(os, "O_DIRECTORY", 0)
            try:
                opened = os.open(component, flags, dir_fd=current)
            except OSError as exc:
                raise R4Error(f"{label} path is unsafe") from exc
            value = os.fstat(opened)
            if (final and not stat.S_ISREG(value.st_mode)) or (
                not final and not stat.S_ISDIR(value.st_mode)
            ):
                os.close(opened)
                raise R4Error(f"{label} path has an unsafe component")
            states.append(
                _authority_fd_state(value)
                if final
                else _authority_directory_state(value)
            )
            os.close(current)
            current = opened
        return states
    finally:
        os.close(current)


def _read_authority_regular(path: Path, *, label: str) -> tuple[bytes, dict[str, int], dict[str, int]]:
    """Read one regular file through stable dirfds and reject path swaps."""

    states = _authority_path_states(path, label=label)
    if len(states) < 2:
        raise R4Error(f"{label} path has no parent")
    original = path.expanduser().absolute()
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    current = os.open("/", directory_flags)
    try:
        for component in original.parts[1:-1]:
            opened = os.open(
                component,
                directory_flags,
                dir_fd=current,
            )
            os.close(current)
            current = opened
        fd = os.open(
            original.name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=current,
        )
        try:
            before = _authority_fd_state(os.fstat(fd))
            if before != states[-1]:
                raise R4Error(f"{label} changed while opening")
            data = b"".join(iter(lambda: os.read(fd, 1024 * 1024), b""))
            if _authority_fd_state(os.fstat(fd)) != before or len(data) != before["st_size"]:
                raise R4Error(f"{label} changed during read")
        finally:
            os.close(fd)
    except OSError as exc:
        raise R4Error(f"{label} path is unsafe") from exc
    finally:
        os.close(current)
    if _authority_path_states(original, label=label) != states:
        raise R4Error(f"{label} path changed during read")
    return data, before, states[-2]


def read_artifact(path: Path) -> dict[str, Any]:
    """Verify one R4 artifact's serialization, identity, and integrity seal.

    This is an integrity/readback check, not an independent production
    authority.  Production certification still requires an external
    completion/watchdog evidence chain.
    """

    _reject_original_symlinks((("R4 artifact", path),))
    try:
        data, _state, _parent = _read_authority_regular(path, label="R4 artifact")
        artifact = json.loads(data)
    except (OSError, ValueError, UnicodeError) as exc:
        raise R4Error("R4 artifact is not valid JSON") from exc
    if not isinstance(artifact, dict) or artifact.get("schema") != R4_SCHEMA:
        raise R4Error("R4 artifact schema mismatch")
    expected_keys = {
        "artifact_id",
        "schema",
        "namespace",
        "seal_sha256",
        "captured_at",
        "source",
        "source_after",
        "source_final",
        "source_contract",
        "production_certification",
        "receipt_files",
        "authority_receipt",
        "provider_calls",
        "production_root_used",
    }
    if set(artifact) != expected_keys:
        raise R4Error("R4 artifact payload shape mismatch")
    _verify_seal(artifact, schema=R4_SCHEMA)
    artifact_id = artifact.get("artifact_id")
    if not isinstance(artifact_id, str) or path.stem != artifact_id:
        raise R4Error("R4 artifact filename identity mismatch")
    if artifact_id != _sha256(
        {
            key: value
            for key, value in artifact.items()
            if key not in {"artifact_id", "seal_sha256"}
        }
    ):
        raise R4Error("R4 artifact identity mismatch")
    canonical = _json_bytes(artifact)
    if data not in {canonical, canonical + b"\n"}:
        raise R4Error("R4 artifact is not canonical")
    if not isinstance(artifact.get("captured_at"), str) or not artifact["captured_at"]:
        raise R4Error("R4 artifact timestamp is invalid")
    source_keys = {
        "commit",
        "clean",
        "status_sha256",
        "status_count",
        "tree_sha256",
        "file_count",
        "symlink_count",
        "ox_identity_sha256",
        "account_uid",
        "account_home",
    }
    snapshots = [
        artifact.get(name) for name in ("source", "source_after", "source_final")
    ]
    if any(
        not isinstance(snapshot, Mapping) or set(snapshot) != source_keys
        for snapshot in snapshots
    ):
        raise R4Error("R4 artifact source snapshot shape mismatch")
    first_snapshot = snapshots[0]
    if not isinstance(first_snapshot, Mapping):
        raise R4Error("R4 artifact source snapshot shape mismatch")
    if not (
        snapshots[0] == snapshots[1] == snapshots[2]
        and isinstance(first_snapshot.get("clean"), bool)
    ):
        raise R4Error("R4 artifact source snapshots disagree")
    source_contract = artifact.get("source_contract")
    legacy_source_contract_keys = {"schema", "passed", "local", "ox"}
    current_source_contract_keys = legacy_source_contract_keys | {"ox_authority"}
    owned_source_contract_keys = current_source_contract_keys | {"owned_faults"}
    if not isinstance(source_contract, Mapping) or set(source_contract) not in {
        frozenset(legacy_source_contract_keys),
        frozenset(current_source_contract_keys),
        frozenset(owned_source_contract_keys),
    }:
        raise R4Error("R4 artifact source contract is missing")
    if (
        source_contract.get("schema") != SOURCE_SCHEMA
        or not isinstance(source_contract.get("passed"), bool)
        or not isinstance(source_contract.get("local"), Mapping)
        or not isinstance(source_contract.get("ox"), Mapping)
    ):
        raise R4Error("R4 artifact source contract is invalid")
    if set(source_contract) == owned_source_contract_keys and not isinstance(
        source_contract.get("owned_faults"), Mapping
    ):
        raise R4Error("R4 artifact owned fault contract is invalid")
    if set(source_contract) in {
        frozenset(current_source_contract_keys),
        frozenset(owned_source_contract_keys),
    }:
        ox_authority = source_contract.get("ox_authority")
        if not isinstance(ox_authority, Mapping):
            raise R4Error("R4 artifact OX authority is invalid")
        if ox_authority.get("kind") == "fixed_root_runtime_sealed_projection":
            if (
                set(ox_authority)
                != {"kind", "projection_sha256", "receipt_inventory"}
                or _SHA.fullmatch(_text(ox_authority.get("projection_sha256")))
                is None
                or ox_authority.get("receipt_inventory")
                != {"files": [], "count": 0}
            ):
                raise R4Error("R4 artifact OX authority is invalid")
        elif ox_authority.get("kind") == "external_sealed_receipts":
            if set(ox_authority) != {"kind", "receipt_inventory"}:
                raise R4Error("R4 artifact OX authority is invalid")
        else:
            raise R4Error("R4 artifact OX authority is invalid")
    production = artifact.get("production_certification")
    unavailable_production_keys = {
        "passed",
        "reasons",
        "collector",
        "provider_calls",
    }
    unavailable_with_root_keys = unavailable_production_keys | {"root"}
    collected_production_keys = unavailable_with_root_keys | {
        "state",
        "config",
        "profile_contract",
        "workset",
        "candidate_checkpoint",
        "candidate_anchor",
        "labels",
        "events",
        "scenarios",
        "quality",
    }
    if not isinstance(production, Mapping) or set(production) not in {
        frozenset(unavailable_production_keys),
        frozenset(unavailable_with_root_keys),
        frozenset(collected_production_keys),
    }:
        raise R4Error("R4 artifact production verdict is missing")
    if (
        not isinstance(production.get("passed"), bool)
        or not isinstance(production.get("reasons"), list)
        or not isinstance(production.get("collector"), str)
        or isinstance(production.get("provider_calls"), bool)
        or not isinstance(production.get("provider_calls"), int)
        or production.get("provider_calls") != 0
    ):
        raise R4Error("R4 artifact production verdict is invalid")
    receipt_files = artifact.get("receipt_files")
    if not isinstance(receipt_files, Mapping) or set(receipt_files) != {
        "local",
        "ox",
        "production",
    }:
        raise R4Error("R4 artifact receipt file shape mismatch")
    for receipt in receipt_files.values():
        if (
            not isinstance(receipt, Mapping)
            or not isinstance(receipt.get("files"), list)
            or isinstance(receipt.get("count"), bool)
            or not isinstance(receipt.get("count"), int)
        ):
            raise R4Error("R4 artifact receipt file entry is invalid")
    authority = artifact.get("authority_receipt")
    if not isinstance(authority, Mapping) or authority.get("available") not in {True, False}:
        raise R4Error("R4 artifact authority reference is invalid")
    if authority.get("available") is True and (
        set(authority)
        != {"available", "artifact_id", "seal_sha256", "relative_path", "file_sha256", "parent_dev", "parent_ino"}
        or _SHA.fullmatch(str(authority.get("artifact_id"))) is None
        or _SHA.fullmatch(str(authority.get("seal_sha256"))) is None
        or _SHA.fullmatch(str(authority.get("file_sha256"))) is None
        or not isinstance(authority.get("relative_path"), str)
        or isinstance(authority.get("parent_dev"), bool)
        or not isinstance(authority.get("parent_dev"), int)
        or isinstance(authority.get("parent_ino"), bool)
        or not isinstance(authority.get("parent_ino"), int)
    ):
        raise R4Error("R4 artifact authority reference is invalid")
    if authority.get("available") is False and set(authority) != {"available", "reason"}:
        raise R4Error("R4 artifact authority reference is invalid")
    if (
        isinstance(artifact.get("provider_calls"), bool)
        or not isinstance(artifact.get("provider_calls"), int)
        or artifact.get("provider_calls") != 0
        or not isinstance(artifact.get("production_root_used"), bool)
    ):
        raise R4Error("R4 artifact provider/root contract is invalid")
    if _read_authority_regular(path, label="R4 artifact")[0] != data:
        raise R4Error("R4 artifact changed during read")
    return artifact


def validate_source_bound_authority_receipt(
    authority_path: Path,
    *,
    artifact_path: Path,
    source_root: Path,
    source_commit: str,
) -> dict[str, Any]:
    """Re-derive the sole authority receipt from source and fixed production."""

    _reject_original_symlinks((("authority receipt", authority_path), ("R4 artifact", artifact_path)))
    source = _assert_source(source_root, source_commit)
    artifact = read_artifact(artifact_path)
    _artifact_raw, _artifact_state, artifact_parent = _read_authority_regular(
        artifact_path, label="R4 artifact"
    )
    reference = artifact.get("authority_receipt")
    if not isinstance(reference, Mapping) or reference.get("available") is not True:
        raise R4Error("R4 artifact has no formal authority receipt")
    expected_reference = {
        "available", "artifact_id", "seal_sha256", "relative_path", "file_sha256", "parent_dev", "parent_ino"
    }
    if (
        set(reference) != expected_reference
        or Path(str(reference.get("relative_path"))).name != reference.get("relative_path")
        or authority_path.name != reference.get("relative_path")
    ):
        raise R4Error("R4 authority reference is invalid")
    parent_fd, output_fd, output_name, output_identity = _open_existing_authority_root(
        authority_path.parent
    )
    try:
        raw, state = _authority_read_fd(output_fd, authority_path.name, label="authority receipt")
        if (
            hashlib.sha256(raw).hexdigest() != reference.get("file_sha256")
            or output_identity != (reference.get("parent_dev"), reference.get("parent_ino"))
            or (artifact_parent["st_dev"], artifact_parent["st_ino"])
            != (reference.get("parent_dev"), reference.get("parent_ino"))
        ):
            raise R4Error("authority receipt path binding is invalid")
        projection = _collect_authoritative_production(
            source_root=source_root, source=source, production_root=PRODUCTION_ROOT
        )
        if projection.get("passed") is not True or projection.get("provider_calls") != 0:
            raise R4Error("formal production authority is unavailable")
        owned_fault_result = _fresh_owned_fault_contract(source_root, source_commit)
        try:
            receipt = json.loads(raw)
        except (ValueError, UnicodeError) as exc:
            raise R4Error("authority receipt is invalid") from exc
        expected = {
            "schema",
            "namespace",
            "artifact_id",
            "seal_sha256",
            "captured_at",
            "source",
            "production_projection_sha256",
            "ox_authority",
            "receipt_inventory",
            "input_payloads",
        }
        if not isinstance(receipt, Mapping) or set(receipt) != expected or _json_bytes(receipt) + b"\n" != raw:
            raise R4Error("authority receipt schema is invalid")
        if (
            receipt.get("schema") != R4_AUTHORITY_RECEIPT_SCHEMA
            or receipt.get("namespace") != "recall-distillation"
        ):
            raise R4Error("authority receipt schema is invalid")
        unsigned = {key: value for key, value in receipt.items() if key not in {"artifact_id", "seal_sha256"}}
        source_binding = {"source_commit": source["commit"], "source_tree_sha256": source["tree_sha256"], "source_ox_identity_sha256": source["ox_identity_sha256"]}
        observed_inventory = receipt.get("receipt_inventory")
        if not isinstance(observed_inventory, Mapping) or set(observed_inventory) != {"local", "ox", "production"}:
            raise R4Error("authority receipt inventory is invalid")
        if observed_inventory.get("production") != {"files": [], "count": 0}:
            raise R4Error("authority receipt inventory is invalid")
        local, ox = _read_embedded_authority_inputs(
            receipt.get("input_payloads"), inventory=observed_inventory
        )
        local_result = _validate_local(local, source)
        if ox:
            raise R4Error("formal authority embeds external OX receipts")
        ox_result = _validate_ox((), source, production_projection=projection)
        ox_authority = _runtime_ox_authority(projection)
        source_contract = artifact.get("source_contract")
        if (
            not isinstance(source_contract, Mapping)
            or source_contract.get("passed") is not True
            or local_result.get("passed") is not True
            or ox_result.get("passed") is not True
            or owned_fault_result.get("passed") is not True
        ):
            raise R4Error("authority receipt source contract is invalid")
        if (
            receipt.get("artifact_id") != reference.get("artifact_id")
            or authority_path.name != f"{receipt.get('artifact_id')}.authority.json"
            or receipt.get("seal_sha256") != reference.get("seal_sha256")
            or receipt.get("artifact_id") != _sha256(unsigned)
            or receipt.get("seal_sha256") != _sha256({"artifact_id": receipt.get("artifact_id"), **unsigned})
            or receipt.get("source") != source_binding
            or receipt.get("production_projection_sha256") != _sha256(projection)
            or receipt.get("ox_authority") != ox_authority
            or artifact.get("production_certification") != projection
            or receipt.get("receipt_inventory") != observed_inventory
            or artifact.get("receipt_files") != observed_inventory
            or source_contract.get("local") != local_result
            or source_contract.get("ox") != ox_result
            or source_contract.get("owned_faults") != owned_fault_result
            or source_contract.get("ox_authority") != ox_authority
        ):
            raise R4Error("authority receipt binding is invalid")
        projection_after = _collect_authoritative_production(
            source_root=source_root, source=source, production_root=PRODUCTION_ROOT
        )
        # Staged files are deliberately not authoritative.  The sealed receipt
        # owns the exact input bytes above, so only its own root/file identity
        # remains to be checked after the final collector pass.
        raw_final, state_final = _authority_read_fd(output_fd, authority_path.name, label="authority receipt")
        _assert_authority_root(parent_fd, output_fd, output_name, output_identity)
        if (
            raw_final != raw
            or state_final != state
            or _assert_source(source_root, source_commit) != source
            or projection_after != projection
        ):
            raise R4Error("authority receipt changed during validation")
        return {"artifact_id": str(receipt["artifact_id"]), "r4_artifact_id": str(artifact["artifact_id"]), "file_state": state}
    finally:
        for fd in (output_fd, parent_fd):
            if fd >= 0:
                try:
                    os.close(fd)
                except OSError:
                    pass


def run(
    *,
    source_root: Path,
    source_commit: str,
    output: Path,
    local_receipts: Path | None = None,
    ox_receipts: Path | None = None,
    production_receipts: Path | None = None,
    production_root: Path | None = None,
) -> tuple[dict[str, Any], Path]:
    """Validate source-bound evidence and persist one immutable artifact."""

    original_paths: list[tuple[str, Path]] = [
        ("source", source_root),
        ("output", output),
    ]
    for name, value in (
        ("local_receipts", local_receipts),
        ("ox_receipts", ox_receipts),
    ):
        if value is not None:
            original_paths.append((name, value))
    if production_root is not None:
        original_paths.append(("production", production_root))
    _reject_original_symlinks(original_paths)
    source_root = source_root.expanduser().resolve(strict=True)
    output = output.expanduser().resolve(strict=False)
    if _overlap(output, PRODUCTION_ROOT.expanduser().absolute()):
        raise R4Error("output/production paths overlap")
    production = production_root.expanduser().absolute() if production_root else None
    input_roots = [
        path.expanduser().resolve(strict=True)
        for path in (local_receipts, ox_receipts)
        if path is not None
    ]
    assert_root_matrix(source_root, output, production, input_roots)
    source_before = _assert_source(source_root, source_commit)
    local, local_files, _local_payloads = _capture_authority_input_payloads(
        local_receipts, kind="local"
    )
    ox, ox_files, _ox_payloads = _capture_authority_input_payloads(
        ox_receipts, kind="ox"
    )
    # Arbitrary production JSON is never an input to certification.  Keep the
    # old parameter only as a compatibility tripwire for callers that still
    # pass it; the resulting verdict remains false and no file is read.
    production_files: dict[str, Any] = {"files": [], "count": 0}
    local_result = (
        _validate_local(local, source_before)
        if local
        else {"passed": False, "reasons": ["local_receipts_missing"], "rows": 0}
    )
    source_after = _assert_source(source_root, source_commit)
    if source_after != source_before:
        raise R4Error("source changed during evidence validation")
    if production is not None and production != PRODUCTION_ROOT.expanduser().absolute():
        production_result = {
            "passed": False,
            "reasons": ["production_root_not_authoritative"],
            "collector": "fixed-production-root-workset-v1",
            "provider_calls": 0,
        }
    elif production is not None:
        production_result = _collect_authoritative_production(
            source_root=source_root,
            source=source_before,
            production_root=PRODUCTION_ROOT,
        )
    elif production_receipts is not None:
        production_result = {
            "passed": False,
            "reasons": ["external_production_receipts_rejected"],
            "collector": "fixed-production-root-workset-v1",
            "provider_calls": 0,
        }
    else:
        production_result = {
            "passed": False,
            "reasons": ["independent_live_provider_attestation_unavailable"],
            "collector": "fixed-production-root-workset-v1",
            "provider_calls": 0,
        }
    if production is not None:
        if ox:
            ox_result = {
                "passed": False,
                "reasons": ["formal_ox_receipts_must_use_fixed_root_runtime"],
                "rows": len(ox),
            }
            ox_authority = {
                "kind": "fixed_root_runtime_sealed_projection",
                "projection_sha256": _sha256(production_result),
                "receipt_inventory": {"files": [], "count": 0},
            }
        else:
            ox_result = _validate_ox(
                (), source_before, production_projection=production_result
            )
            ox_authority = _runtime_ox_authority(production_result)
    elif ox:
        ox_result = _validate_ox(ox, source_before)
        ox_authority = {
            "kind": "external_sealed_receipts",
            "receipt_inventory": ox_files,
        }
    else:
        ox_result = {"passed": False, "reasons": ["ox_receipts_missing"], "rows": 0}
        ox_authority = {
            "kind": "external_sealed_receipts",
            "receipt_inventory": ox_files,
        }
    owned_fault_result = (
        _fresh_owned_fault_contract(source_root, source_commit)
        if local_result.get("passed") is True and ox_result.get("passed") is True
        else {
            "passed": False,
            "reasons": ["source_contract_prerequisite_failed"],
            "count": 0,
            "scenarios": [],
        }
    )
    # The collector reads a large amount of runtime state.  Re-snapshot the
    # source immediately before publishing the artifact so a source mutation
    # during collection cannot be hidden by the earlier before/after pair.
    source_final = _assert_source(source_root, source_commit)
    if source_final != source_before or source_final != source_after:
        raise R4Error("source changed during final evidence validation")
    source_passed = bool(
        local_result["passed"]
        and ox_result["passed"]
        and owned_fault_result.get("passed") is True
        and source_before["clean"]
        and source_after == source_before
        and source_final == source_before
    )
    parent_fd, output_fd, output_name, output_identity = _open_authority_output_root(output)
    try:
        def verify_output() -> None:
            _assert_authority_root(parent_fd, output_fd, output_name, output_identity)

        authority_receipt, authority_files = (
            produce_source_bound_authority_receipt(
                output,
                source_root=source_root,
                source_commit=source_commit,
                local_receipts=local_receipts,
                ox_receipts=ox_receipts,
                _output_context=(parent_fd, output_fd, output_name, output_identity),
                _expected_projection=production_result,
                _expected_inventory={"local": local_files, "ox": ox_files},
                _expected_results={
                    "local": local_result,
                    "ox": ox_result,
                    "owned_faults": owned_fault_result,
                },
            )
            if source_passed and production_result["passed"] is True
            else (
                {"available": False, "reason": "formal_production_authority_unavailable"},
                {
                    "local": local_files,
                    "ox": ox_files,
                    "production": production_files,
                },
            )
        )

        def cleanup_authority() -> None:
            if authority_receipt.get("available") is not True:
                return
            name = authority_receipt.get("relative_path")
            expected_sha256 = authority_receipt.get("file_sha256")
            if not isinstance(name, str) or Path(name).name != name:
                return
            try:
                current, _state = _authority_read_fd(
                    output_fd, name, label="authority receipt"
                )
                if hashlib.sha256(current).hexdigest() == expected_sha256:
                    os.unlink(name, dir_fd=output_fd)
                    os.fsync(output_fd)
            except (OSError, R4Error):
                pass

        payload = {
            "captured_at": datetime.now(UTC).isoformat(),
            "source": source_before,
            "source_after": source_after,
            "source_final": source_final,
            "source_contract": {
                "schema": SOURCE_SCHEMA,
                "passed": source_passed,
                "local": local_result,
                "ox": ox_result,
                "owned_faults": owned_fault_result,
                "ox_authority": ox_authority,
            },
            "production_certification": {
                **production_result,
                "passed": bool(source_passed and production_result["passed"]),
            },
            "receipt_files": authority_files,
            "authority_receipt": authority_receipt,
            "provider_calls": 0,
            "production_root_used": production is not None,
        }

        def _verify_publication_source() -> None:
            nonlocal source_final
            checked = _assert_source(source_root, source_commit)
            if checked != source_before or checked != source_after:
                raise R4Error("source changed during artifact publication")
            source_final = checked

        try:
            _artifact_id, artifact_name, artifact, artifact_preexisted = (
                _write_immutable_pinned(
                    output_fd,
                    payload,
                    verify_directory=verify_output,
                    before_publish=_verify_publication_source,
                )
            )
        except BaseException:
            cleanup_authority()
            raise
        artifact_encoded = _json_bytes(artifact) + b"\n"

        def cleanup_artifact() -> None:
            if artifact_preexisted:
                return
            try:
                current, _state = _authority_read_fd(output_fd, artifact_name, label="R4 artifact")
                if current == artifact_encoded:
                    os.unlink(artifact_name, dir_fd=output_fd)
                    os.fsync(output_fd)
            except (OSError, R4Error):
                pass

        def verify_published_output() -> None:
            try:
                verify_output()
            except BaseException:
                cleanup_artifact()
                cleanup_authority()
                raise

        try:
            checked_after_publish = _assert_source(source_root, source_commit)
        except R4Error as exc:
            cleanup_artifact()
            cleanup_authority()
            raise R4Error("source changed after artifact publication") from exc
        if checked_after_publish != source_before or checked_after_publish != source_final:
            cleanup_artifact()
            cleanup_authority()
            raise R4Error("source changed after artifact publication")
        verify_published_output()
        if artifact["production_certification"]["passed"] is True:
            if authority_receipt.get("available") is not True:
                cleanup_artifact()
                raise R4Error("formal authority receipt is unavailable")
            authority_path = output / str(authority_receipt["relative_path"])
            try:
                validated = validate_source_bound_authority_receipt(
                    authority_path,
                    artifact_path=output / artifact_name,
                    source_root=source_root,
                    source_commit=source_commit,
                )
            except BaseException:
                cleanup_artifact()
                cleanup_authority()
                raise
            if validated.get("r4_artifact_id") != _artifact_id:
                cleanup_artifact()
                cleanup_authority()
                raise R4Error("formal authority readback disagrees with R4 artifact")
        verify_published_output()
        return artifact, output / artifact_name
    finally:
        close_errors: list[OSError] = []
        for fd in (output_fd, parent_fd):
            try:
                os.close(fd)
            except OSError as exc:
                close_errors.append(exc)
        if close_errors:
            primary = sys.exc_info()[1]
            if primary is None:
                raise R4Error("R4 output descriptor cleanup failed") from close_errors[0]
            primary.add_note(
                "R4 output descriptor cleanup also failed: "
                + "; ".join(str(error) for error in close_errors)
            )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--local-receipts", type=Path)
    parser.add_argument("--ox-receipts", type=Path)
    parser.add_argument(
        "--production",
        action="store_true",
        help="read the fixed OS-account Chronovisor root without mutation",
    )
    parser.add_argument(
        "--source-contract-only",
        action="store_true",
        help="validate source/local/OX contracts without claiming production certification",
    )
    parser.add_argument(
        "--run-owned-faults",
        action="store_true",
        help="generate the nine provider-free owned-clone fault artifacts",
    )
    args = parser.parse_args(argv)
    if args.run_owned_faults:
        if any(
            value is not None
            for value in (args.local_receipts, args.ox_receipts)
        ) or args.production or args.source_contract_only:
            parser.error("--run-owned-faults cannot be combined with validation inputs")
        try:
            paths = run_owned_fault_scenarios(
                source_root=args.source_root,
                source_commit=args.source_commit,
                output=args.output,
            )
            source = _assert_source(args.source_root, args.source_commit)
            artifacts, inventory = _load_owned_fault_scenarios(args.output)
            result = _validate_owned_fault_scenarios(artifacts, source)
            if (
                result.get("passed") is not True
                or inventory.get("count") != len(PRODUCTION_FAULT_SCENARIOS)
                or len(paths) != len(PRODUCTION_FAULT_SCENARIOS)
            ):
                raise R4Error("owned fault diagnostic readback failed")
        except (R4Error, OSError, ValueError) as exc:
            print(f"r4 owned faults failed: {exc}", file=sys.stderr)
            return 2
        print(
            json.dumps(
                {
                    "artifacts": [str(path) for path in paths],
                    "authoritative": False,
                    "provider_calls": 0,
                    "validation": "passed",
                },
                sort_keys=True,
            )
        )
        return 0
    try:
        artifact, path = run(
            source_root=args.source_root,
            source_commit=args.source_commit,
            output=args.output,
            local_receipts=args.local_receipts,
            ox_receipts=args.ox_receipts,
            production_root=PRODUCTION_ROOT if args.production else None,
        )
    except (R4Error, OSError, ValueError) as exc:
        print(f"r4 harness failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "artifact_id": artifact["artifact_id"],
                "path": str(path),
                "source_contract": artifact["source_contract"]["passed"],
                "production_certification": artifact["production_certification"][
                    "passed"
                ],
            },
            sort_keys=True,
        )
    )
    if args.source_contract_only:
        return 0 if artifact["source_contract"]["passed"] else 2
    return 0 if artifact["production_certification"]["passed"] else 3


# Keep the names used by the other formal harnesses available to focused tests.
_assert_root_matrix = assert_root_matrix
_source_snapshot = _source_tree_digest
_validate_local_profile = _validate_local
_validate_ox_profile = _validate_ox


if __name__ == "__main__":
    raise SystemExit(main())
