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
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback is fail-closed.
    tomllib = None  # type: ignore[assignment]

R4_SCHEMA = "chronovisor.recall-r4.v1"
RECEIPT_SCHEMA = "chronovisor.recall-r4-receipt.v1"
SOURCE_SCHEMA = "chronovisor.recall-r4-source-contract.v1"
LOCAL_PROFILE = "local-triad-v1"
OX_PROFILE = "ox-alpha-single-v1"
OX_ROUTE = "opencode-go/ox-alpha-free"
OX_MODEL = "ox-alpha-free"
OX_SCHEMA = "chronovisor.recall-distill-teacher-batch.v1"
OX_COHORT = "ox-alpha-backfill-v1"
OX_IDENTITY_REVISION = "ox-alpha-fixed-identity-v1"
OX_PROMPT_SHA256 = "f6a61adb72cafa813a7df9afd6d143c7636069358be17508ac7ad1c0a540bf5a"
OX_SCHEMA_SHA256 = "325a07d3a80d1aa38e9e95569af722b39de962c63994476f57d3baa3444786d7"
OX_ROUTE_SHA256 = "4683cd125fa04ad59ada878a7dbf5ead1bd3941b8bb9a0ca5d02c4eb72e30a98"
OX_MODEL_SHA256 = "29c31b2ca8e6d69bf746ac1a158871549b628562cc05bd5773a3bfbfe501d0b0"
OX_STAGES = (1, 2, 5, 10)
OX_PROBE_REVISION = "single-teacher-repeat-v2"
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
    if production_root is not None and not production_root.is_dir():
        raise R4Error("production root is not a directory")
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


def _source_tree_digest(root: Path) -> dict[str, Any]:
    """Hash the source tree with lstat-before/after TOCTOU checks."""

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
    for row in rows:
        identity = row.get("route_identity")
        if not isinstance(identity, Mapping):
            reasons.add("route_identity_missing")
            continue
        role = _text(identity.get("role"))
        provider = _text(identity.get("provider"))
        model = _text(identity.get("model"))
        location = _text(identity.get("location"))
        if (
            role not in LOCAL_ROLES
            or provider != "local"
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
        lease = row.get("lease")
        if (
            not isinstance(lease, Mapping)
            or lease.get("kind") not in {"LocalStructuredSession", "LLMRuntime"}
            or lease.get("foreground") is not True
            or lease.get("inflight") != 1
        ):
            reasons.add("foreground_lease_invalid")
        live = row.get("live_recall")
        if (
            not isinstance(live, Mapping)
            or live.get("unaffected") is not True
            or live.get("remote_egress") != 0
        ):
            reasons.add("live_recall_egress")
        outcome = row.get("outcome")
        if not isinstance(outcome, Mapping):
            reasons.add("outcome_missing")
            continue
        outcome_class = _text(outcome.get("class"))
        reason = _text(outcome.get("reason"))
        if outcome_class not in _OUTCOME_CLASSES:
            reasons.add("outcome_class_invalid")
        elif outcome_class == "valid":
            valid += 1
            if (
                outcome.get("schema_valid") is not True
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
        if reason:
            categories.add(reason)
        if row.get("max_inflight") != 1:
            reasons.add("local_inflight_not_one")
        if row.get("failure_injection") is not True:
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


def _parse_expiry(value: object) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return None
    return None


def _validate_ox(
    receipts: Sequence[Mapping[str, Any]], source: Mapping[str, Any]
) -> dict[str, Any]:
    reasons: set[str] = set()
    rows = [row for row in receipts if row.get("profile") == OX_PROFILE]
    if len(rows) != len(receipts):
        reasons.add("profile_mixing")
    identity: dict[str, Any] | None = None
    stages: dict[int, dict[str, Any]] = {}
    all_label_ids: list[str] = []
    all_commit_ids: list[str] = []
    all_work_ids: list[str] = []
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
        if not isinstance(contract, Mapping):
            reasons.add("contract_missing")
        else:
            required = {
                "route",
                "model",
                "prompt_sha256",
                "schema",
                "contract_id",
                "expires_at",
                "schema_sha256",
                "route_sha256",
                "model_sha256",
                "cohort",
                "identity_revision",
            }
            if not required.issubset(contract):
                reasons.add("contract_identity_incomplete")
            if (
                contract.get("route") != OX_ROUTE
                or contract.get("model") != OX_MODEL
                or contract.get("schema") != OX_SCHEMA
                or contract.get("prompt_sha256") != OX_PROMPT_SHA256
                or contract.get("schema_sha256") != OX_SCHEMA_SHA256
                or contract.get("route_sha256") != OX_ROUTE_SHA256
                or contract.get("model_sha256") != OX_MODEL_SHA256
                or contract.get("cohort") != OX_COHORT
                or contract.get("identity_revision") != OX_IDENTITY_REVISION
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
                    "expires_at",
                )
            }
            if contract.get("contract_id") != _sha256(contract_identity):
                reasons.add("ox_contract_digest_unbound")
            source_identity = _text(contract.get("source_identity_sha256"))
            expected_source_identity = _text(source.get("ox_identity_sha256"))
            if expected_source_identity and source_identity != expected_source_identity:
                reasons.add("ox_source_identity_mismatch")
            expiry = _parse_expiry(contract.get("expires_at"))
            if expiry is None or expiry <= datetime.now(UTC).timestamp():
                reasons.add("ox_contract_expired_or_missing")
        control = row.get("control")
        if (
            not isinstance(control, Mapping)
            or control.get("ox_enabled") is not True
            or control.get("free_only") is not True
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
                    ):
                        reasons.add("ox_label_identity_invalid")
                    if len(stage_label_ids) != count:
                        reasons.add("ox_label_count_mismatch")
                    else:
                        all_label_ids.extend(stage_label_ids)
                        all_commit_ids.extend(stage_commit_ids)
                        all_work_ids.extend(work_ids)
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
            if set(details) != {"kind", "count", "selection_sha256"}:
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
            if set(details) not in (expected_details, timed_details):
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
            ):
                raise R4Error("production workset commit receipt is invalid")
            if set(details) == timed_details and (
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
        previous = receipt_sha
        prior_after = {
            "counts": after_counts,
            "watermark": after.get("watermark"),
        }
    if prior_after is None or prior_after["counts"] != dict(current_counts):
        raise R4Error("production workset receipt final state mismatch")
    if prior_after["watermark"] != current_watermark:
        raise R4Error("production workset receipt final watermark mismatch")
    return {
        "count": len(rows),
        "generation": len(rows),
        "head_sha256": previous,
        "verified": True,
    }


def _production_workset(path: Path) -> dict[str, Any]:
    """Read and verify the managed SQLite workset without opening a writer."""

    before_files = _production_sqlite_state(path, label="production workset")
    if before_files["main"]["st_size"] > PRODUCTION_MAX_SQLITE_BYTES:
        raise R4Error("production workset exceeds bounded size")
    try:
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
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
            "temporal_split_json, provenance_json, state, attempt_count, "
            "completion_ref, completion_digest FROM work_items ORDER BY sequence"
        ).fetchall()
        if len(rows) != row_count:
            raise R4Error("production workset row count changed during read")
        work_ids: set[str] = set()
        completed: dict[str, dict[str, Any]] = {}
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
                state,
                attempt_count,
                completion_ref,
                completion_digest,
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
                or state not in PRODUCTION_WORKSET_STATES
                or isinstance(attempt_count, bool)
                or not isinstance(attempt_count, int)
                or attempt_count < 0
            ):
                raise R4Error("production workset item identity is invalid")
            try:
                temporal = json.loads(temporal_json)
                provenance = json.loads(provenance_json)
            except (ValueError, TypeError, UnicodeError, RecursionError) as exc:
                raise R4Error("production workset metadata is invalid") from exc
            if not isinstance(temporal, Mapping) or not isinstance(provenance, Mapping):
                raise R4Error("production workset metadata is invalid")
            expected_provenance = {
                key: provenance.get(key)
                for key in ("cohort", "profile", "profile_contract_id", "route")
            }
            if (
                expected_provenance["cohort"] != OX_COHORT
                or expected_provenance["profile"] != OX_PROFILE
                or expected_provenance["route"] != OX_ROUTE
                or _SHA.fullmatch(str(expected_provenance["profile_contract_id"]))
                is None
            ):
                raise R4Error("production workset provenance is invalid")
            if provenance_identity is None:
                provenance_identity = dict(expected_provenance)
            elif provenance_identity != expected_provenance:
                raise R4Error("production workset provenance is mixed")
            work_ids.add(work_id)
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
        receipt = _production_workset_receipts(connection, counts, watermark)
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


def _load_production_anchor(source_root: Path) -> dict[str, Any]:
    """Read the tracked R0 candidate baseline as an immutable ledger anchor."""

    path = source_root / R0_EVIDENCE_RELATIVE
    _reject_original_symlinks((("R0 evidence", path),))
    data, state, file_sha256 = _production_file_bytes(
        path, label="R0 production evidence", max_bytes=_MAX_RECEIPT_BYTES
    )
    try:
        payload = json.loads(data)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise R4Error("R0 production evidence JSON is invalid") from exc
    if not isinstance(payload, dict):
        raise R4Error("R0 production evidence is not an object")
    if payload.get("schema") != R0_SCHEMA:
        raise R4Error("R0 production evidence schema is invalid")
    try:
        verified = _verify_seal(payload, schema=R0_SCHEMA)
    except R4Error as exc:
        raise R4Error("R0 production evidence seal is invalid") from exc
    if verified.get("artifact_id") != R0_EVIDENCE_ID:
        raise R4Error("R0 production evidence artifact id mismatch")
    unsigned = {
        key: value
        for key, value in verified.items()
        if key not in {"artifact_id", "seal_sha256"}
    }
    if _sha256(unsigned) != R0_EVIDENCE_ID:
        raise R4Error("R0 production evidence content digest mismatch")
    canonical = _json_bytes(verified)
    if data not in {canonical, canonical + b"\n"}:
        raise R4Error("R0 production evidence is not canonical")
    if _production_stat(path, label="R0 production evidence") != state:
        raise R4Error("R0 production evidence changed during validation")
    production = verified.get("production")
    ledgers = production.get("ledgers") if isinstance(production, Mapping) else None
    candidate = (
        ledgers.get("candidate-ledger.jsonl")
        if isinstance(ledgers, Mapping)
        else None
    )
    if not isinstance(candidate, Mapping):
        raise R4Error("R0 candidate ledger anchor is missing")
    head = candidate.get("head_sha256")
    records = candidate.get("records")
    bytes_value = candidate.get("bytes")
    file_state = candidate.get("file_state")
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
        raise R4Error("R0 candidate ledger anchor is invalid")
    return {
        "artifact_id": R0_EVIDENCE_ID,
        "file_sha256": file_sha256,
        "seal_sha256": str(verified["seal_sha256"]),
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
    anchor_file_state = expected.get("file_state")
    checkpoint_file_state = checkpoint.get("file_state")
    if (
        not isinstance(anchor_file_state, Mapping)
        or not isinstance(checkpoint_file_state, Mapping)
        or dict(checkpoint_file_state) != dict(anchor_file_state)
        or records != anchor_records
        or current_bytes != anchor_bytes
        or head != anchor_head
    ):
        # A new append requires a fresh offline R0 snapshot.  We deliberately
        # do not trust a current checkpoint to authenticate an unanchored tail.
        raise R4Error("production candidate ledger differs from sealed R0 anchor")
    if _production_stat(path, label="production candidate ledger") != dict(ledger_state):
        raise R4Error("production candidate ledger changed during anchor validation")
    return {
        "anchor_artifact_id": anchor.get("artifact_id"),
        "anchor_head_sha256": anchor_head,
        "anchor_records": anchor_records,
        "anchor_bytes": anchor_bytes,
        "tail_records": 0,
        "tail_bytes": 0,
        "tail_verified": True,
    }


def _production_chain(path: Path, checkpoint_path: Path) -> dict[str, Any]:
    """Verify a bounded label-ledger view against its sealed head checkpoint.

    A small ledger is checked in full.  Once it exceeds the hot-path bound,
    only the bounded tail is parsed; the sealed checkpoint supplies the
    historical record count, head digest, and immutable file identity.
    """

    checkpoint = _production_ledger_checkpoint(
        path, checkpoint_path, ledger_name="label-ledger.jsonl"
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
        "ox_free_only": True,
    }
    for key, expected in required_config.items():
        if distillation.get(key) != expected:
            reasons.add(f"production_config_{key}_invalid")
    relevant_config = {
        key: distillation.get(key)
        for key in (
            "teacher_profile",
            "teacher_max_inflight",
            "ox_enabled",
            "ox_free_only",
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
    if (
        source.get("account_uid") != ACCOUNT_UID
        or source.get("account_home") != str(ACCOUNT_HOME)
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
        "candidate_checkpoint_file_state": candidate.get("file_state"),
        "candidate_anchor_artifact_id": candidate.get("anchor_artifact_id"),
        "candidate_anchor_head_sha256": candidate.get("anchor_head_sha256"),
        "candidate_anchor_records": candidate.get("anchor_records"),
        "candidate_anchor_bytes": candidate.get("anchor_bytes"),
        "candidate_tail_records": candidate.get("tail_records"),
        "candidate_tail_bytes": candidate.get("tail_bytes"),
        "label_receipt_head": labels.get("head_sha256"),
        "label_checkpoint_records": labels.get("count"),
        "label_checkpoint_file_state": labels.get("checkpoint_file_state"),
    }
    if labels.get("sha256") is not None:
        expected_runtime["label_sha256"] = labels.get("sha256")
    for key, expected in expected_runtime.items():
        if runtime.get(key) != expected:
            reasons.add(f"production_runtime_{key}_mismatch")
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
        "request_model": OX_MODEL,
        "required_returned_model": OX_MODEL,
        "free_only": True,
        "no_paid_fallback": True,
    }
    for key, expected in profile_contract.items():
        if contract.get(key) != expected:
            reasons.add(f"production_contract_{key}_invalid")
    if contract.get("max_inflight") != 10:
        reasons.add("production_contract_max_inflight_invalid")
    expiry = _parse_expiry(contract.get("expires_at"))
    if expiry is None or expiry <= datetime.now(UTC).timestamp():
        reasons.add("production_contract_expired_or_missing")
    contract_id = _text(state.get("profile_contract_id"))
    if (
        _SHA.fullmatch(contract_id) is None
        or contract.get("artifact_id") != contract_id
        or workset.get("provenance", {}).get("profile_contract_id") != contract_id
    ):
        reasons.add("production_profile_contract_binding_invalid")
    return reasons


def _production_quality(
    *,
    state: Mapping[str, Any],
    workset: Mapping[str, Any],
    labels: Mapping[str, Any],
    source: Mapping[str, Any],
    contract_id: str,
) -> tuple[set[str], dict[str, Any]]:
    reasons: set[str] = set()
    label_rows = labels.get("rows")
    if not isinstance(label_rows, list) or not label_rows:
        return {"production_labels_missing"}, {"stages": {}}
    completed = workset.get("completed")
    if not isinstance(completed, Mapping):
        return {"production_completed_inventory_missing"}, {"stages": {}}
    label_by_digest: dict[str, Mapping[str, Any]] = {}
    label_by_work: dict[str, Mapping[str, Any]] = {}
    label_ids: set[str] = set()
    commit_ids: set[str] = set()
    stage_work_ids: dict[int, list[str]] = {cap: [] for cap in PRODUCTION_RAMP_CAPS}
    for row in label_rows:
        if not isinstance(row, Mapping):
            reasons.add("production_label_invalid")
            continue
        digest = _text(row.get("record_sha256"))
        work_id = _text(row.get("work_id"))
        label_id = _text(row.get("label_id"))
        commit_id = _text(row.get("commit_id"))
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
            or row.get("teacher_role") != "recall.distill.teacher.ox-alpha"
            or row.get("identity_revision") != OX_IDENTITY_REVISION
            or row.get("route_digest") != OX_ROUTE_SHA256
            or row.get("model_digest") != OX_MODEL_SHA256
            or row.get("prompt_sha256") != OX_PROMPT_SHA256
            or row.get("schema_sha256") != OX_SCHEMA_SHA256
            or row.get("source_ox_identity_sha256") != source.get("ox_identity_sha256")
            or not isinstance(identity, Mapping)
            or dict(identity)
            != {"provider": "opencode-go", "model": OX_ROUTE, "location": "remote"}
            or _SHA.fullmatch(digest) is None
            or _SHA.fullmatch(work_id) is None
            or _SHA.fullmatch(label_id) is None
            or _SHA.fullmatch(commit_id) is None
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
        work = completed.get(work_id)
        if (
            not isinstance(work, Mapping)
            or work.get("completion_digest") != digest
            or isinstance(work.get("attempt_count"), bool)
            or not isinstance(work.get("attempt_count"), int)
            or work.get("attempt_count", 0) < 1
            or row.get("attempt_count") != work.get("attempt_count")
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
            stage_work_ids[cap].append(work_id)
    if set(label_by_work) != set(completed):
        reasons.add("production_completed_label_set_mismatch")
    ramp = state.get("ramp_receipts")
    if not isinstance(ramp, list) or len(ramp) != len(PRODUCTION_RAMP_CAPS):
        reasons.add("production_ramp_receipts_missing")
        ramp = []
    stages: dict[str, Any] = {}
    seen_caps: set[int] = set()
    ramp_caps: list[int] = []
    for row in ramp:
        if not isinstance(row, Mapping):
            reasons.add("production_ramp_receipt_invalid")
            continue
        cap = row.get("cap")
        if isinstance(cap, int) and not isinstance(cap, bool):
            ramp_caps.append(cap)
        valid = row.get("valid_receipts")
        attempts = row.get("attempts")
        work_ids = row.get("work_ids")
        if (
            isinstance(cap, bool)
            or not isinstance(cap, int)
            or cap not in PRODUCTION_RAMP_CAPS
            or cap in seen_caps
            or isinstance(valid, bool)
            or not isinstance(valid, int)
            or isinstance(attempts, bool)
            or not isinstance(attempts, int)
            or valid < 20
            or attempts < valid
            or valid / attempts < 0.95
            or not isinstance(work_ids, list)
            or len(work_ids) != valid
            or len(set(work_ids)) != len(work_ids)
            or set(work_ids) != set(stage_work_ids.get(cap, []))
            or row.get("source_commit") != source.get("commit")
            or row.get("profile_contract_id") != contract_id
        ):
            reasons.add("production_ramp_quality_invalid")
            continue
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
    if workset.get("counts", {}).get("leased") != 0:
        reasons.add("production_leased_work_present")
    for key in (
        "sensitive",
        "raw",
        "billable",
        "unexpected_route",
        "duplicate_label",
        "duplicate_commit",
    ):
        value = state.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value != 0:
            reasons.add(f"production_{key}_veto_invalid")
    lease_recovery = state.get("lease_recovery")
    recovered = (
        lease_recovery.get("recovered") if isinstance(lease_recovery, Mapping) else None
    )
    if (
        not isinstance(lease_recovery, Mapping)
        or isinstance(lease_recovery.get("leased_after"), bool)
        or not isinstance(lease_recovery.get("leased_after"), int)
        or lease_recovery.get("leased_after") != 0
        or isinstance(recovered, bool)
        or not isinstance(recovered, int)
        or recovered < 0
    ):
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
    transitions = state.get("failure_receipts")
    expected_failures = {
        "429": False,
        "5xx": False,
        "timeout": False,
        "402": False,
        "paid": False,
        "model_drift": False,
    }
    if not isinstance(transitions, list):
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
        else:
            expected_failures[category] = True
    if not all(expected_failures.values()):
        reasons.add("production_failure_coverage_incomplete")
    return reasons, {"stages": stages, "labels": len(label_rows)}


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
        directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(
            os, "O_NOFOLLOW", 0
        )
        cwd_fd = os.open(".", directory_flags)
        root_fd = os.open(str(original_root), directory_flags)
        # ponytail: this one-shot CLI uses process-global cwd; concurrent
        # embedding must move the same reads to openat(dir_fd=...) instead.
        os.fchdir(root_fd)
        collection_root = Path(".")
        if _production_directory_fd_identity(root_fd, label="production root") != root_identity:
            raise R4Error("production root changed while opening")
        if _has_symlink_component(source_root):
            unavailable["reasons"] = ["source_root_contains_symlink"]
            return unavailable
        source_identity = _source_ox_identity(source_root)
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
        candidate_anchor = _load_production_anchor(source_root)
        candidate_tail = _production_candidate_tail(
            candidate_path, candidate, candidate_anchor
        )
        candidate = {**candidate, **candidate_tail}
        labels = _production_chain(label_path, label_checkpoint_path)
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
            schema="chronovisor.recall-distill-ox-profile.v1",
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
        )
        quality_reasons, quality = _production_quality(
            state=state,
            workset=workset,
            labels=labels,
            source=source,
            contract_id=contract_id,
        )
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
    if path.is_symlink():
        raise R4Error("immutable artifact path is a symlink")
    if path.exists() and path.read_bytes() != encoded:
        raise R4Error("immutable artifact conflict")
    if not path.exists():
        fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=output)
        try:
            with os.fdopen(fd, "wb") as handle:
                os.fchmod(handle.fileno(), 0o600)
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                if Path(temporary).read_bytes() != encoded:
                    raise R4Error("immutable artifact readback mismatch")
            except OSError as exc:
                raise R4Error("immutable artifact readback failed") from exc
            if before_publish is not None:
                before_publish()
            os.replace(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)
    return artifact_id, path, artifact


def read_artifact(path: Path) -> dict[str, Any]:
    """Verify one R4 artifact's serialization, identity, and integrity seal.

    This is an integrity/readback check, not an independent production
    authority.  Production certification still requires an external
    completion/watchdog evidence chain.
    """

    _reject_original_symlinks((("R4 artifact", path),))
    before = _stat(path)
    try:
        data = path.read_bytes()
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
        {key: value for key, value in artifact.items() if key not in {"artifact_id", "seal_sha256"}}
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
    snapshots = [artifact.get(name) for name in ("source", "source_after", "source_final")]
    if any(not isinstance(snapshot, Mapping) or not source_keys.issubset(snapshot) for snapshot in snapshots):
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
    if not isinstance(source_contract, Mapping) or not {
        "schema", "passed", "local", "ox"
    }.issubset(source_contract):
        raise R4Error("R4 artifact source contract is missing")
    if (
        not isinstance(source_contract.get("passed"), bool)
        or not isinstance(source_contract.get("local"), Mapping)
        or not isinstance(source_contract.get("ox"), Mapping)
    ):
        raise R4Error("R4 artifact source contract is invalid")
    production = artifact.get("production_certification")
    if not isinstance(production, Mapping) or not {
        "passed", "reasons", "collector", "provider_calls"
    }.issubset(production):
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
        "local", "ox", "production"
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
    if (
        isinstance(artifact.get("provider_calls"), bool)
        or not isinstance(artifact.get("provider_calls"), int)
        or artifact.get("provider_calls") != 0
        or not isinstance(artifact.get("production_root_used"), bool)
    ):
        raise R4Error("R4 artifact provider/root contract is invalid")
    if _stat(path) != before:
        raise R4Error("R4 artifact changed during read")
    return artifact


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
    production = production_root.expanduser().absolute() if production_root else None
    input_roots = [
        path.expanduser().resolve(strict=True)
        for path in (local_receipts, ox_receipts)
        if path is not None
    ]
    assert_root_matrix(source_root, output, production, input_roots)
    source_before = _assert_source(source_root, source_commit)
    local, local_files = load_receipts(local_receipts)
    ox, ox_files = load_receipts(ox_receipts)
    # Arbitrary production JSON is never an input to certification.  Keep the
    # old parameter only as a compatibility tripwire for callers that still
    # pass it; the resulting verdict remains false and no file is read.
    production_files: dict[str, Any] = {"files": [], "count": 0}
    local_result = (
        _validate_local(local, source_before)
        if local
        else {"passed": False, "reasons": ["local_receipts_missing"], "rows": 0}
    )
    ox_result = (
        _validate_ox(ox, source_before)
        if ox
        else {"passed": False, "reasons": ["ox_receipts_missing"], "rows": 0}
    )
    source_after = _assert_source(source_root, source_commit)
    if source_after != source_before:
        raise R4Error("source changed during evidence validation")
    if production is not None:
        production_result = _collect_authoritative_production(
            source_root=source_root,
            source=source_before,
            production_root=production,
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
    # The collector reads a large amount of runtime state.  Re-snapshot the
    # source immediately before publishing the artifact so a source mutation
    # during collection cannot be hidden by the earlier before/after pair.
    source_final = _assert_source(source_root, source_commit)
    if source_final != source_before or source_final != source_after:
        raise R4Error("source changed during final evidence validation")
    source_passed = bool(
        local_result["passed"]
        and ox_result["passed"]
        and source_before["clean"]
        and source_after == source_before
        and source_final == source_before
    )
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
        },
        "production_certification": {
            **production_result,
            "passed": bool(source_passed and production_result["passed"]),
        },
        "receipt_files": {
            "local": local_files,
            "ox": ox_files,
            "production": production_files,
        },
        "provider_calls": 0,
        "production_root_used": production is not None,
    }
    expected_unsigned = {
        "schema": R4_SCHEMA,
        "namespace": "recall-distillation",
        **payload,
    }
    expected_artifact_path = output / f"{_sha256(expected_unsigned)}.json"
    artifact_preexisted = (
        expected_artifact_path.exists() or expected_artifact_path.is_symlink()
    )

    def _verify_publication_source() -> None:
        nonlocal source_final
        checked = _assert_source(source_root, source_commit)
        if checked != source_before or checked != source_after:
            raise R4Error("source changed during artifact publication")
        source_final = checked

    artifact_id, artifact_path, artifact = _write_immutable(
        output, payload, before_publish=_verify_publication_source
    )
    try:
        checked_after_publish = _assert_source(source_root, source_commit)
    except R4Error as exc:
        # Never remove a pre-existing immutable artifact.  If this invocation
        # just published a matching regular file, clean up only that file.
        if not artifact_preexisted and not artifact_path.is_symlink():
            try:
                if artifact_path.read_bytes() == _json_bytes(artifact) + b"\n":
                    artifact_path.unlink()
            except OSError:
                pass
        raise R4Error("source changed after artifact publication") from exc
    if checked_after_publish != source_before or checked_after_publish != source_final:
        if not artifact_preexisted and not artifact_path.is_symlink():
            try:
                if artifact_path.read_bytes() == _json_bytes(artifact) + b"\n":
                    artifact_path.unlink()
            except OSError:
                pass
        raise R4Error("source changed after artifact publication")
    return artifact, artifact_path


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
    args = parser.parse_args(argv)
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
