#!/usr/bin/env python3
"""Fail-closed, source-bound evidence harness for Recall R4.

The harness validates receipts; it does not run a teacher or make a provider
request.  A source-contract verdict is intentionally independent from the
production verdict: synthetic receipts can exercise the contract, while
production certification stays disabled until an independently authenticated
runtime/resource/process/egress evidence chain is available.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import subprocess
import sys
import tempfile
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

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
    for row in rows:
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


def _validate_production_attestations(
    receipts: Sequence[Mapping[str, Any]], source: Mapping[str, Any]
) -> dict[str, Any]:
    """Keep production certification disabled until a trusted chain exists.

    JSON seals and a same-process collector are replayable claims, not durable
    provider/runtime attestations.  The harness therefore never promotes them
    to production certification.  Source-contract validation remains usable via
    ``--source-contract-only`` while a future implementation can replace this
    gate with a separately authenticated provider evidence chain.
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
    output: Path, payload: Mapping[str, Any]
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
            os.replace(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)
    return artifact_id, path, artifact


def read_artifact(path: Path) -> dict[str, Any]:
    """Read one R4 artifact and verify its immutable identity and seal."""

    before = _stat(path)
    try:
        artifact = json.loads(path.read_bytes())
    except (OSError, ValueError, UnicodeError) as exc:
        raise R4Error("R4 artifact is not valid JSON") from exc
    if not isinstance(artifact, dict) or artifact.get("schema") != R4_SCHEMA:
        raise R4Error("R4 artifact schema mismatch")
    _verify_seal(artifact, schema=R4_SCHEMA)
    if artifact.get("artifact_id") != _sha256(
        {
            key: value
            for key, value in artifact.items()
            if key not in {"artifact_id", "seal_sha256"}
        }
    ):
        raise R4Error("R4 artifact identity mismatch")
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
    if production_root is not None:
        original_paths.append(("production", production_root))
    for name, value in (
        ("local_receipts", local_receipts),
        ("ox_receipts", ox_receipts),
        ("production_receipts", production_receipts),
    ):
        if value is not None:
            original_paths.append((name, value))
    _reject_original_symlinks(original_paths)
    source_root = source_root.expanduser().resolve(strict=True)
    output = output.expanduser().resolve(strict=False)
    production = (
        production_root.expanduser().resolve(strict=True) if production_root else None
    )
    input_roots = [
        path.expanduser().resolve(strict=True)
        for path in (local_receipts, ox_receipts, production_receipts)
        if path is not None
    ]
    assert_root_matrix(source_root, output, production, input_roots)
    source_before = _assert_source(source_root, source_commit)
    local, local_files = load_receipts(local_receipts)
    ox, ox_files = load_receipts(ox_receipts)
    production_rows, production_files = load_receipts(production_receipts)
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
    production_result = (
        _validate_production_attestations(production_rows, source_before)
        if production_rows
        else {
            "passed": False,
            "reasons": ["independent_live_provider_attestation_unavailable"],
            "kinds": [],
            "producers": [],
        }
    )
    source_passed = bool(
        local_result["passed"]
        and ox_result["passed"]
        and source_before["clean"]
        and source_after == source_before
    )
    payload = {
        "captured_at": datetime.now(UTC).isoformat(),
        "source": source_before,
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
    artifact_id, artifact_path, artifact = _write_immutable(output, payload)
    return artifact, artifact_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--local-receipts", type=Path)
    parser.add_argument("--ox-receipts", type=Path)
    parser.add_argument("--production-receipts", type=Path)
    parser.add_argument("--production-root", type=Path)
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
            production_receipts=args.production_receipts,
            production_root=args.production_root,
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
