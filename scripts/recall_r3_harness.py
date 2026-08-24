#!/usr/bin/env python3
"""Run the offline Recall R3 workset durability/performance gate.

The default path takes a production root, creates the existing forced APFS
clone, and mutates only that clone.  The legacy frozen-clone plus manifest
options fail closed as non-certifying because their provenance is unkeyed.  No
teacher/provider is called; evidence is sealed with the normal immutable-
artifact writer and contains bounded digests/counts, never work payloads.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import selectors
import shutil
import signal
import sqlite3
import stat
import subprocess
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

R3_SCHEMA = "chronovisor.recall-r3.v1"
R3_CLONE_SCHEMA = "chronovisor.recall-r3-workset-clone.v1"
R3_COMPLETION_SCHEMA = "chronovisor.recall-r3-completion.v1"
R2_FROZEN_CLONE_SCHEMA = "chronovisor.recall-r2-frozen-clone.v1"
DEFAULT_SAMPLES = 100
MIN_SAMPLES = 100
UNIT_MIN_SAMPLES = 20
CLAIM_P95_LIMIT_NS = 500_000_000
TEACHER_HANDOFF_LIMIT_NS = 10_000_000_000
RECEIPT_COVERAGE_LIMIT = 99.0
OX_WORKSET_RELATIVE = Path("runtime") / "recall-distillation" / "ox-workset.sqlite3"
OX_WORKSET_EXPECTED_ROWS = 32_522
OX_WORKSET_EXPECTED_STATES = {
    "ready": 19_400,
    "leased": 0,
    "completed": 152,
    "quarantined": 12_970,
}
OX_WORKSET_ROW_LIMIT = 100_000
CLONE_TREE_FILE_LIMIT = 100_000
CLONE_TREE_FILE_BYTES_LIMIT = 512 * 1024 * 1024
# A formal run may inventory production-sized files, but it must never read an
# unbounded amount of append-only data merely to prove clone identity.  The
# bounded body budget applies only to files represented by a content digest;
# Raw and sealed ledgers use their existing R1/R2 checkpoints instead.
# Keep the total body read bounded while leaving room for the existing
# sub-512MiB FTS/workset files in a production clone.
CLONE_TREE_HASH_BYTES_LIMIT = 1024 * 1024 * 1024
CLONE_TREE_DIGEST_REPRESENTATION = "bounded-content+sealed-checkpoint-v1"
CLONE_TREE_RAW_REPRESENTATION = "r2.raw-tree-state+committed-watermark-v1"
CLONE_TREE_LEDGER_REPRESENTATION = "r2.sealed-ledger-checkpoint-v1"
CLONE_TREE_CATALOG_REPRESENTATION = "r2.sealed-catalog-checkpoint-v1"
R3_WORKSET_SCOPE_REPRESENTATION = "workset-security-columns+sealed-state-v1"
R3_EXCLUDED_NOT_EVALUATED = {
    "raw": "ingest-owned; R2/R5 sealed watermark boundary",
    "candidate_ledgers": "R2/R5 sealed ledger checkpoints",
    "historical_catalog": "R2/R5 catalog ownership",
    "historical_fts": "R2/R5 FTS ownership",
    "unrelated_runtime": "outside R3 Workset/state/pointer/lock scope",
}
_WORKSET_SECURITY_COLUMNS = (
    "sequence",
    "work_id",
    "kind",
    "payload_ref",
    "payload_digest",
    "temporal_split_json",
    "provenance_json",
    "priority",
    "watermark_json",
    "stage",
    "state",
    "attempt_count",
    "last_error_class",
    "lease_id",
    "lease_owner",
    "lease_expires_at",
    "next_attempt_at",
    "completion_ref",
    "completion_digest",
    "created_at",
    "updated_at",
)
SIX_STAGES = (
    "snapshot",
    "teacher",
    "counterfactual",
    "retry_wait",
    "dataset",
    "evaluation",
)
_COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
_DIGEST_RE = re.compile(r"[0-9a-f]{64}\Z")
_TREE_RE = re.compile(r"[0-9a-f]{40}\Z")
_FROZEN_MANIFEST_KEYS = {
    "artifact_id",
    "captured_at",
    "clone",
    "clone_root",
    "namespace",
    "production",
    "runtime_identity",
    "schema",
    "seal_sha256",
    "source_commit",
    "source_tree",
    "threshold",
}


class R3Error(ValueError):
    """An R3 gate failed closed."""


def _load_r2() -> Any:
    """Load clone/tree helpers without making ``scripts`` a package."""

    path = Path(__file__).with_name("recall_r2_harness.py")
    spec = importlib.util.spec_from_file_location("chronovisor_r2_harness", path)
    if spec is None or spec.loader is None:
        raise R3Error("R2 helper unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


R2 = _load_r2()


def _load_runtime(source_root: Path) -> tuple[Any, Any]:
    """Bind the workset/store modules to the requested source checkout."""

    source_path = str(source_root / "src")
    if not (source_root / "src" / "chronovisor").is_dir():
        raise R3Error("source root does not contain src/chronovisor")
    if source_path in sys.path:
        sys.path.remove(source_path)
    sys.path.insert(0, source_path)
    try:
        workset = __import__(
            "chronovisor.recall.recall_distillation_workset",
            fromlist=["DistillationWorkset"],
        )
        store = __import__(
            "chronovisor.recall.recall_distillation_store",
            fromlist=["write_immutable"],
        )
    except (ImportError, OSError) as exc:
        raise R3Error("R3 runtime modules are unavailable") from exc
    for module in (workset, store):
        module_file = getattr(module, "__file__", None)
        if not isinstance(module_file, str):
            raise R3Error("R3 runtime module has no source path")
        module_path = Path(module_file).resolve()
        if not module_path.is_relative_to(source_root / "src"):
            raise R3Error("R3 runtime module escaped source root")
    return workset, store


def _hex_digest(value: object, field: str) -> str:
    if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
        raise R3Error(f"{field} must be a lowercase SHA-256")
    return value


def _tree_sha(value: object, field: str) -> str:
    if not isinstance(value, str) or _TREE_RE.fullmatch(value) is None:
        raise R3Error(f"{field} must be a full Git tree SHA")
    return value


def _bounded_integer(value: object, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise R3Error(f"{field} must be a bounded integer")
    return value


def _manifest_raw_tree(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise R3Error(f"{field} is missing")
    required = {"bytes", "content_sha256", "file_count"}
    if not required.issubset(value):
        raise R3Error(f"{field} is incomplete")
    return {
        "bytes": _bounded_integer(value.get("bytes"), f"{field}.bytes"),
        "content_sha256": _hex_digest(
            value.get("content_sha256"), f"{field}.content_sha256"
        ),
        "file_count": _bounded_integer(value.get("file_count"), f"{field}.file_count"),
    }


def _manifest_static(value: object, field: str) -> dict[str, Any]:
    """Validate the path-neutral R2 clone identity projection.

    R0's ``_clone_identity`` intentionally contains only ledger heads/counts/
    bytes, FTS digest/count/checkpoint seal, pointer seals, state seal, and the
    committed Raw watermark.  Keep this validator shape-aware but compare the
    complete projection later so a trusted manifest cannot omit one binding.
    """

    if not isinstance(value, Mapping):
        raise R3Error(f"{field} is missing")
    ledgers = value.get("ledgers")
    if not isinstance(ledgers, Mapping) or not ledgers:
        raise R3Error(f"{field}.ledgers is incomplete")
    normalized_ledgers: dict[str, Any] = {}
    for name, row in ledgers.items():
        if not isinstance(name, str) or not isinstance(row, Mapping):
            raise R3Error(f"{field}.ledgers is malformed")
        records = _bounded_integer(
            row.get("records"), f"{field}.ledgers.{name}.records"
        )
        head = row.get("head_sha256")
        if head != "":
            head = _hex_digest(head, f"{field}.ledgers.{name}.head_sha256")
        elif records != 0:
            raise R3Error(f"{field}.ledgers.{name}.head_sha256 is empty")
        normalized_ledgers[name] = {
            "records": records,
            "head_sha256": head,
            "bytes": _bounded_integer(
                row.get("bytes"), f"{field}.ledgers.{name}.bytes"
            ),
        }
    raw_watermark = _hex_digest(
        value.get("raw_watermark"), f"{field}.raw_watermark"
    )
    fts = value.get("fts")
    if not isinstance(fts, Mapping):
        raise R3Error(f"{field}.fts is incomplete")
    normalized_fts = {
        "content_sha256": _hex_digest(
            fts.get("content_sha256"), f"{field}.fts.content_sha256"
        ),
        "atom_count": _bounded_integer(
            fts.get("atom_count"), f"{field}.fts.atom_count"
        ),
        "fts_count": _bounded_integer(
            fts.get("fts_count"), f"{field}.fts.fts_count"
        ),
        "checkpoint_seal_sha256": _hex_digest(
            fts.get("checkpoint_seal_sha256"),
            f"{field}.fts.checkpoint_seal_sha256",
        ),
    }
    state = value.get("state")
    if not isinstance(state, Mapping):
        raise R3Error(f"{field}.state is incomplete")
    state_seal = _hex_digest(state.get("seal_sha256"), f"{field}.state.seal_sha256")
    state_fields = state.get("fields")
    if not isinstance(state_fields, Mapping):
        raise R3Error(f"{field}.state.fields is incomplete")
    pointers = value.get("pointers")
    if not isinstance(pointers, Mapping):
        raise R3Error(f"{field}.pointers is incomplete")
    normalized_pointers: dict[str, Any] = {}
    for kind, pointer in pointers.items():
        if not isinstance(kind, str):
            raise R3Error(f"{field}.pointers is malformed")
        if pointer is None:
            normalized_pointers[kind] = None
            continue
        if not isinstance(pointer, Mapping):
            raise R3Error(f"{field}.pointers.{kind} is malformed")
        normalized_pointers[kind] = {
            "policy_id": _hex_digest(
                pointer.get("policy_id"), f"{field}.pointers.{kind}.policy_id"
            ),
            "pointer_seal_sha256": _hex_digest(
                pointer.get("pointer_seal_sha256"),
                f"{field}.pointers.{kind}.pointer_seal_sha256",
            ),
            "policy_seal_sha256": _hex_digest(
                pointer.get("policy_seal_sha256"),
                f"{field}.pointers.{kind}.policy_seal_sha256",
            ),
        }
    return {
        "ledgers": normalized_ledgers,
        "raw_watermark": raw_watermark,
        "fts": normalized_fts,
        "state": {"seal_sha256": state_seal, "fields": dict(state_fields)},
        "pointers": normalized_pointers,
    }


def _read_frozen_manifest(
    path: Path,
    store: Any,
    *,
    clone_root: Path,
    source_commit: str,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    """Read and verify the immutable R2 APFS clone manifest once.

    The manifest is an input contract, never an output that this harness can
    replace.  Both the canonical seal and content ``artifact_id`` are checked;
    the caller re-stats and re-hashes the file at run completion for TOCTOU.
    """

    if _has_symlink_component(path) or not path.is_file():
        raise R3Error("R2 frozen manifest is not a regular file")
    before = _regular_file_state(path)
    try:
        encoded = path.read_bytes()
        raw = json.loads(encoded)
    except (OSError, UnicodeError, ValueError) as exc:
        raise R3Error("R2 frozen manifest is invalid JSON") from exc
    after = _regular_file_state(path)
    if before != after:
        raise R3Error("R2 frozen manifest changed during read")
    if not isinstance(raw, Mapping):
        raise R3Error("R2 frozen manifest is not an object")
    try:
        verified = store.verify_seal(dict(raw), schema=R2_FROZEN_CLONE_SCHEMA)
    except Exception as exc:
        raise R3Error("R2 frozen manifest seal/schema verification failed") from exc
    if set(verified) != _FROZEN_MANIFEST_KEYS:
        raise R3Error("R2 frozen manifest top-level schema is not exact")
    unsigned = {
        key: value for key, value in verified.items() if key not in {"artifact_id", "seal_sha256"}
    }
    try:
        content_id = store.canonical_json_sha256_strict(unsigned)
    except Exception as exc:
        raise R3Error("R2 frozen manifest content identity failed") from exc
    if verified.get("artifact_id") != content_id:
        raise R3Error("R2 frozen manifest artifact identity mismatch")
    if path.name != f"{content_id}.json":
        raise R3Error("R2 frozen manifest filename is not content-addressed")
    if not isinstance(verified.get("captured_at"), str) or not verified["captured_at"]:
        raise R3Error("R2 frozen manifest captured_at is invalid")
    if verified.get("source_commit") != source_commit:
        raise R3Error("R2 frozen manifest source commit mismatch")
    if verified.get("namespace") != "recall-distillation":
        raise R3Error("R2 frozen manifest namespace mismatch")
    clone_value = verified.get("clone")
    if not isinstance(clone_value, Mapping) or set(clone_value) != {
        "clone_backend",
        "filesystem",
        "raw_tree",
        "static",
    }:
        raise R3Error("R2 frozen manifest clone contract is incomplete")
    clone_backend = clone_value.get("clone_backend")
    filesystem = clone_value.get("filesystem")
    if (
        not isinstance(clone_backend, str)
        or "apfs" not in clone_backend.lower()
        or not isinstance(filesystem, str)
        or filesystem.lower() != "apfs"
    ):
        raise R3Error("R2 frozen manifest is not an APFS clone")
    clone_root_value = verified.get("clone_root")
    if not isinstance(clone_root_value, str) or clone_root_value != str(clone_root):
        raise R3Error("R2 frozen manifest clone root does not match input")
    production = verified.get("production")
    if not isinstance(production, Mapping) or set(production) != {"raw_tree", "static"}:
        raise R3Error("R2 frozen manifest production identity is incomplete")
    clone_raw = _manifest_raw_tree(clone_value.get("raw_tree"), "clone.raw_tree")
    production_raw = _manifest_raw_tree(
        production.get("raw_tree"), "production.raw_tree"
    )
    if clone_raw != production_raw:
        raise R3Error("R2 frozen manifest Raw parity is not exact")
    clone_static = _manifest_static(clone_value.get("static"), "clone.static")
    production_static = _manifest_static(production.get("static"), "production.static")
    if clone_static != production_static:
        raise R3Error("R2 frozen manifest static clone parity is not exact")
    runtime_identity = verified.get("runtime_identity")
    if not isinstance(runtime_identity, Mapping) or set(runtime_identity) != {
        "runtime_module_sha256",
        "source_commit",
        "source_tree",
    }:
        raise R3Error("R2 frozen manifest runtime identity is incomplete")
    _hex_digest(
        runtime_identity.get("runtime_module_sha256"),
        "runtime_identity.runtime_module_sha256",
    )
    if runtime_identity.get("source_commit") != source_commit:
        raise R3Error("R2 frozen manifest runtime source commit mismatch")
    _tree_sha(runtime_identity.get("source_tree"), "runtime_identity.source_tree")
    source_tree = verified.get("source_tree")
    if not isinstance(source_tree, Mapping) or set(source_tree) != {
        "git_status_count",
        "git_status_sha256",
        "repo",
        "trees",
    }:
        raise R3Error("R2 frozen manifest source tree is incomplete")
    _bounded_integer(source_tree.get("git_status_count"), "source_tree.git_status_count")
    _hex_digest(source_tree.get("git_status_sha256"), "source_tree.git_status_sha256")
    if not isinstance(source_tree.get("repo"), Mapping) or not isinstance(
        source_tree.get("trees"), Mapping
    ):
        raise R3Error("R2 frozen manifest source tree inventory is incomplete")
    threshold = verified.get("threshold")
    if not isinstance(threshold, Mapping) or set(threshold) != {
        "production_unchanged_during_freeze",
        "raw_parity",
    } or threshold.get("production_unchanged_during_freeze") is not True or threshold.get(
        "raw_parity"
    ) is not True:
        raise R3Error("R2 frozen manifest threshold contract is not certified")
    return dict(verified), before, hashlib.sha256(encoded).hexdigest()


def _has_symlink_component(path: Path) -> bool:
    current = path.expanduser()
    while True:
        if current.is_symlink():
            return True
        parent = current.parent
        if parent == current:
            return False
        current = parent


def _path_overlap(left: Path, right: Path) -> bool:
    left_resolved = left.expanduser().resolve(strict=False)
    right_resolved = right.expanduser().resolve(strict=False)
    return (
        left_resolved == right_resolved
        or left_resolved.is_relative_to(right_resolved)
        or right_resolved.is_relative_to(left_resolved)
    )


def _assert_root_matrix(
    production: Path, source_root: Path, output: Path, clones: tuple[Path, ...] = ()
) -> None:
    """Reject symlink entry points and every protected-root overlap."""

    paths = {
        "production": production,
        "source_root": source_root,
        "output": output,
        **{f"clone[{index}]": value for index, value in enumerate(clones)},
    }
    for name, path in paths.items():
        if _has_symlink_component(path):
            raise R3Error(f"{name} path contains a symlink")
    entries = tuple(paths.items())
    for index, (left_name, left) in enumerate(entries):
        for right_name, right in entries[index + 1 :]:
            if _path_overlap(left, right):
                raise R3Error(f"{left_name}/{right_name} paths overlap")
    for name, path in paths.items():
        if name == "output" or name.startswith("clone["):
            continue
        if not path.is_dir():
            raise R3Error(f"{name} root is not a directory")


def _assert_output_safe(output: Path) -> None:
    if output.exists() and not output.is_dir():
        raise R3Error("output root is not a directory")
    if output.is_dir() and any(path.is_symlink() for path in output.rglob("*")):
        raise R3Error("output tree contains a symlink")


def _assert_input_matrix(
    clone_root: Path, source_root: Path, manifest: Path, output: Path
) -> None:
    paths = {
        "clone_root": clone_root,
        "source_root": source_root,
        "manifest": manifest,
        "output": output,
    }
    for name, path in paths.items():
        if _has_symlink_component(path):
            raise R3Error(f"{name} path contains a symlink")
    entries = tuple(paths.items())
    for index, (left_name, left) in enumerate(entries):
        for right_name, right in entries[index + 1 :]:
            if _path_overlap(left, right):
                raise R3Error(f"{left_name}/{right_name} paths overlap")
    if not clone_root.is_dir():
        raise R3Error("clone root is not a directory")
    if not source_root.is_dir():
        raise R3Error("source root is not a directory")
    if not manifest.is_file():
        raise R3Error("manifest is not a regular file")


def _configured_production_root() -> Path:
    configured = os.environ.get("CHRONOVISOR_ROOT", "").strip()
    return Path(configured).expanduser() if configured else Path.home() / ".chronovisor"


def _assert_external_clone_not_production(clone_root: Path) -> None:
    """Reject an external input that resolves to the live production root."""

    production = _configured_production_root()
    if _path_overlap(clone_root, production):
        raise R3Error("external clone overlaps configured production root")


def _git_head(source_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD"],
            cwd=source_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise R3Error("source commit lookup failed") from exc
    head = result.stdout.strip()
    if _COMMIT_RE.fullmatch(head) is None:
        raise R3Error("source HEAD is not a full commit")
    return head


def _source_snapshot(source_root: Path) -> dict[str, Any]:
    try:
        snapshot = R2._source_tree_digest(source_root)
        if not isinstance(snapshot, dict):
            raise R3Error("source snapshot shape is invalid")
        return dict(snapshot)
    except Exception as exc:
        raise R3Error("source snapshot failed") from exc


def _source_runtime_identity(source_root: Path, source_commit: str) -> dict[str, str]:
    module = source_root / "src" / "chronovisor" / "recall" / "recall_runtime.py"
    if _has_symlink_component(module) or not module.is_file():
        raise R3Error("source runtime module is unavailable")
    before = _regular_file_state(module)
    try:
        module_sha = hashlib.sha256(module.read_bytes()).hexdigest()
        result = subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD^{tree}"],
            cwd=source_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise R3Error("source tree identity lookup failed") from exc
    if before != _regular_file_state(module):
        raise R3Error("source runtime module changed during identity capture")
    return {
        "runtime_module_sha256": module_sha,
        "source_commit": source_commit,
        "source_tree": _tree_sha(result.stdout.strip(), "source runtime tree"),
    }


def _assert_manifest_source_binding(
    manifest: Mapping[str, Any],
    source_snapshot: Mapping[str, Any],
    runtime_identity: Mapping[str, str],
) -> None:
    if manifest.get("source_tree") != dict(source_snapshot):
        raise R3Error("R2 frozen manifest source tree differs from source checkout")
    if manifest.get("runtime_identity") != dict(runtime_identity):
        raise R3Error("R2 frozen manifest runtime identity differs from source")


def _assert_source_clean(snapshot: Mapping[str, Any], *, when: str) -> None:
    if snapshot.get("git_status_count") != 0:
        raise R3Error(f"source checkout is dirty {when}")


def _bounded_raw_identity(root: Path, label: str) -> dict[str, Any]:
    """Capture Raw identity through R2's state/watermark contract only.

    Raw is an ingest-owned append-only boundary and can be multi-gigabyte.  A
    formal R3 run therefore records the existing R2 metadata state and
    committed watermark instead of reopening every payload body.
    """

    try:
        state = R2._raw_tree_state_digest(root)
        raw_store = __import__("chronovisor.core.raw_store", fromlist=["RawStore"])
        watermark = raw_store.committed_raw_watermark(root / "raw")
    except Exception as exc:
        raise R3Error(f"{label} bounded Raw checkpoint failed") from exc
    file_count = _bounded_integer(state.get("file_count"), f"{label}.file_count")
    if file_count > CLONE_TREE_FILE_LIMIT:
        raise R3Error(f"{label} exceeds bounded file inventory")
    raw_bytes = _bounded_integer(state.get("bytes"), f"{label}.bytes")
    state_sha256 = _hex_digest(state.get("state_sha256"), f"{label}.state_sha256")
    watermark = _hex_digest(watermark, f"{label}.raw_watermark")
    return {
        "bytes": raw_bytes,
        "content_sha256": None,
        "file_count": file_count,
        "state_sha256": state_sha256,
        "raw_watermark": watermark,
        "representation": CLONE_TREE_RAW_REPRESENTATION,
        "body_hashed": False,
    }


def _ledger_checkpoint_evidence(
    store: Any, path: Path, *, require_checkpoint_file_state: bool
) -> dict[str, Any]:
    """Read an R2 ledger checkpoint without opening its JSONL body."""

    try:
        bounded = R2._bounded_chain(
            store,
            path,
            require_checkpoint_file_state=require_checkpoint_file_state,
        )
        checkpoint_path = store._chain_checkpoint_path(path)
        checkpoint_file_state = R2.R0._stat(checkpoint_path)
    except Exception as exc:
        raise R3Error(f"sealed ledger checkpoint unavailable: {path.name}") from exc
    if checkpoint_file_state is None:
        raise R3Error(f"sealed ledger checkpoint missing: {path.name}")
    return {
        "ledger_name": path.name,
        "records": int(bounded["records"]),
        "head_sha256": str(bounded["head_sha256"]),
        "bytes": int(bounded["bytes"]),
        "file_state": dict(bounded["file_state"]),
        "checkpoint_file_state": checkpoint_file_state,
        "representation": CLONE_TREE_LEDGER_REPRESENTATION,
        "body_hashed": False,
    }


def _catalog_checkpoint_evidence(
    root: Path, path: Path, store: Any, *, require_checkpoint_file_state: bool
) -> dict[str, Any]:
    """Read the sealed historical-catalog checkpoint without scanning SQLite."""

    try:
        catalog = __import__(
            "chronovisor.recall.recall_distillation_catalog",
            fromlist=["catalog_path"],
        )
        before = R2.R0._stat(path)
        if before is None:
            raise R3Error(f"historical catalog missing: {path.name}")
        checkpoint_path = (
            catalog._catalog_checkpoint_path(root)
            if root.name != "recall-distillation"
            else catalog._index_checkpoint_path(path)
        )
        if not checkpoint_path.exists():
            return _catalog_metadata_evidence(path)
        checkpoint = store.read_sealed(checkpoint_path, schema=store.DISTILLATION_SCHEMA)
        after = R2.R0._stat(path)
        checkpoint_file_state = R2.R0._stat(checkpoint_path)
    except R3Error:
        raise
    except Exception as exc:
        raise R3Error("sealed historical catalog checkpoint unavailable") from exc
    checkpoint_state = checkpoint.get("file_state")
    lineage = checkpoint.get("catalog_lineage")
    if lineage is not None and getattr(catalog, "_catalog_lineage", lambda _: None)(
        checkpoint
    ) is None:
        raise R3Error("historical catalog checkpoint lineage is invalid")
    if (
        checkpoint.get("kind") != "historical-catalog-checkpoint"
        or checkpoint.get("catalog_name") != path.name
        or not isinstance(checkpoint.get("catalog_watermark"), str)
        or not isinstance(checkpoint.get("event_rowid"), int)
        or isinstance(checkpoint.get("event_rowid"), bool)
        or checkpoint.get("event_rowid", -1) < 0
        or not isinstance(checkpoint_state, Mapping)
        or checkpoint_state.get("size_bytes") != before["size_bytes"]
        or (require_checkpoint_file_state and dict(checkpoint_state) != before)
        or after != before
        or checkpoint_file_state is None
    ):
        raise R3Error("historical catalog checkpoint is stale")
    return {
        "catalog_name": path.name,
        "event_rowid": int(checkpoint["event_rowid"]),
        "catalog_watermark": checkpoint["catalog_watermark"],
        "catalog_lineage": lineage,
        "bytes": int(before["size_bytes"]),
        "file_state": before,
        "checkpoint_file_state": checkpoint_file_state,
        "representation": CLONE_TREE_CATALOG_REPRESENTATION,
        "body_hashed": False,
    }


def _catalog_metadata_evidence(path: Path) -> dict[str, Any]:
    """Use the catalog's tiny sealed metadata projection when no checkpoint exists.

    Some legacy production roots predate the catalog checkpoint.  The fallback
    remains bounded: it reads only the metadata table and binds the complete
    file-state, never the 500MiB+ SQLite body.  A later run can upgrade this
    representation once the normal checkpoint is present.
    """

    before = R2.R0._stat(path)
    if before is None:
        raise R3Error(f"historical catalog missing: {path.name}")
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
            rows = tuple(
                (str(key), str(value))
                for key, value in connection.execute(
                    "SELECT key,value FROM metadata ORDER BY key"
                )
            )
        after = R2.R0._stat(path)
    except sqlite3.DatabaseError as exc:
        raise R3Error("historical catalog metadata projection failed") from exc
    if after != before:
        raise R3Error("historical catalog changed during metadata projection")
    return {
        "catalog_name": path.name,
        "metadata_sha256": hashlib.sha256(
            json.dumps(rows, separators=(",", ":")).encode()
        ).hexdigest(),
        "metadata_keys": [key for key, _ in rows],
        "bytes": int(before["size_bytes"]),
        "file_state": before,
        "checkpoint_file_state": None,
        "representation": "r2.catalog-metadata+file-state-v1",
        "body_hashed": False,
    }


def _workset_lock_snapshot(runtime_dir: Path) -> dict[str, Any]:
    """Capture lock file state in the R3-owned boundary."""

    locks: dict[str, Any] = {}
    paths = list(runtime_dir.glob("*.lock"))
    immutable_lock = runtime_dir / ".immutable.lock"
    if immutable_lock.exists():
        paths.append(immutable_lock)
    for path in sorted(set(paths)):
        if _has_symlink_component(path) or not path.is_file():
            raise R3Error("production Workset lock/sidecar path is unsafe")
        before = _regular_file_state(path)
        try:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as exc:
            raise R3Error("production Workset lock state read failed") from exc
        if before != _regular_file_state(path):
            raise R3Error("production Workset lock changed during read")
        locks[path.name] = {**before, "sha256": digest}
    return {"files": locks, "bounded": True}


def _production_snapshot(
    production: Path,
    *,
    include_raw: bool = True,
    store: Any | None = None,
    require_checkpoint_file_state: bool = False,
) -> dict[str, Any]:
    """Digest only R3's protected production boundary.

    The live root intentionally contains unrelated application trees (some may
    be symlinked model/cache paths).  R3 never reads those siblings.  The Raw
    tree and Recall distillation directory are the protected inputs used by
    the Workset/static checks; symlinks inside either boundary remain fatal.
    """

    if _has_symlink_component(production) or not production.is_dir():
        raise R3Error("production root is unsafe")
    runtime_dir = production / "runtime" / "recall-distillation"
    if _has_symlink_component(runtime_dir) or not runtime_dir.is_dir():
        raise R3Error("production protected boundary is unavailable")
    if store is None:
        # Keep the compact fixture path used by unit tests; formal runs always
        # pass the loaded runtime store and use the explicit Workset boundary.
        snapshots = {
            "raw": R2._tree_digest(production / "raw", label="production.raw"),
            "recall_distillation": _clone_tree_state_digest(runtime_dir),
        }
    else:
        owned = _production_owned_snapshot(production, store)
        snapshots = {
            "workset": owned["workset"],
            "state_pointers": owned["state_pointers"],
            "locks": _workset_lock_snapshot(runtime_dir),
        }
    return {
        "scope": "runtime/recall-distillation/ox-workset.sqlite3 + state/pointers/locks",
        "protected": snapshots,
        "excluded_not_evaluated": dict(R3_EXCLUDED_NOT_EVALUATED),
        "excluded_scope": _production_excluded_scope(production),
    }


def _production_excluded_scope(production: Path) -> dict[str, Any]:
    if _has_symlink_component(production) or not production.is_dir():
        raise R3Error("production root is unsafe")
    excluded_names = sorted(
        path.name
        for path in production.iterdir()
        if path.name not in {"raw", "runtime"}
    )
    return {
        "root_siblings": len(excluded_names),
        "root_siblings_sha256": hashlib.sha256(
            json.dumps(excluded_names, separators=(",", ":")).encode()
        ).hexdigest(),
        "root_symlink_count": sum(
            1 for path in production.iterdir() if path.is_symlink()
        ),
        "read": False,
    }


def _production_raw_after_observation(
    production: Path,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Observe Raw after the run, classifying only an in-flight source append."""

    raw_root = production / "raw"
    if _has_symlink_component(raw_root) or not raw_root.is_dir():
        raise R3Error("production Raw root is unsafe")
    try:
        value = _bounded_raw_identity(production, "production.raw_tree.after")
    except Exception as exc:
        messages: list[str] = []
        current: BaseException | None = exc
        while current is not None:
            messages.append(str(current).lower())
            current = current.__cause__
        message = " ".join(messages)
        if not any(
            marker in message
            for marker in ("changed during", "disappeared", "capture")
        ):
            raise R3Error("production Raw observation failed") from exc
        return None, {
            "detected": True,
            "raw_tree_changed": True,
            "classification": "ingest-owned-concurrent",
            "observation_error": type(exc).__name__,
        }
    return value, {
        "detected": False,
        "raw_tree_changed": False,
        "classification": "none",
    }


def _production_protected_equal(
    before: Mapping[str, Any] | None, after: Mapping[str, Any] | None
) -> bool:
    """Compare only the boundary R3 reads; root sibling inventory is excluded."""

    return (
        isinstance(before, Mapping)
        and isinstance(after, Mapping)
        and isinstance(before.get("protected"), Mapping)
        and isinstance(after.get("protected"), Mapping)
        and before["protected"] == after["protected"]
    )


def _production_scope_identity(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize R3-owned state/pointer/lock identity across APFS clones."""

    protected = snapshot.get("protected")
    if not isinstance(protected, Mapping):
        raise R3Error("production Workset scope evidence is incomplete")
    pointers = protected.get("state_pointers")
    locks = protected.get("locks")
    lock_files = locks.get("files") if isinstance(locks, Mapping) else None
    if (
        not isinstance(pointers, Mapping)
        or not isinstance(locks, Mapping)
        or not isinstance(lock_files, Mapping)
    ):
        raise R3Error("production Workset state/pointer/lock evidence is incomplete")
    workset = protected.get("workset")
    if not isinstance(workset, Mapping):
        raise R3Error("production Workset identity evidence is incomplete")
    normalized_pointers: dict[str, Any] = {}
    for name, value in pointers.items():
        if value is None:
            normalized_pointers[str(name)] = None
        elif isinstance(value, Mapping):
            normalized_pointers[str(name)] = {
                key: value.get(key) for key in ("sha256", "seal_sha256")
            }
        else:
            raise R3Error("production pointer identity is malformed")
    normalized_locks = {
        str(name): {
            "st_size": value.get("st_size"),
            "sha256": value.get("sha256"),
        }
        for name, value in lock_files.items()
        if isinstance(value, Mapping)
    }
    return {
        # APFS clone inode/device/mtime values are expected to differ.  Keep
        # the logical/security-column digest and schema only for cross-root
        # parity; each side's raw file state remains in its own TOCTOU probe.
        "workset": _workset_identity(workset),
        "state_pointers": normalized_pointers,
        "locks": normalized_locks,
    }


def _production_owned_snapshot(production: Path, store: Any) -> dict[str, Any]:
    """Capture only the Workset and sealed state R3 is allowed to mutate."""

    workset = _clone_workset_inventory(_clone_workset_path(production))
    runtime_dir = production / "runtime" / "recall-distillation"
    if _has_symlink_component(runtime_dir) or not runtime_dir.is_dir():
        raise R3Error("production distillation directory is unsafe")
    state_pointers: dict[str, Any] = {}
    filenames = (store.STATE_FILE, *store.POINTER_FILES.values())
    for filename in filenames:
        path = runtime_dir / filename
        if not path.exists():
            state_pointers[filename] = None
            continue
        if _has_symlink_component(path) or not path.is_file():
            raise R3Error(f"production sealed state path is unsafe: {filename}")
        file_snapshot = _artifact_file_snapshot(path)
        try:
            sealed = store.read_sealed(path, schema=store.DISTILLATION_SCHEMA)
        except Exception as exc:
            raise R3Error(f"production sealed state is invalid: {filename}") from exc
        if _artifact_file_snapshot(path) != file_snapshot:
            raise R3Error(f"production sealed state changed during read: {filename}")
        seal = sealed.get("seal_sha256")
        _hex_digest(seal, f"production.{filename}.seal_sha256")
        state_pointers[filename] = {
            "file_state": file_snapshot["file_state"],
            "sha256": file_snapshot["sha256"],
            "seal_sha256": seal,
        }
    return {
        "workset": workset,
        "workset_identity": _workset_identity(workset),
        "state_pointers": state_pointers,
    }


def _clone_from_root(source: Path) -> Path:
    """Use the existing forced APFS clone implementation."""

    try:
        clone = R2._clone_from_root(source)
        return Path(clone)
    except Exception as exc:
        raise R3Error(str(exc)) from exc


def _cleanup_clone(path: Path) -> None:
    try:
        R2._cleanup_clone(path)
    except Exception as exc:
        raise R3Error("clone/temp cleanup failed") from exc


def _regular_file_state(path: Path) -> dict[str, int]:
    """Capture bounded file identity without reading the database body."""

    try:
        state = path.lstat()
    except OSError as exc:
        raise R3Error("clone workset file disappeared") from exc
    if stat.S_ISLNK(state.st_mode) or not stat.S_ISREG(state.st_mode):
        raise R3Error("clone workset is not a regular file")
    return {
        "st_dev": int(state.st_dev),
        "st_ino": int(state.st_ino),
        "st_size": int(state.st_size),
        "st_mtime_ns": int(state.st_mtime_ns),
    }


def _clone_workset_path(clone: Path) -> Path:
    """Resolve the production workset only inside the APFS clone."""

    if _has_symlink_component(clone) or not clone.is_dir():
        raise R3Error("clone root is unsafe")
    path = clone / OX_WORKSET_RELATIVE
    if _has_symlink_component(path) or not path.is_file():
        raise R3Error("clone production ox workset is unavailable")
    if not path.resolve(strict=True).is_relative_to(clone.resolve(strict=True)):
        raise R3Error("clone production ox workset escaped clone")
    return path


def _sqlite_sidecar_snapshot(path: Path) -> dict[str, Any]:
    """Capture clone-only SQLite sidecars without trusting their generation."""

    sidecars: dict[str, Any] = {}
    for suffix in ("-wal", "-shm", "-journal"):
        sidecar = path.with_name(path.name + suffix)
        try:
            sidecar.lstat()
        except FileNotFoundError:
            sidecars[suffix] = None
            continue
        if _has_symlink_component(sidecar):
            raise R3Error("clone Workset SQLite sidecar contains a symlink")
        sidecars[suffix] = _regular_file_state(sidecar)
    return sidecars


def _assert_checkpointed_clone_sidecars(path: Path) -> None:
    """Reject untrusted copied WAL/journal frames before any SQLite open."""

    sidecars = _sqlite_sidecar_snapshot(path)
    for suffix in ("-wal", "-journal"):
        state = sidecars[suffix]
        if isinstance(state, Mapping) and state.get("st_size") != 0:
            raise R3Error("clone Workset SQLite sidecar is not checkpointed")


def _normalize_clone_workset(
    path: Path, *, expected_identity: Mapping[str, Any], allow_wal: bool = False
) -> dict[str, Any]:
    """Checkpoint clone-only SQLite WAL state without discarding semantic rows."""

    expected = _workset_identity(expected_identity)
    sidecars_before = _sqlite_sidecar_snapshot(path)
    if not allow_wal:
        for suffix in ("-wal", "-journal"):
            state = sidecars_before[suffix]
            if isinstance(state, Mapping) and state.get("st_size") != 0:
                raise R3Error("clone Workset SQLite sidecar is not checkpointed")
    pre_inventory = _clone_workset_inventory(path)
    if _workset_identity(pre_inventory) != expected:
        raise R3Error("clone Workset identity changed before SQLite normalization")
    before = _regular_file_state(path)
    try:
        with sqlite3.connect(path, timeout=30.0) as connection:
            journal_mode = str(connection.execute("PRAGMA journal_mode=WAL").fetchone()[0])
            checkpoint = tuple(connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone())
            integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
    except sqlite3.DatabaseError as exc:
        raise R3Error("clone Workset SQLite normalization failed") from exc
    if (
        journal_mode.lower() != "wal"
        or len(checkpoint) != 3
        or int(checkpoint[0]) != 0
        or integrity.lower() != "ok"
    ):
        raise R3Error("clone Workset SQLite normalization is not healthy")
    after = _regular_file_state(path)
    sidecars_after = _sqlite_sidecar_snapshot(path)
    if (
        before["st_dev"] != after["st_dev"]
        or before["st_ino"] != after["st_ino"]
    ):
        raise R3Error("clone Workset SQLite identity changed during normalization")
    post_inventory = _clone_workset_inventory(path)
    post_identity = _workset_identity(post_inventory)
    if post_identity != expected or post_identity != _workset_identity(pre_inventory):
        raise R3Error("clone Workset identity changed during SQLite normalization")
    return {
        "journal_mode": journal_mode,
        "wal_checkpoint": [int(value) for value in checkpoint],
        "integrity": integrity,
        "file_state_before": before,
        "file_state_after": after,
        "sidecars_before": sidecars_before,
        "sidecars_after": sidecars_after,
        "semantic_before": _workset_identity(pre_inventory),
        "semantic_after": post_identity,
        "clone_only": True,
    }


def _clone_root_identity(path: Path) -> dict[str, int]:
    try:
        value = path.lstat()
    except OSError as exc:
        raise R3Error("clone root disappeared") from exc
    if stat.S_ISLNK(value.st_mode) or not stat.S_ISDIR(value.st_mode):
        raise R3Error("clone root is not a regular directory")
    return {"st_dev": int(value.st_dev), "st_ino": int(value.st_ino)}


def _probe_apfs_clone(path: Path) -> str:
    """Measure the clone's actual filesystem instead of trusting a manifest."""

    if _has_symlink_component(path) or not path.is_dir():
        raise R3Error("clone root is unsafe")
    try:
        filesystem = R2.R0._filesystem_type(path.resolve(strict=True))
    except Exception as exc:
        raise R3Error("clone filesystem probe failed") from exc
    if filesystem != "apfs":
        raise R3Error("trusted clone is not on APFS")
    return cast(str, filesystem)


def _artifact_file_snapshot(path: Path) -> dict[str, Any]:
    """Hash an artifact while proving its path and file state stayed stable."""

    if _has_symlink_component(path) or not path.is_file():
        raise R3Error("immutable artifact path is unsafe")
    before = _regular_file_state(path)
    try:
        encoded = path.read_bytes()
    except (OSError, UnicodeError) as exc:
        raise R3Error("immutable artifact read failed") from exc
    after = _regular_file_state(path)
    if before != after:
        raise R3Error("immutable artifact changed during hash")
    return {
        "path": str(path),
        "file_state": after,
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _clone_tree_state_digest(
    root: Path,
    *,
    ignore_mutable_workset: bool = False,
    store: Any | None = None,
    require_checkpoint_file_state: bool = False,
) -> dict[str, Any]:
    """Capture bounded clone identity without replaying append-only bodies.

    Small files are content-hashed.  R2 ledger/catalog/FTS checkpoints and Raw
    state/watermark bind production-sized files without reading their bodies.
    Every file is lstat'ed again after observation and the returned
    representation makes the distinction explicit in the sealed artifact.
    """

    if _has_symlink_component(root) or not root.is_dir():
        raise R3Error("clone tree root is unsafe")
    digest = hashlib.sha256()
    file_count = 0
    total_bytes = 0
    hashed_bytes = 0
    omitted_bytes = 0
    omitted_file_count = 0
    raw_checkpoint: dict[str, Any] | None = None
    logical_digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        try:
            state = path.lstat()
        except OSError as exc:
            raise R3Error("clone tree changed during state capture") from exc
        if stat.S_ISLNK(state.st_mode):
            raise R3Error("protected clone tree contains a symlink")
        if not stat.S_ISREG(state.st_mode):
            continue
        relative = path.relative_to(root).as_posix()
        if ignore_mutable_workset and (
            relative == OX_WORKSET_RELATIVE.as_posix()
            or relative.startswith(f"{OX_WORKSET_RELATIVE.as_posix()}-")
            or any(
                component.startswith(".r3-harness-")
                for component in Path(relative).parts
            )
        ):
            continue
        if file_count >= CLONE_TREE_FILE_LIMIT:
            raise R3Error("clone tree exceeds bounded file inventory")
        size_bytes = int(state.st_size)
        parts = Path(relative).parts
        evidence: dict[str, Any] | None = None
        representation = "content-sha256"
        content_sha256: str | None = None
        # R2's append-only ledgers are checkpointed even when they are below
        # the per-file body limit.  This keeps repeated before/after scans
        # bounded and binds their head/count/file-state exactly.
        if store is not None and parts and parts[0] == "raw":
            if raw_checkpoint is None:
                raw_checkpoint = _bounded_raw_identity(root, "clone.raw_tree")
            evidence = raw_checkpoint
            representation = CLONE_TREE_RAW_REPRESENTATION
        elif store is not None and path.name in R2.R0.LEDGERS:
            evidence = _ledger_checkpoint_evidence(
                store,
                path,
                require_checkpoint_file_state=require_checkpoint_file_state,
            )
            representation = CLONE_TREE_LEDGER_REPRESENTATION
        elif store is not None and path.name == "historical-catalog.sqlite":
            evidence = _catalog_checkpoint_evidence(
                root,
                path,
                store,
                require_checkpoint_file_state=require_checkpoint_file_state,
            )
            representation = CLONE_TREE_CATALOG_REPRESENTATION
        elif size_bytes > CLONE_TREE_FILE_BYTES_LIMIT:
            raise R3Error("clone tree file exceeds bounded hash limit")
        else:
            if hashed_bytes + size_bytes > CLONE_TREE_HASH_BYTES_LIMIT:
                raise R3Error("clone tree body hash budget exceeded")
            content = hashlib.sha256()
            try:
                with path.open("rb") as handle:
                    while chunk := handle.read(1024 * 1024):
                        content.update(chunk)
                content_sha256 = content.hexdigest()
            except OSError as exc:
                raise R3Error("clone tree file changed during hashing") from exc
            hashed_bytes += size_bytes
        try:
            after = path.lstat()
        except OSError as exc:
            raise R3Error("clone tree file changed during state capture") from exc
        if (
            state.st_dev != after.st_dev
            or state.st_ino != after.st_ino
            or state.st_size != after.st_size
            or state.st_mtime_ns != after.st_mtime_ns
        ):
            raise R3Error("clone tree file changed during hashing")
        row = {
            "mode": int(state.st_mode & 0o7777),
            "path": relative,
            "size_bytes": size_bytes,
            "representation": representation,
        }
        if content_sha256 is not None:
            row["sha256"] = content_sha256
        if evidence is not None:
            row["checkpoint"] = evidence
        digest.update(json.dumps(row, sort_keys=True, separators=(",", ":")).encode())
        digest.update(b"\n")
        logical_row = dict(row)
        if evidence is not None:
            logical_row["checkpoint"] = {
                key: value
                for key, value in evidence.items()
                if key not in {"file_state", "checkpoint_file_state"}
            }
        logical_digest.update(
            json.dumps(
                logical_row, sort_keys=True, separators=(",", ":")
            ).encode()
        )
        logical_digest.update(b"\n")
        file_count += 1
        total_bytes += size_bytes
        if evidence is not None:
            omitted_bytes += size_bytes
            omitted_file_count += 1
    return {
        "file_count": file_count,
        "bytes": total_bytes,
        "state_sha256": digest.hexdigest(),
        "logical_sha256": logical_digest.hexdigest(),
        "representation": CLONE_TREE_DIGEST_REPRESENTATION,
        "hashed_bytes": hashed_bytes,
        "omitted_bytes": omitted_bytes,
        "omitted_file_count": omitted_file_count,
        "hash_byte_limit": CLONE_TREE_HASH_BYTES_LIMIT,
        "file_byte_limit": CLONE_TREE_FILE_BYTES_LIMIT,
        "raw_checkpoint": raw_checkpoint,
    }


def _clone_static_identity(store: Any, clone: Path) -> dict[str, Any]:
    """Read the R2 path-neutral static identity from the supplied clone."""

    try:
        catalog = __import__(
            "chronovisor.recall.recall_distillation_catalog",
            fromlist=["HistoricalCatalog"],
        )
        raw_store = __import__("chronovisor.core.raw_store", fromlist=["RawStore"])
        runtime_dir = store.distillation_dir(clone)
        ledgers = {
            name: R2._bounded_chain(
                store,
                runtime_dir / name,
                require_checkpoint_file_state=False,
            )
            for name in R2.R0.LEDGERS
        }
        watermark = raw_store.committed_raw_watermark(clone / "raw")
        fts = R2.R0._fts(
            store,
            catalog,
            clone,
            watermark,
            require_checkpoint_file_state=False,
        )
        state = store.read_sealed(
            runtime_dir / store.STATE_FILE, schema=store.DISTILLATION_SCHEMA
        )
        compact_state = {
            key: state.get(key)
            for key in R2.R0.STATE_KEYS
            if isinstance(state.get(key), (str, int, bool)) or state.get(key) is None
        }
        pointers: dict[str, Any] = {}
        for kind, filename in store.POINTER_FILES.items():
            pointer_path = runtime_dir / filename
            if not pointer_path.exists():
                pointers[kind] = None
                continue
            pointer = store.read_sealed(pointer_path, schema=store.DISTILLATION_SCHEMA)
            policy_id = pointer.get("policy_id")
            policy = store.read_sealed(
                runtime_dir / "policies" / f"{policy_id}.json",
                schema=R2.R0.POLICY_SCHEMA,
            )
            if policy.get("artifact_id") != policy_id:
                raise R3Error("clone policy pointer identity mismatch")
            pointers[kind] = {
                "policy_id": policy_id,
                "pointer_seal_sha256": pointer.get("seal_sha256", ""),
                "policy_seal_sha256": policy.get("seal_sha256", ""),
            }
        identity = R2.R0._clone_identity(
            {
                "ledgers": ledgers,
                "raw_watermark": watermark,
                "fts": fts,
                "state": {"seal_sha256": state.get("seal_sha256", ""), "fields": compact_state},
                "pointers": pointers,
            }
        )
    except (ImportError, OSError, KeyError, TypeError, ValueError, sqlite3.Error) as exc:
        raise R3Error("clone static identity read failed") from exc
    return _manifest_static(identity, "clone.static")


def _clone_workset_inventory(
    path: Path, *, expected_rows: int | None = None, require_receipts: bool = False
) -> dict[str, Any]:
    """Read a bounded, payload-free inventory and digest of the clone DB."""

    before = _regular_file_state(path)
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
            tables = {
                str(row[0])
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            }
            required = {"work_items", "workset_state"}
            if not required.issubset(tables):
                raise R3Error("clone production ox workset schema is incomplete")
            receipt_table_present = "workset_receipts" in tables
            if require_receipts and not receipt_table_present:
                raise R3Error("clone production ox workset receipts are unavailable")
            work_item_columns = {
                str(row[1])
                for row in connection.execute("PRAGMA table_info(work_items)")
            }
            required_columns = set(_WORKSET_SECURITY_COLUMNS).difference(
                {"stage", "next_attempt_at"}
            )
            if not required_columns.issubset(work_item_columns):
                raise R3Error("clone production ox workset identity schema is incomplete")
            row = connection.execute(
                "SELECT COUNT(*), COUNT(DISTINCT work_id), "
                "COUNT(DISTINCT payload_digest) FROM work_items"
            ).fetchone()
            row_count, unique_work_ids, unique_payload_digests = (int(value) for value in row)
            if row_count > OX_WORKSET_ROW_LIMIT:
                raise R3Error("clone production ox workset exceeds bounded inventory")
            if expected_rows is not None and row_count != expected_rows:
                raise R3Error("clone production ox workset row count is not certified")
            states = {
                str(state): int(count)
                for state, count in connection.execute(
                    "SELECT state, COUNT(*) FROM work_items GROUP BY state ORDER BY state"
                )
            }
            state_rows = connection.execute(
                "SELECT key, value_json FROM workset_state ORDER BY key"
            ).fetchall()
            state_digest = hashlib.sha256(
                json.dumps(
                    [(str(key), str(value_json)) for key, value_json in state_rows],
                    separators=(",", ":"),
                ).encode()
            ).hexdigest()
            state_seal_digest = hashlib.sha256(
                b"chronovisor.r3.workset-state-seal.v1\n"
                + state_digest.encode()
            ).hexdigest()
            receipt_count = 0
            receipt_digest = hashlib.sha256()
            if receipt_table_present:
                receipt_rows = connection.execute(
                    "SELECT generation, previous_sha256, operation, receipt_sha256 "
                    "FROM workset_receipts ORDER BY generation"
                ).fetchall()
                receipt_count = len(receipt_rows)
                for generation, previous, operation, receipt in receipt_rows:
                    receipt_digest.update(
                        json.dumps(
                            {
                                "generation": int(generation),
                                "previous_sha256": str(previous),
                                "operation": str(operation),
                                "receipt_sha256": str(receipt),
                            },
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode()
                    )
                    receipt_digest.update(b"\n")
                if receipt_count > OX_WORKSET_ROW_LIMIT * 2:
                    raise R3Error("clone production ox workset receipts are unbounded")
            rows = connection.execute(
                "SELECT "
                + ", ".join(
                    f'"{column}"' if column in work_item_columns else f'NULL AS "{column}"'
                    for column in _WORKSET_SECURITY_COLUMNS
                )
                + " FROM work_items ORDER BY sequence LIMIT ?",
                (OX_WORKSET_ROW_LIMIT + 1,),
            ).fetchall()
            if len(rows) > OX_WORKSET_ROW_LIMIT:
                raise R3Error("clone production ox workset inventory is unbounded")
    except sqlite3.Error as exc:
        raise R3Error("clone production ox workset read failed") from exc
    digest = hashlib.sha256()
    for row in rows:
        identity = {
            column: value
            for column, value in zip(_WORKSET_SECURITY_COLUMNS, row, strict=True)
        }
        digest.update(
            json.dumps(
                identity,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        )
        digest.update(b"\n")
    after = _regular_file_state(path)
    if before != after:
        raise R3Error("clone production ox workset changed during inventory")
    return {
        "relative_path": OX_WORKSET_RELATIVE.as_posix(),
        "row_count": row_count,
        "unique_work_ids": unique_work_ids,
        "unique_item_digests": unique_payload_digests,
        "states": states,
        "schema": {
            "tables": sorted(tables),
            "receipt_table_present": receipt_table_present,
            "receipt_count": receipt_count,
            "retry_column_present": "next_attempt_at" in work_item_columns,
            "stage_column_present": "stage" in work_item_columns,
        },
        "inventory_sha256": digest.hexdigest(),
        "content_sha256": digest.hexdigest(),
        "state_sha256": state_digest,
        "state_seal_sha256": state_seal_digest,
        "receipt_chain_sha256": receipt_digest.hexdigest(),
        "file_state": after,
        "bounded": True,
        "production_path_used": False,
    }


def _workset_identity(inventory: Mapping[str, Any]) -> dict[str, Any]:
    """Return clone/workset content identity without inode or mtime metadata."""

    keys = (
        "relative_path",
        "row_count",
        "unique_work_ids",
        "unique_item_digests",
        "states",
        "schema",
        "inventory_sha256",
        "content_sha256",
        "state_sha256",
        "state_seal_sha256",
        "receipt_chain_sha256",
        "bounded",
        "production_path_used",
    )
    if any(key not in inventory for key in keys):
        raise R3Error("workset identity evidence is incomplete")
    return {key: inventory[key] for key in keys}


def _run_clone_workset_cycles(
    workset_module: Any, clone: Path, *, cycles: int = MIN_SAMPLES
) -> dict[str, Any]:
    """Run successful claim/commit cycles against the cloned production workset."""

    if cycles < MIN_SAMPLES:
        raise R3Error("clone workset certification requires 100 successful cycles")
    path = _clone_workset_path(clone)
    legacy = _clone_workset_inventory(
        path, expected_rows=OX_WORKSET_EXPECTED_ROWS, require_receipts=False
    )
    legacy_schema = legacy["schema"]
    if not isinstance(legacy_schema, Mapping):
        raise R3Error("clone production ox workset legacy schema evidence is invalid")
    workset = workset_module.DistillationWorkset(path)
    migrated = _clone_workset_inventory(
        path, expected_rows=OX_WORKSET_EXPECTED_ROWS, require_receipts=True
    )
    migrated_schema = migrated["schema"]
    if (
        not isinstance(migrated_schema, Mapping)
        or migrated_schema.get("receipt_table_present") is not True
        or migrated_schema.get("retry_column_present") is not True
        or migrated_schema.get("stage_column_present") is not True
    ):
        raise R3Error("clone production ox workset migration is incomplete")
    migration_audit = workset.audit_transition_receipts()
    if migration_audit.get("status") not in {"verified", "legacy-unverified"}:
        raise R3Error("clone production ox workset migration audit is invalid")
    initial_status = workset.status()
    state_names = ("ready", "leased", "completed", "quarantined")
    legacy_state_counts = {
        state: int(legacy["states"].get(state, 0)) for state in state_names
    }
    migration_state_counts = {
        state: int(initial_status.get(state, 0)) for state in state_names
    }
    if legacy_state_counts != OX_WORKSET_EXPECTED_STATES:
        raise R3Error("clone production ox workset state counts are not certified")
    if migration_state_counts != legacy_state_counts:
        raise R3Error("clone workset migration changed state counts")
    # Legacy production clones have durable state but no receipt progress.  Seed
    # one current-schema progress boundary after migration; the legacy prefix is
    # retained as evidence and is never admitted to the successful-cycle
    # denominator below.
    progress_before_cycles = workset.progress()
    progress_bootstrap = progress_before_cycles is None
    if progress_bootstrap:
        watermark = workset.watermark()
        if watermark is None:
            watermark = {"source": "r3-clone-cycle"}
        workset.advance([], watermark, progress=_progress(0))
        progress_before_cycles = workset.progress()
    if not isinstance(progress_before_cycles, Mapping):
        raise R3Error("clone workset durable progress is unavailable")
    progress_cursor = progress_before_cycles.get("cursor")
    if (
        not isinstance(progress_cursor, Mapping)
        or set(progress_cursor) != {"completed"}
        or isinstance(progress_cursor.get("completed"), bool)
        or not isinstance(progress_cursor.get("completed"), int)
        or progress_cursor["completed"] < 0
    ):
        raise R3Error("clone workset durable progress cursor is not current-schema")
    progress_base_cursor = int(progress_cursor["completed"])
    receipts_before_cycles = _receipt_rows(path)
    initial_status = workset.status()
    initial_generation = int(initial_status["last_durable_receipt"]["generation"])
    if initial_generation != len(receipts_before_cycles):
        raise R3Error("clone workset receipt generation baseline is inconsistent")
    timings: list[int] = []
    for _index in range(cycles):
        started = time.perf_counter_ns()
        claims = workset.claim(None, 1, "r3-clone-cycle", 60.0)
        elapsed = time.perf_counter_ns() - started
        if len(claims) != 1:
            raise R3Error("clone production ox workset did not admit a successful cycle")
        timings.append(elapsed)
        totals = workset.commit(
            claims,
            [_completed(claims[0])],
            progress=_progress(progress_base_cursor + _index + 1),
        )
        if totals.get("completed") != 1:
            raise R3Error("clone production ox workset commit was not completed")
    empty_started = time.perf_counter_ns()
    if workset.claim("r3-empty-probe", 1, "r3-clone-cycle", 60.0):
        raise R3Error("clone production ox workset empty probe selected work")
    empty_probe_ns = time.perf_counter_ns() - empty_started
    claim_p95 = _p95(timings)
    if claim_p95 > CLAIM_P95_LIMIT_NS:
        raise R3Error("clone claim p95 exceeded 500ms")
    final_status = workset.status()
    expected_final_states = dict(OX_WORKSET_EXPECTED_STATES)
    expected_final_states["ready"] -= cycles
    expected_final_states["completed"] += cycles
    final_state_counts = {
        state: int(final_status.get(state, 0)) for state in expected_final_states
    }
    if final_state_counts != expected_final_states or final_state_counts["leased"] != 0:
        raise R3Error("clone production ox workset final state counts are invalid")
    final_generation = int(final_status["last_durable_receipt"]["generation"])
    if final_generation - initial_generation < cycles * 2:
        raise R3Error("clone workset did not receiptize every cycle")
    receipt_rows = _receipt_rows(path)
    cycle_receipts = receipt_rows[len(receipts_before_cycles) :]
    expected_progress_receipts = cycles * 2
    if len(cycle_receipts) != expected_progress_receipts:
        raise R3Error("clone workset cycle receipt denominator is incomplete")

    def progress_equal(left: object, right: object) -> bool:
        return json.dumps(left, sort_keys=True, separators=(",", ":")) == json.dumps(
            right, sort_keys=True, separators=(",", ":")
        )

    def valid_progress(value: object) -> bool:
        return isinstance(value, Mapping) and set(value) == {
            "cursor",
            "ledger_heads",
            "provenance",
            "progress_kind",
        }

    prior_progress: object = progress_before_cycles
    verified_progress_receipts = 0
    for index, row in enumerate(cycle_receipts):
        payload = row.get("payload")
        if (
            row.get("generation") != initial_generation + index + 1
            or row.get("operation") not in {"claim", "commit"}
            or not isinstance(payload, Mapping)
            or payload.get("version") != 2
            or not isinstance(payload.get("before"), Mapping)
            or not isinstance(payload.get("after"), Mapping)
        ):
            raise R3Error("clone cycle receipt is not current-schema progress")
        before = payload["before"].get("progress")
        after = payload["after"].get("progress")
        if (
            not valid_progress(before)
            or not valid_progress(after)
            or not progress_equal(before, prior_progress)
        ):
            raise R3Error("clone cycle progress before/after continuity failed")
        if row["operation"] == "claim" and not progress_equal(before, after):
            raise R3Error("clone claim receipt changed durable progress")
        if row["operation"] == "commit":
            expected_cursor = progress_base_cursor + (index // 2) + 1
            cursor = after.get("cursor")
            if (
                not isinstance(cursor, Mapping)
                or cursor.get("completed") != expected_cursor
            ):
                raise R3Error("clone commit receipt progress cursor is invalid")
            prior_progress = after
        else:
            prior_progress = after
        verified_progress_receipts += 1
    final_progress = workset.progress()
    if not valid_progress(final_progress) or not progress_equal(
        final_progress, prior_progress
    ):
        raise R3Error("clone workset final durable progress parity failed")
    progress_coverage = 100.0 * verified_progress_receipts / expected_progress_receipts
    # The denominator starts after migration/bootstrap, so any legacy or
    # pre-existing receipts are explicitly outside successful-cycle coverage.
    legacy_unverified_excluded = True
    audit = workset.audit_transition_receipts()
    audit_status = audit.get("status")
    durable_final = final_status.get("last_durable_receipt")
    if (
        not isinstance(durable_final, Mapping)
        or audit.get("generation") != final_generation
        or audit.get("head_sha256") != durable_final.get("head_sha256")
        or audit.get("progress") != final_progress
        or final_status.get("last_durable_progress") != final_progress
    ):
        raise R3Error("clone workset last durable receipt/progress parity failed")
    cycle_receipt_chain_verified = (
        verified_progress_receipts == expected_progress_receipts
        and progress_coverage >= RECEIPT_COVERAGE_LIMIT
    )
    if not cycle_receipt_chain_verified:
        raise R3Error("clone production ox workset progress receipts are not verified")
    after = _clone_workset_inventory(
        path, expected_rows=OX_WORKSET_EXPECTED_ROWS, require_receipts=True
    )
    if after["unique_work_ids"] != after["row_count"]:
        raise R3Error("clone production ox workset contains duplicate work ids")
    if legacy["row_count"] != after["row_count"]:
        raise R3Error("clone production ox workset inventory changed size")
    duplicates = _duplicate_count(path)
    if duplicates != 0:
        raise R3Error("clone production ox workset has duplicate receipts")
    return {
        "relative_path": OX_WORKSET_RELATIVE.as_posix(),
        "row_count": after["row_count"],
        "legacy_inventory": legacy,
        "inventory_before": migrated,
        "inventory_after": after,
        "samples": len(timings),
        "successful_cycles": len(timings),
        "claim_samples": len(timings),
        "observation_calls": len(timings),
        "claim_p95_ns": claim_p95,
        "claim_threshold_ns": CLAIM_P95_LIMIT_NS,
        "empty_probe": {
            "kind": "r3-empty-probe",
            "scope": "kind-only",
            "elapsed_ns": empty_probe_ns,
            "excluded_from_p95": True,
        },
        "legacy_status": {
            "states": legacy_state_counts,
            "row_count": legacy["row_count"],
            "receipt_count": legacy_schema["receipt_count"],
        },
        "migration": {
            "schema_before": legacy_schema,
            "schema_after": migrated_schema,
            "status_before_cycles": migration_state_counts,
            "status_unchanged": migration_state_counts == legacy_state_counts,
            "audit_status_before_cycles": migration_audit["status"],
            "receipt_chain_verified": migration_audit["status"] == "verified",
        },
        "final_status": final_state_counts,
        "receipt_generation_before": initial_generation,
        "receipt_generation_after": final_generation,
        "receipt_delta": final_generation - initial_generation,
        "audit_status": audit_status,
        "receipt_chain_verified": cycle_receipt_chain_verified,
        "cycle_audit_status": "verified" if cycle_receipt_chain_verified else "failed",
        "legacy_audit_status": audit_status,
        "legacy_unverified_excluded": legacy_unverified_excluded,
        "progress_receipt_count": verified_progress_receipts,
        "expected_progress_receipt_count": expected_progress_receipts,
        "progress_coverage_pct": progress_coverage,
        "progress_coverage": {
            "denominator": expected_progress_receipts,
            "receipts": verified_progress_receipts,
            "percent": progress_coverage,
            "schema_version": 2,
            "legacy_unverified_excluded": legacy_unverified_excluded,
        },
        "progress_before": dict(progress_before_cycles),
        "progress_after": dict(final_progress),
        "progress_receipt_generations": {
            "before": initial_generation,
            "after": final_generation,
            "delta": final_generation - initial_generation,
        },
        "preexisting_receipt_count": len(receipts_before_cycles),
        "progress_bootstrap": progress_bootstrap,
        "duplicates": duplicates,
        "production_path_used": False,
    }


def _p95(values: list[int]) -> int:
    if not values:
        raise R3Error("p95 sample is empty")
    ordered = sorted(values)
    return ordered[max(0, (len(ordered) * 95 + 99) // 100 - 1)]


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def _item(work_id: str, kind: str, *, priority: int = 0) -> dict[str, Any]:
    digest = _digest({"work_id": work_id, "kind": kind})
    return {
        "work_id": work_id,
        "kind": kind,
        "payload_ref": f"candidate-ledger:r3-{work_id}",
        "payload_digest": digest,
        "priority": priority,
        "temporal_split": {"partition": "train", "cutoff": "2026-08-24"},
        "provenance": {"cohort": "r3-harness-v1", "route": "offline"},
    }


def _progress(cursor: int) -> dict[str, Any]:
    digest = f"{cursor + 1:064x}"[-64:]
    return {
        "cursor": {"completed": cursor},
        "ledger_heads": {"workset": digest},
        "provenance": {"cohort": "r3-harness-v1", "revision": "offline"},
        "progress_kind": "r3-harness-v1",
    }


def _completed(claim: Any) -> dict[str, str]:
    return {
        "status": "completed",
        "completion_ref": f"label-ledger:r3-{claim.work_id}",
        "completion_digest": claim.payload_digest,
    }


def _receipt_rows(path: Path) -> list[dict[str, Any]]:
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
            rows = connection.execute(
                "SELECT generation, previous_sha256, operation, payload_json, "
                "receipt_sha256 FROM workset_receipts ORDER BY generation"
            ).fetchall()
    except sqlite3.Error as exc:
        raise R3Error("workset receipt read failed") from exc
    result: list[dict[str, Any]] = []
    for generation, previous, operation, payload_json, receipt in rows:
        try:
            payload = json.loads(payload_json)
        except (TypeError, ValueError) as exc:
            raise R3Error("workset receipt JSON is invalid") from exc
        result.append(
            {
                "generation": int(generation),
                "previous_sha256": str(previous),
                "operation": str(operation),
                "payload": payload,
                "receipt_sha256": str(receipt),
            }
        )
    return result


def _duplicate_count(path: Path) -> int:
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
            rows = connection.execute(
                "SELECT COALESCE(SUM(n - 1), 0) FROM "
                "(SELECT COUNT(*) AS n FROM work_items GROUP BY work_id)"
            ).fetchone()
            receipt_rows = connection.execute(
                "SELECT COALESCE(SUM(n - 1), 0) FROM "
                "(SELECT COUNT(*) AS n FROM workset_receipts GROUP BY receipt_sha256)"
            ).fetchone()
    except sqlite3.Error as exc:
        raise R3Error("duplicate audit failed") from exc
    return int(rows[0] or 0) + int(receipt_rows[0] or 0)


_TEACHER_CHILD = r'''
import json, sys
from chronovisor.recall.recall_distillation_dispatcher import dispatch_claimed_work
from chronovisor.recall.recall_distillation_workset import DistillationWorkset

path = sys.argv[1]
count = int(sys.argv[2])
kind = sys.argv[3] if len(sys.argv) > 3 else "local-teacher:handoff"
workset = DistillationWorkset(path)
claims = workset.claim(kind, count, "local-fake-teacher", 60.0)
if len(claims) != count:
    raise SystemExit(3)
dispatch_results = dispatch_claimed_work(
    claims,
    lambda _claim: {"accepted": True, "teacher": "local-fake-v1"},
    max_inflight=10,
    max_retries=0,
    min_valid_results_per_cap=1,
    initial_cap=1,
    valid_result_count=lambda _value: 1,
)
if len(dispatch_results) != count or any(
    result.status != "ok" for result in dispatch_results
):
    raise SystemExit(4)
outcomes = [
    {
        "status": "completed",
        "completion_ref": f"label-ledger:r3-handoff-{claim.work_id}",
        "completion_digest": claim.payload_digest,
    }
    for claim in claims
]
totals = workset.commit(claims, outcomes)
audit = workset.audit_transition_receipts()
if totals.get("completed") != count or audit.get("status") not in {
    "verified", "legacy-unverified"
}:
    raise SystemExit(5)
print(json.dumps({
    "teacher": "local-fake-v1",
    "dispatcher": "single-teacher-v1",
    "lease_observed": True,
    "claimed": len(claims),
    "completed": totals["completed"],
    "kind": kind,
    "owner": "local-fake-teacher",
    "audit_status": audit["status"],
    "legacy_unverified_excluded": audit["status"] == "legacy-unverified",
    "receipt_generation": audit["generation"],
}, separators=(",", ":")), flush=True)
'''


_CLONE_TEACHER_CHILD = r'''
import json, sys
from chronovisor.recall.recall_distillation_dispatcher import dispatch_claimed_work
from chronovisor.recall.recall_distillation_workset import DistillationWorkset

path = sys.argv[1]
count = int(sys.argv[2])
kind = sys.argv[3]
workset = DistillationWorkset(path)
for _index in range(count):
    claims = workset.claim(kind, 1, "local-fake-teacher", 60.0)
    if len(claims) != 1:
        raise SystemExit(3)
    dispatch_results = dispatch_claimed_work(
        claims,
        lambda _claim: {"accepted": True, "teacher": "local-fake-v1"},
        max_inflight=1,
        max_retries=0,
        min_valid_results_per_cap=1,
        initial_cap=1,
        valid_result_count=lambda _value: 1,
    )
    if len(dispatch_results) != 1 or dispatch_results[0].status != "ok":
        raise SystemExit(4)
    claim = claims[0]
    totals = workset.commit(
        claims,
        [{
            "status": "completed",
            "completion_ref": f"label-ledger:r3-handoff-{claim.work_id}",
            "completion_digest": claim.payload_digest,
        }],
    )
    if totals.get("completed") != 1:
        raise SystemExit(5)
audit = workset.audit_transition_receipts()
if audit.get("status") not in {"verified", "legacy-unverified"}:
    raise SystemExit(6)
print(json.dumps({
    "teacher": "local-fake-v1",
    "dispatcher": "single-teacher-v1",
    "lease_observed": True,
    "claimed": count,
    "completed": count,
    "kind": kind,
    "owner": "local-fake-teacher",
    "audit_status": audit["status"],
    "legacy_unverified_excluded": audit["status"] == "legacy-unverified",
}, separators=(",", ":")), flush=True)
'''


_SIGTERM_CHILD = r'''
import dataclasses, json, sys, time
from chronovisor.recall.recall_distillation_workset import DistillationWorkset
path = sys.argv[1]
kind = sys.argv[2] if len(sys.argv) > 2 else "r3-sigterm"
workset = DistillationWorkset(path)
claims = workset.claim(kind, 1, "sigterm-child", 0.5)
if len(claims) != 1:
    raise SystemExit(3)
print(json.dumps({"ready": True, "claim": dataclasses.asdict(claims[0])}, separators=(",", ":")), flush=True)
time.sleep(30)
'''


def _child_env(source_root: Path) -> dict[str, str]:
    environment = os.environ.copy()
    source_path = str(source_root / "src")
    environment["PYTHONPATH"] = os.pathsep.join(
        value for value in (source_path, environment.get("PYTHONPATH", "")) if value
    )
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return environment


def _teacher_handoff(
    workset_module: Any, source_root: Path, path: Path, sample_count: int
) -> dict[str, Any]:
    if sample_count < UNIT_MIN_SAMPLES:
        raise R3Error("teacher handoff sample count is below the unit minimum")
    if _has_symlink_component(path):
        raise R3Error("teacher handoff path contains a symlink")
    path.parent.mkdir(parents=True, exist_ok=True)
    workset = workset_module.DistillationWorkset(path)
    items = [
        _item(f"r3-handoff-{index}", "local-teacher:handoff", priority=0)
        for index in range(sample_count)
    ]
    workset.advance(items, {"source": "handoff"}, progress=_progress(0))
    started = time.perf_counter_ns()
    try:
        result = subprocess.run(
            [sys.executable, "-c", _TEACHER_CHILD, str(path), str(sample_count)],
            text=True,
            capture_output=True,
            check=True,
            timeout=10.0,
            env=_child_env(source_root),
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise R3Error("teacher handoff failed") from exc
    elapsed = time.perf_counter_ns() - started
    try:
        response = json.loads(result.stdout)
    except (TypeError, ValueError) as exc:
        raise R3Error("teacher handoff response is invalid") from exc
    expected = {
        "teacher": "local-fake-v1",
        "dispatcher": "single-teacher-v1",
        "lease_observed": True,
        "claimed": sample_count,
        "completed": sample_count,
        "audit_status": "verified",
    }
    if not isinstance(response, Mapping) or any(
        response.get(key) != value for key, value in expected.items()
    ):
        raise R3Error("teacher handoff response coverage is invalid")
    reopened = workset_module.DistillationWorkset(path)
    status = reopened.status("local-teacher:handoff")
    if status.get("completed") != sample_count or status.get("leased") != 0:
        raise R3Error("teacher handoff did not complete the claimed workset")
    audit = reopened.audit_transition_receipts()
    receipts = _receipt_rows(path)
    if audit.get("status") != "verified" or len(receipts) != 3:
        raise R3Error("teacher handoff receipt chain is incomplete")
    if Counter(row["operation"] for row in receipts) != Counter(
        {"advance": 1, "claim": 1, "commit": 1}
    ):
        raise R3Error("teacher handoff seam did not receiptize claim and commit")
    duplicates = _duplicate_count(path)
    if duplicates != 0:
        raise R3Error("teacher handoff produced duplicate rows")
    return {
        "wall_time_ns": elapsed,
        "accepted": sample_count,
        "claimed": response["claimed"],
        "completed": response["completed"],
        "teacher": response["teacher"],
        "dispatcher": response["dispatcher"],
        "lease_observed": True,
        "audit_status": audit["status"],
        "receiptized": True,
        "receipt_generation": audit["generation"],
        "duplicates": duplicates,
        "process_returncode": result.returncode,
    }


def _clone_teacher_handoff(
    workset_module: Any,
    source_root: Path,
    clone: Path,
    sample_count: int = MIN_SAMPLES,
) -> dict[str, Any]:
    """Exercise the real cloned Workset/dispatcher seam with a local teacher."""

    if sample_count < 1:
        raise R3Error("clone teacher handoff requires one successful claim")
    path = _clone_workset_path(clone)
    before_inventory = _clone_workset_inventory(
        path, expected_rows=OX_WORKSET_EXPECTED_ROWS, require_receipts=True
    )
    before_workset = workset_module.DistillationWorkset(path)
    before_progress = before_workset.progress()
    before_receipts = _receipt_rows(path)
    before_status = before_workset.status()
    before_generation = int(before_status["last_durable_receipt"]["generation"])
    if before_generation != len(before_receipts):
        raise R3Error("clone teacher handoff receipt baseline is inconsistent")
    # A cycle/child boundary must not inherit an APFS-copied WAL/SHM frame.
    # Checkpoint only the owned clone before handing the DB to the child; the
    # production Workset is never opened for this normalization.
    normalization_before_child = _normalize_clone_workset(
        path,
        expected_identity=_workset_identity(before_inventory),
        allow_wal=True,
    )
    started = time.perf_counter_ns()
    try:
        process = subprocess.run(
            [
                sys.executable,
                "-c",
                _CLONE_TEACHER_CHILD,
                str(path),
                str(sample_count),
                "ox",
            ],
            text=True,
            capture_output=True,
            check=True,
            timeout=10.0,
            env=_child_env(source_root),
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise R3Error("clone teacher handoff failed") from exc
    elapsed = time.perf_counter_ns() - started
    try:
        response = json.loads(process.stdout)
    except (TypeError, ValueError) as exc:
        raise R3Error("clone teacher handoff response is invalid") from exc
    if not isinstance(response, Mapping) or any(
        response.get(key) != value
        for key, value in {
            "teacher": "local-fake-v1",
            "dispatcher": "single-teacher-v1",
            "kind": "ox",
            "owner": "local-fake-teacher",
            "claimed": sample_count,
            "completed": sample_count,
        }.items()
    ) or response.get("audit_status") not in {"verified", "legacy-unverified"} or (
        response.get("audit_status") == "legacy-unverified"
        and response.get("legacy_unverified_excluded") is not True
    ):
        raise R3Error("clone teacher handoff response coverage is invalid")
    after_child_pre_inventory = _clone_workset_inventory(
        path, expected_rows=OX_WORKSET_EXPECTED_ROWS, require_receipts=True
    )
    normalization_after_child = _normalize_clone_workset(
        path,
        expected_identity=_workset_identity(after_child_pre_inventory),
        allow_wal=True,
    )
    reopened = workset_module.DistillationWorkset(path)
    status = reopened.status()
    if status.get("leased") != 0:
        raise R3Error("clone teacher handoff left an active lease")
    after_receipts = _receipt_rows(path)
    suffix = after_receipts[len(before_receipts) :]
    expected_receipts = sample_count * 2
    if len(suffix) != expected_receipts or Counter(
        row["operation"] for row in suffix
    ) != Counter({"claim": sample_count, "commit": sample_count}):
        raise R3Error("clone teacher handoff receipt seam is incomplete")
    prior_progress: object = before_progress
    verified_progress = 0
    for index, row in enumerate(suffix):
        payload = row.get("payload")
        if (
            row.get("generation") != before_generation + index + 1
            or not isinstance(payload, Mapping)
            or payload.get("version") != 2
            or not isinstance(payload.get("before"), Mapping)
            or not isinstance(payload.get("after"), Mapping)
            or payload["before"].get("progress") != prior_progress
        ):
            raise R3Error("clone teacher handoff progress receipt is invalid")
        prior_progress = payload["after"].get("progress")
        verified_progress += 1
    if reopened.progress() != prior_progress:
        raise R3Error("clone teacher handoff progress parity failed")
    audit = reopened.audit_transition_receipts()
    duplicates = _duplicate_count(path)
    if audit.get("status") not in {"verified", "legacy-unverified"} or duplicates != 0:
        raise R3Error("clone teacher handoff audit or duplicate check failed")
    after_inventory = _clone_workset_inventory(
        path, expected_rows=OX_WORKSET_EXPECTED_ROWS, require_receipts=True
    )
    return {
        "actual_workset": True,
        "wall_time_ns": elapsed,
        "threshold_ns": TEACHER_HANDOFF_LIMIT_NS,
        "accepted": sample_count,
        "claimed": int(response["claimed"]),
        "completed": int(response["completed"]),
        "teacher": response["teacher"],
        "dispatcher": response["dispatcher"],
        "kind": response["kind"],
        "owner": response["owner"],
        "lease_observed": True,
        "leased_after": int(status["leased"]),
        "audit_status": audit["status"],
        "legacy_unverified_excluded": audit["status"] == "legacy-unverified",
        "new_suffix_verified": True,
        "suffix_generation_start": before_generation + 1,
        "suffix_generation_end": int(status["last_durable_receipt"]["generation"]),
        "suffix_receipt_count": len(suffix),
        "suffix_receipt_chain_verified": True,
        "receiptized": True,
        "receipt_generation_before": before_generation,
        "receipt_generation_after": int(status["last_durable_receipt"]["generation"]),
        "progress_receipt_count": verified_progress,
        "expected_progress_receipt_count": expected_receipts,
        "progress_coverage_pct": 100.0 * verified_progress / expected_receipts,
        "progress_receipt_generations": {
            "start": before_generation + 1,
            "end": int(status["last_durable_receipt"]["generation"]),
            "count": verified_progress,
        },
        "progress_before": before_progress,
        "progress_after": reopened.progress(),
        "lock_holder": {
            "owner": response["owner"],
            "leased_before_commit": sample_count,
            "leased_after_commit": int(status["leased"]),
            "verified": True,
        },
        "reclaim": {
            "performed": False,
            "expired_leases": 0,
            "verified": True,
        },
        "duplicates": duplicates,
        "inventory_before": _workset_identity(before_inventory),
        "inventory_after": _workset_identity(after_inventory),
        "process_returncode": process.returncode,
        "normalization_before_child": normalization_before_child,
        "normalization_after_child": normalization_after_child,
    }


def _read_child_line(process: subprocess.Popen[str], timeout: float) -> str:
    if process.stdout is None:
        raise R3Error("SIGTERM child stdout is unavailable")
    selector = selectors.DefaultSelector()
    try:
        selector.register(process.stdout, selectors.EVENT_READ)
        events = selector.select(timeout)
        if not events:
            raise R3Error("SIGTERM child did not report readiness")
        line = process.stdout.readline()
    finally:
        selector.close()
    if not line:
        raise R3Error("SIGTERM child exited before readiness")
    return str(line)


def _sigterm_reopen(
    workset_module: Any,
    source_root: Path,
    path: Path,
    kind: str | None = None,
    expected_rows: int | None = None,
) -> dict[str, Any]:
    if kind is None:
        try:
            with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
                dedicated = connection.execute(
                    "SELECT 1 FROM work_items WHERE kind = 'r3-sigterm' "
                    "AND state = 'ready' LIMIT 1"
                ).fetchone()
        except sqlite3.Error as exc:
            raise R3Error("SIGTERM kind probe failed") from exc
        kind = "r3-sigterm" if dedicated is not None else "ox"
    workset = workset_module.DistillationWorkset(path)
    before_status = workset.status(kind)
    before_completed = int(before_status.get("completed", 0))
    before_receipts = _receipt_rows(path)
    before_inventory = _clone_workset_inventory(
        path, expected_rows=expected_rows, require_receipts=True
    )
    started = time.perf_counter_ns()
    process = subprocess.Popen(
        [sys.executable, "-c", _SIGTERM_CHILD, str(path), kind],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=_child_env(source_root),
    )
    try:
        line = _read_child_line(process, 10.0)
        try:
            ready = json.loads(line)
            claim_value = ready["claim"]
        except (TypeError, ValueError, KeyError) as exc:
            raise R3Error("SIGTERM child claim is invalid") from exc
        if ready.get("ready") is not True or not isinstance(claim_value, Mapping):
            raise R3Error("SIGTERM child did not seal a claim")
        process.terminate()
        try:
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5.0)
            raise R3Error("SIGTERM child did not terminate") from None
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5.0)
        if process.stdout is not None:
            process.stdout.close()
        if process.stderr is not None:
            process.stderr.close()
    if process.returncode != -signal.SIGTERM:
        raise R3Error("SIGTERM child returncode was not -SIGTERM")
    try:
        old_claim = workset_module.WorkClaim(**dict(claim_value))
    except (TypeError, ValueError) as exc:
        raise R3Error("SIGTERM child claim shape is invalid") from exc
    expiry = float(old_claim.lease_expires_at)
    while time.time() <= expiry:
        time.sleep(min(0.05, max(0.001, expiry - time.time())))
    reopened = workset_module.DistillationWorkset(path)
    claims = reopened.claim(kind, 1, "reopened-owner", 5.0)
    if len(claims) != 1 or claims[0].attempt != old_claim.attempt + 1:
        raise R3Error("expired SIGTERM lease was not reclaimed")
    reclaimed = claims[0]
    try:
        workset.commit([old_claim], [_completed(old_claim)])
    except workset_module.DistillationWorksetError:
        old_owner_rejected = True
    else:
        old_owner_rejected = False
    if not old_owner_rejected:
        raise R3Error("expired old owner was accepted")
    outcome = _completed(reclaimed)
    first = reopened.commit([reclaimed], [outcome])
    second = reopened.commit([reclaimed], [outcome])
    if first != second or first.get("completed") != 1:
        raise R3Error("idempotent commit failed after lease reclaim")
    status = reopened.status(kind)
    if (
        status.get("completed") != before_completed + 1
        or status.get("leased") != 0
    ):
        raise R3Error("reopened workset status is invalid")
    after_receipts = _receipt_rows(path)
    receipt_suffix = after_receipts[len(before_receipts) :]
    initial_generation = int(before_status["last_durable_receipt"]["generation"])
    if not receipt_suffix or any(
        row.get("generation") != initial_generation + index + 1
        for index, row in enumerate(receipt_suffix)
    ):
        raise R3Error("SIGTERM recovery receipt suffix is incomplete")
    after_inventory = _clone_workset_inventory(
        path, expected_rows=expected_rows, require_receipts=True
    )
    lock_holder = {
        "before_sigterm": old_claim.owner,
        "after_reclaim": reclaimed.owner,
        "lease_id_changed": reclaimed.lease_id != old_claim.lease_id,
    }
    release = {
        "attempted": True,
        "old_owner_rejected": old_owner_rejected,
        "lease_expired_before_reclaim": True,
    }
    reclaim = {
        "performed": True,
        "owner": reclaimed.owner,
        "attempt_before": old_claim.attempt,
        "attempt_after": reclaimed.attempt,
        "attempt_incremented": reclaimed.attempt == old_claim.attempt + 1,
    }
    return {
        "wall_time_ns": time.perf_counter_ns() - started,
        "child_returncode": process.returncode,
        "sigterm_process": {
            "signal": "SIGTERM",
            "returncode": process.returncode,
            "expected_returncode": -signal.SIGTERM,
            "asserted": True,
            "kind": kind,
            "lock_holder": lock_holder,
            "release": release,
            "reclaim": reclaim,
        },
        "old_owner_rejected": old_owner_rejected,
        "reclaimed_attempt": reclaimed.attempt,
        "completed_before": before_completed,
        "completed_after": int(status["completed"]),
        "actual_workset": True,
        "kind": kind,
        "leased_after": int(status["leased"]),
        "inventory_before": _workset_identity(before_inventory),
        "inventory_after": _workset_identity(after_inventory),
        "receipt_suffix": {
            "generation_start": receipt_suffix[0]["generation"],
            "generation_end": receipt_suffix[-1]["generation"],
            "count": len(receipt_suffix),
            "operations": dict(Counter(row["operation"] for row in receipt_suffix)),
        },
        "idempotent_commit": True,
        "lock_holder": lock_holder,
        "release": release,
        "reclaim": reclaim,
        "duplicates": _duplicate_count(path),
    }


def _run_workset(
    workset_module: Any, source_root: Path, root: Path, samples: int
) -> dict[str, Any]:
    if samples < UNIT_MIN_SAMPLES:
        raise R3Error(f"R3 sample count must be at least {UNIT_MIN_SAMPLES}")
    if _has_symlink_component(root):
        raise R3Error("workset root contains a symlink")
    root.mkdir(parents=True, exist_ok=True)
    stage_path = root / "stages.sqlite3"
    crash_path = root / "sigterm.sqlite3"
    clock_value = [0.0]
    stage_workset = workset_module.DistillationWorkset(
        stage_path, clock=lambda: clock_value[0]
    )
    phase_records: dict[str, dict[str, Any]] = {}
    # Make one teacher item old enough to prove cross-kind fairness over a
    # newer, higher-priority kind.
    teacher_phase_started = time.time_ns()
    stage_workset.advance(
        [_item("r3-old-teacher", "local-teacher:old", priority=0)],
        {"source": 0},
        progress=_progress(0),
    )
    teacher_phase_finished = time.time_ns()
    clock_value[0] = 61.0
    stage_items: list[dict[str, Any]] = []
    kind_by_stage = {
        "snapshot": "snapshot",
        "teacher": "local-teacher:a",
        "counterfactual": "counterfactual",
        "dataset": "dataset",
        "evaluation": "evaluation",
    }
    remaining, extra = divmod(samples - 1, len(kind_by_stage))
    for index, (stage, kind) in enumerate(kind_by_stage.items()):
        phase_started = time.time_ns()
        count = remaining + (1 if index < extra else 0)
        items = [_item(f"r3-{stage}-{index}", kind, priority=99) for index in range(count)]
        stage_items.extend(items)
        stage_workset.advance(items, {"source": stage}, progress=_progress(0))
        phase_finished = time.time_ns()
        phase_records[stage] = {
            "started_at_ns": phase_started,
            "finished_at_ns": phase_finished,
            "elapsed_ns": phase_finished - phase_started,
            "count": count,
        }
    phase_records["teacher"] = {
        "started_at_ns": teacher_phase_started,
        "finished_at_ns": teacher_phase_finished,
        "elapsed_ns": teacher_phase_finished - teacher_phase_started,
        "count": phase_records["teacher"]["count"] + 1
        if "teacher" in phase_records
        else 1,
    }

    claim_samples: list[int] = []
    retry_phase_started = time.time_ns()
    first_started = time.perf_counter_ns()
    first_claims = stage_workset.claim(None, 1, "fairness-worker", 60.0)
    if len(first_claims) != 1 or first_claims[0].work_id != "r3-old-teacher":
        raise R3Error("cross-kind fairness selected a newer kind")
    claim_samples.append(time.perf_counter_ns() - first_started)
    completed_count = 1
    stage_workset.commit(
        first_claims,
        [{"status": "retry", "error_class": "transport_error", "retry_after_seconds": 30}],
        progress=_progress(completed_count),
    )
    retry_status = stage_workset.status(include_timing=True)
    stages = retry_status.get("stages")
    if not isinstance(stages, Mapping) or set(stages) != set(SIX_STAGES):
        raise R3Error("six-stage status projection is incomplete")
    if retry_status.get("retry_wait") != 1 or stages["teacher"].get("retry_wait") != 1:
        raise R3Error("retry_wait observability is missing")
    retry_phase_finished = time.time_ns()
    phase_records["retry_wait"] = {
        "started_at_ns": retry_phase_started,
        "finished_at_ns": retry_phase_finished,
        "elapsed_ns": retry_phase_finished - retry_phase_started,
        "count": 1,
    }

    empty_probe_ns: int | None = None
    while True:
        started = time.perf_counter_ns()
        claims = stage_workset.claim(None, 1, "fairness-worker", 60.0)
        if not claims:
            empty_probe_ns = time.perf_counter_ns() - started
            break
        claim_samples.append(time.perf_counter_ns() - started)
        claim = claims[0]
        completed_count += 1
        stage_workset.commit(
            claims,
            [_completed(claim)],
            progress=_progress(completed_count),
        )
    expected_completed = len(stage_items)
    final_status = stage_workset.status(include_timing=True)
    if final_status.get("completed") != expected_completed:
        raise R3Error("stage workset did not complete every ready item")
    claim_p95 = _p95(claim_samples)
    if claim_p95 > CLAIM_P95_LIMIT_NS:
        raise R3Error("claim p95 exceeded 500ms")

    handoff = _teacher_handoff(
        workset_module, source_root, root / "teacher-handoff.sqlite3", samples
    )
    if int(handoff["wall_time_ns"]) > TEACHER_HANDOFF_LIMIT_NS:
        raise R3Error("teacher handoff exceeded 10 seconds")

    crash_workset = workset_module.DistillationWorkset(crash_path)
    crash_workset.advance([_item("r3-sigterm-item", "r3-sigterm")], {"source": 1})
    crash = _sigterm_reopen(workset_module, source_root, crash_path)
    if crash["duplicates"] != 0:
        raise R3Error("SIGTERM recovery produced duplicate rows")

    audit = stage_workset.audit_transition_receipts()
    if audit.get("status") != "verified":
        raise R3Error("stage receipt chain is not verified")
    durable = final_status.get("last_durable_receipt")
    progress = stage_workset.progress()
    if (
        not isinstance(durable, Mapping)
        or audit.get("generation") != durable.get("generation")
        or audit.get("head_sha256") != durable.get("head_sha256")
        or audit.get("progress") != progress
        or final_status.get("last_durable_progress") != progress
    ):
        raise R3Error("last durable receipt/progress parity failed")
    receipts = _receipt_rows(stage_path)
    if not receipts:
        raise R3Error("no durable receipts were emitted")
    claim_count = len(claim_samples)
    expected_receipts = 6 + claim_count + claim_count
    # Once progress is initialized, Workset seals it into claim/commit receipts
    # too; the formal denominator therefore covers every durable receipt.
    expected_progress_receipts = expected_receipts
    operation_counts = Counter(row["operation"] for row in receipts)
    if operation_counts != Counter(
        {"advance": 6, "claim": claim_count, "commit": claim_count}
    ):
        raise R3Error("durable receipt operation coverage is incomplete")
    receipt_coverage = 100.0 * len(receipts) / expected_receipts
    progress_receipts = sum(
        1
        for row in receipts
        if isinstance(row["payload"], Mapping) and row["payload"].get("version") == 2
    )
    progress_coverage = 100.0 * progress_receipts / expected_progress_receipts
    if receipt_coverage < RECEIPT_COVERAGE_LIMIT or progress_coverage < RECEIPT_COVERAGE_LIMIT:
        raise R3Error("durable receipt/progress coverage is below 99%")
    duplicates = _duplicate_count(stage_path) + _duplicate_count(crash_path)
    if duplicates != 0:
        raise R3Error("workset duplicate count is non-zero")
    return {
        "samples": samples,
        "admitted_cycles": samples,
        "stages": dict(retry_status["stages"]),
        "phases": phase_records,
        "retry_wait": {
            "count": retry_status["retry_wait"],
            "next_retry_in_seconds": retry_status["next_retry_in_seconds"],
            "oldest_retry_wait_age_seconds": retry_status["oldest_retry_wait_age_seconds"],
        },
        "final_status": {
            key: value
            for key, value in final_status.items()
            if key in {"ready", "leased", "completed", "quarantined", "retry_wait", "total"}
        },
        "fairness": {
            "older_kind": "local-teacher:old",
            "newer_high_priority_kind": "counterfactual",
            "selected_older_kind": True,
            "oldest_work_id_sha256": _digest("r3-old-teacher"),
            "passed": True,
        },
        "cross_kind_fairness": True,
        "claim": {
            "samples": claim_count,
            "observation_calls": claim_count + (1 if empty_probe_ns is not None else 0),
            "p95_ns": claim_p95,
            "threshold_ns": CLAIM_P95_LIMIT_NS,
            "successful_count": claim_count,
            "final_empty_excluded": empty_probe_ns is not None,
            "final_empty_observation_ns": empty_probe_ns,
        },
        "teacher_handoff": {**handoff, "threshold_ns": TEACHER_HANDOFF_LIMIT_NS},
        "sigterm_reopen": crash,
        "durability": {
            "samples": samples,
            "receipt_count": len(receipts),
            "expected_receipt_count": expected_receipts,
            "progress_receipt_count": progress_receipts,
            "expected_progress_receipt_count": expected_progress_receipts,
            "receipt_coverage_pct": receipt_coverage,
            "progress_coverage_pct": progress_coverage,
            "coverage": {
                "denominator": expected_receipts,
                "receipts": len(receipts),
                "percent": receipt_coverage,
            },
            "progress_coverage": {
                "denominator": expected_progress_receipts,
                "receipts": progress_receipts,
                "percent": progress_coverage,
            },
            "last_durable_receipt": dict(durable),
            "last_durable_progress": progress,
            "audit_status": audit["status"],
        },
        "duplicates": duplicates,
        "payload_free": True,
    }


def _assert_payload_free(payload: Mapping[str, Any]) -> None:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    forbidden = ("private raw", "authorization:", "bearer ", "api_key")
    if any(marker in encoded.lower() for marker in forbidden):
        raise R3Error("evidence contains a payload or credential marker")
    for key, value in payload.items():
        if "payload" in str(key).lower() and key not in {"payload_free"}:
            raise R3Error("evidence contains a payload field")
        if isinstance(value, Mapping):
            _assert_payload_free(value)
        elif isinstance(value, list):
            for child in value:
                if isinstance(child, Mapping):
                    _assert_payload_free(child)


def _assert_formal_acceptance(
    result: Mapping[str, Any],
    source: Mapping[str, Any],
    *,
    require_completion: bool = True,
) -> None:
    claim = result.get("claim")
    clone = result.get("clone_workset")
    sigterm = result.get("sigterm_process")
    successful_count = claim.get("successful_count") if isinstance(claim, Mapping) else None
    if (
        not isinstance(claim, Mapping)
        or isinstance(successful_count, bool)
        or not isinstance(successful_count, int)
        or successful_count < MIN_SAMPLES
        or claim.get("samples") != successful_count
        or claim.get("observation_calls") != successful_count
        or not isinstance(claim.get("p95_ns"), int)
        or claim["p95_ns"] > CLAIM_P95_LIMIT_NS
    ):
        raise R3Error("formal artifact lacks 100 successful claim cycles")
    synthetic = result.get("synthetic_claim")
    if not isinstance(synthetic, Mapping):
        raise R3Error("formal artifact lacks synthetic claim evidence")
    synthetic_count = synthetic.get("successful_count")
    if (
        not isinstance(synthetic_count, int)
        or isinstance(synthetic_count, bool)
        or synthetic_count < MIN_SAMPLES
        or synthetic.get("observation_calls") != synthetic_count + 1
        or synthetic.get("final_empty_excluded") is not True
    ):
        raise R3Error("synthetic claim empty-observation evidence is incomplete")
    if not isinstance(clone, Mapping):
        raise R3Error("formal artifact lacks clone workset evidence")
    legacy_status = clone.get("legacy_status")
    migration = clone.get("migration")
    progress_coverage_pct = clone.get("progress_coverage_pct")
    normalization = clone.get("normalization")
    if (
        clone.get("relative_path") != OX_WORKSET_RELATIVE.as_posix()
        or clone.get("row_count") != OX_WORKSET_EXPECTED_ROWS
        or clone.get("successful_cycles") != successful_count
        or clone.get("claim_samples") != successful_count
        or clone.get("observation_calls") != successful_count
        or clone.get("production_path_used") is not False
        or clone.get("duplicates") != 0
        or clone.get("receipt_chain_verified") is not True
        or clone.get("cycle_audit_status") != "verified"
        or clone.get("legacy_unverified_excluded") is not True
        or clone.get("progress_receipt_count")
        != clone.get("expected_progress_receipt_count")
        or not isinstance(progress_coverage_pct, (int, float))
        or progress_coverage_pct < RECEIPT_COVERAGE_LIMIT
        or not isinstance(clone.get("progress_before"), Mapping)
        or not isinstance(clone.get("progress_after"), Mapping)
        or not isinstance(clone.get("progress_receipt_generations"), Mapping)
        or clone["progress_receipt_generations"].get("delta", 0)
        < clone.get("expected_progress_receipt_count", 0)
        or not isinstance(clone.get("progress_coverage"), Mapping)
        or clone["progress_coverage"].get("denominator")
        != clone.get("expected_progress_receipt_count")
        or clone["progress_coverage"].get("receipts")
        != clone.get("progress_receipt_count")
        or clone["progress_coverage"].get("legacy_unverified_excluded") is not True
        or not isinstance(legacy_status, Mapping)
        or legacy_status.get("states") != OX_WORKSET_EXPECTED_STATES
        or not isinstance(migration, Mapping)
        or migration.get("status_unchanged") is not True
        or clone.get("final_status", {}).get("leased") != 0
        or not isinstance(normalization, Mapping)
        or normalization.get("clone_only") is not True
        or normalization.get("journal_mode") != "wal"
        or normalization.get("integrity") != "ok"
        or not isinstance(normalization.get("semantic_before"), Mapping)
        or not isinstance(normalization.get("semantic_after"), Mapping)
        or normalization.get("semantic_before")
        != _workset_identity(clone.get("legacy_inventory", {}))
        or normalization.get("semantic_after")
        != normalization.get("semantic_before")
        or not isinstance(normalization.get("sidecars_before"), Mapping)
        or not isinstance(normalization.get("sidecars_after"), Mapping)
        or not isinstance(normalization.get("file_state_before"), Mapping)
        or not isinstance(normalization.get("file_state_after"), Mapping)
        or normalization["file_state_before"].get("st_ino")
        != normalization["file_state_after"].get("st_ino")
    ):
        raise R3Error("clone workset certification is incomplete")
    if not isinstance(sigterm, Mapping) or (
        sigterm.get("returncode") != -signal.SIGTERM
        or sigterm.get("expected_returncode") != -signal.SIGTERM
        or sigterm.get("asserted") is not True
        or result.get("sigterm_reopen", {}).get("old_owner_rejected") is not True
        or result.get("sigterm_reopen", {}).get("idempotent_commit") is not True
    ):
        raise R3Error("SIGTERM child process evidence is incomplete")
    actual_sigterm = result.get("actual_clone_sigterm")
    if (
        not isinstance(actual_sigterm, Mapping)
        or actual_sigterm.get("actual_workset") is not True
        or actual_sigterm.get("kind") != "ox"
        or actual_sigterm.get("leased_after") != 0
        or actual_sigterm.get("duplicates") != 0
        or not isinstance(actual_sigterm.get("inventory_before"), Mapping)
        or not isinstance(actual_sigterm.get("inventory_after"), Mapping)
        or actual_sigterm["inventory_before"].get("row_count")
        != OX_WORKSET_EXPECTED_ROWS
        or actual_sigterm["inventory_after"].get("row_count")
        != OX_WORKSET_EXPECTED_ROWS
        or actual_sigterm.get("inventory_after_normalized")
        != actual_sigterm.get("inventory_after")
        or not isinstance(actual_sigterm.get("receipt_suffix"), Mapping)
        or actual_sigterm["receipt_suffix"].get("count", 0) < 1
        or not isinstance(actual_sigterm.get("normalization"), Mapping)
        or actual_sigterm["normalization"].get("clone_only") is not True
        or actual_sigterm["normalization"].get("integrity") != "ok"
    ):
        raise R3Error("actual clone SIGTERM recovery evidence is incomplete")
    sigterm_reopen = result.get("sigterm_reopen")
    lock_holder = sigterm_reopen.get("lock_holder") if isinstance(sigterm_reopen, Mapping) else None
    release = sigterm_reopen.get("release") if isinstance(sigterm_reopen, Mapping) else None
    reclaim = sigterm_reopen.get("reclaim") if isinstance(sigterm_reopen, Mapping) else None
    if (
        not isinstance(lock_holder, Mapping)
        or lock_holder.get("before_sigterm") != "sigterm-child"
        or lock_holder.get("after_reclaim") != "reopened-owner"
        or lock_holder.get("lease_id_changed") is not True
        or not isinstance(release, Mapping)
        or release.get("attempted") is not True
        or release.get("old_owner_rejected") is not True
        or release.get("lease_expired_before_reclaim") is not True
        or not isinstance(reclaim, Mapping)
        or reclaim.get("performed") is not True
        or reclaim.get("owner") != "reopened-owner"
        or reclaim.get("attempt_incremented") is not True
    ):
        raise R3Error("SIGTERM lease release/reclaim evidence is incomplete")
    handoff = result.get("teacher_handoff")
    if not isinstance(handoff, Mapping) or (
        handoff.get("actual_workset") is not True
        or handoff.get("accepted") != MIN_SAMPLES
        or handoff.get("claimed") != MIN_SAMPLES
        or handoff.get("completed") != MIN_SAMPLES
        or handoff.get("receiptized") is not True
        or handoff.get("duplicates") != 0
        or handoff.get("kind") != "ox"
        or handoff.get("owner") != "local-fake-teacher"
        or handoff.get("leased_after") != 0
        or handoff.get("audit_status") not in {"verified", "legacy-unverified"}
        or (
            handoff.get("audit_status") == "legacy-unverified"
            and handoff.get("legacy_unverified_excluded") is not True
        )
        or handoff.get("new_suffix_verified") is not True
        or handoff.get("suffix_receipt_chain_verified") is not True
        or handoff.get("suffix_receipt_count")
        != handoff.get("expected_progress_receipt_count")
        or handoff.get("expected_progress_receipt_count") != MIN_SAMPLES * 2
        or handoff.get("suffix_generation_end", 0)
        - handoff.get("suffix_generation_start", 0)
        + 1
        != handoff.get("suffix_receipt_count")
        or handoff.get("progress_receipt_count")
        != handoff.get("expected_progress_receipt_count")
        or handoff.get("progress_coverage_pct", 0) < RECEIPT_COVERAGE_LIMIT
        or not isinstance(handoff.get("lock_holder"), Mapping)
        or handoff["lock_holder"].get("verified") is not True
        or handoff["lock_holder"].get("owner") != "local-fake-teacher"
        or handoff["lock_holder"].get("leased_after_commit") != 0
        or not isinstance(handoff.get("reclaim"), Mapping)
        or handoff["reclaim"].get("verified") is not True
        or handoff.get("wall_time_ns", TEACHER_HANDOFF_LIMIT_NS + 1)
        > TEACHER_HANDOFF_LIMIT_NS
    ):
        raise R3Error("teacher handoff evidence is incomplete")
    phases = result.get("phases")
    if not isinstance(phases, Mapping) or set(phases) != set(SIX_STAGES):
        raise R3Error("six formal phases are incomplete")
    for name, phase in phases.items():
        if (
            not isinstance(phase, Mapping)
            or not isinstance(phase.get("started_at_ns"), int)
            or not isinstance(phase.get("finished_at_ns"), int)
            or phase["finished_at_ns"] < phase["started_at_ns"]
            or phase.get("elapsed_ns")
            != phase["finished_at_ns"] - phase["started_at_ns"]
            or not isinstance(phase.get("count"), int)
            or phase["count"] < (1 if name == "retry_wait" else 0)
        ):
            raise R3Error("phase timing/count evidence is incomplete")
    manifest = result.get("manifest")
    if not isinstance(manifest, Mapping):
        raise R3Error("R2 clone evidence is incomplete")
    required_manifest = ("clone_root_exact", "toctou_rechecked")
    if any(manifest.get(key) is not True for key in required_manifest):
        raise R3Error("R2 clone identity evidence is incomplete")
    if (
        manifest.get("raw_parity") is not None
        or manifest.get("static_parity") is not None
        or manifest.get("schema") != R3_CLONE_SCHEMA
        or manifest.get("scope") != "Workset DB/receipt chain/state/pointers/locks"
        or manifest.get("excluded_not_evaluated") != R3_EXCLUDED_NOT_EVALUATED
    ):
        raise R3Error("R3 clone scope evidence is incomplete")
    if manifest.get("filesystem") != "apfs" or manifest.get("filesystem_probe") != "r0":
        raise R3Error("R2 clone filesystem evidence is incomplete")
    if manifest.get("external") is True and any(
        manifest.get(key) is not True
        for key in ("seal_verified", "content_identity_verified")
    ):
        raise R3Error("R2 frozen manifest evidence is incomplete")
    workset_manifest = manifest.get("workset")
    if (
        not isinstance(workset_manifest, Mapping)
        or workset_manifest.get("relative_path") != OX_WORKSET_RELATIVE.as_posix()
        or not isinstance(workset_manifest.get("prestate"), Mapping)
        or not isinstance(workset_manifest.get("poststate"), Mapping)
        or workset_manifest.get("prestate_verified") is not True
        or workset_manifest.get("content_sha256_before")
        != workset_manifest["prestate"].get("content_sha256")
        or workset_manifest.get("state_sha256_before")
        != workset_manifest["prestate"].get("state_sha256")
        or workset_manifest.get("receipt_chain_sha256_before")
        != workset_manifest["prestate"].get("receipt_chain_sha256")
        or workset_manifest.get("content_sha256_after")
        != workset_manifest["poststate"].get("content_sha256")
        or workset_manifest.get("state_sha256_after")
        != workset_manifest["poststate"].get("state_sha256")
        or workset_manifest.get("state_seal_sha256_before")
        != workset_manifest["prestate"].get("state_seal_sha256")
        or workset_manifest.get("state_seal_sha256_after")
        != workset_manifest["poststate"].get("state_seal_sha256")
        or workset_manifest.get("receipt_chain_sha256_after")
        != workset_manifest["poststate"].get("receipt_chain_sha256")
    ):
        raise R3Error("R3 frozen manifest Workset prestate is incomplete")
    for field in (
        "content_sha256_before",
        "content_sha256_after",
        "state_sha256_before",
        "state_sha256_after",
        "state_seal_sha256_before",
        "state_seal_sha256_after",
        "receipt_chain_sha256_before",
        "receipt_chain_sha256_after",
    ):
        _hex_digest(workset_manifest.get(field), f"manifest.workset.{field}")
    input_clone = manifest.get("input_clone")
    if (
        not isinstance(input_clone, Mapping)
        or input_clone.get("external") != (manifest.get("external") is True)
        or input_clone.get("owned_scope_unchanged") is not True
        or input_clone.get("excluded_not_evaluated") != R3_EXCLUDED_NOT_EVALUATED
        or input_clone.get("root_identity_unchanged") is not True
    ):
        raise R3Error("external clone owned-scope evidence is incomplete")
    clone_tree = result.get("clone_tree")
    if (
        not isinstance(clone_tree, Mapping)
        or clone_tree.get("representation") != R3_WORKSET_SCOPE_REPRESENTATION
        or clone_tree.get("excluded_not_evaluated") != R3_EXCLUDED_NOT_EVALUATED
        or clone_tree.get("root_identity_unchanged") is not True
        or clone_tree.get("owned_scope_unchanged") is not True
        or not isinstance(clone_tree.get("before"), Mapping)
        or not isinstance(clone_tree.get("after"), Mapping)
        or not isinstance(clone_tree["before"].get("workset"), Mapping)
        or not isinstance(clone_tree["after"].get("workset"), Mapping)
    ):
        raise R3Error("clone tree evidence is incomplete")
    before_scope = clone_tree["before"].get("scope")
    after_scope = clone_tree["after"].get("scope")
    if (
        not isinstance(before_scope, Mapping)
        or not isinstance(after_scope, Mapping)
        or before_scope.get("state_pointers") != after_scope.get("state_pointers")
        or before_scope.get("locks") != after_scope.get("locks")
    ):
        raise R3Error("clone Workset state/pointer/lock parity is incomplete")
    formal_wall = result.get("formal_wall") or result.get("pre_publication_wall")
    if (
        not isinstance(formal_wall, Mapping)
        or not isinstance(formal_wall.get("started_at_ns"), int)
        or not isinstance(formal_wall.get("finished_at_ns"), int)
        or formal_wall["finished_at_ns"] < formal_wall["started_at_ns"]
        or formal_wall.get("elapsed_ns")
        != formal_wall["finished_at_ns"] - formal_wall["started_at_ns"]
    ):
        raise R3Error("formal wall timing evidence is incomplete")
    completion = result.get("completion_receipt")
    through_main = (
        completion.get("through_main_readback_wall")
        if isinstance(completion, Mapping)
        else None
    )
    completion_boundary = (
        completion.get("completion_boundary")
        if isinstance(completion, Mapping)
        else None
    )
    boundary_source = (
        completion_boundary.get("source")
        if isinstance(completion_boundary, Mapping)
        else None
    )
    boundary_production = (
        completion_boundary.get("production")
        if isinstance(completion_boundary, Mapping)
        else None
    )
    post_completion_readback = result.get("post_completion_readback")
    final_scope_recheck = (
        post_completion_readback.get("scope")
        if isinstance(post_completion_readback, Mapping)
        else None
    )
    final_source = (
        final_scope_recheck.get("source")
        if isinstance(final_scope_recheck, Mapping)
        else None
    )
    final_production = (
        final_scope_recheck.get("production")
        if isinstance(final_scope_recheck, Mapping)
        else None
    )
    if require_completion and (
        not isinstance(completion, Mapping)
        or completion.get("readback_verified") is not True
        or completion.get("sealed_artifact_id") != completion.get("main_artifact_id")
        or completion.get("sealed_artifact_sha256")
        != completion.get("main_artifact_sha256")
        or completion.get("main_artifact_persistence_included") is not True
        or completion.get("main_artifact_readback_included") is not True
        or not isinstance(through_main, Mapping)
        or through_main.get("main_artifact_persistence_included") is not True
        or through_main.get("main_artifact_readback_included") is not True
        or not isinstance(completion_boundary, Mapping)
        or not isinstance(boundary_source, Mapping)
        or boundary_source.get("after") != source.get("after")
        or boundary_source.get("head_after") != source.get("head_after")
        or not isinstance(boundary_production, Mapping)
        or boundary_production.get("production_workset_unchanged") is not True
        or boundary_production.get("excluded_not_evaluated")
        != R3_EXCLUDED_NOT_EVALUATED
        or not isinstance(post_completion_readback, Mapping)
        or not isinstance(final_scope_recheck, Mapping)
        or not isinstance(final_source, Mapping)
        or final_source.get("matches_sealed_boundary") is not True
        or not isinstance(final_production, Mapping)
        or final_production.get("production_workset_unchanged") is not True
        or final_production.get("matches_sealed_boundary") is not True
    ):
        raise R3Error("completion receipt evidence is incomplete")
    cleanup = result.get("cleanup")
    if (
        not isinstance(cleanup, Mapping)
        or cleanup.get("remaining") != 0
        or cleanup.get("clone_owned") is not True
        or cleanup.get("external_input_preserved")
        != (manifest.get("external") is True)
    ):
        raise R3Error("temporary cleanup evidence is incomplete")
    write_boundary = result.get("production_write_boundary")
    owned_root = (
        write_boundary.get("owned_root")
        if isinstance(write_boundary, Mapping)
        else None
    )
    if (
        not isinstance(write_boundary, Mapping)
        or write_boundary.get("path_overlap_rejected") is not True
        or write_boundary.get("owned_clone_only") is not True
        or write_boundary.get("production_workset_unchanged") is not True
        or not isinstance(owned_root, Mapping)
        or owned_root.get("filesystem") != "apfs"
        or owned_root.get("root_identity_unchanged") is not True
        or owned_root.get("cleanup_remaining") != 0
    ):
        raise R3Error("production write boundary evidence is incomplete")
    if (
        source.get("head_before") != source.get("head_after")
        or source.get("status_count_before") != 0
        or source.get("status_count_after") != 0
        or source.get("status_sha256_before") != source.get("status_sha256_after")
        or source.get("tree_unchanged") is not True
        or source.get("head_rechecked_at_exit") is not True
        or source.get("bytecode_disabled_during_run") is not True
    ):
        raise R3Error("source immutability evidence is incomplete")
    if result.get("duplicates") != 0:
        raise R3Error("formal artifact duplicate count is non-zero")
    excluded_observation = result.get("excluded_scope_observation")
    owned_unchanged = result.get("production_workset_unchanged")
    if (
        not isinstance(excluded_observation, Mapping)
        or excluded_observation.get("detected") is not None
        or not isinstance(owned_unchanged, bool)
        or excluded_observation.get("classification") != "excluded-not-evaluated"
        or not owned_unchanged
    ):
        raise R3Error("excluded production scope observation is incomplete")


def _run_once_guarded(
    *,
    production: Path | None = None,
    clone_root: Path | None = None,
    manifest_path: Path | None = None,
    source_root: Path,
    source_commit: str,
    output: Path,
    samples: int,
) -> dict[str, Any]:
    formal_started_ns = time.time_ns()
    if samples < MIN_SAMPLES:
        raise R3Error(f"formal R3 sample count must be at least {MIN_SAMPLES}")
    if _COMMIT_RE.fullmatch(source_commit) is None:
        raise R3Error("source commit must be a full lowercase SHA-1")
    external_clone = clone_root is not None or manifest_path is not None
    if (clone_root is None) != (manifest_path is None):
        raise R3Error("--clone-root and --manifest must be supplied together")
    if external_clone:
        raise R3Error(
            "external frozen clone input is non-certifying; use --production-root"
        )
    if external_clone and production is not None:
        raise R3Error("production root cannot be combined with a trusted clone input")
    if not external_clone and production is None:
        raise R3Error("production root or trusted clone input is required")
    if external_clone:
        assert clone_root is not None and manifest_path is not None
        _assert_external_clone_not_production(clone_root)
        _assert_input_matrix(clone_root, source_root, manifest_path, output)
    else:
        assert production is not None
        _assert_root_matrix(production, source_root, output)
    _assert_output_safe(output)
    source_head_before = _git_head(source_root)
    if source_head_before != source_commit:
        raise R3Error("source commit does not match source HEAD")
    source_before = _source_snapshot(source_root)
    _assert_source_clean(source_before, when="before run")
    workset_module, store = _load_runtime(source_root)
    production_before = (
        _production_snapshot(
            production,
            include_raw=False,
            store=store,
            require_checkpoint_file_state=True,
        )
        if production is not None
        else None
    )
    production_workset_before = (
        _clone_workset_inventory(_clone_workset_path(production))
        if production is not None
        else None
    )
    production_owned_before = (
        _production_owned_snapshot(production, store)
        if production is not None
        else None
    )
    manifest: dict[str, Any] | None = None
    manifest_state_before: dict[str, int] | None = None
    manifest_bytes_sha: str | None = None
    clone_filesystem: str | None = None
    external_input_filesystem: str | None = None
    clone_owned = False
    external_input_root: Path | None = None
    external_input_identity_before: dict[str, int] | None = None
    external_input_stable_tree_before: dict[str, Any] | None = None
    external_input_raw_before: dict[str, Any] | None = None
    external_input_static_before: dict[str, Any] | None = None
    external_input_workset_before: dict[str, Any] | None = None
    external_input_identity_after: dict[str, int] | None = None
    external_input_stable_tree_after: dict[str, Any] | None = None
    external_input_raw_after: dict[str, Any] | None = None
    external_input_static_after: dict[str, Any] | None = None
    external_input_workset_after: dict[str, Any] | None = None
    external_input_verified_after = False
    production_owned_clone_boundary: dict[str, Any] | None = None
    production_protected_clone_boundary: dict[str, Any] | None = None
    clone_scope_before: dict[str, Any] | None = None
    normalization: dict[str, Any] | None = None
    if external_clone:
        assert clone_root is not None and manifest_path is not None
        external_input_root = clone_root
        external_input_filesystem = _probe_apfs_clone(clone_root)
        manifest, manifest_state_before, manifest_bytes_sha = _read_frozen_manifest(
            manifest_path,
            store,
            clone_root=clone_root,
            source_commit=source_commit,
        )
        _assert_manifest_source_binding(
            manifest,
            source_before,
            _source_runtime_identity(source_root, source_commit),
        )
        external_input_identity_before = _clone_root_identity(clone_root)
        external_input_stable_tree_before = _clone_tree_state_digest(
            clone_root,
            ignore_mutable_workset=True,
            store=store,
        )
        external_input_raw_before = _bounded_raw_identity(
            clone_root, "external clone.raw_tree"
        )
        external_input_static_before = _clone_static_identity(store, clone_root)
        external_input_workset_before = _clone_workset_inventory(
            _clone_workset_path(clone_root)
        )
        clone_root = _clone_from_root(clone_root)
        clone_owned = True
        try:
            _assert_input_matrix(clone_root, source_root, manifest_path, output)
            if external_input_root is not None and _path_overlap(
                clone_root, external_input_root
            ):
                raise R3Error("owned clone overlaps external input")
            clone_filesystem = _probe_apfs_clone(clone_root)
        except BaseException:
            _cleanup_clone(clone_root)
            raise
    else:
        clone_owned = True
        clone_root = _clone_from_root(production)
        try:
            _assert_root_matrix(production, source_root, output, (clone_root,))
            clone_filesystem = _probe_apfs_clone(clone_root)
        except BaseException:
            _cleanup_clone(clone_root)
            raise
    try:
        clone_identity_before = _clone_root_identity(clone_root)
        _assert_checkpointed_clone_sidecars(clone_root / OX_WORKSET_RELATIVE)
        # First capture the clone's read-only semantic prestate.  APFS copies
        # SQLite's main DB and WAL/SHM sidecars independently; normalization
        # may discard a stale sidecar frame, so this parity gate must precede
        # checkpointing and fail closed on malformed input.
        clone_workset_before = _clone_workset_inventory(clone_root / OX_WORKSET_RELATIVE)
        if production is not None:
            production_protected_clone_boundary = _production_snapshot(
                production,
                include_raw=False,
                store=store,
                require_checkpoint_file_state=True,
            )
            if not _production_protected_equal(
                production_before, production_protected_clone_boundary
            ):
                raise R3Error("production Workset scope changed during clone creation")
            production_owned_clone_boundary = _production_owned_snapshot(production, store)
            clone_scope_before = _production_snapshot(clone_root, store=store)
            if _production_scope_identity(clone_scope_before) != _production_scope_identity(
                production_protected_clone_boundary
            ):
                raise R3Error("APFS clone Workset scope prestate differs from production")
        if external_clone:
            if _workset_identity(clone_workset_before) != _workset_identity(
                external_input_workset_before or {}
            ):
                raise R3Error("owned clone Workset prestate differs from external input")
        elif production_workset_before is not None and _workset_identity(
            clone_workset_before
        ) != _workset_identity(production_workset_before):
            raise R3Error("APFS clone Workset prestate differs from production")
        # Only after semantic parity is proven, checkpoint the owned clone.  A
        # post-normalization inventory must remain byte-independent but
        # logically identical, proving no stale WAL data was silently lost.
        normalization = _normalize_clone_workset(
            clone_root / OX_WORKSET_RELATIVE,
            expected_identity=_workset_identity(clone_workset_before),
        )
        if production_workset_before is not None and normalization["semantic_after"] != _workset_identity(
            production_workset_before
        ):
            raise R3Error("normalized clone Workset differs from production prestate")
    except BaseException:
        if clone_owned and clone_root.exists():
            _cleanup_clone(clone_root)
        raise
    work_root: Path | None = None
    try:
        clone_workset = _run_clone_workset_cycles(workset_module, clone_root, cycles=samples)
        if normalization is None:
            raise R3Error("clone Workset normalization evidence is missing")
        clone_workset["normalization"] = normalization
        actual_teacher_handoff = _clone_teacher_handoff(
            workset_module, source_root, clone_root, sample_count=samples
        )
        actual_clone_sigterm = _sigterm_reopen(
            workset_module,
            source_root,
            clone_root / OX_WORKSET_RELATIVE,
            expected_rows=OX_WORKSET_EXPECTED_ROWS,
        )
        sigterm_normalization = _normalize_clone_workset(
            clone_root / OX_WORKSET_RELATIVE,
            expected_identity=actual_clone_sigterm["inventory_after"],
            allow_wal=True,
        )
        sigterm_normalized_inventory = _clone_workset_inventory(
            clone_root / OX_WORKSET_RELATIVE,
            expected_rows=OX_WORKSET_EXPECTED_ROWS,
            require_receipts=True,
        )
        if _workset_identity(sigterm_normalized_inventory) != _workset_identity(
            actual_clone_sigterm["inventory_after"]
        ):
            raise R3Error("clone SIGTERM inventory changed during normalization")
        actual_clone_sigterm["normalization"] = sigterm_normalization
        actual_clone_sigterm["inventory_after_normalized"] = _workset_identity(
            sigterm_normalized_inventory
        )
        work_root = Path(tempfile.mkdtemp(prefix=".r3-harness-", dir=clone_root))
        synthetic = _run_workset(workset_module, source_root, work_root, samples)
        result = dict(synthetic)
        result["synthetic_claim"] = result["claim"]
        result["synthetic_teacher_handoff"] = result["teacher_handoff"]
        result["synthetic_sigterm_reopen"] = result["sigterm_reopen"]
        result["actual_clone_sigterm"] = actual_clone_sigterm
        result["sigterm_reopen"] = actual_clone_sigterm
        result["teacher_handoff"] = actual_teacher_handoff
        result["claim"] = {
            "samples": clone_workset["successful_cycles"],
            "observation_calls": clone_workset["observation_calls"],
            "p95_ns": clone_workset["claim_p95_ns"],
            "threshold_ns": CLAIM_P95_LIMIT_NS,
            "successful_count": clone_workset["successful_cycles"],
            "source": "r2-frozen-clone-production-ox-workset",
        }
        result["clone_workset"] = clone_workset
        result["sigterm_process"] = actual_clone_sigterm["sigterm_process"]
        clone_workset_after = _clone_workset_inventory(
            clone_root / OX_WORKSET_RELATIVE,
            expected_rows=OX_WORKSET_EXPECTED_ROWS,
            require_receipts=True,
        )
        clone_workset["inventory_after"] = clone_workset_after
        clone_scope_after = _production_snapshot(clone_root, store=store)
        if clone_scope_before is not None and (
            _production_scope_identity(clone_scope_before)["state_pointers"]
            != _production_scope_identity(clone_scope_after)["state_pointers"]
            or _production_scope_identity(clone_scope_before)["locks"]
            != _production_scope_identity(clone_scope_after)["locks"]
        ):
            raise R3Error("clone Workset state/pointer/lock scope changed unexpectedly")
        clone_identity_after = _clone_root_identity(clone_root)
        if clone_identity_after != clone_identity_before:
            raise R3Error("frozen clone root identity changed during R3")
        source_after = _source_snapshot(source_root)
        source_head_after = _git_head(source_root)
        _assert_source_clean(source_after, when="after run")
        if source_head_after != source_head_before or source_after != source_before:
            raise R3Error("source changed during R3 run")
        shutil.rmtree(work_root, ignore_errors=True)
        if work_root.exists():
            raise R3Error("R3 workset temp cleanup failed")
        work_root = None
        if external_clone:
            assert manifest_path is not None and manifest_state_before is not None
            verified_after, manifest_state_after, manifest_bytes_after = (
                _read_frozen_manifest(
                    manifest_path,
                    store,
                    clone_root=external_input_root or clone_root,
                    source_commit=source_commit,
                )
            )
            if (
                manifest_state_after != manifest_state_before
                or manifest_bytes_after != manifest_bytes_sha
                or verified_after != manifest
            ):
                raise R3Error("R2 frozen manifest changed during R3")
            assert external_input_root is not None
            external_input_identity_after = _clone_root_identity(external_input_root)
            external_input_stable_tree_after = _clone_tree_state_digest(
                external_input_root,
                ignore_mutable_workset=True,
                store=store,
            )
            external_input_raw_after = _bounded_raw_identity(
                external_input_root, "external clone.raw_tree.after"
            )
            external_input_static_after = _clone_static_identity(
                store, external_input_root
            )
            external_input_workset_after = _clone_workset_inventory(
                _clone_workset_path(external_input_root)
            )
            if (
                external_input_identity_after != external_input_identity_before
                or external_input_stable_tree_after
                != external_input_stable_tree_before
                or external_input_raw_after != external_input_raw_before
                or external_input_static_after != external_input_static_before
                or _workset_identity(external_input_workset_after)
                != _workset_identity(external_input_workset_before or {})
            ):
                raise R3Error("external frozen clone changed during R3")
            external_input_verified_after = True
        production_after = (
            _production_snapshot(
                production,
                include_raw=False,
                store=store,
                require_checkpoint_file_state=True,
            )
            if production is not None
            else None
        )
        production_owned_after = (
            _production_owned_snapshot(production, store)
            if production is not None
            else None
        )
        production_workset_after = (
            production_owned_after["workset"]
            if production_owned_after is not None
            else None
        )
        production_owned_unchanged = production is None or (
            production_owned_after == production_owned_before
        )
        if production is not None and not production_owned_unchanged:
            raise R3Error("production changed during R3 run")
        production_workset_unchanged = production is None or _production_protected_equal(
            production_before, production_after
        )
        if production is not None and not production_workset_unchanged:
            raise R3Error("production protected runtime changed during R3 run")
        excluded_scope_observation = {
            "detected": None,
            "raw_tree_changed": None,
            "classification": "excluded-not-evaluated",
            "scope": dict(R3_EXCLUDED_NOT_EVALUATED),
        }
        result["manifest"] = {
            "external": external_clone,
            "schema": manifest["schema"] if manifest else R3_CLONE_SCHEMA,
            "artifact_id": manifest["artifact_id"] if manifest else None,
            "seal_sha256": manifest["seal_sha256"] if manifest else None,
            "content_sha256": manifest_bytes_sha,
            "filesystem": clone_filesystem,
            "filesystem_probe": "r0",
            "seal_verified": external_clone,
            "content_identity_verified": external_clone,
            "clone_root_exact": True,
            "raw_parity": None,
            "static_parity": None,
            "toctou_rechecked": True,
            "scope": "Workset DB/receipt chain/state/pointers/locks",
            "excluded_not_evaluated": dict(R3_EXCLUDED_NOT_EVALUATED),
            "workset": {
                "relative_path": OX_WORKSET_RELATIVE.as_posix(),
                "prestate": _workset_identity(clone_workset_before),
                "poststate": _workset_identity(clone_workset_after),
                "content_sha256_before": clone_workset_before["content_sha256"],
                "content_sha256_after": clone_workset_after["content_sha256"],
                "state_sha256_before": clone_workset_before["state_sha256"],
                "state_sha256_after": clone_workset_after["state_sha256"],
                "state_seal_sha256_before": clone_workset_before[
                    "state_seal_sha256"
                ],
                "state_seal_sha256_after": clone_workset_after["state_seal_sha256"],
                "receipt_chain_sha256_before": clone_workset_before[
                    "receipt_chain_sha256"
                ],
                "receipt_chain_sha256_after": clone_workset_after[
                    "receipt_chain_sha256"
                ],
                "prestate_verified": True,
            },
            "input_clone": {
                "external": external_clone,
                "filesystem": external_input_filesystem,
                "owned_scope_before": external_input_stable_tree_before,
                "owned_scope_after": external_input_stable_tree_after,
                "owned_scope_unchanged": external_clone is False
                or external_input_stable_tree_after == external_input_stable_tree_before,
                "excluded_not_evaluated": dict(R3_EXCLUDED_NOT_EVALUATED),
                "root_identity_unchanged": external_clone is False
                or external_input_identity_after == external_input_identity_before,
                "workset_before": (
                    _workset_identity(external_input_workset_before)
                    if external_input_workset_before is not None
                    else None
                ),
                "workset_after": (
                    _workset_identity(external_input_workset_after)
                    if external_input_workset_after is not None
                    else None
                ),
            },
        }
        result["clone_tree"] = {
            "scope": "clone Workset DB/receipt chain/state/pointers/locks",
            "representation": R3_WORKSET_SCOPE_REPRESENTATION,
            "excluded_not_evaluated": dict(R3_EXCLUDED_NOT_EVALUATED),
            "before": {
                "workset": _workset_identity(clone_workset_before),
                "scope": _production_scope_identity(clone_scope_before)
                if clone_scope_before is not None
                else None,
            },
            "after": {
                "workset": _workset_identity(clone_workset_after),
                "scope": _production_scope_identity(clone_scope_after),
            },
            "owned_scope_unchanged": True,
            "root_identity_unchanged": True,
            "inventory_before": clone_workset_before["inventory_sha256"],
            "inventory_after": clone_workset_after["inventory_sha256"],
        }
        if clone_owned:
            _cleanup_clone(clone_root)
            clone_remaining = int(clone_root.exists())
        else:
            clone_remaining = 0
        result["cleanup"] = {
            "temporary_roots": 1,
            "clone_owned": clone_owned,
            "remaining": clone_remaining,
            "external_input_preserved": external_clone,
        }
        result["excluded_scope_observation"] = excluded_scope_observation
        result["production_workset_unchanged"] = production_workset_unchanged
        result["production_write_boundary"] = {
            "path_overlap_rejected": True,
            "owned_clone_only": True,
            "production_workset_unchanged": production_workset_unchanged,
            "owned_root": {
                "filesystem": clone_filesystem,
                "root_identity_unchanged": True,
                "cleanup_remaining": clone_remaining,
            },
        }
        result["duplicates"] = int(result.get("duplicates", 0)) + int(
            clone_workset.get("duplicates", 0)
        )
        if _git_head(source_root) != source_head_before:
            raise R3Error("source HEAD changed at R3 exit")
        if _has_symlink_component(output):
            raise R3Error("output path contains a symlink before artifact write")
        _assert_output_safe(output)
        if production is not None:
            _assert_root_matrix(production, source_root, output)
        run_formal_finished_ns = time.time_ns()
        result["pre_publication_wall"] = {
            "started_at_ns": formal_started_ns,
            "finished_at_ns": run_formal_finished_ns,
            "elapsed_ns": run_formal_finished_ns - formal_started_ns,
        }
        payload = {
            "runtime": {"source_commit": source_commit, "external_provider_calls": 0},
            "production_scope": "workset-only",
            "source_unchanged": source_after == source_before,
            "source": {
                "before": source_before,
                "after": source_after,
                "head_before": source_head_before,
                "head_after": source_head_after,
                "head_rechecked_at_exit": True,
                "status_count_before": source_before["git_status_count"],
                "status_count_after": source_after["git_status_count"],
                "status_sha256_before": source_before["git_status_sha256"],
                "status_sha256_after": source_after["git_status_sha256"],
                "tree_unchanged": source_before == source_after,
                "clean_before": source_before["git_status_count"] == 0,
                "clean_after": source_after["git_status_count"] == 0,
                "bytecode_disabled_during_run": True,
            },
            "production": {
                "used": production is not None,
                "production_workset_unchanged": production_workset_unchanged,
                "protected_before": production_before,
                "protected_after": production_after,
                "workset_before": production_workset_before,
                "workset_after": production_workset_after,
                "owned_before": production_owned_before,
                "owned_after": production_owned_after,
                "excluded_scope_observation": excluded_scope_observation,
                "excluded_not_evaluated": dict(R3_EXCLUDED_NOT_EVALUATED),
                "clone_boundary": {
                    "protected_before": production_before,
                    "protected_after": production_protected_clone_boundary,
                    "workset_unchanged": production is None
                    or _production_protected_equal(
                        production_before, production_protected_clone_boundary
                    ),
                    "owned_before": production_owned_before,
                    "owned_after": production_owned_clone_boundary,
                    "owned_unchanged": production is None
                    or production_owned_clone_boundary == production_owned_before,
                    "scope_prestate_verified": production is None
                    or (
                        clone_scope_before is not None
                        and _production_scope_identity(clone_scope_before)
                        == _production_scope_identity(production_protected_clone_boundary)
                    ),
                },
                "production_write_boundary": {
                    "path_overlap_rejected": True,
                    "owned_clone_only": True,
                    "production_workset_unchanged": production_workset_unchanged,
                    "owned_root": {
                        "filesystem": clone_filesystem,
                        "root_identity_unchanged": True,
                        "cleanup_remaining": clone_remaining,
                    },
                },
            },
            "manifest": result["manifest"],
            "pre_publication_wall": result["pre_publication_wall"],
            "phases": result["phases"],
            "claim": result["claim"],
            "teacher_handoff": result["teacher_handoff"],
            "synthetic_teacher_handoff": result["synthetic_teacher_handoff"],
            "durability": result["durability"],
            "duplicates": result["duplicates"],
            "clone_workset": clone_workset,
            "sigterm_process": result["sigterm_process"],
            "actual_clone_sigterm": result["actual_clone_sigterm"],
            "synthetic_sigterm_reopen": result["synthetic_sigterm_reopen"],
            "samples": result["samples"],
            "admitted_cycles": result["admitted_cycles"],
            "successful_cycles": result["claim"]["successful_count"],
            "clone_tree": result["clone_tree"],
            "cleanup": result["cleanup"],
            "clone_temp_cleanup_verified": result["cleanup"]["remaining"] == 0,
            "result": result,
            "thresholds": {
                "claim_p95_ns": CLAIM_P95_LIMIT_NS,
                "teacher_handoff_ns": TEACHER_HANDOFF_LIMIT_NS,
                "durable_coverage_pct": RECEIPT_COVERAGE_LIMIT,
            },
        }
        _assert_formal_acceptance(
            result, payload["source"], require_completion=False
        )
        _assert_payload_free(payload)
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        if len(encoded) > 2 * 1024 * 1024:
            raise R3Error("R3 evidence exceeds bounded size")
        if _has_symlink_component(output):
            raise R3Error("output path contains a symlink before artifact write")
        _assert_output_safe(output)
        if production is not None:
            _assert_root_matrix(production, source_root, output)
        artifact_id, artifact_path, artifact = store.write_immutable(
            output, payload, schema=R3_SCHEMA
        )
        artifact_before = _artifact_file_snapshot(artifact_path)
        readback = store.read_sealed(artifact_path, schema=R3_SCHEMA)
        artifact_after = _artifact_file_snapshot(artifact_path)
        if artifact_before != artifact_after:
            raise R3Error("R3 immutable artifact changed during readback")
        if readback != artifact or readback.get("artifact_id") != artifact_id:
            raise R3Error("R3 immutable artifact readback mismatch")
        if _has_symlink_component(output):
            raise R3Error("output path contains a symlink before completion write")
        _assert_output_safe(output)
        if production is not None:
            _assert_root_matrix(production, source_root, output)
        # The completion receipt is published only after an independent
        # source/Workset boundary recheck.  The same semantic values are
        # re-read once more after completion readback below; that post-readback
        # probe is compared with this sealed boundary and the initial pre-run
        # identity, so a late mutation cannot be hidden by artifact timing.
        completion_source = _source_snapshot(source_root)
        completion_source_head = _git_head(source_root)
        _assert_source_clean(completion_source, when="before completion receipt")
        if completion_source != source_before or completion_source_head != source_head_before:
            raise R3Error("source changed before completion receipt")
        completion_production = (
            _production_snapshot(
                production,
                include_raw=False,
                store=store,
                require_checkpoint_file_state=True,
            )
            if production is not None
            else None
        )
        completion_production_owned = (
            _production_owned_snapshot(production, store)
            if production is not None
            else None
        )
        completion_workset_unchanged = production is None or (
            _production_protected_equal(production_before, completion_production)
            and completion_production_owned == production_owned_before
        )
        if not completion_workset_unchanged:
            raise R3Error("production Workset changed before completion receipt")
        completion_boundary = {
            "source": {
                "before": source_before,
                "after": completion_source,
                "head_before": source_head_before,
                "head_after": completion_source_head,
            },
            "production": {
                "scope": "workset-only",
                "protected_before": production_before,
                "protected_after": completion_production,
                "owned_before": production_owned_before,
                "owned_after": completion_production_owned,
                "production_workset_unchanged": completion_workset_unchanged,
                "excluded_not_evaluated": dict(R3_EXCLUDED_NOT_EVALUATED),
            },
        }
        completion_started_ns = time.time_ns()
        completion_probe = {
            "started_at_ns": completion_started_ns,
            "main_artifact_id": artifact_id,
            "main_artifact_sha256": artifact_after["sha256"],
        }
        completion_payload = {
            "sealed_artifact_id": artifact_id,
            "sealed_artifact_sha256": artifact_after["sha256"],
            "sealed_artifact_file_state": artifact_after["file_state"],
            "completion_probe": completion_probe,
            "readback_verified": True,
            "source_commit": source_commit,
            "completion_boundary": completion_boundary,
        }
        through_main_readback_wall = {
            "started_at_ns": formal_started_ns,
            "finished_at_ns": completion_started_ns,
            "elapsed_ns": completion_started_ns - formal_started_ns,
            "main_artifact_persistence_included": True,
            "main_artifact_readback_included": True,
        }
        completion_payload.update(
            {
                "main_artifact_id": artifact_id,
                "main_artifact_sha256": artifact_after["sha256"],
                "main_artifact_persistence_included": True,
                "main_artifact_readback_included": True,
                "through_main_readback_wall": through_main_readback_wall,
            }
        )
        completion_id, completion_path, completion_artifact = store.write_immutable(
            output, completion_payload, schema=R3_COMPLETION_SCHEMA
        )
        completion_before = _artifact_file_snapshot(completion_path)
        completion_readback = store.read_sealed(
            completion_path, schema=R3_COMPLETION_SCHEMA
        )
        completion_after = _artifact_file_snapshot(completion_path)
        if completion_before != completion_after:
            raise R3Error("R3 completion receipt changed during readback")
        if (
            completion_readback != completion_artifact
            or completion_readback.get("main_artifact_id") != artifact_id
            or completion_readback.get("main_artifact_sha256")
            != artifact_after["sha256"]
            or completion_readback.get("through_main_readback_wall")
            != through_main_readback_wall
            or completion_readback.get("completion_boundary") != completion_boundary
        ):
            raise R3Error("R3 completion receipt readback mismatch")
        final_source = _source_snapshot(source_root)
        final_source_head = _git_head(source_root)
        _assert_source_clean(final_source, when="after completion receipt readback")
        final_production = (
            _production_snapshot(
                production,
                include_raw=False,
                store=store,
                require_checkpoint_file_state=True,
            )
            if production is not None
            else None
        )
        final_production_owned = (
            _production_owned_snapshot(production, store)
            if production is not None
            else None
        )
        final_workset_unchanged = production is None or (
            _production_protected_equal(production_before, final_production)
            and final_production_owned == production_owned_before
        )
        if (
            final_source != completion_source
            or final_source_head != completion_source_head
            or not final_workset_unchanged
            or not _production_protected_equal(completion_production, final_production)
            or final_production_owned != completion_production_owned
        ):
            raise R3Error("source or production Workset changed after completion readback")
        final_scope_recheck = {
            "source": {
                "after_completion_readback": final_source,
                "head_after_completion_readback": final_source_head,
                "matches_sealed_boundary": final_source == completion_source
                and final_source_head == completion_source_head,
            },
            "production": {
                "protected_after_completion_readback": final_production,
                "owned_after_completion_readback": final_production_owned,
                "production_workset_unchanged": final_workset_unchanged,
                "matches_sealed_boundary": _production_protected_equal(
                    completion_production, final_production
                )
                and final_production_owned == completion_production_owned,
                "excluded_not_evaluated": dict(R3_EXCLUDED_NOT_EVALUATED),
            },
        }
        final_artifact_probe = _artifact_file_snapshot(artifact_path)
        if final_artifact_probe != artifact_after:
            raise R3Error("R3 main artifact changed after completion publication")
        _assert_output_safe(output)
        completion_finished_ns = time.time_ns()
        completion_wall = {
            "started_at_ns": completion_started_ns,
            "finished_at_ns": completion_finished_ns,
            "elapsed_ns": completion_finished_ns - completion_started_ns,
            "main_artifact_persisted_and_readback": True,
            "completion_receipt_persisted_and_readback": True,
            "main_artifact_rechecked": True,
        }
        outer_end_to_end_wall = {
            "started_at_ns": formal_started_ns,
            "finished_at_ns": completion_finished_ns,
            "elapsed_ns": completion_finished_ns - formal_started_ns,
            "sealed_scope": "external_completion_readback",
            "through_main_readback_wall": through_main_readback_wall,
            "completion_receipt_readback_observed": True,
        }
        result["formal_wall"] = outer_end_to_end_wall
        result["post_completion_readback"] = {
            "scope": final_scope_recheck,
            "completion_wall": completion_wall,
            "outer_end_to_end_wall": outer_end_to_end_wall,
            "authority": "external-watchdog-attestation",
        }
        result["completion_receipt"] = {
            "schema": completion_artifact["schema"],
            "artifact_id": completion_id,
            "sealed_artifact_id": artifact_id,
            "main_artifact_id": artifact_id,
            "path": str(completion_path),
            "artifact_file_state": completion_after["file_state"],
            "artifact_sha256": completion_after["sha256"],
            "sealed_artifact_sha256": artifact_after["sha256"],
            "main_artifact_sha256": artifact_after["sha256"],
            "readback_verified": True,
            "main_artifact_persistence_included": True,
            "main_artifact_readback_included": True,
            "probe": completion_probe,
            "through_main_readback_wall": through_main_readback_wall,
            "completion_boundary": completion_boundary,
        }
        _assert_formal_acceptance(result, payload["source"])
        _assert_payload_free(result["completion_receipt"])
        _assert_payload_free(result["post_completion_readback"])
        return {
            "schema": artifact["schema"],
            "artifact_id": artifact_id,
            "path": str(artifact_path),
            "samples": result["samples"],
            "claim_p95_ns": result["claim"]["p95_ns"],
            "teacher_handoff_ns": result["teacher_handoff"]["wall_time_ns"],
            "receipt_coverage_pct": result["durability"]["receipt_coverage_pct"],
            "progress_coverage_pct": result["durability"]["progress_coverage_pct"],
            "duplicates": result["duplicates"],
            "artifact_seal_verified": True,
            "artifact_readback_verified": True,
            "artifact_file_state": artifact_after["file_state"],
            "artifact_sha256": artifact_after["sha256"],
            "artifact_path_rechecked": artifact_before["path"] == artifact_after["path"],
            "formal_wall": result["formal_wall"],
            "completion_receipt": result["completion_receipt"],
            "post_completion_readback": result["post_completion_readback"],
            "clone_cleanup_verified": True,
        }
    finally:
        if (
            external_input_root is not None
            and external_input_identity_before is not None
            and not external_input_verified_after
        ):
            try:
                if manifest_path is None or manifest is None or store is None:
                    raise R3Error("external frozen clone verification context is missing")
                verified_failure, state_failure, bytes_failure = _read_frozen_manifest(
                    manifest_path,
                    store,
                    clone_root=external_input_root,
                    source_commit=source_commit,
                )
                if (
                    manifest_state_before is None
                    or manifest_bytes_sha is None
                    or state_failure != manifest_state_before
                    or bytes_failure != manifest_bytes_sha
                    or verified_failure != manifest
                ):
                    raise R3Error("external frozen manifest changed during failed R3")
                if (
                    _clone_root_identity(external_input_root)
                    != external_input_identity_before
                    or _clone_tree_state_digest(
                        external_input_root,
                        ignore_mutable_workset=True,
                        store=store,
                    )
                    != external_input_stable_tree_before
                    or _bounded_raw_identity(
                        external_input_root, "external clone.raw_tree.failure"
                    )
                    != external_input_raw_before
                    or _clone_static_identity(store, external_input_root)
                    != external_input_static_before
                    or _workset_identity(
                        _clone_workset_inventory(_clone_workset_path(external_input_root))
                    )
                    != _workset_identity(external_input_workset_before or {})
                ):
                    raise R3Error("external frozen clone changed during failed R3")
            except BaseException as exc:
                raise R3Error("external frozen clone preservation check failed") from exc
        if work_root is not None:
            shutil.rmtree(work_root, ignore_errors=True)
            if work_root.exists():
                raise R3Error("R3 workset temp cleanup failed")
        if clone_owned and clone_root is not None and clone_root.exists():
            _cleanup_clone(clone_root)


def _run_once(
    *,
    production: Path | None = None,
    clone_root: Path | None = None,
    manifest_path: Path | None = None,
    source_root: Path,
    source_commit: str,
    output: Path,
    samples: int,
) -> dict[str, Any]:
    """Run one formal R3 gate with bytecode generation disabled throughout."""

    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        return _run_once_guarded(
            production=production,
            clone_root=clone_root,
            manifest_path=manifest_path,
            source_root=source_root,
            source_commit=source_commit,
            output=output,
            samples=samples,
        )
    finally:
        sys.dont_write_bytecode = previous


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production-root", type=Path)
    parser.add_argument("--clone-root", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--isolated-root", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        if args.isolated_root is not None:
            raise R3Error("--isolated-root is unsupported; use the trusted R2 clone")
        if args.production_root is None and args.clone_root is None:
            raise R3Error("--production-root or --clone-root is required")
        if (args.clone_root is None) != (args.manifest is None):
            raise R3Error("--clone-root and --manifest must be supplied together")
        paths = tuple(
            path
            for path in (args.production_root, args.clone_root, args.manifest, args.source_root, args.output)
            if path is not None
        )
        if any(_has_symlink_component(path.expanduser()) for path in paths):
            raise R3Error("input/output path contains a symlink")
        production = (
            args.production_root.expanduser().resolve(strict=True)
            if args.production_root is not None
            else None
        )
        clone_root = (
            args.clone_root.expanduser().resolve(strict=True)
            if args.clone_root is not None
            else None
        )
        manifest_path = (
            args.manifest.expanduser().resolve(strict=True)
            if args.manifest is not None
            else None
        )
        source_root = args.source_root.expanduser().resolve(strict=True)
        output = args.output.expanduser().resolve(strict=False)
        result = _run_once(
            production=production,
            clone_root=clone_root,
            manifest_path=manifest_path,
            source_root=source_root,
            source_commit=args.source_commit,
            output=output,
            samples=args.samples,
        )
        print(json.dumps(result, sort_keys=True))
    except (R3Error, OSError, ValueError, sqlite3.Error) as exc:
        print(f"r3 harness failed: {str(exc).split(':', 1)[0]}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
