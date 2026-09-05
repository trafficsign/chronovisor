#!/usr/bin/env python3.14
"""Provider-free, fail-closed readiness evidence for Recall R8/P9.

This program only observes a source checkout and a bounded, read-only
production observation.  It never calls a provider, starts a process, or
deletes/mutates production data.  The artifact is a readiness decision for a
future operator; it is not a cleanup executor.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import re
import sqlite3
import stat
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

R8_SCHEMA = "chronovisor.recall-r8-readiness.v1"
R8_PHASE_RECEIPT_SCHEMA = "chronovisor.recall-r8-phase-receipt.v1"
R8_OBSERVATION_SCHEMA = "chronovisor.recall-r8-production-observation.v1"
R7_SCHEMA = "chronovisor.recall-r7.v2"
R7_LIVE_ATTESTATION_SCHEMA = "chronovisor.recall-r7-live-attestation.v1"
MAX_INPUT_BYTES = 64 * 1024 * 1024
MAX_HASH_BYTES = 64 * 1024 * 1024
MAX_FILES = 2_000
MIN_R7_DAYS = 7
MIN_R7_PAIRED = 500
_R7_STAGES = (("shadow", 0), ("5", 5), ("25", 25), ("100", 100))
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_SENSITIVE = re.compile(
    r"(?:^|_)(?:raw|payload|content|body|prompt|query|token|secret|credential)(?:_|$)",
    re.IGNORECASE,
)

PHASES: tuple[dict[str, Any], ...] = (
    {
        "name": "r7_receipt",
        "order": 1,
        "prerequisites": ("r7_certified", "stages_7d_500", "forced_rollback", "active_lkg"),
        "rollback_artifact": "r7-lkg-pointer",
    },
    {
        "name": "ox_drain_archive",
        "order": 2,
        "prerequisites": ("ox_off", "provider_calls_zero", "leased_zero", "distillation_process_lock_absent"),
        "rollback_artifact": "ox-workset-contract-event-label-candidate-archive",
    },
    {
        "name": "r2_fullscan_catalog_parity",
        "order": 3,
        "prerequisites": ("r2_sealed_receipt", "catalog_parity", "raw_watermark_sealed"),
        "rollback_artifact": "r2-catalog-checkpoint",
    },
    {
        "name": "field_growth_authority_retirement",
        "order": 4,
        "prerequisites": ("field_sessions_archived", "authority_receipt_sealed"),
        "rollback_artifact": "field-growth-lkg-snapshot",
    },
    {
        "name": "compat_legacy",
        "order": 5,
        "prerequisites": ("legacy_policy_checkpoint", "legacy_config_checkpoint", "fts_checkpoint"),
        "rollback_artifact": "legacy-compat-lkg-bundle",
    },
    {
        "name": "final_cleanup",
        "order": 6,
        "prerequisites": ("all_prior_phases_sealed", "final_rollback_receipt"),
        "rollback_artifact": "r8-final-lkg-bundle",
    },
)
_R8_PHASE_NAMES = tuple(item["name"] for item in PHASES)

_OX_FILES = {
    "workset": "runtime/recall-distillation/ox-workset.sqlite3",
    "contract": "runtime/recall-distillation/ox-profile-contracts",
    "event": "runtime/recall-distillation/exposure-receipts.jsonl",
    "label": "runtime/recall-distillation/label-ledger.jsonl",
    "candidate_lineage": "runtime/recall-distillation/candidate-ledger.jsonl",
}
_POINTER_FILES = {
    "active": "runtime/recall-distillation/active-policy.json",
    "candidate": "runtime/recall-distillation/candidate-policy.json",
    "lkg": "runtime/recall-distillation/lkg-policy.json",
}
_LEGACY_FILES = {
    "field_sessions": (
        "runtime/recall/field-sessions",
        "runtime/field-sessions",
        "recall/field-sessions",
    ),
    "policy": (
        "runtime/recall/legacy-policy.json",
        "recall/legacy-policy.json",
        "legacy-policy.json",
    ),
    "config": ("config.toml", "runtime/config.toml"),
    "fts_checkpoints": (
        "runtime/fts/checkpoint.json",
        "runtime/search/fts.checkpoint",
        "runtime/recall/fts.checkpoint",
    ),
}
_PROCESS_LOCKS = (
    "runtime/recall-distillation/distillation-worker.lock",
    "runtime/recall-distillation/worker.lock",
    "runtime/recall-distillation/.distillation-worker.lock",
)


class R8Error(ValueError):
    """An R8 evidence or safety-boundary violation."""


def _assert_dependency_tree_bound(source_root: Path) -> None:
    """Reject cached Chronovisor modules loaded from another checkout."""

    bound_root = (source_root / "src").resolve(strict=False)
    for name, module in tuple(sys.modules.items()):
        if not (name == "chronovisor" or name.startswith("chronovisor.")):
            continue
        module_path = getattr(module, "__file__", None)
        if module_path is None:
            continue
        try:
            resolved = Path(module_path).resolve(strict=False)
        except OSError as exc:
            raise R8Error("source-bound Chronovisor dependency cannot be resolved") from exc
        if not resolved.is_relative_to(bound_root):
            raise R8Error("source-bound Chronovisor dependency escaped source root")


def _load_sibling(name: str) -> Any:
    """Compatibility seam for test doubles; production paths are source-bound."""

    raise R8Error(f"unbound sibling validator forbidden: {name}")


def _load_runtime_store(source_root: Path) -> Any:
    """Load the source-bound sealed reader used by R2/R3."""

    source_path = str(source_root / "src")
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    if source_path not in sys.path:
        sys.path.insert(0, source_path)
    try:
        module = importlib.import_module("chronovisor.recall.recall_distillation_store")
    except (ImportError, OSError) as exc:
        raise R8Error("R2/R3 sealed reader unavailable") from exc
    finally:
        sys.dont_write_bytecode = previous
    module_path = getattr(module, "__file__", None)
    if not isinstance(module_path, str) or not Path(module_path).resolve().is_relative_to(source_root / "src"):
        raise R8Error("R2/R3 sealed reader escaped source root")
    _assert_dependency_tree_bound(source_root)
    return module


def _load_r7_evidence(source_root: Path) -> Any:
    """Load the source-bound R7 evidence reader/validator, if available."""

    path = source_root / "src" / "chronovisor" / "recall" / "recall_r7_evidence.py"
    if _has_symlink_component(path) or not path.is_file():
        raise R8Error("R7 official evidence validator unavailable")
    module_name = f"chronovisor_r8_r7_evidence_{source_root.stat().st_ino}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise R8Error("R7 official evidence validator unavailable")
    module = importlib.util.module_from_spec(spec)
    previous_module = sys.modules.get(module_name)
    previous_path = list(sys.path)
    previous_bytecode = sys.dont_write_bytecode
    sys.modules[module_name] = module
    sys.dont_write_bytecode = True
    source_path = str(source_root / "src")
    if source_path not in sys.path:
        sys.path.insert(0, source_path)
    try:
        spec.loader.exec_module(module)
        _assert_dependency_tree_bound(source_root)
    except Exception as exc:
        raise R8Error("R7 official evidence validator unavailable") from exc
    finally:
        sys.path[:] = previous_path
        sys.dont_write_bytecode = previous_bytecode
        if previous_module is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous_module
    validator = getattr(module, "validate_live_attestation", None)
    if not callable(validator):
        raise R8Error("R7 official live validator is missing")
    return module


def _load_source_r7_harness(source_root: Path) -> Any:
    """Load the R7 script validator from the exact source checkout."""

    path = source_root / "scripts" / "recall_r7_harness.py"
    if _has_symlink_component(path) or not path.is_file():
        raise R8Error("R7 source-bound harness unavailable")
    module_name = f"chronovisor_r8_source_r7_{source_root.stat().st_ino}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise R8Error("R7 source-bound harness unavailable")
    module = importlib.util.module_from_spec(spec)
    previous_module = sys.modules.get(module_name)
    previous_path = list(sys.path)
    previous_bytecode = sys.dont_write_bytecode
    sys.modules[module_name] = module
    sys.dont_write_bytecode = True
    source_path = str(source_root / "src")
    if source_path not in sys.path:
        sys.path.insert(0, source_path)
    try:
        spec.loader.exec_module(module)
        _assert_dependency_tree_bound(source_root)
    except Exception as exc:
        raise R8Error("R7 source-bound harness unavailable") from exc
    finally:
        sys.path[:] = previous_path
        sys.dont_write_bytecode = previous_bytecode
        if previous_module is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous_module
    return module


def _load_source_script(source_root: Path, name: str) -> Any:
    """Load one official harness from the exact source checkout.

    R8 never lets the working-tree helper implementation stand in for the
    version named by ``source_commit``.  A missing or old source-bound reader
    is a normal fail-closed condition.
    """

    path = source_root / "scripts" / name
    if _has_symlink_component(path) or not path.is_file():
        raise R8Error(f"{name} source-bound harness unavailable")
    module_name = f"chronovisor_r8_source_{path.stem}_{source_root.stat().st_ino}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise R8Error(f"{name} source-bound harness unavailable")
    module = importlib.util.module_from_spec(spec)
    previous_module = sys.modules.get(module_name)
    previous_path = list(sys.path)
    previous_bytecode = sys.dont_write_bytecode
    sys.modules[module_name] = module
    sys.dont_write_bytecode = True
    source_path = str(source_root / "src")
    if source_path not in sys.path:
        sys.path.insert(0, source_path)
    try:
        spec.loader.exec_module(module)
        _assert_dependency_tree_bound(source_root)
    except Exception as exc:
        raise R8Error(f"{name} source-bound harness unavailable") from exc
    finally:
        sys.path[:] = previous_path
        sys.dont_write_bytecode = previous_bytecode
        if previous_module is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous_module
    return module


def _validate_r7_official_bundle(
    value: Mapping[str, Any], *, source_root: Path, source_commit: str
) -> Mapping[str, Any]:
    """Run the existing R7 formal validator over its source-bound bundle."""

    validator_module = _load_source_r7_harness(source_root)
    validator = getattr(validator_module, "validate_bundle", None)
    if not callable(validator):
        raise R8Error("R7 official bundle validator unavailable")
    evidence = value.get("evidence")
    if not isinstance(evidence, Mapping):
        evidence = {}
    bundle = value.get("formal_bundle", evidence.get("formal_bundle"))
    if not isinstance(bundle, Mapping):
        raise R8Error("R7 official formal bundle is missing")
    required = {
        "locked_replay",
        "stages",
        "forced_failure",
        "receipts",
        "artifacts",
        "baseline_id",
        "candidate_id",
        "lkg_id",
        "source_commit",
        "now",
    }
    if set(bundle) not in (required, required | {"source_tree_sha256"}):
        raise R8Error("R7 official formal bundle schema is not closed")
    if bundle.get("source_commit") != source_commit:
        raise R8Error("R7 official formal bundle source binding is invalid")
    now = bundle.get("now")
    if isinstance(now, str):
        try:
            now = datetime.fromisoformat(now.replace("Z", "+00:00"))
        except ValueError as exc:
            raise R8Error("R7 official formal bundle clock is invalid") from exc
    if not isinstance(now, datetime):
        raise R8Error("R7 official formal bundle clock is invalid")
    kwargs = dict(bundle)
    kwargs["now"] = now
    try:
        result = validator(**kwargs)
    except Exception as exc:
        raise R8Error(f"R7 official formal validator rejected bundle: {exc}") from exc
    if not isinstance(result, Mapping) or result.get("certification") is not True:
        raise R8Error("R7 official formal validator did not certify")
    return result


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise R8Error("value is not canonical JSON") from exc


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _embedded_content_digest(
    value: Mapping[str, Any], *, label: str
) -> str:
    """Digest the sealed object's canonical payload without its envelope.

    ``file_sha256`` on a producer artifact is an embedded payload/content
    digest.  The SHA of the complete file is supplied by the parent reference
    envelope instead; hashing the complete file here would require a
    self-referential fixed point because the field is part of the JSON bytes.
    """

    try:
        payload = {
            key: item
            for key, item in value.items()
            if key not in {"artifact_id", "file_sha256", "seal_sha256"}
        }
        return _digest(payload)
    except (TypeError, ValueError) as exc:
        raise R8Error(f"{label} content is not canonical") from exc


def _sha(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA.fullmatch(value) is None:
        raise R8Error(f"{label} is not a SHA-256")
    return value


def _int(value: object, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise R8Error(f"{label} is invalid")
    return value


def _file_state(path: Path, *, label: str) -> dict[str, int]:
    if _has_symlink_component(path):
        raise R8Error(f"{label} path contains a symlink")
    try:
        value = path.lstat()
    except FileNotFoundError:
        raise
    except OSError as exc:
        raise R8Error(f"{label} path cannot be stated") from exc
    if stat.S_ISLNK(value.st_mode):
        raise R8Error(f"{label} path contains a symlink")
    return {
        "st_dev": int(value.st_dev),
        "st_ino": int(value.st_ino),
        "st_mode": int(value.st_mode & 0o7777),
        "st_uid": int(value.st_uid),
        "st_size": int(value.st_size),
        "st_mtime_ns": int(value.st_mtime_ns),
        "st_ctime_ns": int(value.st_ctime_ns),
    }


def _has_symlink_component(path: Path) -> bool:
    current = path.expanduser().absolute()
    while True:
        if current.is_symlink():
            return True
        parent = current.parent
        if parent == current:
            return False
        current = parent


def _overlap(left: Path, right: Path) -> bool:
    a = left.expanduser().resolve(strict=False)
    b = right.expanduser().resolve(strict=False)
    return a == b or a.is_relative_to(b) or b.is_relative_to(a)


def _assert_paths(
    source_root: Path,
    r7_artifact: Path,
    output: Path,
    production_observation: Path | None = None,
    auxiliary_paths: Sequence[Path] = (),
) -> None:
    paths = [source_root, r7_artifact, output, *auxiliary_paths]
    if production_observation is not None:
        paths.append(production_observation)
    if any(_has_symlink_component(path) for path in paths):
        raise R8Error("path contains a symlink")
    source = source_root.expanduser().resolve(strict=False)
    r7_path = r7_artifact.expanduser().resolve(strict=False)
    out = output.expanduser().resolve(strict=False)
    if not source.is_dir():
        raise R8Error("source root is not a directory")
    if _overlap(source, out):
        raise R8Error("source/output roots overlap")
    if _overlap(source, r7_path):
        raise R8Error("source/R7 input roots overlap")
    if _overlap(out, r7_path):
        raise R8Error("output/R7 input roots overlap")
    for auxiliary in auxiliary_paths:
        if _overlap(source, auxiliary) or _overlap(out, auxiliary):
            raise R8Error("source/output/auxiliary input roots overlap")
    if production_observation is not None:
        observed = production_observation.expanduser().resolve(strict=False)
        if _overlap(source, observed) or _overlap(out, observed):
            raise R8Error("source/output/observation roots overlap")
    if out.exists() and not out.is_dir():
        raise R8Error("output is not a directory")
    if r7_path.suffix != ".json" or (r7_path.exists() and not r7_path.is_file()):
        raise R8Error("R7 input path is unsupported")
    if production_observation is not None and observed.exists() and not (
        observed.is_file() or observed.is_dir()
    ):
        raise R8Error("production observation path is unsupported")


def _stable_bytes(path: Path, *, label: str, maximum: int = MAX_INPUT_BYTES) -> tuple[bytes, dict[str, int]]:
    try:
        before = _file_state(path, label=label)
    except FileNotFoundError as exc:
        raise R8Error(f"{label} path is unavailable") from exc
    if not path.is_file() or before["st_size"] > maximum:
        raise R8Error(f"{label} path is unsafe or exceeds bound")
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise R8Error(f"{label} path cannot be read") from exc
    after = _file_state(path, label=label)
    if before != after or len(data) != before["st_size"]:
        raise R8Error(f"{label} changed during read")
    return data, after


def _read_json(path: Path, *, label: str) -> tuple[dict[str, Any], dict[str, int]]:
    data, state = _stable_bytes(path, label=label)
    try:
        value = json.loads(data)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise R8Error(f"{label} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise R8Error(f"{label} is not an object")
    if data not in {_canonical(value), _canonical(value) + b"\n"}:
        raise R8Error(f"{label} is not canonical")
    return value, state


def _verify_sealed(value: Mapping[str, Any], *, schema: str, label: str) -> None:
    if value.get("schema") != schema or value.get("namespace") != "recall-distillation":
        raise R8Error(f"{label} schema mismatch")
    unsigned = {key: item for key, item in value.items() if key != "seal_sha256"}
    if value.get("seal_sha256") != _digest(unsigned):
        raise R8Error(f"{label} seal mismatch")


def _read_r7_artifact(
    path: Path, *, source_root: Path, source_commit: str | None = None
) -> tuple[dict[str, Any], dict[str, int]]:
    """Read R7 through its stable reader, then bind filename, ID and seal."""

    try:
        # R7 owns the existing bounded JSON/readback contract.  Keep this call
        # separate from the semantic R8 gate so a forged summary cannot pass.
        r7 = _load_source_script(source_root, "recall_r7_harness.py")
        value = r7._read_json(path, "R7 artifact")
    except Exception as exc:
        if isinstance(exc, R8Error):
            raise
        raise R8Error(f"R7 artifact cannot be read: {exc}") from exc
    state = _file_state(path, label="R7 artifact")
    allowed_top = {
        "artifact_id",
        "schema",
        "namespace",
        "seal_sha256",
        "captured_at",
        "certification",
        "certification_reason",
        "synthetic_fixture",
        "source_before",
        "source_after",
        "source_final",
        "production_before",
        "production_after",
        "production_final",
        "cleanup",
        "thresholds",
        "stage_matrix",
        "stages",
        "evidence",
        "formal_bundle",
        "forced_failure",
        "rollback",
        "forced_rollback",
        "collector",
        "pointers",
        "live_attestation_artifact_id",
        "live_attestation_file_sha256",
        "live_attestation_seal_sha256",
        "live_attestation_source_commit",
        "live_attestation_run_id",
        "live_attestation_stage100_artifact_id",
        "live_attestation_rollback_artifact_id",
    }
    if not set(value).issubset(allowed_top):
        raise R8Error("R7 artifact schema contains unsupported fields")
    if path.suffix != ".json" or path.stem != value.get("artifact_id"):
        raise R8Error("R7 artifact filename identity mismatch")
    artifact_id = _sha(value.get("artifact_id"), "R7 artifact id")
    unsigned = {
        key: item
        for key, item in value.items()
        if key not in {"artifact_id", "seal_sha256"}
    }
    if artifact_id != _digest(unsigned):
        raise R8Error("R7 artifact identity mismatch")
    _verify_sealed(value, schema=R7_SCHEMA, label="R7 artifact")
    canonical = _canonical(value)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise R8Error("R7 artifact readback failed") from exc
    if raw not in {canonical, canonical + b"\n"}:
        raise R8Error("R7 artifact is not canonical")
    if _file_state(path, label="R7 artifact") != state:
        raise R8Error("R7 artifact changed during readback")
    if source_commit is not None:
        if not _COMMIT.fullmatch(source_commit):
            raise R8Error("source commit format is invalid")
        snapshots = [value.get(name) for name in ("source_before", "source_after")]
        for snapshot in snapshots:
            commit = _source_commit_from_snapshot(snapshot)
            if commit != source_commit:
                raise R8Error("R7/source commit binding is missing or mismatched")
            if not isinstance(snapshot, Mapping) or snapshot.get("source_clean") not in {
                True,
                "true",
            }:
                raise R8Error("R7/source checkout is not clean")
        source_final = value.get("source_final")
        if source_final is not None and (
            not isinstance(source_final, Mapping)
            or source_final != snapshots[0]
        ):
            raise R8Error("R7 source snapshots disagree")
    return value, state


def _stage_rows(stage: Mapping[str, Any]) -> int | None:
    for key in ("paired", "paired_count", "rows_count", "sample_count", "samples", "count"):
        value = stage.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    rows = stage.get("rows")
    if isinstance(rows, list):
        return len(rows)
    return None


def _stage_days(stage: Mapping[str, Any]) -> float | None:
    for key in ("days", "observed_days", "wall_days", "duration_days"):
        value = stage.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    starts = (stage.get("first_poll"), stage.get("started_at"), stage.get("stage_started_at"))
    ends = (stage.get("last_poll"), stage.get("ended_at"), stage.get("stage_ended_at"))
    for first, last in zip(starts, ends, strict=False):
        if isinstance(first, str) and isinstance(last, str):
            try:
                a = datetime.fromisoformat(first.replace("Z", "+00:00"))
                b = datetime.fromisoformat(last.replace("Z", "+00:00"))
            except ValueError:
                continue
            if a.tzinfo is not None and b.tzinfo is not None:
                return (b - a).total_seconds() / 86400
    return None


def _validate_r7_live_attestation(
    value: Mapping[str, Any],
    *,
    source_commit: str,
    source_root: Path,
    r7_artifact: Path | None,
    label: str = "R7 live attestation",
) -> tuple[str, str, str, dict[str, str]]:
    """Read and validate the independent live receipt referenced by R7.

    A nested mapping is deliberately not accepted: its bytes would be part of
    the outer R7 seal and therefore could only self-attest.  The R7 artifact
    carries an ID and file hash reference; the actual canonical sealed receipt
    is read from the fixed sibling ``r7-live-attestations/<id>.json`` path.
    """

    if r7_artifact is None:
        raise R8Error(f"{label} path is unavailable")
    if value.get("live_attestation") is not None:
        raise R8Error(f"{label} inline mapping is forbidden")
    evidence = value.get("evidence")
    if isinstance(evidence, Mapping) and evidence.get("live_attestation") is not None:
        raise R8Error(f"{label} inline mapping is forbidden")
    artifact_id = _sha(value.get("live_attestation_artifact_id"), f"{label} artifact id reference")
    expected_file_sha = _sha(value.get("live_attestation_file_sha256"), f"{label} file hash reference")
    expected_seal = _sha(value.get("live_attestation_seal_sha256"), f"{label} seal reference")
    expected_run_id = _sha(value.get("live_attestation_run_id"), f"{label} run id reference")
    live_path = r7_artifact.parent / "r7-live-attestations" / f"{artifact_id}.json"
    if _has_symlink_component(live_path):
        raise R8Error(f"{label} path contains a symlink")
    try:
        official = _load_r7_evidence(source_root)
        reference = official.validate_live_attestation(
            live_path, expected_stage="100", expected_run_id=expected_run_id
        )
    except Exception as exc:
        raise R8Error(f"{label} receipt is missing or unreadable") from exc
    if not isinstance(reference, Mapping):
        raise R8Error(f"{label} validator returned no identity")
    actual_id = _sha(reference.get("artifact_id"), f"{label} artifact id")
    actual_file_sha = _sha(reference.get("file_sha256"), f"{label} file hash")
    seal_sha = _sha(reference.get("seal_sha256"), f"{label} seal")
    if actual_id != artifact_id or actual_file_sha != expected_file_sha or seal_sha != expected_seal:
        raise R8Error(f"{label} reference identity mismatch")
    try:
        store = official.store
        schema = official.LIVE_ATTESTATION_SCHEMA
        live = store.read_sealed(live_path, schema=schema)
    except Exception as exc:
        raise R8Error(f"{label} sealed readback unavailable") from exc
    if not isinstance(live, Mapping):
        raise R8Error(f"{label} sealed readback is not an object")
    source = live.get("source")
    if not isinstance(source, Mapping) or source.get("source_commit") != source_commit:
        raise R8Error(f"{label} source binding is invalid")
    if live.get("stage") != "100" or live.get("run_id") != expected_run_id:
        raise R8Error(f"{label} stage/run binding is invalid")
    runtime = live.get("runtime")
    if not isinstance(runtime, Mapping) or runtime.get("archive_commit") != source_commit:
        raise R8Error(f"{label} runtime/archive binding is invalid")
    rollback = live.get("rollback")
    if not isinstance(rollback, Mapping) or rollback.get("status") not in {"not_triggered", "triggered", "rolled_back"}:
        raise R8Error(f"{label} rollback identity is invalid")
    rollback_artifact = rollback.get("artifact_id")
    rollback_receipt = rollback.get("receipt_sha256")
    if rollback_artifact is not None:
        _sha(rollback_artifact, f"{label} rollback artifact id")
    if rollback_receipt is not None:
        _sha(rollback_receipt, f"{label} rollback receipt")
    live_identity = {
        "source_commit": source_commit,
        "run_id": expected_run_id,
        "stage100_artifact_id": actual_id,
        "rollback_artifact_id": str(rollback_artifact) if rollback_artifact is not None else "",
        "collector_artifact_id": actual_id,
        "collector_file_sha256": actual_file_sha,
        "collector_seal_sha256": seal_sha,
        "rollback_file_sha256": str(rollback_receipt) if rollback_receipt is not None else "",
        "rollback_seal_sha256": str(rollback_receipt) if rollback_receipt is not None else "",
    }
    return actual_id, actual_file_sha, seal_sha, live_identity


def _validate_r7_summary(
    value: Mapping[str, Any],
    *,
    source_commit: str,
    source_root: Path,
    file_sha256: str | None = None,
    r7_artifact: Path | None = None,
) -> dict[str, Any]:
    if value.get("certification") is not True:
        raise R8Error("R7 certification is not true")
    evidence = value.get("evidence")
    if not isinstance(evidence, Mapping):
        evidence = {}
    if evidence.get("certification") is False:
        raise R8Error("R7 evidence certification is false")
    if value.get("synthetic_fixture") is True or evidence.get("synthetic_fixture") is True:
        raise R8Error("synthetic R7 evidence cannot authorize cleanup")
    official_bundle = _validate_r7_official_bundle(
        value, source_root=source_root, source_commit=source_commit
    )
    if official_bundle.get("certification") is not True:
        raise R8Error("R7 official bundle certification is false")
    raw_stages: object = value.get("stages", evidence.get("stages"))
    if raw_stages is None:
        raw_stages = value.get("stage_matrix", evidence.get("stage_matrix"))
    if isinstance(raw_stages, Mapping):
        stages = list(raw_stages.values())
    elif isinstance(raw_stages, list):
        stages = raw_stages
    else:
        raise R8Error("R7 stages are missing")
    if len(stages) != 4 or any(not isinstance(stage, Mapping) for stage in stages):
        raise R8Error("R7 stage set is incomplete")
    matrix = value.get("stage_matrix", evidence.get("stage_matrix"))
    if not isinstance(matrix, list) or len(matrix) != len(_R7_STAGES):
        raise R8Error("R7 ordered stage matrix is missing")
    for entry, (expected_name, expected_percent) in zip(matrix, _R7_STAGES, strict=True):
        if (
            not isinstance(entry, Mapping)
            or set(entry) != {"stage", "rollout_percent"}
            or entry.get("stage") != expected_name
            or entry.get("rollout_percent") != expected_percent
        ):
            raise R8Error("R7 stage matrix order or rollout is invalid")
    checked: list[dict[str, Any]] = []
    run_ids: set[str] = set()
    for index, stage in enumerate(stages):
        item = dict(stage)
        expected_name, expected_percent = _R7_STAGES[index]
        allowed_stage_keys = {
            "stage",
            "rollout_percent",
            "days",
            "observed_days",
            "wall_days",
            "duration_days",
            "paired",
            "paired_count",
            "rows_count",
            "sample_count",
            "samples",
            "count",
            "rows",
            "certified",
            "run_id",
            "stage_seal_sha256",
            "last_poll_seal_sha256",
            "artifact_id",
            "stage_artifact_id",
            "file_sha256",
            "stage_file_sha256",
            "artifact_seal_sha256",
            "seal_sha256",
            "poll_artifact_id",
            "last_poll_artifact_id",
            "rows_sha256",
            "metrics",
            "last_poll",
            "reason",
        }
        if not set(item).issubset(allowed_stage_keys):
            raise R8Error("R7 stage schema is not closed")
        if item.get("stage") != expected_name:
            raise R8Error("R7 stage name order is invalid")
        if type(item.get("rollout_percent")) is not int or item.get("rollout_percent") != expected_percent:
            raise R8Error("R7 stage rollout is invalid")
        days = _stage_days(item)
        paired = _stage_rows(item)
        if days is None or days < MIN_R7_DAYS:
            raise R8Error("R7 stage is below seven-day wall time")
        if paired is None or paired < MIN_R7_PAIRED:
            raise R8Error("R7 stage paired denominator is below 500")
        if item.get("certified") is not True:
            raise R8Error("R7 stage is not certified")
        if not isinstance(item.get("run_id"), str) or _SHA.fullmatch(item["run_id"]) is None:
            raise R8Error("R7 stage run binding is missing")
        if item["run_id"] in run_ids:
            raise R8Error("R7 stage run IDs are reused")
        run_ids.add(item["run_id"])
        if not isinstance(item.get("stage_seal_sha256"), str) or _SHA.fullmatch(item["stage_seal_sha256"]) is None:
            raise R8Error("R7 stage seal binding is missing")
        if not isinstance(item.get("last_poll_seal_sha256"), str) or _SHA.fullmatch(item["last_poll_seal_sha256"]) is None:
            raise R8Error("R7 stage poll binding is missing")
        stage_artifact_id = item.get("artifact_id", item.get("stage_artifact_id"))
        stage_file_sha = item.get("file_sha256", item.get("stage_file_sha256"))
        stage_artifact_seal = item.get("artifact_seal_sha256", item.get("seal_sha256"))
        poll_artifact_id = item.get("poll_artifact_id", item.get("last_poll_artifact_id"))
        for candidate, label in (
            (stage_artifact_id, "R7 stage artifact id"),
            (stage_file_sha, "R7 stage file hash"),
            (stage_artifact_seal, "R7 stage artifact seal"),
            (poll_artifact_id, "R7 stage poll artifact id"),
        ):
            _sha(candidate, label)
        if stage_artifact_seal != item["stage_seal_sha256"]:
            raise R8Error("R7 stage receipt/artifact seal binding is mismatched")
        checked.append(
            {
                "stage": expected_name,
                "rollout_percent": expected_percent,
                "days": days,
                "paired": paired,
                "certified": True,
                "run_id": item["run_id"],
                "stage_seal_sha256": item["stage_seal_sha256"],
                "last_poll_seal_sha256": item["last_poll_seal_sha256"],
                "artifact_id": stage_artifact_id,
                "file_sha256": stage_file_sha,
                "artifact_seal_sha256": stage_artifact_seal,
                "poll_artifact_id": poll_artifact_id,
            }
        )
    forced: object = value.get("forced_failure", evidence.get("forced_failure"))
    if forced is None:
        forced = value.get("rollback", evidence.get("rollback"))
    if forced is None:
        forced = value.get("forced_rollback", evidence.get("forced_rollback"))
    if forced is True:
        raise R8Error("R7 forced rollback boolean is not a receipt")
    if not isinstance(forced, Mapping):
        raise R8Error("R7 forced rollback receipt is missing")
    if any(forced.get(key) is not True for key in ("deterministic_failure", "rolled_back", "learning_halted")):
        raise R8Error("R7 forced rollback is incomplete")
    rollout_percent = forced.get("rollout_percent")
    if (
        isinstance(rollout_percent, bool)
        or not isinstance(rollout_percent, (int, float))
        or rollout_percent != 0
    ):
        raise R8Error("R7 forced rollback did not reach zero")
    if forced.get("kind") not in {None, "forced-failure-receipt"}:
        raise R8Error("R7 forced rollback kind is invalid")
    if forced.get("stage") != "100" or forced.get("run_id") != checked[-1]["run_id"]:
        raise R8Error("R7 forced rollback stage binding is missing")
    if forced.get("source_commit") != source_commit:
        raise R8Error("R7 forced rollback/source binding is missing")
    for key in ("stage_seal_sha256", "last_poll_seal_sha256"):
        if not isinstance(forced.get(key), str) or _SHA.fullmatch(forced[key]) is None:
            raise R8Error("R7 forced rollback poll binding is missing")
    if (
        forced.get("stage_seal_sha256") != checked[-1]["stage_seal_sha256"]
        or forced.get("last_poll_seal_sha256") != checked[-1]["last_poll_seal_sha256"]
    ):
        raise R8Error("R7 forced rollback stage/poll seal binding is mismatched")
    for key in ("poll_artifact_id", "process_artifact_id", "archive_artifact_id", "final_stage_artifact_id"):
        if not isinstance(forced.get(key), str) or _SHA.fullmatch(forced[key]) is None:
            raise R8Error("R7 forced rollback evidence chain is incomplete")
    if (
        forced.get("final_stage_artifact_id") != checked[-1]["artifact_id"]
        or forced.get("poll_artifact_id") != checked[-1]["poll_artifact_id"]
    ):
        raise R8Error("R7 forced rollback final-stage artifact binding is mismatched")
    final_file_sha = forced.get("final_stage_file_sha256", forced.get("final_stage_artifact_sha256"))
    final_seal = forced.get("final_stage_seal_sha256", forced.get("final_stage_artifact_seal_sha256"))
    if final_file_sha != checked[-1]["file_sha256"] or final_seal != checked[-1]["artifact_seal_sha256"]:
        raise R8Error("R7 forced rollback final-stage file binding is mismatched")
    if r7_artifact is None:
        raise R8Error("R7 final-stage receipt path is unavailable")
    _r7_content_ref(
        r7_artifact,
        kind="stage",
        artifact_id=checked[-1]["artifact_id"],
        file_sha256=checked[-1]["file_sha256"],
        seal_sha256=checked[-1]["artifact_seal_sha256"],
    )
    for kind in ("poll", "process", "archive"):
        run_id = forced.get(f"{kind}_run_id", forced.get("run_id"))
        if run_id != checked[-1]["run_id"]:
            raise R8Error(f"R7 {kind} receipt run binding is mismatched")
        _r7_content_ref(
            r7_artifact,
            kind=kind,
            artifact_id=forced.get(f"{kind}_artifact_id"),
            file_sha256=forced.get(f"{kind}_file_sha256"),
            seal_sha256=forced.get(f"{kind}_seal_sha256"),
        )
    if forced.get("poll_seal_sha256") != checked[-1]["last_poll_seal_sha256"]:
        raise R8Error("R7 forced rollback poll receipt seal is mismatched")
    state = forced.get("rollback_state")
    if not isinstance(state, Mapping):
        state = value.get("pointers", evidence.get("pointers"))
    if not isinstance(state, Mapping):
        raise R8Error("R7 active/LKG pointer state is missing")
    active = state.get("active_policy_id", state.get("active"))
    lkg = state.get("lkg_policy_id", state.get("lkg"))
    candidate_present = "candidate_policy_id" in state or "candidate" in state
    candidate = state.get("candidate_policy_id", state.get("candidate"))
    if set(state) not in {
        {
            "active_policy_id",
            "candidate_policy_id",
            "lkg_policy_id",
            "active_policy_file_sha256",
            "active_policy_seal_sha256",
            "lkg_policy_file_sha256",
            "lkg_policy_seal_sha256",
        },
        {
            "active",
            "candidate",
            "lkg",
            "active_file_sha256",
            "active_seal_sha256",
            "lkg_file_sha256",
            "lkg_seal_sha256",
        },
    }:
        raise R8Error("R7 rollback pointer schema is not closed")
    if (
        not isinstance(active, str)
        or not isinstance(lkg, str)
        or _SHA.fullmatch(active) is None
        or _SHA.fullmatch(lkg) is None
        or not candidate_present
        or candidate is not None
        or active != lkg
    ):
        raise R8Error("R7 active pointer is not LKG")
    for role, pointer_id in (("active", active), ("lkg", lkg)):
        _r7_content_ref(
            r7_artifact,
            kind="policy",
            artifact_id=pointer_id,
            file_sha256=state.get(f"{role}_policy_file_sha256", state.get(f"{role}_file_sha256")),
            seal_sha256=state.get(f"{role}_policy_seal_sha256", state.get(f"{role}_seal_sha256")),
        )
    live_id, live_file_sha, live_seal, live_identity = _validate_r7_live_attestation(
        value,
        source_commit=source_commit,
        source_root=source_root,
        r7_artifact=r7_artifact,
    )
    live_refs = {
        "live_attestation_artifact_id": live_id,
        "live_attestation_file_sha256": live_file_sha,
        "live_attestation_seal_sha256": live_seal,
        "live_attestation_source_commit": live_identity["source_commit"],
        "live_attestation_run_id": live_identity["run_id"],
        "live_attestation_stage100_artifact_id": live_identity["stage100_artifact_id"],
        "live_attestation_rollback_artifact_id": live_identity["rollback_artifact_id"],
    }
    for field, expected in live_refs.items():
        if value.get(field) != expected:
            raise R8Error(f"R7 top {field} cross-binding is invalid")
    collector = value.get("collector", evidence.get("collector"))
    if not isinstance(collector, Mapping):
        raise R8Error("R7 collector identity is missing")
    for field, expected in {
        **live_refs,
        "artifact_id": live_identity["collector_artifact_id"],
        "file_sha256": live_identity["collector_file_sha256"],
        "seal_sha256": live_identity["collector_seal_sha256"],
        "source_commit": source_commit,
        "run_id": live_identity["run_id"],
    }.items():
        actual = collector.get(field)
        if field in {"artifact_id", "file_sha256", "seal_sha256"}:
            _sha(actual, f"R7 collector {field}")
        if actual != expected:
            raise R8Error(f"R7 collector {field} cross-binding is invalid")
    for field, expected in {
        **live_refs,
        "rollback_artifact_id": live_identity["rollback_artifact_id"],
        "rollback_file_sha256": live_identity["rollback_file_sha256"],
        "rollback_seal_sha256": live_identity["rollback_seal_sha256"],
        "source_commit": source_commit,
        "run_id": checked[-1]["run_id"],
    }.items():
        actual = forced.get(field)
        if field.endswith(("_artifact_id", "_file_sha256", "_seal_sha256", "_run_id")):
            _sha(actual, f"R7 rollback {field}")
        if actual != expected:
            raise R8Error(f"R7 rollback {field} cross-binding is invalid")
    return {
        "artifact_id": _sha(value.get("artifact_id"), "R7 artifact id"),
        "file_sha256": file_sha256,
        "certification": True,
        "stages": checked,
        "forced_rollback": True,
        "active_lkg": True,
        "source_commit": source_commit,
        "live_attestation": True,
        "live_attestation_id": live_id,
        "live_attestation_sha256": live_file_sha,
        "live_attestation_seal_sha256": live_seal,
        "live_attestation_source_commit": live_identity["source_commit"],
        "live_attestation_run_id": live_identity["run_id"],
        "live_attestation_stage100_artifact_id": live_identity["stage100_artifact_id"],
        "live_attestation_rollback_artifact_id": live_identity["rollback_artifact_id"],
        "collector_artifact_id": live_identity["collector_artifact_id"],
        "collector_file_sha256": live_identity["collector_file_sha256"],
        "collector_seal_sha256": live_identity["collector_seal_sha256"],
        "rollback_artifact_id": live_identity["rollback_artifact_id"],
        "rollback_file_sha256": live_identity["rollback_file_sha256"],
        "rollback_seal_sha256": live_identity["rollback_seal_sha256"],
    }


def _safe_file_state(value: object) -> dict[str, int] | None:
    if not isinstance(value, Mapping):
        return None
    keys = {"st_dev", "st_ino", "st_mode", "st_uid", "st_size", "st_mtime_ns", "st_ctime_ns"}
    if set(value) != keys:
        return None
    if any(isinstance(value[key], bool) or not isinstance(value[key], int) or value[key] < 0 for key in keys):
        return None
    return {key: int(value[key]) for key in keys}


def _safe_metadata(value: object, *, label: str) -> dict[str, Any]:
    """Keep only bounded inventory metadata; never copy payload/body fields."""

    if not isinstance(value, Mapping):
        raise R8Error(f"{label} inventory is not an object")
    result: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise R8Error(f"{label} inventory key is invalid")
        if _SENSITIVE.search(key) and not key.endswith(("_sha256", "_digest")):
            raise R8Error(f"{label} contains a sensitive field")
        if key in {"file_state", "before", "after", "pointer_file_state", "policy_file_state"}:
            if key.endswith("state") or key in {"file_state", "pointer_file_state", "policy_file_state"}:
                # A missing inventory is represented by ``file_state: null``.
                # Preserve that closed shape instead of dropping it or
                # rejecting the bounded absence marker.
                if item is None:
                    result[key] = None
                    continue
                state = _safe_file_state(item)
                if state is None:
                    raise R8Error(f"{label}.{key} file state is invalid")
                result[key] = state
            continue
        if key == "state":
            if not isinstance(item, str) or len(item) > 64:
                raise R8Error(f"{label}.state is invalid")
            result[key] = item
            continue
        if isinstance(item, Mapping):
            nested = _safe_metadata(item, label=f"{label}.{key}")
            if nested or key == "sidecars":
                result[key] = nested
            continue
        if key in {"present", "exists", "enabled", "process_lock", "distillation_process_lock", "read_only", "sealed", "bounded"}:
            if not isinstance(item, bool):
                raise R8Error(f"{label}.{key} is invalid")
            result[key] = item
        elif key in {
            "count",
            "records",
            "bytes",
            "provider_calls",
            "leased",
            "process_count",
            "row_count",
            "size_bytes",
        }:
            result[key] = _int(item, f"{label}.{key}")
        elif key in {"sha256", "content_sha256", "inventory_sha256", "head_sha256", "seal_sha256", "artifact_id", "payload_digest", "completion_digest"}:
            if item is None and key in {"sha256", "content_sha256"}:
                result[key] = None
            else:
                result[key] = _sha(item, f"{label}.{key}")
        elif key in {"status", "kind", "schema", "namespace", "relative_path", "read_mode", "source"}:
            if not isinstance(item, str) or len(item) > 128:
                raise R8Error(f"{label}.{key} is invalid")
            result[key] = item
        elif key in {"files", "entries", "children"}:
            if not isinstance(item, list) or len(item) > MAX_FILES:
                raise R8Error(f"{label}.{key} is invalid")
            result[key] = {"count": len(item)}
        elif key == "sidecars":
            if not isinstance(item, Mapping):
                raise R8Error(f"{label}.sidecars is invalid")
            result[key] = _safe_metadata(item, label=f"{label}.sidecars")
    return result


def _empty_inventory() -> dict[str, Any]:
    return {
        "present": False,
        "count": 0,
        "bytes": 0,
        "sha256": None,
        "file_state": None,
        "sidecars": {},
    }


def _sidecar_inventory(path: Path, *, label: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for suffix in ("-wal", "-shm", "-journal"):
        sidecar = path.with_name(path.name + suffix)
        try:
            state = _file_state(sidecar, label=f"{label} sidecar {suffix}")
        except FileNotFoundError:
            result[suffix] = _empty_inventory()
            continue
        if not sidecar.is_file():
            raise R8Error(f"{label} sidecar is not a regular file")
        result[suffix] = {
            "present": True,
            "count": 1,
            "bytes": state["st_size"],
            "sha256": _hash_file(sidecar, state, label=f"{label} sidecar {suffix}"),
            "file_state": state,
            "bounded": state["st_size"] <= MAX_HASH_BYTES,
        }
    return result


def _hash_file(path: Path, state: Mapping[str, int], *, label: str) -> str | None:
    if state["st_size"] > MAX_HASH_BYTES:
        return None
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    except OSError as exc:
        raise R8Error(f"{label} cannot be hashed") from exc
    if _file_state(path, label=label) != dict(state):
        raise R8Error(f"{label} changed during hash")
    return digest.hexdigest()


def _path_inventory(root: Path, relative: str, *, label: str) -> dict[str, Any]:
    path = root / relative
    try:
        state = _file_state(path, label=label)
    except FileNotFoundError:
        return _empty_inventory()
    if path.is_file():
        file_digest = _hash_file(path, state, label=label)
        return {
            "present": True,
            "count": 1,
            "bytes": state["st_size"],
            "sha256": file_digest,
            "file_state": state,
            "sidecars": _sidecar_inventory(path, label=label),
            "bounded": file_digest is not None,
        }
    if not path.is_dir():
        raise R8Error(f"{label} is not a file or directory")
    directory_digest = hashlib.sha256()
    count = total = 0
    states: list[dict[str, int]] = []
    stack = [path]
    while stack:
        current = stack.pop()
        try:
            children = sorted(current.iterdir(), key=lambda item: item.name)
        except OSError as exc:
            raise R8Error(f"{label} directory cannot be read") from exc
        for child in children:
            if _has_symlink_component(child):
                raise R8Error(f"{label} contains a symlink")
            child_state = _file_state(child, label=label)
            if child.is_dir():
                stack.append(child)
                continue
            if not child.is_file():
                continue
            count += 1
            if count > MAX_FILES:
                raise R8Error(f"{label} exceeds bounded file count")
            total += child_state["st_size"]
            child_digest = _hash_file(child, child_state, label=label)
            directory_digest.update(
                _canonical((child.relative_to(root).as_posix(), child_digest, child_state))
            )
            states.append(child_state)
    return {
        "present": True,
        "count": count,
        "bytes": total,
        "sha256": directory_digest.hexdigest() if count else _digest([]),
        "file_state": state,
        "sidecars": {},
        "bounded": all(item["st_size"] <= MAX_HASH_BYTES for item in states),
    }


def _observe_directory(root: Path) -> dict[str, Any]:
    if _has_symlink_component(root) or not root.is_dir():
        raise R8Error("production observation root is unsafe")
    ox = {name: _path_inventory(root, relative, label=f"OX {name}") for name, relative in _OX_FILES.items()}
    pointers = {name: _path_inventory(root, relative, label=f"pointer {name}") for name, relative in _POINTER_FILES.items()}
    legacy: dict[str, Any] = {}
    for name, relatives in _LEGACY_FILES.items():
        entries = [_path_inventory(root, relative, label=f"legacy {name}") for relative in relatives]
        present = [entry for entry in entries if entry["present"]]
        legacy[name] = present[0] if len(present) == 1 else {
            "present": bool(present),
            "count": sum(int(entry["count"]) for entry in present),
            "bytes": sum(int(entry["bytes"]) for entry in present),
            "sha256": _digest(entries),
            "file_state": None,
            "bounded": all(entry.get("bounded", False) for entry in present),
        }
    locks = {relative: _path_inventory(root, relative, label="distillation process lock") for relative in _PROCESS_LOCKS}
    leased = 0
    workset_path = root / _OX_FILES["workset"]
    if workset_path.is_file() and not _has_symlink_component(workset_path):
        workset_before = ox["workset"]
        try:
            with sqlite3.connect(f"file:{workset_path}?mode=ro", uri=True) as connection:
                row = connection.execute(
                    "SELECT COUNT(*) FROM work_items WHERE state = 'leased'"
                ).fetchone()
                leased = int(row[0]) if row else 0
        except sqlite3.Error as exc:
            raise R8Error("OX Workset leased-state read failed") from exc
        workset_after = _path_inventory(
            root, _OX_FILES["workset"], label="OX workset"
        )
        if workset_after != workset_before:
            raise R8Error("OX Workset changed during leased-state read")
    process_count = 1 if any(item["present"] for item in locks.values()) else 0
    return {
        "ox": {
            "enabled": False,
            "provider_calls": 0,
            "leased": leased,
            "process_lock": any(item["present"] for item in locks.values()),
            "process_count": process_count,
            "workset": ox["workset"],
            "contract": ox["contract"],
            "event": ox["event"],
            "label": ox["label"],
            "candidate_lineage": ox["candidate_lineage"],
        },
        "pointers": pointers,
        "legacy": legacy,
        "locks": locks,
        "phase_receipts": {},
    }


def _observe_file(path: Path) -> tuple[dict[str, Any], dict[str, int]]:
    value, state = _read_json(path, label="production observation")
    if value.get("schema") is not None:
        _verify_sealed(value, schema=str(value.get("schema")), label="production observation")
    # A fixed observation may be a sealed envelope or a plain bounded mapping.
    body = value.get("production", value)
    if not isinstance(body, Mapping):
        raise R8Error("production observation body is not an object")
    # Preserve only the bounded sections; unknown fields (including payloads)
    # are deliberately not copied into the artifact.
    result: dict[str, Any] = {}
    for section in ("ox", "pointers", "legacy", "locks", "phase_receipts"):
        if section in body:
            raw = body[section]
            if section == "phase_receipts":
                if not isinstance(raw, Mapping):
                    raise R8Error("phase receipts are not an object")
                result[section] = dict(raw)
            elif isinstance(raw, Mapping):
                result[section] = _safe_metadata(raw, label=f"production {section}")
            else:
                raise R8Error(f"production {section} is not an object")
    return result, state


def _fixed_production_root(source_root: Path) -> Path:
    r4 = _load_source_script(source_root, "recall_r4_harness.py")
    root = getattr(r4, "PRODUCTION_ROOT", None)
    if not isinstance(root, Path) or not root.is_absolute() or _has_symlink_component(root):
        raise R8Error("fixed production root is unavailable")
    return root


def _read_observation(
    path: Path | None, *, source_root: Path | None = None
) -> tuple[dict[str, Any], dict[str, int] | None]:
    if path is None:
        return {}, None
    if path.is_dir():
        observed = _observe_directory(path)
        observed["_sealed"] = False
        return observed, _file_state(path, label="production observation root")
    if not path.is_file() or path.suffix != ".json":
        raise R8Error("production observation is not a supported sealed file")
    value, state = _read_json(path, label="production observation")
    expected = {
        "artifact_id",
        "file_sha256",
        "schema",
        "namespace",
        "kind",
        "source_commit",
        "source_before",
        "source_after",
        "source_final",
        "production",
        "production_root",
        "production_root_state",
        "production_before",
        "production_after",
        "production_final",
        "seal_sha256",
    }
    if set(value) != expected:
        raise R8Error("production observation schema is not closed")
    _verify_sealed(value, schema=R8_OBSERVATION_SCHEMA, label="production observation")
    artifact_id = _sha(value.get("artifact_id"), "production observation id")
    if path.stem != artifact_id:
        raise R8Error("production observation filename identity mismatch")
    unsigned = {key: item for key, item in value.items() if key not in {"artifact_id", "seal_sha256"}}
    if _digest(unsigned) != artifact_id:
        raise R8Error("production observation content identity mismatch")
    declared_content_sha = _sha(
        value.get("file_sha256"), "production observation content hash"
    )
    if declared_content_sha != _embedded_content_digest(
        value, label="production observation"
    ):
        raise R8Error("production observation content hash does not match payload")
    if value.get("kind") != "r8-production-read-only-observation":
        raise R8Error("production observation kind mismatch")
    source_commit = value.get("source_commit")
    if not isinstance(source_commit, str) or _COMMIT.fullmatch(source_commit) is None:
        raise R8Error("production observation source commit is invalid")
    snapshots = [value.get(key) for key in ("source_before", "source_after", "source_final")]
    if any(not isinstance(snapshot, Mapping) for snapshot in snapshots):
        raise R8Error("production observation source snapshots are missing")
    if not (snapshots[0] == snapshots[1] == snapshots[2]):
        raise R8Error("production observation source snapshots differ")
    for snapshot in snapshots:
        if not isinstance(snapshot, Mapping):
            raise R8Error("production observation source snapshot is invalid")
        if snapshot.get("source_commit") != source_commit:
            raise R8Error("production observation/source commit binding is invalid")
        if _safe_file_state(snapshot.get("file_state")) is None:
            raise R8Error("production observation source file state is missing")
    body = value.get("production")
    if not isinstance(body, Mapping):
        raise R8Error("production observation body is not an object")
    expected_sections = {"ox", "pointers", "legacy", "locks", "phase_receipts"}
    if set(body) != expected_sections:
        raise R8Error("production observation sections are not closed")
    if not isinstance(body.get("ox"), Mapping) or set(body["ox"]) != {
        "enabled", "provider_calls", "leased", "process_lock", "process_count", *tuple(_OX_FILES)
    }:
        raise R8Error("production OX observation schema is not closed")
    if not isinstance(body.get("pointers"), Mapping) or set(body["pointers"]) != set(_POINTER_FILES):
        raise R8Error("production pointer observation schema is not closed")
    if not isinstance(body.get("legacy"), Mapping) or set(body["legacy"]) != set(_LEGACY_FILES):
        raise R8Error("production legacy observation schema is not closed")
    if not isinstance(body.get("locks"), Mapping) or set(body["locks"]) != set(_PROCESS_LOCKS):
        raise R8Error("production lock observation schema is not closed")
    if not isinstance(body.get("phase_receipts"), Mapping) or set(body["phase_receipts"]) != set(_R8_PHASE_NAMES):
        raise R8Error("production phase receipt schema is not closed")
    if source_root is None:
        raise R8Error("source-bound production observation root is unavailable")
    fixed_root = _fixed_production_root(source_root)
    if value.get("production_root") != str(fixed_root):
        raise R8Error("production observation root is not the fixed root")
    root_state = _file_state(fixed_root, label="fixed production root")
    if value.get("production_root_state") != root_state:
        raise R8Error("production observation root identity is stale")
    actual_inventory = _observe_directory(fixed_root)
    for name in ("before", "after", "final"):
        snapshot = value.get(f"production_{name}")
        if not isinstance(snapshot, Mapping) or dict(snapshot) != actual_inventory:
            raise R8Error("production observation inventory is not an actual fixed-root readback")
    # The body is a projection of the same fixed-root readback.  Comparing it
    # section-by-section prevents a sealed envelope from claiming
    # ``provider_calls=0`` or clean pointers while the actual root differs.
    for name in ("ox", "pointers", "legacy", "locks"):
        section = body.get(name)
        actual = actual_inventory.get(name)
        if not isinstance(section, Mapping) or section != actual:
            raise R8Error(f"production observation {name} is not actual fixed-root inventory")
    phase_receipts = body.get("phase_receipts")
    if not isinstance(phase_receipts, Mapping):
        raise R8Error("production phase receipts are not an object")
    for phase in _R8_PHASE_NAMES:
        reference = phase_receipts.get(phase)
        if not isinstance(reference, Mapping) or set(reference) != _PHASE_REF_KEYS:
            raise R8Error("production phase receipt reference is not closed")
        if not isinstance(reference.get("path"), str) or not reference["path"].startswith("/"):
            raise R8Error("production phase receipt path is invalid")
    result: dict[str, Any] = {}
    for section in ("ox", "pointers", "legacy", "locks", "phase_receipts"):
        if section not in body:
            continue
        raw = body[section]
        if section == "phase_receipts":
            if not isinstance(raw, Mapping):
                raise R8Error("phase receipts are not an object")
            result[section] = dict(raw)
        elif isinstance(raw, Mapping):
            result[section] = _safe_metadata(raw, label=f"production {section}")
        else:
            raise R8Error(f"production {section} is not an object")
    result["source_snapshots"] = {
        name: dict(snapshot)
        for name, snapshot in zip(("before", "after", "final"), snapshots, strict=True)
        if isinstance(snapshot, Mapping)
    }
    result["_sealed"] = True
    return result, state


def _mapping_bool(value: Mapping[str, Any], keys: Sequence[str], *, label: str) -> bool | None:
    for key in keys:
        if key in value:
            item = value[key]
            if not isinstance(item, bool):
                raise R8Error(f"{label}.{key} is invalid")
            return item
    return None


def _mapping_int(value: Mapping[str, Any], keys: Sequence[str], *, label: str) -> int | None:
    for key in keys:
        if key in value:
            return _int(value[key], f"{label}.{key}")
    return None


def _source_commit_from_snapshot(value: object) -> str | None:
    if not isinstance(value, Mapping):
        return None
    commit = value.get("source_commit", value.get("commit", value.get("runtime_commit")))
    return commit if isinstance(commit, str) else None


def _artifact_source_commit(artifact: Mapping[str, Any]) -> str | None:
    """Extract an explicitly source-bound commit from a known receipt shape.

    R2 records it under ``runtime_identity`` while R3 records it under
    ``runtime``.  R4 uses the three source snapshots.  Do not infer a commit
    from arbitrary payload text: an absent binding must fail closed.
    """

    for key in ("source_commit", "runtime_identity", "runtime"):
        value = artifact.get(key)
        commit = _source_commit_from_snapshot(value)
        if commit is not None:
            return commit
    source = artifact.get("source")
    if isinstance(source, Mapping):
        commit = _source_commit_from_snapshot(source)
        if commit is not None:
            return commit
        after = source.get("after")
        commit = _source_commit_from_snapshot(after)
        if commit is not None:
            return commit
    return None


def _bind_artifact_identity(
    path: Path, artifact: Mapping[str, Any], *, label: str
) -> tuple[str, str, str]:
    if path.suffix != ".json":
        raise R8Error(f"{label} filename suffix is invalid")
    artifact_id = _sha(artifact.get("artifact_id"), f"{label} id")
    if path.stem != artifact_id:
        raise R8Error(f"{label} filename identity mismatch")
    unsigned = {
        key: item
        for key, item in artifact.items()
        if key not in {"artifact_id", "seal_sha256"}
    }
    if _digest(unsigned) != artifact_id:
        raise R8Error(f"{label} content identity mismatch")
    seal = _sha(artifact.get("seal_sha256"), f"{label} seal")
    if _digest({key: item for key, item in artifact.items() if key != "seal_sha256"}) != seal:
        raise R8Error(f"{label} seal is invalid")
    state = _file_state(path, label=label)
    file_sha = _hash_file(path, state, label=label)
    if file_sha is None:
        raise R8Error(f"{label} file hash is unavailable")
    return artifact_id, file_sha, seal


def _receipt_file_identity(path: Path, *, label: str) -> str:
    """Return the stable file hash for a sealed receipt read by an existing reader."""

    state = _file_state(path, label=label)
    digest = _hash_file(path, state, label=label)
    if digest is None:
        raise R8Error(f"{label} file hash is unavailable")
    return digest


def _bind_completion_identity(
    path: Path, artifact: Mapping[str, Any], *, label: str
) -> tuple[str, str, str]:
    if path.suffix != ".json":
        raise R8Error(f"{label} filename suffix is invalid")
    artifact_id = _sha(artifact.get("artifact_id"), f"{label} id")
    if path.stem != artifact_id:
        raise R8Error(f"{label} filename identity mismatch")
    unsigned = {
        key: item for key, item in artifact.items() if key not in {"artifact_id", "seal_sha256"}
    }
    if _digest(unsigned) != artifact_id:
        raise R8Error(f"{label} content identity mismatch")
    seal = _sha(artifact.get("seal_sha256"), f"{label} seal")
    if _digest({key: item for key, item in artifact.items() if key != "seal_sha256"}) != seal:
        raise R8Error(f"{label} seal is invalid")
    return artifact_id, _receipt_file_identity(path, label=label), seal


def _r7_content_ref(
    r7_artifact: Path,
    *,
    kind: str,
    artifact_id: object,
    file_sha256: object,
    seal_sha256: object,
) -> tuple[Path, str, str]:
    """Dereference a content-addressed R7 receipt from its fixed namespace."""

    ref_id = _sha(artifact_id, f"R7 {kind} artifact id")
    expected_file_sha = _sha(file_sha256, f"R7 {kind} file hash")
    expected_seal = _sha(seal_sha256, f"R7 {kind} seal")
    directory = r7_artifact.parent / f"r7-{kind}s"
    path = directory / f"{ref_id}.json"
    if _has_symlink_component(path) or not path.is_file() or path.suffix != ".json":
        raise R8Error(f"R7 {kind} receipt is unavailable")
    value, state = _read_json(path, label=f"R7 {kind} receipt")
    actual_id, actual_file_sha, actual_seal = _bind_artifact_identity(
        path, value, label=f"R7 {kind} receipt"
    )
    if actual_id != ref_id or actual_file_sha != expected_file_sha or actual_seal != expected_seal:
        raise R8Error(f"R7 {kind} receipt identity mismatch")
    if _file_state(path, label=f"R7 {kind} receipt") != state:
        raise R8Error(f"R7 {kind} receipt changed during readback")
    return path, actual_file_sha, actual_seal


def _same_source_commit(artifact: Mapping[str, Any], source_commit: str, *, label: str) -> None:
    actual = _artifact_source_commit(artifact)
    if actual != source_commit:
        raise R8Error(f"{label}/source commit binding is missing or mismatched")


def _r2_parity_projection(value: object, *, label: str) -> tuple[Any, ...]:
    """Check R2's full-rebuild parity projection without copying its contents."""

    if isinstance(value, bool):
        raise R8Error(f"{label} compact boolean is not accepted")
    if not isinstance(value, Mapping):
        raise R8Error(f"{label} is missing")
    if value.get("passed") is False or value.get("verified") is False:
        raise R8Error(f"{label} is false")
    if value.get("passed") is True and not any(
        key in value for key in ("catalog", "fts", "inventory")
    ):
        raise R8Error(f"{label} compact boolean is not accepted")
    projection: list[Any] = []
    for name in ("catalog", "fts"):
        section = value.get(name)
        if not isinstance(section, Mapping) or section.get("exists") is not True:
            raise R8Error(f"{label}.{name} is incomplete")
        duplicates = section.get("duplicates")
        if isinstance(duplicates, Mapping):
            if any(
                isinstance(item, bool) or not isinstance(item, int) or item != 0
                for item in duplicates.values()
            ):
                raise R8Error(f"{label}.{name} contains duplicates")
            duplicate_projection: object = tuple(sorted(duplicates.items()))
        elif isinstance(duplicates, bool) or not isinstance(duplicates, int) or duplicates != 0:
            raise R8Error(f"{label}.{name} contains duplicates")
        else:
            duplicate_projection = 0
        digest = section.get("digest")
        digest = _sha(digest, f"{label}.{name}.digest")
        rows = section.get("rows")
        if not isinstance(rows, Mapping) or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in rows.values()
        ):
            raise R8Error(f"{label}.{name}.rows is incomplete")
        projection.append((name, digest, tuple(sorted(rows.items())), duplicate_projection))
    inventory = value.get("inventory")
    if not isinstance(inventory, Mapping):
        raise R8Error(f"{label}.inventory is incomplete")
    count = inventory.get("count")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise R8Error(f"{label}.inventory.count is invalid")
    ids_sha = _sha(inventory.get("ids_sha256"), f"{label}.inventory.ids_sha256")
    statuses = inventory.get("status_counts")
    if not isinstance(statuses, Mapping) or any(
        isinstance(item, bool) or not isinstance(item, int) or item < 0
        for item in statuses.values()
    ):
        raise R8Error(f"{label}.inventory.status_counts is incomplete")
    projection.append(("inventory", count, ids_sha, tuple(sorted(statuses.items()))))
    fts_digest = _sha(value.get("fts_digest"), f"{label}.fts_digest")
    projection.append(("fts_digest", fts_digest))
    return tuple(projection)


def _r2_production_unchanged(artifact: Mapping[str, Any]) -> bool:
    explicit = artifact.get("production_unchanged")
    if explicit is not None:
        # A caller-provided boolean is not a byte-state receipt.  Require the
        # formal before/after/static snapshots below even when the convenience
        # field is present.
        return False
    production = artifact.get("production")
    if not isinstance(production, Mapping):
        return False
    source_before = production.get("source_tree_before")
    source_after = production.get("source_tree_after")
    static_before = production.get("before_static")
    static_after = production.get("after_static")
    if not all(
        isinstance(before, Mapping)
        and isinstance(after, Mapping)
        and before == after
        for before, after in (
            (source_before, source_after),
            (static_before, static_after),
        )
    ):
        return False
    # Official R2 records the immutable Raw tree digest once and a bounded
    # post-delta inventory validation projection.  Older bespoke
    # raw/derived_before fields remain accepted only when their pairs agree.
    raw_before = production.get("raw_tree_before")
    raw_after = production.get("raw_tree_after")
    if raw_before is not None or raw_after is not None:
        if raw_before != raw_after:
            return False
    elif not isinstance(production.get("raw_tree"), Mapping):
        return False
    raw_validation = artifact.get("raw_inventory_validation")
    if raw_validation is not None and not isinstance(raw_validation, Mapping):
        return False
    derived_before = production.get("derived_tree_before")
    derived_after = production.get("derived_tree_after")
    if derived_before is not None or derived_after is not None:
        if derived_before != derived_after:
            return False
    elif production.get("catalog_after") != production.get("legacy_catalog"):
        return False
    return production.get("catalog_after") == production.get("legacy_catalog")


def _r2_cleanup_zero(artifact: Mapping[str, Any]) -> bool:
    cleanup = artifact.get("cleanup")
    if isinstance(cleanup, Mapping):
        remaining = cleanup.get("remaining", cleanup.get("remaining_count"))
        if remaining is not None:
            return isinstance(remaining, int) and not isinstance(remaining, bool) and remaining == 0
    cleanup = artifact.get("clone_cleanup")
    if not isinstance(cleanup, Mapping) or cleanup.get("verified") is not True:
        return False
    # R2's production receipt records the number of owned temporary objects
    # removed, not a remaining count.  A zero owned-temp count is its sealed
    # no-leftovers signal.
    owned_temp_count = cleanup.get("owned_temp_count")
    return isinstance(owned_temp_count, int) and not isinstance(owned_temp_count, bool) and owned_temp_count == 0


def _r3_production_byte_state(artifact: Mapping[str, Any], result: Mapping[str, Any]) -> bool:
    production = result.get("production", artifact.get("production"))
    if not isinstance(production, Mapping):
        production = None
    if production is not None:
        if production.get("production_workset_unchanged") is not True:
            return False
        for before_key, after_key in (
            ("protected_before", "protected_after"),
            ("workset_before", "workset_after"),
            ("owned_before", "owned_after"),
        ):
            before = production.get(before_key)
            after = production.get(after_key)
            if (before is not None or after is not None) and (
                not isinstance(before, Mapping)
                or not isinstance(after, Mapping)
                or before != after
            ):
                return False
    # The official R3 producer keeps the protected production byte boundary in
    # completion/post-completion projections rather than a synthetic top-level
    # ``sqlite_file_boundary`` key.  Require both independently sealed views
    # and their exact equality when using that schema.
    completion = result.get("completion_receipt")
    completion_boundary = completion.get("completion_boundary") if isinstance(completion, Mapping) else None
    completion_production = completion_boundary.get("production") if isinstance(completion_boundary, Mapping) else None
    post = result.get("post_completion_readback")
    post_scope = post.get("scope") if isinstance(post, Mapping) else None
    final_production = post_scope.get("production") if isinstance(post_scope, Mapping) else None
    if isinstance(completion_production, Mapping) or isinstance(final_production, Mapping):
        if not isinstance(completion_production, Mapping) or not isinstance(final_production, Mapping):
            return False
        if completion_production.get("production_workset_unchanged") is not True:
            return False
        if final_production.get("production_workset_unchanged") is not True:
            return False
        if final_production.get("matches_sealed_boundary") is not True:
            return False
        if (
            completion_production.get("protected_after")
            != final_production.get("protected_after_completion_readback")
            or completion_production.get("owned_after")
            != final_production.get("owned_after_completion_readback")
            or completion_production.get("excluded_not_evaluated")
            != final_production.get("excluded_not_evaluated")
        ):
            return False
    elif production is None:
        return False
    # R3's official artifact keeps the clone SQLite byte boundary under the
    # clone-workset normalization projection; it does not define a synthetic
    # top-level ``sqlite_file_boundary`` field.  Validate that official
    # projection when present, while retaining the production before/after
    # equality asserted by ``_assert_formal_acceptance``.
    clone_workset = result.get("clone_workset", artifact.get("clone_workset"))
    normalization = clone_workset.get("normalization") if isinstance(clone_workset, Mapping) else None
    if isinstance(normalization, Mapping):
        before = normalization.get("file_state_before")
        after = normalization.get("file_state_after")
        sidecars_before = normalization.get("sidecars_before")
        sidecars_after = normalization.get("sidecars_after")
        if (
            not isinstance(before, Mapping)
            or not isinstance(after, Mapping)
            or before.get("st_ino") != after.get("st_ino")
            or not isinstance(sidecars_before, Mapping)
            or not isinstance(sidecars_after, Mapping)
            or sidecars_before != sidecars_after
        ):
            return False
    sqlite_boundary = artifact.get("sqlite_file_boundary", result.get("sqlite_file_boundary"))
    if sqlite_boundary is not None:
        if not isinstance(sqlite_boundary, Mapping):
            return False
        for name in ("main", "wal", "shm"):
            sidecar = sqlite_boundary.get(name)
            if not isinstance(sidecar, Mapping):
                return False
            for suffix in ("sha256", "size", "mtime_ns"):
                if sidecar.get(f"{suffix}_before") != sidecar.get(f"{suffix}_after"):
                    return False
            if not isinstance(sidecar.get("sha256_before"), str) or _SHA.fullmatch(sidecar["sha256_before"]) is None:
                return False
    return True


def _validate_r4_file(
    path: Path, *, source_root: Path, source_commit: str
) -> dict[str, Any]:
    r4 = _load_source_script(source_root, "recall_r4_harness.py")
    try:
        artifact = r4.read_artifact(path)
    except Exception as exc:
        raise R8Error(f"R4 artifact readback failed: {exc}") from exc
    snapshots = [artifact.get(name) for name in ("source", "source_after", "source_final")]
    if any(not isinstance(item, Mapping) for item in snapshots):
        raise R8Error("R4 source snapshots are missing")
    commits = [item.get("commit") for item in snapshots if isinstance(item, Mapping)]
    if commits != [source_commit] * 3:
        raise R8Error("R4/source commit binding is missing or mismatched")
    if any(item.get("clean") is not True for item in snapshots if isinstance(item, Mapping)):
        raise R8Error("R4 source checkout is not clean")
    source_contract = artifact.get("source_contract")
    if not isinstance(source_contract, Mapping) or source_contract.get("passed") is not True:
        raise R8Error("R4 source contract is not certified")
    production = artifact.get("production_certification")
    if not isinstance(production, Mapping):
        raise R8Error("R4 production certification is missing")
    if (
        production.get("passed") is not True
        or production.get("provider_calls") != 0
        or production.get("reasons") != []
        or production.get("collector") != "fixed-production-root-workset-v1"
    ):
        raise R8Error("R4 production/provider gate failed")
    if artifact.get("production_root_used") is not True:
        raise R8Error("R4 fixed production root was not used")
    # The current official R4 closed schema keeps runtime/Workset identity in
    # ``production_certification`` (there is no top-level runtime_identity).
    # Consume that producer shape directly rather than requiring fields R4
    # never emits.  Missing fields still fail closed.
    try:
        expected_root = r4.PRODUCTION_ROOT
    except AttributeError as exc:
        raise R8Error("R4 fixed production root is unavailable") from exc
    if production.get("root") != str(expected_root.absolute()):
        raise R8Error("R4 production root identity is invalid")
    workset = production.get("workset")
    if not isinstance(workset, Mapping):
        raise R8Error("R4 Workset readback is missing")
    identity = _sha(workset.get("sha256"), "R4 Workset identity")
    rows = workset.get("rows")
    counts = workset.get("counts")
    receipts = workset.get("receipts")
    if (
        isinstance(rows, bool)
        or not isinstance(rows, int)
        or rows < 1
        or not isinstance(counts, Mapping)
        or not isinstance(receipts, Mapping)
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in counts.values()
        )
    ):
        raise R8Error("R4 Workset/ledger identity readback is incomplete")
    labels = production.get("labels")
    candidate = production.get("candidate_checkpoint")
    for item, label in ((labels, "R4 label identity"), (candidate, "R4 candidate checkpoint")):
        if not isinstance(item, Mapping):
            raise R8Error(f"{label} is missing")
    if not isinstance(labels, Mapping) or not isinstance(candidate, Mapping):
        raise R8Error("R4 production checkpoints are missing")
    _sha(labels.get("sha256"), "R4 label identity")
    _sha(labels.get("head_sha256"), "R4 label head identity")
    _sha(candidate.get("head_sha256"), "R4 candidate checkpoint identity")
    artifact_id, file_sha256, seal_sha256 = _bind_artifact_identity(
        path, artifact, label="R4 artifact"
    )
    return {
        "artifact_id": artifact_id,
        "file_sha256": file_sha256,
        "seal_sha256": seal_sha256,
        "source_commit": source_commit,
        "provider_calls": 0,
        "production_passed": True,
        "workset_identity": identity,
    }


def _read_sealed_with_existing_reader(
    path: Path, *, source_root: Path, schema: str, label: str
) -> dict[str, Any]:
    r2 = _load_source_script(source_root, "recall_r2_harness.py")
    store = _load_runtime_store(source_root)
    try:
        artifact, _ = r2._read_stable_sealed(store, path, schema=schema, label=label)
    except Exception as exc:
        raise R8Error(f"{label} readback failed: {exc}") from exc
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise R8Error(f"{label} bytes unavailable") from exc
    if raw not in {_canonical(artifact), _canonical(artifact) + b"\n"}:
        raise R8Error(f"{label} is not canonical")
    return dict(artifact)


def _validate_r2_file(path: Path, *, source_root: Path, source_commit: str) -> dict[str, Any]:
    r2 = _load_source_script(source_root, "recall_r2_harness.py")
    store = _load_runtime_store(source_root)
    try:
        artifact = r2._read_content_addressed_artifact(
            store, path, schema=r2.R2_SCHEMA, label="R2 artifact"
        )
    except Exception as exc:
        raise R8Error(f"R2 artifact readback failed: {exc}") from exc
    _same_source_commit(artifact, source_commit, label="R2")
    parity = artifact.get("full_rebuild_parity")
    independent = artifact.get("full_rebuild_parity_independent")
    if parity is None or independent is None:
        # The compact ``parity: true`` convenience field is not an R2
        # completion proof.  Require both official full-rebuild projections;
        # an old or fixture schema remains blocked.
        raise R8Error("R2 full rebuild parity projections are missing")
    parity_projection = _r2_parity_projection(parity, label="R2 full rebuild parity")
    independent_projection = _r2_parity_projection(
        independent, label="R2 independent full rebuild parity"
    )
    if parity_projection != independent_projection:
        raise R8Error("R2 full rebuild parity projections differ")
    if not _r2_production_unchanged(artifact):
        raise R8Error("R2 production byte-state changed or is unsealed")
    if not _r2_cleanup_zero(artifact):
        raise R8Error("R2 cleanup is not zero")
    clone_cleanup = artifact.get("clone_cleanup")
    if not isinstance(clone_cleanup, Mapping) or clone_cleanup.get("count") != 30 or clone_cleanup.get("verified") is not True or clone_cleanup.get("owned_temp_count") != 0:
        raise R8Error("R2 formal clone cleanup receipt is incomplete")
    runtime_comparison = artifact.get("runtime_comparison")
    if isinstance(runtime_comparison, Mapping) and runtime_comparison.get("runtime_drift") is not False:
        raise R8Error("R2 runtime/source drift is present")
    production = artifact.get("production")
    if (
        not isinstance(production, Mapping)
        or production.get("raw_tree") is None
        or production.get("raw_tree_before") != production.get("raw_tree_after")
        or production.get("derived_tree_before") != production.get("derived_tree_after")
        or production.get("source_tree_before") != production.get("source_tree_after")
        or production.get("catalog_after") != production.get("legacy_catalog")
    ):
        raise R8Error("R2 Raw/derived production identity is incomplete")
    artifact_id, file_sha256, seal_sha256 = _bind_artifact_identity(
        path, artifact, label="R2 artifact"
    )
    return {
        "artifact_id": artifact_id,
        "file_sha256": file_sha256,
        "seal_sha256": seal_sha256,
        "source_commit": source_commit,
        "full_rebuild": True,
        "catalog_parity": True,
        "production_unchanged": True,
        "cleanup_remaining": 0,
        "parity_projection_sha256": _digest(parity_projection),
    }


def _validate_r2_external(
    path: Path,
    *,
    source_commit: str,
    r2: Mapping[str, Any],
    r2_completion: Mapping[str, Any],
) -> dict[str, Any]:
    if path.suffix != ".json":
        raise R8Error("R2 external receipt suffix is invalid")
    value, _state = _read_json(path, label="R2 external receipt")
    expected = {
        "artifact_id",
        "schema",
        "namespace",
        "seal_sha256",
        "source_commit",
        "formal_source",
        "verdict",
        "artifacts",
        "integrity",
        "cleanup",
        "supervisor",
    }
    if set(value) != expected:
        raise R8Error("R2 external receipt schema is not closed")
    if value.get("schema") != "chronovisor.recall-r2-external-receipt.v1":
        raise R8Error("R2 external receipt schema mismatch")
    _verify_sealed(value, schema=str(value["schema"]), label="R2 external receipt")
    artifact_id, file_sha256, seal_sha256 = _bind_artifact_identity(
        path, value, label="R2 external receipt"
    )
    if value.get("source_commit") != source_commit:
        raise R8Error("R2 external/source commit binding is invalid")
    formal_source = value.get("formal_source")
    if not isinstance(formal_source, Mapping) or formal_source.get("commit") != source_commit or formal_source.get("clean") is not True:
        raise R8Error("R2 external formal source binding is invalid")
    if value.get("verdict") != "approved":
        raise R8Error("R2 external verdict is not approved")
    supervisor = value.get("supervisor")
    if not isinstance(supervisor, Mapping) or set(supervisor) != {
        "validator",
        "certified",
        "source_commit",
        "source_tree_sha256",
        "harness_path",
        "evidence_path",
        "artifact_id",
        "file_sha256",
        "seal_sha256",
        "production_root",
    }:
        raise R8Error("R2 external supervisor receipt is missing")
    if (
        supervisor.get("validator") != "chronovisor.recall.r2.external-supervisor.v1"
        or supervisor.get("certified") is not True
        or supervisor.get("source_commit") != source_commit
        or supervisor.get("harness_path") != "scripts/recall_r2_harness.py"
        or supervisor.get("evidence_path") != str(path.resolve())
        or not isinstance(supervisor.get("production_root"), str)
        or not supervisor["production_root"].startswith("/")
    ):
        raise R8Error("R2 external supervisor/source identity is invalid")
    _sha(supervisor.get("source_tree_sha256"), "R2 external supervisor source tree")
    for key, expected_value in (
        ("artifact_id", artifact_id),
        ("file_sha256", file_sha256),
        ("seal_sha256", seal_sha256),
    ):
        if supervisor.get(key) != expected_value:
            raise R8Error("R2 external supervisor artifact binding is invalid")
    artifacts = value.get("artifacts")
    if not isinstance(artifacts, Mapping) or not isinstance(artifacts.get("artifact"), Mapping) or not isinstance(artifacts.get("completion"), Mapping):
        raise R8Error("R2 external artifact bindings are missing")
    main_ref = artifacts["artifact"]
    completion_ref = artifacts["completion"]
    if (
        main_ref.get("artifact_id") != r2["artifact_id"]
        or main_ref.get("sha256") != r2["file_sha256"]
        or completion_ref.get("sha256") != r2_completion["file_sha256"]
        or completion_ref.get("artifact_id") != r2_completion.get("artifact_id")
    ):
        raise R8Error("R2 external/main/completion file binding is invalid")
    integrity = value.get("integrity")
    if (
        not isinstance(integrity, Mapping)
        or set(integrity)
        != {"artifact", "completion", "catalog", "fts", "inventory", "source", "raw", "derived", "production"}
        or any(item is not True for item in integrity.values())
    ):
        raise R8Error("R2 external integrity parity is incomplete")
    cleanup = value.get("cleanup")
    if (
        not isinstance(cleanup, Mapping)
        or set(cleanup)
        != {
            "clone_count",
            "verified",
            "owned_temp_count",
            "remaining_clone_paths",
            "remaining_temp_prefixes",
            "formal_processes",
        }
        or any(
        isinstance(cleanup.get(key), bool) if expected_value == 0 else False
        or cleanup.get(key) != expected_value
        for key, expected_value in (("clone_count", 30), ("verified", True), ("owned_temp_count", 0), ("remaining_clone_paths", 0), ("remaining_temp_prefixes", 0), ("formal_processes", 0))
        )
    ):
        raise R8Error("R2 external cleanup is incomplete")
    return {
        "artifact_id": artifact_id,
        "file_sha256": file_sha256,
        "seal_sha256": seal_sha256,
        "source_commit": source_commit,
        "main_artifact_id": r2["artifact_id"],
        "main_artifact_sha256": r2["file_sha256"],
        "completion_file_sha256": r2_completion["file_sha256"],
        "cleanup_remaining": 0,
    }


def _validate_r2_completion_file(
    path: Path,
    *,
    source_root: Path,
    source_commit: str,
    r2: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the official ``.completion.receipt`` producer shape."""

    r2_harness = _load_source_script(source_root, "recall_r2_harness.py")
    store = _load_runtime_store(source_root)
    schema = getattr(store, "DISTILLATION_SCHEMA", "chronovisor.recall-distillation.v1")
    try:
        completion, _ = r2_harness._read_stable_sealed(
            store, path, schema=schema, label="R2 completion receipt"
        )
    except Exception as exc:
        raise R8Error(f"R2 completion receipt readback failed: {exc}") from exc
    if path.suffix != ".receipt" or not path.name.endswith(".completion.receipt"):
        raise R8Error("R2 completion receipt suffix is invalid")
    required = {
        "schema",
        "namespace",
        "kind",
        "r2_schema",
        "r2_artifact_id",
        "r2_artifact_path",
        "r2_artifact_seal_sha256",
        "phase_timing",
        "completion_started_ns",
        "completion_publication_readback_ns",
        "seal_sha256",
    }
    if set(completion) != required:
        raise R8Error("R2 completion receipt schema is not closed")
    if completion.get("kind") != "chronovisor.recall-r2-completion":
        raise R8Error("R2 completion receipt kind is invalid")
    if completion.get("r2_schema") != getattr(r2_harness, "R2_SCHEMA", "chronovisor.recall-r2.v1"):
        raise R8Error("R2 completion/main schema binding is invalid")
    if completion.get("r2_artifact_id") != r2.get("artifact_id"):
        raise R8Error("R2 completion/artifact ID binding is invalid")
    if completion.get("r2_artifact_seal_sha256") != r2.get("seal_sha256"):
        raise R8Error("R2 completion/artifact seal binding is invalid")
    if not isinstance(completion.get("r2_artifact_path"), str) or not completion["r2_artifact_path"].endswith(".json"):
        raise R8Error("R2 completion/main path binding is invalid")
    if not isinstance(completion.get("phase_timing"), Mapping):
        raise R8Error("R2 completion phase timing is missing")
    for key in ("completion_started_ns", "completion_publication_readback_ns"):
        _int(completion.get(key), f"R2 completion {key}")
    completion_id = _digest(completion)
    completion_file_sha = _receipt_file_identity(path, label="R2 completion receipt")
    completion_seal = _sha(completion.get("seal_sha256"), "R2 completion seal")
    if path.name.removesuffix(".completion.receipt") != r2.get("artifact_id"):
        raise R8Error("R2 completion filename/main binding is invalid")
    return {
        "artifact_id": completion_id,
        "file_sha256": completion_file_sha,
        "seal_sha256": completion_seal,
        "r2_artifact_id": r2["artifact_id"],
        "r2_artifact_seal_sha256": r2["seal_sha256"],
        "r2_artifact_sha256": r2["file_sha256"],
        "source_commit": source_commit,
    }


def _validate_r3_file(path: Path, *, source_root: Path, source_commit: str) -> dict[str, Any]:
    r3 = _load_source_script(source_root, "recall_r3_harness.py")
    artifact = _read_sealed_with_existing_reader(
        path, source_root=source_root, schema=r3.R3_SCHEMA, label="R3 artifact"
    )
    source = artifact.get("source")
    _same_source_commit(artifact, source_commit, label="R3")
    result = artifact.get("result", artifact)
    if not isinstance(result, Mapping) or not isinstance(source, Mapping):
        raise R8Error("R3 result/source binding is missing")
    try:
        r3._assert_formal_acceptance(result, source, require_completion=True)
    except Exception as exc:
        raise R8Error(f"R3 completion validator rejected artifact: {exc}") from exc
    clone_workset = result.get("clone_workset")
    final_status = clone_workset.get("final_status") if isinstance(clone_workset, Mapping) else None
    leased = final_status.get("leased") if isinstance(final_status, Mapping) else None
    duplicates = result.get("duplicates")
    cleanup = result.get("cleanup")
    production_unchanged = result.get("production_workset_unchanged")
    if production_unchanged is None:
        boundary = result.get("production_write_boundary")
        production_unchanged = boundary.get("production_workset_unchanged") if isinstance(boundary, Mapping) else None
    if (
        not isinstance(leased, int)
        or isinstance(leased, bool)
        or leased != 0
        or not isinstance(duplicates, int)
        or isinstance(duplicates, bool)
        or duplicates != 0
        or production_unchanged is not True
        or not _r3_production_byte_state(artifact, result)
        or not isinstance(cleanup, Mapping)
        or not isinstance(cleanup.get("remaining"), int)
        or isinstance(cleanup.get("remaining"), bool)
        or cleanup.get("remaining") != 0
    ):
        raise R8Error("R3 leased/duplicate/production-byte-state/cleanup gate failed")
    completion = artifact.get("completion_receipt")
    if completion is None and isinstance(result, Mapping):
        completion = result.get("completion_receipt")
    artifact_id, file_sha256, seal_sha256 = _bind_artifact_identity(
        path, artifact, label="R3 artifact"
    )
    if completion is not None:
        if not isinstance(completion, Mapping) or completion.get("readback_verified") is not True:
            raise R8Error("R3 completion readback is invalid")
        completion_main_id = completion.get("main_artifact_id", completion.get("sealed_artifact_id"))
        if completion_main_id != artifact_id:
            raise R8Error("R3 completion/main artifact ID binding is missing")
        completion_main_sha = completion.get(
            "main_artifact_sha256", completion.get("sealed_artifact_sha256")
        )
        if completion_main_sha is not None and completion_main_sha != file_sha256:
            raise R8Error("R3 completion/main artifact file hash binding is invalid")
    return {
        "artifact_id": artifact_id,
        "file_sha256": file_sha256,
        "seal_sha256": seal_sha256,
        "source_commit": source_commit,
        "leased": 0,
        "duplicates": 0,
        "production_unchanged": True,
        "cleanup_remaining": 0,
        "completion_readback_verified": True,
        "completion_main_artifact_id": artifact_id,
        "completion_main_artifact_sha256": file_sha256,
    }


def _validate_r3_external(
    path: Path,
    *,
    source_commit: str,
    r3: Mapping[str, Any],
    r3_completion: Mapping[str, Any],
) -> dict[str, Any]:
    if path.suffix != ".json":
        raise R8Error("R3 external receipt suffix is invalid")
    value, _state = _read_json(path, label="R3 external receipt")
    required = {
        "artifact_id",
        "schema",
        "namespace",
        "seal_sha256",
        "source_commit",
        "verdict",
        "execution",
        "artifacts",
        "acceptance",
        "production_workset",
        "service_boundary",
    }
    if not required.issubset(value):
        raise R8Error("R3 external receipt schema is incomplete")
    if value.get("schema") != "chronovisor.recall-r3-external-receipt.v1":
        raise R8Error("R3 external receipt schema mismatch")
    _verify_sealed(value, schema=str(value["schema"]), label="R3 external receipt")
    artifact_id, file_sha256, seal_sha256 = _bind_artifact_identity(
        path, value, label="R3 external receipt"
    )
    if value.get("source_commit") != source_commit or value.get("verdict") != "approved":
        raise R8Error("R3 external source/verdict binding is invalid")
    execution = value.get("execution")
    if not isinstance(execution, Mapping) or execution.get("source_commit") != source_commit or execution.get("source_clean_after") is not True or execution.get("provider_calls") != 0 or execution.get("production_mutation", execution.get("production_semantic_mutation")) is not False:
        raise R8Error("R3 external execution boundary is invalid")
    artifacts = value.get("artifacts")
    if not isinstance(artifacts, Mapping) or not isinstance(artifacts.get("main"), Mapping) or not isinstance(artifacts.get("completion"), Mapping):
        raise R8Error("R3 external artifact bindings are missing")
    main_ref = artifacts["main"]
    completion_ref = artifacts["completion"]
    if (
        main_ref.get("artifact_id") != r3["artifact_id"]
        or main_ref.get("file_sha256") != r3["file_sha256"]
        or completion_ref.get("artifact_id") != r3_completion["artifact_id"]
        or completion_ref.get("file_sha256") != r3_completion["file_sha256"]
        or main_ref.get("seal_and_readback_verified") is not True
        or completion_ref.get("seal_and_readback_verified") is not True
    ):
        raise R8Error("R3 external/main/completion file binding is invalid")
    acceptance = value.get("acceptance")
    if not isinstance(acceptance, Mapping) or acceptance.get("duplicates") != 0 or acceptance.get("leased_after") != 0 or acceptance.get("clone_cleanup_remaining") != 0:
        raise R8Error("R3 external acceptance is incomplete")
    workset = value.get("production_workset")
    if not isinstance(workset, Mapping) or workset.get("external_boundary_readback_verified") is not True or workset.get("active_lease_fields") != 0 or workset.get("distillation_lock_holders") != 0 or workset.get("clone_temp_remaining") != 0 or workset.get("content_sha256_before") != workset.get("content_sha256_after") or workset.get("row_count_before") != workset.get("row_count_after"):
        raise R8Error("R3 external Workset boundary is incomplete")
    production_workset = value.get("production_workset")
    sqlite_boundary = value.get("sqlite_file_boundary")
    if sqlite_boundary is None and isinstance(production_workset, Mapping):
        sqlite_boundary = next(
            (
                production_workset.get(key)
                for key in ("sqlite_file_boundary", "sqlite_boundary", "sqlite")
                if production_workset.get(key) is not None
            ),
            None,
        )
    if not isinstance(sqlite_boundary, Mapping):
        raise R8Error("R3 external SQLite boundary is missing")
    for name in ("main", "wal", "shm"):
        entry = sqlite_boundary.get(name)
        if not isinstance(entry, Mapping) or entry.get("sha256_before") != entry.get("sha256_after") or entry.get("size_before") != entry.get("size_after") or entry.get("mtime_ns_before") != entry.get("mtime_ns_after"):
            raise R8Error("R3 external SQLite byte-state changed")
    service = value.get("service_boundary")
    if (
        not isinstance(service, Mapping)
        or service.get("ox_enabled") is not False
        or any(
            isinstance(service.get(key), bool)
            or not isinstance(service.get(key), int)
            or service.get(key) != 0
            for key in ("distillation_processes", "distillation_lock_holders", "workset_leased")
        )
    ):
        raise R8Error("R3 external service boundary is not quiescent")
    return {
        "artifact_id": artifact_id,
        "file_sha256": file_sha256,
        "seal_sha256": seal_sha256,
        "source_commit": source_commit,
        "main_artifact_id": r3["artifact_id"],
        "main_artifact_sha256": r3["file_sha256"],
        "completion_artifact_id": r3_completion["artifact_id"],
        "completion_file_sha256": r3_completion["file_sha256"],
    }


def _validate_auxiliary(
    *,
    source_root: Path,
    source_commit: str,
    r4_artifact: Path | None,
    r2_artifact: Path | None,
    r2_completion: Path | None,
    r2_external: Path | None = None,
    r3_artifact: Path | None,
    r3_completion: Path | None,
    r3_external: Path | None = None,
) -> tuple[dict[str, Any], list[str]]:
    reasons: list[str] = []
    output: dict[str, Any] = {}
    if r4_artifact is None:
        reasons.append("r4_artifact_missing")
    else:
        try:
            output["r4"] = _validate_r4_file(
                r4_artifact, source_root=source_root, source_commit=source_commit
            )
        except R8Error as exc:
            reasons.append(f"r4_invalid:{str(exc).split(':', 1)[0]}")
    if r2_artifact is None:
        reasons.append("r2_artifact_missing")
    else:
        try:
            output["r2"] = _validate_r2_file(r2_artifact, source_root=source_root, source_commit=source_commit)
        except R8Error as exc:
            reasons.append(f"r2_invalid:{str(exc).split(':', 1)[0]}")
    if r2_completion is None:
        reasons.append("r2_completion_missing")
    else:
        try:
            if output.get("r2") is None:
                raise R8Error("R2 artifact is required for completion binding")
            if r2_completion.name.endswith(".completion.receipt"):
                output["r2_completion"] = _validate_r2_completion_file(
                    r2_completion,
                    source_root=source_root,
                    source_commit=source_commit,
                    r2=output["r2"],
                )
            else:
                # Keep compatibility with the pre-R2-completion producer only
                # for its fully closed JSON shape.  Arbitrary ``.txt`` or
                # self-claimed mappings never enter this branch.
                r2_completion_value = _read_sealed_with_existing_reader(
                    r2_completion,
                    source_root=source_root,
                    schema="chronovisor.recall-distillation.v1",
                    label="R2 completion receipt",
                )
                if r2_completion_value.get("kind") != "chronovisor.recall-r2-completion":
                    raise R8Error("R2 completion kind is invalid")
                if r2_completion.suffix != ".json":
                    raise R8Error("R2 completion suffix is invalid")
                if set(r2_completion_value) != {
                    "artifact_id", "schema", "namespace", "kind",
                    "r2_artifact_id", "r2_artifact_seal_sha256",
                    "r2_artifact_sha256", "source_commit", "seal_sha256",
                }:
                    raise R8Error("R2 completion schema is not closed")
                completion_id, completion_file_sha, completion_seal = _bind_completion_identity(
                    r2_completion, r2_completion_value, label="R2 completion receipt"
                )
                if (
                    r2_completion_value.get("r2_artifact_id") != output["r2"]["artifact_id"]
                    or r2_completion_value.get("r2_artifact_seal_sha256") != output["r2"]["seal_sha256"]
                    or r2_completion_value.get("r2_artifact_sha256") != output["r2"]["file_sha256"]
                    or r2_completion_value.get("source_commit") != source_commit
                ):
                    raise R8Error("R2 completion/artifact binding is invalid")
                output["r2_completion"] = {
                    "artifact_id": completion_id,
                    "file_sha256": completion_file_sha,
                    "seal_sha256": completion_seal,
                    "r2_artifact_id": output["r2"]["artifact_id"],
                    "r2_artifact_seal_sha256": output["r2"]["seal_sha256"],
                    "r2_artifact_sha256": output["r2"]["file_sha256"],
                    "source_commit": source_commit,
                }
        except R8Error as exc:
            reasons.append(f"r2_completion_invalid:{str(exc).split(':', 1)[0]}")
    if r2_external is None:
        reasons.append("r2_external_missing")
    else:
        try:
            if output.get("r2") is None or output.get("r2_completion") is None:
                raise R8Error("R2 artifact/completion is required for external binding")
            output["r2_external"] = _validate_r2_external(
                r2_external,
                source_commit=source_commit,
                r2=output["r2"],
                r2_completion=output["r2_completion"],
            )
        except R8Error as exc:
            reasons.append(f"r2_external_invalid:{str(exc).split(':', 1)[0]}")
    if r3_artifact is None:
        reasons.append("r3_artifact_missing")
    else:
        try:
            output["r3"] = _validate_r3_file(r3_artifact, source_root=source_root, source_commit=source_commit)
        except R8Error as exc:
            reasons.append(f"r3_invalid:{str(exc).split(':', 1)[0]}")
    if r3_completion is None:
        reasons.append("r3_completion_missing")
    else:
        try:
            completion = _read_sealed_with_existing_reader(
                r3_completion,
                source_root=source_root,
                schema="chronovisor.recall-r3-completion.v1",
                label="R3 completion receipt",
            )
            if completion.get("readback_verified") is not True:
                raise R8Error("R3 completion is not readback verified")
            required_completion = {
                "artifact_id",
                "schema",
                "namespace",
                "seal_sha256",
                "main_artifact_id",
                "main_artifact_sha256",
                "sealed_artifact_id",
                "sealed_artifact_sha256",
                "source_commit",
                "readback_verified",
                "completion_boundary",
            }
            if not required_completion.issubset(completion):
                raise R8Error("R3 completion schema is incomplete")
            if output.get("r3") is None:
                raise R8Error("R3 artifact is required for completion binding")
            main_id = completion.get("main_artifact_id", completion.get("sealed_artifact_id"))
            main_sha = completion.get(
                "main_artifact_sha256", completion.get("sealed_artifact_sha256")
            )
            if main_id != output["r3"]["artifact_id"]:
                raise R8Error("R3 completion/artifact ID binding is invalid")
            if main_sha != output["r3"]["file_sha256"]:
                raise R8Error("R3 completion/artifact file hash binding is invalid")
            if completion.get("sealed_artifact_id") != output["r3"]["artifact_id"] or completion.get("sealed_artifact_sha256") != output["r3"]["file_sha256"]:
                raise R8Error("R3 completion/sealed artifact binding is invalid")
            completion_source = completion.get("source_commit")
            if completion_source != source_commit:
                raise R8Error("R3 completion/source commit binding is invalid")
            completion_boundary = completion.get("completion_boundary")
            if not isinstance(completion_boundary, Mapping):
                raise R8Error("R3 completion boundary is missing")
            boundary_production = completion_boundary.get("production")
            if not isinstance(boundary_production, Mapping) or not _r3_production_byte_state(
                completion_boundary, completion_boundary
            ):
                raise R8Error("R3 completion production boundary is invalid")
            completion_id, completion_file_sha, completion_seal = _bind_artifact_identity(
                r3_completion, completion, label="R3 completion receipt"
            )
            output["r3_completion"] = {
                "artifact_id": completion_id,
                "file_sha256": completion_file_sha,
                "seal_sha256": completion_seal,
                "main_artifact_id": output["r3"]["artifact_id"],
                "main_artifact_sha256": output["r3"]["file_sha256"],
                "source_commit": source_commit,
                "readback_verified": True,
            }
        except R8Error as exc:
            reasons.append(f"r3_completion_invalid:{str(exc).split(':', 1)[0]}")
    if r3_external is None:
        reasons.append("r3_external_missing")
    else:
        try:
            if output.get("r3") is None or output.get("r3_completion") is None:
                raise R8Error("R3 artifact/completion is required for external binding")
            output["r3_external"] = _validate_r3_external(
                r3_external,
                source_commit=source_commit,
                r3=output["r3"],
                r3_completion=output["r3_completion"],
            )
        except R8Error as exc:
            reasons.append(f"r3_external_invalid:{str(exc).split(':', 1)[0]}")
    return output, sorted(set(reasons))


def _status_value(section: Mapping[str, Any], *keys: str) -> object:
    for key in keys:
        if key in section:
            return section[key]
    for nested_name in ("runtime", "workset", "state", "safety"):
        nested = section.get(nested_name)
        if isinstance(nested, Mapping):
            for key in keys:
                if key in nested:
                    return nested[key]
    return None


def _validate_phase_receipt(
    value: object,
    phase: Mapping[str, Any],
    *,
    source_commit: str,
    inventory_sha256: str,
    previous: Mapping[str, Any] | None,
    actual_prerequisites: Mapping[str, bool],
) -> bool:
    if not isinstance(value, Mapping):
        return False
    expected_keys = {
        "artifact_id",
        "file_sha256",
        "schema",
        "namespace",
        "kind",
        "phase",
        "order",
        "status",
        "rollback_artifact",
        "source_commit",
        "inventory_sha256",
        "previous_artifact_id",
        "previous_artifact_sha256",
        "seal_sha256",
        "official_validation",
    }
    if set(value) != expected_keys:
        return False
    try:
        _verify_sealed(value, schema=R8_PHASE_RECEIPT_SCHEMA, label=f"phase {phase['name']}")
        artifact_id = _sha(value.get("artifact_id"), f"phase {phase['name']} id")
        if artifact_id != _digest(
            {key: item for key, item in value.items() if key not in {"artifact_id", "seal_sha256"}}
        ):
            return False
        if _sha(value.get("file_sha256"), f"phase {phase['name']} content hash") != _embedded_content_digest(
            value, label=f"phase {phase['name']} receipt"
        ):
            return False
        _sha(value.get("inventory_sha256"), f"phase {phase['name']} inventory hash")
    except R8Error:
        return False
    if value.get("source_commit") != source_commit or value.get("inventory_sha256") != inventory_sha256:
        return False
    official = value.get("official_validation")
    if not isinstance(official, Mapping) or set(official) != {
        "validator",
        "certified",
        "source_commit",
        "inventory_sha256",
        "prerequisites",
    }:
        return False
    if (
        official.get("validator") != "chronovisor.recall.r8.phase-validator.v1"
        or official.get("certified") is not True
        or official.get("source_commit") != source_commit
        or official.get("inventory_sha256") != inventory_sha256
        or official.get("prerequisites") != dict(actual_prerequisites)
        or any(item is not True for item in actual_prerequisites.values())
    ):
        return False
    if type(value.get("order")) is not int or value.get("order") != phase["order"]:
        return False
    if previous is None:
        if value.get("previous_artifact_id") is not None or value.get("previous_artifact_sha256") is not None:
            return False
    elif (
        value.get("previous_artifact_id") != previous.get("artifact_id")
        or value.get("previous_artifact_sha256") != previous.get("file_sha256")
    ):
        return False
    return (
        value.get("kind") == "r8-phase-receipt"
        and value.get("phase") == phase["name"]
        and value.get("status") == "sealed"
        and value.get("rollback_artifact") == phase["rollback_artifact"]
    )


_PHASE_REF_KEYS = {"path", "artifact_id", "file_sha256", "seal_sha256", "schema"}


def _read_phase_receipt_ref(
    value: object,
    *,
    source_root: Path | None,
    phase_name: str,
) -> tuple[dict[str, Any], Path, str, str] | None:
    """Read one phase receipt from its immutable, content-addressed file.

    Inline sealed mappings are intentionally not receipts.  The observation
    carries only this fixed reference envelope; the bytes and seal are read
    again from the referenced regular file before the phase can become ready.
    """

    if source_root is None or not isinstance(value, Mapping):
        return None
    if set(value) != _PHASE_REF_KEYS:
        return None
    raw_path = value.get("path")
    if not isinstance(raw_path, str) or not raw_path.startswith("/"):
        return None
    path = Path(raw_path)
    if path.suffix != ".json" or _has_symlink_component(path) or not path.is_file():
        return None
    try:
        artifact, state = _read_json(path, label=f"phase {phase_name} receipt")
        actual_id, actual_file_sha, actual_seal = _bind_artifact_identity(
            path, artifact, label=f"phase {phase_name} receipt"
        )
        if artifact.get("schema") != R8_PHASE_RECEIPT_SCHEMA:
            return None
        if (
            value.get("artifact_id") != actual_id
            or value.get("file_sha256") != actual_file_sha
            or value.get("seal_sha256") != actual_seal
            or value.get("schema") != artifact.get("schema")
        ):
            return None
        # The receipt's embedded ``file_sha256`` is the canonical payload
        # digest.  The parent reference carries the independent SHA of the
        # complete receipt file, avoiding a circular self-hash.
        if artifact.get("file_sha256") != _embedded_content_digest(
            artifact, label=f"phase {phase_name} receipt"
        ):
            return None
        if _file_state(path, label=f"phase {phase_name} receipt") != state:
            return None
    except (OSError, R8Error):
        return None
    return artifact, path, actual_file_sha, actual_seal


def _inventory_summary(observation: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    reasons: list[str] = []
    ox = observation.get("ox")
    pointers = observation.get("pointers")
    legacy = observation.get("legacy")
    locks = observation.get("locks")
    if not isinstance(ox, Mapping):
        reasons.append("ox_inventory_missing")
        ox = {}
    if not isinstance(pointers, Mapping):
        reasons.append("pointer_inventory_missing")
        pointers = {}
    if not isinstance(legacy, Mapping):
        reasons.append("legacy_inventory_missing")
        legacy = {}
    if not isinstance(locks, Mapping):
        reasons.append("lock_inventory_missing")
        locks = {}
    ox_safe = _safe_metadata(ox, label="ox inventory")
    ptr_safe = _safe_metadata(pointers, label="pointer inventory")
    legacy_safe = _safe_metadata(legacy, label="legacy inventory")
    locks_safe = _safe_metadata(locks, label="lock inventory")
    for name in _OX_FILES:
        if name not in ox_safe:
            reasons.append(f"ox_{name}_inventory_missing")
    for name in _POINTER_FILES:
        if name not in ptr_safe:
            reasons.append(f"pointer_{name}_inventory_missing")
    for name in _LEGACY_FILES:
        if name not in legacy_safe:
            reasons.append(f"legacy_{name}_inventory_missing")
    for name in _PROCESS_LOCKS:
        if name not in locks_safe:
            reasons.append(f"lock_{name}_inventory_missing")
    for section_name, expected_names, section in (
        ("ox", tuple(_OX_FILES), ox_safe),
        ("legacy", tuple(_LEGACY_FILES), legacy_safe),
        ("locks", _PROCESS_LOCKS, locks_safe),
    ):
        for name in expected_names:
            entry = section.get(name)
            if not isinstance(entry, Mapping):
                continue
            if entry.get("present") is True and (
                not isinstance(entry.get("sha256"), str)
                or _SHA.fullmatch(entry["sha256"]) is None
                or _safe_file_state(entry.get("file_state")) is None
                or not isinstance(entry.get("sidecars"), Mapping)
            ):
                reasons.append(f"{section_name}_{name}_state_or_hash_incomplete")
    enabled = _status_value(ox_safe, "enabled", "ox_enabled")
    provider_calls = _status_value(ox_safe, "provider_calls", "external_provider_calls")
    leased = _status_value(ox_safe, "leased", "leased_count")
    process_lock = _status_value(ox_safe, "process_lock", "distillation_process_lock", "lock_present")
    process_count = _status_value(ox_safe, "process_count", "distillation_processes")
    if enabled is not False:
        reasons.append("ox_enabled")
    if (
        isinstance(provider_calls, bool)
        or not isinstance(provider_calls, int)
        or provider_calls != 0
    ):
        reasons.append("provider_calls_nonzero")
    if isinstance(leased, bool) or not isinstance(leased, int) or leased != 0:
        reasons.append("leased_work_present")
    if process_lock is not False:
        reasons.append("distillation_process_lock_present")
    if (
        isinstance(process_count, bool)
        or not isinstance(process_count, int)
        or process_count != 0
    ):
        reasons.append("distillation_processes_nonzero_or_unobserved")
    # Candidate may be absent after rollback, but active and LKG inventory must
    # be observed.  No policy contents are copied into the artifact.
    for name in ("active", "candidate", "lkg"):
        entry = ptr_safe.get(name)
        if (
            not isinstance(entry, Mapping)
            or entry.get("present") is not True
            or not isinstance(entry.get("sha256"), str)
            or _SHA.fullmatch(entry["sha256"]) is None
            or not isinstance(entry.get("file_state"), Mapping)
        ):
            reasons.append(f"pointer_{name}_missing")
    return {
        "ox": ox_safe,
        "pointers": ptr_safe,
        "legacy": legacy_safe,
        "locks": locks_safe,
        "safety": {
            "enabled": enabled,
            "provider_calls": provider_calls,
            "leased": leased,
            "process_lock": process_lock,
            "process_count": process_count,
        },
    }, sorted(set(reasons))


def _phase_summary(
    observation: Mapping[str, Any],
    prior_reasons: Sequence[str],
    *,
    source_commit: str,
    inventory_sha256: str,
    source_root: Path | None = None,
    r7_ready: bool = False,
) -> tuple[list[dict[str, Any]], list[str]]:
    supplied = observation.get("phase_receipts")
    if not isinstance(supplied, Mapping):
        supplied = {}
    phases: list[dict[str, Any]] = []
    reasons: list[str] = []
    previous_phase_blocked = False
    previous_receipt: Mapping[str, Any] | None = None
    for phase in PHASES:
        actual_prerequisites = {
            prerequisite: False for prerequisite in phase["prerequisites"]
        }
        if phase["name"] == "r7_receipt":
            for prerequisite in actual_prerequisites:
                actual_prerequisites[prerequisite] = r7_ready and not prior_reasons
        elif phase["name"] == "ox_drain_archive":
            ox = observation.get("ox")
            actual_prerequisites["ox_off"] = isinstance(ox, Mapping) and ox.get("enabled") is False
            actual_prerequisites["provider_calls_zero"] = isinstance(ox, Mapping) and ox.get("provider_calls") == 0
            actual_prerequisites["leased_zero"] = isinstance(ox, Mapping) and ox.get("leased") == 0
            actual_prerequisites["distillation_process_lock_absent"] = isinstance(ox, Mapping) and ox.get("process_lock") is False
        else:
            for prerequisite in actual_prerequisites:
                actual_prerequisites[prerequisite] = not prior_reasons
        receipt_ref = supplied.get(phase["name"])
        loaded = _read_phase_receipt_ref(
            receipt_ref,
            source_root=source_root,
            phase_name=phase["name"],
        )
        receipt = loaded[0] if loaded is not None else None
        valid = _validate_phase_receipt(
            receipt,
            phase,
            source_commit=source_commit,
            inventory_sha256=inventory_sha256,
            previous=previous_receipt,
            actual_prerequisites=actual_prerequisites,
        )
        blocked_by = list(phase["prerequisites"])
        blocked_by.extend(prior_reasons)
        if not valid:
            blocked_by.append("sealed_phase_receipt_missing_or_invalid")
        if previous_phase_blocked:
            blocked_by.append("prior_phase_not_ready")
        status_ready = valid and not prior_reasons and not previous_phase_blocked
        phases.append(
            {
                "name": phase["name"],
                "order": phase["order"],
                "prerequisites": list(phase["prerequisites"]),
                "rollback_artifact": phase["rollback_artifact"],
                "receipt_supplied": receipt_ref is not None,
                "receipt_valid": valid,
                "status": "ready" if status_ready else "blocked",
                "blocked_by": sorted(set(blocked_by)),
            }
        )
        if not valid:
            reasons.append(f"phase_{phase['name']}_receipt_missing_or_invalid")
        if previous_phase_blocked:
            reasons.append(f"phase_{phase['name']}_prior_phase_not_ready")
        previous_phase_blocked = previous_phase_blocked or not valid
        if valid and isinstance(receipt, Mapping):
            previous_receipt = receipt
    return phases, sorted(set(reasons))


def _source_identity(source: Path, commit: str) -> dict[str, Any]:
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        r7 = _load_source_script(source, "recall_r7_harness.py")
        identity = r7._source_identity(source, commit)
    except Exception as exc:
        raise R8Error(str(exc)) from exc
    finally:
        sys.dont_write_bytecode = previous
    if identity.get("source_commit") != commit or identity.get("source_clean") not in {"true", True}:
        raise R8Error("source commit drift or dirty checkout")
    return {
        "source_commit": commit,
        "source_tree_sha256": _sha(identity.get("source_tree_sha256"), "source tree"),
        "source_clean": True,
    }


def _evidence_ref(
    path: Path,
    *,
    artifact_id: object,
    file_sha256: object,
    seal_sha256: object,
    schema: str,
    label: str,
) -> dict[str, Any]:
    """Create a reference whose identity is checked against current bytes."""

    if not path.is_absolute() or _has_symlink_component(path) or not path.is_file():
        raise R8Error(f"{label} path is unavailable")
    if not path.name.endswith((".json", ".completion.receipt")):
        raise R8Error(f"{label} path suffix is invalid")
    actual_id = _sha(artifact_id, f"{label} artifact id")
    actual_file_sha = _sha(file_sha256, f"{label} file hash")
    actual_seal = _sha(seal_sha256, f"{label} seal")
    state = _file_state(path, label=label)
    observed_sha = _hash_file(path, state, label=label)
    if observed_sha != actual_file_sha:
        raise R8Error(f"{label} file hash does not match bytes")
    return {
        "path": str(path),
        "artifact_id": actual_id,
        "file_sha256": actual_file_sha,
        "seal_sha256": actual_seal,
        "schema": schema,
    }


def _build_evidence(
    *,
    source_root: Path,
    r7_artifact: Path,
    r7_summary: Mapping[str, Any],
    observation_path: Path | None,
    observation: Mapping[str, Any],
    auxiliary_paths: Mapping[str, Path | None],
    auxiliary: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind every readiness summary to an actual immutable producer file."""

    try:
        r7_value, _ = _read_json(r7_artifact, label="R7 evidence reference")
        evidence: dict[str, Any] = {
            "status": "bound",
            "source_root": str(source_root.absolute()),
            "observation": None,
            "r7": _evidence_ref(
                r7_artifact,
                artifact_id=r7_summary.get("artifact_id"),
                file_sha256=r7_summary.get("file_sha256"),
                seal_sha256=r7_value.get("seal_sha256"),
                schema=R7_SCHEMA,
                label="R7 evidence",
            ),
            "auxiliary": {},
            "phases": {},
        }
        if observation_path is None or not observation_path.is_file():
            raise R8Error("sealed production observation reference is missing")
        observation_value, observation_state = _read_json(
            observation_path, label="production observation reference"
        )
        if observation_value.get("schema") != R8_OBSERVATION_SCHEMA:
            raise R8Error("production observation schema is invalid")
        observation_file_sha = _hash_file(
            observation_path, observation_state, label="production observation reference"
        )
        if observation_file_sha is None:
            raise R8Error("production observation file hash is unavailable")
        evidence["observation"] = _evidence_ref(
            observation_path,
            artifact_id=observation_value.get("artifact_id"),
            # The observation's embedded ``file_sha256`` is its payload
            # digest; the evidence ref must bind the complete bytes actually
            # read from disk.
            file_sha256=observation_file_sha,
            seal_sha256=observation_value.get("seal_sha256"),
            schema=R8_OBSERVATION_SCHEMA,
            label="production observation",
        )
        phase_receipts = observation_value.get("production", {}).get("phase_receipts")
        if not isinstance(phase_receipts, Mapping):
            raise R8Error("phase receipt references are missing")
        for phase in _R8_PHASE_NAMES:
            ref = phase_receipts.get(phase)
            loaded = _read_phase_receipt_ref(
                ref, source_root=source_root, phase_name=phase
            )
            if loaded is None:
                raise R8Error(f"phase {phase} receipt reference is invalid")
            artifact, receipt_path, file_sha, seal = loaded
            evidence["phases"][phase] = _evidence_ref(
                receipt_path,
                artifact_id=artifact.get("artifact_id"),
                file_sha256=file_sha,
                seal_sha256=seal,
                schema=R8_PHASE_RECEIPT_SCHEMA,
                label=f"phase {phase} receipt",
            )
        for name in (
            "r4",
            "r2",
            "r2_completion",
            "r2_external",
            "r3",
            "r3_completion",
            "r3_external",
        ):
            path = auxiliary_paths.get(name)
            summary = auxiliary.get(name)
            if path is None or not isinstance(summary, Mapping):
                raise R8Error(f"{name} evidence reference is missing")
            value, _state = _read_json(path, label=f"{name} evidence reference")
            evidence["auxiliary"][name] = _evidence_ref(
                path,
                artifact_id=summary.get("artifact_id"),
                file_sha256=summary.get("file_sha256"),
                seal_sha256=value.get("seal_sha256"),
                schema=str(value.get("schema")),
                label=f"{name} evidence",
            )
        return evidence
    except (OSError, R8Error, TypeError):
        return {
            "status": "unavailable",
            "source_root": None,
            "observation": None,
            "r7": None,
            "auxiliary": {},
            "phases": {},
        }


def _false_report(reason: str, *, source_commit: str) -> dict[str, Any]:
    reasons = {
        reason,
        "r4_artifact_missing",
        "r2_artifact_missing",
        "r2_completion_missing",
        "r2_external_missing",
        "r3_artifact_missing",
        "r3_completion_missing",
        "r3_external_missing",
    }
    return {
        "captured_at": "1970-01-01T00:00:00+00:00",
        "source": {"source_commit": source_commit, "source_tree_sha256": None, "source_clean": False},
        "r7": {"status": "invalid", "reason": reason},
        "auxiliary": {},
        "inventory": {"ox": {}, "pointers": {}, "legacy": {}, "locks": {}, "safety": {}},
        "evidence": {
            "status": "unavailable",
            "source_root": None,
            "observation": None,
            "r7": None,
            "auxiliary": {},
            "phases": {},
        },
        "phases": [
            {
                "name": phase["name"],
                "order": phase["order"],
                "prerequisites": list(phase["prerequisites"]),
                "rollback_artifact": phase["rollback_artifact"],
                "receipt_supplied": False,
                "receipt_valid": False,
                "status": "blocked",
                "blocked_by": [reason, "sealed_phase_receipt_missing_or_invalid"],
            }
            for phase in PHASES
        ],
        "reasons": sorted(reasons),
        "cleanup_authorized": False,
        "cleanup_performed": False,
        "provider_calls": 0,
        "production_write_performed": False,
    }


def validate_readiness(
    *,
    source_root: Path,
    source_commit: str,
    r7_artifact: Path,
    production_observation: Path | Mapping[str, Any] | None = None,
    r4_artifact: Path | None = None,
    r2_artifact: Path | None = None,
    r2_completion: Path | None = None,
    r2_external: Path | None = None,
    r3_artifact: Path | None = None,
    r3_completion: Path | None = None,
    r3_external: Path | None = None,
    test_only: bool = False,
) -> dict[str, Any]:
    """Validate readiness without publishing or mutating any path."""

    source = _source_identity(source_root, source_commit)
    r7_value, r7_state = _read_r7_artifact(
        r7_artifact, source_root=source_root, source_commit=source_commit
    )
    r7_sha256 = _hash_file(r7_artifact, r7_state, label="R7 artifact")
    r7_summary = _validate_r7_summary(
        r7_value,
        source_commit=source_commit,
        source_root=source_root,
        file_sha256=r7_sha256,
        r7_artifact=r7_artifact,
    )
    if isinstance(production_observation, Mapping):
        if not test_only:
            raise R8Error("plain production observations require --test-only")
        observation = dict(production_observation)
        observation_state = None
    else:
        observation, observation_state = _read_observation(
            production_observation, source_root=source_root
        )
    observation_source = observation.get("source_snapshots")
    if observation_source is not None:
        if not isinstance(observation_source, Mapping):
            raise R8Error("production observation source snapshots are invalid")
        snapshots = [observation_source.get(name) for name in ("before", "after", "final")]
        if any(not isinstance(snapshot, Mapping) for snapshot in snapshots):
            raise R8Error("production observation source snapshots are incomplete")
        for snapshot in snapshots:
            if not isinstance(snapshot, Mapping):
                raise R8Error("production observation source snapshots are incomplete")
            if (
                snapshot.get("source_commit") != source_commit
                or snapshot.get("source_tree_sha256") != source["source_tree_sha256"]
                or snapshot.get("source_clean") not in {True, "true"}
            ):
                raise R8Error("production observation/source actual identity mismatch")
    observation_sealed = observation.pop("_sealed", False)
    inventory, inventory_reasons = _inventory_summary(observation)
    if observation_sealed is not True:
        inventory_reasons.append("production_observation_unsealed")
    if test_only:
        inventory_reasons.append("test_only_evidence")
    auxiliary, auxiliary_reasons = _validate_auxiliary(
        source_root=source_root,
        source_commit=source_commit,
        r4_artifact=r4_artifact,
        r2_artifact=r2_artifact,
        r2_completion=r2_completion,
        r2_external=r2_external,
        r3_artifact=r3_artifact,
        r3_completion=r3_completion,
        r3_external=r3_external,
    )
    all_reasons = inventory_reasons + auxiliary_reasons
    phases, phase_reasons = _phase_summary(
        observation,
        all_reasons,
        source_commit=source_commit,
        inventory_sha256=_digest(inventory),
        source_root=source_root,
        r7_ready=True,
    )
    reasons = sorted(set(all_reasons + phase_reasons))
    reasons = sorted(set(reasons))
    authorized = (
        not reasons
        and all(item["status"] == "ready" for item in phases)
    )
    observation_path = production_observation if isinstance(production_observation, Path) else None
    evidence = _build_evidence(
        source_root=source_root,
        r7_artifact=r7_artifact,
        r7_summary=r7_summary,
        observation_path=observation_path,
        observation=observation,
        auxiliary_paths={
            "r4": r4_artifact,
            "r2": r2_artifact,
            "r2_completion": r2_completion,
            "r2_external": r2_external,
            "r3": r3_artifact,
            "r3_completion": r3_completion,
            "r3_external": r3_external,
        },
        auxiliary=auxiliary,
    )
    if evidence.get("status") != "bound":
        reasons = sorted(set(reasons) | {"immutable_evidence_references_unavailable"})
        authorized = False
    return {
        "captured_at": r7_value.get("captured_at", "1970-01-01T00:00:00+00:00"),
        "source": source,
        "r7": r7_summary,
        "auxiliary": auxiliary,
        "evidence": evidence,
        "inventory": inventory,
        "observation_state": observation_state,
        "phases": phases,
        "reasons": reasons,
        "cleanup_authorized": authorized,
        "cleanup_performed": False,
        "provider_calls": 0,
        "production_write_performed": False,
    }


def _artifact_file_state(path: Path) -> dict[str, Any]:
    state = _file_state(path, label="R8 artifact")
    return {"file_state": state, "sha256": _hash_file(path, state, label="R8 artifact")}


def _observation_boundary(path: Path) -> object:
    if path.is_dir():
        return {
            "file_state": _file_state(path, label="production observation root"),
            "inventory_sha256": _digest(_observe_directory(path)),
        }
    state = _file_state(path, label="production observation")
    return {
        "file_state": state,
        "sha256": _hash_file(path, state, label="production observation"),
    }


def _input_boundary(path: Path, *, label: str) -> object:
    if path.is_dir():
        return {
            "file_state": _file_state(path, label=label),
            "inventory_sha256": _digest(_observe_directory(path)),
        }
    state = _file_state(path, label=label)
    return {"file_state": state, "sha256": _hash_file(path, state, label=label)}


def _r7_live_reference_path(path: Path) -> Path | None:
    """Derive the fixed external live-receipt path without trusting a path field."""

    try:
        value, _state = _read_json(path, label="R7 artifact reference")
    except (OSError, R8Error):
        return None
    artifact_id = value.get("live_attestation_artifact_id")
    if not isinstance(artifact_id, str) or _SHA.fullmatch(artifact_id) is None:
        return None
    candidate = path.parent / "r7-live-attestations" / f"{artifact_id}.json"
    try:
        if candidate.is_file() and not _has_symlink_component(candidate):
            return candidate
    except OSError:
        return None
    return None


def _observation_phase_reference_paths(path: Path | None) -> dict[str, Path]:
    """Discover phase receipt files for publication TOCTOU probes."""

    if path is None or not path.is_file():
        return {}
    try:
        value, _state = _read_json(path, label="production observation references")
    except (OSError, R8Error):
        return {}
    production = value.get("production")
    if not isinstance(production, Mapping):
        return {}
    receipts = production.get("phase_receipts")
    if not isinstance(receipts, Mapping):
        return {}
    result: dict[str, Path] = {}
    for name in _R8_PHASE_NAMES:
        raw_path = receipts.get(name, {}).get("path") if isinstance(receipts.get(name), Mapping) else None
        if isinstance(raw_path, str) and raw_path.startswith("/"):
            result[name] = Path(raw_path)
    return result


def _write_immutable(
    output: Path,
    payload: Mapping[str, Any],
    *,
    before_publish: Callable[[], None] | None = None,
) -> tuple[str, Path, dict[str, Any]]:
    if _has_symlink_component(output) or (output.exists() and not output.is_dir()):
        raise R8Error("output path is unsafe")
    if any(key in payload for key in ("schema", "namespace", "artifact_id", "seal_sha256")):
        raise R8Error("R8 reserved fields are caller-controlled")
    output.mkdir(parents=True, exist_ok=True)
    if any(path.name.endswith((".pyc", ".pyo")) or path.name == "__pycache__" for path in output.iterdir()):
        raise R8Error("output contains bytecode/cache")
    unsigned = {"schema": R8_SCHEMA, "namespace": "recall-distillation", **payload}
    artifact_id = _digest(unsigned)
    artifact = {"artifact_id": artifact_id, **unsigned}
    artifact["seal_sha256"] = _digest(artifact)
    encoded = _canonical(artifact) + b"\n"
    path = output / f"{artifact_id}.json"
    if path.is_symlink():
        raise R8Error("R8 artifact path is a symlink")
    if path.exists():
        try:
            if path.read_bytes() != encoded:
                raise R8Error("immutable artifact conflict")
        except OSError as exc:
            raise R8Error("immutable artifact readback failed") from exc
    else:
        temporary: Path | None = None
        directory_fd: int | None = None
        published = False
        try:
            try:
                directory_fd = os.open(
                    output,
                    os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
                )
                directory_before = os.fstat(directory_fd)
            except OSError as exc:
                raise R8Error("R8 output directory cannot be opened safely") from exc
            with tempfile.NamedTemporaryFile(dir=output, prefix=f".{path.name}.", delete=False) as handle:
                temporary = Path(handle.name)
                os.fchmod(handle.fileno(), 0o600)
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            if before_publish is not None:
                before_publish()
            directory_after = os.fstat(directory_fd)
            if (
                directory_after.st_dev != directory_before.st_dev
                or directory_after.st_ino != directory_before.st_ino
            ):
                raise R8Error("R8 output directory changed during publication")
            os.replace(
                temporary.name,
                path.name,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
            )
            published = True
            directory_final = os.fstat(directory_fd)
            output_final = _file_state(output, label="R8 output directory")
            if (
                directory_final.st_dev != directory_before.st_dev
                or directory_final.st_ino != directory_before.st_ino
                or output_final["st_dev"] != directory_before.st_dev
                or output_final["st_ino"] != directory_before.st_ino
            ):
                raise R8Error("R8 output directory changed after publication")
        except Exception:
            if published:
                _discard_new_artifact(path, artifact)
            raise
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
            if directory_fd is not None:
                os.close(directory_fd)
    return artifact_id, path, artifact


def _discard_new_artifact(path: Path, artifact: Mapping[str, Any]) -> None:
    """Remove only an artifact this invocation just published on TOCTOU drift."""

    try:
        if _has_symlink_component(path) or not path.is_file():
            return
        encoded = _canonical(artifact) + b"\n"
        if path.read_bytes() == encoded:
            path.unlink()
    except OSError:
        # A drift failure remains a hard failure; never broaden cleanup to an
        # output directory or any protected/production root.
        return


def _reject_sensitive_recursive(value: object, *, label: str = "artifact") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if isinstance(key, str) and _SENSITIVE.search(key) and not key.endswith(("_sha256", "_digest")):
                raise R8Error(f"{label} contains a sensitive field")
            _reject_sensitive_recursive(item, label=f"{label}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_sensitive_recursive(item, label=f"{label}[{index}]")


_EVIDENCE_REF_KEYS = {"path", "artifact_id", "file_sha256", "seal_sha256", "schema"}
_EVIDENCE_REF_SCHEMAS = {
    "r7": R7_SCHEMA,
    "observation": R8_OBSERVATION_SCHEMA,
    "r4": "chronovisor.recall-r4.v1",
    "r2": "chronovisor.recall-r2.v1",
    "r2_completion": "chronovisor.recall-distillation.v1",
    "r2_external": "chronovisor.recall-r2-external-receipt.v1",
    "r3": "chronovisor.recall-r3.v1",
    "r3_completion": "chronovisor.recall-r3-completion.v1",
    "r3_external": "chronovisor.recall-r3-external-receipt.v1",
    **{phase: R8_PHASE_RECEIPT_SCHEMA for phase in _R8_PHASE_NAMES},
}


def _validate_evidence_shape(value: object) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "status", "source_root", "observation", "r7", "auxiliary", "phases"
    }:
        raise R8Error("R8 evidence reference section is not closed")
    status = value.get("status")
    if status not in {"bound", "unavailable"}:
        raise R8Error("R8 evidence reference status is invalid")
    if status == "unavailable":
        if value.get("source_root") is not None or value.get("r7") is not None or value.get("observation") is not None:
            raise R8Error("R8 unavailable evidence contains bound references")
        if value.get("auxiliary") != {} or value.get("phases") != {}:
            raise R8Error("R8 unavailable evidence contains nested references")
        return
    source_root = value.get("source_root")
    if not isinstance(source_root, str) or not source_root.startswith("/"):
        raise R8Error("R8 evidence source root is invalid")
    for name in ("r7", "observation"):
        ref = value.get(name)
        if not isinstance(ref, Mapping) or set(ref) != _EVIDENCE_REF_KEYS:
            raise R8Error(f"R8 evidence {name} reference is invalid")
    auxiliary = value.get("auxiliary")
    if not isinstance(auxiliary, Mapping) or set(auxiliary) != {
        "r4", "r2", "r2_completion", "r2_external", "r3", "r3_completion", "r3_external"
    }:
        raise R8Error("R8 evidence auxiliary references are incomplete")
    phases = value.get("phases")
    if not isinstance(phases, Mapping) or set(phases) != set(_R8_PHASE_NAMES):
        raise R8Error("R8 evidence phase references are incomplete")
    for name, ref in [
        ("r7", value.get("r7")),
        ("observation", value.get("observation")),
        *[(str(key), item) for key, item in auxiliary.items()],
        *[(str(key), item) for key, item in phases.items()],
    ]:
        if not isinstance(ref, Mapping) or set(ref) != _EVIDENCE_REF_KEYS:
            raise R8Error(f"R8 evidence {name} reference is invalid")
        path = ref.get("path")
        if not isinstance(path, str) or not path.startswith("/"):
            raise R8Error(f"R8 evidence {name} path is invalid")
        if Path(path).suffix not in {".json", ".receipt"}:
            raise R8Error(f"R8 evidence {name} suffix is invalid")
        _sha(ref.get("artifact_id"), f"R8 evidence {name} id")
        _sha(ref.get("file_sha256"), f"R8 evidence {name} file hash")
        _sha(ref.get("seal_sha256"), f"R8 evidence {name} seal")
        if ref.get("schema") != _EVIDENCE_REF_SCHEMAS[name]:
            raise R8Error(f"R8 evidence {name} schema is invalid")


def _validate_closed_report(value: Mapping[str, Any]) -> None:
    _reject_sensitive_recursive(value)
    _validate_evidence_shape(value.get("evidence"))
    if not isinstance(value.get("captured_at"), str):
        raise R8Error("R8 captured_at is invalid")
    source = value.get("source")
    if not isinstance(source, Mapping) or set(source) != {"source_commit", "source_tree_sha256", "source_clean"}:
        raise R8Error("R8 source schema is not closed")
    if not isinstance(source.get("source_commit"), str) or _COMMIT.fullmatch(source["source_commit"]) is None:
        raise R8Error("R8 source commit is invalid")
    if source.get("source_tree_sha256") is not None:
        _sha(source.get("source_tree_sha256"), "R8 source tree")
    if not isinstance(source.get("source_clean"), bool):
        raise R8Error("R8 source clean flag is invalid")
    r7 = value.get("r7")
    if not isinstance(r7, Mapping):
        raise R8Error("R8 R7 section is invalid")
    if r7.get("status") == "invalid":
        if set(r7) != {"status", "reason"} or not isinstance(r7.get("reason"), str):
            raise R8Error("R8 invalid R7 section is not closed")
    else:
        expected_r7 = {
            "artifact_id",
            "file_sha256",
            "certification",
            "stages",
            "forced_rollback",
            "active_lkg",
            "source_commit",
            "live_attestation",
            "live_attestation_id",
            "live_attestation_sha256",
            "live_attestation_seal_sha256",
            "live_attestation_source_commit",
            "live_attestation_run_id",
            "live_attestation_stage100_artifact_id",
            "live_attestation_rollback_artifact_id",
            "collector_artifact_id",
            "collector_file_sha256",
            "collector_seal_sha256",
            "rollback_artifact_id",
            "rollback_file_sha256",
            "rollback_seal_sha256",
        }
        if set(r7) != expected_r7:
            raise R8Error("R8 R7 section is not closed")
        _sha(r7.get("artifact_id"), "R8 R7 artifact id")
        _sha(r7.get("file_sha256"), "R8 R7 file hash")
        if (
            r7.get("certification") is not True
            or r7.get("forced_rollback") is not True
            or r7.get("active_lkg") is not True
            or r7.get("live_attestation") is not True
            or r7.get("source_commit") != source["source_commit"]
        ):
            raise R8Error("R8 R7 summary is invalid")
        _sha(r7.get("live_attestation_id"), "R8 R7 live attestation id")
        _sha(r7.get("live_attestation_sha256"), "R8 R7 live attestation file hash")
        _sha(r7.get("live_attestation_seal_sha256"), "R8 R7 live attestation seal")
        if r7.get("live_attestation_source_commit") != source["source_commit"]:
            raise R8Error("R8 R7 live source binding is invalid")
        for field in (
            "live_attestation_run_id",
            "live_attestation_stage100_artifact_id",
            "live_attestation_rollback_artifact_id",
            "collector_artifact_id",
            "collector_file_sha256",
            "collector_seal_sha256",
            "rollback_artifact_id",
            "rollback_file_sha256",
            "rollback_seal_sha256",
        ):
            _sha(r7.get(field), f"R8 R7 {field}")
    inventory = value.get("inventory")
    if not isinstance(inventory, Mapping) or set(inventory) != {"ox", "pointers", "legacy", "locks", "safety"}:
        raise R8Error("R8 inventory schema is not closed")
    for section in ("ox", "pointers", "legacy", "locks", "safety"):
        if not isinstance(inventory[section], Mapping):
            raise R8Error(f"R8 inventory {section} is invalid")
    phases = value.get("phases")
    if not isinstance(phases, list) or len(phases) != len(PHASES):
        raise R8Error("R8 phase list is invalid")
    phase_keys = {"name", "order", "prerequisites", "rollback_artifact", "receipt_supplied", "receipt_valid", "status", "blocked_by"}
    for item, expected in zip(phases, PHASES, strict=True):
        if not isinstance(item, Mapping) or set(item) != phase_keys:
            raise R8Error("R8 phase schema is not closed")
        if item.get("name") != expected["name"] or type(item.get("order")) is not int or item.get("order") != expected["order"] or item.get("rollback_artifact") != expected["rollback_artifact"]:
            raise R8Error("R8 phase order is invalid")
        if item.get("prerequisites") != list(expected["prerequisites"]):
            raise R8Error("R8 phase prerequisites are invalid")
        if item.get("status") not in {"ready", "blocked"} or not isinstance(item.get("receipt_supplied"), bool) or not isinstance(item.get("receipt_valid"), bool):
            raise R8Error("R8 phase status is invalid")
        if not isinstance(item.get("blocked_by"), list) or any(not isinstance(reason, str) for reason in item["blocked_by"]):
            raise R8Error("R8 phase blockers are invalid")
    reasons = value.get("reasons")
    if not isinstance(reasons, list) or any(not isinstance(reason, str) for reason in reasons):
        raise R8Error("R8 reasons are invalid")
    if value.get("observation_state") is not None and _safe_file_state(value["observation_state"]) is None:
        raise R8Error("R8 observation state is invalid")
    auxiliary = value.get("auxiliary")
    if not isinstance(auxiliary, Mapping):
        raise R8Error("R8 auxiliary section is invalid")
    allowed_aux = {"r4", "r2", "r2_completion", "r2_external", "r3", "r3_completion", "r3_external"}
    if any(not isinstance(key, str) or key not in allowed_aux for key in auxiliary):
        raise R8Error("R8 auxiliary schema is not closed")
    auxiliary_keys = {
        "r4": {"artifact_id", "file_sha256", "seal_sha256", "source_commit", "provider_calls", "production_passed", "workset_identity"},
        "r2": {"artifact_id", "file_sha256", "seal_sha256", "source_commit", "full_rebuild", "catalog_parity", "production_unchanged", "cleanup_remaining", "parity_projection_sha256"},
        "r2_completion": {"artifact_id", "file_sha256", "seal_sha256", "r2_artifact_id", "r2_artifact_seal_sha256", "r2_artifact_sha256", "source_commit"},
        "r2_external": {"artifact_id", "file_sha256", "seal_sha256", "source_commit", "main_artifact_id", "main_artifact_sha256", "completion_file_sha256", "cleanup_remaining"},
        "r3": {"artifact_id", "file_sha256", "seal_sha256", "source_commit", "leased", "duplicates", "production_unchanged", "cleanup_remaining", "completion_readback_verified", "completion_main_artifact_id", "completion_main_artifact_sha256"},
        "r3_completion": {"artifact_id", "file_sha256", "seal_sha256", "main_artifact_id", "main_artifact_sha256", "source_commit", "readback_verified"},
        "r3_external": {"artifact_id", "file_sha256", "seal_sha256", "source_commit", "main_artifact_id", "main_artifact_sha256", "completion_artifact_id", "completion_file_sha256"},
    }
    for key, item in auxiliary.items():
        if not isinstance(item, Mapping):
            raise R8Error(f"R8 auxiliary {key} is invalid")
        if set(item) != auxiliary_keys[key]:
            raise R8Error(f"R8 auxiliary {key} schema is not closed")
        for field in ("artifact_id", "file_sha256", "source_commit"):
            if field in item:
                if field == "source_commit" and item[field] != source["source_commit"]:
                    raise R8Error(f"R8 auxiliary {key} source binding is invalid")
                if field != "source_commit":
                    _sha(item[field], f"R8 auxiliary {key}.{field}")
        if key == "r4":
            if item.get("provider_calls") != 0 or item.get("production_passed") is not True:
                raise R8Error("R8 auxiliary r4 production/provider gate is invalid")
            _sha(item.get("workset_identity"), "R8 auxiliary r4 Workset identity")
        elif key == "r2":
            if any(item.get(field) is not True for field in ("full_rebuild", "catalog_parity", "production_unchanged")):
                raise R8Error("R8 auxiliary r2 completion gate is invalid")
            if item.get("cleanup_remaining") != 0:
                raise R8Error("R8 auxiliary r2 cleanup is invalid")
            _sha(item.get("parity_projection_sha256"), "R8 auxiliary r2 parity projection")
        elif key == "r2_completion":
            for field in ("r2_artifact_id", "r2_artifact_seal_sha256", "r2_artifact_sha256"):
                _sha(item.get(field), f"R8 auxiliary r2 completion {field}")
        elif key == "r2_external":
            for field in ("main_artifact_id", "main_artifact_sha256", "completion_file_sha256"):
                _sha(item.get(field), f"R8 auxiliary r2 external {field}")
            if item.get("cleanup_remaining") != 0:
                raise R8Error("R8 auxiliary r2 external cleanup is invalid")
        elif key == "r3":
            if item.get("leased") != 0 or item.get("duplicates") != 0 or item.get("production_unchanged") is not True or item.get("cleanup_remaining") != 0 or item.get("completion_readback_verified") is not True:
                raise R8Error("R8 auxiliary r3 completion gate is invalid")
            _sha(item.get("completion_main_artifact_id"), "R8 auxiliary r3 completion artifact")
            _sha(item.get("completion_main_artifact_sha256"), "R8 auxiliary r3 completion file")
        elif key == "r3_completion":
            if item.get("readback_verified") is not True:
                raise R8Error("R8 auxiliary r3 completion readback is invalid")
            _sha(item.get("main_artifact_id"), "R8 auxiliary r3 completion main id")
            _sha(item.get("main_artifact_sha256"), "R8 auxiliary r3 completion main hash")
        elif key == "r3_external":
            _sha(item.get("main_artifact_id"), "R8 auxiliary r3 external main id")
            _sha(item.get("main_artifact_sha256"), "R8 auxiliary r3 external main hash")
            _sha(item.get("completion_artifact_id"), "R8 auxiliary r3 external completion id")
            _sha(item.get("completion_file_sha256"), "R8 auxiliary r3 external completion hash")
    for key in ("cleanup_authorized", "cleanup_performed", "production_write_performed"):
        if not isinstance(value.get(key), bool):
            raise R8Error(f"R8 {key} is not boolean")
    if value.get("provider_calls") != 0 or isinstance(value.get("provider_calls"), bool) or not isinstance(value.get("provider_calls"), int):
        raise R8Error("R8 provider_calls is invalid")


def _derive_cleanup_authorized(value: Mapping[str, Any]) -> bool:
    """Derive the authorization bit from independently closed report facts.

    ``reasons`` and phase status are diagnostics, not an authority by
    themselves.  This second projection intentionally requires every
    certified section, receipt, and quiescence field so a caller cannot
    reseal a blocked report with only ``cleanup_authorized=true``.
    """

    evidence = value.get("evidence")
    if not isinstance(evidence, Mapping) or evidence.get("status") != "bound":
        return False
    source = value.get("source")
    if not isinstance(source, Mapping) or source.get("source_clean") is not True:
        return False
    r7 = value.get("r7")
    if not isinstance(r7, Mapping) or r7.get("status") == "invalid":
        if isinstance(r7, Mapping):
            return False
        return False
    if any(
        r7.get(key) is not True
        for key in ("certification", "forced_rollback", "active_lkg", "live_attestation")
    ):
        return False
    inventory = value.get("inventory")
    if not isinstance(inventory, Mapping):
        return False
    safety = inventory.get("safety")
    if (
        not isinstance(safety, Mapping)
        or safety.get("enabled") is not False
        or safety.get("provider_calls") != 0
        or safety.get("leased") != 0
        or safety.get("process_lock") is not False
        or safety.get("process_count") != 0
    ):
        return False
    phases = value.get("phases")
    if (
        not isinstance(phases, list)
        or len(phases) != len(PHASES)
        or any(
            not isinstance(item, Mapping)
            or item.get("status") != "ready"
            or item.get("receipt_supplied") is not True
            or item.get("receipt_valid") is not True
            or item.get("blocked_by") != []
            for item in phases
        )
    ):
        return False
    auxiliary = value.get("auxiliary")
    required_auxiliary = {
        "r4",
        "r2",
        "r2_completion",
        "r2_external",
        "r3",
        "r3_completion",
        "r3_external",
    }
    if not isinstance(auxiliary, Mapping) or set(auxiliary) != required_auxiliary:
        return False
    for item in auxiliary.values():
        if not isinstance(item, Mapping) or item.get("source_commit") != source.get("source_commit"):
            return False
    reasons = value.get("reasons")
    return not (not isinstance(reasons, list) or reasons)


def _read_bound_ref(
    ref: Mapping[str, Any], *, label: str, expected_schema: str
) -> tuple[Path, dict[str, Any], str, str]:
    path_value = ref.get("path")
    if not isinstance(path_value, str) or not path_value.startswith("/"):
        raise R8Error(f"{label} path is invalid")
    path = Path(path_value)
    if _has_symlink_component(path) or not path.is_file():
        raise R8Error(f"{label} path is unavailable")
    artifact, state = _read_json(path, label=label)
    if ref.get("schema") != expected_schema or artifact.get("schema") != expected_schema:
        raise R8Error(f"{label} schema mismatch")
    actual_file_sha = _hash_file(path, state, label=label)
    if actual_file_sha is None or actual_file_sha != ref.get("file_sha256"):
        raise R8Error(f"{label} file hash does not match bytes")
    if path.name.endswith(".completion.receipt"):
        actual_id = _digest(artifact)
        actual_seal = _sha(artifact.get("seal_sha256"), f"{label} seal")
    else:
        actual_id, _unused_file_sha, actual_seal = _bind_artifact_identity(
            path, artifact, label=label
        )
    if actual_id != ref.get("artifact_id") or actual_seal != ref.get("seal_sha256"):
        raise R8Error(f"{label} reference identity mismatch")
    return path, artifact, actual_file_sha, actual_seal


def _revalidate_bound_evidence(value: Mapping[str, Any]) -> None:
    """Recompute a bound report exclusively from its producer files."""

    evidence = value.get("evidence")
    if not isinstance(evidence, Mapping) or evidence.get("status") != "bound":
        raise R8Error("R8 authoritative evidence references are missing")
    source_root_value = evidence.get("source_root")
    if not isinstance(source_root_value, str):
        raise R8Error("R8 evidence source root is missing")
    source_root = Path(source_root_value)
    source_commit = value["source"]["source_commit"]
    source = _source_identity(source_root, source_commit)
    if value.get("source") != source:
        raise R8Error("R8 source identity is not an actual readback")

    r7_ref = evidence.get("r7")
    if not isinstance(r7_ref, Mapping):
        raise R8Error("R8 R7 evidence reference is missing")
    r7_path, _r7_raw, r7_file_sha, _r7_seal = _read_bound_ref(
        r7_ref, label="R7 evidence", expected_schema=_EVIDENCE_REF_SCHEMAS["r7"]
    )
    r7_value, r7_state = _read_r7_artifact(
        r7_path, source_root=source_root, source_commit=source_commit
    )
    if _hash_file(r7_path, r7_state, label="R7 evidence") != r7_file_sha:
        raise R8Error("R7 evidence changed during revalidation")
    r7_summary = _validate_r7_summary(
        r7_value,
        source_commit=source_commit,
        source_root=source_root,
        file_sha256=r7_file_sha,
        r7_artifact=r7_path,
    )
    if value.get("r7") != r7_summary:
        raise R8Error("R8 R7 summary is not derived from actual evidence")

    auxiliary_refs = evidence.get("auxiliary")
    if not isinstance(auxiliary_refs, Mapping):
        raise R8Error("R8 auxiliary evidence references are missing")
    aux_paths: dict[str, Path | None] = {}
    for name in (
        "r4", "r2", "r2_completion", "r2_external", "r3", "r3_completion", "r3_external"
    ):
        ref = auxiliary_refs.get(name)
        if not isinstance(ref, Mapping):
            raise R8Error(f"R8 {name} evidence reference is missing")
        path, _artifact, _file_sha, _seal = _read_bound_ref(
            ref,
            label=f"{name} evidence",
            expected_schema=_EVIDENCE_REF_SCHEMAS[name],
        )
        aux_paths[name] = path
    auxiliary, auxiliary_reasons = _validate_auxiliary(
        source_root=source_root,
        source_commit=source_commit,
        r4_artifact=aux_paths["r4"],
        r2_artifact=aux_paths["r2"],
        r2_completion=aux_paths["r2_completion"],
        r2_external=aux_paths["r2_external"],
        r3_artifact=aux_paths["r3"],
        r3_completion=aux_paths["r3_completion"],
        r3_external=aux_paths["r3_external"],
    )
    if auxiliary_reasons or auxiliary != value.get("auxiliary"):
        raise R8Error("R8 auxiliary summary is not derived from actual evidence")

    observation_ref = evidence.get("observation")
    if not isinstance(observation_ref, Mapping):
        raise R8Error("R8 production observation reference is missing")
    observation_path, _observation_raw, _observation_file_sha, _observation_seal = _read_bound_ref(
        observation_ref,
        label="production observation",
        expected_schema=_EVIDENCE_REF_SCHEMAS["observation"],
    )
    observation, observation_state = _read_observation(
        observation_path, source_root=source_root
    )
    if value.get("observation_state") != observation_state:
        raise R8Error("R8 observation state is not an actual readback")
    observation_source = observation.get("source_snapshots")
    if not isinstance(observation_source, Mapping):
        raise R8Error("R8 observation source snapshots are missing")
    for name in ("before", "after", "final"):
        snapshot = observation_source.get(name)
        if (
            not isinstance(snapshot, Mapping)
            or snapshot.get("source_commit") != source_commit
            or snapshot.get("source_tree_sha256") != source["source_tree_sha256"]
            or snapshot.get("source_clean") not in {True, "true"}
        ):
            raise R8Error("R8 observation/source identity is not bound")
    observation.pop("_sealed", None)
    phase_refs_from_observation = observation.get("phase_receipts")
    evidence_phase_refs = evidence.get("phases")
    if not isinstance(phase_refs_from_observation, Mapping) or not isinstance(evidence_phase_refs, Mapping):
        raise R8Error("R8 phase receipt references are missing")
    if any(
        phase_refs_from_observation.get(name) != evidence_phase_refs.get(name)
        for name in _R8_PHASE_NAMES
    ):
        raise R8Error("R8 phase receipt reference binding is invalid")
    inventory, inventory_reasons = _inventory_summary(observation)
    phases, phase_reasons = _phase_summary(
        observation,
        inventory_reasons + auxiliary_reasons,
        source_commit=source_commit,
        inventory_sha256=_digest(inventory),
        source_root=source_root,
        r7_ready=True,
    )
    expected_reasons = sorted(
        set(inventory_reasons + auxiliary_reasons + phase_reasons)
    )
    if value.get("inventory") != inventory:
        raise R8Error("R8 inventory is not derived from actual production readback")
    if value.get("phases") != phases or value.get("reasons") != expected_reasons:
        raise R8Error("R8 phases/reasons are not derived from actual evidence")


def read_artifact(path: Path) -> dict[str, Any]:
    """Read back and verify one R8 artifact, including closed schema."""

    if path.suffix != ".json":
        raise R8Error("R8 artifact suffix is invalid")
    value, state = _read_json(path, label="R8 artifact")
    expected = {
        "artifact_id", "schema", "namespace", "seal_sha256", "captured_at",
        "source", "r7", "evidence", "inventory", "observation_state", "phases", "reasons",
        "auxiliary",
        "cleanup_authorized", "cleanup_performed", "provider_calls", "production_write_performed",
    }
    if set(value) != expected:
        raise R8Error("R8 artifact schema is not closed")
    if path.stem != value.get("artifact_id"):
        raise R8Error("R8 artifact filename identity mismatch")
    artifact_id = _sha(value.get("artifact_id"), "R8 artifact id")
    unsigned = {key: item for key, item in value.items() if key not in {"artifact_id", "seal_sha256"}}
    if artifact_id != _digest(unsigned):
        raise R8Error("R8 artifact identity mismatch")
    _verify_sealed(value, schema=R8_SCHEMA, label="R8 artifact")
    raw = path.read_bytes()
    if raw not in {_canonical(value), _canonical(value) + b"\n"}:
        raise R8Error("R8 artifact is not canonical")
    if _file_state(path, label="R8 artifact") != state:
        raise R8Error("R8 artifact changed during readback")
    _validate_closed_report(value)
    if value["cleanup_performed"] is not False or value["production_write_performed"] is not False:
        raise R8Error("R8 artifact records a forbidden mutation")
    evidence = value.get("evidence")
    if isinstance(evidence, Mapping) and evidence.get("status") == "unavailable":
        if value.get("cleanup_authorized") is True or value.get("reasons") == []:
            raise R8Error("R8 cleanup authorization/report summary lacks immutable evidence references")
    else:
        _revalidate_bound_evidence(value)
    derived_authorized = _derive_cleanup_authorized(value)
    if value["cleanup_authorized"] is not derived_authorized:
        raise R8Error("R8 cleanup authorization does not match the closed evidence projection")
    return value


def run(
    *,
    source_root: Path,
    source_commit: str,
    r7_artifact: Path,
    output: Path,
    production_observation: Path | Mapping[str, Any] | None = None,
    r4_artifact: Path | None = None,
    r2_artifact: Path | None = None,
    r2_completion: Path | None = None,
    r2_external: Path | None = None,
    r3_artifact: Path | None = None,
    r3_completion: Path | None = None,
    r3_external: Path | None = None,
    test_only: bool = False,
) -> dict[str, Any]:
    """Publish exactly one immutable readiness artifact in ``output``."""

    observation_path = (
        production_observation if isinstance(production_observation, Path) else None
    )
    auxiliary_paths = tuple(
        path
        for path in (
            r4_artifact,
            r2_artifact,
            r2_completion,
            r2_external,
            r3_artifact,
            r3_completion,
            r3_external,
        )
        if path is not None
    )
    r7_live_path = _r7_live_reference_path(r7_artifact)
    observation_phase_paths = _observation_phase_reference_paths(observation_path)
    all_input_paths = auxiliary_paths + (
        (r7_live_path,) if r7_live_path is not None else ()
    ) + tuple(observation_phase_paths.values())
    _assert_paths(source_root, r7_artifact, output, observation_path, all_input_paths)
    input_paths = {
        "r7": r7_artifact,
        "r7_live": r7_live_path,
        "r4": r4_artifact,
        "r2": r2_artifact,
        "r2_completion": r2_completion,
        "r2_external": r2_external,
        "r3": r3_artifact,
        "r3_completion": r3_completion,
        "r3_external": r3_external,
    }
    input_paths.update({f"phase_{name}": path for name, path in observation_phase_paths.items()})
    input_before: dict[str, object | None] = {}
    for name, path in input_paths.items():
        if path is None:
            input_before[name] = None
        elif path.exists():
            input_before[name] = _input_boundary(path, label=f"{name} input")
        else:
            input_before[name] = None
    source_before: dict[str, Any] | None = None
    r7_before: dict[str, int] | None = None
    observation_before: object = None
    if r7_artifact.is_file():
        r7_before = _file_state(r7_artifact, label="R7 artifact")
    if observation_path is not None and observation_path.exists():
        observation_before = _observation_boundary(observation_path)
    try:
        source_before = _source_identity(source_root, source_commit)
        report = validate_readiness(
            source_root=source_root,
            source_commit=source_commit,
            r7_artifact=r7_artifact,
            production_observation=production_observation,
            r4_artifact=r4_artifact,
            r2_artifact=r2_artifact,
            r2_completion=r2_completion,
            r2_external=r2_external,
            r3_artifact=r3_artifact,
            r3_completion=r3_completion,
            r3_external=r3_external,
            test_only=test_only,
        )
    except R8Error as exc:
        report = _false_report(str(exc), source_commit=source_commit)
        # Preserve bounded safety reasons even when an earlier formal receipt
        # is invalid.  This is diagnostic only; it can never turn readiness
        # true and never copies observation payloads.
        try:
            if isinstance(production_observation, Mapping):
                diagnostic_observation = dict(production_observation)
            elif observation_path is not None and observation_path.is_dir():
                diagnostic_observation = _observe_directory(observation_path)
            else:
                diagnostic_observation = {}
            diagnostic_observation.pop("_sealed", None)
            _inventory, diagnostic_reasons = _inventory_summary(diagnostic_observation)
            report["inventory"] = _inventory
            report["reasons"] = sorted(set(report["reasons"]) | set(diagnostic_reasons))
        except (OSError, R8Error):
            pass
    # The publication boundary is itself read-only evidence.  A drifted input
    # cannot authorize cleanup; it is recorded as a failed sealed artifact.
    if source_before is not None:
        try:
            source_after = _source_identity(source_root, source_commit)
        except R8Error:
            source_after = None
        if source_after != source_before:
            report = _false_report("source_changed_during_publication", source_commit=source_commit)
        else:
            report["source"] = source_before
    if observation_path is not None and observation_before is not None:
        try:
            observation_after = _observation_boundary(observation_path)
        except (OSError, R8Error):
            observation_after = None
        if observation_after != observation_before:
            report = _false_report("observation_changed_during_publication", source_commit=source_commit)
    for name, path in input_paths.items():
        before = input_before[name]
        if path is None:
            continue
        if before is None:
            if path.exists() or path.is_symlink():
                report = _false_report(f"{name}_appeared_during_publication", source_commit=source_commit)
            continue
        try:
            after = _input_boundary(path, label=f"{name} input")
        except (OSError, R8Error):
            after = None
        if after != before:
            report = _false_report(f"{name}_changed_during_publication", source_commit=source_commit)
    payload = {
        "captured_at": str(report.get("captured_at", "1970-01-01T00:00:00+00:00")),
        "source": report["source"],
        "r7": report["r7"],
        "auxiliary": report.get("auxiliary", {}),
        "evidence": report.get(
            "evidence",
            {
                "status": "unavailable",
                "source_root": None,
                "observation": None,
                "r7": None,
                "auxiliary": {},
                "phases": {},
            },
        ),
        "inventory": report["inventory"],
        "observation_state": report.get("observation_state"),
        "phases": report["phases"],
        "reasons": report["reasons"],
        "cleanup_authorized": report["cleanup_authorized"],
        "cleanup_performed": False,
        "provider_calls": 0,
        "production_write_performed": False,
    }
    expected_unsigned = {"schema": R8_SCHEMA, "namespace": "recall-distillation", **payload}
    expected_id = _digest(expected_unsigned)
    artifact_was_present = (output / f"{expected_id}.json").exists()
    artifact_id, artifact_path, artifact = _write_immutable(output, payload)
    try:
        readback = read_artifact(artifact_path)
        if readback != artifact:
            raise R8Error("R8 artifact readback mismatch")
        # One final input probe after publication.  Any drift invalidates the
        # newly-created artifact; an artifact that existed before this run is
        # never removed or overwritten.
        if source_before is not None and _source_identity(source_root, source_commit) != source_before:
            raise R8Error("source changed after R8 artifact publication")
        if r7_before is not None and _file_state(r7_artifact, label="R7 artifact") != r7_before:
            raise R8Error("R7 artifact changed after R8 artifact publication")
        if observation_path is not None and observation_before is not None:
            final_state = _observation_boundary(observation_path)
            if final_state != observation_before:
                raise R8Error("observation changed after R8 artifact publication")
        for name, path in input_paths.items():
            before = input_before[name]
            if path is None:
                continue
            if before is None:
                if path.exists() or path.is_symlink():
                    raise R8Error(f"{name} input appeared after R8 artifact publication")
                continue
            if _input_boundary(path, label=f"{name} input") != before:
                raise R8Error(f"{name} input changed after R8 artifact publication")
    except Exception:
        if not artifact_was_present:
            _discard_new_artifact(artifact_path, artifact)
        raise
    return {"schema": R8_SCHEMA, "artifact_id": artifact_id, "path": str(artifact_path), **readback}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--r7-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--r4-artifact", type=Path)
    parser.add_argument("--r2-artifact", type=Path)
    parser.add_argument("--r2-completion", type=Path)
    parser.add_argument("--r2-external", type=Path)
    parser.add_argument("--r3-artifact", type=Path)
    parser.add_argument("--r3-completion", type=Path)
    parser.add_argument("--r3-external", type=Path)
    parser.add_argument(
        "--production-observation",
        "--production-read-only-observation",
        "--observation",
        dest="production_observation",
        type=Path,
    )
    parser.add_argument("--test-only", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    try:
        result = run(
            source_root=args.source_root,
            source_commit=args.source_commit,
            r7_artifact=args.r7_artifact,
            output=args.output,
            production_observation=args.production_observation,
            r4_artifact=args.r4_artifact,
            r2_artifact=args.r2_artifact,
            r2_completion=args.r2_completion,
            r2_external=args.r2_external,
            r3_artifact=args.r3_artifact,
            r3_completion=args.r3_completion,
            r3_external=args.r3_external,
            test_only=args.test_only,
        )
    except (R8Error, OSError, ValueError) as exc:
        print(f"r8 readiness failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({key: result[key] for key in ("schema", "artifact_id", "path", "cleanup_authorized")}, sort_keys=True))
    return 0 if result["cleanup_authorized"] is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
