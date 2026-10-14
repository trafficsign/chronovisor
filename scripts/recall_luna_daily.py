#!/usr/bin/env python3
"""Capture, prepare, inspect, and import the private daily native-Luna bundle.

The daily Codex automation owns the native worker calls.  This script owns the
private source boundary and the repository-owned candidate writer: it advances
the catalog, freezes the current candidate ledger, reuses the current grouped
split plan, creates an immutable private bundle when the source is complete,
and imports a completed bundle into the teacher-only ledger.  Source text is
resolved only in memory while capture or bundle work runs; all durable receipts
and stdout contain counts and hashes, never utterances, questions, or
rationales.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import re
import stat
import subprocess
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# The source checkout is intentionally importable when this script is invoked
# directly by the Codex automation instead of through an installed console
# entrypoint.
# ruff: noqa: E402, I001
from chronovisor.core.canonical_json import canonical_json_sha256_strict
from chronovisor.recall import recall_distillation as distill
from chronovisor.recall import recall_distillation_catalog as catalog
from chronovisor.recall import recall_distillation_native as native
from chronovisor.recall import recall_distillation_store as store
from chronovisor.recall.recall_distillation_luna import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_SHARD_COUNT,
    DEFAULT_TEACHER_PROFILE,
    bundle_status,
    create_bundle,
)


SCHEMA = "chronovisor.recall-luna-daily.v1"
NATIVE_COHORT = "native-luna-backfill-v1"
MAX_INPUT_BYTES = 12_000
PROBE_CAP = 100
DEFAULT_ROOT = Path.home() / ".chronovisor"
DEFAULT_BULK_ROOT = DEFAULT_ROOT / "runtime" / "recall-luna-bulk"
DEFAULT_DAY = datetime.now(ZoneInfo("Asia/Tokyo")).strftime("%Y-%m-%d")
DEFAULT_DAY_ROOT = DEFAULT_BULK_ROOT / DEFAULT_DAY
DEFAULT_BUNDLE = DEFAULT_DAY_ROOT / "bundle"
DEFAULT_RECEIPT = DEFAULT_DAY_ROOT / "daily-status.json"
DEFAULT_CAPTURE_PROGRESS = (
    DEFAULT_DAY_ROOT / "inventory" / "candidate-capture-progress.json"
)
_SOURCE_SPLITS = frozenset({"train", "validation", "test", "embargo"})
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}\Z")
_CAPTURE_STALE_AFTER_SECONDS = 15 * 60
_CAPTURE_PROFILE = "native-luna-single-v1"
_CAPTURE_BACKFILL_REVISION = "age-bands-v1"


class DailyError(RuntimeError):
    """A daily preparation or import operation cannot advance safely."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _private_json(path: Path) -> dict[str, Any]:
    try:
        state = path.lstat()
        if stat.S_ISLNK(state.st_mode) or not stat.S_ISREG(state.st_mode):
            raise DailyError("private receipt is not a regular file")
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DailyError("private receipt cannot be read") from exc
    if not isinstance(value, dict):
        raise DailyError("private receipt must be an object")
    return value


def _write_private_json(path: Path, value: Mapping[str, Any]) -> None:
    path = path.expanduser()
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise DailyError("private receipt parent is unsafe")
    os.chmod(path.parent, 0o700)
    if path.exists() and (path.is_symlink() or not path.is_file()):
        raise DailyError("private receipt path is unsafe")
    data = json.dumps(
        dict(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8") + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _error_fields(error: BaseException) -> dict[str, str]:
    return {
        "error_type": type(error).__name__,
        "error_message_sha256": _sha256_bytes(
            str(error).encode("utf-8", "replace")
        ),
    }


def _native_model_cohort_sha256(root: Path = DEFAULT_ROOT) -> str:
    """Return the canonical cohort identity used by grouped split receipts.

    Split plans and the training gate both derive this value through
    ``_active_training_cohort``.  Reimplementing the hash here would create a
    permanently incompatible plan as soon as the native contract changes.
    """

    root = root.expanduser().resolve()
    contract_id = native.current_contract_id(root)
    _rows, cohort = distill._active_training_cohort(  # type: ignore[attr-defined]
        [],
        teacher_profile=DEFAULT_TEACHER_PROFILE,
        profile_contract_id=contract_id,
    )
    value = str(cohort.get("cohort_sha256") or "")
    if not _SHA256.fullmatch(value):
        raise DailyError("native_model_cohort_unavailable")
    return value


def _catalog_identity(root: Path) -> dict[str, Any]:
    path = catalog.catalog_path(root)
    checkpoint = catalog._read_catalog_checkpoint(root)  # type: ignore[attr-defined]
    if checkpoint is None:
        raise DailyError("catalog_checkpoint_missing")
    return {
        "sha256": _sha256_file(path),
        "bytes": path.stat().st_size,
        "watermark": str(checkpoint.get("catalog_watermark") or ""),
        "event_rowid": int(checkpoint.get("event_rowid") or 0),
        "lineage": str(checkpoint.get("catalog_lineage") or ""),
    }


def _candidate_identity(root: Path) -> dict[str, Any]:
    ledger = store.distillation_dir(root) / "candidate-ledger.jsonl"
    connection = catalog._connect(root)  # type: ignore[attr-defined]
    try:
        state = catalog._candidate_index_state_row(connection)  # type: ignore[attr-defined]
    finally:
        connection.close()
    if state is None:
        raise DailyError("candidate_index_state_missing")
    return {
        "sha256": _sha256_file(ledger),
        "bytes": ledger.stat().st_size,
        "record_count": int(state["record_count"]),
        "head_sha256": str(state["head_sha256"]),
        "mtime_ns": int(state["ledger_mtime_ns"]),
        "ctime_ns": int(state["ledger_ctime_ns"]),
    }


def _capture_progress(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    value = _private_json(path)
    status = str(value.get("status") or "unknown")
    if status not in {
        "starting",
        "running",
        "complete",
        "error",
        "stalled",
        "blocked_lock",
    }:
        status = "unknown"
    result: dict[str, Any] = {
        "status": status,
        "captured": int(value.get("captured") or 0),
        "total_rallies": int(value.get("total_rallies") or 0),
        "remaining": int(value.get("remaining") or 0),
        "candidate_index_count": int(value.get("candidate_index_count") or 0),
    }
    for key in (
        "pid",
        "started_at_unix",
        "updated_at_unix",
        "finished_at_unix",
        "error_class",
        "error_message_sha256",
        "reason",
    ):
        if key in value:
            result[key] = value[key]
    return result


def _capture_pid_alive(pid: object) -> bool:
    try:
        value = int(pid)
    except (TypeError, ValueError):
        return False
    if value <= 0:
        return False
    try:
        os.kill(value, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _capture_lock_busy(root: Path) -> bool | None:
    """Check the authoritative writer lock without retaining it."""

    try:
        handle = store.acquire_nonblocking_lock(
            store.distillation_dir(root) / "distillation-worker.lock"
        )
    except (OSError, store.DistillationStoreError):
        # A lock that cannot be inspected is treated as active.  Starting a
        # second writer is less recoverable than waiting for the next tick.
        return None
    if handle is None:
        return True
    store.release_lock(handle)
    return False


def _capture_owner_alive(root: Path, path: Path, progress: Mapping[str, Any]) -> bool:
    """Validate both the recorded PID and the append-only writer lock."""

    lock_busy = _capture_lock_busy(root)
    if lock_busy is True or lock_busy is None:
        return True
    # A running capture must retain the lock.  A live PID with a released lock
    # is an orphaned receipt (or a process that has already stopped writing).
    if progress.get("status") == "starting" and _capture_pid_alive(progress.get("pid")):
        return True
    # Older receipts did not carry a PID.  Give a fresh receipt a short grace
    # period for process startup, but make an old orphan visible and resumable.
    timestamp = progress.get("updated_at_unix") or progress.get("started_at_unix")
    try:
        updated_at = float(timestamp)
    except (TypeError, ValueError):
        try:
            updated_at = path.stat().st_mtime
        except OSError:
            updated_at = 0.0
    return (time.time() - updated_at) < _CAPTURE_STALE_AFTER_SECONDS


def _mark_capture_stalled(path: Path, progress: Mapping[str, Any]) -> dict[str, Any]:
    stalled = {
        **dict(progress),
        "status": "stalled",
        "reason": "capture_owner_stale",
        "finished_at_unix": time.time(),
    }
    _write_private_json(path, stalled)
    return _capture_progress(path) or stalled


def _active_capture_progress(
    root: Path, requested: Path
) -> tuple[Path, dict[str, Any] | None]:
    """Find an active capture receipt across a midnight/date-root rotation."""

    requested = requested.expanduser()
    progress = _capture_progress(requested)
    stalled_path: Path | None = None
    if progress and progress.get("status") in {"starting", "running"}:
        if _capture_owner_alive(root, requested, progress):
            return requested, progress
        progress = _mark_capture_stalled(requested, progress)
        stalled_path = requested
    bulk_root = root / "runtime" / "recall-luna-bulk"
    if bulk_root.name != "recall-luna-bulk" or not bulk_root.is_dir():
        return requested, progress
    try:
        relative = requested.relative_to(bulk_root)
        receipt_suffix = Path(*relative.parts[1:])
    except ValueError:
        receipt_suffix = Path(requested.name)
    for day_root in sorted(bulk_root.iterdir(), reverse=True):
        if not day_root.is_dir() or day_root.is_symlink():
            continue
        candidate = day_root / receipt_suffix
        if candidate == requested:
            continue
        candidate_progress = _capture_progress(candidate)
        if candidate_progress and candidate_progress.get("status") in {
            "starting",
            "running",
        }:
            if _capture_owner_alive(root, candidate, candidate_progress):
                return candidate, candidate_progress
            progress = _mark_capture_stalled(candidate, candidate_progress)
            stalled_path = candidate
    return stalled_path or requested, progress


def _safe_candidate_ids(snapshot: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    candidates = snapshot.get("candidates")
    if not isinstance(candidates, list):
        return []
    return [
        candidate
        for candidate in candidates
        if isinstance(candidate, Mapping)
        and _IDENTIFIER.fullmatch(str(candidate.get("candidate_id") or ""))
    ]


def _selected_candidates(snapshot: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    candidates = _safe_candidate_ids(snapshot)
    if len(candidates) <= 3:
        return candidates
    return [*candidates[:3], candidates[-1]]


def _repeat_pair_id(rally_id: str, candidate_id: str) -> str:
    return canonical_json_sha256_strict(
        {
            "schema": "chronovisor.recall-luna-blind-repeat.v1",
            "rally_id": rally_id,
            "candidate_id": candidate_id,
        }
    )


def _payload(
    rally: Mapping[str, Any],
    candidate: Mapping[str, Any],
    texts: Mapping[str, str],
) -> dict[str, Any] | None:
    return distill._teacher_payload(  # type: ignore[attr-defined]
        rally,
        candidate,
        texts,
        max_input_bytes=MAX_INPUT_BYTES,
        include_context=True,
    )


def _sha256_or_empty(value: object) -> str:
    text = str(value or "")
    return text if _SHA256.fullmatch(text) else ""


def _context_hashes(rally: Mapping[str, Any]) -> list[str]:
    refs = rally.get("context_refs")
    if not isinstance(refs, list):
        return []
    return [
        value
        for ref in refs
        if isinstance(ref, Mapping)
        for value in [_sha256_or_empty(ref.get("semantic_sha256"))]
        if value
    ]


def _feature_metadata(candidate: Mapping[str, Any]) -> dict[str, float | int]:
    """Keep only the numeric feature subset accepted by the native importer."""

    value = candidate.get("features")
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, float | int] = {}
    for key, item in value.items():
        if (
            isinstance(key, str)
            and _IDENTIFIER.fullmatch(key) is not None
            and not isinstance(item, bool)
            and isinstance(item, (int, float))
            and math.isfinite(float(item))
        ):
            result[key] = item
    return result


def _blind_payload(
    rally: Mapping[str, Any],
    target: Mapping[str, Any],
    alternate: Mapping[str, Any],
    target_payload: Mapping[str, Any],
    texts: Mapping[str, str],
    *,
    blind_order: str,
) -> dict[str, Any] | None:
    """Build one blind request with two displayed candidates and one target.

    The target candidate remains the same source row in both requests.  Only
    the order of the two displayed candidates changes; the worker therefore
    receives a real counter-order pair rather than two copies of one input.
    """

    alternate_payload = _payload(rally, alternate, texts)
    if alternate_payload is None:
        return None
    ordered = (
        (target, target_payload)
        if blind_order == "a_first"
        else (alternate, alternate_payload)
    )
    second = (
        (alternate, alternate_payload)
        if blind_order == "a_first"
        else (target, target_payload)
    )
    result = {
        **dict(target_payload),
        "target_candidate_id": str(target["candidate_id"]),
        "candidates": [
            {
                "candidate_id": str(candidate["candidate_id"]),
                "evidence": str(payload.get("candidate") or ""),
                "display_position": position,
            }
            for position, (candidate, payload) in enumerate((ordered, second))
        ],
    }
    # Reserve the complete two-candidate display before fitting recent context.
    # Both orders remove the same oldest context entries; query and candidate
    # text are never truncated to manufacture an eligible probe.
    result["context"] = list(target_payload.get("context") or [])
    while len(
        json.dumps(result, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ) > MAX_INPUT_BYTES:
        if not result["context"]:
            return None
        result["context"].pop(0)
    return result


def _base_row(
    rally: Mapping[str, Any],
    candidate: Mapping[str, Any],
    payload: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    assignment: str,
    *,
    probe: bool = False,
    repeat_pair_id: str = "",
    repeat_base_sha256: str = "",
    fixed_repeat: bool = False,
    order_swap: bool = False,
    blind_order: str = "",
    candidate_position: int | None = None,
) -> dict[str, Any]:
    rally_id = str(rally["rally_id"])
    candidate_id = str(candidate["candidate_id"])
    candidate_text_sha256 = _sha256_or_empty(candidate.get("text_sha256"))
    source_position = int(candidate.get("rank") or 0)
    display_position = source_position if candidate_position is None else int(candidate_position)
    assignment_row: dict[str, Any] = {
        "revision": native.NATIVE_ASSIGNMENT_REVISION,
        "profile": DEFAULT_TEACHER_PROFILE,
        "split": assignment,
        "probe": probe,
        "owner": "native-luna",
    }
    if probe:
        assignment_row.update(
            {
                "repeat_pair_id": repeat_pair_id,
                "fixed_repeat": fixed_repeat,
                "order_swap": order_swap,
                "blind_order": blind_order,
                "candidate_position": display_position,
            }
        )
    return {
        "rally_id": rally_id,
        "rally_key": rally_id,
        "candidate_id": candidate_id,
        "as_of": str(rally.get("as_of") or ""),
        "as_of_us": int(rally["as_of_us"]),
        "input": dict(payload),
        "split": assignment,
        "group_id": str(rally.get("session_cluster_id") or rally_id),
        "assignment": assignment_row,
        "features": _feature_metadata(candidate),
        "snapshot_sha256": str(snapshot["snapshot_sha256"]),
        "candidate_snapshot_sha256": str(snapshot["snapshot_sha256"]),
        "source_snapshot_sha256": str(snapshot["snapshot_sha256"]),
        "query_sha256": _sha256_or_empty(rally.get("query_sha256")),
        "context_sha256": _context_hashes(rally),
        "candidate_text_sha256": candidate_text_sha256,
        "candidate_position": source_position,
        **({"repeat_pair_id": repeat_pair_id} if repeat_pair_id else {}),
        **(
            {"repeat_base_sha256": repeat_base_sha256}
            if repeat_base_sha256
            else {}
        ),
        **({"fixed_repeat": fixed_repeat} if probe else {}),
        **({"order_swap": order_swap} if probe else {}),
        **({"blind_order": blind_order} if probe else {}),
    }


def _build_rows(
    rallies: Sequence[Mapping[str, Any]],
    snapshots: Mapping[str, Mapping[str, Any]],
    assignments: Mapping[str, str],
    texts: Mapping[str, str],
    *,
    excluded_pairs: set[tuple[str, str]] | None = None,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    rows: list[dict[str, Any]] = []
    skipped = Counter[str]()
    excluded_pairs = excluded_pairs or set()
    probe_candidates: list[
        tuple[Mapping[str, Any], Mapping[str, Any], list[Mapping[str, Any]]]
    ] = []
    for rally in rallies:
        rally_id = str(rally["rally_id"])
        assignment = str(assignments.get(rally_id) or "")
        if assignment not in _SOURCE_SPLITS:
            skipped["invalid_split"] += 1
            continue
        snapshot = snapshots.get(rally_id)
        if snapshot is None:
            skipped["missing_snapshot"] += 1
            continue
        selected = _selected_candidates(snapshot)
        for candidate in selected:
            if (rally_id, str(candidate["candidate_id"])) in excluded_pairs:
                skipped["already_labeled"] += 1
                continue
            payload = _payload(rally, candidate, texts)
            if payload is None:
                skipped["payload_over_limit_or_text_missing"] += 1
                continue
            rows.append(_base_row(rally, candidate, payload, snapshot, assignment))
        if assignment == "test":
            safe: list[tuple[int, str, Mapping[str, Any], Mapping[str, Any]]] = []
            for candidate in sorted(
                _safe_candidate_ids(snapshot),
                key=lambda item: str(item["candidate_id"]),
            ):
                candidate_id = str(candidate["candidate_id"])
                if (rally_id, candidate_id) in excluded_pairs:
                    continue
                payload = _payload(rally, candidate, texts)
                if payload is None:
                    continue
                candidate_input = {
                    "candidate_id": candidate["candidate_id"],
                    "rally_id": rally_id,
                    "query": payload["query"],
                    "context": [],
                    "evidence": payload["candidate"],
                }
                encoded_size = len(
                    json.dumps(
                        candidate_input,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ).encode()
                )
                safe.append((encoded_size, str(candidate["candidate_id"]), candidate, payload))
            if len(safe) >= 2:
                safe.sort(key=lambda item: (item[0], item[1]))
                probe_candidates.append(
                    (rally, snapshot, [safe[0][2], safe[1][2]])
                )

    probe_candidates.sort(
        key=lambda item: (int(item[0]["as_of_us"]), str(item[0]["rally_id"]))
    )
    probe_rallies = 0
    for rally, snapshot, candidates in probe_candidates:
        rally_probes: list[dict[str, Any]] = []
        assignment = str(assignments[str(rally["rally_id"])])
        rally_id = str(rally["rally_id"])
        if len(candidates) < 2:
            continue
        for target_index, candidate in enumerate(candidates):
            alternate = candidates[1 - target_index]
            payload = _payload(rally, candidate, texts)
            if payload is None:
                skipped["probe_payload_over_limit_or_text_missing"] += 2
                continue
            repeat_base = canonical_json_sha256_strict(
                {
                    "schema": "chronovisor.recall-luna-blind-repeat-base.v2",
                    "rally_id": rally_id,
                    "target_candidate_id": str(candidate["candidate_id"]),
                    "candidate_ids": sorted(
                        [str(candidate["candidate_id"]), str(alternate["candidate_id"])]
                    ),
                    "query_sha256": _sha256_or_empty(rally.get("query_sha256")),
                    "context_sha256": _context_hashes(rally),
                }
            )
            pair_id = _repeat_pair_id(
                rally_id, str(candidate["candidate_id"])
            )
            for blind_order in ("a_first", "b_first"):
                blind_input = _blind_payload(
                    rally,
                    candidate,
                    alternate,
                    payload,
                    texts,
                    blind_order=blind_order,
                )
                if blind_input is None:
                    skipped["probe_payload_over_limit_or_text_missing"] += 1
                    continue
                display_position = 0 if blind_order == "a_first" else 1
                rally_probes.append(
                    _base_row(
                        rally,
                        candidate,
                        blind_input,
                        snapshot,
                        assignment,
                        probe=True,
                        repeat_pair_id=pair_id,
                        repeat_base_sha256=repeat_base,
                        fixed_repeat=True,
                        order_swap=True,
                        blind_order=blind_order,
                        candidate_position=display_position,
                    )
                )
        if len(rally_probes) == 4:
            rows.extend(rally_probes)
            probe_rallies += 1
            if probe_rallies >= PROBE_CAP:
                break
        else:
            skipped["probe_incomplete_rally_discarded"] += len(rally_probes)
    return rows, skipped


def _read_current_split_plan(root: Path) -> dict[str, Any] | None:
    try:
        plan = distill._read_split_plan(root)  # type: ignore[attr-defined]
    except Exception:
        return None
    if not isinstance(plan, dict):
        return None
    assignments = plan.get("assignments")
    if not isinstance(assignments, Mapping):
        return None
    if any(value not in _SOURCE_SPLITS for value in assignments.values()):
        return None
    return dict(plan)


def _split_plan_path(root: Path, plan: Mapping[str, Any]) -> Path:
    plan_id = str(plan.get("artifact_id") or "")
    if len(plan_id) != 64 or any(char not in "0123456789abcdef" for char in plan_id):
        raise DailyError("split_plan_id_invalid")
    return store.distillation_dir(root) / "split-plans" / f"{plan_id}.json"


def _plan_metadata(
    root: Path, plan: Mapping[str, Any], rally_ids: set[str]
) -> dict[str, Any]:
    assignments = plan.get("assignments")
    if not isinstance(assignments, Mapping):
        raise DailyError("split_plan_assignments_missing")
    values = {str(key): str(value) for key, value in assignments.items()}
    counts = Counter(values.values())
    missing = rally_ids - set(values)
    return {
        "split_plan_id": str(plan.get("artifact_id") or ""),
        "model_cohort_sha256": str(plan.get("model_cohort_sha256") or ""),
        "raw_watermark": str(plan.get("raw_watermark") or ""),
        "assignment_count": len(values),
        "rally_count": len(rally_ids),
        "missing_rallies": len(missing),
        "split_counts": dict(counts),
        "path": str(_split_plan_path(root, plan)),
    }


def _resolve_split_plan(
    root: Path,
    rallies: Sequence[Mapping[str, Any]],
    *,
    explicit_path: Path | None,
    allow_write: bool,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    rally_ids = {str(rally["rally_id"]) for rally in rallies}
    plan: dict[str, Any] | None
    if explicit_path is not None:
        plan = _private_json(explicit_path)
    else:
        plan = _read_current_split_plan(root)
    if plan is None:
        if not allow_write:
            return None, {
                "status": "in_progress",
                "reason": "split_plan_pending",
                "resumable": True,
            }
        checkpoint = catalog._read_catalog_checkpoint(root)  # type: ignore[attr-defined]
        if checkpoint is None:
            raise DailyError("catalog_checkpoint_missing")
        plan = distill._ensure_split_plan(  # type: ignore[attr-defined]
            root,
            rallies,
            raw_watermark=str(checkpoint.get("catalog_watermark") or ""),
            model_cohort_sha256=_native_model_cohort_sha256(root),
        )
    metadata = _plan_metadata(root, plan, rally_ids)
    if metadata["model_cohort_sha256"] != _native_model_cohort_sha256(root):
        return None, {
            "status": "in_progress",
            "reason": "split_plan_pending_native_cohort",
            "resumable": True,
            "split_plan": metadata,
        }
    if metadata["missing_rallies"]:
        if explicit_path is not None:
            return None, {
                "status": "in_progress",
                "reason": "split_plan_missing_latest_rallies",
                "resumable": True,
                "split_plan": metadata,
            }
        # A growth receipt keeps every historical assignment (including the
        # locked test) and marks only the new groups embargo.  This is the one
        # safe split-plan mutation the daily runner may perform; a fresh
        # native cohort still requires the authoritative inventory writer.
        checkpoint = catalog._read_catalog_checkpoint(root)  # type: ignore[attr-defined]
        if checkpoint is None:
            raise DailyError("catalog_checkpoint_missing")
        plan = distill._ensure_split_plan(  # type: ignore[attr-defined]
            root,
            rallies,
            raw_watermark=str(checkpoint.get("catalog_watermark") or ""),
            model_cohort_sha256=_native_model_cohort_sha256(root),
        )
        metadata = _plan_metadata(root, plan, rally_ids)
        metadata["growth_receipt"] = True
    return plan, {"status": "ready", "split_plan": metadata}


def _bundle_metadata(path: Path) -> dict[str, Any] | None:
    manifest = path / "manifest.json"
    if not manifest.exists():
        return None
    try:
        manifest_value = _private_json(manifest)
        status = bundle_status(path)
    except Exception as exc:
        return {
            "status": "error",
            "reason": "bundle_invalid",
            **_error_fields(exc),
        }
    import_marker = path / "import-receipt.json"
    marker_imported = False
    if import_marker.exists():
        try:
            marker_imported = _private_json(import_marker).get("status") == "imported"
        except DailyError:
            marker_imported = False
    imported = bool(manifest_value.get("production_ledger_written")) or marker_imported
    completion_value: dict[str, Any] = {}
    completion_path = path / "completion.json"
    if completion_path.exists():
        try:
            completion_value = _private_json(completion_path)
        except DailyError:
            completion_value = {}
    blind_repeat = completion_value.get("blind_repeat")
    if not isinstance(blind_repeat, Mapping):
        blind_repeat = {}
    pairs = int(blind_repeat.get("pairs") or 0)
    stable = int(blind_repeat.get("stable") or 0)
    return {
        "status": str(status.get("status") or "unknown"),
        "record_count": int(status.get("record_count") or 0),
        "completed": int(status.get("completed") or 0),
        "remaining": int(status.get("remaining") or 0),
        "source_snapshot_sha256": str(status.get("source_snapshot_sha256") or ""),
        "class_ratio": dict(status.get("class_ratio") or {}),
        "verdicts": dict(status.get("verdicts") or {}),
        "blind_repeat": {
            "pairs": pairs,
            "stable": stable,
            "unstable": int(blind_repeat.get("unstable") or 0),
        },
        "agreement_rate": (stable / pairs) if pairs else None,
        "truth_authority": str(status.get("truth_authority") or ""),
        "production_ledger_written": imported,
        "import_pending": bool(status.get("remaining") == 0)
        and not imported,
    }


def _bundle_candidates(requested: Path) -> list[Path]:
    """Find immutable bundles in one private daily directory."""

    parent = requested.expanduser().parent
    search_roots = [parent]
    bulk_root = parent.parent
    if bulk_root.name == "recall-luna-bulk" and bulk_root.is_dir():
        search_roots.extend(
            path
            for path in sorted(bulk_root.iterdir())
            if path.is_dir() and not path.is_symlink() and path != parent
        )
    candidates: list[Path] = []
    for root_path in search_roots:
        paths = (
            (requested.expanduser(), *sorted(root_path.glob("bundle*")))
            if root_path == parent
            else tuple(sorted(root_path.glob("bundle*")))
        )
        for path in paths:
            if path.is_dir() and path not in candidates:
                candidates.append(path)
    return candidates


def _select_resumable_bundle(requested: Path) -> tuple[Path, dict[str, Any]] | None:
    """Prefer unfinished work, including a completed bundle awaiting import."""

    pending: list[tuple[Path, dict[str, Any]]] = []
    for path in _bundle_candidates(requested):
        metadata = _bundle_metadata(path)
        if metadata is None or metadata.get("status") == "error":
            continue
        if int(metadata.get("remaining") or 0) > 0 or metadata.get("import_pending"):
            pending.append((path, metadata))
    pending.sort(
        key=lambda item: (
            0 if int(item[1].get("remaining") or 0) > 0 else 1,
            str(item[0]),
        )
    )
    return pending[0] if pending else None


def _select_import_bundle(requested: Path) -> tuple[Path, dict[str, Any]] | None:
    """Select the import target without replacing an explicit valid bundle.

    The resumable selector is intentionally global: it can find an unfinished
    bundle in a sibling day when the scheduled path is absent.  Import callers
    that explicitly name a completed bundle need a narrower rule, though.  If
    that path has a manifest, its status (including an import-pending status)
    is authoritative and sibling discovery must not replace it.  Discovery is
    allowed only when the requested directory has no manifest.  A file,
    symlink, or malformed manifest is an unsafe request and fails closed.
    """

    requested = requested.expanduser()
    if requested.is_symlink():
        raise DailyError("requested_bundle_unsafe")
    if requested.exists():
        if not requested.is_dir():
            raise DailyError("requested_bundle_invalid")
        manifest = requested / "manifest.json"
        if manifest.is_symlink():
            raise DailyError("requested_bundle_unsafe")
        if manifest.exists():
            metadata = _bundle_metadata(requested)
            if metadata is None:
                raise DailyError("requested_bundle_invalid")
            return requested, metadata
    return _select_resumable_bundle(requested)


def _existing_labeled_pairs(root: Path) -> set[tuple[str, str]]:
    """Read only source identities from the native teacher-only ledger."""

    path = store.distillation_dir(root) / store.LABEL_LEDGER_FILE
    if not path.exists():
        return set()
    try:
        rows = store.read_chain(path)
    except store.DistillationStoreError as exc:
        raise DailyError("native_label_ledger_invalid") from exc
    pairs = {
        (str(row.get("rally_id") or ""), str(row.get("candidate_id") or ""))
        for row in rows
        if isinstance(row, Mapping)
        and row.get("profile") == DEFAULT_TEACHER_PROFILE
        and row.get("native_luna") is True
        and row.get("rally_id")
        and row.get("candidate_id")
    }
    # A completed private bundle is also an immutable label boundary.  Keep a
    # source pair out of the next daily bundle even if an idempotent importer
    # was interrupted after writing its private marker but before the ledger
    # checkpoint became visible.
    bulk_root = root / "runtime" / "recall-luna-bulk"
    day_roots = [
        path
        for path in sorted(bulk_root.iterdir())
        if path.is_dir() and not path.is_symlink()
    ] if bulk_root.is_dir() else []
    for day_root in day_roots:
        for bundle_path in sorted(day_root.glob("bundle*")):
            metadata = _bundle_metadata(bundle_path)
            if not metadata or not metadata.get("production_ledger_written"):
                continue
            source_path = bundle_path / "snapshot" / "source.jsonl"
            if not source_path.is_file() or source_path.is_symlink():
                continue
            try:
                for line in source_path.read_bytes().splitlines():
                    source = json.loads(line.decode("utf-8"))
                    if isinstance(source, Mapping) and source.get("rally_id") and source.get(
                        "candidate_id"
                    ):
                        pairs.add(
                            (str(source["rally_id"]), str(source["candidate_id"]))
                        )
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise DailyError("completed_bundle_source_invalid") from exc
    return pairs


def _new_bundle_path(requested: Path, *, seed: str) -> Path:
    parent = requested.expanduser().parent
    parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    base = parent / f"bundle-{seed[:20]}"
    path = base
    suffix = 1
    while path.exists():
        path = parent / f"{base.name}-{suffix}"
        suffix += 1
    return path


def _update_native_bundle_pointer(root: Path, bundle_path: Path) -> dict[str, Any]:
    """Point an enabled native profile at the newly frozen private bundle."""

    config_path = root / "config.toml"
    try:
        config_state = config_path.lstat()
    except OSError:
        return {"updated": False, "reason": "config_missing"}
    if stat.S_ISLNK(config_state.st_mode) or not stat.S_ISREG(config_state.st_mode):
        return {"updated": False, "reason": "config_unsafe"}
    if not config_path.is_file():
        return {"updated": False, "reason": "config_missing"}
    try:
        config = distill.load_distillation_config(config_path)
    except Exception as exc:
        return {"updated": False, "reason": "config_invalid", **_error_fields(exc)}
    if (
        config.teacher_profile != DEFAULT_TEACHER_PROFILE
        or config.native_luna_enabled is not True
    ):
        return {"updated": False, "reason": "native_luna_not_enabled"}
    try:
        original = config_path.read_text(encoding="utf-8")
    except OSError as exc:
        return {"updated": False, "reason": "config_unreadable", **_error_fields(exc)}
    lines = original.splitlines(keepends=True)
    in_distillation = False
    replaced = False
    section_end = len(lines)
    for index, line in enumerate(lines):
        section = re.match(r"^\s*\[([^]]+)\]\s*$", line)
        if section:
            if in_distillation:
                section_end = index
            in_distillation = section.group(1).strip() == "recall.distillation"
            continue
        if in_distillation and re.match(r"^\s*native_luna_bundle\s*=", line):
            newline = "\n" if line.endswith("\n") else ""
            lines[index] = f'native_luna_bundle = {json.dumps(str(bundle_path))}{newline}'
            replaced = True
            break
    if not replaced:
        insertion = f'native_luna_bundle = {json.dumps(str(bundle_path))}\n'
        lines.insert(section_end, insertion)
    data = "".join(lines).encode("utf-8")
    mode = stat.S_IMODE(config_path.stat().st_mode)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{config_path.name}.", dir=config_path.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, config_path)
        os.chmod(config_path, mode)
        directory_fd = os.open(config_path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return {
            "updated": True,
            "config_sha256": _sha256_file(config_path),
            "bundle_path": str(bundle_path),
        }
    except OSError as exc:
        return {"updated": False, "reason": "config_update_failed", **_error_fields(exc)}
    finally:
        temporary.unlink(missing_ok=True)


def _row_has_native_source_identity(row: Mapping[str, Any]) -> bool:
    required = (
        "source_snapshot_sha256",
        "candidate_snapshot_sha256",
        "query_sha256",
        "candidate_text_sha256",
    )
    if any(not _SHA256.fullmatch(str(row.get(key) or "")) for key in required):
        return False
    context = row.get("context_sha256")
    if not isinstance(context, list) or any(
        not _SHA256.fullmatch(str(value or "")) for value in context
    ):
        return False
    return bool(str(row.get("group_id") or "")) and isinstance(
        row.get("assignment"), Mapping
    )


def _base_result(command: str) -> dict[str, Any]:
    return {"schema": SCHEMA, "command": command, "resumable": True}


def status(
    *,
    root: Path = DEFAULT_ROOT,
    bundle_path: Path = DEFAULT_BUNDLE,
    capture_progress_path: Path = DEFAULT_CAPTURE_PROGRESS,
) -> dict[str, Any]:
    root = root.expanduser().resolve()
    bundle_path = bundle_path.expanduser()
    result = _base_result("status")
    try:
        selected_bundle = _select_resumable_bundle(bundle_path)
        if selected_bundle is not None:
            active_bundle_path, active_bundle = selected_bundle
            bundle_view: dict[str, Any] = {
                "path": str(active_bundle_path),
                **active_bundle,
            }
        else:
            bundle_view = {
                "path": str(bundle_path),
                **(_bundle_metadata(bundle_path) or {}),
            }
        active_progress_path, progress = _active_capture_progress(
            root, capture_progress_path
        )
        if progress and progress.get("status") in {"starting", "running"}:
            result.update(
                {
                    "status": "in_progress",
                    "reason": "authoritative_candidate_writer_active",
                    "capture_progress": progress,
                    "capture_progress_path": str(active_progress_path),
                    "bundle": bundle_view,
                }
            )
            return result
        catalog_identity = _catalog_identity(root)
        candidate_identity = _candidate_identity(root)
        rallies = catalog.rallies(root)
        result.update(
            {
                "status": "ready",
                "catalog": {
                    **catalog_identity,
                    "rallies": len(rallies),
                },
                "candidate": candidate_identity,
                "candidate_snapshot_rallies": len(catalog.candidate_rally_ids(root)),
                "capture_progress": progress,
                "bundle": bundle_view,
            }
        )
    except Exception as exc:
        result.update({"status": "error", **_error_fields(exc)})
    return result


def _advance_catalog(root: Path) -> dict[str, Any]:
    """Advance the text-free catalog before candidate capture starts."""

    advanced = catalog.advance(root / "raw", root, MAX_INPUT_BYTES)
    return {
        "status": str(advanced.status),
        "watermark": str(advanced.watermark),
        "indexed_raw": len(advanced.indexed_raw_ids),
        "archived_raw": len(advanced.archived_raw_ids),
        "rallies": len(advanced.rally_ids),
        "deferred_sessions": len(advanced.deferred_session_keys),
    }


def _advance_catalog_when_idle(root: Path) -> dict[str, Any]:
    """Refresh source metadata before reusing a completed capture receipt."""

    lock = store.acquire_nonblocking_lock(
        store.distillation_dir(root) / "distillation-worker.lock"
    )
    if lock is None:
        return {
            "status": "in_progress",
            "reason": "distillation_worker_lock_busy",
            "resumable": True,
        }
    try:
        return _advance_catalog(root)
    finally:
        store.release_lock(lock)


def _capture_manifest_row(rally: Mapping[str, Any]) -> dict[str, Any]:
    """Return the text-free identity used by a capture progress digest."""

    return {
        key: rally.get(key)
        for key in (
            "rally_id",
            "as_of",
            "as_of_us",
            "host",
            "session_cluster_id",
            "source_index",
            "query_sha256",
        )
    }


def _capture_manifest_digest(rallies: Sequence[Mapping[str, Any]]) -> str:
    return canonical_json_sha256_strict(
        [_capture_manifest_row(rally) for rally in rallies]
    )


def _capture_checkpoint_identity(root: Path) -> dict[str, Any]:
    checkpoint = catalog._read_catalog_checkpoint(root)  # type: ignore[attr-defined]
    if checkpoint is None:
        raise DailyError("catalog_checkpoint_missing")
    return {
        "catalog_watermark": str(checkpoint.get("catalog_watermark") or ""),
        "catalog_event_rowid": int(checkpoint.get("event_rowid") or 0),
        "catalog_lineage": str(checkpoint.get("catalog_lineage") or ""),
    }


def _capture_index_identity(root: Path) -> dict[str, Any]:
    path = catalog.historical_index_path(root)
    checkpoint = catalog._read_index_checkpoint(path)  # type: ignore[attr-defined]
    if checkpoint is None:
        raise DailyError("historical_index_checkpoint_missing")
    return {
        "content_sha256": str(checkpoint.get("content_sha256") or ""),
        "catalog_watermark": str(checkpoint.get("catalog_watermark") or ""),
        "catalog_event_rowid": int(checkpoint.get("catalog_event_rowid") or 0),
    }


def _capture_assert_checkpoint(
    root: Path,
    expected_catalog: Mapping[str, Any],
    expected_index: Mapping[str, Any],
) -> None:
    if _capture_checkpoint_identity(root) != dict(expected_catalog):
        raise DailyError("catalog_identity_changed")
    if _capture_index_identity(root) != dict(expected_index):
        raise DailyError("historical_index_identity_changed")


def _capture_one(
    *, root: Path, rally: Mapping[str, Any], index_path: Path
) -> dict[str, Any]:
    texts = catalog.CatalogTextCache(root / "raw", root)
    query_sha = str(rally.get("query_sha256") or "")
    if not _SHA256.fullmatch(query_sha):
        raise DailyError("query_text_missing")
    texts.prefetch((query_sha,))
    query_text = texts[query_sha]
    if not query_text:
        raise DailyError("query_text_empty")
    snapshot = distill.candidate_snapshot(
        index_path,
        rally,
        query_text,
        limit=200,
        candidate_texts=texts,
    )
    if not isinstance(snapshot.get("candidates"), list):
        raise DailyError("candidate_snapshot_shape_invalid")
    return snapshot


def _capture_progress_payload(
    *,
    root: Path,
    candidate_path: Path,
    manifest_sha256: str,
    catalog_identity: Mapping[str, Any],
    index_identity: Mapping[str, Any],
    total_rallies: int,
    initial_missing: int,
) -> dict[str, Any]:
    return {
        "schema": "chronovisor.recall-luna-bulk.candidate-capture.v1",
        "status": "starting",
        "profile": _CAPTURE_PROFILE,
        "backfill_revision": _CAPTURE_BACKFILL_REVISION,
        "candidate_ledger": str(candidate_path),
        "catalog_watermark": catalog_identity["catalog_watermark"],
        "catalog_event_rowid": catalog_identity["catalog_event_rowid"],
        "catalog_lineage": catalog_identity["catalog_lineage"],
        "historical_index_sha256": index_identity["content_sha256"],
        "historical_index_catalog_event_rowid": index_identity[
            "catalog_event_rowid"
        ],
        "manifest_sha256": manifest_sha256,
        "total_rallies": total_rallies,
        "initial_missing": initial_missing,
        "captured": 0,
        "new_candidate_pairs": 0,
        "zero_candidate_snapshots": 0,
        "last_cursor": 0,
        "candidate_index_count": 0,
        "pid": os.getpid(),
        "started_at_unix": time.time(),
    }


def _capture(
    *,
    root: Path,
    private_dir: Path,
    progress_path: Path,
    chunk_size: int = 100,
    max_rallies: int | None = None,
    read_workers: int = 8,
) -> int:
    """Capture missing point-in-time candidate snapshots as the sole writer."""

    if chunk_size < 1 or chunk_size > 500:
        raise ValueError("chunk_size must be between 1 and 500")
    if read_workers < 1 or read_workers > 32:
        raise ValueError("read_workers must be between 1 and 32")
    root = root.expanduser().resolve()
    private_dir = private_dir.expanduser()
    progress_path = progress_path.expanduser()
    private_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    candidate_path = store.distillation_dir(root) / "candidate-ledger.jsonl"
    lock = store.acquire_nonblocking_lock(
        store.distillation_dir(root) / "distillation-worker.lock"
    )
    if lock is None:
        _write_private_json(
            progress_path,
            {
                "schema": "chronovisor.recall-luna-bulk.candidate-capture.v1",
                "status": "blocked_lock",
                "profile": _CAPTURE_PROFILE,
                "pid": os.getpid(),
                "reason": "distillation_worker_lock_busy",
                "finished_at_unix": time.time(),
            },
        )
        return 2

    try:
        # The lock covers catalog/index synchronization and every candidate
        # append.  Raw is read at its current committed boundary; rally as_of
        # values are never rewritten during backfill.
        _advance_catalog(root)
        catalog.sync_historical_index(root / "raw", root)
        index_state = catalog.sync_candidate_index(root, candidate_path)
        rallies = catalog.rallies(root)
        rallies.sort(key=lambda row: (int(row["as_of_us"]), str(row["rally_id"])))
        catalog_identity = _capture_checkpoint_identity(root)
        historical_identity = _capture_index_identity(root)
        manifest_sha256 = _capture_manifest_digest(rallies)
        known = catalog.candidate_rally_ids(root)
        missing = [
            rally
            for rally in rallies
            if str(rally.get("rally_id") or "") not in known
        ]
        if max_rallies is not None:
            missing = missing[:max_rallies]
        progress = _capture_progress_payload(
            root=root,
            candidate_path=candidate_path,
            manifest_sha256=manifest_sha256,
            catalog_identity=catalog_identity,
            index_identity=historical_identity,
            total_rallies=len(rallies),
            initial_missing=len(missing),
        )
        progress.update(
            {
                "status": "running",
                "candidate_index_count": int(index_state.get("record_count", 0)),
                "chunk_size": chunk_size,
                "read_workers": read_workers,
                "age_boundary_utc": distill._source_age_boundary(rallies),
            }
        )
        _write_private_json(progress_path, progress)

        cursor = 0
        captured = 0
        new_pairs = 0
        zero_candidates = 0
        while cursor < len(missing):
            _capture_assert_checkpoint(root, catalog_identity, historical_identity)
            batch = missing[cursor : cursor + chunk_size]
            index_path = catalog.historical_index_path(root)
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=min(read_workers, len(batch))
            ) as pool:
                snapshots = list(
                    pool.map(
                        lambda rally, index_path=index_path: _capture_one(
                            root=root, rally=rally, index_path=index_path
                        ),
                        batch,
                    )
                )
            payloads: list[dict[str, Any]] = []
            batch_pairs = 0
            batch_zero = 0
            age_bands = distill._source_age_bands(rallies)
            age_boundary = distill._source_age_boundary(rallies)
            for rally, snapshot in zip(batch, snapshots, strict=True):
                candidates = snapshot.get("candidates")
                if not isinstance(candidates, list):
                    raise DailyError("candidate_snapshot_shape_invalid")
                batch_pairs += len(candidates)
                batch_zero += int(not candidates)
                payloads.append(
                    {
                        "kind": "candidate-snapshot",
                        "rally_id": str(rally["rally_id"]),
                        "backfill_revision": _CAPTURE_BACKFILL_REVISION,
                        "backfill_profile": _CAPTURE_PROFILE,
                        "backfill_band": age_bands.get(
                            str(rally["rally_id"]), "old-history"
                        ),
                        "age_boundary_utc": age_boundary,
                        "snapshot": snapshot,
                    }
                )
            if len(payloads) != len(batch):
                raise DailyError("capture_batch_incomplete")
            store.append_chain_batch(candidate_path, payloads)
            index_state = catalog.sync_candidate_index(root, candidate_path)
            _capture_assert_checkpoint(root, catalog_identity, historical_identity)
            cursor += len(batch)
            captured += len(batch)
            new_pairs += batch_pairs
            zero_candidates += batch_zero
            progress.update(
                {
                    "status": "running",
                    "captured": captured,
                    "new_candidate_pairs": new_pairs,
                    "zero_candidate_snapshots": zero_candidates,
                    "last_cursor": cursor,
                    "candidate_index_count": int(index_state.get("record_count", 0)),
                    "last_rally_as_of_us": int(batch[-1]["as_of_us"]),
                    "updated_at_unix": time.time(),
                }
            )
            _write_private_json(progress_path, progress)

        final_known = catalog.candidate_rally_ids(root)
        remaining = sum(
            str(rally.get("rally_id") or "") not in final_known for rally in missing
        )
        if remaining:
            raise DailyError("candidate_index_missing_after_capture")
        progress.update(
            {
                "status": "complete",
                "remaining": 0,
                "finished_at_unix": time.time(),
            }
        )
        _write_private_json(progress_path, progress)
        return 0
    except BaseException as error:
        prior: dict[str, Any] = {}
        try:
            if progress_path.exists():
                prior = _private_json(progress_path)
        except DailyError:
            prior = {}
        prior.update(
            {
                "status": "error",
                "error_class": type(error).__name__,
                "error_message_sha256": _sha256_bytes(
                    str(error).encode("utf-8", "replace")
                ),
                "finished_at_unix": time.time(),
            }
        )
        _write_private_json(progress_path, prior)
        raise
    finally:
        store.release_lock(lock)


def _capture_script_for(
    progress_path: Path | None = None, capture_script: Path | None = None
) -> Path:
    """Resolve the repository-owned capture entrypoint only."""

    script = Path(__file__).resolve()
    if capture_script is not None and capture_script.expanduser().resolve() != script:
        raise DailyError("capture_script_must_be_repo_owned")
    if not script.is_file():
        raise DailyError("capture_script_missing")
    return script


def _start_candidate_capture(
    *,
    root: Path,
    progress_path: Path,
    capture_script: Path | None = None,
) -> dict[str, Any]:
    """Start this repository script as the authoritative private capture writer."""

    _active_path, progress = _active_capture_progress(root, progress_path)
    if progress and progress.get("status") in {"starting", "running"}:
        return {"started": False, "status": "running", "reason": "capture_already_running"}
    stalled_previous = (
        dict(progress) if progress and progress.get("status") == "stalled" else None
    )
    try:
        script = _capture_script_for(capture_script=capture_script)
    except DailyError as exc:
        return {"started": False, "status": "error", **_error_fields(exc)}
    progress_path.expanduser().parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(script),
        "capture",
        "--root",
        str(root),
        "--private-dir",
        str(progress_path.expanduser().parent),
        "--capture-progress",
        str(progress_path.expanduser()),
    ]
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )
    except OSError as exc:
        return {
            "started": False,
            "status": "error",
            "reason": "capture_start_failed",
            "resumable": True,
            **_error_fields(exc),
        }
    current = _capture_progress(progress_path)
    if current is None or current.get("pid") != int(process.pid):
        _write_private_json(
            progress_path,
            {
                "schema": "chronovisor.recall-luna-bulk.candidate-capture.v1",
                "status": "starting",
                "profile": _CAPTURE_PROFILE,
                "pid": int(process.pid),
                "started_at_unix": time.time(),
            },
        )
    result: dict[str, Any] = {
        "started": True,
        "status": "starting",
        "resumable": True,
    }
    if stalled_previous is not None:
        result["stalled_previous"] = stalled_previous
    return result


def run_daily(
    *,
    root: Path = DEFAULT_ROOT,
    bundle_path: Path = DEFAULT_BUNDLE,
    receipt_path: Path = DEFAULT_RECEIPT,
    capture_progress_path: Path = DEFAULT_CAPTURE_PROGRESS,
    capture_script: Path | None = None,
    split_plan_path: Path | None = None,
    allow_split_write: bool = False,
) -> dict[str, Any]:
    """Resume pending work before starting the repository-owned capture writer."""

    root = root.expanduser().resolve()
    receipt_path = receipt_path.expanduser()
    result = _base_result("run")

    def write_and_return() -> dict[str, Any]:
        _write_private_json(receipt_path, result)
        return result

    def promote_capture_failure(capture: Mapping[str, Any]) -> dict[str, Any]:
        result["capture"] = dict(capture)
        result.update(
            {
                "status": "error",
                "reason": str(capture.get("reason") or "candidate_capture_failed"),
            }
        )
        for key in ("error_type", "error_message_sha256"):
            value = capture.get(key)
            if isinstance(value, str) and value:
                result[key] = value
        return write_and_return()

    def publish_prepared(
        prepared: Mapping[str, Any], *, capture: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        if capture is not None:
            result["capture"] = dict(capture)
        result["prepare"] = dict(prepared)
        result.update(
            {
                "status": str(prepared.get("status") or "in_progress"),
                "reason": str(prepared.get("reason") or "daily_prepare"),
            }
        )
        return write_and_return()

    try:
        # A pending bundle is the durable work queue.  Resume it before any
        # catalog/index write so a candidate writer cannot race an export.
        if _select_resumable_bundle(bundle_path) is not None:
            prepared = prepare(
                root=root,
                bundle_path=bundle_path,
                receipt_path=receipt_path,
                capture_progress_path=capture_progress_path,
                split_plan_path=split_plan_path,
                allow_split_write=allow_split_write,
            )
            result["reason"] = prepared.get("reason", "pending_bundle")
            return publish_prepared(prepared)

        active_progress_path, progress = _active_capture_progress(
            root, capture_progress_path
        )
        progress_status = str(progress.get("status") or "") if progress else ""
        if progress is not None and progress_status in {
            "error",
            "blocked_lock",
            "unknown",
        }:
            # Keep the terminal receipt as evidence, but do not let it become
            # a permanent retry loop.  A blocked-lock result is recoverable
            # once the authoritative writer releases the lock; an error or an
            # unknown terminal receipt follows the same no-owner check before
            # starting a fresh capture attempt.
            previous_capture = {
                "status": progress_status,
                "capture_progress": progress,
                "capture_progress_path": str(active_progress_path),
            }
            lock_busy = _capture_lock_busy(root)
            if lock_busy is True or lock_busy is None:
                result.update(
                    {
                        "status": "in_progress",
                        "reason": "distillation_worker_lock_busy",
                        "capture": {
                            "started": False,
                            "status": "in_progress",
                            "reason": "distillation_worker_lock_busy",
                            "previous_capture": previous_capture,
                        },
                    }
                )
                return write_and_return()
            capture = _start_candidate_capture(
                root=root,
                progress_path=capture_progress_path,
                capture_script=capture_script,
            )
            capture["previous_capture"] = previous_capture
            if capture.get("started") is True:
                prepared = {
                    **_base_result("prepare"),
                    "status": "in_progress",
                    "reason": "authoritative_candidate_writer_starting",
                    "capture": capture,
                }
                return publish_prepared(prepared, capture=capture)
            if capture.get("status") == "error" or capture.get(
                "reason"
            ) == "capture_start_failed":
                return promote_capture_failure(capture)
            return publish_prepared(
                {
                    **_base_result("prepare"),
                    "status": "in_progress",
                    "reason": str(capture.get("reason") or "capture_in_progress"),
                },
                capture=capture,
            )

        if progress is not None:
            # A completed (or stale) receipt is evidence that the previous
            # writer reached a boundary.  Refresh the catalog while idle and
            # prepare that boundary before considering another writer.
            if progress_status in {"complete", "stalled"}:
                catalog_result = _advance_catalog_when_idle(root)
                result["catalog"] = catalog_result
                if catalog_result.get("status") == "in_progress":
                    result.update(
                        {
                            "status": "in_progress",
                            "reason": str(
                                catalog_result.get(
                                    "reason", "catalog_refresh_in_progress"
                                )
                            ),
                        }
                    )
                    result["capture"] = {
                        "started": False,
                        "status": progress_status,
                        "reason": str(progress.get("reason") or progress_status),
                        "capture_progress_path": str(active_progress_path),
                    }
                    return write_and_return()
            capture = {
                "started": False,
                "status": progress_status,
                "reason": str(progress.get("reason") or progress_status),
                "capture_progress": progress,
                "capture_progress_path": str(active_progress_path),
            }
            prepared = prepare(
                root=root,
                bundle_path=bundle_path,
                receipt_path=receipt_path,
                capture_progress_path=capture_progress_path,
                split_plan_path=split_plan_path,
                allow_split_write=allow_split_write,
            )
            if (
                prepared.get("status") != "in_progress"
                or prepared.get("reason") != "missing_candidate_snapshots"
            ):
                return publish_prepared(prepared, capture=capture)
            capture = _start_candidate_capture(
                root=root,
                progress_path=capture_progress_path,
                capture_script=capture_script,
            )
            if capture.get("started") is True:
                prepared = {
                    **_base_result("prepare"),
                    "status": "in_progress",
                    "reason": "authoritative_candidate_writer_starting",
                    "capture": capture,
                }
                return publish_prepared(prepared, capture=capture)
            if capture.get("status") == "error" or capture.get("reason") == "capture_start_failed":
                return promote_capture_failure(capture)
            return publish_prepared(
                {
                    **_base_result("prepare"),
                    "status": "in_progress",
                    "reason": str(capture.get("reason") or "capture_in_progress"),
                },
                capture=capture,
            )

        capture = _start_candidate_capture(
            root=root,
            progress_path=capture_progress_path,
            capture_script=capture_script,
        )
        if capture.get("started") is True:
            # The helper publishes its own ``starting`` receipt immediately
            # after process creation.  Avoid exporting while that first write
            # is in flight; the next unattended tick resumes prepare safely.
            prepared = {
                **_base_result("prepare"),
                "status": "in_progress",
                "reason": "authoritative_candidate_writer_starting",
                "capture": capture,
            }
            return publish_prepared(prepared, capture=capture)
        if capture.get("status") == "error" or capture.get("reason") == "capture_start_failed":
            return promote_capture_failure(capture)
        prepared = prepare(
            root=root,
            bundle_path=bundle_path,
            receipt_path=receipt_path,
            capture_progress_path=capture_progress_path,
            split_plan_path=split_plan_path,
            allow_split_write=allow_split_write,
        )
        return publish_prepared(prepared, capture=capture)
    except Exception as exc:
        result.update({"status": "error", **_error_fields(exc)})
    return write_and_return()


def _export_bundle(
    *,
    root: Path,
    rallies: Sequence[Mapping[str, Any]],
    plan: Mapping[str, Any],
    output: Path,
    before_catalog: Mapping[str, Any],
    before_candidate: Mapping[str, Any],
    excluded_pairs: set[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    excluded_pairs = excluded_pairs or set()
    candidate_path = store.distillation_dir(root) / "candidate-ledger.jsonl"
    rally_ids = {str(rally["rally_id"]) for rally in rallies}
    candidate_ids = catalog.candidate_rally_ids(root)
    missing = rally_ids - candidate_ids
    if missing:
        return {
            "status": "in_progress",
            "reason": "missing_candidate_snapshots",
            "resumable": True,
            "candidate_snapshot_rallies": len(candidate_ids),
            "missing_candidate_snapshot_rallies": len(missing),
        }
    snapshots = catalog.read_candidate_snapshots(root, candidate_path, candidate_ids)
    text_cache = catalog.CatalogTextCache(root / "raw", root)
    requested: set[str] = set()
    assignments = {
        str(key): str(value)
        for key, value in (plan.get("assignments") or {}).items()
    }
    for rally in rallies:
        snapshot = snapshots[str(rally["rally_id"])]
        for candidate in _safe_candidate_ids(snapshot):
            if (
                str(rally["rally_id"]),
                str(candidate["candidate_id"]),
            ) in excluded_pairs:
                continue
            if rally.get("query_sha256"):
                requested.add(str(rally["query_sha256"]))
            if candidate.get("text_sha256"):
                requested.add(str(candidate["text_sha256"]))
        for ref in rally.get("context_refs", []):
            if isinstance(ref, Mapping) and ref.get("semantic_sha256"):
                requested.add(str(ref["semantic_sha256"]))
    text_cache.prefetch(requested)
    rows, skipped = _build_rows(
        rallies,
        snapshots,
        assignments,
        text_cache,
        excluded_pairs=excluded_pairs,
    )
    after_catalog = _catalog_identity(root)
    after_candidate = _candidate_identity(root)
    if dict(after_catalog) != dict(before_catalog):
        raise DailyError("catalog_changed_during_export")
    if dict(after_candidate) != dict(before_candidate):
        raise DailyError("candidate_ledger_changed_during_export")
    valid_rows: list[dict[str, Any]] = []
    for row in rows:
        if _row_has_native_source_identity(row):
            valid_rows.append(row)
        else:
            skipped["native_source_metadata_invalid"] += 1
    rows = valid_rows
    if not rows:
        if skipped.get("already_labeled"):
            return {
                "status": "ready",
                "resumable": True,
                "reason": "no_unlabeled_source",
                "record_count": 0,
                "skipped": dict(skipped),
                "catalog_sha256": before_catalog["sha256"],
                "candidate_ledger_sha256": before_candidate["sha256"],
                "split_plan_id": str(plan.get("artifact_id") or ""),
            }
        raise DailyError("bundle_source_empty")
    bundle = create_bundle(
        output,
        rows,
        teacher_profile=DEFAULT_TEACHER_PROFILE,
        cohort=NATIVE_COHORT,
        frozen_as_of_us=max(int(rally["as_of_us"]) for rally in rallies),
        split_plan_id=str(plan.get("artifact_id") or ""),
        shard_count=DEFAULT_SHARD_COUNT,
        batch_size=DEFAULT_BATCH_SIZE,
        metadata={
            "catalog_watermark": before_catalog["watermark"],
            "catalog_sha256": before_catalog["sha256"],
            "candidate_ledger_sha256": before_candidate["sha256"],
            "split_counts": dict(Counter(str(row["split"]) for row in rows)),
            "skipped": dict(skipped),
        },
    )
    bundle_state = bundle.status()
    return {
        "status": "pending",
        "resumable": True,
        "record_count": len(rows),
        "source_snapshot_sha256": str(bundle_state["source_snapshot_sha256"]),
        "remaining": int(bundle_state["remaining"]),
        "split_counts": dict(Counter(str(row["split"]) for row in rows)),
        "skipped": dict(skipped),
        "catalog_sha256": before_catalog["sha256"],
        "candidate_ledger_sha256": before_candidate["sha256"],
        "split_plan_id": str(plan.get("artifact_id") or ""),
    }


def prepare(
    *,
    root: Path = DEFAULT_ROOT,
    bundle_path: Path = DEFAULT_BUNDLE,
    receipt_path: Path = DEFAULT_RECEIPT,
    capture_progress_path: Path = DEFAULT_CAPTURE_PROGRESS,
    split_plan_path: Path | None = None,
    allow_split_write: bool = False,
) -> dict[str, Any]:
    root = root.expanduser().resolve()
    bundle_path = bundle_path.expanduser()
    receipt_path = receipt_path.expanduser()
    result = _base_result("prepare")
    try:
        selected = _select_resumable_bundle(bundle_path)
        if selected is not None:
            active_path, existing = selected
            result.update(existing)
            result["bundle_path"] = str(active_path)
            if existing.get("remaining"):
                result["status"] = "pending"
                result["reason"] = "existing_bundle_resumable"
            elif existing.get("import_pending"):
                result["status"] = "complete_pending_import"
                result["reason"] = "bundle_ready_for_import"
            else:
                result["status"] = "complete"
                result["reason"] = "existing_bundle_imported"
            _write_private_json(receipt_path, result)
            return result
        active_progress_path, progress = _active_capture_progress(
            root, capture_progress_path
        )
        if progress and progress.get("status") in {"starting", "running"}:
            result.update(
                {
                    "status": "in_progress",
                    "reason": "authoritative_candidate_writer_active",
                    "capture_progress": progress,
                    "capture_progress_path": str(active_progress_path),
                }
            )
            _write_private_json(receipt_path, result)
            return result
        before_catalog = _catalog_identity(root)
        before_candidate = _candidate_identity(root)
        rallies = catalog.rallies(root)
        plan, plan_result = _resolve_split_plan(
            root,
            rallies,
            explicit_path=split_plan_path,
            allow_write=allow_split_write,
        )
        if plan is None:
            result.update(plan_result)
            result.update(
                {
                    "catalog": {**before_catalog, "rallies": len(rallies)},
                    "candidate": before_candidate,
                }
            )
            _write_private_json(receipt_path, result)
            return result
        excluded_pairs = _existing_labeled_pairs(root)
        requested_metadata = _bundle_metadata(bundle_path)
        output = bundle_path
        if requested_metadata is not None and not requested_metadata.get(
            "import_pending"
        ) and not requested_metadata.get("remaining"):
            seed = canonical_json_sha256_strict(
                {
                    "catalog_sha256": before_catalog["sha256"],
                    "candidate_sha256": before_candidate["sha256"],
                    "split_plan_id": str(plan.get("artifact_id") or ""),
                    "excluded_pairs": len(excluded_pairs),
                }
            )
            output = _new_bundle_path(bundle_path, seed=seed)
        exported = _export_bundle(
            root=root,
            rallies=rallies,
            plan=plan,
            output=output,
            before_catalog=before_catalog,
            before_candidate=before_candidate,
            excluded_pairs=excluded_pairs,
        )
        exported["bundle_path"] = str(output)
        if exported.get("status") == "pending":
            exported["config_pointer"] = _update_native_bundle_pointer(root, output)
        result.update(
            {
                **exported,
                "catalog": {**before_catalog, "rallies": len(rallies)},
                "candidate": before_candidate,
                "split_plan": plan_result.get("split_plan"),
            }
        )
    except Exception as exc:
        result.update({"status": "error", **_error_fields(exc)})
    _write_private_json(receipt_path, result)
    return result


def import_completed(
    *,
    root: Path = DEFAULT_ROOT,
    bundle_path: Path = DEFAULT_BUNDLE,
    receipt_path: Path = DEFAULT_RECEIPT,
) -> dict[str, Any]:
    root = root.expanduser().resolve()
    bundle_path = bundle_path.expanduser()
    receipt_path = receipt_path.expanduser()
    result = _base_result("import")
    try:
        selected = _select_import_bundle(bundle_path)
        if selected is not None:
            selected_path, selected_metadata = selected
            bundle_path = selected_path
            result["bundle_path"] = str(bundle_path)
            if selected_metadata.get("status") == "error":
                result.update(
                    {
                        "status": "error",
                        "reason": str(
                            selected_metadata.get("reason") or "bundle_invalid"
                        ),
                        **{
                            key: selected_metadata[key]
                            for key in (
                                "error_type",
                                "error_message_sha256",
                            )
                            if key in selected_metadata
                        },
                    }
                )
                _write_private_json(receipt_path, result)
                return result
            if int(selected_metadata.get("remaining") or 0) > 0:
                result.update(
                    {
                        "status": "in_progress",
                        "reason": "bundle_incomplete",
                        "remaining": int(selected_metadata.get("remaining") or 0),
                    }
                )
                _write_private_json(receipt_path, result)
                return result
        config = distill.load_distillation_config(root / "config.toml")
        if (
            config.teacher_profile != DEFAULT_TEACHER_PROFILE
            or config.native_luna_enabled is not True
        ):
            result.update(
                {
                    "status": "in_progress",
                    "reason": "native_luna_not_enabled",
                    "teacher_profile": config.teacher_profile,
                    "native_luna_enabled": config.native_luna_enabled,
                }
            )
        else:
            contract = native.ensure_contract(root, config)
            imported = native.import_bundle(root, bundle_path)
            labels_written = int(
                imported.get("labels_written")
                or imported.get("labels_appended")
                or 0
            )
            result.update(
                {
                    "status": "imported",
                    "labels": int(imported.get("labels") or 0),
                    "relevant": int(imported.get("relevant") or 0),
                    "irrelevant": int(imported.get("irrelevant") or 0),
                    "uncertain": int(imported.get("uncertain") or 0),
                    "labels_appended": labels_written,
                    "labels_written": labels_written,
                    "remaining": int(imported.get("remaining") or 0),
                    "profile_contract_id": str(contract.get("artifact_id") or ""),
                    "source_snapshot_sha256": str(
                        imported.get("source_snapshot_sha256") or ""
                    ),
                    "label_chain_head_sha256": str(
                        imported.get("label_chain_head_sha256") or ""
                    ),
                }
            )
            # A zero-label idempotent import must not make the runtime look as
            # though fresh teacher work completed today.
            if labels_written <= 0:
                result.pop("last_label_at", None)
            elif imported.get("last_label_at"):
                result["last_label_at"] = str(imported["last_label_at"])
            _write_private_json(
                bundle_path / "import-receipt.json",
                {
                    "schema": "chronovisor.recall-luna-daily-import.v1",
                    "status": "imported",
                    "labels_written": labels_written,
                    "labels": int(imported.get("labels") or 0),
                    "source_snapshot_sha256": str(
                        imported.get("source_snapshot_sha256") or ""
                    ),
                    "label_chain_head_sha256": str(
                        imported.get("label_chain_head_sha256") or ""
                    ),
                },
            )
    except DailyError as exc:
        result.update(
            {"status": "error", "reason": str(exc), **_error_fields(exc)}
        )
    except Exception as exc:
        result.update({"status": "error", **_error_fields(exc)})
    _write_private_json(receipt_path, result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument(
        "--capture-progress", type=Path, default=DEFAULT_CAPTURE_PROGRESS
    )
    commands = parser.add_subparsers(dest="command", required=True)

    def runtime_options(command: argparse.ArgumentParser) -> None:
        # Accept both ``script --root ... run`` and the more natural
        # ``script run --root ...`` form used by scheduled jobs.
        command.add_argument("--root", type=Path, default=argparse.SUPPRESS)
        command.add_argument("--bundle", type=Path, default=argparse.SUPPRESS)
        command.add_argument("--receipt", type=Path, default=argparse.SUPPRESS)
        command.add_argument(
            "--capture-progress", type=Path, default=argparse.SUPPRESS
        )

    prepare_command = commands.add_parser("prepare")
    runtime_options(prepare_command)
    prepare_command.add_argument("--split-plan", type=Path, default=None)
    prepare_command.add_argument("--allow-split-write", action="store_true")
    run_command = commands.add_parser("run")
    runtime_options(run_command)
    run_command.add_argument("--capture-script", type=Path, default=None)
    run_command.add_argument("--split-plan", type=Path, default=None)
    run_command.add_argument("--allow-split-write", action="store_true")
    capture_command = commands.add_parser("capture")
    runtime_options(capture_command)
    capture_command.add_argument("--private-dir", type=Path, default=None)
    capture_command.add_argument("--chunk-size", type=int, default=100)
    capture_command.add_argument("--max-rallies", type=int, default=None)
    capture_command.add_argument("--read-workers", type=int, default=8)
    status_command = commands.add_parser("status")
    runtime_options(status_command)
    import_command = commands.add_parser("import")
    runtime_options(import_command)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "prepare":
        result = prepare(
            root=args.root,
            bundle_path=args.bundle,
            receipt_path=args.receipt,
            capture_progress_path=args.capture_progress,
            split_plan_path=args.split_plan,
            allow_split_write=args.allow_split_write,
        )
    elif args.command == "run":
        result = run_daily(
            root=args.root,
            bundle_path=args.bundle,
            receipt_path=args.receipt,
            capture_progress_path=args.capture_progress,
            capture_script=args.capture_script,
            split_plan_path=args.split_plan,
            allow_split_write=args.allow_split_write,
        )
    elif args.command == "capture":
        private_dir = args.private_dir or args.capture_progress.parent
        try:
            exit_code = _capture(
                root=args.root,
                private_dir=private_dir,
                progress_path=args.capture_progress,
                chunk_size=args.chunk_size,
                max_rallies=args.max_rallies,
                read_workers=args.read_workers,
            )
            result = {
                **_base_result("capture"),
                "status": "complete" if exit_code == 0 else "in_progress",
                "progress_path": str(args.capture_progress),
            }
        except Exception as exc:
            result = {**_base_result("capture"), "status": "error", **_error_fields(exc)}
    elif args.command == "status":
        result = status(
            root=args.root,
            bundle_path=args.bundle,
            capture_progress_path=args.capture_progress,
        )
        _write_private_json(args.receipt, result)
    else:
        result = import_completed(
            root=args.root,
            bundle_path=args.bundle,
            receipt_path=args.receipt,
        )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result.get("status") not in {"error"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
