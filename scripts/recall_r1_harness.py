#!/usr/bin/env python3
"""Measure the Recall R1 append/checkpoint/index hot paths on an APFS clone."""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import hashlib
import json
import os
import random
import shutil
import sqlite3
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

# Keep the harness and any source modules it imports from creating bytecode
# artifacts in the production checkout during a read-only measurement.
sys.dont_write_bytecode = True

from recall_r0_harness import (  # noqa: E402
    LEDGERS,
    POLICY_SCHEMA,
    STATE_KEYS,
    R0Error,
    _chain,
    _env,
    _filesystem_type,
    _fts,
    _load,
    _loopback_json,
    _proc_pid_rusage_v2,
    _stage_guards,
    _stat,
)

SCHEMA = "chronovisor.recall-r1.v1"
R0_SCHEMA = "chronovisor.recall-r0.v1"
R0_EVIDENCE_ID = "4de2cfe3f33e5c9c5153b264ebee8fae24d814856e0ac339e53c3077dc7efb33"
R0_EVIDENCE_RELATIVE = Path(
    "_handoff/evidence/2026-08-23-recall-distillation-recovery/"
    "r0-measured-baseline-4de2cfe3.json"
)
EXPECTED_BASELINE = {
    "candidate-ledger.jsonl": {"bytes": 1_680_839_826, "records": 8_050},
    "rally-manifest.jsonl": {"bytes": 383_284_887, "records": 8_050},
    "label-ledger.jsonl": {"bytes": 277_562, "records": 158},
}
UNIQUE_LEDGER_NAMES = (
    "exposure-receipts.jsonl",
    "outcome-receipts.jsonl",
    "negative-veto-receipts.jsonl",
    "shadow-observation-receipts.jsonl",
)
RUSAGE_SAMPLE_INTERVAL_SECONDS = 0.005


class R1Error(R0Error):
    """An R1 measurement or safety contract failed closed."""


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _static(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in snapshot.items()
        if key not in {"fast_snapshot", "live_health"}
    }


def _redact_operations(value: Any) -> Any:
    """Keep operation shape/digests while excluding returned ledger bodies."""

    if isinstance(value, Mapping):
        return {
            str(key): (
                _value_digest(item) if key == "value" else _redact_operations(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_operations(item) for item in value]
    return value


def _value_digest(value: Any) -> dict[str, Any]:
    shape = (
        "mapping"
        if isinstance(value, Mapping)
        else "list"
        if isinstance(value, list)
        else type(value).__name__
    )
    count = len(value) if isinstance(value, (Mapping, list, tuple, set)) else None
    return {
        "shape": shape,
        "count": count,
        "sha256": _digest(value),
    }


def _redact_index_state(state: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(state)
    ledger_path = result.pop("ledger_path", None)
    if isinstance(ledger_path, str):
        result["ledger_path_basename"] = Path(ledger_path).name
        result["ledger_path_sha256"] = hashlib.sha256(ledger_path.encode()).hexdigest()
    return result


def _label_health_readonly(store: Any, path: Path) -> dict[str, Any]:
    """Compute the production label count without checkpoint/projection writes."""

    before = _chain(store, path)
    try:
        rows: list[dict[str, Any]] = []
        with path.open("rb") as handle:
            for line in handle:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise R1Error("label row is not an object")
                rows.append(value)
    except (OSError, UnicodeError, ValueError, store.DistillationStoreError) as exc:
        raise R1Error("label health read failed") from exc
    after = _stat(path)
    if after != before["file_state"]:
        raise R1Error("label ledger changed during read")
    counts = {"teacher_only": 0, "verified_truth": 0, "probe_not_truth": 0}
    for row in rows:
        if row.get("authority") == "teacher-only":
            counts["teacher_only"] += 1
        elif row.get("authority") == "verified":
            counts["verified_truth"] += 1
        assignment = row.get("assignment")
        if isinstance(assignment, Mapping) and assignment.get("probe") is True:
            counts["probe_not_truth"] += 1
    if before["records"] != len(rows):
        raise R1Error("label count/head mismatch")
    return {
        "records": int(before["records"]),
        "head_sha256": str(before["head_sha256"]),
        "bytes": int(before["bytes"]),
        "counts": counts,
        "file_state": before["file_state"],
        "source": "read-only-ledger-replay",
    }


def _bounded_chain(store: Any, path: Path) -> dict[str, Any]:
    """Read a sealed chain checkpoint without opening the ledger body."""

    before = _stat(path)
    if before is None:
        raise R1Error(f"production ledger missing: {path.name}")
    checkpoint_path = store._chain_checkpoint_path(path)
    try:
        checkpoint = store.read_sealed(
            checkpoint_path, schema=store.DISTILLATION_SCHEMA
        )
    except Exception as exc:
        raise R1Error(f"production ledger checkpoint invalid: {path.name}") from exc
    after = _stat(path)
    records = checkpoint.get("records")
    head = checkpoint.get("head_sha256")
    if (
        checkpoint.get("kind") != "ledger-chain-checkpoint"
        or checkpoint.get("ledger_name") != path.name
        or not isinstance(records, int)
        or isinstance(records, bool)
        or records < 0
        or not isinstance(head, str)
        or (head and (len(head) != 64 or set(head) - set("0123456789abcdef")))
        or (records == 0) != (head == "")
        or checkpoint.get("file_state") != before
        or after != before
    ):
        raise R1Error(f"production ledger checkpoint stale: {path.name}")
    return {
        "records": int(records),
        "head_sha256": head,
        "bytes": int(before["size_bytes"]),
        "file_state": before,
    }


def _bounded_production(
    store: Any,
    catalog: Any,
    raw_store: Any,
    root: Path,
    dashboard_url: str,
) -> dict[str, Any]:
    """Capture production state while keeping large ledger bodies unread."""

    directory = store.distillation_dir(root)
    ledgers = {name: _bounded_chain(store, directory / name) for name in LEDGERS}
    try:
        watermark = raw_store.committed_raw_watermark(root / "raw")
    except Exception as exc:
        raise R1Error("committed Raw watermark invalid") from exc
    if (
        not isinstance(watermark, str)
        or len(watermark) != 64
        or set(watermark) - set("0123456789abcdef")
    ):
        raise R1Error("committed Raw watermark malformed")

    state_path = directory / store.STATE_FILE
    state_file_state = _stat(state_path)
    try:
        state = store.read_sealed(state_path, schema=store.DISTILLATION_SCHEMA)
    except Exception as exc:
        raise R1Error("distillation state invalid") from exc
    compact_state = {
        key: state.get(key)
        for key in STATE_KEYS
        if isinstance(state.get(key), (str, int, bool)) or state.get(key) is None
    }

    pointers: dict[str, Any] = {}
    for kind, filename in store.POINTER_FILES.items():
        pointer_path = directory / filename
        if _stat(pointer_path) is None:
            pointers[kind] = None
            continue
        try:
            pointer = store.read_sealed(pointer_path, schema=store.DISTILLATION_SCHEMA)
        except Exception as exc:
            raise R1Error("policy pointer invalid") from exc
        policy_id = pointer.get("policy_id")
        if (
            pointer.get("kind") != f"{kind}-policy-pointer"
            or not isinstance(policy_id, str)
            or len(policy_id) != 64
            or set(policy_id) - set("0123456789abcdef")
        ):
            raise R1Error("policy pointer identity invalid")
        try:
            policy = store.read_sealed(
                directory / "policies" / f"{policy_id}.json",
                schema=POLICY_SCHEMA,
            )
        except Exception as exc:
            raise R1Error("policy artifact invalid") from exc
        if policy.get("artifact_id") != policy_id:
            raise R1Error("policy artifact identity mismatch")
        pointers[kind] = {
            "policy_id": policy_id,
            "pointer_seal_sha256": pointer.get("seal_sha256", ""),
            "policy_seal_sha256": policy.get("seal_sha256", ""),
            "pointer_file_state": _stat(pointer_path),
            "policy_file_state": _stat(directory / "policies" / f"{policy_id}.json"),
        }

    status, body, payload = _loopback_json(dashboard_url, "/api/fast-snapshot")
    fast_snapshot = {"status": status, "payload_sha256": None}
    if payload is not None:
        fast_snapshot.update(
            {
                "payload_sha256": hashlib.sha256(body).hexdigest(),
                "top_level_keys": sorted(str(key) for key in payload),
                "events_count": (
                    len(payload["events"])
                    if isinstance(payload.get("events"), list)
                    else None
                ),
                "metrics_count": (
                    len(payload["metrics"])
                    if isinstance(payload.get("metrics"), list)
                    else None
                ),
            }
        )
    health_status, health_body, health_payload = _loopback_json(
        dashboard_url, "/api/health"
    )
    health = (
        health_payload.get("health")
        if isinstance(health_payload, Mapping)
        and isinstance(health_payload.get("health"), Mapping)
        else {}
    )
    runtime = health.get("runtime") if isinstance(health, Mapping) else {}
    runtime = runtime if isinstance(runtime, Mapping) else {}
    recall = health.get("recall_distillation") if isinstance(health, Mapping) else {}
    recall = recall if isinstance(recall, Mapping) else {}
    live_health = {
        "http_status": health_status,
        "payload_sha256": (
            hashlib.sha256(health_body).hexdigest() if health_body else None
        ),
        "status": health.get("status") if isinstance(health, Mapping) else None,
        "runtime": {
            key: runtime.get(key)
            for key in ("commit_id", "expected_commit", "drift", "package_version")
        },
        "recall_distillation": {
            key: recall.get(key)
            for key in ("status", "worker_status", "rollout", "hold_reason", "alert")
        },
    }
    return {
        "ledgers": ledgers,
        "raw_watermark": watermark,
        "fts": _fts(store, catalog, root, watermark),
        "state": {
            "seal_sha256": state.get("seal_sha256", ""),
            "fields": compact_state,
            "file_state": state_file_state,
        },
        "pointers": pointers,
        "fast_snapshot": fast_snapshot,
        "live_health": live_health,
    }


def _production_snapshot(
    store: Any,
    catalog: Any,
    raw_store: Any,
    root: Path,
    dashboard_url: str,
) -> dict[str, Any]:
    snapshot = _bounded_production(store, catalog, raw_store, root, dashboard_url)
    label = _label_health_readonly(
        store, store.distillation_dir(root) / store.LABEL_LEDGER_FILE
    )
    return {**snapshot, "label_health": label}


def _load_r0_anchor(store: Any, source_root: Path) -> dict[str, Any]:
    path = source_root / R0_EVIDENCE_RELATIVE
    try:
        state = _stat(path)
        if state is None:
            raise R1Error("R0 evidence is missing")
        with path.open("rb") as handle:
            raw = handle.read()
        payload = json.loads(raw)
        verified = store.verify_seal(payload, schema=R0_SCHEMA)
        anchor_id = verified.get("artifact_id")
        if anchor_id != R0_EVIDENCE_ID:
            raise R1Error("R0 evidence artifact id mismatch")
        unsigned = {
            key: value
            for key, value in verified.items()
            if key not in {"artifact_id", "seal_sha256"}
        }
        if store.canonical_json_sha256_strict(unsigned) != R0_EVIDENCE_ID:
            raise R1Error("R0 evidence content digest mismatch")
        if _stat(path) != state:
            raise R1Error("R0 evidence changed during read")
    except (OSError, UnicodeError, ValueError, store.DistillationStoreError) as exc:
        raise R1Error("R0 evidence cannot be verified") from exc
    return {
        "artifact_id": R0_EVIDENCE_ID,
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "seal_sha256": str(verified["seal_sha256"]),
        "production": verified.get("production"),
    }


def _assert_r0_anchor(snapshot: Mapping[str, Any], anchor: Mapping[str, Any]) -> None:
    anchor_production = anchor.get("production")
    current_ledgers = snapshot.get("ledgers")
    if not isinstance(anchor_production, Mapping) or not isinstance(
        current_ledgers, Mapping
    ):
        raise R1Error("R0 anchor production section is invalid")
    anchor_ledgers = anchor_production.get("ledgers")
    if not isinstance(anchor_ledgers, Mapping):
        raise R1Error("R0 anchor ledgers are invalid")
    for name, expected in anchor_ledgers.items():
        actual = current_ledgers.get(name)
        if not isinstance(expected, Mapping) or not isinstance(actual, Mapping):
            raise R1Error(f"R0 anchor ledger is missing: {name}")
        for field in ("head_sha256", "records", "bytes", "file_state"):
            if actual.get(field) != expected.get(field):
                raise R1Error(f"R0 anchor ledger mismatch: {name}.{field}")
    if snapshot.get("raw_watermark") != anchor_production.get("raw_watermark"):
        raise R1Error("R0 anchor raw watermark mismatch")
    current_fts = snapshot.get("fts")
    anchor_fts = anchor_production.get("fts")
    if not isinstance(current_fts, Mapping) or not isinstance(anchor_fts, Mapping):
        raise R1Error("R0 anchor FTS section is invalid")
    for field in (
        "content_sha256",
        "atom_count",
        "fts_count",
        "file_state",
        "checkpoint_seal_sha256",
    ):
        if current_fts.get(field) != anchor_fts.get(field):
            raise R1Error(f"R0 anchor FTS mismatch: {field}")


def _assert_expected_baseline(
    snapshot: Mapping[str, Any], anchor: Mapping[str, Any]
) -> None:
    _assert_r0_anchor(snapshot, anchor)
    ledgers = snapshot.get("ledgers")
    if not isinstance(ledgers, Mapping):
        raise R1Error("production ledger snapshot missing")
    for name, expected in EXPECTED_BASELINE.items():
        actual = ledgers.get(name)
        if not isinstance(actual, Mapping):
            raise R1Error(f"production ledger missing: {name}")
        if (
            actual.get("bytes") != expected["bytes"]
            or actual.get("records") != expected["records"]
        ):
            raise R1Error(f"production baseline drift: {name}")


def _measure(name: str, call: Callable[[], Any]) -> dict[str, Any]:
    before = _proc_pid_rusage_v2()
    started = time.perf_counter_ns()
    value: Any = None
    error: Exception | None = None
    samples = [before]
    sample_errors: list[Exception] = []
    stop_sampling = threading.Event()

    def sample_rusage() -> None:
        while not stop_sampling.wait(RUSAGE_SAMPLE_INTERVAL_SECONDS):
            try:
                samples.append(_proc_pid_rusage_v2())
            except Exception as exc:
                sample_errors.append(exc)

    sampler = threading.Thread(
        target=sample_rusage,
        name=f"r1-rusage-{name}",
        daemon=True,
    )
    sampler.start()
    try:
        value = call()
    except Exception as exc:  # expected crash/fault cases are measured too
        error = exc
    finally:
        stop_sampling.set()
        sampler.join()
    finished = time.perf_counter_ns()
    after = _proc_pid_rusage_v2()
    samples.append(after)
    if sample_errors:
        raise R1Error("rusage sampler failed") from sample_errors[0]
    if before["rusage_uuid"] != after["rusage_uuid"] or finished < started:
        raise R1Error("measurement counter invalid")
    if any(sample["rusage_uuid"] != before["rusage_uuid"] for sample in samples):
        raise R1Error("rusage sampler identity changed")
    read_bytes = int(after["disk_read_bytes"]) - int(before["disk_read_bytes"])
    write_bytes = int(after["disk_write_bytes"]) - int(before["disk_write_bytes"])
    if read_bytes < 0 or write_bytes < 0:
        raise R1Error("measurement counter decreased")
    result: dict[str, Any] = {
        "id": name,
        "status": "error" if error is not None else "ok",
        "metrics": {
            "wall_time_ns": finished - started,
            "disk_read_bytes": read_bytes,
            "disk_write_bytes": write_bytes,
            "resident_before_bytes": int(before["resident_bytes"]),
            "resident_after_bytes": int(after["resident_bytes"]),
            "resident_peak_bytes": max(
                int(sample["resident_bytes"]) for sample in samples
            ),
            "footprint_before_bytes": int(before["footprint_bytes"]),
            "footprint_after_bytes": int(after["footprint_bytes"]),
            "footprint_peak_bytes": max(
                int(sample["footprint_bytes"]) for sample in samples
            ),
            "rusage_sample_count": len(samples),
            "rusage_sample_interval_ns": int(
                RUSAGE_SAMPLE_INTERVAL_SECONDS * 1_000_000_000
            ),
            "rusage_peak_method": "periodic_sampler_with_boundaries",
            "rusage_uuid": str(before["rusage_uuid"]),
        },
    }
    if error is not None:
        result["error"] = {"type": type(error).__name__}
    else:
        result["value"] = value
    return result


class _GuardedHandle:
    def __init__(
        self,
        handle: Any,
        reads: list[tuple[int, int]],
        allowed: list[tuple[int, int]],
        *,
        allow_full: bool,
    ) -> None:
        self._handle = handle
        self._reads = reads
        self._allowed = allowed
        self._allow_full = allow_full

    def __enter__(self) -> _GuardedHandle:
        self._handle.__enter__()
        return self

    def __exit__(self, *args: Any) -> Any:
        return self._handle.__exit__(*args)

    def _record(self, start: int, length: int) -> None:
        if length <= 0:
            return
        if not self._allow_full and not any(
            start >= offset and start + length <= offset + size
            for offset, size in self._allowed
        ):
            raise R1Error("ledger read escaped allowed range")
        self._reads.append((start, length))

    def read(self, size: int = -1) -> bytes:
        start = int(self._handle.tell())
        value = self._handle.read(size)
        self._record(start, len(value))
        return value

    def readline(self, size: int = -1) -> bytes:
        start = int(self._handle.tell())
        value = self._handle.readline(size)
        self._record(start, len(value))
        return value

    def readlines(self, hint: int = -1) -> list[bytes]:
        start = int(self._handle.tell())
        values = self._handle.readlines(hint)
        self._record(start, sum(len(value) for value in values))
        return values

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        return int(self._handle.seek(offset, whence))

    def tell(self) -> int:
        return int(self._handle.tell())

    def __iter__(self) -> Iterator[bytes]:
        return self

    def __next__(self) -> bytes:
        value = self.readline()
        if not value:
            raise StopIteration
        return value

    def __getattr__(self, name: str) -> Any:
        return getattr(self._handle, name)


@contextlib.contextmanager
def _body_guard(
    path: Path,
    *,
    allowed_ranges: list[tuple[int, int]] | None = None,
    allow_full: bool = False,
) -> Iterator[dict[str, Any]]:
    """Guard one ledger body and retain exact ranges read by the operation."""

    target = path.resolve()
    reads: list[tuple[int, int]] = []
    allowed = list(allowed_ranges or [])
    original_open = Path.open
    original_read_bytes = Path.read_bytes
    if allowed:
        prefix_bytes = min(offset for offset, _size in allowed)
    elif allow_full:
        prefix_bytes = path.stat().st_size
    else:
        prefix_bytes = 0

    def guarded_open(self: Path, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        handle = original_open(self, mode, *args, **kwargs)
        if self.resolve() != target or not ({"r", "+"} & set(mode)):
            return handle
        return _GuardedHandle(handle, reads, allowed, allow_full=allow_full)

    def guarded_read_bytes(self: Path) -> bytes:
        if self.resolve() != target:
            return original_read_bytes(self)
        if not allow_full:
            raise R1Error("ledger body read is forbidden")
        value = original_read_bytes(self)
        reads.append((0, len(value)))
        return value

    Path.open = guarded_open  # type: ignore[method-assign]
    Path.read_bytes = guarded_read_bytes  # type: ignore[method-assign]
    try:
        yield {
            "allowed": allowed,
            "ranges": reads,
            "prefix_bytes": prefix_bytes,
        }
    finally:
        Path.open = original_open  # type: ignore[method-assign]
        Path.read_bytes = original_read_bytes  # type: ignore[method-assign]


def _guard_evidence(guard: Mapping[str, Any]) -> dict[str, Any]:
    ranges = [
        {"offset": int(offset), "length": int(length)}
        for offset, length in guard.get("ranges", [])
    ]
    prefix = int(guard.get("prefix_bytes", 0))
    old_prefix_bytes = sum(
        max(0, min(offset + length, prefix) - offset)
        for offset, length in guard.get("ranges", [])
    )
    return {
        "ranges": ranges,
        "bytes_read": sum(row["length"] for row in ranges),
        "old_prefix_bytes": old_prefix_bytes,
        "old_prefix_scanned": old_prefix_bytes > 0,
    }


@contextlib.contextmanager
def _write_guard(path: Path) -> Iterator[dict[str, Any]]:
    """Prove steady writes append to the existing inode via native O_APPEND."""

    target = path.resolve()
    before = _stat(path)
    if before is None:
        raise R1Error("write guard target is missing")
    writes: list[dict[str, Any]] = []
    target_fds: set[int] = set()
    original_os_open = os.open
    original_path_open = Path.open
    original_replace = os.replace
    original_rename = os.rename
    original_truncate = os.truncate
    original_ftruncate = os.ftruncate

    def resolve_target(value: Any) -> bool:
        try:
            return Path(value).resolve() == target
        except (OSError, TypeError, ValueError):
            return False

    def guarded_os_open(
        value: Any,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        write_flags = flags & (os.O_WRONLY | os.O_RDWR | os.O_TRUNC | os.O_APPEND)
        if resolve_target(value) and write_flags:
            if not (flags & os.O_APPEND) or flags & os.O_TRUNC:
                raise R1Error("steady ledger write requires O_APPEND without O_TRUNC")
            if dir_fd is None:
                descriptor = original_os_open(value, flags, mode)
            else:
                descriptor = original_os_open(value, flags, mode, dir_fd=dir_fd)
            stat = os.fstat(descriptor)
            if int(stat.st_ino) != int(before["st_ino"]):
                os.close(descriptor)
                raise R1Error("steady ledger write replaced inode")
            target_fds.add(descriptor)
            writes.append({"flags": int(flags), "append": True})
            return descriptor
        if dir_fd is None:
            return original_os_open(value, flags, mode)
        return original_os_open(value, flags, mode, dir_fd=dir_fd)

    def guarded_path_open(
        self: Path, mode: str = "r", *args: Any, **kwargs: Any
    ) -> Any:
        if self.resolve() == target and (
            any(character in mode for character in "wax") or "+" in mode
        ):
            if (
                "a" not in mode
                or "+" in mode
                or any(character in mode for character in "wx")
            ):
                raise R1Error("steady ledger high-level write is not append-only")
            writes.append({"flags": None, "append": True, "path_open": mode})
        return original_path_open(self, mode, *args, **kwargs)

    def reject_replace(src: Any, dst: Any, *args: Any, **kwargs: Any) -> Any:
        if resolve_target(src) or resolve_target(dst):
            raise R1Error("steady ledger replace is forbidden")
        return original_replace(src, dst, *args, **kwargs)

    def reject_rename(src: Any, dst: Any, *args: Any, **kwargs: Any) -> Any:
        if resolve_target(src) or resolve_target(dst):
            raise R1Error("steady ledger rename is forbidden")
        return original_rename(src, dst, *args, **kwargs)

    def reject_truncate(value: Any, length: int) -> Any:
        if resolve_target(value):
            raise R1Error("steady ledger truncate is forbidden")
        return original_truncate(value, length)

    def reject_ftruncate(descriptor: int, length: int) -> Any:
        if descriptor in target_fds:
            raise R1Error("steady ledger ftruncate is forbidden")
        return original_ftruncate(descriptor, length)

    os.open = guarded_os_open  # type: ignore[method-assign]
    Path.open = guarded_path_open  # type: ignore[method-assign]
    os.replace = reject_replace  # type: ignore[method-assign]
    os.rename = reject_rename  # type: ignore[method-assign]
    os.truncate = reject_truncate  # type: ignore[method-assign]
    os.ftruncate = reject_ftruncate  # type: ignore[method-assign]
    evidence: dict[str, Any] = {
        "before": before,
        "writes": writes,
        "target_fds": target_fds,
    }
    try:
        yield evidence
    finally:
        os.open = original_os_open  # type: ignore[method-assign]
        Path.open = original_path_open  # type: ignore[method-assign]
        os.replace = original_replace  # type: ignore[method-assign]
        os.rename = original_rename  # type: ignore[method-assign]
        os.truncate = original_truncate  # type: ignore[method-assign]
        os.ftruncate = original_ftruncate  # type: ignore[method-assign]
    after = _stat(path)
    if after is None or after["st_ino"] != before["st_ino"]:
        raise R1Error("steady ledger inode changed")
    append_bytes = after["size_bytes"] - before["size_bytes"]
    if append_bytes < 0:
        raise R1Error("steady ledger shrank")
    if any(not write.get("append") for write in writes):
        raise R1Error("steady ledger write lacked O_APPEND")
    evidence.update(
        {
            "native_o_append": bool(writes),
            "write_open_count": len(writes),
            "old_prefix_write_bytes": 0,
            "append_bytes": append_bytes,
            "inode_before": int(before["st_ino"]),
            "inode_after": int(after["st_ino"]),
        }
    )
    evidence.pop("writes", None)
    evidence.pop("target_fds", None)


def _require_ok(measured: Mapping[str, Any]) -> Any:
    if measured.get("status") != "ok":
        raise R1Error(f"operation failed: {measured.get('id')}")
    return measured.get("value")


def _head(store: Any, path: Path) -> dict[str, Any]:
    value = store.chain_head(path)
    if set(value) != {"records", "head_sha256"}:
        raise R1Error("chain head shape invalid")
    return {"records": int(value["records"]), "head_sha256": str(value["head_sha256"])}


def _run_guarded(
    name: str,
    path: Path,
    call: Callable[[], Any],
    *,
    allowed_ranges: list[tuple[int, int]] | None = None,
    allow_full: bool = False,
    write_guard: bool = False,
) -> dict[str, Any]:
    with _body_guard(
        path, allowed_ranges=allowed_ranges, allow_full=allow_full
    ) as guard:
        if write_guard:
            with _write_guard(path) as writes:
                measured = _measure(name, call)
        else:
            writes = None
            measured = _measure(name, call)
    measured["read_guard"] = _guard_evidence(guard)
    if writes is not None:
        measured["write_guard"] = writes
    return measured


def _candidate_snapshot(catalog: Any, rally_id: str) -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "schema": catalog.CANDIDATE_SNAPSHOT_SCHEMA,
        "rally_id": rally_id,
        "as_of": "2026-08-24T00:00:00Z",
        "retriever_revision": "historical-fts-v1",
        "feature_revision": "recall-distill-text-v2",
        "query_feature_text_sha256": "a" * 64,
        "candidates": [
            {
                "candidate_id": f"r1-candidate-{rally_id[:12]}",
                "rank": 1,
                "text_sha256": "b" * 64,
                "candidate_feature_text_sha256": "c" * 64,
            }
        ],
    }
    snapshot["snapshot_sha256"] = catalog.canonical_json_sha256_strict(snapshot)
    return snapshot


COPYFILE_ALL = (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3)
COPYFILE_NOFOLLOW = (1 << 18) | (1 << 19)
COPYFILE_CLONE_FORCE = 1 << 25


def _copyfile_clone(source: Path, destination: Path, flags: int) -> None:
    if sys.platform != "darwin":
        raise R1Error("copyfile(3) clone requires Darwin")
    try:
        library = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
        function = library.copyfile
        function.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_void_p,
            ctypes.c_uint32,
        ]
        function.restype = ctypes.c_int
        result = function(
            os.fsencode(source), os.fsencode(destination), None, int(flags)
        )
    except (AttributeError, OSError) as exc:
        raise R1Error("copyfile(3) unavailable") from exc
    if result != 0:
        error = ctypes.get_errno()
        raise R1Error(f"forced APFS clone failed: errno={error}")


def _paths_overlap(left: Path, right: Path) -> bool:
    """Return whether either path contains the other (including equality)."""

    left_resolved = left.expanduser().resolve(strict=False)
    right_resolved = right.expanduser().resolve(strict=False)
    return (
        left_resolved == right_resolved
        or left_resolved.is_relative_to(right_resolved)
        or right_resolved.is_relative_to(left_resolved)
    )


def _clone_destination_parent() -> Path:
    try:
        destination_parent = Path(tempfile.gettempdir()).resolve(strict=True)
    except OSError as exc:
        raise R1Error("clone destination parent is unavailable") from exc
    if not destination_parent.is_dir():
        raise R1Error("clone destination parent is not a directory")
    return destination_parent


def _assert_clone_destination_safe(
    destination_parent: Path, protected_roots: Iterable[Path]
) -> None:
    for protected in protected_roots:
        if _paths_overlap(destination_parent, protected):
            raise R1Error("clone temp destination overlaps protected root")


def _cleanup_clone(path: Path) -> None:
    try:
        shutil.rmtree(path)
    except OSError as exc:
        raise R1Error("clone cleanup failed") from exc
    if path.exists():
        raise R1Error("clone cleanup failed")


def _forced_clone(
    production: Path, protected_roots: Iterable[Path] = ()
) -> tuple[Path, bool, dict[str, Any]]:
    production = production.resolve(strict=True)
    if sys.platform != "darwin" or not production.is_dir():
        raise R1Error("forced APFS clone requires Darwin directory")
    _assert_clone_destination_safe(
        _clone_destination_parent(), (production, *protected_roots)
    )
    required = [
        production / "raw",
        production / "runtime" / "recall-distillation",
    ]
    optional = production / "recall" / "recall-log.jsonl"
    roots = [*required, optional] if optional.exists() else required
    files: list[Path] = []
    directories: set[Path] = set()
    for root in roots:
        if root.is_symlink():
            raise R1Error("clone source contains symlink")
        if not root.exists():
            if root in required:
                raise R1Error("forced APFS clone source subtree missing")
            continue
        if root.is_file():
            if root in required:
                raise R1Error("forced APFS clone source subtree is not a directory")
            files.append(root)
            continue
        if not root.is_dir():
            raise R1Error("clone source is not a regular file or directory")
        for base, dir_names, file_names in os.walk(root, followlinks=False):
            base_path = Path(base)
            directories.add(base_path)
            for name in sorted(dir_names):
                child = base_path / name
                if child.is_symlink():
                    raise R1Error("clone source contains symlink")
            for name in sorted(file_names):
                child = base_path / name
                if child.is_symlink() or not child.is_file():
                    raise R1Error("clone source contains unsafe file")
                files.append(child)
    clone = Path(tempfile.mkdtemp(prefix="chronovisor-r1-"))
    flags = COPYFILE_ALL | COPYFILE_NOFOLLOW | COPYFILE_CLONE_FORCE
    try:
        for directory in sorted(directories):
            (clone / directory.relative_to(production)).mkdir(
                parents=True, exist_ok=True
            )
        for source in files:
            destination = clone / source.relative_to(production)
            destination.parent.mkdir(parents=True, exist_ok=True)
            _copyfile_clone(source, destination, flags)
    except Exception:
        _cleanup_clone(clone)
        raise
    if (
        clone == production
        or clone.is_relative_to(production)
        or production.is_relative_to(clone)
    ):
        _cleanup_clone(clone)
        raise R1Error("forced APFS clone overlaps production")
    if any(path.is_symlink() for path in clone.rglob("*")):
        _cleanup_clone(clone)
        raise R1Error("forced APFS clone contains symlink")
    return (
        clone,
        True,
        {
            "copy_backend": "copyfile(3)",
            "copy_flags": flags,
            "clone_force": True,
            "files_cloned": len(files),
            "directories_created": len(directories),
        },
    )


def _apfs_clone_preflight(
    production: Path, protected_roots: Iterable[Path] = ()
) -> dict[str, Any]:
    if sys.platform != "darwin":
        raise R1Error("APFS clone requires Darwin")
    source = production.resolve(strict=True)
    destination_parent = _clone_destination_parent()
    _assert_clone_destination_safe(destination_parent, (source, *protected_roots))
    source_type = _filesystem_type(source)
    destination_type = _filesystem_type(destination_parent)
    if source_type != "apfs" or destination_type != "apfs":
        raise R1Error("APFS clone requires APFS source and destination")
    try:
        same_volume = source.stat().st_dev == destination_parent.stat().st_dev
    except OSError as exc:
        raise R1Error("APFS volume identity probe failed") from exc
    if not same_volume:
        raise R1Error("APFS source and destination volumes differ")
    return {
        "source_filesystem": source_type,
        "destination_filesystem": destination_type,
        "same_volume": True,
        "copy_backend": "copyfile(3)",
        "clone_force_required": True,
    }


@contextlib.contextmanager
def _clone_context(
    production: Path,
    clone_evidence: list[dict[str, Any]] | None = None,
    protected_roots: Iterable[Path] = (),
) -> Iterator[Path]:
    protected_roots = tuple(protected_roots)
    contract = _apfs_clone_preflight(production, protected_roots)
    clone, temporary, copy_evidence = _forced_clone(production, protected_roots)
    try:
        clone_type = _filesystem_type(clone)
        if clone_type != contract["destination_filesystem"]:
            raise R1Error("APFS clone destination filesystem changed")
        contract = {
            **contract,
            **copy_evidence,
            "clone_filesystem": clone_type,
            "clone_copy_verified": True,
        }
        if clone_evidence is not None:
            clone_evidence.append(contract)
        yield clone
    finally:
        if temporary:
            _cleanup_clone(clone)


def _warm_head(store: Any, path: Path, name: str) -> dict[str, Any]:
    expected = _head(store, path)
    measured = _run_guarded(
        name,
        path,
        lambda: _head(store, path),
        allowed_ranges=[],
    )
    if _require_ok(measured) != expected or measured["read_guard"]["bytes_read"]:
        raise R1Error(f"warm head scanned ledger: {path.name}")
    return {"measurement": measured, "head": expected}


def _steady_ledger(
    store: Any,
    root: Path,
    name: str,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    path = store.distillation_dir(root) / name
    before_state = _stat(path)
    if before_state is None:
        raise R1Error(f"ledger missing: {name}")
    warm_head = _warm_head(store, path, f"head.{name}")
    before_head = warm_head["head"]
    tail_range = (max(0, before_state["size_bytes"] - 1), 1)
    measured = _run_guarded(
        f"steady_append.{name}",
        path,
        lambda: store.append_chain_batch(path, [dict(payload)]),
        allowed_ranges=[tail_range],
        write_guard=True,
    )
    rows = _require_ok(measured)
    after_state = _stat(path)
    after_head = _head(store, path)
    if not isinstance(rows, list) or len(rows) != 1:
        raise R1Error(f"append result invalid: {name}")
    if after_head["records"] != before_head["records"] + 1:
        raise R1Error(f"append count invalid: {name}")
    if measured["read_guard"]["old_prefix_scanned"]:
        raise R1Error(f"append scanned old prefix: {name}")
    write_evidence = measured.get("write_guard", {})
    if (
        write_evidence.get("old_prefix_write_bytes") != 0
        or not write_evidence.get("native_o_append")
        or write_evidence.get("append_bytes")
        != after_state["size_bytes"] - before_state["size_bytes"]
    ):
        raise R1Error(f"append write guard invalid: {name}")
    result = {
        "name": name,
        "warm_head": warm_head,
        "before": {"head": before_head, "file_state": before_state},
        "after": {"head": after_head, "file_state": after_state},
        "measurement": measured,
    }
    if name == store.LABEL_LEDGER_FILE:
        health = _run_guarded(
            "label_health_projection",
            path,
            lambda: store.label_health_projection(path, repair=False),
            allowed_ranges=[],
        )
        value = _require_ok(health)
        if value["label_records"] != after_head["records"]:
            raise R1Error("label projection record mismatch")
        if health["read_guard"]["old_prefix_scanned"]:
            raise R1Error("label health scanned old prefix")
        result["label_health"] = {"value": value, "measurement": health}
    return result


def _label_projection_bootstrap(store: Any, root: Path) -> dict[str, Any]:
    path = store.distillation_dir(root) / store.LABEL_LEDGER_FILE
    projection_path = store._label_health_projection_path(path)
    before = _stat(projection_path)
    measured = _run_guarded(
        "label_health_bootstrap",
        path,
        lambda: store.label_health_projection(path, repair=True),
        allow_full=True,
    )
    value = _require_ok(measured)
    after = _stat(projection_path)
    if after is None:
        raise R1Error("label health bootstrap did not seal projection")
    return {
        "projection": projection_path.name,
        "before": before,
        "after": after,
        "value": value,
        "measurement": measured,
    }


def _candidate_hot_path(store: Any, catalog: Any, root: Path) -> dict[str, Any]:
    path = store.distillation_dir(root) / "candidate-ledger.jsonl"
    _head(store, path)
    bootstrap = _measure(
        "candidate_index_rebuild",
        lambda: catalog.sync_candidate_index(root, path, rebuild=True),
    )
    bootstrap_value = _require_ok(bootstrap)
    base_state = catalog.candidate_index_state(root)
    expected = EXPECTED_BASELINE["candidate-ledger.jsonl"]
    if base_state["record_count"] != expected["records"]:
        raise R1Error("candidate index bootstrap count mismatch")
    base_file_state = _stat(path)
    if (
        base_file_state is None
        or base_state["indexed_offset"] != base_file_state["size_bytes"]
    ):
        raise R1Error("candidate index bootstrap offset mismatch")
    warm_head = _warm_head(store, path, "candidate_head")
    rally_id = f"r1-{base_state['head_sha256'][:24]}"
    row = {
        "kind": "candidate-snapshot",
        "rally_id": rally_id,
        "snapshot": _candidate_snapshot(catalog, rally_id),
    }
    before_size = int(base_state["indexed_offset"])
    append = _run_guarded(
        "steady_append.candidate-ledger.jsonl",
        path,
        lambda: store.append_chain_batch(path, [row]),
        allowed_ranges=[(max(0, before_size - 1), 1)],
        write_guard=True,
    )
    _require_ok(append)
    after_size = int(_stat(path)["size_bytes"])  # type: ignore[index]
    tail_length = after_size - before_size
    if tail_length <= 0:
        raise R1Error("candidate append did not grow ledger")
    write_evidence = append.get("write_guard", {})
    if (
        write_evidence.get("old_prefix_write_bytes") != 0
        or not write_evidence.get("native_o_append")
        or write_evidence.get("append_bytes") != after_size - before_size
    ):
        raise R1Error("candidate append write guard invalid")
    observed: dict[str, int] = {}
    original_scan = catalog._scan_candidate_ledger

    def scan_wrapper(*args: Any, **kwargs: Any) -> Any:
        start = int(kwargs.get("start", args[1] if len(args) > 1 else -1))
        count = int(kwargs.get("expected_count", args[3] if len(args) > 3 else -1))
        observed.update(start=start, expected_count=count)
        if start != before_size or count != expected["records"]:
            raise R1Error("candidate tail scan started before indexed offset")
        return original_scan(*args, **kwargs)

    catalog._scan_candidate_ledger = scan_wrapper
    try:
        tail_sync = _run_guarded(
            "candidate_index_tail",
            path,
            lambda: catalog.sync_candidate_index(root, path),
            allowed_ranges=[(before_size, tail_length)],
        )
    finally:
        catalog._scan_candidate_ledger = original_scan
    tail_value = _require_ok(tail_sync)
    if tail_value["record_count"] != expected["records"] + 1:
        raise R1Error("candidate tail index count mismatch")
    if tail_sync["read_guard"]["old_prefix_scanned"] or observed != {
        "start": before_size,
        "expected_count": expected["records"],
    }:
        raise R1Error("candidate tail index scanned old prefix")

    state = catalog.candidate_index_state(root)
    with sqlite3.connect(
        f"file:{catalog.catalog_path(root)}?mode=ro", uri=True
    ) as connection:
        rows = connection.execute(
            "SELECT record_index,rally_id,offset,length FROM candidate_records "
            "ORDER BY record_index"
        ).fetchall()
    if not rows:
        raise R1Error("candidate index has no rows")
    randomizer = random.Random(0x5231)
    selected = [
        rows[index] for index in randomizer.sample(range(len(rows)), min(8, len(rows)))
    ]
    ranges = [(int(row[2]), int(row[3])) for row in selected]
    ids = [str(row[1]) for row in selected]
    random_read = _run_guarded(
        "candidate_random_read",
        path,
        lambda: catalog.read_candidate_snapshots(root, path, ids),
        allowed_ranges=ranges,
    )
    snapshots = _require_ok(random_read)
    if set(snapshots) != set(ids) or random_read["read_guard"]["old_prefix_scanned"]:
        raise R1Error("candidate random read scanned old prefix")
    random_read["result_digest"] = _digest(
        {key: snapshots[key].get("snapshot_sha256") for key in sorted(snapshots)}
    )
    return {
        "bootstrap": bootstrap,
        "bootstrap_result": bootstrap_value,
        "warm_head": warm_head,
        "append": append,
        "tail_sync": tail_sync,
        "tail_sync_observed": observed,
        "random_read": random_read,
        "random_read_targets": [
            {
                "rally_sha256": hashlib.sha256(value.encode()).hexdigest(),
                "offset": int(row[2]),
                "length": int(row[3]),
            }
            for value, row in zip(ids, selected, strict=True)
        ],
        "final_index_state": _redact_index_state(state),
    }


def _recovery(
    store: Any,
    root: Path,
    name: str,
    mutation: Callable[[Path, dict[str, Any]], Mapping[str, Any] | None],
) -> dict[str, Any]:
    path = store.distillation_dir(root) / name
    before_state = _stat(path)
    if before_state is None:
        raise R1Error(f"recovery ledger missing: {name}")
    before_head = _head(store, path)
    injection = mutation(path, before_head)
    measured = _run_guarded(
        f"recovery.{name}",
        path,
        lambda: _head(store, path),
        allow_full=True,
    )
    recovered = _require_ok(measured)
    after_state = _stat(path)
    if (
        recovered != before_head
        or after_state is None
        or after_state["size_bytes"] != before_state["size_bytes"]
    ):
        raise R1Error(f"recovery head mismatch: {name}")
    if not measured["read_guard"]["old_prefix_scanned"]:
        raise R1Error(f"recovery did not verify prefix: {name}")
    return {
        "name": name,
        "injection": dict(injection) if isinstance(injection, Mapping) else None,
        "before": {"head": before_head, "file_state": before_state},
        "after": {"head": recovered, "file_state": after_state},
        "measurement": measured,
    }


def _append_partial(path: Path, _head_value: Mapping[str, Any]) -> None:
    with path.open("ab") as handle:
        handle.write(b'{"r1_partial":')
        handle.flush()
        os.fsync(handle.fileno())


def _append_separator(path: Path, _head_value: Mapping[str, Any]) -> None:
    with path.open("ab") as handle:
        handle.write(b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def _mismatch_checkpoint(
    store: Any, path: Path, head: Mapping[str, Any]
) -> dict[str, Any]:
    checkpoint_path = store._chain_checkpoint_path(path)
    checkpoint = store.read_sealed(checkpoint_path, schema=store.DISTILLATION_SCHEMA)
    current_state = _stat(path)
    if current_state is None or checkpoint.get("file_state") != current_state:
        raise R1Error("cannot seed stale checkpoint from unstable ledger")
    stale_state = dict(current_state)
    if stale_state["size_bytes"] > 0:
        stale_state["size_bytes"] -= 1
    else:
        stale_state["st_mtime_ns"] -= 1
    store.write_sealed_state(
        checkpoint_path,
        {
            "kind": "ledger-chain-checkpoint",
            "ledger_name": path.name,
            "records": int(head["records"]),
            "head_sha256": str(head["head_sha256"]),
            "file_state": stale_state,
        },
    )
    return {
        "kind": "stale_checkpoint_file_state",
        "checkpoint_file_state": stale_state,
        "ledger_file_state": current_state,
    }


def _unique_target(store: Any, path: Path) -> dict[str, Any]:
    index_path = store._unique_index_path(path)
    metadata: dict[str, str]
    try:
        metadata = {}
        with sqlite3.connect(f"file:{index_path}?mode=ro", uri=True) as connection:
            metadata = dict(connection.execute("SELECT key,value FROM metadata"))
            row = connection.execute(
                "SELECT entries.identity,entries.binding,rows.offset,rows.length,"
                "rows.record_sha256 "
                "FROM entries JOIN rows USING(offset) ORDER BY rows.offset LIMIT 1"
            ).fetchone()
    except (OSError, sqlite3.Error) as exc:
        raise R1Error("unique sidecar is unreadable") from exc
    if (
        metadata.get("unique_field") != "decision_id"
        or metadata.get("binding_field") != "idempotency_sha256"
        or row is None
    ):
        raise R1Error("no compatible unique receipt ledger")
    try:
        identity = json.loads(bytes(row[0]))
        binding = json.loads(bytes(row[1]))
    except (TypeError, ValueError, UnicodeError) as exc:
        raise R1Error("unique sidecar identity is invalid") from exc
    if not isinstance(identity, str) or not isinstance(binding, str):
        raise R1Error("unique sidecar identity is not text")
    return {
        "decision_id": identity,
        "idempotency_sha256": binding,
        "offset": int(row[2]),
        "length": int(row[3]),
        "record_sha256": str(row[4]),
    }


def _unique_setup(store: Any, root: Path) -> tuple[Path, dict[str, Any]]:
    directory = store.distillation_dir(root)
    for name in UNIQUE_LEDGER_NAMES:
        path = directory / name
        state = _stat(path)
        if state is None or state["size_bytes"] == 0:
            continue
        try:
            head, connection = store._rebuild_unique_index(
                path, unique_field="decision_id", binding_field="idempotency_sha256"
            )
            connection.close()
            target = _unique_target(store, path)
            target["head"] = head
            target["file_state"] = _stat(path)
            return path, target
        except (
            OSError,
            sqlite3.Error,
            ValueError,
            store.DistillationStoreError,
            R1Error,
        ):
            continue
    raise R1Error("no production unique receipt ledger")


def _unique_duplicate(
    store: Any,
    root: Path,
    scenario: str,
    mutation: Callable[[Path, Path], None] | None = None,
) -> dict[str, Any]:
    path, target = _unique_setup(store, root)
    index_path = store._unique_index_path(path)
    before_state = _stat(path)
    before_head = _head(store, path)
    if mutation is not None:
        mutation(path, index_path)
    ranges = [(int(target["offset"]), int(target["length"]))]
    measured = _run_guarded(
        f"unique_duplicate.{scenario}",
        path,
        lambda: store.append_chain_unique(
            path,
            {
                "decision_id": target["decision_id"],
                "idempotency_sha256": target["idempotency_sha256"],
            },
            unique_field="decision_id",
            binding_field="idempotency_sha256",
        ),
        allowed_ranges=ranges,
        allow_full=scenario != "steady",
    )
    result = _require_ok(measured)
    after_state = _stat(path)
    after_head = _head(store, path)
    if (
        after_head != before_head
        or after_state is None
        or after_state["size_bytes"] != before_state["size_bytes"]
    ):
        raise R1Error(f"unique duplicate appended: {scenario}")
    if result.get("record_sha256") != target.get("record_sha256") and target.get(
        "record_sha256"
    ):
        raise R1Error(f"unique duplicate returned wrong row: {scenario}")
    measured["result"] = {
        "record_sha256": result.get("record_sha256"),
        "duplicate_appends": 0,
    }
    return {
        "scenario": scenario,
        "ledger": path.name,
        "target": {
            "identity_sha256": hashlib.sha256(
                target["decision_id"].encode()
            ).hexdigest(),
            "offset": target["offset"],
            "length": target["length"],
        },
        "before": {"head": before_head, "file_state": before_state},
        "after": {"head": after_head, "file_state": after_state},
        "measurement": measured,
    }


def _logical_delete_tamper(_path: Path, index: Path) -> None:
    """Delete indexed identity rows and stale the sealed metadata head."""

    try:
        with sqlite3.connect(index) as connection:
            connection.execute("PRAGMA foreign_keys=OFF")
            connection.execute("DELETE FROM entries")
            updated_records = connection.execute(
                "UPDATE metadata SET value = '0' WHERE key = 'records'"
            ).rowcount
            updated_head = connection.execute(
                "UPDATE metadata SET value = ? WHERE key = 'head_sha256'",
                ("0" * 64,),
            ).rowcount
            if updated_records != 1 or updated_head != 1:
                raise R1Error("unique index metadata tamper did not apply")
    except (OSError, sqlite3.Error) as exc:
        raise R1Error("unique index logical-delete tamper failed") from exc


def _unique_crash(store: Any, root: Path) -> dict[str, Any]:
    path, target = _unique_setup(store, root)
    before_state = _stat(path)
    before_head = _head(store, path)
    payload = {
        "decision_id": f"r1-crash-{before_head['head_sha256'][:24]}",
        "idempotency_sha256": hashlib.sha256(
            f"r1-crash\0{before_head['head_sha256']}".encode()
        ).hexdigest(),
    }
    original = store._write_chain_checkpoint
    crashed = {"value": False}

    def fail_after_data(path_value: Path, head: Mapping[str, Any]) -> None:
        if (
            int(head["records"]) == int(before_head["records"]) + 1
            and not crashed["value"]
        ):
            crashed["value"] = True
            raise RuntimeError("r1 injected checkpoint crash")
        original(path_value, head)

    store._write_chain_checkpoint = fail_after_data
    try:
        first = _run_guarded(
            "unique_crash.append",
            path,
            lambda: store.append_chain_unique(
                path,
                payload,
                unique_field="decision_id",
                binding_field="idempotency_sha256",
            ),
            allowed_ranges=[(max(0, int(before_state["size_bytes"]) - 1), 1)],
        )
    finally:
        store._write_chain_checkpoint = original
    if first.get("status") != "error" or not crashed["value"]:
        raise R1Error("checkpoint crash was not injected")
    retry = _run_guarded(
        "unique_crash.retry_duplicate",
        path,
        lambda: store.append_chain_unique(
            path,
            payload,
            unique_field="decision_id",
            binding_field="idempotency_sha256",
        ),
        allow_full=True,
    )
    result = _require_ok(retry)
    after_state = _stat(path)
    after_head = _head(store, path)
    if after_head["records"] != before_head["records"] + 1 or after_state is None:
        raise R1Error("crash retry did not preserve one durable row")
    return {
        "ledger": path.name,
        "before": {"head": before_head, "file_state": before_state},
        "after": {"head": after_head, "file_state": after_state},
        "payload_identity_sha256": hashlib.sha256(
            payload["decision_id"].encode()
        ).hexdigest(),
        "first": first,
        "retry": retry,
        "retry_result": {
            "record_sha256": result.get("record_sha256"),
            "duplicate_appends": 0,
        },
        "original_target_offset": target["offset"],
    }


def _run(args: argparse.Namespace) -> tuple[str, Path, dict[str, Any]]:
    if sys.platform != "darwin":
        raise R1Error("R1 requires Darwin APFS clone and proc_pid_rusage")
    production_input = args.production_root.expanduser()
    source_input = args.source_root.expanduser()
    output_input = args.output.expanduser()
    if production_input.is_symlink() or source_input.is_symlink():
        raise R1Error("production/source symlink roots are forbidden")
    if output_input.exists() and output_input.is_symlink():
        raise R1Error("output symlink is forbidden")
    production = production_input.resolve(strict=True)
    source_root = source_input.resolve(strict=True)
    output = output_input.resolve(strict=False)
    if not production.is_dir() or _paths_overlap(source_root, production):
        raise R1Error("production/source roots overlap")
    if _paths_overlap(output, production) or _paths_overlap(output, source_root):
        raise R1Error("output overlaps protected root")

    clone_evidence: list[dict[str, Any]] = []
    protected_roots = (source_root, output)
    with _clone_context(production, clone_evidence, protected_roots) as module_clone:
        with _env(
            {
                "CHRONOVISOR_ROOT": str(module_clone),
                "CHRONOVISOR_READ_ONLY": "0",
                "PYTHONDONTWRITEBYTECODE": "1",
            }
        ):
            parity, distill, store, catalog, raw_store = _load(source_root)

    production_before = _production_snapshot(
        store, catalog, raw_store, production, args.dashboard_url
    )
    r0_anchor = _load_r0_anchor(store, source_root)
    _assert_expected_baseline(production_before, r0_anchor)
    runtime_identity = parity._runtime_identity(source_root, args.source_commit)
    operations: list[dict[str, Any]] = []

    with _stage_guards(distill):
        for name, payload in (
            ("rally-manifest.jsonl", {"kind": "r1-rally-probe", "probe": True}),
            (
                "label-ledger.jsonl",
                {"authority": "teacher-only", "assignment": {"probe": False}},
            ),
        ):
            with _clone_context(production, clone_evidence, protected_roots) as clone:
                path = store.distillation_dir(clone) / name
                _head(store, path)
                if name == store.LABEL_LEDGER_FILE:
                    operations.append(_label_projection_bootstrap(store, clone))
                operations.append(_steady_ledger(store, clone, name, payload))

        with _clone_context(production, clone_evidence, protected_roots) as clone:
            operations.append(_candidate_hot_path(store, catalog, clone))

        for mutation_name, mutation in (
            ("partial_tail", _append_partial),
            ("separator_only", _append_separator),
        ):
            with _clone_context(production, clone_evidence, protected_roots) as clone:
                operation = _recovery(
                    store,
                    clone,
                    "candidate-ledger.jsonl",
                    mutation,
                )
                operation["scenario"] = mutation_name
                operations.append(operation)
        with _clone_context(production, clone_evidence, protected_roots) as clone:
            operation = _recovery(
                store,
                clone,
                "candidate-ledger.jsonl",
                lambda path, head: _mismatch_checkpoint(store, path, head),
            )
            operation["scenario"] = "head_mismatch"
            operations.append(operation)

        with _clone_context(production, clone_evidence, protected_roots) as clone:
            operations.append(_unique_duplicate(store, clone, "steady"))
        with _clone_context(production, clone_evidence, protected_roots) as clone:
            operations.append(
                _unique_duplicate(
                    store,
                    clone,
                    "sidecar_deleted",
                    lambda path, index: (
                        index.unlink(missing_ok=True),
                        store._unique_index_checkpoint_path(index).unlink(
                            missing_ok=True
                        ),
                    ),
                )
            )
        with _clone_context(production, clone_evidence, protected_roots) as clone:
            operations.append(
                _unique_duplicate(
                    store, clone, "logical_delete", _logical_delete_tamper
                )
            )
        with _clone_context(production, clone_evidence, protected_roots) as clone:
            operations.append(
                _unique_duplicate(store, clone, "separator_only", _append_separator)
            )
        with _clone_context(production, clone_evidence, protected_roots) as clone:
            operations.append(_unique_crash(store, clone))

    production_after = _production_snapshot(
        store, catalog, raw_store, production, args.dashboard_url
    )
    production_before_static = _static(production_before)
    production_after_static = _static(production_after)
    if production_before_static != production_after_static:
        raise R1Error("production changed during R1 measurement")
    _assert_r0_anchor(production_after, r0_anchor)
    if parity._runtime_identity(source_root, args.source_commit) != runtime_identity:
        raise R1Error("source identity changed during R1 measurement")
    payload = {
        "baseline_reference": {
            "r1_reference_commit": "b20ee23d530b60242a51968e5805cba7c776e962",
            "r0_evidence": r0_anchor["artifact_id"],
            "r0_anchor_file_sha256": r0_anchor["file_sha256"],
            "r0_anchor_seal_sha256": r0_anchor["seal_sha256"],
        },
        "measurement_identity": runtime_identity,
        "contract": {
            "platform": sys.platform,
            "clone_protocol": "apfs-cow",
            "clones": clone_evidence,
            "production_mutation": False,
            "offline_stage_guards": True,
            "raw_body_in_evidence": False,
            "acceptance_scope": "R1 storage gates only",
            "advance_latency_gates": {
                "status": "not_certified",
                "authority": "R2",
            },
        },
        "expected_baseline": EXPECTED_BASELINE,
        "production": {
            "before": production_before,
            "after": production_after,
            "anchor_sha256": _digest(production_before_static),
        },
        "operations": _redact_operations(operations),
        "summary": {
            "production_unchanged": True,
            "duplicate_appends": 0,
            "old_prefix_scans_in_steady_paths": 0,
            "operation_count": len(operations),
        },
    }
    artifact_id, artifact_path, artifact = store.write_immutable(
        output, payload, schema=SCHEMA
    )
    return artifact_id, artifact_path, artifact


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--dashboard-url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        artifact_id, artifact_path, artifact = _run(args)
        print(
            json.dumps(
                {
                    "schema": artifact["schema"],
                    "artifact_id": artifact_id,
                    "path": str(artifact_path),
                },
                sort_keys=True,
            )
        )
    except (R1Error, R0Error, OSError, ValueError, sqlite3.Error) as exc:
        print(f"r1 harness failed: {type(exc).__name__}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
