#!/usr/bin/env python3.14
"""Package-independent observer for the Chronovisor autonomy watchdog.

This file intentionally imports only the Python standard library.  The
installer copies it outside the uv/Chronovisor package archive so a broken import,
bad deployment, or dead main watchdog cannot also remove its observer.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import re
import stat
import subprocess
import tempfile
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

SCHEMA = "chronovisor.ops.deadman-heartbeat.v1"
THRESHOLD_POLICY = {
    "version": 1,
    "minimum_failure_samples": 2,
    "recovery_samples": 2,
    "cooldown_seconds": 3600,
    "incident_budget_per_day": 4,
}
_LIVE_LAYOUT_LOG_SHA256 = "479d37f9f41843b9847e18adee9dcce1fc26cb8862341f94b996ca26037977d0"
_LIVE_LAYOUT_SCHEMA_SHA256 = "0cc24c0be93ed3eef4ab534ccb95e77fc5e377529255ed07a52c4c509abf6a7b"
_EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
_INDEX_PREFIX = b"---\nokf_version: '0.2'\n---\n# Chronovisor pages\n"
_FINAL_RECEIPT_FIELDS = {
    "schema",
    "version",
    "run_id",
    "state",
    "manifest_sha256",
    "before_manifest_sha256",
    "after_manifest_sha256",
    "transaction_version",
    "manifest_schema",
    "okf_version",
    "status_mapping_cohorts",
    "rollback_recutover",
    "rebuild_proof",
    "activity_prefix",
    "activity_suffix",
    "pages_log_sha256",
    "system_schema_sha256",
    "seal_sha256",
}


class WriterGateBlocked(RuntimeError):
    pass


def _reject_symlink_ancestors(path: Path) -> None:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        if stat.S_ISLNK(os.lstat(current).st_mode):
            raise WriterGateBlocked


def _entries(directory_fd: int) -> dict[str, int]:
    with os.scandir(directory_fd) as iterator:
        return {
            entry.name: entry.stat(follow_symlinks=False).st_mode
            for entry in iterator
        }


def _require_safe_tree(directory_fd: int, flags: int) -> None:
    for name, mode in _entries(directory_fd).items():
        if stat.S_ISREG(mode):
            continue
        if not stat.S_ISDIR(mode):
            raise WriterGateBlocked
        child_fd = os.open(name, flags, dir_fd=directory_fd)
        try:
            _require_safe_tree(child_fd, flags)
        finally:
            os.close(child_fd)


def _read_regular(directory_fd: int, name: str, *, limit: int) -> bytes:
    flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(name, flags, dir_fd=directory_fd)
    try:
        snapshot = os.fstat(descriptor)
        if not stat.S_ISREG(snapshot.st_mode) or snapshot.st_size > limit:
            raise WriterGateBlocked
        raw = os.read(descriptor, snapshot.st_size + 1)
        if len(raw) != snapshot.st_size:
            raise WriterGateBlocked
        return raw
    finally:
        os.close(descriptor)


def _require_regular(directory_fd: int, name: str, *, limit: int) -> None:
    flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(name, flags, dir_fd=directory_fd)
    try:
        snapshot = os.fstat(descriptor)
        if not stat.S_ISREG(snapshot.st_mode) or snapshot.st_size > limit:
            raise WriterGateBlocked
    finally:
        os.close(descriptor)


def _require_segment(
    directory_fd: int, name: str, *, offset: int, length: int, sha256: str
) -> None:
    flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(name, flags, dir_fd=directory_fd)
    try:
        snapshot = os.fstat(descriptor)
        if (
            not stat.S_ISREG(snapshot.st_mode)
            or snapshot.st_size < offset + length
            or offset < 0
            or length < 0
        ):
            raise WriterGateBlocked
        os.lseek(descriptor, offset, os.SEEK_SET)
        digest = hashlib.sha256()
        remaining = length
        while remaining:
            chunk = os.read(descriptor, min(64 * 1024, remaining))
            if not chunk:
                raise WriterGateBlocked
            digest.update(chunk)
            remaining -= len(chunk)
        if digest.hexdigest() != sha256:
            raise WriterGateBlocked
    finally:
        os.close(descriptor)


def _read_regular_prefix(directory_fd: int, name: str, prefix: bytes, *, limit: int) -> None:
    flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(name, flags, dir_fd=directory_fd)
    try:
        snapshot = os.fstat(descriptor)
        if (
            not stat.S_ISREG(snapshot.st_mode)
            or snapshot.st_size > limit
            or os.read(descriptor, len(prefix)) != prefix
        ):
            raise WriterGateBlocked
    finally:
        os.close(descriptor)


def _require_live_layout_proof(
    root_fd: int, runtime_fd: int, directory_flags: int
) -> None:
    try:
        proof = verify(
            json.loads(
                _read_regular(runtime_fd, "bootstrap-layout.json", limit=4096)
            )
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise WriterGateBlocked from exc
    if (
        proof.get("schema") != "chronovisor.live-layout.v1"
        or proof.get("version") != 1
        or proof.get("state") != "ready"
        or proof.get("index_renderer_version") != 1
        or proof.get("paths")
        != {
            "index": "pages/index.md",
            "log": "pages/log.md",
            "schema": "system/schema.md",
            "activity": "runtime/activity.jsonl",
        }
        or proof.get("log_sha256") != _LIVE_LAYOUT_LOG_SHA256
        or proof.get("schema_sha256") != _LIVE_LAYOUT_SCHEMA_SHA256
        or proof.get("activity_prefix")
        != {"length": 0, "sha256": _EMPTY_SHA256}
    ):
        raise WriterGateBlocked
    pages_fd = os.open("pages", directory_flags, dir_fd=root_fd)
    system_fd = os.open("system", directory_flags, dir_fd=root_fd)
    try:
        _read_regular_prefix(
            pages_fd, "index.md", _INDEX_PREFIX, limit=16 * 1024 * 1024
        )
        log = _read_regular(pages_fd, "log.md", limit=4096)
        schema = _read_regular(system_fd, "schema.md", limit=64 * 1024)
        _require_regular(runtime_fd, "activity.jsonl", limit=64 * 1024 * 1024)
    finally:
        os.close(system_fd)
        os.close(pages_fd)
    if (
        hashlib.sha256(log).hexdigest() != _LIVE_LAYOUT_LOG_SHA256
        or hashlib.sha256(schema).hexdigest() != _LIVE_LAYOUT_SCHEMA_SHA256
    ):
        raise WriterGateBlocked


def _require_finalized_migration_receipt(
    runtime_fd: int, directory_flags: int
) -> dict[str, Any]:
    migrations_fd = workspace_fd = -1
    try:
        migrations_fd = os.open("migrations", directory_flags, dir_fd=runtime_fd)
        entries = _entries(migrations_fd)
        if len(entries) != 1:
            raise WriterGateBlocked
        run_id, mode = next(iter(entries.items()))
        if (
            not stat.S_ISDIR(mode)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", run_id) is None
        ):
            raise WriterGateBlocked
        workspace_fd = os.open(run_id, directory_flags, dir_fd=migrations_fd)
        workspace_entries = _entries(workspace_fd)
        if set(workspace_entries) != {"receipt.json"} or not stat.S_ISREG(
            workspace_entries["receipt.json"]
        ):
            raise WriterGateBlocked
        raw = _read_regular(workspace_fd, "receipt.json", limit=64 * 1024)
        receipt = verify(json.loads(raw))
        if raw != canonical_bytes(receipt):
            raise WriterGateBlocked
        status_mapping = {
            "missing": "stable",
            "active": "stable",
            "draft": "draft",
            "stable": "stable",
            "deprecated": "deprecated",
            "archived": "deprecated",
        }
        expected_cohorts = [
            (scope, input_status, output_status)
            for scope in ("pages", "system")
            for input_status, output_status in status_mapping.items()
        ]
        cohorts = receipt.get("status_mapping_cohorts")
        rebuild = receipt.get("rebuild_proof")
        if (
            set(receipt) != _FINAL_RECEIPT_FIELDS
            or receipt.get("schema") != "chronovisor.okf-migration-receipt.v2"
            or receipt.get("version") != 2
            or receipt.get("run_id") != run_id
            or receipt.get("state") != "finalized-v2"
            or receipt.get("transaction_version") != 1
            or receipt.get("manifest_schema")
            != "chronovisor.okf-migration-manifest.v1"
            or receipt.get("okf_version") != "0.2"
            or receipt.get("rollback_recutover")
            != {"rollback": "complete", "recutover": "complete"}
            or not isinstance(cohorts, list)
            or len(cohorts) != len(expected_cohorts)
            or not isinstance(rebuild, dict)
            or set(rebuild) != {"derived_generation", "sha256", "stable_page_count"}
        ):
            raise WriterGateBlocked
        sha_fields = (
            "manifest_sha256",
            "before_manifest_sha256",
            "after_manifest_sha256",
            "pages_log_sha256",
            "system_schema_sha256",
        )
        if any(
            not isinstance(receipt.get(field), str)
            or re.fullmatch(r"[0-9a-f]{64}", receipt[field]) is None
            for field in sha_fields
        ):
            raise WriterGateBlocked
        for index, cohort in enumerate(cohorts):
            expected = expected_cohorts[index]
            if (
                not isinstance(cohort, dict)
                or set(cohort)
                != {
                    "scope",
                    "input_status",
                    "output_status",
                    "count",
                    "identity_set_sha256",
                }
                or (
                    cohort.get("scope"),
                    cohort.get("input_status"),
                    cohort.get("output_status"),
                )
                != expected
                or not isinstance(cohort.get("count"), int)
                or isinstance(cohort.get("count"), bool)
                or cohort["count"] < 0
                or not isinstance(cohort.get("identity_set_sha256"), str)
                or re.fullmatch(r"[0-9a-f]{64}", cohort["identity_set_sha256"])
                is None
            ):
                raise WriterGateBlocked
        generation = rebuild.get("derived_generation")
        stable_page_count = rebuild.get("stable_page_count")
        if (
            not isinstance(generation, str)
            or re.fullmatch(r"[a-z0-9-]{1,128}", generation) is None
            or not isinstance(rebuild.get("sha256"), str)
            or re.fullmatch(r"[0-9a-f]{64}", rebuild["sha256"]) is None
            or not isinstance(stable_page_count, int)
            or isinstance(stable_page_count, bool)
            or stable_page_count < 0
        ):
            raise WriterGateBlocked
        for field, with_events in (
            ("activity_prefix", True),
            ("activity_suffix", False),
        ):
            identity = receipt.get(field)
            expected_fields = {"length", "sha256"}
            if with_events:
                expected_fields.add("event_ids_sha256")
            if (
                not isinstance(identity, dict)
                or set(identity) != expected_fields
                or not isinstance(identity.get("length"), int)
                or isinstance(identity.get("length"), bool)
                or identity["length"] < 0
                or not isinstance(identity.get("sha256"), str)
                or re.fullmatch(r"[0-9a-f]{64}", identity["sha256"]) is None
                or (
                    with_events
                    and (
                        not isinstance(identity.get("event_ids_sha256"), str)
                        or re.fullmatch(
                            r"[0-9a-f]{64}", identity["event_ids_sha256"]
                        )
                        is None
                    )
                )
            ):
                raise WriterGateBlocked
        return receipt
    except (TypeError, UnicodeError, ValueError):
        raise WriterGateBlocked from None
    finally:
        if workspace_fd >= 0:
            os.close(workspace_fd)
        if migrations_fd >= 0:
            os.close(migrations_fd)


def _require_final_receipt_layout(
    root_fd: int,
    runtime_fd: int,
    directory_flags: int,
    receipt: dict[str, Any],
) -> None:
    pages_fd = os.open("pages", directory_flags, dir_fd=root_fd)
    system_fd = os.open("system", directory_flags, dir_fd=root_fd)
    try:
        _read_regular_prefix(
            pages_fd, "index.md", _INDEX_PREFIX, limit=16 * 1024 * 1024
        )
        log = _read_regular(pages_fd, "log.md", limit=4096)
        schema = _read_regular(system_fd, "schema.md", limit=64 * 1024)
        if (
            hashlib.sha256(log).hexdigest() != receipt["pages_log_sha256"]
            or hashlib.sha256(schema).hexdigest()
            != receipt["system_schema_sha256"]
        ):
            raise WriterGateBlocked
    finally:
        os.close(system_fd)
        os.close(pages_fd)
    prefix = receipt["activity_prefix"]
    suffix = receipt["activity_suffix"]
    _require_segment(
        runtime_fd,
        "activity.jsonl",
        offset=0,
        length=prefix["length"],
        sha256=prefix["sha256"],
    )
    _require_segment(
        runtime_fd,
        "activity.jsonl",
        offset=prefix["length"],
        length=suffix["length"],
        sha256=suffix["sha256"],
    )


@contextlib.contextmanager
def writer_gate(chronovisor_root: Path):
    """Pin the root parent shared while rejecting non-legacy/migrating roots."""

    entered = False
    try:
        root = chronovisor_root.absolute()
        if root.parent == root:
            raise WriterGateBlocked
        _reject_symlink_ancestors(root)
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        flags |= getattr(os, "O_CLOEXEC", 0)
        parent_fd = os.open(root.parent, flags)
        root_fd = runtime_fd = -1
        try:
            fcntl.flock(parent_fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
            root_fd = os.open(root.name, flags, dir_fd=parent_fd)
            entries = _entries(root_fd)
            if any(
                not (stat.S_ISREG(mode) or stat.S_ISDIR(mode))
                for mode in entries.values()
            ):
                raise WriterGateBlocked
            legacy_reserved = tuple(
                stat.S_ISREG(entries.get(name, 0))
                for name in ("index.md", "log.md", "schema.md")
            )
            if any(legacy_reserved) and not all(legacy_reserved):
                raise WriterGateBlocked
            if "config.toml" in entries and not stat.S_ISREG(entries["config.toml"]):
                raise WriterGateBlocked
            for name in ("logs", "runtime"):
                if name in entries and not stat.S_ISDIR(entries[name]):
                    raise WriterGateBlocked
            for name in ("raw", "pages", "system"):
                if name not in entries:
                    continue
                if not stat.S_ISDIR(entries[name]):
                    raise WriterGateBlocked
                child_fd = os.open(name, flags, dir_fd=root_fd)
                try:
                    _require_safe_tree(child_fd, flags)
                finally:
                    os.close(child_fd)
            runtime_fd = os.open("runtime", flags, dir_fd=root_fd)
            runtime_entries = _entries(runtime_fd)
            final_receipt = None
            if "migrations" in runtime_entries:
                final_receipt = _require_finalized_migration_receipt(
                    runtime_fd, flags
                )
            if final_receipt is not None or not all(legacy_reserved):
                if final_receipt is None and any(legacy_reserved):
                    raise WriterGateBlocked
                if final_receipt is not None and any(
                    name in entries for name in ("index.md", "log.md", "schema.md")
                ):
                    raise WriterGateBlocked
                if not stat.S_ISREG(runtime_entries.get("activity.jsonl", 0)):
                    raise WriterGateBlocked
                for directory, names in (
                    ("pages", ("index.md", "log.md")),
                    ("system", ("schema.md",)),
                ):
                    if not stat.S_ISDIR(entries.get(directory, 0)):
                        raise WriterGateBlocked
                    directory_fd = os.open(directory, flags, dir_fd=root_fd)
                    try:
                        child_entries = _entries(directory_fd)
                        if any(
                            not stat.S_ISREG(child_entries.get(name, 0))
                            for name in names
                        ):
                            raise WriterGateBlocked
                    finally:
                        os.close(directory_fd)
                if final_receipt is None:
                    _require_live_layout_proof(root_fd, runtime_fd, flags)
                else:
                    if not stat.S_ISDIR(entries.get("raw", 0)):
                        raise WriterGateBlocked
                    _require_final_receipt_layout(
                        root_fd, runtime_fd, flags, final_receipt
                    )
            entered = True
            yield
        finally:
            if runtime_fd >= 0:
                os.close(runtime_fd)
            if root_fd >= 0:
                os.close(root_fd)
            with contextlib.suppress(OSError):
                fcntl.flock(parent_fd, fcntl.LOCK_UN)
            os.close(parent_fd)
    except WriterGateBlocked:
        raise
    except OSError as exc:
        if entered:
            raise
        raise WriterGateBlocked from exc


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def seal(payload: dict[str, Any]) -> dict[str, Any]:
    unsigned = {key: value for key, value in payload.items() if key != "seal_sha256"}
    return {
        **unsigned,
        "seal_sha256": hashlib.sha256(canonical_bytes(unsigned)).hexdigest(),
    }


def verify(payload: object) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("heartbeat is not an object")
    observed = payload.get("seal_sha256")
    unsigned = {key: value for key, value in payload.items() if key != "seal_sha256"}
    if observed != hashlib.sha256(canonical_bytes(unsigned)).hexdigest():
        raise ValueError("heartbeat seal mismatch")
    return payload


def read(path: Path) -> dict[str, Any]:
    return verify(json.loads(path.read_text(encoding="utf-8")))


def boot_id() -> str:
    try:
        result = subprocess.run(
            ["/usr/sbin/sysctl", "-n", "kern.boottime"],
            text=True,
            capture_output=True,
            timeout=2,
            check=False,
        )
    except Exception:
        return "unknown"
    return result.stdout.strip() or "unknown"


def inspect(
    path: Path,
    *,
    max_age_seconds: int,
    now: datetime | None = None,
) -> dict[str, Any]:
    try:
        payload = read(path)
    except FileNotFoundError:
        return {"status": "missing"}
    except Exception as exc:
        return {"status": "invalid", "error": str(exc)}
    if payload.get("schema") != SCHEMA or payload.get("role") != "main_watchdog":
        return {"status": "invalid", "error": "unexpected heartbeat identity"}
    wall_time = payload.get("wall_time")
    sequence = payload.get("sequence")
    if not isinstance(wall_time, str) or not isinstance(sequence, int):
        return {"status": "invalid", "error": "heartbeat fields are malformed"}
    try:
        observed = datetime.fromisoformat(wall_time.replace("Z", "+00:00"))
    except ValueError:
        return {"status": "invalid", "error": "heartbeat wall time is invalid"}
    age = (_utc(now) - observed.astimezone(UTC)).total_seconds()
    if age < -300:
        status = "clock_regression"
    elif age > max_age_seconds:
        status = "stale"
    else:
        status = "ok"
    return {
        "status": status,
        "sequence": sequence,
        "age_seconds": round(age, 3),
        "wall_time": wall_time,
    }


def atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = canonical_bytes(seal(payload))
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None:
            with contextlib.suppress(FileNotFoundError):
                temporary.unlink()


def append_incident(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        os.write(descriptor, canonical_bytes(payload))
        os.fsync(descriptor)
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _utc(value: datetime | None = None) -> datetime:
    current = value or datetime.now(UTC)
    if current.tzinfo is None:
        return current.replace(tzinfo=UTC)
    return current.astimezone(UTC)


def _threshold_transition(
    previous: dict[str, Any],
    peer: dict[str, Any],
    *,
    now: datetime,
) -> tuple[dict[str, Any], bool]:
    """Apply versioned debounce, hysteresis, cooldown, and incident budget."""

    if previous.get("threshold_policy") != THRESHOLD_POLICY:
        previous = {}
    peer_status = str(peer.get("status") or "invalid")
    unhealthy = peer_status != "ok"
    prior_status = str(previous.get("status") or "healthy")
    failures = int(previous.get("consecutive_failures") or 0)
    recoveries = int(previous.get("consecutive_recoveries") or 0)
    if unhealthy:
        failures += 1
        recoveries = 0
        status = (
            "alert"
            if failures >= THRESHOLD_POLICY["minimum_failure_samples"]
            else prior_status
        )
    else:
        failures = 0
        recoveries += 1
        status = prior_status
        if recoveries >= THRESHOLD_POLICY["recovery_samples"]:
            status = "healthy"

    cutoff = now - timedelta(days=1)
    incident_times: list[str] = []
    for value in previous.get("incident_times", []):
        if not isinstance(value, str):
            continue
        try:
            parsed = _utc(datetime.fromisoformat(value.replace("Z", "+00:00")))
        except ValueError:
            continue
        if cutoff <= parsed <= now + timedelta(minutes=5):
            incident_times.append(value)
    dedupe_key = f"main_watchdog:{peer_status}:{peer.get('sequence')}"
    last_key = previous.get("last_incident_dedupe_key")
    last_time_raw = previous.get("last_incident_at")
    last_time: datetime | None = None
    if isinstance(last_time_raw, str):
        with contextlib.suppress(ValueError):
            last_time = _utc(
                datetime.fromisoformat(last_time_raw.replace("Z", "+00:00"))
            )
    cooldown_elapsed = (
        last_key != dedupe_key
        or last_time is None
        or (now - last_time).total_seconds()
        >= THRESHOLD_POLICY["cooldown_seconds"]
    )
    emit = bool(
        status == "alert"
        and unhealthy
        and cooldown_elapsed
        and len(incident_times) < THRESHOLD_POLICY["incident_budget_per_day"]
    )
    now_text = now.isoformat(timespec="seconds").replace("+00:00", "Z")
    if emit:
        incident_times.append(now_text)
        last_key = dedupe_key
        last_time_raw = now_text
    state = {
        "schema": "chronovisor.ops.deadman-threshold-state.v1",
        "threshold_policy": THRESHOLD_POLICY,
        "status": status,
        "consecutive_failures": failures,
        "consecutive_recoveries": recoveries,
        "last_peer_status": peer_status,
        "last_peer_sequence": peer.get("sequence"),
        "last_incident_dedupe_key": last_key,
        "last_incident_at": last_time_raw,
        "incident_times": incident_times,
        "incident_budget_remaining": max(
            0,
            THRESHOLD_POLICY["incident_budget_per_day"] - len(incident_times),
        ),
        "updated_at": now_text,
    }
    return state, emit


def run_once(
    chronovisor_root: Path,
    *,
    max_main_age_seconds: int,
    now: datetime | None = None,
) -> dict[str, Any]:
    autonomy = chronovisor_root / "autonomy"
    main_path = autonomy / "watchdog-heartbeat.json"
    observer_path = autonomy / "observer-heartbeat.json"
    incident_path = autonomy / "deadman-incidents.jsonl"
    threshold_path = autonomy / "observer-threshold-state.json"
    lock_path = autonomy / "observer-heartbeat.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        peer = inspect(
            main_path,
            max_age_seconds=max_main_age_seconds,
            now=now,
        )
        try:
            previous_threshold = read(threshold_path)
        except Exception:
            previous_threshold = {}
        observed_now = _utc(now)
        threshold, emit_incident = _threshold_transition(
            previous_threshold,
            peer,
            now=observed_now,
        )
        atomic_write(threshold_path, threshold)
        try:
            previous = read(observer_path)
        except Exception:
            previous = {}
        prior = previous.get("sequence")
        sequence = int(prior) + 1 if isinstance(prior, int) else 1
        now_text = observed_now.isoformat(timespec="seconds").replace(
            "+00:00", "Z"
        )
        payload = {
            "schema": SCHEMA,
            "threshold_schema_version": 1,
            "role": "independent_observer",
            "sequence": sequence,
            "wall_time": now_text,
            "monotonic_ns": time.monotonic_ns(),
            "boot_id": boot_id(),
            "pid": os.getpid(),
            "reported_status": (
                "alert" if threshold.get("status") == "alert" else "ok"
            ),
            "peer_sequence": peer.get("sequence"),
            "peer_status": peer.get("status"),
            "threshold_policy": THRESHOLD_POLICY,
            "threshold_state": threshold.get("status"),
            "incident_budget_remaining": threshold.get(
                "incident_budget_remaining"
            ),
        }
        atomic_write(observer_path, payload)
        if emit_incident:
            append_incident(
                incident_path,
                {
                    "schema": "chronovisor.ops.deadman-incident.v1",
                    "ts": now_text,
                    "dedupe_key": threshold["last_incident_dedupe_key"],
                    "observer_sequence": sequence,
                    "peer": peer,
                    "threshold_policy": THRESHOLD_POLICY,
                    "threshold_state": threshold,
                    "requires_semantic_judgment": False,
                    "frontier_allowed": False,
                },
            )
        return {
            "status": payload["reported_status"],
            "peer": peer,
            "sequence": sequence,
            "threshold": threshold,
            "incident_emitted": emit_incident,
        }
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--chronovisor-root", type=Path, default=Path.home() / ".chronovisor"
    )
    parser.add_argument("--max-main-age-seconds", type=int, default=1_200)
    args = parser.parse_args(argv)
    root = args.chronovisor_root.expanduser().absolute()
    try:
        with writer_gate(root):
            result = run_once(
                root,
                max_main_age_seconds=max(1, args.max_main_age_seconds),
            )
    except WriterGateBlocked:
        print(json.dumps({"status": "blocked", "category": "okf_startup_blocked"}))
        return 75
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
