"""Sealed cross-process heartbeat protocol for the autonomy watchdog."""

from __future__ import annotations

import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llm_wiki_mcp.durable_state import (
    DurableStateError,
    file_lock,
    read_sealed_json,
    write_sealed_json,
)


HEARTBEAT_SCHEMA = "llm-wiki.deadman-heartbeat.v1"
THRESHOLD_SCHEMA_VERSION = 1
THRESHOLD_POLICY = {
    "version": THRESHOLD_SCHEMA_VERSION,
    "minimum_failure_samples": 2,
    "recovery_samples": 2,
    "cooldown_seconds": 3600,
    "incident_budget_per_day": 4,
}


def boot_id() -> str:
    try:
        proc = subprocess.run(
            ["/usr/sbin/sysctl", "-n", "kern.boottime"],
            text=True,
            capture_output=True,
            timeout=2,
            check=False,
        )
    except Exception:
        return "unknown"
    return proc.stdout.strip() or "unknown"


def _as_utc(value: datetime | None) -> datetime:
    current = value or datetime.now(timezone.utc)
    if current.tzinfo is None:
        return current.replace(tzinfo=timezone.utc)
    return current.astimezone(timezone.utc)


def inspect_heartbeat(
    path: Path,
    *,
    expected_role: str,
    max_age_seconds: int,
    now: datetime | None = None,
) -> dict[str, Any]:
    current = _as_utc(now)
    try:
        payload = read_sealed_json(path)
    except (DurableStateError, OSError) as exc:
        return {
            "status": "missing" if not path.exists() else "invalid",
            "role": expected_role,
            "path": str(path),
            "error": str(exc),
        }
    if payload.get("schema") != HEARTBEAT_SCHEMA:
        return {"status": "invalid", "role": expected_role, "path": str(path)}
    if payload.get("role") != expected_role:
        return {"status": "invalid", "role": expected_role, "path": str(path)}
    sequence = payload.get("sequence")
    wall_time = payload.get("wall_time")
    monotonic_ns = payload.get("monotonic_ns")
    heartbeat_boot_id = payload.get("boot_id")
    if (
        isinstance(sequence, bool)
        or not isinstance(sequence, int)
        or sequence < 1
        or not isinstance(wall_time, str)
        or isinstance(monotonic_ns, bool)
        or not isinstance(monotonic_ns, int)
        or monotonic_ns < 0
        or not isinstance(heartbeat_boot_id, str)
        or not heartbeat_boot_id
        or payload.get("threshold_schema_version") != THRESHOLD_SCHEMA_VERSION
    ):
        return {"status": "invalid", "role": expected_role, "path": str(path)}
    try:
        observed = datetime.fromisoformat(wall_time.replace("Z", "+00:00"))
    except ValueError:
        return {"status": "invalid", "role": expected_role, "path": str(path)}
    observed = _as_utc(observed)
    age = (current - observed).total_seconds()
    if age < -300:
        status = "clock_regression"
    elif age > max(1, int(max_age_seconds)):
        status = "stale"
    else:
        status = "ok"
    return {
        "status": status,
        "role": expected_role,
        "path": str(path),
        "sequence": sequence,
        "wall_time": wall_time,
        "age_seconds": round(age, 3),
        "boot_id": payload.get("boot_id"),
        "reported_status": payload.get("reported_status"),
        "peer_sequence": payload.get("peer_sequence"),
        "peer_status": payload.get("peer_status"),
        "threshold_schema_version": payload.get("threshold_schema_version"),
        "threshold_policy": payload.get("threshold_policy"),
        "threshold_state": payload.get("threshold_state"),
        "incident_budget_remaining": payload.get("incident_budget_remaining"),
    }


def write_heartbeat(
    path: Path,
    *,
    role: str,
    reported_status: str,
    peer: dict[str, Any] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    lock = path.with_name(f"{path.name}.lock")
    with file_lock(lock, exclusive=True):
        try:
            previous = read_sealed_json(path)
        except DurableStateError:
            previous = {}
        prior_sequence = previous.get("sequence")
        sequence = (
            int(prior_sequence) + 1
            if isinstance(prior_sequence, int) and not isinstance(prior_sequence, bool)
            else 1
        )
        current = _as_utc(now)
        payload = {
            "schema": HEARTBEAT_SCHEMA,
            "threshold_schema_version": THRESHOLD_SCHEMA_VERSION,
            "role": role,
            "sequence": sequence,
            "wall_time": current.isoformat(timespec="seconds").replace("+00:00", "Z"),
            "monotonic_ns": time.monotonic_ns(),
            "boot_id": boot_id(),
            "pid": os.getpid(),
            "reported_status": reported_status,
            "peer_sequence": peer.get("sequence") if isinstance(peer, dict) else None,
            "peer_status": peer.get("status") if isinstance(peer, dict) else None,
            "threshold_policy": THRESHOLD_POLICY,
        }
        return write_sealed_json(path, payload, backup=True)
