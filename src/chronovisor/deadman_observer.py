#!/usr/bin/python3
"""Package-independent observer for the Chronovisor autonomy watchdog.

This file intentionally imports only the Python standard library.  The
installer copies it outside the uv/Chronovisor package archive so a broken import,
bad deployment, or dead main watchdog cannot also remove its observer.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


SCHEMA = "chronovisor.deadman-heartbeat.v1"
SEALED_PREVIOUS_SCHEMA = "llm-wiki.deadman-heartbeat.v1"
THRESHOLD_POLICY = {
    "version": 1,
    "minimum_failure_samples": 2,
    "recovery_samples": 2,
    "cooldown_seconds": 3600,
    "incident_budget_per_day": 4,
}


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
    if payload.get("schema") not in {SCHEMA, SEALED_PREVIOUS_SCHEMA} or payload.get(
        "role"
    ) != "main_watchdog":
        return {"status": "invalid", "error": "unexpected heartbeat identity"}
    wall_time = payload.get("wall_time")
    sequence = payload.get("sequence")
    if not isinstance(wall_time, str) or not isinstance(sequence, int):
        return {"status": "invalid", "error": "heartbeat fields are malformed"}
    try:
        observed = datetime.fromisoformat(wall_time.replace("Z", "+00:00"))
    except ValueError:
        return {"status": "invalid", "error": "heartbeat wall time is invalid"}
    age = (_utc(now) - observed.astimezone(timezone.utc)).total_seconds()
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
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


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
    current = value or datetime.now(timezone.utc)
    if current.tzinfo is None:
        return current.replace(tzinfo=timezone.utc)
    return current.astimezone(timezone.utc)


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
        try:
            last_time = _utc(
                datetime.fromisoformat(last_time_raw.replace("Z", "+00:00"))
            )
        except ValueError:
            pass
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
        "schema": "chronovisor.deadman-threshold-state.v1",
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
                    "schema": "chronovisor.deadman-incident.v1",
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
    result = run_once(
        args.chronovisor_root.expanduser().resolve(strict=False),
        max_main_age_seconds=max(1, args.max_main_age_seconds),
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
