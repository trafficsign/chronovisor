#!/usr/bin/env python3
"""Read-only duration-configurable burn/soak monitor for Chronovisor services.

Importing this module has no side effects.  Running it explicitly creates one
append-only JSONL evidence file under ``~/.chronovisor/runtime/burn-ins`` and only
reads launchd, process, Ollama, dashboard, and Chronovisor runtime state.  It
never bootstraps, kicks, stops, or otherwise mutates a service.
"""

from __future__ import annotations

import argparse
import dataclasses
import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from chronovisor.canonical_json import (
    canonical_json_bytes_stringifying as canonical_bytes,
)
from chronovisor.store import CHRONOVISOR_ROOT


RUNTIME_ROOT = CHRONOVISOR_ROOT / "runtime"
AUTONOMY_ROOT = CHRONOVISOR_ROOT / "autonomy"
LOG_ROOT = CHRONOVISOR_ROOT / "logs"
BURN_ROOT = RUNTIME_ROOT / "burn-ins"

DEFAULT_DURATION_SECONDS = 35 * 60
DEFAULT_SAMPLE_SECONDS = 60
DEFAULT_PROBE_SECONDS = 5
DEFAULT_PREFLIGHT_WAIT_SECONDS = 180
DEFAULT_FINAL_IDLE_WAIT_SECONDS = 180
DEFAULT_DASHBOARD_URL = "http://127.0.0.1:8765"
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
EXPECTED_COMMIT = os.environ.get("CHRONOVISOR_EXPECTED_COMMIT", "").strip()
MAX_RELATED_RSS_BYTES = 80 * 1024**3

SERVICE_LABELS = {
    "dashboard": "com.trafficsign.chronovisor-dashboard",
    "sleep": "com.trafficsign.chronovisor-sleep",
    "watchdog": "com.trafficsign.chronovisor-watchdog",
    "converge": "com.trafficsign.chronovisor-converge",
    "ingest": "com.trafficsign.chronovisor-ingest-drain",
    "observer": "com.trafficsign.chronovisor-deadman-observer",
}

TRACKED_FILES = {
    "watchdog_history": AUTONOMY_ROOT / "watchdog-history.jsonl",
    "watchdog_err": LOG_ROOT / "watchdog.launchd.err.log",
    "converge_out": LOG_ROOT / "converge.launchd.out.log",
    "converge_err": LOG_ROOT / "converge.launchd.err.log",
    "sleep_history": RUNTIME_ROOT / "sleep-cycle-history.jsonl",
    "ingest_out": LOG_ROOT / "ingest-drain.launchd.out.log",
    "ingest_err": LOG_ROOT / "ingest-drain.launchd.err.log",
    "dashboard_out": LOG_ROOT / "dashboard.launchd.out.log",
    "dashboard_err": LOG_ROOT / "dashboard.launchd.err.log",
}

DASHBOARD_ENDPOINTS = (
    "/",
    "/api/snapshot",
    "/api/status",
    "/api/local-consensus",
    "/api/frontier-repair",
    "/api/health",
    "/api/model-status",
    "/api/save-history",
)

ROLE_PATTERNS = {
    "dashboard": (
        re.compile(r"(?:^|[/\s])chronovisor-dashboard(?:\s|$)"),
        re.compile(r"chronovisor\.dashboard"),
    ),
    "ingest": (
        re.compile(r"(?:^|[/\s])chronovisor-ingest-drain(?:\s|$)"),
        re.compile(r"(?:^|[/\s])chronovisor-ingest-batch(?:\s|$)"),
        re.compile(r"chronovisor\.(?:ingest_batch|ingest_drain)"),
    ),
    "watchdog": (
        re.compile(r"(?:^|[/\s])chronovisor-watchdog(?:\s|$)"),
        re.compile(r"chronovisor\.autonomy.*watchdog"),
    ),
    "converge": (
        re.compile(r"(?:^|[/\s])chronovisor-converge(?:\s|$)"),
        re.compile(r"chronovisor\.converge_worker"),
    ),
    "sleep": (
        re.compile(r"(?:^|[/\s])chronovisor-sleep(?:\s|$)"),
        re.compile(r"chronovisor\.sleep_cycle"),
    ),
}

OLLAMA_PATTERNS = (
    re.compile(r"(?:^|[/\s])ollama(?:\s|$)"),
    re.compile(r"llama-server"),
    re.compile(r"ollama_llama_server"),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_fingerprint(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return {"status": "absent", "sha256": None, "bytes": 0}
    except OSError as exc:
        return {
            "status": "error",
            "error": str(exc),
            "sha256": None,
            "bytes": 0,
        }
    return {
        "status": "present",
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
    }


def repair_active_projection(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"status": "absent", "active": False, "incident_id": None}
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "status": "indeterminate",
            "active": None,
            "incident_id": None,
            "error": str(exc),
        }
    if not isinstance(payload, Mapping):
        return {
            "status": "indeterminate",
            "active": None,
            "incident_id": None,
            "error": "frontier repair state is not an object",
        }
    incident_id = payload.get("active_incident_id")
    if not isinstance(incident_id, str) or not incident_id:
        return {"status": "ok", "active": False, "incident_id": None}
    incidents = payload.get("incidents")
    incident = incidents.get(incident_id) if isinstance(incidents, Mapping) else None
    if not isinstance(incident, Mapping):
        return {
            "status": "indeterminate",
            "active": None,
            "incident_id": incident_id,
            "error": "active frontier repair incident is missing",
        }
    status = str(incident.get("status") or "")
    return {
        "status": "ok",
        "active": status in {"reserved", "started"},
        "incident_id": incident_id,
        "incident_status": status,
    }


def frontier_fingerprint(root: Path = CHRONOVISOR_ROOT) -> dict[str, Any]:
    """Mirror convergence_drain._frontier_fingerprint byte-for-byte in shape."""

    runtime_root = root / "runtime"
    repair_root = runtime_root / "frontier-repair"
    events_path = runtime_root / "events.jsonl"
    frontier_events: list[dict[str, Any]] = []
    event_status = "ok"
    event_error: str | None = None
    try:
        lines = events_path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        lines = []
    except OSError as exc:
        lines = []
        event_status = "error"
        event_error = str(exc)
    invalid_frontier_lines = 0
    if event_error is None:
        for line in lines:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                if '"source"' in line and "frontier" in line:
                    invalid_frontier_lines += 1
                continue
            if isinstance(row, dict) and row.get("source") == "frontier":
                frontier_events.append(row)
    frontier_starts = [
        row
        for row in frontier_events
        if "frontier | review started" in str(row.get("message") or "")
    ]
    event_projection: dict[str, Any] = {
        "status": event_status,
        "count": len(frontier_events),
        "invalid_frontier_lines": invalid_frontier_lines,
        "sha256": sha256_value(frontier_events) if event_error is None else None,
        "start_count": len(frontier_starts),
        "start_sha256": (
            sha256_value(frontier_starts) if event_error is None else None
        ),
    }
    if event_error is not None:
        event_projection["error"] = event_error

    active_root = runtime_root / "frontier-reviews" / "active"
    active_records: list[dict[str, Any]] = []
    active_status = "ok"
    active_error: str | None = None
    try:
        active_paths = sorted(active_root.glob("*.json"))
    except OSError as exc:
        active_paths = []
        active_status = "error"
        active_error = str(exc)
    for path in active_paths:
        active_records.append({"name": path.name, **file_fingerprint(path)})
    return {
        "repair": {
            "state": file_fingerprint(repair_root / "state.json"),
            "events": file_fingerprint(repair_root / "events.jsonl"),
            "active": repair_active_projection(repair_root / "state.json"),
        },
        "frontier_events": event_projection,
        "frontier_active": {
            "status": active_status,
            "count": len(active_records),
            "sha256": sha256_value(active_records),
            "records": active_records,
            **({"error": active_error} if active_error is not None else {}),
        },
    }


def frontier_baseline_error(value: object) -> str | None:
    if not isinstance(value, Mapping):
        return "frontier_baseline_missing"
    repair = value.get("repair")
    if not isinstance(repair, Mapping):
        return "frontier_repair_baseline_missing"
    for name in ("state", "events"):
        state = repair.get(name)
        if not isinstance(state, Mapping) or state.get("status") == "error":
            return f"frontier_repair_{name}_unreadable"
    active = repair.get("active")
    if not isinstance(active, Mapping) or active.get("active") is None:
        return "frontier_repair_activity_indeterminate"
    if active.get("active") is True:
        return "frontier_repair_already_active"
    events = value.get("frontier_events")
    if not isinstance(events, Mapping) or events.get("status") != "ok":
        return "frontier_event_ledger_unreadable"
    if int(events.get("invalid_frontier_lines") or 0) > 0:
        return "frontier_event_ledger_malformed"
    frontier_active = value.get("frontier_active")
    if not isinstance(frontier_active, Mapping) or frontier_active.get("status") != "ok":
        return "frontier_active_markers_unreadable"
    if int(frontier_active.get("count") or 0) > 0:
        return "frontier_review_already_active"
    return None


def run_command(args: list[str], timeout: float = 5.0) -> dict[str, Any]:
    started = time.monotonic()
    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "ok": False,
            "returncode": None,
            "stdout": "",
            "stderr": "",
            "error": f"{exc.__class__.__name__}: {exc}",
            "elapsed_ms": round((time.monotonic() - started) * 1000, 1),
        }
    return {
        "ok": result.returncode == 0,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "elapsed_ms": round((time.monotonic() - started) * 1000, 1),
    }


def parse_launchctl_print(output: str) -> dict[str, Any]:
    def match_int(name: str) -> int | None:
        match = re.search(rf"^\s*{re.escape(name)}\s*=\s*(-?\d+)\s*$", output, re.M)
        return int(match.group(1)) if match else None

    state_match = re.search(r"^\s*state\s*=\s*(.+?)\s*$", output, re.M)
    return {
        "state": state_match.group(1) if state_match else None,
        "runs": match_int("runs"),
        "pid": match_int("pid"),
        "last_exit_code": match_int("last exit code"),
    }


def launchctl_snapshot() -> dict[str, Any]:
    domain = f"gui/{os.getuid()}"
    snapshot: dict[str, Any] = {}
    for role, label in SERVICE_LABELS.items():
        result = run_command(["/bin/launchctl", "print", f"{domain}/{label}"])
        parsed = parse_launchctl_print(result["stdout"] if result["ok"] else "")
        snapshot[role] = {
            "label": label,
            "loaded": bool(result["ok"]),
            **parsed,
            **(
                {
                    "error": result.get("error")
                    or (result.get("stderr") or result.get("stdout") or "").strip()
                }
                if not result["ok"]
                else {}
            ),
        }
    return snapshot


def parse_swapusage(output: str) -> dict[str, Any]:
    match = re.search(
        r"total\s*=\s*([0-9.]+)([KMGTP])\s+"
        r"used\s*=\s*([0-9.]+)([KMGTP])\s+"
        r"free\s*=\s*([0-9.]+)([KMGTP])",
        output,
        re.I,
    )
    if not match:
        return {"available": False, "raw": output.strip(), "error": "unparsed"}
    powers = {"K": 1, "M": 2, "G": 3, "T": 4, "P": 5}

    def to_bytes(value: str, unit: str) -> int:
        return int(float(value) * (1024 ** powers[unit.upper()]))

    return {
        "available": True,
        "total_bytes": to_bytes(match.group(1), match.group(2)),
        "used_bytes": to_bytes(match.group(3), match.group(4)),
        "free_bytes": to_bytes(match.group(5), match.group(6)),
        "raw": output.strip(),
    }


def swap_snapshot() -> dict[str, Any]:
    result = run_command(["/usr/sbin/sysctl", "vm.swapusage"])
    if not result["ok"]:
        return {
            "available": False,
            "error": result.get("error") or result.get("stderr") or "sysctl failed",
        }
    return parse_swapusage(result["stdout"])


def process_snapshot() -> dict[str, Any]:
    result = run_command(
        ["/bin/ps", "-axo", "pid=,ppid=,rss=,etime=,command="], timeout=10
    )
    if not result["ok"]:
        return {"available": False, "error": result.get("error") or result["stderr"]}
    rows: list[dict[str, Any]] = []
    for line in result["stdout"].splitlines():
        match = re.match(r"\s*(\d+)\s+(\d+)\s+(\d+)\s+(\S+)\s+(.*)$", line)
        if not match:
            continue
        rows.append(
            {
                "pid": int(match.group(1)),
                "ppid": int(match.group(2)),
                "rss_bytes": int(match.group(3)) * 1024,
                "etime": match.group(4),
                "command": match.group(5),
            }
        )

    roles: dict[str, list[dict[str, Any]]] = {}
    seed_pids: set[int] = set()
    for role, patterns in ROLE_PATTERNS.items():
        matched = [
            row
            for row in rows
            if any(pattern.search(str(row["command"])) for pattern in patterns)
        ]
        roles[role] = matched
        seed_pids.update(int(row["pid"]) for row in matched)

    related_pids = set(seed_pids)
    changed = True
    while changed:
        changed = False
        for row in rows:
            if int(row["ppid"]) in related_pids and int(row["pid"]) not in related_pids:
                related_pids.add(int(row["pid"]))
                changed = True
    ollama = [
        row
        for row in rows
        if any(pattern.search(str(row["command"])) for pattern in OLLAMA_PATTERNS)
    ]
    related_pids.update(int(row["pid"]) for row in ollama)
    related = [row for row in rows if int(row["pid"]) in related_pids]

    # A uv/launcher parent and its Python child are one worker.  Count only
    # independent roots among processes matching the same role, so a normal
    # exec/spawn chain cannot be misreported as duplicate workers.
    role_roots: dict[str, list[dict[str, Any]]] = {}
    for role, items in roles.items():
        matched_pids = {int(item["pid"]) for item in items}
        role_roots[role] = [
            item for item in items if int(item["ppid"]) not in matched_pids
        ]
    role_counts = {role: len(items) for role, items in role_roots.items()}
    duplicates = {role: max(0, count - 1) for role, count in role_counts.items()}
    compact = lambda row: {
        "pid": row["pid"],
        "ppid": row["ppid"],
        "rss_bytes": row["rss_bytes"],
        "etime": row["etime"],
        "command": str(row["command"])[:500],
    }
    return {
        "available": True,
        "role_counts": role_counts,
        "role_process_counts": {role: len(items) for role, items in roles.items()},
        "role_worker_pids": {
            role: [int(item["pid"]) for item in items]
            for role, items in role_roots.items()
        },
        "duplicates": duplicates,
        "related_rss_bytes": sum(int(row["rss_bytes"]) for row in related),
        "related_rss_max_process_bytes": max(
            (int(row["rss_bytes"]) for row in related), default=0
        ),
        "related": [compact(row) for row in related],
        "ollama_processes": [compact(row) for row in ollama],
    }


def http_get(url: str, *, parse_json: bool, timeout: float = 20.0) -> dict[str, Any]:
    started = time.monotonic()
    request = urllib.request.Request(
        url,
        method="GET",
        headers={"Accept": "application/json" if parse_json else "text/html"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
            status = int(response.status)
            content_type = response.headers.get("Content-Type")
    except urllib.error.HTTPError as exc:
        return {
            "ok": False,
            "status": int(exc.code),
            "error": f"HTTPError: {exc}",
            "elapsed_ms": round((time.monotonic() - started) * 1000, 1),
        }
    except (OSError, TimeoutError) as exc:
        return {
            "ok": False,
            "status": None,
            "error": f"{exc.__class__.__name__}: {exc}",
            "elapsed_ms": round((time.monotonic() - started) * 1000, 1),
        }
    payload: Any = None
    error: str | None = None
    if parse_json:
        try:
            payload = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            error = f"{exc.__class__.__name__}: {exc}"
    return {
        "ok": status == 200 and error is None,
        "status": status,
        "bytes": len(raw),
        "content_type": content_type,
        "payload": payload,
        **({"error": error} if error is not None else {}),
        "elapsed_ms": round((time.monotonic() - started) * 1000, 1),
    }


def normalize_ollama_models(payload: object) -> tuple[list[dict[str, Any]], str | None]:
    if not isinstance(payload, Mapping):
        return [], "Ollama /api/ps payload is not an object"
    raw_models = payload.get("models")
    if not isinstance(raw_models, list):
        return [], "Ollama /api/ps models is not a list"
    models: list[dict[str, Any]] = []
    for index, item in enumerate(raw_models):
        if not isinstance(item, Mapping):
            return [], f"Ollama /api/ps models[{index}] is not an object"
        name = item.get("name") or item.get("model")
        if not isinstance(name, str) or not name.strip():
            return [], f"Ollama /api/ps models[{index}] has invalid name"
        digest = item.get("digest")
        if not isinstance(digest, str) or not digest:
            return [], f"Ollama /api/ps models[{index}] has invalid digest"
        numeric_fields = {
            "size": item.get("size"),
            "size_vram": item.get("size_vram"),
            "context_length": item.get("context_length"),
        }
        if any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < (1 if field == "context_length" else 0)
            for field, value in numeric_fields.items()
        ):
            return [], f"Ollama /api/ps models[{index}] has invalid numeric fields"
        models.append(
            {
                "name": name,
                "digest": digest,
                "size_bytes": numeric_fields["size"],
                "size_vram_bytes": numeric_fields["size_vram"],
                "context_length": numeric_fields["context_length"],
                "expires_at": item.get("expires_at"),
                "details": item.get("details"),
            }
        )
    models.sort(key=lambda row: str(row["name"]))
    return models, None


def ollama_snapshot(base_url: str) -> dict[str, Any]:
    response = http_get(f"{base_url.rstrip('/')}/api/ps", parse_json=True, timeout=5)
    payload = response.get("payload")
    if not response.get("ok"):
        return {
            "available": False,
            "status": response.get("status"),
            "error": response.get("error") or "invalid Ollama /api/ps response",
            "elapsed_ms": response.get("elapsed_ms"),
            "models": [],
        }
    models, schema_error = normalize_ollama_models(payload)
    if schema_error is not None:
        return {
            "available": False,
            "status": response.get("status"),
            "error": schema_error,
            "elapsed_ms": response.get("elapsed_ms"),
            "models": [],
        }
    return {
        "available": True,
        "status": response.get("status"),
        "elapsed_ms": response.get("elapsed_ms"),
        "model_count": len(models),
        "model_bytes": sum(
            int(item["size_bytes"] or item["size_vram_bytes"]) for item in models
        ),
        "models": models,
    }


def snapshot_projection(payload: object) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        return {"valid": False, "error": "snapshot is not an object"}
    status = payload.get("status")
    status = status if isinstance(status, Mapping) else {}
    batch = status.get("batch") if isinstance(status.get("batch"), Mapping) else {}
    llm = status.get("llm") if isinstance(status.get("llm"), Mapping) else {}
    local = (
        status.get("local_consensus")
        if isinstance(status.get("local_consensus"), Mapping)
        else {}
    )
    repair = (
        status.get("frontier_repair")
        if isinstance(status.get("frontier_repair"), Mapping)
        else {}
    )
    frontier = (
        status.get("frontier_review")
        if isinstance(status.get("frontier_review"), Mapping)
        else {}
    )
    operational = (
        status.get("operational_deferred")
        if isinstance(status.get("operational_deferred"), Mapping)
        else {}
    )
    semantic = (
        status.get("semantic_deferred")
        if isinstance(status.get("semantic_deferred"), Mapping)
        else {}
    )
    runtime = payload.get("runtime") if isinstance(payload.get("runtime"), Mapping) else {}
    return {
        "valid": bool(status),
        "runtime": {
            key: runtime.get(key)
            for key in ("commit_id", "expected_commit", "drift", "module_path")
        },
        "state": status.get("state"),
        "stage": status.get("stage"),
        "pending": status.get("pending"),
        "raw_outstanding": status.get("raw_outstanding"),
        "current_job_id": status.get("current_job_id"),
        "current_raw": status.get("current_raw"),
        "batch_active": batch.get("active"),
        "llm_active": llm.get("active"),
        "local_consensus_active": local.get("active"),
        "frontier_repair_active": repair.get("active"),
        "frontier_review_active": frontier.get("active"),
        "semantic_deferred": semantic.get("count"),
        "operational_deferred": operational.get("count"),
    }


def save_history_projection(payload: object) -> dict[str, Any]:
    """Retain the exact pending invariants without copying the full history."""

    if not isinstance(payload, Mapping):
        return {"valid": False, "error": "response is not an object"}
    history = payload.get("save_history")
    if not isinstance(history, Mapping):
        return {"valid": False, "error": "save_history is not an object"}
    totals = history.get("totals")
    days = history.get("days")
    if not isinstance(totals, Mapping) or not isinstance(days, list):
        return {
            "valid": False,
            "error": "save_history totals/days shape is invalid",
        }
    pending_bytes = totals.get("pending_bytes")
    pending_segments: list[dict[str, Any]] = []
    raw_segment_count = 0
    invalid_day_count = 0
    invalid_segment_count = 0
    for day in days:
        if not isinstance(day, Mapping):
            invalid_day_count += 1
            continue
        segments = day.get("raw_segments")
        if not isinstance(segments, list):
            invalid_day_count += 1
            continue
        for segment in segments:
            if not isinstance(segment, Mapping):
                invalid_segment_count += 1
                continue
            raw_segment_count += 1
            if segment.get("status") == "pending":
                pending_segments.append(
                    {
                        "date": day.get("date"),
                        "name": segment.get("name"),
                        "bytes": segment.get("bytes"),
                    }
                )
    pending_bytes_is_int = isinstance(pending_bytes, int) and not isinstance(
        pending_bytes, bool
    )
    return {
        "valid": bool(
            pending_bytes_is_int
            and invalid_day_count == 0
            and invalid_segment_count == 0
        ),
        "pending_bytes": pending_bytes,
        "day_count": len(days),
        "raw_segment_count": raw_segment_count,
        "pending_segment_count": len(pending_segments),
        "pending_segment_samples": pending_segments[:20],
        "invalid_day_count": invalid_day_count,
        "invalid_segment_count": invalid_segment_count,
    }


def hardening_projection(payload: object) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        return {"valid": False, "error": "health response is not an object"}
    health = payload.get("health")
    if not isinstance(health, Mapping):
        return {"valid": False, "error": "health payload is not an object"}
    hardening = health.get("autonomy_hardening")
    read_back = health.get("read_back")
    if not isinstance(hardening, Mapping) or not isinstance(read_back, Mapping):
        return {"valid": False, "error": "hardening health fields are missing"}
    deadman = hardening.get("deadman")
    quality = hardening.get("quality")
    artifacts = hardening.get("decision_artifacts")
    holds = hardening.get("managed_holds")
    provisional = hardening.get("provisional_recall")
    ledger = read_back.get("derived_view_integrity")
    values = (deadman, quality, artifacts, holds, provisional, ledger)
    if not all(isinstance(value, Mapping) for value in values):
        return {"valid": False, "error": "hardening projection shape is invalid"}
    main_heartbeat = deadman.get("main")
    observer_heartbeat = deadman.get("observer")
    probe = quality.get("probe")
    return {
        "valid": True,
        "decision_artifacts": artifacts.get("count"),
        "replay_definition": artifacts.get("replay_definition"),
        "deadman_main": (
            main_heartbeat.get("status")
            if isinstance(main_heartbeat, Mapping)
            else None
        ),
        "deadman_observer": (
            observer_heartbeat.get("status")
            if isinstance(observer_heartbeat, Mapping)
            else None
        ),
        "quality_frozen": quality.get("frozen"),
        "quality_probe": probe.get("status") if isinstance(probe, Mapping) else None,
        "managed_holds": holds.get("total"),
        "provisional_entries": provisional.get("entries"),
        "provisional_mutation_allowed": provisional.get(
            "mutation_evidence_allowed"
        ),
        "ledger_integrity": ledger.get("status"),
        "frontier_semantic_audit_allowed": hardening.get(
            "frontier_semantic_audit_allowed"
        ),
    }


def idle_violations(projection: Mapping[str, Any]) -> list[str]:
    violations: list[str] = []
    if not projection.get("valid"):
        return ["snapshot_invalid"]
    if projection.get("state") != "idle":
        violations.append("status_not_idle")
    for key in (
        "batch_active",
        "llm_active",
        "local_consensus_active",
        "frontier_repair_active",
        "frontier_review_active",
    ):
        if projection.get(key) is True:
            violations.append(key)
    for key in ("current_job_id", "current_raw"):
        if projection.get(key):
            violations.append(key)
    pending = projection.get("pending")
    raw_outstanding = projection.get("raw_outstanding")
    semantic_deferred = projection.get("semantic_deferred")
    operational_deferred = projection.get("operational_deferred")
    pending_is_int = isinstance(pending, int) and not isinstance(pending, bool)
    raw_outstanding_is_int = isinstance(raw_outstanding, int) and not isinstance(
        raw_outstanding, bool
    )
    semantic_deferred_is_int = isinstance(
        semantic_deferred, int
    ) and not isinstance(semantic_deferred, bool)
    operational_deferred_is_int = isinstance(
        operational_deferred, int
    ) and not isinstance(operational_deferred, bool)
    if not pending_is_int:
        violations.append("pending_not_int")
    elif pending != 0:
        violations.append("pending_not_zero")
    if not raw_outstanding_is_int:
        violations.append("raw_outstanding_not_int")
    if not semantic_deferred_is_int:
        violations.append("semantic_deferred_not_int")
    if not operational_deferred_is_int:
        violations.append("operational_deferred_not_int")
    elif operational_deferred != 0:
        violations.append("operational_deferred_not_zero")
    if (
        raw_outstanding_is_int
        and semantic_deferred_is_int
        and raw_outstanding != semantic_deferred
    ):
        violations.append("raw_outstanding_semantic_deferred_mismatch")
    return violations


def dashboard_snapshot(base_url: str, *, all_endpoints: bool) -> dict[str, Any]:
    endpoints = DASHBOARD_ENDPOINTS if all_endpoints else ("/api/snapshot",)
    results: dict[str, Any] = {}
    snapshot_payload: Any = None
    save_history_payload: Any = None
    health_payload: Any = None
    for endpoint in endpoints:
        response = http_get(
            f"{base_url.rstrip('/')}{endpoint}",
            parse_json=endpoint.startswith("/api/"),
        )
        if endpoint == "/api/snapshot":
            snapshot_payload = response.get("payload")
            # The aggregate snapshot already contains both components.  Light
            # preflight/final-idle probes intentionally fetch only this one
            # endpoint, so project health and save history from the same
            # response instead of reporting them as missing.
            save_history_payload = snapshot_payload
            health_payload = snapshot_payload
        elif endpoint == "/api/save-history":
            save_history_payload = response.get("payload")
        elif endpoint == "/api/health":
            health_payload = response.get("payload")
        results[endpoint] = {
            key: value
            for key, value in response.items()
            if key not in {"payload"}
        }
    projection = snapshot_projection(snapshot_payload)
    return {
        "endpoints": results,
        "snapshot": projection,
        "save_history": save_history_projection(save_history_payload),
        "hardening": hardening_projection(health_payload),
        "idle_violations": idle_violations(projection),
    }


@dataclasses.dataclass
class FileDeltaTracker:
    name: str
    path: Path
    initial_size: int = 0
    initial_mtime_ns: int | None = None
    initial_device: int | None = None
    initial_inode: int | None = None
    offset: int = 0
    current_size: int = 0
    appended_bytes: int = 0
    appended_lines: int = 0
    resets: int = 0
    errors: list[str] = dataclasses.field(default_factory=list)
    replace_record_key: str | None = None
    seen_record_ids: set[str] = dataclasses.field(default_factory=set)
    baseline_present: bool = False
    present: bool = False
    missing_after_baseline: bool = False
    _hasher: Any = dataclasses.field(default_factory=hashlib.sha256, repr=False)

    @classmethod
    def start(
        cls,
        name: str,
        path: Path,
        *,
        replace_record_key: str | None = None,
    ) -> "FileDeltaTracker":
        tracker = cls(name=name, path=path, replace_record_key=replace_record_key)
        try:
            stat = path.stat()
        except FileNotFoundError:
            return tracker
        except OSError as exc:
            tracker.errors.append(f"baseline: {exc}")
            return tracker
        tracker.initial_size = stat.st_size
        tracker.initial_mtime_ns = stat.st_mtime_ns
        tracker.initial_device = stat.st_dev
        tracker.initial_inode = stat.st_ino
        tracker.offset = stat.st_size
        tracker.current_size = stat.st_size
        tracker.baseline_present = True
        tracker.present = True
        if replace_record_key is not None:
            try:
                _new_lines, baseline_ids, valid, error = keyed_jsonl_delta(
                    path.read_bytes(),
                    None,
                    key=replace_record_key,
                )
                tracker.seen_record_ids.update(baseline_ids)
                if not valid:
                    tracker.errors.append(f"baseline: {error}")
            except OSError as exc:
                tracker.errors.append(f"baseline read: {exc}")
        return tracker

    def poll(self) -> dict[str, Any]:
        try:
            stat = self.path.stat()
        except FileNotFoundError:
            self.current_size = 0
            self.present = False
            if self.baseline_present and not self.missing_after_baseline:
                self.missing_after_baseline = True
                self.resets += 1
                self.errors.append("file missing after baseline")
            return self.projection(status="absent")
        except OSError as exc:
            self.present = False
            self.errors.append(f"poll: {exc}")
            return self.projection(status="error")
        self.present = True
        if self.replace_record_key is not None:
            try:
                payload = self.path.read_bytes()
            except OSError as exc:
                self.errors.append(f"read: {exc}")
                return self.projection(status="error")
            new_lines, current_ids, valid, error = keyed_jsonl_delta(
                payload,
                self.seen_record_ids,
                key=self.replace_record_key,
            )
            if not valid:
                self.resets += 1
                self.errors.append(f"bounded JSONL invalid after replacement: {error}")
            else:
                for line in new_lines:
                    self._hasher.update(line)
                    self.appended_bytes += len(line)
                    self.appended_lines += 1
                self.seen_record_ids.update(current_ids)
            self.initial_device = stat.st_dev
            self.initial_inode = stat.st_ino
            self.offset = stat.st_size
            self.current_size = stat.st_size
            return self.projection(status="present")
        if self.initial_inode is None:
            self.initial_device = stat.st_dev
            self.initial_inode = stat.st_ino
        identity_changed = (
            self.initial_inode is not None
            and (stat.st_dev != self.initial_device or stat.st_ino != self.initial_inode)
        )
        truncated = stat.st_size < self.offset
        if identity_changed or truncated:
            self.resets += 1
            self.offset = 0
            self.initial_device = stat.st_dev
            self.initial_inode = stat.st_ino
        try:
            with self.path.open("rb") as handle:
                handle.seek(self.offset)
                while True:
                    chunk = handle.read(1024 * 1024)
                    if not chunk:
                        break
                    self._hasher.update(chunk)
                    self.appended_bytes += len(chunk)
                    self.appended_lines += chunk.count(b"\n")
                self.offset = handle.tell()
        except OSError as exc:
            self.errors.append(f"read: {exc}")
        self.current_size = stat.st_size
        return self.projection(status="present")

    def projection(self, *, status: str) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "status": status,
            "initial_size": self.initial_size,
            "current_size": self.current_size,
            "appended_bytes": self.appended_bytes,
            "appended_lines": self.appended_lines,
            "appended_sha256": self._hasher.hexdigest(),
            "resets": self.resets,
            "errors": list(self.errors),
            "baseline_present": self.baseline_present,
            "present": self.present,
            "missing_after_baseline": self.missing_after_baseline,
        }


def keyed_jsonl_delta(
    payload: bytes,
    seen_record_ids: set[str] | None,
    *,
    key: str,
) -> tuple[list[bytes], set[str], bool, str | None]:
    """Return rows with new stable IDs from an atomically replaced JSONL.

    The watchdog intentionally rewrites its bounded history with ``os.replace``.
    Treating an inode change as log rotation therefore produces a false burn-in
    failure.  Its ``ts`` field is a per-run stable identity, so tracking IDs is
    exact across repeated payloads and bounded-window rollover.  Missing,
    duplicate, or non-monotonic IDs fail closed instead of guessing an offset.
    """
    lines = [line for line in payload.splitlines(keepends=True) if line.strip()]
    ids: list[str] = []
    for index, line in enumerate(lines):
        try:
            row = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            return [], set(ids), False, f"row {index} invalid JSON: {exc}"
        if not isinstance(row, Mapping):
            return [], set(ids), False, f"row {index} is not an object"
        value = row.get(key)
        if not isinstance(value, str) or not value.strip():
            return [], set(ids), False, f"row {index} missing string {key}"
        ids.append(value)
    current_ids = set(ids)
    if len(current_ids) != len(ids):
        return [], current_ids, False, f"duplicate {key}"
    if ids != sorted(ids):
        return [], current_ids, False, f"non-monotonic {key}"
    if seen_record_ids is None:
        return lines, current_ids, True, None
    if seen_record_ids and not lines:
        return [], current_ids, False, "history became empty"
    new_pairs = [
        (line, record_id)
        for line, record_id in zip(lines, ids)
        if record_id not in seen_record_ids
    ]
    latest_seen = max(seen_record_ids) if seen_record_ids else None
    if latest_seen is not None and any(record_id <= latest_seen for _, record_id in new_pairs):
        return [], current_ids, False, f"new {key} did not advance"
    return [line for line, _record_id in new_pairs], current_ids, True, None


@dataclasses.dataclass
class ModelStabilityTracker:
    states: dict[str, list[bool]] = dataclasses.field(default_factory=dict)
    contexts: dict[str, Any] = dataclasses.field(default_factory=dict)
    identities: dict[str, tuple[Any, Any, Any]] = dataclasses.field(default_factory=dict)
    context_changes: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    identity_changes: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    transitions: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    unavailable_count: int = 0
    initialized: bool = False

    def update(self, snapshot: Mapping[str, Any], *, elapsed_seconds: float) -> None:
        if not snapshot.get("available"):
            self.unavailable_count += 1
            return
        loaded = {
            str(row.get("name")): row
            for row in snapshot.get("models", [])
            if isinstance(row, Mapping) and row.get("name")
        }
        if not self.initialized:
            for name, row in loaded.items():
                self.states[name] = [True]
                self.contexts[name] = row.get("context_length")
                self.identities[name] = (
                    row.get("digest"),
                    row.get("size_bytes"),
                    row.get("size_vram_bytes"),
                )
            self.initialized = True
            return
        names = set(self.states) | set(loaded)
        for name in sorted(names):
            present = name in loaded
            history = self.states.setdefault(name, [False] if present else [False])
            previous = history[-1]
            if previous != present:
                history.append(present)
                self.transitions.append(
                    {
                        "elapsed_seconds": round(elapsed_seconds, 3),
                        "model": name,
                        "loaded": present,
                    }
                )
            if not present:
                continue
            context = loaded[name].get("context_length")
            if name in self.contexts and self.contexts[name] != context:
                self.context_changes.append(
                    {
                        "elapsed_seconds": round(elapsed_seconds, 3),
                        "model": name,
                        "before": self.contexts[name],
                        "after": context,
                    }
                )
            self.contexts[name] = context
            identity = (
                loaded[name].get("digest"),
                loaded[name].get("size_bytes"),
                loaded[name].get("size_vram_bytes"),
            )
            if name in self.identities and self.identities[name] != identity:
                self.identity_changes.append(
                    {
                        "elapsed_seconds": round(elapsed_seconds, 3),
                        "model": name,
                        "before": self.identities[name],
                        "after": identity,
                    }
                )
            self.identities[name] = identity

    def projection(self) -> dict[str, Any]:
        reload_cycles = 0
        for history in self.states.values():
            for index in range(2, len(history)):
                if (
                    history[index - 2] is True
                    and history[index - 1] is False
                    and history[index] is True
                ):
                    reload_cycles += 1
        return {
            "states": self.states,
            "transitions": self.transitions,
            "transition_count": len(self.transitions),
            "flap_count": reload_cycles,
            "contexts": self.contexts,
            "context_changes": self.context_changes,
            "context_change_count": len(self.context_changes),
            "identities": self.identities,
            "identity_changes": self.identity_changes,
            "identity_change_count": len(self.identity_changes),
            "unavailable_count": self.unavailable_count,
        }


@dataclasses.dataclass
class Aggregate:
    baseline_swap_bytes: int | None = None
    baseline_swap_available: bool = False
    max_swap_bytes: int | None = None
    swap_unavailable_count: int = 0
    light_sample_count: int = 0
    max_related_rss_bytes: int = 0
    max_related_process_rss_bytes: int = 0
    max_ollama_model_bytes: int = 0
    max_duplicates: dict[str, int] = dataclasses.field(default_factory=dict)
    min_role_counts: dict[str, int] = dataclasses.field(default_factory=dict)
    max_role_counts: dict[str, int] = dataclasses.field(default_factory=dict)
    process_unavailable_count: int = 0
    endpoint_failures: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    launchctl_run_regressions: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    services_unloaded_observations: list[dict[str, Any]] = dataclasses.field(
        default_factory=list
    )
    previous_runs: dict[str, int | None] = dataclasses.field(default_factory=dict)
    model_tracker: ModelStabilityTracker = dataclasses.field(
        default_factory=ModelStabilityTracker
    )

    def update_light(self, sample: Mapping[str, Any], *, elapsed_seconds: float) -> None:
        swap = sample.get("swap") if isinstance(sample.get("swap"), Mapping) else {}
        used = swap.get("used_bytes")
        swap_available = swap.get("available") is True and isinstance(used, int)
        if self.light_sample_count == 0:
            self.baseline_swap_available = swap_available
            if swap_available:
                self.baseline_swap_bytes = used
        if swap_available:
            self.max_swap_bytes = max(self.max_swap_bytes or used, used)
        else:
            self.swap_unavailable_count += 1
        self.light_sample_count += 1
        processes = (
            sample.get("processes")
            if isinstance(sample.get("processes"), Mapping)
            else {}
        )
        if not processes.get("available"):
            self.process_unavailable_count += 1
        self.max_related_rss_bytes = max(
            self.max_related_rss_bytes, int(processes.get("related_rss_bytes") or 0)
        )
        self.max_related_process_rss_bytes = max(
            self.max_related_process_rss_bytes,
            int(processes.get("related_rss_max_process_bytes") or 0),
        )
        duplicates = processes.get("duplicates")
        if isinstance(duplicates, Mapping):
            for role, count in duplicates.items():
                self.max_duplicates[str(role)] = max(
                    self.max_duplicates.get(str(role), 0), int(count or 0)
                )
        role_counts = processes.get("role_counts")
        if isinstance(role_counts, Mapping):
            for role, count in role_counts.items():
                role = str(role)
                value = int(count or 0)
                self.min_role_counts[role] = min(
                    self.min_role_counts.get(role, value), value
                )
                self.max_role_counts[role] = max(
                    self.max_role_counts.get(role, value), value
                )
        ollama = sample.get("ollama") if isinstance(sample.get("ollama"), Mapping) else {}
        self.max_ollama_model_bytes = max(
            self.max_ollama_model_bytes, int(ollama.get("model_bytes") or 0)
        )
        self.model_tracker.update(ollama, elapsed_seconds=elapsed_seconds)

    def update_full(self, sample: Mapping[str, Any], *, elapsed_seconds: float) -> None:
        launchd = sample.get("launchd") if isinstance(sample.get("launchd"), Mapping) else {}
        for role in SERVICE_LABELS:
            row = launchd.get(role)
            if not isinstance(row, Mapping) or row.get("loaded") is not True:
                self.services_unloaded_observations.append(
                    {
                        "elapsed_seconds": round(elapsed_seconds, 3),
                        "role": role,
                        "loaded": row.get("loaded") if isinstance(row, Mapping) else None,
                        "state": row.get("state") if isinstance(row, Mapping) else None,
                    }
                )
        for role, row in launchd.items():
            if not isinstance(row, Mapping):
                continue
            runs = row.get("runs")
            previous = self.previous_runs.get(str(role))
            if isinstance(runs, int) and isinstance(previous, int) and runs < previous:
                self.launchctl_run_regressions.append(
                    {
                        "elapsed_seconds": round(elapsed_seconds, 3),
                        "role": role,
                        "before": previous,
                        "after": runs,
                    }
                )
            self.previous_runs[str(role)] = runs if isinstance(runs, int) else None
        dashboard = (
            sample.get("dashboard")
            if isinstance(sample.get("dashboard"), Mapping)
            else {}
        )
        endpoints = dashboard.get("endpoints")
        if isinstance(endpoints, Mapping):
            for endpoint, row in endpoints.items():
                if not isinstance(row, Mapping) or not row.get("ok"):
                    self.endpoint_failures.append(
                        {
                            "elapsed_seconds": round(elapsed_seconds, 3),
                            "endpoint": endpoint,
                            "status": row.get("status") if isinstance(row, Mapping) else None,
                            "error": row.get("error") if isinstance(row, Mapping) else "invalid",
                        }
                    )


class EvidenceWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.handle = path.open("x", encoding="utf-8")
        fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    def write(self, row: Mapping[str, Any]) -> None:
        self.handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        self.handle.flush()
        os.fsync(self.handle.fileno())

    def close(self) -> None:
        self.handle.close()


def light_sample(ollama_url: str) -> dict[str, Any]:
    return {
        "timestamp": utc_now(),
        "swap": swap_snapshot(),
        "processes": process_snapshot(),
        "ollama": ollama_snapshot(ollama_url),
    }


def full_sample(
    dashboard_url: str,
    ollama_url: str,
    trackers: Mapping[str, FileDeltaTracker],
    *,
    include_light: bool,
) -> dict[str, Any]:
    row = light_sample(ollama_url) if include_light else {"timestamp": utc_now()}
    row.update(
        {
            "launchd": launchctl_snapshot(),
            "files": {name: tracker.poll() for name, tracker in trackers.items()},
            "frontier": frontier_fingerprint(),
            "dashboard": dashboard_snapshot(dashboard_url, all_endpoints=True),
        }
    )
    return row


def all_services_loaded(launchd: Mapping[str, Any]) -> bool:
    return all(
        isinstance(launchd.get(role), Mapping) and launchd[role].get("loaded") is True
        for role in SERVICE_LABELS
    )


def runtime_is_expected(projection: Mapping[str, Any], expected_commit: str) -> bool:
    runtime = projection.get("runtime")
    if not isinstance(runtime, Mapping):
        return False
    return bool(
        runtime.get("commit_id") == expected_commit
        and runtime.get("expected_commit") == expected_commit
        and runtime.get("drift") is False
    )


def make_assertions(
    baseline: Mapping[str, Any],
    final: Mapping[str, Any],
    aggregate: Aggregate,
    trackers: Mapping[str, FileDeltaTracker],
    *,
    expected_commit: str,
) -> dict[str, bool]:
    baseline_launchd = (
        baseline.get("launchd")
        if isinstance(baseline.get("launchd"), Mapping)
        else {}
    )
    final_launchd = (
        final.get("launchd")
        if isinstance(final.get("launchd"), Mapping)
        else {}
    )

    def run_delta(role: str) -> int | None:
        before = baseline_launchd.get(role)
        after = final_launchd.get(role)
        if not isinstance(before, Mapping) or not isinstance(after, Mapping):
            return None
        before_runs = before.get("runs")
        after_runs = after.get("runs")
        if not isinstance(before_runs, int) or not isinstance(after_runs, int):
            return None
        return after_runs - before_runs

    baseline_frontier = baseline.get("frontier")
    final_frontier = final.get("frontier")
    frontier_events = (
        final_frontier.get("frontier_events")
        if isinstance(final_frontier, Mapping)
        and isinstance(final_frontier.get("frontier_events"), Mapping)
        else {}
    )
    final_dashboard = final.get("dashboard") if isinstance(final.get("dashboard"), Mapping) else {}
    final_endpoints = (
        final_dashboard.get("endpoints")
        if isinstance(final_dashboard.get("endpoints"), Mapping)
        else {}
    )
    final_projection = (
        final_dashboard.get("snapshot")
        if isinstance(final_dashboard.get("snapshot"), Mapping)
        else {}
    )
    final_save_history = (
        final_dashboard.get("save_history")
        if isinstance(final_dashboard.get("save_history"), Mapping)
        else {}
    )
    final_hardening = (
        final_dashboard.get("hardening")
        if isinstance(final_dashboard.get("hardening"), Mapping)
        else {}
    )
    raw_outstanding = final_projection.get("raw_outstanding")
    semantic_deferred = final_projection.get("semantic_deferred")
    operational_deferred = final_projection.get("operational_deferred")
    raw_outstanding_is_int = isinstance(raw_outstanding, int) and not isinstance(
        raw_outstanding, bool
    )
    semantic_deferred_is_int = isinstance(
        semantic_deferred, int
    ) and not isinstance(semantic_deferred, bool)
    operational_deferred_is_int = isinstance(
        operational_deferred, int
    ) and not isinstance(operational_deferred, bool)
    save_pending_bytes = final_save_history.get("pending_bytes")
    save_pending_bytes_is_int = isinstance(
        save_pending_bytes, int
    ) and not isinstance(save_pending_bytes, bool)
    model = aggregate.model_tracker.projection()
    converge_growth = (
        trackers["converge_out"].appended_bytes
        + trackers["converge_err"].appended_bytes
    )
    return {
        "services_loaded_at_baseline": all_services_loaded(baseline_launchd),
        "services_loaded_at_end": all_services_loaded(final_launchd),
        "services_loaded_always": not aggregate.services_unloaded_observations,
        "watchdog_natural_runs_at_least_2": (run_delta("watchdog") or 0) >= 2,
        "watchdog_history_rows_at_least_2": trackers[
            "watchdog_history"
        ].appended_lines
        >= 2,
        "converge_natural_runs_at_least_1": (run_delta("converge") or 0) >= 1,
        "converge_log_grew": converge_growth > 0,
        "launchctl_runs_never_regressed": not aggregate.launchctl_run_regressions,
        "tracked_files_present_at_baseline": all(
            tracker.baseline_present for tracker in trackers.values()
        ),
        "tracked_files_present_at_end": all(tracker.present for tracker in trackers.values()),
        "tracked_files_not_rotated": all(tracker.resets == 0 for tracker in trackers.values()),
        "tracked_file_errors_empty": all(not tracker.errors for tracker in trackers.values()),
        "frontier_baseline_valid": frontier_baseline_error(baseline_frontier) is None,
        "frontier_exactly_unchanged": baseline_frontier == final_frontier,
        "frontier_start_count_zero": frontier_events.get("start_count") == 0,
        "swap_available": aggregate.baseline_swap_available,
        "swap_always_available": aggregate.swap_unavailable_count == 0,
        "swap_did_not_increase": (
            aggregate.baseline_swap_bytes is not None
            and aggregate.max_swap_bytes is not None
            and aggregate.max_swap_bytes <= aggregate.baseline_swap_bytes
        ),
        "related_rss_at_most_80_gib": aggregate.max_related_rss_bytes
        <= MAX_RELATED_RSS_BYTES,
        "ollama_model_bytes_at_most_80_gib": aggregate.max_ollama_model_bytes
        <= MAX_RELATED_RSS_BYTES,
        "dashboard_duplicate_zero": aggregate.max_duplicates.get("dashboard", 0) == 0,
        "ingest_duplicate_zero": aggregate.max_duplicates.get("ingest", 0) == 0,
        "process_sampling_always_available": aggregate.process_unavailable_count == 0,
        "dashboard_worker_exactly_one": (
            aggregate.min_role_counts.get("dashboard") == 1
            and aggregate.max_role_counts.get("dashboard") == 1
        ),
        "ingest_worker_exactly_one": (
            aggregate.min_role_counts.get("ingest") == 1
            and aggregate.max_role_counts.get("ingest") == 1
        ),
        "all_role_duplicates_zero": all(
            count == 0 for count in aggregate.max_duplicates.values()
        ),
        "ollama_api_always_available": model["unavailable_count"] == 0,
        "ollama_runner_flap_zero": model["flap_count"] == 0,
        "ollama_context_change_zero": model["context_change_count"] == 0,
        "ollama_identity_change_zero": model["identity_change_count"] == 0,
        "dashboard_endpoints_always_200": not aggregate.endpoint_failures,
        "final_endpoints_200": all(
            isinstance(final_endpoints.get(endpoint), Mapping)
            and final_endpoints[endpoint].get("ok") is True
            and final_endpoints[endpoint].get("status") == 200
            for endpoint in DASHBOARD_ENDPOINTS
        ),
        "final_raw_accounting_types_valid": bool(
            raw_outstanding_is_int
            and semantic_deferred_is_int
            and operational_deferred_is_int
        ),
        "final_raw_outstanding_equals_semantic_deferred": bool(
            raw_outstanding_is_int
            and semantic_deferred_is_int
            and raw_outstanding == semantic_deferred
        ),
        "final_save_history_valid": final_save_history.get("valid") is True,
        "final_save_history_pending_bytes_zero": bool(
            save_pending_bytes_is_int and save_pending_bytes == 0
        ),
        "final_save_history_no_pending_segments": (
            final_save_history.get("pending_segment_count") == 0
        ),
        "final_dashboard_idle": not idle_violations(final_projection),
        "dashboard_runtime_is_expected_commit": runtime_is_expected(
            final_projection, expected_commit
        ),
        "hardening_projection_valid": final_hardening.get("valid") is True,
        "decision_artifact_replay_enabled": (
            final_hardening.get("replay_definition")
            == "sealed_execution_fingerprint"
        ),
        "deadman_cross_check_healthy": (
            final_hardening.get("deadman_main") == "ok"
            and final_hardening.get("deadman_observer") == "ok"
        ),
        "quality_containment_clear": (
            final_hardening.get("quality_frozen") == 0
            and final_hardening.get("quality_probe") == "ok"
        ),
        "provisional_mutation_disabled": (
            final_hardening.get("provisional_mutation_allowed") is False
        ),
        "read_back_ledger_integrity_ok": (
            final_hardening.get("ledger_integrity") == "ok"
        ),
        "frontier_semantic_audit_disabled": (
            final_hardening.get("frontier_semantic_audit_allowed") is False
        ),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration-seconds", type=int, default=DEFAULT_DURATION_SECONDS)
    parser.add_argument("--sample-seconds", type=int, default=DEFAULT_SAMPLE_SECONDS)
    parser.add_argument("--probe-seconds", type=int, default=DEFAULT_PROBE_SECONDS)
    parser.add_argument(
        "--preflight-wait-seconds", type=int, default=DEFAULT_PREFLIGHT_WAIT_SECONDS
    )
    parser.add_argument(
        "--final-idle-wait-seconds", type=int, default=DEFAULT_FINAL_IDLE_WAIT_SECONDS
    )
    parser.add_argument("--dashboard-url", default=DEFAULT_DASHBOARD_URL)
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--expected-commit", default=EXPECTED_COMMIT)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run pure helper smoke tests; do not query services or write evidence",
    )
    args = parser.parse_args(argv)
    if not args.self_test:
        for name in ("duration_seconds", "sample_seconds", "probe_seconds"):
            if getattr(args, name) <= 0:
                parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def self_test() -> None:
    launchd = parse_launchctl_print(
        "state = not running\n\truns = 12\n\tlast exit code = 0\n"
    )
    assert launchd == {
        "state": "not running",
        "runs": 12,
        "pid": None,
        "last_exit_code": 0,
    }
    swap = parse_swapusage("vm.swapusage: total = 2048.00M used = 1.50G free = 512.00M")
    assert swap["available"] is True
    assert swap["used_bytes"] == int(1.5 * 1024**3)
    idle = {
        "valid": True,
        "state": "idle",
        "pending": 0,
        "raw_outstanding": 3,
        "semantic_deferred": 3,
        "operational_deferred": 0,
    }
    assert idle_violations(idle) == []
    assert "status_not_idle" in idle_violations({**idle, "state": "running"})
    assert "raw_outstanding_not_int" in idle_violations(
        {**idle, "raw_outstanding": None}
    )
    assert "raw_outstanding_semantic_deferred_mismatch" in idle_violations(
        {**idle, "raw_outstanding": 4}
    )
    save_history = save_history_projection(
        {
            "save_history": {
                "totals": {"pending_bytes": 0},
                "days": [
                    {
                        "date": "2026-07-15",
                        "raw_segments": [
                            {"name": "raw.md", "bytes": 10, "status": "deferred"}
                        ],
                    }
                ],
            }
        }
    )
    assert save_history["valid"] is True
    assert save_history["pending_segment_count"] == 0
    pending_history = save_history_projection(
        {
            "save_history": {
                "totals": {"pending_bytes": 10},
                "days": [
                    {
                        "date": "2026-07-15",
                        "raw_segments": [
                            {"name": "raw.md", "bytes": 10, "status": "pending"}
                        ],
                    }
                ],
            }
        }
    )
    assert pending_history["pending_bytes"] == 10
    assert pending_history["pending_segment_count"] == 1
    hardening = hardening_projection(
        {
            "health": {
                "autonomy_hardening": {
                    "decision_artifacts": {
                        "count": 4,
                        "replay_definition": "sealed_execution_fingerprint",
                    },
                    "deadman": {
                        "main": {"status": "ok"},
                        "observer": {"status": "ok"},
                    },
                    "quality": {"frozen": 0, "probe": {"status": "ok"}},
                    "managed_holds": {"total": 3},
                    "provisional_recall": {
                        "entries": 2,
                        "mutation_evidence_allowed": False,
                    },
                    "frontier_semantic_audit_allowed": False,
                },
                "read_back": {"derived_view_integrity": {"status": "ok"}},
            }
        }
    )
    assert hardening["valid"] is True
    assert hardening["deadman_main"] == "ok"
    assert hardening["deadman_observer"] == "ok"
    assert hardening["quality_frozen"] == 0
    assert hardening["provisional_mutation_allowed"] is False
    assert hardening["ledger_integrity"] == "ok"
    assert hardening["frontier_semantic_audit_allowed"] is False
    tracker = ModelStabilityTracker()
    tracker.update({"available": True, "models": []}, elapsed_seconds=0)
    tracker.update(
        {"available": True, "models": [{"name": "model", "context_length": 32768}]},
        elapsed_seconds=1,
    )
    tracker.update({"available": True, "models": []}, elapsed_seconds=2)
    projection = tracker.projection()
    assert projection["transition_count"] == 2
    assert projection["flap_count"] == 0
    tracker.update(
        {"available": True, "models": [{"name": "model", "context_length": 32768}]},
        elapsed_seconds=3,
    )
    projection = tracker.projection()
    assert projection["transition_count"] == 3
    assert projection["flap_count"] == 1
    first_payload = (
        b'{"ts":"2026-07-15T00:01:00","status":"A"}\n'
        b'{"ts":"2026-07-15T00:02:00","status":"A"}\n'
    )
    first_rows, first_ids, first_valid, first_error = keyed_jsonl_delta(
        first_payload,
        None,
        key="ts",
    )
    assert first_valid is True
    assert first_error is None
    assert len(first_rows) == 2
    second_rows, second_ids, second_valid, second_error = keyed_jsonl_delta(
        first_payload + b'{"ts":"2026-07-15T00:03:00","status":"A"}\n',
        first_ids,
        key="ts",
    )
    assert second_valid is True
    assert second_error is None
    assert second_rows == [b'{"ts":"2026-07-15T00:03:00","status":"A"}\n']
    rolled_payload = (
        b'{"ts":"2026-07-15T00:02:00","status":"A"}\n'
        b'{"ts":"2026-07-15T00:03:00","status":"A"}\n'
        b'{"ts":"2026-07-15T00:04:00","status":"A"}\n'
    )
    rolled_rows, rolled_ids, rolled_valid, rolled_error = keyed_jsonl_delta(
        rolled_payload,
        first_ids | second_ids,
        key="ts",
    )
    assert rolled_valid is True
    assert rolled_error is None
    assert rolled_rows == [b'{"ts":"2026-07-15T00:04:00","status":"A"}\n']
    assert len(rolled_ids) == 3
    duplicate_rows, _duplicate_ids, duplicate_valid, duplicate_error = (
        keyed_jsonl_delta(
            b'{"ts":"2026-07-15T00:04:00"}\n{"ts":"2026-07-15T00:04:00"}\n',
            first_ids | second_ids | rolled_ids,
            key="ts",
        )
    )
    assert duplicate_valid is False
    assert duplicate_rows == []
    assert duplicate_error == "duplicate ts"
    nonmonotonic_rows, _nonmonotonic_ids, nonmonotonic_valid, nonmonotonic_error = (
        keyed_jsonl_delta(
            b'{"ts":"2026-07-15T00:05:00"}\n{"ts":"2026-07-15T00:04:00"}\n',
            set(),
            key="ts",
        )
    )
    assert nonmonotonic_valid is False
    assert nonmonotonic_rows == []
    assert nonmonotonic_error == "non-monotonic ts"
    old_rows, _old_ids, old_valid, old_error = keyed_jsonl_delta(
        b'{"ts":"2026-07-14T23:59:00"}\n',
        first_ids | second_ids | rolled_ids,
        key="ts",
    )
    assert old_valid is False
    assert old_rows == []
    assert old_error == "new ts did not advance"
    emptied_rows, emptied_ids, emptied_valid, emptied_error = keyed_jsonl_delta(
        b"",
        first_ids,
        key="ts",
    )
    assert emptied_valid is False
    assert emptied_rows == []
    assert emptied_ids == set()
    assert emptied_error == "history became empty"
    empty_base_rows, empty_base_ids, empty_base_valid, empty_base_error = (
        keyed_jsonl_delta(
            b"",
            None,
            key="ts",
        )
    )
    assert empty_base_valid is True
    assert empty_base_error is None
    assert empty_base_rows == []
    after_empty_rows, _after_empty_ids, after_empty_valid, after_empty_error = (
        keyed_jsonl_delta(
            b'{"ts":"2026-07-15T00:01:00"}\n',
            empty_base_ids,
            key="ts",
        )
    )
    assert after_empty_valid is True
    assert after_empty_error is None
    assert after_empty_rows == [b'{"ts":"2026-07-15T00:01:00"}\n']
    invalid_rows, _invalid_ids, invalid_valid, invalid_error = keyed_jsonl_delta(
        b"not-json\n",
        set(),
        key="ts",
    )
    assert invalid_valid is False
    assert invalid_rows == []
    assert invalid_error is not None and "invalid JSON" in invalid_error
    missing_rows, _missing_ids, missing_valid, missing_error = keyed_jsonl_delta(
        b'{"status":"alert"}\n',
        set(),
        key="ts",
    )
    assert missing_valid is False
    assert missing_rows == []
    assert missing_error == "row 0 missing string ts"
    missing_tracker = FileDeltaTracker(
        name="missing",
        path=Path(f"/tmp/chronovisor-burn-monitor-missing-{os.getpid()}"),
        initial_inode=1,
        baseline_present=True,
        present=True,
    )
    missing_projection = missing_tracker.poll()
    assert missing_projection["status"] == "absent"
    assert missing_projection["present"] is False
    assert missing_projection["missing_after_baseline"] is True
    assert missing_projection["resets"] == 1
    assert missing_projection["errors"] == ["file missing after baseline"]
    missing_tracker.poll()
    assert missing_tracker.resets == 1
    swap_gap = Aggregate()
    swap_gap.update_light(
        {
            "swap": {"available": False, "used_bytes": None},
            "processes": {"available": True, "role_counts": {}, "duplicates": {}},
            "ollama": {"available": True, "models": []},
        },
        elapsed_seconds=0,
    )
    swap_gap.update_light(
        {
            "swap": {"available": True, "used_bytes": 0},
            "processes": {"available": True, "role_counts": {}, "duplicates": {}},
            "ollama": {"available": True, "models": []},
        },
        elapsed_seconds=1,
    )
    assert swap_gap.baseline_swap_available is False
    assert swap_gap.baseline_swap_bytes is None
    assert swap_gap.swap_unavailable_count == 1
    service_gap = Aggregate()
    loaded_services = {
        role: {"loaded": True, "runs": 1, "state": "not running"}
        for role in SERVICE_LABELS
    }
    service_gap.update_full({"launchd": loaded_services}, elapsed_seconds=0)
    assert service_gap.services_unloaded_observations == []
    unloaded_services = {role: dict(row) for role, row in loaded_services.items()}
    unloaded_services["watchdog"]["loaded"] = False
    service_gap.update_full({"launchd": unloaded_services}, elapsed_seconds=1)
    assert len(service_gap.services_unloaded_observations) == 1
    assert service_gap.services_unloaded_observations[0]["role"] == "watchdog"
    assert normalize_ollama_models({})[1] is not None
    assert normalize_ollama_models({"models": "invalid"})[1] is not None
    assert normalize_ollama_models({"models": [None]})[1] is not None
    assert normalize_ollama_models({"models": [{"name": "model"}]})[1] is not None
    empty_models, empty_models_error = normalize_ollama_models({"models": []})
    assert empty_models == [] and empty_models_error is None
    one_model, one_model_error = normalize_ollama_models(
        {
            "models": [
                {
                    "name": "model:latest",
                    "digest": "abc",
                    "size": 10,
                    "size_vram": 10,
                    "context_length": 8192,
                }
            ]
        }
    )
    assert one_model_error is None and one_model[0]["name"] == "model:latest"
    assert ollama_quiescence_error({"available": True, "model_count": 0}) is None
    assert (
        ollama_quiescence_error({"available": False, "model_count": 0})
        == "ollama_unavailable"
    )
    assert (
        ollama_quiescence_error({"available": True, "model_count": 1})
        == "ollama_not_quiescent"
    )
    print(json.dumps({"status": "ok", "tests": 92}, sort_keys=True))


def ollama_quiescence_error(snapshot: Mapping[str, Any]) -> str | None:
    if snapshot.get("available") is not True:
        return "ollama_unavailable"
    if snapshot.get("model_count") != 0:
        return "ollama_not_quiescent"
    return None


def preflight_ready(args: argparse.Namespace) -> tuple[dict[str, Any], list[str]]:
    launchd = launchctl_snapshot()
    dashboard = dashboard_snapshot(args.dashboard_url, all_endpoints=False)
    ollama = ollama_snapshot(args.ollama_url)
    errors: list[str] = []
    if not all_services_loaded(launchd):
        errors.append("not_all_services_loaded")
    endpoint = dashboard.get("endpoints", {}).get("/api/snapshot", {})
    if not isinstance(endpoint, Mapping) or endpoint.get("ok") is not True:
        errors.append("dashboard_snapshot_unavailable")
    if dashboard.get("idle_violations"):
        errors.append("dashboard_not_idle")
    projection = dashboard.get("snapshot")
    if not isinstance(projection, Mapping) or not runtime_is_expected(
        projection, args.expected_commit
    ):
        errors.append("dashboard_runtime_not_expected_commit")
    ollama_error = ollama_quiescence_error(ollama)
    if ollama_error is not None:
        errors.append(ollama_error)
    return {"launchd": launchd, "dashboard": dashboard, "ollama": ollama}, errors


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    if not args.expected_commit:
        print(
            json.dumps(
                {
                    "status": "configuration_error",
                    "error": "--expected-commit is required for a live burn",
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2

    BURN_ROOT.mkdir(parents=True, exist_ok=True)
    output = args.output or (
        BURN_ROOT
        / f"burn-in-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{os.getpid()}.jsonl"
    )
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = EvidenceWriter(output)
    writer.write(
        {
            "kind": "metadata",
            "timestamp": utc_now(),
            "pid": os.getpid(),
            "duration_seconds": args.duration_seconds,
            "sample_seconds": args.sample_seconds,
            "probe_seconds": args.probe_seconds,
            "dashboard_url": args.dashboard_url,
            "ollama_url": args.ollama_url,
            "expected_commit": args.expected_commit,
            "read_only_service_monitor": True,
            "quiescent_model_baseline_required": True,
        }
    )

    try:
        preflight_deadline = time.monotonic() + args.preflight_wait_seconds
        preflight: dict[str, Any] = {}
        errors: list[str] = []
        while True:
            preflight, errors = preflight_ready(args)
            writer.write(
                {
                    "kind": "preflight",
                    "timestamp": utc_now(),
                    "ready": not errors,
                    "errors": errors,
                    **preflight,
                }
            )
            if not errors:
                break
            if time.monotonic() >= preflight_deadline:
                summary = {
                    "kind": "summary",
                    "timestamp": utc_now(),
                    "status": "preflight_failed",
                    "errors": errors,
                    "evidence_path": str(output),
                }
                writer.write(summary)
                print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
                return 2
            time.sleep(min(5, max(0, preflight_deadline - time.monotonic())))

        trackers = {
            name: FileDeltaTracker.start(
                name,
                path,
                replace_record_key="ts" if name == "watchdog_history" else None,
            )
            for name, path in TRACKED_FILES.items()
        }
        aggregate = Aggregate()
        started = time.monotonic()
        baseline = full_sample(
            args.dashboard_url, args.ollama_url, trackers, include_light=True
        )
        aggregate.update_light(baseline, elapsed_seconds=0)
        aggregate.update_full(baseline, elapsed_seconds=0)
        writer.write(
            {
                "kind": "baseline",
                "elapsed_seconds": 0,
                **baseline,
            }
        )

        baseline_frontier_error = frontier_baseline_error(baseline.get("frontier"))
        baseline_dashboard = baseline.get("dashboard", {})
        baseline_projection = baseline_dashboard.get("snapshot", {})
        baseline_save_history = baseline_dashboard.get("save_history", {})
        baseline_errors: list[str] = []
        if baseline_frontier_error is not None:
            baseline_errors.append(baseline_frontier_error)
        if not all_services_loaded(baseline.get("launchd", {})):
            baseline_errors.append("not_all_services_loaded")
        if idle_violations(baseline_projection):
            baseline_errors.append("baseline_dashboard_not_idle")
        if not runtime_is_expected(baseline_projection, args.expected_commit):
            baseline_errors.append("baseline_runtime_not_expected_commit")
        if baseline_save_history.get("valid") is not True:
            baseline_errors.append("baseline_save_history_invalid")
        if baseline_save_history.get("pending_bytes") != 0:
            baseline_errors.append("baseline_save_history_pending_bytes_nonzero")
        if baseline_save_history.get("pending_segment_count") != 0:
            baseline_errors.append("baseline_save_history_pending_segments")
        baseline_ollama_error = ollama_quiescence_error(baseline.get("ollama", {}))
        if baseline_ollama_error is not None:
            baseline_errors.append(f"baseline_{baseline_ollama_error}")
        events = baseline.get("frontier", {}).get("frontier_events", {})
        if events.get("start_count") != 0:
            baseline_errors.append("frontier_start_count_not_zero")
        if baseline_errors:
            summary = {
                "kind": "summary",
                "timestamp": utc_now(),
                "status": "baseline_failed",
                "errors": baseline_errors,
                "evidence_path": str(output),
            }
            writer.write(summary)
            print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
            return 2

        deadline = started + args.duration_seconds
        next_probe = started + args.probe_seconds
        next_sample = started + args.sample_seconds
        last_full = baseline
        while True:
            now = time.monotonic()
            if now >= deadline:
                break
            wake_at = min(next_probe, next_sample, deadline)
            time.sleep(max(0, wake_at - now))
            now = time.monotonic()
            elapsed = now - started
            if now + 1e-6 >= next_probe:
                probe = light_sample(args.ollama_url)
                aggregate.update_light(probe, elapsed_seconds=elapsed)
                writer.write(
                    {
                        "kind": "probe",
                        "elapsed_seconds": round(elapsed, 3),
                        **probe,
                    }
                )
                while next_probe <= now:
                    next_probe += args.probe_seconds
            if now + 1e-6 >= next_sample:
                sample = full_sample(
                    args.dashboard_url,
                    args.ollama_url,
                    trackers,
                    include_light=False,
                )
                aggregate.update_full(sample, elapsed_seconds=elapsed)
                last_full = sample
                writer.write(
                    {
                        "kind": "sample",
                        "elapsed_seconds": round(elapsed, 3),
                        **sample,
                    }
                )
                while next_sample <= now:
                    next_sample += args.sample_seconds

        final_wait_deadline = time.monotonic() + args.final_idle_wait_seconds
        while True:
            elapsed = time.monotonic() - started
            wait_light = light_sample(args.ollama_url)
            wait_dashboard = dashboard_snapshot(
                args.dashboard_url, all_endpoints=False
            )
            aggregate.update_light(wait_light, elapsed_seconds=elapsed)
            idle = not wait_dashboard.get("idle_violations")
            writer.write(
                {
                    "kind": "final_idle_wait",
                    "elapsed_seconds": round(elapsed, 3),
                    "idle": idle,
                    **wait_light,
                    "dashboard": wait_dashboard,
                }
            )
            if idle or time.monotonic() >= final_wait_deadline:
                break
            time.sleep(min(5, max(0, final_wait_deadline - time.monotonic())))

        elapsed = time.monotonic() - started
        final = full_sample(
            args.dashboard_url, args.ollama_url, trackers, include_light=True
        )
        aggregate.update_light(final, elapsed_seconds=elapsed)
        aggregate.update_full(final, elapsed_seconds=elapsed)
        last_full = final
        writer.write(
            {
                "kind": "final",
                "elapsed_seconds": round(elapsed, 3),
                **final,
            }
        )

        assertions = make_assertions(
            baseline,
            last_full,
            aggregate,
            trackers,
            expected_commit=args.expected_commit,
        )
        failed = sorted(name for name, passed in assertions.items() if not passed)
        baseline_runs = {
            role: row.get("runs")
            for role, row in baseline.get("launchd", {}).items()
            if isinstance(row, Mapping)
        }
        final_runs = {
            role: row.get("runs")
            for role, row in last_full.get("launchd", {}).items()
            if isinstance(row, Mapping)
        }
        summary = {
            "kind": "summary",
            "timestamp": utc_now(),
            "status": "passed" if not failed else "failed",
            "duration_seconds": round(time.monotonic() - started, 3),
            "assertions": assertions,
            "failed_assertions": failed,
            "launchctl_runs": {"baseline": baseline_runs, "final": final_runs},
            "file_deltas": {
                name: tracker.projection(status="final")
                for name, tracker in trackers.items()
            },
            "frontier_sha256": {
                "baseline": sha256_value(baseline.get("frontier")),
                "final": sha256_value(last_full.get("frontier")),
            },
            "swap": {
                "baseline_available": aggregate.baseline_swap_available,
                "baseline_bytes": aggregate.baseline_swap_bytes,
                "max_bytes": aggregate.max_swap_bytes,
                "unavailable_count": aggregate.swap_unavailable_count,
                "delta_bytes": (
                    aggregate.max_swap_bytes - aggregate.baseline_swap_bytes
                    if aggregate.max_swap_bytes is not None
                    and aggregate.baseline_swap_bytes is not None
                    else None
                ),
            },
            "max_related_rss_bytes": aggregate.max_related_rss_bytes,
            "max_related_process_rss_bytes": aggregate.max_related_process_rss_bytes,
            "max_ollama_model_bytes": aggregate.max_ollama_model_bytes,
            "max_duplicates": aggregate.max_duplicates,
            "min_role_counts": aggregate.min_role_counts,
            "max_role_counts": aggregate.max_role_counts,
            "process_unavailable_count": aggregate.process_unavailable_count,
            "model_stability": aggregate.model_tracker.projection(),
            "endpoint_failures": aggregate.endpoint_failures,
            "launchctl_run_regressions": aggregate.launchctl_run_regressions,
            "services_unloaded_observations": aggregate.services_unloaded_observations,
            "evidence_path": str(output),
        }
        writer.write(summary)
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
        return 0 if not failed else 1
    except KeyboardInterrupt:
        interrupted = {
            "kind": "interrupted",
            "timestamp": utc_now(),
            "status": "interrupted",
            "evidence_path": str(output),
        }
        writer.write(interrupted)
        print(json.dumps(interrupted, sort_keys=True), file=sys.stderr)
        return 130
    finally:
        writer.close()


if __name__ == "__main__":
    raise SystemExit(main())
