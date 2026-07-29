"""Deterministic L0/L1 recovery runbooks; no LLM judgment is involved."""

from __future__ import annotations

import argparse
import json
import os
import plistlib
import subprocess
from pathlib import Path
from typing import Any

from chronovisor.core.durable_state import (
    atomic_write_bytes,
    canonical_bytes,
    read_sealed_json,
)
from chronovisor.core.runtime_config import runtime_repo_root
from chronovisor.core.store import CHRONOVISOR_ROOT

RUNBOOK_VERSION = 1
SERVICE_LABELS = {
    "dashboard": "com.trafficsign.chronovisor-dashboard",
    "ingest": "com.trafficsign.chronovisor-ingest-drain",
    "sleep": "com.trafficsign.chronovisor-sleep",
    "watchdog": "com.trafficsign.chronovisor-watchdog",
    "observer": "com.trafficsign.chronovisor-deadman-observer",
    "converge": "com.trafficsign.chronovisor-converge",
    "soak": "com.trafficsign.chronovisor-soak",
}
KEEPALIVE_SERVICES = frozenset({"dashboard", "ingest"})


def _service_plist(service: str) -> Path:
    label = SERVICE_LABELS[service]
    if service in KEEPALIVE_SERVICES:
        return runtime_repo_root() / "launchd" / f"{label}.plist"
    return Path.home() / "Library" / "LaunchAgents" / f"{label}.plist"


def launchd_status(service: str) -> dict[str, Any]:
    label = SERVICE_LABELS[service]
    target = f"gui/{os.getuid()}/{label}"
    proc = subprocess.run(
        ["launchctl", "print", target],
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )
    plist_path = _service_plist(service)
    try:
        with plist_path.open("rb") as stream:
            plist = plistlib.load(stream)
    except (OSError, plistlib.InvalidFileException):
        plist = {}
    keep_alive = plist.get("KeepAlive") is True
    expected_keep_alive = service in KEEPALIVE_SERVICES
    keep_alive_policy = plist.get("KeepAlive")
    scheduled = bool(plist.get("StartInterval") or plist.get("StartCalendarInterval"))
    restart_policy_valid = (
        keep_alive
        if expected_keep_alive
        else scheduled
        or (
            service == "soak"
            and isinstance(keep_alive_policy, dict)
            and keep_alive_policy.get("SuccessfulExit") is False
        )
    )
    return {
        "service": service,
        "label": label,
        "loaded": proc.returncode == 0,
        "returncode": proc.returncode,
        "plist_path": str(plist_path),
        "keep_alive": keep_alive,
        "keep_alive_expected": expected_keep_alive,
        "restart_policy_valid": restart_policy_valid,
    }


def _restore_targets(chronovisor_root: Path) -> dict[str, Path]:
    return {
        "managed-holds": chronovisor_root / "runtime" / "managed-holds" / "state.json",
        "quality-probe": chronovisor_root / "runtime" / "quality" / "probe-latest.json",
        "provisional-index": chronovisor_root
        / "runtime"
        / "provisional-recall"
        / "index.json",
    }


def run_l1(
    action: str,
    *,
    chronovisor_root: Path = CHRONOVISOR_ROOT,
    service: str | None = None,
    target: str | None = None,
    limit: int = 8,
    dry_run: bool = False,
) -> dict[str, Any]:
    if action == "recover-hold-leases":
        from chronovisor.librarian.managed_hold import ManagedHoldStore

        store = ManagedHoldStore(
            chronovisor_root / "runtime" / "managed-holds" / "state.json"
        )
        recovered = [] if dry_run else store.recover_expired()
        return {"status": "ok", "action": action, "recovered": recovered}
    if action == "rebuild-read-back-view":
        from chronovisor.ingest.read_back_repair import run_read_back_repair

        result = run_read_back_repair(
            failure_file=chronovisor_root
            / "runtime"
            / "ingest-read-back-failures.jsonl",
            ledger_file=chronovisor_root / "runtime" / "ingest-read-back-repair.json",
            max_items=0,
            dry_run=dry_run,
        )
        return {"status": result.get("status"), "action": action, "result": result}
    if action == "sync-provisional-recall":
        from chronovisor.recall.provisional_recall import sync_index

        if dry_run:
            return {"status": "ok", "action": action, "dry_run": True}
        result = sync_index(chronovisor_root=chronovisor_root)
        return {
            "status": "ok",
            "action": action,
            "entries": len(result.get("entries", [])),
        }
    if action == "unload-models":
        from chronovisor.core.runtime_config import load_decision_router_config

        config = load_decision_router_config()
        models = sorted(
            {
                config.primary_model,
                config.challenger_model,
                config.tie_break_model,
            }
        )
        if dry_run:
            return {
                "status": "ok",
                "action": action,
                "models": models,
                "dry_run": True,
            }
        from chronovisor.core.ollama import unload_named_model

        results = {model: unload_named_model(model) for model in models}
        return {
            "status": "ok" if all(results.values()) else "error",
            "action": action,
            "models": results,
        }
    if action == "resend-due-queues":
        from chronovisor.ops import background_jobs

        if dry_run:
            return {
                "status": "ok",
                "action": action,
                "dry_run": True,
                "queue": background_jobs.snapshot(),
            }
        result = background_jobs.retry_due(limit=max(0, int(limit)))
        return {"status": result.get("status"), "action": action, "result": result}
    if action == "restore-durable-state":
        targets = _restore_targets(chronovisor_root)
        if target not in targets:
            raise ValueError("restore-durable-state requires an allowlisted target")
        destination = targets[target]
        backup = destination.with_name(f"{destination.name}.bak")
        recovered = read_sealed_json(backup)
        if dry_run:
            return {
                "status": "ok",
                "action": action,
                "target": target,
                "backup": str(backup),
                "dry_run": True,
            }
        atomic_write_bytes(
            destination,
            canonical_bytes(recovered),
            backup=False,
        )
        verified = read_sealed_json(destination)
        return {
            "status": "ok",
            "action": action,
            "target": target,
            "restored_seal_sha256": verified.get("seal_sha256"),
        }
    if action == "restart-service":
        if service not in SERVICE_LABELS:
            raise ValueError("restart-service requires an allowlisted service")
        target = f"gui/{os.getuid()}/{SERVICE_LABELS[service]}"
        if dry_run:
            return {"status": "ok", "action": action, "target": target, "dry_run": True}
        proc = subprocess.run(
            ["launchctl", "kickstart", "-k", target],
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
        return {
            "status": "ok" if proc.returncode == 0 else "error",
            "action": action,
            "target": target,
            "returncode": proc.returncode,
            "stderr": proc.stderr[-500:],
        }
    raise ValueError(f"unknown deterministic runbook action: {action}")


def snapshot() -> dict[str, Any]:
    l0 = {name: launchd_status(name) for name in SERVICE_LABELS}
    return {
        "runbook_version": RUNBOOK_VERSION,
        "l0": l0,
        "l0_restart_policy_valid": all(
            row.get("restart_policy_valid") is True for row in l0.values()
        ),
        "l1_actions": [
            "recover-hold-leases",
            "rebuild-read-back-view",
            "sync-provisional-recall",
            "unload-models",
            "resend-due-queues",
            "restore-durable-state",
            "restart-service",
        ],
        "l2": {
            "role": "code_repair",
            "incident_kind": "system_code_repair",
            "semantic_payload_allowed": False,
            "quality_drift_allowed": False,
        },
    }


def main(argv: list[str] | None = None) -> int:
    """Run the ``chronovisor-repair-runbook`` command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=(
            "status",
            "recover-hold-leases",
            "rebuild-read-back-view",
            "sync-provisional-recall",
            "unload-models",
            "resend-due-queues",
            "restore-durable-state",
            "restart-service",
        ),
    )
    parser.add_argument("--service", choices=tuple(SERVICE_LABELS))
    parser.add_argument("--target", choices=tuple(_restore_targets(CHRONOVISOR_ROOT)))
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    payload = (
        snapshot()
        if args.action == "status"
        else run_l1(
            args.action,
            service=args.service,
            target=args.target,
            limit=args.limit,
            dry_run=args.dry_run,
        )
    )
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0 if payload.get("status", "ok") != "error" else 1
