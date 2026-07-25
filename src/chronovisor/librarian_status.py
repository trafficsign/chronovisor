"""Host-derived Librarian status and false-green prevention."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from chronovisor.durable_state import DurableStateError, read_sealed_json
from chronovisor.merge_ledger import MergeLedger
from chronovisor.page_registry import PageRegistry, PageRegistryError

SNAPSHOT_SCHEMA = "chronovisor.librarian-status.v1"
STATE_SCHEMA = "chronovisor.librarian-state.v1"
STATE_CODES = {
    "BLOCKED",
    "FALLING_BEHIND",
    "NOT_READY",
    "MIGRATING",
    "CATCHING_UP",
    "STEADY_WITH_HOLDS",
    "STEADY_CLEAN",
}


def _now(now: datetime | None = None) -> datetime:
    value = now or datetime.now(timezone.utc)
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def _iso(value: datetime) -> str:
    return value.isoformat(timespec="seconds")


def _empty_state() -> dict[str, Any]:
    return {
        "schema": STATE_SCHEMA,
        "enabled": True,
        "mode": "shadow",
        "generation": 0,
        "scope_generation": None,
        "last_swept_scope_generation": None,
        "initial_organization_complete_at": None,
        "authority": {
            "active": False,
            "package_complete": False,
            "calibrated": False,
            "reason": "shadow_state_not_initialized",
        },
        "progress": {},
        "queue": {},
        "debts": {},
        "quality": {},
        "resources": {},
        "blocked_reasons": [],
        "last_run": None,
    }


def load_librarian_state(root: Path) -> dict[str, Any]:
    path = root / "runtime" / "librarian" / "state.json"
    if not path.exists():
        return _empty_state()
    try:
        state = read_sealed_json(path, recover_backup=False)
    except DurableStateError:
        state = _empty_state()
        state["blocked_reasons"] = ["librarian_state_unreadable"]
    if state.get("schema") != STATE_SCHEMA:
        state = _empty_state()
        state["blocked_reasons"] = ["librarian_state_schema_mismatch"]
    return state


def _observed_scope(root: Path, registry: Mapping[str, Any]) -> dict[str, Any]:
    """Compare live Markdown with the last persisted registry snapshot."""

    actual: dict[str, Any] = {}
    for path in PageRegistry._page_paths(root, include_system=True):
        try:
            actual[str(path.relative_to(root))] = path.stat()
        except FileNotFoundError:
            continue
    registered = {
        str(row.get("path") or ""): row
        for row in (registry.get("pages") or {}).values()
        if isinstance(row, Mapping)
        and row.get("status") != "superseded"
        and row.get("path")
    }
    unregistered = sorted(set(actual) - set(registered))
    missing = sorted(set(registered) - set(actual))
    changed: list[str] = []
    current_classified = 0
    current_held = 0
    rows = []
    for relative, stat in sorted(actual.items()):
        row = registered.get(relative)
        status = str(row.get("status") or "active") if row else "active"
        rows.append((relative, stat.st_size, stat.st_mtime_ns, status))
        if row is None:
            continue
        is_current = (
            row.get("content_size") == stat.st_size
            and row.get("content_mtime_ns") == stat.st_mtime_ns
        )
        if not is_current:
            changed.append(relative)
            continue
        if isinstance(row.get("classification"), Mapping):
            current_classified += 1
            current_held += int(row.get("classification_status") == "held")
    encoded = json.dumps(rows, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return {
        "scope_generation": "sha256:" + hashlib.sha256(encoded).hexdigest(),
        "actual_total": len(actual),
        "registered_current": len(actual) - len(unregistered),
        "current_classified": current_classified,
        "current_held": current_held,
        "unregistered": unregistered,
        "changed": changed,
        "missing": missing,
        "actionable": len(unregistered) + len(changed) + len(missing),
    }


def _read_events(root: Path, limit: int = 4000) -> list[dict[str, Any]]:
    path = root / "runtime" / "librarian" / "events.jsonl"
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines()[-max(0, limit) :]:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _flow(events: Iterable[Mapping[str, Any]], since: datetime) -> dict[str, int]:
    counts = Counter()
    for row in events:
        try:
            timestamp = datetime.fromisoformat(str(row.get("timestamp")))
        except (TypeError, ValueError):
            continue
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        if timestamp < since:
            continue
        counts["runs"] += int(row.get("event") == "shadow_run")
        counts["arrivals"] += int(row.get("created") or 0)
        counts["completed"] += int(row.get("classified") or 0)
        counts["held"] += int(row.get("held") or 0)
    return {
        "runs": counts["runs"],
        "arrivals": counts["arrivals"],
        "completed": counts["completed"],
        "held": counts["held"],
        "net_growth": counts["arrivals"] - counts["completed"],
    }


def _restore_points(root: Path) -> dict[str, Any]:
    base = root / "runtime" / "librarian" / "migration-restore-points"
    rows: list[dict[str, Any]] = []
    if base.exists():
        for path in sorted(
            (value for value in base.iterdir() if value.is_dir()), reverse=True
        ):
            try:
                manifest = json.loads(
                    (path / "manifest.json").read_text(encoding="utf-8")
                )
            except (OSError, ValueError, json.JSONDecodeError):
                rows.append({"restore_id": path.name, "status": "invalid"})
                continue
            rows.append(
                {
                    "restore_id": path.name,
                    "status": str(manifest.get("verification_status") or "recorded"),
                    "created_at": manifest.get("created_at"),
                    "expires_at": manifest.get("expires_at"),
                    "file_count": manifest.get("file_count"),
                }
            )
    return {
        "count": len(rows),
        "verified": sum(
            row.get("status") in {"verified", "checksum_verified"} for row in rows
        ),
        "recent": rows[:5],
    }


def _derive_code(state: Mapping[str, Any], queue: Mapping[str, Any]) -> str:
    blocked = state.get("blocked_reasons") or []
    if blocked:
        return "BLOCKED"
    authority = state.get("authority")
    authority = authority if isinstance(authority, Mapping) else {}
    if not state.get("enabled") or not authority.get("active"):
        return "NOT_READY"
    progress = state.get("progress")
    progress = progress if isinstance(progress, Mapping) else {}
    sweep = progress.get("full_sweep")
    sweep = sweep if isinstance(sweep, Mapping) else {}
    if not state.get("initial_organization_complete_at"):
        return "MIGRATING"
    if int(queue.get("oldest_age_seconds") or 0) > 7 * 86_400:
        return "FALLING_BEHIND"
    if int(queue.get("actionable") or 0) or not sweep.get("current"):
        return "CATCHING_UP"
    if int(queue.get("held") or 0) or int(queue.get("quarantined") or 0):
        return "STEADY_WITH_HOLDS"
    return "STEADY_CLEAN"


def build_librarian_status(
    root: Path,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    current = _now(now)
    state = load_librarian_state(root)
    queue = dict(state.get("queue") or {})
    events = _read_events(root)
    recent_receipts = [
        {
            "timestamp": row.get("timestamp"),
            "event": row.get("event"),
            "status": row.get("status"),
            "classified": row.get("classified"),
            "held": row.get("held"),
            "scope_generation": row.get("scope_generation"),
        }
        for row in events[-12:]
    ][::-1]
    recent_receipts.extend(MergeLedger(root).recent(limit=8)[::-1])
    try:
        registry = PageRegistry(root).load()
        registry_error = None
    except PageRegistryError as exc:
        registry = PageRegistry.empty()
        registry_error = str(exc)
    blocked_reasons = list(state.get("blocked_reasons") or [])
    if registry_error:
        blocked_reasons.append("page_registry_unreadable")
        state = {**state, "blocked_reasons": blocked_reasons}
    observed = _observed_scope(root, registry)
    progress = {
        str(key): dict(value) if isinstance(value, Mapping) else {}
        for key, value in (state.get("progress") or {}).items()
    }
    actual_total = int(observed["actual_total"])
    current_classified = int(observed["current_classified"])
    current_held = int(observed["current_held"])
    observed_generation = str(observed["scope_generation"])
    for key, numerator in (
        ("uid", int(observed["registered_current"])),
        ("classification_shadow", current_classified),
        ("classification_terminal", current_held),
        ("migration_batch", current_classified),
    ):
        progress[key] = {
            **progress.get(key, {}),
            "numerator": numerator,
            "denominator": actual_total,
            "scope_generation": observed_generation,
        }
    sweep_current = bool(
        not observed["actionable"]
        and state.get("last_swept_scope_generation") == observed_generation
    )
    progress["full_sweep"] = {
        **progress.get("full_sweep", {}),
        "numerator": int(sweep_current),
        "denominator": 1,
        "scope_generation": observed_generation,
        "current": sweep_current,
    }
    queue = {
        **queue,
        "queued": actual_total - current_classified,
        "actionable": actual_total - current_classified + len(observed["missing"]),
        "running": 0,
        "held": current_held,
        "completed": current_classified - current_held,
    }
    debts = {
        **dict(state.get("debts") or {}),
        "unclassified": actual_total - current_classified,
        "explicit_hold": current_held,
        "scope_unregistered": len(observed["unregistered"]),
        "scope_changed": len(observed["changed"]),
        "scope_missing": len(observed["missing"]),
    }
    state = {
        **state,
        "scope_generation": observed_generation,
        "progress": progress,
        "queue": queue,
        "debts": debts,
    }
    code = _derive_code(state, queue)
    if code not in STATE_CODES:
        code = "BLOCKED"
    authority = dict(state.get("authority") or {})
    detail = str(authority.get("reason") or "librarian state available")
    if code == "NOT_READY":
        detail = (
            (
                f"Live scope has {observed['actionable']} unswept change(s); "
                if observed["actionable"]
                else "Shadow migration is current; "
            )
            + "classification authority remains "
            "fail-closed until a complete licensed UDC package and calibrated "
            "locked fixture exist."
        )
    return {
        "schema": SNAPSHOT_SCHEMA,
        "generated_at": _iso(current),
        "state": code,
        "mode": state.get("mode") or "shadow",
        "detail": detail,
        "enabled": bool(state.get("enabled")),
        "initial_organization_complete_at": state.get(
            "initial_organization_complete_at"
        ),
        "scope_generation": observed_generation,
        "current_generation": int(registry.get("generation") or 0),
        "last_swept_generation": state.get("last_swept_scope_generation"),
        "authority": authority,
        "progress": progress,
        "queue": queue,
        "debts": debts,
        "quality": dict(state.get("quality") or {}),
        "resources": dict(state.get("resources") or {}),
        "flow": {
            "24h": _flow(events, current - timedelta(hours=24)),
            "7d": _flow(events, current - timedelta(days=7)),
        },
        "eta": state.get("eta"),
        "restore_points": _restore_points(root),
        "recent_receipts": recent_receipts[:20],
        "blocked_reasons": blocked_reasons,
        "last_run": state.get("last_run"),
    }
