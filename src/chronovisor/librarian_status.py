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
    current_terminal = 0
    current_adopted = 0
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
            classification_status = str(row.get("classification_status") or "")
            current_held += int(classification_status == "held")
            current_adopted += int(classification_status == "adopted")
            current_terminal += int(
                classification_status in {"adopted", "held"}
            )
    encoded = json.dumps(rows, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return {
        "scope_generation": "sha256:" + hashlib.sha256(encoded).hexdigest(),
        "actual_total": len(actual),
        "registered_current": len(actual) - len(unregistered),
        "current_classified": current_classified,
        "current_held": current_held,
        "current_terminal": current_terminal,
        "current_adopted": current_adopted,
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
        counts["completed"] += int(
            row.get("classified")
            or row.get("migrated")
            or row.get("completed")
            or 0
        )
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


def _transaction_preimages(root: Path) -> dict[str, Any]:
    base = root / "runtime" / "librarian" / "transaction-preimages"
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
                rows.append({"transaction_id": path.name, "status": "invalid"})
                continue
            rows.append(
                {
                    "transaction_id": path.name,
                    "status": "quarantined",
                    "created_at": manifest.get("created_at"),
                    "expires_at": manifest.get("expires_at"),
                    "file_count": len(manifest.get("files") or []),
                    "input_uids": manifest.get("input_uids") or [],
                    "canonical_uid": manifest.get("canonical_uid"),
                }
            )
    return {"count": len(rows), "recent": rows[:5]}


def _migration_dispositions(root: Path) -> dict[str, Any]:
    path = root / "runtime" / "librarian" / "migration-dispositions.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return {"terminal": 0, "scope_generation": None}
    pages = payload.get("pages")
    pages = pages if isinstance(pages, dict) else {}
    counts = Counter(
        str(row.get("disposition") or "unknown")
        for row in pages.values()
        if isinstance(row, dict)
    )
    recent = [
        {
            "uid": str(uid),
            "disposition": row.get("disposition"),
            "reason": row.get("reason"),
            "canonical_uid": row.get("canonical_uid"),
            "transaction_id": row.get("transaction_id"),
        }
        for uid, row in list(pages.items())[-20:]
        if isinstance(row, dict)
    ][::-1]
    return {
        "terminal": sum(
            isinstance(row, dict)
            and row.get("disposition")
            in {"classified", "merged", "canonical", "keep-both", "explicit-hold"}
            for row in pages.values()
        ),
        "scope_generation": payload.get("scope_generation"),
        "counts": dict(sorted(counts.items())),
        "recent": recent,
    }


def _soak_status(root: Path, now: datetime) -> dict[str, Any]:
    path = root / "runtime" / "librarian" / "soak.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return {"status": "not_started", "remaining_seconds": None}
    if payload.get("observation_mode") == "concurrent_migration":
        try:
            starts = datetime.fromisoformat(str(payload["starts_at"]))
            if starts.tzinfo is None:
                starts = starts.replace(tzinfo=timezone.utc)
        except (KeyError, TypeError, ValueError):
            return {"status": "invalid", "remaining_seconds": None}
        return {
            **payload,
            "remaining_seconds": 0,
            "elapsed_seconds": max(0, int((now - starts).total_seconds())),
        }
    try:
        ends = datetime.fromisoformat(str(payload["ends_at"]))
        if ends.tzinfo is None:
            ends = ends.replace(tzinfo=timezone.utc)
    except (KeyError, TypeError, ValueError):
        return {"status": "invalid", "remaining_seconds": None}
    remaining = max(0, int((ends - now).total_seconds()))
    return {
        **payload,
        "status": "complete" if remaining == 0 else "running",
        "remaining_seconds": remaining,
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


def _rollout_status(root: Path) -> dict[str, Any]:
    path = root / "runtime" / "librarian" / "rollout.json"
    if not path.exists():
        return {
            "status": "not_started",
            "stage": None,
            "updated_at": None,
        }
    try:
        payload = read_sealed_json(path, recover_backup=False)
    except DurableStateError:
        return {
            "status": "unreadable",
            "stage": None,
            "updated_at": None,
        }
    return {
        "status": str(payload.get("status") or "unknown"),
        "stage": payload.get("stage"),
        "updated_at": payload.get("updated_at"),
    }


def _current_quality(
    root: Path,
    persisted: Mapping[str, Any],
) -> dict[str, Any]:
    """Overlay the latest sealed calibration on the last shadow sweep."""

    quality = dict(persisted)
    path = root / "classification" / "calibration.json"
    if not path.exists():
        return quality
    try:
        calibration = read_sealed_json(path, recover_backup=False)
    except DurableStateError:
        quality["locked_holdout"] = "unreadable"
        return quality
    if calibration.get("schema") != "chronovisor.classification-calibration.v1":
        quality["locked_holdout"] = "invalid"
        return quality
    quality["locked_holdout"] = str(calibration.get("status") or "missing")
    metrics = calibration.get("holdout_metrics")
    quality["holdout_metrics"] = dict(metrics) if isinstance(metrics, Mapping) else {}
    gates = calibration.get("gates")
    quality["forced_misclassification_gate"] = (
        gates.get("forced_misclassification")
        if isinstance(gates, Mapping)
        else None
    )
    return quality


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
    try:
        from chronovisor.classification import classification_authority_status

        authority = classification_authority_status(root)
    except Exception:
        authority = {
            "active": False,
            "reason": "classification_authority_unavailable",
        }
    progress = {
        str(key): dict(value) if isinstance(value, Mapping) else {}
        for key, value in (state.get("progress") or {}).items()
    }
    actual_total = int(observed["actual_total"])
    current_classified = int(observed["current_classified"])
    current_held = int(observed["current_held"])
    current_terminal = int(observed["current_terminal"])
    current_adopted = int(observed["current_adopted"])
    observed_generation = str(observed["scope_generation"])
    for key, numerator in (
        ("uid", int(observed["registered_current"])),
        ("classification_shadow", current_classified),
        ("classification_terminal", current_terminal),
        ("migration_batch", current_terminal),
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
    preimages = _transaction_preimages(root)
    queue = {
        **queue,
        "queued": actual_total - current_classified,
        "actionable": actual_total - current_classified + len(observed["missing"]),
        "running": 0,
        "held": current_held,
        "quarantined": int(preimages["count"]),
        "completed": current_adopted,
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
        "authority": authority,
        "scope_generation": observed_generation,
        "progress": progress,
        "queue": queue,
        "debts": debts,
    }
    code = _derive_code(state, queue)
    if code not in STATE_CODES:
        code = "BLOCKED"
    authority = dict(authority)
    if blocked_reasons:
        reason_codes = sorted(set(blocked_reasons))
    elif code == "NOT_READY":
        reason_codes = [
            value
            for value in str(authority.get("reason") or "not_ready").split(",")
            if value
        ]
    elif code == "MIGRATING":
        reason_codes = ["initial_organization_incomplete"]
    elif code == "FALLING_BEHIND":
        reason_codes = ["oldest_actionable_slo_exceeded"]
    elif code == "CATCHING_UP":
        reason_codes = ["scope_or_queue_not_current"]
    elif code == "STEADY_WITH_HOLDS":
        reason_codes = ["terminal_holds_or_quarantine_present"]
    else:
        reason_codes = ["all_release_and_current_scope_gates_passed"]
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
    elif code == "MIGRATING":
        detail = (
            f"{current_terminal} of {actual_total} pages have terminal "
            "classification; initial organization and concurrent migration "
            "observation are still in progress."
        )
    dispositions = _migration_dispositions(root)
    soak = _soak_status(root, current)
    flow_24h = _flow(events, current - timedelta(hours=24))
    flow_7d = _flow(events, current - timedelta(days=7))
    completion_rate = flow_7d["completed"] / 7
    arrival_rate = flow_7d["arrivals"] / 7
    actionable = int(queue.get("actionable") or 0)
    if actionable and completion_rate > arrival_rate:
        eta = {
            "status": "estimated",
            "days": actionable / (completion_rate - arrival_rate),
            "completion_per_day": completion_rate,
            "arrival_per_day": arrival_rate,
        }
    elif actionable:
        eta = {
            "status": "falling_behind_or_unstable",
            "days": None,
            "completion_per_day": completion_rate,
            "arrival_per_day": arrival_rate,
        }
    else:
        eta = {"status": "current", "days": 0}
    return {
        "schema": SNAPSHOT_SCHEMA,
        "generated_at": _iso(current),
        "state": code,
        "reason_codes": reason_codes,
        "threshold_version": authority.get("threshold_version"),
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
        "quality": _current_quality(root, state.get("quality") or {}),
        "resources": dict(state.get("resources") or {}),
        "flow": {
            "24h": flow_24h,
            "7d": flow_7d,
        },
        "eta": eta,
        "growth": dict(state.get("growth") or {}),
        "restore_points": _restore_points(root),
        "transaction_preimages": preimages,
        "migration_dispositions": dispositions,
        "rollout": _rollout_status(root),
        "soak": soak,
        "recent_receipts": recent_receipts[:20],
        "blocked_reasons": blocked_reasons,
        "last_run": state.get("last_run"),
    }
