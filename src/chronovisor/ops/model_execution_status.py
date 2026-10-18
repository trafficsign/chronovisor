"""Payload-free local model identity and observed queue throughput."""

from __future__ import annotations

import tomllib
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from chronovisor.core.model_execution import (
    automatic_cloud_allowed,
    local_reasoning_settings,
)
from chronovisor.recall import recall_distillation_store


def _daily_status(root: Path) -> dict[str, Any]:
    path = recall_distillation_store.distillation_dir(root) / "local-daily/status.json"
    if not path.exists():
        return {"status": "not_started"}
    try:
        record = recall_distillation_store.read_sealed(path)
    except recall_distillation_store.DistillationStoreError:
        return {"status": "receipt_invalid"}
    return {key: record[key] for key in (
        "status", "started_at", "finished_at", "elapsed_seconds", "teacher",
        "labels_written", "counterfactuals_written", "promotion_allowed", "reason", "drain",
    ) if key in record}


def model_execution_status(root: Path) -> dict[str, Any]:
    try:
        config = tomllib.loads((root / "config.toml").read_text())
        role = "recall.distill.teacher.a"
        route = config.get("llm", {}).get("roles", {}).get(role, {})
        model = str(route.get("model") or "")
        reasoning = local_reasoning_settings(role, model, root=root)
        distillation = config.get("recall", {}).get("distillation", {})
        profile = distillation.get("teacher_profile")
        return {
            "automatic_cloud_allowed": automatic_cloud_allowed(root),
            "teacher": {
                "active": distillation.get("enabled") is True and profile in {"local-single-v1", "local-triad-v1"},
                "profile": profile, "role": role, "provider": route.get("provider"),
                "model": model, "revision": route.get("revision"),
                "reasoning": reasoning, "identity_source": "runtime_config",
            },
            "switched_at": config.get("model_execution", {}).get("switched_at"),
            "local_daily": _daily_status(root),
        }
    except (OSError, ValueError, TypeError, AttributeError):
        return {"automatic_cloud_allowed": False, "status": "config_unavailable"}


def _timestamp(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    # Queue writers historically use local naive timestamps; cutover receipts
    # use explicit UTC. Interpret each according to its actual clock.
    return parsed.astimezone(UTC)


def observed_drain(
    rows: Iterable[Mapping[str, Any]], *, status_field: str,
    terminal_statuses: set[str], timestamp_fields: tuple[str, ...],
    pending: int, switched_at: object = None, now: datetime | None = None,
) -> dict[str, Any]:
    """Count terminal outcomes since cutover, never old cloud throughput."""

    now = now or datetime.now(UTC)
    since = max(now - timedelta(hours=24), _timestamp(switched_at) or now - timedelta(hours=24))
    seconds = max(0.0, (now - since).total_seconds())
    completed = 0
    latest: datetime | None = None
    for row in rows:
        if row.get(status_field) not in terminal_statuses:
            continue
        dates = []
        for key in timestamp_fields:
            value: object = row
            for part in key.split("."):
                value = value.get(part) if isinstance(value, Mapping) else None
            if (date := _timestamp(value)) is not None:
                dates.append(date)
        stamp = max(dates) if dates else None
        if stamp is not None and since <= stamp <= now:
            completed += 1
            latest = max(latest, stamp) if latest else stamp
    rate = completed * 3600 / seconds if seconds else 0.0
    return {
        "observation_seconds": round(seconds, 3), "terminal_items": completed,
        "items_per_hour": round(rate, 3), "pending": pending,
        "eta_hours_at_observed_rate": round(pending / rate, 3) if rate else None,
        "last_terminal_at": latest.isoformat() if latest else None,
        "status": "draining" if completed and pending else "idle" if not pending else "awaiting_progress",
        "basis": "terminal_receipts_since_cutover_or_last_24h",
    }
