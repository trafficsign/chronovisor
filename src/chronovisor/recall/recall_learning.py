"""Fail-closed policy history and learning gates for Recall Field."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from chronovisor.core.link_fix import atomic_write


def _sha(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def decide_learning_update(
    *,
    current: dict[str, float],
    proposed: dict[str, float],
    label_counts: dict[str, int],
    metrics: dict[str, float],
    max_change: float = 0.05,
) -> dict[str, Any]:
    """Cap changes and roll back on leakage risk or quality regression."""

    strong = int(label_counts.get("strong_positive") or 0)
    sessions = int(label_counts.get("strong_positive_sessions") or 0)
    total = int(label_counts.get("total") or 0)
    if strong < 200 or sessions < 20:
        return {
            "status": "held",
            "reason": "insufficient_strong_positive_diversity",
            "policy": current,
            "field_learning_allowed": False,
            "calibration_allowed": total >= 500,
        }
    if metrics.get("session_leakage", 0.0) > 0 or metrics.get(
        "query_leakage", 0.0
    ) > 0:
        return {
            "status": "rollback",
            "reason": "holdout_leakage",
            "policy": current,
            "field_learning_allowed": False,
            "calibration_allowed": total >= 500,
        }
    if (
        float(metrics.get("precision_delta_points") or 0.0) < -1.0
        or float(metrics.get("recall_delta_points") or 0.0) < -1.0
    ):
        return {
            "status": "rollback",
            "reason": "teacher_nondegradation_failed",
            "policy": current,
            "field_learning_allowed": False,
            "calibration_allowed": total >= 500,
        }
    bounded = {
        key: round(
            max(
                float(current.get(key, value)) - max_change,
                min(
                    float(current.get(key, value)) + max_change,
                    float(value),
                ),
            ),
            6,
        )
        for key, value in proposed.items()
    }
    return {
        "status": "candidate",
        "reason": "shadow_only_until_locked_gate",
        "policy": {**current, **bounded},
        "field_learning_allowed": True,
        "calibration_allowed": total >= 500,
    }


def append_policy_history(
    decision: dict[str, Any],
    *,
    path: Path,
) -> dict[str, Any]:
    """Append a hash-chained policy record with last-known-good identity."""

    try:
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError):
        rows = []
    previous = str(rows[-1].get("record_sha256") or "") if rows else ""
    record = {
        "schema_version": 1,
        "previous_sha256": previous,
        **decision,
    }
    record["record_sha256"] = _sha(record)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = (
        path.read_text(encoding="utf-8")
        if path.exists()
        else ""
    )
    atomic_write(
        path,
        existing
        + json.dumps(
            record,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
    )
    return record
