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
    label_counts: dict[str, Any],
    metrics: dict[str, Any],
    answer_evaluation: dict[str, Any] | None = None,
    confidence_bounds: dict[str, Any] | None = None,
    max_change: float = 0.05,
) -> dict[str, Any]:
    """Authorize only train-scoped, outcome-grounded, bounded evidence."""

    if label_counts.get("scope") != "train":
        return {
            "status": "held",
            "reason": "learning_counts_not_train_scoped",
            "policy": current,
            "field_learning_allowed": False,
            "calibration_allowed": False,
        }
    strong = int(label_counts.get("strong_positive") or 0)
    sessions = int(label_counts.get("strong_positive_sessions") or 0)
    total = int(label_counts.get("total") or 0)
    answer = answer_evaluation or (
        metrics.get("answer_evaluation")
        if isinstance(metrics.get("answer_evaluation"), dict)
        else {}
    )
    bounds = confidence_bounds or (
        metrics.get("confidence_bounds")
        if isinstance(metrics.get("confidence_bounds"), dict)
        else {}
    )
    answer_bound = bounds.get("answer_reward") if isinstance(bounds, dict) else None
    if not isinstance(answer, dict) or answer.get("passed") is not True:
        return {
            "status": "held",
            "reason": "missing_or_invalid_answer_evaluation",
            "policy": current,
            "field_learning_allowed": False,
            "calibration_allowed": total >= 500,
        }
    if (
        not isinstance(answer_bound, dict)
        or answer_bound.get("valid") is not True
        or not isinstance(answer_bound.get("samples"), int)
        or int(answer_bound["samples"]) < 20
        or not isinstance(answer_bound.get("clusters"), int)
        or int(answer_bound["clusters"]) < 20
        or not isinstance(answer_bound.get("method"), str)
        or not answer_bound["method"]
        or not isinstance(answer_bound.get("manifest_sha256"), str)
        or len(answer_bound["manifest_sha256"]) != 64
    ):
        return {
            "status": "held",
            "reason": "missing_or_invalid_confidence_evidence",
            "policy": current,
            "field_learning_allowed": False,
            "calibration_allowed": total >= 500,
        }
    point = answer_bound.get("point")
    lower = answer_bound.get("lower")
    point_floor = answer_bound.get("point_floor")
    lower_floor = answer_bound.get("lower_floor")
    if not all(
        isinstance(value, int | float) and not isinstance(value, bool)
        for value in (point, lower, point_floor, lower_floor)
    ):
        return {
            "status": "held",
            "reason": "missing_or_invalid_confidence_evidence",
            "policy": current,
            "field_learning_allowed": False,
            "calibration_allowed": total >= 500,
        }
    if float(point) < float(point_floor):
        return {
            "status": "held",
            "reason": "answer_reward_point_floor_failed",
            "policy": current,
            "field_learning_allowed": False,
            "calibration_allowed": total >= 500,
        }
    if float(lower) < float(lower_floor):
        return {
            "status": "held",
            "reason": "answer_reward_lower_bound_failed",
            "policy": current,
            "field_learning_allowed": False,
            "calibration_allowed": total >= 500,
        }
    if strong < 200 or sessions < 20:
        return {
            "status": "held",
            "reason": "insufficient_strong_positive_diversity",
            "policy": current,
            "field_learning_allowed": False,
            "calibration_allowed": total >= 500,
        }
    if any(
        float(metrics.get(key) or 0.0) > 0
        for key in (
            "session_leakage",
            "query_leakage",
            "page_leakage",
            "content_leakage",
            "timestamp_leakage",
            "embargo_leakage",
        )
    ):
        return {
            "status": "rollback",
            "reason": "split_leakage",
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
        "reason": "train_outcome_candidate_only_until_authority_gates",
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
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
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


def verify_policy_history(path: Path) -> dict[str, Any]:
    """Verify every link in an append-only policy chain."""

    try:
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError):
        rows = []
    previous = ""
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            return {"status": "invalid", "row": index, "reason": "not_object"}
        digest = str(row.get("record_sha256") or "")
        unsigned = {key: value for key, value in row.items() if key != "record_sha256"}
        if row.get("previous_sha256") != previous or digest != _sha(unsigned):
            return {"status": "invalid", "row": index, "reason": "chain_mismatch"}
        previous = digest
    return {"status": "ok", "records": len(rows), "head_sha256": previous}


def write_last_known_good(
    policy: dict[str, float],
    *,
    evaluation: dict[str, Any],
    history_head_sha256: str,
    path: Path,
) -> dict[str, Any]:
    """Atomically seal one runtime policy against its evaluation/history."""

    unsigned = {
        "schema_version": 1,
        "status": "active",
        "policy": {key: round(float(value), 6) for key, value in policy.items()},
        "evaluation": evaluation,
        "history_head_sha256": history_head_sha256,
    }
    payload = {**unsigned, "snapshot_sha256": _sha(unsigned)}
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(
        path,
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
    )
    return payload


def load_last_known_good(path: Path) -> dict[str, Any]:
    """Load only a valid sealed active policy; malformed files fail closed."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict) or payload.get("status") != "active":
        return {}
    seal = str(payload.get("snapshot_sha256") or "")
    unsigned = {
        key: value for key, value in payload.items() if key != "snapshot_sha256"
    }
    if not seal or seal != _sha(unsigned):
        return {}
    policy = payload.get("policy")
    return payload if isinstance(policy, dict) else {}
