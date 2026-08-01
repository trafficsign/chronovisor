"""Sealed candidate/canary rollout with automatic fail-closed rollback."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from chronovisor.core.durable_state import read_sealed_json, write_sealed_json
from chronovisor.core.store import CHRONOVISOR_ROOT

PROMOTION_FILE = CHRONOVISOR_ROOT / "runtime" / "typed-graph" / "promotion.json"
CANARY_STEPS = (5, 25, 100)
CANARY_SAMPLE_UNIT = "distinct_applied_session_hashes"


def selected_for_canary(session_id: str, percent: int) -> bool:
    if not session_id or percent <= 0:
        return False
    bucket = int(hashlib.sha256(session_id.encode()).hexdigest()[:8], 16) % 100
    return bucket < min(100, percent)


def applied_canary_session_count(path_ledger: Path) -> int:
    """Count privacy-safe sessions that actually received typed candidates.

    Shadow projections are deliberately excluded.  The count is monotonic for
    the append-only ledger and can therefore advance a rollout beyond the
    finite locked-fixture evaluation set.
    """

    sessions: set[str] = set()
    try:
        lines = path_ledger.read_text(encoding="utf-8").splitlines()
    except OSError:
        return 0
    for line in lines:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict) or row.get("shadow") is not False:
            continue
        session_hash = str(row.get("session_hash") or "")
        path_ids = [
            str(value)
            for value in [
                *(row.get("relation_ids") or []),
                *(row.get("entity_merge_ids") or []),
            ]
            if isinstance(value, str) and value
        ]
        if session_hash and path_ids:
            sessions.add(session_hash)
    return len(sessions)


def advance_rollout(
    *,
    gates: Mapping[str, bool],
    sample_count: int,
    promotion_file: Path = PROMOTION_FILE,
    minimum_step_samples: int = 100,
    manifest_sha256: str = "",
    relation_snapshot_sha256: str = "",
    rubric_sha256: str = "",
    model_manifest_sha256: str = "",
) -> dict[str, Any]:
    try:
        previous = read_sealed_json(promotion_file, recover_backup=True)
    except Exception:
        previous = {}
    prior_percent = int(previous.get("canary_percent") or 0)
    prior_started = int(previous.get("stage_started_sample_count") or 0)
    all_pass = bool(gates) and all(value is True for value in gates.values())
    if not all_pass:
        percent = 0
        mode = "shadow"
        reason = "gate_failed"
        started = sample_count
    elif prior_percent == 0:
        percent = 5
        mode = "candidate"
        reason = "sealed_gate_passed"
        started = sample_count
    elif sample_count - prior_started >= minimum_step_samples:
        percent = next((step for step in CANARY_STEPS if step > prior_percent), 100)
        mode = "active" if percent == 100 else "candidate"
        reason = "canary_advanced"
        started = sample_count
    else:
        percent = prior_percent
        mode = "active" if percent == 100 else "candidate"
        reason = "collecting_canary_samples"
        started = prior_started
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "mode": mode,
        "canary_percent": percent,
        "stage_started_sample_count": started,
        "sample_count": sample_count,
        "sample_unit": CANARY_SAMPLE_UNIT,
        "gates": dict(sorted(gates.items())),
        "reason": reason,
        "rollback_teacher": "current",
        "manifest_sha256": manifest_sha256,
        "relation_snapshot_sha256": relation_snapshot_sha256,
        "rubric_sha256": rubric_sha256,
        "model_manifest_sha256": model_manifest_sha256,
    }
    return write_sealed_json(promotion_file, payload)


def rollback(*, reason: str, promotion_file: Path = PROMOTION_FILE) -> dict[str, Any]:
    try:
        previous = read_sealed_json(promotion_file, recover_backup=True)
    except Exception:
        previous = {}
    payload = advance_rollout(
        gates={"manual_or_runtime_guard": False},
        sample_count=int(previous.get("sample_count") or 0),
        promotion_file=promotion_file,
        manifest_sha256=str(previous.get("manifest_sha256") or ""),
        relation_snapshot_sha256=str(
            previous.get("relation_snapshot_sha256") or ""
        ),
        rubric_sha256=str(previous.get("rubric_sha256") or ""),
        model_manifest_sha256=str(previous.get("model_manifest_sha256") or ""),
    )
    payload = {**payload, "rollback_reason": reason[:160]}
    return write_sealed_json(promotion_file, payload)
