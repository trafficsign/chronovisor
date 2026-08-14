"""Sealed, fail-closed rollout state for distilled Recall policies."""

from __future__ import annotations

import hashlib
import math
import os
import re
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from chronovisor.core.canonical_json import canonical_json_sha256_strict
from chronovisor.core.store import CHRONOVISOR_ROOT
from chronovisor.recall import recall_distillation_store as store

POLICY_SCHEMA = "chronovisor.recall-distill-policy.v2"
EVALUATION_SCHEMA = "chronovisor.recall-distill-rollout-evaluation.v2"
QUARANTINE_SCHEMA = "chronovisor.recall-distill-quarantine.v1"
_STAGES = (5, 25, 100)
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_REVISION = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}\Z")


class RolloutError(ValueError):
    """A rollout input or artifact is not safe to use."""


def _root(root: Path | None) -> Path:
    return root or CHRONOVISOR_ROOT


def _policy(root: Path, policy_id: object) -> str:
    if not isinstance(policy_id, str) or _HEX.fullmatch(policy_id) is None:
        raise RolloutError("invalid policy id")
    artifact = store.read_sealed(
        store.distillation_dir(root) / "policies" / f"{policy_id}.json",
        schema=POLICY_SCHEMA,
    )
    if artifact.get("artifact_id") != policy_id:
        raise RolloutError("policy identity mismatch")
    keys = artifact.get("feature_keys")
    weights = artifact.get("weights")
    revision = artifact.get("feature_revision")
    from chronovisor.recall.recall_distillation import (
        FAST_FEATURE_KEYS,
        TEXT_FEATURE_REVISION,
    )

    if (
        not isinstance(keys, list)
        or tuple(keys) != FAST_FEATURE_KEYS
        or not isinstance(weights, Mapping)
        or set(weights) != set(FAST_FEATURE_KEYS)
        or not isinstance(revision, str)
        or revision != TEXT_FEATURE_REVISION
    ):
        raise RolloutError("policy feature schema is invalid")
    numeric = [
        artifact.get("bias"),
        artifact.get("threshold"),
        artifact.get("abstain_margin"),
        *weights.values(),
    ]
    if not all(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        for value in numeric
    ):
        raise RolloutError("policy numeric schema is invalid")
    if (
        not 0.0 <= float(artifact["threshold"]) <= 1.0
        or not 0.0 <= float(artifact["abstain_margin"]) <= 1.0
    ):
        raise RolloutError("policy threshold schema is invalid")
    if (
        isinstance(artifact.get("max_cards"), bool)
        or not isinstance(artifact.get("max_cards"), int)
        or not 1 <= int(artifact["max_cards"]) <= 3
    ):
        raise RolloutError("policy card schema is invalid")
    return policy_id


def _pointer_policy(root: Path, kind: str) -> str:
    try:
        return _policy(root, store.read_pointer(root, kind)["policy_id"])
    except (KeyError, RolloutError, store.DistillationStoreError) as exc:
        raise RolloutError(f"invalid {kind} policy") from exc


def _lkg(root: Path, state: Mapping[str, Any] | None = None) -> str:
    """Validate LKG, including the state-first rollback copy when necessary."""

    if state is not None and state.get("status") == "rolled_back":
        try:
            return _policy(root, state.get("lkg_policy_id"))
        except (RolloutError, store.DistillationStoreError):
            pass
    try:
        return _pointer_policy(root, "lkg")
    except RolloutError:
        if state is not None:
            try:
                return _policy(root, state.get("lkg_policy_id"))
            except (RolloutError, store.DistillationStoreError):
                pass
        raise


def _state(root: Path) -> dict[str, Any]:
    try:
        value = store.read_sealed(
            store.distillation_dir(root) / store.STATE_FILE,
            schema=store.DISTILLATION_SCHEMA,
        )
    except store.DistillationStoreError as exc:
        raise RolloutError("invalid rollout state") from exc
    if value.get("kind") != "worker-state":
        raise RolloutError("invalid rollout state kind")
    return value


def _write_state(root: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    return store.write_sealed_state(
        store.distillation_dir(root) / store.STATE_FILE,
        {"kind": "worker-state", **payload},
    )


def _merged_state(state: Mapping[str, Any], **updates: Any) -> dict[str, Any]:
    current = {
        key: value
        for key, value in state.items()
        if key not in {"schema", "namespace", "seal_sha256"}
    }
    return {**current, **updates}


def _enabled(root: Path) -> bool:
    override = os.environ.get("CHRONOVISOR_RECALL_DISTILLATION", "").strip().lower()
    if override:
        return override in {"1", "true", "yes", "on"}
    try:
        from chronovisor.recall.recall_distillation import distillation_enabled

        return distillation_enabled(root / "config.toml")
    except Exception:
        return False


def _now(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, ValueError) as exc:
        raise RolloutError("rollout timestamp is invalid") from exc
    if parsed.tzinfo is None:
        raise RolloutError("rollout timestamp has no timezone")
    return parsed.astimezone(UTC)


def _stage_days(state: Mapping[str, Any], now: datetime) -> int:
    started = state.get("stage_started_at")
    if not isinstance(started, str):
        raise RolloutError("rollout stage start is missing")
    elapsed = (now - _now(started)).total_seconds()
    if elapsed < 0:
        raise RolloutError("rollout clock moved backwards")
    return int(elapsed // 86_400)


def _baseline(root: Path, evaluation: Mapping[str, Any]) -> None:
    baseline_id = evaluation["baseline_id"]
    try:
        artifact = store.read_sealed(
            store.distillation_dir(root) / "baselines" / f"{baseline_id}.json",
            schema="chronovisor.recall-distill-baseline.v1",
        )
    except store.DistillationStoreError as exc:
        raise RolloutError("evaluation baseline is invalid") from exc
    hard_floor = artifact.get("hard_floor")
    offline_gate = artifact.get("offline_training_gate")
    if (
        artifact.get("artifact_id") != baseline_id
        or artifact.get("raw_watermark") != evaluation["raw_watermark"]
        or not isinstance(hard_floor, Mapping)
        or hard_floor.get("p5_allowed") is not True
        or not isinstance(offline_gate, Mapping)
        or canonical_json_sha256_strict(offline_gate)
        != evaluation["offline_gate_sha256"]
    ):
        raise RolloutError("evaluation baseline is not P5 eligible")


def _gate(value: object, *, observation_required: bool, observed_days: int) -> str:
    """Return pass/hold/fail from a closed numeric lower-bound gate."""

    if not isinstance(value, Mapping) or set(value) != {
        "denominator",
        "min_denominator",
        "min_days",
        "ci_lower",
        "min_ci_lower",
    }:
        raise RolloutError("gate schema is not closed")
    denominator = value["denominator"]
    minimum = value["min_denominator"]
    min_days = value["min_days"]
    lower = value["ci_lower"]
    min_lower = value["min_ci_lower"]
    if (
        isinstance(denominator, bool)
        or isinstance(minimum, bool)
        or isinstance(min_days, bool)
        or not isinstance(denominator, int)
        or not isinstance(minimum, int)
        or not isinstance(min_days, int)
        or not isinstance(lower, (int, float))
        or not isinstance(min_lower, (int, float))
        or denominator < 0
        or minimum < 1
        or min_days < 0
        or not math.isfinite(float(lower))
        or not math.isfinite(float(min_lower))
        or not 0.0 <= float(lower) <= 1.0
        or not 0.0 <= float(min_lower) <= 1.0
    ):
        raise RolloutError("gate values are invalid")
    if observation_required and min_days < 7:
        raise RolloutError("observation gate minimum must be at least seven days")
    if denominator < minimum or (observation_required and observed_days < min_days):
        return "hold"
    return "pass" if float(lower) >= float(min_lower) else "fail"


_METRICS = (
    "coverage_abstain",
    "latency_timeout",
    "cohort_delta",
    "feature_parity",
)


def _metrics_gate(
    value: object, *, observation_required: bool, observed_days: int
) -> str:
    if not isinstance(value, Mapping) or set(value) != set(_METRICS):
        raise RolloutError("named rollout metrics are incomplete")
    results = [
        _gate(
            value[name],
            observation_required=observation_required,
            observed_days=observed_days,
        )
        for name in _METRICS
    ]
    return "fail" if "fail" in results else "hold" if "hold" in results else "pass"


def _evaluation(
    root: Path, evaluation: Mapping[str, Any]
) -> tuple[str, str, dict[str, Any]]:
    if set(evaluation) != {"run_id", "evaluation_artifact_id"}:
        raise RolloutError("evaluation reference schema is not closed")
    run_id = evaluation.get("run_id")
    artifact_id = evaluation.get("evaluation_artifact_id")
    if not isinstance(run_id, str) or _HEX.fullmatch(run_id) is None:
        raise RolloutError("evaluation run id is invalid")
    if not isinstance(artifact_id, str) or _HEX.fullmatch(artifact_id) is None:
        raise RolloutError("evaluation artifact id is invalid")
    try:
        artifact = store.read_sealed(
            store.distillation_dir(root) / "evaluations" / f"{artifact_id}.json",
            schema=EVALUATION_SCHEMA,
        )
    except store.DistillationStoreError as exc:
        raise RolloutError("evaluation artifact is invalid") from exc
    if artifact.get("artifact_id") != artifact_id or artifact.get("run_id") != run_id:
        raise RolloutError("evaluation artifact identity mismatch")
    required = {
        "schema",
        "namespace",
        "artifact_id",
        "seal_sha256",
        "kind",
        "run_id",
        "policy_id",
        "baseline_id",
        "raw_watermark",
        "incumbent_policy_id",
        "split_sha256",
        "feature_revision",
        "feature_parity_sha256",
        "offline_gate_sha256",
        "observation_mode",
        "replay_metrics",
        "shadow_metrics",
        "canary_metrics",
    }
    if set(artifact) != required:
        raise RolloutError("evaluation schema is not closed")
    policy_id = artifact.get("policy_id")
    if not isinstance(policy_id, str) or _HEX.fullmatch(policy_id) is None:
        raise RolloutError("evaluation policy id is invalid")
    for key in (
        "baseline_id",
        "raw_watermark",
        "incumbent_policy_id",
        "split_sha256",
        "feature_parity_sha256",
        "offline_gate_sha256",
    ):
        if not isinstance(artifact[key], str) or _HEX.fullmatch(artifact[key]) is None:
            raise RolloutError(f"evaluation {key} is invalid")
    revision = artifact["feature_revision"]
    if not isinstance(revision, str) or _REVISION.fullmatch(revision) is None:
        raise RolloutError("evaluation feature revision is invalid")
    if artifact["observation_mode"] not in {
        "paired",
        "candidate_only_legacy_incumbent",
    }:
        raise RolloutError("evaluation observation mode is invalid")
    for gate, observed in (
        (artifact["replay_metrics"], 0),
        (artifact["shadow_metrics"], 7),
        (artifact["canary_metrics"], 7),
    ):
        _metrics_gate(gate, observation_required=observed > 0, observed_days=observed)
    receipt = {
        "kind": "numeric-rollout-evaluation",
        "run_id": run_id,
        "policy_id": policy_id,
        "evaluation_sha256": canonical_json_sha256_strict(artifact),
        "evaluation_artifact_id": artifact_id,
        "baseline_id": artifact["baseline_id"],
        "raw_watermark": artifact["raw_watermark"],
        "incumbent_policy_id": artifact["incumbent_policy_id"],
        "split_sha256": artifact["split_sha256"],
        "feature_revision": revision,
        "feature_parity_sha256": artifact["feature_parity_sha256"],
        "offline_gate_sha256": artifact["offline_gate_sha256"],
    }
    return run_id, policy_id, {**receipt, "metrics": artifact}


def _result(state: Mapping[str, Any], *, changed: bool) -> dict[str, Any]:
    return {
        "status": str(state.get("status") or "unknown"),
        "rollout_percent": int(state.get("rollout_percent") or 0),
        "learning_halted": bool(state.get("learning_halted", False)),
        "last_run_id": str(state.get("last_run_id") or ""),
        "changed": changed,
    }


def _hold(
    root: Path,
    state: Mapping[str, Any],
    run_id: str,
    receipt_id: str,
    reason: str,
) -> dict[str, Any]:
    updated = _write_state(
        root,
        {
            **_merged_state(state),
            "status": str(state.get("status") or "shadow"),
            "last_run_id": run_id,
            "evaluation_receipt_id": receipt_id,
            "hold_reason": reason,
            "learning_halted": False,
        },
    )
    return _result(updated, changed=True)


def _advance(
    root: Path,
    state: Mapping[str, Any],
    run_id: str,
    policy_id: str,
    status: str,
    percent: int,
    receipt_id: str,
    now: datetime,
    *,
    lkg_policy_id: str | None = None,
) -> dict[str, Any]:
    updated = _write_state(
        root,
        _merged_state(
            state,
            **{
                "status": status,
                "rollout_percent": percent,
                "candidate_policy_id": policy_id,
                "lkg_policy_id": lkg_policy_id or _lkg(root, state),
                "last_run_id": run_id,
                "stage_run_id": run_id,
                "evaluation_receipt_id": receipt_id,
                "stage_started_at": now.isoformat().replace("+00:00", "Z"),
                "hold_reason": "",
                "learning_halted": False,
                "error_code": "",
            },
        ),
    )
    return _result(updated, changed=True)


def _rollback_locked(
    root: Path, state: Mapping[str, Any], run_id: str, reason: str
) -> dict[str, Any]:
    lkg = _lkg(root, state)
    candidate_id = str(state.get("candidate_policy_id") or "")
    try:
        candidate_id = _pointer_policy(root, "candidate")
    except RolloutError:
        pass
    quarantine_id, _, _ = store.write_immutable(
        store.distillation_dir(root) / "quarantine",
        {
            "kind": "rollout_rollback",
            "run_id": run_id,
            "candidate_policy_id": candidate_id,
            "reason": reason,
        },
        schema=QUARANTINE_SCHEMA,
    )
    # This is intentionally before pointer writes: serving consults this sealed
    # state first, so even a filesystem failure cannot keep the candidate live.
    updated = _write_state(
        root,
        _merged_state(
            state,
            **{
                "status": "rolled_back",
                "rollout_percent": 0,
                "lkg_policy_id": lkg,
                "last_run_id": run_id,
                "quarantine_id": quarantine_id,
                "learning_halted": True,
                "error_code": reason,
            },
        ),
    )
    try:
        store.write_pointer(root, "active", lkg, rollback_run_id=run_id)
        store.clear_pointer(root, "candidate")
    except (OSError, store.DistillationStoreError):
        pass
    return _result(updated, changed=True)


def evaluate_and_advance(
    root: Path | None, now: str, evaluation: Mapping[str, Any]
) -> dict[str, Any]:
    """Advance only sealed replay/shadow/canary gates; otherwise keep serving LKG."""

    chronovisor_root = _root(root)
    lock = store.distillation_dir(chronovisor_root) / "rollout.lock"
    with store._locked(lock):
        state = _state(chronovisor_root)
        if not _enabled(chronovisor_root):
            return _result(state, changed=False)
        timestamp = _now(now)
        run_id, policy_id, receipt = _evaluation(chronovisor_root, evaluation)
        if state.get("last_run_id") == run_id:
            receipt_id = state.get("evaluation_receipt_id")
            try:
                prior = store.read_sealed(
                    store.distillation_dir(chronovisor_root)
                    / "rollout-runs"
                    / f"{receipt_id}.json",
                    schema=EVALUATION_SCHEMA,
                )
            except store.DistillationStoreError as exc:
                raise RolloutError("prior run receipt is invalid") from exc
            if prior.get("evaluation_sha256") != receipt["evaluation_sha256"]:
                raise RolloutError("run id was reused with different evidence")
            return _result(state, changed=False)
        if state.get("learning_halted"):
            return _result(state, changed=False)
        candidate_id = _pointer_policy(chronovisor_root, "candidate")
        if candidate_id != policy_id:
            raise RolloutError("evaluation does not match sealed candidate")
        if str(state.get("status") or "") == "capture_only":
            raise RolloutError("capture-only state cannot advance")
        candidate_artifact = store.read_sealed(
            store.distillation_dir(chronovisor_root) / "policies" / f"{policy_id}.json",
            schema=POLICY_SCHEMA,
        )
        bound = receipt["metrics"]
        if candidate_artifact.get("feature_revision") != bound["feature_revision"]:
            raise RolloutError("evaluation feature revision does not match policy")
        if bound["incumbent_policy_id"] != _pointer_policy(chronovisor_root, "active"):
            raise RolloutError("evaluation incumbent policy does not match active")
        _baseline(chronovisor_root, bound)
        status = str(state.get("status") or "")
        percent = int(state.get("rollout_percent") or 0)
        if bound["observation_mode"] == "candidate_only_legacy_incumbent":
            incumbent = store.read_sealed(
                store.distillation_dir(chronovisor_root)
                / "policies"
                / f"{bound['incumbent_policy_id']}.json",
                schema=POLICY_SCHEMA,
            )
            if status != "canary" or percent != 100 or incumbent.get(
                "serve_mode"
            ) != "legacy":
                raise RolloutError("candidate-only observation mode is not allowed")
        replay = _metrics_gate(
            bound["replay_metrics"], observation_required=False, observed_days=0
        )
        receipt_id, _, _ = store.write_immutable(
            store.distillation_dir(chronovisor_root) / "rollout-runs",
            receipt,
            schema=EVALUATION_SCHEMA,
            artifact_id=run_id,
        )
        if status in {"ready", "replay", "capture_only"}:
            if replay == "hold":
                return _hold(
                    chronovisor_root, state, run_id, receipt_id, "replay_insufficient"
                )
            if replay == "fail":
                return _rollback_locked(
                    chronovisor_root, state, run_id, "replay_ci_failed"
                )
            return _advance(
                chronovisor_root,
                state,
                run_id,
                policy_id,
                "shadow",
                0,
                receipt_id,
                timestamp,
            )
        if status == "shadow":
            if not state.get("evaluation_receipt_id"):
                if replay == "hold":
                    return _hold(
                        chronovisor_root,
                        state,
                        run_id,
                        receipt_id,
                        "replay_insufficient",
                    )
                if replay == "fail":
                    return _rollback_locked(
                        chronovisor_root, state, run_id, "replay_ci_failed"
                    )
                return _advance(
                    chronovisor_root,
                    state,
                    run_id,
                    policy_id,
                    "shadow",
                    0,
                    receipt_id,
                    timestamp,
                )
            shadow = _metrics_gate(
                bound["shadow_metrics"],
                observation_required=True,
                observed_days=_stage_days(state, timestamp),
            )
            if shadow == "hold":
                return _hold(
                    chronovisor_root, state, run_id, receipt_id, "shadow_insufficient"
                )
            if shadow == "fail":
                return _rollback_locked(
                    chronovisor_root, state, run_id, "shadow_ci_failed"
                )
            return _advance(
                chronovisor_root,
                state,
                run_id,
                policy_id,
                "canary",
                5,
                receipt_id,
                timestamp,
            )
        if status == "canary" and percent in _STAGES:
            observation_mode = bound["observation_mode"]
            if observation_mode == "candidate_only_legacy_incumbent":
                results = [
                    _gate(
                        bound["canary_metrics"][name],
                        observation_required=True,
                        observed_days=_stage_days(state, timestamp),
                    )
                    for name in ("latency_timeout", "feature_parity")
                ]
                canary = (
                    "fail" if "fail" in results else "hold" if "hold" in results else "pass"
                )
            else:
                canary = _metrics_gate(
                    bound["canary_metrics"],
                    observation_required=True,
                    observed_days=_stage_days(state, timestamp),
                )
            if canary == "hold":
                return _hold(
                    chronovisor_root, state, run_id, receipt_id, "canary_insufficient"
                )
            if canary == "fail":
                return _rollback_locked(
                    chronovisor_root, state, run_id, "canary_ci_failed"
                )
            if percent == 100:
                lkg = _lkg(chronovisor_root, state)
                staged = _write_state(
                    chronovisor_root,
                    _merged_state(
                        state,
                        **{
                            "status": "adopting",
                            "rollout_percent": 0,
                            "lkg_policy_id": lkg,
                            "last_run_id": run_id,
                            "evaluation_receipt_id": receipt_id,
                            "learning_halted": True,
                            "error_code": "adoption_incomplete",
                        },
                    ),
                )
                try:
                    store.write_pointer(
                        chronovisor_root, "active", policy_id, adopted_from=lkg
                    )
                    store.write_pointer(
                        chronovisor_root, "lkg", policy_id, adopted=True
                    )
                    store.clear_pointer(chronovisor_root, "candidate")
                except (OSError, store.DistillationStoreError):
                    return _result(staged, changed=True)
                return _advance(
                    chronovisor_root,
                    state,
                    run_id,
                    policy_id,
                    "active",
                    100,
                    receipt_id,
                    timestamp,
                    lkg_policy_id=policy_id,
                )
            return _advance(
                chronovisor_root,
                state,
                run_id,
                policy_id,
                "canary",
                _STAGES[_STAGES.index(percent) + 1],
                receipt_id,
                timestamp,
            )
        return _hold(chronovisor_root, state, run_id, receipt_id, "invalid_state")


def rollback_to_lkg(root: Path | None, run_id: str, reason: str) -> dict[str, Any]:
    """Halt learning state-first, then best-effort restore the active pointer."""

    if not isinstance(run_id, str) or _HEX.fullmatch(run_id) is None:
        raise RolloutError("rollback run id is invalid")
    if not isinstance(reason, str) or not re.fullmatch(r"[a-z0-9_:-]{1,80}", reason):
        raise RolloutError("rollback reason is invalid")
    chronovisor_root = _root(root)
    lock = store.distillation_dir(chronovisor_root) / "rollout.lock"
    with store._locked(lock):
        state = _state(chronovisor_root)
        return _rollback_locked(chronovisor_root, state, run_id, reason)


def select_policy_id(root: Path | None, session_id: str) -> str:
    """Choose the candidate only during sealed canaries; otherwise use validated LKG."""

    if not isinstance(session_id, str) or not session_id:
        return ""
    chronovisor_root = _root(root)
    if not _enabled(chronovisor_root):
        return ""
    try:
        state = _state(chronovisor_root)
        lkg = _lkg(chronovisor_root, state)
        if state.get("learning_halted") or state.get("status") in {
            "rolled_back",
            "quarantined",
        }:
            return lkg
        status = str(state.get("status") or "")
        percent = int(state.get("rollout_percent") or 0)
        if status == "active":
            try:
                return _pointer_policy(chronovisor_root, "active")
            except RolloutError:
                return lkg
        if status != "canary" or percent not in _STAGES:
            return lkg
        candidate = _pointer_policy(chronovisor_root, "candidate")
        bucket = (
            int.from_bytes(
                hashlib.sha256(
                    f"recall-distill-rollout-v2\0{session_id}".encode()
                ).digest()[:8],
                "big",
            )
            % 10_000
        )
        return candidate if bucket < percent * 100 else lkg
    except (RolloutError, store.DistillationStoreError, ValueError):
        try:
            return _pointer_policy(chronovisor_root, "lkg")
        except RolloutError:
            return ""
