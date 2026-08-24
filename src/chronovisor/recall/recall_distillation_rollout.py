"""Sealed, fail-closed rollout state for distilled Recall policies."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from chronovisor.core.canonical_json import (
    canonical_json_bytes_strict,
    canonical_json_sha256_strict,
)
from chronovisor.core.store import CHRONOVISOR_ROOT
from chronovisor.recall import recall_distillation_store as store
from chronovisor.recall.recall_distillation import (
    _operational_rollout_metrics as _PRODUCER_OPERATIONAL_METRICS,
)
from chronovisor.recall.recall_distillation import (
    _operational_rollout_source_ids as _PRODUCER_OPERATIONAL_SOURCE_IDS,
)
from chronovisor.recall.recall_distillation import (
    _shadow_replay_source_fields as _PRODUCER_SHADOW_SOURCE_FIELDS,
)
from chronovisor.recall.recall_distillation import (
    shadow_observation_hashes as _PRODUCER_SHADOW_HASHES,
)

POLICY_SCHEMA = "chronovisor.recall-distill-policy.v2"
EVALUATION_SCHEMA = "chronovisor.recall-distill-rollout-evaluation.v2"
REPLAY_OBSERVATION_SCHEMA = "chronovisor.recall-rollout-replay-observation.v1"
QUARANTINE_SCHEMA = "chronovisor.recall-distill-quarantine.v1"
_STAGES = (5, 25, 100)
_REPLAY_MIN_DENOMINATOR = 500
_DAY_SECONDS = 86_400
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_REVISION = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}\Z")
_REPLAY_REGISTRY_PAYLOAD_KEYS = frozenset(
    {
        "kind",
        "pair_id",
        "observation_artifact_id",
        "run_id",
        "stage",
        "cohort",
        "candidate_policy_id",
        "baseline_policy_id",
        "baseline_artifact_id",
        "split_sha256",
        "row_id",
        "shadow_observation_artifact_id",
        "source_set_sha256",
    }
)


class RolloutError(ValueError):
    """A rollout input or artifact is not safe to use."""


def _root(root: Path | None) -> Path:
    return root or CHRONOVISOR_ROOT


def _policy(root: Path, policy_id: object) -> str:
    if not isinstance(policy_id, str) or _HEX.fullmatch(policy_id) is None:
        raise RolloutError("invalid policy id")
    artifact = _stable_sealed(
        store.distillation_dir(root) / "policies" / f"{policy_id}.json",
        base=store.distillation_dir(root),
        schema=POLICY_SCHEMA,
        label="policy artifact",
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


def _stable_pointer(root: Path, kind: str) -> dict[str, Any]:
    try:
        filename = store.POINTER_FILES[kind]
    except KeyError as exc:
        raise RolloutError("invalid pointer kind") from exc
    return _stable_sealed(
        store.distillation_dir(root) / filename,
        base=store.distillation_dir(root),
        schema=store.DISTILLATION_SCHEMA,
        label=f"{kind} pointer",
    )


def _pointer_policy(root: Path, kind: str) -> str:
    try:
        return _policy(root, _stable_pointer(root, kind)["policy_id"])
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
        value = _stable_sealed(
            store.distillation_dir(root) / store.STATE_FILE,
            base=store.distillation_dir(root),
            schema=store.DISTILLATION_SCHEMA,
            label="rollout state",
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


def _strict_utc(value: object, label: str) -> datetime:
    if not isinstance(value, str):
        raise RolloutError(f"{label} is invalid")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise RolloutError(f"{label} is invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        raise RolloutError(f"{label} is invalid")
    return parsed.astimezone(UTC)


def _immutable_artifact_id(artifact: Mapping[str, Any]) -> str:
    """Return the content identity used by ``store.write_immutable``."""

    unsigned = {
        key: value
        for key, value in artifact.items()
        if key not in {"artifact_id", "seal_sha256"}
    }
    return canonical_json_sha256_strict(unsigned)


def _path_identity(path: Path) -> tuple[int, int, int, int, int]:
    info = os.lstat(path)
    return (info.st_dev, info.st_ino, info.st_mode, info.st_size, info.st_mtime_ns)


def _stable_sealed(
    path: Path, *, base: Path, schema: str, label: str
) -> dict[str, Any]:
    """Read one sealed artifact only if its path and bytes stay unchanged."""

    try:
        for parent in (base.parent.parent, base.parent, base):
            if parent.exists() and stat.S_ISLNK(os.lstat(parent).st_mode):
                raise OSError("artifact path component is a symlink")
        base_identity = _path_identity(base)
        if not stat.S_ISDIR(base_identity[2]):
            raise OSError("artifact root is not a directory")
        relative = path.relative_to(base)
        directories = [base]
        current = base
        for part in relative.parts[:-1]:
            current /= part
            identity = _path_identity(current)
            if not stat.S_ISDIR(identity[2]):
                raise OSError("artifact parent is not a directory")
            directories.append(current)
        directory_identities = [_path_identity(directory) for directory in directories]
        file_identity = _path_identity(path)
        if not stat.S_ISREG(file_identity[2]):
            raise OSError("artifact is not a regular file")
        raw = path.read_bytes()
        artifact = store.verify_seal(json.loads(raw), schema=schema)
        if raw != canonical_json_bytes_strict(artifact) + b"\n":
            raise OSError("artifact bytes are not canonical")
        if raw != path.read_bytes():
            raise OSError("artifact bytes changed while reading")
        if _path_identity(base) != base_identity or _path_identity(path) != file_identity:
            raise OSError("artifact path changed while reading")
        if any(
            _path_identity(directory) != identity
            for directory, identity in zip(
                directories, directory_identities, strict=True
            )
        ):
            raise OSError("artifact parent changed while reading")
        return artifact
    except (OSError, TypeError, ValueError, UnicodeError, store.DistillationStoreError) as exc:
        raise store.DistillationStoreError(f"{label} is not stable") from exc


def _stage_days(state: Mapping[str, Any], now: datetime) -> int:
    started = state.get("stage_started_at")
    if not isinstance(started, str):
        raise RolloutError("rollout stage start is missing")
    elapsed = (now - _now(started)).total_seconds()
    if elapsed < 0:
        raise RolloutError("rollout clock moved backwards")
    return int(elapsed // 86_400)


def _baseline(root: Path, evaluation: Mapping[str, Any]) -> None:
    baseline_id = evaluation["baseline_artifact_id"]
    try:
        artifact = _stable_sealed(
            store.distillation_dir(root) / "baselines" / f"{baseline_id}.json",
            base=store.distillation_dir(root),
            schema="chronovisor.recall-distill-baseline.v1",
            label="evaluation baseline artifact",
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


_LOCKED_REPLAY_KEYS = {
    "schema",
    "namespace",
    "artifact_id",
    "seal_sha256",
    "kind",
    "run_id",
    "stage",
    "cohort",
    "baseline_artifact_id",
    "baseline_policy_id",
    "candidate_policy_id",
    "training_snapshot_id",
    "training_rows",
    "training_rows_sha256",
    "policy_sha256",
    "candidate_head",
    "profile_contract_id",
    "offline_gate_sha256",
    "model_cohort_sha256",
    "split_revision",
}
_LOCKED_REPLAY_ROW_KEYS = {
    "row_id",
    "rally_id",
    "candidate_id",
    "as_of",
    "split",
    "split_role",
    "stage",
    "cohort",
    "candidate_policy_id",
    "baseline_policy_id",
    "baseline_artifact_id",
    "pair_id",
    "shadow_observation_artifact_id",
    "shadow_receipt_record_sha256",
    "qualified_run_id",
    "candidate_decision_sha256",
    "baseline_decision_sha256",
    "candidate_pool_sha256",
    "baseline_pool_sha256",
    "candidate_feature_snapshot_sha256",
    "baseline_feature_snapshot_sha256",
    "candidate_feature_bytes_sha256",
    "baseline_feature_bytes_sha256",
    "feature_snapshot_sha256",
    "runtime_observation_sha256",
    "operational_evidence_sha256",
}
_REPLAY_OBSERVATION_KEYS = {
    "schema",
    "namespace",
    "artifact_id",
    "seal_sha256",
    "kind",
    "run_id",
    "stage",
    "cohort",
    "candidate_policy_id",
    "baseline_policy_id",
    "baseline_artifact_id",
    "split_sha256",
    "first_observed_at",
    "last_observed_at",
    "pair_count",
    "pairs_sha256",
    "pairs",
}
_REPLAY_PAIR_KEYS = {
    "pair_id",
    "row_id",
    "observed_at",
    "run_id",
    "stage",
    "cohort",
    "candidate_policy_id",
    "baseline_policy_id",
    "baseline_artifact_id",
    "split_sha256",
    "shadow_observation_artifact_id",
    "shadow_receipt_record_sha256",
    "qualified_run_id",
    "baseline_decision_sha256",
    "candidate_decision_sha256",
    "candidate_pool_sha256",
    "baseline_pool_sha256",
    "candidate_feature_snapshot_sha256",
    "baseline_feature_snapshot_sha256",
    "candidate_feature_bytes_sha256",
    "baseline_feature_bytes_sha256",
    "feature_snapshot_sha256",
    "runtime_observation_sha256",
    "operational_evidence_sha256",
}
_SHADOW_EVIDENCE_KEYS = {
    "candidate_quality",
    "baseline_quality",
    "candidate_covered",
    "baseline_covered",
    "candidate_anchor_retained",
    "baseline_anchor_retained",
    "candidate_abstained",
    "baseline_abstained",
    "candidate_score_ms",
    "live_latency_ms",
    "resource_ok",
    "integrity_ok",
    "negative_veto",
    "deadline_ms",
    "producer",
    "stage",
    "run_id",
    "cohort",
    "host",
    "pair_id",
    "candidate_decision_sha256",
    "baseline_decision_sha256",
    "candidate_pool_sha256",
    "baseline_pool_sha256",
    "candidate_feature_snapshot_sha256",
    "baseline_feature_snapshot_sha256",
    "candidate_feature_bytes_sha256",
    "baseline_feature_bytes_sha256",
    "feature_snapshot_sha256",
    "feature_parity",
}
_SHADOW_RECEIPT_BINDING_KEYS = {
    "decision_id",
    "host",
    "session_id_sha256",
    "query_semantic_sha256",
    "policy_id",
    "incumbent_policy_id",
    "served_policy_id",
    "stage",
    "stage_started_at",
    "qualified_run_id",
    "run_id",
    "cohort",
    "baseline_artifact_id",
    "candidate_policy_id",
    "baseline_policy_id",
    "row_id",
    "rally_id",
    "candidate_id",
    "as_of",
    "split",
    "split_role",
    "selected_candidate_ids",
    "incumbent_selected_candidate_ids",
    "paired_eligible",
    "candidate_pool_sha256",
    "candidate_feature_snapshot_sha256",
    "candidate_decision_sha256",
    "baseline_decision_sha256",
    "baseline_pool_sha256",
    "baseline_feature_snapshot_sha256",
    "candidate_feature_bytes_sha256",
    "baseline_feature_bytes_sha256",
    "feature_snapshot_sha256",
    "pair_id",
    "runtime_observation_sha256",
    "operational_evidence_sha256",
    "observed_at",
}
_SHADOW_RECEIPT_KEYS = {
    "schema",
    "namespace",
    "previous_sha256",
    "record_sha256",
    "kind",
    "shadow_observation_artifact_id",
    "binding_sha256",
    "idempotency_sha256",
    *_SHADOW_RECEIPT_BINDING_KEYS,
}


def _artifact_directory(root: Path, name: str) -> Path:
    base = store.distillation_dir(root)
    try:
        for parent in (base.parent.parent, base.parent, base):
            if parent.exists() and stat.S_ISLNK(os.lstat(parent).st_mode):
                raise OSError("artifact path component is a symlink")
        if not base.exists():
            base.mkdir(parents=True, exist_ok=True)
        if not stat.S_ISDIR(_path_identity(base)[2]):
            raise OSError("artifact root is not a directory")
        directory = base / name
        if directory.exists() and not stat.S_ISDIR(_path_identity(directory)[2]):
            raise OSError("artifact directory is not a directory")
        directory.mkdir(parents=True, exist_ok=True)
        if not stat.S_ISDIR(_path_identity(directory)[2]):
            raise OSError("artifact directory changed while creating")
        return directory
    except OSError as exc:
        raise RolloutError(f"{name} artifact directory is unsafe") from exc


def _require_hex(value: object, label: str) -> str:
    if not isinstance(value, str) or _HEX.fullmatch(value) is None:
        raise RolloutError(f"{label} is invalid")
    return value


def _require_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 240:
        raise RolloutError(f"{label} is invalid")
    return value


def _nonnegative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _replay_context(
    *,
    run_id: object,
    stage: object,
    cohort: object,
    candidate_policy_id: object,
    baseline_policy_id: object,
    baseline_artifact_id: object,
) -> dict[str, str]:
    candidate = _require_hex(candidate_policy_id, "candidate policy id")
    baseline = _require_hex(baseline_policy_id, "baseline policy id")
    if candidate == baseline:
        raise RolloutError("candidate and baseline policy identities must differ")
    return {
        "run_id": _require_hex(run_id, "replay run id"),
        "stage": _require_text(stage, "replay stage"),
        "cohort": _require_text(cohort, "replay cohort"),
        "candidate_policy_id": candidate,
        "baseline_policy_id": baseline,
        "baseline_artifact_id": _require_hex(
            baseline_artifact_id, "baseline artifact id"
        ),
    }


def _shadow_observation(root: Path, artifact_id: str) -> dict[str, Any]:
    _require_hex(artifact_id, "shadow observation artifact id")
    path = (
        store.distillation_dir(root)
        / "shadow-observations"
        / f"{artifact_id}.json"
    )
    try:
        artifact = _stable_sealed(
            path,
            base=store.distillation_dir(root),
            schema="chronovisor.recall-distill-shadow-observation.v1",
            label="shadow observation artifact",
        )
    except store.DistillationStoreError as exc:
        raise RolloutError("shadow observation artifact is invalid") from exc
    if artifact.get("artifact_id") != artifact_id:
        raise RolloutError("shadow observation identity mismatch")
    if _immutable_artifact_id(artifact) != artifact_id:
        raise RolloutError("shadow observation content identity mismatch")
    if artifact.get("kind") != "non-causal-shadow-observation":
        raise RolloutError("shadow observation kind is invalid")
    return artifact


def _stable_chain_head(root: Path, filename: str) -> str:
    """Read a ledger head only from one canonical, identity-stable file."""

    base = store.distillation_dir(root)
    path = base / filename
    lock = path.with_suffix(path.suffix + ".lock")
    try:
        for parent in (base.parent.parent, base.parent, base):
            if parent.exists() and stat.S_ISLNK(os.lstat(parent).st_mode):
                raise OSError("ledger path component is a symlink")
        base_identity = _path_identity(base)
        if not stat.S_ISDIR(base_identity[2]):
            raise OSError("ledger root is not a directory")
        for candidate in (path, lock):
            if candidate.exists() and stat.S_ISLNK(os.lstat(candidate).st_mode):
                raise OSError("ledger path is a symlink")
        before = _path_identity(path) if path.exists() else None
        rows = store.read_chain(path)
        raw = path.read_bytes() if path.exists() else b""
        if any(
            line != canonical_json_bytes_strict(row)
            for line, row in zip(raw.splitlines(), rows, strict=True)
        ):
            raise OSError("ledger bytes are not canonical")
        after = _path_identity(path) if path.exists() else None
        if (
            _path_identity(base)[:3] != base_identity[:3]
            or before != after
            or (raw and raw != path.read_bytes())
        ):
            raise OSError("ledger changed while reading")
    except (OSError, store.DistillationStoreError) as exc:
        raise RolloutError("ledger is not stable") from exc
    if not rows:
        return ""
    head = rows[-1].get("record_sha256")
    return _require_hex(head, "ledger head")


def _stable_chain_rows(root: Path, filename: str) -> list[Mapping[str, Any]]:
    """Read one canonical append-only ledger with path identity checks."""

    base = store.distillation_dir(root)
    path = base / filename
    try:
        for parent in (base.parent.parent, base.parent, base):
            if parent.exists() and stat.S_ISLNK(os.lstat(parent).st_mode):
                raise OSError("ledger path component is a symlink")
        base_identity = _path_identity(base)
        if not stat.S_ISDIR(base_identity[2]):
            raise OSError("ledger root is not a directory")
        if not path.exists():
            return []
        if path.exists() and stat.S_ISLNK(os.lstat(path).st_mode):
            raise OSError("ledger is a symlink")
        before = _path_identity(path) if path.exists() else None
        rows = store.read_chain(path)
        raw = path.read_bytes() if path.exists() else b""
        if any(
            line != canonical_json_bytes_strict(row)
            for line, row in zip(raw.splitlines(), rows, strict=True)
        ):
            raise OSError("ledger bytes are not canonical")
        after = _path_identity(path) if path.exists() else None
        if (
            _path_identity(base) != base_identity
            or before != after
            or (raw and raw != path.read_bytes())
        ):
            raise OSError("ledger changed while reading")
    except (OSError, store.DistillationStoreError) as exc:
        raise RolloutError("ledger is not stable") from exc
    return cast(list[Mapping[str, Any]], rows)


def _shadow_receipt_index(root: Path) -> dict[str, Mapping[str, Any]]:
    try:
        rows = _stable_chain_rows(root, "shadow-observation-receipts.jsonl")
    except RolloutError as exc:
        raise RolloutError("shadow receipt ledger is invalid") from exc
    index: dict[str, Mapping[str, Any]] = {
        str(row["shadow_observation_artifact_id"]): row
        for row in rows
        if isinstance(row, Mapping)
        and isinstance(row.get("shadow_observation_artifact_id"), str)
    }
    if len(index) != len(rows):
        raise RolloutError("shadow observation receipt identities are not unique")
    seen_decisions: set[object] = set()
    seen_requests: set[tuple[object, object]] = set()
    seen_idempotency: set[object] = set()
    for row in index.values():
        if set(row) != _SHADOW_RECEIPT_KEYS:
            raise RolloutError("shadow receipt schema is not closed")
        if row.get("kind") != "shadow-policy-observation":
            raise RolloutError("shadow receipt kind is invalid")
        _require_text(row.get("decision_id"), "shadow receipt decision id")
        _require_text(row.get("host"), "shadow receipt host")
        _require_text(row.get("stage"), "shadow receipt stage")
        _require_text(row.get("stage_started_at"), "shadow receipt stage start")
        _require_text(row.get("cohort"), "shadow receipt cohort")
        _require_text(row.get("candidate_id"), "shadow receipt candidate id")
        _require_text(row.get("as_of"), "shadow receipt as-of")
        _require_text(row.get("split"), "shadow receipt split")
        _require_text(row.get("split_role"), "shadow receipt split role")
        _require_text(row.get("observed_at"), "shadow receipt observed timestamp")
        for key in (
            "session_id_sha256",
            "query_semantic_sha256",
            "policy_id",
            "incumbent_policy_id",
            "served_policy_id",
            "qualified_run_id",
            "run_id",
            "baseline_artifact_id",
            "candidate_policy_id",
            "baseline_policy_id",
            "row_id",
            "rally_id",
            "candidate_pool_sha256",
            "candidate_feature_snapshot_sha256",
            "candidate_decision_sha256",
            "baseline_decision_sha256",
            "baseline_pool_sha256",
            "baseline_feature_snapshot_sha256",
            "candidate_feature_bytes_sha256",
            "baseline_feature_bytes_sha256",
            "feature_snapshot_sha256",
            "pair_id",
            "runtime_observation_sha256",
            "operational_evidence_sha256",
        ):
            _require_hex(row.get(key), f"shadow receipt {key}")
        for key in ("selected_candidate_ids", "incumbent_selected_candidate_ids"):
            values = row.get(key)
            if (
                not isinstance(values, list)
                or any(
                    not isinstance(value, str) or not value for value in values
                )
                or len(values) != len(set(values))
            ):
                raise RolloutError(f"shadow receipt {key} is invalid")
        if not isinstance(row.get("paired_eligible"), bool):
            raise RolloutError("shadow receipt paired eligibility is invalid")
        _require_hex(
            row.get("shadow_observation_artifact_id"),
            "shadow receipt observation artifact id",
        )
        _require_hex(row.get("record_sha256"), "shadow receipt record id")
        _require_hex(row.get("binding_sha256"), "shadow receipt binding id")
        _require_hex(
            row.get("idempotency_sha256"), "shadow receipt idempotency id"
        )
        unsigned = {
            key: value for key, value in row.items() if key != "record_sha256"
        }
        if row["record_sha256"] != canonical_json_sha256_strict(unsigned):
            raise RolloutError("shadow receipt record hash mismatch")
        binding = {
            key: row[key] for key in _SHADOW_RECEIPT_BINDING_KEYS
        }
        if row["binding_sha256"] != canonical_json_sha256_strict(binding):
            raise RolloutError("shadow receipt binding hash mismatch")
        idempotency = {
            key: value
            for key, value in binding.items()
            if key not in {"observed_at", "as_of", "row_id"}
        }
        if row["idempotency_sha256"] != canonical_json_sha256_strict(idempotency):
            raise RolloutError("shadow receipt idempotency hash mismatch")
        decision_id = row["decision_id"]
        request_id = (row["session_id_sha256"], row["query_semantic_sha256"])
        if decision_id in seen_decisions:
            raise RolloutError("shadow receipt decision identities are not unique")
        if request_id in seen_requests:
            raise RolloutError("shadow receipt request identities are not unique")
        if row["idempotency_sha256"] in seen_idempotency:
            raise RolloutError("shadow receipt idempotency identities are not unique")
        seen_decisions.add(decision_id)
        seen_requests.add(request_id)
        seen_idempotency.add(row["idempotency_sha256"])
    return index


def _shadow_receipt(root: Path, artifact_id: str) -> dict[str, Any]:
    row = _shadow_receipt_index(root).get(artifact_id)
    matches = [row] if row is not None else []
    if len(matches) != 1:
        raise RolloutError("shadow observation receipt binding is missing")
    record_id = matches[0].get("record_sha256")
    _require_hex(record_id, "shadow receipt record id")
    return dict(matches[0])


def _shadow_hashes(
    candidate_feature_snapshot: object,
    baseline_feature_snapshot: object,
    candidate_pool_refs: object,
    baseline_pool_refs: object,
    *,
    selected_candidate_ids: object,
    baseline_selected_candidate_ids: object,
) -> dict[str, str | bool]:
    """Derive hashes through the typed runtime producer's canonical helper."""

    def sequence_of_mappings(value: object) -> bool:
        return isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ) and all(isinstance(row, Mapping) for row in value)

    def sequence_of_strings(value: object) -> bool:
        return isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ) and all(isinstance(item, str) for item in value)

    if not all(
        sequence_of_mappings(value)
        for value in (
            candidate_feature_snapshot,
            baseline_feature_snapshot,
            candidate_pool_refs,
            baseline_pool_refs,
        )
    ) or not sequence_of_strings(selected_candidate_ids) or not sequence_of_strings(
        baseline_selected_candidate_ids
    ):
        raise RolloutError("shadow hash source types are invalid")
    try:
        derived = _PRODUCER_SHADOW_HASHES(
            cast(Sequence[Mapping[str, Any]], candidate_feature_snapshot),
            cast(Sequence[Mapping[str, Any]], baseline_feature_snapshot),
            cast(Sequence[Mapping[str, Any]], candidate_pool_refs),
            cast(Sequence[Mapping[str, Any]], baseline_pool_refs),
            selected_candidate_ids=cast(Sequence[str], selected_candidate_ids),
            baseline_selected_candidate_ids=cast(
                Sequence[str], baseline_selected_candidate_ids
            ),
        )
    except Exception as exc:
        raise RolloutError("shadow hash derivation failed") from exc
    if not isinstance(derived, Mapping):
        raise RolloutError("shadow hash derivation is invalid")
    expected_keys = {
        "candidate_decision_sha256",
        "baseline_decision_sha256",
        "candidate_pool_sha256",
        "baseline_pool_sha256",
        "candidate_feature_snapshot_sha256",
        "baseline_feature_snapshot_sha256",
        "candidate_feature_bytes_sha256",
        "baseline_feature_bytes_sha256",
        "feature_snapshot_sha256",
        "pair_id",
        "feature_parity",
    }
    if set(derived) != expected_keys:
        raise RolloutError("shadow hash derivation schema is invalid")
    for key in expected_keys - {"feature_parity"}:
        _require_hex(derived[key], f"shadow {key}")
    if not isinstance(derived["feature_parity"], bool):
        raise RolloutError("shadow feature parity is invalid")
    return dict(derived)


def _shadow_row(
    root: Path,
    artifact_id: str,
    context: Mapping[str, str],
    *,
    receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    artifact = _shadow_observation(root, artifact_id)
    receipt = receipt or _shadow_receipt(root, artifact_id)
    if set(receipt) != _SHADOW_RECEIPT_KEYS:
        raise RolloutError("shadow receipt schema is not closed")
    _require_hex(receipt.get("record_sha256"), "shadow receipt record id")
    _require_hex(receipt.get("binding_sha256"), "shadow receipt binding id")
    _require_hex(
        receipt.get("idempotency_sha256"), "shadow receipt idempotency id"
    )
    receipt_binding = {
        key: receipt[key] for key in _SHADOW_RECEIPT_BINDING_KEYS
    }
    if receipt["binding_sha256"] != canonical_json_sha256_strict(receipt_binding):
        raise RolloutError("shadow receipt binding hash mismatch")
    receipt_idempotency = {
        key: value
        for key, value in receipt_binding.items()
        if key not in {"observed_at", "as_of", "row_id"}
    }
    if receipt["idempotency_sha256"] != canonical_json_sha256_strict(
        receipt_idempotency
    ):
        raise RolloutError("shadow receipt idempotency hash mismatch")
    unsigned_receipt = {
        key: value for key, value in receipt.items() if key != "record_sha256"
    }
    if receipt["record_sha256"] != canonical_json_sha256_strict(unsigned_receipt):
        raise RolloutError("shadow receipt record hash mismatch")
    run_id = artifact.get("run_id")
    if (
        not isinstance(run_id, str)
        or run_id != context["run_id"]
        or receipt.get("qualified_run_id") != run_id
    ):
        raise RolloutError("shadow observation run binding mismatch")
    if (
        artifact.get("stage") != context["stage"]
        or receipt.get("stage") != context["stage"]
    ):
        raise RolloutError("shadow observation stage binding mismatch")
    if (
        artifact.get("cohort") != context["cohort"]
        or receipt.get("cohort") != context["cohort"]
    ):
        raise RolloutError("shadow observation cohort binding mismatch")
    candidate_policy_id = artifact.get("candidate_policy_id")
    baseline_policy_id = artifact.get("baseline_policy_id")
    baseline_artifact_id = artifact.get("baseline_artifact_id")
    _require_hex(candidate_policy_id, "shadow observation candidate policy id")
    _require_hex(baseline_policy_id, "shadow observation baseline policy id")
    _require_hex(baseline_artifact_id, "shadow observation baseline artifact id")
    if (
        candidate_policy_id != context["candidate_policy_id"]
        or baseline_policy_id != context["baseline_policy_id"]
        or baseline_artifact_id != context["baseline_artifact_id"]
    ):
        raise RolloutError("shadow observation identity binding mismatch")
    # Legacy aliases are retained in the sealed schema for readback, but they
    # must never provide a weaker identity binding than the canonical fields.
    if (
        artifact.get("policy_id") not in {None, candidate_policy_id}
        or artifact.get("incumbent_policy_id") not in {None, baseline_policy_id}
    ):
        raise RolloutError("shadow observation policy alias mismatch")
    receipt_candidate = receipt.get("candidate_policy_id")
    receipt_baseline = receipt.get("baseline_policy_id")
    receipt_baseline_artifact = receipt.get("baseline_artifact_id")
    _require_hex(receipt_candidate, "shadow receipt candidate policy id")
    _require_hex(receipt_baseline, "shadow receipt baseline policy id")
    _require_hex(receipt_baseline_artifact, "shadow receipt baseline artifact id")
    if (
        receipt_candidate != candidate_policy_id
        or receipt_baseline != baseline_policy_id
        or receipt_baseline_artifact != baseline_artifact_id
    ):
        raise RolloutError("shadow receipt identity binding mismatch")
    if (
        receipt.get("policy_id") not in {None, receipt_candidate}
        or receipt.get("incumbent_policy_id") not in {None, receipt_baseline}
    ):
        raise RolloutError("shadow receipt policy alias mismatch")
    if any(receipt.get(key) != artifact.get(key) for key in _SHADOW_RECEIPT_BINDING_KEYS):
        raise RolloutError("shadow receipt binding fields mismatch")
    if receipt.get("shadow_observation_artifact_id") != artifact_id:
        raise RolloutError("shadow receipt artifact binding mismatch")
    if receipt.get("kind") != "shadow-policy-observation":
        raise RolloutError("shadow receipt kind is invalid")
    _strict_utc(artifact.get("observed_at"), "shadow observation timestamp")
    if receipt.get("observed_at") != artifact.get("observed_at"):
        raise RolloutError("shadow receipt timestamp mismatch")
    pool = artifact.get("candidate_pool_refs")
    baseline_pool = artifact.get("baseline_pool_refs")
    features = artifact.get("candidate_feature_snapshot")
    baseline_features = artifact.get("baseline_feature_snapshot")
    selected = artifact.get("selected_candidate_ids")
    baseline_selected = artifact.get("incumbent_selected_candidate_ids")
    runtime = artifact.get("runtime_observation")
    evidence = artifact.get("operational_evidence")
    if (
        not isinstance(pool, list)
        or not isinstance(baseline_pool, list)
        or not isinstance(features, list)
        or not isinstance(baseline_features, list)
        or not isinstance(runtime, Mapping)
        or not isinstance(selected, list)
        or not isinstance(baseline_selected, list)
        or not isinstance(evidence, Mapping)
        or any(
            key not in artifact
            for key in (
                "row_id",
                "rally_id",
                "candidate_id",
                "as_of",
                "split",
                "split_role",
            )
        )
    ):
        raise RolloutError("shadow observation typed source is incomplete")
    if set(evidence) != _SHADOW_EVIDENCE_KEYS:
        raise RolloutError("shadow operational evidence schema is not closed")
    producer = evidence.get("producer")
    if (
        not isinstance(producer, Mapping)
        or set(producer) != {"name", "version", "synthetic_fixture"}
        or producer.get("name") != "chronovisor.recall-runtime"
        or producer.get("version") != 1
        or producer.get("synthetic_fixture") is not False
        or any(
            not isinstance(evidence.get(key), bool)
            for key in (
                "candidate_quality",
                "baseline_quality",
                "candidate_covered",
                "baseline_covered",
                "candidate_anchor_retained",
                "baseline_anchor_retained",
                "candidate_abstained",
                "baseline_abstained",
                "resource_ok",
                "integrity_ok",
                "negative_veto",
                "feature_parity",
            )
        )
        or any(
            not _nonnegative_int(evidence.get(key))
            for key in ("candidate_score_ms", "live_latency_ms", "deadline_ms")
        )
        or evidence.get("stage") != context["stage"]
        or evidence.get("run_id") != context["run_id"]
        or evidence.get("cohort") != context["cohort"]
        or not isinstance(evidence.get("host"), str)
        or not evidence.get("host")
    ):
        raise RolloutError("shadow operational evidence binding is invalid")
    if any(
        not isinstance(row, Mapping)
        or not isinstance(row.get("candidate_id"), str)
        or not row.get("candidate_id")
        or not isinstance(row.get("selected"), bool)
        for rows in (pool, baseline_pool)
        for row in rows
    ) or any(
        not isinstance(row, Mapping)
        or not isinstance(row.get("candidate_id"), str)
        or not row.get("candidate_id")
        for rows in (features, baseline_features)
        for row in rows
    ):
        raise RolloutError("shadow source row schema is invalid")
    candidate_pool_ids = {row["candidate_id"] for row in pool}
    baseline_pool_ids = {row["candidate_id"] for row in baseline_pool}
    if candidate_pool_ids != baseline_pool_ids:
        raise RolloutError("shadow candidate and baseline pools differ")
    if {
        row["candidate_id"] for row in pool if row["selected"]
    } != set(selected):
        raise RolloutError("shadow selected pool does not match decision")
    if {
        row["candidate_id"] for row in baseline_pool if row["selected"]
    } != set(baseline_selected):
        raise RolloutError("shadow baseline selected pool does not match decision")
    if {row["candidate_id"] for row in features} != candidate_pool_ids:
        raise RolloutError("shadow candidate pool and features differ")
    if {row["candidate_id"] for row in baseline_features} != baseline_pool_ids:
        raise RolloutError("shadow baseline pool and features differ")
    if not isinstance(artifact.get("paired_eligible"), bool):
        raise RolloutError("shadow paired eligibility is invalid")
    if receipt.get("paired_eligible") != artifact.get("paired_eligible"):
        raise RolloutError("shadow receipt paired eligibility mismatch")
    decision_id = artifact.get("decision_id")
    query_semantic_sha256 = artifact.get("query_semantic_sha256")
    observed_at = artifact.get("observed_at")
    if (
        not isinstance(decision_id, str)
        or not isinstance(query_semantic_sha256, str)
        or not isinstance(observed_at, str)
    ):
        raise RolloutError("shadow replay source identity fields are invalid")
    decision_id_text = decision_id
    query_semantic_sha256_text = query_semantic_sha256
    observed_at_text = observed_at
    try:
        source_fields = _PRODUCER_SHADOW_SOURCE_FIELDS(
            decision_id=decision_id_text,
            query_semantic_sha256=query_semantic_sha256_text,
            observed_at=observed_at_text,
            pool_rows=pool,
            selected_candidate_ids=selected,
            baseline_pool_rows=baseline_pool,
            baseline_selected_candidate_ids=baseline_selected,
            paired_eligible=artifact["paired_eligible"],
        )
    except RolloutError:
        raise
    except Exception as exc:
        raise RolloutError("shadow replay source derivation failed") from exc
    for key in ("row_id", "rally_id", "candidate_id", "as_of", "split", "split_role"):
        if (
            artifact.get(key) != source_fields.get(key)
            or receipt.get(key) != source_fields.get(key)
        ):
            raise RolloutError("shadow replay source identity mismatch")
    hashes = _shadow_hashes(
        features,
        baseline_features,
        pool,
        baseline_pool,
        selected_candidate_ids=selected,
        baseline_selected_candidate_ids=baseline_selected,
    )
    for key, expected in hashes.items():
        if key != "feature_parity":
            if artifact.get(key) != expected:
                raise RolloutError(f"shadow {key} hash mismatch")
            if receipt.get(key) != expected:
                raise RolloutError(f"shadow receipt {key} hash mismatch")
        if evidence.get(key) != expected:
            raise RolloutError(f"shadow evidence {key} hash mismatch")
    runtime_sha = canonical_json_sha256_strict(runtime)
    evidence_sha = canonical_json_sha256_strict(evidence)
    if artifact.get("runtime_observation_sha256") != runtime_sha:
        raise RolloutError("shadow runtime observation hash mismatch")
    if artifact.get("operational_evidence_sha256") != evidence_sha:
        raise RolloutError("shadow operational evidence hash mismatch")
    if receipt.get("runtime_observation_sha256") != runtime_sha:
        raise RolloutError("shadow receipt runtime hash mismatch")
    if receipt.get("operational_evidence_sha256") != evidence_sha:
        raise RolloutError("shadow receipt operational hash mismatch")
    row_id = artifact["row_id"]
    rally_id = artifact["rally_id"]
    candidate_id = artifact["candidate_id"]
    if (
        not isinstance(candidate_id, str)
        or candidate_id not in candidate_pool_ids
        or candidate_id not in set(selected)
        or not any(
            row["candidate_id"] == candidate_id and row["selected"]
            for row in pool
        )
    ):
        raise RolloutError("shadow candidate identity is not selected in the paired pool")
    split = artifact["split"]
    split_role = artifact["split_role"]
    pair_id = str(hashes["pair_id"])
    row = {
        "row_id": row_id,
        "rally_id": rally_id,
        "candidate_id": candidate_id,
        "as_of": artifact["as_of"],
        "split": split,
        "split_role": split_role,
        "stage": context["stage"],
        "cohort": context["cohort"],
        "candidate_policy_id": context["candidate_policy_id"],
        "baseline_policy_id": context["baseline_policy_id"],
        "baseline_artifact_id": context["baseline_artifact_id"],
        "pair_id": pair_id,
        "shadow_observation_artifact_id": artifact_id,
        "shadow_receipt_record_sha256": receipt["record_sha256"],
        "qualified_run_id": run_id,
        "candidate_decision_sha256": hashes["candidate_decision_sha256"],
        "baseline_decision_sha256": hashes["baseline_decision_sha256"],
        "candidate_pool_sha256": hashes["candidate_pool_sha256"],
        "baseline_pool_sha256": hashes["baseline_pool_sha256"],
        "candidate_feature_snapshot_sha256": hashes[
            "candidate_feature_snapshot_sha256"
        ],
        "baseline_feature_snapshot_sha256": hashes[
            "baseline_feature_snapshot_sha256"
        ],
        "candidate_feature_bytes_sha256": hashes["candidate_feature_bytes_sha256"],
        "baseline_feature_bytes_sha256": hashes["baseline_feature_bytes_sha256"],
        "feature_snapshot_sha256": hashes["feature_snapshot_sha256"],
        "runtime_observation_sha256": runtime_sha,
        "operational_evidence_sha256": evidence_sha,
    }
    if row["split"] != row["split_role"] or row["split"] not in {
        "train",
        "validation",
        "test",
    }:
        raise RolloutError("shadow observation split role is invalid")
    _require_hex(row["row_id"], "replay row id")
    _require_hex(row["rally_id"], "replay rally id")
    _require_text(row["candidate_id"], "replay candidate id")
    _require_text(row["stage"], "replay stage")
    _require_text(row["cohort"], "replay cohort")
    _require_hex(row["candidate_policy_id"], "replay candidate policy id")
    _require_hex(row["baseline_policy_id"], "replay baseline policy id")
    _require_hex(row["baseline_artifact_id"], "replay baseline artifact id")
    _require_hex(row["qualified_run_id"], "replay qualified run id")
    _require_hex(
        row["shadow_receipt_record_sha256"], "shadow receipt record id"
    )
    _strict_utc(row["as_of"], "replay row as-of")
    for key in _LOCKED_REPLAY_ROW_KEYS - {
        "row_id",
        "rally_id",
        "candidate_id",
        "as_of",
        "split",
        "split_role",
        "stage",
        "cohort",
        "candidate_policy_id",
        "baseline_policy_id",
        "baseline_artifact_id",
        "qualified_run_id",
        "shadow_observation_artifact_id",
        "shadow_receipt_record_sha256",
    }:
        _require_hex(row[key], f"replay {key}")
    return row


def _derive_shadow_rows(
    root: Path,
    artifact_ids: object,
    context: Mapping[str, str],
    *,
    allow_empty: bool = False,
) -> list[dict[str, Any]]:
    if isinstance(artifact_ids, (str, bytes, bytearray)) or not isinstance(
        artifact_ids, Sequence
    ):
        raise RolloutError("shadow observation artifact ids are invalid")
    if not artifact_ids and not allow_empty:
        raise RolloutError("shadow observation artifact ids are empty")
    ids = [_require_hex(value, "shadow observation artifact id") for value in artifact_ids]
    if len(set(ids)) != len(ids):
        raise RolloutError("shadow observation artifact ids are not unique")
    receipt_rows: dict[str, Mapping[str, Any]] = {}
    if ids:
        receipt_rows = _shadow_receipt_index(root)
    rows = [
        _shadow_row(root, artifact_id, context, receipt=receipt_rows.get(artifact_id))
        for artifact_id in ids
    ]
    if len({row["row_id"] for row in rows}) != len(rows):
        raise RolloutError("shadow observation row ids are not unique")
    if len({row["pair_id"] for row in rows}) != len(rows):
        raise RolloutError("shadow observation pair ids are not unique")
    return sorted(rows, key=lambda row: (row["as_of"], row["row_id"]))


def _validate_locked_rows(rows: object) -> list[dict[str, Any]]:
    if isinstance(rows, (str, bytes, bytearray)) or not isinstance(rows, list):
        raise RolloutError("locked replay rows are invalid")
    values: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != _LOCKED_REPLAY_ROW_KEYS:
            raise RolloutError("locked replay row schema is not closed")
        values.append(dict(row))
    if not values:
        raise RolloutError("locked replay rows are empty")
    if len({row["row_id"] for row in values}) != len(values):
        raise RolloutError("locked replay row ids are not unique")
    if len({row["shadow_observation_artifact_id"] for row in values}) != len(values):
        raise RolloutError("locked replay shadow sources are not unique")
    for row in values:
        _require_hex(row["row_id"], "replay row id")
        _require_hex(row["rally_id"], "replay rally id")
        _require_text(row["candidate_id"], "replay candidate id")
        _strict_utc(row["as_of"], "replay row as-of")
        if row["split"] != row["split_role"] or row["split"] not in {
            "train",
            "validation",
            "test",
        }:
            raise RolloutError("replay row split role is invalid")
        _require_hex(row["shadow_observation_artifact_id"], "shadow source id")
        _require_text(row["stage"], "replay stage")
        _require_text(row["cohort"], "replay cohort")
        _require_hex(row["candidate_policy_id"], "replay candidate policy id")
        _require_hex(row["baseline_policy_id"], "replay baseline policy id")
        _require_hex(row["baseline_artifact_id"], "replay baseline artifact id")
        _require_hex(row["qualified_run_id"], "replay qualified run id")
        _require_hex(
            row["shadow_receipt_record_sha256"], "shadow receipt record id"
        )
        for key in _LOCKED_REPLAY_ROW_KEYS - {
            "row_id",
            "rally_id",
            "candidate_id",
            "as_of",
            "split",
            "split_role",
            "stage",
            "cohort",
            "candidate_policy_id",
            "baseline_policy_id",
            "baseline_artifact_id",
            "qualified_run_id",
            "shadow_observation_artifact_id",
            "shadow_receipt_record_sha256",
        }:
            _require_hex(row[key], f"replay {key}")
    if {row["split"] for row in values} < {"train", "validation", "test"}:
        raise RolloutError("locked replay split roles are incomplete")
    return values


def _locked_replay(root: Path, split_sha256: str) -> dict[str, Any]:
    _require_hex(split_sha256, "evaluation split identity")
    path = store.distillation_dir(root) / "locked-replays" / f"{split_sha256}.json"
    try:
        replay = _stable_sealed(
            path,
            base=store.distillation_dir(root),
            schema="chronovisor.recall-distill-locked-replay.v1",
            label="locked replay artifact",
        )
    except store.DistillationStoreError as exc:
        raise RolloutError("evaluation split artifact is invalid") from exc
    if set(replay) != _LOCKED_REPLAY_KEYS:
        raise RolloutError("locked replay schema is not closed")
    if (
        replay.get("artifact_id") != split_sha256
        or _immutable_artifact_id(replay) != split_sha256
        or replay.get("kind") != "locked-replay-input"
    ):
        raise RolloutError("evaluation split artifact binding mismatch")
    for key in (
        "run_id",
        "baseline_artifact_id",
        "baseline_policy_id",
        "candidate_policy_id",
        "training_snapshot_id",
        "policy_sha256",
        "candidate_head",
        "offline_gate_sha256",
        "model_cohort_sha256",
    ):
        _require_hex(replay[key], f"locked replay {key}")
    for key in ("stage", "cohort", "profile_contract_id", "split_revision"):
        _require_text(replay[key], f"locked replay {key}")
    rows = _validate_locked_rows(replay.get("training_rows"))
    rows_sha256 = replay.get("training_rows_sha256")
    if not isinstance(rows_sha256, str) or _HEX.fullmatch(rows_sha256) is None:
        raise RolloutError("locked replay row hash is invalid")
    if canonical_json_sha256_strict(rows) != rows_sha256:
        raise RolloutError("locked replay row hash mismatch")
    context = _replay_context(
        run_id=replay["run_id"],
        stage=replay["stage"],
        cohort=replay["cohort"],
        candidate_policy_id=replay["candidate_policy_id"],
        baseline_policy_id=replay["baseline_policy_id"],
        baseline_artifact_id=replay["baseline_artifact_id"],
    )
    candidate_policy = _stable_sealed(
        store.distillation_dir(root)
        / "policies"
        / f"{replay['candidate_policy_id']}.json",
        base=store.distillation_dir(root),
        schema=POLICY_SCHEMA,
        label="locked replay candidate policy",
    )
    baseline_policy = _stable_sealed(
        store.distillation_dir(root)
        / "policies"
        / f"{replay['baseline_policy_id']}.json",
        base=store.distillation_dir(root),
        schema=POLICY_SCHEMA,
        label="locked replay baseline policy",
    )
    baseline = _stable_sealed(
        store.distillation_dir(root)
        / "baselines"
        / f"{replay['baseline_artifact_id']}.json",
        base=store.distillation_dir(root),
        schema="chronovisor.recall-distill-baseline.v1",
        label="locked replay baseline artifact",
    )
    offline_gate = baseline.get("offline_training_gate")
    if (
        candidate_policy.get("artifact_id") != replay["candidate_policy_id"]
        or baseline_policy.get("artifact_id") != replay["baseline_policy_id"]
        or baseline.get("artifact_id") != replay["baseline_artifact_id"]
        or replay["policy_sha256"] != _immutable_artifact_id(candidate_policy)
        or replay["profile_contract_id"] != replay["cohort"]
        or replay["split_revision"] != "locked-shadow-v1"
        or not isinstance(offline_gate, Mapping)
        or replay["offline_gate_sha256"]
        != canonical_json_sha256_strict(offline_gate)
        or replay["model_cohort_sha256"]
        != canonical_json_sha256_strict(
            {
                "cohort": replay["cohort"],
                "candidate_policy_id": replay["candidate_policy_id"],
                "baseline_policy_id": replay["baseline_policy_id"],
            }
        )
    ):
        raise RolloutError("locked replay provenance is not source-bound")
    if replay["training_snapshot_id"] != canonical_json_sha256_strict(
        {"run_id": replay["run_id"], "cohort": replay["cohort"], "rows": rows}
    ):
        raise RolloutError("locked replay training snapshot is not source-bound")
    receipts = _shadow_receipt_index(root)
    for row in rows:
        source = _shadow_row(
            root,
            row["shadow_observation_artifact_id"],
            context,
            receipt=receipts.get(row["shadow_observation_artifact_id"]),
        )
        if source != row:
            raise RolloutError("locked replay row is not source-derived")
    return replay


def _registry_path(root: Path) -> Path:
    base = store.distillation_dir(root)
    try:
        for parent in (base.parent.parent, base.parent, base):
            if parent.exists() and stat.S_ISLNK(os.lstat(parent).st_mode):
                raise OSError("registry path component is a symlink")
        if not stat.S_ISDIR(_path_identity(base)[2]):
            raise OSError("registry root is not a directory")
        ledger = base / "replay-observation-pairs.jsonl"
        lock = ledger.with_suffix(ledger.suffix + ".lock")
        for path in (ledger, lock):
            if path.exists() and stat.S_ISLNK(os.lstat(path).st_mode):
                raise OSError("registry path is a symlink")
        return ledger
    except OSError as exc:
        raise RolloutError("replay pair registry path is unsafe") from exc


def _registry_snapshot(paths: Sequence[Path]) -> dict[Path, bytes | None]:
    snapshot: dict[Path, bytes | None] = {}
    for path in paths:
        try:
            info = os.lstat(path)
        except FileNotFoundError:
            snapshot[path] = None
            continue
        except OSError as exc:
            raise RolloutError("replay pair registry snapshot failed") from exc
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise RolloutError("replay pair registry snapshot path is unsafe")
        try:
            snapshot[path] = path.read_bytes()
        except OSError as exc:
            raise RolloutError("replay pair registry snapshot failed") from exc
    return snapshot


def _restore_registry_snapshot(snapshot: Mapping[Path, bytes | None]) -> None:
    """Restore every registry sidecar after a failed multi-row append."""

    for path, data in snapshot.items():
        try:
            if data is None:
                if path.exists() or path.is_symlink():
                    remove_info = os.lstat(path)
                    if stat.S_ISLNK(remove_info.st_mode) or not stat.S_ISREG(
                        remove_info.st_mode
                    ):
                        raise OSError("registry rollback path is unsafe")
                    path.unlink()
                continue
            current_info: os.stat_result | None = (
                os.lstat(path) if path.exists() else None
            )
            if current_info is not None and (
                stat.S_ISLNK(current_info.st_mode)
                or not stat.S_ISREG(current_info.st_mode)
            ):
                raise OSError("registry rollback path is unsafe")
            path.write_bytes(data)
        except OSError as exc:
            raise RolloutError("replay pair registry rollback failed") from exc


def _replay_source_set_sha256(pairs: Sequence[Mapping[str, Any]]) -> str:
    source_ids = [
        _require_hex(row.get("shadow_observation_artifact_id"), "shadow source id")
        for row in pairs
    ]
    if len(source_ids) != len(set(source_ids)):
        raise RolloutError("replay pair source identities are not unique")
    return canonical_json_sha256_strict(sorted(source_ids))


def _validated_replay_registry_rows(root: Path) -> list[Mapping[str, Any]]:
    """Reject registry rows that are not bound to a sealed source window."""

    rows = _stable_chain_rows(root, "replay-observation-pairs.jsonl")
    if not rows:
        return []
    expected_keys = {
        "schema",
        "namespace",
        "previous_sha256",
        "record_sha256",
        *(_REPLAY_REGISTRY_PAYLOAD_KEYS),
    }
    validated: list[Mapping[str, Any]] = []
    observations: dict[str, dict[str, Any]] = {}
    seen_pairs: set[str] = set()
    seen_sources: set[str] = set()
    for row in rows:
        if set(row) != expected_keys:
            raise RolloutError("replay pair registry schema is not closed")
        if row.get("kind") != "replay-observation-pair":
            raise RolloutError("replay pair registry kind is invalid")
        for key in (
            "pair_id",
            "observation_artifact_id",
            "run_id",
            "candidate_policy_id",
            "baseline_policy_id",
            "baseline_artifact_id",
            "split_sha256",
            "row_id",
            "shadow_observation_artifact_id",
            "source_set_sha256",
        ):
            _require_hex(row.get(key), f"replay pair registry {key}")
        for key in ("stage", "cohort"):
            _require_text(row.get(key), f"replay pair registry {key}")
        pair_id = str(row["pair_id"])
        source_id = str(row["shadow_observation_artifact_id"])
        if pair_id in seen_pairs:
            raise RolloutError("replay pair registry pair identity is duplicated")
        if source_id in seen_sources:
            raise RolloutError("replay pair registry source identity is duplicated")
        seen_pairs.add(pair_id)
        seen_sources.add(source_id)
        observation_id = str(row["observation_artifact_id"])
        observation = observations.get(observation_id)
        if observation is None:
            try:
                observation = _stable_sealed(
                    store.distillation_dir(root)
                    / "rollout-observations"
                    / f"{observation_id}.json",
                    base=store.distillation_dir(root),
                    schema=REPLAY_OBSERVATION_SCHEMA,
                    label="replay pair registry observation artifact",
                )
            except store.DistillationStoreError as exc:
                raise RolloutError(
                    "replay pair registry observation artifact is invalid"
                ) from exc
            observations[observation_id] = observation
            _replay_observation(
                root,
                observation_id,
                run_id=_require_hex(observation.get("run_id"), "observation run id"),
                stage=_require_text(observation.get("stage"), "observation stage"),
                cohort=_require_text(
                    observation.get("cohort"), "observation cohort"
                ),
                candidate_policy_id=_require_hex(
                    observation.get("candidate_policy_id"),
                    "observation candidate policy id",
                ),
                baseline_policy_id=_require_hex(
                    observation.get("baseline_policy_id"),
                    "observation baseline policy id",
                ),
                baseline_artifact_id=_require_hex(
                    observation.get("baseline_artifact_id"),
                    "observation baseline artifact id",
                ),
                split_sha256=_require_hex(
                    observation.get("split_sha256"), "observation split id"
                ),
                now=None,
                register=False,
            )
        pairs = observation.get("pairs")
        if not isinstance(pairs, list) or not all(
            isinstance(pair, Mapping) for pair in pairs
        ):
            raise RolloutError("replay pair registry observation pairs are invalid")
        source_set_sha256 = _replay_source_set_sha256(pairs)
        if row["source_set_sha256"] != source_set_sha256:
            raise RolloutError("replay pair registry source set binding mismatch")
        if any(
            observation.get(key) != row[key]
            for key in (
                "run_id",
                "stage",
                "cohort",
                "candidate_policy_id",
                "baseline_policy_id",
                "baseline_artifact_id",
                "split_sha256",
            )
        ):
            raise RolloutError("replay pair registry provenance mismatch")
        matching = [
            pair
            for pair in pairs
            if pair.get("pair_id") == row["pair_id"]
            and pair.get("shadow_observation_artifact_id")
            == row["shadow_observation_artifact_id"]
            and pair.get("row_id") == row["row_id"]
        ]
        if len(matching) != 1:
            raise RolloutError("replay pair registry source binding mismatch")
        validated.append(row)
    return validated


def _register_replay_pairs(root: Path, observation: Mapping[str, Any]) -> None:
    if not isinstance(observation, Mapping):
        raise RolloutError("replay pair registry observation is invalid")
    artifact_id = _require_hex(
        observation.get("artifact_id"), "observation artifact id"
    )
    observation_path = (
        store.distillation_dir(root)
        / "rollout-observations"
        / f"{artifact_id}.json"
    )
    candidate_pairs = observation.get("pairs")
    if isinstance(candidate_pairs, list):
        existing_rows = _stable_chain_rows(root, "replay-observation-pairs.jsonl")
        prior_by_pair = {
            row.get("pair_id"): row.get("observation_artifact_id")
            for row in existing_rows
            if isinstance(row, Mapping)
        }
        for pair in candidate_pairs:
            if not isinstance(pair, Mapping):
                continue
            pair_id = pair.get("pair_id")
            prior = prior_by_pair.get(pair_id)
            if isinstance(pair_id, str) and prior is not None and prior != artifact_id:
                raise RolloutError("replay observation pair was already registered")
    try:
        sealed = _stable_sealed(
            observation_path,
            base=store.distillation_dir(root),
            schema=REPLAY_OBSERVATION_SCHEMA,
            label="replay pair registry observation artifact",
        )
    except store.DistillationStoreError as exc:
        raise RolloutError(
            "replay pair registry observation artifact is invalid"
        ) from exc
    if dict(sealed) != dict(observation):
        raise RolloutError(
            "replay pair registry observation does not match sealed artifact"
        )
    try:
        _replay_observation(
            root,
            artifact_id,
            run_id=_require_hex(observation.get("run_id"), "observation run id"),
            stage=_require_text(observation.get("stage"), "observation stage"),
            cohort=_require_text(observation.get("cohort"), "observation cohort"),
            candidate_policy_id=_require_hex(
                observation.get("candidate_policy_id"),
                "observation candidate policy id",
            ),
            baseline_policy_id=_require_hex(
                observation.get("baseline_policy_id"),
                "observation baseline policy id",
            ),
            baseline_artifact_id=_require_hex(
                observation.get("baseline_artifact_id"),
                "observation baseline artifact id",
            ),
            split_sha256=_require_hex(
                observation.get("split_sha256"), "observation split id"
            ),
            now=None,
            register=False,
        )
    except RolloutError:
        raise
    except (store.DistillationStoreError, ValueError) as exc:
        raise RolloutError(
            "replay pair registry observation is not source-derived"
        ) from exc
    ledger = _registry_path(root)
    pairs = observation.get("pairs")
    if not isinstance(pairs, list):
        raise RolloutError("replay observation pairs are invalid")
    if any(not isinstance(row, Mapping) for row in pairs):
        raise RolloutError("replay observation pairs are invalid")
    source_set_sha256 = _replay_source_set_sha256(pairs)
    if not pairs:
        return
    lock_path = ledger.with_suffix(ledger.suffix + ".lock")
    base = store.distillation_dir(root)
    base_identity = _path_identity(base)
    ledger_identity = _path_identity(ledger) if ledger.exists() else None
    try:
        with store._locked(lock_path):
            existing = store._read_chain_locked(ledger)
            snapshot = _registry_snapshot(
                (
                    ledger,
                    ledger.with_suffix(ledger.suffix + ".head.json"),
                    store._unique_index_path(ledger),
                    store._unique_index_checkpoint_path(ledger),
                )
            )
            known = {
                row.get("pair_id"): row.get("observation_artifact_id")
                for row in existing
                if isinstance(row, Mapping)
            }
            pending: list[dict[str, Any]] = []
            for row in pairs:
                if not isinstance(row, Mapping) or set(row) != _REPLAY_PAIR_KEYS:
                    raise RolloutError("replay observation pair is invalid")
                pair_id = _require_hex(row.get("pair_id"), "replay pair id")
                prior = known.get(pair_id)
                if prior is not None and prior != artifact_id:
                    raise RolloutError("replay observation pair was already registered")
                if prior is None:
                    pending.append(
                        {
                            "kind": "replay-observation-pair",
                            "pair_id": pair_id,
                            "observation_artifact_id": artifact_id,
                            "run_id": row["run_id"],
                            "stage": row["stage"],
                            "cohort": row["cohort"],
                            "candidate_policy_id": row["candidate_policy_id"],
                            "baseline_policy_id": row["baseline_policy_id"],
                            "baseline_artifact_id": row["baseline_artifact_id"],
                            "split_sha256": row["split_sha256"],
                            "row_id": row["row_id"],
                            "shadow_observation_artifact_id": row[
                                "shadow_observation_artifact_id"
                            ],
                            "source_set_sha256": source_set_sha256,
                        }
                    )
            try:
                for payload in pending:
                    store.append_chain_unique_locked(
                        ledger,
                        payload,
                        unique_field="pair_id",
                        binding_field="observation_artifact_id",
                    )
            except Exception:
                _restore_registry_snapshot(snapshot)
                raise
        try:
            if _path_identity(base)[:3] != base_identity[:3]:
                raise OSError("replay pair registry root changed while writing")
            if stat.S_ISLNK(os.lstat(ledger).st_mode):
                raise OSError("replay pair registry became a symlink")
            if ledger_identity is not None:
                current = _path_identity(ledger)
                if current[:3] != ledger_identity[:3]:
                    raise OSError(
                        "replay pair registry identity changed while writing"
                    )
            elif not stat.S_ISREG(_path_identity(ledger)[2]):
                raise OSError("replay pair registry is not a regular file")
        except Exception:
            _restore_registry_snapshot(snapshot)
            raise
    except RolloutError:
        raise
    except Exception as exc:
        raise RolloutError("replay observation pair registry append failed") from exc


def _replay_pair_conflict_preflight(
    root: Path, pairs: Sequence[Mapping[str, Any]], artifact_id: str
) -> None:
    if not pairs:
        return
    ledger = _registry_path(root)
    lock_path = ledger.with_suffix(ledger.suffix + ".lock")
    try:
        with store._locked(lock_path):
            existing = store._read_chain_locked(ledger)
            known = {
                row.get("pair_id"): row.get("observation_artifact_id")
                for row in existing
                if isinstance(row, Mapping)
            }
            for row in pairs:
                pair_id = _require_hex(row.get("pair_id"), "replay pair id")
                prior = known.get(pair_id)
                if prior is not None and prior != artifact_id:
                    raise RolloutError("replay observation pair was already registered")
    except (OSError, store.DistillationStoreError) as exc:
        raise RolloutError("replay pair registry preflight failed") from exc


def _derived_pairs(
    root: Path,
    artifact_ids: object,
    context: Mapping[str, str],
    split: Mapping[str, Any],
    *,
    now: datetime | None,
) -> list[dict[str, Any]]:
    rows = _derive_shadow_rows(root, artifact_ids, context)
    split_rows = {
        row["shadow_observation_artifact_id"]: row
        for row in _validate_locked_rows(split["training_rows"])
    }
    values: list[dict[str, Any]] = []
    for row in rows:
        source_id = row["shadow_observation_artifact_id"]
        if source_id not in split_rows or split_rows[source_id] != row:
            raise RolloutError("replay pair is not a member of the locked split")
        source = _shadow_observation(root, source_id)
        observed_at = _strict_utc(source["observed_at"], "shadow observation timestamp")
        if now is not None and observed_at > now:
            raise RolloutError("replay observation timestamp is in the future")
        values.append(
            {
                "pair_id": row["pair_id"],
                "row_id": row["row_id"],
                "observed_at": observed_at.isoformat().replace("+00:00", "Z"),
                **{
                    key: context[key]
                    for key in (
                        "run_id",
                        "stage",
                        "cohort",
                        "candidate_policy_id",
                        "baseline_policy_id",
                        "baseline_artifact_id",
                    )
                },
                "split_sha256": split["artifact_id"],
                **{
                    key: row[key]
                    for key in _LOCKED_REPLAY_ROW_KEYS
                    if key
                    not in {
                        "row_id",
                        "rally_id",
                        "candidate_id",
                        "as_of",
                        "split",
                        "split_role",
                        "shadow_observation_artifact_id",
                    }
                },
                "shadow_observation_artifact_id": source_id,
            }
        )
    return values


def write_locked_replay_input(
    root: Path | None,
    *,
    shadow_observation_artifact_ids: Sequence[str],
    run_id: str,
    stage: str,
    cohort: str,
    candidate_policy_id: str,
    baseline_policy_id: str,
    baseline_artifact_id: str,
) -> dict[str, Any]:
    """Derive and publish a typed locked split from sealed shadow artifacts."""

    chronovisor_root = _root(root)
    context = _replay_context(
        run_id=run_id,
        stage=stage,
        cohort=cohort,
        candidate_policy_id=candidate_policy_id,
        baseline_policy_id=baseline_policy_id,
        baseline_artifact_id=baseline_artifact_id,
    )
    rows = _derive_shadow_rows(
        chronovisor_root, shadow_observation_artifact_ids, context
    )
    baseline = _stable_sealed(
        store.distillation_dir(chronovisor_root)
        / "baselines"
        / f"{baseline_artifact_id}.json",
        base=store.distillation_dir(chronovisor_root),
        schema="chronovisor.recall-distill-baseline.v1",
        label="locked replay baseline artifact",
    )
    policy = _stable_sealed(
        store.distillation_dir(chronovisor_root)
        / "policies"
        / f"{candidate_policy_id}.json",
        base=store.distillation_dir(chronovisor_root),
        schema=POLICY_SCHEMA,
        label="locked replay candidate policy",
    )
    baseline_policy = _stable_sealed(
        store.distillation_dir(chronovisor_root)
        / "policies"
        / f"{baseline_policy_id}.json",
        base=store.distillation_dir(chronovisor_root),
        schema=POLICY_SCHEMA,
        label="locked replay baseline policy",
    )
    if (
        policy.get("artifact_id") != candidate_policy_id
        or baseline_policy.get("artifact_id") != baseline_policy_id
    ):
        raise RolloutError("locked replay policy identity mismatch")
    offline_gate = baseline.get("offline_training_gate")
    if not isinstance(offline_gate, Mapping):
        raise RolloutError("locked replay baseline gate is invalid")
    candidate_head = _stable_chain_head(chronovisor_root, "candidate-ledger.jsonl")
    if _HEX.fullmatch(str(candidate_head)) is None:
        candidate_head = canonical_json_sha256_strict({"ledger": "empty"})
    payload: dict[str, Any] = {
        "kind": "locked-replay-input",
        **context,
        "training_snapshot_id": canonical_json_sha256_strict(
            {"run_id": run_id, "cohort": cohort, "rows": rows}
        ),
        "training_rows": rows,
        "training_rows_sha256": canonical_json_sha256_strict(rows),
        "policy_sha256": _immutable_artifact_id(policy),
        "candidate_head": candidate_head,
        "profile_contract_id": cohort,
        "offline_gate_sha256": canonical_json_sha256_strict(offline_gate),
        "model_cohort_sha256": canonical_json_sha256_strict(
            {
                "cohort": cohort,
                "candidate_policy_id": candidate_policy_id,
                "baseline_policy_id": baseline_policy_id,
            }
        ),
        "split_revision": "locked-shadow-v1",
    }
    directory = _artifact_directory(chronovisor_root, "locked-replays")
    try:
        artifact_id, path, _ = store.write_immutable(
            directory,
            payload,
            schema="chronovisor.recall-distill-locked-replay.v1",
        )
        artifact = _stable_sealed(
            path,
            base=store.distillation_dir(chronovisor_root),
            schema="chronovisor.recall-distill-locked-replay.v1",
            label="locked replay artifact",
        )
    except store.DistillationStoreError as exc:
        raise RolloutError("locked replay write failed") from exc
    if artifact.get("artifact_id") != artifact_id:
        raise RolloutError("locked replay writeback identity mismatch")
    _locked_replay(chronovisor_root, artifact_id)
    return artifact


def _write_replay_observation(
    root: Path | None,
    *,
    run_id: str,
    stage: str,
    cohort: str,
    candidate_policy_id: str,
    baseline_policy_id: str,
    baseline_artifact_id: str,
    split_sha256: str,
    shadow_observation_artifact_ids: Sequence[str],
    allow_empty: bool,
) -> dict[str, Any]:
    chronovisor_root = _root(root)
    context = _replay_context(
        run_id=run_id,
        stage=stage,
        cohort=cohort,
        candidate_policy_id=candidate_policy_id,
        baseline_policy_id=baseline_policy_id,
        baseline_artifact_id=baseline_artifact_id,
    )
    split = _locked_replay(chronovisor_root, split_sha256)
    if any(split.get(key) != value for key, value in context.items()):
        raise RolloutError("replay observation split provenance mismatch")
    now = datetime.now(UTC)
    if allow_empty and shadow_observation_artifact_ids == []:
        values: list[dict[str, Any]] = []
        first = last = None
    else:
        values = _derived_pairs(
            chronovisor_root,
            shadow_observation_artifact_ids,
            context,
            split,
            now=now,
        )
        timestamps = [
            _strict_utc(row["observed_at"], "replay observation timestamp")
            for row in values
        ]
        first = min(timestamps).isoformat().replace("+00:00", "Z")
        last = max(timestamps).isoformat().replace("+00:00", "Z")
    payload = {
        "kind": "locked-replay-observation-window",
        **context,
        "split_sha256": split_sha256,
        "first_observed_at": first,
        "last_observed_at": last,
        "pair_count": len(values),
        "pairs_sha256": canonical_json_sha256_strict(values),
        "pairs": values,
    }
    # A complete conflict preflight happens before immutable publication.  If
    # a later append still fails, the newly-created observation is removed.
    _registry_path(chronovisor_root)
    predicted_artifact_id = canonical_json_sha256_strict(
        {
            "schema": REPLAY_OBSERVATION_SCHEMA,
            "namespace": "recall-distillation",
            **payload,
        }
    )
    _replay_pair_conflict_preflight(
        chronovisor_root, values, predicted_artifact_id
    )
    directory = _artifact_directory(chronovisor_root, "rollout-observations")
    path = directory / f"{canonical_json_sha256_strict({**payload, 'schema': REPLAY_OBSERVATION_SCHEMA, 'namespace': 'recall-distillation'})}.json"
    existed = path.exists()
    try:
        artifact_id, path, _ = store.write_immutable(
            directory, payload, schema=REPLAY_OBSERVATION_SCHEMA
        )
        artifact = _stable_sealed(
            path,
            base=store.distillation_dir(chronovisor_root),
            schema=REPLAY_OBSERVATION_SCHEMA,
            label="replay observation artifact",
        )
        _replay_observation(
            chronovisor_root,
            artifact_id,
            run_id=run_id,
            stage=stage,
            cohort=cohort,
            candidate_policy_id=candidate_policy_id,
            baseline_policy_id=baseline_policy_id,
            baseline_artifact_id=baseline_artifact_id,
            split_sha256=split_sha256,
            now=now,
            register=False,
        )
        _register_replay_pairs(chronovisor_root, artifact)
    except (store.DistillationStoreError, RolloutError) as exc:
        if not existed:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        if isinstance(exc, RolloutError):
            raise
        raise RolloutError("replay observation write failed") from exc
    if artifact.get("artifact_id") != artifact_id:
        raise RolloutError("replay observation writeback identity mismatch")
    return artifact


def write_replay_observation(
    root: Path | None,
    *,
    run_id: str,
    stage: str,
    cohort: str,
    candidate_policy_id: str,
    baseline_policy_id: str,
    baseline_artifact_id: str,
    split_artifact_id: str,
    shadow_observation_artifact_ids: Sequence[str],
) -> dict[str, Any]:
    """Publish an observation window derived only from sealed shadow artifacts."""

    return _write_replay_observation(
        root,
        run_id=run_id,
        stage=stage,
        cohort=cohort,
        candidate_policy_id=candidate_policy_id,
        baseline_policy_id=baseline_policy_id,
        baseline_artifact_id=baseline_artifact_id,
        split_sha256=split_artifact_id,
        shadow_observation_artifact_ids=shadow_observation_artifact_ids,
        allow_empty=False,
    )


def write_empty_replay_observation(
    root: Path | None,
    *,
    run_id: str,
    stage: str,
    cohort: str,
    candidate_policy_id: str,
    baseline_policy_id: str,
    baseline_artifact_id: str,
    split_artifact_id: str,
) -> dict[str, Any]:
    """Publish an explicit no-observation artifact that can only hold."""

    return _write_replay_observation(
        root,
        run_id=run_id,
        stage=stage,
        cohort=cohort,
        candidate_policy_id=candidate_policy_id,
        baseline_policy_id=baseline_policy_id,
        baseline_artifact_id=baseline_artifact_id,
        split_sha256=split_artifact_id,
        shadow_observation_artifact_ids=[],
        allow_empty=True,
    )


def _replay_observation(
    root: Path,
    artifact_id: object,
    *,
    run_id: str,
    stage: str,
    cohort: str,
    candidate_policy_id: str,
    baseline_policy_id: str,
    baseline_artifact_id: str,
    split_sha256: str,
    now: datetime | None,
    register: bool = False,
) -> tuple[int, int, dict[str, Any]]:
    if not isinstance(artifact_id, str) or _HEX.fullmatch(artifact_id) is None:
        raise RolloutError("replay observation artifact id is invalid")
    observation_path = (
        store.distillation_dir(root)
        / "rollout-observations"
        / f"{artifact_id}.json"
    )
    try:
        artifact = _stable_sealed(
            observation_path,
            base=store.distillation_dir(root),
            schema=REPLAY_OBSERVATION_SCHEMA,
            label="replay observation artifact",
        )
    except store.DistillationStoreError as exc:
        raise RolloutError("replay observation artifact is invalid") from exc
    if artifact.get("artifact_id") != artifact_id:
        raise RolloutError("replay observation identity mismatch")
    if _immutable_artifact_id(artifact) != artifact_id:
        raise RolloutError("replay observation content identity mismatch")
    if set(artifact) != _REPLAY_OBSERVATION_KEYS:
        raise RolloutError("replay observation schema is not closed")
    if artifact.get("kind") != "locked-replay-observation-window":
        raise RolloutError("replay observation kind is invalid")
    context = _replay_context(
        run_id=run_id,
        stage=stage,
        cohort=cohort,
        candidate_policy_id=candidate_policy_id,
        baseline_policy_id=baseline_policy_id,
        baseline_artifact_id=baseline_artifact_id,
    )
    if any(artifact.get(key) != expected for key, expected in context.items()) or artifact.get(
        "split_sha256"
    ) != split_sha256:
        raise RolloutError("replay observation binding mismatch")
    pair_count = artifact.get("pair_count")
    pairs = artifact.get("pairs")
    pairs_sha256 = artifact.get("pairs_sha256")
    if (
        isinstance(pair_count, bool)
        or not isinstance(pair_count, int)
        or pair_count < 0
        or not isinstance(pairs, list)
        or not isinstance(pairs_sha256, str)
        or _HEX.fullmatch(pairs_sha256) is None
        or canonical_json_sha256_strict(pairs) != pairs_sha256
        or pair_count != len(pairs)
    ):
        raise RolloutError("replay observation rows are invalid")

    observed: list[datetime] = []
    for row in pairs:
        if not isinstance(row, Mapping) or set(row) != _REPLAY_PAIR_KEYS:
            raise RolloutError("replay observation row schema is not closed")
        for key in _REPLAY_PAIR_KEYS - {"observed_at", "stage", "cohort"}:
            _require_hex(row[key], f"replay observation {key}")
        if any(row[key] != value for key, value in context.items()) or row[
            "split_sha256"
        ] != split_sha256:
            raise RolloutError("replay observation row binding mismatch")
        instant = _strict_utc(row["observed_at"], "replay observation timestamp")
        if now is not None and instant > now:
            raise RolloutError("replay observation timestamp is in the future")
        observed.append(instant)

    split = _locked_replay(root, split_sha256)
    source_ids = [row["shadow_observation_artifact_id"] for row in pairs]
    expected_pairs = (
        _derived_pairs(root, source_ids, context, split, now=now)
        if pairs
        else []
    )
    if expected_pairs != pairs:
        raise RolloutError("replay observation hashes are not source-derived")

    first = artifact.get("first_observed_at")
    last = artifact.get("last_observed_at")
    if pair_count == 0:
        if first is not None or last is not None:
            raise RolloutError("empty replay observation window is invalid")
        if register:
            _register_replay_pairs(root, artifact)
        return 0, 0, artifact
    first_instant = _strict_utc(first, "replay observation first timestamp")
    last_instant = _strict_utc(last, "replay observation last timestamp")
    if first_instant != min(observed) or last_instant != max(observed):
        raise RolloutError("replay observation window does not match rows")
    if now is not None and last_instant > now:
        raise RolloutError("replay observation window is in the future")
    span_seconds = int((last_instant - first_instant).total_seconds())
    if span_seconds < 0:
        raise RolloutError("replay observation window is reversed")
    if register:
        _register_replay_pairs(root, artifact)
    observed_days = span_seconds // _DAY_SECONDS
    return observed_days, span_seconds, artifact


def _evaluation(
    root: Path, evaluation: Mapping[str, Any], *, now: datetime | None = None
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
        artifact = _stable_sealed(
            store.distillation_dir(root) / "evaluations" / f"{artifact_id}.json",
            base=store.distillation_dir(root),
            schema=EVALUATION_SCHEMA,
            label="evaluation artifact",
        )
    except store.DistillationStoreError as exc:
        raise RolloutError("evaluation artifact is invalid") from exc
    if artifact.get("artifact_id") != artifact_id or artifact.get("run_id") != run_id:
        raise RolloutError("evaluation artifact identity mismatch")
    if _immutable_artifact_id(artifact) != artifact_id:
        raise RolloutError("evaluation content identity mismatch")
    required = {
        "schema",
        "namespace",
        "artifact_id",
        "seal_sha256",
        "kind",
        "run_id",
        "candidate_policy_id",
        "baseline_artifact_id",
        "raw_watermark",
        "baseline_policy_id",
        "split_sha256",
        "feature_revision",
        "feature_parity_sha256",
        "offline_gate_sha256",
        "observation_mode",
        "replay_observation_artifact_id",
        "replay_metrics",
        "shadow_metrics",
        "canary_metrics",
        "operational_metrics_sha256",
        "operational_source_sha256",
    }
    if set(artifact) != required:
        raise RolloutError("evaluation schema is not closed")
    if artifact.get("kind") != "automatic-closed-metrics":
        raise RolloutError("evaluation kind is not producer-owned")
    policy_id = artifact.get("candidate_policy_id")
    if not isinstance(policy_id, str) or _HEX.fullmatch(policy_id) is None:
        raise RolloutError("evaluation candidate policy id is invalid")
    for key in (
        "baseline_artifact_id",
        "raw_watermark",
        "baseline_policy_id",
        "split_sha256",
        "feature_parity_sha256",
        "offline_gate_sha256",
        "operational_metrics_sha256",
        "operational_source_sha256",
    ):
        if not isinstance(artifact[key], str) or _HEX.fullmatch(artifact[key]) is None:
            raise RolloutError(f"evaluation {key} is invalid")
    if artifact["candidate_policy_id"] == artifact["baseline_policy_id"]:
        raise RolloutError("evaluation candidate and baseline policies must differ")
    revision = artifact["feature_revision"]
    if not isinstance(revision, str) or _REVISION.fullmatch(revision) is None:
        raise RolloutError("evaluation feature revision is invalid")
    if artifact["observation_mode"] not in {
        "paired",
        "candidate_only_legacy_incumbent",
    }:
        raise RolloutError("evaluation observation mode is invalid")
    split = _locked_replay(root, artifact["split_sha256"])
    try:
        operational_source_ids = _PRODUCER_OPERATIONAL_SOURCE_IDS(
            root,
            candidate_id=policy_id,
            incumbent_id=artifact["baseline_policy_id"],
            baseline_artifact_id=artifact["baseline_artifact_id"],
            cohort=split["cohort"],
            qualified_run_id=run_id,
            stage_name=split["stage"],
        )
    except (store.DistillationStoreError, ValueError) as exc:
        raise RolloutError("operational source linkage is unavailable") from exc
    try:
        operational_metrics = _PRODUCER_OPERATIONAL_METRICS(
            root,
            policy_id,
            artifact["baseline_policy_id"],
            baseline_artifact_id=artifact["baseline_artifact_id"],
            cohort=split["cohort"],
            qualified_run_id=run_id,
            stage_name=split["stage"],
            source_ids=operational_source_ids,
        )
    except (store.DistillationStoreError, ValueError) as exc:
        raise RolloutError("operational metrics are unavailable") from exc
    if (
        artifact["shadow_metrics"] != operational_metrics
        or artifact["canary_metrics"] != operational_metrics
        or artifact["operational_metrics_sha256"]
        != canonical_json_sha256_strict(operational_metrics)
    ):
        raise RolloutError("operational metrics are not source-derived")
    if artifact["operational_source_sha256"] != canonical_json_sha256_strict(
        operational_source_ids
    ):
        raise RolloutError("operational source linkage is not producer-derived")
    replay_days, replay_seconds, replay_observation = _replay_observation(
        root,
        artifact["replay_observation_artifact_id"],
        run_id=run_id,
        stage=split["stage"],
        cohort=split["cohort"],
        candidate_policy_id=policy_id,
        baseline_policy_id=artifact["baseline_policy_id"],
        baseline_artifact_id=artifact["baseline_artifact_id"],
        split_sha256=artifact["split_sha256"],
        now=now,
    )
    replay_metrics = artifact["replay_metrics"]
    if not isinstance(replay_metrics, Mapping):
        raise RolloutError("replay metrics are invalid")
    if set(replay_metrics) != set(_METRICS):
        raise RolloutError("named rollout metrics are incomplete")
    for name in _METRICS:
        gate = replay_metrics.get(name)
        if (
            not isinstance(gate, Mapping)
            or gate.get("denominator") != replay_observation["pair_count"]
            or (
                isinstance(gate.get("min_denominator"), bool)
                or not isinstance(gate.get("min_denominator"), int)
                or gate["min_denominator"] < _REPLAY_MIN_DENOMINATOR
            )
        ):
            raise RolloutError("replay denominator does not match observations")
    _metrics_gate(
        replay_metrics,
        observation_required=True,
        observed_days=replay_days,
    )
    for gate in (artifact["shadow_metrics"], artifact["canary_metrics"]):
        _metrics_gate(gate, observation_required=True, observed_days=7)
    receipt = {
        "kind": "numeric-rollout-evaluation",
        "run_id": run_id,
        "candidate_policy_id": policy_id,
        "evaluation_sha256": canonical_json_sha256_strict(artifact),
        "evaluation_artifact_id": artifact_id,
        "baseline_artifact_id": artifact["baseline_artifact_id"],
        "raw_watermark": artifact["raw_watermark"],
        "baseline_policy_id": artifact["baseline_policy_id"],
        "split_sha256": artifact["split_sha256"],
        "stage": split["stage"],
        "cohort": split["cohort"],
        "feature_revision": revision,
        "feature_parity_sha256": artifact["feature_parity_sha256"],
        "offline_gate_sha256": artifact["offline_gate_sha256"],
        "replay_observation_artifact_id": artifact["replay_observation_artifact_id"],
        "replay_observation_sha256": canonical_json_sha256_strict(
            replay_observation
        ),
        "replay_observation_days": replay_days,
        "replay_observation_span_seconds": replay_seconds,
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


_TEST_CLOCK_TOKEN = object()


def _evaluate_at(
    root: Path | None,
    now: str,
    evaluation: Mapping[str, Any],
    *,
    _token: object,
) -> dict[str, Any]:
    """Test-only deterministic clock seam; never part of production callers."""

    if _token is not _TEST_CLOCK_TOKEN:
        raise RolloutError("test clock token is required")
    return _evaluate_at_system_clock(root, _now(now), evaluation)


def evaluate_and_advance(
    root: Path | None, evaluation: Mapping[str, Any]
) -> dict[str, Any]:
    """Advance using the authoritative UTC system clock only."""

    return _evaluate_at_system_clock(root, datetime.now(UTC), evaluation)


def _evaluate_at_system_clock(
    root: Path | None, timestamp: datetime, evaluation: Mapping[str, Any]
) -> dict[str, Any]:
    """Advance only sealed replay/shadow/canary gates; otherwise keep serving LKG."""

    chronovisor_root = _root(root)
    lock = store.distillation_dir(chronovisor_root) / "rollout.lock"
    with store._locked(lock):
        state = _state(chronovisor_root)
        if not _enabled(chronovisor_root):
            return _result(state, changed=False)
        run_id, policy_id, receipt = _evaluation(
            chronovisor_root, evaluation, now=timestamp
        )
        if state.get("last_run_id") == run_id:
            receipt_id = state.get("evaluation_receipt_id")
            try:
                prior = _stable_sealed(
                    store.distillation_dir(chronovisor_root)
                    / "rollout-runs"
                    / f"{receipt_id}.json",
                    base=store.distillation_dir(chronovisor_root),
                    schema=EVALUATION_SCHEMA,
                    label="prior rollout receipt",
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
        candidate_artifact = _stable_sealed(
            store.distillation_dir(chronovisor_root)
            / "policies"
            / f"{policy_id}.json",
            base=store.distillation_dir(chronovisor_root),
            schema=POLICY_SCHEMA,
            label="rollout candidate policy",
        )
        bound = receipt["metrics"]
        if candidate_artifact.get("feature_revision") != bound["feature_revision"]:
            raise RolloutError("evaluation feature revision does not match policy")
        if bound["baseline_policy_id"] != _pointer_policy(chronovisor_root, "active"):
            raise RolloutError("evaluation baseline policy does not match active")
        _baseline(chronovisor_root, bound)
        status = str(state.get("status") or "")
        percent = int(state.get("rollout_percent") or 0)
        if bound["observation_mode"] == "candidate_only_legacy_incumbent":
            incumbent = _stable_sealed(
                store.distillation_dir(chronovisor_root)
                / "policies"
                / f"{bound['baseline_policy_id']}.json",
                base=store.distillation_dir(chronovisor_root),
                schema=POLICY_SCHEMA,
                label="rollout incumbent policy",
            )
            if status != "canary" or percent != 100 or incumbent.get(
                "serve_mode"
            ) != "legacy":
                raise RolloutError("candidate-only observation mode is not allowed")
        replay = _metrics_gate(
            bound["replay_metrics"],
            observation_required=True,
            observed_days=int(receipt["replay_observation_days"]),
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
