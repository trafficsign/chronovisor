"""Local-only semantic quality regression and deterministic containment.

This module deliberately has no frontier import or callback.  Quality drift is
contained by freezing the affected lane, selecting the registered
last-known-good authority artifact, and scheduling local shadow replay.  Only
another subsystem may later classify a reproducible *code* failure for the
repair-only FrontierGuard.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from chronovisor.durable_state import (
    DurableStateError,
    atomic_write_bytes,
    canonical_bytes,
    canonical_sha256,
    file_lock,
    read_sealed_json,
    verify_sealed_object,
    write_sealed_json,
)
from chronovisor.schema_compat import canonical_schema


QUALITY_POLICY_VERSION = 1
CORPUS_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class QualityThresholds:
    version: int = QUALITY_POLICY_VERSION
    # Production lane contracts deliberately contain at least five objective
    # cases per lane.  A larger global minimum would silently disable every
    # lane-specific guard.
    minimum_samples: int = 5
    minimum_anchor_match_rate: float = 0.98
    minimum_metamorphic_pass_rate: float = 0.98
    maximum_flip_rate: float = 0.05
    trip_samples: int = 2
    recovery_samples: int = 3
    cooldown_seconds: int = 3600
    incident_budget_per_day: int = 4


def corpus_layout(root: Path) -> dict[str, Path]:
    return {
        "immutable_anchor": root / "immutable-anchor.jsonl",
        "behavior_snapshots": root / "behavior-snapshots",
        "metamorphic": root / "metamorphic.jsonl",
    }


def _read_sealed_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise DurableStateError(f"cannot read quality corpus: {path}: {exc}") from exc
    if raw and not raw.endswith(b"\n"):
        raise DurableStateError(f"quality corpus has a partial tail: {path}")
    for line_number, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            rows.append(verify_sealed_object(payload))
        except Exception as exc:
            raise DurableStateError(
                f"quality corpus line {line_number} is invalid: {path}"
            ) from exc
    return rows


def _replace_sealed_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    from chronovisor.durable_state import seal_object

    encoded = b"".join(canonical_bytes(seal_object(row)) for row in rows)
    atomic_write_bytes(path, encoded, backup=True)


def _normalized_unsigned(value: Mapping[str, Any]) -> dict[str, Any]:
    normalized = {key: item for key, item in value.items() if key != "seal_sha256"}
    schema = normalized.get("schema")
    if isinstance(schema, str):
        normalized["schema"] = canonical_schema(schema)
    return normalized


def append_immutable_anchor(
    *,
    root: Path,
    lane: str,
    case_id: str,
    source_kind: str,
    source_reference: str,
    expected_signature: Mapping[str, Any],
    expected_decision: str | None,
    expected_effect: str,
) -> dict[str, Any]:
    """Append an objective invariant or explicit user correction exactly once.

    A later model evaluation can never rewrite an existing anchor.  Correcting
    an anchor requires a new explicit source reference; contradictory content
    under the same identity is rejected instead of silently relabelled.
    """

    if source_kind not in {"objective_invariant", "user_correction"}:
        raise ValueError("anchor source must be objective_invariant or user_correction")
    identity = {
        "lane": lane,
        "case_id": case_id,
        "source_kind": source_kind,
        "source_reference": source_reference,
    }
    row = {
        "schema": "chronovisor.quality-anchor.v1",
        "anchor_id": canonical_sha256(identity),
        **identity,
        "expected_signature": dict(expected_signature),
        "expected_signature_sha256": canonical_sha256(dict(expected_signature)),
        "expected_decision": expected_decision,
        "expected_effect": expected_effect,
        "immutable": True,
        "frontier_allowed": False,
    }
    path = corpus_layout(root)["immutable_anchor"]
    lock = path.with_name(f"{path.name}.lock")
    with file_lock(lock, exclusive=True):
        rows = _read_sealed_jsonl(path)
        existing = next(
            (item for item in rows if item.get("anchor_id") == row["anchor_id"]),
            None,
        )
        if existing is not None:
            unsigned = _normalized_unsigned(existing)
            if unsigned != row:
                raise DurableStateError("immutable quality anchor conflict")
            return existing
        _replace_sealed_jsonl(path, [*rows, row])
    return row


def _objective_anchor_rows(artifact: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cases = artifact.get("cases")
    if not isinstance(cases, list):
        return rows
    for case in cases:
        if not isinstance(case, Mapping):
            continue
        expected = case.get("expected_signature")
        lane = case.get("decision_lane")
        case_id = case.get("case_id")
        decision = case.get("expected_decision")
        effect = case.get("expected_effect")
        contract_id = case.get("contract_id")
        if not (
            isinstance(expected, Mapping)
            and isinstance(lane, str)
            and lane
            and isinstance(case_id, str)
            and case_id
            and (decision is None or isinstance(decision, str))
            and isinstance(effect, str)
            and isinstance(contract_id, str)
            and contract_id
        ):
            continue
        identity = {
            "lane": lane,
            "case_id": case_id,
            "source_kind": "objective_invariant",
            "source_reference": contract_id,
        }
        rows.append(
            {
                "schema": "chronovisor.quality-anchor.v1",
                "anchor_id": canonical_sha256(identity),
                **identity,
                "expected_signature": dict(expected),
                "expected_signature_sha256": canonical_sha256(dict(expected)),
                "expected_decision": decision,
                "expected_effect": effect,
                "immutable": True,
                "frontier_allowed": False,
            }
        )
    return rows


def bootstrap_objective_corpus(
    *, root: Path, artifact: Mapping[str, Any]
) -> dict[str, Any]:
    """Freeze contract-backed invariants once without copying model behavior."""

    layout = corpus_layout(root)
    anchor_path = layout["immutable_anchor"]
    metamorphic_path = layout["metamorphic"]
    anchors = _objective_anchor_rows(artifact)
    if not anchors:
        raise ValueError("adoption artifact has no objective contract anchors")
    anchor_lock = anchor_path.with_name(f"{anchor_path.name}.lock")
    with file_lock(anchor_lock, exclusive=True):
        existing = _read_sealed_jsonl(anchor_path)
        existing_by_id = {str(row.get("anchor_id")): row for row in existing}
        additions: list[dict[str, Any]] = []
        for row in anchors:
            prior = existing_by_id.get(str(row["anchor_id"]))
            if prior is None:
                additions.append(row)
                continue
            prior_unsigned = _normalized_unsigned(prior)
            if prior_unsigned != row:
                raise DurableStateError("objective anchor identity conflict")
        if additions:
            _replace_sealed_jsonl(anchor_path, [*existing, *additions])
            existing = [*existing, *additions]
    metamorphic = [
        {
            "schema": "chronovisor.quality-metamorphic.v1",
            "relation_id": canonical_sha256(
                {
                    "anchor_id": row["anchor_id"],
                    "transforms": [
                        "json_key_order",
                        "json_whitespace",
                        "json_chunk_join",
                    ],
                }
            ),
            "anchor_id": row["anchor_id"],
            "lane": row["lane"],
            "case_id": row["case_id"],
            "expected_signature_sha256": row["expected_signature_sha256"],
            "transforms": [
                "json_key_order",
                "json_whitespace",
                "json_chunk_join",
            ],
            "immutable": True,
            "frontier_allowed": False,
        }
        for row in existing
    ]
    metamorphic_lock = metamorphic_path.with_name(f"{metamorphic_path.name}.lock")
    with file_lock(metamorphic_lock, exclusive=True):
        existing_relations = _read_sealed_jsonl(metamorphic_path)
        existing_relation_ids = {
            str(row.get("relation_id")): row for row in existing_relations
        }
        relation_additions: list[dict[str, Any]] = []
        for row in metamorphic:
            prior = existing_relation_ids.get(str(row["relation_id"]))
            if prior is None:
                relation_additions.append(row)
                continue
            prior_unsigned = _normalized_unsigned(prior)
            if prior_unsigned != row:
                raise DurableStateError("metamorphic relation identity conflict")
        if relation_additions:
            _replace_sealed_jsonl(
                metamorphic_path,
                [*existing_relations, *relation_additions],
            )
            existing_relations = [*existing_relations, *relation_additions]
    return {
        "anchors": len(existing),
        "metamorphic_relations": len(existing_relations),
        "anchor_sha256": hashlib.sha256(anchor_path.read_bytes()).hexdigest(),
        "metamorphic_sha256": hashlib.sha256(metamorphic_path.read_bytes()).hexdigest(),
        "behavior_promoted_to_anchor": False,
        "frontier_calls": 0,
    }


def _behavior_snapshot(artifact: Mapping[str, Any]) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    artifact_cases = artifact.get("cases")
    if isinstance(artifact_cases, list):
        for case in artifact_cases:
            if not isinstance(case, Mapping):
                continue
            signature = case.get("actual_signature")
            if not isinstance(signature, Mapping):
                continue
            cases.append(
                {
                    "case_id": case.get("case_id"),
                    "lane": case.get("decision_lane"),
                    "actual_signature_sha256": canonical_sha256(dict(signature)),
                    "actual_decision": case.get("actual_decision"),
                    "actual_effect": case.get("actual_effect"),
                }
            )
    return {
        "schema": "chronovisor.quality-behavior-snapshot.v1",
        "artifact_evidence_sha256": artifact.get("evidence_sha256"),
        "artifact_result_sha256": artifact.get("evaluation_result_sha256"),
        "config_sha256": artifact.get("config_sha256"),
        "model_metadata_sha256": artifact.get("model_metadata_sha256"),
        "cases": cases,
        "oracle": False,
        "behavior_promoted_to_anchor": False,
        "frontier_allowed": False,
    }


def _publish_behavior_snapshot(
    root: Path, artifact: Mapping[str, Any]
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    layout = corpus_layout(root)
    directory = layout["behavior_snapshots"]
    pointer_path = directory / "latest.json"
    pointer_version: str | None = None
    try:
        pointer = read_sealed_json(pointer_path)
        pointer_version = str(pointer.get("version") or "") or None
        previous = read_sealed_json(Path(str(pointer["snapshot_path"])))
    except (DurableStateError, KeyError, OSError):
        previous = None
    snapshot = _behavior_snapshot(artifact)
    version = str(snapshot.get("artifact_evidence_sha256") or canonical_sha256(snapshot))
    if len(version) != 64:
        version = canonical_sha256(snapshot)
    destination = directory / f"{version}.json"
    if destination.exists():
        current = read_sealed_json(destination)
        unsigned = _normalized_unsigned(current)
        if unsigned != snapshot:
            raise DurableStateError("behavior snapshot version collision")
    else:
        write_sealed_json(destination, snapshot, backup=False)
    if pointer_version != version:
        write_sealed_json(
            pointer_path,
            {
                "schema": "chronovisor.quality-behavior-pointer.v1",
                "snapshot_path": str(destination),
                "version": version,
                "oracle": False,
            },
            backup=True,
        )
    return previous, snapshot


def _metamorphic_signature_ok(signature: Mapping[str, Any]) -> bool:
    expected = canonical_sha256(dict(signature))
    reversed_mapping = {key: signature[key] for key in reversed(list(signature))}
    variants = [
        json.dumps(reversed_mapping, ensure_ascii=False, separators=(",", ":")),
        json.dumps(signature, ensure_ascii=False, indent=2, sort_keys=False),
    ]
    compact = json.dumps(signature, ensure_ascii=False, sort_keys=True)
    split = max(1, len(compact) // 3)
    variants.append("".join([compact[:split], compact[split : split * 2], compact[split * 2 :]]))
    try:
        return all(canonical_sha256(json.loads(value)) == expected for value in variants)
    except (TypeError, ValueError, json.JSONDecodeError):
        return False


def _metrics_by_lane(
    *,
    anchors: list[dict[str, Any]],
    relations: list[dict[str, Any]],
    previous: Mapping[str, Any] | None,
    current: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    current_cases = {
        str(row.get("case_id")): row
        for row in current.get("cases", [])
        if isinstance(row, Mapping) and isinstance(row.get("case_id"), str)
    }
    previous_cases = {
        str(row.get("case_id")): row
        for row in (previous or {}).get("cases", [])
        if isinstance(row, Mapping) and isinstance(row.get("case_id"), str)
    }
    anchor_by_id = {str(row.get("anchor_id")): row for row in anchors}
    lanes: dict[str, dict[str, Any]] = {}
    for anchor in anchors:
        lane = str(anchor.get("lane") or "unknown")
        metrics = lanes.setdefault(
            lane,
            {
                "sample_count": 0,
                "anchor_comparable": 0,
                "anchor_matches": 0,
                "metamorphic_comparable": 0,
                "metamorphic_passes": 0,
                "behavior_comparable": 0,
                "behavior_flips": 0,
            },
        )
        case = current_cases.get(str(anchor.get("case_id")))
        if case is None:
            continue
        metrics["sample_count"] += 1
        metrics["anchor_comparable"] += 1
        if (
            case.get("actual_signature_sha256")
            == anchor.get("expected_signature_sha256")
            and case.get("actual_decision") == anchor.get("expected_decision")
            and case.get("actual_effect") == anchor.get("expected_effect")
        ):
            metrics["anchor_matches"] += 1
        prior = previous_cases.get(str(anchor.get("case_id")))
        if prior is not None:
            metrics["behavior_comparable"] += 1
            if prior.get("actual_signature_sha256") != case.get(
                "actual_signature_sha256"
            ):
                metrics["behavior_flips"] += 1
    artifact_cases = current.get("cases", [])
    actual_signatures: dict[str, Mapping[str, Any]] = {}
    # The slim behavior snapshot intentionally stores no semantic payload.  The
    # caller attaches signatures only for in-memory metamorphic checks.
    if isinstance(current.get("_actual_signatures"), Mapping):
        actual_signatures = dict(current["_actual_signatures"])
    for relation in relations:
        anchor = anchor_by_id.get(str(relation.get("anchor_id")))
        if anchor is None:
            continue
        lane = str(relation.get("lane") or anchor.get("lane") or "unknown")
        metrics = lanes.setdefault(lane, {})
        signature = actual_signatures.get(str(relation.get("case_id")))
        if not isinstance(signature, Mapping):
            continue
        metrics["metamorphic_comparable"] = int(
            metrics.get("metamorphic_comparable") or 0
        ) + 1
        if _metamorphic_signature_ok(signature):
            metrics["metamorphic_passes"] = int(
                metrics.get("metamorphic_passes") or 0
            ) + 1
    for lane, metrics in lanes.items():
        anchor_total = int(metrics.get("anchor_comparable") or 0)
        metamorphic_total = int(metrics.get("metamorphic_comparable") or 0)
        behavior_total = int(metrics.get("behavior_comparable") or 0)
        metrics["anchor_match_rate"] = (
            float(metrics.get("anchor_matches") or 0) / anchor_total
            if anchor_total
            else 0.0
        )
        metrics["metamorphic_pass_rate"] = (
            float(metrics.get("metamorphic_passes") or 0) / metamorphic_total
            if metamorphic_total
            else 0.0
        )
        metrics["flip_rate"] = (
            float(metrics.get("behavior_flips") or 0) / behavior_total
            if behavior_total
            else 0.0
        )
    return lanes


def _utc(value: datetime | None = None) -> datetime:
    current = value or datetime.now(timezone.utc)
    if current.tzinfo is None:
        return current.replace(tzinfo=timezone.utc)
    return current.astimezone(timezone.utc)


def lane_state_path(root: Path, lane: str) -> Path:
    safe = "".join(char if char.isalnum() or char in "_.-" else "_" for char in lane)
    return root / "lanes" / f"{safe}.json"


def load_lane_state(root: Path, lane: str) -> dict[str, Any]:
    path = lane_state_path(root, lane)
    try:
        return read_sealed_json(path, recover_backup=True)
    except DurableStateError:
        if path.exists() or path.with_name(f"{path.name}.bak").exists():
            return {
                "schema_version": QUALITY_POLICY_VERSION,
                "lane": lane,
                "status": "frozen",
                "state_integrity": "invalid",
                "frontier_allowed": False,
            }
        return {
            "schema_version": QUALITY_POLICY_VERSION,
            "lane": lane,
            "status": "active",
            "consecutive_failures": 0,
            "consecutive_recoveries": 0,
            "incidents": [],
            "frontier_allowed": False,
        }


def lane_is_frozen(root: Path, lane: str) -> bool:
    return load_lane_state(root, lane).get("status") == "frozen"


def _metric_failure(
    metrics: Mapping[str, Any],
    thresholds: QualityThresholds,
) -> tuple[bool, list[str]]:
    sample_count = int(metrics.get("sample_count") or 0)
    if sample_count < thresholds.minimum_samples:
        return True, ["insufficient_samples"]
    reasons: list[str] = []
    if float(metrics.get("anchor_match_rate") or 0.0) < (
        thresholds.minimum_anchor_match_rate
    ):
        reasons.append("anchor_match_rate")
    if float(metrics.get("metamorphic_pass_rate") or 0.0) < (
        thresholds.minimum_metamorphic_pass_rate
    ):
        reasons.append("metamorphic_pass_rate")
    if float(metrics.get("flip_rate") or 0.0) > thresholds.maximum_flip_rate:
        reasons.append("flip_rate")
    return bool(reasons), reasons


def _parse_incident_time(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value).astimezone(timezone.utc)
    except (TypeError, ValueError):
        return None


def evaluate_quality(
    *,
    root: Path,
    lane: str,
    metrics: Mapping[str, Any],
    thresholds: QualityThresholds | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    policy = thresholds or QualityThresholds()
    current_time = _utc(now)
    path = lane_state_path(root, lane)
    lock = path.with_name(f"{path.name}.lock")
    with file_lock(lock, exclusive=True):
        state = load_lane_state(root, lane)
        failed, reasons = _metric_failure(metrics, policy)
        failures = int(state.get("consecutive_failures") or 0)
        recoveries = int(state.get("consecutive_recoveries") or 0)
        if failed:
            failures += 1
            recoveries = 0
        else:
            failures = 0
            recoveries += 1
        state.update(
            {
                "schema_version": QUALITY_POLICY_VERSION,
                "lane": lane,
                "thresholds": asdict(policy),
                "latest_metrics": dict(metrics),
                "latest_reasons": reasons,
                "consecutive_failures": failures,
                "consecutive_recoveries": recoveries,
                "updated_at": current_time.isoformat(timespec="seconds"),
                "frontier_allowed": False,
            }
        )
        incidents = state.get("incidents")
        incidents = list(incidents) if isinstance(incidents, list) else []
        recent_cutoff = current_time - timedelta(days=1)
        recent = [
            row
            for row in incidents
            if isinstance(row, dict)
            and (_parse_incident_time(row.get("ts")) or datetime.min.replace(tzinfo=timezone.utc))
            >= recent_cutoff
        ]
        cooldown_until = state.get("cooldown_until")
        try:
            cooldown_active = bool(
                isinstance(cooldown_until, str)
                and datetime.fromisoformat(cooldown_until).astimezone(timezone.utc)
                > current_time
            )
        except ValueError:
            cooldown_active = False
        if (
            failed
            and failures >= policy.trip_samples
            and not cooldown_active
            and len(recent) < policy.incident_budget_per_day
        ):
            dedupe_key = hashlib.sha256(
                canonical_bytes(
                    {
                        "lane": lane,
                        "policy_version": policy.version,
                        "reasons": reasons,
                        "metrics_epoch": metrics.get("epoch"),
                    }
                )
            ).hexdigest()
            if not any(row.get("dedupe_key") == dedupe_key for row in recent):
                incident = {
                    "ts": current_time.isoformat(timespec="seconds"),
                    "dedupe_key": dedupe_key,
                    "reasons": reasons,
                    "action": "freeze_rollback_shadow_replay",
                    "frontier_allowed": False,
                }
                incidents.append(incident)
                state["status"] = "frozen"
                state["containment"] = {
                    "lane_frozen": True,
                    "last_known_good_rollback_required": True,
                    "local_shadow_replay_required": True,
                    "frontier_semantic_audit": "prohibited",
                }
                state["cooldown_until"] = (
                    current_time + timedelta(seconds=policy.cooldown_seconds)
                ).isoformat(timespec="seconds")
        elif (
            not failed
            and state.get("status") == "frozen"
            and recoveries >= policy.recovery_samples
            and state.get("rollback_verified") is True
            and state.get("shadow_replay_verified") is True
        ):
            state["status"] = "active"
            state["containment"] = {"lane_frozen": False}
        state["incidents"] = incidents[-100:]
        return write_sealed_json(path, state, backup=True)


def register_last_known_good(
    *,
    root: Path,
    lane: str,
    authority_artifact: Path,
    expected_authority_sha256: str | None = None,
) -> dict[str, Any]:
    raw = authority_artifact.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if (
        expected_authority_sha256 is not None
        and digest != expected_authority_sha256
    ):
        raise RuntimeError("quality authority changed before LKG publication")
    destination = root / "last-known-good" / lane / f"{digest}.json"
    if destination.exists() and destination.read_bytes() != raw:
        raise RuntimeError("last-known-good content-address collision")
    if not destination.exists():
        atomic_write_bytes(destination, raw, backup=False)
    pointer = root / "last-known-good" / lane / "current.json"
    try:
        current = read_sealed_json(pointer)
        if (
            current.get("artifact_sha256") == digest
            and current.get("artifact_path") == str(destination)
        ):
            return current
    except DurableStateError:
        pass
    return write_sealed_json(
        pointer,
        {
            "schema_version": 1,
            "lane": lane,
            "artifact_path": str(destination),
            "artifact_sha256": digest,
        },
        backup=True,
    )


def rollback_last_known_good(
    *,
    root: Path,
    lane: str,
    active_artifact: Path,
    expected_active_sha256: str,
) -> dict[str, Any]:
    pointer = read_sealed_json(root / "last-known-good" / lane / "current.json")
    lkg_path = Path(str(pointer["artifact_path"]))
    replacement = lkg_path.read_bytes()
    if hashlib.sha256(replacement).hexdigest() != pointer.get("artifact_sha256"):
        raise RuntimeError("last-known-good artifact digest mismatch")
    current = active_artifact.read_bytes()
    current_digest = hashlib.sha256(current).hexdigest()
    if current_digest != expected_active_sha256:
        raise RuntimeError("active authority changed before rollback")
    atomic_write_bytes(active_artifact, replacement, backup=True)
    return {
        "status": "rolled_back",
        "lane": lane,
        "preimage_sha256": current_digest,
        "postimage_sha256": hashlib.sha256(replacement).hexdigest(),
        "frontier_calls": 0,
    }


def _mark_containment(
    *,
    root: Path,
    lane: str,
    rollback: Mapping[str, Any],
    shadow_replay_verified: bool,
) -> dict[str, Any]:
    path = lane_state_path(root, lane)
    lock = path.with_name(f"{path.name}.lock")
    with file_lock(lock, exclusive=True):
        state = load_lane_state(root, lane)
        state["rollback_verified"] = rollback.get("status") in {
            "rolled_back",
            "already_last_known_good",
        }
        state["shadow_replay_verified"] = bool(shadow_replay_verified)
        state["containment_result"] = {
            "rollback": dict(rollback),
            "local_shadow_replay": {
                "status": "passed" if shadow_replay_verified else "failed",
                "model_invocations": 0,
                "frontier_calls": 0,
                "source": "sealed_last_known_good_artifact_replay",
            },
            "frontier_calls": 0,
        }
        return write_sealed_json(path, state, backup=True)


def _artifact_actual_signatures(
    artifact: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    signatures: dict[str, Mapping[str, Any]] = {}
    cases = artifact.get("cases")
    if not isinstance(cases, list):
        return signatures
    for row in cases:
        if not isinstance(row, Mapping):
            continue
        case_id = row.get("case_id")
        signature = row.get("actual_signature")
        if isinstance(case_id, str) and isinstance(signature, Mapping):
            signatures[case_id] = signature
    return signatures


def _shadow_replay_matches_anchors(
    artifact: Mapping[str, Any],
    anchors: list[dict[str, Any]],
    *,
    lanes: set[str],
) -> bool:
    signatures = _artifact_actual_signatures(artifact)
    cases = {
        str(row.get("case_id")): row
        for row in artifact.get("cases", [])
        if isinstance(row, Mapping) and isinstance(row.get("case_id"), str)
    }
    selected = [row for row in anchors if str(row.get("lane")) in lanes]
    if not selected:
        return False
    for anchor in selected:
        case_id = str(anchor.get("case_id"))
        case = cases.get(case_id)
        signature = signatures.get(case_id)
        if (
            case is None
            or signature is None
            or canonical_sha256(dict(signature))
            != anchor.get("expected_signature_sha256")
            or case.get("actual_decision") != anchor.get("expected_decision")
            or case.get("actual_effect") != anchor.get("expected_effect")
            or not _metamorphic_signature_ok(signature)
        ):
            return False
    return True


def run_quality_probe(
    *,
    root: Path,
    adoption_artifact: Path,
    thresholds: QualityThresholds | None = None,
    bootstrap_objective_anchors: bool = True,
    now: datetime | None = None,
    artifact_validator: Callable[[Path], Any] | None = None,
) -> dict[str, Any]:
    """Measure a sealed local adoption artifact and contain drift locally.

    The probe never starts a model.  It checks immutable objective anchors,
    deterministic metamorphic encodings, and versioned behavior deltas.  A
    tripped lane is rolled back to a stable last-known-good artifact and the
    same sealed artifact is replayed locally as a zero-inference shadow check.
    """

    from chronovisor.local_model_eval import validate_adoption_evidence

    policy = thresholds or QualityThresholds()
    try:
        artifact_raw = adoption_artifact.read_bytes()
        artifact = json.loads(artifact_raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DurableStateError(f"cannot read quality artifact: {exc}") from exc
    if not isinstance(artifact, dict):
        raise DurableStateError("quality artifact must be a JSON object")
    if (
        artifact.get("status") != "complete"
        or artifact.get("adopted") is not True
        or not validate_adoption_evidence(artifact)
    ):
        raise DurableStateError("quality artifact is not complete adopted evidence")
    if artifact_validator is None:
        # Reuse the production validator so a self-consistent but stale or
        # weakened artifact cannot become either quality evidence or LKG.
        from chronovisor.decision_router import _validated_adoption_artifact

        artifact_validator = _validated_adoption_artifact
    try:
        artifact_validator(adoption_artifact)
    except Exception as exc:
        raise DurableStateError(
            f"quality artifact failed production validation: {exc}"
        ) from exc
    if bootstrap_objective_anchors:
        corpus = bootstrap_objective_corpus(root=root, artifact=artifact)
    else:
        corpus = {"behavior_promoted_to_anchor": False, "frontier_calls": 0}
    layout = corpus_layout(root)
    anchors = _read_sealed_jsonl(layout["immutable_anchor"])
    relations = _read_sealed_jsonl(layout["metamorphic"])
    previous, current = _publish_behavior_snapshot(root, artifact)
    current["_actual_signatures"] = _artifact_actual_signatures(artifact)
    metrics_by_lane = _metrics_by_lane(
        anchors=anchors,
        relations=relations,
        previous=previous,
        current=current,
    )
    epoch = str(artifact.get("evidence_sha256") or "unknown")
    states: dict[str, dict[str, Any]] = {}
    failed_lanes: set[str] = set()
    for lane, metrics in sorted(metrics_by_lane.items()):
        metrics["epoch"] = epoch
        failed, _reasons = _metric_failure(metrics, policy)
        if failed:
            failed_lanes.add(lane)
        states[lane] = evaluate_quality(
            root=root,
            lane=lane,
            metrics=metrics,
            thresholds=policy,
            now=now,
        )

    try:
        current_raw = adoption_artifact.read_bytes()
    except OSError as exc:
        raise DurableStateError("quality artifact disappeared during probe") from exc
    if current_raw != artifact_raw:
        raise DurableStateError("quality artifact changed during probe")
    current_sha256 = hashlib.sha256(artifact_raw).hexdigest()
    for lane, state in states.items():
        if (
            lane not in failed_lanes
            and state.get("status") == "active"
            and int(state.get("consecutive_recoveries") or 0)
            >= policy.recovery_samples
        ):
            register_last_known_good(
                root=root,
                lane=lane,
                authority_artifact=adoption_artifact,
                expected_authority_sha256=current_sha256,
            )

    frozen = {lane for lane, state in states.items() if state.get("status") == "frozen"}
    rollback: dict[str, Any] | None = None
    shadow_verified = False
    if frozen:
        pointer: dict[str, Any] | None = None
        pointer_lane: str | None = None
        for lane in sorted(frozen):
            try:
                pointer = read_sealed_json(
                    root / "last-known-good" / lane / "current.json"
                )
                pointer_lane = lane
                break
            except DurableStateError:
                continue
        if pointer is None or pointer_lane is None:
            rollback = {
                "status": "missing_last_known_good",
                "frontier_calls": 0,
            }
        else:
            lkg_path = Path(str(pointer["artifact_path"]))
            lkg_sha256 = str(pointer.get("artifact_sha256") or "")
            if current_sha256 == lkg_sha256:
                rollback = {
                    "status": "already_last_known_good",
                    "lane": pointer_lane,
                    "postimage_sha256": current_sha256,
                    "frontier_calls": 0,
                }
            else:
                rollback = rollback_last_known_good(
                    root=root,
                    lane=pointer_lane,
                    active_artifact=adoption_artifact,
                    expected_active_sha256=current_sha256,
                )
            try:
                lkg_artifact = json.loads(lkg_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                lkg_artifact = {}
            shadow_verified = isinstance(lkg_artifact, dict) and (
                _shadow_replay_matches_anchors(
                    lkg_artifact,
                    anchors,
                    lanes=frozen,
                )
            )
        for lane in frozen:
            states[lane] = _mark_containment(
                root=root,
                lane=lane,
                rollback=rollback,
                shadow_replay_verified=shadow_verified,
            )
    probe = {
        "schema": "chronovisor.quality-probe.v1",
        "observed_at": _utc(now).isoformat(timespec="seconds"),
        "artifact_path": str(adoption_artifact),
        "artifact_sha256": current_sha256,
        "artifact_evidence_sha256": epoch,
        "corpus": corpus,
        "metrics_by_lane": metrics_by_lane,
        "lane_status": {
            lane: {
                "status": state.get("status"),
                "latest_reasons": state.get("latest_reasons"),
                "rollback_verified": state.get("rollback_verified"),
                "shadow_replay_verified": state.get("shadow_replay_verified"),
            }
            for lane, state in states.items()
        },
        "frozen_lanes": sorted(frozen),
        "rollback": rollback,
        "behavior_promoted_to_anchor": False,
        "frontier_calls": 0,
    }
    write_sealed_json(root / "probe-latest.json", probe, backup=True)
    return probe


def quality_snapshot(root: Path) -> dict[str, Any]:
    lanes: dict[str, Any] = {}
    for path in sorted((root / "lanes").glob("*.json")):
        try:
            state = read_sealed_json(path)
        except DurableStateError:
            lanes[path.stem] = {"status": "invalid"}
            continue
        lanes[str(state.get("lane") or path.stem)] = {
            "status": state.get("status"),
            "latest_metrics": state.get("latest_metrics"),
            "latest_reasons": state.get("latest_reasons"),
            "containment": state.get("containment"),
            "frontier_allowed": False,
        }
    layout = corpus_layout(root)
    try:
        corpus = {
            "immutable_anchor": len(_read_sealed_jsonl(layout["immutable_anchor"])),
            "behavior_snapshots": len(
                [
                    path
                    for path in layout["behavior_snapshots"].glob("*.json")
                    if path.name != "latest.json"
                ]
            ),
            "metamorphic": len(_read_sealed_jsonl(layout["metamorphic"])),
            "behavior_promoted_to_anchor": False,
            "status": "ok",
        }
    except DurableStateError as exc:
        corpus = {"status": "invalid", "error": str(exc)}
    try:
        latest_probe = read_sealed_json(root / "probe-latest.json")
        probe = {
            "status": "ok",
            "observed_at": latest_probe.get("observed_at"),
            "artifact_sha256": latest_probe.get("artifact_sha256"),
            "frozen_lanes": latest_probe.get("frozen_lanes"),
            "frontier_calls": 0,
        }
    except DurableStateError:
        probe = {"status": "missing"}
    return {
        "schema_version": QUALITY_POLICY_VERSION,
        "lanes": lanes,
        "frozen": sum(row.get("status") == "frozen" for row in lanes.values()),
        "corpus": corpus,
        "probe": probe,
        "frontier_semantic_audit_allowed": False,
    }
