"""Content-addressed classification bundles and fail-closed resolution."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from chronovisor.classification.classification import ClassificationError
from chronovisor.core.durable_state import (
    DurableStateError,
    read_sealed_json,
    write_sealed_json,
)
from chronovisor.core.timeutil import utc_iso_milliseconds as _now
from chronovisor.lab.classification_fixture_set import (
    DISABLED_BASELINE_SCHEMA,
    sha256_bytes,
    sha256_file,
)

CANDIDATE_BUNDLE_SCHEMA = "chronovisor.classification-candidate-bundle.v1"
ADOPTION_PAYLOAD_SCHEMA = "chronovisor.classification-adoption-payload.v1"
AUTHORITY_SCHEMA = "chronovisor.classification-authority.v1"
ADOPTED_MANIFEST_SCHEMA = "chronovisor.classification-adopted-manifest.v1"
POINTER_SCHEMA = "chronovisor.classification-pointer.v1"
SUPERVISOR_PROBE_SCHEMA = "chronovisor.classification-authority-probe.v1"




def _digest(payload: Mapping[str, Any]) -> str:
    return sha256_bytes(
        json.dumps(
            dict(payload),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )


def digest_dag(
    *,
    udc_sha256: str,
    source_sha256: list[str],
    crosswalk_sha256: list[str],
    index_sha256: list[str],
    provider_code_sha256: str,
    evidence_template_sha256: str,
    model_policy: Mapping[str, Any],
    run_config: Mapping[str, Any],
    input_sha256: str,
    fixture_set_sha256: str,
    chosen_thresholds: Mapping[str, Any],
    dev_result_sha256: str,
    holdout_result_sha256: str,
    metric_schema: str,
    observed_metrics: Mapping[str, Any],
    evaluation_status: str,
) -> dict[str, str]:
    evidence_root = _digest(
        {
            "udc_sha256": udc_sha256,
            "source_sha256": sorted(source_sha256),
            "crosswalk_sha256": sorted(crosswalk_sha256),
            "index_sha256": sorted(index_sha256),
            "provider_code_sha256": provider_code_sha256,
            "evidence_template_sha256": evidence_template_sha256,
        }
    )
    policy_digest = _digest(dict(model_policy))
    execution_digest = _digest(
        {
            "evidence_root": evidence_root,
            "policy_digest": policy_digest,
            "run_config": dict(run_config),
            "input_sha256": input_sha256,
        }
    )
    calibration_input_digest = _digest(
        {
            "fixture_set_sha256": fixture_set_sha256,
            "evidence_root": evidence_root,
            "policy_digest": policy_digest,
            "chosen_thresholds": dict(chosen_thresholds),
        }
    )
    evaluation_result_digest = _digest(
        {
            "calibration_input_digest": calibration_input_digest,
            "dev_result_sha256": dev_result_sha256,
            "holdout_result_sha256": holdout_result_sha256,
            "metric_schema": metric_schema,
            "observed_metrics": dict(observed_metrics),
            "evaluation_status": evaluation_status,
        }
    )
    return {
        "evidence_root": evidence_root,
        "policy_digest": policy_digest,
        "execution_digest": execution_digest,
        "calibration_input_digest": calibration_input_digest,
        "evaluation_result_digest": evaluation_result_digest,
    }


def create_candidate_bundle(
    output_path: Path,
    *,
    dag: Mapping[str, str],
    evaluation_path: Path,
    provider_manifest_path: Path,
    fixture_manifest_path: Path,
    storage_manifest: Mapping[str, Any],
    attributions: list[Mapping[str, Any]],
    run_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    evaluation = read_sealed_json(evaluation_path)
    if evaluation.get("status") != "passed":
        raise ClassificationError("cannot bundle a failed evaluation")
    payload = {
        "schema": CANDIDATE_BUNDLE_SCHEMA,
        "created_at": _now(),
        "dag": dict(dag),
        "evaluation_path": str(evaluation_path),
        "evaluation_sha256": sha256_file(evaluation_path),
        "provider_manifest_path": str(provider_manifest_path),
        "provider_manifest_sha256": sha256_file(provider_manifest_path),
        "fixture_manifest_path": str(fixture_manifest_path),
        "fixture_manifest_sha256": sha256_file(fixture_manifest_path),
        "storage_manifest": dict(storage_manifest),
        "attributions": [dict(value) for value in attributions],
        "run_config": dict(run_config or {}),
        "mutation_capability": False,
        "status": "inactive-vnext",
    }
    payload["candidate_bundle_digest"] = _digest(
        {
            "evidence_root": dag["evidence_root"],
            "policy_digest": dag["policy_digest"],
            "evaluation_result_digest": dag["evaluation_result_digest"],
            "payload": payload,
        }
    )
    write_sealed_json(output_path, payload, backup=True)
    return payload


def create_adopted_manifest(
    output_path: Path,
    *,
    candidate_bundle_path: Path,
    actor: str,
    decision: str,
    parent_phase4_receipt: Path,
    adoption_policy: Mapping[str, Any],
) -> dict[str, Any]:
    candidate = read_sealed_json(candidate_bundle_path)
    if candidate.get("schema") != CANDIDATE_BUNDLE_SCHEMA:
        raise ClassificationError("candidate bundle schema mismatch")
    if candidate.get("status") != "inactive-vnext":
        raise ClassificationError("candidate bundle is not inactive vNext")
    if decision != "adopt":
        raise ClassificationError("explicit adoption decision is required")
    if not parent_phase4_receipt.is_file():
        raise ClassificationError("parent Phase 4 receipt is missing")
    adoption_payload = {
        "schema": ADOPTION_PAYLOAD_SCHEMA,
        "candidate_bundle_digest": candidate["candidate_bundle_digest"],
        "adoption_policy": dict(adoption_policy),
        "decision": decision,
        "actor": actor,
        "decided_at": _now(),
        "parent_phase4_receipt": str(parent_phase4_receipt),
        "parent_phase4_receipt_sha256": sha256_file(parent_phase4_receipt),
        "mutation_capability": False,
    }
    adoption_payload["adoption_payload_digest"] = _digest(adoption_payload)
    authority = {
        "schema": AUTHORITY_SCHEMA,
        "candidate_bundle_digest": candidate["candidate_bundle_digest"],
        "adoption_payload_digest": adoption_payload["adoption_payload_digest"],
        "mode": "decision-only/canary",
        "mutation_capability": False,
    }
    authority["authority_digest"] = _digest(authority)
    manifest = {
        "schema": ADOPTED_MANIFEST_SCHEMA,
        "candidate_bundle_path": str(candidate_bundle_path),
        "candidate_bundle_sha256": sha256_file(candidate_bundle_path),
        "candidate_bundle_digest": candidate["candidate_bundle_digest"],
        "adoption_payload": adoption_payload,
        "authority": authority,
        "mutation_capability": False,
        "mode": "decision-only/canary",
    }
    manifest["adopted_bundle_manifest_digest"] = _digest(
        {
            "candidate_bundle_digest": candidate["candidate_bundle_digest"],
            "authority_digest": authority["authority_digest"],
        }
    )
    write_sealed_json(output_path, manifest, backup=True)
    return manifest


def pointer_paths(root: Path) -> tuple[Path, Path, Path]:
    base = root / "classification" / "authority"
    return base / "active.json", base / "previous.json", base / "mutation.json"


def resolve_authority(root: Path) -> dict[str, Any]:
    active_path, _previous_path, mutation_path = pointer_paths(root)
    if not active_path.exists():
        return {
            "status": "error",
            "reason": "missing_active_pointer",
            "classification_authority_active": False,
            "mutation_capability": False,
        }
    try:
        pointer = read_sealed_json(active_path)
    except (DurableStateError, OSError, json.JSONDecodeError, ValueError) as exc:
        return {
            "status": "error",
            "reason": "corrupt_active_pointer",
            "detail": str(exc),
            "classification_authority_active": False,
            "mutation_capability": False,
        }
    if pointer.get("schema") != POINTER_SCHEMA:
        return {
            "status": "error",
            "reason": "unsupported_active_pointer",
            "classification_authority_active": False,
            "mutation_capability": False,
        }
    target_path = Path(str(pointer.get("target_path") or ""))
    if not target_path.is_file():
        return {
            "status": "error",
            "reason": "missing_pointer_target",
            "classification_authority_active": False,
            "mutation_capability": False,
        }
    try:
        target = read_sealed_json(target_path)
    except (DurableStateError, OSError, json.JSONDecodeError, ValueError) as exc:
        return {
            "status": "error",
            "reason": "corrupt_pointer_target",
            "detail": str(exc),
            "classification_authority_active": False,
            "mutation_capability": False,
        }
    if sha256_file(target_path) != pointer.get("target_sha256"):
        return {
            "status": "error",
            "reason": "pointer_target_checksum_mismatch",
            "classification_authority_active": False,
            "mutation_capability": False,
        }
    if target.get("schema") == DISABLED_BASELINE_SCHEMA:
        return {
            "status": "disabled",
            "reason": "intentional_disabled_baseline",
            "classification_authority_active": False,
            "candidate_behavior": "A0-production-replay",
            "mutation_capability": False,
            "target": target,
        }
    if target.get("schema") != ADOPTED_MANIFEST_SCHEMA:
        return {
            "status": "error",
            "reason": "unsupported_pointer_target",
            "classification_authority_active": False,
            "mutation_capability": False,
        }
    mutation = read_sealed_json(mutation_path) if mutation_path.exists() else {}
    mutation_enabled = bool(mutation.get("enabled") is True)
    if target.get("mutation_capability") is not False:
        return {
            "status": "error",
            "reason": "decision_only_manifest_claims_mutation",
            "classification_authority_active": False,
            "mutation_capability": False,
        }
    return {
        "status": "active",
        "reason": "adopted_decision_only",
        "classification_authority_active": True,
        "candidate_behavior": "library-evidence-vnext",
        "mutation_capability": mutation_enabled,
        "target": target,
    }


def probe_decision_only_authority(
    root: Path,
    *,
    expected_manifest_path: Path,
) -> dict[str, Any]:
    """Validate the complete decision-only authority chain without mutating it."""

    resolved = resolve_authority(root)
    expected_sha256 = (
        sha256_file(expected_manifest_path)
        if expected_manifest_path.is_file()
        else None
    )
    try:
        expected = (
            read_sealed_json(expected_manifest_path)
            if expected_sha256 is not None
            else {}
        )
    except (DurableStateError, OSError, json.JSONDecodeError, ValueError):
        expected = {}
    target = resolved.get("target")
    target = target if isinstance(target, Mapping) else {}
    gates = {
        "resolver_active": resolved.get("status") == "active",
        "expected_manifest_present": expected_sha256 is not None,
        "expected_manifest_selected": (
            resolved.get("status") == "active"
            and target.get("schema") == ADOPTED_MANIFEST_SCHEMA
            and bool(expected.get("adopted_bundle_manifest_digest"))
            and target.get("adopted_bundle_manifest_digest")
            == expected.get("adopted_bundle_manifest_digest")
        ),
        "mutation_disabled": resolved.get("mutation_capability") is False,
        "decision_only_mode": target.get("mode") == "decision-only/canary",
        "target_declares_no_mutation": target.get("mutation_capability") is False,
    }
    return {
        "schema": SUPERVISOR_PROBE_SCHEMA,
        "status": "passed" if all(gates.values()) else "critical-breach",
        "checked_at": _now(),
        "expected_manifest_path": str(expected_manifest_path),
        "expected_manifest_sha256": expected_sha256,
        "resolved_status": resolved.get("status"),
        "resolved_reason": resolved.get("reason"),
        "authority_digest": (target.get("authority") or {}).get("authority_digest"),
        "gates": gates,
    }


def _write_pointer(path: Path, payload: Mapping[str, Any]) -> None:
    write_sealed_json(path, payload, backup=True)


def activate_decision_only(root: Path, *, target_path: Path) -> dict[str, Any]:
    target = read_sealed_json(target_path)
    if target.get("schema") not in {
        ADOPTED_MANIFEST_SCHEMA,
        DISABLED_BASELINE_SCHEMA,
    }:
        raise ClassificationError("activation target is not adopted or disabled")
    if target.get("mutation_capability") is not False:
        raise ClassificationError("decision-only activation cannot enable mutation")
    active_path, previous_path, mutation_path = pointer_paths(root)
    mutation_path.parent.mkdir(parents=True, exist_ok=True)
    write_sealed_json(
        mutation_path,
        {
            "schema": "chronovisor.classification-mutation-capability.v1",
            "enabled": False,
            "updated_at": _now(),
            "reason": "decision-only-activation",
        },
        backup=True,
    )
    if active_path.exists():
        previous = read_sealed_json(active_path)
        _write_pointer(previous_path, previous)
    pointer = {
        "schema": POINTER_SCHEMA,
        "activated_at": _now(),
        "target_path": str(target_path),
        "target_sha256": sha256_file(target_path),
        "mode": "decision-only/canary",
        "mutation_capability": False,
    }
    _write_pointer(active_path, pointer)
    resolved = resolve_authority(root)
    if resolved["status"] not in {"active", "disabled"}:
        raise ClassificationError(f"activation did not resolve: {resolved}")
    return resolved


def rollback_authority(root: Path) -> dict[str, Any]:
    active_path, previous_path, mutation_path = pointer_paths(root)
    write_sealed_json(
        mutation_path,
        {
            "schema": "chronovisor.classification-mutation-capability.v1",
            "enabled": False,
            "updated_at": _now(),
            "reason": "rollback-first-step",
        },
        backup=True,
    )
    if not previous_path.is_file():
        raise ClassificationError("previous authority pointer is missing")
    previous = read_sealed_json(previous_path)
    _write_pointer(active_path, previous)
    return resolve_authority(root)
