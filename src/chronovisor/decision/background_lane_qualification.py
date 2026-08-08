"""Lane capability seals evaluated beneath the ordinary authority boundary.

The first qualification for a lane cannot call :class:`DecisionRouter` in its
normal artifact-replay mode: that mode already requires the capability this
module is trying to establish.  The only bootstrap boundary here is the
read-only replay evaluator, pinned to the currently adopted router config and
the five deterministic lane-contract cases.  Normal decision callers do not
gain an authority bypass.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from chronovisor.core.canonical_json import (
    canonical_json_sha256_strict as _strict_sha256,
)
from chronovisor.core.durable_state import (
    DurableStateError,
    canonical_sha256,
    read_sealed_json,
    seal_object,
    sidecar_exclusive_lock,
    write_sealed_json,
)
from chronovisor.core.runtime_config import DecisionRouterConfig
from chronovisor.core.store import CHRONOVISOR_ROOT
from chronovisor.decision.decision_lane_contract_cases import (
    BACKGROUND_LANE_CONTRACT_CASE_VERSION,
    background_decision_lane_contract_case_manifest_sha256,
    background_decision_lane_contract_case_specs,
)
from chronovisor.decision.decision_lane_contracts import lane_contract_sha256
from chronovisor.decision.decision_schema_manifest import (
    background_decision_schemas,
    schema_sha256,
    signature_policy,
)
from chronovisor.decision.local_structured import (
    structured_generation_policy_sha256,
)

BACKGROUND_LANE_QUALIFICATION_VERSION = 3
MetadataProvider = Callable[[Sequence[str]], Mapping[str, Any]]
ReplayEvaluator = Callable[..., dict[str, Any]]


def _metadata(
    models: Sequence[str], provider: MetadataProvider | None = None
) -> dict[str, Any]:
    from chronovisor.decision.local_model_eval import (
        _safe_model_metadata,
        fetch_local_model_metadata,
        validate_model_metadata_identity,
    )

    raw = (provider or fetch_local_model_metadata)(models)
    safe = _safe_model_metadata(raw, models)
    validate_model_metadata_identity(safe, models)
    return safe


def _resolved_adopted_config(
    base_authority: Mapping[str, Any],
) -> DecisionRouterConfig:
    from chronovisor.core.runtime_config import load_decision_router_config
    from chronovisor.decision.decision_router import resolve_router_policy

    resolution = resolve_router_policy(load_decision_router_config())
    if resolution.audit_record() != base_authority.get("router"):
        raise ValueError("adopted router authority changed")
    if resolution.source != "adopted_artifact" or resolution.error is not None:
        raise ValueError("adopted router config is unavailable")
    return resolution.config


def _qualification_rows(lane: str) -> list[dict[str, Any]]:
    from chronovisor.decision.decision_router import (
        decision_request_fingerprint_sha256,
    )
    from chronovisor.decision.decision_schema_manifest import (
        background_decision_schemas,
    )

    schemas = background_decision_schemas()
    case_manifest_sha = background_decision_lane_contract_case_manifest_sha256()
    rows: list[dict[str, Any]] = []
    for case in background_decision_lane_contract_case_specs():
        if case.lane != lane:
            continue
        schema = dict(schemas[case.schema_name])
        contract_id = (
            f"background-contract-v{BACKGROUND_LANE_CONTRACT_CASE_VERSION}:"
            f"{lane}:{case.ordinal}"
        )
        rows.append(
            {
                "timestamp": "2026-08-02T00:00:00Z",
                "source": "background_lane_qualification_contract_v1",
                "contract_version": BACKGROUND_LANE_CONTRACT_CASE_VERSION,
                "contract_id": contract_id,
                "decision_lane": lane,
                "lane_contract_sha256": lane_contract_sha256(lane),
                "lane_contract_case_manifest_sha256": case_manifest_sha,
                "lane_contract_effect": case.expected_effect,
                "role": lane,
                "model": "deterministic-background-contract",
                "effort": "contract",
                "prompt": case.prompt,
                "system": case.system,
                "prompt_truncated": False,
                "prompt_original_chars": len(case.prompt),
                "system_original_chars": (
                    len(case.system) if case.system is not None else 0
                ),
                "schema": schema,
                "expected": copy.deepcopy(case.expected),
                "effective_request_sha256": decision_request_fingerprint_sha256(
                    prompt=case.prompt,
                    schema=schema,
                    system=case.system,
                    decision_lane=lane,
                ),
                "evidence_provenance": {
                    "kind": "deterministic_background_lane_contract",
                    "manifest_sha256": case_manifest_sha,
                },
                "latency_seconds": 0.0,
            }
        )
    rows.sort(key=lambda row: str(row.get("contract_id") or ""))
    if len(rows) != 5:
        raise ValueError("background lane qualification requires exact five cases")
    if len({str(row.get("contract_id") or "") for row in rows}) != 5:
        raise ValueError("background lane qualification cases are duplicated")
    return rows


def _source_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return (
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
            for row in rows
        )
    ).encode("utf-8")


def _write_create_once(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != raw:
            raise DurableStateError("qualification source epoch conflicts")
        return
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        try:
            path.unlink()
        except OSError:
            pass
        raise


def _qualification_identity(
    lane: str,
    *,
    base_authority: Mapping[str, Any],
    config: DecisionRouterConfig,
    metadata: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    policy = base_authority.get("policy")
    schema_name = policy.get("schema_name") if isinstance(policy, Mapping) else None
    schema = background_decision_schemas().get(str(schema_name or ""))
    if schema is None:
        raise ValueError("background lane qualification schema is missing")
    source_raw = _source_bytes(rows)
    return {
        "version": BACKGROUND_LANE_QUALIFICATION_VERSION,
        "lane": lane,
        "base_authority_sha256": canonical_sha256(base_authority),
        "adoption_artifact_sha256": base_authority.get("router", {}).get(
            "artifact_sha256"
        ),
        "models": list(base_authority.get("router", {}).get("models", [])),
        "config_sha256": _strict_sha256(asdict(config)),
        "model_metadata": copy.deepcopy(dict(metadata)),
        "model_metadata_sha256": _strict_sha256(metadata),
        "lane_contract_sha256": lane_contract_sha256(lane),
        "lane_contract_case_manifest_sha256": (
            background_decision_lane_contract_case_manifest_sha256()
        ),
        "schema_name": schema_name,
        "schema_sha256": schema_sha256(schema),
        "signature_policy_sha256": canonical_sha256(signature_policy(schema)),
        "generation_policy_sha256": structured_generation_policy_sha256(),
        "source_sha256": hashlib.sha256(source_raw).hexdigest(),
        "source_rows_sha256": _strict_sha256(list(rows)),
    }


def _paths(
    root: Path, lane: str, identity_sha: str
) -> tuple[Path, Path, Path, Path]:
    directory = root / "runtime" / "decision-qualification"
    stem = f"{lane}-{identity_sha}"
    return (
        directory / f"{stem}.jsonl",
        directory / f"{stem}.evaluation.json",
        directory / f"{stem}.capability.json",
        directory / f"{lane}-active.json",
    )


def _required_schema_manifest(
    identity: Mapping[str, Any],
) -> dict[str, str]:
    return {str(identity["schema_name"]): str(identity["schema_sha256"])}


def _validated_evaluation(
    *,
    source_path: Path,
    evaluation_path: Path,
    identity: Mapping[str, Any],
    config: DecisionRouterConfig,
    metadata_provider: MetadataProvider | None,
    evaluator: ReplayEvaluator | None,
    execute: bool,
) -> dict[str, Any]:
    from chronovisor.decision.local_model_eval import (
        AdoptionThresholds,
        adoption_case_derived_evidence,
        adoption_gate,
        adoption_metrics,
        adoption_result_sha256,
        evaluate_replays,
        load_replay_corpus,
        validate_adoption_evidence,
    )

    if execute:
        run_evaluator = evaluator or evaluate_replays
        artifact = run_evaluator(
            source_path,
            evaluation_path,
            resume=evaluation_path.exists(),
            config=config,
            model_metadata_provider=metadata_provider,
            required_schema_manifest=_required_schema_manifest(identity),
            live_resource_control=True,
        )
    else:
        loaded = json.loads(evaluation_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError("qualification evaluation artifact is not an object")
        artifact = loaded
    corpus = load_replay_corpus(
        source_path,
        required_schema_manifest=_required_schema_manifest(identity),
    )
    cases = artifact.get("cases")
    models = list(identity["models"])
    expected_by_id = {case.case_id: case for case in corpus.cases}
    source = artifact.get("source")
    thresholds = asdict(AdoptionThresholds())
    if (
        artifact.get("status") != "complete"
        or artifact.get("processed_cases") != 5
        or artifact.get("selected_cases") != 5
        or not isinstance(cases, list)
        or len(cases) != 5
        or artifact.get("config_sha256") != identity.get("config_sha256")
        or artifact.get("model_metadata") != identity.get("model_metadata")
        or artifact.get("model_metadata_sha256")
        != identity.get("model_metadata_sha256")
        or artifact.get("thresholds") != thresholds
        or not validate_adoption_evidence(artifact)
        or artifact.get("evaluation_result_sha256")
        != adoption_result_sha256(artifact)
        or not isinstance(source, Mapping)
        or source.get("source_path") != str(source_path.resolve())
        or source.get("source_sha256") != identity.get("source_sha256")
        or source.get("selected_cases") != 5
        or source.get("selected_case_ids_sha256")
        != _strict_sha256([case.case_id for case in corpus.cases])
        or source.get("selected_effective_requests_sha256")
        != _strict_sha256(
            [case.effective_request_sha256 for case in corpus.cases]
        )
    ):
        raise ValueError("qualification evaluation identity is incomplete")
    observed_ids = [str(row.get("case_id") or "") for row in cases]
    if observed_ids != [case.case_id for case in corpus.cases]:
        raise ValueError("qualification evaluation case set changed")
    for row in cases:
        expected = expected_by_id[str(row["case_id"])]
        votes = row.get("votes")
        if not isinstance(votes, list) or len(votes) not in {2, 3}:
            raise ValueError("qualification evaluation vote set is invalid")
        vote_identity = [
            (vote.get("role"), vote.get("model"))
            for vote in votes
            if isinstance(vote, Mapping)
        ]
        if vote_identity != list(
            zip(("primary", "challenger", "tie_break"), models, strict=True)
        )[: len(votes)]:
            raise ValueError("qualification evaluation model identity changed")
        signatures = Counter(
            str(vote.get("signature_value") and _strict_sha256(vote["signature_value"]))
            for vote in votes
            if isinstance(vote, Mapping) and vote.get("vote_valid") is True
        )
        derived = adoption_case_derived_evidence(row)
        if any(row.get(key) != value for key, value in derived.items()):
            raise ValueError("qualification evaluation derived evidence changed")
        if (
            row.get("status") != "agreed"
            or row.get("failure_class") is not None
            or row.get("quarantine_reason") is not None
            or row.get("expected_signature_sha256")
            != expected.expected_signature_sha256
            or row.get("actual_signature_sha256")
            != expected.expected_signature_sha256
            or row.get("expected_signature_match") is not True
            or row.get("expected_effect_match") is not True
            or row.get("unsafe_decision_flip") is not False
            or row.get("invalid_output_accepted") != 0
            or row.get("signature_majority_resolved") is not True
            or signatures[expected.expected_signature_sha256] < 2
        ):
            raise ValueError("qualification evaluation case did not pass")
    expected_metrics = adoption_metrics(
        cases, required_context_buckets=artifact.get("context_buckets", ())
    )
    if (
        artifact.get("metrics") != expected_metrics
        or artifact.get("adoption_gate")
        != adoption_gate(expected_metrics, AdoptionThresholds(), source)
    ):
        raise ValueError("qualification evaluation aggregate evidence changed")
    return artifact


def qualify_background_lane(
    lane: str,
    *,
    chronovisor_root: Path = CHRONOVISOR_ROOT,
    metadata_provider: MetadataProvider | None = None,
    evaluator: ReplayEvaluator | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Evaluate and activate one current five-case capability epoch."""

    from chronovisor.decision.decision_authority import base_semantic_authority

    base, error = base_semantic_authority(lane)
    if error is not None or base is None:
        return {"status": "waiting", "reason": error or "base_authority_missing"}
    try:
        config = _resolved_adopted_config(base)
        rows = _qualification_rows(lane)
        metadata = _metadata(tuple(base["router"]["models"]), metadata_provider)
        identity = _qualification_identity(
            lane,
            base_authority=base,
            config=config,
            metadata=metadata,
            rows=rows,
        )
    except Exception as exc:
        return {
            "status": "waiting",
            "reason": f"qualification_identity:{type(exc).__name__}",
        }
    identity_sha = canonical_sha256(identity)
    source_path, evaluation_path, capability_path, active_path = _paths(
        chronovisor_root, lane, identity_sha
    )
    if dry_run:
        current = validate_current_background_lane_qualification(
            lane,
            base_authority=base,
            chronovisor_root=chronovisor_root,
            metadata_provider=metadata_provider,
            evaluator=evaluator,
        )
        return {
            "status": "passed" if current.get("passed") else "waiting",
            "reason": str(current.get("reason") or "qualification_evaluation_required"),
            "identity_sha256": identity_sha,
        }
    try:
        with sidecar_exclusive_lock(active_path):
            _write_create_once(source_path, _source_bytes(rows))
            artifact = _validated_evaluation(
                source_path=source_path,
                evaluation_path=evaluation_path,
                identity=identity,
                config=config,
                metadata_provider=metadata_provider,
                evaluator=evaluator,
                execute=True,
            )
            current_base, current_error = base_semantic_authority(lane)
            if current_error is not None or current_base != base:
                raise ValueError("base authority changed during qualification")
            capability_payload = {
                "schema_version": BACKGROUND_LANE_QUALIFICATION_VERSION,
                "artifact_kind": "background-lane-capability",
                "status": "passed",
                "identity": identity,
                "identity_sha256": identity_sha,
                "source_path": str(source_path),
                "source_sha256": identity["source_sha256"],
                "evaluation_path": str(evaluation_path),
                "evaluation_artifact_sha256": canonical_sha256(artifact),
                "evaluation_evidence_sha256": artifact.get("evidence_sha256"),
                "evaluation_result_sha256": artifact.get(
                    "evaluation_result_sha256"
                ),
            }
            sealed_capability = seal_object(capability_payload)
            if capability_path.exists():
                if read_sealed_json(capability_path) != sealed_capability:
                    raise DurableStateError("qualification capability epoch conflicts")
            else:
                write_sealed_json(capability_path, capability_payload, backup=False)
            pointer = {
                "schema_version": 1,
                "artifact_kind": "background-lane-capability-active-pointer",
                "lane": lane,
                "identity_sha256": identity_sha,
                "capability_sha256": sealed_capability["seal_sha256"],
                "capability_path": str(capability_path),
            }
            write_sealed_json(active_path, pointer, backup=True)
    except Exception as exc:
        detail = str(exc).replace("\n", " ")[:160]
        return {
            "status": "held",
            "reason": f"qualification_failed:{type(exc).__name__}:{detail}",
        }
    checked = validate_current_background_lane_qualification(
        lane,
        base_authority=base,
        chronovisor_root=chronovisor_root,
        metadata_provider=metadata_provider,
        evaluator=evaluator,
    )
    return {
        "status": "passed" if checked.get("passed") else "held",
        "reason": str(checked.get("reason") or "qualification_readback_failed"),
        **checked,
    }


def validate_current_background_lane_qualification(
    lane: str,
    *,
    base_authority: Mapping[str, Any],
    chronovisor_root: Path = CHRONOVISOR_ROOT,
    metadata_provider: MetadataProvider | None = None,
    evaluator: ReplayEvaluator | None = None,
) -> dict[str, Any]:
    """Rejoin the active pointer, immutable capability, and replay evidence."""

    try:
        config = _resolved_adopted_config(base_authority)
        rows = _qualification_rows(lane)
        metadata = _metadata(
            tuple(base_authority.get("router", {}).get("models", [])),
            metadata_provider,
        )
        identity = _qualification_identity(
            lane,
            base_authority=base_authority,
            config=config,
            metadata=metadata,
            rows=rows,
        )
        identity_sha = canonical_sha256(identity)
        source_path, evaluation_path, capability_path, active_path = _paths(
            chronovisor_root, lane, identity_sha
        )
        if not all(
            path.exists()
            for path in (source_path, evaluation_path, capability_path, active_path)
        ):
            raise DurableStateError("qualification epoch is incomplete")
        pointer = read_sealed_json(active_path)
        capability = read_sealed_json(capability_path)
        artifact = _validated_evaluation(
            source_path=source_path,
            evaluation_path=evaluation_path,
            identity=identity,
            config=config,
            metadata_provider=metadata_provider,
            evaluator=evaluator,
            execute=False,
        )
    except (DurableStateError, OSError, TypeError, ValueError, KeyError):
        return {"passed": False, "reason": "decision_lane_qualification_missing"}
    capability_payload = {
        key: value for key, value in capability.items() if key != "seal_sha256"
    }
    expected_capability = {
        "schema_version": BACKGROUND_LANE_QUALIFICATION_VERSION,
        "artifact_kind": "background-lane-capability",
        "status": "passed",
        "identity": identity,
        "identity_sha256": identity_sha,
        "source_path": str(source_path),
        "source_sha256": identity["source_sha256"],
        "evaluation_path": str(evaluation_path),
        "evaluation_artifact_sha256": canonical_sha256(artifact),
        "evaluation_evidence_sha256": artifact.get("evidence_sha256"),
        "evaluation_result_sha256": artifact.get("evaluation_result_sha256"),
    }
    passed = bool(
        capability_payload == expected_capability
        and capability.get("seal_sha256") == seal_object(expected_capability)["seal_sha256"]
        and pointer
        == seal_object(
            {
                "schema_version": 1,
                "artifact_kind": "background-lane-capability-active-pointer",
                "lane": lane,
                "identity_sha256": identity_sha,
                "capability_sha256": capability["seal_sha256"],
                "capability_path": str(capability_path),
            }
        )
    )
    return {
        "passed": passed,
        "reason": "verified" if passed else "decision_lane_qualification_stale",
        "capability_sha256": str(capability.get("seal_sha256") or ""),
        "identity_sha256": identity_sha,
    }


__all__ = [
    "qualify_background_lane",
    "validate_current_background_lane_qualification",
]
