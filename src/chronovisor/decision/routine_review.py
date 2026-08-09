"""Routine structured decisions resolved by local consensus."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Mapping
from contextlib import ExitStack, suppress
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from chronovisor.decision import semantic_hold
from chronovisor.decision.decision_schema_manifest import (
    FRONTIER_DECISION_SCHEMA as FRONTIER_DECISION_SCHEMA,
)

STRUCTURED_REVIEW_HOLD_CACHE_ROOT: Path | None = None
_STRUCTURED_REVIEW_ROUTER_CONFIG: ContextVar[Any | None] = ContextVar(
    "structured_review_router_config",
    default=None,
)

_SECRET_PATTERNS = [
    re.compile(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{12,}"),
    re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]{12,}"),
    re.compile(r"(?i)(api[_-]?key['\"=\s:]+)[A-Za-z0-9._~+/=-]{8,}"),
    re.compile(r"(?i)(token['\"=\s:]+)[A-Za-z0-9._~+/=-]{8,}"),
    re.compile(r"(?i)(cookie['\"=\s:]+)[^\s\n;]{8,}"),
]


@dataclass(frozen=True)
class _RoutineFailure:
    failure_class: str
    rescue_status: str
    summary: str
    human_required: bool = False
    notify_user: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "failure_class": self.failure_class,
            "rescue_status": self.rescue_status,
            "summary": self.summary,
            "human_required": self.human_required,
            "notify_user": self.notify_user,
        }


def _routine_failure(
    failure_class: str,
    rescue_status: str,
    summary: str,
) -> _RoutineFailure:
    return _RoutineFailure(failure_class, rescue_status, summary)


def _redact_sensitive_text(text: str | None) -> str:
    redacted = text or ""
    for pattern in _SECRET_PATTERNS:
        redacted = pattern.sub(_redact_match, redacted)
    return redacted


def _redact_match(match: re.Match[str]) -> str:
    if match.groups():
        return f"{match.group(1)}[REDACTED]"
    return "[REDACTED]"


def _strict_schema_with_repair(
    schema: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    strict_schema = json.loads(json.dumps(schema))
    properties = strict_schema.get("properties")
    if not isinstance(properties, dict):
        return strict_schema, None

    required = strict_schema.get("required")
    expected_required = list(properties.keys())
    changes: list[dict[str, Any]] = []
    if required != expected_required:
        strict_schema["required"] = expected_required
        changes.append(
            {
                "field": "required",
                "before": required,
                "after": expected_required,
            }
        )
    if strict_schema.get("additionalProperties") is not False:
        changes.append(
            {
                "field": "additionalProperties",
                "before": strict_schema.get("additionalProperties"),
                "after": False,
            }
        )
        strict_schema["additionalProperties"] = False

    if not changes:
        return strict_schema, None
    return strict_schema, {
        "type": "schema_strictness_autofix",
        "applied": True,
        "changes": changes,
        "summary": "normalized frontier output schema for strict structured output",
    }


def _structured_route_result(
    routed: Any,
    *,
    schema: dict[str, Any],
    decision_lane: str | None,
    lane_mode: str,
    router_policy_source: str,
    policy_audit: dict[str, Any],
) -> dict[str, Any]:
    """Convert one trusted local-router result into the compatibility envelope."""

    routed_residency = getattr(routed, "residency", None)
    residency = routed_residency if isinstance(routed_residency, Mapping) else {}
    decision_execution = {
        "execution_fingerprint": residency.get("execution_fingerprint"),
        "decision_artifact_seal_sha256": residency.get(
            "decision_artifact_seal_sha256"
        ),
    }

    if lane_mode == "shadow":
        reason = f"decision_lane_shadow:{decision_lane}"
        failure = _routine_failure(
            "local_decision_shadow_only",
            "local_quarantined",
            reason,
        )
        result = _structured_failure_payload(
            schema,
            summary=reason,
            failure=failure,
            reviewer="local_consensus_shadow",
        )
        result["local_consensus"] = routed.audit_record()
        result["decision_policy"] = policy_audit
        result["decision_execution"] = decision_execution
        return result
    if router_policy_source != "adopted_artifact":
        reason = f"decision_lane_unadopted:{decision_lane}"
        failure = _routine_failure(
            "local_decision_artifact_required",
            "local_quarantined",
            reason,
        )
        result = _structured_failure_payload(
            schema,
            summary=reason,
            failure=failure,
            reviewer="local_policy",
        )
        result["local_consensus"] = routed.audit_record()
        result["decision_policy"] = policy_audit
        result["decision_execution"] = decision_execution
        return result
    if routed.ok and isinstance(routed.decision, dict):
        result = _validated_structured_result(
            routed.decision,
            schema,
            reviewer="local_consensus",
        )
        result["local_consensus"] = routed.audit_record()
        result["decision_policy"] = policy_audit
        result["decision_execution"] = decision_execution
        return result

    reason = (
        routed.quarantine_reason or routed.failure_class or "local consensus failed"
    )
    valid_signatures = {
        vote.signature_sha256
        for vote in routed.votes
        if vote.valid and vote.signature_sha256 is not None
    }
    vote_signature_sha256s = [vote.signature_sha256 for vote in routed.votes]
    three_valid_votes = bool(
        len(routed.votes) == 3 and all(vote.valid for vote in routed.votes)
    )
    conservative_veto_semantic_no_quorum = bool(
        routed.failure_class == "local_consensus_failed"
        and routed.quarantine_reason
        == "mutating_local_majority_vetoed_by_conservative_vote"
        and len(vote_signature_sha256s) == 3
        and all(
            isinstance(signature, str)
            and re.fullmatch(r"[0-9a-f]{64}", signature) is not None
            for signature in vote_signature_sha256s
        )
        and sorted(Counter(vote_signature_sha256s).values()) == [1, 2]
    )
    three_way_semantic_no_quorum = bool(
        three_valid_votes
        and (
            (
                routed.quarantine_reason == "local_models_did_not_reach_two_vote_quorum"
                and len(valid_signatures) == 3
            )
            or conservative_veto_semantic_no_quorum
        )
    )
    failure = _routine_failure(
        (
            semantic_hold.LOCAL_SEMANTIC_NO_QUORUM
            if three_way_semantic_no_quorum
            else "local_consensus_failed"
        ),
        "local_quarantined",
        reason,
    )
    result = _structured_failure_payload(
        schema,
        summary=reason,
        failure=failure,
        reviewer="local_consensus",
    )
    result["local_consensus"] = routed.audit_record()
    result["decision_policy"] = policy_audit
    result["decision_execution"] = decision_execution
    return result


def _structured_semantic_cache_result_error(
    result: object,
    *,
    schema: dict[str, Any],
    policy_audit: dict[str, Any],
    authority: dict[str, Any],
) -> str | None:
    """Validate a cached failure against this exact schema and authority."""

    if not isinstance(result, dict):
        return "structured semantic cache result is missing"
    lane = policy_audit.get("lane")
    if not isinstance(lane, str) or not lane:
        return "structured semantic cache lane is missing"
    if result.get("decision_policy") != policy_audit:
        return "structured semantic cache policy changed"
    if not semantic_hold.is_local_semantic_no_quorum(result):
        return "structured semantic cache no-quorum proof is invalid"
    consensus = result.get("local_consensus")
    assert isinstance(consensus, dict)
    reason = consensus.get("quarantine_reason")
    assert isinstance(reason, str)
    decision_execution = result.get("decision_execution")
    if not isinstance(decision_execution, dict) or set(decision_execution) != {
        "execution_fingerprint",
        "decision_artifact_seal_sha256",
    }:
        return "structured semantic cache execution provenance is invalid"
    if any(
        value is not None
        and (
            not isinstance(value, str)
            or re.fullmatch(r"[0-9a-f]{64}", value) is None
        )
        for value in decision_execution.values()
    ):
        return "structured semantic cache execution provenance is invalid"
    expected = _structured_failure_payload(
        schema,
        summary=reason,
        failure=_routine_failure(
            semantic_hold.LOCAL_SEMANTIC_NO_QUORUM,
            "local_quarantined",
            reason,
        ),
        reviewer="local_consensus",
    )
    expected["local_consensus"] = consensus
    expected["decision_policy"] = policy_audit
    expected["decision_execution"] = decision_execution
    if result != expected:
        return "structured semantic cache failure payload is not canonical"
    try:
        semantic_hold.build_semantic_no_quorum_hold(
            lane,
            {"validation": "structured_review_cache"},
            authority,
            result,
        )
    except (TypeError, ValueError) as exc:
        return f"structured semantic cache authority is invalid:{exc}"
    return None


def _structured_authority_observation(
    authority: dict[str, Any],
) -> str | None:
    """Return an opaque mutable-source generation for the in-flight guard."""

    try:
        return semantic_hold.structured_review_authority_observation_sha256(
            authority,
            router_config=_STRUCTURED_REVIEW_ROUTER_CONFIG.get(),
        )
    except Exception:
        # Enabled adopted lanes fail closed before inference when the mutable
        # authority generation cannot be observed.
        return None


def _current_structured_authority(
    lane: str,
) -> tuple[dict[str, Any] | None, str | None]:
    """Resolve lane authority behind a narrow, test-isolated seam."""

    from chronovisor.decision.decision_authority import current_semantic_authority

    return current_semantic_authority(
        lane,
        router_config=_STRUCTURED_REVIEW_ROUTER_CONFIG.get(),
    )


def _schema_type_matches(value: Any, expected: str) -> bool:
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "null":
        return value is None
    return True


def _structured_validation_error(
    value: Any,
    schema: dict[str, Any],
    *,
    path: str = "$",
) -> str | None:
    expected = schema.get("type")
    allowed_types = expected if isinstance(expected, list) else [expected]
    allowed_types = [item for item in allowed_types if isinstance(item, str)]
    if allowed_types and not any(
        _schema_type_matches(value, item) for item in allowed_types
    ):
        return f"{path}: expected {'|'.join(allowed_types)}"
    if "enum" in schema and value not in schema.get("enum", []):
        return f"{path}: value is outside enum"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(float(value)):
            return f"{path}: number must be finite"
        if "minimum" in schema and value < schema["minimum"]:
            return f"{path}: below minimum"
        if "maximum" in schema and value > schema["maximum"]:
            return f"{path}: above maximum"
    if isinstance(value, list):
        if "minItems" in schema and len(value) < int(schema["minItems"]):
            return f"{path}: too few items"
        if "maxItems" in schema and len(value) > int(schema["maxItems"]):
            return f"{path}: too many items"
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, item in enumerate(value):
                error = _structured_validation_error(
                    item, item_schema, path=f"{path}[{index}]"
                )
                if error:
                    return error
    if isinstance(value, dict):
        properties = schema.get("properties")
        properties = properties if isinstance(properties, dict) else {}
        required = schema.get("required")
        required = required if isinstance(required, list) else []
        missing = [name for name in required if name not in value]
        if missing:
            return f"{path}: missing required fields: {', '.join(str(name) for name in missing)}"
        if schema.get("additionalProperties") is False:
            extras = sorted(set(value) - set(properties))
            if extras:
                return f"{path}: unexpected fields: {', '.join(extras)}"
        for name, child_schema in properties.items():
            if name not in value or not isinstance(child_schema, dict):
                continue
            error = _structured_validation_error(
                value[name], child_schema, path=f"{path}.{name}"
            )
            if error:
                return error
    return None


def _failure_default(name: str, schema: dict[str, Any], summary: str) -> Any:
    enum = schema.get("enum")
    if isinstance(enum, list) and enum:
        if name == "decision":
            for safe_decision in (
                "needs_retry",
                "retry",
                "rejected",
                "quarantined",
            ):
                if safe_decision in enum:
                    return safe_decision
        return enum[0]
    expected = schema.get("type")
    types = expected if isinstance(expected, list) else [expected]
    if "null" in types:
        return None
    if name == "decision" and "string" in types:
        return "needs_retry"
    if name in {"summary", "reason", "notes"} and "string" in types:
        return summary
    if name == "confidence" and "number" in types:
        return 0.0
    if "string" in types:
        return ""
    if "number" in types or "integer" in types:
        return 0
    if "boolean" in types:
        return False
    if "array" in types:
        return []
    if "object" in types:
        return {}
    return None


def _structured_failure_payload(
    schema: dict[str, Any],
    *,
    summary: str,
    failure: _RoutineFailure,
    reviewer: str,
    diagnostics: str = "",
) -> dict[str, Any]:
    strict_schema, _repair = _strict_schema_with_repair(schema)
    properties = strict_schema.get("properties")
    properties = properties if isinstance(properties, dict) else {}
    payload = {
        name: _failure_default(name, field_schema, summary)
        for name, field_schema in properties.items()
        if isinstance(field_schema, dict)
    }
    failure_payload = failure.to_dict()
    if diagnostics:
        redacted = _redact_sensitive_text(diagnostics)
        failure_payload["diagnostics_tail"] = redacted[-4000:]
        failure_payload["diagnostics_sha256"] = hashlib.sha256(
            redacted.encode("utf-8")
        ).hexdigest()
    payload["frontier_failure"] = failure_payload
    payload["human_required"] = failure.human_required
    payload["reviewer"] = reviewer
    return payload


def _validated_structured_result(
    parsed: dict[str, Any] | None,
    schema: dict[str, Any],
    *,
    reviewer: str,
) -> dict[str, Any]:
    if parsed is None:
        failure = _routine_failure(
            "schema_invalid",
            "pending_frontier_review",
            "frontier output did not contain JSON",
        )
        return _structured_failure_payload(
            schema, summary=failure.summary, failure=failure, reviewer=reviewer
        )
    # The local router already validates against the caller's production
    # schema. Provider strictness rewrites must not turn legitimate optional
    # fields into required result fields after a successful local quorum.
    error = _structured_validation_error(parsed, schema)
    if error:
        failure = _routine_failure(
            "schema_invalid",
            "pending_frontier_review",
            f"frontier output failed schema validation: {error}",
        )
        return _structured_failure_payload(
            schema, summary=failure.summary, failure=failure, reviewer=reviewer
        )
    result = dict(parsed)
    result["reviewer"] = reviewer
    return result


def run_structured_review(
    prompt: str,
    schema: dict[str, Any],
    *,
    repo_root: Path,
    audit_root: Path | None = None,
    timeout: int | None = None,
    execute_patch: bool = False,
    command_env: str = "CHRONOVISOR_STRUCTURED_REVIEW_CMD",
    model_role: str = "semantic_judge",
    decision_lane: str | None = None,
    model_override: str | None = None,
    reasoning_effort_override: str | None = None,
    record_replay: bool = True,
    system: str | None = None,
) -> dict[str, Any]:
    """Resolve a routine structured decision using local models only."""
    del (
        repo_root,
        timeout,
        execute_patch,
        command_env,
        model_override,
        reasoning_effort_override,
    )

    from chronovisor.decision.decision_policy import resolve_decision_policy
    from chronovisor.decision.decision_router import (
        DecisionRouter,
        decision_request_fingerprint_sha256,
        load_decision_router_config,
    )
    from chronovisor.decision.decision_schema_manifest import (
        production_schema_manifest,
        schema_sha256,
    )

    lane_policy, lane_mode, lane_error = resolve_decision_policy(decision_lane)
    policy_audit = {
        "lane": decision_lane,
        "kind": lane_policy.kind if lane_policy is not None else None,
        "schema_name": lane_policy.schema_name if lane_policy is not None else None,
        "mode": lane_mode,
        "error": lane_error,
    }
    if lane_error is not None or lane_mode == "off":
        reason = lane_error or f"decision_lane_off:{decision_lane}"
        failure = _routine_failure(
            "local_decision_policy_blocked",
            "local_quarantined",
            reason,
        )
        result = _structured_failure_payload(
            schema,
            summary=reason,
            failure=failure,
            reviewer="local_policy",
        )
        result["decision_policy"] = policy_audit
        return result

    if lane_policy is None or lane_policy.kind not in {"consensus", "local_batch"}:
        reason = f"decision_lane_not_structured:{decision_lane}"
        failure = _routine_failure(
            "local_decision_policy_kind_invalid",
            "local_quarantined",
            reason,
        )
        result = _structured_failure_payload(
            schema,
            summary=reason,
            failure=failure,
            reviewer="local_policy",
        )
        result["decision_policy"] = policy_audit
        return result

    expected_digest = production_schema_manifest().get(str(lane_policy.schema_name))
    actual_digest = schema_sha256(schema)
    policy_audit["expected_schema_sha256"] = expected_digest
    policy_audit["actual_schema_sha256"] = actual_digest
    if expected_digest is None or actual_digest != expected_digest:
        reason = f"decision_lane_schema_mismatch:{decision_lane}"
        failure = _routine_failure(
            "local_decision_schema_mismatch",
            "local_quarantined",
            reason,
        )
        result = _structured_failure_payload(
            schema,
            summary=reason,
            failure=failure,
            reviewer="local_policy",
        )
        result["decision_policy"] = policy_audit
        return result

    review_router_config = load_decision_router_config()

    authority: dict[str, Any] | None = None
    authority_observation_sha256: str | None = None
    cache_epoch: dict[str, Any] | None = None
    if isinstance(decision_lane, str) and decision_lane and lane_mode == "enabled":
        config_token = _STRUCTURED_REVIEW_ROUTER_CONFIG.set(review_router_config)
        try:
            candidate_authority, authority_error = _current_structured_authority(
                decision_lane
            )
            if authority_error is None and isinstance(candidate_authority, dict):
                expected_policy_authority = {
                    "kind": lane_policy.kind,
                    "schema_name": lane_policy.schema_name,
                    "mode": lane_mode,
                    "error": lane_error,
                }
                if candidate_authority.get("policy") == expected_policy_authority:
                    candidate_observation_sha256 = _structured_authority_observation(
                        candidate_authority
                    )
                    try:
                        if candidate_observation_sha256 is None:
                            raise ValueError("authority observation unavailable")
                        effective_request_sha256 = decision_request_fingerprint_sha256(
                            prompt=prompt,
                            schema=schema,
                            system=system,
                            decision_lane=decision_lane,
                        )
                        cache_epoch = semantic_hold.build_structured_review_hold_epoch(
                            lane=decision_lane,
                            authority=candidate_authority,
                            schema_sha256=actual_digest,
                            prompt=prompt,
                            system=system,
                            effective_request_sha256=effective_request_sha256,
                        )
                    except (TypeError, ValueError):
                        cache_epoch = None
                    else:
                        authority = candidate_authority
                        authority_observation_sha256 = candidate_observation_sha256
        finally:
            _STRUCTURED_REVIEW_ROUTER_CONFIG.reset(config_token)

    router = DecisionRouter(
        config=review_router_config,
        audit_root=audit_root,
        audit_role=decision_lane or model_role,
        record_replay=record_replay,
        require_adopted=lane_mode == "enabled",
        decision_lane=decision_lane,
    )
    policy_audit["router_policy"] = router.policy.audit_record()
    cache_eligible = bool(
        authority is not None
        and cache_epoch is not None
        and router.policy.source == "adopted_artifact"
        and router.policy.audit_record() == authority.get("router")
    )

    def authority_guard_error(stage: str) -> str | None:
        if (
            authority is None
            or authority_observation_sha256 is None
            or not isinstance(decision_lane, str)
        ):
            return f"decision authority observation unavailable {stage} local review"
        current_authority, current_error = _current_structured_authority(decision_lane)
        current_observation = (
            _structured_authority_observation(current_authority)
            if isinstance(current_authority, dict)
            else None
        )
        if current_error is not None or current_observation is None:
            return f"decision authority observation unavailable {stage} local review"
        if (
            current_authority != authority
            or current_observation != authority_observation_sha256
        ):
            return f"decision authority changed {stage} local review"
        return None

    def authority_failure(reason: str) -> dict[str, Any]:
        failure_class = (
            "decision_authority_changed"
            if "changed" in reason
            else "decision_authority_unavailable"
        )
        result = _structured_failure_payload(
            schema,
            summary=reason,
            failure=_routine_failure(failure_class, "local_retry", reason),
            reviewer="local_policy",
        )
        result["decision_policy"] = policy_audit
        return result

    def route_once() -> dict[str, Any]:
        routed = (
            router.decide(prompt, schema)
            if system is None
            else router.decide(prompt, schema, system=system)
        )
        return _structured_route_result(
            routed,
            schema=schema,
            decision_lane=decision_lane,
            lane_mode=lane_mode,
            router_policy_source=router.policy.source,
            policy_audit=policy_audit,
        )

    if lane_mode == "enabled" and router.policy.source == "adopted_artifact":
        if not cache_eligible:
            return authority_failure(
                "decision authority observation unavailable before local review"
            )
        assert authority is not None
        assert authority_observation_sha256 is not None
        assert cache_epoch is not None
        assert isinstance(decision_lane, str)
        cache = semantic_hold.StructuredReviewSemanticHoldCache(
            root=STRUCTURED_REVIEW_HOLD_CACHE_ROOT
        )
        with ExitStack() as stack:
            try:
                lease = stack.enter_context(
                    cache.locked(
                        lane=decision_lane,
                        epoch=cache_epoch,
                        authority=authority,
                    )
                )
            except (OSError, TypeError, ValueError):
                lease = None
            pre_error = authority_guard_error("before")
            if pre_error is not None:
                return authority_failure(pre_error)
            if lease is not None:
                cached = lease.load()
                if cached is not None and (
                    _structured_semantic_cache_result_error(
                        cached,
                        schema=schema,
                        policy_audit=policy_audit,
                        authority=authority,
                    )
                    is None
                ):
                    post_error = authority_guard_error("during")
                    return (
                        cached if post_error is None else authority_failure(post_error)
                    )
            result = route_once()
            post_error = authority_guard_error("during")
            if post_error is not None:
                return authority_failure(post_error)
            if lease is not None and (
                _structured_semantic_cache_result_error(
                    result,
                    schema=schema,
                    policy_audit=policy_audit,
                    authority=authority,
                )
                is None
            ):
                with suppress(OSError, TypeError, ValueError):
                    lease.store(result)
                post_store_error = authority_guard_error("during")
                if post_store_error is not None:
                    return authority_failure(post_store_error)
            return result

    return route_once()
