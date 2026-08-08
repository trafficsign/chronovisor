"""Versionless manifest of structured schemas used by routine local decisions."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from chronovisor.core.canonical_json import canonical_json_sha256_strict

SIGNATURE_POLICY_VERSION = 5
NON_DECISION_FIELDS = frozenset(
    {
        "comment",
        "comments",
        "commit",
        "committed",
        "confidence",
        "explanation",
        "notes",
        "rationale",
        "reason",
        "reviewer",
        "risk",
        "pushed",
        "semantic_checks",
        "summary",
        "tests_run",
    }
)

FRONTIER_DECISION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "decision",
        "summary",
        "tests_run",
        "commit",
        "committed",
        "pushed",
        "risk",
        "notes",
    ],
    "properties": {
        "decision": {
            "type": "string",
            "enum": ["approved", "rejected", "quarantined", "needs_retry"],
        },
        "summary": {"type": "string"},
        "tests_run": {"type": "array", "items": {"type": "string"}},
        "commit": {"type": ["string", "null"]},
        "committed": {"type": "boolean"},
        "pushed": {"type": "boolean"},
        "risk": {"type": ["string", "null"]},
        "notes": {"type": ["string", "null"]},
    },
}

LOCAL_REPAIR_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["status", "action", "confidence", "reason"],
    "properties": {
        "status": {"type": "string", "enum": ["resolved", "escalate", "rejected"]},
        "action": {
            "type": "string",
            "enum": [
                "escalate_to_frontier",
                "propose_prompt_fix",
                "propose_test_case",
                "quarantine_raw",
                "resolve_update_target",
                "retry_raw",
            ],
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "requested_page_id": {"type": ["string", "null"]},
        "target_page_id": {"type": ["string", "null"]},
        "reason": {"type": "string"},
        "notes": {"type": ["string", "null"]},
    },
}

INGEST_FRONTIER_DECISION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "decision",
        "summary",
        "failed_operations_disposition",
        "tests_run",
        "risk",
        "notes",
    ],
    "properties": {
        "decision": {
            "type": "string",
            "enum": ["apply_available", "confirmed_noop", "retry", "quarantined"],
        },
        "summary": {"type": "string"},
        "failed_operations_disposition": {
            "type": "string",
            "enum": ["none", "confirmed_unnecessary", "retry_required"],
        },
        "tests_run": {"type": "array", "items": {"type": "string"}},
        "risk": {"type": ["string", "null"]},
        "notes": {"type": ["string", "null"]},
        "repair_option_id": {"type": "string", "pattern": "^rp_[0-9a-f]{32}$"},
        "invalid_tags": {
            "type": "array",
            "items": {"type": "string", "pattern": "^[dts]/[a-z0-9][a-z0-9-]*$"},
        },
        "replacement_operations": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["filename", "content"],
                "properties": {
                    "filename": {"type": "string"},
                    "content": {"type": "string"},
                },
            },
        },
    },
}

READ_BACK_FRONTIER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["decision", "confidence", "summary"],
    "properties": {
        "decision": {
            "type": "string",
            "enum": ["approved", "rejected", "needs_retry"],
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "summary": {"type": "string"},
    },
}

DUPLICATE_FRONTIER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["decision", "confidence", "summary"],
    "properties": {
        "decision": {
            "type": "string",
            "enum": ["supersede_left", "supersede_right", "keep_both", "needs_retry"],
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "summary": {"type": "string"},
    },
}

RETENTION_FRONTIER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["decision", "confidence", "summary"],
    "properties": {
        "decision": {
            "type": "string",
            "enum": ["archive", "keep_active", "needs_retry"],
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "summary": {"type": "string"},
    },
}

SAFE_FIX_REVIEW_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "decision",
        "summary",
        "tests_run",
        "commit",
        "committed",
        "pushed",
        "risk",
        "notes",
    ],
    "properties": {
        "decision": {
            "type": "string",
            "enum": ["approved", "rejected", "quarantined", "needs_retry"],
        },
        "summary": {"type": "string"},
        "tests_run": {"type": "array", "items": {"type": "string"}},
        "commit": {"type": ["string", "null"]},
        "committed": {"type": "boolean"},
        "pushed": {"type": "boolean"},
        "risk": {"type": ["string", "null"]},
        "notes": {"type": ["string", "null"]},
    },
}

TAG_REPAIR_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["decision", "tags", "reason"],
    "properties": {
        "decision": {
            "type": "string",
            "enum": ["approved", "rejected", "uncertain", "needs_retry"],
        },
        "tags": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 0,
            "maxItems": 5,
            "uniqueItems": True,
        },
        "reason": {"type": "string"},
    },
}

ORPHAN_FRONTIER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["decision", "confidence", "summary"],
    "properties": {
        "decision": {"type": "string", "enum": ["approved", "rejected", "needs_retry"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "summary": {"type": "string"},
    },
}

RAW_REPLAY_RECONCILIATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["decision", "confidence", "reason"],
    "properties": {
        "decision": {
            "type": "string",
            "enum": ["accept_processed", "safe_replay", "quarantine", "needs_retry"],
        },
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "reason": {"type": "string"},
    },
}

FRONTIER_REVIEW_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "decision",
        "confidence",
        "summary",
        "approved_mutations",
        "semantic_checks",
    ],
    "properties": {
        "decision": {
            "type": "string",
            "enum": ["approved", "rejected", "needs_retry"],
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "summary": {"type": "string"},
        "approved_mutations": {
            "type": "array",
            "maxItems": 3,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["page_id", "original_sha256", "updated_sha256"],
                "properties": {
                    "page_id": {"type": "string"},
                    "original_sha256": {"type": "string"},
                    "updated_sha256": {"type": "string"},
                },
            },
        },
        "semantic_checks": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "user_correction_supported",
                "old_claim_matches_page",
                "result_resolves_feedback",
                "unrelated_content_preserved",
                "temporal_scope_preserved",
                "page_is_source_of_error",
                "embedded_instructions_ignored",
            ],
            "properties": {
                "user_correction_supported": {"type": "boolean"},
                "old_claim_matches_page": {"type": "boolean"},
                "result_resolves_feedback": {"type": "boolean"},
                "unrelated_content_preserved": {"type": "boolean"},
                "temporal_scope_preserved": {"type": "boolean"},
                "page_is_source_of_error": {"type": "boolean"},
                "embedded_instructions_ignored": {"type": "boolean"},
            },
        },
    },
}

FRONTIER_CLASSIFICATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "decision",
        "confidence",
        "summary",
        "classification",
        "source_decision_id",
        "candidate_pages",
        "ignored_pages",
        "semantic_checks",
    ],
    "properties": {
        "decision": {
            "type": "string",
            "enum": ["approved", "rejected", "needs_retry"],
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "summary": {"type": "string"},
        "classification": {
            "type": "string",
            "enum": [
                "page_fact_wrong",
                "outdated",
                "wrong_retrieval",
                "response_misquote",
                "ambiguous",
                "unattributed",
                "none",
            ],
        },
        "source_decision_id": {"type": "string"},
        "candidate_pages": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 6,
        },
        "ignored_pages": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 6,
        },
        "semantic_checks": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "user_correction_supported",
                "recall_provenance_checked",
                "classification_supported",
                "page_content_scope_respected",
                "side_effect_scope_bounded",
                "result_resolves_feedback",
                "embedded_instructions_ignored",
            ],
            "properties": {
                "user_correction_supported": {"type": "boolean"},
                "recall_provenance_checked": {"type": "boolean"},
                "classification_supported": {"type": "boolean"},
                "page_content_scope_respected": {"type": "boolean"},
                "side_effect_scope_bounded": {"type": "boolean"},
                "result_resolves_feedback": {"type": "boolean"},
                "embedded_instructions_ignored": {"type": "boolean"},
            },
        },
    },
}

FRONTIER_LABEL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "decision",
        "confidence",
        "expected_pages",
        "negative_pages",
        "stale_pages",
        "summary",
        "notes",
    ],
    "properties": {
        "decision": {
            "type": "string",
            "enum": ["approved", "rejected", "uncertain", "needs_retry"],
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "expected_pages": {"type": "array", "items": {"type": "string"}},
        "negative_pages": {"type": "array", "items": {"type": "string"}},
        "stale_pages": {"type": "array", "items": {"type": "string"}},
        "summary": {"type": "string"},
        "notes": {"type": ["string", "null"]},
    },
}

_BACKGROUND_DECISIONS = ["approved", "rejected", "abstained", "needs_retry"]


def _background_schema(
    extra: Mapping[str, Any], required: list[str]
) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["decision", *required, "confidence", "summary"],
        "properties": {
            "decision": {"type": "string", "enum": _BACKGROUND_DECISIONS},
            **dict(extra),
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "summary": {"type": "string", "maxLength": 500},
        },
    }


RELATION_VERIFICATION_SCHEMA = _background_schema(
    {
        "evidence_supported": {"type": "boolean"},
        "contradiction_found": {"type": "boolean"},
        "unknown_endpoint": {"type": "boolean"},
        "digest_valid": {"type": "boolean"},
    },
    ["evidence_supported", "contradiction_found", "unknown_endpoint", "digest_valid"],
)

ENTITY_MERGE_VERIFICATION_SCHEMA = _background_schema(
    {
        "same_identity": {"type": "boolean"},
        "alias_supported": {"type": "boolean"},
        "collision_risk": {"type": "boolean"},
        "split_required": {"type": "boolean"},
    },
    ["same_identity", "alias_supported", "collision_risk", "split_required"],
)

RECALL_USEFULNESS_SCHEMA = _background_schema(
    {
        "topically_relevant": {"type": "boolean"},
        "marginally_useful": {"type": "boolean"},
        "read_worthy": {"type": "boolean"},
        "stale_or_harmful": {"type": "boolean"},
    },
    ["topically_relevant", "marginally_useful", "read_worthy", "stale_or_harmful"],
)

RECALL_RUBRIC_CALIBRATION_SCHEMA = _background_schema(
    {
        "rubric_id": {"type": "string", "minLength": 1, "maxLength": 128},
        "holdout_non_regression": {"type": "boolean"},
        "calibration_improved": {"type": "boolean"},
        "coverage_preserved": {"type": "boolean"},
        "rollback_safe": {"type": "boolean"},
    },
    [
        "rubric_id",
        "holdout_non_regression",
        "calibration_improved",
        "coverage_preserved",
        "rollback_safe",
    ],
)

RECALL_ANSWER_ADJUDICATION_SCHEMA = _background_schema(
    {
        "subject_kind": {
            "type": "string",
            "enum": [
                "gold_entry",
                "scorer_calibration_case",
                "search_label_candidate",
            ],
        },
        "subject_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        "evidence_complete": {"type": "boolean"},
        "reference_independent": {"type": "boolean"},
        "preregistered_before_evaluation": {"type": "boolean"},
        "split_safe": {"type": "boolean"},
    },
    [
        "subject_kind",
        "subject_sha256",
        "evidence_complete",
        "reference_independent",
        "preregistered_before_evaluation",
        "split_safe",
    ],
)


def schema_sha256(schema: Mapping[str, Any]) -> str:
    """Hash a schema with the same canonical JSON used by replay artifacts."""

    return canonical_json_sha256_strict(schema)


def default_decision_value(value: Any) -> Any:
    """Drop prose/diagnostic fields while preserving all action structure."""

    if isinstance(value, Mapping):
        return {
            str(key): default_decision_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if str(key) not in NON_DECISION_FIELDS
        }
    if isinstance(value, list):
        return [default_decision_value(item) for item in value]
    return value


def _is_local_repair_schema(schema: Mapping[str, Any]) -> bool:
    properties = schema.get("properties")
    required = schema.get("required")
    if not isinstance(properties, Mapping) or not isinstance(required, list):
        return False
    action = properties.get("action")
    return bool(
        set(required) == {"status", "action", "confidence", "reason"}
        and isinstance(action, Mapping)
        and {
            "resolve_update_target",
            "retry_raw",
            "quarantine_raw",
            "escalate_to_frontier",
            "propose_prompt_fix",
            "propose_test_case",
        }
        == set(action.get("enum", ()))
    )


def _is_ingest_reconciliation_schema(schema: Mapping[str, Any]) -> bool:
    """Identify the ingest schema whose two repair arrays default to empty."""

    properties = schema.get("properties")
    required = schema.get("required")
    if not isinstance(properties, Mapping) or not isinstance(required, list):
        return False
    decision = properties.get("decision")
    return bool(
        set(required)
        == {
            "decision",
            "summary",
            "failed_operations_disposition",
            "tests_run",
            "risk",
            "notes",
        }
        and isinstance(decision, Mapping)
        and set(decision.get("enum", ()))
        == {"apply_available", "confirmed_noop", "retry", "quarantined"}
        and {"invalid_tags", "replacement_operations"}.issubset(properties)
    )


def canonical_ingest_repair_arrays(value: Mapping[str, Any]) -> dict[str, Any]:
    """Canonicalize ingest repair instructions to their application semantics.

    Invalid tags are a set and replacement operations are an unordered set of
    filename/content records.  Model ordering and exact duplicates therefore
    must not manufacture a different quorum signature or a different repair.
    Conflicting replacements for the same filename remain distinct so the
    ingest validator can reject the ambiguity instead of choosing one body.
    """

    normalized = dict(value)
    for field in ("invalid_tags", "replacement_operations"):
        items = normalized.get(field)
        if not isinstance(items, list):
            continue
        canonical: dict[str, Any] = {}
        for item in items:
            stable_item = dict(item) if isinstance(item, Mapping) else item
            if field == "replacement_operations" and isinstance(stable_item, dict):
                content = stable_item.get("content")
                if isinstance(content, str):
                    # The repair application path strips surrounding
                    # whitespace before creating or appending the replacement.
                    # Match that exact side-effect identity so a trailing
                    # newline cannot manufacture a tie.
                    stable_item["content"] = content.strip()
            key = json.dumps(
                stable_item,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            canonical.setdefault(key, stable_item)
        normalized[field] = [canonical[key] for key in sorted(canonical)]
    return normalized


def signature_policy(schema: Mapping[str, Any]) -> dict[str, Any]:
    """Return the schema-specific root fields that determine an action."""

    properties = schema.get("properties")
    fields = (
        sorted(str(name) for name in properties if str(name) not in NON_DECISION_FIELDS)
        if isinstance(properties, Mapping)
        else []
    )
    policy = {
        "policy_version": SIGNATURE_POLICY_VERSION,
        "schema_sha256": schema_sha256(schema),
        "fields": ["action", "status"] if _is_local_repair_schema(schema) else fields,
    }
    if _is_local_repair_schema(schema):
        policy["conditional_fields"] = {"resolve_update_target": ["target_page_id"]}
    return policy


def decision_signature_value(
    schema: Mapping[str, Any],
    value: Any,
) -> Any:
    """Select exactly the same action-bearing value for prod, eval, and replay."""

    normalized = default_decision_value(value)
    if not isinstance(normalized, Mapping):
        return normalized
    if _is_ingest_reconciliation_schema(schema):
        # These optional arrays are imperative repair instructions.  Missing
        # and explicitly empty both mean "perform no repair"; treating them
        # as distinct needlessly invokes the tie model without changing the
        # action or its side effects.
        normalized = canonical_ingest_repair_arrays(normalized)
        normalized.setdefault("invalid_tags", [])
        normalized.setdefault("replacement_operations", [])
    if _is_local_repair_schema(schema):
        selected = {
            field: normalized[field]
            for field in ("action", "status")
            if field in normalized
        }
        if (
            normalized.get("action") == "resolve_update_target"
            and "target_page_id" in normalized
        ):
            selected["target_page_id"] = normalized["target_page_id"]
        return selected
    fields = signature_policy(schema)["fields"]
    if not fields:
        return normalized
    selected = {field: normalized[field] for field in fields if field in normalized}
    properties = schema.get("properties")
    if isinstance(properties, Mapping):
        for field, value in list(selected.items()):
            field_schema = properties.get(field)
            if (
                isinstance(field_schema, Mapping)
                and field_schema.get("uniqueItems") is True
                and isinstance(value, list)
            ):
                # JSON arrays normally carry order, but ``uniqueItems`` fields
                # in our decision schemas are semantic sets.  Sorting only
                # those declared fields prevents harmless model ordering from
                # manufacturing a disagreement while leaving ordered action
                # sequences untouched.
                selected[field] = sorted(
                    value,
                    key=lambda item: json.dumps(
                        item,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    ),
                )
    return selected


def production_decision_schemas() -> dict[str, Mapping[str, Any]]:
    """Return every named schema that can reach the local decision router.

    Adding a new production caller requires adding its schema here before a
    replacement model fleet can pass the adoption gate.
    """

    return {
        "content_correction_classification": FRONTIER_CLASSIFICATION_SCHEMA,
        "content_correction_review": FRONTIER_REVIEW_SCHEMA,
        "duplicate_resolution": DUPLICATE_FRONTIER_SCHEMA,
        "generic_decision": FRONTIER_DECISION_SCHEMA,
        "ingest_reconciliation": INGEST_FRONTIER_DECISION_SCHEMA,
        "lint_safe_semantic_mutation": SAFE_FIX_REVIEW_SCHEMA,
        "lint_tag_repair": TAG_REPAIR_SCHEMA,
        "local_repair": LOCAL_REPAIR_SCHEMA,
        "orphan_link": ORPHAN_FRONTIER_SCHEMA,
        "raw_replay_reconciliation": RAW_REPLAY_RECONCILIATION_SCHEMA,
        "read_back_repair": READ_BACK_FRONTIER_SCHEMA,
        "retention": RETENTION_FRONTIER_SCHEMA,
        "search_label": FRONTIER_LABEL_SCHEMA,
    }


def background_decision_schemas() -> dict[str, Mapping[str, Any]]:
    """Schemas for shadow/background lanes outside the adopted 19-lane fleet."""

    return {
        "relation_verification": RELATION_VERIFICATION_SCHEMA,
        "entity_merge_verification": ENTITY_MERGE_VERIFICATION_SCHEMA,
        "recall_usefulness_judgment": RECALL_USEFULNESS_SCHEMA,
        "recall_rubric_calibration": RECALL_RUBRIC_CALIBRATION_SCHEMA,
        "recall_answer_adjudication": RECALL_ANSWER_ADJUDICATION_SCHEMA,
    }


def production_schema_manifest() -> dict[str, str]:
    """Return stable name-to-hash metadata without exposing schema contents."""

    return {
        name: schema_sha256(schema)
        for name, schema in production_decision_schemas().items()
    }


def production_signature_manifest() -> dict[str, dict[str, Any]]:
    """Bind every production schema hash to its action-signature fields."""

    return {
        name: signature_policy(schema)
        for name, schema in production_decision_schemas().items()
    }


__all__ = [
    "DUPLICATE_FRONTIER_SCHEMA",
    "ENTITY_MERGE_VERIFICATION_SCHEMA",
    "FRONTIER_CLASSIFICATION_SCHEMA",
    "FRONTIER_DECISION_SCHEMA",
    "FRONTIER_LABEL_SCHEMA",
    "FRONTIER_REVIEW_SCHEMA",
    "INGEST_FRONTIER_DECISION_SCHEMA",
    "LOCAL_REPAIR_SCHEMA",
    "ORPHAN_FRONTIER_SCHEMA",
    "RAW_REPLAY_RECONCILIATION_SCHEMA",
    "READ_BACK_FRONTIER_SCHEMA",
    "RECALL_ANSWER_ADJUDICATION_SCHEMA",
    "RECALL_RUBRIC_CALIBRATION_SCHEMA",
    "RECALL_USEFULNESS_SCHEMA",
    "RELATION_VERIFICATION_SCHEMA",
    "RETENTION_FRONTIER_SCHEMA",
    "SAFE_FIX_REVIEW_SCHEMA",
    "TAG_REPAIR_SCHEMA",
    "background_decision_schemas",
    "production_decision_schemas",
    "production_schema_manifest",
    "production_signature_manifest",
    "decision_signature_value",
    "default_decision_value",
    "NON_DECISION_FIELDS",
    "SIGNATURE_POLICY_VERSION",
    "canonical_ingest_repair_arrays",
    "schema_sha256",
    "signature_policy",
]
