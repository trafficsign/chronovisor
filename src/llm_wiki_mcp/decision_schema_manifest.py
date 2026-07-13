"""Versionless manifest of structured schemas used by routine local decisions."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

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


def schema_sha256(schema: Mapping[str, Any]) -> str:
    """Hash a schema with the same canonical JSON used by replay artifacts."""

    encoded = json.dumps(
        schema,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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

    Imports stay inside the function so normal DecisionRouter calls do not
    load unrelated lanes.  Adding a new production caller requires adding its
    schema here before a replacement model fleet can pass the adoption gate.
    """

    from llm_wiki_mcp.autonomy import (
        DUPLICATE_FRONTIER_SCHEMA,
        RETENTION_FRONTIER_SCHEMA,
    )
    from llm_wiki_mcp.content_correction import (
        FRONTIER_CLASSIFICATION_SCHEMA,
        FRONTIER_REVIEW_SCHEMA,
    )
    from llm_wiki_mcp.frontier_review import FRONTIER_DECISION_SCHEMA
    from llm_wiki_mcp.ingest import INGEST_FRONTIER_DECISION_SCHEMA
    from llm_wiki_mcp.lint import SAFE_FIX_REVIEW_SCHEMA
    from llm_wiki_mcp.lint_repair import TAG_REPAIR_SCHEMA
    from llm_wiki_mcp.local_repair import LOCAL_REPAIR_SCHEMA
    from llm_wiki_mcp.orphan_link import ORPHAN_FRONTIER_SCHEMA
    from llm_wiki_mcp.raw_replay import RAW_REPLAY_RECONCILIATION_SCHEMA
    from llm_wiki_mcp.read_back_repair import READ_BACK_FRONTIER_SCHEMA
    from llm_wiki_mcp.search_eval import FRONTIER_LABEL_SCHEMA

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
