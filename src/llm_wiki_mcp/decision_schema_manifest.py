"""Versionless manifest of structured schemas used by routine local decisions."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

SIGNATURE_POLICY_VERSION = 1
NON_DECISION_FIELDS = frozenset(
    {
        "comment",
        "comments",
        "confidence",
        "explanation",
        "notes",
        "rationale",
        "reason",
        "reviewer",
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


def signature_policy(schema: Mapping[str, Any]) -> dict[str, Any]:
    """Return the schema-specific root fields that determine an action."""

    properties = schema.get("properties")
    fields = (
        sorted(
            str(name)
            for name in properties
            if str(name) not in NON_DECISION_FIELDS
        )
        if isinstance(properties, Mapping)
        else []
    )
    return {
        "policy_version": SIGNATURE_POLICY_VERSION,
        "schema_sha256": schema_sha256(schema),
        "fields": fields,
    }


def decision_signature_value(
    schema: Mapping[str, Any],
    value: Any,
) -> Any:
    """Select exactly the same action-bearing value for prod, eval, and replay."""

    normalized = default_decision_value(value)
    if not isinstance(normalized, Mapping):
        return normalized
    fields = signature_policy(schema)["fields"]
    if not fields:
        return normalized
    return {field: normalized[field] for field in fields if field in normalized}


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
    "schema_sha256",
    "signature_policy",
]
