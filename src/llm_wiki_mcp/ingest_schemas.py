"""Stable schema and artifact-version constants for the ingest pipeline."""

from __future__ import annotations

from typing import Any

from llm_wiki_mcp.decision_lane_prompts import INGEST_PROPOSAL_SCHEMA_VERSION


TRIAGE_CATALOG_TOP_N = 100
TRIAGE_MAX_OPERATIONS = 8
TRIAGE_MAX_OUTPUT_BYTES = 8_000
TRIAGE_MAX_FEEDBACK_BYTES = 4_000
TRIAGE_NUM_PREDICT = 4_096

TRIAGE_PLAN_VALIDATION_SCHEMA: dict[str, Any] = {
    "type": "array",
    "maxItems": TRIAGE_MAX_OPERATIONS,
    "items": {
        "type": "object",
        "additionalProperties": False,
        "required": ["type", "filename"],
        "properties": {
            "type": {"type": "string", "enum": ["create", "update"]},
            "filename": {"type": "string", "minLength": 1, "maxLength": 200},
            "title": {"type": "string", "minLength": 1, "maxLength": 300},
            "keywords": {
                "type": "array",
                "minItems": 1,
                "maxItems": 32,
                "items": {"type": "string", "minLength": 1, "maxLength": 200},
            },
            "summary": {"type": "string", "minLength": 1, "maxLength": 2_000},
        },
    },
}

TRIAGE_PLAN_SCHEMA: dict[str, Any] = {
    "type": "array",
    "items": {
        "type": "object",
        "additionalProperties": False,
        "required": ["type", "filename", "title", "keywords", "summary"],
        "properties": {
            "type": {"type": "string", "enum": ["create", "update"]},
            "filename": {"type": "string"},
            "title": {"type": "string"},
            "keywords": {"type": "array", "items": {"type": "string"}},
            "summary": {"type": "string"},
        },
    },
}

RECALL_METADATA_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["summary", "recall_questions"],
    "properties": {
        "summary": {"type": "string", "minLength": 1, "maxLength": 500},
        "recall_questions": {
            "type": "array",
            "minItems": 1,
            "maxItems": 5,
            "uniqueItems": True,
            "items": {"type": "string", "minLength": 1, "maxLength": 300},
        },
    },
}

INGEST_FRONTIER_ARTIFACT_SCHEMA_VERSION = INGEST_PROPOSAL_SCHEMA_VERSION
INGEST_FRONTIER_LEGACY_ARTIFACT_SCHEMA_VERSION = 1
INGEST_FRONTIER_REVIEW_ARTIFACT_SCHEMA_VERSION = 2
INGEST_REVIEW_SHARD_POLICY_VERSION = 1
INGEST_REVIEW_SHARD_SCHEMA_VERSION = 1
MAX_INGEST_REVIEW_SHARDS = 32
INGEST_REVIEW_LIMIT_FIELDS = frozenset(
    {
        "num_ctx",
        "min_num_ctx",
        "num_predict",
        "max_input_chars",
        "max_output_chars",
        "max_feedback_chars",
    }
)
INGEST_REVIEW_SHARD_ROW_FIELDS = frozenset(
    {
        "shard_index",
        "original_operation_indices",
        "proposal_sha256",
        "effective_request_sha256",
        "effective_input_chars",
        "effective_input_bytes",
        "required_num_ctx",
        "selected_num_ctx",
    }
)
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
