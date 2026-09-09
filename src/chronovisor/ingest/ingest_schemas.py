"""Stable schema and artifact-version constants for the ingest pipeline."""

from __future__ import annotations

from typing import Any

from chronovisor.decision.decision_lane_prompts import INGEST_PROPOSAL_SCHEMA_VERSION
from chronovisor.decision.decision_schema_manifest import (
    INGEST_FRONTIER_DECISION_SCHEMA as INGEST_FRONTIER_DECISION_SCHEMA,
)

TRIAGE_CATALOG_TOP_N = 100
TRIAGE_MAX_OPERATIONS = 8
TRIAGE_MAX_OUTPUT_BYTES = 8_000
TRIAGE_MAX_FEEDBACK_BYTES = 4_096
TRIAGE_NUM_PREDICT = 4_096

# C2 is an explicit, source-record-bound extension of triage.  These limits
# keep the optional semantic projection inside the existing ingest budget;
# they do not change the v1 operation contract below.
TRIAGE_C2_SCHEMA_VERSION = "chronovisor.ingest-triage-c2.v1"
TRIAGE_C2_MAX_SOURCE_RECORDS = 64
TRIAGE_C2_MAX_SOURCE_RECORD_ID_CHARS = 200
TRIAGE_C2_MAX_SOURCE_RECORD_TEXT_CHARS = 64_000
TRIAGE_C2_MAX_SEMANTIC_ROWS = 8
TRIAGE_C2_MAX_QUOTE_CHARS = 512
TRIAGE_C2_MAX_QUOTE_LIST = 8

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

# The model-facing C2 wrapper is deliberately separate from TRIAGE_PLAN_SCHEMA.
# Keeping the v1 array untouched preserves its canonical bytes and its
# five-column plain-text fallback.  Host code materializes semantic quotes
# only after this strict wrapper has passed validation against fixed source
# records.
TRIAGE_C2_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["operations", "semantic_evidence"],
    "properties": {
        # Reuse the exact v1 operation schema; host validation still applies
        # the existing effective-target and size rules after this wrapper.
        "operations": TRIAGE_PLAN_SCHEMA,
        "semantic_evidence": {
            "type": "array",
            "maxItems": TRIAGE_C2_MAX_SEMANTIC_ROWS,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "record_id",
                    "quote",
                    "kind",
                    "subject_quote",
                    "scope_quotes",
                    "condition_quotes",
                ],
                "properties": {
                    "record_id": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": TRIAGE_C2_MAX_SOURCE_RECORD_ID_CHARS,
                    },
                    "quote": {
                        "type": ["string", "null"],
                        "minLength": 1,
                        "maxLength": TRIAGE_C2_MAX_QUOTE_CHARS,
                    },
                    "kind": {
                        "type": "string",
                        "enum": ["unknown", "proposal", "decision", "result"],
                    },
                    "subject_quote": {
                        "type": ["string", "null"],
                        "minLength": 1,
                        "maxLength": TRIAGE_C2_MAX_QUOTE_CHARS,
                    },
                    "scope_quotes": {
                        "type": ["array", "null"],
                        "maxItems": TRIAGE_C2_MAX_QUOTE_LIST,
                        "uniqueItems": True,
                        "items": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": TRIAGE_C2_MAX_QUOTE_CHARS,
                        },
                    },
                    "condition_quotes": {
                        "type": ["array", "null"],
                        "maxItems": TRIAGE_C2_MAX_QUOTE_LIST,
                        "uniqueItems": True,
                        "items": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": TRIAGE_C2_MAX_QUOTE_CHARS,
                        },
                    },
                },
            },
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
