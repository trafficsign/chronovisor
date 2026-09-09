"""Stable schema and artifact-version constants for the ingest pipeline."""

from __future__ import annotations

from typing import Any

from chronovisor.core.triage_contract import (
    TRIAGE_C2_MAX_QUOTE_CHARS as TRIAGE_C2_MAX_QUOTE_CHARS,
)
from chronovisor.core.triage_contract import (
    TRIAGE_C2_MAX_QUOTE_LIST as TRIAGE_C2_MAX_QUOTE_LIST,
)
from chronovisor.core.triage_contract import (
    TRIAGE_C2_MAX_SEMANTIC_ROWS as TRIAGE_C2_MAX_SEMANTIC_ROWS,
)
from chronovisor.core.triage_contract import (
    TRIAGE_C2_MAX_SOURCE_RECORD_ID_CHARS as TRIAGE_C2_MAX_SOURCE_RECORD_ID_CHARS,
)
from chronovisor.core.triage_contract import (
    TRIAGE_C2_MAX_SOURCE_RECORD_TEXT_CHARS as TRIAGE_C2_MAX_SOURCE_RECORD_TEXT_CHARS,
)
from chronovisor.core.triage_contract import (
    TRIAGE_C2_MAX_SOURCE_RECORDS as TRIAGE_C2_MAX_SOURCE_RECORDS,
)
from chronovisor.core.triage_contract import (
    TRIAGE_C2_SCHEMA as TRIAGE_C2_SCHEMA,
)
from chronovisor.core.triage_contract import (
    TRIAGE_C2_SCHEMA_VERSION as TRIAGE_C2_SCHEMA_VERSION,
)
from chronovisor.core.triage_contract import (
    TRIAGE_MAX_OPERATIONS as TRIAGE_MAX_OPERATIONS,
)
from chronovisor.core.triage_contract import (
    TRIAGE_PLAN_SCHEMA as TRIAGE_PLAN_SCHEMA,
)
from chronovisor.core.triage_contract import (
    TRIAGE_PLAN_VALIDATION_SCHEMA as TRIAGE_PLAN_VALIDATION_SCHEMA,
)
from chronovisor.decision.decision_lane_prompts import INGEST_PROPOSAL_SCHEMA_VERSION
from chronovisor.decision.decision_schema_manifest import (
    INGEST_FRONTIER_DECISION_SCHEMA as INGEST_FRONTIER_DECISION_SCHEMA,
)

TRIAGE_CATALOG_TOP_N = 100
TRIAGE_MAX_OUTPUT_BYTES = 8_000
TRIAGE_MAX_FEEDBACK_BYTES = 4_096
TRIAGE_NUM_PREDICT = 4_096

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
