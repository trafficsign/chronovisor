"""Pure schemas and quote coordinates shared by ingest and evidence storage."""

from __future__ import annotations

import hashlib
from typing import Any

TRIAGE_MAX_OPERATIONS = 8

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


def unique_quote_byte_range(text: str, quote: str) -> tuple[int, int] | None:
    """Return a unique UTF-8 span; overlapping matches are ambiguous too."""

    if not quote:
        return None
    start = text.find(quote)
    if start < 0:
        return None
    if text.find(quote, start + 1) >= 0:
        return None
    byte_start = len(text[:start].encode("utf-8"))
    byte_end = byte_start + len(quote.encode("utf-8"))
    return byte_start, byte_end


def quote_span_payload(
    text: str,
    quote: str | None,
) -> dict[str, Any] | None:
    if quote is None:
        return None
    span = unique_quote_byte_range(text, quote)
    if span is None:
        raise ValueError("C2 semantic quote is missing or ambiguous")
    byte_start, byte_end = span
    return {
        "byte_range": [byte_start, byte_end],
        "byte_coordinate_space": "decoded_source_text_utf8",
        "span_sha256": hashlib.sha256(quote.encode("utf-8")).hexdigest(),
    }
