"""Source-bound storage and consumption for the explicit C2 triage result.

The ingest C2 triage call is intentionally kept in :mod:`chronovisor.ingest`.
This module only handles the hand-off after that call has succeeded: it seals
the fixed model result together with the host verified source map, stores the
canonical envelope in the existing research CAS, and can turn the rows into
the existing C2 evidence projection/ledger contract.

Semantic ``kind``/subject/scope/condition values remain model judgements.  A
semantic row selects an unchanged native C2 atom only after the caller supplies
the same source records and an independently rebuilt Raw projection.  No Raw
event body is copied into this envelope, and no relation is generated here.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from chronovisor.core.canonical_json import (
    canonical_json_line_bytes_strict,
    canonical_json_sha256_strict,
)
from chronovisor.core.raw_segment import RawSegmentCorrupt
from chronovisor.core.raw_store import RawStore, committed_event_spans
from chronovisor.core.triage_contract import (
    TRIAGE_C2_MAX_QUOTE_CHARS,
    TRIAGE_C2_MAX_QUOTE_LIST,
    TRIAGE_C2_MAX_SEMANTIC_ROWS,
    TRIAGE_C2_MAX_SOURCE_RECORD_ID_CHARS,
    TRIAGE_C2_MAX_SOURCE_RECORD_TEXT_CHARS,
    TRIAGE_C2_MAX_SOURCE_RECORDS,
    TRIAGE_C2_SCHEMA,
    TRIAGE_C2_SCHEMA_VERSION,
    TRIAGE_MAX_OPERATIONS,
    quote_span_payload,
)
from chronovisor.research.evidence_reconstruction import (
    EPISODE_PROJECTION_C2_SCHEMA,
    EpisodeProjection,
    EvidenceAtom,
    RetrievalProgram,
    _event_semantics,
    _identity,
    _projection_atom_c2,
    load_episode_projection_c2,
)
from chronovisor.search.research_store import ResearchStore
from chronovisor.search.research_types import EvidenceArtifact

C2_SEMANTIC_ENVELOPE_SCHEMA = "chronovisor.c2-semantic-evidence-envelope.v1"
C2_SEMANTIC_ENVELOPE_VERSION = 1

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_KINDS = frozenset({"unknown", "proposal", "decision", "result"})
_TRIAGE_KEYS = frozenset({"operations", "semantic_evidence", "audit"})
_AUDIT_KEYS = frozenset(
    {
        "schema_version",
        "schema_sha256",
        "prompt_sha256",
        "request_sha256",
        "source_records_sha256",
        "source_record_count",
        "semantic_evidence_authority",
        "local_structured",
    }
)
_OPERATION_KEYS = frozenset({"type", "filename", "title", "keywords", "summary"})
_SEMANTIC_KEYS = frozenset(
    {
        "record_id",
        "quote",
        "kind",
        "subject_quote",
        "scope_quotes",
        "condition_quotes",
        "source_text_sha256",
        "byte_coordinate_space",
        "quote_span",
        "subject_span",
        "scope_spans",
        "condition_spans",
    }
)
_SOURCE_RECORD_KEYS = frozenset({"record_id", "source_text_sha256", "text_bytes"})
_BINDING_KEYS = frozenset(
    {
        "record_id",
        "source_record_index",
        "raw_id",
        "raw_sha256",
        "receipt_sha256",
        "source_record_sha256",
        "source_text_sha256",
        "range_sha256",
        "byte_range",
        "byte_coordinate_space",
        "source_line",
        "source_character_start",
        "source_character_end",
        "source_record_id",
        "native_text_sha256",
    }
)
_REQUIRED_BINDING_KEYS = _BINDING_KEYS - {"source_record_id"}
_ENVELOPE_KEYS = frozenset(
    {
        "schema",
        "version",
        "projection_id",
        "source_sha256",
        "source_records",
        "source_records_sha256",
        "source_bindings",
        "source_bindings_sha256",
        "triage_result",
        "model_output_sha256",
        "schema_sha256",
        "prompt_sha256",
        "request_sha256",
        "semantic_evidence_authority",
    }
)


class C2SemanticProjectionError(ValueError):
    """A C2 envelope or source binding failed closed validation."""


def _sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise C2SemanticProjectionError(f"{field} must be a lowercase SHA-256")
    return value


def _nonempty(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise C2SemanticProjectionError(f"{field} must be a non-empty string")
    return value.strip()


def _strict_int(value: object, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise C2SemanticProjectionError(f"{field} must be an integer >= {minimum}")
    return value


def _strict_fields(value: object, fields: frozenset[str], field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise C2SemanticProjectionError(f"{field} fields are invalid")
    return value


def _span(text: str, quote: str) -> dict[str, Any]:
    try:
        payload = quote_span_payload(text, quote)
    except ValueError as exc:
        raise C2SemanticProjectionError(str(exc)) from exc
    if payload is None:
        raise C2SemanticProjectionError("C2 source quote must be non-empty")
    return payload


def _validate_span(
    text: str | None,
    value: object,
    quote: object,
    field: str,
    *,
    max_bytes: int | None = None,
) -> None:
    if quote is None:
        if value is not None:
            raise C2SemanticProjectionError(f"{field} must be null when quote is null")
        return
    if not isinstance(quote, str) or not quote or len(quote) > TRIAGE_C2_MAX_QUOTE_CHARS:
        raise C2SemanticProjectionError(f"{field} quote is invalid")
    if text is not None:
        expected = _span(text, quote)
    else:
        if not isinstance(value, Mapping) or set(value) != {
            "byte_range",
            "byte_coordinate_space",
            "span_sha256",
        }:
            raise C2SemanticProjectionError(f"{field} span is invalid")
        byte_range = value["byte_range"]
        if (
            not isinstance(byte_range, list)
            or len(byte_range) != 2
            or any(isinstance(item, bool) or not isinstance(item, int) for item in byte_range)
            or byte_range[0] < 0
            or byte_range[1] <= byte_range[0]
            or byte_range[1] - byte_range[0] != len(quote.encode("utf-8"))
            or value["byte_coordinate_space"] != "decoded_source_text_utf8"
            or value["span_sha256"] != hashlib.sha256(quote.encode("utf-8")).hexdigest()
        ):
            raise C2SemanticProjectionError(f"{field} span is invalid")
        if max_bytes is not None and value["byte_range"][1] > max_bytes:
            raise C2SemanticProjectionError(f"{field} span exceeds source bytes")
        return
    if value != expected:
        raise C2SemanticProjectionError(f"{field} span is invalid")
    if max_bytes is not None and expected["byte_range"][1] > max_bytes:
        raise C2SemanticProjectionError(f"{field} span exceeds source bytes")


def _validate_quote_list(
    text: str | None,
    quotes: object,
    spans: object,
    field: str,
    *,
    max_bytes: int | None = None,
) -> None:
    if quotes is None:
        if spans is not None:
            raise C2SemanticProjectionError(f"{field} spans must be null when quotes are null")
        return
    if not isinstance(quotes, list) or not isinstance(spans, list):
        raise C2SemanticProjectionError(f"{field} must have quote and span lists")
    if len(quotes) != len(spans) or len(quotes) > TRIAGE_C2_MAX_QUOTE_LIST:
        raise C2SemanticProjectionError(f"{field} exceeds its fixed list limit")
    if len(set(quotes)) != len(quotes):
        raise C2SemanticProjectionError(f"{field} contains duplicate quotes")
    for index, quote in enumerate(quotes):
        if not isinstance(quote, str) or not quote or len(quote) > TRIAGE_C2_MAX_QUOTE_CHARS:
            raise C2SemanticProjectionError(f"{field}[{index}] quote is invalid")
        _validate_span(
            text,
            spans[index],
            quote,
            f"{field}[{index}]",
            max_bytes=max_bytes,
        )


def _normalize_source_records(
    source_records: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    if isinstance(source_records, (str, bytes, bytearray)) or not isinstance(
        source_records, Sequence
    ) or not source_records:
        raise C2SemanticProjectionError("source_records must be a non-empty sequence")
    if len(source_records) > TRIAGE_C2_MAX_SOURCE_RECORDS:
        raise C2SemanticProjectionError("source_records exceed the fixed record limit")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, value in enumerate(source_records):
        row = _strict_fields(value, frozenset({"record_id", "text"}), f"source_records[{index}]")
        record_id = row["record_id"]
        text = row["text"]
        if (
            not isinstance(record_id, str)
            or not record_id
            or record_id != record_id.strip()
            or len(record_id) > TRIAGE_C2_MAX_SOURCE_RECORD_ID_CHARS
            or any(ord(char) < 0x20 or char == "\x7f" for char in record_id)
        ):
            raise C2SemanticProjectionError(f"source_records[{index}] record_id is invalid")
        if record_id in seen:
            raise C2SemanticProjectionError(f"source record_id is duplicated: {record_id}")
        if not isinstance(text, str) or not text or len(text) > TRIAGE_C2_MAX_SOURCE_RECORD_TEXT_CHARS:
            raise C2SemanticProjectionError(f"source record {record_id} text is invalid")
        try:
            record_id.encode("utf-8")
            text_bytes = text.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise C2SemanticProjectionError(f"source record {record_id} is not valid UTF-8") from exc
        seen.add(record_id)
        result.append(
            {
                "record_id": record_id,
                "text": text,
                "source_text_sha256": hashlib.sha256(text_bytes).hexdigest(),
            }
        )
    return tuple(result)


def _source_records_sha256(records: Sequence[Mapping[str, str]]) -> str:
    return canonical_json_sha256_strict(
        [{"record_id": row["record_id"], "text": row["text"]} for row in records]
    )


def _validate_binding_row(
    value: object,
    index: int,
    *,
    expected_record_id: str | None = None,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) not in {
        _REQUIRED_BINDING_KEYS,
        _BINDING_KEYS,
    }:
        raise C2SemanticProjectionError(f"source_bindings[{index}] fields are invalid")
    row = dict(value)
    if expected_record_id is not None and row.get("record_id") != expected_record_id:
        raise C2SemanticProjectionError(f"source_bindings[{index}] record order is invalid")
    _nonempty(row.get("record_id"), f"source_bindings[{index}].record_id")
    _nonempty(row.get("raw_id"), f"source_bindings[{index}].raw_id")
    for name in (
        "raw_sha256",
        "receipt_sha256",
        "source_record_sha256",
        "source_text_sha256",
        "native_text_sha256",
        "range_sha256",
    ):
        _sha256(row.get(name), f"source_bindings[{index}].{name}")
    _strict_int(row.get("source_record_index"), f"source_bindings[{index}].source_record_index")
    _strict_int(row.get("source_line"), f"source_bindings[{index}].source_line", minimum=1)
    start = _strict_int(row.get("source_character_start"), f"source_bindings[{index}].source_character_start")
    end = _strict_int(row.get("source_character_end"), f"source_bindings[{index}].source_character_end")
    if end <= start:
        raise C2SemanticProjectionError(f"source_bindings[{index}] decoded range is invalid")
    raw_range = row.get("byte_range")
    if (
        not isinstance(raw_range, list)
        or len(raw_range) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in raw_range)
        or raw_range[0] < 0
        or raw_range[1] <= raw_range[0]
        or row.get("byte_coordinate_space") != "logical_raw"
    ):
        raise C2SemanticProjectionError(f"source_bindings[{index}] native range is invalid")
    if "source_record_id" in row and row["source_record_id"] is not None:
        _nonempty(row["source_record_id"], f"source_bindings[{index}].source_record_id")
    return row


def _normalize_bindings(
    source_bindings: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    if isinstance(source_bindings, (str, bytes, bytearray)) or not isinstance(
        source_bindings, Sequence
    ) or len(source_bindings) != len(records):
        raise C2SemanticProjectionError("source_bindings must align one-to-one with source_records")
    normalized: list[dict[str, Any]] = []
    record_ids = [row["record_id"] for row in records]
    for index, value in enumerate(source_bindings):
        row = _validate_binding_row(
            value,
            index,
            expected_record_id=record_ids[index],
        )
        if row["source_text_sha256"] != records[index]["source_text_sha256"]:
            raise C2SemanticProjectionError(
                f"source_bindings[{index}] selected text hash is invalid"
            )
        normalized.append(row)
    return tuple(normalized)


def _validate_operation(value: object, index: int) -> dict[str, Any]:
    row = dict(_strict_fields(value, _OPERATION_KEYS, f"operations[{index}]"))
    if row["type"] not in {"create", "update"}:
        raise C2SemanticProjectionError(f"operations[{index}].type is invalid")
    for field, maximum in (("filename", 200), ("title", 300), ("summary", 2_000)):
        if not isinstance(row[field], str) or not row[field].strip() or len(row[field]) > maximum:
            raise C2SemanticProjectionError(f"operations[{index}].{field} is invalid")
    keywords = row["keywords"]
    if (
        not isinstance(keywords, list)
        or not 1 <= len(keywords) <= 32
        or any(not isinstance(item, str) or not item.strip() or len(item) > 200 for item in keywords)
    ):
        raise C2SemanticProjectionError(f"operations[{index}].keywords is invalid")
    return row


def _validate_audit(audit: object, record_count: int) -> dict[str, Any]:
    row = dict(_strict_fields(audit, _AUDIT_KEYS, "triage_result.audit"))
    if row["schema_version"] != TRIAGE_C2_SCHEMA_VERSION:
        raise C2SemanticProjectionError("triage C2 schema version is invalid")
    if row["schema_sha256"] != canonical_json_sha256_strict(TRIAGE_C2_SCHEMA):
        raise C2SemanticProjectionError("triage C2 schema hash is invalid")
    for field in ("schema_sha256", "prompt_sha256", "request_sha256", "source_records_sha256"):
        _sha256(row[field], f"triage_result.audit.{field}")
    if row["source_record_count"] != record_count:
        raise C2SemanticProjectionError("triage C2 source record count is invalid")
    if row["semantic_evidence_authority"] != "model_judgment":
        raise C2SemanticProjectionError("C2 semantic evidence authority is invalid")
    local = row["local_structured"]
    if not isinstance(local, Mapping) or local.get("ok") is not True:
        raise C2SemanticProjectionError("triage C2 local structured audit is invalid")
    return row


def _validate_semantic_row(
    value: object,
    index: int,
    records_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    row = dict(_strict_fields(value, _SEMANTIC_KEYS, f"semantic_evidence[{index}]"))
    record_id = row["record_id"]
    if not isinstance(record_id, str) or record_id not in records_by_id:
        raise C2SemanticProjectionError(f"semantic_evidence[{index}].record_id is unknown")
    text_value = records_by_id[record_id].get("text")
    text = text_value if isinstance(text_value, str) else None
    _sha256(row["source_text_sha256"], f"semantic_evidence[{index}].source_text_sha256")
    if row["source_text_sha256"] != records_by_id[record_id]["source_text_sha256"]:
        raise C2SemanticProjectionError(f"semantic_evidence[{index}] source text hash is invalid")
    if row["byte_coordinate_space"] != "decoded_source_text_utf8":
        raise C2SemanticProjectionError(f"semantic_evidence[{index}] coordinate space is invalid")
    kind = row["kind"]
    if kind not in _KINDS:
        raise C2SemanticProjectionError(f"semantic_evidence[{index}].kind is invalid")
    quote = row["quote"]
    if quote is not None and (not isinstance(quote, str) or not quote or len(quote) > TRIAGE_C2_MAX_QUOTE_CHARS):
        raise C2SemanticProjectionError(f"semantic_evidence[{index}].quote is invalid")
    if kind != "unknown" and quote is None:
        raise C2SemanticProjectionError(f"semantic_evidence[{index}] known kind requires a quote")
    text_bytes = (
        len(text.encode("utf-8"))
        if text is not None
        else records_by_id[record_id].get("text_bytes")
    )
    if isinstance(text_bytes, bool) or not isinstance(text_bytes, int) or text_bytes < 1:
        raise C2SemanticProjectionError(f"semantic_evidence[{index}] source byte length is invalid")
    _validate_span(
        text,
        row["quote_span"],
        quote,
        f"semantic_evidence[{index}].quote",
        max_bytes=text_bytes,
    )
    subject = row["subject_quote"]
    if subject is not None and (not isinstance(subject, str) or not subject or len(subject) > TRIAGE_C2_MAX_QUOTE_CHARS):
        raise C2SemanticProjectionError(f"semantic_evidence[{index}].subject_quote is invalid")
    _validate_span(
        text,
        row["subject_span"],
        subject,
        f"semantic_evidence[{index}].subject",
        max_bytes=text_bytes,
    )
    _validate_quote_list(
        text,
        row["scope_quotes"],
        row["scope_spans"],
        f"semantic_evidence[{index}].scope_quotes",
        max_bytes=text_bytes,
    )
    _validate_quote_list(
        text,
        row["condition_quotes"],
        row["condition_spans"],
        f"semantic_evidence[{index}].condition_quotes",
        max_bytes=text_bytes,
    )
    return row


def _model_output(value: Mapping[str, Any]) -> dict[str, Any]:
    return {"operations": value["operations"], "semantic_evidence": value["semantic_evidence"]}


def _validate_envelope(value: object) -> dict[str, Any]:
    root = dict(_strict_fields(value, _ENVELOPE_KEYS, "C2 semantic envelope"))
    if root["schema"] != C2_SEMANTIC_ENVELOPE_SCHEMA or root["version"] != C2_SEMANTIC_ENVELOPE_VERSION:
        raise C2SemanticProjectionError("C2 semantic envelope schema is invalid")
    _sha256(root["projection_id"], "projection_id")
    _sha256(root["source_sha256"], "source_sha256")
    records = root["source_records"]
    if not isinstance(records, list) or not records:
        raise C2SemanticProjectionError("C2 source_records are invalid")
    for index, record in enumerate(records):
        row = dict(_strict_fields(record, _SOURCE_RECORD_KEYS, f"source_records[{index}]"))
        _nonempty(row["record_id"], f"source_records[{index}].record_id")
        _sha256(row["source_text_sha256"], f"source_records[{index}].source_text_sha256")
        _strict_int(row["text_bytes"], f"source_records[{index}].text_bytes", minimum=1)
    _sha256(root["source_records_sha256"], "source_records_sha256")
    _sha256(root["source_bindings_sha256"], "source_bindings_sha256")
    bindings = root["source_bindings"]
    if not isinstance(bindings, list) or len(bindings) != len(records):
        raise C2SemanticProjectionError("C2 source_bindings are invalid")
    for index, binding in enumerate(bindings):
        expected_record_id = records[index].get("record_id")
        binding_row = _validate_binding_row(
            binding,
            index,
            expected_record_id=expected_record_id
            if isinstance(expected_record_id, str)
            else None,
        )
        if binding_row["source_text_sha256"] != records[index]["source_text_sha256"]:
            raise C2SemanticProjectionError(
                f"source_bindings[{index}] selected text hash is invalid"
            )
    triage = root["triage_result"]
    if not isinstance(triage, Mapping) or set(triage) != _TRIAGE_KEYS:
        raise C2SemanticProjectionError("triage_result fields are invalid")
    operations = triage["operations"]
    rows = triage["semantic_evidence"]
    if not isinstance(operations, list) or len(operations) > TRIAGE_MAX_OPERATIONS:
        raise C2SemanticProjectionError("triage operations are invalid")
    for index, operation in enumerate(operations):
        _validate_operation(operation, index)
    if not isinstance(rows, list) or len(rows) > TRIAGE_C2_MAX_SEMANTIC_ROWS:
        raise C2SemanticProjectionError("triage semantic evidence is invalid")
    metadata_by_id = {str(row["record_id"]): row for row in records}
    if len(metadata_by_id) != len(records):
        raise C2SemanticProjectionError("C2 source record IDs are duplicated")
    for index, row in enumerate(rows):
        semantic = _validate_semantic_row(row, index, metadata_by_id)
        if not isinstance(semantic["record_id"], str):
            raise C2SemanticProjectionError("semantic row record_id is invalid")
    audit = _validate_audit(triage["audit"], len(records))
    if audit["source_records_sha256"] != root["source_records_sha256"]:
        raise C2SemanticProjectionError("triage/source record hash mismatch")
    if root["semantic_evidence_authority"] != "model_judgment":
        raise C2SemanticProjectionError("C2 semantic authority is invalid")
    if root["schema_sha256"] != audit["schema_sha256"]:
        raise C2SemanticProjectionError("C2 schema hash mismatch")
    _sha256(root["schema_sha256"], "schema_sha256")
    for field in ("prompt_sha256", "request_sha256"):
        _sha256(root[field], field)
        if root[field] != audit[field]:
            raise C2SemanticProjectionError(f"C2 {field} mismatch")
    if root["model_output_sha256"] != canonical_json_sha256_strict(_model_output(triage)):
        raise C2SemanticProjectionError("C2 model output hash mismatch")
    if root["source_bindings_sha256"] != canonical_json_sha256_strict(bindings):
        raise C2SemanticProjectionError("C2 source binding hash mismatch")
    return root


def materialize_c2_semantic_envelope(
    triage_result: Mapping[str, Any],
    source_records: Sequence[Mapping[str, Any]],
    source_bindings: Sequence[Mapping[str, Any]],
    *,
    projection_id: str,
    source_sha256: str,
) -> dict[str, Any]:
    """Build a canonical C2 envelope from host-validated triage output.

    ``source_records`` are decoded child text supplied by the host.  They are
    used to validate every model quote, but only their IDs, byte lengths and
    digests are persisted.  ``source_bindings`` retain the native Raw range
    and receipt identities separately from decoded UTF-8 quote spans.
    """

    records = _normalize_source_records(source_records)
    bindings = _normalize_bindings(source_bindings, records)
    triage = dict(_strict_fields(triage_result, _TRIAGE_KEYS, "triage_result"))
    operations = triage["operations"]
    rows = triage["semantic_evidence"]
    if not isinstance(operations, list) or len(operations) > TRIAGE_MAX_OPERATIONS:
        raise C2SemanticProjectionError("triage operations are invalid")
    for index, operation in enumerate(operations):
        _validate_operation(operation, index)
    if not isinstance(rows, list) or len(rows) > TRIAGE_C2_MAX_SEMANTIC_ROWS:
        raise C2SemanticProjectionError("triage semantic evidence is invalid")
    source_records_sha256 = _source_records_sha256(records)
    audit = _validate_audit(triage["audit"], len(records))
    if audit["source_records_sha256"] != source_records_sha256:
        raise C2SemanticProjectionError("triage/source record hash mismatch")
    metadata = [
        {
            "record_id": row["record_id"],
            "source_text_sha256": row["source_text_sha256"],
            "text_bytes": len(row["text"].encode("utf-8")),
        }
        for row in records
    ]
    records_by_id = {row["record_id"]: row for row in records}
    for index, row in enumerate(rows):
        _validate_semantic_row(row, index, records_by_id)
    output = {
        "schema": C2_SEMANTIC_ENVELOPE_SCHEMA,
        "version": C2_SEMANTIC_ENVELOPE_VERSION,
        "projection_id": _sha256(projection_id, "projection_id"),
        "source_sha256": _sha256(source_sha256, "source_sha256"),
        "source_records": metadata,
        "source_records_sha256": source_records_sha256,
        "source_bindings": [dict(row) for row in bindings],
        "source_bindings_sha256": canonical_json_sha256_strict(bindings),
        "triage_result": triage,
        "model_output_sha256": canonical_json_sha256_strict(_model_output(triage)),
        "schema_sha256": audit["schema_sha256"],
        "prompt_sha256": audit["prompt_sha256"],
        "request_sha256": audit["request_sha256"],
        "semantic_evidence_authority": "model_judgment",
    }
    return _validate_envelope(output)


def store_c2_semantic_envelope(
    store: ResearchStore,
    envelope: Mapping[str, Any],
    *,
    source_uri: str,
) -> EvidenceArtifact:
    """Store one validated envelope through the existing durable CAS."""

    if not isinstance(store, ResearchStore):
        raise C2SemanticProjectionError("research store is invalid")
    validated = _validate_envelope(envelope)
    uri = _nonempty(source_uri, "source_uri")
    raw = canonical_json_line_bytes_strict(validated)
    artifact = store.put_artifact(
        raw,
        source_type=C2_SEMANTIC_ENVELOPE_SCHEMA,
        source_uri=uri,
        title="Chronovisor C2 semantic evidence",
        mime_type="application/json",
        trust="model_judgment",
        durable=True,
        metadata={
            "schema": C2_SEMANTIC_ENVELOPE_SCHEMA,
            "projection_id": validated["projection_id"],
            "source_sha256": validated["source_sha256"],
            "source_records_sha256": validated["source_records_sha256"],
            "source_bindings_sha256": validated["source_bindings_sha256"],
            "model_output_sha256": validated["model_output_sha256"],
        },
    )
    expected_sha256 = hashlib.sha256(raw).hexdigest()
    if artifact.sha256 != expected_sha256 or artifact.byte_length != len(raw) or not artifact.durable:
        raise C2SemanticProjectionError("C2 CAS artifact identity is invalid")
    return artifact


def load_c2_semantic_envelope(
    store: ResearchStore,
    artifact_id: str,
) -> dict[str, Any]:
    """Read one CAS envelope and verify canonical bytes plus its manifest."""

    if not isinstance(store, ResearchStore):
        raise C2SemanticProjectionError("research store is invalid")
    try:
        raw = store.read_artifact(artifact_id)
    except Exception as exc:
        raise C2SemanticProjectionError("C2 CAS artifact cannot be read") from exc
    try:
        import json

        value = json.loads(raw)
    except (TypeError, ValueError, UnicodeDecodeError) as exc:
        raise C2SemanticProjectionError("C2 CAS envelope is invalid JSON") from exc
    if canonical_json_line_bytes_strict(value) != raw:
        raise C2SemanticProjectionError("C2 CAS envelope is not canonical")
    envelope = _validate_envelope(value)
    manifest = store.artifact_manifest(artifact_id)
    if manifest is None:
        raise C2SemanticProjectionError("C2 CAS manifest is missing")
    if (
        not manifest.durable
        or manifest.sha256 != hashlib.sha256(raw).hexdigest()
        or manifest.byte_length != len(raw)
        or manifest.source_type != C2_SEMANTIC_ENVELOPE_SCHEMA
        or manifest.metadata != {
            "schema": C2_SEMANTIC_ENVELOPE_SCHEMA,
            "projection_id": envelope["projection_id"],
            "source_sha256": envelope["source_sha256"],
            "source_records_sha256": envelope["source_records_sha256"],
            "source_bindings_sha256": envelope["source_bindings_sha256"],
            "model_output_sha256": envelope["model_output_sha256"],
        }
    ):
        raise C2SemanticProjectionError("C2 CAS manifest identity is invalid")
    return envelope


def _resolve_native_raw(raw_dir: Path, raw_id: str) -> tuple[RawStore, Any]:
    # The binding is external input.  Reject path syntax before joining it to
    # either allowed root, so ``../`` and absolute names can never escape the
    # named Raw-reference lookup.
    if Path(raw_id).name != raw_id or raw_id in {".", ".."}:
        raise C2SemanticProjectionError("C2 native Raw ID is invalid")
    store = RawStore(raw_dir, mode="v2")
    for directory in (
        raw_dir,
        raw_dir.parent / "runtime" / "raw-projections" / "parents",
    ):
        reference = directory / raw_id
        if not reference.is_file() or reference.is_symlink():
            continue
        try:
            unit = store.resolve_reference(reference)
        except (OSError, ValueError, RawSegmentCorrupt) as exc:
            raise C2SemanticProjectionError("C2 native Raw reference is invalid") from exc
        if unit is not None:
            return store, unit
    raise C2SemanticProjectionError("C2 native Raw reference is unavailable")


def _validate_native_source_segment(
    raw_dir: Path,
    base: EvidenceAtom,
    binding: Mapping[str, Any],
    source_text: str,
) -> None:
    """Rebuild the named native event and verify decoded child coordinates."""

    try:
        store, unit = _resolve_native_raw(raw_dir, binding["raw_id"])
        commit = unit.commit
        if (
            commit is None
            or unit.sha256 is None
            or unit.raw_id != binding["raw_id"]
            or commit.raw_id != binding["raw_id"]
            or commit.sha256 != binding["raw_sha256"]
            or unit.sha256 != binding["raw_sha256"]
        ):
            raise C2SemanticProjectionError("C2 native Raw commit identity is invalid")
        raw = store.read_bytes(unit)
        if hashlib.sha256(raw).hexdigest() != binding["raw_sha256"]:
            raise C2SemanticProjectionError("C2 native Raw bytes disagree with binding")
        receipt_sha256 = canonical_json_sha256_strict(commit.to_dict())
        if receipt_sha256 != binding["receipt_sha256"]:
            raise C2SemanticProjectionError("C2 native Raw receipt disagrees with binding")
        spans = committed_event_spans(raw, commit.record_count)
        source_index = binding["source_record_index"]
        if source_index >= len(spans):
            raise C2SemanticProjectionError("C2 native Raw event index is invalid")
        native_start, encoded_event = spans[source_index]
        if binding["byte_range"] != [native_start, native_start + len(encoded_event)]:
            raise C2SemanticProjectionError("C2 native Raw byte range disagrees with binding")
        if hashlib.sha256(encoded_event).hexdigest() != binding["range_sha256"]:
            raise C2SemanticProjectionError("C2 native Raw event range hash disagrees")
        expected_line = commit.after_line + source_index + 1
        if binding["source_line"] != expected_line:
            raise C2SemanticProjectionError("C2 native Raw source line disagrees with binding")
        event = json.loads(encoded_event.decode("utf-8"))
        if not isinstance(event, Mapping):
            raise C2SemanticProjectionError("C2 native Raw event is invalid")
        role, text = _event_semantics(commit.host, event)
        if role != base.provenance.source_role or text.strip() != base.claim:
            raise C2SemanticProjectionError("C2 native Raw decoded claim disagrees")
        expected = _projection_atom_c2(
            raw_id=unit.raw_id,
            raw_sha256=unit.sha256,
            receipt_sha256=receipt_sha256,
            host=commit.host,
            session_key=commit.session_key,
            captured_at=commit.captured_at,
            line=encoded_event,
            index=source_index,
            start=native_start,
        )
        if expected is None or expected != base:
            raise C2SemanticProjectionError(
                "C2 native Raw atom metadata disagrees with base projection"
            )
        if hashlib.sha256(text.encode("utf-8")).hexdigest() != binding["native_text_sha256"]:
            raise C2SemanticProjectionError("C2 native Raw decoded text hash disagrees")
        start = binding["source_character_start"]
        end = binding["source_character_end"]
        if start < 0 or end > len(text) or end <= start or text[start:end] != source_text:
            raise C2SemanticProjectionError("C2 native Raw decoded source range disagrees")
    except C2SemanticProjectionError:
        raise
    except (OSError, ValueError, IndexError, UnicodeDecodeError, json.JSONDecodeError, RawSegmentCorrupt) as exc:
        raise C2SemanticProjectionError("C2 native Raw source verification failed") from exc


def _base_atom_for_binding(
    base_projection: EpisodeProjection,
    binding: Mapping[str, Any],
) -> EvidenceAtom:
    raw_id = binding["raw_id"]
    byte_range = binding["byte_range"]
    matches = tuple(
        atom
        for atom in base_projection.atoms
        if atom.evidence.raw_id == raw_id
        and atom.evidence.byte_start == byte_range[0]
        and atom.evidence.byte_end == byte_range[1]
        and atom.evidence.raw_sha256 == binding["raw_sha256"]
        and atom.evidence.receipt_sha256 == binding["receipt_sha256"]
    )
    if len(matches) != 1:
        raise C2SemanticProjectionError("C2 semantic row has no unique base Raw atom")
    atom = matches[0]
    if (
        atom.recorded_at is None
        or atom.to_dict().get("schema") != "chronovisor.evidence-atom.v2"
    ):
        raise C2SemanticProjectionError("C2 semantic row Raw identity disagrees with base atom")
    receipts = tuple(
        receipt
        for receipt in base_projection.source_receipts
        if receipt.get("raw_id") == raw_id
        and receipt.get("raw_sha256") == binding["raw_sha256"]
        and receipt.get("receipt_sha256") == binding["receipt_sha256"]
    )
    if len(receipts) != 1:
        raise C2SemanticProjectionError("C2 semantic row Raw receipt is not unique")
    line_range = receipts[0].get("source_line_range")
    source_line = binding["source_line"]
    if (
        not isinstance(line_range, list)
        or len(line_range) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in line_range)
        or not line_range[0] < source_line <= line_range[1]
    ):
        raise C2SemanticProjectionError("C2 semantic row source line is outside its receipt")
    return atom


def build_c2_semantic_projection(
    envelope: Mapping[str, Any],
    base_projection: EpisodeProjection,
    *,
    source_records: Sequence[Mapping[str, Any]],
    raw_dir: Path,
) -> EpisodeProjection:
    """Select verified native C2 atoms for semantic rows.

    The native atom remains unchanged, so existing Raw verification and
    bounded packet serialization continue to cover the complete event text.
    C2 semantic fields stay in the envelope as model judgement metadata.
    """

    validated = _validate_envelope(envelope)
    if not isinstance(base_projection, EpisodeProjection) or base_projection.schema != EPISODE_PROJECTION_C2_SCHEMA:
        raise C2SemanticProjectionError("base projection must be an explicit C2 projection")
    records = _normalize_source_records(source_records)
    if [row["record_id"] for row in records] != [row["record_id"] for row in validated["source_records"]]:
        raise C2SemanticProjectionError("C2 source record IDs changed before consumption")
    if _source_records_sha256(records) != validated["source_records_sha256"]:
        raise C2SemanticProjectionError("C2 source records changed before consumption")
    records_by_id = {row["record_id"]: row for row in records}
    bindings = validated["source_bindings"]
    binding_by_id = {row["record_id"]: row for row in bindings}
    if len(binding_by_id) != len(bindings):
        raise C2SemanticProjectionError("C2 source binding IDs are duplicated")
    if not isinstance(raw_dir, Path):
        raise C2SemanticProjectionError("C2 raw_dir is invalid")
    atoms_by_id: dict[str, EvidenceAtom] = {}
    for index, row in enumerate(validated["triage_result"]["semantic_evidence"]):
        semantic = _validate_semantic_row(row, index, records_by_id)
        quote = semantic["quote"]
        if quote is None:
            continue
        binding = binding_by_id.get(semantic["record_id"])
        if binding is None:
            raise C2SemanticProjectionError("C2 semantic row has no source binding")
        base = _base_atom_for_binding(base_projection, binding)
        source_text = records_by_id[semantic["record_id"]]["text"]
        if not isinstance(source_text, str):
            raise C2SemanticProjectionError("C2 source text is unavailable")
        _validate_native_source_segment(raw_dir, base, binding, source_text)
        # The quote and all optional subject/scope/condition fields were
        # validated against this exact native decoded segment above.  Keep the
        # canonical native atom as the consumer value; no synthetic claim or
        # relation is introduced by the model metadata.
        atoms_by_id[base.atom_id] = base
    atom_rows = tuple(sorted(atoms_by_id.values(), key=lambda atom: atom.atom_id))
    projection_unsigned = base_projection.to_dict()
    projection_unsigned.pop("projection_id")
    projection_unsigned["atoms"] = [atom.to_dict() for atom in atom_rows]
    # Retain the existing projection identity and loader contract. The CAS
    # envelope separately retains the model's selection and its audit hashes.
    return load_episode_projection_c2(
        canonical_json_line_bytes_strict(
            {
                "projection_id": _identity("projection", projection_unsigned),
                **projection_unsigned,
            }
        )
    )


def run_c2_semantic_retrieval(
    envelope: Mapping[str, Any],
    base_projection: EpisodeProjection,
    *,
    source_records: Sequence[Mapping[str, Any]],
    program: RetrievalProgram,
    actions: Sequence[tuple[str, Any]] = (),
    raw_dir: Path,
    deadline_ms: int = 4_000,
) -> Any:
    """Run the existing C2 EvidenceLedger against bound semantic atoms."""

    projection = build_c2_semantic_projection(
        envelope,
        base_projection,
        source_records=source_records,
        raw_dir=raw_dir,
    )
    from chronovisor.research.evidence_runtime import run_evidence_retrieval_c2

    return run_evidence_retrieval_c2(
        program,
        projection,
        actions=actions,
        raw_dir=raw_dir,
        deadline_ms=deadline_ms,
    )


__all__ = [
    "C2_SEMANTIC_ENVELOPE_SCHEMA",
    "C2_SEMANTIC_ENVELOPE_VERSION",
    "C2SemanticProjectionError",
    "build_c2_semantic_projection",
    "load_c2_semantic_envelope",
    "materialize_c2_semantic_envelope",
    "run_c2_semantic_retrieval",
    "store_c2_semantic_envelope",
]
