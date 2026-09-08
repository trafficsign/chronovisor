"""Versioned, source-backed projections for canonical Wiki pages.

Page projections are retrieval candidates.  They deliberately do not become
Raw conversation evidence: a page has no committed Raw receipt, assistant
authority, or trustworthy validity interval by virtue of being indexed.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from chronovisor.core.canonical_document import CanonicalDocumentError, parse_document
from chronovisor.core.canonical_json import canonical_json_line_bytes_strict
from chronovisor.core.page_identity import normalize_page_uid
from chronovisor.core.search_types import tokenize
from chronovisor.search.research_store import ResearchStore
from chronovisor.search.research_types import EvidenceArtifact

PAGE_EVIDENCE_PROJECTION_SCHEMA = "chronovisor.page-evidence-projection.v1"
PAGE_EVIDENCE_TRANSFORM_VERSION = "semantic-section-v1"
MAX_SECTION_PIECE_CHARS = 900
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")
_FENCE_OPEN_RE = re.compile(
    r"^[ \t]{0,3}(?P<fence>`{3,}|~{3,})[^\r\n]*(?:\r?\n|$)"
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class PageEvidenceProjectionError(ValueError):
    """A page projection is malformed, stale, or not source-backed."""


@dataclass(frozen=True, slots=True)
class PageEvidenceRecord:
    """One bounded source slice and its deterministic retrieval context."""

    record_id: str
    page_id: str
    page_uid: str
    content_sha256: str
    section_ordinal: int
    piece_ordinal: int
    section_context: tuple[str, ...]
    search_keys: tuple[str, ...]
    byte_start: int
    byte_end: int
    span_sha256: str
    byte_coordinate_space: str = "canonical_source"

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "page_id": self.page_id,
            "page_uid": self.page_uid,
            "content_sha256": self.content_sha256,
            "section_ordinal": self.section_ordinal,
            "piece_ordinal": self.piece_ordinal,
            "section_context": list(self.section_context),
            "search_keys": list(self.search_keys),
            "byte_range": [self.byte_start, self.byte_end],
            "byte_coordinate_space": self.byte_coordinate_space,
            "span_sha256": self.span_sha256,
        }


@dataclass(frozen=True, slots=True)
class PageEvidenceProjection:
    """Canonical page-source candidate artifact without generated claims."""

    projection_id: str
    page_id: str
    page_uid: str
    content_sha256: str
    source_body_byte_range: tuple[int, int]
    source_updated: str | None
    recorded_at: str | None
    valid_from: str | None
    valid_to: str | None
    transform_version: str
    records: tuple[PageEvidenceRecord, ...]

    @property
    def schema(self) -> str:
        return PAGE_EVIDENCE_PROJECTION_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "projection_id": self.projection_id,
            "page_id": self.page_id,
            "page_uid": self.page_uid,
            "content_sha256": self.content_sha256,
            "source_body_byte_range": list(self.source_body_byte_range),
            "source_updated": self.source_updated,
            "recorded_at": self.recorded_at,
            "valid_from": self.valid_from,
            "valid_to": self.valid_to,
            "transform_version": self.transform_version,
            "records": [record.to_dict() for record in self.records],
        }

    def canonical_bytes(self) -> bytes:
        return canonical_json_line_bytes_strict(self.to_dict())


def _optional_uid(value: object) -> str | None:
    if value is None or value == "":
        return ""
    try:
        return normalize_page_uid(value)
    except (TypeError, ValueError):
        return None


def _title(metadata: Mapping[str, Any], page_id: str) -> str:
    value = metadata.get("title")
    return value.strip() or page_id if isinstance(value, str) else page_id


def _metadata_keys(
    metadata: Mapping[str, Any], title: str, section_context: tuple[str, ...]
) -> tuple[str, ...]:
    values: list[str] = [title, *section_context]
    for key in ("description", "summary", "page_type", "updated"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            values.append(value)
    entities = metadata.get("entities")
    if isinstance(entities, list):
        values.extend(value for value in entities[:8] if isinstance(value, str))
    recall_questions = metadata.get("recall_questions")
    if isinstance(recall_questions, list):
        # Keep this in lockstep with ``search._recall_questions``: these are
        # retrieval keys only and never source-backed text or claims.
        valid_questions = [
            question
            for question in recall_questions
            if isinstance(question, str) and question.strip()
        ][:8]
        values.extend(valid_questions)
    return tuple(dict.fromkeys(tokenize("\n".join(values))))[:128]


def _body_sections(body: str) -> list[tuple[int, int, tuple[str, ...]]]:
    sections: list[tuple[int, int, tuple[str, ...]]] = []
    stack: list[tuple[int, str]] = []
    section_start = 0
    offset = 0
    fence_char: str | None = None
    fence_length = 0
    for raw_line in body.splitlines(keepends=True):
        if fence_char is not None:
            closing_line = raw_line.rstrip("\r\n")
            closing = re.fullmatch(
                rf"[ \t]{{0,3}}{re.escape(fence_char)}{{{fence_length},}}[ \t]*",
                closing_line,
            )
            if closing:
                fence_char = None
                fence_length = 0
            offset += len(raw_line)
            continue

        opening = _FENCE_OPEN_RE.match(raw_line)
        if opening:
            marker = opening.group("fence")
            fence_char = marker[0]
            fence_length = len(marker)
            offset += len(raw_line)
            continue

        line = raw_line.strip()
        heading = _HEADING_RE.match(line)
        if heading:
            if section_start < offset:
                sections.append(
                    (section_start, offset, tuple(title for _, title in stack))
                )
            level = len(heading.group(1))
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, heading.group(2).strip()))
            section_start = offset
        offset += len(raw_line)
    if section_start < len(body):
        sections.append(
            (section_start, len(body), tuple(title for _, title in stack))
        )
    return sections


def _char_to_byte_offsets(body: str) -> list[int]:
    offsets = [0]
    for char in body:
        offsets.append(offsets[-1] + len(char.encode("utf-8")))
    return offsets


def _record_id(payload: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(canonical_json_line_bytes_strict(dict(payload))).hexdigest()
    return f"page-record:{digest}"


def _projection_id(payload: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(canonical_json_line_bytes_strict(dict(payload))).hexdigest()
    return f"page-projection:{digest}"


def project_page_evidence(source: bytes, page_id: str) -> PageEvidenceProjection:
    """Project every canonical page body section into bounded source slices."""

    if not isinstance(source, bytes) or not isinstance(page_id, str) or not page_id.strip():
        raise PageEvidenceProjectionError("page source identity is invalid")
    try:
        document = parse_document(source)
        body = document.body.decode("utf-8")
    except (CanonicalDocumentError, UnicodeDecodeError) as exc:
        raise PageEvidenceProjectionError("page source is not canonical UTF-8") from exc
    if document.metadata.get("status") != "stable":
        raise PageEvidenceProjectionError("page source is not stable")
    page_uid = _optional_uid(document.metadata.get("uid"))
    if page_uid is None:
        raise PageEvidenceProjectionError("page source UID is invalid")
    content_sha256 = hashlib.sha256(source).hexdigest()
    body_start = len(source) - len(document.body)
    body_end = len(source)
    offsets = _char_to_byte_offsets(body)
    title = _title(document.metadata, page_id)
    source_updated_value = document.metadata.get("updated")
    source_updated = (
        source_updated_value.strip()
        if isinstance(source_updated_value, str) and source_updated_value.strip()
        else source_updated_value.isoformat()
        if isinstance(source_updated_value, date)
        else None
    )
    records: list[PageEvidenceRecord] = []
    for section_ordinal, (section_start, section_end, context) in enumerate(
        _body_sections(body)
    ):
        search_keys = _metadata_keys(document.metadata, title, context)
        for piece_ordinal, start_char in enumerate(
            range(section_start, section_end, MAX_SECTION_PIECE_CHARS)
        ):
            end_char = min(start_char + MAX_SECTION_PIECE_CHARS, section_end)
            byte_start = body_start + offsets[start_char]
            byte_end = body_start + offsets[end_char]
            span_sha256 = hashlib.sha256(source[byte_start:byte_end]).hexdigest()
            unsigned = {
                "page_id": page_id,
                "page_uid": page_uid,
                "content_sha256": content_sha256,
                "section_ordinal": section_ordinal,
                "piece_ordinal": piece_ordinal,
                "section_context": list(context),
                "search_keys": list(search_keys),
                "byte_range": [byte_start, byte_end],
                "byte_coordinate_space": "canonical_source",
                "span_sha256": span_sha256,
            }
            records.append(
                PageEvidenceRecord(
                    record_id=_record_id(unsigned),
                    page_id=page_id,
                    page_uid=page_uid,
                    content_sha256=content_sha256,
                    section_ordinal=section_ordinal,
                    piece_ordinal=piece_ordinal,
                    section_context=context,
                    search_keys=search_keys,
                    byte_start=byte_start,
                    byte_end=byte_end,
                    span_sha256=span_sha256,
                )
            )
    unsigned_projection = {
        "schema": PAGE_EVIDENCE_PROJECTION_SCHEMA,
        "page_id": page_id,
        "page_uid": page_uid,
        "content_sha256": content_sha256,
        "source_body_byte_range": [body_start, body_end],
        "source_updated": source_updated,
        "recorded_at": None,
        "valid_from": None,
        "valid_to": None,
        "transform_version": PAGE_EVIDENCE_TRANSFORM_VERSION,
        "records": [record.to_dict() for record in records],
    }
    return PageEvidenceProjection(
        projection_id=_projection_id(unsigned_projection),
        page_id=page_id,
        page_uid=page_uid,
        content_sha256=content_sha256,
        source_body_byte_range=(body_start, body_end),
        source_updated=source_updated,
        recorded_at=None,
        valid_from=None,
        valid_to=None,
        transform_version=PAGE_EVIDENCE_TRANSFORM_VERSION,
        records=tuple(records),
    )


def build_page_evidence_artifact(source: bytes, page_id: str) -> bytes:
    return project_page_evidence(source, page_id).canonical_bytes()


def _parse_record(value: object, index: int) -> PageEvidenceRecord:
    if not isinstance(value, Mapping):
        raise PageEvidenceProjectionError(f"records[{index}] is invalid")
    expected = {
        "record_id",
        "page_id",
        "page_uid",
        "content_sha256",
        "section_ordinal",
        "piece_ordinal",
        "section_context",
        "search_keys",
        "byte_range",
        "byte_coordinate_space",
        "span_sha256",
    }
    if set(value) != expected:
        raise PageEvidenceProjectionError(f"records[{index}] fields are invalid")
    byte_range = value["byte_range"]
    section_context = value["section_context"]
    search_keys = value["search_keys"]
    if (
        not isinstance(byte_range, list)
        or len(byte_range) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in byte_range)
        or byte_range[0] < 0
        or byte_range[1] <= byte_range[0]
        or not isinstance(section_context, list)
        or any(not isinstance(item, str) for item in section_context)
        or not isinstance(search_keys, list)
        or any(not isinstance(item, str) or not item for item in search_keys)
        or len(set(search_keys)) != len(search_keys)
        or not isinstance(value["section_ordinal"], int)
        or isinstance(value["section_ordinal"], bool)
        or value["section_ordinal"] < 0
        or not isinstance(value["piece_ordinal"], int)
        or isinstance(value["piece_ordinal"], bool)
        or value["piece_ordinal"] < 0
        or value["byte_coordinate_space"] != "canonical_source"
        or not isinstance(value["record_id"], str)
        or not value["record_id"].startswith("page-record:")
        or not isinstance(value["page_id"], str)
        or not value["page_id"]
        or not isinstance(value["page_uid"], str)
        or _optional_uid(value["page_uid"]) != value["page_uid"]
        or not isinstance(value["content_sha256"], str)
        or _SHA256_RE.fullmatch(value["content_sha256"]) is None
        or not isinstance(value["span_sha256"], str)
        or _SHA256_RE.fullmatch(value["span_sha256"]) is None
    ):
        raise PageEvidenceProjectionError(f"records[{index}] values are invalid")
    return PageEvidenceRecord(
        record_id=value["record_id"],
        page_id=value["page_id"],
        page_uid=value["page_uid"],
        content_sha256=value["content_sha256"],
        section_ordinal=value["section_ordinal"],
        piece_ordinal=value["piece_ordinal"],
        section_context=tuple(section_context),
        search_keys=tuple(search_keys),
        byte_start=byte_range[0],
        byte_end=byte_range[1],
        span_sha256=value["span_sha256"],
    )


def _source_body_range(source: bytes) -> tuple[int, int, str]:
    try:
        document = parse_document(source)
        document.body.decode("utf-8")
    except (CanonicalDocumentError, UnicodeDecodeError) as exc:
        raise PageEvidenceProjectionError("page source is not canonical UTF-8") from exc
    return len(source) - len(document.body), len(source), document.metadata.get("status", "")


def load_page_evidence_artifact(
    payload: bytes, *, source: bytes | None = None, page_id: str | None = None
) -> PageEvidenceProjection:
    """Strictly parse and optionally verify one page-source artifact."""

    if not isinstance(payload, bytes):
        raise PageEvidenceProjectionError("page projection bytes are invalid")
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PageEvidenceProjectionError("page projection is invalid JSON") from exc
    if not isinstance(value, Mapping):
        raise PageEvidenceProjectionError("page projection root is invalid")
    expected = {
        "schema",
        "projection_id",
        "page_id",
        "page_uid",
        "content_sha256",
        "source_body_byte_range",
        "source_updated",
        "recorded_at",
        "valid_from",
        "valid_to",
        "transform_version",
        "records",
    }
    if set(value) != expected or value["schema"] != PAGE_EVIDENCE_PROJECTION_SCHEMA:
        raise PageEvidenceProjectionError("page projection fields are invalid")
    body_range = value["source_body_byte_range"]
    if (
        not isinstance(body_range, list)
        or len(body_range) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in body_range)
        or body_range[0] < 0
        or body_range[1] < body_range[0]
    ):
        raise PageEvidenceProjectionError("page body range is invalid")
    if (
        not isinstance(value["page_id"], str)
        or not value["page_id"]
        or (page_id is not None and value["page_id"] != page_id)
        or not isinstance(value["page_uid"], str)
        or _optional_uid(value["page_uid"]) != value["page_uid"]
        or not isinstance(value["content_sha256"], str)
        or _SHA256_RE.fullmatch(value["content_sha256"]) is None
        or value["recorded_at"] is not None
        or value["valid_from"] is not None
        or value["valid_to"] is not None
        or not isinstance(value["transform_version"], str)
        or value["transform_version"] != PAGE_EVIDENCE_TRANSFORM_VERSION
        or (value["source_updated"] is not None and not isinstance(value["source_updated"], str))
        or not isinstance(value["records"], list)
        or not isinstance(value["projection_id"], str)
        or not value["projection_id"].startswith("page-projection:")
    ):
        raise PageEvidenceProjectionError("page projection values are invalid")
    records = tuple(_parse_record(item, index) for index, item in enumerate(value["records"]))
    unsigned = dict(value)
    unsigned.pop("projection_id")
    if value["projection_id"] != _projection_id(unsigned):
        raise PageEvidenceProjectionError("page projection identity mismatch")
    projection = PageEvidenceProjection(
        projection_id=value["projection_id"],
        page_id=value["page_id"],
        page_uid=value["page_uid"],
        content_sha256=value["content_sha256"],
        source_body_byte_range=(body_range[0], body_range[1]),
        source_updated=value["source_updated"],
        recorded_at=None,
        valid_from=None,
        valid_to=None,
        transform_version=value["transform_version"],
        records=records,
    )
    if projection.canonical_bytes() != payload:
        raise PageEvidenceProjectionError("page projection is not canonical")

    previous_end = projection.source_body_byte_range[0]
    expected_section = 0
    expected_piece = 0
    seen_ids: set[str] = set()
    for index, record in enumerate(records):
        if (
            record.record_id in seen_ids
            or record.page_id != projection.page_id
            or record.page_uid != projection.page_uid
            or record.content_sha256 != projection.content_sha256
            or record.byte_start != previous_end
            or record.byte_end <= record.byte_start
            or record.section_ordinal != expected_section
            or record.piece_ordinal != expected_piece
        ):
            raise PageEvidenceProjectionError(f"records[{index}] ordering or identity is invalid")
        record_payload = record.to_dict()
        record_payload.pop("record_id")
        if record.record_id != _record_id(record_payload):
            raise PageEvidenceProjectionError(f"records[{index}] identity mismatch")
        seen_ids.add(record.record_id)
        previous_end = record.byte_end
        expected_piece += 1
        if index + 1 < len(records) and records[index + 1].section_ordinal != expected_section:
            expected_section += 1
            expected_piece = 0
    if previous_end != projection.source_body_byte_range[1]:
        raise PageEvidenceProjectionError("page body coverage is incomplete")

    if source is not None:
        body_start, body_end, status = _source_body_range(source)
        if status != "stable":
            raise PageEvidenceProjectionError("page source is not stable")
        source_uid = _optional_uid(parse_document(source).metadata.get("uid"))
        if (
            hashlib.sha256(source).hexdigest() != projection.content_sha256
            or (source_uid is None or source_uid != projection.page_uid)
            or projection.source_body_byte_range != (body_start, body_end)
        ):
            raise PageEvidenceProjectionError("page source identity is stale")
        # Search keys and section context are derived metadata, so source
        # validation must re-run the same deterministic transform instead of
        # trusting a payload whose identities were recomputed after tampering.
        if project_page_evidence(source, projection.page_id).canonical_bytes() != payload:
            raise PageEvidenceProjectionError("page projection transform mismatch")
        for index, record in enumerate(records):
            span = source[record.byte_start : record.byte_end]
            if hashlib.sha256(span).hexdigest() != record.span_sha256:
                raise PageEvidenceProjectionError(f"records[{index}] span checksum mismatch")
        if b"".join(
            source[record.byte_start : record.byte_end] for record in records
        ) != source[body_start:body_end]:
            raise PageEvidenceProjectionError("page body reconstruction mismatch")
    return projection


def reconstruct_page_body(source: bytes, projection: PageEvidenceProjection) -> bytes:
    """Return the exact body covered by a validated projection."""

    validated = load_page_evidence_artifact(
        projection.canonical_bytes(), source=source, page_id=projection.page_id
    )
    return b"".join(source[row.byte_start : row.byte_end] for row in validated.records)


def page_evidence_projection_key(source: bytes, page_id: str) -> str:
    payload = {
        "schema": PAGE_EVIDENCE_PROJECTION_SCHEMA,
        "page_id": page_id,
        "content_sha256": hashlib.sha256(source).hexdigest(),
        "transform_version": PAGE_EVIDENCE_TRANSFORM_VERSION,
        "max_section_piece_chars": MAX_SECTION_PIECE_CHARS,
    }
    return f"page-key:{hashlib.sha256(canonical_json_line_bytes_strict(payload)).hexdigest()}"


def store_page_evidence_projection(
    store: ResearchStore, source: bytes, page_id: str, *, durable: bool = False
) -> EvidenceArtifact:
    projection = project_page_evidence(source, page_id)
    artifact = store.put_artifact(
        projection.canonical_bytes(),
        source_type="page-evidence-projection",
        source_uri=f"page:{page_id}",
        title=page_id,
        mime_type="application/json",
        trust="page-source",
        durable=durable,
        metadata={
            "schema": PAGE_EVIDENCE_PROJECTION_SCHEMA,
            "projection_id": projection.projection_id,
            "projection_key": page_evidence_projection_key(source, page_id),
            "page_id": page_id,
            "content_sha256": projection.content_sha256,
            "transform_version": projection.transform_version,
        },
    )
    stored = store.read_artifact(artifact.artifact_id)
    if stored != projection.canonical_bytes():
        raise PageEvidenceProjectionError("stored page projection differs from source transform")
    load_page_evidence_artifact(stored, source=source, page_id=page_id)
    return artifact


def checkpoint_page_evidence_projection(
    store: ResearchStore,
    session_id: str,
    artifact: EvidenceArtifact,
    *,
    active: bool = False,
    durable_receipt: bool = False,
) -> Path:
    # ``ResearchStore.checkpoint`` intentionally targets the global runtime
    # checkpoint directory.  Page backfills are resumable run artifacts and
    # must stay inside the caller-owned store root instead.
    session_id_sha256 = hashlib.sha256(session_id.encode("utf-8")).hexdigest()
    return store.write_summary(
        session_id_sha256,
        {
            "schema_version": 1,
            "session_id_sha256": session_id_sha256,
            "active": bool(active),
            "durable_receipt": bool(durable_receipt),
            "payload": {
                "kind": "page-evidence-projection",
                "artifact_id": artifact.artifact_id,
                "projection_id": artifact.metadata.get("projection_id", ""),
                "projection_key": artifact.metadata.get("projection_key", ""),
                "page_id": artifact.metadata.get("page_id", ""),
                "content_sha256": artifact.metadata.get("content_sha256", ""),
                "transform_version": artifact.metadata.get("transform_version", ""),
            },
        },
    )
