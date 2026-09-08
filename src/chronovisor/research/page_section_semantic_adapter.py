"""Isolated semantic-index adapter for page-section projections.

The adapter is deliberately separate from the production semantic service.  It
turns the source-backed page projection into the existing ``SemanticDocument``
shape and resolves selected records back to ``SourcePassage`` objects.  No
claims, conversation evidence, authority, or validity interval is invented
for a page source.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from pathlib import Path

from chronovisor.core.canonical_document import parse_document
from chronovisor.core.search_types import SemanticEvidence
from chronovisor.core.semantic_evidence import SourcePassage
from chronovisor.core.semantic_index import SemanticDocument
from chronovisor.core.store import SYSTEM_DIR
from chronovisor.research.page_evidence_projection import (
    PageEvidenceProjection,
    PageEvidenceProjectionError,
    PageEvidenceRecord,
    load_page_evidence_artifact,
    project_page_evidence,
)

SECTION_SEMANTIC_KIND = "section-v1"
MAX_CONTEXT_FIELD_CHARS = 512
_SOURCE_MARKER = "[source]\n"
_CONTEXT_MARKER = "[section_context]\n"
_KEYS_MARKER = "[search_keys]\n"


class PageSectionSemanticError(ValueError):
    """A page-section candidate or selected identity is invalid."""


def _validated_projection(
    source: bytes,
    page_id: str,
    projection: PageEvidenceProjection | None,
) -> PageEvidenceProjection:
    try:
        if projection is None:
            return project_page_evidence(source, page_id)
        # The source-aware loader re-runs the deterministic transform and
        # rejects mutated keys even when a caller recomputes record IDs.
        return load_page_evidence_artifact(
            projection.canonical_bytes(), source=source, page_id=page_id
        )
    except (AttributeError, PageEvidenceProjectionError, TypeError, ValueError) as exc:
        raise PageSectionSemanticError("page projection is stale or invalid") from exc


def _document_text(
    source: bytes, record: PageEvidenceRecord
) -> str:
    try:
        original = source[record.byte_start : record.byte_end].decode("utf-8")
    except (UnicodeDecodeError, TypeError, ValueError) as exc:
        raise PageSectionSemanticError("section source span is invalid") from exc
    if not original:
        raise PageSectionSemanticError("section source span is empty")
    context = "\n".join(record.section_context)[:MAX_CONTEXT_FIELD_CHARS]
    keys = "\n".join(record.search_keys)[:MAX_CONTEXT_FIELD_CHARS]
    return f"{_CONTEXT_MARKER}{context}\n{_KEYS_MARKER}{keys}\n{_SOURCE_MARKER}{original}"


def _source_classification(metadata: dict[str, object], source_path: str) -> tuple[str, str]:
    if not source_path.strip():
        raise PageSectionSemanticError("semantic source path is invalid")
    is_system = metadata.get("is_system") is True or Path(source_path).is_relative_to(
        SYSTEM_DIR
    )
    return (
        "system" if is_system else "page",
        "normal" if not is_system and metadata.get("sensitivity") == "normal" else "high",
    )


def _validate_document_metadata(*, source_mtime_ns: int) -> None:
    if isinstance(source_mtime_ns, bool) or not isinstance(source_mtime_ns, int):
        raise PageSectionSemanticError("semantic source mtime is invalid")
    if source_mtime_ns < 0:
        raise PageSectionSemanticError("semantic source mtime is invalid")


def build_page_section_documents(
    source: bytes,
    page_id: str,
    *,
    source_path: str | Path,
    source_mtime_ns: int,
    projection: PageEvidenceProjection | None = None,
) -> tuple[SemanticDocument, ...]:
    """Build every section-v1 record; no legacy eight-record tail limit."""

    validated = _validated_projection(source, page_id, projection)
    digest = validated.content_sha256
    page_uid = validated.page_uid
    path_text = str(source_path)
    _validate_document_metadata(source_mtime_ns=source_mtime_ns)
    metadata = parse_document(source).metadata
    source_data_class, source_sensitivity = _source_classification(metadata, path_text)
    documents: list[SemanticDocument] = []
    for ordinal, record in enumerate(validated.records):
        if (
            record.page_id != page_id
            or record.page_uid != page_uid
            or record.content_sha256 != digest
            or not record.record_id
            or record.byte_end <= record.byte_start
        ):
            raise PageSectionSemanticError("page section record identity is invalid")
        documents.append(
            SemanticDocument(
                doc_id=record.record_id,
                page_id=page_id,
                kind=SECTION_SEMANTIC_KIND,
                ordinal=ordinal,
                text=_document_text(source, record),
                source_path=path_text,
                source_sha256=digest,
                source_mtime_ns=source_mtime_ns,
                page_uid=page_uid,
                source_data_class=source_data_class,
                source_sensitivity=source_sensitivity,
            )
        )
    if not documents:
        raise PageSectionSemanticError("page projection has no source sections")
    return tuple(documents)


def resolve_page_section_evidence(
    source: bytes,
    page_id: str,
    evidence: Sequence[SemanticEvidence],
    *,
    projection: PageEvidenceProjection | None = None,
) -> tuple[SourcePassage, ...]:
    """Resolve up to three section-v1 identities to exact source passages."""

    try:
        validated = _validated_projection(source, page_id, projection)
    except PageSectionSemanticError:
        return ()
    digest = validated.content_sha256
    page_uid = validated.page_uid
    rows = {record.record_id: record for record in validated.records}
    if len(rows) != len(validated.records):
        return ()
    try:
        selected = tuple(evidence)
    except TypeError:
        return ()
    if not selected or len(selected) > 3:
        return ()
    passages: list[SourcePassage] = []
    seen: set[str] = set()
    generation_id: str | None = None
    for item in selected:
        if not isinstance(item, SemanticEvidence):
            return ()
        if (
            item.page_id != page_id
            or item.kind != SECTION_SEMANTIC_KIND
            or not isinstance(item.doc_id, str)
            or item.doc_id in seen
            or not isinstance(item.page_uid, str)
            or item.page_uid != page_uid
            or item.source_sha256 != digest
            or not isinstance(item.generation_id, str)
            or not item.generation_id.strip()
            or (generation_id is not None and item.generation_id != generation_id)
            or isinstance(item.ordinal, bool)
            or not isinstance(item.ordinal, int)
            or not 0 <= item.ordinal < len(validated.records)
            or isinstance(item.score, bool)
            or type(item.score) not in {int, float}
        ):
            return ()
        try:
            numeric_score = float(item.score)
        except (OverflowError, ValueError):
            return ()
        if not math.isfinite(numeric_score):
            return ()
        generation_id = item.generation_id
        record = rows.get(item.doc_id)
        if record is None or record != validated.records[item.ordinal]:
            return ()
        span = source[record.byte_start : record.byte_end]
        if hashlib.sha256(span).hexdigest() != record.span_sha256:
            return ()
        try:
            text = span.decode("utf-8")
        except UnicodeDecodeError:
            return ()
        passages.append(
            SourcePassage(
                evidence=item,
                byte_start=record.byte_start,
                byte_end=record.byte_end,
                text=text,
            )
        )
        seen.add(item.doc_id)
    return tuple(passages)
