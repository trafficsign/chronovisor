"""Resolve semantic document identities to byte-stable source passages."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Any

from chronovisor.core.canonical_document import CanonicalDocumentError, parse_document
from chronovisor.core.page_identity import normalize_page_uid
from chronovisor.core.search_types import SemanticEvidence


@dataclass(frozen=True, slots=True)
class SourcePassage:
    """An exact UTF-8 source slice selected by a semantic document identity."""

    evidence: SemanticEvidence
    byte_start: int
    byte_end: int
    text: str


@dataclass(frozen=True, slots=True)
class _MappedGroup:
    text: str
    source_positions: tuple[int, ...]


def _title(metadata: dict[str, Any], page_id: str) -> str:
    value = metadata.get("title")
    return value.strip() or page_id if isinstance(value, str) else page_id


def _line_positions(raw: str, offset: int) -> tuple[str, tuple[int, ...]]:
    stripped = raw.strip()
    if not stripped:
        return "", ()
    left = len(raw) - len(raw.lstrip())
    right = len(raw.rstrip())
    return stripped, tuple(offset + index for index in range(left, right))


def _normalize_group(
    lines: list[tuple[str, tuple[int, ...]]], max_chars: int
) -> _MappedGroup:
    """Match ``re.sub(r"\\s+", " ", ...).strip()`` with source positions."""

    normalized: list[str] = []
    normalized_positions: list[int] = []
    pending_space: int | None = None
    for index, (line, positions) in enumerate(lines):
        chars: list[str] = list(line)
        positions_for_line = list(positions)
        if index:
            # The separator is synthetic in ``_markdown_chunks``.  Its source
            # position is only used for internal whitespace; passage edges are
            # always anchored to non-whitespace characters.
            separator = positions[0] - 1 if positions else normalized_positions[-1] + 1
            chars.insert(0, "\n")
            positions_for_line.insert(0, separator)
        for char, position in zip(chars, positions_for_line, strict=True):
            if char.isspace():
                if pending_space is None:
                    pending_space = position
                continue
            if pending_space is not None and normalized:
                normalized.append(" ")
                normalized_positions.append(pending_space)
            pending_space = None
            normalized.append(char)
            normalized_positions.append(position)
            if len(normalized) >= max_chars:
                return _MappedGroup("".join(normalized), tuple(normalized_positions))
    return _MappedGroup("".join(normalized), tuple(normalized_positions))


def _mapped_groups(body: str) -> list[_MappedGroup]:
    """Build paragraph groups while retaining each normalized character's offset."""

    # Import lazily because the search module imports the shared evidence type.
    from chronovisor.core.search import MAX_CHUNK_CHARS, MAX_CHUNKS_PER_PAGE

    max_chunk_chars = MAX_CHUNK_CHARS
    max_groups = MAX_CHUNKS_PER_PAGE
    max_normalized_chars = max_chunk_chars * max_groups
    groups: list[_MappedGroup] = []
    buffer: list[tuple[str, tuple[int, ...]]] = []
    buffer_length = 0
    offset = 0

    def flush() -> None:
        nonlocal buffer, buffer_length
        if buffer:
            group = _normalize_group(buffer, max_normalized_chars)
            if group.text:
                groups.append(group)
        buffer = []
        buffer_length = 0

    for raw_line in body.splitlines(keepends=True):
        line, positions = _line_positions(raw_line, offset)
        offset += len(raw_line)
        heading_match = re.match(r"^(#{1,6})\s+(.+)$", line)
        if heading_match:
            flush()
            if len(groups) >= max_groups:
                break
            continue
        if not line:
            flush()
            if len(groups) >= max_groups:
                break
            continue
        buffer.append((line, positions))
        buffer_length += len(line)
        if buffer_length >= max_chunk_chars:
            flush()
            if len(groups) >= max_groups:
                break
    if len(groups) < max_groups:
        flush()
    return groups


def _payload(chunk: str) -> str | None:
    prefix, separator, payload = chunk.rpartition("\n\n")
    if not separator or not prefix or not payload:
        return None
    return payload


def _chunk_spans(
    body: str, metadata: dict[str, Any], page_id: str
) -> list[tuple[int, int, str]]:
    """Map the existing semantic chunk payloads to body-relative char spans."""

    try:
        from chronovisor.core import search as search_core

        canonical = search_core._markdown_chunks(body, _title(metadata, page_id), metadata)
    except (AttributeError, TypeError, ValueError):
        return []
    groups = _mapped_groups(body)
    if not canonical or not groups:
        # A title-only fallback is a retrieval key, not a source passage.
        return []

    spans: list[tuple[int, int, str]] = []
    group_index = 0
    group_offset = 0
    for chunk in canonical:
        payload = _payload(chunk)
        if payload is None:
            return []
        while group_index < len(groups):
            group = groups[group_index]
            if group_offset == len(group.text):
                group_index += 1
                group_offset = 0
                continue
            if not group.text.startswith(payload, group_offset):
                # Do not fuzzy-match repeated text: a changed extractor must
                # fail closed instead of attributing a passage to the wrong
                # occurrence.
                return []
            end_offset = group_offset + len(payload)
            if end_offset > len(group.source_positions):
                return []
            start_char = group.source_positions[group_offset]
            end_char = group.source_positions[end_offset - 1] + 1
            if start_char < 0 or end_char <= start_char:
                return []
            spans.append((start_char, end_char, payload))
            group_offset = end_offset
            while group_offset < len(group.text) and group.text[group_offset].isspace():
                group_offset += 1
            break
        else:
            return []
    return spans


def _optional_uid(value: object) -> str | None:
    if value is None or value == "":
        return ""
    try:
        return normalize_page_uid(value)
    except (TypeError, ValueError):
        return None


def resolve_semantic_evidence(
    source: bytes,
    page_id: str,
    evidence: tuple[SemanticEvidence, ...],
) -> tuple[SourcePassage, ...]:
    """Resolve valid chunk identities to exact slices of one canonical source.

    Resolution is deliberately fail-closed.  The source digest, stable status,
    page UID, generation, document identity, and finite score are all checked
    before any text is returned.  Page and question documents remain retrieval
    entrances; only chunk documents have a source span.
    """

    if not isinstance(source, bytes) or not isinstance(page_id, str) or not page_id:
        return ()
    try:
        document = parse_document(source)
        body = document.body.decode("utf-8")
    except (CanonicalDocumentError, UnicodeDecodeError):
        return ()
    if document.metadata.get("status") != "stable":
        return ()
    source_digest = hashlib.sha256(source).hexdigest()
    source_uid = _optional_uid(document.metadata.get("uid"))
    if source_uid is None:
        return ()

    valid: list[SemanticEvidence] = []
    seen_doc_ids: set[str] = set()
    identity = source_uid or page_id
    max_chunks = 8
    for item in evidence:
        if not isinstance(item, SemanticEvidence) or item.doc_id in seen_doc_ids:
            continue
        if item.page_id != page_id or item.kind != "chunk":
            continue
        item_uid = _optional_uid(item.page_uid)
        if item.source_sha256 != source_digest or item_uid is None or item_uid != source_uid:
            continue
        if not isinstance(item.doc_id, str) or item.doc_id != f"{identity}#c{item.ordinal}":
            continue
        if isinstance(item.ordinal, bool) or not isinstance(item.ordinal, int):
            continue
        if not 0 <= item.ordinal < max_chunks:
            continue
        if not isinstance(item.generation_id, str) or not item.generation_id.strip():
            continue
        if isinstance(item.score, bool):
            continue
        try:
            score = float(item.score)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(score):
            continue
        valid.append(item)
        seen_doc_ids.add(item.doc_id)
    if not valid:
        return ()

    spans = _chunk_spans(body, document.metadata, page_id)
    if not spans:
        return ()
    body_start = len(source) - len(document.body)
    if source[body_start:] != document.body:
        return ()
    body_char_to_byte = [0]
    for char in body:
        body_char_to_byte.append(body_char_to_byte[-1] + len(char.encode("utf-8")))

    passages: list[SourcePassage] = []
    for item in valid:
        if item.ordinal >= len(spans):
            continue
        start_char, end_char, _payload_text = spans[item.ordinal]
        if end_char > len(body_char_to_byte) - 1:
            continue
        byte_start = body_start + body_char_to_byte[start_char]
        byte_end = body_start + body_char_to_byte[end_char]
        text = source[byte_start:byte_end].decode("utf-8")
        if not text or byte_end <= byte_start:
            continue
        passages.append(
            SourcePassage(
                evidence=item,
                byte_start=byte_start,
                byte_end=byte_end,
                text=text,
            )
        )
    return tuple(passages)
