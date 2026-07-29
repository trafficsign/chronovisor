"""Deterministic, lossless semantic projections for transcript raw captures.

The session savers intentionally preserve tool protocol, model reasoning events,
and conversational messages in one immutable raw.  Ingest does not need to send
that transport trace to a semantic model.  This module verifies the saver receipt,
selects user/assistant text without rewriting a byte, and durably delegates the
selection to content-addressed child artifacts.

No artifact is ever overwritten.  A retry either reads back the exact bytes it
would have written or fails closed with :class:`ProjectionConflictError`.
"""

from __future__ import annotations

from chronovisor.core.hashutil import sha256_bytes as _sha256

import hashlib
import json
import os
import re
import uuid
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from chronovisor.core.canonical_json import (
    canonical_json_line_bytes_strict as _canonical_bytes,
)
from chronovisor.core.durable_state import fsync_directory as _fsync_directory
from chronovisor.raw.raw_segment import RawSegmentCommit
from chronovisor.core.sealed_artifact_decoder import schema_matches

from chronovisor.raw.save_transaction import (
    SaveTransactionReceipt,
    parse_save_transaction_receipt,
)


PROJECTION_POLICY_VERSION = 2
PROJECTION_MANIFEST_SCHEMA = "chronovisor.raw-semantic-projection-manifest.v1"
PROJECTION_CHILD_SCHEMA = "chronovisor.raw-semantic-projection-child.v1"
PROJECTION_NOOP_SCHEMA = "chronovisor.raw-semantic-projection-noop.v1"
PROJECTION_BUNDLE_RECEIPT_SCHEMA = "chronovisor.raw-semantic-projection-bundle-receipt.v1"

_INDEX_WIDTH = 8
_MAX_INDEX = (10**_INDEX_WIDTH) - 1
_TRANSCRIPT_CLAIM_RE = re.compile(
    r"^# (?:Codex|Claude Code) Session Transcript Delta\s*$", re.MULTILINE
)
_SAVE_TRANSACTION_MARKER_PREFIX = "<!-- chronovisor-save-transaction:"
_SAVE_TRANSACTION_FILENAME_RE = re.compile(
    r"^save-(?:codex|claude-code)-[0-9a-f]{24}-from[0-9]+-to[0-9]+\.md$"
)
_TRANSCRIPT_BLOCK_RE = re.compile(
    r"^# (?P<host>Codex|Claude Code) Session Transcript Delta\s*$"
    r".*?^## Transcript Delta\s*$\s*^```json\s*$\n"
    r"(?P<payload>.*?)\n^```\s*$",
    re.MULTILINE | re.DOTALL,
)
_CHILD_FILENAME_RE = re.compile(
    r"^semantic-(?P<projection>[0-9a-f]{64})-child-"
    r"(?P<index>[0-9]{8})-(?P<child>[0-9a-f]{64})\.md$"
)


class RawSemanticProjectionError(ValueError):
    """The source or a deterministic projection artifact is invalid."""


class ProjectionConflictError(RawSemanticProjectionError):
    """A content-addressed path exists with different bytes."""


class ProjectionCapacityError(RawSemanticProjectionError):
    """Even one UTF-8 code point cannot fit the configured child envelope."""


@dataclass(frozen=True)
class ProjectionChildArtifact:
    """One verified child artifact returned to the ingest orchestrator."""

    path: Path
    index: int
    count: int
    child_id: str
    child_sha256: str
    file_sha256: str
    semantic_bytes: int
    source_record_indices: tuple[int, ...]


@dataclass(frozen=True)
class ProjectionArtifacts:
    """A complete projection decision and its verified durable artifacts."""

    kind: Literal["noop", "children", "passthrough"]
    parent_paths: tuple[Path, ...]
    parent_sha256: str
    projection_sha256: str | None
    manifest_path: Path | None
    projection_paths: tuple[Path, ...]
    child_paths: tuple[Path, ...]
    noop_receipt_path: Path | None
    role_counts: dict[str, int]
    record_count: int
    selected_record_count: int
    child_count: int
    children: tuple[ProjectionChildArtifact, ...]


@dataclass(frozen=True)
class _VerifiedParent:
    path: Path
    raw_bytes: bytes
    raw_sha256: str
    receipt: SaveTransactionReceipt


@dataclass(frozen=True)
class _TranscriptRecord:
    index: int
    role: str
    text: str
    line: int | str | None
    timestamp: str | None
    phase: str | None
    row_sha256: str




def _fixed_index(value: int) -> str:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 1 <= value <= _MAX_INDEX
    ):
        raise ProjectionCapacityError(
            f"projection index must be within 1..{_MAX_INDEX}: {value!r}"
        )
    return f"{value:0{_INDEX_WIDTH}d}"


def _receipt_payload(receipt: SaveTransactionReceipt) -> dict[str, Any]:
    transaction = receipt.transaction
    return {
        "host": transaction.host,
        "session_key": transaction.session_key,
        "after_line": transaction.after_line,
        "until_line": transaction.until_line,
        "idempotency_key": transaction.idempotency_key,
        "payload_sha256": receipt.payload_sha256,
    }


def _read_parent(path: Path, raw_bytes: bytes | None = None) -> tuple[bytes, str]:
    source = path.read_bytes() if raw_bytes is None else raw_bytes
    if not isinstance(source, bytes):
        raise TypeError("raw_bytes must be bytes")
    try:
        text = source.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RawSemanticProjectionError(
            f"parent raw is not valid UTF-8: {path.name}"
        ) from exc
    return source, text


def _verified_parent(path: Path, raw_bytes: bytes | None = None) -> _VerifiedParent:
    source, text = _read_parent(path, raw_bytes)
    receipt = parse_save_transaction_receipt(text)
    if receipt is None:
        raise RawSemanticProjectionError(
            f"transcript raw has no valid save transaction receipt: {path.name}"
        )
    if _SAVE_TRANSACTION_FILENAME_RE.fullmatch(path.name) is not None:
        expected_filename = f"save-{receipt.transaction.idempotency_key}.md"
        if path.name != expected_filename:
            raise RawSemanticProjectionError(
                "save transaction filename does not match its verified receipt"
            )
    return _VerifiedParent(
        path=path,
        raw_bytes=source,
        raw_sha256=_sha256(source),
        receipt=receipt,
    )


def _source_parent_payload(parent: _VerifiedParent) -> dict[str, Any]:
    return {
        "raw_sha256": parent.raw_sha256,
        "raw_bytes": len(parent.raw_bytes),
        "receipt": _receipt_payload(parent.receipt),
    }


def _source_identity(source: Mapping[str, Any]) -> dict[str, Any]:
    parents = source.get("parents")
    if not isinstance(parents, list):
        raise RawSemanticProjectionError("projection source parents are missing")
    identity_parents: list[dict[str, Any]] = []
    for row in parents:
        if not isinstance(row, dict):
            raise RawSemanticProjectionError("projection source parent is malformed")
        raw_sha256 = row.get("raw_sha256")
        receipt = row.get("receipt")
        if not isinstance(raw_sha256, str) or not isinstance(receipt, dict):
            raise RawSemanticProjectionError("projection source identity is malformed")
        identity_parents.append(
            {
                "raw_sha256": raw_sha256,
                "receipt": receipt,
            }
        )
    payload_sha256 = source.get("record_payload_sha256")
    kind = source.get("kind")
    if kind not in {"transcript_delta", "reassembled_transcript_record"}:
        raise RawSemanticProjectionError("projection source kind is invalid")
    if not isinstance(payload_sha256, str):
        raise RawSemanticProjectionError("projection record payload digest is missing")
    record_count = source.get("record_count")
    role_counts = source.get("role_counts")
    if (
        isinstance(record_count, bool)
        or not isinstance(record_count, int)
        or record_count < 0
        or not isinstance(role_counts, dict)
        or any(
            not isinstance(role, str)
            or not role
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count < 1
            for role, count in role_counts.items()
        )
        or sum(role_counts.values()) != record_count
    ):
        raise RawSemanticProjectionError(
            "projection source record audit metadata is malformed"
        )
    return {
        "kind": kind,
        "record_payload_sha256": payload_sha256,
        "parents": identity_parents,
        "record_count": record_count,
        "role_counts": dict(sorted(role_counts.items())),
    }


def _source_sha256(source: Mapping[str, Any]) -> str:
    parents = source.get("parents")
    if (
        source.get("kind") == "transcript_delta"
        and isinstance(parents, list)
        and len(parents) == 1
        and isinstance(parents[0], dict)
        and isinstance(parents[0].get("raw_sha256"), str)
    ):
        return str(parents[0]["raw_sha256"])
    return _sha256(_canonical_bytes(_source_identity(source)))


def _projection_id(
    source: Mapping[str, Any],
    *,
    manifest_schema: str | None = None,
) -> str:
    schema = manifest_schema or PROJECTION_MANIFEST_SCHEMA
    if not schema_matches(schema, PROJECTION_MANIFEST_SCHEMA):
        raise RawSemanticProjectionError("projection manifest schema is unsupported")
    return _sha256(
        _canonical_bytes(
            {
                # The schema spelling is part of the sealed projection ID.
                # Artifacts published before the Chronovisor rename therefore
                # have to retain the previous spelling when their ID is
                # recomputed for read-only verification.
                "schema": schema,
                "projection_policy_version": PROJECTION_POLICY_VERSION,
                "source": _source_identity(source),
            }
        )
    )


def _parse_records(payload_bytes: bytes) -> tuple[_TranscriptRecord, ...]:
    try:
        payload = json.loads(payload_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RawSemanticProjectionError(
            "transcript record payload is not valid UTF-8 JSON"
        ) from exc
    if not isinstance(payload, list):
        raise RawSemanticProjectionError("transcript record payload must be an array")

    records: list[_TranscriptRecord] = []
    for index, row in enumerate(payload):
        if not isinstance(row, dict):
            raise RawSemanticProjectionError(
                f"transcript record {index} must be an object"
            )
        role = row.get("role")
        text = row.get("text")
        line = row.get("line")
        timestamp = row.get("timestamp")
        phase = row.get("phase")
        if not isinstance(role, str) or not role:
            raise RawSemanticProjectionError(
                f"transcript record {index} has no valid role"
            )
        if not isinstance(text, str):
            raise RawSemanticProjectionError(
                f"transcript record {index} has no valid text"
            )
        if line is not None and (
            isinstance(line, bool) or not isinstance(line, (int, str))
        ):
            raise RawSemanticProjectionError(
                f"transcript record {index} has an invalid line"
            )
        if timestamp is not None and not isinstance(timestamp, str):
            raise RawSemanticProjectionError(
                f"transcript record {index} has an invalid timestamp"
            )
        if phase is not None and not isinstance(phase, str):
            raise RawSemanticProjectionError(
                f"transcript record {index} has an invalid phase"
            )
        records.append(
            _TranscriptRecord(
                index=index,
                role=role,
                text=text,
                line=line,
                timestamp=timestamp,
                phase=phase,
                row_sha256=_sha256(_canonical_bytes(row)),
            )
        )
    return tuple(records)


def _extract_transcript_payload(text: str) -> tuple[bytes, str] | None:
    matches = list(_TRANSCRIPT_BLOCK_RE.finditer(text))
    if not matches:
        if _TRANSCRIPT_CLAIM_RE.search(text):
            raise RawSemanticProjectionError(
                "raw claims to be a transcript delta but its record block is malformed"
            )
        return None
    if len(matches) != 1:
        raise RawSemanticProjectionError(
            "transcript raw contains multiple record blocks"
        )
    match = matches[0]
    host = "claude-code" if match.group("host") == "Claude Code" else "codex"
    return match.group("payload").encode("utf-8"), host


def _selected_record_payload(record: _TranscriptRecord) -> dict[str, Any]:
    text_bytes = record.text.encode("utf-8")
    payload: dict[str, Any] = {
        "source_record_index": record.index,
        "source_record_sha256": record.row_sha256,
        "role": record.role,
        "text": record.text,
        "text_bytes": len(text_bytes),
        "text_sha256": _sha256(text_bytes),
    }
    if record.line is not None:
        payload["source_line"] = record.line
    if record.timestamp is not None:
        payload["timestamp"] = record.timestamp
    if record.phase is not None:
        payload["phase"] = record.phase
    return payload


def _selection_row(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key != "text"}


def _segment_payload(
    selected: Mapping[str, Any],
    text: str,
    *,
    segment_index: int,
    segment_count: int,
) -> dict[str, Any]:
    segment_bytes = text.encode("utf-8")
    payload = dict(selected)
    payload["text"] = text
    payload["segment_index"] = _fixed_index(segment_index)
    payload["segment_count"] = _fixed_index(segment_count)
    payload["segment_bytes"] = len(segment_bytes)
    payload["segment_sha256"] = _sha256(segment_bytes)
    return payload


def _child_identity(
    *,
    source_sha256: str,
    child_index: str,
    records_sha256: str,
) -> str:
    return _sha256(
        _canonical_bytes(
            {
                "source_sha256": source_sha256,
                "projection_policy_version": PROJECTION_POLICY_VERSION,
                "child_index": child_index,
                "child_records_sha256": records_sha256,
            }
        )
    )


def _render_child(
    *,
    projection_id: str,
    source_sha256: str,
    records: Sequence[Mapping[str, Any]],
    child_index: str,
    child_count: str,
) -> tuple[bytes, dict[str, Any]]:
    records_list = [dict(row) for row in records]
    records_sha256 = _sha256(_canonical_bytes(records_list))
    child_id = _child_identity(
        source_sha256=source_sha256,
        child_index=child_index,
        records_sha256=records_sha256,
    )
    payload = {
        "schema": PROJECTION_CHILD_SCHEMA,
        "kind": "semantic_projection_child",
        "projection_policy_version": PROJECTION_POLICY_VERSION,
        "projection_id": projection_id,
        "source_sha256": source_sha256,
        "child_index": child_index,
        "child_count": child_count,
        "child_id": child_id,
        "records_sha256": records_sha256,
        "records": records_list,
    }
    return _canonical_bytes(payload), payload


def _fits_single_unit(
    selected: Mapping[str, Any],
    text: str,
    *,
    projection_id: str,
    source_sha256: str,
    max_child_bytes: int,
) -> bool:
    unit = _segment_payload(selected, text, segment_index=1, segment_count=1)
    rendered, _ = _render_child(
        projection_id=projection_id,
        source_sha256=source_sha256,
        records=[unit],
        child_index="9" * _INDEX_WIDTH,
        child_count="9" * _INDEX_WIDTH,
    )
    return len(rendered) <= max_child_bytes


def _split_selected_record(
    selected: Mapping[str, Any],
    *,
    projection_id: str,
    source_sha256: str,
    max_child_bytes: int,
) -> list[dict[str, Any]]:
    text = str(selected["text"])
    if _fits_single_unit(
        selected,
        text,
        projection_id=projection_id,
        source_sha256=source_sha256,
        max_child_bytes=max_child_bytes,
    ):
        return [_segment_payload(selected, text, segment_index=1, segment_count=1)]

    pieces: list[str] = []
    offset = 0
    while offset < len(text):
        low = 1
        high = len(text) - offset
        accepted = 0
        while low <= high:
            middle = (low + high) // 2
            candidate = text[offset : offset + middle]
            placeholder = _segment_payload(
                selected,
                candidate,
                segment_index=_MAX_INDEX,
                segment_count=_MAX_INDEX,
            )
            rendered, _ = _render_child(
                projection_id=projection_id,
                source_sha256=source_sha256,
                records=[placeholder],
                child_index="9" * _INDEX_WIDTH,
                child_count="9" * _INDEX_WIDTH,
            )
            if len(rendered) <= max_child_bytes:
                accepted = middle
                low = middle + 1
            else:
                high = middle - 1
        if accepted < 1:
            raise ProjectionCapacityError(
                "max_child_bytes cannot fit one UTF-8 code point plus the "
                "lossless projection envelope"
            )
        pieces.append(text[offset : offset + accepted])
        offset += accepted
        if len(pieces) > _MAX_INDEX:
            raise ProjectionCapacityError("one record requires too many UTF-8 segments")

    count = len(pieces)
    segments = [
        _segment_payload(
            selected,
            piece,
            segment_index=index,
            segment_count=count,
        )
        for index, piece in enumerate(pieces, start=1)
    ]
    reconstructed = "".join(str(segment["text"]) for segment in segments)
    if reconstructed.encode("utf-8") != text.encode("utf-8"):
        raise RawSemanticProjectionError("UTF-8 segmentation changed source bytes")
    return segments


def _pack_children(
    selected_records: Sequence[Mapping[str, Any]],
    *,
    projection_id: str,
    source_sha256: str,
    max_child_bytes: int,
) -> list[list[dict[str, Any]]]:
    if isinstance(max_child_bytes, bool) or not isinstance(max_child_bytes, int):
        raise TypeError("max_child_bytes must be an integer")
    if max_child_bytes < 1:
        raise ValueError("max_child_bytes must be positive")

    units: list[dict[str, Any]] = []
    for selected in selected_records:
        units.extend(
            _split_selected_record(
                selected,
                projection_id=projection_id,
                source_sha256=source_sha256,
                max_child_bytes=max_child_bytes,
            )
        )

    children: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for unit in units:
        candidate = [*current, unit]
        rendered, _ = _render_child(
            projection_id=projection_id,
            source_sha256=source_sha256,
            records=candidate,
            child_index="9" * _INDEX_WIDTH,
            child_count="9" * _INDEX_WIDTH,
        )
        if len(rendered) <= max_child_bytes:
            current = candidate
            continue
        if not current:
            raise ProjectionCapacityError("one semantic segment exceeds child envelope")
        children.append(current)
        current = [unit]
    if current:
        children.append(current)
    if len(children) > _MAX_INDEX:
        raise ProjectionCapacityError("projection requires too many children")
    return children


def _atomic_create_or_verify(path: Path, payload: bytes) -> bool:
    """Create ``path`` without replacement, then exact-read it back.

    Returns ``True`` when an exact artifact already existed.  A hard link from
    a fully synced same-directory temporary file provides atomic publication
    without the overwrite behavior of ``os.replace``.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        observed = path.read_bytes()
        if observed != payload:
            raise ProjectionConflictError(
                f"projection artifact conflict at {path.name}"
            )
        # A previous publication may have returned an fsync error after the
        # hard link became visible.  Re-sync the directory before accepting an
        # existing exact artifact so a retry can close that crash window.
        _fsync_directory(path.parent)
        return True

    temporary = path.parent / f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            observed = path.read_bytes()
            if observed != payload:
                raise ProjectionConflictError(
                    f"projection artifact conflict at {path.name}"
                )
        _fsync_directory(path.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass

    if path.read_bytes() != payload:
        raise ProjectionConflictError(
            f"projection artifact read-back mismatch at {path.name}"
        )
    return False


def _build_projection(
    *,
    parent_paths: tuple[Path, ...],
    source: dict[str, Any],
    records: Sequence[_TranscriptRecord],
    output_dir: Path,
    max_child_bytes: int,
) -> ProjectionArtifacts:
    source_sha256 = _source_sha256(source)
    if (
        isinstance(max_child_bytes, bool)
        or not isinstance(max_child_bytes, int)
        or max_child_bytes < 1
    ):
        raise ValueError("max_child_bytes must be a positive integer")
    projection_id = _projection_id(source)
    role_counts = dict(sorted(Counter(record.role for record in records).items()))
    if (
        source.get("record_count") != len(records)
        or source.get("role_counts") != role_counts
    ):
        raise RawSemanticProjectionError(
            "projection source audit metadata does not match parsed records"
        )
    selected = [
        _selected_record_payload(record)
        for record in records
        if record.role in {"user", "assistant"} and record.text.strip()
    ]
    projection_sha256 = _sha256(_canonical_bytes(selected))
    selection = [_selection_row(row) for row in selected]
    selected_role_counts = dict(
        sorted(Counter(str(row["role"]) for row in selected).items())
    )
    selected_text_bytes = sum(int(row["text_bytes"]) for row in selected)

    output_dir = output_dir.expanduser().resolve(strict=False)
    manifest_path = output_dir / f"semantic-{projection_id}.manifest.json"
    manifest_preexisted = manifest_path.exists()
    if manifest_preexisted:
        existing_manifest, _existing_bytes = _load_canonical_json(manifest_path)
        existing_limit = existing_manifest.get("max_child_bytes")
        if (
            not schema_matches(
                existing_manifest.get("schema"), PROJECTION_MANIFEST_SCHEMA
            )
            or existing_manifest.get("kind") != "raw_semantic_projection_manifest"
            or existing_manifest.get("projection_policy_version")
            != PROJECTION_POLICY_VERSION
            or existing_manifest.get("projection_id") != projection_id
            or existing_manifest.get("source_sha256") != source_sha256
            or existing_manifest.get("source") != source
            or isinstance(existing_limit, bool)
            or not isinstance(existing_limit, int)
            or existing_limit < 1
        ):
            raise ProjectionConflictError(
                "existing projection intent has invalid identity or byte envelope"
            )
        # The first atomic manifest is the durable fan-out intent. A later
        # configuration change completes that exact child set rather than
        # producing a second bundle for the same immutable source.
        max_child_bytes = existing_limit

    child_artifacts: list[ProjectionChildArtifact] = []
    child_manifest_rows: list[dict[str, Any]] = []
    planned_child_bytes: list[tuple[Path, bytes]] = []
    noop_receipt_path: Path | None = None

    if selected:
        child_records = _pack_children(
            selected,
            projection_id=projection_id,
            source_sha256=source_sha256,
            max_child_bytes=max_child_bytes,
        )
        child_count = len(child_records)
        child_count_label = _fixed_index(child_count)
        for child_index, rows in enumerate(child_records, start=1):
            index_label = _fixed_index(child_index)
            child_bytes, child_payload = _render_child(
                projection_id=projection_id,
                source_sha256=source_sha256,
                records=rows,
                child_index=index_label,
                child_count=child_count_label,
            )
            if len(child_bytes) > max_child_bytes:
                raise ProjectionCapacityError(
                    "final child exceeds max_child_bytes after deterministic packing"
                )
            child_id = str(child_payload["child_id"])
            child_sha256 = str(child_payload["records_sha256"])
            filename = f"semantic-{projection_id}-child-{index_label}-{child_id}.md"
            child_path = output_dir / filename
            file_sha256 = _sha256(child_bytes)
            indices = tuple(
                dict.fromkeys(int(row["source_record_index"]) for row in rows)
            )
            semantic_bytes = sum(int(row["segment_bytes"]) for row in rows)
            artifact = ProjectionChildArtifact(
                path=child_path,
                index=child_index,
                count=child_count,
                child_id=child_id,
                child_sha256=child_sha256,
                file_sha256=file_sha256,
                semantic_bytes=semantic_bytes,
                source_record_indices=indices,
            )
            child_artifacts.append(artifact)
            planned_child_bytes.append((child_path, child_bytes))
            child_manifest_rows.append(
                {
                    "filename": filename,
                    "child_index": index_label,
                    "child_count": child_count_label,
                    "child_id": child_id,
                    "child_sha256": child_sha256,
                    "file_sha256": file_sha256,
                    "file_bytes": len(child_bytes),
                    "semantic_bytes": semantic_bytes,
                    "source_record_indices": list(indices),
                }
            )
        status = "delegated"
    else:
        child_count = 0
        status = "noop"

    manifest: dict[str, Any] = {
        "schema": PROJECTION_MANIFEST_SCHEMA,
        "kind": "raw_semantic_projection_manifest",
        "projection_policy_version": PROJECTION_POLICY_VERSION,
        "max_child_bytes": max_child_bytes,
        "projection_id": projection_id,
        "source_sha256": source_sha256,
        "projection_sha256": projection_sha256,
        "source": source,
        "status": status,
        "record_count": len(records),
        "role_counts": role_counts,
        "selected_record_count": len(selected),
        "selected_role_counts": selected_role_counts,
        "selected_text_bytes": selected_text_bytes,
        "selection": selection,
        "children": child_manifest_rows,
    }
    if not selected:
        manifest["noop_receipt_filename"] = f"semantic-{projection_id}.noop.json"

    manifest_bytes = _canonical_bytes(manifest)
    manifest_sha256 = _sha256(manifest_bytes)
    noop_bytes: bytes | None = None
    bundle_receipt_path: Path | None = None
    bundle_receipt_bytes: bytes | None = None
    if not selected:
        noop_payload = {
            "schema": PROJECTION_NOOP_SCHEMA,
            "kind": "deterministic_semantic_noop_receipt",
            "projection_policy_version": PROJECTION_POLICY_VERSION,
            "projection_id": projection_id,
            "source_sha256": source_sha256,
            "projection_sha256": projection_sha256,
            "manifest_sha256": manifest_sha256,
            "record_count": len(records),
            "role_counts": role_counts,
            "selected_record_count": 0,
        }
        noop_receipt_path = output_dir / str(manifest["noop_receipt_filename"])
        noop_bytes = _canonical_bytes(noop_payload)
    else:
        bundle_receipt_path = output_dir / (
            f"semantic-{projection_id}-manifest-{manifest_sha256}.receipt.json"
        )
        bundle_receipt_bytes = _canonical_bytes(
            {
                "schema": PROJECTION_BUNDLE_RECEIPT_SCHEMA,
                "kind": "raw_semantic_projection_bundle_receipt",
                "projection_policy_version": PROJECTION_POLICY_VERSION,
                "projection_id": projection_id,
                "source_sha256": source_sha256,
                "projection_sha256": projection_sha256,
                "manifest_sha256": manifest_sha256,
                "child_count": child_count,
            }
        )

    try:
        # Intent is committed before any child/noop publication. A crash can
        # leave a deterministically repairable incomplete bundle, but cannot
        # expose an uncommitted alternate child set to the orchestrator.
        _atomic_create_or_verify(manifest_path, manifest_bytes)
    except ProjectionConflictError:
        if not manifest_preexisted:
            # Resolve a first-writer race by adopting the winner's manifest
            # limit. No losing child has been published because intent is the
            # first write in this function.
            return _build_projection(
                parent_paths=parent_paths,
                source=source,
                records=records,
                output_dir=output_dir,
                max_child_bytes=max_child_bytes,
            )
        raise
    for child_path, child_bytes in planned_child_bytes:
        _atomic_create_or_verify(child_path, child_bytes)
    if noop_receipt_path is not None and noop_bytes is not None:
        _atomic_create_or_verify(noop_receipt_path, noop_bytes)
    if bundle_receipt_path is not None and bundle_receipt_bytes is not None:
        _atomic_create_or_verify(bundle_receipt_path, bundle_receipt_bytes)

    verified = verify_projection_bundle(manifest_path)
    if verified != manifest:
        raise ProjectionConflictError("projection manifest changed during read-back")

    child_paths = tuple(artifact.path for artifact in child_artifacts)
    projection_paths = (
        child_paths
        if child_paths
        else ((noop_receipt_path,) if noop_receipt_path is not None else ())
    )
    return ProjectionArtifacts(
        kind="children" if child_paths else "noop",
        parent_paths=parent_paths,
        parent_sha256=source_sha256,
        projection_sha256=projection_sha256,
        manifest_path=manifest_path,
        projection_paths=projection_paths,
        child_paths=child_paths,
        noop_receipt_path=noop_receipt_path,
        role_counts=role_counts,
        record_count=len(records),
        selected_record_count=len(selected),
        child_count=child_count,
        children=tuple(child_artifacts),
    )


def project_parent_raw(
    raw_path: Path,
    *,
    output_dir: Path,
    max_child_bytes: int,
    raw_bytes: bytes | None = None,
) -> ProjectionArtifacts:
    """Project one saver raw, or return ``passthrough`` for a normal wiki raw.

    A raw that claims a Codex/Claude transcript envelope must have a valid
    self-verifying save transaction receipt.  Non-transcript raws are not
    rewritten and need no projection artifact.
    """

    source_bytes, source_text = _read_parent(raw_path, raw_bytes)
    extracted = _extract_transcript_payload(source_text)
    if extracted is None:
        try:
            child = _projected_child_passthrough(raw_path, source_bytes)
        except ProjectionConflictError:
            raise
        except RawSemanticProjectionError as exc:
            # Once an artifact claims the delegated-child contract, any
            # canonical/filename/manifest/read-back defect belongs to the
            # derived projection bundle, never to the immutable source raw.
            raise ProjectionConflictError(
                f"projection child artifact is invalid: {exc}"
            ) from exc
        if child is not None:
            return child
        if (
            _SAVE_TRANSACTION_MARKER_PREFIX in source_text
            or _SAVE_TRANSACTION_FILENAME_RE.fullmatch(raw_path.name) is not None
        ):
            raise RawSemanticProjectionError(
                "save transaction raw has no canonical transcript envelope"
            )
        return ProjectionArtifacts(
            kind="passthrough",
            parent_paths=(raw_path,),
            parent_sha256=_sha256(source_bytes),
            projection_sha256=None,
            manifest_path=None,
            projection_paths=(),
            child_paths=(),
            noop_receipt_path=None,
            role_counts={},
            record_count=0,
            selected_record_count=0,
            child_count=0,
            children=(),
        )

    parent = _verified_parent(raw_path, source_bytes)
    payload_bytes, expected_host = extracted
    if parent.receipt.transaction.host != expected_host:
        raise RawSemanticProjectionError(
            "transcript heading host does not match save transaction receipt"
        )
    records = _parse_records(payload_bytes)
    role_counts = dict(sorted(Counter(record.role for record in records).items()))
    source = {
        "kind": "transcript_delta",
        "record_payload_sha256": _sha256(payload_bytes),
        "record_payload_bytes": len(payload_bytes),
        "record_count": len(records),
        "role_counts": role_counts,
        "parents": [_source_parent_payload(parent)],
    }
    return _build_projection(
        parent_paths=(raw_path,),
        source=source,
        records=records,
        output_dir=output_dir,
        max_child_bytes=max_child_bytes,
    )


def project_native_transcript(
    raw_path: Path,
    raw_bytes: bytes,
    commit: RawSegmentCommit,
    *,
    output_dir: Path,
    max_child_bytes: int,
) -> ProjectionArtifacts:
    """Project source-native JSONL bytes referenced by one v2 commit."""

    if raw_path.name != commit.raw_id:
        raise RawSemanticProjectionError("native Raw reference ID mismatch")
    if len(raw_bytes) != commit.length or _sha256(raw_bytes) != commit.sha256:
        raise RawSemanticProjectionError("native Raw bytes disagree with commit")
    if not raw_bytes.endswith(b"\n"):
        raise RawSemanticProjectionError("native transcript is not line complete")

    records: list[_TranscriptRecord] = []
    for index, encoded_line in enumerate(raw_bytes.splitlines(), start=0):
        try:
            event = json.loads(encoded_line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RawSemanticProjectionError(
                f"native transcript line {index + 1} is invalid JSON"
            ) from exc
        if not isinstance(event, dict):
            raise RawSemanticProjectionError(
                f"native transcript line {index + 1} is not an object"
            )
        source_line = commit.after_line + index + 1
        timestamp = event.get("timestamp")
        timestamp = timestamp if isinstance(timestamp, str) else None
        phase: str | None = None
        event_type: str | None = None
        if commit.host == "codex":
            from chronovisor.hosts.codex_record import _codex_semantic_view

            item_type = event.get("type")
            payload = event.get("payload")
            role, text = _codex_semantic_view(item_type, payload)
            payload_type = payload.get("type") if isinstance(payload, dict) else None
            event_type = (
                payload_type
                if isinstance(payload_type, str)
                else item_type
                if isinstance(item_type, str)
                else None
            )
            phase_value = payload.get("phase") if isinstance(payload, dict) else None
            phase = phase_value if isinstance(phase_value, str) else None
            semantic_row: dict[str, Any] = {
                "line": source_line,
                "role": role,
                "text": text,
                "timestamp": timestamp,
                "phase": phase,
            }
        elif commit.host == "claude-code":
            from chronovisor.hosts.claude_code_record import _claude_semantic_view

            item_type = event.get("type")
            message = event.get("message")
            content = message.get("content") if isinstance(message, dict) else None
            role, text = _claude_semantic_view(item_type, content)
            event_type = item_type if isinstance(item_type, str) else None
            semantic_row = {
                "line": source_line,
                "role": role,
                "text": text,
                "timestamp": timestamp,
            }
        else:
            raise RawSemanticProjectionError(
                f"unsupported native transcript host: {commit.host}"
            )
        if event_type is not None:
            semantic_row["event_type"] = event_type
        semantic_row["event"] = event
        records.append(
            _TranscriptRecord(
                index=index,
                role=role,
                text=text,
                line=source_line,
                timestamp=timestamp,
                phase=phase,
                row_sha256=_sha256(_canonical_bytes(semantic_row)),
            )
        )
    if len(records) != commit.record_count:
        raise RawSemanticProjectionError(
            "native transcript record count disagrees with commit"
        )
    role_counts = dict(sorted(Counter(record.role for record in records).items()))
    receipt = {
        "host": commit.host,
        "session_key": commit.session_key,
        "after_line": commit.after_line,
        "until_line": commit.until_line,
        "idempotency_key": commit.idempotency_key,
        "payload_sha256": commit.sha256,
    }
    source = {
        "kind": "transcript_delta",
        "record_payload_sha256": commit.sha256,
        "record_payload_bytes": len(raw_bytes),
        "record_count": len(records),
        "role_counts": role_counts,
        "parents": [
            {
                "raw_sha256": commit.sha256,
                "raw_bytes": len(raw_bytes),
                "receipt": receipt,
            }
        ],
    }
    return _build_projection(
        parent_paths=(raw_path,),
        source=source,
        records=records,
        output_dir=output_dir,
        max_child_bytes=max_child_bytes,
    )


def verify_projection_child(raw_path: Path) -> ProjectionChildArtifact:
    """Verify one child against its canonical filename and durable manifest."""

    source_bytes = raw_path.read_bytes()
    projected = _projected_child_passthrough(raw_path, source_bytes)
    if (
        projected is None
        or projected.kind != "passthrough"
        or len(projected.children) != 1
        or projected.children[0].path != raw_path
    ):
        raise RawSemanticProjectionError(
            "path is not a verified deterministic semantic projection child"
        )
    return projected.children[0]


def project_reassembled_raws(
    raw_paths: Sequence[Path],
    record_bytes: bytes,
    *,
    output_dir: Path,
    max_child_bytes: int,
    raw_bytes_by_path: Mapping[Path, bytes] | None = None,
) -> ProjectionArtifacts:
    """Project one record reassembled from verified lossless fragment raws.

    ``record_bytes`` is the exact serialized record array produced by the
    saver.  Every parent fragment receipt is verified before those bytes are
    accepted, and the ordered parent set plus reassembled payload hash binds
    the resulting projection identity.
    """

    if not raw_paths:
        raise RawSemanticProjectionError("reassembled projection has no parent raws")
    from chronovisor.raw.raw_capture_fragments import (
        RawCaptureFragmentError,
        group_capture_fragments,
        parse_capture_fragment,
    )

    parents_by_path: dict[Path, _VerifiedParent] = {}
    fragments = []
    for path in raw_paths:
        supplied = (
            raw_bytes_by_path.get(path) if raw_bytes_by_path is not None else None
        )
        parent = _verified_parent(path, supplied)
        parents_by_path[path] = parent
        try:
            fragment = parse_capture_fragment(
                path,
                text=parent.raw_bytes.decode("utf-8"),
            )
        except RawCaptureFragmentError as exc:
            raise RawSemanticProjectionError(
                f"fragment parent is malformed: {path.name}: {exc}"
            ) from exc
        if fragment is None:
            raise RawSemanticProjectionError(
                f"reassembled parent does not contain fragment metadata: {path.name}"
            )
        transaction = parent.receipt.transaction
        if (
            transaction.until_line != fragment.identity.source_line
            or transaction.after_line != max(0, fragment.identity.source_line - 1)
        ):
            raise RawSemanticProjectionError(
                "fragment save transaction interval does not match metadata source line"
            )
        fragments.append(fragment)
    try:
        groups = group_capture_fragments(fragments)
    except RawCaptureFragmentError as exc:
        raise RawSemanticProjectionError(
            f"fragment parent set is invalid: {exc}"
        ) from exc
    if len(groups) != 1 or not groups[0].complete:
        raise RawSemanticProjectionError(
            "reassembled parent set must contain one complete fragment group"
        )
    try:
        assembled = groups[0].assemble_bytes()
    except RawCaptureFragmentError as exc:
        raise RawSemanticProjectionError(f"fragment reassembly failed: {exc}") from exc
    if assembled != record_bytes:
        raise RawSemanticProjectionError(
            "supplied reassembled record differs from verified fragment bytes"
        )
    parents = [parents_by_path[fragment.path] for fragment in groups[0].fragments]
    hosts = {parent.receipt.transaction.host for parent in parents}
    if len(hosts) != 1:
        raise RawSemanticProjectionError("fragment parent receipts use different hosts")
    fragment_host = groups[0].identity.host
    if hosts != {fragment_host}:
        raise RawSemanticProjectionError(
            "fragment metadata host does not match save transaction receipt host"
        )
    records = _parse_records(record_bytes)
    role_counts = dict(sorted(Counter(record.role for record in records).items()))
    source = {
        "kind": "reassembled_transcript_record",
        "record_payload_sha256": _sha256(record_bytes),
        "record_payload_bytes": len(record_bytes),
        "record_count": len(records),
        "role_counts": role_counts,
        "parents": [_source_parent_payload(parent) for parent in parents],
    }
    return _build_projection(
        parent_paths=tuple(parent.path for parent in parents),
        source=source,
        records=records,
        output_dir=output_dir,
        max_child_bytes=max_child_bytes,
    )


def _projected_child_passthrough(
    raw_path: Path,
    source_bytes: bytes,
) -> ProjectionArtifacts | None:
    """Validate a previously delegated child before returning passthrough."""

    filename_match = _CHILD_FILENAME_RE.fullmatch(raw_path.name)
    try:
        decoded = json.loads(source_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        if filename_match is not None:
            raise RawSemanticProjectionError(
                "projection child filename points to invalid canonical JSON"
            )
        return None
    claims_child = isinstance(decoded, dict) and (
        schema_matches(decoded.get("schema"), PROJECTION_CHILD_SCHEMA)
        or decoded.get("kind") == "semantic_projection_child"
    )
    if not claims_child and filename_match is None:
        return None
    if not isinstance(decoded, dict) or _canonical_bytes(decoded) != source_bytes:
        raise RawSemanticProjectionError("projection child is not canonical JSON")

    projection_id = decoded.get("projection_id")
    source_sha256 = decoded.get("source_sha256")
    index = decoded.get("child_index")
    count = decoded.get("child_count")
    child_id = decoded.get("child_id")
    if not all(
        isinstance(value, str)
        for value in (projection_id, source_sha256, index, count, child_id)
    ):
        raise RawSemanticProjectionError("projection child binding fields are missing")
    if (
        re.fullmatch(r"[0-9a-f]{64}", projection_id) is None
        or re.fullmatch(r"[0-9a-f]{64}", source_sha256) is None
        or re.fullmatch(r"[0-9a-f]{64}", child_id) is None
        or re.fullmatch(r"[0-9]{8}", index) is None
        or re.fullmatch(r"[0-9]{8}", count) is None
        or not 1 <= int(index) <= int(count) <= _MAX_INDEX
    ):
        raise RawSemanticProjectionError("projection child binding fields are invalid")
    expected_filename = f"semantic-{projection_id}-child-{index}-{child_id}.md"
    if raw_path.name != expected_filename:
        raise RawSemanticProjectionError("projection child filename identity mismatch")
    records = _validate_child_payload(
        decoded,
        projection_id=projection_id,
        source_sha256=source_sha256,
        index=index,
        count=count,
    )
    manifest_path = raw_path.parent / f"semantic-{projection_id}.manifest.json"
    if not manifest_path.is_file():
        raise RawSemanticProjectionError(
            "projection child has no durable delegation manifest"
        )
    manifest = verify_projection_bundle(manifest_path)
    manifest_children = manifest.get("children")
    file_sha256 = _sha256(source_bytes)
    membership = (
        [
            row
            for row in manifest_children
            if isinstance(row, dict)
            and row.get("filename") == raw_path.name
            and row.get("child_index") == index
            and row.get("child_id") == child_id
            and row.get("file_sha256") == file_sha256
            and row.get("file_bytes") == len(source_bytes)
        ]
        if isinstance(manifest_children, list)
        else []
    )
    if (
        manifest.get("status") != "delegated"
        or manifest.get("projection_id") != projection_id
        or manifest.get("source_sha256") != source_sha256
        or len(membership) != 1
    ):
        raise RawSemanticProjectionError(
            "projection child is not an exact member of its delegation manifest"
        )
    role_counts: Counter[str] = Counter()
    source_indices: list[int] = []
    semantic_bytes = 0
    for record in records:
        role = record.get("role")
        source_index = record.get("source_record_index")
        text = record.get("text")
        if (
            not isinstance(role, str)
            or isinstance(source_index, bool)
            or not isinstance(source_index, int)
            or not isinstance(text, str)
        ):
            raise RawSemanticProjectionError("projection child record is malformed")
        encoded = text.encode("utf-8")
        if record.get("segment_bytes") != len(encoded) or record.get(
            "segment_sha256"
        ) != _sha256(encoded):
            raise RawSemanticProjectionError("projection child segment digest mismatch")
        role_counts[role] += 1
        source_indices.append(source_index)
        semantic_bytes += len(encoded)
    child_sha256 = str(decoded["records_sha256"])
    artifact = ProjectionChildArtifact(
        path=raw_path,
        index=int(index),
        count=int(count),
        child_id=child_id,
        child_sha256=child_sha256,
        file_sha256=file_sha256,
        semantic_bytes=semantic_bytes,
        source_record_indices=tuple(dict.fromkeys(source_indices)),
    )
    return ProjectionArtifacts(
        kind="passthrough",
        parent_paths=(raw_path,),
        parent_sha256=source_sha256,
        projection_sha256=str(manifest["projection_sha256"]),
        manifest_path=manifest_path,
        projection_paths=(raw_path,),
        child_paths=(raw_path,),
        noop_receipt_path=None,
        role_counts=dict(sorted(role_counts.items())),
        record_count=len(records),
        selected_record_count=len(set(source_indices)),
        child_count=int(count),
        children=(artifact,),
    )


def _load_canonical_json(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise RawSemanticProjectionError(
            f"projection artifact cannot be read: {path.name}"
        ) from exc
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RawSemanticProjectionError(
            f"projection artifact is not valid JSON: {path.name}"
        ) from exc
    if not isinstance(payload, dict) or _canonical_bytes(payload) != raw:
        raise RawSemanticProjectionError(
            f"projection artifact is not canonical: {path.name}"
        )
    return payload, raw


def _validate_child_payload(
    payload: Mapping[str, Any],
    *,
    projection_id: str,
    source_sha256: str,
    index: str,
    count: str,
) -> list[dict[str, Any]]:
    records = payload.get("records")
    if not isinstance(records, list) or not all(
        isinstance(row, dict) for row in records
    ):
        raise RawSemanticProjectionError("projection child records are malformed")
    records_sha256 = _sha256(_canonical_bytes(records))
    child_id = _child_identity(
        source_sha256=source_sha256,
        child_index=index,
        records_sha256=records_sha256,
    )
    if (
        not schema_matches(payload.get("schema"), PROJECTION_CHILD_SCHEMA)
        or payload.get("kind") != "semantic_projection_child"
        or payload.get("projection_policy_version") != PROJECTION_POLICY_VERSION
        or payload.get("projection_id") != projection_id
        or payload.get("source_sha256") != source_sha256
        or payload.get("child_index") != index
        or payload.get("child_count") != count
        or payload.get("records_sha256") != records_sha256
        or payload.get("child_id") != child_id
    ):
        raise RawSemanticProjectionError("projection child identity mismatch")
    return [dict(row) for row in records]


def _reconstruct_selected_records(
    units: Sequence[Mapping[str, Any]],
    selection: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    expected_order: list[tuple[int, int]] = []
    grouped: dict[int, list[Mapping[str, Any]]] = {}
    for unit in units:
        source_index = unit.get("source_record_index")
        segment_index = unit.get("segment_index")
        if (
            isinstance(source_index, bool)
            or not isinstance(source_index, int)
            or not isinstance(segment_index, str)
            or not segment_index.isdigit()
        ):
            raise RawSemanticProjectionError("projection segment identity is malformed")
        numeric_segment_index = int(segment_index)
        expected_order.append((source_index, numeric_segment_index))
        grouped.setdefault(source_index, []).append(unit)
    if expected_order != sorted(expected_order):
        raise RawSemanticProjectionError("projection segments are out of source order")

    reconstructed: list[dict[str, Any]] = []
    for selected in selection:
        source_index = selected.get("source_record_index")
        if isinstance(source_index, bool) or not isinstance(source_index, int):
            raise RawSemanticProjectionError("projection selection index is malformed")
        segments = grouped.pop(source_index, None)
        if not segments:
            raise RawSemanticProjectionError(
                "projection selection has no child segment"
            )
        count_labels = {str(segment.get("segment_count")) for segment in segments}
        if len(count_labels) != 1 or not next(iter(count_labels)).isdigit():
            raise RawSemanticProjectionError("projection segment count mismatch")
        expected_count = int(next(iter(count_labels)))
        if len(segments) != expected_count:
            raise RawSemanticProjectionError("projection segment set is incomplete")
        if [int(str(segment["segment_index"])) for segment in segments] != list(
            range(1, expected_count + 1)
        ):
            raise RawSemanticProjectionError(
                "projection segment indices are incomplete"
            )

        texts: list[str] = []
        for segment in segments:
            text = segment.get("text")
            if not isinstance(text, str):
                raise RawSemanticProjectionError("projection segment text is malformed")
            encoded = text.encode("utf-8")
            if segment.get("segment_bytes") != len(encoded) or segment.get(
                "segment_sha256"
            ) != _sha256(encoded):
                raise RawSemanticProjectionError("projection segment digest mismatch")
            for key, value in selected.items():
                if segment.get(key) != value:
                    raise RawSemanticProjectionError(
                        "projection segment does not match selection manifest"
                    )
            texts.append(text)
        text = "".join(texts)
        encoded = text.encode("utf-8")
        if selected.get("text_bytes") != len(encoded) or selected.get(
            "text_sha256"
        ) != _sha256(encoded):
            raise RawSemanticProjectionError("reconstructed projection text mismatch")
        reconstructed.append({**dict(selected), "text": text})
    if grouped:
        raise RawSemanticProjectionError("projection contains unselected child records")
    return reconstructed


def verify_projection_bundle(manifest_path: Path) -> dict[str, Any]:
    """Exact-read and validate a manifest plus every referenced artifact."""

    manifest, manifest_bytes = _load_canonical_json(manifest_path)
    source = manifest.get("source")
    if not isinstance(source, dict):
        raise RawSemanticProjectionError("projection manifest source is malformed")
    max_child_bytes = manifest.get("max_child_bytes")
    if (
        isinstance(max_child_bytes, bool)
        or not isinstance(max_child_bytes, int)
        or max_child_bytes < 1
    ):
        raise RawSemanticProjectionError(
            "projection manifest child byte envelope is invalid"
        )
    observed_schema = manifest.get("schema")
    if not isinstance(observed_schema, str):
        raise RawSemanticProjectionError("projection manifest schema is malformed")
    projection_id = _projection_id(source, manifest_schema=observed_schema)
    source_sha256 = _source_sha256(source)
    if (
        manifest_path.name != f"semantic-{projection_id}.manifest.json"
        or not schema_matches(manifest.get("schema"), PROJECTION_MANIFEST_SCHEMA)
        or manifest.get("kind") != "raw_semantic_projection_manifest"
        or manifest.get("projection_policy_version") != PROJECTION_POLICY_VERSION
        or manifest.get("projection_id") != projection_id
        or manifest.get("source_sha256") != source_sha256
    ):
        raise RawSemanticProjectionError("projection manifest identity mismatch")
    selection = manifest.get("selection")
    children = manifest.get("children")
    if not isinstance(selection, list) or not all(
        isinstance(row, dict) for row in selection
    ):
        raise RawSemanticProjectionError("projection selection manifest is malformed")
    if not isinstance(children, list) or not all(
        isinstance(row, dict) for row in children
    ):
        raise RawSemanticProjectionError("projection child manifest is malformed")

    if manifest.get("status") == "noop":
        if selection or children or manifest.get("selected_record_count") != 0:
            raise RawSemanticProjectionError(
                "noop projection contains semantic records"
            )
        filename = manifest.get("noop_receipt_filename")
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise RawSemanticProjectionError("noop receipt filename is invalid")
        receipt, _ = _load_canonical_json(manifest_path.parent / filename)
        if (
            not schema_matches(receipt.get("schema"), PROJECTION_NOOP_SCHEMA)
            or receipt.get("kind") != "deterministic_semantic_noop_receipt"
            or receipt.get("projection_policy_version") != PROJECTION_POLICY_VERSION
            or receipt.get("projection_id") != projection_id
            or receipt.get("source_sha256") != source_sha256
            or receipt.get("projection_sha256") != manifest.get("projection_sha256")
            or receipt.get("manifest_sha256") != _sha256(manifest_bytes)
            or receipt.get("record_count") != manifest.get("record_count")
            or receipt.get("role_counts") != manifest.get("role_counts")
            or receipt.get("selected_record_count") != 0
        ):
            raise RawSemanticProjectionError("noop receipt identity mismatch")
        return manifest

    if manifest.get("status") != "delegated" or not children:
        raise RawSemanticProjectionError("projection manifest has invalid status")
    manifest_sha256 = _sha256(manifest_bytes)
    bundle_receipt_path = manifest_path.parent / (
        f"semantic-{projection_id}-manifest-{manifest_sha256}.receipt.json"
    )
    bundle_receipt, _ = _load_canonical_json(bundle_receipt_path)
    if (
        not schema_matches(
            bundle_receipt.get("schema"), PROJECTION_BUNDLE_RECEIPT_SCHEMA
        )
        or bundle_receipt.get("kind") != "raw_semantic_projection_bundle_receipt"
        or bundle_receipt.get("projection_policy_version") != PROJECTION_POLICY_VERSION
        or bundle_receipt.get("projection_id") != projection_id
        or bundle_receipt.get("source_sha256") != source_sha256
        or bundle_receipt.get("projection_sha256") != manifest.get("projection_sha256")
        or bundle_receipt.get("manifest_sha256") != manifest_sha256
        or bundle_receipt.get("child_count") != len(children)
    ):
        raise RawSemanticProjectionError("projection bundle receipt identity mismatch")
    if manifest.get("record_count") != source.get("record_count") or manifest.get(
        "role_counts"
    ) != source.get("role_counts"):
        raise RawSemanticProjectionError("projection record audit metadata mismatch")
    child_count_label = _fixed_index(len(children))
    units: list[dict[str, Any]] = []
    for child_index, row in enumerate(children, start=1):
        filename = row.get("filename")
        index_label = _fixed_index(child_index)
        if (
            not isinstance(filename, str)
            or Path(filename).name != filename
            or row.get("child_index") != index_label
            or row.get("child_count") != child_count_label
        ):
            raise RawSemanticProjectionError("projection child order is malformed")
        child_path = manifest_path.parent / filename
        child_payload, child_bytes = _load_canonical_json(child_path)
        if (
            row.get("file_sha256") != _sha256(child_bytes)
            or row.get("file_bytes") != len(child_bytes)
            or len(child_bytes) > max_child_bytes
            or row.get("child_id") != child_payload.get("child_id")
            or row.get("child_sha256") != child_payload.get("records_sha256")
        ):
            raise RawSemanticProjectionError("projection child read-back mismatch")
        child_units = _validate_child_payload(
            child_payload,
            projection_id=projection_id,
            source_sha256=source_sha256,
            index=index_label,
            count=child_count_label,
        )
        expected_indices = list(
            dict.fromkeys(int(unit["source_record_index"]) for unit in child_units)
        )
        expected_semantic_bytes = sum(
            int(unit["segment_bytes"]) for unit in child_units
        )
        if (
            row.get("source_record_indices") != expected_indices
            or row.get("semantic_bytes") != expected_semantic_bytes
        ):
            raise RawSemanticProjectionError("projection child audit metadata mismatch")
        units.extend(child_units)

    reconstructed = _reconstruct_selected_records(units, selection)
    projection_sha256 = _sha256(_canonical_bytes(reconstructed))
    if (
        projection_sha256 != manifest.get("projection_sha256")
        or len(reconstructed) != manifest.get("selected_record_count")
        or sum(int(row["text_bytes"]) for row in reconstructed)
        != manifest.get("selected_text_bytes")
        or dict(sorted(Counter(str(row["role"]) for row in reconstructed).items()))
        != manifest.get("selected_role_counts")
    ):
        raise RawSemanticProjectionError("projection semantic digest mismatch")
    return manifest


def projection_bundle_state_for_parent(
    raw_path: Path,
    *,
    projection_dir: Path | None = None,
) -> Literal["absent", "incomplete", "completed", "invalid"]:
    """Inspect current-policy projection publication for one immutable parent.

    This is intentionally read-only.  It lets failure supervision distinguish
    an intent-first crash (safe to resume) from a fully published bundle and
    from canonical tampering/conflict (must remain deferred).
    """

    try:
        parent_sha256 = _sha256(raw_path.read_bytes())
    except OSError:
        return "absent"

    artifact_dir = projection_dir or raw_path.parent
    observed: list[Literal["incomplete", "completed", "invalid"]] = []
    for manifest_path in sorted(artifact_dir.glob("semantic-*.manifest.json")):
        try:
            manifest, manifest_bytes = _load_canonical_json(manifest_path)
        except RawSemanticProjectionError:
            continue
        source = manifest.get("source")
        if not isinstance(source, dict):
            continue
        parents = source.get("parents")
        if not isinstance(parents, list) or not any(
            isinstance(parent, dict) and parent.get("raw_sha256") == parent_sha256
            for parent in parents
        ):
            continue
        # A prior projection policy is neither a current completion receipt nor
        # evidence that the current policy is corrupt.  It may coexist during
        # a safe upgrade and is deliberately ignored here.
        if manifest.get("projection_policy_version") != PROJECTION_POLICY_VERSION:
            continue

        try:
            observed_schema = manifest.get("schema")
            if not isinstance(observed_schema, str):
                raise RawSemanticProjectionError(
                    "projection manifest schema is malformed"
                )
            projection_id = _projection_id(
                source,
                manifest_schema=observed_schema,
            )
            source_sha256 = _source_sha256(source)
        except (RawSemanticProjectionError, TypeError, ValueError):
            observed.append("invalid")
            continue
        max_child_bytes = manifest.get("max_child_bytes")
        identity_valid = (
            schema_matches(manifest.get("schema"), PROJECTION_MANIFEST_SCHEMA)
            and manifest.get("kind") == "raw_semantic_projection_manifest"
            and manifest_path.name == f"semantic-{projection_id}.manifest.json"
            and manifest.get("projection_id") == projection_id
            and manifest.get("source_sha256") == source_sha256
            and not isinstance(max_child_bytes, bool)
            and isinstance(max_child_bytes, int)
            and max_child_bytes > 0
        )
        if not identity_valid:
            observed.append("invalid")
            continue

        try:
            verify_projection_bundle(manifest_path)
        except RawSemanticProjectionError:
            status = manifest.get("status")
            if status == "delegated":
                manifest_sha256 = _sha256(manifest_bytes)
                expected_receipt = manifest_path.parent / (
                    f"semantic-{projection_id}-manifest-{manifest_sha256}.receipt.json"
                )
                any_receipt = any(
                    manifest_path.parent.glob(
                        f"semantic-{projection_id}-manifest-*.receipt.json"
                    )
                )
                # The manifest is the first durable publication and the bundle
                # receipt is the last.  No receipt at all therefore identifies
                # the one safe crash-resume window.  A differently hashed
                # receipt proves the manifest changed after completion.
                if not expected_receipt.exists() and not any_receipt:
                    observed.append("incomplete")
                else:
                    observed.append("invalid")
            elif status == "noop":
                filename = manifest.get("noop_receipt_filename")
                if (
                    isinstance(filename, str)
                    and Path(filename).name == filename
                    and not (manifest_path.parent / filename).exists()
                ):
                    observed.append("incomplete")
                else:
                    observed.append("invalid")
            else:
                observed.append("invalid")
        else:
            observed.append("completed")

    if "completed" in observed:
        return "completed"
    if "invalid" in observed:
        return "invalid"
    if "incomplete" in observed:
        return "incomplete"
    return "absent"


__all__ = [
    "PROJECTION_BUNDLE_RECEIPT_SCHEMA",
    "PROJECTION_CHILD_SCHEMA",
    "PROJECTION_MANIFEST_SCHEMA",
    "PROJECTION_NOOP_SCHEMA",
    "PROJECTION_POLICY_VERSION",
    "ProjectionArtifacts",
    "ProjectionCapacityError",
    "ProjectionChildArtifact",
    "ProjectionConflictError",
    "RawSemanticProjectionError",
    "project_parent_raw",
    "project_reassembled_raws",
    "projection_bundle_state_for_parent",
    "verify_projection_child",
    "verify_projection_bundle",
]
