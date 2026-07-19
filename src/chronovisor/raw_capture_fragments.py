"""Validate and reassemble deterministic lossless raw-capture fragments.

Oversized transcript records are stored as several immutable raw files so a
Stop hook never drops source bytes.  Those transport fragments are not useful
semantic input by themselves: the ingest boundary must either reconstruct the
entire record or leave/quarantine the complete set without calling a model.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from chronovisor.sealed_artifact_decoder import schema_matches


FRAGMENT_SCHEMA = "chronovisor.raw-capture-fragment.v1"
_FRAGMENT_TITLE_RE = re.compile(
    r"^# (?:Codex|Claude Code) Oversized Transcript Record Fragment\s*$",
    re.MULTILINE,
)
_JSON_FENCE_RE = re.compile(r"```json\s*\n(.*?)\n```", re.DOTALL)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class RawCaptureFragmentError(ValueError):
    """Raised when a file claims to be a capture fragment but is malformed."""


@dataclass(frozen=True)
class CaptureFragmentIdentity:
    host: str
    session_id: str | None
    session_file: str
    source_line: int
    record_sha256: str
    record_bytes: int
    fragment_count: int


@dataclass(frozen=True)
class CaptureFragment:
    path: Path
    identity: CaptureFragmentIdentity
    fragment_index: int
    data: bytes


@dataclass(frozen=True)
class CaptureFragmentGroup:
    identity: CaptureFragmentIdentity
    fragments: tuple[CaptureFragment, ...]

    @property
    def paths(self) -> tuple[Path, ...]:
        return tuple(fragment.path for fragment in self.fragments)

    @property
    def complete(self) -> bool:
        expected = list(range(1, self.identity.fragment_count + 1))
        observed = [fragment.fragment_index for fragment in self.fragments]
        return observed == expected

    @property
    def missing_indices(self) -> tuple[int, ...]:
        observed = {fragment.fragment_index for fragment in self.fragments}
        return tuple(
            index
            for index in range(1, self.identity.fragment_count + 1)
            if index not in observed
        )

    def assemble_bytes(self) -> bytes:
        if not self.complete:
            raise RawCaptureFragmentError(
                "capture fragment group is incomplete: missing "
                + ",".join(str(index) for index in self.missing_indices)
            )
        assembled = b"".join(fragment.data for fragment in self.fragments)
        if len(assembled) != self.identity.record_bytes:
            raise RawCaptureFragmentError(
                "reassembled record byte length does not match fragment metadata"
            )
        observed_sha256 = hashlib.sha256(assembled).hexdigest()
        if observed_sha256 != self.identity.record_sha256:
            raise RawCaptureFragmentError(
                "reassembled record sha256 does not match fragment metadata"
            )
        return assembled

    def assemble_text(self) -> str:
        try:
            return self.assemble_bytes().decode("utf-8")
        except UnicodeDecodeError as exc:
            raise RawCaptureFragmentError(
                "reassembled transcript record is not valid UTF-8"
            ) from exc

    def ingest_content(self) -> str:
        record = self.assemble_text()
        return "\n".join(
            [
                "# Reassembled Oversized Transcript Record",
                "",
                f"- Host: {self.identity.host}",
                f"- Session ID: {self.identity.session_id or ''}",
                f"- Session file: {self.identity.session_file}",
                f"- Source line: {self.identity.source_line}",
                f"- Record SHA-256: {self.identity.record_sha256}",
                "",
                "<transcript-record-json>",
                record,
                "</transcript-record-json>",
                "",
            ]
        )


def _required_string(payload: dict[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise RawCaptureFragmentError(f"fragment field {key!r} must be a string")
    return value


def _required_int(payload: dict[str, object], key: str, *, minimum: int = 1) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise RawCaptureFragmentError(
            f"fragment field {key!r} must be an integer >= {minimum}"
        )
    return value


def _parse_payload(path: Path, payload: dict[str, object]) -> CaptureFragment:
    host = _required_string(payload, "host")
    if host not in {"codex", "claude-code"}:
        raise RawCaptureFragmentError("fragment host is not recognized")
    session_id = payload.get("session_id")
    if session_id is not None and not isinstance(session_id, str):
        raise RawCaptureFragmentError("fragment session_id must be a string or null")
    session_file = _required_string(payload, "session_file")
    source_line = _required_int(payload, "source_line")
    record_sha256 = _required_string(payload, "record_sha256")
    if _SHA256_RE.fullmatch(record_sha256) is None:
        raise RawCaptureFragmentError("fragment record_sha256 is invalid")
    record_bytes = _required_int(payload, "record_bytes")
    fragment_index = _required_int(payload, "fragment_index")
    fragment_count = _required_int(payload, "fragment_count")
    if fragment_count > 100_000 or fragment_index > fragment_count:
        raise RawCaptureFragmentError("fragment index/count is invalid")
    fragment_bytes = _required_int(payload, "fragment_bytes")
    if payload.get("encoding") != "base64":
        raise RawCaptureFragmentError("fragment encoding must be base64")
    encoded = _required_string(payload, "data")
    try:
        data = base64.b64decode(encoded, validate=True)
    except (ValueError, TypeError) as exc:
        raise RawCaptureFragmentError("fragment data is invalid base64") from exc
    if len(data) != fragment_bytes:
        raise RawCaptureFragmentError(
            "decoded fragment byte length does not match fragment metadata"
        )
    if fragment_bytes > record_bytes:
        raise RawCaptureFragmentError("fragment is larger than its source record")
    return CaptureFragment(
        path=path,
        identity=CaptureFragmentIdentity(
            host=host,
            session_id=session_id,
            session_file=session_file,
            source_line=source_line,
            record_sha256=record_sha256,
            record_bytes=record_bytes,
            fragment_count=fragment_count,
        ),
        fragment_index=fragment_index,
        data=data,
    )


def parse_capture_fragment(path: Path, text: str | None = None) -> CaptureFragment | None:
    """Return a validated transport fragment, or ``None`` for a normal raw."""

    source = path.read_text(encoding="utf-8") if text is None else text
    claims_fragment = _FRAGMENT_TITLE_RE.search(source) is not None
    for match in _JSON_FENCE_RE.finditer(source):
        try:
            payload = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict) or not schema_matches(
            payload.get("schema"), FRAGMENT_SCHEMA
        ):
            continue
        return _parse_payload(path, payload)
    if claims_fragment:
        raise RawCaptureFragmentError(
            "raw claims to be a capture fragment but has no valid fragment payload"
        )
    return None


def group_capture_fragments(
    fragments: Iterable[CaptureFragment],
) -> tuple[CaptureFragmentGroup, ...]:
    grouped: dict[CaptureFragmentIdentity, list[CaptureFragment]] = {}
    for fragment in fragments:
        grouped.setdefault(fragment.identity, []).append(fragment)
    groups: list[CaptureFragmentGroup] = []
    for identity, rows in grouped.items():
        rows.sort(key=lambda row: row.fragment_index)
        indices = [row.fragment_index for row in rows]
        if len(indices) != len(set(indices)):
            raise RawCaptureFragmentError(
                f"capture fragment group {identity.record_sha256} has duplicate indices"
            )
        groups.append(CaptureFragmentGroup(identity=identity, fragments=tuple(rows)))
    return tuple(
        sorted(
            groups,
            key=lambda group: min(path.name for path in group.paths),
        )
    )
