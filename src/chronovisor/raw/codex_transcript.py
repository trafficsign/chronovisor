"""Codex transcript parsing and semantic projection."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from chronovisor.raw.transcript import (
    CodexSaveError,
    iter_jsonl,
)
from chronovisor.raw.transcript import (
    content_has_capture_payload as _content_has_capture_payload,
)

FILE_CHANGE_TOOLS = frozenset({"apply_patch", "write_file"})


@dataclass(frozen=True)
class TranscriptRecord:
    line: int
    role: str
    text: str
    timestamp: str | None = None
    phase: str | None = None
    event_type: str | None = None
    event: dict[str, Any] | None = None


@dataclass(frozen=True)
class TranscriptSlice:
    session_file: Path
    scanned_until_line: int
    records: list[TranscriptRecord]
    session_id: str | None = None
    cwd: str | None = None
    originator: str | None = None
    cli_version: str | None = None
    source: str | None = None
    model_provider: str | None = None
    after_line: int = 0
    has_file_changes: bool = False
    user_turn_count: int = 0


def read_session_meta(path: Path) -> dict[str, Any]:
    for _, item in iter_jsonl(path):
        if item.get("type") == "session_meta":
            payload = item.get("payload")
            return payload if isinstance(payload, dict) else {}
    return {}


def codex_home() -> Path:
    """Return the Codex home used for session logs."""
    configured = os.environ.get("CODEX_HOME")
    if configured:
        return Path(configured).expanduser()
    default_config = Path.home() / ".config" / "codex"
    if default_config.exists():
        return default_config
    return Path.home() / ".codex"


def default_sessions_root() -> Path:
    return codex_home() / "sessions"


def hook_hints(payload: dict[str, Any]) -> dict[str, str]:
    hints: dict[str, str] = {}

    session_id = _find_string_value(
        payload,
        ("session_id", "sessionId", "conversation_id", "conversationId", "rollout_id"),
        uuid_like=True,
    )
    if session_id:
        hints["session_id"] = session_id

    cwd = _find_string_value(payload, ("cwd", "working_directory", "workspace"))
    if cwd:
        hints["cwd"] = cwd

    session_file = _find_string_value(
        payload,
        ("session_file", "sessionFile", "transcript_path", "transcriptPath", "path"),
        suffix=".jsonl",
    )
    if session_file:
        hints["session_file"] = session_file

    return hints


def _find_string_value(
    value: Any,
    keys: tuple[str, ...],
    *,
    uuid_like: bool = False,
    suffix: str | None = None,
) -> str | None:
    if isinstance(value, dict):
        for key in keys:
            found = value.get(key)
            if isinstance(found, str) and _matches_hint(found, uuid_like, suffix):
                return found
        for child in value.values():
            found = _find_string_value(child, keys, uuid_like=uuid_like, suffix=suffix)
            if found:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_string_value(child, keys, uuid_like=uuid_like, suffix=suffix)
            if found:
                return found
    return None


def _matches_hint(value: str, uuid_like: bool, suffix: str | None) -> bool:
    text = value.strip()
    if not text:
        return False
    if suffix and not text.endswith(suffix):
        return False
    return not (uuid_like and not _looks_like_uuid(text))


def _looks_like_uuid(value: str) -> bool:
    parts = value.split("-")
    return (
        len(parts) == 5
        and [len(part) for part in parts] == [8, 4, 4, 4, 12]
        and all(all(ch in "0123456789abcdefABCDEF" for ch in part) for part in parts)
    )


def find_session_file(
    *,
    session_id: str | None = None,
    cwd: str | None = None,
    sessions_root: Path | None = None,
) -> Path:
    root = sessions_root or default_sessions_root()
    if not root.exists():
        raise CodexSaveError(f"Codex sessions root does not exist: {root}")

    candidates = sorted(root.rglob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise CodexSaveError(f"No Codex session logs found under: {root}")

    if session_id or cwd:
        for candidate in candidates:
            meta = read_session_meta(candidate)
            if session_id and meta.get("id") == session_id:
                return candidate
            if cwd and meta.get("cwd") == cwd:
                return candidate

    return candidates[0]


def extract_transcript_slice(path: Path, *, after_line: int = 0) -> TranscriptSlice:
    meta: dict[str, Any] = {}
    records: list[TranscriptRecord] = []
    scanned_until_line = 0
    has_file_changes = False
    user_turn_count = 0

    for line_no, item in iter_jsonl(path):
        scanned_until_line = max(scanned_until_line, line_no)
        item_type = item.get("type")
        if item_type == "session_meta":
            payload = item.get("payload")
            if isinstance(payload, dict):
                meta = payload
        if line_no <= after_line:
            continue

        payload = item.get("payload")
        if item_type == "response_item" and isinstance(payload, dict) and not has_file_changes:
            payload_type = payload.get("type")
            fname = str(payload.get("name") or "")
            if payload_type == "function_call" and fname in FILE_CHANGE_TOOLS:
                has_file_changes = True
            elif payload_type == "custom_tool_call" and fname == "exec":
                tool_input = payload.get("input")
                serialized = (
                    tool_input
                    if isinstance(tool_input, str)
                    else json.dumps(tool_input, default=str)
                )
                has_file_changes = bool(
                    re.search(r"\btools\.(?:apply_patch|write_file)\s*\(", serialized)
                )

        payload_type = payload.get("type") if isinstance(payload, dict) else None
        role, text = _codex_semantic_view(item_type, payload)

        if role == "user":
            user_turn_count += 1

        records.append(
            TranscriptRecord(
                line=line_no,
                role=role,
                text=text,
                timestamp=item.get("timestamp")
                if isinstance(item.get("timestamp"), str)
                else None,
                phase=(
                    payload.get("phase")
                    if isinstance(payload, dict) and isinstance(payload.get("phase"), str)
                    else None
                ),
                event_type=(
                    payload_type
                    if isinstance(payload_type, str)
                    else item_type
                    if isinstance(item_type, str)
                    else None
                ),
                # Never replace the original payload with its semantic view.
                event=item,
            )
        )

    return TranscriptSlice(
        session_file=path,
        scanned_until_line=scanned_until_line,
        records=records,
        session_id=meta.get("id") if isinstance(meta.get("id"), str) else None,
        cwd=meta.get("cwd") if isinstance(meta.get("cwd"), str) else None,
        originator=meta.get("originator")
        if isinstance(meta.get("originator"), str)
        else None,
        cli_version=meta.get("cli_version")
        if isinstance(meta.get("cli_version"), str)
        else None,
        source=meta.get("source") if isinstance(meta.get("source"), str) else None,
        model_provider=meta.get("model_provider")
        if isinstance(meta.get("model_provider"), str)
        else None,
        after_line=after_line,
        has_file_changes=has_file_changes,
        user_turn_count=user_turn_count,
    )


def _codex_semantic_view(item_type: Any, payload: Any) -> tuple[str, str]:
    """Return the non-authoritative semantic view for one complete event."""
    if item_type != "response_item" or not isinstance(payload, dict):
        return "event", ""
    payload_type = payload.get("type")
    if payload_type != "message":
        return "tool" if _is_tool_payload_type(payload_type) else "event", ""
    role = payload.get("role")
    if role not in {"user", "assistant"}:
        return "event", ""
    semantic_content = sanitize_message_content(payload.get("content"))
    if not _content_has_capture_payload(semantic_content):
        return "event", ""
    return role, message_content_text(semantic_content)


codex_semantic_view = _codex_semantic_view


def sanitize_message_content(content: Any) -> Any:
    """Build a semantic-only view; the raw ``event`` remains untouched."""
    if isinstance(content, str):
        return None if is_injected_context(content) else content
    if not isinstance(content, list):
        return content

    sanitized: list[Any] = []
    for part in content:
        if isinstance(part, dict):
            text = part.get("text")
            if isinstance(text, str) and is_injected_context(text):
                continue
        elif isinstance(part, str) and is_injected_context(part):
            continue
        sanitized.append(part)
    return sanitized


def _is_tool_payload_type(payload_type: Any) -> bool:
    return isinstance(payload_type, str) and (
        "tool" in payload_type or "function_call" in payload_type
    )


def message_content_text(content: Any) -> str:
    if isinstance(content, str):
        fragments = [content]
    elif isinstance(content, list):
        fragments = []
        for part in content:
            text = ""
            if isinstance(part, dict):
                if part.get("type") in {"input_text", "output_text"}:
                    text = part.get("text") if isinstance(part.get("text"), str) else ""
                elif isinstance(part.get("text"), str):
                    text = part["text"]
            elif isinstance(part, str):
                text = part
            if text and not is_injected_context(text):
                fragments.append(text)
    else:
        fragments = []

    clean = [fragment for fragment in fragments if fragment.strip()]
    return "\n\n".join(clean)


def is_injected_context(text: str) -> bool:
    stripped = text.lstrip()
    return (
        stripped.startswith("# AGENTS.md instructions")
        or stripped.startswith("<environment_context>")
        or stripped.startswith("<developer_context>")
        or stripped.startswith("<system_context>")
    )


def format_transcript(records: list[TranscriptRecord]) -> str:
    parts: list[str] = []
    for record in records:
        if not record.text:
            continue
        phase = f" ({record.phase})" if record.phase else ""
        timestamp = f" @ {record.timestamp}" if record.timestamp else ""
        parts.append(
            f"### {record.role.upper()}{phase} line {record.line}{timestamp}\n"
            f"{record.text}"
        )
    return "\n\n".join(parts)


def serialize_transcript_records(records: list[TranscriptRecord]) -> str:
    """Serialize complete transcript events plus their semantic view."""
    payload: list[dict[str, Any]] = []
    for record in records:
        row: dict[str, Any] = {
            "line": record.line,
            "role": record.role,
            "text": record.text,
            "timestamp": record.timestamp,
            "phase": record.phase,
        }
        if record.event_type is not None:
            row["event_type"] = record.event_type
        if record.event is not None:
            row["event"] = record.event
        payload.append(row)
    return json.dumps(payload, ensure_ascii=False, indent=2)
