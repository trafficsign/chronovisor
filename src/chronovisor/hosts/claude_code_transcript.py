"""Claude Code transcript parsing and semantic projection."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from chronovisor.hosts.agent_save_base import (
    content_has_capture_payload as _content_has_capture_payload,
)
from chronovisor.hosts.agent_save_base import iter_jsonl

FILE_CHANGE_TOOLS = frozenset({"Edit", "Write"})


@dataclass(frozen=True)
class TranscriptRecord:
    line: int
    role: str
    text: str
    timestamp: str | None = None
    event_type: str | None = None
    event: dict[str, Any] | None = None


@dataclass(frozen=True)
class TranscriptSlice:
    session_file: Path
    scanned_until_line: int
    records: list[TranscriptRecord]
    session_id: str | None = None
    cwd: str | None = None
    after_line: int = 0
    has_file_changes: bool = False
    user_turn_count: int = 0


def extract_transcript_slice(path: Path, *, after_line: int = 0) -> TranscriptSlice:
    records: list[TranscriptRecord] = []
    scanned_until_line = 0
    session_id: str | None = None
    cwd: str | None = None
    has_file_changes = False
    user_turn_count = 0

    for line_no, item in iter_jsonl(path):
        scanned_until_line = max(scanned_until_line, line_no)
        item_type = item.get("type")

        if not session_id:
            sid = item.get("sessionId")
            if isinstance(sid, str):
                session_id = sid
        if not cwd:
            c = item.get("cwd")
            if isinstance(c, str):
                cwd = c

        if line_no <= after_line:
            continue

        message = item.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if item_type == "assistant" and not has_file_changes:
            has_file_changes = _content_has_file_changes(content)

        role, text = _claude_semantic_view(item_type, content)

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
                event_type=item_type if isinstance(item_type, str) else None,
                # Canonical JSON serialization remains byte-equivalent to the
                # parsed source object; semantic filtering never mutates it.
                event=item,
            )
        )

    return TranscriptSlice(
        session_file=path,
        scanned_until_line=scanned_until_line,
        records=records,
        session_id=session_id,
        cwd=cwd,
        after_line=after_line,
        has_file_changes=has_file_changes,
        user_turn_count=user_turn_count,
    )


def _claude_semantic_view(item_type: Any, content: Any) -> tuple[str, str]:
    """Return the non-authoritative semantic view for one complete event."""
    if item_type not in {"user", "assistant"}:
        return "event", ""
    semantic_content = sanitize_message_content(content)
    if not _content_has_capture_payload(semantic_content):
        return "event", ""
    text = message_content_text(semantic_content)
    return _claude_record_role(item_type, semantic_content, text), text


claude_semantic_view = _claude_semantic_view


def sanitize_message_content(content: Any) -> Any:
    """Build a semantic-only view; the raw ``event`` remains untouched."""
    if isinstance(content, str):
        return None if is_injected_context(content) else content
    if not isinstance(content, list):
        return content

    sanitized: list[Any] = []
    for part in content:
        if isinstance(part, dict):
            ptype = part.get("type")
            if ptype in {"thinking", "redacted_thinking"}:
                continue
            text = part.get("text")
            if isinstance(text, str) and is_injected_context(text):
                continue
        elif isinstance(part, str) and is_injected_context(part):
            continue
        sanitized.append(part)
    return sanitized


def _claude_record_role(item_type: str, content: Any, text: str) -> str:
    if text:
        return item_type
    if isinstance(content, list) and any(
        isinstance(part, dict)
        and part.get("type")
        in {"tool_use", "tool_result", "server_tool_use", "web_search_tool_result"}
        for part in content
    ):
        return "tool"
    return item_type


def _content_has_file_changes(content: Any) -> bool:
    if not isinstance(content, list):
        return False
    for part in content:
        if (
            isinstance(part, dict)
            and part.get("type") == "tool_use"
            and part.get("name") in FILE_CHANGE_TOOLS
        ):
            return True
    return False


def message_content_text(content: Any) -> str:
    if isinstance(content, str):
        if is_injected_context(content):
            return ""
        return content if content.strip() else ""

    if isinstance(content, list):
        fragments: list[str] = []
        for part in content:
            if not isinstance(part, dict):
                continue
            ptype = part.get("type")
            if ptype == "text":
                text = part.get("text", "")
                if isinstance(text, str) and text.strip() and not is_injected_context(text):
                    fragments.append(text)
        return "\n\n".join(fragments)

    return ""


def is_injected_context(text: str) -> bool:
    stripped = text.lstrip()
    return (
        stripped.startswith("<system-reminder>")
        or stripped.startswith("<command-name>")
        or stripped.startswith("<local-command-caveat>")
        or stripped.startswith("<local-command-stdout>")
        or stripped.startswith("<local-command-stderr>")
        or stripped.startswith("Note:")
        and "was read before the last conversation was summarized" in stripped[:200]
        or stripped.startswith("# AGENTS.md instructions")
        or stripped.startswith("<environment_context>")
        or stripped.startswith("<developer_context>")
        or stripped.startswith("<system_context>")
    )


def format_transcript(records: list[TranscriptRecord]) -> str:
    parts: list[str] = []
    for record in records:
        if not record.text:
            continue
        timestamp = f" @ {record.timestamp}" if record.timestamp else ""
        parts.append(
            f"### {record.role.upper()} line {record.line}{timestamp}\n" f"{record.text}"
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
        }
        if record.event_type is not None:
            row["event_type"] = record.event_type
        if record.event is not None:
            row["event"] = record.event
        payload.append(row)
    return json.dumps(payload, ensure_ascii=False, indent=2)
