"""Pi transcript parsing and semantic projection.

Parses Pi session JSONL (``~/.pi/agent/sessions/**/*.jsonl``) into the same
``TranscriptRecord``/``TranscriptSlice`` shape used by the other hosts, so the
raw save pipeline is host-agnostic.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from chronovisor.core.transcript import (
    ClaudeCodeSaveError,
    iter_jsonl,
)
from chronovisor.core.transcript import (
    content_has_capture_payload as _content_has_capture_payload,
)

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


def pi_projects_root() -> Path:
    return Path.home() / ".pi" / "agent" / "sessions"


def find_session_file(
    *,
    session_id: str | None = None,
    transcript_path: str | None = None,
) -> Path:
    if transcript_path:
        p = Path(transcript_path).expanduser()
        if p.exists():
            return p

    root = pi_projects_root()
    if not root.exists():
        raise ClaudeCodeSaveError(f"Pi sessions root does not exist: {root}")

    candidates = sorted(root.rglob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise ClaudeCodeSaveError(f"No Pi session logs found under: {root}")

    if session_id:
        for candidate in candidates:
            if session_id in candidate.name:
                return candidate

    return candidates[0]


def hook_hints(payload: dict[str, Any]) -> dict[str, str]:
    hints: dict[str, str] = {}
    for key in ("session_id", "sessionId"):
        val = payload.get(key)
        if isinstance(val, str) and val:
            hints["session_id"] = val
            break
    for key in ("transcript_path", "transcriptPath", "session_file", "sessionFile"):
        val = payload.get(key)
        if isinstance(val, str) and val:
            hints["transcript_path"] = val
            break
    for key in ("cwd", "working_directory"):
        val = payload.get(key)
        if isinstance(val, str) and val:
            hints["cwd"] = val
            break
    return hints


def _pi_session_meta(item: dict[str, Any]) -> tuple[str | None, str | None]:
    """Extract (session_id, cwd) from a Pi ``type: "session"`` entry."""
    if item.get("type") == "session":
        sid = item.get("id")
        cwd = item.get("cwd")
        return (
            sid if isinstance(sid, str) else None,
            cwd if isinstance(cwd, str) else None,
        )
    return None, None


def _pi_message_view(item: dict[str, Any]) -> tuple[str, Any]:
    """Return (event_type, content) for a Pi message entry.

    Pi wraps messages as ``{"type": "message", "message": {role, content}}``.
    The event type is the role: user / assistant / toolResult.
    """
    if item.get("type") != "message":
        return "event", None
    msg = item.get("message")
    if not isinstance(msg, dict):
        return "event", None
    role = msg.get("role")
    if not isinstance(role, str) or role not in {"user", "assistant", "toolResult"}:
        return "event", None
    return role, msg.get("content")


def extract_transcript_slice(path: Path, *, after_line: int = 0) -> TranscriptSlice:
    records: list[TranscriptRecord] = []
    scanned_until_line = 0
    session_id: str | None = None
    cwd: str | None = None
    has_file_changes = False
    user_turn_count = 0

    for line_no, item in iter_jsonl(path):
        scanned_until_line = max(scanned_until_line, line_no)

        sid, c = _pi_session_meta(item)
        if not session_id and sid:
            session_id = sid
        if not cwd and c:
            cwd = c

        if line_no <= after_line:
            continue

        item_type, content = _pi_message_view(item)
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
    if item_type not in {"user", "assistant", "toolResult"}:
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
    # Pi tool results are a distinct role, never a user/assistant text turn.
    if item_type == "toolResult":
        return "tool"
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
