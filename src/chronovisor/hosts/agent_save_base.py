"""Host-neutral protocol helpers for Claude Code and Codex session saves."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from chronovisor.core.link_fix import atomic_write

_RAW_KEYWORD_FORBIDDEN_CHARS = frozenset(",[]:#{}\n\r")


class TranscriptSliceProtocol(Protocol):
    """The state fields shared by both host-specific transcript slices."""

    records: list[Any]
    scanned_until_line: int
    session_id: str | None
    cwd: str | None
    has_file_changes: bool


def trim_middle(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    if max_chars < 1_000:
        return text[-max_chars:]
    head_len = max_chars // 5
    tail_len = max_chars - head_len
    return (
        text[:head_len]
        + "\n\n[... transcript trimmed for memory-writer budget ...]\n\n"
        + text[-tail_len:]
    )


def extract_json_object(output: str) -> Any:
    text = output.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 2 and lines[0].startswith("```") and lines[-1].startswith(
            "```"
        ):
            text = "\n".join(lines[1:-1]).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def validate_raw_keyword(keyword: str) -> bool:
    if not keyword:
        return False
    for character in keyword:
        if character in _RAW_KEYWORD_FORBIDDEN_CHARS:
            return False
        if ord(character) < 0x20:
            return False
    return True


def sanitize_keywords(
    values: list[Any], *, limit: int = 20
) -> tuple[list[str], list[str]]:
    accepted: list[str] = []
    rejected: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            rejected.append(repr(value))
            continue
        keyword = value.strip()
        if not validate_raw_keyword(keyword):
            rejected.append(value)
            continue
        key = keyword.casefold()
        if key in seen:
            continue
        accepted.append(keyword)
        seen.add(key)
        if len(accepted) >= limit:
            break
    return accepted, rejected


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"version": 1, "files": {}}
    try:
        parsed = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {"version": 1, "files": {}}
    if not isinstance(parsed, dict):
        return {"version": 1, "files": {}}
    parsed.setdefault("version", 1)
    parsed.setdefault("files", {})
    if not isinstance(parsed["files"], dict):
        parsed["files"] = {}
    return parsed


def saved_line_for(state: dict[str, Any], session_file: Path) -> int:
    entry = state.get("files", {}).get(str(session_file))
    if not isinstance(entry, dict):
        return 0
    value = entry.get("last_saved_line", 0)
    return value if isinstance(value, int) and value > 0 else 0


def last_saved_at(state: dict[str, Any], session_file: Path) -> datetime | None:
    entry = state.get("files", {}).get(str(session_file))
    if not isinstance(entry, dict):
        return None
    timestamp = entry.get("last_saved_at")
    if not isinstance(timestamp, str):
        return None
    try:
        return datetime.fromisoformat(timestamp)
    except (ValueError, TypeError):
        return None


def should_process(
    transcript_slice: TranscriptSliceProtocol, state: dict[str, Any]
) -> tuple[bool, str]:
    del state
    if not transcript_slice.records:
        return False, "no_messages"
    return True, "file_changes" if transcript_slice.has_file_changes else "session_tail"


def update_state(
    state: dict[str, Any],
    *,
    session_file: Path,
    transcript_slice: TranscriptSliceProtocol,
    status: str,
) -> dict[str, Any]:
    now = datetime.now(UTC).isoformat()
    files = state.setdefault("files", {})
    entry: dict[str, Any] = {
        "last_saved_line": transcript_slice.scanned_until_line,
        "session_id": transcript_slice.session_id,
        "cwd": transcript_slice.cwd,
        "status": status,
        "updated_at": now,
    }
    if status == "saved":
        entry["last_saved_at"] = now
    else:
        previous = files.get(str(session_file))
        if isinstance(previous, dict) and "last_saved_at" in previous:
            entry["last_saved_at"] = previous["last_saved_at"]
    files[str(session_file)] = entry
    return state


def write_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(path, json.dumps(state, ensure_ascii=False, indent=2) + "\n")


def save_raw(
    content: str,
    *,
    session_id: str,
    keywords: list[str],
    trigger_ingest: bool,
    idempotency_key: str,
) -> dict[str, Any]:
    from chronovisor.hosts.server import chronovisor_record

    result = chronovisor_record(
        content=content,
        session_id=session_id,
        keywords=keywords,
        trigger_ingest=trigger_ingest,
        idempotency_key=idempotency_key,
    )
    parsed = json.loads(result)
    return parsed if isinstance(parsed, dict) else {"result": parsed}


def read_hook_payload(stdin_text: str | None) -> dict[str, Any]:
    if not stdin_text:
        return {}
    try:
        parsed = json.loads(stdin_text)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}
