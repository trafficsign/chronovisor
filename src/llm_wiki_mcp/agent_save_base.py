"""Host-neutral protocol helpers for Claude Code and Codex session saves."""

from __future__ import annotations

import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Protocol

from llm_wiki_mcp.link_fix import atomic_write


_RAW_KEYWORD_FORBIDDEN_CHARS = frozenset(",[]:#{}\n\r")


class TranscriptSliceProtocol(Protocol):
    """The state fields shared by both host-specific transcript slices."""

    records: list[Any]
    scanned_until_line: int
    session_id: str | None
    cwd: str | None
    has_file_changes: bool


def iter_jsonl(path: Path):
    with path.open() as handle:
        for line_no, line in enumerate(handle, start=1):
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                yield line_no, parsed


def content_has_capture_payload(content: Any) -> bool:
    if content is None:
        return False
    if isinstance(content, str):
        return bool(content.strip())
    if isinstance(content, list):
        return bool(content)
    return True


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
    now = datetime.now(timezone.utc).isoformat()
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
    from llm_wiki_mcp.server import wiki_save_raw

    result = wiki_save_raw(
        content=content,
        session_id=session_id,
        keywords=keywords,
        trigger_ingest=trigger_ingest,
        idempotency_key=idempotency_key,
    )
    parsed = json.loads(result)
    return parsed if isinstance(parsed, dict) else {"result": parsed}


def publish_transcript_capture(
    *,
    raw_dir: Path,
    host: str,
    session_key: str,
    session_id: str | None,
    session_file: Path,
    after_line: int,
    until_line: int,
    idempotency_key: str,
    source_bytes: bytes,
    record_count: int,
    legacy_content: str,
    legacy_session_id: str,
    keywords: list[str],
    trigger_ingest: bool,
    legacy_publisher: Callable[..., dict[str, Any]] = save_raw,
) -> dict[str, Any]:
    """Publish according to the reversible legacy/shadow/v2 feature flag."""

    from llm_wiki_mcp.raw_segment import append_capture
    from llm_wiki_mcp.raw_store import raw_layout_mode

    mode = raw_layout_mode(wiki_root=raw_dir.parent)
    if mode == "legacy":
        return legacy_publisher(
            legacy_content,
            session_id=legacy_session_id,
            keywords=keywords,
            trigger_ingest=trigger_ingest,
            idempotency_key=idempotency_key,
        )

    if mode == "shadow":
        authority = legacy_publisher(
            legacy_content,
            session_id=legacy_session_id,
            keywords=keywords,
            trigger_ingest=trigger_ingest,
            idempotency_key=idempotency_key,
        )
        try:
            shadow = append_capture(
                raw_dir=raw_dir,
                raw_id=f"save-{idempotency_key}.md",
                idempotency_key=idempotency_key,
                host=host,
                session_key=session_key,
                session_id=session_id,
                source_file=session_file,
                after_line=after_line,
                until_line=until_line,
                source_bytes=source_bytes,
                record_count=record_count,
            )
        except Exception as exc:
            # Legacy is the explicit authority in shadow mode.  Its durable
            # receipt may advance the cursor, while the mismatch stays visible
            # for the adoption gate instead of taking down Stop capture.
            return {
                **authority,
                "layout": "shadow",
                "shadow_error": f"{type(exc).__name__}: {exc}",
            }
        return {
            **authority,
            "layout": "shadow",
            "shadow_result": shadow.to_result(),
            "shadow_comparison": _compare_shadow_capture(
                legacy_content=legacy_content,
                source_bytes=source_bytes,
                after_line=after_line,
                until_line=until_line,
            ),
        }

    receipt = append_capture(
        raw_dir=raw_dir,
        raw_id=f"save-{idempotency_key}.md",
        idempotency_key=idempotency_key,
        host=host,
        session_key=session_key,
        session_id=session_id,
        source_file=session_file,
        after_line=after_line,
        until_line=until_line,
        source_bytes=source_bytes,
        record_count=record_count,
    )
    return {**receipt.to_result(), "layout": "v2"}


def _compare_shadow_capture(
    *,
    legacy_content: str,
    source_bytes: bytes,
    after_line: int,
    until_line: int,
) -> dict[str, Any]:
    """Compare logical source records without requiring byte-identical envelopes."""

    try:
        source_rows = [json.loads(line) for line in source_bytes.splitlines()]
        payload_text = legacy_content.split("```json\n", 1)[1].split("\n```", 1)[0]
        legacy_rows = json.loads(payload_text)
        if not isinstance(legacy_rows, list) or any(
            not isinstance(row, dict) for row in legacy_rows
        ):
            raise ValueError("legacy transcript payload is not an object array")
        legacy_events = [row.get("event") for row in legacy_rows]
        legacy_lines = [row.get("line") for row in legacy_rows]
        expected_lines = list(range(after_line + 1, until_line + 1))
        matched = sum(
            source == legacy
            for source, legacy in zip(source_rows, legacy_events, strict=False)
        )
        duplicate_lines = len(legacy_lines) - len(set(legacy_lines))
        missing = max(0, len(source_rows) - matched)
        extra = max(0, len(legacy_events) - matched)
        status = (
            "match"
            if source_rows == legacy_events
            and legacy_lines == expected_lines
            and duplicate_lines == 0
            else "mismatch"
        )
        return {
            "status": status,
            "source_records": len(source_rows),
            "legacy_records": len(legacy_events),
            "matched_records": matched,
            "missing": missing,
            "extra": extra,
            "duplicate_lines": duplicate_lines,
            "line_identity_match": legacy_lines == expected_lines,
            "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        }
    except Exception as exc:
        return {
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
        }


def publish_oversized_shadow(
    *,
    raw_dir: Path,
    host: str,
    session_file: Path,
    session_id: str | None,
    source_line: int,
) -> dict[str, Any]:
    """Mirror a legacy fragment set as one native source record in shadow mode."""

    from llm_wiki_mcp.raw_segment import append_capture, copy_source_interval
    from llm_wiki_mcp.raw_store import raw_layout_mode
    from llm_wiki_mcp.save_transaction import make_save_transaction

    if raw_layout_mode(wiki_root=raw_dir.parent) != "shadow":
        return {}
    transaction = make_save_transaction(
        host=host,
        session_file=session_file,
        session_id=session_id,
        after_line=source_line - 1,
        until_line=source_line,
    )
    source_bytes = copy_source_interval(
        session_file,
        after_line=source_line - 1,
        until_line=source_line,
    )
    try:
        receipt = append_capture(
            raw_dir=raw_dir,
            raw_id=f"save-{transaction.idempotency_key}.md",
            idempotency_key=transaction.idempotency_key,
            host=host,
            session_key=transaction.session_key,
            session_id=session_id,
            source_file=session_file,
            after_line=transaction.after_line,
            until_line=transaction.until_line,
            source_bytes=source_bytes,
            record_count=1,
        )
    except Exception as exc:
        return {"shadow_error": f"{type(exc).__name__}: {exc}"}
    return {
        "shadow_result": receipt.to_result(),
        "shadow_comparison": {
            "status": "match",
            "source_records": 1,
            "legacy_fragment_count": None,
            "missing": 0,
            "extra": 0,
            "duplicate_lines": 0,
            "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        },
    }


def read_hook_payload(stdin_text: str | None) -> dict[str, Any]:
    if not stdin_text:
        return {}
    try:
        parsed = json.loads(stdin_text)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}
