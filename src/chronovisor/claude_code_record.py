"""Claude Code session saver for Chronovisor raw entries.

Reads Claude Code JSONL session transcripts and losslessly captures every
unpublished transcript record into ~/.chronovisor/raw/ without starting an LLM.

The complete source event is immutable evidence. ``role`` and ``text`` are a
deterministic semantic view used only by the downstream projection; they never
replace or sanitize the captured event.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sys
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from chronovisor.agent_save_base import (
    content_has_capture_payload as _content_has_capture_payload,
    extract_json_object,
    iter_jsonl,
    last_saved_at,
    load_state,
    publish_transcript_capture,
    publish_oversized_shadow,
    read_hook_payload,
    sanitize_keywords,
    save_raw,
    saved_line_for,
    should_process,
    trim_middle,
    update_state,
    validate_raw_keyword,
    write_state,
)
from chronovisor.decision_policy import resolve_decision_policy
from chronovisor.evidence_grounding import (
    ProtectedLiteralGroundingError,
    validate_protected_literals,
)
from chronovisor.save_transaction import (
    SaveTransaction,
    attach_save_transaction_marker,
    find_published_save_transaction,
    make_save_transaction,
    save_transaction_lock,
    validate_published_save_receipt,
)
from chronovisor.raw_segment import copy_source_interval
from chronovisor.raw_store import raw_layout_mode
from chronovisor.store import RAW_DIR, CHRONOVISOR_ROOT, init_chronovisor

DEFAULT_STATE_FILE = CHRONOVISOR_ROOT / "claude-code-save-state.json"
DEFAULT_MAX_CHARS = 120_000
# Kept as parser/API compatibility values for the legacy manual writer helpers.
# The normal save path is deterministic and never resolves or starts a model.
DEFAULT_MEMORY_MODEL = "deterministic-capture"
DEFAULT_REASONING_EFFORT = "none"
DEFAULT_TIMEOUT_SECONDS = 300
HOOK_ENABLE_ENV = "CLAUDE_CODE_CHRONOVISOR_RECORD_ENABLED"

TURN_INTERVAL = 10
COOLDOWN_SECONDS = 900
FILE_CHANGE_TOOLS = frozenset({"Edit", "Write"})

MEMORY_WRITER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["should_save", "content", "keywords", "reason", "evidence_quotes"],
    "properties": {
        "should_save": {"type": "boolean"},
        "content": {"type": "string"},
        "keywords": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 0,
            "maxItems": 20,
        },
        "reason": {"type": "string"},
        "evidence_quotes": {
            "type": "array",
            "items": {"type": "string", "minLength": 1, "maxLength": 500},
            "minItems": 0,
            "maxItems": 8,
        },
    },
}


class ClaudeCodeSaveError(RuntimeError):
    pass


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


@dataclass(frozen=True)
class WriterResult:
    should_save: bool
    content: str
    keywords: list[str]
    reason: str
    rejected_keywords: list[str]
    evidence_quotes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Session file discovery
# ---------------------------------------------------------------------------

def claude_code_projects_root() -> Path:
    return Path.home() / ".claude" / "projects"


def find_session_file(
    *,
    session_id: str | None = None,
    transcript_path: str | None = None,
) -> Path:
    if transcript_path:
        p = Path(transcript_path).expanduser()
        if p.exists():
            return p

    root = claude_code_projects_root()
    if not root.exists():
        raise ClaudeCodeSaveError(f"Claude Code projects root does not exist: {root}")

    candidates = sorted(root.rglob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise ClaudeCodeSaveError(f"No Claude Code session logs found under: {root}")

    if session_id:
        for candidate in candidates:
            if session_id in candidate.name:
                return candidate

    return candidates[0]


# ---------------------------------------------------------------------------
# JSONL parsing
# ---------------------------------------------------------------------------

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
                timestamp=item.get("timestamp") if isinstance(item.get("timestamp"), str) else None,
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
        and part.get("type") in {"tool_use", "tool_result", "server_tool_use", "web_search_tool_result"}
        for part in content
    ):
        return "tool"
    return item_type


def _content_has_file_changes(content: Any) -> bool:
    if not isinstance(content, list):
        return False
    for part in content:
        if isinstance(part, dict) and part.get("type") == "tool_use":
            if part.get("name") in FILE_CHANGE_TOOLS:
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
        or stripped.startswith("Note:") and "was read before the last conversation was summarized" in stripped[:200]
        or stripped.startswith("# AGENTS.md instructions")
        or stripped.startswith("<environment_context>")
        or stripped.startswith("<developer_context>")
        or stripped.startswith("<system_context>")
    )


# ---------------------------------------------------------------------------
# Transcript formatting
# ---------------------------------------------------------------------------

def format_transcript(records: list[TranscriptRecord]) -> str:
    parts: list[str] = []
    for record in records:
        if not record.text:
            continue
        timestamp = f" @ {record.timestamp}" if record.timestamp else ""
        parts.append(
            f"### {record.role.upper()} line {record.line}{timestamp}\n"
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
        }
        if record.event_type is not None:
            row["event_type"] = record.event_type
        if record.event is not None:
            row["event"] = record.event
        payload.append(row)
    return json.dumps(payload, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# LLM interaction
# ---------------------------------------------------------------------------

def build_writer_prompt(transcript_slice: TranscriptSlice, *, max_chars: int) -> str:
    transcript = trim_middle(format_transcript(transcript_slice.records), max_chars)
    metadata = {
        "source": "claude-code",
        "session_id": transcript_slice.session_id,
        "cwd": transcript_slice.cwd,
        "session_file": str(transcript_slice.session_file),
        "after_line": transcript_slice.after_line,
        "scanned_until_line": transcript_slice.scanned_until_line,
    }
    return (
        "You are the memory-writer subagent for Chronovisor.\n"
        "Read the Claude Code session delta and decide whether it contains durable memory.\n"
        "Durable memory includes user preferences, project decisions, implementation outcomes, "
        "environment facts, debugging findings, or lessons likely to help future sessions.\n"
        "Ignore transient command output unless it supports a durable fact.\n"
        "Do not use tools, browse the web, or modify files.\n\n"
        "Return JSON only, matching this shape:\n"
        '{"should_save": true, "content": "...", "keywords": ["..."], '
        '"reason": "...", "evidence_quotes": ["exact USER quote"]}\n\n'
        "Rules:\n"
        "- Prefer Japanese when the source conversation is Japanese.\n"
        "- Preserve exact dates, paths, commands, model names, and decisions.\n"
        "- Mention that the source is Claude Code when relevant.\n"
        "- If there is no durable memory, set should_save=false, content=\"\", keywords=[].\n"
        "- Use 10 to 20 specific keywords when saving; avoid generic words by themselves.\n"
        "- Keywords must not contain commas, brackets, colons, braces, or newlines.\n"
        "- A saved memory MUST be grounded in one or more evidence_quotes copied verbatim from USER text.\n"
        "- evidence_quotes must contain exact substrings from USER messages only; never quote ASSISTANT text.\n"
        "- If should_save=false, set evidence_quotes=[].\n"
        "- Treat ASSISTANT text only as secondary execution context, never as authority for what the user named.\n"
        "- Preserve model names, product names, versions, and spellings exactly as stated in USER evidence.\n"
        "- Never normalize, correct, expand, translate, supplement, or replace a USER-stated model/product name "
        "from ASSISTANT context (for example, do not replace Q-KUN with Qwen).\n"
        "- If a model/product identity appears only in ASSISTANT text, omit it from the memory.\n"
        "- Do not invent facts beyond the transcript.\n\n"
        "Session metadata:\n"
        f"{json.dumps(metadata, ensure_ascii=False, indent=2)}\n\n"
        "Transcript delta:\n"
        f"{transcript}\n"
    )


def run_memory_writer(
    prompt: str,
    *,
    model: str = DEFAULT_MEMORY_MODEL,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> WriterResult:
    """Reject the retired frontier writer without starting any process."""
    del prompt, model, reasoning_effort, timeout
    raise ClaudeCodeSaveError(
        "memory writer is disabled; session saves use deterministic-lossless capture"
    )


def parse_writer_output(output: str) -> WriterResult:
    parsed = extract_json_object(output)
    if not isinstance(parsed, dict):
        raise ClaudeCodeSaveError("memory writer did not return a JSON object")

    should_save = parsed.get("should_save")
    content = parsed.get("content")
    keywords = parsed.get("keywords")
    reason = parsed.get("reason")
    evidence_quotes = parsed.get("evidence_quotes")

    if not isinstance(should_save, bool):
        raise ClaudeCodeSaveError("memory writer JSON missing boolean should_save")
    if not isinstance(content, str):
        raise ClaudeCodeSaveError("memory writer JSON missing string content")
    if not isinstance(keywords, list):
        raise ClaudeCodeSaveError("memory writer JSON missing keyword list")
    if not isinstance(reason, str):
        raise ClaudeCodeSaveError("memory writer JSON missing string reason")
    if not isinstance(evidence_quotes, list):
        raise ClaudeCodeSaveError("memory writer JSON missing evidence_quotes list")

    cleaned_quotes = _validate_evidence_quote_shape(evidence_quotes)
    if should_save and not cleaned_quotes:
        raise ClaudeCodeSaveError(
            "memory writer must provide user evidence_quotes when should_save=true"
        )

    accepted, rejected = sanitize_keywords(keywords)
    return WriterResult(
        should_save=should_save,
        content=content.strip(),
        keywords=accepted,
        reason=reason.strip(),
        rejected_keywords=rejected,
        evidence_quotes=cleaned_quotes,
    )


def _validate_evidence_quote_shape(values: list[Any], *, limit: int = 8) -> list[str]:
    """Validate the structured-output quote field without changing quote text."""
    if len(values) > limit:
        raise ClaudeCodeSaveError(f"memory writer returned more than {limit} evidence quotes")
    quotes: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            raise ClaudeCodeSaveError(
                "memory writer evidence_quotes must contain only strings"
            )
        quote = value
        if not quote.strip():
            raise ClaudeCodeSaveError(
                "memory writer evidence_quotes must not contain empty strings"
            )
        if len(quote) > 500:
            raise ClaudeCodeSaveError("memory writer evidence quote exceeds 500 characters")
        if quote not in seen:
            quotes.append(quote)
            seen.add(quote)
    return quotes


def validate_user_evidence_quotes(
    writer: WriterResult,
    transcript_slice: TranscriptSlice,
) -> None:
    """Fail closed unless every save citation exists verbatim in a USER record."""
    if not writer.should_save:
        return
    if not writer.evidence_quotes:
        raise ClaudeCodeSaveError("memory writer save is missing user evidence quotes")
    user_texts = [record.text for record in transcript_slice.records if record.role == "user"]
    if not user_texts:
        raise ClaudeCodeSaveError("memory writer save has no USER transcript evidence")
    missing = [
        quote
        for quote in writer.evidence_quotes
        if not any(quote in user_text for user_text in user_texts)
    ]
    if missing:
        preview = ", ".join(repr(quote[:120]) for quote in missing[:3])
        raise ClaudeCodeSaveError(
            "memory writer evidence quote was not found verbatim in USER messages: "
            + preview
        )
    assistant_texts = [
        record.text for record in transcript_slice.records if record.role == "assistant"
    ]
    try:
        validate_protected_literals(
            {"content": writer.content, "keywords": writer.keywords},
            evidence_quotes=writer.evidence_quotes,
            context_texts=assistant_texts,
            trusted_literals=("Claude Code", "Chronovisor"),
        )
    except ProtectedLiteralGroundingError as exc:
        raise ClaudeCodeSaveError(f"memory writer {exc}") from exc


def run_grounded_memory_writer(
    prompt: str,
    transcript_slice: TranscriptSlice,
    *,
    model: str,
    reasoning_effort: str,
    timeout: int,
) -> WriterResult:
    effective_prompt = prompt
    last_error: ClaudeCodeSaveError | None = None
    for attempt in range(2):
        writer = run_memory_writer(
            effective_prompt,
            model=model,
            reasoning_effort=reasoning_effort,
            timeout=timeout,
        )
        try:
            validate_user_evidence_quotes(writer, transcript_slice)
            return writer
        except ClaudeCodeSaveError as exc:
            last_error = exc
            if attempt:
                break
            effective_prompt = (
                prompt
                + "\n\nYour previous structured answer was rejected by the deterministic grounding gate:\n"
                + str(exc)
                + "\nRegenerate the full JSON. Copy evidence_quotes character-for-character from USER text. "
                "Remove any content or keyword literal that is not explicitly present in those exact quotes. "
                "Do not paraphrase quotes. If no grounded durable memory remains, return should_save=false.\n"
            )
    assert last_error is not None
    raise last_error


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------

def bounded_transcript_slice(
    transcript_slice: TranscriptSlice,
    *,
    max_chars: int,
) -> TranscriptSlice:
    """Return a byte-bounded ordered prefix; never admit an oversized first row."""
    if max_chars < 1:
        raise ClaudeCodeSaveError("max_chars must be a positive byte limit")
    if len(_serialized_records_bytes(transcript_slice.records)) <= max_chars:
        return transcript_slice
    selected: list[TranscriptRecord] = []
    for record in transcript_slice.records:
        candidate = [*selected, record]
        if len(_serialized_records_bytes(candidate)) > max_chars:
            break
        selected.append(record)
    return replace(
        transcript_slice,
        records=selected,
        scanned_until_line=selected[-1].line if selected else transcript_slice.after_line,
        user_turn_count=sum(record.role == "user" for record in selected),
    )


def bounded_transcript_slice_for_layout(
    transcript_slice: TranscriptSlice,
    *,
    max_chars: int,
    layout: str,
) -> TranscriptSlice:
    """Allow one oversized native JSONL row only in the lossless v2 layout."""

    bounded = bounded_transcript_slice(transcript_slice, max_chars=max_chars)
    if layout != "v2" or bounded.records or not transcript_slice.records:
        return bounded
    first = transcript_slice.records[0]
    return replace(
        transcript_slice,
        records=[first],
        scanned_until_line=first.line,
        user_turn_count=1 if first.role == "user" else 0,
    )


def _serialized_records_bytes(records: list[TranscriptRecord]) -> bytes:
    return serialize_transcript_records(records).encode("utf-8")


def _oversized_fragment_transaction(
    transcript_slice: TranscriptSlice,
    *,
    record: TranscriptRecord,
    record_sha256: str,
    fragment_index: int,
    fragment_count: int,
    fragment_bytes: int,
) -> SaveTransaction:
    identity = "\0".join(
        [
            transcript_slice.session_id or "",
            "oversized-record-v1",
            str(record.line),
            record_sha256,
            str(fragment_index),
            str(fragment_count),
            str(fragment_bytes),
        ]
    )
    return make_save_transaction(
        host="claude-code",
        session_file=transcript_slice.session_file,
        session_id=identity,
        after_line=max(0, record.line - 1),
        until_line=record.line,
    )


def _build_oversized_fragment_content(
    transcript_slice: TranscriptSlice,
    *,
    record: TranscriptRecord,
    record_bytes: bytes,
    record_sha256: str,
    fragment: bytes,
    fragment_index: int,
    fragment_count: int,
    transaction: SaveTransaction,
) -> str:
    payload = {
        "schema": "chronovisor.raw-capture-fragment.v1",
        "host": "claude-code",
        "session_id": transcript_slice.session_id,
        "session_file": str(transcript_slice.session_file),
        "source_line": record.line,
        "record_sha256": record_sha256,
        "record_bytes": len(record_bytes),
        "fragment_index": fragment_index,
        "fragment_count": fragment_count,
        "fragment_bytes": len(fragment),
        "encoding": "base64",
        "data": base64.b64encode(fragment).decode("ascii"),
    }
    content = "\n".join(
        [
            "# Claude Code Oversized Transcript Record Fragment",
            "",
            "```json",
            json.dumps(payload, ensure_ascii=False, indent=2),
            "```",
            "",
        ]
    )
    return attach_save_transaction_marker(transaction, content)


def _capture_oversized_record(
    *,
    args: argparse.Namespace,
    transcript_slice: TranscriptSlice,
    state: dict[str, Any],
    state_file: Path,
    base_result: dict[str, Any],
) -> dict[str, Any]:
    """Publish one oversized record as idempotent reassemblable fragments."""
    record = transcript_slice.records[0]
    record_bytes = _serialized_records_bytes([record])
    limit = args.max_chars
    record_sha256 = hashlib.sha256(record_bytes).hexdigest()
    fragments = [record_bytes[i : i + limit] for i in range(0, len(record_bytes), limit)]
    metadata = {
        **base_result,
        "capture_mode": "deterministic-lossless",
        "oversized_record": True,
        "scanned_until_line": record.line,
        "record_count": 1,
        "record_bytes": len(record_bytes),
        "record_sha256": record_sha256,
        "fragment_bytes_limit": limit,
        "fragment_count": len(fragments),
        "keywords": ["Claude Code", "transcript-delta", "transcript-fragment"],
    }
    if args.extract_only:
        return {**metadata, "status": "extracted"}
    if args.dry_run or not args.save:
        return {**metadata, "status": "dry_run"}

    save_results: list[dict[str, Any]] = []
    for offset, fragment in enumerate(fragments, start=1):
        transaction = _oversized_fragment_transaction(
            transcript_slice,
            record=record,
            record_sha256=record_sha256,
            fragment_index=offset,
            fragment_count=len(fragments),
            fragment_bytes=limit,
        )
        content = _build_oversized_fragment_content(
            transcript_slice,
            record=record,
            record_bytes=record_bytes,
            record_sha256=record_sha256,
            fragment=fragment,
            fragment_index=offset,
            fragment_count=len(fragments),
            transaction=transaction,
        )
        save_result = save_raw(
            content,
            session_id=raw_session_id(transcript_slice),
            keywords=metadata["keywords"],
            trigger_ingest=False,
            idempotency_key=transaction.idempotency_key,
        )
        try:
            validate_published_save_receipt(
                raw_dir=RAW_DIR,
                save_result=save_result,
                expected=transaction,
            )
        except ValueError as exc:
            raise ClaudeCodeSaveError(
                f"raw fragment receipt validation failed: {exc}"
            ) from exc
        save_results.append(save_result)

    shadow = publish_oversized_shadow(
        raw_dir=RAW_DIR,
        host="claude-code",
        session_file=transcript_slice.session_file,
        session_id=transcript_slice.session_id,
        source_line=record.line,
    )
    if isinstance(shadow.get("shadow_comparison"), dict):
        shadow["shadow_comparison"]["legacy_fragment_count"] = len(fragments)
    committed_slice = replace(
        transcript_slice,
        records=[record],
        scanned_until_line=record.line,
        user_turn_count=int(record.role == "user"),
    )
    update_state(
        state,
        session_file=transcript_slice.session_file,
        transcript_slice=committed_slice,
        status="saved",
    )
    write_state(state_file, state)
    return {
        **metadata,
        "status": "saved",
        "save_result": save_results[-1],
        "save_results": save_results,
        **shadow,
    }


# ---------------------------------------------------------------------------
# Raw content building & saving
# ---------------------------------------------------------------------------

def raw_session_id(transcript_slice: TranscriptSlice) -> str:
    if transcript_slice.session_id:
        return f"claude-code-{transcript_slice.session_id}"
    digest = hashlib.sha1(str(transcript_slice.session_file).encode()).hexdigest()[:12]
    return f"claude-code-{digest}"


def build_raw_content(
    transcript_slice: TranscriptSlice,
    *,
    transaction: SaveTransaction,
) -> str:
    header = [
        "# Claude Code Session Transcript Delta",
        "",
        "- Source: Claude Code",
        "- Capture mode: deterministic-lossless",
        f"- Session ID: {transcript_slice.session_id or 'unknown'}",
        f"- CWD: {transcript_slice.cwd or 'unknown'}",
        f"- Session file: {transcript_slice.session_file}",
        f"- Lines: {transcript_slice.after_line + 1}-{transcript_slice.scanned_until_line}",
        f"- Record count: {len(transcript_slice.records)}",
        f"- Chunk order: after={transaction.after_line}; until={transaction.until_line}",
        "",
        "## Transcript Delta",
        "",
        "```json",
        serialize_transcript_records(transcript_slice.records),
        "```",
        "",
    ]
    return attach_save_transaction_marker(transaction, "\n".join(header))


# ---------------------------------------------------------------------------
# Hook payload helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def _run_save_transaction(
    args: argparse.Namespace,
    *,
    stdin_text: str | None = None,
) -> dict[str, Any]:
    if args.hook and os.environ.get(HOOK_ENABLE_ENV) != "1":
        return {
            "status": "disabled",
            "reason": f"{HOOK_ENABLE_ENV}=1 is required for hook execution",
        }

    policy, policy_mode, policy_error = resolve_decision_policy("raw_capture")
    policy_kind = policy.kind if policy is not None else None
    policy_result = {
        "lane": "raw_capture",
        "kind": policy_kind,
        "mode": policy_mode,
        "error": policy_error,
    }
    if (
        policy_error is not None
        or policy_kind != "validated_local"
        or policy_mode != "enabled"
    ):
        return {
            "status": "deferred",
            "reason": policy_error
            or (
                "raw_capture_policy_kind_invalid"
                if policy_kind != "validated_local"
                else f"raw_capture_policy_not_enabled:{policy_mode}"
            ),
            "decision_policy": policy_result,
            "model_calls": 0,
        }

    init_chronovisor()

    hints = hook_hints(read_hook_payload(stdin_text)) if args.hook else {}
    session_file = find_session_file(
        session_id=args.session_id or hints.get("session_id"),
        transcript_path=args.session_file or hints.get("transcript_path"),
    )
    if not session_file.exists():
        raise ClaudeCodeSaveError(f"session file does not exist: {session_file}")

    state_file = Path(args.state_file).expanduser()
    state = load_state(state_file)
    committed_line = saved_line_for(state, session_file)
    after_line = 0 if args.ignore_state else committed_line
    transcript_slice = extract_transcript_slice(session_file, after_line=after_line)

    # A complete raw can outlive a crash before the state cursor replace.
    # Recover that receipt before capturing overlapping transcript text again
    # (including when the transcript grew meanwhile).
    recovery_probe = (
        transcript_slice
        if after_line == committed_line
        else extract_transcript_slice(session_file, after_line=committed_line)
    )
    recovered = find_published_save_transaction(
        raw_dir=RAW_DIR,
        host="claude-code",
        session_file=session_file,
        session_id=recovery_probe.session_id,
        after_line=committed_line,
    )
    if (
        recovered is not None
        and recovered.transaction.until_line > recovery_probe.scanned_until_line
    ):
        raise ClaudeCodeSaveError(
            "published save receipt extends beyond the current transcript; "
            "refusing to publish an overlapping replacement"
        )
    if (
        recovered is not None
        and recovered.transaction.until_line <= recovery_probe.scanned_until_line
    ):
        recovered_slice = TranscriptSlice(
            session_file=session_file,
            scanned_until_line=recovered.transaction.until_line,
            records=[],
            session_id=recovery_probe.session_id,
            cwd=recovery_probe.cwd,
            after_line=committed_line,
        )
        update_state(
            state,
            session_file=session_file,
            transcript_slice=recovered_slice,
            status="saved",
        )
        write_state(state_file, state)
        after_line = recovered.transaction.until_line
        transcript_slice = extract_transcript_slice(session_file, after_line=after_line)

    base_result: dict[str, Any] = {
        "session_file": str(session_file),
        "session_id": transcript_slice.session_id,
        "cwd": transcript_slice.cwd,
        "after_line": after_line,
        "scanned_until_line": transcript_slice.scanned_until_line,
        "record_count": len(transcript_slice.records),
    }
    if recovered is not None and after_line == recovered.transaction.until_line:
        base_result["recovered_save"] = {
            "path": str(recovered.path),
            "idempotency_key": recovered.transaction.idempotency_key,
            "until_line": recovered.transaction.until_line,
        }

    if not transcript_slice.records:
        if "recovered_save" in base_result:
            return {
                **base_result,
                "status": "recovered",
                "reason": "published raw recovered before state cursor commit",
            }
        return {**base_result, "status": "skipped", "reason": "no new transcript records"}

    if not args.ignore_state:
        proceed, trigger_reason = should_process(transcript_slice, state)
        if not proceed:
            return {
                **base_result,
                "status": "skipped",
                "reason": trigger_reason,
                "has_file_changes": transcript_slice.has_file_changes,
                "user_turn_count": transcript_slice.user_turn_count,
            }
        base_result["trigger"] = trigger_reason

    if args.max_chars < 1:
        raise ClaudeCodeSaveError("max_chars must be a positive byte limit")
    layout = raw_layout_mode(chronovisor_root=RAW_DIR.parent)
    if (
        layout != "v2"
        and len(_serialized_records_bytes([transcript_slice.records[0]]))
        > args.max_chars
    ):
        return _capture_oversized_record(
            args=args,
            transcript_slice=transcript_slice,
            state=state,
            state_file=state_file,
            base_result={**base_result, "decision_policy": policy_result},
        )

    transcript_slice = bounded_transcript_slice_for_layout(
        transcript_slice,
        max_chars=args.max_chars,
        layout=layout,
    )
    base_result["scanned_until_line"] = transcript_slice.scanned_until_line
    base_result["record_count"] = len(transcript_slice.records)
    transcript_json = serialize_transcript_records(transcript_slice.records)
    transcript_bytes = transcript_json.encode("utf-8")
    if args.extract_only:
        return {
            **base_result,
            "status": "extracted",
            "capture_mode": "deterministic-lossless",
            "transcript_bytes": len(transcript_bytes),
            "transcript_sha256": hashlib.sha256(transcript_bytes).hexdigest(),
            "decision_policy": policy_result,
        }

    capture_result = {
        "capture_mode": "deterministic-lossless",
        "keywords": ["Claude Code", "transcript-delta"],
    }

    transaction = make_save_transaction(
        host="claude-code",
        session_file=session_file,
        session_id=transcript_slice.session_id,
        after_line=transcript_slice.after_line,
        until_line=transcript_slice.scanned_until_line,
    )
    raw_content = build_raw_content(
        transcript_slice,
        transaction=transaction,
    )
    source_bytes = copy_source_interval(
        session_file,
        after_line=transaction.after_line,
        until_line=transaction.until_line,
    )
    if args.dry_run or not args.save:
        raw_bytes = (
            source_bytes if layout == "v2" else raw_content.encode("utf-8")
        )
        return {
            **base_result,
            **capture_result,
            "status": "dry_run",
            "raw_content_bytes": len(raw_bytes),
            "raw_content_sha256": hashlib.sha256(raw_bytes).hexdigest(),
            "raw_layout": layout,
            "decision_policy": policy_result,
        }

    save_result = publish_transcript_capture(
        raw_dir=RAW_DIR,
        host="claude-code",
        session_key=transaction.session_key,
        session_id=transcript_slice.session_id,
        session_file=session_file,
        after_line=transaction.after_line,
        until_line=transaction.until_line,
        idempotency_key=transaction.idempotency_key,
        source_bytes=source_bytes,
        record_count=len(transcript_slice.records),
        legacy_content=raw_content,
        legacy_session_id=raw_session_id(transcript_slice),
        keywords=capture_result["keywords"],
        trigger_ingest=False,
        legacy_publisher=save_raw,
    )
    try:
        validate_published_save_receipt(
            raw_dir=RAW_DIR,
            save_result=save_result,
            expected=transaction,
        )
    except ValueError as exc:
        raise ClaudeCodeSaveError(
            f"raw save receipt validation failed: {exc}"
        ) from exc
    update_state(state, session_file=session_file, transcript_slice=transcript_slice, status="saved")
    write_state(state_file, state)
    return {
        **base_result,
        **capture_result,
        "status": "saved",
        "save_result": save_result,
        "decision_policy": policy_result,
    }


def run(args: argparse.Namespace, *, stdin_text: str | None = None) -> dict[str, Any]:
    """Run one state-serialized Claude Code save transaction."""
    state_file = Path(args.state_file).expanduser()
    session_hint = Path(args.session_file).expanduser() if args.session_file else Path(".")
    with save_transaction_lock(
        host="claude-code",
        session_file=session_hint,
        state_file=state_file,
    ):
        first = _run_save_transaction(args, stdin_text=stdin_text)
        if (
            first.get("status") != "saved"
            or not args.save
            or args.dry_run
            or args.extract_only
        ):
            return first

        # One Stop event must drain every ordered prefix, not leave the tail
        # dependent on a future Stop. Keep the serialization lock for the
        # entire drain and honor the committed cursor after the first chunk.
        chunks = [first]
        continuation_args = argparse.Namespace(**vars(args))
        continuation_args.ignore_state = False
        while True:
            current = _run_save_transaction(
                continuation_args,
                stdin_text=stdin_text,
            )
            if current.get("status") != "saved":
                break
            chunks.append(current)

        result = dict(first)
        result["chunk_count"] = len(chunks)
        result["scanned_until_line"] = chunks[-1].get(
            "scanned_until_line", result.get("scanned_until_line")
        )
        if len(chunks) > 1:
            result["save_results"] = [chunk.get("save_result") for chunk in chunks]
        return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Save Claude Code sessions into Chronovisor raw entries.")
    parser.add_argument("--session-id")
    parser.add_argument("--session-file")
    parser.add_argument("--state-file", default=str(DEFAULT_STATE_FILE))
    parser.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS)
    parser.add_argument("--model", default=DEFAULT_MEMORY_MODEL, help=argparse.SUPPRESS)
    parser.add_argument(
        "--reasoning-effort",
        default=DEFAULT_REASONING_EFFORT,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build deterministic raw data but do not save it.",
    )
    parser.add_argument("--save", action="store_true")
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help="Only parse and serialize the transcript.",
    )
    parser.add_argument("--ignore-state", action="store_true")
    parser.add_argument("--hook", action="store_true")
    parser.add_argument(
        "--trigger-ingest",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    stdin_text = sys.stdin.read() if args.hook else None
    try:
        result = run(args, stdin_text=stdin_text)
    except Exception as exc:
        result = {"status": "error", "error": str(exc)}
        print(json.dumps(result, ensure_ascii=False))
        return 1

    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
