"""Claude Code session saver for LLM Wiki raw entries.

Reads Claude Code JSONL session transcripts, extracts deltas since last save,
asks an LLM whether the content is worth saving, and writes to ~/.wiki/raw/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llm_wiki_mcp.evidence_grounding import (
    ProtectedLiteralGroundingError,
    validate_protected_literals,
)
from llm_wiki_mcp.link_fix import atomic_write
from llm_wiki_mcp.save_transaction import (
    SaveTransaction,
    attach_save_transaction_marker,
    find_published_save_transaction,
    make_save_transaction,
    save_transaction_lock,
    validate_published_save_receipt,
)
from llm_wiki_mcp.wiki import RAW_DIR, WIKI_ROOT, init_wiki

DEFAULT_STATE_FILE = WIKI_ROOT / "claude-code-save-state.json"
DEFAULT_MAX_CHARS = 120_000
DEFAULT_MEMORY_MODEL = "gpt-5.4-mini"
DEFAULT_REASONING_EFFORT = "medium"
DEFAULT_TIMEOUT_SECONDS = 300
HOOK_ENABLE_ENV = "CLAUDE_CODE_WIKI_SAVE_ENABLED"

TURN_INTERVAL = 10
COOLDOWN_SECONDS = 900
FILE_CHANGE_TOOLS = frozenset({"Edit", "Write"})
_RAW_KEYWORD_FORBIDDEN_CHARS = frozenset(",[]:#{}\n\r")

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

def iter_jsonl(path: Path):
    with path.open() as f:
        for line_no, line in enumerate(f, start=1):
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                yield line_no, parsed


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
        if item_type not in ("user", "assistant"):
            continue

        content = item.get("message", {}).get("content")

        if item_type == "assistant" and not has_file_changes:
            has_file_changes = _content_has_file_changes(content)

        role = item_type
        text = message_content_text(content)
        if not text:
            continue

        if role == "user":
            user_turn_count += 1

        records.append(
            TranscriptRecord(
                line=line_no,
                role=role,
                text=text,
                timestamp=item.get("timestamp") if isinstance(item.get("timestamp"), str) else None,
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
        return content.strip()

    if isinstance(content, list):
        fragments: list[str] = []
        for part in content:
            if not isinstance(part, dict):
                continue
            ptype = part.get("type")
            if ptype == "text":
                text = part.get("text", "")
                if isinstance(text, str) and text.strip() and not is_injected_context(text):
                    fragments.append(text.strip())
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
        timestamp = f" @ {record.timestamp}" if record.timestamp else ""
        parts.append(
            f"### {record.role.upper()} line {record.line}{timestamp}\n"
            f"{record.text.strip()}"
        )
    return "\n\n".join(parts)


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
        "You are the memory-writer subagent for LLM Wiki.\n"
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
    """Use the frontier writer; a local proposal never finalizes data loss."""
    from llm_wiki_mcp.codex_save import run_memory_writer as run_frontier_writer

    result = run_frontier_writer(
        prompt,
        model=model,
        reasoning_effort=reasoning_effort,
        timeout=timeout,
    )
    return WriterResult(
        should_save=result.should_save,
        content=result.content,
        keywords=list(result.keywords),
        reason=result.reason,
        rejected_keywords=list(result.rejected_keywords),
        evidence_quotes=list(result.evidence_quotes),
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
            trusted_literals=("Claude Code", "LLM Wiki"),
        )
    except ProtectedLiteralGroundingError as exc:
        raise ClaudeCodeSaveError(f"memory writer {exc}") from exc


def extract_json_object(output: str) -> Any:
    text = output.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 2 and lines[0].startswith("```") and lines[-1].startswith("```"):
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


def sanitize_keywords(values: list[Any], *, limit: int = 20) -> tuple[list[str], list[str]]:
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


def validate_raw_keyword(keyword: str) -> bool:
    if not keyword:
        return False
    for ch in keyword:
        if ch in _RAW_KEYWORD_FORBIDDEN_CHARS:
            return False
        if ord(ch) < 0x20:
            return False
    return True


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------

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
    ts = entry.get("last_saved_at")
    if not isinstance(ts, str):
        return None
    try:
        return datetime.fromisoformat(ts)
    except (ValueError, TypeError):
        return None


def should_process(transcript_slice: TranscriptSlice, state: dict[str, Any]) -> tuple[bool, str]:
    if not transcript_slice.records:
        return False, "no_messages"
    return True, "file_changes" if transcript_slice.has_file_changes else "session_tail"


def bounded_transcript_slice(
    transcript_slice: TranscriptSlice,
    *,
    max_chars: int,
) -> TranscriptSlice:
    """Return an ordered prefix so a large delta is dispositioned in chunks."""
    if len(format_transcript(transcript_slice.records)) <= max_chars:
        return transcript_slice
    selected: list[TranscriptRecord] = []
    for record in transcript_slice.records:
        candidate = [*selected, record]
        if selected and len(format_transcript(candidate)) > max_chars:
            break
        selected.append(record)
    if not selected:
        selected = [transcript_slice.records[0]]
    return replace(
        transcript_slice,
        records=selected,
        scanned_until_line=selected[-1].line,
        user_turn_count=sum(record.role == "user" for record in selected),
    )


def update_state(
    state: dict[str, Any],
    *,
    session_file: Path,
    transcript_slice: TranscriptSlice,
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
        prev = files.get(str(session_file))
        if isinstance(prev, dict) and "last_saved_at" in prev:
            entry["last_saved_at"] = prev["last_saved_at"]
    files[str(session_file)] = entry
    return state


def write_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(path, json.dumps(state, ensure_ascii=False, indent=2) + "\n")


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
    writer: WriterResult,
    *,
    transaction: SaveTransaction,
) -> str:
    header = [
        "# Claude Code Session Memory Save",
        "",
        f"- Source: Claude Code",
        f"- Session ID: {transcript_slice.session_id or 'unknown'}",
        f"- CWD: {transcript_slice.cwd or 'unknown'}",
        f"- Session file: {transcript_slice.session_file}",
        f"- Lines: {transcript_slice.after_line + 1}-{transcript_slice.scanned_until_line}",
        f"- Generated at: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Memory",
        "",
        writer.content.strip(),
        "",
        "## User Evidence",
        "",
    ]
    for quote in writer.evidence_quotes:
        header.extend(["\n".join(f"> {line}" for line in quote.splitlines()), ""])
    if writer.rejected_keywords:
        header.extend(
            [
                "## Rejected Keywords",
                "",
                ", ".join(writer.rejected_keywords),
                "",
            ]
        )
    return attach_save_transaction_marker(transaction, "\n".join(header))


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


# ---------------------------------------------------------------------------
# Hook payload helpers
# ---------------------------------------------------------------------------

def read_hook_payload(stdin_text: str | None) -> dict[str, Any]:
    if not stdin_text:
        return {}
    try:
        parsed = json.loads(stdin_text)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


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
    init_wiki()

    if args.hook and os.environ.get(HOOK_ENABLE_ENV) != "1":
        return {
            "status": "disabled",
            "reason": f"{HOOK_ENABLE_ENV}=1 is required for hook execution",
        }

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
    # Recover that receipt before asking the writer to summarize overlapping
    # transcript text again (including when the transcript grew meanwhile).
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
        return {**base_result, "status": "skipped", "reason": "no new user/assistant messages"}

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

    transcript_slice = bounded_transcript_slice(transcript_slice, max_chars=args.max_chars)
    base_result["scanned_until_line"] = transcript_slice.scanned_until_line
    base_result["record_count"] = len(transcript_slice.records)
    prompt = build_writer_prompt(transcript_slice, max_chars=args.max_chars)
    if args.extract_only:
        return {
            **base_result,
            "status": "extracted",
            "prompt_chars": len(prompt),
            "transcript_preview": format_transcript(transcript_slice.records)[:4000],
        }

    writer = run_memory_writer(
        prompt,
        model=getattr(args, "model", DEFAULT_MEMORY_MODEL),
        reasoning_effort=getattr(args, "reasoning_effort", DEFAULT_REASONING_EFFORT),
        timeout=getattr(args, "timeout", DEFAULT_TIMEOUT_SECONDS),
    )
    validate_user_evidence_quotes(writer, transcript_slice)
    writer_result = {
        "should_save": writer.should_save,
        "keywords": writer.keywords,
        "rejected_keywords": writer.rejected_keywords,
        "writer_reason": writer.reason,
        "evidence_quotes": writer.evidence_quotes,
    }

    if not writer.should_save or not writer.content:
        if args.save and not args.dry_run:
            update_state(state, session_file=session_file, transcript_slice=transcript_slice, status="declined")
            write_state(state_file, state)
        return {**base_result, **writer_result, "status": "skipped"}

    transaction = make_save_transaction(
        host="claude-code",
        session_file=session_file,
        session_id=transcript_slice.session_id,
        after_line=transcript_slice.after_line,
        until_line=transcript_slice.scanned_until_line,
    )
    raw_content = build_raw_content(
        transcript_slice,
        writer,
        transaction=transaction,
    )
    if args.dry_run or not args.save:
        return {
            **base_result,
            **writer_result,
            "status": "dry_run",
            "raw_content_preview": raw_content[:4000],
        }

    save_result = save_raw(
        raw_content,
        session_id=raw_session_id(transcript_slice),
        keywords=writer.keywords,
        trigger_ingest=args.trigger_ingest,
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
            f"raw save receipt validation failed: {exc}"
        ) from exc
    update_state(state, session_file=session_file, transcript_slice=transcript_slice, status="saved")
    write_state(state_file, state)
    return {**base_result, **writer_result, "status": "saved", "save_result": save_result}


def run(args: argparse.Namespace, *, stdin_text: str | None = None) -> dict[str, Any]:
    """Run one state-serialized Claude Code save transaction."""
    state_file = Path(args.state_file).expanduser()
    session_hint = Path(args.session_file).expanduser() if args.session_file else Path(".")
    with save_transaction_lock(
        host="claude-code",
        session_file=session_hint,
        state_file=state_file,
    ):
        return _run_save_transaction(args, stdin_text=stdin_text)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Save Claude Code sessions into LLM Wiki raw entries.")
    parser.add_argument("--session-id")
    parser.add_argument("--session-file")
    parser.add_argument("--state-file", default=str(DEFAULT_STATE_FILE))
    parser.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS)
    parser.add_argument("--model", default=DEFAULT_MEMORY_MODEL)
    parser.add_argument("--reasoning-effort", default=DEFAULT_REASONING_EFFORT)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument("--ignore-state", action="store_true")
    parser.add_argument("--hook", action="store_true")
    parser.add_argument(
        "--trigger-ingest",
        action="store_true",
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
