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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llm_wiki_mcp.wiki import WIKI_ROOT, init_wiki

DEFAULT_STATE_FILE = WIKI_ROOT / "claude-code-save-state.json"
DEFAULT_MAX_CHARS = 120_000
HOOK_ENABLE_ENV = "CLAUDE_CODE_WIKI_SAVE_ENABLED"
_RAW_KEYWORD_FORBIDDEN_CHARS = frozenset(",[]:#{}\n\r")

MEMORY_WRITER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["should_save", "content", "keywords", "reason"],
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


@dataclass(frozen=True)
class WriterResult:
    should_save: bool
    content: str
    keywords: list[str]
    reason: str
    rejected_keywords: list[str]


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

        role = item_type
        text = message_content_text(item.get("message", {}).get("content"))
        if not text:
            continue

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
    )


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
        '{"should_save": true, "content": "...", "keywords": ["..."], "reason": "..."}\n\n'
        "Rules:\n"
        "- Prefer Japanese when the source conversation is Japanese.\n"
        "- Preserve exact dates, paths, commands, model names, and decisions.\n"
        "- Mention that the source is Claude Code when relevant.\n"
        "- If there is no durable memory, set should_save=false, content=\"\", keywords=[].\n"
        "- Use 10 to 20 specific keywords when saving; avoid generic words by themselves.\n"
        "- Keywords must not contain commas, brackets, colons, braces, or newlines.\n"
        "- Do not invent facts beyond the transcript.\n\n"
        "Session metadata:\n"
        f"{json.dumps(metadata, ensure_ascii=False, indent=2)}\n\n"
        "Transcript delta:\n"
        f"{transcript}\n"
    )


def run_memory_writer(prompt: str) -> WriterResult:
    from llm_wiki_mcp.ollama import generate, is_available

    if not is_available():
        raise ClaudeCodeSaveError("Ollama is not running")

    output = generate(prompt, format=MEMORY_WRITER_SCHEMA)
    return parse_writer_output(output)


def parse_writer_output(output: str) -> WriterResult:
    parsed = extract_json_object(output)
    if not isinstance(parsed, dict):
        raise ClaudeCodeSaveError("memory writer did not return a JSON object")

    should_save = parsed.get("should_save")
    content = parsed.get("content")
    keywords = parsed.get("keywords")
    reason = parsed.get("reason")

    if not isinstance(should_save, bool):
        raise ClaudeCodeSaveError("memory writer JSON missing boolean should_save")
    if not isinstance(content, str):
        raise ClaudeCodeSaveError("memory writer JSON missing string content")
    if not isinstance(keywords, list):
        raise ClaudeCodeSaveError("memory writer JSON missing keyword list")
    if not isinstance(reason, str):
        raise ClaudeCodeSaveError("memory writer JSON missing string reason")

    accepted, rejected = sanitize_keywords(keywords)
    return WriterResult(
        should_save=should_save,
        content=content.strip(),
        keywords=accepted,
        reason=reason.strip(),
        rejected_keywords=rejected,
    )


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


def update_state(
    state: dict[str, Any],
    *,
    session_file: Path,
    transcript_slice: TranscriptSlice,
    status: str,
) -> dict[str, Any]:
    files = state.setdefault("files", {})
    files[str(session_file)] = {
        "last_saved_line": transcript_slice.scanned_until_line,
        "session_id": transcript_slice.session_id,
        "cwd": transcript_slice.cwd,
        "status": status,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    return state


def write_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


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
        "## Writer Reason",
        "",
        writer.reason.strip(),
        "",
    ]
    if writer.rejected_keywords:
        header.extend(
            [
                "## Rejected Keywords",
                "",
                ", ".join(writer.rejected_keywords),
                "",
            ]
        )
    return "\n".join(header)


def save_raw(
    content: str,
    *,
    session_id: str,
    keywords: list[str],
    trigger_ingest: bool,
) -> dict[str, Any]:
    from llm_wiki_mcp.server import wiki_save_raw

    result = wiki_save_raw(
        content=content,
        session_id=session_id,
        keywords=keywords,
        trigger_ingest=trigger_ingest,
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

def run(args: argparse.Namespace, *, stdin_text: str | None = None) -> dict[str, Any]:
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
    after_line = 0 if args.ignore_state else saved_line_for(state, session_file)
    transcript_slice = extract_transcript_slice(session_file, after_line=after_line)

    base_result: dict[str, Any] = {
        "session_file": str(session_file),
        "session_id": transcript_slice.session_id,
        "cwd": transcript_slice.cwd,
        "after_line": after_line,
        "scanned_until_line": transcript_slice.scanned_until_line,
        "record_count": len(transcript_slice.records),
    }

    if not transcript_slice.records:
        if args.save and not args.dry_run:
            update_state(state, session_file=session_file, transcript_slice=transcript_slice, status="empty")
            write_state(state_file, state)
        return {**base_result, "status": "skipped", "reason": "no new user/assistant messages"}

    prompt = build_writer_prompt(transcript_slice, max_chars=args.max_chars)
    if args.extract_only:
        return {
            **base_result,
            "status": "extracted",
            "prompt_chars": len(prompt),
            "transcript_preview": format_transcript(transcript_slice.records)[:4000],
        }

    writer = run_memory_writer(prompt)
    writer_result = {
        "should_save": writer.should_save,
        "keywords": writer.keywords,
        "rejected_keywords": writer.rejected_keywords,
        "writer_reason": writer.reason,
    }

    if not writer.should_save or not writer.content:
        if args.save and not args.dry_run:
            update_state(state, session_file=session_file, transcript_slice=transcript_slice, status="declined")
            write_state(state_file, state)
        return {**base_result, **writer_result, "status": "skipped"}

    raw_content = build_raw_content(transcript_slice, writer)
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
    )
    update_state(state, session_file=session_file, transcript_slice=transcript_slice, status="saved")
    write_state(state_file, state)
    return {**base_result, **writer_result, "status": "saved", "save_result": save_result}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Save Claude Code sessions into LLM Wiki raw entries.")
    parser.add_argument("--session-id")
    parser.add_argument("--session-file")
    parser.add_argument("--state-file", default=str(DEFAULT_STATE_FILE))
    parser.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS)
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
