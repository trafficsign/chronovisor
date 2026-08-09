"""Codex session saver for lossless Chronovisor raw entries.

Every unpublished JSONL event is captured unchanged. ``role`` and ``text`` are
a deterministic semantic view for downstream projection only; privileged,
injected, reasoning and tool events remain intact in the raw evidence.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from chronovisor.core.store import (
    DEFAULT_CONTEXT,
    RAW_DIR,
    RuntimeContext,
    init_chronovisor,
)
from chronovisor.decision.decision_policy import resolve_decision_policy
from chronovisor.hosts.agent_save_base import (
    extract_json_object,
    load_state,
    read_hook_payload,
    sanitize_keywords,
    save_raw,
    saved_line_for,
    should_process,
    trim_middle,
    update_state,
    write_state,
)
from chronovisor.hosts.codex_capture_delta import (
    CodexSaveError,
    _serialized_records_bytes,
    bounded_transcript_slice_for_layout,
)
from chronovisor.hosts.codex_capture_delta import (
    bounded_transcript_slice as bounded_transcript_slice,
)
from chronovisor.raw import codex_transcript as _codex_transcript
from chronovisor.raw.evidence_grounding import (
    ProtectedLiteralGroundingError,
    validate_protected_literals,
)
from chronovisor.raw.raw_segment import copy_source_interval
from chronovisor.raw.raw_store import raw_layout_mode
from chronovisor.raw.save_transaction import (
    SaveTransaction,
    attach_save_transaction_marker,
    find_published_save_transaction,
    make_save_transaction,
    publish_oversized_shadow,
    publish_transcript_capture,
    save_transaction_lock,
    validate_published_save_receipt,
)

FILE_CHANGE_TOOLS = _codex_transcript.FILE_CHANGE_TOOLS
TranscriptRecord = _codex_transcript.TranscriptRecord
TranscriptSlice = _codex_transcript.TranscriptSlice
_codex_semantic_view = _codex_transcript._codex_semantic_view
_is_tool_payload_type = _codex_transcript._is_tool_payload_type
codex_semantic_view = _codex_transcript.codex_semantic_view
codex_home = _codex_transcript.codex_home
default_sessions_root = _codex_transcript.default_sessions_root
extract_transcript_slice = _codex_transcript.extract_transcript_slice
find_session_file = _codex_transcript.find_session_file
format_transcript = _codex_transcript.format_transcript
hook_hints = _codex_transcript.hook_hints
is_injected_context = _codex_transcript.is_injected_context
message_content_text = _codex_transcript.message_content_text
read_session_meta = _codex_transcript.read_session_meta
sanitize_message_content = _codex_transcript.sanitize_message_content
serialize_transcript_records = _codex_transcript.serialize_transcript_records

# Kept as parser/API compatibility values for the legacy manual writer helpers.
# The normal save path is deterministic and never resolves or starts a model.
DEFAULT_MEMORY_MODEL = "deterministic-capture"
DEFAULT_REASONING_EFFORT = "none"
DEFAULT_STATE_FILE = DEFAULT_CONTEXT.codex_state_file
DEFAULT_MAX_CHARS = 120_000
DEFAULT_TIMEOUT_SECONDS = 300
HOOK_ENABLE_ENV = "CODEX_CHRONOVISOR_RECORD_ENABLED"

TURN_INTERVAL = 10
COOLDOWN_SECONDS = 900

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


@dataclass(frozen=True)
class WriterResult:
    should_save: bool
    content: str
    keywords: list[str]
    reason: str
    rejected_keywords: list[str]
    evidence_quotes: list[str] = field(default_factory=list)


def build_writer_prompt(transcript_slice: TranscriptSlice, *, max_chars: int) -> str:
    transcript = trim_middle(format_transcript(transcript_slice.records), max_chars)
    metadata = {
        "source": "codex",
        "session_id": transcript_slice.session_id,
        "cwd": transcript_slice.cwd,
        "originator": transcript_slice.originator,
        "cli_version": transcript_slice.cli_version,
        "session_file": str(transcript_slice.session_file),
        "after_line": transcript_slice.after_line,
        "scanned_until_line": transcript_slice.scanned_until_line,
    }
    return (
        "You are the memory-writer subagent for Chronovisor.\n"
        "Read the Codex session delta and decide whether it contains durable memory.\n"
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
        "- Mention that the source is Codex when relevant.\n"
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
    """Reject the retired frontier writer without starting any process.

    Raw session persistence is deliberately deterministic and lossless.  This
    compatibility function remains so older callers fail with a useful error
    instead of silently reintroducing a routine ``codex exec`` path.
    """
    del prompt, model, reasoning_effort, timeout
    raise CodexSaveError(
        "memory writer is disabled; session saves use deterministic-lossless capture"
    )


def parse_writer_output(output: str) -> WriterResult:
    parsed = extract_json_object(output)
    if not isinstance(parsed, dict):
        raise CodexSaveError("memory writer did not return a JSON object")

    should_save = parsed.get("should_save")
    content = parsed.get("content")
    keywords = parsed.get("keywords")
    reason = parsed.get("reason")
    evidence_quotes = parsed.get("evidence_quotes")

    if not isinstance(should_save, bool):
        raise CodexSaveError("memory writer JSON missing boolean should_save")
    if not isinstance(content, str):
        raise CodexSaveError("memory writer JSON missing string content")
    if not isinstance(keywords, list):
        raise CodexSaveError("memory writer JSON missing keyword list")
    if not isinstance(reason, str):
        raise CodexSaveError("memory writer JSON missing string reason")
    if not isinstance(evidence_quotes, list):
        raise CodexSaveError("memory writer JSON missing evidence_quotes list")

    cleaned_quotes = _validate_evidence_quote_shape(evidence_quotes)
    if should_save and not cleaned_quotes:
        raise CodexSaveError("memory writer must provide user evidence_quotes when should_save=true")

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
        raise CodexSaveError(f"memory writer returned more than {limit} evidence quotes")
    quotes: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            raise CodexSaveError("memory writer evidence_quotes must contain only strings")
        quote = value
        if not quote.strip():
            raise CodexSaveError("memory writer evidence_quotes must not contain empty strings")
        if len(quote) > 500:
            raise CodexSaveError("memory writer evidence quote exceeds 500 characters")
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
        raise CodexSaveError("memory writer save is missing user evidence quotes")
    user_texts = [record.text for record in transcript_slice.records if record.role == "user"]
    if not user_texts:
        raise CodexSaveError("memory writer save has no USER transcript evidence")
    missing = [
        quote
        for quote in writer.evidence_quotes
        if not any(quote in user_text for user_text in user_texts)
    ]
    if missing:
        preview = ", ".join(repr(quote[:120]) for quote in missing[:3])
        raise CodexSaveError(
            "memory writer evidence quote was not found verbatim in USER messages: " + preview
        )
    assistant_texts = [
        record.text for record in transcript_slice.records if record.role == "assistant"
    ]
    try:
        validate_protected_literals(
            {"content": writer.content, "keywords": writer.keywords},
            evidence_quotes=writer.evidence_quotes,
            context_texts=assistant_texts,
            trusted_literals=("Codex", "Chronovisor"),
        )
    except ProtectedLiteralGroundingError as exc:
        raise CodexSaveError(f"memory writer {exc}") from exc


def run_grounded_memory_writer(
    prompt: str,
    transcript_slice: TranscriptSlice,
    *,
    model: str,
    reasoning_effort: str,
    timeout: int,
) -> WriterResult:
    """Validate a legacy writer result; the default writer is disabled."""
    effective_prompt = prompt
    last_error: CodexSaveError | None = None
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
        except CodexSaveError as exc:
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


def _oversized_fragment_transaction(
    transcript_slice: TranscriptSlice,
    *,
    record: TranscriptRecord,
    record_sha256: str,
    fragment_index: int,
    fragment_count: int,
    fragment_bytes: int,
) -> SaveTransaction:
    # The synthetic session identity makes every fragment a distinct,
    # deterministic idempotency key without changing the source interval.
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
        host="codex",
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
        "host": "codex",
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
            "# Codex Oversized Transcript Record Fragment",
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
    raw_dir: Path,
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
        "keywords": ["Codex", "transcript-delta", "transcript-fragment"],
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
                raw_dir=raw_dir,
                save_result=save_result,
                expected=transaction,
            )
        except ValueError as exc:
            raise CodexSaveError(
                f"raw fragment receipt validation failed: {exc}"
            ) from exc
        save_results.append(save_result)

    shadow = publish_oversized_shadow(
        raw_dir=raw_dir,
        host="codex",
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
    # The cursor moves only after every fragment has a validated receipt.
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


def raw_session_id(transcript_slice: TranscriptSlice) -> str:
    if transcript_slice.session_id:
        return f"codex-{transcript_slice.session_id}"
    digest = hashlib.sha1(str(transcript_slice.session_file).encode()).hexdigest()[:12]
    return f"codex-{digest}"


def build_raw_content(
    transcript_slice: TranscriptSlice,
    *,
    transaction: SaveTransaction,
) -> str:
    header = [
        "# Codex Session Transcript Delta",
        "",
        "- Source: Codex",
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


def resolve_session_file(args: argparse.Namespace, hints: dict[str, str]) -> Path:
    if args.session_file:
        return Path(args.session_file).expanduser()
    if hints.get("session_file"):
        return Path(hints["session_file"]).expanduser()

    session_id = args.session_id or hints.get("session_id")
    cwd = args.cwd or hints.get("cwd") or os.environ.get("PWD")
    return find_session_file(
        session_id=session_id,
        cwd=cwd,
        sessions_root=Path(args.sessions_root).expanduser() if args.sessions_root else None,
    )


def _run_save_transaction(
    args: argparse.Namespace,
    *,
    stdin_text: str | None = None,
    context: RuntimeContext | None = None,
) -> dict[str, Any]:
    if args.hook and os.environ.get(HOOK_ENABLE_ENV) != "1":
        return {
            "status": "disabled",
            "reason": f"{HOOK_ENABLE_ENV}=1 is required for hook execution",
        }

    if context is None:
        policy, policy_mode, policy_error = resolve_decision_policy("raw_capture")
    else:
        policy, policy_mode, policy_error = resolve_decision_policy(
            "raw_capture", config_path=context.config_file
        )
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

    if context is None:
        init_chronovisor()
    else:
        init_chronovisor(context=context)

    raw_dir = RAW_DIR if context is None else context.raw_dir

    hints = hook_hints(read_hook_payload(stdin_text)) if args.hook else {}
    session_file = resolve_session_file(args, hints)
    if not session_file.exists():
        raise CodexSaveError(f"session file does not exist: {session_file}")

    state_file = _resolve_state_file(args, context=context)
    state = load_state(state_file)
    committed_line = saved_line_for(state, session_file)
    after_line = 0 if args.ignore_state else committed_line
    transcript_slice = extract_transcript_slice(session_file, after_line=after_line)

    # A process may die after the raw becomes visible but before the cursor
    # state replace. The raw itself is the receipt, so recover it before
    # considering a wider new delta.
    recovery_probe = (
        transcript_slice
        if after_line == committed_line
        else extract_transcript_slice(session_file, after_line=committed_line)
    )
    recovered = find_published_save_transaction(
        raw_dir=raw_dir,
        host="codex",
        session_file=session_file,
        session_id=recovery_probe.session_id,
        after_line=committed_line,
    )
    if (
        recovered is not None
        and recovered.transaction.until_line > recovery_probe.scanned_until_line
    ):
        raise CodexSaveError(
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
            originator=recovery_probe.originator,
            cli_version=recovery_probe.cli_version,
            source=recovery_probe.source,
            model_provider=recovery_probe.model_provider,
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
        raise CodexSaveError("max_chars must be a positive byte limit")
    layout = raw_layout_mode(chronovisor_root=raw_dir.parent)
    if context is not None and layout != "v2":
        raise CodexSaveError("RuntimeContext capture requires raw layout v2")
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
            raw_dir=raw_dir,
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
        "keywords": ["Codex", "transcript-delta"],
    }

    transaction = make_save_transaction(
        host="codex",
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
        raw_dir=raw_dir,
        host="codex",
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
            raw_dir=raw_dir,
            save_result=save_result,
            expected=transaction,
        )
    except ValueError as exc:
        raise CodexSaveError(f"raw save receipt validation failed: {exc}") from exc
    update_state(state, session_file=session_file, transcript_slice=transcript_slice, status="saved")
    write_state(state_file, state)
    return {
        **base_result,
        **capture_result,
        "status": "saved",
        "save_result": save_result,
        "decision_policy": policy_result,
    }


def _resolve_state_file(
    args: argparse.Namespace, *, context: RuntimeContext | None
) -> Path:
    if args.state_file is not None:
        return Path(args.state_file).expanduser()
    return DEFAULT_STATE_FILE if context is None else context.codex_state_file


def run(
    args: argparse.Namespace,
    *,
    stdin_text: str | None = None,
    context: RuntimeContext | None = None,
) -> dict[str, Any]:
    """Run one state-serialized Codex save transaction.

    The lock deliberately covers raw publication, not just the final JSON
    replace. A second Stop worker therefore reloads the committed cursor and
    cannot publish the same delta.
    """
    state_file = _resolve_state_file(args, context=context)
    session_hint = Path(args.session_file).expanduser() if args.session_file else Path(".")
    with save_transaction_lock(
        host="codex",
        session_file=session_hint,
        state_file=state_file,
    ):
        first = _run_save_transaction(
            args, stdin_text=stdin_text, context=context
        )
        if (
            first.get("status") != "saved"
            or not args.save
            or args.dry_run
            or args.extract_only
        ):
            return first

        # ``bounded_transcript_slice`` intentionally publishes an ordered
        # prefix. Drain every remaining prefix while this same transaction
        # lock is held so a single Stop event is genuinely lossless even when
        # its delta exceeds ``max_chars``. After the first chunk, state must be
        # honored even when a manual caller requested ``--ignore-state``.
        chunks = [first]
        continuation_args = argparse.Namespace(**vars(args))
        continuation_args.ignore_state = False
        while True:
            current = _run_save_transaction(
                continuation_args,
                stdin_text=stdin_text,
                context=context,
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
    parser = argparse.ArgumentParser(description="Save Codex sessions into Chronovisor raw entries.")
    parser.add_argument("--session-id")
    parser.add_argument("--cwd")
    parser.add_argument("--session-file")
    parser.add_argument("--sessions-root")
    parser.add_argument("--state-file")
    parser.add_argument("--model", default=DEFAULT_MEMORY_MODEL, help=argparse.SUPPRESS)
    parser.add_argument(
        "--reasoning-effort",
        default=DEFAULT_REASONING_EFFORT,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS)
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--dry-run", action="store_true", help="Build deterministic raw data but do not save it.")
    parser.add_argument("--save", action="store_true", help="Write to chronovisor_record and update state.")
    parser.add_argument("--extract-only", action="store_true", help="Only parse and serialize the transcript.")
    parser.add_argument("--ignore-state", action="store_true", help="Read the whole session instead of the delta.")
    parser.add_argument("--hook", action="store_true", help="Read Codex hook JSON from stdin.")
    parser.add_argument(
        "--trigger-ingest",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the ``chronovisor-codex-record`` command-line entry point."""
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
