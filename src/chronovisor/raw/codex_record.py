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
from dataclasses import replace
from pathlib import Path
from typing import Any

from chronovisor.core import codex_transcript as _codex_transcript
from chronovisor.core.background_jobs import PERMANENT_EXIT_CODE
from chronovisor.core.paths import (
    DEFAULT_CONTEXT,
    RuntimeContext,
)
from chronovisor.core.save_transaction import (
    SaveTransaction,
    attach_save_transaction_marker,
    make_save_transaction,
    publish_oversized_shadow,
    save_transaction_lock,
    validate_published_save_receipt,
)
from chronovisor.raw.agent_save_base import (
    load_state,  # noqa: F401 - re-exported for session sweeper callers
    save_raw,
    saved_line_for,  # noqa: F401 - re-exported for session sweeper callers
    update_state,
    write_state,
)
from chronovisor.raw.codex_capture_delta import (
    CodexSaveError,
    bounded_transcript_slice_for_layout,
)
from chronovisor.raw.codex_capture_delta import (
    bounded_transcript_slice as bounded_transcript_slice,
)
from chronovisor.raw.codex_capture_delta import (
    serialized_records_bytes as _serialized_records_bytes,
)
from chronovisor.raw.record_transaction import (
    RecordTransactionSpec,
    run_save_transaction,
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
    fragments = [
        record_bytes[i : i + limit] for i in range(0, len(record_bytes), limit)
    ]
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
    session_id = (
        args.session_id if args.session_id is not None else hints.get("session_id")
    )
    cwd = args.cwd if args.cwd is not None else hints.get("cwd")

    explicit_path = (
        args.session_file
        if args.session_file is not None
        else hints.get("session_file")
    )
    if explicit_path:
        session_file = Path(explicit_path).expanduser()
        if not session_file.is_file():
            raise CodexSaveError(f"Codex session file does not exist: {session_file}")
        meta = read_session_meta(session_file)
        if session_id is not None and meta.get("id") != session_id:
            raise CodexSaveError("Codex session file does not match session_id")
        return session_file

    if session_id is None and cwd is None:
        cwd = os.environ.get("PWD")
    return find_session_file(
        session_id=session_id,
        cwd=cwd,
        sessions_root=Path(args.sessions_root).expanduser()
        if args.sessions_root
        else None,
    )


def _transaction_spec() -> RecordTransactionSpec:
    return RecordTransactionSpec(
        host="codex",
        hook_enable_env=HOOK_ENABLE_ENV,
        keywords=("Codex", "transcript-delta"),
        error_type=CodexSaveError,
        resolve_session_file=resolve_session_file,
        resolve_state_file=_resolve_state_file,
        hook_hints=hook_hints,
        extract_transcript_slice=extract_transcript_slice,
        bounded_transcript_slice_for_layout=bounded_transcript_slice_for_layout,
        serialized_records_bytes=_serialized_records_bytes,
        serialize_transcript_records=serialize_transcript_records,
        build_raw_content=build_raw_content,
        raw_session_id=raw_session_id,
        capture_oversized_record=_capture_oversized_record,
    )


def _run_save_transaction(
    args: argparse.Namespace,
    *,
    stdin_text: str | None = None,
    context: RuntimeContext | None = None,
) -> dict[str, Any]:
    return run_save_transaction(
        args,
        spec=_transaction_spec(),
        stdin_text=stdin_text,
        context=context,
    )


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
    session_hint = (
        Path(args.session_file).expanduser() if args.session_file else Path(".")
    )
    with save_transaction_lock(
        host="codex",
        session_file=session_hint,
        state_file=state_file,
    ):
        first = _run_save_transaction(args, stdin_text=stdin_text, context=context)
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
    parser = argparse.ArgumentParser(
        description="Save Codex sessions into Chronovisor raw entries."
    )
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
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build deterministic raw data but do not save it.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Write to chronovisor_record and update state.",
    )
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help="Only parse and serialize the transcript.",
    )
    parser.add_argument(
        "--ignore-state",
        action="store_true",
        help="Read the whole session instead of the delta.",
    )
    parser.add_argument(
        "--hook", action="store_true", help="Read Codex hook JSON from stdin."
    )
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
        return PERMANENT_EXIT_CODE if getattr(exc, "permanent", False) else 1

    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
