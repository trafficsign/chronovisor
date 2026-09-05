"""Shared deterministic transcript save transaction for raw host recorders."""

from __future__ import annotations

import argparse
import hashlib
import os
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from chronovisor.core.raw_segment import copy_source_interval
from chronovisor.core.raw_store import raw_layout_mode
from chronovisor.core.save_transaction import (
    find_published_save_transaction,
    make_save_transaction,
    publish_transcript_capture,
    validate_published_save_receipt,
)
from chronovisor.core.store import RAW_DIR, RuntimeContext, init_chronovisor
from chronovisor.decision.decision_policy import resolve_decision_policy
from chronovisor.raw.agent_save_base import (
    load_state,
    read_hook_payload,
    save_raw,
    saved_line_for,
    should_process,
    update_state,
    write_state,
)


@dataclass(frozen=True)
class RecordTransactionSpec:
    """Host-specific transcript and publication formatting operations."""

    host: str
    hook_enable_env: str
    keywords: tuple[str, ...]
    error_type: type[Exception]
    resolve_session_file: Callable[[argparse.Namespace, dict[str, str]], Path]
    resolve_state_file: Callable[..., Path]
    hook_hints: Callable[[dict[str, Any]], dict[str, str]]
    extract_transcript_slice: Callable[..., Any]
    bounded_transcript_slice_for_layout: Callable[..., Any]
    serialized_records_bytes: Callable[[list[Any]], bytes]
    serialize_transcript_records: Callable[[list[Any]], str]
    build_raw_content: Callable[..., str]
    raw_session_id: Callable[[Any], str]
    capture_oversized_record: Callable[..., dict[str, Any]]


def run_save_transaction(
    args: argparse.Namespace,
    *,
    spec: RecordTransactionSpec,
    stdin_text: str | None = None,
    context: RuntimeContext | None = None,
) -> dict[str, Any]:
    """Run one deterministic, state-serialized host transcript capture."""
    if args.hook and os.environ.get(spec.hook_enable_env) != "1":
        return {
            "status": "disabled",
            "reason": f"{spec.hook_enable_env}=1 is required for hook execution",
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

    hints = spec.hook_hints(read_hook_payload(stdin_text)) if args.hook else {}
    session_file = spec.resolve_session_file(args, hints)
    if not session_file.exists():
        raise spec.error_type(f"session file does not exist: {session_file}")

    state_file = spec.resolve_state_file(args, context=context)
    state = load_state(state_file)
    committed_line = saved_line_for(state, session_file)
    after_line = 0 if args.ignore_state else committed_line
    transcript_slice = spec.extract_transcript_slice(
        session_file, after_line=after_line
    )

    # A complete raw can outlive a crash before the state cursor replace.
    # Recover that receipt before capturing overlapping transcript text again
    # (including when the transcript grew meanwhile).
    recovery_probe = (
        transcript_slice
        if after_line == committed_line
        else spec.extract_transcript_slice(session_file, after_line=committed_line)
    )
    recovered = find_published_save_transaction(
        raw_dir=raw_dir,
        host=spec.host,
        session_file=session_file,
        session_id=recovery_probe.session_id,
        after_line=committed_line,
    )
    if (
        recovered is not None
        and recovered.transaction.until_line > recovery_probe.scanned_until_line
    ):
        raise spec.error_type(
            "published save receipt extends beyond the current transcript; "
            "refusing to publish an overlapping replacement"
        )
    if (
        recovered is not None
        and recovered.transaction.until_line <= recovery_probe.scanned_until_line
    ):
        recovered_slice = replace(
            recovery_probe,
            session_file=session_file,
            scanned_until_line=recovered.transaction.until_line,
            records=[],
            after_line=committed_line,
            has_file_changes=False,
            user_turn_count=0,
        )
        update_state(
            state,
            session_file=session_file,
            transcript_slice=recovered_slice,
            status="saved",
        )
        write_state(state_file, state)
        after_line = recovered.transaction.until_line
        transcript_slice = spec.extract_transcript_slice(
            session_file, after_line=after_line
        )

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
        raise spec.error_type("max_chars must be a positive byte limit")
    layout = raw_layout_mode(chronovisor_root=raw_dir.parent)
    if context is not None and layout != "v2":
        raise spec.error_type("RuntimeContext capture requires raw layout v2")
    if (
        layout != "v2"
        and len(spec.serialized_records_bytes([transcript_slice.records[0]]))
        > args.max_chars
    ):
        return spec.capture_oversized_record(
            args=args,
            transcript_slice=transcript_slice,
            state=state,
            state_file=state_file,
            raw_dir=raw_dir,
            base_result={**base_result, "decision_policy": policy_result},
        )

    transcript_slice = spec.bounded_transcript_slice_for_layout(
        transcript_slice,
        max_chars=args.max_chars,
        layout=layout,
    )
    base_result["scanned_until_line"] = transcript_slice.scanned_until_line
    base_result["record_count"] = len(transcript_slice.records)
    transcript_json = spec.serialize_transcript_records(transcript_slice.records)
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

    capture_result: dict[str, Any] = {
        "capture_mode": "deterministic-lossless",
        "keywords": list(spec.keywords),
    }

    transaction = make_save_transaction(
        host=spec.host,
        session_file=session_file,
        session_id=transcript_slice.session_id,
        after_line=transcript_slice.after_line,
        until_line=transcript_slice.scanned_until_line,
    )
    raw_content = spec.build_raw_content(
        transcript_slice,
        transaction=transaction,
    )
    source_bytes = copy_source_interval(
        session_file,
        after_line=transaction.after_line,
        until_line=transaction.until_line,
    )
    if args.dry_run or not args.save:
        raw_bytes = source_bytes if layout == "v2" else raw_content.encode("utf-8")
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
        host=spec.host,
        session_key=transaction.session_key,
        session_id=transcript_slice.session_id,
        session_file=session_file,
        after_line=transaction.after_line,
        until_line=transaction.until_line,
        idempotency_key=transaction.idempotency_key,
        source_bytes=source_bytes,
        record_count=len(transcript_slice.records),
        legacy_content=raw_content,
        legacy_session_id=spec.raw_session_id(transcript_slice),
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
        raise spec.error_type(f"raw save receipt validation failed: {exc}") from exc
    update_state(
        state,
        session_file=session_file,
        transcript_slice=transcript_slice,
        status="saved",
    )
    write_state(state_file, state)
    return {
        **base_result,
        **capture_result,
        "status": "saved",
        "save_result": save_result,
        "decision_policy": policy_result,
    }
