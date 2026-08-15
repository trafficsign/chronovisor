"""Deterministic Hermes state.db transcript capture for Chronovisor."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from chronovisor.core.link_fix import atomic_write
from chronovisor.core.save_transaction import (
    SaveTransaction,
    attach_save_transaction_marker,
    make_save_transaction,
    publish_transcript_capture,
    save_session_key,
    save_transaction_lock,
)
from chronovisor.core.store import DEFAULT_CONTEXT, RAW_DIR, init_chronovisor
from chronovisor.raw.agent_save_base import read_hook_payload, save_raw

HOOK_ENABLE_ENV = "HERMES_CHRONOVISOR_RECORD_ENABLED"
DEFAULT_STATE_FILE = DEFAULT_CONTEXT.hermes_state_file


@dataclass(frozen=True)
class HermesSession:
    session_id: str
    host: str
    platform: str
    provider: str | None
    model: str | None
    cwd: str | None
    profile: str | None
    capture_failed: bool | None


@dataclass(frozen=True)
class HermesRecord:
    line: int
    message_id: int
    role: str
    text: str
    timestamp: str | None
    event: dict[str, Any]


@dataclass(frozen=True)
class HermesTranscriptDelta:
    state_db: Path
    session: HermesSession
    records: tuple[HermesRecord, ...]
    after_line: int
    scanned_until_line: int
    last_message_id: int


def _message_columns(conn: sqlite3.Connection) -> tuple[str, ...]:
    return tuple(str(row[1]) for row in conn.execute("PRAGMA table_info(messages)"))


def _row_dict(row: sqlite3.Row) -> dict[str, Any]:
    return dict(row)


def _timestamp(value: object) -> str | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return datetime.fromtimestamp(float(value), UTC).isoformat()
    return value if isinstance(value, str) else None


def _semantic_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def extract_transcript_delta(
    state_db: Path,
    *,
    session_id: str,
    after_message_id: int,
    after_line: int,
    capture_failed: bool | None = None,
) -> HermesTranscriptDelta:
    """Read one stable Hermes message delta from a read-only state database."""

    resolved = state_db.expanduser().resolve(strict=True)
    with sqlite3.connect(f"{resolved.as_uri()}?mode=ro", uri=True) as conn:
        conn.row_factory = sqlite3.Row
        session_row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
        if session_row is None:
            raise ValueError(f"Hermes session does not exist: {session_id}")
        session_data = _row_dict(session_row)
        message_columns = _message_columns(conn)
        if not {"id", "session_id", "role", "content", "timestamp"}.issubset(
            message_columns
        ):
            raise ValueError("Hermes messages table is missing required columns")
        rows = conn.execute(
            "SELECT * FROM messages WHERE session_id = ? AND id > ? ORDER BY id",
            (session_id, after_message_id),
        ).fetchall()

    session = HermesSession(
        session_id=session_id,
        host="hermes",
        platform=str(session_data.get("source") or "unknown"),
        provider=session_data.get("billing_provider"),
        model=session_data.get("model"),
        cwd=session_data.get("cwd"),
        profile=session_data.get("profile_name"),
        capture_failed=capture_failed,
    )
    records: list[HermesRecord] = []
    session_event = {
        "session_id": session.session_id,
        "platform": session.platform,
        "provider": session.provider,
        "model": session.model,
        "cwd": session.cwd,
        "profile": session.profile,
        "capture_failed": session.capture_failed,
    }
    for index, row in enumerate(rows, start=1):
        message = _row_dict(row)
        role = str(message.get("role") or "unknown")
        timestamp = _timestamp(message.get("timestamp"))
        event = {
            "schema": "chronovisor.hermes-message.v1",
            "host": "hermes",
            "session": session_event,
            "message": message,
            "timestamp": timestamp,
        }
        records.append(
            HermesRecord(
                line=after_line + index,
                message_id=int(message["id"]),
                role=role,
                text=_semantic_text(message.get("content")),
                timestamp=timestamp,
                event=event,
            )
        )
    last_message_id = records[-1].message_id if records else after_message_id
    return HermesTranscriptDelta(
        state_db=resolved,
        session=session,
        records=tuple(records),
        after_line=after_line,
        scanned_until_line=after_line + len(records),
        last_message_id=last_message_id,
    )


def _serialized_records(delta: HermesTranscriptDelta) -> str:
    rows = [
        {
            "line": record.line,
            "role": record.role,
            "text": record.text,
            "timestamp": record.timestamp,
            "event": record.event,
        }
        for record in delta.records
    ]
    return json.dumps(rows, ensure_ascii=False, indent=2)


def build_raw_content(
    delta: HermesTranscriptDelta, *, transaction: SaveTransaction
) -> str:
    """Build the self-verifying legacy envelope used by transcript projection."""

    session = delta.session
    header = [
        "# Hermes Session Transcript Delta",
        "",
        "- Source: Hermes",
        "- Capture mode: deterministic-full-row",
        f"- Session ID: {session.session_id}",
        f"- Platform: {session.platform}",
        f"- Provider: {session.provider or 'unknown'}",
        f"- Model: {session.model or 'unknown'}",
        f"- Capture Failed: {str(session.capture_failed).lower() if session.capture_failed is not None else 'unknown'}",
        f"- CWD: {session.cwd or 'unknown'}",
        f"- State DB: {delta.state_db}",
        f"- Messages: {delta.after_line + 1}-{delta.scanned_until_line}",
        f"- Record count: {len(delta.records)}",
        f"- Chunk order: after={transaction.after_line}; until={transaction.until_line}",
        "",
        "## Transcript Delta",
        "",
        "```json",
        _serialized_records(delta),
        "```",
        "",
    ]
    return attach_save_transaction_marker(transaction, "\n".join(header))


def _load_cursor(
    path: Path, cursor_key: str, *, legacy_session_id: str
) -> tuple[int, int]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return 0, 0
    sessions = payload.get("sessions") if isinstance(payload, dict) else None
    entry = sessions.get(cursor_key) if isinstance(sessions, dict) else None
    if not isinstance(entry, dict) and isinstance(sessions, dict):
        entry = sessions.get(legacy_session_id)
    if not isinstance(entry, dict):
        return 0, 0
    message_id = entry.get("last_saved_message_id")
    line = entry.get("last_saved_line")
    return (
        message_id if isinstance(message_id, int) and message_id > 0 else 0,
        line if isinstance(line, int) and line > 0 else 0,
    )


def _write_cursor(
    path: Path,
    *,
    cursor_key: str,
    session_id: str,
    state_db: Path,
    message_id: int,
    line: int,
) -> None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    sessions = payload.setdefault("sessions", {})
    if not isinstance(sessions, dict):
        sessions = {}
        payload["sessions"] = sessions
    sessions.pop(session_id, None)
    sessions[cursor_key] = {
        "session_id": session_id,
        "state_db": str(state_db),
        "last_saved_message_id": message_id,
        "last_saved_line": line,
        "updated_at": datetime.now(UTC).isoformat(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _source_bytes(delta: HermesTranscriptDelta) -> bytes:
    return b"".join(
        json.dumps(record.event, ensure_ascii=False, sort_keys=True).encode("utf-8") + b"\n"
        for record in delta.records
    )


def save_session(
    *,
    state_db: Path,
    session_id: str,
    state_file: Path,
    raw_dir: Path = RAW_DIR,
    capture_failed: bool | None = None,
    trigger_ingest: bool = False,
) -> dict[str, Any]:
    """Capture every previously unpublished Hermes DB message exactly once."""

    init_chronovisor()
    with save_transaction_lock(host="hermes", session_file=state_db, state_file=state_file):
        cursor_key = save_session_key(
            host="hermes", session_file=state_db, session_id=session_id
        )
        after_message_id, after_line = _load_cursor(
            state_file, cursor_key, legacy_session_id=session_id
        )
        delta = extract_transcript_delta(
            state_db,
            session_id=session_id,
            after_message_id=after_message_id,
            after_line=after_line,
            capture_failed=capture_failed,
        )
        if not delta.records:
            return {"status": "skipped", "reason": "no_messages", "host": "hermes"}
        transaction = make_save_transaction(
            host="hermes",
            session_file=delta.state_db,
            session_id=session_id,
            after_line=after_line,
            until_line=delta.scanned_until_line,
        )
        result = publish_transcript_capture(
            raw_dir=raw_dir,
            host="hermes",
            session_key=transaction.session_key,
            session_id=session_id,
            session_file=delta.state_db,
            after_line=transaction.after_line,
            until_line=transaction.until_line,
            idempotency_key=transaction.idempotency_key,
            source_bytes=_source_bytes(delta),
            record_count=len(delta.records),
            legacy_content=build_raw_content(delta, transaction=transaction),
            legacy_session_id=f"hermes-{session_id}",
            keywords=["hermes", delta.session.platform, delta.session.model or "unknown-model"],
            trigger_ingest=trigger_ingest,
            legacy_publisher=save_raw,
        )
        _write_cursor(
            state_file,
            cursor_key=cursor_key,
            session_id=session_id,
            state_db=delta.state_db,
            message_id=delta.last_message_id,
            line=delta.scanned_until_line,
        )
        return {
            **result,
            "status": str(result.get("status") or "saved"),
            "host": "hermes",
            "session_id": session_id,
            "last_message_id": delta.last_message_id,
            "record_count": len(delta.records),
        }


def hermes_state_db() -> Path:
    """Default Hermes state database (HERMES_HOME or ~/.hermes/state.db)."""
    hermes_home = Path(os.environ.get("HERMES_HOME") or Path.home() / ".hermes")
    return hermes_home / "state.db"


def list_sessions(
    state_db: Path, *, include_subagent: bool = False
) -> list[dict[str, Any]]:
    """Return Hermes session rows, oldest last-activity first."""
    resolved = state_db.expanduser().resolve(strict=True)
    with sqlite3.connect(f"{resolved.as_uri()}?mode=ro", uri=True) as conn:
        conn.row_factory = sqlite3.Row
        where = "" if include_subagent else "WHERE source <> 'subagent'"
        rows = conn.execute(
            f"""
            SELECT id, source, model, billing_provider, profile_name, cwd,
                   message_count, started_at, ended_at, last_activity_at
            FROM sessions
            {where}
            ORDER BY COALESCE(last_activity_at, started_at) ASC
            """
        ).fetchall()
    return [_row_dict(row) for row in rows]


def _pending_message_count(
    conn: sqlite3.Connection, *, session_id: str, after_message_id: int
) -> int:
    row = conn.execute(
        "SELECT COUNT(*) FROM messages WHERE session_id = ? AND id > ?",
        (session_id, after_message_id),
    ).fetchone()
    return int(row[0]) if row else 0


def pending_session_ids(
    state_db: Path,
    *,
    state_file: Path = DEFAULT_STATE_FILE,
    idle_seconds: int = 300,
    include_subagent: bool = False,
) -> list[str]:
    """Return Hermes session IDs that still have unpublished messages.

    Skips sessions whose last activity is newer than ``idle_seconds`` (mirroring
    the file-mtime cutoff used by the file-backed hosts) and, by default, skips
    ``subagent`` sources (mirroring ``_is_user_pi_session``).
    """
    resolved = state_db.expanduser().resolve(strict=True)
    cutoff = time.time() - max(0, idle_seconds)
    sessions = list_sessions(resolved, include_subagent=include_subagent)
    pending: list[str] = []
    for session in sessions:
        session_id = str(session["id"])
        last_activity = session.get("last_activity_at")
        if isinstance(last_activity, (int, float)) and last_activity > cutoff:
            continue
        cursor_key = save_session_key(
            host="hermes", session_file=resolved, session_id=session_id
        )
        after_message_id, _after_line = _load_cursor(
            state_file, cursor_key, legacy_session_id=session_id
        )
        with sqlite3.connect(f"{resolved.as_uri()}?mode=ro", uri=True) as conn:
            count = _pending_message_count(
                conn, session_id=session_id, after_message_id=after_message_id
            )
        if count > 0:
            pending.append(session_id)
    return pending


def save_pending_sessions(
    *,
    state_db: Path,
    state_file: Path = DEFAULT_STATE_FILE,
    raw_dir: Path = RAW_DIR,
    idle_seconds: int = 300,
    include_subagent: bool = False,
    limit: int | None = None,
) -> dict[str, Any]:
    """Capture every pending Hermes session in a state database, oldest first.

    Session-level failures are captured per-session so one broken session never
    blocks the rest of the database.
    """
    resolved = state_db.expanduser().resolve(strict=True)
    session_ids = pending_session_ids(
        resolved,
        state_file=state_file,
        idle_seconds=idle_seconds,
        include_subagent=include_subagent,
    )
    if not session_ids:
        return {
            "status": "skipped",
            "reason": "no_pending_sessions",
            "host": "hermes",
        }
    session_results: list[dict[str, Any]] = []
    total_records = 0
    for session_id in session_ids:
        if limit is not None and len(session_results) >= limit:
            break
        try:
            result = save_session(
                state_db=resolved,
                session_id=session_id,
                state_file=state_file,
                raw_dir=raw_dir,
                trigger_ingest=True,
            )
            record_count = result.get("record_count")
        except Exception as exc:
            result = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
            record_count = 0
        session_results.append({"session_id": session_id, **result})
        if isinstance(record_count, int):
            total_records += record_count
    saved = sum(1 for r in session_results if r.get("status") == "saved")
    errors = [r for r in session_results if r.get("status") == "error"]
    status = "saved" if saved else ("error" if errors else "skipped")
    return {
        "status": status,
        "host": "hermes",
        "session_count": len(session_results),
        "record_count": total_records,
        "session_results": session_results,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Capture a Hermes session into Chronovisor")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--hook", action="store_true")
    parser.add_argument("--state-db")
    parser.add_argument("--session-id")
    parser.add_argument("--state-file")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.hook and os.environ.get(HOOK_ENABLE_ENV) != "1":
        print(
            json.dumps(
                {
                    "status": "disabled",
                    "reason": f"{HOOK_ENABLE_ENV}=1 is required for hook execution",
                }
            )
        )
        return 0
    payload = read_hook_payload(sys.stdin.read()) if args.hook else {}
    session_id = args.session_id or payload.get("session_id")
    state_db_value = args.state_db or payload.get("state_db")
    if not isinstance(session_id, str) or not session_id:
        print(json.dumps({"status": "error", "error": "session_id is required"}))
        return 2
    if not isinstance(state_db_value, str) or not state_db_value:
        state_db_value = str(hermes_state_db())
    state_file = (
        Path(args.state_file).expanduser() if args.state_file else DEFAULT_STATE_FILE
    )
    try:
        payload_failed = payload.get("failed")
        result = save_session(
            state_db=Path(state_db_value).expanduser(),
            session_id=session_id,
            state_file=state_file,
            capture_failed=payload_failed if isinstance(payload_failed, bool) else None,
        )
    except Exception as exc:
        print(json.dumps({"status": "error", "error": f"{type(exc).__name__}: {exc}"}))
        return 1
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
