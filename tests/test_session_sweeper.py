"""Session sweeper tests, focused on the Pi host integration.

Pi main sessions live directly under ``~/.pi/agent/sessions/**`` as
``<timestamp>_<uuid>.jsonl`` while subagent runs nest deeper as
``<session>/<agent>/run-0/session.jsonl``. The sweeper must discover the
former, ignore the latter, and only propose sessions with unsaved records.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from chronovisor.ops import session_sweeper


def _write_pi_session(path: Path, *, lines: int = 3, session_id: str = "abc") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "type": "session",
                    "version": 3,
                    "id": session_id,
                    "cwd": "/tmp",
                    "timestamp": "2026-08-13T00:00:00Z",
                }
            )
            + "\n"
        )
        for index in range(lines):
            handle.write(
                json.dumps(
                    {
                        "type": "message",
                        "timestamp": "2026-08-13T00:00:00Z",
                        "message": {
                            "role": "user" if index % 2 == 0 else "assistant",
                            "content": f"line {index}",
                        },
                    }
                )
                + "\n"
            )


def _age(path: Path, *, seconds: int = 3600) -> None:
    stamp = time.time() - seconds
    import os

    os.utime(path, (stamp, stamp))


def _isolate_roots(
    tmp_path: Path, monkeypatch, *, pi_root: Path
) -> None:
    """Point every sweeper root at temp locations so tests never touch $HOME."""
    absent = tmp_path / "absent-roots"
    monkeypatch.setattr(
        "chronovisor.raw.codex_record.default_sessions_root", lambda: absent
    )
    monkeypatch.setattr(
        "chronovisor.raw.claude_code_record.claude_code_projects_root",
        lambda: absent,
    )
    monkeypatch.setattr(
        "chronovisor.raw.pi_record.pi_projects_root", lambda: pi_root
    )
    monkeypatch.setattr(
        "chronovisor.raw.pi_record.DEFAULT_STATE_FILE",
        tmp_path / "pi-save-state.json",
    )


def test_is_user_pi_session_accepts_main_and_rejects_subagent_runs(
    tmp_path: Path,
) -> None:
    main = (
        tmp_path
        / "sessions"
        / "--Users-x--"
        / "2026-08-13T00-00-00-000Z_abc.jsonl"
    )
    sub = (
        tmp_path
        / "sessions"
        / "--Users-x--"
        / "abc"
        / "worker"
        / "run-0"
        / "session.jsonl"
    )
    assert session_sweeper._is_user_pi_session(main) is True
    assert session_sweeper._is_user_pi_session(sub) is False


def test_pending_pi_reflects_save_state_cursor(tmp_path: Path, monkeypatch) -> None:
    state_file = tmp_path / "pi-save-state.json"
    session = (
        tmp_path / "sessions" / "2026-08-13T00-00-00-000Z_abc.jsonl"
    )
    _write_pi_session(session, lines=3)
    monkeypatch.setattr(
        "chronovisor.raw.pi_record.DEFAULT_STATE_FILE", state_file
    )

    assert session_sweeper._pending_pi(session) is True

    state = {
        "version": 1,
        "files": {str(session): {"last_saved_line": 4, "status": "saved"}},
    }
    state_file.write_text(json.dumps(state), encoding="utf-8")
    assert session_sweeper._pending_pi(session) is False


def test_discover_pending_includes_pi_and_skips_subagent_runs(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "sessions"
    main = root / "--Users-x--" / "2026-08-13T00-00-00-000Z_abc.jsonl"
    sub = (
        root
        / "--Users-x--"
        / "abc"
        / "worker"
        / "run-0"
        / "session.jsonl"
    )
    _write_pi_session(main, lines=3)
    _write_pi_session(sub, lines=2, session_id="sub")
    _age(main)
    _age(sub)
    _isolate_roots(tmp_path, monkeypatch, pi_root=root)

    pending = session_sweeper.discover_pending(idle_seconds=300)

    assert ("pi", main) in pending
    assert all(host != "pi" or path != sub for host, path in pending)


def test_discover_pending_skips_fresh_pi_sessions(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "sessions"
    fresh = root / "--Users-x--" / "2026-08-13T00-00-00-000Z_abc.jsonl"
    _write_pi_session(fresh, lines=3)
    _age(fresh, seconds=10)  # newer than the idle cutoff
    _isolate_roots(tmp_path, monkeypatch, pi_root=root)

    pending = session_sweeper.discover_pending(idle_seconds=300)

    assert ("pi", fresh) not in pending
