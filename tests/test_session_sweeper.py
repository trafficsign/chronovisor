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
    tmp_path: Path, monkeypatch, *, pi_root: Path, hermes_db: Path | None = None
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
    monkeypatch.setattr(
        "chronovisor.raw.hermes_record.hermes_state_db",
        lambda: hermes_db if hermes_db is not None else absent / "state.db",
    )
    monkeypatch.setattr(
        "chronovisor.raw.hermes_record.DEFAULT_STATE_FILE",
        tmp_path / "hermes-save-state.json",
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


def _write_hermes_db(path: Path) -> None:
    """Hermes state.db with one idle pending session and one subagent session."""
    import sqlite3
    import time as time_module

    path.parent.mkdir(parents=True, exist_ok=True)
    now = time_module.time()
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE sessions (
                id TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                model TEXT,
                cwd TEXT,
                billing_provider TEXT,
                profile_name TEXT,
                message_count INTEGER DEFAULT 0,
                started_at REAL NOT NULL,
                ended_at REAL,
                last_activity_at REAL
            );
            CREATE TABLE messages (
                id INTEGER PRIMARY KEY,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT,
                timestamp REAL NOT NULL,
                active INTEGER NOT NULL DEFAULT 1,
                compacted INTEGER NOT NULL DEFAULT 0
            );
            """
        )
        conn.executemany(
            "INSERT INTO sessions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                ("hermes-main", "cli", "gpt-test", "/tmp/a", None, "default", 2, now - 3600, None, now - 3600),
                ("hermes-sub", "subagent", "deepseek", "/tmp/b", None, "default", 2, now - 3600, None, now - 3600),
            ],
        )
        conn.executemany(
            "INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (1, "hermes-main", "user", "hello", now - 3600, 1, 0),
                (2, "hermes-main", "assistant", "world", now - 3599, 1, 0),
                (3, "hermes-sub", "user", "sub only", now - 3600, 1, 0),
            ],
        )


def test_pending_hermes_reflects_save_state_cursor(tmp_path: Path, monkeypatch) -> None:
    state_file = tmp_path / "hermes-save-state.json"
    state_db = tmp_path / "state.db"
    _write_hermes_db(state_db)
    monkeypatch.setattr(
        "chronovisor.raw.hermes_record.DEFAULT_STATE_FILE", state_file
    )

    assert session_sweeper._pending_hermes(state_db, idle_seconds=300) is True

    # Advance the cursor past every message of the only user session.
    import json

    from chronovisor.core.save_transaction import save_session_key

    cursor_key = save_session_key(
        host="hermes", session_file=state_db, session_id="hermes-main"
    )
    state_file.write_text(
        json.dumps(
            {
                "version": 1,
                "sessions": {
                    cursor_key: {
                        "session_id": "hermes-main",
                        "state_db": str(state_db),
                        "last_saved_message_id": 2,
                        "last_saved_line": 2,
                        "updated_at": "2026-08-14T00:00:00Z",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    assert session_sweeper._pending_hermes(state_db, idle_seconds=300) is False


def test_discover_pending_includes_hermes_db(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "sessions"
    main = root / "--Users-x--" / "2026-08-13T00-00-00-000Z_abc.jsonl"
    _write_pi_session(main, lines=3)
    _age(main)
    hermes_db = tmp_path / "hermes" / "state.db"
    _write_hermes_db(hermes_db)
    _isolate_roots(tmp_path, monkeypatch, pi_root=root, hermes_db=hermes_db)

    pending = session_sweeper.discover_pending(idle_seconds=300)

    assert ("hermes", hermes_db) in pending
    assert ("pi", main) in pending


def test_discover_pending_skips_hermes_when_nothing_pending(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "sessions"
    hermes_db = tmp_path / "hermes" / "state.db"
    _write_hermes_db(hermes_db)
    # Mark every user-session message as already saved.
    import json

    from chronovisor.core.save_transaction import save_session_key

    cursor_key = save_session_key(
        host="hermes", session_file=hermes_db, session_id="hermes-main"
    )
    state_file = tmp_path / "hermes-save-state.json"
    state_file.write_text(
        json.dumps(
            {
                "version": 1,
                "sessions": {
                    cursor_key: {
                        "session_id": "hermes-main",
                        "state_db": str(hermes_db),
                        "last_saved_message_id": 2,
                        "last_saved_line": 2,
                        "updated_at": "2026-08-14T00:00:00Z",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    _isolate_roots(tmp_path, monkeypatch, pi_root=root, hermes_db=hermes_db)

    pending = session_sweeper.discover_pending(idle_seconds=300)

    assert all(host != "hermes" for host, _path in pending)


def test_discover_pending_skips_fresh_hermes_sessions(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "sessions"
    hermes_db = tmp_path / "hermes" / "state.db"
    _write_hermes_db(hermes_db)
    # Make the only user session fresh (last activity within idle window).
    import sqlite3
    import time as time_module

    with sqlite3.connect(hermes_db) as conn:
        conn.execute(
            "UPDATE sessions SET last_activity_at = ? WHERE id = 'hermes-main'",
            (time_module.time() - 10,),
        )
    _isolate_roots(tmp_path, monkeypatch, pi_root=root, hermes_db=hermes_db)

    pending = session_sweeper.discover_pending(idle_seconds=300)

    assert all(host != "hermes" for host, _path in pending)


def test_run_one_hermes_saves_pending_sessions(tmp_path: Path, monkeypatch) -> None:
    state_db = tmp_path / "state.db"
    _write_hermes_db(state_db)
    state_file = tmp_path / "hermes-save-state.json"
    monkeypatch.setattr(
        "chronovisor.raw.hermes_record.DEFAULT_STATE_FILE", state_file
    )
    monkeypatch.setattr(
        "chronovisor.raw.hermes_record.init_chronovisor", lambda **_kwargs: None
    )
    published: list[dict[str, object]] = []

    def fake_publish(**kwargs: object) -> dict[str, object]:
        published.append(kwargs)
        return {"status": "saved"}

    monkeypatch.setattr(
        "chronovisor.raw.hermes_record.publish_transcript_capture", fake_publish
    )

    result = session_sweeper._run_one("hermes", state_db)

    assert result["status"] == "saved"
    assert result["session_count"] == 1
    assert result["session_results"][0]["session_id"] == "hermes-main"
    assert len(published) == 1
    assert published[0]["trigger_ingest"] is True
