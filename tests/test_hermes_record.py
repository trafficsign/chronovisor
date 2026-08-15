from __future__ import annotations

import json
import sqlite3
import sys
import time
from io import StringIO
from pathlib import Path

from chronovisor.core.save_transaction import make_save_transaction
from chronovisor.ingest.raw_semantic_projection import project_parent_raw
from chronovisor.raw import hermes_record
from chronovisor.raw.record_raw import raw_source_label


def _create_state_db(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE sessions (
                id TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                model TEXT,
                cwd TEXT,
                billing_provider TEXT,
                profile_name TEXT
            );
            CREATE TABLE messages (
                id INTEGER PRIMARY KEY,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT,
                tool_call_id TEXT,
                tool_calls TEXT,
                tool_name TEXT,
                timestamp REAL NOT NULL,
                reasoning TEXT,
                active INTEGER NOT NULL DEFAULT 1,
                compacted INTEGER NOT NULL DEFAULT 0
            );
            """
        )
        conn.execute(
            "INSERT INTO sessions VALUES (?, ?, ?, ?, ?, ?)",
            ("session-1", "cli", "gpt-test", "/tmp/project", "provider-test", "default"),
        )
        conn.executemany(
            "INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (10, "session-1", "user", "hello", None, None, None, 1.0, None, 1, 0),
                (12, "session-1", "assistant", "world", None, None, None, 2.0, "private chain", 1, 0),
                (13, "other", "user", "not ours", None, None, None, 3.0, None, 1, 0),
            ],
        )


def _create_sweep_state_db(path: Path) -> None:
    """state.db matching the real Hermes schema for pending-session discovery."""
    now = time.time()
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
                ("cli-old", "cli", "gpt-test", "/tmp/a", None, "default", 2, now - 2000, None, now - 2000),
                ("cli-fresh", "cli", "gpt-test", "/tmp/b", None, "default", 2, now - 10, None, now - 10),
                ("sub-agent", "subagent", "deepseek", "/tmp/c", None, "default", 2, now - 2000, None, now - 2000),
            ],
        )
        conn.executemany(
            "INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (1, "cli-old", "user", "alpha", now - 2000, 1, 0),
                (2, "cli-old", "assistant", "beta", now - 1999, 1, 0),
                (3, "cli-fresh", "user", "gamma", now - 10, 1, 0),
                (4, "cli-fresh", "assistant", "delta", now - 9, 1, 0),
                (5, "sub-agent", "user", "epsilon", now - 2000, 1, 0),
            ],
        )


def test_extract_delta_preserves_hermes_provenance_and_full_rows(tmp_path: Path) -> None:
    state_db = tmp_path / "state.db"
    _create_state_db(state_db)

    delta = hermes_record.extract_transcript_delta(
        state_db, session_id="session-1", after_message_id=0, after_line=0
    )

    assert delta.session.host == "hermes"
    assert delta.session.platform == "cli"
    assert delta.session.provider == "provider-test"
    assert delta.session.model == "gpt-test"
    assert delta.session.cwd == "/tmp/project"
    assert [record.line for record in delta.records] == [1, 2]
    assert [record.message_id for record in delta.records] == [10, 12]
    assert [record.role for record in delta.records] == ["user", "assistant"]
    assert [record.text for record in delta.records] == ["hello", "world"]
    assert delta.records[1].event["message"]["reasoning"] == "private chain"
    assert delta.last_message_id == 12


def test_extract_delta_escapes_sqlite_uri_metacharacters(tmp_path: Path) -> None:
    state_db = tmp_path / "profile?#1" / "state.db"
    state_db.parent.mkdir()
    _create_state_db(state_db)

    delta = hermes_record.extract_transcript_delta(
        state_db, session_id="session-1", after_message_id=0, after_line=0
    )

    assert [record.text for record in delta.records] == ["hello", "world"]


def test_raw_filename_source_label_recognizes_hermes() -> None:
    assert raw_source_label("hermes-session-1") == "hermes"


def test_hook_execution_requires_explicit_enable_env(monkeypatch, capsys) -> None:
    monkeypatch.delenv("HERMES_CHRONOVISOR_RECORD_ENABLED", raising=False)
    monkeypatch.setattr(
        sys,
        "stdin",
        StringIO('{"session_id":"session-1","state_db":"/missing/state.db"}'),
    )

    assert hermes_record.main(["--hook", "--save"]) == 0
    assert json.loads(capsys.readouterr().out) == {
        "status": "disabled",
        "reason": "HERMES_CHRONOVISOR_RECORD_ENABLED=1 is required for hook execution",
    }


def test_build_raw_content_marks_filename_source_and_metadata_as_hermes(tmp_path: Path) -> None:
    state_db = tmp_path / "state.db"
    _create_state_db(state_db)
    delta = hermes_record.extract_transcript_delta(
        state_db,
        session_id="session-1",
        after_message_id=0,
        after_line=0,
        capture_failed=True,
    )
    transaction = make_save_transaction(
        host="hermes",
        session_file=state_db,
        session_id="session-1",
        after_line=0,
        until_line=2,
    )

    content = hermes_record.build_raw_content(delta, transaction=transaction)

    assert "# Hermes Session Transcript Delta" in content
    assert "- Source: Hermes" in content
    assert "- Platform: cli" in content
    assert "- Provider: provider-test" in content
    assert "- Model: gpt-test" in content
    assert "- Capture Failed: true" in content
    payload = json.loads(content.split("```json\n", 1)[1].split("\n```", 1)[0])
    assert payload[0]["event"]["host"] == "hermes"
    assert payload[0]["event"]["session"]["provider"] == "provider-test"
    assert payload[0]["event"]["session"]["capture_failed"] is True
    assert transaction.idempotency_key.startswith("hermes-")


def test_extract_delta_uses_message_id_cursor_but_contiguous_capture_lines(tmp_path: Path) -> None:
    state_db = tmp_path / "state.db"
    _create_state_db(state_db)

    delta = hermes_record.extract_transcript_delta(
        state_db, session_id="session-1", after_message_id=10, after_line=7
    )

    assert [record.message_id for record in delta.records] == [12]
    assert [record.line for record in delta.records] == [8]
    assert delta.scanned_until_line == 8


def test_save_session_publishes_once_and_advances_cursor(tmp_path: Path, monkeypatch) -> None:
    state_db = tmp_path / "state.db"
    state_file = tmp_path / "hermes-save-state.json"
    _create_state_db(state_db)
    published: list[dict[str, object]] = []

    def fake_publish(**kwargs: object) -> dict[str, object]:
        published.append(kwargs)
        return {"status": "saved", "raw_path": "/tmp/raw"}

    monkeypatch.setattr(hermes_record, "publish_transcript_capture", fake_publish)
    monkeypatch.setattr(hermes_record, "init_chronovisor", lambda **_kwargs: None)

    first = hermes_record.save_session(
        state_db=state_db,
        session_id="session-1",
        state_file=state_file,
        raw_dir=tmp_path / "raw",
    )
    second = hermes_record.save_session(
        state_db=state_db,
        session_id="session-1",
        state_file=state_file,
        raw_dir=tmp_path / "raw",
    )

    assert first["status"] == "saved"
    assert first["host"] == "hermes"
    assert second == {"status": "skipped", "reason": "no_messages", "host": "hermes"}
    assert len(published) == 1
    assert published[0]["host"] == "hermes"
    assert published[0]["legacy_session_id"] == "hermes-session-1"
    assert str(published[0]["idempotency_key"]).startswith("hermes-")
    state = json.loads(state_file.read_text())
    entry = next(iter(state["sessions"].values()))
    assert entry["last_saved_message_id"] == 12
    assert entry["last_saved_line"] == 2


def test_save_cursors_are_isolated_by_profile_state_db(tmp_path: Path, monkeypatch) -> None:
    first_db = tmp_path / "profile-a" / "state.db"
    second_db = tmp_path / "profile-b" / "state.db"
    first_db.parent.mkdir()
    second_db.parent.mkdir()
    _create_state_db(first_db)
    _create_state_db(second_db)
    state_file = tmp_path / "hermes-save-state.json"
    published: list[dict[str, object]] = []

    monkeypatch.setattr(
        hermes_record,
        "publish_transcript_capture",
        lambda **kwargs: published.append(kwargs) or {"status": "saved"},
    )
    monkeypatch.setattr(hermes_record, "init_chronovisor", lambda **_kwargs: None)

    for state_db in (first_db, second_db):
        hermes_record.save_session(
            state_db=state_db,
            session_id="session-1",
            state_file=state_file,
            raw_dir=tmp_path / "raw",
        )

    state = json.loads(state_file.read_text())
    assert len(published) == 2
    assert len(state["sessions"]) == 2


def test_hermes_raw_projects_into_semantic_children(tmp_path: Path) -> None:
    state_db = tmp_path / "state.db"
    _create_state_db(state_db)
    delta = hermes_record.extract_transcript_delta(
        state_db, session_id="session-1", after_message_id=0, after_line=0
    )
    transaction = make_save_transaction(
        host="hermes",
        session_file=state_db,
        session_id="session-1",
        after_line=0,
        until_line=2,
    )
    raw_path = tmp_path / f"save-{transaction.idempotency_key}.md"
    raw_path.write_text(
        hermes_record.build_raw_content(delta, transaction=transaction), encoding="utf-8"
    )

    projection = project_parent_raw(
        raw_path, output_dir=tmp_path / "projection", max_child_bytes=4_000
    )

    assert projection.role_counts == {"assistant": 1, "user": 1}
    projected = "\n".join(path.read_text() for path in projection.child_paths)
    assert "hello" in projected
    assert "world" in projected
    assert '"host":"hermes"' in raw_path.read_text().replace(" ", "")


def test_pending_session_ids_excludes_subagent_and_fresh_sessions(
    tmp_path: Path,
) -> None:
    state_db = tmp_path / "state.db"
    _create_sweep_state_db(state_db)
    state_file = tmp_path / "hermes-save-state.json"

    pending = hermes_record.pending_session_ids(
        state_db, state_file=state_file, idle_seconds=300
    )

    # cli-old is idle and has messages; cli-fresh is too recent; sub-agent is
    # filtered by source even though it is idle.
    assert pending == ["cli-old"]


def test_pending_session_ids_include_subagent_when_requested(tmp_path: Path) -> None:
    state_db = tmp_path / "state.db"
    _create_sweep_state_db(state_db)
    state_file = tmp_path / "hermes-save-state.json"

    pending = hermes_record.pending_session_ids(
        state_db,
        state_file=state_file,
        idle_seconds=300,
        include_subagent=True,
    )

    assert pending == ["cli-old", "sub-agent"]


def test_pending_session_ids_advances_cursor(tmp_path: Path, monkeypatch) -> None:
    state_db = tmp_path / "state.db"
    _create_sweep_state_db(state_db)
    state_file = tmp_path / "hermes-save-state.json"
    published: list[dict[str, object]] = []

    monkeypatch.setattr(
        hermes_record,
        "publish_transcript_capture",
        lambda **kwargs: published.append(kwargs) or {"status": "saved"},
    )
    monkeypatch.setattr(hermes_record, "init_chronovisor", lambda **_kwargs: None)

    hermes_record.save_session(
        state_db=state_db,
        session_id="cli-old",
        state_file=state_file,
        raw_dir=tmp_path / "raw",
    )

    pending = hermes_record.pending_session_ids(
        state_db, state_file=state_file, idle_seconds=300
    )
    assert pending == []


def test_save_pending_sessions_captures_all_and_reports_errors(
    tmp_path: Path, monkeypatch
) -> None:
    state_db = tmp_path / "state.db"
    _create_sweep_state_db(state_db)
    state_file = tmp_path / "hermes-save-state.json"
    published: list[dict[str, object]] = []

    def fake_publish(**kwargs: object) -> dict[str, object]:
        published.append(kwargs)
        return {"status": "saved"}

    monkeypatch.setattr(hermes_record, "publish_transcript_capture", fake_publish)
    monkeypatch.setattr(hermes_record, "init_chronovisor", lambda **_kwargs: None)

    result = hermes_record.save_pending_sessions(
        state_db=state_db,
        state_file=state_file,
        raw_dir=tmp_path / "raw",
        idle_seconds=300,
    )

    assert result["status"] == "saved"
    assert result["host"] == "hermes"
    assert [r["session_id"] for r in result["session_results"]] == ["cli-old"]
    assert result["session_results"][0]["status"] == "saved"
    assert len(published) == 1
    assert published[0]["host"] == "hermes"
    # A second sweep has nothing left to publish.
    again = hermes_record.save_pending_sessions(
        state_db=state_db,
        state_file=state_file,
        raw_dir=tmp_path / "raw",
        idle_seconds=300,
    )
    assert again["status"] == "skipped"
    assert again["reason"] == "no_pending_sessions"


def test_save_pending_sessions_tolerates_session_failures(
    tmp_path: Path, monkeypatch
) -> None:
    state_db = tmp_path / "state.db"
    _create_sweep_state_db(state_db)
    state_file = tmp_path / "hermes-save-state.json"

    def boom(**kwargs: object) -> dict[str, object]:
        raise RuntimeError("db locked")

    monkeypatch.setattr(hermes_record, "publish_transcript_capture", boom)
    monkeypatch.setattr(hermes_record, "init_chronovisor", lambda **_kwargs: None)

    result = hermes_record.save_pending_sessions(
        state_db=state_db,
        state_file=state_file,
        raw_dir=tmp_path / "raw",
        idle_seconds=300,
    )

    assert result["status"] == "error"
    assert result["session_results"][0]["status"] == "error"
    assert "RuntimeError" in result["session_results"][0]["error"]


def test_hermes_state_db_defaults_to_home(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("HERMES_HOME", raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))

    assert hermes_record.hermes_state_db() == tmp_path / ".hermes" / "state.db"
