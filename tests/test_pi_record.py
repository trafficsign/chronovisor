"""Focused parity coverage for the Pi raw recorder."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from chronovisor.hosts import pi_record
from chronovisor.raw import record_transaction


def _write_pi_session(path: Path) -> None:
    rows = [
        {
            "type": "session",
            "id": "pi-test",
            "cwd": "/tmp/project",
            "timestamp": "2026-08-13T00:00:00Z",
        },
        {
            "type": "message",
            "timestamp": "2026-08-13T00:00:01Z",
            "message": {"role": "user", "content": "Pi の保存を確認したい"},
        },
        {
            "type": "message",
            "timestamp": "2026-08-13T00:00:02Z",
            "message": {"role": "assistant", "content": "保存しました"},
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _args_for(session: Path, state: Path) -> SimpleNamespace:
    return SimpleNamespace(
        session_id=None,
        session_file=str(session),
        state_file=str(state),
        max_chars=pi_record.DEFAULT_MAX_CHARS,
        dry_run=False,
        save=True,
        extract_only=False,
        ignore_state=True,
        hook=False,
        trigger_ingest=False,
    )


def test_pi_v2_save_preserves_source_bytes_and_cursor(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "wiki"
    root.mkdir()
    for name in ("index.md", "log.md", "schema.md"):
        (root / name).write_text("legacy\n", encoding="utf-8")
    monkeypatch.setattr(pi_record, "CHRONOVISOR_ROOT", root)

    session = tmp_path / "session.jsonl"
    state = tmp_path / "state.json"
    raw_dir = tmp_path / "wiki" / "raw"
    _write_pi_session(session)
    monkeypatch.setenv("CHRONOVISOR_RAW_LAYOUT", "v2")
    monkeypatch.setattr(record_transaction, "RAW_DIR", raw_dir)
    monkeypatch.setattr(record_transaction, "init_chronovisor", lambda: None)

    args = _args_for(session, state)
    result = pi_record.run(args)

    assert result["status"] == "saved"
    assert Path(result["save_result"]["path"]).read_bytes() == session.read_bytes()
    assert json.loads(state.read_text())["files"][str(session)]["last_saved_line"] == 3

    args.ignore_state = False
    second = pi_record.run(args)
    assert second["status"] == "skipped"
    assert second["record_count"] == 0
