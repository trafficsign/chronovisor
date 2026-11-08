"""Regression tests for host transcript session resolution."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from chronovisor.core import claude_code_transcript, codex_transcript
from chronovisor.core.transcript import ClaudeCodeSaveError, CodexSaveError
from chronovisor.raw import claude_code_record, codex_record


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def _write_codex_session(path: Path, *, session_id: str, cwd: str) -> None:
    _write_jsonl(
        path,
        [
            {
                "type": "session_meta",
                "payload": {"id": session_id, "cwd": cwd},
            }
        ],
    )


def _write_claude_session(
    path: Path, *, session_id: str, cwd: str = "/tmp/project"
) -> None:
    _write_jsonl(
        path,
        [
            {
                "type": "user",
                "sessionId": session_id,
                "cwd": cwd,
                "message": {"content": "hello"},
            }
        ],
    )


def _codex_args(
    *,
    session_id: str | None = None,
    cwd: str | None = None,
    session_file: str | None = None,
    sessions_root: Path | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        session_id=session_id,
        cwd=cwd,
        session_file=session_file,
        sessions_root=str(sessions_root) if sessions_root is not None else None,
    )


def test_codex_missing_session_id_does_not_fall_back_to_newest(tmp_path: Path) -> None:
    root = tmp_path / "sessions"
    root.mkdir()
    _write_codex_session(root / "other.jsonl", session_id="other", cwd="/tmp/other")

    with pytest.raises(CodexSaveError, match="session_id"):
        codex_transcript.find_session_file(session_id="missing", sessions_root=root)


def test_codex_missing_cwd_does_not_fall_back_to_newest(tmp_path: Path) -> None:
    root = tmp_path / "sessions"
    root.mkdir()
    _write_codex_session(root / "other.jsonl", session_id="other", cwd="/tmp/other")

    with pytest.raises(CodexSaveError, match="cwd"):
        codex_transcript.find_session_file(cwd="/tmp/missing", sessions_root=root)


def test_codex_conflicting_identity_inputs_do_not_select_either_session(
    tmp_path: Path,
) -> None:
    root = tmp_path / "sessions"
    root.mkdir()
    _write_codex_session(root / "first.jsonl", session_id="first", cwd="/tmp/first")
    _write_codex_session(root / "second.jsonl", session_id="second", cwd="/tmp/second")

    with pytest.raises(CodexSaveError, match="session_id.*cwd|cwd.*session_id"):
        codex_transcript.find_session_file(
            session_id="first",
            cwd="/tmp/second",
            sessions_root=root,
        )


def test_codex_session_id_does_not_fall_through_to_implicit_pwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "sessions"
    root.mkdir()
    _write_codex_session(root / "other.jsonl", session_id="other", cwd="/tmp/project")
    monkeypatch.setenv("PWD", "/tmp/project")

    with pytest.raises(CodexSaveError, match="session_id"):
        codex_record.resolve_session_file(
            _codex_args(session_id="missing", sessions_root=root), {}
        )


def test_codex_empty_session_id_is_still_an_explicit_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "sessions"
    root.mkdir()
    _write_codex_session(root / "other.jsonl", session_id="other", cwd="/tmp/project")
    monkeypatch.setenv("PWD", "/tmp/project")

    with pytest.raises(CodexSaveError, match="session_id"):
        codex_record.resolve_session_file(
            _codex_args(session_id="", sessions_root=root), {}
        )


def test_codex_explicit_session_file_must_exist(tmp_path: Path) -> None:
    missing = tmp_path / "missing.jsonl"

    with pytest.raises(CodexSaveError, match="session file"):
        codex_record.resolve_session_file(
            _codex_args(session_file=str(missing)),
            {},
        )


def test_codex_explicit_session_file_must_match_session_id(tmp_path: Path) -> None:
    session = tmp_path / "session.jsonl"
    _write_codex_session(session, session_id="actual", cwd="/tmp/project")

    with pytest.raises(CodexSaveError, match="session_id"):
        codex_record.resolve_session_file(
            _codex_args(session_file=str(session), session_id="other"),
            {},
        )


def test_claude_missing_transcript_path_does_not_fall_back_to_newest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "projects"
    root.mkdir()
    _write_claude_session(root / "other.jsonl", session_id="other")
    monkeypatch.setattr(
        claude_code_transcript, "claude_code_projects_root", lambda: root
    )

    with pytest.raises(ClaudeCodeSaveError, match="transcript_path"):
        claude_code_transcript.find_session_file(
            transcript_path=str(tmp_path / "missing.jsonl")
        )


def test_claude_explicit_transcript_path_must_match_session_id(
    tmp_path: Path,
) -> None:
    session = tmp_path / "session.jsonl"
    _write_claude_session(session, session_id="actual")

    with pytest.raises(ClaudeCodeSaveError, match="session_id"):
        claude_code_transcript.find_session_file(
            transcript_path=str(session), session_id="other"
        )


def test_claude_missing_session_id_does_not_fall_back_to_newest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "projects"
    root.mkdir()
    _write_claude_session(root / "other.jsonl", session_id="other")
    monkeypatch.setattr(
        claude_code_transcript, "claude_code_projects_root", lambda: root
    )

    with pytest.raises(ClaudeCodeSaveError, match="session_id"):
        claude_code_transcript.find_session_file(session_id="missing")


@pytest.mark.parametrize("session_id", ["", "other"])
def test_claude_identity_is_not_a_filename_substring(tmp_path, monkeypatch, session_id):
    _write_claude_session(tmp_path / "other-session.jsonl", session_id="other-session")
    monkeypatch.setattr(
        claude_code_transcript, "claude_code_projects_root", lambda: tmp_path
    )
    with pytest.raises(ClaudeCodeSaveError, match="session_id"):
        claude_code_transcript.find_session_file(session_id=session_id)


def test_claude_empty_session_id_is_still_an_explicit_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_claude_session(tmp_path / "other.jsonl", session_id="other")
    monkeypatch.setattr(
        claude_code_transcript, "claude_code_projects_root", lambda: tmp_path
    )

    with pytest.raises(ClaudeCodeSaveError, match="session_id"):
        claude_code_record._transaction_spec().resolve_session_file(
            SimpleNamespace(session_id="", session_file=None), {}
        )
