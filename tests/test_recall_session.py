from __future__ import annotations

import json
import os
from pathlib import Path

from chronovisor.recall import recall_session


def test_session_path_sanitizes_host_input_and_bounds_filename(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(recall_session, "SESSIONS_DIR", tmp_path)

    path = recall_session.session_path(" ../Codex session/" + "x" * 200)

    assert path.parent == tmp_path
    assert path.suffix == ".json"
    assert len(path.stem) <= 96
    assert "/" not in path.name
    assert ".." not in path.name


def test_corrupt_session_fails_open_without_rewriting_source(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(recall_session, "SESSIONS_DIR", tmp_path)
    path = recall_session.session_path("session-a")
    path.write_text("{broken", encoding="utf-8")

    state = recall_session.load_session_state("session-a")

    assert state is not None
    assert state.session_id == "session-a"
    assert state.recent_queries == []
    assert path.read_text(encoding="utf-8") == "{broken"


def test_update_session_persists_bounded_unique_state(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(recall_session, "SESSIONS_DIR", tmp_path)
    monkeypatch.setattr(recall_session.time, "time", lambda: 1_234.5)
    state = recall_session.RecallSessionState(session_id="session-a")

    recall_session.update_session_after_recall(
        state,
        queries=["  LLM   Wiki recall  ", "Chronovisor recall", "検索精度 改善"],
        page_ids=["page-a", "", "page-b"],
        page_updated={"page-a": "2026-07-17"},
    )

    payload = json.loads(recall_session.session_path("session-a").read_text())
    assert payload["version"] == 1
    assert payload["recent_queries"] == ["Chronovisor recall", "検索精度 改善"]
    assert payload["last_seen"] == 1_234.5
    assert payload["injected_pages"] == {
        "page-a": {"last_injected_at": 1_234.5, "updated": "2026-07-17"},
        "page-b": {"last_injected_at": 1_234.5, "updated": ""},
    }
    assert recall_session.should_skip_page(state, "page-a", "2026-07-17") is True
    assert recall_session.should_skip_page(state, "page-a", "newer") is False


def test_load_session_repairs_host_transport_query_and_topic_pollution(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(recall_session, "SESSIONS_DIR", tmp_path)
    path = recall_session.session_path("session-a")
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "session_id": "session-a",
                "recent_queries": [
                    '<in-app-browser-context source="ambient-ui-state"> truncated',
                    "AI 2040 Plan D",
                    "<heartbeat> <automation_id>chronovisor</automation_id>",
                ],
                "recent_topics": [
                    "in-app-browser-context",
                    "ambient-ui-state",
                    "automation_id",
                    "plan",
                ],
            }
        ),
        encoding="utf-8",
    )

    state = recall_session.load_session_state("session-a")

    assert state is not None
    assert state.recent_queries == ["AI 2040 Plan D"]
    assert "in-app-browser-context" not in state.recent_topics
    assert "ambient-ui-state" not in state.recent_topics
    assert "automation_id" not in state.recent_topics
    assert "plan" in state.recent_topics


def test_update_session_rejects_new_transport_only_queries(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(recall_session, "SESSIONS_DIR", tmp_path)
    state = recall_session.RecallSessionState(
        session_id="session-a",
        recent_queries=[
            '<in-app-browser-context source="ambient-ui-state"> truncated'
        ],
        recent_topics=["ambient-ui-state"],
    )

    recall_session.update_session_after_recall(
        state,
        queries=[
            "<heartbeat> <automation_id>chronovisor</automation_id>",
            "Chronovisor recall",
        ],
        page_ids=[],
    )

    assert state.recent_queries == ["Chronovisor recall"]
    assert "ambient-ui-state" not in state.recent_topics
    assert "automation_id" not in state.recent_topics


def test_cleanup_sessions_uses_mtime_for_corrupt_state(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(recall_session, "SESSIONS_DIR", tmp_path)
    monkeypatch.setattr(recall_session.time, "time", lambda: 10_000.0)
    old = tmp_path / "old.json"
    fresh = tmp_path / "fresh.json"
    old.write_text("{broken", encoding="utf-8")
    fresh.write_text("{broken", encoding="utf-8")
    os.utime(old, (1_000, 1_000))
    os.utime(fresh, (9_990, 9_990))

    removed = recall_session.cleanup_sessions(ttl_seconds=100)

    assert removed == 1
    assert not old.exists()
    assert fresh.exists()
