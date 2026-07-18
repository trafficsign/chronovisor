from __future__ import annotations

import json
from pathlib import Path

from llm_wiki_mcp import server


def _tool(function):
    return function.fn if hasattr(function, "fn") else function


def test_wiki_recall_used_records_only_explicit_used_pages(monkeypatch) -> None:
    recorded: list[dict] = []
    monkeypatch.setattr(
        server,
        "_append_pull_log",
        lambda record: recorded.append(record) is None,
    )
    tool = _tool(server.wiki_recall_used)

    result = json.loads(
        tool(
            decision_id="decision-1",
            session_id="session-1",
            page_ids=["page-a", "page-a", "page-b"],
            note="materially used",
        )
    )

    assert result["status"] == "recorded"
    assert result["event_id"] == recorded[0]["event_id"]
    assert len(result["event_id"]) == 32
    assert recorded[0] == {
        "type": "used",
        "stage": "used",
        "event_id": result["event_id"],
        "session_id": "session-1",
        "decision_id": "decision-1",
        "page_ids": ["page-a", "page-b"],
        "note": "materially used",
    }


def test_wiki_recall_used_fails_when_durable_append_fails(monkeypatch) -> None:
    monkeypatch.setattr(server, "_append_pull_log", lambda _record: False)

    result = json.loads(
        _tool(server.wiki_recall_used)(
            decision_id="decision-1",
            session_id="session-1",
            page_ids=["page-a"],
        )
    )

    assert result == {
        "status": "error",
        "error": "used receipt was not durably recorded",
        "decision_id": "decision-1",
    }


def test_wiki_read_forwards_turn_trace_without_marking_page_used(
    monkeypatch,
    tmp_path: Path,
) -> None:
    page = tmp_path / "page-a.md"
    page.write_text("# Page A\n\nEvidence only.", encoding="utf-8")
    recorded: list[dict] = []

    class FakeStore:
        def refresh(self) -> None:
            return None

        def outlinks(self, _page: str) -> list[str]:
            return []

        def backlinks(self, _page: str) -> list[str]:
            return []

    monkeypatch.setattr(server, "get_store", FakeStore)
    monkeypatch.setattr(server, "find_page", lambda _page: page)
    monkeypatch.setattr(server, "_append_pull_log", recorded.append)

    result = json.loads(
        _tool(server.wiki_read)(
            "page-a", session_id="session-1", decision_id="decision-1"
        )
    )

    assert result["page_id"] == "page-a"
    assert recorded == [
        {
            "type": "read",
            "stage": "read",
            "session_id": "session-1",
            "decision_id": "decision-1",
            "page_id": "page-a",
        }
    ]
