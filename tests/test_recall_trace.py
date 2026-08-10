from __future__ import annotations

import json
from pathlib import Path

from chronovisor.hosts import server
from chronovisor.recall import recall_runtime


def _tool(function):
    return function.fn if hasattr(function, "fn") else function


def test_chronovisor_recall_used_records_only_explicit_used_pages(monkeypatch) -> None:
    recorded: list[dict] = []
    monkeypatch.setattr(
        server,
        "_validate_used_recall_decision",
        lambda _decision, session: {
            "status": "ok",
            "session_id": session,
            "observable_page_ids": ["page-a", "page-b"],
            "processor_shadow_page_ids": ["page-a"],
        },
    )
    monkeypatch.setattr(server, "_existing_used_receipt", lambda *_args: None)
    monkeypatch.setattr(
        server,
        "_append_pull_log",
        lambda record: recorded.append(record) is None,
    )
    tool = _tool(server.chronovisor_recall_used)

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
    assert result["learning_join"] == "ready"
    assert result["processor_shadow_covered_page_ids"] == ["page-a"]
    assert result["new_page_ids"] == ["page-a", "page-b"]
    assert recorded[0] == {
        "type": "used",
        "stage": "used",
        "event_id": result["event_id"],
        "session_id": "session-1",
        "decision_id": "decision-1",
        "page_ids": ["page-a", "page-b"],
        "note": "materially used",
    }


def test_chronovisor_recall_used_fails_when_durable_append_fails(monkeypatch) -> None:
    monkeypatch.setattr(
        server,
        "_validate_used_recall_decision",
        lambda _decision, session: {
            "status": "ok",
            "session_id": session,
            "observable_page_ids": ["page-a"],
            "processor_shadow_page_ids": [],
        },
    )
    monkeypatch.setattr(server, "_existing_used_receipt", lambda *_args: None)
    monkeypatch.setattr(server, "_append_pull_log", lambda _record: False)

    result = json.loads(
        _tool(server.chronovisor_recall_used)(
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


def test_chronovisor_recall_used_rejects_unjoinable_decision(monkeypatch) -> None:
    monkeypatch.setattr(
        server,
        "_validate_used_recall_decision",
        lambda *_args: {"status": "error", "error": "unknown recall decision"},
    )
    recorded: list[dict] = []
    monkeypatch.setattr(server, "_append_pull_log", recorded.append)

    result = json.loads(
        _tool(server.chronovisor_recall_used)(
            decision_id="missing",
            session_id="session-1",
            page_ids=["page-a"],
        )
    )

    assert result == {
        "status": "error",
        "error": "unknown recall decision",
        "decision_id": "missing",
    }
    assert recorded == []


def test_chronovisor_recall_used_rejects_unobserved_pages(monkeypatch) -> None:
    monkeypatch.setattr(
        server,
        "_validate_used_recall_decision",
        lambda *_args: {
            "status": "ok",
            "session_id": "session-1",
            "observable_page_ids": ["page-a"],
            "processor_shadow_page_ids": [],
        },
    )
    recorded: list[dict] = []
    monkeypatch.setattr(server, "_append_pull_log", recorded.append)

    result = json.loads(
        _tool(server.chronovisor_recall_used)(
            decision_id="decision-1",
            session_id="session-1",
            page_ids=["page-b"],
        )
    )

    assert result == {
        "status": "error",
        "error": "used pages were not returned, injected, or read",
        "decision_id": "decision-1",
        "page_ids": ["page-b"],
    }
    assert recorded == []


def test_chronovisor_recall_used_is_idempotent_per_decision(monkeypatch) -> None:
    monkeypatch.setattr(
        server,
        "_validate_used_recall_decision",
        lambda _decision, _session: {
            "status": "ok",
            "session_id": "canonical-session",
            "observable_page_ids": ["page-a"],
            "processor_shadow_page_ids": ["page-a"],
        },
    )
    monkeypatch.setattr(
        server,
        "_existing_used_receipt",
        lambda *_args: {
            "event_id": "existing-event",
            "page_ids": ["page-a"],
        },
    )
    recorded: list[dict] = []
    monkeypatch.setattr(server, "_append_pull_log", recorded.append)

    result = json.loads(
        _tool(server.chronovisor_recall_used)(
            decision_id="decision-1",
            page_ids=["page-a"],
        )
    )

    assert result == {
        "status": "already_recorded",
        "event_id": "existing-event",
        "decision_id": "decision-1",
        "page_ids": ["page-a"],
        "learning_join": "ready",
    }
    assert recorded == []


def test_chronovisor_recall_used_appends_only_new_pages(monkeypatch) -> None:
    monkeypatch.setattr(
        server,
        "_validate_used_recall_decision",
        lambda _decision, _session: {
            "status": "ok",
            "session_id": "session-1",
            "observable_page_ids": ["page-a", "page-b"],
            "processor_shadow_page_ids": ["page-a", "page-b"],
        },
    )
    monkeypatch.setattr(
        server,
        "_existing_used_receipt",
        lambda *_args: {
            "event_id": "existing-event",
            "page_ids": ["page-a"],
        },
    )
    recorded: list[dict] = []
    monkeypatch.setattr(
        server,
        "_append_pull_log",
        lambda record: recorded.append(record) is None,
    )

    result = json.loads(
        _tool(server.chronovisor_recall_used)(
            decision_id="decision-1",
            session_id="session-1",
            page_ids=["page-a", "page-b"],
        )
    )

    assert result["status"] == "recorded"
    assert result["page_ids"] == ["page-a", "page-b"]
    assert result["new_page_ids"] == ["page-b"]
    assert recorded[0]["page_ids"] == ["page-b"]


def test_used_decision_validation_fills_session_and_reports_processor_overlap(
    monkeypatch,
    tmp_path: Path,
) -> None:
    recall_log = tmp_path / "recall.jsonl"
    recall_log.write_text(
        json.dumps(
            {
                "decision_id": "decision-1",
                "session_id": "session-1",
                "pages": ["page-a"],
                "evidence_features": {
                    "processor_shadow": {"committed_page_ids": ["page-a", "page-b"]}
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(recall_runtime, "RECALL_LOG_FILE", recall_log)
    pull_log = tmp_path / "pull.jsonl"
    pull_log.write_text("", encoding="utf-8")
    monkeypatch.setattr(recall_runtime, "RECALL_PULL_LOG_FILE", pull_log)

    validation = server._validate_used_recall_decision("decision-1", "")

    assert validation == {
        "status": "ok",
        "session_id": "session-1",
        "observable_page_ids": ["page-a"],
        "processor_shadow_page_ids": ["page-a", "page-b"],
    }
    assert server._validate_used_recall_decision("decision-1", "wrong") == {
        "status": "error",
        "error": "recall session mismatch",
    }


def test_chronovisor_read_forwards_turn_trace_without_marking_page_used(
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
    monkeypatch.setattr(server, "_find_page_with_alias", lambda _page: page)
    monkeypatch.setattr(server, "_append_pull_log", recorded.append)

    result = json.loads(
        _tool(server.chronovisor_read)(
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


def test_mcp_client_host_and_read_field_attribution(
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

    class ClientInfo:
        name = "Claude Code"

    class ClientParams:
        clientInfo = ClientInfo()

    class Session:
        client_params = ClientParams()

    class Context:
        client_id = ""
        session = Session()

    monkeypatch.setattr(server, "get_store", FakeStore)
    monkeypatch.setattr(server, "_find_page_with_alias", lambda _page: page)
    monkeypatch.setattr(server, "_append_pull_log", recorded.append)
    monkeypatch.setattr(
        server,
        "_record_mcp_field_activity",
        lambda **_kwargs: {
            "status": "ok",
            "host": "claude-code",
            "session_hash": "0123456789abcdef",
        },
    )

    result = json.loads(_tool(server.chronovisor_read)("page-a", ctx=Context()))

    assert result["page_id"] == "page-a"
    assert server._mcp_client_host(Context()) == "claude-code"
    assert recorded[0]["host"] == "claude-code"
    assert recorded[0]["field_session_hash"] == "0123456789abcdef"
