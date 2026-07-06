from __future__ import annotations

import json

from llm_wiki_mcp import lint, server


def test_wiki_check_returns_compact_limited_issues(monkeypatch) -> None:
    issues = [
        {
            "type": "tag_missing",
            "severity": "high",
            "page": f"p{i}",
            "detail": "x" * 400,
            "auto_fixable": False,
        }
        for i in range(50)
    ]
    monkeypatch.setattr(lint, "check", lambda: issues)

    tool_fn = server.wiki_check.fn if hasattr(server.wiki_check, "fn") else server.wiki_check
    payload = json.loads(tool_fn())

    assert payload["total_issues"] == 50
    assert len(payload["issues"]) == 40
    assert payload["omitted_issues"] == 10
    assert len(payload["issues"][0]["detail"]) <= 180
    assert payload["issues"][0]["lane"] == "heavy_model_batch"
