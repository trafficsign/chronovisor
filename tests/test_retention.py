from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from llm_wiki_mcp import retention


class FakeStore:
    def refresh(self) -> None:
        pass

    def all_pages_meta(self, include_system: bool = False):
        return [
            {"page_id": "used", "page_type": "knowledge", "updated": "2026-07-01"},
            {"page_id": "ref", "page_type": "reference", "updated": "2026-07-01"},
        ]


def test_build_retention_scores_strengthens_used_pages(tmp_path: Path, monkeypatch) -> None:
    feedback = tmp_path / "feedback.jsonl"
    feedback.write_text(
        json.dumps({"kind": "injection_used", "expected_pages": ["used"]}) + "\n",
        encoding="utf-8",
    )
    recall_log = tmp_path / "recall.jsonl"
    recall_log.write_text("", encoding="utf-8")
    output = tmp_path / "retention.json"
    monkeypatch.setattr(retention, "get_store", lambda: FakeStore())

    payload = retention.build_retention_scores(
        feedback_file=feedback,
        recall_log_file=recall_log,
        output_file=output,
        today=date(2026, 7, 6),
    )

    assert payload["pages"]["used"]["score"] > 0
    assert payload["pages"]["ref"]["score"] == 0.0
    assert retention.retention_score("used", path=output) > 0

