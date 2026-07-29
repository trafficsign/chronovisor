from __future__ import annotations

import json
from pathlib import Path

from chronovisor.ops import golden_expand


class FakeStore:
    def refresh(self) -> None:
        pass

    def all_pages_meta(self, include_system: bool = False):
        return [{"page_id": "target", "page_type": "knowledge"}]

    def meta(self, page_id: str):
        return {"recall_questions": ["Where is the target?"]}


def test_expand_golden_from_recall_questions_appends_new_rows(tmp_path: Path, monkeypatch) -> None:
    golden = tmp_path / "search-golden.jsonl"
    monkeypatch.setattr(golden_expand, "get_store", lambda: FakeStore())

    payload = golden_expand.expand_golden_from_recall_questions(golden_file=golden)

    assert payload["added"] == 1
    row = json.loads(golden.read_text(encoding="utf-8"))
    assert row["query"] == "Where is the target?"
    assert row["expected_pages"] == ["target"]

