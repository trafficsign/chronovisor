from __future__ import annotations

import json
from pathlib import Path

from llm_wiki_mcp import distill


class FakeStore:
    def refresh(self) -> None:
        pass

    def all_pages_meta(self, include_system: bool = False):
        return [{"page_id": "target", "page_type": "knowledge"}]

    def meta(self, page_id: str):
        return {"summary": "Target answer", "recall_questions": ["What is target?"]}


def test_export_distill_dataset_writes_qa_rows(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "wiki-qa.jsonl"
    monkeypatch.setattr(distill, "get_store", lambda: FakeStore())

    payload = distill.export_distill_dataset(output_file=output)

    assert payload["rows"] == 1
    row = json.loads(output.read_text(encoding="utf-8"))
    assert row["instruction"] == "What is target?"
    assert row["output"] == "Target answer"

