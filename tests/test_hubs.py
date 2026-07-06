from __future__ import annotations

from pathlib import Path

from llm_wiki_mcp import hubs


class FakeStore:
    def refresh(self) -> None:
        pass

    def all_pages_meta(self, include_system: bool = False):
        return [
            {"page_id": "a", "title": "A", "updated": "2026-07-01", "path": "/tmp/wiki/pages/ai/a.md", "page_type": "knowledge", "entities": ["Codex"]},
            {"page_id": "b", "title": "B", "updated": "2026-07-01", "path": "/tmp/wiki/pages/ai/b.md", "page_type": "knowledge", "entities": ["Codex"]},
        ]


def test_build_hub_pages_writes_folder_hub(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(hubs, "get_store", lambda: FakeStore())

    payload = hubs.build_hub_pages(output_dir=tmp_path, min_pages=2, max_hubs=3)

    assert payload["hubs"] >= 1
    assert any(path.name.endswith("-hub.md") for path in tmp_path.iterdir())

