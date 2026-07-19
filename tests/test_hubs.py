from __future__ import annotations

from pathlib import Path

from chronovisor import hubs


class FakeStore:
    def refresh(self) -> None:
        pass

    def all_pages_meta(self, include_system: bool = False):
        return [
            {"page_id": "a", "title": "A", "updated": "2026-07-01", "path": "/tmp/wiki/pages/ai/a.md", "page_type": "knowledge", "entities": ["Codex"]},
            {"page_id": "b", "title": "B", "updated": "2026-07-01", "path": "/tmp/wiki/pages/ai/b.md", "page_type": "knowledge", "entities": ["Codex"]},
        ]


class ExistingHubStore(FakeStore):
    def __init__(self, existing_hub: Path) -> None:
        self.existing_hub = existing_hub

    def meta(self, page_id: str):
        if page_id == "folder-ai-hub":
            return {"page_id": page_id, "path": str(self.existing_hub)}
        return None


def test_build_hub_pages_writes_folder_hub(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(hubs, "get_store", lambda: FakeStore())

    payload = hubs.build_hub_pages(output_dir=tmp_path, min_pages=2, max_hubs=3)

    assert payload["hubs"] >= 1
    assert payload["mutation"]["status"] == "applied"
    assert any(path.name.endswith("-hub.md") for path in tmp_path.iterdir())


def test_build_hub_pages_updates_existing_page_id_without_relocation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    pages_dir = tmp_path / "pages"
    legacy_dir = pages_dir / "chronovisor"
    output_dir = pages_dir / "hubs"
    legacy_dir.mkdir(parents=True)
    output_dir.mkdir()
    existing = legacy_dir / "folder-ai-hub.md"
    existing.write_text("old hub", encoding="utf-8")
    monkeypatch.setattr(hubs, "PAGES_DIR", pages_dir)
    monkeypatch.setattr(hubs, "get_store", lambda: ExistingHubStore(existing))

    payload = hubs.build_hub_pages(
        output_dir=output_dir,
        min_pages=2,
        max_hubs=1,
    )

    assert payload["paths"] == [str(existing)]
    assert "Auto-maintained folder hub" in existing.read_text(encoding="utf-8")
    assert not (output_dir / "folder-ai-hub.md").exists()
