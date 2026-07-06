from __future__ import annotations

from pathlib import Path

from llm_wiki_mcp import raw_replay


def test_build_queue_selects_raws_by_date(tmp_path: Path, monkeypatch) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "20260701-codex-a.md").write_text("a", encoding="utf-8")
    (raw_dir / "20260706-codex-b.md").write_text("bb", encoding="utf-8")
    queue = tmp_path / "queue.jsonl"
    monkeypatch.setattr(raw_replay, "RAW_DIR", raw_dir)

    payload = raw_replay.build_queue(since="2026-07-05", path=queue)

    assert payload["count"] == 1
    text = queue.read_text(encoding="utf-8")
    assert "20260706-codex-b.md" in text
    assert "20260701-codex-a.md" not in text
