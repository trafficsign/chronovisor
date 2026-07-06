from __future__ import annotations

import json
from pathlib import Path

from llm_wiki_mcp import health


def test_capture_kpi_counts_raw_claim_coverage(tmp_path: Path, monkeypatch) -> None:
    wiki_root = tmp_path / "wiki"
    raw_dir = wiki_root / "raw"
    claims_dir = wiki_root / "claims"
    raw_dir.mkdir(parents=True)
    claims_dir.mkdir(parents=True)
    (raw_dir / "20260706-codex-a.md").write_text("a", encoding="utf-8")
    (raw_dir / "20260706-claude-b.md").write_text("b", encoding="utf-8")
    (claims_dir / "claims.jsonl").write_text(
        json.dumps({"source_raw": "20260706-codex-a.md"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(health, "WIKI_ROOT", wiki_root)
    monkeypatch.setattr(health, "RAW_DIR", raw_dir)

    payload = health.capture_kpi()

    assert payload["raw_files"] == 2
    assert payload["claimed_raw_files"] == 1
    assert payload["claim_coverage"] == 0.5
    assert payload["raw_by_host"] == {"codex": 1, "claude": 1}
