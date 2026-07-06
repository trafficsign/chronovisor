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


def test_latest_memory_integrity_reads_summary(tmp_path: Path, monkeypatch) -> None:
    wiki_root = tmp_path / "wiki"
    eval_dir = wiki_root / "eval"
    eval_dir.mkdir(parents=True)
    (eval_dir / "memory-integrity-latest.json").write_text(
        json.dumps({"status": "ok", "total": 2, "passed": 1, "missed": 1, "capture_rate": 0.5}),
        encoding="utf-8",
    )
    monkeypatch.setattr(health, "WIKI_ROOT", wiki_root)

    payload = health.latest_memory_integrity()

    assert payload["status"] == "ok"
    assert payload["capture_rate"] == 0.5


def test_cofire_kpi_reads_graph_summary(tmp_path: Path, monkeypatch) -> None:
    wiki_root = tmp_path / "wiki"
    recall_dir = wiki_root / "recall"
    recall_dir.mkdir(parents=True)
    (recall_dir / "cofire.json").write_text(
        json.dumps({"status": "ok", "nodes": 3, "edges": 4}),
        encoding="utf-8",
    )
    monkeypatch.setattr(health, "WIKI_ROOT", wiki_root)

    payload = health.cofire_kpi()

    assert payload["nodes"] == 3
    assert payload["edges"] == 4
