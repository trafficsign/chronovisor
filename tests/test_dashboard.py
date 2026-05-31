"""Tests for dashboard data assembly."""

from __future__ import annotations

from pathlib import Path

from llm_wiki_mcp import dashboard, runtime_status


def test_build_snapshot_combines_runtime_and_queue(tmp_path: Path, monkeypatch) -> None:
    wiki_root = tmp_path / "wiki"
    raw_dir = wiki_root / "raw"
    runtime_dir = wiki_root / "runtime"
    raw_dir.mkdir(parents=True)
    (wiki_root / "pages").mkdir()
    (wiki_root / "system").mkdir()
    (raw_dir / "r1.md").write_text("raw")

    monkeypatch.setattr(dashboard, "WIKI_ROOT", wiki_root)
    monkeypatch.setattr(dashboard, "LOG_FILE", wiki_root / "log.md")
    monkeypatch.setattr(runtime_status, "RUNTIME_DIR", runtime_dir)
    monkeypatch.setattr(runtime_status, "STATUS_FILE", runtime_dir / "status.json")
    monkeypatch.setattr(runtime_status, "EVENTS_FILE", runtime_dir / "events.jsonl")
    monkeypatch.setattr(runtime_status, "METRICS_FILE", runtime_dir / "metrics.jsonl")

    from llm_wiki_mcp import orchestrator, wiki

    monkeypatch.setattr(wiki, "WIKI_ROOT", wiki_root)
    monkeypatch.setattr(wiki, "RAW_DIR", raw_dir)
    monkeypatch.setattr(wiki, "PAGES_DIR", wiki_root / "pages")
    monkeypatch.setattr(wiki, "SYSTEM_DIR", wiki_root / "system")
    monkeypatch.setattr(wiki, "INDEX_FILE", wiki_root / "index.md")
    monkeypatch.setattr(wiki, "LOG_FILE", wiki_root / "log.md")
    monkeypatch.setattr(orchestrator, "RAW_DIR", raw_dir)
    monkeypatch.setattr(orchestrator, "STATE_FILE", wiki_root / ".orchestrator_state.json")
    monkeypatch.setattr(dashboard, "_ollama_snapshot", lambda: {"available": False, "models": []})

    runtime_status.write_status({"state": "running", "stage": "generate"})
    runtime_status.append_event("info", "ingest | stage 1: triage started")
    runtime_status.append_metric("batch", pending_before=2, pending_after=1, files_processed=1)

    snapshot = dashboard.build_snapshot()

    assert snapshot["status"]["pending"] == 1
    assert snapshot["status"]["state"] == "running"
    assert snapshot["events"]
    assert snapshot["metrics"][0]["pending_after"] == 1
