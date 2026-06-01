from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture()
def isolated_wiki(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    wiki_root = tmp_path / "wiki"
    pages = wiki_root / "pages"
    raw = wiki_root / "raw"
    system = wiki_root / "system"
    runtime = wiki_root / "runtime"
    for d in (pages, raw, system, runtime):
        d.mkdir(parents=True, exist_ok=True)

    from llm_wiki_mcp import wiki, runtime_status

    monkeypatch.setattr(wiki, "WIKI_ROOT", wiki_root)
    monkeypatch.setattr(wiki, "PAGES_DIR", pages)
    monkeypatch.setattr(wiki, "RAW_DIR", raw)
    monkeypatch.setattr(wiki, "SYSTEM_DIR", system)
    monkeypatch.setattr(runtime_status, "RUNTIME_DIR", runtime)
    monkeypatch.setattr(runtime_status, "STATUS_FILE", runtime / "status.json")
    monkeypatch.setattr(runtime_status, "EVENTS_FILE", runtime / "events.jsonl")
    monkeypatch.setattr(runtime_status, "METRICS_FILE", runtime / "metrics.jsonl")
    return wiki_root


def _seed_page(wiki_root: Path, rel: str) -> None:
    path = wiki_root / "pages" / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("---\ntitle: T\nupdated: 2026-01-01\n---\nbody\n")


def _write_packet(wiki_root: Path) -> Path:
    packet = {
        "failure_id": "f1",
        "raw_file": "broken.md",
        "failure_class": "apply.update_target_not_found",
        "fingerprint": "apply.update_target_not_found:model-made-up-target",
        "attempts": 3,
        "error": "update target not found for page_id 'model-made-up-target'",
        "requested_page_id": "model-made-up-target",
        "similar_existing_pages": ["ai/canonical-target"],
        "status": "pending_local_repair",
    }
    path = wiki_root / "runtime" / "failures" / "packets" / "f1.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(packet))
    return path


def test_local_repair_adds_alias_restores_raw_and_retries(
    isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from llm_wiki_mcp import self_heal
    from llm_wiki_mcp.alias_store import load_aliases

    _seed_page(isolated_wiki, "ai/canonical-target.md")
    packet_path = _write_packet(isolated_wiki)
    quarantined = isolated_wiki / "runtime" / "failures" / "quarantined-raw" / "broken.md"
    quarantined.parent.mkdir(parents=True, exist_ok=True)
    quarantined.write_text("raw body")

    monkeypatch.setattr(
        self_heal,
        "_retry_ingest",
        lambda *, dry_run: {"triggered": True, "files_processed": ["broken.md"]},
    )

    result = self_heal.handle_packet(
        packet_path,
        use_qwen=False,
        enable_frontier=False,
        dry_run=False,
    )

    assert result["status"] == "local_repair_applied"
    assert load_aliases()["model-made-up-target"] == "ai/canonical-target"
    assert not quarantined.exists()
    assert (isolated_wiki / "raw" / "broken.md").exists()
    updated_packet = json.loads(packet_path.read_text())
    assert updated_packet["status"] == "local_repair_applied"


def test_drill_returns_local_repair_decision(isolated_wiki: Path) -> None:
    from llm_wiki_mcp.self_heal import run_drill

    result = run_drill(use_qwen=False)

    assert result["decision"]["status"] == "resolved"
    assert result["decision"]["action"] == "resolve_update_target"
