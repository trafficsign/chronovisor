from __future__ import annotations

from llm_wiki_mcp import sleep_cycle


def test_run_sleep_cycle_coordinates_safe_steps(monkeypatch) -> None:
    monkeypatch.setattr("llm_wiki_mcp.wiki_snapshot.snapshot_wiki", lambda reason: {"status": "clean", "reason": reason})
    monkeypatch.setattr("llm_wiki_mcp.cofire.build_cofire_graph", lambda write=True: {"edges": 2, "nodes": 2, "graph": {}})
    monkeypatch.setattr("llm_wiki_mcp.prefetch.build_prefetch_cache", lambda write=True: {"status": "ok", "episodes": 1, "buckets": {"a": []}, "tokens": {"b": []}})
    monkeypatch.setattr("llm_wiki_mcp.retention.build_retention_scores", lambda write=True: {"counts": {"pages": 2}, "pages": {}})
    monkeypatch.setattr("llm_wiki_mcp.claims.rebuild_claim_index", lambda write=True: {"claims": 3})
    monkeypatch.setattr("llm_wiki_mcp.golden_expand.expand_golden_from_recall_questions", lambda limit=0, write=True: {"added": 4})
    monkeypatch.setattr("llm_wiki_mcp.distill.export_distill_dataset", lambda write=True: {"rows": 5})
    monkeypatch.setattr("llm_wiki_mcp.hubs.build_hub_pages", lambda write=True: {"hubs": 6, "paths": []})
    monkeypatch.setattr("llm_wiki_mcp.reflection.write_reflection_page", lambda write=True: {"path": "/tmp/reflection.md"})
    monkeypatch.setattr("llm_wiki_mcp.state_register.refresh_state_register", lambda write=True: {"pages": ["p"]})
    monkeypatch.setattr("llm_wiki_mcp.memory_integrity.run_eval", lambda limit, write=True: {"capture_rate": 0.5, "rows": []})
    monkeypatch.setattr("llm_wiki_mcp.raw_replay.build_queue", lambda limit: {"count": limit})
    monkeypatch.setattr("llm_wiki_mcp.duplicate_review.build_duplicate_review_queue", lambda limit: [{"id": "a"}])
    monkeypatch.setattr("llm_wiki_mcp.duplicate_review.write_review_queue", lambda records: "/tmp/dupes.jsonl")
    monkeypatch.setattr(sleep_cycle, "_append_history", lambda row: None)

    payload = sleep_cycle.run_sleep_cycle(raw_limit=3, eval_limit=4, duplicate_limit=5)

    assert payload["wiki_snapshot"]["status"] == "clean"
    assert payload["cofire"]["edges"] == 2
    assert payload["prefetch"]["buckets"] == 1
    assert payload["retention"]["counts"]["pages"] == 2
    assert payload["claims"]["claims"] == 3
    assert payload["golden"]["added"] == 4
    assert payload["distill"]["rows"] == 5
    assert payload["hubs"]["hubs"] == 6
    assert payload["state_register"]["pages"] == ["p"]
    assert payload["memory_integrity"]["capture_rate"] == 0.5
    assert payload["raw_replay"]["count"] == 3
    assert payload["duplicates"]["count"] == 1
