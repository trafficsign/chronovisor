from __future__ import annotations

from llm_wiki_mcp import sleep_cycle


def test_run_sleep_cycle_coordinates_safe_steps(monkeypatch) -> None:
    monkeypatch.setattr("llm_wiki_mcp.wiki_snapshot.snapshot_wiki", lambda reason: {"status": "clean", "reason": reason})
    monkeypatch.setattr("llm_wiki_mcp.cofire.build_cofire_graph", lambda write=True: {"edges": 2, "nodes": 2, "graph": {}})
    monkeypatch.setattr("llm_wiki_mcp.prefetch.build_prefetch_cache", lambda write=True: {"status": "ok", "episodes": 1, "buckets": {"a": []}, "tokens": {"b": []}})
    monkeypatch.setattr("llm_wiki_mcp.memory_integrity.run_eval", lambda limit, write=True: {"capture_rate": 0.5, "rows": []})
    monkeypatch.setattr("llm_wiki_mcp.raw_replay.build_queue", lambda limit: {"count": limit})
    monkeypatch.setattr("llm_wiki_mcp.duplicate_review.build_duplicate_review_queue", lambda limit: [{"id": "a"}])
    monkeypatch.setattr("llm_wiki_mcp.duplicate_review.write_review_queue", lambda records: "/tmp/dupes.jsonl")
    monkeypatch.setattr(sleep_cycle, "_append_history", lambda row: None)

    payload = sleep_cycle.run_sleep_cycle(raw_limit=3, eval_limit=4, duplicate_limit=5)

    assert payload["wiki_snapshot"]["status"] == "clean"
    assert payload["cofire"]["edges"] == 2
    assert payload["prefetch"]["buckets"] == 1
    assert payload["memory_integrity"]["capture_rate"] == 0.5
    assert payload["raw_replay"]["count"] == 3
    assert payload["duplicates"]["count"] == 1
