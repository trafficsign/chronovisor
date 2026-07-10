from __future__ import annotations

import os
from pathlib import Path

from llm_wiki_mcp import sleep_cycle


def test_sleep_lock_is_single_flight(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(sleep_cycle, "LOCK_FILE", tmp_path / "sleep-cycle.lock")
    first = sleep_cycle._try_acquire_lock()
    assert first is not None
    try:
        assert sleep_cycle._try_acquire_lock() is None
        assert sleep_cycle._try_acquire_read_lock() is None
    finally:
        first.close()
    reader = sleep_cycle._try_acquire_read_lock()
    assert reader not in (None, False)
    reader.close()


def _patch_sleep_dependencies(monkeypatch) -> None:
    monkeypatch.setattr("llm_wiki_mcp.wiki_snapshot.snapshot_wiki", lambda reason: {"status": "clean", "reason": reason})
    monkeypatch.setattr("llm_wiki_mcp.health.health_snapshot", lambda: {"memory_integrity": {"capture_rate": 0.5}, "queues": {}})
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
    monkeypatch.setattr("llm_wiki_mcp.raw_replay.build_queue", lambda **kwargs: {"count": kwargs["limit"], "status": "ok"})
    monkeypatch.setattr("llm_wiki_mcp.raw_replay.run_pending_queue", lambda **kwargs: {"count": kwargs["max_runs"], "status": "ok"})
    monkeypatch.setattr("llm_wiki_mcp.orchestrator.run_lint_if_due", lambda **kwargs: {"status": "ok", "dry_run": kwargs["dry_run"]})
    monkeypatch.setattr("llm_wiki_mcp.lint_repair.run_lint_repair", lambda **kwargs: {"status": "ok", "dry_run": kwargs["dry_run"]})
    monkeypatch.setattr("llm_wiki_mcp.search_eval.build_label_queue", lambda **kwargs: {"status": "ok", "dry_run": kwargs["dry_run"]})
    monkeypatch.setattr("llm_wiki_mcp.search_eval.review_label_queue_with_frontier", lambda **kwargs: {"status": "ok", "dry_run": kwargs["dry_run"]})
    monkeypatch.setattr("llm_wiki_mcp.search_eval.run_self_tune_due", lambda **kwargs: {"status": "skipped", "dry_run": kwargs["dry_run"]})
    monkeypatch.setattr("llm_wiki_mcp.recall_auto_apply.apply_feedback_file", lambda **kwargs: {"status": "ok", "dry_run": kwargs["dry_run"]})
    monkeypatch.setattr("llm_wiki_mcp.read_back_repair.run_read_back_repair", lambda **kwargs: {"status": "ok", "dry_run": kwargs["dry_run"]})
    monkeypatch.setattr("llm_wiki_mcp.self_heal.run_pending", lambda **kwargs: {"status": "ok", "dry_run": kwargs["dry_run"]})
    monkeypatch.setattr("llm_wiki_mcp.recall_calibration.run_due", lambda **kwargs: {"status": "skipped", "dry_run": kwargs["dry_run"]})
    monkeypatch.setattr("llm_wiki_mcp.duplicate_review.build_duplicate_review_queue", lambda limit: [{"id": "a"}])
    monkeypatch.setattr("llm_wiki_mcp.duplicate_review.write_review_queue", lambda records: "/tmp/dupes.jsonl")
    monkeypatch.setattr("llm_wiki_mcp.recall_improvement.run_due", lambda **kwargs: {"status": "skipped", "dry_run": kwargs["dry_run"]})
    monkeypatch.setattr("llm_wiki_mcp.autonomy.run_autonomy_cycle", lambda **kwargs: {"status": "ok", "dry_run": kwargs["dry_run"]})
    monkeypatch.setattr("llm_wiki_mcp.autonomy.resolve_deferred_duplicates_with_frontier", lambda records, **kwargs: {"status": "ok", "seen": len(records), "dry_run": kwargs["dry_run"]})
    monkeypatch.setattr("llm_wiki_mcp.orphan_link.run_autonomous", lambda **kwargs: {"status": "ok", "dry_run": kwargs["dry_run"]})
    monkeypatch.setattr(sleep_cycle, "_append_history", lambda row: None)


def test_run_sleep_cycle_coordinates_safe_steps(monkeypatch) -> None:
    _patch_sleep_dependencies(monkeypatch)
    drain_kwargs: dict = {}

    def fake_drain(**kwargs):
        drain_kwargs.update(kwargs)
        return {"count": kwargs["max_runs"], "status": "ok"}

    monkeypatch.setattr("llm_wiki_mcp.raw_replay.run_pending_queue", fake_drain)

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
    assert payload["raw_replay"]["queue_refresh"]["count"] == 3
    assert payload["raw_replay"]["drain"]["count"] == 1
    assert drain_kwargs["eligible_sources"] == frozenset(
        {"ingest_failure", "memory_integrity_miss"}
    )
    assert drain_kwargs["eligible_keys"] == set()
    assert payload["duplicates"]["count"] == 1
    assert payload["recall_improve"]["status"] == "skipped"
    assert payload["autonomy"]["status"] == "ok"
    assert payload["lint_repair"]["status"] == "ok"
    assert payload["search_label_review"]["status"] == "ok"
    assert payload["duplicate_frontier"]["status"] == "ok"
    assert payload["orphan_links"]["status"] == "ok"
    assert payload["wiki_snapshot_after"]["status"] == "clean"


def test_sleep_lane_error_isolated_as_partial(monkeypatch) -> None:
    _patch_sleep_dependencies(monkeypatch)
    monkeypatch.setattr(
        "llm_wiki_mcp.search_eval.build_label_queue",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    payload = sleep_cycle.run_sleep_cycle(raw_limit=0, eval_limit=1, duplicate_limit=0)

    assert payload["status"] == "partial"
    assert "search_labels" in payload["lane_errors"]
    assert payload["read_back_repair"]["status"] == "ok"


def test_sleep_artifact_lane_error_does_not_block_consumers(monkeypatch) -> None:
    _patch_sleep_dependencies(monkeypatch)
    monkeypatch.setattr(
        "llm_wiki_mcp.cofire.build_cofire_graph",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("cofire boom")),
    )

    payload = sleep_cycle.run_sleep_cycle(raw_limit=0, eval_limit=0, duplicate_limit=0)

    assert payload["status"] == "partial"
    assert "cofire" in payload["lane_errors"]
    assert payload["raw_replay"]["status"] == "ok"
    assert payload["read_back_repair"]["status"] == "ok"
    assert payload["orphan_links"]["status"] == "ok"


def test_non_json_cli_renders_partial_cycle_without_key_error(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        sleep_cycle,
        "run_sleep_cycle",
        lambda **_kwargs: {
            "status": "partial",
            "lane_errors": ["cofire"],
            "cofire": {"status": "error", "error": "boom"},
            "autonomy": {"status": "ok"},
        },
    )

    assert sleep_cycle.main([]) == 0
    output = capsys.readouterr().out
    assert "sleep_cycle\tpartial" in output
    assert "cofire_edges\tunavailable" in output
    assert "lane_errors\tcofire" in output


def test_sleep_reserves_one_frontier_slot_for_every_decision_lane(monkeypatch) -> None:
    _patch_sleep_dependencies(monkeypatch)
    seen: list[str] = []

    def spend(name: str, budget) -> dict:
        assert budget.consume("frontier") == (True, "ok")
        seen.append(name)
        return {"status": "ok"}

    monkeypatch.setattr(
        "llm_wiki_mcp.lint_repair.run_lint_repair",
        lambda **kwargs: spend("lint", kwargs["budget"]),
    )
    monkeypatch.setattr(
        "llm_wiki_mcp.search_eval.review_label_queue_with_frontier",
        lambda **kwargs: spend("labels", kwargs["budget"]),
    )
    monkeypatch.setattr(
        "llm_wiki_mcp.self_heal.run_pending",
        lambda **kwargs: spend("self_heal", kwargs["frontier_budget"]),
    )
    monkeypatch.setattr(
        "llm_wiki_mcp.recall_improvement.run_due",
        lambda **kwargs: spend("recall", kwargs["frontier_budget"]),
    )
    monkeypatch.setattr(
        "llm_wiki_mcp.recall_calibration.run_due",
        lambda **kwargs: spend("calibration", kwargs["budget"]),
    )
    monkeypatch.setattr(
        "llm_wiki_mcp.search_eval.run_self_tune_due",
        lambda **kwargs: spend("self_tune", kwargs["budget"]),
    )
    monkeypatch.setattr(
        "llm_wiki_mcp.autonomy.resolve_deferred_duplicates_with_frontier",
        lambda records, **kwargs: spend("duplicates", kwargs["budget"]),
    )
    monkeypatch.setattr(
        "llm_wiki_mcp.orphan_link.run_autonomous",
        lambda **kwargs: spend("orphans", kwargs["budget"]),
    )

    payload = sleep_cycle.run_sleep_cycle(raw_limit=0, eval_limit=2, duplicate_limit=0)

    assert payload["status"] == "ok"
    assert seen == [
        "lint",
        "labels",
        "self_heal",
        "recall",
        "calibration",
        "self_tune",
        "duplicates",
        "orphans",
    ]
    assert payload["convergence_budget"]["used"]["frontier"] == 8


def test_sleep_dry_run_does_not_snapshot_or_write_history(monkeypatch) -> None:
    _patch_sleep_dependencies(monkeypatch)
    writes: list[dict] = []
    monkeypatch.setattr(sleep_cycle, "_append_history", lambda row: writes.append(row))
    monkeypatch.setattr(
        "llm_wiki_mcp.wiki_snapshot.snapshot_wiki",
        lambda _reason: (_ for _ in ()).throw(AssertionError("snapshot must not run")),
    )

    payload = sleep_cycle.run_sleep_cycle(raw_limit=1, eval_limit=1, duplicate_limit=1, dry_run=True)

    assert payload["status"] == "ok"
    assert payload["wiki_snapshot"] == {"status": "skipped", "reason": "dry_run"}
    assert "wiki_snapshot_after" not in payload
    assert writes == []


def test_sleep_dry_run_sets_and_restores_process_read_only_guard(monkeypatch) -> None:
    _patch_sleep_dependencies(monkeypatch)
    observed: list[str | None] = []
    monkeypatch.delenv("LLM_WIKI_READ_ONLY", raising=False)
    monkeypatch.setattr(
        "llm_wiki_mcp.cofire.build_cofire_graph",
        lambda write=True: (
            observed.append(os.environ.get("LLM_WIKI_READ_ONLY"))
            or {"edges": 0, "nodes": 0, "graph": {}}
        ),
    )

    sleep_cycle.run_sleep_cycle(raw_limit=0, eval_limit=0, duplicate_limit=0, dry_run=True)

    assert observed == ["1"]
    assert "LLM_WIKI_READ_ONLY" not in os.environ
