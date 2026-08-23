from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path

import pytest

from chronovisor.core.okf_cutover import OKFStartupBlocked, OKFStartupDecision
from chronovisor.ops import sleep_cycle


@pytest.fixture(autouse=True)
def _valid_okf_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "wiki"
    root.mkdir()
    for name in ("index.md", "log.md", "schema.md"):
        (root / name).write_text("legacy\n", encoding="utf-8")
    monkeypatch.setattr(sleep_cycle, "CHRONOVISOR_ROOT", root)


def test_direct_sleep_entrypoint_holds_shared_writer_lease(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    checked = False

    def run(**_kwargs):
        nonlocal checked
        descriptor = os.open(sleep_cycle.CHRONOVISOR_ROOT, os.O_RDONLY)
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(descriptor)
        checked = True
        return {"status": "ok"}

    monkeypatch.setattr(sleep_cycle, "_run_sleep_cycle_operation", run)

    assert sleep_cycle.main(["--json"]) == 0
    assert json.loads(capsys.readouterr().out) == {"status": "ok"}
    assert checked


def test_direct_sleep_entrypoint_reports_content_free_startup_block(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        sleep_cycle,
        "run_sleep_cycle",
        lambda **_kwargs: (_ for _ in ()).throw(
            OKFStartupBlocked(
                OKFStartupDecision(False, "blocked", "blocked", "unsafe_detail")
            )
        ),
    )

    assert sleep_cycle.main(["--json"]) == 75
    assert json.loads(capsys.readouterr().out) == {
        "status": "blocked",
        "category": "okf_startup_blocked",
    }


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


def test_sleep_history_compacts_recursive_rows_and_bounds_file(
    tmp_path: Path,
    monkeypatch,
) -> None:
    history = tmp_path / "sleep-cycle-history.jsonl"
    legacy_rows = [
        {
            "status": "ok",
            "started_at": f"2026-07-{(index % 9) + 1:02d}T03:40:00",
            "autonomy": {
                "watchdog": {
                    "latest_sleep": {"autonomy": {"watchdog": {"blob": "x" * 1000}}}
                }
            },
        }
        for index in range(12)
    ]
    history.write_text(
        "".join(json.dumps(row) + "\n" for row in legacy_rows),
        encoding="utf-8",
    )
    monkeypatch.setattr(sleep_cycle, "HISTORY_FILE", history)

    sleep_cycle._append_history(
        {
            "status": "ok",
            "run_id": "run-11",
            "started_at": "2026-07-11T03:40:00",
            "finished_at": "2026-07-11T03:43:00",
            "dry_run": False,
            "lint_repair": {"status": "ok", "processed": 5, "applied": 3},
            "convergence_budget": {
                "used": {"frontier": 1},
                "limits": {"frontier": 9},
            },
        },
        max_lines=10,
    )

    rows = [
        json.loads(line) for line in history.read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 10
    assert all("autonomy" not in row for row in rows)
    assert rows[-1]["run_id"] == "run-11"
    assert rows[-1]["finished_at"] == "2026-07-11T03:43:00"
    assert rows[-1]["work"]["lint_repair"] == {"applied": 3, "processed": 5}
    assert rows[-1]["convergence_budget"] == {
        "limits": {"frontier": 9},
        "used": {"frontier": 1},
    }
    assert sleep_cycle._sleep_history_summary(rows[-1]) == rows[-1]
    assert history.stat().st_size < 20_000


def test_atomic_sleep_history_failure_preserves_previous_file(
    tmp_path: Path,
    monkeypatch,
) -> None:
    history = tmp_path / "sleep-cycle-history.jsonl"
    history.write_text('{"status":"old"}\n', encoding="utf-8")
    before = history.read_bytes()
    monkeypatch.setattr(
        sleep_cycle.os,
        "replace",
        lambda *_args: (_ for _ in ()).throw(OSError("replace failed")),
    )

    with pytest.raises(OSError, match="replace failed"):
        sleep_cycle._atomic_write_history(history, [{"status": "new"}])

    assert history.read_bytes() == before
    assert list(tmp_path.glob(".sleep-cycle-history.jsonl.*.tmp")) == []


def test_run_lane_interrupts_at_lane_runtime_budget(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        sleep_cycle,
        "ACTIVE_LANE_FILE",
        tmp_path / "sleep-cycle-active-lane.json",
    )

    completed = False

    def slow_lane() -> None:
        nonlocal completed
        sleep_cycle.time.sleep(0.2)
        completed = True

    payload = sleep_cycle._run_lane(
        "slow", slow_lane, max_elapsed_seconds=0.02
    )

    assert payload["status"] == "budget_deferred"
    assert payload["lane"] == "slow"
    assert payload["reason"] == "lane runtime budget exhausted"
    assert completed is False
    active = json.loads(sleep_cycle.ACTIVE_LANE_FILE.read_text(encoding="utf-8"))
    assert active["lane"] == "slow"
    assert active["status"] == "budget_deferred"


def test_run_lane_defers_before_cycle_finalization_reserve(monkeypatch) -> None:
    monkeypatch.setenv(
        "CHRONOVISOR_CYCLE_DEADLINE_MONOTONIC",
        str(sleep_cycle.time.monotonic() + 5),
    )
    called = False

    def should_not_run():
        nonlocal called
        called = True

    payload = sleep_cycle._run_lane("late", should_not_run)

    assert payload["status"] == "budget_deferred"
    assert payload["reason"] == "sleep cycle runtime budget exhausted"
    assert called is False


def _patch_sleep_dependencies(monkeypatch) -> None:
    monkeypatch.setattr(
        "chronovisor.recall.recall_distillation.distillation_enabled", lambda: False
    )
    monkeypatch.setattr(
        "chronovisor.ingest.snapshot.snapshot_chronovisor",
        lambda reason: {"status": "clean", "reason": reason},
    )
    monkeypatch.setattr(
        "chronovisor.ops.health.health_snapshot",
        lambda: {"memory_integrity": {"capture_rate": 0.5}, "queues": {}},
    )
    monkeypatch.setattr(
        "chronovisor.recall.cofire.build_cofire_graph",
        lambda write=True: {"edges": 2, "nodes": 2, "graph": {}},
    )
    monkeypatch.setattr(
        sleep_cycle,
        "run_graph_maintenance",
        lambda **kwargs: {
            "status": "ok",
            "relation_counts": {"proposed": 2},
            "dry_run": kwargs["dry_run"],
            "external_model_calls": 0,
        },
    )
    monkeypatch.setattr(
        "chronovisor.ops.sleep_cycle.run_growth_cycle",
        lambda **kwargs: {
            "status": "ok",
            "stage": "collecting_labels",
            "dry_run": kwargs["dry_run"],
        },
    )
    monkeypatch.setattr(
        "chronovisor.core.prefetch.build_prefetch_cache",
        lambda write=True: {
            "status": "ok",
            "episodes": 1,
            "buckets": {"a": []},
            "tokens": {"b": []},
        },
    )
    monkeypatch.setattr(
        "chronovisor.core.retention.build_retention_scores",
        lambda write=True: {"counts": {"pages": 2}, "pages": {}},
    )
    monkeypatch.setattr(
        "chronovisor.core.claims.rebuild_claim_index",
        lambda write=True: {"claims": 3},
    )
    monkeypatch.setattr(
        "chronovisor.recall.claims.review_claim_conflicts",
        lambda **kwargs: {"status": "ok", "write": kwargs["write"]},
    )
    monkeypatch.setattr(
        "chronovisor.ops.page_normalize.normalize_pages",
        lambda **kwargs: {"status": "ok", "write": kwargs["write"]},
    )
    monkeypatch.setattr(
        "chronovisor.ops.metadata_backfill.backfill_metadata",
        lambda **kwargs: {"status": "ok", "dry_run": kwargs["dry_run"]},
    )
    monkeypatch.setattr(
        "chronovisor.ops.entities.backfill_entities",
        lambda **kwargs: {"status": "ok", "dry_run": kwargs["dry_run"]},
    )
    monkeypatch.setattr(
        "chronovisor.recall.librarian.run_shadow",
        lambda **kwargs: {"status": "ok", "dry_run": kwargs["dry_run"]},
    )
    monkeypatch.setattr(
        "chronovisor.recall.librarian_release.finalize_if_ready",
        lambda _root: {"status": "not_started"},
    )
    monkeypatch.setattr(
        "chronovisor.core.migration_snapshot.cleanup_expired_restore_points",
        lambda _root: {"deleted": [], "retained": []},
    )
    monkeypatch.setattr(
        "chronovisor.recall.merge_transaction.cleanup_expired_preimages",
        lambda _root: {"deleted": [], "retained": []},
    )
    monkeypatch.setattr(
        "chronovisor.recall.content_correction.run_pending_corrections",
        lambda **kwargs: {"status": "ok", "dry_run": kwargs["dry_run"]},
    )
    monkeypatch.setattr(
        "chronovisor.decision.frontier_review.run_frontier_preflight",
        lambda: {"ok": True},
    )
    monkeypatch.setattr(
        "chronovisor.ingest.convergence.ConvergenceStore.resume_due_quarantined",
        lambda self, **kwargs: {"status": "ok"},
    )
    monkeypatch.setattr(
        "chronovisor.ingest.convergence.ConvergenceStore.resume_due_human_required",
        lambda self, **kwargs: {"status": "ok"},
    )
    monkeypatch.setattr(
        "chronovisor.ingest.convergence.ConvergenceStore.reap_expired_leases",
        lambda self, **kwargs: {"status": "ok"},
    )
    monkeypatch.setattr(
        "chronovisor.ops.golden_expand.expand_golden_from_recall_questions",
        lambda limit=0, write=True: {"added": 4},
    )
    monkeypatch.setattr(
        "chronovisor.ops.distill.export_distill_dataset", lambda write=True: {"rows": 5}
    )
    monkeypatch.setattr(
        "chronovisor.ops.hubs.build_hub_pages",
        lambda write=True: {"hubs": 6, "paths": []},
    )
    monkeypatch.setattr(
        "chronovisor.ops.reflection.write_reflection_page",
        lambda write=True: {"path": "/tmp/reflection.md"},
    )
    monkeypatch.setattr(
        "chronovisor.ingest.state_register.refresh_state_register",
        lambda write=True: {"pages": ["p"]},
    )
    monkeypatch.setattr(
        "chronovisor.ops.memory_integrity.run_eval",
        lambda limit, write=True: {"capture_rate": 0.5, "rows": []},
    )
    monkeypatch.setattr(
        "chronovisor.ingest.raw_replay.build_queue",
        lambda **kwargs: {"count": kwargs["limit"], "status": "ok"},
    )
    monkeypatch.setattr(
        "chronovisor.ingest.raw_replay.run_pending_queue",
        lambda **kwargs: {"count": kwargs["max_runs"], "status": "ok"},
    )
    monkeypatch.setattr(
        "chronovisor.ingest.orchestrator.run_lint_if_due",
        lambda **kwargs: {"status": "ok", "dry_run": kwargs["dry_run"]},
    )
    monkeypatch.setattr(
        "chronovisor.ops.lint_repair.run_lint_repair",
        lambda **kwargs: {"status": "ok", "dry_run": kwargs["dry_run"]},
    )
    monkeypatch.setattr(
        "chronovisor.search.search_eval.build_label_queue",
        lambda **kwargs: {"status": "ok", "dry_run": kwargs["dry_run"]},
    )
    monkeypatch.setattr(
        "chronovisor.search.search_eval.review_label_queue_with_frontier",
        lambda **kwargs: {"status": "ok", "dry_run": kwargs["dry_run"]},
    )
    monkeypatch.setattr(
        "chronovisor.search.search_eval.run_self_tune_due",
        lambda **kwargs: {"status": "skipped", "dry_run": kwargs["dry_run"]},
    )
    monkeypatch.setattr(
        "chronovisor.recall.recall_auto_apply.apply_feedback_file",
        lambda **kwargs: {"status": "ok", "dry_run": kwargs["dry_run"]},
    )
    monkeypatch.setattr(
        "chronovisor.ingest.read_back_repair.run_read_back_repair",
        lambda **kwargs: {"status": "ok", "dry_run": kwargs["dry_run"]},
    )
    monkeypatch.setattr(
        "chronovisor.ingest.self_heal.run_pending",
        lambda **kwargs: {"status": "ok", "dry_run": kwargs["dry_run"]},
    )
    monkeypatch.setattr(
        "chronovisor.recall.recall_calibration.run_due",
        lambda **kwargs: {"status": "skipped", "dry_run": kwargs["dry_run"]},
    )
    monkeypatch.setattr(
        "chronovisor.recall.duplicate_review.build_duplicate_review_queue",
        lambda limit: [{"id": "a"}],
    )
    monkeypatch.setattr(
        "chronovisor.recall.duplicate_review.write_review_queue",
        lambda records: "/tmp/dupes.jsonl",
    )
    monkeypatch.setattr(
        "chronovisor.recall.recall_improvement.run_due",
        lambda **kwargs: {"status": "skipped", "dry_run": kwargs["dry_run"]},
    )
    monkeypatch.setattr(
        "chronovisor.search.research_consolidation.run_consolidation",
        lambda **kwargs: {
            "status": "ok",
            "dry_run": kwargs["dry_run"],
            "mutation_mode": "proposal_only",
        },
    )
    monkeypatch.setattr(
        "chronovisor.ops.autonomy.run_autonomy_cycle",
        lambda **kwargs: {"status": "ok", "dry_run": kwargs["dry_run"]},
    )
    monkeypatch.setattr(
        "chronovisor.ops.autonomy.resolve_deferred_duplicates_with_frontier",
        lambda records, **kwargs: {
            "status": "ok",
            "seen": len(records),
            "dry_run": kwargs["dry_run"],
        },
    )
    monkeypatch.setattr(
        "chronovisor.ops.orphan_link.run_autonomous",
        lambda **kwargs: {"status": "ok", "dry_run": kwargs["dry_run"]},
    )
    monkeypatch.setattr(sleep_cycle, "_append_history", lambda row: None)


def test_run_sleep_cycle_coordinates_safe_steps(monkeypatch) -> None:
    _patch_sleep_dependencies(monkeypatch)
    drain_kwargs: dict = {}

    def fake_drain(**kwargs):
        drain_kwargs.update(kwargs)
        return {"count": kwargs["max_runs"], "status": "ok"}

    monkeypatch.setattr("chronovisor.ingest.raw_replay.run_pending_queue", fake_drain)

    payload = sleep_cycle.run_sleep_cycle(raw_limit=3, eval_limit=4, duplicate_limit=5)

    assert payload["snapshot"]["status"] == "clean"
    assert len(payload["run_id"]) == 32
    assert payload["cofire"]["edges"] == 2
    assert payload["typed_graph"]["relation_counts"] == {"proposed": 2}
    assert payload["recall_growth"]["stage"] == "collecting_labels"
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
    assert payload["research_consolidation"]["mutation_mode"] == "proposal_only"
    assert payload["autonomy"]["status"] == "ok"
    assert payload["lint_repair"]["status"] == "ok"
    assert payload["search_label_review"]["status"] == "ok"
    assert payload["duplicate_frontier"]["status"] == "ok"
    assert payload["orphan_links"]["status"] == "ok"
    assert payload["snapshot_after"]["status"] == "clean"


def test_sleep_cycle_runs_distillation_as_the_only_recall_writer(monkeypatch) -> None:
    _patch_sleep_dependencies(monkeypatch)
    calls: list[bool] = []
    monkeypatch.setattr(
        "chronovisor.recall.recall_distillation.distillation_enabled", lambda: True
    )
    monkeypatch.setattr(
        "chronovisor.recall.recall_distillation.run_distillation_chunk",
        lambda **kwargs: calls.append(kwargs["dry_run"]) or {"status": "ok"},
    )
    monkeypatch.setattr(
        sleep_cycle,
        "run_growth_cycle",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("growth must stay off")),
    )
    monkeypatch.setattr(
        "chronovisor.recall.recall_improvement.run_due",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("improvement must stay off")),
    )
    monkeypatch.setattr(
        "chronovisor.recall.recall_calibration.run_due",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("calibration must stay off")),
    )

    payload = sleep_cycle.run_sleep_cycle(raw_limit=0, eval_limit=0, duplicate_limit=0)

    assert calls == [False]
    assert payload["recall_distillation"]["status"] == "ok"
    for lane in ("recall_growth", "recall_improve", "recall_calibration"):
        assert payload[lane]["reason"] == "distillation_single_writer"


def test_sleep_cycle_always_writes_watchdog_receipt(
    tmp_path: Path,
    monkeypatch,
) -> None:
    append_history = sleep_cycle._append_history
    _patch_sleep_dependencies(monkeypatch)
    history = tmp_path / "sleep-cycle-history.jsonl"
    monkeypatch.setattr(sleep_cycle, "HISTORY_FILE", history)
    monkeypatch.setattr(sleep_cycle, "_append_history", append_history)

    payload = sleep_cycle.run_sleep_cycle(
        raw_limit=0,
        eval_limit=0,
        duplicate_limit=0,
    )

    rows = [
        json.loads(line)
        for line in history.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert payload["history"] == {"status": "ok"}
    assert rows[-1]["run_id"] == payload["run_id"]
    assert rows[-1]["status"] == payload["status"]


def test_sleep_lane_error_isolated_as_partial(monkeypatch) -> None:
    _patch_sleep_dependencies(monkeypatch)
    monkeypatch.setattr(
        "chronovisor.search.search_eval.build_label_queue",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    payload = sleep_cycle.run_sleep_cycle(raw_limit=0, eval_limit=1, duplicate_limit=0)

    assert payload["status"] == "partial"
    assert "search_labels" in payload["lane_errors"]
    assert payload["read_back_repair"]["status"] == "ok"


def test_sleep_artifact_lane_error_does_not_block_consumers(monkeypatch) -> None:
    _patch_sleep_dependencies(monkeypatch)
    monkeypatch.setattr(
        "chronovisor.recall.cofire.build_cofire_graph",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("cofire boom")),
    )

    payload = sleep_cycle.run_sleep_cycle(raw_limit=0, eval_limit=0, duplicate_limit=0)

    assert payload["status"] == "partial"
    assert "cofire" in payload["lane_errors"]
    assert payload["raw_replay"]["status"] == "ok"
    assert payload["read_back_repair"]["status"] == "ok"
    assert payload["orphan_links"]["status"] == "ok"


def test_non_json_cli_renders_partial_cycle_without_key_error(
    monkeypatch, capsys
) -> None:
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
        "chronovisor.ops.lint_repair.run_lint_repair",
        lambda **kwargs: spend("lint", kwargs["budget"]),
    )
    monkeypatch.setattr(
        "chronovisor.search.search_eval.review_label_queue_with_frontier",
        lambda **kwargs: spend("labels", kwargs["budget"]),
    )
    monkeypatch.setattr(
        "chronovisor.ingest.self_heal.run_pending",
        lambda **kwargs: spend("self_heal", kwargs["frontier_budget"]),
    )
    monkeypatch.setattr(
        "chronovisor.recall.recall_improvement.run_due",
        lambda **kwargs: spend("recall", kwargs["frontier_budget"]),
    )
    monkeypatch.setattr(
        "chronovisor.recall.recall_calibration.run_due",
        lambda **kwargs: spend("calibration", kwargs["budget"]),
    )
    monkeypatch.setattr(
        "chronovisor.search.search_eval.run_self_tune_due",
        lambda **kwargs: spend("self_tune", kwargs["budget"]),
    )
    monkeypatch.setattr(
        "chronovisor.ops.autonomy.resolve_deferred_duplicates_with_frontier",
        lambda records, **kwargs: spend("duplicates", kwargs["budget"]),
    )
    monkeypatch.setattr(
        "chronovisor.ops.orphan_link.run_autonomous",
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
        "chronovisor.ingest.snapshot.snapshot_chronovisor",
        lambda _reason: (_ for _ in ()).throw(AssertionError("snapshot must not run")),
    )

    payload = sleep_cycle.run_sleep_cycle(
        raw_limit=1, eval_limit=1, duplicate_limit=1, dry_run=True
    )

    assert payload["status"] == "ok"
    assert payload["snapshot"] == {"status": "skipped", "reason": "dry_run"}
    assert "snapshot_after" not in payload
    assert writes == []


def test_sleep_dry_run_sets_and_restores_process_read_only_guard(monkeypatch) -> None:
    _patch_sleep_dependencies(monkeypatch)
    observed: list[str | None] = []
    monkeypatch.delenv("CHRONOVISOR_READ_ONLY", raising=False)
    monkeypatch.setattr(
        "chronovisor.recall.cofire.build_cofire_graph",
        lambda write=True: (
            observed.append(os.environ.get("CHRONOVISOR_READ_ONLY"))
            or {"edges": 0, "nodes": 0, "graph": {}}
        ),
    )

    sleep_cycle.run_sleep_cycle(
        raw_limit=0, eval_limit=0, duplicate_limit=0, dry_run=True
    )

    assert observed == ["1"]
    assert "CHRONOVISOR_READ_ONLY" not in os.environ
