from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from chronovisor.ingest import read_back_repair, recall_hints
from chronovisor.raw import raw_replay
from chronovisor.recall import recall_auto_apply, recall_calibration
from chronovisor.search import search_eval
from tests.semantic_hold_support import semantic_authority, semantic_review


@pytest.fixture(autouse=True)
def isolate_decision_authority_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chronovisor.core import page_mutation

    monkeypatch.setattr(
        page_mutation,
        "DECISION_AUTHORITY_LOCK",
        tmp_path / "runtime" / "decision-authority.lock",
    )
    # Raw replay binds semantic holds to the current runtime-status evidence.
    # Never let the live ingest worker update that evidence mid-test.
    monkeypatch.setattr(
        raw_replay,
        "RUNTIME_STATUS_FILE",
        tmp_path / "runtime" / "status.json",
    )


def test_raw_replay_reuses_exact_semantic_hold_without_review(
    monkeypatch, tmp_path: Path
) -> None:
    authority = semantic_authority(
        raw_replay.RAW_REPLAY_DECISION_LANE,
        schema_name="raw_replay_reconciliation",
    )
    authority_b = semantic_authority(
        raw_replay.RAW_REPLAY_DECISION_LANE,
        artifact_sha256="f" * 64,
        schema_name="raw_replay_reconciliation",
    )
    current_authority = {"value": authority}
    monkeypatch.setattr(
        raw_replay,
        "_current_raw_replay_authority",
        lambda **_kwargs: (current_authority["value"], None),
    )
    calls = 0

    def reviewer(_prompt: str, _schema: dict) -> dict:
        nonlocal calls
        calls += 1
        return semantic_review(
            current_authority["value"], lane=raw_replay.RAW_REPLAY_DECISION_LANE
        )

    row = {
        "key": "raw-a",
        "raw": "missing.md",
        "status": "indeterminate",
        "frontier_attempts": 0,
        "priority": 1,
    }
    history = tmp_path / "history.jsonl"
    first = raw_replay._review_indeterminate_rows(
        [row],
        claims_file=tmp_path / "claims.jsonl",
        history_file=history,
        now=datetime(2026, 7, 15, tzinfo=UTC),
        budget=None,
        retry_delay_seconds=0,
        reviewer=reviewer,
    )
    second = raw_replay._review_indeterminate_rows(
        [row],
        claims_file=tmp_path / "claims.jsonl",
        history_file=history,
        now=datetime(2027, 7, 15, tzinfo=UTC),
        budget=None,
        retry_delay_seconds=0,
        reviewer=lambda *_args: (_ for _ in ()).throw(
            AssertionError("exact hold must not be reviewed again")
        ),
    )

    assert first["record"]["frontier_decision"] == "semantic_hold"
    assert second["reviewed"] == 0
    assert row["status"] == "quarantined"
    assert calls == 1

    current_authority["value"] = authority_b
    raw_replay._review_indeterminate_rows(
        [row],
        claims_file=tmp_path / "claims.jsonl",
        history_file=history,
        now=datetime(2028, 7, 15, tzinfo=UTC),
        budget=None,
        retry_delay_seconds=0,
        reviewer=reviewer,
    )
    current_authority["value"] = authority
    restored = raw_replay._review_indeterminate_rows(
        [row],
        claims_file=tmp_path / "claims.jsonl",
        history_file=history,
        now=datetime(2029, 7, 15, tzinfo=UTC),
        budget=None,
        retry_delay_seconds=0,
        reviewer=lambda *_args: (_ for _ in ()).throw(
            AssertionError("A-B-A must restore A before a model call")
        ),
    )
    assert restored["reviewed"] == 0
    assert row["semantic_hold"]["authority"] == authority
    assert calls == 2


def test_read_back_repair_semantic_hold_ignores_cooldown(
    monkeypatch, tmp_path: Path
) -> None:
    page = tmp_path / "page.md"
    page.write_text("---\ntitle: Target\n---\nbody\n", encoding="utf-8")
    monkeypatch.setattr(recall_hints.chronovisor_store, "find_page", lambda _page_id: page)
    monkeypatch.setattr(recall_hints.chronovisor_store, "SYSTEM_DIR", tmp_path / "system")
    authority = semantic_authority(
        read_back_repair.READ_BACK_DECISION_LANE,
        schema_name="read_back_repair",
    )
    current_authority = {"value": authority}
    monkeypatch.setattr(
        read_back_repair,
        "_current_query_hint_authority",
        lambda **_kwargs: (current_authority["value"], None),
    )
    failure_file = tmp_path / "failures.jsonl"
    failure_file.write_text(
        json.dumps(
            {
                "timestamp": "2026-07-15T00:00:00+00:00",
                "failed": [
                    {
                        "page_id": "target",
                        "query": "target fact",
                        "reason": "not-in-top-results",
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    ledger = tmp_path / "ledger.json"
    calls = 0

    def reviewer(_proposal: dict) -> dict:
        nonlocal calls
        calls += 1
        return semantic_review(
            current_authority["value"],
            lane=read_back_repair.READ_BACK_DECISION_LANE,
        )

    first = read_back_repair.run_read_back_repair(
        failure_file=failure_file,
        ledger_file=ledger,
        hints_file=tmp_path / "hints.jsonl",
        reviewer=reviewer,
        now=datetime(2026, 7, 15, tzinfo=UTC),
    )
    second = read_back_repair.run_read_back_repair(
        failure_file=failure_file,
        ledger_file=ledger,
        hints_file=tmp_path / "hints.jsonl",
        reviewer=lambda _proposal: (_ for _ in ()).throw(
            AssertionError("cooldown must not reopen semantic hold")
        ),
        now=datetime(2027, 7, 15, tzinfo=UTC),
        quarantine_cooldown_seconds=0,
    )

    assert first["semantic_hold"] == 1
    assert second["semantic_hold"] == 1
    assert second["resumed_quarantined"] == 0
    assert calls == 1

    # The hold ledger must not behave like an eight-entry retry cache.  Even
    # after more than eight authority changes, returning to A restores A
    # before a reviewer call.
    for index in range(1, 11):
        current_authority["value"] = semantic_authority(
            read_back_repair.READ_BACK_DECISION_LANE,
            artifact_sha256=f"{index:064x}",
            schema_name="read_back_repair",
        )
        changed = read_back_repair.run_read_back_repair(
            failure_file=failure_file,
            ledger_file=ledger,
            hints_file=tmp_path / "hints.jsonl",
            reviewer=reviewer,
            now=datetime(2027 + index, 7, 15, tzinfo=UTC),
            quarantine_cooldown_seconds=0,
        )
        assert changed["semantic_hold"] == 1
    current_authority["value"] = authority
    restored = read_back_repair.run_read_back_repair(
        failure_file=failure_file,
        ledger_file=ledger,
        hints_file=tmp_path / "hints.jsonl",
        reviewer=lambda _proposal: (_ for _ in ()).throw(
            AssertionError("oldest hold must survive more than eight transitions")
        ),
        now=datetime(2038, 7, 15, tzinfo=UTC),
        quarantine_cooldown_seconds=0,
    )
    assert restored["semantic_hold"] == 1
    assert calls == 11


def _auto_apply_candidate() -> dict:
    return {
        "kind": "missed_candidate",
        "source": "auditor",
        "lane": "auto",
        "auto_apply_eligible": True,
        "action_type": "query_hint",
        "normalize_key": "target-fact",
        "expected_pages": ["target"],
        "action_payload": {"page_id": "target", "query": "target fact"},
        "ref": "feedback-1",
    }


def test_auto_apply_semantic_hold_is_append_only_terminal(
    monkeypatch, tmp_path: Path
) -> None:
    page = tmp_path / "target.md"
    page.write_text("---\ntitle: Target\n---\nbody\n", encoding="utf-8")
    monkeypatch.setattr(recall_auto_apply.chronovisor_store, "find_page", lambda _page_id: page)
    monkeypatch.setattr(recall_auto_apply, "load_query_hints", lambda: [])
    authority = semantic_authority(recall_auto_apply.AUTO_APPLY_DECISION_LANE)
    authority_b = semantic_authority(
        recall_auto_apply.AUTO_APPLY_DECISION_LANE,
        artifact_sha256="f" * 64,
    )
    current_authority = {"value": authority}
    monkeypatch.setattr(
        recall_auto_apply,
        "_current_review_authority",
        lambda **_kwargs: (current_authority["value"], None),
    )
    calls = 0

    def reviewer(_proposal: dict) -> dict:
        nonlocal calls
        calls += 1
        return semantic_review(
            current_authority["value"],
            lane=recall_auto_apply.AUTO_APPLY_DECISION_LANE,
        )

    kwargs = {
        "policy": recall_auto_apply.AutoApplyPolicy(),
        "log_file": tmp_path / "auto-apply.jsonl",
        "review_dir": tmp_path / "reviews",
        "frontier_reviewer": reviewer,
        "now": datetime(2026, 7, 15),
    }
    first = recall_auto_apply.apply_feedback_records(
        [_auto_apply_candidate()], **kwargs
    )
    second = recall_auto_apply.apply_feedback_records(
        [_auto_apply_candidate()],
        **{
            **kwargs,
            "frontier_reviewer": lambda _proposal: (_ for _ in ()).throw(
                AssertionError("exact hold must bypass reviewer")
            ),
            "now": datetime(2027, 7, 15),
        },
    )

    assert first["actions"][0]["status"] == "semantic_hold"
    assert first["actions"][0]["convergence_status"] == "quarantined"
    assert second["actions"][0]["status"] == "semantic_hold_reused"
    assert calls == 1

    current_authority["value"] = authority_b
    third = recall_auto_apply.apply_feedback_records(
        [_auto_apply_candidate()],
        **{**kwargs, "now": datetime(2028, 7, 15)},
    )
    current_authority["value"] = authority
    fourth = recall_auto_apply.apply_feedback_records(
        [_auto_apply_candidate()],
        **{
            **kwargs,
            "frontier_reviewer": lambda _proposal: (_ for _ in ()).throw(
                AssertionError("A-B-A must restore A before reviewer")
            ),
            "now": datetime(2029, 7, 15),
        },
    )
    assert third["actions"][0]["status"] == "semantic_hold"
    assert fourth["actions"][0]["status"] == "semantic_hold_reused"
    assert fourth["actions"][0]["semantic_hold"]["authority"] == authority
    assert calls == 2


def _calibration_rows() -> list[dict]:
    return [
        {"ts": f"2026-07-{index:02d}", "features": {}, "label": index % 2}
        for index in range(1, 11)
    ]


def test_calibration_semantic_hold_reuses_authority_sidecar(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        recall_calibration, "load_labeled_rows", lambda **_kwargs: _calibration_rows()
    )
    score_calls = iter(
        [
            {"accuracy": 0.5, "precision": 0.5, "recall": 0.5},
            {"accuracy": 0.8, "precision": 0.8, "recall": 0.8},
        ]
        * 2
    )
    monkeypatch.setattr(
        recall_calibration,
        "score_probabilities",
        lambda *_args, **_kwargs: next(score_calls),
    )
    monkeypatch.setattr(
        recall_calibration, "CALIBRATION_FILE", tmp_path / "active.json"
    )
    authority = semantic_authority(
        recall_calibration.CALIBRATION_DECISION_LANE,
        kind="local_batch",
    )
    monkeypatch.setattr(
        recall_calibration,
        "_current_calibration_authority",
        lambda **_kwargs: (authority, None),
    )
    calls = 0

    def reviewer(_proposal: dict) -> dict:
        nonlocal calls
        calls += 1
        return semantic_review(
            authority, lane=recall_calibration.CALIBRATION_DECISION_LANE
        )

    policy = recall_calibration.CalibrationPolicy(
        min_samples=4,
        min_class_samples=1,
        min_improvement=0.1,
    )
    first = recall_calibration.calibrate(
        policy=policy,
        frontier_reviewer=reviewer,
        review_dir=tmp_path / "reviews",
    )
    second = recall_calibration.calibrate(
        policy=policy,
        frontier_reviewer=lambda _proposal: (_ for _ in ()).throw(
            AssertionError("exact sidecar hold must bypass reviewer")
        ),
        review_dir=tmp_path / "reviews",
    )

    assert first["status"] == "semantic_hold"
    assert second["status"] == "semantic_hold"
    assert calls == 1


def test_search_label_semantic_hold_is_not_cooldown_reopened(
    monkeypatch, tmp_path: Path
) -> None:
    queue = tmp_path / "queue.jsonl"
    golden = tmp_path / "golden.jsonl"
    queue.write_text(
        json.dumps(
            {
                "query": "target fact",
                "expected_pages": ["target"],
                "negative_pages": [],
                "stale_pages": [],
                "queue_status": "pending_frontier_review",
                "split": "dev",
                "source": "manual",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    authority = semantic_authority(
        search_eval.SEARCH_LABEL_LANE,
        schema_name="search_label",
    )
    current_authority = {"value": authority}
    monkeypatch.setattr(
        search_eval.decision_authority,
        "current_semantic_authority",
        lambda lane, **_kwargs: (current_authority["value"], None),
    )
    calls = 0

    def reviewer(_row: dict) -> dict:
        nonlocal calls
        calls += 1
        return semantic_review(
            current_authority["value"], lane=search_eval.SEARCH_LABEL_LANE
        )

    first = search_eval.review_label_queue_with_frontier(
        queue_file=queue,
        golden_file=golden,
        reviewer=reviewer,
        now=datetime(2026, 7, 15),
    )
    second = search_eval.review_label_queue_with_frontier(
        queue_file=queue,
        golden_file=golden,
        reviewer=lambda _row: (_ for _ in ()).throw(
            AssertionError("semantic hold must not be cooldown reopened")
        ),
        now=datetime(2027, 7, 15),
    )

    assert first["status_counts"]["semantic_hold"] == 1
    assert second["attempted"] == 0
    assert calls == 1

    for index in range(1, 11):
        current_authority["value"] = semantic_authority(
            search_eval.SEARCH_LABEL_LANE,
            artifact_sha256=f"{index:064x}",
            schema_name="search_label",
        )
        changed = search_eval.review_label_queue_with_frontier(
            queue_file=queue,
            golden_file=golden,
            reviewer=reviewer,
            now=datetime(2027 + index, 7, 15),
        )
        assert changed["status_counts"]["semantic_hold"] == 1
    current_authority["value"] = authority
    restored = search_eval.review_label_queue_with_frontier(
        queue_file=queue,
        golden_file=golden,
        reviewer=lambda _row: (_ for _ in ()).throw(
            AssertionError("oldest label hold must survive authority churn")
        ),
        now=datetime(2038, 7, 15),
    )
    assert restored["attempted"] == 0
    assert calls == 11


def _metric(mrr: float) -> dict[str, float]:
    return {
        "recall_at_5": 0.8,
        "mrr_at_10": mrr,
        "ndcg_at_10": mrr,
        "stale_hit_rate_at_20": 0.0,
        "negative_hit_rate_at_20": 0.0,
    }


def test_search_self_tune_history_restores_exact_hold(
    monkeypatch, tmp_path: Path
) -> None:
    examples = [
        search_eval.SearchExample("dev", ("a",), split="dev", reviewed=True),
        search_eval.SearchExample("locked", ("a",), split="locked-test", reviewed=True),
    ]
    monkeypatch.setattr(search_eval, "load_examples", lambda _path: examples)
    monkeypatch.setattr(search_eval, "_rows_for_weight_eval", lambda *_args: [])
    monkeypatch.setattr(
        search_eval, "load_active_fusion_weights", lambda _path: {"semantic": 0.4}
    )
    authority = semantic_authority(search_eval.SEARCH_SELF_TUNE_LANE)
    monkeypatch.setattr(
        search_eval.decision_authority,
        "current_semantic_authority",
        lambda lane, **_kwargs: (authority, None),
    )
    history = tmp_path / "history.jsonl"
    policy = tmp_path / "policy.json"
    calls = 0

    def run_once(reviewer) -> dict:
        metrics = iter(
            [
                _metric(0.1),
                _metric(0.5),
                _metric(0.2),
                _metric(0.3),
                _metric(0.4),
                _metric(0.6),
                _metric(0.5),
                _metric(0.5),
            ]
        )
        monkeypatch.setattr(search_eval, "_metrics", lambda _rows: next(metrics))
        return search_eval.self_tune(
            golden_file=tmp_path / "golden.jsonl",
            history_file=history,
            policy_file=policy,
            apply=True,
            frontier_mode="auto",
            frontier_reviewer=reviewer,
        )

    def reviewer(_record: dict) -> dict:
        nonlocal calls
        calls += 1
        return semantic_review(authority, lane=search_eval.SEARCH_SELF_TUNE_LANE)

    first = run_once(reviewer)
    second = run_once(
        lambda _record: (_ for _ in ()).throw(
            AssertionError("exact history hold must bypass reviewer")
        )
    )

    assert first["status"] == "semantic_hold"
    assert second["status"] == "semantic_hold"
    assert calls == 1
