from __future__ import annotations

import hashlib
import json

from llm_wiki_mcp import recall_calibration
from llm_wiki_mcp.convergence import CycleBudget
from llm_wiki_mcp.recall_runtime import RecallPolicy


def test_split_holdout_keeps_latest_rows_for_holdout() -> None:
    rows = [{"ts": f"2026-06-{i:02d}", "features": {}, "label": i % 2} for i in range(1, 11)]

    train, holdout = recall_calibration.split_holdout(rows, holdout_ratio=0.2)

    assert [row["ts"] for row in train] == [f"2026-06-{i:02d}" for i in range(1, 9)]
    assert [row["ts"] for row in holdout] == ["2026-06-09", "2026-06-10"]


def test_min_samples_guard_skips_calibration(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(recall_calibration, "RECALL_LOG_FILE", tmp_path / "recall-log.jsonl")
    monkeypatch.setattr(recall_calibration, "RECALL_FEEDBACK_FILE", tmp_path / "feedback.jsonl")

    result = recall_calibration.calibrate(
        policy=recall_calibration.CalibrationPolicy(min_samples=5),
        log_file=tmp_path / "recall-log.jsonl",
        feedback_file=tmp_path / "feedback.jsonl",
        dry_run=True,
    )

    assert result["status"] == "skipped"
    assert "not enough labeled samples" in result["reason"]


def test_disabled_calibration_does_not_load_training_data(monkeypatch) -> None:
    monkeypatch.setattr(
        recall_calibration,
        "load_labeled_rows",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("disabled lane must do no work")),
    )

    result = recall_calibration.calibrate(
        policy=recall_calibration.CalibrationPolicy(enabled=False)
    )

    assert result["status"] == "disabled"


def test_temporal_holdout_requires_both_classes(monkeypatch) -> None:
    rows = [
        {
            "ts": f"2026-07-10T10:{index:02d}:00",
            "features": {"top1_score_norm": index / 10},
            "label": index % 2 if index < 8 else 1,
        }
        for index in range(10)
    ]
    monkeypatch.setattr(recall_calibration, "load_labeled_rows", lambda **_kwargs: rows)

    result = recall_calibration.calibrate(
        policy=recall_calibration.CalibrationPolicy(
            min_samples=10,
            min_class_samples=2,
            holdout_ratio=0.2,
        ),
        dry_run=True,
    )

    assert result["status"] == "skipped"
    assert "each contain both calibration classes" in result["reason"]
    assert result["split_label_counts"]["holdout"] == {0: 0, 1: 2}


def test_run_due_uses_runtime_disabled_switch(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        recall_calibration,
        "load_policy",
        lambda: RecallPolicy(calibration_enabled=False),
    )

    result = recall_calibration.run_due(
        run_history_file=tmp_path / "history.jsonl",
        dry_run=True,
    )

    assert result["status"] == "disabled"


def test_frontier_retry_quarantines_same_calibration_candidate(monkeypatch, tmp_path) -> None:
    artifact = {
        "weights": {"top1_score_norm": 1.0},
        "bias": 0.0,
        "thresholds": {"search": 0.35, "read": 0.65},
        "samples": 100,
        "holdout": {"improvement": 0.1},
    }
    candidate_hash = hashlib.sha256(
        json.dumps(artifact, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    history_file = tmp_path / "history.jsonl"
    history_file.write_text(
        json.dumps(
            {
                "ts": "2026-07-01T00:00:00",
                "status": "frontier_retry",
                "candidate_hash": candidate_hash,
                "frontier_attempts": 2,
                "next_retry_at": "2026-07-01T00:00:00",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        recall_calibration,
        "calibrate",
        lambda **_kwargs: {
            "status": "frontier_retry",
            "reason": "temporary",
            "candidate": artifact,
        },
    )

    result = recall_calibration.run_due(
        policy=recall_calibration.CalibrationPolicy(),
        run_history_file=history_file,
    )

    assert result["status"] == "frontier_quarantined"
    assert json.loads(history_file.read_text().splitlines()[-1])["frontier_attempts"] == 3


def test_calibration_mutation_budget_guards_authoritative_artifact(monkeypatch, tmp_path) -> None:
    rows = [
        {
            "ts": f"2026-07-10T10:{index:02d}:00",
            "features": {"top1_score_norm": float(index % 2)},
            "label": index % 2,
        }
        for index in range(10)
    ]
    calibration_file = tmp_path / "calibration.json"
    history_file = tmp_path / "calibration-history.jsonl"
    calibration_file.write_text('{"weights":{"old":1}}\n', encoding="utf-8")
    before = calibration_file.read_bytes()
    monkeypatch.setattr(recall_calibration, "CALIBRATION_FILE", calibration_file)
    monkeypatch.setattr(recall_calibration, "CALIBRATION_HISTORY_FILE", history_file)
    monkeypatch.setattr(recall_calibration, "load_calibration", lambda: {"weights": {"old": 1}})
    monkeypatch.setattr(recall_calibration, "load_labeled_rows", lambda **_kwargs: rows)
    monkeypatch.setattr(
        recall_calibration,
        "train_logistic",
        lambda _rows, _policy: ([0.0 for _ in recall_calibration.FEATURE_KEYS], 0.0),
    )
    monkeypatch.setattr(recall_calibration, "evidence_score", lambda _features, _policy: 0.0)
    monkeypatch.setattr(
        recall_calibration,
        "predict",
        lambda _weights, _bias, features: features[0],
    )
    policy = recall_calibration.CalibrationPolicy(
        min_samples=10,
        min_class_samples=1,
        holdout_ratio=0.2,
        min_improvement=0.1,
    )

    denied_budget = CycleBudget(max_mutations=0)
    deferred = recall_calibration.calibrate(policy=policy, budget=denied_budget)
    assert deferred["status"] == "budget_deferred"
    assert calibration_file.read_bytes() == before
    assert not history_file.exists()
    assert denied_budget.snapshot()["used"]["mutation"] == 0

    reject_budget = CycleBudget(max_frontier_calls=1, max_mutations=0)
    rejected = recall_calibration.calibrate(
        policy=policy,
        frontier_mode="auto",
        frontier_reviewer=lambda _artifact: {"decision": "rejected", "summary": "unsafe"},
        budget=reject_budget,
    )
    assert rejected["status"] == "frontier_rejected"
    assert calibration_file.read_bytes() == before
    assert reject_budget.snapshot()["used"]["mutation"] == 0

    apply_budget = CycleBudget(max_mutations=1)
    applied = recall_calibration.calibrate(policy=policy, budget=apply_budget)
    assert applied["status"] == "applied"
    assert calibration_file.read_bytes() != before
    assert apply_budget.snapshot()["used"]["mutation"] == 1
