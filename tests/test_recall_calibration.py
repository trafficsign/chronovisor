from __future__ import annotations

import hashlib
import json
from contextlib import nullcontext

from chronovisor import recall_calibration
from chronovisor.convergence import CycleBudget
from chronovisor.decision_router import canonical_agreement_signature
from chronovisor.decision_schema_manifest import production_decision_schemas
from chronovisor.recall_runtime import RecallPolicy


def _calibration_authority(epoch: str) -> dict:
    return {
        "source": "adopted_local_consensus",
        "authority_version": 1,
        "lane": "recall_calibration",
        "lane_contract_sha256": "1" * 64,
        "lane_contract_manifest_sha256": "2" * 64,
        "lane_contract_case_manifest_sha256": "3" * 64,
        "policy": {
            "kind": "local_batch",
            "schema_name": "generic_decision",
            "mode": "enabled",
            "error": None,
        },
        "router": {
            "source": "adopted_artifact",
            "artifact_sha256": epoch * 64,
            "error": None,
            "models": ["primary", "challenger", "tie"],
        },
    }


def _calibration_review(authority: dict, *, decision: str = "approved") -> dict:
    review = {
        "decision": decision,
        "summary": "authority-bound calibration verdict",
        "tests_run": [],
        "commit": None,
        "committed": False,
        "pushed": False,
        "risk": None,
        "notes": None,
        "decision_policy": {
            **authority["policy"],
            "router_policy": authority["router"],
        },
    }
    schema_name = authority["policy"]["schema_name"]
    signature = canonical_agreement_signature(
        review,
        schema=production_decision_schemas()[schema_name],
    )
    agreement = hashlib.sha256(signature.encode("utf-8")).hexdigest()
    models = authority["router"]["models"]
    review["local_consensus"] = {
        "status": "agreed",
        "ok": True,
        "agreement_sha256": agreement,
        "failure_class": None,
        "quarantine_reason": None,
        "votes": [
            {
                "role": "primary",
                "model": models[0],
                "valid": True,
                "signature_sha256": agreement,
                "invalid_reason": None,
            },
            {
                "role": "challenger",
                "model": models[1],
                "valid": True,
                "signature_sha256": agreement,
                "invalid_reason": None,
            },
        ],
    }
    return review


def test_split_holdout_keeps_latest_rows_for_holdout() -> None:
    rows = [
        {"ts": f"2026-06-{i:02d}", "features": {}, "label": i % 2} for i in range(1, 11)
    ]

    train, holdout = recall_calibration.split_holdout(rows, holdout_ratio=0.2)

    assert [row["ts"] for row in train] == [f"2026-06-{i:02d}" for i in range(1, 9)]
    assert [row["ts"] for row in holdout] == ["2026-06-09", "2026-06-10"]


def test_min_samples_guard_skips_calibration(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        recall_calibration, "RECALL_LOG_FILE", tmp_path / "recall-log.jsonl"
    )
    monkeypatch.setattr(
        recall_calibration, "RECALL_FEEDBACK_FILE", tmp_path / "feedback.jsonl"
    )

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
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("disabled lane must do no work")
        ),
    )

    result = recall_calibration.calibrate(
        policy=recall_calibration.CalibrationPolicy(enabled=False)
    )

    assert result["status"] == "disabled"


def test_rollback_last_restores_exact_applied_preimage_under_nested_locks(
    monkeypatch,
    tmp_path,
) -> None:
    calibration_file = tmp_path / "calibration.json"
    history_file = tmp_path / "history.jsonl"
    old = {"weights": {"old": 1.0}}
    applied = {"weights": {"new": 2.0}}
    recall_calibration._atomic_write_json(calibration_file, applied)
    recall_calibration.append_jsonl(
        history_file,
        {"ts": "2026-07-13T10:00:00", "action": "apply", "old": old, "new": applied},
    )
    monkeypatch.setattr(recall_calibration, "CALIBRATION_FILE", calibration_file)
    monkeypatch.setattr(
        recall_calibration,
        "CALIBRATION_HISTORY_FILE",
        history_file,
    )
    lock_order: list[str] = []

    class TrackedLock:
        def __init__(self, name: str) -> None:
            self.name = name

        def __enter__(self):
            lock_order.append(f"enter:{self.name}")

        def __exit__(self, *_args):
            lock_order.append(f"exit:{self.name}")

    monkeypatch.setattr(
        recall_calibration,
        "decision_authority_lock",
        lambda: TrackedLock("authority"),
    )
    monkeypatch.setattr(
        recall_calibration,
        "chronovisor_mutation_lock",
        lambda: TrackedLock("wiki"),
    )

    result = recall_calibration.rollback_last()

    assert result == {
        "status": "rolled_back",
        "restored_from": "2026-07-13T10:00:00",
        "restored": old,
    }
    assert json.loads(calibration_file.read_text(encoding="utf-8")) == old
    assert lock_order == [
        "enter:authority",
        "enter:wiki",
        "exit:wiki",
        "exit:authority",
    ]


def test_rollback_last_cas_rejects_calibration_applied_while_waiting_for_lock(
    monkeypatch,
    tmp_path,
) -> None:
    calibration_file = tmp_path / "calibration.json"
    history_file = tmp_path / "history.jsonl"
    old = {"weights": {"old": 1.0}}
    applied = {"weights": {"applied": 2.0}}
    raced = {"weights": {"raced": 3.0}}
    recall_calibration._atomic_write_json(calibration_file, applied)
    recall_calibration.append_jsonl(
        history_file,
        {"ts": "2026-07-13T10:00:00", "action": "apply", "old": old, "new": applied},
    )
    monkeypatch.setattr(recall_calibration, "CALIBRATION_FILE", calibration_file)
    monkeypatch.setattr(
        recall_calibration,
        "CALIBRATION_HISTORY_FILE",
        history_file,
    )
    monkeypatch.setattr(recall_calibration, "decision_authority_lock", nullcontext)

    class RacingWikiLock:
        def __enter__(self):
            recall_calibration._atomic_write_json(calibration_file, raced)

        def __exit__(self, *_args):
            return None

    monkeypatch.setattr(recall_calibration, "chronovisor_mutation_lock", RacingWikiLock)

    result = recall_calibration.rollback_last()

    assert result == {
        "status": "conflict",
        "reason": "active calibration changed before rollback",
        "restored_from": "2026-07-13T10:00:00",
    }
    assert json.loads(calibration_file.read_text(encoding="utf-8")) == raced
    rows = recall_calibration.read_jsonl(history_file)
    assert [row["action"] for row in rows] == ["apply"]


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


def test_frontier_retry_quarantines_same_calibration_candidate(
    monkeypatch, tmp_path
) -> None:
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
    assert (
        json.loads(history_file.read_text().splitlines()[-1])["frontier_attempts"] == 3
    )


def test_run_due_rechecks_rejected_authority_before_terminal_history(
    monkeypatch, tmp_path
) -> None:
    authority_a = _calibration_authority("a")
    authority_b = _calibration_authority("b")
    review = _calibration_review(authority_a, decision="rejected")
    history_file = tmp_path / "history.jsonl"
    lock_active = [False]

    class AuthorityLease:
        def __enter__(self):
            lock_active[0] = True

        def __exit__(self, *_args):
            lock_active[0] = False

    monkeypatch.setattr(
        recall_calibration,
        "calibrate",
        lambda **_kwargs: {
            "status": "frontier_rejected",
            "reason": "unsafe",
            "candidate": {"samples": 10},
            "frontier_review": review,
            "frontier_review_authority": authority_a,
        },
    )
    monkeypatch.setattr(
        recall_calibration,
        "_current_calibration_authority",
        lambda **_kwargs: (authority_b, None),
    )
    monkeypatch.setattr(
        recall_calibration,
        "decision_authority_lock",
        AuthorityLease,
    )
    real_append = recall_calibration.append_jsonl

    def append_while_locked(path, payload):
        assert lock_active[0] is True
        real_append(path, payload)

    monkeypatch.setattr(recall_calibration, "append_jsonl", append_while_locked)

    result = recall_calibration.run_due(
        policy=recall_calibration.CalibrationPolicy(),
        run_history_file=history_file,
    )

    recorded = json.loads(history_file.read_text(encoding="utf-8").splitlines()[-1])
    assert result["status"] == "frontier_retry"
    assert result["reason"] == "decision authority changed before effect"
    assert recorded["status"] == "frontier_retry"
    assert recorded["status"] != "frontier_rejected"


def test_calibration_mutation_budget_guards_authoritative_artifact(
    monkeypatch, tmp_path
) -> None:
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
    monkeypatch.setattr(
        recall_calibration,
        "load_calibration",
        lambda _path=None: {"weights": {"old": 1}},
    )
    monkeypatch.setattr(recall_calibration, "chronovisor_mutation_lock", nullcontext)
    monkeypatch.setattr(recall_calibration, "load_labeled_rows", lambda **_kwargs: rows)
    monkeypatch.setattr(
        recall_calibration,
        "train_logistic",
        lambda _rows, _policy: ([0.0 for _ in recall_calibration.FEATURE_KEYS], 0.0),
    )
    monkeypatch.setattr(
        recall_calibration, "evidence_score", lambda _features, _policy: 0.0
    )
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

    denied_budget = CycleBudget(max_frontier_calls=1, max_mutations=0)
    deferred = recall_calibration.calibrate(
        policy=policy,
        budget=denied_budget,
        frontier_reviewer=lambda _proposal: {
            "decision": "approved",
            "summary": "safe",
        },
        review_dir=tmp_path / "denied-reviews",
    )
    assert deferred["status"] == "budget_deferred"
    assert calibration_file.read_bytes() == before
    assert not history_file.exists()
    assert denied_budget.snapshot()["used"]["mutation"] == 0
    assert denied_budget.snapshot()["used"]["frontier"] == 1

    recovered = recall_calibration.calibrate(
        policy=policy,
        budget=CycleBudget(max_frontier_calls=0, max_mutations=1),
        frontier_reviewer=lambda _proposal: (_ for _ in ()).throw(
            AssertionError("durable approval must be reused")
        ),
        review_dir=tmp_path / "denied-reviews",
    )
    assert recovered["status"] == "applied"
    assert recovered["frontier_review_reused"] is True
    calibration_file.write_bytes(before)
    history_file.unlink()

    reject_budget = CycleBudget(max_frontier_calls=1, max_mutations=0)
    rejected = recall_calibration.calibrate(
        policy=policy,
        frontier_mode="auto",
        frontier_reviewer=lambda _artifact: {
            "decision": "rejected",
            "summary": "unsafe",
        },
        budget=reject_budget,
        review_dir=tmp_path / "rejected-reviews",
    )
    assert rejected["status"] == "frontier_rejected"
    assert calibration_file.read_bytes() == before
    assert reject_budget.snapshot()["used"]["mutation"] == 0

    apply_budget = CycleBudget(max_frontier_calls=1, max_mutations=1)
    applied = recall_calibration.calibrate(
        policy=policy,
        budget=apply_budget,
        frontier_reviewer=lambda _proposal: {
            "decision": "approved",
            "summary": "safe",
        },
        review_dir=tmp_path / "apply-reviews",
    )
    assert applied["status"] == "applied"
    assert calibration_file.read_bytes() != before
    assert apply_budget.snapshot()["used"]["mutation"] == 1


def test_calibration_apply_rechecks_authority_inside_effect_lock(
    monkeypatch,
    tmp_path,
) -> None:
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
    monkeypatch.setattr(
        recall_calibration,
        "load_calibration",
        lambda _path=None: {"weights": {"old": 1}},
    )
    monkeypatch.setattr(recall_calibration, "chronovisor_mutation_lock", nullcontext)
    monkeypatch.setattr(recall_calibration, "decision_authority_lock", nullcontext)
    monkeypatch.setattr(recall_calibration, "load_labeled_rows", lambda **_kwargs: rows)
    monkeypatch.setattr(
        recall_calibration,
        "train_logistic",
        lambda _rows, _policy: (
            [0.0 for _ in recall_calibration.FEATURE_KEYS],
            0.0,
        ),
    )
    monkeypatch.setattr(
        recall_calibration,
        "evidence_score",
        lambda _features, _policy: 0.0,
    )
    monkeypatch.setattr(
        recall_calibration,
        "predict",
        lambda _weights, _bias, features: features[0],
    )
    authority_a = _calibration_authority("a")
    authority_b = _calibration_authority("b")
    resolutions = iter([authority_a, authority_a, authority_b])
    monkeypatch.setattr(
        recall_calibration,
        "_current_calibration_authority",
        lambda **_kwargs: (next(resolutions), None),
    )

    result = recall_calibration.calibrate(
        policy=recall_calibration.CalibrationPolicy(
            min_samples=10,
            min_class_samples=1,
            holdout_ratio=0.2,
            min_improvement=0.1,
        ),
        frontier_reviewer=lambda _proposal: _calibration_review(authority_a),
        review_dir=tmp_path / "reviews",
    )

    assert result["status"] == "frontier_retry"
    assert result["reason"] == "decision authority changed before effect"
    assert calibration_file.read_bytes() == before
    assert not history_file.exists()
