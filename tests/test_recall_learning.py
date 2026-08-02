from __future__ import annotations

import json

from chronovisor.recall.recall_learning import (
    append_policy_history,
    decide_learning_update,
    load_last_known_good,
    verify_policy_history,
    write_last_known_good,
)


def _answer_evidence(
    *, point: float = 0.1, lower: float = 0.05
) -> tuple[dict, dict]:
    return (
        {"passed": True},
        {
            "answer_reward": {
                "valid": True,
                "samples": 20,
                "clusters": 20,
                "method": "connected-cluster-bootstrap-percentile",
                "manifest_sha256": "a" * 64,
                "point": point,
                "lower": lower,
                "point_floor": 0.02,
                "lower_floor": 0.0,
            }
        },
    )


def test_learning_is_held_below_strong_positive_and_session_gates() -> None:
    answer, bounds = _answer_evidence()
    decision = decide_learning_update(
        current={"spread_gain": 0.35},
        proposed={"spread_gain": 0.8},
        label_counts={
            "scope": "train",
            "strong_positive": 199,
            "strong_positive_sessions": 50,
            "total": 499,
        },
        metrics={},
        answer_evaluation=answer,
        confidence_bounds=bounds,
    )

    assert decision["status"] == "held"
    assert decision["policy"]["spread_gain"] == 0.35
    assert decision["calibration_allowed"] is False


def test_learning_caps_change_and_rolls_back_on_leakage_or_regression() -> None:
    answer, bounds = _answer_evidence()
    counts = {
        "scope": "train",
        "strong_positive": 250,
        "strong_positive_sessions": 25,
        "total": 600,
    }
    candidate = decide_learning_update(
        current={"spread_gain": 0.35},
        proposed={"spread_gain": 0.8},
        label_counts=counts,
        metrics={"precision_delta_points": 0.0, "recall_delta_points": 0.0},
        answer_evaluation=answer,
        confidence_bounds=bounds,
    )
    leaked = decide_learning_update(
        current={"spread_gain": 0.35},
        proposed={"spread_gain": 0.36},
        label_counts=counts,
        metrics={"session_leakage": 1.0},
        answer_evaluation=answer,
        confidence_bounds=bounds,
    )
    degraded = decide_learning_update(
        current={"spread_gain": 0.35},
        proposed={"spread_gain": 0.36},
        label_counts=counts,
        metrics={"recall_delta_points": -1.1},
        answer_evaluation=answer,
        confidence_bounds=bounds,
    )

    assert candidate["policy"]["spread_gain"] == 0.4
    assert candidate["calibration_allowed"] is True
    assert leaked["status"] == "rollback"
    assert degraded["status"] == "rollback"


def test_learning_separates_point_and_lower_bound_failures() -> None:
    counts = {
        "scope": "train",
        "strong_positive": 250,
        "strong_positive_sessions": 25,
        "total": 600,
    }
    answer, point_failed = _answer_evidence(point=0.01, lower=0.01)
    _, lower_failed = _answer_evidence(point=0.03, lower=-0.01)

    point = decide_learning_update(
        current={"spread_gain": 0.35},
        proposed={"spread_gain": 0.36},
        label_counts=counts,
        metrics={},
        answer_evaluation=answer,
        confidence_bounds=point_failed,
    )
    lower = decide_learning_update(
        current={"spread_gain": 0.35},
        proposed={"spread_gain": 0.36},
        label_counts=counts,
        metrics={},
        answer_evaluation=answer,
        confidence_bounds=lower_failed,
    )

    assert point["reason"] == "answer_reward_point_floor_failed"
    assert lower["reason"] == "answer_reward_lower_bound_failed"


def test_policy_history_is_hash_chained(tmp_path) -> None:
    path = tmp_path / "history.jsonl"
    first = append_policy_history({"status": "candidate"}, path=path)
    second = append_policy_history({"status": "rollback"}, path=path)

    assert first["previous_sha256"] == ""
    assert second["previous_sha256"] == first["record_sha256"]
    assert verify_policy_history(path)["head_sha256"] == second["record_sha256"]


def test_last_known_good_is_sealed_and_tamper_evident(tmp_path) -> None:
    path = tmp_path / "last-known-good.json"
    written = write_last_known_good(
        {"spread_gain": 0.36},
        evaluation={"locked": "passed"},
        history_head_sha256="a" * 64,
        path=path,
    )

    assert load_last_known_good(path)["policy"]["spread_gain"] == 0.36
    written["policy"]["spread_gain"] = 0.99
    path.write_text(json.dumps(written), encoding="utf-8")
    assert load_last_known_good(path) == {}
