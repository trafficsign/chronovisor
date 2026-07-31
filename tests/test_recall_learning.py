from __future__ import annotations

import json

from chronovisor.recall.recall_learning import (
    append_policy_history,
    decide_learning_update,
    load_last_known_good,
    verify_policy_history,
    write_last_known_good,
)


def test_learning_is_held_below_strong_positive_and_session_gates() -> None:
    decision = decide_learning_update(
        current={"spread_gain": 0.35},
        proposed={"spread_gain": 0.8},
        label_counts={
            "strong_positive": 199,
            "strong_positive_sessions": 50,
            "total": 499,
        },
        metrics={},
    )

    assert decision["status"] == "held"
    assert decision["policy"]["spread_gain"] == 0.35
    assert decision["calibration_allowed"] is False


def test_learning_caps_change_and_rolls_back_on_leakage_or_regression() -> None:
    counts = {
        "strong_positive": 250,
        "strong_positive_sessions": 25,
        "total": 600,
    }
    candidate = decide_learning_update(
        current={"spread_gain": 0.35},
        proposed={"spread_gain": 0.8},
        label_counts=counts,
        metrics={"precision_delta_points": 0.0, "recall_delta_points": 0.0},
    )
    leaked = decide_learning_update(
        current={"spread_gain": 0.35},
        proposed={"spread_gain": 0.36},
        label_counts=counts,
        metrics={"session_leakage": 1.0},
    )
    degraded = decide_learning_update(
        current={"spread_gain": 0.35},
        proposed={"spread_gain": 0.36},
        label_counts=counts,
        metrics={"recall_delta_points": -1.1},
    )

    assert candidate["policy"]["spread_gain"] == 0.4
    assert candidate["calibration_allowed"] is True
    assert leaked["status"] == "rollback"
    assert degraded["status"] == "rollback"


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
