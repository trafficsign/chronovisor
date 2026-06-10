from __future__ import annotations

from llm_wiki_mcp import recall_calibration


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
