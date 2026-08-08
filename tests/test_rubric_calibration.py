from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from chronovisor.recall import rubric_calibration
from chronovisor.recall.rubric_calibration import (
    DEFAULT_RUBRIC,
    STRATA,
    build_locked_gold_cycle,
    build_rubric_artifact,
    evaluate_judges,
    load_active_rubric,
    promote_candidate,
    run_calibration_cycle,
    select_diverse_cases,
    write_candidate,
)
from chronovisor.search.search_eval import SearchExample, write_sealed_manifest


def test_diverse_selection_excludes_query_and_session_duplicates() -> None:
    rows = [
        {
            "stratum": stratum,
            "query_sha256": f"q-{index}",
            "session_hash": f"s-{index}",
        }
        for index, stratum in enumerate(
            ["relevant", "multi_hop", "hub_false_positive", "topic_switch"]
        )
    ]
    rows.append({"stratum": "stale_info", "query_sha256": "q-0", "session_hash": "new"})

    selected = select_diverse_cases(rows, limit=10)

    assert len(selected) == 4
    assert len({row["query_sha256"] for row in selected}) == len(selected)
    assert len({row["session_hash"] for row in selected}) == len(selected)


def test_judge_metrics_include_calibration_correlation_and_ensemble_gain() -> None:
    rows = [
        {
            "gold": True,
            "primary": True,
            "challenger": False,
            "tie_break": True,
            "ensemble": True,
            "primary_confidence": 0.9,
            "challenger_confidence": 0.6,
            "tie_break_confidence": 0.8,
            "ensemble_confidence": 0.9,
        },
        {
            "gold": False,
            "primary": False,
            "challenger": False,
            "tie_break": True,
            "ensemble": False,
            "primary_confidence": 0.9,
            "challenger_confidence": 0.8,
            "tie_break_confidence": 0.6,
            "ensemble_confidence": 0.9,
        },
    ]

    metrics = evaluate_judges(rows)

    assert metrics["models"]["ensemble"]["accuracy"] == 1.0
    assert "primary:challenger" in metrics["pairwise_error_correlation"]
    assert metrics["unanimous_wrong_rate"] == 0.0
    assert metrics["ensemble_gain"] == 0.0


def test_active_rubric_is_sealed_and_promotion_fails_closed(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.json"
    active = tmp_path / "active.json"
    lkg = tmp_path / "lkg.json"
    artifact = build_rubric_artifact(rubric_text="Only useful grounded evidence.")
    write_candidate(artifact, candidate)

    held = promote_candidate(
        candidate_file=candidate,
        active_file=active,
        last_known_good_file=lkg,
        metrics={
            "ensemble_gain": 0.1,
            "models": {"ensemble": {"precision": 1.0, "ece": 0.0, "abstention": 0.0}},
            "strata_counts": {stratum: 1 for stratum in STRATA},
            "session_count": 5,
            "split_counts": {
                split: {"positive": 1, "negative": 1}
                for split in ("train", "dev", "locked-test")
            },
        },
        gold_count=10,
    )
    assert held["status"] == "held"
    assert not active.exists()

    insufficient_sessions = promote_candidate(
        candidate_file=candidate,
        active_file=active,
        last_known_good_file=lkg,
        metrics={
            "ensemble_gain": 0.1,
            "models": {"ensemble": {"precision": 1.0, "ece": 0.0, "abstention": 0.0}},
            "strata_counts": {stratum: 1 for stratum in STRATA},
            "session_count": 4,
            "split_counts": {
                split: {"positive": 1, "negative": 1}
                for split in ("train", "dev", "locked-test")
            },
        },
        gold_count=30,
    )
    assert insufficient_sessions["status"] == "held"
    assert insufficient_sessions["gates"]["session_diversity"] is False
    assert not active.exists()

    no_ensemble_gain = promote_candidate(
        candidate_file=candidate,
        active_file=active,
        last_known_good_file=lkg,
        metrics={
            "ensemble_gain": 0.0,
            "models": {"ensemble": {"precision": 1.0, "ece": 0.0, "abstention": 0.0}},
            "strata_counts": {stratum: 1 for stratum in STRATA},
            "session_count": 5,
            "split_counts": {
                split: {"positive": 1, "negative": 1}
                for split in ("train", "dev", "locked-test")
            },
        },
        gold_count=30,
    )
    assert no_ensemble_gain["status"] == "held"
    assert no_ensemble_gain["gates"]["ensemble_value"] is False
    assert not active.exists()

    adopted = promote_candidate(
        candidate_file=candidate,
        active_file=active,
        last_known_good_file=lkg,
        metrics={
            "ensemble_gain": 0.1,
            "models": {"ensemble": {"precision": 1.0, "ece": 0.0, "abstention": 0.0}},
            "strata_counts": {stratum: 1 for stratum in STRATA},
            "session_count": 5,
            "split_counts": {
                split: {"positive": 1, "negative": 1}
                for split in ("train", "dev", "locked-test")
            },
        },
        gold_count=30,
    )
    assert adopted["status"] == "adopted"
    assert load_active_rubric(active)["rubric_text"] == "Only useful grounded evidence."

    active.write_text("{}", encoding="utf-8")
    assert load_active_rubric(active)["rubric_text"] == DEFAULT_RUBRIC


def test_calibration_cycle_requires_background_local_consensus(
    tmp_path: Path, monkeypatch
) -> None:
    rows_file = tmp_path / "locked-gold.jsonl"
    rows = []
    for index in range(30):
        gold = index % 2 == 0
        rows.append(
            {
                "case_id": f"case-{index}",
                "reviewed": True,
                "stratum": STRATA[index % len(STRATA)],
                "split": ("train", "dev", "locked-test")[index % 3],
                "query_sha256": f"query-{index}",
                "session_hash": f"{index:064x}",
                "gold": gold,
                "current": gold,
                "generated": gold,
                "diverse_few_shot": gold,
                "calibrated": gold,
                "primary": (not gold) if index % 5 == 0 else gold,
                "challenger": (not gold) if index % 7 == 0 else gold,
                "tie_break": (not gold) if index % 11 == 0 else gold,
                "ensemble": gold,
                "primary_confidence": 0.95,
                "challenger_confidence": 0.95,
                "tie_break_confidence": 0.95,
                "ensemble_confidence": 0.95,
            }
        )
    rows_file.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    monkeypatch.setattr(
        rubric_calibration,
        "router_for_producer",
        lambda *_args: SimpleNamespace(
            decide=lambda *_call_args, **_kwargs: SimpleNamespace(
                ok=True,
                value={
                    "decision": "approved",
                    "holdout_non_regression": True,
                    "calibration_improved": True,
                    "coverage_preserved": True,
                    "rollback_safe": True,
                },
                agreement_sha256="a" * 64,
                failure_class=None,
                votes=(),
            )
        ),
    )

    result = run_calibration_cycle(
        rows_file=rows_file,
        candidate_file=tmp_path / "candidate.json",
        active_file=tmp_path / "active.json",
        last_known_good_file=tmp_path / "lkg.json",
        status_file=tmp_path / "status.json",
        outcomes_file=tmp_path / "outcomes.jsonl",
    )

    assert result["status"] == "adopted"
    assert result["gates"]["local_consensus"] is True
    assert result["consensus"]["passed"] is True
    assert result["external_model_calls"] == 0


def test_locked_gold_builder_is_incremental_local_and_privacy_safe(
    tmp_path: Path, monkeypatch
) -> None:
    (tmp_path / "pages").mkdir()
    (tmp_path / "pages" / "answer.md").write_text(
        "Grounded answer body.", encoding="utf-8"
    )
    golden = tmp_path / "recall" / "search-golden.jsonl"
    golden.parent.mkdir()
    golden.write_text(
        json.dumps(
            {
                "query": "What is the grounded answer?",
                "expected_pages": ["answer"],
                "negative_pages": [],
                "stale_pages": [],
                "reviewed": True,
                "ref": "manual-1",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "recall" / "feedback.jsonl").write_text(
        json.dumps(
            {
                "ref": "manual-1",
                "snapshot": {"session_id": "private-session-id"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "runtime" / "search-eval" / "manual-94-manifest.json"
    manifest.parent.mkdir(parents=True)
    write_sealed_manifest(
        [
            SearchExample(
                query="What is the grounded answer?",
                expected_pages=("answer",),
                reviewed=True,
                ref="manual-1",
            )
        ],
        manifest,
    )
    monkeypatch.setattr(
        rubric_calibration,
        "_judge_variant",
        lambda *_args, **_kwargs: (True, 0.95),
    )
    monkeypatch.setattr(
        rubric_calibration,
        "_judge_consensus",
        lambda *_args, **_kwargs: {
            "primary": True,
            "primary_confidence": 0.95,
            "challenger": True,
            "challenger_confidence": 0.95,
            "tie_break": "abstain",
            "tie_break_confidence": 0.0,
            "ensemble": True,
            "ensemble_confidence": 0.95,
            "consensus_receipt_sha256": "a" * 64,
        },
    )
    output = tmp_path / "runtime" / "recall-rubric" / "locked-gold.jsonl"
    state = output.parent / "state.json"

    for _ in range(5):
        result = build_locked_gold_cycle(
            root=tmp_path,
            golden_file=golden,
            output_file=output,
            state_file=state,
            max_steps_per_day=10,
        )

    row = json.loads(output.read_text(encoding="utf-8"))
    assert result["cases"] == 1
    assert row["gold"] is True
    assert row["ensemble"] is True
    assert len(row["session_hash"]) == 64
    assert row["session_hash"] != "private-session-id"
    assert "query" not in row
    assert "page_id" not in row
    assert "Grounded answer body" not in output.read_text(encoding="utf-8")

    tampered = json.loads(manifest.read_text(encoding="utf-8"))
    tampered["entries"][0]["source"] = "tampered"
    manifest.write_text(json.dumps(tampered), encoding="utf-8")
    assert rubric_calibration._gold_cases(tmp_path, golden) == []
