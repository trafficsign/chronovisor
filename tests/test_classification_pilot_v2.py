from __future__ import annotations

from chronovisor.lab.classification_pilot_v2 import (
    V2PilotRunner,
    reciprocal_rank_fusion,
    summarize_v2,
)


def test_rrf_combines_relative_ranks_and_requires_unanimous_rejection() -> None:
    left = {
        "ranking": [
            {
                "notation": "575",
                "incidental_match": 0,
                "fatal_contradiction": 0,
            },
            {
                "notation": "57",
                "incidental_match": 0,
                "fatal_contradiction": 0,
            },
        ]
    }
    right = {
        "ranking": [
            {
                "notation": "57",
                "incidental_match": 0,
                "fatal_contradiction": 0,
            },
            {
                "notation": "575",
                "incidental_match": 1,
                "fatal_contradiction": 0,
            },
        ]
    }

    fused = reciprocal_rank_fusion([left, right])

    assert {row["notation"] for row in fused} == {"575", "57"}
    assert fused[0]["notation"] == "57"


def test_rrf_drops_only_unanimous_hard_negative() -> None:
    rejected = {
        "ranking": [
            {
                "notation": "225",
                "incidental_match": 1,
                "fatal_contradiction": 0,
            }
        ]
    }

    assert reciprocal_rank_fusion([rejected, rejected]) == []


def test_v2_summary_cannot_qualify_without_candidate_recall() -> None:
    cases = [
        {
            "reference": {
                "expected_disposition": "leaf",
                "primary_notation": "575",
            },
            "candidate_retrieval": {"notations": ["57"]},
            "variants": {
                "method": {
                    "status": "provisional",
                    "primary_notation": "57",
                }
            },
        }
        for _ in range(10)
    ]

    summary = summarize_v2(cases)

    assert summary["candidate_primary_recall"] == 0
    assert summary["production_qualified"] is False
    assert summary["pilot_winner"] is None


def test_tournament_keeps_groups_small_and_runs_a_final(
    monkeypatch,
) -> None:
    runner = object.__new__(V2PilotRunner)
    calls: list[tuple[int, str, str]] = []

    def fake_rank(
        page,
        candidates,
        normalization,
        *,
        role,
        model,
        keep_alive,
        stage_suffix="",
    ):
        calls.append((len(candidates), stage_suffix, model))
        return {
            "ranking": [
                {
                    "notation": row["notation"],
                    "incidental_match": 0,
                    "fatal_contradiction": 0,
                    "specificity_supported": 1,
                    "rationale": "test",
                }
                for row in candidates[:5]
            ],
            "no_fit": 0,
            "certain_parent": "__NONE__",
        }

    monkeypatch.setattr(runner, "_rank", fake_rank)
    monkeypatch.setattr(runner, "_rank_group", fake_rank)
    candidates = [{"notation": str(index)} for index in range(128)]

    result = runner._rank_tournament(
        {"uid": "uid"},
        candidates,
        {},
        role="test",
        model="test",
        keep_alive="0",
    )

    assert calls[:4] == [
        (32, "-group-1", "gemma4:26b"),
        (32, "-group-2", "gemma4:26b"),
        (32, "-group-3", "gemma4:26b"),
        (32, "-group-4", "gemma4:26b"),
    ]
    assert calls[4] == (20, "-final", "test")
    assert result["group_stage"]["finalist_count"] == 20
    assert result["group_stage"]["model"] == "gemma4:26b"
