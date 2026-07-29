from __future__ import annotations

from chronovisor.classification.classification import default_udc_package
from chronovisor.lab.classification_library_eval import (
    evaluate_candidate_results,
    evaluate_external_test_results,
    evaluate_paired_decisions,
    unsupported_candidate_notations,
)


def test_candidate_eval_joins_gold_only_in_evaluator() -> None:
    fixture = [
        {
            "uid": "a",
            "gold_primary_notation": "004.8",
            "gold_allowed_primary_notations": ["004.8"],
        },
        {
            "uid": "b",
            "gold_primary_notation": "51",
            "gold_allowed_primary_notations": ["51"],
        },
    ]
    provider = [
        {
            "uid": "a",
            "official_baseline": [{"notation": "0"}, {"notation": "004.8"}],
            "external_only": [{"notation": "004.8"}],
            "union": [{"notation": "0"}, {"notation": "004.8"}],
        },
        {
            "uid": "b",
            "official_baseline": [{"notation": "51"}],
            "external_only": [],
            "union": [{"notation": "51"}],
        },
    ]

    result = evaluate_candidate_results(fixture, provider)

    assert result["provider_payload_gold_free"] is True
    assert result["metrics"]["official_baseline"]["recall_at_5"] == 1.0
    assert result["metrics"]["external_only"]["recall_at_5"] == 0.5


def test_paired_eval_uses_unexpected_hold_not_total_hold() -> None:
    fixture = [
        {
            "uid": "expected",
            "gold_primary_notation": "0",
            "gold_allowed_primary_notations": ["0"],
            "gold_expected_status": "held",
        },
        {
            "uid": "assignable",
            "gold_primary_notation": "004.8",
            "gold_allowed_primary_notations": ["004.8"],
            "gold_expected_status": "proposed",
        },
    ]
    arms = {
        "J1": [
            {"uid": "expected", "status": "held"},
            {
                "uid": "assignable",
                "status": "proposed",
                "primary_notation": "0",
            },
        ],
        "J2": [
            {"uid": "expected", "status": "held"},
            {
                "uid": "assignable",
                "status": "proposed",
                "primary_notation": "004.8",
            },
        ],
    }

    result = evaluate_paired_decisions(
        fixture, arms, baseline_arm="J1", seed=7, resamples=100
    )

    assert result["unexpected_hold_rate"] == {"J1": 0.0, "J2": 0.0}
    assert result["exact_rate"]["J2"] > result["exact_rate"]["J1"]
    assert result["severe_error_count"]["J2"] == 0


def test_external_test_reports_slices_and_unsupported_notations() -> None:
    fixture = [
        {
            "uid": "external-a",
            "language": "cze",
            "external_major_class": "0",
            "external_year_bucket": "2020s",
            "external_assignment_count": 1,
            "gold_primary_notation": "004.8",
            "gold_allowed_primary_notations": ["004.8"],
        },
        {
            "uid": "external-b",
            "language": "jpn",
            "external_major_class": "5",
            "external_year_bucket": "2010s",
            "external_assignment_count": 2,
            "gold_primary_notation": "51",
            "gold_allowed_primary_notations": ["51", "510"],
        },
    ]
    provider = [
        {
            "uid": "external-a",
            "official_baseline": [{"notation": "0"}],
            "external_only": [{"notation": "004.8"}],
            "union": [{"notation": "0"}, {"notation": "004.8"}],
        },
        {
            "uid": "external-b",
            "official_baseline": [{"notation": "51"}],
            "external_only": [{"notation": "999.unknown"}],
            "union": [{"notation": "51"}, {"notation": "999.unknown"}],
        },
    ]

    result = evaluate_external_test_results(fixture, provider)
    unsupported = unsupported_candidate_notations(
        provider,
        package=default_udc_package(),
    )

    assert result["n"] == 2
    assert result["group_held_out"] is True
    assert result["slice_counts"]["language:cze"] == 1
    assert result["slice_counts"]["assignment:multiple"] == 1
    assert unsupported == ["999.unknown"]
