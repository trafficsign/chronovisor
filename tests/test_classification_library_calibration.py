from __future__ import annotations

from chronovisor.lab.classification_library_calibration import (
    clopper_pearson_zero_upper,
    evaluate_holdout_gates,
    optional_ablation_decision,
    select_dev_configuration,
)


def test_300_zero_severe_passes_one_sided_exact_upper() -> None:
    assert clopper_pearson_zero_upper(300) < 0.01
    result = evaluate_holdout_gates(
        n=300,
        exact_difference=0.01,
        exact_ci_lower=-0.005,
        unexpected_hold_relative_reduction=0.25,
        unexpected_hold_reduction_ci_lower=0.05,
        severe_error_count=0,
        unexpected_hold_rate=0.07,
        expected_hold_escape_count=0,
        proposal_availability=0.99,
        gold_non_hold_system_exact_rate=0.91,
        gold_non_hold_system_hierarchy_rate=0.98,
        required_facet_macro_f1=0.95,
        rerun_consistency=0.99,
        secondary_comparator_passed=True,
        recall_gate_passed=True,
        resource_gate_passed=True,
        storage_gate_passed=True,
        powered_slices_passed=True,
    )
    assert result["status"] == "passed"


def test_dev_selection_and_optional_conditions_are_fail_closed() -> None:
    selected = select_dev_configuration(
        {
            "P0": {
                "exact_match_rate": 0.9,
                "unexpected_hold_rate": 0.05,
                "severe_error_count": 0,
            },
            "J3": {
                "exact_match_rate": 0.91,
                "unexpected_hold_rate": 0.07,
                "severe_error_count": 0,
            },
        },
        baseline_arm="P0",
    )
    optional = optional_ablation_decision(
        c1_passed=True,
        auto_assignable_hold_count=100,
        explicit_link_gap_count=4,
        clean_training_pairs=9_999,
        llm_wall_time_dominant=True,
    )
    assert selected["arm"] == "J3"
    assert optional == {
        "run_gnd": False,
        "run_yso_after_gnd": False,
        "run_annif": False,
    }
