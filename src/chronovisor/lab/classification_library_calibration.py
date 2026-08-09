"""Pre-registered dev selection and one-time Holdout safety gates."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from chronovisor.core.durable_state import write_sealed_json
from chronovisor.recall.classification import ClassificationError
from chronovisor.recall.classification_fixture_set import sha256_bytes

PREREGISTRATION_SCHEMA = "chronovisor.library-evidence-preregistration.v1"
EVALUATION_SCHEMA = "chronovisor.library-evidence-evaluation.v1"


def clopper_pearson_zero_upper(n: int, *, alpha: float = 0.05) -> float:
    if n <= 0 or not 0 < alpha < 1:
        raise ClassificationError("invalid exact-binomial parameters")
    return 1.0 - alpha ** (1.0 / n)


def mcnemar_counts(
    baseline: Sequence[bool],
    candidate: Sequence[bool],
) -> dict[str, int | float]:
    if len(baseline) != len(candidate) or not baseline:
        raise ClassificationError("McNemar requires paired non-empty outcomes")
    baseline_only = sum(a and not b for a, b in zip(baseline, candidate, strict=True))
    candidate_only = sum(b and not a for a, b in zip(baseline, candidate, strict=True))
    discordant = baseline_only + candidate_only
    statistic = (
        (abs(baseline_only - candidate_only) - 1) ** 2 / discordant
        if discordant
        else 0.0
    )
    return {
        "baseline_only": baseline_only,
        "candidate_only": candidate_only,
        "discordant": discordant,
        "continuity_corrected_chi_square": statistic,
    }


def select_dev_configuration(
    arm_metrics: Mapping[str, Mapping[str, Any]],
    *,
    baseline_arm: str,
    exact_noninferiority_margin: float = -0.01,
    maximum_unexpected_hold_rate: float = 0.08,
) -> dict[str, Any]:
    if baseline_arm not in arm_metrics:
        raise ClassificationError("dev metrics lack baseline arm")
    baseline_exact = float(arm_metrics[baseline_arm]["exact_match_rate"])
    eligible = []
    for arm, metrics in arm_metrics.items():
        if arm == baseline_arm:
            continue
        exact = float(metrics["exact_match_rate"])
        unexpected_hold = float(metrics["unexpected_hold_rate"])
        severe = int(metrics.get("severe_error_count") or 0)
        if (
            exact - baseline_exact >= exact_noninferiority_margin
            and unexpected_hold <= maximum_unexpected_hold_rate
            and severe == 0
        ):
            eligible.append((arm, metrics))
    if not eligible:
        return {
            "status": "blocked",
            "reason": "no_dev_configuration_passed",
            "baseline_arm": baseline_arm,
        }
    arm, selected = max(
        eligible,
        key=lambda item: (
            float(item[1]["exact_match_rate"]),
            -float(item[1]["unexpected_hold_rate"]),
            item[0],
        ),
    )
    return {
        "status": "selected",
        "arm": arm,
        "metrics": dict(selected),
        "baseline_arm": baseline_arm,
        "exact_noninferiority_margin": exact_noninferiority_margin,
        "maximum_unexpected_hold_rate": maximum_unexpected_hold_rate,
    }


def preregister_evaluation(
    output_path: Path,
    *,
    fixture_manifest_sha256: str,
    evidence_root: str,
    policy_digest: str,
    selected_configuration: Mapping[str, Any],
    seed: int = 20260726,
    resamples: int = 10_000,
    slice_power: Mapping[str, Any],
) -> dict[str, Any]:
    if selected_configuration.get("status") != "selected":
        raise ClassificationError("cannot preregister an unselected configuration")
    if not slice_power or any(
        not bool(value.get("powered"))
        for value in slice_power.values()
        if isinstance(value, Mapping) and bool(value.get("mandatory"))
    ):
        raise ClassificationError("every mandatory slice requires preregistered power")
    if not any(
        bool(value.get("mandatory")) and bool(value.get("powered"))
        for value in slice_power.values()
        if isinstance(value, Mapping)
    ):
        raise ClassificationError("preregistration requires at least one powered slice")
    payload = {
        "schema": PREREGISTRATION_SCHEMA,
        "fixture_manifest_sha256": fixture_manifest_sha256,
        "evidence_root": evidence_root,
        "policy_digest": policy_digest,
        "selected_configuration": dict(selected_configuration),
        "seed": seed,
        "resamples": resamples,
        "primary_test": "paired-bootstrap-exact-noninferiority-margin-minus-0.01",
        "secondary_test": "mcnemar",
        "severe_error_gate": "zero-and-one-sided-95pct-exact-upper<=0.01",
        "unexpected_hold_gate": "<=0.08-on-auto-assignable-denominator",
        "slice_power": dict(slice_power),
        "holdout_opened": False,
    }
    payload["preregistration_digest"] = sha256_bytes(
        json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    )
    write_sealed_json(output_path, payload, backup=True)
    return payload


def evaluate_holdout_gates(
    *,
    n: int,
    exact_difference: float,
    exact_ci_lower: float,
    unexpected_hold_relative_reduction: float,
    unexpected_hold_reduction_ci_lower: float,
    severe_error_count: int,
    unexpected_hold_rate: float,
    expected_hold_escape_count: int,
    proposal_availability: float,
    gold_non_hold_system_exact_rate: float,
    gold_non_hold_system_hierarchy_rate: float,
    required_facet_macro_f1: float,
    rerun_consistency: float,
    secondary_comparator_passed: bool,
    recall_gate_passed: bool,
    resource_gate_passed: bool,
    storage_gate_passed: bool,
    powered_slices_passed: bool,
    require_severe_exact_upper: bool = True,
    require_primary_effect: bool = True,
) -> dict[str, Any]:
    severe_upper = clopper_pearson_zero_upper(n) if severe_error_count == 0 else 1.0
    primary_effect = (exact_difference >= 0.05 and exact_ci_lower > 0) or (
        exact_ci_lower >= -0.01
        and unexpected_hold_relative_reduction >= 0.20
        and unexpected_hold_reduction_ci_lower > 0
    )
    gates = {
        "exact_noninferiority": exact_ci_lower >= -0.01,
        "secondary_comparator_noninferiority": secondary_comparator_passed,
        "primary_effect": primary_effect if require_primary_effect else True,
        "proposal_availability": proposal_availability >= 0.98,
        "gold_non_hold_system_exact": gold_non_hold_system_exact_rate >= 0.90,
        "gold_non_hold_system_hierarchy": (gold_non_hold_system_hierarchy_rate >= 0.97),
        "severe_zero": severe_error_count == 0,
        "severe_exact_upper": (
            severe_upper <= 0.01 if require_severe_exact_upper else True
        ),
        "unexpected_hold": unexpected_hold_rate <= 0.08,
        "expected_hold_escape": expected_hold_escape_count == 0,
        "required_facets": required_facet_macro_f1 >= 0.90,
        "rerun_consistency": rerun_consistency >= 0.98,
        "recall": recall_gate_passed,
        "resource_ready": resource_gate_passed,
        "storage": storage_gate_passed,
        "powered_slices": powered_slices_passed,
    }
    return {
        "schema": EVALUATION_SCHEMA,
        "status": "passed" if all(gates.values()) else "rejected",
        "n": n,
        "exact_difference": exact_difference,
        "exact_ci_lower": exact_ci_lower,
        "unexpected_hold_relative_reduction": (unexpected_hold_relative_reduction),
        "unexpected_hold_reduction_ci_lower": (unexpected_hold_reduction_ci_lower),
        "severe_error_count": severe_error_count,
        "severe_error_one_sided_95pct_upper": severe_upper,
        "unexpected_hold_rate": unexpected_hold_rate,
        "expected_hold_escape_count": expected_hold_escape_count,
        "proposal_availability": proposal_availability,
        "gold_non_hold_system_exact_rate": gold_non_hold_system_exact_rate,
        "gold_non_hold_system_hierarchy_rate": (gold_non_hold_system_hierarchy_rate),
        "required_facet_macro_f1": required_facet_macro_f1,
        "rerun_consistency": rerun_consistency,
        "gates": gates,
    }


def optional_ablation_decision(
    *,
    c1_passed: bool,
    auto_assignable_hold_count: int,
    explicit_link_gap_count: int,
    clean_training_pairs: int,
    llm_wall_time_dominant: bool,
) -> dict[str, bool]:
    link_gap_rate = explicit_link_gap_count / max(1, auto_assignable_hold_count)
    return {
        "run_gnd": c1_passed and link_gap_rate >= 0.05,
        "run_yso_after_gnd": c1_passed and link_gap_rate >= 0.05,
        "run_annif": (
            c1_passed and clean_training_pairs >= 10_000 and llm_wall_time_dominant
        ),
    }
