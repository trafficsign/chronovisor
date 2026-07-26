"""Gold-isolated candidate and paired-decision evaluation."""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from chronovisor.classification import ClassificationError, UDCPackage
from chronovisor.classification_fixture_set import read_jsonl, sha256_bytes
from chronovisor.durable_state import write_sealed_json

CANDIDATE_EVAL_SCHEMA = "chronovisor.classification-candidate-eval.v1"
PAIRED_EVAL_SCHEMA = "chronovisor.classification-paired-eval.v1"


def _allowed(row: Mapping[str, Any]) -> set[str]:
    primary = str(row.get("gold_primary_notation") or "")
    return {
        str(value)
        for value in row.get("gold_allowed_primary_notations") or [primary]
        if str(value)
    }


def _ranked_notations(result: Mapping[str, Any], field: str) -> list[str]:
    rows = result.get(field) or []
    return [
        str(row.get("notation") or "")
        for row in rows
        if isinstance(row, Mapping) and row.get("notation")
    ]


def _within_one(
    predicted: str,
    allowed: set[str],
    package: UDCPackage | None,
) -> bool:
    if predicted in allowed:
        return True
    if package is not None:
        predicted_row = package.by_notation(predicted)
        predicted_uri = (
            str(predicted_row.get("uri") or "") if predicted_row is not None else ""
        )
        predicted_broader = (
            str(predicted_row.get("broader_uri") or "")
            if predicted_row is not None
            else ""
        )
        for notation in allowed:
            gold = package.by_notation(notation)
            if gold is None:
                continue
            gold_uri = str(gold.get("uri") or "")
            gold_broader = str(gold.get("broader_uri") or "")
            if predicted_broader == gold_uri or gold_broader == predicted_uri:
                return True
    return any(
        predicted.startswith(f"{notation}.") or notation.startswith(f"{predicted}.")
        for notation in allowed
    )


def _host_facets(row: Mapping[str, Any]) -> dict[str, str]:
    form = str(row.get("page_type") or "")
    if form not in {
        "decision",
        "event",
        "howto",
        "reference",
        "architecture",
        "analysis",
        "state",
        "profile",
        "knowledge",
    }:
        form = "knowledge"
    lifecycle = str(row.get("lifecycle") or "")
    if lifecycle not in {
        "active",
        "historical",
        "superseded",
        "experimental",
        "held",
    }:
        lifecycle = "active"
    sensitivity = str(row.get("sensitivity") or "")
    if sensitivity not in {"normal", "personal", "restricted", "high"}:
        sensitivity = "normal"
    return {
        "form": form,
        "lifecycle": lifecycle,
        "evidence": "mixed",
        "sensitivity": sensitivity,
    }


def _facet_macro_f1(rows: Sequence[Mapping[str, Any]]) -> float:
    scores = []
    for field in ("form", "lifecycle", "evidence", "sensitivity"):
        correct = 0
        total = 0
        for row in rows:
            gold = row.get("gold_facets")
            if not isinstance(gold, Mapping):
                continue
            total += 1
            correct += str(gold.get(field) or "") == _host_facets(row)[field]
        if total:
            scores.append(correct / total)
    return sum(scores) / len(scores) if scores else 0.0


def decision_rerun_consistency(
    first: Sequence[Mapping[str, Any]],
    second: Sequence[Mapping[str, Any]],
) -> float:
    right = {str(row.get("uid") or ""): row for row in second}
    matches = 0
    for row in first:
        other = right.get(str(row.get("uid") or ""))
        if other is None:
            continue
        if (
            str(row.get("status") or "") == str(other.get("status") or "")
            and str(row.get("primary_notation") or "")
            == str(other.get("primary_notation") or "")
            and [str(value) for value in row.get("secondary_notations") or []]
            == [str(value) for value in other.get("secondary_notations") or []]
        ):
            matches += 1
    return matches / max(1, len(first))


def evaluate_candidate_results(
    fixture_rows: Sequence[Mapping[str, Any]],
    provider_results: Sequence[Mapping[str, Any]],
    *,
    fields: Sequence[str] = ("official_baseline", "external_only", "union"),
    cutoffs: Sequence[int] = (5, 12, 20, 128),
) -> dict[str, Any]:
    by_uid = {str(row.get("uid") or ""): row for row in provider_results}
    metrics: dict[str, Any] = {}
    for field in fields:
        ranks: list[int | None] = []
        major_errors = 0
        for fixture in fixture_rows:
            uid = str(fixture.get("uid") or "")
            ranked = _ranked_notations(by_uid.get(uid, {}), field)
            allowed = _allowed(fixture)
            rank = next(
                (
                    index
                    for index, notation in enumerate(ranked, start=1)
                    if notation in allowed
                ),
                None,
            )
            ranks.append(rank)
            if ranked and allowed:
                predicted_head = ranked[0].split(".", 1)[0]
                gold_heads = {value.split(".", 1)[0] for value in allowed}
                if predicted_head not in gold_heads:
                    major_errors += 1
        count = max(1, len(ranks))
        values: dict[str, Any] = {
            "n": len(ranks),
            "mrr": sum(0 if rank is None else 1 / rank for rank in ranks) / count,
            "ndcg_at_5": sum(
                0 if rank is None or rank > 5 else 1 / math.log2(rank + 1)
                for rank in ranks
            )
            / count,
            "major_class_error_rate": major_errors / count,
        }
        for cutoff in cutoffs:
            values[f"recall_at_{cutoff}"] = (
                sum(rank is not None and rank <= cutoff for rank in ranks) / count
            )
        metrics[field] = values
    return {
        "schema": CANDIDATE_EVAL_SCHEMA,
        "metrics": metrics,
        "gold_join_location": "evaluator-only",
        "provider_payload_gold_free": all(
            "gold_" not in json.dumps(row, ensure_ascii=False)
            for row in provider_results
        ),
    }


def unsupported_candidate_notations(
    provider_results: Sequence[Mapping[str, Any]],
    *,
    package: UDCPackage,
    fields: Sequence[str] = (
        "official_baseline",
        "official_baseline_tail",
        "official_semantic",
        "external_only",
        "union",
    ),
) -> list[str]:
    unsupported = set()
    for result in provider_results:
        for field in fields:
            for row in result.get(field) or []:
                if not isinstance(row, Mapping):
                    continue
                notation = str(row.get("notation") or "")
                if notation and package.by_notation(notation) is None:
                    unsupported.add(notation)
    return sorted(unsupported)


def evaluate_external_test_results(
    fixture_rows: Sequence[Mapping[str, Any]],
    provider_results: Sequence[Mapping[str, Any]],
    *,
    fields: Sequence[str] = ("official_baseline", "external_only", "union"),
) -> dict[str, Any]:
    """Evaluate the source-group-held-out corpus with fixed diagnostic slices."""

    overall = evaluate_candidate_results(
        fixture_rows,
        provider_results,
        fields=fields,
    )
    notation_counts: dict[str, int] = defaultdict(int)
    for row in fixture_rows:
        for notation in _allowed(row):
            notation_counts[notation] += 1
    ordered = sorted(
        notation_counts, key=lambda value: (-notation_counts[value], value)
    )
    head_count = max(1, math.ceil(len(ordered) * 0.2))
    head = set(ordered[:head_count])

    slices: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in fixture_rows:
        allowed = _allowed(row)
        language = str(row.get("language") or "unknown")
        major = str(row.get("external_major_class") or "unknown")
        year = str(row.get("external_year_bucket") or "unknown")
        assignment = (
            "multiple"
            if int(row.get("external_assignment_count") or len(allowed)) > 1
            else "single"
        )
        frequency = "head" if allowed & head else "tail"
        for name in (
            f"language:{language}",
            f"major:{major}",
            f"year:{year}",
            f"assignment:{assignment}",
            f"frequency:{frequency}",
        ):
            slices[name].append(row)

    by_uid = {str(row.get("uid") or ""): row for row in provider_results}
    slice_metrics = {}
    for name, rows in sorted(slices.items()):
        uid_set = {str(row.get("uid") or "") for row in rows}
        results = [by_uid[uid] for uid in sorted(uid_set) if uid in by_uid]
        slice_metrics[name] = evaluate_candidate_results(
            rows,
            results,
            fields=fields,
        )["metrics"]
    return {
        "schema": "chronovisor.classification-external-test-eval.v1",
        "n": len(fixture_rows),
        "group_held_out": True,
        "metrics": overall["metrics"],
        "slices": slice_metrics,
        "slice_counts": {name: len(rows) for name, rows in sorted(slices.items())},
        "provider_payload_gold_free": overall["provider_payload_gold_free"],
    }


def bootstrap_difference(
    left: Sequence[float],
    right: Sequence[float],
    *,
    seed: int,
    resamples: int = 10_000,
) -> dict[str, float]:
    if len(left) != len(right) or not left:
        raise ClassificationError("paired bootstrap requires equal non-empty arrays")
    randomizer = random.Random(seed)
    differences = []
    for _ in range(resamples):
        indexes = [randomizer.randrange(len(left)) for _ in left]
        differences.append(
            sum(right[index] - left[index] for index in indexes) / len(indexes)
        )
    differences.sort()
    lower = differences[int(resamples * 0.025)]
    upper = differences[min(resamples - 1, int(resamples * 0.975))]
    observed = sum(b - a for a, b in zip(left, right, strict=True)) / len(left)
    return {"difference": observed, "ci_lower": lower, "ci_upper": upper}


def bootstrap_relative_reduction(
    baseline: Sequence[float],
    candidate: Sequence[float],
    *,
    seed: int,
    resamples: int = 10_000,
) -> dict[str, float]:
    if len(baseline) != len(candidate) or not baseline:
        raise ClassificationError(
            "paired relative reduction requires equal non-empty arrays"
        )
    randomizer = random.Random(seed)

    def reduction(indexes: Sequence[int]) -> float:
        baseline_rate = sum(baseline[index] for index in indexes) / len(indexes)
        candidate_rate = sum(candidate[index] for index in indexes) / len(indexes)
        if baseline_rate == 0:
            return 0.0 if candidate_rate == 0 else -1.0
        return (baseline_rate - candidate_rate) / baseline_rate

    distributions = []
    for _ in range(resamples):
        indexes = [randomizer.randrange(len(baseline)) for _ in baseline]
        distributions.append(reduction(indexes))
    distributions.sort()
    observed = reduction(list(range(len(baseline))))
    return {
        "difference": observed,
        "ci_lower": distributions[int(resamples * 0.025)],
        "ci_upper": distributions[min(resamples - 1, int(resamples * 0.975))],
    }


def evaluate_paired_decisions(
    fixture_rows: Sequence[Mapping[str, Any]],
    decisions_by_arm: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    baseline_arm: str,
    seed: int,
    resamples: int = 10_000,
    package: UDCPackage | None = None,
) -> dict[str, Any]:
    fixture_by_uid = {str(row["uid"]): row for row in fixture_rows}
    correctness: dict[str, list[float]] = {}
    unexpected_hold: dict[str, list[float]] = {}
    severe: dict[str, int] = {}
    system_metrics: dict[str, dict[str, Any]] = {}
    facet_macro_f1 = _facet_macro_f1(fixture_rows)
    for arm, rows in decisions_by_arm.items():
        by_uid = {str(row["uid"]): row for row in rows}
        arm_correct: list[float] = []
        arm_unexpected_hold: list[float] = []
        arm_severe = 0
        proposal_available = 0
        published = 0
        non_hold_count = 0
        exact_system = 0
        hierarchy_system = 0
        expected_hold_count = 0
        expected_hold_escape = 0
        total_holds = 0
        for uid, fixture in fixture_by_uid.items():
            expected_hold = str(fixture.get("gold_expected_status") or "") == "held"
            decision = by_uid.get(uid)
            held = decision is None or decision.get("status") == "held"
            if decision is not None and str(decision.get("primary_notation") or ""):
                proposal_available += 1
            if held:
                total_holds += 1
            predicted = (
                "" if decision is None else str(decision.get("primary_notation") or "")
            )
            allowed = _allowed(fixture)
            if expected_hold:
                expected_hold_count += 1
                if not held:
                    expected_hold_escape += 1
            else:
                non_hold_count += 1
            if not held:
                published += 1
            if not expected_hold and not held:
                if predicted in allowed:
                    exact_system += 1
                if _within_one(predicted, allowed, package):
                    hierarchy_system += 1
            arm_correct.append(float(not held and predicted in allowed))
            arm_unexpected_hold.append(float(held and not expected_hold))
            if not held and not expected_hold and predicted not in allowed:
                heads = {value.split(".", 1)[0] for value in allowed}
                if predicted.split(".", 1)[0] not in heads:
                    arm_severe += 1
        correctness[arm] = arm_correct
        unexpected_hold[arm] = arm_unexpected_hold
        severe[arm] = arm_severe
        system_metrics[arm] = {
            "proposal_availability": proposal_available / max(1, len(fixture_by_uid)),
            "published_assignment_rate": published / max(1, len(fixture_by_uid)),
            "gold_non_hold_system_exact_rate": exact_system / max(1, non_hold_count),
            "gold_non_hold_system_hierarchy_rate": hierarchy_system
            / max(1, non_hold_count),
            "published_assignment_conditional_exact_rate": exact_system
            / max(1, published),
            "unexpected_hold_rate": sum(arm_unexpected_hold) / max(1, non_hold_count),
            "total_hold_rate": total_holds / max(1, len(fixture_by_uid)),
            "expected_hold_count": expected_hold_count,
            "expected_hold_escape_count": expected_hold_escape,
            "required_facet_macro_f1": facet_macro_f1,
        }
    baseline = correctness[baseline_arm]
    comparisons = {}
    for arm, values in correctness.items():
        if arm == baseline_arm:
            continue
        comparisons[arm] = {
            "exact": bootstrap_difference(
                baseline, values, seed=seed, resamples=resamples
            ),
            "unexpected_hold": bootstrap_difference(
                unexpected_hold[baseline_arm],
                unexpected_hold[arm],
                seed=seed + 1,
                resamples=resamples,
            ),
            "unexpected_hold_relative_reduction": bootstrap_relative_reduction(
                unexpected_hold[baseline_arm],
                unexpected_hold[arm],
                seed=seed + 2,
                resamples=resamples,
            ),
        }
    return {
        "schema": PAIRED_EVAL_SCHEMA,
        "n": len(fixture_rows),
        "baseline_arm": baseline_arm,
        "seed": seed,
        "resamples": resamples,
        "exact_rate": {
            arm: sum(values) / max(1, len(values))
            for arm, values in correctness.items()
        },
        "unexpected_hold_rate": {
            arm: sum(values) / max(1, len(values))
            for arm, values in unexpected_hold.items()
        },
        "severe_error_count": severe,
        "system_metrics": system_metrics,
        "comparisons": comparisons,
    }


def evaluate_files(
    *,
    fixture_path: Path,
    provider_result_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    result = evaluate_candidate_results(
        read_jsonl(fixture_path), read_jsonl(provider_result_path)
    )
    result["fixture_sha256"] = sha256_bytes(fixture_path.read_bytes())
    result["provider_result_sha256"] = sha256_bytes(provider_result_path.read_bytes())
    write_sealed_json(output_path, result, backup=True)
    return result
