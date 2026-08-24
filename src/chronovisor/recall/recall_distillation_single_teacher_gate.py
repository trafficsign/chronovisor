"""Fail-closed offline gate for a temporary single-teacher cohort."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any


def _wilson_lower(successes: int, total: int) -> float:
    if total <= 0 or successes < 0 or successes > total:
        return 0.0
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1 + z * z / total
    center = proportion + z * z / (2 * total)
    spread = z * math.sqrt(
        proportion * (1 - proportion) / total + z * z / (4 * total * total)
    )
    return max(0.0, (center - spread) / denominator)


def _text(value: object) -> str:
    return value if isinstance(value, str) else ""


def _complete_row(row: Mapping[str, Any]) -> bool:
    return (
        row.get("status") == "completed"
        and row.get("error_class") in (None, "")
        and row.get("verdict") in {"relevant", "irrelevant"}
    )


def _one(values: Sequence[str], name: str, reasons: list[str]) -> str:
    distinct = set(values)
    if not values or "" in distinct or len(distinct) != 1:
        reasons.append(f"{name}_identity_not_exactly_one")
        return ""
    return next(iter(distinct))


def evaluate_single_teacher_gate(
    rows: Sequence[Mapping[str, Any]],
    *,
    profile: str,
    cohort: str,
    profile_contract_id: str = "",
    min_labels: int = 500,
    min_per_class: int = 100,
    min_repeat_pairs: int = 100,
    min_repeat_stability: float = 0.60,
) -> dict[str, Any]:
    """Evaluate sealed-like rows without granting them verified-truth authority.

    A usable row has ``status='completed'``, no error, and a relevance verdict.
    It must carry immutable ``profile``, ``cohort``, ``route``, ``model_digest``,
    ``prompt_sha256``, ``schema_sha256``, and ``profile_contract_id`` identities.
    The captured provider/model/location route must match exactly.  Completed
    test rows additionally need ``locked_test_read_only=True`` and a non-empty
    ``locked_test_evidence_ref``.  Repeat probes use ``repeat_pair_id``,
    ``fixed_repeat=True``, ``order_swap=True``, and opposite ``blind_order``s.
    """

    reasons: list[str] = []
    valid_rows = [row for row in rows if isinstance(row, Mapping)]
    if len(valid_rows) != len(rows):
        reasons.append("input_row_invalid")
    rows = valid_rows
    if not isinstance(profile, str) or not profile:
        reasons.append("profile_argument_invalid")
    if not isinstance(cohort, str) or not cohort:
        reasons.append("cohort_argument_invalid")
    if min_labels < 1 or min_per_class < 1 or min_repeat_pairs < 1:
        reasons.append("minimum_argument_invalid")
    if not 0.0 <= min_repeat_stability <= 1.0:
        reasons.append("repeat_stability_argument_invalid")

    current_rows = [
        row
        for row in rows
        if not profile_contract_id
        or row.get("profile_contract_id") == profile_contract_id
    ]
    excluded_prior_contract_rows = len(rows) - len(current_rows)
    completed = [row for row in current_rows if _complete_row(row)]
    required_splits = {"train", "validation", "test"}
    eligible = [row for row in completed if row.get("split") in required_splits]
    excluded = len(rows) - len(eligible)
    labels = [row for row in eligible if row.get("probe") is not True]
    probes = [row for row in eligible if row.get("probe") is True]
    relevant = sum(row.get("verdict") == "relevant" for row in labels)
    irrelevant = sum(row.get("verdict") == "irrelevant" for row in labels)

    if len(labels) < min_labels:
        reasons.append("teacher_labels_below_floor")
    if relevant < min_per_class or irrelevant < min_per_class:
        reasons.append("teacher_class_below_floor")
    if any(
        row.get("negative_veto_conflict") is not None
        and row.get("negative_veto_conflict") is not False
        for row in rows
    ):
        reasons.append("negative_veto_conflict")

    identities = {
        name: _one([_text(row.get(name)) for row in completed], name, reasons)
        for name in (
            "profile",
            "cohort",
            "route",
            "model_digest",
            "prompt_sha256",
            "schema_sha256",
            "split_plan_id",
            "profile_contract_id",
        )
    }
    if identities["profile"] and identities["profile"] != profile:
        reasons.append("profile_identity_mismatch")
    if identities["cohort"] and identities["cohort"] != cohort:
        reasons.append("cohort_identity_mismatch")
    if profile_contract_id and identities["profile_contract_id"] != profile_contract_id:
        reasons.append("profile_contract_mismatch")
    if identities["route"] and identities["route"] != "opencode-go/ox-alpha-free":
        reasons.append("route_mismatch")
    for name in (
        "model_digest",
        "prompt_sha256",
        "schema_sha256",
        "split_plan_id",
        "profile_contract_id",
    ):
        if identities[name] and re.fullmatch(r"[0-9a-f]{64}", identities[name]) is None:
            reasons.append(f"{name}_identity_invalid")
    if any(row.get("route_identity_exact") is not True for row in completed):
        reasons.append("route_identity_mismatch")
    if any(row.get("fixed_split_plan") is not True for row in completed):
        reasons.append("fixed_split_plan_missing")

    if len(eligible) != len(completed):
        reasons.append("split_assignment_invalid")
    split_rows = [row for row in completed if row.get("split") in required_splits]
    if {str(row.get("split")) for row in split_rows} != required_splits:
        reasons.append("chronological_split_incomplete")
    missing_group = any(not _text(row.get("group_id")) for row in completed)
    missing_time = any(not _text(row.get("as_of")) for row in completed)
    if missing_group:
        reasons.append("group_identity_missing")
    if any(row.get("group_identity_exact") is not True for row in completed):
        reasons.append("group_identity_mismatch")
    if missing_time:
        reasons.append("chronological_timestamp_missing")
    groups: dict[str, set[str]] = defaultdict(set)
    for row in split_rows:
        group = _text(row.get("group_id"))
        if group:
            groups[group].add(str(row["split"]))
    if any(len(splits) > 1 for splits in groups.values()):
        reasons.append("group_split_leakage")
    times = {
        split: [_text(row.get("as_of")) for row in split_rows if row["split"] == split]
        for split in required_splits
    }
    if (
        all(times.values())
        and not missing_time
        and not (max(times["train"]) < min(times["validation"]) < min(times["test"]))
    ):
        reasons.append("chronological_order_invalid")

    completed_flags = completed
    if any("future_leakage" not in row for row in completed_flags):
        reasons.append("future_leakage_flag_missing")
    if any(
        "future_leakage" in row and row.get("future_leakage") is not False
        for row in completed_flags
    ):
        reasons.append("future_leakage_detected")
    if any("feature_parity" not in row for row in completed_flags):
        reasons.append("feature_parity_flag_missing")
    if any(row.get("feature_parity") is not True for row in completed_flags):
        reasons.append("feature_parity_failed")

    test_rows = [row for row in completed if row.get("split") == "test"]
    evidence_refs = sorted(
        {
            _text(row.get("locked_test_evidence_ref"))
            for row in test_rows
            if _text(row.get("locked_test_evidence_ref"))
        }
    )
    if not test_rows or any(
        row.get("locked_test_read_only") is not True
        or _text(row.get("locked_test_evidence_ref"))
        != f"split-plan:{_text(row.get('split_plan_id'))}"
        for row in test_rows
    ):
        reasons.append("locked_test_read_only_evidence_missing")

    pairs: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    incomplete_pairs = 0
    for row in probes:
        pair_id = _text(row.get("repeat_pair_id"))
        if pair_id:
            pairs[pair_id].append(row)
        else:
            incomplete_pairs += 1
    complete_pairs: list[list[Mapping[str, Any]]] = []
    pair_identity_mismatch = False
    for pair in pairs.values():
        orders = {row.get("blind_order") for row in pair}
        targets = {
            (_text(row.get("rally_id")), _text(row.get("candidate_id"))) for row in pair
        }
        identity_exact = len(targets) == 1 and all(next(iter(targets), ()))
        pair_identity_mismatch = pair_identity_mismatch or not identity_exact
        valid = (
            len(pair) == 2
            and all(row.get("fixed_repeat") is True for row in pair)
            and all(row.get("order_swap") is True for row in pair)
            and orders == {"a_first", "b_first"}
            and identity_exact
        )
        if valid:
            complete_pairs.append(pair)
        else:
            incomplete_pairs += 1
    stable = sum(
        pair[0].get("verdict") == pair[1].get("verdict") for pair in complete_pairs
    )
    stability = _wilson_lower(stable, len(complete_pairs))
    if incomplete_pairs:
        reasons.append("blind_repeat_pair_incomplete")
    if pair_identity_mismatch:
        reasons.append("blind_repeat_identity_mismatch")
    if len(complete_pairs) < min_repeat_pairs:
        reasons.append("blind_repeat_pairs_below_floor")
    if stability < min_repeat_stability:
        reasons.append("blind_repeat_stability_below_gate")

    return {
        "schema": "chronovisor.recall-single-teacher-gate.v1",
        "truth_authority": "teacher_only_not_verified",
        "profile": profile,
        "cohort": cohort,
        "passed": not reasons,
        "reasons": sorted(set(reasons)),
        "labels": {
            "eligible": len(labels),
            "relevant": relevant,
            "irrelevant": irrelevant,
            "excluded": excluded,
        },
        "excluded_prior_contract_rows": excluded_prior_contract_rows,
        "identity": identities,
        "chronological_grouped": {
            "splits": sorted({str(row.get("split")) for row in split_rows}),
            "groups": len(groups),
        },
        "locked_test": {
            "rows": len(test_rows),
            "read_only": not any(
                row.get("locked_test_read_only") is not True for row in test_rows
            ),
            "evidence_refs": evidence_refs,
        },
        "blind_repeat": {
            "complete_pairs": len(complete_pairs),
            "incomplete_pairs": incomplete_pairs,
            "stable": stable,
            "wilson_lower": round(stability, 8),
        },
    }
