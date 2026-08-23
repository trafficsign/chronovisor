from __future__ import annotations

from typing import cast

from chronovisor.recall.recall_distillation_single_teacher_gate import (
    evaluate_single_teacher_gate,
)

PROFILE = "ox-alpha-single-v1"
COHORT = "ox-alpha-backfill-v1"


def _row(
    row_id: str,
    *,
    split: str,
    verdict: str,
    probe: bool = False,
    group_id: str | None = None,
    **override: object,
) -> dict[str, object]:
    as_of = {"train": "2026-08-01", "validation": "2026-08-02", "test": "2026-08-03"}[
        split
    ]
    return {
        "row_id": row_id,
        "rally_id": row_id,
        "candidate_id": row_id,
        "status": "completed",
        "verdict": verdict,
        "probe": probe,
        "profile": PROFILE,
        "cohort": COHORT,
        "route": "opencode-go/ox-alpha-free",
        "model_digest": "a" * 64,
        "prompt_sha256": "b" * 64,
        "schema_sha256": "c" * 64,
        "profile_contract_id": "e" * 64,
        "route_identity_exact": True,
        "split": split,
        "split_plan_id": "d" * 64,
        "fixed_split_plan": True,
        "group_id": group_id or row_id,
        "group_identity_exact": True,
        "as_of": as_of,
        "future_leakage": False,
        "feature_parity": True,
        "locked_test_read_only": split == "test",
        "locked_test_evidence_ref": f"split-plan:{'d' * 64}" if split == "test" else "",
        **override,
    }


def _passing_rows() -> list[dict[str, object]]:
    rows = [
        _row("train-r", split="train", verdict="relevant"),
        _row("train-i", split="train", verdict="irrelevant"),
        _row("validation-r", split="validation", verdict="relevant"),
        _row("validation-i", split="validation", verdict="irrelevant"),
    ]
    for pair_id, verdict in (("one", "relevant"), ("two", "irrelevant")):
        rows.extend(
            [
                _row(
                    f"{pair_id}-a",
                    split="test",
                    verdict=verdict,
                    probe=True,
                    group_id=f"repeat-{pair_id}",
                    rally_id=f"rally-{pair_id}",
                    candidate_id=f"candidate-{pair_id}",
                    repeat_pair_id=pair_id,
                    fixed_repeat=True,
                    order_swap=True,
                    blind_order="a_first",
                ),
                _row(
                    f"{pair_id}-b",
                    split="test",
                    verdict=verdict,
                    probe=True,
                    group_id=f"repeat-{pair_id}",
                    rally_id=f"rally-{pair_id}",
                    candidate_id=f"candidate-{pair_id}",
                    repeat_pair_id=pair_id,
                    fixed_repeat=True,
                    order_swap=True,
                    blind_order="b_first",
                ),
            ]
        )
    return rows


def _gate(rows: list[dict[str, object]]) -> dict[str, object]:
    return evaluate_single_teacher_gate(
        rows,
        profile=PROFILE,
        cohort=COHORT,
        min_labels=4,
        min_per_class=2,
        min_repeat_pairs=2,
        min_repeat_stability=0.10,
    )


def test_single_teacher_gate_accepts_fixed_chronological_cohort() -> None:
    gate = _gate(_passing_rows())

    assert gate["passed"] is True
    assert gate["truth_authority"] == "teacher_only_not_verified"
    assert gate["identity"]["route"] == "opencode-go/ox-alpha-free"
    assert gate["locked_test"] == {
        "rows": 4,
        "read_only": True,
        "evidence_refs": [f"split-plan:{'d' * 64}"],
    }
    assert gate["blind_repeat"]["complete_pairs"] == 2


def test_invalid_uncertain_and_retry_rows_are_excluded_from_label_counts() -> None:
    rows = _passing_rows()
    rows.extend(
        [
            _row("invalid", split="train", verdict="relevant", status="invalid"),
            _row("uncertain", split="train", verdict="relevant", status="uncertain"),
            _row(
                "retry",
                split="train",
                verdict="relevant",
                status="retry",
                error_class="invalid_teacher_output",
            ),
        ]
    )

    gate = _gate(rows)

    assert gate["passed"] is True
    assert gate["labels"] == {
        "eligible": 4,
        "relevant": 2,
        "irrelevant": 2,
        "excluded": 3,
    }


def test_negative_veto_conflict_fails_even_when_other_gates_pass() -> None:
    rows = _passing_rows()
    rows[0]["negative_veto_conflict"] = True

    gate = _gate(rows)

    assert gate["passed"] is False
    assert gate["reasons"] == ["negative_veto_conflict"]


def test_gate_fails_closed_for_repeat_identity_and_parity_contracts() -> None:
    rows = _passing_rows()
    rows[4]["blind_order"] = "b_first"
    rows[0]["route"] = "different-route"
    rows[1].pop("future_leakage")
    rows[2]["feature_parity"] = False
    rows[3].pop("schema_sha256")
    rows[5]["locked_test_read_only"] = False
    rows[4]["group_id"] = rows[0]["group_id"]
    rows[0]["fixed_split_plan"] = False
    rows[1]["route_identity_exact"] = False
    rows[2]["group_identity_exact"] = False

    gate = _gate(rows)

    assert gate["passed"] is False
    assert gate["reasons"] == sorted(gate["reasons"])
    assert {
        "blind_repeat_pair_incomplete",
        "blind_repeat_pairs_below_floor",
        "group_split_leakage",
        "group_identity_mismatch",
        "future_leakage_flag_missing",
        "feature_parity_failed",
        "fixed_split_plan_missing",
        "locked_test_read_only_evidence_missing",
        "route_identity_not_exactly_one",
        "route_identity_mismatch",
        "schema_sha256_identity_not_exactly_one",
    } <= set(gate["reasons"])


def test_gate_rejects_noncanonical_identity_values_and_invalid_rows() -> None:
    expected = {
        "route": ("evil-route", "route_mismatch"),
        "model_digest": ("not-a-digest", "model_digest_identity_invalid"),
        "prompt_sha256": ("x", "prompt_sha256_identity_invalid"),
        "schema_sha256": ("y", "schema_sha256_identity_invalid"),
        "profile_contract_id": ("z", "profile_contract_id_identity_invalid"),
    }
    for field, (value, reason) in expected.items():
        rows = _passing_rows()
        for row in rows:
            row[field] = value
        assert reason in _gate(rows)["reasons"]

    invalid = cast(list[dict[str, object]], [None])
    assert _gate(invalid)["reasons"] == sorted(_gate(invalid)["reasons"])
    assert "input_row_invalid" in _gate(invalid)["reasons"]

    mismatched_pair = _passing_rows()
    mismatched_pair[4]["candidate_id"] = "different-candidate"
    mismatch_reasons = _gate(mismatched_pair)["reasons"]
    assert "blind_repeat_identity_mismatch" in mismatch_reasons
    assert "blind_repeat_pair_incomplete" in mismatch_reasons

    malformed_boolean = _passing_rows()
    malformed_boolean[0]["future_leakage"] = 0
    assert "future_leakage_detected" in _gate(malformed_boolean)["reasons"]

    malformed_veto = _passing_rows()
    malformed_veto[0]["negative_veto_conflict"] = "true"
    assert "negative_veto_conflict" in _gate(malformed_veto)["reasons"]


def test_gate_excludes_embargo_rows_from_quality_floors() -> None:
    rows = _passing_rows()
    rows.extend(
        {
            **rows[0],
            "row_id": f"embargo-{index}",
            "rally_id": f"embargo-{index}",
            "candidate_id": f"embargo-{index}",
            "group_id": f"embargo-{index}",
            "split": "embargo",
            "verdict": "relevant" if index % 2 == 0 else "irrelevant",
        }
        for index in range(20)
    )

    gate = _gate(rows)

    assert gate["passed"] is False
    assert "split_assignment_invalid" in gate["reasons"]
    assert gate["labels"] == {
        "eligible": 4,
        "relevant": 2,
        "irrelevant": 2,
        "excluded": 20,
    }
