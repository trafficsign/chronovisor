from __future__ import annotations

from chronovisor.lab.classification_decision_trial import (
    MINIMUM_CORRECT,
    decision_gate_passed,
    score_decision,
)
from chronovisor.classification.classification_decision_worker import (
    DECISION_SCHEMA,
    HOLD,
    validate_decision,
)


def _candidates() -> list[dict[str, str]]:
    return [
        {"notation": "004.4", "label_en": "Software", "label_ja": "ソフトウェア"},
        {
            "notation": "629.02",
            "label_en": "Vehicle structure",
            "label_ja": "車両構造",
        },
    ]


def test_validator_accepts_supported_candidate() -> None:
    decision = validate_decision(
        {
            "assessments": [
                {
                    "notation": "004.4",
                    "support": "yes",
                    "evidence": "direct",
                    "reason": "The document is about software.",
                },
                {
                    "notation": "629.02",
                    "support": "no",
                    "evidence": "contradicted",
                    "reason": "No vehicle subject is present.",
                },
            ],
            "principal_class": "0",
            "disposition": "assign",
            "selected_notation": "004.4",
            "specificity_safe": True,
            "rationale": "Software is the principal subject.",
        },
        _candidates(),
    )

    assert decision["schema"] == DECISION_SCHEMA
    assert decision["disposition"] == "assign"
    assert decision["selected_notation"] == "004.4"
    assert decision["invalid_reason"] == ""


def test_validator_fails_closed_on_candidate_coverage_mismatch() -> None:
    decision = validate_decision(
        {
            "assessments": [
                {
                    "notation": "004.4",
                    "support": "yes",
                    "evidence": "direct",
                    "reason": "Software.",
                },
                {
                    "notation": "004.4",
                    "support": "yes",
                    "evidence": "direct",
                    "reason": "Duplicate.",
                },
            ],
            "principal_class": "0",
            "disposition": "assign",
            "selected_notation": "004.4",
            "specificity_safe": True,
            "rationale": "Invalid duplicate coverage.",
        },
        _candidates(),
    )

    assert decision["disposition"] == "hold"
    assert decision["selected_notation"] == HOLD
    assert decision["invalid_reason"] == "candidate_assessment_coverage_mismatch"


def test_validator_applies_principal_class_veto() -> None:
    decision = validate_decision(
        {
            "assessments": [
                {
                    "notation": "004.4",
                    "support": "yes",
                    "evidence": "direct",
                    "reason": "Software.",
                },
                {
                    "notation": "629.02",
                    "support": "no",
                    "evidence": "none",
                    "reason": "Not supported.",
                },
            ],
            "principal_class": "6",
            "disposition": "assign",
            "selected_notation": "004.4",
            "specificity_safe": True,
            "rationale": "Inconsistent principal class.",
        },
        _candidates(),
    )

    assert decision["disposition"] == "hold"
    assert decision["invalid_reason"] == "principal_class_veto"


def test_score_and_gate_require_accuracy_without_catastrophic_errors() -> None:
    assert score_decision(
        {"disposition": "assign", "selected_notation": "629.02"},
        ["629"],
    ) == {
        "held": False,
        "correct": True,
        "incorrect": False,
        "catastrophic": False,
    }
    assert score_decision(
        {"disposition": "assign", "selected_notation": "004.4"},
        ["629.02"],
    )["catastrophic"]
    assert decision_gate_passed(
        {
            "correct": MINIMUM_CORRECT,
            "holds": 6,
            "catastrophic": 0,
        }
    )
    assert not decision_gate_passed(
        {
            "correct": MINIMUM_CORRECT,
            "holds": 5,
            "catastrophic": 1,
        }
    )
