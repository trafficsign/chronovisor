from __future__ import annotations

from chronovisor.classification_direct_decision_worker import (
    DECISION_SCHEMA,
    HOLD,
    validate_direct_decision,
)


def _candidates() -> list[dict[str, str]]:
    return [
        {
            "notation": "629.02",
            "label_en": "Vehicle structure",
            "label_ja": "車両構造",
        },
        {"notation": "004.4", "label_en": "Software", "label_ja": "ソフトウェア"},
    ]


def test_direct_decision_accepts_inclusive_broader_shelf() -> None:
    decision = validate_direct_decision(
        {
            "central_subject": "Vehicle mudguard attachment design",
            "principal_class": "6",
            "disposition": "assign",
            "selected_notation": "629.02",
            "rationale": "A mudguard is a specific vehicle structural part.",
        },
        _candidates(),
    )

    assert decision["schema"] == DECISION_SCHEMA
    assert decision["selected_notation"] == "629.02"
    assert decision["invalid_reason"] == ""


def test_direct_decision_fails_closed_on_principal_class_mismatch() -> None:
    decision = validate_direct_decision(
        {
            "central_subject": "Vehicle mudguard attachment design",
            "principal_class": "0",
            "disposition": "assign",
            "selected_notation": "629.02",
            "rationale": "Inconsistent class.",
        },
        _candidates(),
    )

    assert decision["disposition"] == "hold"
    assert decision["selected_notation"] == HOLD
    assert decision["invalid_reason"] == "principal_class_veto"
