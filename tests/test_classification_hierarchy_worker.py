from __future__ import annotations

from chronovisor.classification.classification_hierarchy_worker import (
    AUDIT_SCHEMA,
    HOLD,
    STEP_STOP,
    STEP_SCHEMA,
    validate_audit,
    validate_step,
)


def test_step_accepts_two_explicit_branches() -> None:
    step = validate_step(
        {
            "selected_notations": ["005", "331"],
            "corrected_central_subject": "",
            "rationale": "Management and labour are both principal aspects.",
        },
        ["004", "005", "331"],
    )

    assert step["schema"] == STEP_SCHEMA
    assert step["selected_notations"] == ["005", "331"]
    assert step["invalid_reason"] == ""


def test_invalid_step_fails_closed_to_stop() -> None:
    step = validate_step(
        {
            "selected_notations": ["543.6"],
            "rationale": "Outside the offered siblings.",
        },
        ["004.4", "004.5"],
    )

    assert step["action"] == "stop"
    assert step["selected_notations"] == []
    assert step["invalid_reason"] == "invalid_navigation_action"


def test_explicit_stop_token_normalizes_to_parent_stop() -> None:
    step = validate_step(
        {
            "selected_notations": [STEP_STOP],
            "corrected_central_subject": "",
            "rationale": "The current parent is the deepest justified shelf.",
        },
        ["004.4", "004.5"],
    )

    assert step["action"] == "stop"
    assert step["selected_notations"] == []


def test_audit_cannot_jump_to_unexplored_sibling() -> None:
    audit = validate_audit(
        {"selected_notation": "543.6", "rationale": "Wrong branch."},
        ["0", "004", "004.4"],
    )

    assert audit["schema"] == AUDIT_SCHEMA
    assert audit["selected_notation"] == HOLD
    assert audit["invalid_reason"] == "audit_left_explored_paths"


def test_root_only_audit_is_hold() -> None:
    audit = validate_audit(
        {"selected_notation": "0", "rationale": "Only the root is defensible."},
        ["0", "004", "004.4"],
    )

    assert audit["selected_notation"] == HOLD
    assert audit["invalid_reason"] == "root_equivalent_hold"
