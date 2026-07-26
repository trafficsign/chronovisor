from __future__ import annotations

import json

import pytest

from chronovisor.classification import ClassificationError
from chronovisor.classification_anchor import UNRESOLVED_ANCHOR_ID, load_anchor_set
from chronovisor.classification_anchor_second_auditor import validate_audit
from chronovisor.classification_anchor_set_dev import (
    score_anchor_set,
    summarize_metrics,
)
from chronovisor.classification_anchor_set_unseen import (
    GOLD_SCHEMA,
    OUTPUT_CONTRACT_EPOCH,
    _load_manual_gold,
)
from chronovisor.classification_anchor_set_worker import validate_selection
from chronovisor.classification_intent_lexicon import classify_complement


def test_set_worker_normalizes_order_and_unresolved_contract() -> None:
    ids = ["cvo:anchor:0001", "cvo:anchor:0002", UNRESOLVED_ANCHOR_ID]
    result = validate_selection(
        {
            "anchor_ids": ["cvo:anchor:0002", "cvo:anchor:0001"],
            "rationale": "Both subjects are independently central.",
        },
        ids,
    )
    assert result["anchor_ids"] == ["cvo:anchor:0001", "cvo:anchor:0002"]

    held = validate_selection(
        {
            "anchor_ids": [UNRESOLVED_ANCHOR_ID, "cvo:anchor:0001"],
            "rationale": "Invalid mixed unresolved result.",
        },
        ids,
    )
    assert held["anchor_ids"] == [UNRESOLVED_ANCHOR_ID]


def test_set_score_separates_partial_excess_and_major_error() -> None:
    exact = score_anchor_set(
        ["cvo:anchor:0001", "cvo:anchor:0003"],
        ["cvo:anchor:0003", "cvo:anchor:0001"],
        ["cvo:anchor:0001", "cvo:anchor:0003"],
    )
    assert exact["exact_set"]
    assert not exact["major_error"]

    valid_singleton = score_anchor_set(
        ["cvo:anchor:0001"],
        ["cvo:anchor:0001", "cvo:anchor:0003"],
        ["cvo:anchor:0001", "cvo:anchor:0003"],
    )
    assert valid_singleton["exact_set"]
    assert valid_singleton["missing_anchor_ids"] == []

    explicit_alternative = score_anchor_set(
        ["cvo:anchor:0002"],
        ["cvo:anchor:0001"],
        ["cvo:anchor:0001", "cvo:anchor:0002"],
        [["cvo:anchor:0001"], ["cvo:anchor:0002"]],
    )
    assert explicit_alternative["exact_set"]
    assert explicit_alternative["nearest_acceptable_anchor_set"] == [
        "cvo:anchor:0002"
    ]

    defensible_partial = score_anchor_set(
        ["cvo:anchor:0001", "cvo:anchor:0002"],
        ["cvo:anchor:0001"],
        ["cvo:anchor:0001", "cvo:anchor:0002"],
    )
    assert defensible_partial["partial_set"]
    assert defensible_partial["excess_anchor_ids"] == ["cvo:anchor:0002"]
    assert not defensible_partial["major_error"]

    major = score_anchor_set(
        ["cvo:anchor:0001", "cvo:anchor:0018"],
        ["cvo:anchor:0001"],
        ["cvo:anchor:0001", "cvo:anchor:0002"],
    )
    assert major["major_error"]
    assert major["indefensible_anchor_ids"] == ["cvo:anchor:0018"]


def test_metrics_count_direct_selected_anchor_ids() -> None:
    case = {
        "selected_anchor_ids": ["cvo:anchor:0001", "cvo:anchor:0002"],
        **score_anchor_set(
            ["cvo:anchor:0001", "cvo:anchor:0002"],
            ["cvo:anchor:0001"],
            ["cvo:anchor:0001", "cvo:anchor:0002"],
            [["cvo:anchor:0001"]],
        ),
    }
    metrics = summarize_metrics([case])
    assert metrics["selected_anchor_count"] == 2
    assert metrics["excess_anchor_count"] == 1
    assert metrics["excess_anchor_rate"] == 0.5


def test_second_anchor_requires_every_admission_axis() -> None:
    anchors = ["cvo:anchor:0001", "cvo:anchor:0002"]
    rejected = validate_audit(
        {
            "second_anchor_id": "cvo:anchor:0002",
            "independent_principal_subject": True,
            "not_subsumed_by_core": True,
            "not_incidental_context": False,
            "explicit_document_evidence": True,
            "rationale": "AI is only implementation context.",
        },
        core_anchor_id="cvo:anchor:0001",
        anchor_ids=anchors,
    )
    assert not rejected["admitted"]
    assert rejected["second_anchor_id"] == "NONE"

    admitted = validate_audit(
        {
            "second_anchor_id": "cvo:anchor:0002",
            "independent_principal_subject": True,
            "not_subsumed_by_core": True,
            "not_incidental_context": True,
            "explicit_document_evidence": True,
            "rationale": "The document independently analyzes both domains.",
        },
        core_anchor_id="cvo:anchor:0001",
        anchor_ids=anchors,
    )
    assert admitted["admitted"]


def test_unseen_gold_requires_explicit_sets_inside_defensible_scope(
    tmp_path,
) -> None:
    anchor_set = load_anchor_set()
    gold_path = tmp_path / "gold.json"
    payload = {
        "schema": GOLD_SCHEMA,
        "anchor_epoch": anchor_set.epoch,
        "output_contract_epoch": OUTPUT_CONTRACT_EPOCH,
        "fixture_status": "sealed-unseen-before-inference",
        "cases": [
            {
                "uid": "case-1",
                "acceptable_anchor_sets": [
                    ["cvo:anchor:0001", "cvo:anchor:0002"]
                ],
                "defensible_anchor_ids": [
                    "cvo:anchor:0001",
                    "cvo:anchor:0002",
                ],
            }
        ],
    }
    gold_path.write_text(json.dumps(payload), encoding="utf-8")
    gold = _load_manual_gold(gold_path, anchor_set)
    assert gold["case-1"]["acceptable_sets"] == [
        ["cvo:anchor:0001", "cvo:anchor:0002"]
    ]

    payload["cases"][0]["defensible_anchor_ids"] = ["cvo:anchor:0001"]
    gold_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ClassificationError):
        _load_manual_gold(gold_path, anchor_set)


def test_intent_lexicon_is_high_precision_and_core_aware() -> None:
    career = classify_complement(
        {
            "title": "Defense Engineering Interview Preparation",
            "summary": "",
        },
        core_anchor_id="cvo:anchor:0024",
    )
    assert career["second_anchor_id"] == "cvo:anchor:0008"

    core_suppressed = classify_complement(
        {
            "title": "Career Strategy",
            "summary": "",
        },
        core_anchor_id="cvo:anchor:0008",
    )
    assert core_suppressed["second_anchor_id"] == "NONE"

    implementation_only = classify_complement(
        {
            "title": "LLM Wiki Frontier Review Architecture and Call Sites",
            "summary": "",
        },
        core_anchor_id="cvo:anchor:0001",
    )
    assert implementation_only["second_anchor_id"] == "NONE"

    career_purpose = classify_complement(
        {
            "title": "Defense Engineering Profile",
            "summary": "",
            "excerpt": (
                "This serves as a strategic reference for evaluating this "
                "specific site as a career option."
            ),
        },
        core_anchor_id="cvo:anchor:0024",
    )
    assert career_purpose["second_anchor_id"] == "cvo:anchor:0008"
