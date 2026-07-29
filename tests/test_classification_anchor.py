from __future__ import annotations

import json

from chronovisor.classification.classification_anchor import (
    UNRESOLVED_ANCHOR_ID,
    default_anchor_gold_path,
    load_anchor_set,
    validate_anchor_gold,
)
from chronovisor.classification.classification_anchor_worker import (
    SELECTION_SCHEMA,
    validate_selection,
)


def test_anchor_set_is_complete_and_crosswalk_is_not_model_input() -> None:
    anchor_set = load_anchor_set()

    assert len(anchor_set.anchors) == 34
    assert UNRESOLVED_ANCHOR_ID in anchor_set.by_id
    assert all("udc_scope" not in card for card in anchor_set.model_cards())


def test_dev_gold_covers_unique_known_anchor_ids() -> None:
    anchor_set = load_anchor_set()
    payload = json.loads(default_anchor_gold_path().read_text(encoding="utf-8"))
    expected_uids = [str(row["uid"]) for row in payload["cases"]]

    gold = validate_anchor_gold(payload, anchor_set, expected_uids)

    assert len(gold) == 40


def test_anchor_selection_normalizes_secondary_and_unresolved() -> None:
    anchor_ids = ["cvo:anchor:0001", "cvo:anchor:0002", UNRESOLVED_ANCHOR_ID]
    selected = validate_selection(
        {
            "primary_anchor_id": "cvo:anchor:0001",
            "secondary_anchor_ids": [
                "cvo:anchor:0001",
                "cvo:anchor:0002",
            ],
            "rationale": "Software is primary and AI is secondary.",
        },
        anchor_ids,
    )

    assert selected["schema"] == SELECTION_SCHEMA
    assert selected["secondary_anchor_ids"] == ["cvo:anchor:0002"]

    held = validate_selection(
        {
            "primary_anchor_id": UNRESOLVED_ANCHOR_ID,
            "secondary_anchor_ids": ["cvo:anchor:0001"],
            "rationale": "No anchor fits.",
        },
        anchor_ids,
    )
    assert held["secondary_anchor_ids"] == []
