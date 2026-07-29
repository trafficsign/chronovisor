from __future__ import annotations

from chronovisor.classification.classification_anchor import (
    UNRESOLVED_ANCHOR_ID,
    load_anchor_set,
)
from chronovisor.lab.classification_anchor_dev import score_anchor_selection


def test_anchor_score_distinguishes_exact_hold_related_and_catastrophic() -> None:
    anchor_set = load_anchor_set()

    exact = score_anchor_selection(
        anchor_set,
        "cvo:anchor:0001",
        [],
        ["cvo:anchor:0001"],
    )
    assert exact["relation"] == "exact"
    assert exact["exact"]

    held = score_anchor_selection(
        anchor_set,
        UNRESOLVED_ANCHOR_ID,
        [],
        ["cvo:anchor:0001"],
    )
    assert held["relation"] == "hold"
    assert held["held"]

    related = score_anchor_selection(
        anchor_set,
        "cvo:anchor:0002",
        [],
        ["cvo:anchor:0001"],
    )
    assert related["relation"] == "related-family"
    assert related["related_error"]

    catastrophic = score_anchor_selection(
        anchor_set,
        "cvo:anchor:0011",
        [],
        ["cvo:anchor:0001"],
    )
    assert catastrophic["relation"] == "catastrophic-family"
    assert catastrophic["catastrophic"]


def test_anchor_score_records_secondary_rescue_without_passing_primary() -> None:
    anchor_set = load_anchor_set()

    score = score_anchor_selection(
        anchor_set,
        "cvo:anchor:0002",
        ["cvo:anchor:0001"],
        ["cvo:anchor:0001"],
    )

    assert not score["exact"]
    assert score["secondary_rescue"]
