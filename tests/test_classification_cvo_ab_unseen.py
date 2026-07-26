from __future__ import annotations

from chronovisor.classification_anchor import UNRESOLVED_ANCHOR_ID
from chronovisor.classification_controlled_vocabulary import (
    load_controlled_vocabulary,
)
from chronovisor.classification_cvo_ab_unseen import _arm_passed


def _passing_metrics() -> dict[str, int | float]:
    return {
        "case_count": 40,
        "exact_sets": 36,
        "semantic_coverage_cases": 38,
        "excess_anchor_rate": 0.10,
        "missing_anchor_rate": 0.10,
        "dual_assignment_rate": 0.40,
        "holds": 2,
        "major_errors": 0,
    }


def test_controlled_vocabulary_is_frozen_and_fully_audited() -> None:
    vocabulary = load_controlled_vocabulary()

    assert vocabulary.epoch == "cvo-controlled-vocabulary-v1"
    assert vocabulary.checksum.startswith("sha256:")
    assert len(vocabulary.terms) == 60
    assert (
        vocabulary.derivation["corpus_snapshot"]["specific_in_domain_terms"]
        == 25
    )
    assert (
        vocabulary.derivation["page_label_associations_used"] is False
    )
    assert (
        vocabulary.derivation["literal_regex_matching_used"] is False
    )
    assert {
        term.source_kind for term in vocabulary.terms
    } >= {"raw-keyword", "wikilink", "tag"}
    assert len(vocabulary.model_cards()) == 60
    assert (
        sum(
            card["id"] == UNRESOLVED_ANCHOR_ID
            for card in vocabulary.model_cards()
        )
        == 1
    )
    assert {entry["surface"] for entry in vocabulary.ambiguous_registry} >= {
        "採用",
        "memory",
        "analysis",
        "lock",
    }
    assert all(term.source_ids for term in vocabulary.terms)


def test_controlled_terms_map_deterministically_to_max_two_anchors() -> None:
    vocabulary = load_controlled_vocabulary()

    assert vocabulary.anchors_for_terms(
        [
            "cvo:term:software-implementation",
            "cvo:term:data-pipeline-format",
        ]
    ) == ["cvo:anchor:0001"]
    assert vocabulary.anchors_for_terms(
        [
            "cvo:term:career-recruitment",
            "cvo:term:defense-geopolitics",
        ]
    ) == ["cvo:anchor:0008", "cvo:anchor:0024"]
    assert vocabulary.anchors_for_terms(
        [
            "cvo:term:software-implementation",
            "cvo:term:career-recruitment",
            "cvo:term:defense-geopolitics",
        ]
    ) == [UNRESOLVED_ANCHOR_ID]
    assert vocabulary.anchors_for_terms(
        ["cvo:term:vocabulary-gap"]
    ) == [UNRESOLVED_ANCHOR_ID]


def test_two_arm_gate_is_fail_closed() -> None:
    passing = _passing_metrics()
    assert _arm_passed(passing)

    for field, value in (
        ("exact_sets", 35),
        ("semantic_coverage_cases", 37),
        ("excess_anchor_rate", 0.1001),
        ("missing_anchor_rate", 0.1001),
        ("dual_assignment_rate", 0.4001),
        ("holds", 3),
        ("major_errors", 1),
    ):
        failing = dict(passing)
        failing[field] = value
        assert not _arm_passed(failing)
