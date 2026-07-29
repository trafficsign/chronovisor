from __future__ import annotations

from chronovisor.classification import default_udc_package
from chronovisor.classification_hierarchy import build_navigation_graph
from chronovisor.classification_hierarchy_dev import (
    _requires_adjudication,
    score_selection,
)


def test_root_probe_disagreement_requires_adjudication() -> None:
    assert _requires_adjudication(
        {"action": "descend", "selected_notations": ["0"]},
        {"action": "descend", "selected_notations": ["3"]},
    )
    assert not _requires_adjudication(
        {"action": "descend", "selected_notations": ["0"]},
        {"action": "descend", "selected_notations": ["0"]},
    )


def test_hierarchy_score_distinguishes_ancestor_sibling_and_root_error() -> None:
    graph = build_navigation_graph(default_udc_package())

    assert score_selection(graph, "004", ["004.4"])["relation"] == "ancestor"
    assert (
        score_selection(graph, "004.3", ["004.4"])["relation"]
        == "same-parent-sibling"
    )
    assert (
        score_selection(graph, "543.6", ["004.4"])["relation"]
        == "catastrophic-root"
    )
