from __future__ import annotations

from chronovisor.classification.classification import default_udc_package
from chronovisor.classification.classification_hierarchy import (
    ROOT_NOTATIONS,
    build_navigation_graph,
    deterministic_evidence_capsule,
    is_primary_navigation_concept,
)


def test_navigation_graph_preserves_ranges_and_contracts_auxiliary_headers() -> None:
    package = default_udc_package()
    graph = build_navigation_graph(package)

    assert len(graph.nodes) == 1_850
    assert tuple(node.notation for node in graph.children(None)) == ROOT_NOTATIONS
    assert graph.by_notation("005.95/.96") is not None
    assert graph.by_notation("355/359") is not None
    assert graph.by_notation("004.01/.08") is None
    assert graph.by_notation("004.05").parent_uri == graph.by_notation("004").uri
    assert graph.contracted_parent_count == 186
    # Contracting auxiliary headers lifts their children into the nearest
    # primary parent and raises the observed maximum from 25 to 31.
    assert max(len(node.children_uris) for node in graph.nodes.values()) == 31


def test_auxiliary_notations_are_not_primary_navigation_concepts() -> None:
    package = default_udc_package()

    assert not is_primary_navigation_concept(package.by_notation("(075)"))
    assert not is_primary_navigation_concept(package.by_notation("-057.1"))
    assert not is_primary_navigation_concept(package.by_notation("004.01/.08"))
    assert is_primary_navigation_concept(package.by_notation("005.95/.96"))


def test_evidence_capsule_removes_frontmatter_and_duplicate_title() -> None:
    capsule = deterministic_evidence_capsule(
        {
            "title": "Memory Reflection",
            "summary": "A journal about knowledge retention.",
            "excerpt": (
                "---\ntitle: Memory Reflection\n---\n"
                "# Memory Reflection\n\n"
                "This page records lessons from prior work.\n"
                "## Details\nIt is not a hardware memory benchmark."
            ),
        }
    )

    assert capsule["title"] == "Memory Reflection"
    assert "title:" not in capsule["evidence_excerpt"]
    assert "This page records lessons" in capsule["evidence_excerpt"]
