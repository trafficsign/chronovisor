from __future__ import annotations

from chronovisor.classification_collection_pilot import (
    _collection_from_path,
    audit_native_collections,
)


def test_collection_from_path_is_fail_closed() -> None:
    assert _collection_from_path("pages/career/a.md") == "career"
    assert _collection_from_path("pages//a.md") == ""
    assert _collection_from_path("system/current-state.md") == ""
    assert _collection_from_path("../pages/career/a.md") == ""


def test_collection_audit_measures_coverage_and_link_locality() -> None:
    registry = {
        "pages": {
            "u1": {
                "uid": "u1",
                "page_id": "a",
                "path": "pages/career/a.md",
                "status": "active",
            },
            "u2": {
                "uid": "u2",
                "page_id": "b",
                "path": "pages/career/b.md",
                "status": "active",
            },
            "u3": {
                "uid": "u3",
                "page_id": "c",
                "path": "pages/ai/c.md",
                "status": "active",
            },
            "u4": {
                "uid": "u4",
                "page_id": "",
                "path": "pages/ai/missing.md",
                "status": "active",
            },
            "system": {
                "uid": "system",
                "page_id": "system",
                "path": "system/current-state.md",
                "status": "active",
            },
        }
    }
    index = {
        "entries": {
            "a": {"outlinks": ["b", "c", "missing"]},
            "b": {"outlinks": ["a"]},
            "c": {"outlinks": ["a"]},
        }
    }

    audit = audit_native_collections(registry, index)

    assert audit["active_page_count"] == 4
    assert audit["assigned_collection_count"] == 3
    assert audit["assignment_coverage"] == 0.75
    assert audit["unassigned_uids"] == ["u4"]
    assert audit["collection_count"] == 2
    assert audit["resolved_link_edges"] == 4
    assert audit["intra_collection_edges"] == 2
    assert audit["cross_collection_edges"] == 2
    assert audit["unresolved_or_external_edges"] == 1
    assert audit["intra_collection_link_rate"] == 0.5
