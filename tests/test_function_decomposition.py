"""Pure seams extracted from high-ROI orchestration functions."""

from __future__ import annotations


def test_ingest_review_artifact_projection_is_structural_and_fail_closed() -> None:
    from llm_wiki_mcp.ingest_review_apply import inspect_ingest_review_artifact

    authority = {"lane": "ingest_reconciliation"}
    state = inspect_ingest_review_artifact(
        {
            "review": {"decision": "apply_available"},
            "authority": authority,
        },
        has_planned_operations=True,
        planned_postimages_fully_applied=True,
    )
    assert state.review == {"decision": "apply_available"}
    assert state.authority == authority
    assert state.exact_postimages_already_applied is True

    malformed = inspect_ingest_review_artifact(
        {"review": [], "authority": "copied"},
        has_planned_operations=True,
        planned_postimages_fully_applied=True,
    )
    assert malformed.review is None
    assert malformed.authority is None
    assert malformed.exact_postimages_already_applied is False
