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


def test_run_ingest_metadata_normalization_accepts_only_exact_side_channels() -> None:
    from llm_wiki_mcp.ingest import _normalize_ingest_source_metadata

    keywords = ["qwen", "recall"]
    normalized, source = _normalize_ingest_source_metadata(
        {"raw_keywords": keywords, "source_raw": "raw/session.md", "ignored": 1}
    )
    assert normalized == keywords
    assert normalized is not keywords
    assert source == "raw/session.md"
    assert _normalize_ingest_source_metadata(
        {"raw_keywords": ["ok", 1], "source_raw": 3}
    ) == (None, None)
    assert _normalize_ingest_source_metadata(None) == (None, None)
