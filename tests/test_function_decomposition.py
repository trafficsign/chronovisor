"""Pure seams extracted from high-ROI orchestration functions."""

from __future__ import annotations


def test_ingest_review_artifact_projection_is_structural_and_fail_closed() -> None:
    from chronovisor.ingest.ingest_review_apply import inspect_ingest_review_artifact

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
    from chronovisor.ingest.ingest import _normalize_ingest_source_metadata

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


def test_orchestrator_unit_keyword_and_event_decisions_are_deterministic() -> None:
    from chronovisor.ingest.orchestrator import _raw_unit_event, _raw_unit_keywords

    raw = "---\nraw_keywords: [current]\nkeywords: [legacy]\n---\nbody\n"
    assert _raw_unit_keywords(None, raw) == ["current"]
    assert _raw_unit_keywords(("bound",), raw) == ["bound"]
    assert _raw_unit_event(succeeded=True, deferred=True, continued=True) == (
        "success",
        "processed",
    )
    assert _raw_unit_event(succeeded=False, deferred=True, continued=True) == (
        "info",
        "shard review continuation pending",
    )
    assert _raw_unit_event(succeeded=False, deferred=True, continued=False) == (
        "info",
        "semantic deferred",
    )
    assert _raw_unit_event(succeeded=False, deferred=False, continued=False) == (
        "warn",
        "not processed",
    )


def test_self_heal_read_back_retirement_classification_is_ordered() -> None:
    import json

    from chronovisor.ingest.self_heal import _read_back_packet_retirement_kind

    transient = {
        "failure_class": "read_back.repeated_miss",
        "error": "request timed out",
        "raw_preview": json.dumps({"failure": {"reason": "unknown"}}),
    }
    empty = {
        "failure_class": "read_back.repeated_miss",
        "raw_preview": json.dumps({"failure": {"reason": "empty-query"}}),
    }
    assert _read_back_packet_retirement_kind(transient) == "transient"
    assert _read_back_packet_retirement_kind(empty) == "empty_query"
    assert _read_back_packet_retirement_kind({"failure_class": "other"}) is None


def test_content_correction_frontier_evidence_projection_is_strict() -> None:
    from chronovisor.recall.content_correction import (
        _frontier_item_evidence_inputs,
        _page_evidence_hashes,
    )

    event, context, page_ids = _frontier_item_evidence_inputs(
        {
            "metadata": {
                "source_prompt": "prompt",
                "source_assistant_response": "answer",
                "correction_prompt": "fix",
                "candidate_pages": ["memory/a", 3, "memory/b"],
            }
        }
    )
    assert event["source_prompt"] == "prompt"
    assert context == "prompt answer fix"
    assert page_ids == ["memory/a", "memory/b"]
    assert _page_evidence_hashes(
        [
            {"page_id": "memory/a", "sha256": "a" * 64},
            {"page_id": "memory/b"},
            "invalid",
        ]
    ) == {"memory/a": "a" * 64}
    assert _frontier_item_evidence_inputs({"metadata": []}) == ({}, "  ", [])
