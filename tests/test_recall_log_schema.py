from __future__ import annotations

from chronovisor.recall.recall_log_schema import (
    canonicalize_page_ids,
    join_used_recall_episodes,
    page_ids_from_record,
)


def test_page_ids_from_record_supports_current_and_legacy_fields() -> None:
    row = {
        "pages": ["current", "shared"],
        "injected_pages": ["injected", "shared"],
        "expected_pages": ["expected"],
        "context_items": [
            {"page_id": "context"},
            {"page_id": "current"},
            {"page_id": ""},
            {"other": "ignored"},
        ],
    }

    assert page_ids_from_record(row) == [
        "current",
        "shared",
        "context",
        "injected",
        "expected",
    ]


def test_page_ids_from_record_ignores_malformed_field_values() -> None:
    row = {
        "pages": "not-a-list",
        "injected_pages": [None, 42],
        "expected_pages": None,
        "context_items": [None, "bad", {"page_id": 42}],
    }

    assert page_ids_from_record(row) == []


def test_canonicalize_page_ids_resolves_alias_targets_and_deduplicates() -> None:
    assert canonicalize_page_ids(
        ["former-page", "current-page", "nested-former"],
        {
            "former-page": "current-page",
            "nested-former": "chronovisor/nested-current.md",
        },
    ) == ["current-page", "nested-current"]


def test_join_used_recall_episodes_is_exact_and_deduplicated() -> None:
    recalls = [
        {
            "decision_id": "accepted",
            "session_id": "session-a",
            "prompt_preview": "recall runtime",
            "pages": ["exposed-only"],
        },
        {"decision_id": "ambiguous", "session_id": "session-a"},
        {"decision_id": "ambiguous", "session_id": "session-a"},
    ]
    pulls = [
        {
            "type": "used",
            "event_id": "event-1",
            "decision_id": "accepted",
            "session_id": "session-a",
            "page_ids": ["used-a", "used-a", "used-b"],
        },
        {
            "type": "used",
            "event_id": "event-1",
            "decision_id": "accepted",
            "session_id": "session-a",
            "page_ids": ["ignored-duplicate"],
        },
        {
            "type": "used",
            "event_id": "event-orphan",
            "decision_id": "orphan",
            "session_id": "session-a",
            "page_ids": ["ignored"],
        },
        {
            "type": "used",
            "event_id": "event-mismatch",
            "decision_id": "accepted",
            "session_id": "session-b",
            "page_ids": ["ignored"],
        },
        {
            "type": "used",
            "event_id": "event-ambiguous",
            "decision_id": "ambiguous",
            "session_id": "session-a",
            "page_ids": ["ignored"],
        },
    ]

    joined = join_used_recall_episodes(recalls, pulls)

    assert joined["accepted"] == 1
    assert joined["episodes"][0]["page_ids"] == ["used-a", "used-b"]
    assert joined["rejected_by_reason"] == {
        "ambiguous_decision": 1,
        "duplicate_event": 1,
        "orphan_decision": 1,
        "session_mismatch": 1,
    }
