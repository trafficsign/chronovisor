from __future__ import annotations

from chronovisor.core.recall_log_schema import (
    canonicalize_page_ids,
    join_used_recall_episodes,
    page_ids_from_record,
    recall_identity_from_record,
    recall_identity_is_complete,
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


def test_join_used_recall_episodes_merges_monotonic_page_amendments() -> None:
    recalls = [
        {
            "decision_id": "decision-1",
            "session_id": "session-1",
            "prompt_hash": "a" * 64,
        }
    ]
    pulls = [
        {
            "type": "used",
            "event_id": "event-1",
            "decision_id": "decision-1",
            "session_id": "session-1",
            "page_ids": ["page-a"],
        },
        {
            "type": "used",
            "event_id": "event-2",
            "decision_id": "decision-1",
            "session_id": "session-1",
            "page_ids": ["page-b", "page-a"],
            "model": "answer-model",
            "model_revision": "r1",
            "system_sha256": "a" * 64,
            "sampler_sha256": "b" * 64,
        },
    ]

    joined = join_used_recall_episodes(recalls, pulls)

    assert joined["accepted"] == 1
    assert joined["episodes"][0]["event_ids"] == ["event-1", "event-2"]
    assert joined["episodes"][0]["page_ids"] == ["page-a", "page-b"]
    assert joined["episodes"][0]["identity"] == {
        "model": "answer-model",
        "model_revision": "r1",
        "system_sha256": "a" * 64,
        "sampler_sha256": "b" * 64,
    }


def test_join_used_recall_episodes_keeps_unused_decisions_out_of_used_receipts() -> None:
    recalls = [
        {
            "decision_id": f"decision-{index}",
            "session_id": f"session-{index}",
            "pages": [f"page-{index}"],
            "model": "answer-model",
            "model_revision": "answer-model-r1",
            "system_sha256": "a" * 64,
            "sampler_sha256": "b" * 64,
        }
        for index in range(100)
    ]
    pulls = [
        {
            "type": "used",
            "event_id": f"used-{index}",
            "decision_id": f"decision-{index}",
            "session_id": f"session-{index}",
            "page_ids": [f"page-{index}"],
        }
        for index in range(0, 100, 2)
    ]

    joined = join_used_recall_episodes(recalls, pulls)

    joined_ids = {episode["decision_id"] for episode in joined["episodes"]}
    assert joined["accepted"] == 50
    assert len(joined_ids) == 50
    assert joined_ids == {f"decision-{index}" for index in range(0, 100, 2)}
    assert not joined_ids & {f"decision-{index}" for index in range(1, 100, 2)}
    assert joined["rejected"] == 0
    assert joined["rejected_by_reason"] == {}
    assert joined["identity_missing"] == 0
    assert joined["eligible_decisions"] == 100
    assert joined["used_decisions"] == 50
    assert joined["unused_decisions"] == 50
    assert joined["used_rate"] == 0.5
    assert joined["production_replayable"] == 50


def test_identity_is_strict_and_receipts_amend_per_field_without_overriding_source() -> None:
    source = {
        "decision_id": "decision-1",
        "session_id": "session-1",
        "model": " source-model ",
        "model_revision": 42,
        "system_sha256": "not-a-sha",
        "sampler_sha256": "A" * 64,
    }
    pulls = [
        {
            "type": "used",
            "event_id": "event-1",
            "decision_id": "decision-1",
            "session_id": "session-1",
            "page_ids": ["page-a"],
            "model": "receipt-model",
            "model_revision": "receipt-r1",
            "system_sha256": "b" * 64,
            "sampler_sha256": "c" * 64,
        },
        {
            "type": "used",
            "event_id": "event-2",
            "decision_id": "decision-1",
            "session_id": "session-1",
            "page_ids": ["page-b"],
            "model": "conflicting-model",
            "model_revision": "conflicting-r2",
            "system_sha256": "d" * 64,
            "sampler_sha256": "e" * 64,
        },
    ]

    joined = join_used_recall_episodes([source], pulls)

    assert joined["episodes"][0]["identity"] == {
        "model": "source-model",
        "model_revision": "receipt-r1",
        "system_sha256": "b" * 64,
        "sampler_sha256": "A" * 64,
    }
    assert joined["identity_missing"] == 0
    assert joined["episodes"][0]["event_ids"] == ["event-1", "event-2"]
    assert recall_identity_from_record({"model": 3, "system_sha256": "bad"}) == {}
    assert not recall_identity_is_complete({"model": "m", "model_revision": "r"})


def test_partial_receipts_complete_identity_but_first_conflict_wins() -> None:
    source = [{"decision_id": "decision-1", "session_id": "session-1"}]
    pulls = [
        {
            "type": "used",
            "event_id": "event-1",
            "decision_id": "decision-1",
            "session_id": "session-1",
            "page_ids": ["page-a"],
            "model": "receipt-model",
        },
        {
            "type": "used",
            "event_id": "event-2",
            "decision_id": "decision-1",
            "session_id": "session-1",
            "page_ids": ["page-b"],
            "model_revision": "receipt-r1",
            "system_sha256": "a" * 64,
        },
        {
            "type": "used",
            "event_id": "event-3",
            "decision_id": "decision-1",
            "session_id": "session-1",
            "page_ids": ["page-c"],
            "sampler_sha256": "b" * 64,
        },
        {
            "type": "used",
            "event_id": "event-4",
            "decision_id": "decision-1",
            "session_id": "session-1",
            "page_ids": ["page-d"],
            "model": "later-conflict",
            "system_sha256": "c" * 64,
        },
    ]

    joined = join_used_recall_episodes(source, pulls)

    assert joined["episodes"][0]["identity"] == {
        "model": "receipt-model",
        "model_revision": "receipt-r1",
        "system_sha256": "a" * 64,
        "sampler_sha256": "b" * 64,
    }
    assert joined["identity_missing"] == 0
