from __future__ import annotations

from llm_wiki_mcp.recall_log_schema import page_ids_from_record


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
