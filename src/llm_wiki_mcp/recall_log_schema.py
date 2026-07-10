"""Compatibility helpers for recall-log JSON records."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def page_ids_from_record(row: Mapping[str, Any]) -> list[str]:
    """Return deduplicated page IDs from current and legacy recall records."""
    page_ids: list[str] = []

    current_pages = row.get("pages")
    if isinstance(current_pages, list):
        page_ids.extend(value for value in current_pages if isinstance(value, str) and value)

    context_items = row.get("context_items")
    if isinstance(context_items, list):
        for item in context_items:
            if not isinstance(item, Mapping):
                continue
            page_id = item.get("page_id")
            if isinstance(page_id, str) and page_id:
                page_ids.append(page_id)

    for key in ("injected_pages", "expected_pages"):
        values = row.get(key)
        if not isinstance(values, list):
            continue
        page_ids.extend(value for value in values if isinstance(value, str) and value)

    return list(dict.fromkeys(page_ids))
