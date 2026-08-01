"""Compatibility helpers for recall-log JSON records."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def page_ids_from_record(row: Mapping[str, Any]) -> list[str]:
    """Return deduplicated page IDs from current and legacy recall records."""
    page_ids: list[str] = []

    current_pages = row.get("pages")
    if isinstance(current_pages, list):
        page_ids.extend(
            value for value in current_pages if isinstance(value, str) and value
        )

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


def canonicalize_page_ids(page_ids: list[str], aliases: Mapping[str, str]) -> list[str]:
    """Resolve immutable historical page IDs into current derived-index IDs."""

    canonical: list[str] = []
    for page_id in page_ids:
        target = aliases.get(page_id, page_id)
        target_id = Path(target.removesuffix(".md")).name
        if target_id:
            canonical.append(target_id)
    return list(dict.fromkeys(canonical))


def used_page_ids_from_record(row: Mapping[str, Any]) -> list[str]:
    """Return only pages explicitly declared as materially used."""

    if row.get("type") != "used":
        return []
    values = row.get("page_ids")
    if not isinstance(values, list):
        return []
    return list(
        dict.fromkeys(value for value in values if isinstance(value, str) and value)
    )


def pull_event_id(row: Mapping[str, Any]) -> str:
    """Return the durable event ID, with a stable legacy-row fallback."""

    event_id = row.get("event_id")
    if isinstance(event_id, str) and event_id:
        return event_id
    encoded = json.dumps(
        dict(row), ensure_ascii=False, sort_keys=True, default=str
    ).encode("utf-8")
    return "legacy-" + hashlib.sha256(encoded).hexdigest()[:24]


def join_used_recall_episodes(
    recall_rows: list[Mapping[str, Any]],
    pull_rows: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Join explicit usage receipts to one unambiguous recall decision.

    Positive supervision is deliberately stricter than telemetry analysis:
    decision IDs must be unique, the session identity must be present and
    equal whenever the recall decision has one, and duplicate event receipts
    are ignored.  Rejected counts are returned so a silent loss of learning
    signal becomes observable.
    """

    decisions: dict[str, list[Mapping[str, Any]]] = {}
    for row in recall_rows:
        decision_id = str(row.get("decision_id") or "")
        if decision_id:
            decisions.setdefault(decision_id, []).append(row)

    episodes_by_decision: dict[tuple[str, str], dict[str, Any]] = {}
    rejected: Counter[str] = Counter()
    seen_events: set[str] = set()
    for pull in pull_rows:
        if pull.get("type") != "used":
            continue
        event_id = pull_event_id(pull)
        if event_id in seen_events:
            rejected["duplicate_event"] += 1
            continue
        seen_events.add(event_id)
        decision_id = str(pull.get("decision_id") or "")
        if not decision_id:
            rejected["missing_decision_id"] += 1
            continue
        candidates = decisions.get(decision_id, [])
        if not candidates:
            rejected["orphan_decision"] += 1
            continue
        if len(candidates) != 1:
            rejected["ambiguous_decision"] += 1
            continue
        recall = candidates[0]
        recall_session = str(recall.get("session_id") or "")
        pull_session = str(pull.get("session_id") or "")
        if recall_session and not pull_session:
            rejected["missing_session_id"] += 1
            continue
        if recall_session and pull_session != recall_session:
            rejected["session_mismatch"] += 1
            continue
        pages = used_page_ids_from_record(pull)
        if not pages:
            rejected["missing_page_ids"] += 1
            continue
        episode_session = pull_session or recall_session
        episode_key = (decision_id, episode_session)
        existing = episodes_by_decision.get(episode_key)
        if existing is None:
            episodes_by_decision[episode_key] = {
                "event_id": event_id,
                "event_ids": [event_id],
                "decision_id": decision_id,
                "session_id": episode_session,
                "page_ids": pages,
                "recall": dict(recall),
                "pull": dict(pull),
            }
            continue
        existing["event_ids"].append(event_id)
        existing["page_ids"] = list(dict.fromkeys([*existing["page_ids"], *pages]))

    episodes = list(episodes_by_decision.values())
    return {
        "episodes": episodes,
        "accepted": len(episodes),
        "rejected": sum(rejected.values()),
        "rejected_by_reason": dict(sorted(rejected.items())),
    }
