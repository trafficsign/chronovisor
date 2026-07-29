from __future__ import annotations

import json

from chronovisor.recall.recall_policy_store import (
    append_live_episode,
    apply_policy_overrides,
    normalize_policy_overrides,
)
from chronovisor.recall.recall_runtime import RecallPolicy


def test_normalize_policy_overrides_keeps_allowed_fields_only() -> None:
    overrides = normalize_policy_overrides(
        {
            "max-pages": "4",
            "fusion_semantic": 0.8,
            "unknown": 123,
            "read_threshold": 0.3,
            "search_threshold": 0.35,
        }
    )

    assert overrides["max_pages"] == 4
    assert overrides["fusion_semantic"] == 0.8
    assert "unknown" not in overrides
    assert overrides["read_threshold"] > overrides["search_threshold"]


def test_apply_policy_overrides_mutates_recall_policy() -> None:
    policy = RecallPolicy(max_pages=3, fusion_semantic=0.6)

    applied = apply_policy_overrides(policy, {"max_pages": 5, "fusion_semantic": 0.9})

    assert applied == ["max_pages", "fusion_semantic"]
    assert policy.max_pages == 5
    assert policy.fusion_semantic == 0.9


def test_append_live_episode_writes_unlabeled_quality_record(tmp_path) -> None:
    path = tmp_path / "live-episodes.jsonl"

    append_live_episode(
        {
            "ts": "2026-07-05T12:00:00",
            "decision_id": "d1",
            "host": "codex",
            "decision": "read",
            "queries": ["chronovisor recall"],
            "pages": ["page-a"],
            "prompt_preview": "Chronovisor recall",
        },
        path=path,
    )

    row = json.loads(path.read_text(encoding="utf-8"))
    assert row["schema_version"] == 1
    assert row["decision_id"] == "d1"
    assert row["quality"]["source"] == "unlabeled-live"
    assert row["pages"] == ["page-a"]
