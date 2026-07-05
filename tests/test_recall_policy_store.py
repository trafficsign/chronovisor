from __future__ import annotations

from llm_wiki_mcp.recall_policy_store import apply_policy_overrides, normalize_policy_overrides
from llm_wiki_mcp.recall_runtime import RecallPolicy


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
