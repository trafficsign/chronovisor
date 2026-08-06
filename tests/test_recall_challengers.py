from __future__ import annotations

import pytest

from chronovisor.lab import recall_challengers
from chronovisor.lab.recall_challengers import (
    adoption_gate,
    run_report,
    top_k_reproduction,
)
from chronovisor.recall.judge_prefix_cache import (
    PrefixCacheKey,
    PrefixHandleCache,
)
from chronovisor.recall.recall_late_interaction import LateInteractionIndex


def test_late_interaction_index_is_separate_deterministic_and_incremental(
    tmp_path,
) -> None:
    index = LateInteractionIndex(tmp_path / "colbert.sqlite")
    index.upsert("a", revision="1", vectors=[[1.0, 0.0], [0.0, 1.0]])
    index.upsert("b", revision="1", vectors=[[-1.0, 0.0]])
    first = index.search([[1.0, 0.0]], limit=2)
    index.upsert("b", revision="2", vectors=[[1.0, 0.0]])
    second = index.search([[1.0, 0.0]], limit=2)

    assert first[0][0] == "a"
    assert [row[0] for row in second] == ["a", "b"]
    assert index.stats()["documents"] == 2
    assert index.stats()["bytes"] > 0


def test_prefix_cache_key_uses_digest_and_lru_not_page_identity() -> None:
    cache = PrefixHandleCache(max_entries=1)
    first = PrefixCacheKey.build(
        model_revision="r1",
        fixed_prompt="fixed",
        support_span="bounded support",
        position=12,
    )
    second = PrefixCacheKey.build(
        model_revision="r1",
        fixed_prompt="fixed",
        support_span="other support",
        position=12,
    )
    cache.put(first, "handle-1")
    cache.put(second, "handle-2")

    assert "bounded support" not in repr(first)
    assert cache.get(first) is None
    assert cache.get(second) == "handle-2"
    assert len(cache) == 1


def test_challenger_requires_complete_nondegrading_speed_metrics() -> None:
    baseline = {
        "recall_at_5": 0.563,
        "negative_hit_at_20": 0.229,
        "p95_ms": 437.7,
        "max_ms": 501.2,
        "over_4s": 0.0,
        "resource_bytes": 1.0,
    }
    missing = adoption_gate(baseline, {})
    winner = adoption_gate(
        baseline,
        {
            "recall_at_5": 0.563,
            "negative_hit_at_20": 0.20,
            "p95_ms": 200.0,
            "max_ms": 300.0,
            "over_4s": 0.0,
            "resource_bytes": 10.0,
        },
    )

    assert missing["status"] == "rejected"
    assert winner["status"] == "passed"
    assert top_k_reproduction([["a", "b"]], [["a", "b"]], k=2) == 1.0


@pytest.fixture()
def no_live_ollama_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    original_which = recall_challengers.shutil.which

    def isolated_which(command: str) -> str | None:
        if command == "ollama":
            return None
        return original_which(command)

    monkeypatch.setattr(recall_challengers.shutil, "which", isolated_which)


def test_report_never_adopts_unmeasured_challengers(
    tmp_path,
    no_live_ollama_cli: None,
) -> None:
    del no_live_ollama_cli
    report = run_report(
        baseline={
            "recall_at_5": 0.563,
            "negative_hit_at_20": 0.229,
            "p95_ms": 437.7,
            "max_ms": 501.2,
            "over_4s": 0.0,
            "resource_bytes": 1.0,
        },
        output_file=tmp_path / "report.json",
    )

    assert report["winner"] is None
    assert report["production_changed"] is False
    assert report["environment"]["ollama_executable"] == ""
    assert report["environment"]["ollama_version"] == ""
    assert set(report["excluded_production"]) == {
        "dsi",
        "memory_lora",
        "speculative_injection",
    }


def test_report_selects_only_a_measured_gate_winner(
    tmp_path,
    no_live_ollama_cli: None,
) -> None:
    del no_live_ollama_cli
    baseline = {
        "recall_at_5": 0.563,
        "negative_hit_at_20": 0.229,
        "p95_ms": 437.7,
        "max_ms": 501.2,
        "over_4s": 0.0,
        "resource_bytes": 1.0,
    }
    report = run_report(
        baseline=baseline,
        measurements={
            "colbert": {
                "status": "measured",
                "recall_at_5": 0.563,
                "negative_hit_at_20": 0.20,
                "p95_ms": 200.0,
                "max_ms": 300.0,
                "over_4s": 0.0,
                "resource_bytes": 10.0,
            }
        },
        output_file=tmp_path / "report.json",
    )

    assert report["winner"] == "colbert"
    assert report["challengers"]["colbert"]["adopted"] is True
    assert report["production_changed"] is False
    assert report["environment"]["ollama_executable"] == ""
    assert report["environment"]["ollama_version"] == ""
