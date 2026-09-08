"""Protect ranking measurements without loading any model or live store."""

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/benchmark_recall_rerankers.py"
SPEC = importlib.util.spec_from_file_location("reranker_benchmark", SCRIPT)
assert SPEC and SPEC.loader
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


def test_metrics_distinguish_partial_retrieval_and_reject_bad_outputs():
    candidates = [
        {"id": "a", "relevance": 1},
        {"id": "b", "relevance": 0},
        {"id": "c", "relevance": 1},
        {"id": "d", "relevance": 0},
    ]
    metrics = benchmark.ranking_metrics(candidates, [3, 4, 1, 2])
    assert metrics["ranking"] == ["b", "a", "d", "c"]
    assert metrics["mrr"] == 0.5
    assert metrics["recall_at3"] == 0.5
    assert metrics["all_evidence_at3"] == 0
    assert 0 < metrics["ndcg_at3"] < 1
    for scores in ([1], [0, 1, float("nan"), 2]):
        with pytest.raises(ValueError):
            benchmark.ranking_metrics(candidates, scores)
    no_evidence = benchmark.ranking_metrics([{"id": "a", "relevance": 0}], [0.99])
    assert no_evidence["has_evidence"] is False
    assert "top1" not in no_evidence  # A high score is not correct abstention.


def test_fixture_is_reproducible_and_context_is_not_a_gold_label():
    fixture = SCRIPT.parent.parent / "tests/fixtures/recall_reranker_japanese.json"
    cases = benchmark.load_cases(fixture)
    assert cases == benchmark.load_cases(fixture)
    assert len(cases) == 24
    assert len({c["category"] for c in cases}) == 6
    case = next(c for c in cases if c["category"] == "coreference")
    assert case["context"] in benchmark.query_text(case)
    assert benchmark.query_text(case, with_context=False) == case["query"]
    assert case["rationale"] not in benchmark.query_text(case)
