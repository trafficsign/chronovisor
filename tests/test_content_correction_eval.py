from __future__ import annotations

from llm_wiki_mcp.content_correction_eval import load_cases, run_eval


def test_correction_detection_corpus_has_independent_splits_and_passes() -> None:
    cases = load_cases()
    result = run_eval()

    assert {case["split"] for case in cases} == {"golden", "holdout"}
    assert {case["category"] for case in cases} >= {
        "positive_explicit",
        "positive_provenance",
        "hard_negative",
        "metamorphic",
    }
    assert result["status"] == "passed"
    assert result["metrics"]["precision"] == 1.0
    assert result["metrics"]["recall"] == 1.0
