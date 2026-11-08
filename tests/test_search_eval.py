import tomllib
from pathlib import Path

from chronovisor.ops.search_eval import compare, rank_of, summarize

ROOT = Path(__file__).resolve().parents[1]


def test_rank_summary_and_compare() -> None:
    assert rank_of(["b"], ["a", "b"]) == 2
    assert rank_of(["z"], ["a"]) is None
    assert rank_of(["a"], ["x", "a-part-02"]) == 2
    s = summarize(
        [
            {"rank": 1, "ms": 10, "rerank": "applied"},
            {"rank": None, "ms": 30, "rerank": "applied"},
        ]
    )
    assert s["hit@1"] == 0.5 and s["mrr"] == 0.5 and s["p50_ms"] == 20
    a = {"rows": [{"id": "q", "rank": 3}]}
    b = {"rows": [{"id": "q", "rank": 1}]}
    assert compare(a, b) == {"b_better": 1, "b_worse": 0, "same": 0}


def test_search_eval_entry_point_is_declared() -> None:
    scripts = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]["scripts"]
    assert scripts["chronovisor-search-eval"] == "chronovisor.ops.search_eval:main"
