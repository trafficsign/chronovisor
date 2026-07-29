from __future__ import annotations

from chronovisor.classification_query2doc_v2 import (
    _matching_rank,
    coverage_first_fusion,
    heading_query_pages,
    interleave_heading_candidates,
)


def _rows(prefix: str, count: int = 12) -> list[dict[str, str]]:
    return [
        {
            "notation": f"{prefix}{index}",
            "label_en": f"{prefix} {index}",
            "label_ja": f"{prefix} {index}",
        }
        for index in range(1, count + 1)
    ]


def test_heading_query_pages_keep_roles_independent() -> None:
    artifact = {
        "query": {
            "headings": [
                {
                    "role": "principal_shelf",
                    "ja": "知識管理",
                    "en": "Knowledge management",
                },
                {
                    "role": "problem_or_activity",
                    "ja": "記録の整理",
                    "en": "Record organization",
                },
                {
                    "role": "context",
                    "ja": "個人的な振り返り",
                    "en": "Personal reflection",
                },
            ]
        }
    }

    pages = heading_query_pages(artifact)

    assert [role for role, _page in pages] == [
        "principal_shelf",
        "problem_or_activity",
        "context",
    ]
    assert pages[0][1]["title"] == "知識管理\nKnowledge management"
    assert "記録の整理" not in pages[0][1]["title"]


def test_heading_candidates_are_round_robin_not_concatenated() -> None:
    rows = {
        "principal_shelf": _rows("P"),
        "problem_or_activity": _rows("A"),
        "context": _rows("C"),
    }

    result = interleave_heading_candidates(rows, limit=6)

    assert [row["notation"] for row in result] == [
        "P1",
        "A1",
        "C1",
        "P2",
        "A2",
        "C2",
    ]


def test_coverage_first_fusion_reserves_single_channel_hits() -> None:
    query_lexical = _rows("Q")
    raw_lexical = _rows("R")
    query_dense = _rows("D")
    raw_dense = _rows("M")
    shared = {
        "notation": "004.33",
        "label_en": "Storage devices",
        "label_ja": "記憶装置",
    }
    for rows in (query_lexical, raw_lexical, query_dense, raw_dense):
        rows.insert(0, dict(shared))
    raw_lexical[1] = {
        "notation": "331.4",
        "label_en": "Working environment",
        "label_ja": "労働環境",
    }
    query_dense[3] = {
        "notation": "551.24",
        "label_en": "Geotectonics",
        "label_ja": "地殻構造学",
    }

    result = coverage_first_fusion(
        {
            "raw_lexical": raw_lexical,
            "raw_dense": raw_dense,
            "query2doc_lexical": query_lexical,
            "query2doc_dense": query_dense,
        }
    )

    by_notation = {row["notation"]: row for row in result}
    assert len(result) == 12
    assert "331.4" in by_notation
    assert by_notation["331.4"]["reserved_by"] == ["raw_lexical"]
    assert "551.24" in by_notation
    assert by_notation["551.24"]["reserved_by"] == ["query2doc_dense"]
    assert any(row["selection"] == "backfill" for row in result)


def test_coverage_first_fusion_is_deterministic() -> None:
    channels = {
        "raw_lexical": _rows("R"),
        "raw_dense": _rows("M"),
        "query2doc_lexical": _rows("Q"),
        "query2doc_dense": _rows("D"),
    }

    first = coverage_first_fusion(channels)
    second = coverage_first_fusion(channels)

    assert first == second


def test_matching_rank_accepts_target_descendants() -> None:
    candidates = [{"notation": "331.4.1"}]

    assert _matching_rank(candidates, ["331.4"]) == 1
