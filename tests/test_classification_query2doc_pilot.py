from __future__ import annotations

from chronovisor.lab.classification_query2doc_pilot import (
    candidate_blind_page,
    query_text,
    reciprocal_rank_fusion,
)


def test_candidate_blind_projection_excludes_labels_and_metadata() -> None:
    page = {
        "uid": "page-1",
        "title": "Horse analogy",
        "summary": "A labor displacement argument",
        "excerpt": "x" * 3_000,
        "candidates": [{"notation": "331"}],
        "expected_primary_notations": ["331"],
        "gold_primary_notation": "331",
        "case_number": 4,
        "tags": ["d/work"],
        "raw_keywords": ["employment"],
    }

    projected = candidate_blind_page(page)

    assert set(projected) == {"uid", "title", "summary", "excerpt"}
    assert len(projected["excerpt"]) == 2_400
    assert "331" not in str(projected)


def test_query_text_uses_subject_headings_but_not_ignored_literals() -> None:
    text = query_text(
        {
            "query": {
                "subject_headings_ja": ["雇用", "職業指導"],
                "subject_headings_en": ["Employment", "Vocational guidance"],
                "literal_terms_to_ignore": ["horse", "operator"],
            }
        }
    )

    assert text == "雇用\n職業指導\nEmployment\nVocational guidance"
    assert "horse" not in text
    assert "operator" not in text


def test_equal_weight_rrf_is_fixed_and_deduplicates_notations() -> None:
    rows = reciprocal_rank_fusion(
        {
            "raw_lexical": [
                {"notation": "004.4", "label_en": "Software"},
                {"notation": "331", "label_en": "Employment"},
            ],
            "raw_dense": [
                {"notation": "796", "label_en": "Sport"},
                {"notation": "004.4", "label_en": "Software"},
            ],
            "query2doc_lexical": [
                {"notation": "331", "label_en": "Employment"},
            ],
            "query2doc_dense": [
                {"notation": "331", "label_en": "Employment"},
            ],
        },
        limit=3,
        k=60,
    )

    assert [row["notation"] for row in rows] == ["331", "004.4", "796"]
    assert rows[0]["channel_ranks"] == {
        "raw_lexical": 2,
        "query2doc_lexical": 1,
        "query2doc_dense": 1,
    }
    assert rows[1]["channel_ranks"] == {
        "raw_lexical": 1,
        "raw_dense": 2,
    }
