from __future__ import annotations

import math

from chronovisor.duplicate_review import (
    DuplicateCandidate,
    embedding_duplicate_candidates,
    title_duplicate_candidates,
)


def test_title_duplicate_candidates_detects_near_titles() -> None:
    metas = [
        {
            "page_id": "workplace-stress-and-mentoring",
            "title": "Workplace Stress and Mentoring",
        },
        {
            "page_id": "workplace-stress-mentoring",
            "title": "Workplace Stress Mentoring",
        },
        {"page_id": "ollama-model-status", "title": "Ollama Model Status"},
    ]

    candidates = title_duplicate_candidates(metas, threshold=0.85)

    assert [(c.left, c.right, c.method) for c in candidates] == [
        ("workplace-stress-and-mentoring", "workplace-stress-mentoring", "title")
    ]


def test_title_duplicate_candidates_blocks_large_corpus() -> None:
    metas = [
        {"page_id": f"page-{index}", "title": f"Unrelated knowledge page {index}"}
        for index in range(501)
    ]
    metas.extend(
        [
            {
                "page_id": "workplace-stress-and-mentoring",
                "title": "Workplace Stress and Mentoring",
            },
            {
                "page_id": "workplace-stress-mentoring",
                "title": "Workplace Stress Mentoring",
            },
        ]
    )

    candidates = title_duplicate_candidates(metas, threshold=0.85)

    assert any(
        {candidate.left, candidate.right}
        == {"workplace-stress-and-mentoring", "workplace-stress-mentoring"}
        for candidate in candidates
    )


def test_embedding_duplicate_candidates_keeps_highest_bounded_results(
    monkeypatch,
) -> None:
    metas = [
        {"page_id": "a", "title": "A"},
        {"page_id": "b", "title": "B"},
        {"page_id": "c", "title": "C"},
    ]
    monkeypatch.setattr(
        "chronovisor.duplicate_review._iter_all_embeddings",
        lambda: [
            ("a", [1.0, 0.0], 0.0, 1.0),
            ("b", [0.99, 0.1], 0.0, math.sqrt(0.9901)),
            ("c", [0.98, 0.2], 0.0, math.sqrt(1.0004)),
        ],
    )

    candidates = embedding_duplicate_candidates(metas, threshold=0.9, limit=1)

    assert len(candidates) == 1
    assert isinstance(candidates[0], DuplicateCandidate)
    assert {candidates[0].left, candidates[0].right} == {"b", "c"}
