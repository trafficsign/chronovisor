from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from chronovisor.core.semantic_index import SemanticIndexError
from chronovisor.recall.duplicate_review import (
    DuplicateCandidate,
    build_duplicate_review_queue,
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
        "chronovisor.recall.duplicate_review.load_active_generation",
        lambda: SimpleNamespace(
            manifest=SimpleNamespace(dimensions=2),
            vectors=np.asarray(
                [
                    [1.0, 0.0],
                    [0.99, 0.1],
                    [0.98, 0.2],
                ],
                dtype=np.float32,
            ),
            page_ids=["a", "b", "c"],
            kinds=["page", "page", "page"],
            overridden_pages=set(),
            delta_vectors=np.empty((0, 2), dtype=np.float32),
            delta_page_ids=[],
            delta_kinds=[],
        ),
    )

    candidates = embedding_duplicate_candidates(metas, threshold=0.9, limit=1)

    assert len(candidates) == 1
    assert isinstance(candidates[0], DuplicateCandidate)
    assert {candidates[0].left, candidates[0].right} == {"b", "c"}


def test_embedding_candidates_use_authoritative_page_vectors(monkeypatch) -> None:
    metas = [
        {"page_id": "a", "title": "A"},
        {"page_id": "b", "title": "B"},
        {"page_id": "c", "title": "C"},
    ]
    monkeypatch.setattr(
        "chronovisor.recall.duplicate_review.load_active_generation",
        lambda: SimpleNamespace(
            manifest=SimpleNamespace(dimensions=2),
            vectors=np.asarray(
                [
                    [1.0, 0.0],  # base page a
                    [0.0, 1.0],  # overridden base page b
                    [0.0, 1.0],  # base page c
                    [1.0, 0.0],  # non-page row for c
                ],
                dtype=np.float32,
            ),
            page_ids=["a", "b", "c", "c"],
            kinds=["page", "page", "page", "chunk"],
            overridden_pages={"b"},
            delta_vectors=np.asarray(
                [
                    [1.0, 0.0],  # authoritative page b
                    [0.0, 1.0],  # non-page row for b
                ],
                dtype=np.float32,
            ),
            delta_page_ids=["b", "b"],
            delta_kinds=["page", "chunk"],
        ),
    )

    candidates = embedding_duplicate_candidates(metas, threshold=0.999, limit=10)

    assert [(candidate.left, candidate.right) for candidate in candidates] == [
        ("a", "b")
    ]


def test_invalid_active_vector_respects_queue_strict_mode(monkeypatch) -> None:
    monkeypatch.setattr(
        "chronovisor.recall.duplicate_review._knowledge_metas",
        lambda: [{"page_id": "a", "title": "A"}],
    )
    monkeypatch.setattr(
        "chronovisor.recall.duplicate_review.load_active_generation",
        lambda: SimpleNamespace(
            manifest=SimpleNamespace(dimensions=2),
            vectors=np.asarray([[0.0, 0.0]], dtype=np.float32),
            page_ids=["a"],
            kinds=["page"],
            overridden_pages=set(),
            delta_vectors=np.empty((0, 2), dtype=np.float32),
            delta_page_ids=[],
            delta_kinds=[],
        ),
    )

    assert build_duplicate_review_queue(strict=False) == []
    with pytest.raises(SemanticIndexError, match="page vector is invalid"):
        build_duplicate_review_queue(strict=True)
