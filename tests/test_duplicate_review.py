from __future__ import annotations

from chronovisor.duplicate_review import title_duplicate_candidates


def test_title_duplicate_candidates_detects_near_titles() -> None:
    metas = [
        {"page_id": "workplace-stress-and-mentoring", "title": "Workplace Stress and Mentoring"},
        {"page_id": "workplace-stress-mentoring", "title": "Workplace Stress Mentoring"},
        {"page_id": "ollama-model-status", "title": "Ollama Model Status"},
    ]

    candidates = title_duplicate_candidates(metas, threshold=0.85)

    assert [(c.left, c.right, c.method) for c in candidates] == [
        ("workplace-stress-and-mentoring", "workplace-stress-mentoring", "title")
    ]
