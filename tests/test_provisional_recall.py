from __future__ import annotations

from pathlib import Path

from chronovisor.recall import provisional_recall


def _entry(provisional_id: str, text: str, record_index: int) -> dict:
    return {
        "provisional_id": provisional_id,
        "raw_file": f"raw-{provisional_id}.md",
        "records": [
            {
                "source_record_index": record_index,
                "role": "user",
                "text": text,
            }
        ],
        "host_boundary": "citation_only",
        "source_host": "codex",
    }


def test_provisional_ranking_uses_idf_coverage_not_a_fixed_score(
    tmp_path: Path,
    monkeypatch,
) -> None:
    entries = [
        _entry("full", "alpha beta gamma delta", 1),
        _entry("rare", "delta only", 2),
        _entry("common-1", "alpha only", 3),
        _entry("common-2", "alpha elsewhere", 4),
    ]
    monkeypatch.setattr(
        provisional_recall,
        "sync_index",
        lambda **_kwargs: {"entries": entries},
    )

    hits = provisional_recall.search_provisional(
        "alpha beta gamma delta", chronovisor_root=tmp_path
    )

    assert hits[0]["provisional_id"] == "full"
    assert len({hit["score"] for hit in hits}) > 1
    assert hits[0]["ranking_basis"] == "idf_weighted_coverage+density+phrase"
    assert hits[0]["score"] <= provisional_recall.RANK_CAP


def test_provisional_ranking_tokenizes_japanese_and_breaks_ties_by_record_recency(
    tmp_path: Path,
    monkeypatch,
) -> None:
    entries = [
        _entry("older", "検索精度を改善する", 2),
        _entry("newer", "検索精度を改善する", 8),
    ]
    monkeypatch.setattr(
        provisional_recall,
        "sync_index",
        lambda **_kwargs: {"entries": entries},
    )

    hits = provisional_recall.search_provisional("検索精度", chronovisor_root=tmp_path)

    assert [hit["provisional_id"] for hit in hits] == ["newer", "older"]
    assert hits[0]["matched_terms"]
