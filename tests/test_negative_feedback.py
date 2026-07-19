from __future__ import annotations

import hashlib
import json

import pytest

from chronovisor import negative_feedback
from chronovisor.feedback_ledger import feedback_row_sha256
from chronovisor.runtime_config import NegativeFeedbackConfig, load_negative_feedback_config
from chronovisor.search import ScoredPage


def page(page_id: str, score: float) -> ScoredPage:
    return ScoredPage(
        page_id=page_id,
        title=page_id,
        folder="",
        updated="2026-06-12",
        score=score,
    )


def write_feedback(path, rows) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


@pytest.fixture
def feedback_file(tmp_path, monkeypatch):
    path = tmp_path / "feedback.jsonl"
    monkeypatch.setattr(negative_feedback, "FEEDBACK_FILE_OVERRIDE", path)
    monkeypatch.setattr(negative_feedback, "GOLDEN_FILE_OVERRIDE", tmp_path / "golden.jsonl")
    monkeypatch.setattr(negative_feedback, "_CACHE", negative_feedback._Cache())
    monkeypatch.setattr(negative_feedback, "_PROTECT_CACHE", negative_feedback._Cache())
    return path


CONFIG = NegativeFeedbackConfig(enabled=True, similarity_threshold=0.35, penalty=0.85)


def test_similar_query_penalizes_feedback_pages(feedback_file) -> None:
    write_feedback(feedback_file, [
        {
            "ts": "2026-06-11T10:23:35",
            "kind": "injection_ignored",
            "prompt": "メニューバーにショートカットを置く設定の話",
            "expected_pages": ["irrelevant-page"],
        },
    ])
    penalties = negative_feedback.penalties_for_query(
        "メニューバーにショートカットを置く設定の話", CONFIG
    )
    assert "irrelevant-page" in penalties
    assert penalties["irrelevant-page"] == pytest.approx(0.85)


def test_local_auditor_precision_label_cannot_penalize_without_frontier(
    feedback_file,
) -> None:
    row = {
        "ts": "2026-06-11T10:23:35",
        "kind": "injection_ignored",
        "source": "auditor_precision",
        "prompt": "メニューバーにショートカットを置く設定の話",
        "expected_pages": ["possibly-relevant-page"],
    }
    write_feedback(feedback_file, [row])

    assert negative_feedback.penalties_for_query(row["prompt"], CONFIG) == {}

    write_feedback(feedback_file, [{**row, "frontier_reviewed": True}])
    negative_feedback._CACHE = negative_feedback._Cache()
    assert negative_feedback.penalties_for_query(row["prompt"], CONFIG) == {
        "possibly-relevant-page": pytest.approx(0.85)
    }


def test_page_ignored_penalizes_only_explicit_negative_page(feedback_file) -> None:
    write_feedback(feedback_file, [
        {
            "ts": "2026-06-11T10:23:35",
            "kind": "page_ignored",
            "prompt": "G32P と P24U のレビューを比較して",
            "expected_pages": ["g32p-review", "p24u-review"],
            "negative_pages": ["p24u-review"],
            "frontier_reviewed": True,
        },
    ])

    penalties = negative_feedback.penalties_for_query(
        "G32P と P24U のレビューを比較して", CONFIG
    )

    assert penalties == {"p24u-review": pytest.approx(0.85)}
    adjusted = negative_feedback.apply_penalties(
        [page("g32p-review", 0.8), page("p24u-review", 0.9)], penalties
    )
    assert [item.page_id for item in adjusted] == ["g32p-review", "p24u-review"]
    assert adjusted[0].score == pytest.approx(0.8)
    assert adjusted[1].score == pytest.approx(0.9 * 0.15)


def test_exact_page_ignored_retraction_preserves_other_feedback(feedback_file) -> None:
    legacy = {
        "ts": "2026-07-11T10:23:35Z",
        "kind": "page_ignored",
        "source": "content_correction",
        "content_correction_key": "legacy-key",
        "prompt": "same query",
        "negative_pages": ["legacy-noise"],
        "frontier_reviewed": True,
    }
    valid = {
        **legacy,
        "content_correction_key": "valid-key",
        "negative_pages": ["valid-noise"],
    }
    write_feedback(
        feedback_file,
        [
            legacy,
            valid,
            {
                "ts": "2026-07-11T11:00:00Z",
                "kind": "page_ignored_retracted",
                "source": "content_correction",
                "content_correction_key": "legacy-key",
                "target_kind": "page_ignored",
                "target_feedback_sha256": feedback_row_sha256(legacy),
            },
        ],
    )

    assert negative_feedback.penalties_for_query("same query", CONFIG) == {
        "valid-noise": pytest.approx(0.85)
    }


def test_page_ignored_hash_binding_expires_when_page_content_changes(
    feedback_file,
    tmp_path,
    monkeypatch,
) -> None:
    page_path = tmp_path / "p24u-review.md"
    page_path.write_text("Old irrelevant content.\n", encoding="utf-8")
    original_hash = hashlib.sha256(page_path.read_bytes()).hexdigest()
    monkeypatch.setattr(
        negative_feedback,
        "find_mutation_page",
        lambda page_id: page_path if page_id == "p24u-review" else None,
    )
    write_feedback(
        feedback_file,
        [
            {
                "ts": "2026-07-11T10:23:35Z",
                "kind": "page_ignored",
                "prompt": "G32P と P24U のレビューを比較して",
                "negative_pages": ["p24u-review"],
                "negative_page_hashes": {"p24u-review": original_hash},
                "frontier_reviewed": True,
            }
        ],
    )

    query = "G32P と P24U のレビューを比較して"
    assert "p24u-review" in negative_feedback.penalties_for_query(query, CONFIG)
    page_path.write_text("Now relevant corrected content.\n", encoding="utf-8")
    assert negative_feedback.penalties_for_query(query, CONFIG) == {}


def test_page_ignored_without_negative_pages_fails_closed(feedback_file) -> None:
    write_feedback(feedback_file, [
        {
            "ts": "2026-06-11T10:23:35",
            "kind": "page_ignored",
            "prompt": "G32P と P24U のレビューを比較して",
            "expected_pages": ["g32p-review", "p24u-review"],
        },
    ])

    assert negative_feedback.penalties_for_query(
        "G32P と P24U のレビューを比較して", CONFIG
    ) == {}


def test_page_ignored_without_frontier_confirmation_fails_closed(feedback_file) -> None:
    write_feedback(
        feedback_file,
        [
            {
                "ts": "2026-06-11T10:23:35",
                "kind": "page_ignored",
                "prompt": "G32P と P24U のレビューを比較して",
                "negative_pages": ["p24u-review"],
            }
        ],
    )

    assert negative_feedback.penalties_for_query(
        "G32P と P24U のレビューを比較して", CONFIG
    ) == {}


def test_newer_frontier_page_ignored_overrides_older_positive_golden(feedback_file) -> None:
    golden_file = negative_feedback.GOLDEN_FILE_OVERRIDE
    assert golden_file is not None
    write_feedback(
        golden_file,
        [
            {
                "ts": "2026-07-10T08:00:00Z",
                "query": "G32P と P24U のレビューを比較して",
                "expected_pages": ["p24u-review"],
                "reviewed": True,
            }
        ],
    )
    write_feedback(
        feedback_file,
        [
            {
                "ts": "2026-07-11T08:00:00Z",
                "kind": "page_ignored",
                "prompt": "G32P と P24U のレビューを比較して",
                "negative_pages": ["p24u-review"],
                "frontier_reviewed": True,
            }
        ],
    )

    assert negative_feedback.penalties_for_query(
        "G32P と P24U のレビューを比較して", CONFIG
    ) == {"p24u-review": pytest.approx(0.85)}


def test_newer_reviewed_positive_can_supersede_older_page_ignored(feedback_file) -> None:
    golden_file = negative_feedback.GOLDEN_FILE_OVERRIDE
    assert golden_file is not None
    write_feedback(
        golden_file,
        [
            {
                "ts": "2026-07-11T08:00:00+00:00",
                "query": "G32P と P24U のレビューを比較して",
                "expected_pages": ["p24u-review"],
                "reviewed": True,
            }
        ],
    )
    write_feedback(
        feedback_file,
        [
            {
                "ts": "2026-07-10T08:00:00",
                "kind": "page_ignored",
                "prompt": "G32P と P24U のレビューを比較して",
                "negative_pages": ["p24u-review"],
                "frontier_reviewed": True,
            }
        ],
    )

    assert negative_feedback.penalties_for_query(
        "G32P と P24U のレビューを比較して", CONFIG
    ) == {}


def test_dissimilar_query_is_not_penalized(feedback_file) -> None:
    write_feedback(feedback_file, [
        {
            "ts": "2026-06-11T10:23:35",
            "kind": "injection_ignored",
            "prompt": "メニューバーにショートカットを置く設定の話",
            "expected_pages": ["irrelevant-page"],
        },
    ])
    assert negative_feedback.penalties_for_query("JT ファイルの NURBS デコード", CONFIG) == {}


def test_unrelated_kinds_are_ignored(feedback_file) -> None:
    write_feedback(feedback_file, [
        {
            "ts": "2026-06-11T10:23:35",
            "kind": "missed_candidate",
            "prompt": "メニューバーにショートカットを置く設定の話",
            "expected_pages": ["good-page"],
        },
    ])
    assert negative_feedback.penalties_for_query(
        "メニューバーにショートカットを置く設定の話", CONFIG
    ) == {}


def test_old_entries_expire(feedback_file) -> None:
    write_feedback(feedback_file, [
        {
            "ts": "2020-01-01T00:00:00",
            "kind": "injection_ignored",
            "prompt": "メニューバーにショートカットを置く設定の話",
            "expected_pages": ["irrelevant-page"],
        },
    ])
    assert negative_feedback.penalties_for_query(
        "メニューバーにショートカットを置く設定の話", CONFIG
    ) == {}


def test_disabled_config_is_noop(feedback_file) -> None:
    write_feedback(feedback_file, [
        {
            "ts": "2026-06-11T10:23:35",
            "kind": "injection_ignored",
            "prompt": "メニューバーにショートカットを置く設定の話",
            "expected_pages": ["irrelevant-page"],
        },
    ])
    config = NegativeFeedbackConfig(enabled=False)
    assert negative_feedback.penalties_for_query(
        "メニューバーにショートカットを置く設定の話", config
    ) == {}


def test_reviewed_positive_protects_page_from_penalty(feedback_file, tmp_path) -> None:
    write_feedback(feedback_file, [
        {
            "ts": "2026-06-11T10:23:35",
            "kind": "injection_ignored",
            "prompt": "保存みたいに自律的に改善できるようになってほしい",
            "expected_pages": ["wiki-save-hook", "unrelated-noise"],
        },
    ])
    write_feedback(tmp_path / "golden.jsonl", [
        {
            "query": "保存みたいに自律的に改善できるようになってほしい",
            "expected_pages": ["wiki-save-hook"],
            "negative_pages": [],
            "reviewed": True,
        },
    ])
    penalties = negative_feedback.penalties_for_query(
        "保存みたいに自律的に改善できるようになってほしい", CONFIG
    )
    assert "wiki-save-hook" not in penalties
    assert "unrelated-noise" in penalties


def test_unreviewed_golden_rows_do_not_protect(feedback_file, tmp_path) -> None:
    write_feedback(feedback_file, [
        {
            "ts": "2026-06-11T10:23:35",
            "kind": "injection_ignored",
            "prompt": "保存みたいに自律的に改善できるようになってほしい",
            "expected_pages": ["wiki-save-hook"],
        },
    ])
    write_feedback(tmp_path / "golden.jsonl", [
        {
            "query": "保存みたいに自律的に改善できるようになってほしい",
            "expected_pages": ["wiki-save-hook"],
            "negative_pages": [],
            "reviewed": False,
        },
    ])
    penalties = negative_feedback.penalties_for_query(
        "保存みたいに自律的に改善できるようになってほしい", CONFIG
    )
    assert "wiki-save-hook" in penalties


def test_apply_penalties_demotes_but_keeps_page() -> None:
    results = [page("noise", 0.08), page("relevant", 0.05)]
    adjusted = negative_feedback.apply_penalties(results, {"noise": 0.85})
    assert [p.page_id for p in adjusted] == ["relevant", "noise"]
    assert adjusted[1].score == pytest.approx(0.08 * 0.15)


def test_apply_penalties_without_matches_is_identity() -> None:
    results = [page("a", 0.08), page("b", 0.05)]
    assert negative_feedback.apply_penalties(results, {}) is results


def test_cache_invalidates_on_file_change(feedback_file) -> None:
    write_feedback(feedback_file, [
        {
            "ts": "2026-06-11T10:23:35",
            "kind": "injection_ignored",
            "prompt": "メニューバーにショートカットを置く設定の話",
            "expected_pages": ["page-one"],
        },
    ])
    first = negative_feedback.penalties_for_query(
        "メニューバーにショートカットを置く設定の話", CONFIG
    )
    assert "page-one" in first
    import os

    write_feedback(feedback_file, [
        {
            "ts": "2026-06-11T10:23:35",
            "kind": "injection_ignored",
            "prompt": "メニューバーにショートカットを置く設定の話",
            "expected_pages": ["page-two"],
        },
    ])
    os.utime(feedback_file, (feedback_file.stat().st_atime, feedback_file.stat().st_mtime + 5))
    second = negative_feedback.penalties_for_query(
        "メニューバーにショートカットを置く設定の話", CONFIG
    )
    assert "page-two" in second
    assert "page-one" not in second


def test_load_config_from_toml(tmp_path) -> None:
    config_file = tmp_path / "config.toml"
    config_file.write_text(
        "\n".join([
            "[search.negative_feedback]",
            "enabled = true",
            "similarity_threshold = 0.5",
            "penalty = 0.7",
            "max_age_days = 90",
        ]),
        encoding="utf-8",
    )
    config = load_negative_feedback_config(config_file)
    assert config.enabled is True
    assert config.similarity_threshold == pytest.approx(0.5)
    assert config.penalty == pytest.approx(0.7)
    assert config.max_age_days == 90
    assert config.kinds == ("page_ignored", "injection_ignored", "false-positive")


def test_load_config_defaults_when_missing(tmp_path) -> None:
    config = load_negative_feedback_config(tmp_path / "missing.toml")
    assert config.enabled is False
    assert config.similarity_threshold == pytest.approx(0.35)
    assert config.penalty == pytest.approx(0.85)
