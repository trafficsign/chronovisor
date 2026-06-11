from __future__ import annotations

import json

import pytest

from llm_wiki_mcp import negative_feedback
from llm_wiki_mcp.runtime_config import NegativeFeedbackConfig, load_negative_feedback_config
from llm_wiki_mcp.search import ScoredPage


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
    assert config.kinds == ("injection_ignored", "false-positive")


def test_load_config_defaults_when_missing(tmp_path) -> None:
    config = load_negative_feedback_config(tmp_path / "missing.toml")
    assert config.enabled is False
    assert config.similarity_threshold == pytest.approx(0.35)
    assert config.penalty == pytest.approx(0.85)
