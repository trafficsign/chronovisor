from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from chronovisor.core import feedback_ledger, negative_feedback
from chronovisor.core.feedback_ledger import (
    active_feedback_rows,
    feedback_row_sha256,
)
from chronovisor.core.runtime_config import (
    NegativeFeedbackConfig,
    load_negative_feedback_config,
)
from chronovisor.core.search import ScoredPage


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
    # Most tests below isolate query-conditioning behavior. Authority binding
    # itself is exercised separately with the real central validator.
    monkeypatch.setattr(
        negative_feedback,
        "trusted_negative_feedback_rows",
        lambda candidate, **_kwargs: active_feedback_rows(candidate),
    )
    return path


CONFIG = NegativeFeedbackConfig(enabled=True, similarity_threshold=0.35, penalty=0.85)


def test_similar_query_penalizes_feedback_pages(feedback_file) -> None:
    write_feedback(feedback_file, [
        {
            "ts": "2026-06-11T10:23:35",
            "kind": "false-positive",
            "frontier_reviewed": True,
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
    assert negative_feedback.penalties_for_query(row["prompt"], CONFIG) == {}


def test_page_ignored_is_not_a_standalone_true_negative(feedback_file) -> None:
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

    assert penalties == {}


def test_exact_page_ignored_retraction_preserves_other_feedback(
    feedback_file,
    tmp_path,
    monkeypatch,
) -> None:
    legacy_path = tmp_path / "legacy-noise.md"
    valid_path = tmp_path / "valid-noise.md"
    legacy_path.write_text("legacy", encoding="utf-8")
    valid_path.write_text("valid", encoding="utf-8")
    page_paths = {
        "legacy-noise": legacy_path,
        "valid-noise": valid_path,
    }
    monkeypatch.setattr(
        negative_feedback,
        "find_mutation_page",
        lambda page_id: page_paths.get(page_id),
    )
    legacy = {
        "ts": "2026-07-11T10:23:35Z",
        "kind": "page_ignored",
        "source": "content_correction",
        "content_correction_key": "legacy-key",
        "prompt": "same query",
        "negative_pages": ["legacy-noise"],
        "negative_page_hashes": {
            "legacy-noise": hashlib.sha256(legacy_path.read_bytes()).hexdigest()
        },
        "frontier_reviewed": True,
        "label_quality": "strong",
    }
    valid = {
        **legacy,
        "content_correction_key": "valid-key",
        "negative_pages": ["valid-noise"],
        "negative_page_hashes": {
            "valid-noise": hashlib.sha256(valid_path.read_bytes()).hexdigest()
        },
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
                    "label_quality": "strong",
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
                "kind": "false-positive",
                "prompt": "G32P と P24U のレビューを比較して",
                "expected_pages": ["p24u-review"],
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


def test_default_feedback_uses_process_shared_derived_cache(
    tmp_path,
    monkeypatch,
) -> None:
    feedback = tmp_path / "feedback.jsonl"
    recall_log = tmp_path / "recall-log.jsonl"
    cache = tmp_path / "negative-feedback-cache.json"
    row = {
        "ts": "2026-06-11T10:23:35",
        "kind": "false-positive",
        "frontier_reviewed": True,
        "prompt": "Chronovisor sync recall",
        "expected_pages": ["recall-page"],
    }
    write_feedback(feedback, [row])
    monkeypatch.setattr(negative_feedback, "FEEDBACK_FILE_OVERRIDE", None)
    monkeypatch.setattr(negative_feedback, "PERSISTENT_CACHE_FILE", cache)
    monkeypatch.setattr(negative_feedback, "_feedback_file", lambda: feedback)
    monkeypatch.setattr(negative_feedback, "_recall_log_file", lambda: recall_log)
    monkeypatch.setattr(negative_feedback, "_CACHE", negative_feedback._Cache())
    scanned_logs = []

    def load_rows(candidate, *, recall_log_file):
        scanned_logs.append(recall_log_file)
        return active_feedback_rows(candidate)

    monkeypatch.setattr(
        negative_feedback,
        "trusted_negative_feedback_rows",
        load_rows,
    )

    expected = negative_feedback._load_entries(CONFIG)
    assert len(expected) == 1
    assert cache.exists()
    assert scanned_logs == [recall_log]

    recall_log.write_text('{"unrelated":true}\n', encoding="utf-8")
    monkeypatch.setattr(
        negative_feedback,
        "trusted_negative_feedback_rows",
        lambda *_args, **_kwargs: pytest.fail(
            "recall-log append must not invalidate feedback cache"
        ),
    )
    assert negative_feedback._load_entries(CONFIG) == expected

    monkeypatch.setattr(negative_feedback, "_CACHE", negative_feedback._Cache())
    assert negative_feedback._load_entries(CONFIG) == expected


def test_production_penalty_requires_exact_recall_bound_negative(
    tmp_path,
    monkeypatch,
) -> None:
    feedback = tmp_path / "feedback.jsonl"
    recall_log = tmp_path / "recall.jsonl"
    page_path = tmp_path / "noise.md"
    page_path.write_text("noise", encoding="utf-8")
    prompt = "same exact query"
    prompt_hash = feedback_ledger._prompt_hash(prompt)
    decision_id = "decision-exact"
    field_evidence = {
        "field_shadow": {"topic_epoch": 0, "session_hash": "field-session"}
    }
    exact = {
        "ts": "2026-08-01T00:00:00Z",
        "kind": "page_ignored",
        "host": "codex",
        "frontier_reviewed": True,
        "label_quality": "strong",
        "content_correction_key": "correction-exact",
        "ref": decision_id,
        "prompt": prompt,
        "snapshot": {
            "decision_id": decision_id,
            "decision": "search",
            "host": "codex",
            "session_id": "session-exact",
            "prompt_hash": prompt_hash,
            "evidence_features": field_evidence,
        },
        "source_turn_ref": {
            "session_id": "session-exact",
            "prompt_hash": prompt_hash,
            "user_line": 1,
            "assistant_line": 2,
        },
        "negative_pages": ["noise"],
        "negative_page_hashes": {
            "noise": hashlib.sha256(page_path.read_bytes()).hexdigest()
        },
    }
    write_feedback(
        recall_log,
        [
            {
                "decision_id": decision_id,
                "decision": "search",
                "host": "codex",
                "session_id": "session-exact",
                "prompt_hash": prompt_hash,
                "pages": ["noise"],
                "evidence_features": field_evidence,
            }
        ],
    )
    monkeypatch.setattr(negative_feedback, "FEEDBACK_FILE_OVERRIDE", feedback)
    monkeypatch.setattr(negative_feedback, "RECALL_LOG_FILE_OVERRIDE", recall_log)
    monkeypatch.setattr(
        negative_feedback,
        "trusted_negative_feedback_rows",
        feedback_ledger.trusted_negative_feedback_rows,
    )
    monkeypatch.setattr(feedback_ledger, "find_page", lambda _page_id: page_path)
    monkeypatch.setattr(negative_feedback, "find_mutation_page", lambda _page_id: page_path)
    monkeypatch.setattr(
        feedback_ledger, "find_mutation_page", lambda _page_id: page_path
    )
    monkeypatch.setattr(negative_feedback, "_CACHE", negative_feedback._Cache())

    write_feedback(feedback, [{key: value for key, value in exact.items() if key != "ref"}])
    assert negative_feedback.penalties_for_query(prompt, CONFIG) == {}

    write_feedback(feedback, [exact])
    negative_feedback._CACHE = negative_feedback._Cache()
    assert feedback_ledger.trusted_negative_feedback_row_error(
        exact, recall_log_file=recall_log
    ) == ""
    assert negative_feedback.penalties_for_query(prompt, CONFIG) == {
        "noise": pytest.approx(0.85)
    }


def test_reviewed_positive_protects_page_from_penalty(feedback_file, tmp_path) -> None:
    write_feedback(feedback_file, [
        {
            "ts": "2026-06-11T10:23:35",
            "kind": "false-positive",
            "frontier_reviewed": True,
            "prompt": "保存みたいに自律的に改善できるようになってほしい",
            "expected_pages": ["wiki-save-hook", "unrelated-noise"],
        },
    ])
    write_feedback(tmp_path / "golden.jsonl", [
            {
                "ts": "2026-06-12T10:23:35Z",
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
            "kind": "false-positive",
            "frontier_reviewed": True,
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
            "kind": "false-positive",
            "frontier_reviewed": True,
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
            "kind": "false-positive",
            "frontier_reviewed": True,
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


def _reset_golden_memory(monkeypatch):
    monkeypatch.setattr(negative_feedback, "_PROTECT_CACHE", negative_feedback._Cache())
    monkeypatch.setattr(negative_feedback, "_GOLD_NEGATIVE_CACHE", negative_feedback._Cache())


def test_golden_sidecar_parses_once_and_reuses_tokens(feedback_file, tmp_path, monkeypatch):
    golden = tmp_path / "golden.jsonl"
    write_feedback(golden, [
        {"query": "alpha 日本語 alpha", "reviewed": True, "expected_pages": ["yes"], "negative_pages": ["no"]},
        {"query": "holdout", "reviewed": True, "split": "holdout", "negative_pages": ["hidden"]},
    ])
    _reset_golden_memory(monkeypatch)
    original = negative_feedback._parse_golden_entries
    calls = []

    def parse(text):
        calls.append(text)
        return original(text)

    monkeypatch.setattr(negative_feedback, "_parse_golden_entries", parse)
    negatives = negative_feedback._load_reviewed_train_negatives()
    positives = negative_feedback._load_protections()
    assert len(calls) == 1
    assert [e.pages for e in negatives] == [("no",)]
    assert [e.pages for e in positives] == [("yes",)]
    sidecar = golden.with_suffix(".tokens.json")
    assert sidecar.exists()
    assert sidecar.stat().st_mode & 0o777 == 0o600
    _reset_golden_memory(monkeypatch)
    monkeypatch.setattr(negative_feedback, "tokenize", lambda _: pytest.fail("cache hit retokenized"))
    assert negative_feedback._load_golden_entries() == (negatives, positives)
    assert len(calls) == 1


@pytest.mark.parametrize("corruption", ["truncated", "checksum", "invalid_entry", "schema", "tokenizer"])
def test_golden_corrupt_or_versioned_cache_falls_back(feedback_file, tmp_path, monkeypatch, corruption):
    golden = tmp_path / "golden.jsonl"
    write_feedback(golden, [{"query": "alpha", "reviewed": True, "negative_pages": ["no"]}])
    _reset_golden_memory(monkeypatch)
    expected = negative_feedback._load_golden_entries()
    sidecar = golden.with_suffix(".tokens.json")
    data = json.loads(sidecar.read_text())
    if corruption == "truncated":
        sidecar.write_text("{")
    else:
        if corruption == "checksum":
            data["entries_sha256"] = "0" * 64
        elif corruption == "invalid_entry":
            data["entries_json"] = '[{"tokens":[],"pages":["bad"],"kind":"explicit_eval_negative"}]'
            data["entries_sha256"] = hashlib.sha256(data["entries_json"].encode()).hexdigest()
        else:
            data["source"]["schema_version" if corruption == "schema" else "tokenizer_version"] = -1
        sidecar.write_text(json.dumps(data))
    _reset_golden_memory(monkeypatch)
    assert negative_feedback._load_golden_entries() == expected


def test_golden_content_hash_invalidates_preserved_stat(feedback_file, tmp_path, monkeypatch):
    import os

    golden = tmp_path / "golden.jsonl"
    write_feedback(golden, [{"query": "alpha", "reviewed": True, "negative_pages": ["no"]}])
    _reset_golden_memory(monkeypatch)
    assert negative_feedback._load_reviewed_train_negatives()[0].tokens == {"alpha"}
    stat = golden.stat()
    golden.write_text(golden.read_text().replace("alpha", "gamma"))
    os.utime(golden, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    _reset_golden_memory(monkeypatch)
    assert negative_feedback._load_reviewed_train_negatives()[0].tokens == {"gamma"}


def test_golden_concurrent_update_discards_cache(feedback_file, tmp_path, monkeypatch):
    from pathlib import Path

    golden = tmp_path / "golden.jsonl"
    write_feedback(golden, [{"query": "alpha", "reviewed": True, "negative_pages": ["no"]}])
    _reset_golden_memory(monkeypatch)
    original = Path.read_bytes
    changed = False

    def read(path):
        nonlocal changed
        content = original(path)
        if path == golden and not changed:
            changed = True
            path.write_bytes(content.replace(b"alpha", b"gamma"))
        return content

    monkeypatch.setattr(Path, "read_bytes", read)
    assert negative_feedback._load_reviewed_train_negatives()[0].tokens == {"gamma"}
    assert not golden.with_suffix(".tokens.json").exists()
    assert negative_feedback._GOLD_NEGATIVE_CACHE.key is None


def test_read_only_never_writes_feedback_sidecars(feedback_file, tmp_path, monkeypatch):
    golden = tmp_path / "golden.jsonl"
    write_feedback(golden, [{"query": "alpha", "reviewed": True, "negative_pages": ["no"]}])
    _reset_golden_memory(monkeypatch)
    monkeypatch.setenv("CHRONOVISOR_READ_ONLY", "1")
    assert negative_feedback._load_reviewed_train_negatives()
    assert not golden.with_suffix(".tokens.json").exists()
    cache = tmp_path / "missing" / "feedback-cache.json"
    monkeypatch.setattr(negative_feedback, "FEEDBACK_FILE_OVERRIDE", None)
    monkeypatch.setattr(negative_feedback, "PERSISTENT_CACHE_FILE", cache)
    negative_feedback._write_persistent_entries({}, [])
    assert not cache.parent.exists()


def test_golden_unicode_separator_preserves_legacy_views(feedback_file, tmp_path, monkeypatch):
    golden = tmp_path / "golden.jsonl"
    write_feedback(golden, [{"query": "alpha\u2028beta", "reviewed": True, "negative_pages": ["no"], "expected_pages": ["yes"]}])
    _reset_golden_memory(monkeypatch)
    negatives, positives = negative_feedback._load_golden_entries()
    assert negatives == []
    assert len(positives) == 1


def test_golden_update_during_parse_is_not_published(feedback_file, tmp_path, monkeypatch):
    golden = tmp_path / "golden.jsonl"
    write_feedback(golden, [{"query": "alpha", "reviewed": True, "negative_pages": ["no"]}])
    _reset_golden_memory(monkeypatch)
    original = negative_feedback._parse_golden_entries
    changed = False

    def parse(text):
        nonlocal changed
        values = original(text)
        if not changed:
            changed = True
            golden.write_text(golden.read_text().replace("alpha", "gamma"))
        return values

    monkeypatch.setattr(negative_feedback, "_parse_golden_entries", parse)
    assert negative_feedback._load_reviewed_train_negatives()[0].tokens == {"gamma"}
    assert not golden.with_suffix(".tokens.json").exists()
    assert negative_feedback._GOLD_NEGATIVE_CACHE.key is None


def _golden_calls(monkeypatch):
    calls = []
    original = negative_feedback._parse_golden_entries

    def parse(text):
        calls.append(text)
        return original(text)

    monkeypatch.setattr(negative_feedback, "_parse_golden_entries", parse)
    return calls, original


def _full_golden(golden: Path, original) -> tuple[list, list]:
    return negative_feedback._decode_golden_entries(
        original(golden.read_text(encoding="utf-8"))
    )


def test_golden_sidecar_stores_stat_signature_and_append_state(
    feedback_file, tmp_path, monkeypatch
) -> None:
    golden = tmp_path / "golden.jsonl"
    write_feedback(golden, [{"query": "alpha", "reviewed": True, "negative_pages": ["no"]}])
    _reset_golden_memory(monkeypatch)
    negative_feedback._load_golden_entries()
    stored = json.loads(golden.with_suffix(".tokens.json").read_text())["source"]
    stat = golden.stat()
    assert stored["size"] == stat.st_size
    assert stored["signature"] == [
        stat.st_dev,
        stat.st_ino,
        stat.st_size,
        stat.st_mtime_ns,
        stat.st_ctime_ns,
    ]
    assert stored["ends_with_newline"] is True
    assert stored["schema_version"] == negative_feedback.GOLDEN_CACHE_SCHEMA


def test_golden_append_parses_only_the_tail_and_matches_a_full_parse(
    feedback_file, tmp_path, monkeypatch
) -> None:
    golden = tmp_path / "golden.jsonl"
    write_feedback(
        golden,
        [
            {"query": "alpha 日本語", "reviewed": True, "expected_pages": ["yes"], "negative_pages": ["no"]},
            {"query": "beta\u2028separator", "reviewed": True, "negative_pages": ["sep-row"]},
        ],
    )
    _reset_golden_memory(monkeypatch)
    negative_feedback._load_golden_entries()
    appended = [
        {"query": "delta δ", "reviewed": True, "negative_pages": ["d"]},
        {"query": "epsilon\u2029para", "reviewed": True, "expected_pages": ["e"], "negative_pages": ["en"]},
        {"query": "zeta", "reviewed": True, "split": "holdout", "negative_pages": ["hidden"]},
    ]
    blob = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in appended)
    with golden.open("ab") as stream:
        stream.write(blob.encode("utf-8"))
    _reset_golden_memory(monkeypatch)
    calls, original = _golden_calls(monkeypatch)
    incremental = negative_feedback._load_golden_entries()
    assert calls == [blob]
    assert incremental == _full_golden(golden, original)
    assert [entry.pages for entry in incremental[0]] == [("no",), ("d",)]
    assert [entry.pages for entry in incremental[1]] == [("yes",), ("e",)]
    stored = json.loads(golden.with_suffix(".tokens.json").read_text())["source"]
    assert stored["size"] == golden.stat().st_size
    assert stored["source_sha256"] == hashlib.sha256(golden.read_bytes()).hexdigest()


def test_golden_append_falls_back_when_the_cached_source_lacks_a_newline(
    feedback_file, tmp_path, monkeypatch
) -> None:
    golden = tmp_path / "golden.jsonl"
    golden.write_bytes(
        b'{"query":"alpha","reviewed":true,"negative_pages":["no"]}'
    )
    _reset_golden_memory(monkeypatch)
    negative_feedback._load_golden_entries()
    stored = json.loads(golden.with_suffix(".tokens.json").read_text())["source"]
    assert stored["ends_with_newline"] is False
    with golden.open("ab") as stream:
        stream.write(b'\n{"query":"beta","reviewed":true,"negative_pages":["b"]}\n')
    _reset_golden_memory(monkeypatch)
    calls, original = _golden_calls(monkeypatch)
    entries = negative_feedback._load_golden_entries()
    assert calls == [golden.read_text(encoding="utf-8")]
    assert entries == _full_golden(golden, original)


@pytest.mark.parametrize("mutation", ["truncate", "prefix-hash", "rewrite"])
def test_golden_append_falls_back_on_truncation_and_rewrites(
    feedback_file, tmp_path, monkeypatch, mutation
) -> None:
    golden = tmp_path / "golden.jsonl"
    write_feedback(
        golden,
        [
            {"query": "alpha", "reviewed": True, "negative_pages": ["no"]},
            {"query": "beta", "reviewed": True, "negative_pages": ["b"]},
        ],
    )
    _reset_golden_memory(monkeypatch)
    negative_feedback._load_golden_entries()
    if mutation == "truncate":
        golden.write_bytes(b'{"query":"alpha","reviewed":true,"negative_pages":["no"]}\n')
    elif mutation == "prefix-hash":
        grown = golden.read_bytes().replace(b'"alpha"', b'"alphx"', 1)
        golden.write_bytes(grown + b'{"query":"gamma","reviewed":true,"negative_pages":["g"]}\n')
    else:
        write_feedback(golden, [{"query": "omega", "reviewed": True, "negative_pages": ["o"]}])
    _reset_golden_memory(monkeypatch)
    calls, original = _golden_calls(monkeypatch)
    entries = negative_feedback._load_golden_entries()
    assert calls == [golden.read_text(encoding="utf-8")]
    assert entries == _full_golden(golden, original)


def test_golden_append_after_a_crlf_terminated_cache_matches_a_full_parse(
    feedback_file, tmp_path, monkeypatch
) -> None:
    golden = tmp_path / "golden.jsonl"
    prefix = (
        '{"query":"alpha","reviewed":true,"negative_pages":["no"]}\r\n'
        '{"query":"beta","reviewed":true,"expected_pages":["y"]}\r\n'
    )
    golden.write_bytes(prefix.encode("utf-8"))
    _reset_golden_memory(monkeypatch)
    negative_feedback._load_golden_entries()
    stored = json.loads(golden.with_suffix(".tokens.json").read_text())["source"]
    assert stored["ends_with_newline"] is True
    blob = '\n{"query":"gamma","reviewed":true,"negative_pages":["g"]}\n'
    with golden.open("ab") as stream:
        stream.write(blob.encode("utf-8"))
    _reset_golden_memory(monkeypatch)
    calls, original = _golden_calls(monkeypatch)
    incremental = negative_feedback._load_golden_entries()
    assert calls == [blob]
    assert incremental == _full_golden(golden, original)


def test_golden_fast_path_never_reads_or_hashes_the_source(
    feedback_file, tmp_path, monkeypatch
) -> None:
    golden = tmp_path / "golden.jsonl"
    write_feedback(golden, [{"query": "alpha", "reviewed": True, "negative_pages": ["no"]}])
    _reset_golden_memory(monkeypatch)
    expected = negative_feedback._load_golden_entries()
    _reset_golden_memory(monkeypatch)
    original = Path.read_bytes
    reads = []

    def read(path):
        if path == golden:
            reads.append(path)
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", read)
    monkeypatch.setattr(
        negative_feedback, "_parse_golden_entries", lambda _text: pytest.fail("reparsed")
    )
    assert negative_feedback._load_golden_entries() == expected
    assert reads == []


def test_golden_in_memory_cache_survives_without_touching_disk(
    feedback_file, tmp_path, monkeypatch
) -> None:
    golden = tmp_path / "golden.jsonl"
    write_feedback(golden, [{"query": "alpha", "reviewed": True, "negative_pages": ["no"]}])
    _reset_golden_memory(monkeypatch)
    expected = negative_feedback._load_golden_entries()
    monkeypatch.setattr(
        negative_feedback, "_read_golden_cache", lambda _path: pytest.fail("read cache")
    )
    monkeypatch.setattr(Path, "read_bytes", lambda _path: pytest.fail("read source"))
    assert negative_feedback._load_golden_entries() == expected


@pytest.fixture
def incremental_feedback(tmp_path, monkeypatch):
    feedback = tmp_path / "feedback.jsonl"
    cache = tmp_path / "feedback-cache.json"
    monkeypatch.delenv("CHRONOVISOR_READ_ONLY", raising=False)
    monkeypatch.setattr(negative_feedback, "FEEDBACK_FILE_OVERRIDE", None)
    monkeypatch.setattr(negative_feedback, "PERSISTENT_CACHE_FILE", cache)
    monkeypatch.setattr(negative_feedback, "_feedback_file", lambda: feedback)
    monkeypatch.setattr(negative_feedback, "_CACHE", negative_feedback._Cache())
    row = {"kind": "false-positive", "frontier_reviewed": True,
           "prompt": "recall cache", "expected_pages": ["page"]}
    write_feedback(feedback, [row])
    scans = []

    def validate(path, **_kwargs):
        scans.append(True)
        return active_feedback_rows(path)

    monkeypatch.setattr(negative_feedback, "trusted_negative_feedback_rows", validate)
    expected = negative_feedback._load_entries(CONFIG)
    assert len(expected) == 1
    return feedback, cache, scans, expected


@pytest.mark.parametrize("read_only", [False, True])
def test_feedback_non_target_append_reuses_prefix(incremental_feedback, monkeypatch, read_only):
    feedback, cache, scans, expected = incremental_feedback
    before = cache.read_bytes(), cache.stat().st_mtime_ns
    if read_only:
        monkeypatch.setenv("CHRONOVISOR_READ_ONLY", "1")
    for _ in range(2):
        with feedback.open("a") as stream:
            stream.write('{"kind":"missed_candidate","prompt":"irrelevant"}\n')
        monkeypatch.setattr(negative_feedback, "_CACHE", negative_feedback._Cache())
        assert negative_feedback._load_entries(CONFIG) == expected
        assert len(scans) == 1
    if read_only:
        assert (cache.read_bytes(), cache.stat().st_mtime_ns) == before
    else:
        payload = json.loads(cache.read_text())
        assert payload["source"]["size"] == feedback.stat().st_size
        assert payload["source_sha256"] == hashlib.sha256(feedback.read_bytes()).hexdigest()


@pytest.mark.parametrize("kind", ["page_ignored", "contradiction", "strong-contradiction",
                                  "false-positive", "page_ignored_retracted"])
def test_feedback_target_append_fully_validates(incremental_feedback, monkeypatch, kind):
    feedback, _cache, scans, _expected = incremental_feedback
    with feedback.open("a") as stream:
        stream.write(json.dumps({"kind": kind}) + "\n")
    monkeypatch.setattr(negative_feedback, "_CACHE", negative_feedback._Cache())
    negative_feedback._load_entries(CONFIG)
    assert len(scans) == 2


def test_feedback_same_stat_prefix_tamper_fully_validates(incremental_feedback, monkeypatch):
    import os

    feedback, _cache, scans, _expected = incremental_feedback
    stat = feedback.stat()
    feedback.write_bytes(feedback.read_bytes().replace(b'recall cache', b'recall other'))
    os.utime(feedback, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    # Even the in-process key must notice preserved size/mtime tampering.
    result = negative_feedback._load_entries(CONFIG)
    assert len(scans) == 2
    assert result[0].tokens == negative_feedback._tokenize("recall other")
    monkeypatch.setattr(negative_feedback, "_CACHE", negative_feedback._Cache())
    assert negative_feedback._load_entries(CONFIG) == result
    assert len(scans) == 2


@pytest.mark.parametrize("tail", ['{"kind":"missed_candidate"}', '{broken}\n', '[]\n'])
def test_feedback_ambiguous_tail_fully_validates(incremental_feedback, monkeypatch, tail):
    feedback, _cache, scans, _expected = incremental_feedback
    with feedback.open("a") as stream:
        stream.write(tail)
    monkeypatch.setattr(negative_feedback, "_CACHE", negative_feedback._Cache())
    negative_feedback._load_entries(CONFIG)
    assert len(scans) == 2


def test_feedback_concurrent_rebuild_does_not_publish(incremental_feedback, monkeypatch):
    feedback, cache, scans, _expected = incremental_feedback
    before = cache.read_bytes()
    with feedback.open("a") as stream:
        stream.write('{"kind":"page_ignored"}\n')
    original = negative_feedback.trusted_negative_feedback_rows

    def racing_validation(path, **kwargs):
        rows = original(path, **kwargs)
        with path.open("a") as stream:
            stream.write('{"kind":"page_ignored_retracted"}\n')
        return rows

    monkeypatch.setattr(negative_feedback, "trusted_negative_feedback_rows", racing_validation)
    monkeypatch.setattr(negative_feedback, "_CACHE", negative_feedback._Cache())
    negative_feedback._load_entries(CONFIG)
    assert len(scans) == 2
    assert cache.read_bytes() == before
    assert negative_feedback._CACHE.key is None
