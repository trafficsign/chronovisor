from __future__ import annotations

import json

from llm_wiki_mcp import search_eval
from llm_wiki_mcp.convergence import CycleBudget
from llm_wiki_mcp.feedback_ledger import feedback_row_sha256
from llm_wiki_mcp.reranker import RerankOutcome
from llm_wiki_mcp.runtime_config import RerankerConfig
from llm_wiki_mcp.search import ScoredPage


def page(page_id: str, score: float = 1.0, *, status: str = "active") -> ScoredPage:
    return ScoredPage(
        page_id=page_id,
        title=page_id,
        folder="",
        updated="2026-06-11",
        score=score,
        status=status,
    )


def write_jsonl(path, rows) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_language_and_kind_buckets() -> None:
    assert search_eval.language_bucket("LLM Wiki 検索") == "mixed"
    assert search_eval.language_bucket("検索エンジン") == "ja"
    assert search_eval.language_bucket("search engine") == "en"
    assert search_eval.query_kind("短い質問?") == "short"
    assert search_eval.query_kind("この検索結果はどうして外れているのかを確認したい？") == "question"
    assert search_eval.query_kind("uv run pytest -q") == "short"


def test_build_candidates_uses_feedback_labels(tmp_path) -> None:
    feedback_file = tmp_path / "feedback.jsonl"
    log_file = tmp_path / "recall-log.jsonl"
    write_jsonl(
        feedback_file,
        [
            {
                "kind": "missed_candidate",
                "prompt": "LLM Wiki 検索 ロードマップ",
                "expected_pages": ["llm-wiki-search-improvement-roadmap"],
                "source": "auditor",
                "ref": "d1",
            },
            {
                "kind": "injection_ignored",
                "prompt": "軽い雑談",
                "expected_pages": ["noisy-page"],
                "source": "auditor_precision",
                "ref": "d2",
            },
            {
                "kind": "page_ignored",
                "prompt": "G32P と P24U のレビューを比較して",
                "expected_pages": ["g32p-review", "p24u-review"],
                "negative_pages": ["p24u-review"],
                "source": "content_correction",
                "ref": "d3",
            },
        ],
    )
    write_jsonl(log_file, [])

    examples = search_eval.build_candidates(
        feedback_file=feedback_file,
        log_file=log_file,
        limit=10,
    )

    assert len(examples) == 3
    assert examples[0].expected_pages == ("llm-wiki-search-improvement-roadmap",)
    assert examples[0].negative_pages == ()
    assert examples[0].language == "mixed"
    assert examples[0].reviewed is False
    assert examples[1].expected_pages == ()
    assert examples[1].negative_pages == ("noisy-page",)
    assert examples[2].expected_pages == ()
    assert examples[2].negative_pages == ("p24u-review",)
    assert examples[2].kind == "page_ignored"


def test_build_candidates_excludes_only_exactly_retracted_page_feedback(tmp_path) -> None:
    feedback_file = tmp_path / "feedback.jsonl"
    log_file = tmp_path / "recall-log.jsonl"
    legacy = {
        "kind": "page_ignored",
        "content_correction_key": "legacy",
        "prompt": "same prompt",
        "negative_pages": ["old-noise"],
        "source": "content_correction",
    }
    valid = {
        **legacy,
        "content_correction_key": "valid",
        "negative_pages": ["real-noise"],
    }
    write_jsonl(
        feedback_file,
        [
            legacy,
            valid,
            {
                "kind": "page_ignored_retracted",
                "content_correction_key": "legacy",
                "target_kind": "page_ignored",
                "target_feedback_sha256": feedback_row_sha256(legacy),
            },
        ],
    )
    write_jsonl(log_file, [])

    examples = search_eval.build_candidates(
        feedback_file=feedback_file,
        log_file=log_file,
        limit=10,
    )

    assert [(row.kind, row.negative_pages) for row in examples] == [
        ("page_ignored", ("real-noise",))
    ]


def test_evaluate_examples_reports_ranking_metrics(monkeypatch) -> None:
    examples = [
        search_eval.SearchExample(
            query="q1",
            expected_pages=("target",),
            split="dev",
            language="en",
            kind="manual",
        ),
        search_eval.SearchExample(
            query="q2",
            expected_pages=("other",),
            negative_pages=("stale",),
            stale_pages=("stale",),
            split="locked-test",
            language="ja",
            kind="manual",
        ),
    ]

    def fake_run_variant(query: str, variant: str, *, top_n: int = 20):
        if query == "q1":
            results = [page("noise"), page("target"), page("later")]
        else:
            results = [page("stale"), page("other")]
        return {
            "variant": variant,
            "results": results,
            "latency_ms": 7,
            "channels": {"bm25": [p.page_id for p in results], "semantic": [], "graph": [], "usage_prior": []},
        }

    monkeypatch.setattr(search_eval, "run_variant", fake_run_variant)

    payload = search_eval.evaluate_examples(examples, variants=("bm25",), top_n=20)

    metrics = payload["variants"]["bm25"]["metrics"]
    assert metrics["recall_at_5"] == 1.0
    assert metrics["recall_at_20"] == 1.0
    assert metrics["mrr_at_10"] == 0.5
    assert metrics["negative_hit_rate_at_20"] == 1.0
    assert metrics["stale_hit_rate_at_20"] == 1.0
    assert metrics["latency_ms"]["p95"] == 7.0
    assert payload["variants"]["bm25"]["by_bucket"]["split:dev"]["examples"] == 1


def test_ci_gate_fails_when_threshold_missed() -> None:
    payload = {
        "variants": {
            "hybrid-current": {
                "metrics": {
                    "recall_at_5": 0.4,
                    "mrr_at_10": 0.8,
                    "negative_hit_rate_at_20": 0.1,
                }
            }
        }
    }

    gate = search_eval.ci_gate(payload, min_recall_at_5=0.5)

    assert gate["status"] == "failed"
    assert gate["failures"] == ["recall_at_5"]


def test_run_report_respects_limit(tmp_path, monkeypatch) -> None:
    golden_file = tmp_path / "golden.jsonl"
    write_jsonl(
        golden_file,
        [
            {"query": "q1", "expected_pages": ["a"], "reviewed": True},
            {"query": "q2", "expected_pages": ["b"], "reviewed": True},
            {"query": "q3", "expected_pages": ["c"], "reviewed": True},
        ],
    )
    seen = {}

    def fake_evaluate(examples, variants, top_n):
        seen["count"] = len(examples)
        return {"variants": {"bm25": {"metrics": {"examples": len(examples)}, "by_bucket": {}}}, "debug_rows": []}

    monkeypatch.setattr(search_eval, "evaluate_examples", fake_evaluate)

    payload = search_eval.run_report(golden_file=golden_file, variants=("bm25",), limit=2)

    assert seen["count"] == 2
    assert payload["dataset"]["examples"] == 2


def test_run_report_can_filter_auto_golden_sources(tmp_path, monkeypatch) -> None:
    golden_file = tmp_path / "golden.jsonl"
    write_jsonl(
        golden_file,
        [
            {"query": "manual", "expected_pages": ["a"], "source": "manual-curated-from-feedback", "reviewed": True},
            {"query": "auto", "expected_pages": ["b"], "source": "recall_questions", "reviewed": True},
        ],
    )
    seen = {}

    def fake_evaluate(examples, variants, top_n):
        seen["queries"] = [example.query for example in examples]
        return {"variants": {"bm25": {"metrics": {"examples": len(examples)}, "by_bucket": {}}}, "debug_rows": []}

    monkeypatch.setattr(search_eval, "evaluate_examples", fake_evaluate)

    payload = search_eval.run_report(golden_file=golden_file, variants=("bm25",), source_filter="manual")

    assert seen["queries"] == ["manual"]
    assert payload["dataset"]["sources"] == {"manual-curated-from-feedback": 1}


def test_run_variant_filters_lifecycle_pages(monkeypatch) -> None:
    class FakeBM25:
        def build(self) -> None:
            pass

        def query(self, query: str, top_n: int = 20):
            return [
                page("old", 2.0, status="deprecated"),
                page("active", 1.0),
            ]

    monkeypatch.setattr(search_eval, "get_bm25", lambda: FakeBM25())

    payload = search_eval.run_variant("anything", "bm25", top_n=10)

    assert [result.page_id for result in payload["results"]] == ["active"]


def test_run_variant_can_apply_hybrid_reranker(monkeypatch) -> None:
    class FakeBM25:
        def build(self) -> None:
            pass

        def query(self, query: str, top_n: int = 20):
            return [page("a", 2.0), page("b", 1.0)]

    def fake_rerank(query, candidates, *, config):
        assert query == "anything"
        assert config.enabled is True
        return RerankOutcome(
            [candidates[1], candidates[0]],
            {"status": "applied", "candidate_count": 2},
        )

    monkeypatch.setattr(search_eval, "get_bm25", lambda: FakeBM25())
    monkeypatch.setattr(search_eval, "semantic_search", lambda query, top_n=20: [])
    monkeypatch.setattr(search_eval, "load_reranker_config", lambda: RerankerConfig(enabled=True))
    monkeypatch.setattr(search_eval, "rerank_results", fake_rerank)

    payload = search_eval.run_variant("anything", "hybrid-rerank", top_n=2)

    assert [result.page_id for result in payload["results"]] == ["b", "a"]
    assert payload["channels"]["reranker"]["status"] == "applied"


def test_cli_build_golden_json(tmp_path, capsys) -> None:
    feedback_file = tmp_path / "feedback.jsonl"
    log_file = tmp_path / "recall-log.jsonl"
    output_file = tmp_path / "search-golden.jsonl"
    write_jsonl(
        feedback_file,
        [
            {
                "kind": "missed_candidate",
                "prompt": "検索エンジン",
                "expected_pages": ["search-page"],
                "ref": "d1",
            }
        ],
    )
    write_jsonl(log_file, [])

    rc = search_eval.main(
        [
            "--build-golden",
            "--feedback-file",
            str(feedback_file),
            "--log-file",
            str(log_file),
            "--output-file",
            str(output_file),
            "--json",
        ]
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["examples"] == 1
    assert payload["status"] == "queued_for_frontier_review"
    assert payload["authoritative_golden_unchanged"] is True
    assert output_file.exists()
    row = json.loads(output_file.read_text(encoding="utf-8").splitlines()[0])
    assert row["queue_status"] == "pending_frontier_review"
    assert row["promoted_to_golden"] is False


def test_unreviewed_rows_are_never_loaded_as_active_golden(tmp_path) -> None:
    golden_file = tmp_path / "search-golden.jsonl"
    write_jsonl(
        golden_file,
        [
            {"query": "local proposal", "expected_pages": ["unsafe"], "reviewed": False},
            {"query": "frontier approved", "expected_pages": ["safe"], "reviewed": True},
        ],
    )

    examples = search_eval.load_examples(golden_file)

    assert [example.query for example in examples] == ["frontier approved"]


def test_legacy_build_golden_reroutes_away_from_authoritative_file(
    tmp_path,
    monkeypatch,
) -> None:
    feedback_file = tmp_path / "feedback.jsonl"
    log_file = tmp_path / "recall-log.jsonl"
    golden_file = tmp_path / "search-golden.jsonl"
    queue_file = tmp_path / "search-label-queue.jsonl"
    golden_file.write_text('{"query":"safe","expected_pages":["p"],"reviewed":true}\n')
    before = golden_file.read_bytes()
    write_jsonl(
        feedback_file,
        [
            {
                "kind": "missed_candidate",
                "prompt": "local candidate",
                "expected_pages": ["candidate"],
            }
        ],
    )
    write_jsonl(log_file, [])
    monkeypatch.setattr(search_eval, "GOLDEN_FILE", golden_file)
    monkeypatch.setattr(search_eval, "LABEL_QUEUE_FILE", queue_file)

    payload = search_eval.build_golden(
        feedback_file=feedback_file,
        log_file=log_file,
        output_file=golden_file,
    )

    assert payload["status"] == "queued_for_frontier_review"
    assert golden_file.read_bytes() == before
    assert queue_file.exists()


def test_build_label_queue_does_not_touch_golden(tmp_path) -> None:
    feedback_file = tmp_path / "feedback.jsonl"
    log_file = tmp_path / "recall-log.jsonl"
    output_file = tmp_path / "label-queue.jsonl"
    golden_file = tmp_path / "search-golden.jsonl"
    golden_file.write_text("", encoding="utf-8")
    write_jsonl(
        feedback_file,
        [
            {
                "kind": "missed_candidate",
                "prompt": "query",
                "expected_pages": ["target"],
                "ref": "r1",
            }
        ],
    )
    write_jsonl(log_file, [])

    payload = search_eval.build_label_queue(
        feedback_file=feedback_file,
        log_file=log_file,
        output_file=output_file,
    )

    rows = [json.loads(line) for line in output_file.read_text(encoding="utf-8").splitlines()]
    assert payload["examples"] == 1
    assert rows[0]["queue_status"] == "pending_frontier_review"
    assert rows[0]["promoted_to_golden"] is False
    assert rows[0]["reviewer"] == ""
    assert golden_file.read_text(encoding="utf-8") == ""


def test_frontier_review_promotes_trusted_label_queue_rows(tmp_path) -> None:
    queue_file = tmp_path / "label-queue.jsonl"
    golden_file = tmp_path / "search-golden.jsonl"
    write_jsonl(
        queue_file,
        [
            {
                "query": "query",
                "expected_pages": ["target"],
                "negative_pages": [],
                "stale_pages": [],
                "split": "dev",
                "language": "en",
                "kind": "manual",
                "source": "feedback",
                "ref": "r1",
                "queue_status": "pending_frontier_review",
                "promoted_to_golden": False,
            }
        ],
    )
    golden_file.write_text("", encoding="utf-8")

    def reviewer(row):
        assert row["query"] == "query"
        return {
            "decision": "approved",
            "confidence": 0.93,
            "expected_pages": ["target"],
            "negative_pages": [],
            "stale_pages": [],
            "summary": "target matches the query",
            "notes": "ok",
            "reviewer": "frontier:test",
        }

    payload = search_eval.review_label_queue_with_frontier(
        queue_file=queue_file,
        golden_file=golden_file,
        min_confidence=0.8,
        reviewer=reviewer,
    )

    queue_rows = [json.loads(line) for line in queue_file.read_text(encoding="utf-8").splitlines()]
    golden_rows = [json.loads(line) for line in golden_file.read_text(encoding="utf-8").splitlines()]
    assert payload["promoted"] == 1
    assert queue_rows[0]["queue_status"] == "frontier_approved"
    assert queue_rows[0]["promoted_to_golden"] is True
    assert queue_rows[0]["reviewer"] == "frontier:test"
    assert golden_rows[0]["reviewed"] is True
    assert golden_rows[0]["reviewer"] == "frontier:test"
    assert golden_rows[0]["review_confidence"] == 0.93
    assert golden_rows[0]["expected_pages"] == ["target"]


def test_frontier_review_commits_golden_before_queue_ack(tmp_path, monkeypatch) -> None:
    queue_file = tmp_path / "label-queue.jsonl"
    golden_file = tmp_path / "search-golden.jsonl"
    write_jsonl(
        queue_file,
        [
            {
                "query": "query",
                "expected_pages": ["target"],
                "negative_pages": [],
                "stale_pages": [],
                "queue_status": "pending_frontier_review",
                "promoted_to_golden": False,
            }
        ],
    )
    writes = []
    original_write = search_eval.write_jsonl

    def recording_write(path, rows):
        writes.append(path)
        original_write(path, rows)

    monkeypatch.setattr(search_eval, "write_jsonl", recording_write)
    search_eval.review_label_queue_with_frontier(
        queue_file=queue_file,
        golden_file=golden_file,
        reviewer=lambda _row: {
            "decision": "approved",
            "confidence": 0.95,
            "expected_pages": ["target"],
            "negative_pages": [],
            "stale_pages": [],
            "summary": "trusted",
            "notes": None,
        },
    )

    assert writes == [golden_file, queue_file]


def test_frontier_review_recovers_either_cross_file_crash_window(tmp_path) -> None:
    queue_file = tmp_path / "label-queue.jsonl"
    golden_file = tmp_path / "search-golden.jsonl"
    trusted_review = {
        "decision": "approved",
        "confidence": 0.94,
        "expected_pages": ["target"],
        "negative_pages": [],
        "stale_pages": [],
        "summary": "trusted",
        "notes": None,
        "reviewer": "frontier:test",
    }

    # Legacy queue-first crash: acknowledgement exists but golden is absent.
    write_jsonl(
        queue_file,
        [
            {
                "query": "legacy",
                "expected_pages": ["target"],
                "negative_pages": [],
                "stale_pages": [],
                "queue_status": "frontier_approved",
                "promoted_to_golden": True,
                "frontier_review": trusted_review,
            }
        ],
    )
    result = search_eval.review_label_queue_with_frontier(
        queue_file=queue_file,
        golden_file=golden_file,
        reviewer=lambda _row: (_ for _ in ()).throw(AssertionError("must not review again")),
    )
    assert result["recovered"] == 1
    assert json.loads(golden_file.read_text().splitlines()[0])["query"] == "legacy"

    # Golden-first crash: trusted golden exists but the queue is still pending.
    write_jsonl(
        queue_file,
        [
            {
                "query": "legacy",
                "expected_pages": ["old-candidate"],
                "negative_pages": [],
                "stale_pages": [],
                "ref": "r1",
                "queue_status": "pending_frontier_review",
                "promoted_to_golden": False,
            }
        ],
    )
    golden_row = json.loads(golden_file.read_text().splitlines()[0])
    golden_row["ref"] = "r1"
    write_jsonl(golden_file, [golden_row])
    result = search_eval.review_label_queue_with_frontier(
        queue_file=queue_file,
        golden_file=golden_file,
        reviewer=lambda _row: (_ for _ in ()).throw(AssertionError("must not review again")),
    )
    queue_row = json.loads(queue_file.read_text().splitlines()[0])
    assert result["recovered"] == 1
    assert result["attempted"] == 0
    assert queue_row["queue_status"] == "frontier_approved"
    assert queue_row["promoted_to_golden"] is True


def test_frontier_review_does_not_use_confidence_as_promotion_gate(tmp_path) -> None:
    queue_file = tmp_path / "label-queue.jsonl"
    golden_file = tmp_path / "search-golden.jsonl"
    write_jsonl(
        queue_file,
        [
            {
                "query": "query",
                "expected_pages": ["target"],
                "negative_pages": [],
                "stale_pages": [],
                "queue_status": "pending_review",
                "promoted_to_golden": False,
            }
        ],
    )

    payload = search_eval.review_label_queue_with_frontier(
        queue_file=queue_file,
        golden_file=golden_file,
        min_confidence=0.8,
        reviewer=lambda _row: {
            "decision": "approved",
            "confidence": 0.5,
            "expected_pages": ["target"],
            "negative_pages": [],
            "stale_pages": [],
            "summary": "maybe",
            "notes": None,
        },
    )

    queue_rows = [json.loads(line) for line in queue_file.read_text(encoding="utf-8").splitlines()]
    assert payload["promoted"] == 1
    assert payload["status_counts"] == {"frontier_approved": 1}
    assert queue_rows[0]["queue_status"] == "frontier_approved"
    assert search_eval.read_jsonl(golden_file)[0]["expected_pages"] == ["target"]


def test_same_label_action_is_order_independent_across_confidence_metadata() -> None:
    primary = {
        "decision": "approved",
        "confidence": 0.1,
        "expected_pages": ["target"],
        "negative_pages": [],
        "stale_pages": [],
        "summary": "primary",
        "notes": None,
    }
    challenger = {**primary, "confidence": 0.9, "summary": "challenger"}

    actions = []
    for reviews in ([primary, challenger], [challenger, primary]):
        combined = search_eval._combine_frontier_label_reviews(
            list(reviews),
            min_confidence=0.8,
        )
        actions.append(
            (
                search_eval._queue_status_for_review(
                    combined,
                    min_confidence=0.8,
                ),
                search_eval._label_tuple_from_review(combined),
            )
        )

    assert actions == [
        ("frontier_approved", (("target",), (), ())),
        ("frontier_approved", (("target",), (), ())),
    ]


def test_frontier_review_votes_require_same_label_set(tmp_path) -> None:
    queue_file = tmp_path / "label-queue.jsonl"
    golden_file = tmp_path / "search-golden.jsonl"
    write_jsonl(
        queue_file,
        [
            {
                "query": "query",
                "expected_pages": ["target"],
                "negative_pages": [],
                "stale_pages": [],
                "queue_status": "pending_frontier_review",
                "promoted_to_golden": False,
            }
        ],
    )
    responses = iter(
        [
            {
                "decision": "approved",
                "confidence": 0.9,
                "expected_pages": ["target"],
                "negative_pages": [],
                "stale_pages": [],
                "summary": "first",
                "notes": None,
            },
            {
                "decision": "approved",
                "confidence": 0.92,
                "expected_pages": ["other"],
                "negative_pages": [],
                "stale_pages": [],
                "summary": "second",
                "notes": None,
            },
        ]
    )

    payload = search_eval.review_label_queue_with_frontier(
        queue_file=queue_file,
        golden_file=golden_file,
        min_confidence=0.8,
        votes=2,
        reviewer=lambda _row: next(responses),
    )

    queue_rows = [json.loads(line) for line in queue_file.read_text(encoding="utf-8").splitlines()]
    assert payload["promoted"] == 0
    assert queue_rows[0]["queue_status"] == "frontier_uncertain"
    assert queue_rows[0]["frontier_review"]["reviewer"] == "frontier_consensus"


def test_frontier_tool_unavailable_retries_without_human_queue(tmp_path) -> None:
    queue_file = tmp_path / "label-queue.jsonl"
    golden_file = tmp_path / "search-golden.jsonl"
    write_jsonl(
        queue_file,
        [
            {
                "query": "query",
                "expected_pages": ["target"],
                "negative_pages": [],
                "stale_pages": [],
                "queue_status": "pending_frontier_review",
                "promoted_to_golden": False,
            }
        ],
    )

    payload = search_eval.review_label_queue_with_frontier(
        queue_file=queue_file,
        golden_file=golden_file,
        reviewer=lambda _row: {
            "decision": "needs_retry",
            "confidence": 0,
            "expected_pages": [],
            "negative_pages": [],
            "stale_pages": [],
            "summary": "codex executable not found",
            "notes": None,
            "frontier_failure": {
                "failure_class": "frontier_tool_unavailable",
                "rescue_status": "human_required",
                "summary": "codex executable not found",
                "human_required": True,
            },
        },
    )

    queue_rows = [json.loads(line) for line in queue_file.read_text(encoding="utf-8").splitlines()]
    assert payload["promoted"] == 0
    assert queue_rows[0]["queue_status"] == "frontier_retry"
    assert queue_rows[0]["frontier_review"]["human_required"] is False


def test_legacy_tool_unavailable_human_label_is_quarantined_without_human_wait(
    tmp_path,
) -> None:
    queue_file = tmp_path / "label-queue.jsonl"
    golden_file = tmp_path / "search-golden.jsonl"
    write_jsonl(
        queue_file,
        [
            {
                "query": "query",
                "expected_pages": ["target"],
                "negative_pages": [],
                "stale_pages": [],
                "queue_status": "human_required",
                "promoted_to_golden": False,
                "frontier_attempts": 3,
                "frontier_review": {
                    "decision": "needs_retry",
                    "human_required": True,
                    "frontier_failure": {
                        "failure_class": "frontier_tool_unavailable",
                        "human_required": True,
                    },
                },
            }
        ],
    )

    payload = search_eval.review_label_queue_with_frontier(
        queue_file=queue_file,
        golden_file=golden_file,
        reviewer=lambda _row: (_ for _ in ()).throw(AssertionError("must not review")),
    )

    queue_row = json.loads(queue_file.read_text(encoding="utf-8"))
    assert payload["attempted"] == 0
    assert queue_row["queue_status"] == "frontier_quarantined"
    assert "human_boundary_reclassified_at" in queue_row


def test_cli_frontier_review_labels_json(tmp_path, capsys, monkeypatch) -> None:
    queue_file = tmp_path / "label-queue.jsonl"
    golden_file = tmp_path / "search-golden.jsonl"
    write_jsonl(
        queue_file,
        [
            {
                "query": "query",
                "expected_pages": ["target"],
                "negative_pages": [],
                "stale_pages": [],
                "queue_status": "pending_frontier_review",
                "promoted_to_golden": False,
            }
        ],
    )
    monkeypatch.setattr(
        search_eval,
        "run_frontier_label_review",
        lambda row, **_kwargs: {
            "decision": "approved",
            "confidence": 0.91,
            "expected_pages": row["expected_pages"],
            "negative_pages": [],
            "stale_pages": [],
            "summary": "ok",
            "notes": None,
        },
    )

    rc = search_eval.main(
        [
            "--frontier-review-labels",
            "--label-queue-file",
            str(queue_file),
            "--golden-file",
            str(golden_file),
            "--json",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload["promoted"] == 1
    assert golden_file.exists()


def test_failure_index_records_missed_expected_pages(tmp_path) -> None:
    output_file = tmp_path / "failures.jsonl"
    debug_rows = [
        {
            "variant": "hybrid-current",
            "query": "query",
            "split": "dev",
            "language": "en",
            "kind": "question",
            "expected_pages": ["target"],
            "result_pages": ["other"],
            "channels": {"bm25": ["target"], "semantic": ["other"]},
        }
    ]

    payload = search_eval.write_failure_index(debug_rows, output_file)

    rows = [json.loads(line) for line in output_file.read_text(encoding="utf-8").splitlines()]
    assert payload["failures"] == 1
    assert rows[0]["failed_stage"] == "fusion"
    assert rows[0]["reason_code"] == "fusion_missed"


def test_self_tune_shadow_blocks_when_locked_regresses(tmp_path, monkeypatch) -> None:
    golden_file = tmp_path / "golden.jsonl"
    history_file = tmp_path / "self-tune.jsonl"
    write_jsonl(
        golden_file,
        [
            {
                "query": "dev",
                "expected_pages": ["target"],
                "split": "dev",
                "reviewed": True,
            },
            {
                "query": "locked",
                "expected_pages": ["target"],
                "split": "locked-test",
                "reviewed": True,
            },
        ],
    )

    def fake_rows(examples, weights):
        semantic_weight = weights["semantic"]
        rows = []
        for example in examples:
            hit = example.query == "dev" and semantic_weight == 0.8
            if example.query == "locked" and semantic_weight == 0.8:
                hit = False
            if example.query == "locked" and semantic_weight != 0.8:
                hit = True
            rows.append(
                {
                    "expected_pages": list(example.expected_pages),
                    "negative_pages": [],
                    "stale_pages": [],
                    "result_pages": ["target"] if hit else ["other"],
                    "latency_ms": 1,
                }
            )
        return rows

    monkeypatch.setattr(search_eval, "_rows_for_weight_eval", fake_rows)

    result = search_eval.self_tune(golden_file=golden_file, history_file=history_file)

    assert result["status"] == "blocked"
    assert result["applied"] is False
    assert history_file.exists()


def test_self_tune_runtime_budget_defers_without_history_mutation(tmp_path, monkeypatch) -> None:
    golden_file = tmp_path / "golden.jsonl"
    history_file = tmp_path / "self-tune.jsonl"
    write_jsonl(
        golden_file,
        [
            {"query": "dev", "expected_pages": ["target"], "split": "dev", "reviewed": True},
            {
                "query": "locked",
                "expected_pages": ["target"],
                "split": "locked-test",
                "reviewed": True,
            },
        ],
    )
    monkeypatch.setattr(
        search_eval,
        "_rows_for_weight_eval",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("budget must stop evaluation")),
    )

    result = search_eval.self_tune(
        golden_file=golden_file,
        history_file=history_file,
        max_elapsed_seconds=0,
    )

    assert result["status"] == "budget_deferred"
    assert not history_file.exists()


def test_build_label_queue_preserves_terminal_decisions(tmp_path) -> None:
    feedback_file = tmp_path / "feedback.jsonl"
    log_file = tmp_path / "log.jsonl"
    queue_file = tmp_path / "queue.jsonl"
    write_jsonl(
        feedback_file,
        [{"kind": "missed_candidate", "prompt": "query", "expected_pages": ["target"], "ref": "r1"}],
    )
    write_jsonl(log_file, [])
    search_eval.build_label_queue(feedback_file=feedback_file, log_file=log_file, output_file=queue_file)
    row = json.loads(queue_file.read_text().splitlines()[0])
    row.update({"queue_status": "frontier_rejected", "frontier_attempts": 1, "review_note": "wrong"})
    write_jsonl(queue_file, [row])

    search_eval.build_label_queue(feedback_file=feedback_file, log_file=log_file, output_file=queue_file)

    refreshed = json.loads(queue_file.read_text().splitlines()[0])
    assert refreshed["queue_status"] == "frontier_rejected"
    assert refreshed["frontier_attempts"] == 1
    assert refreshed["review_note"] == "wrong"


def test_frontier_label_retry_quarantines_after_bound(tmp_path) -> None:
    queue_file = tmp_path / "queue.jsonl"
    golden_file = tmp_path / "golden.jsonl"
    write_jsonl(
        queue_file,
        [{"query": "q", "expected_pages": ["p"], "queue_status": "frontier_retry", "frontier_attempts": 2}],
    )

    result = search_eval.review_label_queue_with_frontier(
        queue_file=queue_file,
        golden_file=golden_file,
        max_attempts=3,
        backoff_base_seconds=0,
        reviewer=lambda _row: {
            "decision": "needs_retry",
            "confidence": 0,
            "expected_pages": [],
            "negative_pages": [],
            "stale_pages": [],
            "summary": "temporary",
            "notes": None,
        },
    )

    row = json.loads(queue_file.read_text().splitlines()[0])
    assert result["status_counts"] == {"frontier_quarantined": 1}
    assert row["queue_status"] == "frontier_quarantined"


def test_frontier_label_quarantine_reopens_after_cooldown(tmp_path, monkeypatch) -> None:
    queue_file = tmp_path / "queue.jsonl"
    golden_file = tmp_path / "golden.jsonl"
    write_jsonl(
        queue_file,
        [
            {
                "query": "q",
                "expected_pages": ["p"],
                "queue_status": "frontier_quarantined",
                "frontier_attempts": 3,
                "last_attempt_at": "2000-01-01T00:00:00",
            }
        ],
    )
    monkeypatch.setenv("LLM_WIKI_CONVERGENCE_QUARANTINE_RETRY_SECONDS", "1")

    result = search_eval.review_label_queue_with_frontier(
        queue_file=queue_file,
        golden_file=golden_file,
        reviewer=lambda _row: {
            "decision": "approved",
            "confidence": 0.99,
            "expected_pages": ["p"],
            "negative_pages": [],
            "stale_pages": [],
            "summary": "recovered",
            "notes": None,
        },
    )

    row = json.loads(queue_file.read_text().splitlines()[0])
    assert result["promoted"] == 1
    assert row["queue_status"] == "frontier_approved"
    assert row["quarantine_reopen_count"] == 1
    assert row["frontier_attempts"] == 1


def test_self_tune_applies_frontier_approved_policy(tmp_path, monkeypatch) -> None:
    golden_file = tmp_path / "golden.jsonl"
    history_file = tmp_path / "history.jsonl"
    policy_file = tmp_path / "search-policy.json"
    write_jsonl(
        golden_file,
        [
            {"query": "dev", "expected_pages": ["target"], "split": "dev", "reviewed": True},
            {"query": "locked", "expected_pages": ["target"], "split": "locked-test", "reviewed": True},
        ],
    )

    def fake_rows(examples, weights):
        rows = []
        for example in examples:
            hit = example.query == "locked" or float(weights["semantic"]) >= 0.7
            rows.append(
                {
                    "expected_pages": list(example.expected_pages),
                    "negative_pages": [],
                    "stale_pages": [],
                    "result_pages": ["target"] if hit else ["other"],
                    "latency_ms": 1,
                }
            )
        return rows

    monkeypatch.setattr(search_eval, "_rows_for_weight_eval", fake_rows)
    apply_budget = CycleBudget(max_mutations=1)
    result = search_eval.self_tune(
        golden_file=golden_file,
        history_file=history_file,
        policy_file=policy_file,
        apply=True,
        frontier_mode="auto",
        frontier_reviewer=lambda _record: {"decision": "approved", "summary": "safe"},
        budget=apply_budget,
    )

    assert result["status"] == "applied"
    assert result["applied"] is True
    assert json.loads(policy_file.read_text())["weights"]["semantic"] == 0.7
    assert apply_budget.snapshot()["used"]["mutation"] == 1

    deferred_policy = tmp_path / "deferred-policy.json"
    deferred_policy.write_text(
        json.dumps({"weights": dict(search_eval.DEFAULT_FUSION_WEIGHTS)}) + "\n",
        encoding="utf-8",
    )
    before = deferred_policy.read_bytes()
    deferred_history = tmp_path / "deferred-history.jsonl"
    denied_budget = CycleBudget(max_mutations=0)
    deferred = search_eval.self_tune(
        golden_file=golden_file,
        history_file=deferred_history,
        policy_file=deferred_policy,
        apply=True,
        frontier_mode="auto",
        frontier_reviewer=lambda _record: {"decision": "approved", "summary": "safe"},
        budget=denied_budget,
    )
    assert deferred["status"] == "budget_deferred"
    assert deferred_policy.read_bytes() == before
    assert not deferred_history.exists()
    assert denied_budget.snapshot()["used"]["mutation"] == 0

    rejected_policy = tmp_path / "rejected-policy.json"
    rejected_policy.write_bytes(before)
    rejected_budget = CycleBudget(max_mutations=0)
    rejected = search_eval.self_tune(
        golden_file=golden_file,
        history_file=tmp_path / "rejected-history.jsonl",
        policy_file=rejected_policy,
        apply=True,
        frontier_mode="auto",
        frontier_reviewer=lambda _record: {"decision": "rejected", "summary": "unsafe"},
        budget=rejected_budget,
    )
    assert rejected["status"] == "frontier_rejected"
    assert rejected_policy.read_bytes() == before
    assert rejected_budget.snapshot()["used"]["mutation"] == 0
