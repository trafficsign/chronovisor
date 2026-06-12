from __future__ import annotations

import json

from llm_wiki_mcp import search_eval
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
        ],
    )
    write_jsonl(log_file, [])

    examples = search_eval.build_candidates(
        feedback_file=feedback_file,
        log_file=log_file,
        limit=10,
    )

    assert len(examples) == 2
    assert examples[0].expected_pages == ("llm-wiki-search-improvement-roadmap",)
    assert examples[0].negative_pages == ()
    assert examples[0].language == "mixed"
    assert examples[0].reviewed is False
    assert examples[1].expected_pages == ()
    assert examples[1].negative_pages == ("noisy-page",)


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
    assert output_file.exists()


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


def test_frontier_review_keeps_low_confidence_rows_out_of_golden(tmp_path) -> None:
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
    assert payload["promoted"] == 0
    assert payload["status_counts"] == {"frontier_uncertain": 1}
    assert queue_rows[0]["queue_status"] == "frontier_uncertain"
    assert not golden_file.exists()


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


def test_frontier_review_marks_environment_failures_human_required(tmp_path) -> None:
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
    assert queue_rows[0]["queue_status"] == "human_required"


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
