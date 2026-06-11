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
    assert rows[0]["queue_status"] == "pending_review"
    assert rows[0]["promoted_to_golden"] is False
    assert golden_file.read_text(encoding="utf-8") == ""


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
