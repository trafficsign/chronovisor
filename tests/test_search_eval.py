from __future__ import annotations

import fcntl
import hashlib
import json
import multiprocessing
import os
import stat
from contextlib import contextmanager
from pathlib import Path

import pytest

from chronovisor.core.feedback_ledger import feedback_row_sha256
from chronovisor.core.reranker import RerankOutcome
from chronovisor.core.runtime_config import RerankerConfig
from chronovisor.core.search import ScoredPage
from chronovisor.decision import decision_lane_prompts
from chronovisor.decision.decision_router import canonical_agreement_signature
from chronovisor.decision.decision_schema_manifest import production_decision_schemas
from chronovisor.ingest.convergence import CycleBudget
from chronovisor.ops import golden_expand
from chronovisor.search import search_eval


def test_frontier_label_prompt_reexports_decision_implementation() -> None:
    assert (
        search_eval.build_frontier_label_prompt
        is decision_lane_prompts.build_frontier_label_prompt
    )
    assert search_eval._str_tuple is decision_lane_prompts._str_tuple
    assert search_eval._str_list is decision_lane_prompts._str_list
    assert search_eval._page_for_label is decision_lane_prompts._page_for_label
    assert search_eval._page_excerpt is decision_lane_prompts._page_excerpt
    assert (
        search_eval._candidate_label_pages
        is decision_lane_prompts._candidate_label_pages
    )


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


def _wait_for_search_label_lock(path_value, ready, start, attempted, acquired) -> None:
    ready.set()
    if not start.wait(5):
        return
    attempted.set()
    with search_eval._search_label_queue_lock(Path(path_value)):
        acquired.set()


def _assert_lock_is_available(path: Path) -> None:
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def authority(lane: str, marker: str = "a") -> dict:
    schema_name = (
        "search_label" if lane == search_eval.SEARCH_LABEL_LANE else "generic_decision"
    )
    router = {
        "source": "adopted_artifact",
        "artifact_sha256": marker * 64,
        "error": None,
        "models": ["primary", "challenger", "tie"],
    }
    return {
        "source": "adopted_local_consensus",
        "authority_version": 1,
        "lane": lane,
        "lane_contract_sha256": marker * 64,
        "lane_contract_manifest_sha256": marker * 64,
        "lane_contract_case_manifest_sha256": marker * 64,
        "policy": {
            "kind": "local_batch",
            "schema_name": schema_name,
            "mode": "enabled",
            "error": None,
        },
        "router": router,
    }


def authority_review(lane: str, marker: str = "a", **values) -> dict:
    expected = authority(lane, marker)
    review = {
        **values,
        "decision_policy": {
            **expected["policy"],
            "router_policy": expected["router"],
        },
        "local_consensus": {
            "status": "agreed",
            "ok": True,
            "agreement_sha256": marker * 64,
            "failure_class": None,
            "quarantine_reason": None,
            "votes": [
                {
                    "model": "primary",
                    "role": "primary",
                    "valid": True,
                    "signature_sha256": marker * 64,
                    "invalid_reason": None,
                },
                {
                    "model": "challenger",
                    "role": "challenger",
                    "valid": True,
                    "signature_sha256": marker * 64,
                    "invalid_reason": None,
                },
            ],
        },
    }
    schema_name = expected["policy"]["schema_name"]
    signature = canonical_agreement_signature(
        review,
        schema=production_decision_schemas()[schema_name],
    )
    agreement_sha256 = hashlib.sha256(signature.encode("utf-8")).hexdigest()
    review["local_consensus"]["agreement_sha256"] = agreement_sha256
    for vote in review["local_consensus"]["votes"]:
        vote["signature_sha256"] = agreement_sha256
    return review


def label_artifact(row: dict, review: dict) -> dict:
    injected, error = search_eval.decision_authority.current_semantic_authority(
        search_eval.SEARCH_LABEL_LANE,
        injected_reviewer=True,
    )
    assert error is None and injected is not None
    return search_eval._seal_search_review(
        kind="search_label_verdict",
        lane=search_eval.SEARCH_LABEL_LANE,
        evidence=search_eval._label_candidate_payload(row),
        review=review,
        authority=injected,
    )


def self_tune_policy(marker: str = "a", *, previous: dict | None = None) -> dict:
    lane = search_eval.SEARCH_SELF_TUNE_LANE
    previous_policy = {} if previous is None else previous
    weights = dict(search_eval.DEFAULT_FUSION_WEIGHTS)
    holdout: dict = {}
    review = authority_review(
        lane,
        marker,
        decision="approved",
        summary="safe",
    )
    evidence = {
        "baseline": {},
        "best": {"weights": weights, "locked-test": holdout},
        "guardrails": {},
        "previous_sha256": search_eval._canonical_json_sha256(previous_policy),
        "previous_summary": search_eval._self_tune_previous_summary(previous_policy),
    }
    artifact = search_eval._seal_search_review(
        kind="search_self_tune_verdict",
        lane=lane,
        evidence=evidence,
        review=review,
        authority=authority(lane, marker),
    )
    policy = {
        "version": 1,
        "created_at": "2026-07-13T00:00:00",
        "source": "search_eval.self_tune",
        "weights": weights,
        "holdout": holdout,
        "previous": previous_policy,
        "decision_artifact": artifact,
    }
    policy["policy_id"] = search_eval._self_tune_policy_id(policy)
    return policy


def test_language_and_kind_buckets() -> None:
    assert search_eval.language_bucket("Chronovisor 検索") == "mixed"
    assert search_eval.language_bucket("検索エンジン") == "ja"
    assert search_eval.language_bucket("search engine") == "en"
    assert search_eval.query_kind("短い質問?") == "short"
    assert (
        search_eval.query_kind("この検索結果はどうして外れているのかを確認したい？")
        == "question"
    )
    assert search_eval.query_kind("uv run pytest -q") == "short"


def test_build_candidates_uses_feedback_labels(tmp_path) -> None:
    feedback_file = tmp_path / "feedback.jsonl"
    log_file = tmp_path / "recall-log.jsonl"
    write_jsonl(
        feedback_file,
        [
            {
                "kind": "missed_candidate",
                "prompt": "Chronovisor 検索 ロードマップ",
                "expected_pages": ["chronovisor-search-improvement-roadmap"],
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
    assert examples[0].expected_pages == ("chronovisor-search-improvement-roadmap",)
    assert examples[0].negative_pages == ()
    assert examples[0].language == "mixed"
    assert examples[0].reviewed is False
    assert examples[1].expected_pages == ()
    assert examples[1].negative_pages == ("noisy-page",)
    assert examples[2].expected_pages == ()
    assert examples[2].negative_pages == ("p24u-review",)
    assert examples[2].kind == "page_ignored"


def test_build_candidates_excludes_only_exactly_retracted_page_feedback(
    tmp_path,
) -> None:
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
            "channels": {
                "bm25": [p.page_id for p in results],
                "semantic": [],
                "graph": [],
                "usage_prior": [],
            },
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
        return {
            "variants": {
                "bm25": {"metrics": {"examples": len(examples)}, "by_bucket": {}}
            },
            "debug_rows": [],
        }

    monkeypatch.setattr(search_eval, "evaluate_examples", fake_evaluate)

    payload = search_eval.run_report(
        golden_file=golden_file, variants=("bm25",), limit=2
    )

    assert seen["count"] == 2
    assert payload["dataset"]["examples"] == 2


def test_run_report_can_filter_auto_golden_sources(tmp_path, monkeypatch) -> None:
    golden_file = tmp_path / "golden.jsonl"
    write_jsonl(
        golden_file,
        [
            {
                "query": "manual",
                "expected_pages": ["a"],
                "source": "manual-curated-from-feedback",
                "reviewed": True,
            },
            {
                "query": "auto",
                "expected_pages": ["b"],
                "source": "recall_questions",
                "reviewed": True,
            },
        ],
    )
    seen = {}

    def fake_evaluate(examples, variants, top_n):
        seen["queries"] = [example.query for example in examples]
        return {
            "variants": {
                "bm25": {"metrics": {"examples": len(examples)}, "by_bucket": {}}
            },
            "debug_rows": [],
        }

    monkeypatch.setattr(search_eval, "evaluate_examples", fake_evaluate)

    payload = search_eval.run_report(
        golden_file=golden_file, variants=("bm25",), source_filter="manual"
    )

    assert seen["queries"] == ["manual"]
    assert payload["dataset"]["sources"] == {"manual-curated-from-feedback": 1}


def test_sealed_manifest_is_deterministic_and_never_contains_query(
    tmp_path,
) -> None:
    examples = [
        search_eval.SearchExample(
            query="private query text",
            expected_pages=("target",),
            source="manual-curated-from-feedback",
            ref="ref-1",
            reviewed=True,
        )
    ]
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"

    first_result = search_eval.write_sealed_manifest(examples, first)
    second_result = search_eval.write_sealed_manifest(examples, second)

    assert first.read_bytes() == second.read_bytes()
    assert first_result["manifest_sha256"] == second_result["manifest_sha256"]
    assert "private query text" not in first.read_text(encoding="utf-8")
    payload = json.loads(first.read_text(encoding="utf-8"))
    assert payload["examples"] == 1
    assert (
        payload["entries"][0]["query_sha256"]
        == hashlib.sha256(b"private query text").hexdigest()
    )


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
    monkeypatch.setattr(
        search_eval, "load_reranker_config", lambda: RerankerConfig(enabled=True)
    )
    monkeypatch.setattr(search_eval, "rerank_results", fake_rerank)

    payload = search_eval.run_variant("anything", "hybrid-rerank", top_n=2)

    assert [result.page_id for result in payload["results"]] == ["b", "a"]
    assert payload["channels"]["reranker"]["status"] == "applied"
    assert payload["stages"]["candidate_union"] == ["a", "b"]
    assert payload["stages"]["fused"] == ["a", "b"]
    assert payload["stages"]["reranked"] == ["b", "a"]
    assert payload["stages"]["page_gate"] == []
    assert payload["stages"]["committed"] == []
    assert payload["stages"]["observed"]["page_gate"] is True


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
            {
                "query": "local proposal",
                "expected_pages": ["unsafe"],
                "reviewed": False,
            },
            {
                "query": "frontier approved",
                "expected_pages": ["safe"],
                "reviewed": True,
            },
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

    rows = [
        json.loads(line)
        for line in output_file.read_text(encoding="utf-8").splitlines()
    ]
    assert payload["examples"] == 1
    assert rows[0]["queue_status"] == "pending_frontier_review"
    assert rows[0]["promoted_to_golden"] is False
    assert rows[0]["reviewer"] == ""
    assert golden_file.read_text(encoding="utf-8") == ""


def test_build_label_queue_uses_shared_jsonl_sidecar(tmp_path, monkeypatch) -> None:
    feedback_file = tmp_path / "feedback.jsonl"
    log_file = tmp_path / "recall-log.jsonl"
    output_file = tmp_path / "label-queue.jsonl"
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
    calls = []
    real_lock = search_eval._search_label_queue_lock

    @contextmanager
    def recording_lock(path):
        calls.append(path)
        with real_lock(path):
            yield

    monkeypatch.setattr(search_eval, "_search_label_queue_lock", recording_lock)

    search_eval.build_label_queue(
        feedback_file=feedback_file,
        log_file=log_file,
        output_file=output_file,
    )

    lock = output_file.with_suffix(".jsonl.lock")
    assert calls == [output_file]
    assert lock.exists()
    assert stat.S_IMODE(os.stat(lock).st_mode) == 0o600


def test_build_label_queue_defers_concurrent_append_and_nests_locks(
    tmp_path, monkeypatch
) -> None:
    feedback_file = tmp_path / "feedback.jsonl"
    log_file = tmp_path / "recall-log.jsonl"
    output_file = tmp_path / "label-queue.jsonl"
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
    competitor = {
        "query": "competitor",
        "expected_pages": ["competitor-page"],
        "queue_status": "pending_frontier_review",
    }
    original_build = search_eval.build_candidates
    real_queue_lock = search_eval._search_label_queue_lock
    order: list[str] = []

    def competing_build(**kwargs):
        lock = output_file.with_suffix(".jsonl.lock")
        _assert_lock_is_available(lock)
        with golden_expand._search_label_queue_lock(output_file):
            write_jsonl(output_file, [competitor])
        return original_build(**kwargs)

    @contextmanager
    def authority_lock():
        order.append("authority enter")
        try:
            yield
        finally:
            order.append("authority exit")

    @contextmanager
    def queue_lock(path):
        order.append("queue enter")
        try:
            with real_queue_lock(path):
                yield
        finally:
            order.append("queue exit")

    monkeypatch.setattr(search_eval, "build_candidates", competing_build)
    monkeypatch.setattr(search_eval, "decision_authority_lock", authority_lock)
    monkeypatch.setattr(search_eval, "_search_label_queue_lock", queue_lock)

    payload = search_eval.build_label_queue(
        feedback_file=feedback_file,
        log_file=log_file,
        output_file=output_file,
    )

    assert payload["status"] == "concurrent_update_deferred"
    assert [row["query"] for row in search_eval.read_jsonl(output_file)] == [
        "competitor"
    ]
    assert order == [
        "authority enter",
        "queue enter",
        "queue exit",
        "authority exit",
    ]


def test_search_label_sidecar_blocks_across_processes(tmp_path) -> None:
    queue_file = tmp_path / "label-queue.jsonl"
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    start = context.Event()
    attempted = context.Event()
    acquired = context.Event()
    process = context.Process(
        target=_wait_for_search_label_lock,
        args=(str(queue_file), ready, start, attempted, acquired),
    )
    process.start()
    try:
        assert ready.wait(5)
        with golden_expand._search_label_queue_lock(queue_file):
            start.set()
            assert attempted.wait(5)
            assert not acquired.wait(0.2)
        assert acquired.wait(5)
        process.join(5)
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.terminate()
            process.join(5)


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

    queue_rows = [
        json.loads(line) for line in queue_file.read_text(encoding="utf-8").splitlines()
    ]
    golden_rows = [
        json.loads(line)
        for line in golden_file.read_text(encoding="utf-8").splitlines()
    ]
    assert payload["promoted"] == 1
    assert queue_rows[0]["queue_status"] == "frontier_approved"
    assert queue_rows[0]["promoted_to_golden"] is True
    assert queue_rows[0]["reviewer"] == "frontier:test"
    assert golden_rows[0]["reviewed"] is True
    assert golden_rows[0]["reviewer"] == "frontier:test"
    assert golden_rows[0]["review_confidence"] == 0.93
    assert golden_rows[0]["expected_pages"] == ["target"]
    assert queue_rows[0]["decision_artifact"]["schema_version"] == 2
    assert golden_rows[0]["decision_artifact"] == queue_rows[0]["decision_artifact"]


def test_frontier_review_reopens_when_callback_appends_to_queue(
    tmp_path, monkeypatch
) -> None:
    queue_file = tmp_path / "label-queue.jsonl"
    golden_file = tmp_path / "search-golden.jsonl"
    initial = {
        "query": "query",
        "expected_pages": ["target"],
        "negative_pages": [],
        "stale_pages": [],
        "queue_status": "pending_frontier_review",
        "promoted_to_golden": False,
    }
    competitor = {
        "query": "competitor",
        "expected_pages": ["competitor-page"],
        "negative_pages": [],
        "stale_pages": [],
        "queue_status": "pending_frontier_review",
        "promoted_to_golden": False,
    }
    write_jsonl(queue_file, [initial])
    golden_file.write_bytes(b"")
    real_queue_lock = search_eval._search_label_queue_lock
    order: list[str] = []

    def reviewer(_row):
        _assert_lock_is_available(queue_file.with_suffix(".jsonl.lock"))
        with golden_expand._search_label_queue_lock(queue_file):
            write_jsonl(queue_file, [initial, competitor])
        return {
            "decision": "approved",
            "confidence": 0.95,
            "expected_pages": ["target"],
            "negative_pages": [],
            "stale_pages": [],
            "summary": "trusted",
            "notes": None,
        }

    @contextmanager
    def authority_lock():
        order.append("authority enter")
        try:
            yield
        finally:
            order.append("authority exit")

    @contextmanager
    def queue_lock(path):
        order.append("queue enter")
        try:
            with real_queue_lock(path):
                yield
        finally:
            order.append("queue exit")

    monkeypatch.setattr(search_eval, "decision_authority_lock", authority_lock)
    monkeypatch.setattr(search_eval, "_search_label_queue_lock", queue_lock)

    payload = search_eval.review_label_queue_with_frontier(
        queue_file=queue_file,
        golden_file=golden_file,
        reviewer=reviewer,
    )

    assert payload["status"] == "concurrent_update_deferred"
    assert [row["query"] for row in search_eval.read_jsonl(queue_file)] == [
        "query",
        "competitor",
    ]
    assert golden_file.read_bytes() == b""
    assert order[-4:] == [
        "authority enter",
        "queue enter",
        "queue exit",
        "authority exit",
    ]


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

    queue_row = {
        "query": "legacy",
        "expected_pages": ["target"],
        "negative_pages": [],
        "stale_pages": [],
        "queue_status": "frontier_approved",
        "promoted_to_golden": True,
        "frontier_review": trusted_review,
    }
    queue_row["decision_artifact"] = label_artifact(queue_row, trusted_review)

    # Queue-first crash: a sealed acknowledgement exists but golden is absent.
    write_jsonl(
        queue_file,
        [queue_row],
    )
    result = search_eval.review_label_queue_with_frontier(
        queue_file=queue_file,
        golden_file=golden_file,
        reviewer=lambda _row: (_ for _ in ()).throw(
            AssertionError("must not review again")
        ),
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
    review_calls = 0

    def reject_false_recovery(_row):
        nonlocal review_calls
        review_calls += 1
        raise AssertionError("different candidate must be reviewed")

    result = search_eval.review_label_queue_with_frontier(
        queue_file=queue_file,
        golden_file=golden_file,
        reviewer=reject_false_recovery,
    )
    queue_row = json.loads(queue_file.read_text().splitlines()[0])
    assert result["recovered"] == 0
    assert result["attempted"] == 1
    assert review_calls == 1
    assert queue_row["queue_status"] == "frontier_retry"
    assert queue_row["promoted_to_golden"] is False
    assert "authority_recovery" not in queue_row


def test_frontier_review_denies_stale_sealed_queue_verdict(
    tmp_path, monkeypatch
) -> None:
    queue_file = tmp_path / "queue.jsonl"
    golden_file = tmp_path / "golden.jsonl"
    lane = search_eval.SEARCH_LABEL_LANE
    row = {
        "query": "stale",
        "expected_pages": ["target"],
        "negative_pages": [],
        "stale_pages": [],
        "queue_status": "frontier_approved",
        "promoted_to_golden": True,
    }
    review = authority_review(
        lane,
        "a",
        decision="approved",
        confidence=0.9,
        expected_pages=["target"],
        negative_pages=[],
        stale_pages=[],
        summary="old",
        notes=None,
    )
    row["frontier_review"] = review
    row["decision_artifact"] = search_eval._seal_search_review(
        kind="search_label_verdict",
        lane=lane,
        evidence=search_eval._label_candidate_payload(row),
        review=review,
        authority=authority(lane, "a"),
    )
    write_jsonl(queue_file, [row])
    monkeypatch.setattr(
        search_eval.decision_authority,
        "current_semantic_authority",
        lambda *_args, **_kwargs: (authority(lane, "b"), None),
    )
    calls = 0

    def reviewer(_row):
        nonlocal calls
        calls += 1
        raise RuntimeError("do not reuse stale verdict")

    result = search_eval.review_label_queue_with_frontier(
        queue_file=queue_file,
        golden_file=golden_file,
        reviewer=reviewer,
    )

    stored = search_eval.read_jsonl(queue_file)[0]
    assert result["promoted"] == 0
    assert calls == 1
    assert not search_eval.read_jsonl(golden_file)
    assert stored["queue_status"] == "frontier_retry"
    assert "decision_artifact" not in stored


def test_frontier_review_revalidates_authority_inside_final_effect_lock(
    tmp_path, monkeypatch
) -> None:
    queue_file = tmp_path / "queue.jsonl"
    golden_file = tmp_path / "golden.jsonl"
    lane = search_eval.SEARCH_LABEL_LANE
    write_jsonl(
        queue_file,
        [
            {
                "query": "race",
                "expected_pages": ["target"],
                "negative_pages": [],
                "stale_pages": [],
                "queue_status": "pending_frontier_review",
                "promoted_to_golden": False,
            }
        ],
    )
    calls = 0

    def current(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return (authority(lane, "a" if calls <= 2 else "b"), None)

    monkeypatch.setattr(
        search_eval.decision_authority,
        "current_semantic_authority",
        current,
    )
    review = authority_review(
        lane,
        "a",
        decision="approved",
        confidence=0.99,
        expected_pages=["target"],
        negative_pages=[],
        stale_pages=[],
        summary="safe",
        notes=None,
    )

    result = search_eval.review_label_queue_with_frontier(
        queue_file=queue_file,
        golden_file=golden_file,
        reviewer=lambda _row: review,
    )

    stored = search_eval.read_jsonl(queue_file)[0]
    assert calls == 3
    assert result["promoted"] == 0
    assert result["status_counts"] == {"frontier_retry": 1}
    assert not search_eval.read_jsonl(golden_file)
    assert stored["queue_status"] == "frontier_retry"
    assert "decision_artifact" not in stored


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

    queue_rows = [
        json.loads(line) for line in queue_file.read_text(encoding="utf-8").splitlines()
    ]
    assert payload["promoted"] == 1
    assert payload["status_counts"] == {"frontier_approved": 1}
    assert queue_rows[0]["queue_status"] == "frontier_approved"
    assert search_eval.read_jsonl(golden_file)[0]["expected_pages"] == ["target"]


def test_frontier_review_cannot_approve_invented_label_ids(tmp_path) -> None:
    queue_file = tmp_path / "queue.jsonl"
    golden_file = tmp_path / "golden.jsonl"
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

    result = search_eval.review_label_queue_with_frontier(
        queue_file=queue_file,
        golden_file=golden_file,
        reviewer=lambda _row: {
            "decision": "approved",
            "confidence": 1.0,
            "expected_pages": ["invented"],
            "negative_pages": [],
            "stale_pages": [],
            "summary": "invented replacement",
            "notes": None,
        },
    )

    stored = search_eval.read_jsonl(queue_file)[0]
    assert result["promoted"] == 0
    assert result["status_counts"] == {"frontier_retry": 1}
    assert not search_eval.read_jsonl(golden_file)
    assert stored["queue_status"] == "frontier_retry"
    assert "decision_artifact" not in stored


def test_frontier_review_requires_exact_unmodified_label_arrays(tmp_path) -> None:
    queue_file = tmp_path / "queue.jsonl"
    golden_file = tmp_path / "golden.jsonl"
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

    result = search_eval.review_label_queue_with_frontier(
        queue_file=queue_file,
        golden_file=golden_file,
        reviewer=lambda _row: {
            "decision": "approved",
            "confidence": 1.0,
            "expected_pages": ["target", "target"],
            "negative_pages": [],
            "stale_pages": [],
            "summary": "duplicated candidate",
            "notes": None,
        },
    )

    stored = search_eval.read_jsonl(queue_file)[0]
    assert result["promoted"] == 0
    assert not search_eval.read_jsonl(golden_file)
    assert stored["queue_status"] == "frontier_retry"
    assert stored["decision_artifact"]["review"]["decision"] == "needs_retry"


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

    queue_rows = [
        json.loads(line) for line in queue_file.read_text(encoding="utf-8").splitlines()
    ]
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

    queue_rows = [
        json.loads(line) for line in queue_file.read_text(encoding="utf-8").splitlines()
    ]
    assert payload["promoted"] == 0
    assert queue_rows[0]["queue_status"] == "frontier_retry"
    assert queue_rows[0]["frontier_review"]["human_required"] is False


def test_legacy_unsealed_human_label_is_rereviewed_before_quarantine(
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
    assert payload["attempted"] == 1
    assert queue_row["queue_status"] == "frontier_quarantined"
    assert "human_boundary_reclassified_at" in queue_row
    assert queue_row["decision_artifact"]["schema_version"] == 2


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

    rows = [
        json.loads(line)
        for line in output_file.read_text(encoding="utf-8").splitlines()
    ]
    assert payload["failures"] == 1
    assert rows[0]["failed_stage"] == "fusion"
    assert rows[0]["reason_code"] == "fusion_missed"


def test_failure_index_uses_explicit_stage_trace(tmp_path) -> None:
    output_file = tmp_path / "failures.jsonl"
    debug_rows = [
        {
            "variant": "hybrid-rerank",
            "query": "query",
            "split": "locked-test",
            "language": "ja",
            "kind": "question",
            "expected_pages": ["target"],
            "result_pages": ["other"],
            "channels": {"bm25": ["target"], "semantic": ["other"]},
            "stages": {
                "candidate_union": ["target", "other"],
                "fused": ["target", "other"],
                "reranked": ["other"],
                "page_gate": None,
                "committed": None,
                "host_used": None,
            },
        }
    ]

    search_eval.write_failure_index(debug_rows, output_file)

    row = json.loads(output_file.read_text(encoding="utf-8").splitlines()[0])
    assert row["failed_stage"] == "reranker"
    assert row["reason_code"] == "reranker_missed"
    assert row["fix_kind"] == "reranker"


def test_failure_index_distinguishes_top_50_from_evaluation_cutoff(
    tmp_path,
) -> None:
    output_file = tmp_path / "failures.jsonl"
    below_cutoff = [f"noise-{index}" for index in range(20)] + ["target"]
    debug_rows = [
        {
            "variant": "hybrid-rerank",
            "query": "query",
            "split": "locked-test",
            "language": "ja",
            "kind": "question",
            "expected_pages": ["target"],
            "result_pages": below_cutoff,
            "channels": {"bm25": ["target"]},
            "stages": {
                "candidate_union": ["target"],
                "fused": ["target"],
                "reranked": below_cutoff,
                "page_gate": None,
                "committed": None,
                "host_used": None,
            },
        }
    ]

    search_eval.write_failure_index(debug_rows, output_file)

    row = json.loads(output_file.read_text(encoding="utf-8").splitlines()[0])
    assert row["failed_stage"] == "rank_cutoff"
    assert row["reason_code"] == "below_evaluation_cutoff"


@pytest.mark.parametrize(
    ("stages", "failed_stage", "reason_code"),
    [
        (
            {
                "candidate_union": ["target"],
                "fused": ["target"],
                "reranked": ["target"],
                "page_gate": [],
                "committed": [],
                "host_used": None,
            },
            "page_gate",
            "page_gate_rejected",
        ),
        (
            {
                "candidate_union": ["target"],
                "fused": ["target"],
                "reranked": ["target"],
                "page_gate": ["target"],
                "committed": [],
                "host_used": None,
            },
            "commit",
            "commit_missed",
        ),
        (
            {
                "candidate_union": ["target"],
                "fused": ["target"],
                "reranked": ["target"],
                "page_gate": ["target"],
                "committed": ["target"],
                "host_used": [],
                "observed": {"host_used": True},
            },
            "host_used",
            "host_did_not_use",
        ),
    ],
)
def test_failure_index_classifies_post_ranking_stages(
    tmp_path, stages, failed_stage, reason_code
) -> None:
    output_file = tmp_path / "failures.jsonl"
    debug_rows = [
        {
            "variant": "hybrid-rerank",
            "query": "query",
            "split": "locked-test",
            "language": "ja",
            "kind": "question",
            "expected_pages": ["target"],
            "result_pages": ["target"],
            "channels": {"bm25": ["target"]},
            "stages": stages,
        }
    ]

    search_eval.write_failure_index(debug_rows, output_file)

    row = json.loads(output_file.read_text(encoding="utf-8").splitlines()[0])
    assert row["failed_stage"] == failed_stage
    assert row["reason_code"] == reason_code


def test_locked_e2e_artifact_seals_all_promotion_gates(tmp_path) -> None:
    examples = [
        search_eval.SearchExample(
            query=f"query-{index}",
            expected_pages=(f"page-{index}",),
            ref=f"ref-{index}",
            reviewed=True,
        )
        for index in range(94)
    ]
    metrics = {
        "recall_at_5": 0.60,
        "negative_hit_rate_at_20": 0.10,
        "latency_ms": {"max": 900.0},
        "processor": {
            "precision": 0.95,
            "related_recall": 0.60,
            "labeled_selected_pages": 40,
            "true_positive_pages": 38,
            "evidence_kind": {
                "rich": {"precision": 0.95},
                "pointer": {"precision": 0.95},
            },
        },
    }
    path = tmp_path / "locked.json"

    artifact = search_eval.write_locked_e2e_artifact(
        {
            "generated_at": "2026-07-31T00:00:00",
            "variants": {"hybrid-rerank": {"metrics": metrics}},
        },
        examples,
        path=path,
    )

    assert artifact["status"] == "failed"
    assert artifact["gates"]["sealed_manual_94"] is False
    assert artifact["precision_lower_95"] is not None
    assert (
        json.loads(path.read_text())["snapshot_sha256"] == artifact["snapshot_sha256"]
    )


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


def test_self_tune_runtime_budget_defers_without_history_mutation(
    tmp_path, monkeypatch
) -> None:
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
    monkeypatch.setattr(
        search_eval,
        "_rows_for_weight_eval",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("budget must stop evaluation")
        ),
    )

    result = search_eval.self_tune(
        golden_file=golden_file,
        history_file=history_file,
        max_elapsed_seconds=0,
    )

    assert result["status"] == "budget_deferred"
    assert not history_file.exists()


def test_run_self_tune_due_persists_budget_backoff(tmp_path, monkeypatch) -> None:
    history_file = tmp_path / "history.jsonl"
    attempt_file = tmp_path / "attempt.json"
    calls = 0

    def fake_self_tune(**_kwargs):
        nonlocal calls
        calls += 1
        return {"status": "budget_deferred", "reason": "runtime budget exhausted"}

    monkeypatch.setattr(search_eval, "self_tune", fake_self_tune)
    first = search_eval.run_self_tune_due(
        history_file=history_file,
        attempt_file=attempt_file,
        min_interval_hours=24,
    )
    second = search_eval.run_self_tune_due(
        history_file=history_file,
        attempt_file=attempt_file,
        min_interval_hours=24,
    )

    assert first["status"] == "budget_deferred"
    assert second["status"] == "skipped"
    assert second["reason"] == "budget_backoff"
    assert calls == 1
    assert json.loads(attempt_file.read_text())["next_attempt_at"]


def test_build_label_queue_reopens_unsealed_terminal_decisions(tmp_path) -> None:
    feedback_file = tmp_path / "feedback.jsonl"
    log_file = tmp_path / "log.jsonl"
    queue_file = tmp_path / "queue.jsonl"
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
    search_eval.build_label_queue(
        feedback_file=feedback_file, log_file=log_file, output_file=queue_file
    )
    row = json.loads(queue_file.read_text().splitlines()[0])
    row.update(
        {
            "queue_status": "frontier_rejected",
            "frontier_attempts": 1,
            "review_note": "wrong",
        }
    )
    write_jsonl(queue_file, [row])

    search_eval.build_label_queue(
        feedback_file=feedback_file, log_file=log_file, output_file=queue_file
    )

    refreshed = json.loads(queue_file.read_text().splitlines()[0])
    assert refreshed["queue_status"] == "frontier_retry"
    assert refreshed["frontier_attempts"] == 1
    assert refreshed["review_note"] == "wrong"
    assert "decision_artifact" not in refreshed
    assert "decision_authority_error" in refreshed


def test_frontier_label_retry_quarantines_after_bound(tmp_path) -> None:
    queue_file = tmp_path / "queue.jsonl"
    golden_file = tmp_path / "golden.jsonl"
    write_jsonl(
        queue_file,
        [
            {
                "query": "q",
                "expected_pages": ["p"],
                "queue_status": "frontier_retry",
                "frontier_attempts": 2,
            }
        ],
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


def test_frontier_label_quarantine_reopens_after_cooldown(
    tmp_path, monkeypatch
) -> None:
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
    monkeypatch.setenv("CHRONOVISOR_CONVERGENCE_QUARANTINE_RETRY_SECONDS", "1")

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
    active_policy = json.loads(policy_file.read_text())
    assert active_policy["weights"]["semantic"] == 0.7
    assert active_policy["decision_artifact"]["schema_version"] == 2
    assert (
        active_policy["decision_artifact"]["authority"]["source"]
        == "injected_reviewer_boundary"
    )
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


def test_self_tune_revalidates_authority_before_policy_and_terminal_history(
    tmp_path, monkeypatch
) -> None:
    golden_file = tmp_path / "golden.jsonl"
    history_file = tmp_path / "history.jsonl"
    policy_file = tmp_path / "policy.json"
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
        return [
            {
                "expected_pages": list(example.expected_pages),
                "negative_pages": [],
                "stale_pages": [],
                "result_pages": ["target"]
                if example.query == "locked" or float(weights["semantic"]) >= 0.7
                else ["other"],
                "latency_ms": 1,
            }
            for example in examples
        ]

    monkeypatch.setattr(search_eval, "_rows_for_weight_eval", fake_rows)
    lane = search_eval.SEARCH_SELF_TUNE_LANE
    calls = 0

    def current(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return (authority(lane, "a" if calls <= 2 else "b"), None)

    monkeypatch.setattr(
        search_eval.decision_authority,
        "current_semantic_authority",
        current,
    )
    review = authority_review(lane, "a", decision="approved", summary="safe")

    result = search_eval.self_tune(
        golden_file=golden_file,
        history_file=history_file,
        policy_file=policy_file,
        apply=True,
        frontier_mode="auto",
        frontier_reviewer=lambda _record: review,
    )

    assert calls == 3
    assert result["status"] == "frontier_retry"
    assert result["applied"] is False
    assert not policy_file.exists()
    assert not history_file.exists()


def test_self_tune_exact_postimage_recovery_is_audit_only(tmp_path) -> None:
    policy_file = tmp_path / "policy.json"
    history_file = tmp_path / "history.jsonl"
    golden_file = tmp_path / "golden.jsonl"
    policy = self_tune_policy()
    policy_file.write_text(json.dumps(policy), encoding="utf-8")
    before = policy_file.read_bytes()

    result = search_eval.self_tune(
        golden_file=golden_file,
        history_file=history_file,
        policy_file=policy_file,
    )

    assert result["status"] == "applied_recovered"
    assert result["authority_recovery"]["kind"] == "already_applied_exact_postimage"
    assert policy_file.read_bytes() == before
    assert search_eval.read_jsonl(history_file)[0]["status"] == "applied_recovered"
    assert (
        search_eval._recover_applied_self_tune_receipt(
            policy_file=policy_file,
            history_file=history_file,
        )
        is None
    )
    assert len(search_eval.read_jsonl(history_file)) == 1


def test_self_tune_recovery_rejects_spliced_policy_and_stale_receipt(
    tmp_path, monkeypatch
) -> None:
    policy_file = tmp_path / "policy.json"
    history_file = tmp_path / "history.jsonl"
    policy = self_tune_policy()
    policy["weights"] = {**policy["weights"], "semantic": 0.99}
    # Recomputing the public content id cannot make mismatched reviewed
    # evidence authorize a different policy body.
    policy["policy_id"] = search_eval._self_tune_policy_id(policy)
    policy_file.write_text(json.dumps(policy), encoding="utf-8")

    assert (
        search_eval._recover_applied_self_tune_receipt(
            policy_file=policy_file,
            history_file=history_file,
        )
        is None
    )
    assert not history_file.exists()
    lane = search_eval.SEARCH_SELF_TUNE_LANE
    monkeypatch.setattr(
        search_eval.decision_authority,
        "current_semantic_authority",
        lambda *_args, **_kwargs: (authority(lane, "a"), None),
    )
    blocked = search_eval.rollback_search_policy(
        policy_file=policy_file,
        history_file=history_file,
        expected_policy_id=policy["policy_id"],
        injected_reviewer=True,
    )
    assert blocked["status"] == "rollback_blocked"

    current = self_tune_policy()
    policy_file.write_text(json.dumps(current), encoding="utf-8")
    write_jsonl(
        history_file,
        [
            {
                "ts": "2026-07-14T00:00:00",
                "status": "applied",
                "policy_id": "newer-different-policy",
            }
        ],
    )
    assert (
        search_eval._recover_applied_self_tune_receipt(
            policy_file=policy_file,
            history_file=history_file,
        )
        is None
    )
    assert len(search_eval.read_jsonl(history_file)) == 1


def test_self_tune_recovery_receipt_uses_policy_and_history_cas(
    tmp_path, monkeypatch
) -> None:
    policy_file = tmp_path / "policy.json"
    history_file = tmp_path / "history.jsonl"
    policy = self_tune_policy()
    policy_file.write_text(json.dumps(policy), encoding="utf-8")
    original_read_jsonl = search_eval.read_jsonl

    def concurrent_history_update(path):
        rows = original_read_jsonl(path)
        if path == history_file and not rows:
            write_jsonl(
                history_file,
                [
                    {
                        "ts": "2026-07-14T00:00:00",
                        "status": "applied",
                        "policy_id": "concurrent-policy",
                    }
                ],
            )
        return rows

    monkeypatch.setattr(search_eval, "read_jsonl", concurrent_history_update)

    assert (
        search_eval._recover_applied_self_tune_receipt(
            policy_file=policy_file,
            history_file=history_file,
        )
        is None
    )
    assert original_read_jsonl(history_file) == [
        {
            "ts": "2026-07-14T00:00:00",
            "status": "applied",
            "policy_id": "concurrent-policy",
        }
    ]


def test_search_policy_rollback_requires_current_sealed_authority(
    tmp_path, monkeypatch
) -> None:
    policy_file = tmp_path / "policy.json"
    history_file = tmp_path / "history.jsonl"
    previous = {
        "version": 1,
        "source": "search_eval.self_tune",
        "holdout": {},
        "weights": {**search_eval.DEFAULT_FUSION_WEIGHTS, "semantic": 0.5},
    }
    policy = self_tune_policy(previous=previous)
    policy_file.write_text(json.dumps(policy), encoding="utf-8")
    lane = search_eval.SEARCH_SELF_TUNE_LANE
    monkeypatch.setattr(
        search_eval.decision_authority,
        "current_semantic_authority",
        lambda *_args, **_kwargs: (authority(lane, "b"), None),
    )
    before = policy_file.read_bytes()

    blocked = search_eval.rollback_search_policy(
        policy_file=policy_file,
        history_file=history_file,
        expected_policy_id=policy["policy_id"],
        injected_reviewer=True,
    )

    assert blocked["status"] == "rollback_blocked"
    assert policy_file.read_bytes() == before
    assert not history_file.exists()

    monkeypatch.setattr(
        search_eval.decision_authority,
        "current_semantic_authority",
        lambda *_args, **_kwargs: (authority(lane, "a"), None),
    )
    applied = search_eval.rollback_search_policy(
        policy_file=policy_file,
        history_file=history_file,
        expected_policy_id=policy["policy_id"],
        injected_reviewer=True,
    )

    assert applied["status"] == "rolled_back"
    assert json.loads(policy_file.read_text()) == previous
    assert search_eval.read_jsonl(history_file)[0]["status"] == "rolled_back"
