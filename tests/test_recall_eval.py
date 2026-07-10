from __future__ import annotations

import json
from types import SimpleNamespace

from llm_wiki_mcp import recall_eval
from llm_wiki_mcp.recall_runtime import ContextItem, RecallPolicy, RecallResult


def test_build_dataset_uses_feedback_and_snapshot(tmp_path) -> None:
    log_file = tmp_path / "recall-log.jsonl"
    feedback_file = tmp_path / "feedback.jsonl"
    log_file.write_text(
        json.dumps(
            {
                "decision_id": "d1",
                "host": "codex",
                "prompt_preview": "前回の件",
                "pages": ["old-page"],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    feedback_file.write_text(
        json.dumps(
            {
                "kind": "missed_candidate",
                "prompt": "前回の件",
                "expected_pages": ["target-page"],
                "ref": "d1",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    examples = recall_eval.build_dataset(log_file=log_file, feedback_file=feedback_file)

    assert len(examples) == 1
    assert examples[0].expected_pages == ("target-page",)
    assert examples[0].injected_pages == ("old-page",)


def test_replay_metrics_are_deterministic(monkeypatch) -> None:
    examples = [
        recall_eval.RecallExample(prompt="p1", expected_pages=("a",), kind="missed_candidate"),
        recall_eval.RecallExample(prompt="p2", injected_pages=("b",), kind="false-positive"),
    ]

    def fake_run_recall(request, policy, *, perform_search):
        page = "a" if request.prompt == "p1" else "b"
        return RecallResult(
            status="ok",
            decision="search",
            confidence=0.7,
            queries=[request.prompt],
            reasons=[],
            matched_terms={},
            context_items=[ContextItem(page_id=page, title=page, updated="", score=1.0)],
            latency_ms=5,
        )

    monkeypatch.setattr(recall_eval, "run_recall", fake_run_recall)

    payload = recall_eval.evaluate_examples(examples, policy=RecallPolicy(log_decisions=False))

    assert payload["metrics"]["recall_at_3"] == 1.0
    assert payload["metrics"]["waste_injection_rate"] == 1.0
    assert payload["metrics"]["latency_ms"]["p95"] == 5.0


def test_page_ignored_is_neither_positive_nor_prompt_false_positive(monkeypatch) -> None:
    examples = [
        recall_eval.RecallExample(
            prompt="mixed recall",
            expected_pages=("relevant-page",),
            negative_pages=("wrong-page",),
            injected_pages=("relevant-page", "wrong-page"),
            kind="page_ignored",
        ),
        recall_eval.RecallExample(
            prompt="legacy ignored",
            injected_pages=("legacy-noise",),
            kind="injection_ignored",
        ),
    ]

    def fake_run_recall(request, policy, *, perform_search):
        page_id = "wrong-page" if request.prompt == "mixed recall" else "legacy-noise"
        return RecallResult(
            status="ok",
            decision="read",
            confidence=0.8,
            queries=[request.prompt],
            reasons=[],
            matched_terms={},
            context_items=[ContextItem(page_id=page_id, title=page_id, updated="", score=1.0)],
            latency_ms=1,
        )

    monkeypatch.setattr(recall_eval, "run_recall", fake_run_recall)

    payload = recall_eval.evaluate_examples(examples, policy=RecallPolicy(log_decisions=False))

    assert payload["metrics"]["positives"] == 0
    assert payload["metrics"]["false_positives"] == 1
    assert payload["metrics"]["waste_injection_rate"] == 1.0
