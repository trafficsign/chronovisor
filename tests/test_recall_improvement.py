from __future__ import annotations

import json

from llm_wiki_mcp import recall_improvement
from llm_wiki_mcp.recall_improvement import PolicyProposal
from llm_wiki_mcp.recall_runtime import RecallPolicy


def _write_feedback(log_file, feedback_file) -> None:
    log_file.write_text(
        json.dumps(
            {
                "decision_id": "d1",
                "host": "codex",
                "prompt_preview": "LLM Wiki recall policy",
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
                "prompt": "LLM Wiki recall policy",
                "expected_pages": ["target-page"],
                "ref": "d1",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def test_run_improvement_adopts_candidate_policy(tmp_path, monkeypatch) -> None:
    log_file = tmp_path / "recall-log.jsonl"
    feedback_file = tmp_path / "feedback.jsonl"
    _write_feedback(log_file, feedback_file)

    monkeypatch.setattr(
        recall_improvement,
        "load_policy",
        lambda _config=None: RecallPolicy(log_decisions=False, max_pages=3),
    )

    def fake_propose(**_kwargs):
        return [
            PolicyProposal(
                source="test",
                model="qwen",
                proposal_id="p1",
                summary="Add one page",
                rationale="Expected page needs a wider top-k.",
                overrides={"max_pages": 4},
                risk="low",
            )
        ]

    def fake_evaluate(examples, *, policy, replay):
        improved = policy.max_pages == 4
        return {
            "metrics": {
                "examples": len(examples),
                "positives": len(examples),
                "false_positives": 0,
                "recall_at_1": 1.0 if improved else 0.0,
                "recall_at_3": 1.0 if improved else 0.0,
                "mrr": 1.0 if improved else 0.0,
                "waste_injection_rate": 0.0,
                "avg_pages": float(policy.max_pages),
                "decision_counts": {"read": len(examples)},
                "latency_ms": {"p50": 10.0, "p95": 10.0, "max": 10.0},
            },
            "rows": [
                {
                    "prompt": "LLM Wiki recall policy",
                    "kind": "missed_candidate",
                    "expected_pages": ["target-page"],
                    "pages": ["target-page"] if improved else ["old-page"],
                    "decision": "read",
                    "latency_ms": 10,
                }
            ],
        }

    monkeypatch.setattr(recall_improvement, "propose_with_models", fake_propose)
    monkeypatch.setattr(recall_improvement, "evaluate_examples", fake_evaluate)

    active_file = tmp_path / "active-policy.json"
    registry_file = tmp_path / "policy-registry.jsonl"
    runs_dir = tmp_path / "runs"
    episodes_file = tmp_path / "episodes.jsonl"
    payload = recall_improvement.run_improvement(
        log_file=log_file,
        feedback_file=feedback_file,
        models=("qwen", "gemma"),
        include_heuristic=False,
        active_file=active_file,
        registry_file=registry_file,
        runs_dir=runs_dir,
        episodes_file=episodes_file,
    )

    assert payload["status"] == "applied"
    assert payload["applied"] is True
    active = json.loads(active_file.read_text(encoding="utf-8"))
    assert active["overrides"] == {"max_pages": 4}
    assert registry_file.exists()
    assert episodes_file.exists()
    assert list(runs_dir.glob("*.json"))


def test_improvement_snapshot_surfaces_active_policy(tmp_path) -> None:
    active_file = tmp_path / "active-policy.json"
    registry_file = tmp_path / "policy-registry.jsonl"
    active_file.write_text(
        json.dumps({"run_id": "r1", "overrides": {"max_pages": 4}}, ensure_ascii=False),
        encoding="utf-8",
    )
    registry_file.write_text(
        json.dumps({"run_id": "r1", "status": "applied"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    snapshot = recall_improvement.improvement_snapshot(
        active_file=active_file,
        registry_file=registry_file,
    )

    assert snapshot["status"] == "active"
    assert snapshot["active"]["run_id"] == "r1"
    assert snapshot["history"][0]["status"] == "applied"
