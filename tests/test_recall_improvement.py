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
    monkeypatch.setattr(recall_improvement, "safe_append_event", lambda *_args, **_kwargs: None)

    active_file = tmp_path / "active-policy.json"
    registry_file = tmp_path / "policy-registry.jsonl"
    runs_dir = tmp_path / "runs"
    episodes_file = tmp_path / "episodes.jsonl"
    live_file = tmp_path / "live-episodes.jsonl"
    payload = recall_improvement.run_improvement(
        log_file=log_file,
        feedback_file=feedback_file,
        models=("qwen", "gemma"),
        include_heuristic=False,
        frontier_mode="off",
        active_file=active_file,
        registry_file=registry_file,
        runs_dir=runs_dir,
        episodes_file=episodes_file,
        live_episodes_file=live_file,
    )

    assert payload["status"] == "applied"
    assert payload["applied"] is True
    active = json.loads(active_file.read_text(encoding="utf-8"))
    assert active["overrides"] == {"max_pages": 4}
    assert registry_file.exists()
    assert episodes_file.exists()
    assert list(runs_dir.glob("*.json"))


def test_run_improvement_frontier_rejection_blocks_active_policy(tmp_path, monkeypatch) -> None:
    log_file = tmp_path / "recall-log.jsonl"
    feedback_file = tmp_path / "feedback.jsonl"
    _write_feedback(log_file, feedback_file)

    monkeypatch.setattr(
        recall_improvement,
        "load_policy",
        lambda _config=None: RecallPolicy(log_decisions=False, max_pages=3),
    )
    monkeypatch.setattr(recall_improvement, "safe_append_event", lambda *_args, **_kwargs: None)

    def fake_propose(**_kwargs):
        return [
            PolicyProposal(
                source="test",
                model="qwen",
                proposal_id="p1",
                summary="Risky widen",
                rationale="Expected page needs a wider top-k.",
                overrides={"max_pages": 4},
                risk="high",
                audit_recommended=True,
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
    monkeypatch.setattr(
        recall_improvement,
        "run_frontier_policy_audit",
        lambda *_args, **_kwargs: {
            "decision": "rejected",
            "summary": "holdout evidence is too thin",
            "human_required": False,
        },
    )

    active_file = tmp_path / "active-policy.json"
    payload = recall_improvement.run_improvement(
        log_file=log_file,
        feedback_file=feedback_file,
        models=("qwen",),
        include_heuristic=False,
        frontier_mode="always",
        active_file=active_file,
        registry_file=tmp_path / "policy-registry.jsonl",
        runs_dir=tmp_path / "runs",
        episodes_file=tmp_path / "episodes.jsonl",
        live_episodes_file=tmp_path / "live-episodes.jsonl",
    )

    assert payload["status"] == "frontier_rejected"
    assert payload["applied"] is False
    assert payload["active_policy"] is None
    assert payload["frontier_audit"]["decision"] == "rejected"
    assert not active_file.exists()


def test_run_due_dry_run_is_read_only(tmp_path, monkeypatch) -> None:
    feedback_file = tmp_path / "feedback.jsonl"
    feedback_file.write_text(
        json.dumps({"kind": "missed_candidate", "prompt": "x"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    schedule_file = tmp_path / "schedule-state.json"

    def fail_run(**_kwargs):
        raise AssertionError("dry-run must not execute improvement")

    monkeypatch.setattr(recall_improvement, "run_improvement", fail_run)

    payload = recall_improvement.run_due(
        log_file=tmp_path / "recall-log.jsonl",
        feedback_file=feedback_file,
        min_total_feedback=1,
        min_new_feedback=1,
        dry_run=True,
        schedule_file=schedule_file,
    )

    assert payload["status"] == "due"
    assert payload["dry_run"] is True
    assert payload["would_update_schedule"]["last_status"] == "due"
    assert not schedule_file.exists()


def test_run_due_executes_and_updates_schedule_when_due(tmp_path, monkeypatch) -> None:
    feedback_file = tmp_path / "feedback.jsonl"
    feedback_file.write_text(
        "\n".join(json.dumps({"kind": "missed_candidate", "prompt": str(i)}) for i in range(3)) + "\n",
        encoding="utf-8",
    )
    schedule_file = tmp_path / "schedule-state.json"
    seen: dict[str, object] = {}

    def fake_run_improvement(**kwargs):
        seen.update(kwargs)
        return {
            "ts": "2026-07-05T12:00:00",
            "run_id": "r1",
            "status": "applied",
            "applied": True,
            "dataset": {"examples": 3},
            "eval_cache_entries": 2,
        }

    monkeypatch.setattr(recall_improvement, "run_improvement", fake_run_improvement)
    monkeypatch.setattr(recall_improvement, "safe_append_metric", lambda *_args, **_kwargs: None)

    payload = recall_improvement.run_due(
        log_file=tmp_path / "recall-log.jsonl",
        feedback_file=feedback_file,
        models=("qwen", "gemma"),
        min_total_feedback=1,
        min_new_feedback=1,
        schedule_file=schedule_file,
    )

    state = json.loads(schedule_file.read_text(encoding="utf-8"))
    assert payload["status"] == "ran"
    assert payload["result"]["run_id"] == "r1"
    assert state["last_run_id"] == "r1"
    assert state["last_feedback_count"] == 3
    assert seen["models"] == ("qwen", "gemma")


def test_live_episode_summary_compacts_unlabeled_traffic(tmp_path) -> None:
    path = tmp_path / "live-episodes.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"ts": "2026-07-05T12:00:00", "host": "codex", "decision": "read", "pages": ["a"], "latency_ms": 10}),
                json.dumps({"ts": "2026-07-05T12:01:00", "host": "codex", "decision": "none", "pages": [], "latency_ms": 20}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = recall_improvement.live_episode_summary(path)

    assert summary["episodes"] == 2
    assert summary["decisions"] == {"read": 1, "none": 1}
    assert summary["hosts"] == {"codex": 2}
    assert summary["avg_pages"] == 0.5


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
