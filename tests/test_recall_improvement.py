from __future__ import annotations

import json
from pathlib import Path

from llm_wiki_mcp import recall_improvement
from llm_wiki_mcp.convergence import CycleBudget
from llm_wiki_mcp.recall_improvement import PolicyProposal
from llm_wiki_mcp.recall_runtime import RecallPolicy


def test_default_improvement_models_use_ornith_and_gemma_pair(monkeypatch) -> None:
    monkeypatch.delenv("LLM_WIKI_RECALL_IMPROVEMENT_MODELS", raising=False)
    monkeypatch.setattr(recall_improvement, "load_toml_file", lambda: {})

    assert recall_improvement.configured_models() == (
        "maxwell1500/ornith-35b:Q5_K_M",
        "gemma4:26b-mxfp8",
    )


def test_ollama_proposer_accepts_direct_override_response(monkeypatch) -> None:
    class FakeResponse:
        def raise_for_status(self) -> None:
            pass

        def json(self):
            return {
                "response": "```json\n"
                + json.dumps({"search_threshold": 0.25, "read_threshold": 0.7, "top_k": 8})
                + "\n```"
            }

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            pass

        def post(self, *args, **kwargs):
            return FakeResponse()

    monkeypatch.setattr(recall_improvement.httpx, "Client", FakeClient)

    proposal = recall_improvement._call_ollama_proposer(
        model="gemma4:26b-mxfp8",
        baseline_policy=RecallPolicy(log_decisions=False),
        baseline_eval={"score": 0.0, "metrics": {}},
        failure_samples=[],
        live_summary={},
    )

    assert proposal.error == ""
    assert proposal.model == "gemma4:26b-mxfp8"
    assert proposal.overrides == {"search_threshold": 0.25, "read_threshold": 0.7}
    assert "directly" in proposal.rationale


def test_proposal_prompt_includes_adoption_gate_and_rejection_blockers() -> None:
    prompt = recall_improvement._proposal_prompt(
        model="qwen",
        baseline_policy=RecallPolicy(log_decisions=False),
        baseline_eval={
            "score": 0.0,
            "metrics": {"recall_at_3": 0.1, "waste_injection_rate": 0.8},
        },
        baseline_holdout={
            "score": 0.08,
            "metrics": {
                "recall_at_3": 0.16,
                "waste_injection_rate": 0.5,
                "latency_ms": {"p95": 3000.0},
            },
        },
        failure_samples=[],
        live_summary={},
        min_improvement=0.05,
        recent_rejection_blockers={
            "counts": {"latency_ok": 3, "holdout_recall_ok": 2},
            "runs": [],
        },
    )

    gate = prompt["adoption_gate"]
    assert "Your patch is rejected" in " ".join(prompt["rules"])
    assert "relative_gain >= 0.050" in gate["candidate_must_pass_public_checks"]["dev_improved"]
    assert gate["baseline_for_public_gate"]["dev_score"] == 0.0
    assert gate["candidate_must_pass_public_checks"]["private_stability_checks"]
    assert "holdout" not in json.dumps(gate, ensure_ascii=False)
    assert "baseline_holdout_metrics" not in prompt
    assert "baseline_holdout_score" not in prompt
    assert prompt["recent_rejection_blockers"]["counts"]["latency_ok"] == 3
    assert "holdout_recall_ok" not in prompt["recent_rejection_blockers"]["counts"]


def test_proposer_visible_rejection_blockers_hide_private_quality_signals() -> None:
    visible = recall_improvement._proposer_visible_rejection_blockers(
        {
            "counts": {
                "dev_improved": 4,
                "holdout_recall_ok": 3,
                "holdout_score_ok": 2,
                "latency_ok": 1,
            },
            "runs": [
                {
                    "run_id": "r1",
                    "counts": {
                        "holdout_recall_ok": 3,
                        "latency_ok": 1,
                    },
                }
            ],
        }
    )

    assert visible["counts"] == {"dev_improved": 4, "latency_ok": 1}
    assert visible["runs"][0]["counts"] == {"latency_ok": 1}
    assert "holdout" not in json.dumps(visible, ensure_ascii=False)


def test_candidate_blocker_summary_counts_failed_gate_checks() -> None:
    summary = recall_improvement._candidate_blocker_summary(
        [
            {"checks": {"dev_improved": False, "latency_ok": False, "holdout_score_ok": True}},
            {"checks": {"dev_improved": True, "latency_ok": False, "holdout_recall_ok": False}},
        ]
    )

    assert summary["blocked_candidates"] == 2
    assert summary["counts"] == {
        "latency_ok": 2,
        "dev_improved": 1,
        "holdout_recall_ok": 1,
    }


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


def test_run_improvement_excludes_page_ignored_from_policy_dataset(tmp_path) -> None:
    log_file = tmp_path / "recall-log.jsonl"
    log_file.write_text("", encoding="utf-8")
    feedback_file = tmp_path / "feedback.jsonl"
    feedback_file.write_text(
        json.dumps(
            {
                "kind": "page_ignored",
                "prompt": "G32P と P24U のレビューを比較して",
                "negative_pages": ["p24u-review"],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    payload = recall_improvement.run_improvement(
        log_file=log_file,
        feedback_file=feedback_file,
        models=(),
        include_heuristic=False,
        active_file=tmp_path / "active.json",
        registry_file=tmp_path / "registry.jsonl",
        runs_dir=tmp_path / "runs",
        episodes_file=tmp_path / "episodes.jsonl",
        live_episodes_file=tmp_path / "live.jsonl",
    )

    assert payload["status"] == "blocked"
    assert payload["reason"] == "no recall feedback examples available"
    assert payload["dataset"]["examples"] == 0


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
    audit_dir = tmp_path / "frontier-audits"
    frontier_calls = 0

    def approve(_prompt, _best=None):
        nonlocal frontier_calls
        frontier_calls += 1
        return {"decision": "approved", "summary": "frontier approved exact candidate"}

    apply_budget = CycleBudget(max_frontier_calls=1, max_mutations=1)
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
        frontier_budget=apply_budget,
        frontier_audit_dir=audit_dir,
        frontier_reviewer=approve,
    )

    assert payload["status"] == "applied"
    assert payload["applied"] is True
    active = json.loads(active_file.read_text(encoding="utf-8"))
    assert active["overrides"] == {"max_pages": 4}
    assert active["frontier_verdict"]["decision"] == "approved"
    assert Path(active["frontier_verdict"]["artifact_path"]).exists()
    assert registry_file.exists()
    assert episodes_file.exists()
    assert list(runs_dir.glob("*.json"))
    assert apply_budget.snapshot()["used"]["frontier"] == 1
    assert apply_budget.snapshot()["used"]["mutation"] == 1

    preserved_active = tmp_path / "preserved-active-policy.json"
    preserved_active.write_text('{"run_id":"old","overrides":{"max_pages":3}}\n')
    before = preserved_active.read_bytes()
    denied_budget = CycleBudget(max_mutations=0)
    deferred = recall_improvement.run_improvement(
        log_file=log_file,
        feedback_file=feedback_file,
        models=("qwen", "gemma"),
        include_heuristic=False,
        frontier_mode="off",
        active_file=preserved_active,
        registry_file=tmp_path / "deferred-registry.jsonl",
        runs_dir=tmp_path / "deferred-runs",
        episodes_file=tmp_path / "deferred-episodes.jsonl",
        live_episodes_file=live_file,
        frontier_budget=denied_budget,
        frontier_audit_dir=audit_dir,
        frontier_reviewer=approve,
    )
    assert deferred["status"] == "budget_deferred"
    assert deferred["applied"] is False
    assert deferred["active_policy"] is None
    assert preserved_active.read_bytes() == before
    assert denied_budget.snapshot()["used"]["mutation"] == 0
    assert denied_budget.snapshot()["used"]["frontier"] == 0
    assert frontier_calls == 1


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
    reject_budget = CycleBudget(max_frontier_calls=1, max_mutations=0)
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
        frontier_budget=reject_budget,
    )

    assert payload["status"] == "frontier_rejected"
    assert payload["applied"] is False
    assert payload["active_policy"] is None
    assert payload["frontier_audit"]["decision"] == "rejected"
    assert not active_file.exists()
    assert reject_budget.snapshot()["used"] == {
        "local": 0,
        "frontier": 1,
        "mutation": 0,
        "raw_bytes": 0,
    }

    monkeypatch.setattr(
        recall_improvement,
        "run_frontier_policy_audit",
        lambda *_args, **_kwargs: {
            "decision": "needs_retry",
            "summary": "frontier temporarily unavailable",
            "rescue_status": "pending_frontier_review",
            "human_required": False,
        },
    )
    retry_payload = recall_improvement.run_improvement(
        log_file=log_file,
        feedback_file=feedback_file,
        models=("qwen",),
        include_heuristic=False,
        frontier_mode="always",
        active_file=tmp_path / "retry-active-policy.json",
        registry_file=tmp_path / "retry-policy-registry.jsonl",
        runs_dir=tmp_path / "retry-runs",
        episodes_file=tmp_path / "retry-episodes.jsonl",
        live_episodes_file=tmp_path / "retry-live-episodes.jsonl",
    )
    assert retry_payload["status"] == "pending_frontier_review"
    assert retry_payload["applied"] is False
    assert retry_payload["active_policy"] is None

    monkeypatch.setattr(
        recall_improvement,
        "run_frontier_policy_audit",
        lambda *_args, **_kwargs: {
            "decision": "approved",
            "summary": "approval was not durably recorded",
        },
    )
    undurable_active = tmp_path / "undurable-active-policy.json"
    undurable = recall_improvement.run_improvement(
        log_file=log_file,
        feedback_file=feedback_file,
        models=("qwen",),
        include_heuristic=False,
        frontier_mode="always",
        active_file=undurable_active,
        registry_file=tmp_path / "undurable-registry.jsonl",
        runs_dir=tmp_path / "undurable-runs",
        episodes_file=tmp_path / "undurable-episodes.jsonl",
        live_episodes_file=tmp_path / "undurable-live-episodes.jsonl",
    )
    assert undurable["status"] == "pending_frontier_review"
    assert undurable["applied"] is False
    assert not undurable_active.exists()

    preserved_active = tmp_path / "approved-but-deferred-active.json"
    preserved_active.write_text('{"run_id":"old","overrides":{"max_pages":3}}\n')
    before = preserved_active.read_bytes()
    monkeypatch.setattr(
        recall_improvement,
        "run_frontier_policy_audit",
        lambda *_args, **_kwargs: {
            "decision": "approved",
            "summary": "safe",
            "human_required": False,
            "_artifact_durable": True,
            "candidate_sha256": "test-candidate",
            "_artifact_path": str(tmp_path / "approved-verdict.json"),
        },
    )
    mutation_denied = CycleBudget(max_frontier_calls=1, max_mutations=0)
    deferred_payload = recall_improvement.run_improvement(
        log_file=log_file,
        feedback_file=feedback_file,
        models=("qwen",),
        include_heuristic=False,
        frontier_mode="always",
        active_file=preserved_active,
        registry_file=tmp_path / "approved-deferred-registry.jsonl",
        runs_dir=tmp_path / "approved-deferred-runs",
        episodes_file=tmp_path / "approved-deferred-episodes.jsonl",
        live_episodes_file=tmp_path / "approved-deferred-live-episodes.jsonl",
        frontier_budget=mutation_denied,
    )
    assert deferred_payload["status"] == "budget_deferred"
    assert deferred_payload["active_policy"] is None
    assert preserved_active.read_bytes() == before
    assert mutation_denied.snapshot()["used"] == {
        "local": 0,
        "frontier": 1,
        "mutation": 0,
        "raw_bytes": 0,
    }

    monkeypatch.setattr(
        recall_improvement,
        "run_frontier_policy_audit",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("frontier budget must defer before review")
        ),
    )
    frontier_denied = CycleBudget(max_frontier_calls=0, max_mutations=1)
    frontier_deferred = recall_improvement.run_improvement(
        log_file=log_file,
        feedback_file=feedback_file,
        models=("qwen",),
        include_heuristic=False,
        frontier_mode="always",
        active_file=preserved_active,
        registry_file=tmp_path / "frontier-deferred-registry.jsonl",
        runs_dir=tmp_path / "frontier-deferred-runs",
        episodes_file=tmp_path / "frontier-deferred-episodes.jsonl",
        live_episodes_file=tmp_path / "frontier-deferred-live-episodes.jsonl",
        frontier_budget=frontier_denied,
    )
    assert frontier_deferred["status"] == "budget_deferred"
    assert preserved_active.read_bytes() == before
    assert frontier_denied.snapshot()["used"] == {
        "local": 0,
        "frontier": 0,
        "mutation": 0,
        "raw_bytes": 0,
    }


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


def test_run_due_does_not_count_page_ignored_as_policy_feedback(tmp_path) -> None:
    feedback_file = tmp_path / "feedback.jsonl"
    feedback_file.write_text(
        "\n".join(
            json.dumps(
                {
                    "kind": "page_ignored",
                    "prompt": f"mixed result {index}",
                    "negative_pages": ["noise"],
                }
            )
            for index in range(3)
        )
        + "\n",
        encoding="utf-8",
    )

    payload = recall_improvement.run_due(
        log_file=tmp_path / "recall-log.jsonl",
        feedback_file=feedback_file,
        min_total_feedback=1,
        min_new_feedback=1,
        dry_run=True,
        schedule_file=tmp_path / "schedule.json",
    )

    assert payload["status"] == "skipped"
    assert payload["feedback_count"] == 0
    assert payload["new_feedback"] == 0


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


def test_run_due_respects_interval_even_with_new_feedback(tmp_path, monkeypatch) -> None:
    feedback_file = tmp_path / "feedback.jsonl"
    feedback_file.write_text(
        "\n".join(json.dumps({"kind": "missed_candidate", "prompt": str(i)}) for i in range(20)) + "\n",
        encoding="utf-8",
    )
    schedule_file = tmp_path / "schedule-state.json"
    schedule_file.write_text(
        json.dumps(
            {
                "last_run_at": recall_improvement._now_iso(),
                "last_feedback_count": 0,
                "last_status": "applied",
            }
        ),
        encoding="utf-8",
    )

    def fail_run(**_kwargs):
        raise AssertionError("cooldown should prevent improvement run")

    monkeypatch.setattr(recall_improvement, "run_improvement", fail_run)

    payload = recall_improvement.run_due(
        log_file=tmp_path / "recall-log.jsonl",
        feedback_file=feedback_file,
        min_total_feedback=1,
        min_new_feedback=1,
        min_interval_hours=24.0,
        schedule_file=schedule_file,
        lock_file=tmp_path / "run-due.lock",
    )

    state = json.loads(schedule_file.read_text(encoding="utf-8"))
    assert payload["status"] == "skipped"
    assert payload["reasons"]["feedback_due"] is True
    assert payload["reasons"]["interval_due"] is False
    assert state["last_feedback_count"] == 0


def test_run_due_preserves_pending_frontier_state_during_backoff(tmp_path, monkeypatch) -> None:
    feedback_file = tmp_path / "feedback.jsonl"
    feedback_file.write_text(
        json.dumps({"kind": "missed_candidate", "prompt": "x"}) + "\n",
        encoding="utf-8",
    )
    schedule_file = tmp_path / "schedule-state.json"
    future = (recall_improvement.datetime.now() + recall_improvement.timedelta(hours=1)).isoformat(
        timespec="seconds"
    )
    schedule_file.write_text(
        json.dumps(
            {
                "last_status": "pending_frontier_review",
                "last_feedback_count": 0,
                "frontier_retry_candidate": "candidate",
                "frontier_retry_attempts": 1,
                "frontier_next_retry_at": future,
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        recall_improvement,
        "run_improvement",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("backoff must suppress retry")),
    )
    result = recall_improvement.run_due(
        feedback_file=feedback_file,
        log_file=tmp_path / "log.jsonl",
        min_total_feedback=1,
        min_new_feedback=1,
        schedule_file=schedule_file,
        lock_file=tmp_path / "run-due.lock",
    )

    state = json.loads(schedule_file.read_text(encoding="utf-8"))
    assert result["status"] == "skipped"
    assert state["last_status"] == "pending_frontier_review"
    assert state["frontier_retry_attempts"] == 1
    assert state["frontier_next_retry_at"] == future


def test_run_due_reopens_frontier_quarantine_without_new_feedback(
    tmp_path, monkeypatch
) -> None:
    feedback_file = tmp_path / "feedback.jsonl"
    feedback_file.write_text(
        json.dumps({"kind": "missed_candidate", "prompt": "x"}) + "\n",
        encoding="utf-8",
    )
    schedule_file = tmp_path / "schedule-state.json"
    schedule_file.write_text(
        json.dumps(
            {
                "last_status": "frontier_quarantined",
                "last_feedback_count": 1,
                "frontier_retry_candidate": "candidate",
                "frontier_retry_attempts": 3,
                "frontier_quarantine_retry_at": "2000-01-01T00:00:00",
            }
        ),
        encoding="utf-8",
    )
    calls = 0

    def recovered(**_kwargs):
        nonlocal calls
        calls += 1
        return {
            "ts": "2026-07-11T12:00:00",
            "run_id": "recovered",
            "status": "applied",
            "applied": True,
            "dataset": {"examples": 1},
        }

    monkeypatch.setattr(recall_improvement, "run_improvement", recovered)
    monkeypatch.setattr(recall_improvement, "safe_append_metric", lambda *_args, **_kwargs: None)

    result = recall_improvement.run_due(
        feedback_file=feedback_file,
        log_file=tmp_path / "log.jsonl",
        min_total_feedback=1,
        min_new_feedback=5,
        schedule_file=schedule_file,
        lock_file=tmp_path / "run-due.lock",
    )

    state = json.loads(schedule_file.read_text(encoding="utf-8"))
    assert result["status"] == "ran"
    assert calls == 1
    assert state["last_status"] == "applied"
    assert state["frontier_retry_attempts"] == 0
    assert state["frontier_quarantine_retry_at"] is None


def test_frontier_quarantine_does_not_ack_unresolved_feedback(
    tmp_path, monkeypatch
) -> None:
    feedback_file = tmp_path / "feedback.jsonl"
    feedback_file.write_text(
        "".join(
            json.dumps({"kind": "missed_candidate", "prompt": str(index)}) + "\n"
            for index in range(3)
        ),
        encoding="utf-8",
    )
    schedule_file = tmp_path / "schedule-state.json"
    schedule_file.write_text(
        json.dumps(
            {
                "last_status": "pending_frontier_review",
                "last_feedback_count": 1,
                "frontier_retry_candidate": "different-candidate",
                "frontier_retry_attempts": 2,
                "frontier_next_retry_at": "2000-01-01T00:00:00",
            }
        ),
        encoding="utf-8",
    )
    pending_result = {
        "ts": "2026-07-11T12:00:00",
        "run_id": "still-pending",
        "status": "pending_frontier_review",
        "applied": False,
        "dataset": {"examples": 3},
        "best": {"proposal": {"overrides": {"fusion_semantic": 0.7}}},
        "frontier_audit": {"summary": "temporary model outage"},
    }
    candidate_payload = {
        "overrides": {"fusion_semantic": 0.7},
        "examples": 3,
        "feedback_count": 3,
    }
    candidate_hash = recall_improvement.hashlib.sha256(
        json.dumps(candidate_payload, ensure_ascii=False, sort_keys=True, default=str).encode(
            "utf-8"
        )
    ).hexdigest()
    persisted = json.loads(schedule_file.read_text(encoding="utf-8"))
    persisted["frontier_retry_candidate"] = candidate_hash
    persisted["frontier_retry_attempts"] = 2
    schedule_file.write_text(json.dumps(persisted), encoding="utf-8")
    monkeypatch.setattr(
        recall_improvement,
        "run_improvement",
        lambda **_kwargs: pending_result,
    )
    monkeypatch.setattr(recall_improvement, "safe_append_metric", lambda *_args, **_kwargs: None)

    result = recall_improvement.run_due(
        feedback_file=feedback_file,
        log_file=tmp_path / "log.jsonl",
        min_total_feedback=1,
        min_new_feedback=1,
        schedule_file=schedule_file,
        lock_file=tmp_path / "run-due.lock",
    )

    state = json.loads(schedule_file.read_text(encoding="utf-8"))
    assert result["result"]["status"] == "frontier_quarantined"
    assert state["last_feedback_count"] == 1
    assert state["frontier_quarantine_retry_at"]


def test_run_due_budget_defer_keeps_schedule_byte_for_byte(tmp_path, monkeypatch) -> None:
    feedback_file = tmp_path / "feedback.jsonl"
    feedback_file.write_text(
        json.dumps({"kind": "missed_candidate", "prompt": "x"}) + "\n",
        encoding="utf-8",
    )
    schedule_file = tmp_path / "schedule-state.json"
    schedule_file.write_text(
        json.dumps(
            {
                "last_run_at": "2026-01-01T00:00:00",
                "last_feedback_count": 0,
                "last_status": "applied",
                "sentinel": "unchanged",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    before = schedule_file.read_bytes()
    monkeypatch.setattr(
        recall_improvement,
        "run_improvement",
        lambda **_kwargs: {
            "ts": "2026-07-10T12:00:00",
            "run_id": "deferred",
            "status": "budget_deferred",
            "applied": False,
            "dataset": {"examples": 1},
        },
    )
    monkeypatch.setattr(recall_improvement, "safe_append_metric", lambda *_args, **_kwargs: None)

    result = recall_improvement.run_due(
        feedback_file=feedback_file,
        log_file=tmp_path / "log.jsonl",
        min_total_feedback=1,
        min_new_feedback=1,
        schedule_file=schedule_file,
        lock_file=tmp_path / "run-due.lock",
    )

    assert result["status"] == "budget_deferred"
    assert result["result"]["status"] == "budget_deferred"
    assert schedule_file.read_bytes() == before


def test_run_due_skips_when_another_run_holds_lock(tmp_path, monkeypatch) -> None:
    feedback_file = tmp_path / "feedback.jsonl"
    feedback_file.write_text(
        json.dumps({"kind": "missed_candidate", "prompt": "x"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    lock_file = tmp_path / "run-due.lock"
    lock_handle = recall_improvement._try_acquire_run_due_lock(lock_file)
    assert lock_handle is not None

    def fail_run(**_kwargs):
        raise AssertionError("locked run_due must not execute improvement")

    monkeypatch.setattr(recall_improvement, "run_improvement", fail_run)

    try:
        payload = recall_improvement.run_due(
            log_file=tmp_path / "recall-log.jsonl",
            feedback_file=feedback_file,
            min_total_feedback=1,
            min_new_feedback=1,
            schedule_file=tmp_path / "schedule-state.json",
            lock_file=lock_file,
        )
    finally:
        lock_handle.close()

    assert payload["status"] == "skipped"
    assert payload["locked"] is True
    assert "already in progress" in payload["reason"]


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
