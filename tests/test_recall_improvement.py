from __future__ import annotations

import hashlib
import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from chronovisor.decision.decision_router import canonical_agreement_signature
from chronovisor.decision.decision_schema_manifest import production_decision_schemas
from chronovisor.decision.recall_improvement_contract import PolicyProposal
from chronovisor.decision.recall_policy_contract import RecallPolicy
from chronovisor.ingest.convergence import CycleBudget
from chronovisor.recall import recall_improvement


def _improvement_authority(epoch: str) -> dict:
    return {
        "source": "adopted_local_consensus",
        "authority_version": 1,
        "lane": "recall_improvement",
        "lane_contract_sha256": "1" * 64,
        "lane_contract_manifest_sha256": "2" * 64,
        "lane_contract_case_manifest_sha256": "3" * 64,
        "policy": {
            "kind": "local_batch",
            "schema_name": "generic_decision",
            "mode": "enabled",
            "error": None,
        },
        "router": {
            "source": "adopted_artifact",
            "artifact_sha256": epoch * 64,
            "error": None,
            "models": ["primary", "challenger", "tie"],
        },
    }


def _local_consensus_proof(review: dict, authority: dict) -> dict:
    schema_name = authority["policy"]["schema_name"]
    signature = canonical_agreement_signature(
        review,
        schema=production_decision_schemas()[schema_name],
    )
    agreement = hashlib.sha256(signature.encode("utf-8")).hexdigest()
    models = authority["router"]["models"]
    return {
        "status": "agreed",
        "ok": True,
        "agreement_sha256": agreement,
        "failure_class": None,
        "quarantine_reason": None,
        "votes": [
            {
                "role": "primary",
                "model": models[0],
                "valid": True,
                "signature_sha256": agreement,
                "invalid_reason": None,
            },
            {
                "role": "challenger",
                "model": models[1],
                "valid": True,
                "signature_sha256": agreement,
                "invalid_reason": None,
            },
        ],
    }


def _improvement_review(authority: dict, *, decision: str = "approved") -> dict:
    review = {
        "decision": decision,
        "summary": "authority-bound recall policy verdict",
        "tests_run": [],
        "commit": None,
        "committed": False,
        "pushed": False,
        "risk": None,
        "notes": None,
        "decision_policy": {
            **authority["policy"],
            "router_policy": authority["router"],
        },
    }
    review["local_consensus"] = _local_consensus_proof(review, authority)
    return review


def _durable_schedule_result(
    authority: dict,
    *,
    decision: str,
    run_id: str = "held",
) -> dict:
    status = (
        "frontier_quarantined"
        if decision == "quarantined"
        else "pending_frontier_review"
    )
    candidate_sha256 = hashlib.sha256(decision.encode("utf-8")).hexdigest()
    review = _improvement_review(authority, decision=decision)
    return {
        "ts": "2026-07-11T12:00:00",
        "run_id": run_id,
        "status": status,
        "applied": False,
        "dataset": {"examples": 1},
        "best": {"proposal": {"overrides": {"fusion_semantic": 0.7}}},
        "frontier_audit": {
            **review,
            "_artifact_durable": True,
            "candidate_sha256": candidate_sha256,
            "_artifact_path": f"/tmp/{candidate_sha256}.json",
            "_artifact_authority": authority,
        },
    }


def test_default_improvement_models_use_ornith_and_gemma_pair(monkeypatch) -> None:
    monkeypatch.delenv("CHRONOVISOR_RECALL_IMPROVEMENT_MODELS", raising=False)
    monkeypatch.setattr(recall_improvement, "load_toml_file", lambda: {})

    assert recall_improvement.configured_models() == (
        "maxwell1500/ornith-35b:Q5_K_M",
        "gemma4:26b",
    )


def test_ollama_proposer_accepts_direct_override_response(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeSession:
        def __init__(self, **kwargs: object) -> None:
            captured["session"] = kwargs

        def run(
            self,
            prompt: str,
            schema: dict[str, object],
            **kwargs: object,
        ) -> SimpleNamespace:
            captured["prompt"] = prompt
            captured["schema"] = schema
            captured["run"] = kwargs
            return SimpleNamespace(
                ok=True,
                value={"search_threshold": 0.25, "read_threshold": 0.7},
                failure_class=None,
                failure_reason=None,
            )

    monkeypatch.setattr(recall_improvement, "LocalStructuredSession", FakeSession)

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
    session = captured["session"]
    assert isinstance(session, dict)
    assert session["role"] == "recall_policy_proposer"
    assert session["num_ctx"] == 32768
    assert "transport" not in session
    schema = captured["schema"]
    assert isinstance(schema, dict)
    assert "search_threshold" in schema["properties"]
    run = captured["run"]
    assert isinstance(run, dict)
    assert run["value_validator"] is recall_improvement._proposal_value_issues


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
    assert (
        "relative_gain >= 0.050"
        in gate["candidate_must_pass_public_checks"]["dev_improved"]
    )
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
            {
                "checks": {
                    "dev_improved": False,
                    "latency_ok": False,
                    "holdout_score_ok": True,
                }
            },
            {
                "checks": {
                    "dev_improved": True,
                    "latency_ok": False,
                    "holdout_recall_ok": False,
                }
            },
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
                "prompt_preview": "Chronovisor recall policy",
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
                "prompt": "Chronovisor recall policy",
                "expected_pages": ["target-page"],
                "ref": "d1",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def test_run_improvement_persists_runtime_budget_deferred(
    tmp_path,
    monkeypatch,
) -> None:
    log_file = tmp_path / "recall-log.jsonl"
    feedback_file = tmp_path / "feedback.jsonl"
    _write_feedback(log_file, feedback_file)

    def exhaust_budget(*_args, **_kwargs):
        raise TimeoutError("recall evaluation runtime budget exhausted")

    monkeypatch.setattr(recall_improvement, "_evaluate_cached", exhaust_budget)
    runs_dir = tmp_path / "runs"
    registry_file = tmp_path / "policy-registry.jsonl"

    payload = recall_improvement.run_improvement(
        log_file=log_file,
        feedback_file=feedback_file,
        models=(),
        include_heuristic=False,
        registry_file=registry_file,
        runs_dir=runs_dir,
        episodes_file=tmp_path / "episodes.jsonl",
        live_episodes_file=tmp_path / "live-episodes.jsonl",
        max_elapsed_seconds=0.001,
    )

    assert payload["status"] == "budget_deferred"
    assert payload["applied"] is False
    assert payload["reason"] == "recall evaluation runtime budget exhausted"
    assert payload["dataset"] == {"examples": 1, "dev": 1, "holdout": 1}
    assert len(list(runs_dir.glob("*.json"))) == 1
    assert "budget_deferred" in registry_file.read_text(encoding="utf-8")


def test_policy_verdict_artifact_is_bound_to_current_adopted_authority(
    tmp_path,
    monkeypatch,
) -> None:
    record = {
        "run_id": "authority-run",
        "dataset": {"examples": 10, "dev": 8, "holdout": 2},
        "baseline": {"dev": {"score": 0.2}, "holdout": {"score": 0.2}},
        "failure_samples": [],
    }
    best = {
        "proposal": {"overrides": {"max_pages": 4}, "summary": "widen"},
        "checks": {"dev_improved": True},
        "dev": {"score": 0.5},
        "holdout": {"score": 0.5},
    }
    authority_a = _improvement_authority("a")
    authority_b = _improvement_authority("b")
    active_authority = {"value": authority_a}
    monkeypatch.setattr(
        recall_improvement.decision_authority,
        "current_semantic_authority",
        lambda *_args, **_kwargs: (active_authority["value"], None),
    )
    calls: list[str] = []

    def reviewer_a(_prompt, _best) -> dict:
        calls.append("a")
        return _improvement_review(authority_a)

    first = recall_improvement.run_frontier_policy_audit(
        record,
        best,
        reasons=["mandatory"],
        audit_dir=tmp_path / "audits",
        reviewer=reviewer_a,
        authority=authority_a,
    )
    assert first["_artifact_reused"] is False
    reused = recall_improvement.load_frontier_policy_audit(
        record,
        best,
        reasons=["mandatory"],
        audit_dir=tmp_path / "audits",
        authority=authority_a,
    )
    assert reused is not None
    assert reused["_artifact_reused"] is True

    active_authority["value"] = authority_b

    def reviewer_b(_prompt, _best) -> dict:
        calls.append("b")
        return _improvement_review(authority_b)

    second = recall_improvement.run_frontier_policy_audit(
        record,
        best,
        reasons=["mandatory"],
        audit_dir=tmp_path / "audits",
        reviewer=reviewer_b,
        authority=authority_b,
    )

    assert calls == ["a", "b"]
    assert second["_artifact_reused"] is False
    assert second["_artifact_authority"] == authority_b
    artifact = json.loads(Path(second["_artifact_path"]).read_text(encoding="utf-8"))
    assert artifact["schema_version"] == 3
    assert artifact["authority"] == authority_b


@pytest.mark.parametrize(
    "decision",
    ["approved", "rejected", "quarantined", "needs_retry"],
)
def test_policy_verdict_artifact_reuses_every_authority_bound_decision(
    tmp_path,
    monkeypatch,
    decision,
) -> None:
    record = {
        "run_id": f"reuse-{decision}",
        "dataset": {"examples": 10, "dev": 8, "holdout": 2},
        "baseline": {"dev": {"score": 0.2}, "holdout": {"score": 0.2}},
        "failure_samples": [],
    }
    best = {
        "proposal": {"overrides": {"max_pages": 4}, "summary": "widen"},
        "checks": {"dev_improved": True},
        "dev": {"score": 0.5},
        "holdout": {"score": 0.5},
    }
    authority = _improvement_authority("a")
    monkeypatch.setattr(
        recall_improvement.decision_authority,
        "current_semantic_authority",
        lambda *_args, **_kwargs: (authority, None),
    )
    calls = 0

    def reviewer(_prompt, _best) -> dict:
        nonlocal calls
        calls += 1
        return _improvement_review(authority, decision=decision)

    first = recall_improvement.run_frontier_policy_audit(
        record,
        best,
        reasons=["mandatory"],
        audit_dir=tmp_path / "audits",
        reviewer=reviewer,
        authority=authority,
    )
    reused = recall_improvement.run_frontier_policy_audit(
        record,
        best,
        reasons=["mandatory"],
        audit_dir=tmp_path / "audits",
        reviewer=reviewer,
        authority=authority,
    )

    assert calls == 1
    assert first["decision"] == decision
    assert first["_artifact_reused"] is False
    assert reused["decision"] == decision
    assert reused["_artifact_reused"] is True


def test_policy_verdict_artifact_rejects_old_schema(tmp_path, monkeypatch) -> None:
    record = {
        "run_id": "old-schema",
        "dataset": {"examples": 10, "dev": 8, "holdout": 2},
        "baseline": {"dev": {"score": 0.2}, "holdout": {"score": 0.2}},
        "failure_samples": [],
    }
    best = {
        "proposal": {"overrides": {"max_pages": 4}, "summary": "widen"},
        "checks": {"dev_improved": True},
        "dev": {"score": 0.5},
        "holdout": {"score": 0.5},
    }
    authority = _improvement_authority("a")
    monkeypatch.setattr(
        recall_improvement.decision_authority,
        "current_semantic_authority",
        lambda *_args, **_kwargs: (authority, None),
    )

    first = recall_improvement.run_frontier_policy_audit(
        record,
        best,
        reasons=["mandatory"],
        audit_dir=tmp_path / "audits",
        reviewer=lambda *_args: _improvement_review(authority),
        authority=authority,
    )
    artifact_path = Path(first["_artifact_path"])
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    artifact["schema_version"] = 2
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")

    assert (
        recall_improvement.load_frontier_policy_audit(
            record,
            best,
            reasons=["mandatory"],
            audit_dir=tmp_path / "audits",
            authority=authority,
        )
        is None
    )


def test_active_policy_writer_rechecks_authority_inside_shared_lock(
    tmp_path,
    monkeypatch,
) -> None:
    authority_a = _improvement_authority("a")
    authority_b = _improvement_authority("b")
    active_file = tmp_path / "active-policy.json"
    monkeypatch.setattr(recall_improvement, "decision_authority_lock", nullcontext)
    monkeypatch.setattr(
        recall_improvement.decision_authority,
        "current_semantic_authority",
        lambda *_args, **_kwargs: (authority_b, None),
    )

    error = recall_improvement._write_active_policy_under_authority(
        active_file,
        {"overrides": {"max_pages": 4}},
        review=_improvement_review(authority_a),
        authority=authority_a,
        injected_reviewer=False,
    )

    assert error == "decision authority changed before effect"
    assert not active_file.exists()


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

    def fake_evaluate(examples, *, policy, replay, deadline=None):
        del deadline
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
                    "prompt": "Chronovisor recall policy",
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
        recall_improvement, "safe_append_event", lambda *_args, **_kwargs: None
    )

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


def test_run_improvement_frontier_rejection_blocks_active_policy(
    tmp_path, monkeypatch
) -> None:
    log_file = tmp_path / "recall-log.jsonl"
    feedback_file = tmp_path / "feedback.jsonl"
    _write_feedback(log_file, feedback_file)

    monkeypatch.setattr(
        recall_improvement,
        "load_policy",
        lambda _config=None: RecallPolicy(log_decisions=False, max_pages=3),
    )
    monkeypatch.setattr(
        recall_improvement, "safe_append_event", lambda *_args, **_kwargs: None
    )

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

    def fake_evaluate(examples, *, policy, replay, deadline=None):
        del deadline
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
                    "prompt": "Chronovisor recall policy",
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
            "decision": "quarantined",
            "summary": "local quorum found a typed policy conflict",
            "human_required": False,
        },
    )
    quarantined_payload = recall_improvement.run_improvement(
        log_file=log_file,
        feedback_file=feedback_file,
        models=("qwen",),
        include_heuristic=False,
        frontier_mode="always",
        active_file=tmp_path / "quarantined-active-policy.json",
        registry_file=tmp_path / "quarantined-policy-registry.jsonl",
        runs_dir=tmp_path / "quarantined-runs",
        episodes_file=tmp_path / "quarantined-episodes.jsonl",
        live_episodes_file=tmp_path / "quarantined-live-episodes.jsonl",
    )
    assert quarantined_payload["status"] == "frontier_quarantined"
    assert quarantined_payload["applied"] is False
    assert quarantined_payload["active_policy"] is None

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


def test_run_improvement_frontier_human_required_failure_maps_to_review_retry(
    tmp_path, monkeypatch
) -> None:
    log_file = tmp_path / "recall-log.jsonl"
    feedback_file = tmp_path / "feedback.jsonl"
    _write_feedback(log_file, feedback_file)

    monkeypatch.setattr(
        recall_improvement,
        "load_policy",
        lambda _config=None: RecallPolicy(log_decisions=False, max_pages=3),
    )
    monkeypatch.setattr(
        recall_improvement, "safe_append_event", lambda *_args, **_kwargs: None
    )

    def fake_propose(**_kwargs):
        return [
            PolicyProposal(
                source="test",
                model="qwen",
                proposal_id="p1",
                summary="Auth failure path",
                rationale="Expected page needs a wider top-k.",
                overrides={"max_pages": 4},
                risk="low",
                audit_recommended=True,
            )
        ]

    def fake_evaluate(examples, *, policy, replay, deadline=None):
        del deadline
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
                    "prompt": "Chronovisor recall policy",
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
            "decision": "needs_retry",
            "failure_class": "auth_required",
            "summary": "frontier authority boundary not available now",
            "human_required": True,
        },
    )

    payload = recall_improvement.run_improvement(
        log_file=log_file,
        feedback_file=feedback_file,
        models=("qwen",),
        include_heuristic=False,
        frontier_mode="always",
        active_file=tmp_path / "human-required-active-policy.json",
        registry_file=tmp_path / "human-required-registry.jsonl",
        runs_dir=tmp_path / "human-required-runs",
        episodes_file=tmp_path / "human-required-episodes.jsonl",
        live_episodes_file=tmp_path / "human-required-live-episodes.jsonl",
    )

    assert payload["status"] == "pending_frontier_review"
    assert payload["frontier_audit"]["decision"] == "needs_retry"
    assert payload["frontier_audit"]["failure_class"] == "auth_required"
    assert payload["frontier_audit"]["human_required"] is True
    assert payload["applied"] is False
    assert payload["active_policy"] is None


def test_run_due_dry_run_is_read_only(tmp_path, monkeypatch) -> None:
    feedback_file = tmp_path / "feedback.jsonl"
    feedback_file.write_text(
        json.dumps({"kind": "missed_candidate", "prompt": "x"}, ensure_ascii=False)
        + "\n",
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
        "\n".join(
            json.dumps({"kind": "missed_candidate", "prompt": str(i)}) for i in range(3)
        )
        + "\n",
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
    monkeypatch.setattr(
        recall_improvement, "safe_append_metric", lambda *_args, **_kwargs: None
    )

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


def test_run_due_respects_interval_even_with_new_feedback(
    tmp_path, monkeypatch
) -> None:
    feedback_file = tmp_path / "feedback.jsonl"
    feedback_file.write_text(
        "\n".join(
            json.dumps({"kind": "missed_candidate", "prompt": str(i)})
            for i in range(20)
        )
        + "\n",
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


def test_run_due_preserves_pending_frontier_state_during_backoff(
    tmp_path, monkeypatch
) -> None:
    feedback_file = tmp_path / "feedback.jsonl"
    feedback_file.write_text(
        json.dumps({"kind": "missed_candidate", "prompt": "x"}) + "\n",
        encoding="utf-8",
    )
    schedule_file = tmp_path / "schedule-state.json"
    future = (
        recall_improvement.datetime.now() + recall_improvement.timedelta(hours=1)
    ).isoformat(timespec="seconds")
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
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("backoff must suppress retry")
        ),
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


@pytest.mark.parametrize(
    ("decision", "status"),
    [
        ("quarantined", "frontier_quarantined"),
        ("needs_retry", "pending_frontier_review"),
    ],
)
def test_run_due_holds_durable_nonterminal_verdict_without_new_evidence(
    tmp_path,
    monkeypatch,
    decision,
    status,
) -> None:
    feedback_file = tmp_path / "feedback.jsonl"
    feedback_file.write_text(
        json.dumps({"kind": "missed_candidate", "prompt": "x"}) + "\n",
        encoding="utf-8",
    )
    schedule_file = tmp_path / "schedule-state.json"
    authority = _improvement_authority("a")
    monkeypatch.setattr(
        recall_improvement.decision_authority,
        "current_semantic_authority",
        lambda *_args, **_kwargs: (authority, None),
    )
    calls = 0

    def held(**_kwargs):
        nonlocal calls
        calls += 1
        return _durable_schedule_result(
            authority,
            decision=decision,
            run_id=f"held-{calls}",
        )

    monkeypatch.setattr(recall_improvement, "run_improvement", held)
    monkeypatch.setattr(
        recall_improvement, "safe_append_metric", lambda *_args, **_kwargs: None
    )

    first = recall_improvement.run_due(
        feedback_file=feedback_file,
        log_file=tmp_path / "log.jsonl",
        min_total_feedback=1,
        min_new_feedback=1,
        min_interval_hours=0,
        schedule_file=schedule_file,
        lock_file=tmp_path / "run-due.lock",
    )
    second = recall_improvement.run_due(
        feedback_file=feedback_file,
        log_file=tmp_path / "log.jsonl",
        min_total_feedback=1,
        min_new_feedback=1,
        min_interval_hours=0,
        schedule_file=schedule_file,
        lock_file=tmp_path / "run-due.lock",
    )

    state = json.loads(schedule_file.read_text(encoding="utf-8"))
    assert first["status"] == "ran"
    assert second["status"] == "skipped"
    assert calls == 1
    assert state["last_status"] == status
    assert state["frontier_hold_decision"] == decision
    assert state["frontier_hold_feedback_count"] == 1
    assert len(state["frontier_hold_candidate_sha256"]) == 64
    assert len(state["frontier_hold_authority_sha256"]) == 64
    assert state["frontier_retry_attempts"] == 0
    assert state["frontier_quarantine_retry_at"] is None
    assert second["reasons"]["frontier_durable_hold_pending"] is True
    assert second["reasons"]["frontier_hold_feedback_release"] is False
    assert second["reasons"]["frontier_hold_authority_release"] is False


def test_run_due_releases_durable_hold_for_new_feedback(tmp_path, monkeypatch) -> None:
    feedback_file = tmp_path / "feedback.jsonl"
    feedback_file.write_text(
        json.dumps({"kind": "missed_candidate", "prompt": "first"}) + "\n",
        encoding="utf-8",
    )
    schedule_file = tmp_path / "schedule-state.json"
    authority = _improvement_authority("a")
    monkeypatch.setattr(
        recall_improvement.decision_authority,
        "current_semantic_authority",
        lambda *_args, **_kwargs: (authority, None),
    )
    calls = 0

    def improve(**_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return _durable_schedule_result(authority, decision="needs_retry")
        return {
            "ts": "2026-07-11T13:00:00",
            "run_id": "released-by-feedback",
            "status": "applied",
            "applied": True,
            "dataset": {"examples": 2},
        }

    monkeypatch.setattr(recall_improvement, "run_improvement", improve)
    monkeypatch.setattr(
        recall_improvement, "safe_append_metric", lambda *_args, **_kwargs: None
    )
    recall_improvement.run_due(
        feedback_file=feedback_file,
        log_file=tmp_path / "log.jsonl",
        min_total_feedback=1,
        min_new_feedback=99,
        schedule_file=schedule_file,
        lock_file=tmp_path / "run-due.lock",
    )
    with feedback_file.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps({"kind": "missed_candidate", "prompt": "second"}) + "\n"
        )

    released = recall_improvement.run_due(
        feedback_file=feedback_file,
        log_file=tmp_path / "log.jsonl",
        min_total_feedback=1,
        min_new_feedback=99,
        schedule_file=schedule_file,
        lock_file=tmp_path / "run-due.lock",
    )

    state = json.loads(schedule_file.read_text(encoding="utf-8"))
    assert released["status"] == "ran"
    assert released["decision"]["reasons"]["frontier_hold_feedback_release"] is True
    assert calls == 2
    assert state["last_status"] == "applied"
    assert state["frontier_hold_candidate_sha256"] is None
    assert state["frontier_hold_authority_sha256"] is None


def test_run_due_releases_durable_hold_for_authority_change(
    tmp_path, monkeypatch
) -> None:
    feedback_file = tmp_path / "feedback.jsonl"
    feedback_file.write_text(
        json.dumps({"kind": "missed_candidate", "prompt": "same"}) + "\n",
        encoding="utf-8",
    )
    schedule_file = tmp_path / "schedule-state.json"
    authority_a = _improvement_authority("a")
    authority_b = _improvement_authority("b")
    current_authority = {"value": authority_a}
    monkeypatch.setattr(
        recall_improvement.decision_authority,
        "current_semantic_authority",
        lambda *_args, **_kwargs: (current_authority["value"], None),
    )
    calls = 0

    def improve(**_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return _durable_schedule_result(authority_a, decision="quarantined")
        return {
            "ts": "2026-07-11T13:00:00",
            "run_id": "released-by-authority",
            "status": "frontier_rejected",
            "applied": False,
            "dataset": {"examples": 1},
        }

    monkeypatch.setattr(recall_improvement, "run_improvement", improve)
    monkeypatch.setattr(
        recall_improvement, "safe_append_metric", lambda *_args, **_kwargs: None
    )
    recall_improvement.run_due(
        feedback_file=feedback_file,
        log_file=tmp_path / "log.jsonl",
        min_total_feedback=1,
        min_new_feedback=1,
        schedule_file=schedule_file,
        lock_file=tmp_path / "run-due.lock",
    )
    current_authority["value"] = authority_b

    released = recall_improvement.run_due(
        feedback_file=feedback_file,
        log_file=tmp_path / "log.jsonl",
        min_total_feedback=1,
        min_new_feedback=99,
        schedule_file=schedule_file,
        lock_file=tmp_path / "run-due.lock",
    )

    state = json.loads(schedule_file.read_text(encoding="utf-8"))
    assert released["status"] == "ran"
    assert released["decision"]["reasons"]["frontier_hold_authority_release"] is True
    assert calls == 2
    assert state["last_status"] == "frontier_rejected"
    assert state["frontier_hold_candidate_sha256"] is None
    assert state["frontier_hold_authority_sha256"] is None


def test_run_due_keeps_hold_when_current_authority_is_unavailable(
    tmp_path, monkeypatch
) -> None:
    feedback_file = tmp_path / "feedback.jsonl"
    feedback_file.write_text(
        json.dumps({"kind": "missed_candidate", "prompt": "first"}) + "\n",
        encoding="utf-8",
    )
    schedule_file = tmp_path / "schedule-state.json"
    authority = _improvement_authority("a")
    current_authority = {"value": authority, "error": None}
    monkeypatch.setattr(
        recall_improvement.decision_authority,
        "current_semantic_authority",
        lambda *_args, **_kwargs: (
            current_authority["value"],
            current_authority["error"],
        ),
    )
    calls = 0

    def improve(**_kwargs):
        nonlocal calls
        calls += 1
        return _durable_schedule_result(authority, decision="needs_retry")

    monkeypatch.setattr(recall_improvement, "run_improvement", improve)
    monkeypatch.setattr(
        recall_improvement, "safe_append_metric", lambda *_args, **_kwargs: None
    )
    recall_improvement.run_due(
        feedback_file=feedback_file,
        log_file=tmp_path / "log.jsonl",
        min_total_feedback=1,
        min_new_feedback=1,
        schedule_file=schedule_file,
        lock_file=tmp_path / "run-due.lock",
    )
    with feedback_file.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps({"kind": "missed_candidate", "prompt": "second"}) + "\n"
        )
    current_authority["value"] = None
    current_authority["error"] = "adopted authority unavailable"

    held = recall_improvement.run_due(
        feedback_file=feedback_file,
        log_file=tmp_path / "log.jsonl",
        min_total_feedback=1,
        min_new_feedback=1,
        schedule_file=schedule_file,
        lock_file=tmp_path / "run-due.lock",
    )

    assert held["status"] == "skipped"
    assert held["reasons"]["frontier_hold_feedback_release"] is True
    assert held["reasons"]["frontier_hold_authority_available"] is False
    assert calls == 1


def test_run_due_retries_legacy_quarantine_once_through_backoff(
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
                "frontier_retry_candidate": "legacy-candidate",
                "frontier_retry_attempts": 3,
                "frontier_quarantine_retry_at": "2000-01-01T00:00:00",
            }
        ),
        encoding="utf-8",
    )
    calls = 0

    def unavailable(**_kwargs):
        nonlocal calls
        calls += 1
        return {
            "ts": "2026-07-11T12:00:00",
            "run_id": "legacy-retry",
            "status": "pending_frontier_review",
            "applied": False,
            "dataset": {"examples": 1},
            "frontier_audit": {"summary": "temporary model outage"},
        }

    monkeypatch.setattr(recall_improvement, "run_improvement", unavailable)
    monkeypatch.setattr(
        recall_improvement, "safe_append_metric", lambda *_args, **_kwargs: None
    )
    first = recall_improvement.run_due(
        feedback_file=feedback_file,
        log_file=tmp_path / "log.jsonl",
        min_total_feedback=1,
        min_new_feedback=99,
        schedule_file=schedule_file,
        lock_file=tmp_path / "run-due.lock",
    )
    expired_backoff = json.loads(schedule_file.read_text(encoding="utf-8"))
    expired_backoff["frontier_next_retry_at"] = "2000-01-01T00:00:00"
    schedule_file.write_text(json.dumps(expired_backoff), encoding="utf-8")
    second = recall_improvement.run_due(
        feedback_file=feedback_file,
        log_file=tmp_path / "log.jsonl",
        min_total_feedback=1,
        min_new_feedback=99,
        schedule_file=schedule_file,
        lock_file=tmp_path / "run-due.lock",
    )

    state = json.loads(schedule_file.read_text(encoding="utf-8"))
    assert first["status"] == "ran"
    assert second["status"] == "skipped"
    assert calls == 1
    assert state["frontier_legacy_hold_retried"] is True
    assert second["reasons"]["frontier_legacy_retry_hold_pending"] is True
    assert second["reasons"]["frontier_legacy_feedback_release"] is False


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
        json.dumps(
            candidate_payload, ensure_ascii=False, sort_keys=True, default=str
        ).encode("utf-8")
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
    monkeypatch.setattr(
        recall_improvement, "safe_append_metric", lambda *_args, **_kwargs: None
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
    assert result["result"]["status"] == "frontier_quarantined"
    assert state["last_feedback_count"] == 1
    assert state["frontier_quarantine_retry_at"]


def test_run_due_budget_defer_keeps_schedule_byte_for_byte(
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
    monkeypatch.setattr(
        recall_improvement, "safe_append_metric", lambda *_args, **_kwargs: None
    )

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
        json.dumps({"kind": "missed_candidate", "prompt": "x"}, ensure_ascii=False)
        + "\n",
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
                json.dumps(
                    {
                        "ts": "2026-07-05T12:00:00",
                        "host": "codex",
                        "decision": "read",
                        "pages": ["a"],
                        "latency_ms": 10,
                    }
                ),
                json.dumps(
                    {
                        "ts": "2026-07-05T12:01:00",
                        "host": "codex",
                        "decision": "none",
                        "pages": [],
                        "latency_ms": 20,
                    }
                ),
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
