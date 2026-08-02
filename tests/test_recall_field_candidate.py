from __future__ import annotations

import json

from chronovisor.recall import recall_field_candidate
from chronovisor.recall.recall_field_schema import RecallFieldConfig
from chronovisor.search.search_types import ScoredPage


def page(page_id: str, score: float = 1.0) -> ScoredPage:
    return ScoredPage(page_id, page_id, "", "2026-07-30", score)


def test_candidate_and_teacher_run_but_teacher_keeps_authority(monkeypatch) -> None:
    cfg = RecallFieldConfig(mode="candidate", canary_percent=100)
    monkeypatch.setattr(
        recall_field_candidate.semantic_client,
        "verify",
        lambda *_args, **_kwargs: [page("a"), page("b")],
    )
    teacher_calls = 0

    def teacher():
        nonlocal teacher_calls
        teacher_calls += 1
        return [page("a"), page("c")], "hybrid"

    results, mode, metadata = recall_field_candidate.run_candidate_teacher_pair(
        query="stateful recall",
        field_turn={
            "session_hash": "0123456789abcdef",
            "candidate_page_ids": ["a", "b"],
            "full_search_fallback": False,
        },
        teacher_search=teacher,
        timeout_ms=500,
        config=cfg,
    )

    assert teacher_calls == 1
    assert [row.page_id for row in results] == ["a", "c"]
    assert mode == "hybrid"
    assert metadata["authority"] == "teacher"
    assert metadata["missed_page_ids"] == ["c"]
    assert metadata["quality_eligible"] is True
    assert metadata["field_attempted"] is True
    assert metadata["field_verified"] is True


def test_topic_reset_skips_field_verify_but_never_full_search(monkeypatch) -> None:
    cfg = RecallFieldConfig(mode="candidate", canary_percent=100)
    monkeypatch.setattr(
        recall_field_candidate.semantic_client,
        "verify",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("verify must not run")
        ),
    )

    results, _mode, metadata = recall_field_candidate.run_candidate_teacher_pair(
        query="new topic",
        field_turn={
            "session_hash": "0123456789abcdef",
            "candidate_page_ids": ["old"],
            "full_search_fallback": True,
        },
        teacher_search=lambda: ([page("fresh")], "hybrid"),
        timeout_ms=500,
        config=cfg,
    )

    assert [row.page_id for row in results] == ["fresh"]
    assert metadata["reason"] == "topic_reset"
    assert metadata["quality_eligible"] is False
    assert metadata["field_attempted"] is False
    assert metadata["field_verified"] is False


def test_active_mode_fails_closed_when_field_verification_times_out(
    tmp_path,
    monkeypatch,
) -> None:
    cfg = RecallFieldConfig(mode="active", canary_percent=100)
    artifact = tmp_path / "promotion.json"
    payload = {
        "schema_version": 1,
        "status": "passed",
        "metrics": {
            "teacher_commit_coverage": 0.99,
            "precision_delta_points": -0.5,
            "recall_delta_points": -0.5,
            "over_4s": 0,
            "processor_used_precision_proxy": 0.95,
        },
    }
    artifact.write_text(
        json.dumps(
            {
                **payload,
                "snapshot_sha256": recall_field_candidate._canonical_sha256(payload),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(recall_field_candidate, "PROMOTION_ARTIFACT", artifact)
    monkeypatch.setattr(
        recall_field_candidate.semantic_client,
        "verify",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(TimeoutError),
    )

    results, mode, metadata = recall_field_candidate.run_candidate_teacher_pair(
        query="query",
        field_turn={
            "session_hash": "0123456789abcdef",
            "candidate_page_ids": ["field"],
            "full_search_fallback": False,
        },
        teacher_search=lambda: ([page("teacher")], "hybrid"),
        timeout_ms=500,
        config=cfg,
    )

    assert [row.page_id for row in results] == ["teacher"]
    assert mode == "hybrid"
    assert metadata["status"] == "fallback"
    assert metadata["reason"] == "TimeoutError"
    assert metadata["authority"] == "teacher"
    assert metadata["full_search_required"] is True
    assert metadata["quality_eligible"] is True
    assert metadata["field_attempted"] is True
    assert metadata["field_verified"] is False

    trace = recall_field_candidate.append_candidate_trace(
        session_hash="0123456789abcdef",
        prompt="private prompt",
        observer=metadata,
        committed_page_ids=["teacher"],
        latency_ms=120,
        path=tmp_path / "trace.jsonl",
    )
    assert trace["quality_eligible"] is True
    assert trace["field_attempted"] is True
    assert trace["field_verified"] is False


def test_successful_empty_verification_falls_back_to_teacher(monkeypatch) -> None:
    cfg = RecallFieldConfig(mode="active", canary_percent=100)
    monkeypatch.setattr(
        recall_field_candidate,
        "authority_allowed",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        recall_field_candidate.semantic_client,
        "verify",
        lambda *_args, **_kwargs: [],
    )

    results, mode, metadata = recall_field_candidate.run_candidate_teacher_pair(
        query="query",
        field_turn={
            "session_hash": "0123456789abcdef",
            "candidate_page_ids": ["field"],
            "full_search_fallback": False,
        },
        teacher_search=lambda: ([page("teacher")], "hybrid"),
        timeout_ms=500,
        config=cfg,
    )

    assert [row.page_id for row in results] == ["teacher"]
    assert mode == "hybrid"
    assert metadata["status"] == "fallback"
    assert metadata["reason"] == "empty_verified_field"
    assert metadata["authority"] == "teacher"
    assert metadata["full_search_required"] is True
    assert metadata["field_verified"] is True
    assert metadata["quality_eligible"] is True


def test_active_mode_rolls_back_without_passing_artifact(
    tmp_path,
    monkeypatch,
) -> None:
    cfg = RecallFieldConfig(mode="active", canary_percent=100)
    monkeypatch.setattr(
        recall_field_candidate,
        "PROMOTION_ARTIFACT",
        tmp_path / "missing.json",
    )
    monkeypatch.setattr(
        recall_field_candidate.semantic_client,
        "verify",
        lambda *_args, **_kwargs: [page("field")],
    )

    results, _mode, metadata = recall_field_candidate.run_candidate_teacher_pair(
        query="query",
        field_turn={
            "session_hash": "0123456789abcdef",
            "candidate_page_ids": ["field"],
            "full_search_fallback": False,
        },
        teacher_search=lambda: ([page("teacher")], "hybrid"),
        timeout_ms=500,
        config=cfg,
    )

    assert [row.page_id for row in results] == ["teacher"]
    assert metadata["rollback"] is True
    assert metadata["effective_mode"] == "shadow"


def test_active_mode_cannot_bypass_certificate_boundary(monkeypatch) -> None:
    cfg = RecallFieldConfig(mode="active", canary_percent=100)
    monkeypatch.setattr(
        recall_field_candidate,
        "authority_allowed",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        recall_field_candidate.semantic_client,
        "verify",
        lambda *_args, **_kwargs: [page("field")],
    )

    results, _mode, metadata = recall_field_candidate.run_candidate_teacher_pair(
        query="query",
        field_turn={
            "session_hash": "0123456789abcdef",
            "candidate_page_ids": ["field"],
            "full_search_fallback": False,
        },
        teacher_search=lambda: ([page("teacher")], "hybrid"),
        timeout_ms=500,
        config=cfg,
        certificate_boundary_enabled=False,
    )

    assert [row.page_id for row in results] == ["teacher"]
    assert metadata["rollback_reason"] == "certificate_boundary_disabled"


def test_canary_rollouts_are_deterministic_and_nested() -> None:
    sessions = [f"{index:016x}" for index in range(500)]
    selected_5 = {
        session
        for session in sessions
        if recall_field_candidate.selected_for_canary(
            session,
            RecallFieldConfig(mode="candidate", canary_percent=5),
        )
    }
    selected_25 = {
        session
        for session in sessions
        if recall_field_candidate.selected_for_canary(
            session,
            RecallFieldConfig(mode="candidate", canary_percent=25),
        )
    }
    selected_100 = {
        session
        for session in sessions
        if recall_field_candidate.selected_for_canary(
            session,
            RecallFieldConfig(mode="candidate", canary_percent=100),
        )
    }

    assert selected_5
    assert selected_5 < selected_25 < selected_100
    assert len(selected_100) == len(sessions)


def test_auto_rollout_uses_durable_growth_state(monkeypatch) -> None:
    from chronovisor.recall import recall_growth

    monkeypatch.setattr(
        recall_growth,
        "automatic_rollout",
        lambda **_kwargs: ("active", 25),
    )

    effective = recall_field_candidate.effective_rollout(
        RecallFieldConfig(
            mode="candidate",
            canary_percent=100,
            auto_promote=True,
        )
    )

    assert effective.mode == "active"
    assert effective.canary_percent == 25


def test_promotion_artifact_is_hash_bound_and_requires_nondegradation(
    tmp_path,
) -> None:
    path = tmp_path / "promotion.json"
    payload = {
        "schema_version": 3,
        "status": "passed",
        "metrics": {
            "stable_traces": 100,
            "stable_sessions": 20,
            "coverage_evidence_traces": 100,
            "coverage_evidence_sessions": 20,
            "commit_evidence_traces": 100,
            "commit_evidence_sessions": 20,
            "paired_latency_traces": 100,
            "paired_latency_sessions": 20,
            "teacher_commit_coverage": 0.99,
            "precision_delta_points": -0.5,
            "recall_delta_points": -0.5,
            "over_4s": 0,
                "processor_used_precision_proxy": 0.95,
            },
        "confidence_evidence": {
            "candidate": {
                "samples": 100,
                "clusters": 20,
                "manifest_sha256": "a" * 64,
                    "teacher_coverage": {"valid": True, "method": "connected-cluster-wilson-score", "point": 1.0, "lower": 0.99, "upper": 1.0},
                    "commit_coverage": {"valid": True, "method": "connected-cluster-wilson-score", "point": 1.0, "lower": 0.99, "upper": 1.0},
                    "field_precision": {"valid": True, "method": "connected-cluster-wilson-score", "point": 1.0, "lower": 0.95, "upper": 1.0},
            },
            "processor_used": {
                "samples": 50,
                "clusters": 20,
                "manifest_sha256": "b" * 64,
                    "coverage": {"valid": True, "method": "connected-cluster-wilson-score", "point": 1.0, "lower": 0.99, "upper": 1.0},
                    "precision": {"valid": True, "method": "connected-cluster-wilson-score", "point": 1.0, "lower": 0.95, "upper": 1.0},
            },
            "answer_reward": {
                    "valid": True,
                    "method": "connected-cluster-bootstrap-percentile",
                "samples": 20,
                "clusters": 20,
                "manifest_sha256": "c" * 64,
                "point": 0.05,
                    "lower": 0.01,
                    "upper": 0.1,
                "point_floor": 0.02,
                "lower_floor": 0.0,
            },
            },
            "answer_evaluation": {"passed": True, "split_manifest_sha256": "d" * 64},
            "locked_answer_evaluation": {"passed": True, "split_manifest_sha256": "d" * 64},
            "retrieval_locked_e2e": {"passed": True, "examples": 94, "manifest_sha256": "e" * 64},
            "answer_artifact_set": {"passed": True, "split_manifest_sha256": "d" * 64},
    }
    path.write_text(
        json.dumps(
            {
                **payload,
                "snapshot_sha256": recall_field_candidate._canonical_sha256(payload),
            }
        ),
        encoding="utf-8",
    )

    # Summary-only confidence is not authority evidence. The central gate now
    # requires case receipts, live source artifacts, and a current trace chain.
    assert recall_field_candidate.authority_allowed(path) is False
    legacy = {
        **payload,
        "schema_version": 1,
        "metrics": {
            key: value
            for key, value in payload["metrics"].items()
            if not key.startswith(("stable_", "coverage_", "commit_", "paired_"))
        },
    }
    path.write_text(
        json.dumps(
            {
                **legacy,
                "snapshot_sha256": recall_field_candidate._canonical_sha256(legacy),
            }
        ),
        encoding="utf-8",
    )
    assert recall_field_candidate.authority_allowed(path) is False

    degraded = {
        **payload,
        "confidence_evidence": {
            **payload["confidence_evidence"],
            "answer_reward": {
                **payload["confidence_evidence"]["answer_reward"],
                "lower": -0.01,
            },
        },
    }
    path.write_text(
        json.dumps(
            {
                **degraded,
                "snapshot_sha256": recall_field_candidate._canonical_sha256(
                    degraded
                ),
            }
        ),
        encoding="utf-8",
    )
    assert recall_field_candidate.authority_allowed(path) is False

    path.write_text(
        json.dumps(
            {
                **payload,
                "snapshot_sha256": recall_field_candidate._canonical_sha256(payload),
            }
        ),
        encoding="utf-8",
    )
    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["metrics"]["recall_delta_points"] = -2.0
    path.write_text(json.dumps(tampered), encoding="utf-8")
    assert recall_field_candidate.authority_allowed(path) is False

    malformed = {
        **payload,
        "metrics": {**payload["metrics"], "over_4s": {"invalid": True}},
    }
    path.write_text(
        json.dumps(
            {
                **malformed,
                "snapshot_sha256": recall_field_candidate._canonical_sha256(malformed),
            }
        ),
        encoding="utf-8",
    )
    assert recall_field_candidate.authority_allowed(path) is False


def test_candidate_trace_hashes_prompt_and_measures_commit_coverage(tmp_path) -> None:
    path = tmp_path / "trace.jsonl"
    record = recall_field_candidate.append_candidate_trace(
        session_hash="0123456789abcdef",
        prompt="private prompt",
        observer={
            "status": "observed",
            "authority": "teacher",
            "quality_eligible": True,
            "field_attempted": True,
            "field_verified": True,
            "field_page_ids": ["a", "b"],
            "teacher_page_ids": ["a", "c"],
            "missed_page_ids": ["c"],
        },
        committed_page_ids=["a", "c"],
        latency_ms=120,
        path=path,
    )
    serialized = path.read_text(encoding="utf-8")

    assert record["teacher_commit_coverage"] == 0.5
    assert record["quality_eligible"] is True
    assert record["field_attempted"] is True
    assert record["field_verified"] is True
    assert "private prompt" not in serialized
    assert json.loads(serialized)["prompt_sha256"] == record["prompt_sha256"]
