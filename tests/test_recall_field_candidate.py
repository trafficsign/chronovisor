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


def test_promotion_artifact_is_hash_bound_and_requires_nondegradation(
    tmp_path,
) -> None:
    path = tmp_path / "promotion.json"
    payload = {
        "schema_version": 1,
        "status": "passed",
        "metrics": {
            "teacher_commit_coverage": 0.99,
            "precision_delta_points": -0.5,
            "recall_delta_points": -0.5,
            "over_4s": 0,
        },
    }
    path.write_text(
        json.dumps(
            {
                **payload,
                "snapshot_sha256": recall_field_candidate._canonical_sha256(
                    payload
                ),
            }
        ),
        encoding="utf-8",
    )

    assert recall_field_candidate.authority_allowed(path) is True
    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["metrics"]["recall_delta_points"] = -2.0
    path.write_text(json.dumps(tampered), encoding="utf-8")
    assert recall_field_candidate.authority_allowed(path) is False


def test_candidate_trace_hashes_prompt_and_measures_commit_coverage(tmp_path) -> None:
    path = tmp_path / "trace.jsonl"
    record = recall_field_candidate.append_candidate_trace(
        session_hash="0123456789abcdef",
        prompt="private prompt",
        observer={
            "status": "observed",
            "authority": "teacher",
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
    assert "private prompt" not in serialized
    assert json.loads(serialized)["prompt_sha256"] == record["prompt_sha256"]
