from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

from chronovisor.recall import recall_growth


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _label_inputs(tmp_path: Path) -> dict[str, Path]:
    return {
        "certificate_file": tmp_path / "certificates.jsonl",
        "recall_log_file": tmp_path / "recall.jsonl",
        "pull_log_file": tmp_path / "pull.jsonl",
        "golden_file": tmp_path / "golden.jsonl",
    }


def _locked_gate(tmp_path: Path) -> Path:
    path = tmp_path / "locked-e2e.json"
    unsigned = {
        "status": "passed",
        "gates": {"quality": True},
        "manifest_sha256": "a" * 64,
        "precision_delta_points": 0.0,
        "recall_delta_points": 0.0,
        "precision_lower_95": 0.9,
    }
    payload = {
        **unsigned,
        "snapshot_sha256": hashlib.sha256(
            json.dumps(
                unsigned,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest(),
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_growth_cycle_collects_without_authorizing_weak_evidence(
    tmp_path: Path,
) -> None:
    inputs = _label_inputs(tmp_path)
    for path in inputs.values():
        _write_jsonl(path, [])
    state = tmp_path / "growth-state.json"
    promotion = tmp_path / "promotion.json"

    result = recall_growth.run_growth_cycle(
        state_file=state,
        history_file=tmp_path / "history.jsonl",
        candidate_trace_file=tmp_path / "candidate.jsonl",
        promotion_file=promotion,
        label_inputs=inputs,
        now=datetime(2026, 7, 31, 20, 0, 0),
    )

    assert result["stage"] == "collecting_labels"
    assert result["effective_mode"] == "candidate"
    assert result["canary_percent"] == 100
    assert result["field_learning_allowed"] is False
    assert result["authority_enabled"] is False
    assert json.loads(promotion.read_text())["status"] == "held"


def test_growth_cycle_reads_default_locked_e2e_artifact(
    tmp_path: Path, monkeypatch
) -> None:
    inputs = _label_inputs(tmp_path)
    for path in inputs.values():
        _write_jsonl(path, [])
    locked = _locked_gate(tmp_path)
    monkeypatch.setattr(recall_growth, "LOCKED_E2E_ARTIFACT", locked)

    result = recall_growth.run_growth_cycle(
        state_file=tmp_path / "growth-state.json",
        history_file=tmp_path / "history.jsonl",
        candidate_trace_file=tmp_path / "candidate.jsonl",
        promotion_file=tmp_path / "promotion.json",
        label_inputs=inputs,
    )

    assert result["locked_e2e"]["passed"] is True
    assert result["gates"]["locked_e2e"] is True


def test_positive_learning_unlocks_before_production_authority(tmp_path: Path) -> None:
    inputs = _label_inputs(tmp_path)
    recalls: list[dict] = []
    pulls: list[dict] = []
    for index in range(200):
        decision = f"decision-{index}"
        session = f"session-{index % 20}"
        page = f"page-{index}"
        recalls.append(
            {
                "decision_id": decision,
                "session_id": session,
                "prompt_hash": f"{index:064x}",
            }
        )
        pulls.append(
            {
                "type": "used",
                "event_id": f"used-{index}",
                "decision_id": decision,
                "session_id": session,
                "page_ids": [page],
            }
        )
    _write_jsonl(inputs["recall_log_file"], recalls)
    _write_jsonl(inputs["pull_log_file"], pulls)
    _write_jsonl(inputs["certificate_file"], [])
    _write_jsonl(inputs["golden_file"], [])
    state = tmp_path / "growth-state.json"
    last_known_good = tmp_path / "last-known-good.json"

    result = recall_growth.run_growth_cycle(
        state_file=state,
        history_file=tmp_path / "history.jsonl",
        candidate_trace_file=tmp_path / "candidate.jsonl",
        promotion_file=tmp_path / "promotion.json",
        last_known_good_file=last_known_good,
        locked_e2e_file=_locked_gate(tmp_path),
        label_inputs=inputs,
    )

    assert result["stage"] == "collecting_candidate_evidence"
    assert result["positive_learning_allowed"] is True
    assert result["field_learning_allowed"] is True
    assert result["policy_update_allowed"] is False
    assert result["authority_enabled"] is False
    assert recall_growth.automatic_learning_allowed(
        enabled=True,
        state_file=state,
    )
    assert not last_known_good.exists()


def test_processor_used_metrics_penalize_unused_shadow_cards() -> None:
    metrics = recall_growth.processor_used_metrics(
        [
            {
                "decision_id": "decision-1",
                "session_id": "session-1",
                "prompt_hash": "a" * 64,
                "evidence_features": {
                    "processor_shadow": {"committed_page_ids": ["used", "unused"]}
                },
            }
        ],
        [
            {
                "type": "used",
                "event_id": "used-1",
                "decision_id": "decision-1",
                "session_id": "session-1",
                "page_ids": ["used"],
            }
        ],
    )

    assert metrics["used_page_coverage"] == 1.0
    assert metrics["used_precision_proxy"] == 0.5


def test_candidate_metrics_separate_stable_quality_from_fallback_e2e() -> None:
    metrics = recall_growth.candidate_metrics(
        [
            {
                "session_hash": "stable-observed",
                "status": "observed",
                "authority": "teacher",
                "field_page_ids": ["page-a"],
                "teacher_page_ids": ["page-a"],
                "committed_page_ids": ["page-a"],
                "latency_ms": 100,
                "field_latency_ms": 20,
                "teacher_latency_ms": 100,
                "full_search_required": False,
                "over_4s": False,
            },
            {
                "session_hash": "stable-active",
                "status": "active",
                "authority": "field",
                "field_page_ids": ["page-b"],
                "teacher_page_ids": ["page-b"],
                "committed_page_ids": ["page-b"],
                "latency_ms": 110,
                "field_latency_ms": 30,
                "teacher_latency_ms": 110,
                "full_search_required": False,
                "over_4s": False,
            },
            {
                "session_hash": "legacy-observed",
                "status": "observed",
                "authority": "teacher",
                "field_page_ids": ["page-legacy"],
                "teacher_page_ids": ["page-legacy"],
                "committed_page_ids": ["page-legacy"],
                "latency_ms": 120,
                "field_latency_ms": 40,
                "teacher_latency_ms": 120,
                "over_4s": False,
            },
            {
                "session_hash": "topic-reset",
                "status": "fallback",
                "fallback_reason": "topic_reset",
                "authority": "teacher",
                "field_page_ids": [],
                "teacher_page_ids": ["fallback-page"],
                "committed_page_ids": ["fallback-page"],
                "latency_ms": 5_000,
                "teacher_latency_ms": 90,
                "full_search_required": True,
                "over_4s": True,
            },
            {
                "session_hash": "not-verified",
                "status": "fallback",
                "authority": "teacher",
                "quality_eligible": True,
                "field_attempted": True,
                "field_verified": False,
                "field_page_ids": [],
                "teacher_page_ids": ["other-page"],
                "committed_page_ids": ["other-page"],
                "latency_ms": 400,
                "field_latency_ms": 50,
                "teacher_latency_ms": 400,
                "full_search_required": True,
                "over_4s": False,
            },
            {
                "session_hash": "malformed-full-search-flag",
                "status": "observed",
                "authority": "teacher",
                "field_page_ids": ["malformed-page"],
                "teacher_page_ids": ["malformed-page"],
                "committed_page_ids": ["malformed-page"],
                "latency_ms": 100,
                "field_latency_ms": 10,
                "teacher_latency_ms": 100,
                "full_search_required": "false",
                "over_4s": False,
            },
        ]
    )

    # Stable-topic attempts are quality evidence; verifier failures stay misses.
    assert metrics["traces"] == 6
    assert metrics["sessions"] == 6
    assert metrics["stable_traces"] == 4
    assert metrics["stable_sessions"] == 4
    assert metrics["coverage_evidence_traces"] == 4
    assert metrics["coverage_evidence_sessions"] == 4
    assert metrics["commit_evidence_traces"] == 4
    assert metrics["commit_evidence_sessions"] == 4
    assert metrics["paired_latency_traces"] == 4
    assert metrics["paired_latency_sessions"] == 4
    assert metrics["teacher_top30_coverage"] == 0.75
    assert metrics["teacher_commit_coverage"] == 0.75
    assert metrics["field_precision_against_teacher"] == 1.0
    assert metrics["field_latency_ms"]["p95"] == 50.0
    assert metrics["teacher_latency_ms"]["p95"] == 400.0
    assert metrics["p95_improvement_ms"] == 350.0
    assert metrics["active_traces"] == 1

    # Fallbacks remain visible to whole-request safety and cost metrics.
    assert metrics["latency_ms"]["p95"] == 5_000.0
    assert metrics["over_4s"] == 1
    assert metrics["fallbacks"] == 2
    assert metrics["fallback_rate"] == 0.333333
    assert metrics["full_searches"] == 3
    assert metrics["full_search_rate"] == 0.5


def test_candidate_precision_counts_field_only_false_positives() -> None:
    metrics = recall_growth.candidate_metrics(
        [
            {
                "session_hash": "perfect",
                "status": "observed",
                "full_search_required": False,
                "field_page_ids": ["page-a"],
                "teacher_page_ids": ["page-a"],
            },
            {
                "session_hash": "field-only",
                "status": "observed",
                "full_search_required": False,
                "field_page_ids": ["false-positive"],
                "teacher_page_ids": [],
            },
        ]
    )

    assert metrics["field_pages"] == 2
    assert metrics["field_teacher_overlap"] == 1
    assert metrics["field_precision_against_teacher"] == 0.5
    assert metrics["teacher_top30_coverage"] == 1.0
    assert metrics["coverage_evidence_traces"] == 1


def test_all_fallback_candidate_evidence_fails_quality_gates(tmp_path: Path) -> None:
    inputs = _label_inputs(tmp_path)
    recalls: list[dict] = []
    pulls: list[dict] = []
    for index in range(200):
        page = f"page-{index}"
        recalls.append(
            {
                "decision_id": f"decision-{index}",
                "session_id": f"session-{index % 20}",
                "prompt_hash": f"{index:064x}",
                "evidence_features": {
                    "processor_shadow": {"committed_page_ids": [page]}
                },
            }
        )
        pulls.append(
            {
                "type": "used",
                "event_id": f"used-{index}",
                "decision_id": f"decision-{index}",
                "session_id": f"session-{index % 20}",
                "page_ids": [page],
            }
        )
    _write_jsonl(inputs["recall_log_file"], recalls)
    _write_jsonl(inputs["pull_log_file"], pulls)
    _write_jsonl(inputs["certificate_file"], [])
    _write_jsonl(inputs["golden_file"], [])
    fallback_rows = [
        {
            "session_hash": f"fallback-{index % 20}",
            "status": "fallback",
            "fallback_reason": "topic_reset",
            "authority": "teacher",
            "field_page_ids": [],
            "teacher_page_ids": [f"page-{index}"],
            "committed_page_ids": [f"page-{index}"],
            "latency_ms": 5_000 if index == 0 else 100,
            "teacher_latency_ms": 100,
            "full_search_required": True,
            "over_4s": index == 0,
        }
        for index in range(100)
    ]
    candidate_trace = tmp_path / "candidate.jsonl"
    _write_jsonl(candidate_trace, fallback_rows)

    result = recall_growth.run_growth_cycle(
        state_file=tmp_path / "growth-state.json",
        history_file=tmp_path / "history.jsonl",
        candidate_trace_file=candidate_trace,
        promotion_file=tmp_path / "promotion.json",
        locked_e2e_file=_locked_gate(tmp_path),
        label_inputs=inputs,
    )

    candidate = result["metrics"]["candidate"]
    assert candidate["traces"] == 100
    assert candidate["sessions"] == 20
    assert candidate["stable_traces"] == 0
    assert candidate["stable_sessions"] == 0
    assert candidate["quality_window_traces"] == 100
    assert candidate["quality_window_stable_traces"] == 0
    assert candidate["quality_window_stable_sessions"] == 0
    assert candidate["teacher_top30_coverage"] == 0.0
    assert candidate["teacher_commit_coverage"] == 0.0
    assert candidate["field_latency_ms"]["p95"] is None
    assert candidate["p95_improvement_ms"] is None
    assert candidate["full_search_rate"] == 1.0
    assert candidate["over_4s"] == 1
    assert result["gates"]["candidate_samples"] is False
    assert result["gates"]["candidate_sessions"] is False
    assert result["gates"]["candidate_coverage_evidence"] is False
    assert result["gates"]["candidate_commit_evidence"] is False
    assert result["gates"]["candidate_latency_evidence"] is False
    assert result["gates"]["teacher_top30_coverage"] is False
    assert result["gates"]["teacher_commit_coverage"] is False
    assert result["gates"]["p95_improvement"] is False
    assert result["authority_enabled"] is False


def test_sparse_quality_evidence_cannot_satisfy_candidate_gates(
    tmp_path: Path,
) -> None:
    inputs = _label_inputs(tmp_path)
    for path in inputs.values():
        _write_jsonl(path, [])
    empty_rows = [
        {
            "session_hash": f"empty-{index}",
            "status": "observed",
            "full_search_required": False,
            "latency_ms": 100,
        }
        for index in range(99)
    ]
    perfect = {
        "session_hash": "perfect",
        "status": "observed",
        "field_page_ids": ["page"],
        "teacher_page_ids": ["page"],
        "committed_page_ids": ["page"],
        "latency_ms": 100,
        "field_latency_ms": 10,
        "teacher_latency_ms": 100,
        "full_search_required": False,
    }
    candidate_trace = tmp_path / "candidate.jsonl"
    _write_jsonl(candidate_trace, [*empty_rows, perfect])

    result = recall_growth.run_growth_cycle(
        state_file=tmp_path / "growth-state.json",
        history_file=tmp_path / "history.jsonl",
        candidate_trace_file=candidate_trace,
        promotion_file=tmp_path / "promotion.json",
        locked_e2e_file=_locked_gate(tmp_path),
        label_inputs=inputs,
    )

    candidate = result["metrics"]["candidate"]
    assert candidate["quality_window_stable_traces"] == 100
    assert candidate["teacher_top30_coverage"] == 1.0
    assert candidate["teacher_commit_coverage"] == 1.0
    assert candidate["p95_improvement_ms"] == 90.0
    assert candidate["coverage_evidence_traces"] == 1
    assert candidate["commit_evidence_traces"] == 1
    assert candidate["paired_latency_traces"] == 1
    assert result["gates"]["candidate_samples"] is True
    assert result["gates"]["candidate_coverage_evidence"] is False
    assert result["gates"]["candidate_commit_evidence"] is False
    assert result["gates"]["candidate_latency_evidence"] is False
    assert result["authority_enabled"] is False


def test_canary_counter_migration_rebases_stable_units() -> None:
    assert recall_growth._advance_rollout(
        {
            "effective_mode": "active",
            "canary_percent": 25,
            "stage_started_trace_count": 500,
        },
        authority_eligible=True,
        candidate_trace_count=100,
    ) == ("active", 25, 100)
    assert recall_growth._advance_rollout(
        {
            "effective_mode": "active",
            "canary_percent": 5,
            "stage_started_stable_trace_count": 0,
        },
        authority_eligible=True,
        candidate_trace_count=100,
    ) == ("active", 25, 100)


def test_growth_cycle_promotes_qualified_evidence_through_canary(
    tmp_path: Path,
) -> None:
    inputs = _label_inputs(tmp_path)
    recalls: list[dict] = []
    pulls: list[dict] = []
    for index in range(200):
        decision = f"decision-{index}"
        session = f"session-{index % 20}"
        page = f"page-{index}"
        recalls.append(
            {
                "decision_id": decision,
                "session_id": session,
                "prompt_hash": f"{index:064x}",
                "evidence_features": {
                    "processor_shadow": {"committed_page_ids": [page]}
                },
            }
        )
        pulls.append(
            {
                "type": "used",
                "event_id": f"used-{index}",
                "decision_id": decision,
                "session_id": session,
                "page_ids": [page],
            }
        )
    _write_jsonl(inputs["recall_log_file"], recalls)
    _write_jsonl(inputs["pull_log_file"], pulls)
    _write_jsonl(inputs["certificate_file"], [])
    _write_jsonl(inputs["golden_file"], [])
    traces = [
        {
            "session_hash": f"{index % 20:016x}",
            "status": "observed",
            "field_page_ids": [f"page-{index}"],
            "teacher_page_ids": [f"page-{index}"],
            "committed_page_ids": [f"page-{index}"],
            "latency_ms": 120,
            "field_latency_ms": 20,
            "teacher_latency_ms": 120,
            "full_search_required": False,
            "over_4s": False,
            "authority": "teacher",
        }
        for index in range(100)
    ]
    candidate_trace = tmp_path / "candidate.jsonl"
    _write_jsonl(candidate_trace, traces)
    state = tmp_path / "growth-state.json"
    promotion = tmp_path / "promotion.json"

    result = recall_growth.run_growth_cycle(
        state_file=state,
        history_file=tmp_path / "history.jsonl",
        candidate_trace_file=candidate_trace,
        promotion_file=promotion,
        locked_e2e_file=_locked_gate(tmp_path),
        label_inputs=inputs,
    )

    assert result["stage"] == "canary"
    assert result["effective_mode"] == "active"
    assert result["canary_percent"] == 5
    assert result["field_learning_allowed"] is True
    assert result["positive_learning_allowed"] is True
    assert result["policy_update_allowed"] is True
    assert result["authority_enabled"] is True
    assert result["gates"]["processor_used_precision"] is True
    assert json.loads(promotion.read_text())["status"] == "passed"

    _write_jsonl(
        candidate_trace,
        traces
        + [
            {
                **row,
                "session_hash": f"{index % 20 + 20:016x}",
                "status": "active",
                "authority": "field",
            }
            for index, row in enumerate(traces)
        ],
    )
    advanced = recall_growth.run_growth_cycle(
        state_file=state,
        history_file=tmp_path / "history.jsonl",
        candidate_trace_file=candidate_trace,
        promotion_file=promotion,
        locked_e2e_file=_locked_gate(tmp_path),
        label_inputs=inputs,
    )

    assert advanced["canary_percent"] == 25

    regressed_rows = [
        {
            "session_hash": f"regressed-{index}",
            "status": "active",
            "field_page_ids": [],
            "teacher_page_ids": [f"miss-{index}"],
            "committed_page_ids": [f"miss-{index}"],
            "latency_ms": 120,
            "full_search_required": False,
            "over_4s": False,
            "authority": "field",
        }
        for index in range(3)
    ]
    _write_jsonl(candidate_trace, traces + traces + regressed_rows)
    rolled_back = recall_growth.run_growth_cycle(
        state_file=state,
        history_file=tmp_path / "history.jsonl",
        candidate_trace_file=candidate_trace,
        promotion_file=promotion,
        locked_e2e_file=_locked_gate(tmp_path),
        label_inputs=inputs,
    )

    assert rolled_back["effective_mode"] == "candidate"
    assert rolled_back["authority_enabled"] is False
    assert json.loads(promotion.read_text())["status"] == "held"


def test_automatic_rollout_and_learning_fail_closed(tmp_path: Path) -> None:
    state = tmp_path / "growth-state.json"
    assert recall_growth.automatic_rollout(
        enabled=True,
        state_file=state,
    ) == ("candidate", 100)
    assert not recall_growth.automatic_learning_allowed(
        enabled=True,
        state_file=state,
    )
    state.write_text(
        json.dumps(
            {
                "effective_mode": "active",
                "canary_percent": 25,
                "authority_enabled": True,
                "field_learning_allowed": True,
            }
        ),
        encoding="utf-8",
    )

    assert recall_growth.automatic_rollout(
        enabled=True,
        state_file=state,
    ) == ("active", 25)
    assert recall_growth.automatic_learning_allowed(
        enabled=True,
        state_file=state,
    )


def test_growth_quality_gate_uses_recent_window(tmp_path: Path) -> None:
    inputs = _label_inputs(tmp_path)
    recalls: list[dict] = []
    pulls: list[dict] = []
    for index in range(220):
        page = f"page-{index}"
        recalls.append(
            {
                "decision_id": f"decision-{index}",
                "session_id": f"session-{index % 20}",
                "prompt_hash": f"{index:064x}",
                "evidence_features": {
                    "processor_shadow": {"committed_page_ids": [page]}
                },
            }
        )
        pulls.append(
            {
                "type": "used",
                "event_id": f"used-{index}",
                "decision_id": f"decision-{index}",
                "session_id": f"session-{index % 20}",
                "page_ids": [page],
            }
        )
    _write_jsonl(inputs["recall_log_file"], recalls)
    _write_jsonl(inputs["pull_log_file"], pulls)
    _write_jsonl(inputs["certificate_file"], [])
    _write_jsonl(inputs["golden_file"], [])
    bad = [
        {
            "session_hash": f"bad-{index}",
            "status": "observed",
            "field_page_ids": [],
            "teacher_page_ids": [f"old-{index}"],
            "committed_page_ids": [f"old-{index}"],
            "latency_ms": 5_000,
            "full_search_required": False,
            "over_4s": True,
            "authority": "teacher",
        }
        for index in range(20)
    ]
    good = [
        {
            "session_hash": f"good-{index % 20}",
            "status": "observed",
            "field_page_ids": [f"page-{index}"],
            "teacher_page_ids": [f"page-{index}"],
            "committed_page_ids": [f"page-{index}"],
            "latency_ms": 100,
            "field_latency_ms": 20,
            "teacher_latency_ms": 100,
            "full_search_required": False,
            "over_4s": False,
            "authority": "teacher",
        }
        for index in range(200)
    ]
    fallback = [
        {
            "session_hash": f"fallback-{index}",
            "status": "fallback",
            "fallback_reason": "topic_reset",
            "field_page_ids": [],
            "teacher_page_ids": [f"fallback-page-{index}"],
            "committed_page_ids": [f"fallback-page-{index}"],
            "latency_ms": 100,
            "teacher_latency_ms": 100,
            "full_search_required": True,
            "over_4s": False,
            "authority": "teacher",
        }
        for index in range(50)
    ]
    candidate_trace = tmp_path / "candidate.jsonl"
    _write_jsonl(candidate_trace, bad + good + fallback)

    result = recall_growth.run_growth_cycle(
        state_file=tmp_path / "growth-state.json",
        history_file=tmp_path / "history.jsonl",
        candidate_trace_file=candidate_trace,
        promotion_file=tmp_path / "promotion.json",
        locked_e2e_file=_locked_gate(tmp_path),
        label_inputs=inputs,
    )

    assert result["authority_enabled"] is True
    assert result["metrics"]["candidate"]["traces"] == 270
    assert result["metrics"]["candidate"]["stable_traces"] == 220
    assert result["metrics"]["candidate"]["quality_window_traces"] == 200
    assert result["metrics"]["candidate"]["quality_window_stable_traces"] == 200
    assert result["metrics"]["candidate"]["quality_window_stable_sessions"] == 20
    assert result["metrics"]["candidate"]["teacher_top30_coverage"] == 1.0
    assert result["metrics"]["candidate"]["full_search_rate"] == 0.25
    assert result["metrics"]["candidate"]["over_4s"] == 0
    assert result["metrics"]["processor_used"]["episodes"] == 220
    assert result["metrics"]["processor_used"]["quality_window_episodes"] == 200
