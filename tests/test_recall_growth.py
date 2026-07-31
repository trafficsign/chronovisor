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
            "field_page_ids": [],
            "teacher_page_ids": [f"miss-{index}"],
            "committed_page_ids": [f"miss-{index}"],
            "latency_ms": 120,
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
            "field_page_ids": [],
            "teacher_page_ids": [f"old-{index}"],
            "committed_page_ids": [f"old-{index}"],
            "latency_ms": 5_000,
            "over_4s": True,
            "authority": "teacher",
        }
        for index in range(20)
    ]
    good = [
        {
            "session_hash": f"good-{index % 20}",
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
    candidate_trace = tmp_path / "candidate.jsonl"
    _write_jsonl(candidate_trace, bad + good)

    result = recall_growth.run_growth_cycle(
        state_file=tmp_path / "growth-state.json",
        history_file=tmp_path / "history.jsonl",
        candidate_trace_file=candidate_trace,
        promotion_file=tmp_path / "promotion.json",
        locked_e2e_file=_locked_gate(tmp_path),
        label_inputs=inputs,
    )

    assert result["authority_enabled"] is True
    assert result["metrics"]["candidate"]["traces"] == 220
    assert result["metrics"]["candidate"]["quality_window_traces"] == 200
    assert result["metrics"]["candidate"]["over_4s"] == 0
    assert result["metrics"]["processor_used"]["episodes"] == 220
    assert result["metrics"]["processor_used"]["quality_window_episodes"] == 200
