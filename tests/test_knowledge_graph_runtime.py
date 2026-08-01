from __future__ import annotations

from pathlib import Path
from typing import Any

from chronovisor.core.durable_state import read_sealed_json
from chronovisor.knowledge_graph import runtime
from chronovisor.knowledge_graph.config import (
    GraphRetrievalConfig,
    KnowledgeGraphConfig,
)
from chronovisor.knowledge_graph.evaluation import capture_baseline
from chronovisor.knowledge_graph.rollout import advance_rollout
from chronovisor.knowledge_graph.schema import sha256
from chronovisor.knowledge_graph.store import KnowledgeGraphStore


def _config() -> KnowledgeGraphConfig:
    return KnowledgeGraphConfig(
        local_extraction_enabled=False,
        retrieval=GraphRetrievalConfig(mode="shadow"),
    )


def _capture_baseline(root: Path) -> None:
    capture_baseline(
        output_file=root / "runtime" / "typed-graph" / "baseline.json",
        git_head="a" * 40,
        runtime_commit="b" * 40,
        config_sha256="c" * 64,
        model_inventory=["gemma4:26b"],
        artifact_counts={"manual_locked": 94},
    )


def _stub_lanes(monkeypatch: Any, *, busy: bool) -> None:
    monkeypatch.setattr(runtime, "foreground_resource_busy", lambda _root: busy)
    monkeypatch.setattr(
        runtime,
        "run_builder_cycle",
        lambda **_kwargs: {"status": "ok", "external_model_calls": 0},
    )
    monkeypatch.setattr(
        runtime,
        "verify_pending_relations",
        lambda **_kwargs: {"status": "ok", "external_model_calls": 0},
    )
    monkeypatch.setattr(
        runtime,
        "advance_used_relations",
        lambda **_kwargs: {"status": "ok"},
    )
    monkeypatch.setattr(
        runtime,
        "advance_used_entities",
        lambda **_kwargs: {"status": "ok"},
    )
    monkeypatch.setattr(
        runtime,
        "consolidate_entity_candidates",
        lambda **_kwargs: {"status": "ok", "external_model_calls": 0},
    )
    monkeypatch.setattr(runtime, "build_communities", lambda _rows: [])
    monkeypatch.setattr(
        runtime,
        "summarize_communities",
        lambda rows, **_kwargs: (
            rows,
            {"status": "ok", "external_model_calls": 0},
        ),
    )
    monkeypatch.setattr(
        runtime,
        "build_locked_gold_cycle",
        lambda **_kwargs: {
            "status": "ok",
            "cases": 0,
            "external_model_calls": 0,
        },
    )
    monkeypatch.setattr(
        runtime,
        "run_calibration_cycle",
        lambda **_kwargs: {
            "status": "collecting",
            "samples": 0,
            "sessions": 0,
            "external_model_calls": 0,
        },
    )
    monkeypatch.setattr(
        runtime,
        "run_four_arm_fixture_cycle",
        lambda **_kwargs: {"status": "ok", "external_model_calls": 0},
    )
    monkeypatch.setattr(
        runtime,
        "run_evaluation_cycle",
        lambda **_kwargs: {
            "status": "collecting_or_held",
            "manifest_sha256": sha256("locked-manifest"),
            "gates": {"all_arms": False, "all_categories": False},
            "comparison": {
                "samples": {
                    "current": 0,
                    "graph_only": 0,
                    "rubric_only": 0,
                    "graph_and_rubric": 0,
                }
            },
            "external_model_calls": 0,
        },
    )
    monkeypatch.setattr(
        runtime,
        "promote_authoritative_relations",
        lambda **_kwargs: {"status": "held", "eligible": 0, "promoted": 0},
    )
    monkeypatch.setattr(
        runtime,
        "promote_authoritative_entities",
        lambda **_kwargs: {"status": "held", "eligible": 0, "promoted": 0},
    )


def test_runtime_reports_machine_checked_engineering_and_maturity_gates(
    tmp_path: Path, monkeypatch: Any
) -> None:
    _capture_baseline(tmp_path)
    _stub_lanes(monkeypatch, busy=False)

    result = runtime.run_graph_maintenance(root=tmp_path, config=_config())
    snapshot = KnowledgeGraphStore(tmp_path / "knowledge-graph").load_snapshot()

    assert result["engineering_complete"] is True
    assert all(result["engineering_gates"].values())
    assert result["authority_mature"] is False
    assert result["rollout"]["relation_snapshot_sha256"] == snapshot["seal_sha256"]
    assert result["rollout"]["rubric_sha256"] == sha256(
        {"rubric": "builtin", "version": 1}
    )
    assert result["rollout"]["sample_unit"] == "distinct_applied_session_hashes"
    assert result["rubric_gold"]["steps_run"] == 4
    assert result["authority"]["current"]["relation_strong"] == 0
    assert result["rollout"]["gates"]["authority:relation_strong"] is False
    assert result["rollout"]["gates"]["authority:rubric_sessions"] is False
    assert result["rollout"]["gates"]["learning:rubric_adopted"] is False


def test_resource_pause_preserves_existing_canary_artifact(
    tmp_path: Path, monkeypatch: Any
) -> None:
    _capture_baseline(tmp_path)
    _stub_lanes(monkeypatch, busy=False)
    first = runtime.run_graph_maintenance(root=tmp_path, config=_config())
    promotion_file = tmp_path / "runtime" / "typed-graph" / "promotion.json"
    candidate = advance_rollout(
        gates={"quality": True},
        sample_count=0,
        promotion_file=promotion_file,
        manifest_sha256=first["rollout"]["manifest_sha256"],
        relation_snapshot_sha256=first["rollout"]["relation_snapshot_sha256"],
        rubric_sha256=first["rollout"]["rubric_sha256"],
        model_manifest_sha256=first["rollout"]["model_manifest_sha256"],
    )
    before = read_sealed_json(promotion_file)
    assert candidate["canary_percent"] == 5

    _stub_lanes(monkeypatch, busy=True)
    paused = runtime.run_graph_maintenance(root=tmp_path, config=_config())
    after = read_sealed_json(promotion_file)

    assert paused["rollout"]["canary_percent"] == 5
    assert paused["rollout"]["maintenance_status"] == "paused"
    assert after["seal_sha256"] == before["seal_sha256"]
