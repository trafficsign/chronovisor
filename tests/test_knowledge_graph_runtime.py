from __future__ import annotations

from pathlib import Path
from typing import Any

from chronovisor.core.durable_state import read_sealed_json, write_sealed_json
from chronovisor.core.knowledge_graph_config import (
    GraphRetrievalConfig,
    KnowledgeGraphConfig,
)
from chronovisor.core.knowledge_graph_rollout import advance_rollout
from chronovisor.core.knowledge_graph_schema import sha256
from chronovisor.core.knowledge_graph_store import KnowledgeGraphStore
from chronovisor.knowledge_graph.evaluation import capture_baseline
from chronovisor.ops import graph_maintenance as runtime


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
        lambda **_kwargs: {
            "status": "ok",
            "external_model_calls": 0,
            "route_identity": {
                "role": "knowledge.relation_extraction",
                "provider": "ollama",
                "model": "gemma4:26b",
                "location": "local",
            },
            "model_sha256": "a" * 64,
            "local_model_digest": "local-a",
        },
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
            {
                "status": "ok",
                "external_model_calls": 0,
                "route_identity": {
                    "role": "knowledge.community_summary",
                    "provider": "ollama",
                    "model": "gemma4:26b",
                    "location": "local",
                },
                "model_sha256": "b" * 64,
                "local_model_digest": "local-b",
            },
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
    assert result["rollout"]["model_manifest_sha256"] == sha256(
        [
            {
                "role": "knowledge.relation_extraction",
                "provider": "ollama",
                "model": "gemma4:26b",
                "location": "local",
                "model_sha256": "a" * 64,
                "local_model_digest": "local-a",
            },
            {
                "role": "knowledge.community_summary",
                "provider": "ollama",
                "model": "gemma4:26b",
                "location": "local",
                "model_sha256": "b" * 64,
                "local_model_digest": "local-b",
            },
        ]
    )
    assert result["rollout"]["sample_unit"] == "distinct_applied_session_hashes"
    assert result["rubric_gold"]["steps_run"] == 4
    assert result["authority"]["current"]["relation_strong"] == 0
    assert result["rollout"]["gates"]["authority:relation_strong"] is False
    assert result["rollout"]["gates"]["authority:rubric_sessions"] is False
    assert result["rollout"]["gates"]["learning:rubric_adopted"] is False


def test_rollout_fails_closed_when_runtime_route_is_unresolved(
    tmp_path: Path, monkeypatch: Any
) -> None:
    _capture_baseline(tmp_path)
    _stub_lanes(monkeypatch, busy=False)
    monkeypatch.setattr(
        runtime,
        "run_builder_cycle",
        lambda **_kwargs: {
            "status": "partial",
            "external_model_calls": 0,
            "route_identity": {},
            "model_sha256": "",
            "local_model_digest": "",
        },
    )

    result = runtime.run_graph_maintenance(root=tmp_path, config=_config())

    assert result["rollout"]["gates"]["runtime_routes_resolved"] is False
    assert result["rollout"]["canary_percent"] == 0


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


def test_resource_pause_migrates_pre_contract_rollout_without_advancing(
    tmp_path: Path, monkeypatch: Any
) -> None:
    _capture_baseline(tmp_path)
    promotion_file = tmp_path / "runtime" / "typed-graph" / "promotion.json"
    write_sealed_json(
        promotion_file,
        {
            "schema_version": 1,
            "mode": "shadow",
            "canary_percent": 0,
            "sample_count": 12,
            "gates": {"legacy": False},
        },
    )
    _stub_lanes(monkeypatch, busy=True)

    paused = runtime.run_graph_maintenance(root=tmp_path, config=_config())
    migrated = read_sealed_json(promotion_file)

    assert paused["engineering_complete"] is True
    assert all(paused["engineering_gates"].values())
    assert paused["rollout"]["maintenance_status"] == "paused"
    assert migrated["canary_percent"] == 0
    assert migrated["sample_count"] == 0
    assert migrated["stage_started_sample_count"] == 0
    assert migrated["sample_unit"] == "distinct_applied_session_hashes"
    assert migrated["rollback_teacher"] == "current"
    assert "maintenance_status" not in migrated
    assert all(
        len(migrated[key]) == 64
        for key in (
            "manifest_sha256",
            "relation_snapshot_sha256",
            "rubric_sha256",
            "model_manifest_sha256",
        )
    )
