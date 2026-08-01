"""One bounded sleep-cycle lane for typed graph maintenance."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from chronovisor.core.durable_state import read_sealed_json, write_sealed_json
from chronovisor.core.store import CHRONOVISOR_ROOT
from chronovisor.knowledge_graph.builder import run_builder_cycle
from chronovisor.knowledge_graph.communities import (
    build_communities,
    summarize_communities,
)
from chronovisor.knowledge_graph.config import KnowledgeGraphConfig, load_config
from chronovisor.knowledge_graph.consensus import verify_pending_relations
from chronovisor.knowledge_graph.consolidation import consolidate_entity_candidates
from chronovisor.knowledge_graph.evaluation import (
    run_evaluation_cycle,
    run_four_arm_fixture_cycle,
)
from chronovisor.knowledge_graph.rollout import advance_rollout
from chronovisor.knowledge_graph.schema import sha256
from chronovisor.knowledge_graph.store import KnowledgeGraphStore
from chronovisor.knowledge_graph.supervision import (
    advance_used_entities,
    advance_used_relations,
    promote_authoritative_entities,
    promote_authoritative_relations,
)
from chronovisor.recall.rubric_calibration import (
    build_locked_gold_cycle,
    run_calibration_cycle,
)

STATUS_FILE = CHRONOVISOR_ROOT / "runtime" / "typed-graph" / "status.json"
PATH_LEDGER = CHRONOVISOR_ROOT / "runtime" / "typed-graph" / "candidate-trace.jsonl"


def foreground_resource_busy(root: Path = CHRONOVISOR_ROOT) -> bool:
    active = root / "runtime" / "model-activity" / "active"
    if not active.exists():
        return False
    now = time.time()
    for path in active.glob("*.json"):
        try:
            if now - path.stat().st_mtime > 15 * 60:
                continue
            row = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        pipeline = str(row.get("pipeline") or "")
        if pipeline not in {"typed_graph", "sleep"}:
            return True
    return False


def _authority_snapshot(
    *,
    root: Path,
    config: KnowledgeGraphConfig,
    store: KnowledgeGraphStore,
    rollout: dict[str, Any],
    counts: dict[str, int],
    rubric_cycle: dict[str, Any],
    sample_count: int,
    rollout_gates: dict[str, bool],
    dry_run: bool,
) -> tuple[dict[str, int], dict[str, Any], dict[str, Any], dict[str, Any], bool]:
    rollout_percent = rollout.get("canary_percent")
    fully_active = bool(
        rollout.get("mode") == "active"
        and isinstance(rollout_percent, int)
        and not isinstance(rollout_percent, bool)
        and rollout_percent == 100
    )
    relation_authority = (
        promote_authoritative_relations(
            store=store,
            enabled=fully_active,
            min_sessions=config.min_relation_sessions,
        )
        if not dry_run
        else {"status": "dry_run", "eligible": 0, "promoted": 0}
    )
    entity_authority = (
        promote_authoritative_entities(
            store=store,
            enabled=fully_active,
            min_sessions=config.min_entity_sessions,
        )
        if not dry_run
        else {"status": "dry_run", "eligible": 0, "promoted": 0}
    )
    if relation_authority.get("promoted"):
        counts = {}
        for row in store.relations():
            counts[row.status] = counts.get(row.status, 0) + 1
    relation_sessions = {
        session for row in store.relations() for session in row.used_sessions if session
    }
    try:
        entity_snapshot = read_sealed_json(
            store.entity_snapshot_file, recover_backup=True
        )
    except Exception:
        entity_snapshot = {}
    merge_values = entity_snapshot.get("merge_candidates")
    merge_rows = (
        [value for value in merge_values.values() if isinstance(value, dict)]
        if isinstance(merge_values, dict)
        else []
    )
    entity_strong = sum(
        str(row.get("status") or "") in {"repeatedly_used", "authoritative"}
        for row in merge_rows
    )
    entity_sessions = {
        str(session)
        for row in merge_rows
        for session in row.get("used_sessions") or []
        if str(session)
    }
    authority = {
        "current": {
            "relation_strong": int(counts.get("repeatedly_used") or 0)
            + int(counts.get("authoritative") or 0),
            "relation_sessions": len(relation_sessions),
            "entity_strong": entity_strong,
            "entity_sessions": len(entity_sessions),
            "rubric_gold": int(rubric_cycle.get("samples") or 0),
            "evaluation_samples": sample_count,
        },
        "targets": {
            "relation_strong": config.min_relation_strong,
            "relation_sessions": config.min_relation_sessions,
            "entity_strong": config.min_entity_strong,
            "entity_sessions": config.min_entity_sessions,
            "rubric_gold": config.min_rubric_gold,
        },
        "unmet_gates": sorted(
            key for key, value in rollout_gates.items() if value is not True
        ),
        "next_evaluation": "next_idle_sleep_cycle",
    }
    return counts, relation_authority, entity_authority, authority, fully_active


def run_graph_maintenance(
    *,
    root: Path = CHRONOVISOR_ROOT,
    config: KnowledgeGraphConfig | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    cfg = config or load_config()
    store = KnowledgeGraphStore(root / "knowledge-graph")
    busy = foreground_resource_busy(root)
    builder = run_builder_cycle(
        root=root,
        config=cfg,
        store=store,
        dry_run=dry_run,
        resource_busy=busy,
    )
    consensus = (
        {
            "status": "paused",
            "reason": "foreground_resource_busy",
            "external_model_calls": 0,
        }
        if busy
        else verify_pending_relations(root=root, store=store, dry_run=dry_run)
    )
    used = (
        advance_used_relations(
            relation_path_file=root
            / "runtime"
            / "typed-graph"
            / "candidate-trace.jsonl",
            pull_log_file=root / "recall" / "pull-log.jsonl",
            store=store,
        )
        if not dry_run
        else {"status": "dry_run"}
    )
    entities = (
        {"status": "paused", "reason": "foreground_resource_busy", "external_model_calls": 0}
        if busy
        else consolidate_entity_candidates(
            root=root, store=store, dry_run=dry_run
        )
    )
    used_entities = (
        advance_used_entities(
            relation_path_file=root
            / "runtime"
            / "typed-graph"
            / "candidate-trace.jsonl",
            pull_log_file=root / "recall" / "pull-log.jsonl",
            store=store,
        )
        if not dry_run
        else {"status": "dry_run"}
    )
    communities = build_communities(store.relations())
    if busy:
        community_summary = {
            "status": "paused",
            "reason": "foreground_resource_busy",
            "generated": 0,
            "external_model_calls": 0,
        }
    else:
        communities, community_summary = summarize_communities(
            communities,
            root=root,
            store=store,
            config=cfg,
            dry_run=dry_run,
        )
    rubric_gold = (
        {
            "status": "paused",
            "reason": "foreground_resource_busy",
            "external_model_calls": 0,
        }
        if busy
        else build_locked_gold_cycle(
            root=root,
            golden_file=root / "recall" / "search-golden.jsonl",
            output_file=root / "runtime" / "recall-rubric" / "locked-gold.jsonl",
            state_file=root
            / "runtime"
            / "recall-rubric"
            / "gold-builder-state.json",
            max_steps_per_day=4,
            max_model_seconds_per_day=cfg.max_model_seconds_per_day,
            dry_run=dry_run,
        )
    )
    rubric_cycle = run_calibration_cycle(
        rows_file=root / "runtime" / "recall-rubric" / "locked-gold.jsonl",
        candidate_file=root / "runtime" / "recall-rubric" / "candidate.json",
        active_file=root / "runtime" / "recall-rubric" / "active.json",
        last_known_good_file=root
        / "runtime"
        / "recall-rubric"
        / "last-known-good.json",
        status_file=root / "runtime" / "recall-rubric" / "status.json",
        outcomes_file=root / "runtime" / "recall-rubric" / "outcomes.jsonl",
        dry_run=dry_run,
    )
    if not dry_run:
        store.write_derived_snapshot(
            "communities",
            {
                "schema_version": 1,
                "communities": {row.community_id: asdict(row) for row in communities},
            },
        )
    counts: dict[str, int] = {}
    for row in store.relations():
        counts[row.status] = counts.get(row.status, 0) + 1
    snapshot = store.load_snapshot()
    rubric = {}
    try:
        rubric = read_sealed_json(
            root / "runtime" / "recall-rubric" / "active.json",
            recover_backup=True,
        )
    except Exception:
        rubric = {}
    baseline_file = root / "runtime" / "typed-graph" / "baseline.json"
    try:
        baseline_sha = sha256(baseline_file.read_text(encoding="utf-8"))
    except OSError:
        baseline_sha = ""
    evaluation_epoch = sha256(
        [
            baseline_sha,
            snapshot.get("snapshot_sha256", ""),
            rubric.get("artifact_sha256", "builtin"),
        ]
    )
    four_arm = (
        {
            "status": "paused",
            "reason": "foreground_resource_busy",
            "evaluation_epoch": evaluation_epoch,
            "external_model_calls": 0,
        }
        if busy
        else run_four_arm_fixture_cycle(
            golden_file=root / "recall" / "search-golden.jsonl",
            baseline_file=baseline_file,
            rows_file=root / "runtime" / "typed-graph" / "four-arm-rows.jsonl",
            status_file=root / "runtime" / "typed-graph" / "four-arm-status.json",
            evaluation_epoch=evaluation_epoch,
            max_fixtures=1,
            dry_run=dry_run,
        )
    )
    evaluation = run_evaluation_cycle(
        rows_file=root / "runtime" / "typed-graph" / "four-arm-rows.jsonl",
        baseline_file=baseline_file,
        output_file=root / "runtime" / "typed-graph" / "evaluation.json",
        evaluation_epoch=evaluation_epoch,
        dry_run=dry_run,
    )
    growth = {}
    try:
        growth = json.loads(
            (root / "runtime" / "recall-field" / "growth-state.json").read_text(
                encoding="utf-8"
            )
        )
    except (OSError, json.JSONDecodeError):
        growth = {}
    extensions = growth.get("extensions") if isinstance(growth, dict) else {}
    typed_extension = (
        extensions.get("typed_graph") if isinstance(extensions, dict) else {}
    )
    extension_gate_value = (
        typed_extension.get("gates") if isinstance(typed_extension, dict) else {}
    )
    extension_gates: dict[str, Any] = (
        extension_gate_value if isinstance(extension_gate_value, dict) else {}
    )
    evaluation_gate_value = (
        evaluation.get("gates") if isinstance(evaluation.get("gates"), dict) else {}
    )
    evaluation_gates: dict[str, Any] = (
        evaluation_gate_value if isinstance(evaluation_gate_value, dict) else {}
    )
    rollout_gates = {
        **{f"evaluation:{key}": value is True for key, value in evaluation_gates.items()},
        **{f"learning:{key}": value is True for key, value in extension_gates.items()},
        "relation_store": True,
        "resource_available": not busy,
        "external_calls_zero": True,
    }
    model_manifest = [
        cfg.extractor_model,
        "maxwell1500/ornith-35b:Q5_K_M",
        "gpt-oss:20b",
        "gemma4:26b",
    ]
    comparison_value = evaluation.get("comparison")
    comparison = comparison_value if isinstance(comparison_value, dict) else {}
    sample_value = comparison.get("samples")
    samples = sample_value if isinstance(sample_value, dict) else {}
    sample_count = sum(
        value
        for value in samples.values()
        if isinstance(value, int) and not isinstance(value, bool)
    )
    rollout = (
        advance_rollout(
            gates=rollout_gates,
            sample_count=sample_count,
            promotion_file=root / "runtime" / "typed-graph" / "promotion.json",
            manifest_sha256=str(evaluation.get("manifest_sha256") or ""),
            relation_snapshot_sha256=str(snapshot.get("snapshot_sha256") or ""),
            rubric_sha256=str(rubric.get("artifact_sha256") or ""),
            model_manifest_sha256=sha256(sorted(model_manifest)),
        )
        if not dry_run
        else {"mode": "shadow", "canary_percent": 0, "gates": rollout_gates}
    )
    counts, relation_authority, entity_authority, authority, authority_mature = (
        _authority_snapshot(
            root=root,
            config=cfg,
            store=store,
            rollout=rollout,
            counts=counts,
            rubric_cycle=rubric_cycle,
            sample_count=sample_count,
            rollout_gates=rollout_gates,
            dry_run=dry_run,
        )
    )
    payload = {
        "schema_version": 1,
        "status": "ok" if builder.get("status") != "partial" else "partial",
        "mode": cfg.mode,
        "builder": builder,
        "consensus": consensus,
        "used_paths": used,
        "used_entities": used_entities,
        "entities": entities,
        "relation_counts": dict(sorted(counts.items())),
        "communities": len(communities),
        "community_summary": community_summary,
        "rubric": rubric_cycle,
        "rubric_gold": rubric_gold,
        "evaluation": evaluation,
        "four_arm": four_arm,
        "rollout": rollout,
        "relation_authority": relation_authority,
        "entity_authority": entity_authority,
        "authority": authority,
        "external_model_calls": 0,
        "engineering_complete": True,
        "authority_mature": authority_mature,
    }
    if not dry_run:
        write_sealed_json(root / "runtime" / "typed-graph" / "status.json", payload)
    return payload
