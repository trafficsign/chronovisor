"""One bounded sleep-cycle lane for typed graph maintenance."""

from __future__ import annotations

import json
import re
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
    baseline_manifest_sha256,
    run_evaluation_cycle,
    run_four_arm_fixture_cycle,
    validate_baseline,
)
from chronovisor.knowledge_graph.rollout import (
    CANARY_SAMPLE_UNIT,
    advance_rollout,
    applied_canary_session_count,
)
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
REQUIRED_LEARNING_GATES = (
    "relation_learning",
    "entity_learning",
    "rubric_learning",
    "rubric_adopted",
    "four_arm_evaluation",
    "external_calls_zero",
)


def _is_sha256(value: object) -> bool:
    return bool(re.fullmatch(r"[0-9a-f]{64}", str(value or "")))


def _external_model_call_count(*payloads: dict[str, Any]) -> int:
    return sum(
        max(0, int(payload.get("external_model_calls") or 0))
        for payload in payloads
        if isinstance(payload, dict)
    )


def _evaluation_external_call_detected(evaluation: dict[str, Any]) -> int:
    comparison = evaluation.get("comparison")
    metrics = comparison.get("metrics") if isinstance(comparison, dict) else None
    if not isinstance(metrics, dict):
        return 0
    return int(
        any(
            isinstance(values, dict)
            and isinstance(values.get("external_model_calls"), int | float)
            and float(values["external_model_calls"]) > 0
            for values in metrics.values()
        )
    )


def _current_authority_counts(
    *,
    store: KnowledgeGraphStore,
    relation_counts: dict[str, int],
    rubric_cycle: dict[str, Any],
    evaluation_sample_count: int,
) -> dict[str, int]:
    strong_relations = [
        row
        for row in store.relations()
        if row.status in {"repeatedly_used", "authoritative"}
    ]
    relation_sessions = {
        session for row in strong_relations for session in row.used_sessions if session
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
    entity_sessions = {
        str(session)
        for row in merge_rows
        for session in row.get("used_sessions") or []
        if str(session)
    }
    return {
        "relation_strong": int(relation_counts.get("repeatedly_used") or 0)
        + int(relation_counts.get("authoritative") or 0),
        "relation_sessions": len(relation_sessions),
        "entity_strong": sum(
            str(row.get("status") or "") in {"repeatedly_used", "authoritative"}
            for row in merge_rows
        ),
        "entity_sessions": len(entity_sessions),
        "rubric_gold": int(rubric_cycle.get("samples") or 0),
        "rubric_sessions": int(rubric_cycle.get("sessions") or 0),
        "evaluation_samples": evaluation_sample_count,
    }


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


def _build_rubric_gold_batch(
    *,
    root: Path,
    config: KnowledgeGraphConfig,
    dry_run: bool,
    max_steps_per_cycle: int = 4,
) -> dict[str, Any]:
    """Use the declared daily budget within the single nightly sleep run."""

    result: dict[str, Any] = {
        "status": "waiting",
        "reason": "no_steps_requested",
        "external_model_calls": 0,
    }
    steps_run = 0
    for _ in range(max(0, max_steps_per_cycle)):
        result = build_locked_gold_cycle(
            root=root,
            golden_file=root / "recall" / "search-golden.jsonl",
            output_file=root / "runtime" / "recall-rubric" / "locked-gold.jsonl",
            state_file=root
            / "runtime"
            / "recall-rubric"
            / "gold-builder-state.json",
            max_steps_per_day=4,
            max_model_seconds_per_day=config.max_model_seconds_per_day,
            dry_run=dry_run,
        )
        if result.get("status") != "ok":
            break
        steps_run += 1
    return {**result, "steps_run": steps_run}


def _authority_snapshot(
    *,
    root: Path,
    config: KnowledgeGraphConfig,
    store: KnowledgeGraphStore,
    rollout: dict[str, Any],
    counts: dict[str, int],
    rubric_cycle: dict[str, Any],
    evaluation_sample_count: int,
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
    current = _current_authority_counts(
        store=store,
        relation_counts=counts,
        rubric_cycle=rubric_cycle,
        evaluation_sample_count=evaluation_sample_count,
    )
    targets = {
        "relation_strong": config.min_relation_strong,
        "relation_sessions": config.min_relation_sessions,
        "entity_strong": config.min_entity_strong,
        "entity_sessions": config.min_entity_sessions,
        "rubric_gold": config.min_rubric_gold,
        "rubric_sessions": config.min_rubric_sessions,
    }
    authority = {
        "current": current,
        "targets": targets,
        "unmet_gates": sorted(
            key for key, value in rollout_gates.items() if value is not True
        ),
        "next_evaluation": "next_idle_sleep_cycle",
    }
    mature = fully_active and all(
        current.get(key, 0) >= target for key, target in targets.items()
    )
    return counts, relation_authority, entity_authority, authority, mature


def _paused(reason: str = "foreground_resource_busy") -> dict[str, Any]:
    return {"status": "paused", "reason": reason, "external_model_calls": 0}


def _run_maintenance_lanes(
    *,
    root: Path,
    config: KnowledgeGraphConfig,
    store: KnowledgeGraphStore,
    busy: bool,
    dry_run: bool,
) -> dict[str, Any]:
    builder = run_builder_cycle(
        root=root,
        config=config,
        store=store,
        dry_run=dry_run,
        resource_busy=busy,
    )
    consensus = (
        _paused()
        if busy
        else verify_pending_relations(root=root, store=store, dry_run=dry_run)
    )
    trace_file = root / "runtime" / "typed-graph" / "candidate-trace.jsonl"
    pull_file = root / "recall" / "pull-log.jsonl"
    used = (
        advance_used_relations(
            relation_path_file=trace_file, pull_log_file=pull_file, store=store
        )
        if not dry_run
        else {"status": "dry_run"}
    )
    entities = (
        _paused()
        if busy
        else consolidate_entity_candidates(root=root, store=store, dry_run=dry_run)
    )
    used_entities = (
        advance_used_entities(
            relation_path_file=trace_file, pull_log_file=pull_file, store=store
        )
        if not dry_run
        else {"status": "dry_run"}
    )
    communities = build_communities(store.relations())
    if busy:
        community_summary = {**_paused(), "generated": 0}
    else:
        communities, community_summary = summarize_communities(
            communities,
            root=root,
            store=store,
            config=config,
            dry_run=dry_run,
        )
    rubric_gold = (
        _paused()
        if busy
        else _build_rubric_gold_batch(root=root, config=config, dry_run=dry_run)
    )
    rubric_status_file = root / "runtime" / "recall-rubric" / "status.json"
    if busy:
        try:
            rubric_cycle = read_sealed_json(rubric_status_file, recover_backup=True)
        except Exception:
            rubric_cycle = _paused()
    else:
        rubric_cycle = run_calibration_cycle(
            rows_file=root / "runtime" / "recall-rubric" / "locked-gold.jsonl",
            candidate_file=root / "runtime" / "recall-rubric" / "candidate.json",
            active_file=root / "runtime" / "recall-rubric" / "active.json",
            last_known_good_file=root
            / "runtime"
            / "recall-rubric"
            / "last-known-good.json",
            status_file=rubric_status_file,
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
    return {
        "builder": builder,
        "consensus": consensus,
        "used": used,
        "entities": entities,
        "used_entities": used_entities,
        "communities": communities,
        "community_summary": community_summary,
        "rubric_gold": rubric_gold,
        "rubric_cycle": rubric_cycle,
    }


def _evaluation_artifacts(
    *,
    root: Path,
    config: KnowledgeGraphConfig,
    store: KnowledgeGraphStore,
    busy: bool,
    dry_run: bool,
) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for row in store.relations():
        counts[row.status] = counts.get(row.status, 0) + 1
    snapshot = store.load_snapshot()
    try:
        rubric = read_sealed_json(
            root / "runtime" / "recall-rubric" / "active.json",
            recover_backup=True,
        )
    except Exception:
        rubric = {}
    baseline_file = root / "runtime" / "typed-graph" / "baseline.json"
    try:
        baseline = read_sealed_json(baseline_file, recover_backup=True)
        baseline_validation = validate_baseline(baseline_file)
    except Exception:
        baseline = {}
        baseline_validation = {"status": "failed", "checks": {}}
    rubric_sha = str(rubric.get("artifact_sha256") or "") or sha256(
        {"rubric": "builtin", "version": 1}
    )
    relation_sha = str(snapshot.get("seal_sha256") or "")
    evaluation_epoch = sha256(
        [baseline_manifest_sha256(baseline), relation_sha, rubric_sha]
    )
    four_arm = (
        {
            **_paused(),
            "evaluation_epoch": evaluation_epoch,
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
    return {
        "counts": counts,
        "snapshot": snapshot,
        "baseline_validation": baseline_validation,
        "rubric_sha": rubric_sha,
        "relation_sha": relation_sha,
        "four_arm": four_arm,
        "evaluation": evaluation,
        "model_manifest": [
            config.extractor_model,
            "maxwell1500/ornith-35b:Q5_K_M",
            "gpt-oss:20b",
            "gemma4:26b",
        ],
    }


def _learning_extension_gates(root: Path) -> dict[str, Any]:
    try:
        growth = json.loads(
            (root / "runtime" / "recall-field" / "growth-state.json").read_text(
                encoding="utf-8"
            )
        )
    except (OSError, json.JSONDecodeError):
        return {}
    extensions = growth.get("extensions") if isinstance(growth, dict) else None
    typed = extensions.get("typed_graph") if isinstance(extensions, dict) else None
    gates = typed.get("gates") if isinstance(typed, dict) else None
    return gates if isinstance(gates, dict) else {}


def _paused_rollout(
    *,
    promotion_file: Path,
    fallback: dict[str, Any],
    manifest_sha256: str,
    relation_snapshot_sha256: str,
    rubric_sha256: str,
    model_manifest_sha256: str,
) -> dict[str, Any]:
    """Preserve a valid canary while migrating pre-contract artifacts in place."""

    try:
        rollout = read_sealed_json(promotion_file, recover_backup=True)
    except Exception:
        rollout = dict(fallback)
    migrated = False
    digest_defaults = {
        "manifest_sha256": manifest_sha256,
        "relation_snapshot_sha256": relation_snapshot_sha256,
        "rubric_sha256": rubric_sha256,
        "model_manifest_sha256": model_manifest_sha256,
    }
    for key, value in digest_defaults.items():
        if not _is_sha256(rollout.get(key)) and _is_sha256(value):
            rollout[key] = value
            migrated = True
    if rollout.get("sample_unit") != CANARY_SAMPLE_UNIT:
        sample_count = int(fallback.get("sample_count") or 0)
        rollout["sample_count"] = sample_count
        rollout["stage_started_sample_count"] = sample_count
        rollout["sample_unit"] = CANARY_SAMPLE_UNIT
        migrated = True
    if rollout.get("rollback_teacher") != "current":
        rollout["rollback_teacher"] = "current"
        migrated = True
    if not isinstance(rollout.get("gates"), dict):
        rollout["gates"] = fallback["gates"]
        migrated = True
    if migrated:
        rollout = write_sealed_json(promotion_file, rollout)
    return {
        **rollout,
        "maintenance_status": "paused",
        "maintenance_reason": "foreground_resource_busy",
    }


def _rollout_and_authority(
    *,
    root: Path,
    config: KnowledgeGraphConfig,
    store: KnowledgeGraphStore,
    busy: bool,
    dry_run: bool,
    lanes: dict[str, Any],
    artifacts: dict[str, Any],
) -> dict[str, Any]:
    evaluation = artifacts["evaluation"]
    evaluation_gates = (
        evaluation.get("gates") if isinstance(evaluation.get("gates"), dict) else {}
    )
    extension_gates = _learning_extension_gates(root)
    rollout_gates = {
        **{f"evaluation:{key}": value is True for key, value in evaluation_gates.items()},
        **{
            f"learning:{key}": extension_gates.get(key) is True
            for key in REQUIRED_LEARNING_GATES
        },
        "relation_store": _is_sha256(artifacts["snapshot"].get("seal_sha256")),
    }
    comparison = evaluation.get("comparison")
    samples = comparison.get("samples") if isinstance(comparison, dict) else None
    evaluation_samples = sum(
        value
        for value in (samples.values() if isinstance(samples, dict) else [])
        if isinstance(value, int) and not isinstance(value, bool)
    )
    current = _current_authority_counts(
        store=store,
        relation_counts=artifacts["counts"],
        rubric_cycle=lanes["rubric_cycle"],
        evaluation_sample_count=evaluation_samples,
    )
    targets = {
        "relation_strong": config.min_relation_strong,
        "relation_sessions": config.min_relation_sessions,
        "entity_strong": config.min_entity_strong,
        "entity_sessions": config.min_entity_sessions,
        "rubric_gold": config.min_rubric_gold,
        "rubric_sessions": config.min_rubric_sessions,
    }
    rollout_gates.update(
        {
            f"authority:{key}": current.get(key, 0) >= target
            for key, target in targets.items()
        }
    )
    external_calls = _external_model_call_count(
        lanes["builder"],
        lanes["consensus"],
        lanes["entities"],
        lanes["community_summary"],
        lanes["rubric_gold"],
        lanes["rubric_cycle"],
        artifacts["four_arm"],
        evaluation,
    ) + _evaluation_external_call_detected(evaluation)
    rollout_gates["external_calls_zero"] = (
        not config.external_models_allowed and external_calls == 0
    )
    canary_samples = applied_canary_session_count(
        root / "runtime" / "typed-graph" / "candidate-trace.jsonl"
    )
    promotion_file = root / "runtime" / "typed-graph" / "promotion.json"
    fallback = {
        "mode": "shadow",
        "canary_percent": 0,
        "gates": rollout_gates,
        "sample_count": canary_samples,
        "sample_unit": CANARY_SAMPLE_UNIT,
        "rollback_teacher": "current",
    }
    if dry_run:
        rollout = fallback
    elif busy:
        rollout = _paused_rollout(
            promotion_file=promotion_file,
            fallback=fallback,
            manifest_sha256=str(evaluation.get("manifest_sha256") or ""),
            relation_snapshot_sha256=artifacts["relation_sha"],
            rubric_sha256=artifacts["rubric_sha"],
            model_manifest_sha256=sha256(sorted(artifacts["model_manifest"])),
        )
    else:
        rollout = advance_rollout(
            gates=rollout_gates,
            sample_count=canary_samples,
            promotion_file=promotion_file,
            manifest_sha256=str(evaluation.get("manifest_sha256") or ""),
            relation_snapshot_sha256=artifacts["relation_sha"],
            rubric_sha256=artifacts["rubric_sha"],
            model_manifest_sha256=sha256(sorted(artifacts["model_manifest"])),
        )
    counts, relation_authority, entity_authority, authority, mature = (
        _authority_snapshot(
            root=root,
            config=config,
            store=store,
            rollout=rollout,
            counts=artifacts["counts"],
            rubric_cycle=lanes["rubric_cycle"],
            evaluation_sample_count=evaluation_samples,
            rollout_gates=rollout_gates,
            dry_run=dry_run,
        )
    )
    engineering_gates = {
        "baseline_valid": artifacts["baseline_validation"].get("status") == "passed",
        "sealed_manifest": _is_sha256(rollout.get("manifest_sha256")),
        "sealed_relation_snapshot": _is_sha256(
            rollout.get("relation_snapshot_sha256")
        ),
        "sealed_rubric": _is_sha256(rollout.get("rubric_sha256")),
        "sealed_model_manifest": _is_sha256(rollout.get("model_manifest_sha256")),
        "automatic_canary": rollout.get("sample_unit") == CANARY_SAMPLE_UNIT,
        "current_fallback": rollout.get("rollback_teacher") == "current",
        "external_calls_zero": (
            not config.external_models_allowed and external_calls == 0
        ),
    }
    return {
        "counts": counts,
        "rollout": rollout,
        "relation_authority": relation_authority,
        "entity_authority": entity_authority,
        "authority": authority,
        "authority_mature": mature,
        "external_model_calls": external_calls,
        "engineering_gates": engineering_gates,
    }


def run_graph_maintenance(
    *,
    root: Path = CHRONOVISOR_ROOT,
    config: KnowledgeGraphConfig | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    cfg = config or load_config()
    store = KnowledgeGraphStore(root / "knowledge-graph")
    busy = foreground_resource_busy(root)
    lanes = _run_maintenance_lanes(
        root=root, config=cfg, store=store, busy=busy, dry_run=dry_run
    )
    artifacts = _evaluation_artifacts(
        root=root, config=cfg, store=store, busy=busy, dry_run=dry_run
    )
    decision = _rollout_and_authority(
        root=root,
        config=cfg,
        store=store,
        busy=busy,
        dry_run=dry_run,
        lanes=lanes,
        artifacts=artifacts,
    )
    payload = {
        "schema_version": 1,
        "status": "ok" if lanes["builder"].get("status") != "partial" else "partial",
        "mode": cfg.mode,
        "builder": lanes["builder"],
        "consensus": lanes["consensus"],
        "used_paths": lanes["used"],
        "used_entities": lanes["used_entities"],
        "entities": lanes["entities"],
        "relation_counts": dict(sorted(decision["counts"].items())),
        "communities": len(lanes["communities"]),
        "community_summary": lanes["community_summary"],
        "rubric": lanes["rubric_cycle"],
        "rubric_gold": lanes["rubric_gold"],
        "baseline": artifacts["baseline_validation"],
        "evaluation": artifacts["evaluation"],
        "four_arm": artifacts["four_arm"],
        "rollout": decision["rollout"],
        "relation_authority": decision["relation_authority"],
        "entity_authority": decision["entity_authority"],
        "authority": decision["authority"],
        "external_model_calls": decision["external_model_calls"],
        "engineering_gates": decision["engineering_gates"],
        "engineering_complete": all(decision["engineering_gates"].values()),
        "authority_mature": decision["authority_mature"],
    }
    if not dry_run:
        write_sealed_json(root / "runtime" / "typed-graph" / "status.json", payload)
    return payload
