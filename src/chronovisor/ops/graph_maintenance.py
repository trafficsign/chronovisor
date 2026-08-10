"""One bounded sleep-cycle lane for typed graph maintenance."""

from __future__ import annotations

import json
import re
import time
from collections.abc import Mapping
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from chronovisor.core.durable_state import read_sealed_json, write_sealed_json
from chronovisor.core.jsonl_write import append_jsonl_durable
from chronovisor.core.knowledge_graph_config import KnowledgeGraphConfig, load_config
from chronovisor.core.knowledge_graph_rollout import (
    CANARY_SAMPLE_UNIT,
    advance_rollout,
    applied_canary_session_count,
)
from chronovisor.core.knowledge_graph_schema import sha256
from chronovisor.core.knowledge_graph_store import KnowledgeGraphStore
from chronovisor.core.store import CHRONOVISOR_ROOT
from chronovisor.knowledge_graph.builder import run_builder_cycle
from chronovisor.knowledge_graph.communities import (
    build_communities,
    summarize_communities,
)
from chronovisor.knowledge_graph.consensus import verify_pending_relations
from chronovisor.knowledge_graph.consolidation import consolidate_entity_candidates
from chronovisor.knowledge_graph.evaluation import (
    EVALUATION_ARMS,
    baseline_manifest_sha256,
    run_evaluation_cycle,
    validate_baseline,
)
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
)


def _arm_metrics(
    result: Mapping[str, Any],
    *,
    expected: set[str],
    negative: set[str],
) -> dict[str, Any]:
    ranked = [
        str(getattr(page, "page_id", ""))
        for page in result.get("results") or []
        if str(getattr(page, "page_id", ""))
    ]
    processor = result.get("processor")
    processor = processor if isinstance(processor, dict) else {}
    selected_values = processor.get("selected")
    selected = selected_values if isinstance(selected_values, list) else []
    committed = [
        str(row.get("page_id") or "")
        for row in selected
        if isinstance(row, dict) and row.get("page_id")
    ]
    by_kind: dict[str, list[str]] = {"pointer": [], "rich": []}
    for row in selected:
        if not isinstance(row, dict):
            continue
        kind = str(row.get("evidence_kind") or "")
        if kind in by_kind:
            by_kind[kind].append(str(row.get("page_id") or ""))

    def precision(values: list[str]) -> float:
        return sum(value in expected for value in values) / len(values) if values else 1.0

    first_rank = next(
        (index for index, page_id in enumerate(ranked, 1) if page_id in expected), None
    )
    stages = result.get("stages")
    stages = stages if isinstance(stages, dict) else {}
    trace_value = result.get("search_trace")
    trace = trace_value if isinstance(trace_value, dict) else {}
    paths_value = trace.get("paths")
    paths = paths_value if isinstance(paths_value, dict) else {}
    typed_targets = {
        str(page_id)
        for page_id, path in paths.items()
        if isinstance(path, dict)
        and (bool(path.get("relation_ids")) or bool(path.get("community_id")))
    }
    latency = int(result.get("latency_ms") or 0)
    pointer_correct = sum(page_id in expected for page_id in by_kind["pointer"])
    pointer_total = len(by_kind["pointer"])
    return {
        "recall_at_5": len(expected.intersection(ranked[:5])) / max(1, len(expected)),
        "recall_at_10": len(expected.intersection(ranked[:10])) / max(1, len(expected)),
        "mrr": 1.0 / first_rank if first_rank else 0.0,
        "used_page_coverage": len(expected.intersection(committed)) / max(1, len(expected)),
        "pointer_precision": precision(by_kind["pointer"]),
        "rich_precision": precision(by_kind["rich"]),
        "pointer_correct": pointer_correct,
        "pointer_total": pointer_total,
        "negative_hit": float(bool(negative.intersection(committed))),
        "abstention": float(not committed),
        "merge_error": 0.0,
        "relation_path_precision": precision(sorted(typed_targets)),
        "latency_p50_ms": latency,
        "latency_p95_ms": latency,
        "latency_max_ms": latency,
        "over_4s": int(latency > 4_000),
        "memory_mb": 0.0,
        "model_seconds": round(latency / 1_000, 6),
        "external_model_calls": 0,
        "candidate_generated": bool(stages.get("candidate_union")),
        "rerank_passed": bool(stages.get("reranked")),
        "certificate_passed": bool(stages.get("page_gate")),
        "committed": bool(committed),
        "actually_used": False,
    }


def run_four_arm_fixture_cycle(
    *,
    golden_file: Path,
    baseline_file: Path,
    rows_file: Path,
    status_file: Path,
    evaluation_epoch: str,
    max_fixtures: int = 1,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Incrementally execute the four preregistered arms on locked real queries."""

    try:
        baseline = read_sealed_json(baseline_file, recover_backup=True)
    except Exception:
        return {"status": "waiting", "reason": "baseline_missing", "processed": 0}
    fixture_value = baseline.get("fixture_manifest")
    fixture_rows = fixture_value.get("fixtures") if isinstance(fixture_value, dict) else []
    fixtures: list[Any] = fixture_rows if isinstance(fixture_rows, list) else []
    golden_rows: list[dict[str, Any]] = []
    try:
        for line in golden_file.read_text(encoding="utf-8").splitlines():
            value = json.loads(line)
            if isinstance(value, dict) and value.get("reviewed") is True:
                golden_rows.append(value)
    except (OSError, json.JSONDecodeError):
        golden_rows = []
    by_query = {
        sha256(str(row.get("query") or "")): row
        for row in golden_rows
        if str(row.get("query") or "")
    }
    existing: set[tuple[str, str, str]] = set()
    try:
        for line in rows_file.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if isinstance(row, dict):
                existing.add(
                    (
                        str(row.get("evaluation_epoch") or ""),
                        str(row.get("query_sha256") or ""),
                        str(row.get("arm") or ""),
                    )
                )
    except (OSError, json.JSONDecodeError):
        pass
    from chronovisor.search.search_eval import run_variant

    written: list[dict[str, Any]] = []
    errors: list[str] = []
    processed = 0
    for fixture in fixtures:
        if processed >= max(0, max_fixtures) or not isinstance(fixture, dict):
            break
        query_sha = str(fixture.get("query_sha256") or "")
        golden = by_query.get(query_sha)
        if golden is None:
            continue
        pending_arms = [
            arm
            for arm in EVALUATION_ARMS
            if (evaluation_epoch, query_sha, arm) not in existing
        ]
        if not pending_arms:
            continue
        query = str(golden.get("query") or "")
        expected = {
            str(value)
            for value in golden.get("expected_pages") or []
            if isinstance(value, str)
        }
        negative = {
            str(value)
            for value in (
                list(golden.get("negative_pages") or [])
                + list(golden.get("stale_pages") or [])
            )
            if isinstance(value, str)
        }
        for arm in pending_arms:
            typed = "candidate" if arm in {"graph_only", "graph_and_rubric"} else "off"
            calibrated = arm in {"rubric_only", "graph_and_rubric"}
            try:
                result = run_variant(
                    query,
                    "hybrid-current",
                    top_n=20,
                    typed_retrieval_mode=typed,
                    calibrated_judge=calibrated,
                )
            except Exception as exc:
                errors.append(f"{arm}:{type(exc).__name__}")
                continue
            written.append(
                {
                    "schema_version": 1,
                    "evaluation_epoch": evaluation_epoch,
                    "fixture_id": str(fixture.get("fixture_id") or ""),
                    "category": str(fixture.get("category") or ""),
                    "query_sha256": query_sha,
                    "arm": arm,
                    **_arm_metrics(result, expected=expected, negative=negative),
                }
            )
        processed += 1
    if written and not dry_run:
        append_jsonl_durable(rows_file, written, sort_keys=True)
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "status": "ok" if not errors else "partial",
        "evaluation_epoch": evaluation_epoch,
        "processed": processed,
        "rows_written": len(written),
        "errors": errors[:20],
        "external_model_calls": 0,
    }
    return payload if dry_run else write_sealed_json(status_file, payload)


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
    if builder.get("status") == "blocked":
        blocked = {
            "status": "blocked",
            "reason": str(builder.get("reason") or "builder_blocked"),
            "external_model_calls": 0,
        }
        return {
            "builder": builder,
            "consensus": blocked,
            "used": blocked,
            "entities": blocked,
            "used_entities": blocked,
            "communities": [],
            "community_summary": {**blocked, "generated": 0},
            "rubric_gold": blocked,
            "rubric_cycle": blocked,
        }
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
    store: KnowledgeGraphStore,
    lanes: Mapping[str, Any],
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
            {
                "role": str(route.get("role") or role),
                "provider": str(route.get("provider") or "unresolved"),
                "model": str(route.get("model") or "unresolved"),
                "location": str(route.get("location") or "invalid"),
                "model_sha256": str(status.get("model_sha256") or ""),
                "local_model_digest": str(status.get("local_model_digest") or ""),
            }
            for role, status in (
                ("knowledge.relation_extraction", lanes["builder"]),
                ("knowledge.community_summary", lanes["community_summary"]),
            )
            for route in (
                status.get("route_identity")
                if isinstance(status.get("route_identity"), dict)
                else {},
            )
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
    rollout_gates["runtime_routes_resolved"] = all(
        isinstance(identity, dict)
        and set(identity) == {"role", "provider", "model", "location"}
        and identity.get("role") == role
        and isinstance(identity.get("provider"), str)
        and bool(identity["provider"])
        and isinstance(identity.get("model"), str)
        and bool(identity["model"])
        and identity.get("location") in {"local", "remote"}
        and _is_sha256(status.get("model_sha256"))
        for role, status in (
            ("knowledge.relation_extraction", lanes["builder"]),
            ("knowledge.community_summary", lanes["community_summary"]),
        )
        for identity in (status.get("route_identity"),)
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
            model_manifest_sha256=sha256(artifacts["model_manifest"]),
        )
    else:
        rollout = advance_rollout(
            gates=rollout_gates,
            sample_count=canary_samples,
            promotion_file=promotion_file,
            manifest_sha256=str(evaluation.get("manifest_sha256") or ""),
            relation_snapshot_sha256=artifacts["relation_sha"],
            rubric_sha256=artifacts["rubric_sha"],
            model_manifest_sha256=sha256(artifacts["model_manifest"]),
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
    if lanes["builder"].get("status") == "blocked":
        return {
            "schema_version": 1,
            "status": "blocked",
            "mode": cfg.mode,
            "builder": lanes["builder"],
            "consensus": lanes["consensus"],
            "used_paths": lanes["used"],
            "used_entities": lanes["used_entities"],
            "entities": lanes["entities"],
            "communities": 0,
            "community_summary": lanes["community_summary"],
            "rubric": lanes["rubric_cycle"],
            "rubric_gold": lanes["rubric_gold"],
            "external_model_calls": 0,
        }
    artifacts = _evaluation_artifacts(
        root=root,
        store=store,
        lanes=lanes,
        busy=busy,
        dry_run=dry_run,
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
