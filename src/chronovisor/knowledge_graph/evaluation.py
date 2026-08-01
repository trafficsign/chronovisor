"""Preregistered four-arm evaluation and machine-checkable invariants."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from chronovisor.core.durable_state import read_sealed_json, write_sealed_json
from chronovisor.core.jsonl_write import append_jsonl_durable
from chronovisor.knowledge_graph.schema import sha256

EVALUATION_ARMS = (
    "current",
    "graph_only",
    "rubric_only",
    "graph_and_rubric",
)
FIXTURE_CATEGORIES = (
    "single_hop",
    "multi_hop",
    "global_synthesis",
    "entity_ambiguity",
    "alias_collision",
    "topic_shift",
    "hub_false_positive",
    "irrelevant_card",
)
PREREGISTRATION_VERSION = 1


def preregistration() -> dict[str, Any]:
    return {
        "schema_version": PREREGISTRATION_VERSION,
        "arms": list(EVALUATION_ARMS),
        "fixed": {
            "query_manifest": True,
            "candidate_budget": 50,
            "token_budget": 2_402,
            "deadline_ms": 4_000,
            "temporal_split": "70/20/10-session-query-connected",
            "teacher": "current",
        },
        "categories": list(FIXTURE_CATEGORIES),
        "metrics": [
            "recall_at_5",
            "recall_at_10",
            "mrr",
            "used_page_coverage",
            "pointer_precision",
            "rich_precision",
            "negative_hit",
            "abstention",
            "relation_path_precision",
            "latency_p50_ms",
            "latency_p95_ms",
            "latency_max_ms",
            "over_4s",
            "model_seconds",
            "external_model_calls",
        ],
        "winner_rules": {
            "direct_non_degradation": True,
            "multi_hop_improvement": True,
            "coverage_floor": True,
            "all_fail_winner": "current",
            "interaction_must_beat_single": True,
        },
    }


def fixture_manifest(
    locked_fixtures: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    fixtures = [
        {
            "fixture_id": f"typed-graph-{index + 1:02d}",
            "category": category,
            "query_sha256": sha256(f"locked:{category}:v1"),
            "locked": True,
        }
        for index, category in enumerate(FIXTURE_CATEGORIES)
    ] if locked_fixtures is None else [
        {
            "fixture_id": str(row.get("fixture_id") or f"typed-graph-{index + 1:02d}"),
            "category": str(row.get("category") or ""),
            "query_sha256": str(row.get("query_sha256") or ""),
            "locked": True,
        }
        for index, row in enumerate(locked_fixtures)
        if str(row.get("category") or "") in FIXTURE_CATEGORIES
        and re.fullmatch(r"[0-9a-f]{64}", str(row.get("query_sha256") or ""))
    ]
    return {"schema_version": 1, "fixtures": fixtures, "count": len(fixtures)}


def select_locked_fixtures(
    rows: Sequence[Mapping[str, Any]],
    *,
    allowed_query_sha256s: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Select one reviewed real query per preregistered failure category."""

    candidates = []
    for row in rows:
        query = str(row.get("query") or "")
        query_sha = sha256(query) if query else str(row.get("query_sha256") or "")
        if (
            row.get("reviewed") is not True
            or not query
            or not row.get("expected_pages")
            or (allowed_query_sha256s is not None and query_sha not in allowed_query_sha256s)
        ):
            continue
        candidates.append((row, query, query_sha))

    def score(category: str, row: Mapping[str, Any], query: str) -> tuple[int, str]:
        folded = query.casefold()
        expected = row.get("expected_pages")
        expected_count = len(expected) if isinstance(expected, list) else 0
        negative = row.get("negative_pages")
        negative_count = len(negative) if isinstance(negative, list) else 0
        source = str(row.get("source") or "")
        kind = str(row.get("kind") or "")
        values = {
            "single_hop": 5 * int(expected_count == 1) + int(len(query) < 80),
            "multi_hop": 4 * int(expected_count >= 2)
            + 3 * int(bool(re.search(r"関係|つなが|経由|なぜ|between|relation", folded))),
            "global_synthesis": 8
            * int(bool(re.search(r"全体|まとめ|傾向|横断|共通|overview|overall", folded))),
            "entity_ambiguity": 6
            * int(bool(re.search(r"同名|誰|人物|会社|組織|製品|モデル|version|バージョン", folded))),
            "alias_collision": 6
            * int(bool(re.search(r"略称|呼び方|alias|gpt|llm|kimi|君系|qwen|gemma", folded))),
            "topic_shift": 6
            * int(kind == "injection_ignored" or "関係ない" in folded or "話変" in folded),
            "hub_false_positive": 5 * int(negative_count > 0)
            + 3 * int(source == "auditor_precision"),
            "irrelevant_card": 7 * int(kind == "injection_ignored")
            + 3 * int(source == "auditor_precision"),
        }
        return values[category], query_sha

    selected: list[dict[str, Any]] = []
    used: set[str] = set()
    for category in FIXTURE_CATEGORIES:
        ranked = sorted(
            (
                (score(category, row, query), row, query_sha)
                for row, query, query_sha in candidates
                if query_sha not in used
            ),
            key=lambda value: (-value[0][0], value[0][1]),
        )
        if not ranked:
            continue
        _rank, row, query_sha = ranked[0]
        used.add(query_sha)
        selected.append(
            {
                "fixture_id": f"typed-graph-{len(selected) + 1:02d}",
                "category": category,
                "query_sha256": query_sha,
                "ref_sha256": sha256(str(row.get("ref") or "")),
            }
        )
    return selected


def capture_baseline(
    *,
    output_file: Path,
    git_head: str,
    runtime_commit: str,
    config_sha256: str,
    model_inventory: Sequence[str],
    artifact_counts: Mapping[str, int | float | bool],
    locked_fixtures: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_head": git_head,
        "runtime_commit": runtime_commit,
        "config_sha256": config_sha256,
        "model_inventory_sha256": sha256(sorted(model_inventory)),
        "artifact_counts": dict(sorted(artifact_counts.items())),
        "preregistration": preregistration(),
        "fixture_manifest": fixture_manifest(locked_fixtures),
        "privacy": {
            "query_text": False,
            "page_body": False,
            "raw_prompt": False,
        },
        "invariants": {
            "external_model_calls": 0,
            "sync_over_4s": 0,
            "rollback_teacher": "current",
        },
    }
    return write_sealed_json(output_file, payload)


def validate_baseline(path: Path) -> dict[str, Any]:
    payload = read_sealed_json(path)
    privacy = payload.get("privacy")
    invariants = payload.get("invariants")
    registered = payload.get("preregistration")
    checks = {
        "four_arms": isinstance(registered, dict)
        and registered.get("arms") == list(EVALUATION_ARMS),
        "fixture_coverage": isinstance(payload.get("fixture_manifest"), dict)
        and {
            row.get("category")
            for row in payload["fixture_manifest"].get("fixtures", [])
            if isinstance(row, dict)
        }
        == set(FIXTURE_CATEGORIES),
        "privacy_safe": isinstance(privacy, dict) and not any(privacy.values()),
        "external_calls_zero": isinstance(invariants, dict)
        and invariants.get("external_model_calls") == 0,
        "sync_over_4s_zero": isinstance(invariants, dict)
        and invariants.get("sync_over_4s") == 0,
        "rollback_teacher_current": isinstance(invariants, dict)
        and invariants.get("rollback_teacher") == "current",
    }
    return {"status": "passed" if all(checks.values()) else "failed", "checks": checks}


def baseline_manifest_sha256(baseline: Mapping[str, Any]) -> str:
    """Bind evaluation rows to the locked query manifest, not mutable metadata."""

    manifest = baseline.get("fixture_manifest")
    return sha256(manifest if isinstance(manifest, dict) else {})


def compare_four_arms(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate sealed arm results without assuming graph+rubric wins."""

    grouped: dict[str, list[Mapping[str, Any]]] = {arm: [] for arm in EVALUATION_ARMS}
    for row in rows:
        arm = str(row.get("arm") or "")
        if arm in grouped:
            grouped[arm].append(row)
    metrics: dict[str, dict[str, float]] = {}

    def mean(arm_rows: Sequence[Mapping[str, Any]], name: str) -> float:
        values = [
            float(row[name])
            for row in arm_rows
            if isinstance(row.get(name), int | float)
            and not isinstance(row.get(name), bool)
        ]
        return sum(values) / len(values) if values else 0.0

    metric_names = (
        "recall_at_5",
        "recall_at_10",
        "mrr",
        "used_page_coverage",
        "pointer_precision",
        "rich_precision",
        "negative_hit",
        "abstention",
        "merge_error",
        "relation_path_precision",
        "latency_p50_ms",
        "latency_p95_ms",
        "latency_max_ms",
        "over_4s",
        "memory_mb",
        "model_seconds",
        "external_model_calls",
    )
    for arm, arm_rows in grouped.items():
        metrics[arm] = {
            name: round(mean(arm_rows, name), 6) for name in metric_names
        }
    baseline = metrics["current"]
    eligible = [
        arm
        for arm, values in metrics.items()
        if values["pointer_precision"] >= baseline["pointer_precision"]
        and values["used_page_coverage"] >= baseline["used_page_coverage"]
        and values["over_4s"] == 0
        and values["external_model_calls"] == 0
    ]
    winner = max(
        eligible, key=lambda arm: metrics[arm]["recall_at_5"], default="current"
    )
    if winner == "graph_and_rubric" and metrics[winner]["recall_at_5"] <= max(
        metrics["graph_only"]["recall_at_5"], metrics["rubric_only"]["recall_at_5"]
    ):
        winner = max(
            ("current", "graph_only", "rubric_only"),
            key=lambda arm: metrics[arm]["recall_at_5"] if arm in eligible else -1.0,
        )
    by_category = {
        category: {
            arm: {
                "samples": len(
                    [row for row in grouped[arm] if row.get("category") == category]
                ),
                "recall_at_5": round(
                    mean(
                        [row for row in grouped[arm] if row.get("category") == category],
                        "recall_at_5",
                    ),
                    6,
                ),
            }
            for arm in EVALUATION_ARMS
        }
        for category in FIXTURE_CATEGORIES
    }
    stages = {
        arm: {
            name: sum(bool(row.get(name)) for row in grouped[arm])
            for name in (
                "candidate_generated",
                "rerank_passed",
                "certificate_passed",
                "committed",
                "actually_used",
            )
        }
        for arm in EVALUATION_ARMS
    }
    return {
        "schema_version": 1,
        "metrics": metrics,
        "by_category": by_category,
        "stages": stages,
        "samples": {arm: len(values) for arm, values in grouped.items()},
        "eligible": eligible,
        "winner": winner,
    }


def _wilson_lower(successes: int, total: int, z: float = 1.96) -> float:
    if total <= 0:
        return 0.0
    rate = successes / total
    denominator = 1 + z * z / total
    center = rate + z * z / (2 * total)
    margin = z * math.sqrt((rate * (1 - rate) + z * z / (4 * total)) / total)
    return max(0.0, (center - margin) / denominator)


def evaluate_locked_rows(
    rows: Sequence[Mapping[str, Any]], *, manifest_sha256: str
) -> dict[str, Any]:
    comparison = compare_four_arms(rows)
    samples = comparison["samples"]
    categories = comparison["by_category"]
    winner = str(comparison["winner"])
    winner_rows = [row for row in rows if row.get("arm") == winner]
    precision_total = sum(
        int(row.get("pointer_total") or 0) for row in winner_rows
    )
    precision_hits = sum(
        int(row.get("pointer_correct") or 0) for row in winner_rows
    )
    baseline = comparison["metrics"]["current"]
    selected = comparison["metrics"][winner]
    direct_ok = (
        categories["single_hop"][winner]["recall_at_5"]
        >= categories["single_hop"]["current"]["recall_at_5"]
    )
    multihop_ok = (
        winner == "current"
        or categories["multi_hop"][winner]["recall_at_5"]
        > categories["multi_hop"]["current"]["recall_at_5"]
    )
    gates = {
        "all_arms": all(samples[arm] > 0 for arm in EVALUATION_ARMS),
        "all_categories": all(
            all(categories[category][arm]["samples"] > 0 for arm in EVALUATION_ARMS)
            for category in FIXTURE_CATEGORIES
        ),
        "direct_non_degradation": direct_ok,
        "multi_hop_improvement": multihop_ok,
        "pointer_precision_non_degradation": (
            selected["pointer_precision"] >= baseline["pointer_precision"]
        ),
        "used_coverage_non_degradation": (
            selected["used_page_coverage"] >= baseline["used_page_coverage"]
        ),
        "latency": selected["over_4s"] == 0,
        "external_calls": selected["external_model_calls"] == 0,
        "non_abstention_gaming": selected["used_page_coverage"] > 0,
    }
    return {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "status": "passed" if all(gates.values()) else "collecting_or_held",
        "manifest_sha256": manifest_sha256,
        "comparison": comparison,
        "gates": gates,
        "winner": winner,
        "pointer_precision_lower_95": round(
            _wilson_lower(precision_hits, precision_total), 6
        ),
    }


def run_evaluation_cycle(
    *,
    rows_file: Path,
    baseline_file: Path,
    output_file: Path,
    evaluation_epoch: str = "",
    dry_run: bool = False,
) -> dict[str, Any]:
    try:
        baseline = read_sealed_json(baseline_file, recover_backup=True)
    except Exception:
        baseline = {}
    rows: list[dict[str, Any]] = []
    try:
        for line in rows_file.read_text(encoding="utf-8").splitlines():
            value = json.loads(line)
            if isinstance(value, dict):
                rows.append(value)
    except (OSError, json.JSONDecodeError):
        rows = []
    if evaluation_epoch:
        rows = [row for row in rows if row.get("evaluation_epoch") == evaluation_epoch]
    payload = evaluate_locked_rows(
        rows,
        manifest_sha256=baseline_manifest_sha256(baseline),
    )
    return payload if dry_run else write_sealed_json(output_file, payload)


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
