"""Pre-registered unseen evaluation for candidate-blind query2doc retrieval."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from chronovisor.classification.classification_query_worker import (
    QUERY_POLICY,
    QUERY_PROMPT_SHA256,
)
from chronovisor.core import ollama
from chronovisor.core.durable_state import read_sealed_json, write_sealed_json
from chronovisor.core.runtime_config import load_decision_router_config
from chronovisor.core.store import CHRONOVISOR_ROOT
from chronovisor.core.timeutil import utc_iso_milliseconds as _now
from chronovisor.lab.classification_profile_pilot import (
    notation_matches,
    query_profile_index,
)
from chronovisor.lab.classification_query2doc_pilot import (
    CHANNEL_LIMIT,
    CHANNELS,
    FUSED_LIMIT,
    RRF_K,
    generate_query_artifact,
    query_text,
    reciprocal_rank_fusion,
)
from chronovisor.lab.harness import (
    LabHarness,
    aggregate_channel_metrics,
    require_contract,
    require_file_hashes,
)
from chronovisor.recall.classification import (
    ClassificationError,
    default_udc_package,
)
from chronovisor.recall.classification_engine import CandidateIndex
from chronovisor.recall.classification_fixture_set import (
    read_jsonl,
    sha256_file,
)

SELECTION_SCHEMA = "chronovisor.classification-query2doc-unseen-selection.v1"
MANUAL_GOLD_SCHEMA = "chronovisor.classification-query2doc-manual-gold.v1"
FIXTURE_SCHEMA = "chronovisor.classification-query2doc-unseen-fixture.v1"
PREREGISTRATION_SCHEMA = (
    "chronovisor.classification-query2doc-unseen-preregistration.v1"
)
EVALUATION_SCHEMA = "chronovisor.classification-query2doc-unseen-evaluation.v1"
MANIFEST_SCHEMA = "chronovisor.classification-query2doc-unseen-manifest.v1"
STATE_SCHEMA = "chronovisor.classification-query2doc-unseen-state.v1"
EXPERIMENT = "query2doc-unseen"
SELECTION_SEED = "query2doc-unseen-v1"
SAMPLE_SIZE = 30
PASS_RATE = 0.8
PASS_HITS = math.ceil(SAMPLE_SIZE * PASS_RATE)




def unseen_root(root: Path) -> Path:
    return LabHarness(root, EXPERIMENT).output_root


def _dev_path(root: Path) -> Path:
    return root / "classification" / "fixtures" / "classification-dev-200.jsonl"


def _epoch3_path(root: Path) -> Path:
    return (
        root
        / "classification"
        / "fixtures"
        / "epochs"
        / "epoch-3-library-evidence-v1"
        / "adjudication.jsonl"
    )


def _fixed_evaluation_path(root: Path) -> Path:
    return root / "classification" / "query2doc-pilot" / "evaluation.json"


def _write_state(
    root: Path,
    *,
    status: str,
    stage: str,
    **detail: Any,
) -> dict[str, Any]:
    return write_sealed_json(
        unseen_root(root) / "state.json",
        {
            "schema": STATE_SCHEMA,
            "status": status,
            "stage": stage,
            "updated_at": _now(),
            **detail,
        },
        backup=True,
    )


def _selection_key(row: Mapping[str, Any]) -> str:
    identity = (
        f"{SELECTION_SEED}\0{row.get('uid') or ''}\0"
        f"{row.get('source_sha256') or ''}"
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def select_unseen_rows(
    root: Path,
    *,
    sample_size: int = SAMPLE_SIZE,
) -> list[dict[str, Any]]:
    """Select deterministic dev rows never used by the fixed-ten design set."""

    fixed = read_sealed_json(_fixed_evaluation_path(root))
    seen = {
        str(case.get("uid") or "")
        for case in fixed.get("cases") or []
        if str(case.get("uid") or "")
    }
    seen.update(
        str(row.get("uid") or "")
        for row in read_jsonl(_epoch3_path(root))
        if str(row.get("uid") or "")
    )
    eligible = []
    source_hashes = set()
    for row in read_jsonl(_dev_path(root)):
        uid = str(row.get("uid") or "")
        source_sha256 = str(row.get("source_sha256") or "")
        if (
            not uid
            or uid in seen
            or not source_sha256
            or source_sha256 in source_hashes
            or row.get("adjudication_status") != "accepted"
            or row.get("gold_expected_status") != "proposed"
        ):
            continue
        source_hashes.add(source_sha256)
        eligible.append(dict(row))
    eligible.sort(key=_selection_key)
    if len(eligible) < sample_size:
        raise ClassificationError(
            f"only {len(eligible)} unseen rows are eligible; need {sample_size}"
        )
    return eligible[:sample_size]


def prepare_selection(root: Path) -> dict[str, Any]:
    output_root = unseen_root(root)
    if (output_root / "evaluation.json").exists():
        raise ClassificationError("unseen evaluation is already sealed")
    rows = select_unseen_rows(root)
    selection = {
        "schema": SELECTION_SCHEMA,
        "created_at": _now(),
        "selection_seed": SELECTION_SEED,
        "selection_algorithm": (
            "accepted proposed dev rows; exclude fixed-ten and epoch-3 early "
            "rows; deduplicate source_sha256; sort sha256(seed,uid,source); take 30"
        ),
        "sample_size": SAMPLE_SIZE,
        "source_dev_path": str(_dev_path(root)),
        "source_dev_sha256": sha256_file(_dev_path(root)),
        "excluded_fixed_evaluation_path": str(_fixed_evaluation_path(root)),
        "excluded_fixed_evaluation_sha256": sha256_file(
            _fixed_evaluation_path(root)
        ),
        "excluded_epoch3_path": str(_epoch3_path(root)),
        "excluded_epoch3_sha256": sha256_file(_epoch3_path(root)),
        "legacy_consensus_gold_used_for_scoring": False,
        "cases": [
            {
                "position": position,
                "uid": str(row.get("uid") or ""),
                "source_sha256": str(row.get("source_sha256") or ""),
                "title": str(row.get("title") or ""),
            }
            for position, row in enumerate(rows, start=1)
        ],
    }
    LabHarness(root, EXPERIMENT).seal_selection(selection)
    _write_state(
        root,
        status="prepared",
        stage="unseen-selection-sealed",
        case_count=SAMPLE_SIZE,
        generated_count=0,
        evaluated_count=0,
    )
    return read_sealed_json(output_root / "selection.json")


def _read_manual_gold(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ClassificationError(f"manual gold is unreadable: {exc}") from exc
    if not isinstance(value, dict) or value.get("schema") != MANUAL_GOLD_SCHEMA:
        raise ClassificationError("manual gold schema mismatch")
    return value


def lock_preregistration(root: Path, manual_gold_path: Path) -> dict[str, Any]:
    output_root = unseen_root(root)
    if (output_root / "evaluation.json").exists():
        raise ClassificationError("unseen evaluation is already sealed")
    selection = read_sealed_json(output_root / "selection.json")
    selected_rows = select_unseen_rows(root)
    selected_by_uid = {
        str(row.get("uid") or ""): row
        for row in selected_rows
    }
    manual = _read_manual_gold(manual_gold_path)
    gold_rows = manual.get("cases")
    if not isinstance(gold_rows, list):
        raise ClassificationError("manual gold cases are missing")
    gold_by_uid = {
        str(row.get("uid") or ""): row
        for row in gold_rows
        if isinstance(row, Mapping)
    }
    if set(gold_by_uid) != set(selected_by_uid):
        raise ClassificationError("manual gold UIDs do not match sealed selection")
    package = default_udc_package()
    fixture_cases = []
    for selected in selection.get("cases") or []:
        uid = str(selected.get("uid") or "")
        source = selected_by_uid[uid]
        gold = gold_by_uid[uid]
        expected = [
            str(value).strip()
            for value in gold.get("expected_primary_notations") or []
            if str(value).strip()
        ]
        if not expected:
            raise ClassificationError(f"manual gold has no expected notation: {uid}")
        unknown = [
            notation
            for notation in expected
            if package.by_notation(notation) is None
        ]
        if unknown:
            raise ClassificationError(
                f"manual gold has unknown UDC notation for {uid}: "
                + ", ".join(unknown)
            )
        fixture_cases.append(
            {
                "position": int(selected.get("position") or 0),
                "uid": uid,
                "source_sha256": str(source.get("source_sha256") or ""),
                "title": str(source.get("title") or ""),
                "summary": str(source.get("summary") or ""),
                "excerpt": str(source.get("excerpt") or ""),
                "tags": list(source.get("tags") or []),
                "raw_keywords": list(source.get("raw_keywords") or []),
                "expected_primary_notations": expected,
                "gold_rationale": str(gold.get("rationale") or ""),
                "gold_basis": "codex semantic review before query generation",
            }
        )
    fixture = {
        "schema": FIXTURE_SCHEMA,
        "locked_at": _now(),
        "selection_path": str(output_root / "selection.json"),
        "selection_sha256": sha256_file(output_root / "selection.json"),
        "manual_gold_path": str(manual_gold_path),
        "manual_gold_sha256": sha256_file(manual_gold_path),
        "manual_gold_reviewer": str(manual.get("reviewer") or ""),
        "manual_gold_completed_before_query_generation": True,
        "legacy_consensus_gold_used_for_scoring": False,
        "case_count": len(fixture_cases),
        "cases": fixture_cases,
    }
    write_sealed_json(output_root / "fixture.json", fixture, backup=True)
    config = load_decision_router_config()
    model = config.primary_model
    model_digest = ollama.model_digests([model]).get(model, "")
    if not model_digest:
        raise ClassificationError(f"query2doc model is not installed: {model}")
    preregistration = {
        "schema": PREREGISTRATION_SCHEMA,
        "locked_at": _now(),
        "fixture_path": str(output_root / "fixture.json"),
        "fixture_sha256": sha256_file(output_root / "fixture.json"),
        "case_count": SAMPLE_SIZE,
        "model": model,
        "model_digest": model_digest,
        "prompt_sha256": QUERY_PROMPT_SHA256,
        "input_contract": {
            "included": list(QUERY_POLICY["input_fields"]),
            "forbidden": list(QUERY_POLICY["forbidden_input_fields"]),
            "candidates_exposed_to_model": False,
            "expected_notations_exposed_to_model": False,
        },
        "channel_limit": CHANNEL_LIMIT,
        "fused_limit": FUSED_LIMIT,
        "fusion": {
            "algorithm": "equal-weight reciprocal-rank-fusion",
            "k": RRF_K,
            "channels": list(CHANNELS),
            "weights": {channel: 1 for channel in CHANNELS},
            "tuning_after_fixed_ten": False,
        },
        "gate": {
            "minimum_fused_hits": PASS_HITS,
            "minimum_fused_recall_at_12": PASS_RATE,
            "require_not_worse_than_best_raw_channel": True,
        },
        "successful_model_call_budget": SAMPLE_SIZE,
        "classification_judge_calls": 0,
        "page_mutations": 0,
    }
    LabHarness(root, EXPERIMENT).lock_preregistration(preregistration)
    _write_state(
        root,
        status="locked",
        stage="unseen-preregistration-locked",
        case_count=SAMPLE_SIZE,
        minimum_fused_hits=PASS_HITS,
        model=model,
        generated_count=0,
        evaluated_count=0,
    )
    return read_sealed_json(output_root / "preregistration.json")


def _matching_rank(
    rows: Sequence[Mapping[str, Any]],
    expected: Sequence[str],
) -> int | None:
    return next(
        (
            rank
            for rank, row in enumerate(rows, start=1)
            if notation_matches(str(row.get("notation") or ""), expected)
        ),
        None,
    )


def _metrics(
    cases: Sequence[Mapping[str, Any]],
    channel: str,
) -> dict[str, Any]:
    ranks = [
        int(case[f"{channel}_rank"])
        for case in cases
        if isinstance(case.get(f"{channel}_rank"), int)
    ]
    count = max(1, len(cases))
    return {
        "hit_count": len(ranks),
        "recall_at_12": len(ranks) / count,
        "mrr": sum(1 / rank for rank in ranks) / count,
    }


def unseen_gate_passed(metrics: Mapping[str, Mapping[str, Any]]) -> bool:
    fused_hits = int((metrics.get("fused") or {}).get("hit_count") or 0)
    best_raw_hits = max(
        int((metrics.get("raw_lexical") or {}).get("hit_count") or 0),
        int((metrics.get("raw_dense") or {}).get("hit_count") or 0),
    )
    return fused_hits >= PASS_HITS and fused_hits >= best_raw_hits


def _validate_preregistration(
    root: Path,
    preregistration: Mapping[str, Any],
    fixture: Mapping[str, Any],
) -> tuple[str, str]:
    harness = LabHarness(root, EXPERIMENT)
    require_contract(
        preregistration,
        schema=PREREGISTRATION_SCHEMA,
        exact={
            "prompt_sha256": QUERY_PROMPT_SHA256,
        },
        error_type=ClassificationError,
        message="unseen preregistration contract changed",
    )
    if int(preregistration.get("case_count") or 0) != SAMPLE_SIZE:
        raise ClassificationError("unseen case count is not pre-registered")
    require_contract(
        fixture,
        schema=FIXTURE_SCHEMA,
        error_type=ClassificationError,
        message="unseen fixture schema mismatch",
    )
    require_file_hashes(
        {harness.path("fixture.json"): str(preregistration.get("fixture_sha256") or "")},
        digest=sha256_file,
        error_type=ClassificationError,
        message="unseen fixture changed after preregistration",
    )
    config = load_decision_router_config()
    model = config.primary_model
    model_digest = ollama.model_digests([model]).get(model, "")
    if (
        model != preregistration.get("model")
        or not model_digest
        or model_digest != preregistration.get("model_digest")
    ):
        raise ClassificationError("query2doc model changed after preregistration")
    return model, model_digest


def evaluate_unseen(root: Path) -> dict[str, Any]:
    output_root = unseen_root(root)
    evaluation_path = output_root / "evaluation.json"
    if evaluation_path.is_file():
        return read_sealed_json(evaluation_path)
    preregistration = read_sealed_json(output_root / "preregistration.json")
    fixture = read_sealed_json(output_root / "fixture.json")
    model, model_digest = _validate_preregistration(
        root,
        preregistration,
        fixture,
    )
    config = load_decision_router_config()
    lexical = CandidateIndex(default_udc_package())
    cases = []
    total_model_calls = 0
    total_model_attempts = 0
    fixture_cases = list(fixture.get("cases") or [])
    for offset, source in enumerate(fixture_cases, start=1):
        uid = str(source.get("uid") or "")
        _write_state(
            root,
            status="running",
            stage="generating-unseen-candidate-blind-queries",
            case_count=SAMPLE_SIZE,
            generated_count=offset - 1,
            evaluated_count=offset - 1,
            model=model,
        )
        artifact_path = output_root / "queries" / f"{uid}.json"
        artifact = generate_query_artifact(
            root,
            source,
            model=model,
            model_digest=model_digest,
            keep_alive=config.primary_keep_alive,
            read_timeout_ms=config.read_timeout_ms,
            artifact_path=artifact_path,
        )
        total_model_calls += int(artifact.get("model_calls") or 0)
        total_model_attempts += int(artifact.get("model_attempts") or 0)
        abstract_query = query_text(artifact)
        query_page = {
            "title": abstract_query,
            "summary": "",
            "tags": [],
            "raw_keywords": [],
            "excerpt": "",
        }
        raw_page = {
            "title": str(source.get("title") or ""),
            "summary": str(source.get("summary") or ""),
            "tags": list(source.get("tags") or []),
            "raw_keywords": list(source.get("raw_keywords") or []),
            "excerpt": str(source.get("excerpt") or ""),
        }
        channel_rows = {
            "raw_lexical": lexical.candidates(raw_page, limit=CHANNEL_LIMIT),
            "raw_dense": query_profile_index(
                root,
                raw_page,
                limit=CHANNEL_LIMIT,
            ),
            "query2doc_lexical": lexical.candidates(
                query_page,
                limit=CHANNEL_LIMIT,
            ),
            "query2doc_dense": query_profile_index(
                root,
                query_page,
                limit=CHANNEL_LIMIT,
            ),
        }
        fused = reciprocal_rank_fusion(channel_rows)
        expected = [
            str(value)
            for value in source.get("expected_primary_notations") or []
            if str(value)
        ]
        case = {
            "position": int(source.get("position") or 0),
            "uid": uid,
            "title": str(source.get("title") or ""),
            "source_sha256": str(source.get("source_sha256") or ""),
            "expected_primary_notations": expected,
            "gold_rationale": str(source.get("gold_rationale") or ""),
            "query_artifact_path": str(artifact_path),
            "query_artifact_sha256": sha256_file(artifact_path),
            "query": artifact["query"],
            "fused_candidates": fused,
        }
        for channel, rows in channel_rows.items():
            case[f"{channel}_candidates"] = rows
            case[f"{channel}_rank"] = _matching_rank(rows, expected)
            case[f"{channel}_hit"] = case[f"{channel}_rank"] is not None
        case["fused_rank"] = _matching_rank(fused, expected)
        case["fused_hit"] = case["fused_rank"] is not None
        cases.append(case)
        _write_state(
            root,
            status="running",
            stage="retrieving-unseen-query2doc-channels",
            case_count=SAMPLE_SIZE,
            generated_count=offset,
            evaluated_count=offset,
            model=model,
        )
    metrics = aggregate_channel_metrics(cases, (*CHANNELS, "fused"), _metrics)
    fused_hits = int(metrics["fused"]["hit_count"])
    best_raw_hits = max(
        int(metrics["raw_lexical"]["hit_count"]),
        int(metrics["raw_dense"]["hit_count"]),
    )
    passed = unseen_gate_passed(metrics)
    decision = (
        "qualify-unseen-query2doc-retrieval"
        if passed
        else "reject-unseen-query2doc-retrieval"
    )
    receipt = {
        "schema": EVALUATION_SCHEMA,
        "evaluated_at": _now(),
        "preregistration_path": str(output_root / "preregistration.json"),
        "preregistration_sha256": sha256_file(
            output_root / "preregistration.json"
        ),
        "fixture_path": str(output_root / "fixture.json"),
        "fixture_sha256": sha256_file(output_root / "fixture.json"),
        "input_contract": dict(preregistration.get("input_contract") or {}),
        "model": model,
        "model_digest": model_digest,
        "prompt_sha256": QUERY_PROMPT_SHA256,
        "model_calls": total_model_calls,
        "model_attempts": total_model_attempts,
        "case_count": len(cases),
        "channel_limit": CHANNEL_LIMIT,
        "fused_limit": FUSED_LIMIT,
        "fusion": dict(preregistration.get("fusion") or {}),
        "match_policy": "exact target, target descendant, or explicit UDC range",
        "metrics": metrics,
        "minimum_fused_hits": PASS_HITS,
        "minimum_fused_recall_at_12": PASS_RATE,
        "best_raw_hit_count": best_raw_hits,
        "decision": decision,
        "decision_trial_authorized": passed,
        "larger_corpus_evaluation_authorized": False,
        "classification_judge_calls": 0,
        "page_mutations": 0,
        "cases": cases,
    }
    write_sealed_json(evaluation_path, receipt, backup=True)
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "created_at": _now(),
        "evaluation_path": str(evaluation_path),
        "evaluation_sha256": sha256_file(evaluation_path),
        "preregistration_path": str(output_root / "preregistration.json"),
        "preregistration_sha256": sha256_file(
            output_root / "preregistration.json"
        ),
        "query_count": len(cases),
        "model": model,
        "model_digest": model_digest,
        "prompt_sha256": QUERY_PROMPT_SHA256,
        "query_artifacts_separate_from_fixed_ten": True,
        "classification_judge_calls": 0,
        "page_mutations": 0,
    }
    write_sealed_json(output_root / "manifest.json", manifest, backup=True)
    _write_state(
        root,
        status="qualified" if passed else "rejected",
        stage="unseen-query2doc-complete",
        decision=decision,
        case_count=len(cases),
        fused_hit_count=fused_hits,
        minimum_fused_hits=PASS_HITS,
        best_raw_hit_count=best_raw_hits,
        decision_trial_authorized=passed,
        larger_corpus_evaluation_authorized=False,
        model_calls=total_model_calls,
        classification_judge_calls=0,
        page_mutations=0,
    )
    return read_sealed_json(evaluation_path)


def run_unseen(root: Path) -> dict[str, Any]:
    try:
        return evaluate_unseen(root)
    except Exception as exc:
        _write_state(
            root,
            status="failed",
            stage="unseen-query2doc-failed",
            error=f"{type(exc).__name__}: {exc}",
        )
        raise


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Prepare, lock, or run the unseen query2doc retrieval gate"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--lock", type=Path, metavar="MANUAL_GOLD_JSON")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--root", type=Path, default=CHRONOVISOR_ROOT)
    arguments = parser.parse_args(argv)
    if arguments.prepare:
        result = prepare_selection(arguments.root)
    elif arguments.lock:
        result = lock_preregistration(arguments.root, arguments.lock)
    else:
        result = run_unseen(arguments.root)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
