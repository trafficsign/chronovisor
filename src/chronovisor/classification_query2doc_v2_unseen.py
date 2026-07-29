"""Preregistered group-separated unseen gate for Query2doc v2."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from chronovisor import ollama
from chronovisor.classification import ClassificationError, default_udc_package
from chronovisor.classification_fixture_set import read_jsonl, sha256_file
from chronovisor.classification_query2doc_v2 import (
    CHANNEL_LIMIT,
    CHANNEL_QUOTAS,
    CHANNEL_WEIGHTS,
    FUSED_LIMIT,
    RRF_K,
    evaluate_fixture,
    retrieval_contract,
    retrieval_contract_sha256,
)
from chronovisor.classification_query_worker_v2 import (
    QUERY_POLICY,
    QUERY_PROMPT_SHA256,
)
from chronovisor.durable_state import read_sealed_json, write_sealed_json
from chronovisor.runtime_config import load_decision_router_config
from chronovisor.store import CHRONOVISOR_ROOT

SELECTION_SCHEMA = "chronovisor.classification-query2doc-v2-3-selection.v1"
MANUAL_GOLD_SCHEMA = "chronovisor.classification-query2doc-v2-3-manual-gold.v1"
FIXTURE_SCHEMA = "chronovisor.classification-query2doc-v2-3-fixture.v1"
PREREGISTRATION_SCHEMA = (
    "chronovisor.classification-query2doc-v2-3-preregistration.v1"
)
EVALUATION_SCHEMA = "chronovisor.classification-query2doc-v2-3-unseen-evaluation.v1"
MANIFEST_SCHEMA = "chronovisor.classification-query2doc-v2-3-unseen-manifest.v1"
STATE_SCHEMA = "chronovisor.classification-query2doc-v2-3-unseen-state.v1"
SELECTION_SEED = "query2doc-v2.3-group-separated-unseen-v1"
SAMPLE_SIZE = 30
PASS_RATE = 0.8
PASS_HITS = 24
FIXTURE_EPOCH = "epoch-3-library-evidence-v1"


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def unseen_root(root: Path) -> Path:
    return root / "classification" / "query2doc-v2-3-unseen"


def _candidate_path(root: Path) -> Path:
    return (
        root
        / "classification"
        / "fixtures"
        / "epochs"
        / FIXTURE_EPOCH
        / "candidates.jsonl"
    )


def _adjudication_path(root: Path) -> Path:
    return (
        root
        / "classification"
        / "fixtures"
        / "epochs"
        / FIXTURE_EPOCH
        / "adjudication.jsonl"
    )


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


def _selection_key(row: Mapping[str, Any]) -> tuple[str, str]:
    uid = str(row.get("uid") or "")
    group_id = str(row.get("fixture_group_id") or "")
    digest = hashlib.sha256(
        f"{SELECTION_SEED}|{group_id}|{uid}".encode()
    ).hexdigest()
    return digest, uid


def _seen_uids(root: Path) -> set[str]:
    seen: set[str] = set()
    receipt_paths = (
        root / "classification" / "query2doc-pilot" / "evaluation.json",
        root / "classification" / "query2doc-unseen" / "selection.json",
        root / "classification" / "query2doc-v2-unseen" / "selection.json",
        root / "classification" / "query2doc-v2-2-unseen" / "selection.json",
    )
    for path in receipt_paths:
        receipt = read_sealed_json(path)
        seen.update(
            str(case.get("uid") or "")
            for case in receipt.get("cases") or []
            if isinstance(case, Mapping) and str(case.get("uid") or "")
        )
    seen.update(
        str(row.get("uid") or "")
        for row in read_jsonl(_adjudication_path(root))
        if str(row.get("uid") or "")
    )
    return seen


def select_unseen_rows(
    root: Path,
    *,
    sample_size: int = SAMPLE_SIZE,
) -> list[dict[str, Any]]:
    """Select one row per provenance group, disjoint from every design fixture."""

    rows = read_jsonl(_candidate_path(root))
    by_uid = {
        str(row.get("uid") or ""): row
        for row in rows
        if str(row.get("uid") or "")
    }
    seen_uids = _seen_uids(root)
    seen_groups = {
        str(by_uid[uid].get("fixture_group_id") or "")
        for uid in seen_uids
        if uid in by_uid and str(by_uid[uid].get("fixture_group_id") or "")
    }
    eligible = []
    for row in rows:
        uid = str(row.get("uid") or "")
        source_sha256 = str(row.get("source_sha256") or "")
        group_id = str(row.get("fixture_group_id") or "")
        if (
            not uid
            or uid in seen_uids
            or not source_sha256
            or not group_id
            or group_id in seen_groups
            or row.get("lifecycle") != "active"
            or str(row.get("sensitivity") or "normal") != "normal"
        ):
            continue
        eligible.append(dict(row))
    eligible.sort(key=_selection_key)
    selected = []
    selected_groups: set[str] = set()
    selected_hashes: set[str] = set()
    for row in eligible:
        group_id = str(row["fixture_group_id"])
        source_sha256 = str(row["source_sha256"])
        if group_id in selected_groups or source_sha256 in selected_hashes:
            continue
        selected.append(row)
        selected_groups.add(group_id)
        selected_hashes.add(source_sha256)
        if len(selected) >= sample_size:
            break
    if len(selected) < sample_size:
        raise ClassificationError(
            f"only {len(selected)} group-separated rows are eligible; "
            f"need {sample_size}"
        )
    return selected


def prepare_selection(root: Path) -> dict[str, Any]:
    output_root = unseen_root(root)
    evaluation_path = output_root / "evaluation.json"
    if evaluation_path.exists():
        raise ClassificationError("query2doc v2.3 unseen evaluation is already sealed")
    selection_path = output_root / "selection.json"
    if selection_path.exists():
        return read_sealed_json(selection_path)
    rows = select_unseen_rows(root)
    selection = {
        "schema": SELECTION_SCHEMA,
        "created_at": _now(),
        "selection_seed": SELECTION_SEED,
        "source_candidate_path": str(_candidate_path(root)),
        "source_candidate_sha256": sha256_file(_candidate_path(root)),
        "excluded_uid_count": len(_seen_uids(root)),
        "group_disjoint_from_prior_design_data": True,
        "one_case_per_fixture_group": True,
        "case_count": len(rows),
        "cases": [
            {
                "position": position,
                "uid": str(row["uid"]),
                "source_sha256": str(row["source_sha256"]),
                "fixture_group_id": str(row["fixture_group_id"]),
                "fixture_group_basis": str(row.get("fixture_group_basis") or ""),
                "title": str(row.get("title") or ""),
            }
            for position, row in enumerate(rows, start=1)
        ],
    }
    write_sealed_json(selection_path, selection, backup=True)
    _write_state(
        root,
        status="prepared",
        stage="v2.3-unseen-selection-sealed",
        case_count=len(rows),
        generated_count=0,
        evaluated_count=0,
        classification_judge_calls=0,
        page_mutations=0,
    )
    return read_sealed_json(selection_path)


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
        raise ClassificationError("query2doc v2.3 unseen evaluation is already sealed")
    if any((output_root / "queries").glob("*.json")):
        raise ClassificationError("query artifacts exist before preregistration")
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
        if str(source.get("source_sha256") or "") != str(
            selected.get("source_sha256") or ""
        ):
            raise ClassificationError("selected source changed before lock")
        if str(source.get("fixture_group_id") or "") != str(
            selected.get("fixture_group_id") or ""
        ):
            raise ClassificationError("selected provenance group changed before lock")
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
                "fixture_group_id": str(source.get("fixture_group_id") or ""),
                "title": str(source.get("title") or ""),
                "summary": str(source.get("summary") or ""),
                "excerpt": str(source.get("excerpt") or ""),
                "tags": list(source.get("tags") or []),
                "raw_keywords": list(source.get("raw_keywords") or []),
                "expected_primary_notations": expected,
                "gold_rationale": str(gold.get("rationale") or ""),
                "gold_ambiguity": str(gold.get("ambiguity") or ""),
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
        "group_disjoint_from_prior_design_data": True,
        "case_count": len(fixture_cases),
        "cases": fixture_cases,
    }
    write_sealed_json(output_root / "fixture.json", fixture, backup=True)
    config = load_decision_router_config()
    model = config.primary_model
    model_digest = ollama.model_digests([model]).get(model, "")
    if not model_digest:
        raise ClassificationError(f"query2doc v2 model is not installed: {model}")
    preregistration = {
        "schema": PREREGISTRATION_SCHEMA,
        "locked_at": _now(),
        "fixture_path": str(output_root / "fixture.json"),
        "fixture_sha256": sha256_file(output_root / "fixture.json"),
        "case_count": SAMPLE_SIZE,
        "model": model,
        "model_digest": model_digest,
        "prompt_sha256": QUERY_PROMPT_SHA256,
        "retrieval_contract": retrieval_contract(),
        "retrieval_contract_sha256": retrieval_contract_sha256(),
        "input_contract": {
            "included": list(QUERY_POLICY["input_fields"]),
            "forbidden": list(QUERY_POLICY["forbidden_input_fields"]),
            "candidates_exposed_to_model": False,
            "expected_notations_exposed_to_model": False,
        },
        "channel_limit": CHANNEL_LIMIT,
        "fused_limit": FUSED_LIMIT,
        "fusion": {
            "algorithm": (
                "coverage-first-reserved-union-with-weighted-rrf-backfill"
            ),
            "k": RRF_K,
            "channel_quotas": dict(CHANNEL_QUOTAS),
            "channel_weights": dict(CHANNEL_WEIGHTS),
            "broad_and_role_views_from_one_model_call": True,
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
    write_sealed_json(
        output_root / "preregistration.json",
        preregistration,
        backup=True,
    )
    _write_state(
        root,
        status="locked",
        stage="v2.3-unseen-preregistration-locked",
        case_count=SAMPLE_SIZE,
        minimum_fused_hits=PASS_HITS,
        model=model,
        generated_count=0,
        evaluated_count=0,
        classification_judge_calls=0,
        page_mutations=0,
    )
    return read_sealed_json(output_root / "preregistration.json")


def _validate_preregistration(root: Path) -> dict[str, Any]:
    output_root = unseen_root(root)
    preregistration = read_sealed_json(output_root / "preregistration.json")
    if preregistration.get("schema") != PREREGISTRATION_SCHEMA:
        raise ClassificationError("query2doc v2 preregistration schema mismatch")
    if preregistration.get("fixture_sha256") != sha256_file(
        output_root / "fixture.json"
    ):
        raise ClassificationError("query2doc v2 fixture changed after lock")
    if preregistration.get("prompt_sha256") != QUERY_PROMPT_SHA256:
        raise ClassificationError("query2doc v2 prompt changed after lock")
    if (
        preregistration.get("retrieval_contract_sha256")
        != retrieval_contract_sha256()
    ):
        raise ClassificationError("query2doc v2 retrieval contract changed after lock")
    config = load_decision_router_config()
    model = config.primary_model
    digest = ollama.model_digests([model]).get(model, "")
    if (
        preregistration.get("model") != model
        or preregistration.get("model_digest") != digest
    ):
        raise ClassificationError("query2doc v2 model changed after lock")
    return preregistration


def unseen_gate_passed(metrics: Mapping[str, Mapping[str, Any]]) -> bool:
    fused_hits = int((metrics.get("fused") or {}).get("hit_count") or 0)
    best_raw_hits = max(
        int((metrics.get("raw_lexical") or {}).get("hit_count") or 0),
        int((metrics.get("raw_dense") or {}).get("hit_count") or 0),
    )
    return fused_hits >= PASS_HITS and fused_hits >= best_raw_hits


def evaluate_unseen(root: Path) -> dict[str, Any]:
    output_root = unseen_root(root)
    evaluation_path = output_root / "evaluation.json"
    if evaluation_path.exists():
        raise ClassificationError("query2doc v2.3 unseen evaluation is already sealed")
    _validate_preregistration(root)
    fixture = read_sealed_json(output_root / "fixture.json")
    evaluation = evaluate_fixture(
        root,
        fixture=fixture,
        output_root=output_root,
        prior_cases=None,
        state_stage_prefix="unseen",
    )
    metrics = evaluation["metrics"]
    fused_hits = int((metrics.get("fused") or {}).get("hit_count") or 0)
    best_raw_hits = max(
        int((metrics.get("raw_lexical") or {}).get("hit_count") or 0),
        int((metrics.get("raw_dense") or {}).get("hit_count") or 0),
    )
    passed = unseen_gate_passed(metrics)
    decision = (
        "qualify-query2doc-v2.3-unseen-retrieval"
        if passed
        else "reject-query2doc-v2.3-unseen-retrieval"
    )
    receipt = {
        "schema": EVALUATION_SCHEMA,
        "preregistration_path": str(output_root / "preregistration.json"),
        "preregistration_sha256": sha256_file(
            output_root / "preregistration.json"
        ),
        "fixture_path": str(output_root / "fixture.json"),
        "fixture_sha256": sha256_file(output_root / "fixture.json"),
        **evaluation,
        "minimum_fused_hits": PASS_HITS,
        "minimum_fused_recall_at_12": PASS_RATE,
        "best_raw_hit_count": best_raw_hits,
        "decision": decision,
        "decision_trial_authorized": passed,
        "larger_corpus_evaluation_authorized": False,
        "classification_judge_calls": 0,
        "page_mutations": 0,
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
        "query_count": int(receipt["case_count"]),
        "model": receipt["model"],
        "model_digest": receipt["model_digest"],
        "prompt_sha256": receipt["prompt_sha256"],
        "retrieval_contract_sha256": receipt["retrieval_contract_sha256"],
        "classification_judge_calls": 0,
        "page_mutations": 0,
    }
    write_sealed_json(output_root / "manifest.json", manifest, backup=True)
    _write_state(
        root,
        status="qualified" if passed else "rejected",
        stage="v2.3-unseen-evaluation-complete",
        decision=decision,
        case_count=receipt["case_count"],
        fused_hit_count=fused_hits,
        minimum_fused_hits=PASS_HITS,
        best_raw_hit_count=best_raw_hits,
        decision_trial_authorized=passed,
        larger_corpus_evaluation_authorized=False,
        model_calls=receipt["model_calls"],
        classification_judge_calls=0,
        page_mutations=0,
    )
    return read_sealed_json(evaluation_path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=CHRONOVISOR_ROOT)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare")
    lock_parser = subparsers.add_parser("lock")
    lock_parser.add_argument("--manual-gold", type=Path, required=True)
    subparsers.add_parser("run")
    args = parser.parse_args(argv)
    root = args.root.expanduser()
    try:
        if args.command == "prepare":
            result = prepare_selection(root)
        elif args.command == "lock":
            result = lock_preregistration(root, args.manual_gold.expanduser())
        else:
            result = evaluate_unseen(root)
    except (
        ClassificationError,
        OSError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        _write_state(
            root,
            status="failed",
            stage=f"v2.3-unseen-{args.command}-failed",
            error=str(exc),
            classification_judge_calls=0,
            page_mutations=0,
        )
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
