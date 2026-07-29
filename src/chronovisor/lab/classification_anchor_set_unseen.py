"""Preregistered unseen gate for the conservative CVO anchor-set contract."""

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
from chronovisor.classification import ClassificationError
from chronovisor.classification_anchor import (
    UNRESOLVED_ANCHOR_ID,
    AnchorSet,
    default_anchor_gold_path,
    default_anchor_set_path,
    load_anchor_set,
)
from chronovisor.lab.classification_anchor_dev import run_case
from chronovisor.lab.classification_anchor_second_auditor import (
    AUDIT_SCHEMA,
)
from chronovisor.lab.classification_anchor_second_auditor import (
    PROMPT_SHA256 as AUDITOR_PROMPT_SHA256,
)
from chronovisor.lab.classification_anchor_second_dev import (
    _auditor_payload,
    _call_auditor,
)
from chronovisor.lab.classification_anchor_set_dev import (
    default_dev_gold_path,
    score_anchor_set,
    summarize_metrics,
)
from chronovisor.lab.harness import (
    LabHarness,
    require_contract,
    require_file_hashes,
)
from chronovisor.classification_anchor_worker import (
    PROMPT_SHA256 as CORE_PROMPT_SHA256,
)
from chronovisor.lab.classification_fixture_set import read_jsonl, sha256_file
from chronovisor.durable_state import read_sealed_json, write_sealed_json
from chronovisor.runtime_config import load_decision_router_config
from chronovisor.store import CHRONOVISOR_ROOT

SELECTION_SCHEMA = "chronovisor.classification-anchor-set-unseen-selection.v1"
GOLD_SCHEMA = "chronovisor.classification-anchor-set-unseen-gold.v1"
FIXTURE_SCHEMA = "chronovisor.classification-anchor-set-unseen-fixture.v1"
PREREGISTRATION_SCHEMA = (
    "chronovisor.classification-anchor-set-unseen-preregistration.v1"
)
EVALUATION_SCHEMA = "chronovisor.classification-anchor-set-unseen-evaluation.v1"
STATE_SCHEMA = "chronovisor.classification-anchor-set-unseen-state.v1"
EXPERIMENT = "cvo-anchor-set-v1-unseen40"
SELECTION_SEED = "cvo-anchor-set-v1-group-separated-unseen40-v1"
FIXTURE_EPOCH = "epoch-3-library-evidence-v1"
OUTPUT_CONTRACT_EPOCH = "cvo-anchor-set-v1"
SAMPLE_SIZE = 40
MINIMUM_EXACT_SETS = 36
MINIMUM_SEMANTIC_COVERAGE = 38
MAXIMUM_EXCESS_RATE = 0.10
MAXIMUM_MISSING_RATE = 0.10
MAXIMUM_DUAL_RATE = 0.40
MAXIMUM_HOLDS = 2
MAXIMUM_MAJOR_ERRORS = 0


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def unseen_root(root: Path) -> Path:
    return LabHarness(root, EXPERIMENT).output_root


def default_unseen_gold_path() -> Path:
    return (
        Path(__file__).parent
        / "data"
        / "cvo-anchor-set-unseen-gold-v1.json"
    )


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


def _write_state(root: Path, *, status: str, stage: str, **detail: Any) -> None:
    write_sealed_json(
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


def _selection_key(
    row: Mapping[str, Any],
    *,
    selection_seed: str = SELECTION_SEED,
) -> tuple[str, str]:
    uid = str(row.get("uid") or "")
    group_id = str(row.get("fixture_group_id") or "")
    return (
        hashlib.sha256(
            f"{selection_seed}|{group_id}|{uid}".encode()
        ).hexdigest(),
        uid,
    )


def _receipt_uids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    receipt = read_sealed_json(path)
    return {
        str(case.get("uid") or "")
        for case in receipt.get("cases") or []
        if isinstance(case, Mapping) and str(case.get("uid") or "")
    }


def _gold_uids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        return set()
    return {
        str(case.get("uid") or "")
        for case in payload.get("cases") or []
        if isinstance(case, Mapping) and str(case.get("uid") or "")
    }


def _seen_uids(
    root: Path,
    *,
    extra_receipt_paths: Sequence[Path] = (),
    extra_gold_paths: Sequence[Path] = (),
) -> set[str]:
    receipt_paths = (
        root / "classification" / "query2doc-pilot" / "evaluation.json",
        root / "classification" / "query2doc-unseen" / "selection.json",
        root / "classification" / "query2doc-v2-unseen" / "selection.json",
        root / "classification" / "query2doc-v2-2-unseen" / "selection.json",
        root / "classification" / "query2doc-v2-3-unseen" / "selection.json",
        root / "classification" / "cvo-anchor-v0-unseen30" / "selection.json",
    )
    seen: set[str] = set()
    for path in (*receipt_paths, *extra_receipt_paths):
        seen.update(_receipt_uids(path))
    adjudication = _adjudication_path(root)
    if adjudication.is_file():
        seen.update(
            str(row.get("uid") or "")
            for row in read_jsonl(adjudication)
            if str(row.get("uid") or "")
        )
    seen.update(_gold_uids(default_anchor_gold_path()))
    seen.update(_gold_uids(default_dev_gold_path()))
    for path in extra_gold_paths:
        seen.update(_gold_uids(path))
    return seen


def select_unseen_rows(
    root: Path,
    *,
    sample_size: int = SAMPLE_SIZE,
    selection_seed: str = SELECTION_SEED,
    extra_receipt_paths: Sequence[Path] = (),
    extra_gold_paths: Sequence[Path] = (),
) -> list[dict[str, Any]]:
    rows = read_jsonl(_candidate_path(root))
    by_uid = {
        str(row.get("uid") or ""): row
        for row in rows
        if str(row.get("uid") or "")
    }
    seen_uids = _seen_uids(
        root,
        extra_receipt_paths=extra_receipt_paths,
        extra_gold_paths=extra_gold_paths,
    )
    seen_groups = {
        str(by_uid[uid].get("fixture_group_id") or "")
        for uid in seen_uids
        if uid in by_uid and str(by_uid[uid].get("fixture_group_id") or "")
    }
    eligible = []
    for row in rows:
        uid = str(row.get("uid") or "")
        group_id = str(row.get("fixture_group_id") or "")
        if (
            not uid
            or uid in seen_uids
            or not str(row.get("source_sha256") or "")
            or not group_id
            or group_id in seen_groups
            or row.get("lifecycle") != "active"
            or str(row.get("sensitivity") or "normal") != "normal"
        ):
            continue
        eligible.append(dict(row))
    eligible.sort(
        key=lambda row: _selection_key(row, selection_seed=selection_seed)
    )
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
    if len(selected) != sample_size:
        raise ClassificationError(
            f"only {len(selected)} group-separated CVO rows are eligible; "
            f"need {sample_size}"
        )
    return selected


def prepare_selection(root: Path) -> dict[str, Any]:
    destination = unseen_root(root)
    selection_path = destination / "selection.json"
    if (destination / "evaluation.json").is_file():
        raise ClassificationError("CVO anchor-set unseen evaluation is already sealed")
    if selection_path.is_file():
        return read_sealed_json(selection_path)
    rows = select_unseen_rows(root)
    selection = {
        "schema": SELECTION_SCHEMA,
        "created_at": _now(),
        "selection_seed": SELECTION_SEED,
        "source_candidate_path": str(_candidate_path(root)),
        "source_candidate_sha256": sha256_file(_candidate_path(root)),
        "excluded_uid_count": len(_seen_uids(root)),
        "group_disjoint_from_all_opened_design_data": True,
        "one_case_per_fixture_group": True,
        "unique_source_hashes": True,
        "model_calls_before_gold_lock": 0,
        "case_count": len(rows),
        "cases": [
            {
                "position": position,
                "uid": str(row["uid"]),
                "source_sha256": str(row["source_sha256"]),
                "fixture_group_id": str(row["fixture_group_id"]),
                "fixture_group_basis": str(
                    row.get("fixture_group_basis") or ""
                ),
                "title": str(row.get("title") or ""),
                "summary": str(row.get("summary") or ""),
                "excerpt": str(row.get("excerpt") or ""),
                "tags": list(row.get("tags") or []),
                "raw_keywords": list(row.get("raw_keywords") or []),
            }
            for position, row in enumerate(rows, start=1)
        ],
    }
    LabHarness(root, EXPERIMENT).seal_selection(selection)
    _write_state(
        root,
        status="prepared",
        stage="selection-sealed-before-gold",
        case_count=SAMPLE_SIZE,
        inference_calls=0,
        page_mutations=0,
    )
    return read_sealed_json(selection_path)


def _load_manual_gold(
    path: Path,
    anchor_set: AnchorSet,
) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema") != GOLD_SCHEMA
        or payload.get("anchor_epoch") != anchor_set.epoch
        or payload.get("output_contract_epoch") != OUTPUT_CONTRACT_EPOCH
        or payload.get("fixture_status") != "sealed-unseen-before-inference"
    ):
        raise ClassificationError("CVO anchor-set unseen gold contract mismatch")
    output: dict[str, dict[str, Any]] = {}
    for row in payload.get("cases") or []:
        if not isinstance(row, Mapping):
            raise ClassificationError("CVO anchor-set unseen gold case is invalid")
        uid = str(row.get("uid") or "")
        raw_sets = row.get("acceptable_anchor_sets")
        if not isinstance(raw_sets, list):
            raise ClassificationError(
                "CVO anchor-set unseen acceptable sets are missing"
            )
        acceptable_sets: list[list[str]] = []
        seen_sets: set[tuple[str, ...]] = set()
        for raw_set in raw_sets:
            if not isinstance(raw_set, list):
                raise ClassificationError(
                    "CVO anchor-set unseen acceptable set is invalid"
                )
            normalized = tuple(
                sorted(
                    dict.fromkeys(str(value) for value in raw_set if str(value))
                )
            )
            if not 1 <= len(normalized) <= 2 or normalized in seen_sets:
                raise ClassificationError(
                    "CVO anchor-set unseen acceptable set is incomplete"
                )
            seen_sets.add(normalized)
            acceptable_sets.append(list(normalized))
        defensible = sorted(
            dict.fromkeys(
                str(value)
                for value in row.get("defensible_anchor_ids") or []
                if str(value)
            )
        )
        acceptable_union = sorted(
            {value for acceptable in acceptable_sets for value in acceptable}
        )
        if (
            not uid
            or uid in output
            or not acceptable_sets
            or not set(acceptable_union) <= set(defensible)
            or any(value not in anchor_set.by_id for value in defensible)
            or UNRESOLVED_ANCHOR_ID in defensible
        ):
            raise ClassificationError(
                "CVO anchor-set unseen gold case is incomplete"
            )
        output[uid] = {
            "acceptable_sets": acceptable_sets,
            "acceptable_union": acceptable_union,
            "defensible": defensible,
            "gold_basis": str(row.get("gold_basis") or ""),
        }
    return output


def lock_preregistration(root: Path, gold_path: Path) -> dict[str, Any]:
    destination = unseen_root(root)
    preregistration_path = destination / "preregistration.json"
    if preregistration_path.is_file():
        return read_sealed_json(preregistration_path)
    if (destination / "evaluation.json").is_file():
        raise ClassificationError("CVO anchor-set unseen evaluation is sealed")
    case_root = destination / "cases"
    if case_root.exists() and any(case_root.rglob("*.json")):
        raise ClassificationError("CVO inference artifacts exist before gold lock")
    selection_path = destination / "selection.json"
    selection = read_sealed_json(selection_path)
    if selection.get("schema") != SELECTION_SCHEMA:
        raise ClassificationError("CVO anchor-set unseen selection mismatch")
    selected_rows = select_unseen_rows(root)
    selected_by_uid = {
        str(row.get("uid") or ""): row for row in selected_rows
    }
    anchor_set = load_anchor_set()
    gold = _load_manual_gold(gold_path, anchor_set)
    if set(gold) != set(selected_by_uid):
        raise ClassificationError(
            "CVO anchor-set unseen gold UIDs do not match selection"
        )
    fixture_cases = []
    for selected in selection.get("cases") or []:
        if not isinstance(selected, Mapping):
            raise ClassificationError(
                "CVO anchor-set unseen selection case is invalid"
            )
        uid = str(selected.get("uid") or "")
        source = selected_by_uid[uid]
        if (
            str(source.get("source_sha256") or "")
            != str(selected.get("source_sha256") or "")
            or str(source.get("fixture_group_id") or "")
            != str(selected.get("fixture_group_id") or "")
        ):
            raise ClassificationError(
                "CVO anchor-set unseen source changed before lock"
            )
        fixture_cases.append(
            {
                "position": int(selected.get("position") or 0),
                "uid": uid,
                "source_sha256": str(source.get("source_sha256") or ""),
                "fixture_group_id": str(
                    source.get("fixture_group_id") or ""
                ),
                "title": str(source.get("title") or ""),
                "summary": str(source.get("summary") or ""),
                "excerpt": str(source.get("excerpt") or ""),
                "tags": list(source.get("tags") or []),
                "raw_keywords": list(source.get("raw_keywords") or []),
                "acceptable_anchor_sets": gold[uid]["acceptable_sets"],
                "acceptable_anchor_ids": gold[uid]["acceptable_union"],
                "defensible_anchor_ids": gold[uid]["defensible"],
                "gold_basis": gold[uid]["gold_basis"],
            }
        )
    fixture = {
        "schema": FIXTURE_SCHEMA,
        "locked_at": _now(),
        "selection_path": str(selection_path),
        "selection_sha256": sha256_file(selection_path),
        "manual_gold_path": str(gold_path),
        "manual_gold_sha256": sha256_file(gold_path),
        "anchor_set_path": str(default_anchor_set_path()),
        "anchor_set_sha256": sha256_file(default_anchor_set_path()),
        "manual_gold_completed_before_inference": True,
        "case_count": len(fixture_cases),
        "cases": fixture_cases,
    }
    fixture_path = destination / "fixture.json"
    write_sealed_json(fixture_path, fixture, backup=True)
    config = load_decision_router_config()
    models = {
        "extractor": {
            "model": config.primary_model,
            "keep_alive": config.primary_keep_alive,
        },
        "classifier": {
            "model": config.tie_break_model,
            "keep_alive": config.tie_break_keep_alive,
        },
        "second_anchor_auditor": {
            "model": config.primary_model,
            "keep_alive": config.primary_keep_alive,
        },
    }
    digests = ollama.model_digests(
        sorted({str(spec["model"]) for spec in models.values()})
    )
    for spec in models.values():
        spec["model_digest"] = digests.get(str(spec["model"]), "")
        if not spec["model_digest"]:
            raise ClassificationError(
                f"CVO anchor-set unseen model unavailable: {spec['model']}"
            )
    preregistration = {
        "schema": PREREGISTRATION_SCHEMA,
        "locked_at": _now(),
        "selection_path": str(selection_path),
        "selection_sha256": sha256_file(selection_path),
        "fixture_path": str(fixture_path),
        "fixture_sha256": sha256_file(fixture_path),
        "gold_path": str(gold_path),
        "gold_sha256": sha256_file(gold_path),
        "anchor_set_path": str(default_anchor_set_path()),
        "anchor_set_sha256": sha256_file(default_anchor_set_path()),
        "anchor_epoch": anchor_set.epoch,
        "anchor_checksum": anchor_set.checksum,
        "output_contract_epoch": OUTPUT_CONTRACT_EPOCH,
        "models": models,
        "prompt_sha256": {
            "core": CORE_PROMPT_SHA256,
            "second_anchor_auditor": AUDITOR_PROMPT_SHA256,
        },
        "gate": {
            "sample_size": SAMPLE_SIZE,
            "minimum_exact_sets": MINIMUM_EXACT_SETS,
            "minimum_semantic_coverage": MINIMUM_SEMANTIC_COVERAGE,
            "maximum_excess_anchor_rate": MAXIMUM_EXCESS_RATE,
            "maximum_missing_anchor_rate": MAXIMUM_MISSING_RATE,
            "maximum_dual_assignment_rate": MAXIMUM_DUAL_RATE,
            "maximum_holds": MAXIMUM_HOLDS,
            "maximum_major_errors": MAXIMUM_MAJOR_ERRORS,
            "major_error_definition": (
                "Any assigned anchor outside the independently "
                "predeclared defensible set."
            ),
        },
        "successful_model_call_budget": SAMPLE_SIZE * 3,
        "page_mutations": 0,
    }
    LabHarness(root, EXPERIMENT).lock_preregistration(preregistration)
    _write_state(
        root,
        status="locked",
        stage="gold-and-preregistration-sealed",
        case_count=SAMPLE_SIZE,
        inference_calls=0,
        page_mutations=0,
    )
    return read_sealed_json(preregistration_path)


def _validate_preregistration(root: Path) -> dict[str, Any]:
    harness = LabHarness(root, EXPERIMENT)
    preregistration = harness.read_preregistration()
    require_contract(
        preregistration,
        schema=PREREGISTRATION_SCHEMA,
        exact={
            "gate": {
                "sample_size": SAMPLE_SIZE,
                "minimum_exact_sets": MINIMUM_EXACT_SETS,
                "minimum_semantic_coverage": MINIMUM_SEMANTIC_COVERAGE,
                "maximum_excess_anchor_rate": MAXIMUM_EXCESS_RATE,
                "maximum_missing_anchor_rate": MAXIMUM_MISSING_RATE,
                "maximum_dual_assignment_rate": MAXIMUM_DUAL_RATE,
                "maximum_holds": MAXIMUM_HOLDS,
                "maximum_major_errors": MAXIMUM_MAJOR_ERRORS,
                "major_error_definition": (
                    "Any assigned anchor outside the independently "
                    "predeclared defensible set."
                ),
            },
            "prompt_sha256": {
                "core": CORE_PROMPT_SHA256,
                "second_anchor_auditor": AUDITOR_PROMPT_SHA256,
            },
        },
        error_type=ClassificationError,
        message="CVO anchor-set unseen preregistration contract changed",
    )
    require_file_hashes(
        {
            Path(str(preregistration["selection_path"])): str(
                preregistration["selection_sha256"]
            ),
            Path(str(preregistration["fixture_path"])): str(
                preregistration["fixture_sha256"]
            ),
            Path(str(preregistration["gold_path"])): str(
                preregistration["gold_sha256"]
            ),
            Path(str(preregistration["anchor_set_path"])): str(
                preregistration["anchor_set_sha256"]
            ),
        },
        digest=sha256_file,
        error_type=ClassificationError,
        message="CVO anchor-set unseen sealed input changed",
    )
    models = dict(preregistration.get("models") or {})
    names = {
        str(spec.get("model") or "")
        for spec in models.values()
        if isinstance(spec, Mapping)
    }
    current_digests = ollama.model_digests(sorted(names))
    for spec in models.values():
        if (
            not isinstance(spec, Mapping)
            or current_digests.get(str(spec.get("model") or ""))
            != str(spec.get("model_digest") or "")
        ):
            raise ClassificationError("CVO anchor-set unseen model digest changed")
    return preregistration


def evaluate_unseen(root: Path) -> dict[str, Any]:
    destination = unseen_root(root)
    evaluation_path = destination / "evaluation.json"
    if evaluation_path.is_file():
        return read_sealed_json(evaluation_path)
    preregistration = _validate_preregistration(root)
    fixture = read_sealed_json(destination / "fixture.json")
    anchor_set = load_anchor_set()
    models = dict(preregistration["models"])
    config = load_decision_router_config()
    cases = []
    model_calls = 0
    for page in fixture.get("cases") or []:
        if not isinstance(page, Mapping):
            raise ClassificationError(
                "CVO anchor-set unseen fixture case is invalid"
            )
        uid = str(page.get("uid") or "")
        core = run_case(
            root,
            page,
            anchor_set,
            extractor=dict(models["extractor"]),
            classifier=dict(models["classifier"]),
            read_timeout_ms=config.read_timeout_ms,
            experiment=EXPERIMENT,
        )
        core_selection = dict(core.get("selection") or {})
        core_anchor_id = str(
            core_selection.get("primary_anchor_id")
            or UNRESOLVED_ANCHOR_ID
        )
        core_calls = int(core.get("model_calls") or 0)
        if core_anchor_id == UNRESOLVED_ANCHOR_ID:
            audit = {
                "schema": AUDIT_SCHEMA,
                "second_anchor_id": "NONE",
                "independent_principal_subject": False,
                "not_subsumed_by_core": False,
                "not_incidental_context": False,
                "explicit_document_evidence": False,
                "admitted": False,
                "rationale": "Core classifier held the page.",
                "invalid_reason": "core_hold",
            }
            audit_calls = 0
        else:
            auditor = dict(models["second_anchor_auditor"])
            artifact = _call_auditor(
                root,
                uid,
                _auditor_payload(
                    page_result=core,
                    anchor_set=anchor_set,
                    core_anchor_id=core_anchor_id,
                    model=str(auditor["model"]),
                    model_digest=str(auditor["model_digest"]),
                    keep_alive=str(auditor["keep_alive"]),
                    read_timeout_ms=config.read_timeout_ms,
                ),
                experiment=EXPERIMENT,
            )
            audit = dict(artifact["audit"])
            audit_calls = int(artifact.get("model_calls") or 0)
        selected = [core_anchor_id]
        if audit.get("admitted"):
            selected.append(str(audit["second_anchor_id"]))
        selected = sorted(dict.fromkeys(selected))
        acceptable_sets = [
            [str(value) for value in acceptable]
            for acceptable in page.get("acceptable_anchor_sets") or []
        ]
        acceptable_union = [
            str(value)
            for value in page.get("acceptable_anchor_ids") or []
        ]
        defensible = [
            str(value)
            for value in page.get("defensible_anchor_ids") or []
        ]
        score = score_anchor_set(
            selected,
            acceptable_union,
            defensible,
            acceptable_sets,
        )
        model_calls += core_calls + audit_calls
        cases.append(
            {
                "uid": uid,
                "title": str(page.get("title") or ""),
                "core_anchor_id": core_anchor_id,
                "subject": core.get("subject"),
                "audit": audit,
                "selected_anchor_ids": selected,
                "acceptable_anchor_sets": acceptable_sets,
                "defensible_anchor_ids": defensible,
                **score,
            }
        )
    metrics = summarize_metrics(cases)
    passed = (
        metrics["case_count"] == SAMPLE_SIZE
        and metrics["exact_sets"] >= MINIMUM_EXACT_SETS
        and metrics["semantic_coverage_cases"] >= MINIMUM_SEMANTIC_COVERAGE
        and metrics["excess_anchor_rate"] <= MAXIMUM_EXCESS_RATE
        and metrics["missing_anchor_rate"] <= MAXIMUM_MISSING_RATE
        and metrics["dual_assignment_rate"] <= MAXIMUM_DUAL_RATE
        and metrics["holds"] <= MAXIMUM_HOLDS
        and metrics["major_errors"] <= MAXIMUM_MAJOR_ERRORS
    )
    evaluation = {
        "schema": EVALUATION_SCHEMA,
        "evaluated_at": _now(),
        "preregistration_path": str(
            destination / "preregistration.json"
        ),
        "preregistration_sha256": sha256_file(
            destination / "preregistration.json"
        ),
        "models": models,
        "anchor_epoch": anchor_set.epoch,
        "anchor_checksum": anchor_set.checksum,
        "output_contract_epoch": OUTPUT_CONTRACT_EPOCH,
        "prompt_sha256": preregistration["prompt_sha256"],
        "model_calls": model_calls,
        "page_mutations": 0,
        "gate": preregistration["gate"],
        "metrics": metrics,
        "cases": cases,
        "decision": (
            "qualify-cvo-anchor-set-v1-unseen40"
            if passed
            else "reject-cvo-anchor-set-v1-unseen40"
        ),
        "layer2_implementation_authorized": passed,
    }
    write_sealed_json(evaluation_path, evaluation, backup=True)
    _write_state(
        root,
        status="passed" if passed else "rejected",
        stage="unseen-evaluation-sealed",
        metrics=metrics,
        inference_calls=model_calls,
        page_mutations=0,
    )
    return read_sealed_json(evaluation_path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prepare, lock, or evaluate the CVO anchor-set unseen gate"
    )
    parser.add_argument("--root", type=Path, default=CHRONOVISOR_ROOT)
    parser.add_argument(
        "--stage",
        choices=("prepare", "lock", "evaluate"),
        required=True,
    )
    parser.add_argument("--gold", type=Path, default=default_unseen_gold_path())
    args = parser.parse_args(argv)
    root = args.root.expanduser()
    try:
        if args.stage == "prepare":
            result = prepare_selection(root)
        elif args.stage == "lock":
            result = lock_preregistration(root, args.gold.expanduser())
        else:
            result = evaluate_unseen(root)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    except (
        ClassificationError,
        OSError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
