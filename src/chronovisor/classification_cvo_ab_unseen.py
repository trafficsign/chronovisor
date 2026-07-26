"""Preregistered A/B gate for direct max-2 and controlled-vocabulary CVO."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from chronovisor import ollama
from chronovisor.classification import ClassificationError
from chronovisor.classification_anchor import (
    UNRESOLVED_ANCHOR_ID,
    default_anchor_set_path,
    load_anchor_set,
)
from chronovisor.classification_anchor_set_dev import (
    default_dev_gold_path,
    load_dev70,
    run_case,
    run_case_with_subject,
    score_anchor_set,
    summarize_metrics,
    validate_set_gold,
)
from chronovisor.classification_anchor_set_unseen import select_unseen_rows
from chronovisor.classification_anchor_set_worker import PROMPT_SHA256
from chronovisor.classification_controlled_vocabulary import (
    default_vocabulary_path,
    load_controlled_vocabulary,
)
from chronovisor.classification_fixture_set import sha256_file
from chronovisor.durable_state import read_sealed_json, write_sealed_json
from chronovisor.runtime_config import load_decision_router_config
from chronovisor.store import CHRONOVISOR_ROOT

SELECTION_SCHEMA = "chronovisor.cvo-ab-unseen-selection.v1"
GOLD_SCHEMA = "chronovisor.cvo-ab-unseen-gold.v1"
FIXTURE_SCHEMA = "chronovisor.cvo-ab-unseen-fixture.v1"
PREREGISTRATION_SCHEMA = "chronovisor.cvo-ab-unseen-preregistration.v1"
EVALUATION_SCHEMA = "chronovisor.cvo-ab-unseen-evaluation.v1"
RESCORE_SCHEMA = "chronovisor.cvo-opened30-max2-rescore.v1"
CONTROLLED_VOCABULARY_DEV_SCHEMA = (
    "chronovisor.controlled-vocabulary-dev.v1"
)
STATE_SCHEMA = "chronovisor.cvo-ab-unseen-state.v1"

EXPERIMENT = "cvo-ab-v1-unseen40"
ARM_A_EXPERIMENT = f"{EXPERIMENT}-arm-a"
ARM_B_EXPERIMENT = f"{EXPERIMENT}-arm-b"
ARM_B_DEV_EXPERIMENT = "cvo-controlled-vocabulary-v1-dev70"
SELECTION_SEED = "cvo-ab-v1-group-separated-unseen40-v1"
OUTPUT_CONTRACT_EPOCH = "cvo-anchor-set-v1"
SAMPLE_SIZE = 40
MINIMUM_EXACT_SETS = 36
MINIMUM_SEMANTIC_COVERAGE = 38
MAXIMUM_EXCESS_RATE = 0.10
MAXIMUM_MISSING_RATE = 0.10
MAXIMUM_DUAL_RATE = 0.40
MAXIMUM_HOLDS = 2
MAXIMUM_MAJOR_ERRORS = 0
DEV_CASES = 70
DEV_MINIMUM_EXACT_SETS = 63
DEV_MINIMUM_SEMANTIC_COVERAGE = 67
DEV_MAXIMUM_HOLDS = 3


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def output_root(root: Path) -> Path:
    return root / "classification" / EXPERIMENT


def default_gold_path() -> Path:
    return Path(__file__).parent / "data" / "cvo-ab-unseen-gold-v1.json"


def _extra_receipt_paths(root: Path) -> tuple[Path, ...]:
    return tuple(
        root / "classification" / name / "selection.json"
        for name in (
            "cvo-anchor-set-v1-unseen40",
            "cvo-intent-v1-unseen40",
            "cvo-intent-v2-unseen40",
        )
    )


def _extra_gold_paths() -> tuple[Path, ...]:
    data = Path(__file__).parent / "data"
    return tuple(
        data / name
        for name in (
            "cvo-anchor-set-unseen-gold-v1.json",
            "cvo-intent-unseen-gold-v1.json",
            "cvo-intent-unseen-gold-v2.json",
        )
    )


def selected_rows(root: Path) -> list[dict[str, Any]]:
    return select_unseen_rows(
        root,
        sample_size=SAMPLE_SIZE,
        selection_seed=SELECTION_SEED,
        extra_receipt_paths=_extra_receipt_paths(root),
        extra_gold_paths=_extra_gold_paths(),
    )


def _write_state(
    root: Path,
    *,
    status: str,
    stage: str,
    **detail: Any,
) -> None:
    write_sealed_json(
        output_root(root) / "state.json",
        {
            "schema": STATE_SCHEMA,
            "status": status,
            "stage": stage,
            "updated_at": _now(),
            **detail,
        },
        backup=True,
    )


def prepare_selection(root: Path) -> dict[str, Any]:
    destination = output_root(root)
    selection_path = destination / "selection.json"
    if (destination / "evaluation.json").is_file():
        raise ClassificationError("CVO A/B evaluation is already sealed")
    if selection_path.is_file():
        return read_sealed_json(selection_path)
    rows = selected_rows(root)
    selection = {
        "schema": SELECTION_SCHEMA,
        "created_at": _now(),
        "selection_seed": SELECTION_SEED,
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
                "title": str(row.get("title") or ""),
                "summary": str(row.get("summary") or ""),
                "excerpt": str(row.get("excerpt") or ""),
                "tags": list(row.get("tags") or []),
                "raw_keywords": list(row.get("raw_keywords") or []),
            }
            for position, row in enumerate(rows, start=1)
        ],
    }
    write_sealed_json(selection_path, selection, backup=True)
    _write_state(
        root,
        status="prepared",
        stage="selection-sealed-before-gold",
        case_count=len(rows),
        inference_calls=0,
        page_mutations=0,
    )
    return read_sealed_json(selection_path)


def _load_gold(path: Path) -> dict[str, dict[str, Any]]:
    anchor_set = load_anchor_set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema") != GOLD_SCHEMA
        or payload.get("anchor_epoch") != anchor_set.epoch
        or payload.get("output_contract_epoch") != OUTPUT_CONTRACT_EPOCH
        or payload.get("fixture_status") != "sealed-unseen-before-inference"
    ):
        raise ClassificationError("CVO A/B gold contract mismatch")
    output: dict[str, dict[str, Any]] = {}
    for row in payload.get("cases") or []:
        if not isinstance(row, Mapping):
            raise ClassificationError("CVO A/B gold row is invalid")
        uid = str(row.get("uid") or "")
        acceptable_sets: list[list[str]] = []
        for values in row.get("acceptable_anchor_sets") or []:
            if not isinstance(values, list):
                raise ClassificationError("CVO A/B acceptable set is invalid")
            normalized = sorted(
                dict.fromkeys(str(value) for value in values if str(value))
            )
            if not 1 <= len(normalized) <= 2:
                raise ClassificationError(
                    "CVO A/B acceptable set exceeds max-2"
                )
            acceptable_sets.append(normalized)
        defensible = sorted(
            dict.fromkeys(
                str(value)
                for value in row.get("defensible_anchor_ids") or []
                if str(value)
            )
        )
        acceptable_union = sorted(
            {
                value
                for acceptable in acceptable_sets
                for value in acceptable
            }
        )
        if (
            not uid
            or uid in output
            or not acceptable_sets
            or len({tuple(values) for values in acceptable_sets})
            != len(acceptable_sets)
            or not set(acceptable_union) <= set(defensible)
            or any(value not in anchor_set.by_id for value in defensible)
            or UNRESOLVED_ANCHOR_ID in defensible
            or not str(row.get("gold_basis") or "").strip()
        ):
            raise ClassificationError("CVO A/B gold row is incomplete")
        output[uid] = {
            "acceptable_sets": acceptable_sets,
            "acceptable_union": acceptable_union,
            "defensible": defensible,
            "gold_basis": str(row["gold_basis"]),
        }
    return output


def _gate() -> dict[str, Any]:
    return {
        "sample_size": SAMPLE_SIZE,
        "minimum_exact_sets": MINIMUM_EXACT_SETS,
        "minimum_semantic_coverage": MINIMUM_SEMANTIC_COVERAGE,
        "maximum_excess_anchor_rate": MAXIMUM_EXCESS_RATE,
        "maximum_missing_anchor_rate": MAXIMUM_MISSING_RATE,
        "maximum_dual_assignment_rate": MAXIMUM_DUAL_RATE,
        "maximum_holds": MAXIMUM_HOLDS,
        "maximum_major_errors": MAXIMUM_MAJOR_ERRORS,
        "major_error_definition": (
            "Any assigned anchor outside the independently predeclared "
            "defensible set."
        ),
    }


def _arm_passed(metrics: Mapping[str, Any]) -> bool:
    return (
        int(metrics.get("case_count") or 0) == SAMPLE_SIZE
        and int(metrics.get("exact_sets") or 0) >= MINIMUM_EXACT_SETS
        and int(metrics.get("semantic_coverage_cases") or 0)
        >= MINIMUM_SEMANTIC_COVERAGE
        and float(metrics.get("excess_anchor_rate") or 0.0)
        <= MAXIMUM_EXCESS_RATE
        and float(metrics.get("missing_anchor_rate") or 0.0)
        <= MAXIMUM_MISSING_RATE
        and float(metrics.get("dual_assignment_rate") or 0.0)
        <= MAXIMUM_DUAL_RATE
        and int(metrics.get("holds") or 0) <= MAXIMUM_HOLDS
        and int(metrics.get("major_errors") or 0) <= MAXIMUM_MAJOR_ERRORS
    )


def _development_arm_passed(metrics: Mapping[str, Any]) -> bool:
    return (
        int(metrics.get("case_count") or 0) == DEV_CASES
        and int(metrics.get("exact_sets") or 0)
        >= DEV_MINIMUM_EXACT_SETS
        and int(metrics.get("semantic_coverage_cases") or 0)
        >= DEV_MINIMUM_SEMANTIC_COVERAGE
        and float(metrics.get("excess_anchor_rate") or 0.0)
        <= MAXIMUM_EXCESS_RATE
        and float(metrics.get("missing_anchor_rate") or 0.0)
        <= MAXIMUM_MISSING_RATE
        and float(metrics.get("dual_assignment_rate") or 0.0)
        <= MAXIMUM_DUAL_RATE
        and int(metrics.get("holds") or 0) <= DEV_MAXIMUM_HOLDS
        and int(metrics.get("major_errors") or 0)
        <= MAXIMUM_MAJOR_ERRORS
    )


def calibrate_controlled_vocabulary_opened70(
    root: Path,
) -> dict[str, Any]:
    """Early-kill Arm B on opened development data before unseen inference."""

    destination = (
        root
        / "classification"
        / ARM_B_DEV_EXPERIMENT
        / "evaluation.json"
    )
    if destination.is_file():
        return read_sealed_json(destination)
    pages = load_dev70(root)
    anchor_set = load_anchor_set()
    vocabulary = load_controlled_vocabulary()
    gold_path = default_dev_gold_path()
    payload = json.loads(gold_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ClassificationError("CVO Arm B development gold is invalid")
    gold = validate_set_gold(
        payload,
        anchor_set,
        [str(page.get("uid") or "") for page in pages],
    )
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
    }
    digests = ollama.model_digests(
        sorted({str(spec["model"]) for spec in models.values()})
    )
    for spec in models.values():
        spec["model_digest"] = digests.get(str(spec["model"]), "")
        if not spec["model_digest"]:
            raise ClassificationError("CVO Arm B model is unavailable")
    cases = []
    new_model_calls = 0
    inherited_extractor_calls = 0
    gap_cases = []
    arm_a_root = (
        root / "classification" / "cvo-anchor-set-v1-dev70" / "cases"
    )
    for page in pages:
        uid = str(page.get("uid") or "")
        arm_a_case = read_sealed_json(arm_a_root / uid / "result.json")
        subject = arm_a_case.get("subject")
        if not isinstance(subject, Mapping):
            raise ClassificationError(
                "CVO Arm B requires the opened Arm A subject receipt"
            )
        result = run_case_with_subject(
            root,
            page,
            vocabulary,
            subject=subject,
            classifier=models["classifier"],
            read_timeout_ms=config.read_timeout_ms,
            experiment=ARM_B_DEV_EXPERIMENT,
            inherited_model_calls=1,
        )
        selected_terms = sorted(
            dict.fromkeys(
                str(value)
                for value in (result.get("selection") or {}).get(
                    "anchor_ids"
                )
                or []
            )
        )
        selected_anchors = vocabulary.anchors_for_terms(selected_terms)
        score = score_anchor_set(
            selected_anchors,
            gold[uid]["target"],
            gold[uid]["defensible"],
            gold[uid]["acceptable_sets"],
        )
        new_model_calls += int(result.get("new_model_calls") or 0)
        inherited_extractor_calls += 1
        if selected_anchors == [UNRESOLVED_ANCHOR_ID]:
            gap_cases.append(uid)
        cases.append(
            {
                "uid": uid,
                "title": str(page.get("title") or ""),
                "subject": dict(subject),
                "selected_term_ids": selected_terms,
                "selected_anchor_ids": selected_anchors,
                "acceptable_anchor_sets": gold[uid]["acceptable_sets"],
                "defensible_anchor_ids": gold[uid]["defensible"],
                **score,
            }
        )
    metrics = summarize_metrics(cases)
    passed = _development_arm_passed(metrics)
    receipt = {
        "schema": CONTROLLED_VOCABULARY_DEV_SCHEMA,
        "evaluated_at": _now(),
        "fixture_set": "opened70-development-only",
        "development_only": True,
        "anchor_set_path": str(default_anchor_set_path()),
        "anchor_set_sha256": sha256_file(default_anchor_set_path()),
        "vocabulary_path": str(default_vocabulary_path()),
        "vocabulary_sha256": sha256_file(default_vocabulary_path()),
        "vocabulary_epoch": vocabulary.epoch,
        "vocabulary_checksum": vocabulary.checksum,
        "gold_path": str(gold_path),
        "gold_sha256": sha256_file(gold_path),
        "models": models,
        "prompt_sha256": PROMPT_SHA256,
        "inherited_extractor_calls": inherited_extractor_calls,
        "new_model_calls": new_model_calls,
        "effective_pipeline_calls": (
            inherited_extractor_calls + new_model_calls
        ),
        "page_mutations": 0,
        "gap_count": len(gap_cases),
        "gap_uids": gap_cases,
        "gate": {
            "case_count": DEV_CASES,
            "minimum_exact_sets": DEV_MINIMUM_EXACT_SETS,
            "minimum_semantic_coverage": (
                DEV_MINIMUM_SEMANTIC_COVERAGE
            ),
            "maximum_excess_anchor_rate": MAXIMUM_EXCESS_RATE,
            "maximum_missing_anchor_rate": MAXIMUM_MISSING_RATE,
            "maximum_dual_assignment_rate": MAXIMUM_DUAL_RATE,
            "maximum_holds": DEV_MAXIMUM_HOLDS,
            "maximum_major_errors": MAXIMUM_MAJOR_ERRORS,
        },
        "metrics": metrics,
        "cases": cases,
        "decision": (
            "continue-to-sealed-unseen40"
            if passed
            else "early-kill-before-unseen40"
        ),
    }
    write_sealed_json(destination, receipt, backup=True)
    return read_sealed_json(destination)


def lock_preregistration(root: Path, gold_path: Path) -> dict[str, Any]:
    destination = output_root(root)
    preregistration_path = destination / "preregistration.json"
    if preregistration_path.is_file():
        return read_sealed_json(preregistration_path)
    for experiment in (ARM_A_EXPERIMENT, ARM_B_EXPERIMENT):
        case_root = root / "classification" / experiment / "cases"
        if case_root.exists() and any(case_root.rglob("*.json")):
            raise ClassificationError(
                "CVO A/B inference exists before preregistration"
            )
    selection_path = destination / "selection.json"
    selection = read_sealed_json(selection_path)
    if selection.get("schema") != SELECTION_SCHEMA:
        raise ClassificationError("CVO A/B selection contract mismatch")
    rows = selected_rows(root)
    rows_by_uid = {str(row["uid"]): row for row in rows}
    gold = _load_gold(gold_path)
    if set(gold) != set(rows_by_uid):
        raise ClassificationError("CVO A/B gold UIDs differ from selection")
    fixture_cases = []
    for selected in selection.get("cases") or []:
        if not isinstance(selected, Mapping):
            raise ClassificationError("CVO A/B selection row is invalid")
        uid = str(selected.get("uid") or "")
        source = rows_by_uid[uid]
        if (
            str(source.get("source_sha256") or "")
            != str(selected.get("source_sha256") or "")
            or str(source.get("fixture_group_id") or "")
            != str(selected.get("fixture_group_id") or "")
        ):
            raise ClassificationError("CVO A/B source changed before lock")
        fixture_cases.append(
            {
                **dict(selected),
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
    }
    digests = ollama.model_digests(
        sorted({str(spec["model"]) for spec in models.values()})
    )
    for spec in models.values():
        spec["model_digest"] = digests.get(str(spec["model"]), "")
        if not spec["model_digest"]:
            raise ClassificationError("CVO A/B model is unavailable")
    vocabulary = load_controlled_vocabulary()
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
        "vocabulary_path": str(default_vocabulary_path()),
        "vocabulary_sha256": sha256_file(default_vocabulary_path()),
        "vocabulary_epoch": vocabulary.epoch,
        "vocabulary_checksum": vocabulary.checksum,
        "output_contract_epoch": OUTPUT_CONTRACT_EPOCH,
        "models": models,
        "arm_prompts": {
            "arm_a_direct_max2": PROMPT_SHA256,
            "arm_b_term_selection": PROMPT_SHA256,
        },
        "gate": _gate(),
        "successful_model_call_budget": SAMPLE_SIZE * 4,
        "same_fixture_for_both_arms": True,
        "cross_arm_adjustment_forbidden": True,
        "page_mutations": 0,
    }
    write_sealed_json(preregistration_path, preregistration, backup=True)
    _write_state(
        root,
        status="locked",
        stage="both-arms-sealed-before-inference",
        case_count=SAMPLE_SIZE,
        inference_calls=0,
        page_mutations=0,
    )
    return read_sealed_json(preregistration_path)


def _validate_preregistration(root: Path) -> dict[str, Any]:
    destination = output_root(root)
    prereg = read_sealed_json(destination / "preregistration.json")
    if (
        prereg.get("schema") != PREREGISTRATION_SCHEMA
        or prereg.get("gate") != _gate()
        or prereg.get("arm_prompts")
        != {
            "arm_a_direct_max2": PROMPT_SHA256,
            "arm_b_term_selection": PROMPT_SHA256,
        }
    ):
        raise ClassificationError("CVO A/B preregistration changed")
    checks = {
        Path(str(prereg["selection_path"])): str(
            prereg["selection_sha256"]
        ),
        Path(str(prereg["fixture_path"])): str(prereg["fixture_sha256"]),
        Path(str(prereg["gold_path"])): str(prereg["gold_sha256"]),
        Path(str(prereg["anchor_set_path"])): str(
            prereg["anchor_set_sha256"]
        ),
        Path(str(prereg["vocabulary_path"])): str(
            prereg["vocabulary_sha256"]
        ),
    }
    for path, expected in checks.items():
        if not path.is_file() or sha256_file(path) != expected:
            raise ClassificationError(f"sealed CVO A/B input changed: {path}")
    models = dict(prereg.get("models") or {})
    digests = ollama.model_digests(
        sorted(
            {
                str(spec.get("model") or "")
                for spec in models.values()
                if isinstance(spec, Mapping)
            }
        )
    )
    for spec in models.values():
        if (
            not isinstance(spec, Mapping)
            or digests.get(str(spec.get("model") or ""))
            != str(spec.get("model_digest") or "")
        ):
            raise ClassificationError("CVO A/B model digest changed")
    vocabulary = load_controlled_vocabulary()
    if (
        vocabulary.epoch != prereg.get("vocabulary_epoch")
        or vocabulary.checksum != prereg.get("vocabulary_checksum")
    ):
        raise ClassificationError("CVO A/B vocabulary changed")
    return prereg


def _score_case(
    page: Mapping[str, Any],
    selected: list[str],
) -> dict[str, Any]:
    acceptable_sets = [
        [str(value) for value in values]
        for values in page.get("acceptable_anchor_sets") or []
    ]
    acceptable_union = [
        str(value) for value in page.get("acceptable_anchor_ids") or []
    ]
    defensible = [
        str(value) for value in page.get("defensible_anchor_ids") or []
    ]
    return score_anchor_set(
        selected,
        acceptable_union,
        defensible,
        acceptable_sets,
    )


def evaluate(root: Path) -> dict[str, Any]:
    destination = output_root(root)
    evaluation_path = destination / "evaluation.json"
    if evaluation_path.is_file():
        return read_sealed_json(evaluation_path)
    prereg = _validate_preregistration(root)
    fixture = read_sealed_json(destination / "fixture.json")
    anchor_set = load_anchor_set()
    vocabulary = load_controlled_vocabulary()
    models = dict(prereg["models"])
    config = load_decision_router_config()
    arm_a_cases = []
    arm_b_cases = []
    gap_cases = []
    model_calls = {"arm_a": 0, "arm_b": 0}
    for page in fixture.get("cases") or []:
        if not isinstance(page, Mapping):
            raise ClassificationError("CVO A/B fixture row is invalid")
        uid = str(page.get("uid") or "")
        arm_a = run_case(
            root,
            page,
            anchor_set,
            extractor=dict(models["extractor"]),
            classifier=dict(models["classifier"]),
            read_timeout_ms=config.read_timeout_ms,
            experiment=ARM_A_EXPERIMENT,
        )
        selected_a = sorted(
            dict.fromkeys(
                str(value)
                for value in (arm_a.get("selection") or {}).get(
                    "anchor_ids"
                )
                or []
            )
        )
        score_a = _score_case(page, selected_a)
        model_calls["arm_a"] += int(arm_a.get("model_calls") or 0)
        arm_a_cases.append(
            {
                "uid": uid,
                "title": str(page.get("title") or ""),
                "subject": arm_a.get("subject"),
                "selected_anchor_ids": selected_a,
                "acceptable_anchor_sets": page.get(
                    "acceptable_anchor_sets"
                ),
                **score_a,
            }
        )
        arm_b = run_case(
            root,
            page,
            vocabulary,  # type: ignore[arg-type]
            extractor=dict(models["extractor"]),
            classifier=dict(models["classifier"]),
            read_timeout_ms=config.read_timeout_ms,
            experiment=ARM_B_EXPERIMENT,
        )
        selected_terms = sorted(
            dict.fromkeys(
                str(value)
                for value in (arm_b.get("selection") or {}).get(
                    "anchor_ids"
                )
                or []
            )
        )
        selected_b = vocabulary.anchors_for_terms(selected_terms)
        score_b = _score_case(page, selected_b)
        model_calls["arm_b"] += int(arm_b.get("model_calls") or 0)
        gap = selected_b == [UNRESOLVED_ANCHOR_ID]
        if gap:
            gap_cases.append(
                {
                    "uid": uid,
                    "title": str(page.get("title") or ""),
                    "selected_term_ids": selected_terms,
                    "reason": "controlled-vocabulary-gap",
                }
            )
        arm_b_cases.append(
            {
                "uid": uid,
                "title": str(page.get("title") or ""),
                "subject": arm_b.get("subject"),
                "selected_term_ids": selected_terms,
                "selected_anchor_ids": selected_b,
                "vocabulary_gap": gap,
                "acceptable_anchor_sets": page.get(
                    "acceptable_anchor_sets"
                ),
                **score_b,
            }
        )
    metrics_a = summarize_metrics(arm_a_cases)
    metrics_b = summarize_metrics(arm_b_cases)
    passed_a = _arm_passed(metrics_a)
    passed_b = _arm_passed(metrics_b)
    write_sealed_json(
        destination / "vocabulary-gap-queue.json",
        {
            "schema": "chronovisor.controlled-vocabulary-gap-queue.v1",
            "created_at": _now(),
            "vocabulary_epoch": vocabulary.epoch,
            "case_count": len(gap_cases),
            "cases": gap_cases,
            "page_mutations": 0,
        },
        backup=True,
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
        "gate": _gate(),
        "same_fixture_for_both_arms": True,
        "cross_arm_adjustment_performed": False,
        "model_calls": model_calls,
        "page_mutations": 0,
        "arms": {
            "A": {
                "method": "ornith-subject-gemma-direct-unordered-max2",
                "passed": passed_a,
                "metrics": metrics_a,
                "cases": arm_a_cases,
            },
            "B": {
                "method": "ornith-subject-gemma-controlled-term-to-anchor",
                "passed": passed_b,
                "metrics": metrics_b,
                "vocabulary_epoch": vocabulary.epoch,
                "vocabulary_checksum": vocabulary.checksum,
                "gap_count": len(gap_cases),
                "cases": arm_b_cases,
            },
        },
        "phase4_candidate": "A" if passed_a else ("B" if passed_b else None),
        "layer2_implementation_authorized": passed_b,
        "decision": (
            "qualify-arm-a"
            if passed_a
            else ("qualify-arm-b" if passed_b else "reject-both-arms")
        ),
    }
    write_sealed_json(evaluation_path, evaluation, backup=True)
    _write_state(
        root,
        status="qualified" if passed_a or passed_b else "rejected",
        stage="sealed-two-arm-evaluation-complete",
        arm_a_passed=passed_a,
        arm_b_passed=passed_b,
        inference_calls=sum(model_calls.values()),
        page_mutations=0,
    )
    return read_sealed_json(evaluation_path)


def rescore_opened30(root: Path) -> dict[str, Any]:
    destination = (
        root
        / "classification"
        / "cvo-anchor-v0-unseen30-max2-rescore"
        / "evaluation.json"
    )
    if destination.is_file():
        return read_sealed_json(destination)
    source_path = (
        root
        / "classification"
        / "cvo-anchor-v0-unseen30"
        / "evaluation.json"
    )
    source = read_sealed_json(source_path)
    payload = json.loads(default_dev_gold_path().read_text(encoding="utf-8"))
    gold = validate_set_gold(
        payload,
        load_anchor_set(),
        [str(row.get("uid") or "") for row in payload.get("cases") or []],
    )
    cases = []
    for row in source.get("cases") or []:
        if not isinstance(row, Mapping):
            raise ClassificationError("opened30 row is invalid")
        uid = str(row.get("uid") or "")
        selection = dict(row.get("selection") or {})
        selected = [
            str(selection.get("primary_anchor_id") or ""),
            *[
                str(value)
                for value in selection.get("secondary_anchor_ids") or []
            ],
        ]
        selected = [value for value in selected if value]
        score = score_anchor_set(
            selected,
            gold[uid]["target"],
            gold[uid]["defensible"],
            gold[uid]["acceptable_sets"],
        )
        cases.append(
            {
                "uid": uid,
                "title": str(row.get("title") or ""),
                "selected_anchor_ids": selected,
                "acceptable_anchor_sets": gold[uid]["acceptable_sets"],
                **score,
            }
        )
    metrics = summarize_metrics(cases)
    receipt = {
        "schema": RESCORE_SCHEMA,
        "evaluated_at": _now(),
        "source_path": str(source_path),
        "source_sha256": sha256_file(source_path),
        "gold_path": str(default_dev_gold_path()),
        "gold_sha256": sha256_file(default_dev_gold_path()),
        "development_only": True,
        "new_model_calls": 0,
        "page_mutations": 0,
        "gate": _gate(),
        "metrics": metrics,
        "cases": cases,
        "decision": (
            "qualify-opened30-rescore"
            if _arm_passed(metrics)
            else "reject-opened30-rescore"
        ),
    }
    write_sealed_json(destination, receipt, backup=True)
    return read_sealed_json(destination)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=CHRONOVISOR_ROOT,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("rescore-opened30")
    subparsers.add_parser("calibrate-opened70")
    subparsers.add_parser("prepare")
    lock = subparsers.add_parser("lock")
    lock.add_argument("--gold", type=Path, default=default_gold_path())
    subparsers.add_parser("evaluate")
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        if args.command == "rescore-opened30":
            payload = rescore_opened30(args.root)
        elif args.command == "calibrate-opened70":
            payload = calibrate_controlled_vocabulary_opened70(args.root)
        elif args.command == "prepare":
            payload = prepare_selection(args.root)
        elif args.command == "lock":
            payload = lock_preregistration(args.root, args.gold)
        else:
            payload = evaluate(args.root)
    except (ClassificationError, OSError, ValueError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}))
        return 1
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
