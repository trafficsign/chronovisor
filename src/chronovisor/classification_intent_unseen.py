"""Preregistered unseen gate for the CVO core plus intent-lexicon lane."""

from __future__ import annotations

import argparse
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
    default_anchor_set_path,
    load_anchor_set,
)
from chronovisor.classification_anchor_dev import run_case
from chronovisor.classification_anchor_set_dev import (
    score_anchor_set,
    summarize_metrics,
)
from chronovisor.classification_anchor_set_unseen import (
    _candidate_path,
    _load_manual_gold,
    select_unseen_rows,
)
from chronovisor.classification_anchor_worker import PROMPT_SHA256
from chronovisor.classification_fixture_set import sha256_file
from chronovisor.classification_intent_lexicon import (
    classify_complement,
    default_intent_lexicon_path,
    load_intent_lexicon,
)
from chronovisor.durable_state import read_sealed_json, write_sealed_json
from chronovisor.runtime_config import load_decision_router_config
from chronovisor.store import CHRONOVISOR_ROOT

SELECTION_SCHEMA = "chronovisor.cvo-intent-unseen-selection.v2"
FIXTURE_SCHEMA = "chronovisor.cvo-intent-unseen-fixture.v2"
PREREGISTRATION_SCHEMA = "chronovisor.cvo-intent-unseen-preregistration.v2"
EVALUATION_SCHEMA = "chronovisor.cvo-intent-unseen-evaluation.v2"
STATE_SCHEMA = "chronovisor.cvo-intent-unseen-state.v2"
EXPERIMENT = "cvo-intent-v2-unseen40"
PRIOR_EXPERIMENTS = (
    "cvo-anchor-set-v1-unseen40",
    "cvo-intent-v1-unseen40",
)
SELECTION_SEED = "cvo-intent-v2-group-separated-unseen40-v1"
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
    return root / "classification" / EXPERIMENT


def default_unseen_gold_path() -> Path:
    return (
        Path(__file__).parent / "data" / "cvo-intent-unseen-gold-v2.json"
    )


def _prior_selection_paths(root: Path) -> tuple[Path, ...]:
    return tuple(
        root / "classification" / experiment / "selection.json"
        for experiment in PRIOR_EXPERIMENTS
    )


def _selected_rows(root: Path) -> list[dict[str, Any]]:
    return select_unseen_rows(
        root,
        sample_size=SAMPLE_SIZE,
        selection_seed=SELECTION_SEED,
        extra_receipt_paths=_prior_selection_paths(root),
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


def prepare_selection(root: Path) -> dict[str, Any]:
    destination = unseen_root(root)
    selection_path = destination / "selection.json"
    if (destination / "evaluation.json").is_file():
        raise ClassificationError("CVO intent unseen evaluation is sealed")
    if selection_path.is_file():
        return read_sealed_json(selection_path)
    rows = _selected_rows(root)
    selection = {
        "schema": SELECTION_SCHEMA,
        "created_at": _now(),
        "selection_seed": SELECTION_SEED,
        "source_candidate_path": str(_candidate_path(root)),
        "source_candidate_sha256": sha256_file(_candidate_path(root)),
        "prior_selection_paths": [
            str(path) for path in _prior_selection_paths(root)
        ],
        "prior_selection_sha256s": [
            sha256_file(path) for path in _prior_selection_paths(root)
        ],
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
    write_sealed_json(selection_path, selection, backup=True)
    _write_state(
        root,
        status="prepared",
        stage="selection-sealed-before-gold",
        case_count=SAMPLE_SIZE,
        inference_calls=0,
        page_mutations=0,
    )
    return read_sealed_json(selection_path)


def lock_preregistration(root: Path, gold_path: Path) -> dict[str, Any]:
    destination = unseen_root(root)
    preregistration_path = destination / "preregistration.json"
    if preregistration_path.is_file():
        return read_sealed_json(preregistration_path)
    if (destination / "evaluation.json").is_file():
        raise ClassificationError("CVO intent unseen evaluation is sealed")
    case_root = destination / "cases"
    if case_root.exists() and any(case_root.rglob("*.json")):
        raise ClassificationError("CVO inference artifacts exist before gold lock")
    selection_path = destination / "selection.json"
    selection = read_sealed_json(selection_path)
    if selection.get("schema") != SELECTION_SCHEMA:
        raise ClassificationError("CVO intent unseen selection mismatch")
    rows = _selected_rows(root)
    rows_by_uid = {str(row.get("uid") or ""): row for row in rows}
    anchor_set = load_anchor_set()
    gold = _load_manual_gold(gold_path, anchor_set)
    if set(gold) != set(rows_by_uid):
        raise ClassificationError("CVO intent unseen gold UIDs mismatch")
    fixture_cases = []
    for selected in selection.get("cases") or []:
        if not isinstance(selected, Mapping):
            raise ClassificationError("CVO intent unseen selected case invalid")
        uid = str(selected.get("uid") or "")
        source = rows_by_uid[uid]
        if (
            str(source.get("source_sha256") or "")
            != str(selected.get("source_sha256") or "")
            or str(source.get("fixture_group_id") or "")
            != str(selected.get("fixture_group_id") or "")
        ):
            raise ClassificationError(
                "CVO intent unseen source changed before lock"
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
        "intent_lexicon_path": str(default_intent_lexicon_path()),
        "intent_lexicon_sha256": sha256_file(default_intent_lexicon_path()),
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
            raise ClassificationError(
                f"CVO intent unseen model unavailable: {spec['model']}"
            )
    lexicon = load_intent_lexicon(anchor_set=anchor_set)
    gate = {
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
    }
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
        "intent_lexicon_path": str(default_intent_lexicon_path()),
        "intent_lexicon_sha256": sha256_file(default_intent_lexicon_path()),
        "intent_lexicon_epoch": lexicon.epoch,
        "intent_lexicon_checksum": lexicon.checksum,
        "models": models,
        "core_prompt_sha256": PROMPT_SHA256,
        "gate": gate,
        "successful_model_call_budget": SAMPLE_SIZE * 2,
        "page_mutations": 0,
    }
    write_sealed_json(preregistration_path, preregistration, backup=True)
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
    destination = unseen_root(root)
    preregistration = read_sealed_json(
        destination / "preregistration.json"
    )
    expected_gate = {
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
    }
    if (
        preregistration.get("schema") != PREREGISTRATION_SCHEMA
        or preregistration.get("gate") != expected_gate
        or preregistration.get("core_prompt_sha256") != PROMPT_SHA256
    ):
        raise ClassificationError(
            "CVO intent unseen preregistration contract changed"
        )
    checks = {
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
        Path(str(preregistration["intent_lexicon_path"])): str(
            preregistration["intent_lexicon_sha256"]
        ),
    }
    for path, expected in checks.items():
        if not path.is_file() or sha256_file(path) != expected:
            raise ClassificationError(
                f"CVO intent unseen sealed input changed: {path}"
            )
    lexicon = load_intent_lexicon()
    if (
        preregistration.get("intent_lexicon_epoch") != lexicon.epoch
        or preregistration.get("intent_lexicon_checksum") != lexicon.checksum
    ):
        raise ClassificationError("CVO intent lexicon contract changed")
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
            raise ClassificationError("CVO intent unseen model digest changed")
    return preregistration


def evaluate_unseen(root: Path) -> dict[str, Any]:
    destination = unseen_root(root)
    evaluation_path = destination / "evaluation.json"
    if evaluation_path.is_file():
        return read_sealed_json(evaluation_path)
    preregistration = _validate_preregistration(root)
    fixture = read_sealed_json(destination / "fixture.json")
    anchor_set = load_anchor_set()
    lexicon = load_intent_lexicon(anchor_set=anchor_set)
    models = dict(preregistration["models"])
    config = load_decision_router_config()
    cases = []
    model_calls = 0
    for page in fixture.get("cases") or []:
        if not isinstance(page, Mapping):
            raise ClassificationError("CVO intent unseen fixture case invalid")
        core = run_case(
            root,
            page,
            anchor_set,
            extractor=dict(models["extractor"]),
            classifier=dict(models["classifier"]),
            read_timeout_ms=config.read_timeout_ms,
            experiment=EXPERIMENT,
        )
        core_anchor_id = str(
            core.get("selection", {}).get("primary_anchor_id")
            or UNRESOLVED_ANCHOR_ID
        )
        intent = classify_complement(
            page,
            core_anchor_id=core_anchor_id,
            lexicon=lexicon,
        )
        selected = [core_anchor_id]
        if intent["second_anchor_id"] != "NONE":
            selected.append(str(intent["second_anchor_id"]))
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
        model_calls += int(core.get("model_calls") or 0)
        cases.append(
            {
                "uid": str(page.get("uid") or ""),
                "title": str(page.get("title") or ""),
                "core_anchor_id": core_anchor_id,
                "subject": core.get("subject"),
                "intent": intent,
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
        "core_prompt_sha256": PROMPT_SHA256,
        "intent_lexicon_epoch": lexicon.epoch,
        "intent_lexicon_checksum": lexicon.checksum,
        "model_calls": model_calls,
        "page_mutations": 0,
        "gate": preregistration["gate"],
        "metrics": metrics,
        "cases": cases,
        "decision": (
            "qualify-cvo-intent-v2-unseen40"
            if passed
            else "reject-cvo-intent-v2-unseen40"
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
        description="Prepare, lock, or evaluate the CVO intent unseen gate"
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
