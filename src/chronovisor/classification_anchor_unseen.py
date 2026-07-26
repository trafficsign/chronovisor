"""Preregistered group-separated unseen gate for CVO anchor selection."""

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
from chronovisor.classification_anchor_dev import (
    run_case,
    score_anchor_selection,
)
from chronovisor.classification_anchor_worker import PROMPT_SHA256
from chronovisor.classification_fixture_set import read_jsonl, sha256_file
from chronovisor.durable_state import read_sealed_json, write_sealed_json
from chronovisor.runtime_config import load_decision_router_config
from chronovisor.store import CHRONOVISOR_ROOT

SELECTION_SCHEMA = "chronovisor.classification-anchor-unseen-selection.v1"
GOLD_SCHEMA = "chronovisor.classification-anchor-unseen-gold.v1"
FIXTURE_SCHEMA = "chronovisor.classification-anchor-unseen-fixture.v1"
PREREGISTRATION_SCHEMA = (
    "chronovisor.classification-anchor-unseen-preregistration.v1"
)
EVALUATION_SCHEMA = "chronovisor.classification-anchor-unseen-evaluation.v1"
STATE_SCHEMA = "chronovisor.classification-anchor-unseen-state.v1"
EXPERIMENT = "cvo-anchor-v0-unseen30"
SELECTION_SEED = "cvo-anchor-v0-group-separated-unseen30-v1"
FIXTURE_EPOCH = "epoch-3-library-evidence-v1"
SAMPLE_SIZE = 30
MINIMUM_EXACT = 27
MAXIMUM_HOLDS = 3
MAXIMUM_CATASTROPHIC = 0


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def unseen_root(root: Path) -> Path:
    return root / "classification" / EXPERIMENT


def default_unseen_gold_path() -> Path:
    return Path(__file__).parent / "data" / "cvo-anchor-unseen-gold-v0.json"


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


def _selection_key(row: Mapping[str, Any]) -> tuple[str, str]:
    uid = str(row.get("uid") or "")
    group_id = str(row.get("fixture_group_id") or "")
    return (
        hashlib.sha256(f"{SELECTION_SEED}|{group_id}|{uid}".encode()).hexdigest(),
        uid,
    )


def _seen_uids(root: Path) -> set[str]:
    seen: set[str] = set()
    receipt_paths = (
        root / "classification" / "query2doc-pilot" / "evaluation.json",
        root / "classification" / "query2doc-unseen" / "selection.json",
        root / "classification" / "query2doc-v2-unseen" / "selection.json",
        root / "classification" / "query2doc-v2-2-unseen" / "selection.json",
        root / "classification" / "query2doc-v2-3-unseen" / "selection.json",
    )
    for path in receipt_paths:
        if not path.is_file():
            continue
        receipt = read_sealed_json(path)
        seen.update(
            str(case.get("uid") or "")
            for case in receipt.get("cases") or []
            if isinstance(case, Mapping) and str(case.get("uid") or "")
        )
    adjudication = _adjudication_path(root)
    if adjudication.is_file():
        seen.update(
            str(row.get("uid") or "")
            for row in read_jsonl(adjudication)
            if str(row.get("uid") or "")
        )
    dev_gold = json.loads(default_anchor_gold_path().read_text(encoding="utf-8"))
    seen.update(
        str(row.get("uid") or "")
        for row in dev_gold.get("cases") or []
        if isinstance(row, Mapping) and str(row.get("uid") or "")
    )
    return seen


def select_unseen_rows(
    root: Path,
    *,
    sample_size: int = SAMPLE_SIZE,
) -> list[dict[str, Any]]:
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
        raise ClassificationError("CVO unseen evaluation is already sealed")
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
        stage="selection-sealed-before-gold",
        case_count=SAMPLE_SIZE,
        inference_calls=0,
        page_mutations=0,
    )
    return read_sealed_json(selection_path)


def _load_manual_gold(path: Path, anchor_set: AnchorSet) -> dict[str, list[str]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ClassificationError(f"CVO unseen gold is unreadable: {exc}") from exc
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema") != GOLD_SCHEMA
        or payload.get("anchor_epoch") != anchor_set.epoch
        or payload.get("fixture_status") != "sealed-unseen-before-inference"
    ):
        raise ClassificationError("CVO unseen gold contract mismatch")
    rows = payload.get("cases")
    if not isinstance(rows, list):
        raise ClassificationError("CVO unseen gold cases are missing")
    output: dict[str, list[str]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ClassificationError("CVO unseen gold case is invalid")
        uid = str(row.get("uid") or "")
        expected = list(
            dict.fromkeys(
                str(value)
                for value in row.get("expected_primary_anchor_ids") or []
                if str(value)
            )
        )
        if (
            not uid
            or uid in output
            or not expected
            or any(value not in anchor_set.by_id for value in expected)
        ):
            raise ClassificationError("CVO unseen gold case is incomplete")
        output[uid] = expected
    return output


def lock_preregistration(root: Path, gold_path: Path) -> dict[str, Any]:
    destination = unseen_root(root)
    preregistration_path = destination / "preregistration.json"
    if preregistration_path.is_file():
        return read_sealed_json(preregistration_path)
    if (destination / "evaluation.json").is_file():
        raise ClassificationError("CVO unseen evaluation is already sealed")
    case_root = destination / "cases"
    if case_root.exists() and any(case_root.rglob("*.json")):
        raise ClassificationError("CVO inference artifacts exist before gold lock")
    selection_path = destination / "selection.json"
    selection = read_sealed_json(selection_path)
    if selection.get("schema") != SELECTION_SCHEMA:
        raise ClassificationError("CVO unseen selection contract mismatch")
    selected_rows = select_unseen_rows(root)
    selected_by_uid = {
        str(row.get("uid") or ""): row for row in selected_rows
    }
    anchor_set = load_anchor_set()
    gold = _load_manual_gold(gold_path, anchor_set)
    if set(gold) != set(selected_by_uid):
        raise ClassificationError("CVO unseen gold UIDs do not match selection")
    fixture_cases = []
    for selected in selection.get("cases") or []:
        if not isinstance(selected, Mapping):
            raise ClassificationError("CVO unseen selection case is invalid")
        uid = str(selected.get("uid") or "")
        source = selected_by_uid[uid]
        if (
            str(source.get("source_sha256") or "")
            != str(selected.get("source_sha256") or "")
            or str(source.get("fixture_group_id") or "")
            != str(selected.get("fixture_group_id") or "")
        ):
            raise ClassificationError("CVO unseen source changed before lock")
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
                "expected_primary_anchor_ids": gold[uid],
                "gold_basis": "Codex semantic review before any unseen inference call",
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
    }
    digests = ollama.model_digests(
        sorted({str(spec["model"]) for spec in models.values()})
    )
    for spec in models.values():
        spec["model_digest"] = digests.get(str(spec["model"]), "")
        if not spec["model_digest"]:
            raise ClassificationError(
                f"CVO unseen model is unavailable: {spec['model']}"
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
        "models": models,
        "prompt_sha256": PROMPT_SHA256,
        "gate": {
            "minimum_exact": MINIMUM_EXACT,
            "maximum_holds": MAXIMUM_HOLDS,
            "maximum_catastrophic": MAXIMUM_CATASTROPHIC,
        },
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
    preregistration = read_sealed_json(destination / "preregistration.json")
    if preregistration.get("schema") != PREREGISTRATION_SCHEMA:
        raise ClassificationError("CVO unseen preregistration schema mismatch")
    checks = {
        "selection_sha256": destination / "selection.json",
        "fixture_sha256": destination / "fixture.json",
        "gold_sha256": Path(str(preregistration.get("gold_path") or "")),
        "anchor_set_sha256": default_anchor_set_path(),
    }
    for key, path in checks.items():
        if not path.is_file() or preregistration.get(key) != sha256_file(path):
            raise ClassificationError(f"CVO unseen locked artifact changed: {key}")
    if preregistration.get("prompt_sha256") != PROMPT_SHA256:
        raise ClassificationError("CVO unseen prompt changed after lock")
    config = load_decision_router_config()
    live_models = {
        "extractor": config.primary_model,
        "classifier": config.tie_break_model,
    }
    digests = ollama.model_digests(sorted(set(live_models.values())))
    for role, model in live_models.items():
        spec = (preregistration.get("models") or {}).get(role) or {}
        if (
            spec.get("model") != model
            or spec.get("model_digest") != digests.get(model, "")
        ):
            raise ClassificationError(f"CVO unseen {role} changed after lock")
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
            raise ClassificationError("CVO unseen fixture case is invalid")
        result = run_case(
            root,
            page,
            anchor_set,
            extractor=dict(models["extractor"]),
            classifier=dict(models["classifier"]),
            read_timeout_ms=config.read_timeout_ms,
            experiment=EXPERIMENT,
        )
        selection = dict(result.get("selection") or {})
        expected = [
            str(value)
            for value in page.get("expected_primary_anchor_ids") or []
        ]
        score = score_anchor_selection(
            anchor_set,
            str(selection.get("primary_anchor_id") or UNRESOLVED_ANCHOR_ID),
            [
                str(value)
                for value in selection.get("secondary_anchor_ids") or []
            ],
            expected,
        )
        model_calls += int(result.get("model_calls") or 0)
        cases.append(
            {
                "uid": str(page.get("uid") or ""),
                "title": str(page.get("title") or ""),
                "expected_primary_anchor_ids": expected,
                "subject": result.get("subject"),
                "selection": selection,
                **score,
            }
        )
    metrics = {
        "case_count": len(cases),
        "exact": sum(bool(case["exact"]) for case in cases),
        "holds": sum(bool(case["held"]) for case in cases),
        "related_errors": sum(bool(case["related_error"]) for case in cases),
        "catastrophic": sum(bool(case["catastrophic"]) for case in cases),
        "secondary_rescues": sum(
            bool(case["secondary_rescue"]) for case in cases
        ),
    }
    metrics["exact_rate"] = round(metrics["exact"] / max(1, len(cases)), 4)
    passed = (
        len(cases) == SAMPLE_SIZE
        and metrics["exact"] >= MINIMUM_EXACT
        and metrics["holds"] <= MAXIMUM_HOLDS
        and metrics["catastrophic"] <= MAXIMUM_CATASTROPHIC
    )
    evaluation = {
        "schema": EVALUATION_SCHEMA,
        "evaluated_at": _now(),
        "preregistration_path": str(destination / "preregistration.json"),
        "preregistration_sha256": sha256_file(
            destination / "preregistration.json"
        ),
        "models": models,
        "anchor_epoch": anchor_set.epoch,
        "anchor_checksum": anchor_set.checksum,
        "prompt_sha256": PROMPT_SHA256,
        "model_calls": model_calls,
        "page_mutations": 0,
        "gate": preregistration["gate"],
        "metrics": metrics,
        "cases": cases,
        "decision": (
            "qualify-cvo-anchor-unseen30"
            if passed
            else "reject-cvo-anchor-unseen30"
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
        description="Prepare, lock, or run the unseen CVO anchor gate"
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
