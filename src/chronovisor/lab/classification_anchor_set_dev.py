"""Opened-seventy development calibration for co-primary CVO anchor sets."""

from __future__ import annotations

from chronovisor.core.timeutil import utc_iso_milliseconds as _now

import argparse
import hashlib
import json
import re
import sys
import time
import uuid
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from chronovisor.core import ollama
from chronovisor.classification.classification import ClassificationError
from chronovisor.classification.classification_anchor import (
    UNRESOLVED_ANCHOR_ID,
    AnchorSet,
    default_anchor_set_path,
    load_anchor_set,
)
from chronovisor.lab.classification_anchor_dev import (
    deterministic_evidence_capsule,
    load_burned40,
)
from chronovisor.classification.classification_anchor_set_worker import (
    PROMPT_SHA256,
    SELECTION_SCHEMA,
    SUBJECT_SCHEMA,
    WORKER_SCHEMA,
)
from chronovisor.lab.classification_fixture_set import sha256_file
from chronovisor.core.durable_state import read_sealed_json, write_sealed_json
from chronovisor.research.research_scheduler import (
    research_lane,
    run_cancellable_command,
    sync_pending,
)
from chronovisor.core.runtime_config import load_decision_router_config
from chronovisor.core.store import CHRONOVISOR_ROOT

EVALUATION_SCHEMA = "chronovisor.classification-anchor-set-dev.v1"
CASE_SCHEMA = "chronovisor.classification-anchor-set-case.v1"
CALL_SCHEMA = "chronovisor.classification-anchor-set-call.v1"
EXPERIMENT = "cvo-anchor-set-v1-dev70"
DEV_CASES = 70
MAXIMUM_DUAL_RATE = 0.40
_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")




def default_dev_gold_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "data"
        / "cvo-anchor-set-dev-gold-v1.json"
    )


def output_root(root: Path, experiment: str = EXPERIMENT) -> Path:
    safe = _SAFE_NAME.sub("-", experiment).strip("-")
    if not safe or safe != experiment:
        raise ClassificationError("CVO anchor-set experiment name is unsafe")
    return root / "classification" / safe


def _json_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _call_path(
    root: Path,
    uid: str,
    call_id: str,
    *,
    experiment: str,
) -> Path:
    safe = _SAFE_NAME.sub("-", call_id).strip("-")[:80]
    digest = hashlib.sha256(call_id.encode()).hexdigest()[:12]
    return (
        output_root(root, experiment)
        / "cases"
        / uid
        / "calls"
        / f"{safe}-{digest}.json"
    )


def _model_payload(
    *,
    operation: str,
    model: str,
    model_digest: str,
    keep_alive: str,
    read_timeout_ms: int,
    page: Mapping[str, str],
    **extra: Any,
) -> dict[str, Any]:
    return {
        "schema": WORKER_SCHEMA,
        "operation": operation,
        "model": model,
        "model_digest": model_digest,
        "keep_alive": keep_alive,
        "read_timeout_ms": read_timeout_ms,
        "page": dict(page),
        **extra,
    }


def _call_worker(
    root: Path,
    uid: str,
    call_id: str,
    payload: Mapping[str, Any],
    *,
    experiment: str,
) -> dict[str, Any]:
    path = _call_path(root, uid, call_id, experiment=experiment)
    input_sha256 = _json_sha256(payload)
    if path.is_file():
        artifact = read_sealed_json(path)
        if (
            artifact.get("schema") != CALL_SCHEMA
            or artifact.get("input_sha256") != input_sha256
            or artifact.get("prompt_sha256") != PROMPT_SHA256
        ):
            raise ClassificationError(
                f"sealed CVO anchor-set call contract changed: {call_id}"
            )
        return artifact
    attempts = 0
    timeout_ms = int(payload.get("read_timeout_ms") or 660_000)
    deadline = time.monotonic() + max(60.0, timeout_ms / 1_000 + 30)
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ClassificationError(
                f"CVO anchor-set call exceeded deadline: {call_id}"
            )
        attempts += 1
        with research_lane(
            f"anchor-set-{uid[:10]}-{uuid.uuid4().hex[:8]}",
            enabled=True,
            mode="on",
            purpose="explicit",
            needs_model=True,
        ) as lease:
            result = run_cancellable_command(
                [
                    sys.executable,
                    "-m",
                    "chronovisor.classification_anchor_set_worker",
                ],
                json.dumps(dict(payload), ensure_ascii=False, sort_keys=True),
                lease,
                timeout_seconds=remaining,
            )
        if result.status in {"cancelled", "deferred"}:
            while sync_pending():
                if time.monotonic() >= deadline:
                    raise ClassificationError(
                        f"CVO anchor-set foreground wait exceeded deadline: {call_id}"
                    )
                time.sleep(0.05)
            continue
        if result.status != "completed" or not isinstance(result.value, Mapping):
            raise ClassificationError(
                result.error or f"CVO anchor-set call failed: {call_id}"
            )
        worker = dict(result.value)
        operation = str(payload.get("operation") or "")
        expected_schema = {
            "extract": SUBJECT_SCHEMA,
            "classify": SELECTION_SCHEMA,
        }[operation]
        worker_result = worker.get("result")
        if (
            worker.get("schema") != WORKER_SCHEMA
            or worker.get("operation") != operation
            or worker.get("model") != payload.get("model")
            or worker.get("model_digest") != payload.get("model_digest")
            or worker.get("prompt_sha256") != PROMPT_SHA256
            or int(worker.get("model_calls") or 0) != 1
            or not isinstance(worker_result, Mapping)
            or worker_result.get("schema") != expected_schema
        ):
            raise ClassificationError(
                f"CVO anchor-set worker contract mismatch: {call_id}"
            )
        artifact = {
            "schema": CALL_SCHEMA,
            "created_at": _now(),
            "uid": uid,
            "call_id": call_id,
            "operation": operation,
            "model": payload.get("model"),
            "model_digest": payload.get("model_digest"),
            "prompt_sha256": PROMPT_SHA256,
            "input_sha256": input_sha256,
            "attempts": attempts,
            "model_calls": 1,
            "result": dict(worker_result),
        }
        write_sealed_json(path, artifact, backup=True)
        return read_sealed_json(path)


def run_case(
    root: Path,
    page: Mapping[str, Any],
    anchor_set: Any,
    *,
    extractor: Mapping[str, str],
    classifier: Mapping[str, str],
    read_timeout_ms: int,
    experiment: str = EXPERIMENT,
) -> dict[str, Any]:
    uid = str(page.get("uid") or "")
    source_sha256 = str(page.get("source_sha256") or "")
    if not uid or not source_sha256:
        raise ClassificationError(
            "CVO anchor-set page requires UID and source hash"
        )
    case_path = output_root(root, experiment) / "cases" / uid / "result.json"
    if case_path.is_file():
        return read_sealed_json(case_path)
    capsule = deterministic_evidence_capsule(page)
    extraction = _call_worker(
        root,
        uid,
        "subject",
        _model_payload(
            operation="extract",
            page=capsule,
            read_timeout_ms=read_timeout_ms,
            **extractor,
        ),
        experiment=experiment,
    )
    return run_case_with_subject(
        root,
        page,
        anchor_set,
        subject=dict(extraction["result"]),
        classifier=classifier,
        read_timeout_ms=read_timeout_ms,
        experiment=experiment,
        inherited_model_calls=1,
    )


def run_case_with_subject(
    root: Path,
    page: Mapping[str, Any],
    classification_space: Any,
    *,
    subject: Mapping[str, Any],
    classifier: Mapping[str, str],
    read_timeout_ms: int,
    experiment: str,
    inherited_model_calls: int = 0,
) -> dict[str, Any]:
    """Classify against a frozen card space using a precomputed subject.

    Development arms may share the already-opened subject extraction while
    keeping their classification calls and result artifacts independent.
    Fresh sealed evaluations continue to use :func:`run_case`, which performs
    both calls inside each arm.
    """

    uid = str(page.get("uid") or "")
    source_sha256 = str(page.get("source_sha256") or "")
    if not uid or not source_sha256:
        raise ClassificationError(
            "CVO anchor-set page requires UID and source hash"
        )
    case_path = output_root(root, experiment) / "cases" / uid / "result.json"
    if case_path.is_file():
        return read_sealed_json(case_path)
    capsule = deterministic_evidence_capsule(page)
    selection = _call_worker(
        root,
        uid,
        "anchor-set-selection",
        _model_payload(
            operation="classify",
            page=capsule,
            subject=dict(subject),
            anchors=classification_space.model_cards(),
            read_timeout_ms=read_timeout_ms,
            **classifier,
        ),
        experiment=experiment,
    )
    result = {
        "schema": CASE_SCHEMA,
        "created_at": _now(),
        "uid": uid,
        "source_sha256": source_sha256,
        "anchor_epoch": classification_space.epoch,
        "anchor_checksum": classification_space.checksum,
        "output_contract_epoch": "cvo-anchor-set-v1",
        "prompt_sha256": PROMPT_SHA256,
        "capsule": capsule,
        "subject": dict(subject),
        "selection": dict(selection["result"]),
        "model_calls": inherited_model_calls + 1,
        "new_model_calls": 1,
        "page_mutations": 0,
    }
    write_sealed_json(case_path, result, backup=True)
    return read_sealed_json(case_path)


def validate_set_gold(
    payload: Mapping[str, Any],
    anchor_set: AnchorSet,
    expected_uids: Sequence[str],
    *,
    schema: str = "chronovisor.classification-anchor-set-dev-gold.v1",
) -> dict[str, dict[str, Any]]:
    if (
        payload.get("schema") != schema
        or payload.get("anchor_epoch") != anchor_set.epoch
        or payload.get("output_contract_epoch") != "cvo-anchor-set-v1"
    ):
        raise ClassificationError("CVO anchor-set gold contract mismatch")
    rows = payload.get("cases")
    if not isinstance(rows, list):
        raise ClassificationError("CVO anchor-set gold cases are missing")
    output: dict[str, dict[str, list[str]]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ClassificationError("CVO anchor-set gold case is invalid")
        uid = str(row.get("uid") or "")
        target = sorted(
            dict.fromkeys(
                str(value)
                for value in row.get("target_anchor_ids") or []
                if str(value)
            )
        )
        defensible = sorted(
            dict.fromkeys(
                str(value)
                for value in row.get("defensible_anchor_ids") or []
                if str(value)
            )
        )
        raw_acceptable_sets = row.get("acceptable_anchor_sets")
        if raw_acceptable_sets is None:
            acceptable_sets = [[value] for value in target]
            if len(target) == 2:
                acceptable_sets.append(list(target))
        elif isinstance(raw_acceptable_sets, list):
            acceptable_sets = []
            seen_sets: set[tuple[str, ...]] = set()
            for raw_set in raw_acceptable_sets:
                if not isinstance(raw_set, list):
                    raise ClassificationError(
                        "CVO anchor-set acceptable set is invalid"
                    )
                normalized = tuple(
                    sorted(
                        dict.fromkeys(
                            str(value) for value in raw_set if str(value)
                        )
                    )
                )
                if (
                    not 1 <= len(normalized) <= 2
                    or normalized in seen_sets
                ):
                    raise ClassificationError(
                        "CVO anchor-set acceptable set is incomplete"
                    )
                seen_sets.add(normalized)
                acceptable_sets.append(list(normalized))
        else:
            raise ClassificationError(
                "CVO anchor-set acceptable sets are invalid"
            )
        acceptable_union = sorted(
            {value for acceptable in acceptable_sets for value in acceptable}
        )
        if (
            not uid
            or uid in output
            or not 1 <= len(target) <= 2
            or not set(target) <= set(acceptable_union)
            or not set(acceptable_union) <= set(defensible)
            or any(value not in anchor_set.by_id for value in defensible)
            or UNRESOLVED_ANCHOR_ID in defensible
        ):
            raise ClassificationError("CVO anchor-set gold case is incomplete")
        output[uid] = {
            "target": acceptable_union,
            "defensible": defensible,
            "acceptable_sets": acceptable_sets,
        }
    if set(output) != set(expected_uids):
        raise ClassificationError("CVO anchor-set gold UIDs do not match fixture")
    return output


def score_anchor_set(
    selected_anchor_ids: Sequence[str],
    target_anchor_ids: Sequence[str],
    defensible_anchor_ids: Sequence[str],
    acceptable_anchor_sets: Sequence[Sequence[str]] | None = None,
) -> dict[str, Any]:
    selected = set(selected_anchor_ids)
    target = set(target_anchor_ids)
    defensible = set(defensible_anchor_ids)
    held = selected == {UNRESOLVED_ANCHOR_ID}
    assigned = selected - {UNRESOLVED_ANCHOR_ID}
    if acceptable_anchor_sets is None:
        acceptable_sets = [{value} for value in sorted(target)]
        if len(target) == 2:
            acceptable_sets.append(set(target))
    else:
        acceptable_sets = [
            set(values) for values in acceptable_anchor_sets if values
        ]
        if not acceptable_sets:
            raise ClassificationError("CVO score has no acceptable anchor set")
    nearest = min(
        acceptable_sets,
        key=lambda candidate: (
            len(assigned ^ candidate),
            len(candidate - assigned),
            len(candidate),
        ),
    )
    exact = assigned in acceptable_sets and not held
    overlap = assigned & target
    partial = bool(overlap) and not exact
    excess = assigned - nearest
    missing = nearest - assigned
    indefensible = assigned - defensible
    major_error = bool(indefensible)
    return {
        "acceptable_anchor_sets": [
            sorted(candidate)
            for candidate in sorted(
                acceptable_sets,
                key=lambda candidate: (len(candidate), sorted(candidate)),
            )
        ],
        "nearest_acceptable_anchor_set": sorted(nearest),
        "exact_set": exact,
        "partial_set": partial,
        "target_miss": not overlap and not held,
        "held": held,
        "dual_assigned": len(assigned) == 2,
        "excess_anchor_ids": sorted(excess),
        "missing_anchor_ids": sorted(missing),
        "indefensible_anchor_ids": sorted(indefensible),
        "major_error": major_error,
        "semantic_coverage": bool(overlap) and not major_error,
    }


def load_dev70(root: Path) -> list[dict[str, Any]]:
    opened_unseen = read_sealed_json(
        root / "classification" / "cvo-anchor-v0-unseen30" / "fixture.json"
    )
    pages = [
        *load_burned40(root),
        *[
            dict(row)
            for row in opened_unseen.get("cases") or []
            if isinstance(row, Mapping)
        ],
    ]
    uids = [str(page.get("uid") or "") for page in pages]
    if len(pages) != DEV_CASES or len(set(uids)) != DEV_CASES:
        raise ClassificationError(
            "CVO anchor-set dev fixture is not exactly 70 unique pages"
        )
    return pages


def summarize_metrics(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def selected_anchor_ids(case: Mapping[str, Any]) -> set[str]:
        direct = case.get("selected_anchor_ids")
        if isinstance(direct, Sequence) and not isinstance(direct, str):
            values = direct
        else:
            selection = case.get("selection")
            values = (
                selection.get("anchor_ids") or []
                if isinstance(selection, Mapping)
                else []
            )
        return {
            str(value)
            for value in values
            if str(value) != UNRESOLVED_ANCHOR_ID
        }

    selected_assignment_count = sum(
        len(selected_anchor_ids(case)) for case in cases
    )
    target_assignment_count = sum(
        len(case.get("nearest_acceptable_anchor_set") or []) for case in cases
    )
    excess_count = sum(
        len(case.get("excess_anchor_ids") or []) for case in cases
    )
    missing_count = sum(
        len(case.get("missing_anchor_ids") or []) for case in cases
    )
    count = len(cases)
    metrics = {
        "case_count": count,
        "exact_sets": sum(bool(case.get("exact_set")) for case in cases),
        "partial_sets": sum(bool(case.get("partial_set")) for case in cases),
        "target_misses": sum(bool(case.get("target_miss")) for case in cases),
        "semantic_coverage_cases": sum(
            bool(case.get("semantic_coverage")) for case in cases
        ),
        "holds": sum(bool(case.get("held")) for case in cases),
        "dual_assignments": sum(
            bool(case.get("dual_assigned")) for case in cases
        ),
        "major_errors": sum(
            bool(case.get("major_error")) for case in cases
        ),
        "excess_anchor_count": excess_count,
        "missing_anchor_count": missing_count,
        "selected_anchor_count": selected_assignment_count,
        "target_anchor_count": target_assignment_count,
    }
    metrics.update(
        {
            "exact_set_rate": round(metrics["exact_sets"] / max(1, count), 4),
            "semantic_coverage_rate": round(
                metrics["semantic_coverage_cases"] / max(1, count), 4
            ),
            "dual_assignment_rate": round(
                metrics["dual_assignments"] / max(1, count), 4
            ),
            "excess_anchor_rate": round(
                excess_count / max(1, selected_assignment_count), 4
            ),
            "missing_anchor_rate": round(
                missing_count / max(1, target_assignment_count), 4
            ),
        }
    )
    return metrics


def run_dev(root: Path) -> dict[str, Any]:
    destination = output_root(root)
    evaluation_path = destination / "evaluation.json"
    if evaluation_path.is_file():
        return read_sealed_json(evaluation_path)
    pages = load_dev70(root)
    anchor_set = load_anchor_set()
    gold_path = default_dev_gold_path()
    payload = json.loads(gold_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ClassificationError("CVO anchor-set gold root must be an object")
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
            raise ClassificationError(
                f"CVO anchor-set model is unavailable: {spec['model']}"
            )
    cases = []
    model_calls = 0
    for page in pages:
        uid = str(page.get("uid") or "")
        result = run_case(
            root,
            page,
            anchor_set,
            extractor=models["extractor"],
            classifier=models["classifier"],
            read_timeout_ms=config.read_timeout_ms,
        )
        selection = dict(result.get("selection") or {})
        target = gold[uid]["target"]
        defensible = gold[uid]["defensible"]
        score = score_anchor_set(
            [
                str(value)
                for value in selection.get("anchor_ids") or []
            ],
            target,
            defensible,
            gold[uid]["acceptable_sets"],
        )
        model_calls += int(result.get("model_calls") or 0)
        cases.append(
            {
                "uid": uid,
                "title": str(page.get("title") or ""),
                "target_anchor_ids": target,
                "acceptable_anchor_sets": gold[uid]["acceptable_sets"],
                "defensible_anchor_ids": defensible,
                "subject": result.get("subject"),
                "selection": selection,
                **score,
            }
        )
    metrics = summarize_metrics(cases)
    evaluation = {
        "schema": EVALUATION_SCHEMA,
        "evaluated_at": _now(),
        "fixture_set": "opened70-development-only",
        "anchor_set_path": str(default_anchor_set_path()),
        "anchor_set_sha256": sha256_file(default_anchor_set_path()),
        "anchor_epoch": anchor_set.epoch,
        "anchor_checksum": anchor_set.checksum,
        "output_contract_epoch": "cvo-anchor-set-v1",
        "gold_path": str(gold_path),
        "gold_sha256": sha256_file(gold_path),
        "models": models,
        "prompt_sha256": PROMPT_SHA256,
        "model_calls": model_calls,
        "page_mutations": 0,
        "fixed_invariants": {
            "maximum_anchors_per_page": 2,
            "maximum_dual_assignment_rate": MAXIMUM_DUAL_RATE,
            "maximum_major_errors": 0,
            "major_error_definition": (
                "any assigned anchor outside the predeclared independently "
                "defensible set"
            ),
        },
        "metrics": metrics,
        "cases": cases,
        "decision": "calibrate-anchor-set-v1-unseen-gates",
    }
    write_sealed_json(evaluation_path, evaluation, backup=True)
    return read_sealed_json(evaluation_path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run opened-seventy CVO anchor-set development calibration"
    )
    parser.add_argument("--root", type=Path, default=CHRONOVISOR_ROOT)
    args = parser.parse_args(argv)
    try:
        print(
            json.dumps(
                run_dev(args.root.expanduser()),
                ensure_ascii=False,
                sort_keys=True,
            )
        )
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
