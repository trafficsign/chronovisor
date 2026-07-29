"""Opened-data evaluation for the complementary retrieval-anchor audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import uuid
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from chronovisor import ollama
from chronovisor.classification import ClassificationError
from chronovisor.classification_anchor import (
    UNRESOLVED_ANCHOR_ID,
    AnchorSet,
    load_anchor_set,
)
from chronovisor.classification_anchor_complement_auditor import (
    AUDIT_SCHEMA,
    PROMPT_SHA256,
    WORKER_SCHEMA,
)
from chronovisor.classification_anchor_set_dev import (
    score_anchor_set,
    summarize_metrics,
)
from chronovisor.durable_state import read_sealed_json, write_sealed_json
from chronovisor.research_scheduler import (
    research_lane,
    run_cancellable_command,
    sync_pending,
)
from chronovisor.runtime_config import load_decision_router_config
from chronovisor.store import CHRONOVISOR_ROOT

CALL_SCHEMA = "chronovisor.classification-anchor-complement-call.v1"
EVALUATION_SCHEMA = "chronovisor.classification-anchor-complement-dev.v1"
EXPERIMENT = "cvo-anchor-set-v2-complement-dev40"
SOURCE_EXPERIMENT = "cvo-anchor-set-v1-unseen40"
EARLY_LIMIT = 15
EARLY_REQUIRED_REPAIRS = 7
EARLY_MAXIMUM_BROKEN_CONTROLS = 1


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def _model_slug(model: str) -> str:
    return "".join(character if character.isalnum() else "-" for character in model)


def output_root(root: Path, *, model_override: str | None = None) -> Path:
    experiment = (
        EXPERIMENT
        if model_override is None
        else f"{EXPERIMENT}-{_model_slug(model_override)}"
    )
    return root / "classification" / experiment


def _json_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _call_auditor(
    root: Path,
    uid: str,
    payload: Mapping[str, Any],
    *,
    model_override: str | None = None,
) -> dict[str, Any]:
    path = (
        output_root(root, model_override=model_override)
        / "cases"
        / uid
        / "complement-audit.json"
    )
    input_sha256 = _json_sha256(payload)
    if path.is_file():
        artifact = read_sealed_json(path)
        if (
            artifact.get("schema") != CALL_SCHEMA
            or artifact.get("input_sha256") != input_sha256
            or artifact.get("prompt_sha256") != PROMPT_SHA256
        ):
            raise ClassificationError(
                f"sealed complement audit contract changed: {uid}"
            )
        return artifact
    attempts = 0
    timeout_ms = int(payload.get("read_timeout_ms") or 660_000)
    deadline = time.monotonic() + max(60.0, timeout_ms / 1_000 + 30)
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ClassificationError(
                f"complement audit exceeded deadline: {uid}"
            )
        attempts += 1
        with research_lane(
            f"complement-audit-{uid[:10]}-{uuid.uuid4().hex[:8]}",
            enabled=True,
            mode="on",
            purpose="explicit",
            needs_model=True,
        ) as lease:
            result = run_cancellable_command(
                [
                    sys.executable,
                    "-m",
                    "chronovisor.classification_anchor_complement_auditor",
                ],
                json.dumps(dict(payload), ensure_ascii=False, sort_keys=True),
                lease,
                timeout_seconds=remaining,
            )
        if result.status in {"cancelled", "deferred"}:
            while sync_pending():
                if time.monotonic() >= deadline:
                    raise ClassificationError(
                        f"complement wait exceeded deadline: {uid}"
                    )
                time.sleep(0.05)
            continue
        if result.status != "completed" or not isinstance(result.value, Mapping):
            raise ClassificationError(
                result.error or f"complement auditor failed: {uid}"
            )
        worker = dict(result.value)
        audit = worker.get("result")
        if (
            worker.get("schema") != WORKER_SCHEMA
            or worker.get("model") != payload.get("model")
            or worker.get("model_digest") != payload.get("model_digest")
            or worker.get("prompt_sha256") != PROMPT_SHA256
            or int(worker.get("model_calls") or 0) != 1
            or not isinstance(audit, Mapping)
            or audit.get("schema") != AUDIT_SCHEMA
        ):
            raise ClassificationError(
                f"complement worker contract mismatch: {uid}"
            )
        artifact = {
            "schema": CALL_SCHEMA,
            "created_at": _now(),
            "uid": uid,
            "model": payload.get("model"),
            "model_digest": payload.get("model_digest"),
            "prompt_sha256": PROMPT_SHA256,
            "input_sha256": input_sha256,
            "attempts": attempts,
            "model_calls": 1,
            "audit": dict(audit),
        }
        write_sealed_json(path, artifact, backup=True)
        return read_sealed_json(path)


def _auditor_payload(
    *,
    page_result: Mapping[str, Any],
    anchor_set: AnchorSet,
    core_anchor_id: str,
    model: str,
    model_digest: str,
    keep_alive: str,
    read_timeout_ms: int,
) -> dict[str, Any]:
    core = anchor_set.by_id.get(core_anchor_id)
    if core is None or core_anchor_id == UNRESOLVED_ANCHOR_ID:
        raise ClassificationError("complement audit has no valid core anchor")
    return {
        "schema": WORKER_SCHEMA,
        "model": model,
        "model_digest": model_digest,
        "keep_alive": keep_alive,
        "read_timeout_ms": read_timeout_ms,
        "page": dict(page_result.get("capsule") or {}),
        "subject": dict(page_result.get("subject") or {}),
        "core_anchor": core.model_card(),
        "alternative_anchors": [
            anchor.model_card()
            for anchor in anchor_set.anchors
            if anchor.anchor_id != core_anchor_id
        ],
    }


def _source_paths(root: Path) -> tuple[Path, Path]:
    source_root = root / "classification" / SOURCE_EXPERIMENT
    return source_root / "evaluation.json", source_root / "cases"


def _ordered_cases(root: Path) -> list[dict[str, Any]]:
    evaluation_path, _ = _source_paths(root)
    evaluation = read_sealed_json(evaluation_path)
    rows = [
        dict(case)
        for case in evaluation.get("cases") or []
        if isinstance(case, Mapping)
    ]
    misses = [case for case in rows if not bool(case.get("exact_set"))]
    controls = [case for case in rows if bool(case.get("exact_set"))]
    return [*misses, *controls]


def run_dev(
    root: Path,
    *,
    limit: int,
    model_override: str | None = None,
) -> dict[str, Any]:
    if not 1 <= limit <= 40:
        raise ClassificationError("complement dev limit must be 1 to 40")
    evaluation_name = (
        "evaluation.json" if limit == 40 else f"evaluation-{limit}.json"
    )
    evaluation_path = (
        output_root(root, model_override=model_override) / evaluation_name
    )
    if evaluation_path.is_file():
        return read_sealed_json(evaluation_path)
    source_evaluation_path, source_cases_root = _source_paths(root)
    source_evaluation = read_sealed_json(source_evaluation_path)
    pages = _ordered_cases(root)[:limit]
    anchor_set = load_anchor_set()
    config = load_decision_router_config()
    model = model_override or config.primary_model
    keep_alive = (
        config.tie_break_keep_alive
        if model == config.tie_break_model
        else config.primary_keep_alive
    )
    model_digest = ollama.model_digests([model]).get(model, "")
    if not model_digest:
        raise ClassificationError("complement auditor model is unavailable")
    cases = []
    model_calls = 0
    for source_score in pages:
        uid = str(source_score.get("uid") or "")
        page_result = read_sealed_json(
            source_cases_root / uid / "result.json"
        )
        core_anchor_id = str(
            page_result.get("selection", {}).get("primary_anchor_id")
            or UNRESOLVED_ANCHOR_ID
        )
        if core_anchor_id == UNRESOLVED_ANCHOR_ID:
            audit = {
                "schema": AUDIT_SCHEMA,
                "second_anchor_id": "NONE",
                "different_principal_axis": False,
                "independent_retrieval_route": False,
                "explicit_document_evidence": False,
                "not_incidental_context": False,
                "admitted": False,
                "rationale": "Core classifier held the page.",
                "invalid_reason": "core_hold",
            }
            call_count = 0
        else:
            artifact = _call_auditor(
                root,
                uid,
                _auditor_payload(
                    page_result=page_result,
                    anchor_set=anchor_set,
                    core_anchor_id=core_anchor_id,
                    model=model,
                    model_digest=model_digest,
                    keep_alive=keep_alive,
                    read_timeout_ms=config.read_timeout_ms,
                ),
                model_override=model_override,
            )
            audit = dict(artifact["audit"])
            call_count = int(artifact.get("model_calls") or 0)
        selected = [core_anchor_id]
        if audit.get("admitted"):
            selected.append(str(audit["second_anchor_id"]))
        selected = sorted(dict.fromkeys(selected))
        acceptable_sets = [
            [str(value) for value in acceptable]
            for acceptable in source_score.get("acceptable_anchor_sets") or []
        ]
        acceptable_union = sorted(
            {value for acceptable in acceptable_sets for value in acceptable}
        )
        defensible = [
            str(value)
            for value in source_score.get("defensible_anchor_ids") or []
        ]
        score = score_anchor_set(
            selected,
            acceptable_union,
            defensible,
            acceptable_sets,
        )
        was_exact = bool(source_score.get("exact_set"))
        model_calls += call_count
        cases.append(
            {
                "uid": uid,
                "title": str(source_score.get("title") or ""),
                "source_was_exact": was_exact,
                "source_selected_anchor_ids": list(
                    source_score.get("selected_anchor_ids") or []
                ),
                "core_anchor_id": core_anchor_id,
                "audit": audit,
                "selected_anchor_ids": selected,
                "acceptable_anchor_sets": acceptable_sets,
                "defensible_anchor_ids": defensible,
                "repaired_source_miss": not was_exact and score["exact_set"],
                "broke_source_control": was_exact and not score["exact_set"],
                **score,
            }
        )
    metrics = summarize_metrics(cases)
    metrics["source_misses_in_sample"] = sum(
        not bool(case["source_was_exact"]) for case in cases
    )
    metrics["source_controls_in_sample"] = sum(
        bool(case["source_was_exact"]) for case in cases
    )
    metrics["repaired_source_misses"] = sum(
        bool(case["repaired_source_miss"]) for case in cases
    )
    metrics["broken_source_controls"] = sum(
        bool(case["broke_source_control"]) for case in cases
    )
    if limit == EARLY_LIMIT:
        passed = (
            metrics["repaired_source_misses"] >= EARLY_REQUIRED_REPAIRS
            and metrics["broken_source_controls"]
            <= EARLY_MAXIMUM_BROKEN_CONTROLS
            and metrics["major_errors"] == 0
        )
    else:
        passed = (
            metrics["exact_sets"] >= 36
            and metrics["semantic_coverage_cases"] >= 38
            and metrics["excess_anchor_rate"] <= 0.10
            and metrics["missing_anchor_rate"] <= 0.10
            and metrics["dual_assignment_rate"] <= 0.40
            and metrics["holds"] <= 2
            and metrics["major_errors"] == 0
        )
    evaluation = {
        "schema": EVALUATION_SCHEMA,
        "evaluated_at": _now(),
        "fixture_status": "opened-development-only",
        "source_evaluation_path": str(source_evaluation_path),
        "source_evaluation_sha256": _json_sha256(source_evaluation),
        "case_limit": limit,
        "model": model,
        "model_digest": model_digest,
        "prompt_sha256": PROMPT_SHA256,
        "model_calls": model_calls,
        "page_mutations": 0,
        "early_gate": {
            "case_order": "nine source misses then exact controls",
            "required_repairs": EARLY_REQUIRED_REPAIRS,
            "maximum_broken_controls": EARLY_MAXIMUM_BROKEN_CONTROLS,
            "maximum_major_errors": 0,
        },
        "metrics": metrics,
        "cases": cases,
        "decision": (
            "continue-complement-auditor"
            if passed
            else "kill-complement-auditor"
        ),
    }
    write_sealed_json(evaluation_path, evaluation, backup=True)
    return read_sealed_json(evaluation_path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate the complement auditor on opened data"
    )
    parser.add_argument("--root", type=Path, default=CHRONOVISOR_ROOT)
    parser.add_argument("--limit", type=int, default=EARLY_LIMIT)
    parser.add_argument("--model")
    args = parser.parse_args(argv)
    try:
        print(
            json.dumps(
                run_dev(
                    args.root.expanduser(),
                    limit=args.limit,
                    model_override=args.model,
                ),
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
