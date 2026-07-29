"""Preregistered decision-only trial over the qualified Query2doc v2.2 fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from chronovisor.classification.classification import ClassificationError
from chronovisor.classification.classification_decision_worker import (
    DECISION_PROMPT_SHA256,
    DECISION_SCHEMA,
    HOLD,
    WORKER_SCHEMA,
)
from chronovisor.core import ollama
from chronovisor.core.durable_state import read_sealed_json, write_sealed_json
from chronovisor.core.runtime_config import load_decision_router_config
from chronovisor.core.store import CHRONOVISOR_ROOT
from chronovisor.core.timeutil import utc_iso_milliseconds as _now
from chronovisor.lab.classification_fixture_set import sha256_file
from chronovisor.lab.classification_profile_pilot import notation_matches
from chronovisor.lab.classification_query2doc_pilot import candidate_blind_page
from chronovisor.research.research_scheduler import (
    research_lane,
    run_cancellable_command,
    sync_pending,
)

PREREGISTRATION_SCHEMA = (
    "chronovisor.classification-query2doc-v2-2-decision-preregistration.v1"
)
ARTIFACT_SCHEMA = "chronovisor.classification-query2doc-v2-2-decision-artifact.v1"
EVALUATION_SCHEMA = "chronovisor.classification-query2doc-v2-2-decision-evaluation.v1"
MANIFEST_SCHEMA = "chronovisor.classification-query2doc-v2-2-decision-manifest.v1"
STATE_SCHEMA = "chronovisor.classification-query2doc-v2-2-decision-state.v1"
MINIMUM_CORRECT = 24
MAXIMUM_HOLDS = 6
MAXIMUM_CATASTROPHIC = 0




def trial_root(root: Path) -> Path:
    return root / "classification" / "query2doc-v2-2-decision-trial"


def retrieval_root(root: Path) -> Path:
    return root / "classification" / "query2doc-v2-2-unseen"


def decision_contract() -> dict[str, Any]:
    return {
        "version": "candidate-bounded-independent-assessment-v1",
        "worker_schema": WORKER_SCHEMA,
        "decision_schema": DECISION_SCHEMA,
        "prompt_sha256": DECISION_PROMPT_SHA256,
        "candidate_source": "sealed-query2doc-v2.2-fused-top12",
        "candidate_count": 12,
        "model_calls_per_case": 1,
        "independent_candidate_assessment": True,
        "no_match_exit": HOLD,
        "principal_class_veto": True,
        "specificity_gate": True,
        "gold_exposed_to_model": False,
        "page_mutations": 0,
    }


def decision_contract_sha256() -> str:
    payload = json.dumps(
        decision_contract(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _write_state(
    root: Path,
    *,
    status: str,
    stage: str,
    **detail: Any,
) -> dict[str, Any]:
    return write_sealed_json(
        trial_root(root) / "state.json",
        {
            "schema": STATE_SCHEMA,
            "status": status,
            "stage": stage,
            "updated_at": _now(),
            **detail,
        },
        backup=True,
    )


def preregister(root: Path) -> dict[str, Any]:
    output_root = trial_root(root)
    if (output_root / "evaluation.json").exists():
        raise ClassificationError("decision trial evaluation is already sealed")
    if any((output_root / "decisions").glob("*.json")):
        raise ClassificationError("decision artifacts exist before preregistration")
    retrieval_evaluation = read_sealed_json(
        retrieval_root(root) / "evaluation.json"
    )
    if not retrieval_evaluation.get("decision_trial_authorized"):
        raise ClassificationError("retrieval gate did not authorize a decision trial")
    fixture_path = retrieval_root(root) / "fixture.json"
    fixture = read_sealed_json(fixture_path)
    if int(fixture.get("case_count") or 0) != 30:
        raise ClassificationError("decision trial requires the sealed 30-case fixture")
    config = load_decision_router_config()
    model = config.primary_model
    model_digest = ollama.model_digests([model]).get(model, "")
    if not model_digest:
        raise ClassificationError("decision trial model digest is unavailable")
    preregistration = {
        "schema": PREREGISTRATION_SCHEMA,
        "locked_at": _now(),
        "fixture_path": str(fixture_path),
        "fixture_sha256": sha256_file(fixture_path),
        "retrieval_evaluation_path": str(
            retrieval_root(root) / "evaluation.json"
        ),
        "retrieval_evaluation_sha256": sha256_file(
            retrieval_root(root) / "evaluation.json"
        ),
        "retrieval_contract_sha256": str(
            retrieval_evaluation.get("retrieval_contract_sha256") or ""
        ),
        "decision_contract": decision_contract(),
        "decision_contract_sha256": decision_contract_sha256(),
        "model": model,
        "model_digest": model_digest,
        "successful_model_call_budget": int(fixture["case_count"]),
        "input_contract": {
            "included": [
                "uid",
                "title",
                "summary",
                "excerpt",
                "fused official candidate notation and bilingual labels",
            ],
            "forbidden": [
                "expected_primary_notations",
                "gold_rationale",
                "gold_ambiguity",
                "case_number",
                "tags",
                "raw_keywords",
                "retrieval channel scores and ranks",
            ],
        },
        "gate": {
            "minimum_correct": MINIMUM_CORRECT,
            "maximum_holds": MAXIMUM_HOLDS,
            "maximum_catastrophic": MAXIMUM_CATASTROPHIC,
            "selected_notation_must_be_in_fused_candidates": True,
        },
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
        stage="decision-trial-preregistered",
        case_count=int(fixture["case_count"]),
        generated_count=0,
        evaluated_count=0,
        model_calls=0,
        page_mutations=0,
    )
    return read_sealed_json(output_root / "preregistration.json")


def _artifact_path(root: Path, uid: str) -> Path:
    return trial_root(root) / "decisions" / f"{uid}.json"


def _candidate_hash(candidates: Sequence[Mapping[str, Any]]) -> str:
    payload = [
        {
            "notation": str(row.get("notation") or ""),
            "label_en": str(row.get("label_en") or ""),
            "label_ja": str(row.get("label_ja") or ""),
        }
        for row in candidates
    ]
    return "sha256:" + hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _generate_decision(
    root: Path,
    page: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    *,
    source_sha256: str,
    model: str,
    model_digest: str,
    keep_alive: str,
    read_timeout_ms: int,
) -> dict[str, Any]:
    projected = candidate_blind_page(page)
    uid = projected["uid"]
    path = _artifact_path(root, uid)
    candidate_sha256 = _candidate_hash(candidates)
    if path.is_file():
        artifact = read_sealed_json(path)
        if (
            artifact.get("schema") == ARTIFACT_SCHEMA
            and artifact.get("source_sha256") == source_sha256
            and artifact.get("candidate_sha256") == candidate_sha256
            and artifact.get("model") == model
            and artifact.get("model_digest") == model_digest
            and artifact.get("prompt_sha256") == DECISION_PROMPT_SHA256
        ):
            return artifact
        raise ClassificationError("stale decision artifact conflicts with trial")
    payload = {
        "schema": WORKER_SCHEMA,
        "model": model,
        "model_digest": model_digest,
        "keep_alive": keep_alive,
        "read_timeout_ms": read_timeout_ms,
        "page": projected,
        "candidates": [
            {
                "notation": str(row.get("notation") or ""),
                "label_en": str(row.get("label_en") or ""),
                "label_ja": str(row.get("label_ja") or ""),
            }
            for row in candidates
        ],
    }
    attempts = 0
    deadline = time.monotonic() + max(60.0, read_timeout_ms / 1_000 + 30)
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ClassificationError("decision generation exceeded its deadline")
        attempts += 1
        run_id = f"decision-v2-2-{uid[:12]}-{uuid.uuid4().hex[:8]}"
        with research_lane(
            run_id,
            enabled=True,
            mode="on",
            purpose="explicit",
            needs_model=True,
        ) as lease:
            result = run_cancellable_command(
                [
                    sys.executable,
                    "-m",
                    "chronovisor.classification.classification_decision_worker",
                ],
                json.dumps(payload, ensure_ascii=False, sort_keys=True),
                lease,
                timeout_seconds=remaining,
            )
        if result.status in {"cancelled", "deferred"}:
            while sync_pending():
                if time.monotonic() >= deadline:
                    raise ClassificationError(
                        "decision foreground wait exceeded deadline"
                    )
                time.sleep(0.05)
            continue
        if result.status != "completed" or not isinstance(result.value, Mapping):
            raise ClassificationError(result.error or "decision worker failed")
        worker = dict(result.value)
        decision = worker.get("decision")
        if (
            worker.get("schema") != WORKER_SCHEMA
            or worker.get("uid") != uid
            or worker.get("model") != model
            or worker.get("model_digest") != model_digest
            or worker.get("prompt_sha256") != DECISION_PROMPT_SHA256
            or int(worker.get("model_calls") or 0) != 1
            or not isinstance(decision, Mapping)
            or decision.get("schema") != DECISION_SCHEMA
        ):
            raise ClassificationError("decision worker contract mismatch")
        artifact = {
            "schema": ARTIFACT_SCHEMA,
            "created_at": _now(),
            "uid": uid,
            "source_sha256": source_sha256,
            "candidate_sha256": candidate_sha256,
            "model": model,
            "model_digest": model_digest,
            "prompt_sha256": DECISION_PROMPT_SHA256,
            "attempts": attempts,
            "model_calls": 1,
            "decision": dict(decision),
        }
        write_sealed_json(path, artifact, backup=True)
        return read_sealed_json(path)


def score_decision(
    decision: Mapping[str, Any],
    expected: Sequence[str],
) -> dict[str, Any]:
    disposition = str(decision.get("disposition") or "hold")
    notation = str(decision.get("selected_notation") or HOLD)
    held = disposition != "assign" or notation == HOLD
    correct = not held and notation_matches(notation, expected)
    expected_majors = {str(value)[:1] for value in expected if str(value)}
    catastrophic = (
        not held
        and not correct
        and bool(notation)
        and notation[:1] not in expected_majors
    )
    return {
        "held": held,
        "correct": correct,
        "incorrect": not held and not correct,
        "catastrophic": catastrophic,
    }


def decision_gate_passed(metrics: Mapping[str, Any]) -> bool:
    return (
        int(metrics.get("correct") or 0) >= MINIMUM_CORRECT
        and int(metrics.get("holds") or 0) <= MAXIMUM_HOLDS
        and int(metrics.get("catastrophic") or 0) <= MAXIMUM_CATASTROPHIC
    )


def _validate_preregistration(root: Path) -> dict[str, Any]:
    preregistration = read_sealed_json(
        trial_root(root) / "preregistration.json"
    )
    if preregistration.get("schema") != PREREGISTRATION_SCHEMA:
        raise ClassificationError("decision preregistration schema mismatch")
    if preregistration.get("decision_contract_sha256") != decision_contract_sha256():
        raise ClassificationError("decision contract changed after lock")
    if preregistration.get("decision_contract") != decision_contract():
        raise ClassificationError("decision contract body changed after lock")
    if sha256_file(Path(str(preregistration["fixture_path"]))) != str(
        preregistration.get("fixture_sha256") or ""
    ):
        raise ClassificationError("decision fixture changed after lock")
    if sha256_file(Path(str(preregistration["retrieval_evaluation_path"]))) != str(
        preregistration.get("retrieval_evaluation_sha256") or ""
    ):
        raise ClassificationError("retrieval evaluation changed after lock")
    observed_digest = ollama.model_digests(
        [str(preregistration["model"])]
    ).get(str(preregistration["model"]), "")
    if observed_digest != str(preregistration.get("model_digest") or ""):
        raise ClassificationError("decision model digest changed after lock")
    return preregistration


def evaluate(root: Path) -> dict[str, Any]:
    output_root = trial_root(root)
    evaluation_path = output_root / "evaluation.json"
    if evaluation_path.exists():
        raise ClassificationError("decision trial evaluation is already sealed")
    preregistration = _validate_preregistration(root)
    fixture = read_sealed_json(Path(str(preregistration["fixture_path"])))
    retrieval = read_sealed_json(
        Path(str(preregistration["retrieval_evaluation_path"]))
    )
    fixture_by_uid = {
        str(case.get("uid") or ""): case
        for case in fixture.get("cases") or []
        if isinstance(case, Mapping)
    }
    retrieval_cases = [
        case
        for case in retrieval.get("cases") or []
        if isinstance(case, Mapping)
    ]
    if set(fixture_by_uid) != {
        str(case.get("uid") or "") for case in retrieval_cases
    }:
        raise ClassificationError("decision fixture and retrieval cases differ")
    config = load_decision_router_config()
    cases = []
    model_calls = 0
    model_attempts = 0
    for position, retrieval_case in enumerate(retrieval_cases, start=1):
        uid = str(retrieval_case.get("uid") or "")
        source = fixture_by_uid[uid]
        candidates = [
            row
            for row in retrieval_case.get("fused_candidates") or []
            if isinstance(row, Mapping)
        ]
        if len(candidates) != 12:
            raise ClassificationError(f"decision case {uid} lacks 12 candidates")
        _write_state(
            root,
            status="running",
            stage="candidate-bounded-decisions",
            case_count=len(retrieval_cases),
            generated_count=position - 1,
            evaluated_count=position - 1,
            model_calls=model_calls,
            page_mutations=0,
        )
        artifact = _generate_decision(
            root,
            source,
            candidates,
            source_sha256=str(source.get("source_sha256") or ""),
            model=str(preregistration["model"]),
            model_digest=str(preregistration["model_digest"]),
            keep_alive=config.primary_keep_alive,
            read_timeout_ms=config.read_timeout_ms,
        )
        model_calls += int(artifact.get("model_calls") or 0)
        model_attempts += int(artifact.get("attempts") or 0)
        decision = dict(artifact.get("decision") or {})
        expected = [
            str(value)
            for value in source.get("expected_primary_notations") or []
        ]
        score = score_decision(decision, expected)
        cases.append(
            {
                "position": position,
                "uid": uid,
                "title": str(source.get("title") or ""),
                "source_sha256": str(source.get("source_sha256") or ""),
                "expected_primary_notations": expected,
                "candidate_notations": [
                    str(row.get("notation") or "") for row in candidates
                ],
                "retrieval_hit": bool(retrieval_case.get("fused_hit")),
                "decision_artifact_path": str(_artifact_path(root, uid)),
                "decision_artifact_sha256": sha256_file(
                    _artifact_path(root, uid)
                ),
                "decision": decision,
                **score,
            }
        )
    correct = sum(bool(case["correct"]) for case in cases)
    holds = sum(bool(case["held"]) for case in cases)
    incorrect = sum(bool(case["incorrect"]) for case in cases)
    catastrophic = sum(bool(case["catastrophic"]) for case in cases)
    metrics = {
        "case_count": len(cases),
        "correct": correct,
        "holds": holds,
        "incorrect": incorrect,
        "catastrophic": catastrophic,
        "accuracy": correct / len(cases),
        "hold_rate": holds / len(cases),
        "assignment_accuracy": (
            correct / (len(cases) - holds) if len(cases) > holds else 0.0
        ),
    }
    passed = decision_gate_passed(metrics)
    evaluation = {
        "schema": EVALUATION_SCHEMA,
        "evaluated_at": _now(),
        "fixture_path": str(preregistration["fixture_path"]),
        "fixture_sha256": str(preregistration["fixture_sha256"]),
        "retrieval_evaluation_path": str(
            preregistration["retrieval_evaluation_path"]
        ),
        "retrieval_evaluation_sha256": str(
            preregistration["retrieval_evaluation_sha256"]
        ),
        "preregistration_path": str(output_root / "preregistration.json"),
        "preregistration_sha256": sha256_file(
            output_root / "preregistration.json"
        ),
        "decision_contract": decision_contract(),
        "decision_contract_sha256": decision_contract_sha256(),
        "model": str(preregistration["model"]),
        "model_digest": str(preregistration["model_digest"]),
        "model_calls": model_calls,
        "model_attempts": model_attempts,
        "classification_judge_calls": 0,
        "page_mutations": 0,
        "gate": dict(preregistration["gate"]),
        "metrics": metrics,
        "cases": cases,
        "decision": (
            "qualify-candidate-bounded-decision"
            if passed
            else "reject-candidate-bounded-decision"
        ),
        "larger_decision_evaluation_authorized": passed,
        "classification_authority_activation_authorized": False,
    }
    write_sealed_json(evaluation_path, evaluation, backup=True)
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "created_at": _now(),
        "evaluation_path": str(evaluation_path),
        "evaluation_sha256": sha256_file(evaluation_path),
        "decision": evaluation["decision"],
        "larger_decision_evaluation_authorized": passed,
        "classification_authority_activation_authorized": False,
        "page_mutations": 0,
    }
    write_sealed_json(output_root / "manifest.json", manifest, backup=True)
    _write_state(
        root,
        status="qualified" if passed else "rejected",
        stage="decision-trial-complete",
        case_count=len(cases),
        generated_count=len(cases),
        evaluated_count=len(cases),
        model_calls=model_calls,
        metrics=metrics,
        decision=evaluation["decision"],
        page_mutations=0,
    )
    return read_sealed_json(evaluation_path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the preregistered Query2doc v2.2 decision-only trial"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=CHRONOVISOR_ROOT,
    )
    parser.add_argument("command", choices=("preregister", "run"))
    args = parser.parse_args(argv)
    root = args.root.expanduser()
    try:
        result = preregister(root) if args.command == "preregister" else evaluate(root)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
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
            stage=f"decision-trial-{args.command}-failed",
            error=str(exc),
            page_mutations=0,
        )
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
