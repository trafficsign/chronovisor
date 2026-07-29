"""Ten-case development probe for direct Query2doc candidate decisions."""

from __future__ import annotations

import argparse
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
from chronovisor.classification_decision_trial import score_decision
from chronovisor.classification_direct_decision_worker import (
    DECISION_SCHEMA,
    DIRECT_PROMPT_SHA256,
    WORKER_SCHEMA,
)
from chronovisor.classification_fixture_set import sha256_file
from chronovisor.classification_query2doc_pilot import candidate_blind_page
from chronovisor.durable_state import read_sealed_json, write_sealed_json
from chronovisor.research_scheduler import (
    research_lane,
    run_cancellable_command,
    sync_pending,
)
from chronovisor.runtime_config import load_decision_router_config
from chronovisor.store import CHRONOVISOR_ROOT

EVALUATION_SCHEMA = "chronovisor.classification-direct-decision-dev.v2"
ARTIFACT_SCHEMA = "chronovisor.classification-direct-decision-artifact.v2"
DEV_CASES = 10
DEV_TARGET_CORRECT = 7
DEV_MAX_HOLDS = 3
DEV_MAX_CATASTROPHIC = 0


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def dev_root(root: Path, model_role: str) -> Path:
    return (
        root
        / "classification"
        / f"query2doc-v2-2-direct-decision-v2-{model_role}-dev"
    )


def _generate(
    root: Path,
    page: Mapping[str, Any],
    retrieval_case: Mapping[str, Any],
    *,
    model_role: str,
    model: str,
    model_digest: str,
    keep_alive: str,
    read_timeout_ms: int,
) -> dict[str, Any]:
    projected = candidate_blind_page(page)
    uid = projected["uid"]
    path = dev_root(root, model_role) / "decisions" / f"{uid}.json"
    if path.is_file():
        return read_sealed_json(path)
    candidates = [
        {
            "notation": str(row.get("notation") or ""),
            "label_en": str(row.get("label_en") or ""),
            "label_ja": str(row.get("label_ja") or ""),
        }
        for row in retrieval_case.get("fused_candidates") or []
        if isinstance(row, Mapping)
    ]
    headings = [
        value.strip()
        for value in str(
            (retrieval_case.get("broad_query_page") or {}).get("title") or ""
        ).splitlines()
        if value.strip()
    ]
    payload = {
        "schema": WORKER_SCHEMA,
        "model": model,
        "model_digest": model_digest,
        "keep_alive": keep_alive,
        "read_timeout_ms": read_timeout_ms,
        "page": projected,
        "subject_headings": headings,
        "candidates": candidates,
    }
    attempts = 0
    deadline = time.monotonic() + max(60.0, read_timeout_ms / 1_000 + 30)
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ClassificationError("direct decision dev exceeded its deadline")
        attempts += 1
        with research_lane(
            f"direct-dev-{uid[:12]}-{uuid.uuid4().hex[:8]}",
            enabled=True,
            mode="on",
            purpose="explicit",
            needs_model=True,
        ) as lease:
            result = run_cancellable_command(
                [
                    sys.executable,
                    "-m",
                    "chronovisor.classification_direct_decision_worker",
                ],
                json.dumps(payload, ensure_ascii=False, sort_keys=True),
                lease,
                timeout_seconds=remaining,
            )
        if result.status in {"cancelled", "deferred"}:
            while sync_pending():
                if time.monotonic() >= deadline:
                    raise ClassificationError(
                        "direct decision dev foreground wait exceeded deadline"
                    )
                time.sleep(0.05)
            continue
        if result.status != "completed" or not isinstance(result.value, Mapping):
            raise ClassificationError(result.error or "direct decision worker failed")
        worker = dict(result.value)
        decision = worker.get("decision")
        if (
            worker.get("schema") != WORKER_SCHEMA
            or worker.get("uid") != uid
            or worker.get("model") != model
            or worker.get("model_digest") != model_digest
            or worker.get("prompt_sha256") != DIRECT_PROMPT_SHA256
            or int(worker.get("model_calls") or 0) != 1
            or not isinstance(decision, Mapping)
            or decision.get("schema") != DECISION_SCHEMA
        ):
            raise ClassificationError("direct decision worker contract mismatch")
        artifact = {
            "schema": ARTIFACT_SCHEMA,
            "created_at": _now(),
            "uid": uid,
            "source_sha256": str(page.get("source_sha256") or ""),
            "model": model,
            "model_digest": model_digest,
            "prompt_sha256": DIRECT_PROMPT_SHA256,
            "attempts": attempts,
            "model_calls": 1,
            "subject_headings": headings,
            "candidate_notations": [
                str(row.get("notation") or "") for row in candidates
            ],
            "decision": dict(decision),
        }
        write_sealed_json(path, artifact, backup=True)
        return read_sealed_json(path)


def run_dev(root: Path, *, model_role: str = "primary") -> dict[str, Any]:
    output_root = dev_root(root, model_role)
    evaluation_path = output_root / "evaluation.json"
    if evaluation_path.exists():
        return read_sealed_json(evaluation_path)
    retrieval_path = (
        root
        / "classification"
        / "query2doc-v2-2-unseen"
        / "evaluation.json"
    )
    fixture_path = (
        root / "classification" / "query2doc-v2-2-unseen" / "fixture.json"
    )
    retrieval = read_sealed_json(retrieval_path)
    fixture = read_sealed_json(fixture_path)
    fixture_by_uid = {
        str(case.get("uid") or ""): case
        for case in fixture.get("cases") or []
        if isinstance(case, Mapping)
    }
    selected = [
        case
        for case in retrieval.get("cases") or []
        if isinstance(case, Mapping)
    ][:DEV_CASES]
    config = load_decision_router_config()
    model_settings = {
        "primary": (config.primary_model, config.primary_keep_alive),
        "challenger": (config.challenger_model, config.challenger_keep_alive),
        "tie_break": (config.tie_break_model, config.tie_break_keep_alive),
    }
    if model_role not in model_settings:
        raise ClassificationError(f"unsupported direct decision role: {model_role}")
    model, keep_alive = model_settings[model_role]
    model_digest = ollama.model_digests([model]).get(model, "")
    if not model_digest:
        raise ClassificationError("direct decision model digest is unavailable")
    cases = []
    model_calls = 0
    attempts = 0
    for retrieval_case in selected:
        uid = str(retrieval_case.get("uid") or "")
        source = fixture_by_uid[uid]
        artifact = _generate(
            root,
            source,
            retrieval_case,
            model_role=model_role,
            model=model,
            model_digest=model_digest,
            keep_alive=keep_alive,
            read_timeout_ms=config.read_timeout_ms,
        )
        model_calls += int(artifact.get("model_calls") or 0)
        attempts += int(artifact.get("attempts") or 0)
        decision = dict(artifact.get("decision") or {})
        expected = [
            str(value)
            for value in source.get("expected_primary_notations") or []
        ]
        cases.append(
            {
                "uid": uid,
                "title": str(source.get("title") or ""),
                "expected_primary_notations": expected,
                "candidate_notations": list(artifact["candidate_notations"]),
                "decision": decision,
                **score_decision(decision, expected),
            }
        )
    metrics = {
        "case_count": len(cases),
        "correct": sum(bool(case["correct"]) for case in cases),
        "holds": sum(bool(case["held"]) for case in cases),
        "incorrect": sum(bool(case["incorrect"]) for case in cases),
        "catastrophic": sum(bool(case["catastrophic"]) for case in cases),
    }
    passed = (
        metrics["correct"] >= DEV_TARGET_CORRECT
        and metrics["holds"] <= DEV_MAX_HOLDS
        and metrics["catastrophic"] <= DEV_MAX_CATASTROPHIC
    )
    evaluation = {
        "schema": EVALUATION_SCHEMA,
        "evaluated_at": _now(),
        "retrieval_evaluation_path": str(retrieval_path),
        "retrieval_evaluation_sha256": sha256_file(retrieval_path),
        "fixture_path": str(fixture_path),
        "fixture_sha256": sha256_file(fixture_path),
        "model": model,
        "model_role": model_role,
        "model_digest": model_digest,
        "prompt_sha256": DIRECT_PROMPT_SHA256,
        "model_calls": model_calls,
        "model_attempts": attempts,
        "page_mutations": 0,
        "gate": {
            "minimum_correct": DEV_TARGET_CORRECT,
            "maximum_holds": DEV_MAX_HOLDS,
            "maximum_catastrophic": DEV_MAX_CATASTROPHIC,
        },
        "metrics": metrics,
        "cases": cases,
        "decision": (
            "qualify-direct-decision-dev"
            if passed
            else "reject-direct-decision-dev"
        ),
        "new_unseen_decision_trial_authorized": passed,
    }
    write_sealed_json(evaluation_path, evaluation, backup=True)
    return read_sealed_json(evaluation_path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the ten-case direct decision development probe"
    )
    parser.add_argument("--root", type=Path, default=CHRONOVISOR_ROOT)
    parser.add_argument(
        "--model-role",
        choices=("primary", "challenger", "tie_break"),
        default="primary",
    )
    args = parser.parse_args(argv)
    try:
        print(
            json.dumps(
                run_dev(args.root.expanduser(), model_role=args.model_role),
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
