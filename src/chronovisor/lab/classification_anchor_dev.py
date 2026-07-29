"""Burned-forty development gate for flat Chronovisor anchor selection."""

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
    default_anchor_gold_path,
    default_anchor_set_path,
    load_anchor_set,
    validate_anchor_gold,
)
from chronovisor.classification.classification_anchor_worker import (
    PROMPT_SHA256,
    SELECTION_SCHEMA,
    SUBJECT_SCHEMA,
    WORKER_SCHEMA,
)
from chronovisor.lab.classification_fixture_set import read_jsonl, sha256_file
from chronovisor.core.durable_state import read_sealed_json, write_sealed_json
from chronovisor.research.research_scheduler import (
    research_lane,
    run_cancellable_command,
    sync_pending,
)
from chronovisor.core.runtime_config import load_decision_router_config
from chronovisor.core.store import CHRONOVISOR_ROOT

EVALUATION_SCHEMA = "chronovisor.classification-anchor-dev.v1"
CASE_SCHEMA = "chronovisor.classification-anchor-case.v1"
CALL_SCHEMA = "chronovisor.classification-anchor-call.v1"
FIXTURE_SET = "burned40"
DEV_CASES = 40
DEV_MINIMUM_EXACT = 36
DEV_MAXIMUM_HOLDS = 4
DEV_MAXIMUM_CATASTROPHIC = 0
_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")
_MARKDOWN_HEADING = re.compile(r"^\s*#{1,6}\s+")
_FRONTMATTER_FENCE = re.compile(r"^\s*---\s*$")
DEFAULT_EXPERIMENT = "cvo-anchor-v0-dev40"




def _json_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def deterministic_evidence_capsule(
    page: Mapping[str, Any],
    *,
    max_excerpt_chars: int = 1_200,
) -> dict[str, str]:
    """Retain raw correction evidence without frontmatter or duplicate headings."""

    title = str(page.get("title") or "").strip()
    summary = str(page.get("summary") or "").strip()
    lines = str(page.get("excerpt") or "").splitlines()
    in_frontmatter = False
    body_lines = []
    for line in lines:
        if _FRONTMATTER_FENCE.match(line):
            in_frontmatter = not in_frontmatter
            continue
        if in_frontmatter:
            continue
        cleaned = _MARKDOWN_HEADING.sub("", line).strip()
        if cleaned and cleaned != title:
            body_lines.append(cleaned)
    return {
        "title": title,
        "summary": summary,
        "evidence_excerpt": "\n".join(body_lines)[:max_excerpt_chars].strip(),
    }


def output_root(root: Path, experiment: str = DEFAULT_EXPERIMENT) -> Path:
    safe = _SAFE_NAME.sub("-", experiment).strip("-")
    if not safe or safe != experiment:
        raise ClassificationError("CVO anchor experiment name is unsafe")
    return root / "classification" / safe


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
                f"sealed CVO anchor call contract changed: {call_id}"
            )
        return artifact
    attempts = 0
    timeout_ms = int(payload.get("read_timeout_ms") or 660_000)
    deadline = time.monotonic() + max(60.0, timeout_ms / 1_000 + 30)
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ClassificationError(f"CVO anchor call exceeded deadline: {call_id}")
        attempts += 1
        with research_lane(
            f"anchor-{uid[:10]}-{uuid.uuid4().hex[:8]}",
            enabled=True,
            mode="on",
            purpose="explicit",
            needs_model=True,
        ) as lease:
            result = run_cancellable_command(
                [
                    sys.executable,
                    "-m",
                    "chronovisor.classification_anchor_worker",
                ],
                json.dumps(dict(payload), ensure_ascii=False, sort_keys=True),
                lease,
                timeout_seconds=remaining,
            )
        if result.status in {"cancelled", "deferred"}:
            while sync_pending():
                if time.monotonic() >= deadline:
                    raise ClassificationError(
                        f"CVO anchor foreground wait exceeded deadline: {call_id}"
                    )
                time.sleep(0.05)
            continue
        if result.status != "completed" or not isinstance(result.value, Mapping):
            raise ClassificationError(
                result.error or f"CVO anchor call failed: {call_id}"
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
            raise ClassificationError(f"CVO anchor worker contract mismatch: {call_id}")
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
    anchor_set: AnchorSet,
    *,
    extractor: Mapping[str, str],
    classifier: Mapping[str, str],
    read_timeout_ms: int,
    experiment: str = DEFAULT_EXPERIMENT,
) -> dict[str, Any]:
    uid = str(page.get("uid") or "")
    source_sha256 = str(page.get("source_sha256") or "")
    if not uid or not source_sha256:
        raise ClassificationError("CVO anchor dev page requires UID and source hash")
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
    subject = dict(extraction["result"])
    selection = _call_worker(
        root,
        uid,
        "anchor-selection",
        _model_payload(
            operation="classify",
            page=capsule,
            subject=subject,
            anchors=anchor_set.model_cards(),
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
        "anchor_epoch": anchor_set.epoch,
        "anchor_checksum": anchor_set.checksum,
        "prompt_sha256": PROMPT_SHA256,
        "capsule": capsule,
        "subject": subject,
        "selection": dict(selection["result"]),
        "model_calls": 2,
        "page_mutations": 0,
    }
    write_sealed_json(case_path, result, backup=True)
    return read_sealed_json(case_path)


def score_anchor_selection(
    anchor_set: AnchorSet,
    primary_anchor_id: str,
    secondary_anchor_ids: Sequence[str],
    expected_anchor_ids: Sequence[str],
) -> dict[str, Any]:
    expected = [value for value in expected_anchor_ids if value in anchor_set.by_id]
    secondaries = [
        value for value in secondary_anchor_ids if value in anchor_set.by_id
    ]
    exact = primary_anchor_id in expected
    held = primary_anchor_id == UNRESOLVED_ANCHOR_ID
    secondary_rescue = not exact and any(value in expected for value in secondaries)
    same_family = bool(
        not exact
        and not held
        and primary_anchor_id in anchor_set.by_id
        and any(
            anchor_set.by_id[primary_anchor_id].family
            == anchor_set.by_id[value].family
            for value in expected
        )
    )
    catastrophic = not exact and not held and not same_family
    relation = (
        "exact"
        if exact
        else "hold"
        if held
        else "related-family"
        if same_family
        else "catastrophic-family"
    )
    return {
        "relation": relation,
        "exact": exact,
        "held": held,
        "related_error": same_family,
        "catastrophic": catastrophic,
        "secondary_rescue": secondary_rescue,
    }


def _candidate_rows(root: Path) -> dict[str, dict[str, Any]]:
    path = (
        root
        / "classification"
        / "fixtures"
        / "epochs"
        / "epoch-3-library-evidence-v1"
        / "candidates.jsonl"
    )
    return {
        str(row.get("uid") or ""): row
        for row in read_jsonl(path)
        if str(row.get("uid") or "")
    }


def load_dev_cases(root: Path, fixture_set: str) -> list[dict[str, Any]]:
    if fixture_set == "early10":
        evaluation = read_sealed_json(
            root / "classification" / "query2doc-pilot" / "evaluation.json"
        )
        candidates = _candidate_rows(root)
        output = []
        for row in evaluation.get("cases") or []:
            uid = str(row.get("uid") or "")
            page = dict(candidates[uid])
            page["expected_primary_notations"] = list(
                row.get("expected_primary_notations") or []
            )
            output.append(page)
        return output
    if fixture_set == "opened30":
        fixture = read_sealed_json(
            root / "classification" / "query2doc-v2-unseen" / "fixture.json"
        )
        return [
            dict(case)
            for case in fixture.get("cases") or []
            if isinstance(case, Mapping)
        ]
    raise ClassificationError(f"unsupported CVO anchor dev fixture: {fixture_set}")


def load_burned40(root: Path) -> list[dict[str, Any]]:
    pages = [*load_dev_cases(root, "early10"), *load_dev_cases(root, "opened30")]
    uids = [str(page.get("uid") or "") for page in pages]
    if len(pages) != DEV_CASES or len(set(uids)) != DEV_CASES:
        raise ClassificationError("CVO anchor burned40 fixture is not exactly 40 unique pages")
    return pages


def run_dev(root: Path) -> dict[str, Any]:
    destination = output_root(root)
    evaluation_path = destination / "evaluation.json"
    if evaluation_path.is_file():
        return read_sealed_json(evaluation_path)
    pages = load_burned40(root)
    anchor_set = load_anchor_set()
    gold_path = default_anchor_gold_path()
    gold_payload = json.loads(gold_path.read_text(encoding="utf-8"))
    if not isinstance(gold_payload, Mapping):
        raise ClassificationError("CVO anchor gold root must be an object")
    gold = validate_anchor_gold(
        gold_payload,
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
                f"CVO anchor model is unavailable: {spec['model']}"
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
        expected = gold[uid]
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
                "uid": uid,
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
        metrics["exact"] >= DEV_MINIMUM_EXACT
        and metrics["holds"] <= DEV_MAXIMUM_HOLDS
        and metrics["catastrophic"] <= DEV_MAXIMUM_CATASTROPHIC
    )
    anchor_path = default_anchor_set_path()
    evaluation = {
        "schema": EVALUATION_SCHEMA,
        "evaluated_at": _now(),
        "fixture_set": FIXTURE_SET,
        "anchor_set_path": str(anchor_path),
        "anchor_set_sha256": sha256_file(anchor_path),
        "anchor_epoch": anchor_set.epoch,
        "anchor_checksum": anchor_set.checksum,
        "gold_path": str(gold_path),
        "gold_sha256": sha256_file(gold_path),
        "models": models,
        "prompt_sha256": PROMPT_SHA256,
        "model_calls": model_calls,
        "page_mutations": 0,
        "gate": {
            "minimum_exact": DEV_MINIMUM_EXACT,
            "maximum_holds": DEV_MAXIMUM_HOLDS,
            "maximum_catastrophic": DEV_MAXIMUM_CATASTROPHIC,
        },
        "metrics": metrics,
        "cases": cases,
        "decision": (
            "qualify-cvo-anchor-dev40"
            if passed
            else "reject-cvo-anchor-dev40"
        ),
        "new_unseen30_authorized": passed,
    }
    write_sealed_json(evaluation_path, evaluation, backup=True)
    return read_sealed_json(evaluation_path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the burned-forty flat CVO anchor development gate"
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
