"""Development audit for conservative second-anchor admission."""

from __future__ import annotations

from chronovisor.timeutil import utc_iso_milliseconds as _now

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
from chronovisor.lab.classification_anchor_second_auditor import (
    AUDIT_SCHEMA,
    PROMPT_SHA256,
    WORKER_SCHEMA,
)
from chronovisor.lab.classification_anchor_set_dev import (
    MAXIMUM_DUAL_RATE,
    default_dev_gold_path,
    load_dev70,
    score_anchor_set,
    summarize_metrics,
    validate_set_gold,
)
from chronovisor.durable_state import read_sealed_json, write_sealed_json
from chronovisor.research_scheduler import (
    research_lane,
    run_cancellable_command,
    sync_pending,
)
from chronovisor.runtime_config import load_decision_router_config
from chronovisor.store import CHRONOVISOR_ROOT

CALL_SCHEMA = "chronovisor.classification-anchor-second-call.v1"
EVALUATION_SCHEMA = "chronovisor.classification-anchor-second-dev.v1"
EXPERIMENT = "cvo-anchor-set-v1-second-audit-dev70"




def output_root(root: Path, experiment: str = EXPERIMENT) -> Path:
    return root / "classification" / experiment


def _json_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _source_result_path(root: Path, uid: str) -> Path:
    dev_path = (
        root
        / "classification"
        / "cvo-anchor-v0-dev40"
        / "cases"
        / uid
        / "result.json"
    )
    unseen_path = (
        root
        / "classification"
        / "cvo-anchor-v0-unseen30"
        / "cases"
        / uid
        / "result.json"
    )
    if dev_path.is_file():
        return dev_path
    if unseen_path.is_file():
        return unseen_path
    raise ClassificationError(f"single-anchor source result is missing: {uid}")


def _call_auditor(
    root: Path,
    uid: str,
    payload: Mapping[str, Any],
    *,
    experiment: str = EXPERIMENT,
) -> dict[str, Any]:
    path = output_root(root, experiment) / "cases" / uid / "second-audit.json"
    input_sha256 = _json_sha256(payload)
    if path.is_file():
        artifact = read_sealed_json(path)
        if (
            artifact.get("schema") != CALL_SCHEMA
            or artifact.get("input_sha256") != input_sha256
            or artifact.get("prompt_sha256") != PROMPT_SHA256
        ):
            raise ClassificationError(
                f"sealed second-anchor audit contract changed: {uid}"
            )
        return artifact
    attempts = 0
    timeout_ms = int(payload.get("read_timeout_ms") or 660_000)
    deadline = time.monotonic() + max(60.0, timeout_ms / 1_000 + 30)
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ClassificationError(
                f"second-anchor audit exceeded deadline: {uid}"
            )
        attempts += 1
        with research_lane(
            f"second-audit-{uid[:10]}-{uuid.uuid4().hex[:8]}",
            enabled=True,
            mode="on",
            purpose="explicit",
            needs_model=True,
        ) as lease:
            result = run_cancellable_command(
                [
                    sys.executable,
                    "-m",
                    "chronovisor.lab.classification_anchor_second_auditor",
                ],
                json.dumps(dict(payload), ensure_ascii=False, sort_keys=True),
                lease,
                timeout_seconds=remaining,
            )
        if result.status in {"cancelled", "deferred"}:
            while sync_pending():
                if time.monotonic() >= deadline:
                    raise ClassificationError(
                        f"second-anchor foreground wait exceeded deadline: {uid}"
                    )
                time.sleep(0.05)
            continue
        if result.status != "completed" or not isinstance(result.value, Mapping):
            raise ClassificationError(
                result.error or f"second-anchor auditor failed: {uid}"
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
                f"second-anchor worker contract mismatch: {uid}"
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
        raise ClassificationError("second-anchor audit has no valid core anchor")
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


def run_dev(root: Path, *, limit: int) -> dict[str, Any]:
    if not 1 <= limit <= 70:
        raise ClassificationError("second-anchor dev limit must be 1 to 70")
    evaluation_name = "evaluation.json" if limit == 70 else f"evaluation-{limit}.json"
    evaluation_path = output_root(root) / evaluation_name
    if evaluation_path.is_file():
        return read_sealed_json(evaluation_path)
    pages = load_dev70(root)[:limit]
    anchor_set = load_anchor_set()
    gold_payload = json.loads(default_dev_gold_path().read_text(encoding="utf-8"))
    if not isinstance(gold_payload, Mapping):
        raise ClassificationError("second-anchor dev gold root is invalid")
    gold = validate_set_gold(
        gold_payload,
        anchor_set,
        [str(page.get("uid") or "") for page in load_dev70(root)],
    )
    config = load_decision_router_config()
    model = config.primary_model
    model_digest = ollama.model_digests([model]).get(model, "")
    if not model_digest:
        raise ClassificationError("second-anchor auditor model is unavailable")
    cases = []
    model_calls = 0
    for page in pages:
        uid = str(page.get("uid") or "")
        source_path = _source_result_path(root, uid)
        source = read_sealed_json(source_path)
        selection = dict(source.get("selection") or {})
        core_anchor_id = str(
            selection.get("primary_anchor_id") or UNRESOLVED_ANCHOR_ID
        )
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
            model_call_count = 0
        else:
            artifact = _call_auditor(
                root,
                uid,
                _auditor_payload(
                    page_result=source,
                    anchor_set=anchor_set,
                    core_anchor_id=core_anchor_id,
                    model=model,
                    model_digest=model_digest,
                    keep_alive=config.primary_keep_alive,
                    read_timeout_ms=config.read_timeout_ms,
                ),
            )
            audit = dict(artifact["audit"])
            model_call_count = int(artifact.get("model_calls") or 0)
        selected = [core_anchor_id]
        if audit.get("admitted"):
            selected.append(str(audit["second_anchor_id"]))
        selected = sorted(dict.fromkeys(selected))
        target = gold[uid]["target"]
        defensible = gold[uid]["defensible"]
        score = score_anchor_set(
            selected,
            target,
            defensible,
            gold[uid]["acceptable_sets"],
        )
        model_calls += model_call_count
        cases.append(
            {
                "uid": uid,
                "title": str(page.get("title") or ""),
                "source_result_path": str(source_path),
                "core_anchor_id": core_anchor_id,
                "audit": audit,
                "selected_anchor_ids": selected,
                "selection": {"anchor_ids": selected},
                "target_anchor_ids": target,
                "acceptable_anchor_sets": gold[uid]["acceptable_sets"],
                "defensible_anchor_ids": defensible,
                **score,
            }
        )
    metrics = summarize_metrics(cases)
    evaluation = {
        "schema": EVALUATION_SCHEMA,
        "evaluated_at": _now(),
        "fixture_set": f"opened-development-first-{limit}",
        "case_limit": limit,
        "source_contract": "cvo-anchor-v0 single primary",
        "output_contract_epoch": "cvo-anchor-set-v1",
        "auditor_model": model,
        "auditor_model_digest": model_digest,
        "auditor_prompt_sha256": PROMPT_SHA256,
        "model_calls": model_calls,
        "page_mutations": 0,
        "fixed_invariants": {
            "maximum_anchors_per_page": 2,
            "maximum_dual_assignment_rate": MAXIMUM_DUAL_RATE,
            "maximum_major_errors": 0,
            "all_four_admission_axes_required": True,
        },
        "metrics": metrics,
        "cases": cases,
        "decision": (
            "continue-second-anchor-audit-dev"
            if metrics["dual_assignment_rate"] <= MAXIMUM_DUAL_RATE
            and metrics["major_errors"] == 0
            else "reject-second-anchor-audit-dev"
        ),
    }
    write_sealed_json(evaluation_path, evaluation, backup=True)
    return read_sealed_json(evaluation_path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit conservative second-anchor admission on opened dev data"
    )
    parser.add_argument("--root", type=Path, default=CHRONOVISOR_ROOT)
    parser.add_argument("--limit", type=int, default=15)
    args = parser.parse_args(argv)
    try:
        print(
            json.dumps(
                run_dev(args.root.expanduser(), limit=args.limit),
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
