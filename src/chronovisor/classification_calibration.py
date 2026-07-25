"""CLI workflow for frozen classification fixtures and authority calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from chronovisor.classification import CALIBRATION_SCHEMA, load_udc_package
from chronovisor.classification_engine import (
    ENGINE_VERSION,
    adopt_calibration,
    build_fixture_candidates,
    evaluate_predictions,
    fixture_paths,
    lock_fixtures,
    run_consensus_batches,
)
from chronovisor.durable_state import write_sealed_json
from chronovisor.runtime_config import load_decision_router_config
from chronovisor.store import CHRONOVISOR_ROOT

DISTRIBUTION_SCHEMA = "chronovisor.classification-distribution.v1"


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def candidate_path(root: Path) -> Path:
    return root / "classification" / "fixtures" / "classification-candidates-300.jsonl"


def adjudication_path(root: Path) -> Path:
    return root / "classification" / "fixtures" / "classification-adjudication-300.jsonl"


def prepare(root: Path) -> dict[str, Any]:
    rows = build_fixture_candidates(root, count=300)
    path = candidate_path(root)
    _write_jsonl(path, rows)
    return {
        "status": "prepared",
        "path": str(path),
        "count": len(rows),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def adjudicate(root: Path, *, batch_size: int) -> dict[str, Any]:
    source = candidate_path(root)
    rows = _jsonl(source) if source.exists() else build_fixture_candidates(root)
    decisions = run_consensus_batches(
        rows,
        root=root,
        batch_size=batch_size,
        purpose="explicit",
        timeout_seconds=1_800,
        run_namespace="adjudication-epoch2-v2",
    )
    initial_by_uid = {str(row["uid"]): row for row in decisions}
    refinement_rows = [
        row
        for row in rows
        if int(initial_by_uid[str(row["uid"])].get("quorum") or 0) < 2
    ]
    if refinement_rows:
        refined = run_consensus_batches(
            refinement_rows,
            root=root,
            batch_size=batch_size,
            purpose="explicit",
            timeout_seconds=1_800,
            run_namespace="adjudication-tie-policy-v3",
        )
        initial_by_uid.update({str(row["uid"]): row for row in refined})
        decisions = [initial_by_uid[str(row["uid"])] for row in rows]
    by_uid = {str(row["uid"]): row for row in decisions}
    output = []
    holds = []
    for row in rows:
        decision = by_uid[str(row["uid"])]
        merged = {
            **row,
            "gold_primary_notation": str(decision["primary_notation"]),
            "gold_secondary_notations": list(
                decision.get("secondary_notations") or []
            ),
            "gold_rationale": str(decision.get("rationale") or ""),
            "gold_consensus_sha256": str(decision["consensus_sha256"]),
            "gold_quorum": int(decision.get("quorum") or 0),
            "gold_expected_status": (
                "proposed" if int(decision.get("quorum") or 0) >= 2 else "held"
            ),
            "gold_allowed_primary_notations": [
                str(decision["primary_notation"])
            ],
            "gold_models": {
                "primary": decision.get("primary_model"),
                "challenger": decision.get("challenger_model"),
                "tie_break": decision.get("tie_break_model"),
            },
            "adjudication_status": "accepted",
        }
        output.append(merged)
        if merged["gold_expected_status"] == "held":
            holds.append(
                {
                    "uid": merged["uid"],
                    "title": merged["title"],
                    "proposed": merged["gold_primary_notation"],
                }
            )
    path = adjudication_path(root)
    _write_jsonl(path, output)
    return {
        "status": "adjudicated",
        "path": str(path),
        "count": len(output),
        "expected_holds": holds,
        "adjudication_basis": (
            "independent local 2-of-3 consensus; no-quorum rows are locked "
            "as expected-hold safety cases"
        ),
        "model_calls_are_local_only": True,
        "refined_no_quorum_count": len(refinement_rows),
    }


def lock(root: Path) -> dict[str, Any]:
    rows = _jsonl(adjudication_path(root))
    rejected = [
        row["uid"] for row in rows if row.get("adjudication_status") != "accepted"
    ]
    if rejected:
        return {
            "status": "blocked",
            "reason": "adjudication_rejected",
            "uids": rejected,
        }
    manifest = lock_fixtures(
        root,
        rows,
        adjudicator=(
            "local-ornith-gpt-oss-gemma-independent-consensus-with-"
            "deterministic-host-validation-and-per-page-inference-isolation"
        ),
    )
    return {"status": "locked", "manifest": manifest}


def distribution(root: Path) -> dict[str, Any]:
    rows = _jsonl(adjudication_path(root))
    counts = Counter(str(row["gold_primary_notation"]) for row in rows)
    total = max(1, len(rows))
    entropy = -sum(
        (count / total) * math.log2(count / total) for count in counts.values()
    )
    ordered = counts.most_common()
    top_occupancy = {
        str(limit): sum(count for _notation, count in ordered[:limit]) / total
        for limit in (1, 5, 10)
    }
    cvo_extension_required = top_occupancy["1"] >= 0.35
    payload = {
        "schema": DISTRIBUTION_SCHEMA,
        "generated_at": _now(),
        "fixture_count": len(rows),
        "class_count": len(counts),
        "entropy_bits": entropy,
        "top_occupancy": top_occupancy,
        "top_classes": dict(ordered[:30]),
        "cvo_subject_extension_required": cvo_extension_required,
        "decision": (
            "adopt_versioned_cvo_subject_extension"
            if cvo_extension_required
            else "udcs_granularity_sufficient"
        ),
    }
    write_sealed_json(
        root / "classification" / "distribution-analysis.json",
        payload,
        backup=True,
    )
    return payload


def _config_digest() -> str:
    config = load_decision_router_config()
    payload = json.dumps(
        {
            "primary_model": config.primary_model,
            "challenger_model": config.challenger_model,
            "tie_break_model": config.tie_break_model,
            "engine": f"classification-consensus-v{ENGINE_VERSION}",
        },
        sort_keys=True,
    )
    return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()


def calibrate(root: Path) -> dict[str, Any]:
    dev_path, holdout_path, manifest_path = fixture_paths(root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["holdout"].get("opened_at"):
        calibration_path = root / "classification" / "calibration.json"
        if calibration_path.exists():
            return json.loads(calibration_path.read_text(encoding="utf-8"))
        raise RuntimeError("locked holdout was opened without a calibration artifact")
    dev = _jsonl(dev_path)
    holdout = _jsonl(holdout_path)
    dev_raw = run_consensus_batches(
        dev,
        root=root,
        batch_size=20,
        purpose="explicit",
        timeout_seconds=1_800,
        run_namespace="calibration-dev-epoch2-v2",
    )
    sweep = []
    for minimum_confidence in (0.45, 0.50, 0.55, 0.60, 0.65, 0.70):
        decisions = [
            {
                **row,
                "status": (
                    "proposed"
                    if int(row.get("quorum") or 0) >= 2
                    and float(row.get("confidence") or 0.0) >= minimum_confidence
                    else "held"
                ),
            }
            for row in dev_raw
        ]
        metrics = evaluate_predictions(dev, decisions)
        sweep.append(
            {
                "minimum_confidence": minimum_confidence,
                "metrics": metrics,
            }
        )
    eligible = [
        row
        for row in sweep
        if row["metrics"]["forced_misclassification_rate"] <= 0.01
        and row["metrics"]["expected_hold_escape_rate"] == 0.0
        and row["metrics"]["hold_rate"] <= 0.08
        and row["metrics"]["primary_assignment_rate"] >= 0.98
        and row["metrics"]["exact_match_rate"] >= 0.90
        and row["metrics"]["hierarchy_within_one_rate"] >= 0.97
    ]
    if not eligible:
        rejection = {
            "schema": CALIBRATION_SCHEMA,
            "status": "rejected",
            "reason": "dev_gate_unmet_holdout_remains_sealed",
            "package_checksum": load_udc_package(root).checksum,
            "fixture_locked": True,
            "config_digest": _config_digest(),
            "dev_sweep": sweep,
            "forced_misclassification_rate": min(
                row["metrics"]["forced_misclassification_rate"] for row in sweep
            ),
        }
        write_sealed_json(
            root / "classification" / "calibration.json",
            rejection,
            backup=True,
        )
        return rejection
    selected = min(
        eligible,
        key=lambda row: (
            row["metrics"]["hold_rate"],
            -row["minimum_confidence"],
        ),
    )
    dev_metrics = dict(selected["metrics"])
    thresholds = {
        "minimum_quorum": 2,
        "minimum_confidence": float(selected["minimum_confidence"]),
        "maximum_hold_rate": 0.08,
        "maximum_forced_misclassification_rate": 0.01,
        "maximum_expected_hold_escape_rate": 0.0,
        "minimum_exact_match_rate": 0.90,
        "minimum_hierarchy_within_one_rate": 0.97,
    }
    preregistration = {
        "schema": "chronovisor.classification-preregistration.v1",
        "registered_at": _now(),
        "authority_epoch": 1,
        "fixture_manifest_sha256": (
            "sha256:" + hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        ),
        "package_checksum": load_udc_package(root).checksum,
        "config_digest": _config_digest(),
        "thresholds": thresholds,
        "dev_metrics": dev_metrics,
        "dev_sweep": sweep,
        "holdout_opened": False,
    }
    write_sealed_json(
        root / "classification" / "calibration-preregistration.json",
        preregistration,
        backup=True,
    )
    manifest["holdout"]["opened_at"] = _now()
    manifest["holdout"]["opening_reason"] = "single_locked_authority_evaluation"
    write_sealed_json(manifest_path, manifest, backup=True)
    holdout_raw = run_consensus_batches(
        holdout,
        root=root,
        batch_size=20,
        purpose="explicit",
        timeout_seconds=1_800,
        run_namespace="calibration-holdout-epoch2-v2",
    )
    holdout_decisions = [
        {
            **row,
            "status": (
                "proposed"
                if int(row.get("quorum") or 0) >= 2
                and float(row.get("confidence") or 0.0)
                >= thresholds["minimum_confidence"]
                else "held"
            ),
        }
        for row in holdout_raw
    ]
    _write_jsonl(
        root
        / "classification"
        / "fixtures"
        / "classification-holdout-epoch2-results.jsonl",
        holdout_decisions,
    )
    holdout_metrics = evaluate_predictions(holdout, holdout_decisions)
    return adopt_calibration(
        root,
        dev_metrics=dev_metrics,
        holdout_metrics=holdout_metrics,
        config_digest=_config_digest(),
        thresholds=thresholds,
    )


def install_package(root: Path, source: Path) -> dict[str, Any]:
    package = load_udc_package(None)
    source_package = json.loads(source.read_text(encoding="utf-8"))
    if source_package.get("release") != package.release:
        from chronovisor.classification import UDCPackage

        package = UDCPackage.load(source)
    target = root / "classification" / "udc-package.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(source.read_bytes())
    installed = load_udc_package(root)
    return {
        "status": "installed",
        "path": str(target),
        "release": installed.release,
        "checksum": installed.checksum,
        "concepts": len(installed.concepts),
        "complete": installed.complete,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=(
            "install-package",
            "prepare",
            "adjudicate",
            "lock",
            "distribution",
            "calibrate",
            "all",
        ),
    )
    parser.add_argument("--root", type=Path, default=CHRONOVISOR_ROOT)
    parser.add_argument("--package", type=Path)
    parser.add_argument("--batch-size", type=int, default=20)
    args = parser.parse_args(argv)
    if args.command == "install-package":
        if args.package is None:
            parser.error("--package is required")
        result = install_package(args.root, args.package)
    elif args.command == "prepare":
        result = prepare(args.root)
    elif args.command == "adjudicate":
        result = adjudicate(args.root, batch_size=args.batch_size)
    elif args.command == "lock":
        result = lock(args.root)
    elif args.command == "distribution":
        result = distribution(args.root)
    elif args.command == "calibrate":
        result = calibrate(args.root)
    else:
        steps = [
            prepare(args.root),
            adjudicate(args.root, batch_size=args.batch_size),
        ]
        if steps[-1]["status"] == "needs_review":
            result = {"status": "blocked", "steps": steps}
        else:
            steps.extend(
                [
                    lock(args.root),
                    distribution(args.root),
                    calibrate(args.root),
                ]
            )
            result = {"status": "ok", "steps": steps}
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result.get("status") not in {"error", "blocked"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
