"""CLI workflow for frozen classification fixtures and authority calibration."""

from __future__ import annotations

from chronovisor.core.jsonl import write_jsonl as _write_jsonl

from chronovisor.core.timeutil import utc_iso_milliseconds as _now

import argparse
import hashlib
import json
import math
import shutil
from collections import Counter
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from chronovisor.classification.classification import CALIBRATION_SCHEMA, load_udc_package
from chronovisor.classification.classification_engine import (
    ENGINE_VERSION,
    adopt_calibration,
    build_fixture_candidates,
    evaluate_predictions,
    fixture_paths,
    lock_fixtures,
    run_consensus_batches,
)
from chronovisor.lab.classification_fixture_set import load_fixture_set
from chronovisor.core.durable_state import read_sealed_json, write_sealed_json
from chronovisor.core.runtime_config import load_decision_router_config
from chronovisor.core.store import CHRONOVISOR_ROOT

DISTRIBUTION_SCHEMA = "chronovisor.classification-distribution.v1"
DEV_AUDIT_SCHEMA = "chronovisor.classification-dev-audit.v1"
DEV_AUDIT_RECEIPT_SCHEMA = "chronovisor.classification-dev-audit-receipt.v1"
PREREGISTRATION_SCHEMA = "chronovisor.classification-preregistration.v1"




def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]




def candidate_path(root: Path) -> Path:
    return root / "classification" / "fixtures" / "classification-candidates-300.jsonl"


def adjudication_path(root: Path) -> Path:
    return (
        root / "classification" / "fixtures" / "classification-adjudication-300.jsonl"
    )


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
            "gold_secondary_notations": list(decision.get("secondary_notations") or []),
            "gold_rationale": str(decision.get("rationale") or ""),
            "gold_consensus_sha256": str(decision["consensus_sha256"]),
            "gold_quorum": int(decision.get("quorum") or 0),
            "gold_expected_status": (
                "proposed" if int(decision.get("quorum") or 0) >= 2 else "held"
            ),
            "gold_allowed_primary_notations": [str(decision["primary_notation"])],
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


def apply_dev_audit(root: Path, audit_path: Path) -> dict[str, Any]:
    """Apply reviewed dev-label corrections without opening locked Holdout."""

    dev_path, _holdout_path, manifest_path = fixture_paths(root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["holdout"].get("opened_at"):
        raise RuntimeError("dev audit is forbidden after Holdout opening")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("schema") != DEV_AUDIT_SCHEMA:
        raise RuntimeError("unsupported dev audit schema")
    audit_id = str(audit.get("audit_id") or "")
    if not audit_id or any(
        value not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for value in audit_id
    ):
        raise RuntimeError("dev audit requires a safe audit_id")
    corrections = audit.get("corrections")
    if not isinstance(corrections, list) or not corrections:
        raise RuntimeError("dev audit requires corrections")

    rows = _jsonl(dev_path)
    by_uid = {str(row["uid"]): row for row in rows}
    reviewed_at = str(audit.get("reviewed_at") or _now())
    applied: list[dict[str, Any]] = []
    for correction in corrections:
        if not isinstance(correction, Mapping):
            raise TypeError("dev audit correction must be an object")
        uid = str(correction.get("uid") or "")
        row = by_uid.get(uid)
        if row is None:
            raise RuntimeError(f"dev audit UID is not in dev fixture: {uid}")
        if str(row.get("source_sha256") or "") != str(
            correction.get("source_sha256") or ""
        ):
            raise RuntimeError(f"dev audit source hash drifted: {uid}")
        original = str(correction.get("original_gold_primary_notation") or "")
        if str(row.get("gold_primary_notation") or "") != original:
            raise RuntimeError(f"dev audit original gold drifted: {uid}")
        primary = str(correction.get("gold_primary_notation") or "")
        allowed = [
            str(value)
            for value in correction.get("gold_allowed_primary_notations") or []
        ]
        candidate_notations = {
            str(candidate.get("notation") or "")
            for candidate in row.get("candidates") or []
            if isinstance(candidate, Mapping)
        }
        if (
            not primary
            or primary not in candidate_notations
            or not allowed
            or primary not in allowed
            or any(value not in candidate_notations for value in allowed)
        ):
            raise RuntimeError(f"dev audit notation is outside candidates: {uid}")
        rationale = str(correction.get("gold_rationale") or "").strip()
        reason = str(correction.get("reason") or "").strip()
        if not rationale or not reason:
            raise RuntimeError(f"dev audit rationale is missing: {uid}")
        row["gold_primary_notation"] = primary
        row["gold_allowed_primary_notations"] = allowed
        row["gold_rationale"] = rationale
        row["gold_review"] = {
            "audit_id": audit_id,
            "reviewed_at": reviewed_at,
            "original_gold_primary_notation": original,
            "reason": reason,
        }
        applied.append(
            {
                "uid": uid,
                "original": original,
                "primary": primary,
                "allowed": allowed,
            }
        )

    archive = root / "classification" / "fixtures" / "epochs" / f"{audit_id}-pre-audit"
    archive.mkdir(parents=True, exist_ok=True)
    shutil.copy2(dev_path, archive / dev_path.name)
    shutil.copy2(manifest_path, archive / manifest_path.name)
    before_sha256 = hashlib.sha256(dev_path.read_bytes()).hexdigest()
    _write_jsonl(dev_path, rows)
    after_sha256 = hashlib.sha256(dev_path.read_bytes()).hexdigest()
    audit_sha256 = hashlib.sha256(audit_path.read_bytes()).hexdigest()
    manifest["dev"] = {
        **manifest["dev"],
        "sha256": f"sha256:{after_sha256}",
        "audit_id": audit_id,
        "audit_sha256": f"sha256:{audit_sha256}",
        "reviewed_at": reviewed_at,
    }
    write_sealed_json(manifest_path, manifest, backup=True)
    receipt = {
        "schema": DEV_AUDIT_RECEIPT_SCHEMA,
        "status": "verified",
        "audit_id": audit_id,
        "audit_sha256": f"sha256:{audit_sha256}",
        "reviewed_at": reviewed_at,
        "holdout_opened": False,
        "correction_count": len(applied),
        "corrections": applied,
        "dev_before_sha256": f"sha256:{before_sha256}",
        "dev_after_sha256": f"sha256:{after_sha256}",
        "archive": str(archive),
    }
    write_sealed_json(
        root / "classification" / "dev-audit-receipt.json",
        receipt,
        backup=True,
    )
    return receipt


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


def calibration_input_fingerprint(
    root: Path,
    *,
    fixture_manifest_path: Path | None = None,
) -> dict[str, str]:
    """Return the immutable inputs that determine a calibration result."""

    if fixture_manifest_path is None:
        dev_path, _holdout_path, manifest_path = fixture_paths(root)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest_path = fixture_manifest_path
        manifest = load_fixture_set(manifest_path)
        dev_path = Path(str(manifest["dev"]["path"]))
    dev_sha256 = "sha256:" + hashlib.sha256(dev_path.read_bytes()).hexdigest()
    if str(manifest.get("dev", {}).get("sha256") or "") != dev_sha256:
        raise RuntimeError("dev fixture hash does not match its locked manifest")
    return {
        "dev_fixture_sha256": dev_sha256,
        "fixture_manifest_sha256": (
            "sha256:" + hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        ),
        "fixture_epoch": str(manifest.get("fixture_epoch") or "legacy"),
        "engine_version": str(manifest.get("engine_version") or ENGINE_VERSION),
        "package_checksum": load_udc_package(root).checksum,
        "config_digest": _config_digest(),
    }


def _verify_locked_holdout(manifest: Mapping[str, Any], holdout_path: Path) -> None:
    observed = "sha256:" + hashlib.sha256(holdout_path.read_bytes()).hexdigest()
    expected = str(manifest.get("holdout", {}).get("sha256") or "")
    if observed != expected:
        raise RuntimeError("locked Holdout hash does not match its manifest")


def _evaluate_locked_holdout(
    root: Path,
    holdout: list[dict[str, Any]],
    *,
    dev_metrics: Mapping[str, Any],
    thresholds: Mapping[str, Any],
    input_fingerprint: Mapping[str, str],
    manifest_path: Path | None = None,
    calibration_path: Path | None = None,
    holdout_results_path: Path | None = None,
    run_namespace: str = "calibration-holdout-epoch2-v2",
) -> dict[str, Any]:
    holdout_raw = run_consensus_batches(
        holdout,
        root=root,
        batch_size=20,
        purpose="explicit",
        timeout_seconds=1_800,
        run_namespace=run_namespace,
    )
    holdout_decisions = [
        {
            **row,
            "status": (
                "proposed"
                if int(row.get("quorum") or 0) >= 2
                and float(row.get("confidence") or 0.0)
                >= float(thresholds["minimum_confidence"])
                else "held"
            ),
        }
        for row in holdout_raw
    ]
    holdout_results_path = holdout_results_path or (
        root
        / "classification"
        / "fixtures"
        / "classification-holdout-epoch2-results.jsonl"
    )
    _write_jsonl(holdout_results_path, holdout_decisions)
    holdout_metrics = evaluate_predictions(holdout, holdout_decisions)
    return adopt_calibration(
        root,
        dev_metrics=dev_metrics,
        holdout_metrics=holdout_metrics,
        config_digest=str(input_fingerprint["config_digest"]),
        thresholds=thresholds,
        manifest_path=manifest_path,
        output_path=calibration_path,
        authority_epoch=(
            int(str(input_fingerprint.get("fixture_epoch") or "1").split("-")[1])
            if str(input_fingerprint.get("fixture_epoch") or "").startswith("epoch-")
            and str(input_fingerprint.get("fixture_epoch")).split("-")[1].isdigit()
            else 1
        ),
    )


def calibrate(
    root: Path,
    *,
    fixture_manifest_path: Path | None = None,
) -> dict[str, Any]:
    if fixture_manifest_path is None:
        dev_path, holdout_path, manifest_path = fixture_paths(root)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        calibration_path = root / "classification" / "calibration.json"
        preregistration_path = (
            root / "classification" / "calibration-preregistration.json"
        )
        holdout_results_path = (
            root
            / "classification"
            / "fixtures"
            / "classification-holdout-epoch2-results.jsonl"
        )
        namespace_suffix = "epoch2-v2"
    else:
        manifest_path = fixture_manifest_path
        manifest = load_fixture_set(manifest_path)
        dev_path = Path(str(manifest["dev"]["path"]))
        holdout_path = Path(str(manifest["holdout"]["path"]))
        calibration_path = manifest_path.parent / "calibration.json"
        preregistration_path = manifest_path.parent / "calibration-preregistration.json"
        holdout_results_path = manifest_path.parent / "holdout-results.jsonl"
        namespace_suffix = str(manifest["fixture_epoch"])
    _verify_locked_holdout(manifest, holdout_path)
    input_fingerprint = (
        calibration_input_fingerprint(root)
        if fixture_manifest_path is None
        else calibration_input_fingerprint(
            root, fixture_manifest_path=fixture_manifest_path
        )
    )
    holdout = _jsonl(holdout_path)
    if manifest["holdout"].get("opened_at"):
        if calibration_path.exists():
            existing = json.loads(calibration_path.read_text(encoding="utf-8"))
            if existing.get("status") == "adopted":
                return existing
        preregistration = read_sealed_json(preregistration_path)
        if (
            preregistration.get("schema") != PREREGISTRATION_SCHEMA
            or preregistration.get("input_fingerprint") != input_fingerprint
            or not isinstance(preregistration.get("thresholds"), Mapping)
            or not isinstance(preregistration.get("dev_metrics"), Mapping)
        ):
            raise RuntimeError("opened Holdout lacks a matching sealed preregistration")
        return _evaluate_locked_holdout(
            root,
            holdout,
            dev_metrics=preregistration["dev_metrics"],
            thresholds=preregistration["thresholds"],
            input_fingerprint=input_fingerprint,
            manifest_path=manifest_path,
            calibration_path=calibration_path,
            holdout_results_path=holdout_results_path,
            run_namespace=f"calibration-holdout-{namespace_suffix}",
        )
    dev = _jsonl(dev_path)
    dev_raw = run_consensus_batches(
        dev,
        root=root,
        batch_size=20,
        purpose="explicit",
        timeout_seconds=1_800,
        run_namespace=f"calibration-dev-{namespace_suffix}",
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
        and row["metrics"]["unexpected_hold_rate"] <= 0.08
        and row["metrics"]["primary_assignment_rate"] >= 0.98
        and row["metrics"]["exact_match_rate"] >= 0.90
        and row["metrics"]["hierarchy_within_one_rate"] >= 0.97
    ]
    if not eligible:
        rejection = {
            "schema": CALIBRATION_SCHEMA,
            "status": "rejected",
            "reason": "dev_gate_unmet_holdout_remains_sealed",
            "input_fingerprint": input_fingerprint,
            "package_checksum": input_fingerprint["package_checksum"],
            "fixture_locked": True,
            "config_digest": input_fingerprint["config_digest"],
            "dev_sweep": sweep,
            "forced_misclassification_rate": min(
                row["metrics"]["forced_misclassification_rate"] for row in sweep
            ),
        }
        write_sealed_json(
            calibration_path,
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
        "maximum_unexpected_hold_rate": 0.08,
        "maximum_forced_misclassification_rate": 0.01,
        "maximum_expected_hold_escape_rate": 0.0,
        "minimum_exact_match_rate": 0.90,
        "minimum_hierarchy_within_one_rate": 0.97,
    }
    preregistration = {
        "schema": PREREGISTRATION_SCHEMA,
        "registered_at": _now(),
        "authority_epoch": (
            int(str(manifest.get("fixture_epoch") or "1").split("-")[1])
            if str(manifest.get("fixture_epoch") or "").startswith("epoch-")
            and str(manifest.get("fixture_epoch")).split("-")[1].isdigit()
            else 1
        ),
        "fixture_manifest_sha256": (
            "sha256:" + hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        ),
        "input_fingerprint": input_fingerprint,
        "package_checksum": input_fingerprint["package_checksum"],
        "config_digest": input_fingerprint["config_digest"],
        "thresholds": thresholds,
        "dev_metrics": dev_metrics,
        "dev_sweep": sweep,
        "holdout_opened": False,
    }
    write_sealed_json(
        preregistration_path,
        preregistration,
        backup=True,
    )
    manifest["holdout"]["opened_at"] = _now()
    manifest["holdout"]["opening_reason"] = "single_locked_authority_evaluation"
    write_sealed_json(manifest_path, manifest, backup=True)
    return _evaluate_locked_holdout(
        root,
        dev_metrics=dev_metrics,
        thresholds=thresholds,
        holdout=holdout,
        input_fingerprint=input_fingerprint,
        manifest_path=manifest_path,
        calibration_path=calibration_path,
        holdout_results_path=holdout_results_path,
        run_namespace=f"calibration-holdout-{namespace_suffix}",
    )


def install_package(root: Path, source: Path) -> dict[str, Any]:
    package = load_udc_package(None)
    source_package = json.loads(source.read_text(encoding="utf-8"))
    if source_package.get("release") != package.release:
        from chronovisor.classification.classification import UDCPackage

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
            "audit-dev",
            "calibrate",
            "all",
        ),
    )
    parser.add_argument("--root", type=Path, default=CHRONOVISOR_ROOT)
    parser.add_argument("--package", type=Path)
    parser.add_argument("--audit", type=Path)
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
    elif args.command == "audit-dev":
        if args.audit is None:
            parser.error("--audit is required")
        result = apply_dev_audit(args.root, args.audit)
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
