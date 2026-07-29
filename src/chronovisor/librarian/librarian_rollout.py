"""Crash-resumable production rollout for the classification Librarian."""

from __future__ import annotations

from chronovisor.core.timeutil import utc_iso_milliseconds as _now

import argparse
import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from chronovisor.lab.classification_calibration import (
    adjudicate,
    adjudication_path,
    calibrate,
    calibration_input_fingerprint,
    distribution,
    lock,
)
from chronovisor.classification.classification_engine import fixture_paths
from chronovisor.lab.classification_migration import (
    migrate_active_metadata,
    run_full_model_shadow,
)
from chronovisor.core.durable_state import file_lock, write_sealed_json
from chronovisor.lab.librarian_burn import run_burn
from chronovisor.librarian.librarian_merge import run_merge_migration
from chronovisor.librarian.librarian_release import (
    advance_migration_observation,
    capture_phase0_artifacts,
    finalize_if_ready,
    reconcile_librarian_state,
    start_soak,
)
from chronovisor.core.store import CHRONOVISOR_ROOT

ROLLOUT_SCHEMA = "chronovisor.librarian-rollout.v1"


class RolloutBlocked(RuntimeError):
    """A hard quality gate correctly stopped the rollout."""




def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _state_path(root: Path) -> Path:
    return root / "runtime" / "librarian" / "rollout.json"


def _write_state(
    root: Path,
    *,
    status: str,
    stage: str,
    detail: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "schema": ROLLOUT_SCHEMA,
        "status": status,
        "stage": stage,
        "updated_at": _now(),
        "detail": detail or {},
    }
    write_sealed_json(_state_path(root), payload, backup=True)
    return payload


def _receipt_ok(root: Path, filename: str, status: str) -> bool:
    payload = _read_json(root / "runtime" / "librarian" / filename)
    return payload.get("status") == status


def _run_stage(
    root: Path,
    stage: str,
    operation: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    _write_state(root, status="running", stage=stage)
    result = operation()
    _write_state(root, status="running", stage=stage, detail=result)
    return result


def _run_collection_rollout(root: Path) -> dict[str, Any]:
    """Resume ADR-0001 through the shared burn, merge and release gates."""

    from chronovisor.librarian.collection_authority import (
        collection_authority_status,
        run_collection_librarian,
    )

    result = _run_stage(
        root,
        "phase5_collection_sync",
        lambda: run_collection_librarian(root, evaluate_unseen=True),
    )
    authority = collection_authority_status(root)
    if not authority.get("active"):
        raise RolloutBlocked(
            "collection authority rejected by preregistered quality gates"
        )
    quality = result["quality"]
    phase5 = {
        "status": "ok",
        "method": "collection-first-authority",
        "assignment_count": result["sync"]["assignment_count"],
        "collection_count": result["sync"]["collection_count"],
        "quality_status": quality["status"],
        "review_queue_open": result["queue"]["open"],
        "page_mutations": 0,
        "model_calls": 0,
    }
    write_sealed_json(
        root / "runtime" / "librarian" / "phase5-receipt.json",
        phase5,
        backup=True,
    )
    write_sealed_json(
        root / "runtime" / "librarian" / "phase6-receipt.json",
        {
            "status": "ok",
            "method": "legacy-page-udc-metadata-superseded",
            "collection_registry_mirror_generation": result["sync"][
                "page_registry_mirror"
            ]["generation"],
            "page_mutations": 0,
        },
        backup=True,
    )
    review_queue = result["queue"]
    queue_open = int(review_queue.get("open", 0))
    if queue_open:
        next_stage = "collection_review_queue"
        next_gate = "review_queue_budget_and_split_proposal_audit"
        return _write_state(
            root,
            status="observing",
            stage=next_stage,
            detail={
                "authority": authority,
                "quality": {
                    "status": quality["status"],
                    "warnings": quality["warnings"],
                    "hard_failures": quality["hard_failures"],
                },
                "review_queue": review_queue,
                "next_gate": next_gate,
            },
        )

    _run_stage(
        root,
        "phase5_migration_observation",
        lambda: start_soak(root),
    )
    advance_migration_observation(
        root,
        stage="phase5_collection_shadow_complete",
    )
    advance_migration_observation(
        root,
        stage="phase6_collection_authority_complete",
    )
    if not _receipt_ok(root, "phase7-burn.json", "passed"):
        burn = _run_stage(
            root,
            "phase7_preemption_burn",
            lambda: run_burn(root, foreground_admissions=200),
        )
        if burn.get("status") != "passed":
            raise RolloutBlocked("P0 preemption burn gate failed")
    advance_migration_observation(
        root,
        stage="phase7_preemption_burn_complete",
    )
    if not _receipt_ok(root, "phase10-pilot.json", "ok"):
        _run_stage(
            root,
            "phase10_pilot",
            lambda: run_merge_migration(root, pilot_limit=3),
        )
    advance_migration_observation(
        root,
        stage="phase10_pilot_complete",
    )
    if not _receipt_ok(root, "phase11-receipt.json", "ok"):
        _run_stage(
            root,
            "phase11_migration",
            lambda: run_merge_migration(root, pilot_limit=None),
        )
    advance_migration_observation(
        root,
        stage="phase11_full_migration_complete",
    )

    postflight = _run_stage(
        root,
        "phase12_collection_postflight",
        lambda: run_collection_librarian(root, evaluate_unseen=True),
    )
    postflight_authority = collection_authority_status(root)
    if not postflight_authority.get("active"):
        raise RolloutBlocked("collection authority drifted during postflight")
    if int(postflight["queue"].get("open", 0)):
        return _write_state(
            root,
            status="observing",
            stage="collection_review_queue",
            detail={
                "authority": postflight_authority,
                "quality": postflight["quality"],
                "review_queue": postflight["queue"],
                "next_gate": "review_queue_budget_and_split_proposal_audit",
            },
        )
    _run_stage(
        root,
        "phase12_reconcile",
        lambda: reconcile_librarian_state(root, mode="active"),
    )
    release = _run_stage(
        root,
        "phase12_release",
        lambda: finalize_if_ready(root),
    )
    if release.get("status") == "released":
        return _write_state(
            root,
            status="released",
            stage="complete",
            detail=release,
        )
    return _write_state(
        root,
        status="observing",
        stage="phase12_release_gates",
        detail=release,
    )


def run_rollout(
    root: Path = CHRONOVISOR_ROOT,
    *,
    batch_size: int = 20,
    pilot_limit: int = 3,
    foreground_admissions: int = 200,
) -> dict[str, Any]:
    """Resume every rollout phase through evidence-gated release."""

    lock_path = root / "runtime" / "librarian" / "rollout.lock"
    with file_lock(lock_path):
        release = _read_json(
            root / "runtime" / "librarian" / "phase12-release.json"
        )
        if release.get("status") == "released":
            return _write_state(
                root,
                status="released",
                stage="complete",
                detail=release,
            )
        collection_receipt = _read_json(
            root
            / "runtime"
            / "librarian"
            / "phase4-collection-authority.json"
        )
        if collection_receipt.get("status") == "adopted":
            try:
                return _run_collection_rollout(root)
            except RolloutBlocked as exc:
                return _write_state(
                    root,
                    status="blocked",
                    stage="quality_gate",
                    detail={"error": str(exc)},
                )
        try:
            if not adjudication_path(root).is_file():
                _run_stage(
                    root,
                    "phase0_fixture_adjudication",
                    lambda: adjudicate(root, batch_size=max(1, batch_size)),
                )
            manifest_path = fixture_paths(root)[2]
            if not manifest_path.is_file():
                locked = _run_stage(
                    root,
                    "phase0_fixture_lock",
                    lambda: lock(root),
                )
                if locked.get("status") != "locked":
                    raise RolloutBlocked("classification fixture lock was rejected")
            if not _receipt_ok(root, "phase0-receipt.json", "ok"):
                _run_stage(
                    root,
                    "phase0_baseline",
                    lambda: capture_phase0_artifacts(root),
                )
            distribution_path = root / "classification" / "distribution-analysis.json"
            if not distribution_path.is_file():
                _run_stage(root, "phase4_distribution", lambda: distribution(root))
            calibration_path = root / "classification" / "calibration.json"
            calibration = _read_json(calibration_path)
            manifest = _read_json(manifest_path)
            holdout_opened = bool(manifest.get("holdout", {}).get("opened_at"))
            rejected_inputs_changed = (
                calibration.get("status") == "rejected"
                and not holdout_opened
                and calibration.get("input_fingerprint")
                != calibration_input_fingerprint(root)
            )
            opened_holdout_incomplete = (
                holdout_opened and calibration.get("status") != "adopted"
            )
            if (
                not calibration
                or rejected_inputs_changed
                or opened_holdout_incomplete
            ):
                calibration = _run_stage(
                    root,
                    "phase4_calibration",
                    lambda: calibrate(root),
                )
            if calibration.get("status") != "adopted":
                raise RolloutBlocked(
                    "classification authority rejected by locked quality gates"
                )
            _run_stage(
                root,
                "phase5_migration_observation",
                lambda: start_soak(root),
            )
            while not _receipt_ok(root, "phase5-receipt.json", "ok"):
                shadow = _run_stage(
                    root,
                    "phase5_shadow",
                    lambda: run_full_model_shadow(
                        root,
                        limit=-1,
                        batch_size=max(1, batch_size),
                    ),
                )
                if shadow.get("status") == "ok":
                    break
            advance_migration_observation(
                root,
                stage="phase5_full_shadow_complete",
            )
            if not _receipt_ok(root, "phase6-receipt.json", "ok"):
                _run_stage(
                    root,
                    "phase6_active_metadata",
                    lambda: migrate_active_metadata(root, batch_size=100),
                )
            advance_migration_observation(
                root,
                stage="phase6_active_metadata_complete",
            )
            if not _receipt_ok(root, "phase7-burn.json", "passed"):
                burn = _run_stage(
                    root,
                    "phase7_preemption_burn",
                    lambda: run_burn(
                        root,
                        foreground_admissions=max(3, foreground_admissions),
                    ),
                )
                if burn.get("status") != "passed":
                    raise RolloutBlocked("P0 preemption burn gate failed")
            advance_migration_observation(
                root,
                stage="phase7_preemption_burn_complete",
            )
            if not _receipt_ok(root, "phase10-pilot.json", "ok"):
                _run_stage(
                    root,
                    "phase10_pilot",
                    lambda: run_merge_migration(
                        root,
                        pilot_limit=max(0, pilot_limit),
                    ),
                )
            advance_migration_observation(
                root,
                stage="phase10_pilot_complete",
            )
            if not _receipt_ok(root, "phase11-receipt.json", "ok"):
                _run_stage(
                    root,
                    "phase11_migration",
                    lambda: run_merge_migration(root, pilot_limit=None),
                )
            advance_migration_observation(
                root,
                stage="phase11_full_migration_complete",
            )
            _run_stage(
                root,
                "phase12_reconcile",
                lambda: reconcile_librarian_state(root, mode="active"),
            )
            release = _run_stage(
                root,
                "phase12_release",
                lambda: finalize_if_ready(root),
            )
            if release.get("status") == "released":
                return _write_state(
                    root,
                    status="released",
                    stage="complete",
                    detail=release,
                )
            return _write_state(
                root,
                status="observing",
                stage="phase12_release_gates",
                detail=release,
            )
        except RolloutBlocked as exc:
            return _write_state(
                root,
                status="blocked",
                stage="quality_gate",
                detail={"reason": str(exc)},
            )
        except Exception as exc:
            _write_state(
                root,
                status="failed",
                stage="retryable_failure",
                detail={"error": f"{type(exc).__name__}: {exc}"},
            )
            raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=CHRONOVISOR_ROOT)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--pilot-limit", type=int, default=3)
    parser.add_argument("--foreground-admissions", type=int, default=200)
    args = parser.parse_args(argv)
    result = run_rollout(
        args.root,
        batch_size=max(1, args.batch_size),
        pilot_limit=max(0, args.pilot_limit),
        foreground_admissions=max(3, args.foreground_admissions),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["status"] in {"released", "observing", "blocked"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
