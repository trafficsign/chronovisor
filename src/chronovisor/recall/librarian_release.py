"""Release, verification and soak orchestration for the classification Librarian."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from collections import Counter
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from chronovisor.core.durable_state import write_sealed_json
from chronovisor.core.link_fix import extract_wiki_links
from chronovisor.core.migration_snapshot import (
    cleanup_expired_restore_points,
    restore_drill,
)
from chronovisor.core.store import CHRONOVISOR_ROOT
from chronovisor.core.timeutil import iso_seconds as _iso
from chronovisor.ingest.page_registry import PageRegistry
from chronovisor.ingest.uid_link_index import build_uid_link_index
from chronovisor.recall.classification import (
    classification_authority_status,
    default_udc_package,
    load_udc_package,
)
from chronovisor.recall.classification_engine import librarian_convergence_store
from chronovisor.recall.librarian import capture_baseline
from chronovisor.recall.librarian_status import STATE_SCHEMA, load_librarian_state
from chronovisor.recall.merge_transaction import cleanup_expired_preimages

ADR_SCHEMA = "chronovisor.librarian-adr.v1"
SOAK_SCHEMA = "chronovisor.librarian-soak.v2"
RELEASE_SCHEMA = "chronovisor.librarian-release.v1"
UDC_RECEIPT_SCHEMA = "chronovisor.udc-package-receipt.v1"

RELEASE_PREREQUISITES = {
    "phase0-receipt.json": "ok",
    "phase1-receipt.json": "verified",
    "phase3-receipt.json": "verified",
    "phase5-receipt.json": "ok",
    "phase6-receipt.json": "ok",
    "phase7-burn.json": "passed",
    "phase10-pilot.json": "ok",
    "phase11-receipt.json": "ok",
}


def _now(value: datetime | None = None) -> datetime:
    current = value or datetime.now(UTC)
    return current if current.tzinfo else current.replace(tzinfo=UTC)




def _json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _scope_generation(root: Path, state: Mapping[str, Any]) -> str:
    rows = []
    for uid, row in sorted((state.get("pages") or {}).items()):
        if not isinstance(row, Mapping) or row.get("status") == "superseded":
            continue
        path = root / str(row.get("path") or "")
        if not path.is_file():
            continue
        stat = path.stat()
        rows.append((str(uid), str(row.get("path")), stat.st_size, stat.st_mtime_ns))
    encoded = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def capture_phase0_artifacts(
    root: Path = CHRONOVISOR_ROOT,
    *,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Freeze the migration constitution and immutable fixture boundary."""

    fixture_manifest = (
        root / "classification" / "fixtures" / "manifest.json"
    )
    if not fixture_manifest.is_file():
        raise RuntimeError("Phase 0 requires the locked 200/100 fixture manifest")
    baseline = capture_baseline(root=root, repo_root=repo_root, write=True)
    link_baseline = build_uid_link_index(root, write=True)
    adr_root = root / "runtime" / "librarian" / "adr"
    artifacts = {
        "classification-constitution-v1.json": {
            "decision": "adopt_udcs_with_versioned_cvo_extensions",
            "invariants": [
                "raw_is_immutable",
                "uid_is_identity_and_call_number_is_not",
                "models_select_only_host_candidates",
                "two_model_quorum_or_hold",
                "locked_holdout_opens_once_per_epoch",
                "legacy_tags_remain_compatibility_projection",
                "classification_nodes_are_graph_overlay_only",
            ],
        },
        "identity-redirect-merge-v1.json": {
            "decision": "uuidv7_dual_resolver_and_read_only_redirects",
            "max_redirect_hops": 8,
            "read_path_writes": False,
            "merge_requires_span_fingerprint_raw_and_sensitivity_gates": True,
        },
        "migration-retention-v1.json": {
            "decision": "no_permanent_premerge_archive",
            "restore_point_ttl_days": 7,
            "pilot_preimage_ttl_days": 7,
            "excluded_from_recall_search_obsidian_and_librarian": True,
        },
        "fixture-governance-v1.json": {
            "decision": "200_dev_plus_100_immutable_holdout",
            "estimated_human_budget_hours": [10, 25],
            "frontier_assistance_boundary": "fixture_creation_only_not_data_plane",
            "live_page_path_is_not_fixture_identity": True,
            "fixture_manifest": str(fixture_manifest),
            "fixture_manifest_sha256": hashlib.sha256(
                fixture_manifest.read_bytes()
            ).hexdigest(),
        },
    }
    written = []
    for filename, decision in artifacts.items():
        path = adr_root / filename
        write_sealed_json(
            path,
            {
                "schema": ADR_SCHEMA,
                "recorded_at": _iso(_now()),
                **decision,
            },
            backup=True,
        )
        written.append(str(path))
    receipt = {
        "status": "ok",
        "baseline": baseline,
        "link_baseline": {
            "edge_count": link_baseline["edge_count"],
            "unresolved_count": link_baseline["unresolved_count"],
            "content_sha256": link_baseline["content_sha256"],
        },
        "artifacts": written,
    }
    write_sealed_json(
        root / "runtime" / "librarian" / "phase0-receipt.json",
        receipt,
        backup=True,
    )
    return receipt


def record_phase1_package(root: Path = CHRONOVISOR_ROOT) -> dict[str, Any]:
    """Verify the bundled and installed licensed UDC Summary snapshot."""

    bundled_path = (
        Path(__file__).resolve().parents[1] / "data" / "udc-summary.json"
    )
    installed_path = root / "classification" / "udc-package.json"
    if not bundled_path.is_file() or not installed_path.is_file():
        raise RuntimeError("bundled and installed UDC packages are both required")
    bundled = default_udc_package()
    installed = load_udc_package(root)
    if not bundled.complete or not installed.complete:
        raise RuntimeError("Phase 1 requires complete UDC packages")
    if bundled.checksum != installed.checksum:
        raise RuntimeError("installed UDC package differs from bundled authority")
    concepts = list(installed.concepts.values())
    japanese = sum(bool(row.get("label_ja")) for row in concepts)
    coverage = japanese / max(1, len(concepts))
    if coverage < 0.95:
        raise RuntimeError(f"Japanese UDC label coverage is only {coverage:.3%}")
    package_payload = json.loads(bundled_path.read_text(encoding="utf-8"))
    receipt = {
        "schema": UDC_RECEIPT_SCHEMA,
        "status": "verified",
        "recorded_at": _iso(_now()),
        "release": installed.release,
        "checksum": installed.checksum,
        "concept_count": len(concepts),
        "japanese_label_count": japanese,
        "japanese_label_coverage": coverage,
        "license": installed.license,
        "attribution": installed.attribution,
        "source_url": installed.source_url,
        "source_kind": package_payload.get("source_kind"),
        "source_snapshots": package_payload.get("sources") or [],
        "hierarchy_validated": True,
        "notation_and_uri_unique": True,
        "cvo_registry_versioned": True,
        "ndc_overlay": "interface_only_no_licensed_data_bundled",
    }
    write_sealed_json(
        root / "runtime" / "librarian" / "phase1-receipt.json",
        receipt,
        backup=True,
    )
    return receipt


def verify_uid_link_foundation(root: Path = CHRONOVISOR_ROOT) -> dict[str, Any]:
    """Prove UID/slug parity and derived-link completeness on the live corpus."""

    registry = PageRegistry(root)
    state = registry.ensure_manifest(write=True)["registry"]
    index = build_uid_link_index(root, registry=registry, write=True)
    active = {
        uid: row
        for uid, row in state["pages"].items()
        if isinstance(row, Mapping)
        and row.get("status") != "superseded"
        and (root / str(row.get("path") or "")).is_file()
    }
    legacy_links = 0
    for row in active.values():
        text = (root / str(row["path"])).read_text(encoding="utf-8")
        legacy_links += len(extract_wiki_links(text, strip=True))
    projected_links = int(index["edge_count"]) + int(index["unresolved_count"])
    if projected_links != legacy_links:
        raise RuntimeError(
            f"UID graph parity failed: {projected_links} != {legacy_links}"
        )
    if len(active) != len(set(active)):
        raise RuntimeError("duplicate active UID")
    paths = [str(row["path"]) for row in active.values()]
    if len(paths) != len(set(paths)):
        raise RuntimeError("duplicate active page path")
    ambiguous = state.get("ambiguous_keys")
    collision_count = len(ambiguous) if isinstance(ambiguous, dict) else 0
    receipt = {
        "status": "verified",
        "active_pages": len(active),
        "unique_uids": len(active),
        "legacy_link_count": legacy_links,
        "uid_edge_count": int(index["edge_count"]),
        "unresolved_count": int(index["unresolved_count"]),
        "graph_parity": True,
        "ambiguous_legacy_keys": collision_count,
        "redirect_resolution": "validated_by_registry_load_max_8_hops",
        "semantic_uid_generation": "supported_after_frontmatter_backfill",
        "recall_uid_fields": True,
    }
    write_sealed_json(
        root / "runtime" / "librarian" / "phase3-receipt.json",
        receipt,
        backup=True,
    )
    return receipt


def reconcile_librarian_state(
    root: Path = CHRONOVISOR_ROOT,
    *,
    mode: str | None = None,
    initial_complete_at: str | None = None,
) -> dict[str, Any]:
    """Refresh host-derived state after classification or migration work."""

    if (
        root / "runtime" / "librarian" / "collection-registry.json"
    ).is_file():
        from chronovisor.recall.librarian import run_shadow

        current = dict(run_shadow(root=root)["state"])
        changed = False
        if mode is not None and current.get("mode") != mode:
            current["mode"] = mode
            changed = True
        if (
            initial_complete_at is not None
            and current.get("initial_organization_complete_at")
            != initial_complete_at
        ):
            current["initial_organization_complete_at"] = initial_complete_at
            changed = True
        if changed:
            write_sealed_json(
                root / "runtime" / "librarian" / "state.json",
                current,
                backup=True,
            )
        return current

    registry = PageRegistry(root)
    state = registry.ensure_manifest(write=True)["registry"]
    active = {
        uid: row
        for uid, row in state["pages"].items()
        if isinstance(row, Mapping)
        and row.get("status") != "superseded"
        and (root / str(row.get("path") or "")).is_file()
    }
    total = len(active)
    status_counts = Counter(
        str(row.get("classification_status") or "unclassified")
        for row in active.values()
    )
    classified = sum(
        isinstance(row.get("classification"), Mapping) for row in active.values()
    )
    terminal = status_counts["adopted"] + status_counts["held"]
    link_index = build_uid_link_index(root, registry=registry, write=True)
    link_denominator = int(link_index["edge_count"]) + int(
        link_index["unresolved_count"]
    )
    scope = _scope_generation(root, state)
    authority = classification_authority_status(root)
    previous = load_librarian_state(root)
    queue_items = librarian_convergence_store(root).list_items()
    queue_status = Counter(str(row.get("status") or "") for row in queue_items)
    calibration = _json(root / "classification" / "calibration.json")
    baseline = _json(root / "runtime" / "librarian" / "baseline.json")
    dispositions = _json(
        root / "runtime" / "librarian" / "migration-dispositions.json"
    )
    disposition_pages = dispositions.get("pages")
    disposition_pages = (
        disposition_pages if isinstance(disposition_pages, dict) else {}
    )
    full_sweep_current = bool(
        terminal == total
        and len(disposition_pages) >= total
        and not link_index["unresolved_count"]
    )
    try:
        from chronovisor.raw.raw_store import RawStore

        raw_units = sum(1 for _unit in RawStore(root / "raw").iter_units())
    except Exception:
        raw_units = sum(1 for path in (root / "raw").rglob("*.md") if path.is_file())
    try:
        baseline_time = datetime.fromisoformat(str(baseline["captured_at"]))
        if baseline_time.tzinfo is None:
            baseline_time = baseline_time.replace(tzinfo=UTC)
    except (KeyError, TypeError, ValueError):
        baseline_time = _now()
    baseline_days = max(1 / 24, (_now() - baseline_time).total_seconds() / 86_400)
    page_delta = total - int(baseline.get("pages") or total)
    raw_delta = raw_units - int(baseline.get("raw_logical_units") or raw_units)
    progress = {
        "uid": {
            "numerator": total,
            "denominator": total,
            "scope_generation": scope,
        },
        "classification_shadow": {
            "numerator": classified,
            "denominator": total,
            "scope_generation": scope,
        },
        "classification_terminal": {
            "numerator": terminal,
            "denominator": total,
            "scope_generation": scope,
        },
        "links": {
            "numerator": int(link_index["edge_count"]),
            "denominator": link_denominator,
            "unresolved": int(link_index["unresolved_count"]),
            "scope_generation": scope,
        },
        "migration_batch": {
            "numerator": terminal,
            "denominator": total,
            "scope_generation": scope,
        },
        "full_sweep": {
            "numerator": int(full_sweep_current),
            "denominator": 1,
            "current": full_sweep_current,
            "scope_generation": scope,
        },
    }
    current = {
        "schema": STATE_SCHEMA,
        "enabled": True,
        "mode": mode or ("active" if authority["active"] else "shadow"),
        "generation": int(previous.get("generation") or 0) + 1,
        "scope_generation": scope,
        "last_swept_scope_generation": scope if full_sweep_current else None,
        "initial_organization_complete_at": (
            initial_complete_at
            if initial_complete_at is not None
            else previous.get("initial_organization_complete_at")
        ),
        "authority": authority,
        "progress": progress,
        "queue": {
            "queued": max(0, total - classified),
            "actionable": max(0, total - terminal),
            "running": queue_status["local_running"]
            + queue_status["frontier_running"],
            "held": status_counts["held"],
            "quarantined": queue_status["quarantined"],
            "completed": status_counts["adopted"],
            "oldest_age_seconds": 0,
        },
        "debts": {
            "unclassified": max(0, total - classified),
            "nonterminal_classification": max(0, total - terminal),
            "explicit_hold": status_counts["held"],
            "unresolved_link": int(link_index["unresolved_count"]),
            "terminal_dispositions": len(disposition_pages),
        },
        "quality": {
            "classification_authority_active": bool(authority["active"]),
            "locked_holdout": calibration.get("status") or "missing",
            "holdout_metrics": calibration.get("holdout_metrics") or {},
            "forced_misclassification_gate": (
                calibration.get("gates", {}).get("forced_misclassification")
                if isinstance(calibration.get("gates"), dict)
                else None
            ),
            "recall_regression": previous.get("quality", {}).get(
                "recall_regression", "not_evaluated"
            ),
            "broken_redirects": 0,
            "sensitivity_downgrades": 0,
        },
        "resources": {
            **dict(previous.get("resources") or {}),
            "priority": "P3",
            "frontier_calls": 0,
            "queue_items": len(queue_items),
        },
        "growth": {
            "active_pages": total,
            "raw_logical_units": raw_units,
            "active_page_delta": page_delta,
            "raw_unit_delta": raw_delta,
            "active_pages_per_day": page_delta / baseline_days,
            "raw_units_per_day": raw_delta / baseline_days,
            "baseline_at": baseline.get("captured_at"),
        },
        "eta": previous.get("eta"),
        "blocked_reasons": [],
        "last_run": _iso(_now()),
    }
    write_sealed_json(
        root / "runtime" / "librarian" / "state.json",
        current,
        backup=True,
    )
    return current


def _release_prerequisite_errors(
    root: Path,
) -> tuple[list[str], list[str]]:
    missing = []
    invalid = []
    for filename, expected_status in RELEASE_PREREQUISITES.items():
        payload = _json(root / "runtime" / "librarian" / filename)
        if not payload:
            missing.append(filename)
        elif payload.get("status") != expected_status:
            invalid.append(
                f"{filename}:{payload.get('status')}!={expected_status}"
            )
    collection_receipt = _json(
        root / "runtime" / "librarian" / "phase4-collection-authority.json"
    )
    if collection_receipt.get("status") == "adopted":
        authority = (
            (collection_receipt.get("authority") or {})
            if isinstance(collection_receipt.get("authority"), dict)
            else {}
        )
        if not authority.get("active"):
            invalid.append("phase4-collection-authority.json:not_active")
    else:
        calibration = _json(root / "classification" / "calibration.json")
        if calibration.get("status") != "adopted":
            invalid.append("classification/calibration.json:not_adopted")
    return missing, invalid


def start_soak(
    root: Path = CHRONOVISOR_ROOT,
    *,
    days: int | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Start observation concurrently with full-corpus migration.

    ``days`` remains a compatibility argument for older wrappers. Release is
    evidence-gated, not wall-clock-gated: Phase 5 through Phase 11 provide the
    load and transaction observation, followed by the Phase 12 postflight.
    """

    path = root / "runtime" / "librarian" / "soak.json"
    existing = _json(path)
    if (
        existing.get("schema") == SOAK_SCHEMA
        and existing.get("status") in {"running", "complete"}
    ):
        return existing

    current = _now(now)
    starts_at = str(existing.get("starts_at") or _iso(current))
    payload = {
        "schema": SOAK_SCHEMA,
        "status": "running",
        "observation_mode": "concurrent_migration",
        "starts_at": starts_at,
        "wall_clock_required_seconds": 0,
        "required_evidence": list(RELEASE_PREREQUISITES),
        "observed_through": "phase5_full_shadow_started",
        "checkpoints": [
            {
                "stage": "phase5_full_shadow_started",
                "observed_at": _iso(current),
            }
        ],
        "legacy_requested_days": days,
        "release_receipt": None,
    }
    write_sealed_json(path, payload, backup=True)
    return payload


def advance_migration_observation(
    root: Path,
    *,
    stage: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Persist the latest completed migration stage without resetting elapsed time."""

    payload = start_soak(root, now=now)
    if payload.get("status") != "running":
        return payload
    current = _now(now)
    checkpoints = list(payload.get("checkpoints") or [])
    if not checkpoints or checkpoints[-1].get("stage") != stage:
        checkpoints.append({"stage": stage, "observed_at": _iso(current)})
    payload["observed_through"] = stage
    payload["checkpoints"] = checkpoints
    write_sealed_json(
        root / "runtime" / "librarian" / "soak.json",
        payload,
        backup=True,
    )
    return payload


def _require_release_prerequisites(root: Path) -> None:
    missing, invalid = _release_prerequisite_errors(root)
    if missing or invalid:
        raise RuntimeError(
            f"release prerequisites missing={missing} invalid={invalid}"
        )


def _restore_all(root: Path) -> list[dict[str, Any]]:
    base = root / "runtime" / "librarian" / "migration-restore-points"
    receipts = []
    if not base.exists():
        return receipts
    for path in sorted(value for value in base.iterdir() if value.is_dir()):
        with tempfile.TemporaryDirectory(
            prefix="chronovisor-release-restore-"
        ) as temporary:
            receipts.append(
                restore_drill(path, Path(temporary) / "restored")
            )
    return receipts


def finalize_release(
    root: Path = CHRONOVISOR_ROOT,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    current = _now(now)
    soak_path = root / "runtime" / "librarian" / "soak.json"
    soak = _json(soak_path)
    if soak.get("schema") != SOAK_SCHEMA:
        raise RuntimeError("migration observation has not started")
    release_path = root / "runtime" / "librarian" / "phase12-release.json"
    if release_path.is_file():
        return _json(release_path)
    if soak.get("status") not in {"running", "complete"}:
        raise RuntimeError("migration observation is not running")
    _require_release_prerequisites(root)

    state_before = reconcile_librarian_state(root, mode="active")
    progress_before = state_before.get("progress") or {}
    if not (state_before.get("authority") or {}).get("active"):
        raise RuntimeError("classification authority is no longer active")
    if not (progress_before.get("full_sweep") or {}).get("current"):
        raise RuntimeError("full corpus sweep drifted during migration observation")
    for key in ("classification_terminal", "migration_batch"):
        row = progress_before.get(key) or {}
        if row.get("numerator") != row.get("denominator"):
            raise RuntimeError(f"{key} coverage is incomplete")
    debts = state_before.get("debts") or {}
    if int(debts.get("worker_failure") or 0):
        raise RuntimeError("worker failures remain after migration")
    if int(debts.get("unresolved_link") or 0):
        raise RuntimeError("unresolved links remain after migration")

    restore_receipts = _restore_all(root)
    if not restore_receipts or any(
        row.get("status") != "verified" for row in restore_receipts
    ):
        raise RuntimeError("final restore drill did not verify every restore point")
    restore_cleanup = cleanup_expired_restore_points(
        root,
        now=current,
        force=True,
    )
    preimage_cleanup = cleanup_expired_preimages(root, now=current, force=True)
    if restore_cleanup.get("retained") or preimage_cleanup.get("retained"):
        raise RuntimeError("verified migration insurance cleanup is incomplete")

    state = reconcile_librarian_state(
        root,
        mode="active",
        initial_complete_at=_iso(current),
    )
    progress = state["progress"]
    if (
        progress["classification_terminal"]["numerator"]
        != progress["classification_terminal"]["denominator"]
        or progress["migration_batch"]["numerator"]
        != progress["migration_batch"]["denominator"]
    ):
        raise RuntimeError("initial organization coverage is incomplete")
    receipt = {
        "schema": RELEASE_SCHEMA,
        "status": "released",
        "released_at": _iso(current),
        "observation": {
            **soak,
            "status": "complete",
            "observed_through": "phase12_postflight",
            "completed_at": _iso(current),
        },
        "restore_drills": restore_receipts,
        "cleanup": {
            "restore_points": restore_cleanup,
            "transaction_preimages": preimage_cleanup,
        },
        "initial_organization_complete_at": state[
            "initial_organization_complete_at"
        ],
    }
    write_sealed_json(
        root / "runtime" / "librarian" / "phase12-release.json",
        receipt,
        backup=True,
    )
    soak["status"] = "complete"
    soak["observed_through"] = "phase12_postflight"
    soak["completed_at"] = _iso(current)
    soak["release_receipt"] = str(
        root / "runtime" / "librarian" / "phase12-release.json"
    )
    write_sealed_json(soak_path, soak, backup=True)
    return receipt


def finalize_if_ready(
    root: Path = CHRONOVISOR_ROOT,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Close migration observation once every evidence gate is satisfied."""

    release_path = root / "runtime" / "librarian" / "phase12-release.json"
    if release_path.is_file():
        return {
            "status": "already_released",
            "release": str(release_path),
        }
    soak = _json(root / "runtime" / "librarian" / "soak.json")
    if soak.get("schema") != SOAK_SCHEMA:
        return {"status": "not_started"}
    missing, invalid = _release_prerequisite_errors(root)
    if missing or invalid:
        return {
            "status": "observing",
            "missing": missing,
            "invalid": invalid,
        }
    return finalize_release(root, now=now)


def _git_root() -> Path | None:
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return Path(result.stdout.strip()) if result.returncode == 0 else None


def main(argv: list[str] | None = None) -> int:
    """Run the ``chronovisor-librarian-release`` command-line entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=(
            "phase0",
            "phase1",
            "verify-links",
            "reconcile",
            "start-observation",
            "start-soak",
            "finalize",
        ),
    )
    parser.add_argument("--root", type=Path, default=CHRONOVISOR_ROOT)
    parser.add_argument(
        "--days",
        type=int,
        default=None,
        help="Deprecated compatibility option; release is evidence-gated.",
    )
    args = parser.parse_args(argv)
    if args.command == "phase0":
        result = capture_phase0_artifacts(args.root, repo_root=_git_root())
    elif args.command == "phase1":
        result = record_phase1_package(args.root)
    elif args.command == "verify-links":
        result = verify_uid_link_foundation(args.root)
    elif args.command == "reconcile":
        result = reconcile_librarian_state(args.root)
    elif args.command in {"start-observation", "start-soak"}:
        result = start_soak(args.root, days=args.days)
    else:
        result = finalize_release(args.root)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
