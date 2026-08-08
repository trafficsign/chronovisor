"""Artifact-only full-corpus classification sweep.

This runner is intentionally unable to update Pages, Page Registry, Recall, or
Graph artifacts.  It writes only into its caller-provided overlay directory and
proves the protected paths remained byte-identical.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from chronovisor.classification.classification import (
    ClassificationError,
    load_udc_package,
)
from chronovisor.classification.classification_bundle import (
    ADOPTED_MANIFEST_SCHEMA,
    activate_decision_only,
    pointer_paths,
    resolve_authority,
    rollback_authority,
)
from chronovisor.classification.classification_engine import page_payload
from chronovisor.core.durable_state import read_sealed_json, write_sealed_json
from chronovisor.core.timeutil import utc_iso_milliseconds as _now
from chronovisor.ingest.page_registry import PageRegistry
from chronovisor.lab.classification_fixture_set import (
    _write_jsonl,
    inference_dto,
    sha256_bytes,
    sha256_file,
)
from chronovisor.lab.classification_library_evidence import (
    LibraryEvidenceIndex,
    LibraryEvidenceProvider,
)

ARTIFACT_SWEEP_SCHEMA = "chronovisor.classification-artifact-sweep.v1"




def _tree_manifest(paths: Sequence[Path]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for base in paths:
        if not base.exists():
            output[str(base)] = {"kind": "missing"}
            continue
        if base.is_file():
            output[str(base)] = {
                "kind": "file",
                "size": base.stat().st_size,
                "sha256": sha256_file(base),
            }
            continue
        for path in sorted(value for value in base.rglob("*") if value.is_file()):
            output[str(path)] = {
                "kind": "file",
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    return output


def protected_paths(root: Path) -> list[Path]:
    return [
        root / "pages",
        root / "system",
        root / "runtime" / "librarian" / "page-registry.json",
        root / "recall",
        root / "graph",
    ]


def storage_manifest(
    *,
    working_paths: Sequence[Path],
    audit_paths: Sequence[Path],
) -> dict[str, Any]:
    def size(paths: Sequence[Path]) -> int:
        files = {
            path.resolve()
            for base in paths
            if base.exists()
            for path in (
                [base]
                if base.is_file()
                else [value for value in base.rglob("*") if value.is_file()]
            )
        }
        return sum(path.stat().st_size for path in files)

    working = size(working_paths)
    audit = size(audit_paths)
    return {
        "working_set_bytes": working,
        "working_set_limit_bytes": 3 * 1024**3,
        "working_set_passed": working <= 3 * 1024**3,
        "audit_store_bytes": audit,
        "audit_annual_budget_bytes": 1024**3,
        "audit_warning": audit >= int(0.8 * 1024**3),
        "audit_adoption_blocked": audit > 1024**3,
    }


def _resolver_drill(disabled_manifest: Path) -> dict[str, Any]:
    if not disabled_manifest.is_file():
        raise ClassificationError("disabled baseline manifest is missing")
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="chronovisor-resolver-drill-") as raw:
        drill_root = Path(raw)
        bundle_dir = drill_root / "classification" / "library-evidence" / "bundles"
        bundle_dir.mkdir(parents=True)
        disabled_copy = bundle_dir / "disabled.json"
        shutil.copy2(disabled_manifest, disabled_copy)
        disabled = activate_decision_only(drill_root, target_path=disabled_copy)
        adopted_path = bundle_dir / "synthetic-adopted.json"
        write_sealed_json(
            adopted_path,
            {
                "schema": ADOPTED_MANIFEST_SCHEMA,
                "candidate_bundle_path": "synthetic",
                "adoption_payload": {"adoption_policy": {"authority_epoch": 0}},
                "authority": {"authority_digest": "sha256:synthetic"},
                "mutation_capability": False,
                "mode": "decision-only/canary",
            },
            backup=True,
        )
        active = activate_decision_only(drill_root, target_path=adopted_path)
        rollback_started = time.monotonic()
        rolled_back = rollback_authority(drill_root)
        rollback_seconds = time.monotonic() - rollback_started

        missing_root = drill_root / "missing"
        missing = resolve_authority(missing_root)
        corrupt_root = drill_root / "corrupt"
        corrupt_active, _previous, _mutation = pointer_paths(corrupt_root)
        corrupt_active.parent.mkdir(parents=True)
        corrupt_active.write_bytes(b"{not-json")
        corrupt = resolve_authority(corrupt_root)
    gates = {
        "disabled_distinct_from_corrupt": (
            disabled.get("status") == "disabled" and corrupt.get("status") == "error"
        ),
        "adopted_resolves": active.get("status") == "active",
        "rollback_to_disabled": rolled_back.get("status") == "disabled",
        "rollback_deadline": rollback_seconds <= 60,
        "missing_fail_closed": missing.get("status") == "error",
        "corrupt_fail_closed": corrupt.get("status") == "error",
    }
    return {
        "status": "passed" if all(gates.values()) else "failed",
        "gates": gates,
        "rollback_seconds": round(rollback_seconds, 6),
        "duration_seconds": round(time.monotonic() - started, 6),
        "missing_reason": missing.get("reason"),
        "corrupt_reason": corrupt.get("reason"),
    }


def run_artifact_only_sweep(
    *,
    root: Path,
    evidence_index_manifest: Path,
    output_dir: Path,
    arms: Sequence[str],
    candidate_limit: int = 20,
    additional_working_paths: Sequence[Path] = (),
    audit_paths: Sequence[Path] = (),
    disabled_baseline_manifest: Path | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    output_dir.mkdir(parents=True, exist_ok=True)
    before = _tree_manifest(protected_paths(root))
    registry = PageRegistry(root).load()
    provider = LibraryEvidenceProvider(
        package=load_udc_package(root),
        evidence_index=LibraryEvidenceIndex(evidence_index_manifest),
    )
    pages = []
    for uid, registry_row in sorted(registry.get("pages", {}).items()):
        if (
            not isinstance(registry_row, Mapping)
            or registry_row.get("status") != "active"
        ):
            continue
        pages.append(inference_dto(page_payload(root, str(uid), registry_row)))
    if provider.evidence_index is not None:
        provider.evidence_index.prefetch_dense_queries(
            [provider._page_text(page) for page in pages],
            purpose="explicit",
        )
    rows = [
        provider.candidates(page, arms=arms, limit=candidate_limit) for page in pages
    ]
    restart_check = {"status": "passed", "rows_checked": 0}
    if pages:
        restarted_index = LibraryEvidenceIndex(evidence_index_manifest)
        restarted_index.prefetch_dense_queries(
            [LibraryEvidenceProvider._page_text(pages[0])],
            purpose="explicit",
        )
        restarted_provider = LibraryEvidenceProvider(
            package=load_udc_package(root),
            evidence_index=restarted_index,
        )
        replay = restarted_provider.candidates(
            pages[0],
            arms=arms,
            limit=candidate_limit,
        )
        restart_check = {
            "status": "passed" if replay == rows[0] else "failed",
            "rows_checked": 1,
        }
    overlay_path = output_dir / "full-corpus-overlay.jsonl"
    _write_jsonl(overlay_path, rows)
    after = _tree_manifest(protected_paths(root))
    if before != after:
        raise ClassificationError("artifact-only sweep mutated a protected path")
    storage = storage_manifest(
        working_paths=[
            evidence_index_manifest.parent,
            overlay_path,
            *additional_working_paths,
        ],
        audit_paths=[output_dir / "audit", *audit_paths],
    )
    index_manifest = read_sealed_json(evidence_index_manifest)
    storage.update(
        {
            "build_peak_bytes": int(index_manifest.get("build_peak_bound_bytes") or 0),
            "build_peak_limit_bytes": 6 * 1024**3,
            "build_peak_passed": bool(index_manifest.get("build_peak_gate")),
            "resource_preflight": index_manifest.get("resource_preflight"),
        }
    )
    resolver_drill = (
        _resolver_drill(disabled_baseline_manifest)
        if disabled_baseline_manifest is not None
        else {"status": "skipped", "reason": "disabled baseline not supplied"}
    )
    receipt = {
        "schema": ARTIFACT_SWEEP_SCHEMA,
        "created_at": _now(),
        "status": (
            "passed"
            if storage["working_set_passed"]
            and storage["build_peak_passed"]
            and not storage["audit_adoption_blocked"]
            and restart_check["status"] == "passed"
            and resolver_drill["status"] in {"passed", "skipped"}
            else "failed"
        ),
        "page_count": len(rows),
        "arms": list(arms),
        "candidate_limit": candidate_limit,
        "overlay_path": str(overlay_path),
        "overlay_sha256": sha256_file(overlay_path),
        "protected_manifest_sha256": sha256_bytes(
            json.dumps(
                before,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ),
        "protected_paths_mutated": False,
        "storage": storage,
        "restart_check": restart_check,
        "resolver_drill": resolver_drill,
        "duration_seconds": round(time.monotonic() - started, 3),
    }
    write_sealed_json(output_dir / "receipt.json", receipt, backup=True)
    return receipt
