"""Checkpointed runner for the library-evidence classification pilot."""

from __future__ import annotations

import argparse
import gzip
import json
import subprocess
import time
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from chronovisor.classification.classification_evidence_judgment import (
    ARMS,
    paired_rows,
)
from chronovisor.classification.classification_retention import (
    build_audit_retention_manifest,
    required_update_validation,
)
from chronovisor.core.durable_state import (
    DurableStateError,
    read_sealed_json,
    write_sealed_json,
)
from chronovisor.core.jsonl_write import write_jsonl_atomic as _write_jsonl
from chronovisor.core.runtime_config import (
    load_decision_router_config,
)
from chronovisor.core.store import (
    CHRONOVISOR_ROOT,
    okf_runtime_operation,
    okf_startup_status,
)
from chronovisor.lab.classification_artifact_runner import (
    run_artifact_only_sweep,
    storage_manifest,
)
from chronovisor.lab.classification_library_calibration import (
    evaluate_holdout_gates,
    optional_ablation_decision,
    preregister_evaluation,
    select_dev_configuration,
)
from chronovisor.lab.classification_library_eval import (
    decision_rerun_consistency,
    evaluate_candidate_results,
    evaluate_external_test_results,
    evaluate_paired_decisions,
    unsupported_candidate_notations,
)
from chronovisor.lab.classification_pilot import AuthoritativeCandidateIndex
from chronovisor.recall.classification import (
    ClassificationError,
    classification_authority_status,
    load_udc_package,
)
from chronovisor.recall.classification_bundle import (
    activate_decision_only,
    create_adopted_manifest,
    create_candidate_bundle,
    digest_dag,
    pointer_paths,
    probe_decision_only_authority,
    rollback_authority,
)
from chronovisor.recall.classification_engine import (
    DEFAULT_CANDIDATE_LIMIT,
    ENGINE_VERSION,
    run_consensus_batches,
)
from chronovisor.recall.classification_fixture_set import (
    build_fixture_pool,
    create_disabled_baseline_manifest,
    fixture_set_paths,
    fixture_slice_flags,
    inference_rows,
    lock_fixture_set,
    read_jsonl,
    sha256_bytes,
    sha256_file,
)
from chronovisor.recall.classification_library_evidence import (
    LibraryEvidenceIndex,
    LibraryEvidenceProvider,
    build_dense_index,
    build_source_index,
    external_test_cases,
    resolved_route_identity,
)
from chronovisor.recall.classification_library_sources import (
    atomic_write,
    czech_authority_contract,
    czech_bibliography_contract,
    download_file,
    fetch_oai_window,
    ndl_bibliography_contract,
    ndlsh_contract,
    normalize_ndl_oai_records,
    normalize_ndlsh_rdf,
    parse_marcxml_records,
    stable_sample,
    validate_ndl_provider,
    write_external_package,
)
from chronovisor.recall.classification_resource_burn import run_resource_burn

PILOT_STATE_SCHEMA = "chronovisor.classification-library-pilot-state.v1"
FIXTURE_EPOCH = "epoch-3-library-evidence-v1"
SOURCE_RELEASE = "2026-07-25"
STAGES = (
    "e0_baseline",
    "e0_adjudicate",
    "e1_czech_bibliography",
    "e1_czech_authority",
    "e1_ndlsh",
    "e1_ndl_bibliography",
    "e2_index",
    "e3_candidates",
    "e4_p0",
    "e4_a0f",
    "e4_a1f",
    "e4_paired",
    "e4_resource",
    "e5_dev",
    "e5_holdout_candidates",
    "e5_holdout_p0",
    "e5_holdout_a0f",
    "e5_holdout_a1f",
    "e5_holdout_paired",
    "e5_holdout_evaluate",
    "e6_optional",
    "e7a_sweep",
    "awaiting_explicit_adoption",
    "complete",
)
PROVIDER_ARMS = {
    "B1a": ("B1a",),
    "B1b": ("B1b",),
    "B2": ("B1b", "B2"),
    "B3": ("B1b", "B2", "B3"),
    "C1": ("B1b", "B2", "B3", "C1"),
    "C2": ("B1b", "B2", "B3", "C1", "C2"),
}


def pilot_root(root: Path) -> Path:
    return root / "classification" / "library-evidence"


def state_path(root: Path) -> Path:
    return pilot_root(root) / "state.json"


def _new_state() -> dict[str, Any]:
    return {
        "schema": PILOT_STATE_SCHEMA,
        "status": "running",
        "stage": STAGES[0],
        "fixture_cursor": 0,
        "fixture_accepted": 0,
        "attempts": {},
        "last_error": None,
    }


def load_state(root: Path) -> dict[str, Any]:
    path = state_path(root)
    if not path.exists():
        return _new_state()
    value = read_sealed_json(path)
    if value.get("schema") != PILOT_STATE_SCHEMA:
        raise ClassificationError("library pilot state schema mismatch")
    return value


def save_state(root: Path, state: Mapping[str, Any]) -> None:
    write_sealed_json(state_path(root), dict(state), backup=True)


def _advance(root: Path, state: dict[str, Any], stage: str) -> dict[str, Any]:
    state = {
        **state,
        "status": "running",
        "stage": stage,
        "last_error": None,
    }
    save_state(root, state)
    return state


def _run_stage_timed(
    root: Path,
    stage: str,
    callback: Any,
) -> dict[str, Any]:
    started = time.monotonic()
    callback()
    latest = load_state(root)
    timings = dict(latest.get("stage_timings") or {})
    previous = timings.get(stage)
    previous = previous if isinstance(previous, Mapping) else {}
    timings[stage] = {
        "invocations": int(previous.get("invocations") or 0) + 1,
        "wall_seconds": round(
            float(previous.get("wall_seconds") or 0.0) + (time.monotonic() - started),
            6,
        ),
    }
    latest["stage_timings"] = timings
    save_state(root, latest)
    return latest


def _receipt(root: Path, phase: int, payload: Mapping[str, Any]) -> None:
    write_sealed_json(
        pilot_root(root) / "receipts" / f"phase-e{phase}.json",
        {
            "schema": "chronovisor.classification-library-phase-receipt.v1",
            "phase": f"E{phase}",
            **dict(payload),
        },
        backup=True,
    )


def _git_snapshot(repo_root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    return {
        "head": run("rev-parse", "HEAD"),
        "origin_main": run("rev-parse", "origin/main"),
        "status_porcelain": run("status", "--porcelain=v1"),
    }


def _model_policy() -> dict[str, Any]:
    return {
        "decision_router": load_decision_router_config().__dict__,
        "embedding_route": resolved_route_identity(),
        "embedding_contract": {
            "source_data_class": "derived_snippet",
            "source_sensitivity": "normal",
            "embedding_purpose": "document",
        },
        "engine_version": ENGINE_VERSION,
        "worker_code_sha256": sha256_file(
            Path(__file__).resolve().parents[1]
            / "recall"
            / "classification_model_worker.py"
        ),
        "calibration_code_sha256": sha256_file(
            Path(__file__).with_name("classification_library_calibration.py")
        ),
        "metric_code_sha256": sha256_file(
            Path(__file__).with_name("classification_library_eval.py")
        ),
    }


def _phase_e0_baseline(
    root: Path, state: dict[str, Any], repo_root: Path
) -> dict[str, Any]:
    paths = fixture_set_paths(root, FIXTURE_EPOCH)
    config = load_decision_router_config()
    a0_config = {
        "engine_version": ENGINE_VERSION,
        "candidate_limit": DEFAULT_CANDIDATE_LIMIT,
        "primary_model": config.primary_model,
        "challenger_model": config.challenger_model,
        "tie_break_model": config.tie_break_model,
        "quorum": config.quorum,
        "worker_policy": "current-production-one-page-per-call",
    }
    disabled_path = pilot_root(root) / "bundles" / "disabled-baseline.json"
    create_disabled_baseline_manifest(
        root, a0_config=a0_config, receipt_path=disabled_path
    )
    if not paths.candidates.exists():
        prior_fixture_uids = set()
        for manifest_path in sorted(
            (root / "classification" / "fixtures").rglob("manifest.json")
        ):
            if paths.root in manifest_path.parents:
                continue
            try:
                prior_manifest = read_sealed_json(manifest_path)
            except (DurableStateError, OSError, json.JSONDecodeError):
                prior_manifest = {}
            if not isinstance(prior_manifest, Mapping):
                prior_manifest = {}
            for split in ("dev", "holdout", "reserve"):
                entry = prior_manifest.get(split)
                if not isinstance(entry, Mapping):
                    continue
                split_path = Path(str(entry.get("path") or ""))
                if not split_path.is_file():
                    continue
                for row in read_jsonl(split_path):
                    uid = str(row.get("uid") or "")
                    if uid:
                        prior_fixture_uids.add(uid)
        build_fixture_pool(
            root,
            fixture_epoch=FIXTURE_EPOCH,
            initial_groups=550,
            maximum_groups=800,
            prior_fixture_uids=sorted(prior_fixture_uids),
        )
    legacy_fixture = root / "classification" / "fixtures" / "manifest.json"
    baseline = {
        "schema": "chronovisor.classification-library-baseline.v1",
        "status": "prepared",
        "git": _git_snapshot(repo_root),
        "classification_authority": classification_authority_status(root),
        "decision_authority_unchanged": True,
        "legacy_fixture_path": str(legacy_fixture),
        "legacy_fixture_sha256": (
            sha256_file(legacy_fixture) if legacy_fixture.exists() else None
        ),
        "legacy_holdout_opened_at": (
            read_sealed_json(legacy_fixture).get("holdout", {}).get("opened_at")
            if legacy_fixture.exists()
            else None
        ),
        "a0_config": a0_config,
        "a1_config": {
            "implementation": "committed AuthoritativeCandidateIndex",
            "candidate_limit": 20,
            "embedding_route": resolved_route_identity(),
            "mandatory_secondary_comparator": True,
        },
        "a2_config": {
            "status": "excluded",
            "reason": "unowned-working-tree-challenger-not-a-dependency",
        },
        "disabled_baseline_path": str(disabled_path),
        "disabled_baseline_sha256": sha256_file(disabled_path),
        "production_behavior_mutated": False,
    }
    write_sealed_json(pilot_root(root) / "baseline.json", baseline, backup=True)
    return _advance(root, state, "e0_adjudicate")


def _phase_e0_adjudicate(root: Path, state: dict[str, Any]) -> dict[str, Any]:
    paths = fixture_set_paths(root, FIXTURE_EPOCH)
    candidates = read_jsonl(paths.candidates)
    adjudication_path = paths.root / "adjudication.jsonl"
    accepted = read_jsonl(adjudication_path) if adjudication_path.exists() else []
    cursor = int(state.get("fixture_cursor") or 0)
    minimum_examined = 550
    if cursor >= minimum_examined and len(accepted) >= 500:
        config = load_decision_router_config()
        manifest = lock_fixture_set(
            root,
            fixture_epoch=FIXTURE_EPOCH,
            adjudicated_rows=accepted,
            adjudicator=(
                "local-dual-blind-with-third-adjudication:"
                f"{config.primary_model},{config.challenger_model},"
                f"{config.tie_break_model}:host-notation-validation"
            ),
            dev_count=200,
            holdout_count=300,
        )
        _receipt(
            root,
            0,
            {
                "status": "passed",
                "fixture_manifest": str(paths.manifest),
                "fixture_manifest_sha256": sha256_file(paths.manifest),
                "dev_count": manifest["dev"]["count"],
                "holdout_count": manifest["holdout"]["count"],
                "reserve_count": manifest["reserve"]["count"],
                "gold_free_inference_boundary": True,
                "current_fixture_pointer_changed": False,
            },
        )
        return _advance(root, state, "e1_czech_bibliography")
    if cursor >= len(candidates):
        state.update(
            {
                "status": "blocked",
                "last_error": (
                    f"only {len(accepted)} evaluable fixtures after "
                    f"{len(candidates)} independent groups"
                ),
            }
        )
        save_state(root, state)
        return state
    batch = candidates[cursor : min(len(candidates), cursor + 5)]
    decisions = run_consensus_batches(
        batch,
        root=root,
        batch_size=5,
        purpose="explicit",
        timeout_seconds=1_800,
        run_namespace=f"library-fixture-{FIXTURE_EPOCH}",
        adjudication_mode="dual-blind",
        authority_kind="quorum_v1",
    )
    by_uid = {str(row["uid"]): row for row in decisions}
    for row in batch:
        decision = by_uid[str(row["uid"])]
        if int(decision.get("quorum") or 0) < 2 or decision.get("_invalid_reason"):
            continue
        primary = str(decision.get("primary_notation") or "")
        allowed = {
            str(value.get("notation") or "")
            for value in row.get("candidates") or []
            if isinstance(value, Mapping)
        }
        if primary not in allowed:
            continue
        accepted.append(
            {
                **row,
                "adjudication_status": "accepted",
                "gold_primary_notation": primary,
                "gold_allowed_primary_notations": [primary],
                "gold_secondary_notations": list(
                    decision.get("secondary_notations") or []
                ),
                "gold_rationale": str(decision.get("rationale") or ""),
                "gold_consensus_sha256": str(decision.get("consensus_sha256") or ""),
                "gold_quorum": int(decision.get("quorum") or 0),
                "gold_expected_status": str(
                    decision.get("expected_status") or "proposed"
                ),
                "gold_models": {
                    "primary": decision.get("primary_model"),
                    "challenger": decision.get("challenger_model"),
                    "tie_break": decision.get("tie_break_model"),
                },
                "gold_facets": {
                    "form": (
                        str(row.get("page_type"))
                        if str(row.get("page_type"))
                        in {
                            "decision",
                            "event",
                            "howto",
                            "reference",
                            "architecture",
                            "analysis",
                            "state",
                            "profile",
                            "knowledge",
                        }
                        else "knowledge"
                    ),
                    "lifecycle": (
                        str(row.get("lifecycle"))
                        if str(row.get("lifecycle"))
                        in {"draft", "stable", "deprecated"}
                        else "stable"
                    ),
                    "evidence": "mixed",
                    "sensitivity": (
                        str(row.get("sensitivity"))
                        if str(row.get("sensitivity"))
                        in {"normal", "personal", "restricted", "high"}
                        else "normal"
                    ),
                },
            }
        )
    _write_jsonl(adjudication_path, accepted)
    state.update(
        {
            "fixture_cursor": cursor + len(batch),
            "fixture_accepted": len(accepted),
        }
    )
    save_state(root, state)
    return state


def _source_dir(root: Path, name: str) -> Path:
    return pilot_root(root) / "sources" / name / SOURCE_RELEASE


def _phase_e1_czech_bibliography(root: Path, state: dict[str, Any]) -> dict[str, Any]:
    target = _source_dir(root, "czech-national-bibliography")
    raw, acquisition = fetch_oai_window(
        base_url="https://aleph.nkp.cz/OAI",
        metadata_prefix="marc21",
        from_date="2025-01-01",
        until_date="2026-07-25",
        set_spec="CNB",
        size_cap_bytes=1024**3,
        checkpoint_dir=target / "oai-checkpoint",
    )
    atomic_write(target / "source.xml", raw)
    rows = stable_sample(parse_marcxml_records(raw, authority=False), limit=100_000)
    manifest = write_external_package(
        target,
        contract=czech_bibliography_contract(
            "https://aleph.nkp.cz/OAI?set=CNB&metadataPrefix=marc21"
        ),
        source_release=SOURCE_RELEASE,
        rows=rows,
        acquisition=acquisition,
    )
    if manifest["record_count"] < 1:
        raise ClassificationError("Czech bibliography window returned no records")
    return _advance(root, state, "e1_czech_authority")


def _phase_e1_czech_authority(root: Path, state: dict[str, Any]) -> dict[str, Any]:
    target = _source_dir(root, "czech-topical-authorities")
    archive = target / "aut_ph.xml.gz"
    acquisition = download_file(
        "https://aleph.nkp.cz/data/aut_ph.xml.gz",
        archive,
        size_cap_bytes=64 * 1024**2,
    )
    with gzip.open(archive, "rb") as stream:
        rows = list(parse_marcxml_records(stream, authority=True))
    manifest = write_external_package(
        target,
        contract=czech_authority_contract("https://aleph.nkp.cz/data/aut_ph.xml.gz"),
        source_release=SOURCE_RELEASE,
        rows=rows,
        acquisition=acquisition,
    )
    if manifest["record_count"] < 1:
        raise ClassificationError("Czech topical authority archive is empty")
    return _advance(root, state, "e1_ndlsh")


def _phase_e1_ndlsh(root: Path, state: dict[str, Any]) -> dict[str, Any]:
    target = _source_dir(root, "ndlsh-authority")
    archive = target / "ndlsh-rdf.zip"
    acquisition = download_file(
        "https://id.ndl.go.jp/auth/data/download/ndlsh-rdf.zip",
        archive,
        size_cap_bytes=64 * 1024**2,
    )
    with zipfile.ZipFile(archive) as bundle:
        members = [
            name
            for name in bundle.namelist()
            if name.casefold().endswith((".rdf", ".xml"))
        ]
        if not members:
            raise ClassificationError("NDLSH archive has no RDF/XML member")
        archive_member = min(members)
        with bundle.open(archive_member) as stream:
            rows = list(normalize_ndlsh_rdf(stream))
    manifest = write_external_package(
        target,
        contract=ndlsh_contract(
            "https://id.ndl.go.jp/auth/data/download/ndlsh-rdf.zip"
        ),
        source_release=SOURCE_RELEASE,
        rows=rows,
        acquisition={**acquisition, "archive_member": archive_member},
    )
    if manifest["record_count"] < 1:
        raise ClassificationError("NDLSH authority archive is empty")
    return _advance(root, state, "e1_ndl_bibliography")


def _phase_e1_ndl_bibliography(root: Path, state: dict[str, Any]) -> dict[str, Any]:
    target = _source_dir(root, "ndl-created-bibliography")
    raw, acquisition = fetch_oai_window(
        base_url="https://ndlsearch.ndl.go.jp/api/oaipmh",
        metadata_prefix="dcndl",
        from_date="2026-07-20",
        until_date="2026-07-20",
        set_spec="iss-ndl-opac",
        size_cap_bytes=256 * 1024**2,
        extract_oai_records=True,
        checkpoint_dir=target / "oai-checkpoint",
    )
    atomic_write(target / "source.xml", raw)
    contract = ndl_bibliography_contract(
        "https://ndlsearch.ndl.go.jp/api/oaipmh",
        provider_allowlist=("iss-ndl-opac",),
    )
    rows = []
    rejected = []
    for row in normalize_ndl_oai_records(raw):
        allowed, reason = validate_ndl_provider(row, contract)
        if allowed:
            rows.append(row)
        else:
            rejected.append(str(reason))
    manifest = write_external_package(
        target,
        contract=contract,
        source_release=SOURCE_RELEASE,
        rows=stable_sample(rows, limit=50_000),
        acquisition=acquisition,
        rejected_counts={
            reason: rejected.count(reason) for reason in sorted(set(rejected))
        },
    )
    if manifest["record_count"] < 1:
        raise ClassificationError("NDL allowlisted bibliography window is empty")
    _receipt(
        root,
        1,
        {
            "status": "passed",
            "source_release": SOURCE_RELEASE,
            "packages": [
                str(path)
                for path in sorted(pilot_root(root).glob("sources/*/*/manifest.json"))
            ],
            "repo_or_wheel_bundled": False,
        },
    )
    return _advance(root, state, "e2_index")


def _phase_e2_index(root: Path, state: dict[str, Any]) -> dict[str, Any]:
    manifests = sorted(pilot_root(root).glob("sources/*/*/manifest.json"))
    output = pilot_root(root) / "index" / "evidence.sqlite3"
    index_manifest = output.with_suffix(".manifest.json")
    if index_manifest.exists():
        manifest = read_sealed_json(index_manifest)
    else:
        manifest = build_source_index(
            package_manifest_paths=manifests,
            output_path=output,
            root=root,
        )
    manifest = build_dense_index(
        index_manifest,
        scheduler_purpose="explicit",
    )
    _receipt(
        root,
        2,
        {
            "status": "passed",
            "index_manifest": str(output.with_suffix(".manifest.json")),
            "index_sha256": manifest["index_sha256"],
            "support_count": manifest["support_count"],
            "vocabulary_count": manifest["vocabulary_count"],
            "dense_count": manifest["dense_count"],
            "dense_model": manifest["dense_model"],
            "dense_route_identity": manifest["dense_route_identity"],
            "dense_source_data_class": manifest["dense_source_data_class"],
            "dense_source_sensitivity": manifest["dense_source_sensitivity"],
            "dense_embedding_purpose": manifest["dense_embedding_purpose"],
            "working_set_bytes": manifest["working_set_bytes"],
            "build_peak_bound_bytes": manifest["build_peak_bound_bytes"],
            "build_peak_gate": manifest["build_peak_gate"],
        },
    )
    return _advance(root, state, "e3_candidates")


def _provider(root: Path) -> LibraryEvidenceProvider:
    manifest = pilot_root(root) / "index" / "evidence.manifest.json"
    package = load_udc_package(root)
    return LibraryEvidenceProvider(
        package=package,
        evidence_index=LibraryEvidenceIndex(manifest),
        semantic_index=AuthoritativeCandidateIndex(package),
    )


def _provider_results(
    root: Path,
    *,
    fixture_rows: Sequence[Mapping[str, Any]],
    provider_arms: Sequence[str],
    limit: int,
    provider: LibraryEvidenceProvider | None = None,
) -> list[dict[str, Any]]:
    provider = provider or _provider(root)
    inference = inference_rows(fixture_rows)
    if provider.evidence_index is not None:
        provider.evidence_index.prefetch_dense_queries(
            [provider._page_text(row) for row in inference],
            source_sensitivities=[
                "normal" if row.get("sensitivity") == "normal" else "high"
                for row in inference
            ],
            scheduler_purpose="explicit",
        )
    return [
        provider.candidates(row, arms=provider_arms, limit=limit) for row in inference
    ]


def _phase_e3_candidates(root: Path, state: dict[str, Any]) -> dict[str, Any]:
    paths = fixture_set_paths(root, FIXTURE_EPOCH)
    dev = read_jsonl(paths.dev)
    evaluation_dir = pilot_root(root) / "evaluation"
    all_metrics: dict[str, Any] = {}
    all_external_metrics: dict[str, Any] = {}
    all_external_counts: dict[str, int] = {}
    unsupported_by_arm: dict[str, list[str]] = {}
    provider = _provider(root)
    package = load_udc_package(root)
    source_manifests = sorted(pilot_root(root).glob("sources/*/*/manifest.json"))
    for arm, provider_arms in PROVIDER_ARMS.items():
        result_path = evaluation_dir / f"dev-provider-{arm.casefold()}.jsonl"
        if result_path.exists():
            results = read_jsonl(result_path)
        else:
            results = _provider_results(
                root,
                fixture_rows=dev,
                provider_arms=provider_arms,
                limit=128,
                provider=provider,
            )
            _write_jsonl(result_path, results)
        evaluation = evaluate_candidate_results(
            dev,
            results,
            fields=(
                "official_baseline",
                "official_baseline_tail",
                "official_semantic",
                "external_only",
                "union",
            ),
        )
        all_metrics[arm] = evaluation["metrics"]
        external_fixture_path = (
            evaluation_dir / f"external-test-{arm.casefold()}-fixture.jsonl"
        )
        external_result_path = (
            evaluation_dir / f"external-test-{arm.casefold()}-provider.jsonl"
        )
        if external_fixture_path.exists():
            external_fixture = read_jsonl(external_fixture_path)
        else:
            external_fixture = external_test_cases(
                package_manifest_paths=source_manifests,
                package=package,
                arms=provider_arms,
            )
            _write_jsonl(external_fixture_path, external_fixture)
        if external_result_path.exists():
            external_results = read_jsonl(external_result_path)
        else:
            external_provider = LibraryEvidenceProvider(
                package=package,
                evidence_index=provider.evidence_index,
            )
            external_results = _provider_results(
                root,
                fixture_rows=external_fixture,
                provider_arms=provider_arms,
                limit=128,
                provider=external_provider,
            )
            _write_jsonl(external_result_path, external_results)
        external_evaluation = evaluate_external_test_results(
            external_fixture,
            external_results,
        )
        all_external_metrics[arm] = external_evaluation
        all_external_counts[arm] = len(external_fixture)
        unsupported_by_arm[arm] = unsupported_candidate_notations(
            [*results, *external_results],
            package=package,
        )
    eligible_arms = [
        arm
        for arm in PROVIDER_ARMS
        if (
            float(all_metrics[arm]["union"]["recall_at_12"])
            >= float(all_metrics[arm]["official_baseline"]["recall_at_12"])
            and float(all_metrics[arm]["union"]["recall_at_20"]) >= 0.98
            and float(all_metrics[arm]["union"]["recall_at_128"]) >= 1.0
            and float(all_metrics[arm]["union"]["major_class_error_rate"])
            <= float(all_metrics[arm]["official_baseline"]["major_class_error_rate"])
            and all_external_counts[arm] >= 10_000
            and float(all_external_metrics[arm]["metrics"]["union"]["recall_at_20"])
            >= float(
                all_external_metrics[arm]["metrics"]["official_baseline"][
                    "recall_at_20"
                ]
            )
            and not unsupported_by_arm[arm]
        )
    ]
    ranked_arms = eligible_arms or list(PROVIDER_ARMS)
    best_arm = max(
        ranked_arms,
        key=lambda arm: (
            float(all_metrics[arm]["union"]["recall_at_20"]),
            float(all_metrics[arm]["union"]["ndcg_at_5"]),
            arm,
        ),
    )
    p0 = all_metrics[best_arm]["official_baseline"]
    winner = all_metrics[best_arm]["union"]
    passed = best_arm in eligible_arms
    payload = {
        "schema": "chronovisor.classification-candidate-eval.v1",
        "status": "passed" if passed else "rejected",
        "selected_arm": best_arm,
        "metrics": all_metrics[best_arm],
        "all_arm_metrics": all_metrics,
        "external_test": all_external_metrics[best_arm],
        "all_external_test_metrics": all_external_metrics,
        "mandatory_secondary_a1": all_metrics[best_arm]["official_semantic"],
        "candidate_gates": {
            "baseline_not_degraded": (
                float(winner["recall_at_12"]) >= float(p0["recall_at_12"])
            ),
            "recall_at_20": float(winner["recall_at_20"]) >= 0.98,
            "recall_at_128": float(winner["recall_at_128"]) >= 1.0,
            "major_class_nonincrease": (
                float(winner["major_class_error_rate"])
                <= float(p0["major_class_error_rate"])
            ),
            "external_test_n": all_external_counts[best_arm] >= 10_000,
            "external_test_baseline_not_degraded": (
                float(
                    all_external_metrics[best_arm]["metrics"]["union"]["recall_at_20"]
                )
                >= float(
                    all_external_metrics[best_arm]["metrics"]["official_baseline"][
                        "recall_at_20"
                    ]
                )
            ),
            "unsupported_authority_ids": len(unsupported_by_arm[best_arm]) == 0,
        },
        "unsupported_authority_notations": unsupported_by_arm[best_arm],
        "gold_join_location": "evaluator-only",
        "provider_payload_gold_free": True,
    }
    write_sealed_json(
        evaluation_dir / "candidate-evaluation.json",
        payload,
        backup=True,
    )
    _receipt(
        root,
        3,
        {
            "status": payload["status"],
            "selected_arm": best_arm,
            "candidate_evaluation": str(evaluation_dir / "candidate-evaluation.json"),
        },
    )
    if not passed:
        state.update(
            {
                "status": "blocked",
                "last_error": "candidate-only gate rejected all library evidence arms",
            }
        )
        save_state(root, state)
        return state
    state["selected_arm"] = best_arm
    save_state(root, state)
    return _advance(root, state, "e4_p0")


def _dev_paired_rows(root: Path) -> dict[str, list[dict[str, Any]]]:
    paths = fixture_set_paths(root, FIXTURE_EPOCH)
    dev = read_jsonl(paths.dev)
    selected = str(load_state(root).get("selected_arm") or "")
    results = read_jsonl(
        pilot_root(root) / "evaluation" / f"dev-provider-{selected.casefold()}.jsonl"
    )
    return paired_rows(inference_rows(dev), results)


def _run_decision_stage(
    root: Path,
    state: dict[str, Any],
    *,
    arm: str,
    next_stage: str,
) -> dict[str, Any]:
    paths = fixture_set_paths(root, FIXTURE_EPOCH)
    output = pilot_root(root) / "evaluation" / f"dev-{arm.casefold()}-decisions.jsonl"
    if not output.exists():
        rows = _rows_for_judgment_arm(
            root,
            fixture=read_jsonl(paths.dev),
            arm=arm,
            split="dev",
        )
        decisions = run_consensus_batches(
            rows,
            root=root,
            batch_size=20,
            purpose="explicit",
            timeout_seconds=1_800,
            run_namespace=f"library-evidence-dev-{arm.casefold()}",
            authority_kind="quorum_v1",
        )
        _write_jsonl(output, decisions)
    return _advance(root, state, next_stage)


def _run_latin_square_decisions(
    root: Path,
    *,
    split: str,
) -> dict[str, list[dict[str, Any]]]:
    if split not in {"dev", "holdout"}:
        raise ClassificationError("paired judgment split is invalid")
    paths = fixture_set_paths(root, FIXTURE_EPOCH)
    fixture = read_jsonl(paths.dev if split == "dev" else paths.holdout)
    rows_by_arm = (
        _dev_paired_rows(root) if split == "dev" else _holdout_paired_rows(root)
    )
    evaluation = pilot_root(root) / "evaluation"
    final_paths = {
        arm: evaluation / f"{split}-{arm.casefold()}-decisions.jsonl" for arm in ARMS
    }
    if all(path.exists() for path in final_paths.values()):
        return {arm: read_jsonl(path) for arm, path in final_paths.items()}
    for position in range(len(ARMS)):
        for arm in ARMS:
            cohort_path = (
                evaluation / f"{split}-latin-{position}-{arm.casefold()}.jsonl"
            )
            if cohort_path.exists():
                continue
            cohort = [
                row
                for row in rows_by_arm[arm]
                if list(row.get("latin_square_order") or [])[position] == arm
            ]
            decisions = (
                run_consensus_batches(
                    cohort,
                    root=root,
                    batch_size=20,
                    purpose="explicit",
                    timeout_seconds=1_800,
                    run_namespace=(
                        f"library-evidence-{split}-latin-{position}-{arm.casefold()}"
                    ),
                    authority_kind="quorum_v1",
                )
                if cohort
                else []
            )
            _write_jsonl(cohort_path, decisions)
    fixture_uids = [str(row["uid"]) for row in fixture]
    output: dict[str, list[dict[str, Any]]] = {}
    for arm in ARMS:
        by_uid = {}
        for position in range(len(ARMS)):
            cohort_path = (
                evaluation / f"{split}-latin-{position}-{arm.casefold()}.jsonl"
            )
            for row in read_jsonl(cohort_path):
                by_uid[str(row["uid"])] = row
        if set(by_uid) != set(fixture_uids):
            raise ClassificationError(
                f"{split} Latin-square {arm} result coverage is incomplete"
            )
        output[arm] = [by_uid[uid] for uid in fixture_uids]
        _write_jsonl(final_paths[arm], output[arm])
    return output


def _phase_e4_paired(root: Path, state: dict[str, Any]) -> dict[str, Any]:
    _run_latin_square_decisions(root, split="dev")
    return _advance(root, state, "e4_resource")


def _phase_e4_resource(root: Path, state: dict[str, Any]) -> dict[str, Any]:
    path = pilot_root(root) / "evaluation" / "resource-gate.json"
    if path.exists():
        receipt = read_sealed_json(path)
    else:
        receipt = run_resource_burn(root, samples_per_stage=50)
    _receipt(
        root,
        4,
        {
            "status": ("passed" if receipt.get("status") == "passed" else "failed"),
            "resource_gate": str(path),
        },
    )
    if receipt.get("status") != "passed":
        state.update(
            {
                "status": "blocked",
                "last_error": "resource overlap burn failed closed",
            }
        )
        save_state(root, state)
        return state
    return _advance(root, state, "e5_dev")


def _rows_for_judgment_arm(
    root: Path,
    *,
    fixture: Sequence[Mapping[str, Any]],
    arm: str,
    split: str,
) -> list[dict[str, Any]]:
    if arm == "P0":
        rows = inference_rows(fixture)
        for row, source in zip(rows, fixture, strict=True):
            row["candidates"] = list(source.get("candidates") or [])
        return rows
    if arm in {"A0F", "A1F"}:
        selected = str(load_state(root).get("selected_arm") or "")
        provider_rows = read_jsonl(
            pilot_root(root)
            / "evaluation"
            / f"{split}-provider-{selected.casefold()}.jsonl"
        )
        provider_by_uid = {str(row.get("uid") or ""): row for row in provider_rows}
        field = "official_baseline_tail" if arm == "A0F" else "official_semantic"
        rows = inference_rows(fixture)
        for row in rows:
            provider = provider_by_uid.get(str(row["uid"])) or {}
            row["candidates"] = [
                dict(value)
                for value in (provider.get(field) or [])[:20]
                if isinstance(value, Mapping)
            ]
            if not row["candidates"]:
                raise ClassificationError(
                    f"{arm} candidate set is empty for {row['uid']}"
                )
        return rows
    paired = _dev_paired_rows(root) if split == "dev" else _holdout_paired_rows(root)
    return paired[arm]


def _run_replay(
    root: Path,
    *,
    fixture: Sequence[Mapping[str, Any]],
    arm: str,
    split: str,
) -> list[dict[str, Any]]:
    output = (
        pilot_root(root)
        / "evaluation"
        / f"{split}-{arm.casefold()}-replay-decisions.jsonl"
    )
    if not output.exists():
        decisions = run_consensus_batches(
            _rows_for_judgment_arm(
                root,
                fixture=fixture,
                arm=arm,
                split=split,
            ),
            root=root,
            batch_size=20,
            purpose="explicit",
            timeout_seconds=1_800,
            run_namespace=f"library-evidence-{split}-{arm.casefold()}-replay",
            stage_cache_epoch=f"library-evidence-{split}-replay-2",
            authority_kind="quorum_v1",
        )
        _write_jsonl(output, decisions)
    return read_jsonl(output)


def _slice_power_from_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    holdout = manifest.get("holdout")
    holdout = holdout if isinstance(holdout, Mapping) else {}
    counts = holdout.get("slice_counts")
    counts = counts if isinstance(counts, Mapping) else {}
    required_n = 80
    result = {
        "global": {
            "n": int(holdout.get("count") or 0),
            "required_n": required_n,
            "powered": int(holdout.get("count") or 0) >= required_n,
            "mandatory": True,
        }
    }
    for name, raw_count in counts.items():
        count = int(raw_count or 0)
        powered = count >= required_n
        result[str(name)] = {
            "n": count,
            "required_n": required_n,
            "powered": powered,
            "mandatory": powered,
            "evaluation": "hard" if powered else "reference-only",
        }
    return result


def _powered_slice_gates(
    root: Path,
    *,
    fixture: Sequence[Mapping[str, Any]],
    decisions: Mapping[str, Sequence[Mapping[str, Any]]],
    selected_arm: str,
) -> dict[str, Any]:
    preregistration = read_sealed_json(
        pilot_root(root) / "evaluation" / "preregistration.json"
    )
    results = {}
    for offset, (name, power) in enumerate(
        sorted((preregistration.get("slice_power") or {}).items())
    ):
        if not isinstance(power, Mapping) or not power.get("mandatory"):
            continue
        subset = (
            list(fixture)
            if name == "global"
            else [row for row in fixture if name in fixture_slice_flags(row)]
        )
        subset_uids = {str(row["uid"]) for row in subset}
        subset_decisions = {
            arm: [row for row in arm_rows if str(row.get("uid") or "") in subset_uids]
            for arm, arm_rows in decisions.items()
            if arm in {"A0F", selected_arm}
        }
        evaluation = evaluate_paired_decisions(
            subset,
            subset_decisions,
            baseline_arm="A0F",
            seed=20260726 + offset,
            package=load_udc_package(root),
        )
        comparison = evaluation["comparisons"].get(selected_arm) or {}
        system = evaluation["system_metrics"].get(selected_arm) or {}
        passed = (
            len(subset) >= int(power.get("required_n") or 0)
            and float(
                (comparison.get("exact") or {}).get("ci_lower")
                if (comparison.get("exact") or {}).get("ci_lower") is not None
                else -1.0
            )
            >= -0.01
            and int(evaluation["severe_error_count"].get(selected_arm) or 0) == 0
            and float(system.get("gold_non_hold_system_exact_rate") or 0.0) >= 0.90
            and float(system.get("gold_non_hold_system_hierarchy_rate") or 0.0) >= 0.97
        )
        results[name] = {
            "status": "passed" if passed else "rejected",
            "n": len(subset),
            "required_n": power.get("required_n"),
            "evaluation": evaluation,
        }
    return {
        "status": (
            "passed"
            if results
            and all(value["status"] == "passed" for value in results.values())
            else "rejected"
        ),
        "slices": results,
    }


def _phase_e5_dev(root: Path, state: dict[str, Any]) -> dict[str, Any]:
    paths = fixture_set_paths(root, FIXTURE_EPOCH)
    fixture = read_jsonl(paths.dev)
    decision_dir = pilot_root(root) / "evaluation"
    decisions = {
        arm: read_jsonl(decision_dir / f"dev-{arm.casefold()}-decisions.jsonl")
        for arm in ("P0", "A0F", "A1F", *ARMS)
    }
    paired = evaluate_paired_decisions(
        fixture,
        decisions,
        baseline_arm="A0F",
        seed=20260726,
        package=load_udc_package(root),
    )
    paired_a1 = evaluate_paired_decisions(
        fixture,
        decisions,
        baseline_arm="A1F",
        seed=20260727,
        package=load_udc_package(root),
    )
    arm_metrics = {
        arm: {
            "exact_match_rate": paired["exact_rate"][arm],
            "unexpected_hold_rate": paired["system_metrics"][arm][
                "unexpected_hold_rate"
            ],
            "severe_error_count": paired["severe_error_count"][arm],
            **paired["system_metrics"][arm],
        }
        for arm in decisions
    }
    selection_metrics = {arm: arm_metrics[arm] for arm in ("A0F", *ARMS)}
    selected = select_dev_configuration(
        selection_metrics,
        baseline_arm="A0F",
    )
    payload = {
        **paired,
        "a1_comparisons": paired_a1["comparisons"],
        "selection": selected,
        "arm_metrics": arm_metrics,
    }
    write_sealed_json(decision_dir / "dev-evaluation.json", payload, backup=True)
    if selected.get("status") != "selected":
        _receipt(
            root,
            5,
            {
                "status": "rejected",
                "dev_evaluation": str(decision_dir / "dev-evaluation.json"),
            },
        )
        state.update(
            {"status": "blocked", "last_error": "dev configuration gate failed"}
        )
        save_state(root, state)
        return state
    selected_arm = str(selected["arm"])
    replay = _run_replay(
        root,
        fixture=fixture,
        arm=selected_arm,
        split="dev",
    )
    rerun_consistency = decision_rerun_consistency(
        decisions[selected_arm],
        replay,
    )
    resource = read_sealed_json(decision_dir / "resource-gate.json")
    comparison = paired["comparisons"].get(selected_arm) or {}
    a1_comparison = paired_a1["comparisons"].get(selected_arm) or {}
    exact = comparison.get("exact") or {}
    a1_exact = a1_comparison.get("exact") or {}
    hold_reduction = comparison.get("unexpected_hold_relative_reduction") or {}
    system = paired["system_metrics"][selected_arm]
    candidate_gate = read_sealed_json(decision_dir / "candidate-evaluation.json")
    index_manifest = read_sealed_json(
        pilot_root(root) / "index" / "evidence.manifest.json"
    )
    dev_gate = evaluate_holdout_gates(
        n=len(fixture),
        exact_difference=float(exact.get("difference") or 0.0),
        exact_ci_lower=float(
            exact["ci_lower"] if exact.get("ci_lower") is not None else -1.0
        ),
        unexpected_hold_relative_reduction=float(
            hold_reduction.get("difference") or 0.0
        ),
        unexpected_hold_reduction_ci_lower=float(
            hold_reduction["ci_lower"]
            if hold_reduction.get("ci_lower") is not None
            else -1.0
        ),
        severe_error_count=int(paired["severe_error_count"][selected_arm]),
        unexpected_hold_rate=float(system["unexpected_hold_rate"]),
        expected_hold_escape_count=int(system["expected_hold_escape_count"]),
        proposal_availability=float(system["proposal_availability"]),
        gold_non_hold_system_exact_rate=float(
            system["gold_non_hold_system_exact_rate"]
        ),
        gold_non_hold_system_hierarchy_rate=float(
            system["gold_non_hold_system_hierarchy_rate"]
        ),
        required_facet_macro_f1=float(system["required_facet_macro_f1"]),
        rerun_consistency=rerun_consistency,
        secondary_comparator_passed=(
            float(
                a1_exact["ci_lower"] if a1_exact.get("ci_lower") is not None else -1.0
            )
            >= -0.01
        ),
        recall_gate_passed=candidate_gate.get("status") == "passed",
        resource_gate_passed=resource.get("status") == "passed",
        storage_gate_passed=bool(index_manifest.get("working_set_gate")),
        powered_slices_passed=True,
        require_severe_exact_upper=False,
        require_primary_effect=False,
    )
    if dev_gate["status"] != "passed":
        _receipt(root, 5, {"status": "rejected", "dev_gate": dev_gate})
        state.update({"status": "blocked", "last_error": "dev hard gate failed"})
        save_state(root, state)
        return state
    fixture_manifest = read_sealed_json(paths.manifest)
    preregister_evaluation(
        decision_dir / "preregistration.json",
        fixture_manifest_sha256=sha256_file(paths.manifest),
        evidence_root=sha256_file(
            pilot_root(root) / "index" / "evidence.manifest.json"
        ),
        policy_digest=sha256_bytes(
            json.dumps(
                _model_policy(),
                sort_keys=True,
            ).encode("utf-8")
        ),
        selected_configuration=selected,
        slice_power=_slice_power_from_manifest(fixture_manifest),
    )
    state["selected_judgment_arm"] = selected_arm
    save_state(root, state)
    return _advance(root, state, "e5_holdout_candidates")


def _open_holdout_once(root: Path) -> dict[str, Any]:
    paths = fixture_set_paths(root, FIXTURE_EPOCH)
    manifest = read_sealed_json(paths.manifest)
    opened = manifest.get("holdout", {}).get("opened_at")
    if opened:
        return manifest
    preregistration = read_sealed_json(
        pilot_root(root) / "evaluation" / "preregistration.json"
    )
    if preregistration.get("holdout_opened") is not False:
        raise ClassificationError("Holdout preregistration is not sealed")
    opened_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    manifest["holdout"]["opened_at"] = opened_at
    manifest["holdout"]["opening_reason"] = (
        "single-preregistered-library-evidence-evaluation"
    )
    write_sealed_json(paths.manifest, manifest, backup=True)
    write_sealed_json(
        pilot_root(root) / "evaluation" / "holdout-opening.json",
        {
            "schema": "chronovisor.library-evidence-holdout-opening.v1",
            "opened_at": opened_at,
            "fixture_manifest_sha256": sha256_file(paths.manifest),
            "preregistration_path": str(
                pilot_root(root) / "evaluation" / "preregistration.json"
            ),
            "preregistration_sha256": sha256_file(
                pilot_root(root) / "evaluation" / "preregistration.json"
            ),
            "one_time": True,
        },
        backup=True,
    )
    return manifest


def _phase_e5_holdout_candidates(root: Path, state: dict[str, Any]) -> dict[str, Any]:
    _open_holdout_once(root)
    paths = fixture_set_paths(root, FIXTURE_EPOCH)
    holdout = read_jsonl(paths.holdout)
    selected = str(state.get("selected_arm") or "")
    result_path = (
        pilot_root(root)
        / "evaluation"
        / f"holdout-provider-{selected.casefold()}.jsonl"
    )
    if not result_path.exists():
        results = _provider_results(
            root,
            fixture_rows=holdout,
            provider_arms=PROVIDER_ARMS[selected],
            limit=128,
        )
        _write_jsonl(result_path, results)
    return _advance(root, state, "e5_holdout_p0")


def _holdout_paired_rows(root: Path) -> dict[str, list[dict[str, Any]]]:
    paths = fixture_set_paths(root, FIXTURE_EPOCH)
    holdout = read_jsonl(paths.holdout)
    selected = str(load_state(root).get("selected_arm") or "")
    results = read_jsonl(
        pilot_root(root)
        / "evaluation"
        / f"holdout-provider-{selected.casefold()}.jsonl"
    )
    return paired_rows(inference_rows(holdout), results)


def _run_holdout_decision_stage(
    root: Path,
    state: dict[str, Any],
    *,
    arm: str,
    next_stage: str,
) -> dict[str, Any]:
    paths = fixture_set_paths(root, FIXTURE_EPOCH)
    output = (
        pilot_root(root) / "evaluation" / f"holdout-{arm.casefold()}-decisions.jsonl"
    )
    if not output.exists():
        rows = _rows_for_judgment_arm(
            root,
            fixture=read_jsonl(paths.holdout),
            arm=arm,
            split="holdout",
        )
        decisions = run_consensus_batches(
            rows,
            root=root,
            batch_size=20,
            purpose="explicit",
            timeout_seconds=1_800,
            run_namespace=f"library-evidence-holdout-{arm.casefold()}",
            authority_kind="quorum_v1",
        )
        _write_jsonl(output, decisions)
    return _advance(root, state, next_stage)


def _phase_e5_holdout_paired(
    root: Path,
    state: dict[str, Any],
) -> dict[str, Any]:
    _run_latin_square_decisions(root, split="holdout")
    return _advance(root, state, "e5_holdout_evaluate")


def _phase_e5_holdout_evaluate(root: Path, state: dict[str, Any]) -> dict[str, Any]:
    paths = fixture_set_paths(root, FIXTURE_EPOCH)
    fixture = read_jsonl(paths.holdout)
    evaluation_dir = pilot_root(root) / "evaluation"
    decisions = {
        arm: read_jsonl(evaluation_dir / f"holdout-{arm.casefold()}-decisions.jsonl")
        for arm in ("P0", "A0F", "A1F", *ARMS)
    }
    paired = evaluate_paired_decisions(
        fixture,
        decisions,
        baseline_arm="A0F",
        seed=20260726,
        package=load_udc_package(root),
    )
    paired_a1 = evaluate_paired_decisions(
        fixture,
        decisions,
        baseline_arm="A1F",
        seed=20260727,
        package=load_udc_package(root),
    )
    selected_arm = str(state.get("selected_judgment_arm") or "")
    replay = _run_replay(
        root,
        fixture=fixture,
        arm=selected_arm,
        split="holdout",
    )
    rerun_consistency = decision_rerun_consistency(
        decisions[selected_arm],
        replay,
    )
    comparison = paired["comparisons"].get(selected_arm) or {}
    a1_comparison = paired_a1["comparisons"].get(selected_arm) or {}
    exact = comparison.get("exact") or {}
    a1_exact = a1_comparison.get("exact") or {}
    hold_reduction = comparison.get("unexpected_hold_relative_reduction") or {}
    resource = read_sealed_json(evaluation_dir / "resource-gate.json")
    expected_by_uid = {
        str(row["uid"]): str(row.get("gold_expected_status") or "") == "held"
        for row in fixture
    }
    selected_by_uid = {str(row["uid"]): row for row in decisions[selected_arm]}
    expected_escape_count = sum(
        expected
        and selected_by_uid.get(uid) is not None
        and selected_by_uid[uid].get("status") != "held"
        for uid, expected in expected_by_uid.items()
    )
    system = paired["system_metrics"].get(selected_arm) or {}
    slice_gates = _powered_slice_gates(
        root,
        fixture=fixture,
        decisions=decisions,
        selected_arm=selected_arm,
    )
    candidate_gate = read_sealed_json(evaluation_dir / "candidate-evaluation.json")
    index_manifest = read_sealed_json(
        pilot_root(root) / "index" / "evidence.manifest.json"
    )
    result = evaluate_holdout_gates(
        n=len(fixture),
        exact_difference=float(exact.get("difference") or 0.0),
        exact_ci_lower=float(
            exact["ci_lower"] if exact.get("ci_lower") is not None else -1.0
        ),
        unexpected_hold_relative_reduction=float(
            hold_reduction.get("difference") or 0.0
        ),
        unexpected_hold_reduction_ci_lower=float(
            hold_reduction["ci_lower"]
            if hold_reduction.get("ci_lower") is not None
            else -1.0
        ),
        severe_error_count=int(paired["severe_error_count"].get(selected_arm) or 0),
        unexpected_hold_rate=float(system.get("unexpected_hold_rate") or 0.0),
        expected_hold_escape_count=expected_escape_count,
        proposal_availability=float(system.get("proposal_availability") or 0.0),
        gold_non_hold_system_exact_rate=float(
            system.get("gold_non_hold_system_exact_rate") or 0.0
        ),
        gold_non_hold_system_hierarchy_rate=float(
            system.get("gold_non_hold_system_hierarchy_rate") or 0.0
        ),
        required_facet_macro_f1=float(system.get("required_facet_macro_f1") or 0.0),
        rerun_consistency=rerun_consistency,
        secondary_comparator_passed=(
            float(
                a1_exact["ci_lower"] if a1_exact.get("ci_lower") is not None else -1.0
            )
            >= -0.01
        ),
        recall_gate_passed=candidate_gate.get("status") == "passed",
        resource_gate_passed=resource.get("status") == "passed",
        storage_gate_passed=bool(index_manifest.get("working_set_gate")),
        powered_slices_passed=slice_gates.get("status") == "passed",
    )
    payload = {
        **result,
        "holdout_metrics": {
            "selected_arm": selected_arm,
            "exact_match_rate": paired["exact_rate"].get(selected_arm),
            **paired["system_metrics"].get(selected_arm, {}),
            "severe_error_count": paired["severe_error_count"].get(selected_arm),
            "expected_hold_escape_count": expected_escape_count,
            "exact_difference": exact.get("difference"),
            "exact_ci_lower": exact.get("ci_lower"),
            "exact_ci_upper": exact.get("ci_upper"),
        },
        "paired": paired,
        "a1_comparisons": paired_a1["comparisons"],
        "rerun_consistency": rerun_consistency,
        "powered_slice_gates": slice_gates,
        "fixture_manifest_sha256": sha256_file(paths.manifest),
    }
    output_path = evaluation_dir / "holdout-evaluation.json"
    write_sealed_json(output_path, payload, backup=True)
    _receipt(
        root,
        5,
        {
            "status": result["status"],
            "holdout_evaluation": str(output_path),
            "holdout_evaluation_sha256": sha256_file(output_path),
        },
    )
    if result["status"] != "passed":
        state.update(
            {
                "status": "blocked",
                "last_error": "sealed Holdout hard gate failed",
            }
        )
        save_state(root, state)
        return state
    return _advance(root, state, "e6_optional")


def _phase_e6_optional(root: Path, state: dict[str, Any]) -> dict[str, Any]:
    selected = str(state.get("selected_judgment_arm") or "")
    paths = fixture_set_paths(root, FIXTURE_EPOCH)
    fixture = read_jsonl(paths.dev)
    selected_decisions = read_jsonl(
        pilot_root(root) / "evaluation" / f"dev-{selected.casefold()}-decisions.jsonl"
    )
    provider_rows = read_jsonl(
        pilot_root(root)
        / "evaluation"
        / f"dev-provider-{str(state.get('selected_arm') or '').casefold()}.jsonl"
    )
    decision_by_uid = {str(row.get("uid") or ""): row for row in selected_decisions}
    provider_by_uid = {str(row.get("uid") or ""): row for row in provider_rows}
    auto_holds = sum(
        str(row.get("gold_expected_status") or "") != "held"
        and (decision_by_uid.get(str(row.get("uid") or "")) or {}).get("status")
        == "held"
        for row in fixture
    )
    explicit_link_gap_uids = []
    for row in fixture:
        uid = str(row.get("uid") or "")
        judgment = decision_by_uid.get(uid) or {}
        expected_hold = str(row.get("gold_expected_status") or "") == "held"
        if expected_hold or judgment.get("status") != "held":
            continue
        provider_row = provider_by_uid.get(uid) or {}
        allowed = {
            str(value)
            for value in row.get("gold_allowed_primary_notations")
            or [row.get("gold_primary_notation")]
            if str(value)
        }
        candidate_present = any(
            str(value.get("notation") or "") in allowed
            for value in provider_row.get("union") or []
            if isinstance(value, Mapping)
        )
        c1_link_present = any(
            value.get("vocabulary_role") == "C1" and bool(value.get("relations"))
            for value in provider_row.get("query_expansion") or []
            if isinstance(value, Mapping)
        )
        if candidate_present and not c1_link_present:
            explicit_link_gap_uids.append(uid)

    timings = state.get("stage_timings")
    timings = timings if isinstance(timings, Mapping) else {}
    model_stages = {
        "e4_p0",
        "e4_a0f",
        "e4_a1f",
        "e4_paired",
        "e5_dev",
        "e5_holdout_p0",
        "e5_holdout_a0f",
        "e5_holdout_a1f",
        "e5_holdout_paired",
        "e5_holdout_evaluate",
    }
    observed_stages = {
        name: float(value.get("wall_seconds") or 0.0)
        for name, value in timings.items()
        if isinstance(value, Mapping)
        and name.startswith(("e1_", "e2_", "e3_", "e4_", "e5_"))
    }
    model_wall_seconds = sum(
        seconds for name, seconds in observed_stages.items() if name in model_stages
    )
    pipeline_wall_seconds = sum(observed_stages.values())
    llm_wall_share = (
        model_wall_seconds / pipeline_wall_seconds if pipeline_wall_seconds > 0 else 0.0
    )
    decision = optional_ablation_decision(
        c1_passed=str(state.get("selected_arm") or "") in {"C1", "C2"},
        auto_assignable_hold_count=auto_holds,
        explicit_link_gap_count=len(explicit_link_gap_uids),
        clean_training_pairs=int(
            read_sealed_json(pilot_root(root) / "index" / "evidence.manifest.json").get(
                "support_count"
            )
            or 0
        ),
        llm_wall_time_dominant=llm_wall_share >= 0.5,
    )
    required = any(decision.values())
    _receipt(
        root,
        6,
        {
            "status": "blocked" if required else "skipped-not-required",
            "ablation_decision": decision,
            "measurements": {
                "auto_assignable_hold_count": auto_holds,
                "explicit_link_gap_count": len(explicit_link_gap_uids),
                "explicit_link_gap_uids": explicit_link_gap_uids,
                "model_wall_seconds": round(model_wall_seconds, 6),
                "pipeline_wall_seconds": round(pipeline_wall_seconds, 6),
                "llm_wall_share": round(llm_wall_share, 6),
                "stage_timings": observed_stages,
            },
        },
    )
    if required:
        state.update(
            {
                "status": "blocked",
                "last_error": "optional ablation condition requires a separate run",
            }
        )
        save_state(root, state)
        return state
    return _advance(root, state, "e7a_sweep")


def _phase_e7a_sweep(root: Path, state: dict[str, Any]) -> dict[str, Any]:
    selected = str(state.get("selected_arm") or "")
    sweep_dir = pilot_root(root) / "sweep"
    sweep = run_artifact_only_sweep(
        root=root,
        evidence_index_manifest=(pilot_root(root) / "index" / "evidence.manifest.json"),
        output_dir=sweep_dir,
        arms=PROVIDER_ARMS[selected],
        candidate_limit=20,
        additional_working_paths=(
            pilot_root(root) / "sources",
            pilot_root(root) / "bundles",
        ),
        audit_paths=(pilot_root(root) / "receipts",),
        disabled_baseline_manifest=(
            pilot_root(root) / "bundles" / "disabled-baseline.json"
        ),
    )
    if sweep["status"] != "passed":
        _receipt(root, 7, {"status": "blocked", "sweep": sweep})
        state.update({"status": "blocked", "last_error": "artifact sweep gate failed"})
        save_state(root, state)
        return state
    paths = fixture_set_paths(root, FIXTURE_EPOCH)
    evaluation_path = pilot_root(root) / "evaluation" / "holdout-evaluation.json"
    provider_manifest = pilot_root(root) / "index" / "evidence.manifest.json"
    evaluation = read_sealed_json(evaluation_path)
    index_manifest = read_sealed_json(provider_manifest)
    source_manifests = sorted(pilot_root(root).glob("sources/*/*/manifest.json"))
    dag = digest_dag(
        udc_sha256=load_udc_package(root).checksum,
        source_sha256=sorted(
            {
                digest
                for path in source_manifests
                for digest in (
                    sha256_file(path),
                    str(read_sealed_json(path).get("package_sha256") or ""),
                )
                if digest
            }
            | {
                sha256_file(
                    Path(__file__).resolve().parents[1]
                    / "recall"
                    / "classification_library_sources.py"
                )
            }
        ),
        crosswalk_sha256=[],
        index_sha256=sorted(
            {
                digest
                for digest in (
                    sha256_file(provider_manifest),
                    str(index_manifest.get("index_sha256") or ""),
                    str(index_manifest.get("dense_vectors_sha256") or ""),
                    str(index_manifest.get("dense_row_ids_sha256") or ""),
                    str(index_manifest.get("dense_selection_sha256") or ""),
                )
                if digest
            }
        ),
        provider_code_sha256=sha256_bytes(
            "\n".join(
                (
                    sha256_file(
                        Path(__file__).resolve().parents[1]
                        / "recall"
                        / "classification_library_evidence.py"
                    ),
                    sha256_file(
                        Path(__file__).resolve().parents[1]
                        / "recall"
                        / "classification_embedding_worker.py"
                    ),
                    sha256_file(
                        Path(__file__).resolve().parents[1]
                        / "recall"
                        / "classification_resolver.py"
                    ),
                )
            ).encode("utf-8")
        ),
        evidence_template_sha256=sha256_file(
            Path(__file__).resolve().parents[1]
            / "classification"
            / "classification_evidence_judgment.py"
        ),
        model_policy=_model_policy(),
        run_config={
            "selected_provider_arm": selected,
            "selected_judgment_arm": state.get("selected_judgment_arm"),
            "embedding_route": index_manifest.get("dense_route_identity"),
        },
        input_sha256=str(sweep["overlay_sha256"]),
        fixture_set_sha256=sha256_file(paths.manifest),
        chosen_thresholds={
            "unexpected_hold": 0.08,
            "exact_noninferiority": -0.01,
            "severe": 0,
        },
        dev_result_sha256=sha256_file(
            pilot_root(root) / "evaluation" / "dev-evaluation.json"
        ),
        holdout_result_sha256=sha256_file(evaluation_path),
        metric_schema=str(evaluation.get("schema") or ""),
        observed_metrics=dict(evaluation.get("holdout_metrics") or {}),
        evaluation_status=str(evaluation.get("status") or ""),
    )
    bundle_path = pilot_root(root) / "bundles" / "vnext-candidate.json"
    create_candidate_bundle(
        bundle_path,
        dag=dag,
        evaluation_path=evaluation_path,
        provider_manifest_path=provider_manifest,
        fixture_manifest_path=paths.manifest,
        storage_manifest=sweep["storage"],
        attributions=[
            {
                "source": manifest.get("source_name"),
                "attribution": manifest.get("attribution"),
                "record_license": manifest.get("record_license"),
                "scheme_license": manifest.get("scheme_license"),
                "vocabulary_license": manifest.get("vocabulary_license"),
                "software_license": manifest.get("software_license"),
                "model_license": manifest.get("model_license"),
                "training_corpus_license": manifest.get("training_corpus_license"),
            }
            for manifest in (read_sealed_json(path) for path in source_manifests)
        ],
        run_config={
            "provider_arm": selected,
            "provider_arms": list(PROVIDER_ARMS[selected]),
            "judgment_arm": state.get("selected_judgment_arm"),
            "dense_model": index_manifest.get("dense_model"),
            "embedding_route": index_manifest.get("dense_route_identity"),
            "embedding_source_data_class": index_manifest.get(
                "dense_source_data_class"
            ),
            "embedding_source_sensitivity": index_manifest.get(
                "dense_source_sensitivity"
            ),
            "embedding_purpose": index_manifest.get("dense_embedding_purpose"),
            "dense_model_license": index_manifest.get("dense_model_license"),
            "dense_training_corpus_license": index_manifest.get(
                "dense_training_corpus_license"
            ),
        },
    )
    final_storage = storage_manifest(
        working_paths=(
            pilot_root(root) / "sources",
            pilot_root(root) / "index",
            pilot_root(root) / "bundles",
            sweep_dir / "full-corpus-overlay.jsonl",
        ),
        audit_paths=(
            pilot_root(root) / "receipts",
            pilot_root(root) / "evaluation",
        ),
    )
    final_storage.update(
        {
            "build_peak_bytes": sweep["storage"].get("build_peak_bytes"),
            "build_peak_limit_bytes": sweep["storage"].get("build_peak_limit_bytes"),
            "build_peak_passed": sweep["storage"].get("build_peak_passed"),
            "resource_preflight": sweep["storage"].get("resource_preflight"),
        }
    )
    if (
        not final_storage["working_set_passed"]
        or not final_storage["build_peak_passed"]
        or final_storage["audit_adoption_blocked"]
    ):
        raise ClassificationError("installed vNext exceeds final storage budget")
    _receipt(
        root,
        7,
        {
            "status": "passed",
            "candidate_bundle": str(bundle_path),
            "candidate_bundle_sha256": sha256_file(bundle_path),
            "active_pointer_changed": False,
            "mutation_capability": False,
            "artifact_sweep": str(sweep_dir / "receipt.json"),
            "final_storage": final_storage,
        },
    )
    state.update(
        {
            "status": "awaiting_user",
            "stage": "awaiting_explicit_adoption",
            "last_error": (
                "explicit adoption decision and parent Phase 4 receipt required"
            ),
        }
    )
    save_state(root, state)
    return state


def _supervise_adopted_authority(
    root: Path,
    state: dict[str, Any],
) -> dict[str, Any]:
    receipt_path = pilot_root(root) / "receipts" / "phase-e8.json"
    receipt = read_sealed_json(receipt_path)
    adopted_manifest = Path(str(receipt.get("adopted_manifest") or ""))
    probe = probe_decision_only_authority(
        root,
        expected_manifest_path=adopted_manifest,
    )
    write_sealed_json(
        pilot_root(root) / "supervisor" / "latest.json",
        probe,
        backup=True,
    )
    if probe["status"] == "passed":
        return state

    started = time.monotonic()
    rollback_result: dict[str, Any]
    try:
        rollback_result = rollback_authority(root)
        rollback_error = None
    except (
        ClassificationError,
        DurableStateError,
        OSError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        rollback_result = {}
        rollback_error = f"{type(exc).__name__}: {exc}"
    elapsed = round(time.monotonic() - started, 6)
    rollback_receipt = {
        "schema": "chronovisor.classification-authority-rollback.v1",
        "status": (
            "passed"
            if rollback_error is None and elapsed <= 60.0
            else "manual-recovery-required"
        ),
        "trigger_probe": probe,
        "mutation_disabled_first": True,
        "rollback_result": rollback_result,
        "rollback_error": rollback_error,
        "elapsed_seconds": elapsed,
        "deadline_seconds": 60,
        "deadline_met": elapsed <= 60.0,
    }
    write_sealed_json(
        pilot_root(root) / "supervisor" / "rollback-latest.json",
        rollback_receipt,
        backup=True,
    )
    state.update(
        {
            "status": "blocked",
            "stage": "authority_rolled_back",
            "last_error": (
                "decision-only authority supervisor detected a critical breach"
            ),
        }
    )
    save_state(root, state)
    return state


def run_once(
    *,
    root: Path = CHRONOVISOR_ROOT,
    repo_root: Path,
) -> dict[str, Any]:
    state = load_state(root)
    stage = str(state.get("stage") or STAGES[0])
    if state.get("status") == "complete" and stage == "complete":
        return _supervise_adopted_authority(root, state)
    if state.get("status") in {
        "blocked",
        "observing",
        "awaiting_user",
        "complete",
    }:
        return state
    try:
        handlers = {
            "e0_baseline": lambda: _phase_e0_baseline(root, state, repo_root),
            "e0_adjudicate": lambda: _phase_e0_adjudicate(root, state),
            "e1_czech_bibliography": lambda: _phase_e1_czech_bibliography(root, state),
            "e1_czech_authority": lambda: _phase_e1_czech_authority(root, state),
            "e1_ndlsh": lambda: _phase_e1_ndlsh(root, state),
            "e1_ndl_bibliography": lambda: _phase_e1_ndl_bibliography(root, state),
            "e2_index": lambda: _phase_e2_index(root, state),
            "e3_candidates": lambda: _phase_e3_candidates(root, state),
            "e4_p0": lambda: _run_decision_stage(
                root, state, arm="P0", next_stage="e4_a0f"
            ),
            "e4_a0f": lambda: _run_decision_stage(
                root, state, arm="A0F", next_stage="e4_a1f"
            ),
            "e4_a1f": lambda: _run_decision_stage(
                root, state, arm="A1F", next_stage="e4_paired"
            ),
            "e4_paired": lambda: _phase_e4_paired(root, state),
            "e4_resource": lambda: _phase_e4_resource(root, state),
            "e5_dev": lambda: _phase_e5_dev(root, state),
            "e5_holdout_candidates": lambda: _phase_e5_holdout_candidates(root, state),
            "e5_holdout_p0": lambda: _run_holdout_decision_stage(
                root, state, arm="P0", next_stage="e5_holdout_a0f"
            ),
            "e5_holdout_a0f": lambda: _run_holdout_decision_stage(
                root, state, arm="A0F", next_stage="e5_holdout_a1f"
            ),
            "e5_holdout_a1f": lambda: _run_holdout_decision_stage(
                root, state, arm="A1F", next_stage="e5_holdout_paired"
            ),
            "e5_holdout_paired": lambda: _phase_e5_holdout_paired(root, state),
            "e5_holdout_evaluate": lambda: _phase_e5_holdout_evaluate(root, state),
            "e6_optional": lambda: _phase_e6_optional(root, state),
            "e7a_sweep": lambda: _phase_e7a_sweep(root, state),
        }
        handler = handlers.get(stage)
        return _run_stage_timed(root, stage, handler) if handler is not None else state
    except Exception as exc:
        attempts = dict(state.get("attempts") or {})
        attempts[stage] = int(attempts.get(stage) or 0) + 1
        state.update(
            {
                "status": "retrying" if attempts[stage] < 3 else "blocked",
                "attempts": attempts,
                "last_error": f"{type(exc).__name__}: {exc}",
            }
        )
        save_state(root, state)
        raise


def adopt(
    *,
    root: Path,
    actor: str,
    parent_phase4_receipt: Path,
) -> dict[str, Any]:
    state = load_state(root)
    if state.get("stage") != "awaiting_explicit_adoption":
        raise ClassificationError("pilot is not awaiting explicit adoption")
    candidate_path = pilot_root(root) / "bundles" / "vnext-candidate.json"
    adopted_path = pilot_root(root) / "bundles" / "adopted.json"
    create_adopted_manifest(
        adopted_path,
        candidate_bundle_path=candidate_path,
        actor=actor,
        decision="adopt",
        parent_phase4_receipt=parent_phase4_receipt,
        adoption_policy={
            "authority_epoch": 3,
            "mode": "decision-only/canary",
            "mutation_capability": False,
        },
    )
    paths = fixture_set_paths(root, FIXTURE_EPOCH)
    selected_arm = str(state.get("selected_judgment_arm") or "")
    artifact_paths = [
        *sorted((pilot_root(root) / "receipts").glob("*.json")),
        *sorted((pilot_root(root) / "evaluation").glob("*.json")),
        *sorted((pilot_root(root) / "sources").glob("*/*/manifest.json")),
        paths.manifest,
        pilot_root(root) / "index" / "evidence.manifest.json",
        pilot_root(root)
        / "evaluation"
        / f"dev-{selected_arm.casefold()}-decisions.jsonl",
        pilot_root(root)
        / "evaluation"
        / f"holdout-{selected_arm.casefold()}-decisions.jsonl",
        pilot_root(root) / "evaluation" / "dev-a0f-decisions.jsonl",
        pilot_root(root) / "evaluation" / "holdout-a0f-decisions.jsonl",
        pilot_root(root) / "evaluation" / "dev-a1f-decisions.jsonl",
        pilot_root(root) / "evaluation" / "holdout-a1f-decisions.jsonl",
        candidate_path,
        adopted_path,
    ]
    retention = build_audit_retention_manifest(
        pilot_root(root) / "audit" / "retention.json",
        artifact_paths=artifact_paths,
    )
    if retention["adoption_blocked"]:
        raise ClassificationError("audit retention budget exceeded")
    active_path, _previous_path, _mutation_path = pointer_paths(root)
    if not active_path.exists():
        activate_decision_only(
            root,
            target_path=pilot_root(root) / "bundles" / "disabled-baseline.json",
        )
    try:
        resolved = activate_decision_only(root, target_path=adopted_path)
        activation_probe = probe_decision_only_authority(
            root,
            expected_manifest_path=adopted_path,
        )
        if activation_probe["status"] != "passed":
            raise ClassificationError("post-activation authority probe failed")
    except (
        ClassificationError,
        DurableStateError,
        OSError,
        ValueError,
        json.JSONDecodeError,
    ):
        rollback_authority(root)
        raise
    _receipt(
        root,
        8,
        {
            "status": "passed",
            "authority": resolved,
            "activation_probe": activation_probe,
            "adopted_manifest": str(adopted_path),
            "adopted_manifest_sha256": sha256_file(adopted_path),
            "retention": retention,
            "source_semantic_update_policy": required_update_validation(
                "source-or-index-semantic"
            ),
            "model_policy_update_policy": required_update_validation(
                "model-policy-taxonomy"
            ),
            "mutation_capability": False,
        },
    )
    state.update(
        {
            "status": "complete",
            "stage": "complete",
            "last_error": None,
        }
    )
    save_state(root, state)
    return state


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one checkpoint of the library-evidence pilot."
    )
    parser.add_argument("--root", type=Path, default=CHRONOVISOR_ROOT)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[3],
    )
    parser.add_argument(
        "command",
        choices=("run-once", "status", "adopt"),
        default="run-once",
        nargs="?",
    )
    parser.add_argument("--actor", default="")
    parser.add_argument("--parent-phase4-receipt", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "status":
        return _main_locked(args)
    from chronovisor.core.okf_cutover import OKFStartupBlocked

    try:
        with okf_runtime_operation(args.root):
            return _main_locked(args)
    except OKFStartupBlocked:
        print(
            json.dumps(
                {"status": "blocked", "category": "okf_startup_blocked"},
                sort_keys=True,
            )
        )
        return 75


def _main_locked(args: argparse.Namespace) -> int:
    if args.command != "status" and not okf_startup_status(args.root).allowed:
        print(
            json.dumps(
                {"status": "blocked", "category": "okf_startup_blocked"},
                sort_keys=True,
            )
        )
        return 75
    if args.command == "status":
        result = load_state(args.root)
    elif args.command == "adopt":
        if not args.actor or args.parent_phase4_receipt is None:
            raise SystemExit("adopt requires --actor and --parent-phase4-receipt")
        result = adopt(
            root=args.root,
            actor=args.actor,
            parent_phase4_receipt=args.parent_phase4_receipt,
        )
    else:
        result = run_once(root=args.root, repo_root=args.repo_root)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
