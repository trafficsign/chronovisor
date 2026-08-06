from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).parents[1]
BASELINE = ROOT / "docs" / "refactoring" / "architecture-baseline.json"
EMPTY_EXCEPTION_VIOLATIONS = {
    "ledger_load_error": "",
    "ledger_schema_version": [],
    "ledger_source_baseline_head_drift": [],
    "ledger_baseline_sha256_drift": [],
    "new_exception_ids": [],
    "unrecorded_exception_ids": [],
    "stale_exception_ids": [],
    "baseline_semantic_id_non_subset": [],
    "exception_identity_mismatches": [],
    "exception_content_mismatches": [],
    "exception_metadata_missing": [],
    "duplicate_exception_ids": [],
    "new_cross_domain_site_ids": [],
    "unrecorded_cross_domain_site_ids": [],
    "stale_cross_domain_site_ids": [],
    "baseline_site_semantic_id_non_subset": [],
    "site_identity_mismatches": [],
    "site_content_mismatches": [],
    "duplicate_site_ids": [],
    "site_count_drift": {},
    "ledger_count_drift": {},
    "production_to_lab_edge_growth": [],
    "production_to_lab_static_site_growth": [],
    "production_to_lab_dynamic_site_growth": [],
    "compatibility_contract_drift": {},
    "compatibility_metadata_missing": [],
    "duplicate_compatibility_ids": [],
    "seed_load_error": "",
    "seed_schema_version": [],
    "seed_source_baseline_head_drift": [],
    "previous_seed_source_baseline_head_drift": [],
    "seed_source_baseline_head_history_drift": {},
    "seed_structure_errors": [],
    "seed_universe_drift": {},
    "seed_active_retired_overlap": {},
    "seed_current_drift": {},
    "retired_id_reintroductions": {},
    "seed_retired_regressions": {},
    "seed_active_growth": {},
    "duplicate_seed_ids": {},
    "seed_count_drift": {},
    "previous_seed_load_error": "",
    "previous_seed_schema_version": [],
}


def _load_script() -> ModuleType:
    path = ROOT / "scripts" / "architecture_baseline.py"
    spec = importlib.util.spec_from_file_location("architecture_baseline", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def architecture() -> ModuleType:
    return _load_script()


@pytest.fixture(scope="module")
def baseline() -> dict[str, Any]:
    return json.loads(BASELINE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def current(architecture: ModuleType) -> dict[str, Any]:
    return architecture.scan_repository(ROOT, captured_at="verification")


def _exception_inputs(
    current: dict[str, Any],
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    return (
        copy.deepcopy(current["worktree_source"]),
        copy.deepcopy(current["architecture_exception_ledger"]),
        copy.deepcopy(current["compatibility_contracts"]),
        copy.deepcopy(current["architecture_exception_baseline"]),
        copy.deepcopy(current["frozen_architecture_exception_reference"]),
        copy.deepcopy(current["previous_architecture_exception_baseline"]),
    )


def _exception_violations(
    architecture: ModuleType,
    source: dict[str, Any],
    ledger: dict[str, Any],
    compatibility: list[dict[str, Any]],
    seed: dict[str, Any],
    frozen: dict[str, Any],
    previous_seed: dict[str, Any],
) -> dict[str, Any]:
    return architecture._architecture_exception_violations(
        source,
        ledger,
        compatibility,
        seed,
        frozen,
        previous_seed,
    )


def _sync_seed_and_ledger_counts(
    architecture: ModuleType,
    source: dict[str, Any],
    ledger: dict[str, Any],
    compatibility: list[dict[str, Any]],
    seed: dict[str, Any],
) -> None:
    retired = {
        field: architecture._seed_ids(seed, field, "retired")
        for field in architecture.EXCEPTION_BASELINE_ID_FIELDS
    }
    counts = architecture._exception_counts(source, compatibility, retired)
    seed["counts"] = counts
    ledger["counts"] = counts["active"]
    ledger["baseline_sha256"] = architecture._canonical_sha256(seed)


def _move_seed_id(
    seed: dict[str, Any],
    field: str,
    semantic_id: str,
    *,
    target: str,
) -> None:
    source = "retired" if target == "active" else "active"
    seed[field][source].remove(semantic_id)
    seed[field][target].append(semantic_id)
    seed[field][target].sort()


def test_baseline_records_complete_pre_campaign_inventory(
    baseline: dict[str, Any],
) -> None:
    assert baseline["schema_version"] == 1
    assert baseline["campaign"] == "O"
    assert baseline["campaign_started_at"] == "2026-08-06T12:44:00+09:00"
    assert baseline["captured_at"] != baseline["campaign_started_at"]
    assert (
        "before the authoritative isolated full suite"
        in baseline["captured_at_semantics"]
    )
    assert baseline["repository"]["head_at_capture"] == (
        "d341d575f56c1f3217840e20a0dd144799244a89"
    )
    assert baseline["repository"]["worktree"]["capture_phase"] == (
        "Campaign O frozen pre-full, pre-commit implementation worktree"
    )
    assert baseline["repository"]["pre_campaign_source_head"] == (
        "a17b8704e2a69e1df1dc3466e956edee77fec870"
    )
    assert baseline["source"]["totals"] == {
        "modules": 281,
        "lines": 200634,
        "functions": 4743,
    }
    assert baseline["source"]["package_count"] == 13
    assert "knowledge_graph" in baseline["source"]["packages"]
    assert baseline["source"]["namespace_packages"] == []
    assert len(baseline["source"]["modules"]) == 281
    assert len(baseline["source"]["module_hotspots"]) == 25
    assert len(baseline["source"]["function_hotspots"]) == 50
    assert len(baseline["source"]["python_source_bytes_sha256"]) == 64
    assert baseline["architecture"]["edge_count"] == 95
    assert len(baseline["architecture"]["strongly_connected_components"][0]) == 12
    assert baseline["architecture"]["strongly_connected_components"][1] == ["core"]
    assert len(baseline["console_entrypoints"]) == 51
    assert len(baseline["tracked_launchd_plists"]) == 7
    assets = baseline["source"]["tracked_non_python_assets"]
    assert assets["file_count"] > 0
    assert assets["total_bytes"] > 0
    assert len(assets["manifest_sha256"]) == 64
    assert len(assets["files"]) == assets["file_count"]
    assert all(
        row["path"] and row["bytes"] >= 0 and len(row["sha256"]) == 64
        for row in assets["files"]
    )
    assert assets["frontend_totals"] == {"file_count": 11, "lines": 16979}
    assert assets["asset_hotspots"][0]["path"] == (
        "src/chronovisor/dashboard_static/cortex.js"
    )
    assert assets["asset_hotspots"][0]["lines"] == 5606


def test_baseline_labels_repository_contract_hash_semantics(
    baseline: dict[str, Any], current: dict[str, Any]
) -> None:
    hashes = baseline["contract_hashes"]
    authority = hashes["decision_authority"]
    schema = hashes["production_schema_manifest"]
    signature = hashes["production_signature_manifest"]
    assert authority["lane_contract_case_manifest_sha256"] == (
        "a3a8b84e249b4a6bf36ba3f3584bd6fae45ac4fa521c83c34637879e9b2473eb"
    )
    assert schema["canonical_mapping_sha256"]["sha256"] == (
        "1541981873a0669f5ef7234c9b4490fe3c3f00872d1a584b182a3a33799fbea2"
    )
    assert schema["artifact_validator_sorted_rows_sha256"]["sha256"] == (
        "299b9e5c7c1b5f0195e6437890c111c82cbf63545333eebab83e7b42a870ed58"
    )
    assert signature["sha256"] == (
        "057a9edf3c0d88f579bef8c0836535714aefba73fdba6a15b9b9072f46540f05"
    )
    assert hashes == current["contract_hashes"]


def test_baseline_keeps_live_evidence_out_of_repository_gates(
    baseline: dict[str, Any],
) -> None:
    exclusions = {row["evidence"] for row in baseline["live_only_exclusions"]}
    assert exclusions == {
        "production_runtime_archives_and_running_processes",
        "production_authority_artifact_identity",
        "recall_save_ingest_and_repair_live_behavior",
        "dashboard_cortex_dom_latency_frame_and_memory",
    }


def test_current_architecture_does_not_weaken_baseline(
    architecture: ModuleType,
    baseline: dict[str, Any],
    current: dict[str, Any],
) -> None:
    report = architecture.architecture_fitness(baseline, current)

    assert report["passed"] is True
    assert report["violations"] == {
        "new_edges": [],
        "scc_regressions": [],
        "namespace_packages": [],
        "entrypoint_drift": {},
        "launchd_drift": {},
        "contract_hash_drift": {},
        "architecture_contract_drift": {},
        **EMPTY_EXCEPTION_VIOLATIONS,
    }


def test_architecture_fitness_rejects_new_edge_and_scc_growth(
    architecture: ModuleType,
    baseline: dict[str, Any],
    current: dict[str, Any],
) -> None:
    regressed = copy.deepcopy(current)
    worktree = regressed["worktree_architecture"]
    worktree["edges"].append(["core", "ops"])
    worktree["strongly_connected_components"] = (
        architecture._strongly_connected_components(
            regressed["worktree_source"]["packages"], worktree["edges"]
        )
    )

    report = architecture.architecture_fitness(baseline, regressed)

    assert report["passed"] is False
    assert report["violations"]["new_edges"] == [["core", "ops"]]
    assert len(report["violations"]["scc_regressions"][0]) == 13


def test_namespace_package_is_inventoried_and_rejected(
    architecture: ModuleType,
    baseline: dict[str, Any],
    current: dict[str, Any],
    tmp_path: Path,
) -> None:
    package_root = tmp_path / "src" / "chronovisor"
    (package_root / "core").mkdir(parents=True)
    (package_root / "core" / "__init__.py").write_text("", encoding="utf-8")
    (package_root / "shadow").mkdir()
    (package_root / "shadow" / "worker.py").write_text(
        "from chronovisor import core\n", encoding="utf-8"
    )

    source, edges = architecture._source_inventory(package_root)

    assert source["packages"] == ["core", "shadow"]
    assert source["namespace_packages"] == ["shadow"]
    assert edges == [["shadow", "core"]]

    regressed = copy.deepcopy(current)
    regressed["worktree_source"]["packages"].append("shadow")
    regressed["worktree_source"]["namespace_packages"] = ["shadow"]
    regressed["worktree_architecture"]["strongly_connected_components"].append(
        ["shadow"]
    )
    report = architecture.architecture_fitness(baseline, regressed)
    assert report["passed"] is False
    assert report["violations"]["namespace_packages"] == ["shadow"]


def test_python_source_digest_distinguishes_crlf_from_lf(
    architecture: ModuleType, tmp_path: Path
) -> None:
    package_root = tmp_path / "src" / "chronovisor"
    package = package_root / "core"
    package.mkdir(parents=True)
    (package / "__init__.py").write_bytes(b"")
    module = package / "sample.py"
    module.write_bytes(b"def value():\r\n    return 1\r\n")
    crlf, _edges = architecture._source_inventory(package_root)
    module.write_bytes(b"def value():\n    return 1\n")
    lf, _edges = architecture._source_inventory(package_root)

    assert crlf["totals"] == lf["totals"]
    assert crlf["python_source_bytes_sha256"] != lf["python_source_bytes_sha256"]


def test_entrypoint_and_launchd_surfaces_still_match_baseline(
    baseline: dict[str, Any], current: dict[str, Any]
) -> None:
    assert current["console_entrypoints"] == baseline["console_entrypoints"]
    assert current["tracked_launchd_plists"] == baseline["tracked_launchd_plists"]
    core_contract = next(
        contract
        for contract in baseline["architecture"]["contracts"]
        if contract["name"] == "Core cannot depend on domain or outer layers"
    )
    assert "chronovisor.knowledge_graph" in core_contract["forbidden_modules"]


def test_architecture_fitness_allows_and_reports_package_edge_removal(
    architecture: ModuleType,
    baseline: dict[str, Any],
    current: dict[str, Any],
) -> None:
    reduced = copy.deepcopy(current)
    reduced["worktree_source"]["packages"].remove("knowledge_graph")
    worktree = reduced["worktree_architecture"]
    worktree["edges"] = [
        edge for edge in worktree["edges"] if "knowledge_graph" not in edge
    ]
    worktree["strongly_connected_components"] = (
        architecture._strongly_connected_components(
            reduced["worktree_source"]["packages"], worktree["edges"]
        )
    )

    report = architecture.architecture_fitness(baseline, reduced)

    assert report["passed"] is True
    assert report["observations"]["missing_packages"] == ["knowledge_graph"]
    assert report["violations"]["new_edges"] == []
    assert report["violations"]["scc_regressions"] == []
    assert report["violations"]["namespace_packages"] == []


def test_import_scanner_covers_function_scope_root_and_relative_imports(
    architecture: ModuleType, tmp_path: Path
) -> None:
    package_root = tmp_path / "src" / "chronovisor"
    for package in ("core", "ops"):
        path = package_root / package
        path.mkdir(parents=True)
        (path / "__init__.py").write_text("", encoding="utf-8")
    (package_root / "core" / "sample.py").write_text(
        "def load():\n    from chronovisor import ops\n",
        encoding="utf-8",
    )
    (package_root / "ops" / "sample.py").write_text(
        "def load():\n    from .. import core\n",
        encoding="utf-8",
    )

    _source, edges = architecture._source_inventory(package_root)

    assert edges == [["core", "ops"], ["ops", "core"]]


def test_statement_inventory_tracks_registry_to_implementation_direction(
    architecture: ModuleType, tmp_path: Path
) -> None:
    package_root = tmp_path / "src" / "chronovisor"
    for package in ("decision", "lab", "ops"):
        path = package_root / package
        path.mkdir(parents=True)
        (path / "__init__.py").write_text("", encoding="utf-8")
    (package_root / "decision" / "decision_schema_manifest.py").write_text(
        "def production_decision_schemas():\n"
        "    from chronovisor.ops.schemas import ALPHA_SCHEMA, BETA_SCHEMA\n"
        "\n"
        "def background_decision_schemas():\n"
        "    from chronovisor.decision.graph_decisions import BACKGROUND_SCHEMA\n",
        encoding="utf-8",
    )
    (package_root / "ops" / "sample.py").write_text(
        "import importlib\n"
        "from chronovisor.lab._private import _value\n"
        "from chronovisor.decision.decision_schema_manifest import "
        "NON_DECISION_FIELDS\n"
        "\n"
        "def load():\n"
        "    importlib.import_module('chronovisor.lab.worker')\n"
        "    return __import__('chronovisor.lab.plugin')\n",
        encoding="utf-8",
    )

    source, edges = architecture._source_inventory(package_root)
    rows = source["import_sites"]
    schema_rows = [
        row
        for row in rows
        if row["category"] == "schema_manifest_implementation_import"
    ]

    assert edges == [
        ["decision", "ops"],
        ["ops", "decision"],
        ["ops", "lab"],
    ]
    assert len(schema_rows) == 2
    assert {row["source_module"] for row in schema_rows} == {
        "chronovisor.decision.decision_schema_manifest"
    }
    assert {row["scope"] for row in schema_rows} == {
        "background_decision_schemas",
        "production_decision_schemas",
    }
    assert sum(len(row["symbols"]) for row in schema_rows) == 3
    assert all(
        row["target_module"] != "chronovisor.decision.decision_schema_manifest"
        for row in schema_rows
    )
    assert len({row["semantic_id"] for row in rows}) == len(rows)


def test_schema_registry_detects_all_uppercase_same_package_constants(
    architecture: ModuleType, tmp_path: Path
) -> None:
    package_root = tmp_path / "src" / "chronovisor"
    decision = package_root / "decision"
    decision.mkdir(parents=True)
    (decision / "__init__.py").write_text("", encoding="utf-8")
    (decision / "decision_schema_manifest.py").write_text(
        "def production_decision_schemas():\n"
        "    from chronovisor.decision.schemas import (\n"
        "        SCHEMA, FOO_SCHEMA_VERSION, FOO_SCHEMA_V2, lower_name\n"
        "    )\n"
        "\n"
        "def background_decision_schemas():\n"
        "    from chronovisor.decision.schemas import (\n"
        "        BACKGROUND_SCHEMA_VERSION, mixed_Name\n"
        "    )\n",
        encoding="utf-8",
    )

    source, edges = architecture._source_inventory(package_root)
    schema_rows = [
        row
        for row in source["import_sites"]
        if row["category"] == "schema_manifest_implementation_import"
    ]

    assert edges == []
    assert len(schema_rows) == 2
    assert {row["target_module"] for row in schema_rows} == {
        "chronovisor.decision.schemas"
    }
    assert {row["scope"]: row["symbols"] for row in schema_rows} == {
        "production_decision_schemas": [
            "FOO_SCHEMA_V2",
            "FOO_SCHEMA_VERSION",
            "SCHEMA",
        ],
        "background_decision_schemas": ["BACKGROUND_SCHEMA_VERSION"],
    }


def test_statement_semantic_identity_and_content_ignore_line_moves(
    architecture: ModuleType, tmp_path: Path
) -> None:
    package_root = tmp_path / "src" / "chronovisor"
    for package in ("core", "ops"):
        path = package_root / package
        path.mkdir(parents=True)
        (path / "__init__.py").write_text("", encoding="utf-8")
    module = package_root / "core" / "sample.py"
    module.write_text(
        "from chronovisor.ops import public_api\n",
        encoding="utf-8",
    )
    before, _edges = architecture._source_inventory(package_root)
    module.write_text(
        "\n\n\nfrom chronovisor.ops import public_api\n",
        encoding="utf-8",
    )
    after, _edges = architecture._source_inventory(package_root)

    before_site = before["import_sites"][0]
    after_site = after["import_sites"][0]
    assert before_site["line"] == 1
    assert after_site["line"] == 4
    assert before_site["semantic_id"] == after_site["semantic_id"]
    assert before_site["content_sha256"] == after_site["content_sha256"]
    assert (
        architecture._architecture_exception_rows(before)[0]["semantic_id"]
        == architecture._architecture_exception_rows(after)[0]["semantic_id"]
    )


def test_current_exception_ledger_seed_and_schema_inventory_are_exact(
    architecture: ModuleType,
    current: dict[str, Any],
) -> None:
    source, ledger, compatibility, seed, frozen, previous = _exception_inputs(current)
    previous = copy.deepcopy(seed)
    detected = architecture._architecture_exception_rows(source)
    detected_ids = {row["semantic_id"] for row in detected}
    ledger_ids = {row["semantic_id"] for row in ledger["exceptions"]}
    edge_rows = [
        row for row in ledger["exceptions"] if row["category"] == "cross_domain_edge"
    ]
    raw_cross_sites = [
        row
        for row in source["import_sites"]
        if row["category"] == "cross_domain_import"
    ]
    counts = seed["counts"]["active"]

    assert detected_ids == ledger_ids == set(seed["exception_semantic_ids"]["active"])
    assert all(
        not seed[field]["retired"]
        for field in architecture.EXCEPTION_BASELINE_ID_FIELDS
    )
    assert len(edge_rows) == current["worktree_architecture"]["edge_count"] == 95
    assert sum(len(row["sites"]) for row in edge_rows) == len(raw_cross_sites) == 1267
    assert counts["exceptions"] == 162
    assert counts["by_category"] == {
        "cross_domain_edge": 95,
        "dynamic_import": 24,
        "private_symbol_import": 31,
        "schema_manifest_implementation_import": 12,
    }
    assert counts["production_to_lab_static_sites"] == 20
    assert counts["production_to_lab_dynamic_sites"] == 1
    assert counts["compatibility_by_kind"] == {
        "console_entrypoint": 51,
        "lab_dispatch": 15,
        "module_string": 223,
    }
    assert counts["schema_manifest_implementation"] == {
        "background_decision_schemas": {"statements": 1, "symbols": 5},
        "production_decision_schemas": {"statements": 11, "symbols": 13},
    }
    assert (
        _exception_violations(
            architecture,
            source,
            ledger,
            compatibility,
            seed,
            frozen,
            previous,
        )
        == EMPTY_EXCEPTION_VIOLATIONS
    )


def test_actual_schema_exceptions_are_registry_imports_not_consumers(
    current: dict[str, Any],
) -> None:
    rows = [
        row
        for row in current["architecture_exception_ledger"]["exceptions"]
        if row["category"] == "schema_manifest_implementation_import"
    ]
    production = [row for row in rows if row["scope"] == "production_decision_schemas"]
    background = [row for row in rows if row["scope"] == "background_decision_schemas"]

    assert len(production) == 11
    assert sum(len(row["symbols"]) for row in production) == 13
    assert len(background) == 1
    assert sum(len(row["symbols"]) for row in background) == 5
    assert {row["source_module"] for row in rows} == {
        "chronovisor.decision.decision_schema_manifest"
    }
    assert all(symbol.endswith("_SCHEMA") for row in rows for symbol in row["symbols"])
    assert all(row["target_module"] != row["source_module"] for row in production)
    assert background[0]["target_module"] == "chronovisor.decision.graph_decisions"


def test_new_sensitive_exception_cannot_self_authorize_in_ledger_and_seed(
    architecture: ModuleType,
    current: dict[str, Any],
) -> None:
    source, ledger, compatibility, seed, frozen, previous = _exception_inputs(current)
    previous = copy.deepcopy(seed)
    template = next(
        row for row in source["import_sites"] if row["category"] == "dynamic_import"
    )
    import_site = {
        **template,
        "source_module": f"{template['source_module']}.new_dynamic_site",
        "scope": "<module>",
        "scope_kind": "module",
        "occurrence": 1,
        "line": 1,
    }
    import_site["semantic_id"] = architecture._semantic_id(import_site)
    source["import_sites"].append(import_site)
    ledger["exceptions"].append(
        {
            **import_site,
            "owner": "chronovisor.ops",
            "deadline": "2026-12-31",
            "removal_campaign": "S",
            "rationale": "Synthetic forbidden exception.",
        }
    )
    seed["exception_semantic_ids"]["active"].append(import_site["semantic_id"])
    seed["exception_semantic_ids"]["active"].sort()
    _sync_seed_and_ledger_counts(architecture, source, ledger, compatibility, seed)

    violations = _exception_violations(
        architecture,
        source,
        ledger,
        compatibility,
        seed,
        frozen,
        previous,
    )

    assert violations["new_exception_ids"] == []
    assert violations["unrecorded_exception_ids"] == []
    assert violations["seed_universe_drift"]["exception_semantic_ids"]["added"] == [
        import_site["semantic_id"]
    ]
    assert violations["seed_active_growth"]["exception_semantic_ids"] == [
        import_site["semantic_id"]
    ]


def test_existing_edge_site_growth_cannot_self_authorize_in_ledger_and_seed(
    architecture: ModuleType,
    current: dict[str, Any],
) -> None:
    source, ledger, compatibility, seed, frozen, previous = _exception_inputs(current)
    previous = copy.deepcopy(seed)
    edge = next(
        row
        for row in ledger["exceptions"]
        if row["category"] == "cross_domain_edge"
        and row["target_package"] != "lab"
        and len(row["sites"]) > 1
    )
    template = next(
        row
        for row in source["import_sites"]
        if row["category"] == "cross_domain_import"
        and row["source_package"] == edge["source_package"]
        and row["target_package"] == edge["target_package"]
    )
    import_site = {
        **template,
        "source_module": f"chronovisor.{edge['source_package']}.new_public_site",
        "scope": "<module>",
        "scope_kind": "module",
        "occurrence": 1,
        "line": 1,
        "content_sha256": "a" * 64,
    }
    import_site["semantic_id"] = architecture._semantic_id(import_site)
    source["import_sites"].append(import_site)
    replacement = next(
        row
        for row in architecture._architecture_exception_rows(source)
        if row["category"] == "cross_domain_edge"
        and row["source_package"] == edge["source_package"]
        and row["target_package"] == edge["target_package"]
    )
    replacement.update(
        {field: edge[field] for field in architecture.EXCEPTION_METADATA_FIELDS}
    )
    ledger["exceptions"] = [
        replacement if row["semantic_id"] == edge["semantic_id"] else row
        for row in ledger["exceptions"]
    ]
    seed["cross_domain_site_semantic_ids"]["active"].append(import_site["semantic_id"])
    seed["cross_domain_site_semantic_ids"]["active"].sort()
    _sync_seed_and_ledger_counts(architecture, source, ledger, compatibility, seed)

    violations = _exception_violations(
        architecture,
        source,
        ledger,
        compatibility,
        seed,
        frozen,
        previous,
    )

    assert violations["new_cross_domain_site_ids"] == []
    assert violations["seed_universe_drift"]["cross_domain_site_semantic_ids"][
        "added"
    ] == [import_site["semantic_id"]]
    assert violations["seed_active_growth"]["cross_domain_site_semantic_ids"] == [
        import_site["semantic_id"]
    ]


def test_site_gate_rejects_unrecorded_stale_duplicate_identity_content_and_count(
    architecture: ModuleType,
    current: dict[str, Any],
) -> None:
    inputs = _exception_inputs(current)
    source, ledger, compatibility, seed, frozen, previous = inputs
    edge = next(
        row
        for row in ledger["exceptions"]
        if row["category"] == "cross_domain_edge" and len(row["sites"]) > 1
    )
    site = edge["sites"][0]
    edge["sites"].remove(site)
    violations = _exception_violations(
        architecture, source, ledger, compatibility, seed, frozen, previous
    )
    assert violations["unrecorded_cross_domain_site_ids"] == [site["semantic_id"]]
    assert violations["baseline_site_semantic_id_non_subset"] == [site["semantic_id"]]
    assert violations["site_count_drift"]

    source, ledger, compatibility, seed, frozen, previous = _exception_inputs(current)
    edge = next(
        row
        for row in ledger["exceptions"]
        if row["category"] == "cross_domain_edge" and len(row["sites"]) > 1
    )
    original = edge["sites"][0]
    stale = {
        **original,
        "source_module": f"{original['source_module']}.stale",
        "occurrence": 1,
    }
    stale_identity = {
        "category": "cross_domain_import",
        "source_package": edge["source_package"],
        "target_package": edge["target_package"],
        **stale,
    }
    stale["semantic_id"] = architecture._semantic_id(stale_identity)
    edge["sites"].append(stale)
    edge["sites"].append(copy.deepcopy(original))
    for recorded in edge["sites"]:
        if recorded["semantic_id"] == original["semantic_id"]:
            recorded["content_sha256"] = "0" * 64
    edge["sites"][1]["target_module"] += ".tampered"
    violations = _exception_violations(
        architecture, source, ledger, compatibility, seed, frozen, previous
    )
    assert stale["semantic_id"] in violations["stale_cross_domain_site_ids"]
    assert original["semantic_id"] in violations["duplicate_site_ids"]
    assert original["semantic_id"] in violations["site_content_mismatches"]
    assert violations["site_identity_mismatches"]
    assert violations["site_count_drift"]


def test_exception_rows_reject_unrecorded_stale_duplicate_content_and_metadata(
    architecture: ModuleType,
    current: dict[str, Any],
) -> None:
    source, ledger, compatibility, seed, frozen, previous = _exception_inputs(current)
    dynamic = next(
        row for row in ledger["exceptions"] if row["category"] == "dynamic_import"
    )
    ledger["exceptions"].remove(dynamic)
    violations = _exception_violations(
        architecture, source, ledger, compatibility, seed, frozen, previous
    )
    assert violations["unrecorded_exception_ids"] == [dynamic["semantic_id"]]
    assert violations["baseline_semantic_id_non_subset"] == [dynamic["semantic_id"]]

    source, ledger, compatibility, seed, frozen, previous = _exception_inputs(current)
    dynamic = next(
        row for row in ledger["exceptions"] if row["category"] == "dynamic_import"
    )
    stale = {
        **dynamic,
        "source_module": f"{dynamic['source_module']}.stale",
        "occurrence": 1,
    }
    stale["semantic_id"] = architecture._semantic_id(stale)
    ledger["exceptions"].append(stale)
    duplicate = copy.deepcopy(dynamic)
    ledger["exceptions"].append(duplicate)
    for row in ledger["exceptions"]:
        if row["semantic_id"] == dynamic["semantic_id"]:
            row["content_sha256"] = "0" * 64
    private = next(
        row
        for row in ledger["exceptions"]
        if row["category"] == "private_symbol_import"
    )
    private["target_module"] += ".tampered"
    private.pop("owner")
    ledger["counts"]["exceptions"] += 1
    violations = _exception_violations(
        architecture, source, ledger, compatibility, seed, frozen, previous
    )
    assert stale["semantic_id"] in violations["stale_exception_ids"]
    assert dynamic["semantic_id"] in violations["duplicate_exception_ids"]
    assert dynamic["semantic_id"] in violations["exception_content_mismatches"]
    assert private["semantic_id"] in violations["exception_identity_mismatches"]
    assert {
        "semantic_id": private["semantic_id"],
        "missing": ["owner"],
    } in violations["exception_metadata_missing"]
    assert violations["ledger_count_drift"]


def test_site_deletion_retires_monotonically_and_reintroduction_is_rejected(
    architecture: ModuleType,
    current: dict[str, Any],
) -> None:
    source, ledger, compatibility, seed, frozen, _previous = _exception_inputs(current)
    previous_seed = copy.deepcopy(seed)
    edge = next(
        row
        for row in ledger["exceptions"]
        if row["category"] == "cross_domain_edge"
        and row["target_package"] != "lab"
        and len(row["sites"]) > 1
    )
    site = edge["sites"][0]
    source["import_sites"] = [
        row
        for row in source["import_sites"]
        if row["semantic_id"] != site["semantic_id"]
    ]
    edge["sites"] = [
        row for row in edge["sites"] if row["semantic_id"] != site["semantic_id"]
    ]
    _move_seed_id(
        seed,
        "cross_domain_site_semantic_ids",
        site["semantic_id"],
        target="retired",
    )
    _sync_seed_and_ledger_counts(architecture, source, ledger, compatibility, seed)

    violations = _exception_violations(
        architecture,
        source,
        ledger,
        compatibility,
        seed,
        frozen,
        previous_seed,
    )
    assert violations == EMPTY_EXCEPTION_VIOLATIONS

    retired_seed = copy.deepcopy(seed)
    source, ledger, compatibility, _initial_seed, frozen, _previous = _exception_inputs(
        current
    )
    seed = copy.deepcopy(retired_seed)
    _move_seed_id(
        seed,
        "cross_domain_site_semantic_ids",
        site["semantic_id"],
        target="active",
    )
    _sync_seed_and_ledger_counts(architecture, source, ledger, compatibility, seed)
    violations = _exception_violations(
        architecture,
        source,
        ledger,
        compatibility,
        seed,
        frozen,
        retired_seed,
    )
    assert violations["seed_retired_regressions"]["cross_domain_site_semantic_ids"] == [
        site["semantic_id"]
    ]
    assert violations["seed_active_growth"]["cross_domain_site_semantic_ids"] == [
        site["semantic_id"]
    ]


def test_production_to_lab_static_and_dynamic_site_growth_are_explicit(
    architecture: ModuleType,
    current: dict[str, Any],
) -> None:
    source, ledger, compatibility, seed, frozen, previous = _exception_inputs(current)
    static_template = next(
        row
        for row in source["import_sites"]
        if row["category"] == "cross_domain_import"
        and row["source_package"] == "classification"
        and row["target_package"] == "lab"
    )
    static_site = {
        **static_template,
        "source_module": "chronovisor.classification.new_lab_static",
        "occurrence": 1,
        "line": 1,
    }
    static_site["semantic_id"] = architecture._semantic_id(static_site)
    dynamic_template = next(
        row
        for row in source["import_sites"]
        if row["category"] == "dynamic_import" and row["target_package"] == "lab"
    )
    dynamic_site = {
        **dynamic_template,
        "source_module": "chronovisor.ops.new_lab_dynamic",
        "occurrence": 1,
        "line": 1,
    }
    dynamic_site["semantic_id"] = architecture._semantic_id(dynamic_site)
    source["import_sites"].extend((static_site, dynamic_site))

    violations = _exception_violations(
        architecture,
        source,
        ledger,
        compatibility,
        seed,
        frozen,
        previous,
    )

    assert violations["production_to_lab_static_site_growth"] == [
        static_site["semantic_id"]
    ]
    assert violations["production_to_lab_dynamic_site_growth"] == [
        dynamic_site["semantic_id"]
    ]


def test_exception_metadata_routes_to_real_owner_and_removal_campaign(
    architecture: ModuleType,
    current: dict[str, Any],
) -> None:
    rows = current["architecture_exception_ledger"]["exceptions"]
    by_key = {
        (row["category"], row["source_package"], row["target_package"]): row
        for row in rows
    }

    classification_edge = by_key[("cross_domain_edge", "classification", "lab")]
    assert classification_edge["removal_campaign"] == "P3"
    assert classification_edge["deadline"] == architecture.CAMPAIGN_DEADLINES["P3"]
    assert (
        by_key[("private_symbol_import", "classification", "lab")]["removal_campaign"]
        == "P3"
    )
    assert by_key[("cross_domain_edge", "decision", "lab")]["removal_campaign"] == "P4"
    assert by_key[("cross_domain_edge", "librarian", "lab")]["removal_campaign"] == "P5"
    assert {
        row["removal_campaign"]
        for row in rows
        if row["category"] == "schema_manifest_implementation_import"
    } == {"P6"}
    assert {
        row["removal_campaign"]
        for row in rows
        if row["category"] == "private_symbol_import" and row["target_package"] != "lab"
    } == {"P8"}
    assert {row["removal_campaign"] for row in rows} == {
        "P3",
        "P4",
        "P5",
        "P6",
        "P8",
        "Q",
        "R",
        "S",
    }
    assert all(row["owner"].startswith("chronovisor.") for row in rows)
    assert all(row["owner"] != "chronovisor-architecture" for row in rows)
    assert all(
        row["deadline"] == architecture.CAMPAIGN_DEADLINES[row["removal_campaign"]]
        for row in rows
    )
    custom = {
        "owner": "chronovisor.custom.owner",
        "deadline": "2099-01-01",
        "removal_campaign": "S",
        "rationale": "Keep this reviewed metadata.",
    }
    assert (
        architecture._preserved_metadata(
            custom,
            {field: "fallback" for field in architecture.EXCEPTION_METADATA_FIELDS},
            legacy_owner="chronovisor-architecture",
        )
        == custom
    )


def test_lab_dispatch_compatibility_and_drift_are_protected(
    architecture: ModuleType,
    current: dict[str, Any],
) -> None:
    source, ledger, compatibility, seed, frozen, previous = _exception_inputs(current)
    dispatch = [row for row in compatibility if row["kind"] == "lab_dispatch"]
    assert len(dispatch) == 15
    assert {row["name"] for row in dispatch} == {
        "adoption-corpus",
        "classification-annif",
        "classification-calibrate",
        "classification-library-pilot",
        "classification-migrate",
        "classification-pilot",
        "classification-pilot-v2",
        "classification-profile-pilot",
        "classification-query2doc-pilot",
        "classification-query2doc-unseen",
        "librarian-burn",
        "local-model-eval",
        "model",
        "recall-challengers",
        "research-eval",
    }
    previous_id = dispatch[0]["semantic_id"]
    dispatch[0]["target"] += ".moved"
    dispatch[0]["semantic_id"] = architecture._compatibility_semantic_id(dispatch[0])

    violations = _exception_violations(
        architecture,
        source,
        ledger,
        compatibility,
        seed,
        frozen,
        previous,
    )
    assert violations["compatibility_contract_drift"] == {
        "unrecorded": [dispatch[0]["semantic_id"]],
        "stale": [previous_id],
        "identity_mismatches": [],
    }

    source, ledger, compatibility, seed, frozen, previous = _exception_inputs(current)
    duplicate = copy.deepcopy(ledger["compatibility_contracts"][0])
    ledger["compatibility_contracts"].append(duplicate)
    violations = _exception_violations(
        architecture,
        source,
        ledger,
        compatibility,
        seed,
        frozen,
        previous,
    )
    assert violations["duplicate_compatibility_ids"] == [duplicate["semantic_id"]]


def test_exception_artifacts_are_fresh_and_head_independent(
    architecture: ModuleType,
) -> None:
    recorded_seed = json.loads(
        (
            ROOT / "docs" / "refactoring" / "architecture-exception-baseline.json"
        ).read_text(encoding="utf-8")
    )
    recorded_ledger = json.loads(
        (ROOT / "docs" / "refactoring" / "architecture-exceptions.json").read_text(
            encoding="utf-8"
        )
    )

    assert architecture.build_architecture_exception_baseline(ROOT) == recorded_seed
    assert architecture.build_architecture_exception_ledger(ROOT) == recorded_ledger
    assert "captured_from_head" not in recorded_seed
    assert "captured_from_head" not in recorded_ledger
    assert (
        architecture.FROZEN_EXCEPTION_SOURCE_HEAD
        == "d404a6b20d00e3bcd1d4cdb89edfa5a718c51833"
    )
    assert (
        recorded_seed["source_baseline_head"]
        == architecture.FROZEN_EXCEPTION_SOURCE_HEAD
    )
    assert recorded_ledger["source_baseline_head"] == (
        architecture.FROZEN_EXCEPTION_SOURCE_HEAD
    )


def test_frozen_source_head_rejects_coordinated_current_artifact_drift(
    architecture: ModuleType,
    current: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, ledger, compatibility, seed, frozen, _previous = _exception_inputs(current)
    previous = copy.deepcopy(seed)
    original_head = seed["source_baseline_head"]
    replacement_head = "0" * 40
    monkeypatch.setattr(
        architecture,
        "FROZEN_EXCEPTION_SOURCE_HEAD",
        replacement_head,
    )
    seed["source_baseline_head"] = replacement_head
    frozen["source_baseline_head"] = replacement_head
    ledger["source_baseline_head"] = replacement_head
    ledger["baseline_sha256"] = architecture._canonical_sha256(seed)

    violations = _exception_violations(
        architecture,
        source,
        ledger,
        compatibility,
        seed,
        frozen,
        previous,
    )

    assert violations["ledger_source_baseline_head_drift"] == []
    assert violations["seed_source_baseline_head_drift"] == []
    assert violations["previous_seed_source_baseline_head_drift"] == [original_head]
    assert violations["seed_source_baseline_head_history_drift"] == {
        "previous": original_head,
        "current": replacement_head,
    }


def test_compatibility_policy_requires_mixed_version_observation_and_rollback() -> None:
    adr = (
        ROOT
        / "docs"
        / "architecture"
        / "adr"
        / "0001-layering-dependency-and-compatibility.md"
    ).read_text(encoding="utf-8")
    policy = adr.lower()
    assert "mixed-version" in policy
    assert "observation" in policy
    assert "rollback" in policy
