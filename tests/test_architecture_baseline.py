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
    "new_exception_ids": [],
    "unrecorded_exception_ids": [],
    "stale_exception_ids": [],
    "baseline_semantic_id_non_subset": [],
    "exception_identity_mismatches": [],
    "exception_metadata_missing": [],
    "duplicate_exception_ids": [],
    "production_to_lab_edge_growth": [],
    "compatibility_contract_drift": {},
    "compatibility_metadata_missing": [],
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
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    return (
        copy.deepcopy(current["worktree_source"]),
        copy.deepcopy(current["architecture_exception_ledger"]),
        copy.deepcopy(current["compatibility_contracts"]),
    )


def _exception_violations(
    architecture: ModuleType,
    source: dict[str, Any],
    ledger: dict[str, Any],
    compatibility: list[dict[str, Any]],
) -> dict[str, Any]:
    return architecture._architecture_exception_violations(
        source,
        ledger,
        compatibility,
    )


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


def test_statement_inventory_groups_edges_and_keeps_sensitive_sites(
    architecture: ModuleType, tmp_path: Path
) -> None:
    package_root = tmp_path / "src" / "chronovisor"
    for package in ("decision", "lab", "ops"):
        path = package_root / package
        path.mkdir(parents=True)
        (path / "__init__.py").write_text("", encoding="utf-8")
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
    exceptions = architecture._architecture_exception_rows(source)

    assert edges == [["ops", "decision"], ["ops", "lab"]]
    assert source["import_site_counts"] == {
        "cross_domain_import": 2,
        "dynamic_import": 2,
        "private_symbol_import": 1,
        "schema_manifest_implementation_import": 1,
    }
    assert len({row["semantic_id"] for row in rows}) == len(rows)
    assert {
        row["statement_kind"] for row in rows if row["category"] == "dynamic_import"
    } == {"__import__", "importlib.import_module"}
    assert (
        next(row for row in rows if row["category"] == "dynamic_import")["scope_kind"]
        == "function"
    )
    edge_rows = [row for row in exceptions if row["category"] == "cross_domain_edge"]
    assert {(row["source_package"], row["target_package"]) for row in edge_rows} == {
        ("ops", "decision"),
        ("ops", "lab"),
    }
    assert sum(len(row["sites"]) for row in edge_rows) == 2
    assert {row["category"] for row in exceptions} == {
        "cross_domain_edge",
        "dynamic_import",
        "private_symbol_import",
        "schema_manifest_implementation_import",
    }


def test_statement_semantic_identity_ignores_source_line_moves(
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
    assert (
        architecture._architecture_exception_rows(before)[0]["semantic_id"]
        == architecture._architecture_exception_rows(after)[0]["semantic_id"]
    )


def test_current_exception_ledger_is_complete_and_grouped_by_edge(
    architecture: ModuleType,
    current: dict[str, Any],
) -> None:
    source, ledger, compatibility = _exception_inputs(current)
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

    assert detected_ids == ledger_ids == set(ledger["baseline_semantic_ids"])
    assert len(detected_ids) == len(detected)
    assert len(edge_rows) == current["worktree_architecture"]["edge_count"]
    assert sum(len(row["sites"]) for row in edge_rows) == len(raw_cross_sites)
    assert ledger["counts"]["cross_domain_sites"] == len(raw_cross_sites)
    assert {row["kind"] for row in compatibility} == {
        "console_entrypoint",
        "module_string",
    }
    assert {row["semantic_id"] for row in ledger["compatibility_contracts"]} == {
        row["semantic_id"] for row in compatibility
    }
    assert (
        _exception_violations(
            architecture,
            source,
            ledger,
            compatibility,
        )
        == EMPTY_EXCEPTION_VIOLATIONS
    )


def test_exception_gate_rejects_new_edge_even_when_detailed_row_is_added(
    architecture: ModuleType,
    current: dict[str, Any],
) -> None:
    source, ledger, compatibility = _exception_inputs(current)
    existing_edges = {
        (row["source_package"], row["target_package"])
        for row in source["import_sites"]
        if row["category"] == "cross_domain_import"
    }
    source_package, target_package = next(
        (source_name, target_name)
        for source_name in source["packages"]
        for target_name in source["packages"]
        if source_name != target_name
        and target_name != "lab"
        and (source_name, target_name) not in existing_edges
    )
    template = next(
        row
        for row in source["import_sites"]
        if row["category"] == "cross_domain_import"
    )
    import_site = {
        **template,
        "source_package": source_package,
        "source_module": f"chronovisor.{source_package}.new_dependency",
        "scope_kind": "module",
        "scope": "<module>",
        "statement_kind": "from",
        "target_package": target_package,
        "target_module": f"chronovisor.{target_package}.public_api",
        "symbols": ["public_api"],
        "occurrence": 1,
        "line": 1,
    }
    import_site["semantic_id"] = architecture._semantic_id(import_site)
    source["import_sites"].append(import_site)
    new_edge = next(
        row
        for row in architecture._architecture_exception_rows(source)
        if row["category"] == "cross_domain_edge"
        and row["source_package"] == source_package
        and row["target_package"] == target_package
    )
    ledger["exceptions"].append(
        {
            **new_edge,
            "owner": "test-owner",
            "deadline": "2099-12-31",
            "removal_campaign": "P9",
            "rationale": "Synthetic new edge.",
        }
    )

    violations = _exception_violations(
        architecture,
        source,
        ledger,
        compatibility,
    )

    assert violations["new_exception_ids"] == [new_edge["semantic_id"]]
    assert violations["unrecorded_exception_ids"] == []


def test_exception_gate_rejects_new_sensitive_statement_on_existing_edge(
    architecture: ModuleType,
    current: dict[str, Any],
) -> None:
    source, ledger, compatibility = _exception_inputs(current)
    template = next(
        row for row in source["import_sites"] if row["category"] == "dynamic_import"
    )
    import_site = {
        **template,
        "source_module": f"{template['source_module']}.new_dynamic_site",
        "scope_kind": "module",
        "scope": "<module>",
        "occurrence": 1,
        "line": 1,
    }
    import_site["semantic_id"] = architecture._semantic_id(import_site)
    source["import_sites"].append(import_site)
    ledger["exceptions"].append(
        {
            **import_site,
            "owner": "test-owner",
            "deadline": "2099-12-31",
            "removal_campaign": "P9",
            "rationale": "Synthetic sensitive statement.",
        }
    )

    violations = _exception_violations(
        architecture,
        source,
        ledger,
        compatibility,
    )

    assert violations["new_exception_ids"] == [import_site["semantic_id"]]
    assert violations["unrecorded_exception_ids"] == []


def test_exception_gate_rejects_stale_rows_and_missing_metadata(
    architecture: ModuleType,
    current: dict[str, Any],
) -> None:
    source, ledger, compatibility = _exception_inputs(current)
    edge = next(
        row for row in ledger["exceptions"] if row["category"] == "cross_domain_edge"
    )
    source["import_sites"] = [
        row
        for row in source["import_sites"]
        if row["category"] != "cross_domain_import"
        or row["source_package"] != edge["source_package"]
        or row["target_package"] != edge["target_package"]
    ]
    ledger["exceptions"][0].pop("owner", None)

    violations = _exception_violations(
        architecture,
        source,
        ledger,
        compatibility,
    )

    assert violations["stale_exception_ids"] == [edge["semantic_id"]]
    assert violations["exception_metadata_missing"] == [
        {
            "semantic_id": ledger["exceptions"][0]["semantic_id"],
            "missing": ["owner"],
        }
    ]


def test_exception_gate_rejects_baseline_ids_missing_from_detailed_ledger(
    architecture: ModuleType,
    current: dict[str, Any],
) -> None:
    source, ledger, compatibility = _exception_inputs(current)
    missing_id = "arch:baseline-without-ledger-row"
    ledger["baseline_semantic_ids"].append(missing_id)

    violations = _exception_violations(
        architecture,
        source,
        ledger,
        compatibility,
    )

    assert violations["baseline_semantic_id_non_subset"] == [missing_id]


def test_exception_gate_allows_code_and_ledger_deletion_together(
    architecture: ModuleType,
    current: dict[str, Any],
) -> None:
    source, ledger, compatibility = _exception_inputs(current)
    edge = next(
        row
        for row in ledger["exceptions"]
        if row["category"] == "cross_domain_edge" and row["target_package"] != "lab"
    )
    source["import_sites"] = [
        row
        for row in source["import_sites"]
        if row["source_package"] != edge["source_package"]
        or row["target_package"] != edge["target_package"]
    ]
    remaining_ids = {
        row["semantic_id"] for row in architecture._architecture_exception_rows(source)
    }
    ledger["exceptions"] = [
        row for row in ledger["exceptions"] if row["semantic_id"] in remaining_ids
    ]
    ledger["baseline_semantic_ids"] = [
        semantic_id
        for semantic_id in ledger["baseline_semantic_ids"]
        if semantic_id in remaining_ids
    ]
    ledger["production_to_lab_baseline_semantic_ids"] = [
        semantic_id
        for semantic_id in ledger["production_to_lab_baseline_semantic_ids"]
        if semantic_id in remaining_ids
    ]

    violations = _exception_violations(
        architecture,
        source,
        ledger,
        compatibility,
    )

    assert violations == EMPTY_EXCEPTION_VIOLATIONS


def test_exception_gate_rejects_production_to_lab_growth(
    architecture: ModuleType,
    current: dict[str, Any],
) -> None:
    source, ledger, compatibility = _exception_inputs(current)
    existing_sources = {
        row["source_package"]
        for row in source["import_sites"]
        if row["category"] == "cross_domain_import" and row["target_package"] == "lab"
    }
    source_package = next(
        package
        for package in sorted(architecture.PRODUCTION_PACKAGES)
        if package not in existing_sources
    )
    template = next(
        row
        for row in source["import_sites"]
        if row["category"] == "cross_domain_import"
    )
    import_site = {
        **template,
        "source_package": source_package,
        "source_module": f"chronovisor.{source_package}.new_lab_dependency",
        "scope_kind": "module",
        "scope": "<module>",
        "statement_kind": "from",
        "target_package": "lab",
        "target_module": "chronovisor.lab.contract",
        "symbols": ["contract"],
        "occurrence": 1,
        "line": 1,
    }
    import_site["semantic_id"] = architecture._semantic_id(import_site)
    source["import_sites"].append(import_site)
    new_edge = next(
        row
        for row in architecture._architecture_exception_rows(source)
        if row["category"] == "cross_domain_edge"
        and row["source_package"] == source_package
        and row["target_package"] == "lab"
    )

    violations = _exception_violations(
        architecture,
        source,
        ledger,
        compatibility,
    )

    assert violations["production_to_lab_edge_growth"] == [new_edge["semantic_id"]]


def test_exception_gate_rejects_compatibility_contract_drift(
    architecture: ModuleType,
    current: dict[str, Any],
) -> None:
    source, ledger, compatibility = _exception_inputs(current)
    previous_id = compatibility[0]["semantic_id"]
    compatibility[0]["target"] += ":moved"
    compatibility[0]["semantic_id"] = architecture._compatibility_semantic_id(
        compatibility[0]
    )

    violations = _exception_violations(
        architecture,
        source,
        ledger,
        compatibility,
    )

    assert violations["compatibility_contract_drift"] == {
        "unrecorded": [compatibility[0]["semantic_id"]],
        "stale": [previous_id],
        "identity_mismatches": [],
    }
