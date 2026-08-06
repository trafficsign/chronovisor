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


def test_baseline_records_complete_pre_campaign_inventory(
    baseline: dict[str, Any],
) -> None:
    assert baseline["schema_version"] == 1
    assert baseline["campaign"] == "O"
    assert baseline["campaign_started_at"] == "2026-08-06T12:44:00+09:00"
    assert baseline["captured_at"] != baseline["campaign_started_at"]
    assert "before the authoritative isolated full suite" in baseline[
        "captured_at_semantics"
    ]
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
    exclusions = {
        row["evidence"] for row in baseline["live_only_exclusions"]
    }
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
    assert (
        crlf["python_source_bytes_sha256"]
        != lf["python_source_bytes_sha256"]
    )


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
