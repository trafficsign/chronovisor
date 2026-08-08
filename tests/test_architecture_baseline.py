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
P2_RETIRED_PRIVATE_EXCEPTION_ID = (
    "arch:97b784f974ebdf78ae0226731f9b421e381ac04bad98dd104435367c618f52e9"
)
P2_RETIRED_SITE_IDS = (
    "arch:0d799ce1e29887c64caf095e2640a148aa9e90326aaef4c45a476ae13c33e85b",
    "arch:7d200838738a191b653ce5829735d820b786746b2a6329bf9cc98610ff57dd32",
    "arch:a36a5c65819a0aab93f8c971dee29537ef65da12069950b9fdab4b270bb9c9d7",
    "arch:c7056bd3c53d85c0dfb27edae2aead5bac51065e169b7b87ef3f21b4363d3ca2",
    "arch:e821d535f1969c514f6f49a2338ec36e382d3a6b82040c593bf28f580d86c97d",
)
P3_RETIRED_CLASSIFICATION_LAB_EDGE_ID = (
    "arch:0f37f016df9c2c328a0edc59fd3b3c4b8039921bde3fdaaede27d59f34be9f60"
)
P3_RETIRED_PROVIDER_SITE_ID = (
    "arch:7923c8117584e014f1a6d93283fb7ce9eb012f2cdb7e6228171c5fffb58aecc1"
)
P4_RETIRED_SEARCH_LAB_PRIVATE_EXCEPTION_ID = (
    "arch:93b9e7819071ee5c3aecca9db43262f5ef56ac9a8b6085178911353e0c0d25b7"
)
P4_RETIRED_SEARCH_LAB_EDGE_ID = (
    "arch:95ae240af520b7872ef42f68074e124a12bc04c5f937d508f8135a6a8a23d03a"
)
P4_RETIRED_SEARCH_LAB_SITE_ID = (
    "arch:d4b968e37807dc285676f8f73e74c3740782f10133aa0fc2e09acf42dfab3636"
)
P4C_RETIRED_EXCEPTION_IDS = (
    "arch:53527ec1ad690243e3d5a03be50728716d34df8f1c222a10e652b40e872e01c0",
    "arch:61a565cf4549c1807a69940ad985ef4c66cce6a47176773d4d10add426b238fd",
    "arch:89e88a117c7d284e8325a0d39a301530316177b1e0054b3eaf0108d82f727f2d",
)
P4C_RETIRED_DECISION_LAB_SITE_IDS = (
    "arch:1bfb36c1612203a1ed27ea9feb74fe67449555be6e6f1663624f13584342514c",
    "arch:260382e328ebb07af32a060e01f277e15c3903d29e58307404e2f21c8e703b0c",
    "arch:67561ef5193c9f26cc02f98c0b2b59ffb78f6d722fb1972227a0ff1778195d7d",
    "arch:6b6b9de176ed950d9f3bc08bd8157fff4affee7a28dd0c2bb58981ee4d7a63e5",
    "arch:c4f90d606485dac03d16fa59b11938455864828332e2ee5f13418a0a7a0c54b3",
    "arch:c70da36211b3b4a165f59ba6281e2a47bc9bcbd134eaa5b155dd29de5b1f5db1",
    "arch:c8f86543c1dee07e9e0bfe5594699726c3f688b1405bcb3570ec8670e37ebe4a",
    "arch:cdf4e056944483b580134c81faa876f2d16dc7d92545a4f5b24ed6bca6a52431",
    "arch:fd60ca9f13016ac01869596b80083d7f7cf3c21ffeff491d104c60643b703003",
)
P4C_RETIRED_MOVED_SITE_IDS = (
    "arch:0e2f4f55e57ae72b55a9ac06a4657a51c188922dd8e2e714e98024f49138251c",
    "arch:1379aba196c611f97d7178a0364f4486f243e9da8b8840e141cb6faae1b6b3ab",
    "arch:1ac58df524d4b41f324f4dff7dd17ddf5fad7635b4213d633cb030ccebabe29e",
    "arch:1b6f386c9960dbb4ea63b3d1ae5284ec9d721aaf22b74790a86197d00f13a090",
    "arch:1dd8a86eae5bc73ea226a6e5a70fabbe49d987ccc6bd203029c84910d2b371c0",
    "arch:3b17a90248ef53b2292ff341faac23a1e5e788bb3ccf457b5214abedbd057892",
    "arch:54d33e971f9381c55ef8197914dac30c05485f7270ab62c9cc8ae368534519cd",
    "arch:59c991a217d25c0c8d89ecb4919405927c090fd27377ce37a12cce437d21aaaf",
    "arch:7225cb98e0b4893dc35179535642999cd467c97f6d18a02e531ce34ae6a765bc",
    "arch:7e6db621af61ead96ab24fde3184087f82cb86d9e80ed4c1a7155ccaf29c2fbc",
    "arch:8734a683868e5bb81a039b9e9332751da5a6e70e7d9161c4574a9c8d86d70a3d",
    "arch:cb3f2a5f7e521eafb107c37f60252650bf07a67300160a2eb4336647adcb87ce",
)
P4A_RETIRED_OPS_LAB_EDGE_ID = (
    "arch:eb2b813a6f4420f60df925a80375c90dea7fc85dcc15240272b743e739b509a7"
)
P4A_RETIRED_OPS_LAB_STATIC_SITE_ID = (
    "arch:7dc55845add1f43bc44723139de24db48ad92ca39be88c436b8d3c6bc5f960b0"
)
P4A_RETIRED_OPS_LAB_DYNAMIC_SITE_ID = (
    "arch:f2679d78c5af6d67a6f7c00c1740b8328bf32cd33253d61369db71f29ee0b8e9"
)
P4A_RETIRED_MOVED_SITE_IDS = (
    "arch:054633d2f397258b2ae27fe2e8771ff11a3b07d2ba2b3237d552518e463a73f7",
    "arch:37ff531c153962a561682fc13ead43a4142dc0d1258701d3feb477586f5b2522",
    "arch:57a4f30b14d566feba89b017d42d0932e701d035baa2e52a81828cd72db17fd1",
    "arch:6f981f4b5cc6d6f3f7d50c5e65d194bf5dcbee6820a647f47030f43f52f76016",
    "arch:dd7c91d6e2347693b73918a9e01e96c08d74031108131b0c44655012ba1a023e",
    "arch:e3e8f26b41ac88d106b2d9a357f71293eba715a14a27cebd717278fa49d66f95",
    "arch:e767a86ee1c2c1ba318ad5fec467355b673266eedea548500ecf197b340eebbc",
)
RETIREMENT_HISTORY = {
    "exception_semantic_ids": (
        P3_RETIRED_CLASSIFICATION_LAB_EDGE_ID,
        *P4C_RETIRED_EXCEPTION_IDS,
        P4_RETIRED_SEARCH_LAB_PRIVATE_EXCEPTION_ID,
        P4_RETIRED_SEARCH_LAB_EDGE_ID,
        P2_RETIRED_PRIVATE_EXCEPTION_ID,
        P4A_RETIRED_OPS_LAB_EDGE_ID,
        P4A_RETIRED_OPS_LAB_DYNAMIC_SITE_ID,
    ),
    "cross_domain_site_semantic_ids": tuple(
        sorted(
            (
                *P2_RETIRED_SITE_IDS,
                P3_RETIRED_PROVIDER_SITE_ID,
                *P4C_RETIRED_DECISION_LAB_SITE_IDS,
                *P4C_RETIRED_MOVED_SITE_IDS,
                P4A_RETIRED_OPS_LAB_STATIC_SITE_ID,
                *P4A_RETIRED_MOVED_SITE_IDS,
                P4_RETIRED_SEARCH_LAB_SITE_ID,
            )
        )
    ),
    "production_to_lab_edge_semantic_ids": (
        P3_RETIRED_CLASSIFICATION_LAB_EDGE_ID,
        P4C_RETIRED_EXCEPTION_IDS[1],
        P4_RETIRED_SEARCH_LAB_EDGE_ID,
        P4A_RETIRED_OPS_LAB_EDGE_ID,
    ),
    "production_to_lab_static_site_semantic_ids": tuple(
        sorted(
            (
                *P2_RETIRED_SITE_IDS,
                P3_RETIRED_PROVIDER_SITE_ID,
                *P4C_RETIRED_DECISION_LAB_SITE_IDS,
                P4A_RETIRED_OPS_LAB_STATIC_SITE_ID,
                P4_RETIRED_SEARCH_LAB_SITE_ID,
            )
        )
    ),
    "production_to_lab_dynamic_site_semantic_ids": (
        P4A_RETIRED_OPS_LAB_DYNAMIC_SITE_ID,
    ),
    "compatibility_semantic_ids": (),
}

DiagnosticPath = tuple[str | int, ...]


def _diagnostic_line_entries(
    value: Any, path: DiagnosticPath = ()
) -> list[tuple[DiagnosticPath, Any]]:
    if isinstance(value, dict):
        entries: list[tuple[DiagnosticPath, Any]] = []
        for key, item in value.items():
            item_path = (*path, key)
            if key == "line":
                entries.append((item_path, item))
            else:
                entries.extend(_diagnostic_line_entries(item, item_path))
        return entries
    if isinstance(value, list):
        return [
            entry
            for index, item in enumerate(value)
            for entry in _diagnostic_line_entries(item, (*path, index))
        ]
    return []


def _without_diagnostic_lines(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _without_diagnostic_lines(item)
            for key, item in value.items()
            if key != "line"
        }
    if isinstance(value, list):
        return [_without_diagnostic_lines(item) for item in value]
    return value


def _assert_diagnostic_line_contract(recorded: Any, built: Any) -> None:
    recorded_entries = _diagnostic_line_entries(recorded)
    built_entries = _diagnostic_line_entries(built)
    assert recorded_entries
    assert built_entries
    assert all(
        type(line) is int
        for _path, line in (*recorded_entries, *built_entries)
    )
    recorded_paths = [path for path, _line in recorded_entries]
    built_paths = [path for path, _line in built_entries]
    assert len(recorded_paths) == len(built_paths)
    assert set(recorded_paths) == set(built_paths)
    assert _without_diagnostic_lines(built) == _without_diagnostic_lines(recorded)


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


def _assert_exact_retirement_history(
    architecture: ModuleType,
    seed: dict[str, Any],
) -> None:
    assert set(RETIREMENT_HISTORY) == set(
        architecture.EXCEPTION_BASELINE_ID_FIELDS
    )
    assert {
        field: tuple(seed[field]["retired"])
        for field in architecture.EXCEPTION_BASELINE_ID_FIELDS
    } == RETIREMENT_HISTORY
    assert seed["counts"]["retired"] == {
        field: len(retired_ids)
        for field, retired_ids in RETIREMENT_HISTORY.items()
    }
    assert all(
        set(seed[field]["active"]).isdisjoint(seed[field]["retired"])
        for field in architecture.EXCEPTION_BASELINE_ID_FIELDS
    )


def _without_persisted_retirement_history(
    architecture: ModuleType,
    seed: dict[str, Any],
) -> dict[str, Any]:
    normalized = copy.deepcopy(seed)
    _assert_exact_retirement_history(architecture, normalized)
    for field, retired_ids in RETIREMENT_HISTORY.items():
        normalized[field]["active"] = sorted(
            (*normalized[field]["active"], *retired_ids)
        )
        normalized[field]["retired"] = []
    normalized["counts"]["retired"] = {
        field: 0 for field in RETIREMENT_HISTORY
    }
    active_counts = normalized["counts"]["active"]
    active_counts["exceptions"] += len(RETIREMENT_HISTORY["exception_semantic_ids"])
    active_counts["by_category"]["cross_domain_edge"] += 4
    active_counts["by_category"]["dynamic_import"] += 1
    active_counts["by_category"]["private_symbol_import"] += 4
    active_counts["cross_domain_sites"] += len(
        RETIREMENT_HISTORY["cross_domain_site_semantic_ids"]
    )
    active_counts["production_to_lab_edges"] += len(
        RETIREMENT_HISTORY["production_to_lab_edge_semantic_ids"]
    )
    active_counts["production_to_lab_static_sites"] += len(
        RETIREMENT_HISTORY["production_to_lab_static_site_semantic_ids"]
    )
    active_counts["production_to_lab_dynamic_sites"] += len(
        RETIREMENT_HISTORY["production_to_lab_dynamic_site_semantic_ids"]
    )
    return normalized


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
    _assert_exact_retirement_history(architecture, seed)
    assert len(edge_rows) == current["worktree_architecture"]["edge_count"] == 91
    assert sum(len(row["sites"]) for row in edge_rows) == len(raw_cross_sites) == 1251
    assert {
        field: counts[field]
        for field in (
            "exceptions",
            "cross_domain_sites",
            "production_to_lab_edges",
            "production_to_lab_static_sites",
            "production_to_lab_dynamic_sites",
            "compatibility_contracts",
        )
    } == {
        "exceptions": 153,
        "cross_domain_sites": 1251,
        "production_to_lab_edges": 1,
        "production_to_lab_static_sites": 3,
        "production_to_lab_dynamic_sites": 0,
        "compatibility_contracts": 289,
    }
    assert counts["by_category"] == {
        "cross_domain_edge": 91,
        "dynamic_import": 23,
        "private_symbol_import": 27,
        "schema_manifest_implementation_import": 12,
    }
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


@pytest.mark.parametrize("mutation", ["extra", "wrong"])
def test_retirement_history_rejects_extra_or_wrong_id(
    architecture: ModuleType,
    current: dict[str, Any],
    mutation: str,
) -> None:
    seed = copy.deepcopy(current["architecture_exception_baseline"])
    if mutation == "extra":
        seed["exception_semantic_ids"]["retired"].append("arch:" + "0" * 64)
    else:
        seed["cross_domain_site_semantic_ids"]["retired"][0] = "arch:" + "f" * 64

    with pytest.raises(AssertionError):
        _assert_exact_retirement_history(architecture, seed)


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


def test_existing_edge_replacement_site_can_be_seeded_with_exact_ledger_match(
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

    assert violations == EMPTY_EXCEPTION_VIOLATIONS


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
    assert violations["seed_active_growth"] == {}


def test_production_to_lab_static_and_dynamic_site_growth_are_explicit(
    architecture: ModuleType,
    current: dict[str, Any],
) -> None:
    source, ledger, compatibility, seed, frozen, previous = _exception_inputs(current)
    static_template = next(
        row
        for row in source["import_sites"]
        if row["category"] == "cross_domain_import"
        and row["target_package"] == "lab"
    )
    static_site = {
        **static_template,
        "source_package": "classification",
        "source_module": "chronovisor.classification.new_lab_static",
        "occurrence": 1,
        "line": 1,
    }
    static_site["semantic_id"] = architecture._semantic_id(static_site)
    dynamic_template = next(
        row
        for row in source["import_sites"]
        if row["category"] == "dynamic_import"
    )
    dynamic_site = {
        **dynamic_template,
        "source_package": "ops",
        "source_module": "chronovisor.ops.new_lab_dynamic",
        "target_package": "lab",
        "target_module": "chronovisor.lab.new_dynamic",
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

    assert ("cross_domain_edge", "classification", "lab") not in by_key
    assert ("private_symbol_import", "classification", "lab") not in by_key
    assert ("cross_domain_edge", "search", "lab") not in by_key
    assert ("private_symbol_import", "search", "lab") not in by_key
    assert ("cross_domain_edge", "decision", "lab") not in by_key
    assert ("private_symbol_import", "decision", "lab") not in by_key
    assert ("cross_domain_edge", "ops", "lab") not in by_key
    assert P2_RETIRED_PRIVATE_EXCEPTION_ID not in {
        row["semantic_id"] for row in rows
    }
    assert current["architecture_exception_baseline"]["exception_semantic_ids"][
        "retired"
    ] == list(RETIREMENT_HISTORY["exception_semantic_ids"])
    assert current["architecture_exception_baseline"][
        "cross_domain_site_semantic_ids"
    ]["retired"] == list(RETIREMENT_HISTORY["cross_domain_site_semantic_ids"])
    assert current["architecture_exception_baseline"][
        "production_to_lab_edge_semantic_ids"
    ]["retired"] == [
        P3_RETIRED_CLASSIFICATION_LAB_EDGE_ID,
        P4C_RETIRED_EXCEPTION_IDS[1],
        P4_RETIRED_SEARCH_LAB_EDGE_ID,
        P4A_RETIRED_OPS_LAB_EDGE_ID,
    ]
    assert current["architecture_exception_baseline"][
        "production_to_lab_static_site_semantic_ids"
    ]["retired"] == list(
        RETIREMENT_HISTORY["production_to_lab_static_site_semantic_ids"]
    )
    assert current["architecture_exception_baseline"][
        "production_to_lab_dynamic_site_semantic_ids"
    ]["retired"] == [P4A_RETIRED_OPS_LAB_DYNAMIC_SITE_ID]
    assert P2_RETIRED_PRIVATE_EXCEPTION_ID not in current[
        "architecture_exception_baseline"
    ]["exception_semantic_ids"]["active"]
    assert P3_RETIRED_CLASSIFICATION_LAB_EDGE_ID not in current[
        "architecture_exception_baseline"
    ]["exception_semantic_ids"]["active"]
    assert P3_RETIRED_PROVIDER_SITE_ID not in current[
        "architecture_exception_baseline"
    ]["cross_domain_site_semantic_ids"]["active"]
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
    built_ledger = architecture.build_architecture_exception_ledger(ROOT)
    built_seed = architecture.build_architecture_exception_baseline(ROOT)

    normalized_seed = _without_persisted_retirement_history(
        architecture,
        recorded_seed,
    )
    normalized_seed["cross_domain_site_semantic_ids"] = built_seed[
        "cross_domain_site_semantic_ids"
    ]
    normalized_seed["counts"]["active"]["cross_domain_sites"] = built_seed["counts"][
        "active"
    ]["cross_domain_sites"]
    assert built_seed == normalized_seed
    _assert_diagnostic_line_contract(recorded_ledger, built_ledger)
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


def test_diagnostic_line_contract_rejects_bool_and_path_drift() -> None:
    with pytest.raises(AssertionError):
        _assert_diagnostic_line_contract(
            {"items": [{"line": 1}]},
            {"items": [{"line": True}]},
        )
    with pytest.raises(AssertionError):
        _assert_diagnostic_line_contract(
            {"items": [{"line": 1}, {}]},
            {"items": [{}, {"line": 99}]},
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
