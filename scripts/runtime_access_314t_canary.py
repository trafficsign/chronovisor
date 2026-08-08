#!/usr/bin/env python3
"""Run the sealed runtime-access compatibility canary under CPython 3.14t."""

from __future__ import annotations

import hashlib
import json
import sys
import sysconfig
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

EXPECTED_ANALYZER_REVISION = "02ee06fb125f51b5aa7cc9acdc7ac56a36b7f541"
EXPECTED_ANALYZER_FILES_SHA256 = (
    "d910085ffd1796f19ac0b63fad3d18a93b59e0c1c9f9606539396034737185e0"
)
EXPECTED_ANALYZER_MANIFEST_SHA256 = (
    "84363513e70b4d9c394adc6ee753d29935344e8eaaebc8839ec37df196fbdbb3"
)
EXPECTED_SOURCE_REVISION = "f90202f1d1b9b2ed44075f38b0668c91fc0f196f"
EXPECTED_SOURCE_FILES_SHA256 = (
    "be2ad06f687bc619a89d12ad6274d6843b26278e2094d420146105c398e73cee"
)
EXPECTED_SOURCE_MANIFEST_SHA256 = (
    "268a6d8ca2fbd7d4877f78a3f5c6b14fd0e7e36d760173be9ce1a05e6703f43a"
)
EXPECTED_ANALYZER_MODULE_COUNT = 15
EXPECTED_RESULT_BYTE_COUNT = 1725
EXPECTED_RESULT_SHA256 = (
    "6abc14507c31ebb76fb0ab3757bfdcba0a1cf37c26736972537625dabb6f1660"
)
EXPECTED_SHARD_PLAN_ID = "monolithic-v1"
EXPECTED_SHARDING_DISABLED_REASON = "semantic_non_equivalence_risk"

RESOURCE_ID = "runtime-resource:" + "1" * 64
ANALYZER_CANDIDATES: tuple[dict[str, Any], ...] = (
    {
        "id": RESOURCE_ID,
        "module": "chronovisor.a",
        "symbol": "RESOURCE",
        "locator": {"type": "path", "value": "$CHRONOVISOR_ROOT/demo"},
    },
)
SOURCE_FILES: Mapping[str, bytes] = {
    "src/chronovisor/a.py": (
        b'RESOURCE = "ignored"\n'
        b"def mode():\n"
        b'    return "r"\n'
    ),
    "src/chronovisor/b.py": (
        b"from chronovisor.a import RESOURCE, mode\n"
        b"def run():\n"
        b"    return open(RESOURCE, mode())\n"
    ),
}


class CanaryError(RuntimeError):
    """Raised when the runtime or compatibility contract drifts."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CanaryError(message)


def _assert_runtime_values(
    *,
    implementation_name: str,
    version: tuple[int, int],
    isolated: object,
    dont_write_bytecode: object,
    gil_enabled: object,
    py_gil_disabled: object,
) -> None:
    _require(implementation_name == "cpython", "CPython is required")
    _require(version == (3, 14), "exactly CPython 3.14 is required")
    _require(type(isolated) is int and isolated == 1, "the canary must run with -I")
    _require(dont_write_bytecode is True, "the canary must run with -B")
    _require(gil_enabled is False, "the GIL must be disabled")
    _require(
        type(py_gil_disabled) is int and py_gil_disabled == 1,
        "Py_GIL_DISABLED must be exact integer 1",
    )


def _assert_free_threaded_runtime() -> None:
    gil_probe = getattr(sys, "_is_gil_enabled", None)
    if not callable(gil_probe):
        raise CanaryError("sys._is_gil_enabled is required")
    _assert_runtime_values(
        implementation_name=sys.implementation.name,
        version=(sys.version_info.major, sys.version_info.minor),
        isolated=sys.flags.isolated,
        dont_write_bytecode=sys.dont_write_bytecode,
        gil_enabled=gil_probe(),
        py_gil_disabled=sysconfig.get_config_var("Py_GIL_DISABLED"),
    )


def _assert_manifest_seal(
    manifest: Mapping[str, Any],
    *,
    revision: str,
    files_sha256: str,
    manifest_sha256: str,
) -> None:
    _require(manifest.get("revision") == revision, "manifest revision drifted")
    _require(
        manifest.get("files_sha256") == files_sha256,
        "manifest files seal drifted",
    )
    _require(
        manifest.get("manifest_sha256") == manifest_sha256,
        "manifest content seal drifted",
    )


def _fixture_adapter(canonical_bytes: Any) -> dict[str, Any]:
    candidates = [dict(ANALYZER_CANDIDATES[0])]
    candidate_hash = hashlib.sha256(canonical_bytes(candidates)).hexdigest()
    return {
        "adapter_schema_version": 1,
        "analyzer_candidates": candidates,
        "unsupported_declarations": [],
        "excluded_declarations": [],
        "supported_resource_ids": [RESOURCE_ID],
        "candidate_subset_sha256": candidate_hash,
        "counts": {
            "resource_candidates": 1,
            "supported_candidates": 1,
            "supported_resources": 1,
            "unsupported_declarations": 0,
            "excluded_declarations": 0,
        },
    }


def _run_contract_checks() -> dict[str, Any]:
    """Exercise the compatibility contract; tests call this in a fresh process."""

    from scripts.runtime_ownership import machine_facts, manifests

    expected_module_names = frozenset(
        f"scripts.runtime_ownership.{Path(path).stem}"
        for path in manifests.ANALYZER_PATHS
    )
    _require(
        len(manifests.ANALYZER_PATHS) == EXPECTED_ANALYZER_MODULE_COUNT,
        "analyzer path count drifted",
    )
    _require(
        not expected_module_names.intersection(sys.modules),
        "analyzer modules were imported before seal verification",
    )
    constants = {
        "analyzer_revision": (
            machine_facts.ANALYZER_REVISION,
            EXPECTED_ANALYZER_REVISION,
        ),
        "analyzer_files_sha256": (
            machine_facts.ANALYZER_FILES_SHA256,
            EXPECTED_ANALYZER_FILES_SHA256,
        ),
        "analyzer_manifest_sha256": (
            machine_facts.ANALYZER_MANIFEST_SHA256,
            EXPECTED_ANALYZER_MANIFEST_SHA256,
        ),
        "source_revision": (
            machine_facts.EFFECTIVE_SOURCE_REVISION,
            EXPECTED_SOURCE_REVISION,
        ),
        "source_files_sha256": (
            machine_facts.SOURCE_FILES_SHA256,
            EXPECTED_SOURCE_FILES_SHA256,
        ),
        "source_manifest_sha256": (
            machine_facts.SOURCE_MANIFEST_SHA256,
            EXPECTED_SOURCE_MANIFEST_SHA256,
        ),
        "shard_plan_id": (
            machine_facts.SHARD_PLAN_ID,
            EXPECTED_SHARD_PLAN_ID,
        ),
        "sharding_disabled_reason": (
            machine_facts.SHARDING_DISABLED_REASON,
            EXPECTED_SHARDING_DISABLED_REASON,
        ),
    }
    for label, (actual, expected) in constants.items():
        _require(actual == expected, f"{label} drifted")

    analyzer_manifest = manifests.build_manifest(
        REPOSITORY,
        EXPECTED_ANALYZER_REVISION,
        manifest_kind=manifests.ANALYZER_MANIFEST_KIND,
        expected_revision=EXPECTED_ANALYZER_REVISION,
    )
    _assert_manifest_seal(
        analyzer_manifest,
        revision=EXPECTED_ANALYZER_REVISION,
        files_sha256=EXPECTED_ANALYZER_FILES_SHA256,
        manifest_sha256=EXPECTED_ANALYZER_MANIFEST_SHA256,
    )
    analyzer_snapshot = manifests.verify_manifest(
        REPOSITORY,
        analyzer_manifest,
        expected_kind=manifests.ANALYZER_MANIFEST_KIND,
        expected_revision=EXPECTED_ANALYZER_REVISION,
    )
    _require(
        tuple(row.path for row in analyzer_snapshot.files)
        == tuple(manifests.ANALYZER_PATHS),
        "verified analyzer paths drifted",
    )

    source_manifest = manifests.build_manifest(
        REPOSITORY,
        EXPECTED_SOURCE_REVISION,
        manifest_kind=manifests.SOURCE_MANIFEST_KIND,
        expected_revision=EXPECTED_SOURCE_REVISION,
    )
    _assert_manifest_seal(
        source_manifest,
        revision=EXPECTED_SOURCE_REVISION,
        files_sha256=EXPECTED_SOURCE_FILES_SHA256,
        manifest_sha256=EXPECTED_SOURCE_MANIFEST_SHA256,
    )
    manifests.verify_manifest(
        REPOSITORY,
        source_manifest,
        expected_kind=manifests.SOURCE_MANIFEST_KIND,
        expected_revision=EXPECTED_SOURCE_REVISION,
    )

    machine_facts.verify_loaded_analyzer_code(
        machine_facts.current_loaded_analyzer_files(REPOSITORY),
        analyzer_snapshot,
    )
    analyzer = machine_facts._import_verified_fresh_analyzer(
        REPOSITORY, analyzer_snapshot
    )
    loaded_names = expected_module_names.intersection(sys.modules)
    _require(
        loaded_names == expected_module_names,
        "fresh import did not bind exactly 15 analyzer modules",
    )
    for path in manifests.ANALYZER_PATHS:
        name = f"scripts.runtime_ownership.{Path(path).stem}"
        loaded_path = getattr(sys.modules[name], "__file__", None)
        _require(
            type(loaded_path) is str
            and Path(loaded_path).resolve() == (REPOSITORY / path).resolve(),
            f"analyzer module path drifted: {name}",
        )

    adapter = _fixture_adapter(machine_facts.canonical_bytes)
    result = machine_facts.analyze_runtime_access_unsealed(
        SOURCE_FILES,
        adapter,
        analyzer=analyzer,
    )
    raw = machine_facts.canonical_bytes(result)
    digest = hashlib.sha256(raw).hexdigest()
    _require(len(raw) == EXPECTED_RESULT_BYTE_COUNT, "canonical byte count drifted")
    _require(digest == EXPECTED_RESULT_SHA256, "canonical result digest drifted")
    counts = result.get("counts")
    _require(type(counts) is dict, "runtime-access counts are missing")
    count_values = cast(dict[str, Any], counts)
    _require(count_values.get("accesses") == 0, "fixture access count drifted")
    _require(count_values.get("escapes") == 1, "fixture escape count drifted")
    escape_facts = result.get("escape_facts")
    _require(
        type(escape_facts) is list and len(escape_facts) == 1,
        "fixture escape facts drifted",
    )
    escape_rows = cast(list[Any], escape_facts)
    escape = escape_rows[0]
    _require(type(escape) is dict, "fixture escape fact is malformed")
    escape_row = cast(dict[str, Any], escape)
    _require(
        escape_row.get("reason") == "dynamic_open_mode",
        "fixture escape reason drifted",
    )
    _require(escape_row.get("sink") == "builtins.open", "fixture sink drifted")

    try:
        machine_facts.reject_sharded_analysis()
    except machine_facts.MachineFactError as exc:
        _require(
            EXPECTED_SHARDING_DISABLED_REASON in str(exc),
            "sharding rejection reason drifted",
        )
    else:
        raise CanaryError("sharded analysis was not rejected")

    return {
        "canonical_byte_count": len(raw),
        "canonical_sha256": digest,
        "escape_reason": escape_row["reason"],
        "escape_sink": escape_row["sink"],
        "shard_plan_id": machine_facts.SHARD_PLAN_ID,
        "status": "passed",
    }


def main() -> int:
    """Fail closed unless running the isolated CPython 3.14t lane."""

    _assert_free_threaded_runtime()
    receipt = _run_contract_checks()
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CanaryError as exc:
        print(f"runtime-access 3.14t canary failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
