from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from scripts import runtime_access_314t_canary as canary

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/runtime_access_314t_canary.py"
WORKFLOW = ROOT / ".github/workflows/runtime-access-314t-canary.yml"
EXPECTED_ANALYZER_PATHS = (
    "scripts/runtime_ownership/access.py",
    "scripts/runtime_ownership/access_bindings.py",
    "scripts/runtime_ownership/access_class_scopes.py",
    "scripts/runtime_ownership/access_control.py",
    "scripts/runtime_ownership/access_definition_execution.py",
    "scripts/runtime_ownership/access_export_flow.py",
    "scripts/runtime_ownership/access_expressions.py",
    "scripts/runtime_ownership/access_facts.py",
    "scripts/runtime_ownership/access_imports.py",
    "scripts/runtime_ownership/access_model.py",
    "scripts/runtime_ownership/access_outcome_control.py",
    "scripts/runtime_ownership/access_outcomes.py",
    "scripts/runtime_ownership/access_resolver.py",
    "scripts/runtime_ownership/access_sinks.py",
    "scripts/runtime_ownership/access_statements.py",
)


def _run_contract_subprocess() -> subprocess.CompletedProcess[str]:
    code = f"""
import json
import sys
sys.path.insert(0, {str(ROOT)!r})
from scripts import runtime_access_314t_canary as canary
print(json.dumps(canary._run_contract_checks(), sort_keys=True))
"""
    return subprocess.run(
        [sys.executable, "-I", "-B", "-c", code],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )


def test_independent_fixture_and_seals_are_locked() -> None:
    from scripts.runtime_ownership import machine_facts, manifests

    assert canary.EXPECTED_ANALYZER_REVISION == (
        "418dd0d1dbb61857766b637087788b2ed9fe9c6c"
    )
    assert canary.EXPECTED_ANALYZER_FILES_SHA256 == (
        "b60940f9e1525db5259999668b8600c2026428c6c6bd7f181d3eb45d22eb25d7"
    )
    assert canary.EXPECTED_ANALYZER_MANIFEST_SHA256 == (
        "149077eb33bbbc26dc03b93fa5fd7cbe4c78188d41be9462a9d1c39dbd838603"
    )
    assert canary.EXPECTED_SOURCE_REVISION == (
        "f90202f1d1b9b2ed44075f38b0668c91fc0f196f"
    )
    assert canary.EXPECTED_SOURCE_FILES_SHA256 == (
        "be2ad06f687bc619a89d12ad6274d6843b26278e2094d420146105c398e73cee"
    )
    assert canary.EXPECTED_SOURCE_MANIFEST_SHA256 == (
        "268a6d8ca2fbd7d4877f78a3f5c6b14fd0e7e36d760173be9ce1a05e6703f43a"
    )
    assert tuple(manifests.ANALYZER_PATHS) == EXPECTED_ANALYZER_PATHS
    assert len(EXPECTED_ANALYZER_PATHS) == 15
    assert machine_facts.SHARD_PLAN_ID == "monolithic-v1"
    assert machine_facts.SHARDING_DISABLED_REASON == (
        "semantic_non_equivalence_risk"
    )

    candidates = [
        {
            "id": "runtime-resource:" + "1" * 64,
            "module": "chronovisor.a",
            "symbol": "RESOURCE",
            "locator": {
                "type": "path",
                "value": "$CHRONOVISOR_ROOT/demo",
            },
        }
    ]
    source_files = {
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
    assert list(canary.ANALYZER_CANDIDATES) == candidates
    assert source_files == canary.SOURCE_FILES
    adapter = canary._fixture_adapter(machine_facts.canonical_bytes)
    assert len(source_files) == 2
    assert len(candidates) == 1
    assert adapter["candidate_subset_sha256"] == hashlib.sha256(
        machine_facts.canonical_bytes(candidates)
    ).hexdigest()

    analyzer_manifest = manifests.build_manifest(
        ROOT,
        canary.EXPECTED_ANALYZER_REVISION,
        manifest_kind=manifests.ANALYZER_MANIFEST_KIND,
        expected_revision=canary.EXPECTED_ANALYZER_REVISION,
    )
    source_manifest = manifests.build_manifest(
        ROOT,
        canary.EXPECTED_SOURCE_REVISION,
        manifest_kind=manifests.SOURCE_MANIFEST_KIND,
        expected_revision=canary.EXPECTED_SOURCE_REVISION,
    )
    assert analyzer_manifest["files_sha256"] == (
        "b60940f9e1525db5259999668b8600c2026428c6c6bd7f181d3eb45d22eb25d7"
    )
    assert analyzer_manifest["manifest_sha256"] == (
        "149077eb33bbbc26dc03b93fa5fd7cbe4c78188d41be9462a9d1c39dbd838603"
    )
    assert source_manifest["files_sha256"] == (
        "be2ad06f687bc619a89d12ad6274d6843b26278e2094d420146105c398e73cee"
    )
    assert source_manifest["manifest_sha256"] == (
        "268a6d8ca2fbd7d4877f78a3f5c6b14fd0e7e36d760173be9ce1a05e6703f43a"
    )
    with pytest.raises(
        machine_facts.MachineFactError,
        match="semantic_non_equivalence_risk",
    ):
        machine_facts.reject_sharded_analysis()


def test_standard_gil_subprocess_locks_monolithic_contract() -> None:
    probe = _run_contract_subprocess()
    assert probe.returncode == 0, probe.stderr
    receipt: dict[str, Any] = json.loads(probe.stdout)
    assert receipt == {
        "canonical_byte_count": 1725,
        "canonical_sha256": (
            "6abc14507c31ebb76fb0ab3757bfdcba0a1cf37c26736972537625dabb6f1660"
        ),
        "escape_reason": "dynamic_open_mode",
        "escape_sink": "builtins.open",
        "shard_plan_id": "monolithic-v1",
        "status": "passed",
    }


def test_public_main_has_no_runtime_bypass_and_matches_current_runtime() -> None:
    assert list(__import__("inspect").signature(canary.main).parameters) == []
    probe = subprocess.run(
        [sys.executable, "-I", "-B", str(SCRIPT)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    gil_probe = getattr(sys, "_is_gil_enabled", None)
    py_gil_disabled = __import__("sysconfig").get_config_var("Py_GIL_DISABLED")
    is_exact_314t = (
        sys.implementation.name == "cpython"
        and sys.version_info[:2] == (3, 14)
        and callable(gil_probe)
        and gil_probe() is False
        and type(py_gil_disabled) is int
        and py_gil_disabled == 1
    )
    if is_exact_314t:
        assert probe.returncode == 0, probe.stderr
        receipt = json.loads(probe.stdout)
        assert receipt["canonical_byte_count"] == 1725
        assert receipt["canonical_sha256"] == (
            "6abc14507c31ebb76fb0ab3757bfdcba0a1cf37c26736972537625dabb6f1660"
        )
        assert receipt["status"] == "passed"
    else:
        assert probe.returncode == 1
        assert "runtime-access 3.14t canary failed:" in probe.stderr


def test_runtime_check_requires_exact_integer_py_gil_disabled() -> None:
    canary._assert_runtime_values(
        implementation_name="cpython",
        version=(3, 14),
        isolated=1,
        dont_write_bytecode=True,
        gil_enabled=False,
        py_gil_disabled=1,
    )
    with pytest.raises(canary.CanaryError):
        canary._assert_runtime_values(
            implementation_name="cpython",
            version=(3, 14),
            isolated=1,
            dont_write_bytecode=True,
            gil_enabled=False,
            py_gil_disabled=True,
        )


def test_script_imports_only_stdlib_and_local_modules() -> None:
    tree = ast.parse(SCRIPT.read_text())
    allowed_roots = {
        "__future__",
        "collections",
        "hashlib",
        "json",
        "pathlib",
        "scripts",
        "sys",
        "sysconfig",
        "typing",
    }
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_roots.add(node.module.split(".", 1)[0])
    assert imported_roots <= allowed_roots
    source = SCRIPT.read_text()
    assert "ProcessPoolExecutor" not in source
    assert "multiprocessing" not in source
    assert "run_sealed_effective_analysis" not in source


def test_workflow_is_an_isolated_direct_314t_canary() -> None:
    workflow = WORKFLOW.read_text()
    assert "timeout-minutes: 10" in workflow
    assert "fetch-depth: 0" in workflow
    assert "astral-sh/setup-uv@v6" in workflow
    assert 'python-version: "3.14t"' in workflow
    install = workflow.index("uv python install 3.14t")
    resolve = workflow.index(
        "uv python find 3.14t --no-project --no-python-downloads --resolve-links"
    )
    assert install < resolve
    assert (
        "uv python find 3.14t --no-project --no-python-downloads --resolve-links"
        in workflow
    )
    assert '"$PYTHON_314T" -I -B - <<\'PY\'' in workflow
    assert (
        '"$PYTHON_314T" -I -B scripts/runtime_access_314t_canary.py'
        in workflow
    )
    assert "uv sync" not in workflow
    assert "uv run" not in workflow
    assert "pytest" not in workflow
    assert "sys._is_gil_enabled() is False" in workflow
    assert "type(py_gil_disabled) is int and py_gil_disabled == 1" in workflow
