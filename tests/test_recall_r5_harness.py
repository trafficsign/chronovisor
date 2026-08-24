from __future__ import annotations

import builtins
import contextlib
import importlib
import importlib.util
import json
import os
import socket
import sqlite3
import subprocess
import sys
import time
import types
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "recall_r5_harness_test", ROOT / "scripts" / "recall_r5_harness.py"
)
assert SPEC and SPEC.loader
HARNESS = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = HARNESS
SPEC.loader.exec_module(HARNESS)


@pytest.fixture(autouse=True)
def _restore_chronovisor_modules() -> Any:
    snapshot = HARNESS._chronovisor_module_snapshot()
    try:
        yield
    finally:
        HARNESS._restore_chronovisor_modules(snapshot)


def _id(number: int) -> str:
    return f"{number:064x}"


def _base() -> tuple[Any, Any, Any, Any, Any]:
    labels = [
        {
            "record_sha256": _id(1),
            "status": "completed",
            "profile": "p",
            "cohort": "c",
            "assignment_revision": "a",
            "label_set_revision": "l",
        }
    ]
    rows = [
        {
            "label_record_sha256": _id(1),
            "source": "teacher-label",
            "probe": False,
            "verdict": "relevant",
            "feature_parity": True,
            "future_leakage": False,
        }
    ]
    preflight = {
        "counts": {"eligible_native_rallies": 1000, "span_days": 30, "windows": 3},
        "hard_floor": {"p5_allowed": True},
        "label_chain_head": _id(2),
        "baseline": {"label_chain_head": _id(2), "training_snapshot_sha256": _id(3)},
    }
    gate = {"passed": True}
    workset = {"counts": {"leased": 0}, "completed_refs": []}
    return rows, labels, preflight, gate, workset


def _result() -> dict[str, object]:
    rows, labels, preflight, gate, workset = _base()
    return cast(
        dict[str, object],
        HARNESS.validate_dataset(
            rows=rows, labels=labels, preflight=preflight, gate=gate, workset=workset
        ),
    )


def test_contract_constants_and_root_boundary(tmp_path: Path) -> None:
    assert HARNESS.R5_SCHEMA == "chronovisor.recall-r5.v1"
    production, source, output = (
        tmp_path / "production",
        tmp_path / "source",
        tmp_path / "output",
    )
    production.mkdir()
    source.mkdir()
    HARNESS.assert_root_matrix(production, source, output)
    with pytest.raises(HARNESS.R5Error, match="overlap"):
        HARNESS.assert_root_matrix(production, production, output)
    link = tmp_path / "link"
    link.symlink_to(source, target_is_directory=True)
    with pytest.raises(HARNESS.R5Error, match="symlink"):
        HARNESS.assert_root_matrix(production, link, output)


def test_source_state_rejects_non_git_source(tmp_path: Path) -> None:
    source = tmp_path / "not-git"
    source.mkdir()
    (source / "marker").write_text("not a checkout")
    with pytest.raises(HARNESS.R5Error, match="trusted command failed"):
        HARNESS.source_state(source, "a" * 40)


def test_direct_run_rejects_python_only_kernel_attestation() -> None:
    with pytest.raises(HARNESS.R5Error, match="kernel sandbox attestation"):
        HARNESS.run(
            production=Path("/missing-production"),
            source=Path("/missing-source"),
            source_commit="a" * 40,
            output=Path("/missing-output"),
            r4_artifact=Path("/missing-r4"),
            kernel_attested=True,
        )


def test_direct_inner_run_rejects_forged_environment_attestation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("R5_KERNEL_SANDBOX_ATTESTATION", "forged")
    assert HARNESS.main(
        [
            "--inner-run", "--kernel-sandbox-attestation", "forged",
            "--source-root", "/missing-source", "--source-commit", "a" * 40,
            "--output", "/missing-output", "--r4-artifact", "/missing-r4",
        ]
    ) == 2


def test_missing_native_metric_and_fake_flag_decline() -> None:
    rows, labels, preflight, gate, workset = _base()
    preflight["counts"] = {
        "answer_utility_eligible": 9999,
        "span_days": 30,
        "windows": 3,
    }
    rows[0]["eligible"] = True
    result = HARNESS.validate_dataset(
        rows=rows, labels=labels, preflight=preflight, gate=gate, workset=workset
    )
    assert result["passed"] is False
    assert "eligible_native_predicate_missing_or_empty" in result["reasons"]


def test_crafted_floor_counts_cannot_pass_without_exact_label_age_and_workset_bindings() -> (
    None
):
    rows, labels, preflight, gate, workset = _base()
    rows[:] = [
        {
            "label_id": _id(index + 10),
            "source": "teacher-label",
            "probe": False,
            "verdict": "relevant" if index < 250 else "irrelevant",
            "feature_parity": True,
            "future_leakage": False,
        }
        for index in range(500)
    ]
    # Producer-supplied counts and rows do not replace actual native rallies,
    # ledger identities, age observations, or completed-work bijection.
    preflight["counts"] = {
        "eligible_native_rallies": 1000,
        "span_days": 30,
        "windows": 3,
    }
    result = HARNESS.validate_dataset(
        rows=rows, labels=labels, preflight=preflight, gate=gate, workset=workset
    )
    assert result["passed"] is False
    assert "materialized_label_ledger_not_exact_1_to_1" in result["reasons"]
    assert "eligible_native_predicate_missing_or_empty" in result["reasons"]


def test_caller_sealed_r5_metadata_cannot_replace_official_rederivation() -> None:
    """A self-consistent producer envelope is not canonical evidence."""
    rows, labels, preflight, gate, workset = _base()
    preflight["official_r5_evidence"] = {
        "materialization_artifact_id": _id(3),
        "label_chain_head": _id(2),
        "rows_sha256": HARNESS._sha(rows),
        "gate_sha256": HARNESS._sha(gate),
    }
    result = HARNESS.validate_dataset(
        rows=rows, labels=labels, preflight=preflight, gate=gate, workset=workset
    )
    assert result["passed"] is False
    assert "independent_official_evidence_rederivation_failed" in result["reasons"]


@pytest.mark.parametrize(
    "mutate, reason",
    [
        (
            lambda rows, labels, preflight, gate, workset: labels.append(
                dict(labels[0])
            ),
            "label_identity_duplicate_or_missing",
        ),
        (
            lambda rows, labels, preflight, gate, workset: labels[0].update(
                status="retry"
            ),
            "invalid_or_uncertain_label_status",
        ),
        (
            lambda rows, labels, preflight, gate, workset: workset.update(
                completed_refs=["missing"]
            ),
            "workset_label_reconciliation_failed",
        ),
        (
            lambda rows, labels, preflight, gate, workset: preflight.update(
                baseline={"label_chain_head": _id(9)}
            ),
            "baseline_label_head_or_snapshot_binding_missing",
        ),
    ],
)
def test_adversarial_metadata_declines(
    mutate: Callable[[Any, Any, Any, Any, Any], None], reason: str
) -> None:
    rows, labels, preflight, gate, workset = _base()
    mutate(rows, labels, preflight, gate, workset)
    result = HARNESS.validate_dataset(
        rows=rows, labels=labels, preflight=preflight, gate=gate, workset=workset
    )
    assert result["passed"] is False
    assert reason in result["reasons"]


def test_counterfactual_duplicate_and_missing_seal_decline() -> None:
    rows, labels, preflight, gate, workset = _base()
    for _index in range(100):
        rows.append(
            {
                "source": "counterfactual-label",
                "verdict": "helpful",
                "counterfactual_ref": _id(20),
                "feature_parity": True,
                "future_leakage": False,
            }
        )
    result = HARNESS.validate_dataset(
        rows=rows, labels=labels, preflight=preflight, gate=gate, workset=workset
    )
    assert "sealed_counterfactual_pairs_below_floor_or_duplicate" in result["reasons"]
    assert "counterfactual_sealed_exposure_binding_missing" in result["reasons"]


def test_artifact_tamper_is_detected_and_payload_is_absent(tmp_path: Path) -> None:
    payload = {
        "captured_at": "2026-08-24T00:00:00Z",
        "source": {},
        "source_after": {},
        "production": {},
        "production_after": {},
        "clone": {},
        "r4_dependency": {},
        "dataset": {"passed": False},
        "phases": [],
        "cleanup": {"clone_removed": True, "remaining": 0},
        "provider_calls": 0,
        "egress_attempts": 0,
        "process_attempts": 0,
        "supervised": False,
        "test_only": False,
    }
    artifact = HARNESS._sealed_artifact(payload)
    path = HARNESS._write_immutable(tmp_path, artifact)
    assert HARNESS.read_artifact(path)["artifact_id"] == artifact["artifact_id"]
    assert b"payload" not in path.read_bytes().lower()
    altered = json.loads(path.read_text())
    altered["egress"] = 1
    path.write_text(json.dumps(altered, sort_keys=True))
    with pytest.raises(HARNESS.R5Error, match="closed"):
        HARNESS.read_artifact(path)


def test_truthful_floor_decline_retains_valid_r4_dependency_but_cannot_formally_pass(
    tmp_path: Path,
) -> None:
    inner = _formal_inner()
    decline = _reseal_inner(
        inner,
        dataset={
            "passed": False,
            "capture_only": True,
                "reasons": ["valid_nonprobe_labels_below_floor"],
            "metrics": cast(dict[str, object], inner["dataset"])["metrics"],
        },
    )
    path = HARNESS._write_immutable(tmp_path, decline)
    assert HARNESS.read_artifact(path)["r4_dependency"] == inner["r4_dependency"]
    with pytest.raises(HARNESS.R5Error, match="cannot certify"):
        HARNESS.assert_formal_acceptance(
            decline, {}, source_root=tmp_path, r4_artifact=tmp_path / "r4.json",
        )


@pytest.mark.parametrize("field", ["source", "production", "clone"])
def test_artifact_identity_mappings_reject_safe_unknown_fields(field: str) -> None:
    inner = _formal_inner()
    payload = {
        key: value
        for key, value in inner.items()
        if key not in {"artifact_id", "schema", "namespace", "seal_sha256"}
    }
    value = dict(cast(dict[str, object], payload[field]))
    value["benign_extra"] = True
    payload[field] = value
    with pytest.raises(HARNESS.R5Error, match="closed"):
        HARNESS._sealed_artifact(payload)


@pytest.mark.parametrize("target", ["protected", "files", "phase", "metrics"])
def test_artifact_deep_nested_values_reject_non_primitives(target: str) -> None:
    inner = _formal_inner()
    payload = {
        key: value
        for key, value in inner.items()
        if key not in {"artifact_id", "schema", "namespace", "seal_sha256"}
    }
    if target == "protected":
        protected = cast(dict[str, object], cast(dict[str, object], payload["production"])["protected_inventory"])
        cast(list[object], protected["included"]).append({"not": "primitive"})
    elif target == "files":
        cast(dict[str, object], cast(dict[str, object], payload["clone"])["state"])["files"] = {"not": "digest"}
    elif target == "phase":
        cast(list[dict[str, object]], payload["phases"]).append({"name": "bad", "elapsed_ms": {"not": "number"}})
    else:
        cast(dict[str, object], cast(dict[str, object], payload["dataset"])["metrics"])["rows"] = {"not": "primitive"}
    with pytest.raises(HARNESS.R5Error, match="artifact"):
        HARNESS._sealed_artifact(payload)


@pytest.mark.parametrize("target", ["captured", "phase", "cleanup", "dependency", "test_only"])
def test_artifact_receipt_boundary_rejects_malformed_control_values(target: str) -> None:
    inner = _formal_inner()
    payload = {
        key: value
        for key, value in inner.items()
        if key not in {"artifact_id", "schema", "namespace", "seal_sha256"}
    }
    if target == "captured":
        payload["captured_at"] = {"not": "timestamp"}
    elif target == "phase":
        payload["phases"] = [{"name": "ok", "elapsed_ms": 1, "extra": True}]
    elif target == "cleanup":
        payload["cleanup"] = {"clone_removed": 1, "remaining": -1}
    elif target == "dependency":
        payload["r4_dependency"] = {"artifact_id": "bad", "seal_sha256": _id(1)}
    else:
        payload["test_only"] = "false"
    with pytest.raises(HARNESS.R5Error, match="artifact"):
        HARNESS._sealed_artifact(payload)


def test_evidence_inventory_rejects_late_workset_sidecar(tmp_path: Path) -> None:
    base = tmp_path / "runtime" / "recall-distillation"
    base.mkdir(parents=True)
    inventory = HARNESS._evidence_inventory(tmp_path)
    sidecar = base / "ox-workset.sqlite3-wal"
    sidecar.write_bytes(b"late")
    relative = "runtime/recall-distillation/ox-workset.sqlite3-wal"
    assert HARNESS._inventory_matches(tmp_path, relative, inventory[relative]) is False


def test_load_runtime_rejects_preloaded_outside_transitive_dependency(tmp_path: Path) -> None:
    source = tmp_path / "source"
    recall = source / "src" / "chronovisor" / "recall"
    recall.mkdir(parents=True)
    (recall.parent / "__init__.py").write_text("")
    (recall / "__init__.py").write_text("")
    (recall / "recall_distillation.py").write_text("import fake_dependency\n")
    (recall / "recall_distillation_store.py").write_text("")
    (recall / "recall_distillation_workset.py").write_text("")
    fake = types.ModuleType("fake_dependency")
    fake.__file__ = str(tmp_path / "outside" / "fake_dependency.py")
    prior_chronovisor = {
        name: module
        for name, module in sys.modules.items()
        if name == "chronovisor" or name.startswith("chronovisor.")
    }
    prior_fake = sys.modules.get("fake_dependency")
    for name in prior_chronovisor:
        sys.modules.pop(name, None)
    sys.modules["fake_dependency"] = fake
    try:
        with pytest.raises(HARNESS.R5Error, match="allowed origins"):
            HARNESS._load_runtime(source)
    finally:
        for name in [name for name in sys.modules if name == "chronovisor" or name.startswith("chronovisor.")]:
            sys.modules.pop(name, None)
        sys.modules.update(prior_chronovisor)
        if prior_fake is None:
            sys.modules.pop("fake_dependency", None)
        else:
            sys.modules["fake_dependency"] = prior_fake
        with contextlib.suppress(ValueError):
            sys.path.remove(str(source / "src"))


def test_module_provenance_rejects_spoofed_source_path_module(tmp_path: Path) -> None:
    source = tmp_path / "source"
    module_path = source / "src" / "fake_dependency.py"
    module_path.parent.mkdir(parents=True)
    module_path.write_text("value = 1\n")
    fake = types.ModuleType("fake_dependency")
    fake.__file__ = str(module_path)
    assert HARNESS._module_provenance_allowed(fake, source) is False


def test_module_provenance_accepts_genuine_source_stdlib_and_environment(tmp_path: Path) -> None:
    source = tmp_path / "source"
    recall = source / "src" / "chronovisor" / "recall"
    recall.mkdir(parents=True)
    (recall.parent / "__init__.py").write_text("")
    (recall / "__init__.py").write_text("")
    (recall / "recall_distillation.py").write_text("")
    (recall / "recall_distillation_store.py").write_text("")
    (recall / "recall_distillation_workset.py").write_text("")
    prior = {
        name: module
        for name, module in sys.modules.items()
        if name == "chronovisor" or name.startswith("chronovisor.")
    }
    for name in prior:
        sys.modules.pop(name, None)
    try:
        distill, _, _ = HARNESS._load_runtime(source)
        assert HARNESS._module_provenance_allowed(distill, source)
        assert HARNESS._module_provenance_allowed(json, source)
        assert HARNESS._module_provenance_allowed(pytest, source)
    finally:
        for name in [name for name in sys.modules if name == "chronovisor" or name.startswith("chronovisor.")]:
            sys.modules.pop(name, None)
        sys.modules.update(prior)
        with contextlib.suppress(ValueError):
            sys.path.remove(str(source / "src"))


def test_load_runtime_evicts_forged_preloaded_module_with_real_loader(tmp_path: Path) -> None:
    source = tmp_path / "source"
    recall = source / "src" / "chronovisor" / "recall"
    recall.mkdir(parents=True)
    (recall.parent / "__init__.py").write_text("")
    (recall / "__init__.py").write_text("")
    dependency = source / "src" / "fake_dependency.py"
    dependency.write_text("value = 'genuine'\n")
    (recall / "recall_distillation.py").write_text("import fake_dependency\n")
    (recall / "recall_distillation_store.py").write_text("")
    (recall / "recall_distillation_workset.py").write_text("")
    fake = types.ModuleType("fake_dependency")
    fake.__file__ = str(dependency)
    fake.__spec__ = importlib.util.spec_from_file_location("fake_dependency", dependency)
    fake.__dict__["value"] = "forged"
    prior_fake = sys.modules.get("fake_dependency")
    prior_chronovisor = {
        name: module for name, module in sys.modules.items()
        if name == "chronovisor" or name.startswith("chronovisor.")
    }
    sys.modules["fake_dependency"] = fake
    for name in prior_chronovisor:
        sys.modules.pop(name, None)
    try:
        HARNESS._load_runtime(source)
        assert sys.modules["fake_dependency"] is not fake
        assert sys.modules["fake_dependency"].__dict__["value"] == "genuine"
    finally:
        for name in [name for name in sys.modules if name == "chronovisor" or name.startswith("chronovisor.")]:
            sys.modules.pop(name, None)
        sys.modules.update(prior_chronovisor)
        if prior_fake is None:
            sys.modules.pop("fake_dependency", None)
        else:
            sys.modules["fake_dependency"] = prior_fake
        with contextlib.suppress(ValueError):
            sys.path.remove(str(source / "src"))


def test_load_runtime_reloads_source_injected_stdlib_lookalike(tmp_path: Path) -> None:
    source = tmp_path / "source"
    recall = source / "src" / "chronovisor" / "recall"
    recall.mkdir(parents=True)
    (source / "src" / "chronovisor" / "__init__.py").write_text("")
    (source / "src" / "chronovisor" / "recall" / "__init__.py").write_text("")
    (recall / "recall_distillation.py").write_text(
        "import importlib.util, json as genuine, sys, types\n"
        "fake = types.ModuleType('json')\n"
        "fake.__file__ = genuine.__file__\n"
        "fake.__spec__ = importlib.util.spec_from_file_location('json', genuine.__file__)\n"
        "fake.forged = True\n"
        "sys.modules['json'] = fake\n"
        "dependency = __import__('json')\n"
    )
    (recall / "recall_distillation_store.py").write_text("")
    (recall / "recall_distillation_workset.py").write_text("")
    prior = {name: sys.modules.get(name) for name in ("chronovisor", "json")}
    try:
        distill, _, _ = HARNESS._load_runtime(source)
        assert distill.dependency is sys.modules["json"]
        assert getattr(distill.dependency, "forged", False) is False
        assert HARNESS._module_provenance_allowed(distill.dependency, source)
    finally:
        for name in [name for name in sys.modules if name == "chronovisor" or name.startswith("chronovisor.")]:
            sys.modules.pop(name, None)
        for name, module in prior.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
        with contextlib.suppress(ValueError):
            sys.path.remove(str(source / "src"))


def test_load_runtime_evicts_dynamic_import_injected_module(tmp_path: Path) -> None:
    source = tmp_path / "source"
    recall = source / "src" / "chronovisor" / "recall"
    recall.mkdir(parents=True)
    (recall.parent / "__init__.py").write_text("")
    (recall / "__init__.py").write_text("")
    dependency = source / "src" / "fake_dependency.py"
    dependency.write_text("value = 'genuine'\n")
    (recall / "recall_distillation.py").write_text("dependency = __import__('fake_dependency')\n")
    (recall / "recall_distillation_store.py").write_text("")
    (recall / "recall_distillation_workset.py").write_text("")
    fake = types.ModuleType("fake_dependency")
    fake.__file__ = str(dependency)
    fake.__spec__ = importlib.util.spec_from_file_location("fake_dependency", dependency)
    fake.__dict__["value"] = "forged"
    previous = sys.modules.get("fake_dependency")
    prior_chronovisor = {
        name: module for name, module in sys.modules.items()
        if name == "chronovisor" or name.startswith("chronovisor.")
    }
    sys.modules["fake_dependency"] = fake
    for name in prior_chronovisor:
        sys.modules.pop(name, None)
    try:
        distill, _, _ = HARNESS._load_runtime(source)
        assert distill.dependency is not fake
        assert distill.dependency.__dict__["value"] == "genuine"
    finally:
        for name in [name for name in sys.modules if name == "chronovisor" or name.startswith("chronovisor.")]:
            sys.modules.pop(name, None)
        sys.modules.update(prior_chronovisor)
        if previous is None:
            sys.modules.pop("fake_dependency", None)
        else:
            sys.modules["fake_dependency"] = previous
        with contextlib.suppress(ValueError):
            sys.path.remove(str(source / "src"))


def test_load_runtime_evicts_dynamic_module_forging_builtin_origin(tmp_path: Path) -> None:
    source = tmp_path / "source"
    recall = source / "src" / "chronovisor" / "recall"
    recall.mkdir(parents=True)
    (recall.parent / "__init__.py").write_text("")
    (recall / "__init__.py").write_text("")
    (recall / "recall_distillation.py").write_text("dependency = __import__('time')\n")
    (recall / "recall_distillation_store.py").write_text("")
    (recall / "recall_distillation_workset.py").write_text("")
    fake = types.ModuleType("time")
    fake.__file__ = None
    fake.__spec__ = importlib.machinery.ModuleSpec(
        "time", cast(Any, importlib.machinery.BuiltinImporter), origin="built-in"
    )
    fake.__dict__["forged"] = True
    prior_time = sys.modules.get("time")
    prior_chronovisor = {
        name: module for name, module in sys.modules.items()
        if name == "chronovisor" or name.startswith("chronovisor.")
    }
    sys.modules["time"] = fake
    for name in prior_chronovisor:
        sys.modules.pop(name, None)
    try:
        distill, _, _ = HARNESS._load_runtime(source)
        assert distill.dependency is not fake
        assert getattr(distill.dependency, "forged", False) is False
    finally:
        for name in [name for name in sys.modules if name == "chronovisor" or name.startswith("chronovisor.")]:
            sys.modules.pop(name, None)
        sys.modules.update(prior_chronovisor)
        if prior_time is None:
            sys.modules.pop("time", None)
        else:
            sys.modules["time"] = prior_time
        with contextlib.suppress(ValueError):
            sys.path.remove(str(source / "src"))


def test_load_runtime_preserves_pinned_bootstrap_during_stdlib_import(tmp_path: Path) -> None:
    source = tmp_path / "source"
    recall = source / "src" / "chronovisor" / "recall"
    recall.mkdir(parents=True)
    (recall.parent / "__init__.py").write_text("")
    (recall / "__init__.py").write_text("")
    (recall / "recall_distillation.py").write_text("import json\ndependency = json\n")
    (recall / "recall_distillation_store.py").write_text("")
    (recall / "recall_distillation_workset.py").write_text("")
    prior_chronovisor = {
        name: module for name, module in sys.modules.items()
        if name == "chronovisor" or name.startswith("chronovisor.")
    }
    for name in prior_chronovisor:
        sys.modules.pop(name, None)
    try:
        distill, _, _ = HARNESS._load_runtime(source)
        assert distill.dependency.dumps({"ok": True}) == '{"ok": true}'
        assert sys.modules["sys"] is HARNESS._TRUSTED_BOOTSTRAP_MODULES["sys"]
    finally:
        for name in [name for name in sys.modules if name == "chronovisor" or name.startswith("chronovisor.")]:
            sys.modules.pop(name, None)
        sys.modules.update(prior_chronovisor)
        with contextlib.suppress(ValueError):
            sys.path.remove(str(source / "src"))


def test_load_runtime_rejects_dynamic_module_forging_frozen_origin(tmp_path: Path) -> None:
    source = tmp_path / "source"
    recall = source / "src" / "chronovisor" / "recall"
    recall.mkdir(parents=True)
    (recall.parent / "__init__.py").write_text("")
    (recall / "__init__.py").write_text("")
    (recall / "recall_distillation.py").write_text("dependency = __import__('frozen_fake')\n")
    (recall / "recall_distillation_store.py").write_text("")
    (recall / "recall_distillation_workset.py").write_text("")
    fake = types.ModuleType("frozen_fake")
    fake.__file__ = None
    fake.__spec__ = importlib.machinery.ModuleSpec(
        "frozen_fake", cast(Any, importlib.machinery.FrozenImporter), origin="frozen"
    )
    assert HARNESS._module_provenance_allowed(fake, source) is False
    prior = sys.modules.get("frozen_fake")
    prior_chronovisor = {
        name: module for name, module in sys.modules.items()
        if name == "chronovisor" or name.startswith("chronovisor.")
    }
    sys.modules["frozen_fake"] = fake
    for name in prior_chronovisor:
        sys.modules.pop(name, None)
    try:
        with pytest.raises(HARNESS.R5Error, match="allowed origins"):
            HARNESS._load_runtime(source)
    finally:
        for name in [name for name in sys.modules if name == "chronovisor" or name.startswith("chronovisor.")]:
            sys.modules.pop(name, None)
        sys.modules.update(prior_chronovisor)
        if prior is None:
            sys.modules.pop("frozen_fake", None)
        else:
            sys.modules["frozen_fake"] = prior
        with contextlib.suppress(ValueError):
            sys.path.remove(str(source / "src"))


def test_provider_sentinel_fails_closed_on_depth_and_slots() -> None:
    root: Any = types.SimpleNamespace()
    current = root
    for _ in range(520):
        child = types.SimpleNamespace()
        current.adapter = child
        current = child
    attempts = {"provider_calls": 0, "egress_attempts": 0, "process_attempts": 0}
    with pytest.raises(HARNESS.R5Error, match="bounded inspection"):
        with HARNESS._provider_sentinel(attempts, root):
            pass

    class Adapter:
        def evaluate(self, _payload: object) -> object:
            return {}

    class Holder:
        __slots__ = ("adapter",)

        def __init__(self) -> None:
            self.adapter = Adapter()

    holder = Holder()
    with HARNESS._provider_sentinel(attempts, types.SimpleNamespace(holder=holder)):
        with pytest.raises(HARNESS.R5Error, match="provider evaluation"):
            holder.adapter.evaluate({})


def test_tree_state_detects_toctou_and_symlink(tmp_path: Path) -> None:
    root = tmp_path / "safe"
    root.mkdir()
    (root / "item").write_text("one")
    before = HARNESS.tree_state(root, label="fixture")
    (root / "item").write_text("two")
    assert (
        before["tree_sha256"]
        != HARNESS.tree_state(root, label="fixture")["tree_sha256"]
    )
    (root / "link").symlink_to(root / "item")
    with pytest.raises(HARNESS.R5Error, match="symlink"):
        HARNESS.tree_state(root, label="fixture")


def test_egress_sentinel_blocks_os_process_primitives_and_pycache() -> None:
    original = sys.dont_write_bytecode
    with HARNESS._egress_sentinel() as attempts:
        assert sys.dont_write_bytecode is True
        with pytest.raises(HARNESS.R5Error, match="process"):
            os.system("true")
        assert attempts["process_attempts"] == 1
    assert sys.dont_write_bytecode is original


def test_egress_sentinel_blocks_fork_session_spawn_and_exec() -> None:
    with HARNESS._egress_sentinel() as attempts:
        for function, args in (
            (os.fork, ()),
            (os.setsid, ()),
            (os.execv, ("/bin/true", ["true"])),
            (os.posix_spawn, ("/bin/true", ["true"], {})),
        ):
            with pytest.raises(HARNESS.R5Error, match="process"):
                function(*args)
    assert attempts["process_attempts"] == 4


def test_egress_sentinel_allows_owned_write_and_blocks_outside_write(tmp_path: Path) -> None:
    owned = tmp_path / "owned"
    owned.mkdir()
    with HARNESS._egress_sentinel(owned) as attempts:
        fd = os.open(owned / "ok", os.O_WRONLY | os.O_CREAT, 0o600)
        try:
            assert os.write(fd, b"ok") == 2
        finally:
            os.close(fd)
        with pytest.raises(HARNESS.R5Error, match="filesystem mutation"):
            os.open(tmp_path / "outside", os.O_WRONLY | os.O_CREAT, 0o600)
    assert (owned / "ok").read_bytes() == b"ok"
    assert attempts["process_attempts"] == 1


def test_egress_sentinel_allows_pinned_owned_dirfd_and_rejects_preopened_outside_fd(
    tmp_path: Path,
) -> None:
    owned, outside = tmp_path / "owned", tmp_path / "outside"
    owned.mkdir()
    nested = owned / "nested"
    nested.mkdir()
    outside_fd = os.open(outside, os.O_WRONLY | os.O_CREAT, 0o600)
    owned_dir_fd = os.open(nested, os.O_RDONLY)
    try:
        with HARNESS._egress_sentinel(owned) as attempts:
            fd = os.open("relative", os.O_WRONLY | os.O_CREAT, 0o600, dir_fd=owned_dir_fd)
            try:
                assert os.write(fd, b"ok") == 2
            finally:
                os.close(fd)
            with pytest.raises(HARNESS.R5Error, match="filesystem mutation"):
                os.write(outside_fd, b"blocked")
        assert attempts["process_attempts"] == 1
    finally:
        os.close(owned_dir_fd)
        os.close(outside_fd)
    assert (nested / "relative").read_bytes() == b"ok"


@pytest.mark.skipif(not hasattr(os, "writev"), reason="platform lacks writev")
def test_egress_sentinel_blocks_writev_to_preopened_outside_fd(tmp_path: Path) -> None:
    owned = tmp_path / "owned"
    owned.mkdir()
    outside_fd = os.open(tmp_path / "outside", os.O_WRONLY | os.O_CREAT, 0o600)
    try:
        with HARNESS._egress_sentinel(owned) as attempts:
            with pytest.raises(HARNESS.R5Error, match="filesystem mutation"):
                os.writev(outside_fd, [b"blocked"])
        assert attempts["process_attempts"] == 1
    finally:
        os.close(outside_fd)


@pytest.mark.parametrize("name", [name for name in ("pwrite", "pwritev", "sendfile") if hasattr(os, name)])
def test_egress_sentinel_blocks_low_level_outside_mutation(name: str, tmp_path: Path) -> None:
    owned = tmp_path / "owned"
    owned.mkdir()
    outside_fd = os.open(tmp_path / "outside", os.O_WRONLY | os.O_CREAT, 0o600)
    source_fd = os.open(tmp_path / "source", os.O_WRONLY | os.O_CREAT, 0o600)
    try:
        os.write(source_fd, b"x")
        with HARNESS._egress_sentinel(owned) as attempts:
            with pytest.raises(HARNESS.R5Error, match="filesystem mutation"):
                if name == "pwrite":
                    os.pwrite(outside_fd, b"x", 0)
                elif name == "pwritev":
                    os.pwritev(outside_fd, [b"x"], 0)
                else:
                    os.sendfile(outside_fd, source_fd, 0, 1)
        assert attempts["process_attempts"] == 1
    finally:
        os.close(source_fd)
        os.close(outside_fd)


@pytest.mark.skipif(not hasattr(socket.socket, "sendfile"), reason="platform lacks socket.sendfile")
def test_egress_sentinel_blocks_socket_sendfile(tmp_path: Path) -> None:
    owned = tmp_path / "owned"
    owned.mkdir()
    source = tmp_path / "source"
    source.write_bytes(b"x")
    left, right = socket.socketpair()
    try:
        with source.open("rb") as handle, HARNESS._egress_sentinel(owned) as attempts:
            with pytest.raises(HARNESS.R5Error, match="network egress"):
                left.sendfile(handle)
        assert attempts["egress_attempts"] == 1
    finally:
        left.close()
        right.close()


@pytest.mark.skipif(not hasattr(socket, "send_fds"), reason="platform lacks socket.send_fds")
def test_egress_sentinel_blocks_socket_send_fds(tmp_path: Path) -> None:
    owned = tmp_path / "owned"
    owned.mkdir()
    left, right = socket.socketpair()
    try:
        with HARNESS._egress_sentinel(owned) as attempts:
            with pytest.raises(HARNESS.R5Error, match="network egress"):
                socket.send_fds(left, [b"blocked"], [])
        assert attempts["egress_attempts"] == 1
    finally:
        left.close()
        right.close()


def test_provider_sentinel_counts_and_fails_adapter_evaluation() -> None:
    class Adapter:
        def evaluate(self, _payload: object) -> object:
            return {"unexpected": True}

    runtime = types.SimpleNamespace(Adapter=Adapter)
    attempts = {"provider_calls": 0, "egress_attempts": 0, "process_attempts": 0}
    with HARNESS._provider_sentinel(attempts, runtime):
        with pytest.raises(HARNESS.R5Error, match="provider evaluation"):
            Adapter().evaluate({})
    assert attempts["provider_calls"] == 1


def test_provider_sentinel_reaches_nested_adapter_instance() -> None:
    class Adapter:
        def generate(self, _payload: object) -> object:
            return {"unexpected": True}

    runtime = types.SimpleNamespace(holder=types.SimpleNamespace(adapter=Adapter()))
    attempts = {"provider_calls": 0, "egress_attempts": 0, "process_attempts": 0}
    with HARNESS._provider_sentinel(attempts, runtime):
        with pytest.raises(HARNESS.R5Error, match="provider evaluation"):
            runtime.holder.adapter.generate({})
    assert attempts["provider_calls"] == 1


def test_provider_sentinel_blocks_instance_slots_property_and_module_boundaries() -> None:
    class InstanceAdapter:
        def __init__(self) -> None:
            self.evaluate = lambda _payload: {"unexpected": True}
            self.helper = lambda value: value + 1

    class SlotsAdapter:
        __slots__ = ("generate", "helper")

        def __init__(self) -> None:
            self.generate = lambda _payload: {"unexpected": True}
            self.helper = lambda value: value + 1

    class PropertyAdapter:
        @property
        def compare(self) -> Callable[[object], object]:
            return lambda _payload: {"unexpected": True}

    runtime: Any = types.ModuleType("provider_boundary_fixture")
    runtime.provider = lambda _payload: {"unexpected": True}
    runtime.evaluate = lambda _payload: {"unexpected": True}
    runtime.helper = lambda value: value + 1
    holder = types.SimpleNamespace(
        instance=InstanceAdapter(), slots=SlotsAdapter(), property=PropertyAdapter()
    )
    runtime.holder = holder
    attempts = {"provider_calls": 0, "egress_attempts": 0, "process_attempts": 0}
    with HARNESS._provider_sentinel(attempts, runtime):
        for callback in (
            runtime.provider, runtime.evaluate, holder.instance.evaluate,
            holder.slots.generate, holder.property.compare,
        ):
            with pytest.raises(HARNESS.R5Error, match="provider evaluation"):
                callback({})
        assert runtime.helper(1) == 2
        assert holder.instance.helper(1) == 2
        assert holder.slots.helper(1) == 2
    assert attempts["provider_calls"] == 5
    assert runtime.helper(1) == 2
    assert holder.instance.evaluate({}) == {"unexpected": True}
    assert holder.slots.generate({}) == {"unexpected": True}
    assert holder.property.compare({}) == {"unexpected": True}


def test_provider_sentinel_rejects_importlib_dynamic_module(tmp_path: Path) -> None:
    module_name = "provider_importlib_fixture"
    prior = sys.modules.get(module_name)
    (tmp_path / f"{module_name}.py").write_text("value = 1\n")
    sys.path.insert(0, str(tmp_path))
    attempts = {"provider_calls": 0, "egress_attempts": 0, "process_attempts": 0}
    try:
        with HARNESS._provider_sentinel(attempts, types.SimpleNamespace()):
            with pytest.raises(HARNESS.R5Error, match="dynamic imports"):
                importlib.import_module(module_name)
    finally:
        if prior is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = prior
        with contextlib.suppress(ValueError):
            sys.path.remove(str(tmp_path))
    assert attempts["provider_calls"] == 0


def test_provider_sentinel_allows_large_benign_module_graph() -> None:
    runtime: Any = types.ModuleType("benign_runtime_fixture")
    for index in range(600):
        setattr(runtime, f"helper_{index}", types.SimpleNamespace(value=index))
    runtime.helper = lambda value: value + 1
    attempts = {"provider_calls": 0, "egress_attempts": 0, "process_attempts": 0}
    with HARNESS._provider_sentinel(attempts, runtime):
        assert runtime.helper(1) == 2
    assert attempts["provider_calls"] == 0


def test_provider_sentinel_blocks_inherited_and_aliased_provider_callables() -> None:
    class Service:
        def evaluate(self, _payload: object) -> object:
            return {"unexpected": True}

    class SlotsService(Service):
        __slots__ = ("helper",)

        def __init__(self) -> None:
            self.helper = Service.evaluate

    runtime = types.SimpleNamespace(service=SlotsService())
    attempts = {"provider_calls": 0, "egress_attempts": 0, "process_attempts": 0}
    with HARNESS._provider_sentinel(attempts, runtime):
        with pytest.raises(HARNESS.R5Error, match="provider evaluation"):
            runtime.service.evaluate({})
        with pytest.raises(HARNESS.R5Error, match="provider evaluation"):
            runtime.service.helper(runtime.service, {})
    assert attempts["provider_calls"] == 2


def test_provider_sentinel_reaches_neutral_class_descriptors() -> None:
    class Service:
        def evaluate(self, _payload: object) -> object:
            return {"unexpected": True}

    class PropertyService:
        @property
        def evaluate(self) -> Callable[[object], object]:
            return lambda _payload: {"unexpected": True}

    class NeutralSlots:
        __slots__ = ()

        def evaluate(self, _payload: object) -> object:
            return {"unexpected": True}

    runtime = types.SimpleNamespace(
        service=Service(), property_service=PropertyService(), slots=NeutralSlots(),
    )
    attempts = {"provider_calls": 0, "egress_attempts": 0, "process_attempts": 0}
    with HARNESS._provider_sentinel(attempts, runtime):
        for callback in (
            runtime.service.evaluate, runtime.property_service.evaluate, runtime.slots.evaluate,
        ):
            with pytest.raises(HARNESS.R5Error, match="provider evaluation"):
                callback({})
    assert attempts["provider_calls"] == 3


def test_provider_sentinel_blocks_pre_captured_import_aliases(tmp_path: Path) -> None:
    module_name = "provider_alias_import_fixture"
    (tmp_path / f"{module_name}.py").write_text("value = 1\n")
    sys.path.insert(0, str(tmp_path))
    direct_import, module_import = builtins.__import__, importlib.import_module
    attempts = {"provider_calls": 0, "egress_attempts": 0, "process_attempts": 0}
    try:
        with HARNESS._provider_sentinel(attempts, types.SimpleNamespace()):
            with pytest.raises(HARNESS.R5Error, match="dynamic imports"):
                direct_import(module_name)
            with pytest.raises(HARNESS.R5Error, match="dynamic imports"):
                module_import(module_name)
    finally:
        sys.modules.pop(module_name, None)
        with contextlib.suppress(ValueError):
            sys.path.remove(str(tmp_path))


def test_module_provenance_requires_canonical_cache_identity() -> None:
    trusted = HARNESS._TRUSTED_BOOTSTRAP_MODULES["time"]
    fake = types.ModuleType("time")
    fake.__spec__ = trusted.__spec__
    fake.__file__ = getattr(trusted, "__file__", None)
    assert HARNESS._module_provenance_allowed(fake, ROOT) is False


def test_workset_inventory_rejects_audit_sidecar_mutation(tmp_path: Path) -> None:
    root = tmp_path / "clone"
    db = root / "runtime" / "recall-distillation" / "ox-workset.sqlite3"
    db.parent.mkdir(parents=True)
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE work_items (stage TEXT, state TEXT, completion_ref TEXT, completion_digest TEXT, sequence INTEGER)")
        conn.execute("CREATE TABLE workset_receipts (generation INTEGER, previous_sha256 TEXT, operation TEXT, payload_json TEXT, receipt_sha256 TEXT)")

    class Runtime:
        class DistillationWorkset:
            def __init__(self, _path: Path) -> None:
                self.path = _path

            def audit_transition_receipts(self) -> dict[str, object]:
                self.path.with_name(f"{self.path.name}-wal").write_bytes(b"late")
                return {}

    with pytest.raises(HARNESS.R5Error, match="receipt audit"):
        HARNESS._workset_inventory(root, Runtime)


def test_workset_inventory_replays_wal_only_schema_and_rows(tmp_path: Path) -> None:
    root = tmp_path / "clone"
    db = root / "runtime" / "recall-distillation" / "ox-workset.sqlite3"
    db.parent.mkdir(parents=True)
    conn = sqlite3.connect(db)
    try:
        assert conn.execute("PRAGMA journal_mode=WAL").fetchone() == ("wal",)
        conn.execute("PRAGMA wal_autocheckpoint=0")
        conn.execute("CREATE TABLE work_items (state TEXT, completion_ref TEXT, completion_digest TEXT, sequence INTEGER)")
        conn.execute("CREATE TABLE workset_receipts (generation INTEGER, previous_sha256 TEXT, operation TEXT, payload_json TEXT, receipt_sha256 TEXT)")
        conn.execute("INSERT INTO work_items VALUES ('completed', 'ref', 'digest', 1)")
        conn.commit()
        assert db.with_name(f"{db.name}-wal").exists()
        inventory = HARNESS._workset_inventory(root)
        assert inventory["counts"] == {"completed": 1}
        assert inventory["completed_refs"] == [{"ref": "ref", "digest": "digest"}]
        assert inventory["receipt_count"] == 0
    finally:
        conn.close()


def test_workset_inventory_rejects_forged_runtime_audit(tmp_path: Path) -> None:
    root = tmp_path / "clone"
    db = root / "runtime" / "recall-distillation" / "ox-workset.sqlite3"
    db.parent.mkdir(parents=True)
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE work_items (state TEXT, completion_ref TEXT, completion_digest TEXT, sequence INTEGER)")
        conn.execute("CREATE TABLE workset_receipts (generation INTEGER, previous_sha256 TEXT, operation TEXT, payload_json TEXT, receipt_sha256 TEXT)")

    class Runtime:
        class DistillationWorkset:
            def __init__(self, _path: Path) -> None:
                pass

            def audit_transition_receipts(self) -> dict[str, object]:
                return {"receipts": 1, "generation": 1, "head_sha256": "forged", "counts": {}}

    with pytest.raises(HARNESS.R5Error, match="does not bind"):
        HARNESS._workset_inventory(root, Runtime)


def test_workset_inventory_rejects_self_consistent_forged_receipt_row(tmp_path: Path) -> None:
    root = tmp_path / "clone"
    db = root / "runtime" / "recall-distillation" / "ox-workset.sqlite3"
    db.parent.mkdir(parents=True)
    sys.path.insert(0, str(ROOT / "src"))
    workset = importlib.import_module("chronovisor.recall.recall_distillation_workset")
    workset.DistillationWorkset(db)
    with sqlite3.connect(db) as conn:
        conn.execute("INSERT INTO workset_receipts VALUES (1, '', 'advance', '{}', 'not-a-hash')")
    with pytest.raises(HARNESS.R5Error, match="receipt"):
        HARNESS._workset_inventory(root)


def test_workset_inventory_rejects_canonical_noop_advance_receipt(tmp_path: Path) -> None:
    root = tmp_path / "clone"
    db = root / "runtime" / "recall-distillation" / "ox-workset.sqlite3"
    db.parent.mkdir(parents=True)
    counts = {state: 0 for state in ("ready", "leased", "completed", "quarantined")}
    payload = {
        "before": {"counts": counts, "watermark": None},
        "after": {"counts": counts, "watermark": None},
        "delta": counts,
        "details": {"inserted": 0, "rebound": 0, "watermark_changed": False, "selection_sha256": _id(1)},
    }
    receipt = HARNESS._sha({"generation": 1, "previous_sha256": "", "operation": "advance", "payload": payload})
    sys.path.insert(0, str(ROOT / "src"))
    workset = importlib.import_module("chronovisor.recall.recall_distillation_workset")
    workset.DistillationWorkset(db)
    with sqlite3.connect(db) as conn:
        conn.execute(
            "INSERT INTO workset_receipts VALUES (?, ?, ?, ?, ?)",
            (1, "", "advance", HARNESS._canonical(payload).decode(), receipt),
        )
    with pytest.raises(HARNESS.R5Error, match="advance"):
        HARNESS._workset_inventory(root)


def test_actual_workset_and_materialization_api_on_owned_clone(tmp_path: Path) -> None:
    """The formal path uses real project APIs; its empty owned clone declines."""
    sys.path.insert(0, str(ROOT / "src"))
    distill = importlib.import_module("chronovisor.recall.recall_distillation")
    workset = importlib.import_module("chronovisor.recall.recall_distillation_workset")

    clone = tmp_path / "clone"
    (clone / "runtime" / "recall-distillation").mkdir(parents=True)
    queue = workset.DistillationWorkset(
        clone / "runtime" / "recall-distillation" / "ox-workset.sqlite3"
    )
    queue.advance(
        [
            {
                "work_id": "r5-item",
                "kind": "ox",
                "payload_ref": "label-ledger:r5-item",
                "payload_digest": _id(7),
                "temporal_split": {},
                "provenance": {},
                "priority": 1,
            }
        ],
        {"source": "test"},
    )
    # This is the real materialization function, with a clone-owned root and
    # no fabricated rows.  An empty dataset is the expected capture-only case.
    materialized = distill.materialize_training_rows(
        clone, _rallies=[], _snapshots={}, _label_rows=[]
    )
    assert materialized["rows"] == []
    inventory = HARNESS._workset_inventory(clone)
    assert inventory["counts"]["ready"] == 1
    assert (
        HARNESS.validate_dataset(
            rows=[], labels=[], preflight={}, gate={"passed": False}, workset=inventory
        )["passed"]
        is False
    )


def test_clone_cleanup_receipt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    production, source, output, clone = (
        tmp_path / name for name in ("production", "source", "output", "clone")
    )
    for root in (production, source, clone):
        root.mkdir()
    for root in (production, clone):
        (root / "raw").mkdir()
        (root / "runtime" / "recall-distillation").mkdir(parents=True)
    (source / "src").mkdir()
    (source / "marker").write_text("source")
    monkeypatch.setattr(
        HARNESS,
        "source_state",
        lambda _source, _commit: {"commit": "a" * 40, "tree_sha256": "b" * 64},
    )
    monkeypatch.setattr(
        HARNESS, "_verify_r4", lambda *_args: {"artifact_id": "c" * 64}
    )
    monkeypatch.setattr(HARNESS, "_kernel_sandbox_attested", lambda: True)
    monkeypatch.setattr(HARNESS, "_clone", lambda _production: (clone, {"owned": True}))
    monkeypatch.setattr(
        HARNESS, "_load_runtime", lambda _source: (object(), object(), object())
    )
    monkeypatch.setattr(HARNESS, "_read_rows", lambda _store, _clone: [])
    # Avoid runtime import here; force a truthful capture failure after the clone
    # boundary while still exercising finally cleanup.
    artifact, _ = HARNESS.run(
        production=production,
        source=source,
        source_commit="a" * 40,
        output=output,
        r4_artifact=tmp_path / "r4.json",
        test_only=True,
        kernel_attested=True,
    )
    assert artifact["cleanup"] == {"clone_removed": True, "remaining": 0}
    assert not clone.exists()


def test_run_detects_production_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    production, source, output, clone = (
        tmp_path / name for name in ("production", "source", "output", "clone")
    )
    for root in (production, source, clone):
        root.mkdir()
    for root in (production, clone):
        (root / "raw").mkdir()
        (root / "runtime" / "recall-distillation").mkdir(parents=True)
    (source / "src").mkdir()
    (source / "marker").write_text("source")
    monkeypatch.setattr(
        HARNESS,
        "source_state",
        lambda _source, _commit: {"commit": "a" * 40, "tree_sha256": "b" * 64},
    )
    monkeypatch.setattr(
        HARNESS, "_verify_r4", lambda *_args: {"artifact_id": "c" * 64}
    )
    monkeypatch.setattr(HARNESS, "_kernel_sandbox_attested", lambda: True)
    monkeypatch.setattr(HARNESS, "_clone", lambda _production: (clone, {"owned": True}))
    monkeypatch.setattr(
        HARNESS, "_load_runtime", lambda _source: (object(), object(), object())
    )
    monkeypatch.setattr(HARNESS, "_read_rows", lambda _store, _clone: [])

    def mutate(_clone: Path, _runtime: object | None = None) -> dict[str, object]:
        (production / "raw" / "mutation").write_text("detected")
        return {
            "present": False,
            "counts": {},
            "completed_refs": [],
            "receipt_head": None,
        }

    monkeypatch.setattr(HARNESS, "_workset_inventory", mutate)
    with pytest.raises(HARNESS.R5Error, match="production changed"):
        HARNESS.run(
            production=production,
            source=source,
            source_commit="a" * 40,
            output=output,
            r4_artifact=tmp_path / "r4.json",
            test_only=True,
            kernel_attested=True,
        )
    assert not clone.exists()


def test_run_restores_preexisting_chronovisor_modules(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    production, source, clone = (tmp_path / name for name in ("production", "source", "clone"))
    for root in (production, source, clone):
        root.mkdir()
    for root in (production, clone):
        (root / "raw").mkdir()
        (root / "runtime" / "recall-distillation").mkdir(parents=True)
    previous = {
        name: module for name, module in sys.modules.items()
        if name == "chronovisor" or name.startswith("chronovisor.")
    }
    sentinel = types.ModuleType("chronovisor.r5_cache_fixture")
    sys.modules[sentinel.__name__] = sentinel
    monkeypatch.setattr(HARNESS, "source_state", lambda *_args: {"commit": "a" * 40, "tree_sha256": _id(1)})
    monkeypatch.setattr(HARNESS, "_verify_r4", lambda *_args: {"artifact_id": _id(2)})
    monkeypatch.setattr(HARNESS, "_kernel_sandbox_attested", lambda: True)
    monkeypatch.setattr(HARNESS, "_clone", lambda _production: (clone, {"owned": True}))

    def contaminate(_source: Path) -> tuple[object, object, object]:
        sys.modules["chronovisor"] = types.ModuleType("chronovisor")
        sys.modules["chronovisor.recall"] = types.ModuleType("chronovisor.recall")
        raise HARNESS.R5Error("runtime stop")

    monkeypatch.setattr(HARNESS, "_load_runtime", contaminate)
    try:
        with pytest.raises(HARNESS.R5Error, match="runtime stop"):
            HARNESS.run(
                production=production, source=source, source_commit="a" * 40,
                output=tmp_path / "output", r4_artifact=tmp_path / "r4.json",
                test_only=True, kernel_attested=True,
            )
        assert sys.modules[sentinel.__name__] is sentinel
        restored = {
            name: module for name, module in sys.modules.items()
            if name == "chronovisor" or name.startswith("chronovisor.")
        }
        assert restored == {**previous, sentinel.__name__: sentinel}
        for name, module in restored.items():
            parent_name, _, attribute = name.rpartition(".")
            if parent_name in restored:
                assert getattr(restored[parent_name], attribute) is module
    finally:
        for name in [
            name for name in sys.modules
            if name == "chronovisor" or name.startswith("chronovisor.")
        ]:
            sys.modules.pop(name, None)
        sys.modules.update(previous)


def _formal_inner() -> dict[str, object]:
    file_state = {"bytes": 1, "mtime_ns": 1, "dev": 1, "ino": 1, "sha256": _id(1)}
    identity = {
        "commit": "a" * 40, "clean": True, "status_sha256": _id(2), "status_count": 0,
        "tree_sha256": _id(3), "file_count": 1, "symlink_count": 0,
        "ox_identity_sha256": _id(4), "account_uid": 1, "account_home": "/tmp",
        "tree": "b" * 40, "index_count": 1, "index_sha256": _id(5),
        "git_index": file_state, "tracked_bytes_sha256": _id(6),
        "tool_identities": [{"path": "/usr/bin/git", "uid": 0, "mode": 0o755, "sha256": _id(7)}],
    }
    production = {
        "root": {"dev": 1, "ino": 1, "uid": 1, "gid": 1, "mode": 0o700, "ctime_ns": 1},
        "managed_inventory": {"entries": 1, "sha256": _id(8)},
        "raw": {"file_count": 1, "tree_sha256": _id(9)},
        "runtime": {"file_count": 1, "tree_sha256": _id(10)},
        "config": file_state,
        "protected_inventory": {"included": ["raw"], "excluded": ["other"], "excluded_sha256": _id(11)},
    }
    clone = {
        "owned": True, "cow": "copyfile(3):COPYFILE_CLONE_FORCE", "dev": 1, "ino": 2, "volume": "apfs",
        "tool": {"backend": "copyfile(3)", "flags": HARNESS.R2.COPYFILE_CLONE_FORCE}, "parity": production,
        "state": {"file_count": 1, "tree_sha256": _id(12), "files": _id(13)},
    }
    dataset = {
        "passed": True, "capture_only": False, "reasons": [],
        "metrics": {
            "rows": 500, "labels": 500, "valid_labels": 500,
            "classes": {"relevant": 250, "irrelevant": 250}, "local_probe_pairs": 100,
            "blind_pairs": 100, "counterfactual_pairs": 100,
            "age_bands": {"0_7": 1, "8_30": 1, "31_plus": 1},
            "future_leakage": 0, "feature_parity_percent": 100,
        },
    }
    return cast(dict[str, object], HARNESS._sealed_artifact(
        {
            "captured_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "source": identity,
            "source_after": identity,
            "production": production,
            "production_after": production,
            "clone": clone,
            "r4_dependency": {
                "artifact_id": _id(2), "seal_sha256": _id(3),
                "artifact_path": "/tmp/r4-artifact.json", "artifact_file_state": file_state,
                "authority_artifact_id": _id(14), "authority_seal_sha256": _id(15),
                "authority_relative_path": f"{_id(14)}.authority.json",
                "authority_file_state": {**file_state, "sha256": _id(16)},
                "source_root": str(Path("/tmp").resolve()), "source_commit": "a" * 40,
                "source_tree_sha256": _id(3),
            },
            "dataset": dataset,
            "phases": [],
            "cleanup": {"clone_removed": True, "remaining": 0},
            "provider_calls": 0,
            "egress_attempts": 0,
            "process_attempts": 0,
            "supervised": False,
            "test_only": False,
        }
    ))


def _reseal_inner(inner: dict[str, object], **changes: object) -> dict[str, object]:
    payload = {
        key: value
        for key, value in inner.items()
        if key not in {"artifact_id", "schema", "namespace", "seal_sha256"}
    }
    payload.update(changes)
    return cast(dict[str, object], HARNESS._sealed_artifact(payload))


@pytest.mark.parametrize(
    "dataset_change",
    (
        {"passed": True, "capture_only": True, "reasons": ["clone_api_capture_failed"]},
        {"passed": True, "capture_only": False, "reasons": ["future_leakage_detected"]},
    ),
)
def test_passed_artifact_rejects_any_decline_state(dataset_change: dict[str, object]) -> None:
    inner = _formal_inner()
    dataset = dict(cast(dict[str, object], inner["dataset"]))
    dataset.update(dataset_change)
    with pytest.raises(HARNESS.R5Error, match="decline state"):
        _reseal_inner(inner, dataset=dataset)


@pytest.mark.parametrize("metric", ("future_leakage", "feature_parity_percent", "rows"))
def test_formal_artifact_rederives_metric_floor(metric: str) -> None:
    inner = _formal_inner()
    dataset = dict(cast(dict[str, object], inner["dataset"]))
    metrics = dict(cast(dict[str, object], dataset["metrics"]))
    metrics[metric] = 1 if metric == "future_leakage" else 0
    dataset["metrics"] = metrics
    with pytest.raises(HARNESS.R5Error, match="dataset floors"):
        _reseal_inner(inner, dataset=dataset)


def test_formal_clone_requires_production_parity_and_rederived_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    inner = _formal_inner()
    clone = dict(cast(dict[str, object], inner["clone"]))
    clone["parity"] = {}
    with pytest.raises(HARNESS.R5Error, match="clone parity"):
        _reseal_inner(inner, clone=clone)

    clone = dict(cast(dict[str, object], inner["clone"]))
    state = dict(cast(dict[str, object], clone["state"]))
    state["files"] = _id(99)
    clone["state"] = state
    altered = _reseal_inner(inner, clone=clone)
    completion = _completion(altered, tmp_path)
    monkeypatch.setattr(HARNESS, "_verify_r4", lambda *_args: altered["r4_dependency"])
    monkeypatch.setattr(HARNESS, "production_state", lambda _root: inner["production"])
    monkeypatch.setattr(HARNESS, "_clone_input_state", lambda _root: inner["clone"]["state"])
    with pytest.raises(HARNESS.R5Error, match="clone or production"):
        HARNESS.assert_formal_acceptance(
            altered, completion, source_root=Path("/tmp"), r4_artifact=tmp_path / "r4.json",
        )


@pytest.mark.parametrize("replace", ("artifact", "authority", "swapped"))
def test_verify_r4_rejects_post_validation_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, replace: str,
) -> None:
    artifact_path = tmp_path / "r4.json"
    authority_path = tmp_path / f"{_id(14)}.authority.json"
    artifact_path.write_text("artifact")
    authority_path.write_text("authority")
    source = {"commit": "a" * 40, "tree_sha256": _id(3)}
    authority_state = HARNESS._file_state(authority_path)
    artifact = {
        "artifact_id": _id(2), "seal_sha256": _id(3),
        "production_certification": {
            "passed": True, "provider_calls": 0,
            "collector": "fixed-production-root-workset-v1",
        },
        "production_root_used": True,
        "source": {**source, "clean": True, "status_count": 0},
        "source_after": {**source, "clean": True, "status_count": 0},
        "source_final": {**source, "clean": True, "status_count": 0},
        "source_contract": {"passed": True},
        "receipt_files": {
            key: {"count": 1, "files": [key]} for key in ("local", "ox", "production")
        },
        "authority_receipt": {
            "available": True, "artifact_id": _id(14), "seal_sha256": _id(15),
            "relative_path": authority_path.name, "file_sha256": authority_state["sha256"],
            "parent_dev": authority_path.parent.stat().st_dev,
            "parent_ino": authority_path.parent.stat().st_ino,
        },
    }
    monkeypatch.setattr(HARNESS.R4, "read_artifact", lambda _path: artifact)

    def validate(_authority: Path, **_kwargs: object) -> dict[str, object]:
        if replace in {"artifact", "authority"}:
            (artifact_path if replace == "artifact" else authority_path).write_text("replaced")
        return {
            "artifact_id": _id(17) if replace == "swapped" else _id(14),
            "r4_artifact_id": _id(2),
            "file_state": authority_state,
        }

    monkeypatch.setattr(HARNESS.R4, "validate_source_bound_authority_receipt", validate)
    with pytest.raises(HARNESS.R5Error, match="changed during binding"):
        HARNESS._verify_r4(artifact_path, source, tmp_path)


def test_formal_acceptance_rejects_r4_path_or_source_root_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    inner = _formal_inner()
    completion = _completion(inner, tmp_path)
    expected_path = Path("/tmp/r4-artifact.json")

    def verify(path: Path, _source: object, root: Path) -> object:
        if path != expected_path or root != Path("/tmp"):
            raise HARNESS.R5Error("R4 authority receipt cannot be source-bound")
        return inner["r4_dependency"]

    monkeypatch.setattr(HARNESS, "_verify_r4", verify)
    monkeypatch.setattr(HARNESS, "production_state", lambda _root: inner["production"])
    monkeypatch.setattr(HARNESS, "_clone_input_state", lambda _root: inner["clone"]["state"])
    with pytest.raises(HARNESS.R5Error, match="R4 authority"):
        HARNESS.assert_formal_acceptance(
            inner, completion, source_root=tmp_path, r4_artifact=expected_path,
        )
    with pytest.raises(HARNESS.R5Error, match="R4 authority"):
        HARNESS.assert_formal_acceptance(
            inner, completion, source_root=Path("/tmp"), r4_artifact=tmp_path / "other-r4.json",
        )


@pytest.mark.parametrize(
    ("key", "value"),
    (
        ("commit", "A" * 40),
        ("tree", "b" * 39),
        ("status_sha256", "c" * 63),
        ("index_sha256", "d" * 63),
        ("tracked_bytes_sha256", "e" * 63),
    ),
)
def test_formal_source_identity_rejects_nonexact_hashes(key: str, value: str) -> None:
    inner = _formal_inner()
    source = cast(dict[str, object], inner["source"])
    source[key] = value
    with pytest.raises(HARNESS.R5Error, match="identity values"):
        _reseal_inner(inner)


@pytest.mark.parametrize("captured_at", ("2099-01-01T00:00:00Z", "2026-08-24T00:00:00+00:00"))
def test_artifact_rejects_future_or_noncanonical_captured_at(captured_at: str) -> None:
    with pytest.raises(HARNESS.R5Error, match="captured_at|UTC"):
        _reseal_inner(_formal_inner(), captured_at=captured_at)


@pytest.mark.parametrize("field", ("source", "production", "clone", "reasons"))
def test_decline_union_fields_reject_nested_or_sensitive_values(field: str) -> None:
    inner = _formal_inner()
    payload = {
        key: value for key, value in inner.items()
        if key not in {"artifact_id", "schema", "namespace", "seal_sha256"}
    }
    payload["dataset"] = {
        "passed": False, "capture_only": True, "reasons": ["dataset_floor_not_met"],
        "metrics": cast(dict[str, object], inner["dataset"])["metrics"],
    }
    if field == "source":
        payload["source"] = {"commit": {"nested": True}, "tree_sha256": []}
    elif field == "production":
        payload["production"] = {"raw": {"nested": True}, "runtime": [], "config": {}}
    elif field == "clone":
        payload["clone"] = {"test_only": {"nested": True}}
    else:
        cast(dict[str, object], payload["dataset"])["reasons"] = ["raw transcript SECRET password=foo"]
    with pytest.raises(HARNESS.R5Error, match="artifact"):
        HARNESS._sealed_artifact(payload)


def _completion(inner: dict[str, object], tmp_path: Path) -> dict[str, object]:
    inner_path = HARNESS._write_immutable(tmp_path / "r5-inner", inner)
    inner_parent = HARNESS._directory_identity(inner_path.parent)
    output_identity = HARNESS._directory_identity(tmp_path)
    ended_at = datetime.now(UTC).replace(microsecond=0)
    started_at = ended_at - timedelta(seconds=1)
    return cast(dict[str, object], HARNESS._sealed_completion(
        {
            "inner": {
                "artifact_id": inner["artifact_id"],
                "path": str(inner_path),
                "file_sha256": HARNESS._file_state(inner_path)["sha256"],
                "source_commit": "a" * 40,
                "r4_artifact_id": _id(2),
                "parent_dev": inner_parent["dev"],
                "parent_ino": inner_parent["ino"],
            },
            "supervisor": {
                "pid": 1,
                "pgid": 1,
                "started_at": started_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "ended_at": ended_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "elapsed_ms": 1000,
                "deadline_seconds": 1,
                "returncode": 0,
                "signal": None,
                "timeout": False,
                "descendants_remaining": 0,
                "observed_descendant_pids": [],
            },
            "source": inner["source"],
            "source_after": inner["source_after"],
            "source_final": inner["source_after"],
            "production": inner["production"],
            "production_after": inner["production_after"],
            "production_final": inner["production_after"],
            "output": {"dev": output_identity["dev"], "ino": output_identity["ino"]},
            "cleanup": {"clone_remaining": 0, "temporary_remaining": 0},
            "formal_passed": True,
        }
    ))


def test_completion_requires_matching_inner_and_detects_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    inner = _formal_inner()
    completion = _completion(inner, tmp_path)
    monkeypatch.setattr(HARNESS, "_verify_r4", lambda *_args: inner["r4_dependency"])
    monkeypatch.setattr(HARNESS, "production_state", lambda _root: inner["production"])
    monkeypatch.setattr(
        HARNESS, "_clone_input_state", lambda _root: inner["clone"]["state"],
    )
    HARNESS.assert_formal_acceptance(
        inner, completion, source_root=Path("/tmp"), r4_artifact=tmp_path / "r4.json",
    )
    path = HARNESS._write_completion(tmp_path, completion)
    assert HARNESS.read_completion(path)["artifact_id"] == completion["artifact_id"]
    altered = json.loads(path.read_text())
    altered["formal_passed"] = False
    path.write_text(json.dumps(altered, sort_keys=True))
    with pytest.raises(HARNESS.R5Error, match="completion"):
        HARNESS.read_completion(path)


def test_completion_rejects_non_utc_or_inconsistent_supervisor_timing(tmp_path: Path) -> None:
    inner = _formal_inner()
    completion = _completion(inner, tmp_path)
    supervisor = cast(dict[str, object], completion["supervisor"])
    supervisor["started_at"] = "2026-08-25T00:00:02+00:00"
    supervisor["ended_at"] = "2026-08-25T00:00:01+00:00"
    payload = {key: value for key, value in completion.items() if key not in {"artifact_id", "schema", "namespace", "seal_sha256"}}
    sealed = HARNESS._sealed_completion(payload)
    path = tmp_path / "bad.completion.json"
    HARNESS._write_private_json(path, sealed)
    with pytest.raises(HARNESS.R5Error, match="UTC|nested values"):
        HARNESS.read_completion(path)


@pytest.mark.parametrize(
    ("started_at", "ended_at", "elapsed_ms", "deadline_seconds"),
    (
        ("2099-01-01T00:00:00Z", "2099-01-01T00:00:01Z", 1000, 1),
        ("2026-08-24T00:00:00Z", "2026-08-24T01:00:00Z", 3_600_000, 1),
    ),
)
def test_completion_rejects_future_or_deadline_exceeding_supervisor_timing(
    tmp_path: Path, started_at: str, ended_at: str, elapsed_ms: int, deadline_seconds: int,
) -> None:
    inner = _formal_inner()
    completion = _completion(inner, tmp_path)
    supervisor = cast(dict[str, object], completion["supervisor"])
    supervisor.update(
        started_at=started_at, ended_at=ended_at, elapsed_ms=elapsed_ms,
        deadline_seconds=deadline_seconds,
    )
    payload = {key: value for key, value in completion.items() if key not in {"artifact_id", "schema", "namespace", "seal_sha256"}}
    path = tmp_path / "bad-timing.completion.json"
    HARNESS._write_private_json(path, HARNESS._sealed_completion(payload))
    with pytest.raises(HARNESS.R5Error, match="UTC|nested values"):
        HARNESS.read_completion(path)


def test_completion_rejects_deleted_or_replaced_persistent_inner(tmp_path: Path) -> None:
    inner = _formal_inner()
    completion = _completion(inner, tmp_path)
    path = HARNESS._write_completion(tmp_path, completion)
    inner_path = Path(str(cast(dict[str, object], completion["inner"])["path"]))
    inner_path.unlink()
    with pytest.raises(HARNESS.R5Error):
        HARNESS.read_completion(path)


def test_completion_rejects_persistent_inner_parent_inode_swap(tmp_path: Path) -> None:
    inner = _formal_inner()
    completion = _completion(inner, tmp_path)
    path = HARNESS._write_completion(tmp_path, completion)
    root = tmp_path / "r5-inner"
    root.rename(tmp_path / "r5-inner-replaced")
    root.mkdir(mode=0o700)
    HARNESS._write_immutable(root, inner)
    with pytest.raises(HARNESS.R5Error, match="unavailable"):
        HARNESS.read_completion(path)


def test_managed_inventory_hashes_content_not_restorable_metadata(tmp_path: Path) -> None:
    root = tmp_path / "production"
    root.mkdir()
    item = root / "managed"
    item.write_text("one")
    before = HARNESS._managed_inventory(root)
    state = item.stat()
    item.write_text("two")
    os.utime(item, ns=(state.st_atime_ns, state.st_mtime_ns))
    assert HARNESS._managed_inventory(root) != before


def test_inner_alone_or_mixed_completion_cannot_formally_pass(tmp_path: Path) -> None:
    inner = _formal_inner()
    with pytest.raises(HARNESS.R5Error, match="completion"):
        HARNESS.assert_formal_acceptance(
            inner, {}, source_root=tmp_path, r4_artifact=tmp_path / "r4.json",
        )
    other = _reseal_inner(inner, captured_at="2026-08-24T00:00:01Z")
    mixed = _completion(other, tmp_path)
    with pytest.raises(HARNESS.R5Error, match="completion"):
        HARNESS.assert_formal_acceptance(
            inner, mixed, source_root=tmp_path, r4_artifact=tmp_path / "r4.json",
        )


@pytest.mark.parametrize("field", ["unknown", "raw", "secret"])
def test_artifact_nested_schema_rejects_unknown_or_sensitive_fields(field: str) -> None:
    inner = _formal_inner()
    dataset = cast(dict[str, object], inner["dataset"])
    dataset[field] = "injected"
    with pytest.raises(HARNESS.R5Error, match="artifact"):
        HARNESS._sealed_artifact(
            {
                key: value
                for key, value in inner.items()
                if key not in {"artifact_id", "schema", "namespace", "seal_sha256"}
            }
        )


@pytest.mark.parametrize("field, value", [("returncode", 1), ("timeout", True)])
def test_completion_rejects_fake_return_or_timeout(
    field: str, value: object, tmp_path: Path
) -> None:
    inner = _formal_inner()
    completion = _completion(inner, tmp_path)
    cast(dict[str, object], completion["supervisor"])[field] = value
    with pytest.raises(HARNESS.R5Error, match="completion"):
        HARNESS.assert_formal_acceptance(
            inner, completion, source_root=tmp_path, r4_artifact=tmp_path / "r4.json",
        )


def test_completion_rejects_untruthful_descendant_pid_receipt(tmp_path: Path) -> None:
    inner = _formal_inner()
    completion = _completion(inner, tmp_path)
    cast(dict[str, object], completion["supervisor"])["observed_descendant_pids"] = [2, 1, 2]
    with pytest.raises(HARNESS.R5Error, match="completion"):
        HARNESS.assert_formal_acceptance(
            inner, completion, source_root=tmp_path, r4_artifact=tmp_path / "r4.json",
        )


def test_pid_registry_rejects_unrelated_pid_with_exact_start_token(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    root = {"pid": 10, "ppid": 1, "uid": 501, "start": "exact-token"}
    unrelated = {"pid": 20, "ppid": 1, "uid": 501, "start": "exact-token"}
    monkeypatch.setattr(HARNESS, "_process_record", lambda pid: {10: root, 20: unrelated}.get(pid))
    read_fd, write_fd = os.pipe()
    try:
        os.set_blocking(read_fd, False)
        os.write(write_fd, b"20\n")
        os.close(write_fd)
        registry = HARNESS._ChildPidRegistry(read_fd, root)
        registry.drain()
        killed: list[int] = []
        monkeypatch.setattr(HARNESS.os, "kill", lambda pid, _sig: killed.append(pid))
        HARNESS._signal_registered(registry, HARNESS.signal.SIGKILL, root["pid"])
        assert registry.records == {}
        assert killed == []
    finally:
        with contextlib.suppress(OSError):
            os.close(write_fd)
        os.close(read_fd)


def test_pid_registry_records_live_double_fork_chain(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    root = {"pid": 10, "ppid": 1, "uid": 501, "start": "root"}
    child = {"pid": 11, "ppid": 10, "uid": 501, "start": "child"}
    grandchild = {"pid": 12, "ppid": 11, "uid": 501, "start": "grandchild"}
    records = {10: root, 11: child, 12: grandchild}
    monkeypatch.setattr(HARNESS, "_process_record", lambda pid: records.get(pid))
    read_fd, write_fd = os.pipe()
    try:
        os.set_blocking(read_fd, False)
        os.write(write_fd, b"12\n")
        os.close(write_fd)
        registry = HARNESS._ChildPidRegistry(read_fd, root)
        registry.drain()
        assert registry.records[12] == {
            "pid": 12,
            "uid": 501,
            "start": "grandchild",
            "chain": [
                {"pid": 12, "uid": 501, "start": "grandchild"},
                {"pid": 11, "uid": 501, "start": "child"},
                {"pid": 10, "uid": 501, "start": "root"},
            ],
        }
    finally:
        with contextlib.suppress(OSError):
            os.close(write_fd)
        os.close(read_fd)


def test_pid_registry_does_not_signal_reused_pid_with_mismatched_start(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    root = {"pid": 10, "ppid": 1, "uid": 501, "start": "root"}
    read_fd, write_fd = os.pipe()
    registry = HARNESS._ChildPidRegistry(read_fd, root)
    registry.records[20] = {
        "pid": 20,
        "uid": 501,
        "start": "original",
        "chain": [{"pid": 20, "uid": 501, "start": "original"}],
    }
    monkeypatch.setattr(
        HARNESS, "_process_record", lambda pid: {"pid": pid, "uid": 501, "start": "reused"}
    )
    killed: list[int] = []
    monkeypatch.setattr(HARNESS.os, "kill", lambda pid, _sig: killed.append(pid))
    try:
        HARNESS._signal_registered(registry, HARNESS.signal.SIGKILL, root["pid"])
        assert registry.live_pids() == set()
        assert killed == []
    finally:
        os.close(write_fd)
        os.close(registry.fd)


def test_process_lookup_is_a_single_trusted_pid_query(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def fake_run(argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[bytes]:
        calls.append(argv)
        return subprocess.CompletedProcess(
            argv, 0, stdout=b"123 1 501 Mon Aug 25 00:00:00 2026\n"
        )

    monkeypatch.setattr(HARNESS, "_trusted_tool", lambda _path: {})
    monkeypatch.setattr(HARNESS.subprocess, "run", fake_run)
    assert HARNESS._process_record(123) == {
        "pid": 123,
        "ppid": 1,
        "uid": 501,
        "start": "Mon Aug 25 00:00:00 2026",
    }
    assert calls == [["/bin/ps", "-o", "pid=", "-o", "ppid=", "-o", "uid=", "-o", "lstart=", "-p", "123"]]


@pytest.mark.parametrize("field", ["source", "production"])
def test_completion_rejects_source_or_production_swap(field: str, tmp_path: Path) -> None:
    inner = _formal_inner()
    completion = _completion(inner, tmp_path)
    completion[field] = {"swapped": True}
    with pytest.raises(HARNESS.R5Error, match="source or production"):
        HARNESS.assert_formal_acceptance(
            inner, completion, source_root=tmp_path, r4_artifact=tmp_path / "r4.json",
        )


@pytest.mark.parametrize("field", ["source_final", "production_final"])
def test_completion_identity_mappings_reject_safe_unknown_fields(
    field: str, tmp_path: Path
) -> None:
    inner = _formal_inner()
    completion = _completion(inner, tmp_path)
    value = dict(cast(dict[str, object], completion[field]))
    value["benign_extra"] = True
    completion[field] = value
    with pytest.raises(HARNESS.R5Error, match="closed"):
        HARNESS._sealed_completion(completion)


@pytest.mark.parametrize("action", ["fork-sleep-ignore", "fork-setsid-ignore", "fork-double-setsid-ignore"])
def test_supervisor_timeout_kills_test_child_and_cleans_owned_paths(
    action: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    production, source, output = (tmp_path / name for name in ("production", "source", "output"))
    production.mkdir()
    source.mkdir()
    identity = {"commit": "a" * 40, "tree_sha256": _id(1)}
    monkeypatch.setattr(HARNESS, "source_state", lambda *_args: identity)
    monkeypatch.setattr(HARNESS, "production_state", lambda *_args: {"state": "same"})
    monkeypatch.setattr(HARNESS, "_verify_r4", lambda *_args: {"artifact_id": _id(2)})
    with pytest.raises(HARNESS.R5Error, match="inner child failed"):
        HARNESS.run_supervised(
            production=production,
            source=source,
            source_commit="a" * 40,
            output=output,
            r4_artifact=tmp_path / "r4.json",
            watchdog_seconds=1,
            test_only=True,
            test_child_action=action,
        )
    assert not list(production.parent.glob("chronovisor-r5-supervisor-clone-*"))
    assert not list(output.glob(".chronovisor-r5-supervisor-*"))


def test_supervisor_sandbox_rejects_double_fork_before_registry_authentication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    production, source, output = (tmp_path / name for name in ("production", "source", "output"))
    production.mkdir()
    source.mkdir()
    identity = {"commit": "a" * 40, "tree_sha256": _id(1)}
    monkeypatch.setattr(HARNESS, "source_state", lambda *_args: identity)
    monkeypatch.setattr(HARNESS, "production_state", lambda *_args: {"state": "same"})
    monkeypatch.setattr(HARNESS, "_verify_r4", lambda *_args: {"artifact_id": _id(2)})
    with pytest.raises(HARNESS.R5Error, match="inner child failed"):
        HARNESS.run_supervised(
            production=production,
            source=source,
            source_commit="a" * 40,
            output=output,
            r4_artifact=tmp_path / "r4.json",
            watchdog_seconds=1,
            test_only=True,
            test_child_action="fork-double-setsid-ignore",
        )
    assert not list(production.parent.glob("chronovisor-r5-supervisor-clone-*"))


@pytest.mark.parametrize("action", ["alias-write-outside", "alias-network", "alias-process"])
def test_supervisor_kernel_sandbox_rejects_pre_sentinel_aliases(
    action: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    production, source, output = (tmp_path / name for name in ("production", "source", "output"))
    production.mkdir()
    source.mkdir()
    identity = {"commit": "a" * 40, "tree_sha256": _id(1)}
    monkeypatch.setattr(HARNESS, "source_state", lambda *_args: identity)
    monkeypatch.setattr(HARNESS, "production_state", lambda *_args: {"state": "same"})
    monkeypatch.setattr(HARNESS, "_verify_r4", lambda *_args: {"artifact_id": _id(2)})
    with pytest.raises(HARNESS.R5Error, match="inner child failed"):
        HARNESS.run_supervised(
            production=production,
            source=source,
            source_commit="a" * 40,
            output=output,
            r4_artifact=tmp_path / "r4.json",
            test_only=True,
            test_child_action=action,
        )
    assert not list(production.parent.glob("chronovisor-r5-supervisor-clone-*"))
    assert not list(output.glob(".chronovisor-r5-supervisor-*"))


@pytest.mark.skipif(sys.platform != "darwin", reason="requires macOS sandbox-exec")
def test_formal_macos_sandbox_blocks_double_fork_without_touching_unrelated_pid() -> None:
    """The formal profile denies the first fork, so setsid escape is impossible."""
    interpreter = HARNESS._trusted_interpreter()
    harness = HARNESS._trusted_harness_script()
    unrelated = subprocess.Popen([str(interpreter), "-c", "import time; time.sleep(10)"])
    try:
        result = subprocess.run(
            [
                "/usr/bin/sandbox-exec", "-p", HARNESS._formal_sandbox_policy(interpreter),
                str(interpreter), "-I", str(harness), "--test-child-action",
                "fork-double-setsid-ignore", "--test-only-child",
            ],
            check=False,
            capture_output=True,
            timeout=5,
        )
        assert result.returncode != 0
        assert unrelated.poll() is None
    finally:
        unrelated.terminate()
        unrelated.wait(timeout=5)


def test_supervisor_fast_test_child_cleans_but_cannot_issue_a_formal_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    production, source, output = (tmp_path / name for name in ("production", "source", "output"))
    production.mkdir()
    source.mkdir()
    identity = {"commit": "a" * 40, "tree_sha256": _id(1)}
    production_identity = {"raw": None, "runtime": None, "config": None}
    monkeypatch.setattr(HARNESS, "source_state", lambda *_args: identity)
    monkeypatch.setattr(HARNESS, "production_state", lambda *_args: production_identity)
    monkeypatch.setattr(HARNESS, "_verify_r4", lambda *_args: {"artifact_id": _id(2)})
    with pytest.raises(HARNESS.R5Error, match="inner child failed"):
        HARNESS.run_supervised(
            production=production,
            source=source,
            source_commit="a" * 40,
            output=output,
            r4_artifact=tmp_path / "r4.json",
            test_only=True,
            test_child_action="fast-valid",
        )
    assert not list(production.parent.glob("chronovisor-r5-supervisor-clone-*"))
    assert not list(output.glob(".chronovisor-r5-supervisor-*"))
    assert not list(output.glob("*.completion.json"))


def test_test_only_inner_cannot_satisfy_formal_acceptance(tmp_path: Path) -> None:
    inner = _reseal_inner(_formal_inner(), test_only=True)
    with pytest.raises(HARNESS.R5Error, match="execution receipt"):
        HARNESS.assert_formal_acceptance(
            inner, _completion(inner, tmp_path), source_root=tmp_path,
            r4_artifact=tmp_path / "r4.json",
        )


def test_supervisor_child_exception_cleans_owned_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    production, source, output = (tmp_path / name for name in ("production", "source", "output"))
    production.mkdir()
    source.mkdir()
    identity = {"commit": "a" * 40, "tree_sha256": _id(1)}
    monkeypatch.setattr(HARNESS, "source_state", lambda *_args: identity)
    monkeypatch.setattr(HARNESS, "production_state", lambda *_args: {"state": "same"})
    monkeypatch.setattr(HARNESS, "_verify_r4", lambda *_args: {"artifact_id": _id(2)})
    with pytest.raises(HARNESS.R5Error, match="return code"):
        HARNESS.run_supervised(
            production=production,
            source=source,
            source_commit="a" * 40,
            output=output,
            r4_artifact=tmp_path / "r4.json",
            test_only=True,
            test_child_action="raise",
        )
    assert not list(production.parent.glob("chronovisor-r5-supervisor-clone-*"))
    assert not list(output.glob(".chronovisor-r5-supervisor-*"))


def test_supervisor_setup_failure_cleans_every_owned_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    production, source, output = (tmp_path / name for name in ("production", "source", "output"))
    production.mkdir()
    source.mkdir()
    identity = {"commit": "a" * 40, "tree_sha256": _id(1)}
    monkeypatch.setattr(HARNESS, "source_state", lambda *_args: identity)
    monkeypatch.setattr(HARNESS, "production_state", lambda *_args: {"state": "same"})
    monkeypatch.setattr(HARNESS, "_verify_r4", lambda *_args: {"artifact_id": _id(2)})
    monkeypatch.setattr(HARNESS, "_write_private_json", lambda *_args: (_ for _ in ()).throw(HARNESS.R5Error("setup fail")))
    with pytest.raises(HARNESS.R5Error, match="setup fail"):
        HARNESS.run_supervised(
            production=production,
            source=source,
            source_commit="a" * 40,
            output=output,
            r4_artifact=tmp_path / "r4.json",
            test_only=True,
            test_child_action="exit-0",
        )
    assert not list(production.parent.glob("chronovisor-r5-supervisor-clone-*"))
    assert not list(output.glob(".chronovisor-r5-supervisor-*"))


def test_supervisor_pre_registry_process_lookup_failure_reaps_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    production, source, output = (tmp_path / name for name in ("production", "source", "output"))
    production.mkdir()
    source.mkdir()
    identity = {"commit": "a" * 40, "tree_sha256": _id(1)}
    monkeypatch.setattr(HARNESS, "source_state", lambda *_args: identity)
    monkeypatch.setattr(HARNESS, "production_state", lambda *_args: {"state": "same"})
    monkeypatch.setattr(HARNESS, "_verify_r4", lambda *_args: {"artifact_id": _id(2)})
    original_lookup = HARNESS._process_record
    calls = 0

    def fail_root_lookup(pid: int) -> dict[str, object] | None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise HARNESS.R5Error("injected process lookup failure")
        return cast(dict[str, object] | None, original_lookup(pid))

    reaped: list[int] = []
    original_cleanup = HARNESS._terminate_unregistered_child

    def track_cleanup(process: subprocess.Popen[bytes], pgid: int) -> int:
        result = original_cleanup(process, pgid)
        reaped.append(process.pid)
        return int(cast(int, result))

    monkeypatch.setattr(HARNESS, "_process_record", fail_root_lookup)
    monkeypatch.setattr(HARNESS, "_terminate_unregistered_child", track_cleanup)
    with pytest.raises(HARNESS.R5Error, match="injected process lookup failure"):
        HARNESS.run_supervised(
            production=production,
            source=source,
            source_commit="a" * 40,
            output=output,
            r4_artifact=tmp_path / "r4.json",
            test_only=True,
            test_child_action="fork-sleep-ignore",
        )
    assert reaped
    assert not list(production.parent.glob("chronovisor-r5-supervisor-clone-*"))
    assert not list(output.glob(".chronovisor-r5-supervisor-*"))


def test_supervisor_delayed_none_identity_before_registry_leaves_no_descendants(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    production, source, output = (tmp_path / name for name in ("production", "source", "output"))
    production.mkdir()
    source.mkdir()
    identity = {"commit": "a" * 40, "tree_sha256": _id(1)}
    monkeypatch.setattr(HARNESS, "source_state", lambda *_args: identity)
    monkeypatch.setattr(HARNESS, "production_state", lambda *_args: {"state": "same"})
    monkeypatch.setattr(HARNESS, "_verify_r4", lambda *_args: {"artifact_id": _id(2)})
    def delayed_none(_pid: int) -> None:
        time.sleep(0.02)
        return None

    monkeypatch.setattr(HARNESS, "_process_record", delayed_none)
    cleaned: list[int] = []
    original = HARNESS._terminate_unregistered_child

    def observe(process: subprocess.Popen[bytes], pgid: int) -> int:
        result = original(process, pgid)
        cleaned.append(process.pid)
        return int(cast(int, result))

    monkeypatch.setattr(HARNESS, "_terminate_unregistered_child", observe)
    with pytest.raises(HARNESS.R5Error, match="identity failed"):
        HARNESS.run_supervised(
            production=production,
            source=source,
            source_commit="a" * 40,
            output=output,
            r4_artifact=tmp_path / "r4.json",
            test_only=True,
            test_child_action="fork-double-setsid-ignore",
        )
    assert cleaned
    assert not list(production.parent.glob("chronovisor-r5-supervisor-clone-*"))
    assert not list(output.glob(".chronovisor-r5-supervisor-*"))
