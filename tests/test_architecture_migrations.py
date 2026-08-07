from __future__ import annotations

import ast
import copy
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).parents[1]
PLAN = (
    ROOT
    / "docs"
    / "refactoring"
    / "architecture-migrations"
    / "plans"
    / "P2-classification-fixture-contract.json"
)


def _load_script() -> ModuleType:
    path = ROOT / "scripts" / "architecture_migrations.py"
    spec = importlib.util.spec_from_file_location("architecture_migrations", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def migrations() -> ModuleType:
    return _load_script()


def _resign(migrations: ModuleType, payload: dict[str, Any], field: str) -> None:
    payload[field] = migrations._seal(payload, field)


def _write_plan(
    migrations: ModuleType,
    path: Path,
    payload: dict[str, Any],
) -> None:
    migrations.write_canonical_json(path, payload)


def _valid_receipt(migrations: ModuleType, plan: dict[str, Any]) -> dict[str, Any]:
    receipt = {
        "schema": migrations.RECEIPT_SCHEMA,
        "migration_id": migrations.MIGRATION_ID,
        "plan_sha256": plan["plan_sha256"],
        "h1_commit": "1" * 40,
        "h2_artifacts": {
            "baseline": {
                "path": migrations.BASELINE_PATH.as_posix(),
                "sha256": "2" * 64,
            },
            "ledger": {
                "path": migrations.LEDGER_PATH.as_posix(),
                "sha256": "3" * 64,
            },
        },
        "active_counts": migrations.EXPECTED_H2_ACTIVE_COUNTS,
        "retired_counts": migrations.EXPECTED_H2_RETIRED_COUNTS,
        "retired_exception_ids": [migrations.PRIVATE_EXCEPTION_ID],
        "retired_site_ids": list(migrations.MIGRATED_SITE_IDS),
        "p3_retained_edge_ids": [migrations.CLASSIFICATION_LAB_EDGE_ID],
        "p3_retained_site_ids": [migrations.PROVIDER_SITE_ID],
    }
    _resign(migrations, receipt, "receipt_sha256")
    return receipt


def test_p2_plan_validates_from_sealed_git_objects(migrations: ModuleType) -> None:
    plan = migrations.load_plan(PLAN)
    report = migrations.validate_plan(ROOT, plan)

    assert report == {
        "migration_id": migrations.MIGRATION_ID,
        "h0_parent_commit": "602ab1efd46b3c74447887cf430cc77962fec7bd",
        "site_count": 5,
        "state": "valid-h0-plan",
    }
    assert not (ROOT / migrations.RECEIPT_PATH).exists()


def test_p2_plan_separates_h2_retirement_from_p3_retention(
    migrations: ModuleType,
) -> None:
    plan = migrations.load_plan(PLAN)
    policy = plan["retirement_policy"]

    assert policy["h0_ledger_removal_campaign"] == "P3"
    assert policy["h2_retire_exception_ids"] == [migrations.PRIVATE_EXCEPTION_ID]
    assert policy["h2_retire_site_ids"] == list(migrations.MIGRATED_SITE_IDS)
    assert policy["p3_retain_edge_ids"] == [migrations.CLASSIFICATION_LAB_EDGE_ID]
    assert policy["p3_retain_site_ids"] == [migrations.PROVIDER_SITE_ID]
    assert set(policy["h2_retire_site_ids"]).isdisjoint(
        policy["p3_retain_site_ids"]
    )


@pytest.mark.parametrize(
    "mutation, expected",
    [
        ("tamper", "plan_sha256 mismatch"),
        ("unknown", "unexpected"),
        ("missing", "campaign"),
        ("duplicate", "missing, duplicated, or reordered"),
        ("git-object", "byte digest mismatch"),
    ],
)
def test_plan_rejects_tamper_unknown_missing_duplicate_and_git_object_drift(
    migrations: ModuleType,
    tmp_path: Path,
    mutation: str,
    expected: str,
) -> None:
    payload = json.loads(PLAN.read_text(encoding="utf-8"))
    if mutation == "tamper":
        payload["campaign"] = "P3"
    elif mutation == "unknown":
        payload["unexpected"] = True
        _resign(migrations, payload, "plan_sha256")
    elif mutation == "missing":
        del payload["campaign"]
        _resign(migrations, payload, "plan_sha256")
    elif mutation == "duplicate":
        payload["sites"].append(copy.deepcopy(payload["sites"][0]))
        _resign(migrations, payload, "plan_sha256")
    else:
        payload["inputs"]["ledger"]["sha256"] = "f" * 64
        _resign(migrations, payload, "plan_sha256")
    path = tmp_path / f"{mutation}.json"
    _write_plan(migrations, path, payload)

    with pytest.raises(migrations.MigrationValidationError, match=expected):
        plan = migrations.load_plan(path)
        migrations.validate_plan(ROOT, plan)


def test_plan_rejects_noncanonical_json(
    migrations: ModuleType, tmp_path: Path
) -> None:
    payload = json.loads(PLAN.read_text(encoding="utf-8"))
    path = tmp_path / "noncanonical.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(migrations.MigrationValidationError, match="not canonical JSON"):
        migrations.load_plan(path)


def test_receipt_schema_is_strict_canonical_and_self_hashed(
    migrations: ModuleType, tmp_path: Path
) -> None:
    plan = migrations.load_plan(PLAN)
    receipt = _valid_receipt(migrations, plan)
    path = tmp_path / "receipt.json"
    migrations.write_canonical_json(path, receipt)
    assert migrations.load_receipt(path) == receipt

    receipt["unexpected"] = True
    _resign(migrations, receipt, "receipt_sha256")
    migrations.write_canonical_json(path, receipt)
    with pytest.raises(migrations.MigrationValidationError, match="unknown"):
        migrations.load_receipt(path)


def test_production_contract_has_no_lab_import() -> None:
    path = ROOT / "src" / "chronovisor" / "classification" / (
        "classification_fixture_contract.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    assert not any(
        module == "chronovisor.lab" or module.startswith("chronovisor.lab.")
        for module in imported
    )
