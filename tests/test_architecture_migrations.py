from __future__ import annotations

import ast
import copy
import importlib.util
import json
import os
import shutil
import subprocess
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


def _git(repo: Path, *args: str, input_bytes: bytes | None = None) -> bytes:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        input=input_bytes,
        check=True,
        capture_output=True,
    ).stdout


def _git_text(repo: Path, *args: str) -> str:
    return _git(repo, *args).decode().strip()


def _commit(repo: Path, message: str) -> str:
    _git(
        repo,
        "-c",
        "user.name=Migration Test",
        "-c",
        "user.email=migration@example.invalid",
        "commit",
        "--quiet",
        "-m",
        message,
    )
    return _git_text(repo, "rev-parse", "HEAD")


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


def _clone_with_evidence_parent(
    tmp_path: Path, migrations: ModuleType
) -> tuple[Path, dict[str, Any], str]:
    repo = tmp_path / "repo"
    subprocess.run(
        ["git", "clone", "--quiet", "--shared", str(ROOT), str(repo)],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    _git(repo, "checkout", "--quiet", "--detach", migrations.GATE_HYGIENE_COMMIT)
    for relative in migrations.EVIDENCE_HARDENING_PATHS:
        shutil.copyfile(ROOT / relative, repo / relative)
    _git(repo, "add", "--", *migrations.EVIDENCE_HARDENING_PATHS)
    evidence_parent = _commit(repo, "refactor: harden P2 migration evidence")
    plan = migrations.load_plan(repo / migrations.PLAN_PATH)
    return repo, plan, evidence_parent


def _write_h1(
    repo: Path,
    migrations: ModuleType,
    plan: dict[str, Any],
    *,
    mutation: bytes = b"",
) -> str:
    expected = migrations._expected_h1_sources(repo, plan)
    for path, raw in expected.items():
        if path == migrations.H1_PATHS[3]:
            raw += mutation
        (repo / path).write_bytes(raw)
    _git(repo, "add", "--", *migrations.H1_PATHS)
    return _commit(repo, "refactor: migrate P2 classification callsites")


def _stage_h2(repo: Path, migrations: ModuleType, h1: str) -> tuple[bytes, bytes]:
    h1_baseline = migrations._git_file(repo, h1, migrations.BASELINE_PATH.as_posix())
    h1_ledger = migrations._git_file(repo, h1, migrations.LEDGER_PATH.as_posix())
    baseline, ledger = migrations._expected_h2_payloads(h1_baseline, h1_ledger)
    (repo / migrations.BASELINE_PATH).write_bytes(baseline)
    (repo / migrations.LEDGER_PATH).write_bytes(ledger)
    _git(
        repo,
        "add",
        "--",
        migrations.BASELINE_PATH.as_posix(),
        migrations.LEDGER_PATH.as_posix(),
    )
    return baseline, ledger


def test_p2_plan_validates_fixed_h0_and_gate_history(
    migrations: ModuleType,
) -> None:
    plan = migrations.load_plan(PLAN)
    report = migrations.validate_plan(ROOT, plan)

    assert report == {
        "migration_id": migrations.MIGRATION_ID,
        "h0_parent_commit": migrations.H0_PARENT_COMMIT,
        "h0_seed_commit": migrations.H0_SEED_COMMIT,
        "gate_hygiene_commit": migrations.GATE_HYGIENE_COMMIT,
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
    assert set(policy["h2_retire_site_ids"]).isdisjoint(policy["p3_retain_site_ids"])


@pytest.mark.parametrize(
    "mutation, expected",
    [
        ("tamper", "plan_sha256 mismatch"),
        ("unknown", "unexpected"),
        ("missing", "campaign"),
        ("duplicate", "missing, duplicated, or reordered"),
        ("git-object", "byte digest mismatch"),
        ("history", "history chain drift"),
        ("snippet", "source digest drift"),
    ],
)
def test_plan_rejects_semantic_and_git_object_drift(
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
    elif mutation == "git-object":
        payload["inputs"]["ledger"]["sha256"] = "f" * 64
        _resign(migrations, payload, "plan_sha256")
    elif mutation == "history":
        payload["history"]["gate_hygiene_commit"] = "0" * 40
        _resign(migrations, payload, "plan_sha256")
    else:
        payload["sites"][0]["source_transform"]["new_snippet"] += "# drift\n"
        _resign(migrations, payload, "plan_sha256")
    path = tmp_path / f"{mutation}.json"
    _write_plan(migrations, path, payload)

    with pytest.raises(migrations.MigrationValidationError, match=expected):
        plan = migrations.load_plan(path)
        migrations.validate_plan(ROOT, plan)


@pytest.mark.parametrize("mutation", ["format", "duplicate", "nan"])
def test_plan_rejects_noncanonical_duplicate_and_nonfinite_json(
    migrations: ModuleType, tmp_path: Path, mutation: str
) -> None:
    raw = PLAN.read_bytes()
    if mutation == "format":
        raw = json.dumps(json.loads(raw)).encode()
    elif mutation == "duplicate":
        raw = raw.replace(
            b'{\n  "campaign"', b'{\n  "campaign": "P2",\n  "campaign"', 1
        )
    else:
        raw = raw.replace(b'"campaign": "P2"', b'"nonfinite": NaN', 1)
    path = tmp_path / f"{mutation}.json"
    path.write_bytes(raw)

    with pytest.raises(migrations.MigrationValidationError):
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


def test_external_revisions_require_exact_lowercase_full_sha(
    migrations: ModuleType,
) -> None:
    invalid = [
        migrations.H0_SEED_COMMIT[:12],
        "HEAD",
        "main",
        migrations.H0_SEED_COMMIT.upper(),
        "f" * 40,
    ]
    for revision in invalid:
        with pytest.raises(migrations.MigrationValidationError):
            migrations._commit(ROOT, revision)


def test_git_reads_ignore_replace_objects_and_git_environment(
    migrations: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, plan, _evidence_parent = _clone_with_evidence_parent(tmp_path, migrations)
    _git(
        repo,
        "replace",
        migrations.H0_SEED_COMMIT,
        migrations.GATE_HYGIENE_COMMIT,
    )
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    _git(foreign, "init", "--quiet")
    monkeypatch.setenv("GIT_DIR", str(foreign / ".git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(foreign))
    monkeypatch.setenv("GIT_OBJECT_DIRECTORY", str(foreign / ".git" / "objects"))
    monkeypatch.setenv("GIT_ALTERNATE_OBJECT_DIRECTORIES", str(foreign))
    monkeypatch.setenv("GIT_INDEX_FILE", str(foreign / "index"))
    monkeypatch.setenv("GIT_COMMON_DIR", str(foreign / ".git"))
    monkeypatch.setenv("GIT_NAMESPACE", "attacker")

    assert migrations.validate_plan(repo, plan)["state"] == "valid-h0-plan"


def test_foreign_repository_without_fixed_objects_is_rejected(
    migrations: ModuleType, tmp_path: Path
) -> None:
    repo = tmp_path / "foreign"
    repo.mkdir()
    _git(repo, "init", "--quiet")
    (repo / "only.txt").write_text("foreign\n", encoding="utf-8")
    _git(repo, "add", "only.txt")
    _commit(repo, "foreign")

    with pytest.raises(migrations.MigrationValidationError):
        migrations.validate_plan(repo, migrations.load_plan(PLAN))


def test_tree_entries_reject_symlink_executable_gitlink_and_missing(
    migrations: ModuleType, tmp_path: Path
) -> None:
    repo = tmp_path / "modes"
    repo.mkdir()
    _git(repo, "init", "--quiet")
    (repo / "regular").write_text("regular\n", encoding="utf-8")
    (repo / "executable").write_text("executable\n", encoding="utf-8")
    (repo / "executable").chmod(0o755)
    os.symlink("regular", repo / "symlink")
    _git(repo, "add", "regular", "executable", "symlink")
    first = _commit(repo, "mode fixtures")
    _git(repo, "update-index", "--add", "--cacheinfo", f"160000,{first},gitlink")
    gitlink_commit = _commit(repo, "gitlink fixture")

    assert migrations._git_file(repo, gitlink_commit, "regular") == b"regular\n"
    for path in ("executable", "symlink", "gitlink", "missing"):
        with pytest.raises(migrations.MigrationValidationError):
            migrations._git_file(repo, gitlink_commit, path)


def test_history_h1_h2_and_receipt_verify_end_to_end(
    migrations: ModuleType, tmp_path: Path
) -> None:
    repo, plan, evidence_parent = _clone_with_evidence_parent(tmp_path, migrations)
    assert migrations.validate_history(repo, plan, evidence_parent) == {
        "migration_id": migrations.MIGRATION_ID,
        "evidence_parent_commit": evidence_parent,
        "state": "valid-evidence-history",
    }
    h1 = _write_h1(repo, migrations, plan)
    report = migrations.validate_h1(repo, plan, h1)
    assert report["evidence_parent_commit"] == evidence_parent
    assert report["h1_commit"] == h1

    baseline, ledger = _stage_h2(repo, migrations, h1)
    receipt = migrations.build_receipt(repo, plan, h1)
    assert (
        receipt["h2_artifacts"]["baseline"]["sha256"]
        == migrations.hashlib.sha256(baseline).hexdigest()
    )
    assert (
        receipt["h2_artifacts"]["ledger"]["sha256"]
        == migrations.hashlib.sha256(ledger).hexdigest()
    )
    migrations.write_canonical_json(repo / migrations.RECEIPT_PATH, receipt)
    _git(repo, "add", "--", migrations.RECEIPT_PATH.as_posix())
    h2 = _commit(repo, "refactor: retire P2 architecture exceptions")

    verified = migrations.verify_receipt(repo, plan, receipt, h2)
    assert verified["h1_commit"] == h1
    assert verified["h2_commit"] == h2
    assert verified["state"] == "valid-h2-receipt"


def test_evidence_history_rejects_extra_c_path_and_plan_byte_drift(
    migrations: ModuleType, tmp_path: Path
) -> None:
    repo, plan, evidence_parent = _clone_with_evidence_parent(tmp_path, migrations)
    _git(repo, "checkout", "--quiet", "--detach", migrations.GATE_HYGIENE_COMMIT)
    for relative in migrations.EVIDENCE_HARDENING_PATHS:
        shutil.copyfile(ROOT / relative, repo / relative)
    (repo / "pyproject.toml").write_bytes(
        (repo / "pyproject.toml").read_bytes() + b"\n# unrelated\n"
    )
    _git(repo, "add", "--", *migrations.EVIDENCE_HARDENING_PATHS, "pyproject.toml")
    extra = _commit(repo, "invalid extra C path")
    with pytest.raises(migrations.MigrationValidationError, match="scope mismatch"):
        migrations.validate_history(repo, plan, extra)

    _git(repo, "checkout", "--quiet", "--detach", evidence_parent)
    payload = json.loads((repo / migrations.PLAN_PATH).read_bytes())
    payload["campaign"] = "tampered-but-self-consistent"
    _resign(migrations, payload, "plan_sha256")
    migrations.write_canonical_json(repo / migrations.PLAN_PATH, payload)
    _git(repo, "add", "--", migrations.PLAN_PATH.as_posix())
    drift = _commit(repo, "plan byte drift")
    with pytest.raises(migrations.MigrationValidationError):
        migrations.validate_history(repo, plan, drift)


def test_h1_rejects_logic_import_symbol_and_alias_drift(
    migrations: ModuleType, tmp_path: Path
) -> None:
    repo, plan, evidence_parent = _clone_with_evidence_parent(tmp_path, migrations)
    expected = migrations._expected_h1_sources(repo, plan)
    assert (
        b"chronovisor.lab.classification_library_evidence"
        in expected[migrations.H1_PATHS[3]]
    )
    mutations = [
        b"\nUNRELATED_LOGIC = True\n",
        b"\nimport chronovisor.lab.extra\n",
        b"\ndef nested():\n    import chronovisor.lab.extra\n",
        b"\nimport chronovisor.lab\n",
        b"\n__import__('chronovisor.lab.extra')\n",
        b"\nfrom chronovisor.classification.classification_fixture_contract import sha256_file\n",
        b"\nfrom chronovisor.classification.classification_fixture_contract import inference_dto as dto\n",
    ]
    for mutation in mutations:
        _git(repo, "checkout", "--quiet", "--detach", "--force", evidence_parent)
        h1 = _write_h1(repo, migrations, plan, mutation=mutation)
        with pytest.raises(migrations.MigrationValidationError, match="source bytes"):
            migrations.validate_h1(repo, plan, h1)


def test_h1_rejects_non_100644_source_mode(
    migrations: ModuleType, tmp_path: Path
) -> None:
    repo, plan, _evidence_parent = _clone_with_evidence_parent(tmp_path, migrations)
    expected = migrations._expected_h1_sources(repo, plan)
    for path, raw in expected.items():
        (repo / path).write_bytes(raw)
    (repo / migrations.H1_PATHS[0]).chmod(0o755)
    _git(repo, "add", "--", *migrations.H1_PATHS)
    h1 = _commit(repo, "invalid executable H1 source")

    with pytest.raises(migrations.MigrationValidationError, match="100644"):
        migrations.validate_h1(repo, plan, h1)


def test_h2_exact_transform_rejects_semantic_and_encoding_attacks(
    migrations: ModuleType, tmp_path: Path
) -> None:
    repo, plan, _evidence_parent = _clone_with_evidence_parent(tmp_path, migrations)
    h1 = _write_h1(repo, migrations, plan)
    h1_baseline = migrations._git_file(repo, h1, migrations.BASELINE_PATH.as_posix())
    h1_ledger = migrations._git_file(repo, h1, migrations.LEDGER_PATH.as_posix())
    baseline_raw, ledger_raw = migrations._expected_h2_payloads(h1_baseline, h1_ledger)
    migrations._validate_h2_payloads(h1_baseline, h1_ledger, baseline_raw, ledger_raw)
    baseline = json.loads(baseline_raw)
    ledger = json.loads(ledger_raw)
    edge = next(
        row
        for row in ledger["exceptions"]
        if row["semantic_id"] == migrations.CLASSIFICATION_LAB_EDGE_ID
    )
    assert [site["semantic_id"] for site in edge["sites"]] == [
        migrations.PROVIDER_SITE_ID
    ]

    mutations: list[tuple[dict[str, Any], dict[str, Any]]] = []
    overlap = copy.deepcopy(baseline)
    overlap["cross_domain_site_semantic_ids"]["retired"].append(
        migrations.PROVIDER_SITE_ID
    )
    overlap["cross_domain_site_semantic_ids"]["retired"].sort()
    mutations.append((overlap, ledger))
    unknown = copy.deepcopy(baseline)
    unknown["exception_semantic_ids"]["active"].append("arch:" + "f" * 64)
    unknown["exception_semantic_ids"]["active"].sort()
    mutations.append((unknown, ledger))
    missing = copy.deepcopy(baseline)
    missing["exception_semantic_ids"]["active"].pop()
    mutations.append((missing, ledger))
    source_drift = copy.deepcopy(baseline)
    source_drift["source_baseline_head"] = "0" * 40
    mutations.append((source_drift, ledger))
    schema_drift = copy.deepcopy(baseline)
    schema_drift["schema_version"] += 1
    mutations.append((schema_drift, ledger))
    count_drift = copy.deepcopy(baseline)
    count_drift["counts"]["active"]["exceptions"] += 1
    mutations.append((count_drift, ledger))
    unrelated = copy.deepcopy(ledger)
    unrelated["exceptions"][0]["rationale"] = "unrelated edit"
    mutations.append((baseline, unrelated))
    missing_semantic_id = copy.deepcopy(ledger)
    del missing_semantic_id["exceptions"][0]["semantic_id"]
    mutations.append((baseline, missing_semantic_id))
    non_list_sites = copy.deepcopy(ledger)
    next(
        row
        for row in non_list_sites["exceptions"]
        if row["semantic_id"] == migrations.CLASSIFICATION_LAB_EDGE_ID
    )["sites"] = {}
    mutations.append((baseline, non_list_sites))

    for attacked_baseline, attacked_ledger in mutations:
        with pytest.raises(migrations.MigrationValidationError):
            migrations._validate_h2_payloads(
                h1_baseline,
                h1_ledger,
                migrations._document_json_bytes(attacked_baseline),
                migrations._document_json_bytes(attacked_ledger),
            )

    duplicate = baseline_raw.replace(
        b'{\n  "schema_version": 1,',
        b'{\n  "schema_version": 1,\n  "schema_version": 1,',
        1,
    )
    nonfinite = baseline_raw.replace(b'"exceptions": 161', b'"exceptions": NaN', 1)
    reordered = (
        json.dumps(baseline, ensure_ascii=False, sort_keys=True, indent=2).encode()
        + b"\n"
    )
    for attacked in (duplicate, nonfinite, reordered):
        with pytest.raises(migrations.MigrationValidationError):
            migrations._validate_h2_payloads(
                h1_baseline, h1_ledger, attacked, ledger_raw
            )


def test_build_receipt_requires_clean_exact_two_path_index(
    migrations: ModuleType, tmp_path: Path
) -> None:
    repo, plan, _evidence_parent = _clone_with_evidence_parent(tmp_path, migrations)
    h1 = _write_h1(repo, migrations, plan)
    baseline, _ledger = _stage_h2(repo, migrations, h1)
    assert migrations.build_receipt(repo, plan, h1)["h1_commit"] == h1

    (repo / migrations.BASELINE_PATH).write_bytes(baseline + b" ")
    with pytest.raises(migrations.MigrationValidationError, match="unstaged"):
        migrations.build_receipt(repo, plan, h1)
    (repo / migrations.BASELINE_PATH).write_bytes(baseline)

    pyproject = repo / "pyproject.toml"
    pyproject.write_bytes(pyproject.read_bytes() + b"\n# staged attack\n")
    _git(repo, "add", "pyproject.toml")
    with pytest.raises(migrations.MigrationValidationError, match="scope mismatch"):
        migrations.build_receipt(repo, plan, h1)
    for invalid in ("HEAD", h1[:12], h1.upper()):
        with pytest.raises(migrations.MigrationValidationError):
            migrations.build_receipt(repo, plan, invalid)


def test_receipt_verification_rejects_non_additive_receipt_history(
    migrations: ModuleType, tmp_path: Path
) -> None:
    repo, plan, _evidence_parent = _clone_with_evidence_parent(tmp_path, migrations)
    h1 = _write_h1(repo, migrations, plan)
    _stage_h2(repo, migrations, h1)
    receipt = migrations.build_receipt(repo, plan, h1)
    migrations.write_canonical_json(repo / migrations.RECEIPT_PATH, receipt)
    _git(repo, "add", "--", migrations.RECEIPT_PATH.as_posix())
    _commit(repo, "valid H2")
    payload = json.loads((repo / migrations.RECEIPT_PATH).read_bytes())
    payload["receipt_sha256"] = "0" * 64
    migrations.write_canonical_json(repo / migrations.RECEIPT_PATH, payload)
    _git(repo, "add", "--", migrations.RECEIPT_PATH.as_posix())
    tip = _commit(repo, "non-additive receipt history")

    with pytest.raises(migrations.MigrationValidationError, match="non-additive"):
        migrations.verify_receipt(repo, plan, receipt, tip)


def test_production_contract_has_no_lab_import() -> None:
    path = (
        ROOT
        / "src"
        / "chronovisor"
        / "classification"
        / ("classification_fixture_contract.py")
    )
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    assert not any(
        module == "chronovisor.lab" or module.startswith("chronovisor.lab.")
        for module in imported
    )
