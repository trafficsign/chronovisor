#!/usr/bin/env python3
"""Validate sealed Campaign P architecture migrations at Git object boundaries."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

PLAN_SCHEMA = "chronovisor.architecture-migration-plan.v1"
RECEIPT_SCHEMA = "chronovisor.architecture-migration-receipt.v1"
MIGRATION_ID = "P2-classification-fixture-contract"
FROZEN_SOURCE_HEAD = "d404a6b20d00e3bcd1d4cdb89edfa5a718c51833"
H0_PARENT_COMMIT = "602ab1efd46b3c74447887cf430cc77962fec7bd"
PLAN_PATH = Path(
    "docs/refactoring/architecture-migrations/plans/"
    "P2-classification-fixture-contract.json"
)
RECEIPT_PATH = Path(
    "docs/refactoring/architecture-migrations/receipts/"
    "P2-classification-fixture-contract.json"
)
LEDGER_PATH = Path("docs/refactoring/architecture-exceptions.json")
BASELINE_PATH = Path("docs/refactoring/architecture-exception-baseline.json")
CONTRACT_PATH = Path(
    "src/chronovisor/classification/classification_fixture_contract.py"
)
CONTRACT_MODULE = (
    "chronovisor.classification.classification_fixture_contract"
)
PRIVATE_EXCEPTION_ID = (
    "arch:97b784f974ebdf78ae0226731f9b421e381ac04bad98dd104435367c618f52e9"
)
PROVIDER_SITE_ID = (
    "arch:7923c8117584e014f1a6d93283fb7ce9eb012f2cdb7e6228171c5fffb58aecc1"
)
CLASSIFICATION_LAB_EDGE_ID = (
    "arch:0f37f016df9c2c328a0edc59fd3b3c4b8039921bde3fdaaede27d59f34be9f60"
)
MIGRATED_SITE_IDS = (
    "arch:0d799ce1e29887c64caf095e2640a148aa9e90326aaef4c45a476ae13c33e85b",
    "arch:7d200838738a191b653ce5829735d820b786746b2a6329bf9cc98610ff57dd32",
    "arch:a36a5c65819a0aab93f8c971dee29537ef65da12069950b9fdab4b270bb9c9d7",
    "arch:c7056bd3c53d85c0dfb27edae2aead5bac51065e169b7b87ef3f21b4363d3ca2",
    "arch:e821d535f1969c514f6f49a2338ec36e382d3a6b82040c593bf28f580d86c97d",
)
H1_PATHS = (
    "src/chronovisor/classification/classification_bundle.py",
    "src/chronovisor/classification/classification_evidence_judgment.py",
    "src/chronovisor/classification/classification_library_sources.py",
    "src/chronovisor/classification/classification_resolver.py",
    "src/chronovisor/classification/classification_retention.py",
)
H0_PATHS = tuple(
    sorted(
        {
            PLAN_PATH.as_posix(),
            "scripts/architecture_migrations.py",
            CONTRACT_PATH.as_posix(),
            "tests/test_architecture_migrations.py",
            "tests/test_classification_fixture_contract.py",
        }
    )
)
H2_PATHS = tuple(
    sorted(
        {
            BASELINE_PATH.as_posix(),
            LEDGER_PATH.as_posix(),
            RECEIPT_PATH.as_posix(),
        }
    )
)
EXPECTED_H0_ACTIVE_COUNTS = {
    "exceptions": 162,
    "cross_domain_sites": 1267,
    "production_to_lab_edges": 5,
    "production_to_lab_static_sites": 20,
    "production_to_lab_dynamic_sites": 1,
    "compatibility_contracts": 289,
}
EXPECTED_H2_ACTIVE_COUNTS = {
    "exceptions": 161,
    "cross_domain_sites": 1262,
    "production_to_lab_edges": 5,
    "production_to_lab_static_sites": 15,
    "production_to_lab_dynamic_sites": 1,
    "compatibility_contracts": 289,
}
EXPECTED_H2_RETIRED_COUNTS = {
    "exception_semantic_ids": 1,
    "cross_domain_site_semantic_ids": 5,
    "production_to_lab_edge_semantic_ids": 0,
    "production_to_lab_static_site_semantic_ids": 5,
    "production_to_lab_dynamic_site_semantic_ids": 0,
    "compatibility_semantic_ids": 0,
}
EXPECTED_NEW_IMPORTS = {
    MIGRATED_SITE_IDS[0]: ("inference_dto",),
    MIGRATED_SITE_IDS[1]: (
        "DISABLED_BASELINE_SCHEMA",
        "sha256_bytes",
        "sha256_file",
    ),
    MIGRATED_SITE_IDS[2]: ("sha256_bytes", "sha256_file"),
    MIGRATED_SITE_IDS[3]: ("sha256_bytes", "sha256_file"),
    MIGRATED_SITE_IDS[4]: ("sha256_bytes", "write_jsonl"),
}
EXPECTED_LOCAL_NAMES = {
    MIGRATED_SITE_IDS[0]: {"inference_dto": "inference_dto"},
    MIGRATED_SITE_IDS[1]: {
        "DISABLED_BASELINE_SCHEMA": "DISABLED_BASELINE_SCHEMA",
        "sha256_bytes": "sha256_bytes",
        "sha256_file": "sha256_file",
    },
    MIGRATED_SITE_IDS[2]: {
        "sha256_bytes": "sha256_bytes",
        "sha256_file": "sha256_file",
    },
    MIGRATED_SITE_IDS[3]: {
        "sha256_bytes": "sha256_bytes",
        "sha256_file": "sha256_file",
    },
    MIGRATED_SITE_IDS[4]: {
        "sha256_bytes": "sha256_bytes",
        "write_jsonl": "_write_jsonl",
    },
}
_ARCH_ID_RE = re.compile(r"arch:[0-9a-f]{64}\Z")
_HEX_SHA_RE = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
_BLOB_RE = re.compile(r"[0-9a-f]{40,64}\Z")


class MigrationValidationError(ValueError):
    """Raised when a migration artifact or Git boundary fails closed."""


def _canonical_json_bytes(payload: Any) -> bytes:
    try:
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise MigrationValidationError(f"payload is not canonical JSON: {exc}") from exc
    return (encoded + "\n").encode("utf-8")


def _seal(payload: Mapping[str, Any], field: str) -> str:
    unsigned = {key: value for key, value in payload.items() if key != field}
    compact = json.dumps(
        unsigned,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(compact).hexdigest()


def _require_keys(
    payload: Mapping[str, Any], expected: set[str], *, context: str
) -> None:
    actual = set(payload)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise MigrationValidationError(
            f"{context} keys mismatch: missing={missing}, unknown={unknown}"
        )


def _object(value: Any, *, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise MigrationValidationError(f"{context} must be an object")
    return value


def _objects(value: Any, *, context: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise MigrationValidationError(f"{context} must be an array of objects")
    return value


def _strings(value: Any, *, context: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise MigrationValidationError(f"{context} must be an array of strings")
    if len(value) != len(set(value)):
        raise MigrationValidationError(f"{context} contains duplicates")
    return value


def _relative_path(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise MigrationValidationError(f"{context} must be a non-empty path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise MigrationValidationError(f"{context} must be canonical repo-relative path")
    return value


def _load_canonical(path: Path, *, seal_field: str) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise MigrationValidationError(f"cannot load {path}: {exc}") from exc
    artifact = _object(payload, context=str(path))
    if raw != _canonical_json_bytes(artifact):
        raise MigrationValidationError(f"{path} is not canonical JSON")
    recorded = artifact.get(seal_field)
    if not isinstance(recorded, str) or not _HEX_SHA_RE.fullmatch(recorded):
        raise MigrationValidationError(f"{path} has invalid {seal_field}")
    expected = _seal(artifact, seal_field)
    if recorded != expected:
        raise MigrationValidationError(f"{path} {seal_field} mismatch")
    return artifact


def _git(root: Path, *args: str) -> bytes:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.decode("utf-8", errors="replace").strip()
        raise MigrationValidationError(
            f"git {' '.join(args)} failed: {detail}"
        ) from exc


def _commit(root: Path, revision: str) -> str:
    resolved = _git(root, "rev-parse", "--verify", f"{revision}^{{commit}}").decode().strip()
    if not _COMMIT_RE.fullmatch(resolved):
        raise MigrationValidationError(f"noncanonical commit identity: {resolved!r}")
    return resolved


def _git_file(root: Path, commit: str, path: str) -> bytes:
    return _git(root, "show", f"{commit}:{path}")


def _blob_oid(root: Path, commit: str, path: str) -> str:
    output = _git(root, "ls-tree", commit, "--", path).decode().strip()
    fields = output.split(maxsplit=3)
    if len(fields) != 4 or fields[1] != "blob" or fields[3] != path:
        raise MigrationValidationError(f"{path} is not one Git blob at {commit}")
    oid = fields[2]
    if not _BLOB_RE.fullmatch(oid):
        raise MigrationValidationError(f"invalid blob identity for {path}: {oid!r}")
    return oid


def _json_bytes(raw: bytes, *, context: str) -> dict[str, Any]:
    try:
        return _object(json.loads(raw), context=context)
    except json.JSONDecodeError as exc:
        raise MigrationValidationError(f"invalid JSON at {context}: {exc}") from exc


def _single_parent(root: Path, commit: str) -> str:
    line = _git(root, "rev-list", "--parents", "-n", "1", commit).decode().strip()
    parts = line.split()
    if len(parts) != 2:
        raise MigrationValidationError(f"{commit} must have exactly one parent")
    return parts[1]


def _changed_paths(root: Path, parent: str, child: str) -> tuple[str, ...]:
    output = _git(
        root,
        "diff-tree",
        "--no-commit-id",
        "--name-only",
        "-r",
        parent,
        child,
    ).decode()
    return tuple(sorted(line for line in output.splitlines() if line))


def _import_rows(raw: bytes, *, path: str) -> list[dict[str, Any]]:
    try:
        tree = ast.parse(raw.decode("utf-8"), filename=path)
    except (UnicodeDecodeError, SyntaxError) as exc:
        raise MigrationValidationError(f"cannot parse {path}: {exc}") from exc
    rows: list[dict[str, Any]] = []
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom) or node.level or node.module is None:
            continue
        rows.append(
            {
                "target_module": node.module,
                "symbols": sorted(alias.name for alias in node.names),
                "local_names": {
                    alias.name: alias.asname or alias.name for alias in node.names
                },
            }
        )
    return rows


def _validate_import_state(
    root: Path,
    commit: str,
    sites: Sequence[dict[str, Any]],
    *,
    state: str,
) -> None:
    if state not in {"old_import", "new_import"}:
        raise AssertionError(state)
    for site in sites:
        path = str(site["path"])
        expected = site[state]
        rows = _import_rows(_git_file(root, commit, path), path=path)
        matches = [row for row in rows if row == expected]
        if len(matches) != 1:
            raise MigrationValidationError(
                f"{commit}:{path} must contain exactly one sealed {state}"
            )
        opposite = site["new_import" if state == "old_import" else "old_import"]
        if any(row == opposite for row in rows):
            raise MigrationValidationError(
                f"{commit}:{path} contains both old and new imports"
            )


def _find_rows(ledger: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    exceptions = _objects(ledger.get("exceptions"), context="ledger.exceptions")
    exception_by_id: dict[str, dict[str, Any]] = {}
    site_by_id: dict[str, dict[str, Any]] = {}
    for row in exceptions:
        semantic_id = row.get("semantic_id")
        if isinstance(semantic_id, str):
            if semantic_id in exception_by_id:
                raise MigrationValidationError(f"duplicate ledger exception {semantic_id}")
            exception_by_id[semantic_id] = row
        sites = row.get("sites", [])
        if isinstance(sites, list):
            for site in sites:
                if not isinstance(site, dict) or not isinstance(
                    site.get("semantic_id"), str
                ):
                    raise MigrationValidationError("invalid ledger site")
                site_id = str(site["semantic_id"])
                if site_id in site_by_id:
                    raise MigrationValidationError(f"duplicate ledger site {site_id}")
                site_by_id[site_id] = site
    return exception_by_id, site_by_id


def _validate_artifact_binding(
    root: Path, commit: str, binding: Mapping[str, Any], *, context: str
) -> bytes:
    _require_keys(
        binding,
        {"path", "blob_oid", "sha256"},
        context=context,
    )
    path = _relative_path(binding["path"], context=f"{context}.path")
    blob_oid = binding["blob_oid"]
    digest = binding["sha256"]
    if not isinstance(blob_oid, str) or not _BLOB_RE.fullmatch(blob_oid):
        raise MigrationValidationError(f"{context}.blob_oid is invalid")
    if not isinstance(digest, str) or not _HEX_SHA_RE.fullmatch(digest):
        raise MigrationValidationError(f"{context}.sha256 is invalid")
    raw = _git_file(root, commit, path)
    if _blob_oid(root, commit, path) != blob_oid:
        raise MigrationValidationError(f"{context} Git blob mismatch")
    if hashlib.sha256(raw).hexdigest() != digest:
        raise MigrationValidationError(f"{context} byte digest mismatch")
    return raw


def load_plan(path: Path) -> dict[str, Any]:
    """Load a canonical, self-hashed migration plan with no schema extensions."""

    plan = _load_canonical(path, seal_field="plan_sha256")
    _require_keys(
        plan,
        {
            "schema",
            "migration_id",
            "campaign",
            "h0_parent_commit",
            "inputs",
            "production_contract",
            "sites",
            "retirement_policy",
            "phases",
            "plan_sha256",
        },
        context="plan",
    )
    if plan["schema"] != PLAN_SCHEMA or plan["migration_id"] != MIGRATION_ID:
        raise MigrationValidationError("unsupported migration plan identity")
    if plan["campaign"] != "P2":
        raise MigrationValidationError("migration plan campaign must be P2")
    if not isinstance(plan["h0_parent_commit"], str) or not _COMMIT_RE.fullmatch(
        plan["h0_parent_commit"]
    ):
        raise MigrationValidationError("plan.h0_parent_commit is invalid")
    if plan["h0_parent_commit"] != H0_PARENT_COMMIT:
        raise MigrationValidationError("plan H0 parent commit drift")

    inputs = _object(plan["inputs"], context="plan.inputs")
    _require_keys(inputs, {"baseline", "ledger"}, context="plan.inputs")
    contract = _object(plan["production_contract"], context="production_contract")
    _require_keys(
        contract,
        {"module", "path", "public_symbols", "forbidden_import_prefixes"},
        context="production_contract",
    )
    if contract["module"] != CONTRACT_MODULE:
        raise MigrationValidationError("production contract module drift")
    if _relative_path(contract["path"], context="production_contract.path") != (
        CONTRACT_PATH.as_posix()
    ):
        raise MigrationValidationError("production contract path drift")
    public_symbols = _strings(
        contract["public_symbols"], context="production_contract.public_symbols"
    )
    expected_public = sorted(
        {
            "DISABLED_BASELINE_SCHEMA",
            "GOLD_FIELD_PREFIXES",
            "INFERENCE_DTO_SCHEMA",
            "inference_dto",
            "sha256_bytes",
            "sha256_file",
            "write_jsonl",
        }
    )
    if public_symbols != expected_public:
        raise MigrationValidationError("production contract public surface drift")
    forbidden = _strings(
        contract["forbidden_import_prefixes"],
        context="production_contract.forbidden_import_prefixes",
    )
    if forbidden != ["chronovisor.lab"]:
        raise MigrationValidationError("production contract must forbid chronovisor.lab")

    sites = _objects(plan["sites"], context="plan.sites")
    if [row.get("semantic_id") for row in sites] != list(MIGRATED_SITE_IDS):
        raise MigrationValidationError("plan sites are missing, duplicated, or reordered")
    if [row.get("path") for row in sorted(sites, key=lambda row: row["path"])] != list(
        H1_PATHS
    ):
        raise MigrationValidationError("plan H1 path set drift")
    for site in sites:
        _require_keys(
            site,
            {"semantic_id", "path", "source_module", "old_import", "new_import"},
            context=f"site {site.get('semantic_id')}",
        )
        site_id = site["semantic_id"]
        if not isinstance(site_id, str) or not _ARCH_ID_RE.fullmatch(site_id):
            raise MigrationValidationError("site semantic ID is invalid")
        _relative_path(site["path"], context=f"site {site_id}.path")
        source_module = site["source_module"]
        if not isinstance(source_module, str) or not source_module.startswith(
            "chronovisor."
        ):
            raise MigrationValidationError(f"site {site_id} source module is invalid")
        expected_path = f"src/{source_module.replace('.', '/')}.py"
        if site["path"] != expected_path:
            raise MigrationValidationError(f"site {site_id} source path drift")
        for state in ("old_import", "new_import"):
            import_row = _object(site[state], context=f"site {site_id}.{state}")
            _require_keys(
                import_row,
                {"target_module", "symbols", "local_names"},
                context=f"site {site_id}.{state}",
            )
            symbols = _strings(
                import_row["symbols"], context=f"site {site_id}.{state}.symbols"
            )
            local_names = _object(
                import_row["local_names"],
                context=f"site {site_id}.{state}.local_names",
            )
            if set(local_names) != set(symbols) or not all(
                isinstance(value, str) and value for value in local_names.values()
            ):
                raise MigrationValidationError(
                    f"site {site_id}.{state} local names do not match symbols"
                )
        if site["old_import"]["target_module"] != (
            "chronovisor.lab.classification_fixture_set"
        ):
            raise MigrationValidationError(f"site {site_id} old target drift")
        if site["new_import"]["target_module"] != CONTRACT_MODULE:
            raise MigrationValidationError(f"site {site_id} new target drift")
        if tuple(site["new_import"]["symbols"]) != EXPECTED_NEW_IMPORTS[site_id]:
            raise MigrationValidationError(f"site {site_id} new symbols drift")
        if site["new_import"]["local_names"] != EXPECTED_LOCAL_NAMES[site_id]:
            raise MigrationValidationError(f"site {site_id} local binding drift")

    retirement = _object(plan["retirement_policy"], context="retirement_policy")
    _require_keys(
        retirement,
        {
            "early_retirement_reason",
            "h0_ledger_removal_campaign",
            "h2_retire_exception_ids",
            "h2_retire_site_ids",
            "p3_retain_edge_ids",
            "p3_retain_site_ids",
        },
        context="retirement_policy",
    )
    if retirement["h0_ledger_removal_campaign"] != "P3":
        raise MigrationValidationError("private exception historical campaign drift")
    if _strings(
        retirement["h2_retire_exception_ids"],
        context="retirement_policy.h2_retire_exception_ids",
    ) != [PRIVATE_EXCEPTION_ID]:
        raise MigrationValidationError("H2 exception retirement drift")
    if _strings(
        retirement["h2_retire_site_ids"],
        context="retirement_policy.h2_retire_site_ids",
    ) != list(MIGRATED_SITE_IDS):
        raise MigrationValidationError("H2 site retirement drift")
    if _strings(
        retirement["p3_retain_edge_ids"],
        context="retirement_policy.p3_retain_edge_ids",
    ) != [CLASSIFICATION_LAB_EDGE_ID]:
        raise MigrationValidationError("P3 edge retention drift")
    if _strings(
        retirement["p3_retain_site_ids"],
        context="retirement_policy.p3_retain_site_ids",
    ) != [PROVIDER_SITE_ID]:
        raise MigrationValidationError("P3 provider retention drift")
    if not isinstance(retirement["early_retirement_reason"], str) or not retirement[
        "early_retirement_reason"
    ].strip():
        raise MigrationValidationError("early retirement reason is required")

    phases = _object(plan["phases"], context="plan.phases")
    _require_keys(phases, {"h0", "h1", "h2"}, context="plan.phases")
    h0 = _object(phases["h0"], context="plan.phases.h0")
    h1 = _object(phases["h1"], context="plan.phases.h1")
    h2 = _object(phases["h2"], context="plan.phases.h2")
    _require_keys(
        h0,
        {"changed_paths", "callsite_state", "ledger_state", "active_counts"},
        context="plan.phases.h0",
    )
    _require_keys(
        h1,
        {"changed_paths", "callsite_state", "ledger_state"},
        context="plan.phases.h1",
    )
    _require_keys(
        h2,
        {
            "changed_paths",
            "callsite_state",
            "ledger_state",
            "receipt_path",
            "active_counts",
            "retired_counts",
        },
        context="plan.phases.h2",
    )
    if _strings(h0["changed_paths"], context="h0.changed_paths") != list(H0_PATHS):
        raise MigrationValidationError("H0 changed path scope drift")
    if _strings(h1["changed_paths"], context="h1.changed_paths") != list(H1_PATHS):
        raise MigrationValidationError("H1 changed path scope drift")
    if _strings(h2["changed_paths"], context="h2.changed_paths") != list(H2_PATHS):
        raise MigrationValidationError("H2 changed path scope drift")
    if (
        h0["callsite_state"] != "legacy-lab-imports"
        or h0["ledger_state"] != "pre-p2"
        or h1["callsite_state"] != "production-contract-imports"
        or h1["ledger_state"] != "pre-p2"
        or h2["callsite_state"] != "production-contract-imports"
        or h2["ledger_state"] != "post-p2"
    ):
        raise MigrationValidationError("phase state contract drift")
    if h2["receipt_path"] != RECEIPT_PATH.as_posix():
        raise MigrationValidationError("H2 receipt path drift")
    if h0["active_counts"] != EXPECTED_H0_ACTIVE_COUNTS:
        raise MigrationValidationError("H0 active counts drift")
    if h2["active_counts"] != EXPECTED_H2_ACTIVE_COUNTS:
        raise MigrationValidationError("H2 active counts drift")
    if h2["retired_counts"] != EXPECTED_H2_RETIRED_COUNTS:
        raise MigrationValidationError("H2 retired counts drift")
    return plan


def _selected_counts(active: Mapping[str, Any]) -> dict[str, Any]:
    return {key: active.get(key) for key in EXPECTED_H0_ACTIVE_COUNTS}


def validate_plan(root: Path, plan: Mapping[str, Any]) -> dict[str, Any]:
    """Validate H0 inputs and legacy callsites from the sealed parent commit."""

    commit = _commit(root, str(plan["h0_parent_commit"]))
    if commit != plan["h0_parent_commit"]:
        raise MigrationValidationError("H0 parent must be a full commit identity")
    inputs = _object(plan["inputs"], context="plan.inputs")
    baseline_raw = _validate_artifact_binding(
        root,
        commit,
        _object(inputs["baseline"], context="plan.inputs.baseline"),
        context="plan.inputs.baseline",
    )
    ledger_raw = _validate_artifact_binding(
        root,
        commit,
        _object(inputs["ledger"], context="plan.inputs.ledger"),
        context="plan.inputs.ledger",
    )
    baseline = _json_bytes(baseline_raw, context=f"{commit}:{BASELINE_PATH}")
    ledger = _json_bytes(ledger_raw, context=f"{commit}:{LEDGER_PATH}")
    if inputs["baseline"]["path"] != BASELINE_PATH.as_posix():
        raise MigrationValidationError("baseline input path drift")
    if inputs["ledger"]["path"] != LEDGER_PATH.as_posix():
        raise MigrationValidationError("ledger input path drift")
    active = _object(
        _object(baseline.get("counts"), context="baseline.counts").get("active"),
        context="baseline.counts.active",
    )
    if _selected_counts(active) != EXPECTED_H0_ACTIVE_COUNTS:
        raise MigrationValidationError("sealed baseline H0 counts drift")
    if _selected_counts(_object(ledger.get("counts"), context="ledger.counts")) != (
        EXPECTED_H0_ACTIVE_COUNTS
    ):
        raise MigrationValidationError("sealed ledger H0 counts drift")
    if (
        baseline.get("source_baseline_head") != FROZEN_SOURCE_HEAD
        or ledger.get("source_baseline_head") != FROZEN_SOURCE_HEAD
    ):
        raise MigrationValidationError("sealed exception source commit drift")
    compact_seed_sha = hashlib.sha256(
        json.dumps(
            baseline,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    if ledger.get("baseline_sha256") != compact_seed_sha:
        raise MigrationValidationError("sealed ledger is not bound to sealed baseline")
    retired_counts = _object(
        _object(baseline.get("counts"), context="baseline.counts").get("retired"),
        context="baseline.counts.retired",
    )
    if any(retired_counts.values()):
        raise MigrationValidationError("H0 baseline already contains retired IDs")
    active_membership = {
        "exception_semantic_ids": [PRIVATE_EXCEPTION_ID],
        "cross_domain_site_semantic_ids": [*MIGRATED_SITE_IDS, PROVIDER_SITE_ID],
        "production_to_lab_edge_semantic_ids": [CLASSIFICATION_LAB_EDGE_ID],
        "production_to_lab_static_site_semantic_ids": [
            *MIGRATED_SITE_IDS,
            PROVIDER_SITE_ID,
        ],
    }
    for field, expected_ids in active_membership.items():
        active_ids = _seed_ids(baseline, field, "active")
        missing = sorted(set(expected_ids) - active_ids)
        if missing:
            raise MigrationValidationError(
                f"sealed baseline lacks active {field}: {missing}"
            )

    exception_by_id, site_by_id = _find_rows(ledger)
    private = exception_by_id.get(PRIVATE_EXCEPTION_ID)
    if private is None or private.get("removal_campaign") != "P3":
        raise MigrationValidationError("sealed private exception is absent or changed")
    edge = exception_by_id.get(CLASSIFICATION_LAB_EDGE_ID)
    if edge is None or edge.get("removal_campaign") != "P3":
        raise MigrationValidationError("sealed classification-to-lab edge drift")
    if PROVIDER_SITE_ID not in site_by_id:
        raise MigrationValidationError("sealed P3 provider site is absent")

    sites = _objects(plan["sites"], context="plan.sites")
    for planned in sites:
        site_id = str(planned["semantic_id"])
        recorded = site_by_id.get(site_id)
        if recorded is None:
            raise MigrationValidationError(f"sealed site is absent: {site_id}")
        if recorded.get("source_module") != planned["source_module"]:
            raise MigrationValidationError(f"sealed site source module drift: {site_id}")
        old_import = planned["old_import"]
        if sorted(recorded.get("symbols", [])) != old_import["symbols"]:
            raise MigrationValidationError(f"sealed site symbols drift: {site_id}")
        if recorded.get("target_module") != old_import["target_module"]:
            raise MigrationValidationError(f"sealed site target drift: {site_id}")
    _validate_import_state(root, commit, sites, state="old_import")
    return {
        "migration_id": MIGRATION_ID,
        "h0_parent_commit": commit,
        "site_count": len(sites),
        "state": "valid-h0-plan",
    }


def _validate_contract_source(root: Path, commit: str, plan: Mapping[str, Any]) -> None:
    raw = _git_file(root, commit, CONTRACT_PATH.as_posix())
    try:
        tree = ast.parse(raw.decode("utf-8"), filename=CONTRACT_PATH.as_posix())
    except (UnicodeDecodeError, SyntaxError) as exc:
        raise MigrationValidationError(f"production contract cannot be parsed: {exc}") from exc
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            targets = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            targets = [node.module]
        else:
            continue
        if any(target == "chronovisor.lab" or target.startswith("chronovisor.lab.") for target in targets):
            raise MigrationValidationError("production contract imports chronovisor.lab")
    namespace: dict[str, Any] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            namespace[node.name] = True
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    namespace[target.id] = True
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                namespace[alias.asname or alias.name] = True
    expected = plan["production_contract"]["public_symbols"]
    missing = sorted(set(expected) - set(namespace))
    if missing:
        raise MigrationValidationError(f"production contract lacks symbols: {missing}")


def validate_h1(
    root: Path, plan: Mapping[str, Any], h1_revision: str
) -> dict[str, Any]:
    """Validate the exact additive H0 then exact five-file H1 transition."""

    validate_plan(root, plan)
    h1 = _commit(root, h1_revision)
    h0 = _single_parent(root, h1)
    parent = _single_parent(root, h0)
    if parent != plan["h0_parent_commit"]:
        raise MigrationValidationError("H1 is not the direct child of sealed H0")
    if _changed_paths(root, parent, h0) != H0_PATHS:
        raise MigrationValidationError("H0 commit path scope mismatch")
    if _changed_paths(root, h0, h1) != H1_PATHS:
        raise MigrationValidationError("H1 commit path scope mismatch")
    if _git_file(root, h0, PLAN_PATH.as_posix()) != _canonical_json_bytes(plan):
        raise MigrationValidationError("H0 committed plan differs from supplied plan")
    _validate_contract_source(root, h0, plan)
    if _git_file(root, h0, CONTRACT_PATH.as_posix()) != _git_file(
        root, h1, CONTRACT_PATH.as_posix()
    ):
        raise MigrationValidationError("H1 changed the production contract")
    sites = _objects(plan["sites"], context="plan.sites")
    _validate_import_state(root, h0, sites, state="old_import")
    _validate_import_state(root, h1, sites, state="new_import")
    for path in (BASELINE_PATH.as_posix(), LEDGER_PATH.as_posix()):
        if _git_file(root, h0, path) != _git_file(root, h1, path):
            raise MigrationValidationError(f"H1 changed sealed ledger artifact: {path}")
    return {
        "migration_id": MIGRATION_ID,
        "h0_commit": h0,
        "h1_commit": h1,
        "state": "valid-h1-transition",
    }


def _seed_ids(seed: Mapping[str, Any], field: str, state: str) -> set[str]:
    bucket = _object(seed.get(field), context=f"baseline.{field}")
    return set(_strings(bucket.get(state), context=f"baseline.{field}.{state}"))


def _validate_h2_payloads(
    baseline_raw: bytes, ledger_raw: bytes, plan: Mapping[str, Any]
) -> None:
    baseline = _json_bytes(baseline_raw, context="H2 baseline")
    ledger = _json_bytes(ledger_raw, context="H2 ledger")
    counts = _object(baseline.get("counts"), context="H2 baseline.counts")
    active = _object(counts.get("active"), context="H2 baseline.counts.active")
    retired = _object(counts.get("retired"), context="H2 baseline.counts.retired")
    if _selected_counts(active) != EXPECTED_H2_ACTIVE_COUNTS:
        raise MigrationValidationError("H2 active count invariant failed")
    if retired != EXPECTED_H2_RETIRED_COUNTS:
        raise MigrationValidationError("H2 retired count invariant failed")
    if _selected_counts(_object(ledger.get("counts"), context="H2 ledger.counts")) != (
        EXPECTED_H2_ACTIVE_COUNTS
    ):
        raise MigrationValidationError("H2 ledger count invariant failed")
    compact_seed_sha = hashlib.sha256(
        json.dumps(
            baseline,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    if ledger.get("baseline_sha256") != compact_seed_sha:
        raise MigrationValidationError("H2 ledger is not bound to H2 baseline")

    if PRIVATE_EXCEPTION_ID not in _seed_ids(
        baseline, "exception_semantic_ids", "retired"
    ):
        raise MigrationValidationError("H2 private exception was not retired")
    for site_id in MIGRATED_SITE_IDS:
        if site_id not in _seed_ids(
            baseline, "cross_domain_site_semantic_ids", "retired"
        ) or site_id not in _seed_ids(
            baseline, "production_to_lab_static_site_semantic_ids", "retired"
        ):
            raise MigrationValidationError(f"H2 site was not retired: {site_id}")
    if CLASSIFICATION_LAB_EDGE_ID not in _seed_ids(
        baseline, "production_to_lab_edge_semantic_ids", "active"
    ):
        raise MigrationValidationError("P3 classification-to-lab edge was retired early")
    if PROVIDER_SITE_ID not in _seed_ids(
        baseline, "cross_domain_site_semantic_ids", "active"
    ) or PROVIDER_SITE_ID not in _seed_ids(
        baseline, "production_to_lab_static_site_semantic_ids", "active"
    ):
        raise MigrationValidationError("P3 provider site was retired early")

    exception_by_id, site_by_id = _find_rows(ledger)
    if PRIVATE_EXCEPTION_ID in exception_by_id:
        raise MigrationValidationError("H2 ledger retains the private exception")
    if any(site_id in site_by_id for site_id in MIGRATED_SITE_IDS):
        raise MigrationValidationError("H2 ledger retains a migrated site")
    edge = exception_by_id.get(CLASSIFICATION_LAB_EDGE_ID)
    if edge is None or edge.get("removal_campaign") != "P3":
        raise MigrationValidationError("H2 ledger removed the P3 edge")
    if PROVIDER_SITE_ID not in site_by_id:
        raise MigrationValidationError("H2 ledger removed the P3 provider site")


def build_receipt(
    root: Path, plan: Mapping[str, Any], h1_revision: str
) -> dict[str, Any]:
    """Build an H2 receipt from validated worktree ledger artifacts."""

    h1_report = validate_h1(root, plan, h1_revision)
    baseline_raw = (root / BASELINE_PATH).read_bytes()
    ledger_raw = (root / LEDGER_PATH).read_bytes()
    _validate_h2_payloads(baseline_raw, ledger_raw, plan)
    payload: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "migration_id": MIGRATION_ID,
        "plan_sha256": plan["plan_sha256"],
        "h1_commit": h1_report["h1_commit"],
        "h2_artifacts": {
            "baseline": {
                "path": BASELINE_PATH.as_posix(),
                "sha256": hashlib.sha256(baseline_raw).hexdigest(),
            },
            "ledger": {
                "path": LEDGER_PATH.as_posix(),
                "sha256": hashlib.sha256(ledger_raw).hexdigest(),
            },
        },
        "active_counts": EXPECTED_H2_ACTIVE_COUNTS,
        "retired_counts": EXPECTED_H2_RETIRED_COUNTS,
        "retired_exception_ids": [PRIVATE_EXCEPTION_ID],
        "retired_site_ids": list(MIGRATED_SITE_IDS),
        "p3_retained_edge_ids": [CLASSIFICATION_LAB_EDGE_ID],
        "p3_retained_site_ids": [PROVIDER_SITE_ID],
    }
    payload["receipt_sha256"] = _seal(payload, "receipt_sha256")
    return payload


def load_receipt(path: Path) -> dict[str, Any]:
    """Load and strictly validate a canonical H2 receipt."""

    receipt = _load_canonical(path, seal_field="receipt_sha256")
    _require_keys(
        receipt,
        {
            "schema",
            "migration_id",
            "plan_sha256",
            "h1_commit",
            "h2_artifacts",
            "active_counts",
            "retired_counts",
            "retired_exception_ids",
            "retired_site_ids",
            "p3_retained_edge_ids",
            "p3_retained_site_ids",
            "receipt_sha256",
        },
        context="receipt",
    )
    if receipt["schema"] != RECEIPT_SCHEMA or receipt["migration_id"] != MIGRATION_ID:
        raise MigrationValidationError("unsupported receipt identity")
    if not isinstance(receipt["plan_sha256"], str) or not _HEX_SHA_RE.fullmatch(
        receipt["plan_sha256"]
    ):
        raise MigrationValidationError("receipt plan SHA is invalid")
    if not isinstance(receipt["h1_commit"], str) or not _COMMIT_RE.fullmatch(
        receipt["h1_commit"]
    ):
        raise MigrationValidationError("receipt H1 commit is invalid")
    if receipt["active_counts"] != EXPECTED_H2_ACTIVE_COUNTS:
        raise MigrationValidationError("receipt active counts drift")
    if receipt["retired_counts"] != EXPECTED_H2_RETIRED_COUNTS:
        raise MigrationValidationError("receipt retired counts drift")
    expected_lists = {
        "retired_exception_ids": [PRIVATE_EXCEPTION_ID],
        "retired_site_ids": list(MIGRATED_SITE_IDS),
        "p3_retained_edge_ids": [CLASSIFICATION_LAB_EDGE_ID],
        "p3_retained_site_ids": [PROVIDER_SITE_ID],
    }
    for field, expected in expected_lists.items():
        if _strings(receipt[field], context=f"receipt.{field}") != expected:
            raise MigrationValidationError(f"receipt {field} drift")
    artifacts = _object(receipt["h2_artifacts"], context="receipt.h2_artifacts")
    _require_keys(artifacts, {"baseline", "ledger"}, context="receipt.h2_artifacts")
    for name, path in (("baseline", BASELINE_PATH), ("ledger", LEDGER_PATH)):
        binding = _object(artifacts[name], context=f"receipt.h2_artifacts.{name}")
        _require_keys(binding, {"path", "sha256"}, context=f"receipt.{name}")
        if binding["path"] != path.as_posix() or not isinstance(
            binding["sha256"], str
        ) or not _HEX_SHA_RE.fullmatch(binding["sha256"]):
            raise MigrationValidationError(f"receipt {name} binding is invalid")
    return receipt


def _receipt_additions(
    root: Path, h1: str, tip: str, receipt_path: str
) -> list[str]:
    commits = _git(
        root,
        "rev-list",
        "--first-parent",
        "--ancestry-path",
        f"{h1}..{tip}",
    ).decode().splitlines()
    additions = []
    for commit in commits:
        parent = _single_parent(root, commit)
        status = _git(
            root,
            "diff-tree",
            "--no-commit-id",
            "--name-status",
            "-r",
            parent,
            commit,
            "--",
            receipt_path,
        ).decode().strip()
        if status == f"A\t{receipt_path}":
            additions.append(commit)
        elif status:
            raise MigrationValidationError("receipt path has a non-additive history")
    return additions


def verify_receipt(
    root: Path,
    plan: Mapping[str, Any],
    receipt: Mapping[str, Any],
    tip_revision: str,
) -> dict[str, Any]:
    """Find and verify the unique first-parent H2 commit and exact H1->H2 diff."""

    if receipt["plan_sha256"] != plan["plan_sha256"]:
        raise MigrationValidationError("receipt does not bind the supplied plan")
    h1 = _commit(root, str(receipt["h1_commit"]))
    validate_h1(root, plan, h1)
    tip = _commit(root, tip_revision)
    additions = _receipt_additions(root, h1, tip, RECEIPT_PATH.as_posix())
    if len(additions) != 1:
        raise MigrationValidationError(
            f"expected one first-parent H2 receipt addition; found {len(additions)}"
        )
    h2 = additions[0]
    if _single_parent(root, h2) != h1:
        raise MigrationValidationError("H2 is not the direct child of H1")
    if _changed_paths(root, h1, h2) != H2_PATHS:
        raise MigrationValidationError("H1->H2 changed path scope mismatch")
    committed_receipt = _git_file(root, h2, RECEIPT_PATH.as_posix())
    if committed_receipt != _canonical_json_bytes(receipt):
        raise MigrationValidationError("committed H2 receipt bytes differ")
    artifacts = receipt["h2_artifacts"]
    baseline_raw = _git_file(root, h2, BASELINE_PATH.as_posix())
    ledger_raw = _git_file(root, h2, LEDGER_PATH.as_posix())
    for name, raw in (("baseline", baseline_raw), ("ledger", ledger_raw)):
        if hashlib.sha256(raw).hexdigest() != artifacts[name]["sha256"]:
            raise MigrationValidationError(f"committed H2 {name} digest differs")
    _validate_h2_payloads(baseline_raw, ledger_raw, plan)
    return {
        "migration_id": MIGRATION_ID,
        "h1_commit": h1,
        "h2_commit": h2,
        "tip_commit": tip,
        "state": "valid-h2-receipt",
    }


def write_canonical_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write one deterministic migration artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_json_bytes(payload))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).parents[1])
    parser.add_argument("--plan", type=Path, default=PLAN_PATH)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("validate-plan")
    h1 = commands.add_parser("validate-h1")
    h1.add_argument("--h1", required=True)
    build = commands.add_parser("build-receipt")
    build.add_argument("--h1", required=True)
    build.add_argument("--output", type=Path, default=RECEIPT_PATH)
    verify = commands.add_parser("verify-receipt")
    verify.add_argument("--receipt", type=Path, default=RECEIPT_PATH)
    verify.add_argument("--tip", default="HEAD")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.root.resolve()
    plan_path = args.plan if args.plan.is_absolute() else root / args.plan
    try:
        plan = load_plan(plan_path)
        if args.command == "validate-plan":
            report = validate_plan(root, plan)
        elif args.command == "validate-h1":
            report = validate_h1(root, plan, args.h1)
        elif args.command == "build-receipt":
            receipt = build_receipt(root, plan, args.h1)
            output = args.output if args.output.is_absolute() else root / args.output
            write_canonical_json(output, receipt)
            report = {
                "migration_id": MIGRATION_ID,
                "output": output.relative_to(root).as_posix(),
                "state": "h2-receipt-built",
            }
        else:
            receipt_path = (
                args.receipt if args.receipt.is_absolute() else root / args.receipt
            )
            receipt = load_receipt(receipt_path)
            report = verify_receipt(root, plan, receipt, args.tip)
    except MigrationValidationError as exc:
        print(json.dumps({"passed": False, "error": str(exc)}, sort_keys=True))
        return 1
    print(json.dumps({"passed": True, **report}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
