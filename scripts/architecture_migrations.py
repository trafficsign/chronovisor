#!/usr/bin/env python3
"""Validate sealed Campaign P architecture migrations at Git object boundaries."""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import os
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
H0_SEED_COMMIT = "6cbcd1e617c631bee23de0cf8c6597f324485205"
GATE_HYGIENE_COMMIT = "f1208ee3914e7ec402b84ebda16423f57065e041"
EVIDENCE_HARDENING_COMMIT = "63371cf32937948da784292fc8fc6db15e6a4679"
CANONICAL_PLAN_SHA256 = (
    "bedd3ad58736a1489115a76ef5638cc9adf522c687226f7c0a6156e20868ae69"
)
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
CONTRACT_MODULE = "chronovisor.classification.classification_fixture_contract"
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
GATE_HYGIENE_PATHS = ("tests/test_architecture_baseline.py",)
EVIDENCE_HARDENING_PATHS = tuple(
    sorted(
        {
            PLAN_PATH.as_posix(),
            "scripts/architecture_migrations.py",
            "tests/test_architecture_migrations.py",
        }
    )
)
REVIEW_FIX_PATHS = tuple(
    sorted(
        {
            "scripts/architecture_migrations.py",
            "tests/test_architecture_baseline.py",
            "tests/test_architecture_migrations.py",
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
EXPECTED_SOURCE_TRANSFORMS = {
    MIGRATED_SITE_IDS[0]: {
        "h0_sha256": (
            "7b65c798842ce4c3269028664fd42437835f0f38a5cdbf92b2c84588d0c090ed"
        ),
        "h1_sha256": (
            "cfa318c7a235168914209d2d4bbf860ac52e1c65bcb1c965d34fa18b91db57ea"
        ),
        "new_snippet": (
            "from chronovisor.classification.classification_fixture_contract "
            "import inference_dto\n"
        ),
        "old_snippet": (
            "from chronovisor.lab.classification_fixture_set import "
            "inference_dto\n"
        ),
    },
    MIGRATED_SITE_IDS[1]: {
        "h0_sha256": (
            "52748c3d51a8f282ee4841371d83d2ebfde1452615a74dc8ca3994e6fcb33f27"
        ),
        "h1_sha256": (
            "0f93ca7c9bbadee526779968682a1f094cedd4a3b86dbb80971b21f48e6a72c4"
        ),
        "new_snippet": (
            "from chronovisor.classification.classification_fixture_contract "
            "import (\n"
            "    DISABLED_BASELINE_SCHEMA,\n"
            "    sha256_bytes,\n"
            "    sha256_file,\n"
            ")\n"
        ),
        "old_snippet": (
            "from chronovisor.lab.classification_fixture_set import (\n"
            "    DISABLED_BASELINE_SCHEMA,\n"
            "    sha256_bytes,\n"
            "    sha256_file,\n"
            ")\n"
        ),
    },
    MIGRATED_SITE_IDS[2]: {
        "h0_sha256": (
            "b160333fbde80d5e75beb1964cfc54c5d203c15eb0522b99ae5c2c980ebbb37c"
        ),
        "h1_sha256": (
            "89b5c5b13a371ecb6e7091d1a9a0555880ae85b896323ff61f3978f77692c183"
        ),
        "new_snippet": (
            "from chronovisor.classification.classification_fixture_contract "
            "import (\n"
            "    sha256_bytes,\n"
            "    sha256_file,\n"
            ")\n"
        ),
        "old_snippet": (
            "from chronovisor.lab.classification_fixture_set import "
            "sha256_bytes, sha256_file\n"
        ),
    },
    MIGRATED_SITE_IDS[3]: {
        "h0_sha256": (
            "a8a59f9da35f1d06b60caf0f39583426dc7940568f6505d465cb21e3d29d0d9b"
        ),
        "h1_sha256": (
            "a6dc204fa813f97f21e5c810d8da1a34d6392eecb46d2227a61284b4a2e4e62b"
        ),
        "new_snippet": (
            "from chronovisor.classification.classification_fixture_contract "
            "import (\n"
            "    sha256_bytes,\n"
            "    sha256_file,\n"
            ")\n"
        ),
        "old_snippet": (
            "from chronovisor.lab.classification_fixture_set import "
            "sha256_bytes, sha256_file\n"
        ),
    },
    MIGRATED_SITE_IDS[4]: {
        "h0_sha256": (
            "2e86604c7285f76e75b5b5be5ca8177fda12527f68488c275cafa4fc93da4e7c"
        ),
        "h1_sha256": (
            "b34864549a5be45620c14171aa61cf9f5c23060d6f02c965a1806ce407a81f41"
        ),
        "new_snippet": (
            "from chronovisor.classification.classification_fixture_contract "
            "import (\n"
            "    sha256_bytes,\n"
            "    write_jsonl as _write_jsonl,\n"
            ")\n"
        ),
        "old_snippet": (
            "from chronovisor.lab.classification_fixture_set import "
            "_write_jsonl, sha256_bytes\n"
        ),
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


def _document_json_bytes(payload: Any) -> bytes:
    """Encode ledger documents while preserving their established key order."""

    try:
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise MigrationValidationError(f"payload is not strict JSON: {exc}") from exc
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
    if (
        path.is_absolute()
        or value == "."
        or ".." in path.parts
        or path.as_posix() != value
    ):
        raise MigrationValidationError(
            f"{context} must be canonical repo-relative path"
        )
    return value


def _load_canonical(path: Path, *, seal_field: str) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        payload = _decode_json(raw, context=str(path))
    except OSError as exc:
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
    env = {
        key: value for key, value in os.environ.items() if not key.startswith("GIT_")
    }
    try:
        return subprocess.run(
            ["git", "--no-replace-objects", "--literal-pathspecs", *args],
            cwd=root,
            check=True,
            capture_output=True,
            env=env,
        ).stdout
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.decode("utf-8", errors="replace").strip()
        raise MigrationValidationError(
            f"git {' '.join(args)} failed: {detail}"
        ) from exc


def _commit(root: Path, revision: str) -> str:
    if not isinstance(revision, str) or not _COMMIT_RE.fullmatch(revision):
        raise MigrationValidationError(
            "revision must be a lowercase full 40-character commit SHA"
        )
    resolved = (
        _git(
            root,
            "rev-parse",
            "--verify",
            "--end-of-options",
            f"{revision}^{{commit}}",
        )
        .decode()
        .strip()
    )
    if not _COMMIT_RE.fullmatch(resolved):
        raise MigrationValidationError(f"noncanonical commit identity: {resolved!r}")
    if resolved != revision:
        raise MigrationValidationError("resolved commit identity differs from revision")
    return resolved


def _tree_entry(root: Path, commit: str, path: str) -> tuple[str, str]:
    canonical = _relative_path(path, context="Git tree path")
    output = _git(root, "ls-tree", "-z", "--full-tree", commit, "--", canonical)
    records = [record for record in output.split(b"\0") if record]
    if len(records) != 1 or b"\t" not in records[0]:
        raise MigrationValidationError(
            f"{canonical} is not one Git tree entry at {commit}"
        )
    metadata, raw_path = records[0].split(b"\t", 1)
    try:
        mode, object_type, oid = metadata.decode("ascii").split()
        recorded_path = raw_path.decode("utf-8")
    except (UnicodeDecodeError, ValueError) as exc:
        raise MigrationValidationError(
            f"malformed Git tree entry for {canonical} at {commit}"
        ) from exc
    if recorded_path != canonical:
        raise MigrationValidationError(f"Git tree path mismatch for {canonical}")
    if mode != "100644" or object_type != "blob":
        raise MigrationValidationError(
            f"{canonical} must be an exact 100644 blob at {commit}"
        )
    if not _BLOB_RE.fullmatch(oid):
        raise MigrationValidationError(
            f"invalid blob identity for {canonical}: {oid!r}"
        )
    return mode, oid


def _git_file(root: Path, commit: str, path: str) -> bytes:
    _mode, oid = _tree_entry(root, commit, path)
    return _git(root, "cat-file", "blob", oid)


def _blob_oid(root: Path, commit: str, path: str) -> str:
    _mode, oid = _tree_entry(root, commit, path)
    return oid


def _reject_constant(value: str) -> Any:
    raise MigrationValidationError(f"non-finite JSON number is forbidden: {value}")


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise MigrationValidationError(f"duplicate JSON key is forbidden: {key}")
        payload[key] = value
    return payload


def _decode_json(raw: bytes, *, context: str) -> Any:
    try:
        return json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_constant,
        )
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise MigrationValidationError(f"invalid JSON at {context}: {exc}") from exc


def _json_bytes(
    raw: bytes, *, context: str, canonical_document: bool = False
) -> dict[str, Any]:
    payload = _object(_decode_json(raw, context=context), context=context)
    if canonical_document and raw != _document_json_bytes(payload):
        raise MigrationValidationError(f"{context} is not canonical document JSON")
    return payload


def _single_parent(root: Path, commit: str) -> str:
    line = _git(root, "rev-list", "--parents", "-n", "1", commit).decode().strip()
    parts = line.split()
    if len(parts) != 2:
        raise MigrationValidationError(f"{commit} must have exactly one parent")
    return parts[1]


def _changed_entries(root: Path, parent: str, child: str) -> dict[str, str]:
    raw = _git(
        root,
        "diff-tree",
        "--no-commit-id",
        "--name-status",
        "-z",
        "--no-renames",
        "-r",
        parent,
        child,
    )
    fields = [field for field in raw.split(b"\0") if field]
    if len(fields) % 2:
        raise MigrationValidationError("malformed Git diff-tree output")
    entries: dict[str, str] = {}
    for index in range(0, len(fields), 2):
        try:
            status = fields[index].decode("ascii")
            path = fields[index + 1].decode("utf-8")
        except UnicodeDecodeError as exc:
            raise MigrationValidationError("noncanonical Git diff path") from exc
        _relative_path(path, context="Git diff path")
        if status not in {"A", "M", "D"} or path in entries:
            raise MigrationValidationError("unsupported or duplicate Git diff entry")
        entries[path] = status
    return entries


def _require_transition(
    root: Path,
    parent: str,
    child: str,
    expected: Mapping[str, str],
    *,
    context: str,
) -> None:
    actual = _changed_entries(root, parent, child)
    if actual != dict(expected):
        raise MigrationValidationError(
            f"{context} changed path/status scope mismatch: {actual}"
        )
    for path, status in actual.items():
        if status != "A":
            _tree_entry(root, parent, path)
        if status != "D":
            _tree_entry(root, child, path)


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


def _exact_import_snippet(raw: bytes, *, path: str) -> dict[str, Any]:
    try:
        tree = ast.parse(raw.decode("utf-8"), filename=path)
    except (UnicodeDecodeError, SyntaxError) as exc:
        raise MigrationValidationError(f"cannot parse {path}: {exc}") from exc
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.ImportFrom):
        raise MigrationValidationError(
            f"{path} must contain exactly one from-import statement"
        )
    node = tree.body[0]
    if node.level or node.module is None:
        raise MigrationValidationError(f"{path} must use an absolute from-import")
    return {
        "target_module": node.module,
        "symbols": sorted(alias.name for alias in node.names),
        "local_names": {
            alias.name: alias.asname or alias.name for alias in node.names
        },
    }


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
        if not isinstance(semantic_id, str) or not _ARCH_ID_RE.fullmatch(semantic_id):
            raise MigrationValidationError(
                "invalid or missing ledger exception semantic_id"
            )
        if semantic_id in exception_by_id:
            raise MigrationValidationError(f"duplicate ledger exception {semantic_id}")
        exception_by_id[semantic_id] = row
        sites = row.get("sites", [])
        if not isinstance(sites, list):
            raise MigrationValidationError(
                f"ledger sites must be a list: {semantic_id}"
            )
        for site in sites:
            if (
                not isinstance(site, dict)
                or not isinstance(site.get("semantic_id"), str)
                or not _ARCH_ID_RE.fullmatch(str(site["semantic_id"]))
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
            "history",
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

    history = _object(plan["history"], context="plan.history")
    _require_keys(
        history,
        {
            "frozen_source_parent",
            "h0_seed_commit",
            "h0_seed_added_paths",
            "gate_hygiene_commit",
            "gate_hygiene_changed_paths",
            "evidence_hardening_changed_paths",
        },
        context="plan.history",
    )
    expected_history = {
        "frozen_source_parent": H0_PARENT_COMMIT,
        "h0_seed_commit": H0_SEED_COMMIT,
        "h0_seed_added_paths": list(H0_PATHS),
        "gate_hygiene_commit": GATE_HYGIENE_COMMIT,
        "gate_hygiene_changed_paths": list(GATE_HYGIENE_PATHS),
        "evidence_hardening_changed_paths": list(EVIDENCE_HARDENING_PATHS),
    }
    if history != expected_history:
        raise MigrationValidationError("plan history chain drift")

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
        raise MigrationValidationError(
            "production contract must forbid chronovisor.lab"
        )

    sites = _objects(plan["sites"], context="plan.sites")
    if [row.get("semantic_id") for row in sites] != list(MIGRATED_SITE_IDS):
        raise MigrationValidationError(
            "plan sites are missing, duplicated, or reordered"
        )
    if [row.get("path") for row in sorted(sites, key=lambda row: row["path"])] != list(
        H1_PATHS
    ):
        raise MigrationValidationError("plan H1 path set drift")
    for site in sites:
        _require_keys(
            site,
            {
                "semantic_id",
                "path",
                "source_module",
                "old_import",
                "new_import",
                "source_transform",
            },
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
        transform = _object(
            site["source_transform"], context=f"site {site_id}.source_transform"
        )
        _require_keys(
            transform,
            {"h0_sha256", "h1_sha256", "old_snippet", "new_snippet"},
            context=f"site {site_id}.source_transform",
        )
        for field in ("h0_sha256", "h1_sha256"):
            if not isinstance(transform[field], str) or not _HEX_SHA_RE.fullmatch(
                transform[field]
            ):
                raise MigrationValidationError(
                    f"site {site_id} has invalid source {field}"
                )
        for field in ("old_snippet", "new_snippet"):
            snippet = transform[field]
            if not isinstance(snippet, str) or not snippet.endswith("\n"):
                raise MigrationValidationError(
                    f"site {site_id} has invalid source {field}"
                )
        if transform["old_snippet"] == transform["new_snippet"]:
            raise MigrationValidationError(f"site {site_id} source transform is empty")
        if transform != EXPECTED_SOURCE_TRANSFORMS[site_id]:
            raise MigrationValidationError(f"site {site_id} source transform drift")
        if _exact_import_snippet(
            transform["old_snippet"].encode("utf-8"), path=f"{site_id}:old"
        ) != site["old_import"]:
            raise MigrationValidationError(f"site {site_id} old source snippet drift")
        if _exact_import_snippet(
            transform["new_snippet"].encode("utf-8"), path=f"{site_id}:new"
        ) != site["new_import"]:
            raise MigrationValidationError(f"site {site_id} new source snippet drift")

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
    if (
        not isinstance(retirement["early_retirement_reason"], str)
        or not retirement["early_retirement_reason"].strip()
    ):
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


def _validate_fixed_history(root: Path, plan: Mapping[str, Any]) -> None:
    source_parent = _commit(root, str(plan["h0_parent_commit"]))
    h0_seed = _commit(root, str(plan["history"]["h0_seed_commit"]))
    gate_hygiene = _commit(root, str(plan["history"]["gate_hygiene_commit"]))
    evidence_hardening = _commit(root, EVIDENCE_HARDENING_COMMIT)
    if _single_parent(root, h0_seed) != source_parent:
        raise MigrationValidationError("H0 seed parent identity drift")
    _require_transition(
        root,
        source_parent,
        h0_seed,
        {path: "A" for path in H0_PATHS},
        context="source-parent->H0-seed",
    )
    if _single_parent(root, gate_hygiene) != h0_seed:
        raise MigrationValidationError("gate hygiene parent identity drift")
    _require_transition(
        root,
        h0_seed,
        gate_hygiene,
        {path: "M" for path in GATE_HYGIENE_PATHS},
        context="H0-seed->gate-hygiene",
    )
    if _single_parent(root, evidence_hardening) != gate_hygiene:
        raise MigrationValidationError("evidence hardening parent identity drift")
    _require_transition(
        root,
        gate_hygiene,
        evidence_hardening,
        {path: "M" for path in EVIDENCE_HARDENING_PATHS},
        context="gate-hygiene->evidence-hardening",
    )
    if _git_file(root, evidence_hardening, PLAN_PATH.as_posix()) != (
        _canonical_json_bytes(plan)
    ):
        raise MigrationValidationError(
            "fixed evidence hardening plan differs from supplied canonical plan"
        )


def validate_plan(root: Path, plan: Mapping[str, Any]) -> dict[str, Any]:
    """Validate H0 inputs and legacy callsites from the sealed parent commit."""

    commit = _commit(root, str(plan["h0_parent_commit"]))
    _validate_fixed_history(root, plan)
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
            raise MigrationValidationError(
                f"sealed site source module drift: {site_id}"
            )
        old_import = planned["old_import"]
        if sorted(recorded.get("symbols", [])) != old_import["symbols"]:
            raise MigrationValidationError(f"sealed site symbols drift: {site_id}")
        if recorded.get("target_module") != old_import["target_module"]:
            raise MigrationValidationError(f"sealed site target drift: {site_id}")
    _validate_import_state(root, commit, sites, state="old_import")
    _expected_h1_sources(root, plan)
    return {
        "migration_id": MIGRATION_ID,
        "h0_parent_commit": commit,
        "h0_seed_commit": H0_SEED_COMMIT,
        "gate_hygiene_commit": GATE_HYGIENE_COMMIT,
        "site_count": len(sites),
        "state": "valid-h0-plan",
    }


def _validate_contract_source(root: Path, commit: str, plan: Mapping[str, Any]) -> None:
    raw = _git_file(root, commit, CONTRACT_PATH.as_posix())
    try:
        tree = ast.parse(raw.decode("utf-8"), filename=CONTRACT_PATH.as_posix())
    except (UnicodeDecodeError, SyntaxError) as exc:
        raise MigrationValidationError(
            f"production contract cannot be parsed: {exc}"
        ) from exc
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            targets = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            targets = [node.module]
        else:
            continue
        if any(
            target == "chronovisor.lab" or target.startswith("chronovisor.lab.")
            for target in targets
        ):
            raise MigrationValidationError(
                "production contract imports chronovisor.lab"
            )
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


def validate_history(
    root: Path, plan: Mapping[str, Any], evidence_parent_revision: str
) -> dict[str, Any]:
    """Validate the fixed source/H0/B/C chain and exact review-fix parent D."""

    validate_plan(root, plan)
    evidence_parent = _commit(root, evidence_parent_revision)
    if _single_parent(root, evidence_parent) != EVIDENCE_HARDENING_COMMIT:
        raise MigrationValidationError("review-fix parent identity drift")
    _require_transition(
        root,
        EVIDENCE_HARDENING_COMMIT,
        evidence_parent,
        {path: "M" for path in REVIEW_FIX_PATHS},
        context="evidence-hardening->review-fix",
    )
    if _git_file(root, evidence_parent, PLAN_PATH.as_posix()) != _canonical_json_bytes(
        plan
    ):
        raise MigrationValidationError(
            "review-fix committed plan differs from supplied canonical plan"
        )
    verifier_path = Path(__file__).resolve()
    try:
        verifier_raw = verifier_path.read_bytes()
    except OSError as exc:
        raise MigrationValidationError(
            f"cannot read current trusted verifier: {exc}"
        ) from exc
    if _git_file(
        root,
        evidence_parent,
        "scripts/architecture_migrations.py",
    ) != verifier_raw:
        raise MigrationValidationError(
            "review-fix verifier differs from the current trusted verifier"
        )
    _validate_contract_source(root, H0_SEED_COMMIT, plan)
    return {
        "migration_id": MIGRATION_ID,
        "evidence_parent_commit": evidence_parent,
        "state": "valid-evidence-history",
    }


def _expected_h1_sources(root: Path, plan: Mapping[str, Any]) -> dict[str, bytes]:
    expected: dict[str, bytes] = {}
    for site in _objects(plan["sites"], context="plan.sites"):
        path = str(site["path"])
        transform = _object(
            site["source_transform"],
            context=f"site {site['semantic_id']}.source_transform",
        )
        h0_raw = _git_file(root, H0_SEED_COMMIT, path)
        if hashlib.sha256(h0_raw).hexdigest() != transform["h0_sha256"]:
            raise MigrationValidationError(f"sealed H0 source digest drift: {path}")
        old = str(transform["old_snippet"]).encode("utf-8")
        new = str(transform["new_snippet"]).encode("utf-8")
        if h0_raw.count(old) != 1:
            raise MigrationValidationError(
                f"sealed H0 source must contain one exact old snippet: {path}"
            )
        h1_raw = h0_raw.replace(old, new, 1)
        if hashlib.sha256(h1_raw).hexdigest() != transform["h1_sha256"]:
            raise MigrationValidationError(f"sealed H1 source digest drift: {path}")
        if path in expected:
            raise MigrationValidationError(f"duplicate H1 source path: {path}")
        expected[path] = h1_raw
    if tuple(sorted(expected)) != H1_PATHS:
        raise MigrationValidationError("sealed H1 source path set drift")
    return expected


def validate_h1(
    root: Path, plan: Mapping[str, Any], h1_revision: str
) -> dict[str, Any]:
    """Validate H1 as the exact five-source transform after the fixed C parent."""

    h1 = _commit(root, h1_revision)
    evidence_parent = _single_parent(root, h1)
    validate_history(root, plan, evidence_parent)
    _require_transition(
        root,
        evidence_parent,
        h1,
        {path: "M" for path in H1_PATHS},
        context="evidence-hardening->H1",
    )
    for path, expected_raw in _expected_h1_sources(root, plan).items():
        if _git_file(root, h1, path) != expected_raw:
            raise MigrationValidationError(f"H1 source bytes differ: {path}")
    return {
        "migration_id": MIGRATION_ID,
        "h0_commit": H0_SEED_COMMIT,
        "evidence_parent_commit": evidence_parent,
        "h1_commit": h1,
        "state": "valid-h1-transition",
    }


def _seed_ids(seed: Mapping[str, Any], field: str, state: str) -> set[str]:
    bucket = _object(seed.get(field), context=f"baseline.{field}")
    return set(_strings(bucket.get(state), context=f"baseline.{field}.{state}"))


def _ordered_seed_ids(baseline: Mapping[str, Any], field: str, state: str) -> list[str]:
    bucket = _object(baseline.get(field), context=f"baseline.{field}")
    values = _strings(bucket.get(state), context=f"baseline.{field}.{state}")
    if values != sorted(values):
        raise MigrationValidationError(f"baseline.{field}.{state} must be sorted")
    return values


def _assert_seed_integrity(baseline: Mapping[str, Any], *, context: str) -> None:
    fields = (
        "exception_semantic_ids",
        "cross_domain_site_semantic_ids",
        "production_to_lab_edge_semantic_ids",
        "production_to_lab_static_site_semantic_ids",
        "production_to_lab_dynamic_site_semantic_ids",
        "compatibility_semantic_ids",
    )
    counts = _object(baseline.get("counts"), context=f"{context}.counts")
    active_counts = _object(counts.get("active"), context=f"{context}.counts.active")
    retired_counts = _object(counts.get("retired"), context=f"{context}.counts.retired")
    count_names = {
        "exception_semantic_ids": "exceptions",
        "cross_domain_site_semantic_ids": "cross_domain_sites",
        "production_to_lab_edge_semantic_ids": "production_to_lab_edges",
        "production_to_lab_static_site_semantic_ids": (
            "production_to_lab_static_sites"
        ),
        "production_to_lab_dynamic_site_semantic_ids": (
            "production_to_lab_dynamic_sites"
        ),
        "compatibility_semantic_ids": "compatibility_contracts",
    }
    for field in fields:
        active = _ordered_seed_ids(baseline, field, "active")
        retired = _ordered_seed_ids(baseline, field, "retired")
        if set(active) & set(retired):
            raise MigrationValidationError(f"{context}.{field} active/retired overlap")
        if active_counts.get(count_names[field]) != len(active):
            raise MigrationValidationError(f"{context}.{field} active count drift")
        if retired_counts.get(field) != len(retired):
            raise MigrationValidationError(f"{context}.{field} retired count drift")
    by_category = _object(
        active_counts.get("by_category"), context=f"{context}.counts.active.by_category"
    )
    if not all(isinstance(value, int) and value >= 0 for value in by_category.values()):
        raise MigrationValidationError(f"{context} category counts are invalid")
    if sum(by_category.values()) != active_counts.get("exceptions"):
        raise MigrationValidationError(f"{context} category counts do not sum")


def _move_seed_ids(
    baseline: dict[str, Any], field: str, expected_ids: Sequence[str]
) -> None:
    bucket = _object(baseline.get(field), context=f"baseline.{field}")
    active = _strings(bucket.get("active"), context=f"baseline.{field}.active")
    retired = _strings(bucket.get("retired"), context=f"baseline.{field}.retired")
    expected = set(expected_ids)
    if expected & set(retired) or not expected <= set(active):
        raise MigrationValidationError(f"sealed H1 {field} retirement membership drift")
    bucket["active"] = sorted(set(active) - expected)
    bucket["retired"] = sorted(set(retired) | expected)


def _expected_h2_payloads(
    h1_baseline_raw: bytes, h1_ledger_raw: bytes
) -> tuple[bytes, bytes]:
    baseline = copy.deepcopy(
        _json_bytes(
            h1_baseline_raw,
            context="H1 baseline",
            canonical_document=True,
        )
    )
    ledger = copy.deepcopy(
        _json_bytes(h1_ledger_raw, context="H1 ledger", canonical_document=True)
    )
    _assert_seed_integrity(baseline, context="H1 baseline")

    _move_seed_ids(baseline, "exception_semantic_ids", [PRIVATE_EXCEPTION_ID])
    _move_seed_ids(baseline, "cross_domain_site_semantic_ids", MIGRATED_SITE_IDS)
    _move_seed_ids(
        baseline,
        "production_to_lab_static_site_semantic_ids",
        MIGRATED_SITE_IDS,
    )
    baseline_counts = _object(baseline["counts"], context="baseline.counts")
    baseline_active = _object(
        baseline_counts["active"], context="baseline.counts.active"
    )
    baseline_retired = _object(
        baseline_counts["retired"], context="baseline.counts.retired"
    )
    baseline_active["exceptions"] = len(
        _ordered_seed_ids(baseline, "exception_semantic_ids", "active")
    )
    baseline_active["cross_domain_sites"] = len(
        _ordered_seed_ids(baseline, "cross_domain_site_semantic_ids", "active")
    )
    baseline_active["production_to_lab_static_sites"] = len(
        _ordered_seed_ids(
            baseline, "production_to_lab_static_site_semantic_ids", "active"
        )
    )
    baseline_by_category = _object(
        baseline_active["by_category"], context="baseline.counts.active.by_category"
    )
    private_count = baseline_by_category.get("private_symbol_import")
    if not isinstance(private_count, int) or private_count < 1:
        raise MigrationValidationError("sealed H1 private exception count drift")
    baseline_by_category["private_symbol_import"] = private_count - 1
    for field in EXPECTED_H2_RETIRED_COUNTS:
        baseline_retired[field] = len(_ordered_seed_ids(baseline, field, "retired"))
    if _selected_counts(baseline_active) != EXPECTED_H2_ACTIVE_COUNTS:
        raise MigrationValidationError("derived H2 active count invariant failed")
    if baseline_retired != EXPECTED_H2_RETIRED_COUNTS:
        raise MigrationValidationError("derived H2 retired count invariant failed")
    _assert_seed_integrity(baseline, context="derived H2 baseline")

    exceptions = _objects(ledger.get("exceptions"), context="H1 ledger.exceptions")
    private_rows = [
        row for row in exceptions if row.get("semantic_id") == PRIVATE_EXCEPTION_ID
    ]
    if len(private_rows) != 1:
        raise MigrationValidationError("sealed H1 private ledger row drift")
    private_category = private_rows[0].get("category")
    if private_category != "private_symbol_import":
        raise MigrationValidationError("sealed H1 private ledger category drift")
    ledger["exceptions"] = [
        row for row in exceptions if row.get("semantic_id") != PRIVATE_EXCEPTION_ID
    ]
    exception_by_id, _site_by_id = _find_rows(ledger)
    edge = exception_by_id.get(CLASSIFICATION_LAB_EDGE_ID)
    if edge is None:
        raise MigrationValidationError("sealed H1 classification edge is absent")
    sites = _objects(edge.get("sites"), context="H1 classification edge.sites")
    site_ids = [site.get("semantic_id") for site in sites]
    if any(site_ids.count(site_id) != 1 for site_id in MIGRATED_SITE_IDS):
        raise MigrationValidationError("sealed H1 migrated edge site membership drift")
    if site_ids.count(PROVIDER_SITE_ID) != 1:
        raise MigrationValidationError("sealed H1 provider site membership drift")
    edge["sites"] = [
        site for site in sites if site.get("semantic_id") not in set(MIGRATED_SITE_IDS)
    ]
    if [site.get("semantic_id") for site in edge["sites"]] != [PROVIDER_SITE_ID]:
        raise MigrationValidationError("derived H2 classification edge site drift")

    ledger_counts = _object(ledger.get("counts"), context="H1 ledger.counts")
    ledger_counts["exceptions"] = baseline_active["exceptions"]
    ledger_counts["cross_domain_sites"] = baseline_active["cross_domain_sites"]
    ledger_counts["production_to_lab_static_sites"] = baseline_active[
        "production_to_lab_static_sites"
    ]
    ledger_by_category = _object(
        ledger_counts.get("by_category"), context="H1 ledger.counts.by_category"
    )
    ledger_by_category["private_symbol_import"] = baseline_by_category[
        "private_symbol_import"
    ]
    if ledger_counts != baseline_active:
        raise MigrationValidationError("derived H2 ledger count structure drift")
    ledger["baseline_sha256"] = hashlib.sha256(
        json.dumps(
            baseline,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    _find_rows(ledger)
    return _document_json_bytes(baseline), _document_json_bytes(ledger)


def _validate_h2_payloads(
    h1_baseline_raw: bytes,
    h1_ledger_raw: bytes,
    h2_baseline_raw: bytes,
    h2_ledger_raw: bytes,
) -> None:
    _json_bytes(h2_baseline_raw, context="H2 baseline", canonical_document=True)
    _json_bytes(h2_ledger_raw, context="H2 ledger", canonical_document=True)
    expected_baseline, expected_ledger = _expected_h2_payloads(
        h1_baseline_raw, h1_ledger_raw
    )
    if h2_baseline_raw != expected_baseline:
        raise MigrationValidationError("H2 baseline is not the exact sealed transform")
    if h2_ledger_raw != expected_ledger:
        raise MigrationValidationError("H2 ledger is not the exact sealed transform")


def _parse_name_status(raw: bytes, *, context: str) -> dict[str, str]:
    fields = [field for field in raw.split(b"\0") if field]
    if len(fields) % 2:
        raise MigrationValidationError(f"malformed {context} name-status output")
    entries: dict[str, str] = {}
    for index in range(0, len(fields), 2):
        try:
            status = fields[index].decode("ascii")
            path = fields[index + 1].decode("utf-8")
        except UnicodeDecodeError as exc:
            raise MigrationValidationError(f"noncanonical {context} path") from exc
        _relative_path(path, context=f"{context} path")
        if status not in {"A", "M", "D"} or path in entries:
            raise MigrationValidationError(f"unsupported or duplicate {context} entry")
        entries[path] = status
    return entries


def _index_file(root: Path, path: str) -> bytes:
    canonical = _relative_path(path, context="Git index path")
    output = _git(root, "ls-files", "--stage", "-z", "--", canonical)
    records = [record for record in output.split(b"\0") if record]
    if len(records) != 1 or b"\t" not in records[0]:
        raise MigrationValidationError(f"{canonical} is not one Git index entry")
    metadata, raw_path = records[0].split(b"\t", 1)
    try:
        mode, oid, stage = metadata.decode("ascii").split()
        recorded_path = raw_path.decode("utf-8")
    except (UnicodeDecodeError, ValueError) as exc:
        raise MigrationValidationError(
            f"malformed Git index entry for {canonical}"
        ) from exc
    if recorded_path != canonical or mode != "100644" or stage != "0":
        raise MigrationValidationError(
            f"{canonical} must be one stage-0 exact 100644 index blob"
        )
    if not _BLOB_RE.fullmatch(oid):
        raise MigrationValidationError(f"invalid index blob identity for {canonical}")
    return _git(root, "cat-file", "blob", oid)


def _current_head(root: Path) -> str:
    resolved = _git(root, "rev-parse", "--verify", "HEAD^{commit}").decode().strip()
    if not _COMMIT_RE.fullmatch(resolved):
        raise MigrationValidationError("current HEAD is not a canonical commit")
    return resolved


def _validate_exact_counts(
    value: Any, expected: Mapping[str, Any], *, context: str
) -> dict[str, Any]:
    counts = _object(value, context=context)
    _require_keys(counts, set(expected), context=context)
    for key, expected_value in expected.items():
        actual_value = counts[key]
        if isinstance(expected_value, Mapping):
            _validate_exact_counts(
                actual_value,
                expected_value,
                context=f"{context}.{key}",
            )
        elif type(actual_value) is not int or actual_value != expected_value:
            raise MigrationValidationError(f"{context}.{key} count drift")
    return counts


def _validate_receipt_object(
    value: Any, *, expected_plan_sha256: str
) -> dict[str, Any]:
    receipt = _object(value, context="receipt")
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
    if (
        not isinstance(expected_plan_sha256, str)
        or not _HEX_SHA_RE.fullmatch(expected_plan_sha256)
        or expected_plan_sha256 != CANONICAL_PLAN_SHA256
    ):
        raise MigrationValidationError("expected plan SHA is invalid")
    if receipt["plan_sha256"] != expected_plan_sha256:
        raise MigrationValidationError("receipt does not bind the canonical plan")
    if not isinstance(receipt["h1_commit"], str) or not _COMMIT_RE.fullmatch(
        receipt["h1_commit"]
    ):
        raise MigrationValidationError("receipt H1 commit is invalid")
    _validate_exact_counts(
        receipt["active_counts"],
        EXPECTED_H2_ACTIVE_COUNTS,
        context="receipt.active_counts",
    )
    _validate_exact_counts(
        receipt["retired_counts"],
        EXPECTED_H2_RETIRED_COUNTS,
        context="receipt.retired_counts",
    )
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
        _require_keys(
            binding,
            {"path", "sha256"},
            context=f"receipt.h2_artifacts.{name}",
        )
        if binding["path"] != path.as_posix():
            raise MigrationValidationError(f"receipt {name} path is invalid")
        digest = binding["sha256"]
        if not isinstance(digest, str) or not _HEX_SHA_RE.fullmatch(digest):
            raise MigrationValidationError(f"receipt {name} SHA is invalid")
    recorded_seal = receipt["receipt_sha256"]
    if not isinstance(recorded_seal, str) or not _HEX_SHA_RE.fullmatch(recorded_seal):
        raise MigrationValidationError("receipt self-seal is invalid")
    if recorded_seal != _seal(receipt, "receipt_sha256"):
        raise MigrationValidationError("receipt self-seal mismatch")
    return receipt


def build_receipt(
    root: Path, plan: Mapping[str, Any], h1_revision: str
) -> dict[str, Any]:
    """Build an H2 receipt only from the exact staged H2 artifact blobs."""

    h1_report = validate_h1(root, plan, h1_revision)
    h1 = str(h1_report["h1_commit"])
    if _current_head(root) != h1:
        raise MigrationValidationError(
            "current HEAD must be the supplied full H1 commit"
        )
    staged = _parse_name_status(
        _git(
            root,
            "diff",
            "--cached",
            "--name-status",
            "-z",
            "--no-renames",
            h1,
        ),
        context="staged H2",
    )
    expected_staged = {
        BASELINE_PATH.as_posix(): "M",
        LEDGER_PATH.as_posix(): "M",
    }
    if staged != expected_staged:
        raise MigrationValidationError(
            f"staged H2 path/status scope mismatch: {staged}"
        )
    unstaged = _git(
        root,
        "diff",
        "--name-only",
        "-z",
        "--",
        BASELINE_PATH.as_posix(),
        LEDGER_PATH.as_posix(),
    )
    if unstaged:
        raise MigrationValidationError("H2 staged artifacts have unstaged differences")
    baseline_raw = _index_file(root, BASELINE_PATH.as_posix())
    ledger_raw = _index_file(root, LEDGER_PATH.as_posix())
    h1_baseline_raw = _git_file(root, h1, BASELINE_PATH.as_posix())
    h1_ledger_raw = _git_file(root, h1, LEDGER_PATH.as_posix())
    _validate_h2_payloads(
        h1_baseline_raw,
        h1_ledger_raw,
        baseline_raw,
        ledger_raw,
    )
    payload: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "migration_id": MIGRATION_ID,
        "plan_sha256": plan["plan_sha256"],
        "h1_commit": h1,
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
        "active_counts": dict(EXPECTED_H2_ACTIVE_COUNTS),
        "retired_counts": dict(EXPECTED_H2_RETIRED_COUNTS),
        "retired_exception_ids": [PRIVATE_EXCEPTION_ID],
        "retired_site_ids": list(MIGRATED_SITE_IDS),
        "p3_retained_edge_ids": [CLASSIFICATION_LAB_EDGE_ID],
        "p3_retained_site_ids": [PROVIDER_SITE_ID],
    }
    payload["receipt_sha256"] = _seal(payload, "receipt_sha256")
    return _validate_receipt_object(
        payload,
        expected_plan_sha256=str(plan["plan_sha256"]),
    )


def load_receipt(path: Path) -> dict[str, Any]:
    """Load and strictly validate a canonical H2 receipt."""

    receipt = _load_canonical(path, seal_field="receipt_sha256")
    return _validate_receipt_object(
        receipt,
        expected_plan_sha256=CANONICAL_PLAN_SHA256,
    )


def _receipt_additions(root: Path, h1: str, tip: str, receipt_path: str) -> list[str]:
    commits = (
        _git(
            root,
            "rev-list",
            "--first-parent",
            "--ancestry-path",
            f"{h1}..{tip}",
        )
        .decode()
        .splitlines()
    )
    additions = []
    for commit in commits:
        parent = _single_parent(root, commit)
        status = (
            _git(
                root,
                "diff-tree",
                "--no-commit-id",
                "--name-status",
                "-r",
                parent,
                commit,
                "--",
                receipt_path,
            )
            .decode()
            .strip()
        )
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

    plan_sha256 = plan.get("plan_sha256")
    receipt = _validate_receipt_object(
        receipt,
        expected_plan_sha256=plan_sha256,
    )
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
    _require_transition(
        root,
        h1,
        h2,
        {
            BASELINE_PATH.as_posix(): "M",
            LEDGER_PATH.as_posix(): "M",
            RECEIPT_PATH.as_posix(): "A",
        },
        context="H1->H2",
    )
    committed_receipt = _git_file(root, h2, RECEIPT_PATH.as_posix())
    if committed_receipt != _canonical_json_bytes(receipt):
        raise MigrationValidationError("committed H2 receipt bytes differ")
    artifacts = receipt["h2_artifacts"]
    h1_baseline_raw = _git_file(root, h1, BASELINE_PATH.as_posix())
    h1_ledger_raw = _git_file(root, h1, LEDGER_PATH.as_posix())
    baseline_raw = _git_file(root, h2, BASELINE_PATH.as_posix())
    ledger_raw = _git_file(root, h2, LEDGER_PATH.as_posix())
    for name, raw in (("baseline", baseline_raw), ("ledger", ledger_raw)):
        if hashlib.sha256(raw).hexdigest() != artifacts[name]["sha256"]:
            raise MigrationValidationError(f"committed H2 {name} digest differs")
    _validate_h2_payloads(
        h1_baseline_raw,
        h1_ledger_raw,
        baseline_raw,
        ledger_raw,
    )
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


def _untracked_output_path(root: Path, value: Path) -> tuple[Path, str]:
    candidate = value if value.is_absolute() else root / value
    if candidate.is_symlink():
        raise MigrationValidationError("receipt output must not be a symlink")
    if candidate.exists():
        raise MigrationValidationError("receipt output must not already exist")
    resolved = candidate.resolve()
    try:
        relative = resolved.relative_to(root).as_posix()
    except ValueError as exc:
        raise MigrationValidationError(
            "receipt output must stay inside the repository"
        ) from exc
    _relative_path(relative, context="receipt output")
    if _git(root, "ls-files", "-z", "--", relative):
        raise MigrationValidationError(
            "receipt output must be untracked before generation"
        )
    return resolved, relative


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).parents[1])
    parser.add_argument("--plan", type=Path, default=PLAN_PATH)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("validate-plan")
    history = commands.add_parser("validate-history")
    history.add_argument("--evidence-parent", required=True)
    h1 = commands.add_parser("validate-h1")
    h1.add_argument("--h1", required=True)
    build = commands.add_parser("build-receipt")
    build.add_argument("--h1", required=True)
    build.add_argument("--output", type=Path, default=RECEIPT_PATH)
    verify = commands.add_parser("verify-receipt")
    verify.add_argument("--receipt", type=Path, default=RECEIPT_PATH)
    verify.add_argument("--tip", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.root.resolve()
    plan_path = args.plan if args.plan.is_absolute() else root / args.plan
    try:
        plan = load_plan(plan_path)
        if args.command == "validate-plan":
            report = validate_plan(root, plan)
        elif args.command == "validate-history":
            report = validate_history(root, plan, args.evidence_parent)
        elif args.command == "validate-h1":
            report = validate_h1(root, plan, args.h1)
        elif args.command == "build-receipt":
            receipt = build_receipt(root, plan, args.h1)
            output, output_relative = _untracked_output_path(root, args.output)
            write_canonical_json(output, receipt)
            report = {
                "migration_id": MIGRATION_ID,
                "output": output_relative,
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
