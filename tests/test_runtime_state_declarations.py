from __future__ import annotations

import ast
import hashlib
import inspect
from collections import Counter
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

import pytest

from scripts.runtime_ownership import declarations
from scripts.runtime_ownership.declarations import (
    EFFECTIVE_FEEDBACK_REVISION,
    ConcreteDeclarations,
    DeclarationError,
    apply_resource_amendment,
    discover_concrete,
    load_frozen,
    record_protocol_transition,
)
from scripts.runtime_ownership.manifests import (
    FROZEN_SOURCE_REVISION,
    SOURCE_MANIFEST_KIND,
    CommittedFile,
    CommittedSnapshot,
    committed_snapshot,
)

ROOT = Path(__file__).resolve().parents[1]
RESOURCE_FIELDS = {"id", "kind", "locator", "protects", "evidence", "discovery_ids"}
FORBIDDEN_FIELDS = {
    "owner",
    "owner_package",
    "owner_symbol",
    "writers",
    "readers",
    "lifecycle",
    "coordination",
}
FROZEN_HASHES = {
    "rows": "3a8291d534e65d4dadbdb1c1b4b3d7e9e03eb465fc883471bc0b4c08824883d0",
    "resource_candidates": "c7214ebb7fd4d84be5ac0aad4fed9eb99dc09542340981568263a072fac270f5",
    "exclusion_candidates": "5e40a4142e3a2abd1c283ca836eec8ac72cd1eb376e6a34c9dbebbd96c6ede82",
    "lock_protocol_candidates": "3bffbd1791c08c8b75cd5960617b3e3d969b1ac22a8c391a367ebf1c985a871e",
    "resources": "3a77631467721e0a261a419b95e6e6fdc23e889f5af4dd832ca9f6f742f1d326",
}


@pytest.fixture(scope="module")
def frozen_snapshot() -> CommittedSnapshot:
    return committed_snapshot(
        ROOT,
        FROZEN_SOURCE_REVISION,
        manifest_kind=SOURCE_MANIFEST_KIND,
    )


@pytest.fixture(scope="module")
def effective_snapshot() -> CommittedSnapshot:
    return committed_snapshot(
        ROOT,
        EFFECTIVE_FEEDBACK_REVISION,
        manifest_kind=SOURCE_MANIFEST_KIND,
    )


@pytest.fixture(scope="module")
def frozen(frozen_snapshot: CommittedSnapshot) -> ConcreteDeclarations:
    return load_frozen(frozen_snapshot)


def _mapping(snapshot: CommittedSnapshot) -> dict[str, bytes]:
    return {row.path: row.raw_bytes for row in snapshot.files}


def _blob_oid(raw: bytes) -> str:
    return hashlib.sha1(
        f"blob {len(raw)}\0".encode("ascii") + raw,
        usedforsecurity=False,
    ).hexdigest()


def _replace_blob(
    snapshot: CommittedSnapshot, path: str, raw: bytes
) -> CommittedSnapshot:
    files = tuple(
        replace(row, raw_bytes=raw, blob_oid=_blob_oid(raw))
        if row.path == path
        else row
        for row in snapshot.files
    )
    assert files != snapshot.files
    return replace(snapshot, files=files)


def _id_hash(rows: tuple[Mapping[str, Any], ...], field: str) -> str:
    return hashlib.sha256(
        declarations._canonical_bytes(sorted(str(row[field]) for row in rows))
    ).hexdigest()


def _assert_forbidden_fields_absent(value: Any) -> None:
    if isinstance(value, Mapping):
        assert FORBIDDEN_FIELDS.isdisjoint(value)
        for item in value.values():
            _assert_forbidden_fields_absent(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _assert_forbidden_fields_absent(item)


def _assert_no_mutable_json_containers(value: Any) -> None:
    if isinstance(value, Mapping):
        assert not isinstance(value, dict)
        for item in value.values():
            _assert_no_mutable_json_containers(item)
    elif isinstance(value, tuple):
        for item in value:
            _assert_no_mutable_json_containers(item)
    else:
        assert not isinstance(value, list)


def test_forbidden_field_probe_traverses_nested_mapping_proxies() -> None:
    nested = MappingProxyType(
        {"outer": (MappingProxyType({"owner_symbol": "forbidden"}),)}
    )

    with pytest.raises(AssertionError):
        _assert_forbidden_fields_absent(nested)


def test_canonicalization_and_freeze_reject_container_cycles() -> None:
    mapping_cycle: dict[str, Any] = {}
    mapping_cycle["self"] = mapping_cycle
    list_cycle: list[Any] = []
    list_cycle.append(list_cycle)
    tuple_bridge: list[Any] = []
    tuple_cycle = (tuple_bridge,)
    tuple_bridge.append(tuple_cycle)

    for cycle in (mapping_cycle, list_cycle, tuple_cycle):
        with pytest.raises(DeclarationError, match="container cycle"):
            declarations._plain_json(cycle)
        with pytest.raises(DeclarationError, match="container cycle"):
            declarations._deep_freeze(cycle)
        with pytest.raises(DeclarationError, match="container cycle"):
            declarations._canonical_bytes(cycle)


def test_freeze_requires_exact_string_keys_and_copies_shared_containers() -> None:
    class StringSubclass(str):
        pass

    invalid = {StringSubclass("key"): "value"}
    with pytest.raises(DeclarationError, match="keys must be strings"):
        declarations._plain_json(invalid)
    with pytest.raises(DeclarationError, match="keys must be strings"):
        declarations._deep_freeze(invalid)

    shared = {"nested": [1]}
    frozen = cast(
        Mapping[str, Any],
        declarations._deep_freeze({"first": shared, "second": shared}),
    )
    assert frozen["first"] == frozen["second"]
    assert frozen["first"] is not frozen["second"]


def test_frozen_concrete_counts_id_sets_and_resource_kinds(
    frozen: ConcreteDeclarations,
) -> None:
    collections = {
        "rows": frozen.rows,
        "resource_candidates": frozen.resource_candidates,
        "exclusion_candidates": frozen.exclusion_candidates,
        "lock_protocol_candidates": frozen.lock_protocol_candidates,
        "resources": frozen.resources,
    }
    assert {name: len(rows) for name, rows in collections.items()} == {
        "rows": 732,
        "resource_candidates": 490,
        "exclusion_candidates": 138,
        "lock_protocol_candidates": 104,
        "resources": 454,
    }
    assert {
        name: _id_hash(rows, "id" if name == "resources" else "discovery_id")
        for name, rows in collections.items()
    } == FROZEN_HASHES
    assert Counter(str(row["kind"]) for row in frozen.resources) == Counter(
        {
            "artifact": 182,
            "lock": 21,
            "queue": 5,
            "schema": 154,
            "socket": 6,
            "worker": 86,
        }
    )


def test_source_and_declaration_full_seals_are_exact(
    frozen_snapshot: CommittedSnapshot,
    effective_snapshot: CommittedSnapshot,
    frozen: ConcreteDeclarations,
) -> None:
    assert declarations._snapshot_files_sha256(frozen_snapshot) == (
        "6693cc159f8ab213a513225b73096a30e4ae629404d6b5b7906d63cb6a52e4ef"
    )
    assert declarations._snapshot_files_sha256(effective_snapshot) == (
        "be2ad06f687bc619a89d12ad6274d6843b26278e2094d420146105c398e73cee"
    )
    assert declarations._declaration_sha256(frozen) == (
        "03c456e12a698dedf78f2618851b18c1916b1eb45f1b326b665af6959e8bccef"
    )


def test_frozen_direct_flock_shape_is_exact(frozen: ConcreteDeclarations) -> None:
    direct = [
        row
        for row in frozen.lock_protocol_candidates
        if not str(row["operation"]).startswith("helper:")
    ]
    assert len(direct) == 52
    assert len({str(row["module"]) for row in direct}) == 37
    assert len({(str(row["module"]), str(row["scope"])) for row in direct}) == 51


def test_resources_are_structural_only_and_references_are_known(
    frozen: ConcreteDeclarations,
) -> None:
    identifiers = {str(row["id"]) for row in frozen.resources}
    assert len(identifiers) == 454
    for resource in frozen.resources:
        assert set(resource) == RESOURCE_FIELDS
        assert set(str(item) for item in resource["protects"]) <= identifiers
    _assert_forbidden_fields_absent(frozen.__dict__)


def test_resource_id_uses_kind_and_locator_value_string_only(
    frozen: ConcreteDeclarations,
) -> None:
    resource = next(
        row
        for row in frozen.resources
        if row["locator"]["value"] == "$CHRONOVISOR_ROOT/recall/feedback.jsonl"
    )
    assert resource["id"] == (
        "runtime-resource:8dcd7e44878e638768b6866bb2b002ff59e5c75a9cecba7a8d54f4794897a417"
    )
    assert resource["id"] == declarations._resource_id(
        str(resource["kind"]), str(resource["locator"]["value"])
    )


@pytest.mark.parametrize(
    ("locator", "resource_id", "discovery_id"),
    [
        (
            "$CHRONOVISOR_ROOT/recall/search-label-queue.jsonl.lock",
            "runtime-resource:1302dc2326a49adb03cca5821456e4d62e02df9b3b0e9147c051fd2cab828f1b",
            "runtime-site:03e1d780beb5b697d1ee468cd3ea3173b5d1a2ec4b287bb59516a5eeb27de1b8",
        ),
        (
            "$CHRONOVISOR_ROOT/claims/claims.jsonl.lock",
            "runtime-resource:4e71a3e2f7c2d6a5c5e5a042258644431d8452ae10610246ca8d6c302d9c68ca",
            "runtime-site:bec3ffaa1407e14ad9738fdd4e04c91f21cb55c2e12110bbfc2b00ea7d0543a4",
        ),
    ],
)
def test_golden_and_claims_sidecar_resources_have_exact_ids(
    frozen: ConcreteDeclarations,
    locator: str,
    resource_id: str,
    discovery_id: str,
) -> None:
    resource = next(
        row for row in frozen.resources if row["locator"]["value"] == locator
    )
    assert resource["id"] == resource_id
    assert resource["discovery_ids"] == (discovery_id,)


def test_golden_and_claims_concrete_lock_protocol_ids_replace_planned_ids(
    frozen: ConcreteDeclarations,
) -> None:
    protocol_ids = {str(row["discovery_id"]) for row in frozen.lock_protocol_candidates}

    assert {
        "runtime-site:c6e259acaafa9b9c486f879fb6c830b1b22c4da08eb45586128c7125eed9a4e7",
        "runtime-site:dfcc8b463ad3bd770ad1a15b1a3bb477507b909a44c0d2ae73b080287e6957c9",
        "runtime-site:1b61419be70e7739ba3570e71687ac6dd92caac999c8a5e141794765fbd2f875",
        "runtime-site:c707eb3f9c7370b352af3f5bb618f4a2b1ef91833dac08cf9660e5166df3ee91",
        "runtime-site:5f1b276d76fc77c041cf2a17a8aaa1711189ad1d94868d101ed9df9966ad2566",
    } <= protocol_ids
    assert {
        "runtime-site:95d55f5f445e177214f681ba29e79f38c792043fea9201fe13670e5201c19f6e",
        "runtime-site:169a475077d3631793e280c24569085f7427fe5090532d6e777687b4cac126cc",
    }.isdisjoint(protocol_ids)


def test_same_resource_candidates_are_explicitly_grouped_and_merged(
    frozen: ConcreteDeclarations,
) -> None:
    resource = next(
        row
        for row in frozen.resources
        if row["locator"]["value"] == "$CHRONOVISOR_ROOT/.embeddings.json"
    )
    candidates = [
        row
        for row in frozen.resource_candidates
        if row["kind"] == resource["kind"]
        and row["locator"]["value"] == resource["locator"]["value"]
    ]
    assert len(candidates) == 2
    assert resource["discovery_ids"] == tuple(
        sorted(str(row["discovery_id"]) for row in candidates)
    )
    assert resource["evidence"] == tuple(
        sorted(
            (
                {"path": str(row["path"]), "line": int(row["line"])}
                for row in candidates
            ),
            key=lambda row: (row["path"], row["line"]),
        )
    )


def test_discovery_has_no_include_planned_api_or_rows(
    frozen: ConcreteDeclarations,
) -> None:
    assert "include_planned" not in inspect.signature(discover_concrete).parameters
    assert all("planned" not in row for row in frozen.rows)


def test_mapping_input_is_explicit_deterministic_and_does_not_read_worktree(
    frozen_snapshot: CommittedSnapshot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _mapping(frozen_snapshot)
    reversed_source = dict(reversed(tuple(source.items())))

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("cwd, HEAD, git, and worktree reads are forbidden")

    monkeypatch.setattr(Path, "read_bytes", forbidden)
    monkeypatch.setattr(Path, "glob", forbidden)
    monkeypatch.setattr(Path, "rglob", forbidden)
    first = discover_concrete(source)
    second = discover_concrete(reversed_source)

    assert first.source_head is None
    assert first.rows == second.rows
    assert first.resources == second.resources


def test_discovery_ids_are_line_independent(frozen_snapshot: CommittedSnapshot) -> None:
    source = _mapping(frozen_snapshot)
    shifted = dict(source)
    path = "src/chronovisor/recall/recall_runtime.py"
    shifted[path] = b"\n" + shifted[path]

    original = discover_concrete(source)
    changed = discover_concrete(shifted)

    assert {str(row["discovery_id"]) for row in changed.rows} == {
        str(row["discovery_id"]) for row in original.rows
    }
    assert {str(row["id"]) for row in changed.resources} == {
        str(row["id"]) for row in original.resources
    }
    shifted_row = next(
        row
        for row in changed.rows
        if row["module"] == "chronovisor.recall.recall_runtime"
        and row["symbol"] == "RECALL_FEEDBACK_FILE"
    )
    original_row = next(
        row
        for row in original.rows
        if row["module"] == "chronovisor.recall.recall_runtime"
        and row["symbol"] == "RECALL_FEEDBACK_FILE"
    )
    assert shifted_row["line"] == original_row["line"] + 1


@pytest.mark.parametrize(
    "source",
    [
        {},
        {"/absolute.py": b""},
        {"../escape.py": b""},
        {"pyproject.toml": "not bytes"},
    ],
)
def test_invalid_mapping_inputs_fail_closed(source: Any) -> None:
    with pytest.raises(DeclarationError):
        discover_concrete(source)


@pytest.mark.parametrize(
    "path",
    [
        "./pyproject.toml",
        "src//chronovisor/extra.py",
        "src/chronovisor/./extra.py",
        "src/chronovisor/../extra.py",
        "src/chronovisor/extra.py/",
    ],
)
def test_mapping_path_aliases_fail_closed(path: str) -> None:
    with pytest.raises(DeclarationError, match="canonical relative path"):
        discover_concrete({path: b""})


def test_forged_committed_snapshot_bytes_fail_closed(
    frozen_snapshot: CommittedSnapshot,
) -> None:
    first = frozen_snapshot.files[0]
    forged = replace(
        frozen_snapshot,
        files=(
            replace(first, raw_bytes=first.raw_bytes + b"forged"),
            *frozen_snapshot.files[1:],
        ),
    )

    with pytest.raises(DeclarationError, match="not verified"):
        discover_concrete(forged)


def test_forged_frozen_snapshots_with_resealed_blobs_fail_manifest_seal(
    frozen_snapshot: CommittedSnapshot,
) -> None:
    extra_raw = b"FORGED = True\n"
    extra = CommittedFile(
        path="src/chronovisor/zz_forged.py",
        git_mode="100644",
        git_type="blob",
        blob_oid=_blob_oid(extra_raw),
        raw_bytes=extra_raw,
    )
    extra_snapshot = replace(
        frozen_snapshot,
        files=tuple(sorted((*frozen_snapshot.files, extra), key=lambda row: row.path)),
    )
    pyproject = _mapping(frozen_snapshot)["pyproject.toml"] + b"\n# forged\n"
    calibration_path = "src/chronovisor/recall/recall_runtime.py"
    calibration = _mapping(frozen_snapshot)[calibration_path].replace(
        b'RECALL_CALIBRATION_FILE = RECALL_DIR / "calibration.json"',
        b'RECALL_CALIBRATION_FILE = RECALL_DIR / "calibration-forged.json"',
    )
    assert calibration != _mapping(frozen_snapshot)[calibration_path]

    for forged in (
        extra_snapshot,
        _replace_blob(frozen_snapshot, "pyproject.toml", pyproject),
        _replace_blob(frozen_snapshot, calibration_path, calibration),
    ):
        with pytest.raises(DeclarationError, match="files manifest seal mismatch"):
            load_frozen(forged)


def test_effective_amendment_and_transition_reject_resealed_unrelated_changes(
    frozen: ConcreteDeclarations,
    effective_snapshot: CommittedSnapshot,
) -> None:
    forged = _replace_blob(
        effective_snapshot,
        "pyproject.toml",
        _mapping(effective_snapshot)["pyproject.toml"] + b"\n# forged\n",
    )

    with pytest.raises(DeclarationError, match="files manifest seal mismatch"):
        apply_resource_amendment(frozen, forged)
    with pytest.raises(DeclarationError, match="files manifest seal mismatch"):
        record_protocol_transition(frozen, forged)


def test_committed_snapshot_field_and_file_metadata_types_fail_closed(
    frozen_snapshot: CommittedSnapshot,
) -> None:
    first = frozen_snapshot.files[0]
    invalid = (
        replace(frozen_snapshot, revision=cast(Any, frozen_snapshot.revision.encode())),
        replace(frozen_snapshot, git_object_format=cast(Any, b"sha1")),
        replace(frozen_snapshot, files=cast(Any, list(frozen_snapshot.files))),
        replace(
            frozen_snapshot,
            files=(cast(Any, {"path": first.path}), *frozen_snapshot.files[1:]),
        ),
        replace(
            frozen_snapshot,
            files=(replace(first, path=f"./{first.path}"), *frozen_snapshot.files[1:]),
        ),
        replace(
            frozen_snapshot,
            files=(
                replace(first, git_mode=cast(Any, 100644)),
                *frozen_snapshot.files[1:],
            ),
        ),
        replace(
            frozen_snapshot,
            files=(
                replace(first, raw_bytes=cast(Any, bytearray(first.raw_bytes))),
                *frozen_snapshot.files[1:],
            ),
        ),
    )

    for snapshot in invalid:
        with pytest.raises(DeclarationError):
            discover_concrete(snapshot)


def test_load_frozen_rejects_nonfrozen_revision(
    effective_snapshot: CommittedSnapshot,
) -> None:
    with pytest.raises(DeclarationError, match="exact revision"):
        load_frozen(effective_snapshot)


def test_duplicate_discovery_ids_fail_instead_of_last_write_wins(
    frozen_snapshot: CommittedSnapshot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = declarations._ast_discovery

    def duplicate(index: Any) -> list[dict[str, Any]]:
        rows = original(index)
        return [*rows, dict(rows[0])]

    monkeypatch.setattr(declarations, "_ast_discovery", duplicate)
    with pytest.raises(DeclarationError, match="duplicate discovery ids"):
        discover_concrete(frozen_snapshot)


def test_duplicate_final_resource_ids_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = {"one.py": b"a\nb\n"}
    candidates = [
        {
            "kind": "artifact",
            "locator": {
                "type": "path",
                "value": f"$CHRONOVISOR_ROOT/test/{name}",
            },
            "path": "one.py",
            "line": line,
            "discovery_id": f"runtime-site:{name * 64}",
        }
        for name, line in (("a", 1), ("b", 2))
    ]
    monkeypatch.setattr(declarations, "_resource_id", lambda _kind, _value: "same")

    with pytest.raises(DeclarationError, match="duplicate final resource"):
        declarations._group_resources(snapshot, candidates)


def test_unknown_protects_reference_fails_closed(
    frozen_snapshot: CommittedSnapshot,
    frozen: ConcreteDeclarations,
) -> None:
    target = ("artifact", "$CHRONOVISOR_ROOT/claims/claims.jsonl")
    candidates = [
        row
        for row in frozen.resource_candidates
        if (str(row["kind"]), str(row["locator"]["value"])) != target
    ]

    with pytest.raises(DeclarationError, match="unknown resource references"):
        declarations._group_resources(_mapping(frozen_snapshot), candidates)


@pytest.mark.parametrize(
    "evidence",
    [
        {"path": "missing.py", "line": 1},
        {"path": "one.py", "line": 0},
        {"path": "one.py", "line": 2},
        {"path": "one.py", "line": 1, "extra": True},
    ],
)
def test_missing_or_invalid_evidence_fails_closed(evidence: dict[str, Any]) -> None:
    with pytest.raises(DeclarationError):
        declarations._validate_evidence({"one.py": b"one\n"}, evidence)


def test_malformed_resource_candidates_fail_as_declaration_errors() -> None:
    valid = {
        "kind": "artifact",
        "locator": {
            "type": "path",
            "value": "$CHRONOVISOR_ROOT/test/artifact.json",
        },
        "path": "pyproject.toml",
        "line": 1,
        "discovery_id": f"runtime-site:{'a' * 64}",
    }
    invalid: list[dict[str, Any]] = []
    for field, value in (
        ("kind", []),
        ("kind", True),
        ("discovery_id", None),
        ("discovery_id", True),
        ("discovery_id", []),
        ("line", True),
        ("line", 1.5),
        ("line", "1"),
    ):
        row = dict(valid)
        row[field] = value
        invalid.append(row)
    for locator in (
        {"type": "path", "value": "/tmp/absolute"},
        {"type": [], "value": "$CHRONOVISOR_ROOT/test/artifact.json"},
        {"type": "path", "value": "$CHRONOVISOR_ROOT/test/../artifact.json"},
        {"type": "path"},
    ):
        row = dict(valid)
        row["locator"] = locator
        invalid.append(row)
    for evidence in (
        {"path": "pyproject.toml", "line": True},
        {"path": "pyproject.toml", "line": 1.5},
        {"path": "pyproject.toml", "line": "1"},
        {"path": "/tmp/absolute", "line": 1},
        {"path": "missing.py", "line": 1},
    ):
        row = dict(valid)
        row["additional_evidence"] = [evidence]
        invalid.append(row)

    for candidate in invalid:
        with pytest.raises(DeclarationError):
            declarations._group_resources(
                {"pyproject.toml": b"[project]\n"}, [candidate]
            )


def test_mixed_resource_field_key_types_fail_without_sort_type_error() -> None:
    locator = "$CHRONOVISOR_ROOT/test/artifact.json"
    resource: dict[Any, Any] = {
        "id": declarations._resource_id("artifact", locator),
        "kind": "artifact",
        "locator": {"type": "path", "value": locator},
        "protects": [],
        "evidence": [{"path": "pyproject.toml", "line": 1}],
        "discovery_ids": [f"runtime-site:{'a' * 64}"],
        1: "mixed-key",
    }

    with pytest.raises(DeclarationError, match="field names must be exact strings"):
        declarations._validate_resource_collection([resource], {})


@pytest.mark.parametrize(
    "locator_value",
    [
        "$CHRONOVISOR_ROOT/runtime/*.json",
        "$CHRONOVISOR_ROOT/runtime/file?.json",
        "$CHRONOVISOR_ROOT/runtime/[abc].json",
        "$CHRONOVISOR_ROOT/runtime/white space.json",
        "$CHRONOVISOR_ROOT/runtime/tab\tname.json",
        "$CHRONOVISOR_ROOT/runtime/control\x01name.json",
        "$CHRONOVISOR_ROOT/runtime/$GEN/file.json",
        "$CHRONOVISOR_ROOT/runtime/$HOME/file.json",
        "$CHRONOVISOR_ROOT/$CHRONOVISOR_ROOT/file.json",
    ],
)
def test_unresolved_wildcard_or_noncanonical_path_locators_fail_closed(
    locator_value: str,
) -> None:
    with pytest.raises(DeclarationError, match="not canonical"):
        declarations._validate_locator(
            "artifact", {"type": "path", "value": locator_value}
        )


def test_unknown_but_syntactically_canonical_locator_remains_discoverable() -> None:
    assert declarations._validate_locator(
        "artifact",
        {"type": "path", "value": "$CHRONOVISOR_ROOT/unknown"},
    ) == ("artifact", "$CHRONOVISOR_ROOT/unknown")


def test_resource_identity_is_kind_scoped_for_same_physical_locator() -> None:
    locator = "$CHRONOVISOR_ROOT/test/shared.jsonl"
    candidates = [
        {
            "kind": kind,
            "locator": {"type": "path", "value": locator},
            "path": "pyproject.toml",
            "line": line,
            "discovery_id": f"runtime-site:{character * 64}",
        }
        for kind, line, character in (
            ("artifact", 1, "a"),
            ("queue", 2, "b"),
        )
    ]

    resources = declarations._group_resources(
        {"pyproject.toml": b"one\ntwo\n"}, candidates
    )

    assert len(resources) == 2
    assert {str(row["kind"]) for row in resources} == {"artifact", "queue"}
    assert len({str(row["id"]) for row in resources}) == 2


def test_effective_current_scan_is_delta_evidence_not_frozen_replacement(
    frozen: ConcreteDeclarations,
    effective_snapshot: CommittedSnapshot,
) -> None:
    current = discover_concrete(effective_snapshot)

    assert current.source_head == EFFECTIVE_FEEDBACK_REVISION
    assert len(current.rows) == 731
    assert len(current.lock_protocol_candidates) == 103
    assert len(frozen.rows) == 732
    assert len(frozen.lock_protocol_candidates) == 104


def test_feedback_resource_amendment_is_exact_and_structural(
    frozen: ConcreteDeclarations,
    effective_snapshot: CommittedSnapshot,
) -> None:
    amendment = apply_resource_amendment(frozen, effective_snapshot)
    resource = amendment.payload["resource"]

    assert amendment.amendment_id == (
        "runtime-amendment:bd589372354c147f9a5a6695a361b04b9768f7549cc37931bf66e6e646be2bb3"
    )
    assert set(amendment.payload) == {
        "schema_version",
        "kind",
        "operation",
        "base_source_head",
        "effective_source_head",
        "frozen_count",
        "effective_count",
        "resource",
    }
    assert set(resource) == RESOURCE_FIELDS
    assert resource == {
        "id": "runtime-resource:036b07eabe6998c8389bc2221e3074aca1b59dad202458cde1836ad40feaa512",
        "kind": "lock",
        "locator": {
            "type": "path",
            "value": "$CHRONOVISOR_ROOT/recall/feedback.jsonl.lock",
        },
        "protects": (
            "runtime-resource:8dcd7e44878e638768b6866bb2b002ff59e5c75a9cecba7a8d54f4794897a417",
        ),
        "evidence": (
            {"path": "src/chronovisor/recall/recall_runtime.py", "line": 51},
            {"path": "src/chronovisor/recall/recall_runtime.py", "line": 3073},
            {"path": "src/chronovisor/recall/recall_runtime.py", "line": 3078},
        ),
        "discovery_ids": (
            "runtime-site:f3310a069b43f8fdec587d108ff04c67a9686fea61389148ee7e192dd86528c4",
        ),
    }
    assert len(amendment.resources) == 455
    assert Counter(str(row["kind"]) for row in amendment.resources)["lock"] == 22
    _assert_forbidden_fields_absent(amendment.payload)


def test_effective_feedback_lock_mapping_exactly_covers_all_twenty_two_locks(
    frozen: ConcreteDeclarations,
    effective_snapshot: CommittedSnapshot,
) -> None:
    amendment = apply_resource_amendment(frozen, effective_snapshot)
    lock_locators = {
        str(row["locator"]["value"])
        for row in amendment.resources
        if row["kind"] == "lock"
    }

    assert lock_locators == set(declarations._EFFECTIVE_LOCK_PROTECTS_LOCATORS)
    declarations._validate_resource_collection(
        amendment.resources, declarations._EFFECTIVE_LOCK_PROTECTS_LOCATORS
    )


def test_feedback_amendment_requires_exact_effective_source(
    frozen: ConcreteDeclarations,
    frozen_snapshot: CommittedSnapshot,
) -> None:
    with pytest.raises(DeclarationError, match="exact effective revision"):
        apply_resource_amendment(frozen, frozen_snapshot)


def test_full_declaration_seal_rejects_same_id_nonidentity_tampering(
    frozen: ConcreteDeclarations,
    effective_snapshot: CommittedSnapshot,
) -> None:
    changed_row = dict(frozen.rows[0])
    changed_row["line"] = int(changed_row["line"]) + 1
    fake_row = ConcreteDeclarations(
        source_head=frozen.source_head,
        rows=declarations._frozen_rows((changed_row, *frozen.rows[1:])),
        resource_candidates=frozen.resource_candidates,
        exclusion_candidates=frozen.exclusion_candidates,
        lock_protocol_candidates=frozen.lock_protocol_candidates,
        resources=frozen.resources,
    )
    changed_resource = dict(frozen.resources[0])
    changed_resource["evidence"] = (
        {
            "path": changed_resource["evidence"][0]["path"],
            "line": int(changed_resource["evidence"][0]["line"]) + 1,
        },
        *changed_resource["evidence"][1:],
    )
    fake_resource = ConcreteDeclarations(
        source_head=frozen.source_head,
        rows=frozen.rows,
        resource_candidates=frozen.resource_candidates,
        exclusion_candidates=frozen.exclusion_candidates,
        lock_protocol_candidates=frozen.lock_protocol_candidates,
        resources=declarations._frozen_rows((changed_resource, *frozen.resources[1:])),
    )

    with pytest.raises(DeclarationError, match="full declaration seal mismatch"):
        apply_resource_amendment(fake_row, effective_snapshot)
    with pytest.raises(DeclarationError, match="full declaration seal mismatch"):
        record_protocol_transition(fake_resource, effective_snapshot)

    original = frozen.resources[0]
    tampered_resources: list[dict[str, Any]] = []
    locator_tamper = dict(original)
    locator_tamper["locator"] = {
        **dict(original["locator"]),
        "value": f"{original['locator']['value']}-tampered",
    }
    tampered_resources.append(locator_tamper)
    kind_tamper = dict(original)
    kind_tamper["kind"] = "queue"
    tampered_resources.append(kind_tamper)
    protects_tamper = dict(original)
    protects_tamper["protects"] = ("runtime-resource:" + "a" * 64,)
    tampered_resources.append(protects_tamper)
    discoveries_tamper = dict(original)
    discoveries_tamper["discovery_ids"] = (
        *original["discovery_ids"],
        "runtime-site:" + "a" * 64,
    )
    tampered_resources.append(discoveries_tamper)

    for tampered in tampered_resources:
        fake = ConcreteDeclarations(
            source_head=frozen.source_head,
            rows=frozen.rows,
            resource_candidates=frozen.resource_candidates,
            exclusion_candidates=frozen.exclusion_candidates,
            lock_protocol_candidates=frozen.lock_protocol_candidates,
            resources=declarations._frozen_rows((tampered, *frozen.resources[1:])),
        )
        with pytest.raises(DeclarationError):
            apply_resource_amendment(fake, effective_snapshot)


def test_returned_declarations_and_amendments_are_deeply_immutable_without_aliases(
    frozen: ConcreteDeclarations,
    effective_snapshot: CommittedSnapshot,
) -> None:
    amendment = apply_resource_amendment(frozen, effective_snapshot)
    before = declarations._canonical_bytes(amendment.payload)
    payload_resource = cast(Mapping[str, Any], amendment.payload["resource"])
    effective_resource = next(
        row for row in amendment.resources if row["id"] == payload_resource["id"]
    )

    assert payload_resource is not effective_resource
    _assert_no_mutable_json_containers(amendment.payload)
    _assert_no_mutable_json_containers(amendment.resources)
    with pytest.raises(TypeError):
        cast(Any, frozen.rows[0])["line"] = 0
    with pytest.raises(TypeError):
        cast(Any, amendment.payload)["kind"] = "tampered"
    with pytest.raises(TypeError):
        cast(Any, payload_resource)["kind"] = "tampered"
    with pytest.raises(TypeError):
        cast(Any, effective_resource)["kind"] = "tampered"
    with pytest.raises(TypeError):
        dict.__setitem__(cast(Any, payload_resource), "kind", "tampered")
    with pytest.raises(TypeError):
        dict.__setitem__(cast(Any, effective_resource), "kind", "tampered")
    assert declarations._canonical_bytes(amendment.payload) == before
    assert amendment.amendment_id == declarations._semantic_id(
        "runtime-amendment", amendment.payload
    )


def test_protocol_transition_is_exact_v1_and_keeps_frozen_unchanged(
    frozen: ConcreteDeclarations,
    effective_snapshot: CommittedSnapshot,
) -> None:
    frozen_before = tuple(
        str(row["discovery_id"]) for row in frozen.lock_protocol_candidates
    )
    transition = record_protocol_transition(frozen, effective_snapshot)

    assert transition.transition_id == (
        "004ea1090792ded52dfa8f21f7d0d46cbd51bfccef12ca3e40c7426d3eb6a160"
    )
    assert set(transition.payload) == {
        "schema_version",
        "kind",
        "operation",
        "base_source_head",
        "effective_source_head",
        "counts",
        "removed",
        "added",
        "shared_helper",
    }
    assert transition.payload["counts"] == {
        "frozen_lock_protocol_sites": 104,
        "effective_lock_protocol_sites": 103,
    }
    assert [row["discovery_id"] for row in transition.payload["removed"]] == [
        "runtime-site:0e7d4ca4e49982b83440b9592c1b1eb9d53f9d254335ab51ac077065f1633a6a",
        "runtime-site:58e71139ca0945c3405f3956cb7705a481fb74639f7e0e50b3333bd5dacc5a81",
    ]
    assert [row["discovery_id"] for row in transition.payload["added"]] == [
        "runtime-site:bf488c9fa72d0998f8ff25c78053ad9f6f378a32b1e92dcfebf7bffdae807beb"
    ]
    assert transition.payload["shared_helper"] == {
        "path": "src/chronovisor/recall/recall_runtime.py",
        "module": "chronovisor.recall.recall_runtime",
        "symbol": "_feedback_exclusive_lock",
        "definition_line": 3064,
        "sidecar_derivation_line": 3073,
        "acquisition_line": 3078,
        "callsites": (
            {
                "path": "src/chronovisor/recall/content_correction.py",
                "line": 2145,
                "scope": "_retract_legacy_unfiltered_page_ignored_feedback",
            },
            {
                "path": "src/chronovisor/recall/content_correction.py",
                "line": 3535,
                "scope": "_record_wrong_retrieval",
            },
            {
                "path": "src/chronovisor/recall/recall_runtime.py",
                "line": 3019,
                "scope": "append_feedback",
            },
        ),
    }
    assert len(transition.lock_protocol_candidates) == 103
    assert (
        tuple(str(row["discovery_id"]) for row in frozen.lock_protocol_candidates)
        == frozen_before
    )
    _assert_forbidden_fields_absent(transition.payload)


def test_protocol_transition_payload_is_deeply_immutable_and_hash_stable(
    frozen: ConcreteDeclarations,
    effective_snapshot: CommittedSnapshot,
) -> None:
    transition = record_protocol_transition(frozen, effective_snapshot)
    before = declarations._canonical_bytes(transition.payload)
    helper = cast(Mapping[str, Any], transition.payload["shared_helper"])
    first_callsite = cast(Mapping[str, Any], helper["callsites"][0])

    _assert_no_mutable_json_containers(transition.payload)
    _assert_no_mutable_json_containers(transition.lock_protocol_candidates)
    with pytest.raises(TypeError):
        cast(Any, transition.payload)["operation"] = "tampered"
    with pytest.raises(TypeError):
        cast(Any, helper)["symbol"] = "tampered"
    with pytest.raises(TypeError):
        cast(Any, first_callsite)["line"] = 0
    with pytest.raises(TypeError):
        dict.__setitem__(cast(Any, first_callsite), "line", 0)
    assert declarations._canonical_bytes(transition.payload) == before
    assert transition.transition_id == hashlib.sha256(before).hexdigest()


def test_protocol_transition_requires_exact_effective_source(
    frozen: ConcreteDeclarations,
    frozen_snapshot: CommittedSnapshot,
) -> None:
    with pytest.raises(DeclarationError, match="exact effective revision"):
        record_protocol_transition(frozen, frozen_snapshot)


def test_declarations_do_not_import_rejected_provisional_modules() -> None:
    path = ROOT / "scripts/runtime_ownership/declarations.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imported = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }

    assert imported.isdisjoint(
        {
            "scripts.runtime_ownership.model",
            "scripts.runtime_ownership.source",
            "scripts.runtime_ownership.discovery",
            "runtime_ownership.model",
            "runtime_ownership.source",
            "runtime_ownership.discovery",
        }
    )
