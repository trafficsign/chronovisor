from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import pytest

from scripts.runtime_ownership.access import discover_access_facts
from scripts.runtime_ownership.declarations import discover_concrete
from scripts.runtime_ownership.machine_facts import (
    ADAPTER_SCHEMA_VERSION,
    ANALYZER_FILES_SHA256,
    ANALYZER_MANIFEST_SHA256,
    ANALYZER_REVISION,
    CANDIDATE_SUBSET_SHA256,
    EFFECTIVE_SOURCE_REVISION,
    SHARD_PLAN_ID,
    SHARDING_DISABLED_REASON,
    SOURCE_FILES_SHA256,
    SOURCE_MANIFEST_SHA256,
    MachineFactError,
    _assemble_sealed_document,
    analyze_runtime_access_unsealed,
    build_declaration_adapter,
    cache_key_metadata,
    canonical_bytes,
    load_machine_fact_cache,
    machine_fact_cache_bytes,
    reject_sharded_analysis,
    run_sealed_effective_analysis,
    validate_effective_adapter,
    validate_machine_fact_document,
    validate_runtime_access_v2,
    verify_loaded_analyzer_code,
)
from scripts.runtime_ownership.manifests import (
    ANALYZER_MANIFEST_KIND,
    SOURCE_MANIFEST_KIND,
    CommittedSnapshot,
    build_manifest,
    committed_snapshot,
)
from tests.runtime_access_v2_helpers import validate_runtime_access_v2_result

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_FIELDS = {
    "owner",
    "owner_package",
    "owner_symbol",
    "writers",
    "readers",
    "lifecycle",
    "coordination",
}


@pytest.fixture(scope="module")
def source_snapshot() -> CommittedSnapshot:
    return committed_snapshot(
        ROOT, EFFECTIVE_SOURCE_REVISION, manifest_kind=SOURCE_MANIFEST_KIND
    )


@pytest.fixture(scope="module")
def analyzer_snapshot() -> CommittedSnapshot:
    return committed_snapshot(
        ROOT, ANALYZER_REVISION, manifest_kind=ANALYZER_MANIFEST_KIND
    )


@pytest.fixture(scope="module")
def adapter(source_snapshot: CommittedSnapshot) -> dict[str, Any]:
    return build_declaration_adapter(discover_concrete(source_snapshot))


@pytest.fixture(scope="module")
def seals() -> tuple[dict[str, str], dict[str, str]]:
    return (
        {
            "revision": EFFECTIVE_SOURCE_REVISION,
            "files_sha256": SOURCE_FILES_SHA256,
            "manifest_sha256": SOURCE_MANIFEST_SHA256,
        },
        {
            "revision": ANALYZER_REVISION,
            "files_sha256": ANALYZER_FILES_SHA256,
            "manifest_sha256": ANALYZER_MANIFEST_SHA256,
        },
    )


def _empty_v2() -> dict[str, Any]:
    return {
        "schema_version": 2,
        "sites": [],
        "provenances": [],
        "provenance_ids": [],
        "access_facts": [],
        "escape_facts": [],
        "access_fact_ids": [],
        "escape_fact_ids": [],
        "counts": {
            "accesses": 0,
            "escapes": 0,
            "read": 0,
            "write": 0,
            "read_write": 0,
        },
    }


def _independent_id(prefix: str, identity: Mapping[str, Any]) -> str:
    raw = json.dumps(
        identity,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(raw).hexdigest()}"


def _escape_identity(row: Mapping[str, Any]) -> dict[str, Any]:
    identity = {
        key: row[key]
        for key in ("site_id", "resource_id", "operation", "sink", "reason")
    }
    source_kind = row.get("source_kind")
    if source_kind is not None:
        identity.update(
            {
                key: row[key]
                for key in (
                    "source_kind",
                    "source_fact_id",
                    "limit",
                    "retention_policy",
                )
            }
        )
        if source_kind == "access":
            identity.update({key: row[key] for key in ("mode", "sink_actor")})
        else:
            identity["source_reason"] = row["source_reason"]
    return identity


def _rekey_escape(result: dict[str, Any], row: dict[str, Any]) -> None:
    old_id = str(row["escape_fact_id"])
    new_id = _independent_id("runtime-escape-fact", _escape_identity(row))
    row["escape_fact_id"] = new_id
    result["escape_fact_ids"] = sorted(
        new_id if item == old_id else item for item in result["escape_fact_ids"]
    )
    result["escape_facts"].sort(key=lambda item: item["escape_fact_id"])


def _resource_locators(adapter: Mapping[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for row in adapter["analyzer_candidates"]:
        result[str(row["id"])] = str(row["locator"]["value"])
    return result


def _nonempty_v2(adapter: Mapping[str, Any]) -> dict[str, Any]:
    candidate = adapter["analyzer_candidates"][0]
    resource_id = str(candidate["id"])
    locator = str(candidate["locator"]["value"])
    site_identity = {
        "path": "src/chronovisor/example.py",
        "scope": "chronovisor.example:run",
        "kind": "call",
        "syntax": "Call(func=Name(id='open', ctx=Load()), args=[], keywords=[])",
        "occurrence": 1,
    }
    site_id = _independent_id("runtime-site", site_identity)
    provenance_identity = {
        "resource_id": resource_id,
        "actor": "chronovisor.example:run",
        "binding_chain": ["origin:chronovisor.example:STATE_FILE"],
    }
    provenance_id = _independent_id("runtime-provenance", provenance_identity)
    access_identity = {
        "site_id": site_id,
        "resource_id": resource_id,
        "mode": "read",
        "operation": "open:r",
        "sink": "stdlib-open",
        "sink_actor": "chronovisor.example:run",
    }
    access_id = _independent_id("runtime-access-fact", access_identity)
    return {
        "schema_version": 2,
        "sites": [
            {
                "site_id": site_id,
                "scope": site_identity["scope"],
                "kind": site_identity["kind"],
                "syntax": site_identity["syntax"],
                "occurrence": site_identity["occurrence"],
                "evidence": {"path": site_identity["path"], "line": 12},
            }
        ],
        "provenances": [
            {
                "provenance_id": provenance_id,
                **provenance_identity,
                "locator": locator,
                "call_site_ids": [],
            }
        ],
        "provenance_ids": [provenance_id],
        "access_facts": [
            {
                "access_fact_id": access_id,
                **access_identity,
                "locator": locator,
                "provenance_ids": [provenance_id],
                "actors": [provenance_identity["actor"]],
                "provenance_complete": True,
            }
        ],
        "escape_facts": [],
        "access_fact_ids": [access_id],
        "escape_fact_ids": [],
        "counts": {
            "accesses": 1,
            "escapes": 0,
            "read": 1,
            "write": 0,
            "read_write": 0,
        },
    }


def _with_overflows(adapter: Mapping[str, Any]) -> dict[str, Any]:
    result = _nonempty_v2(adapter)
    access = result["access_facts"][0]
    normal_identity = {
        "site_id": access["site_id"],
        "resource_id": access["resource_id"],
        "operation": "open:dynamic",
        "sink": "stdlib-open",
        "reason": "dynamic_open_mode",
    }
    normal_id = _independent_id("runtime-escape-fact", normal_identity)
    normal = {
        "escape_fact_id": normal_id,
        **normal_identity,
        "locator": access["locator"],
        "provenance_ids": list(access["provenance_ids"]),
        "actors": list(access["actors"]),
        "provenance_complete": False,
    }
    escape_overflow_identity = {
        **normal_identity,
        "reason": "provenance_overflow",
        "source_kind": "escape",
        "source_fact_id": normal_id,
        "limit": 64,
        "retention_policy": "shortest_then_lexicographic",
        "source_reason": "dynamic_open_mode",
    }
    escape_overflow = {
        "escape_fact_id": _independent_id(
            "runtime-escape-fact", escape_overflow_identity
        ),
        **escape_overflow_identity,
        "locator": access["locator"],
        "provenance_ids": [],
        "actors": [],
        "provenance_complete": False,
    }
    access_overflow_identity = {
        key: access[key]
        for key in (
            "site_id",
            "resource_id",
            "operation",
            "sink",
            "mode",
            "sink_actor",
        )
    }
    access_overflow_identity.update(
        {
            "reason": "provenance_overflow",
            "source_kind": "access",
            "source_fact_id": access["access_fact_id"],
            "limit": 64,
            "retention_policy": "shortest_then_lexicographic",
        }
    )
    access_overflow = {
        "escape_fact_id": _independent_id(
            "runtime-escape-fact", access_overflow_identity
        ),
        **access_overflow_identity,
        "locator": access["locator"],
        "provenance_ids": [],
        "actors": [],
        "provenance_complete": False,
    }
    access["provenance_complete"] = False
    result["escape_facts"] = sorted(
        [normal, escape_overflow, access_overflow],
        key=lambda row: row["escape_fact_id"],
    )
    result["escape_fact_ids"] = [
        row["escape_fact_id"] for row in result["escape_facts"]
    ]
    result["counts"]["escapes"] = 3
    return result


def _document(
    adapter: Mapping[str, Any],
    seals: tuple[dict[str, str], dict[str, str]],
    *,
    result: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source_seal, analyzer_seal = seals

    def fake_analyzer(_source: Mapping[str, bytes], _candidates: Any) -> dict[str, Any]:
        return copy.deepcopy(_empty_v2() if result is None else result)

    result_value = analyze_runtime_access_unsealed(
        {"src/chronovisor/unused.py": b"pass\n"},
        adapter,
        analyzer=fake_analyzer,
    )
    document = _assemble_sealed_document(
        result_value,
        adapter,
        source_seal=source_seal,
        analyzer_seal=analyzer_seal,
    )
    return document, cast(dict[str, Any], document["cache_key"])


def _assert_no_ownership(value: Any) -> None:
    if isinstance(value, Mapping):
        assert FORBIDDEN_FIELDS.isdisjoint(value)
        for item in value.values():
            _assert_no_ownership(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _assert_no_ownership(item)


def test_exact_source_and_analyzer_manifests_are_separately_sealed() -> None:
    source = build_manifest(
        ROOT,
        EFFECTIVE_SOURCE_REVISION,
        manifest_kind=SOURCE_MANIFEST_KIND,
        expected_revision=EFFECTIVE_SOURCE_REVISION,
    )
    analyzer = build_manifest(
        ROOT,
        ANALYZER_REVISION,
        manifest_kind=ANALYZER_MANIFEST_KIND,
        expected_revision=ANALYZER_REVISION,
    )

    assert (source["files_sha256"], source["manifest_sha256"]) == (
        SOURCE_FILES_SHA256,
        SOURCE_MANIFEST_SHA256,
    )
    assert (analyzer["files_sha256"], analyzer["manifest_sha256"]) == (
        ANALYZER_FILES_SHA256,
        ANALYZER_MANIFEST_SHA256,
    )
    assert source["counts"] == {"files": 296, "bytes": 7_456_781}
    assert analyzer["counts"] == {"files": 15, "bytes": 556_664}


def test_real_adapter_partition_hash_aliases_and_reasons(
    adapter: Mapping[str, Any],
) -> None:
    validate_effective_adapter(adapter)
    assert adapter["adapter_schema_version"] == ADAPTER_SCHEMA_VERSION
    assert adapter["counts"] == {
        "resource_candidates": 490,
        "supported_candidates": 243,
        "supported_resources": 210,
        "unsupported_declarations": 247,
        "excluded_declarations": 138,
    }
    assert adapter["candidate_subset_sha256"] == CANDIDATE_SUBSET_SHA256
    candidates = adapter["analyzer_candidates"]
    assert all(set(row) == {"id", "module", "symbol", "locator"} for row in candidates)
    assert len(candidates) - len({row["id"] for row in candidates}) == 33
    assert Counter(row["reason"] for row in adapter["unsupported_declarations"]) == {
        "unsupported_resource_kind:schema": 155,
        "unsupported_resource_kind:worker": 88,
        "unsupported_socket_locator": 4,
    }
    unsupported_sockets = [
        row
        for row in adapter["unsupported_declarations"]
        if row["reason"] == "unsupported_socket_locator"
    ]
    assert {
        row["locator"]["value"].split(":", 1)[0] for row in unsupported_sockets
    } == {
        "tcp",
        "stdio",
    }
    assert all(row["evidence"] for row in unsupported_sockets)
    unix = [
        row for row in candidates if str(row["locator"]["value"]).startswith("unix://")
    ]
    assert len(unix) == 2
    _assert_no_ownership(adapter)


def test_real_partition_covers_each_resource_discovery_once(
    source_snapshot: CommittedSnapshot, adapter: Mapping[str, Any]
) -> None:
    declarations = discover_concrete(source_snapshot)
    all_ids = [str(row["discovery_id"]) for row in declarations.resource_candidates]
    supported_ids = {
        str(row["discovery_id"])
        for row in declarations.resource_candidates
        if row["kind"] in {"artifact", "queue", "lock"}
        or (
            row["kind"] == "socket"
            and str(row["locator"]["value"]).startswith("unix://")
        )
    }
    unsupported_ids = {
        str(row["discovery_id"]) for row in adapter["unsupported_declarations"]
    }
    assert len(all_ids) == len(set(all_ids)) == 490
    assert len(supported_ids) == 243
    assert len(unsupported_ids) == 247
    assert supported_ids.isdisjoint(unsupported_ids)
    assert supported_ids | unsupported_ids == set(all_ids)
    original_exclusions = {
        str(row["discovery_id"]): str(row["reason"])
        for row in declarations.exclusion_candidates
    }
    assert {
        str(row["discovery_id"]): str(row["reason"])
        for row in adapter["excluded_declarations"]
    } == original_exclusions


def test_candidate_ids_are_group_ids_not_discovery_ids(
    source_snapshot: CommittedSnapshot, adapter: Mapping[str, Any]
) -> None:
    declarations = discover_concrete(source_snapshot)
    discovery_ids = {row["discovery_id"] for row in declarations.resource_candidates}
    resource_ids = {row["id"] for row in declarations.resources}
    candidate_ids = {row["id"] for row in adapter["analyzer_candidates"]}
    assert candidate_ids <= resource_ids
    assert candidate_ids.isdisjoint(discovery_ids)


def test_fake_analyzer_is_called_once_with_all_alias_rows(
    adapter: Mapping[str, Any], seals: tuple[dict[str, str], dict[str, str]]
) -> None:
    calls: list[tuple[Mapping[str, bytes], list[Mapping[str, Any]]]] = []

    def fake_analyzer(source: Mapping[str, bytes], candidates: Any) -> dict[str, Any]:
        calls.append((source, list(candidates)))
        return _empty_v2()

    source_seal, analyzer_seal = seals
    result = analyze_runtime_access_unsealed(
        {"src/chronovisor/a.py": b"pass\n"},
        adapter,
        analyzer=fake_analyzer,
    )
    assert len(calls) == 1
    assert len(calls[0][1]) == 243
    assert result == _empty_v2()
    with pytest.raises(MachineFactError, match="keys mismatch"):
        machine_fact_cache_bytes(result, adapter=adapter)


def test_sealed_entrypoint_rejects_dependency_injection_and_preloaded_state() -> None:
    def fake_analyzer(_source: Mapping[str, bytes], _candidates: Any) -> dict[str, Any]:
        return _empty_v2()

    with pytest.raises(TypeError):
        run_sealed_effective_analysis(  # type: ignore[call-arg]
            ROOT, analyzer=fake_analyzer
        )
    with pytest.raises(TypeError):
        run_sealed_effective_analysis(  # type: ignore[call-arg]
            ROOT, loaded_analyzer_files={}
        )
    with pytest.raises(MachineFactError, match="fresh analyzer import state"):
        run_sealed_effective_analysis(ROOT)


def test_importing_machine_facts_does_not_preload_analyzer_modules() -> None:
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import scripts.runtime_ownership.machine_facts; "
                "assert not any(name == 'scripts.runtime_ownership.access' or "
                "name.startswith('scripts.runtime_ownership.access_') "
                "for name in sys.modules)"
            ),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert probe.returncode == 0, probe.stderr


def test_fresh_subprocess_binds_all_15_verified_analyzer_modules() -> None:
    code = """
import sys
from pathlib import Path
from scripts.runtime_ownership.machine_facts import (
    ANALYZER_REVISION,
    _ANALYZER_MODULE_NAMES,
    _import_verified_fresh_analyzer,
)
from scripts.runtime_ownership.manifests import (
    ANALYZER_MANIFEST_KIND,
    committed_snapshot,
)
repository = Path.cwd()
snapshot = committed_snapshot(
    repository,
    ANALYZER_REVISION,
    manifest_kind=ANALYZER_MANIFEST_KIND,
)
analyzer = _import_verified_fresh_analyzer(repository, snapshot)
assert callable(analyzer)
assert analyzer.__module__ == "scripts.runtime_ownership.access"
assert _ANALYZER_MODULE_NAMES.intersection(sys.modules) == _ANALYZER_MODULE_NAMES
"""
    probe = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert probe.returncode == 0, probe.stderr


def test_loaded_analyzer_code_drift_fails_closed(
    analyzer_snapshot: CommittedSnapshot,
) -> None:
    loaded = {row.path: row.raw_bytes for row in analyzer_snapshot.files}
    verify_loaded_analyzer_code(loaded, analyzer_snapshot)
    path = sorted(loaded)[0]
    loaded[path] += b"\n# drift\n"
    with pytest.raises(MachineFactError, match="loaded/current analyzer code drifted"):
        verify_loaded_analyzer_code(loaded, analyzer_snapshot)
    loaded.pop(path)
    with pytest.raises(MachineFactError, match="exact 15 files"):
        verify_loaded_analyzer_code(loaded, analyzer_snapshot)


def test_strict_v2_validator_matches_independent_helper(
    adapter: Mapping[str, Any],
) -> None:
    result = _nonempty_v2(adapter)
    validate_runtime_access_v2(result, resource_locators=_resource_locators(adapter))
    validate_runtime_access_v2_result(result)


@pytest.mark.parametrize(
    "tamper",
    [
        lambda value: value["sites"][0].__setitem__("syntax", "tampered"),
        lambda value: value["provenances"][0].__setitem__("actor", "tampered"),
        lambda value: value["access_facts"][0].__setitem__("sink", "tampered"),
        lambda value: value["access_facts"][0].__setitem__("locator", "tampered"),
        lambda value: value["access_facts"][0].__setitem__("actors", ["tampered"]),
        lambda value: value["counts"].__setitem__("read", 0),
    ],
)
def test_semantic_field_tamper_with_unchanged_id_is_rejected(
    adapter: Mapping[str, Any], tamper: Any
) -> None:
    value = _nonempty_v2(adapter)
    tamper(value)
    with pytest.raises(MachineFactError):
        validate_runtime_access_v2(value, resource_locators=_resource_locators(adapter))


def test_v2_rejects_legacy_keys_bad_foreign_keys_ids_and_order(
    adapter: Mapping[str, Any],
) -> None:
    resources = _resource_locators(adapter)
    legacy = _nonempty_v2(adapter)
    legacy["accesses"] = []
    with pytest.raises(MachineFactError, match="keys mismatch"):
        validate_runtime_access_v2(legacy, resource_locators=resources)

    bad_fk = _nonempty_v2(adapter)
    bad_fk["access_facts"][0]["provenance_ids"] = ["runtime-provenance:" + "0" * 64]
    with pytest.raises(MachineFactError, match="unknown provenance"):
        validate_runtime_access_v2(bad_fk, resource_locators=resources)

    bad_id = _nonempty_v2(adapter)
    bad_id["access_facts"][0]["access_fact_id"] = "bad"
    bad_id["access_fact_ids"] = ["bad"]
    with pytest.raises(MachineFactError, match="invalid identity"):
        validate_runtime_access_v2(bad_id, resource_locators=resources)

    duplicate = _nonempty_v2(adapter)
    duplicate["provenance_ids"] *= 2
    with pytest.raises(MachineFactError, match="sorted and unique"):
        validate_runtime_access_v2(duplicate, resource_locators=resources)

    mixed_key = cast(dict[Any, Any], _empty_v2())
    mixed_key[1] = "not-a-string-key"
    with pytest.raises(MachineFactError, match="keys must be exact strings"):
        validate_runtime_access_v2(mixed_key, resource_locators=resources)


def test_normal_and_overflow_escape_identities_are_recomputed(
    adapter: Mapping[str, Any],
) -> None:
    result = _nonempty_v2(adapter)
    access = result["access_facts"][0]
    normal_identity = {
        "site_id": access["site_id"],
        "resource_id": access["resource_id"],
        "operation": "open:dynamic",
        "sink": "stdlib-open",
        "reason": "dynamic_open_mode",
    }
    normal_id = _independent_id("runtime-escape-fact", normal_identity)
    normal = {
        "escape_fact_id": normal_id,
        **normal_identity,
        "locator": access["locator"],
        "provenance_ids": list(access["provenance_ids"]),
        "actors": list(access["actors"]),
        "provenance_complete": False,
    }
    overflow_identity = {
        "site_id": access["site_id"],
        "resource_id": access["resource_id"],
        "operation": "open:dynamic",
        "sink": "stdlib-open",
        "reason": "provenance_overflow",
        "source_kind": "escape",
        "source_fact_id": normal_id,
        "limit": 64,
        "retention_policy": "shortest_then_lexicographic",
        "source_reason": "dynamic_open_mode",
    }
    overflow_id = _independent_id("runtime-escape-fact", overflow_identity)
    overflow = {
        "escape_fact_id": overflow_id,
        **overflow_identity,
        "locator": access["locator"],
        "provenance_ids": [],
        "actors": [],
        "provenance_complete": False,
    }
    access_overflow_identity = {
        "site_id": access["site_id"],
        "resource_id": access["resource_id"],
        "operation": access["operation"],
        "sink": access["sink"],
        "reason": "provenance_overflow",
        "source_kind": "access",
        "source_fact_id": access["access_fact_id"],
        "limit": 64,
        "retention_policy": "shortest_then_lexicographic",
        "mode": access["mode"],
        "sink_actor": access["sink_actor"],
    }
    access_overflow_id = _independent_id(
        "runtime-escape-fact", access_overflow_identity
    )
    access_overflow = {
        "escape_fact_id": access_overflow_id,
        **access_overflow_identity,
        "locator": access["locator"],
        "provenance_ids": [],
        "actors": [],
        "provenance_complete": False,
    }
    access["provenance_complete"] = False
    result["escape_facts"] = sorted(
        [normal, overflow, access_overflow], key=lambda row: row["escape_fact_id"]
    )
    result["escape_fact_ids"] = sorted([normal_id, overflow_id, access_overflow_id])
    result["counts"]["escapes"] = 3
    validate_runtime_access_v2(result, resource_locators=_resource_locators(adapter))

    result["escape_facts"][result["escape_fact_ids"].index(overflow_id)]["limit"] = 63
    with pytest.raises(MachineFactError, match="overflow retention contract"):
        validate_runtime_access_v2(
            result, resource_locators=_resource_locators(adapter)
        )


@pytest.mark.parametrize(
    ("source_kind", "field", "replacement"),
    [
        ("access", "limit", 63),
        ("access", "retention_policy", "tampered-policy"),
        ("escape", "limit", 63),
        ("escape", "retention_policy", "tampered-policy"),
    ],
)
def test_overflow_retention_contract_is_exact(
    adapter: Mapping[str, Any], source_kind: str, field: str, replacement: Any
) -> None:
    result = _with_overflows(adapter)
    overflow = next(
        row for row in result["escape_facts"] if row.get("source_kind") == source_kind
    )
    overflow[field] = replacement
    _rekey_escape(result, overflow)
    with pytest.raises(MachineFactError, match="overflow retention contract"):
        validate_runtime_access_v2(
            result, resource_locators=_resource_locators(adapter)
        )


@pytest.mark.parametrize(
    ("source_kind", "field", "replacement"),
    [
        ("access", "operation", "tampered-operation"),
        ("access", "sink", "tampered-sink"),
        ("access", "mode", "write"),
        ("access", "sink_actor", "tampered-actor"),
        ("escape", "operation", "tampered-operation"),
        ("escape", "sink", "tampered-sink"),
        ("escape", "source_reason", "tampered-reason"),
    ],
)
def test_overflow_must_match_all_non_fk_source_fields(
    adapter: Mapping[str, Any], source_kind: str, field: str, replacement: str
) -> None:
    result = _with_overflows(adapter)
    overflow = next(
        row for row in result["escape_facts"] if row.get("source_kind") == source_kind
    )
    overflow[field] = replacement
    _rekey_escape(result, overflow)
    with pytest.raises(MachineFactError, match="does not match its source fact"):
        validate_runtime_access_v2(
            result, resource_locators=_resource_locators(adapter)
        )


@pytest.mark.parametrize("source_kind", ["access", "escape"])
def test_overflow_must_match_source_site_and_resource(
    adapter: Mapping[str, Any], source_kind: str
) -> None:
    resources = _resource_locators(adapter)
    for field in ("site_id", "resource_id"):
        result = _with_overflows(adapter)
        overflow = next(
            row
            for row in result["escape_facts"]
            if row.get("source_kind") == source_kind
        )
        if field == "site_id":
            original = result["sites"][0]
            identity = {
                "path": original["evidence"]["path"],
                "scope": original["scope"],
                "kind": original["kind"],
                "syntax": original["syntax"],
                "occurrence": 2,
            }
            site_id = _independent_id("runtime-site", identity)
            result["sites"].append(
                {
                    "site_id": site_id,
                    "scope": identity["scope"],
                    "kind": identity["kind"],
                    "syntax": identity["syntax"],
                    "occurrence": identity["occurrence"],
                    "evidence": dict(original["evidence"]),
                }
            )
            result["sites"].sort(key=lambda row: row["site_id"])
            overflow["site_id"] = site_id
        else:
            resource_id = next(
                item for item in resources if item != overflow["resource_id"]
            )
            overflow["resource_id"] = resource_id
            overflow["locator"] = resources[resource_id]
        _rekey_escape(result, overflow)
        with pytest.raises(MachineFactError, match="does not match its source fact"):
            validate_runtime_access_v2(result, resource_locators=resources)


@pytest.mark.parametrize("source_kind", ["access", "escape"])
def test_overflow_shape_forces_false_complete_and_rejects_overflow_chains(
    adapter: Mapping[str, Any], source_kind: str
) -> None:
    resources = _resource_locators(adapter)
    result = _with_overflows(adapter)
    overflow = next(
        row for row in result["escape_facts"] if row.get("source_kind") == source_kind
    )
    overflow["provenance_complete"] = True
    with pytest.raises(MachineFactError, match="invalid overflow shape"):
        validate_runtime_access_v2(result, resource_locators=resources)

    if source_kind == "escape":
        result = _with_overflows(adapter)
        escape_overflow = next(
            row for row in result["escape_facts"] if row.get("source_kind") == "escape"
        )
        access_overflow = next(
            row for row in result["escape_facts"] if row.get("source_kind") == "access"
        )
        escape_overflow["source_fact_id"] = access_overflow["escape_fact_id"]
        _rekey_escape(result, escape_overflow)
        with pytest.raises(MachineFactError, match="non-overflow escape fact"):
            validate_runtime_access_v2(result, resource_locators=resources)


def test_source_fact_provenance_complete_exactly_matches_overflow_coverage(
    adapter: Mapping[str, Any],
) -> None:
    resources = _resource_locators(adapter)
    no_overflow = _nonempty_v2(adapter)
    no_overflow["access_facts"][0]["provenance_complete"] = False
    with pytest.raises(MachineFactError, match="overflow coverage"):
        validate_runtime_access_v2(no_overflow, resource_locators=resources)

    for source_kind in ("access", "escape"):
        covered = _with_overflows(adapter)
        if source_kind == "access":
            source = covered["access_facts"][0]
        else:
            source = next(
                row for row in covered["escape_facts"] if row.get("source_kind") is None
            )
        source["provenance_complete"] = True
        with pytest.raises(MachineFactError, match="overflow coverage"):
            validate_runtime_access_v2(covered, resource_locators=resources)

    uncovered_escape = _with_overflows(adapter)
    uncovered_escape["escape_facts"] = [
        row
        for row in uncovered_escape["escape_facts"]
        if row.get("source_kind") != "escape"
    ]
    uncovered_escape["escape_fact_ids"] = [
        row["escape_fact_id"] for row in uncovered_escape["escape_facts"]
    ]
    uncovered_escape["counts"]["escapes"] = len(uncovered_escape["escape_facts"])
    with pytest.raises(MachineFactError, match="overflow coverage"):
        validate_runtime_access_v2(uncovered_escape, resource_locators=resources)


def test_normal_escape_cannot_claim_provenance_overflow_reason(
    adapter: Mapping[str, Any],
) -> None:
    result = _with_overflows(adapter)
    normal = next(
        row for row in result["escape_facts"] if row.get("source_kind") is None
    )
    normal["reason"] = "provenance_overflow"
    _rekey_escape(result, normal)
    with pytest.raises(MachineFactError, match="requires the extended overflow shape"):
        validate_runtime_access_v2(
            result, resource_locators=_resource_locators(adapter)
        )


def test_cache_key_has_exact_metadata_and_invalidates_both_inputs(
    seals: tuple[dict[str, str], dict[str, str]],
) -> None:
    source_seal, analyzer_seal = seals
    key = cache_key_metadata(
        source_seal=source_seal,
        analyzer_seal=analyzer_seal,
        candidate_subset_sha256=CANDIDATE_SUBSET_SHA256,
    )
    assert set(key) == {
        "adapter_schema_version",
        "source_revision",
        "source_files_sha256",
        "source_manifest_sha256",
        "analyzer_revision",
        "analyzer_files_sha256",
        "analyzer_manifest_sha256",
        "candidate_subset_sha256",
        "shard_plan_id",
    }
    assert key["adapter_schema_version"] == ADAPTER_SCHEMA_VERSION
    assert key["shard_plan_id"] == SHARD_PLAN_ID
    changed_source = {**source_seal, "files_sha256": "0" * 64}
    changed_analyzer = {**analyzer_seal, "manifest_sha256": "1" * 64}
    assert (
        cache_key_metadata(
            source_seal=changed_source,
            analyzer_seal=analyzer_seal,
            candidate_subset_sha256=CANDIDATE_SUBSET_SHA256,
        )
        != key
    )
    assert (
        cache_key_metadata(
            source_seal=source_seal,
            analyzer_seal=changed_analyzer,
            candidate_subset_sha256=CANDIDATE_SUBSET_SHA256,
        )
        != key
    )


def test_adapter_candidate_drift_and_cache_input_invalidation_fail_closed(
    adapter: Mapping[str, Any], seals: tuple[dict[str, str], dict[str, str]]
) -> None:
    document, _key = _document(adapter, seals)
    for seal_name, key_name, replacement in [
        ("source_seal", "source_files_sha256", "0" * 64),
        ("analyzer_seal", "analyzer_files_sha256", "1" * 64),
    ]:
        invalidated = copy.deepcopy(document)
        invalidated[seal_name]["files_sha256"] = replacement
        invalidated["cache_key"][key_name] = replacement
        with pytest.raises(MachineFactError, match="official sealed cache key"):
            validate_machine_fact_document(invalidated, adapter=adapter)

    drifted_adapter = copy.deepcopy(adapter)
    drifted_adapter["analyzer_candidates"][0]["module"] = "tampered"
    with pytest.raises(MachineFactError, match="candidate subset hash"):
        validate_machine_fact_document(document, adapter=drifted_adapter)


def test_cache_round_trip_and_tamper_rejection(
    adapter: Mapping[str, Any], seals: tuple[dict[str, str], dict[str, str]]
) -> None:
    document, _key = _document(adapter, seals, result=_nonempty_v2(adapter))
    raw = machine_fact_cache_bytes(document, adapter=adapter)
    assert load_machine_fact_cache(raw, adapter=adapter) == document

    tampered_cases: list[dict[str, Any]] = []
    for path, replacement in [
        (("cache_key", "source_files_sha256"), "0" * 64),
        (("source_seal", "manifest_sha256"), "1" * 64),
        (("binding_mode",), "unsealed"),
        (("candidate_subset_sha256",), "2" * 64),
        (("counts", "supported_candidates"), 242),
        (("runtime_access", "access_facts", 0, "operation"), "tampered"),
    ]:
        value = copy.deepcopy(document)
        target: Any = value
        for component in path[:-1]:
            target = target[component]
        target[path[-1]] = replacement
        tampered_cases.append(value)
    for tampered in tampered_cases:
        with pytest.raises(MachineFactError):
            validate_machine_fact_document(tampered, adapter=adapter)


def test_cache_rejects_duplicate_nan_malformed_and_noncanonical_json(
    adapter: Mapping[str, Any], seals: tuple[dict[str, str], dict[str, str]]
) -> None:
    document, _key = _document(adapter, seals)
    raw = canonical_bytes(document)
    duplicate = b'{"adapter_schema_version":1,' + raw[1:]
    noncanonical = json.dumps(document, indent=2).encode("utf-8")
    for invalid, message in [
        (duplicate, "duplicate key"),
        (b'{"value":NaN}', "non-finite"),
        (b"{", "malformed"),
        (noncanonical, "canonical byte form"),
    ]:
        with pytest.raises(MachineFactError, match=message):
            load_machine_fact_cache(invalid, adapter=adapter)


def test_strict_integer_fields_reject_booleans(
    adapter: Mapping[str, Any], seals: tuple[dict[str, str], dict[str, str]]
) -> None:
    invalid_adapter = cast(dict[str, Any], copy.deepcopy(adapter))
    invalid_adapter["adapter_schema_version"] = True
    with pytest.raises(MachineFactError, match="adapter schema version"):
        validate_effective_adapter(invalid_adapter)

    document, _key = _document(adapter, seals)
    for path in [
        ("cache_schema_version",),
        ("adapter_schema_version",),
        ("cache_key", "adapter_schema_version"),
        ("counts", "resource_candidates"),
    ]:
        invalid = copy.deepcopy(document)
        target: Any = invalid
        for component in path[:-1]:
            target = target[component]
        target[path[-1]] = True
        with pytest.raises(MachineFactError):
            validate_machine_fact_document(invalid, adapter=adapter)

    runtime = _empty_v2()
    runtime["counts"]["accesses"] = True
    with pytest.raises(MachineFactError, match="must be an integer"):
        validate_runtime_access_v2(
            runtime, resource_locators=_resource_locators(adapter)
        )


def test_deep_json_recursion_is_normalized_to_machine_fact_error(
    adapter: Mapping[str, Any],
) -> None:
    nested: Any = 0
    for _ in range(2_000):
        nested = [nested]
    with pytest.raises(MachineFactError, match="canonical JSON"):
        canonical_bytes(nested)

    deeply_nested_json = b"[" * 2_000 + b"0" + b"]" * 2_000
    with pytest.raises(MachineFactError, match="JSON"):
        load_machine_fact_cache(deeply_nested_json, adapter=adapter)


def test_sharding_api_is_rejected_and_dynamic_open_proves_non_equivalence(
    adapter: Mapping[str, Any],
) -> None:
    with pytest.raises(MachineFactError, match=SHARDING_DISABLED_REASON):
        reject_sharded_analysis()
    with pytest.raises(MachineFactError, match=SHARDING_DISABLED_REASON):
        analyze_runtime_access_unsealed(
            {},
            adapter,
            analyzer=lambda _source, _candidates: _empty_v2(),
            shard_plan_id="two-shards",
        )

    resource_id = "runtime-resource:" + "1" * 64
    candidates = [
        {
            "id": resource_id,
            "module": "chronovisor.a",
            "symbol": "RESOURCE",
            "locator": {"type": "path", "value": "$CHRONOVISOR_ROOT/demo"},
        }
    ]
    snapshot = {
        "src/chronovisor/a.py": b'RESOURCE = "ignored"\ndef mode():\n    return "r"\n',
        "src/chronovisor/b.py": (
            b"from chronovisor.a import RESOURCE, mode\n"
            b"def run():\n    return open(RESOURCE, mode())\n"
        ),
    }
    monolithic = discover_access_facts(snapshot, candidates)
    validate_runtime_access_v2(
        monolithic, resource_locators={resource_id: "$CHRONOVISOR_ROOT/demo"}
    )
    validate_runtime_access_v2_result(monolithic)
    sharded = [
        discover_access_facts({path: raw}, candidates) for path, raw in snapshot.items()
    ]
    assert [row["reason"] for row in monolithic["escape_facts"]] == [
        "dynamic_open_mode"
    ]
    assert sum(result["counts"]["escapes"] for result in sharded) == 0
