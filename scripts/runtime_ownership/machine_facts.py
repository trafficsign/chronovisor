"""Seal declaration candidates into one schema-v2 runtime-access artifact.

The runtime access analysis is intentionally monolithic.  Splitting either the
source snapshot or the candidate set changes the analyzer's whole-program flow
semantics, so this adapter exposes no sharded execution path.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import re
import sys
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, TypeAlias, cast

from scripts.runtime_ownership.declarations import (
    EFFECTIVE_FEEDBACK_REVISION,
    ConcreteDeclarations,
    discover_concrete,
)
from scripts.runtime_ownership.manifests import (
    ANALYZER_MANIFEST_KIND,
    ANALYZER_PATHS,
    MACHINE_FACT_TOOLCHAIN_MANIFEST_KIND,
    MACHINE_FACT_TOOLCHAIN_PATHS,
    SOURCE_MANIFEST_KIND,
    CommittedSnapshot,
    ManifestError,
    build_manifest,
    current_head_revision,
    verify_manifest,
)

ADAPTER_SCHEMA_VERSION = 1
CACHE_SCHEMA_VERSION = 2
EFFECTIVE_SOURCE_REVISION = EFFECTIVE_FEEDBACK_REVISION
SOURCE_FILES_SHA256 = "be2ad06f687bc619a89d12ad6274d6843b26278e2094d420146105c398e73cee"
SOURCE_MANIFEST_SHA256 = (
    "268a6d8ca2fbd7d4877f78a3f5c6b14fd0e7e36d760173be9ce1a05e6703f43a"
)
ANALYZER_REVISION = "b934ecb3425f2121c0375f5bcd174736c21a8546"
ANALYZER_FILES_SHA256 = (
    "4c54b26db7e28d16de69b9c00183fab7f2c8fb411edd5d28c1d1728e1f3062f3"
)
ANALYZER_MANIFEST_SHA256 = (
    "3789c8b41f22e99217419e7bc1abfa49e6de8c3d9b6f18091847f294e1a709ec"
)
CANDIDATE_SUBSET_SHA256 = (
    "5a4ced36e086b4ee764b6ddb6e9decb1aa9b58b4e12ee3f77f2193ccf9ba96ec"
)
SHARD_PLAN_ID = "monolithic-v1"
SHARDING_DISABLED_REASON = "semantic_non_equivalence_risk"
PROVENANCE_PATH_LIMIT = 64
PROVENANCE_RETENTION_POLICY = "shortest_then_lexicographic"

_EXPECTED_RESOURCE_CANDIDATES = 490
_EXPECTED_SUPPORTED_CANDIDATES = 243
_EXPECTED_SUPPORTED_RESOURCES = 210
_EXPECTED_UNSUPPORTED_DECLARATIONS = 247
_EXPECTED_EXCLUDED_DECLARATIONS = 138
_SUPPORTED_PATH_KINDS = frozenset({"artifact", "queue", "lock"})
_OWNER_FIELDS = frozenset(
    {
        "owner",
        "owner_package",
        "owner_symbol",
        "writers",
        "readers",
        "lifecycle",
        "coordination",
    }
)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SHA1 = re.compile(r"[0-9a-f]{40}\Z")
_CALL_SITE_ID = re.compile(r"\|site_id=(runtime-site:[0-9a-f]{64})")
_RUNTIME_ID = {
    "site_id": re.compile(r"runtime-site:[0-9a-f]{64}\Z"),
    "provenance_id": re.compile(r"runtime-provenance:[0-9a-f]{64}\Z"),
    "access_fact_id": re.compile(r"runtime-access-fact:[0-9a-f]{64}\Z"),
    "escape_fact_id": re.compile(r"runtime-escape-fact:[0-9a-f]{64}\Z"),
    "resource_id": re.compile(r"runtime-resource:[0-9a-f]{64}\Z"),
    "discovery_id": re.compile(r"runtime-site:[0-9a-f]{64}\Z"),
}
_V2_ROOT_KEYS = frozenset(
    {
        "schema_version",
        "sites",
        "provenances",
        "provenance_ids",
        "access_facts",
        "escape_facts",
        "access_fact_ids",
        "escape_fact_ids",
        "counts",
    }
)
_CACHE_KEY_KEYS = frozenset(
    {
        "adapter_schema_version",
        "source_revision",
        "source_files_sha256",
        "source_manifest_sha256",
        "analyzer_revision",
        "analyzer_files_sha256",
        "analyzer_manifest_sha256",
        "toolchain_revision",
        "toolchain_files_sha256",
        "toolchain_manifest_sha256",
        "candidate_subset_sha256",
        "shard_plan_id",
    }
)
_SEAL_KEYS = frozenset({"revision", "files_sha256", "manifest_sha256"})
_WRAPPER_KEYS = frozenset(
    {
        "binding_mode",
        "cache_schema_version",
        "adapter_schema_version",
        "cache_key",
        "source_seal",
        "analyzer_seal",
        "toolchain_seal",
        "candidate_subset_sha256",
        "shard_plan",
        "runtime_access",
        "unsupported_declarations",
        "excluded_declarations",
        "counts",
    }
)
_WRAPPER_COUNT_KEYS = frozenset(
    {
        "resource_candidates",
        "supported_candidates",
        "supported_resources",
        "unsupported_declarations",
        "excluded_declarations",
        "sites",
        "provenances",
        "access_facts",
        "escape_facts",
    }
)

JsonObject: TypeAlias = dict[str, Any]
Analyzer: TypeAlias = Callable[
    [Mapping[str, bytes], Iterable[Mapping[str, Any]]], dict[str, Any]
]
_ANALYZER_MODULE_NAMES = frozenset(
    f"scripts.runtime_ownership.{Path(path).stem}" for path in ANALYZER_PATHS
)
_TOOLCHAIN_MODULE_BY_PATH = {
    path: f"scripts.runtime_ownership.{Path(path).stem}"
    for path in MACHINE_FACT_TOOLCHAIN_PATHS
}


class MachineFactError(ValueError):
    """Raised when adapter input, analysis output, or cache data drifts."""


def _plain_json(value: Any, active: set[int] | None = None) -> Any:
    seen = set() if active is None else active
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in seen:
            raise MachineFactError("canonical JSON contains a container cycle")
        seen.add(identity)
        try:
            result: dict[str, Any] = {}
            for key, item in value.items():
                if type(key) is not str:
                    raise MachineFactError("canonical JSON keys must be exact strings")
                result[key] = _plain_json(item, seen)
            return result
        finally:
            seen.remove(identity)
    if isinstance(value, (list, tuple)):
        identity = id(value)
        if identity in seen:
            raise MachineFactError("canonical JSON contains a container cycle")
        seen.add(identity)
        try:
            return [_plain_json(item, seen) for item in value]
        finally:
            seen.remove(identity)
    if value is None or type(value) in {str, int, float, bool}:
        return value
    raise MachineFactError(f"value is not JSON-compatible: {type(value).__name__}")


def canonical_bytes(value: Any) -> bytes:
    """Return the sole accepted byte representation for cache JSON."""

    try:
        return json.dumps(
            _plain_json(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except MachineFactError:
        raise
    except (TypeError, ValueError, UnicodeEncodeError, RecursionError) as exc:
        raise MachineFactError("value is not canonical JSON") from exc


def _sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _stable_id(prefix: str, value: Mapping[str, Any]) -> str:
    return f"{prefix}:{_sha256(value)}"


def _object(value: object, keys: frozenset[str], *, label: str) -> JsonObject:
    if type(value) is not dict:
        raise MachineFactError(f"{label} must be a JSON object")
    row = cast(JsonObject, value)
    raw_keys = tuple(row)
    if any(type(key) is not str for key in raw_keys):
        raise MachineFactError(f"{label} keys must be exact strings")
    actual_keys = set(raw_keys)
    if actual_keys != keys:
        raise MachineFactError(
            f"{label} keys mismatch: missing={sorted(keys - actual_keys)}, "
            f"unknown={sorted(actual_keys - keys)}"
        )
    return row


def _array(value: object, *, label: str) -> list[Any]:
    if type(value) is not list:
        raise MachineFactError(f"{label} must be a JSON array")
    return value


def _string(value: object, *, label: str) -> str:
    if type(value) is not str or not value or "\0" in value:
        raise MachineFactError(f"{label} must be a non-empty exact string")
    return value


def _integer(value: object, *, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise MachineFactError(f"{label} must be an integer >= {minimum}")
    return value


def _boolean(value: object, *, label: str) -> bool:
    if type(value) is not bool:
        raise MachineFactError(f"{label} must be an exact boolean")
    return value


def _identity(value: object, kind: str, *, label: str) -> str:
    text = _string(value, label=label)
    if _RUNTIME_ID[kind].fullmatch(text) is None:
        raise MachineFactError(f"{label} has an invalid identity")
    return text


def _strings(value: object, *, label: str, sorted_unique: bool = False) -> list[str]:
    rows = _array(value, label=label)
    result = [_string(item, label=f"{label}[]") for item in rows]
    if sorted_unique and (result != sorted(result) or len(result) != len(set(result))):
        raise MachineFactError(f"{label} must be sorted and unique")
    return result


def _assert_no_owner_fields(value: Any, *, label: str = "artifact") -> None:
    if isinstance(value, Mapping):
        forbidden = _OWNER_FIELDS.intersection(value)
        if forbidden:
            raise MachineFactError(
                f"{label} contains ownership-policy fields: {sorted(forbidden)}"
            )
        for key, item in value.items():
            _assert_no_owner_fields(item, label=f"{label}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_no_owner_fields(item, label=f"{label}[{index}]")


def _evidence(candidate: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = [
        {
            "path": _string(candidate.get("path"), label="candidate.path"),
            "line": _integer(candidate.get("line"), label="candidate.line", minimum=1),
        }
    ]
    additional = candidate.get("additional_evidence", ())
    if not isinstance(additional, (list, tuple)):
        raise MachineFactError("candidate.additional_evidence must be a sequence")
    for index, item in enumerate(additional):
        if not isinstance(item, Mapping) or set(item) != {"path", "line"}:
            raise MachineFactError(
                f"candidate.additional_evidence[{index}] has invalid keys"
            )
        rows.append(
            {
                "path": _string(item["path"], label="evidence.path"),
                "line": _integer(item["line"], label="evidence.line", minimum=1),
            }
        )
    distinct = {(str(row["path"]), int(row["line"])) for row in rows}
    return [{"path": path, "line": line} for path, line in sorted(distinct)]


def _locator(candidate: Mapping[str, Any]) -> dict[str, str]:
    value = candidate.get("locator")
    if not isinstance(value, Mapping) or set(value) != {"type", "value"}:
        raise MachineFactError("candidate.locator must contain exactly type and value")
    return {
        "type": _string(value["type"], label="candidate.locator.type"),
        "value": _string(value["value"], label="candidate.locator.value"),
    }


def _candidate_reason(kind: str, locator: Mapping[str, str]) -> str | None:
    if kind in _SUPPORTED_PATH_KINDS:
        if locator["type"] != "path":
            raise MachineFactError(
                f"supported {kind} candidate must use a path locator"
            )
        return None
    if kind == "socket":
        if locator["type"] != "socket":
            raise MachineFactError("socket candidate must use a socket locator")
        return (
            None
            if locator["value"].startswith("unix://")
            else ("unsupported_socket_locator")
        )
    if kind in {"schema", "worker"}:
        return f"unsupported_resource_kind:{kind}"
    raise MachineFactError(f"unknown declaration resource kind: {kind}")


def _index_grouped_resources(
    declarations: ConcreteDeclarations,
) -> dict[tuple[str, str], Mapping[str, Any]]:
    resources: dict[tuple[str, str], Mapping[str, Any]] = {}
    resource_ids: set[str] = set()
    for index, resource in enumerate(declarations.resources):
        if not isinstance(resource, Mapping):
            raise MachineFactError(f"resources[{index}] must be an object")
        kind = _string(resource.get("kind"), label=f"resources[{index}].kind")
        locator = _locator(resource)
        resource_id = _identity(
            resource.get("id"), "resource_id", label=f"resources[{index}].id"
        )
        key = (kind, locator["value"])
        if key in resources or resource_id in resource_ids:
            raise MachineFactError("declaration resources must be uniquely grouped")
        resources[key] = resource
        resource_ids.add(resource_id)
    return resources


def _partition_resource_candidates(
    declarations: ConcreteDeclarations,
    resources: Mapping[tuple[str, str], Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    analyzer_candidates: list[dict[str, Any]] = []
    unsupported: list[dict[str, Any]] = []
    all_discovery_ids: list[str] = []
    supported_discovery_ids: list[str] = []
    for index, candidate in enumerate(declarations.resource_candidates):
        if not isinstance(candidate, Mapping):
            raise MachineFactError(f"resource_candidates[{index}] must be an object")
        discovery_id = _identity(
            candidate.get("discovery_id"),
            "discovery_id",
            label=f"resource_candidates[{index}].discovery_id",
        )
        kind = _string(
            candidate.get("kind"), label=f"resource_candidates[{index}].kind"
        )
        locator = _locator(candidate)
        reason = _candidate_reason(kind, locator)
        all_discovery_ids.append(discovery_id)
        if reason is not None:
            unsupported.append(
                {
                    "discovery_id": discovery_id,
                    "kind": kind,
                    "locator": locator,
                    "evidence": _evidence(candidate),
                    "reason": reason,
                }
            )
            continue
        grouped_resource = resources.get((kind, locator["value"]))
        if grouped_resource is None:
            raise MachineFactError(
                f"supported declaration has no grouped resource: {kind}:{locator['value']}"
            )
        analyzer_candidates.append(
            {
                "id": _identity(
                    grouped_resource.get("id"), "resource_id", label="resource.id"
                ),
                "module": _string(candidate.get("module"), label="candidate.module"),
                "symbol": _string(candidate.get("symbol"), label="candidate.symbol"),
                "locator": locator,
            }
        )
        supported_discovery_ids.append(discovery_id)

    if len(all_discovery_ids) != len(set(all_discovery_ids)):
        raise MachineFactError("resource candidate discovery IDs must be unique")
    if set(supported_discovery_ids).intersection(
        str(row["discovery_id"]) for row in unsupported
    ):
        raise MachineFactError("supported and unsupported declaration IDs overlap")
    if len(analyzer_candidates) + len(unsupported) != len(all_discovery_ids):
        raise MachineFactError("resource candidate partition is incomplete")
    return analyzer_candidates, unsupported, all_discovery_ids


def _normalize_exclusions(
    declarations: ConcreteDeclarations, resource_discovery_ids: Sequence[str]
) -> list[dict[str, Any]]:
    excluded: list[dict[str, Any]] = []
    exclusion_ids: list[str] = []
    for index, candidate in enumerate(declarations.exclusion_candidates):
        if not isinstance(candidate, Mapping):
            raise MachineFactError(f"exclusion_candidates[{index}] must be an object")
        discovery_id = _identity(
            candidate.get("discovery_id"),
            "discovery_id",
            label=f"exclusion_candidates[{index}].discovery_id",
        )
        exclusion_ids.append(discovery_id)
        excluded.append(
            {
                "discovery_id": discovery_id,
                "module": _string(candidate.get("module"), label="exclusion.module"),
                "symbol": _string(candidate.get("symbol"), label="exclusion.symbol"),
                "evidence": _evidence(candidate),
                "reason": _string(candidate.get("reason"), label="exclusion.reason"),
            }
        )
    if len(exclusion_ids) != len(set(exclusion_ids)):
        raise MachineFactError("excluded declaration IDs must be unique")
    if set(exclusion_ids).intersection(resource_discovery_ids):
        raise MachineFactError("resource and excluded declaration IDs overlap")
    return excluded


def build_declaration_adapter(declarations: ConcreteDeclarations) -> dict[str, Any]:
    """Build a generic strict adapter without applying the effective seals."""

    if type(declarations) is not ConcreteDeclarations:
        raise MachineFactError("declarations must be ConcreteDeclarations")
    resources = _index_grouped_resources(declarations)
    analyzer_candidates, unsupported, all_discovery_ids = (
        _partition_resource_candidates(declarations, resources)
    )
    excluded = _normalize_exclusions(declarations, all_discovery_ids)

    analyzer_candidates.sort(key=lambda row: canonical_bytes(row))
    unsupported.sort(key=lambda row: str(row["discovery_id"]))
    excluded.sort(key=lambda row: str(row["discovery_id"]))
    supported_resource_ids = sorted({str(row["id"]) for row in analyzer_candidates})
    result = {
        "adapter_schema_version": ADAPTER_SCHEMA_VERSION,
        "analyzer_candidates": analyzer_candidates,
        "unsupported_declarations": unsupported,
        "excluded_declarations": excluded,
        "supported_resource_ids": supported_resource_ids,
        "candidate_subset_sha256": _sha256(analyzer_candidates),
        "counts": {
            "resource_candidates": len(all_discovery_ids),
            "supported_candidates": len(analyzer_candidates),
            "supported_resources": len(supported_resource_ids),
            "unsupported_declarations": len(unsupported),
            "excluded_declarations": len(excluded),
        },
    }
    _assert_no_owner_fields(result)
    return result


def validate_effective_adapter(adapter: Mapping[str, Any]) -> None:
    """Require the reviewed effective declaration partition and subset seal."""

    _validate_adapter_integrity(adapter)
    counts = adapter.get("counts")
    expected = {
        "resource_candidates": _EXPECTED_RESOURCE_CANDIDATES,
        "supported_candidates": _EXPECTED_SUPPORTED_CANDIDATES,
        "supported_resources": _EXPECTED_SUPPORTED_RESOURCES,
        "unsupported_declarations": _EXPECTED_UNSUPPORTED_DECLARATIONS,
        "excluded_declarations": _EXPECTED_EXCLUDED_DECLARATIONS,
    }
    if counts != expected:
        raise MachineFactError(f"effective adapter counts drifted: {counts!r}")
    if adapter.get("candidate_subset_sha256") != CANDIDATE_SUBSET_SHA256:
        raise MachineFactError("effective candidate subset seal drifted")
    reasons = Counter(
        str(row["reason"])
        for row in cast(
            Sequence[Mapping[str, Any]], adapter["unsupported_declarations"]
        )
    )
    if reasons != Counter(
        {
            "unsupported_resource_kind:schema": 155,
            "unsupported_resource_kind:worker": 88,
            "unsupported_socket_locator": 4,
        }
    ):
        raise MachineFactError(f"unsupported declaration reasons drifted: {reasons}")


def _validate_adapter_evidence(value: object, *, label: str) -> None:
    rows = _array(value, label=label)
    identities: list[tuple[str, int]] = []
    for index, value_row in enumerate(rows):
        row_label = f"{label}[{index}]"
        row = _object(value_row, frozenset({"path", "line"}), label=row_label)
        identities.append(
            (
                _string(row["path"], label=f"{row_label}.path"),
                _integer(row["line"], label=f"{row_label}.line", minimum=1),
            )
        )
    if not identities or identities != sorted(set(identities)):
        raise MachineFactError(f"{label} must be non-empty, sorted, and unique")


def _validate_adapter_integrity(adapter: Mapping[str, Any]) -> None:
    expected_keys = {
        "adapter_schema_version",
        "analyzer_candidates",
        "unsupported_declarations",
        "excluded_declarations",
        "supported_resource_ids",
        "candidate_subset_sha256",
        "counts",
    }
    if set(adapter) != expected_keys:
        raise MachineFactError("adapter keys drifted")
    if (
        type(adapter["adapter_schema_version"]) is not int
        or adapter["adapter_schema_version"] != ADAPTER_SCHEMA_VERSION
    ):
        raise MachineFactError("adapter schema version drifted")
    candidates_value = _plain_json(adapter["analyzer_candidates"])
    candidates = _array(candidates_value, label="analyzer_candidates")
    for index, value_row in enumerate(candidates):
        label = f"analyzer_candidates[{index}]"
        row = _object(
            value_row,
            frozenset({"id", "module", "symbol", "locator"}),
            label=label,
        )
        _identity(row["id"], "resource_id", label=f"{label}.id")
        _string(row["module"], label=f"{label}.module")
        _string(row["symbol"], label=f"{label}.symbol")
        _locator(row)
    if candidates != sorted(candidates, key=canonical_bytes):
        raise MachineFactError("analyzer candidates must be in canonical row order")
    candidate_hash = _string(
        adapter["candidate_subset_sha256"], label="candidate_subset_sha256"
    )
    if (
        _SHA256.fullmatch(candidate_hash) is None
        or _sha256(candidates) != candidate_hash
    ):
        raise MachineFactError(
            "candidate subset hash does not match analyzer candidates"
        )

    unsupported_value = _plain_json(adapter["unsupported_declarations"])
    unsupported = _array(unsupported_value, label="unsupported_declarations")
    unsupported_ids: list[str] = []
    for index, value_row in enumerate(unsupported):
        label = f"unsupported_declarations[{index}]"
        row = _object(
            value_row,
            frozenset({"discovery_id", "kind", "locator", "evidence", "reason"}),
            label=label,
        )
        unsupported_ids.append(
            _identity(
                row["discovery_id"], "discovery_id", label=f"{label}.discovery_id"
            )
        )
        kind = _string(row["kind"], label=f"{label}.kind")
        locator = _locator(row)
        reason = _string(row["reason"], label=f"{label}.reason")
        if _candidate_reason(kind, locator) != reason:
            raise MachineFactError(f"{label}.reason does not match its declaration")
        _validate_adapter_evidence(row["evidence"], label=f"{label}.evidence")
    if unsupported_ids != sorted(unsupported_ids) or len(unsupported_ids) != len(
        set(unsupported_ids)
    ):
        raise MachineFactError("unsupported declaration IDs must be sorted and unique")

    excluded_value = _plain_json(adapter["excluded_declarations"])
    excluded = _array(excluded_value, label="excluded_declarations")
    excluded_ids: list[str] = []
    for index, value_row in enumerate(excluded):
        label = f"excluded_declarations[{index}]"
        row = _object(
            value_row,
            frozenset({"discovery_id", "module", "symbol", "evidence", "reason"}),
            label=label,
        )
        excluded_ids.append(
            _identity(
                row["discovery_id"], "discovery_id", label=f"{label}.discovery_id"
            )
        )
        _string(row["module"], label=f"{label}.module")
        _string(row["symbol"], label=f"{label}.symbol")
        _string(row["reason"], label=f"{label}.reason")
        _validate_adapter_evidence(row["evidence"], label=f"{label}.evidence")
    if excluded_ids != sorted(excluded_ids) or len(excluded_ids) != len(
        set(excluded_ids)
    ):
        raise MachineFactError("excluded declaration IDs must be sorted and unique")
    if set(unsupported_ids).intersection(excluded_ids):
        raise MachineFactError("unsupported and excluded declaration IDs overlap")

    resource_ids = _strings(
        _plain_json(adapter["supported_resource_ids"]),
        label="supported_resource_ids",
        sorted_unique=True,
    )
    candidate_resource_ids = sorted({str(row["id"]) for row in candidates})
    if resource_ids != candidate_resource_ids:
        raise MachineFactError("supported resource IDs do not match candidates")
    expected_counts = {
        "resource_candidates": len(candidates) + len(unsupported),
        "supported_candidates": len(candidates),
        "supported_resources": len(resource_ids),
        "unsupported_declarations": len(unsupported),
        "excluded_declarations": len(excluded),
    }
    counts = _object(
        adapter["counts"], frozenset(expected_counts), label="adapter.counts"
    )
    for key, value in counts.items():
        _integer(value, label=f"adapter.counts.{key}")
    if counts != expected_counts:
        raise MachineFactError("adapter counts do not match declaration rows")
    _assert_no_owner_fields(adapter)


def _validate_evidence_row(value: object, *, label: str) -> None:
    row = _object(value, frozenset({"path", "line"}), label=label)
    _string(row["path"], label=f"{label}.path")
    _integer(row["line"], label=f"{label}.line", minimum=1)


def _validate_sites(value: object) -> tuple[dict[str, JsonObject], set[str]]:
    rows = _array(value, label="runtime_access.sites")
    indexed: dict[str, JsonObject] = {}
    identities: list[str] = []
    for index, value_row in enumerate(rows):
        label = f"runtime_access.sites[{index}]"
        row = _object(
            value_row,
            frozenset({"site_id", "scope", "kind", "syntax", "occurrence", "evidence"}),
            label=label,
        )
        site_id = _identity(row["site_id"], "site_id", label=f"{label}.site_id")
        scope = _string(row["scope"], label=f"{label}.scope")
        kind = _string(row["kind"], label=f"{label}.kind")
        syntax = _string(row["syntax"], label=f"{label}.syntax")
        occurrence = _integer(row["occurrence"], label=f"{label}.occurrence", minimum=1)
        evidence = _object(
            row["evidence"], frozenset({"path", "line"}), label=f"{label}.evidence"
        )
        _validate_evidence_row(evidence, label=f"{label}.evidence")
        expected = _stable_id(
            "runtime-site",
            {
                "path": evidence["path"],
                "scope": scope,
                "kind": kind,
                "syntax": syntax,
                "occurrence": occurrence,
            },
        )
        if site_id != expected:
            raise MachineFactError(f"{label}.site_id does not match its identity")
        identities.append(site_id)
        indexed[site_id] = row
    if identities != sorted(identities) or len(identities) != len(set(identities)):
        raise MachineFactError("site identities must be sorted and unique")
    return indexed, set(identities)


def _validate_provenances(
    value: object, resource_locators: Mapping[str, str]
) -> tuple[dict[str, JsonObject], set[str], set[str]]:
    rows = _array(value, label="runtime_access.provenances")
    indexed: dict[str, JsonObject] = {}
    identities: list[str] = []
    called_sites: set[str] = set()
    for index, value_row in enumerate(rows):
        label = f"runtime_access.provenances[{index}]"
        row = _object(
            value_row,
            frozenset(
                {
                    "provenance_id",
                    "resource_id",
                    "actor",
                    "binding_chain",
                    "locator",
                    "call_site_ids",
                }
            ),
            label=label,
        )
        provenance_id = _identity(
            row["provenance_id"], "provenance_id", label=f"{label}.provenance_id"
        )
        resource_id = _identity(
            row["resource_id"], "resource_id", label=f"{label}.resource_id"
        )
        if resource_id not in resource_locators:
            raise MachineFactError(f"{label} references an unsupported resource")
        actor = _string(row["actor"], label=f"{label}.actor")
        binding_chain = _strings(row["binding_chain"], label=f"{label}.binding_chain")
        locator = _string(row["locator"], label=f"{label}.locator")
        if locator != resource_locators[resource_id]:
            raise MachineFactError(f"{label}.locator does not match its resource")
        call_site_ids = _strings(
            row["call_site_ids"], label=f"{label}.call_site_ids", sorted_unique=True
        )
        for site_id in call_site_ids:
            _identity(site_id, "site_id", label=f"{label}.call_site_ids[]")
        expected_call_site_ids = sorted(
            {
                match.group(1)
                for step in binding_chain
                for match in _CALL_SITE_ID.finditer(step)
            }
        )
        if call_site_ids != expected_call_site_ids:
            raise MachineFactError(
                f"{label}.call_site_ids do not match its binding chain"
            )
        expected = _stable_id(
            "runtime-provenance",
            {
                "resource_id": resource_id,
                "actor": actor,
                "binding_chain": binding_chain,
            },
        )
        if provenance_id != expected:
            raise MachineFactError(f"{label}.provenance_id does not match its identity")
        identities.append(provenance_id)
        indexed[provenance_id] = row
        called_sites.update(call_site_ids)
    if identities != sorted(identities) or len(identities) != len(set(identities)):
        raise MachineFactError("provenance identities must be sorted and unique")
    return indexed, set(identities), called_sites


def _validate_fact_common(
    row: JsonObject,
    *,
    label: str,
    resource_locators: Mapping[str, str],
) -> tuple[str, str, list[str], list[str]]:
    site_id = _identity(row["site_id"], "site_id", label=f"{label}.site_id")
    resource_id = _identity(
        row["resource_id"], "resource_id", label=f"{label}.resource_id"
    )
    if resource_id not in resource_locators:
        raise MachineFactError(f"{label} references an unsupported resource")
    _string(row["operation"], label=f"{label}.operation")
    _string(row["sink"], label=f"{label}.sink")
    locator = _string(row["locator"], label=f"{label}.locator")
    if locator != resource_locators[resource_id]:
        raise MachineFactError(f"{label}.locator does not match its resource")
    provenance_ids = _strings(
        row["provenance_ids"], label=f"{label}.provenance_ids", sorted_unique=True
    )
    for provenance_id in provenance_ids:
        _identity(provenance_id, "provenance_id", label=f"{label}.provenance_ids[]")
    actors = _strings(row["actors"], label=f"{label}.actors", sorted_unique=True)
    _boolean(row["provenance_complete"], label=f"{label}.provenance_complete")
    return site_id, resource_id, provenance_ids, actors


def _validate_accesses(
    value: object, resource_locators: Mapping[str, str]
) -> tuple[dict[str, JsonObject], list[str], set[str]]:
    rows = _array(value, label="runtime_access.access_facts")
    indexed: dict[str, JsonObject] = {}
    identities: list[str] = []
    sites: set[str] = set()
    keys = frozenset(
        {
            "access_fact_id",
            "site_id",
            "resource_id",
            "mode",
            "operation",
            "sink",
            "sink_actor",
            "locator",
            "provenance_ids",
            "actors",
            "provenance_complete",
        }
    )
    for index, value_row in enumerate(rows):
        label = f"runtime_access.access_facts[{index}]"
        row = _object(value_row, keys, label=label)
        fact_id = _identity(
            row["access_fact_id"], "access_fact_id", label=f"{label}.access_fact_id"
        )
        site_id, resource_id, _provenance_ids, _actors = _validate_fact_common(
            row, label=label, resource_locators=resource_locators
        )
        mode = _string(row["mode"], label=f"{label}.mode")
        if mode not in {"read", "write", "read_write"}:
            raise MachineFactError(f"{label}.mode is unsupported")
        sink_actor = _string(row["sink_actor"], label=f"{label}.sink_actor")
        expected = _stable_id(
            "runtime-access-fact",
            {
                "site_id": site_id,
                "resource_id": resource_id,
                "mode": mode,
                "operation": row["operation"],
                "sink": row["sink"],
                "sink_actor": sink_actor,
            },
        )
        if fact_id != expected:
            raise MachineFactError(
                f"{label}.access_fact_id does not match its identity"
            )
        identities.append(fact_id)
        indexed[fact_id] = row
        sites.add(site_id)
    if identities != sorted(identities) or len(identities) != len(set(identities)):
        raise MachineFactError("access fact identities must be sorted and unique")
    return indexed, identities, sites


def _validate_escapes(
    value: object, resource_locators: Mapping[str, str]
) -> tuple[dict[str, JsonObject], list[str], set[str]]:
    rows = _array(value, label="runtime_access.escape_facts")
    indexed: dict[str, JsonObject] = {}
    identities: list[str] = []
    sites: set[str] = set()
    normal = frozenset(
        {
            "escape_fact_id",
            "site_id",
            "resource_id",
            "operation",
            "sink",
            "reason",
            "locator",
            "provenance_ids",
            "actors",
            "provenance_complete",
        }
    )
    overflow_base = normal.union(
        {"source_kind", "source_fact_id", "limit", "retention_policy"}
    )
    for index, value_row in enumerate(rows):
        label = f"runtime_access.escape_facts[{index}]"
        if type(value_row) is not dict:
            raise MachineFactError(f"{label} must be a JSON object")
        raw = cast(JsonObject, value_row)
        source_kind = raw.get("source_kind")
        if source_kind is None:
            keys = normal
        elif source_kind == "access":
            keys = overflow_base.union({"mode", "sink_actor"})
        elif source_kind == "escape":
            keys = overflow_base.union({"source_reason"})
        else:
            raise MachineFactError(f"{label}.source_kind is invalid")
        row = _object(value_row, keys, label=label)
        fact_id = _identity(
            row["escape_fact_id"], "escape_fact_id", label=f"{label}.escape_fact_id"
        )
        site_id, resource_id, provenance_ids, actors = _validate_fact_common(
            row, label=label, resource_locators=resource_locators
        )
        reason = _string(row["reason"], label=f"{label}.reason")
        if source_kind is None and reason == "provenance_overflow":
            raise MachineFactError(
                f"{label} provenance_overflow requires the extended overflow shape"
            )
        identity: dict[str, Any] = {
            "site_id": site_id,
            "resource_id": resource_id,
            "operation": row["operation"],
            "sink": row["sink"],
            "reason": reason,
        }
        if source_kind is not None:
            if (
                reason != "provenance_overflow"
                or provenance_ids
                or actors
                or row["provenance_complete"] is not False
            ):
                raise MachineFactError(f"{label} has an invalid overflow shape")
            limit = _integer(row["limit"], label=f"{label}.limit", minimum=1)
            retention_policy = _string(
                row["retention_policy"], label=f"{label}.retention_policy"
            )
            if (
                limit != PROVENANCE_PATH_LIMIT
                or retention_policy != PROVENANCE_RETENTION_POLICY
            ):
                raise MachineFactError(f"{label} overflow retention contract drifted")
            source_fact_id = _string(
                row["source_fact_id"], label=f"{label}.source_fact_id"
            )
            if source_kind == "access":
                _identity(
                    source_fact_id, "access_fact_id", label=f"{label}.source_fact_id"
                )
                mode = _string(row["mode"], label=f"{label}.mode")
                if mode not in {"read", "write", "read_write"}:
                    raise MachineFactError(f"{label}.mode is unsupported")
                _string(row["sink_actor"], label=f"{label}.sink_actor")
            else:
                _identity(
                    source_fact_id, "escape_fact_id", label=f"{label}.source_fact_id"
                )
                _string(row["source_reason"], label=f"{label}.source_reason")
            identity = {
                **identity,
                "source_kind": source_kind,
                "source_fact_id": source_fact_id,
                "limit": row["limit"],
                "retention_policy": row["retention_policy"],
            }
            if source_kind == "access":
                identity.update({"mode": row["mode"], "sink_actor": row["sink_actor"]})
            else:
                identity["source_reason"] = row["source_reason"]
        expected = _stable_id("runtime-escape-fact", identity)
        if fact_id != expected:
            raise MachineFactError(
                f"{label}.escape_fact_id does not match its identity"
            )
        identities.append(fact_id)
        indexed[fact_id] = row
        sites.add(site_id)
    if identities != sorted(identities) or len(identities) != len(set(identities)):
        raise MachineFactError("escape fact identities must be sorted and unique")
    return indexed, identities, sites


def validate_runtime_access_v2(
    result: object,
    *,
    resource_locators: Mapping[str, str],
) -> None:
    """Validate exact schema-v2 structure, identities, and all relations."""

    root = _object(result, _V2_ROOT_KEYS, label="runtime_access")
    if type(root["schema_version"]) is not int or root["schema_version"] != 2:
        raise MachineFactError("runtime_access.schema_version must be integer 2")
    supported = set(resource_locators)
    if not supported or any(
        _RUNTIME_ID["resource_id"].fullmatch(item) is None for item in supported
    ):
        raise MachineFactError("supported resource identities are invalid")
    if any(type(value) is not str or not value for value in resource_locators.values()):
        raise MachineFactError("supported resource locators are invalid")
    sites, site_ids = _validate_sites(root["sites"])
    provenances, provenance_ids, call_site_ids = _validate_provenances(
        root["provenances"], resource_locators
    )
    access_facts, access_ids, fact_site_ids = _validate_accesses(
        root["access_facts"], resource_locators
    )
    escape_facts, escape_ids, escape_site_ids = _validate_escapes(
        root["escape_facts"], resource_locators
    )
    if _strings(
        root["provenance_ids"],
        label="runtime_access.provenance_ids",
        sorted_unique=True,
    ) != sorted(provenance_ids):
        raise MachineFactError("provenance_ids do not match provenance rows")
    if (
        _strings(
            root["access_fact_ids"],
            label="runtime_access.access_fact_ids",
            sorted_unique=True,
        )
        != access_ids
    ):
        raise MachineFactError("access_fact_ids do not match access fact rows")
    if (
        _strings(
            root["escape_fact_ids"],
            label="runtime_access.escape_fact_ids",
            sorted_unique=True,
        )
        != escape_ids
    ):
        raise MachineFactError("escape_fact_ids do not match escape fact rows")
    if not call_site_ids.issubset(site_ids):
        raise MachineFactError("provenance references an unknown call site")
    referenced_sites = fact_site_ids | escape_site_ids | call_site_ids
    if referenced_sites != site_ids:
        raise MachineFactError(
            "runtime access sites contain unknown references or orphans"
        )

    referenced_provenances: set[str] = set()
    for label, fact in [
        *(("access fact", row) for row in access_facts.values()),
        *(("escape fact", row) for row in escape_facts.values()),
    ]:
        fact_resource = str(fact["resource_id"])
        fact_provenance_ids = cast(list[str], fact["provenance_ids"])
        fact_actors = cast(list[str], fact["actors"])
        joined_actors: set[str] = set()
        for provenance_id in fact_provenance_ids:
            provenance = provenances.get(provenance_id)
            if provenance is None:
                raise MachineFactError(f"{label} references unknown provenance")
            if provenance["resource_id"] != fact_resource:
                raise MachineFactError(f"{label} provenance resource does not match")
            if provenance["locator"] != fact["locator"]:
                raise MachineFactError(f"{label} provenance locator does not match")
            joined_actors.add(str(provenance["actor"]))
            referenced_provenances.add(provenance_id)
        if fact_actors != sorted(joined_actors):
            raise MachineFactError(f"{label} actors do not match its provenances")
    if referenced_provenances != provenance_ids:
        raise MachineFactError("runtime access provenances contain orphans")
    for fact in escape_facts.values():
        source_kind = fact.get("source_kind")
        if source_kind == "access":
            access_source = access_facts.get(str(fact["source_fact_id"]))
            if access_source is None:
                raise MachineFactError("overflow escape references unknown access fact")
            access_fields = (
                "site_id",
                "resource_id",
                "operation",
                "sink",
                "mode",
                "sink_actor",
            )
            if any(fact[field] != access_source[field] for field in access_fields):
                raise MachineFactError(
                    "access overflow identity does not match its source fact"
                )
        if source_kind == "escape":
            escape_source = escape_facts.get(str(fact["source_fact_id"]))
            if escape_source is None:
                raise MachineFactError("overflow escape references unknown escape fact")
            if escape_source is fact or escape_source.get("source_kind") is not None:
                raise MachineFactError(
                    "escape overflow source must be a non-overflow escape fact"
                )
            escape_fields = ("site_id", "resource_id", "operation", "sink")
            if any(fact[field] != escape_source[field] for field in escape_fields) or (
                fact["source_reason"] != escape_source["reason"]
            ):
                raise MachineFactError(
                    "escape overflow identity does not match its source fact"
                )
    access_overflow_sources = {
        str(fact["source_fact_id"])
        for fact in escape_facts.values()
        if fact.get("source_kind") == "access"
    }
    escape_overflow_sources = {
        str(fact["source_fact_id"])
        for fact in escape_facts.values()
        if fact.get("source_kind") == "escape"
    }
    for fact_id, fact in access_facts.items():
        expected_complete = fact_id not in access_overflow_sources
        if fact["provenance_complete"] is not expected_complete:
            raise MachineFactError(
                "access fact provenance_complete does not match overflow coverage"
            )
    for fact_id, fact in escape_facts.items():
        if fact.get("source_kind") is not None:
            continue
        expected_complete = fact_id not in escape_overflow_sources
        if fact["provenance_complete"] is not expected_complete:
            raise MachineFactError(
                "escape fact provenance_complete does not match overflow coverage"
            )

    del sites
    counts = _object(
        root["counts"],
        frozenset({"accesses", "escapes", "read", "write", "read_write"}),
        label="runtime_access.counts",
    )
    for key, value in counts.items():
        _integer(value, label=f"runtime_access.counts.{key}")
    expected_counts = {
        "accesses": len(access_facts),
        "escapes": len(escape_facts),
        "read": sum(row["mode"] == "read" for row in access_facts.values()),
        "write": sum(row["mode"] == "write" for row in access_facts.values()),
        "read_write": sum(row["mode"] == "read_write" for row in access_facts.values()),
    }
    if counts != expected_counts:
        raise MachineFactError("runtime access counts do not match fact rows")


def cache_key_metadata(
    *,
    source_seal: Mapping[str, Any],
    analyzer_seal: Mapping[str, Any],
    toolchain_seal: Mapping[str, Any],
    candidate_subset_sha256: str,
    shard_plan_id: str = SHARD_PLAN_ID,
) -> dict[str, Any]:
    """Build the exact metadata object that identifies one reusable result."""

    if any(
        set(seal) != _SEAL_KEYS for seal in (source_seal, analyzer_seal, toolchain_seal)
    ):
        raise MachineFactError(
            "source, analyzer, and toolchain seals must have exact keys"
        )
    for label, seal in (
        ("source", source_seal),
        ("analyzer", analyzer_seal),
        ("toolchain", toolchain_seal),
    ):
        revision = seal["revision"]
        files_hash = seal["files_sha256"]
        manifest_hash = seal["manifest_sha256"]
        if type(revision) is not str or _SHA1.fullmatch(revision) is None:
            raise MachineFactError(f"{label} seal revision is invalid")
        if type(files_hash) is not str or _SHA256.fullmatch(files_hash) is None:
            raise MachineFactError(f"{label} files seal is invalid")
        if type(manifest_hash) is not str or _SHA256.fullmatch(manifest_hash) is None:
            raise MachineFactError(f"{label} manifest seal is invalid")
    if _SHA256.fullmatch(candidate_subset_sha256) is None:
        raise MachineFactError("candidate subset hash is invalid")
    if shard_plan_id != SHARD_PLAN_ID:
        raise MachineFactError(f"sharding is forbidden: {SHARDING_DISABLED_REASON}")
    return {
        "adapter_schema_version": ADAPTER_SCHEMA_VERSION,
        "source_revision": source_seal["revision"],
        "source_files_sha256": source_seal["files_sha256"],
        "source_manifest_sha256": source_seal["manifest_sha256"],
        "analyzer_revision": analyzer_seal["revision"],
        "analyzer_files_sha256": analyzer_seal["files_sha256"],
        "analyzer_manifest_sha256": analyzer_seal["manifest_sha256"],
        "toolchain_revision": toolchain_seal["revision"],
        "toolchain_files_sha256": toolchain_seal["files_sha256"],
        "toolchain_manifest_sha256": toolchain_seal["manifest_sha256"],
        "candidate_subset_sha256": candidate_subset_sha256,
        "shard_plan_id": shard_plan_id,
    }


def _wrapper_counts(
    adapter: Mapping[str, Any], result: Mapping[str, Any]
) -> dict[str, int]:
    adapter_counts = cast(Mapping[str, int], adapter["counts"])
    return {
        **{key: int(value) for key, value in adapter_counts.items()},
        "sites": len(cast(Sequence[Any], result["sites"])),
        "provenances": len(cast(Sequence[Any], result["provenances"])),
        "access_facts": len(cast(Sequence[Any], result["access_facts"])),
        "escape_facts": len(cast(Sequence[Any], result["escape_facts"])),
    }


def _adapter_resource_locators(adapter: Mapping[str, Any]) -> dict[str, str]:
    locators: dict[str, str] = {}
    candidates = cast(Sequence[Mapping[str, Any]], adapter["analyzer_candidates"])
    for index, candidate in enumerate(candidates):
        resource_id = _identity(
            candidate.get("id"), "resource_id", label=f"analyzer_candidates[{index}].id"
        )
        locator = _locator(candidate)["value"]
        previous = locators.setdefault(resource_id, locator)
        if previous != locator:
            raise MachineFactError(
                "analyzer candidate aliases have conflicting locators"
            )
    expected_ids = _strings(
        _plain_json(adapter["supported_resource_ids"]),
        label="supported_resource_ids",
        sorted_unique=True,
    )
    if sorted(locators) != expected_ids:
        raise MachineFactError(
            "supported resource IDs do not match analyzer candidates"
        )
    return locators


def analyze_runtime_access_unsealed(
    source_files: Mapping[str, bytes],
    adapter: Mapping[str, Any],
    *,
    analyzer: Analyzer,
    shard_plan_id: str = SHARD_PLAN_ID,
) -> dict[str, Any]:
    """Return validated raw schema-v2 facts for caller-supplied, unsealed inputs.

    This generic API cannot construct cache documents or attach source/analyzer
    seals.  It is suitable for unit tests and synthetic snapshots only.
    """

    if shard_plan_id != SHARD_PLAN_ID:
        raise MachineFactError(f"sharding is forbidden: {SHARDING_DISABLED_REASON}")
    _validate_adapter_integrity(adapter)
    candidates = cast(Sequence[Mapping[str, Any]], adapter["analyzer_candidates"])
    result = analyzer(source_files, candidates)
    resource_locators = _adapter_resource_locators(adapter)
    validate_runtime_access_v2(result, resource_locators=resource_locators)
    return result


def _verified_supplied_toolchain_seal(
    repository: Path,
    manifest: Mapping[str, Any],
    snapshot: CommittedSnapshot,
) -> dict[str, str]:
    if snapshot.manifest_kind != MACHINE_FACT_TOOLCHAIN_MANIFEST_KIND:
        raise MachineFactError("supplied toolchain snapshot has the wrong kind")
    try:
        rebuilt = verify_manifest(
            repository,
            manifest,
            expected_kind=MACHINE_FACT_TOOLCHAIN_MANIFEST_KIND,
            expected_revision=snapshot.revision,
        )
    except ManifestError as exc:
        raise MachineFactError("supplied toolchain manifest is not verified") from exc
    if rebuilt != snapshot:
        raise MachineFactError(
            "supplied toolchain snapshot does not match its manifest"
        )
    return _seal(manifest)


def _rebuild_toolchain_seal(
    repository: Path, claimed_seal: Mapping[str, Any]
) -> dict[str, str]:
    if set(claimed_seal) != _SEAL_KEYS:
        raise MachineFactError("toolchain seal must have exact keys")
    revision = _string(claimed_seal.get("revision"), label="toolchain_seal.revision")
    if _SHA1.fullmatch(revision) is None:
        raise MachineFactError("toolchain seal revision is invalid")
    try:
        manifest = build_manifest(
            repository,
            revision,
            manifest_kind=MACHINE_FACT_TOOLCHAIN_MANIFEST_KIND,
            expected_revision=revision,
        )
        verify_manifest(
            repository,
            manifest,
            expected_kind=MACHINE_FACT_TOOLCHAIN_MANIFEST_KIND,
            expected_revision=revision,
        )
    except ManifestError as exc:
        raise MachineFactError(
            "toolchain seal does not resolve to a verified Git commit"
        ) from exc
    rebuilt_seal = _seal(manifest)
    if dict(claimed_seal) != rebuilt_seal:
        raise MachineFactError("toolchain seal does not match its committed manifest")
    return rebuilt_seal


def _assemble_sealed_document(
    result: Mapping[str, Any],
    adapter: Mapping[str, Any],
    *,
    repository: Path,
    source_seal: Mapping[str, Any],
    analyzer_seal: Mapping[str, Any],
    toolchain_manifest: Mapping[str, Any],
    toolchain_snapshot: CommittedSnapshot,
) -> dict[str, Any]:
    toolchain_seal = _verified_supplied_toolchain_seal(
        repository, toolchain_manifest, toolchain_snapshot
    )
    validate_effective_adapter(adapter)
    validate_runtime_access_v2(
        result, resource_locators=_adapter_resource_locators(adapter)
    )
    cache_key = cache_key_metadata(
        source_seal=source_seal,
        analyzer_seal=analyzer_seal,
        toolchain_seal=toolchain_seal,
        candidate_subset_sha256=CANDIDATE_SUBSET_SHA256,
    )
    document = {
        "binding_mode": "sealed_effective",
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "adapter_schema_version": ADAPTER_SCHEMA_VERSION,
        "cache_key": cache_key,
        "source_seal": dict(source_seal),
        "analyzer_seal": dict(analyzer_seal),
        "toolchain_seal": toolchain_seal,
        "candidate_subset_sha256": CANDIDATE_SUBSET_SHA256,
        "shard_plan": {
            "id": SHARD_PLAN_ID,
            "mode": "monolithic",
            "reason": SHARDING_DISABLED_REASON,
        },
        "runtime_access": result,
        "unsupported_declarations": _plain_json(adapter["unsupported_declarations"]),
        "excluded_declarations": _plain_json(adapter["excluded_declarations"]),
        "counts": _wrapper_counts(adapter, result),
    }
    validate_machine_fact_document(repository, document, adapter=adapter)
    return document


def validate_machine_fact_document(
    repository: Path,
    value: object,
    *,
    adapter: Mapping[str, Any],
) -> None:
    """Accept only the official sealed-effective cache/document binding."""

    document = _object(value, _WRAPPER_KEYS, label="machine fact document")
    validate_effective_adapter(adapter)
    if document["binding_mode"] != "sealed_effective":
        raise MachineFactError("machine fact document is not sealed-effective")
    if (
        type(document["cache_schema_version"]) is not int
        or document["cache_schema_version"] != CACHE_SCHEMA_VERSION
    ):
        raise MachineFactError("cache schema version drifted")
    if (
        type(document["adapter_schema_version"]) is not int
        or document["adapter_schema_version"] != ADAPTER_SCHEMA_VERSION
    ):
        raise MachineFactError("adapter schema version drifted")
    key = _object(document["cache_key"], _CACHE_KEY_KEYS, label="cache_key")
    if (
        type(key["adapter_schema_version"]) is not int
        or key["adapter_schema_version"] != ADAPTER_SCHEMA_VERSION
    ):
        raise MachineFactError("cache key adapter schema version drifted")
    source_seal = _object(document["source_seal"], _SEAL_KEYS, label="source_seal")
    analyzer_seal = _object(
        document["analyzer_seal"], _SEAL_KEYS, label="analyzer_seal"
    )
    toolchain_seal = _object(
        document["toolchain_seal"], _SEAL_KEYS, label="toolchain_seal"
    )
    verified_toolchain_seal = _rebuild_toolchain_seal(repository, toolchain_seal)
    rebuilt_key = cache_key_metadata(
        source_seal=source_seal,
        analyzer_seal=analyzer_seal,
        toolchain_seal=verified_toolchain_seal,
        candidate_subset_sha256=_string(
            document["candidate_subset_sha256"], label="candidate_subset_sha256"
        ),
        shard_plan_id=SHARD_PLAN_ID,
    )
    official_source_seal = {
        "revision": EFFECTIVE_SOURCE_REVISION,
        "files_sha256": SOURCE_FILES_SHA256,
        "manifest_sha256": SOURCE_MANIFEST_SHA256,
    }
    official_analyzer_seal = {
        "revision": ANALYZER_REVISION,
        "files_sha256": ANALYZER_FILES_SHA256,
        "manifest_sha256": ANALYZER_MANIFEST_SHA256,
    }
    official_key = cache_key_metadata(
        source_seal=official_source_seal,
        analyzer_seal=official_analyzer_seal,
        toolchain_seal=verified_toolchain_seal,
        candidate_subset_sha256=CANDIDATE_SUBSET_SHA256,
    )
    if (
        source_seal != official_source_seal
        or analyzer_seal != official_analyzer_seal
        or toolchain_seal != verified_toolchain_seal
        or key != official_key
        or rebuilt_key != official_key
    ):
        raise MachineFactError("official sealed cache key metadata drifted")
    if document["candidate_subset_sha256"] != CANDIDATE_SUBSET_SHA256:
        raise MachineFactError("document candidate subset drifted")
    shard_plan = _object(
        document["shard_plan"],
        frozenset({"id", "mode", "reason"}),
        label="shard_plan",
    )
    if shard_plan != {
        "id": SHARD_PLAN_ID,
        "mode": "monolithic",
        "reason": SHARDING_DISABLED_REASON,
    }:
        raise MachineFactError(f"shard plan is invalid: {SHARDING_DISABLED_REASON}")
    if document["unsupported_declarations"] != _plain_json(
        adapter["unsupported_declarations"]
    ):
        raise MachineFactError("unsupported declarations drifted")
    if document["excluded_declarations"] != _plain_json(
        adapter["excluded_declarations"]
    ):
        raise MachineFactError("excluded declarations drifted")
    runtime_access = cast(Mapping[str, Any], document["runtime_access"])
    validate_runtime_access_v2(
        runtime_access,
        resource_locators=_adapter_resource_locators(adapter),
    )
    expected_counts = _wrapper_counts(adapter, runtime_access)
    counts = _object(document["counts"], _WRAPPER_COUNT_KEYS, label="document.counts")
    for count_name, count_value in counts.items():
        _integer(count_value, label=f"document.counts.{count_name}")
    if counts != expected_counts:
        raise MachineFactError("machine fact document counts drifted")
    _assert_no_owner_fields(document)


def machine_fact_cache_bytes(
    repository: Path,
    document: object,
    *,
    adapter: Mapping[str, Any],
) -> bytes:
    validate_machine_fact_document(repository, document, adapter=adapter)
    return canonical_bytes(document)


def load_machine_fact_cache(
    repository: Path,
    raw: bytes,
    *,
    adapter: Mapping[str, Any],
) -> dict[str, Any]:
    """Parse only unique-key, finite, exact-canonical cache JSON."""

    if type(raw) is not bytes:
        raise MachineFactError("cache input must be exact bytes")

    def reject_constant(token: str) -> Any:
        raise MachineFactError(f"cache JSON contains a non-finite number: {token}")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise MachineFactError(f"cache JSON contains duplicate key: {key}")
            result[key] = value
        return result

    try:
        decoded = raw.decode("utf-8")
        value = json.loads(
            decoded,
            parse_constant=reject_constant,
            object_pairs_hook=unique_object,
        )
    except MachineFactError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise MachineFactError("cache JSON is malformed") from exc
    if canonical_bytes(value) != raw:
        raise MachineFactError("cache JSON is not in canonical byte form")
    validate_machine_fact_document(repository, value, adapter=adapter)
    return cast(dict[str, Any], value)


def reject_sharded_analysis(*_args: Any, **_kwargs: Any) -> None:
    """Fail the explicit negative API used by callers probing for sharding."""

    raise MachineFactError(f"sharding is forbidden: {SHARDING_DISABLED_REASON}")


def _seal(manifest: Mapping[str, Any]) -> dict[str, str]:
    return {
        "revision": _string(manifest.get("revision"), label="manifest.revision"),
        "files_sha256": _string(
            manifest.get("files_sha256"), label="manifest.files_sha256"
        ),
        "manifest_sha256": _string(
            manifest.get("manifest_sha256"), label="manifest.manifest_sha256"
        ),
    }


def _snapshot_mapping(snapshot: CommittedSnapshot) -> dict[str, bytes]:
    return {row.path: row.raw_bytes for row in snapshot.files}


def verify_loaded_analyzer_code(
    loaded_files: Mapping[str, bytes], committed: CommittedSnapshot
) -> None:
    """Make the imported/current-code versus committed-analyzer boundary explicit."""

    expected = _snapshot_mapping(committed)
    if set(loaded_files) != set(ANALYZER_PATHS):
        raise MachineFactError(
            "loaded analyzer path set drifted from the exact 15 files"
        )
    if any(
        type(path) is not str or type(raw) is not bytes
        for path, raw in loaded_files.items()
    ):
        raise MachineFactError(
            "loaded analyzer files must be exact path-to-bytes entries"
        )
    if dict(loaded_files) != expected:
        raise MachineFactError(
            "loaded/current analyzer code drifted from committed input"
        )


def current_loaded_analyzer_files(repository: Path) -> dict[str, bytes]:
    """Read the exact worktree files from which the analyzer package is loaded."""

    return {path: (repository / path).read_bytes() for path in ANALYZER_PATHS}


def verify_current_toolchain_code(
    repository: Path, committed: CommittedSnapshot
) -> None:
    """Verify exact current bytes and loaded module paths for the toolchain."""

    if committed.manifest_kind != MACHINE_FACT_TOOLCHAIN_MANIFEST_KIND:
        raise MachineFactError("committed toolchain snapshot has the wrong kind")
    expected = _snapshot_mapping(committed)
    if set(expected) != set(MACHINE_FACT_TOOLCHAIN_PATHS):
        raise MachineFactError("committed toolchain snapshot is not the exact 3 paths")
    try:
        current = {
            path: (repository / path).read_bytes()
            for path in MACHINE_FACT_TOOLCHAIN_PATHS
        }
    except OSError as exc:
        raise MachineFactError("current toolchain files cannot be read") from exc
    if current != expected:
        raise MachineFactError("current toolchain bytes drifted from committed HEAD")
    for path, module_name in _TOOLCHAIN_MODULE_BY_PATH.items():
        module = sys.modules.get(module_name)
        if module is None:
            raise MachineFactError(f"toolchain module is not loaded: {module_name}")
        loaded_path = getattr(module, "__file__", None)
        if (
            type(loaded_path) is not str
            or Path(loaded_path).resolve() != (repository / path).resolve()
        ):
            raise MachineFactError(
                f"toolchain module loaded from an unexpected path: {module_name}"
            )


def _require_fresh_analyzer_import_state() -> None:
    loaded = sorted(_ANALYZER_MODULE_NAMES.intersection(sys.modules))
    if loaded:
        raise MachineFactError(
            "sealed analysis requires a fresh analyzer import state; "
            f"already loaded={loaded}"
        )


def _import_verified_fresh_analyzer(
    repository: Path, committed: CommittedSnapshot
) -> Analyzer:
    """Import only after committed and current analyzer bytes were verified."""

    importlib.invalidate_caches()
    for module_name in sorted(_ANALYZER_MODULE_NAMES):
        importlib.import_module(module_name)
    module = sys.modules["scripts.runtime_ownership.access"]
    loaded = _ANALYZER_MODULE_NAMES.intersection(sys.modules)
    if loaded != _ANALYZER_MODULE_NAMES:
        raise MachineFactError(
            "fresh analyzer import did not load the exact 15-module package"
        )
    expected_paths = {
        name: (
            repository / f"scripts/runtime_ownership/{name.rsplit('.', 1)[1]}.py"
        ).resolve()
        for name in _ANALYZER_MODULE_NAMES
    }
    for name, expected_path in expected_paths.items():
        loaded_module = sys.modules[name]
        module_path = getattr(loaded_module, "__file__", None)
        if type(module_path) is not str or Path(module_path).resolve() != expected_path:
            raise MachineFactError(
                f"fresh analyzer module loaded from an unexpected path: {name}"
            )
    verify_loaded_analyzer_code(current_loaded_analyzer_files(repository), committed)
    analyzer = getattr(module, "discover_access_facts", None)
    if (
        not callable(analyzer)
        or getattr(analyzer, "__module__", None) != module.__name__
    ):
        raise MachineFactError("fresh analyzer entrypoint binding is invalid")
    return cast(Analyzer, analyzer)


def run_sealed_effective_analysis(repository: Path) -> dict[str, Any]:
    """Bind exact source, analyzer, and current-HEAD toolchain Git inputs."""

    _require_fresh_analyzer_import_state()
    toolchain_revision = current_head_revision(repository)
    toolchain_manifest = build_manifest(
        repository,
        toolchain_revision,
        manifest_kind=MACHINE_FACT_TOOLCHAIN_MANIFEST_KIND,
        expected_revision=toolchain_revision,
    )
    toolchain_snapshot = verify_manifest(
        repository,
        toolchain_manifest,
        expected_kind=MACHINE_FACT_TOOLCHAIN_MANIFEST_KIND,
        expected_revision=toolchain_revision,
    )
    verify_current_toolchain_code(repository, toolchain_snapshot)
    source_manifest = build_manifest(
        repository,
        EFFECTIVE_SOURCE_REVISION,
        manifest_kind=SOURCE_MANIFEST_KIND,
        expected_revision=EFFECTIVE_SOURCE_REVISION,
    )
    analyzer_manifest = build_manifest(
        repository,
        ANALYZER_REVISION,
        manifest_kind=ANALYZER_MANIFEST_KIND,
        expected_revision=ANALYZER_REVISION,
    )
    if (
        source_manifest.get("files_sha256") != SOURCE_FILES_SHA256
        or source_manifest.get("manifest_sha256") != SOURCE_MANIFEST_SHA256
    ):
        raise MachineFactError("effective source manifest seal drifted")
    if (
        analyzer_manifest.get("files_sha256") != ANALYZER_FILES_SHA256
        or analyzer_manifest.get("manifest_sha256") != ANALYZER_MANIFEST_SHA256
    ):
        raise MachineFactError("effective analyzer manifest seal drifted")
    source_snapshot = verify_manifest(
        repository,
        source_manifest,
        expected_kind=SOURCE_MANIFEST_KIND,
        expected_revision=EFFECTIVE_SOURCE_REVISION,
    )
    analyzer_snapshot = verify_manifest(
        repository,
        analyzer_manifest,
        expected_kind=ANALYZER_MANIFEST_KIND,
        expected_revision=ANALYZER_REVISION,
    )
    verify_loaded_analyzer_code(
        current_loaded_analyzer_files(repository), analyzer_snapshot
    )
    analyzer = _import_verified_fresh_analyzer(repository, analyzer_snapshot)
    declarations = discover_concrete(source_snapshot)
    adapter = build_declaration_adapter(declarations)
    validate_effective_adapter(adapter)
    result = analyze_runtime_access_unsealed(
        _snapshot_mapping(source_snapshot),
        adapter,
        analyzer=analyzer,
    )
    return _assemble_sealed_document(
        result,
        adapter,
        repository=repository,
        source_seal=_seal(source_manifest),
        analyzer_seal=_seal(analyzer_manifest),
        toolchain_manifest=toolchain_manifest,
        toolchain_snapshot=toolchain_snapshot,
    )


__all__ = [
    "ADAPTER_SCHEMA_VERSION",
    "ANALYZER_FILES_SHA256",
    "ANALYZER_MANIFEST_SHA256",
    "ANALYZER_REVISION",
    "CACHE_SCHEMA_VERSION",
    "CANDIDATE_SUBSET_SHA256",
    "EFFECTIVE_SOURCE_REVISION",
    "MachineFactError",
    "PROVENANCE_PATH_LIMIT",
    "PROVENANCE_RETENTION_POLICY",
    "SHARDING_DISABLED_REASON",
    "SHARD_PLAN_ID",
    "SOURCE_FILES_SHA256",
    "SOURCE_MANIFEST_SHA256",
    "analyze_runtime_access_unsealed",
    "build_declaration_adapter",
    "cache_key_metadata",
    "canonical_bytes",
    "current_loaded_analyzer_files",
    "load_machine_fact_cache",
    "machine_fact_cache_bytes",
    "reject_sharded_analysis",
    "run_sealed_effective_analysis",
    "validate_effective_adapter",
    "validate_machine_fact_document",
    "validate_runtime_access_v2",
    "verify_current_toolchain_code",
    "verify_loaded_analyzer_code",
]
