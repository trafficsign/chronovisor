# ruff: noqa: F401, F403, F405
"""Runtime ownership gate layer."""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import io
import json
import plistlib
import re
import subprocess
import tarfile
import tomllib
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

from .discovery import *
from .model import *
from .registry import *
from .seed import *
from .source import *


def _duplicate_values(values: list[str]) -> list[str]:
    counts = Counter(values)
    return sorted(value for value, count in counts.items() if count > 1)


def _unknown_value_paths(value: Any, path: str = "$") -> list[str]:
    violations: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            violations.extend(_unknown_value_paths(item, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            violations.extend(_unknown_value_paths(item, f"{path}[{index}]"))
    elif isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in UNKNOWN_VALUES or "*" in value:
            violations.append(path)
    return violations


def _valid_evidence(snapshot: dict[str, bytes], evidence: Any) -> tuple[bool, str]:
    if not isinstance(evidence, list) or not evidence:
        return False, "evidence must be a non-empty list"
    for item in evidence:
        if not isinstance(item, dict):
            return False, "evidence entry must be an object"
        path = item.get("path")
        line = item.get("line")
        if not isinstance(path, str) or path not in snapshot:
            return False, f"evidence path does not exist: {path}"
        if type(line) is not int or line < 1:
            return False, f"invalid evidence line: {line}"
        if line > len(_text(snapshot, path).splitlines()):
            return False, f"evidence line exceeds file: {path}:{line}"
    return True, ""


def _valid_worker_module(
    index: _SourceIndex, snapshot: dict[str, bytes], module: Any
) -> bool:
    if not isinstance(module, str) or not module:
        return False
    if module.startswith("script:"):
        return module.removeprefix("script:") in snapshot
    return module in index.paths


def _registry_structure_errors(registry: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if tuple(registry) != REGISTRY_ROOT_KEYS:
        errors.append("root_keys")
    if type(registry.get("schema_version")) is not int:
        errors.append("schema_version")
    policy = registry.get("policy")
    if not isinstance(policy, dict) or tuple(policy) != (
        "discovery",
        "exclusions",
        "new_resources",
    ):
        errors.append("policy")
    elif not _same_json_value(policy, REGISTRY_POLICY):
        errors.append("policy.values")
    resources = registry.get("resources")
    if not isinstance(resources, list):
        errors.append("resources")
    else:
        suffixes = {
            "artifact": ("format",),
            "queue": ("format",),
            "lock": ("scope", "protects"),
            "schema": (),
            "socket": ("socket",),
            "worker": ("worker",),
        }
        for index, row in enumerate(resources):
            if not isinstance(row, dict):
                errors.append(f"resources[{index}]")
                continue
            kind = row.get("kind")
            suffix = suffixes.get(kind) if isinstance(kind, str) else None
            if suffix is None or tuple(row) != RESOURCE_BASE_KEYS + suffix:
                errors.append(f"resources[{index}].keys")
            locator = row.get("locator")
            if not isinstance(locator, dict) or tuple(locator) != ("type", "value"):
                errors.append(f"resources[{index}].locator")
            lifecycle = row.get("lifecycle")
            if not isinstance(lifecycle, dict) or tuple(lifecycle) != (
                "retention",
                "recovery_owner",
                "recovery_contract",
            ):
                errors.append(f"resources[{index}].lifecycle")
            evidence = row.get("evidence")
            if not isinstance(evidence, list) or any(
                not isinstance(item, dict) or tuple(item) != ("path", "line")
                for item in evidence
            ):
                errors.append(f"resources[{index}].evidence")
    exclusions = registry.get("exclusions")
    exclusion_keys = ("discovery_id", "path", "line", "module", "symbol", "reason")
    if not isinstance(exclusions, list):
        errors.append("exclusions")
    else:
        for index, row in enumerate(exclusions):
            if not isinstance(row, dict) or tuple(row) != exclusion_keys:
                errors.append(f"exclusions[{index}].keys")
    sites = registry.get("lock_protocol_sites")
    protocol_keys = (
        "discovery_id",
        "path",
        "line",
        "module",
        "scope",
        "operation",
        "protocol",
        "coverage",
        "lock_ids",
        "protects",
    )
    if not isinstance(sites, list):
        errors.append("lock_protocol_sites")
    else:
        for index, row in enumerate(sites):
            expected = protocol_keys + (
                ("reason",) if isinstance(row, dict) and "reason" in row else ()
            )
            if not isinstance(row, dict) or tuple(row) != expected:
                errors.append(f"lock_protocol_sites[{index}].keys")
    return errors


def _resource_contract_drift(
    generated: list[dict[str, Any]], registered: list[dict[str, Any]]
) -> dict[str, list[str]]:
    generated_by_id = {str(row.get("id") or ""): row for row in generated}
    registered_by_id = {str(row.get("id") or ""): row for row in registered}
    fields = ("kind", "locator") + MANUAL_RESOURCE_FIELDS + (
        "discovery_ids",
        "format",
        "scope",
        "protects",
        "socket",
        "worker",
    )
    drift: dict[str, list[str]] = {}
    for resource_id in sorted(generated_by_id.keys() & registered_by_id.keys()):
        expected = generated_by_id[resource_id]
        recorded = registered_by_id[resource_id]
        changed = [
            field
            for field in fields
            if not _same_json_value(expected.get(field), recorded.get(field))
        ]
        if changed:
            drift[resource_id] = changed
    return drift


def _resource_validation_errors(
    index: _SourceIndex,
    snapshot: dict[str, bytes],
    resources: list[dict[str, Any]],
) -> list[dict[str, str]]:
    errors: list[dict[str, str]] = []
    by_id = {str(row.get("id") or ""): row for row in resources}
    lock_ids = {
        str(row.get("id") or "") for row in resources if row.get("kind") == "lock"
    }
    for row in resources:
        resource_id = str(row.get("id") or "")

        def reject(message: str, current_id: str = resource_id) -> None:
            errors.append({"id": current_id, "error": message})

        kind = row.get("kind")
        locator = row.get("locator")
        if kind not in RESOURCE_KINDS:
            reject(f"invalid kind: {kind}")
            continue
        if (
            not isinstance(locator, dict)
            or not isinstance(locator.get("type"), str)
            or not isinstance(locator.get("value"), str)
        ):
            reject("locator must contain string type and value")
            continue
        locator_types_by_kind = {
            "artifact": {"path"},
            "queue": {"path"},
            "lock": {"path"},
            "schema": {"schema_id"},
            "socket": {"socket"},
            "worker": {"entrypoint", "launchd", "lab_dispatch", "module_worker"},
        }
        if locator.get("type") not in locator_types_by_kind[str(kind)]:
            reject(f"locator type is invalid for {kind}: {locator.get('type')}")
        expected_id = _resource_id(str(kind), str(locator["value"]))
        if resource_id != expected_id:
            reject("resource ID does not match kind and locator")
        owner_symbol = row.get("owner_symbol")
        owner_package = row.get("owner_package")
        if not isinstance(owner_symbol, str) or not index.symbol_exists(owner_symbol):
            reject(f"owner symbol does not exist: {owner_symbol}")
        elif owner_package != _owner_package(owner_symbol):
            reject("owner package does not match owner symbol")
        for field in ("writers", "readers", "compatibility", "discovery_ids"):
            values = row.get(field)
            if (
                not isinstance(values, list)
                or not values
                or any(not isinstance(value, str) or not value for value in values)
            ):
                reject(f"{field} must be a non-empty string list")
            elif len(values) != len(set(values)):
                reject(f"{field} must not contain duplicates")
        for field in ("writers", "readers"):
            values = row.get(field)
            if isinstance(values, list):
                for reference in values:
                    if isinstance(reference, str) and not index.symbol_exists(
                        reference
                    ):
                        reject(f"{field} symbol does not exist: {reference}")
        coordination = row.get("coordination")
        if (
            not isinstance(coordination, dict)
            or not isinstance(coordination.get("protocol"), str)
            or not coordination.get("protocol")
        ):
            reject("coordination.protocol is required")
        writers = row.get("writers")
        if isinstance(writers, list) and len(writers) > 1:
            lock_id = (
                coordination.get("lock_id") if isinstance(coordination, dict) else None
            )
            transactional = (
                isinstance(coordination, dict)
                and coordination.get("protocol") == "sqlite-wal-transactions"
                and coordination.get("transaction") == "BEGIN IMMEDIATE"
            )
            if not transactional and (
                not isinstance(lock_id, str) or lock_id not in lock_ids
            ):
                reject(
                    "multiple writers require an explicit valid lock_id or "
                    "reviewed SQLite transaction protocol"
                )
        lifecycle = row.get("lifecycle")
        if not isinstance(lifecycle, dict):
            reject("lifecycle is required")
        else:
            for field in ("retention", "recovery_owner", "recovery_contract"):
                if not isinstance(lifecycle.get(field), str) or not lifecycle.get(
                    field
                ):
                    reject(f"lifecycle.{field} is required")
            recovery_owner = lifecycle.get("recovery_owner")
            if isinstance(recovery_owner, str) and not index.symbol_exists(
                recovery_owner
            ):
                reject(f"recovery owner does not exist: {recovery_owner}")
        evidence_ok, evidence_error = _valid_evidence(snapshot, row.get("evidence"))
        if not evidence_ok:
            reject(evidence_error)
        if kind in {"artifact", "queue"}:
            resource_format = row.get("format")
            if not isinstance(resource_format, dict):
                reject("artifact/queue format contract is required")
            elif resource_format.get("status") == "versioned":
                if (
                    not isinstance(resource_format.get("schema_id"), str)
                    or not resource_format.get("schema_id")
                    or type(resource_format.get("version")) is not int
                    or resource_format.get("version", 0) < 1
                ):
                    reject("versioned format requires schema_id and positive version")
                source = resource_format.get("source")
                if not isinstance(source, str) or not index.symbol_exists(source):
                    reject(f"format source symbol does not exist: {source}")
            elif resource_format.get("status") == "unversioned":
                if "version" in resource_format or "schema_id" in resource_format:
                    reject("unversioned format cannot invent schema_id or version")
                rationale = resource_format.get("rationale")
                migration_owner = resource_format.get("migration_owner")
                if not isinstance(rationale, str) or not rationale:
                    reject("unversioned format requires rationale")
                if not isinstance(migration_owner, str) or not index.symbol_exists(
                    migration_owner
                ):
                    reject(
                        f"unversioned migration owner does not exist: {migration_owner}"
                    )
            else:
                reject("format.status must be versioned or unversioned")
        elif kind == "lock":
            scope = row.get("scope")
            if scope not in {"artifact_sidecar", "worker_lease", "global_protocol"}:
                reject("lock scope is required and must be reviewed")
            protects = row.get("protects")
            if not isinstance(protects, list) or not protects:
                reject("lock protects must be non-empty")
            elif any(
                not isinstance(target, str)
                or target not in by_id
                or target == resource_id
                for target in protects
            ):
                reject("lock protects references an unregistered or self resource")
            elif scope == "artifact_sidecar" and any(
                by_id[target].get("kind") not in {"artifact", "queue"}
                for target in protects
            ):
                reject("artifact sidecar locks may protect only artifact/queue state")
        elif kind == "socket":
            socket = row.get("socket")
            if not isinstance(socket, dict):
                reject("socket contract is required")
            else:
                if socket.get("address") != locator.get("value"):
                    reject("socket address must equal locator value")
                if socket.get("server") != owner_symbol:
                    reject("socket server must equal owner symbol")
                clients = socket.get("clients")
                if not isinstance(clients, list) or not clients:
                    reject("socket clients must be non-empty")
                elif any(
                    not isinstance(client, str) or not index.symbol_exists(client)
                    for client in clients
                ):
                    reject("socket client symbol does not exist")
                elif owner_symbol in clients:
                    reject("socket server cannot be registered as its own client")
        elif kind == "schema":
            if row.get("writers") != [owner_symbol]:
                reject("schema must have exactly one owner writer")
        elif kind == "worker":
            worker = row.get("worker")
            if not isinstance(worker, dict):
                reject("worker contract is required")
            else:
                if not _valid_worker_module(index, snapshot, worker.get("module")):
                    reject("worker module does not exist")
                if not isinstance(worker.get("entrypoint"), str) or not worker.get(
                    "entrypoint"
                ):
                    reject("worker entrypoint is required")
                if locator.get("type") == "launchd" and worker.get(
                    "launchd_label"
                ) != locator.get("value"):
                    reject("launchd worker label must equal locator value")
                if locator.get("type") == "launchd":
                    invocations = worker.get("invocations")
                    if not isinstance(invocations, list) or not invocations:
                        reject("launchd worker requires concrete wrapper invocations")
                    else:
                        for invocation in invocations:
                            if not isinstance(invocation, dict):
                                reject("worker invocation must be an object")
                                continue
                            if not isinstance(invocation.get("entrypoint"), str):
                                reject("worker invocation entrypoint is required")
                            argv = invocation.get("argv")
                            if not isinstance(argv, list) or any(
                                not isinstance(value, str) for value in argv
                            ):
                                reject("worker invocation argv must be a string list")
                            if not isinstance(invocation.get("role"), str) or not invocation.get(
                                "role"
                            ):
                                reject("worker invocation role is required")
                            runtime = invocation.get("runtime")
                            if runtime is not None:
                                if not isinstance(runtime, dict) or tuple(runtime) != (
                                    "executable",
                                    "resolution",
                                    "search_path",
                                    "source",
                                    "evidence",
                                ):
                                    reject("worker invocation runtime contract is invalid")
                                else:
                                    if runtime.get("executable") != "uvx":
                                        reject("worker runtime executable must be bare uvx")
                                    if runtime.get("resolution") != "PATH":
                                        reject("worker runtime resolution must use PATH")
                                    for field in ("search_path", "source"):
                                        if not isinstance(runtime.get(field), str) or not runtime.get(
                                            field
                                        ):
                                            reject(f"worker runtime {field} is required")
                                    valid_evidence, evidence_error = _valid_evidence(
                                        snapshot, runtime.get("evidence")
                                    )
                                    if not valid_evidence:
                                        reject(
                                            "worker runtime evidence is invalid: "
                                            f"{evidence_error}"
                                        )
                            invocation_evidence = invocation.get("evidence")
                            if invocation_evidence is not None:
                                valid_evidence, evidence_error = _valid_evidence(
                                    snapshot, [invocation_evidence]
                                )
                                if not valid_evidence:
                                    reject(
                                        "worker invocation evidence is invalid: "
                                        f"{evidence_error}"
                                    )
                            link = invocation.get("linked_worker")
                            if isinstance(link, dict):
                                linked = any(
                                    candidate.get("kind") == link.get("kind")
                                    and candidate.get("locator", {}).get("type")
                                    == link.get("locator_type")
                                    and candidate.get("locator", {}).get("value")
                                    == link.get("locator_value")
                                    for candidate in resources
                                    if isinstance(candidate.get("locator"), dict)
                                )
                                if not linked:
                                    reject("worker invocation link is not registered")
        for unknown_path in _unknown_value_paths(row):
            reject(f"unknown or wildcard value at {unknown_path}")
    return errors


def _exclusion_validation_errors(
    snapshot: dict[str, bytes], exclusions: list[dict[str, Any]]
) -> list[dict[str, str]]:
    errors: list[dict[str, str]] = []
    for row in exclusions:
        discovery_id = str(row.get("discovery_id") or "")
        reason = row.get("reason")
        if not isinstance(reason, str) or not reason:
            errors.append({"id": discovery_id, "error": "exclusion reason is required"})
        evidence_ok, evidence_error = _valid_evidence(
            snapshot,
            [{"path": row.get("path"), "line": row.get("line")}],
        )
        if not evidence_ok:
            errors.append({"id": discovery_id, "error": evidence_error})
        for unknown_path in _unknown_value_paths(row):
            errors.append(
                {
                    "id": discovery_id,
                    "error": f"unknown or wildcard value at {unknown_path}",
                }
            )
    return errors


def _lock_protocol_validation_errors(
    sites: list[dict[str, Any]], resources: list[dict[str, Any]]
) -> list[dict[str, str]]:
    errors: list[dict[str, str]] = []
    by_id = {str(row.get("id") or ""): row for row in resources}
    for row in sites:
        discovery_id = str(row.get("discovery_id") or "")

        def reject(message: str, current_id: str = discovery_id) -> None:
            errors.append({"id": current_id, "error": message})

        for field in ("path", "module", "scope", "operation", "protocol", "coverage"):
            if not isinstance(row.get(field), str) or not row.get(field):
                reject(f"{field} is required")
        if type(row.get("line")) is not int or row.get("line", 0) < 1:
            reject("positive evidence line is required")
        lock_ids = row.get("lock_ids")
        protects = row.get("protects")
        if not isinstance(lock_ids, list) or any(
            not isinstance(lock_id, str)
            or lock_id not in by_id
            or by_id[lock_id].get("kind") != "lock"
            for lock_id in lock_ids
        ):
            reject("lock_ids must reference registered locks")
        if not isinstance(protects, list) or any(
            not isinstance(resource_id, str)
            or resource_id not in by_id
            or by_id[resource_id].get("kind") == "lock"
            for resource_id in protects
        ):
            reject("protects must reference registered non-lock resources")
        if row.get("coverage") == "primitive_only":
            if protects:
                reject("primitive-only lock sites cannot claim protected state")
            if not isinstance(row.get("reason"), str) or not row.get("reason"):
                reject("primitive-only lock sites require a reason")
        elif row.get("coverage") == "concrete_call_site":
            if not protects:
                reject("concrete lock sites require protected state")
        else:
            reject("unsupported lock protocol coverage")
    return errors


def runtime_state_fitness(root: Path) -> dict[str, Any]:
    root = root.resolve()
    snapshot = _snapshot_current(root)
    index, detection = discover(snapshot)
    registry = _load_json(root / REGISTRY_PATH)
    seed = _load_json(root / BASELINE_PATH)
    previous = _load_previous_baseline(root)
    frozen = build_runtime_state_baseline(root)
    generated_resources = _base_resources(detection)
    generated_exclusions = _exclusion_rows(detection)
    registry_resources = (
        [row for row in registry.get("resources", []) if isinstance(row, dict)]
        if isinstance(registry.get("resources"), list)
        else []
    )
    registry_exclusions = (
        [row for row in registry.get("exclusions", []) if isinstance(row, dict)]
        if isinstance(registry.get("exclusions"), list)
        else []
    )
    registry_lock_protocols = (
        [
            row
            for row in registry.get("lock_protocol_sites", [])
            if isinstance(row, dict)
        ]
        if isinstance(registry.get("lock_protocol_sites"), list)
        else []
    )
    detected_resource_ids = {str(row["id"]) for row in generated_resources}
    registered_resource_ids = {str(row.get("id") or "") for row in registry_resources}
    detected_discovery_ids = {str(row["discovery_id"]) for row in detection["rows"]}
    registered_discovery_values = (
        [
            str(value)
            for row in registry_resources
            for value in (
                row.get("discovery_ids", [])
                if isinstance(row.get("discovery_ids"), list)
                else []
            )
            if isinstance(value, str)
        ]
        + [str(row.get("discovery_id") or "") for row in registry_exclusions]
        + [str(row.get("discovery_id") or "") for row in registry_lock_protocols]
    )
    registered_discovery_ids = set(registered_discovery_values)
    resource_ids = [str(row.get("id") or "") for row in registry_resources]
    locator_keys = [
        f"{row.get('kind')}:{row.get('locator', {}).get('value')}"
        for row in registry_resources
        if isinstance(row.get("locator"), dict)
    ]
    exclusion_ids = [str(row.get("discovery_id") or "") for row in registry_exclusions]
    protocol_ids = [
        str(row.get("discovery_id") or "") for row in registry_lock_protocols
    ]
    expected_evidence = {str(row["id"]): row["evidence"] for row in generated_resources}
    registered_evidence = {
        str(row.get("id") or ""): row.get("evidence") for row in registry_resources
    }
    evidence_drift = [
        resource_id
        for resource_id, evidence in sorted(expected_evidence.items())
        if not _same_json_value(registered_evidence.get(resource_id), evidence)
    ]
    expected_entrypoints = {
        str(row["locator"]["value"])
        for row in generated_resources
        if row["kind"] == "worker" and row["locator"]["type"] == "entrypoint"
    }
    expected_launchd = {
        str(row["locator"]["value"])
        for row in generated_resources
        if row["kind"] == "worker" and row["locator"]["type"] == "launchd"
    }
    registered_entrypoints = {
        str(row.get("locator", {}).get("value") or "")
        for row in registry_resources
        if row.get("kind") == "worker"
        and isinstance(row.get("locator"), dict)
        and row["locator"].get("type") == "entrypoint"
    }
    registered_launchd = {
        str(row.get("locator", {}).get("value") or "")
        for row in registry_resources
        if row.get("kind") == "worker"
        and isinstance(row.get("locator"), dict)
        and row["locator"].get("type") == "launchd"
    }
    expected_counts = _resource_counts(
        generated_resources,
        generated_exclusions,
        discovery_count=len(detection["rows"]),
        lock_protocols=detection["lock_protocol_candidates"],
    )
    expected_lock_protocols = _lock_protocol_rows(detection, generated_resources)
    resource_contract_drift = _resource_contract_drift(
        generated_resources, registry_resources
    )
    worker_contract_drift = {
        resource_id: fields
        for resource_id, fields in resource_contract_drift.items()
        if "worker" in fields
    }
    generated_resource_order = [str(row["id"]) for row in generated_resources]
    generated_exclusion_order = [
        str(row["discovery_id"]) for row in generated_exclusions
    ]
    violations = {
        "registry_load_error": str(registry.get("load_error") or ""),
        "registry_structure_errors": _registry_structure_errors(registry),
        "registry_schema_version": (
            []
            if type(registry.get("schema_version")) is int
            and registry.get("schema_version") == REGISTRY_SCHEMA_VERSION
            else [registry.get("schema_version")]
        ),
        "registry_source_head_drift": (
            []
            if registry.get("source_baseline_head") == FROZEN_SOURCE_HEAD
            else [registry.get("source_baseline_head")]
        ),
        "registry_baseline_sha256_drift": (
            []
            if registry.get("baseline_sha256") == _canonical_sha256(seed)
            else [registry.get("baseline_sha256")]
        ),
        "unregistered_resource_ids": sorted(
            detected_resource_ids - registered_resource_ids
        ),
        "stale_resource_ids": sorted(registered_resource_ids - detected_resource_ids),
        "unregistered_discovery_ids": sorted(
            detected_discovery_ids - registered_discovery_ids
        ),
        "stale_discovery_ids": sorted(
            registered_discovery_ids - detected_discovery_ids
        ),
        "duplicate_resource_ids": _duplicate_values(resource_ids),
        "duplicate_locator_keys": _duplicate_values(locator_keys),
        "duplicate_discovery_ids": _duplicate_values(registered_discovery_values),
        "duplicate_exclusion_ids": _duplicate_values(exclusion_ids),
        "duplicate_lock_protocol_ids": _duplicate_values(protocol_ids),
        "resource_order_drift": (
            []
            if resource_ids == generated_resource_order
            else ["recorded resource order differs from discovery"]
        ),
        "exclusion_order_drift": (
            []
            if exclusion_ids == generated_exclusion_order
            else ["recorded exclusion order differs from discovery"]
        ),
        "exclusion_contract_drift": (
            []
            if _same_json_value(registry_exclusions, generated_exclusions)
            else ["recorded exclusions differ from discovery"]
        ),
        "resource_contract_drift": resource_contract_drift,
        "worker_contract_drift": worker_contract_drift,
        "resource_validation_errors": _resource_validation_errors(
            index, snapshot, registry_resources
        ),
        "exclusion_validation_errors": _exclusion_validation_errors(
            snapshot, registry_exclusions
        ),
        "lock_protocol_validation_errors": _lock_protocol_validation_errors(
            registry_lock_protocols, registry_resources
        ),
        "lock_protocol_drift": (
            []
            if _same_json_value(registry_lock_protocols, expected_lock_protocols)
            else ["recorded lock protocol inventory differs from discovery"]
        ),
        "resource_evidence_drift": evidence_drift,
        "registry_count_drift": (
            {}
            if _same_json_value(registry.get("counts"), expected_counts)
            else {"recorded": registry.get("counts"), "expected": expected_counts}
        ),
        "entrypoint_worker_drift": {
            "missing": sorted(expected_entrypoints - registered_entrypoints),
            "stale": sorted(registered_entrypoints - expected_entrypoints),
        }
        if expected_entrypoints != registered_entrypoints
        else {},
        "launchd_worker_drift": {
            "missing": sorted(expected_launchd - registered_launchd),
            "stale": sorted(registered_launchd - expected_launchd),
        }
        if expected_launchd != registered_launchd
        else {},
        **_seed_state_violations(detection, seed, frozen, previous),
    }
    return {
        "schema_version": 1,
        "passed": not any(violations.values()),
        "counts": expected_counts,
        "violations": violations,
    }


__all__ = [
    "_duplicate_values",
    "_unknown_value_paths",
    "_valid_evidence",
    "_valid_worker_module",
    "_registry_structure_errors",
    "_resource_contract_drift",
    "_resource_validation_errors",
    "_exclusion_validation_errors",
    "_lock_protocol_validation_errors",
    "runtime_state_fitness",
]
