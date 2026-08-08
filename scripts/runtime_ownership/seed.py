# ruff: noqa: F401, F403, F405
"""Runtime ownership seed layer."""

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
from .source import *


def _format_for_locator(locator: str, owner_symbol: str) -> dict[str, Any]:
    return {
        "status": "unversioned",
        "rationale": (
            "No reviewed application schema version is declared for this locator; "
            "container syntax and directory shape are not schema versions"
        ),
        "migration_owner": owner_symbol,
    }


def _default_lifecycle(kind: str, owner_symbol: str) -> dict[str, str]:
    retention = {
        "artifact": "retained until owner compaction or superseding snapshot",
        "queue": "retained until acknowledged consumption",
        "lock": "process lifetime",
        "socket": "server process lifetime",
        "schema": "mixed-version compatibility window",
        "worker": "configured launch lifetime",
    }[kind]
    return {
        "retention": retention,
        "recovery_owner": owner_symbol,
        "recovery_contract": (
            "Owner validates durable state before atomic repair or documented replay."
            if kind in {"artifact", "queue"}
            else "Owner recreates the resource from reviewed configuration."
        ),
    }


def _resource_id(kind: str, locator_value: str) -> str:
    return _semantic_id("runtime-resource", {"kind": kind, "locator": locator_value})


def _evidence_for_group(group: list[dict[str, Any]]) -> list[dict[str, Any]]:
    evidence = {(str(row["path"]), int(row["line"])) for row in group}
    for row in group:
        for item in row.get("additional_evidence", []):
            if isinstance(item, dict):
                evidence.add((str(item.get("path") or ""), int(item.get("line") or 0)))
    return [{"path": path, "line": line} for path, line in sorted(evidence)]


def _base_resources(detection: dict[str, Any]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for candidate in detection["resource_candidates"]:
        key = (str(candidate["kind"]), str(candidate["locator"]["value"]))
        groups.setdefault(key, []).append(candidate)
    resources: list[dict[str, Any]] = []
    for (kind, locator_value), group in sorted(groups.items()):
        locator_types = {str(row["locator"]["type"]) for row in group}
        if len(locator_types) != 1:
            raise ValueError(f"locator type conflict for {kind}:{locator_value}")
        owner_symbol = sorted(str(row["owner_symbol"]) for row in group)[0]
        resource_id = _resource_id(kind, locator_value)
        readers = sorted({str(row["owner_symbol"]) for row in group})
        resource: dict[str, Any] = {
            "id": resource_id,
            "kind": kind,
            "locator": {"type": locator_types.pop(), "value": locator_value},
            "owner_package": _owner_package(owner_symbol),
            "owner_symbol": owner_symbol,
            "writers": [owner_symbol],
            "readers": readers,
            "coordination": {"protocol": _coordination_protocol(kind)},
            "lifecycle": _default_lifecycle(kind, owner_symbol),
            "evidence": _evidence_for_group(group),
            "compatibility": sorted(
                {str(value) for row in group for value in row.get("compatibility", [])}
            )
            or [f"stable-locator:{locator_value}"],
            "discovery_ids": sorted(str(row["discovery_id"]) for row in group),
        }
        if kind in {"artifact", "queue"}:
            resource["format"] = _format_for_locator(locator_value, owner_symbol)
        elif kind == "socket":
            resource["socket"] = copy.deepcopy(group[0]["socket"])
        elif kind == "worker":
            resource["worker"] = copy.deepcopy(group[0]["worker"])
        override = RESOURCE_OWNERSHIP_OVERRIDES.get((kind, locator_value))
        if override is not None:
            resource.update(copy.deepcopy(override))
            resource["owner_package"] = _owner_package(str(resource["owner_symbol"]))
            resource["lifecycle"]["recovery_owner"] = resource["owner_symbol"]
            if resource.get("format", {}).get("status") == "unversioned":
                resource["format"]["migration_owner"] = resource["owner_symbol"]
        resources.append(resource)
    resource_by_id = {str(row["id"]): row for row in resources}
    resource_by_key = {
        (str(row["kind"]), str(row["locator"]["value"])): row for row in resources
    }
    worker_leases = {
        "$CHRONOVISOR_ROOT/recall/audit.lock",
        "$CHRONOVISOR_ROOT/runtime/accelerator-inference.lock",
        "$CHRONOVISOR_ROOT/runtime/recall-improvement/run-due.lock",
        "$CHRONOVISOR_ROOT/runtime/research/research-generation.lock",
        "$CHRONOVISOR_ROOT/runtime/sleep-cycle.lock",
    }
    global_protocols = {
        "$CHRONOVISOR_ROOT/runtime/chronovisor-mutation.lock",
        "$CHRONOVISOR_ROOT/runtime/decision-authority.lock",
    }
    for resource in resources:
        if resource["kind"] != "lock":
            continue
        locator = str(resource["locator"]["value"])
        target_keys = LOCK_PROTECTS_LOCATORS.get(locator)
        if not target_keys:
            raise ValueError(f"lock has no reviewed scope mapping: {locator}")
        missing = [target for target in target_keys if target not in resource_by_key]
        if missing:
            raise ValueError(f"lock target is not registered: {locator}: {missing}")
        resource["scope"] = (
            "worker_lease"
            if locator in worker_leases
            else "global_protocol"
            if locator in global_protocols
            else "artifact_sidecar"
        )
        resource["protects"] = sorted(
            str(resource_by_key[target]["id"]) for target in target_keys
        )
    coordinated = {
        "$CHRONOVISOR_ROOT/review/raw-replay-queue.jsonl": (
            "$CHRONOVISOR_ROOT/review/raw-replay-queue.jsonl.lock"
        ),
        "$CHRONOVISOR_ROOT/claims/claims.jsonl": (
            "$CHRONOVISOR_ROOT/claims/claims.jsonl.lock"
        ),
        "$CHRONOVISOR_ROOT/recall/search-label-queue.jsonl": (
            "$CHRONOVISOR_ROOT/recall/search-label-queue.jsonl.lock"
        ),
    }
    for locator, lock_locator in coordinated.items():
        state = next(
            row for row in resources if str(row["locator"]["value"]) == locator
        )
        lock = resource_by_key[("lock", lock_locator)]
        state["coordination"]["lock_id"] = lock["id"]
    assert len(resource_by_id) == len(resources)
    return resources


def _coordination_protocol(kind: str) -> str:
    return {
        "artifact": "single-writer-atomic-replace",
        "queue": "single-writer-append-and-ack",
        "lock": "fcntl-exclusive-lease",
        "socket": "single-server-request-response",
        "schema": "versioned-schema-owner",
        "worker": "single-entrypoint-process-supervision",
    }[kind]


def _exclusion_rows(detection: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "discovery_id": row["discovery_id"],
            "path": row["path"],
            "line": row["line"],
            "module": row["module"],
            "symbol": row["symbol"],
            "reason": row["reason"],
        }
        for row in detection["exclusion_candidates"]
    ]


def _lock_protocol_rows(
    detection: dict[str, Any], resources: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    by_locator = {
        (str(row["kind"]), str(row["locator"]["value"])): row for row in resources
    }
    module_locks = {
        "chronovisor.ops.background_jobs": "$CHRONOVISOR_ROOT/runtime/background-jobs/state.lock",
        "chronovisor.ops.convergence": "$CHRONOVISOR_ROOT/runtime/convergence/state.lock",
        "chronovisor.ingest.page_registry": "$CHRONOVISOR_ROOT/runtime/librarian/page-registry.lock",
        "chronovisor.librarian.collection_authority": "$CHRONOVISOR_ROOT/runtime/librarian/collection-registry.lock",
        "chronovisor.librarian.managed_hold": "$CHRONOVISOR_ROOT/runtime/managed-holds/state.json.lock",
        "chronovisor.knowledge_graph.store": "$CHRONOVISOR_ROOT/knowledge-graph/store.lock",
        "chronovisor.decision.failure_supervisor": "$CHRONOVISOR_ROOT/runtime/failures/state.lock",
        "chronovisor.lab.model_lab": "$CHRONOVISOR_ROOT/runtime/model-lab/model-lab.lock",
        "chronovisor.research.research_consolidation": "$CHRONOVISOR_ROOT/runtime/research/consolidation.lock",
        "chronovisor.recall.recall_auditor": "$CHRONOVISOR_ROOT/recall/audit.lock",
        "chronovisor.recall.recall_improvement": "$CHRONOVISOR_ROOT/runtime/recall-improvement/run-due.lock",
        "chronovisor.research.research_scheduler": "$CHRONOVISOR_ROOT/runtime/research/research-generation.lock",
        "chronovisor.ops.sleep_cycle": "$CHRONOVISOR_ROOT/runtime/sleep-cycle.lock",
        "chronovisor.search.accelerator_lease": "$CHRONOVISOR_ROOT/runtime/accelerator-inference.lock",
    }
    scoped_locks = {
        ("chronovisor.ops.golden_expand", "expand_golden_from_recall_questions"): (
            "$CHRONOVISOR_ROOT/recall/search-label-queue.jsonl.lock"
        ),
        ("chronovisor.search.search_eval", "build_label_queue"): (
            "$CHRONOVISOR_ROOT/recall/search-label-queue.jsonl.lock"
        ),
        (
            "chronovisor.search.search_eval",
            "review_label_queue_with_frontier",
        ): "$CHRONOVISOR_ROOT/recall/search-label-queue.jsonl.lock",
        ("chronovisor.ops.golden_expand", "_search_label_queue_lock"): (
            "$CHRONOVISOR_ROOT/recall/search-label-queue.jsonl.lock"
        ),
        ("chronovisor.search.search_eval", "_search_label_queue_lock"): (
            "$CHRONOVISOR_ROOT/recall/search-label-queue.jsonl.lock"
        ),
        ("chronovisor.recall.claims", "append_page_claims"): (
            "$CHRONOVISOR_ROOT/claims/claims.jsonl.lock"
        ),
        ("chronovisor.recall.claims", "sanitize_claim_ledger"): (
            "$CHRONOVISOR_ROOT/claims/claims.jsonl.lock"
        ),
        ("chronovisor.recall.claims", "_claims_ledger_lock"): (
            "$CHRONOVISOR_ROOT/claims/claims.jsonl.lock"
        ),
        ("chronovisor.raw.raw_replay", "build_queue"): (
            "$CHRONOVISOR_ROOT/review/raw-replay-queue.jsonl.lock"
        ),
        ("chronovisor.raw.raw_replay", "run_pending_queue"): (
            "$CHRONOVISOR_ROOT/review/raw-replay-queue.jsonl.lock"
        ),
        ("chronovisor.recall.evidence_certificate", "append_certificates"): (
            "$CHRONOVISOR_ROOT/recall/evidence-certificate-ledger.jsonl.lock"
        ),
        ("chronovisor.search.semantic_index", "activate_generation"): (
            "$CHRONOVISOR_ROOT/.index/semantic/activation.lock"
        ),
    }
    rows: list[dict[str, Any]] = []
    for candidate in detection["lock_protocol_candidates"]:
        path = str(candidate["path"])
        module = str(candidate["module"])
        scope = str(candidate["scope"])
        lock_locator = scoped_locks.get((module, scope)) or module_locks.get(module)
        lock = by_locator.get(("lock", lock_locator)) if lock_locator else None
        protects = list(lock.get("protects", [])) if isinstance(lock, dict) else []
        primitive_only = lock is None
        row: dict[str, Any] = {
            "discovery_id": candidate["discovery_id"],
            "path": path,
            "line": candidate["line"],
            "module": module,
            "scope": candidate["scope"],
            "operation": candidate["operation"],
            "protocol": candidate["protocol"],
            "coverage": "primitive_only" if primitive_only else "concrete_call_site",
            "lock_ids": [] if lock is None else [str(lock["id"])],
            "protects": []
            if primitive_only
            else sorted(str(item) for item in protects),
        }
        if primitive_only:
            row["reason"] = (
                "low-level or dynamic lock site has no reviewed durable-state binding; "
                "it cannot claim unrelated same-package resources"
            )
        rows.append(row)
    return rows


def _resource_counts(
    resources: list[dict[str, Any]],
    exclusions: list[dict[str, Any]],
    *,
    discovery_count: int,
    lock_protocols: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    by_kind = Counter(str(row["kind"]) for row in resources)
    exclusion_reasons = Counter(str(row["reason"]) for row in exclusions)
    protocols = lock_protocols or []
    direct_protocols = [
        row for row in protocols if not str(row["operation"]).startswith("helper:")
    ]
    return {
        "resources": len(resources),
        "discoveries": discovery_count,
        "exclusions": len(exclusions),
        "by_kind": {kind: by_kind.get(kind, 0) for kind in sorted(RESOURCE_KINDS)},
        "exclusions_by_reason": dict(sorted(exclusion_reasons.items())),
        "entrypoint_workers": sum(
            row["kind"] == "worker" and row["locator"]["type"] == "entrypoint"
            for row in resources
        ),
        "launchd_workers": sum(
            row["kind"] == "worker" and row["locator"]["type"] == "launchd"
            for row in resources
        ),
        "lock_protocol_sites": len(protocols),
        "direct_flock_acquisitions": len(direct_protocols),
        "direct_flock_modules": len({str(row["module"]) for row in direct_protocols}),
        "direct_flock_functions": len(
            {(str(row["module"]), str(row["scope"])) for row in direct_protocols}
        ),
        "lock_protocols_by_kind": dict(
            sorted(Counter(str(row["protocol"]) for row in protocols).items())
        ),
    }


def _id_sets(detection: dict[str, Any]) -> dict[str, set[str]]:
    resources = _base_resources(detection)
    return {
        "discovery_ids": {str(row["discovery_id"]) for row in detection["rows"]},
        "resource_ids": {str(row["id"]) for row in resources},
    }


def _baseline_payload(
    detection: dict[str, Any],
    *,
    retired: dict[str, set[str]] | None = None,
) -> dict[str, Any]:
    resources = _base_resources(detection)
    exclusions = _exclusion_rows(detection)
    active = _id_sets(detection)
    retired_ids = retired or {field: set() for field in BASELINE_ID_FIELDS}
    return {
        "schema_version": BASELINE_SCHEMA_VERSION,
        "source_baseline_head": FROZEN_SOURCE_HEAD,
        "semantic_identity": BASELINE_SEMANTIC_IDENTITY,
        **{
            field: {
                "active": sorted(active[field]),
                "retired": sorted(retired_ids.get(field, set())),
            }
            for field in BASELINE_ID_FIELDS
        },
        "counts": {
            "active": _resource_counts(
                resources,
                exclusions,
                discovery_count=len(detection["rows"]),
                lock_protocols=detection["lock_protocol_candidates"],
            ),
            "retired": {
                field: len(retired_ids.get(field, set()))
                for field in BASELINE_ID_FIELDS
            },
        },
    }


def build_runtime_state_baseline(root: Path) -> dict[str, Any]:
    _index, detection = discover(_snapshot_revision(root.resolve(), FROZEN_SOURCE_HEAD))
    return _baseline_payload(detection)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return _decode_json_document(path.read_bytes())
    except FileNotFoundError:
        return {"absent": True}
    except (OSError, ValueError) as exc:
        return {"load_error": f"{type(exc).__name__}: {exc}"}


def _json_at_revision(root: Path, revision: str, path: Path) -> dict[str, Any]:
    completed = subprocess.run(
        ["git", "show", f"{revision}:{path.as_posix()}"],
        cwd=root,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        return {"absent": True}
    try:
        return _decode_json_document(completed.stdout)
    except ValueError as exc:
        return {"load_error": f"ValueError: {exc}"}


def _load_previous_baseline(root: Path) -> dict[str, Any]:
    empty_retired = {field: [] for field in BASELINE_ID_FIELDS}
    if _run_git(root, "rev-parse", "--is-shallow-repository").strip() != "false":
        message = (
            "runtime ownership history requires a complete Git checkout "
            "(fetch-depth: 0)"
        )
        return {
            "latest": {"load_error": message},
            "historical_retired": empty_retired,
            "history_errors": [message],
        }
    status = _run_git(root, "status", "--porcelain", "--", BASELINE_PATH.as_posix())
    head = subprocess.run(
        ["git", "rev-list", "--parents", "-n", "1", "HEAD"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    if head.returncode != 0 or not head.stdout.strip():
        return {
            "latest": {"absent": True},
            "historical_retired": empty_retired,
            "history_errors": [],
        }
    head_and_parents = head.stdout.split()
    roots = [head_and_parents[0]] if status.strip() else head_and_parents[1:]
    if not roots:
        return {
            "latest": {"absent": True},
            "historical_retired": empty_retired,
            "history_errors": [],
        }
    latest: dict[str, Any] = {"absent": True}
    historical_retired = {field: set() for field in BASELINE_ID_FIELDS}
    history_errors: list[str] = []
    visited: set[str] = set()
    for root_revision in roots:
        revisions = _run_git(
            root,
            "log",
            "--full-history",
            "--format=%H",
            root_revision,
            "--",
            BASELINE_PATH.as_posix(),
        ).splitlines()
        for revision in revisions:
            if revision in visited:
                continue
            visited.add(revision)
            payload = _json_at_revision(root, revision, BASELINE_PATH)
            if payload.get("absent") is True:
                continue
            if latest.get("absent") is True:
                latest = payload
            errors = _historical_seed_errors(payload)
            history_errors.extend(f"{revision}: {error}" for error in errors)
            if not errors:
                for field in BASELINE_ID_FIELDS:
                    historical_retired[field].update(
                        _seed_ids(payload, field, "retired")
                    )
    return {
        "latest": latest,
        "historical_retired": {
            field: sorted(historical_retired[field]) for field in BASELINE_ID_FIELDS
        },
        "history_errors": history_errors,
    }


def _historical_seed_errors(seed: dict[str, Any]) -> list[str]:
    if seed.get("load_error"):
        return [str(seed["load_error"])]
    errors = list(_seed_structure_errors(seed))
    if type(seed.get("schema_version")) is not int or seed.get(
        "schema_version"
    ) != BASELINE_SCHEMA_VERSION:
        errors.append("schema_version")
    if seed.get("source_baseline_head") != FROZEN_SOURCE_HEAD:
        errors.append("source_baseline_head")
    if seed.get("semantic_identity") != BASELINE_SEMANTIC_IDENTITY:
        errors.append("semantic_identity")
    errors.extend(_historical_count_errors(seed))
    for field in BASELINE_ID_FIELDS:
        active = _seed_ids(seed, field, "active")
        retired = _seed_ids(seed, field, "retired")
        if active & retired:
            errors.append(f"{field}.active_retired_overlap")
        bucket = seed.get(field)
        if isinstance(bucket, dict):
            for state in ("active", "retired"):
                values = bucket.get(state)
                if isinstance(values, list) and len(values) != len(set(values)):
                    errors.append(f"{field}.{state}.duplicates")
    return sorted(set(errors))


def _historical_count_errors(seed: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    counts = seed.get("counts")
    if not isinstance(counts, dict) or tuple(counts) != ("active", "retired"):
        return ["counts.keys"]
    active = counts.get("active")
    retired = counts.get("retired")
    active_keys = (
        "resources",
        "discoveries",
        "exclusions",
        "by_kind",
        "exclusions_by_reason",
        "entrypoint_workers",
        "launchd_workers",
        "lock_protocol_sites",
        "direct_flock_acquisitions",
        "direct_flock_modules",
        "direct_flock_functions",
        "lock_protocols_by_kind",
    )
    if not isinstance(active, dict) or tuple(active) != active_keys:
        errors.append("counts.active.keys")
        return errors
    if not isinstance(retired, dict) or tuple(retired) != BASELINE_ID_FIELDS:
        errors.append("counts.retired.keys")
        return errors

    scalar_fields = (
        "resources",
        "discoveries",
        "exclusions",
        "entrypoint_workers",
        "launchd_workers",
        "lock_protocol_sites",
        "direct_flock_acquisitions",
        "direct_flock_modules",
        "direct_flock_functions",
    )
    for field in scalar_fields:
        value = active.get(field)
        if type(value) is not int or value < 0:
            errors.append(f"counts.active.{field}")
    for field in BASELINE_ID_FIELDS:
        value = retired.get(field)
        if type(value) is not int or value < 0:
            errors.append(f"counts.retired.{field}")

    count_maps = ("by_kind", "exclusions_by_reason", "lock_protocols_by_kind")
    for field in count_maps:
        values = active.get(field)
        if not isinstance(values, dict) or any(
            not isinstance(key, str)
            or not key
            or type(value) is not int
            or value < 0
            for key, value in (values.items() if isinstance(values, dict) else ())
        ):
            errors.append(f"counts.active.{field}")
    by_kind = active.get("by_kind")
    if isinstance(by_kind, dict) and tuple(by_kind) != tuple(sorted(RESOURCE_KINDS)):
        errors.append("counts.active.by_kind.keys")

    discovery_ids = seed.get("discovery_ids")
    resource_ids = seed.get("resource_ids")
    active_discovery_ids = (
        discovery_ids.get("active") if isinstance(discovery_ids, dict) else None
    )
    active_resource_ids = (
        resource_ids.get("active") if isinstance(resource_ids, dict) else None
    )
    retired_discovery_ids = (
        discovery_ids.get("retired") if isinstance(discovery_ids, dict) else None
    )
    retired_resource_ids = (
        resource_ids.get("retired") if isinstance(resource_ids, dict) else None
    )
    if isinstance(active_discovery_ids, list) and active.get("discoveries") != len(
        active_discovery_ids
    ):
        errors.append("counts.active.discoveries.length")
    if isinstance(active_resource_ids, list) and active.get("resources") != len(
        active_resource_ids
    ):
        errors.append("counts.active.resources.length")
    for field, values in (
        ("discovery_ids", retired_discovery_ids),
        ("resource_ids", retired_resource_ids),
    ):
        if isinstance(values, list) and retired.get(field) != len(values):
            errors.append(f"counts.retired.{field}.length")

    if (
        isinstance(by_kind, dict)
        and all(type(value) is int for value in by_kind.values())
        and active.get("resources") != sum(by_kind.values())
    ):
        errors.append("counts.active.by_kind.total")
    exclusions_by_reason = active.get("exclusions_by_reason")
    if (
        isinstance(exclusions_by_reason, dict)
        and all(type(value) is int for value in exclusions_by_reason.values())
        and active.get("exclusions") != sum(exclusions_by_reason.values())
    ):
        errors.append("counts.active.exclusions_by_reason.total")
    protocols = active.get("lock_protocols_by_kind")
    if (
        isinstance(protocols, dict)
        and all(type(value) is int for value in protocols.values())
        and active.get("lock_protocol_sites") != sum(protocols.values())
    ):
        errors.append("counts.active.lock_protocols_by_kind.total")
    workers = by_kind.get("worker") if isinstance(by_kind, dict) else None
    if (
        type(workers) is int
        and all(
            type(active.get(field)) is int
            for field in ("entrypoint_workers", "launchd_workers")
        )
        and active["entrypoint_workers"] + active["launchd_workers"] > workers
    ):
        errors.append("counts.active.worker_totals")
    if all(
        type(active.get(field)) is int
        for field in (
            "lock_protocol_sites",
            "direct_flock_acquisitions",
            "direct_flock_modules",
            "direct_flock_functions",
        )
    ):
        if active["direct_flock_acquisitions"] > active["lock_protocol_sites"]:
            errors.append("counts.active.direct_flock_acquisitions.total")
        if active["direct_flock_modules"] > active["direct_flock_acquisitions"]:
            errors.append("counts.active.direct_flock_modules.total")
        if active["direct_flock_functions"] > active["direct_flock_acquisitions"]:
            errors.append("counts.active.direct_flock_functions.total")
    return errors


def _seed_ids(seed: dict[str, Any], field: str, state: str) -> set[str]:
    bucket = seed.get(field)
    if not isinstance(bucket, dict):
        return set()
    values = bucket.get(state)
    if not isinstance(values, list):
        return set()
    return {str(value) for value in values if isinstance(value, str) and value}


def _seed_structure_errors(seed: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if tuple(seed) != BASELINE_ROOT_KEYS:
        errors.append("root_keys")
    if type(seed.get("schema_version")) is not int:
        errors.append("schema_version")
    for field in BASELINE_ID_FIELDS:
        bucket = seed.get(field)
        if not isinstance(bucket, dict):
            errors.append(field)
            continue
        if tuple(bucket) != ("active", "retired"):
            errors.append(f"{field}.keys")
        prefix = {
            "discovery_ids": "runtime-site",
            "resource_ids": "runtime-resource",
        }[field]
        for state in ("active", "retired"):
            values = bucket.get(state)
            if not isinstance(values, list) or any(
                not isinstance(value, str)
                or re.fullmatch(rf"{prefix}:[0-9a-f]{{64}}", value) is None
                for value in values
            ):
                errors.append(f"{field}.{state}")
            elif values != sorted(values):
                errors.append(f"{field}.{state}.order")
    return sorted(errors)


def _seed_state_violations(
    detection: dict[str, Any],
    seed: dict[str, Any],
    frozen: dict[str, Any],
    previous: dict[str, Any],
) -> dict[str, Any]:
    if tuple(previous) == ("latest", "historical_retired", "history_errors"):
        latest = previous.get("latest")
        previous_seed = latest if isinstance(latest, dict) else {"load_error": ""}
        historical_retired_payload = previous.get("historical_retired")
        history_errors_payload = previous.get("history_errors")
        history_errors = (
            [str(error) for error in history_errors_payload]
            if isinstance(history_errors_payload, list)
            else ["runtime ownership history result is malformed"]
        )
        historical_retired = {
            field: (
                {
                    str(value)
                    for value in historical_retired_payload.get(field, [])
                    if isinstance(value, str)
                }
                if isinstance(historical_retired_payload, dict)
                and isinstance(historical_retired_payload.get(field), list)
                else set()
            )
            for field in BASELINE_ID_FIELDS
        }
    else:
        previous_seed = previous
        historical_retired = {
            field: _seed_ids(previous_seed, field, "retired")
            for field in BASELINE_ID_FIELDS
        }
        history_errors = []
    current = _id_sets(detection)
    universe_drift: dict[str, Any] = {}
    current_drift: dict[str, Any] = {}
    overlaps: dict[str, list[str]] = {}
    reintroductions: dict[str, list[str]] = {}
    active_growth: dict[str, list[str]] = {}
    retired_regressions: dict[str, list[str]] = {}
    duplicates: dict[str, list[str]] = {}
    previous_structure_errors = (
        []
        if previous_seed.get("absent") is True or previous_seed.get("load_error")
        else _seed_structure_errors(previous_seed)
    )
    previous_valid = (
        previous_seed.get("absent") is not True
        and not previous_seed.get("load_error")
        and type(previous_seed.get("schema_version")) is int
        and previous_seed.get("schema_version") == BASELINE_SCHEMA_VERSION
        and previous_seed.get("semantic_identity") == BASELINE_SEMANTIC_IDENTITY
        and not previous_structure_errors
    )
    for field in BASELINE_ID_FIELDS:
        active = _seed_ids(seed, field, "active")
        retired = _seed_ids(seed, field, "retired")
        frozen_universe = _seed_ids(frozen, field, "active") | _seed_ids(
            frozen, field, "retired"
        )
        added = sorted((active | retired) - frozen_universe)
        missing = sorted(frozen_universe - (active | retired))
        if added or missing:
            universe_drift[field] = {"added": added, "missing": missing}
        unseeded = sorted(current[field] - active)
        absent = sorted(active - current[field])
        if unseeded or absent:
            current_drift[field] = {
                "unseeded_current": unseeded,
                "seeded_but_absent": absent,
            }
        shared = sorted(active & retired)
        if shared:
            overlaps[field] = shared
        reintroduced = sorted(current[field] & retired)
        if reintroduced:
            reintroductions[field] = reintroduced
        bucket = seed.get(field)
        if isinstance(bucket, dict):
            duplicate_values: set[str] = set()
            for state in ("active", "retired"):
                values = bucket.get(state)
                if isinstance(values, list):
                    duplicate_values.update(
                        str(value) for value in values if values.count(value) > 1
                    )
            if duplicate_values:
                duplicates[field] = sorted(duplicate_values)
        if previous_valid:
            previous_active = _seed_ids(previous_seed, field, "active")
            grown = sorted(active - previous_active)
            regressed = sorted(historical_retired[field] - retired)
            if grown:
                active_growth[field] = grown
            if regressed:
                retired_regressions[field] = regressed
    resources = _base_resources(detection)
    exclusions = _exclusion_rows(detection)
    retired_by_field = {
        field: _seed_ids(seed, field, "retired") for field in BASELINE_ID_FIELDS
    }
    expected_counts = {
        "active": _resource_counts(
            resources,
            exclusions,
            discovery_count=len(detection["rows"]),
            lock_protocols=detection["lock_protocol_candidates"],
        ),
        "retired": {
            field: len(retired_by_field[field]) for field in BASELINE_ID_FIELDS
        },
    }
    current_head = seed.get("source_baseline_head")
    previous_head = previous_seed.get("source_baseline_head")
    return {
        "seed_load_error": str(seed.get("load_error") or ""),
        "seed_schema_version": (
            []
            if type(seed.get("schema_version")) is int
            and seed.get("schema_version") == BASELINE_SCHEMA_VERSION
            else [seed.get("schema_version")]
        ),
        "seed_source_head_drift": (
            [] if current_head == FROZEN_SOURCE_HEAD else [current_head]
        ),
        "seed_semantic_identity_drift": (
            []
            if seed.get("semantic_identity") == BASELINE_SEMANTIC_IDENTITY
            else [seed.get("semantic_identity")]
        ),
        "previous_seed_source_head_drift": (
            []
            if not previous_valid or previous_head == FROZEN_SOURCE_HEAD
            else [previous_head]
        ),
        "seed_source_head_history_drift": (
            {}
            if not previous_valid or current_head == previous_head
            else {"previous": previous_head, "current": current_head}
        ),
        "seed_structure_errors": _seed_structure_errors(seed),
        "seed_universe_drift": universe_drift,
        "seed_current_drift": current_drift,
        "seed_active_retired_overlap": overlaps,
        "retired_id_reintroductions": reintroductions,
        "seed_active_growth": active_growth,
        "seed_retired_regressions": retired_regressions,
        "duplicate_seed_ids": duplicates,
        "seed_count_drift": (
            {}
            if _same_json_value(seed.get("counts"), expected_counts)
            else {"recorded": seed.get("counts"), "expected": expected_counts}
        ),
        "previous_seed_load_error": str(previous_seed.get("load_error") or ""),
        "previous_seed_history_errors": history_errors,
        "previous_seed_structure_errors": previous_structure_errors,
        "previous_seed_schema_version": (
            []
            if previous_seed.get("absent") is True
            or (
                type(previous_seed.get("schema_version")) is int
                and previous_seed.get("schema_version") == BASELINE_SCHEMA_VERSION
            )
            else [previous_seed.get("schema_version")]
        ),
        "previous_seed_semantic_identity_drift": (
            []
            if previous_seed.get("absent") is True
            or previous_seed.get("semantic_identity") == BASELINE_SEMANTIC_IDENTITY
            else [previous_seed.get("semantic_identity")]
        ),
    }


__all__ = [
    "_format_for_locator",
    "_default_lifecycle",
    "_resource_id",
    "_evidence_for_group",
    "_base_resources",
    "_coordination_protocol",
    "_exclusion_rows",
    "_lock_protocol_rows",
    "_resource_counts",
    "_id_sets",
    "_baseline_payload",
    "build_runtime_state_baseline",
    "_load_json",
    "_json_at_revision",
    "_load_previous_baseline",
    "_seed_ids",
    "_seed_structure_errors",
    "_seed_state_violations",
]
