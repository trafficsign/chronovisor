# ruff: noqa: F401, F403, F405
"""Runtime ownership registry layer."""

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
from .seed import *
from .source import *


def _preserve_manual_resource(
    generated: dict[str, Any], existing: dict[str, Any] | None
) -> dict[str, Any]:
    # Reviewed contracts live in RESOURCE_OWNERSHIP_OVERRIDES.  Existing JSON
    # is deliberately not an authority: otherwise editing an owner in the
    # generated artifact would cause the next generation to preserve it.
    del existing
    return generated


def _registry_payload(
    detection: dict[str, Any],
    seed: dict[str, Any],
    existing: dict[str, Any],
) -> dict[str, Any]:
    existing_resources = {
        str(row.get("id") or ""): row
        for row in existing.get("resources", [])
        if isinstance(row, dict)
    }
    resources = [
        _preserve_manual_resource(row, existing_resources.get(str(row["id"])))
        for row in _base_resources(detection)
    ]
    exclusions = _exclusion_rows(detection)
    lock_protocol_sites = _lock_protocol_rows(detection, resources)
    return {
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "source_baseline_head": FROZEN_SOURCE_HEAD,
        "baseline_sha256": _canonical_sha256(seed),
        "policy": copy.deepcopy(REGISTRY_POLICY),
        "counts": _resource_counts(
            resources,
            exclusions,
            discovery_count=len(detection["rows"]),
            lock_protocols=detection["lock_protocol_candidates"],
        ),
        "resources": resources,
        "exclusions": exclusions,
        "lock_protocol_sites": lock_protocol_sites,
    }


def build_runtime_state_registry(root: Path) -> dict[str, Any]:
    root = root.resolve()
    snapshot = _snapshot_current(root)
    _index, detection = discover(snapshot)
    seed = _load_json(root / BASELINE_PATH)
    previous = _load_previous_baseline(root)
    frozen = build_runtime_state_baseline(root)
    violations = _seed_state_violations(detection, seed, frozen, previous)
    if any(violations.values()):
        raise ValueError(f"runtime state baseline mismatch: {violations}")
    existing = _load_json(root / REGISTRY_PATH)
    return _registry_payload(detection, seed, existing)


def retire_missing_runtime_state(root: Path) -> dict[str, Any]:
    root = root.resolve()
    _index, detection = discover(_snapshot_current(root))
    seed = _load_json(root / BASELINE_PATH)
    current = _id_sets(detection)
    retired: dict[str, set[str]] = {}
    for field in BASELINE_ID_FIELDS:
        active = _seed_ids(seed, field, "active")
        already_retired = _seed_ids(seed, field, "retired")
        additions = current[field] - active
        if additions:
            raise ValueError(
                f"cannot retire {field}; unseeded current IDs: {sorted(additions)}"
            )
        retired[field] = already_retired | (active - current[field])
    updated = _baseline_payload(detection, retired=retired)
    violations = _seed_state_violations(
        detection,
        updated,
        build_runtime_state_baseline(root),
        seed,
    )
    if any(violations.values()):
        raise ValueError(f"invalid runtime state retirement: {violations}")
    return updated


__all__ = [
    "_preserve_manual_resource",
    "_registry_payload",
    "build_runtime_state_registry",
    "retire_missing_runtime_state",
]
