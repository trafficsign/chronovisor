"""Versioned, group-isolated classification fixtures.

The legacy classification fixture remains readable for rollback.  New
experiments live under an explicit epoch directory and never move the current
fixture pointer implicitly.  Gold labels are stripped before a row crosses the
inference boundary.
"""

from __future__ import annotations

from chronovisor.core.hashutil import sha256_prefixed_bytes as sha256_bytes

from chronovisor.core.jsonl_write import atomic_replace_bytes as _atomic_write

from chronovisor.core.jsonl_write import write_jsonl_atomic as _write_jsonl

from chronovisor.core.timeutil import utc_iso_milliseconds as _now

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from chronovisor.core import frontmatter
from chronovisor.classification.classification import (
    ClassificationError,
    classification_authority_status,
    load_udc_package,
)
from chronovisor.classification.classification_engine import (
    DEFAULT_CANDIDATE_LIMIT,
    ENGINE_VERSION,
    CandidateIndex,
    _page_payload,
)
from chronovisor.core.durable_state import read_sealed_json, write_sealed_json
from chronovisor.ingest.page_registry import PageRegistry

FIXTURE_SET_SCHEMA = "chronovisor.classification-fixture-set.v1"
DISABLED_BASELINE_SCHEMA = "chronovisor.classification-disabled-baseline.v1"
INFERENCE_DTO_SCHEMA = "chronovisor.classification-inference-dto.v1"
GOLD_FIELD_PREFIXES = ("gold_", "adjudication_")
_GROUP_TOKEN_RE = re.compile(r"[A-Za-z0-9\u3040-\u30ff\u3400-\u9fff]+")
_JAPANESE_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")
_ASCII_WORD_RE = re.compile(r"[A-Za-z]{3,}")




def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")




def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())






def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


@dataclass(frozen=True)
class FixtureSetPaths:
    root: Path
    dev: Path
    holdout: Path
    reserve: Path
    candidates: Path
    manifest: Path
    calibration: Path
    preregistration: Path
    holdout_results: Path


def fixture_set_paths(root: Path, fixture_epoch: str) -> FixtureSetPaths:
    epoch = fixture_epoch.strip()
    if not epoch or epoch in {".", ".."} or "/" in epoch or "\\" in epoch:
        raise ClassificationError("fixture epoch must be one safe path segment")
    base = root / "classification" / "fixtures" / "epochs" / epoch
    return FixtureSetPaths(
        root=base,
        dev=base / "dev.jsonl",
        holdout=base / "holdout.jsonl",
        reserve=base / "reserve.jsonl",
        candidates=base / "candidates.jsonl",
        manifest=base / "manifest.json",
        calibration=base / "calibration.json",
        preregistration=base / "calibration-preregistration.json",
        holdout_results=base / "holdout-results.jsonl",
    )


def load_fixture_set(manifest_path: Path) -> dict[str, Any]:
    manifest = read_sealed_json(manifest_path)
    if manifest.get("schema") != FIXTURE_SET_SCHEMA:
        raise ClassificationError("unsupported FixtureSet manifest schema")
    base = manifest_path.parent.resolve()
    for name in ("dev", "holdout", "reserve"):
        entry = manifest.get(name)
        if not isinstance(entry, Mapping):
            raise ClassificationError(f"FixtureSet lacks {name} entry")
        path = Path(str(entry.get("path") or "")).resolve()
        if path.parent != base:
            raise ClassificationError(f"FixtureSet {name} escapes epoch directory")
        if not path.is_file():
            raise ClassificationError(f"FixtureSet {name} artifact is missing")
        if sha256_file(path) != entry.get("sha256"):
            raise ClassificationError(f"FixtureSet {name} checksum mismatch")
        if len(read_jsonl(path)) != int(entry.get("count") or -1):
            raise ClassificationError(f"FixtureSet {name} count mismatch")
    if str(manifest.get("fixture_epoch") or "") == str(
        manifest.get("engine_version") or ""
    ):
        raise ClassificationError("fixture epoch must be independent of engine version")
    return manifest


def _first_scalar(meta: Mapping[str, Any], keys: Iterable[str]) -> str:
    for key in keys:
        value = meta.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, list) and value:
            first = value[0]
            if isinstance(first, str) and first.strip():
                return first.strip()
    return ""


def _near_duplicate_key(payload: Mapping[str, Any]) -> str:
    text = " ".join(
        (
            str(payload.get("title") or ""),
            str(payload.get("summary") or ""),
            str(payload.get("excerpt") or "")[:600],
        )
    ).casefold()
    tokens = _GROUP_TOKEN_RE.findall(text)
    signature = " ".join(sorted(set(tokens))[:48])
    return hashlib.sha256(signature.encode("utf-8")).hexdigest()[:20]


def fixture_group(
    root: Path,
    payload: Mapping[str, Any],
) -> tuple[str, str]:
    """Return a conservative provenance group without inventing provenance."""

    path = root / str(payload.get("path") or "")
    text = path.read_text(encoding="utf-8")
    meta, _body = frontmatter.parse(text)
    source = _first_scalar(
        meta,
        (
            "raw_record_sha256",
            "record_sha256",
            "source_session_id",
            "session_id",
            "source_id",
        ),
    )
    if source:
        return f"source:{source}", "frontmatter-source"
    project = _first_scalar(meta, ("project_uid", "project_id"))
    if project:
        return f"project:{project}", "frontmatter-project"
    duplicate_key = _near_duplicate_key(payload)
    return f"near-duplicate:{duplicate_key}", "content-signature"


def build_fixture_pool(
    root: Path,
    *,
    fixture_epoch: str,
    initial_groups: int = 550,
    maximum_groups: int = 800,
    prior_manifest_paths: Sequence[Path] = (),
    prior_fixture_uids: Sequence[str] = (),
) -> dict[str, Any]:
    if not (500 <= initial_groups <= maximum_groups <= 800):
        raise ClassificationError(
            "FixtureSet group limits must satisfy 500<=initial<=max<=800"
        )
    used_groups: set[str] = set()
    used_uids = {str(value) for value in prior_fixture_uids if str(value)}
    for path in prior_manifest_paths:
        prior = load_fixture_set(path)
        used_groups.update(str(value) for value in prior.get("group_ids") or [])

    registry = PageRegistry(root)
    manifest_result = registry.ensure_manifest(include_system=False, write=False)
    state = (
        manifest_result.get("registry", manifest_result)
        if isinstance(manifest_result, Mapping)
        else manifest_result
    )
    package = load_udc_package(root)
    index = CandidateIndex(package)
    representative: dict[str, dict[str, Any]] = {}
    basis_by_group: dict[str, str] = {}
    for uid, row in state["pages"].items():
        if not isinstance(row, Mapping) or row.get("status") != "active":
            continue
        if str(uid) in used_uids:
            continue
        payload = _page_payload(root, str(uid), row)
        group_id, basis = fixture_group(root, payload)
        if group_id in used_groups:
            continue
        payload["fixture_group_id"] = group_id
        payload["fixture_group_basis"] = basis
        payload["candidates"] = index.candidates(payload, limit=DEFAULT_CANDIDATE_LIMIT)
        current = representative.get(group_id)
        if current is None or str(payload["uid"]) < str(current["uid"]):
            representative[group_id] = payload
            basis_by_group[group_id] = basis

    ordered_groups = sorted(
        representative,
        key=lambda value: hashlib.sha256(
            f"{fixture_epoch}:{value}".encode()
        ).hexdigest(),
    )
    selected_groups = ordered_groups[:maximum_groups]
    if len(selected_groups) < 500:
        raise ClassificationError(
            f"FixtureSet requires at least 500 independent groups; found {len(selected_groups)}"
        )
    rows = [representative[group_id] for group_id in selected_groups]
    paths = fixture_set_paths(root, fixture_epoch)
    _write_jsonl(paths.candidates, rows)
    candidate_lock = {
        "schema": "chronovisor.classification-fixture-candidate-lock.v1",
        "fixture_epoch": fixture_epoch,
        "created_at": _now(),
        "engine_version": ENGINE_VERSION,
        "initial_groups": initial_groups,
        "maximum_groups": maximum_groups,
        "available_groups": len(ordered_groups),
        "selected_groups": len(selected_groups),
        "excluded_prior_uid_count": len(used_uids),
        "excluded_prior_group_count": len(used_groups),
        "group_basis_counts": {
            basis: sum(value == basis for value in basis_by_group.values())
            for basis in sorted(set(basis_by_group.values()))
        },
        "candidate_path": str(paths.candidates),
        "candidate_sha256": sha256_file(paths.candidates),
        "group_order_sha256": sha256_bytes("\n".join(selected_groups).encode("utf-8")),
        "adjudication_started": False,
    }
    write_sealed_json(paths.root / "candidate-lock.json", candidate_lock, backup=True)
    return candidate_lock


def inference_dto(row: Mapping[str, Any]) -> dict[str, Any]:
    """Strip labels and adjudication state before model/provider execution."""

    output = {
        key: value
        for key, value in row.items()
        if not key.startswith(GOLD_FIELD_PREFIXES)
        and key not in {"fixture_split", "fixture_rank"}
    }
    output["schema"] = INFERENCE_DTO_SCHEMA
    leaked = [key for key in output if key.startswith(GOLD_FIELD_PREFIXES)]
    if leaked:
        raise ClassificationError(f"gold fields crossed inference boundary: {leaked}")
    return output


def inference_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [inference_dto(row) for row in rows]


def fixture_slice_flags(row: Mapping[str, Any]) -> set[str]:
    text = " ".join(
        (
            str(row.get("title") or ""),
            str(row.get("summary") or ""),
            str(row.get("excerpt") or "")[:1_200],
        )
    )
    flags = set()
    has_japanese = bool(_JAPANESE_RE.search(text))
    has_ascii = bool(_ASCII_WORD_RE.search(text))
    if has_japanese:
        flags.add("japanese")
    if has_japanese and has_ascii:
        flags.add("mixed_language")
    if len(text.strip()) < 240:
        flags.add("short_text")
    tags = row.get("tags") or []
    if isinstance(tags, list) and len(tags) >= 3:
        flags.add("compound_subject")
    created = str(row.get("created_at") or row.get("updated_at") or "")
    if created.startswith(("2025-", "2026-")):
        flags.add("recent")
    return flags


def fixture_slice_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    names = (
        "japanese",
        "mixed_language",
        "short_text",
        "compound_subject",
        "recent",
    )
    return {
        name: sum(name in fixture_slice_flags(row) for row in rows) for name in names
    }


def lock_fixture_set(
    root: Path,
    *,
    fixture_epoch: str,
    adjudicated_rows: Sequence[Mapping[str, Any]],
    adjudicator: str,
    dev_count: int = 200,
    holdout_count: int = 300,
) -> dict[str, Any]:
    paths = fixture_set_paths(root, fixture_epoch)
    candidate_lock = read_sealed_json(paths.root / "candidate-lock.json")
    if candidate_lock.get("adjudication_started") is True:
        raise ClassificationError("FixtureSet candidate lock was already consumed")
    if sha256_file(paths.candidates) != candidate_lock.get("candidate_sha256"):
        raise ClassificationError("FixtureSet candidate pool changed after hash lock")

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen_groups: set[str] = set()
    for source in adjudicated_rows:
        row = dict(source)
        group_id = str(row.get("fixture_group_id") or "")
        valid = bool(
            group_id
            and group_id not in seen_groups
            and row.get("adjudication_status") == "accepted"
            and row.get("gold_primary_notation")
            and isinstance(row.get("gold_allowed_primary_notations"), list)
            and row.get("gold_allowed_primary_notations")
            and row.get("gold_rationale")
            and row.get("source_sha256")
        )
        if not valid:
            rejected.append(row)
            continue
        seen_groups.add(group_id)
        accepted.append(row)
    required = dev_count + holdout_count
    if len(accepted) < required:
        raise ClassificationError(
            f"FixtureSet requires {required} evaluable independent groups; "
            f"found {len(accepted)}"
        )
    accepted.sort(
        key=lambda row: hashlib.sha256(
            f"{fixture_epoch}:{row['fixture_group_id']}".encode()
        ).hexdigest()
    )
    dev = sorted(accepted[:dev_count], key=lambda row: str(row["uid"]))
    holdout = sorted(accepted[dev_count:required], key=lambda row: str(row["uid"]))
    reserve = sorted(accepted[required:], key=lambda row: str(row["uid"]))
    split_groups = [
        {str(row["fixture_group_id"]) for row in split}
        for split in (dev, holdout, reserve)
    ]
    if any(
        split_groups[left] & split_groups[right]
        for left, right in ((0, 1), (0, 2), (1, 2))
    ):
        raise ClassificationError("FixtureSet provenance group leaked across splits")

    for path, rows in (
        (paths.dev, dev),
        (paths.holdout, holdout),
        (paths.reserve, reserve),
    ):
        _write_jsonl(path, rows)
    manifest = {
        "schema": FIXTURE_SET_SCHEMA,
        "fixture_epoch": fixture_epoch,
        "engine_version": ENGINE_VERSION,
        "locked_at": _now(),
        "adjudicator": adjudicator,
        "inference_isolation": "gold-free-dto-one-page-per-model-call",
        "candidate_lock_sha256": sha256_file(paths.root / "candidate-lock.json"),
        "group_ids": sorted(seen_groups),
        "rejected_count": len(rejected),
        "dev": _artifact_entry(paths.dev, opened_at=None),
        "holdout": _artifact_entry(paths.holdout, opened_at=None),
        "reserve": _artifact_entry(paths.reserve, opened_at=None),
        "current_pointer_changed": False,
        "source_scope_sha256": sha256_bytes(
            "\n".join(
                sorted(f"{row['uid']}:{row['source_sha256']}" for row in accepted)
            ).encode("utf-8")
        ),
    }
    write_sealed_json(paths.manifest, manifest, backup=True)
    candidate_lock["adjudication_started"] = True
    candidate_lock["consumed_at"] = _now()
    write_sealed_json(paths.root / "candidate-lock.json", candidate_lock, backup=True)
    return manifest


def _artifact_entry(path: Path, *, opened_at: str | None) -> dict[str, Any]:
    rows = read_jsonl(path)
    return {
        "path": str(path),
        "count": len(rows),
        "sha256": sha256_file(path),
        "opened_at": opened_at,
        "slice_counts": fixture_slice_counts(rows),
    }


def create_disabled_baseline_manifest(
    root: Path,
    *,
    a0_config: Mapping[str, Any],
    receipt_path: Path,
) -> dict[str, Any]:
    package = load_udc_package(root)
    authority = classification_authority_status(root, package=package)
    calibration_path = root / "classification" / "calibration.json"
    calibration_sha = (
        sha256_file(calibration_path) if calibration_path.exists() else None
    )
    payload = {
        "schema": DISABLED_BASELINE_SCHEMA,
        "created_at": _now(),
        "authority_active": False,
        "classification_authority": authority,
        "decision_authority_unchanged": True,
        "candidate_behavior": "A0-production-replay",
        "candidate_limit": DEFAULT_CANDIDATE_LIMIT,
        "a0_config": dict(a0_config),
        "a0_config_sha256": sha256_bytes(_canonical_json(a0_config)),
        "udc_checksum": package.checksum,
        "previous_calibration_path": str(calibration_path),
        "previous_calibration_sha256": calibration_sha,
        "mutation_capability": False,
        "intentional_disabled_sentinel": True,
    }
    write_sealed_json(receipt_path, payload, backup=True)
    return payload
