"""Closed native Luna contract and private-bundle importer.

The native worker is an approved OpenAI boundary, but its output remains
teacher-only.  This module is the small boundary between a private Luna
bundle and the durable distillation ledgers: contracts are content addressed,
labels contain hash references rather than source text, and every imported
row is bound to an immutable batch and native receipt record.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import stat
from collections import Counter
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from chronovisor.core import canonical_json
from chronovisor.core.durable_state import file_lock
from chronovisor.recall import recall_distillation_store as store
from chronovisor.recall.recall_distillation_ledger import (
    LedgerMapping,
    ledger_index,
)
from chronovisor.recall.recall_distillation_luna import (
    BATCH_SCHEMA,
    NATIVE_MODEL,
    NATIVE_WORKER_KIND,
    LunaBulkBundle,
    LunaBulkError,
    native_output_sha256,
)
from chronovisor.recall.recall_distillation_luna import (
    SCHEMA as LUNA_BUNDLE_SCHEMA,
)
from chronovisor.recall.recall_distillation_single_teacher_gate import (
    NATIVE_LUNA_MODEL,
    NATIVE_LUNA_REQUEST_REVISION,
    NATIVE_LUNA_ROUTE,
    NATIVE_LUNA_SINGLE_PROFILE,
    NATIVE_LUNA_WORKER_KIND,
    expected_teacher_provider_request_sha256,
    expected_teacher_request_sha256,
    teacher_identity_for_profile,
)

NATIVE_PROFILE_SCHEMA = "chronovisor.recall-distill-native-profile.v1"
NATIVE_PROFILE_CONTRACT_FILE = "native-luna-profile-contract.json"
NATIVE_PROFILE_CONTRACTS_DIR = "native-luna-profile-contracts"
NATIVE_RECEIPTS_FILE = "native-teacher-receipts.jsonl"
NATIVE_IMPORT_LOCK_FILE = "native-luna-import.lock"
NATIVE_COHORT = "native-luna-backfill-v1"
NATIVE_ASSIGNMENT_REVISION = "native-luna-bulk-v1"
NATIVE_RECEIPT_SCHEMA = "chronovisor.recall-native-teacher-receipt.v1"
NATIVE_PAYLOAD_REF_SCHEMA = "chronovisor.recall-native-luna-payload-ref.v1"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}\Z")
_CONTRACT_POINTER_KIND = "native-luna-profile-contract-pointer"
_CONTRACT_KIND = "native-luna-profile"
_RECEIPTS_LEDGER = "native-teacher-receipts.jsonl"
_LABEL_LEDGER = store.LABEL_LEDGER_FILE
# These fields are deterministically copied from the frozen source by
# ``LunaBulkBundle.submit`` after the worker response is received.  They are
# provenance, not part of the native response bytes covered by its receipt.
_BOUND_SOURCE_FIELDS = frozenset(
    {
        "rally_id",
        "candidate_id",
        "as_of_us",
        "repeat_pair_id",
        "repeat_base_sha256",
        "fixed_repeat",
        "order_swap",
        "blind_order",
        "split",
        "group_id",
    }
)
_RAW_FIELDS = frozenset(
    {
        "input",
        "query",
        "context",
        "candidate",
        "candidate_text",
        "rationale",
        "native_receipt",
        "provider_receipt",
    }
)


class NativeDistillationError(ValueError):
    """A native contract, bundle, or imported row is not safe to use."""


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical_sha(value: object) -> str:
    return canonical_json.canonical_json_sha256_strict(value)


def _native_output_sha(label: Mapping[str, Any]) -> str:
    """Hash worker fields after removing deterministic bundle provenance."""

    return native_output_sha256(
        {key: value for key, value in label.items() if key not in _BOUND_SOURCE_FIELDS}
    )


def _is_sha(value: object) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _text(value: object) -> str:
    return value if isinstance(value, str) else ""


def _safe_identifier(value: object) -> bool:
    return isinstance(value, str) and _IDENTIFIER.fullmatch(value) is not None


def _config_value(config: object, name: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def _distillation_dir(root: Path) -> Path:
    return store.distillation_dir(root)


def _contract_pointer_path(root: Path) -> Path:
    return _distillation_dir(root) / NATIVE_PROFILE_CONTRACT_FILE


def _contract_dir(root: Path) -> Path:
    return _distillation_dir(root) / NATIVE_PROFILE_CONTRACTS_DIR


def _receipt_path(root: Path) -> Path:
    return _distillation_dir(root) / NATIVE_RECEIPTS_FILE


def _label_path(root: Path) -> Path:
    return _distillation_dir(root) / _LABEL_LEDGER


def _required_identity() -> dict[str, Any]:
    return teacher_identity_for_profile(NATIVE_LUNA_SINGLE_PROFILE)


def _relevant_config(config: object) -> dict[str, Any]:
    return {
        "teacher_profile": _config_value(
            config, "teacher_profile", NATIVE_LUNA_SINGLE_PROFILE
        ),
        "native_luna_enabled": _config_value(config, "native_luna_enabled", False),
        "teacher_max_inflight": _config_value(config, "teacher_max_inflight", 10),
        "teacher_claim_limit": _config_value(config, "teacher_claim_limit", 500),
        "max_input_bytes": _config_value(config, "max_input_bytes", 12_000),
        "max_candidates": _config_value(config, "max_candidates", 200),
    }


def _validate_config(config: object) -> dict[str, Any]:
    values = _relevant_config(config)
    if values["teacher_profile"] != NATIVE_LUNA_SINGLE_PROFILE:
        raise NativeDistillationError("native Luna teacher profile is required")
    if values["native_luna_enabled"] is not True:
        raise NativeDistillationError("native Luna teacher must be enabled")
    for name, maximum in (
        ("teacher_max_inflight", 10),
        ("teacher_claim_limit", 500),
    ):
        value = values[name]
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
            raise NativeDistillationError(f"{name} is outside the native Luna limit")
    for name, maximum in (("max_input_bytes", 12_000), ("max_candidates", 200)):
        value = values[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 1 <= value <= maximum
        ):
            raise NativeDistillationError(f"{name} is outside the native Luna limit")
    return values


def _contract_payload(config: object) -> dict[str, Any]:
    values = _validate_config(config)
    identity = _required_identity()
    return {
        "kind": _CONTRACT_KIND,
        "profile": NATIVE_LUNA_SINGLE_PROFILE,
        "cohort": NATIVE_COHORT,
        "route": NATIVE_LUNA_ROUTE,
        "teacher_role": NATIVE_LUNA_WORKER_KIND,
        "configured_model": NATIVE_LUNA_MODEL,
        "request_revision": NATIVE_LUNA_REQUEST_REVISION,
        "fixed_identity": identity,
        "native_luna_enabled": True,
        "expires_at": "",
        "teacher_max_inflight": values["teacher_max_inflight"],
        "teacher_claim_limit": values["teacher_claim_limit"],
        "max_input_bytes": values["max_input_bytes"],
        "max_candidates": values["max_candidates"],
        "ledger_authority": "teacher_only_not_verified",
        "relevant_config_sha256": _canonical_sha(values),
    }


def ensure_contract(root: Path, config: object) -> dict[str, Any]:
    """Seal and point the current permanent native Luna profile contract."""

    root = Path(root).expanduser()
    payload = _contract_payload(config)
    artifact_id, _path, artifact = store.write_immutable(
        _contract_dir(root), payload, schema=NATIVE_PROFILE_SCHEMA
    )
    store.write_sealed_state(
        _contract_pointer_path(root),
        {"kind": _CONTRACT_POINTER_KIND, "profile_contract_id": artifact_id},
    )
    checked = _validate_contract(artifact, artifact_id)
    if not checked:
        raise NativeDistillationError("native profile contract failed read-back")
    return checked


def _valid_contract_id(value: object) -> bool:
    return _is_sha(value)


def _contract_keys() -> set[str]:
    return {
        "schema",
        "namespace",
        "artifact_id",
        "kind",
        "profile",
        "cohort",
        "route",
        "teacher_role",
        "configured_model",
        "request_revision",
        "fixed_identity",
        "native_luna_enabled",
        "expires_at",
        "teacher_max_inflight",
        "teacher_claim_limit",
        "max_input_bytes",
        "max_candidates",
        "ledger_authority",
        "relevant_config_sha256",
        "seal_sha256",
    }


def _validate_contract(
    contract: Mapping[str, Any], contract_id: str
) -> dict[str, Any]:
    if set(contract) != _contract_keys() or not _valid_contract_id(contract_id):
        return {}
    if contract.get("artifact_id") != contract_id:
        return {}
    unsigned = {
        key: value
        for key, value in contract.items()
        if key not in {"artifact_id", "seal_sha256"}
    }
    if _canonical_sha(unsigned) != contract_id:
        return {}
    expected_identity = _required_identity()
    if any(
        contract.get(key) != value
        for key, value in {
            "schema": NATIVE_PROFILE_SCHEMA,
            "namespace": "recall-distillation",
            "kind": _CONTRACT_KIND,
            "profile": NATIVE_LUNA_SINGLE_PROFILE,
            "cohort": NATIVE_COHORT,
            "route": NATIVE_LUNA_ROUTE,
            "teacher_role": NATIVE_LUNA_WORKER_KIND,
            "configured_model": NATIVE_LUNA_MODEL,
            "request_revision": NATIVE_LUNA_REQUEST_REVISION,
            "fixed_identity": expected_identity,
            "native_luna_enabled": True,
            "expires_at": "",
            "ledger_authority": "teacher_only_not_verified",
        }.items()
    ):
        return {}
    if not _is_sha(contract.get("relevant_config_sha256")):
        return {}
    for key, maximum in (
        ("teacher_max_inflight", 10),
        ("teacher_claim_limit", 500),
    ):
        value = contract.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
            return {}
    for key, maximum in (("max_input_bytes", 12_000), ("max_candidates", 200)):
        value = contract.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 1 <= value <= maximum
        ):
            return {}
    return dict(contract)


def _pointer_contract_id(root: Path) -> str:
    try:
        state = _contract_pointer_path(root).lstat()
        if (
            stat.S_ISLNK(state.st_mode)
            or not stat.S_ISREG(state.st_mode)
            or stat.S_IMODE(state.st_mode) != 0o600
        ):
            return ""
        pointer = store.read_sealed(
            _contract_pointer_path(root), schema=store.DISTILLATION_SCHEMA
        )
    except (OSError, store.DistillationStoreError):
        return ""
    if pointer.get("kind") != _CONTRACT_POINTER_KIND:
        return ""
    value = pointer.get("profile_contract_id")
    return str(value) if _valid_contract_id(value) else ""


def read_contract(root: Path, contract_id: str = "") -> dict[str, Any]:
    """Read one immutable, closed native contract; return empty on any mismatch."""

    root = Path(root).expanduser()
    requested = contract_id or _pointer_contract_id(root)
    if not _valid_contract_id(requested):
        return {}
    path = _contract_dir(root) / f"{requested}.json"
    try:
        state = path.lstat()
        if (
            stat.S_ISLNK(state.st_mode)
            or not stat.S_ISREG(state.st_mode)
            or stat.S_IMODE(state.st_mode) != 0o600
        ):
            return {}
        contract = store.read_sealed(path, schema=NATIVE_PROFILE_SCHEMA)
    except (OSError, store.DistillationStoreError):
        return {}
    return _validate_contract(contract, requested)


def current_contract_id(root: Path) -> str:
    """Return the current contract id only when its immutable artifact verifies."""

    root = Path(root).expanduser()
    contract_id = _pointer_contract_id(root)
    if not contract_id or not read_contract(root, contract_id):
        return ""
    return contract_id


def _source_ref(source: Mapping[str, Any], manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Project source identity without persisting query, context, or candidate text."""

    task_id = _text(source.get("task_id"))
    input_sha = _text(source.get("input_sha256"))
    source_record_sha = _text(source.get("source_record_sha256"))
    snapshot_sha = _text(manifest.get("source_snapshot_sha256"))
    rally_key = _text(source.get("rally_key")) or _text(source.get("rally_id"))
    rally_id = _text(source.get("rally_id"))
    candidate_id = _text(source.get("candidate_id"))
    candidate_snapshot_sha = _text(source.get("candidate_snapshot_sha256")) or _text(
        source.get("snapshot_sha256")
    )
    query_sha = _text(source.get("query_sha256"))
    candidate_text_sha = _text(source.get("candidate_text_sha256"))
    context_sha = source.get("context_sha256")
    if not isinstance(context_sha, list) or not all(_is_sha(item) for item in context_sha):
        raise NativeDistillationError("native context hash reference is invalid")
    if not (
        _valid_contract_id(task_id)
        and _is_sha(input_sha)
        and _is_sha(source_record_sha)
        and _is_sha(snapshot_sha)
        and _safe_identifier(rally_key)
        and (not rally_id or rally_key == rally_id)
        and _safe_identifier(candidate_id)
        and _is_sha(candidate_snapshot_sha)
        and _is_sha(query_sha)
        and _is_sha(candidate_text_sha)
    ):
        raise NativeDistillationError("native source hash reference is invalid")
    return {
        "schema": NATIVE_PAYLOAD_REF_SCHEMA,
        "task_id": task_id,
        "rally_key": rally_key,
        "candidate_id": candidate_id,
        "candidate_snapshot_sha256": candidate_snapshot_sha,
        "query_sha256": query_sha,
        "context_sha256": list(context_sha),
        "candidate_text_sha256": candidate_text_sha,
        "input_sha256": input_sha,
        "source_record_sha256": source_record_sha,
        "source_snapshot_sha256": snapshot_sha,
    }


def _read_jsonl_private(path: Path) -> tuple[bytes, list[dict[str, Any]]]:
    try:
        state = path.lstat()
        if stat.S_ISLNK(state.st_mode) or not stat.S_ISREG(state.st_mode):
            raise NativeDistillationError("bundle artifact is unsafe")
        raw = path.read_bytes()
    except OSError as exc:
        raise NativeDistillationError("bundle artifact is unavailable") from exc
    if state.st_mode & 0o777 != 0o600 or not raw.endswith(b"\n"):
        raise NativeDistillationError("bundle artifact permissions or tail are invalid")
    rows: list[dict[str, Any]] = []
    for line in raw.splitlines():
        try:
            value = json.loads(line)
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise NativeDistillationError("bundle JSONL is invalid") from exc
        if not isinstance(value, dict):
            raise NativeDistillationError("bundle JSONL row is not an object")
        rows.append(value)
    return raw, rows


def _source_rows(bundle: LunaBulkBundle) -> dict[str, dict[str, Any]]:
    manifest = bundle.manifest
    rows: dict[str, dict[str, Any]] = {}
    for spec in manifest.get("shards", []):
        shard = spec.get("shard") if isinstance(spec, Mapping) else None
        if not isinstance(shard, int):
            raise NativeDistillationError("bundle shard identity is invalid")
        path = bundle.root / "shards" / f"{shard:02d}" / "source.jsonl"
        raw, values = _read_jsonl_private(path)
        if not isinstance(spec.get("source_sha256"), str) or _sha256_bytes(raw) != spec["source_sha256"]:
            raise NativeDistillationError("bundle source hash mismatch")
        for row in values:
            task_id = _text(row.get("task_id"))
            if not _valid_contract_id(task_id) or task_id in rows:
                raise NativeDistillationError("bundle source task identity is invalid")
            if row.get("source_record_sha256") != _canonical_sha(
                {key: value for key, value in row.items() if key != "source_record_sha256"}
            ):
                raise NativeDistillationError("bundle source record hash mismatch")
            rows[task_id] = row
    return rows


def _batch_records(bundle: LunaBulkBundle) -> list[tuple[dict[str, Any], dict[str, Any], str]]:
    manifest = bundle.manifest
    expected_snapshot = _text(manifest.get("source_snapshot_sha256"))
    expected_profile = _text(manifest.get("teacher_profile"))
    all_records: list[tuple[dict[str, Any], dict[str, Any], str]] = []
    seen: set[str] = set()
    for spec in manifest.get("shards", []):
        if not isinstance(spec, Mapping):
            raise NativeDistillationError("bundle shard manifest is invalid")
        shard = spec.get("shard")
        if not isinstance(shard, int):
            raise NativeDistillationError("bundle shard is invalid")
        batches_dir = bundle.root / "shards" / f"{shard:02d}" / "batches"
        try:
            paths = sorted(batches_dir.glob("batch-*.json"))
        except OSError as exc:
            raise NativeDistillationError("bundle batches are unavailable") from exc
        cursor = 0
        for path in paths:
            try:
                state = path.lstat()
                raw = path.read_bytes()
            except OSError as exc:
                raise NativeDistillationError("bundle batch is unavailable") from exc
            if stat.S_ISLNK(state.st_mode) or not stat.S_ISREG(state.st_mode) or state.st_mode & 0o777 != 0o600:
                raise NativeDistillationError("bundle batch is unsafe")
            if not raw.endswith(b"\n"):
                raise NativeDistillationError("bundle batch has incomplete tail")
            try:
                batch = json.loads(raw)
            except (UnicodeError, json.JSONDecodeError) as exc:
                raise NativeDistillationError("bundle batch is invalid") from exc
            if not isinstance(batch, dict):
                raise NativeDistillationError("bundle batch is not an object")
            if (
                batch.get("schema") != BATCH_SCHEMA
                or batch.get("shard") != shard
                or batch.get("cursor") != cursor
                or path.name != f"batch-{cursor:06d}.json"
                or batch.get("source_sha256") != spec.get("source_sha256")
                or batch.get("source_snapshot_sha256") != expected_snapshot
                or batch.get("teacher_profile") != expected_profile
                or batch.get("native_worker") != NATIVE_WORKER_KIND
            ):
                raise NativeDistillationError("bundle batch provenance mismatch")
            records = batch.get("records")
            if not isinstance(records, list) or not records:
                raise NativeDistillationError("bundle batch records are invalid")
            if batch.get("records_sha256") != _canonical_sha(records):
                raise NativeDistillationError("bundle batch records hash mismatch")
            # Bind receipts to the exact immutable batch bytes, including the
            # canonical JSONL tail and all provenance fields.
            batch_sha = _sha256_bytes(raw)
            for label in records:
                if not isinstance(label, dict):
                    raise NativeDistillationError("bundle label is not an object")
                task_id = _text(label.get("task_id"))
                if not _valid_contract_id(task_id) or task_id in seen:
                    raise NativeDistillationError("bundle label task identity is invalid")
                seen.add(task_id)
                all_records.append((batch, label, batch_sha))
            cursor += len(records)
        if cursor != int(spec.get("records", -1)):
            raise NativeDistillationError("bundle batch cursor is incomplete")
    return all_records


def _safe_features(source: Mapping[str, Any]) -> dict[str, Any]:
    value = source.get("features")
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, Any] = {}
    for key, item in value.items():
        if (
            not _safe_identifier(key)
            or isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
        ):
            raise NativeDistillationError("native feature metadata is invalid")
        result[key] = item
    return result


def _safe_assignment(source: Mapping[str, Any]) -> dict[str, Any]:
    value = source.get("assignment")
    if not isinstance(value, Mapping):
        value = {}
    allowed = {
        "revision",
        "profile",
        "split",
        "probe",
        "owner",
        "routes",
        "probe_revision",
        "repeat_pair_id",
        "fixed_repeat",
        "order_swap",
        "blind_order",
        "probe_batch_id",
        "order_variant",
        "candidate_position",
    }
    unexpected = set(value) - allowed
    if unexpected:
        raise NativeDistillationError("native assignment metadata is unknown")
    result = {key: value[key] for key in allowed if key in value}
    for key in result:
        if key in {"probe", "fixed_repeat", "order_swap"}:
            if not isinstance(result[key], bool):
                raise NativeDistillationError("native assignment boolean is invalid")
        elif key == "candidate_position":
            if (
                isinstance(result[key], bool)
                or not isinstance(result[key], int)
                or result[key] < 0
            ):
                raise NativeDistillationError("native assignment position is invalid")
        elif key == "routes":
            continue
        elif not _safe_identifier(result[key]):
            raise NativeDistillationError("native assignment metadata is invalid")
    if "routes" in result:
        routes = result["routes"]
        if not isinstance(routes, list) or not all(_safe_identifier(item) for item in routes):
            raise NativeDistillationError("native assignment routes are invalid")
        result["routes"] = list(routes)
    return result


def _receipt_payload(
    *,
    contract_id: str,
    manifest: Mapping[str, Any],
    source: Mapping[str, Any],
    label: Mapping[str, Any],
    batch_sha: str,
) -> dict[str, Any]:
    receipt = label.get("native_receipt")
    if not isinstance(receipt, Mapping):
        raise NativeDistillationError("native receipt is missing")
    output_sha_value = receipt.get("output_sha256")
    response_sha_value = receipt.get("response_sha256")
    if (
        output_sha_value is not None
        and response_sha_value is not None
        and output_sha_value != response_sha_value
    ):
        raise NativeDistillationError("native output receipt hash fields conflict")
    output_sha = (
        output_sha_value
        if output_sha_value is not None
        else response_sha_value
    )
    if not _is_sha(output_sha):
        raise NativeDistillationError("native output receipt hash is invalid")
    task_id = _text(source.get("task_id"))
    if receipt.get("task_id") != task_id or label.get("task_id") != task_id:
        raise NativeDistillationError("native receipt task binding mismatch")
    assignment_id = receipt.get("assignment_id")
    run_id = receipt.get("run_id")
    if not _safe_identifier(assignment_id) or not _safe_identifier(run_id):
        raise NativeDistillationError("native receipt provenance is invalid")
    input_sha = _text(source.get("input_sha256"))
    source_record_sha = _text(source.get("source_record_sha256"))
    snapshot_sha = _text(manifest.get("source_snapshot_sha256"))
    if (
        not _is_sha(input_sha)
        or not _is_sha(source_record_sha)
        or not _is_sha(snapshot_sha)
        or not _is_sha(batch_sha)
    ):
        raise NativeDistillationError("native receipt source hash is invalid")
    source_ref = _source_ref(source, manifest)
    return {
        "receipt_schema": NATIVE_RECEIPT_SCHEMA,
        "profile_contract_id": contract_id,
        "source_snapshot_sha256": snapshot_sha,
        "source_record_sha256": source_record_sha,
        "rally_key": source_ref["rally_key"],
        "candidate_id": source_ref["candidate_id"],
        "candidate_snapshot_sha256": source_ref["candidate_snapshot_sha256"],
        "query_sha256": source_ref["query_sha256"],
        "context_sha256": source_ref["context_sha256"],
        "candidate_text_sha256": source_ref["candidate_text_sha256"],
        "task_id": task_id,
        "input_sha256": input_sha,
        "output_sha256": output_sha,
        "assignment_id": assignment_id,
        "run_id": run_id,
        "batch_sha256": batch_sha,
        "receipt_binding": "native-worker-output-and-batch-v1",
    }


def _receipt_sha(receipt: Mapping[str, Any]) -> str:
    return _canonical_sha(receipt)


def _row_from_label(
    *,
    contract: Mapping[str, Any],
    manifest: Mapping[str, Any],
    source: Mapping[str, Any],
    label: Mapping[str, Any],
    receipt: Mapping[str, Any],
    receipt_sha: str,
) -> dict[str, Any]:
    identity = cast(dict[str, Any], contract["fixed_identity"])
    profile_contract_id = _text(contract.get("artifact_id"))
    payload_source = _source_ref(source, manifest)
    payload_digest = _canonical_sha(payload_source)
    verdict = _text(label.get("verdict"))
    if verdict not in {"relevant", "irrelevant", "uncertain"}:
        raise NativeDistillationError("native label verdict is invalid")
    split = _text(source.get("split"))
    if split not in {"train", "validation", "test", "embargo"}:
        raise NativeDistillationError("native source split is invalid")
    split_plan_id = _text(manifest.get("split_plan_id"))
    if not _is_sha(split_plan_id):
        raise NativeDistillationError("native split plan is missing")
    group_id = _text(source.get("group_id"))
    rally_id = _text(source.get("rally_id"))
    candidate_id = _text(source.get("candidate_id"))
    as_of_us = source.get("as_of_us")
    if (
        not _safe_identifier(group_id)
        or not _safe_identifier(rally_id)
        or not _safe_identifier(candidate_id)
        or isinstance(as_of_us, bool)
        or not isinstance(as_of_us, int)
        or as_of_us < 0
    ):
        raise NativeDistillationError("native source identity metadata is invalid")
    as_of = _text(source.get("as_of"))
    if as_of:
        try:
            parsed_as_of = datetime.fromisoformat(as_of.replace("Z", "+00:00"))
        except ValueError as exc:
            raise NativeDistillationError("native temporal identity is invalid") from exc
        if parsed_as_of.tzinfo is None or parsed_as_of.utcoffset() is None:
            raise NativeDistillationError("native temporal identity is invalid")
        normalized_as_of = parsed_as_of.astimezone(UTC)
        if int(normalized_as_of.timestamp() * 1_000_000) != as_of_us:
            raise NativeDistillationError("native temporal identity is inconsistent")
        as_of = normalized_as_of.isoformat().replace("+00:00", "Z")
    else:
        as_of = datetime.fromtimestamp(as_of_us / 1_000_000, UTC).isoformat().replace(
            "+00:00", "Z"
        )
    assignment = _safe_assignment(source)
    if assignment:
        if any(
            assignment.get(key) != expected
            for key, expected in {
                "revision": NATIVE_ASSIGNMENT_REVISION,
                "profile": NATIVE_LUNA_SINGLE_PROFILE,
                "split": split,
                "owner": "native-luna",
            }.items()
            if key in assignment
        ):
            raise NativeDistillationError("native assignment identity is inconsistent")
        expected_probe = bool(source.get("repeat_pair_id"))
        if "probe" in assignment and assignment["probe"] is not expected_probe:
            raise NativeDistillationError("native assignment probe is inconsistent")
        for key in ("repeat_pair_id", "fixed_repeat", "order_swap", "blind_order"):
            expected = (
                source.get(key) is True
                if key in {"fixed_repeat", "order_swap"}
                else source.get(key, "")
            )
            if key in assignment and assignment[key] != expected:
                raise NativeDistillationError("native assignment repeat metadata is inconsistent")
    if not assignment:
        assignment = {
            "revision": NATIVE_ASSIGNMENT_REVISION,
            "profile": NATIVE_LUNA_SINGLE_PROFILE,
            "split": split,
            "probe": bool(source.get("repeat_pair_id")),
            "owner": "native-luna",
            "repeat_pair_id": _text(source.get("repeat_pair_id")),
            "fixed_repeat": source.get("fixed_repeat") is True,
            "order_swap": source.get("order_swap") is True,
            "blind_order": _text(source.get("blind_order")),
        }
    rationale = label.get("rationale")
    reason_code = _text(label.get("reason_code"))
    if not _safe_identifier(reason_code):
        raise NativeDistillationError("native reason code is invalid")
    confidence = label.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not math.isfinite(float(confidence)) or not 0 <= confidence <= 1:
        raise NativeDistillationError("native confidence is invalid")
    row: dict[str, Any] = {
        "kind": "teacher-label",
        "source": "teacher-label",
        "status": "completed" if verdict in {"relevant", "irrelevant"} else "abstained",
        "error_class": None,
        "authority": "teacher-only",
        "truth_authority": "teacher_only_not_verified",
        "dimension": "relevance",
        "verdict": verdict,
        "confidence": confidence,
        "reason_code": reason_code,
        "rationale_sha256": _sha256_bytes(_text(rationale).encode("utf-8")),
        "rationale_chars": len(_text(rationale)),
        "profile": NATIVE_LUNA_SINGLE_PROFILE,
        "cohort": NATIVE_COHORT,
        "profile_contract_id": profile_contract_id,
        "route": NATIVE_LUNA_ROUTE,
        "teacher_role": NATIVE_LUNA_WORKER_KIND,
        "native_luna": True,
        "route_identity": dict(identity["route_identity"]),
        "route_identity_exact": True,
        "identity_revision": identity["revision"],
        "request_revision": identity["request_revision"],
        "model_digest": identity["model_digest"],
        "prompt_sha256": identity["prompt_template_sha256"],
        "schema_sha256": identity["schema_revision_sha256"],
        "route_digest": identity["route_digest"],
        "expires_at": "",
        "payload_digest": payload_digest,
        "payload_source": payload_source,
        "work_id": _text(source["task_id"]),
        "provider_receipt_sha256": receipt_sha,
        "native_receipt_sha256": receipt_sha,
        "native_assignment_id": receipt["assignment_id"],
        "native_run_id": receipt["run_id"],
        "native_batch_sha256": receipt["batch_sha256"],
        "assignment_revision": NATIVE_ASSIGNMENT_REVISION,
        "assignment": assignment,
        "features": _safe_features(source),
        "rally_id": rally_id,
        "candidate_id": candidate_id,
        "as_of": as_of,
        "as_of_us": source.get("as_of_us"),
        "group_id": group_id,
        "group_identity_exact": True,
        "split": split,
        "split_plan_id": split_plan_id,
        "fixed_split_plan": True,
        "probe": bool(source.get("repeat_pair_id")),
        "repeat_pair_id": _text(source.get("repeat_pair_id")),
        "fixed_repeat": source.get("fixed_repeat") is True,
        "order_swap": source.get("order_swap") is True,
        "blind_order": _text(source.get("blind_order")),
        "future_leakage": False,
        "feature_parity": True,
        "negative_veto_conflict": False,
        "locked_test_read_only": split == "test",
        "locked_test_evidence_ref": f"split-plan:{split_plan_id}" if split == "test" else "",
    }
    row["request_sha256"] = expected_teacher_request_sha256(
        profile_contract_id=profile_contract_id,
        payload_digest=payload_digest,
        profile=NATIVE_LUNA_SINGLE_PROFILE,
    )
    row["provider_request_sha256"] = expected_teacher_provider_request_sha256(
        profile_contract_id=profile_contract_id,
        payload_digest=payload_digest,
        work_id=row["work_id"],
        expires_at="",
        profile=NATIVE_LUNA_SINGLE_PROFILE,
    )
    return row


def _receipt_matches_row(
    root: Path,
    row: Mapping[str, Any],
    receipts: Mapping[str, Mapping[str, Any]] | None = None,
) -> bool:
    receipt_id = _text(row.get("provider_receipt_sha256"))
    if not _is_sha(receipt_id):
        return False
    matches: list[Mapping[str, Any]]
    if receipts is None:
        receipts = _receipt_index(root)
        if receipts is None:
            return False
    try:
        item = receipts.get(receipt_id)
    except (
        OSError,
        TypeError,
        ValueError,
        sqlite3.Error,
        store.DistillationStoreError,
    ):
        return False
    matches = [item] if item is not None else []
    if len(matches) != 1 or not isinstance(matches[0], Mapping):
        return False
    receipt = matches[0]
    payload_source = row.get("payload_source")
    if not isinstance(payload_source, Mapping):
        return False
    unsigned = {
        key: value
        for key, value in receipt.items()
        if key
        not in {
            "schema",
            "namespace",
            "previous_sha256",
            "record_sha256",
            "seal_sha256",
            "receipt_sha256",
        }
    }
    return (
        receipt.get("receipt_schema") == NATIVE_RECEIPT_SCHEMA
        and receipt.get("task_id") == row.get("work_id")
        and receipt.get("profile_contract_id") == row.get("profile_contract_id")
        and receipt.get("source_record_sha256")
        == payload_source.get("source_record_sha256")
        and receipt.get("source_snapshot_sha256")
        == payload_source.get("source_snapshot_sha256")
        and receipt.get("rally_key") == payload_source.get("rally_key")
        and receipt.get("candidate_id") == payload_source.get("candidate_id")
        and receipt.get("candidate_snapshot_sha256")
        == payload_source.get("candidate_snapshot_sha256")
        and receipt.get("query_sha256") == payload_source.get("query_sha256")
        and receipt.get("context_sha256") == payload_source.get("context_sha256")
        and receipt.get("candidate_text_sha256")
        == payload_source.get("candidate_text_sha256")
        and receipt.get("input_sha256") == payload_source.get("input_sha256")
        and _is_sha(receipt.get("output_sha256"))
        and receipt.get("batch_sha256") == row.get("native_batch_sha256")
        and _receipt_sha(unsigned) == receipt_id
    )


def _ledger_mapping(path: Path, field: str) -> LedgerMapping | None:
    """Open one verified projection while preserving native ledger checks."""

    try:
        state = path.lstat()
    except FileNotFoundError:
        return None
    except OSError:
        return None
    if (
        stat.S_ISLNK(state.st_mode)
        or not stat.S_ISREG(state.st_mode)
        or stat.S_IMODE(state.st_mode) != 0o600
    ):
        return None
    try:
        return LedgerMapping(ledger_index(path), field)
    except (
        OSError,
        TypeError,
        ValueError,
        sqlite3.Error,
        store.DistillationStoreError,
    ):
        return None


def _receipt_index(root: Path) -> Mapping[str, dict[str, Any]] | None:
    """Return a verified receipt mapping backed by an incremental projection."""

    return _ledger_mapping(_receipt_path(root), "receipt_sha256")


def _label_index(root: Path) -> Mapping[str, dict[str, Any]] | None:
    """Return a verified label mapping backed by an incremental projection."""

    return _ledger_mapping(_label_path(root), "record_sha256")


def _label_work_index(root: Path) -> Mapping[str, dict[str, Any]] | None:
    """Return labels keyed by work id for bounded import idempotency checks."""

    return _ledger_mapping(_label_path(root), "work_id")


def _existing_ledger_mapping(
    path: Path,
    mapping: Mapping[str, dict[str, Any]] | None,
    message: str,
) -> Mapping[str, dict[str, Any]] | None:
    """Treat a missing ledger as empty, but reject an invalid existing one."""

    if mapping is not None:
        return mapping
    try:
        path.lstat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise NativeDistillationError(message) from exc
    raise NativeDistillationError(message)


def _mapping_has(
    mapping: Mapping[str, dict[str, Any]] | None,
    key: str,
    message: str,
) -> bool:
    if mapping is None:
        return False
    try:
        return mapping.get(key) is not None
    except (
        OSError,
        TypeError,
        ValueError,
        sqlite3.Error,
        store.DistillationStoreError,
    ) as exc:
        raise NativeDistillationError(message) from exc


def _row_integrity_with_state(
    row: Mapping[str, Any],
    root: Path | None = None,
    *,
    contract: Mapping[str, Any] | None = None,
    receipts: Mapping[str, Mapping[str, Any]] | None = None,
) -> bool:
    """Validate a sanitized native row and, when available, its sealed receipt."""

    if not isinstance(row, Mapping):
        return False
    # Production rows are hash references only.  A caller that tries to carry
    # private source or native response text into the ledger must fail closed.
    if any(key in row for key in _RAW_FIELDS):
        return False
    identity = _required_identity()
    if any(
        row.get(key) != value
        for key, value in {
            "source": "teacher-label",
            "profile": NATIVE_LUNA_SINGLE_PROFILE,
            "cohort": NATIVE_COHORT,
            "route": NATIVE_LUNA_ROUTE,
            "teacher_role": NATIVE_LUNA_WORKER_KIND,
            "native_luna": True,
            "route_identity": identity["route_identity"],
            "route_identity_exact": True,
            "identity_revision": identity["revision"],
            "request_revision": identity["request_revision"],
            "model_digest": identity["model_digest"],
            "prompt_sha256": identity["prompt_template_sha256"],
            "schema_sha256": identity["schema_revision_sha256"],
            "route_digest": identity["route_digest"],
            "expires_at": "",
            "truth_authority": "teacher_only_not_verified",
            "authority": "teacher-only",
            "dimension": "relevance",
            "future_leakage": False,
            "feature_parity": True,
            "negative_veto_conflict": False,
        }.items()
    ):
        return False
    contract_id = _text(row.get("profile_contract_id"))
    payload_digest = _text(row.get("payload_digest"))
    if not _valid_contract_id(contract_id) or not _is_sha(payload_digest):
        return False
    source = row.get("payload_source")
    if (
        not isinstance(source, Mapping)
        or set(source)
        != {
            "schema",
            "task_id",
            "rally_key",
            "candidate_id",
            "candidate_snapshot_sha256",
            "query_sha256",
            "context_sha256",
            "candidate_text_sha256",
            "input_sha256",
            "source_record_sha256",
            "source_snapshot_sha256",
        }
        or source.get("schema") != NATIVE_PAYLOAD_REF_SCHEMA
        or not _valid_contract_id(source.get("task_id"))
        or not _safe_identifier(source.get("rally_key"))
        or not _safe_identifier(source.get("candidate_id"))
        or not _is_sha(source.get("candidate_snapshot_sha256"))
        or not _is_sha(source.get("query_sha256"))
        or not isinstance(source.get("context_sha256"), list)
        or not all(_is_sha(item) for item in source.get("context_sha256", []))
        or not _is_sha(source.get("candidate_text_sha256"))
        or not _is_sha(source.get("input_sha256"))
        or not _is_sha(source.get("source_record_sha256"))
        or not _is_sha(source.get("source_snapshot_sha256"))
        or source.get("task_id") != row.get("work_id")
        or source.get("rally_key") != row.get("rally_id")
        or source.get("candidate_id") != row.get("candidate_id")
        or _canonical_sha(source) != payload_digest
    ):
        return False
    if root is not None:
        checked_contract = contract or read_contract(root, contract_id)
        if not checked_contract or checked_contract.get("artifact_id") != contract_id:
            return False
        if contract is None and current_contract_id(root) != contract_id:
            return False
        if not _receipt_matches_row(root, row, receipts):
            return False
    if row.get("status") not in {"completed", "abstained"}:
        return False
    if row.get("verdict") not in {"relevant", "irrelevant", "uncertain"}:
        return False
    confidence = row.get("confidence")
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isfinite(float(confidence))
        or not 0 <= confidence <= 1
    ):
        return False
    if not _safe_identifier(row.get("reason_code")):
        return False
    rationale_chars = row.get("rationale_chars")
    if (
        isinstance(rationale_chars, bool)
        or not isinstance(rationale_chars, int)
        or rationale_chars < 0
        or not _is_sha(row.get("rationale_sha256"))
    ):
        return False
    for key in (
        "model_digest",
        "prompt_sha256",
        "schema_sha256",
        "route_digest",
        "split_plan_id",
        "provider_receipt_sha256",
        "request_sha256",
        "provider_request_sha256",
        "native_receipt_sha256",
        "native_batch_sha256",
    ):
        if not _is_sha(row.get(key)):
            return False
    expected_request = expected_teacher_request_sha256(
        profile_contract_id=contract_id,
        payload_digest=payload_digest,
        profile=NATIVE_LUNA_SINGLE_PROFILE,
    )
    expected_provider_request = expected_teacher_provider_request_sha256(
        profile_contract_id=contract_id,
        payload_digest=payload_digest,
        work_id=_text(row.get("work_id")),
        expires_at="",
        profile=NATIVE_LUNA_SINGLE_PROFILE,
    )
    return (
        row.get("request_sha256") == expected_request
        and row.get("provider_request_sha256") == expected_provider_request
        and row.get("provider_receipt_sha256") != row.get("provider_request_sha256")
        and row.get("native_receipt_sha256") == row.get("provider_receipt_sha256")
        and _valid_contract_id(row.get("work_id"))
        and _safe_identifier(row.get("native_assignment_id"))
        and _safe_identifier(row.get("native_run_id"))
        and _safe_identifier(row.get("group_id"))
        and _text(row.get("split")) in {"train", "validation", "test", "embargo"}
    )


def row_integrity(row: Mapping[str, Any], root: Path | None = None) -> bool:
    """Validate a sanitized native row and, when available, its sealed receipt."""

    if _row_integrity_with_state(row, root=root):
        return True
    # ``materialize_training_rows`` intentionally projects away teacher
    # confidence/rationale fields.  When the projection carries the sealed
    # label record id, validate the complete source row once and compare every
    # overlapping field.  This keeps projections hash-bound without requiring
    # private teacher response text in the training snapshot.
    if root is None or not isinstance(row, Mapping) or any(
        key in row for key in _RAW_FIELDS
    ):
        return False
    record_id = _text(row.get("label_record_sha256"))
    if not _is_sha(record_id):
        return False
    labels = _label_index(root)
    if labels is None:
        return False
    try:
        source = labels.get(record_id)
    except (
        OSError,
        TypeError,
        ValueError,
        sqlite3.Error,
        store.DistillationStoreError,
    ):
        return False
    if source is None:
        return False
    contract_id = _text(source.get("profile_contract_id"))
    contract = read_contract(root, contract_id)
    receipts = _receipt_index(root)
    if (
        not contract
        or current_contract_id(root) != contract_id
        or receipts is None
        or not _row_integrity_with_state(
        source,
        root=root,
        contract=contract,
        receipts=receipts,
        )
    ):
        return False
    return all(source.get(key) == value for key, value in row.items() if key in source)


def validate_label(root: Path, label: Mapping[str, Any]) -> bool:
    """Validate one already-sanitized native teacher row against current state."""

    return row_integrity(label, root=root)


@contextmanager
def _import_lock(root: Path) -> Iterator[None]:
    lock_path = _distillation_dir(root) / NATIVE_IMPORT_LOCK_FILE
    with file_lock(lock_path):
        yield


def import_bundle(root: Path, bundle_path: Path) -> dict[str, Any]:
    """Validate and idempotently import one complete private Luna bundle."""

    root = Path(root).expanduser()
    bundle_path = Path(bundle_path).expanduser()
    with _import_lock(root):
        contract = read_contract(root)
        if not contract:
            raise NativeDistillationError("native Luna contract is unavailable")
        contract_id = _text(contract["artifact_id"])
        try:
            bundle = LunaBulkBundle.open(bundle_path)
        except (LunaBulkError, OSError) as exc:
            raise NativeDistillationError("native Luna bundle is invalid") from exc
        manifest = bundle.manifest
        if (
            manifest.get("schema") != LUNA_BUNDLE_SCHEMA
            or manifest.get("teacher_profile") != NATIVE_LUNA_SINGLE_PROFILE
            or manifest.get("cohort") != NATIVE_COHORT
            or manifest.get("teacher", {}).get("route") != NATIVE_LUNA_ROUTE
            or manifest.get("teacher", {}).get("model") != NATIVE_MODEL
            or not _is_sha(manifest.get("source_snapshot_sha256"))
            or not _is_sha(manifest.get("split_plan_id"))
        ):
            raise NativeDistillationError("native Luna bundle contract mismatch")
        status = bundle.status()
        if status.get("remaining") != 0:
            raise NativeDistillationError("native Luna bundle is incomplete")
        try:
            completion = bundle.finalize()
        except (LunaBulkError, OSError) as exc:
            raise NativeDistillationError("native Luna bundle completion is invalid") from exc
        source_rows = _source_rows(bundle)
        batch_records = _batch_records(bundle)
        if len(source_rows) != int(manifest["cardinality"]["records"]) or len(batch_records) != len(source_rows):
            raise NativeDistillationError("native Luna bundle cardinality mismatch")
        rows: list[dict[str, Any]] = []
        receipts: list[dict[str, Any]] = []
        seen_receipts: set[str] = set()
        for _batch, label, batch_sha in batch_records:
            task_id = _text(label.get("task_id"))
            source = source_rows.get(task_id)
            if source is None or label.get("input_sha256") not in (None, source.get("input_sha256")):
                raise NativeDistillationError("native label/source binding mismatch")
            receipt_payload = _receipt_payload(
                contract_id=contract_id,
                manifest=manifest,
                source=source,
                label=label,
                batch_sha=batch_sha,
            )
            # The native receipt's output hash is a claim made by the worker;
            # verify it against the immutable response fields before accepting
            # the claim into the durable ledger.
            if receipt_payload["output_sha256"] != _native_output_sha(label):
                raise NativeDistillationError("native output receipt hash mismatch")
            receipt_sha = _receipt_sha(receipt_payload)
            if receipt_sha in seen_receipts:
                raise NativeDistillationError("native receipt is duplicated")
            seen_receipts.add(receipt_sha)
            claimed_receipt = label.get("provider_receipt_sha256")
            if claimed_receipt not in (None, "", receipt_sha):
                raise NativeDistillationError("native receipt self-report mismatch")
            receipt = {**receipt_payload, "receipt_sha256": receipt_sha}
            row = _row_from_label(
                contract=contract,
                manifest=manifest,
                source=source,
                label=label,
                receipt=receipt,
                receipt_sha=receipt_sha,
            )
            if not row_integrity(row):
                raise NativeDistillationError("native row integrity failed")
            rows.append(row)
            receipts.append(receipt)
        receipt_path = _receipt_path(root)
        receipt_view = _existing_ledger_mapping(
            receipt_path,
            _receipt_index(root),
            "native receipt ledger is invalid",
        )
        label_path = _label_path(root)
        label_view = _existing_ledger_mapping(
            label_path,
            _label_work_index(root),
            "label ledger is invalid",
        )
        existing_receipts = {
            receipt["receipt_sha256"]
            for receipt in receipts
            if _mapping_has(
                receipt_view,
                receipt["receipt_sha256"],
                "native receipt ledger is invalid",
            )
        }
        existing_labels = {
            row["work_id"]
            for row in rows
            if _mapping_has(label_view, row["work_id"], "label ledger is invalid")
        }
        for receipt in receipts:
            store.append_chain_unique(
                receipt_path,
                receipt,
                unique_field="task_id",
                binding_field="receipt_sha256",
            )
        for row in rows:
            store.append_chain_unique(
                label_path,
                row,
                unique_field="work_id",
                binding_field="provider_receipt_sha256",
            )
        receipt_index = _receipt_index(root)
        if receipt_index is None:
            raise NativeDistillationError("native receipt ledger is invalid")
        if any(
            not _row_integrity_with_state(
                row,
                root=root,
                contract=contract,
                receipts=receipt_index,
            )
            for row in rows
        ):
            raise NativeDistillationError("native ledger receipt binding failed")
        label_head = store.chain_head(label_path)
        receipt_head = store.chain_head(receipt_path)
        counts = Counter(_text(row.get("verdict")) for row in rows)
        return {
            "status": "imported",
            "schema": NATIVE_PROFILE_SCHEMA,
            "profile": NATIVE_LUNA_SINGLE_PROFILE,
            "cohort": NATIVE_COHORT,
            "profile_contract_id": contract_id,
            "source_snapshot_sha256": manifest["source_snapshot_sha256"],
            "completion_labels_sha256": completion.get("labels_sha256", ""),
            "labels": len(rows),
            "relevant": counts["relevant"],
            "irrelevant": counts["irrelevant"],
            "uncertain": counts["uncertain"],
            "receipts": len(receipts),
            "receipts_appended": sum(
                receipt["receipt_sha256"] not in existing_receipts
                for receipt in receipts
            ),
            "labels_appended": sum(
                row["work_id"] not in existing_labels for row in rows
            ),
            # Compatibility names consumed by the distillation runner.  The
            # importer only returns zero remaining work after a complete
            # bundle has passed all frozen-cursor checks.
            "labels_written": sum(
                row["work_id"] not in existing_labels for row in rows
            ),
            # This is the durable import boundary, not another teacher run.
            "last_label_at": (
                datetime.now(UTC).isoformat().replace("+00:00", "Z")
                if any(row["work_id"] not in existing_labels for row in rows)
                else ""
            ),
            "remaining": 0,
            "label_chain_head_sha256": label_head["head_sha256"],
            "receipt_chain_head_sha256": receipt_head["head_sha256"],
            "truth_authority": "teacher_only_not_verified",
        }


__all__ = [
    "NATIVE_ASSIGNMENT_REVISION",
    "NATIVE_COHORT",
    "NATIVE_PAYLOAD_REF_SCHEMA",
    "NATIVE_PROFILE_CONTRACT_FILE",
    "NATIVE_PROFILE_CONTRACTS_DIR",
    "NATIVE_PROFILE_SCHEMA",
    "NATIVE_RECEIPT_SCHEMA",
    "NATIVE_RECEIPTS_FILE",
    "NativeDistillationError",
    "current_contract_id",
    "ensure_contract",
    "import_bundle",
    "read_contract",
    "row_integrity",
    "validate_label",
]
