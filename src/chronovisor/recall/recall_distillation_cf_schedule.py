"""Immutable, source-only scheduling for future counterfactual cohorts.

The normal/certification split pointer is deliberately not changed here.  This
module records a separate, append-only schedule whose plans copy the anchor
assignment and add only complete future source components as locked test rows.
The caller is expected to hold the repository's ``distillation-worker.lock``
while reading the catalog and calling :func:`ensure_counterfactual_schedule`.
The module takes a smaller private lock so two schedule producers cannot race.
No model call, label write, or outcome is consulted by this producer.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from chronovisor.core.canonical_json import canonical_json_sha256_strict
from chronovisor.recall import recall_distillation_store as store

CF_SCHEDULE_SCHEMA = "chronovisor.recall-distillation.v1"
SPLIT_PLAN_SCHEMA = "chronovisor.recall-distill-split-plan.v1"
SCHEDULE_KIND = "counterfactual-schedule"
PLAN_KIND = "fixed-counterfactual-scheduling-cohort"
SPLIT_REVISION = "grouped-rolling-v1"
FEATURE_REVISION = "recall-distill-text-v2"
SOURCE_KIND = "exact-exposure-page-v1"
EXPOSURE_SCHEMA = "chronovisor.recall-exact-exposure.v1"
SCHEDULE_FILE = "counterfactual-schedule.json"
SCHEDULE_LOCK = "counterfactual-schedule.lock"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class CounterfactualScheduleError(ValueError):
    """A schedule or its source inventory failed closed validation."""


def _sha(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise CounterfactualScheduleError(f"{name} is invalid")
    return value


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise CounterfactualScheduleError(f"{name} is invalid")
    return value


def _rally_id(value: object) -> str:
    return _text(value, "rally_id")


def _epoch_us(row: Mapping[str, Any]) -> int:
    value = row.get("as_of_us")
    if isinstance(value, bool):
        raise CounterfactualScheduleError("source epoch is invalid")
    if isinstance(value, int):
        if value < 0:
            raise CounterfactualScheduleError("source epoch is invalid")
        return value
    if isinstance(value, str) and value.isdecimal():
        parsed = int(value)
        if parsed >= 0:
            return parsed
    raw = row.get("as_of")
    if not isinstance(raw, str) or not raw:
        raise CounterfactualScheduleError("source epoch is invalid")
    try:
        parsed_dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise CounterfactualScheduleError("source epoch is invalid") from exc
    if parsed_dt.tzinfo is None:
        raise CounterfactualScheduleError("source epoch is invalid")
    return int(parsed_dt.astimezone(UTC).timestamp() * 1_000_000)


def _group_keys(row: Mapping[str, Any]) -> tuple[str, ...]:
    session = row.get("session_cluster_id") or row.get("session_key")
    if not isinstance(session, str) or not session:
        raise CounterfactualScheduleError("source group identity is invalid")
    keys = [f"session:{session}"]
    for field in ("task_id", "entity_id"):
        value = row.get(field)
        if value is None or value == "":
            continue
        if isinstance(value, bool) or not isinstance(value, (str, int)):
            raise CounterfactualScheduleError("source group identity is invalid")
        keys.append(f"{field}:{value}")
    return tuple(sorted(set(keys)))


def _rallies_by_id(
    rallies: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for row in rallies:
        if not isinstance(row, Mapping):
            raise CounterfactualScheduleError("rally is invalid")
        rally_id = _rally_id(row.get("rally_id"))
        if rally_id in result:
            raise CounterfactualScheduleError("rally identity is duplicated")
        result[rally_id] = row
    if not result:
        raise CounterfactualScheduleError("rally inventory is empty")
    return result


def _plan_path(root: Path, plan_id: str) -> Path:
    _sha(plan_id, "split plan id")
    return store.distillation_dir(root) / "split-plans" / f"{plan_id}.json"


def _read_plan(root: Path, plan_id: str) -> dict[str, Any]:
    try:
        artifact = store.read_sealed(
            _plan_path(root, plan_id), schema=SPLIT_PLAN_SCHEMA
        )
    except (OSError, store.DistillationStoreError) as exc:
        raise CounterfactualScheduleError("split plan is unavailable") from exc
    if (
        artifact.get("artifact_id") != plan_id
        or not isinstance(artifact.get("assignments"), Mapping)
        or any(
            value not in {"train", "validation", "test", "embargo"}
            for value in artifact["assignments"].values()
        )
    ):
        raise CounterfactualScheduleError("split plan is invalid")
    return artifact


def _current_anchor(root: Path) -> tuple[str, dict[str, Any]]:
    distillation = store.distillation_dir(root)
    certification = distillation / "certification-plan.json"
    normal = distillation / "split-plan.json"
    pointer_path = certification if certification.exists() else normal
    try:
        pointer = store.read_sealed(pointer_path, schema=store.DISTILLATION_SCHEMA)
    except (OSError, store.DistillationStoreError) as exc:
        raise CounterfactualScheduleError("anchor pointer is unavailable") from exc
    expected_kind = (
        "fixed-certification-cohort"
        if pointer_path == certification
        else "split-plan-pointer"
    )
    if pointer.get("kind") != expected_kind:
        raise CounterfactualScheduleError("anchor pointer kind is invalid")
    plan_id = _sha(pointer.get("split_plan_id"), "anchor split plan id")
    plan = _read_plan(root, plan_id)
    if pointer_path == certification:
        expected = ("fixed-certification-cohort", "fixed-certification-v1")
    else:
        expected = ("fixed-chronological-group-split", "grouped-rolling-v1")
    if (
        plan.get("kind") != expected[0]
        or plan.get("split_revision") != expected[1]
        or plan.get("feature_revision") != FEATURE_REVISION
    ):
        raise CounterfactualScheduleError("anchor plan identity is invalid")
    return plan_id, plan


def _snapshot_digest(snapshot: Mapping[str, Any]) -> str:
    digest = snapshot.get("snapshot_sha256")
    unsigned = {
        key: value for key, value in snapshot.items() if key != "snapshot_sha256"
    }
    try:
        calculated = canonical_json_sha256_strict(unsigned)
    except (TypeError, ValueError, OverflowError) as exc:
        raise CounterfactualScheduleError(
            "candidate snapshot is not canonical"
        ) from exc
    if digest is not None:
        if digest != calculated:
            raise CounterfactualScheduleError("candidate snapshot digest drift")
        return _sha(digest, "candidate snapshot digest")
    return calculated


def _selected_ids(snapshot: Mapping[str, Any]) -> tuple[str, ...]:
    selected = snapshot.get("selected")
    if selected is None:
        candidates = snapshot.get("candidates")
        if not isinstance(candidates, list):
            raise CounterfactualScheduleError(
                "selected exposure candidates are invalid"
            )
        selected = [
            item
            for item in candidates
            if isinstance(item, Mapping) and item.get("source_kind") == SOURCE_KIND
        ]
    if not isinstance(selected, Sequence) or isinstance(
        selected, (str, bytes, bytearray)
    ):
        raise CounterfactualScheduleError("selected exposure candidates are invalid")
    result: list[str] = []
    for item in selected:
        candidate = item.get("candidate_id") if isinstance(item, Mapping) else item
        if not isinstance(candidate, str) or not candidate:
            raise CounterfactualScheduleError("selected exposure candidate is invalid")
        result.append(candidate)
    if not result or len(result) != len(set(result)):
        raise CounterfactualScheduleError("selected exposure candidates are invalid")
    return tuple(result)


def _exposure_id(
    root: Path,
    rally: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    selected: Sequence[str],
) -> tuple[str, Mapping[str, Any]]:
    capture = snapshot.get("counterfactual_capture")
    capture_id = (
        capture.get("exposure_artifact_id") if isinstance(capture, Mapping) else None
    )
    supplied = (
        snapshot.get("exposure_artifact_id")
        or snapshot.get("exposure_id")
        or capture_id
    )
    receipts = rally.get("exposure_receipts")
    if not isinstance(receipts, list):
        raise CounterfactualScheduleError("exposure receipt is unavailable")
    if supplied is not None:
        supplied = _sha(supplied, "exposure artifact id")
    matches = [
        receipt
        for receipt in receipts
        if isinstance(receipt, Mapping)
        and (supplied is None or receipt.get("exposure_artifact_id") == supplied)
    ]
    if len(matches) != 1:
        raise CounterfactualScheduleError("exposure receipt is ambiguous")
    exposure_id = _sha(matches[0].get("exposure_artifact_id"), "exposure artifact id")
    if supplied is not None and supplied != exposure_id:
        raise CounterfactualScheduleError("exposure receipt conflicts")
    try:
        artifact = store.read_sealed(
            store.distillation_dir(root) / "exposures" / f"{exposure_id}.json",
            schema=EXPOSURE_SCHEMA,
        )
    except (OSError, store.DistillationStoreError) as exc:
        raise CounterfactualScheduleError("exposure artifact is unavailable") from exc
    candidate_ids = artifact.get("candidate_ids")
    if (
        artifact.get("artifact_id") != exposure_id
        or artifact.get("kind") != "exact-rendered-exposure"
        or not isinstance(candidate_ids, list)
        or not candidate_ids
        or any(not isinstance(value, str) or not value for value in candidate_ids)
        or len(candidate_ids) != len(set(candidate_ids))
        or not set(selected).issubset(set(candidate_ids))
        or not isinstance(artifact.get("candidate_refs"), list)
        or not artifact["candidate_refs"]
    ):
        raise CounterfactualScheduleError("exposure artifact is invalid")
    unsigned = {
        key: value
        for key, value in artifact.items()
        if key not in {"artifact_id", "seal_sha256"}
    }
    if canonical_json_sha256_strict(unsigned) != exposure_id:
        raise CounterfactualScheduleError("exposure artifact identity drift")
    if matches[0].get("candidate_ids") != candidate_ids:
        raise CounterfactualScheduleError("exposure receipt candidate binding drift")
    for artifact_key, rally_key in (
        ("host", "host"),
        ("session_id_sha256", "session_id_sha256"),
        ("query_semantic_sha256", "query_sha256"),
    ):
        if artifact.get(artifact_key) is not None and artifact.get(
            artifact_key
        ) != rally.get(rally_key):
            raise CounterfactualScheduleError("exposure source binding drift")
    return exposure_id, artifact


def _source_record(
    root: Path, rally: Mapping[str, Any], snapshot: Mapping[str, Any]
) -> dict[str, Any]:
    capture = snapshot.get("counterfactual_capture")
    if isinstance(capture, Mapping):
        if (
            capture.get("status") != "ready"
            or capture.get("source_kind") != SOURCE_KIND
        ):
            raise CounterfactualScheduleError("candidate snapshot source is not ready")
    elif snapshot.get("source_kind") != SOURCE_KIND:
        raise CounterfactualScheduleError("candidate snapshot source kind is invalid")
    selected = _selected_ids(snapshot)
    if (
        not isinstance(rally.get("actual_answer_refs"), list)
        or not rally["actual_answer_refs"]
    ):
        raise CounterfactualScheduleError("actual answer refs are unavailable")
    exposure_id, artifact = _exposure_id(root, rally, snapshot, selected)
    candidates = snapshot.get("candidates")
    if candidates is not None:
        if not isinstance(candidates, list):
            raise CounterfactualScheduleError(
                "candidate snapshot candidates are invalid"
            )
        selected_rows: dict[str, Mapping[str, Any]] = {}
        for item in candidates:
            candidate_id = (
                item.get("candidate_id") if isinstance(item, Mapping) else None
            )
            if (
                isinstance(candidate_id, str)
                and candidate_id in selected
                and item.get("source_kind") == SOURCE_KIND
            ):
                selected_rows[candidate_id] = item
        if set(selected_rows) != set(selected) or any(
            row.get("exposure_artifact_id") != exposure_id
            for row in selected_rows.values()
        ):
            raise CounterfactualScheduleError("candidate exposure binding drift")
    epoch = _epoch_us(rally)
    return {
        "rally_id": _rally_id(rally["rally_id"]),
        "as_of_us": epoch,
        "group_keys": list(_group_keys(rally)),
        "snapshot_sha256": _snapshot_digest(snapshot),
        "exposure_artifact_id": exposure_id,
        "selected_candidate_ids": list(selected),
        "candidate_ids_sha256": canonical_json_sha256_strict(artifact["candidate_ids"]),
        "actual_answer_refs_sha256": canonical_json_sha256_strict(
            rally["actual_answer_refs"]
        ),
    }


def _anchor_record(rally: Mapping[str, Any], assignment: str) -> dict[str, Any]:
    return {
        "rally_id": _rally_id(rally["rally_id"]),
        "as_of_us": _epoch_us(rally),
        "group_keys": list(_group_keys(rally)),
        "split": assignment,
    }


def _inventory_hash(records: Sequence[Mapping[str, Any]]) -> str:
    return canonical_json_sha256_strict(list(records))


def _epoch_iso(epoch_us: int) -> str:
    return (
        datetime.fromtimestamp(epoch_us / 1_000_000, UTC)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _validate_chronology(
    assignments: Mapping[str, str], rallies: Mapping[str, Mapping[str, Any]]
) -> None:
    epochs: dict[str, list[int]] = {
        split: [] for split in ("train", "validation", "test")
    }
    for rally_id, split in assignments.items():
        if split in epochs:
            if rally_id not in rallies:
                raise CounterfactualScheduleError(
                    "split plan rally is missing from source"
                )
            epochs[split].append(_epoch_us(rallies[rally_id]))
    if any(not values for values in epochs.values()):
        raise CounterfactualScheduleError("split chronology is incomplete")
    if not (
        max(epochs["train"]) < min(epochs["validation"])
        and max(epochs["validation"]) < min(epochs["test"])
    ):
        raise CounterfactualScheduleError("split chronology is invalid")


def _component_ids(rows: Mapping[str, Mapping[str, Any]]) -> list[set[str]]:
    ids = sorted(rows)
    parent = {rally_id: rally_id for rally_id in ids}

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    owners: dict[str, str] = {}
    for rally_id in ids:
        for key in _group_keys(rows[rally_id]):
            prior = owners.get(key)
            if prior is None:
                owners[key] = rally_id
                continue
            left, right = find(rally_id), find(prior)
            if left != right:
                parent[max(left, right)] = min(left, right)
    result: dict[str, set[str]] = {}
    for rally_id in ids:
        result.setdefault(find(rally_id), set()).add(rally_id)
    return list(result.values())


def _session_identity(row: Mapping[str, Any]) -> str:
    session = row.get("session_cluster_id") or row.get("session_key")
    if not isinstance(session, str) or not session:
        raise CounterfactualScheduleError("source group identity is invalid")
    return session


def _has_later_query(
    rally: Mapping[str, Any], rallies: Mapping[str, Mapping[str, Any]]
) -> bool:
    """Require a later same-session query before freezing answer refs.

    A current session tail can still receive more answer references.  The
    producer therefore records only rows followed by another catalog query;
    the tail remains test-assigned but unmapped until a later generation.
    """

    session = _session_identity(rally)
    epoch = _epoch_us(rally)
    rally_id = _rally_id(rally.get("rally_id"))
    for other_id, other in rallies.items():
        if other_id == rally_id or _session_identity(other) != session:
            continue
        try:
            if _epoch_us(other) > epoch:
                return True
        except CounterfactualScheduleError:
            continue
    return False


def _read_state(root: Path) -> dict[str, Any] | None:
    path = store.distillation_dir(root) / SCHEDULE_FILE
    if not path.exists():
        return None
    try:
        state = store.read_sealed(path, schema=CF_SCHEDULE_SCHEMA)
    except (OSError, store.DistillationStoreError) as exc:
        raise CounterfactualScheduleError("counterfactual schedule is invalid") from exc
    rally_plan_ids = state.get("rally_plan_ids")
    if (
        state.get("kind") != SCHEDULE_KIND
        or not isinstance(state.get("generations"), list)
        or not state["generations"]
        or not isinstance(rally_plan_ids, Mapping)
        or not isinstance(state.get("anchor_plan_id"), str)
    ):
        raise CounterfactualScheduleError("counterfactual schedule shape is invalid")
    _sha(state["anchor_plan_id"], "anchor plan id")
    cohort = _sha(state.get("model_cohort_sha256"), "schedule cohort")
    generation_rally_ids: set[str] = set()
    for generation, entry in enumerate(state["generations"]):
        if not isinstance(entry, Mapping) or entry.get("generation") != generation:
            raise CounterfactualScheduleError("counterfactual generation is invalid")
        plan_id = _sha(entry.get("plan_id"), "counterfactual plan id")
        generation_rallies = entry.get("rally_ids")
        if (
            not isinstance(generation_rallies, list)
            or any(not isinstance(rally_id, str) for rally_id in generation_rallies)
            or len(generation_rallies) != len(set(generation_rallies))
        ):
            raise CounterfactualScheduleError(
                "counterfactual generation map is invalid"
            )
        if any(
            state["rally_plan_ids"].get(rally_id) != plan_id
            for rally_id in generation_rallies
        ):
            raise CounterfactualScheduleError(
                "counterfactual generation map is invalid"
            )
        if generation_rally_ids.intersection(generation_rallies):
            raise CounterfactualScheduleError(
                "counterfactual generation map is invalid"
            )
        generation_rally_ids.update(generation_rallies)
        plan = _read_plan(root, plan_id)
        if (
            plan.get("kind") != PLAN_KIND
            or plan.get("split_revision") != SPLIT_REVISION
            or plan.get("model_cohort_sha256") != cohort
            or plan.get("generation") != generation
            or entry.get("source_inventory_sha256")
            != plan.get("source_inventory_sha256")
        ):
            raise CounterfactualScheduleError("counterfactual plan identity is invalid")
    for rally_id, plan_id in state["rally_plan_ids"].items():
        _rally_id(rally_id)
        _sha(plan_id, "counterfactual plan id")
    if generation_rally_ids != set(state["rally_plan_ids"]):
        raise CounterfactualScheduleError("counterfactual generation map is invalid")
    return state


def read_schedule(root: Path) -> dict[str, Any] | None:
    """Read the sealed schedule, or return ``None`` before the first anchor."""

    return _read_state(root.expanduser().resolve())


def read_plans(
    root: Path,
    *,
    rallies: Mapping[str, Mapping[str, Any]],
    snapshots: Mapping[str, Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    """Read all source-bound plans for the supplied catalog rows.

    The daily catalog may not yet contain every rally in a prior schedule, so
    absent rally or snapshot entries are skipped.  Once both are supplied,
    their source and exposure identity must exactly match the immutable record
    before a plan is returned.  This keeps a historical generation resumable
    while making source drift fail closed.
    """

    root = root.expanduser().resolve()
    state = _read_state(root)
    if state is None:
        return {}
    if not isinstance(rallies, Mapping) or not isinstance(snapshots, Mapping):
        raise CounterfactualScheduleError("source catalog is invalid")
    normalized_rallies: dict[str, Mapping[str, Any]] = {}
    for key, value in rallies.items():
        rally_id = _rally_id(key)
        if (
            not isinstance(value, Mapping)
            or _rally_id(value.get("rally_id")) != rally_id
        ):
            raise CounterfactualScheduleError("source catalog is invalid")
        normalized_rallies[rally_id] = value
    normalized_snapshots: dict[str, Mapping[str, Any]] = {}
    for key, value in snapshots.items():
        rally_id = _rally_id(key)
        if not isinstance(value, Mapping):
            raise CounterfactualScheduleError("source catalog is invalid")
        snapshot_rally_id = value.get("rally_id")
        if snapshot_rally_id is not None and snapshot_rally_id != rally_id:
            raise CounterfactualScheduleError("source catalog is invalid")
        normalized_snapshots[rally_id] = value
    plans = _all_plan_rows(root, state)
    result: dict[str, Mapping[str, Any]] = {}
    for rally_id, plan_id in state["rally_plan_ids"].items():
        if rally_id not in normalized_rallies or rally_id not in normalized_snapshots:
            continue
        rally = normalized_rallies[rally_id]
        snapshot = normalized_snapshots[rally_id]
        record = next(
            (
                item
                for item in plans[str(plan_id)].get("source_inventory", [])
                if isinstance(item, Mapping) and item.get("rally_id") == rally_id
            ),
            None,
        )
        if record is None or _source_record(root, rally, snapshot) != record:
            raise CounterfactualScheduleError("mapped source inventory drift")
        result[str(rally_id)] = plans[str(plan_id)]
    return result


def _all_plan_rows(root: Path, state: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    plans: dict[str, dict[str, Any]] = {}
    for entry in state["generations"]:
        plan_id = str(entry["plan_id"])
        plan = _read_plan(root, plan_id)
        source_inventory = plan.get("source_inventory")
        if not isinstance(source_inventory, list) or _inventory_hash(
            source_inventory
        ) != plan.get("source_inventory_sha256"):
            raise CounterfactualScheduleError("source inventory hash drift")
        source_ids = [
            _rally_id(record.get("rally_id"))
            for record in source_inventory
            if isinstance(record, Mapping)
        ]
        if len(source_ids) != len(source_inventory) or len(source_ids) != len(
            set(source_ids)
        ):
            raise CounterfactualScheduleError("source inventory is invalid")
        plans[plan_id] = plan
        for rally_id in entry["rally_ids"]:
            if str(state["rally_plan_ids"].get(rally_id) or "") != plan_id:
                raise CounterfactualScheduleError("schedule rally binding drift")
    known = set(plans)
    if any(str(plan_id) not in known for plan_id in state["rally_plan_ids"].values()):
        raise CounterfactualScheduleError("schedule rally binding drift")
    return plans


def _validate_anchor_source(
    state: Mapping[str, Any],
    anchor_plan: Mapping[str, Any],
    rallies: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Mapping[str, Any]], int]:
    if anchor_plan.get("anchor_plan_id") != state.get(
        "anchor_plan_id"
    ) or anchor_plan.get("anchor_inventory_sha256") != state.get(
        "anchor_inventory_sha256"
    ):
        raise CounterfactualScheduleError("anchor plan identity drift")
    anchor_records = anchor_plan.get("anchor_inventory")
    if not isinstance(anchor_records, list) or not anchor_records:
        raise CounterfactualScheduleError("anchor inventory is invalid")
    if _inventory_hash(anchor_records) != anchor_plan.get("anchor_inventory_sha256"):
        raise CounterfactualScheduleError("anchor inventory hash drift")
    by_id: dict[str, Mapping[str, Any]] = {}
    for record in anchor_records:
        if not isinstance(record, Mapping):
            raise CounterfactualScheduleError("anchor inventory is invalid")
        rally_id = _rally_id(record.get("rally_id"))
        if rally_id in by_id or rally_id not in rallies:
            raise CounterfactualScheduleError("anchor source is missing")
        current = rallies[rally_id]
        if _epoch_us(current) != record.get("as_of_us") or list(
            _group_keys(current)
        ) != record.get("group_keys"):
            raise CounterfactualScheduleError("anchor source identity drift")
        by_id[rally_id] = current
    cutoff = max(int(record["as_of_us"]) for record in anchor_records)
    return by_id, cutoff


def _validate_mapped_sources(
    root: Path,
    state: Mapping[str, Any],
    rallies: Mapping[str, Mapping[str, Any]],
    snapshots: Mapping[str, Mapping[str, Any]],
    plans: Mapping[str, Mapping[str, Any]],
) -> None:
    for rally_id, plan_id in state["rally_plan_ids"].items():
        plan = plans.get(str(plan_id))
        if plan is None:
            raise CounterfactualScheduleError("mapped plan is unavailable")
        records = {
            str(record.get("rally_id")): record
            for record in plan.get("source_inventory", [])
            if isinstance(record, Mapping)
        }
        record = records.get(str(rally_id))
        if record is None or rally_id not in rallies or rally_id not in snapshots:
            raise CounterfactualScheduleError("mapped source is missing")
        current = _source_record(root, rallies[rally_id], snapshots[rally_id])
        if current != record:
            raise CounterfactualScheduleError("mapped source inventory drift")


def _all_result(
    state: Mapping[str, Any], plans: Mapping[str, Mapping[str, Any]]
) -> dict[str, Mapping[str, Any]]:
    return {
        str(rally_id): plans[str(plan_id)]
        for rally_id, plan_id in state["rally_plan_ids"].items()
        if str(plan_id) in plans
    }


def ensure_counterfactual_schedule(
    root: Path,
    rallies: Sequence[Mapping[str, Any]],
    snapshots: Mapping[str, Mapping[str, Any]],
    *,
    model_cohort_sha256: str,
    raw_watermark: str,
) -> dict[str, Mapping[str, Any]]:
    """Create or extend a source-only future CF schedule.

    The caller owns the repository distillation-worker lease.  This function
    takes only its own schedule lock, writes content-addressed plans, and then
    atomically seals the independent schedule pointer.  It never reads model
    outcomes and never changes the normal/certification split pointers.
    """

    root = root.expanduser().resolve()
    cohort = _sha(model_cohort_sha256, "model cohort")
    watermark = _sha(raw_watermark, "raw watermark")
    rally_by_id = _rallies_by_id(rallies)
    if not isinstance(snapshots, Mapping):
        raise CounterfactualScheduleError("candidate snapshot inventory is invalid")
    normalized_snapshots: dict[str, Mapping[str, Any]] = {}
    for key, value in snapshots.items():
        rally_id = _rally_id(key)
        if not isinstance(value, Mapping):
            raise CounterfactualScheduleError("candidate snapshot is invalid")
        snapshot_rally_id = value.get("rally_id")
        if snapshot_rally_id is not None and snapshot_rally_id != rally_id:
            raise CounterfactualScheduleError("candidate snapshot identity drift")
        normalized_snapshots[rally_id] = value
    snapshots = normalized_snapshots
    with store._locked(store.distillation_dir(root) / SCHEDULE_LOCK):
        state = _read_state(root)
        if state is None:
            anchor_id, anchor = _current_anchor(root)
            if anchor.get("model_cohort_sha256") != cohort:
                raise CounterfactualScheduleError("anchor cohort mismatch")
            anchor_assignments = {
                _rally_id(key): _text(value, "anchor assignment")
                for key, value in anchor["assignments"].items()
            }
            if not set(anchor_assignments).issubset(rally_by_id):
                raise CounterfactualScheduleError("anchor rally inventory is missing")
            anchor_records = [
                _anchor_record(
                    rally_by_id[rally_id], anchor_assignments.get(rally_id, "embargo")
                )
                for rally_id in sorted(rally_by_id)
            ]
            assignments = {
                rally_id: anchor_assignments.get(rally_id, "embargo")
                for rally_id in sorted(rally_by_id)
            }
            _validate_chronology(assignments, rally_by_id)
            anchor_hash = _inventory_hash(anchor_records)
            plan_payload = {
                "kind": PLAN_KIND,
                "raw_watermark": watermark,
                "feature_revision": FEATURE_REVISION,
                "model_cohort_sha256": cohort,
                "split_revision": SPLIT_REVISION,
                "assignments": assignments,
                "age_boundary_utc": _epoch_iso(
                    max(int(record["as_of_us"]) for record in anchor_records)
                ),
                "age_bands": {
                    rally_id: "locked-test" if split == "test" else "old-history"
                    for rally_id, split in assignments.items()
                },
                "anchor_plan_id": anchor_id,
                "anchor_inventory_sha256": anchor_hash,
                "anchor_inventory": anchor_records,
                "source_inventory_sha256": _inventory_hash([]),
                "source_ref_sha256": _inventory_hash([]),
                "source_inventory": [],
                "eligible_cf_rally_ids": [],
                "generation": 0,
            }
            plan_id, _path, plan = store.write_immutable(
                store.distillation_dir(root) / "split-plans",
                plan_payload,
                schema=SPLIT_PLAN_SCHEMA,
            )
            state = {
                "kind": SCHEDULE_KIND,
                "anchor_plan_id": anchor_id,
                "anchor_inventory_sha256": anchor_hash,
                "anchor_cutoff_us": max(
                    int(record["as_of_us"]) for record in anchor_records
                ),
                "model_cohort_sha256": cohort,
                "generations": [
                    {
                        "generation": 0,
                        "plan_id": plan_id,
                        "rally_ids": [],
                        "source_inventory_sha256": _inventory_hash([]),
                    }
                ],
                "rally_plan_ids": {},
            }
            store.write_sealed_state(
                store.distillation_dir(root) / SCHEDULE_FILE, state
            )
            return {}

        if state.get("model_cohort_sha256") != cohort:
            raise CounterfactualScheduleError("schedule cohort mismatch")
        plans = _all_plan_rows(root, state)
        anchor_plan = _read_plan(root, str(state["generations"][0]["plan_id"]))
        anchor_by_id, cutoff = _validate_anchor_source(state, anchor_plan, rally_by_id)
        if int(state.get("anchor_cutoff_us") or -1) != cutoff:
            raise CounterfactualScheduleError("anchor cutoff drift")
        _validate_mapped_sources(root, state, rally_by_id, snapshots, plans)
        mapped_ids = {str(rally_id) for rally_id in state["rally_plan_ids"]}
        current_ids = set(rally_by_id)
        new_ids = current_ids - set(anchor_by_id) - mapped_ids
        if not new_ids:
            return _all_result(state, plans)
        new_rows = {rally_id: rally_by_id[rally_id] for rally_id in sorted(new_ids)}
        anchor_keys = {
            key for rally in anchor_by_id.values() for key in _group_keys(rally)
        }
        eligible: list[dict[str, Any]] = []
        scheduled_ids: set[str] = set()
        for component in _component_ids(new_rows):
            component_keys = {
                key for rally_id in component for key in _group_keys(new_rows[rally_id])
            }
            if component_keys & anchor_keys:
                continue
            if any(_epoch_us(new_rows[rally_id]) <= cutoff for rally_id in component):
                continue
            records: list[dict[str, Any]] = []
            for rally_id in sorted(component):
                row = new_rows[rally_id]
                if rally_id not in snapshots:
                    continue
                if not _has_later_query(row, rally_by_id):
                    continue
                try:
                    records.append(_source_record(root, row, snapshots[rally_id]))
                except CounterfactualScheduleError:
                    continue
            if records:
                eligible.extend(records)
                scheduled_ids.update(component)
        if not eligible:
            return _all_result(state, plans)
        eligible.sort(
            key=lambda record: (int(record["as_of_us"]), str(record["rally_id"]))
        )
        assignments = {
            rally_id: anchor_plan["assignments"].get(rally_id, "embargo")
            for rally_id in sorted(current_ids)
        }
        for rally_id in scheduled_ids:
            assignments[rally_id] = "test"
        _validate_chronology(assignments, rally_by_id)
        generation = len(state["generations"])
        source_hash = _inventory_hash(eligible)
        ref_hash = canonical_json_sha256_strict(
            [
                {
                    "rally_id": record["rally_id"],
                    "exposure_artifact_id": record["exposure_artifact_id"],
                    "selected_candidate_ids": record["selected_candidate_ids"],
                }
                for record in eligible
            ]
        )
        plan_payload = {
            "kind": PLAN_KIND,
            "raw_watermark": watermark,
            "feature_revision": FEATURE_REVISION,
            "model_cohort_sha256": cohort,
            "split_revision": SPLIT_REVISION,
            "assignments": assignments,
            "age_boundary_utc": _epoch_iso(
                max(int(record["as_of_us"]) for record in eligible)
            ),
            "age_bands": {
                rally_id: "locked-test" if split == "test" else "old-history"
                for rally_id, split in assignments.items()
            },
            "anchor_plan_id": str(state["anchor_plan_id"]),
            "anchor_inventory_sha256": str(state["anchor_inventory_sha256"]),
            "source_inventory_sha256": source_hash,
            "source_ref_sha256": ref_hash,
            "source_inventory": eligible,
            "eligible_cf_rally_ids": [record["rally_id"] for record in eligible],
            "generation": generation,
        }
        plan_id, _path, plan = store.write_immutable(
            store.distillation_dir(root) / "split-plans",
            plan_payload,
            schema=SPLIT_PLAN_SCHEMA,
        )
        state = {
            **{
                key: value
                for key, value in state.items()
                if key not in {"schema", "namespace", "seal_sha256"}
            },
            "generations": [
                *state["generations"],
                {
                    "generation": generation,
                    "plan_id": plan_id,
                    "rally_ids": [record["rally_id"] for record in eligible],
                    "source_inventory_sha256": source_hash,
                },
            ],
            "rally_plan_ids": {
                **dict(state["rally_plan_ids"]),
                **{record["rally_id"]: plan_id for record in eligible},
            },
        }
        store.write_sealed_state(store.distillation_dir(root) / SCHEDULE_FILE, state)
        plans[plan_id] = plan
        return _all_result(state, plans)


__all__ = [
    "CounterfactualScheduleError",
    "ensure_counterfactual_schedule",
    "read_plans",
    "read_schedule",
]
