#!/usr/bin/env python3
"""Build a fresh, fixed native-Luna certification cohort.

The normal daily producer is intentionally a growth receipt: it preserves all
historical assignments and places new source in embargo.  Certification needs
a different boundary.  This module builds an immutable split-plan artifact
from source metadata, binds a new task identity to that plan, and delegates
source normalization/bundle writing to the existing daily and native-Luna
APIs.  It never reads or sorts by teacher verdicts and never changes the
current split-plan pointer.

The module is safe to import in an unattended status/dry-run process.  Source
text is only handled by ``recall_luna_daily._build_rows`` when a caller asks to
prepare a bundle; all durable plan fields are identifiers, counts, and hashes.
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# ruff: noqa: E402, I001
from chronovisor.core.canonical_json import canonical_json_sha256_strict
from chronovisor.recall import recall_distillation as distill
from chronovisor.recall import recall_distillation_store as store
from chronovisor.recall.recall_distillation_luna import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_SHARD_COUNT,
    DEFAULT_TEACHER_PROFILE,
    create_bundle,
)
from scripts import recall_luna_daily as daily


CERTIFICATION_REVISION = "native-luna-certification-v1"
REPEAT_PAIR_ID_REVISION = "chronovisor.recall-luna-certification-repeat.v1"
NATIVE_COHORT = "native-luna-backfill-v1"
SOURCE_SPLITS = frozenset({"train", "validation", "test"})
ALL_SPLITS = frozenset({"train", "validation", "test", "embargo"})
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
MAX_CERTIFICATION_ROWS = 10_000
# The native gate requires at least 500 owner labels and 100 complete blind
# pairs.  Freeze a little headroom above those floors while keeping the source
# bundle small enough for unattended Luna requests.  These limits are applied
# before any worker call and are independent of verdicts.
MAX_BASE_ROWS = 2_000
MAX_PROBE_PAIRS = 200
MAX_RALLIES_BY_SPLIT = {"train": 350, "validation": 150, "test": 500}
MAX_PROBE_RALLIES = 100
MAX_BASE_CANDIDATES_PER_RALLY = 2
MAX_PROBE_PAIRS_PER_RALLY = 2
ROW_SAMPLING_REVISION = "chronovisor.recall-luna-certification-row-sampling.v1"
MIN_CERTIFICATION_RALLIES = 1_000
MIN_CERTIFICATION_DAYS = 30
MIN_CERTIFICATION_WINDOWS = 3


def _fixed_row_sampling() -> dict[str, Any]:
    return {
        "revision": ROW_SAMPLING_REVISION,
        "base_rows": MAX_BASE_ROWS,
        "probe_pairs": MAX_PROBE_PAIRS,
        "rally_caps": dict(MAX_RALLIES_BY_SPLIT),
        "base_candidates_per_rally": MAX_BASE_CANDIDATES_PER_RALLY,
        "probe_rallies": MAX_PROBE_RALLIES,
        "probe_pairs_per_rally": MAX_PROBE_PAIRS_PER_RALLY,
        "selection": "metadata-chronological-rally-candidate-v1",
    }


class CertificationError(RuntimeError):
    """A certification cohort cannot be frozen without relaxing a gate."""


def _identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise CertificationError(f"{field} is missing")
    return value


def _source_epoch(row: Mapping[str, Any]) -> float:
    value = row.get("as_of_us")
    if isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    text = str(row.get("as_of") or "")
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise CertificationError("source timestamp is invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise CertificationError("source timestamp is invalid")
    return parsed.astimezone(UTC).timestamp() * 1_000_000


def _group_key(row: Mapping[str, Any]) -> str:
    # Keep this ordering identical to ``grouped_rolling_split``.  ``group_id``
    # is a derived bundle field and must not silently change the atomic source
    # component when callers pass already-normalized rows.
    for field in ("session_cluster_id", "session_key", "group_id"):
        value = row.get(field)
        if value:
            return str(value)
    return str(row.get("rally_id") or "")


def _component_groups(
    rallies: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Union the same metadata keys as ``grouped_rolling_split``."""
    rows = [dict(row) for row in rallies]
    if not rows:
        return []
    parent = list(range(len(rows)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        a, b = find(left), find(right)
        if a != b:
            parent[max(a, b)] = min(a, b)

    owners: dict[str, int] = {}
    for index, row in enumerate(rows):
        keys = [f"session:{_group_key(row)}"] + [
            f"{field}:{row[field]}"
            for field in ("task_id", "entity_id")
            if row.get(field)
        ]
        for key in keys:
            previous = owners.setdefault(key, index)
            if previous != index:
                union(index, previous)

    components: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for index, row in enumerate(rows):
        components[find(index)].append(row)

    result: list[dict[str, Any]] = []
    for items in components.values():
        normalized = sorted(items, key=lambda row: (_source_epoch(row), str(row.get("rally_id") or "")))
        ids = [_identifier(row.get("rally_id"), "rally_id") for row in normalized]
        result.append(
            {
                "component_id": min(ids),
                "rally_ids": ids,
                "rows": normalized,
                "min_epoch": min(_source_epoch(row) for row in normalized),
                "max_epoch": max(_source_epoch(row) for row in normalized),
            }
        )
    return sorted(
        result,
        key=lambda item: (item["max_epoch"], item["min_epoch"], item["component_id"]),
    )


def _assignment_map(
    current_plan: Mapping[str, Any] | Mapping[str, str],
) -> dict[str, str]:
    values = current_plan.get("assignments", current_plan)
    if not isinstance(values, Mapping):
        raise CertificationError("current split assignments are missing")
    result = {str(key): str(value) for key, value in values.items()}
    if any(value not in ALL_SPLITS for value in result.values()):
        raise CertificationError("current split assignment is invalid")
    return result


def _mandatory_rows(
    mandatory_pairs: Iterable[Mapping[str, Any]] | None,
) -> list[dict[str, str]]:
    """Normalize preregistered blind-pair metadata without verdicts."""
    normalized: list[dict[str, str]] = []
    for item in mandatory_pairs or ():
        pair_id = _identifier(item.get("repeat_pair_id"), "repeat_pair_id")
        order = _identifier(item.get("blind_order"), "blind_order")
        if order not in {"a_first", "b_first"}:
            raise CertificationError("mandatory blind order is invalid")
        row = {
            "repeat_pair_id": pair_id,
            "blind_order": order,
            "rally_id": _identifier(item.get("rally_id"), "rally_id"),
            "candidate_id": _identifier(item.get("candidate_id"), "candidate_id"),
        }
        if item.get("input_sha256") is not None:
            digest = _identifier(item["input_sha256"], "input_sha256")
            if not SHA256.fullmatch(digest):
                raise CertificationError("mandatory source input hash is invalid")
            row["input_sha256"] = digest
        normalized.append(row)
    by_pair: dict[str, set[str]] = defaultdict(set)
    for item in normalized:
        pair = item["repeat_pair_id"]
        if item["blind_order"] in by_pair[pair]:
            raise CertificationError("mandatory blind pair has duplicate order")
        by_pair[pair].add(item["blind_order"])
    if any(orders != {"a_first", "b_first"} for orders in by_pair.values()):
        raise CertificationError("mandatory blind pair must contain both orders")
    return sorted(normalized, key=lambda item: tuple(item.get(field, "") for field in (
        "repeat_pair_id", "blind_order", "rally_id", "candidate_id", "input_sha256"
    )))


def preregister_mandatory_pairs(
    source_rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, str]]:
    """Create immutable plan metadata from exact old source rows."""

    normalized: list[dict[str, Any]] = []
    for row in source_rows:
        if row.get("input") is None:
            raise CertificationError("mandatory source input is missing")
        input_sha256 = canonical_json_sha256_strict(row["input"])
        supplied = row.get("input_sha256")
        if supplied is not None and supplied != input_sha256:
            raise CertificationError("mandatory source input hash mismatch")
        normalized.append({
            "repeat_pair_id": row.get("repeat_pair_id"),
            "blind_order": row.get("blind_order"),
            "rally_id": row.get("rally_id"),
            "candidate_id": row.get("candidate_id"),
            "input_sha256": input_sha256,
        })
    return _mandatory_rows(normalized)


def _intervals_strict(
    components: Sequence[Mapping[str, Any]],
    assignment: Mapping[str, str],
) -> bool:
    by_split: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for component in components:
        splits = {assignment.get(rally_id) for rally_id in component["rally_ids"]}
        split = next(iter(splits), None)
        if split in SOURCE_SPLITS and len(splits) == 1:
            by_split[str(split)].append(component)
    if any(not by_split[split] for split in SOURCE_SPLITS):
        return False
    return (
        max(item["max_epoch"] for item in by_split["train"])
        < min(item["min_epoch"] for item in by_split["validation"])
        and max(item["max_epoch"] for item in by_split["validation"])
        < min(item["min_epoch"] for item in by_split["test"])
    )


def _window_count(epochs: Sequence[float]) -> int:
    """Count deterministic seven-day source windows without using verdicts."""

    window = 7 * 86_400_000_000
    return len({int(epoch // window) for epoch in epochs})


def plan_certification_assignments(
    rallies: Sequence[Mapping[str, Any]],
    current_plan: Mapping[str, Any] | Mapping[str, str],
    *,
    mandatory_pairs: Iterable[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Choose a metadata-only fixed plan with closed component intervals.

    Existing test-only components may be reused as *new test evidence*; their
    old rows remain under their old plan and are never rebound.  Existing
    train/validation/embargo components cannot become new test evidence.
    Components whose historical assignment is mixed are quarantined rather
    than repaired by reassignment.
    """

    rows = [dict(row) for row in rallies]
    row_ids = [str(row.get("rally_id") or "") for row in rows]
    if any(not rally_id for rally_id in row_ids) or len(row_ids) != len(set(row_ids)):
        raise CertificationError("source inventory rally identity is invalid")
    assignment = _assignment_map(current_plan)
    mandatory = _mandatory_rows(mandatory_pairs)
    mandatory_rallies = {item["rally_id"] for item in mandatory}
    components = _component_groups(rows)
    if not components:
        raise CertificationError("source inventory is empty")

    selected: dict[str, str] = {}
    quarantined: dict[str, str] = {}
    old_test_components: list[dict[str, Any]] = []
    mutable_components: list[dict[str, Any]] = []
    mandatory_components: set[str] = set()
    historical: dict[str, set[str]] = {}
    inventory_ids = {
        str(row.get("rally_id")) for row in rows if row.get("rally_id") is not None
    }

    for component in components:
        component_id = component["component_id"]
        old_splits = {
            assignment[rally_id]
            for rally_id in component["rally_ids"]
            if rally_id in assignment
        }
        historical[component_id] = old_splits
        if any(rally_id in mandatory_rallies for rally_id in component["rally_ids"]):
            mandatory_components.add(component_id)
        if len(old_splits) > 1:
            quarantined[component_id] = "historical_cross_split"
            continue
        if old_splits == {"test"}:
            old_test_components.append(component)
        elif old_splits == {"embargo"}:
            quarantined[component_id] = "historical_embargo"
        else:
            mutable_components.append(component)

    missing_mandatory = mandatory_rallies - inventory_ids
    if missing_mandatory:
        raise CertificationError("mandatory blind source is outside the inventory")
    if mandatory_components.intersection(quarantined):
        raise CertificationError("mandatory blind source crosses a historical boundary")
    mandatory_historical_splits = {
        assignment[rally_id]
        for rally_id in mandatory_rallies
        if rally_id in assignment
    }
    if mandatory_rallies and mandatory_historical_splits != {"test"}:
        raise CertificationError("mandatory blind source is not an old test source")

    # Historical train/validation components retain their old split.  The
    # current inventory is the fixed historical cohort; source without an
    # assignment stays quarantined until a separately reviewed plan is made.
    fixed_train = [
        component
        for component in mutable_components
        if historical[component["component_id"]] == {"train"}
    ]
    fixed_validation = [
        component
        for component in mutable_components
        if historical[component["component_id"]] == {"validation"}
    ]
    fixed_train.sort(key=lambda item: (item["max_epoch"], item["component_id"]))
    fixed_validation.sort(key=lambda item: (item["max_epoch"], item["component_id"]))

    # Mandatory old-test pairs are preregistered before selecting extra test
    # components.  No verdict, completion status, or class is consulted.
    mandatory_test_ids = {
        component["component_id"]
        for component in old_test_components
        if component["component_id"] in mandatory_components
    }
    old_test_components.sort(
        key=lambda item: (item["min_epoch"], item["max_epoch"], item["component_id"])
    )

    # Start with all eligible groups and then close the temporal intervals. A
    # group is either selected whole or quarantined; no individual rally is
    # moved across a boundary.
    train = fixed_train[: max(1, len(fixed_train) // 2)]
    train_end = max((item["max_epoch"] for item in train), default=None)
    if train_end is None:
        raise CertificationError("strict_temporal_windows_unavailable")

    validation_candidates = [
        component for component in fixed_validation if component["min_epoch"] > train_end
    ]
    validation_candidates.sort(
        key=lambda item: (item["max_epoch"], item["component_id"])
    )
    validation = validation_candidates[: max(1, len(validation_candidates) // 2)]
    validation_end = max((item["max_epoch"] for item in validation), default=None)
    if validation_end is None:
        raise CertificationError("strict_temporal_windows_unavailable")

    test_pool = {
        component["component_id"]: component
        for component in [
            *old_test_components,
            *[
                component
                for component in mutable_components
                if component["component_id"] in mandatory_components
            ],
        ]
    }
    train_ids = {item["component_id"] for item in train}
    validation_ids = {item["component_id"] for item in validation}
    test_candidates = [
        component for component in test_pool.values()
        if component["min_epoch"] > validation_end
        and component["component_id"] not in train_ids | validation_ids
    ]
    test_candidates.sort(
        key=lambda item: (
            item["component_id"] not in mandatory_test_ids,
            item["min_epoch"],
            item["component_id"],
        )
    )
    test = test_candidates
    if not test:
        raise CertificationError("strict_temporal_windows_unavailable")

    for split, selected_components_for_split in (
        ("train", train), ("validation", validation), ("test", test)
    ):
        for component in selected_components_for_split:
            selected.update(dict.fromkeys(component["rally_ids"], split))

    selected_components = train_ids | validation_ids | {
        item["component_id"] for item in test
    }
    if any(selected.get(rally_id) != "test" for rally_id in mandatory_rallies):
        raise CertificationError("mandatory blind source is outside the new test window")
    for component in components:
        if component["component_id"] not in selected_components:
            quarantined.setdefault(component["component_id"], "outside_strict_windows")

    chosen_rows = [row for row in rows if str(row.get("rally_id")) in selected]
    if len(chosen_rows) < MIN_CERTIFICATION_RALLIES:
        raise CertificationError("insufficient_new_inventory")
    epochs = [_source_epoch(row) for row in chosen_rows]
    span_days = (max(epochs) - min(epochs)) / 86_400_000_000
    if span_days < MIN_CERTIFICATION_DAYS:
        raise CertificationError("insufficient_temporal_span")
    window_count = _window_count(epochs)
    if window_count < MIN_CERTIFICATION_WINDOWS:
        raise CertificationError("insufficient_temporal_windows")
    if not _intervals_strict(components, selected):
        raise CertificationError("strict_temporal_windows_unavailable")

    return {
        "assignments": dict(sorted(selected.items())),
        "quarantined_groups": dict(sorted(quarantined.items())),
        "group_count": len(selected_components),
        "rally_count": len(chosen_rows),
        "span_days": int(span_days),
        "window_count": window_count,
        "row_sampling": _fixed_row_sampling(),
        "mandatory_repeat_pairs": mandatory,
        "protected_old_test_groups": sorted(
            component["component_id"] for component in old_test_components
        ),
        "historical_plan_id": str(current_plan.get("artifact_id") or "")
        if isinstance(current_plan, Mapping)
        else "",
    }


def write_certification_plan(
    root: Path,
    plan: Mapping[str, Any],
    *,
    rallies: Sequence[Mapping[str, Any]] | None = None,
    raw_watermark: str,
    model_cohort_sha256: str,
    candidate_snapshot_sha256: str = "",
    reason: str = "fresh-native-certification-cohort",
) -> dict[str, Any]:
    """Write a new immutable split-plan artifact without changing its pointer."""

    root = Path(root).expanduser().resolve()
    if not SHA256.fullmatch(str(model_cohort_sha256)):
        raise CertificationError("model cohort identity is invalid")
    if candidate_snapshot_sha256 and not SHA256.fullmatch(str(candidate_snapshot_sha256)):
        raise CertificationError("candidate snapshot identity is invalid")
    assignments = _assignment_map(plan)
    mandatory = _mandatory_rows(plan.get("mandatory_repeat_pairs"))
    if any(not item.get("input_sha256") for item in mandatory):
        raise CertificationError("mandatory source input hash is missing")
    if plan.get("row_sampling") not in (None, _fixed_row_sampling()):
        raise CertificationError("row sampling policy is not fixed")
    source_rows = [
        dict(rally)
        for rally in (rallies or ())
        if str(rally.get("rally_id") or "") in assignments
    ]
    source_ids = [str(row.get("rally_id") or "") for row in source_rows]
    if len(source_ids) != len(set(source_ids)) or set(source_ids) != set(assignments):
        raise CertificationError("certification source inventory is incomplete")
    age_boundary = distill._source_age_boundary(source_rows)  # type: ignore[attr-defined]
    age_bands = distill._source_age_bands(  # type: ignore[attr-defined]
        source_rows, assignments=assignments
    )
    payload = {
        "kind": "fixed-certification-cohort",
        "raw_watermark": str(raw_watermark),
        "feature_revision": distill.TEXT_FEATURE_REVISION,
        "model_cohort_sha256": str(model_cohort_sha256),
        "split_revision": "fixed-certification-v1",
        "assignments": assignments,
        "age_boundary_utc": age_boundary,
        "age_bands": age_bands or {rally_id: "old-history" for rally_id in assignments},
        "certification_revision": CERTIFICATION_REVISION,
        "certification_reason": str(reason),
        "candidate_snapshot_sha256": str(candidate_snapshot_sha256),
        "historical_plan_id": str(plan.get("historical_plan_id") or ""),
        "protected_old_test_groups": list(plan.get("protected_old_test_groups") or []),
        "quarantined_groups": dict(plan.get("quarantined_groups") or {}),
        "mandatory_repeat_pairs": mandatory,
        "mandatory_repeat_pair_revision": REPEAT_PAIR_ID_REVISION,
        "rally_count": int(plan.get("rally_count") or len(assignments)),
        "group_count": int(plan.get("group_count") or 0),
        "span_days": int(plan.get("span_days") or 0),
        "window_count": int(plan.get("window_count") or 0),
        "row_sampling": _fixed_row_sampling(),
    }
    plan_id, _path, artifact = store.write_immutable(
        store.distillation_dir(root) / "split-plans",
        payload,
        schema=distill.SPLIT_PLAN_SCHEMA,
    )
    return artifact | {"artifact_id": plan_id}


def certification_task_id(row: Mapping[str, Any], plan_id: str) -> str:
    """Derive a new plan-bound task ID while leaving request bytes untouched."""

    try:
        _identifier(row.get("rally_id"), "rally_id")
        _identifier(row.get("candidate_id"), "candidate_id")
        return distill.certification_task_id(row, plan_id)
    except (CertificationError, distill.DistillationError, KeyError, TypeError) as exc:
        raise CertificationError("certification task source is invalid") from exc


def certification_repeat_pair_id(source_pair_id: str, plan_id: str) -> str:
    """Derive a repeat identity that cannot collide with the old pair."""

    if not SHA256.fullmatch(str(plan_id)):
        raise CertificationError("certification plan id is invalid")
    source_pair_id = _identifier(source_pair_id, "source_repeat_pair_id")
    return canonical_json_sha256_strict(
        {
            "schema": REPEAT_PAIR_ID_REVISION,
            "source_repeat_pair_id": source_pair_id,
            "split_plan_id": plan_id,
        }
    )


def bind_rows_to_plan(
    rows: Sequence[Mapping[str, Any]],
    plan: Mapping[str, Any],
    plan_id: str,
) -> list[dict[str, Any]]:
    """Bind source attempts to a fixed plan and reject embargo/leakage rows."""

    assignments = _assignment_map(plan)
    mandatory = _mandatory_rows(plan.get("mandatory_repeat_pairs"))
    mandatory_by_key = {
        (
            item["repeat_pair_id"],
            item["blind_order"],
            item["rally_id"],
            item["candidate_id"],
        ): item
        for item in mandatory
    }
    mandatory_by_bound_key = {
        (
            certification_repeat_pair_id(item["repeat_pair_id"], plan_id),
            item["blind_order"],
            item["rally_id"],
            item["candidate_id"],
        ): item
        for item in mandatory
    }
    bound: list[dict[str, Any]] = []
    for row in rows:
        rally_id = _identifier(row.get("rally_id"), "rally_id")
        expected_split = assignments.get(rally_id)
        if expected_split not in SOURCE_SPLITS:
            raise CertificationError("certification source contains an unselected rally")
        if str(row.get("split") or "") != expected_split:
            raise CertificationError("certification source split assignment mismatch")
        copy = dict(row)
        source_pair_key = (
            str(copy.get("repeat_pair_id") or ""),
            str(copy.get("blind_order") or ""),
            rally_id,
            str(copy.get("candidate_id") or ""),
        )
        expected = mandatory_by_key.get(source_pair_key) or mandatory_by_bound_key.get(
            source_pair_key
        )
        if expected is not None and expected.get("input_sha256"):
            if copy.get("input") is None:
                raise CertificationError("mandatory source input is missing")
            if canonical_json_sha256_strict(copy["input"]) != expected["input_sha256"]:
                raise CertificationError("mandatory source input hash mismatch")
        already_bound = (
            bool(copy.get("task_id"))
            and str(copy.get("task_id")) == certification_task_id(copy, plan_id)
        )
        if source_pair_key[0] and not already_bound:
            # Every blind pair receives a plan-bound identity.  Restricting
            # this to the six mandatory pairs would let a newly generated
            # probe collide with an old pair in downstream projections.
            copy["repeat_pair_id"] = certification_repeat_pair_id(
                source_pair_key[0], plan_id
            )
            assignment = copy.get("assignment")
            if isinstance(assignment, Mapping):
                assignment_copy = dict(assignment)
                assignment_copy["repeat_pair_id"] = copy["repeat_pair_id"]
                copy["assignment"] = assignment_copy
        assignment = copy.get("assignment")
        if (
            source_pair_key[0]
            and isinstance(assignment, Mapping)
            and assignment.get("repeat_pair_id") != copy.get("repeat_pair_id")
        ):
            raise CertificationError("certification assignment repeat identity mismatch")
        copy.pop("source_record_sha256", None)
        copy["task_id"] = certification_task_id(copy, plan_id)
        bound.append(copy)

    ids = [str(row["task_id"]) for row in bound]
    if len(ids) != len(set(ids)):
        raise CertificationError("certification task identity collision")
    required = {
        (
            certification_repeat_pair_id(item["repeat_pair_id"], plan_id),
            item["blind_order"],
            item["rally_id"],
            item["candidate_id"],
        )
        for item in mandatory
    }
    present = {
        (
            str(row.get("repeat_pair_id") or ""),
            str(row.get("blind_order") or ""),
            str(row.get("rally_id") or ""),
            str(row.get("candidate_id") or ""),
        )
        for row in bound
    }
    if not required.issubset(present):
        raise CertificationError("mandatory blind source is missing from bundle")
    return bound


def _row_sort_key(row: Mapping[str, Any]) -> tuple[int, str, str, str, str]:
    """Return a verdict-independent source ordering."""

    as_of = row.get("as_of_us")
    epoch = int(as_of) if isinstance(as_of, int) and not isinstance(as_of, bool) else 0
    return (
        epoch,
        str(row.get("rally_id") or ""),
        str(row.get("candidate_id") or ""),
        str(row.get("repeat_pair_id") or ""),
        str(row.get("blind_order") or ""),
    )


def _sample_source_rows(
    rows: Sequence[Mapping[str, Any]],
    plan: Mapping[str, Any],
    *,
    plan_id: str | None = None,
) -> list[dict[str, Any]]:
    """Bound source records before a Luna bundle is created.

    The allowlist is a stable function of source metadata.  It always retains
    every preregistered mandatory pair and takes the earliest complete probe
    pairs, then the earliest owner rows.  No label, verdict, confidence, or
    completion field participates in the selection.
    """

    # Sampling is a frozen producer policy.  A caller cannot lower a limit or
    # substitute an outcome-conditioned policy in the plan artifact.
    base_limit = MAX_BASE_ROWS
    probe_limit = MAX_PROBE_PAIRS
    rally_caps = MAX_RALLIES_BY_SPLIT
    candidates_per_rally = MAX_BASE_CANDIDATES_PER_RALLY
    probe_rally_limit = MAX_PROBE_RALLIES
    pairs_per_rally = MAX_PROBE_PAIRS_PER_RALLY
    mandatory = _mandatory_rows(plan.get("mandatory_repeat_pairs"))
    mandatory_rallies = {item["rally_id"] for item in mandatory}
    probe_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    base_rows: list[dict[str, Any]] = []
    for row in rows:
        copy = dict(row)
        pair_id = str(copy.get("repeat_pair_id") or "")
        if pair_id:
            probe_groups[pair_id].append(copy)
        else:
            base_rows.append(copy)

    complete_pairs: dict[str, list[dict[str, Any]]] = {}
    pair_rallies: dict[str, str] = {}
    for pair_id, pair_rows in probe_groups.items():
        if len(pair_rows) != 2 or {
            str(item.get("blind_order") or "") for item in pair_rows
        } != {"a_first", "b_first"}:
            raise CertificationError("blind repeat pair is incomplete")
        complete_pairs[pair_id] = sorted(pair_rows, key=_row_sort_key)
        pair_rallies[pair_id] = str(pair_rows[0].get("rally_id") or "")

    mandatory_ids: set[str] = set()
    for item in mandatory:
        old_id = item["repeat_pair_id"]
        new_id = certification_repeat_pair_id(old_id, plan_id) if plan_id else ""
        if old_id in complete_pairs:
            mandatory_ids.add(old_id)
        elif new_id and new_id in complete_pairs:
            mandatory_ids.add(new_id)
        else:
            raise CertificationError("mandatory blind source is missing from bundle")
    ordered_pair_ids = sorted(
        complete_pairs,
        key=lambda pair_id: (_row_sort_key(complete_pairs[pair_id][0]), pair_id),
    )
    mandatory_order = [
        pair_id for pair_id in ordered_pair_ids if pair_id in mandatory_ids
    ]
    mandatory_probe_rallies = {
        pair_rallies[pair_id] for pair_id in mandatory_order
    }
    if len(mandatory_probe_rallies) > probe_rally_limit:
        raise CertificationError("mandatory blind pairs exceed source budget")
    selected_rallies = set(mandatory_probe_rallies)
    ordered_rallies: list[str] = []
    for pair_id in ordered_pair_ids:
        rally_id = pair_rallies[pair_id]
        if rally_id not in ordered_rallies:
            ordered_rallies.append(rally_id)
    for rally_id in ordered_rallies:
        if len(selected_rallies) >= probe_rally_limit:
            break
        selected_rallies.add(rally_id)
    # A rally-level cap avoids giving one source timestamp a disproportionate
    # share of the probe budget.  Mandatory pairs remain first within their
    # rally; all additional pair selection remains metadata-only.
    selected_pair_ids: list[str] = []
    for rally_id in ordered_rallies:
        if rally_id not in selected_rallies:
            continue
        rally_pairs = [
            pair_id
            for pair_id in ordered_pair_ids
            if pair_rallies[pair_id] == rally_id
        ]
        required_pairs = [pair_id for pair_id in rally_pairs if pair_id in mandatory_ids]
        extra_pairs = [pair_id for pair_id in rally_pairs if pair_id not in mandatory_ids]
        selected_pair_ids.extend(
            [*required_pairs, *extra_pairs[: max(0, pairs_per_rally - len(required_pairs))]]
        )
    if len(selected_pair_ids) > probe_limit:
        selected_pair_ids = sorted(
            selected_pair_ids,
            key=lambda pair_id: (
                pair_id not in mandatory_ids,
                _row_sort_key(complete_pairs[pair_id][0]),
                pair_id,
            ),
        )[:probe_limit]
    if not mandatory_ids.issubset(selected_pair_ids):
        raise CertificationError("mandatory blind pairs exceed source budget")
    selected_probe_rows = [
        row for pair_id in selected_pair_ids for row in complete_pairs[pair_id]
    ]
    by_split_rally: dict[str, dict[str, list[dict[str, Any]]]] = {
        split: defaultdict(list) for split in SOURCE_SPLITS
    }
    for row in base_rows:
        split = str(row.get("split") or "")
        rally_id = str(row.get("rally_id") or "")
        if split in by_split_rally and rally_id:
            by_split_rally[split][rally_id].append(row)
    selected_base_rows: list[dict[str, Any]] = []
    for split in ("train", "validation", "test"):
        groups = by_split_rally[split]
        rally_order = sorted(
            groups,
            key=lambda rally_id: (_row_sort_key(groups[rally_id][0]), rally_id),
        )
        mandatory_in_split = [
            rally_id for rally_id in rally_order if rally_id in mandatory_rallies
        ]
        rally_order = [
            *mandatory_in_split,
            *[rally_id for rally_id in rally_order if rally_id not in mandatory_rallies],
        ]
        for rally_id in rally_order[: rally_caps[split]]:
            selected_base_rows.extend(
                sorted(groups[rally_id], key=_row_sort_key)[:candidates_per_rally]
            )
    selected_base_rows = sorted(selected_base_rows, key=_row_sort_key)[:base_limit]
    selected = [*selected_base_rows, *selected_probe_rows]
    return sorted(selected, key=_row_sort_key)


def build_source_rows(
    plan: Mapping[str, Any],
    rallies: Sequence[Mapping[str, Any]],
    snapshots: Mapping[str, Mapping[str, Any]],
    texts: Mapping[str, str],
    *,
    plan_id: str,
    existing_pairs: Iterable[tuple[str, str]] = (),
    mandatory_source_rows: Sequence[Mapping[str, Any]] = (),
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Use the daily producer plus exact preregistered source retries.

    ``mandatory_source_rows`` is supplied from the old immutable bundle in
    memory.  It is never written to the plan artifact; this keeps the original
    input bytes intact without copying private text into repository metadata.
    """

    assignments = _assignment_map(plan)
    mandatory_pairs = {
        (item["rally_id"], item["candidate_id"])
        for item in _mandatory_rows(plan.get("mandatory_repeat_pairs"))
    }
    excluded = {
        (str(rally_id), str(candidate_id))
        for rally_id, candidate_id in existing_pairs
    }
    excluded.difference_update(mandatory_pairs)
    selected_rallies = [
        dict(rally)
        for rally in rallies
        if str(rally.get("rally_id") or "") in assignments
    ]
    rows, skipped = daily._build_rows(  # type: ignore[attr-defined]
        selected_rallies,
        snapshots,
        assignments,
        texts,
        excluded_pairs=excluded,
    )
    mandatory_keys = {
        (
            str(row.get("rally_id") or ""),
            str(row.get("candidate_id") or ""),
            str(row.get("repeat_pair_id") or ""),
            str(row.get("blind_order") or ""),
        )
        for row in mandatory_source_rows
    }
    if mandatory_source_rows:
        allowed_keys = {
            (
                item["rally_id"],
                item["candidate_id"],
                item["repeat_pair_id"],
                item["blind_order"],
            )
            for item in _mandatory_rows(plan.get("mandatory_repeat_pairs"))
        }
        if not mandatory_keys.issubset(allowed_keys):
            raise CertificationError("mandatory source row is not preregistered")
        rows = [
            row
            for row in rows
            if (
                str(row.get("rally_id") or ""),
                str(row.get("candidate_id") or ""),
                str(row.get("repeat_pair_id") or ""),
                str(row.get("blind_order") or ""),
            )
            not in mandatory_keys
        ]
        rows.extend(dict(row) for row in mandatory_source_rows)
    rows = _sample_source_rows(rows, plan, plan_id=plan_id)
    bound = bind_rows_to_plan(rows, plan, plan_id)
    if len(bound) > MAX_CERTIFICATION_ROWS:
        raise CertificationError("certification_source_over_limit")
    return bound, dict(skipped)


def create_certification_bundle(
    output: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    plan: Mapping[str, Any],
    plan_id: str,
    frozen_as_of_us: int,
    metadata: Mapping[str, Any] | None = None,
) -> Any:
    """Create a native bundle through the existing validated API."""

    sampled = _sample_source_rows(rows, plan, plan_id=plan_id)
    bound = bind_rows_to_plan(sampled, plan, plan_id)
    if not bound:
        raise CertificationError("bundle_source_empty")
    if len(bound) > MAX_CERTIFICATION_ROWS:
        raise CertificationError("certification_source_over_limit")
    bundle_metadata = dict(metadata or {})
    bundle_metadata["certification_revision"] = CERTIFICATION_REVISION
    return create_bundle(
        output,
        bound,
        teacher_profile=DEFAULT_TEACHER_PROFILE,
        cohort=NATIVE_COHORT,
        frozen_as_of_us=frozen_as_of_us,
        split_plan_id=plan_id,
        shard_count=DEFAULT_SHARD_COUNT,
        batch_size=DEFAULT_BATCH_SIZE,
        metadata=bundle_metadata,
    )
