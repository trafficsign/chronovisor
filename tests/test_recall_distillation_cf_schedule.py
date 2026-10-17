from __future__ import annotations

import json
from pathlib import Path

import pytest

from chronovisor.core.canonical_json import canonical_json_sha256_strict
from chronovisor.recall import recall_distillation_cf_schedule as schedule
from chronovisor.recall import recall_distillation_store as store

COHORT = "a" * 64
WATERMARK = "b" * 64
NEXT_WATERMARK = "c" * 64


def _rally(rally_id: str, epoch: int, session: str) -> dict[str, object]:
    candidate_id = f"candidate-{rally_id}"
    exposure_id = ""
    return {
        "rally_id": rally_id,
        "as_of_us": epoch,
        "session_cluster_id": session,
        "task_id": f"task-{rally_id}",
        "entity_id": f"entity-{rally_id}",
        "host": "fixture.example",
        "session_id_sha256": "d" * 64,
        "query_sha256": "e" * 64,
        "actual_answer_refs": [{"answer_id": f"answer-{rally_id}"}],
        "exposure_receipts": [
            {
                "exposure_artifact_id": exposure_id,
                "candidate_ids": [candidate_id],
            }
        ],
    }


def _source(root: Path, rally: dict[str, object]) -> dict[str, object]:
    rally_id = str(rally["rally_id"])
    candidate_id = f"candidate-{rally_id}"
    exposure_id = _write_exposure(root, rally, candidate_id)
    rally["exposure_receipts"] = [
        {"exposure_artifact_id": exposure_id, "candidate_ids": [candidate_id]}
    ]
    snapshot: dict[str, object] = {
        "rally_id": rally_id,
        "source_kind": schedule.SOURCE_KIND,
        "selected": [candidate_id],
        "exposure_artifact_id": exposure_id,
    }
    snapshot["snapshot_sha256"] = canonical_json_sha256_strict(
        {key: value for key, value in snapshot.items() if key != "snapshot_sha256"}
    )
    return snapshot


def _write_exposure(root: Path, rally: dict[str, object], candidate_id: str) -> str:
    _, _, artifact = store.write_immutable(
        store.distillation_dir(root) / "exposures",
        {
            "kind": "exact-rendered-exposure",
            "decision_id": f"decision-{rally['rally_id']}",
            "host": rally["host"],
            "session_id_sha256": rally["session_id_sha256"],
            "query_semantic_sha256": rally["query_sha256"],
            "policy_id": "policy-fixture",
            "candidate_ids": [candidate_id],
            "candidate_refs": [{"candidate_id": candidate_id}],
        },
        schema=schedule.EXPOSURE_SCHEMA,
    )
    # `_source` fills the receipt after the artifact ID is known.  The helper
    # is deliberately separate so fixtures exercise the real receipt binding.
    assert artifact["kind"] == "exact-rendered-exposure"
    return str(artifact["artifact_id"])


def _install_anchor(
    root: Path,
    rallies: list[dict[str, object]],
    assignments: dict[str, str] | None = None,
) -> bytes:
    assignments = assignments or {
        "anchor-train": "train",
        "anchor-validation": "validation",
        "anchor-test": "test",
    }
    _, _, plan = store.write_immutable(
        store.distillation_dir(root) / "split-plans",
        {
            "kind": "fixed-chronological-group-split",
            "raw_watermark": WATERMARK,
            "feature_revision": schedule.FEATURE_REVISION,
            "model_cohort_sha256": COHORT,
            "split_revision": "grouped-rolling-v1",
            "assignments": assignments,
        },
        schema=schedule.SPLIT_PLAN_SCHEMA,
    )
    store.write_sealed_state(
        store.distillation_dir(root) / "split-plan.json",
        {"kind": "split-plan-pointer", "split_plan_id": plan["artifact_id"]},
    )
    return (store.distillation_dir(root) / "split-plan.json").read_bytes()


def _anchor_rallies() -> list[dict[str, object]]:
    return [
        _rally("anchor-train", 1, "session-train"),
        _rally("anchor-validation", 2, "session-validation"),
        _rally("anchor-test", 3, "session-test"),
    ]


def test_initial_anchor_has_no_eligible_rows_and_does_not_change_pointer(
    tmp_path: Path,
) -> None:
    rallies = _anchor_rallies()
    pointer_before = _install_anchor(tmp_path, rallies)
    assert (
        schedule.ensure_counterfactual_schedule(
            tmp_path, rallies, {}, model_cohort_sha256=COHORT, raw_watermark=WATERMARK
        )
        == {}
    )
    state = schedule.read_schedule(tmp_path)
    assert state is not None
    assert state["generations"][0]["rally_ids"] == []
    assert (
        pointer_before
        == (store.distillation_dir(tmp_path) / "split-plan.json").read_bytes()
    )


def test_future_component_gets_own_test_plan_and_read_plans_validates_source(
    tmp_path: Path,
) -> None:
    rallies = _anchor_rallies()
    _install_anchor(tmp_path, rallies)
    schedule.ensure_counterfactual_schedule(
        tmp_path, rallies, {}, model_cohort_sha256=COHORT, raw_watermark=WATERMARK
    )
    future = _rally("future-1", 4, "session-future-1")
    open_tail = _rally("future-open", 5, "session-future-1")
    snapshot = _source(tmp_path, future)
    all_rallies = [*rallies, future, open_tail]
    result = schedule.ensure_counterfactual_schedule(
        tmp_path,
        all_rallies,
        {future["rally_id"]: snapshot},
        model_cohort_sha256=COHORT,
        raw_watermark=NEXT_WATERMARK,
    )
    assert future["rally_id"] in result
    plan = result[str(future["rally_id"])]
    assert plan["assignments"][future["rally_id"]] == "test"
    assert plan["assignments"][open_tail["rally_id"]] == "test"
    assert plan["assignments"]["anchor-train"] == "train"
    assert plan["assignments"]["anchor-validation"] == "validation"
    assert plan["assignments"]["anchor-test"] == "test"
    assert (
        schedule.read_plans(
            tmp_path,
            rallies={future["rally_id"]: future},
            snapshots={future["rally_id"]: snapshot},
        )[future["rally_id"]]["artifact_id"]
        == plan["artifact_id"]
    )
    with pytest.raises(schedule.CounterfactualScheduleError, match="drift|invalid"):
        schedule.read_plans(
            tmp_path,
            rallies={future["rally_id"]: future},
            snapshots={
                future["rally_id"]: {**snapshot, "selected": ["other-candidate"]}
            },
        )


def test_cross_component_with_anchor_is_skipped(tmp_path: Path) -> None:
    rallies = _anchor_rallies()
    _install_anchor(tmp_path, rallies)
    schedule.ensure_counterfactual_schedule(
        tmp_path, rallies, {}, model_cohort_sha256=COHORT, raw_watermark=WATERMARK
    )
    crossing = _rally("future-crossing", 4, "session-train")
    snapshot = _source(tmp_path, crossing)
    assert (
        schedule.ensure_counterfactual_schedule(
            tmp_path,
            [*rallies, crossing],
            {crossing["rally_id"]: snapshot},
            model_cohort_sha256=COHORT,
            raw_watermark=NEXT_WATERMARK,
        )
        == {}
    )
    state = schedule.read_schedule(tmp_path)
    assert state is not None and len(state["generations"]) == 1


def test_anchor_continuation_does_not_block_independent_future_component(
    tmp_path: Path,
) -> None:
    rallies = _anchor_rallies()
    _install_anchor(tmp_path, rallies)
    schedule.ensure_counterfactual_schedule(
        tmp_path, rallies, {}, model_cohort_sha256=COHORT, raw_watermark=WATERMARK
    )

    # The continued anchor session is ineligible as a whole component.  An
    # independent session with a closed source row must still be scheduled.
    anchor_continuation = _rally("anchor-continuation", 4, "session-train")
    future = _rally("future-independent", 5, "session-independent")
    future_tail = _rally("future-independent-tail", 6, "session-independent")
    snapshot = _source(tmp_path, future)
    result = schedule.ensure_counterfactual_schedule(
        tmp_path,
        [*rallies, anchor_continuation, future, future_tail],
        {future["rally_id"]: snapshot},
        model_cohort_sha256=COHORT,
        raw_watermark=NEXT_WATERMARK,
    )

    assert future["rally_id"] in result
    plan = result[future["rally_id"]]
    assert plan["assignments"][anchor_continuation["rally_id"]] == "embargo"
    assert plan["assignments"][future["rally_id"]] == "test"
    assert plan["assignments"][future_tail["rally_id"]] == "test"
    state = schedule.read_schedule(tmp_path)
    assert state is not None and len(state["generations"]) == 2


def test_latest_source_waits_for_later_same_session_query(tmp_path: Path) -> None:
    rallies = _anchor_rallies()
    _install_anchor(tmp_path, rallies)
    schedule.ensure_counterfactual_schedule(
        tmp_path, rallies, {}, model_cohort_sha256=COHORT, raw_watermark=WATERMARK
    )
    future = _rally("future-latest", 4, "session-future")
    snapshot = _source(tmp_path, future)
    assert (
        schedule.ensure_counterfactual_schedule(
            tmp_path,
            [*rallies, future],
            {future["rally_id"]: snapshot},
            model_cohort_sha256=COHORT,
            raw_watermark=NEXT_WATERMARK,
        )
        == {}
    )
    state = schedule.read_schedule(tmp_path)
    assert state is not None and len(state["generations"]) == 1


def test_invalid_anchor_chronology_fails_closed(tmp_path: Path) -> None:
    rallies = _anchor_rallies()
    _install_anchor(
        tmp_path,
        rallies,
        {
            "anchor-train": "validation",
            "anchor-validation": "train",
            "anchor-test": "test",
        },
    )
    with pytest.raises(schedule.CounterfactualScheduleError, match="chronology"):
        schedule.ensure_counterfactual_schedule(
            tmp_path, rallies, {}, model_cohort_sha256=COHORT, raw_watermark=WATERMARK
        )


def test_generations_preserve_old_source_mapping_and_tampered_state_fails(
    tmp_path: Path,
) -> None:
    rallies = _anchor_rallies()
    _install_anchor(tmp_path, rallies)
    schedule.ensure_counterfactual_schedule(
        tmp_path, rallies, {}, model_cohort_sha256=COHORT, raw_watermark=WATERMARK
    )
    future_one = _rally("future-1", 4, "session-future-1")
    open_tail = _rally("future-open", 5, "session-future-1")
    snapshot_one = _source(tmp_path, future_one)
    first = schedule.ensure_counterfactual_schedule(
        tmp_path,
        [*rallies, future_one, open_tail],
        {future_one["rally_id"]: snapshot_one},
        model_cohort_sha256=COHORT,
        raw_watermark=NEXT_WATERMARK,
    )
    # A later rally in an already scheduled future component is admitted into
    # its own immutable generation; the previous generation remains untouched.
    future_two = _rally("future-2", 6, "session-future-1")
    future_three = _rally("future-3", 7, "session-future-1")
    snapshot_two = _source(tmp_path, future_two)
    capture_snapshot = dict(snapshot_two)
    capture_snapshot.pop("selected")
    capture_snapshot.pop("source_kind")
    capture_snapshot["candidates"] = [
        {
            "candidate_id": "candidate-future-2",
            "source_kind": schedule.SOURCE_KIND,
            "exposure_artifact_id": snapshot_two["exposure_artifact_id"],
        }
    ]
    capture_snapshot["counterfactual_capture"] = {
        "status": "ready",
        "source_kind": schedule.SOURCE_KIND,
        "exposure_artifact_id": snapshot_two["exposure_artifact_id"],
    }
    capture_snapshot["snapshot_sha256"] = canonical_json_sha256_strict(
        {
            key: value
            for key, value in capture_snapshot.items()
            if key != "snapshot_sha256"
        }
    )
    all_result = schedule.ensure_counterfactual_schedule(
        tmp_path,
        [*rallies, future_one, open_tail, future_two, future_three],
        {
            future_one["rally_id"]: snapshot_one,
            future_two["rally_id"]: capture_snapshot,
        },
        model_cohort_sha256=COHORT,
        raw_watermark="d" * 64,
    )
    state = schedule.read_schedule(tmp_path)
    assert state is not None
    assert len(state["generations"]) == 3
    assert (
        all_result[future_one["rally_id"]]["artifact_id"]
        == first[future_one["rally_id"]]["artifact_id"]
    )
    assert (
        all_result[future_two["rally_id"]]["artifact_id"]
        != first[future_one["rally_id"]]["artifact_id"]
    )
    schedule_path = store.distillation_dir(tmp_path) / schedule.SCHEDULE_FILE
    original = schedule_path.read_bytes()
    payload = json.loads(original)
    payload["rally_plan_ids"][future_one["rally_id"]] = "0" * 64
    schedule_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(schedule.CounterfactualScheduleError):
        schedule.read_schedule(tmp_path)
