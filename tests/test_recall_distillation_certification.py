from __future__ import annotations

from pathlib import Path

import pytest

from chronovisor.recall import recall_distillation as distill
from chronovisor.recall import recall_distillation_native as native
from chronovisor.recall import recall_distillation_store as store
from chronovisor.recall.recall_distillation_luna import (
    LunaBulkBundle,
    native_output_sha256,
)

COHORT_SHA256 = "c" * 64


def _patch_certification_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(native, "current_contract_id", lambda _root: "d" * 64)
    monkeypatch.setattr(
        distill,
        "_active_training_cohort",
        lambda _rows, **_kwargs: ([], {"cohort_sha256": COHORT_SHA256}),
    )


def _plan(root: Path, *, assignments: dict[str, str] | None = None) -> str:
    plan_id, _path, _artifact = store.write_immutable(
        store.distillation_dir(root) / "split-plans",
        {
            "kind": "fixed-certification-cohort",
            "raw_watermark": "a" * 64,
            "feature_revision": distill.TEXT_FEATURE_REVISION,
            "model_cohort_sha256": COHORT_SHA256,
            "split_revision": "fixed-certification-v1",
            "assignments": assignments or {"rally-a": "test"},
        },
        schema=distill.SPLIT_PLAN_SCHEMA,
    )
    return plan_id


def _source_row(*, rally_id: str = "rally-a", split: str = "test") -> dict[str, object]:
    return {
        "rally_id": rally_id,
        "candidate_id": f"candidate-{rally_id}",
        "as_of_us": 1_000,
        "input": {"query": f"query-{rally_id}", "candidate": "candidate"},
        "split": split,
        "group_id": f"group-{rally_id}",
    }


def _bundle(root: Path, plan_id: str, *, split: str = "test") -> LunaBulkBundle:
    row = _source_row(split=split)
    row["task_id"] = distill.certification_task_id(row, plan_id)
    return LunaBulkBundle.create(
        root / "bundle",
        [row],
        cohort=native.NATIVE_COHORT,
        frozen_as_of_us=1_000,
        split_plan_id=plan_id,
        shard_count=1,
        batch_size=1,
    )


def _label(
    request_row: dict[str, object], *, verdict: str = "irrelevant"
) -> dict[str, object]:
    task_id = str(request_row["task_id"])
    label: dict[str, object] = {
        "task_id": task_id,
        "input_sha256": request_row["input_sha256"],
        "teacher_profile": "native-luna-single-v1",
        "native_luna": True,
        "verdict": verdict,
        "confidence": 0.9,
        "reason_code": "direct_match" if verdict == "relevant" else "not_relevant",
        "rationale": None,
    }
    label["native_receipt"] = {
        "task_id": task_id,
        "output_sha256": native_output_sha256(label),
        "assignment_id": "assignment-1",
        "run_id": "run-1",
    }
    return label


def _register_pointer(
    root: Path,
    plan_id: str,
    *,
    source_snapshot_sha256: str = "e" * 64,
    task_ids_sha256: str = "f" * 64,
    kind: str = "fixed-certification-cohort",
) -> Path:
    path = store.distillation_dir(root) / "certification-plan.json"
    store.write_sealed_state(
        path,
        {
            "kind": kind,
            "split_plan_id": plan_id,
            "source_snapshot_sha256": source_snapshot_sha256,
            "task_ids_sha256": task_ids_sha256,
            "record_count": 1,
        },
    )
    return path


def test_native_import_rejects_certification_bundle_without_registration(
    tmp_path: Path,
) -> None:
    root = tmp_path / "native-root"
    plan_id = _plan(root)
    bundle = _bundle(tmp_path / "unregistered", plan_id)
    native.ensure_contract(
        root,
        {
            "teacher_profile": "native-luna-single-v1",
            "native_luna_enabled": True,
        },
    )

    with pytest.raises(
        native.NativeDistillationError,
        match="certification inventory is not registered",
    ):
        native.import_bundle(root, bundle.root)


def test_native_import_rejects_bundle_from_different_registered_inventory(
    tmp_path: Path,
) -> None:
    root = tmp_path / "native-root"
    plan_id = _plan(root, assignments={"rally-a": "test", "rally-b": "test"})
    registered_bundle = _bundle(tmp_path / "registered", plan_id)
    other_row = _source_row(rally_id="rally-b")
    other_row["task_id"] = distill.certification_task_id(other_row, plan_id)
    other_bundle = LunaBulkBundle.create(
        tmp_path / "other" / "bundle",
        [other_row],
        cohort=native.NATIVE_COHORT,
        frozen_as_of_us=1_000,
        split_plan_id=plan_id,
        shard_count=1,
        batch_size=1,
    )
    _register_pointer(
        root,
        plan_id,
        source_snapshot_sha256=registered_bundle.manifest["source_snapshot_sha256"],
        task_ids_sha256=registered_bundle.manifest["task_ids_sha256"],
    )
    native.ensure_contract(
        root,
        {
            "teacher_profile": "native-luna-single-v1",
            "native_luna_enabled": True,
        },
    )

    with pytest.raises(
        native.NativeDistillationError,
        match="certification inventory is not registered",
    ):
        native.import_bundle(root, other_bundle.root)


def test_certification_activation_must_precede_first_judgment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_certification_identity(monkeypatch)
    plan_id = _plan(tmp_path)
    bundle = _bundle(tmp_path, plan_id)
    request = bundle.next_batch(0)
    bundle.submit(0, cursor=0, records=[_label(request["records"][0])])

    with pytest.raises(distill.DistillationError, match="before judgments"):
        distill.activate_certification_plan(tmp_path, bundle.root)


def test_certification_activation_is_idempotent_for_the_same_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_certification_identity(monkeypatch)
    plan_id = _plan(tmp_path)
    bundle = _bundle(tmp_path, plan_id)

    first = distill.activate_certification_plan(tmp_path, bundle.root)
    second = distill.activate_certification_plan(tmp_path, bundle.root)

    assert second == first
    assert second["split_plan_id"] == plan_id
    assert second["source_snapshot_sha256"] == bundle.manifest["source_snapshot_sha256"]


def test_different_post_outcome_bundle_cannot_register_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_certification_identity(monkeypatch)
    first_plan_id = _plan(tmp_path)
    first_bundle = _bundle(tmp_path, first_plan_id)
    distill.activate_certification_plan(tmp_path, first_bundle.root)

    second_plan_id = _plan(tmp_path, assignments={"rally-a": "test", "rally-b": "test"})
    second_row = _source_row(rally_id="rally-b")
    second_row["task_id"] = distill.certification_task_id(second_row, second_plan_id)
    second_bundle = LunaBulkBundle.create(
        tmp_path / "second-bundle",
        [second_row],
        frozen_as_of_us=1_000,
        split_plan_id=second_plan_id,
        shard_count=1,
        batch_size=1,
    )
    request = second_bundle.next_batch(0)
    second_bundle.submit(0, cursor=0, records=[_label(request["records"][0])])

    with pytest.raises(distill.DistillationError, match="already registered"):
        distill.activate_certification_plan(tmp_path, second_bundle.root)


def test_certification_activation_rejects_mismatched_source_assignment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_certification_identity(monkeypatch)
    plan_id = _plan(tmp_path)
    bundle = _bundle(tmp_path, plan_id, split="train")

    with pytest.raises(distill.DistillationError, match="assignment"):
        distill.activate_certification_plan(tmp_path, bundle.root)


def test_certification_activation_rejects_malformed_existing_pointer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_certification_identity(monkeypatch)
    plan_id = _plan(tmp_path)
    bundle = _bundle(tmp_path, plan_id)
    _register_pointer(
        tmp_path,
        plan_id,
        source_snapshot_sha256=bundle.manifest["source_snapshot_sha256"],
        kind="not-certification",
    )

    with pytest.raises(distill.DistillationError, match="pointer|identity|registered"):
        distill.activate_certification_plan(tmp_path, bundle.root)


def test_certification_materialization_keeps_new_adverse_pair_and_own_cf_plan(
    tmp_path: Path,
) -> None:
    certification_plan_id = _plan(
        tmp_path,
        assignments={"new-test": "test", "new-train": "train"},
    )
    old_plan_id, old_path, _old_artifact = store.write_immutable(
        store.distillation_dir(tmp_path) / "split-plans",
        {
            "kind": "fixed-chronological-group-split",
            "raw_watermark": "b" * 64,
            "feature_revision": distill.TEXT_FEATURE_REVISION,
            "model_cohort_sha256": COHORT_SHA256,
            "split_revision": "grouped-rolling-v1",
            "assignments": {"old-cf": "test"},
        },
        schema=distill.SPLIT_PLAN_SCHEMA,
    )
    pointer = _register_pointer(tmp_path, certification_plan_id)
    old_plan_before = old_path.read_bytes()

    def native_row(rally_id: str, verdict: str, plan_id: str) -> dict[str, object]:
        return {
            "source": "teacher-label",
            "profile": distill.NATIVE_LUNA_SINGLE_PROFILE,
            "label_split_plan_id": plan_id,
            "rally_id": rally_id,
            "candidate_id": f"candidate-{rally_id}",
            "as_of": "2026-01-01T00:00:00Z",
            "verdict": verdict,
            "payload_source": {"source_snapshot_sha256": "e" * 64},
        }

    materialized = [
        native_row("new-test", "irrelevant", certification_plan_id),
        native_row("new-test", "relevant", certification_plan_id),
        native_row("new-train", "relevant", certification_plan_id),
        native_row("old-cf", "irrelevant", old_plan_id),
        {
            "source": "counterfactual-label",
            "profile": distill.NATIVE_LUNA_SINGLE_PROFILE,
            "label_split_plan_id": old_plan_id,
            "rally_id": "old-cf",
            "candidate_id": "candidate-old-cf",
            "as_of": "2026-01-01T00:00:00Z",
            "verdict": "harmful",
        },
    ]

    artifact = distill._finalize_materialized_training_rows(
        tmp_path, materialized, limit=10
    )
    rows = artifact["rows"]
    assert sum(row["rally_id"] == "new-test" for row in rows) == 2
    assert any(
        row["rally_id"] == "old-cf"
        and row["source"] == "counterfactual-label"
        and row["split_plan_id"] == old_plan_id
        for row in rows
    )
    assert not any(
        row["rally_id"] == "old-cf" and row["source"] == "teacher-label" for row in rows
    )
    assert old_path.read_bytes() == old_plan_before
    assert pointer.read_bytes()


def test_certification_materialization_rejects_second_bundle_same_plan(
    tmp_path: Path,
) -> None:
    certification_plan_id = _plan(tmp_path)
    _register_pointer(tmp_path, certification_plan_id)
    injected = {
        "source": "teacher-label",
        "profile": distill.NATIVE_LUNA_SINGLE_PROFILE,
        "label_split_plan_id": certification_plan_id,
        "rally_id": "rally-a",
        "candidate_id": "candidate-rally-a",
        "as_of": "2026-01-01T00:00:00Z",
        "verdict": "irrelevant",
        "payload_source": {"source_snapshot_sha256": "b" * 64},
    }

    with pytest.raises(distill.DistillationError, match="outside.*inventory"):
        distill._finalize_materialized_training_rows(tmp_path, [injected], limit=10)
