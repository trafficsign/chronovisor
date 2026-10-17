from pathlib import Path
from typing import Any

import pytest

from chronovisor.core.canonical_json import canonical_json_sha256_strict
from chronovisor.recall import recall_distillation as distill
from chronovisor.recall import recall_distillation_store as store


def _plan(root: Path, name: str, cohort: str = "a" * 64) -> dict[str, Any]:
    return store.write_immutable(
        store.distillation_dir(root) / "split-plans",
        {
            "kind": "fixed-counterfactual-scheduling-cohort",
            "feature_revision": distill.TEXT_FEATURE_REVISION,
            "model_cohort_sha256": cohort,
            "assignments": {name: "test"},
        },
        schema=distill.SPLIT_PLAN_SCHEMA,
    )[2]


def _lineage(*plans: dict[str, Any]) -> dict[str, Any]:
    ids = sorted(plan["artifact_id"] for plan in plans)
    return {
        "counterfactual_plan_ids": ids,
        "counterfactual_plan_ids_sha256": canonical_json_sha256_strict(ids),
    }


def test_cf_lineage_keeps_each_immutable_plan_and_teacher_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    owner = _plan(tmp_path, "teacher")
    first = _plan(tmp_path, "cf-first")
    second = _plan(tmp_path, "cf-second")
    rows = [
        {"source": source, "split_plan_id": plan["artifact_id"], "rally_id": rally}
        for source, plan, rally in (
            ("teacher-label", owner, "teacher"),
            ("counterfactual-label", first, "cf-first"),
            ("counterfactual-label", second, "cf-second"),
        )
    ]
    seen: list[str] = []

    def verify(row: dict[str, Any], *, root: Path, split_plan: dict[str, Any]) -> bool:
        assert root == tmp_path
        seen.append(split_plan["artifact_id"])
        return (
            row["split_plan_id"] == split_plan["artifact_id"]
            and row["rally_id"] in split_plan["assignments"]
        )

    monkeypatch.setattr(distill, "_materialized_row_integrity", verify)
    assert distill._lineage_rows_integrity(
        tmp_path, rows, owner, _lineage(first, second)
    )
    assert seen == [owner["artifact_id"], first["artifact_id"], second["artifact_id"]]
    assert not distill._lineage_rows_integrity(tmp_path, rows, owner, {})
    assert not distill._lineage_rows_integrity(tmp_path, rows, owner, _lineage(first))
    assert not distill._lineage_rows_integrity(
        tmp_path,
        rows,
        owner,
        {**_lineage(first, second), "counterfactual_plan_ids_sha256": "0" * 64},
    )
    # Correct plan inventory cannot hide a failed existing per-row validator.
    rows[-1]["rally_id"] = "not-in-the-frozen-plan"
    assert not distill._lineage_rows_integrity(
        tmp_path, rows, owner, _lineage(first, second)
    )


def test_cf_lineage_rejects_missing_or_other_cohort_plan(tmp_path: Path) -> None:
    owner = _plan(tmp_path, "teacher")
    other = _plan(tmp_path, "other", "b" * 64)
    row = {"source": "counterfactual-label", "split_plan_id": other["artifact_id"]}
    assert not distill._lineage_rows_integrity(tmp_path, [row], owner, _lineage(other))
    missing = {"artifact_id": "f" * 64}
    row["split_plan_id"] = missing["artifact_id"]
    assert not distill._lineage_rows_integrity(
        tmp_path, [row], owner, _lineage(missing)
    )


def test_legacy_single_plan_lineage_still_requires_existing_row_validator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    owner = _plan(tmp_path, "teacher")
    row = {"source": "counterfactual-label", "split_plan_id": owner["artifact_id"]}
    monkeypatch.setattr(distill, "_materialized_row_integrity", lambda *_a, **_kw: True)
    assert distill._lineage_rows_integrity(tmp_path, [row], owner, {})
    monkeypatch.setattr(
        distill, "_materialized_row_integrity", lambda *_a, **_kw: False
    )
    assert not distill._lineage_rows_integrity(tmp_path, [row], owner, {})
