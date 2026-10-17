from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from chronovisor.recall import recall_distillation as distill
from chronovisor.recall import recall_distillation_store as store
from scripts import recall_luna_certification as certification


def _rally(index: int, *, split: str | None = None, group: str | None = None) -> dict:
    instant = datetime(2026, 1, 1, tzinfo=UTC) + timedelta(days=index)
    row = {
        "rally_id": f"rally-{index:02d}",
        "as_of": instant.isoformat().replace("+00:00", "Z"),
        "as_of_us": int(instant.timestamp() * 1_000_000),
        "session_cluster_id": group or f"group-{index:02d}",
        "query_sha256": "a" * 64,
    }
    if split:
        row["split"] = split
    return row


def _inventory() -> tuple[list[dict], dict[str, str], list[dict]]:
    rallies = [
        *[_rally(index, split="train") for index in range(600)],
        *[_rally(index, split="validation") for index in range(700, 1100)],
        *[_rally(index, split="test") for index in range(1200, 1700)],
        *[_rally(index) for index in range(1800, 1810)],
    ]
    assignments = {
        row["rally_id"]: row["split"]
        for row in rallies
        if row.get("split") is not None
    }
    mandatory = [
        {
            "repeat_pair_id": "pair-01",
            "blind_order": order,
            "rally_id": "rally-1200",
            "candidate_id": "candidate-1200",
            "input_sha256": "a" * 64,
        }
        for order in ("a_first", "b_first")
    ]
    return rallies, assignments, mandatory


def test_plan_is_strict_and_reuses_old_test_only_as_new_test() -> None:
    rallies, assignments, mandatory = _inventory()
    result = certification.plan_certification_assignments(
        rallies,
        {"artifact_id": "c" * 64, "assignments": assignments},
        mandatory_pairs=mandatory,
    )

    selected = result["assignments"]
    assert selected["rally-1200"] == "test"
    assert set(selected.values()) == {"train", "validation", "test"}
    assert result["protected_old_test_groups"]
    by_split = {
        split: [row["as_of_us"] for row in rallies if selected.get(row["rally_id"]) == split]
        for split in ("train", "validation", "test")
    }
    assert max(by_split["train"]) < min(by_split["validation"])
    assert max(by_split["validation"]) < min(by_split["test"])


def test_cross_split_group_is_quarantined_and_mandatory_pair_fails_closed() -> None:
    rallies = [
        _rally(1, split="train", group="cross"),
        _rally(20, split="test", group="cross"),
        _rally(2, split="train"),
        _rally(12, split="validation"),
        _rally(24, split="test"),
    ]
    assignments = {
        row["rally_id"]: row["split"] for row in rallies if row.get("split")
    }
    mandatory = [
        {
            "repeat_pair_id": "pair-cross",
            "blind_order": order,
            "rally_id": "rally-20",
            "candidate_id": "candidate-cross",
        }
        for order in ("a_first", "b_first")
    ]
    with pytest.raises(certification.CertificationError, match="boundary|quarantined"):
        certification.plan_certification_assignments(
            rallies,
            {"assignments": assignments},
            mandatory_pairs=mandatory,
        )


def test_plan_bound_task_ids_change_without_mutating_input() -> None:
    row = {
        "rally_id": "rally-1",
        "candidate_id": "candidate-1",
        "as_of_us": 1,
        "split": "train",
        "input": {"query": "q", "candidate": "c", "context": []},
    }
    before = dict(row["input"])
    first = certification.certification_task_id(row, "a" * 64)
    second = certification.certification_task_id(row, "b" * 64)
    assert first != second
    assert row["input"] == before


def test_bind_rows_requires_preregistered_blind_pair_orders() -> None:
    plan = {
        "assignments": {"rally-1": "test"},
        "mandatory_repeat_pairs": [
            {
                "repeat_pair_id": "pair-1",
                "blind_order": order,
                "rally_id": "rally-1",
                "candidate_id": "candidate-1",
            }
            for order in ("a_first", "b_first")
        ],
    }
    row = {
        "rally_id": "rally-1",
        "candidate_id": "candidate-1",
        "as_of_us": 1,
        "split": "test",
        "repeat_pair_id": "pair-1",
        "blind_order": "a_first",
        "input": {"query": "q", "candidate": "c", "context": []},
    }
    with pytest.raises(certification.CertificationError, match="mandatory"):
        certification.bind_rows_to_plan([row], plan, "a" * 64)
    complete = [row, {**row, "blind_order": "b_first"}]
    bound = certification.bind_rows_to_plan(complete, plan, "a" * 64)
    assert len({item["task_id"] for item in bound}) == 2
    assert {item["repeat_pair_id"] for item in bound} == {
        certification.certification_repeat_pair_id("pair-1", "a" * 64)
    }


def test_bind_rows_rebinds_every_probe_and_is_idempotent() -> None:
    plan = {
        "assignments": {"rally-1": "test"},
        "mandatory_repeat_pairs": [
            {
                "repeat_pair_id": "pair-1",
                "blind_order": order,
                "rally_id": "rally-1",
                "candidate_id": "candidate-1",
            }
            for order in ("a_first", "b_first")
        ],
    }
    rows = []
    for pair_id in ("pair-1", "pair-extra"):
        rows.extend(
            {
                "rally_id": "rally-1",
                "candidate_id": "candidate-1",
                "as_of_us": 1,
                "split": "test",
                "repeat_pair_id": pair_id,
                "blind_order": order,
                "fixed_repeat": True,
                "order_swap": True,
                "assignment": {"repeat_pair_id": pair_id},
                "input": {"query": "q", "candidate": pair_id, "context": []},
            }
            for order in ("a_first", "b_first")
        )
    first = certification.bind_rows_to_plan(rows, plan, "b" * 64)
    second = certification.bind_rows_to_plan(first, plan, "b" * 64)
    assert [row["task_id"] for row in first] == [row["task_id"] for row in second]
    assert {
        row["repeat_pair_id"] for row in first if row["repeat_pair_id"]
    } == {
        certification.certification_repeat_pair_id("pair-1", "b" * 64),
        certification.certification_repeat_pair_id("pair-extra", "b" * 64),
    }
    assert all(
        row["assignment"]["repeat_pair_id"] == row["repeat_pair_id"]
        for row in first
    )


def test_bind_rows_checks_preregistered_input_hash_before_rebinding() -> None:
    input_value = {"query": "frozen", "candidate": "original", "context": []}
    plan = {
        "assignments": {"rally-1": "test"},
        "mandatory_repeat_pairs": [
            {
                "repeat_pair_id": "pair-1",
                "blind_order": order,
                "rally_id": "rally-1",
                "candidate_id": "candidate-1",
                "input_sha256": certification.canonical_json_sha256_strict(input_value),
            }
            for order in ("a_first", "b_first")
        ],
    }
    rows = [
        {
            "rally_id": "rally-1",
            "candidate_id": "candidate-1",
            "as_of_us": 1,
            "split": "test",
            "repeat_pair_id": "pair-1",
            "blind_order": order,
            "input": {**input_value, "candidate": "substituted"},
        }
        for order in ("a_first", "b_first")
    ]
    with pytest.raises(certification.CertificationError, match="input hash"):
        certification.bind_rows_to_plan(rows, plan, "c" * 64)


def test_source_sampling_is_stratified_and_keeps_mandatory_pairs() -> None:
    mandatory = [
        {
            "repeat_pair_id": "pair-required",
            "blind_order": order,
            "rally_id": "test-00",
            "candidate_id": "candidate-00",
        }
        for order in ("a_first", "b_first")
    ]
    plan = {
        "assignments": {
            **{f"train-{index:02d}": "train" for index in range(3)},
            **{f"val-{index:02d}": "validation" for index in range(3)},
            **{f"test-{index:02d}": "test" for index in range(3)},
        },
        "mandatory_repeat_pairs": mandatory,
    }
    rows = []
    for split, prefix in (("train", "train"), ("validation", "val"), ("test", "test")):
        for index in range(3):
            rally_id = f"{prefix}-{index:02d}"
            rows.append(
                {
                    "rally_id": rally_id,
                    "candidate_id": f"base-{index}",
                    "as_of_us": index + (0 if split == "train" else 10 if split == "validation" else 20),
                    "split": split,
                    "input": {"rally": rally_id},
                }
            )
    for pair_id, rally_id in (("pair-required", "test-00"), ("pair-extra", "test-01")):
        for order in ("a_first", "b_first"):
            rows.append(
                {
                    "rally_id": rally_id,
                    "candidate_id": "candidate-00",
                    "as_of_us": 20,
                    "split": "test",
                    "repeat_pair_id": pair_id,
                    "blind_order": order,
                    "input": {"rally": rally_id, "order": order},
                }
            )
    selected = certification._sample_source_rows(rows, plan)
    assert {row["repeat_pair_id"] for row in selected if row.get("repeat_pair_id")} == {
        "pair-required",
        "pair-extra",
    }
    base = [row for row in selected if not row.get("repeat_pair_id")]
    assert len(base) == 9
    assert {row["split"] for row in base} == {"train", "validation", "test"}


def test_write_plan_uses_existing_split_schema_without_pointer(tmp_path: Path) -> None:
    rallies, assignments, mandatory = _inventory()
    plan = certification.plan_certification_assignments(
        rallies,
        {"artifact_id": "d" * 64, "assignments": assignments},
        mandatory_pairs=mandatory,
    )
    artifact = certification.write_certification_plan(
        tmp_path,
        plan,
        rallies=rallies,
        raw_watermark="e" * 64,
        model_cohort_sha256="f" * 64,
        candidate_snapshot_sha256="a" * 64,
    )
    assert artifact["schema"] == distill.SPLIT_PLAN_SCHEMA
    path = store.distillation_dir(tmp_path) / "split-plans" / f"{artifact['artifact_id']}.json"
    sealed = store.read_sealed(path, schema=distill.SPLIT_PLAN_SCHEMA)
    assert sealed["artifact_id"] == artifact["artifact_id"]
    assert not (store.distillation_dir(tmp_path) / "split-plan.json").exists()
    assert sealed["certification_revision"] == certification.CERTIFICATION_REVISION
    assert sealed["kind"] == "fixed-certification-cohort"
    assert sealed["split_revision"] == "fixed-certification-v1"


def test_build_source_rows_allows_only_preregistered_existing_pairs(monkeypatch) -> None:
    plan = {
        "assignments": {"rally-1": "test"},
        "mandatory_repeat_pairs": [
            {
                "repeat_pair_id": "pair-1",
                "blind_order": order,
                "rally_id": "rally-1",
                "candidate_id": "candidate-1",
            }
            for order in ("a_first", "b_first")
        ],
    }
    rows = [
        {
            "rally_id": "rally-1",
            "candidate_id": "candidate-1",
            "as_of_us": 1,
            "split": "test",
            "repeat_pair_id": "pair-1",
            "blind_order": order,
            "input": {"query": "q", "candidate": "c", "context": []},
        }
        for order in ("a_first", "b_first")
    ]
    observed: dict[str, set[tuple[str, str]]] = {}

    def fake_build(rallies, snapshots, assignments, texts, *, excluded_pairs):
        del rallies, snapshots, assignments, texts
        observed["excluded"] = set(excluded_pairs)
        return rows, {}

    monkeypatch.setattr(certification.daily, "_build_rows", fake_build)
    bound, skipped = certification.build_source_rows(
        plan,
        [{"rally_id": "rally-1"}],
        {},
        {},
        plan_id="a" * 64,
        existing_pairs={("rally-1", "candidate-1"), ("rally-2", "candidate-2")},
    )
    assert not skipped
    assert observed["excluded"] == {("rally-2", "candidate-2")}
    assert len(bound) == 2


def test_build_source_rows_keeps_exact_mandatory_payload(monkeypatch) -> None:
    plan = {
        "assignments": {"rally-1": "test"},
        "mandatory_repeat_pairs": [
            {
                "repeat_pair_id": "pair-1",
                "blind_order": order,
                "rally_id": "rally-1",
                "candidate_id": "candidate-1",
            }
            for order in ("a_first", "b_first")
        ],
    }
    generated = [
        {
            "rally_id": "rally-1",
            "candidate_id": "candidate-1",
            "as_of_us": 1,
            "split": "test",
            "repeat_pair_id": "pair-1",
            "blind_order": order,
            "input": {"query": "generated", "candidate": "c", "context": []},
        }
        for order in ("a_first", "b_first")
    ]
    exact = [
        {
            **row,
            "input": {"query": "frozen", "candidate": "original", "context": []},
        }
        for row in generated
    ]
    monkeypatch.setattr(
        certification.daily,
        "_build_rows",
        lambda *args, **kwargs: (generated, {}),
    )
    bound, _ = certification.build_source_rows(
        plan,
        [{"rally_id": "rally-1"}],
        {},
        {},
        plan_id="a" * 64,
        mandatory_source_rows=exact,
    )
    assert {row["input"]["query"] for row in bound} == {"frozen"}
