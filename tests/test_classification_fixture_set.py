from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from chronovisor.classification import ClassificationError
from chronovisor.classification_fixture_set import (
    DISABLED_BASELINE_SCHEMA,
    FIXTURE_SET_SCHEMA,
    create_disabled_baseline_manifest,
    fixture_set_paths,
    inference_dto,
    load_fixture_set,
    lock_fixture_set,
)
from chronovisor.durable_state import write_sealed_json


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _rows(count: int) -> list[dict]:
    return [
        {
            "uid": f"uid-{index:03d}",
            "fixture_group_id": f"group-{index:03d}",
            "source_sha256": f"sha256:{index:064x}",
            "adjudication_status": "accepted",
            "gold_primary_notation": "004.8",
            "gold_allowed_primary_notations": ["004.8"],
            "gold_rationale": "AI subject",
            "title": f"Page {index}",
        }
        for index in range(count)
    ]


def test_fixture_epoch_is_isolated_and_variable_count(tmp_path: Path) -> None:
    paths = fixture_set_paths(tmp_path, "epoch-3-library-evidence-v1")
    candidates = _rows(7)
    _write_jsonl(paths.candidates, candidates)
    write_sealed_json(
        paths.root / "candidate-lock.json",
        {
            "schema": "chronovisor.classification-fixture-candidate-lock.v1",
            "fixture_epoch": "epoch-3-library-evidence-v1",
            "candidate_sha256": (
                "sha256:" + hashlib.sha256(paths.candidates.read_bytes()).hexdigest()
            ),
            "adjudication_started": False,
        },
    )

    manifest = lock_fixture_set(
        tmp_path,
        fixture_epoch="epoch-3-library-evidence-v1",
        adjudicated_rows=candidates,
        adjudicator="test",
        dev_count=2,
        holdout_count=3,
    )

    assert manifest["schema"] == FIXTURE_SET_SCHEMA
    assert manifest["fixture_epoch"] != manifest["engine_version"]
    assert manifest["dev"]["count"] == 2
    assert manifest["holdout"]["count"] == 3
    assert manifest["reserve"]["count"] == 2
    assert manifest["current_pointer_changed"] is False
    assert not (tmp_path / "classification" / "fixtures" / "manifest.json").exists()
    assert load_fixture_set(paths.manifest)["holdout"]["count"] == 3


def test_fixture_set_rejects_group_leak_and_insufficient_evaluable(
    tmp_path: Path,
) -> None:
    paths = fixture_set_paths(tmp_path, "epoch-test")
    candidates = _rows(3)
    _write_jsonl(paths.candidates, candidates)
    digest = "sha256:" + hashlib.sha256(paths.candidates.read_bytes()).hexdigest()
    write_sealed_json(
        paths.root / "candidate-lock.json",
        {"candidate_sha256": digest, "adjudication_started": False},
    )
    rows = _rows(3)
    rows[2]["fixture_group_id"] = rows[1]["fixture_group_id"]

    with pytest.raises(ClassificationError, match="requires 3 evaluable"):
        lock_fixture_set(
            tmp_path,
            fixture_epoch="epoch-test",
            adjudicated_rows=rows,
            adjudicator="test",
            dev_count=1,
            holdout_count=2,
        )


def test_inference_dto_is_gold_free() -> None:
    dto = inference_dto(
        {
            "uid": "uid-1",
            "title": "AI",
            "gold_primary_notation": "004.8",
            "gold_allowed_primary_notations": ["004.8"],
            "adjudication_status": "accepted",
            "fixture_split": "holdout",
        }
    )

    assert dto["uid"] == "uid-1"
    assert not any(key.startswith(("gold_", "adjudication_")) for key in dto)
    assert "fixture_split" not in dto


def test_disabled_baseline_is_first_class_not_corruption(tmp_path: Path) -> None:
    receipt = tmp_path / "classification" / "bundles" / "disabled.json"
    payload = create_disabled_baseline_manifest(
        tmp_path,
        a0_config={"candidate_limit": 12, "worker_policy": "current"},
        receipt_path=receipt,
    )

    assert payload["schema"] == DISABLED_BASELINE_SCHEMA
    assert payload["intentional_disabled_sentinel"] is True
    assert payload["mutation_capability"] is False
    assert payload["candidate_limit"] == 12
    assert receipt.exists()
