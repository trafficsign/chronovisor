from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from chronovisor.core import canonical_json
from chronovisor.recall import recall_distillation as distill
from chronovisor.recall import recall_distillation_native as native
from chronovisor.recall import recall_distillation_store as store
from chronovisor.recall.recall_distillation_luna import (
    LunaBulkBundle,
    native_output_sha256,
)


def _native_config(bundle: Path = Path()) -> distill.DistillationConfig:
    return distill.DistillationConfig(
        teacher_profile=distill.NATIVE_LUNA_SINGLE_PROFILE,
        native_luna_enabled=True,
        native_luna_bundle=str(bundle),
    )


def _native_fixture(
    root: Path, *, count: int = 3
) -> tuple[
    distill.DistillationConfig,
    dict[str, Any],
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
    LunaBulkBundle,
]:
    """Build a private synthetic native bundle and its canonical source views."""

    config = _native_config(root / "bundle")
    contract = native.ensure_contract(root, config)
    cohort = {
        "revision": "single-teacher-cohort-v1",
        "profile": distill.NATIVE_LUNA_SINGLE_PROFILE,
        "cohort": native.NATIVE_COHORT,
        "profile_contract_id": contract["artifact_id"],
        "model_digests": [distill.NATIVE_LUNA_FIXED_IDENTITY["model_digest"]],
    }
    cohort_sha256 = canonical_json.canonical_json_sha256_strict(cohort)
    as_of_values = (
        "2026-10-08T14:59:59Z",
        "2026-10-09T00:00:00Z",
        "2026-10-10T00:00:00Z",
    )
    features = distill.build_fast_features(
        query_chargram_coverage=0.75, candidate_chargram_precision=0.65
    )
    rallies: list[dict[str, Any]] = []
    snapshots: dict[str, dict[str, Any]] = {}
    for index in range(count):
        rally_id = f"native-rally-{index}"
        candidate_id = f"native-candidate-{index}"
        as_of = as_of_values[index % len(as_of_values)]
        as_of_us = int(
            datetime.fromisoformat(as_of.replace("Z", "+00:00")).timestamp() * 1_000_000
        )
        query_sha256 = f"{index + 1:064x}"
        context_sha256 = f"{index + 101:064x}"
        snapshot_sha256 = f"{index + 201:064x}"
        candidate_text_sha256 = f"{index + 301:064x}"
        group_id = f"native-group-{index}"
        rallies.append(
            {
                "rally_id": rally_id,
                "host": "synthetic",
                "session_id_sha256": f"{index + 401:064x}",
                "session_cluster_id": group_id,
                "as_of": as_of,
                "as_of_us": as_of_us,
                "source_index": index,
                "query_sha256": query_sha256,
                "context_refs": [{"semantic_sha256": context_sha256}],
                "actual_answer_refs": [{"semantic_sha256": f"{index + 501:064x}"}],
            }
        )
        snapshots[rally_id] = {
            "rally_id": rally_id,
            "as_of": as_of,
            "snapshot_sha256": snapshot_sha256,
            "feature_revision": distill.TEXT_FEATURE_REVISION,
            "candidates": [
                {
                    "candidate_id": candidate_id,
                    "text_sha256": candidate_text_sha256,
                    "features": dict(features),
                }
            ],
        }

    plan = distill._ensure_split_plan(
        root,
        rallies,
        raw_watermark="a" * 64,
        model_cohort_sha256=cohort_sha256,
    )
    source_rows: list[dict[str, Any]] = []
    for index, rally in enumerate(rallies):
        rally_id = str(rally["rally_id"])
        candidate = snapshots[rally_id]["candidates"][0]
        source_rows.append(
            {
                "rally_id": rally_id,
                "candidate_id": candidate["candidate_id"],
                "as_of": rally["as_of"],
                "as_of_us": rally["as_of_us"],
                "input": {"synthetic_ordinal": index},
                "snapshot_sha256": snapshots[rally_id]["snapshot_sha256"],
                "query_sha256": rally["query_sha256"],
                "context_sha256": [
                    ref["semantic_sha256"] for ref in rally["context_refs"]
                ],
                "candidate_text_sha256": candidate["text_sha256"],
                "split": plan["assignments"][rally_id],
                "group_id": rally["session_cluster_id"],
                "features": dict(features),
                "assignment": {
                    "revision": native.NATIVE_ASSIGNMENT_REVISION,
                    "profile": distill.NATIVE_LUNA_SINGLE_PROFILE,
                    "split": plan["assignments"][rally_id],
                    "probe": False,
                    "owner": "native-luna",
                },
            }
        )

    bundle = LunaBulkBundle.create(
        root / "bundle",
        source_rows,
        cohort=native.NATIVE_COHORT,
        split_plan_id=plan["artifact_id"],
        shard_count=1,
        batch_size=count,
    )
    request = bundle.next_batch(0)
    labels: list[dict[str, Any]] = []
    for index, source in enumerate(request["records"]):
        label: dict[str, Any] = {
            "task_id": source["task_id"],
            "input_sha256": source["input_sha256"],
            "teacher_profile": distill.NATIVE_LUNA_SINGLE_PROFILE,
            "native_luna": True,
            "verdict": "relevant" if index % 2 == 0 else "irrelevant",
            "confidence": 0.9,
            "reason_code": "synthetic_match" if index % 2 == 0 else "synthetic_miss",
            "rationale": "synthetic integration result",
        }
        label["native_receipt"] = {
            "task_id": source["task_id"],
            "output_sha256": native_output_sha256(label),
            "assignment_id": f"synthetic-assignment-{index}",
            "run_id": f"synthetic-run-{index}",
        }
        labels.append(label)
    bundle.submit(0, cursor=0, records=labels)
    bundle.finalize()
    return config, contract, rallies, snapshots, bundle


def _materialize_native_fixture(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[distill.DistillationConfig, dict[str, Any], list[dict[str, Any]]]:
    config, contract, rallies, snapshots, bundle = _native_fixture(root)
    native.import_bundle(root, bundle.root)
    rally_map = {str(rally["rally_id"]): rally for rally in rallies}
    monkeypatch.setattr(
        distill,
        "_materialization_rallies",
        lambda _root, _supplied: rally_map,
    )
    monkeypatch.setattr(
        distill,
        "_materialization_snapshots",
        lambda _root, _supplied: snapshots,
    )
    rows = distill.materialize_training_rows(
        root, _rallies=rallies, _snapshots=snapshots
    )["rows"]
    return config, contract, rows


def test_native_bundle_materializes_to_authoritative_rows_and_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, contract, rows = _materialize_native_fixture(tmp_path, monkeypatch)

    assert len(rows) == 3
    assert all(row["profile"] == distill.NATIVE_LUNA_SINGLE_PROFILE for row in rows)
    assert all("native_receipt" not in row for row in rows)
    assert all(native.row_integrity(row, root=tmp_path) for row in rows)
    assert all(
        distill._materialized_row_integrity(row, root=tmp_path) for row in rows
    )

    gate = distill._offline_training_gate(rows, config, root=tmp_path)
    assert gate["identity"]["profile_contract_id"] == contract["artifact_id"]
    assert "row_integrity_failed" not in gate["reasons"]
    assert "row_authority_mismatch" not in gate["reasons"]
    assert "payload_source_digest_mismatch" not in gate["reasons"]
    assert "profile_contract_mismatch" not in gate["reasons"]
    assert "expiry_missing" not in gate["reasons"]
    assert "teacher_labels_below_floor" in gate["reasons"]


def test_native_import_rejects_tampered_input_hash(tmp_path: Path) -> None:
    config, _contract, _rallies, _snapshots, bundle = _native_fixture(tmp_path)
    batch_path = bundle.root / "shards" / "00" / "batches" / "batch-000000.json"
    payload = json.loads(batch_path.read_text(encoding="utf-8"))
    payload["records"][0]["input_sha256"] = "f" * 64
    batch_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(native.NativeDistillationError, match="bundle is invalid"):
        native.import_bundle(tmp_path, bundle.root)
    assert not (store.distillation_dir(tmp_path) / store.LABEL_LEDGER_FILE).exists()
    assert config.native_luna_enabled is True


def test_native_counterfactual_work_excludes_pre_10_october_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _native_config(tmp_path / "bundle")
    native.ensure_contract(tmp_path, config)
    old = {
        "rally_id": "before-cutoff",
        "as_of": "2026-10-08T14:59:59Z",
        "actual_answer_refs": [{"semantic_sha256": "a" * 64}],
    }
    new = {
        "rally_id": "after-cutoff",
        "as_of": "2026-10-08T15:00:00Z",
        "actual_answer_refs": [{"semantic_sha256": "b" * 64}],
    }
    rallies = {old["rally_id"]: old, new["rally_id"]: new}
    snapshots = {
        key: {
            "rally_id": key,
            "as_of": value["as_of"],
            "snapshot_sha256": f"{index + 1:064x}",
            "candidates": [],
        }
        for index, (key, value) in enumerate(rallies.items())
    }
    captured: dict[str, set[str]] = {}

    def prepare(**kwargs: Any) -> tuple[list[dict[str, Any]], dict[tuple[str, str], str]]:
        captured["rally_ids"] = set(kwargs["rally_by_id"])
        return [], {}

    monkeypatch.setattr(distill, "_prepare_counterfactual_work", prepare)

    class Counterfactual:
        local = True

        def compare(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
            raise AssertionError(f"compare must not run in a dry-run cutoff check: {payload}")

    result = distill._run_counterfactual_block(
        execute=False,
        root=tmp_path,
        raw_dir=tmp_path / "raw",
        config=config,
        counterfactual=Counterfactual(),
        snapshots=snapshots,
        rally_by_id=rallies,
        texts={},
        label_path=store.distillation_dir(tmp_path) / store.LABEL_LEDGER_FILE,
        label_rows=[],
    )

    assert result.pending is False
    assert captured["rally_ids"] == {"after-cutoff"}
