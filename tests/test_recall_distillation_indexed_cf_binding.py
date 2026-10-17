from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from chronovisor.core.store import RuntimeContext, init_chronovisor
from chronovisor.recall import recall_distillation as distill
from chronovisor.recall import recall_distillation_catalog as catalog
from chronovisor.recall import recall_distillation_store as store


def _snapshot(candidate_id: str = "candidate-1", text_sha256: str = "text-sha") -> dict[str, Any]:
    return {
        "schema": "chronovisor.recall-candidate-snapshot.v1",
        "rally_id": "rally-1",
        "as_of": "2026-01-01T00:00:00Z",
        "retriever_revision": "test-retriever-v1",
        "feature_revision": distill.TEXT_FEATURE_REVISION,
        "query_feature_text_sha256": "q" * 64,
        "candidates": [
            {
                "candidate_id": candidate_id,
                "rank": 1,
                "text_sha256": text_sha256,
            }
        ],
        "snapshot_sha256": "s" * 64,
    }


def test_materialization_snapshots_uses_indexed_target_without_full_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = _snapshot()
    calls: list[tuple[Path, Path, object]] = []

    monkeypatch.setattr(catalog, "_read_catalog_checkpoint", lambda _root: {"valid": True})

    def read_indexed(root: Path, path: Path, rally_ids: object) -> dict[str, dict[str, Any]]:
        calls.append((root, path, rally_ids))
        return {"rally-1": expected}

    monkeypatch.setattr(catalog, "read_candidate_snapshots", read_indexed)
    monkeypatch.setattr(
        distill,
        "_read_chain",
        lambda _path: (_ for _ in ()).throw(AssertionError("full candidate read")),
    )

    result = distill._materialization_snapshots(
        tmp_path, None, rally_ids=("rally-1",)
    )

    assert result == {"rally-1": expected}
    assert len(calls) == 1
    assert calls[0][0] == tmp_path
    assert calls[0][1] == store.distillation_dir(tmp_path) / "candidate-ledger.jsonl"
    assert calls[0][2] == ("rally-1",)


def test_materialization_snapshots_legacy_checkpoint_fallback_filters_full_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = _snapshot()
    second = {**_snapshot(), "rally_id": "rally-2"}
    monkeypatch.setattr(catalog, "_read_catalog_checkpoint", lambda _root: None)
    monkeypatch.setattr(
        catalog,
        "read_candidate_snapshots",
        lambda *_args: (_ for _ in ()).throw(AssertionError("indexed path")),
    )
    monkeypatch.setattr(
        distill,
        "_read_chain",
        lambda _path: [
            {"rally_id": "rally-1", "snapshot": first},
            {"rally_id": "rally-2", "snapshot": second},
        ],
    )

    result = distill._materialization_snapshots(
        tmp_path, None, rally_ids=("rally-2",)
    )

    assert result == {"rally-2": second}


def test_materialization_snapshots_index_error_is_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(catalog, "_read_catalog_checkpoint", lambda _root: {"valid": True})
    monkeypatch.setattr(
        catalog,
        "read_candidate_snapshots",
        lambda *_args: (_ for _ in ()).throw(catalog.CatalogError("index mismatch")),
    )
    monkeypatch.setattr(
        distill,
        "_read_chain",
        lambda _path: (_ for _ in ()).throw(AssertionError("legacy fallback")),
    )

    with pytest.raises(store.DistillationStoreError):
        distill._materialization_snapshots(
            tmp_path, None, rally_ids=("rally-1",)
        )


def _binding_fixture(root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    init_chronovisor(RuntimeContext(root))
    artifact_id, _, _ = store.write_immutable(
        store.distillation_dir(root) / "exposures",
        {
            "kind": "exact-rendered-exposure",
            "candidate_ids": ["candidate-1"],
            "candidate_refs": [
                {"candidate_id": "candidate-1", "content_sha256": "text-sha"}
            ],
            "candidate_pool_refs": [],
            "candidate_feature_snapshot": [],
        },
        schema="chronovisor.recall-exact-exposure.v1",
    )
    receipt = {
        "exposure_artifact_id": artifact_id,
        "candidate_ids": ["candidate-1"],
    }
    rally = {
        "rally_id": "rally-1",
        "host": "host-1",
        "session_id_sha256": "session-1",
        "query_sha256": "query-1",
        "exposure_receipts": [receipt],
    }
    assignment = {"revision": distill.ASSIGNMENT_REVISION, "kind": "counterfactual"}
    authority = {
        "rally_id": "rally-1",
        "profile": "profile",
        "cohort": "cohort",
        "assignment_revision": distill.ASSIGNMENT_REVISION,
        "as_of": "2026-01-01T00:00:00Z",
        "group_id": "group-1",
        "a0_sha256": "a" * 64,
        "a1_sha256": "b" * 64,
        "blind_orders": ["a0_first", "a1_first"],
        "counterfactual_producer": "producer",
        "counterfactual_revision": "revision",
        "generator_route_identity": {"role": "generator"},
        "generator_model_digest": "c" * 64,
        "judge_route_identity": {"role": "judge"},
        "judge_model_digest": "d" * 64,
    }
    label = {
        **authority,
        "candidate_id": "candidate-1",
        "exposure_artifact_id": artifact_id,
        "split_plan_id": "plan-1",
        "label_split_plan_id": "plan-1",
        "assignment": assignment,
    }
    row = {
        **authority,
        "candidate_id": "candidate-1",
        "counterfactual_ref": artifact_id,
        "label_split_plan_id": "plan-1",
        "assignment_authority": distill._training_assignment_authority(assignment),
    }
    monkeypatch_target = (rally, label, row)
    return monkeypatch_target


@pytest.mark.parametrize("text_sha256, expected", [("text-sha", True), ("other-sha", False)])
def test_counterfactual_binding_preserves_hash_predicate_with_targeted_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    text_sha256: str,
    expected: bool,
) -> None:
    rally, label, row = _binding_fixture(tmp_path)
    monkeypatch.setattr(
        distill,
        "_exposure_map",
        lambda _root: {
            ("host-1", "session-1", "query-1"): rally["exposure_receipts"]
        },
    )
    calls: list[object] = []

    def targeted(_root: Path, _supplied: object, *, rally_ids: object = None) -> dict[str, Any]:
        calls.append(rally_ids)
        assert rally_ids == ("rally-1",)
        return {"rally-1": _snapshot(text_sha256=text_sha256)}

    monkeypatch.setattr(distill, "_materialization_snapshots", targeted)

    assert distill._sealed_counterfactual_exposure_binding(tmp_path, row, label, rally) is expected
    assert calls == [("rally-1",)]
