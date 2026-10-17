from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from chronovisor.core import canonical_json
from chronovisor.ingest.page_registry import PageRegistry
from chronovisor.recall import recall_distillation as distill
from chronovisor.recall import recall_distillation_catalog as catalog
from chronovisor.recall import recall_distillation_cf_capture as capture
from chronovisor.recall import recall_distillation_store as store
from scripts import recall_luna_daily as daily


def _source_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, dict[str, Any], dict[str, Any], str]:
    """Create one source-verified page exposure and a historical snapshot."""

    root = tmp_path / "root"
    root.mkdir()
    page_id = "fixture-page"
    page_path = root / "pages" / f"{page_id}.md"
    page_path.parent.mkdir(parents=True)
    page_path.write_text(
        "---\n"
        f"title: {page_id}\n"
        "status: stable\n"
        "type: knowledge\n"
        "updated: 2026-10-12\n"
        "---\n\n"
        "synthetic integration fixture\n",
        encoding="utf-8",
    )
    registry = PageRegistry(root)
    registry.ensure_manifest()
    resolved = registry.resolve(page_id)
    assert resolved is not None
    resolved_path = registry.path_for(page_id, require_stable=True)
    assert resolved_path == page_path
    page_uid = str(resolved["uid"])
    page_sha = hashlib.sha256(page_path.read_bytes()).hexdigest()
    rendered = json.dumps(
        {
            "items": [
                {
                    "page_id": page_id,
                    "source_ref": {
                        "doc_id": f"page-record:{page_sha}",
                        "uid": page_uid,
                        "sha256": page_sha,
                        "generation_id": "integration-fixture",
                        "byte_start": 0,
                        "byte_end": page_path.stat().st_size,
                    },
                }
            ]
        },
        separators=(",", ":"),
    )
    candidate_ref = {
        "candidate_id": page_id,
        "page_id": page_id,
        "page_content_sha256": page_sha,
        "rendered_context": rendered,
        "rendered_context_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
    }
    feature_row = {
        "candidate_id": page_id,
        "features": {
            "query_chargram_coverage": 0.5,
            "candidate_chargram_precision": 0.75,
        },
    }
    payload = {
        "kind": "exact-rendered-exposure",
        "decision_id": "integration-decision",
        "host": "integration-host",
        "session_id_sha256": "a" * 64,
        "query_semantic_sha256": "b" * 64,
        "policy_id": "c" * 64,
        "candidate_ids": [page_id],
        "candidate_refs_sha256": canonical_json.canonical_json_sha256_strict(
            [candidate_ref]
        ),
        "candidate_pool_refs_sha256": canonical_json.canonical_json_sha256_strict(
            [{**candidate_ref, "selected": True}]
        ),
        "candidate_feature_snapshot_sha256": canonical_json.canonical_json_sha256_strict(
            [feature_row]
        ),
        "runtime_observation_sha256": "d" * 64,
        "render_sha256": "e" * 64,
        "renderer_revision": "recall-card-v1",
        "context_style": "default",
        "candidate_snapshot_sha256": "f" * 64,
        "observed_at": "2026-10-12T03:39:00Z",
        "candidate_refs": [candidate_ref],
        "candidate_pool_refs": [{**candidate_ref, "selected": True}],
        "candidate_feature_snapshot": [feature_row],
        "runtime_observation": None,
    }
    artifact_id, _path, _artifact = store.write_immutable(
        store.distillation_dir(root) / "exposures",
        payload,
        schema="chronovisor.recall-exact-exposure.v1",
    )
    receipt = {
        "exposure_artifact_id": artifact_id,
        "candidate_ids": [page_id],
        "candidate_snapshot_sha256": payload["candidate_snapshot_sha256"],
    }
    rally = {
        "rally_id": "integration-rally",
        "host": payload["host"],
        "session_id_sha256": payload["session_id_sha256"],
        "query_sha256": payload["query_semantic_sha256"],
        "as_of": "2026-10-12T03:38:56Z",
        "session_cluster_id": "integration-session",
        "actual_answer_refs": [{"answer_id": "integration-answer"}],
        "exposure_receipts": [receipt],
    }
    canonical = {
        "decision_id": payload["decision_id"],
        "policy_id": payload["policy_id"],
        "candidate_ids": [page_id],
        "candidate_snapshot_sha256": payload["candidate_snapshot_sha256"],
        "candidate_feature_snapshot_sha256": payload[
            "candidate_feature_snapshot_sha256"
        ],
        "observed_at": payload["observed_at"],
        "record_sha256": "1" * 64,
        "exposure_artifact_id": artifact_id,
    }
    monkeypatch.setattr(
        distill,
        "_exposure_map",
        lambda _root: {
            (rally["host"], rally["session_id_sha256"], rally["query_sha256"]): [
                canonical
            ]
        },
    )
    as_of = rally["as_of"]
    snapshot = {
        "schema": catalog.CANDIDATE_SNAPSHOT_SCHEMA,
        "rally_id": rally["rally_id"],
        "as_of": as_of,
        "retriever_revision": "historical-fts-v1",
        "feature_revision": distill.TEXT_FEATURE_REVISION,
        "query_feature_text_sha256": "2" * 64,
        "candidates": [
            {"candidate_id": "historical-atom", "rank": 1, "text_sha256": "3" * 64}
        ],
    }
    snapshot["snapshot_sha256"] = canonical_json.canonical_json_sha256_strict(
        {key: value for key, value in snapshot.items() if key != "snapshot_sha256"}
    )
    augmented = capture.augment_candidate_snapshot(root, rally, snapshot)
    assert augmented["counterfactual_capture"]["status"] == "ready"
    return root, rally, augmented, artifact_id


def _validate_candidate_append(snapshot: dict[str, Any]) -> None:
    row = {
        "schema": store.DISTILLATION_SCHEMA,
        "namespace": "recall-distillation",
        "kind": "candidate-snapshot",
        "rally_id": snapshot["rally_id"],
        "snapshot": snapshot,
        "previous_sha256": "",
    }
    row["record_sha256"] = canonical_json.canonical_json_sha256_strict(
        {key: value for key, value in row.items() if key != "record_sha256"}
    )
    encoded = canonical_json.canonical_json_bytes_strict(row) + b"\n"
    metadata = catalog._candidate_snapshot(
        row,
        offset=0,
        length=len(encoded),
        expected_previous="",
    )
    assert metadata["snapshot_sha256"] == snapshot["snapshot_sha256"]


def _binding_rows(
    rally: dict[str, Any], artifact_id: str, candidate_id: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    assignment = {
        "revision": distill.ASSIGNMENT_REVISION,
        "kind": "counterfactual",
    }
    authority = {
        "rally_id": rally["rally_id"],
        "profile": "integration-profile",
        "cohort": "integration-cohort",
        "assignment_revision": distill.ASSIGNMENT_REVISION,
        "as_of": rally["as_of"],
        "group_id": rally["session_cluster_id"],
        "a0_sha256": "4" * 64,
        "a1_sha256": "5" * 64,
        "blind_orders": ["a0_first", "a1_first"],
        "counterfactual_producer": "integration-producer",
        "counterfactual_revision": "integration-revision",
        "generator_route_identity": {"role": "generator"},
        "generator_model_digest": "6" * 64,
        "judge_route_identity": {"role": "judge"},
        "judge_model_digest": "7" * 64,
    }
    label = {
        **authority,
        "candidate_id": candidate_id,
        "exposure_artifact_id": artifact_id,
        "split_plan_id": "8" * 64,
        "assignment": assignment,
    }
    row = {
        **authority,
        "candidate_id": candidate_id,
        "counterfactual_ref": artifact_id,
        "label_split_plan_id": label["split_plan_id"],
        "assignment_authority": distill._training_assignment_authority(assignment),
    }
    return label, row


def test_source_snapshot_append_and_strict_binding_accept_page_cf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, rally, snapshot, artifact_id = _source_fixture(tmp_path, monkeypatch)
    _validate_candidate_append(snapshot)
    page = snapshot["candidates"][-1]
    label, row = _binding_rows(rally, artifact_id, str(page["candidate_id"]))
    monkeypatch.setattr(
        distill,
        "_materialization_snapshots",
        lambda _root, _supplied, *, rally_ids=None: {
            rally["rally_id"]: snapshot
        },
    )

    assert distill._sealed_counterfactual_exposure_binding(
        root, row, label, rally
    )

    changed = copy.deepcopy(snapshot)
    changed["candidates"][-1]["text_sha256"] = "9" * 64
    monkeypatch.setattr(
        distill,
        "_materialization_snapshots",
        lambda _root, _supplied, *, rally_ids=None: {
            rally["rally_id"]: changed
        },
    )
    assert not distill._sealed_counterfactual_exposure_binding(
        root, row, label, rally
    )

    bad_rally = {
        **rally,
        "exposure_receipts": [
            {**rally["exposure_receipts"][0], "candidate_ids": ["other-page"]}
        ],
    }
    monkeypatch.setattr(
        distill,
        "_materialization_snapshots",
        lambda _root, _supplied, *, rally_ids=None: {
            rally["rally_id"]: snapshot
        },
    )
    assert not distill._sealed_counterfactual_exposure_binding(
        root, row, label, bad_rally
    )


def test_teacher_and_daily_selection_keep_historical_shape_and_drop_cf_rows() -> None:
    historical = [
        {"candidate_id": f"historical-{index}", "rank": index + 1}
        for index in range(1, 7)
    ]
    snapshot = {
        "candidates": [
            historical[0],
            {"candidate_id": "cf-page-1", "source_kind": "exact-exposure-page-v1"},
            historical[1],
            historical[2],
            historical[3],
            {"candidate_id": "cf-page-2", "source_kind": "exact-exposure-page-v1"},
            historical[4],
            historical[5],
        ]
    }

    teacher_rows = distill.teacher_candidate_rows(snapshot)
    safe_rows = daily._safe_candidate_ids(snapshot)
    selected = daily._selected_candidates(snapshot)

    assert [row["candidate_id"] for row in teacher_rows] == [
        row["candidate_id"] for row in historical
    ]
    assert [row["candidate_id"] for row in safe_rows] == [
        row["candidate_id"] for row in historical
    ]
    assert [row["candidate_id"] for row in selected] == [
        "historical-1",
        "historical-2",
        "historical-3",
        "historical-6",
    ]


def test_counterfactual_own_plan_is_bound_without_changing_legacy_work_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, rally, snapshot, _artifact_id = _source_fixture(tmp_path, monkeypatch)
    rally_id = str(rally["rally_id"])
    legacy_id = "a" * 64
    own_id = "b" * 64
    legacy_plan = {
        "artifact_id": legacy_id,
        "assignments": {rally_id: "train"},
    }
    own_plan = {
        "artifact_id": own_id,
        "assignments": {rally_id: "test"},
    }
    monkeypatch.setattr(distill, "_read_split_plan", lambda _root: legacy_plan)
    monkeypatch.setattr(distill, "_scheduling_age_bands", lambda *_args: {})

    legacy_items, legacy_keys = distill._prepare_counterfactual_work(
        root=root,
        snapshots={rally_id: snapshot},
        rally_by_id={rally_id: rally},
    )
    repeated_items, repeated_keys = distill._prepare_counterfactual_work(
        root=root,
        snapshots={rally_id: snapshot},
        rally_by_id={rally_id: rally},
    )
    own_items, own_keys = distill._prepare_counterfactual_work(
        root=root,
        snapshots={rally_id: snapshot},
        rally_by_id={rally_id: rally},
        counterfactual_plans={rally_id: own_plan},
    )

    page_key = (rally_id, "fixture-page")
    assert legacy_keys[page_key] == repeated_keys[page_key]
    assert legacy_keys[page_key] != own_keys[page_key]
    legacy_page = next(
        item for item in legacy_items if item["work_id"] == legacy_keys[page_key]
    )
    repeated_page = next(
        item for item in repeated_items if item["work_id"] == repeated_keys[page_key]
    )
    own_page = next(item for item in own_items if item["work_id"] == own_keys[page_key])
    assert legacy_page["temporal_split"] == {
        "as_of": rally["as_of"],
        "group_id": rally["session_cluster_id"],
        "split": "train",
        "split_plan_id": legacy_id,
    }
    assert repeated_page == legacy_page
    assert own_page["temporal_split"] == {
        "as_of": rally["as_of"],
        "group_id": rally["session_cluster_id"],
        "split": "test",
        "split_plan_id": own_id,
    }
