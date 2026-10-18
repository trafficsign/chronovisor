from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from chronovisor.core import canonical_json
from chronovisor.recall import recall_distillation as distill
from chronovisor.recall import recall_distillation_catalog as catalog
from chronovisor.recall import recall_distillation_store as store
from chronovisor.recall import recall_distillation_validation as validation

_RALLY_ID = "fixture-rally"
_HOST = "fixture-host"
_SESSION_ID_SHA256 = "s" * 64
_QUERY_SHA256 = "q" * 64
_LIVE_CANDIDATE = "live-candidate"
_SELECTED_CANDIDATE = "selected-candidate"


def _features(*, coverage: float = 0.75, precision: float = 0.5) -> dict[str, float]:
    return distill.build_fast_features(
        query_chargram_coverage=coverage,
        candidate_chargram_precision=precision,
    )


def _write_exposure(
    root: Path,
    *,
    mode: str,
    candidate_id: str,
    features: dict[str, float] | None = None,
    feature_rows: list[dict[str, Any]] | None = None,
    include_candidate: bool = True,
    candidate_ids: list[str] | None = None,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    """Write a sealed synthetic exposure and its rally receipt."""

    if features is None:
        features = _features()
    if feature_rows is None:
        feature_rows = (
            [{"candidate_id": candidate_id, "features": features}]
            if include_candidate
            else []
        )
    refs = (
        [{"candidate_id": candidate_id}]
        if mode == "remove" and include_candidate
        else []
    )
    pool = (
        [{"candidate_id": candidate_id, "selected": False}]
        if mode == "add" and include_candidate
        else []
    )
    selected_ids = (
        [candidate_id]
        if candidate_ids is None and mode == "remove" and include_candidate
        else ([] if candidate_ids is None else candidate_ids)
    )
    feature_sha256 = canonical_json.canonical_json_sha256_strict(feature_rows)
    refs_sha256 = canonical_json.canonical_json_sha256_strict(refs)
    pool_sha256 = canonical_json.canonical_json_sha256_strict(pool)
    candidate_snapshot_sha256 = "c" * 64
    payload = {
        "kind": "exact-rendered-exposure",
        "host": _HOST,
        "session_id_sha256": _SESSION_ID_SHA256,
        "query_semantic_sha256": _QUERY_SHA256,
        "candidate_ids": selected_ids,
        "candidate_refs_sha256": refs_sha256,
        "candidate_pool_refs_sha256": pool_sha256,
        "candidate_feature_snapshot_sha256": feature_sha256,
        "candidate_snapshot_sha256": candidate_snapshot_sha256,
        "candidate_refs": refs,
        "candidate_pool_refs": pool,
        "candidate_feature_snapshot": feature_rows,
    }
    artifact_id, _, _ = store.write_immutable(
        store.distillation_dir(root) / "exposures",
        payload,
        schema="chronovisor.recall-exact-exposure.v1",
    )
    receipt = {
        "exposure_artifact_id": artifact_id,
        "candidate_ids": selected_ids,
        "candidate_snapshot_sha256": candidate_snapshot_sha256,
        "candidate_feature_snapshot_sha256": feature_sha256,
    }
    rally = {
        "rally_id": _RALLY_ID,
        "host": _HOST,
        "session_id_sha256": _SESSION_ID_SHA256,
        "query_sha256": _QUERY_SHA256,
        "exposure_receipts": [receipt],
    }
    return artifact_id, receipt, rally


def _label(
    *,
    candidate_id: str,
    kind: str = "counterfactual-label",
    mode: str = "add",
    exposure_artifact_id: str = "",
    features: dict[str, float] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "record_sha256": "r" * 64,
        "rally_id": _RALLY_ID,
        "candidate_id": candidate_id,
        "kind": kind,
        "features": _features() if features is None else features,
    }
    if kind == "counterfactual-label":
        row.update(mode=mode, exposure_artifact_id=exposure_artifact_id)
    return row


def _stub_catalog(
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
    rally: dict[str, Any],
    *,
    snapshot_candidates: list[dict[str, Any]] | None = None,
) -> list[dict[str, str]]:
    seen: list[dict[str, str]] = []
    candidates = list(snapshot_candidates or [])
    monkeypatch.setattr(
        catalog,
        "_read_catalog_checkpoint",
        lambda _root: {"catalog_watermark": "fixture"},
    )
    monkeypatch.setattr(
        catalog,
        "rallies",
        lambda _root, _ids: [rally],
    )

    def read_snapshots(
        _root: Path, _ledger: Path, claims: list[dict[str, str]]
    ) -> dict[str, dict[str, Any]]:
        seen.extend(dict(claim) for claim in claims)
        for claim in claims:
            candidate_id = claim.get("candidate_id")
            if candidate_id and candidate_id not in {
                str(row.get("candidate_id") or "") for row in candidates
            }:
                raise catalog.CatalogError("candidate claim is absent from snapshot")
        return {_RALLY_ID: {"rally_id": _RALLY_ID, "candidates": candidates}}

    monkeypatch.setattr(catalog, "read_candidate_snapshots", read_snapshots)
    return seen


def _catalog_for_label(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    label: dict[str, Any],
    rally: dict[str, Any],
    *,
    snapshot_candidates: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, str]], tuple[dict[str, Any], dict[str, Any]]]:
    seen = _stub_catalog(
        monkeypatch,
        root,
        rally,
        snapshot_candidates=snapshot_candidates,
    )
    claim = validation._label_source_claim(label)
    result = validation._catalog_sources(root, [_RALLY_ID], [claim])
    assert result is not None
    return seen, result


def test_exposure_only_add_uses_rally_claim_for_unselected_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact_id, _, rally = _write_exposure(
        tmp_path, mode="add", candidate_id=_LIVE_CANDIDATE
    )
    label = _label(
        candidate_id=_LIVE_CANDIDATE,
        exposure_artifact_id=artifact_id,
    )

    seen, sources = _catalog_for_label(tmp_path, monkeypatch, label, rally)

    assert len(seen) == 1
    assert seen[0]["rally_id"] == _RALLY_ID
    assert "candidate_id" not in seen[0]
    assert sources[0][_RALLY_ID] == rally


def test_exposure_only_remove_uses_rally_claim_for_selected_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact_id, _, rally = _write_exposure(
        tmp_path, mode="remove", candidate_id=_SELECTED_CANDIDATE
    )
    label = _label(
        candidate_id=_SELECTED_CANDIDATE,
        mode="remove",
        exposure_artifact_id=artifact_id,
    )

    seen, _ = _catalog_for_label(tmp_path, monkeypatch, label, rally)

    assert len(seen) == 1
    assert seen[0]["rally_id"] == _RALLY_ID
    assert "candidate_id" not in seen[0]


def test_teacher_same_candidate_keeps_historical_candidate_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact_id, _, rally = _write_exposure(
        tmp_path, mode="add", candidate_id=_LIVE_CANDIDATE
    )
    label = _label(
        candidate_id=_LIVE_CANDIDATE,
        kind="teacher-label",
        exposure_artifact_id=artifact_id,
    )
    seen = _stub_catalog(monkeypatch, tmp_path, rally)

    with pytest.raises(validation.SourceBindingError, match="catalog source"):
        validation._catalog_sources(
            tmp_path,
            [_RALLY_ID],
            [validation._label_source_claim(label)],
        )
    assert seen[0]["candidate_id"] == _LIVE_CANDIDATE
    assert "kind" not in seen[0]


def test_unbound_exposure_fails_closed_before_rally_only_lookup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact_id, _, rally = _write_exposure(
        tmp_path, mode="add", candidate_id=_LIVE_CANDIDATE
    )
    label = _label(
        candidate_id=_LIVE_CANDIDATE,
        exposure_artifact_id=artifact_id,
    )
    rally = {**rally, "exposure_receipts": []}
    seen = _stub_catalog(monkeypatch, tmp_path, rally)

    with pytest.raises(validation.SourceBindingError, match="catalog source"):
        validation._catalog_sources(
            tmp_path,
            [_RALLY_ID],
            [validation._label_source_claim(label)],
        )
    assert seen[0]["candidate_id"] == _LIVE_CANDIDATE
    assert seen[0]["kind"] == "counterfactual-label"


def test_tampered_exposure_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact_id, _, rally = _write_exposure(
        tmp_path, mode="add", candidate_id=_LIVE_CANDIDATE
    )
    path = store.distillation_dir(tmp_path) / "exposures" / f"{artifact_id}.json"
    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["host"] = "tampered-host"
    path.write_text(json.dumps(tampered), encoding="utf-8")
    label = _label(
        candidate_id=_LIVE_CANDIDATE,
        exposure_artifact_id=artifact_id,
    )
    seen = _stub_catalog(monkeypatch, tmp_path, rally)

    with pytest.raises(validation.SourceBindingError, match="catalog source"):
        validation._catalog_sources(
            tmp_path,
            [_RALLY_ID],
            [validation._label_source_claim(label)],
        )
    assert seen[0]["candidate_id"] == _LIVE_CANDIDATE
    assert seen[0]["kind"] == "counterfactual-label"


def test_resealed_exposure_with_same_path_identity_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact_id, _, rally = _write_exposure(
        tmp_path, mode="add", candidate_id=_LIVE_CANDIDATE
    )
    path = store.distillation_dir(tmp_path) / "exposures" / f"{artifact_id}.json"
    resealed = json.loads(path.read_text(encoding="utf-8"))
    resealed["test_annotation"] = "rewritten"
    resealed.pop("seal_sha256", None)
    resealed["seal_sha256"] = canonical_json.canonical_json_sha256_strict(resealed)
    path.write_text(canonical_json.canonical_json_strict(resealed) + "\n", encoding="utf-8")
    label = _label(
        candidate_id=_LIVE_CANDIDATE,
        exposure_artifact_id=artifact_id,
    )
    seen = _stub_catalog(monkeypatch, tmp_path, rally)

    with pytest.raises(validation.SourceBindingError, match="catalog source"):
        validation._catalog_sources(
            tmp_path,
            [_RALLY_ID],
            [validation._label_source_claim(label)],
        )
    assert seen[0]["candidate_id"] == _LIVE_CANDIDATE


def test_missing_exposure_feature_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact_id, _, rally = _write_exposure(
        tmp_path,
        mode="add",
        candidate_id=_LIVE_CANDIDATE,
        include_candidate=False,
        feature_rows=[],
    )
    label = _label(
        candidate_id=_LIVE_CANDIDATE,
        exposure_artifact_id=artifact_id,
    )
    seen = _stub_catalog(monkeypatch, tmp_path, rally)

    with pytest.raises(validation.SourceBindingError, match="catalog source"):
        validation._catalog_sources(
            tmp_path,
            [_RALLY_ID],
            [validation._label_source_claim(label)],
        )
    assert seen[0]["candidate_id"] == _LIVE_CANDIDATE
    assert seen[0]["kind"] == "counterfactual-label"


def test_mismatched_exposure_feature_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact_id, _, rally = _write_exposure(
        tmp_path,
        mode="add",
        candidate_id=_LIVE_CANDIDATE,
        feature_rows=[
            {"candidate_id": _LIVE_CANDIDATE, "features": _features(coverage=0.1)}
        ],
    )
    label = _label(
        candidate_id=_LIVE_CANDIDATE,
        exposure_artifact_id=artifact_id,
    )
    seen = _stub_catalog(monkeypatch, tmp_path, rally)

    with pytest.raises(validation.SourceBindingError, match="catalog source"):
        validation._catalog_sources(
            tmp_path,
            [_RALLY_ID],
            [validation._label_source_claim(label)],
        )
    assert seen[0]["candidate_id"] == _LIVE_CANDIDATE
    assert seen[0]["kind"] == "counterfactual-label"


def test_fake_exposure_candidate_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact_id, _, rally = _write_exposure(
        tmp_path, mode="add", candidate_id=_LIVE_CANDIDATE
    )
    label = _label(
        candidate_id="fake-candidate",
        exposure_artifact_id=artifact_id,
    )
    seen = _stub_catalog(monkeypatch, tmp_path, rally)

    with pytest.raises(validation.SourceBindingError, match="catalog source"):
        validation._catalog_sources(
            tmp_path,
            [_RALLY_ID],
            [validation._label_source_claim(label)],
        )
    assert seen[0]["candidate_id"] == "fake-candidate"
    assert seen[0]["kind"] == "counterfactual-label"


def test_add_exposure_rejects_selected_candidate_contradiction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact_id, _, rally = _write_exposure(
        tmp_path,
        mode="add",
        candidate_id=_LIVE_CANDIDATE,
        candidate_ids=[_LIVE_CANDIDATE],
    )
    label = _label(
        candidate_id=_LIVE_CANDIDATE,
        exposure_artifact_id=artifact_id,
    )
    seen = _stub_catalog(monkeypatch, tmp_path, rally)

    with pytest.raises(validation.SourceBindingError, match="catalog source"):
        validation._catalog_sources(
            tmp_path,
            [_RALLY_ID],
            [validation._label_source_claim(label)],
        )
    assert seen[0]["candidate_id"] == _LIVE_CANDIDATE


def test_selected_only_counterfactual_gate_still_rejects_unselected_pool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact_id, receipt, rally = _write_exposure(
        tmp_path,
        mode="add",
        candidate_id=_LIVE_CANDIDATE,
    )
    label_source = _label(
        candidate_id=_LIVE_CANDIDATE,
        exposure_artifact_id=artifact_id,
    )
    claim = validation._label_source_claim(label_source)
    assert validation._exposure_source_claim(tmp_path, rally, claim) is True
    monkeypatch.setattr(
        distill,
        "_exposure_map",
        lambda _root: {
            (_HOST, _SESSION_ID_SHA256, _QUERY_SHA256): [receipt]
        },
    )
    assignment = {"revision": distill.ASSIGNMENT_REVISION, "kind": "counterfactual"}
    row = {
        "rally_id": _RALLY_ID,
        "candidate_id": _LIVE_CANDIDATE,
        "counterfactual_ref": artifact_id,
        "assignment": assignment,
        "assignment_authority": distill._training_assignment_authority(assignment),
    }
    label = {
        "rally_id": _RALLY_ID,
        "candidate_id": _LIVE_CANDIDATE,
        "exposure_artifact_id": artifact_id,
        "assignment": assignment,
    }
    assert distill._sealed_counterfactual_exposure_binding(
        tmp_path, row, label, rally
    ) is False


@pytest.mark.parametrize("feature_mismatch", [False, True])
def test_selected_gate_binds_same_exact_exposure_as_materialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, feature_mismatch: bool
) -> None:
    artifact_id, receipt, rally = _write_exposure(
        tmp_path, mode="remove", candidate_id=_SELECTED_CANDIDATE
    )
    rally["as_of"] = "2026-10-12T00:00:00Z"
    label = _label(
        candidate_id=_SELECTED_CANDIDATE,
        mode="remove",
        exposure_artifact_id=artifact_id,
        features=_features(coverage=0.25) if feature_mismatch else _features(),
    )
    assignment = {"revision": distill.ASSIGNMENT_REVISION, "kind": "counterfactual"}
    label["assignment"] = assignment
    row = {
        "rally_id": _RALLY_ID,
        "candidate_id": _SELECTED_CANDIDATE,
        "counterfactual_ref": artifact_id,
        "assignment_authority": distill._training_assignment_authority(assignment),
    }
    monkeypatch.setattr(
        distill, "_exposure_map",
        lambda _root: {(_HOST, _SESSION_ID_SHA256, _QUERY_SHA256): [receipt]},
    )
    monkeypatch.setattr(
        distill, "_materialization_snapshots",
        lambda *_args, **_kwargs: pytest.fail("CF must not rebind to historical FTS"),
    )

    assert distill._sealed_counterfactual_exposure_binding(
        tmp_path, row, label, rally
    ) is (not feature_mismatch)
