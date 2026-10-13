from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from chronovisor.recall import recall_distillation as distill
from chronovisor.recall import recall_distillation_catalog as catalog
from chronovisor.recall import recall_distillation_store as store
from chronovisor.recall import recall_distillation_validation as validation


def _source_files(root: Path) -> tuple[Path, ...]:
    paths = validation._runtime_paths(root)
    paths[0].parent.mkdir(parents=True, exist_ok=True)
    for path in paths:
        path.touch()
    return paths


def _labels() -> list[dict[str, Any]]:
    return [
        {
            "record_sha256": "a" * 64,
            "rally_id": "rally-a",
            "candidate_id": "candidate-a",
        },
        {
            "record_sha256": "b" * 64,
            "rally_id": "rally-b",
            "candidate_id": "candidate-b",
        },
    ]


def _stub_sources(
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
    *,
    catalog_enabled: bool,
    labels: list[dict[str, Any]] | None = None,
) -> dict[str, int]:
    labels = labels or _labels()
    label_path = store.distillation_dir(root) / store.LABEL_LEDGER_FILE
    calls = {"labels": 0, "rallies": 0, "snapshots": 0, "features": 0}

    def read_chain(path: Path) -> list[dict[str, Any]]:
        if path == label_path:
            calls["labels"] += 1
            return [dict(row) for row in labels]
        return []

    monkeypatch.setattr(store, "read_chain", read_chain)
    monkeypatch.setattr(
        catalog,
        "_read_catalog_checkpoint",
        lambda _root: {"catalog_watermark": "c"} if catalog_enabled else None,
    )

    def read_rallies(_root: Path, ids: tuple[str, ...]) -> list[dict[str, Any]]:
        calls["rallies"] += 1
        return [{"rally_id": rally_id} for rally_id in ids]

    def read_snapshots(
        _root: Path, _ledger: Path, claims: list[dict[str, str]]
    ) -> dict[str, dict[str, Any]]:
        calls["snapshots"] += 1
        return {
            claim["rally_id"]: {
                "rally_id": claim["rally_id"],
                "candidates": [{"candidate_id": claim.get("candidate_id", "")}],
            }
            for claim in claims
        }

    monkeypatch.setattr(catalog, "rallies", read_rallies)
    monkeypatch.setattr(catalog, "read_candidate_snapshots", read_snapshots)
    monkeypatch.setattr(
        distill,
        "_materialization_rallies",
        lambda _root, _supplied: {
            "rally-a": {"rally_id": "rally-a"},
            "rally-b": {"rally_id": "rally-b"},
            "rally-extra": {"rally_id": "rally-extra"},
        },
    )
    monkeypatch.setattr(
        distill,
        "_materialization_snapshots",
        lambda _root, _supplied: {
            "rally-a": {"rally_id": "rally-a"},
            "rally-b": {"rally_id": "rally-b"},
            "rally-extra": {"rally_id": "rally-extra"},
        },
    )

    def feature_pairs(
        _root: Path,
        rallies: dict[str, Any],
        snapshots: dict[str, Any],
    ) -> tuple[dict[tuple[str, str], dict[str, float]], dict[str, dict[str, Any]]]:
        calls["features"] += 1
        assert set(rallies) == {"rally-a", "rally-b"}
        assert set(snapshots) == {"rally-a", "rally-b"}
        return {
            ("rally-a", "candidate-a"): {"score": 1.0},
            ("rally-b", "candidate-b"): {"score": 0.0},
        }, {rally_id: {} for rally_id in rallies}

    monkeypatch.setattr(distill, "_materialization_feature_pairs", feature_pairs)
    validation.clear_cache(root)
    return calls


def test_binding_inputs_uses_catalog_and_reuses_one_root_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _source_files(tmp_path)
    calls = _stub_sources(monkeypatch, tmp_path, catalog_enabled=True)

    first = validation.binding_inputs(tmp_path)
    second = validation.binding_inputs(tmp_path)

    assert first is second
    assert calls == {"labels": 1, "rallies": 1, "snapshots": 1, "features": 1}
    assert set(first[0]) == {"a" * 64, "b" * 64}
    assert set(first[1]) == {"rally-a", "rally-b"}


def test_binding_inputs_falls_back_to_legacy_only_without_catalog_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _source_files(tmp_path)
    calls = _stub_sources(monkeypatch, tmp_path, catalog_enabled=False)

    values = validation.binding_inputs(tmp_path)

    assert set(values[1]) == {"rally-a", "rally-b"}
    assert calls == {"labels": 1, "rallies": 0, "snapshots": 0, "features": 1}


def test_binding_inputs_invalidates_on_append_tamper_and_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _source_files(tmp_path)
    calls = _stub_sources(monkeypatch, tmp_path, catalog_enabled=True)
    candidate_path = store.distillation_dir(tmp_path) / "candidate-ledger.jsonl"

    validation.binding_inputs(tmp_path)
    candidate_path.write_bytes(b"append")
    validation.binding_inputs(tmp_path)
    assert calls["labels"] == 2

    original_rallies = catalog.rallies
    changed = {"count": 0}

    def mutate_during_read(root: Path, ids: tuple[str, ...]) -> list[dict[str, Any]]:
        if changed["count"] == 0:
            changed["count"] += 1
            paths[2].write_bytes(b"tamper")
        return original_rallies(root, ids)

    monkeypatch.setattr(catalog, "rallies", mutate_during_read)
    candidate_path.write_bytes(b"tamper-trigger")
    with pytest.raises(validation.SourceBindingError, match="source changed"):
        validation.binding_inputs(tmp_path)
    assert calls["features"] == 3

    monkeypatch.setattr(catalog, "rallies", original_rallies)
    candidate_path.unlink()
    candidate_path.write_bytes(b"replacement")
    validation.binding_inputs(tmp_path)
    assert calls["labels"] == 4


def test_binding_inputs_does_not_cache_invalid_label_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _source_files(tmp_path)
    duplicate = _labels() + [_labels()[0]]
    calls = _stub_sources(
        monkeypatch, tmp_path, catalog_enabled=True, labels=duplicate
    )

    with pytest.raises(validation.SourceBindingError, match="duplicated"):
        validation.binding_inputs(tmp_path)
    assert calls["labels"] == 1

    calls = _stub_sources(monkeypatch, tmp_path, catalog_enabled=True)
    validation.binding_inputs(tmp_path)
    assert calls["labels"] == 1
