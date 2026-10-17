from __future__ import annotations

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


def _page(root: Path, page_id: str = "alpha") -> tuple[str, str, Path]:
    path = root / "pages" / f"{page_id}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "---\n"
        f"title: {page_id}\n"
        "status: stable\n"
        "type: knowledge\n"
        "updated: 2026-10-12\n"
        "---\n\n"
        f"# {page_id}\n\nfixture page body\n",
        encoding="utf-8",
    )
    registry = PageRegistry(root)
    registry.ensure_manifest()
    resolved = registry.resolve(page_id)
    assert resolved is not None
    path = registry.path_for(page_id, require_stable=True)
    assert path is not None
    page_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    return str(resolved["uid"]), page_sha, path


def _fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    as_of: str = "2026-10-12T03:38:56.107Z",
    include_context_receipt: bool = True,
) -> tuple[Path, dict[str, Any], dict[str, Any], str, Path]:
    root = tmp_path / "root"
    root.mkdir()
    page_uid, page_sha, page_path = _page(root)
    page_id = "alpha"
    rendered = json.dumps(
        {
            "items": [
                {
                    "page_id": page_id,
                    "source_ref": {
                        "doc_id": f"page-record:{page_sha}",
                        "uid": page_uid,
                        "sha256": page_sha,
                        "generation_id": "fixture-generation",
                        "byte_start": 0,
                        "byte_end": page_path.stat().st_size,
                    },
                }
            ]
        },
        separators=(",", ":"),
    )
    rendered_sha = hashlib.sha256(rendered.encode()).hexdigest()
    features = {
        "query_chargram_coverage": 0.5,
        "candidate_chargram_precision": 0.75,
    }
    candidate_ref = {
        "candidate_id": page_id,
        "page_id": page_id,
        "page_content_sha256": page_sha,
        "rendered_context": rendered,
        "rendered_context_sha256": rendered_sha,
    }
    feature_row = {"candidate_id": page_id, "features": features}
    payload = {
        "kind": "exact-rendered-exposure",
        "decision_id": "fixture-decision",
        "host": "codex",
        "session_id_sha256": "a" * 64,
        "query_semantic_sha256": "b" * 64,
        "policy_id": "p" * 64,
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
        "runtime_observation_sha256": "r" * 64,
        "render_sha256": "e" * 64,
        "renderer_revision": "recall-card-v1",
        "context_style": "default",
        "candidate_snapshot_sha256": "a" * 64,
        "observed_at": "2026-10-12T03:39:00.062120+00:00",
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
        "candidate_snapshot_sha256": "a" * 64,
    }
    rally = {
        "rally_id": "rally-fixture",
        "host": "codex",
        "session_id_sha256": "a" * 64,
        "query_sha256": "b" * 64,
        "as_of": as_of,
        "exposure_receipts": [receipt],
    }
    canonical = {
        "decision_id": "fixture-decision",
        "policy_id": "p" * 64,
        "candidate_ids": [page_id],
        "candidate_snapshot_sha256": "a" * 64,
        "candidate_feature_snapshot_sha256": payload[
            "candidate_feature_snapshot_sha256"
        ],
        "observed_at": payload["observed_at"],
        "record_sha256": "c" * 64,
        "exposure_artifact_id": artifact_id,
    }
    monkeypatch.setattr(
        distill,
        "_exposure_map",
        lambda _root: {
            ("codex", "a" * 64, "b" * 64): [canonical],
        },
    )
    if include_context_receipt:
        context_receipt = {
            "schema_version": 1,
            "renderer_protocol": "recall-result-context-v1",
            "context_style": "default",
            "rendered_context": "fixture",
            "rendered_context_sha256": hashlib.sha256(b"fixture").hexdigest(),
            "page_bindings": [
                {
                    "page_id": page_id,
                    "page_uid": page_uid,
                    "content_sha256": page_sha,
                }
            ],
        }
        context_receipt["receipt_sha256"] = hashlib.sha256(
            json.dumps(
                context_receipt,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        log_path = root / "recall" / "recall-log.jsonl"
        log_path.parent.mkdir(parents=True)
        log_path.write_text(
            json.dumps(
                {
                    "decision_id": "fixture-decision",
                    "context_receipt": context_receipt,
                },
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
        )
    snapshot = {
        "schema": catalog.CANDIDATE_SNAPSHOT_SCHEMA,
        "rally_id": rally["rally_id"],
        "as_of": as_of,
        "retriever_revision": "historical-fts-v1",
        "feature_revision": distill.TEXT_FEATURE_REVISION,
        "query_feature_text_sha256": "d" * 64,
        "candidates": [
            {
                "candidate_id": "historical-atom",
                "rank": 1,
                "text_sha256": "e" * 64,
            }
        ],
    }
    snapshot["snapshot_sha256"] = canonical_json.canonical_json_sha256_strict(
        {key: value for key, value in snapshot.items() if key != "snapshot_sha256"}
    )
    return root, rally, snapshot, page_sha, page_path


def _validate_snapshot_row(snapshot: dict[str, Any]) -> None:
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


def test_augment_snapshot_adds_verified_cf_page_row_and_catalog_accepts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, rally, snapshot, page_sha, _ = _fixture(tmp_path, monkeypatch)
    result = capture.augment_candidate_snapshot(root, rally, snapshot)

    assert result != snapshot
    assert result["retriever_revision"] == "historical-fts-v1+exact-exposure-page-v1"
    assert result["counterfactual_capture"]["status"] == "ready"
    assert len(result["candidates"]) == 2
    page = result["candidates"][-1]
    assert page["source_kind"] == "exact-exposure-page-v1"
    assert page["text_sha256"] == page_sha
    assert page["page_content_sha256"] == page_sha
    assert "features" not in page
    assert result["candidates"][0]["candidate_id"] == "historical-atom"
    _validate_snapshot_row(result)


def test_augment_snapshot_fails_closed_on_page_hash_without_touching_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, rally, snapshot, _page_sha, page_path = _fixture(tmp_path, monkeypatch)
    original = json.loads(json.dumps(snapshot))
    page_path.write_text(page_path.read_text(encoding="utf-8") + "changed\n", encoding="utf-8")

    result = capture.augment_candidate_snapshot(root, rally, snapshot)

    assert result["counterfactual_capture"] == {
        "status": "unavailable",
        "reason": "page_hash_mismatch",
    }
    assert [item["candidate_id"] for item in result["candidates"]] == [
        item["candidate_id"] for item in original["candidates"]
    ]
    assert result["retriever_revision"] == snapshot["retriever_revision"]
    assert result["snapshot_sha256"] != snapshot["snapshot_sha256"]


def test_augment_snapshot_keeps_pre_cutoff_or_missing_exposure_byte_identical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, rally, snapshot, _page_sha, _ = _fixture(
        tmp_path, monkeypatch, as_of="2026-10-08T14:59:59Z"
    )
    assert capture.augment_candidate_snapshot(root, rally, snapshot) == snapshot

    rally_without_receipt = {**rally, "exposure_receipts": []}
    assert capture.augment_candidate_snapshot(root, rally_without_receipt, snapshot) == snapshot


def test_augment_snapshot_rejects_receipt_candidate_or_uid_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, rally, snapshot, _page_sha, _ = _fixture(tmp_path, monkeypatch)
    bad_rally = {
        **rally,
        "exposure_receipts": [
            {**rally["exposure_receipts"][0], "candidate_ids": ["other-page"]}
        ],
    }
    result = capture.augment_candidate_snapshot(root, bad_rally, snapshot)
    assert result["counterfactual_capture"]["reason"] == "receipt_mismatch"

    # A context receipt with a different UID is a source identity mismatch;
    # no page row is emitted even though the page bytes still match.
    log = root / "recall" / "recall-log.jsonl"
    record = json.loads(log.read_text(encoding="utf-8"))
    receipt = record["context_receipt"]
    receipt["page_bindings"][0]["page_uid"] = "wrong-uid"
    receipt["receipt_sha256"] = hashlib.sha256(
        json.dumps(
            {key: value for key, value in receipt.items() if key != "receipt_sha256"},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    log.write_text(json.dumps(record, separators=(",", ":")) + "\n", encoding="utf-8")
    result = capture.augment_candidate_snapshot(root, rally, snapshot)
    assert result["counterfactual_capture"]["reason"] == "context_receipt_mismatch"
