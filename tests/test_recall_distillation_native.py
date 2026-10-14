from __future__ import annotations

import json
from pathlib import Path

import pytest

from chronovisor.recall import recall_distillation_store as store
from chronovisor.recall.recall_distillation_luna import (
    LunaBulkBundle,
    native_output_sha256,
)
from chronovisor.recall.recall_distillation_native import (
    NATIVE_COHORT,
    NATIVE_PAYLOAD_REF_SCHEMA,
    NATIVE_PROFILE_CONTRACT_FILE,
    NATIVE_RECEIPTS_FILE,
    NativeDistillationError,
    current_contract_id,
    ensure_contract,
    import_bundle,
    read_contract,
    row_integrity,
    validate_label,
)


def _config(**overrides: object) -> dict[str, object]:
    return {
        "teacher_profile": "native-luna-single-v1",
        "native_luna_enabled": True,
        **overrides,
    }


def _source_rows(count: int = 2) -> list[dict[str, object]]:
    return [
        {
            "rally_id": f"rally-{index}",
            "candidate_id": f"candidate-{index}",
            "as_of_us": 1_000 + index,
            "input": {"token": index},
            "snapshot_sha256": f"{index + 1:064x}",
            "query_sha256": f"{index + 11:064x}",
            "context_sha256": [f"{index + 21:064x}"],
            "candidate_text_sha256": f"{index + 31:064x}",
            "split": "train" if index == 0 else "validation",
            "group_id": f"group-{index}",
            "features": {"candidate_rank": index},
            "assignment": {
                "revision": "native-luna-bulk-v1",
                "profile": "native-luna-single-v1",
                "split": "train" if index == 0 else "validation",
                "probe": False,
                "owner": "native-luna",
            },
        }
        for index in range(count)
    ]


def _complete_bundle(
    root: Path, *, count: int = 2, output_hash: str | None = None,
    assignment_overrides: dict[str, object] | None = None,
) -> LunaBulkBundle:
    sources = _source_rows(count)
    for source in sources:
        source["assignment"].update(assignment_overrides or {})
    bundle = LunaBulkBundle.create(
        root,
        sources,
        cohort=NATIVE_COHORT,
        split_plan_id="a" * 64,
        shard_count=1,
        batch_size=count,
    )
    request = bundle.next_batch(0)
    labels: list[dict[str, object]] = []
    for index, source in enumerate(request["records"]):
        label: dict[str, object] = {
            "task_id": source["task_id"],
            "input_sha256": source["input_sha256"],
            "teacher_profile": "native-luna-single-v1",
            "native_luna": True,
            "verdict": "relevant" if index == 0 else "irrelevant",
            "confidence": 0.9,
            "reason_code": "direct_match" if index == 0 else "not_relevant",
            "rationale": None,
        }
        # Submit a valid immutable batch first.  A caller-supplied hash is
        # applied below only to exercise the importer's re-open validation.
        output_sha = native_output_sha256(label)
        label["native_receipt"] = {
            "task_id": source["task_id"],
            "output_sha256": output_sha,
            "assignment_id": f"assignment-{index}",
            "run_id": f"run-{index}",
        }
        labels.append(label)
    bundle.submit(0, cursor=0, records=labels)
    bundle.finalize()
    if output_hash is not None:
        batch_path = bundle.root / "shards" / "00" / "batches" / "batch-000000.json"
        payload = json.loads(batch_path.read_text(encoding="utf-8"))
        payload["records"][0]["native_receipt"]["output_sha256"] = output_hash
        batch_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    return bundle


def test_contract_is_closed_and_native_expiry_is_permanent(tmp_path: Path) -> None:
    root = tmp_path / "root"
    assert current_contract_id(root) == ""
    assert read_contract(root) == {}
    contract = ensure_contract(root, _config())

    assert current_contract_id(root) == contract["artifact_id"]
    assert read_contract(root)["expires_at"] == ""
    assert (store.distillation_dir(root) / NATIVE_PROFILE_CONTRACT_FILE).is_file()
    with pytest.raises(NativeDistillationError, match="enabled"):
        ensure_contract(root, _config(native_luna_enabled=False))
    with pytest.raises(NativeDistillationError, match="profile"):
        ensure_contract(root, _config(teacher_profile="deepseek-v4-flash-single-v1"))


def test_import_seals_receipts_and_sanitizes_production_rows(tmp_path: Path) -> None:
    root = tmp_path / "root"
    bundle = _complete_bundle(tmp_path / "bundle")
    ensure_contract(root, _config())

    result = import_bundle(root, bundle.root)
    assert result["labels"] == 2
    assert result["relevant"] == 1
    assert result["irrelevant"] == 1
    assert result["receipts"] == 2
    assert result["receipts_appended"] == 2
    assert result["labels_appended"] == 2

    receipt_rows = store.read_chain(
        store.distillation_dir(root) / NATIVE_RECEIPTS_FILE
    )
    label_rows = store.read_chain(store.distillation_dir(root) / store.LABEL_LEDGER_FILE)
    assert len(receipt_rows) == len(label_rows) == 2
    assert all(row["receipt_schema"].startswith("chronovisor.recall-native-") for row in receipt_rows)
    assert all(validate_label(root, row) for row in label_rows)
    for row in label_rows:
        assert row["payload_source"]["schema"] == NATIVE_PAYLOAD_REF_SCHEMA
        assert "input" not in row and "rationale" not in row and "native_receipt" not in row
    production_bytes = b"\n".join(
        json.dumps(row, sort_keys=True).encode("utf-8") for row in [*receipt_rows, *label_rows]
    )
    assert b'"query"' not in production_bytes
    assert b'"context"' not in production_bytes


def test_import_is_idempotent_by_work_and_receipt_identity(tmp_path: Path) -> None:
    root = tmp_path / "root"
    bundle = _complete_bundle(tmp_path / "bundle", count=1)
    ensure_contract(root, _config())

    first = import_bundle(root, bundle.root)
    second = import_bundle(root, bundle.root)
    assert first["receipts_appended"] == first["labels_appended"] == 1
    assert first["last_label_at"]
    assert second["receipts_appended"] == second["labels_appended"] == 0
    assert second["last_label_at"] == ""
    assert second["label_chain_head_sha256"] == first["label_chain_head_sha256"]
    assert second["receipt_chain_head_sha256"] == first["receipt_chain_head_sha256"]


def test_import_rejects_self_reported_native_output_hash(tmp_path: Path) -> None:
    root = tmp_path / "root"
    bundle = _complete_bundle(tmp_path / "bundle", count=1, output_hash="b" * 64)
    ensure_contract(root, _config())

    with pytest.raises(NativeDistillationError, match="bundle is invalid"):
        import_bundle(root, bundle.root)
    assert not (store.distillation_dir(root) / NATIVE_RECEIPTS_FILE).exists()


def test_row_integrity_rejects_raw_fields_and_tampering(tmp_path: Path) -> None:
    root = tmp_path / "root"
    bundle = _complete_bundle(tmp_path / "bundle", count=1)
    ensure_contract(root, _config())
    import_bundle(root, bundle.root)
    row = store.read_chain(store.distillation_dir(root) / store.LABEL_LEDGER_FILE)[0]

    assert row_integrity(row, root=root)
    raw_row = {**row, "query": "private"}
    assert not row_integrity(raw_row, root=root)
    tampered = {**row, "payload_source": {**row["payload_source"], "input_sha256": "c" * 64}}
    assert not row_integrity(tampered, root=root)

    receipt_path = store.distillation_dir(root) / NATIVE_RECEIPTS_FILE
    receipt_path.chmod(0o644)
    assert not validate_label(root, row)


def test_explicit_assignment_repair_preserves_original_and_native_responses(tmp_path: Path) -> None:
    from scripts.recall_luna_repair_assignment import (
        _frozen_hashes,
        repair_assignment_bundle,
    )

    root = tmp_path / "root"
    ensure_contract(root, _config())
    original = _complete_bundle(tmp_path / "original", assignment_overrides={
        "revision": "native-luna-assignment-v1",
    })
    before = _frozen_hashes(original.root)
    with pytest.raises(NativeDistillationError, match="assignment identity"):
        import_bundle(root, original.root)
    repaired_path = tmp_path / "repaired"
    proof = repair_assignment_bundle(original.root, repaired_path)
    repaired = LunaBulkBundle.open(repaired_path)
    assert _frozen_hashes(original.root) == before
    assert original.labels() == repaired.labels()
    assert proof["records"] == 2 and proof["quarantined"] == 0
    assert proof["original_source_snapshot_sha256"] != proof["repaired_source_snapshot_sha256"]
    result = import_bundle(root, repaired_path)
    assert result["labels_appended"] == 2
    assert result["truth_authority"] == "teacher_only_not_verified"
    assert import_bundle(root, repaired_path)["labels_appended"] == 0
    with pytest.raises(NativeDistillationError, match="assignment identity"):
        import_bundle(root, original.root)


@pytest.mark.parametrize("invalid", [
    {"revision": "unknown"}, {"owner": "other"}, {"profile": "other"},
    {"split": "test"}, {"probe": True}, {"extra": "unexpected"},
])
def test_assignment_repair_refuses_other_identity_changes(tmp_path: Path, invalid: dict) -> None:
    from scripts.recall_luna_repair_assignment import repair_assignment_bundle

    original = _complete_bundle(tmp_path / "original", assignment_overrides={
        "revision": "native-luna-assignment-v1", **invalid,
    })
    with pytest.raises(ValueError, match="exact legacy assignment repair"):
        repair_assignment_bundle(original.root, tmp_path / "repaired")
    assert not (tmp_path / "repaired").exists()


def test_assignment_repair_refuses_changed_frozen_input(tmp_path: Path) -> None:
    from scripts.recall_luna_repair_assignment import repair_assignment_bundle

    original = _complete_bundle(tmp_path / "original", assignment_overrides={
        "revision": "native-luna-assignment-v1",
    })
    path = original.root / "snapshot/source.jsonl"
    path.write_text(path.read_text().replace('"token":0', '"token":9'))
    with pytest.raises(ValueError, match="hash changed"):
        repair_assignment_bundle(original.root, tmp_path / "repaired")
    assert not (tmp_path / "repaired").exists()
