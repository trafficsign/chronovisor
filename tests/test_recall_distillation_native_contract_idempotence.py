from __future__ import annotations

from pathlib import Path

import pytest

from chronovisor.recall import recall_distillation_ledger as ledger
from chronovisor.recall import recall_distillation_materialization as materialization
from chronovisor.recall import recall_distillation_native as native
from chronovisor.recall import recall_distillation_store as store


def _config(**overrides: object) -> dict[str, object]:
    return {
        "teacher_profile": native.NATIVE_LUNA_SINGLE_PROFILE,
        "native_luna_enabled": True,
        **overrides,
    }


def _projection(root: Path) -> materialization.Projection:
    """Create the smallest sealed projection that binds the source state."""

    path = materialization._path(root)
    metadata = {"contracts": {"files": {}}}
    connection = ledger.connect(path)
    try:
        materialization._schema(connection)
        materialization._save(connection, path, metadata)
    finally:
        connection.close()
    source_state = materialization._source_state(root, metadata["contracts"])
    return materialization.Projection(root, metadata, source_state)


def test_same_contract_ensure_does_not_invalidate_source_view(tmp_path: Path) -> None:
    root = tmp_path / "root"
    config = _config()
    first = native.ensure_contract(root, config)
    view = _projection(root)
    source_state = materialization._source_state(root, {"files": {}})
    pointer = native._contract_pointer_path(root)
    contract_path = native._contract_dir(root) / f"{first['artifact_id']}.json"
    checkpoint_path = materialization._path(root).with_suffix(
        materialization._path(root).suffix + ".checkpoint.json"
    )
    pointer_state = ledger.file_state(pointer)
    pointer_bytes = pointer.read_bytes()
    contract_state = ledger.file_state(contract_path)
    contract_bytes = contract_path.read_bytes()
    checkpoint_state = ledger.file_state(checkpoint_path)
    checkpoint_bytes = checkpoint_path.read_bytes()

    second = native.ensure_contract(root, config)

    assert second["artifact_id"] == first["artifact_id"]
    assert ledger.file_state(pointer) == pointer_state
    assert pointer.read_bytes() == pointer_bytes
    assert ledger.file_state(contract_path) == contract_state
    assert contract_path.read_bytes() == contract_bytes
    assert ledger.file_state(checkpoint_path) == checkpoint_state
    assert checkpoint_path.read_bytes() == checkpoint_bytes
    assert materialization._source_state(root, {"files": {}}) == source_state
    view.check()
    view.close()


def test_changed_config_repoints_contract_and_recovers_invalid_pointer(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    first = native.ensure_contract(root, _config())
    changed = native.ensure_contract(root, _config(teacher_claim_limit=7))

    assert changed["artifact_id"] != first["artifact_id"]
    assert native.current_contract_id(root) == changed["artifact_id"]
    assert native.read_contract(root)["teacher_claim_limit"] == 7

    store.write_sealed_state(
        native._contract_pointer_path(root),
        {
            "kind": "not-a-native-contract-pointer",
            "profile_contract_id": "f" * 64,
        },
    )
    recovered = native.ensure_contract(root, _config(teacher_claim_limit=7))

    assert recovered["artifact_id"] == changed["artifact_id"]
    assert native.current_contract_id(root) == changed["artifact_id"]
    assert native.read_contract(root)["teacher_claim_limit"] == 7


@pytest.mark.parametrize("pointer_mode", ["missing", "wrong_id"])
def test_invalid_pointer_is_resealed_for_existing_contract(
    tmp_path: Path, pointer_mode: str
) -> None:
    root = tmp_path / "root"
    contract = native.ensure_contract(root, _config())
    pointer = native._contract_pointer_path(root)
    if pointer_mode == "missing":
        pointer.unlink()
    else:
        store.write_sealed_state(
            pointer,
            {
                "kind": "native-luna-profile-contract-pointer",
                "profile_contract_id": "0" * 64,
            },
        )

    recovered = native.ensure_contract(root, _config())

    assert recovered["artifact_id"] == contract["artifact_id"]
    assert native.current_contract_id(root) == contract["artifact_id"]
    assert native.read_contract(root)["artifact_id"] == contract["artifact_id"]
