from __future__ import annotations

import inspect
import json
import stat
from pathlib import Path

from chronovisor.classification import classification_fixture_contract as contract
from chronovisor.classification import classification_fixture_set as fixture_set
from chronovisor.core.jsonl_write import write_jsonl_atomic as fixture_writer


def test_contract_preserves_legacy_schema_hash_and_dto_behavior(tmp_path: Path) -> None:
    assert contract.DISABLED_BASELINE_SCHEMA == fixture_set.DISABLED_BASELINE_SCHEMA
    assert contract.INFERENCE_DTO_SCHEMA == fixture_set.INFERENCE_DTO_SCHEMA
    assert contract.GOLD_FIELD_PREFIXES == fixture_set.GOLD_FIELD_PREFIXES

    payload = b"classification fixture bytes\n\x00"
    source = tmp_path / "source.bin"
    source.write_bytes(payload)
    assert contract.sha256_bytes(payload) == fixture_set.sha256_bytes(payload)
    assert contract.sha256_file(source) == fixture_set.sha256_file(source)

    row = {
        "uid": "uid-1",
        "title": "AI",
        "gold_primary_notation": "004.8",
        "adjudication_status": "accepted",
        "fixture_split": "holdout",
        "fixture_rank": 1,
        "nested": {"preserved": True},
    }
    assert contract.inference_dto(row) == fixture_set.inference_dto(row)


def test_contract_preserves_legacy_jsonl_bytes(tmp_path: Path) -> None:
    rows = [
        {"z": 1, "日本語": "分類"},
        {"nested": {"b": 2, "a": 1}, "flag": True},
    ]
    contract_path = tmp_path / "contract" / "rows.jsonl"
    implementation_path = tmp_path / "implementation" / "rows.jsonl"

    contract.write_jsonl(contract_path, rows)
    fixture_writer(implementation_path, rows)

    assert inspect.signature(contract.write_jsonl) == inspect.signature(fixture_writer)
    assert contract_path.read_bytes() == implementation_path.read_bytes()
    assert [json.loads(line) for line in contract_path.read_text().splitlines()] == rows
    assert contract_path.read_text().splitlines()[0] == '{"z": 1, "日本語": "分類"}'
    assert contract_path.read_bytes().endswith(b"\n")
    assert stat.S_IMODE(contract_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(implementation_path.stat().st_mode) == 0o600

    old_inode = contract_path.stat().st_ino
    contract.write_jsonl(contract_path, [{"replacement": True}])
    assert contract_path.stat().st_ino != old_inode
    assert contract_path.read_bytes() == b'{"replacement": true}\n'
    assert not list(contract_path.parent.glob(f".{contract_path.name}.*"))


def test_contract_writer_fsyncs_before_atomic_replace(
    tmp_path: Path, monkeypatch
) -> None:
    events: list[str] = []
    original_fsync = contract.os.fsync
    original_replace = contract.os.replace

    def observed_fsync(descriptor: int) -> None:
        events.append("fsync")
        original_fsync(descriptor)

    def observed_replace(source: Path, target: Path) -> None:
        events.append("replace")
        original_replace(source, target)

    monkeypatch.setattr(contract.os, "fsync", observed_fsync)
    monkeypatch.setattr(contract.os, "replace", observed_replace)
    contract.write_jsonl(tmp_path / "rows.jsonl", [{"value": 1}])

    assert events == ["fsync", "replace"]
