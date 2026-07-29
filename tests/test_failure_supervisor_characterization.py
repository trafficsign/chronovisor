from __future__ import annotations

import json
from pathlib import Path

from chronovisor.core import store
from chronovisor.decision import failure_supervisor


def test_corrupt_failure_state_fails_closed_to_empty_snapshot(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(store, "CHRONOVISOR_ROOT", tmp_path)
    path = tmp_path / "runtime" / "failures" / "state.json"
    path.parent.mkdir(parents=True)
    path.write_text("{broken", encoding="utf-8")

    assert failure_supervisor._load_state() == {"failures": {}}
    assert path.read_text(encoding="utf-8") == "{broken"


def test_save_failure_state_preserves_utf8_and_trailing_newline(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(store, "CHRONOVISOR_ROOT", tmp_path)
    payload = {"failures": {"raw-a.md": {"error": "意味的な失敗"}}}

    failure_supervisor._save_state(payload)

    path = tmp_path / "runtime" / "failures" / "state.json"
    text = path.read_text(encoding="utf-8")
    assert text.endswith("\n")
    assert "意味的な失敗" in text
    assert json.loads(text) == payload


def test_group_snapshot_is_sorted_and_does_not_create_lock_file(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(store, "CHRONOVISOR_ROOT", tmp_path)
    failures = tmp_path / "runtime" / "failures"
    failures.mkdir(parents=True)
    packet = tmp_path / "review" / "packet.json"
    failure_supervisor._save_state(
        {
            "failures": {
                "z.md": {"packet_path": str(packet), "attempts": 2},
                "a.md": {"packet_path": str(packet), "attempts": 1},
                "other.md": {"packet_path": str(tmp_path / "other.json")},
            }
        }
    )

    snapshot = failure_supervisor.operational_failure_group_snapshot(packet)

    assert [name for name, _entry in snapshot] == ["a.md", "z.md"]
    assert not (failures / "state.lock").exists()
