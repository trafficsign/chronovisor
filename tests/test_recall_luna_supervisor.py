from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path

import pytest

from chronovisor.ops.health import _native_luna_supervisor_health
from chronovisor.recall.recall_distillation_luna import LunaBulkBundle
from scripts import recall_luna_supervisor as supervisor


def make_bundle(root: Path) -> LunaBulkBundle:
    return LunaBulkBundle.create(root / "runtime/recall-luna-bulk/2026-10-13/bundle-A", [{
        "rally_id": f"rally-{i}", "candidate_id": f"candidate-{i}", "as_of_us": 1000 + i,
        "input": {"query": "synthetic only"}, "split": "train", "group_id": "group",
    } for i in range(2)], shard_count=2)


def test_supervisor_never_selects_a_shard_owned_by_another_worker(tmp_path: Path) -> None:
    bundle = make_bundle(tmp_path)
    path = bundle.root / "shards/00/.native-worker.lock"
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        row = supervisor.inventory(tmp_path)[0]
        assert supervisor.available_shards(row, 10) == [1]
    finally:
        os.close(descriptor)
    assert supervisor.available_shards(row, 10) == [0, 1]
    assert path.exists()


def test_supervisor_retries_a_failed_native_process_and_publishes_the_hold(tmp_path: Path, monkeypatch) -> None:
    make_bundle(tmp_path)
    (tmp_path / "config.toml").write_text(
        '[recall.distillation]\nenabled=true\nnative_luna_enabled=true\nteacher_profile="native-luna-single-v1"\n'
    )
    attempts = []

    def fail_round(args, bundle, shards, publish):
        attempts.append((bundle["completed"], shards))
        return 7

    monkeypatch.setattr(supervisor, "run_round", fail_round)
    args = argparse.Namespace(root=tmp_path, max_workers=10, once=True, poll_seconds=1)
    assert supervisor.supervise(args) == 1
    assert supervisor.supervise(args) == 1
    assert attempts == [(0, [0, 1]), (0, [0, 1])]
    state = json.loads((tmp_path / "runtime/recall-luna-bulk/supervisor-status.json").read_text())
    assert state["status"] == "retrying"
    assert state["last_error_code"] == "native_cli_failed"
    assert state["pending_total"] == 2
    diagnosis, alerts = _native_luna_supervisor_health(tmp_path)
    assert diagnosis["completed_total"] == 0
    assert alerts[0]["reason"] == "native_luna_executor_error"
    assert "synthetic only" not in json.dumps(state)


def test_supervisor_does_not_start_native_workers_after_explicit_disable(tmp_path: Path, monkeypatch) -> None:
    make_bundle(tmp_path)
    (tmp_path / "config.toml").write_text('[recall.distillation]\nenabled=false\n')
    monkeypatch.setattr(supervisor, "run_round", lambda *_: pytest.fail("disabled worker started"))
    args = argparse.Namespace(root=tmp_path, max_workers=10, once=True, poll_seconds=1)
    assert supervisor.supervise(args) == 1
    state = json.loads((tmp_path / "runtime/recall-luna-bulk/supervisor-status.json").read_text())
    assert state["status"] == "disabled"


def test_cli_role_must_keep_the_native_luna_identity(tmp_path: Path) -> None:
    role = tmp_path / "agents/luna_worker.toml"
    role.parent.mkdir()
    role.write_text('model="gpt-5.6-luna"\nmodel_reasoning_effort="max"\n')
    command = supervisor.cli_command("codex", tmp_path, tmp_path / "result.json")
    assert "--ignore-user-config" in command
    assert "gpt-6-astra" in command
    role.write_text('model="other-model"\nmodel_reasoning_effort="max"\n')
    with pytest.raises(ValueError, match="native_luna_role_identity_mismatch"):
        supervisor.cli_command("codex", tmp_path, tmp_path / "result.json")
