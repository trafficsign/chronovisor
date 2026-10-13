from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from chronovisor.ops.health import _native_luna_supervisor_health
from chronovisor.recall.recall_distillation_luna import LunaBulkBundle
from scripts import recall_luna_supervisor as supervisor
from scripts.recall_luna_worker_io import bind_decisions


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


def test_scheduled_blind_workers_have_independent_run_receipts(tmp_path: Path) -> None:
    bundle = LunaBulkBundle.create(tmp_path / "bundle", [{
        "rally_id": "rally", "candidate_id": "candidate", "as_of_us": 1000,
        "input": {"query": "synthetic same evidence"}, "repeat_pair_id": "pair",
        "fixed_repeat": True, "order_swap": True, "blind_order": order,
    } for order in ("a_first", "b_first")], shard_count=2)
    prompt = supervisor.worker_prompt(bundle.root, [0, 1], "round-a", 1)
    flags = re.search(r"--assignment-id (\S+) --run-id (\S+)", prompt)
    assert flags is not None
    for shard in range(2):
        request = bundle.next_batch(shard, limit=1)
        decision = {
            "task_id": request["records"][0]["task_id"], "verdict": "relevant",
            "confidence": 0.9, "reason_code": "direct_match",
        }
        labels = bind_decisions(
            request, [decision],
            assignment_id=flags[1].replace("SHARD", str(shard)),
            run_id=flags[2].replace("SHARD", str(shard)),
        )
        bundle.submit(shard, cursor=0, records=labels)
    assert bundle.finalize()["blind_repeat"] == {"pairs": 1, "stable": 1, "unstable": 0}


def test_idle_supervisor_does_not_recapture_the_whole_catalog_each_tick(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "config.toml").write_text(
        '[recall.distillation]\nenabled=true\nnative_luna_enabled=true\nteacher_profile="native-luna-single-v1"\n'
    )
    monkeypatch.setattr(supervisor, "inventory", lambda _root: [])
    calls = []

    def no_new_source(*args):
        calls.append(args[1])
        return {"status": "ready", "reason": "no_unlabeled_source"}

    monkeypatch.setattr(supervisor, "run_daily_operation", no_new_source)
    args = argparse.Namespace(root=tmp_path, max_workers=10, once=True, poll_seconds=1)
    supervisor.supervise(args)
    supervisor.supervise(args)
    assert calls == ["run"]


@pytest.mark.parametrize("elapsed", [1, 1801])
def test_stale_progress_allows_a_fresh_retry_but_still_bounds_a_stalled_round(
    tmp_path: Path, monkeypatch, elapsed: int,
) -> None:
    private = tmp_path / "runtime/recall-luna-bulk"
    private.mkdir(parents=True)
    status = private / "supervisor-status.json"
    status.write_text(json.dumps({"last_progress_at": "2000-01-01T00:00:00Z"}))
    start = datetime(2026, 10, 14, tzinfo=UTC)
    ticks = iter([start, start, start + timedelta(seconds=elapsed)])

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return next(ticks)

    class Child:
        pid = 12345
        returncode = 0
        polls = 0

        def poll(self):
            self.polls += 1
            return None if self.polls == 1 else self.returncode

    monkeypatch.setattr(supervisor, "datetime", Clock)
    monkeypatch.setattr(supervisor, "cli_command", lambda *_: ["synthetic"])
    monkeypatch.setattr(supervisor.subprocess, "Popen", lambda *_args, **_kwargs: Child())
    monkeypatch.setattr(supervisor.time, "sleep", lambda _: None)
    monkeypatch.setattr(supervisor.daily.distill, "load_distillation_config", lambda _: SimpleNamespace(enabled=True, native_luna_enabled=True))
    args = argparse.Namespace(root=tmp_path, records_per_round=1, codex_home=tmp_path, codex="synthetic", poll_seconds=1)
    result = supervisor.run_round(args, {"path": str(tmp_path / "bundle")}, [0], lambda *_args, **_kwargs: None)
    assert result == (0 if elapsed == 1 else 124)
    assert json.loads(status.read_text())["last_progress_at"] == "2000-01-01T00:00:00Z"
