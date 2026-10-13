from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import subprocess
import sys
import time
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


class FakeChild:
    """Small deterministic stand-in for the native Codex parent process."""

    _next_pid = 1000

    def __init__(self, *results: int | None) -> None:
        self.pid = FakeChild._next_pid
        FakeChild._next_pid += 1
        self._results = list(results) or [None]
        self.terminated = False
        self.waited = False

    def poll(self) -> int | None:
        if len(self._results) > 1:
            return self._results.pop(0)
        return self._results[0]

    def wait(self, timeout: int | None = None) -> int:
        self.waited = True
        result = self.poll()
        return 0 if result is None else result

    def terminate(self) -> None:
        self.terminated = True


def fake_row(path: Path, shards: int, completed: int = 0) -> dict:
    return {
        "path": str(path),
        "completed": completed,
        "remaining": shards - completed,
        "imported": False,
        "source_snapshot_sha256": "snapshot",
        "shards": [
            {"shard": shard, "completed": 0, "remaining": 1}
            for shard in range(shards)
        ],
    }


def fake_round(bundle: dict, shard: int, child: FakeChild, *, completed: int = 0) -> supervisor.NativeRound:
    return supervisor.NativeRound(
        child=child,
        bundle_path=bundle["path"],
        shard=shard,
        round_id=f"round-{Path(bundle['path']).name}-{shard}-{child.pid}",
        completed=completed,
        initial_completed=completed,
        last_progress=0.0,
    )


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
    bundle_path = str(tmp_path / "runtime/recall-luna-bulk/2026-10-13/bundle-A")
    row = fake_row(Path(bundle_path), 2)

    def fail_round(args, bundle, shard):
        attempts.append((bundle["path"], shard))
        return fake_round(bundle, shard, FakeChild(7))

    monkeypatch.setattr(supervisor, "inventory", lambda _root: [row])
    monkeypatch.setattr(supervisor, "available_shards", lambda _row, _limit: [0, 1])
    monkeypatch.setattr(supervisor, "start_round", fail_round)
    monkeypatch.setattr(supervisor, "stop_round", lambda job: None)
    monkeypatch.setattr(supervisor.time, "sleep", lambda _: None)
    args = argparse.Namespace(root=tmp_path, max_workers=10, once=True, poll_seconds=1)
    assert supervisor.supervise(args) == 1
    assert attempts == [(bundle_path, 0), (bundle_path, 1)]
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
    monkeypatch.setattr(supervisor, "start_round", lambda *_: pytest.fail("disabled worker started"))
    args = argparse.Namespace(root=tmp_path, max_workers=10, once=True, poll_seconds=1)
    assert supervisor.supervise(args) == 1
    state = json.loads((tmp_path / "runtime/recall-luna-bulk/supervisor-status.json").read_text())
    assert state["status"] == "disabled"


def test_disable_stops_active_rounds_without_advancing_progress_watermark(tmp_path: Path, monkeypatch) -> None:
    private = tmp_path / "runtime/recall-luna-bulk"
    private.mkdir(parents=True)
    old_progress = "2026-10-14T00:00:00Z"
    (private / "supervisor-status.json").write_text(json.dumps({
        "completed_total": 0,
        "last_progress_at": old_progress,
    }))
    row = fake_row(tmp_path / "bundle-A", 1)
    child = FakeChild(None)
    starts = []
    stopped = []
    configs = iter([
        SimpleNamespace(
            enabled=True,
            native_luna_enabled=True,
            teacher_profile=supervisor.daily.DEFAULT_TEACHER_PROFILE,
        ),
        SimpleNamespace(
            enabled=False,
            native_luna_enabled=True,
            teacher_profile=supervisor.daily.DEFAULT_TEACHER_PROFILE,
        ),
    ])

    def start(args, bundle, shard):
        starts.append(shard)
        return fake_round(bundle, shard, child)

    monkeypatch.setattr(supervisor, "inventory", lambda _root: [row])
    monkeypatch.setattr(supervisor, "available_shards", lambda _row, _limit: [0])
    monkeypatch.setattr(supervisor, "start_round", start)
    monkeypatch.setattr(supervisor, "stop_round", lambda job: stopped.append(job.shard))
    monkeypatch.setattr(supervisor.daily.distill, "load_distillation_config", lambda _path: next(configs))
    monkeypatch.setattr(supervisor.time, "sleep", lambda _seconds: None)
    args = argparse.Namespace(
        root=tmp_path, max_workers=1, records_per_round=100,
        codex_home=tmp_path, codex="synthetic", poll_seconds=1, once=True,
    )

    assert supervisor.supervise(args) == 1
    assert starts == [0]
    assert stopped == [0]
    state = json.loads((private / "supervisor-status.json").read_text())
    assert state["status"] == "disabled"
    assert state["active_workers"] == 0
    assert state["last_progress_at"] == old_progress


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


def test_stalled_shard_timeout_is_independent_of_other_shard_progress(monkeypatch) -> None:
    stale_key = ("/private/bundle-A", 0)
    fast_key = ("/private/bundle-A", 1)
    stalled = fake_round({"path": stale_key[0]}, 0, FakeChild(None), completed=0)
    fast = fake_round({"path": fast_key[0]}, 1, FakeChild(None), completed=0)
    active = {stale_key: stalled, fast_key: fast}
    rows = [{
        "path": "/private/bundle-A",
        "shards": [
            {"shard": 0, "completed": 0, "remaining": 1},
            {"shard": 1, "completed": 1, "remaining": 0},
        ],
    }]
    failures = {}
    retry_after = {}
    monkeypatch.setattr(supervisor.time, "monotonic", lambda: 1801.0)
    stopped = []
    monkeypatch.setattr(supervisor, "stop_round", lambda job: stopped.append(job.shard))

    supervisor.poll_rounds(active, rows, failures, retry_after)

    assert stale_key not in active
    assert fast_key in active
    assert failures[stale_key] == "native_shard_stalled"
    assert retry_after[stale_key] == 1861.0
    assert stopped == [0]


def test_fresh_round_uses_a_fresh_progress_deadline(tmp_path: Path, monkeypatch) -> None:
    private = tmp_path / "runtime/recall-luna-bulk"
    private.mkdir(parents=True)
    child = FakeChild(None)
    row = {
        "path": str(tmp_path / "bundle-A"),
        "shards": [{"shard": 0, "completed": 0, "remaining": 1}],
    }
    monkeypatch.setattr(supervisor, "cli_command", lambda *_args: ["synthetic"])
    monkeypatch.setattr(supervisor.subprocess, "Popen", lambda *_args, **_kwargs: child)
    monkeypatch.setattr(supervisor.time, "monotonic", lambda: 123.0)
    args = argparse.Namespace(
        root=tmp_path, records_per_round=100,
        codex_home=tmp_path, codex="synthetic", poll_seconds=1,
    )

    job = supervisor.start_round(args, row, 0)

    assert job.last_progress == 123.0
    assert job.initial_completed == 0
    assert job.completed == 0


def test_stop_round_terminates_descendant_and_releases_shard_lease(tmp_path: Path) -> None:
    lock_path = tmp_path / ".native-worker.lock"
    descendant_code = (
        "import fcntl, os, sys, time; "
        "fd=os.open(sys.argv[1], os.O_RDWR|os.O_CREAT, 0o600); "
        "fcntl.flock(fd, fcntl.LOCK_EX); print('ready', flush=True); time.sleep(60)"
    )
    parent_code = (
        "import subprocess, sys, time; "
        f"child=subprocess.Popen([sys.executable, '-c', {descendant_code!r}, sys.argv[1]], "
        "stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True); time.sleep(60)"
    )
    parent = subprocess.Popen(
        [sys.executable, "-c", parent_code, str(lock_path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    descendant_pid = None
    try:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
            try:
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    break
            finally:
                os.close(descriptor)
            time.sleep(0.05)
        else:
            pytest.fail("descendant worker did not acquire the shard lease")

        pairs = [
            tuple(map(int, line.split()))
            for line in subprocess.check_output(["ps", "-axo", "pid=,ppid="], text=True).splitlines()
            if line.strip()
        ]
        descendant_pid = next(pid for pid, ppid in pairs if ppid == parent.pid)

        supervisor.stop_round(supervisor.NativeRound(
            child=parent,
            bundle_path=str(tmp_path),
            shard=0,
            round_id="round",
            completed=0,
            initial_completed=0,
            last_progress=0.0,
        ))

        assert parent.poll() is not None
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
            try:
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    pass
                else:
                    break
            finally:
                os.close(descriptor)
            time.sleep(0.05)
        else:
            pytest.fail("descendant worker still owns the shard lease")
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait(timeout=5)
        if descendant_pid is not None:
            # The supervisor should already have signalled this process. If a
            # platform reparented it before the snapshot, make the test safe.
            try:
                os.kill(descendant_pid, 0)
            except ProcessLookupError:
                pass
            else:
                try:
                    os.kill(descendant_pid, 9)
                except ProcessLookupError:
                    pass


def test_fast_shard_is_refilled_without_waiting_for_slow_shard(tmp_path: Path, monkeypatch) -> None:
    private = tmp_path / "runtime/recall-luna-bulk"
    private.mkdir(parents=True)
    bundle = fake_row(tmp_path / "bundle-A", 2)
    bundle["shards"] = [
        {"shard": 0, "completed": 0, "remaining": 2},
        {"shard": 1, "completed": 0, "remaining": 2},
    ]
    bundle["remaining"] = 4
    starts = []
    children = {
        0: [FakeChild(0), FakeChild(None)],
        1: [FakeChild(None)],
    }

    def start(args, row, shard):
        starts.append((row["path"], shard, args.records_per_round))
        return fake_round(row, shard, children[shard].pop(0))

    monkeypatch.setattr(supervisor, "inventory", lambda _root: [bundle])
    monkeypatch.setattr(supervisor, "available_shards", lambda _row, _limit: [0, 1])
    monkeypatch.setattr(supervisor, "start_round", start)
    monkeypatch.setattr(supervisor, "stop_round", lambda _job: None)
    monkeypatch.setattr(
        supervisor.daily.distill,
        "load_distillation_config",
        lambda _path: SimpleNamespace(
            enabled=True,
            native_luna_enabled=True,
            teacher_profile=supervisor.daily.DEFAULT_TEACHER_PROFILE,
        ),
    )

    class StopLoop(BaseException):
        pass

    sleeps = 0

    def sleep(_seconds):
        nonlocal sleeps
        sleeps += 1
        if sleeps == 1:
            bundle["shards"][0].update(completed=1, remaining=1)
            bundle["completed"] = 1
            bundle["remaining"] = 3
        else:
            raise StopLoop

    monkeypatch.setattr(supervisor.time, "sleep", sleep)
    args = argparse.Namespace(
        root=tmp_path, max_workers=2, records_per_round=100,
        codex_home=tmp_path, codex="synthetic", poll_seconds=1, once=False,
    )

    with pytest.raises(StopLoop):
        supervisor.supervise(args)

    assert [(path, shard) for path, shard, _cap in starts] == [
        (str(tmp_path / "bundle-A"), 0),
        (str(tmp_path / "bundle-A"), 1),
        (str(tmp_path / "bundle-A"), 0),
    ]
    assert all(cap == 100 for _path, _shard, cap in starts)


def test_once_wave_does_not_refill_a_fast_shard(tmp_path: Path, monkeypatch) -> None:
    private = tmp_path / "runtime/recall-luna-bulk"
    private.mkdir(parents=True)
    bundle = fake_row(tmp_path / "bundle-A", 2)
    bundle["shards"] = [
        {"shard": 0, "completed": 0, "remaining": 2},
        {"shard": 1, "completed": 0, "remaining": 2},
    ]
    bundle["remaining"] = 4
    starts = []

    def start(args, row, shard):
        starts.append(shard)
        return fake_round(row, shard, FakeChild(0))

    monkeypatch.setattr(supervisor, "inventory", lambda _root: [bundle])
    monkeypatch.setattr(supervisor, "available_shards", lambda _row, _limit: [0, 1])
    monkeypatch.setattr(supervisor, "start_round", start)
    monkeypatch.setattr(supervisor, "stop_round", lambda _job: None)
    monkeypatch.setattr(
        supervisor.daily.distill,
        "load_distillation_config",
        lambda _path: SimpleNamespace(
            enabled=True,
            native_luna_enabled=True,
            teacher_profile=supervisor.daily.DEFAULT_TEACHER_PROFILE,
        ),
    )

    def sleep(_seconds):
        for shard in bundle["shards"]:
            shard.update(completed=1, remaining=1)
        bundle["completed"] = 2
        bundle["remaining"] = 2

    monkeypatch.setattr(supervisor.time, "sleep", sleep)
    args = argparse.Namespace(
        root=tmp_path, max_workers=2, records_per_round=100,
        codex_home=tmp_path, codex="synthetic", poll_seconds=1, once=True,
    )

    assert supervisor.supervise(args) == 0
    assert starts == [0, 1]
    state = json.loads((private / "supervisor-status.json").read_text())
    assert state["reason"] == "wave_complete"
    assert state["active_workers"] == 0


def test_pending_shards_interleave_bundles_for_a_small_worker_cap(tmp_path: Path) -> None:
    rows = [fake_row(tmp_path / "bundle-A", 3), fake_row(tmp_path / "bundle-B", 3)]
    selected = supervisor.pending_shards(rows)
    assert [(Path(row["path"]).name, shard) for row, shard in selected] == [
        ("bundle-A", 0), ("bundle-B", 0),
        ("bundle-A", 1), ("bundle-B", 1),
        ("bundle-A", 2), ("bundle-B", 2),
    ]


def test_supervisor_can_start_twenty_workers_across_bundles(tmp_path: Path, monkeypatch) -> None:
    private = tmp_path / "runtime/recall-luna-bulk"
    private.mkdir(parents=True)
    rows = [fake_row(tmp_path / "bundle-A", 12), fake_row(tmp_path / "bundle-B", 12)]
    starts = []

    def start(args, row, shard):
        starts.append((Path(row["path"]).name, shard))
        return fake_round(row, shard, FakeChild(None))

    monkeypatch.setattr(supervisor, "inventory", lambda _root: rows)
    monkeypatch.setattr(supervisor, "available_shards", lambda row, _limit: [item["shard"] for item in row["shards"]])
    monkeypatch.setattr(supervisor, "start_round", start)
    monkeypatch.setattr(supervisor, "stop_round", lambda _job: None)
    monkeypatch.setattr(
        supervisor.daily.distill,
        "load_distillation_config",
        lambda _path: SimpleNamespace(
            enabled=True,
            native_luna_enabled=True,
            teacher_profile=supervisor.daily.DEFAULT_TEACHER_PROFILE,
        ),
    )

    class StopLoop(BaseException):
        pass

    monkeypatch.setattr(supervisor.time, "sleep", lambda _seconds: (_ for _ in ()).throw(StopLoop()))
    args = argparse.Namespace(
        root=tmp_path, max_workers=20, records_per_round=100,
        codex_home=tmp_path, codex="synthetic", poll_seconds=1, once=False,
    )

    with pytest.raises(StopLoop):
        supervisor.supervise(args)

    assert len(starts) == 20
    assert sum(bundle == "bundle-A" for bundle, _shard in starts) == 10
    assert sum(bundle == "bundle-B" for bundle, _shard in starts) == 10


def test_maintenance_marker_keeps_supervisor_running_without_new_workers(tmp_path: Path, monkeypatch) -> None:
    private = tmp_path / "runtime/recall-luna-bulk"
    private.mkdir(parents=True)
    marker = private / "supervisor-maintenance.json"
    marker.write_text(json.dumps({"reason": "test"}))
    row = fake_row(tmp_path / "bundle-A", 1)
    starts = []
    monkeypatch.setattr(supervisor, "inventory", lambda _root: [row])
    monkeypatch.setattr(supervisor, "start_round", lambda *_args: starts.append(True))
    monkeypatch.setattr(
        supervisor.daily.distill,
        "load_distillation_config",
        lambda _path: SimpleNamespace(
            enabled=True,
            native_luna_enabled=True,
            teacher_profile=supervisor.daily.DEFAULT_TEACHER_PROFILE,
        ),
    )
    args = argparse.Namespace(
        root=tmp_path, max_workers=20, records_per_round=100,
        codex_home=tmp_path, codex="synthetic", poll_seconds=1, once=True,
    )

    assert supervisor.supervise(args) == 1
    assert starts == []
    state = json.loads((private / "supervisor-status.json").read_text())
    assert state["status"] == "running"
    assert state["reason"] == "maintenance"
    assert state["active_workers"] == 0


def test_worker_prompt_keeps_the_per_child_cap_at_one_hundred(tmp_path: Path) -> None:
    prompt = supervisor.worker_prompt(tmp_path / "bundle-A", [0], "round", 100)
    assert '"records_per_worker": 100' in prompt
    assert "--max-records 100" in prompt
    assert '// @exec: {"max_output_tokens": 20000}' in prompt
    assert "Start the helper exactly once." in prompt
    assert "only the durable supervisor may retry with a fresh child." in prompt
