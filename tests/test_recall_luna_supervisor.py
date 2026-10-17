from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from chronovisor.decision.native_luna_authority import (
    NativeLunaCapacityUnavailable,
    native_luna_capacity_lease,
)
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


def test_native_capacity_is_shared_between_supervisor_and_authority(tmp_path: Path) -> None:
    supervisor_lease = native_luna_capacity_lease(
        tmp_path,
        owner="supervisor",
        slots=10,
    )
    supervisor_lease.__enter__()
    try:
        with pytest.raises(NativeLunaCapacityUnavailable):
            with native_luna_capacity_lease(tmp_path, owner="authority"):
                pass
    finally:
        supervisor_lease.__exit__(None, None, None)

    with native_luna_capacity_lease(tmp_path, owner="authority"):
        assert list(
            (tmp_path / "runtime/native-luna-authority").glob("slot-*.json")
        )


def test_native_capacity_honors_supervisor_status_without_tokens(tmp_path: Path) -> None:
    status_dir = tmp_path / "runtime/recall-luna-bulk"
    status_dir.mkdir(parents=True)
    holder = _supervisor_lock_holder(tmp_path)
    try:
        assert holder.stdout is not None
        assert holder.stdout.readline().strip() == "ready"
        status = {"active_workers": 10, "pid": holder.pid}
        (status_dir / "supervisor-status.json").write_text(json.dumps(status))
        with pytest.raises(NativeLunaCapacityUnavailable):
            with native_luna_capacity_lease(tmp_path, owner="authority"):
                pass
    finally:
        if holder.stdin is not None:
            holder.stdin.close()
        holder.wait(timeout=5)


def test_native_capacity_missing_status_fails_closed_when_supervisor_is_live(
    tmp_path: Path,
) -> None:
    holder = _supervisor_lock_holder(tmp_path)
    try:
        assert holder.stdout is not None
        assert holder.stdout.readline().strip() == "ready"
        with pytest.raises(NativeLunaCapacityUnavailable):
            with native_luna_capacity_lease(tmp_path, owner="authority"):
                pass
    finally:
        if holder.stdin is not None:
            holder.stdin.close()
        holder.wait(timeout=5)


def _supervisor_lock_holder(root: Path) -> subprocess.Popen[str]:
    script = "\n".join(
        (
            "import fcntl",
            "import os",
            "import sys",
            "from pathlib import Path",
            "path = Path(sys.argv[1])",
            "path.parent.mkdir(parents=True, exist_ok=True)",
            "fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)",
            "fcntl.flock(fd, fcntl.LOCK_EX)",
            "print('ready', flush=True)",
            "sys.stdin.read()",
        )
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    return subprocess.Popen(
        [sys.executable, "-c", script, str(root / "runtime/recall-luna-bulk/supervisor.lock")],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
    )


def test_native_capacity_ignores_stale_supervisor_status(tmp_path: Path) -> None:
    status_dir = tmp_path / "runtime/recall-luna-bulk"
    status_dir.mkdir(parents=True)
    (status_dir / "supervisor-status.json").write_text(
        json.dumps({"active_workers": 10})
    )
    with native_luna_capacity_lease(tmp_path, owner="authority"):
        pass


def test_native_capacity_registry_allows_ten_concurrent_reservations(
    tmp_path: Path,
) -> None:
    barrier = threading.Barrier(10)
    entered = threading.Event()
    release = threading.Event()
    guard = threading.Lock()
    entered_count = 0

    def worker() -> bool:
        nonlocal entered_count
        barrier.wait(timeout=5)
        try:
            with native_luna_capacity_lease(tmp_path, owner="authority"):
                with guard:
                    entered_count += 1
                    if entered_count == 10:
                        entered.set()
                entered.wait(timeout=5)
                release.wait(timeout=5)
            return True
        except (NativeLunaCapacityUnavailable, TimeoutError):
            return False

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(worker) for _ in range(10)]
        assert entered.wait(timeout=5)
        with pytest.raises(NativeLunaCapacityUnavailable):
            with native_luna_capacity_lease(tmp_path, owner="authority"):
                pass
        release.set()
        assert all(future.result(timeout=5) for future in futures)


def _capacity_holder(root: Path, owner: str, slots: int) -> subprocess.Popen[str]:
    script = "\n".join(
        (
            "from pathlib import Path",
            "import sys",
            "from chronovisor.decision.native_luna_authority import native_luna_capacity_lease",
            "root = Path(sys.argv[1])",
            "owner = sys.argv[2]",
            "slots = int(sys.argv[3])",
            "with native_luna_capacity_lease(root, owner=owner, slots=slots):",
            "    print('ready', flush=True)",
            "    sys.stdin.read()",
        )
    )
    environment = dict(os.environ)
    environment["CHRONOVISOR_ROOT"] = str(root)
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    return subprocess.Popen(
        [sys.executable, "-c", script, str(root), owner, str(slots)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
    )


def test_native_capacity_allows_ten_across_processes_and_rejects_eleven(
    tmp_path: Path,
) -> None:
    holder = _capacity_holder(tmp_path, "supervisor", 9)
    try:
        assert holder.stdout is not None
        assert holder.stdout.readline().strip() == "ready"
        with native_luna_capacity_lease(tmp_path, owner="authority", slots=1):
            with pytest.raises(NativeLunaCapacityUnavailable):
                with native_luna_capacity_lease(tmp_path, owner="authority"):
                    pass
    finally:
        if holder.stdin is not None:
            holder.stdin.close()
        holder.wait(timeout=5)
        assert holder.returncode == 0


def test_native_capacity_ignores_tokens_after_holder_process_dies(tmp_path: Path) -> None:
    holder = _capacity_holder(tmp_path, "authority", 10)
    assert holder.stdout is not None
    assert holder.stdout.readline().strip() == "ready"
    try:
        holder.kill()
        holder.wait(timeout=5)
        with native_luna_capacity_lease(tmp_path, owner="supervisor", slots=10):
            pass
    finally:
        if holder.poll() is None:
            holder.kill()
            holder.wait(timeout=5)


def test_native_capacity_lock_follows_child_after_parent_dies(tmp_path: Path) -> None:
    parent_code = "\n".join(
        (
            "import os",
            "import subprocess",
            "import sys",
            "import time",
            "from pathlib import Path",
            "from chronovisor.decision.native_luna_authority import native_luna_capacity_lease",
            "root = Path(sys.argv[1])",
            "child = 'import time; time.sleep(30)'",
            "with native_luna_capacity_lease(root, owner='supervisor', slots=10) as lease:",
            "    process = subprocess.Popen([sys.executable, '-c', child], pass_fds=lease.pass_fds)",
            "    print(process.pid, flush=True)",
            "    os._exit(0)",
        )
    )
    environment = dict(os.environ)
    environment["CHRONOVISOR_ROOT"] = str(tmp_path)
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    parent = subprocess.Popen(
        [sys.executable, "-c", parent_code, str(tmp_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
    )
    child_pid: int | None = None
    try:
        assert parent.stdout is not None
        child_pid = int(parent.stdout.readline().strip())
        assert parent.wait(timeout=5) == 0
        with pytest.raises(NativeLunaCapacityUnavailable):
            with native_luna_capacity_lease(tmp_path, owner="authority"):
                pass
    finally:
        if child_pid is not None:
            try:
                os.kill(child_pid, 15)
            except ProcessLookupError:
                pass
        parent.wait(timeout=5)
    deadline = time.monotonic() + 5
    while True:
        try:
            with native_luna_capacity_lease(tmp_path, owner="supervisor", slots=10):
                break
        except NativeLunaCapacityUnavailable:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.05)


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
    assert "For every nonempty write_stdin verdict submission, set yield_time_ms=1000." in prompt
    assert "a shorter tool wait never permits partial reading." in prompt


def test_daily_operation_failure_keeps_receipt_identity_and_durable_backoff(
    tmp_path: Path,
) -> None:
    state: dict[str, object] = {"operations": {}}
    bundle = tmp_path / "bundle-A"
    first = supervisor.DailyOperationError("NativeDistillationError", "a" * 64)
    second = supervisor.DailyOperationError("NativeDistillationError", "a" * 64)

    initial = supervisor._record_operation_failure(state, "import", bundle, first)
    repeated = supervisor._record_operation_failure(state, "import", bundle, second)

    assert initial["failure_count"] == 1
    assert repeated["failure_count"] == 2
    assert repeated["last_error_code"] == "NativeDistillationError"
    assert repeated["last_error_message_sha256"] == "a" * 64
    assert repeated["next_retry_at_unix"] > initial["next_retry_at_unix"]
    assert supervisor._operation_backoff_until(state, "import", bundle) == repeated[
        "next_retry_at_unix"
    ]


def test_non_run_completion_keeps_success_timestamp(tmp_path: Path) -> None:
    state: dict[str, object] = {"operations": {}}
    bundle = tmp_path / "bundle-A"

    completed = supervisor._complete_operation(state, "import", bundle)

    assert completed["last_success_at"]


def test_daily_run_terminal_success_survives_next_attempt_and_failure(
    tmp_path: Path,
) -> None:
    state: dict[str, object] = {"operations": {}}
    bundle = tmp_path / "runtime/recall-luna-bulk/2026-10-15/bundle"

    succeeded = supervisor._complete_operation(
        state, "run", bundle, reason="daily_prepare"
    )

    assert succeeded["last_success_status"] == "succeeded"
    assert succeeded["last_success_reason"] == "daily_prepare"
    success_at = succeeded["last_success_at"]

    running = supervisor._start_operation(state, "run", bundle)
    assert running["last_success_at"] == success_at
    assert running["last_success_status"] == "succeeded"
    assert running["last_success_reason"] == "daily_prepare"

    failed = supervisor._record_operation_failure(
        state,
        "run",
        bundle,
        supervisor.DailyOperationError("daily_prepare", "a" * 64),
    )
    assert failed["last_success_at"] == success_at
    assert failed["last_success_status"] == "succeeded"
    assert failed["last_success_reason"] == "daily_prepare"

    handoff = supervisor._set_waiting_operation(
        state, "run", bundle, reason="daily_capture_in_progress"
    )
    assert handoff["state"] == "waiting"
    assert handoff["status"] == "waiting"
    assert handoff["failure_count"] == failed["failure_count"]
    assert handoff["last_error_code"] == failed["last_error_code"]
    assert handoff["last_error_message_sha256"] == failed["last_error_message_sha256"]
    assert handoff["reason"] == "daily_capture_in_progress"
    assert handoff["last_success_at"] == success_at
    assert handoff["last_success_status"] == "succeeded"
    assert handoff["last_success_reason"] == "daily_prepare"


def test_daily_capture_terminal_success_survives_wait_failure_and_resolves_health(
    tmp_path: Path,
) -> None:
    private = tmp_path / "runtime/recall-luna-bulk"
    private.mkdir(parents=True)
    state: dict[str, object] = {"operations": {}}
    old_day = private / "2026-10-16"
    today = private / "2026-10-17"

    old_failure = supervisor._record_operation_failure(
        state,
        "capture",
        old_day,
        supervisor.DailyOperationError("daily_capture_receipt_stale", "a" * 64),
    )
    old_failure_at = datetime.now(UTC) - timedelta(hours=3)
    old_failure["updated_at"] = old_failure_at.isoformat()
    old_failure["next_retry_at"] = (old_failure_at + timedelta(minutes=5)).isoformat()
    old_failure["next_retry_at_unix"] = old_failure_at.timestamp() + 300

    def publish_health_state(operation: dict[str, object]) -> None:
        payload = {
            "schema": supervisor.SCHEMA,
            "status": "running",
            "updated_at": supervisor.now(),
            "last_progress_at": supervisor.now(),
            "completed_total": 0,
            "pending_total": 0,
            "bundles": {},
            "operation": operation,
            "operations": state["operations"],
        }
        supervisor.daily._write_private_json(private / "supervisor-status.json", payload)

    succeeded = supervisor._complete_operation(
        state, "capture", today, reason="capture_complete"
    )
    assert succeeded["last_success_status"] == "succeeded"
    assert succeeded["last_success_reason"] == "capture_complete"
    publish_health_state(succeeded)
    _diagnosis, alerts = _native_luna_supervisor_health(tmp_path)
    assert not any(
        item["reason"] == "native_luna_daily_capture_stalled" for item in alerts
    )

    running = supervisor._start_operation(state, "capture", today)
    assert running["last_success_at"] == succeeded["last_success_at"]
    assert running["last_success_status"] == "succeeded"
    assert running["last_success_reason"] == "capture_complete"
    publish_health_state(running)
    _diagnosis, alerts = _native_luna_supervisor_health(tmp_path)
    assert not any(
        item["reason"] == "native_luna_daily_capture_stalled" for item in alerts
    )

    waiting = supervisor._set_waiting_operation(
        state, "capture", today, reason="daily_capture_in_progress"
    )
    assert waiting["last_success_at"] == succeeded["last_success_at"]
    assert waiting["last_success_status"] == "succeeded"
    assert waiting["last_success_reason"] == "capture_complete"
    publish_health_state(waiting)
    _diagnosis, alerts = _native_luna_supervisor_health(tmp_path)
    assert not any(
        item["reason"] == "native_luna_daily_capture_stalled" for item in alerts
    )

    failed = supervisor._record_operation_failure(
        state,
        "capture",
        today,
        supervisor.DailyOperationError("daily_capture_receipt_stale", "b" * 64),
    )
    assert failed["last_success_at"] == succeeded["last_success_at"]
    assert failed["last_success_status"] == "succeeded"
    assert failed["last_success_reason"] == "capture_complete"
    publish_health_state(failed)
    _diagnosis, alerts = _native_luna_supervisor_health(tmp_path)
    assert any(
        item["reason"] == "native_luna_daily_capture_stalled" for item in alerts
    )

    recovered = supervisor._complete_operation(
        state, "capture", today, reason="capture_complete"
    )
    assert recovered["failure_count"] == 0
    assert recovered["last_success_status"] == "succeeded"
    assert recovered["last_success_reason"] == "capture_complete"
    publish_health_state(recovered)
    _diagnosis, alerts = _native_luna_supervisor_health(tmp_path)
    assert not any(
        item["reason"] == "native_luna_daily_capture_stalled" for item in alerts
    )


def test_legacy_capture_terminal_success_is_promoted_on_next_transition(
    tmp_path: Path,
) -> None:
    state: dict[str, object] = {"operations": {}}
    bundle = tmp_path / "runtime/recall-luna-bulk/2026-10-16"
    legacy = supervisor._complete_operation(
        state, "capture", bundle, reason="capture_complete"
    )
    legacy.pop("last_success_status")
    legacy.pop("last_success_reason")

    waiting = supervisor._set_waiting_operation(
        state, "capture", bundle, reason="daily_capture_in_progress"
    )

    assert waiting["last_success_at"] == legacy["last_success_at"]
    assert waiting["last_success_status"] == "succeeded"
    assert waiting["last_success_reason"] == "capture_complete"


def test_capture_handoff_keeps_run_failure_visible_in_supervisor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    private = tmp_path / "runtime/recall-luna-bulk"
    private.mkdir(parents=True)
    (tmp_path / "config.toml").write_text(
        '[recall.distillation]\nenabled=true\nnative_luna_enabled=true\n'
        'teacher_profile="native-luna-single-v1"\n'
    )
    day = datetime.now(supervisor.daily.ZoneInfo("Asia/Tokyo")).strftime("%Y-%m-%d")
    daily_bundle = tmp_path / "runtime/recall-luna-bulk" / day / "bundle"
    operation_key = f"run:{day}-bundle:"
    state: dict[str, object] = {"operations": {}}
    success = supervisor._complete_operation(
        state, "run", daily_bundle, reason="daily_prepare"
    )
    success_at = success["last_success_at"]
    failure = supervisor._record_operation_failure(
        state,
        "run",
        daily_bundle,
        supervisor.DailyOperationError("daily_prepare", "a" * 64),
    )
    # Make the pre-existing failure immediately eligible for this synthetic
    # retry; the production supervisor normally waits for this backoff.
    failure["next_retry_at_unix"] = time.time() - 1
    (private / "supervisor-status.json").write_text(
        json.dumps({
            "schema": supervisor.SCHEMA,
            "completed_total": 0,
            "last_progress_at": supervisor.now(),
            "operation": failure,
            "operations": {operation_key: failure},
        })
    )
    monkeypatch.setattr(supervisor, "inventory", lambda _root: [])
    calls: list[str] = []

    class StopLoop(BaseException):
        pass

    def run_daily(_args, command, _bundle, _publish):
        calls.append(command)
        if len(calls) == 1:
            return {
                "status": "ready",
                "reason": "capture_already_running",
                "capture": {"status": "running", "pid": 4321},
            }
        raise StopLoop()

    monkeypatch.setattr(supervisor, "run_daily_operation", run_daily)
    monkeypatch.setattr(supervisor.time, "sleep", lambda _seconds: None)
    args = argparse.Namespace(root=tmp_path, max_workers=1, poll_seconds=0, once=False)

    with pytest.raises(StopLoop):
        supervisor.supervise(args)

    assert calls == ["run", "run"]
    state = json.loads((private / "supervisor-status.json").read_text())
    run_entry = state["operations"][operation_key]
    assert run_entry["state"] == "waiting"
    assert run_entry["status"] == "waiting"
    assert run_entry["reason"] == "daily_capture_in_progress"
    assert run_entry["failure_count"] == 1
    assert run_entry["last_error_code"] == "daily_prepare"
    assert run_entry["last_error_message_sha256"] == "a" * 64
    assert run_entry["last_success_at"] == success_at
    assert run_entry["last_success_reason"] == "daily_prepare"

    _diagnosis, alerts = _native_luna_supervisor_health(tmp_path)
    assert any(item["reason"] == "native_luna_daily_run_failed" for item in alerts)

    completed = supervisor._complete_operation(
        state, "run", daily_bundle, reason="daily_prepare"
    )
    assert completed["failure_count"] == 0
    state["status"] = "running"
    state["updated_at"] = supervisor.now()
    supervisor.daily._write_private_json(private / "supervisor-status.json", state)
    _diagnosis, alerts = _native_luna_supervisor_health(tmp_path)
    assert not any(item["reason"] == "native_luna_daily_run_failed" for item in alerts)


def test_waiting_operation_preserves_start_boundary_across_heartbeats(
    tmp_path: Path,
) -> None:
    state: dict[str, object] = {"operations": {}}
    bundle = tmp_path / "bundle-A"
    first = supervisor._set_waiting_operation(
        state, "run", bundle, reason="distillation_worker_lock_busy"
    )
    second = supervisor._set_waiting_operation(
        state, "run", bundle, reason="daily_run_in_progress"
    )

    assert second["started_at"] == first["started_at"]
    assert second["updated_at"] != ""
    assert second["reason"] == "daily_run_in_progress"


def test_supervise_keeps_lock_wait_age_across_repeated_in_progress_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    private = tmp_path / "runtime/recall-luna-bulk"
    private.mkdir(parents=True)
    (tmp_path / "config.toml").write_text(
        '[recall.distillation]\nenabled=true\nnative_luna_enabled=true\n'
        'teacher_profile="native-luna-single-v1"\n'
    )
    monkeypatch.setattr(supervisor, "inventory", lambda _root: [])
    old = (datetime.now(UTC) - timedelta(minutes=31)).isoformat()
    fresh = {"value": False}
    monkeypatch.setattr(
        supervisor,
        "now",
        lambda: datetime.now(UTC).isoformat() if fresh["value"] else old,
    )
    calls: list[str] = []

    def run_daily(_args, command, _bundle, _publish):
        calls.append(command)
        if len(calls) == 1:
            fresh["value"] = True
        return {
            "status": "in_progress",
            "reason": "distillation_worker_lock_busy",
            "capture": {
                "status": "in_progress",
                "reason": "distillation_worker_lock_busy",
            },
        }

    monkeypatch.setattr(supervisor, "run_daily_operation", run_daily)
    original_write = supervisor.daily._write_private_json
    writes = 0

    class StopLoop(BaseException):
        pass

    def write_status(path: Path, value: object) -> None:
        nonlocal writes
        original_write(path, value)
        if isinstance(value, dict) and value.get("reason") == "daily_run_in_progress":
            writes += 1
            if writes == 2:
                raise StopLoop()

    monkeypatch.setattr(supervisor.daily, "_write_private_json", write_status)
    args = argparse.Namespace(
        root=tmp_path,
        max_workers=1,
        poll_seconds=0,
        once=False,
    )

    with pytest.raises(StopLoop):
        supervisor.supervise(args)

    assert calls == ["run", "run"]
    state = json.loads((private / "supervisor-status.json").read_text())
    assert state["operation"]["state"] == "waiting"
    assert state["operation"]["started_at"] == old
    _diagnosis, alerts = _native_luna_supervisor_health(tmp_path)
    assert any(item["reason"] == "native_luna_daily_run_failed" for item in alerts)


def test_daily_run_operation_identity_includes_day_without_a_manifest(
    tmp_path: Path,
) -> None:
    first = tmp_path / "runtime/recall-luna-bulk/2026-10-13/bundle"
    second = tmp_path / "runtime/recall-luna-bulk/2026-10-14/bundle"

    assert supervisor._bundle_identity(first)["bundle"] == "2026-10-13-bundle"
    assert supervisor._operation_key("run", supervisor._bundle_identity(first)) != (
        supervisor._operation_key("run", supervisor._bundle_identity(second))
    )


def test_daily_operation_uses_receipt_error_instead_of_generic_exit_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    private = tmp_path / "runtime/recall-luna-bulk"
    private.mkdir(parents=True)
    receipt = private / "supervisor-import.json"
    receipt.write_text(
        json.dumps({
            "status": "error",
            "error_type": "NativeDistillationError",
            "error_message_sha256": "c" * 64,
            "resumable": True,
        })
    )

    class DoneChild:
        pid = 9876
        returncode = 1

        def poll(self) -> int:
            return self.returncode

    def launch(argv, **_kwargs):
        # The old fixed receipt must be ignored; the child writes its own
        # attempt-scoped receipt selected by the supervisor.
        attempt_receipt = Path(argv[argv.index("--receipt") + 1])
        attempt_receipt.write_text(
            json.dumps(
                {
                    "status": "error",
                    "error_type": "NativeDistillationError",
                    "error_message_sha256": "c" * 64,
                    "resumable": True,
                }
            )
        )
        return DoneChild()

    monkeypatch.setattr(supervisor.subprocess, "Popen", launch)
    args = argparse.Namespace(root=tmp_path, poll_seconds=1)

    with pytest.raises(supervisor.DailyOperationError) as raised:
        supervisor.run_daily_operation(
            args,
            "import",
            tmp_path / "bundle-A",
            lambda *_args, **_kwargs: None,
        )
    assert raised.value.code == "NativeDistillationError"
    assert raised.value.message_sha256 == "c" * 64


def test_completed_labels_are_visible_and_import_failure_does_not_spin_after_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    private = tmp_path / "runtime/recall-luna-bulk"
    private.mkdir(parents=True)
    (tmp_path / "config.toml").write_text(
        '[recall.distillation]\nenabled=true\nnative_luna_enabled=true\nteacher_profile="native-luna-single-v1"\n'
    )
    bundle_path = tmp_path / "runtime/recall-luna-bulk/2026-10-13/bundle-A"
    row = fake_row(bundle_path, 2)
    row["completed"] = 2
    row["remaining"] = 0
    row["shards"] = [
        {"shard": 0, "completed": 1, "remaining": 0},
        {"shard": 1, "completed": 1, "remaining": 0},
    ]
    row["labels_done_at"] = "2026-10-14T07:00:00Z"
    calls: list[str] = []

    def fail_import(args, command, bundle, publish):
        calls.append(command)
        raise supervisor.DailyOperationError("NativeDistillationError", "b" * 64)

    monkeypatch.setattr(supervisor, "inventory", lambda _root: [row])
    monkeypatch.setattr(supervisor, "run_daily_operation", fail_import)
    monkeypatch.setattr(supervisor.time, "sleep", lambda _seconds: None)
    args = argparse.Namespace(root=tmp_path, max_workers=1, poll_seconds=1, once=True)

    assert supervisor.supervise(args) == 1
    first_state = json.loads((private / "supervisor-status.json").read_text())
    assert calls == ["import"]
    assert first_state["import_waiting"]["bundles"] == 1
    assert first_state["import_waiting"]["labels"] == 2
    assert first_state["import_waiting"]["oldest_at"] == "2026-10-14T07:00:00Z"
    assert first_state["operation"]["state"] == "backoff"
    assert first_state["operation"]["last_error_code"] == "NativeDistillationError"

    # A restarted supervisor loads the same operation identity and respects
    # its retry deadline instead of starting another two-minute import loop.
    assert supervisor.supervise(args) == 1
    second_state = json.loads((private / "supervisor-status.json").read_text())
    assert calls == ["import"]
    assert second_state["reason"] == "daily_import_backoff"
    assert second_state["operation"]["failure_count"] == 1


def test_import_backoff_wakes_for_earliest_bundle_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    private = tmp_path / "runtime/recall-luna-bulk"
    private.mkdir(parents=True)
    (tmp_path / "config.toml").write_text(
        '[recall.distillation]\nenabled=true\nnative_luna_enabled=true\n'
        'teacher_profile="native-luna-single-v1"\n'
    )
    rows = [
        fake_row(tmp_path / "bundle-A", 1, completed=1),
        fake_row(tmp_path / "bundle-B", 1, completed=1),
    ]
    for row in rows:
        row["remaining"] = 0
        row["shards"] = [{"shard": 0, "completed": 1, "remaining": 0}]
    updated = supervisor.now()
    operations = {}
    for bundle, retry_at in (("bundle-A", 100.0), ("bundle-B", 50.0)):
        operations[f"import:{bundle}:"] = {
            "name": "import",
            "state": "backoff",
            "status": "failed",
            "bundle": bundle,
            "failure_count": 1,
            "last_error_code": "NativeDistillationError",
            "last_error_message_sha256": "a" * 64,
            "next_retry_at_unix": retry_at,
            "next_retry_at": supervisor._iso_from_unix(retry_at),
            "updated_at": updated,
        }
    (private / "supervisor-status.json").write_text(
        json.dumps({
            "schema": supervisor.SCHEMA,
            "completed_total": 2,
            "last_progress_at": updated,
            "operations": operations,
        })
    )
    monkeypatch.setattr(supervisor, "inventory", lambda _root: rows)
    monkeypatch.setattr(supervisor.time, "time", lambda: 0.0)
    monkeypatch.setattr(
        supervisor,
        "run_daily_operation",
        lambda *_args: pytest.fail("all bundles are still in backoff"),
    )
    args = argparse.Namespace(root=tmp_path, max_workers=1, poll_seconds=1, once=True)

    assert supervisor.supervise(args) == 1
    state = json.loads((private / "supervisor-status.json").read_text())
    assert state["reason"] == "daily_import_backoff"
    assert state["operation"]["bundle"] == "bundle-B"


def test_active_capture_is_reported_without_restarting_daily_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    day = datetime.now(supervisor.daily.ZoneInfo("Asia/Tokyo")).strftime("%Y-%m-%d")
    private = tmp_path / "runtime/recall-luna-bulk"
    progress_path = private / f"{day}/inventory/candidate-capture-progress.json"
    progress_path.parent.mkdir(parents=True)
    now_unix = time.time()
    old_capture_failure = {
        "name": "capture",
        "state": "backoff",
        "status": "failed",
        "bundle": day,
        "failure_count": 2,
        "last_error_code": "capture_owner_stale",
        "last_error_message_sha256": "a" * 64,
        "next_retry_at_unix": now_unix - 1,
        "next_retry_at": supervisor._iso_from_unix(now_unix - 1),
        "updated_at": supervisor.now(),
    }
    progress_path.write_text(
        json.dumps({
            "status": "running",
            "pid": 4321,
            "started_at_unix": now_unix - 5,
            "updated_at_unix": now_unix,
        })
    )
    (private / "supervisor-status.json").write_text(
        json.dumps({
            "schema": supervisor.SCHEMA,
            "completed_total": 0,
            "operations": {f"capture:{day}:": old_capture_failure},
            "operation": old_capture_failure,
        })
    )
    (tmp_path / "config.toml").write_text(
        '[recall.distillation]\nenabled=true\nnative_luna_enabled=true\nteacher_profile="native-luna-single-v1"\n'
    )
    monkeypatch.setattr(supervisor, "inventory", lambda _root: [])
    monkeypatch.setattr(
        supervisor,
        "run_daily_operation",
        lambda *_args: pytest.fail("daily run was restarted while capture was active"),
    )
    args = argparse.Namespace(root=tmp_path, max_workers=1, poll_seconds=1, once=True)

    assert supervisor.supervise(args) == 1
    state = json.loads((private / "supervisor-status.json").read_text())
    assert state["status"] == "running"
    assert state["reason"] == "daily_capture_in_progress"
    assert state["operation"]["name"] == "capture"
    assert state["operation"]["failure_count"] == 0
    started_at = supervisor._iso_from_unix(now_unix - 5)
    assert state["operation"]["started_at"] == started_at

    # A subsequent heartbeat keeps the new owner's age; it must not reset the
    # boundary or resurrect the previous capture failure.
    assert supervisor.supervise(args) == 1
    repeated = json.loads((private / "supervisor-status.json").read_text())
    assert repeated["operation"]["failure_count"] == 0
    assert repeated["operation"]["started_at"] == started_at


def test_stalled_capture_is_backed_off_and_remains_visible_after_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    day = datetime.now(supervisor.daily.ZoneInfo("Asia/Tokyo")).strftime("%Y-%m-%d")
    private = tmp_path / "runtime/recall-luna-bulk"
    progress_path = private / f"{day}/inventory/candidate-capture-progress.json"
    progress_path.parent.mkdir(parents=True)
    progress_path.write_text(
        json.dumps({
            "status": "error",
            "reason": "capture_owner_stale",
            "error_message_sha256": "d" * 64,
        })
    )
    (tmp_path / "config.toml").write_text(
        '[recall.distillation]\nenabled=true\nnative_luna_enabled=true\nteacher_profile="native-luna-single-v1"\n'
    )
    monkeypatch.setattr(supervisor, "inventory", lambda _root: [])
    monkeypatch.setattr(
        supervisor,
        "run_daily_operation",
        lambda *_args: pytest.fail("daily run was started while capture was stalled"),
    )
    args = argparse.Namespace(root=tmp_path, max_workers=1, poll_seconds=1, once=True)

    assert supervisor.supervise(args) == 1
    first = json.loads((private / "supervisor-status.json").read_text())
    assert first["operation"]["name"] == "capture"
    assert first["operation"]["state"] == "backoff"
    assert first["operation"]["last_error_code"] == "capture_owner_stale"

    assert supervisor.supervise(args) == 1
    second = json.loads((private / "supervisor-status.json").read_text())
    assert second["reason"] == "daily_capture_backoff"
    assert second["operation"]["failure_count"] == 1


def test_old_capture_error_from_previous_day_does_not_block_daily_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    private = tmp_path / "runtime/recall-luna-bulk"
    old_progress = private / "2026-10-13/inventory/candidate-capture-progress.json"
    old_progress.parent.mkdir(parents=True)
    old_progress.write_text(
        json.dumps({"status": "error", "reason": "capture_owner_stale"})
    )
    (tmp_path / "config.toml").write_text(
        '[recall.distillation]\nenabled=true\nnative_luna_enabled=true\n'
        'teacher_profile="native-luna-single-v1"\n'
    )
    calls: list[str] = []
    monkeypatch.setattr(supervisor, "inventory", lambda _root: [])

    def recover(_args, command, _bundle, _publish):
        calls.append(command)
        return {"status": "ready", "reason": "no_unlabeled_source"}

    monkeypatch.setattr(supervisor, "run_daily_operation", recover)
    args = argparse.Namespace(root=tmp_path, max_workers=1, poll_seconds=1, once=True)

    supervisor.supervise(args)

    assert calls == ["run"]


def test_capture_backoff_allows_one_daily_run_recovery_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    day = datetime.now(supervisor.daily.ZoneInfo("Asia/Tokyo")).strftime("%Y-%m-%d")
    private = tmp_path / "runtime/recall-luna-bulk"
    progress_path = private / f"{day}/inventory/candidate-capture-progress.json"
    progress_path.parent.mkdir(parents=True)
    progress_path.write_text(
        json.dumps({
            "status": "error",
            "reason": "capture_owner_stale",
            "error_message_sha256": "e" * 64,
        })
    )
    (tmp_path / "config.toml").write_text(
        '[recall.distillation]\nenabled=true\nnative_luna_enabled=true\n'
        'teacher_profile="native-luna-single-v1"\n'
    )
    monkeypatch.setattr(supervisor, "inventory", lambda _root: [])
    args = argparse.Namespace(root=tmp_path, max_workers=1, poll_seconds=1, once=True)
    monkeypatch.setattr(
        supervisor,
        "run_daily_operation",
        lambda *_args: pytest.fail("capture error must be recorded before recovery"),
    )

    assert supervisor.supervise(args) == 1
    status_path = private / "supervisor-status.json"
    state = json.loads(status_path.read_text())
    assert state["operation"]["name"] == "capture"
    key = f"capture:{day}:"
    state["operations"][key]["next_retry_at_unix"] = time.time() - 1
    state["operations"][key]["next_retry_at"] = supervisor._iso_from_unix(
        time.time() - 1
    )
    state["operation"]["next_retry_at_unix"] = state["operations"][key][
        "next_retry_at_unix"
    ]
    state["operation"]["next_retry_at"] = state["operations"][key]["next_retry_at"]
    status_path.write_text(json.dumps(state))

    calls: list[str] = []

    def recover(_args, command, _bundle, _publish):
        calls.append(command)
        return {"status": "ready", "reason": "no_unlabeled_source"}

    monkeypatch.setattr(supervisor, "run_daily_operation", recover)
    assert supervisor.supervise(args) == 1
    assert calls == ["run"]
    recovered = json.loads(status_path.read_text())
    assert recovered["operations"][f"capture:{day}:"]["state"] == "succeeded"


def test_daily_in_progress_lock_wait_does_not_backoff_capture_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    day = datetime.now(supervisor.daily.ZoneInfo("Asia/Tokyo")).strftime("%Y-%m-%d")
    private = tmp_path / "runtime/recall-luna-bulk"
    progress_path = private / f"{day}/inventory/candidate-capture-progress.json"
    progress_path.parent.mkdir(parents=True)
    progress_path.write_text(
        json.dumps({"status": "error", "reason": "capture_owner_stale"})
    )
    (tmp_path / "config.toml").write_text(
        '[recall.distillation]\nenabled=true\nnative_luna_enabled=true\n'
        'teacher_profile="native-luna-single-v1"\n'
    )
    run_entry = {
        "name": "run",
        "state": "waiting",
        "status": "waiting",
        "bundle": f"{day}-bundle",
        "failure_count": 0,
        "reason": "distillation_worker_lock_busy",
        "updated_at": supervisor.now(),
    }
    (private / "supervisor-status.json").write_text(
        json.dumps(
            {
                "schema": supervisor.SCHEMA,
                "completed_total": 0,
                "operations": {f"run:{day}-bundle:": run_entry},
                "operation": run_entry,
            }
        )
    )
    monkeypatch.setattr(supervisor, "inventory", lambda _root: [])
    calls: list[str] = []

    def lock_wait(_args, command, _bundle, _publish):
        calls.append(command)
        return {
            "status": "in_progress",
            "reason": "distillation_worker_lock_busy",
            "capture": {
                "status": "in_progress",
                "reason": "distillation_worker_lock_busy",
            },
        }

    monkeypatch.setattr(supervisor, "run_daily_operation", lock_wait)
    args = argparse.Namespace(root=tmp_path, max_workers=1, poll_seconds=1, once=True)

    supervisor.supervise(args)

    state = json.loads((private / "supervisor-status.json").read_text())
    assert calls == ["run"]
    assert state["operation"]["name"] == "run"
    assert state["operation"]["state"] == "waiting"
    assert not any(key.startswith("capture:") for key in state["operations"])


def test_import_backoff_on_bundle_a_does_not_starve_bundle_b(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    private = tmp_path / "runtime/recall-luna-bulk"
    private.mkdir(parents=True)
    (tmp_path / "config.toml").write_text(
        '[recall.distillation]\nenabled=true\nnative_luna_enabled=true\n'
        'teacher_profile="native-luna-single-v1"\n'
    )
    bundle_a = tmp_path / "runtime/recall-luna-bulk/2026-10-13/bundle-A"
    bundle_b = tmp_path / "runtime/recall-luna-bulk/2026-10-14/bundle-B"
    rows = [fake_row(bundle_a, 1, completed=1), fake_row(bundle_b, 1, completed=1)]
    for row in rows:
        row["remaining"] = 0
        row["shards"] = [{"shard": 0, "completed": 1, "remaining": 0}]
    monkeypatch.setattr(supervisor, "inventory", lambda _root: rows)
    retry_at = time.time() + 3600
    state = {
        "schema": supervisor.SCHEMA,
        "completed_total": 2,
        "operations": {
            "import:bundle-A:": {
                "name": "import",
                "state": "backoff",
                "status": "failed",
                "bundle": "bundle-A",
                "failure_count": 2,
                "last_error_code": "NativeDistillationError",
                "last_error_message_sha256": "f" * 64,
                "next_retry_at_unix": retry_at,
                "next_retry_at": supervisor._iso_from_unix(retry_at),
                "updated_at": supervisor.now(),
            }
        },
    }
    (private / "supervisor-status.json").write_text(json.dumps(state))
    calls: list[str] = []

    def import_b(args, command, bundle, _publish):
        calls.append(bundle.name)
        rows[1]["imported"] = True
        return {"status": "imported", "labels": 1, "labels_written": 1}

    monkeypatch.setattr(supervisor, "run_daily_operation", import_b)
    args = argparse.Namespace(root=tmp_path, max_workers=1, poll_seconds=1, once=True)

    supervisor.supervise(args)

    assert calls == ["bundle-B"]
    state_after = json.loads((private / "supervisor-status.json").read_text())
    assert state_after["last_import"]["labels_written"] == 1
    assert "last_label_at" in state_after["last_import"]
    assert state_after["last_imported_label_at"] == state_after["last_import"][
        "last_label_at"
    ]


def test_daily_pending_bundle_is_successful_preparation_for_worker_wave(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "config.toml").write_text(
        '[recall.distillation]\nenabled=true\nnative_luna_enabled=true\n'
        'teacher_profile="native-luna-single-v1"\n'
    )
    monkeypatch.setattr(supervisor, "inventory", lambda _root: [])
    outcomes = iter(
        [
            {"status": "pending", "reason": "bundle_created"},
            {"status": "ready", "reason": "no_unlabeled_source"},
        ]
    )
    monkeypatch.setattr(
        supervisor,
        "run_daily_operation",
        lambda *_args: next(outcomes),
    )
    monkeypatch.setattr(supervisor.time, "sleep", lambda _seconds: None)
    args = argparse.Namespace(root=tmp_path, max_workers=1, poll_seconds=1, once=True)

    assert supervisor.supervise(args) == 1
    state = json.loads(
        (tmp_path / "runtime/recall-luna-bulk/supervisor-status.json").read_text()
    )
    assert state["reason"] == "waiting_for_incremental_capture"
    assert state["operation"]["name"] == "idle"


def test_idempotent_import_does_not_advance_label_watermark(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    private = tmp_path / "runtime/recall-luna-bulk"
    private.mkdir(parents=True)
    (tmp_path / "config.toml").write_text(
        '[recall.distillation]\nenabled=true\nnative_luna_enabled=true\n'
        'teacher_profile="native-luna-single-v1"\n'
    )
    bundle = tmp_path / "runtime/recall-luna-bulk/2026-10-14/bundle-A"
    row = fake_row(bundle, 1, completed=1)
    row["remaining"] = 0
    row["shards"] = [{"shard": 0, "completed": 1, "remaining": 0}]
    rows = [row]
    old = "2026-10-14T00:00:00Z"
    (private / "supervisor-status.json").write_text(
        json.dumps(
            {
                "schema": supervisor.SCHEMA,
                "completed_total": 1,
                "last_imported_label_at": old,
                "operations": {},
            }
        )
    )
    monkeypatch.setattr(supervisor, "inventory", lambda _root: rows)
    calls: list[str] = []

    def import_once(_args, command, _bundle, _publish):
        calls.append(command)
        if command == "import":
            rows[0]["imported"] = True
            return {"status": "imported", "labels": 1, "labels_written": 0}
        return {"status": "ready", "reason": "no_unlabeled_source"}

    monkeypatch.setattr(supervisor, "run_daily_operation", import_once)
    args = argparse.Namespace(root=tmp_path, max_workers=1, poll_seconds=1, once=True)

    supervisor.supervise(args)

    state = json.loads((private / "supervisor-status.json").read_text())
    assert calls == ["import", "run"]
    assert state["last_imported_label_at"] == old
    assert state["last_import"]["labels_written"] == 0
    assert "last_label_at" not in state["last_import"]
