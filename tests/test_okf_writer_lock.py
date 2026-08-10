from __future__ import annotations

import fcntl
import importlib
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from chronovisor.core import okf_cutover, store
from chronovisor.core.durable_state import okf_writer_lock
from chronovisor.core.okf_cutover import (
    OKFStartupBlocked,
    OKFStartupDecision,
    discover_okf_startup,
    execute_okf_cutover,
)
from chronovisor.core.okf_workspace import prepare_okf_workspace
from chronovisor.hosts import cli

FIXTURE = Path(__file__).parent / "fixtures" / "okf_workspace" / "source"


def _legacy(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "source"
    shutil.copytree(FIXTURE, root)
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    return root, runtime


def _exclusive_is_blocked(lock_path: Path) -> bool:
    descriptor = os.open(lock_path, os.O_RDWR)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        return False
    finally:
        os.close(descriptor)


def test_shared_operation_evaluates_gate_while_lock_is_held(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _runtime = _legacy(tmp_path)
    with okf_writer_lock(root):
        pass
    lock_path = root / "runtime" / "okf-writer.lock"

    def require(_source: Path, _runtime: Path) -> OKFStartupDecision:
        assert _exclusive_is_blocked(lock_path)
        return OKFStartupDecision(True, "legacy", "unmigrated", "ok")

    monkeypatch.setattr(okf_cutover, "require_okf_startup_allowed", require)
    with store.okf_runtime_operation(root):
        assert _exclusive_is_blocked(lock_path)


def test_shared_and_exclusive_writer_leases_are_mutually_exclusive(
    tmp_path: Path,
) -> None:
    root, _runtime = _legacy(tmp_path)
    script = """
import sys
from pathlib import Path
from chronovisor.core.durable_state import okf_writer_lock
try:
    with okf_writer_lock(Path(sys.argv[1]), exclusive=True):
        raise SystemExit(2)
except RuntimeError as exc:
    raise SystemExit(0 if str(exc) == "OKF writer lease is busy" else 3)
"""
    with okf_writer_lock(root):
        result = subprocess.run(
            [sys.executable, "-c", script, str(root)],
            check=False,
            timeout=2,
        )
    assert result.returncode == 0


def test_concurrent_first_shared_leases_use_one_lock_inode(tmp_path: Path) -> None:
    root = tmp_path / "bootstrap"
    start = tmp_path / "start"
    script = """
import os
import sys
import time
from pathlib import Path
from chronovisor.core.durable_state import okf_writer_lock
root = Path(sys.argv[1])
start = Path(sys.argv[2])
while not start.exists():
    time.sleep(0.001)
with okf_writer_lock(root):
    observed = (root / "runtime" / "okf-writer.lock").stat()
    print(f"{observed.st_dev}:{observed.st_ino}", flush=True)
    time.sleep(0.05)
"""
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", script, str(root), str(start)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(2)
    ]
    start.touch()
    results = [process.communicate(timeout=2) for process in processes]

    assert [process.returncode for process in processes] == [0, 0]
    assert not [stderr for _stdout, stderr in results if stderr]
    assert len({stdout.strip() for stdout, _stderr in results}) == 1


def test_blocked_gate_runs_no_operation_and_creates_no_lock(tmp_path: Path) -> None:
    root = tmp_path / "unsafe"
    root.mkdir()
    (root / "private.txt").write_text("canary", encoding="utf-8")
    before = sorted(path.relative_to(root) for path in root.rglob("*"))
    operations = 0

    with pytest.raises(OKFStartupBlocked):
        with store.okf_runtime_operation(root):
            operations += 1

    assert operations == 0
    assert sorted(path.relative_to(root) for path in root.rglob("*")) == before


def test_writer_lock_rejects_symlink_leaf_and_absent_exclusive_root(
    tmp_path: Path,
) -> None:
    root, _runtime = _legacy(tmp_path / "leaf")
    runtime = root / "runtime"
    runtime.mkdir()
    target = tmp_path / "outside.lock"
    target.write_text("canary", encoding="utf-8")
    (runtime / "okf-writer.lock").symlink_to(target)
    with pytest.raises(RuntimeError, match="unsafe"):
        with okf_writer_lock(root):
            pass
    assert target.read_text(encoding="utf-8") == "canary"

    absent = tmp_path / "absent"
    with pytest.raises(ValueError, match="must exist"):
        with okf_writer_lock(absent, exclusive=True):
            pass
    assert not absent.exists()


def test_bootstrap_allows_only_the_regular_writer_lock(tmp_path: Path) -> None:
    root = tmp_path / "bootstrap"
    with okf_writer_lock(root):
        pass
    lock_path = root / "runtime" / "okf-writer.lock"
    assert lock_path.is_file() and not lock_path.is_symlink()
    assert discover_okf_startup(root, root / "runtime").allowed

    (root / "runtime" / "extra").write_text("residue", encoding="utf-8")
    assert not discover_okf_startup(root, root / "runtime").allowed


def test_first_writer_lock_creation_fsyncs_file_and_runtime_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "bootstrap"
    synced: list[str] = []

    def fsync(descriptor: int) -> None:
        mode = os.fstat(descriptor).st_mode
        synced.append("directory" if stat.S_ISDIR(mode) else "file")

    monkeypatch.setattr(os, "fsync", fsync)
    with okf_writer_lock(root):
        pass

    assert synced == ["file", "directory", "directory", "directory"]


def test_unlinked_writer_lock_cannot_split_a_held_root_lease(tmp_path: Path) -> None:
    root, _runtime = _legacy(tmp_path)
    with okf_writer_lock(root):
        lock_path = root / "runtime" / "okf-writer.lock"
        lock_path.unlink()
        script = """
import sys
from pathlib import Path
from chronovisor.core.durable_state import okf_writer_lock
try:
    with okf_writer_lock(Path(sys.argv[1]), exclusive=True):
        raise SystemExit(2)
except RuntimeError as exc:
    raise SystemExit(0 if "busy" in str(exc) else 3)
"""
        result = subprocess.run(
            [sys.executable, "-c", script, str(root)], check=False, timeout=2
        )
        assert result.returncode == 0


def test_runtime_swap_cannot_split_a_held_root_lease_or_touch_victim(
    tmp_path: Path,
) -> None:
    root, _runtime = _legacy(tmp_path)
    victim = tmp_path / "victim"
    victim.mkdir()
    (victim / "canary").write_text("safe", encoding="utf-8")
    with okf_writer_lock(root):
        runtime = root / "runtime"
        pinned = root / "runtime-pinned"
        runtime.rename(pinned)
        runtime.symlink_to(victim, target_is_directory=True)
        script = """
import sys
from pathlib import Path
from chronovisor.core.durable_state import okf_writer_lock
try:
    with okf_writer_lock(Path(sys.argv[1]), exclusive=True):
        raise SystemExit(2)
except RuntimeError as exc:
    raise SystemExit(0 if "busy" in str(exc) else 3)
"""
        result = subprocess.run(
            [sys.executable, "-c", script, str(root)], check=False, timeout=2
        )
        assert result.returncode == 0
        assert not (victim / "okf-writer.lock").exists()
        assert (victim / "canary").read_text(encoding="utf-8") == "safe"
        runtime.unlink()
        pinned.rename(runtime)


def test_fork_child_unwind_does_not_unlock_parent_lease(tmp_path: Path) -> None:
    root, _runtime = _legacy(tmp_path)
    lease = okf_writer_lock(root)
    lease.__enter__()
    try:
        child = os.fork()
        if child == 0:
            lease.__exit__(None, None, None)
            os._exit(0)
        waited, status = os.waitpid(child, 0)
        assert waited == child
        assert os.waitstatus_to_exitcode(status) == 0
        script = """
import sys
from pathlib import Path
from chronovisor.core.durable_state import okf_writer_lock
try:
    with okf_writer_lock(Path(sys.argv[1]), exclusive=True):
        raise SystemExit(2)
except RuntimeError as exc:
    raise SystemExit(0 if "busy" in str(exc) else 3)
"""
        result = subprocess.run(
            [sys.executable, "-c", script, str(root)], check=False, timeout=2
        )
        assert result.returncode == 0
    finally:
        lease.__exit__(None, None, None)


def test_root_replacement_cannot_split_held_parent_lease_or_touch_victim(
    tmp_path: Path,
) -> None:
    root, _runtime = _legacy(tmp_path)
    moved = tmp_path / "source-pinned"
    victim = tmp_path / "victim"
    victim.mkdir()
    (victim / "canary").write_text("safe", encoding="utf-8")
    script = """
import sys
from pathlib import Path
from chronovisor.core.durable_state import okf_writer_lock
try:
    with okf_writer_lock(Path(sys.argv[1]), exclusive=True):
        raise SystemExit(2)
except RuntimeError as exc:
    raise SystemExit(0 if "busy" in str(exc) else 3)
"""

    with okf_writer_lock(root):
        root.rename(moved)
        root.mkdir()
        result = subprocess.run(
            [sys.executable, "-c", script, str(root)], check=False, timeout=2
        )
        assert result.returncode == 0
        root.rmdir()
        root.symlink_to(victim, target_is_directory=True)
        result = subprocess.run(
            [sys.executable, "-c", script, str(root)], check=False, timeout=2
        )
        assert result.returncode == 0
        assert not (victim / "runtime").exists()
        assert (victim / "canary").read_text(encoding="utf-8") == "safe"
        root.unlink()
        moved.rename(root)


def test_runtime_symlink_without_holder_fails_closed(tmp_path: Path) -> None:
    root, _runtime = _legacy(tmp_path)
    victim = tmp_path / "victim"
    victim.mkdir()
    (root / "runtime").symlink_to(victim, target_is_directory=True)

    with pytest.raises(OSError):
        with okf_writer_lock(root):
            pass

    assert not (victim / "okf-writer.lock").exists()


def test_missing_writer_lock_with_migration_workspace_is_not_recreated(
    tmp_path: Path,
) -> None:
    root, runtime = _legacy(tmp_path)
    prepare_okf_workspace(root, runtime, "run-001")
    lock_path = root / "runtime" / "okf-writer.lock"
    lock_path.unlink()

    with pytest.raises(RuntimeError, match="missing"):
        prepare_okf_workspace(root, runtime, "run-002")

    assert not lock_path.exists()


def test_external_migration_runtime_uses_source_root_writer_lock(
    tmp_path: Path,
) -> None:
    root, runtime = _legacy(tmp_path)

    with okf_writer_lock(root):
        with pytest.raises(RuntimeError, match="upgrade"):
            prepare_okf_workspace(root, runtime, "run-001")

    assert not (runtime / "migrations").exists()


def test_cutover_fault_injection_runs_under_exclusive_writer_lease(
    tmp_path: Path,
) -> None:
    root, runtime = _legacy(tmp_path)
    prepare_okf_workspace(root, runtime, "run-001")
    lock_path = root / "runtime" / "okf-writer.lock"
    checked = False

    def fault(point: str) -> None:
        nonlocal checked
        if point == "before-start-journal":
            descriptor = os.open(lock_path, os.O_RDWR)
            try:
                with pytest.raises(BlockingIOError):
                    fcntl.flock(descriptor, fcntl.LOCK_SH | fcntl.LOCK_NB)
            finally:
                os.close(descriptor)
            checked = True

    assert execute_okf_cutover(
        root,
        runtime,
        "run-001",
        is_quiescent=lambda: True,
        fault_inject=fault,
    ) == "committed-needs-rebuild"
    assert checked


@pytest.mark.parametrize(
    ("argv", "bypass"),
    [
        (["status"], False),
        (["hooks", "inspect"], True),
        (["raw", "status"], False),
        (["claims", "search", "needle"], False),
        (["hooks", "install", "--host", "codex"], False),
        (["raw", "seal"], False),
        (["claims", "rebuild"], False),
        (["recall-improve", "rollback"], False),
    ],
)
def test_hosts_cli_writer_lease_boundary(argv: list[str], bypass: bool) -> None:
    args = cli.build_parser().parse_args(argv)
    assert cli._okf_lease_bypass(args) is bypass


@pytest.mark.parametrize(
    ("module_name", "argv", "mutating"),
    [
        ("chronovisor.recall.recall_eval", ["--save-baseline"], True),
        ("chronovisor.recall.recall_eval", [], False),
        ("chronovisor.search.search_eval", ["--build-golden"], True),
        ("chronovisor.search.search_eval", [], False),
        ("chronovisor.research.oracle", ["query"], True),
        ("chronovisor.research.oracle", ["query", "--no-index-build"], False),
        ("chronovisor.recall.claims", ["rebuild"], True),
        ("chronovisor.recall.claims", ["search", "query"], False),
        ("chronovisor.recall.duplicate_review", ["--write"], True),
        ("chronovisor.recall.duplicate_review", [], False),
        ("chronovisor.search.cofire", [], True),
        ("chronovisor.search.cofire", ["--no-write"], False),
        ("chronovisor.search.prefetch", [], True),
        ("chronovisor.search.prefetch", ["--no-write"], False),
        ("chronovisor.ops.golden_expand", [], True),
        ("chronovisor.ops.golden_expand", ["--no-write"], False),
        ("chronovisor.ops.distill", [], True),
        ("chronovisor.ops.distill", ["--no-write"], False),
        ("chronovisor.research.research_service", ["query"], True),
        ("chronovisor.ops.repair_runbook", ["recover-hold-leases"], True),
        ("chronovisor.ops.repair_runbook", ["status"], False),
        ("chronovisor.ops.autonomy", ["watchdog"], True),
        ("chronovisor.ops.autonomy", ["status"], False),
        ("chronovisor.recall.recall_answer_eval", [], True),
        ("chronovisor.recall.recall_answer_eval", ["--status"], False),
        ("chronovisor.ops.entities", ["init"], True),
    ],
)
def test_direct_entrypoint_writer_lease_modes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
    argv: list[str],
    mutating: bool,
) -> None:
    module = importlib.import_module(module_name)
    root = tmp_path / module_name.replace(".", "-")
    root.mkdir()
    (root / "private.txt").write_text("canary", encoding="utf-8")
    monkeypatch.setattr(module, "CHRONOVISOR_ROOT", root)
    called = 0

    def inner(_args) -> int:
        nonlocal called
        called += 1
        return 0

    monkeypatch.setattr(module, "_main_locked", inner)

    assert module.main(argv) == (75 if mutating else 0)
    assert called == (0 if mutating else 1)
    assert [path.name for path in root.iterdir()] == ["private.txt"]
