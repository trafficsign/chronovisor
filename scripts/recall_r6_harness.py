#!/usr/bin/env python3.14
"""Fail-closed R6/P5 harness using only the official clone worker.

It has exactly two outcomes: an R5 read-only decline, or evidence from
``run_distillation_chunk(..., teachers={})`` in a disposable APFS clone.
"""

from __future__ import annotations

import argparse
import builtins
import contextlib
import fcntl
import hashlib
import hmac
import importlib
import importlib.machinery
import importlib.util
import io
import json
import math
import os
import secrets
import shutil
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Iterator, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

R6_SCHEMA = "chronovisor.recall-r6.v2"
BASELINE_SCHEMA = "chronovisor.recall-distill-baseline.v1"
MAX_FILE_BYTES = 32 * 1024 * 1024
_SHA = set("0123456789abcdef")
_CLONE_TARGETS = (Path("config.toml"), Path("raw"), Path("runtime/recall-distillation"))
_TRUSTED_EXECUTABLES = {
    "/usr/bin/git": "b8763cf250e607a778bb4603cecb5b90338814d0a3dfcba0d57b1de242f610e9",
    "/bin/cp": "f0629f462c6535f7b1a19f559b7093638e714961bea9228cfb2ae7896f8557f4",
}
_GIT_ENV_PREFIX = "GIT_"
_GIT_REDIRECT_KEYS = frozenset({"GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"})
_GIT_LAYOUT_KEYS = frozenset({"entry", "entry_sha256", "entry_bytes", "entry_inode", "entry_mode", "git_dir", "git_dir_inode", "git_dir_mode", "work_tree", "index", "index_sha256", "index_bytes", "index_inode", "index_mode"})
_CLONE_PROOF_KEYS = frozenset(
    {
        "schema",
        "namespace",
        "kind",
        "method",
        "production_path",
        "source_path",
        "clone_path",
        "euid",
        "filesystem",
        "production",
        "clone",
        "target_before",
        "same_device",
        "nonoverlap",
    }
)
_CLONE_STAT_KEYS = frozenset({"path", "dev", "ino", "uid", "mode"})
_CLONE_METHOD = "cp-cR-cow-requested"
_VOLATILE_STATE_FIELDS = frozenset({"stage_started_at", "last_success_at", "seal_sha256"})
_PHASE_TIMEOUT_SECONDS = 120.0

# Keep the stdlib objects captured at harness import time.  A worker can import
# arbitrary names, but it must not replace the objects on which our guards are
# installed (nor preload a fake module before the exact runtime is loaded).
_TRUSTED_BUILTINS = builtins
_TRUSTED_OS_MODULE = os
_TRUSTED_SOCKET_MODULE = socket
_TRUSTED_SUBPROCESS_MODULE = subprocess
_TRUSTED_SIGNAL_MODULE = signal
_TRUSTED_IO_MODULE = io
_GUARDED_STDLIB_MODULES = {
    "os": _TRUSTED_OS_MODULE,
    "socket": _TRUSTED_SOCKET_MODULE,
    "subprocess": _TRUSTED_SUBPROCESS_MODULE,
    "signal": _TRUSTED_SIGNAL_MODULE,
    "io": _TRUSTED_IO_MODULE,
}
_FORBIDDEN_NATIVE_IMPORTS = frozenset({"ctypes", "_ctypes", "cffi"})


class R6Error(ValueError):
    """An R6 safety, source, or official-store invariant failed."""


class R6GuardError(R6Error):
    """A rejected side effect, retaining the observed (not completed) counts."""

    def __init__(self, reason: str, *, egress_attempts: int, provider_attempts: int) -> None:
        super().__init__(reason)
        self.egress_attempts = egress_attempts
        self.provider_attempts = provider_attempts


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise R6Error("value is not canonical JSON") from exc


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _artifact_id(value: object) -> str:
    """Accept only an immutable-store filename identity."""

    if (
        not isinstance(value, str)
        or len(value) != 64
        or set(value) - _SHA
        or Path(value).name != value
        or "/" in value
        or "\\" in value
    ):
        raise R6Error("artifact identity is invalid")
    return value


def _trusted_executable(path: str) -> Path:
    if path not in _TRUSTED_EXECUTABLES:
        raise R6Error("trusted executable is unknown")
    original = Path(path)
    if not original.is_absolute() or _symlink_component(original):
        raise R6Error("trusted executable has a symlink component")
    state = original.lstat()
    if (
        not stat.S_ISREG(state.st_mode)
        or state.st_uid != 0
        or state.st_mode & 0o022
    ):
        raise R6Error("trusted executable is unsafe")
    executable = original.resolve(strict=True)
    if executable != original or hashlib.sha256(executable.read_bytes()).hexdigest() != _TRUSTED_EXECUTABLES[path]:
        raise R6Error("trusted executable identity mismatch")
    return executable


def _filesystem_type(path: Path) -> str:
    """Reuse R0's Darwin filesystem probe for the R6 clone contract."""

    probe_path = Path(__file__).with_name("recall_r0_harness.py")
    original_run = subprocess.run

    def bounded_probe_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[Any]:
        timeout = kwargs.get("timeout", 5)
        if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or not 0 < float(timeout) <= 5:
            timeout = 5
        kwargs["timeout"] = float(timeout)
        try:
            return original_run(*args, **kwargs)
        except subprocess.TimeoutExpired as exc:
            raise R6Error("APFS volume probe timed out") from exc

    try:
        spec = importlib.util.spec_from_file_location("recall_r0_apfs_probe", probe_path)
        if spec is None or spec.loader is None:
            raise R6Error("R0 APFS probe is unavailable")
        probe = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(probe)
        # R0 predates the R6 timeout contract.  Install a bounded adapter for
        # both its ``df`` and ``diskutil`` calls while preserving the module
        # implementation and restoring the global afterwards.
        probe.subprocess.run = bounded_probe_run
        value = probe._filesystem_type(path)
    except (OSError, ValueError, AttributeError, subprocess.TimeoutExpired, R6Error) as exc:
        raise R6Error("APFS volume probe failed") from exc
    finally:
        subprocess.run = original_run
    if not isinstance(value, str):
        raise R6Error("APFS volume probe is invalid")
    return value.lower()


def _seal(unsigned: Mapping[str, Any]) -> dict[str, Any]:
    if "seal_sha256" in unsigned:
        raise R6Error("unsigned evidence already has a seal")
    return {**unsigned, "seal_sha256": _digest(unsigned)}


def _assert_artifact_integrity(value: Mapping[str, Any], label: str, *, has_id: bool) -> None:
    """Recompute both immutable content identity and the enclosing seal."""

    if has_id:
        _artifact_id(value.get("artifact_id"))
        unsigned_content = {
            key: item for key, item in value.items()
            if key not in {"artifact_id", "seal_sha256"}
        }
        if value["artifact_id"] != _digest(unsigned_content):
            raise R6Error(f"{label} content identity is invalid")
    expected_seal = _digest({key: item for key, item in value.items() if key != "seal_sha256"})
    if value.get("seal_sha256") != expected_seal:
        raise R6Error(f"{label} seal is invalid")


def _policy_payload_digest(value: Mapping[str, Any]) -> str:
    fields = (
        "feature_keys", "feature_revision", "weights", "bias", "threshold",
        "abstain_margin", "max_cards", "training_rows", "validation_rows",
    )
    return _digest({field: value[field] for field in fields})


def _read_sealed(path: Path, schema: str) -> dict[str, Any]:
    try:
        before = path.lstat()
        if path.is_symlink() or not stat.S_ISREG(before.st_mode) or before.st_size > MAX_FILE_BYTES:
            raise R6Error("evidence path is unsafe")
        raw = path.read_bytes()
        after = path.lstat()
    except OSError as exc:
        raise R6Error("evidence cannot be read") from exc
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns
    ):
        raise R6Error("evidence changed during read")
    try:
        value = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise R6Error("evidence is not JSON") from exc
    if not isinstance(value, dict) or value.get("schema") != schema or value.get("namespace") != "recall-distillation":
        raise R6Error("evidence schema mismatch")
    unsigned = {key: item for key, item in value.items() if key != "seal_sha256"}
    if value.get("seal_sha256") != _digest(unsigned):
        raise R6Error("evidence seal mismatch")
    return value


def _symlink_component(path: Path) -> bool:
    current = path.expanduser().absolute()
    while True:
        if current.is_symlink():
            return True
        if current.parent == current:
            return False
        current = current.parent


def _overlap(left: Path, right: Path) -> bool:
    a, b = left.resolve(strict=False), right.resolve(strict=False)
    return a == b or a.is_relative_to(b) or b.is_relative_to(a)


def assert_root_matrix(production: Path, source: Path, output: Path) -> None:
    roots = (production, source, output)
    if any(_symlink_component(path) for path in roots):
        raise R6Error("root path contains a symlink")
    if any(_overlap(left, right) for index, left in enumerate(roots) for right in roots[index + 1 :]):
        raise R6Error("root paths overlap")
    if not production.is_dir() or not source.is_dir():
        raise R6Error("production/source root is not a directory")
    if output.exists() and not output.is_dir():
        raise R6Error("output root is not a directory")


def _assert_no_overlap(*paths: Path) -> None:
    if any(path.is_symlink() for path in paths):
        raise R6Error("clone/root path contains a symlink")
    if any(_overlap(left, right) for index, left in enumerate(paths) for right in paths[index + 1 :]):
        raise R6Error("clone/root paths overlap")


def _reject_ambient_git_env() -> None:
    """Never let an ambient GIT_* variable redirect source inspection."""

    present = sorted(name for name in os.environ if name in _GIT_REDIRECT_KEYS)
    if present:
        raise R6Error("ambient GIT_* environment is forbidden")
    for name in tuple(os.environ):
        if name.startswith(_GIT_ENV_PREFIX):
            os.environ.pop(name, None)


def _git_env() -> dict[str, str]:
    _reject_ambient_git_env()
    return {key: value for key, value in os.environ.items() if not key.startswith(_GIT_ENV_PREFIX)}


def _assert_guard_modules() -> None:
    """Fail closed if a preloaded/fake stdlib module would evade our guards."""

    globals_by_name = {
        "os": os,
        "socket": socket,
        "subprocess": subprocess,
        "signal": signal,
        "io": io,
    }
    for name, expected in _GUARDED_STDLIB_MODULES.items():
        if globals_by_name[name] is not expected or sys.modules.get(name) is not expected:
            raise R6Error(f"preloaded {name} module is not the trusted stdlib")


def _assert_source_import_surface(source: Path) -> None:
    """Reject ignored bytecode/symlink import surfaces before any module load."""

    root = source.resolve(strict=True)
    import_root = root
    try:
        entries = sorted(import_root.rglob("*"))
    except OSError as exc:
        raise R6Error("source import surface cannot be inspected") from exc
    for path in entries:
        # Git/virtualenv/build metadata is inspected separately and is not an
        # import surface.  The source tree itself (including ignored files)
        # remains fail-closed.
        ignored_roots = {".git", ".venv", "venv", "build", "dist", "node_modules"}
        relative = path.relative_to(import_root)
        if relative.parts and relative.parts[0] in ignored_roots:
            continue
        try:
            state = path.lstat()
        except OSError as exc:
            raise R6Error("source import surface changed during inspection") from exc
        if stat.S_ISLNK(state.st_mode):
            raise R6Error("source import surface contains a symlink")
        if path.is_dir() and path.name == "__pycache__":
            raise R6Error("source import surface contains __pycache__")
        if path.is_file() and path.suffix.lower() in {".pyc", ".pyo"}:
            raise R6Error("source import surface contains bytecode")


@contextlib.contextmanager
def _import_side_effect_guards() -> Iterator[None]:
    """Prevent import-time egress, child execution, and writes."""

    _assert_guard_modules()
    egress_attempts = 0
    provider_attempts = 0

    def forbidden(reason: str, *, egress: bool = True) -> Any:
        nonlocal egress_attempts
        if egress:
            egress_attempts += 1
        raise R6GuardError(
            reason,
            egress_attempts=egress_attempts,
            provider_attempts=provider_attempts,
        )

    def forbidden_egress(*_args: Any, **_kwargs: Any) -> Any:
        return forbidden("import-time network/process side effect")

    def guarded_open(file: Any, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        if any(flag in mode for flag in ("w", "a", "x", "+")):
            return forbidden("import-time write side effect", egress=False)
        return original_builtin_open(file, mode, *args, **kwargs)

    def guarded_io_open(file: Any, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        if any(flag in mode for flag in ("w", "a", "x", "+")):
            return forbidden("import-time write side effect", egress=False)
        return original_io_open(file, mode, *args, **kwargs)

    def guarded_os_open(file: Any, flags: int, mode: int = 0o777, *args: Any, **kwargs: Any) -> Any:
        write_flags = (
            getattr(os, "O_WRONLY", 1)
            | getattr(os, "O_RDWR", 2)
            | getattr(os, "O_CREAT", 0)
            | getattr(os, "O_TRUNC", 0)
            | getattr(os, "O_APPEND", 0)
        )
        if flags & write_flags:
            return forbidden("import-time write side effect", egress=False)
        return original_os_open(file, flags, mode, *args, **kwargs)

    def guarded_import(name: str, globals: Any = None, locals: Any = None, fromlist: Any = (), level: int = 0) -> Any:
        if level == 0 and name.partition(".")[0] in _FORBIDDEN_NATIVE_IMPORTS:
            return forbidden("import-time native FFI side effect", egress=False)
        return original_import(name, globals, locals, fromlist, level)

    original_socket = {
        "connect": socket.socket.connect,
        "connect_ex": socket.socket.connect_ex,
        "send": socket.socket.send,
        "sendall": socket.socket.sendall,
        "sendto": socket.socket.sendto,
        "sendmsg": getattr(socket.socket, "sendmsg", None),
        "sendfile": getattr(socket.socket, "sendfile", None),
    }
    original_create_connection = socket.create_connection
    original_getaddrinfo = socket.getaddrinfo
    original_run = subprocess.run
    original_popen = subprocess.Popen
    original_system = os.system
    original_popen_os = getattr(os, "popen", None)
    original_os_open = os.open
    original_builtin_open = builtins.open
    original_io_open = io.open
    original_import = builtins.__import__
    original_path_methods = {
        name: getattr(Path, name)
        for name in ("write_text", "write_bytes", "touch", "unlink", "mkdir", "rmdir")
    }
    original_os_methods = {
        name: getattr(os, name)
        for name in ("unlink", "remove", "rename", "replace", "mkdir", "makedirs", "rmdir")
        if hasattr(os, name)
    }
    exec_names = (
        "execl", "execle", "execlp", "execv", "execve", "execvp", "execvpe",
        "spawnv", "spawnve", "spawnvp", "spawnvpe", "spawnl", "spawnle", "spawnlp", "spawnlpe",
        "fork", "forkpty", "kill", "killpg", "_exit", "abort",
    )
    original_exec = {name: getattr(os, name) for name in exec_names if hasattr(os, name)}
    original_sendfile = getattr(os, "sendfile", None)
    try:
        for name in original_socket:
            if original_socket[name] is not None:
                setattr(socket.socket, name, forbidden_egress)
        socket.create_connection = forbidden_egress
        socket.getaddrinfo = forbidden_egress
        subprocess.run = forbidden_egress
        subprocess.Popen = forbidden_egress  # type: ignore[assignment, misc]
        os.system = forbidden_egress
        if original_popen_os is not None:
            os.popen = forbidden_egress
        for name in original_exec:
            setattr(os, name, forbidden_egress)
        if original_sendfile is not None:
            os.sendfile = forbidden_egress
        builtins.open = guarded_open
        builtins.__import__ = guarded_import
        io.open = guarded_io_open
        os.open = guarded_os_open  # type: ignore[assignment]
        for name in original_path_methods:
            setattr(Path, name, forbidden_egress)
        for name in original_os_methods:
            setattr(os, name, forbidden_egress)
        yield
    finally:
        for name, original in original_socket.items():
            if original is not None:
                setattr(socket.socket, name, original)
        socket.create_connection = original_create_connection
        socket.getaddrinfo = original_getaddrinfo
        subprocess.run = original_run
        subprocess.Popen = original_popen  # type: ignore[misc]
        os.system = original_system
        if original_popen_os is not None:
            os.popen = original_popen_os
        for name, original in original_exec.items():
            setattr(os, name, original)
        if original_sendfile is not None:
            os.sendfile = original_sendfile
        builtins.open = original_builtin_open
        builtins.__import__ = original_import
        io.open = original_io_open
        os.open = original_os_open
        for name, original in original_path_methods.items():
            setattr(Path, name, original)
        for name, original in original_os_methods.items():
            setattr(os, name, original)


def _run_external_bounded(
    command: list[str],
    *,
    timeout: float,
    check: bool = True,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    worker_roots: tuple[Path, Path] | None = None,
    popen: Any = subprocess.Popen,
) -> subprocess.CompletedProcess[str]:
    """Run one harness-approved child without targeting ambient processes."""

    if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or not 0 < float(timeout) <= 120:
        raise R6Error("external child timeout is invalid")
    if not command or not all(isinstance(part, str) for part in command):
        raise R6Error("external child command is invalid")

    # ``sandbox-exec`` replaces the active sandbox instead of composing with
    # it.  Reject a direct nested invocation before allocating any child
    # resources, and prohibit it from the outer policy as well so a
    # subsequently-execed command cannot re-enable fork.  The basename check
    # covers a PATH invocation; canonicalization covers spelling variants of
    # the trusted absolute path.
    sandbox = _sandbox_identity()
    command_program = command[0]
    try:
        command_canonical = os.path.realpath(command_program)
    except (OSError, TypeError):
        command_canonical = ""
    if Path(command_program).name == "sandbox-exec" or command_canonical == sandbox["path"]:
        raise R6Error("nested sandbox-exec command is forbidden")

    # The read end is parent-owned.  A child may request only a PID; this parent
    # attests current ancestry and supplies the start token itself.  The stored
    # PID/start-token pair then survives setsid reparenting without trusting a
    # child-supplied identity claim.
    registry_read, registry_seed = os.pipe()
    # Keep the child capability out of subprocess's low-FD stdio/error-pipe
    # plumbing.  On Darwin those transient descriptors can otherwise replace a
    # low-numbered pass_fds entry during exec.
    registry_write = fcntl.fcntl(registry_seed, fcntl.F_DUPFD, 64)
    os.close(registry_seed)
    registry_ack_seed, registry_ack_write = os.pipe()
    registry_ack_read = fcntl.fcntl(registry_ack_seed, fcntl.F_DUPFD, 65)
    os.close(registry_ack_seed)
    os.set_blocking(registry_read, False)
    os.set_inheritable(registry_write, True)
    os.set_inheritable(registry_ack_read, True)
    registry_buffer = ""
    registry_eof = False
    registry_rejected = 0
    descendants: dict[int, str] = {}
    rejected_pids: set[int] = set()

    def process_table() -> dict[int, tuple[int, str, str]]:
        try:
            probe = popen(
                ["/bin/ps", "-axo", "pid=,ppid=,stat=,lstart="],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                start_new_session=False,
            )
            stdout_probe, _ = probe.communicate(timeout=0.5)
        except (OSError, subprocess.TimeoutExpired, ValueError):
            return {}
        table: dict[int, tuple[int, str, str]] = {}
        for line in stdout_probe.splitlines():
            fields = line.strip().split(None, 7)
            if len(fields) != 8:
                continue
            try:
                child_pid, parent_pid = int(fields[0]), int(fields[1])
            except ValueError:
                continue
            table[child_pid] = (parent_pid, fields[2], " ".join(fields[3:]))
        return table

    child_env = dict(env) if env is not None else _git_env()
    child_env["R6_DESCENDANT_REGISTRY_FD"] = str(registry_write)
    child_env["R6_DESCENDANT_REGISTRY_ACK_FD"] = str(registry_ack_read)
    # The generic helper must not rely on a child honestly registering forks:
    # a late setsid grandchild can otherwise outlive both its group and parent.
    # The parent profile is always outermost.  In particular, a caller cannot
    # supply a permissive sandbox-exec profile that re-enables fork beneath it.
    if worker_roots is None:
        parent_profile = "(version 1) (allow default) (allow process-exec) (deny process-fork)"
    else:
        parent_profile = _worker_sandbox_profile(source=worker_roots[0], clone=worker_roots[1])
    parent_profile += "\n(deny process-exec (literal \"/usr/bin/sandbox-exec\"))"
    launch_command = [sandbox["path"], "-p", parent_profile, "--", *command]
    try:
        process = popen(
            launch_command,
            cwd=cwd,
            env=child_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
            pass_fds=(registry_write, registry_ack_read),
        )
    except BaseException:
        os.close(registry_read)
        os.close(registry_write)
        os.close(registry_ack_read)
        os.close(registry_ack_write)
        raise
    os.close(registry_write)
    os.close(registry_ack_read)

    def is_descendant(pid: int, table: Mapping[int, tuple[int, str, str]]) -> bool:
        seen: set[int] = set()
        current = pid
        while current not in seen:
            seen.add(current)
            row = table.get(current)
            if row is None:
                return False
            if row[0] == process.pid:
                return True
            current = row[0]
        return False

    def drain_registry(table: Mapping[int, tuple[int, str, str]]) -> None:
        nonlocal registry_buffer, registry_eof, registry_rejected
        while not registry_eof:
            try:
                payload = os.read(registry_read, 4096)
            except BlockingIOError:
                break
            if not payload:
                registry_eof = True
                break
            try:
                registry_buffer += payload.decode("ascii")
            except UnicodeDecodeError:
                registry_rejected += 1
                registry_buffer = ""
                continue
            lines = registry_buffer.split("\n")
            registry_buffer = lines.pop()
            for line in lines:
                try:
                    child_pid = int(line)
                except ValueError:
                    registry_rejected += 1
                    continue
                row = table.get(child_pid)
                if row is None or not is_descendant(child_pid, table):
                    registry_rejected += 1
                    continue
                descendants.setdefault(child_pid, row[2])
                try:
                    os.write(registry_ack_write, b"A\n")
                except (OSError, ProcessLookupError):
                    pass

    def capture_descendants() -> dict[int, tuple[int, str, str]]:
        table = process_table()
        for child_pid, (_parent_pid, _state, started) in table.items():
            if child_pid != process.pid and is_descendant(child_pid, table):
                descendants.setdefault(child_pid, started)
        drain_registry(table)
        return table

    def live_descendants(_table: Mapping[int, tuple[int, str, str]]) -> list[int]:
        """Re-check only registered PIDs, so reparenting never loses them."""

        nonlocal registry_rejected
        live: list[int] = []
        for child_pid, started in descendants.items():
            try:
                probe = popen(
                    ["/bin/ps", "-o", "pid=,stat=,lstart=", "-p", str(child_pid)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True,
                    start_new_session=False,
                )
                stdout_probe, _ = probe.communicate(timeout=0.5)
            except (OSError, subprocess.TimeoutExpired, ValueError):
                continue
            fields = stdout_probe.strip().split()
            if len(fields) == 7 and fields[0] == str(child_pid) and " ".join(fields[2:]) == started:
                if not fields[1].startswith("Z"):
                    live.append(child_pid)
                continue
            if child_pid not in rejected_pids:
                rejected_pids.add(child_pid)
                registry_rejected += 1
        return live

    def kill_descendants() -> None:
        table = capture_descendants()
        for child_pid in sorted(live_descendants(table), reverse=True):
            try:
                os.kill(child_pid, signal.SIGKILL)
            except (OSError, ProcessLookupError):
                pass

    def close_registry() -> None:
        os.close(registry_read)
        os.close(registry_ack_write)

    def settle_registry(deadline: float) -> dict[int, tuple[int, str, str]]:
        while True:
            table = capture_descendants()
            if live_descendants(table) or registry_eof or time.monotonic() >= deadline:
                return table
            time.sleep(0.01)

    try:
        deadline = time.monotonic() + float(timeout)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(process.args, float(timeout))
            try:
                stdout, stderr = process.communicate(timeout=min(0.01, remaining))
                break
            except subprocess.TimeoutExpired:
                capture_descendants()
    except subprocess.TimeoutExpired as exc:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except (OSError, ProcessLookupError):
            try:
                process.kill()
            except OSError:
                pass
        kill_descendants()
        try:
            process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                process.kill()
            except OSError:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired as wait_exc:
                close_registry()
                raise R6Error("external child did not exit after watchdog kill") from wait_exc
        kill_descendants()
        table = settle_registry(time.monotonic() + 1)
        survivors = live_descendants(table)
        close_registry()
        if survivors:
            raise R6Error("external child descendant survived watchdog kill") from exc
        raise R6Error("external child watchdog timed out") from exc

    # A quick parent can exit before its reparented grandchild gets scheduled.
    # Keep the parent-held capability open long enough to drain that final
    # registration; the worker phase watchdog remains the outer bound.
    table = settle_registry(time.monotonic() + 1.0)
    if live_descendants(table):
        kill_descendants()
        close_registry()
        raise R6Error("external child left descendants")
    if not registry_eof:
        # This is fail-closed, but it must also actively terminate everything
        # still within parent-owned kernel/process-group containment.
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except (OSError, ProcessLookupError):
            try:
                process.kill()
            except OSError:
                pass
        kill_descendants()
        table = settle_registry(time.monotonic() + 1)
        if live_descendants(table):
            close_registry()
            raise R6Error("external child descendant survived registry failure")
        close_registry()
        raise R6Error("external child registry FD remained open")
    close_registry()
    if check and process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, command, stdout, stderr)
    completed = subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
    completed.r6_containment = {  # type: ignore[attr-defined]
        "schema": "chronovisor.recall-r6-child-containment.v1",
        "registered_descendants": len(descendants),
        "rejected_registry_entries": registry_rejected,
        "remaining_descendants": 0,
        "registry_fd_closed": True,
    }
    return completed


@contextlib.contextmanager
def _phase_watchdog(timeout: float = _PHASE_TIMEOUT_SECONDS) -> Iterator[None]:
    """Bound the whole phase with an authoritative parent watchdog child.

    The SIGALRM timer remains a local auxiliary for prompt failure.  The
    independent watchdog child sends SIGALRM after the deadline, so a worker
    cannot cancel the phase by clearing the in-process timer.
    """

    if threading.current_thread() is not threading.main_thread() or not hasattr(signal, "setitimer"):
        yield
        return
    if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or not 0 < float(timeout) <= 120:
        raise R6Error("phase watchdog timeout is invalid")
    _assert_guard_modules()
    auxiliary_setitimer = signal.setitimer
    auxiliary_signal = signal.signal
    # Capture process-group controls before entering the worker's guard
    # window.  The SIGALRM handler runs in this parent process while the
    # worker monkey-patches ``os.killpg``/``os.getpgid``; using the live module
    # attributes here would turn watchdog cleanup into a false egress attempt.
    watchdog_killpg = os.killpg
    watchdog_getpgid = os.getpgid
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    watchdog = None

    def alarm(_signum: int, _frame: Any) -> None:
        # Stop the independent child immediately.  Otherwise it could deliver
        # a second SIGALRM while the failure path is proving source/production
        # immutability and cleaning the clone.
        if watchdog is not None:
            try:
                watchdog_killpg(watchdog_getpgid(watchdog.pid), signal.SIGKILL)
            except (OSError, ProcessLookupError):
                pass
        auxiliary_setitimer(signal.ITIMER_REAL, 0)
        raise R6Error("R6 phase watchdog timed out")

    auxiliary_signal(signal.SIGALRM, alarm)
    auxiliary_setitimer(signal.ITIMER_REAL, float(timeout))
    try:
        code = (
            "import os,signal,time; "
            "time.sleep(float(os.environ['R6_WATCHDOG_TIMEOUT'])); "
            "os.kill(int(os.environ['R6_WATCHDOG_PARENT']), signal.SIGALRM)"
        )
        watchdog_env = _git_env()
        watchdog_env.update({
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "R6_WATCHDOG_TIMEOUT": str(float(timeout)),
            "R6_WATCHDOG_PARENT": str(os.getpid()),
        })
        watchdog = subprocess.Popen(
            [sys.executable, "-I", "-S", "-B", "-c", code],
            env=watchdog_env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        yield
    finally:
        if watchdog is not None:
            try:
                watchdog_killpg(watchdog_getpgid(watchdog.pid), signal.SIGKILL)
            except (OSError, ProcessLookupError):
                try:
                    watchdog.kill()
                except OSError:
                    pass
            try:
                watchdog.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    watchdog.kill()
                except OSError:
                    pass
                try:
                    watchdog.wait(timeout=5)
                except subprocess.TimeoutExpired as wait_exc:
                    raise R6Error("phase watchdog child did not exit") from wait_exc
        auxiliary_setitimer(signal.ITIMER_REAL, 0)
        auxiliary_signal(signal.SIGALRM, previous_handler)
        if previous_timer != (0.0, 0.0):
            auxiliary_setitimer(signal.ITIMER_REAL, *previous_timer)


def _git_layout(source: Path) -> dict[str, Any]:
    """Capture the actual .git file/dir and absolute index identities."""

    _reject_ambient_git_env()
    root = source.resolve(strict=True)
    entry = root / ".git"
    try:
        entry_state = entry.lstat()
    except OSError as exc:
        raise R6Error("source .git entry is unavailable") from exc
    if entry.is_symlink() or not (stat.S_ISDIR(entry_state.st_mode) or stat.S_ISREG(entry_state.st_mode)):
        raise R6Error("source .git entry is unsafe")
    entry_bytes = entry.read_bytes() if stat.S_ISREG(entry_state.st_mode) else b""
    entry_after = entry.lstat()
    if (entry_state.st_dev, entry_state.st_ino, entry_state.st_size, entry_state.st_mtime_ns) != (
        entry_after.st_dev, entry_after.st_ino, entry_after.st_size, entry_after.st_mtime_ns
    ):
        raise R6Error("source .git entry changed during read")
    if stat.S_ISREG(entry_state.st_mode):
        try:
            marker = entry_bytes.decode("utf-8").strip()
        except UnicodeDecodeError as exc:
            raise R6Error("source .git file is invalid") from exc
        if not marker.startswith("gitdir:"):
            raise R6Error("source .git file has no gitdir")
        raw_git_dir = marker.partition(":")[2].strip()
        if not raw_git_dir:
            raise R6Error("source .git file has an empty gitdir")
        git_dir = Path(raw_git_dir)
        if not git_dir.is_absolute():
            git_dir = (entry.parent / git_dir).resolve(strict=True)
        else:
            git_dir = git_dir.resolve(strict=True)
    else:
        git_dir = entry.resolve(strict=True)
    if _symlink_component(git_dir) or not git_dir.is_dir():
        raise R6Error("source git directory is unsafe")
    index = git_dir / "index"
    index_state = index.lstat() if index.exists() else None
    if index_state is not None and (index.is_symlink() or not stat.S_ISREG(index_state.st_mode)):
        raise R6Error("source git index is unsafe")
    index_bytes = index.read_bytes() if index_state is not None else b""
    if index_state is not None:
        index_after = index.lstat()
        if (index_state.st_dev, index_state.st_ino, index_state.st_size, index_state.st_mtime_ns) != (
            index_after.st_dev, index_after.st_ino, index_after.st_size, index_after.st_mtime_ns
        ):
            raise R6Error("source git index changed during read")
    git_dir_state = git_dir.stat()
    return {
        "entry": str(entry.resolve(strict=True)),
        "entry_sha256": hashlib.sha256(entry_bytes).hexdigest(),
        "entry_bytes": entry_state.st_size,
        "entry_inode": entry_state.st_ino,
        "entry_mode": entry_state.st_mode & 0o7777,
        "git_dir": str(git_dir),
        "git_dir_inode": git_dir_state.st_ino,
        "git_dir_mode": git_dir_state.st_mode & 0o7777,
        "work_tree": str(root),
        "index": str(index.resolve(strict=False)),
        "index_sha256": hashlib.sha256(index_bytes).hexdigest() if index_state is not None else "",
        "index_bytes": index_state.st_size if index_state is not None else 0,
        "index_inode": index_state.st_ino if index_state is not None else 0,
        "index_mode": index_state.st_mode & 0o7777 if index_state is not None else 0,
    }


def _assert_git_layout_unchanged(source: Path, expected: Mapping[str, Any]) -> None:
    actual = _git_layout(source)
    if set(expected) != _GIT_LAYOUT_KEYS or actual != dict(expected):
        raise R6Error("source git layout changed")


def _git(source: Path, *args: str) -> str:
    layout = _git_layout(source)
    try:
        return _run_external_bounded(
            [str(_trusted_executable("/usr/bin/git")), "--no-optional-locks", f"--git-dir={layout['git_dir']}", f"--work-tree={layout['work_tree']}", *args],
            cwd=source, timeout=5, env=_git_env(),
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired, R6Error) as exc:
        raise R6Error("source git inspection failed") from exc


def source_snapshot(source: Path) -> dict[str, Any]:
    _assert_source_import_surface(source)
    layout_before = _git_layout(source)
    module_path = source / "src" / "chronovisor" / "recall" / "recall_distillation.py"
    try:
        module_sha256 = hashlib.sha256(module_path.read_bytes()).hexdigest()
    except OSError:
        module_sha256 = ""
    status = _git(source, "status", "--porcelain=v1", "-z")
    tracked = _git(source, "ls-files", "-z").split("\0")
    files: list[dict[str, Any]] = []
    for relative in sorted(value for value in tracked if value):
        path = source / relative
        try:
            before = path.lstat()
            if path.is_symlink() or not stat.S_ISREG(before.st_mode):
                raise R6Error("source tree contains an unsafe path")
            payload = path.read_bytes()
            after = path.lstat()
        except OSError as exc:
            raise R6Error("source tree cannot be read") from exc
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns
        ):
            raise R6Error("source tree changed during read")
        files.append({
            "path": relative,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "nlink": before.st_nlink,
        })
    layout_after = _git_layout(source)
    if layout_after != layout_before:
        raise R6Error("source git layout changed during inspection")
    return {
        "head": _git(source, "rev-parse", "HEAD"),
        "status": _digest(status),
        "status_count": status.count("\0"),
        "tree_sha256": _digest(files),
        "git_executable_sha256": hashlib.sha256(_trusted_executable("/usr/bin/git").read_bytes()).hexdigest(),
        "module_sha256": module_sha256,
        "git_layout": layout_after,
    }


def _tree_identity(root: Path) -> str:
    rows: list[dict[str, Any]] = []
    try:
        root_before = root.lstat()
        if root.is_symlink() or not stat.S_ISDIR(root_before.st_mode):
            raise R6Error("root identity path is unsafe")
        for path in sorted(root.rglob("*")):
            entry = path.lstat()
            if stat.S_ISLNK(entry.st_mode):
                raise R6Error("root contains a symlink")
            if stat.S_ISDIR(entry.st_mode):
                continue
            if not stat.S_ISREG(entry.st_mode) or entry.st_size > MAX_FILE_BYTES:
                raise R6Error("root contains an unsupported file")
            payload = path.read_bytes()
            final = path.lstat()
            if (entry.st_dev, entry.st_ino, entry.st_size, entry.st_mtime_ns) != (
                final.st_dev, final.st_ino, final.st_size, final.st_mtime_ns
            ):
                raise R6Error("root changed during identity read")
            rows.append({"path": str(path.relative_to(root)), "bytes": entry.st_size, "sha256": hashlib.sha256(payload).hexdigest()})
        root_after = root.lstat()
        if (root_before.st_dev, root_before.st_ino, root_before.st_mtime_ns) != (
            root_after.st_dev, root_after.st_ino, root_after.st_mtime_ns
        ):
            raise R6Error("root changed during identity read")
    except OSError as exc:
        raise R6Error("root identity cannot be read") from exc
    return _digest(rows)


def _stable_file_identity(path: Path) -> str:
    try:
        before = path.lstat()
        if path.is_symlink() or not stat.S_ISREG(before.st_mode) or before.st_size > MAX_FILE_BYTES:
            raise R6Error("artifact file is unsafe")
        payload = path.read_bytes()
        after = path.lstat()
    except OSError as exc:
        raise R6Error("artifact file cannot be read") from exc
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns
    ):
        raise R6Error("artifact file changed during read")
    return hashlib.sha256(payload).hexdigest()


def _assert_clone_contained(clone: Path) -> None:
    """Reject links that could turn a clone write into a source write."""

    try:
        root = clone.resolve(strict=True)
        root_state = root.lstat()
        if root.is_symlink() or not stat.S_ISDIR(root_state.st_mode):
            raise R6Error("clone root is unsafe")
        for path in root.rglob("*"):
            state = path.lstat()
            if stat.S_ISLNK(state.st_mode):
                raise R6Error("clone contains a symlink")
            if stat.S_ISREG(state.st_mode) and state.st_nlink != 1:
                raise R6Error("clone contains a hardlink")
    except OSError as exc:
        raise R6Error("clone containment cannot be inspected") from exc


def _r5_preflight(module: Any, clone: Path, source: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute the official P5 baseline from the frozen clone, never trust a name."""

    config_path = clone / "config.toml"
    try:
        baseline = module.preflight(
            raw_dir=clone / "raw",
            root=clone,
            config_path=config_path if config_path.exists() else None,
            runtime_commit=str(source["head"]),
        )
        store = module.store
        baseline_id = _artifact_id(baseline.get("artifact_id"))
        stored = store.read_sealed(
            store.distillation_dir(clone) / "baselines" / f"{baseline_id}.json",
            schema=module.BASELINE_SCHEMA,
        )
        unsigned = {key: value for key, value in stored.items() if key not in {"artifact_id", "seal_sha256"}}
        if stored != baseline or store.canonical_json_sha256_strict(unsigned) != baseline_id:
            raise R6Error("official baseline readback identity mismatch")
        config_bytes = config_path.read_bytes() if config_path.exists() else b""
        if stored.get("config_sha256") != hashlib.sha256(config_bytes).hexdigest():
            raise R6Error("official baseline config identity mismatch")
        if stored.get("runtime_commit") != source["head"]:
            raise R6Error("official baseline source commit mismatch")
        if stored.get("raw_watermark") != module.committed_raw_watermark(clone / "raw"):
            raise R6Error("official baseline raw watermark mismatch")
        config = module.load_distillation_config(config_path if config_path.exists() else None)
        training = module.materialize_training_rows(clone)
        rows = training.get("rows")
        if not isinstance(rows, list):
            raise R6Error("official training rows are unavailable")
        profile_contract = (
            module._current_ox_profile_contract_id(clone)
            if config.teacher_profile == module.OX_SINGLE_PROFILE
            else ""
        )
        _, cohort = module._active_training_cohort(
            rows,
            teacher_profile=config.teacher_profile,
            profile_contract_id=profile_contract,
        )
        offline = module._offline_training_gate(rows, config, root=clone)
        split_plan = module._read_split_plan(clone)
        if (
            stored.get("offline_training_gate") != offline
            or offline.get("passed") is not True
            or stored.get("hard_floor", {}).get("p5_allowed") is not True
            or split_plan.get("model_cohort_sha256") != cohort.get("cohort_sha256")
            or split_plan.get("feature_revision") != module.TEXT_FEATURE_REVISION
            or module._matching_p5_baseline(clone, stored) is None
        ):
            raise R6Error("official baseline current cohort/split gate failed")
        return {
            "passed": True,
            "reason": "",
            "baseline_id": baseline_id,
            "raw_watermark": stored["raw_watermark"],
            "label_head": stored["label_chain_head"],
            "cohort_sha256": cohort["cohort_sha256"],
            "split_plan_id": split_plan["artifact_id"],
            "profile_contract_id": profile_contract,
        }
    except (module.DistillationError, module.store.DistillationStoreError, KeyError, OSError, R6Error) as exc:
        return {"passed": False, "reason": str(exc).split(":", 1)[0], "baseline_id": ""}


def _target_identity(root: Path) -> str:
    """Pre-copy identity for only the paths the official worker can consume."""

    rows: list[dict[str, Any]] = []
    for relative in _CLONE_TARGETS:
        path = root / relative
        if not path.exists():
            if relative == Path("raw"):
                raise R6Error("required Raw root is missing")
            continue
        if path.is_symlink():
            raise R6Error("official clone target contains a symlink")
        paths = [path] if path.is_file() else [path, *sorted(path.rglob("*"))]
        for item in paths:
            state = item.lstat()
            if stat.S_ISLNK(state.st_mode):
                raise R6Error("official clone target contains a symlink")
            if stat.S_ISDIR(state.st_mode):
                continue
            if not stat.S_ISREG(state.st_mode):
                raise R6Error("official clone target contains an unsafe file")
            digest = hashlib.sha256()
            with item.open("rb") as handle:
                while chunk := handle.read(1024 * 1024):
                    digest.update(chunk)
            final = item.lstat()
            if (state.st_dev, state.st_ino, state.st_size, state.st_mtime_ns) != (
                final.st_dev, final.st_ino, final.st_size, final.st_mtime_ns
            ):
                raise R6Error("official clone target changed during identity read")
            rows.append({
                "path": item.relative_to(root).as_posix(), "size": state.st_size,
                "mtime_ns": state.st_mtime_ns, "mode": state.st_mode & 0o7777,
                "sha256": digest.hexdigest(),
            })
    return _digest(rows)


def _clone_stat(path: Path) -> dict[str, Any]:
    try:
        state = path.lstat()
    except OSError as exc:
        raise R6Error("clone stat is unavailable") from exc
    if path.is_symlink() or not stat.S_ISDIR(state.st_mode):
        raise R6Error("clone stat is unsafe")
    return {
        "path": str(path.resolve(strict=True)),
        "dev": state.st_dev,
        "ino": state.st_ino,
        "uid": state.st_uid,
        "mode": state.st_mode & 0o7777,
    }


def _clone(
    production: Path,
    *,
    source: Path | None = None,
    with_proof: bool = False,
) -> Any:
    _reject_ambient_git_env()
    if sys.platform != "darwin" or _filesystem_type(production) != "apfs":
        raise R6Error("production volume is not APFS")
    before = _target_identity(production)
    parent = Path(tempfile.mkdtemp(prefix="chronovisor-r6-")).resolve(strict=True)
    clone = parent / "root"
    try:
        clone.mkdir()
        for relative in _CLONE_TARGETS:
            source_path = production / relative
            if not source_path.exists():
                continue
            destination = clone / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            _run_external_bounded(
                [str(_trusted_executable("/bin/cp")), "-cR", str(source_path), str(destination)],
                timeout=120, env=_git_env(),
            )
        if _target_identity(clone) == "":
            raise R6Error("APFS clone target is empty")
        production_state, clone_state = production.stat(), clone.stat()
        clone_filesystem = _filesystem_type(clone)
        if (
            production_state.st_dev != clone_state.st_dev
            or clone_state.st_uid != os.geteuid()
            or _overlap(production, clone)
            or clone_filesystem != "apfs"
        ):
            raise R6Error("clone volume, owner, or overlap proof failed")
        proof = {
            "schema": "chronovisor.recall-r6-clone-proof.v1",
            "namespace": "recall-distillation",
            "kind": "clone-provenance",
            "method": _CLONE_METHOD,
            "production_path": str(production.resolve(strict=True)),
            "source_path": str(source.resolve(strict=True)) if source is not None else "",
            "clone_path": str(clone.resolve(strict=True)),
            "euid": os.geteuid(),
            "filesystem": "apfs",
            "production": _clone_stat(production),
            "clone": _clone_stat(clone),
            "target_before": before,
            "same_device": True,
            "nonoverlap": True,
        }
        if set(proof) != _CLONE_PROOF_KEYS:
            raise R6Error("clone provenance proof is not closed")
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired, R6Error) as exc:
        try:
            _cleanup_clone(clone)
        except R6Error as cleanup_exc:
            raise cleanup_exc from exc
        raise R6Error("APFS clone failed") from exc
    return (clone, before, proof) if with_proof else (clone, before)


def _cleanup_clone(clone: Path) -> None:
    parent = clone.parent
    if parent.is_symlink():
        try:
            parent.unlink()
        except OSError as exc:
            raise R6Error("clone cleanup parent is a symlink") from exc
        if parent.exists():
            raise R6Error("clone cleanup failed")
        return
    try:
        if clone.is_symlink() or clone.exists() and not clone.is_dir():
            clone.unlink()
        if parent.exists():
            if parent.is_dir():
                shutil.rmtree(parent)
            else:
                parent.unlink()
    except OSError as exc:
        raise R6Error("clone cleanup failed") from exc
    if parent.exists():
        raise R6Error("clone cleanup failed")


@contextlib.contextmanager
def _clone_runtime_context(clone: Path, source: Path) -> Iterator[None]:
    """Keep every import-time root/config/repo default inside this clone."""

    names = ("CHRONOVISOR_ROOT", "CHRONOVISOR_REPO_ROOT", "PWD")
    original = {name: os.environ.get(name) for name in names}
    original_modules = {
        name: value for name, value in sys.modules.items() if name == "chronovisor" or name.startswith("chronovisor.")
    }
    original_path = list(sys.path)
    try:
        os.environ["CHRONOVISOR_ROOT"] = str(clone)
        os.environ["CHRONOVISOR_REPO_ROOT"] = str(source)
        os.environ["PWD"] = str(clone)
        for name in tuple(original_modules):
            sys.modules.pop(name, None)
        yield
    finally:
        for name in tuple(sys.modules):
            if name == "chronovisor" or name.startswith("chronovisor."):
                sys.modules.pop(name, None)
        sys.modules.update(original_modules)
        sys.path[:] = original_path
        for name, value in original.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _load_runtime(source: Path, clone: Path) -> tuple[Any, dict[str, str]]:
    _assert_guard_modules()
    _assert_source_import_surface(source)
    source_module = (source / "src" / "chronovisor" / "recall" / "recall_distillation.py").resolve(strict=True)
    source_bytes = source_module.read_bytes()
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    source_src = str((source / "src").resolve(strict=True))
    # Remove every preloaded package member, not just the leaf module.  The
    # exact source loader below then recreates the package from this source
    # root, so a user-site module cannot supply a relative import.
    for name in tuple(sys.modules):
        if name == "chronovisor" or name.startswith("chronovisor."):
            sys.modules.pop(name, None)
    sys.path[:] = [source_src, *[item for item in sys.path if item != source_src]]
    original_dont_write = sys.dont_write_bytecode
    original_env = {
        name: os.environ.get(name)
        for name in ("PYTHONNOUSERSITE", "PYTHONDONTWRITEBYTECODE")
    }
    user_site = None
    try:
        import site

        user_site = site.getusersitepackages()
    except (AttributeError, OSError):
        user_site = None
    if user_site:
        user_site_path = str(Path(user_site).resolve(strict=False))
        sys.path[:] = [
            item for item in sys.path
            if not item or str(Path(item).resolve(strict=False)) != user_site_path
        ]

    class _ExactSourceLoader(importlib.machinery.SourceFileLoader):
        def get_code(self, fullname: str) -> Any:
            current = self.get_data(self.path)
            if hashlib.sha256(current).hexdigest() != source_sha256:
                raise R6Error("runtime source changed during exact load")
            return compile(current, self.path, "exec", dont_inherit=True)

    module_name = "chronovisor.recall.recall_distillation"
    try:
        os.environ["PYTHONNOUSERSITE"] = "1"
        os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
        sys.dont_write_bytecode = True
        importlib.invalidate_caches()
        spec = importlib.util.spec_from_file_location(
            module_name,
            source_module,
            loader=_ExactSourceLoader(module_name, str(source_module)),
        )
        if spec is None or spec.loader is None or not isinstance(spec.loader, _ExactSourceLoader):
            raise R6Error("runtime exact source loader is unavailable")
        module = importlib.util.module_from_spec(spec)
        with _import_side_effect_guards():
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = original_dont_write
        for name, value in original_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    if Path(str(module.__file__)).resolve() != source_module:
        raise R6Error("runtime module did not load from exact source root")
    _assert_source_import_surface(source)
    if hashlib.sha256(source_module.read_bytes()).hexdigest() != source_sha256:
        raise R6Error("runtime source changed after exact load")
    clone_resolved = clone.resolve()
    source_resolved = source.resolve()
    if (
        Path(module.CHRONOVISOR_ROOT).resolve() != clone_resolved
        or Path(module.store.CHRONOVISOR_ROOT).resolve() != clone_resolved
        or Path(module.runtime_config.CONFIG_FILE).resolve() != clone_resolved / "config.toml"
        or Path(module.runtime_config.runtime_repo_root()).resolve() != source_resolved
    ):
        raise R6Error("runtime root/config/repository isolation failed")
    return module, {
        "module_path": str(source_module.relative_to(source)),
        "module_sha256": source_sha256,
        "runtime_root": str(clone),
        "config_path": str(clone / "config.toml"),
        "repository_root": str(source),
    }


def _isolated_child_chunk(module: Any, clone: Path, source: Path) -> dict[str, Any]:
    """Run only inside the private isolated-child entry point."""

    _reject_ambient_git_env()
    _assert_guard_modules()
    guarded_module_entries = {
        name: sys.modules[name] for name in _GUARDED_STDLIB_MODULES
    }
    git_layout = _git_layout(source)
    git_env_before = {name: value for name, value in os.environ.items() if name.startswith(_GIT_ENV_PREFIX)}
    original_factory = module._default_workers
    original_connect = socket.socket.connect
    original_socket_class = socket.socket
    original_connect_ex = socket.socket.connect_ex
    original_create_connection = socket.create_connection
    original_getaddrinfo = socket.getaddrinfo
    original_sendto = socket.socket.sendto
    original_send = socket.socket.send
    original_sendall = socket.socket.sendall
    original_sendmsg = getattr(socket.socket, "sendmsg", None)
    original_sendfile = getattr(socket.socket, "sendfile", None)
    original_process = {
        name: getattr(subprocess, name)
        for name in ("run", "Popen", "call", "check_call", "check_output")
        if hasattr(subprocess, name)
    }
    original_os_process = {
        name: getattr(os, name)
        for name in (
            "system", "popen", "posix_spawn", "posix_spawnp", "sendfile",
            "kill", "killpg",
            "fork", "forkpty", "_exit", "abort",
            "execl", "execle", "execlp", "execv", "execve", "execvp", "execvpe",
            "spawnv", "spawnve", "spawnvp", "spawnvpe", "spawnl", "spawnle", "spawnlp", "spawnlpe",
        )
        if hasattr(os, name)
    }
    original_alarm = getattr(signal, "alarm", None)
    original_signal = getattr(signal, "signal", None)
    original_setitimer = getattr(signal, "setitimer", None)
    original_reload = importlib.reload
    original_invalidate_caches = importlib.invalidate_caches
    original_sys_exit = sys.exit
    socket_module: Any = socket
    socket_type: Any = socket.socket
    process_module: Any = subprocess
    original_popen = original_process.get("Popen", subprocess.Popen)
    egress_attempts = 0
    provider_attempts = 0
    guard_violations = 0
    trusted_git_sha256 = ""
    source_root = source.resolve(strict=True)

    def forbidden_factory(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal provider_attempts
        provider_attempts += 1
        raise R6GuardError(
            "provider worker factory was called",
            egress_attempts=egress_attempts,
            provider_attempts=provider_attempts,
        )

    def forbidden_connect(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal egress_attempts
        egress_attempts += 1
        raise R6GuardError(
            "network egress was attempted",
            egress_attempts=egress_attempts,
            provider_attempts=provider_attempts,
        )

    def guarded_signal_change(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal guard_violations
        guard_violations += 1
        raise R6GuardError(
            "worker attempted to replace a signal handler",
            egress_attempts=egress_attempts,
            provider_attempts=provider_attempts,
        )

    def guarded_alarm(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal guard_violations
        guard_violations += 1
        raise R6GuardError(
            "worker attempted to alter the phase watchdog alarm",
            egress_attempts=egress_attempts,
            provider_attempts=provider_attempts,
        )

    def guarded_setitimer(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal guard_violations
        guard_violations += 1
        raise R6GuardError(
            "worker attempted to alter the phase watchdog timer",
            egress_attempts=egress_attempts,
            provider_attempts=provider_attempts,
        )

    def guarded_reload(target: Any) -> Any:
        nonlocal guard_violations
        guard_violations += 1
        if target in _GUARDED_STDLIB_MODULES.values() or getattr(target, "__name__", "") in _GUARDED_STDLIB_MODULES:
            raise R6GuardError(
                "worker attempted to reload a guarded stdlib module",
                egress_attempts=egress_attempts,
                provider_attempts=provider_attempts,
            )
        raise R6GuardError(
            "worker attempted to reload a module during official execution",
            egress_attempts=egress_attempts,
            provider_attempts=provider_attempts,
        )

    def guarded_invalidate_caches() -> None:
        nonlocal guard_violations
        guard_violations += 1
        raise R6GuardError(
            "worker attempted to invalidate import caches",
            egress_attempts=egress_attempts,
            provider_attempts=provider_attempts,
        )

    def guarded_sys_exit(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal guard_violations
        guard_violations += 1
        raise R6GuardError(
            "worker attempted to exit the harness process",
            egress_attempts=egress_attempts,
            provider_attempts=provider_attempts,
        )

    def guarded_run(command: object, *args: Any, **kwargs: Any) -> Any:
        """The runtime's local commit probe is not provider egress."""

        safe = {"cwd", "capture_output", "text", "timeout", "check"}
        if (
            command == ["git", "rev-parse", "origin/main"]
            and not args
            and set(kwargs).issubset(safe)
            and kwargs.get("capture_output") is True
            and kwargs.get("text") is True
            and kwargs.get("check", False) is False
            and isinstance(kwargs.get("timeout", 5), (int, float))
            and 0 < float(kwargs.get("timeout", 5)) <= 5
            and Path(str(kwargs.get("cwd", ""))).resolve(strict=True) == source_root
        ):
            trusted_git = Path("/usr/bin/git").resolve(strict=True)
            if not trusted_git.is_file() or trusted_git.is_symlink():
                raise R6Error("trusted git executable is unsafe")
            nonlocal trusted_git_sha256
            trusted_git_sha256 = hashlib.sha256(trusted_git.read_bytes()).hexdigest()
            completed = _run_external_bounded(
                [str(trusted_git), "--no-optional-locks", f"--git-dir={git_layout['git_dir']}", f"--work-tree={git_layout['work_tree']}", "rev-parse", "origin/main"],
                cwd=source_root,
                timeout=float(kwargs.get("timeout", 5)),
                check=False,
                env=_git_env(),
                popen=original_popen,
            )
            return subprocess.CompletedProcess(command, completed.returncode, completed.stdout, completed.stderr)
        return forbidden_connect(command, *args, **kwargs)

    module._default_workers = forbidden_factory
    socket_type.connect = forbidden_connect
    socket_type.connect_ex = forbidden_connect
    socket_module.create_connection = forbidden_connect
    socket_module.getaddrinfo = forbidden_connect
    socket_type.sendto = forbidden_connect
    socket_type.send = forbidden_connect
    socket_type.sendall = forbidden_connect
    if original_sendmsg is not None:
        socket_type.sendmsg = forbidden_connect
    if original_sendfile is not None:
        socket_type.sendfile = forbidden_connect
    process_module.run = guarded_run
    for name in original_process:
        if name != "run":
            setattr(process_module, name, forbidden_connect)
    for name in original_os_process:
        setattr(os, name, forbidden_connect)
    if original_alarm is not None:
        signal.alarm = guarded_alarm
    if original_signal is not None:
        signal.signal = guarded_signal_change
    if original_setitimer is not None:
        signal.setitimer = guarded_setitimer
    importlib.reload = guarded_reload  # type: ignore[assignment]
    importlib.invalidate_caches = guarded_invalidate_caches
    sys.exit = guarded_sys_exit
    worker_error: BaseException | None = None
    try:
        result = module.run_distillation_chunk(
            root=clone,
            raw_dir=clone / "raw",
            config_path=(clone / "config.toml") if (clone / "config.toml").exists() else None,
            teachers={},
            counterfactual=None,
            dry_run=False,
            cold_start=False,
            max_elapsed_seconds=60,
        )
    except BaseException as exc:
        worker_error = exc
    finally:
        module._default_workers = original_factory
        socket_type.connect = original_connect
        socket_type.connect_ex = original_connect_ex
        socket_module.create_connection = original_create_connection
        socket_module.getaddrinfo = original_getaddrinfo
        socket_type.sendto = original_sendto
        socket_type.send = original_send
        socket_type.sendall = original_sendall
        if original_sendmsg is not None:
            socket_type.sendmsg = original_sendmsg
        if original_sendfile is not None:
            socket_type.sendfile = original_sendfile
        for name, original in original_process.items():
            setattr(process_module, name, original)
        for name, original in original_os_process.items():
            setattr(os, name, original)
        if original_alarm is not None:
            signal.alarm = original_alarm
        if original_signal is not None:
            signal.signal = original_signal
        if original_setitimer is not None:
            signal.setitimer = original_setitimer
        importlib.reload = original_reload
        importlib.invalidate_caches = original_invalidate_caches
        sys.exit = original_sys_exit
        layout_error: R6Error | None = None
        try:
            _assert_git_layout_unchanged(source, git_layout)
        except R6Error as exc:
            layout_error = exc
        changed_git_env = [
            name for name in os.environ
            if name.startswith(_GIT_ENV_PREFIX) and os.environ.get(name) != git_env_before.get(name)
        ]
        removed_git_env = [name for name in git_env_before if name not in os.environ]
        for name in changed_git_env:
            if name in git_env_before:
                os.environ[name] = git_env_before[name]
            else:
                os.environ.pop(name, None)
        for name in removed_git_env:
            os.environ[name] = git_env_before[name]
        if layout_error is not None:
            raise layout_error
        if changed_git_env or removed_git_env:
            raise R6Error("worker introduced ambient GIT_* environment")
        if any(sys.modules.get(name) is not original for name, original in guarded_module_entries.items()):
            raise R6Error("worker replaced a guarded stdlib module")
        if socket.socket is not original_socket_class:
            raise R6Error("worker replaced the guarded socket class")
    if worker_error is not None:
        if egress_attempts or provider_attempts or guard_violations:
            reason = str(worker_error) or "guarded side effect was wrapped by worker"
            raise R6GuardError(
                reason,
                egress_attempts=egress_attempts,
                provider_attempts=provider_attempts,
            ) from worker_error
        raise worker_error
    if not isinstance(result, Mapping):
        raise R6Error("official worker result is invalid")
    if egress_attempts or provider_attempts or guard_violations:
        raise R6GuardError(
            "guarded provider or network attempt was swallowed by worker",
            egress_attempts=egress_attempts,
            provider_attempts=provider_attempts,
        )
    return {
        **result, "r6_egress_attempts": egress_attempts,
        "r6_provider_attempts": provider_attempts,
        "r6_git_sha256": trusted_git_sha256,
    }


def _test_only_official_chunk(module: Any, clone: Path, source: Path) -> dict[str, Any]:
    """Test seam for provider-guard unit cases; never the public worker API."""

    return _isolated_child_chunk(module, clone, source)


def _worker_sandbox_profile(*, source: Path, clone: Path) -> str:
    """Allow the disposable clone to change, but nothing outside it.

    The profile is supplied by the parent (not a file in the clone), so a
    worker cannot replace it.  ``sandbox-exec`` applies it before Python has a
    chance to import sitecustomize or user code.
    """

    def quoted(path: Path) -> str:
        # SBPL strings use a deliberately small escape surface.  Resolved
        # roots have already passed the no-symlink root matrix check.
        return str(path).replace("\\", "\\\\").replace('"', '\\"')

    return "\n".join((
        "(version 1)",
        "(deny default)",
        "(allow file-read*)",
        f'(allow file-write* (subpath "{quoted(clone)}"))',
        # No fork means no double-fork or post-return background process.  An
        # exec replacement remains inside this same sandbox/process group and
        # is reaped by the parent watchdog.
        "(deny process-fork)",
        "(allow process-exec)",
        "(deny network*)",
        "(deny signal)",
    ))


def _sandbox_identity() -> dict[str, Any]:
    """Read the exact parent-approved Darwin sandbox executable twice safely."""

    sandbox = Path("/usr/bin/sandbox-exec")
    try:
        if sandbox.resolve(strict=True) != sandbox or sandbox.is_symlink():
            raise R6Error("worker sandbox executable is unsafe")
        before = sandbox.lstat()
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid != 0
            or before.st_mode & 0o022
            or before.st_nlink != 1
        ):
            raise R6Error("worker sandbox executable is unsafe")
        payload = sandbox.read_bytes()
        after = sandbox.lstat()
    except OSError as exc:
        raise R6Error("worker sandbox executable is unavailable") from exc
    if (before.st_dev, before.st_ino, before.st_uid, before.st_mode, before.st_size, before.st_mtime_ns) != (
        after.st_dev, after.st_ino, after.st_uid, after.st_mode, after.st_size, after.st_mtime_ns
    ):
        raise R6Error("worker sandbox executable changed during read")
    return {
        "path": str(sandbox), "dev": before.st_dev, "ino": before.st_ino,
        "uid": before.st_uid, "mode": before.st_mode & 0o7777,
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _isolated_worker_env(*, source: Path, clone: Path, capability: str = "") -> dict[str, str]:
    """A fixed clean environment; inherited hooks never reach the worker."""

    env = {
        "HOME": str(clone),
        "PATH": "/usr/bin:/bin",
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "CHRONOVISOR_ROOT": str(clone),
        "CHRONOVISOR_REPO_ROOT": str(source),
        "PWD": str(clone),
    }
    if capability:
        env["R6_ISOLATED_CHILD_CAPABILITY"] = capability
    return env


def _official_chunk_isolated(source: Path, clone: Path) -> dict[str, Any]:
    """Run the official worker out-of-process and accept one sealed result."""

    source_before = source_snapshot(source)
    _assert_clone_contained(clone)
    clone_before = _clone_stat(clone)
    script = Path(__file__).resolve(strict=True)
    sandbox_before = _sandbox_identity()
    capability = secrets.token_urlsafe(32)
    command = [
        sys.executable, "-I", "-S", "-B", str(script), "--official-worker",
        "--source-root", str(source), "--clone-root", str(clone),
        "--child-capability", capability,
    ]
    try:
        completed = _run_external_bounded(
            command,
            cwd=clone,
            timeout=_PHASE_TIMEOUT_SECONDS,
            check=True,
            env=_isolated_worker_env(source=source, clone=clone, capability=capability),
            worker_roots=(source, clone),
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired, R6Error) as exc:
        raise R6Error("isolated official worker failed") from exc
    if source_snapshot(source) != source_before:
        raise R6Error("source changed during isolated official worker")
    if _clone_stat(clone) != clone_before:
        raise R6Error("clone root changed during isolated official worker")
    if _sandbox_identity() != sandbox_before:
        raise R6Error("worker sandbox executable changed during isolated official worker")
    _assert_clone_contained(clone)
    try:
        lines = [line for line in completed.stdout.splitlines() if line.strip()]
        if len(lines) != 1:
            raise ValueError("result count")
        value = json.loads(lines[0])
    except (ValueError, json.JSONDecodeError) as exc:
        raise R6Error("isolated official worker result is invalid") from exc
    if not isinstance(value, Mapping):
        raise R6Error("isolated official worker result is invalid")
    worker = dict(value)
    unsigned = {key: item for key, item in worker.items() if key != "seal_sha256"}
    if worker.get("seal_sha256") != _digest(unsigned):
        raise R6Error("isolated official worker result seal is invalid")
    worker.pop("seal_sha256")
    raw_containment = getattr(completed, "r6_containment", None)
    if not isinstance(raw_containment, Mapping):
        raise R6Error("isolated official worker containment receipt is absent")
    worker["r6_child_containment"] = {
        **dict(raw_containment),
        "sandbox": sandbox_before,
    }
    _assert_child_containment(worker["r6_child_containment"])
    return worker


def _official_chunk(*_args: Any, **_kwargs: Any) -> None:
    """Reject the historical in-process entry point unconditionally."""

    raise R6Error("official worker is available only in the isolated child")


def _child_official_chunk(module: Any, clone: Path, source: Path, capability: str) -> dict[str, Any]:
    """Private child entry reached only after the OS sandbox is active."""

    received = os.environ.get("R6_ISOLATED_CHILD_CAPABILITY", "")
    if not capability or not hmac.compare_digest(received, capability):
        raise R6Error("official worker child capability is absent")
    return _isolated_child_chunk(module, clone, source)


def _official_worker_main(*, source: Path, clone: Path, capability: str) -> int:
    """Child-only entry point; its process image is disposable evidence."""

    try:
        assert_root_matrix(clone, source, Path(tempfile.gettempdir()) / "r6-worker-output")
        _assert_no_overlap(source, clone)
        with _clone_runtime_context(clone, source):
            module, _runtime = _load_runtime(source, clone)
            result = _child_official_chunk(module, clone, source, capability)
        print(json.dumps(_seal(result), sort_keys=True))
        return 0
    except (R6Error, OSError, ValueError) as exc:
        print(f"r6 isolated worker failed: {str(exc).split(':', 1)[0]}", file=sys.stderr)
        return 2


def _candidate_pointer_id(module: Any, clone: Path) -> str:
    try:
        pointer = module.store.read_pointer(clone, "candidate")
    except module.store.DistillationStoreError:
        return ""
    policy_id = pointer.get("policy_id")
    if set(pointer) != {"schema", "namespace", "kind", "policy_id", "seal_sha256"} or pointer.get("kind") != "candidate-policy-pointer":
        raise R6Error("candidate pointer is malformed")
    return _artifact_id(policy_id)


def _worker_snapshot(clone: Path) -> dict[str, str]:
    """Record all durable artifacts that one official chunk may create or alter."""

    root = clone / "runtime" / "recall-distillation"
    relevant = (
        "candidate-policy.json",
        "candidate-ledger.jsonl",
        "label-ledger.jsonl",
        "rally-manifest.jsonl",
        "state.json",
        "policies",
        "locked-replays",
        "runs",
    )
    result: dict[str, str] = {}
    for name in relevant:
        path = root / name
        if not path.exists():
            result[name] = ""
            continue
        if path.is_symlink():
            raise R6Error("official worker artifact path is a symlink")
        result[name] = _tree_identity(path) if path.is_dir() else _stable_file_identity(path)
    return result


def _candidate_ledger_state(store: Any, clone: Path) -> dict[str, Any]:
    state = store.chain_head(store.distillation_dir(clone) / "candidate-ledger.jsonl")
    if set(state) != {"records", "head_sha256"} or not isinstance(state["records"], int):
        raise R6Error("candidate ledger state is malformed")
    head = state["head_sha256"]
    if head and _artifact_id(head) != head:
        raise R6Error("candidate ledger head is malformed")
    return {"records": state["records"], "head_sha256": head}


def _require_candidate_newness(
    before_snapshot: Mapping[str, str],
    after_snapshot: Mapping[str, str],
    before_candidate_ledger: Mapping[str, Any],
    after_candidate_ledger: Mapping[str, Any],
    lineage: Mapping[str, Any],
) -> None:
    """Require new candidate artifacts; a fully bound existing ledger may stay put."""

    if before_snapshot.get("candidate-policy.json"):
        raise R6Error("official candidate was already present; R6 generation is not initial")
    for name in ("candidate-policy.json", "state.json", "policies", "locked-replays", "runs"):
        if after_snapshot.get(name) == before_snapshot.get(name):
            raise R6Error("official worker did not create a complete new R6 candidate")
    if (
        before_snapshot.get("candidate-ledger.jsonl") == after_snapshot.get("candidate-ledger.jsonl")
        and (
            after_candidate_ledger != before_candidate_ledger
            or lineage.get("candidate_head") != before_candidate_ledger.get("head_sha256")
        )
    ):
        raise R6Error("unchanged candidate ledger is not bound to candidate lineage")


def _stable_state_digest(state: Mapping[str, Any]) -> str:
    """Compare deterministic worker state while preserving sealed volatile timestamps."""

    if "stage_started_at" in state:
        _strict_timestamp(state["stage_started_at"], "worker state stage_started_at")
    if "last_success_at" in state:
        _strict_timestamp(state["last_success_at"], "worker state last_success_at")
    if "seal_sha256" in state:
        _artifact_id(state["seal_sha256"])
    return _digest({
        key: value for key, value in state.items()
        if key not in _VOLATILE_STATE_FIELDS
    })


_RUN_KEYS = frozenset(
    {
        "schema",
        "namespace",
        "artifact_id",
        "seal_sha256",
        "kind",
        "raw_watermark",
        "baseline_artifact_id",
        "manifest_head",
        "candidate_head",
        "label_head",
        "processed",
        "candidate_snapshots",
        "labels_written",
        "ox_workset",
        "local_workset",
        "ox_profile_contract_id",
        "ox_profile_stopped",
        "counterfactuals_written",
        "p5_allowed",
    }
)
_OX_RAMP_KEYS = frozenset(
    {
        "ox_ramp_cap",
        "ox_ramp_valid_receipts",
        "ox_ramp_provider_attempts",
        "ox_ramp_request_revision",
    }
)
_STATE_KEYS = frozenset(
    {
        "schema",
        "namespace",
        "seal_sha256",
        "kind",
        "status",
        "worker_status",
        "rollout_percent",
        "raw_watermark",
        "baseline_artifact_id",
        "historical_index_sha256",
        "manifest_chain_head",
        "run_id",
        "processed",
        "candidate_snapshots",
        "labels_written",
        "ox_workset",
        "local_workset",
        "ox_profile_contract_id",
        "ox_profile_stopped",
        "counterfactuals_written",
        "teacher_model_calls",
        "counterfactual_model_calls",
        "cold_start_pending",
        "cold_start_lane_turn",
        "split_plan_id",
        "manifest_backlog",
        "candidate_backlog",
        "promotion_status",
        "promotion_reason",
        "incumbent_policy_id",
        "rollout_evaluation_status",
        "hold_reason",
        "capture_only_reasons",
        "last_success_at",
        "error_code",
    }
)
# Rollout may have already written these fields.  _persist_distillation_chunk
# deliberately carries them forward in transition["previous"].
_STATE_ROLLOUT_KEYS = frozenset(
    {
        "stage_started_at",
        "learning_halted",
        "candidate_policy_id",
        "lkg_policy_id",
        "last_run_id",
        "stage_run_id",
        "evaluation_receipt_id",
        "quarantine_id",
    }
)
_WORKER_KEYS = frozenset(
    {
        "status", "processed", "p5_allowed", "teachers_available",
        "counterfactual_available", "candidate_snapshots", "labels_written",
        "ox_workset", "local_workset", "ox_profile_contract_id",
        "ox_profile_stopped", "counterfactuals_written", "cold_start_pending",
        "split_plan_id", "manifest_backlog", "candidate_backlog", "promotion",
        "rollout_evaluation", "run_id", "state_sha256", "r6_egress_attempts",
        "r6_provider_attempts", "r6_git_sha256", "r6_child_containment",
    }
)
_CHILD_CONTAINMENT_KEYS = frozenset(
    {
        "schema", "registered_descendants", "rejected_registry_entries",
        "remaining_descendants", "registry_fd_closed", "sandbox",
    }
)
_SANDBOX_IDENTITY_KEYS = frozenset({"path", "dev", "ino", "uid", "mode", "sha256"})
_WORKSET_COUNT_KEYS = frozenset({"ready", "leased", "completed", "quarantined"})
_WORKSET_STATE_KEYS = _WORKSET_COUNT_KEYS | frozenset(
    {"backlog", "total", "last_durable_receipt", "last_durable_progress"}
)
_WORKSET_RECEIPT_KEYS = frozenset({"generation", "head_sha256"})
_PROGRESS_KEYS = frozenset({"cursor", "ledger_heads", "provenance", "progress_kind"})
_REPLAY_ROW_BASE_KEYS = frozenset(
    {
        "rally_id", "candidate_id", "session_cluster_id", "as_of", "dimension",
        "verdict", "authority", "features", "route", "route_identity",
        "teacher_role", "model_digest", "generator_model_digest",
        "judge_model_digest", "generator_route_identity", "judge_route_identity",
        "counterfactual_ref", "a0_sha256", "a1_sha256", "blind_orders",
        "counterfactual_producer", "counterfactual_revision", "probe", "source",
        "profile", "cohort", "assignment_revision", "assignment_authority",
        "profile_contract_id", "expires_at", "identity_revision", "request_revision",
        "group_id", "label_split_plan_id", "order_agreement", "label_record_sha256",
        "payload_digest", "payload_source", "work_id", "negative_veto_conflict",
        "feature_parity", "future_leakage", "split", "split_plan_id",
        "locked_test_read_only", "locked_test_evidence_ref",
    }
)
_REPLAY_ROW_OX_KEYS = _REPLAY_ROW_BASE_KEYS | frozenset(
    {
        "status", "error_class", "route_digest", "route_identity_exact",
        "prompt_sha256", "schema_sha256", "request_sha256", "provider_request_sha256",
        "provider_response_request_sha256", "group_identity_exact",
        "future_leakage_evidence_ref", "repeat_pair_id", "fixed_repeat", "fixed_split_plan",
        "order_swap",
        "blind_order",
    }
)


def _strict_int(value: object, field: str, *, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise R6Error(f"{field} type or bounds are invalid")
    if maximum is not None and value > maximum:
        raise R6Error(f"{field} type or bounds are invalid")
    return value


def _strict_bool(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise R6Error(f"{field} type is invalid")
    return value


def _strict_text(value: object, field: str, *, allow_empty: bool = True, limit: int = 512) -> str:
    if not isinstance(value, str) or len(value) > limit or (not allow_empty and not value):
        raise R6Error(f"{field} type or bounds are invalid")
    if any(ord(char) < 0x20 or ord(char) == 0x7F for char in value):
        raise R6Error(f"{field} contains control data")
    return value


def _strict_head(value: object, field: str) -> str:
    _strict_text(value, field)
    if value != "":
        try:
            _artifact_id(value)
        except R6Error as exc:
            raise R6Error(f"{field} identity is invalid") from exc
    return str(value)


def _strict_id_or_empty(value: object, field: str) -> str:
    _strict_text(value, field)
    if value:
        try:
            _artifact_id(value)
        except R6Error as exc:
            raise R6Error(f"{field} identity is invalid") from exc
    return str(value)


def _strict_string_list(value: object, field: str) -> list[str]:
    if not isinstance(value, list):
        raise R6Error(f"{field} type is invalid")
    if len(value) > 256:
        raise R6Error(f"{field} exceeds bounds")
    return [_strict_text(item, f"{field}[{index}]", allow_empty=False) for index, item in enumerate(value)]


def _strict_timestamp(value: object, field: str) -> str:
    text = _strict_text(value, field)
    if text:
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError as exc:
            raise R6Error(f"{field} timestamp is invalid") from exc
        if parsed.tzinfo is None:
            raise R6Error(f"{field} timestamp is not timezone-aware")
    return text


def _assert_progress_schema(
    value: object,
    *,
    flavor: str,
    expected_heads: Mapping[str, str] | None = None,
    expected_profile_contract: str | None = None,
) -> None:
    if value is None:
        return
    if not isinstance(value, Mapping) or set(value) != _PROGRESS_KEYS:
        raise R6Error(f"{flavor} workset progress schema is not closed")
    cursor = value["cursor"]
    heads = value["ledger_heads"]
    provenance = value["provenance"]
    if not isinstance(cursor, Mapping) or not isinstance(heads, Mapping) or not isinstance(provenance, Mapping):
        raise R6Error(f"{flavor} workset progress types are invalid")
    if flavor == "ox":
        if set(cursor) != {"candidate_count", "label_count", "revision_epoch"}:
            raise R6Error("OX workset progress cursor schema is not closed")
        for key in cursor:
            _strict_int(cursor[key], f"ox_workset.last_durable_progress.cursor.{key}")
        if set(heads) != {"candidate", "labels"}:
            raise R6Error("OX workset progress heads schema is not closed")
        if set(provenance) != {"profile", "profile_contract_id", "probe_revision", "split_plan_id"}:
            raise R6Error("OX workset progress provenance schema is not closed")
        if provenance.get("profile") != "ox-alpha-single-v1":
            raise R6Error("OX workset progress profile is invalid")
        if provenance.get("probe_revision") != "single-teacher-repeat-v2":
            raise R6Error("OX workset progress probe revision is invalid")
        _strict_id_or_empty(provenance.get("profile_contract_id"), "OX workset progress profile contract")
        _strict_id_or_empty(provenance.get("split_plan_id"), "OX workset progress split plan")
        if expected_profile_contract is not None and provenance.get("profile_contract_id") != expected_profile_contract:
            raise R6Error("OX workset progress profile contract mismatch")
        if value.get("progress_kind") != "ox-workset-v2":
            raise R6Error("OX workset progress kind is invalid")
    else:
        if set(cursor) != {"candidate_count", "label_count"}:
            raise R6Error("local workset progress cursor schema is not closed")
        for key in cursor:
            _strict_int(cursor[key], f"local_workset.last_durable_progress.cursor.{key}")
        if set(heads) != {"candidate", "labels"}:
            raise R6Error("local workset progress heads schema is not closed")
        if set(provenance) != {"assignment_revision", "probe_revision", "split_plan_id"}:
            raise R6Error("local workset progress provenance schema is not closed")
        if provenance.get("assignment_revision") != "assignment-v2" or provenance.get("probe_revision") != "probe-v2":
            raise R6Error("local workset progress provenance is invalid")
        _strict_id_or_empty(provenance.get("split_plan_id"), "local workset progress split plan")
        if value.get("progress_kind") != "local-workset-v2":
            raise R6Error("local workset progress kind is invalid")
    for key, head in heads.items():
        _strict_head(head, f"{flavor}_workset.last_durable_progress.ledger_heads.{key}")
    if expected_heads is not None and dict(heads) != expected_heads:
        raise R6Error(f"{flavor} workset progress ledger head mismatch")


def _assert_workset_schema(
    value: object,
    *,
    flavor: str,
    include_timing: bool,
    expected_heads: Mapping[str, str] | None = None,
    expected_profile_contract: str | None = None,
) -> None:
    if not isinstance(value, Mapping):
        raise R6Error(f"{flavor} workset is not an object")
    expected = set(_WORKSET_STATE_KEYS) | ({
        "retry_wait",
        "oldest_backlog_age_seconds",
        "oldest_ready_age_seconds",
        "oldest_retry_wait_age_seconds",
        "next_retry_in_seconds",
        "stages",
    } if include_timing else set())
    if set(value) != expected:
        raise R6Error(f"{flavor} workset schema is not closed")
    counts = {key: _strict_int(value[key], f"{flavor}_workset.{key}") for key in _WORKSET_COUNT_KEYS}
    if value["backlog"] != counts["ready"] + counts["leased"] or value["total"] != sum(counts.values()):
        raise R6Error(f"{flavor} workset count binding is invalid")
    receipt = value["last_durable_receipt"]
    if not isinstance(receipt, Mapping) or set(receipt) != _WORKSET_RECEIPT_KEYS:
        raise R6Error(f"{flavor} workset receipt schema is not closed")
    _strict_int(receipt["generation"], f"{flavor}_workset.last_durable_receipt.generation")
    _strict_head(receipt["head_sha256"], f"{flavor}_workset.last_durable_receipt.head_sha256")
    _assert_progress_schema(
        value["last_durable_progress"],
        flavor=flavor,
        expected_heads=expected_heads,
        expected_profile_contract=expected_profile_contract,
    )
    if include_timing:
        for key in (
            "retry_wait",
            "oldest_backlog_age_seconds",
            "oldest_ready_age_seconds",
            "oldest_retry_wait_age_seconds",
            "next_retry_in_seconds",
        ):
            _strict_int(value[key], f"{flavor}_workset.{key}")
        stages = value["stages"]
        stage_names = {"snapshot", "teacher", "counterfactual", "dataset", "evaluation", "retry_wait"}
        stage_states = set(_WORKSET_COUNT_KEYS) | {"retry_wait", "backlog"}
        if not isinstance(stages, Mapping) or set(stages) != stage_names:
            raise R6Error(f"{flavor} workset stages schema is not closed")
        for stage, stage_value in stages.items():
            if not isinstance(stage_value, Mapping) or set(stage_value) != stage_states:
                raise R6Error(f"{flavor} workset stage schema is not closed")
            for key in stage_states:
                _strict_int(stage_value[key], f"{flavor}_workset.stages.{stage}.{key}")


def _assert_replay_rows_schema(
    rows: object,
    *,
    lineage: Mapping[str, Any],
) -> None:
    if not isinstance(rows, list) or not rows:
        raise R6Error("locked replay rows are invalid")
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or set(row) not in {_REPLAY_ROW_BASE_KEYS, _REPLAY_ROW_OX_KEYS}:
            raise R6Error(f"locked replay row {index} schema is not closed")
        prefix = f"locked replay row {index}"
        for field in (
            "rally_id", "candidate_id", "session_cluster_id", "as_of", "dimension",
            "verdict", "authority", "route", "teacher_role", "model_digest",
            "generator_model_digest", "judge_model_digest", "counterfactual_producer",
            "counterfactual_revision", "source", "profile", "cohort",
            "assignment_revision", "assignment_authority", "profile_contract_id",
            "expires_at", "identity_revision", "request_revision", "group_id",
            "label_split_plan_id", "label_record_sha256", "payload_digest", "work_id",
            "split", "split_plan_id", "locked_test_evidence_ref",
        ):
            _strict_text(row[field], f"{prefix}.{field}")
        _strict_timestamp(row["as_of"], f"{prefix}.as_of")
        if row["expires_at"]:
            _strict_timestamp(row["expires_at"], f"{prefix}.expires_at")
        if row["source"] not in {"teacher-label", "counterfactual-label"}:
            raise R6Error(f"{prefix}.source is invalid")
        if row["split"] not in {"train", "validation", "test", "embargo"}:
            raise R6Error(f"{prefix}.split is invalid")
        for field in (
            "rally_id", "candidate_id", "session_cluster_id", "model_digest",
            "generator_model_digest", "judge_model_digest", "counterfactual_ref",
            "a0_sha256", "a1_sha256", "label_record_sha256", "payload_digest",
            "split_plan_id",
        ):
            value = row[field]
            if value and (not isinstance(value, str) or len(value) != 64 or set(value) - _SHA):
                raise R6Error(f"{prefix}.{field} identity is invalid")
        if row["split_plan_id"] != lineage["split_plan_id"]:
            raise R6Error(f"{prefix}.split_plan_id is not bound to lineage")
        if row["profile_contract_id"] != lineage["profile_contract_id"]:
            raise R6Error(f"{prefix}.profile_contract_id is not bound to lineage")
        if lineage["profile_contract_id"]:
            if row["profile"] != "ox-alpha-single-v1" or row["cohort"] != "ox-alpha-single-v1":
                raise R6Error(f"{prefix}.profile/cohort is not bound to the R5 OX profile")
            if row["route"] != "opencode-go/ox-alpha-free":
                raise R6Error(f"{prefix}.route is not bound to the R5 OX profile")
        elif row["profile"] != "local-triad-v1" or row["cohort"] != "local-triad-v1":
            raise R6Error(f"{prefix}.profile/cohort is not bound to the R5 local profile")
        if not isinstance(row["features"], Mapping) or set(row["features"]) != {
            "query_chargram_coverage", "candidate_chargram_precision",
        }:
            raise R6Error(f"{prefix}.features schema is invalid")
        for field in row["features"]:
            feature = row["features"][field]
            if isinstance(feature, bool) or not isinstance(feature, (int, float)) or not 0 <= feature <= 1:
                raise R6Error(f"{prefix}.features type is invalid")
        for field in ("route_identity", "generator_route_identity", "judge_route_identity", "payload_source"):
            if not isinstance(row[field], Mapping):
                raise R6Error(f"{prefix}.{field} type is invalid")
        for field in ("route_identity", "generator_route_identity", "judge_route_identity"):
            identity = row[field]
            if set(identity) not in (set(), {"provider", "model", "location"}):
                raise R6Error(f"{prefix}.{field} schema is not closed")
            for identity_field in identity:
                _strict_text(identity[identity_field], f"{prefix}.{field}.{identity_field}", allow_empty=False)
        payload_source = row["payload_source"]
        payload_keys = {
            "rally_id", "candidate_id", "snapshot_sha256", "query_sha256",
            "candidate_text_sha256", "context_sha256",
        }
        if set(payload_source) not in (set(), payload_keys, payload_keys | {"assignment"}):
            raise R6Error(f"{prefix}.payload_source schema is not closed")
        _strict_string_list(row["blind_orders"], f"{prefix}.blind_orders")
        for field in (
            "probe", "order_agreement", "negative_veto_conflict", "feature_parity",
            "future_leakage", "locked_test_read_only",
        ):
            _strict_bool(row[field], f"{prefix}.{field}")
        if row["negative_veto_conflict"] or row["future_leakage"]:
            raise R6Error(f"{prefix} contains a forbidden leakage/veto conflict")
        if row["source"] == "counterfactual-label":
            if (
                not row["counterfactual_ref"]
                or not row["a0_sha256"]
                or not row["a1_sha256"]
                or row["counterfactual_producer"] != "chronovisor-local-blind-v1"
                or row["counterfactual_revision"] != "two-order-locked-v1"
                or row["order_agreement"] is not True
                or set(row["blind_orders"]) != {"a0_first", "a1_first"}
                or len(row["blind_orders"]) != 2
                or not row["generator_route_identity"]
                or not row["judge_route_identity"]
                or row["generator_route_identity"] == row["judge_route_identity"]
            ):
                raise R6Error(f"{prefix} counterfactual binding is invalid")
            for identity_name in ("generator_route_identity", "judge_route_identity"):
                identity = row[identity_name]
                if (
                    set(identity) != {"provider", "model", "location"}
                    or identity["location"] != "local"
                    or not identity["provider"]
                    or not identity["model"]
                ):
                    raise R6Error(f"{prefix}.{identity_name} is invalid")
        else:
            if (
                row["counterfactual_ref"]
                or row["a0_sha256"]
                or row["a1_sha256"]
                or row["counterfactual_producer"]
                or row["counterfactual_revision"]
                or row["blind_orders"]
            ):
                raise R6Error(f"{prefix} teacher row carries counterfactual fields")
            route_identity = row["route_identity"]
            if lineage["profile_contract_id"]:
                if route_identity != {
                    "provider": "opencode-go",
                    "model": "opencode-go/ox-alpha-free",
                    "location": "remote",
                }:
                    raise R6Error(f"{prefix} OX route identity is invalid")
            elif route_identity.get("location") != "local":
                raise R6Error(f"{prefix} local route identity is invalid")
        if set(row) == _REPLAY_ROW_OX_KEYS:
            for field in ("route_identity_exact", "group_identity_exact", "fixed_repeat", "fixed_split_plan", "order_swap"):
                _strict_bool(row[field], f"{prefix}.{field}")
            if row["error_class"] is not None:
                _strict_text(row["error_class"], f"{prefix}.error_class")


def _assert_source_snapshot_schema(value: object) -> None:
    expected = {"head", "status", "status_count", "tree_sha256", "git_executable_sha256", "module_sha256", "git_layout"}
    if not isinstance(value, Mapping) or set(value) != expected:
        raise R6Error("source binding schema is not closed")
    head = value["head"]
    if not isinstance(head, str) or len(head) != 40 or set(head) - _SHA:
        raise R6Error("source binding head is invalid")
    for field in ("status", "tree_sha256", "git_executable_sha256"):
        _artifact_id(value[field])
    _strict_int(value["status_count"], "source status_count")
    if value["status_count"] != 0:
        raise R6Error("source status is not clean")
    _strict_id_or_empty(value["module_sha256"], "source runtime module digest")
    layout = value["git_layout"]
    if not isinstance(layout, Mapping) or set(layout) != _GIT_LAYOUT_KEYS:
        raise R6Error("source git layout schema is not closed")
    for field in ("entry", "git_dir", "work_tree", "index"):
        path = layout[field]
        if not isinstance(path, str) or not Path(path).is_absolute():
            raise R6Error("source git layout path is invalid")
    _artifact_id(layout["entry_sha256"])
    if layout["index_sha256"]:
        _artifact_id(layout["index_sha256"])
    for field in ("entry_bytes", "entry_inode", "entry_mode", "git_dir_inode", "git_dir_mode", "index_bytes", "index_inode", "index_mode"):
        _strict_int(layout[field], f"source git layout {field}")


def _assert_runtime_schema(value: object, *, source: Mapping[str, Any] | None = None) -> None:
    expected = {"module_path", "module_sha256", "runtime_root", "config_path", "repository_root"}
    if not isinstance(value, Mapping) or set(value) != expected:
        raise R6Error("runtime binding schema is not closed")
    if not isinstance(value["module_path"], str) or not value["module_path"] or Path(value["module_path"]).is_absolute():
        raise R6Error("runtime module path is invalid")
    _artifact_id(value["module_sha256"])
    for field in ("runtime_root", "config_path", "repository_root"):
        if not isinstance(value[field], str) or not Path(value[field]).is_absolute():
            raise R6Error(f"runtime {field} is invalid")
    if source is not None and value["repository_root"] != source["git_layout"]["work_tree"]:
        raise R6Error("runtime repository root is not source-bound")
    if source is not None:
        expected_module = "src/chronovisor/recall/recall_distillation.py"
        if value["module_path"] != expected_module or value["module_sha256"] != source["module_sha256"]:
            raise R6Error("runtime module is not source-bound")


def _assert_r5_schema(value: object) -> None:
    expected = {"passed", "reason", "baseline_id", "raw_watermark", "label_head", "cohort_sha256", "split_plan_id", "profile_contract_id"}
    if not isinstance(value, Mapping) or set(value) != expected:
        raise R6Error("R5 binding schema is not closed")
    _strict_bool(value["passed"], "R5 passed")
    _strict_text(value["reason"], "R5 reason")
    for field in ("baseline_id", "raw_watermark", "label_head", "cohort_sha256", "split_plan_id"):
        _artifact_id(value[field])
    _strict_id_or_empty(value["profile_contract_id"], "R5 profile contract")


def _assert_output_artifact_envelopes(
    value: Mapping[str, Any],
    *,
    r5: Mapping[str, Any],
    heads: Mapping[str, str],
) -> None:
    pointer = value["pointer"]
    policy = value["policy"]
    replay = value["locked_replay"]
    run = value["run"]
    state = value["state"]
    if not all(isinstance(item, Mapping) for item in (pointer, policy, replay, run, state)):
        raise R6Error("formal candidate artifact is not an object")
    if set(pointer) != {"schema", "namespace", "kind", "policy_id", "seal_sha256"}:
        raise R6Error("formal candidate pointer schema is not closed")
    policy_keys = {
        "schema", "namespace", "artifact_id", "seal_sha256", "kind", "lineage",
        "feature_keys", "feature_revision", "weights", "bias", "threshold",
        "abstain_margin", "max_cards", "training_rows", "validation_rows",
    }
    if set(policy) != policy_keys:
        raise R6Error("formal candidate policy schema is not closed")
    replay_keys = {
        "schema", "namespace", "artifact_id", "seal_sha256", "kind", "training_snapshot_id",
        "training_rows", "baseline_artifact_id", "policy_sha256", "training_rows_sha256",
        "candidate_head", "profile_contract_id", "offline_gate_sha256", "model_cohort_sha256",
        "split_revision",
    }
    if set(replay) != replay_keys:
        raise R6Error("formal locked replay schema is not closed")
    if set(run) - (_RUN_KEYS | _OX_RAMP_KEYS) or not _RUN_KEYS.issubset(run):
        raise R6Error("formal run schema is not closed")
    if set(state) - (_STATE_KEYS | _STATE_ROLLOUT_KEYS | _OX_RAMP_KEYS) or not _STATE_KEYS.issubset(state):
        raise R6Error("formal state schema is not closed")
    for owner_name, owner in (("run", run), ("state", state)):
        ramp = set(owner).intersection(_OX_RAMP_KEYS)
        if ramp and ramp != set(_OX_RAMP_KEYS):
            raise R6Error(f"formal {owner_name} OX ramp schema is not closed")
    _assert_artifact_integrity(pointer, "formal candidate pointer", has_id=False)
    _assert_artifact_integrity(policy, "formal candidate policy", has_id=True)
    _assert_artifact_integrity(replay, "formal locked replay", has_id=True)
    _assert_artifact_integrity(run, "formal run", has_id=True)
    _assert_artifact_integrity(state, "formal worker state", has_id=False)
    _artifact_id(pointer["policy_id"])
    _artifact_id(policy["artifact_id"])
    _artifact_id(replay["artifact_id"])
    _artifact_id(run["artifact_id"])
    _assert_run_schema(run, r5=r5, heads=heads)
    _assert_worker_state_schema(None, state, run=run, r5=r5, heads=heads)
    for field in (
        "training_snapshot_id", "baseline_artifact_id", "policy_sha256",
        "training_rows_sha256", "candidate_head", "offline_gate_sha256", "model_cohort_sha256",
    ):
        _artifact_id(replay[field])
    _strict_id_or_empty(replay["profile_contract_id"], "formal replay profile contract")
    if replay["split_revision"] != "grouped-rolling-v1":
        raise R6Error("formal replay split revision is invalid")
    if policy["feature_keys"] != ["query_chargram_coverage", "candidate_chargram_precision"]:
        raise R6Error("formal policy feature keys are invalid")
    if policy.get("schema") != "chronovisor.recall-distill-policy.v2" or policy.get("namespace") != "recall-distillation" or policy.get("kind") != "tiny-logistic-policy":
        raise R6Error("formal policy identity is invalid")
    if policy.get("feature_revision") != "recall-distill-text-v2":
        raise R6Error("formal policy feature revision is invalid")
    if not isinstance(policy["weights"], Mapping) or set(policy["weights"]) != set(policy["feature_keys"]):
        raise R6Error("formal policy weights schema is invalid")
    for field in ("bias", "threshold", "abstain_margin"):
        if isinstance(policy[field], bool) or not isinstance(policy[field], (int, float)) or not math.isfinite(policy[field]):
            raise R6Error(f"formal policy {field} type is invalid")
    if not 0 <= policy["threshold"] <= 1 or not 0 <= policy["abstain_margin"] <= 1:
        raise R6Error("formal policy threshold bounds are invalid")
    for field in policy["weights"]:
        if isinstance(policy["weights"][field], bool) or not isinstance(policy["weights"][field], (int, float)) or not math.isfinite(policy["weights"][field]):
            raise R6Error(f"formal policy weight {field} type is invalid")
    for field in ("max_cards", "training_rows", "validation_rows"):
        _strict_int(policy[field], f"formal policy {field}")
    if policy["training_rows"] <= 0 or policy["validation_rows"] <= 0:
        raise R6Error("formal policy has no training or validation rows")
    if pointer["policy_id"] != policy["artifact_id"] or value["candidate_id"] != pointer["policy_id"]:
        raise R6Error("formal candidate pointer identity mismatch")
    lineage = policy["lineage"]
    if not isinstance(lineage, Mapping) or set(lineage) != {
        "training_snapshot_id", "locked_replay_id", "baseline_artifact_id", "model_cohort_sha256",
        "raw_watermark", "label_chain_head", "feature_revision", "split_plan_id",
        "offline_gate_sha256", "training_rows_sha256", "candidate_head", "profile_contract_id",
    }:
        raise R6Error("formal policy lineage schema is not closed")
    for field in (
        "training_snapshot_id", "locked_replay_id", "baseline_artifact_id", "model_cohort_sha256",
        "raw_watermark", "label_chain_head", "split_plan_id", "offline_gate_sha256",
        "training_rows_sha256", "candidate_head",
    ):
        _artifact_id(lineage[field])
    if lineage["feature_revision"] != "recall-distill-text-v2":
        raise R6Error("formal policy lineage feature revision is invalid")
    _strict_id_or_empty(lineage["profile_contract_id"], "formal policy lineage profile contract")
    _assert_replay_rows_schema(replay["training_rows"], lineage=lineage)
    if replay["training_rows_sha256"] != _digest(replay["training_rows"]):
        raise R6Error("formal replay training rows digest is invalid")
    if replay["policy_sha256"] != _policy_payload_digest(policy):
        raise R6Error("formal replay policy digest is invalid")
    if lineage["training_rows_sha256"] != replay["training_rows_sha256"]:
        raise R6Error("formal policy training rows digest is not bound")
    if (
        r5.get("passed") is not True
        or r5.get("cohort_sha256") != replay["model_cohort_sha256"]
        or r5.get("cohort_sha256") != lineage["model_cohort_sha256"]
        or r5.get("split_plan_id") != lineage["split_plan_id"]
        or r5.get("split_plan_id") != replay.get("split_plan_id", lineage["split_plan_id"])
        or r5.get("profile_contract_id") != replay["profile_contract_id"]
        or r5.get("profile_contract_id") != lineage["profile_contract_id"]
        or r5.get("label_head") != lineage["label_chain_head"]
    ):
        raise R6Error("formal R5 lineage binding is invalid")
    if (
        lineage["locked_replay_id"] != replay["artifact_id"]
        or replay["baseline_artifact_id"] != lineage["baseline_artifact_id"]
        or replay["candidate_head"] != lineage["candidate_head"]
        or replay["profile_contract_id"] != lineage["profile_contract_id"]
        or run["artifact_id"] != state["run_id"]
        or run["raw_watermark"] != r5["raw_watermark"]
        or run["baseline_artifact_id"] != r5["baseline_id"]
        or run["manifest_head"] != heads["manifest"]
        or run["candidate_head"] != heads["candidate"]
        or run["label_head"] != heads["label"]
        or state["raw_watermark"] != r5["raw_watermark"]
        or state["baseline_artifact_id"] != r5["baseline_id"]
        or state["manifest_chain_head"] != heads["manifest"]
    ):
        raise R6Error("formal candidate artifact binding is invalid")
    for field in ("raw_watermark", "baseline_artifact_id"):
        _artifact_id(run[field])
    for field in ("manifest_head", "candidate_head", "label_head"):
        _strict_head(run[field], f"formal run {field}")
    _artifact_id(state["raw_watermark"])
    _artifact_id(state["baseline_artifact_id"])
    _strict_head(state["manifest_chain_head"], "formal state manifest head")
    for owner_name, owner in (("run", run), ("state", state)):
        if owner["ox_workset"] == {} and owner["local_workset"] == {}:
            continue
        if owner["ox_workset"] != {} and owner["local_workset"] == {}:
            _assert_workset_schema(
                owner["ox_workset"],
                flavor="ox",
                include_timing=False,
                expected_heads={"candidate": run["candidate_head"], "labels": run["label_head"]},
                expected_profile_contract=owner["ox_profile_contract_id"],
            )
        elif owner["local_workset"] != {} and owner["ox_workset"] == {}:
            _assert_workset_schema(
                owner["local_workset"],
                flavor="local",
                include_timing=True,
                expected_heads={"candidate": run["candidate_head"], "labels": run["label_head"]},
            )
        else:
            raise R6Error(f"formal {owner_name} workset selection is invalid")


def _assert_run_schema(run: Mapping[str, Any], *, r5: Mapping[str, Any], heads: Mapping[str, str]) -> None:
    """Validate the bounded-chunk run envelope independently on readback."""

    present_ramp = set(run).intersection(_OX_RAMP_KEYS)
    if set(run) - (_RUN_KEYS | _OX_RAMP_KEYS) or not _RUN_KEYS.issubset(run):
        raise R6Error("official run schema is not closed")
    if present_ramp and present_ramp != set(_OX_RAMP_KEYS):
        raise R6Error("official run OX ramp schema is not closed")
    if run.get("schema") != "chronovisor.recall-distill-run.v1" or run.get("namespace") != "recall-distillation" or run.get("kind") != "bounded-chunk":
        raise R6Error("official run identity is invalid")
    _artifact_id(run["artifact_id"])
    for field in ("raw_watermark", "baseline_artifact_id"):
        _artifact_id(run[field])
    for field in ("manifest_head", "candidate_head", "label_head"):
        _strict_head(run[field], f"official run {field}")
    for field in ("processed", "candidate_snapshots", "labels_written", "counterfactuals_written"):
        _strict_int(run[field], f"official run {field}")
    if run["processed"] <= 0 or run["candidate_snapshots"] <= 0 or run["labels_written"] <= 0:
        raise R6Error("official run has no durable training progress")
    _strict_bool(run["p5_allowed"], "official run p5_allowed")
    _strict_bool(run["ox_profile_stopped"], "official run ox_profile_stopped")
    _strict_id_or_empty(run["ox_profile_contract_id"], "official run OX profile contract")
    if present_ramp:
        if run["ox_ramp_cap"] not in {1, 2, 5, 10}:
            raise R6Error("official run OX ramp cap is invalid")
        _strict_int(run["ox_ramp_cap"], "official run ox_ramp_cap")
        _strict_int(run["ox_ramp_valid_receipts"], "official run ox_ramp_valid_receipts")
        _strict_int(run["ox_ramp_provider_attempts"], "official run ox_ramp_provider_attempts")
        if run["ox_ramp_request_revision"] != "json-schema-core-label-abstain-16k-240s-v6":
            raise R6Error("official run OX ramp revision is invalid")
    if (
        run["raw_watermark"] != r5["raw_watermark"]
        or run["baseline_artifact_id"] != r5["baseline_id"]
        or run["label_head"] != r5["label_head"]
        or run["candidate_head"] != heads["candidate"]
        or run["label_head"] != heads["label"]
        or run["manifest_head"] != heads["manifest"]
    ):
        raise R6Error("official run lineage binding is invalid")
    _assert_artifact_integrity(run, "official run", has_id=True)


def _assert_worker_state_schema(module: Any | None, state: Mapping[str, Any], *, run: Mapping[str, Any], r5: Mapping[str, Any], heads: Mapping[str, str]) -> None:
    """Validate the exact state emitted by _persist_distillation_chunk."""

    if (
        not isinstance(r5, Mapping)
        or not all(key in r5 for key in ("baseline_id", "raw_watermark", "label_head"))
        or not isinstance(heads, Mapping)
        or set(heads) != {"candidate", "label", "manifest"}
    ):
        raise R6Error("worker state binding inputs are malformed")
    if set(state) - (_STATE_KEYS | _STATE_ROLLOUT_KEYS | _OX_RAMP_KEYS) or not _STATE_KEYS.issubset(state):
        raise R6Error("official worker state schema is not closed")
    state_schema = getattr(getattr(module, "store", None), "DISTILLATION_SCHEMA", "chronovisor.recall-distillation.v1")
    if state.get("schema") != state_schema or state.get("namespace") != "recall-distillation" or state.get("kind") != "worker-state":
        raise R6Error("official worker state identity is invalid")
    if state.get("status") not in {"ready", "capture_only", "shadow", "replay", "canary", "active", "rolled_back", "quarantined", "adopting"}:
        raise R6Error("official worker state status is invalid")
    if state.get("worker_status") not in {"deferred", "ready", "capture_only"}:
        raise R6Error("official worker state worker status is invalid")
    _strict_int(state["rollout_percent"], "worker state rollout_percent", maximum=100)
    for field in ("raw_watermark", "baseline_artifact_id", "historical_index_sha256", "run_id", "incumbent_policy_id"):
        _artifact_id(state[field])
    _strict_head(state["manifest_chain_head"], "worker state manifest_chain_head")
    _strict_int(state["processed"], "worker state processed")
    _strict_int(state["candidate_snapshots"], "worker state candidate_snapshots")
    _strict_int(state["labels_written"], "worker state labels_written")
    _strict_int(state["counterfactuals_written"], "worker state counterfactuals_written")
    _strict_int(state["teacher_model_calls"], "worker state teacher_model_calls")
    _strict_int(state["counterfactual_model_calls"], "worker state counterfactual_model_calls")
    _strict_int(state["cold_start_lane_turn"], "worker state cold_start_lane_turn")
    _strict_int(state["manifest_backlog"], "worker state manifest_backlog")
    _strict_int(state["candidate_backlog"], "worker state candidate_backlog")
    for field in ("ox_profile_stopped", "cold_start_pending", "learning_halted"):
        if field in state:
            _strict_bool(state[field], f"worker state {field}")
    _strict_id_or_empty(state["ox_profile_contract_id"], "worker state OX profile contract")
    _strict_id_or_empty(state["split_plan_id"], "worker state split plan")
    _strict_text(state["promotion_status"], "worker state promotion_status", allow_empty=False)
    _strict_text(state["promotion_reason"], "worker state promotion_reason")
    _strict_text(state["rollout_evaluation_status"], "worker state rollout_evaluation_status", allow_empty=False)
    _strict_text(state["hold_reason"], "worker state hold_reason")
    _strict_string_list(state["capture_only_reasons"], "worker state capture_only_reasons")
    _strict_timestamp(state["last_success_at"], "worker state last_success_at")
    _strict_text(state["error_code"], "worker state error_code")
    for field in ("stage_started_at",):
        if field in state:
            _strict_timestamp(state[field], f"worker state {field}")
    for field in ("candidate_policy_id", "lkg_policy_id", "last_run_id", "stage_run_id", "evaluation_receipt_id", "quarantine_id"):
        if field in state:
            _artifact_id(state[field])
    present_ramp = set(state).intersection(_OX_RAMP_KEYS)
    if present_ramp and present_ramp != set(_OX_RAMP_KEYS):
        raise R6Error("worker state OX ramp schema is not closed")
    if present_ramp:
        if state["ox_ramp_cap"] not in {1, 2, 5, 10}:
            raise R6Error("worker state OX ramp cap is invalid")
        _strict_int(state["ox_ramp_cap"], "worker state ox_ramp_cap")
        _strict_int(state["ox_ramp_valid_receipts"], "worker state ox_ramp_valid_receipts")
        _strict_int(state["ox_ramp_provider_attempts"], "worker state ox_ramp_provider_attempts")
        expected_revision = getattr(module, "OX_RAMP_REQUEST_REVISION", "json-schema-core-label-abstain-16k-240s-v6")
        if state["ox_ramp_request_revision"] != expected_revision:
            raise R6Error("worker state OX ramp request revision is invalid")
    if state["ox_workset"] == {} and state["local_workset"] == {}:
        pass
    elif state["ox_workset"] != {} and state["local_workset"] == {}:
        _assert_workset_schema(
            state["ox_workset"],
            flavor="ox",
            include_timing=False,
            expected_heads={"candidate": run["candidate_head"], "labels": run["label_head"]},
            expected_profile_contract=state["ox_profile_contract_id"],
        )
    elif state["local_workset"] != {} and state["ox_workset"] == {}:
        _assert_workset_schema(
            state["local_workset"],
            flavor="local",
            include_timing=True,
            expected_heads={"candidate": run["candidate_head"], "labels": run["label_head"]},
        )
    else:
        raise R6Error("worker state workset selection is invalid")
    if state["run_id"] != run["artifact_id"] or state["raw_watermark"] != run["raw_watermark"] or state["baseline_artifact_id"] != run["baseline_artifact_id"] or state["manifest_chain_head"] != run["manifest_head"] or state["raw_watermark"] != r5["raw_watermark"] or state["baseline_artifact_id"] != r5["baseline_id"] or state["manifest_chain_head"] != heads["manifest"]:
        raise R6Error("official worker state is not transitively bound")
    for field in ("processed", "candidate_snapshots", "labels_written", "counterfactuals_written"):
        if state[field] != run[field]:
            raise R6Error(f"official worker state {field} is not bound to run")
    _assert_artifact_integrity(state, "official worker state", has_id=False)


def _assert_candidate_artifact_schemas(
    module: Any,
    pointer: Mapping[str, Any],
    policy: Mapping[str, Any],
    replay: Mapping[str, Any],
    run: Mapping[str, Any],
    *,
    state: Mapping[str, Any] | None = None,
    r5: Mapping[str, Any] | None = None,
    heads: Mapping[str, str] | None = None,
) -> None:
    """Close immutable worker artifacts to their official v1/v2 field contracts."""

    policy_keys = {
        "schema", "namespace", "artifact_id", "seal_sha256", "kind", "lineage",
        *module.train_tiny_policy(()).keys(),
    }
    replay_keys = {
        "schema", "namespace", "artifact_id", "seal_sha256", "kind", "training_snapshot_id",
        "training_rows", "baseline_artifact_id", "policy_sha256", "training_rows_sha256",
        "candidate_head", "profile_contract_id", "offline_gate_sha256", "model_cohort_sha256",
        "split_revision",
    }
    lineage_keys = {
        "training_snapshot_id", "locked_replay_id", "baseline_artifact_id",
        "model_cohort_sha256", "raw_watermark", "label_chain_head",
        "feature_revision", "split_plan_id", "offline_gate_sha256",
        "training_rows_sha256", "candidate_head", "profile_contract_id",
    }
    if (
        set(pointer) != {"schema", "namespace", "kind", "policy_id", "seal_sha256"}
        or set(policy) != policy_keys
        or not isinstance(policy.get("lineage"), Mapping)
        or set(policy["lineage"]) != lineage_keys
        or set(replay) != replay_keys
        or set(run) - (_RUN_KEYS | _OX_RAMP_KEYS)
        or not _RUN_KEYS.issubset(run)
    ):
        raise R6Error("official candidate artifact schema is not closed")
    if run.get("schema") != "chronovisor.recall-distill-run.v1" or run.get("namespace") != "recall-distillation" or run.get("kind") != "bounded-chunk":
        raise R6Error("official run identity is invalid")
    _artifact_id(run["artifact_id"])
    for field in ("raw_watermark", "baseline_artifact_id"):
        _artifact_id(run[field])
    for field in ("manifest_head", "candidate_head", "label_head"):
        _strict_head(run[field], f"run {field}")
    for field in ("processed", "candidate_snapshots", "labels_written", "counterfactuals_written"):
        _strict_int(run[field], f"run {field}")
    _strict_bool(run["ox_profile_stopped"], "run ox_profile_stopped")
    _strict_bool(run["p5_allowed"], "run p5_allowed")
    if run["p5_allowed"] is not True:
        raise R6Error("run p5 gate is not admitted")
    _strict_id_or_empty(run["ox_profile_contract_id"], "run OX profile contract")
    present_ramp = set(run).intersection(_OX_RAMP_KEYS)
    if present_ramp and present_ramp != set(_OX_RAMP_KEYS):
        raise R6Error("run OX ramp schema is not closed")
    if present_ramp:
        if run["ox_ramp_cap"] not in {1, 2, 5, 10}:
            raise R6Error("run OX ramp cap is invalid")
        _strict_int(run["ox_ramp_cap"], "run ox_ramp_cap")
        _strict_int(run["ox_ramp_valid_receipts"], "run ox_ramp_valid_receipts")
        _strict_int(run["ox_ramp_provider_attempts"], "run ox_ramp_provider_attempts")
        expected_revision = getattr(module, "OX_RAMP_REQUEST_REVISION", "json-schema-core-label-abstain-16k-240s-v6")
        if run["ox_ramp_request_revision"] != expected_revision:
            raise R6Error("run OX ramp request revision is invalid")
    if run["ox_workset"] == {} and run["local_workset"] == {}:
        pass
    elif run["ox_workset"] != {} and run["local_workset"] == {}:
        _assert_workset_schema(
            run["ox_workset"],
            flavor="ox",
            include_timing=False,
            expected_heads={"candidate": run["candidate_head"], "labels": run["label_head"]},
            expected_profile_contract=run["ox_profile_contract_id"],
        )
    elif run["local_workset"] != {} and run["ox_workset"] == {}:
        _assert_workset_schema(
            run["local_workset"],
            flavor="local",
            include_timing=True,
            expected_heads={"candidate": run["candidate_head"], "labels": run["label_head"]},
        )
    else:
        raise R6Error("run workset selection is invalid")
    for field in lineage_keys - {"feature_revision", "profile_contract_id"}:
        _artifact_id(policy["lineage"][field])
    feature_revision = getattr(module, "TEXT_FEATURE_REVISION", "recall-distill-text-v2")
    if policy["lineage"]["feature_revision"] != feature_revision:
        raise R6Error("policy lineage feature revision is invalid")
    _strict_id_or_empty(policy["lineage"]["profile_contract_id"], "policy lineage profile contract")
    policy_schema = getattr(module, "POLICY_SCHEMA", "chronovisor.recall-distill-policy.v2")
    if policy.get("schema") != policy_schema or policy.get("namespace") != "recall-distillation" or policy.get("kind") != "tiny-logistic-policy":
        raise R6Error("candidate policy identity is invalid")
    if policy.get("feature_keys") != ["query_chargram_coverage", "candidate_chargram_precision"]:
        raise R6Error("candidate policy feature keys are invalid")
    if policy.get("feature_revision") != getattr(module, "TEXT_FEATURE_REVISION", "recall-distill-text-v2"):
        raise R6Error("candidate policy feature revision is invalid")
    if not isinstance(policy.get("weights"), Mapping) or set(policy["weights"]) != set(policy["feature_keys"]):
        raise R6Error("candidate policy weights schema is invalid")
    for field in ("bias", "threshold", "abstain_margin"):
        number = policy.get(field)
        if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number):
            raise R6Error(f"candidate policy {field} type is invalid")
    if not 0 <= policy["threshold"] <= 1 or not 0 <= policy["abstain_margin"] <= 1:
        raise R6Error("candidate policy threshold bounds are invalid")
    for field in ("max_cards", "training_rows", "validation_rows"):
        _strict_int(policy.get(field), f"candidate policy {field}")
    if policy["training_rows"] <= 0 or policy["validation_rows"] <= 0:
        raise R6Error("candidate policy has no training or validation rows")
    _artifact_id(policy["artifact_id"])
    _artifact_id(pointer["policy_id"])
    _artifact_id(replay["artifact_id"])
    state_schema = getattr(getattr(module, "store", None), "DISTILLATION_SCHEMA", "chronovisor.recall-distillation.v1")
    if pointer.get("schema") != state_schema or pointer.get("namespace") != "recall-distillation" or pointer.get("kind") != "candidate-policy-pointer":
        raise R6Error("candidate pointer identity is invalid")
    if policy["artifact_id"] != pointer["policy_id"]:
        raise R6Error("candidate pointer policy identity mismatch")
    for field in (
        "training_snapshot_id", "baseline_artifact_id", "policy_sha256", "training_rows_sha256", "candidate_head", "offline_gate_sha256", "model_cohort_sha256",
    ):
        _artifact_id(replay[field])
    _strict_id_or_empty(replay["profile_contract_id"], "locked replay profile contract")
    if replay.get("schema") != "chronovisor.recall-distill-locked-replay.v1" or replay.get("namespace") != "recall-distillation" or replay.get("kind") != "locked-replay-input":
        raise R6Error("locked replay identity is invalid")
    if replay.get("split_revision") != "grouped-rolling-v1":
        raise R6Error("locked replay split revision is invalid")
    lineage = policy["lineage"]
    _assert_replay_rows_schema(replay["training_rows"], lineage=lineage)
    if replay["training_rows_sha256"] != _digest(replay["training_rows"]):
        raise R6Error("locked replay training rows digest is invalid")
    if replay["policy_sha256"] != _policy_payload_digest(policy):
        raise R6Error("locked replay policy digest is invalid")
    _assert_artifact_integrity(run, "official run", has_id=True)
    _assert_artifact_integrity(policy, "official candidate policy", has_id=True)
    _assert_artifact_integrity(pointer, "official candidate pointer", has_id=False)
    if (
        lineage["locked_replay_id"] != replay["artifact_id"]
        or replay["training_snapshot_id"] != lineage["training_snapshot_id"]
        or replay["baseline_artifact_id"] != lineage["baseline_artifact_id"]
        or replay["candidate_head"] != lineage["candidate_head"]
        or replay["profile_contract_id"] != lineage["profile_contract_id"]
        or replay["offline_gate_sha256"] != lineage["offline_gate_sha256"]
        or replay["training_rows_sha256"] != lineage["training_rows_sha256"]
        or replay["model_cohort_sha256"] != lineage["model_cohort_sha256"]
    ):
        raise R6Error("locked replay lineage binding is invalid")
    _assert_artifact_integrity(replay, "official locked replay", has_id=True)
    if r5 is not None and heads is not None and (
        not isinstance(r5, Mapping)
        or not all(key in r5 for key in ("baseline_id", "raw_watermark", "label_head"))
        or not isinstance(heads, Mapping)
        or set(heads) != {"candidate", "label", "manifest"}
    ):
        raise R6Error("candidate lineage binding inputs are malformed")
    if r5 is not None and heads is not None and (
        run["raw_watermark"] != r5["raw_watermark"]
        or run["baseline_artifact_id"] != r5["baseline_id"]
            or run["manifest_head"] != heads["manifest"]
            or run["candidate_head"] != heads["candidate"]
            or run["label_head"] != heads["label"]
            or run["label_head"] != r5["label_head"]
        or lineage["candidate_head"] != heads["candidate"]
        or lineage["label_chain_head"] != r5["label_head"]
        or lineage["raw_watermark"] != r5["raw_watermark"]
        or lineage["baseline_artifact_id"] != r5["baseline_id"]
        or lineage["model_cohort_sha256"] != r5.get("cohort_sha256")
        or lineage["split_plan_id"] != r5.get("split_plan_id")
        or lineage["profile_contract_id"] != r5.get("profile_contract_id")
    ):
        raise R6Error("official candidate lineage head binding is invalid")
    if state is not None:
        if r5 is None or heads is None:
            raise R6Error("worker state binding inputs are unavailable")
        if not isinstance(r5, Mapping) or not isinstance(heads, Mapping):
            raise R6Error("worker state binding inputs are malformed")
        _assert_worker_state_schema(module, state, run=run, r5=r5, heads=heads)


def _official_candidate(
    module: Any,
    clone: Path,
    r5: Mapping[str, Any],
    result: Mapping[str, Any],
    *,
    before_id: str,
    before_snapshot: Mapping[str, str],
    before_candidate_ledger: Mapping[str, Any],
) -> dict[str, Any]:
    """Read official store artifacts and rerun its lineage verifier."""

    if result.get("p5_allowed") is not True or result.get("teachers_available") is not False:
        raise R6Error("official worker did not reach empty-teacher R6 gate")
    promotion = result.get("promotion")
    if not isinstance(promotion, Mapping) or promotion.get("status") != "candidate":
        raise R6Error("official worker did not generate a candidate")
    store = module.store
    try:
        pointer = store.read_pointer(clone, "candidate")
        policy = module._load_policy(str(pointer["policy_id"]), clone)
        baseline = store.read_sealed(store.distillation_dir(clone) / "baselines" / f"{r5['baseline_id']}.json", schema=module.BASELINE_SCHEMA)
    except (KeyError, module.DistillationError, store.DistillationStoreError) as exc:
        raise R6Error("official candidate artifacts are unavailable") from exc
    lineage = policy.get("lineage")
    if not isinstance(lineage, Mapping):
        raise R6Error("official candidate lineage is unavailable")
    replay_id = lineage.get("locked_replay_id")
    replay_id = _artifact_id(replay_id)
    replay = store.read_sealed(store.distillation_dir(clone) / "locked-replays" / f"{replay_id}.json", schema="chronovisor.recall-distill-locked-replay.v1")
    if module._verify_candidate_lineage(clone, pointer, policy, baseline) is not None:
        raise R6Error("official candidate lineage verifier rejected artifacts")
    if (
        set(pointer) != {"schema", "namespace", "kind", "policy_id", "seal_sha256"}
        or pointer.get("kind") != "candidate-policy-pointer"
        or policy.get("kind") != "tiny-logistic-policy"
        or replay.get("kind") != "locked-replay-input"
        or replay.get("artifact_id") != replay_id
        or policy.get("artifact_id") != pointer.get("policy_id")
    ):
        raise R6Error("official candidate artifact kind is invalid")
    if before_id:
        raise R6Error("official candidate was already present; R6 generation is not initial")
    expected = {
        "baseline_artifact_id": r5["baseline_id"],
        "raw_watermark": r5["raw_watermark"],
        "label_chain_head": r5["label_head"],
        "model_cohort_sha256": r5["cohort_sha256"],
        "split_plan_id": r5["split_plan_id"],
        "profile_contract_id": r5["profile_contract_id"],
    }
    if any(lineage.get(key) != value for key, value in expected.items()):
        raise R6Error("official candidate lineage does not bind current R5 facts")
    ledger_snapshots: dict[str, Mapping[str, Any]] = {}
    for name, filename in (
        ("candidate", "candidate-ledger.jsonl"),
        ("label", "label-ledger.jsonl"),
        ("manifest", "rally-manifest.jsonl"),
    ):
        snapshot = store.chain_head(store.distillation_dir(clone) / filename)
        if (
            not isinstance(snapshot, Mapping)
            or set(snapshot) != {"records", "head_sha256"}
            or isinstance(snapshot.get("records"), bool)
            or not isinstance(snapshot.get("records"), int)
            or snapshot["records"] < 0
        ):
            raise R6Error("official clone ledger checkpoint schema is invalid")
        _strict_head(snapshot.get("head_sha256"), f"{name} ledger checkpoint head")
        ledger_snapshots[name] = snapshot
    heads = {
        name: str(snapshot["head_sha256"])
        for name, snapshot in ledger_snapshots.items()
    }
    if lineage.get("candidate_head") != heads["candidate"]:
        raise R6Error("official candidate lineage candidate head mismatch")
    after_candidate_ledger = _candidate_ledger_state(store, clone)
    run_id = result.get("run_id")
    state_sha = result.get("state_sha256")
    run_id = _artifact_id(run_id)
    state_sha = _artifact_id(state_sha)
    run = store.read_sealed(
        store.distillation_dir(clone) / "runs" / f"{run_id}.json",
        schema="chronovisor.recall-distill-run.v1",
    )
    state = store.read_sealed(store.distillation_dir(clone) / store.STATE_FILE, schema=store.DISTILLATION_SCHEMA)
    _assert_candidate_artifact_schemas(module, pointer, policy, replay, run)
    if (
        run.get("kind") != "bounded-chunk"
        or run.get("artifact_id") != run_id
        or state.get("kind") != "worker-state"
        or state.get("seal_sha256") != state_sha
        or state.get("run_id") != run_id
        or run.get("baseline_artifact_id") != r5["baseline_id"]
        or run.get("raw_watermark") != r5["raw_watermark"]
        or run.get("label_head") != r5["label_head"]
        or run.get("candidate_head") != heads["candidate"]
        or run.get("label_head") != heads["label"]
        or run.get("manifest_head") != heads["manifest"]
        or state.get("baseline_artifact_id") != r5["baseline_id"]
        or state.get("raw_watermark") != r5["raw_watermark"]
        or state.get("manifest_chain_head") != heads["manifest"]
    ):
        raise R6Error("official worker run/state readback is invalid")
    _assert_candidate_artifact_schemas(
        module,
        pointer,
        policy,
        replay,
        run,
        state=state,
        r5=r5,
        heads=heads,
    )
    after_snapshot = _worker_snapshot(clone)
    _require_candidate_newness(
        before_snapshot, after_snapshot, before_candidate_ledger,
        after_candidate_ledger, lineage,
    )
    return {
        "candidate_id": _artifact_id(pointer["policy_id"]), "pointer": pointer,
        "policy": policy, "locked_replay": replay, "run": run, "state": state,
        "baseline_id": _artifact_id(baseline["artifact_id"]), "heads": heads,
        "clone_candidate_status": "created", "worker_before": dict(before_snapshot),
        "worker_after": after_snapshot, "candidate_ledger_before": dict(before_candidate_ledger),
        "candidate_ledger_after": after_candidate_ledger,
    }


def _assert_output_schema(value: Mapping[str, Any]) -> None:
    kind = value.get("kind")
    if not isinstance(kind, str):
        raise R6Error("formal output kind is invalid")
    common = {
        "schema", "namespace", "kind", "source_commit", "external_provider_calls",
        "provider_calls", "provider_attempts", "egress_attempts", "clone_candidate_published",
        "clone_candidate_status", "production_candidate_published", "clone_cleanup_remaining",
        "cleanup_receipt", "seal_sha256",
    }
    expected = {
        "r6-preflight-declined": common | {"r5", "teachers", "production_identity_before", "production_identity_after"},
        "r6-official-worker-blocked": common | {"baseline_id", "reason", "teachers", "production_identity_before", "production_identity_after"},
        "r6-official-clone-evidence": common | {"baseline_id", "candidate_id", "completion", "pointer", "policy", "locked_replay", "run", "state", "candidate_lineage_binding", "candidate_worker_newness", "idempotence", "production_identity_before", "production_identity_after"},
    }.get(kind)
    if expected is None or set(value) != expected:
        raise R6Error("formal output schema is not closed")
    if value.get("schema") != R6_SCHEMA or value.get("namespace") != "recall-distillation":
        raise R6Error("formal output identity is invalid")
    if value.get("seal_sha256") != _digest({key: item for key, item in value.items() if key != "seal_sha256"}):
        raise R6Error("formal output seal is invalid")
    source_commit = value.get("source_commit")
    if not isinstance(source_commit, str) or len(source_commit) != 40 or set(source_commit) - _SHA:
        raise R6Error("formal source commit is invalid")
    for field in ("external_provider_calls", "provider_calls", "provider_attempts", "egress_attempts", "clone_cleanup_remaining"):
        _strict_int(value[field], f"formal output {field}")
    for field in ("clone_candidate_published", "production_candidate_published"):
        _strict_bool(value[field], f"formal output {field}")
    if value["clone_candidate_status"] not in {"none", "created"}:
        raise R6Error("formal clone candidate status is invalid")
    for field in ("production_identity_before", "production_identity_after"):
        _artifact_id(value[field])
    if value["clone_cleanup_remaining"] != 0:
        raise R6Error("formal clone cleanup is incomplete")
    if not isinstance(value["teachers"], Mapping):
        raise R6Error("formal teachers mapping is invalid")
    cleanup = value["cleanup_receipt"]
    if (
        not isinstance(cleanup, Mapping)
        or set(cleanup) != {"schema", "namespace", "kind", "removed_roots", "clone_proofs", "remaining", "seal_sha256"}
        or cleanup.get("schema") != "chronovisor.recall-r6-cleanup.v1"
        or cleanup.get("namespace") != "recall-distillation"
        or cleanup.get("kind") != "clone-cleanup"
        or cleanup.get("remaining") != 0
        or cleanup.get("seal_sha256") != _digest({key: item for key, item in cleanup.items() if key != "seal_sha256"})
        or not isinstance(cleanup.get("removed_roots"), list)
        or not isinstance(cleanup.get("clone_proofs"), list)
        or len(cleanup["removed_roots"]) != len(cleanup["clone_proofs"])
    ):
        raise R6Error("formal cleanup receipt is invalid")
    for root in cleanup["removed_roots"]:
        if not isinstance(root, str) or not Path(root).is_absolute():
            raise R6Error("formal cleanup root is invalid")
    for proof in cleanup["clone_proofs"]:
        if not isinstance(proof, Mapping) or set(proof) != _CLONE_PROOF_KEYS | {"target_after"}:
            raise R6Error("formal cleanup clone proof is not closed")
        if (
            proof.get("schema") != "chronovisor.recall-r6-clone-proof.v1"
            or proof.get("namespace") != "recall-distillation"
            or proof.get("kind") != "clone-provenance"
            or proof.get("method") != _CLONE_METHOD
            or proof.get("filesystem") != "apfs"
            or proof.get("same_device") is not True
            or proof.get("nonoverlap") is not True
        ):
            raise R6Error("formal cleanup clone proof identity is invalid")
        _strict_int(proof["euid"], "formal cleanup clone proof euid")
        for path_name in ("production_path", "source_path", "clone_path"):
            if not isinstance(proof.get(path_name), str) or not Path(proof[path_name]).is_absolute():
                raise R6Error("formal cleanup clone proof path is invalid")
        for stat_name in ("production", "clone"):
            stat_value = proof[stat_name]
            if not isinstance(stat_value, Mapping) or set(stat_value) != _CLONE_STAT_KEYS:
                raise R6Error("formal cleanup clone proof stat is invalid")
            if stat_value["path"] != proof[f"{stat_name}_path"]:
                raise R6Error("formal cleanup clone proof stat path is invalid")
            for field in ("path",):
                if not isinstance(stat_value[field], str) or not Path(stat_value[field]).is_absolute():
                    raise R6Error("formal cleanup clone proof stat path is invalid")
            for field in ("dev", "ino", "uid", "mode"):
                _strict_int(stat_value[field], f"formal cleanup clone proof {stat_name}.{field}")
        _artifact_id(proof["target_before"])
        _artifact_id(proof["target_after"])
    for removed_root, proof in zip(cleanup["removed_roots"], cleanup["clone_proofs"], strict=True):
        if removed_root != str(Path(proof["clone_path"]).parent) or Path(removed_root).exists():
            raise R6Error("formal cleanup root is not bound to a removed clone")
    if kind != "r6-official-clone-evidence":
        if kind == "r6-preflight-declined":
            if (
                value["external_provider_calls"] != 0
                or value["provider_calls"] != 0
                or value["provider_attempts"] != 0
                or value["egress_attempts"] != 0
                or value["clone_candidate_published"] is not False
                or value["clone_candidate_status"] != "none"
                or value["production_candidate_published"] is not False
            ):
                raise R6Error("formal preflight-declined status is inconsistent")
            r5 = value["r5"]
            if (
                not isinstance(r5, Mapping)
                or set(r5) != {"passed", "reason", "baseline_id"}
                or r5.get("passed") is not False
                or r5.get("baseline_id") != ""
            ):
                raise R6Error("formal declined R5 schema is invalid")
            _strict_text(r5["reason"], "formal declined R5 reason")
        else:
            baseline_id = value["baseline_id"]
            if baseline_id != "":
                _artifact_id(baseline_id)
            _strict_text(value["reason"], "formal blocked reason", allow_empty=False)
            if (
                value["external_provider_calls"] != 0
                or value["provider_calls"] != 0
                or value["clone_candidate_published"] is not False
                or value["clone_candidate_status"] != "none"
                or value["production_candidate_published"] is not False
            ):
                raise R6Error("formal blocked status is inconsistent")
        return
    completion = value["completion"]
    if not isinstance(completion, Mapping):
        raise R6Error("formal output completion is not an object")
    _assert_completion_schema(completion)
    if (
        completion["source_commit"] != value["source_commit"]
        or completion["baseline_id"] != value["baseline_id"]
        or completion["candidate_id"] != value["candidate_id"]
        or completion["provider_calls"] != value["provider_calls"]
        or completion["provider_attempts"] != value["provider_attempts"]
        or completion["egress_attempts"] != value["egress_attempts"]
    ):
        raise R6Error("formal output completion binding is invalid")
    if (
        value["external_provider_calls"] != 0
        or value["clone_candidate_published"] is not True
        or value["clone_candidate_status"] != "created"
        or value["production_candidate_published"] is not False
    ):
        raise R6Error("formal clone-evidence status is inconsistent")
    binding = value["candidate_lineage_binding"]
    newness = value["candidate_worker_newness"]
    idempotence = value["idempotence"]
    if not isinstance(binding, Mapping) or set(binding) != {"source", "runtime", "r5", "heads"}:
        raise R6Error("formal output nested schema is not closed")
    _assert_source_snapshot_schema(binding["source"])
    _artifact_id(binding["source"]["module_sha256"])
    _assert_runtime_schema(binding["runtime"], source=binding["source"])
    runtime_root = Path(binding["runtime"]["runtime_root"])
    clone_paths = {
        proof["clone_path"] for proof in cleanup["clone_proofs"] if isinstance(proof, Mapping)
    }
    if (
        binding["runtime"]["runtime_root"] not in clone_paths
        or binding["runtime"]["config_path"] != str(runtime_root / "config.toml")
        or runtime_root.is_symlink()
    ):
        raise R6Error("formal runtime root is not clone-bound")
    _assert_r5_schema(binding["r5"])
    if binding["r5"]["passed"] is not True:
        raise R6Error("formal candidate R5 gate is not passed")
    if value["source_commit"] != binding["source"]["head"] or value["baseline_id"] != binding["r5"]["baseline_id"]:
        raise R6Error("formal output source/R5 identity is not bound")
    if completion["runtime"] != binding["runtime"]:
        raise R6Error("formal output completion runtime is not bound")
    if not isinstance(binding["heads"], Mapping) or set(binding["heads"]) != {"candidate", "label", "manifest"}:
        raise R6Error("formal output heads schema is not closed")
    for head in binding["heads"].values():
        _strict_head(head, "formal output head")
    if not isinstance(newness, Mapping) or set(newness) != {"before", "after", "candidate_ledger_before", "candidate_ledger_after"}:
        raise R6Error("formal output worker newness schema is not closed")
    for snapshot_name in ("before", "after"):
        snapshot = newness[snapshot_name]
        if not isinstance(snapshot, Mapping) or set(snapshot) != {"candidate-policy.json", "candidate-ledger.jsonl", "label-ledger.jsonl", "rally-manifest.jsonl", "state.json", "policies", "locked-replays", "runs"}:
            raise R6Error("formal output worker snapshot schema is not closed")
        for digest in snapshot.values():
            if not isinstance(digest, str) or (digest and (len(digest) != 64 or set(digest) - _SHA)):
                raise R6Error("formal output worker snapshot digest is invalid")
    before_snapshot = newness["before"]
    after_snapshot = newness["after"]
    if before_snapshot["candidate-policy.json"]:
        raise R6Error("formal output candidate was not initially absent")
    for artifact_name in ("candidate-policy.json", "state.json", "policies", "locked-replays", "runs"):
        if not after_snapshot[artifact_name] or after_snapshot[artifact_name] == before_snapshot[artifact_name]:
            raise R6Error("formal output candidate worker newness is invalid")
    for ledger_name in ("candidate_ledger_before", "candidate_ledger_after"):
        ledger = newness[ledger_name]
        if not isinstance(ledger, Mapping) or set(ledger) != {"records", "head_sha256"}:
            raise R6Error("formal output ledger snapshot schema is not closed")
        _strict_int(ledger["records"], f"formal output {ledger_name}.records")
        _strict_head(ledger["head_sha256"], f"formal output {ledger_name}.head_sha256")
    if (
        newness["candidate_ledger_after"]["records"] <= newness["candidate_ledger_before"]["records"]
        or newness["candidate_ledger_after"]["head_sha256"] == newness["candidate_ledger_before"]["head_sha256"]
        or newness["candidate_ledger_after"]["head_sha256"] != binding["heads"]["candidate"]
    ):
        raise R6Error("formal output candidate ledger newness is invalid")
    expected_idempotence = {
        "candidate_id", "heads", "second_clone_match", "pointer_sha256", "policy_sha256",
        "locked_replay_sha256", "run_sha256", "state_sha256", "state_stable_sha256",
        "second_pointer_sha256", "second_policy_sha256", "second_locked_replay_sha256",
        "second_run_sha256", "second_state_stable_sha256", "second_state_sha256", "state_full_match",
        "state_stable_match", "volatile_state_fields",
    }
    if not isinstance(idempotence, Mapping) or set(idempotence) != expected_idempotence:
        raise R6Error("formal output idempotence schema is not closed")
    if value["candidate_id"] != idempotence["candidate_id"] or binding["heads"] != idempotence["heads"]:
        raise R6Error("formal output idempotence binding is invalid")
    _artifact_id(value["candidate_id"])
    for name in (
        "pointer_sha256", "policy_sha256", "locked_replay_sha256", "run_sha256", "state_sha256",
        "state_stable_sha256", "second_pointer_sha256", "second_policy_sha256",
        "second_locked_replay_sha256", "second_run_sha256", "second_state_stable_sha256",
        "second_state_sha256",
    ):
        _artifact_id(idempotence[name])
    _strict_bool(idempotence["second_clone_match"], "formal output second_clone_match")
    _strict_bool(idempotence["state_full_match"], "formal output state_full_match")
    _strict_bool(idempotence["state_stable_match"], "formal output state_stable_match")
    second_artifacts_match = all(
        idempotence[first] == idempotence[second]
        for first, second in (
            ("pointer_sha256", "second_pointer_sha256"),
            ("policy_sha256", "second_policy_sha256"),
            ("locked_replay_sha256", "second_locked_replay_sha256"),
            ("run_sha256", "second_run_sha256"),
        )
    )
    if idempotence["second_clone_match"] != second_artifacts_match:
        raise R6Error("formal output clone match receipt is inconsistent")
    if idempotence["second_clone_match"] != idempotence["state_full_match"]:
        raise R6Error("formal output full-match receipt is inconsistent")
    if idempotence["state_full_match"] != (idempotence["state_sha256"] == idempotence["second_state_sha256"]):
        raise R6Error("formal output full digest receipt is inconsistent")
    if idempotence["state_stable_match"] != (idempotence["state_stable_sha256"] == idempotence["second_state_stable_sha256"]):
        raise R6Error("formal output stable digest receipt is inconsistent")
    if idempotence["state_stable_match"] is not True:
        raise R6Error("formal output stable idempotence gate failed")
    if idempotence["volatile_state_fields"] != sorted(_VOLATILE_STATE_FIELDS):
        raise R6Error("formal output volatile state field list is invalid")
    for artifact_name in ("pointer", "policy", "locked_replay", "run", "state"):
        if not isinstance(value[artifact_name], Mapping):
            raise R6Error(f"formal output {artifact_name} is not an object")
    expected_artifact_digests = {
        "pointer_sha256": value["pointer"],
        "policy_sha256": value["policy"],
        "locked_replay_sha256": value["locked_replay"],
        "run_sha256": value["run"],
        "state_sha256": value["state"],
    }
    for digest_name, artifact in expected_artifact_digests.items():
        if idempotence[digest_name] != _digest(artifact):
            raise R6Error(f"formal output {digest_name} is not bound to artifact")
    if idempotence["state_stable_sha256"] != _stable_state_digest(value["state"]):
        raise R6Error("formal output stable state digest is invalid")
    policy_lineage = value["policy"].get("lineage")
    if not isinstance(policy_lineage, Mapping) or not isinstance(value["locked_replay"].get("training_rows"), list):
        raise R6Error("formal output candidate lineage is invalid")
    _assert_output_artifact_envelopes(value, r5=binding["r5"], heads=binding["heads"])
    worker = completion["worker"]
    if (
        worker["run_id"] != value["run"]["artifact_id"]
        or worker["state_sha256"] != value["state"]["seal_sha256"]
        or worker["ox_workset"] != value["run"]["ox_workset"]
        or worker["local_workset"] != value["run"]["local_workset"]
        or worker["ox_profile_contract_id"] != value["run"]["ox_profile_contract_id"]
        or worker["split_plan_id"] != value["state"]["split_plan_id"]
        or worker["p5_allowed"] != value["run"]["p5_allowed"]
        or worker["r6_provider_attempts"] != completion["provider_attempts"]
        or worker["r6_egress_attempts"] != completion["egress_attempts"]
        or worker["r6_git_sha256"] != binding["source"]["git_executable_sha256"]
    ):
        raise R6Error("formal output completion worker artifacts are not bound")


def _write_output(output: Path, value: Mapping[str, Any]) -> Path:
    _assert_output_schema(value)
    output.mkdir(parents=True, exist_ok=True)
    if _symlink_component(output) or any(item.is_symlink() for item in output.iterdir()):
        raise R6Error("output root is unsafe")
    payload = _canonical(value)
    name = f"{_digest(value)}.json"
    path = output / name
    directory_fd = os.open(
        output,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        try:
            descriptor = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600, dir_fd=directory_fd)
        except FileExistsError:
            if _read_sealed(path, R6_SCHEMA) != value:
                raise R6Error("formal output immutable conflict") from None
        else:
            try:
                written = os.write(descriptor, payload)
                if written != len(payload) or os.fstat(descriptor).st_size != len(payload):
                    raise R6Error("formal output write failed")
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            os.fsync(directory_fd)
        state = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if not stat.S_ISREG(state.st_mode) or state.st_size != len(payload):
            raise R6Error("formal output final state is invalid")
    finally:
        os.close(directory_fd)
    if _read_sealed(path, R6_SCHEMA) != value:
        raise R6Error("formal artifact independent readback failed")
    return path


def _assert_clone_proof(
    clone: Path,
    proof: Mapping[str, Any],
    *,
    output: Path,
    post_target: str,
    other_clones: tuple[Path, ...] = (),
) -> dict[str, Any]:
    """Validate the parent-held clone attestation immediately before cleanup."""

    if not isinstance(proof, Mapping) or set(proof) != _CLONE_PROOF_KEYS:
        raise R6Error("clone provenance proof is not closed")
    if (
        proof.get("schema") != "chronovisor.recall-r6-clone-proof.v1"
        or proof.get("namespace") != "recall-distillation"
        or proof.get("kind") != "clone-provenance"
        or proof.get("method") != _CLONE_METHOD
        or proof.get("filesystem") != "apfs"
        or proof.get("same_device") is not True
        or proof.get("nonoverlap") is not True
        or proof.get("euid") != os.geteuid()
        or not isinstance(proof.get("production_path"), str)
        or not isinstance(proof.get("source_path"), str)
        or not proof["source_path"]
        or not isinstance(proof.get("clone_path"), str)
        or not isinstance(proof.get("target_before"), str)
    ):
        raise R6Error("clone provenance proof is invalid")
    _artifact_id(proof["target_before"])
    _artifact_id(post_target)
    production = Path(str(proof["production_path"]))
    source = Path(str(proof["source_path"]))
    expected_clone = Path(str(proof["clone_path"]))
    if (
        not production.is_absolute()
        or not source.is_absolute()
        or not expected_clone.is_absolute()
        or expected_clone.resolve(strict=False) != clone.resolve(strict=False)
        or _symlink_component(production)
        or _symlink_component(source)
        or _symlink_component(expected_clone)
    ):
        raise R6Error("clone provenance paths are invalid")
    production_stat = proof.get("production")
    clone_stat = proof.get("clone")
    if (
        not isinstance(production_stat, Mapping)
        or set(production_stat) != _CLONE_STAT_KEYS
        or not isinstance(clone_stat, Mapping)
        or set(clone_stat) != _CLONE_STAT_KEYS
    ):
        raise R6Error("clone provenance stat schema is invalid")
    if dict(production_stat) != _clone_stat(production):
        raise R6Error("production clone provenance stat changed")
    if dict(clone_stat) != _clone_stat(clone):
        raise R6Error("clone provenance stat changed or was replaced")
    if production_stat["dev"] != clone_stat["dev"] or _overlap(production, clone):
        raise R6Error("clone provenance device or overlap binding failed")
    if _target_identity(production) != proof["target_before"]:
        raise R6Error("production contents changed before provenance cleanup")
    if _filesystem_type(clone) != "apfs":
        raise R6Error("clone provenance filesystem changed")
    _assert_no_overlap(production, source, output, clone, *other_clones)
    if _target_identity(clone) != post_target:
        raise R6Error("clone contents changed before provenance cleanup")
    receipt = {**dict(proof), "target_after": post_target}
    if set(receipt) != _CLONE_PROOF_KEYS | {"target_after"}:
        raise R6Error("clone cleanup proof is not closed")
    return receipt


def _finalize_output(
    output: Path,
    unsigned: Mapping[str, Any],
    *clone_records: tuple[Path, Mapping[str, Any], str],
) -> dict[str, Any]:
    """Validate parent-held attestations, then remove every harness clone."""

    if not clone_records:
        raise R6Error("clone provenance records are unavailable")
    if any(
        not isinstance(record, tuple)
        or len(record) != 3
        or not isinstance(record[0], Path)
        or not isinstance(record[1], Mapping)
        or not isinstance(record[2], str)
        for record in clone_records
    ):
        raise R6Error("clone provenance records are invalid")
    removed: list[str] = []
    proofs: list[dict[str, Any]] = []
    clones = tuple(record[0] for record in clone_records)
    for clone, proof, post_target in clone_records:
        receipt_proof = _assert_clone_proof(
            clone, proof, output=output, post_target=post_target,
            other_clones=tuple(item for item in clones if item != clone),
        )
        proofs.append(receipt_proof)
        _cleanup_clone(clone)
        removed.append(str(clone.parent))
    cleanup_receipt = _seal({
        "schema": "chronovisor.recall-r6-cleanup.v1",
        "namespace": "recall-distillation",
        "kind": "clone-cleanup",
        "removed_roots": removed,
        "clone_proofs": proofs,
        "remaining": sum(Path(path).exists() for path in removed),
    })
    if (
        set(cleanup_receipt) != {"schema", "namespace", "kind", "removed_roots", "clone_proofs", "remaining", "seal_sha256"}
        or cleanup_receipt["schema"] != "chronovisor.recall-r6-cleanup.v1"
        or cleanup_receipt["namespace"] != "recall-distillation"
        or cleanup_receipt["kind"] != "clone-cleanup"
        or cleanup_receipt["remaining"] != 0
        or cleanup_receipt["seal_sha256"] != _digest({key: item for key, item in cleanup_receipt.items() if key != "seal_sha256"})
    ):
        raise R6Error("clone cleanup receipt is invalid")
    value = _seal({**unsigned, "clone_cleanup_remaining": 0, "cleanup_receipt": cleanup_receipt})
    _write_output(output, value)
    return value


def _guard_counts(exc: BaseException) -> dict[str, int]:
    """Report rejected attempts separately from completed provider calls."""

    if isinstance(exc, R6GuardError):
        return {
            "provider_calls": 0,
            "provider_attempts": exc.provider_attempts,
            "egress_attempts": exc.egress_attempts,
        }
    return {"provider_calls": 0, "provider_attempts": 0, "egress_attempts": 0}


def _assert_child_containment(value: object) -> None:
    """Require the parent-owned isolated-child receipt in every completion."""

    if not isinstance(value, Mapping) or set(value) != _CHILD_CONTAINMENT_KEYS:
        raise R6Error("completion child containment schema is not closed")
    if value["schema"] != "chronovisor.recall-r6-child-containment.v1":
        raise R6Error("completion child containment schema is invalid")
    for field in ("registered_descendants", "rejected_registry_entries", "remaining_descendants"):
        _strict_int(value[field], f"completion child containment {field}")
        if value[field] < 0:
            raise R6Error(f"completion child containment {field} is invalid")
    if value["remaining_descendants"] != 0 or value["registry_fd_closed"] is not True:
        raise R6Error("completion child containment is incomplete")
    sandbox = value["sandbox"]
    if not isinstance(sandbox, Mapping) or set(sandbox) != _SANDBOX_IDENTITY_KEYS:
        raise R6Error("completion child sandbox identity is invalid")
    for field in ("dev", "ino", "uid", "mode"):
        _strict_int(sandbox[field], f"completion child sandbox {field}")
        if sandbox[field] < 0:
            raise R6Error("completion child sandbox identity is invalid")
    if sandbox["path"] != "/usr/bin/sandbox-exec":
        raise R6Error("completion child sandbox identity is invalid")
    _artifact_id(sandbox["sha256"])
    if dict(sandbox) != _sandbox_identity():
        raise R6Error("completion child sandbox identity is not parent-bound")


def _assert_worker_result_schema(value: object) -> None:
    """Validate the exact mapping returned by _persist_distillation_chunk."""

    if not isinstance(value, Mapping):
        raise R6Error("completion worker result is not an object")
    present_ramp = set(value).intersection(_OX_RAMP_KEYS)
    expected = _WORKER_KEYS | (set(_OX_RAMP_KEYS) if present_ramp else set())
    if set(value) != expected or (present_ramp and present_ramp != set(_OX_RAMP_KEYS)):
        raise R6Error("completion worker schema is not closed")
    _strict_text(value["status"], "completion worker status", allow_empty=False)
    if value["status"] not in {"deferred", "ready", "capture_only"}:
        raise R6Error("completion worker status is invalid")
    for field in (
        "processed", "candidate_snapshots", "labels_written", "counterfactuals_written",
        "manifest_backlog", "candidate_backlog", "r6_egress_attempts", "r6_provider_attempts",
    ):
        _strict_int(value[field], f"completion worker {field}")
    for field in (
        "p5_allowed", "teachers_available", "counterfactual_available",
        "ox_profile_stopped", "cold_start_pending",
    ):
        _strict_bool(value[field], f"completion worker {field}")
    _strict_id_or_empty(value["ox_profile_contract_id"], "completion worker OX profile contract")
    _strict_id_or_empty(value["split_plan_id"], "completion worker split plan")
    _artifact_id(value["run_id"])
    _artifact_id(value["state_sha256"])
    _strict_id_or_empty(value["r6_git_sha256"], "completion worker git digest")
    _assert_child_containment(value["r6_child_containment"])

    promotion = value["promotion"]
    if not isinstance(promotion, Mapping) or set(promotion) not in (
        {"status", "policy_id"}, {"status", "reason"}
    ):
        raise R6Error("completion worker promotion schema is not closed")
    _strict_text(promotion["status"], "completion worker promotion status", allow_empty=False)
    if promotion["status"] == "candidate":
        if set(promotion) != {"status", "policy_id"}:
            raise R6Error("completion worker candidate promotion is invalid")
        _artifact_id(promotion["policy_id"])
    else:
        if set(promotion) != {"status", "reason"}:
            raise R6Error("completion worker held promotion is invalid")
        _strict_text(promotion["reason"], "completion worker promotion reason", allow_empty=False)

    evaluation = value["rollout_evaluation"]
    if not isinstance(evaluation, Mapping):
        raise R6Error("completion worker rollout evaluation is not an object")
    evaluation_sets = (
        {"status"},
        {"status", "reason", "rollout_percent", "learning_halted", "last_run_id", "changed"},
        {"status", "negative_vetoes", "rollout_percent", "learning_halted", "last_run_id", "changed"},
    )
    if set(evaluation) not in evaluation_sets:
        raise R6Error("completion worker rollout evaluation schema is not closed")
    _strict_text(evaluation["status"], "completion worker rollout status", allow_empty=False)
    if "reason" in evaluation:
        _strict_text(evaluation["reason"], "completion worker rollout reason", allow_empty=False)
    if "negative_vetoes" in evaluation:
        _strict_int(evaluation["negative_vetoes"], "completion worker negative vetoes")
    if "rollout_percent" in evaluation:
        _strict_int(evaluation["rollout_percent"], "completion worker rollout percent", maximum=100)
        _strict_bool(evaluation["learning_halted"], "completion worker learning halted")
        _strict_id_or_empty(evaluation["last_run_id"], "completion worker last run")
        _strict_bool(evaluation["changed"], "completion worker rollout changed")

    for owner_name, owner in (("ox", value["ox_workset"]), ("local", value["local_workset"])):
        if not isinstance(owner, Mapping):
            raise R6Error(f"completion worker {owner_name} workset is not an object")
    if value["ox_workset"] == {} and value["local_workset"] == {}:
        pass
    elif value["ox_workset"] != {} and value["local_workset"] == {}:
        _assert_workset_schema(
            value["ox_workset"], flavor="ox", include_timing=False,
            expected_profile_contract=value["ox_profile_contract_id"],
        )
    elif value["local_workset"] != {} and value["ox_workset"] == {}:
        _assert_workset_schema(value["local_workset"], flavor="local", include_timing=True)
    else:
        raise R6Error("completion worker workset selection is invalid")
    if present_ramp:
        if value["ox_ramp_cap"] not in {1, 2, 5, 10}:
            raise R6Error("completion worker OX ramp cap is invalid")
        _strict_int(value["ox_ramp_cap"], "completion worker ox_ramp_cap")
        _strict_int(value["ox_ramp_valid_receipts"], "completion worker ox_ramp_valid_receipts")
        _strict_int(value["ox_ramp_provider_attempts"], "completion worker ox_ramp_provider_attempts")
        if value["ox_ramp_request_revision"] != "json-schema-core-label-abstain-16k-240s-v6":
            raise R6Error("completion worker OX ramp revision is invalid")


def _assert_completion_schema(value: Mapping[str, Any]) -> None:
    expected = {
        "schema", "namespace", "kind", "source_commit", "baseline_id", "candidate_id",
        "worker", "external_provider_calls", "provider_calls", "provider_attempts", "egress_attempts",
        "clone_candidate_published", "clone_candidate_status",
        "production_candidate_published", "runtime", "seal_sha256",
    }
    if (
        set(value) != expected
        or value.get("schema") != "chronovisor.recall-r6-completion.v2"
        or value.get("namespace") != "recall-distillation"
        or value.get("kind") != "official-worker-clone-completion"
        or value.get("seal_sha256") != _digest({key: item for key, item in value.items() if key != "seal_sha256"})
    ):
        raise R6Error("completion schema is not closed")
    source_commit = value.get("source_commit")
    if not isinstance(source_commit, str) or len(source_commit) != 40 or set(source_commit) - _SHA:
        raise R6Error("completion source commit is invalid")
    _artifact_id(value["baseline_id"])
    _artifact_id(value["candidate_id"])
    _assert_worker_result_schema(value["worker"])
    _assert_runtime_schema(value["runtime"])
    for field in ("external_provider_calls", "provider_calls", "provider_attempts", "egress_attempts"):
        _strict_int(value[field], f"completion {field}")


def run_once(*, production: Path, source: Path, output: Path, source_commit: str) -> dict[str, Any]:
    _reject_ambient_git_env()
    _assert_guard_modules()
    assert_root_matrix(production, source, output)
    before_source = source_snapshot(source)
    if before_source["head"] != source_commit or before_source["status_count"] != 0:
        raise R6Error("source identity is not exact and clean")
    clone: Path | None = None
    target_before = ""
    clone_proof: dict[str, Any] | None = None
    clone_second: Path | None = None
    clone_second_proof: dict[str, Any] | None = None
    before_production = ""
    phase_watchdog = _phase_watchdog(timeout=_PHASE_TIMEOUT_SECONDS)
    phase_entered = False
    try:
        phase_watchdog.__enter__()
        phase_entered = True
        clone, target_before, clone_proof = _clone(production, source=source, with_proof=True)
        before_production = target_before
        assert clone_proof is not None
        clone_state = clone.lstat()
        if clone.is_symlink() or not stat.S_ISDIR(clone_state.st_mode) or _target_identity(clone) != target_before:
            raise R6Error("APFS clone ownership or identity failed")
        _assert_no_overlap(production, source, output, clone)
        with _clone_runtime_context(clone, source):
            try:
                module, runtime = _load_runtime(source, clone)
                preflight = _r5_preflight(module, clone, before_source)
            except (R6Error, OSError, ValueError) as exc:
                after_production = _target_identity(production)
                if after_production != before_production or source_snapshot(source) != before_source:
                    raise R6Error("production/source changed during R5 preflight") from exc
                return _finalize_output(output, {
                    "schema": R6_SCHEMA, "namespace": "recall-distillation", "kind": "r6-official-worker-blocked",
                    "source_commit": source_commit, "baseline_id": "", "reason": str(exc).split(":", 1)[0],
                    "teachers": {}, "external_provider_calls": 0, **_guard_counts(exc), "clone_candidate_published": False,
                    "clone_candidate_status": "none",
                    "production_candidate_published": False, "production_identity_before": before_production,
                    "production_identity_after": after_production,
                }, (clone, clone_proof, _target_identity(clone)))

            if not preflight["passed"]:
                before_production = target_before
                after_production = _target_identity(production)
                if source_snapshot(source) != before_source:
                    raise R6Error("source changed during R5 preflight")
                if after_production != before_production:
                    return _finalize_output(output, {
                        "schema": R6_SCHEMA, "namespace": "recall-distillation", "kind": "r6-official-worker-blocked",
                        "source_commit": source_commit, "baseline_id": "", "reason": "production target changed during R5 preflight",
                        "teachers": {}, "external_provider_calls": 0, "provider_calls": 0, "provider_attempts": 0, "egress_attempts": 0, "clone_candidate_published": False,
                        "clone_candidate_status": "none", "production_candidate_published": False,
                        "production_identity_before": before_production, "production_identity_after": after_production,
                    }, (clone, clone_proof, _target_identity(clone)))
                return _finalize_output(output, {
                    "schema": R6_SCHEMA, "namespace": "recall-distillation", "kind": "r6-preflight-declined",
                    "source_commit": source_commit, "r5": preflight, "teachers": {}, "external_provider_calls": 0, "provider_calls": 0, "provider_attempts": 0, "egress_attempts": 0,
                    "clone_candidate_published": False, "production_candidate_published": False,
                    "clone_candidate_status": "none",
                    "production_identity_before": before_production, "production_identity_after": after_production,
                }, (clone, clone_proof, _target_identity(clone)))
            before_production = _target_identity(production)
            if before_production != target_before or _target_identity(clone) != target_before:
                raise R6Error("APFS clone identity differs from production")
            try:
                candidate_before_id = _candidate_pointer_id(module, clone)
                worker_before = _worker_snapshot(clone)
                candidate_ledger_before = _candidate_ledger_state(module.store, clone)
                worker = _official_chunk_isolated(source, clone)
                candidate = _official_candidate(
                    module, clone, preflight, worker, before_id=candidate_before_id,
                    before_snapshot=worker_before, before_candidate_ledger=candidate_ledger_before,
                )
            except (R6Error, OSError, ValueError) as exc:
                after_production = _target_identity(production)
                if after_production != before_production or source_snapshot(source) != before_source:
                    raise R6Error("production/source changed during clone evaluation") from exc
                return _finalize_output(output, {
                    "schema": R6_SCHEMA, "namespace": "recall-distillation", "kind": "r6-official-worker-blocked",
                    "source_commit": source_commit, "baseline_id": preflight["baseline_id"],
                    "reason": str(exc).split(":", 1)[0], "teachers": {}, "external_provider_calls": 0, **_guard_counts(exc),
                    "clone_candidate_published": False, "production_candidate_published": False,
                    "clone_candidate_status": "none",
                    "production_identity_before": before_production, "production_identity_after": after_production,
                }, (clone, clone_proof, _target_identity(clone)))
            clone_second, second_target_before, clone_second_proof = _clone(production, source=source, with_proof=True)
            try:
                if _target_identity(clone_second) != second_target_before or second_target_before != target_before:
                    raise R6Error("second APFS clone identity failed")
                with _clone_runtime_context(clone_second, source):
                    module_second, _ = _load_runtime(source, clone_second)
                    repeat_r5 = _r5_preflight(module_second, clone_second, before_source)
                    if repeat_r5 != preflight:
                        raise R6Error("second clone R5 facts differ")
                    repeat_before_id = _candidate_pointer_id(module_second, clone_second)
                    repeat_worker_before = _worker_snapshot(clone_second)
                    repeat_candidate_ledger_before = _candidate_ledger_state(module_second.store, clone_second)
                    repeat_worker = _official_chunk_isolated(source, clone_second)
                    repeat_candidate = _official_candidate(
                        module_second, clone_second, repeat_r5, repeat_worker,
                        before_id=repeat_before_id, before_snapshot=repeat_worker_before,
                        before_candidate_ledger=repeat_candidate_ledger_before,
                    )
                first_state_full_sha256 = _digest(candidate["state"])
                first_state_stable_sha256 = _stable_state_digest(candidate["state"])
                second_state_full_sha256 = _digest(repeat_candidate["state"])
                second_state_stable_sha256 = _stable_state_digest(repeat_candidate["state"])
                if (
                    repeat_candidate["candidate_id"] != candidate["candidate_id"]
                    or repeat_candidate["heads"] != candidate["heads"]
                    or _digest(repeat_candidate["pointer"]) != _digest(candidate["pointer"])
                    or _digest(repeat_candidate["policy"]) != _digest(candidate["policy"])
                    or _digest(repeat_candidate["locked_replay"]) != _digest(candidate["locked_replay"])
                    or _digest(repeat_candidate["run"]) != _digest(candidate["run"])
                    or second_state_stable_sha256 != first_state_stable_sha256
                ):
                    raise R6Error("second frozen clone is not idempotent")
            except (R6Error, OSError, ValueError) as exc:
                after_production = _target_identity(production)
                if after_production != before_production or source_snapshot(source) != before_source:
                    raise R6Error("production/source changed during idempotence check") from exc
                return _finalize_output(output, {
                    "schema": R6_SCHEMA, "namespace": "recall-distillation", "kind": "r6-official-worker-blocked",
                    "source_commit": source_commit, "baseline_id": preflight["baseline_id"],
                    "reason": str(exc).split(":", 1)[0], "teachers": {}, "external_provider_calls": 0, **_guard_counts(exc),
                    "clone_candidate_published": False, "production_candidate_published": False,
                    "clone_candidate_status": "none",
                    "production_identity_before": before_production, "production_identity_after": after_production,
                },
                    (clone_second, clone_second_proof, _target_identity(clone_second)),
                    (clone, clone_proof, _target_identity(clone)),
                )
        after_production = _target_identity(production)
        if after_production != before_production or source_snapshot(source) != before_source:
            raise R6Error("production/source changed during clone evaluation")
        completion = _seal({
            "schema": "chronovisor.recall-r6-completion.v2", "namespace": "recall-distillation",
            "kind": "official-worker-clone-completion", "source_commit": source_commit,
            "baseline_id": candidate["baseline_id"], "candidate_id": candidate["candidate_id"],
            "worker": worker, "external_provider_calls": 0, "provider_calls": 0, "provider_attempts": worker["r6_provider_attempts"],
            "egress_attempts": worker["r6_egress_attempts"], "clone_candidate_published": True,
            "clone_candidate_status": candidate["clone_candidate_status"],
            "production_candidate_published": False, "runtime": runtime,
        })
        _assert_completion_schema(completion)
        unsigned_artifact = {
            "schema": R6_SCHEMA, "namespace": "recall-distillation", "kind": "r6-official-clone-evidence",
            "source_commit": source_commit, "baseline_id": candidate["baseline_id"],
            "candidate_id": candidate["candidate_id"], "completion": completion,
            "pointer": candidate["pointer"], "policy": candidate["policy"], "locked_replay": candidate["locked_replay"], "run": candidate["run"], "state": candidate["state"],
            "candidate_lineage_binding": {"source": before_source, "runtime": runtime, "r5": preflight, "heads": candidate["heads"]},
            "candidate_worker_newness": {
                "before": candidate["worker_before"], "after": candidate["worker_after"],
                "candidate_ledger_before": candidate["candidate_ledger_before"],
                "candidate_ledger_after": candidate["candidate_ledger_after"],
            },
            "idempotence": {
                "candidate_id": candidate["candidate_id"], "heads": candidate["heads"],
                "second_clone_match": first_state_full_sha256 == second_state_full_sha256,
                "state_full_match": first_state_full_sha256 == second_state_full_sha256,
                "state_stable_match": first_state_stable_sha256 == second_state_stable_sha256,
                "pointer_sha256": _digest(candidate["pointer"]), "policy_sha256": _digest(candidate["policy"]),
                "locked_replay_sha256": _digest(candidate["locked_replay"]), "run_sha256": _digest(candidate["run"]),
                "state_sha256": first_state_full_sha256,
                "second_pointer_sha256": _digest(repeat_candidate["pointer"]),
                "second_policy_sha256": _digest(repeat_candidate["policy"]),
                "second_locked_replay_sha256": _digest(repeat_candidate["locked_replay"]),
                "second_run_sha256": _digest(repeat_candidate["run"]),
                "state_stable_sha256": first_state_stable_sha256,
                "second_state_stable_sha256": second_state_stable_sha256,
                "second_state_sha256": second_state_full_sha256,
                "volatile_state_fields": sorted(_VOLATILE_STATE_FIELDS),
            },
            "external_provider_calls": 0, "provider_calls": 0, "provider_attempts": worker["r6_provider_attempts"], "egress_attempts": worker["r6_egress_attempts"], "clone_candidate_published": True, "clone_candidate_status": candidate["clone_candidate_status"], "production_candidate_published": False,
            "production_identity_before": before_production, "production_identity_after": after_production,
        }
        if clone_second is None:
            raise R6Error("second clone is missing")
        if clone_second_proof is None:
            raise R6Error("second clone provenance proof is missing")
        return _finalize_output(
            output,
            unsigned_artifact,
            (clone_second, clone_second_proof, _target_identity(clone_second)),
            (clone, clone_proof, _target_identity(clone)),
        )
    finally:
        try:
            if phase_entered:
                phase_watchdog.__exit__(None, None, None)
        finally:
            try:
                if clone_second is not None:
                    _cleanup_clone(clone_second)
            finally:
                if clone is not None:
                    _cleanup_clone(clone)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-worker", action="store_true")
    parser.add_argument("--production-root", type=Path)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-commit")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--clone-root", type=Path)
    parser.add_argument("--child-capability")
    args = parser.parse_args(argv)
    if args.official_worker:
        if args.clone_root is None or args.child_capability is None:
            parser.error("--official-worker requires --clone-root and --child-capability")
        return _official_worker_main(
            source=args.source_root.resolve(strict=True),
            clone=args.clone_root.resolve(strict=True),
            capability=args.child_capability,
        )
    if args.production_root is None or args.source_commit is None or args.output is None:
        parser.error("--production-root, --source-commit, and --output are required")
    try:
        result = run_once(production=args.production_root.resolve(strict=True), source=args.source_root.resolve(strict=True), output=args.output.resolve(strict=False), source_commit=args.source_commit)
    except (R6Error, OSError, ValueError) as exc:
        print(f"r6 harness failed: {str(exc).split(':', 1)[0]}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
