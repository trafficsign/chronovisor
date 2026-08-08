"""Run the sealed runtime-access analysis with hard operational bounds.

This runner deliberately lives outside the sealed analyzer and machine-fact
toolchain manifests.  It never imports either toolchain in the supervising
process; analysis and validation each happen in their own clean detached Git
worktree and fresh Python process.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final, Protocol, cast

SCHEMA_VERSION: Final = 1
PROGRESS_KIND: Final = "chronovisor-runtime-access-watchdog-progress"
_FULL_SHA1 = re.compile(r"[0-9a-f]{40}\Z")
_MAX_WALL_CLOCK_SECONDS: Final = 43_200
_ANALYSIS_EXIT_NONCONVERGENCE: Final = 20
_ANALYSIS_EXIT_FAILED: Final = 21
_GIT_TIMEOUT_SECONDS: Final = 30.0
_STDERR_LIMIT_BYTES: Final = 16_384
_MAX_TERM_GRACE_SECONDS: Final = 60
_MAX_POLL_SECONDS: Final = 60
_KILL_REAP_SECONDS: Final = 5.0
_TERMINAL_STATUSES: Final = frozenset(
    {"completed", "nonconvergence", "wall_clock_timeout", "stalled", "failed"}
)
_RUNNER_PATH: Final = "scripts/runtime_access_watchdog.py"


class WatchdogError(RuntimeError):
    """Raised when watchdog setup cannot safely begin."""


class WatchdogDeadlineError(WatchdogError):
    """Raised when a bounded setup stage consumes the hard analysis deadline."""


class WatchdogLiveProcessError(WatchdogError):
    """Raised when a subprocess group could still be alive after bounded teardown."""


class _Process(Protocol):
    pid: int
    returncode: int | None

    def poll(self) -> int | None: ...

    def wait(self, timeout: float | None = None) -> int: ...


class _Popen(Protocol):
    def __call__(
        self,
        args: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
        stdin: int,
        stdout: int,
        stderr: int,
        shell: bool,
        start_new_session: bool,
    ) -> _Process: ...


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError, RecursionError) as exc:
        raise WatchdogError("value cannot be encoded as canonical JSON") from exc


def _load_canonical_json(raw: bytes) -> object:
    def reject_constant(token: str) -> Any:
        raise WatchdogError(f"JSON contains a non-finite number: {token}")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise WatchdogError(f"JSON contains duplicate key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=unique_object,
        )
    except WatchdogError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise WatchdogError("JSON is malformed") from exc
    if _canonical_json_bytes(value) != raw:
        raise WatchdogError("JSON is not in canonical byte form")
    return value


def _exact_positive_seconds(value: object, *, name: str, maximum: int | None = None) -> float:
    if type(value) not in {int, float}:
        raise ValueError(f"{name} must be an exact positive number")
    numeric = cast(float, value)
    if not math.isfinite(numeric) or numeric <= 0:
        raise ValueError(f"{name} must be an exact positive number")
    if maximum is not None and numeric > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return float(numeric)


def _absolute_without_symlink_resolution(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _assert_safe_parent(path: Path) -> Path:
    absolute = _absolute_without_symlink_resolution(path)
    parent = absolute.parent
    chain: list[Path] = []
    cursor = parent
    while cursor != cursor.parent:
        chain.append(cursor)
        cursor = cursor.parent
    chain.append(cursor)
    for component in reversed(chain):
        try:
            stat_result = component.lstat()
        except FileNotFoundError as exc:
            raise WatchdogError(f"output parent does not exist: {component}") from exc
        if component.is_symlink():
            raise WatchdogError(f"symlink parent is unsafe: {component}")
        if not component.is_dir():
            raise WatchdogError(f"output parent is not a directory: {component}")
        if stat_result.st_nlink < 1:
            raise WatchdogError(f"output parent is unlinked: {component}")
    return absolute


def _directory_fd(parent: Path) -> int:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    return os.open(parent, flags)


def _write_all(descriptor: int, raw: bytes) -> None:
    view = memoryview(raw)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("short write")
        view = view[written:]


def _exclusive_write(path: Path, raw: bytes) -> None:
    absolute = _assert_safe_parent(path)
    parent_fd = _directory_fd(absolute.parent)
    temporary_name: str | None = None
    descriptor = -1
    try:
        for attempt in range(128):
            candidate = f".{absolute.name}.publish-{os.getpid()}-{attempt}"
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
            try:
                descriptor = os.open(candidate, flags, 0o600, dir_fd=parent_fd)
            except FileExistsError:
                continue
            temporary_name = candidate
            break
        if descriptor < 0 or temporary_name is None:
            raise WatchdogError("cannot allocate a private publication file")
        _write_all(descriptor, raw)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.link(
            temporary_name,
            absolute.name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
            follow_symlinks=False,
        )
        os.fsync(parent_fd)
        os.unlink(temporary_name, dir_fd=parent_fd)
        temporary_name = None
        os.fsync(parent_fd)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary_name is not None:
            try:
                os.unlink(temporary_name, dir_fd=parent_fd)
            except FileNotFoundError:
                pass
        os.close(parent_fd)


def _atomic_replace(path: Path, raw: bytes) -> None:
    absolute = _assert_safe_parent(path)
    parent_fd = _directory_fd(absolute.parent)
    temporary_name: str | None = None
    descriptor = -1
    try:
        for attempt in range(128):
            candidate = f".{absolute.name}.tmp-{os.getpid()}-{attempt}"
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
            try:
                descriptor = os.open(
                    candidate, flags, 0o600, dir_fd=parent_fd
                )
            except FileExistsError:
                continue
            temporary_name = candidate
            break
        if descriptor < 0 or temporary_name is None:
            raise WatchdogError("cannot allocate an atomic progress file")
        _write_all(descriptor, raw)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.replace(
            temporary_name,
            absolute.name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
        )
        temporary_name = None
        os.fsync(parent_fd)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary_name is not None:
            try:
                os.unlink(temporary_name, dir_fd=parent_fd)
            except FileNotFoundError:
                pass
        os.close(parent_fd)


def _path_exists_even_if_dangling(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    return True


def _clean_environment(*, pythonpath: Path | None = None) -> dict[str, str]:
    blocked_python = {
        "VIRTUAL_ENV",
        "__PYVENV_LAUNCHER__",
        "PYTHONHOME",
        "PYTHONINSPECT",
        "PYTHONSTARTUP",
        "PYTHONUSERBASE",
        "PYTHONPATH",
    }
    clean = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("GIT_", "DYLD_"))
        and key not in blocked_python
        and key not in {"LD_PRELOAD", "BASH_ENV", "ENV"}
    }
    clean["PYTHONNOUSERSITE"] = "1"
    clean["PYTHONDONTWRITEBYTECODE"] = "1"
    clean["GIT_NO_REPLACE_OBJECTS"] = "1"
    clean["GIT_CONFIG_NOSYSTEM"] = "1"
    clean["GIT_CONFIG_GLOBAL"] = os.devnull
    clean["GIT_TERMINAL_PROMPT"] = "0"
    if pythonpath is not None:
        clean["PYTHONPATH"] = os.fspath(pythonpath)
    return clean


def _remaining_seconds(deadline_monotonic: float | None) -> float:
    if deadline_monotonic is None:
        return _GIT_TIMEOUT_SECONDS
    remaining = deadline_monotonic - time.monotonic()
    if remaining <= 0:
        raise WatchdogDeadlineError("wall-clock deadline expired during setup")
    return min(_GIT_TIMEOUT_SECONDS, remaining)


def _stop_git_process(process: subprocess.Popen[bytes]) -> None:
    cleanup_errors: list[str] = []
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except OSError as exc:
            cleanup_errors.append(f"group SIGTERM failed: {exc}")
    try:
        process.communicate(timeout=1.0)
    except subprocess.TimeoutExpired:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except OSError as exc:
                cleanup_errors.append(f"group SIGKILL failed: {exc}")
                try:
                    process.kill()
                except OSError as kill_exc:
                    cleanup_errors.append(f"direct kill failed: {kill_exc}")
        try:
            process.communicate(timeout=_KILL_REAP_SECONDS)
        except (OSError, subprocess.TimeoutExpired) as exc:
            cleanup_errors.append(f"bounded reap failed: {exc}")
    except OSError as exc:
        cleanup_errors.append(f"communicate failed: {exc}")
        if process.poll() is None:
            try:
                process.kill()
            except OSError as kill_exc:
                cleanup_errors.append(f"direct kill failed: {kill_exc}")
            try:
                process.wait(timeout=_KILL_REAP_SECONDS)
            except (OSError, subprocess.TimeoutExpired) as wait_exc:
                cleanup_errors.append(f"bounded direct reap failed: {wait_exc}")
    if process.poll() is None:
        cleanup_errors.append("git child remains alive after termination")
    try:
        _drain_process_group(process.pid, grace_seconds=1.0)
    except BaseException as exc:
        cleanup_errors.append(f"git descendant drain failed: {exc}")
    if cleanup_errors:
        raise WatchdogLiveProcessError("; ".join(cleanup_errors))


def _git(
    repository: Path,
    *arguments: str,
    deadline_monotonic: float | None = None,
) -> subprocess.CompletedProcess[bytes]:
    argv = ["git", "--no-replace-objects", *arguments]
    timeout = _remaining_seconds(deadline_monotonic)
    try:
        process = subprocess.Popen(
            argv,
            cwd=repository,
            env=_clean_environment(),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            start_new_session=True,
        )
    except OSError as exc:
        raise WatchdogError(f"git execution failed: {exc}") from exc
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except BaseException as exc:
        try:
            _stop_git_process(process)
        except WatchdogError as cleanup_exc:
            error_class = (
                WatchdogLiveProcessError
                if isinstance(cleanup_exc, WatchdogLiveProcessError)
                else WatchdogError
            )
            raise error_class(
                f"git {' '.join(arguments)} failed and cleanup failed: {cleanup_exc}"
            ) from exc
        if isinstance(exc, subprocess.TimeoutExpired):
            if (
                deadline_monotonic is not None
                and time.monotonic() >= deadline_monotonic
            ):
                raise WatchdogDeadlineError(
                    f"git {' '.join(arguments)} reached the hard deadline"
                ) from exc
            raise WatchdogError(f"git {' '.join(arguments)} timed out") from exc
        raise
    try:
        _drain_process_group(process.pid, grace_seconds=1.0)
    except BaseException as exc:
        raise WatchdogLiveProcessError(
            f"git descendant process group could not be drained: {exc}"
        ) from exc
    completed = subprocess.CompletedProcess(argv, process.returncode, stdout, stderr)
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise WatchdogError(
            f"git {' '.join(arguments)} failed ({completed.returncode}): {detail}"
        )
    return completed


def _resolve_revision(
    repository: Path,
    revision: object,
    *,
    deadline_monotonic: float | None = None,
) -> str:
    if type(revision) is not str or _FULL_SHA1.fullmatch(revision) is None:
        raise ValueError(
            "revision must be an explicit lowercase full 40-character commit SHA"
        )
    object_format = _git(
        repository,
        "rev-parse",
        "--show-object-format",
        deadline_monotonic=deadline_monotonic,
    ).stdout
    if object_format.strip() != b"sha1":
        raise WatchdogError("watchdog requires a sha1 Git repository")
    resolved = _git(
        repository,
        "rev-parse",
        "--verify",
        f"{revision}^{{commit}}",
        deadline_monotonic=deadline_monotonic,
    ).stdout.decode("ascii", errors="strict").strip()
    if resolved != revision:
        raise WatchdogError("revision did not resolve exactly to itself")
    return revision


def _verify_runner_at_revision(
    repository: Path,
    revision: str,
    *,
    deadline_monotonic: float | None = None,
) -> None:
    runner_path = repository / _RUNNER_PATH
    if runner_path.is_symlink():
        raise WatchdogError("runner path must not be a symlink")
    expected = runner_path.resolve()
    if Path(__file__).resolve() != expected:
        raise WatchdogError("runner must execute from the target repository")
    head = _git(
        repository,
        "rev-parse",
        "--verify",
        "HEAD^{commit}",
        deadline_monotonic=deadline_monotonic,
    ).stdout
    if head.decode("ascii", errors="strict").strip() != revision:
        raise WatchdogError("runner commit must be the requested target revision")
    committed = _git(
        repository,
        "show",
        f"{revision}:{_RUNNER_PATH}",
        deadline_monotonic=deadline_monotonic,
    ).stdout
    try:
        current = expected.read_bytes()
    except OSError as exc:
        raise WatchdogError("runner source cannot be read") from exc
    if current != committed:
        raise WatchdogError("runner source differs from the target revision")


def _add_worktree(
    repository: Path,
    path: Path,
    revision: str,
    *,
    deadline_monotonic: float | None = None,
) -> None:
    _git(
        repository,
        "worktree",
        "add",
        "--detach",
        os.fspath(path),
        revision,
        deadline_monotonic=deadline_monotonic,
    )


def _remove_worktree(
    repository: Path,
    path: Path,
    *,
    deadline_monotonic: float | None = None,
) -> None:
    _git(
        repository,
        "worktree",
        "remove",
        "--force",
        os.fspath(path),
        deadline_monotonic=deadline_monotonic,
    )
    if _path_exists_even_if_dangling(path):
        raise WatchdogError(f"worktree cleanup left a directory: {path}")
    if _worktree_registered(
        repository, path, deadline_monotonic=deadline_monotonic
    ):
        raise WatchdogError(f"worktree cleanup left registry metadata: {path}")


def _worktree_registered(
    repository: Path,
    path: Path,
    *,
    deadline_monotonic: float | None = None,
) -> bool:
    raw = _git(
        repository,
        "worktree",
        "list",
        "--porcelain",
        "-z",
        deadline_monotonic=deadline_monotonic,
    ).stdout
    expected = os.fsencode(_absolute_without_symlink_resolution(path))
    return any(
        field.removeprefix(b"worktree ") == expected
        for field in raw.split(b"\0")
        if field.startswith(b"worktree ")
    )


_ANALYSIS_CHILD = r'''
open(__import__("sys").argv[3], "xb").write(b"0")
import json
import os
import sys
import threading
import time

repo, result_path, heartbeat_path, outcome_path = sys.argv[1:5]
sys.path.insert(0, repo)
os.chmod(heartbeat_path, 0o600)

def canonical(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False,
                      sort_keys=True, separators=(",", ":")).encode("utf-8")

def atomic(path, raw):
    temporary = path + ".tmp"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short write")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)

def beat():
    sequence = 1
    while True:
        atomic(heartbeat_path, canonical({"sequence": sequence,
                                         "monotonic": time.monotonic()}))
        sequence += 1
        time.sleep(5.0)

threading.Thread(target=beat, daemon=True, name="runtime-watchdog-heartbeat").start()

try:
    from scripts.runtime_ownership import machine_facts
    document = machine_facts.run_sealed_effective_analysis(__import__("pathlib").Path(repo))
    for name, module in tuple(sys.modules.items()):
        if name == "scripts" or name.startswith("scripts.runtime_ownership"):
            source = getattr(module, "__file__", None)
            if source is not None and os.path.commonpath((repo, os.path.realpath(source))) != repo:
                raise RuntimeError("project module loaded outside clean worktree")
    raw = machine_facts.canonical_bytes(document)
    atomic(result_path, raw)
    atomic(outcome_path, canonical({"status": "success"}))
except BaseException as exc:
    exact_nonconvergence = (exc.__class__.__module__ ==
                            "scripts.runtime_ownership.access_model" and
                            exc.__class__.__name__ == "AnalysisNonConvergenceError")
    if exact_nonconvergence and type(getattr(exc, "payload", None)) is dict:
        atomic(outcome_path, canonical({"status": "nonconvergence",
                                       "payload": exc.payload}))
        raise SystemExit(20)
    atomic(outcome_path, canonical({"status": "failed",
                                   "error": f"{exc.__class__.__module__}."
                                            f"{exc.__class__.__name__}: {exc}"}))
    raise SystemExit(21)
'''


_VALIDATION_CHILD = r'''
import hashlib
import json
import os
import sys
repo, result_path, outcome_path = sys.argv[1:4]
sys.path.insert(0, repo)
from pathlib import Path
from scripts.runtime_ownership.declarations import discover_concrete
from scripts.runtime_ownership.machine_facts import (
    EFFECTIVE_SOURCE_REVISION,
    build_declaration_adapter,
    load_machine_fact_cache,
)
from scripts.runtime_ownership.manifests import SOURCE_MANIFEST_KIND, committed_snapshot
for name, module in tuple(sys.modules.items()):
    if name == "scripts" or name.startswith("scripts.runtime_ownership"):
        source = getattr(module, "__file__", None)
        if source is not None and os.path.commonpath((repo, os.path.realpath(source))) != repo:
            raise RuntimeError("validation module loaded outside clean worktree")
snapshot = committed_snapshot(repo, EFFECTIVE_SOURCE_REVISION,
                              manifest_kind=SOURCE_MANIFEST_KIND)
adapter = build_declaration_adapter(discover_concrete(snapshot))
raw = Path(result_path).read_bytes()
load_machine_fact_cache(Path(repo), raw, adapter=adapter)
outcome = json.dumps({"sha256": hashlib.sha256(raw).hexdigest()},
                     sort_keys=True, separators=(",", ":")).encode("utf-8")
descriptor = os.open(outcome_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
try:
    view = memoryview(outcome)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("short write")
        view = view[written:]
    os.fsync(descriptor)
finally:
    os.close(descriptor)
'''


def _launch_python(
    code: str,
    arguments: Sequence[Path],
    *,
    worktree: Path,
    stderr_path: Path,
    popen: _Popen = subprocess.Popen,
) -> _Process:
    argv = [sys.executable, "-I", "-c", code, *(os.fspath(item) for item in arguments)]
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(stderr_path, flags, 0o600)
    try:
        return popen(
            argv,
            cwd=worktree,
            env=_clean_environment(pythonpath=worktree),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=descriptor,
            shell=False,
            start_new_session=True,
        )
    finally:
        os.close(descriptor)


def _heartbeat_fingerprint(path: Path) -> tuple[int, int, int] | None:
    try:
        stat_result = path.stat()
    except FileNotFoundError:
        return None
    return (stat_result.st_ino, stat_result.st_mtime_ns, stat_result.st_size)


def _progress_document(
    revision: str,
    status: str,
    sequence: int,
    *,
    started_monotonic: float,
    now_monotonic: float,
    pid: int | None = None,
    heartbeat_age_seconds: float | None = None,
    reason: str | None = None,
    error: str | None = None,
    payload: object | None = None,
) -> dict[str, object]:
    document: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "kind": PROGRESS_KIND,
        "revision": revision,
        "status": status,
        "sequence": sequence,
        "elapsed_seconds": round(max(0.0, now_monotonic - started_monotonic), 6),
    }
    if pid is not None:
        document["pid"] = pid
    if heartbeat_age_seconds is not None:
        document["heartbeat_age_seconds"] = round(heartbeat_age_seconds, 6)
    if reason is not None:
        document["reason"] = reason
    if error is not None:
        document["error"] = error
    if payload is not None:
        document["payload"] = payload
    return document


def _monitor_process(
    process: _Process,
    heartbeat_path: Path,
    progress_path: Path,
    *,
    revision: str,
    started_monotonic: float,
    wall_clock_seconds: float,
    stale_after_seconds: float,
    poll_seconds: float,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    replace_progress: Callable[[Path, bytes], None] = _atomic_replace,
) -> tuple[str, int, float]:
    sequence = 0
    last_fingerprint: tuple[int, int, int] | None = None
    last_seen_monotonic = started_monotonic
    while True:
        now_monotonic = monotonic()
        elapsed = max(0.0, now_monotonic - started_monotonic)
        fingerprint = _heartbeat_fingerprint(heartbeat_path)
        if fingerprint is not None and fingerprint != last_fingerprint:
            last_fingerprint = fingerprint
            last_seen_monotonic = now_monotonic
        age = max(0.0, now_monotonic - last_seen_monotonic)
        # Deadline wins ties; a completed process wins over stale heartbeat.
        if elapsed >= wall_clock_seconds:
            return "wall_clock_timeout", sequence, age
        returncode = process.poll()
        if returncode is not None:
            return "exited", sequence, age
        if age >= stale_after_seconds:
            return "stalled", sequence, age
        sequence += 1
        running = _progress_document(
            revision,
            "running",
            sequence,
            started_monotonic=started_monotonic,
            now_monotonic=now_monotonic,
            pid=process.pid,
            heartbeat_age_seconds=age,
        )
        replace_progress(progress_path, _canonical_json_bytes(running))
        sleep(
            min(
                poll_seconds,
                wall_clock_seconds - elapsed,
                stale_after_seconds - age,
            )
        )


def _terminate_process_group(
    process: _Process,
    *,
    grace_seconds: float,
    kill_group: Callable[[int, int], None] = os.killpg,
) -> None:
    if process.poll() is not None:
        process.wait()
        return
    try:
        kill_group(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    except OSError as exc:
        raise WatchdogError(f"cannot send SIGTERM to child group: {exc}") from exc
    try:
        process.wait(timeout=grace_seconds)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        kill_group(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except OSError as exc:
        raise WatchdogError(f"cannot send SIGKILL to child group: {exc}") from exc
    try:
        process.wait(timeout=_KILL_REAP_SECONDS)
    except subprocess.TimeoutExpired as exc:
        raise WatchdogError("child could not be reaped after SIGKILL") from exc


def _process_group_exists(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _drain_process_group(
    process_group_id: int,
    *,
    grace_seconds: float,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    if not _process_group_exists(process_group_id):
        return
    try:
        os.killpg(process_group_id, signal.SIGTERM)
    except ProcessLookupError:
        return
    except OSError as exc:
        raise WatchdogError(f"cannot drain child group with SIGTERM: {exc}") from exc
    deadline = monotonic() + grace_seconds
    while _process_group_exists(process_group_id) and monotonic() < deadline:
        sleep(min(0.05, max(0.0, deadline - monotonic())))
    if not _process_group_exists(process_group_id):
        return
    try:
        os.killpg(process_group_id, signal.SIGKILL)
    except ProcessLookupError:
        return
    except OSError as exc:
        raise WatchdogError(f"cannot drain child group with SIGKILL: {exc}") from exc
    kill_deadline = monotonic() + _KILL_REAP_SECONDS
    while _process_group_exists(process_group_id) and monotonic() < kill_deadline:
        sleep(min(0.05, max(0.0, kill_deadline - monotonic())))
    if _process_group_exists(process_group_id):
        raise WatchdogError("child process group remains alive after SIGKILL")


def _wait_until_deadline(
    process: _Process,
    *,
    started_monotonic: float,
    wall_clock_seconds: float,
    grace_seconds: float,
    monotonic: Callable[[], float] = time.monotonic,
) -> int | None:
    remaining = wall_clock_seconds - (monotonic() - started_monotonic)
    if remaining <= 0:
        _terminate_process_group(process, grace_seconds=grace_seconds)
        return None
    try:
        return process.wait(timeout=remaining)
    except subprocess.TimeoutExpired:
        _terminate_process_group(process, grace_seconds=grace_seconds)
        return None


def _parse_outcome(raw: bytes, returncode: int) -> tuple[str, object | None, str | None]:
    value = _load_canonical_json(raw)
    if type(value) is not dict:
        raise WatchdogError("child outcome must be an object")
    outcome = cast(dict[str, Any], value)
    status = outcome.get("status")
    if returncode == 0 and outcome == {"status": "success"}:
        return "success", None, None
    if (
        returncode == _ANALYSIS_EXIT_NONCONVERGENCE
        and set(outcome) == {"status", "payload"}
        and status == "nonconvergence"
        and type(outcome["payload"]) is dict
    ):
        return "nonconvergence", outcome["payload"], None
    if (
        returncode == _ANALYSIS_EXIT_FAILED
        and set(outcome) == {"status", "error"}
        and status == "failed"
        and type(outcome["error"]) is str
        and outcome["error"]
    ):
        return "failed", None, outcome["error"]
    raise WatchdogError("child outcome does not match its exit status")


def _cleanup_directory(path: Path) -> None:
    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        return
    if _path_exists_even_if_dangling(path):
        raise WatchdogError(f"temporary cleanup left a directory: {path}")


def _create_temporary_root() -> Path:
    path = Path(tempfile.mkdtemp(prefix="chronovisor-access-watchdog-"))
    try:
        os.chmod(path, 0o700)
    except BaseException as exc:
        try:
            _cleanup_directory(path)
        except BaseException as cleanup_exc:
            raise WatchdogError(
                f"private temporary setup failed: {exc}; cleanup failed and "
                f"state was retained at {path}: {cleanup_exc}"
            ) from exc
        raise
    return path


def _bounded_stderr(path: Path) -> str | None:
    try:
        with path.open("rb") as stream:
            raw = stream.read(_STDERR_LIMIT_BYTES + 1)
    except OSError:
        return None
    truncated = len(raw) > _STDERR_LIMIT_BYTES
    text = raw[:_STDERR_LIMIT_BYTES].decode("utf-8", errors="replace").strip()
    if not text:
        return None
    return text + (" [truncated]" if truncated else "")


def _replace_terminal_progress(path: Path, raw: bytes) -> None:
    first_error: BaseException | None = None
    for _attempt in range(2):
        try:
            _atomic_replace(path, raw)
            return
        except BaseException as exc:
            if first_error is None:
                first_error = exc
    assert first_error is not None
    raise WatchdogError(
        f"terminal progress update failed after bounded retry: {first_error}"
    ) from first_error


def _cleanup_worktrees(
    repository: Path,
    rows: Sequence[tuple[str, bool, bool, Path]],
    *,
    deadline_monotonic: float,
) -> tuple[list[str], bool]:
    errors: list[str] = []
    live_process = False
    for label, attempted, registered_hint, path in rows:
        if not attempted or live_process:
            continue
        try:
            registered = registered_hint or _worktree_registered(
                repository, path, deadline_monotonic=deadline_monotonic
            )
            if registered:
                _remove_worktree(
                    repository, path, deadline_monotonic=deadline_monotonic
                )
        except BaseException as exc:
            errors.append(f"{label} worktree cleanup failed: {exc}")
            if isinstance(exc, WatchdogLiveProcessError):
                live_process = True
    return errors, live_process


def run_analysis(
    repository: Path,
    output_path: Path,
    progress_path: Path,
    *,
    revision: str,
    wall_clock_seconds: int | float = 1800,
    stale_after_seconds: int | float = 60,
    term_grace_seconds: int | float = 10,
    poll_seconds: int | float = 1,
) -> dict[str, object]:
    """Run, independently validate, and no-clobber publish one sealed result."""

    wall_clock = _exact_positive_seconds(
        wall_clock_seconds,
        name="wall_clock_seconds",
        maximum=_MAX_WALL_CLOCK_SECONDS,
    )
    stale_after = _exact_positive_seconds(
        stale_after_seconds, name="stale_after_seconds"
    )
    term_grace = _exact_positive_seconds(
        term_grace_seconds,
        name="term_grace_seconds",
        maximum=_MAX_TERM_GRACE_SECONDS,
    )
    poll = _exact_positive_seconds(
        poll_seconds, name="poll_seconds", maximum=_MAX_POLL_SECONDS
    )
    started_monotonic = time.monotonic()
    deadline_monotonic = started_monotonic + wall_clock
    repository = repository.resolve(strict=True)
    revision = _resolve_revision(
        repository, revision, deadline_monotonic=deadline_monotonic
    )
    _verify_runner_at_revision(
        repository, revision, deadline_monotonic=deadline_monotonic
    )

    output = _assert_safe_parent(output_path)
    progress = _assert_safe_parent(progress_path)
    if output == progress:
        raise WatchdogError("output and progress paths must be distinct")
    if _path_exists_even_if_dangling(output):
        raise FileExistsError(os.fspath(output))
    if _path_exists_even_if_dangling(progress):
        raise FileExistsError(os.fspath(progress))

    temporary_root = _create_temporary_root()
    initial = _progress_document(
        revision,
        "running",
        0,
        started_monotonic=started_monotonic,
        now_monotonic=started_monotonic,
    )
    try:
        _exclusive_write(progress, _canonical_json_bytes(initial))
    except BaseException:
        _cleanup_directory(temporary_root)
        raise
    analysis_worktree = temporary_root / "analysis-worktree"
    validation_worktree = temporary_root / "validation-worktree"
    private_result = temporary_root / "result.json"
    private_heartbeat = temporary_root / "heartbeat.json"
    private_outcome = temporary_root / "outcome.json"
    analysis_stderr = temporary_root / "analysis.stderr"
    validation_stderr = temporary_root / "validation.stderr"
    validation_outcome = temporary_root / "validation-outcome.json"
    process: _Process | None = None
    validator_process: _Process | None = None
    analysis_registered = False
    validation_registered = False
    analysis_attempted = False
    validation_attempted = False
    sequence = 0
    final_status = "failed"
    reason: str | None = None
    error: str | None = None
    payload: object | None = None
    validated_raw: bytes | None = None
    heartbeat_age: float | None = None
    untracked_live_process = False

    try:
        analysis_attempted = True
        _add_worktree(
            repository,
            analysis_worktree,
            revision,
            deadline_monotonic=deadline_monotonic,
        )
        analysis_registered = True
        process = _launch_python(
            _ANALYSIS_CHILD,
            (analysis_worktree, private_result, private_heartbeat, private_outcome),
            worktree=analysis_worktree,
            stderr_path=analysis_stderr,
        )
        monitor_status, sequence, heartbeat_age = _monitor_process(
            process,
            private_heartbeat,
            progress,
            revision=revision,
            started_monotonic=started_monotonic,
            wall_clock_seconds=wall_clock,
            stale_after_seconds=stale_after,
            poll_seconds=poll,
        )
        if monitor_status in {"wall_clock_timeout", "stalled"}:
            final_status = monitor_status
            reason = monitor_status
            _terminate_process_group(process, grace_seconds=term_grace)
        else:
            returncode = process.wait()
            _drain_process_group(process.pid, grace_seconds=term_grace)
            try:
                outcome_status, payload, error = _parse_outcome(
                    private_outcome.read_bytes(), returncode
                )
            except (OSError, WatchdogError) as exc:
                final_status = "failed"
                error = f"invalid child outcome: {exc}"
                stderr_detail = _bounded_stderr(analysis_stderr)
                if stderr_detail is not None:
                    error += f"; child stderr: {stderr_detail}"
            else:
                if outcome_status == "nonconvergence":
                    final_status = "nonconvergence"
                    reason = "analysis_nonconvergence"
                elif outcome_status == "failed":
                    final_status = "failed"
                    reason = "analysis_failed"
                else:
                    try:
                        candidate_raw = private_result.read_bytes()
                        _load_canonical_json(candidate_raw)
                    except OSError as exc:
                        final_status = "failed"
                        error = f"private result missing or unreadable: {exc}"
                    except WatchdogError as exc:
                        final_status = "failed"
                        error = f"private result is not canonical JSON: {exc}"
                    else:
                        _remove_worktree(
                            repository,
                            analysis_worktree,
                            deadline_monotonic=deadline_monotonic,
                        )
                        analysis_registered = False
                        analysis_attempted = False
                        validation_attempted = True
                        _add_worktree(
                            repository,
                            validation_worktree,
                            revision,
                            deadline_monotonic=deadline_monotonic,
                        )
                        validation_registered = True
                        validator_process = _launch_python(
                            _VALIDATION_CHILD,
                            (
                                validation_worktree,
                                private_result,
                                validation_outcome,
                            ),
                            worktree=validation_worktree,
                            stderr_path=validation_stderr,
                        )
                        validator_returncode = _wait_until_deadline(
                            validator_process,
                            started_monotonic=started_monotonic,
                            wall_clock_seconds=wall_clock,
                            grace_seconds=term_grace,
                        )
                        _drain_process_group(
                            validator_process.pid, grace_seconds=term_grace
                        )
                        if validator_returncode is None:
                            final_status = "wall_clock_timeout"
                            reason = "wall_clock_timeout"
                            error = "independent result validation exceeded deadline"
                        elif validator_returncode != 0:
                            final_status = "failed"
                            error = "independent result validation failed"
                            stderr_detail = _bounded_stderr(validation_stderr)
                            if stderr_detail is not None:
                                error += f"; validator stderr: {stderr_detail}"
                        else:
                            try:
                                validation_receipt = _load_canonical_json(
                                    validation_outcome.read_bytes()
                                )
                                validated_raw = private_result.read_bytes()
                            except (OSError, WatchdogError) as exc:
                                final_status = "failed"
                                error = f"validation receipt or result is invalid: {exc}"
                            else:
                                expected_receipt = {
                                    "sha256": hashlib.sha256(candidate_raw).hexdigest()
                                }
                                if validation_receipt != expected_receipt:
                                    validated_raw = None
                                    final_status = "failed"
                                    error = "validator receipt does not match candidate bytes"
                                elif validated_raw != candidate_raw:
                                    validated_raw = None
                                    final_status = "failed"
                                    error = "private result changed during validation"
                                elif time.monotonic() >= deadline_monotonic:
                                    validated_raw = None
                                    final_status = "wall_clock_timeout"
                                    reason = "wall_clock_timeout"
                                    error = "validation completed after hard deadline"
                                else:
                                    final_status = "completed"
                                    reason = "validated"
    except BaseException as exc:
        error = str(exc)
        if isinstance(exc, WatchdogLiveProcessError):
            untracked_live_process = True
        if isinstance(exc, WatchdogDeadlineError):
            final_status = "wall_clock_timeout"
            reason = "wall_clock_timeout"
        else:
            final_status = "failed"
            reason = "watchdog_failure"
    finally:
        primary_error = error
        cleanup_errors: list[str] = []
        live_process = untracked_live_process
        if untracked_live_process:
            cleanup_errors.append("an untracked Git process group may remain alive")
        for label, active in (
            ("validator", validator_process),
            ("analysis", process),
        ):
            if active is None:
                continue
            if active.poll() is None:
                try:
                    _terminate_process_group(active, grace_seconds=term_grace)
                except BaseException as exc:
                    cleanup_errors.append(f"{label} termination failed: {exc}")
            try:
                _drain_process_group(active.pid, grace_seconds=term_grace)
            except BaseException as exc:
                cleanup_errors.append(f"{label} process-group drain failed: {exc}")
            if active.poll() is None or _process_group_exists(active.pid):
                live_process = True
                cleanup_errors.append(f"{label} process remains alive")
        cleanup_deadline = time.monotonic() + _GIT_TIMEOUT_SECONDS
        if not live_process:
            worktree_errors, cleanup_found_live_process = _cleanup_worktrees(
                repository,
                (
                    (
                        "validation",
                        validation_attempted,
                        validation_registered,
                        validation_worktree,
                    ),
                    (
                        "analysis",
                        analysis_attempted,
                        analysis_registered,
                        analysis_worktree,
                    ),
                ),
                deadline_monotonic=cleanup_deadline,
            )
            cleanup_errors.extend(worktree_errors)
            live_process = cleanup_found_live_process
        if not cleanup_errors and not live_process:
            try:
                _cleanup_directory(temporary_root)
            except BaseException as exc:
                cleanup_errors.append(
                    f"temporary cleanup failed, retained at {temporary_root}: {exc}"
                )
        else:
            cleanup_errors.append(f"temporary state retained at {temporary_root}")
        if cleanup_errors:
            final_status = "failed"
            validated_raw = None
            reason = "cleanup_failed"
            details = cleanup_errors
            if primary_error:
                details = [primary_error, *details]
            error = "; ".join(details)

    if final_status == "completed" and validated_raw is not None:
        try:
            _exclusive_write(output, validated_raw)
        except BaseException as exc:
            final_status = "failed"
            reason = "publish_failed"
            error = str(exc)
    elif final_status == "completed":
        final_status = "failed"
        reason = "validation_failed"
        error = "completed analysis has no validated bytes"

    if final_status not in _TERMINAL_STATUSES:
        final_status = "failed"
        reason = "invalid_terminal_status"
        error = "watchdog produced an invalid terminal status"
    sequence += 1
    finished_monotonic = time.monotonic()
    terminal = _progress_document(
        revision,
        final_status,
        sequence,
        started_monotonic=started_monotonic,
        now_monotonic=finished_monotonic,
        pid=process.pid if process is not None else None,
        heartbeat_age_seconds=heartbeat_age,
        reason=reason,
        error=error,
        payload=payload,
    )
    _replace_terminal_progress(progress, _canonical_json_bytes(terminal))
    return terminal


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repository", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("progress_path", type=Path)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--wall-clock-seconds", type=float, default=1800)
    parser.add_argument("--stale-after-seconds", type=float, default=60)
    parser.add_argument("--term-grace-seconds", type=float, default=10)
    parser.add_argument("--poll-seconds", type=float, default=1)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        result = run_analysis(
            arguments.repository,
            arguments.output_path,
            arguments.progress_path,
            revision=arguments.revision,
            wall_clock_seconds=arguments.wall_clock_seconds,
            stale_after_seconds=arguments.stale_after_seconds,
            term_grace_seconds=arguments.term_grace_seconds,
            poll_seconds=arguments.poll_seconds,
        )
    except (OSError, ValueError, WatchdogError) as exc:
        print(f"runtime access watchdog: {exc}", file=sys.stderr)
        return 2
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
