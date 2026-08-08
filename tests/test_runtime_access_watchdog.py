from __future__ import annotations

import inspect
import json
import os
import shutil
import signal
import subprocess
from pathlib import Path
from typing import Any

import pytest

from scripts import runtime_access_watchdog as watchdog
from scripts.runtime_ownership.manifests import (
    ANALYZER_PATHS,
    MACHINE_FACT_TOOLCHAIN_PATHS,
    _is_direct_analyzer_path,
    _source_category,
)


def _run_git(repository: Path, *arguments: str) -> bytes:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repository,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr.decode(
        "utf-8", errors="replace"
    )
    return completed.stdout


def _repository(tmp_path: Path) -> tuple[Path, str]:
    repository = tmp_path / "repository"
    repository.mkdir()
    _run_git(repository, "init", "--quiet", "--object-format=sha1")
    _run_git(repository, "config", "user.name", "Watchdog Test")
    _run_git(repository, "config", "user.email", "watchdog@example.test")
    (repository / "tracked.txt").write_text("first\n")
    _run_git(repository, "add", "tracked.txt")
    _run_git(repository, "commit", "--quiet", "-m", "first")
    revision = _run_git(repository, "rev-parse", "HEAD").decode("ascii").strip()
    return repository, revision


class FakeProcess:
    def __init__(
        self,
        *,
        pid: int = 4321,
        returncode: int | None = 0,
        polls: list[int | None] | None = None,
        waits: list[object] | None = None,
    ) -> None:
        self.pid = pid
        self.returncode = returncode
        self._polls = list(polls or [])
        self._waits = list(waits or [])
        self.wait_timeouts: list[float | None] = []

    def poll(self) -> int | None:
        if self._polls:
            result = self._polls.pop(0)
            if result is not None:
                self.returncode = result
            return result
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        self.wait_timeouts.append(timeout)
        if self._waits:
            result = self._waits.pop(0)
            if isinstance(result, BaseException):
                raise result
            self.returncode = int(result)
        if self.returncode is None:
            self.returncode = 0
        return self.returncode


class Clock:
    def __init__(self, value: float = 0.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value

    def sleep(self, seconds: float) -> None:
        self.value += seconds


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def test_revision_requires_exact_lowercase_existing_full_commit(
    tmp_path: Path,
) -> None:
    repository, revision = _repository(tmp_path)
    assert watchdog._resolve_revision(repository, revision) == revision
    for invalid in ("HEAD", revision[:12], revision.upper(), "0" * 40, True):
        if invalid == "0" * 40:
            with pytest.raises(watchdog.WatchdogError, match="git rev-parse"):
                watchdog._resolve_revision(repository, invalid)
        else:
            with pytest.raises(ValueError, match="lowercase full 40-character"):
                watchdog._resolve_revision(repository, invalid)  # type: ignore[arg-type]


def test_git_ignores_replace_objects_and_environment_contamination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, original = _repository(tmp_path)
    (repository / "tracked.txt").write_text("replacement\n")
    _run_git(repository, "commit", "--quiet", "-am", "replacement")
    replacement = _run_git(repository, "rev-parse", "HEAD").decode().strip()
    _run_git(repository, "replace", original, replacement)

    monkeypatch.setenv("GIT_DIR", os.fspath(tmp_path / "poison-git-dir"))
    monkeypatch.setenv("GIT_REPLACE_REF_BASE", "refs/poison/")
    monkeypatch.setenv("PYTHONPATH", os.fspath(tmp_path / "poison-python"))
    monkeypatch.setenv("PYTHONHOME", os.fspath(tmp_path / "poison-home"))
    monkeypatch.setenv("VIRTUAL_ENV", os.fspath(tmp_path / "poison-venv"))

    assert watchdog._resolve_revision(repository, original) == original
    assert watchdog._git(repository, "show", f"{original}:tracked.txt").stdout == b"first\n"
    clean = watchdog._clean_environment(pythonpath=repository)
    assert {key for key in clean if key.startswith("GIT_")} == {
        "GIT_CONFIG_GLOBAL",
        "GIT_CONFIG_NOSYSTEM",
        "GIT_NO_REPLACE_OBJECTS",
        "GIT_TERMINAL_PROMPT",
    }
    assert clean["PYTHONPATH"] == os.fspath(repository)
    assert "PYTHONHOME" not in clean
    assert "VIRTUAL_ENV" not in clean


def test_git_expired_deadline_never_starts_a_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, _revision = _repository(tmp_path)
    started: list[bool] = []
    monkeypatch.setattr(watchdog.time, "monotonic", lambda: 10.0)
    monkeypatch.setattr(
        watchdog.subprocess,
        "Popen",
        lambda *_args, **_kwargs: started.append(True),
    )
    with pytest.raises(watchdog.WatchdogDeadlineError):
        watchdog._git(repository, "status", deadline_monotonic=10.0)
    assert started == []


def test_git_drains_process_group_after_normal_leader_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, _revision = _repository(tmp_path)
    drained: list[tuple[int, float]] = []
    monkeypatch.setattr(
        watchdog,
        "_drain_process_group",
        lambda pid, *, grace_seconds: drained.append((pid, grace_seconds)),
    )
    completed = watchdog._git(repository, "status", "--porcelain")
    assert completed.returncode == 0
    assert len(drained) == 1
    assert drained[0][1] == 1.0


def test_git_cleanup_uses_direct_kill_and_reports_combined_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timeout = subprocess.TimeoutExpired(["git"], 1)

    class BrokenGitProcess:
        pid = 987654

        def poll(self) -> None:
            return None

        def communicate(self, timeout: float) -> tuple[bytes, bytes]:
            del timeout
            raise timeout_error

        def kill(self) -> None:
            raise PermissionError("direct denied")

        def wait(self, timeout: float) -> int:
            del timeout
            raise PermissionError("wait denied")

    timeout_error = timeout
    monkeypatch.setattr(
        watchdog.os,
        "killpg",
        lambda *_args: (_ for _ in ()).throw(PermissionError("group denied")),
    )
    with pytest.raises(watchdog.WatchdogError, match="direct kill failed"):
        watchdog._stop_git_process(BrokenGitProcess())  # type: ignore[arg-type]


def test_runner_is_bound_to_repository_head_and_exact_committed_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, _revision = _repository(tmp_path)
    runner = repository / "scripts/runtime_access_watchdog.py"
    runner.parent.mkdir()
    runner.write_bytes(Path(watchdog.__file__).read_bytes())
    _run_git(repository, "add", watchdog._RUNNER_PATH)
    _run_git(repository, "commit", "--quiet", "-m", "runner")
    revision = _run_git(repository, "rev-parse", "HEAD").decode().strip()
    monkeypatch.setattr(watchdog, "__file__", os.fspath(runner))

    watchdog._verify_runner_at_revision(repository, revision)
    runner.write_text("tampered\n")
    with pytest.raises(watchdog.WatchdogError, match="differs"):
        watchdog._verify_runner_at_revision(repository, revision)
    runner.write_bytes(
        _run_git(repository, "show", f"{revision}:{watchdog._RUNNER_PATH}")
    )
    (repository / "tracked.txt").write_text("third\n")
    _run_git(repository, "commit", "--quiet", "-am", "third")
    with pytest.raises(watchdog.WatchdogError, match="runner commit"):
        watchdog._verify_runner_at_revision(repository, revision)

    runner.unlink()
    runner.symlink_to(repository / "tracked.txt")
    current = _run_git(repository, "rev-parse", "HEAD").decode().strip()
    with pytest.raises(watchdog.WatchdogError, match="must not be a symlink"):
        watchdog._verify_runner_at_revision(repository, current)


def test_public_api_has_no_runner_verification_bypass() -> None:
    parameters = inspect.signature(watchdog.run_analysis).parameters
    assert "_enforce_runner_revision" not in parameters


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("wall_clock_seconds", 0),
        ("wall_clock_seconds", 43_201),
        ("wall_clock_seconds", True),
        ("stale_after_seconds", 0),
        ("term_grace_seconds", 61),
        ("poll_seconds", 61),
    ],
)
def test_operational_bounds_are_exact_and_finite(
    tmp_path: Path,
    name: str,
    value: object,
) -> None:
    arguments: dict[str, object] = {name: value}
    with pytest.raises(ValueError):
        watchdog.run_analysis(
            tmp_path,
            tmp_path / "output",
            tmp_path / "progress",
            revision="0" * 40,
            **arguments,  # type: ignore[arg-type]
        )


def test_child_uses_explicit_argv_clean_worktree_and_fresh_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    stderr = tmp_path / "child.stderr"
    captured: dict[str, Any] = {}

    def popen(args: list[str], **kwargs: Any) -> FakeProcess:
        captured["args"] = args
        captured.update(kwargs)
        return FakeProcess()

    monkeypatch.setenv("GIT_INDEX_FILE", "poison")
    monkeypatch.setenv("PYTHONPATH", "poison")
    process = watchdog._launch_python(
        "pass",
        (worktree, tmp_path / "result"),
        worktree=worktree,
        stderr_path=stderr,
        popen=popen,
    )
    assert process.pid == 4321
    assert captured["args"][:4] == [watchdog.sys.executable, "-I", "-c", "pass"]
    assert captured["args"][4:] == [os.fspath(worktree), os.fspath(tmp_path / "result")]
    assert captured["cwd"] == worktree
    assert captured["shell"] is False
    assert captured["start_new_session"] is True
    assert captured["stdin"] == subprocess.DEVNULL
    assert captured["stdout"] == subprocess.DEVNULL
    assert isinstance(captured["stderr"], int)
    environment = captured["env"]
    assert environment["PYTHONPATH"] == os.fspath(worktree)
    assert environment["GIT_NO_REPLACE_OBJECTS"] == "1"
    assert environment["GIT_CONFIG_NOSYSTEM"] == "1"
    assert environment["GIT_CONFIG_GLOBAL"] == os.devnull
    assert environment["GIT_TERMINAL_PROMPT"] == "0"
    assert stderr.exists()


def test_child_heartbeat_precedes_imports_and_nonconvergence_match_is_exact() -> None:
    first_line = watchdog._ANALYSIS_CHILD.lstrip().splitlines()[0]
    assert first_line.startswith("open(")
    assert "scripts.runtime_ownership.access_model" in watchdog._ANALYSIS_CHILD
    assert "AnalysisNonConvergenceError" in watchdog._ANALYSIS_CHILD
    assert "from scripts.runtime_ownership import machine_facts" in watchdog._ANALYSIS_CHILD


def test_monitor_deadline_wins_over_exit_and_stale(tmp_path: Path) -> None:
    process = FakeProcess(returncode=0)
    clock = Clock(10)
    status, sequence, age = watchdog._monitor_process(
        process,
        tmp_path / "missing-heartbeat",
        tmp_path / "unused-progress",
        revision="1" * 40,
        started_monotonic=0,
        wall_clock_seconds=10,
        stale_after_seconds=1,
        poll_seconds=1,
        monotonic=clock,
        sleep=clock.sleep,
        replace_progress=lambda _path, _raw: None,
    )
    assert (status, sequence, age) == ("wall_clock_timeout", 0, 10)


def test_monitor_exit_wins_over_stale(tmp_path: Path) -> None:
    process = FakeProcess(returncode=None, polls=[None, 7])
    clock = Clock()

    def sleep(_seconds: float) -> None:
        clock.value += 6

    status, sequence, age = watchdog._monitor_process(
        process,
        tmp_path / "missing-heartbeat",
        tmp_path / "progress",
        revision="2" * 40,
        started_monotonic=0,
        wall_clock_seconds=100,
        stale_after_seconds=5,
        poll_seconds=1,
        monotonic=clock,
        sleep=sleep,
        replace_progress=lambda _path, _raw: None,
    )
    assert status == "exited"
    assert sequence == 1
    assert age == 6


def test_monitor_uses_parent_observed_monotonic_heartbeat_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fingerprints = iter([(1, 999, 1), (1, 999, 1), (2, 1, 1), (2, 1, 1)])
    monkeypatch.setattr(watchdog, "_heartbeat_fingerprint", lambda _path: next(fingerprints))
    process = FakeProcess(returncode=None, polls=[None, None, None, 0])
    clock = Clock()
    status, _sequence, age = watchdog._monitor_process(
        process,
        tmp_path / "heartbeat",
        tmp_path / "progress",
        revision="3" * 40,
        started_monotonic=0,
        wall_clock_seconds=100,
        stale_after_seconds=5,
        poll_seconds=3,
        monotonic=clock,
        sleep=clock.sleep,
        replace_progress=lambda _path, _raw: None,
    )
    assert status == "exited"
    assert age == 3


def test_monitor_detects_stalled_child(tmp_path: Path) -> None:
    process = FakeProcess(returncode=None, polls=[None, None, None])
    clock = Clock()
    status, sequence, age = watchdog._monitor_process(
        process,
        tmp_path / "missing-heartbeat",
        tmp_path / "progress",
        revision="4" * 40,
        started_monotonic=0,
        wall_clock_seconds=100,
        stale_after_seconds=5,
        poll_seconds=3,
        monotonic=clock,
        sleep=clock.sleep,
        replace_progress=lambda _path, _raw: None,
    )
    assert (status, sequence, age) == ("stalled", 2, 5)


def test_monitor_clips_poll_sleep_to_next_hard_boundary(tmp_path: Path) -> None:
    process = FakeProcess(returncode=None, polls=[None, None])
    clock = Clock()
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        clock.value += seconds

    status, _sequence, _age = watchdog._monitor_process(
        process,
        tmp_path / "heartbeat",
        tmp_path / "progress",
        revision="5" * 40,
        started_monotonic=0,
        wall_clock_seconds=2,
        stale_after_seconds=50,
        poll_seconds=60,
        monotonic=clock,
        sleep=sleep,
        replace_progress=lambda _path, _raw: None,
    )
    assert status == "wall_clock_timeout"
    assert sleeps == [2]


def test_termination_escalates_term_to_kill_and_reaps() -> None:
    timeout = subprocess.TimeoutExpired(["child"], 2)
    process = FakeProcess(returncode=None, waits=[timeout, -signal.SIGKILL])
    signals: list[tuple[int, int]] = []
    watchdog._terminate_process_group(
        process,
        grace_seconds=2,
        kill_group=lambda pid, sig: signals.append((pid, sig)),
    )
    assert signals == [(process.pid, signal.SIGTERM), (process.pid, signal.SIGKILL)]
    assert process.wait_timeouts == [2, watchdog._KILL_REAP_SECONDS]


def test_termination_does_not_kill_child_that_exits_during_grace() -> None:
    process = FakeProcess(returncode=None, waits=[-signal.SIGTERM])
    signals: list[tuple[int, int]] = []
    watchdog._terminate_process_group(
        process,
        grace_seconds=2,
        kill_group=lambda pid, sig: signals.append((pid, sig)),
    )
    assert signals == [(process.pid, signal.SIGTERM)]
    assert process.wait_timeouts == [2]


def test_validator_wait_obeys_remaining_deadline_and_terminates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timeout = subprocess.TimeoutExpired(["validator"], 2)
    process = FakeProcess(returncode=None, waits=[timeout])
    terminated: list[tuple[FakeProcess, float]] = []
    monkeypatch.setattr(
        watchdog,
        "_terminate_process_group",
        lambda child, *, grace_seconds: terminated.append((child, grace_seconds)),
    )
    assert (
        watchdog._wait_until_deadline(
            process,
            started_monotonic=10,
            wall_clock_seconds=2,
            grace_seconds=1,
            monotonic=lambda: 10,
        )
        is None
    )
    assert process.wait_timeouts == [2]
    assert terminated == [(process, 1)]


def test_outcome_classification_is_exact() -> None:
    payload = {"phase": "cfg", "limit": 2}
    assert watchdog._parse_outcome(_canonical({"status": "success"}), 0) == (
        "success",
        None,
        None,
    )
    assert watchdog._parse_outcome(
        _canonical({"status": "nonconvergence", "payload": payload}), 20
    ) == ("nonconvergence", payload, None)
    assert watchdog._parse_outcome(
        _canonical({"status": "failed", "error": "builtins.ValueError: bad"}), 21
    ) == ("failed", None, "builtins.ValueError: bad")
    for raw, returncode in [
        (_canonical({"status": "success"}), 1),
        (_canonical({"status": "nonconvergence", "payload": payload}), 21),
        (b"{", 20),
        (b'{"status":"failed","status":"success"}', 21),
    ]:
        with pytest.raises(watchdog.WatchdogError):
            watchdog._parse_outcome(raw, returncode)


@pytest.mark.parametrize(
    "raw",
    [
        b"{",
        b'{"value":NaN}',
        b'{"value":1,"value":2}',
        b'{ "value": 1 }',
    ],
)
def test_canonical_loader_rejects_truncated_nan_duplicate_and_noncanonical(
    raw: bytes,
) -> None:
    with pytest.raises(watchdog.WatchdogError):
        watchdog._load_canonical_json(raw)


def test_exclusive_output_rejects_existing_regular_symlink_and_dangling(
    tmp_path: Path,
) -> None:
    victim = tmp_path / "victim"
    victim.write_text("victim")
    regular = tmp_path / "regular"
    regular.write_text("old")
    symlink = tmp_path / "symlink"
    symlink.symlink_to(victim)
    dangling = tmp_path / "dangling"
    dangling.symlink_to(tmp_path / "absent")
    for path in (regular, symlink, dangling):
        with pytest.raises(FileExistsError):
            watchdog._exclusive_write(path, b"new")
    assert victim.read_text() == "victim"
    assert regular.read_text() == "old"


def test_symlink_parent_is_rejected(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    with pytest.raises(watchdog.WatchdogError, match="symlink parent"):
        watchdog._exclusive_write(alias / "result.json", b"value")


def test_post_validation_symlink_swap_cannot_modify_victim(tmp_path: Path) -> None:
    victim = tmp_path / "victim"
    victim.write_bytes(b"victim")
    output = tmp_path / "result"
    assert not watchdog._path_exists_even_if_dangling(output)
    output.symlink_to(victim)
    with pytest.raises(FileExistsError):
        watchdog._exclusive_write(output, b"validated")
    assert victim.read_bytes() == b"victim"


def test_atomic_progress_failure_preserves_prior_json_and_cleans_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    progress = tmp_path / "progress.json"
    prior = _canonical({"sequence": 1, "status": "running"})
    watchdog._exclusive_write(progress, prior)

    def fail_replace(*_args: object, **_kwargs: object) -> None:
        raise OSError("injected replace failure")

    monkeypatch.setattr(watchdog.os, "replace", fail_replace)
    with pytest.raises(OSError, match="injected"):
        watchdog._atomic_replace(progress, _canonical({"sequence": 2}))
    assert progress.read_bytes() == prior
    assert list(tmp_path.glob(".progress.json.tmp-*")) == []


def test_publication_write_failure_never_exposes_partial_final(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "output.json"

    def partial_then_fail(descriptor: int, _raw: bytes) -> None:
        os.write(descriptor, b"p")
        raise OSError("injected short publication")

    monkeypatch.setattr(watchdog, "_write_all", partial_then_fail)
    with pytest.raises(OSError, match="injected"):
        watchdog._exclusive_write(output, b"published")
    assert not watchdog._path_exists_even_if_dangling(output)
    assert list(tmp_path.glob(".output.json.publish-*")) == []


def test_chmod_setup_failure_cleans_private_temp_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    temporary = tmp_path / "allocated"

    def allocate(*_args: object, **_kwargs: object) -> str:
        temporary.mkdir()
        return os.fspath(temporary)

    monkeypatch.setattr(watchdog.tempfile, "mkdtemp", allocate)
    monkeypatch.setattr(
        watchdog.os,
        "chmod",
        lambda *_args: (_ for _ in ()).throw(OSError("chmod failed")),
    )
    with pytest.raises(OSError, match="chmod failed"):
        watchdog._create_temporary_root()
    assert not temporary.exists()


def test_chmod_and_cleanup_failure_reports_retained_private_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    temporary = tmp_path / "retained-setup"

    def allocate(*_args: object, **_kwargs: object) -> str:
        temporary.mkdir()
        return os.fspath(temporary)

    monkeypatch.setattr(watchdog.tempfile, "mkdtemp", allocate)
    monkeypatch.setattr(
        watchdog.os,
        "chmod",
        lambda *_args: (_ for _ in ()).throw(OSError("chmod failed")),
    )
    monkeypatch.setattr(
        watchdog,
        "_cleanup_directory",
        lambda _path: (_ for _ in ()).throw(OSError("cleanup failed")),
    )
    with pytest.raises(watchdog.WatchdogError) as raised:
        watchdog._create_temporary_root()
    assert "chmod failed" in str(raised.value)
    assert "cleanup failed" in str(raised.value)
    assert os.fspath(temporary) in str(raised.value)
    assert temporary.exists()
    shutil.rmtree(temporary)


def test_terminal_progress_update_retries_once_and_stays_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    progress = tmp_path / "progress.json"
    watchdog._exclusive_write(progress, _canonical({"status": "running"}))
    replacement = _canonical({"status": "failed"})
    original = watchdog._atomic_replace
    attempts = 0

    def flaky(path: Path, raw: bytes) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("transient")
        original(path, raw)

    monkeypatch.setattr(watchdog, "_atomic_replace", flaky)
    watchdog._replace_terminal_progress(progress, replacement)
    assert attempts == 2
    assert progress.read_bytes() == replacement


def test_worktree_is_detached_and_cleaned(tmp_path: Path) -> None:
    repository, revision = _repository(tmp_path)
    worktree = tmp_path / "detached"
    watchdog._add_worktree(repository, worktree, revision)
    assert watchdog._worktree_registered(repository, worktree)
    assert _run_git(worktree, "rev-parse", "HEAD").decode().strip() == revision
    symbolic = subprocess.run(
        ["git", "symbolic-ref", "-q", "HEAD"],
        cwd=worktree,
        capture_output=True,
        check=False,
    )
    assert symbolic.returncode == 1
    watchdog._remove_worktree(repository, worktree)
    assert not worktree.exists()
    assert not watchdog._worktree_registered(repository, worktree)
    assert os.fspath(worktree).encode() not in _run_git(repository, "worktree", "list", "--porcelain")


def _patch_run_harness(
    monkeypatch: pytest.MonkeyPatch,
    *,
    result_raw: bytes | None = b'{"result":"validated"}',
    outcome: dict[str, object] | bytes | None = None,
    analysis_returncode: int = 0,
    validator_returncode: int = 0,
    mutate_during_validation: bytes | None = None,
    cleanup_failure: bool = False,
    validation_digest: str | None = None,
    events: list[str] | None = None,
) -> list[Path]:
    retained: list[Path] = []
    monkeypatch.setattr(
        watchdog,
        "_resolve_revision",
        lambda _repository, revision, **_kwargs: revision,
    )
    monkeypatch.setattr(watchdog, "_verify_runner_at_revision", lambda *_args, **_kwargs: None)

    def add_worktree(
        _repository: Path, path: Path, _revision: str, **_kwargs: object
    ) -> None:
        if events is not None:
            events.append(f"add:{path.name}")
        path.mkdir()

    def remove_worktree(
        _repository: Path, path: Path, **_kwargs: object
    ) -> None:
        if events is not None:
            events.append(f"remove:{path.name}")
        if cleanup_failure:
            retained.append(path.parent)
            raise watchdog.WatchdogError("injected cleanup failure")
        shutil.rmtree(path)

    launches = 0

    def launch(
        _code: str,
        arguments: tuple[Path, ...],
        *,
        worktree: Path,
        stderr_path: Path,
    ) -> FakeProcess:
        nonlocal launches
        del worktree
        stderr_path.write_bytes(b"")
        launches += 1
        if launches == 1:
            _clean_repo, private_result, heartbeat, private_outcome = arguments
            heartbeat.write_bytes(b"0")
            if result_raw is not None:
                private_result.write_bytes(result_raw)
            selected_outcome: dict[str, object] | bytes = (
                {"status": "success"} if outcome is None else outcome
            )
            raw_outcome = (
                selected_outcome
                if isinstance(selected_outcome, bytes)
                else _canonical(selected_outcome)
            )
            private_outcome.write_bytes(raw_outcome)
            return FakeProcess(returncode=analysis_returncode)
        if mutate_during_validation is not None:
            arguments[1].write_bytes(mutate_during_validation)
        arguments[2].write_bytes(
            _canonical(
                {
                    "sha256": validation_digest
                    or watchdog.hashlib.sha256(arguments[1].read_bytes()).hexdigest()
                }
            )
        )
        return FakeProcess(returncode=validator_returncode)

    monkeypatch.setattr(watchdog, "_add_worktree", add_worktree)
    monkeypatch.setattr(watchdog, "_remove_worktree", remove_worktree)
    monkeypatch.setattr(watchdog, "_launch_python", launch)
    monkeypatch.setattr(watchdog, "_drain_process_group", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(watchdog, "_process_group_exists", lambda _pid: False)
    monkeypatch.setattr(
        watchdog,
        "_monitor_process",
        lambda *_args, **_kwargs: ("exited", 2, 0.0),
    )
    return retained


def test_success_is_validated_and_published_once_after_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_run_harness(monkeypatch)
    output = tmp_path / "output.json"
    progress = tmp_path / "progress.json"
    result = watchdog.run_analysis(
        tmp_path,
        output,
        progress,
        revision="a" * 40,
    )
    assert result["status"] == "completed"
    assert output.read_bytes() == b'{"result":"validated"}'
    assert json.loads(progress.read_bytes())["status"] == "completed"


def test_validator_digest_mismatch_is_not_published(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_run_harness(monkeypatch, validation_digest="0" * 64)
    output = tmp_path / "output.json"
    result = watchdog.run_analysis(
        tmp_path,
        output,
        tmp_path / "progress.json",
        revision="9" * 40,
    )
    assert result["status"] == "failed"
    assert result["error"] == "validator receipt does not match candidate bytes"
    assert not output.exists()


def test_validation_before_deadline_allows_bounded_cleanup_slack(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_run_harness(monkeypatch)
    monkeypatch.setattr(
        watchdog,
        "_wait_until_deadline",
        lambda process, **_kwargs: process.wait(),
    )
    times = iter([0.0, 9.0, 11.0, 11.0])
    monkeypatch.setattr(watchdog.time, "monotonic", lambda: next(times))
    output = tmp_path / "output.json"
    result = watchdog.run_analysis(
        tmp_path,
        output,
        tmp_path / "progress.json",
        revision="7" * 40,
        wall_clock_seconds=10,
    )
    assert result["status"] == "completed"
    assert output.exists()


def test_validation_receipt_after_deadline_is_timeout_without_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_run_harness(monkeypatch)
    monkeypatch.setattr(
        watchdog,
        "_wait_until_deadline",
        lambda process, **_kwargs: process.wait(),
    )
    times = iter([0.0, 10.0, 10.0, 10.0])
    monkeypatch.setattr(watchdog.time, "monotonic", lambda: next(times))
    output = tmp_path / "output.json"
    result = watchdog.run_analysis(
        tmp_path,
        output,
        tmp_path / "progress.json",
        revision="8" * 40,
        wall_clock_seconds=10,
    )
    assert result["status"] == "wall_clock_timeout"
    assert not output.exists()


def test_exited_leader_with_live_descendant_retains_worktree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    _patch_run_harness(monkeypatch, events=events)
    monkeypatch.setattr(watchdog, "_process_group_exists", lambda _pid: True)
    monkeypatch.setattr(
        watchdog,
        "_drain_process_group",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            watchdog.WatchdogError("descendant remains")
        ),
    )
    output = tmp_path / "output.json"
    result = watchdog.run_analysis(
        tmp_path,
        output,
        tmp_path / "progress.json",
        revision="6" * 40,
    )
    assert result["status"] == "failed"
    assert result["reason"] == "cleanup_failed"
    assert "process remains alive" in str(result["error"])
    assert not any(event.startswith("remove:") for event in events)
    assert not output.exists()
    retained_text = str(result["error"]).rsplit("temporary state retained at ", 1)[1]
    retained = Path(retained_text)
    assert retained.exists()
    shutil.rmtree(retained)


@pytest.mark.parametrize(
    "result_raw",
    [
        None,
        b"{",
        b'{"value":NaN}',
        b'{"value":1,"value":2}',
        b'{ "value": 1 }',
    ],
)
def test_missing_or_invalid_private_result_is_not_published(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    result_raw: bytes | None,
) -> None:
    _patch_run_harness(monkeypatch, result_raw=result_raw)
    output = tmp_path / "output.json"
    result = watchdog.run_analysis(
        tmp_path,
        output,
        tmp_path / "progress.json",
        revision="b" * 40,
    )
    assert result["status"] == "failed"
    assert not output.exists()


def test_semantically_tampered_result_rejected_by_independent_validator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_run_harness(monkeypatch, validator_returncode=1)
    output = tmp_path / "output.json"
    result = watchdog.run_analysis(
        tmp_path,
        output,
        tmp_path / "progress.json",
        revision="c" * 40,
    )
    assert result["status"] == "failed"
    assert result["error"] == "independent result validation failed"
    assert not output.exists()


def test_result_change_after_validation_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_run_harness(
        monkeypatch, mutate_during_validation=b'{"result":"swapped"}'
    )
    output = tmp_path / "output.json"
    result = watchdog.run_analysis(
        tmp_path,
        output,
        tmp_path / "progress.json",
        revision="d" * 40,
    )
    assert result["status"] == "failed"
    assert result["error"] == "validator receipt does not match candidate bytes"
    assert not output.exists()


def test_nonconvergence_generic_and_malformed_child_exit_are_distinct(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cases = [
        (
            {"status": "nonconvergence", "payload": {"phase": "cfg"}},
            20,
            "nonconvergence",
        ),
        ({"status": "failed", "error": "builtins.ValueError: bad"}, 21, "failed"),
        (b"{", 20, "failed"),
    ]
    for index, (outcome, returncode, expected) in enumerate(cases):
        case = tmp_path / str(index)
        case.mkdir()
        with monkeypatch.context() as scoped:
            _patch_run_harness(
                scoped,
                result_raw=None,
                outcome=outcome,
                analysis_returncode=returncode,
            )
            result = watchdog.run_analysis(
                case,
                case / "output.json",
                case / "progress.json",
                revision="e" * 40,
            )
        assert result["status"] == expected
        assert not (case / "output.json").exists()


def test_existing_output_progress_and_dangling_paths_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_run_harness(monkeypatch)
    for name, dangling in [("output", False), ("progress", False), ("dangling", True)]:
        case = tmp_path / name
        case.mkdir()
        output = case / "output.json"
        progress = case / "progress.json"
        target = output if name != "progress" else progress
        if dangling:
            target.symlink_to(case / "missing")
        else:
            target.write_text("existing")
        with pytest.raises(FileExistsError):
            watchdog.run_analysis(
                case,
                output,
                progress,
                revision="f" * 40,
            )


def test_cleanup_failure_retains_temp_state_and_prevents_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    retained = _patch_run_harness(monkeypatch, cleanup_failure=True)
    output = tmp_path / "output.json"
    result = watchdog.run_analysis(
        tmp_path,
        output,
        tmp_path / "progress.json",
        revision="1" * 40,
    )
    assert result["status"] == "failed"
    assert result["reason"] == "cleanup_failed"
    assert not output.exists()
    assert retained and retained[0].exists()
    shutil.rmtree(retained[0])


def test_private_temp_setup_failure_never_creates_running_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_run_harness(monkeypatch)
    monkeypatch.setattr(
        watchdog,
        "_create_temporary_root",
        lambda: (_ for _ in ()).throw(OSError("mkdtemp failed")),
    )
    progress = tmp_path / "progress.json"
    with pytest.raises(OSError, match="mkdtemp failed"):
        watchdog.run_analysis(
            tmp_path,
            tmp_path / "output.json",
            progress,
            revision="2" * 40,
        )
    assert not progress.exists()


def test_partial_worktree_registration_is_discovered_and_removed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_run_harness(monkeypatch)
    removed: list[Path] = []

    def partial_add(
        _repository: Path, path: Path, _revision: str, **_kwargs: object
    ) -> None:
        path.mkdir()
        raise watchdog.WatchdogError("injected add failure after registration")

    def remove(
        _repository: Path, path: Path, **_kwargs: object
    ) -> None:
        removed.append(path)
        shutil.rmtree(path)

    monkeypatch.setattr(watchdog, "_add_worktree", partial_add)
    monkeypatch.setattr(watchdog, "_worktree_registered", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(watchdog, "_remove_worktree", remove)
    output = tmp_path / "output.json"
    result = watchdog.run_analysis(
        tmp_path,
        output,
        tmp_path / "progress.json",
        revision="3" * 40,
    )
    assert result["status"] == "failed"
    assert "injected add failure" in str(result["error"])
    assert len(removed) == 1
    assert not output.exists()


def test_git_drain_failure_retains_attempted_worktree_and_temp_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    _patch_run_harness(monkeypatch, events=events)

    def add_with_live_descendant(
        _repository: Path, path: Path, _revision: str, **_kwargs: object
    ) -> None:
        path.mkdir()
        raise watchdog.WatchdogLiveProcessError("git descendant remains")

    monkeypatch.setattr(watchdog, "_add_worktree", add_with_live_descendant)
    result = watchdog.run_analysis(
        tmp_path,
        tmp_path / "output.json",
        tmp_path / "progress.json",
        revision="5" * 40,
    )
    assert result["status"] == "failed"
    assert result["reason"] == "cleanup_failed"
    assert "untracked Git process group" in str(result["error"])
    assert not any(event.startswith("remove:") for event in events)
    retained_text = str(result["error"]).rsplit("temporary state retained at ", 1)[1]
    retained = Path(retained_text)
    assert retained.exists()
    shutil.rmtree(retained)


def test_live_failure_in_validation_cleanup_skips_analysis_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    validation = tmp_path / "validation"
    analysis = tmp_path / "analysis"
    removed: list[Path] = []

    def registered(_repository: Path, path: Path, **_kwargs: object) -> bool:
        if path == validation:
            raise watchdog.WatchdogLiveProcessError("validation git child remains")
        return True

    monkeypatch.setattr(watchdog, "_worktree_registered", registered)
    monkeypatch.setattr(
        watchdog,
        "_remove_worktree",
        lambda _repository, path, **_kwargs: removed.append(path),
    )
    errors, live = watchdog._cleanup_worktrees(
        tmp_path,
        (
            ("validation", True, False, validation),
            ("analysis", True, False, analysis),
        ),
        deadline_monotonic=10,
    )
    assert live is True
    assert errors == [
        "validation worktree cleanup failed: validation git child remains"
    ]
    assert removed == []


def test_deadline_during_analysis_worktree_removal_is_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_run_harness(monkeypatch)
    calls = 0

    def remove(
        _repository: Path, path: Path, **_kwargs: object
    ) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise watchdog.WatchdogDeadlineError("deadline during remove")
        shutil.rmtree(path)

    monkeypatch.setattr(watchdog, "_remove_worktree", remove)
    output = tmp_path / "output.json"
    result = watchdog.run_analysis(
        tmp_path,
        output,
        tmp_path / "progress.json",
        revision="4" * 40,
    )
    assert result["status"] == "wall_clock_timeout"
    assert result["reason"] == "wall_clock_timeout"
    assert calls == 2
    assert not output.exists()


def test_watchdog_remains_outside_all_sealed_selections() -> None:
    assert watchdog._RUNNER_PATH not in ANALYZER_PATHS
    assert watchdog._RUNNER_PATH not in MACHINE_FACT_TOOLCHAIN_PATHS
    assert len(ANALYZER_PATHS) == 15
    assert len(MACHINE_FACT_TOOLCHAIN_PATHS) == 3
    assert not _is_direct_analyzer_path(watchdog._RUNNER_PATH)
    assert _source_category(watchdog._RUNNER_PATH) is None
