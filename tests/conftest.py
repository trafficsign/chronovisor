from __future__ import annotations

import json
import os
import secrets
import socket
import subprocess
import tempfile
from pathlib import Path
from typing import Final

import pytest
from _pytest.runner import CallInfo

from chronovisor.core.runtime_config import SearchEmbeddingConfig


class ForbiddenLiveOllamaConnection(BaseException):
    """Fail fast when a test attempts to reach the operator's live Ollama."""


class ForbiddenLiveOllamaProcess(BaseException):
    """Fail before a test can spawn the Ollama CLI against the live daemon."""


class ForbiddenUnsafeShellProcess(BaseException):
    """Fail before a test crosses an uninspectable native shell boundary."""


_ORIGINAL_SOCKET_CONNECT = socket.socket.connect
_ORIGINAL_SOCKET_CONNECT_EX = socket.socket.connect_ex
_ORIGINAL_SUBPROCESS_RUN = subprocess.run
_ORIGINAL_SUBPROCESS_POPEN = subprocess.Popen
_HERMETIC_CHILD_SUPPORT = Path(__file__).parent / "support" / "live_ollama_guard"
_LEDGER_ENV: Final = "CHRONOVISOR_TEST_OLLAMA_GUARD_LEDGER"
_PROBE_TOKEN_ENV: Final = "CHRONOVISOR_TEST_OLLAMA_GUARD_PROBE_TOKEN"
_EXPECTED_PROBE_ENV: Final = "CHRONOVISOR_TEST_OLLAMA_GUARD_EXPECTED_PROBE"
_MAX_TEST_FIELD: Final = 1024
_MAX_ADDRESS_FIELD: Final = 512
_MAX_REPORT_BYTES: Final = 8192
_ORIGINAL_PYTHONPATH: str | None = None
_ORIGINAL_GUARD_ENV: dict[str, str | None] = {}
_LEDGER_PATH: Path | None = None
_ACCOUNTED_LEDGER_RANGES: list[tuple[int, int]] = []


def _expected_probe_is_active() -> bool:
    token = os.environ.get(_PROBE_TOKEN_ENV)
    return bool(token) and os.environ.get(_EXPECTED_PROBE_ENV) == token


def _append_guard_attempt(method: str, address: object) -> None:
    if _expected_probe_is_active():
        return

    ledger = os.environ.get(_LEDGER_ENV)
    if not ledger:
        return

    row = {
        "method": method,
        "pid": os.getpid(),
        "test": os.environ.get("PYTEST_CURRENT_TEST", "<collection/background>")[
            :_MAX_TEST_FIELD
        ],
        "address": repr(address)[:_MAX_ADDRESS_FIELD],
    }
    payload = (json.dumps(row, ensure_ascii=True, separators=(",", ":")) + "\n").encode()
    flags = os.O_WRONLY | os.O_APPEND
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(ledger, flags)
    try:
        written = os.write(descriptor, payload)
        if written != len(payload):
            raise OSError(f"short write to live Ollama guard ledger: {written}/{len(payload)}")
    finally:
        os.close(descriptor)


def _forbid_live_ollama(method: str, address: object) -> None:
    if not (isinstance(address, tuple) and len(address) >= 2 and address[1] == 11434):
        return

    _append_guard_attempt(method, address)
    raise ForbiddenLiveOllamaConnection(
        f"ForbiddenLiveOllamaConnection: tests must not connect to {address!r}"
    )


def _guarded_socket_connect(
    sock: socket.socket,
    address: object,
) -> None:
    _forbid_live_ollama("connect", address)
    _ORIGINAL_SOCKET_CONNECT(sock, address)


def _guarded_socket_connect_ex(
    sock: socket.socket,
    address: object,
) -> int:
    _forbid_live_ollama("connect_ex", address)
    return _ORIGINAL_SOCKET_CONNECT_EX(sock, address)


_SHELL_INTERPRETERS: Final = frozenset({"bash", "dash", "ksh", "sh", "zsh"})
_SHELL_OPTIONS_WITH_VALUE: Final = frozenset(
    {"-O", "-o", "--init-file", "--rcfile"}
)
_LAUNCHER_WRAPPERS: Final = frozenset({"command", "env", "exec", "nohup", "time"})


def _argv_tokens(command: object) -> list[str]:
    if not isinstance(command, (list, tuple)) or not command:
        return []
    try:
        return [os.fsdecode(token) for token in command]
    except (TypeError, UnicodeError):
        return []


def _shell_command_option_is_present(tokens: list[str]) -> bool:
    index = 1
    while index < len(tokens):
        option = tokens[index]
        if option == "--" or not option.startswith("-"):
            return False
        if (
            not option.startswith("--")
            and option != "-"
            and "c" in option[1:]
        ):
            return True
        index += 2 if option in _SHELL_OPTIONS_WITH_VALUE else 1
    return False


def _resolved_executable_name(
    command: object,
    executable: object = None,
) -> str:
    tokens = _argv_tokens(command)
    candidate = executable
    if candidate is None:
        if isinstance(command, (str, bytes, os.PathLike)):
            candidate = command
        elif tokens:
            candidate = tokens[0]
    if not isinstance(candidate, (str, bytes, os.PathLike)):
        return ""
    try:
        return os.path.basename(os.fsdecode(candidate)).casefold()
    except (TypeError, UnicodeError):
        return ""


def _unsafe_shell_boundary(
    command: object,
    *,
    executable: object = None,
    shell: bool = False,
) -> str | None:
    if shell:
        return "shell=True"
    executable_name = _resolved_executable_name(command, executable)
    if executable_name in _LAUNCHER_WRAPPERS:
        return f"launcher-wrapper:{executable_name}"
    if executable_name in _SHELL_INTERPRETERS and _shell_command_option_is_present(
        _argv_tokens(command)
    ):
        return "shell-command-string"
    return None


def _command_executable(
    command: object,
    executable: object = None,
) -> str:
    return _resolved_executable_name(command, executable)


def _forbid_live_ollama_process(
    method: str,
    command: object,
    *,
    executable: object = None,
    shell: bool = False,
) -> None:
    unsafe_reason = _unsafe_shell_boundary(
        command,
        executable=executable,
        shell=shell,
    )
    if unsafe_reason is not None:
        _append_guard_attempt(
            method,
            {"reason": unsafe_reason, "command": command},
        )
        raise ForbiddenUnsafeShellProcess(
            "ForbiddenUnsafeShellProcess: tests must not spawn an uninspectable "
            f"native shell ({unsafe_reason}): {command!r}"
        )

    if _command_executable(command, executable) != "ollama":
        return

    _append_guard_attempt(method, command)
    raise ForbiddenLiveOllamaProcess(
        f"ForbiddenLiveOllamaProcess: tests must not spawn the Ollama CLI: {command!r}"
    )


class _GuardedSubprocessPopen(_ORIGINAL_SUBPROCESS_POPEN):
    def __init__(self, args: object, *popenargs: object, **kwargs: object) -> None:
        _forbid_live_ollama_process(
            "subprocess.Popen",
            args,
            executable=kwargs.get("executable"),
            shell=bool(kwargs.get("shell", False)),
        )
        super().__init__(args, *popenargs, **kwargs)


def _guarded_subprocess_run(
    *popenargs: object,
    **kwargs: object,
) -> subprocess.CompletedProcess:
    # A test that replaces Popen owns the fake process boundary.  Let the real
    # subprocess.run implementation exercise that mock without treating the
    # inert argv as an attempted native spawn.
    if subprocess.Popen is _GuardedSubprocessPopen:
        command = popenargs[0] if popenargs else kwargs.get("args")
        _forbid_live_ollama_process(
            "subprocess.run",
            command,
            executable=kwargs.get("executable"),
            shell=bool(kwargs.get("shell", False)),
        )
    return _ORIGINAL_SUBPROCESS_RUN(*popenargs, **kwargs)


def _ledger_size() -> int:
    if _LEDGER_PATH is None:
        return 0
    try:
        return _LEDGER_PATH.stat().st_size
    except OSError:
        return 0


def _read_ledger_range(start: int, end: int) -> str:
    if _LEDGER_PATH is None or end <= start:
        return ""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(_LEDGER_PATH, flags)
        try:
            payload = os.pread(descriptor, min(end - start, _MAX_REPORT_BYTES), start)
        finally:
            os.close(descriptor)
    except OSError as exc:
        return f"<unable to read live Ollama guard ledger: {exc}>"

    details = payload.decode(errors="replace").rstrip()
    if end - start > len(payload):
        details += f"\n<truncated {end - start - len(payload)} bytes>"
    return details


def _emit_owned_attempt_failure(item: pytest.Item, start: int, end: int) -> None:
    details = _read_ledger_range(start, end)
    message = (
        "live Ollama guard ledger recorded forbidden access attempt(s) "
        f"during {item.nodeid}:\n{details}"
    )

    def fail_forbidden_attempt() -> None:
        pytest.fail(message, pytrace=False)

    call = CallInfo.from_call(fail_forbidden_attempt, when="call")
    report = pytest.TestReport.from_item_and_call(item, call)
    item.ihook.pytest_runtest_logreport(report=report)


def _unaccounted_ledger_ranges(end: int) -> list[tuple[int, int]]:
    gaps: list[tuple[int, int]] = []
    cursor = 0
    for start, stop in sorted(_ACCOUNTED_LEDGER_RANGES):
        if start > cursor:
            gaps.append((cursor, min(start, end)))
        cursor = max(cursor, stop)
        if cursor >= end:
            break
    if cursor < end:
        gaps.append((cursor, end))
    return [(start, stop) for start, stop in gaps if stop > start]


def pytest_sessionstart(session: pytest.Session) -> None:
    """Make collection and execution hermetic in this process and children."""

    del session
    global _LEDGER_PATH, _ORIGINAL_PYTHONPATH
    descriptor, ledger_name = tempfile.mkstemp(prefix="chronovisor-ollama-guard-", suffix=".jsonl")
    os.fchmod(descriptor, 0o600)
    os.close(descriptor)
    _LEDGER_PATH = Path(ledger_name)
    _ACCOUNTED_LEDGER_RANGES.clear()

    for name in (_LEDGER_ENV, _PROBE_TOKEN_ENV, _EXPECTED_PROBE_ENV):
        _ORIGINAL_GUARD_ENV[name] = os.environ.get(name)
    os.environ[_LEDGER_ENV] = ledger_name
    os.environ[_PROBE_TOKEN_ENV] = secrets.token_hex(32)
    os.environ.pop(_EXPECTED_PROBE_ENV, None)

    socket.socket.connect = _guarded_socket_connect
    socket.socket.connect_ex = _guarded_socket_connect_ex
    subprocess.Popen = _GuardedSubprocessPopen
    subprocess.run = _guarded_subprocess_run
    _ORIGINAL_PYTHONPATH = os.environ.get("PYTHONPATH")
    child_path = str(_HERMETIC_CHILD_SUPPORT)
    os.environ["PYTHONPATH"] = (
        os.pathsep.join((child_path, _ORIGINAL_PYTHONPATH))
        if _ORIGINAL_PYTHONPATH
        else child_path
    )


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_runtest_protocol(item: pytest.Item, nextitem: pytest.Item | None):
    """Charge all ledger writes during a protocol to the owning test."""

    del nextitem
    start = _ledger_size()
    yield
    end = _ledger_size()
    if end <= start:
        return

    _ACCOUNTED_LEDGER_RANGES.append((start, end))
    _emit_owned_attempt_failure(item, start, end)


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Fail on leftover attempts, then restore and remove process-global state."""

    del exitstatus
    try:
        ledger_end = _ledger_size()
        gaps = _unaccounted_ledger_ranges(ledger_end)
        if gaps:
            details = "\n".join(_read_ledger_range(start, end) for start, end in gaps)
            reporter = session.config.pluginmanager.get_plugin("terminalreporter")
            message = (
                "live Ollama guard ledger recorded forbidden access attempt(s) "
                f"outside a test protocol:\n{details}"
            )
            if reporter is not None:
                reporter.write_sep("=", "LIVE OLLAMA GUARD FAILURE", red=True)
                reporter.write_line(message, red=True)
            session.exitstatus = pytest.ExitCode.TESTS_FAILED
    finally:
        socket.socket.connect = _ORIGINAL_SOCKET_CONNECT
        socket.socket.connect_ex = _ORIGINAL_SOCKET_CONNECT_EX
        subprocess.run = _ORIGINAL_SUBPROCESS_RUN
        subprocess.Popen = _ORIGINAL_SUBPROCESS_POPEN
        if _ORIGINAL_PYTHONPATH is None:
            os.environ.pop("PYTHONPATH", None)
        else:
            os.environ["PYTHONPATH"] = _ORIGINAL_PYTHONPATH
        for name, value in _ORIGINAL_GUARD_ENV.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        if _LEDGER_PATH is not None:
            try:
                _LEDGER_PATH.unlink()
            except OSError:
                pass


@pytest.fixture()
def forbidden_live_ollama_exception() -> type[BaseException]:
    return ForbiddenLiveOllamaConnection


@pytest.fixture()
def forbidden_live_ollama_process_exception() -> type[BaseException]:
    return ForbiddenLiveOllamaProcess


@pytest.fixture()
def forbidden_unsafe_shell_process_exception() -> type[BaseException]:
    return ForbiddenUnsafeShellProcess


@pytest.fixture()
def expected_live_ollama_guard_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    """Suppress ledger accounting only for explicit guard behavior probes."""

    monkeypatch.setenv(_EXPECTED_PROBE_ENV, os.environ[_PROBE_TOKEN_ENV])


@pytest.fixture(autouse=True)
def isolate_operator_raw_layout(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep unit behavior independent from the operator's live rollout mode."""

    monkeypatch.setenv("CHRONOVISOR_RAW_LAYOUT", "legacy")


@pytest.fixture(autouse=True)
def isolate_operator_search_rollout(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep tests from enqueueing work against the live semantic service."""

    from chronovisor.core import search

    monkeypatch.setattr(
        search,
        "load_search_embedding_config",
        lambda: SearchEmbeddingConfig(enabled=False),
    )
