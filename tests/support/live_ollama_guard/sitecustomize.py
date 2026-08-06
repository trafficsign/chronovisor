"""Hermetic socket guard automatically loaded by Python test subprocesses."""

from __future__ import annotations

import json
import os
import socket
import subprocess
from typing import Final


class ForbiddenLiveOllamaConnection(BaseException):
    """Fail before a Python test subprocess reaches the live Ollama port."""


class ForbiddenLiveOllamaProcess(BaseException):
    """Fail before a Python test subprocess spawns the Ollama CLI."""


class ForbiddenUnsafeShellProcess(BaseException):
    """Fail before a Python child crosses an uninspectable shell boundary."""


_ORIGINAL_CONNECT = socket.socket.connect
_ORIGINAL_CONNECT_EX = socket.socket.connect_ex
_ORIGINAL_RUN = subprocess.run
_ORIGINAL_POPEN = subprocess.Popen
_LEDGER_ENV: Final = "CHRONOVISOR_TEST_OLLAMA_GUARD_LEDGER"
_PROBE_TOKEN_ENV: Final = "CHRONOVISOR_TEST_OLLAMA_GUARD_PROBE_TOKEN"
_EXPECTED_PROBE_ENV: Final = "CHRONOVISOR_TEST_OLLAMA_GUARD_EXPECTED_PROBE"
_MAX_TEST_FIELD: Final = 1024
_MAX_ADDRESS_FIELD: Final = 512


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


def _forbid(method: str, address: object) -> None:
    if not (isinstance(address, tuple) and len(address) >= 2 and address[1] == 11434):
        return

    _append_guard_attempt(method, address)
    raise ForbiddenLiveOllamaConnection(
        f"ForbiddenLiveOllamaConnection: tests must not connect to {address!r}"
    )


def _connect(sock: socket.socket, address: object) -> None:
    _forbid("connect", address)
    _ORIGINAL_CONNECT(sock, address)


def _connect_ex(sock: socket.socket, address: object) -> int:
    _forbid("connect_ex", address)
    return _ORIGINAL_CONNECT_EX(sock, address)


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


def _resolved_executable_name(command: object, executable: object = None) -> str:
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


def _forbid_process(
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


class _GuardedPopen(_ORIGINAL_POPEN):
    def __init__(self, args: object, *popenargs: object, **kwargs: object) -> None:
        _forbid_process(
            "subprocess.Popen",
            args,
            executable=kwargs.get("executable"),
            shell=bool(kwargs.get("shell", False)),
        )
        super().__init__(args, *popenargs, **kwargs)


def _run(*popenargs: object, **kwargs: object) -> subprocess.CompletedProcess:
    if subprocess.Popen is _GuardedPopen:
        command = popenargs[0] if popenargs else kwargs.get("args")
        _forbid_process(
            "subprocess.run",
            command,
            executable=kwargs.get("executable"),
            shell=bool(kwargs.get("shell", False)),
        )
    return _ORIGINAL_RUN(*popenargs, **kwargs)


socket.socket.connect = _connect
socket.socket.connect_ex = _connect_ex
subprocess.Popen = _GuardedPopen
subprocess.run = _run
