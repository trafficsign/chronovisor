from __future__ import annotations

import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest


def _run_nested_pytest(test_path: Path) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    tests_path = str(Path(__file__).parent)
    pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        os.pathsep.join((tests_path, pythonpath)) if pythonpath else tests_path
    )
    environment.pop("CHRONOVISOR_TEST_OLLAMA_GUARD_EXPECTED_PROBE", None)

    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "conftest",
            "-q",
            str(test_path),
        ],
        cwd=Path(__file__).parents[1],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize(
    ("family", "address"),
    (
        (socket.AF_INET, ("127.0.0.1", 11434)),
        (socket.AF_INET6, ("::1", 11434, 0, 0)),
    ),
)
@pytest.mark.parametrize("method_name", ("connect", "connect_ex"))
def test_parent_process_rejects_live_ollama_port(
    family: socket.AddressFamily,
    address: tuple[object, ...],
    method_name: str,
    forbidden_live_ollama_exception: type[BaseException],
    expected_live_ollama_guard_probe: None,
) -> None:
    del expected_live_ollama_guard_probe
    with socket.socket(family) as client:
        method = getattr(client, method_name)
        with pytest.raises(
            forbidden_live_ollama_exception,
            match="ForbiddenLiveOllamaConnection",
        ):
            method(address)


@pytest.mark.parametrize(
    ("family_name", "address"),
    (
        ("AF_INET", ("127.0.0.1", 11434)),
        ("AF_INET6", ("::1", 11434, 0, 0)),
    ),
)
@pytest.mark.parametrize("method_name", ("connect", "connect_ex"))
def test_python_child_rejects_live_ollama_port(
    family_name: str,
    address: tuple[object, ...],
    method_name: str,
    expected_live_ollama_guard_probe: None,
) -> None:
    del expected_live_ollama_guard_probe
    script = (
        "import socket\n"
        f"client = socket.socket(socket.{family_name})\n"
        f"client.{method_name}({address!r})\n"
    )

    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "ForbiddenLiveOllamaConnection" in completed.stderr


@pytest.mark.parametrize("method_name", ("run", "Popen"))
@pytest.mark.parametrize(
    "command",
    (
        ["/opt/homebrew/bin/ollama", "--version"],
    ),
)
def test_parent_process_rejects_ollama_cli(
    method_name: str,
    command: list[str],
    forbidden_live_ollama_process_exception: type[BaseException],
    expected_live_ollama_guard_probe: None,
) -> None:
    del expected_live_ollama_guard_probe
    method = getattr(subprocess, method_name)

    with pytest.raises(
        forbidden_live_ollama_process_exception,
        match="ForbiddenLiveOllamaProcess",
    ):
        method(command)


@pytest.mark.parametrize("method_name", ("run", "Popen"))
@pytest.mark.parametrize(
    "command",
    (
        ["ollama", "--version"],
    ),
)
def test_python_child_rejects_ollama_cli(
    method_name: str,
    command: list[str],
    expected_live_ollama_guard_probe: None,
) -> None:
    del expected_live_ollama_guard_probe
    script = (
        "import subprocess\n"
        f"process = subprocess.{method_name}({command!r})\n"
        + ("process.wait()\n" if method_name == "Popen" else "")
    )

    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "ForbiddenLiveOllamaProcess" in completed.stderr


@pytest.mark.parametrize("method_name", ("run", "Popen"))
@pytest.mark.parametrize(
    "shell_command",
    (
        "printf '%s' safe",
        ">/tmp/chronovisor-unreachable-shell-output printf safe",
        'echo "$(ollama --version)"',
        'echo "`ollama --version`"',
        "printf '%s' 'ollama --version'",
    ),
)
def test_parent_process_rejects_any_real_shell_true(
    method_name: str,
    shell_command: str,
    forbidden_unsafe_shell_process_exception: type[BaseException],
    expected_live_ollama_guard_probe: None,
) -> None:
    del expected_live_ollama_guard_probe
    method = getattr(subprocess, method_name)

    with pytest.raises(
        forbidden_unsafe_shell_process_exception,
        match="ForbiddenUnsafeShellProcess.*shell=True",
    ):
        method(shell_command, shell=True)


@pytest.mark.parametrize("method_name", ("run", "Popen"))
@pytest.mark.parametrize(
    "command",
    (
        ["/bin/sh", "-c", "printf safe"],
        ["/bin/bash", "-lc", "printf safe"],
        ["/bin/bash", "-ec", "printf safe"],
        ["/bin/bash", "--rcfile", "/tmp/unused", "-c", "printf safe"],
        ["/bin/bash", "--init-file", "/tmp/unused", "-c", "printf safe"],
        ["/usr/bin/env", "OLLAMA_HOST=127.0.0.1", "ollama", "--version"],
        ["/usr/bin/env", "FOO=1", "/bin/bash", "-lc", "printf safe"],
        ["/usr/bin/env", "-S", "python ollama"],
        ["/usr/bin/env", "-S", "sh -c", "printf safe"],
        ["/usr/bin/env", "--split-string=python ollama"],
        ["/usr/bin/nohup", "ollama", "--version"],
        ["/usr/bin/time", "ollama", "--version"],
        ["command", "ollama", "--version"],
        ["exec", "ollama", "--version"],
    ),
)
def test_parent_process_rejects_unsafe_shell_argv_boundaries(
    method_name: str,
    command: list[str],
    forbidden_unsafe_shell_process_exception: type[BaseException],
    expected_live_ollama_guard_probe: None,
) -> None:
    del expected_live_ollama_guard_probe
    method = getattr(subprocess, method_name)

    with pytest.raises(
        forbidden_unsafe_shell_process_exception,
        match="ForbiddenUnsafeShellProcess",
    ):
        method(command)


@pytest.mark.parametrize("method_name", ("run", "Popen"))
def test_parent_process_rejects_launcher_executable_override(
    method_name: str,
    forbidden_unsafe_shell_process_exception: type[BaseException],
    expected_live_ollama_guard_probe: None,
) -> None:
    del expected_live_ollama_guard_probe
    method = getattr(subprocess, method_name)

    with pytest.raises(
        forbidden_unsafe_shell_process_exception,
        match="ForbiddenUnsafeShellProcess.*launcher-wrapper:env",
    ):
        method(["ignored-argv0", "python"], executable="/usr/bin/env")


@pytest.mark.parametrize("method_name", ("run", "Popen"))
@pytest.mark.parametrize(
    "shell_command",
    (
        "printf '%s' safe",
        ">/tmp/chronovisor-unreachable-shell-output printf safe",
        'echo "$(ollama --version)"',
        'echo "`ollama --version`"',
        "printf '%s' 'ollama --version'",
    ),
)
def test_python_child_rejects_any_real_shell_true(
    method_name: str,
    shell_command: str,
    expected_live_ollama_guard_probe: None,
) -> None:
    del expected_live_ollama_guard_probe
    script = (
        "import subprocess\n"
        f"process = subprocess.{method_name}({shell_command!r}, shell=True)\n"
        + ("process.wait()\n" if method_name == "Popen" else "")
    )

    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "ForbiddenUnsafeShellProcess" in completed.stderr
    assert "shell=True" in completed.stderr


@pytest.mark.parametrize("method_name", ("run", "Popen"))
@pytest.mark.parametrize(
    "command",
    (
        ["/bin/sh", "-c", "printf safe"],
        ["/bin/bash", "-lc", "printf safe"],
        ["/bin/bash", "-ec", "printf safe"],
        ["/bin/bash", "--rcfile", "/tmp/unused", "-c", "printf safe"],
        ["/bin/bash", "--init-file", "/tmp/unused", "-c", "printf safe"],
        ["/usr/bin/env", "OLLAMA_HOST=127.0.0.1", "ollama", "--version"],
        ["/usr/bin/env", "FOO=1", "/bin/bash", "-lc", "printf safe"],
        ["/usr/bin/env", "-S", "python ollama"],
        ["/usr/bin/env", "-S", "sh -c", "printf safe"],
        ["/usr/bin/env", "--split-string=python ollama"],
        ["/usr/bin/nohup", "ollama", "--version"],
        ["/usr/bin/time", "ollama", "--version"],
        ["command", "ollama", "--version"],
        ["exec", "ollama", "--version"],
    ),
)
def test_python_child_rejects_unsafe_shell_argv_boundaries(
    method_name: str,
    command: list[str],
    expected_live_ollama_guard_probe: None,
) -> None:
    del expected_live_ollama_guard_probe
    script = (
        "import subprocess\n"
        f"process = subprocess.{method_name}({command!r})\n"
        + ("process.wait()\n" if method_name == "Popen" else "")
    )

    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "ForbiddenUnsafeShellProcess" in completed.stderr


@pytest.mark.parametrize("method_name", ("run", "Popen"))
def test_python_child_rejects_launcher_executable_override(
    method_name: str,
    expected_live_ollama_guard_probe: None,
) -> None:
    del expected_live_ollama_guard_probe
    script = (
        "import subprocess\n"
        f"process = subprocess.{method_name}"
        "(['ignored-argv0', 'python'], executable='/usr/bin/env')\n"
        + ("process.wait()\n" if method_name == "Popen" else "")
    )

    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "ForbiddenUnsafeShellProcess" in completed.stderr
    assert "launcher-wrapper:env" in completed.stderr


def test_other_local_tcp_ports_are_not_blocked() -> None:
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        address = server.getsockname()
        assert address[1] != 11434

        with socket.socket() as client:
            client.connect(address)
            connection, _peer = server.accept()
            with connection:
                client.sendall(b"ok")
                assert connection.recv(2) == b"ok"


def test_python_child_other_local_socket_behavior_is_not_blocked() -> None:
    script = (
        "import socket\n"
        "left, right = socket.socketpair()\n"
        "left.sendall(b'ok')\n"
        "assert right.recv(2) == b'ok'\n"
        "left.close(); right.close()\n"
    )

    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_unrelated_commands_are_not_blocked() -> None:
    completed = subprocess.run(
        [sys.executable, "-c", "print('run-ok')"],
        capture_output=True,
        text=True,
        check=False,
    )
    process = subprocess.Popen(
        [sys.executable, "-c", "print('popen-ok')"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    stdout, stderr = process.communicate(timeout=5)
    python_word = subprocess.run(
        [sys.executable, "-c", "print('ollama-is-data')"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0
    assert completed.stdout.strip() == "run-ok"
    assert process.returncode == 0, stderr
    assert stdout.strip() == "popen-ok"
    assert python_word.returncode == 0
    assert python_word.stdout.strip() == "ollama-is-data"


def test_shell_false_path_commands_are_not_blocked() -> None:
    completed = subprocess.run("/usr/bin/true", shell=False, check=False)
    process = subprocess.Popen("/usr/bin/true", shell=False)

    assert completed.returncode == 0
    assert process.wait(timeout=5) == 0


def test_subprocess_run_allows_explicitly_mocked_popen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []

    class FakePopen:
        def __init__(self, args: object, *_args: object, **_kwargs: object) -> None:
            calls.append(args)
            self.args = args
            self.returncode = 0

        def __enter__(self):
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def communicate(self, *_args: object, **_kwargs: object) -> tuple[bytes, bytes]:
            return b"", b""

        def poll(self) -> int:
            return 0

    monkeypatch.setattr(subprocess, "Popen", FakePopen)

    completed = subprocess.run(["ollama", "--version"], check=False)
    shell_completed = subprocess.run("ollama --version", shell=True, check=False)

    assert completed.returncode == 0
    assert shell_completed.returncode == 0
    assert calls == [["ollama", "--version"], "ollama --version"]


@pytest.mark.parametrize(
    ("child_attempt", "ledger_method"),
    (
        (
            "import socket\nsocket.socket().connect(('127.0.0.1', 11434))\n",
            "connect",
        ),
        (
            "import subprocess\nsubprocess.run(['ollama', '--version'])\n",
            "subprocess.run",
        ),
    ),
)
def test_swallowed_child_nonzero_still_fails_owning_nested_pytest(
    tmp_path: Path,
    child_attempt: str,
    ledger_method: str,
) -> None:
    nested_test = tmp_path / "test_swallowed_live_ollama_attempt.py"
    nested_test.write_text(
        f"""\
import subprocess
import sys

import pytest


@pytest.fixture(scope="session", autouse=True)
def swallowed_forbidden_child_attempt():
    script = {child_attempt!r}
    completed = subprocess.run(
        [sys.executable, \"-c\", script],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode != 0


def test_caller_remains_green_after_fixture_swallows_child_nonzero():
    pass
""",
        encoding="utf-8",
    )

    completed = _run_nested_pytest(nested_test)

    output = completed.stdout + completed.stderr
    assert completed.returncode == pytest.ExitCode.TESTS_FAILED
    assert "live Ollama guard ledger recorded forbidden access attempt" in output
    assert "test_caller_remains_green_after_fixture_swallows_child_nonzero" in output
    assert f'"method":"{ledger_method}"' in output
    assert "INTERNALERROR" not in output


def test_swallowed_collection_attempt_fails_at_session_finish(tmp_path: Path) -> None:
    nested_test = tmp_path / "test_swallowed_collection_attempt.py"
    nested_test.write_text(
        """\
import socket


try:
    socket.socket().connect(("127.0.0.1", 11434))
except BaseException:
    pass


def test_body_is_green():
    pass
""",
        encoding="utf-8",
    )

    completed = _run_nested_pytest(nested_test)

    output = completed.stdout + completed.stderr
    assert completed.returncode == pytest.ExitCode.TESTS_FAILED
    assert "LIVE OLLAMA GUARD FAILURE" in output
    assert "outside a test protocol" in output
    assert '"method":"connect"' in output
    assert "INTERNALERROR" not in output
