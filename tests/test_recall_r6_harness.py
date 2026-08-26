from __future__ import annotations

import importlib
import importlib.util
import io
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from chronovisor.core.store import (  # type: ignore[import-untyped]
    RuntimeContext,
    init_chronovisor,
)
from chronovisor.recall import (  # type: ignore[import-untyped]
    recall_distillation as distill,
)

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("recall_r6_harness_test", ROOT / "scripts" / "recall_r6_harness.py")
assert SPEC is not None and SPEC.loader is not None
HARNESS = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = HARNESS
SPEC.loader.exec_module(HARNESS)


def _source(tmp_path: Path) -> tuple[Path, str]:
    source = tmp_path / "source"
    source.mkdir()
    (source / "README.md").write_text("r6\n")
    subprocess.run(["git", "init", "-q"], cwd=source, check=True)
    subprocess.run(["git", "config", "user.email", "r6@example.invalid"], cwd=source, check=True)
    subprocess.run(["git", "config", "user.name", "r6"], cwd=source, check=True)
    subprocess.run(["git", "add", "."], cwd=source, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=source, check=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=source, check=True, capture_output=True, text=True).stdout.strip()
    return source, commit


def _production(tmp_path: Path, *, admitted: bool) -> Path:
    root = tmp_path / "production"
    (root / "runtime" / "recall-distillation").mkdir(parents=True)
    (root / "raw").mkdir()
    if admitted:
        baseline = HARNESS._seal({"schema": HARNESS.BASELINE_SCHEMA, "namespace": "recall-distillation", "artifact_id": "a" * 64, "hard_floor": {"p5_allowed": True}})
        directory = root / "runtime" / "recall-distillation" / "baselines"
        directory.mkdir()
        (directory / "baseline.json").write_text(json.dumps(baseline))
    return root


@pytest.mark.darwin_contract
def test_missing_exact_runtime_blocks_without_provider(tmp_path: Path) -> None:
    source, commit = _source(tmp_path)
    result = HARNESS.run_once(production=_production(tmp_path, admitted=False), source=source, output=tmp_path / "output", source_commit=commit)
    assert result["kind"] == "r6-official-worker-blocked"
    assert result["external_provider_calls"] == 0
    assert result["clone_candidate_published"] is False
    assert result["production_candidate_published"] is False


def test_official_worker_is_called_once_with_empty_teachers_and_no_factory(tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []

    def original_factory(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("factory must be replaced")

    module = SimpleNamespace(_default_workers=original_factory)

    def worker(**kwargs: object) -> dict[str, object]:
        calls.append(kwargs)
        assert kwargs["teachers"] == {}
        assert kwargs["counterfactual"] is None
        assert module._default_workers is not original_factory
        return {"status": "capture_only"}

    module.run_distillation_chunk = worker
    assert HARNESS._test_only_official_chunk(module, tmp_path, ROOT) == {
        "status": "capture_only", "r6_egress_attempts": 0,
        "r6_provider_attempts": 0, "r6_git_sha256": "",
    }
    assert len(calls) == 1
    assert module._default_workers is original_factory


def test_provider_factory_attempt_fails_closed(tmp_path: Path) -> None:
    module = SimpleNamespace(_default_workers=lambda: None, store=SimpleNamespace())

    def worker(**_kwargs: object) -> dict[str, object]:
        return {"factory": module._default_workers()}

    module.run_distillation_chunk = worker
    with pytest.raises(HARNESS.R6Error, match="provider worker factory"):
        HARNESS._test_only_official_chunk(module, tmp_path, ROOT)


def test_swallowed_provider_factory_attempt_still_fails_closed(tmp_path: Path) -> None:
    module = SimpleNamespace(_default_workers=lambda: None)

    def worker(**_kwargs: object) -> dict[str, object]:
        try:
            module._default_workers()
        except HARNESS.R6GuardError:
            pass
        return {"status": "forged-success"}

    module.run_distillation_chunk = worker
    with pytest.raises(HARNESS.R6GuardError, match="swallowed") as caught:
        HARNESS._test_only_official_chunk(module, tmp_path, ROOT)
    assert caught.value.provider_attempts == 1
    assert caught.value.egress_attempts == 0


@pytest.mark.parametrize("method", ("send", "sendall", "sendto"))
def test_socket_egress_aliases_fail_closed(tmp_path: Path, method: str) -> None:
    module = SimpleNamespace(_default_workers=lambda: None)

    def worker(**_kwargs: object) -> dict[str, object]:
        with socket.socket() as client:
            getattr(client, method)(b"forbidden")
        return {"status": "unexpected"}

    module.run_distillation_chunk = worker
    with pytest.raises(HARNESS.R6GuardError) as caught:
        HARNESS._test_only_official_chunk(module, tmp_path, ROOT)
    assert caught.value.egress_attempts == 1
    assert caught.value.provider_attempts == 0


def test_posix_spawnp_fails_closed(tmp_path: Path) -> None:
    module = SimpleNamespace(_default_workers=lambda: None)

    def worker(**_kwargs: object) -> dict[str, object]:
        assert hasattr(os, "posix_spawnp")
        os.posix_spawnp("curl", ("curl",), {})
        return {"status": "unexpected"}

    module.run_distillation_chunk = worker
    with pytest.raises(HARNESS.R6GuardError) as caught:
        HARNESS._test_only_official_chunk(module, tmp_path, ROOT)
    assert caught.value.egress_attempts == 1


@pytest.mark.parametrize("method", ("sendmsg", "sendfile"))
def test_socket_extended_egress_aliases_fail_closed(tmp_path: Path, method: str) -> None:
    if not hasattr(socket.socket, method):
        pytest.skip(f"socket.{method} is unavailable")
    module = SimpleNamespace(_default_workers=lambda: None)

    def worker(**_kwargs: object) -> dict[str, object]:
        with socket.socket() as client:
            if method == "sendmsg":
                client.sendmsg([b"forbidden"])
            else:
                client.sendfile(io.BytesIO(b"forbidden"))
        return {"status": "unexpected"}

    module.run_distillation_chunk = worker
    with pytest.raises(HARNESS.R6GuardError) as caught:
        HARNESS._test_only_official_chunk(module, tmp_path, ROOT)
    assert caught.value.egress_attempts == 1


@pytest.mark.parametrize("api", ("alarm", "signal", "setitimer"))
def test_worker_cannot_disable_parent_phase_watchdog(
    tmp_path: Path, api: str
) -> None:
    module = SimpleNamespace(_default_workers=lambda: None)

    def worker(**_kwargs: object) -> dict[str, object]:
        if api == "alarm":
            signal.alarm(0)
        elif api == "signal":
            signal.signal(signal.SIGALRM, signal.SIG_DFL)
        else:
            signal.setitimer(signal.ITIMER_REAL, 0)
        return {"status": "unexpected"}

    module.run_distillation_chunk = worker
    with pytest.raises(HARNESS.R6GuardError, match="watchdog|signal"):
        HARNESS._test_only_official_chunk(module, tmp_path, ROOT)


@pytest.mark.parametrize(
    "api",
    tuple(
        name
        for name in (
            "spawnv", "spawnvp", "spawnve", "spawnvpe", "fork", "forkpty",
            "sendfile", "kill", "killpg",
        )
        if hasattr(os, name)
    ),
)
def test_worker_process_and_os_sendfile_apis_are_guarded(tmp_path: Path, api: str) -> None:
    module = SimpleNamespace(_default_workers=lambda: None)

    def worker(**_kwargs: object) -> dict[str, object]:
        if api.startswith("spawn"):
            getattr(os, api)(os.P_WAIT, "/usr/bin/true", ("true",))
        elif api == "sendfile":
            os.sendfile(1, 1, 0, 1)
        elif api == "kill":
            os.kill(os.getpid(), signal.SIGCONT)
        elif api == "killpg":
            os.killpg(os.getpgrp(), signal.SIGCONT)
        else:
            getattr(os, api)()
        return {"status": "unexpected"}

    module.run_distillation_chunk = worker
    with pytest.raises(HARNESS.R6GuardError) as caught:
        HARNESS._test_only_official_chunk(module, tmp_path, ROOT)
    assert caught.value.egress_attempts == 1


def test_wrapped_guard_attempt_preserves_parent_counter(tmp_path: Path) -> None:
    module = SimpleNamespace(_default_workers=lambda: None)

    def worker(**_kwargs: object) -> dict[str, object]:
        try:
            os.system("true")
        except HARNESS.R6GuardError:
            raise ValueError("worker wrapped guard") from None
        return {"status": "unexpected"}

    module.run_distillation_chunk = worker
    with pytest.raises(HARNESS.R6GuardError, match="wrapped") as caught:
        HARNESS._test_only_official_chunk(module, tmp_path, ROOT)
    assert caught.value.egress_attempts == 1


def test_preloaded_fake_guard_module_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "socket", SimpleNamespace(socket=object()))
    module = SimpleNamespace(_default_workers=lambda: None)
    module.run_distillation_chunk = lambda **_kwargs: {"status": "unexpected"}
    with pytest.raises(HARNESS.R6Error, match="preloaded|stdlib"):
        HARNESS._test_only_official_chunk(module, tmp_path, ROOT)


def test_worker_cannot_reload_guarded_stdlib(tmp_path: Path) -> None:
    module = SimpleNamespace(_default_workers=lambda: None)
    module.run_distillation_chunk = lambda **_kwargs: importlib.reload(subprocess)
    with pytest.raises(HARNESS.R6GuardError, match="reload"):
        HARNESS._test_only_official_chunk(module, tmp_path, ROOT)


def test_worker_cannot_exit_harness_process(tmp_path: Path) -> None:
    module = SimpleNamespace(_default_workers=lambda: None)
    module.run_distillation_chunk = lambda **_kwargs: sys.exit(0)
    with pytest.raises(HARNESS.R6GuardError, match="exit"):
        HARNESS._test_only_official_chunk(module, tmp_path, ROOT)


def test_source_ignored_bytecode_and_pycache_are_formally_rejected(tmp_path: Path) -> None:
    source = tmp_path / "source"
    (source / "src" / "chronovisor" / "recall" / "__pycache__").mkdir(parents=True)
    (source / "src" / "chronovisor" / "recall" / "__pycache__" / "evil.pyc").write_bytes(b"bad")
    with pytest.raises(HARNESS.R6Error, match="bytecode|__pycache__"):
        HARNESS._assert_source_import_surface(source)


def test_exec_family_is_forbidden(tmp_path: Path) -> None:
    module = SimpleNamespace(_default_workers=lambda: None)

    def worker(**_kwargs: object) -> dict[str, object]:
        os.execv("/bin/echo", ("echo", "forbidden"))
        return {"status": "unexpected"}

    module.run_distillation_chunk = worker
    with pytest.raises(HARNESS.R6GuardError) as caught:
        HARNESS._test_only_official_chunk(module, tmp_path, ROOT)
    assert caught.value.egress_attempts == 1


@pytest.mark.darwin_contract
def test_external_child_watchdog_kills_process_group() -> None:
    started = time.monotonic()
    with pytest.raises(HARNESS.R6Error, match="watchdog"):
        HARNESS._run_external_bounded(
            [sys.executable, "-c", "import time; time.sleep(30)"], timeout=0.05
        )
    assert time.monotonic() - started < 5


@pytest.mark.darwin_contract
def test_external_watchdog_never_kills_an_unrelated_same_python_process() -> None:
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        with pytest.raises(HARNESS.R6Error, match="watchdog"):
            HARNESS._run_external_bounded(
                [sys.executable, "-c", "import time; time.sleep(30)"], timeout=0.05
            )
        assert unrelated.poll() is None
    finally:
        unrelated.kill()
        unrelated.wait(timeout=5)


@pytest.mark.parametrize("delay", (0.3, 0.5, 0.8))
@pytest.mark.darwin_contract
def test_external_rejects_permissive_nested_sandbox_before_delayed_registration(
    tmp_path: Path, delay: float
) -> None:
    source = tmp_path / "source"
    source.write_text("unchanged")
    pid_path = tmp_path / "descendant.pid"
    code = (
        "import os,time,pathlib; "
        "status=pathlib.Path(" + repr(str(tmp_path / "status")) + "); "
        "pid_path=pathlib.Path(" + repr(str(pid_path)) + "); "
        "fd=int(os.environ['R6_DESCENDANT_REGISTRY_FD']); "
        "\ntry: pid=os.fork()\n"
        "except OSError: status.write_text('fork-blocked')\n"
        "else:\n"
        " os._exit(0) if pid else None; "
        f" time.sleep({delay!r}); os.setsid(); "
        " os.write(fd, f'{os.getpid()}\\n'.encode()); "
        " pid_path.write_text(str(os.getpid())); os.close(0); os.close(1); os.close(2); time.sleep(30)"
    )
    with pytest.raises(HARNESS.R6Error, match="nested sandbox-exec"):
        HARNESS._run_external_bounded(
            [
                "/usr/bin/sandbox-exec",
                "-p",
                "(version 1) (allow default) (allow process-exec) (allow process-fork)",
                "--",
                sys.executable,
                "-c",
                code,
            ],
            timeout=5,
        )
    assert not pid_path.exists()
    assert source.read_text() == "unchanged"


@pytest.mark.darwin_contract
def test_external_containment_receipt_closes_registry_fd() -> None:
    completed = HARNESS._run_external_bounded([sys.executable, "-c", "pass"], timeout=5)
    assert completed.r6_containment == {
        "schema": "chronovisor.recall-r6-child-containment.v1",
        "registered_descendants": 0,
        "rejected_registry_entries": 0,
        "remaining_descendants": 0,
        "registry_fd_closed": True,
    }


@pytest.mark.darwin_contract
def test_external_registry_rejects_forged_unrelated_pid() -> None:
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        code = (
            "import os; "
            "pid=" + repr(str(unrelated.pid)) + "; "
            "os.write(int(os.environ['R6_DESCENDANT_REGISTRY_FD']), "
            "f'{pid}\\n'.encode())"
        )
        completed = HARNESS._run_external_bounded([sys.executable, "-c", code], timeout=5)
        assert completed.r6_containment["rejected_registry_entries"] == 1
        assert unrelated.poll() is None
    finally:
        unrelated.kill()
        unrelated.wait(timeout=5)


@pytest.mark.darwin_contract
def test_external_profile_blocks_subsequent_sandbox_exec(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.write_text("unchanged")
    status_path = tmp_path / "status"
    pid_path = tmp_path / "escaped.pid"
    delayed_payload = (
        "import os,time,pathlib; "
        "pid=os.fork(); os._exit(0) if pid else None; "
        "time.sleep(.5); os.setsid(); "
        "os.write(int(os.environ['R6_DESCENDANT_REGISTRY_FD']), f'{os.getpid()}\\n'.encode()); "
        "pathlib.Path(" + repr(str(pid_path)) + ").write_text(str(os.getpid())); "
        "os.close(0); os.close(1); os.close(2); time.sleep(30)"
    )
    code = (
        "import os,pathlib; "
        "status=pathlib.Path(" + repr(str(status_path)) + "); "
        "\ntry: os.execv('/usr/bin/sandbox-exec', ('sandbox-exec', '-p', "
        "'(version 1) (allow default) (allow process-exec) (allow process-fork)', '--', "
        + repr(sys.executable) + ", '-c', " + repr(delayed_payload) + "))\n"
        "except OSError: status.write_text('sandbox-exec-blocked')\n"
    )
    completed = HARNESS._run_external_bounded([sys.executable, "-c", code], timeout=5)
    assert completed.returncode == 0
    assert status_path.read_text() == "sandbox-exec-blocked"
    assert not pid_path.exists()
    assert source.read_text() == "unchanged"


def test_isolated_worker_uses_a_fixed_clean_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYTHONPATH", "/tmp/evil")
    monkeypatch.setenv("PYTHONHOME", "/tmp/evil")
    env = HARNESS._isolated_worker_env(source=ROOT, clone=tmp_path)
    assert "PYTHONPATH" not in env
    assert "PYTHONHOME" not in env
    assert env["HOME"] == str(tmp_path)
    assert env["PATH"] == "/usr/bin:/bin"
    profile = HARNESS._worker_sandbox_profile(source=ROOT, clone=tmp_path)
    assert "(deny process-fork)" in profile
    assert "(deny network*)" in profile
    assert str(ROOT) not in profile
    assert str(tmp_path) in profile


def test_clone_containment_rejects_symlink_and_hardlink(tmp_path: Path) -> None:
    clone = tmp_path / "clone"
    clone.mkdir()
    source = tmp_path / "source"
    source.write_text("immutable")
    (clone / "link").symlink_to(source)
    with pytest.raises(HARNESS.R6Error, match="symlink"):
        HARNESS._assert_clone_contained(clone)
    (clone / "link").unlink()
    os.link(source, clone / "hardlink")
    with pytest.raises(HARNESS.R6Error, match="hardlink"):
        HARNESS._assert_clone_contained(clone)


def test_public_official_worker_rejects_in_process_invocation(tmp_path: Path) -> None:
    module = SimpleNamespace(_default_workers=lambda: None)
    with pytest.raises(HARNESS.R6Error, match="isolated child"):
        HARNESS._official_chunk(module, tmp_path, ROOT)


def test_worker_sandbox_allows_only_clone_writes(tmp_path: Path) -> None:
    if sys.platform != "darwin":
        pytest.skip("R6 uses the Darwin sandbox boundary")
    source = tmp_path / "source"
    clone = tmp_path / "clone"
    source.write_text("source")
    clone.mkdir()
    code = (
        "from pathlib import Path; "
        f"Path({str(clone)!r}, 'inside').write_text('ok'); "
        f"Path({str(source)!r}).write_text('escape')"
    )
    with pytest.raises(subprocess.CalledProcessError):
        HARNESS._run_external_bounded(
            [
                sys.executable, "-I", "-S", "-B", "-c", code,
            ],
            cwd=clone,
            timeout=5,
            env=HARNESS._isolated_worker_env(source=source, clone=clone),
            worker_roots=(source, clone),
        )
    assert (clone / "inside").read_text() == "ok"
    assert source.read_text() == "source"


def test_worker_sandbox_blocks_linked_source_writes_and_escape_primitives(tmp_path: Path) -> None:
    if sys.platform != "darwin":
        pytest.skip("R6 uses the Darwin sandbox boundary")
    source = tmp_path / "source"
    clone = tmp_path / "clone"
    source.write_text("source")
    clone.mkdir()
    code = (
        "import ctypes,os,signal; from pathlib import Path; "
        f"source=Path({str(source)!r}); clone=Path({str(clone)!r}); "
        "attempts=[]; "
        "\nfor kind, action in ("
        "('hardlink', lambda: (os.link(source, clone / 'hard'), (clone / 'hard').write_text('escape'))), "
        "('symlink', lambda: (os.symlink(source, clone / 'soft'), (clone / 'soft').write_text('escape'))), "
        "('ctypes', lambda: ctypes.CDLL(None).system(b'/usr/bin/true')), "
        "('fork', lambda: os.fork()), "
        "('sigmask', lambda: signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGALRM}))"
        "):\n "
        " try: action(); attempts.append(kind + ':allowed')\n "
        " except BaseException: attempts.append(kind + ':blocked')\n "
        "print(','.join(attempts))"
    )
    completed = HARNESS._run_external_bounded(
        [
            sys.executable, "-I", "-S", "-B", "-c", code,
        ],
        cwd=clone,
        timeout=5,
        check=False,
        env=HARNESS._isolated_worker_env(source=source, clone=clone),
        worker_roots=(source, clone),
    )
    assert source.read_text() == "source"
    assert "hardlink:allowed" not in completed.stdout
    assert "symlink:allowed" not in completed.stdout
    assert "ctypes:allowed" not in completed.stdout
    assert "fork:allowed" not in completed.stdout


@pytest.mark.darwin_contract
def test_phase_watchdog_returns_blocker_and_cleans_clone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, commit = _source(tmp_path)
    production = _production(tmp_path, admitted=False)
    module = SimpleNamespace(_default_workers=lambda: None, store=SimpleNamespace())

    def worker(**_kwargs: object) -> dict[str, object]:
        time.sleep(3.0)
        return {"status": "unexpected"}

    module.run_distillation_chunk = worker
    monkeypatch.setattr(HARNESS, "_PHASE_TIMEOUT_SECONDS", 2.0)
    monkeypatch.setattr(HARNESS, "_load_runtime", lambda *_args: (module, {}))
    monkeypatch.setattr(
        HARNESS,
        "_r5_preflight",
        lambda *_args: {"passed": True, "baseline_id": "a" * 64},
    )
    monkeypatch.setattr(HARNESS, "_candidate_pointer_id", lambda *_args: "")
    monkeypatch.setattr(HARNESS, "_worker_snapshot", lambda *_args: {})
    monkeypatch.setattr(HARNESS, "_candidate_ledger_state", lambda *_args: {"records": 0, "head_sha256": ""})
    result = HARNESS.run_once(
        production=production, source=source, output=tmp_path / "output", source_commit=commit,
    )
    assert result["kind"] == "r6-official-worker-blocked"
    assert result["cleanup_receipt"]["remaining"] == 0


@pytest.mark.darwin_contract
def test_parent_monkeypatch_does_not_reach_isolated_worker_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, commit = _source(tmp_path)
    production = _production(tmp_path, admitted=False)
    module = SimpleNamespace(_default_workers=lambda: None, store=SimpleNamespace())

    def worker(**_kwargs: object) -> dict[str, object]:
        with socket.socket() as client:
            client.send(b"forbidden")
        return {"status": "unexpected"}

    module.run_distillation_chunk = worker
    monkeypatch.setattr(HARNESS, "_load_runtime", lambda *_args: (module, {}))
    monkeypatch.setattr(
        HARNESS,
        "_r5_preflight",
        lambda *_args: {"passed": True, "baseline_id": "a" * 64},
    )
    monkeypatch.setattr(HARNESS, "_candidate_pointer_id", lambda *_args: "")
    monkeypatch.setattr(HARNESS, "_worker_snapshot", lambda *_args: {})
    monkeypatch.setattr(HARNESS, "_candidate_ledger_state", lambda *_args: {"records": 0, "head_sha256": ""})
    result = HARNESS.run_once(
        production=production, source=source, output=tmp_path / "output", source_commit=commit,
    )
    assert result["kind"] == "r6-official-worker-blocked"
    assert result["provider_calls"] == 0
    assert result["provider_attempts"] == 0
    # The parent stub is deliberately not serializable into the clean child.
    # The child fails on this minimal source fixture before any provider or
    # egress action, which is the only truthful counter value.
    assert result["egress_attempts"] == 0
    assert result["cleanup_receipt"]["remaining"] == 0
    assert len(result["cleanup_receipt"]["removed_roots"]) == 1


def test_new_candidate_allows_unchanged_bound_candidate_ledger() -> None:
    before = {
        "candidate-policy.json": "", "candidate-ledger.jsonl": "stable",
        "state.json": "old", "policies": "old", "locked-replays": "old", "runs": "old",
    }
    after = {
        "candidate-policy.json": "new", "candidate-ledger.jsonl": "stable",
        "state.json": "new", "policies": "new", "locked-replays": "new", "runs": "new",
    }
    ledger = {"records": 3, "head_sha256": "a" * 64}
    HARNESS._require_candidate_newness(before, after, ledger, ledger, {"candidate_head": "a" * 64})
    with pytest.raises(HARNESS.R6Error, match="candidate lineage"):
        HARNESS._require_candidate_newness(before, after, ledger, ledger, {"candidate_head": "b" * 64})


def test_idempotence_state_digest_excludes_only_volatile_worker_timestamps() -> None:
    stable = {"schema": "x", "kind": "worker-state", "run_id": "a" * 64}
    assert HARNESS._stable_state_digest({**stable, "stage_started_at": "2026-01-01T00:00:00+00:00"}) == HARNESS._stable_state_digest(
        {**stable, "stage_started_at": "2026-01-01T00:00:01+00:00", "last_success_at": "2026-01-01T00:00:02+00:00"}
    )
    assert HARNESS._stable_state_digest(stable) != HARNESS._stable_state_digest({**stable, "run_id": "b" * 64})
    first = {**stable, "stage_started_at": "2026-01-01T00:00:00+00:00", "last_success_at": "2026-01-01T00:00:02+00:00", "seal_sha256": "a" * 64}
    second = {**stable, "stage_started_at": "2026-01-01T00:00:01+00:00", "last_success_at": "2026-01-01T00:00:03+00:00", "seal_sha256": "b" * 64}
    assert HARNESS._stable_state_digest(first) == HARNESS._stable_state_digest(second)
    assert HARNESS._digest(first) != HARNESS._digest(second)


def test_nested_replay_row_extra_key_is_rejected() -> None:
    module, pointer, policy, replay, run, state, r5, heads = _closed_candidate_fixture()
    replay["training_rows"][0]["unexpected"] = True
    with pytest.raises(HARNESS.R6Error, match="row.*closed"):
        HARNESS._assert_candidate_artifact_schemas(
            module, pointer, policy, replay, run, state=state, r5=r5, heads=heads
        )


def test_nested_artifact_seal_and_content_id_are_recomputed() -> None:
    module, pointer, policy, replay, run, state, r5, heads = _closed_candidate_fixture()
    policy["threshold"] = -1.0
    with pytest.raises(HARNESS.R6Error, match="content identity|seal|policy"):
        HARNESS._assert_candidate_artifact_schemas(
            module, pointer, policy, replay, run, state=state, r5=r5, heads=heads
        )


@pytest.mark.darwin_contract
def test_trusted_executable_rejects_symlink_and_path_spoof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spoof = tmp_path / "git"
    spoof.write_text("not git")
    link = tmp_path / "git-link"
    link.symlink_to(spoof)
    monkeypatch.setitem(HARNESS._TRUSTED_EXECUTABLES, str(link), "0" * 64)
    with pytest.raises(HARNESS.R6Error, match="symlink"):
        HARNESS._trusted_executable(str(link))
    monkeypatch.setenv("PATH", str(tmp_path))
    assert HARNESS._trusted_executable("/usr/bin/git") == Path("/usr/bin/git")


@pytest.mark.darwin_contract
def test_local_git_probe_is_not_egress_but_provider_cli_is(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = SimpleNamespace(_default_workers=lambda: None)

    def local_git(**_kwargs: object) -> dict[str, object]:
        completed = subprocess.run(
            ["git", "rev-parse", "origin/main"], cwd=ROOT, capture_output=True,
            text=True, timeout=5,
        )
        assert completed.returncode in {0, 128}
        return {"status": "ok"}

    module.run_distillation_chunk = local_git
    assert HARNESS._test_only_official_chunk(module, tmp_path, ROOT)["r6_egress_attempts"] == 0

    monkeypatch.chdir(tmp_path)
    assert HARNESS._test_only_official_chunk(module, tmp_path, ROOT)["r6_git_sha256"]

    def provider_cli(**_kwargs: object) -> dict[str, object]:
        subprocess.run(["curl", "https://example.invalid"], check=False)
        return {"status": "unexpected"}

    module.run_distillation_chunk = provider_cli
    with pytest.raises(HARNESS.R6Error, match="network egress"):
        HARNESS._test_only_official_chunk(module, tmp_path, ROOT)

    def injected_kwargs(**_kwargs: object) -> dict[str, object]:
        subprocess.run(
            ["git", "rev-parse", "origin/main"], cwd=ROOT, capture_output=True,
            text=True, timeout=5, executable="/bin/sh",
        )
        return {"status": "unexpected"}

    module.run_distillation_chunk = injected_kwargs
    with pytest.raises(HARNESS.R6Error, match="network egress"):
        HARNESS._test_only_official_chunk(module, tmp_path, ROOT)

    for unsafe in ({"shell": True}, {"pass_fds": ()}, {"preexec_fn": lambda: None}):
        def injected_option(*, _unsafe: object = unsafe, **_kwargs: object) -> dict[str, object]:
            subprocess.run(
                ["git", "rev-parse", "origin/main"], cwd=ROOT, capture_output=True,
                text=True, timeout=5, **_unsafe,  # type: ignore[arg-type]
            )
            return {"status": "unexpected"}

        module.run_distillation_chunk = injected_option
        with pytest.raises(HARNESS.R6Error, match="network egress"):
            HARNESS._test_only_official_chunk(module, tmp_path, ROOT)

    def injected_git(**_kwargs: object) -> dict[str, object]:
        subprocess.run(
            ["git", "rev-parse", "origin/main"], cwd=ROOT, capture_output=True,
            text=True, timeout=5, env={"PATH": "/tmp/evil"},
        )
        return {"status": "unexpected"}

    module.run_distillation_chunk = injected_git
    with pytest.raises(HARNESS.R6Error, match="network egress"):
        HARNESS._test_only_official_chunk(module, tmp_path, ROOT)


@pytest.mark.darwin_contract
def test_clone_proof_is_parent_held_and_forgery_or_replace_fails(
    tmp_path: Path,
) -> None:
    source, _commit = _source(tmp_path)
    production = _production(tmp_path, admitted=False)
    output = tmp_path / "output"
    clone, _before, proof = HARNESS._clone(production, source=source, with_proof=True)
    try:
        assert not (clone / ".r6-clone-proof.json").exists()
        post_target = HARNESS._target_identity(clone)
        receipt = HARNESS._assert_clone_proof(clone, proof, output=output, post_target=post_target)
        assert receipt["clone_path"] == str(clone)
        forged = dict(proof)
        forged["clone_path"] = str(tmp_path / "forged")
        with pytest.raises(HARNESS.R6Error, match="paths|proof"):
            HARNESS._assert_clone_proof(clone, forged, output=output, post_target=post_target)
    finally:
        HARNESS._cleanup_clone(clone)


@pytest.mark.darwin_contract
def test_clone_symlink_replacement_is_rejected_and_cleaned(tmp_path: Path) -> None:
    source, _commit = _source(tmp_path)
    production = _production(tmp_path, admitted=False)
    clone, _before, proof = HARNESS._clone(production, source=source, with_proof=True)
    parent = clone.parent
    try:
        HARNESS._cleanup_clone(clone)
        parent.mkdir()
        clone.symlink_to(production, target_is_directory=True)
        with pytest.raises(HARNESS.R6Error, match="paths|stat|symlink"):
            HARNESS._assert_clone_proof(clone, proof, output=tmp_path / "output", post_target="a" * 64)
    finally:
        if parent.exists():
            clone.unlink(missing_ok=True)
            parent.rmdir()


def test_external_git_redirect_environment_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source, commit = _source(tmp_path)
    monkeypatch.setenv("GIT_DIR", str(tmp_path / "external.git"))
    with pytest.raises(HARNESS.R6Error, match="GIT"):
        HARNESS.run_once(
            production=_production(tmp_path, admitted=False),
            source=source,
            output=tmp_path / "output",
            source_commit=commit,
        )


@pytest.mark.darwin_contract
def test_source_snapshot_binds_actual_git_layout(tmp_path: Path) -> None:
    source, _commit = _source(tmp_path)
    snapshot = HARNESS.source_snapshot(source)
    assert snapshot["git_layout"]["work_tree"] == str(source.resolve())
    assert snapshot["git_layout"]["entry"] == str((source / ".git").resolve())
    assert snapshot["git_layout"]["git_dir"] == str((source / ".git").resolve())


def test_official_worker_integration_accepts_empty_teacher_mapping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real worker's empty-teacher route is bounded and never builds defaults."""

    init_chronovisor(RuntimeContext(tmp_path))
    (tmp_path / "config.toml").write_text("[recall.distillation]\nenabled = true\n")
    monkeypatch.setattr(distill, "_default_workers", lambda *_args, **_kwargs: pytest.fail("default provider factory called"))
    result = distill.run_distillation_chunk(
        root=tmp_path,
        raw_dir=tmp_path / "raw",
        config_path=tmp_path / "config.toml",
        teachers={},
        counterfactual=None,
        max_elapsed_seconds=60,
    )
    assert result["teachers_available"] is False
    assert result["status"] in {"capture_only", "deferred"}


def test_official_candidate_requires_official_empty_teacher_promotion(tmp_path: Path) -> None:
    module = SimpleNamespace()
    with pytest.raises(HARNESS.R6Error, match="empty-teacher"):
        HARNESS._official_candidate(
            module, tmp_path, {}, {"p5_allowed": False, "teachers_available": False},
            before_id="", before_snapshot={}, before_candidate_ledger={},
        )
    with pytest.raises(HARNESS.R6Error, match="generate a candidate"):
        HARNESS._official_candidate(
            module, tmp_path, {},
            {"p5_allowed": True, "teachers_available": False, "promotion": {"status": "held"}},
            before_id="", before_snapshot={}, before_candidate_ledger={},
        )


@pytest.mark.darwin_contract
def test_admitted_but_unrunnable_official_path_returns_blocker(tmp_path: Path) -> None:
    source, commit = _source(tmp_path)
    production = _production(tmp_path, admitted=True)
    result = HARNESS.run_once(production=production, source=source, output=tmp_path / "output", source_commit=commit)
    assert result["kind"] == "r6-official-worker-blocked"
    assert result["external_provider_calls"] == 0
    assert result["clone_candidate_published"] is False
    assert result["production_candidate_published"] is False


@pytest.mark.darwin_contract
def test_rejects_dirty_source_overlap_symlink_and_tampered_output(tmp_path: Path) -> None:
    source, commit = _source(tmp_path)
    production = _production(tmp_path, admitted=False)
    with pytest.raises(HARNESS.R6Error, match="overlap"):
        HARNESS.run_once(production=production, source=source, output=production / "out", source_commit=commit)
    link = tmp_path / "link"
    link.symlink_to(production, target_is_directory=True)
    with pytest.raises(HARNESS.R6Error, match="symlink"):
        HARNESS.run_once(production=link, source=source, output=tmp_path / "out", source_commit=commit)
    (source / "dirty").write_text("x")
    with pytest.raises(HARNESS.R6Error, match="exact and clean"):
        HARNESS.run_once(production=production, source=source, output=tmp_path / "out", source_commit=commit)
    sealed = HARNESS._seal({"schema": HARNESS.R6_SCHEMA, "namespace": "recall-distillation", "kind": "x"})
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(sealed))
    sealed["kind"] = "tampered"
    path.write_text(json.dumps(sealed))
    with pytest.raises(HARNESS.R6Error, match="seal"):
        HARNESS._read_sealed(path, HARNESS.R6_SCHEMA)


def _closed_workset(*, include_timing: bool) -> dict[str, object]:
    value: dict[str, object] = {
        "ready": 0,
        "leased": 0,
        "completed": 0,
        "quarantined": 0,
        "backlog": 0,
        "total": 0,
        "last_durable_receipt": {"generation": 0, "head_sha256": ""},
        "last_durable_progress": None,
    }
    if include_timing:
        value.update(
            {
                "retry_wait": 0,
                "oldest_backlog_age_seconds": 0,
                "oldest_ready_age_seconds": 0,
                "oldest_retry_wait_age_seconds": 0,
                "next_retry_in_seconds": 0,
                "stages": {
                    stage: {
                        key: 0
                        for key in (
                            "ready",
                            "leased",
                            "completed",
                            "quarantined",
                            "retry_wait",
                            "backlog",
                        )
                    }
                    for stage in (
                        "snapshot",
                        "teacher",
                        "counterfactual",
                        "dataset",
                        "evaluation",
                        "retry_wait",
                    )
                },
            }
        )
    return value


def _closed_candidate_fixture() -> tuple[Any, ...]:
    ids = {
        "raw": "a" * 64,
        "baseline": "b" * 64,
        "snapshot": "c" * 64,
        "split": "d" * 64,
        "offline": "e" * 64,
        "rows": "f" * 64,
        "cohort": "1" * 64,
        "candidate_head": "2" * 64,
        "label_head": "3" * 64,
        "manifest_head": "4" * 64,
        "policy": "5" * 64,
        "replay": "6" * 64,
        "run": "7" * 64,
        "historical": "8" * 64,
    }
    module = SimpleNamespace(
        POLICY_SCHEMA=distill.POLICY_SCHEMA,
        OX_RAMP_REQUEST_REVISION=distill.OX_RAMP_REQUEST_REVISION,
        train_tiny_policy=distill.train_tiny_policy,
        store=SimpleNamespace(DISTILLATION_SCHEMA="chronovisor.recall-distillation.v1"),
    )
    lineage = {
        "training_snapshot_id": ids["snapshot"],
        "locked_replay_id": ids["replay"],
        "baseline_artifact_id": ids["baseline"],
        "model_cohort_sha256": ids["cohort"],
        "raw_watermark": ids["raw"],
        "label_chain_head": ids["label_head"],
        "feature_revision": "recall-distill-text-v2",
        "split_plan_id": ids["split"],
        "offline_gate_sha256": ids["offline"],
        "training_rows_sha256": ids["rows"],
        "candidate_head": ids["candidate_head"],
        "profile_contract_id": "",
    }
    policy = {
        "schema": distill.POLICY_SCHEMA,
        "namespace": "recall-distillation",
        "artifact_id": ids["policy"],
        "seal_sha256": "9" * 64,
        "kind": "tiny-logistic-policy",
        **distill.train_tiny_policy(()),
        "lineage": lineage,
    }
    policy["training_rows"] = 1
    policy["validation_rows"] = 1
    pointer = {
        "schema": "chronovisor.recall-distillation.v1",
        "namespace": "recall-distillation",
        "kind": "candidate-policy-pointer",
        "policy_id": ids["policy"],
        "seal_sha256": "a" * 64,
    }
    replay = {
        "schema": "chronovisor.recall-distill-locked-replay.v1",
        "namespace": "recall-distillation",
        "artifact_id": ids["replay"],
        "seal_sha256": "b" * 64,
        "kind": "locked-replay-input",
        "training_snapshot_id": ids["snapshot"],
        "training_rows": [{
            "rally_id": ids["snapshot"],
            "candidate_id": ids["candidate_head"],
            "session_cluster_id": ids["snapshot"],
            "as_of": "2026-01-01T00:00:00+00:00",
            "dimension": "answer_utility",
            "verdict": "helpful",
            "authority": "teacher-only",
            "features": {"query_chargram_coverage": 0.5, "candidate_chargram_precision": 0.5},
            "route": "local/teacher",
            "route_identity": {"provider": "local", "model": "teacher", "location": "local"},
            "teacher_role": "critic",
            "model_digest": ids["cohort"],
            "generator_model_digest": "",
            "judge_model_digest": "",
            "generator_route_identity": {},
            "judge_route_identity": {},
            "counterfactual_ref": "",
            "a0_sha256": "",
            "a1_sha256": "",
            "blind_orders": [],
            "counterfactual_producer": "",
            "counterfactual_revision": "",
            "probe": False,
            "source": "teacher-label",
            "profile": "local-triad-v1",
            "cohort": "local-triad-v1",
            "assignment_revision": "assignment-v2",
            "assignment_authority": "",
            "profile_contract_id": "",
            "expires_at": "",
            "identity_revision": "",
            "request_revision": "",
            "group_id": ids["snapshot"],
            "label_split_plan_id": ids["split"],
            "order_agreement": False,
            "label_record_sha256": ids["snapshot"],
            "payload_digest": "",
            "payload_source": {},
            "work_id": "",
            "negative_veto_conflict": False,
            "feature_parity": True,
            "future_leakage": False,
            "split": "train",
            "split_plan_id": ids["split"],
            "locked_test_read_only": False,
            "locked_test_evidence_ref": "",
        }],
        "baseline_artifact_id": ids["baseline"],
        "policy_sha256": ids["policy"],
        "training_rows_sha256": ids["rows"],
        "candidate_head": ids["candidate_head"],
        "profile_contract_id": "",
        "offline_gate_sha256": ids["offline"],
        "model_cohort_sha256": ids["cohort"],
        "split_revision": "grouped-rolling-v1",
    }
    run = {
        "schema": "chronovisor.recall-distill-run.v1",
        "namespace": "recall-distillation",
        "artifact_id": ids["run"],
        "seal_sha256": "c" * 64,
        "kind": "bounded-chunk",
        "raw_watermark": ids["raw"],
        "baseline_artifact_id": ids["baseline"],
        "manifest_head": ids["manifest_head"],
        "candidate_head": ids["candidate_head"],
        "label_head": ids["label_head"],
        "processed": 1,
        "candidate_snapshots": 1,
        "labels_written": 1,
        "ox_workset": {},
        "local_workset": _closed_workset(include_timing=True),
        "ox_profile_contract_id": "",
        "ox_profile_stopped": False,
        "counterfactuals_written": 0,
        "p5_allowed": True,
    }
    state = {
        "schema": "chronovisor.recall-distillation.v1",
        "namespace": "recall-distillation",
        "seal_sha256": "d" * 64,
        "kind": "worker-state",
        "status": "ready",
        "worker_status": "capture_only",
        "rollout_percent": 0,
        "raw_watermark": ids["raw"],
        "baseline_artifact_id": ids["baseline"],
        "historical_index_sha256": ids["historical"],
        "manifest_chain_head": ids["manifest_head"],
        "run_id": ids["run"],
        "processed": 1,
        "candidate_snapshots": 1,
        "labels_written": 1,
        "ox_workset": {},
        "local_workset": _closed_workset(include_timing=True),
        "ox_profile_contract_id": "",
        "ox_profile_stopped": False,
        "counterfactuals_written": 0,
        "teacher_model_calls": 0,
        "counterfactual_model_calls": 0,
        "cold_start_pending": False,
        "cold_start_lane_turn": 0,
        "split_plan_id": ids["split"],
        "manifest_backlog": 0,
        "candidate_backlog": 0,
        "promotion_status": "candidate",
        "promotion_reason": "",
        "incumbent_policy_id": ids["policy"],
        "rollout_evaluation_status": "not_applicable",
        "hold_reason": "",
        "capture_only_reasons": [],
        "last_success_at": "",
        "error_code": "",
    }
    # The official store derives content IDs from the unsigned payload and
    # seals the complete object (including artifact_id).  Keep this fixture an
    # exact valid artifact rather than relying on shape-only placeholder IDs.
    rows_digest = HARNESS._digest(replay["training_rows"])
    ids["rows"] = rows_digest
    lineage["training_rows_sha256"] = rows_digest
    replay["training_rows_sha256"] = rows_digest
    replay["policy_sha256"] = HARNESS._policy_payload_digest(policy)
    replay_unsigned = {key: item for key, item in replay.items() if key not in {"artifact_id", "seal_sha256"}}
    replay["artifact_id"] = HARNESS._digest({key: item for key, item in replay_unsigned.items() if key != "artifact_id"})
    replay["seal_sha256"] = HARNESS._digest({key: item for key, item in replay.items() if key != "seal_sha256"})
    lineage["locked_replay_id"] = str(replay["artifact_id"])
    policy["artifact_id"] = HARNESS._digest({key: item for key, item in policy.items() if key not in {"artifact_id", "seal_sha256"}})
    policy["seal_sha256"] = HARNESS._digest({key: item for key, item in policy.items() if key != "seal_sha256"})
    pointer["policy_id"] = policy["artifact_id"]
    pointer["seal_sha256"] = HARNESS._digest({key: item for key, item in pointer.items() if key != "seal_sha256"})
    run["artifact_id"] = HARNESS._digest({key: item for key, item in run.items() if key not in {"artifact_id", "seal_sha256"}})
    run["seal_sha256"] = HARNESS._digest({key: item for key, item in run.items() if key != "seal_sha256"})
    state["run_id"] = run["artifact_id"]
    state["seal_sha256"] = HARNESS._digest({key: item for key, item in state.items() if key != "seal_sha256"})
    r5 = {
        "passed": True,
        "reason": "",
        "baseline_id": ids["baseline"],
        "raw_watermark": ids["raw"],
        "label_head": ids["label_head"],
        "cohort_sha256": ids["cohort"],
        "split_plan_id": ids["split"],
        "profile_contract_id": "",
    }
    heads = {"candidate": ids["candidate_head"], "label": ids["label_head"], "manifest": ids["manifest_head"]}
    return module, pointer, policy, replay, run, state, r5, heads


def test_exact_official_worker_fixture_passes_closed_schemas() -> None:
    module, pointer, policy, replay, run, state, r5, heads = _closed_candidate_fixture()
    HARNESS._assert_candidate_artifact_schemas(
        module, pointer, policy, replay, run, state=state, r5=r5, heads=heads
    )


@pytest.mark.darwin_contract
def test_completion_worker_nested_schema_is_closed_and_range_checked() -> None:
    module, pointer, policy, replay, run, state, r5, heads = _closed_candidate_fixture()
    worker = {
        "status": "capture_only",
        "processed": 0,
        "p5_allowed": True,
        "teachers_available": False,
        "counterfactual_available": False,
        "candidate_snapshots": 0,
        "labels_written": 0,
        "ox_workset": {},
        "local_workset": _closed_workset(include_timing=True),
        "ox_profile_contract_id": "",
        "ox_profile_stopped": False,
        "counterfactuals_written": 0,
        "cold_start_pending": False,
        "split_plan_id": heads["candidate"],
        "manifest_backlog": 0,
        "candidate_backlog": 0,
        "promotion": {"status": "candidate", "policy_id": policy["artifact_id"]},
        "rollout_evaluation": {"status": "not_applicable"},
        "run_id": run["artifact_id"],
        "state_sha256": state["seal_sha256"],
        "r6_egress_attempts": 0,
        "r6_provider_attempts": 0,
        "r6_git_sha256": "",
        "r6_child_containment": {
            "schema": "chronovisor.recall-r6-child-containment.v1",
            "registered_descendants": 0,
            "rejected_registry_entries": 0,
            "remaining_descendants": 0,
            "registry_fd_closed": True,
            "sandbox": HARNESS._sandbox_identity(),
        },
    }
    HARNESS._assert_worker_result_schema(worker)
    worker["r6_child_containment"]["sandbox"]["sha256"] = "0" * 64
    with pytest.raises(HARNESS.R6Error, match="sandbox identity"):
        HARNESS._assert_worker_result_schema(worker)
    worker["r6_child_containment"]["sandbox"] = HARNESS._sandbox_identity()
    worker["promotion"]["unexpected"] = True
    with pytest.raises(HARNESS.R6Error, match="promotion.*closed"):
        HARNESS._assert_worker_result_schema(worker)


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda state, _run: state.__setitem__("unexpected", 1), "state schema"),
        (lambda state, _run: state.pop("processed"), "state schema"),
        (lambda state, _run: state.__setitem__("processed", "0"), "processed"),
        (lambda _state, run: run["local_workset"].__setitem__("unexpected", 1), "workset schema"),
        (lambda _state, run: run["local_workset"].pop("stages"), "workset schema"),
        (lambda _state, run: run.__setitem__("label_head", "wrong"), "run label_head"),
    ],
)
def test_worker_run_state_schema_rejects_adversarial_mutations(
    mutator: object, message: str
) -> None:
    module, pointer, policy, replay, run, state, r5, heads = _closed_candidate_fixture()
    mutator(state, run)  # type: ignore[operator]
    with pytest.raises(HARNESS.R6Error, match=message):
        HARNESS._assert_candidate_artifact_schemas(
            module, pointer, policy, replay, run, state=state, r5=r5, heads=heads
        )


def test_candidate_pointer_and_replay_ids_are_transitively_bound() -> None:
    module, pointer, policy, replay, run, state, r5, heads = _closed_candidate_fixture()
    pointer["policy_id"] = "e" * 64
    with pytest.raises(HARNESS.R6Error, match="policy identity"):
        HARNESS._assert_candidate_artifact_schemas(
            module, pointer, policy, replay, run, state=state, r5=r5, heads=heads
        )
    module, pointer, policy, replay, run, state, r5, heads = _closed_candidate_fixture()
    replay["artifact_id"] = "e" * 64
    with pytest.raises(HARNESS.R6Error, match="replay lineage"):
        HARNESS._assert_candidate_artifact_schemas(
            module, pointer, policy, replay, run, state=state, r5=r5, heads=heads
        )


@pytest.mark.parametrize("field", ("run_id", "raw_watermark", "baseline_artifact_id", "manifest_chain_head"))
def test_worker_state_transitive_bindings_reject_mismatch(field: str) -> None:
    module, pointer, policy, replay, run, state, r5, heads = _closed_candidate_fixture()
    state[field] = "e" * 64
    with pytest.raises(HARNESS.R6Error, match="transitively bound"):
        HARNESS._assert_candidate_artifact_schemas(
            module, pointer, policy, replay, run, state=state, r5=r5, heads=heads
        )
