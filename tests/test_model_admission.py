"""Admission handoff across continuous background work and isolated children."""

from __future__ import annotations

import fcntl
import json
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from chronovisor.core import ollama_lease, research_scheduler


def _isolate(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CHRONOVISOR_OLLAMA_RESOURCE_LOCK", str(tmp_path / "model.lock"))
    monkeypatch.delenv("CHRONOVISOR_MODEL_ADMISSION_FD", raising=False)
    monkeypatch.setattr(research_scheduler, "SYNC_DIR", tmp_path / "sync")
    monkeypatch.setattr(research_scheduler, "RESEARCH_LOCK", tmp_path / "research.lock")
    monkeypatch.setattr(research_scheduler, "ACTIVE_FILE", tmp_path / "active.json")
    monkeypatch.setattr(research_scheduler, "SCHEDULER_LOG", tmp_path / "scheduler.jsonl")


def _await_locked(path) -> None:
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        if path.exists():
            with path.open("a+") as handle:
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    return
        time.sleep(0.005)
    raise AssertionError("waiter did not acquire the handoff queue")


def test_waiter_gets_next_model_turn_before_continuous_drain(tmp_path, monkeypatch) -> None:
    _isolate(tmp_path, monkeypatch)
    entered = threading.Event()
    release = threading.Event()
    failures = []

    def waiter() -> None:
        try:
            with ollama_lease.model_admission_turn(timeout_ms=2000):
                entered.set()
                assert release.wait(3)
        except BaseException as exc:
            failures.append(exc)

    with ollama_lease.model_resource_lease(exclusive=True):
        thread = threading.Thread(target=waiter)
        thread.start()
        _await_locked(tmp_path / "model.lock.admission.queue")
    try:
        # The prior owner immediately asks again, as ingest-drain does.
        with pytest.raises(TimeoutError):
            with ollama_lease.model_resource_lease(exclusive=True, timeout_ms=0):
                pytest.fail("continuous drain overtook its waiter")
        assert entered.wait(2)
    finally:
        release.set()
        thread.join(3)
    assert not thread.is_alive()
    assert not failures
    with ollama_lease.model_resource_lease(exclusive=True, timeout_ms=0):
        pass


def test_one_turn_covers_children_but_never_bypasses_model_lock(tmp_path, monkeypatch) -> None:
    _isolate(tmp_path, monkeypatch)
    script = """
import json
from chronovisor.core.ollama_lease import model_resource_lease
try:
    with model_resource_lease(exclusive=True, timeout_ms=0):
        print(json.dumps({"acquired": True}))
except TimeoutError:
    print(json.dumps({"acquired": False}))
"""
    with ollama_lease.model_admission_turn(timeout_ms=1000):
        options = ollama_lease.model_admission_subprocess_options()

        def child() -> bool:
            result = subprocess.run(
                [sys.executable, "-c", script], capture_output=True, text=True,
                timeout=5, check=True, **options,
            )
            return json.loads(result.stdout)["acquired"]

        assert all(child() for _ in range(4))
        with (tmp_path / "model.lock").open("a+") as actual_lease:
            fcntl.flock(actual_lease, fcntl.LOCK_EX)
            assert child() is False
        assert child() is True
    assert ollama_lease.model_admission_subprocess_options() == {}


def test_unlocked_inherited_descriptor_is_rejected(tmp_path, monkeypatch) -> None:
    _isolate(tmp_path, monkeypatch)
    with (tmp_path / "model.lock.admission").open("a+") as handle:
        monkeypatch.setenv("CHRONOVISOR_MODEL_ADMISSION_FD", str(handle.fileno()))
        with pytest.raises(RuntimeError, match="invalid inherited"):
            with ollama_lease.model_resource_lease(exclusive=True, timeout_ms=0):
                pytest.fail("unowned admission must not be inherited")


def test_waiting_research_gets_next_lane_before_continuous_librarian(tmp_path, monkeypatch) -> None:
    _isolate(tmp_path, monkeypatch)
    entered = threading.Event()
    release = threading.Event()
    failures = []

    def waiter() -> None:
        try:
            with research_scheduler.research_lane(
                "waiting", enabled=True, mode="sleep", purpose="sleep",
                needs_model=True, wait_timeout_seconds=2,
            ) as lease:
                assert lease.admission.admitted
                entered.set()
                assert release.wait(3)
        except BaseException as exc:
            failures.append(exc)

    with research_scheduler.research_lane(
        "owner", enabled=True, mode="sleep", purpose="sleep", needs_model=True,
    ) as lease:
        assert lease.admission.admitted
        thread = threading.Thread(target=waiter)
        thread.start()
        _await_locked(tmp_path / "research.lock.queue")
    try:
        with research_scheduler.research_lane(
            "repeat", enabled=True, mode="sleep", purpose="sleep", needs_model=True,
        ) as repeat:
            assert not repeat.admission.admitted
        assert entered.wait(2)
    finally:
        release.set()
        thread.join(3)
    assert not thread.is_alive()
    assert not failures


def test_research_child_inherits_turn_and_foreground_still_cancels(tmp_path, monkeypatch) -> None:
    _isolate(tmp_path, monkeypatch)
    with research_scheduler.research_lane(
        "cancel", enabled=True, mode="sleep", purpose="sleep", needs_model=True,
    ) as lease, ollama_lease.model_admission_turn(timeout_ms=1000):
        result = research_scheduler.run_cancellable_command(
            [sys.executable, "-c", (
                "import json; from chronovisor.core.ollama_lease import model_resource_lease; "
                "\nwith model_resource_lease(exclusive=True,timeout_ms=0): "
                "print(json.dumps({'ok':True}))"
            )], "", lease, timeout_seconds=5, inherit_model_admission=True,
        )
        assert result.status == "completed"
        assert result.value == {"ok": True}
        monkeypatch.setattr(lease, "cancelled", lambda: True)
        cancelled = research_scheduler.run_cancellable_command(
            [sys.executable, "-c", "raise AssertionError('must not start')"],
            "", lease, timeout_seconds=5, inherit_model_admission=True,
        )
        assert cancelled.status == "cancelled"


def test_counterfactual_keeps_one_turn_for_both_answers_and_blind_orders(tmp_path, monkeypatch) -> None:
    from chronovisor.core import ollama
    from chronovisor.recall import recall_distillation as distill

    _isolate(tmp_path, monkeypatch)
    roles = ("recall.distill.answer_generator", "recall.distill.utility_judge")
    identities = {
        role: {"role": role, "provider": "ollama", "model": role,
               "location": "local"}
        for role in roles
    }
    digests = dict(zip(roles, ("a" * 64, "b" * 64), strict=True))
    monkeypatch.setattr(ollama, "runtime_generation_routes", lambda requested: (
        SimpleNamespace(**identities[requested[0]], structured_output=True),
    ))
    original = ollama_lease.model_admission_turn
    turns = []
    calls = []

    @contextmanager
    def turn(**kwargs):
        turns.append("enter")
        with original(**kwargs):
            yield
        turns.append("exit")

    def command(_argv, encoded, lease, **kwargs):
        assert lease.admission.admitted
        assert kwargs["inherit_model_admission"]
        assert ollama_lease.model_admission_subprocess_options()["pass_fds"]
        assert turns == ["enter"]
        request = json.loads(encoded)
        calls.append(request["operation"])
        return SimpleNamespace(status="completed", value={
            **{key: request[key] for key in ("schema", "operation", "role", "request_id")},
            "ok": True, "route_identity": identities[request["role"]],
            "model_digest": digests[request["role"]],
            "result": {"answer": "synthetic", "blind_choice": "tie"},
        })

    monkeypatch.setattr(ollama_lease, "model_admission_turn", turn)
    monkeypatch.setattr(research_scheduler, "run_cancellable_command", command)
    worker = distill._WorkerCounterfactual(12_000, identities, digests)
    result = worker.compare({
        "rally_id": "synthetic", "candidate_id": "synthetic",
        "query": "synthetic", "context": [],
    })
    assert result["verdict"] == "neutral"
    assert calls == ["answer", "answer", "utility", "utility"]
    assert turns == ["enter", "exit"]


def test_mainline_deadline_reaps_child_and_releases_reserved_turn(tmp_path, monkeypatch) -> None:
    _isolate(tmp_path, monkeypatch)

    class MainlineDeadline(BaseException):
        pass

    started = []
    popen = subprocess.Popen

    def record_child(*args, **kwargs):
        child = popen(*args, **kwargs)
        started.append(child)
        return child

    monkeypatch.setattr(research_scheduler.subprocess, "Popen", record_child)
    with pytest.raises(MainlineDeadline):
        with research_scheduler.research_lane(
            "deadline", enabled=True, mode="sleep", purpose="sleep",
            needs_model=True, wait_timeout_seconds=1, reserve_model_turn=True,
        ) as lease:
            checks = 0

            def cancelled():
                nonlocal checks
                checks += 1
                if checks == 2:
                    raise MainlineDeadline()
                return False

            monkeypatch.setattr(lease, "cancelled", cancelled)
            research_scheduler.run_cancellable_command(
                [sys.executable, "-c", "import time; time.sleep(60)"], "", lease,
                timeout_seconds=10, inherit_model_admission=True,
            )
    assert len(started) == 1 and started[0].poll() is not None
    with ollama_lease.model_resource_lease(exclusive=True, timeout_ms=0):
        pass
