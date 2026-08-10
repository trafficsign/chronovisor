from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from chronovisor.core import semantic_client
from chronovisor.hosts import hook_dispatcher, server
from chronovisor.ingest import convergence, ingest_drain
from chronovisor.lab import classification_library_pilot
from chronovisor.ops import burn_monitor, converge_worker, dashboard
from chronovisor.raw import claude_code_record, codex_record
from chronovisor.recall import librarian, recall_runtime
from chronovisor.search import reranker_service, semantic_service

BLOCKED = SimpleNamespace(allowed=False)
BLOCKED_PAYLOAD = {"status": "blocked", "category": "okf_startup_blocked"}


def _deny(*_args: object, **_kwargs: object) -> SimpleNamespace:
    return BLOCKED


def _boom(*_args: object, **_kwargs: object) -> None:
    raise AssertionError("blocked entrypoint crossed the startup gate")


class _UnreadableStdin:
    def read(self) -> str:
        raise AssertionError("blocked entrypoint read stdin")


class _NoMkdir:
    def mkdir(self, *_args: object, **_kwargs: object) -> None:
        raise AssertionError("blocked entrypoint created an evidence directory")


def _assert_blocked(capsys: pytest.CaptureFixture[str]) -> None:
    captured = capsys.readouterr()
    output = captured.out or captured.err
    assert json.loads(output) == BLOCKED_PAYLOAD


def test_ingest_drain_blocks_watch_and_direct_release(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(ingest_drain, "okf_startup_status", _deny)
    monkeypatch.setattr(ingest_drain, "watch", _boom)
    monkeypatch.setattr(ingest_drain, "_release_ingest_runner", _boom)

    assert ingest_drain.main(["--watch"]) == 75
    _assert_blocked(capsys)
    assert ingest_drain.drain() == BLOCKED_PAYLOAD


def test_hook_dispatcher_blocks_before_stdin_but_prompt_submit_is_noop(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(hook_dispatcher, "okf_startup_status", _deny)
    monkeypatch.setattr(hook_dispatcher.sys, "stdin", _UnreadableStdin())

    assert hook_dispatcher.main(["--host", "codex", "--event", "Stop", "--hook"]) == 75
    _assert_blocked(capsys)
    assert (
        hook_dispatcher.main(
            ["--host", "codex", "--event", "UserPromptSubmit", "--hook"]
        )
        == 0
    )
    assert capsys.readouterr().out == "{}\n"


@pytest.mark.parametrize(
    "command", ["serve", "rebuild", "rollback", "upgrade-ann", "archive-legacy"]
)
def test_semantic_mutating_commands_block_before_config(
    command: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(semantic_service, "okf_startup_status", _deny)
    monkeypatch.setattr(semantic_service, "load_search_embedding_config", _boom)

    assert semantic_service.main([command]) == 75
    _assert_blocked(capsys)


@pytest.mark.parametrize("command", ["serve", "warm"])
def test_reranker_mutating_commands_block_before_config(
    command: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(reranker_service, "okf_startup_status", _deny)
    monkeypatch.setattr(reranker_service, "load_reranker_config", _boom)

    assert reranker_service.main([command, "--json"]) == 75
    _assert_blocked(capsys)


def test_burn_monitor_blocks_before_evidence_directory(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(burn_monitor, "okf_startup_status", _deny)
    monkeypatch.setattr(burn_monitor, "BURN_ROOT", _NoMkdir())

    assert burn_monitor.main(["--expected-commit", "test"]) == 75
    _assert_blocked(capsys)


def test_converge_worker_blocks_before_any_lane(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(convergence, "okf_startup_status", _deny)
    monkeypatch.setattr(converge_worker, "run_converge", _boom)

    assert converge_worker.main([]) == 75
    _assert_blocked(capsys)


def test_librarian_has_no_blocked_read_or_dry_run_bypass(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(librarian, "okf_startup_status", _deny)
    monkeypatch.setattr(librarian, "build_librarian_status", _boom)

    assert (
        librarian.main(["--root", str(tmp_path), "--status", "--dry-run", "--json"])
        == 75
    )
    _assert_blocked(capsys)


@pytest.mark.parametrize("command", ["run-once", "adopt"])
def test_classification_pilot_mutating_commands_block(
    command: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(classification_library_pilot, "okf_startup_status", _deny)
    monkeypatch.setattr(classification_library_pilot, "run_once", _boom)
    monkeypatch.setattr(classification_library_pilot, "adopt", _boom)

    assert classification_library_pilot.main([command]) == 75
    _assert_blocked(capsys)


@pytest.mark.parametrize(
    "argv",
    [
        ["--feedback", "useful"],
        ["--warmup"],
        ["--hook"],
    ],
)
def test_recall_feedback_warmup_and_hook_block_before_side_effects(
    argv: list[str],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(recall_runtime, "okf_startup_status", _deny)
    monkeypatch.setattr(recall_runtime, "append_feedback", _boom)
    monkeypatch.setattr(recall_runtime, "load_policy", _boom)
    monkeypatch.setattr(recall_runtime.sys, "stdin", _UnreadableStdin())

    assert recall_runtime.main(argv) == 75
    _assert_blocked(capsys)


@pytest.mark.parametrize("module", [codex_record, claude_code_record])
def test_raw_recorders_block_before_hook_stdin(
    module,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(module, "okf_startup_status", _deny)
    monkeypatch.setattr(module, "run", _boom)
    monkeypatch.setattr(module.sys, "stdin", _UnreadableStdin())

    assert module.main(["--hook", "--dry-run"]) == 75
    _assert_blocked(capsys)


def test_diagnostic_commands_bypass_the_gate(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(semantic_service, "okf_startup_status", _boom)
    monkeypatch.setattr(semantic_service, "load_search_embedding_config", object)
    monkeypatch.setattr(semantic_client, "health", lambda _config: {"status": "ok"})
    assert semantic_service.main(["status"]) == 0

    monkeypatch.setattr(reranker_service, "okf_startup_status", _boom)
    monkeypatch.setattr(reranker_service, "_read_status", lambda: {"status": "ok"})
    monkeypatch.setattr(reranker_service, "load_reranker_config", object)
    monkeypatch.setattr(
        "chronovisor.core.reranker_client.health",
        lambda _config: {"status": "ok"},
    )
    assert reranker_service.main(["status", "--json"]) == 0
    assert reranker_service.main(["health", "--json"]) == 0

    monkeypatch.setattr(burn_monitor, "okf_startup_status", _boom)
    monkeypatch.setattr(burn_monitor, "self_test", lambda: None)
    assert burn_monitor.main(["--self-test"]) == 0

    monkeypatch.setattr(classification_library_pilot, "okf_startup_status", _boom)
    monkeypatch.setattr(
        classification_library_pilot,
        "load_state",
        lambda _root: {"status": "ok"},
    )
    assert classification_library_pilot.main(["status", "--root", str(tmp_path)]) == 0

    monkeypatch.setattr(recall_runtime, "okf_startup_status", _boom)
    monkeypatch.setattr(recall_runtime, "recent_recall_logs", lambda _limit: [])
    assert recall_runtime.main(["--recent", "1"]) == 0


def test_existing_server_gates_precede_bind(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runs = 0

    def run_server(*_args: object, **_kwargs: object) -> None:
        nonlocal runs
        runs += 1

    monkeypatch.setattr(server, "init_chronovisor", _boom)
    monkeypatch.setattr(server.mcp, "run", run_server)
    with pytest.raises(AssertionError):
        server.main()
    assert runs == 0

    monkeypatch.setattr(dashboard, "init_chronovisor", _boom)
    monkeypatch.setattr(dashboard, "ThreadingHTTPServer", run_server)
    with pytest.raises(AssertionError):
        dashboard.serve("127.0.0.1", 0)
    assert runs == 0
