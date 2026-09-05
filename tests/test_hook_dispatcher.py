from __future__ import annotations

import io
import json
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from chronovisor.core.durable_state import okf_writer_lock
from chronovisor.core.search import ScoredPage
from chronovisor.hosts import hook_dispatcher
from chronovisor.ops import background_jobs
from chronovisor.recall import recall_breaker, recall_runtime


def test_recall_deadline_api_is_reexported_from_recall_runtime() -> None:
    assert hook_dispatcher.RecallWallClockTimeout is recall_runtime.RecallWallClockTimeout
    assert hook_dispatcher.recall_outer_deadline_ms is recall_runtime.recall_outer_deadline_ms
    assert hook_dispatcher.recall_wall_clock_deadline is recall_runtime.recall_wall_clock_deadline


@pytest.fixture(autouse=True)
def isolate_recall_breaker(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "wiki"
    root.mkdir()
    for name in ("index.md", "log.md", "schema.md"):
        (root / name).write_text("legacy\n", encoding="utf-8")
    monkeypatch.setattr(recall_breaker, "BREAKER_FILE", tmp_path / "breaker.json")
    monkeypatch.setattr(hook_dispatcher, "CHRONOVISOR_ROOT", root)
    monkeypatch.setattr(
        hook_dispatcher,
        "okf_startup_status",
        lambda _root: SimpleNamespace(allowed=True, layout="legacy"),
    )
    monkeypatch.setattr(hook_dispatcher, "init_chronovisor", lambda: None)


def _isolate_background_jobs(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(background_jobs, "JOB_DIR", tmp_path / "jobs")
    monkeypatch.setattr(background_jobs, "STATE_FILE", tmp_path / "jobs" / "state.json")
    monkeypatch.setattr(background_jobs, "LOCK_FILE", tmp_path / "jobs" / "state.lock")
    monkeypatch.setattr(hook_dispatcher, "init_chronovisor", lambda: None)


def test_user_prompt_fails_open_without_reading_stdin_when_writer_is_exclusive(
    tmp_path: Path,
) -> None:
    root = tmp_path / "wiki"
    root.mkdir(exist_ok=True)
    for name in ("index.md", "log.md", "schema.md"):
        (root / name).write_text("legacy\n", encoding="utf-8")
    script = """
import sys
from pathlib import Path
from types import SimpleNamespace
from chronovisor.hosts import hook_dispatcher
hook_dispatcher.CHRONOVISOR_ROOT = Path(sys.argv[1])
hook_dispatcher.recall_enabled = lambda: True
hook_dispatcher.load_hook_policy = lambda _path: SimpleNamespace(user_prompt_recall=True)
hook_dispatcher.run_user_prompt = lambda *_args: (_ for _ in ()).throw(RuntimeError("called"))
class NoRead:
    def read(self):
        raise RuntimeError("stdin read")
hook_dispatcher.sys.stdin = NoRead()
raise SystemExit(hook_dispatcher.main(["--host", "codex", "--event", "UserPromptSubmit", "--hook"]))
"""

    with okf_writer_lock(root, exclusive=True):
        result = subprocess.run(
            [sys.executable, "-c", script, str(root)],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )

    assert result.returncode == 0
    assert result.stdout == "{}\n"
    assert result.stderr == ""


def test_user_prompt_fails_open_for_unsafe_runtime_lease(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = tmp_path / "unsafe-wiki"
    root.mkdir()
    for name in ("index.md", "log.md", "schema.md"):
        (root / name).write_text("legacy\n", encoding="utf-8")
    victim = tmp_path / "victim"
    victim.mkdir()
    (root / "runtime").symlink_to(victim, target_is_directory=True)
    monkeypatch.setattr(hook_dispatcher, "CHRONOVISOR_ROOT", root)
    monkeypatch.setattr(hook_dispatcher, "recall_enabled", lambda: True)
    monkeypatch.setattr(
        hook_dispatcher,
        "load_hook_policy",
        lambda _path: SimpleNamespace(user_prompt_recall=True),
    )
    monkeypatch.setattr(
        hook_dispatcher,
        "run_user_prompt",
        lambda *_args: pytest.fail("unsafe lease must fail open before recall"),
    )
    monkeypatch.setattr(
        "sys.stdin",
        SimpleNamespace(read=lambda: pytest.fail("unsafe lease must not read stdin")),
    )

    assert (
        hook_dispatcher.main(
            ["--host", "codex", "--event", "UserPromptSubmit", "--hook"]
        )
        == 0
    )
    assert capsys.readouterr().out == "{}\n"
    assert not (victim / "okf-writer.lock").exists()


def test_user_prompt_dispatches_to_recall_runtime(monkeypatch, capsys) -> None:
    seen: dict[str, object] = {}

    def fake_run_recall(request, policy, *, perform_search: bool, _telemetry):
        seen["request"] = request
        seen["perform_search"] = perform_search
        return recall_runtime.RecallResult(
            status="ok",
            decision="none",
            confidence=0.1,
            queries=[],
            reasons=["test"],
            matched_terms={},
        )

    monkeypatch.setattr(recall_runtime, "run_recall", fake_run_recall)
    monkeypatch.setattr(
        recall_runtime,
        "load_policy",
        lambda _path: recall_runtime.RecallPolicy(log_decisions=False),
    )
    monkeypatch.setattr(
        "sys.stdin",
        io.StringIO(
            json.dumps({"prompt": "今日暑いな", "cwd": "/repo"}, ensure_ascii=False)
        ),
    )

    assert (
        hook_dispatcher.main(
            ["--host", "codex", "--event", "UserPromptSubmit", "--hook"]
        )
        == 0
    )

    output = capsys.readouterr().out.strip()
    assert output == "{}"
    request = seen["request"]
    assert request.host == "codex"
    assert request.prompt == "今日暑いな"
    assert seen["perform_search"] is True


def test_user_prompt_logs_degraded_result_after_render(monkeypatch, capsys) -> None:
    events: list[str] = []
    recorded: list[recall_runtime.RecallResult] = []
    monkeypatch.setattr(
        recall_runtime,
        "load_policy",
        lambda _path: recall_runtime.RecallPolicy(log_decisions=True),
    )
    monkeypatch.setattr(
        recall_runtime,
        "run_recall",
        lambda *_a, **_k: recall_runtime.RecallResult(
            status="degraded",
            decision="none",
            confidence=0.0,
            queries=[],
            reasons=[],
            matched_terms={},
            search_mode="bm25-fallback",
            evidence_features={"authority": "teacher"},
        ),
    )
    monkeypatch.setattr(
        recall_runtime,
        "render_output",
        lambda *_a, **_k: (events.append("render") or "{}"),
    )
    monkeypatch.setattr(
        recall_runtime,
        "append_recall_log",
        lambda _request, result: (recorded.append(result), events.append(f"log:{result.status}")),
    )
    monkeypatch.setattr("sys.stdin", io.StringIO('{"prompt":"remember"}'))

    assert hook_dispatcher.main(
        ["--host", "codex", "--event", "UserPromptSubmit", "--hook"]
    ) == 0
    assert capsys.readouterr().out == "{}\n"
    assert events == ["render", "log:degraded"]
    features = recorded[0].evidence_features
    assert features["host"] == "codex"
    assert features["policy_enabled"] is True
    assert features["semantic_configured"] is True
    assert features["perform_search"] is True
    assert features["semantic_eligible_before_breaker"] is True
    assert features["breaker_open"] is False
    assert features["normalized_prompt_hash"] == recall_runtime.stable_prompt_hash(
        "remember"
    )
    assert features["normalized_prompt_chars"] == len("remember")
    assert features["dispatcher_wall_ms"] >= 0
    assert (
        features["dispatcher_wall_scope"]
        == "run_user_prompt_start_to_render_flush_excluding_uvx_process_startup"
    )
    assert "remember" not in json.dumps(features)


@pytest.mark.parametrize("status", ["ok", "degraded"])
def test_user_prompt_owns_exactly_once_log_even_when_append_raises(
    monkeypatch,
    capsys,
    status: str,
) -> None:
    events: list[str] = []
    monkeypatch.setattr(
        recall_runtime,
        "load_policy",
        lambda _path: recall_runtime.RecallPolicy(log_decisions=True),
    )

    def run(_request, policy, **_kwargs):
        assert policy.log_decisions is False
        return recall_runtime.RecallResult(
            status=status,
            decision="none",
            confidence=0.0,
            queries=[],
            reasons=[],
            matched_terms={},
        )

    def append(_request, _result):
        events.append("append")
        raise RuntimeError("append completed then raised")

    monkeypatch.setattr(recall_runtime, "run_recall", run)
    monkeypatch.setattr(
        recall_runtime,
        "render_output",
        lambda *_a, **_k: (events.append("render") or "{}"),
    )
    monkeypatch.setattr(recall_runtime, "append_recall_log", append)
    monkeypatch.setattr("sys.stdin", io.StringIO('{"prompt":"remember"}'))

    assert hook_dispatcher.main(
        ["--host", "codex", "--event", "UserPromptSubmit", "--hook"]
    ) == 0
    assert capsys.readouterr().out == "{}\n"
    assert events == ["render", "append"]


def test_user_prompt_log_merges_conditional_pipeline_and_finalize_timings(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    log_file = tmp_path / "recall-log.jsonl"
    policy = recall_runtime.RecallPolicy(
        judge_mode="off",
        rewrite_enabled=False,
        processor_shadow_enabled=False,
        log_decisions=True,
    )
    monkeypatch.setattr(recall_runtime, "RECALL_LOG_FILE", log_file)
    monkeypatch.setattr(recall_runtime, "load_policy", lambda _path: policy)

    def evidence_search(*, deadline_at, _telemetry, **_kwargs):
        for stage in ("cleanup", "session_load", "teacher", "reranker"):
            recall_runtime._stage_started(_telemetry, stage, deadline_at)
            recall_runtime._stage_completed(_telemetry, stage, deadline_at)
        return recall_runtime._EvidenceSearchOutcome(
            score=1.0,
            session_state=None,
            pre_results=[ScoredPage("page", "Page", "", "", 1.0)],
            search_mode="hybrid",
            evidence_features={
                "stage_timings_ms": {
                    "bm25_query": 4,
                    "semantic": 5,
                    "graph": 6,
                }
            },
            rewrite_queries=[],
            reranker_metadata={"status": "disabled", "mode": "off"},
            field_shadow_metadata={},
            post_authority={},
        )

    monkeypatch.setattr(recall_runtime, "_run_evidence_search", evidence_search)
    monkeypatch.setattr(
        recall_runtime,
        "collect_context",
        lambda *_a, **_k: [
            recall_runtime.ContextItem("page", "Page", "", 1.0)
        ],
    )
    monkeypatch.setattr(recall_runtime, "state_context_for_request", lambda *_a: "")
    monkeypatch.setattr(
        recall_runtime,
        "observe_evidence_reconstruction",
        lambda *_a, **_k: {"status": "skipped"},
    )
    monkeypatch.setattr(
        "chronovisor.recall.recall_policy_store.append_live_episode",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr("sys.stdin", io.StringIO('{"prompt":"前回の続き"}'))

    assert hook_dispatcher.main(
        ["--host", "codex", "--event", "UserPromptSubmit", "--hook"]
    ) == 0
    assert capsys.readouterr().out
    record = json.loads(log_file.read_text(encoding="utf-8").splitlines()[-1])
    timings = record["stage_timings_ms"]
    assert {
        "scheduler",
        "prepare",
        "evidence_search",
        "cleanup",
        "session_load",
        "teacher",
        "reranker",
        "context",
        "evidence_reconstruction",
        "finalize",
        "bm25_query",
        "semantic",
        "graph",
    } <= set(timings)
    assert all(isinstance(value, int) and value >= 0 for value in timings.values())
    assert "finalize" in record["evidence_features"]["stage_timings_ms"]


def test_user_prompt_unexpected_failure_is_exit_zero_fail_open(
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(hook_dispatcher, "init_chronovisor", lambda: None)
    monkeypatch.setattr(
        recall_runtime,
        "load_policy",
        lambda _path: recall_runtime.RecallPolicy(log_decisions=False),
    )
    monkeypatch.setattr(
        recall_runtime,
        "run_recall",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("offline")),
    )
    monkeypatch.setattr(
        "sys.stdin", io.StringIO('{"prompt":"remember","thread_id":"t1"}')
    )

    assert (
        hook_dispatcher.main(
            ["--host", "codex", "--event", "UserPromptSubmit", "--hook"]
        )
        == 0
    )
    assert capsys.readouterr().out.strip() == "{}"
    assert recall_breaker.snapshot()["failures"] == 1


def test_user_prompt_outer_timeout_does_not_start_second_fallback(
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(hook_dispatcher, "init_chronovisor", lambda: None)
    monkeypatch.setattr(
        recall_runtime,
        "load_policy",
        lambda _path: recall_runtime.RecallPolicy(log_decisions=False),
    )
    monkeypatch.setattr(
        recall_runtime,
        "run_recall",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            hook_dispatcher.RecallWallClockTimeout("primary timeout")
        ),
    )
    fallback_calls: list[bool] = []
    monkeypatch.setattr(
        recall_runtime,
        "run_deterministic_fallback",
        lambda *_args, **_kwargs: fallback_calls.append(True),
    )
    monkeypatch.setattr("sys.stdin", io.StringIO('{"prompt":"前回の続き"}'))

    assert (
        hook_dispatcher.main(
            ["--host", "codex", "--event", "UserPromptSubmit", "--hook"]
        )
        == 0
    )
    assert capsys.readouterr().out.strip() == "{}"
    assert fallback_calls == []
    assert recall_breaker.snapshot()["failures"] == 1


def test_user_prompt_outer_timeout_logs_anonymous_stage_telemetry(
    monkeypatch,
    capsys,
) -> None:
    recorded: list[recall_runtime.RecallResult] = []
    monkeypatch.setattr(
        recall_runtime,
        "load_policy",
        lambda _path: recall_runtime.RecallPolicy(log_decisions=True),
    )

    def timeout(*_args, _telemetry, **_kwargs):
        _telemetry.update(
            scheduler_wait_ms=240,
            last_stage_started="context",
            last_stage_completed="judge",
            remaining_ms=75,
            fallback_started=False,
        )
        raise hook_dispatcher.RecallWallClockTimeout("primary timeout")

    monkeypatch.setattr(recall_runtime, "run_recall", timeout)
    monkeypatch.setattr(
        recall_runtime,
        "append_recall_log",
        lambda _request, result: recorded.append(result),
    )
    monkeypatch.setattr("sys.stdin", io.StringIO('{"prompt":"private prompt"}'))

    assert (
        hook_dispatcher.main(
            ["--host", "codex", "--event", "UserPromptSubmit", "--hook"]
        )
        == 0
    )
    assert capsys.readouterr().out.strip() == "{}"
    assert recorded[0].evidence_features == {
        "host": "codex",
        "policy_enabled": True,
        "semantic_configured": True,
        "perform_search": True,
        "semantic_eligible_before_breaker": True,
        "breaker_open": False,
        "normalized_prompt_hash": recall_runtime.stable_prompt_hash(
            "private prompt"
        ),
        "normalized_prompt_chars": len("private prompt"),
        "scheduler_wait_ms": 240,
        "last_stage_started": "context",
        "last_stage_completed": "judge",
        "remaining_ms": 75,
        "fallback_started": False,
        "dispatcher_wall_ms": recorded[0].evidence_features["dispatcher_wall_ms"],
        "dispatcher_wall_scope": "run_user_prompt_start_to_render_flush_excluding_uvx_process_startup",
    }
    assert recorded[0].evidence_features["dispatcher_wall_ms"] >= 0
    assert recorded[0].latency_ms == 4_000
    assert "private prompt" not in json.dumps(recorded[0].evidence_features)


def test_outer_mid_context_timeout_logs_partial_stage_exactly_once(
    monkeypatch,
    capsys,
) -> None:
    recorded: list[recall_runtime.RecallResult] = []
    monkeypatch.setattr(
        recall_runtime,
        "load_policy",
        lambda _path: recall_runtime.RecallPolicy(log_decisions=True),
    )

    def interrupt(*_args, _telemetry, _final_deadline_at, **_kwargs):
        recall_runtime._stage_started(_telemetry, "context", _final_deadline_at)
        time.sleep(0.002)
        raise hook_dispatcher.RecallWallClockTimeout("outer timeout")

    monkeypatch.setattr(recall_runtime, "_run_recall_impl", interrupt)
    monkeypatch.setattr(
        recall_runtime,
        "append_recall_log",
        lambda _request, result: recorded.append(result),
    )
    monkeypatch.setattr("sys.stdin", io.StringIO('{"prompt":"private"}'))

    assert hook_dispatcher.main(
        ["--host", "codex", "--event", "UserPromptSubmit", "--hook"]
    ) == 0
    assert capsys.readouterr().out == "{}\n"
    assert len(recorded) == 1
    assert recorded[0].status == "timeout"
    timings = recorded[0].evidence_features["stage_timings_ms"]
    assert timings["context"] >= 1


def test_outer_recall_deadline_is_the_total_budget() -> None:
    policy = recall_runtime.RecallPolicy(
        total_timeout_ms=4_000,
        deterministic_fallback_reserve_ms=600,
    )

    assert hook_dispatcher.recall_outer_deadline_ms(policy) == 4_000
    assert hook_dispatcher.recall_inner_budget_ms(policy) == 3_750


def test_user_prompt_reserves_host_headroom_inside_total_deadline(
    monkeypatch,
    capsys,
) -> None:
    seen: dict[str, int] = {}

    monkeypatch.setattr(hook_dispatcher, "init_chronovisor", lambda: None)
    monkeypatch.setattr(
        recall_runtime,
        "load_policy",
        lambda _path: recall_runtime.RecallPolicy(
            total_timeout_ms=4_000,
            deterministic_fallback_reserve_ms=600,
            log_decisions=False,
        ),
    )
    monkeypatch.setattr(recall_breaker, "is_open", lambda: False)

    def fake_deadline(timeout_ms: int):
        class Deadline:
            def __enter__(self):
                seen["outer_timeout_ms"] = timeout_ms

            def __exit__(self, *_args):
                return False

        return Deadline()

    def fake_run(_request, policy, *, perform_search: bool, _telemetry):
        seen["total_timeout_ms"] = policy.total_timeout_ms
        seen["fallback_reserve_ms"] = policy.deterministic_fallback_reserve_ms
        return recall_runtime.RecallResult(
            status="ok",
            decision="none",
            confidence=0.0,
            queries=[],
            reasons=[],
            matched_terms={},
        )

    monkeypatch.setattr(hook_dispatcher, "recall_wall_clock_deadline", fake_deadline)
    monkeypatch.setattr(recall_runtime, "run_recall", fake_run)
    monkeypatch.setattr("sys.stdin", io.StringIO('{"prompt":"remember"}'))

    assert (
        hook_dispatcher.main(
            ["--host", "codex", "--event", "UserPromptSubmit", "--hook"]
        )
        == 0
    )
    assert capsys.readouterr().out.strip() == "{}"
    assert seen == {
        "outer_timeout_ms": 4_000,
        "total_timeout_ms": 3_750,
        "fallback_reserve_ms": 600,
    }


def test_user_prompt_open_breaker_uses_bm25_only_policy(monkeypatch, capsys) -> None:
    seen: dict[str, object] = {}

    def fake_run(_request, policy, *, perform_search: bool, _telemetry):
        seen.update(
            semantic=policy.semantic,
            judge_mode=policy.judge_mode,
            rewrite_enabled=policy.rewrite_enabled,
            perform_search=perform_search,
        )
        return recall_runtime.RecallResult(
            status="ok",
            decision="none",
            confidence=0.0,
            queries=[],
            reasons=[],
            matched_terms={},
        )

    monkeypatch.setattr(hook_dispatcher, "init_chronovisor", lambda: None)
    monkeypatch.setattr(recall_breaker, "is_open", lambda: True)
    monkeypatch.setattr(recall_runtime, "run_recall", fake_run)
    monkeypatch.setattr(
        recall_runtime,
        "load_policy",
        lambda _path: recall_runtime.RecallPolicy(log_decisions=False),
    )
    monkeypatch.setattr("sys.stdin", io.StringIO('{"prompt":"remember"}'))

    assert (
        hook_dispatcher.main(
            ["--host", "codex", "--event", "UserPromptSubmit", "--hook"]
        )
        == 0
    )
    assert capsys.readouterr().out.strip() == "{}"
    assert seen == {
        "semantic": False,
        "judge_mode": "off",
        "rewrite_enabled": False,
        "perform_search": True,
    }


def test_user_prompt_open_breaker_keeps_semantic_denominator_and_anonymous_metadata(
    monkeypatch, capsys
) -> None:
    recorded: list[recall_runtime.RecallResult] = []
    monkeypatch.setattr(hook_dispatcher, "init_chronovisor", lambda: None)
    monkeypatch.setattr(recall_breaker, "is_open", lambda: True)
    monkeypatch.setattr(
        recall_runtime,
        "load_policy",
        lambda _path: recall_runtime.RecallPolicy(log_decisions=True, semantic=True),
    )
    monkeypatch.setattr(
        recall_runtime,
        "run_recall",
        lambda *_args, **_kwargs: recall_runtime.RecallResult(
            status="ok",
            decision="none",
            confidence=0.0,
            queries=[],
            reasons=[],
            search_mode="bm25",
        ),
    )
    monkeypatch.setattr(
        recall_runtime, "render_output", lambda *_args, **_kwargs: "{}"
    )
    monkeypatch.setattr(
        recall_runtime,
        "append_recall_log",
        lambda _request, result: recorded.append(result),
    )

    assert hook_dispatcher.main(
        [
            "--host",
            "codex",
            "--event",
            "UserPromptSubmit",
            "--prompt",
            "  remember\n  this  ",
        ]
    ) == 0
    assert capsys.readouterr().out == "{}\n"

    features = recorded[0].evidence_features
    assert features["policy_enabled"] is True
    assert features["semantic_configured"] is True
    assert features["perform_search"] is True
    assert features["semantic_eligible_before_breaker"] is True
    assert features["breaker_open"] is True
    assert features["normalized_prompt_hash"] == recall_runtime.stable_prompt_hash(
        "  remember\n  this  "
    )
    assert features["normalized_prompt_chars"] == len("remember this")
    assert "remember" not in json.dumps(features)


def test_user_prompt_policy_failure_is_still_exit_zero_fail_open(
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(hook_dispatcher, "init_chronovisor", lambda: None)
    monkeypatch.setattr(
        recall_runtime,
        "load_policy",
        lambda _path: (_ for _ in ()).throw(PermissionError("config unavailable")),
    )
    monkeypatch.setattr("sys.stdin", io.StringIO('{"prompt":"remember"}'))

    assert (
        hook_dispatcher.main(
            ["--host", "codex", "--event", "UserPromptSubmit", "--hook"]
        )
        == 0
    )
    assert capsys.readouterr().out.strip() == "{}"


def test_user_prompt_init_failure_is_still_exit_zero_fail_open(
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(
        hook_dispatcher,
        "init_chronovisor",
        lambda: (_ for _ in ()).throw(PermissionError("wiki unavailable")),
    )
    monkeypatch.setattr("sys.stdin", io.StringIO('{"prompt":"remember"}'))

    assert (
        hook_dispatcher.main(
            ["--host", "codex", "--event", "UserPromptSubmit", "--hook"]
        )
        == 0
    )
    assert capsys.readouterr().out.strip() == "{}"


def test_user_prompt_skips_init_only_for_allowed_okf_v0_2(
    monkeypatch,
) -> None:
    startup = SimpleNamespace(allowed=True, layout="okf_v0_2")
    init_calls: list[None] = []
    monkeypatch.setattr(hook_dispatcher, "okf_startup_status", lambda _root: startup)
    monkeypatch.setattr(
        hook_dispatcher, "init_chronovisor", lambda: init_calls.append(None)
    )
    monkeypatch.setattr(hook_dispatcher, "run_user_prompt", lambda *_args: 0)

    argv = ["--host", "codex", "--event", "UserPromptSubmit"]
    assert hook_dispatcher._main_locked(argv) == 0
    assert init_calls == []

    startup.layout = "legacy"
    assert hook_dispatcher._main_locked(argv) == 0
    assert init_calls == [None]


def test_stop_dispatch_enqueues_save_and_receipt_audit(
    monkeypatch, tmp_path, capsys
) -> None:
    config = tmp_path / "config.toml"
    config.write_text(
        "[hooks.stop]\nsave = true\naudit = true\ncontent_correction = false\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("CODEX_CHRONOVISOR_RECORD_ENABLED", "1")
    monkeypatch.setenv("CHRONOVISOR_RECALL_AUDIT_ENABLED", "1")
    monkeypatch.setattr("sys.stdin", io.StringIO("{}"))

    assert (
        hook_dispatcher.main(
            [
                "--host",
                "codex",
                "--event",
                "Stop",
                "--hook",
                "--config",
                str(config),
                "--dry-run",
                "--format",
                "json",
            ]
        )
        == 0
    )
    output = json.loads(capsys.readouterr().out)

    assert [task["name"] for task in output["tasks"]] == ["codex-save"]
    save = output["tasks"][0]
    assert save["module"] == "chronovisor.hosts.codex_record"
    assert save["args"] == ["--hook", "--save"]
    assert "--trigger-ingest" not in save["args"]
    assert save["on_success"] == [
        {
            "name": "recall-audit-candidate",
            "module": "chronovisor.recall.recall_auditor",
            "args": ["--host", "codex", "--hook"],
            "env": {},
            "when_output_status": "saved",
        },
        {
            "name": "recall-answer-capture",
            "module": "chronovisor.recall.recall_answer_eval",
            "args": ["--host", "codex", "--hook", "--capture-only"],
            "env": {"CHRONOVISOR_RECALL_ANSWER_CAPTURE_ENABLED": "1"},
            "when_output_statuses": ["saved", "recovered"],
            "stdin_from_output": True,
        },
    ]


def test_stop_dispatch_supports_hermes_save(monkeypatch, tmp_path, capsys) -> None:
    config = tmp_path / "config.toml"
    config.write_text(
        "[hooks.stop]\nsave = true\naudit = false\ncontent_correction = false\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_CHRONOVISOR_RECORD_ENABLED", "1")
    monkeypatch.setattr("sys.stdin", io.StringIO("{}"))

    assert hook_dispatcher.main(
        [
            "--host", "hermes", "--event", "Stop", "--hook",
            "--config", str(config), "--dry-run", "--format", "json",
        ]
    ) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["tasks"] == [
        {
            "name": "hermes-save",
            "module": "chronovisor.hosts.hermes_record",
            "args": ["--hook", "--save"],
            "dry_run": True,
        }
    ]


def test_stop_dispatch_full_entrypoint_enqueues_only_capture_work(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    config = tmp_path / "config.toml"
    config.write_text(
        "[hooks.stop]\n"
        "save = true\n"
        "audit = true\n"
        "content_correction = true\n"
        "recall_improve = true\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("CLAUDE_CODE_CHRONOVISOR_RECORD_ENABLED", "1")
    monkeypatch.setenv("CHRONOVISOR_RECALL_AUDIT_ENABLED", "1")
    monkeypatch.setenv("CHRONOVISOR_CONTENT_CORRECTION_ENABLED", "1")
    monkeypatch.setenv("CHRONOVISOR_RECALL_IMPROVE_ENABLED", "1")
    monkeypatch.setattr("sys.stdin", io.StringIO("{}"))

    assert (
        hook_dispatcher.main(
            [
                "--host",
                "claude-code",
                "--event",
                "Stop",
                "--hook",
                "--config",
                str(config),
                "--dry-run",
                "--format",
                "json",
            ]
        )
        == 0
    )
    output = json.loads(capsys.readouterr().out)

    assert [task["name"] for task in output["tasks"]] == [
        "claude-code-save",
        "content-correction-capture",
    ]
    assert output["tasks"][0]["args"] == ["--hook", "--save"]
    correction = output["tasks"][1]
    assert correction["module"] == "chronovisor.recall.content_correction"
    assert correction["args"] == [
        "--host",
        "claude-code",
        "--hook",
        "--capture-only",
    ]


def test_stop_dispatch_content_correction_uses_capture_only_worker(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    config = tmp_path / "config.toml"
    config.write_text(
        "[hooks.stop]\nsave = false\ncontent_correction = true\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("CHRONOVISOR_CONTENT_CORRECTION_ENABLED", "1")
    monkeypatch.setattr("sys.stdin", io.StringIO("{}"))

    assert (
        hook_dispatcher.main(
            [
                "--host",
                "codex",
                "--event",
                "Stop",
                "--hook",
                "--config",
                str(config),
                "--dry-run",
                "--format",
                "json",
            ]
        )
        == 0
    )
    output = json.loads(capsys.readouterr().out)

    assert output["tasks"] == [
        {
            "name": "content-correction-capture",
            "module": "chronovisor.recall.content_correction",
            "args": ["--host", "codex", "--hook", "--capture-only"],
            "dry_run": True,
        }
    ]


def test_stop_content_correction_false_enqueues_nothing(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    config = tmp_path / "config.toml"
    config.write_text(
        "[hooks.stop]\nsave = false\ncontent_correction = false\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("CHRONOVISOR_CONTENT_CORRECTION_ENABLED", "1")
    monkeypatch.setattr("sys.stdin", io.StringIO('{"session_id":"session-1"}'))

    assert (
        hook_dispatcher.main(
            [
                "--host",
                "codex",
                "--event",
                "Stop",
                "--hook",
                "--config",
                str(config),
                "--format",
                "json",
            ]
        )
        == 0
    )

    assert json.loads(capsys.readouterr().out)["tasks"] == []


def test_stop_capture_enqueue_coalesces_and_never_starts_a_subprocess(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    _isolate_background_jobs(monkeypatch, tmp_path)
    config = tmp_path / "config.toml"
    config.write_text(
        "[hooks.stop]\nsave = false\ncontent_correction = true\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("CHRONOVISOR_CONTENT_CORRECTION_ENABLED", "1")
    monkeypatch.setattr(
        background_jobs.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("Stop must not run a subprocess"),
    )
    monkeypatch.setattr(
        background_jobs.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("Stop must not spawn a subprocess"),
    )
    session_id = "11111111-2222-3333-4444-555555555555"
    session_file = tmp_path / "session.jsonl"
    argv = [
        "--host",
        "codex",
        "--event",
        "Stop",
        "--hook",
        "--config",
        str(config),
        "--format",
        "json",
    ]

    monkeypatch.setattr(
        "sys.stdin",
        io.StringIO(
            json.dumps(
                {"session_id": session_id, "session_file": str(session_file), "turn": 1}
            )
        ),
    )
    assert hook_dispatcher.main(argv) == 0
    first = json.loads(capsys.readouterr().out)["tasks"][0]

    monkeypatch.setattr(
        "sys.stdin",
        io.StringIO(
            json.dumps(
                {"session_id": session_id, "session_file": str(session_file), "turn": 2}
            )
        ),
    )
    assert hook_dispatcher.main(argv) == 0
    second = json.loads(capsys.readouterr().out)["tasks"][0]

    assert first["name"] == "content-correction-capture"
    assert first["enqueued"] is True
    assert second["job_id"] == first["job_id"]
    assert second["enqueued"] is False
    assert second["coalesced"] is True
    state = json.loads(background_jobs.STATE_FILE.read_text(encoding="utf-8"))
    assert len(state["jobs"]) == 1
    stored = state["jobs"][first["job_id"]]
    assert stored["lane_key"] == "content-correction-capture"
    assert stored["module"] == "chronovisor.recall.content_correction"
    assert stored["args"] == ["--host", "codex", "--hook", "--capture-only"]
    assert json.loads(stored["stdin"])["turn"] == 2


def test_spawn_task_only_enqueues_without_process(monkeypatch) -> None:
    seen: dict[str, object] = {}

    def fake_enqueue_job(**kwargs):
        seen.update(kwargs)
        return {
            "job_id": "job-1",
            "status": "queued",
            "enqueued": True,
            "coalesced": False,
        }

    monkeypatch.setattr(background_jobs, "enqueue_job", fake_enqueue_job)
    task = hook_dispatcher.BackgroundTask(
        name="codex-save",
        module="chronovisor.hosts.codex_record",
        args=["--hook", "--save"],
        env={"CODEX_CHRONOVISOR_RECORD_ENABLED": "1"},
    )

    result = hook_dispatcher.spawn_task(task, '{"session_id":"session-1"}')

    assert result == {
        "job_id": "job-1",
        "status": "queued",
        "enqueued": True,
        "coalesced": False,
    }
    assert seen["stdin_text"] == '{"session_id":"session-1"}'


def test_stop_dispatch_requires_env_for_noncanonical_config_override(
    monkeypatch, tmp_path, capsys
) -> None:
    override_config = tmp_path / "override.toml"
    override_config.write_text("enabled = true\n", encoding="utf-8")
    monkeypatch.delenv("CODEX_CHRONOVISOR_RECORD_ENABLED", raising=False)
    monkeypatch.delenv("CHRONOVISOR_RECALL_AUDIT_ENABLED", raising=False)
    monkeypatch.delenv("CHRONOVISOR_RECALL_IMPROVE_ENABLED", raising=False)
    monkeypatch.delenv("CHRONOVISOR_CONTENT_CORRECTION_ENABLED", raising=False)
    monkeypatch.setattr("sys.stdin", io.StringIO("{}"))

    assert (
        hook_dispatcher.main(
            [
                "--host",
                "codex",
                "--event",
                "Stop",
                "--hook",
                "--config",
                str(override_config),
                "--dry-run",
                "--format",
                "json",
            ]
        )
        == 0
    )
    output = json.loads(capsys.readouterr().out)

    assert output["tasks"] == []


def test_internal_frontier_stop_never_spawns_tasks(monkeypatch, capsys) -> None:
    monkeypatch.setenv("CHRONOVISOR_INTERNAL_FRONTIER", "1")
    monkeypatch.setattr("sys.stdin", io.StringIO("{}"))

    assert (
        hook_dispatcher.main(
            [
                "--host",
                "codex",
                "--event",
                "Stop",
                "--hook",
                "--dry-run",
                "--format",
                "json",
            ]
        )
        == 0
    )

    output = json.loads(capsys.readouterr().out)
    assert output == {
        "status": "suppressed",
        "reason": "internal_frontier",
        "tasks": [],
    }
