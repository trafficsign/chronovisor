"""Single host hook dispatcher for Chronovisor.

Host-specific shell scripts should be thin wrappers around this entry point.
The dispatcher owns event routing; recall/save/audit modules keep the domain
logic.
"""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import suppress
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from chronovisor.core.runtime_config import (
    active_config_file,
    env_flag,
    load_hook_policy,
)
from chronovisor.core.store import (
    CHRONOVISOR_ROOT,
    init_chronovisor,
    okf_runtime_operation,
    okf_startup_status,
)
from chronovisor.hosts import evidence_composition
from chronovisor.recall import recall_runtime

LOG_DIR = CHRONOVISOR_ROOT / "logs"
RECALL_HOST_HEADROOM_MS = 250

HOSTS = {"codex", "claude-code", "pi", "generic"}
USER_PROMPT_EVENTS = {"user-prompt-submit", "userpromptsubmit", "prompt-submit"}
STOP_EVENTS = {"stop"}


RecallWallClockTimeout = recall_runtime.RecallWallClockTimeout
recall_outer_deadline_ms = recall_runtime.recall_outer_deadline_ms
recall_wall_clock_deadline = recall_runtime.recall_wall_clock_deadline


def recall_inner_budget_ms(policy: recall_runtime.RecallPolicy) -> int:
    """Reserve process/render headroom inside the configured host deadline."""

    return max(500, int(policy.total_timeout_ms) - RECALL_HOST_HEADROOM_MS)


@dataclass(frozen=True)
class BackgroundTask:
    name: str
    module: str
    args: list[str]
    env: dict[str, str] = field(default_factory=dict)
    log_prefix: str = "hook"
    on_success: list[dict[str, Any]] = field(default_factory=list)


def normalize_host(value: str) -> str:
    host = value.strip().lower().replace("_", "-")
    if host == "claude":
        host = "claude-code"
    if host not in HOSTS:
        raise ValueError(f"unsupported host: {value}")
    return host


def normalize_event(value: str) -> str:
    event = value.strip().lower().replace("_", "-")
    if event in USER_PROMPT_EVENTS:
        return "user-prompt-submit"
    if event in STOP_EVENTS:
        return "stop"
    raise ValueError(f"unsupported hook event: {value}")


def host_output_format(host: str) -> str:
    if host == "claude-code":
        return "claude"
    if host == "codex":
        return "codex"
    return "json"


def read_hook_payload(stdin_text: str) -> dict[str, Any]:
    if not stdin_text.strip():
        return {}
    try:
        parsed = json.loads(stdin_text)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _print_host_noop(host: str) -> None:
    if host == "codex":
        print("{}")


def recall_enabled() -> bool:
    flag = env_flag("CHRONOVISOR_RECALL_ENABLED")
    return True if flag is None else flag


def save_enabled(host: str, explicit_config: Path | None = None) -> bool:
    policy = load_hook_policy(explicit_config)
    if not policy.stop_save:
        return False
    env_name = (
        "CODEX_CHRONOVISOR_RECORD_ENABLED"
        if host == "codex"
        else "PI_CHRONOVISOR_RECORD_ENABLED"
        if host == "pi"
        else "CLAUDE_CODE_CHRONOVISOR_RECORD_ENABLED"
    )
    flag = env_flag(env_name)
    if flag is not None:
        return flag
    return active_config_file(explicit_config).name == "config.toml"


def audit_enabled(explicit_config: Path | None = None) -> bool:
    policy = load_hook_policy(explicit_config)
    if not policy.stop_audit:
        return False
    flag = env_flag("CHRONOVISOR_RECALL_AUDIT_ENABLED")
    if flag is not None:
        return flag
    return active_config_file(explicit_config).name == "config.toml"


def content_correction_enabled(explicit_config: Path | None = None) -> bool:
    policy = load_hook_policy(explicit_config)
    if not policy.stop_content_correction:
        return False
    flag = env_flag("CHRONOVISOR_CONTENT_CORRECTION_ENABLED")
    if flag is not None:
        return flag
    return active_config_file(explicit_config).name == "config.toml"


def recall_improve_enabled(explicit_config: Path | None = None) -> bool:
    policy = load_hook_policy(explicit_config)
    if not policy.stop_recall_improve:
        return False
    flag = env_flag("CHRONOVISOR_RECALL_IMPROVE_ENABLED")
    if flag is not None:
        return flag
    return active_config_file(explicit_config).name == "config.toml"


def run_user_prompt(args: argparse.Namespace, stdin_text: str) -> int:
    evidence_composition.bind_recall_provider()
    host = normalize_host(args.host)
    if env_flag("CHRONOVISOR_INTERNAL_FRONTIER") is True:
        _print_host_noop(host)
        return 0
    if not recall_enabled() or not load_hook_policy(args.config).user_prompt_recall:
        _print_host_noop(host)
        return 0

    payload = read_hook_payload(stdin_text) if args.hook else {}
    request = (
        recall_runtime.request_from_hook_payload(
            payload, host=host, event="UserPromptSubmit"
        )
        if args.hook
        else recall_runtime.RecallRequest(
            host=host,
            event="UserPromptSubmit",
            prompt=args.prompt or "",
            cwd=args.cwd or "",
            session_id=args.session_id or "",
        )
    )
    if args.prompt:
        request.prompt = args.prompt
    if args.cwd:
        request.cwd = args.cwd
    if args.session_id:
        request.session_id = args.session_id

    policy = recall_runtime.load_policy(active_config_file(args.config))
    from chronovisor.recall import recall_breaker

    breaker_was_open = recall_breaker.is_open()
    effective_policy = replace(
        policy,
        total_timeout_ms=recall_inner_budget_ms(policy),
    )
    if breaker_was_open:
        effective_policy = replace(
            effective_policy,
            semantic=False,
            judge_mode="off",
            rewrite_enabled=False,
        )
    telemetry: dict[str, Any] = {"host": host}
    try:
        with recall_wall_clock_deadline(recall_outer_deadline_ms(policy)):
            result = recall_runtime.run_recall(
                request,
                effective_policy,
                perform_search=not args.no_search,
                _telemetry=telemetry,
            )
    except RecallWallClockTimeout as exc:
        recall_breaker.record_failure(
            str(exc),
            threshold=policy.circuit_breaker_failures,
            cooldown_seconds=policy.circuit_breaker_cooldown_seconds,
        )
        # ``run_recall`` already reserves and runs its deterministic fallback
        # inside this total deadline. Reaching the outer timer means a lower
        # layer ignored its own timeout, so starting any second pass here would
        # violate the host's four-second contract.
        _record_recall_fail_open(
            request,
            policy,
            status="timeout",
            error=str(exc),
            telemetry=telemetry,
        )
        _print_host_noop(host)
        return 0
    except Exception as exc:
        error = f"{exc.__class__.__name__}: {exc}"
        recall_breaker.record_failure(
            error,
            threshold=policy.circuit_breaker_failures,
            cooldown_seconds=policy.circuit_breaker_cooldown_seconds,
        )
        _record_recall_fail_open(
            request,
            policy,
            status="error",
            error=error,
            telemetry=telemetry,
        )
        _print_host_noop(host)
        return 0

    if result.status in {"timeout", "degraded"}:
        recall_breaker.record_failure(
            result.error or "recall soft deadline exhausted",
            threshold=policy.circuit_breaker_failures,
            cooldown_seconds=policy.circuit_breaker_cooldown_seconds,
        )
    elif not breaker_was_open:
        recall_breaker.record_success()
    if breaker_was_open:
        result.reasons.append("circuit breaker open; expensive recall stages disabled")
    output = recall_runtime.render_output(
        result, args.format or host_output_format(host)
    )
    if output:
        print(output)
    return 0


def _record_recall_fail_open(
    request: recall_runtime.RecallRequest,
    policy: recall_runtime.RecallPolicy,
    *,
    status: str,
    error: str,
    telemetry: dict[str, Any] | None = None,
) -> None:
    if not policy.log_decisions:
        return
    with suppress(Exception):
        evidence_features = dict(telemetry or {})
        evidence_features.setdefault("fallback_started", False)
        recall_runtime.append_recall_log(
            request,
            recall_runtime.RecallResult(
                status=status,
                decision="none",
                confidence=0.0,
                queries=[],
                reasons=["synchronous recall failed open"],
                matched_terms={},
                evidence_features=evidence_features,
                latency_ms=policy.total_timeout_ms if status == "timeout" else 0,
                error=error,
            ),
        )


def log_file(prefix: str) -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime("%Y%m%d")
    return LOG_DIR / f"{prefix}-{today}.log"


def spawn_task(task: BackgroundTask, stdin_text: str) -> dict[str, Any]:
    """Durably enqueue hook work without starting a detached process."""
    from chronovisor.core.background_jobs import enqueue_job

    job = enqueue_job(
        name=task.name,
        module=task.module,
        args=task.args,
        env=task.env,
        stdin_text=stdin_text,
        on_success=task.on_success,
    )
    return {
        "job_id": job["job_id"],
        "status": job.get("status"),
        "enqueued": bool(job.get("enqueued", True)),
        "coalesced": bool(job.get("coalesced", False)),
    }


def stop_tasks(host: str, args: argparse.Namespace) -> list[BackgroundTask]:
    config = active_config_file(args.config)
    tasks: list[BackgroundTask] = []
    if save_enabled(host, config):
        if host == "codex":
            tasks.append(
                BackgroundTask(
                    name="codex-save",
                    module="chronovisor.hosts.codex_record",
                    args=["--hook", "--save"],
                    env={"CODEX_CHRONOVISOR_RECORD_ENABLED": "1"},
                    log_prefix="codex-save",
                    on_success=[
                        {
                            "name": "recall-audit-candidate",
                            "module": "chronovisor.recall.recall_auditor",
                            "args": ["--host", "codex", "--hook"],
                            "env": {},
                            "when_output_status": "saved",
                        },
                        {
                            # Capture only immutable answer references after the
                            # host save has reported durable publication. Replay
                            # and scoring remain sleep/offline work.
                            "name": "recall-answer-capture",
                            "module": "chronovisor.recall.recall_answer_eval",
                            "args": ["--host", "codex", "--hook", "--capture-only"],
                            "env": {
                                "CHRONOVISOR_RECALL_ANSWER_CAPTURE_ENABLED": "1"
                            },
                            "when_output_statuses": ["saved", "recovered"],
                            "stdin_from_output": True,
                        }
                    ],
                )
            )
        elif host == "claude-code":
            tasks.append(
                BackgroundTask(
                    name="claude-code-save",
                    module="chronovisor.hosts.claude_code_record",
                    args=["--hook", "--save"],
                    env={"CLAUDE_CODE_CHRONOVISOR_RECORD_ENABLED": "1"},
                    log_prefix="claude-code-save",
                    on_success=[
                        {
                            "name": "recall-audit-candidate",
                            "module": "chronovisor.recall.recall_auditor",
                            "args": ["--host", "claude-code", "--hook"],
                            "env": {},
                            "when_output_status": "saved",
                        },
                        {
                            "name": "recall-answer-capture",
                            "module": "chronovisor.recall.recall_answer_eval",
                            "args": [
                                "--host",
                                "claude-code",
                                "--hook",
                                "--capture-only",
                            ],
                            "env": {
                                "CHRONOVISOR_RECALL_ANSWER_CAPTURE_ENABLED": "1"
                            },
                            "when_output_statuses": ["saved", "recovered"],
                            "stdin_from_output": True,
                        }
                    ],
                )
            )
        elif host == "pi":
            tasks.append(
                BackgroundTask(
                    name="pi-save",
                    module="chronovisor.hosts.pi_record",
                    args=["--hook", "--save"],
                    env={"PI_CHRONOVISOR_RECORD_ENABLED": "1"},
                    log_prefix="pi-save",
                    on_success=[
                        {
                            "name": "recall-audit-candidate",
                            "module": "chronovisor.recall.recall_auditor",
                            "args": ["--host", "pi", "--hook"],
                            "env": {},
                            "when_output_status": "saved",
                        },
                        {
                            "name": "recall-answer-capture",
                            "module": "chronovisor.recall.recall_answer_eval",
                            "args": [
                                "--host",
                                "pi",
                                "--hook",
                                "--capture-only",
                            ],
                            "env": {
                                "CHRONOVISOR_RECALL_ANSWER_CAPTURE_ENABLED": "1"
                            },
                            "when_output_statuses": ["saved", "recovered"],
                            "stdin_from_output": True,
                        }
                    ],
                )
            )
    if (
        host in {"codex", "claude-code", "pi"}
        and content_correction_enabled(config)
    ):
        tasks.append(
            BackgroundTask(
                # One shared lane makes transcript inspection single-flight
                # across hosts. Session identity remains part of the durable
                # dedupe key, so unrelated sessions never coalesce.
                name="content-correction-capture",
                module="chronovisor.recall.content_correction",
                args=["--host", host, "--hook", "--capture-only"],
                env={"CHRONOVISOR_CONTENT_CORRECTION_ENABLED": "1"},
                log_prefix="content-correction-capture",
            )
        )
    return tasks


def run_stop(args: argparse.Namespace, stdin_text: str) -> int:
    host = normalize_host(args.host)
    if env_flag("CHRONOVISOR_INTERNAL_FRONTIER") is True:
        if args.format == "json":
            print(
                json.dumps(
                    {"status": "suppressed", "reason": "internal_frontier", "tasks": []}
                )
            )
        else:
            print("{}")
        return 0
    tasks = stop_tasks(host, args)
    spawned: list[dict[str, Any]] = []
    for task in tasks:
        if args.dry_run:
            dry_run_task: dict[str, Any] = {
                "name": task.name,
                "module": task.module,
                "args": task.args,
                "dry_run": True,
            }
            if task.on_success:
                dry_run_task["on_success"] = task.on_success
            spawned.append(dry_run_task)
        else:
            spawned.append({"name": task.name, **spawn_task(task, stdin_text)})
    if args.format == "json":
        payload: dict[str, Any] = {"status": "ok", "tasks": spawned}
        print(json.dumps(payload, ensure_ascii=False))
    else:
        print("{}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Dispatch Codex/Claude Code hooks into Chronovisor."
    )
    parser.add_argument("--host", choices=sorted(HOSTS), default="generic")
    parser.add_argument("--event", required=True)
    parser.add_argument(
        "--hook", action="store_true", help="Read hook JSON from stdin."
    )
    parser.add_argument(
        "--config",
        help="Config file override. Defaults to ~/.chronovisor/config.toml.",
    )
    parser.add_argument(
        "--format", choices=["json", "plain", "claude", "codex", "hook-json"]
    )
    parser.add_argument("--prompt")
    parser.add_argument("--cwd")
    parser.add_argument("--session-id")
    parser.add_argument("--no-search", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the ``chronovisor-hook`` command-line entry point."""
    preliminary = build_parser().parse_args(argv)
    is_user_prompt = normalize_event(preliminary.event) == "user-prompt-submit"
    if is_user_prompt:
        try:
            is_noop = (
                env_flag("CHRONOVISOR_INTERNAL_FRONTIER") is True
                or not recall_enabled()
                or not load_hook_policy(preliminary.config).user_prompt_recall
            )
        except Exception:
            is_noop = True
        if is_noop:
            _print_host_noop(normalize_host(preliminary.host))
            return 0
    from chronovisor.core.okf_cutover import OKFStartupBlocked
    try:
        with okf_runtime_operation(CHRONOVISOR_ROOT, blocking=not is_user_prompt):
            return _main_locked(argv)
    except OKFStartupBlocked:
        if is_user_prompt:
            _print_host_noop(normalize_host(preliminary.host))
            return 0
        print(json.dumps({"status": "blocked", "category": "okf_startup_blocked"}))
        return 75


def _main_locked(argv: list[str] | None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        host = normalize_host(args.host)
        event = normalize_event(args.event)
        args.host = host
        args.event = event
        startup = okf_startup_status(CHRONOVISOR_ROOT)
        if not startup.allowed:
            if event == "user-prompt-submit":
                _print_host_noop(host)
                return 0
            print(
                json.dumps(
                    {"status": "blocked", "category": "okf_startup_blocked"}
                )
            )
            return 75
        stdin_text = sys.stdin.read() if args.hook else ""
        if event != "user-prompt-submit" or startup.layout != "okf_v0_2":
            try:
                init_chronovisor()
            except Exception:
                if event == "user-prompt-submit":
                    _print_host_noop(host)
                    return 0
                raise
        if event == "user-prompt-submit":
            try:
                return run_user_prompt(args, stdin_text)
            except Exception:
                # UserPromptSubmit is a host availability boundary. Even
                # policy/breaker/render failures outside run_recall must not
                # reject the user's prompt.
                _print_host_noop(host)
                return 0
        if event == "stop":
            return run_stop(args, stdin_text)
    except Exception as exc:
        result = {"status": "error", "error": f"{exc.__class__.__name__}: {exc}"}
        print(json.dumps(result, ensure_ascii=False))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
