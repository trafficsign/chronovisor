"""Single host hook dispatcher for Chronovisor.

Host-specific shell scripts should be thin wrappers around this entry point.
The dispatcher only enqueues deterministic transcript capture.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from chronovisor.common.config import (
    active_config_file,
    env_flag,
    load_hook_policy,
)

HOSTS = {"codex", "claude-code", "generic"}
USER_PROMPT_EVENTS = {"user-prompt-submit", "userpromptsubmit", "prompt-submit"}
STOP_EVENTS = {"stop"}


@dataclass(frozen=True)
class BackgroundTask:
    name: str
    module: str
    args: list[str]
    env: dict[str, str] = field(default_factory=dict)
    log_prefix: str = "hook"


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


def _print_host_noop(host: str) -> None:
    if host == "codex":
        print("{}", flush=True)


def save_enabled(host: str, explicit_config: Path | None = None) -> bool:
    if (
        host not in {"codex", "claude-code"}
        or not load_hook_policy(explicit_config).stop_save
    ):
        return False
    env_name = (
        "CODEX_CHRONOVISOR_RECORD_ENABLED"
        if host == "codex"
        else "CLAUDE_CODE_CHRONOVISOR_RECORD_ENABLED"
    )
    flag = env_flag(env_name)
    return (
        flag
        if flag is not None
        else active_config_file(explicit_config).name == "config.toml"
    )


def run_user_prompt(args: argparse.Namespace, stdin_text: str) -> int:
    _print_host_noop(normalize_host(args.host))
    return 0


def spawn_task(task: BackgroundTask, stdin_text: str) -> dict[str, Any]:
    """Durably enqueue hook work without starting a detached process."""
    from chronovisor.capture.background_jobs import enqueue_job

    job = enqueue_job(
        name=task.name,
        module=task.module,
        args=task.args,
        env=task.env,
        stdin_text=stdin_text,
    )
    return {
        "job_id": job["job_id"],
        "status": job.get("status"),
        "enqueued": bool(job.get("enqueued", True)),
        "coalesced": bool(job.get("coalesced", False)),
    }


def stop_tasks(host: str, args: argparse.Namespace) -> list[BackgroundTask]:
    if not save_enabled(host, args.config):
        return []
    module = "codex_record" if host == "codex" else "claude_code_record"
    enable_env = (
        "CODEX_CHRONOVISOR_RECORD_ENABLED"
        if host == "codex"
        else "CLAUDE_CODE_CHRONOVISOR_RECORD_ENABLED"
    )
    return [
        BackgroundTask(
            name=f"{host}-save",
            module=f"chronovisor.hosts.{module}",
            args=["--hook", "--save"],
            env={enable_env: "1"},
            log_prefix=f"{host}-save",
        )
    ]


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
    args = build_parser().parse_args(argv)
    try:
        args.host = normalize_host(args.host)
        args.event = normalize_event(args.event)
        if args.event == "user-prompt-submit":
            return run_user_prompt(args, "")
        return run_stop(args, sys.stdin.read() if args.hook else "")
    except Exception as exc:
        print(
            json.dumps({"status": "error", "error": f"{exc.__class__.__name__}: {exc}"})
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
