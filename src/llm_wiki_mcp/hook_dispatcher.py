"""Single host hook dispatcher for LLM Wiki.

Host-specific shell scripts should be thin wrappers around this entry point.
The dispatcher owns event routing; recall/save/audit modules keep the domain
logic.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from llm_wiki_mcp import recall_runtime
from llm_wiki_mcp.runtime_config import (
    active_config_file,
    env_flag,
    load_hook_policy,
)
from llm_wiki_mcp.wiki import WIKI_ROOT, init_wiki

LOG_DIR = WIKI_ROOT / "logs"

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
    flag = env_flag("LLM_WIKI_RECALL_ENABLED")
    return True if flag is None else flag


def save_enabled(host: str, explicit_config: Path | None = None) -> bool:
    policy = load_hook_policy(explicit_config)
    if not policy.stop_save:
        return False
    env_name = "CODEX_WIKI_SAVE_ENABLED" if host == "codex" else "CLAUDE_CODE_WIKI_SAVE_ENABLED"
    flag = env_flag(env_name)
    if flag is not None:
        return flag
    return active_config_file(explicit_config).name == "config.toml"


def audit_enabled(explicit_config: Path | None = None) -> bool:
    policy = load_hook_policy(explicit_config)
    if not policy.stop_audit:
        return False
    flag = env_flag("LLM_WIKI_RECALL_AUDIT_ENABLED")
    if flag is not None:
        return flag
    return active_config_file(explicit_config).name == "config.toml"


def content_correction_enabled(explicit_config: Path | None = None) -> bool:
    policy = load_hook_policy(explicit_config)
    if not policy.stop_content_correction:
        return False
    flag = env_flag("LLM_WIKI_CONTENT_CORRECTION_ENABLED")
    if flag is not None:
        return flag
    return active_config_file(explicit_config).name == "config.toml"


def recall_improve_enabled(explicit_config: Path | None = None) -> bool:
    policy = load_hook_policy(explicit_config)
    if not policy.stop_recall_improve:
        return False
    flag = env_flag("LLM_WIKI_RECALL_IMPROVE_ENABLED")
    if flag is not None:
        return flag
    return active_config_file(explicit_config).name == "config.toml"


def run_user_prompt(args: argparse.Namespace, stdin_text: str) -> int:
    host = normalize_host(args.host)
    if env_flag("LLM_WIKI_INTERNAL_FRONTIER") is True:
        _print_host_noop(host)
        return 0
    if not recall_enabled() or not load_hook_policy(args.config).user_prompt_recall:
        _print_host_noop(host)
        return 0

    payload = read_hook_payload(stdin_text) if args.hook else {}
    request = (
        recall_runtime.request_from_hook_payload(payload, host=host, event="UserPromptSubmit")
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
    result = recall_runtime.run_recall(request, policy, perform_search=not args.no_search)
    output = recall_runtime.render_output(result, args.format or host_output_format(host))
    if output:
        print(output)
    return 0


def log_file(prefix: str) -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime("%Y%m%d")
    return LOG_DIR / f"{prefix}-{today}.log"


def spawn_task(task: BackgroundTask, stdin_text: str) -> dict[str, Any]:
    """Durably enqueue hook work without starting a detached process."""
    from llm_wiki_mcp.background_jobs import enqueue_job

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
    config = active_config_file(args.config)
    tasks: list[BackgroundTask] = []
    run_save = args.only in {None, "save"}
    if run_save and save_enabled(host, config):
        if host == "codex":
            tasks.append(
                BackgroundTask(
                    name="codex-save",
                    module="llm_wiki_mcp.codex_save",
                    args=["--hook", "--save"],
                    env={"CODEX_WIKI_SAVE_ENABLED": "1"},
                    log_prefix="codex-save",
                )
            )
        elif host == "claude-code":
            tasks.append(
                BackgroundTask(
                    name="claude-code-save",
                    module="llm_wiki_mcp.claude_code_save",
                    args=["--hook", "--save"],
                    env={"CLAUDE_CODE_WIKI_SAVE_ENABLED": "1"},
                    log_prefix="claude-code-save",
                )
            )
    run_correction_capture = args.only in {None, "correction"}
    if (
        run_correction_capture
        and host in {"codex", "claude-code"}
        and content_correction_enabled(config)
    ):
        tasks.append(
            BackgroundTask(
                # One shared lane makes transcript inspection single-flight
                # across hosts. Session identity remains part of the durable
                # dedupe key, so unrelated sessions never coalesce.
                name="content-correction-capture",
                module="llm_wiki_mcp.content_correction",
                args=["--host", host, "--hook", "--capture-only"],
                env={"LLM_WIKI_CONTENT_CORRECTION_ENABLED": "1"},
                log_prefix="content-correction-capture",
            )
        )
    return tasks


def run_stop(args: argparse.Namespace, stdin_text: str) -> int:
    host = normalize_host(args.host)
    if env_flag("LLM_WIKI_INTERNAL_FRONTIER") is True:
        if args.format == "json":
            print(json.dumps({"status": "suppressed", "reason": "internal_frontier", "tasks": []}))
        else:
            print("{}")
        return 0
    tasks = stop_tasks(host, args)
    spawned: list[dict[str, Any]] = []
    for task in tasks:
        if args.dry_run:
            spawned.append({"name": task.name, "module": task.module, "args": task.args, "dry_run": True})
        else:
            spawned.append({"name": task.name, **spawn_task(task, stdin_text)})
    if args.format == "json":
        print(json.dumps({"status": "ok", "tasks": spawned}, ensure_ascii=False))
    else:
        print("{}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dispatch Codex/Claude Code hooks into LLM Wiki.")
    parser.add_argument("--host", choices=sorted(HOSTS), default="generic")
    parser.add_argument("--event", required=True)
    parser.add_argument("--hook", action="store_true", help="Read hook JSON from stdin.")
    parser.add_argument("--config", help="Config file override. Defaults to ~/.wiki/config.toml then recall.toml.")
    parser.add_argument("--format", choices=["json", "plain", "claude", "codex", "hook-json"])
    parser.add_argument("--prompt")
    parser.add_argument("--cwd")
    parser.add_argument("--session-id")
    parser.add_argument("--no-search", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--only",
        choices=["save", "audit", "correction", "improve"],
        help="Limit Stop dispatch for legacy wrappers.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        host = normalize_host(args.host)
        event = normalize_event(args.event)
        args.host = host
        args.event = event
        stdin_text = sys.stdin.read() if args.hook else ""
        init_wiki()
        if event == "user-prompt-submit":
            return run_user_prompt(args, stdin_text)
        if event == "stop":
            return run_stop(args, stdin_text)
    except Exception as exc:
        result = {"status": "error", "error": f"{exc.__class__.__name__}: {exc}"}
        print(json.dumps(result, ensure_ascii=False))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
