"""Top-level operational CLI for LLM Wiki."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from llm_wiki_mcp import runtime_status, wiki
from llm_wiki_mcp.recall_runtime import RECALL_DIR, RECALL_FEEDBACK_FILE, RECALL_LOG_FILE
from llm_wiki_mcp.runtime_config import config_summary, load_hook_policy

CODEX_HOOKS_FILE = Path.home() / ".config/codex/hooks.json"
CODEX_CONFIG_FILE = Path.home() / ".config/codex/config.toml"
CLAUDE_SETTINGS_FILE = Path.home() / "dotfiles/claude/settings.json"


def read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def read_jsonl_counter(path: Path, field: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return counts
    for line in lines:
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            counts[str(data.get(field, ""))] += 1
    counts.pop("", None)
    return counts


def count_files(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for _ in path.rglob("*") if _.is_file())


def build_status() -> dict[str, Any]:
    return {
        "wiki": {
            "root": str(wiki.WIKI_ROOT),
            "raw_files": count_files(wiki.RAW_DIR),
            "pages": count_files(wiki.PAGES_DIR),
            "system_files": count_files(wiki.SYSTEM_DIR),
        },
        "config": config_summary(),
        "recall": {
            "dir": str(RECALL_DIR),
            "decisions": dict(read_jsonl_counter(RECALL_LOG_FILE, "decision")),
            "feedback": dict(read_jsonl_counter(RECALL_FEEDBACK_FILE, "kind")),
            "query_hints": str(RECALL_DIR / "query-hints.json"),
            "auto_apply_log": str(RECALL_DIR / "auto-apply.jsonl"),
        },
        "runtime": runtime_status.read_status(),
    }


def _hook_entries(data: dict[str, Any], event: str) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for group_i, group in enumerate(data.get("hooks", {}).get(event, [])):
        if not isinstance(group, dict):
            continue
        for hook_i, hook in enumerate(group.get("hooks", [])):
            if isinstance(hook, dict):
                entries.append({
                    "event": event,
                    "group": group_i,
                    "index": hook_i,
                    "command": hook.get("command", ""),
                    "timeout": hook.get("timeout"),
                })
    return entries


def _canonical_hook_hash(event_name: str, hook: dict[str, Any]) -> str:
    identity = {
        "event_name": event_name,
        "hooks": [
            {
                "async": False,
                "command": hook.get("command", ""),
                "timeout": hook.get("timeout"),
                "type": hook.get("type", "command"),
            }
        ],
    }

    def canon(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: canon(value[key]) for key in sorted(value)}
        if isinstance(value, list):
            return [canon(item) for item in value]
        return value

    encoded = json.dumps(canon(identity), separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _codex_event_name(event: str) -> str:
    mapping = {
        "UserPromptSubmit": "user_prompt_submit",
        "Stop": "stop",
        "SessionStart": "session_start",
    }
    return mapping.get(event, event.lower())


def inspect_hooks() -> dict[str, Any]:
    codex_data = read_json(CODEX_HOOKS_FILE)
    claude_data = read_json(CLAUDE_SETTINGS_FILE)
    codex: list[dict[str, Any]] = []
    for event in ("UserPromptSubmit", "Stop", "SessionStart"):
        for entry in _hook_entries(codex_data, event):
            hook = {
                "type": "command",
                "command": entry["command"],
                "timeout": entry["timeout"],
            }
            entry["trusted_hash"] = _canonical_hook_hash(_codex_event_name(event), hook)
            codex.append(entry)
    claude: list[dict[str, Any]] = []
    for event in ("UserPromptSubmit", "Stop", "SessionStart", "PostToolUse"):
        claude.extend(_hook_entries(claude_data, event))
    return {
        "codex": {
            "hooks_file": str(CODEX_HOOKS_FILE),
            "config_file": str(CODEX_CONFIG_FILE),
            "entries": codex,
        },
        "claude_code": {
            "settings_file": str(CLAUDE_SETTINGS_FILE),
            "entries": claude,
        },
        "hook_policy": load_hook_policy().__dict__,
    }


def doctor() -> dict[str, Any]:
    status = build_status()
    hooks = inspect_hooks()
    checks: list[dict[str, Any]] = []

    def check(name: str, ok: bool, detail: str) -> None:
        checks.append({"name": name, "ok": ok, "detail": detail})

    check("wiki.root", Path(status["wiki"]["root"]).exists(), status["wiki"]["root"])
    check("wiki.raw", status["wiki"]["raw_files"] > 0, f"{status['wiki']['raw_files']} raw files")
    check("wiki.pages", status["wiki"]["pages"] > 0, f"{status['wiki']['pages']} pages")
    check("config", status["config"]["exists"], status["config"]["path"])
    check(
        "codex.recall_hook",
        any("llm-wiki" in e.get("command", "") and "recall" in e.get("command", "") for e in hooks["codex"]["entries"]),
        "Codex UserPromptSubmit recall hook",
    )
    check(
        "claude.recall_hook",
        any("llm-wiki" in e.get("command", "") and "recall" in e.get("command", "") for e in hooks["claude_code"]["entries"]),
        "Claude Code UserPromptSubmit recall hook",
    )
    check(
        "audit.feedback",
        status["recall"]["feedback"].get("missed_candidate", 0) >= 0,
        str(status["recall"]["feedback"]),
    )
    return {
        "status": "ok" if all(item["ok"] for item in checks) else "warn",
        "checks": checks,
        "summary": {
            "wiki": status["wiki"],
            "config": status["config"],
            "recall": status["recall"],
        },
    }


def print_plain_status(data: dict[str, Any]) -> None:
    print(f"wiki: {data['wiki']['root']}")
    print(
        "content: "
        f"raw={data['wiki']['raw_files']} pages={data['wiki']['pages']} "
        f"system={data['wiki']['system_files']}"
    )
    print(f"config: {data['config']['path']} ({data['config']['mode']})")
    print(f"recall decisions: {data['recall']['decisions']}")
    print(f"feedback: {data['recall']['feedback']}")
    runtime = data.get("runtime", {})
    print(f"runtime: {runtime.get('state', 'unknown')} stage={runtime.get('stage')}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Operate and inspect LLM Wiki.")
    sub = parser.add_subparsers(dest="command", required=True)
    status_parser = sub.add_parser("status", help="Show content, recall, and runtime status.")
    status_parser.add_argument("--json", action="store_true")
    doctor_parser = sub.add_parser("doctor", help="Run operational checks.")
    doctor_parser.add_argument("--json", action="store_true")
    hooks_parser = sub.add_parser("hooks", help="Inspect host hook configuration.")
    hooks_sub = hooks_parser.add_subparsers(dest="hooks_command", required=True)
    hooks_inspect = hooks_sub.add_parser("inspect", help="List configured host hooks.")
    hooks_inspect.add_argument("--json", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "status":
        data = build_status()
        if args.json:
            print(json.dumps(data, ensure_ascii=False, indent=2, default=str))
        else:
            print_plain_status(data)
        return 0
    if args.command == "doctor":
        data = doctor()
        if args.json:
            print(json.dumps(data, ensure_ascii=False, indent=2, default=str))
        else:
            for item in data["checks"]:
                mark = "ok" if item["ok"] else "warn"
                print(f"{mark}\t{item['name']}\t{item['detail']}")
            print(f"status\t{data['status']}")
        return 0 if data["status"] == "ok" else 1
    if args.command == "hooks" and args.hooks_command == "inspect":
        data = inspect_hooks()
        if args.json:
            print(json.dumps(data, ensure_ascii=False, indent=2, default=str))
        else:
            for host, section in (("codex", data["codex"]), ("claude-code", data["claude_code"])):
                print(f"== {host} ==")
                for entry in section["entries"]:
                    print(f"{entry['event']}:{entry['group']}:{entry['index']}\t{entry['command']}")
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
