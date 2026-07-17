"""Top-level operational CLI for LLM Wiki."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
from collections import Counter
from pathlib import Path
from typing import Any

from llm_wiki_mcp import runtime_status, wiki
from llm_wiki_mcp.recall_runtime import (
    RECALL_DIR,
    RECALL_FEEDBACK_FILE,
    RECALL_LOG_FILE,
)
from llm_wiki_mcp.runtime_config import (
    config_summary,
    load_hook_policy,
    runtime_identity,
    uvx_runtime_command,
)

CODEX_HOOKS_FILE = Path.home() / ".config/codex/hooks.json"
CODEX_CONFIG_FILE = Path.home() / ".config/codex/config.toml"
CLAUDE_SETTINGS_FILE = Path.home() / "dotfiles/claude/settings.json"
USER_PROMPT_HOOK_TIMEOUT_MS = 7000
STOP_HOOK_TIMEOUT_MS = 5000


def read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


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
    from llm_wiki_mcp.health import health_snapshot

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
        "health": health_snapshot(),
    }


def _hook_entries(data: dict[str, Any], event: str) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for group_i, group in enumerate(data.get("hooks", {}).get(event, [])):
        if not isinstance(group, dict):
            continue
        for hook_i, hook in enumerate(group.get("hooks", [])):
            if isinstance(hook, dict):
                entry = {
                    "event": event,
                    "group": group_i,
                    "index": hook_i,
                    "command": hook.get("command", ""),
                    "timeout": hook.get("timeout"),
                }
                entry.update(_hook_compatibility(str(entry["command"])))
                entries.append(entry)
    return entries


def _hook_compatibility(command: str) -> dict[str, Any]:
    noop_removal_after = "2026-10-01"
    no_op_markers = (
        "codex_recall_audit_hook.sh",
        "claude_code_recall_audit_hook.sh",
        "--only audit",
        "--only improve",
    )
    legacy_markers = (
        "codex_recall_hook.sh",
        "codex_wiki_save_hook.sh",
        "claude_code_recall_hook.sh",
        "claude_code_wiki_save_hook.sh",
        "llm-wiki-recall",
        "llm-wiki-codex-save",
        "llm-wiki-claude-code-save",
    )
    if any(marker in command for marker in no_op_markers):
        return {
            "compatibility": "deprecated_noop",
            "deprecated": True,
            "warning": "This legacy audit/improve wrapper is a compatibility no-op.",
            "replacement": "llm-wiki-hook --event Stop; audit is queued after durable save receipt",
            "removal_after": noop_removal_after,
        }
    if "llm-wiki-hook" not in command and any(
        marker in command for marker in legacy_markers
    ):
        return {
            "compatibility": "legacy_wrapper",
            "deprecated": True,
            "warning": "Legacy wrapper; migrate with `llm-wiki hooks install`.",
            "replacement": "direct llm-wiki-hook entry",
        }
    return {"compatibility": "current", "deprecated": False}


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


def _codex_state_index(event: str, group_i: int, hook_i: int) -> str:
    return f"{_codex_event_name(event)}:{group_i}:{hook_i}"


def default_hook_command_prefix() -> str:
    return shlex.join(uvx_runtime_command("llm-wiki-hook"))


def _is_llm_wiki_command(command: object) -> bool:
    if not isinstance(command, str):
        return False
    markers = (
        "llm-wiki-hook",
        "llm-wiki-recall",
        "llm-wiki-codex-save",
        "llm-wiki-claude-code-save",
        "codex_recall_hook.sh",
        "codex_wiki_save_hook.sh",
        "codex_recall_audit_hook.sh",
        "claude_code_recall_hook.sh",
        "claude_code_wiki_save_hook.sh",
        "claude_code_recall_audit_hook.sh",
    )
    return any(marker in command for marker in markers)


def _hook(type_: str, command: str, timeout: int | None = None) -> dict[str, Any]:
    data: dict[str, Any] = {"type": type_, "command": command}
    if timeout is not None:
        data["timeout"] = timeout
    return data


def _replace_event_llm_wiki_hooks(
    data: dict[str, Any],
    event: str,
    new_hooks: list[dict[str, Any]],
) -> None:
    hooks = data.setdefault("hooks", {})
    groups = hooks.setdefault(event, [{"hooks": []}])
    if not groups:
        groups.append({"hooks": []})
    group = groups[0]
    existing = group.setdefault("hooks", [])
    kept: list[dict[str, Any]] = []
    insert_at: int | None = None
    for hook in existing:
        command = hook.get("command") if isinstance(hook, dict) else ""
        if _is_llm_wiki_command(command):
            if insert_at is None:
                insert_at = len(kept)
            continue
        kept.append(hook)
    if insert_at is None:
        insert_at = len(kept)
    group["hooks"] = kept[:insert_at] + new_hooks + kept[insert_at:]


def _llm_wiki_codex_state_indexes(data: dict[str, Any], event: str) -> set[str]:
    indexes: set[str] = set()
    for group_i, group in enumerate(data.get("hooks", {}).get(event, [])):
        if not isinstance(group, dict):
            continue
        for hook_i, hook in enumerate(group.get("hooks", [])):
            if isinstance(hook, dict) and _is_llm_wiki_command(hook.get("command")):
                indexes.add(_codex_state_index(event, group_i, hook_i))
    return indexes


def _find_codex_state_index(data: dict[str, Any], event: str, command: str) -> str:
    for group_i, group in enumerate(data.get("hooks", {}).get(event, [])):
        if not isinstance(group, dict):
            continue
        for hook_i, hook in enumerate(group.get("hooks", [])):
            if isinstance(hook, dict) and hook.get("command") == command:
                return _codex_state_index(event, group_i, hook_i)
    raise ValueError(f"installed hook not found for {event}: {command}")


def install_codex_hooks(
    command_prefix: str | None = None, *, dry_run: bool = False
) -> dict[str, Any]:
    prefix = command_prefix or default_hook_command_prefix()
    data = read_json(CODEX_HOOKS_FILE)
    if not data:
        data = {"hooks": {}}
    stale_state_indexes = _llm_wiki_codex_state_indexes(
        data, "UserPromptSubmit"
    ) | _llm_wiki_codex_state_indexes(data, "Stop")

    user_command = (
        "CODEX_HOME=/Users/trafficsign/.config/codex "
        f"{prefix} --host codex --event UserPromptSubmit --hook"
    )
    stop_command = (
        "CODEX_HOME=/Users/trafficsign/.config/codex "
        f"{prefix} --host codex --event Stop --hook"
    )
    user_hook = _hook("command", user_command, USER_PROMPT_HOOK_TIMEOUT_MS)
    stop_hook = _hook("command", stop_command, STOP_HOOK_TIMEOUT_MS)
    _replace_event_llm_wiki_hooks(data, "UserPromptSubmit", [user_hook])
    _replace_event_llm_wiki_hooks(data, "Stop", [stop_hook])

    hashes = {
        _find_codex_state_index(
            data, "UserPromptSubmit", user_command
        ): _canonical_hook_hash(
            "user_prompt_submit",
            user_hook,
        ),
        _find_codex_state_index(data, "Stop", stop_command): _canonical_hook_hash(
            "stop",
            stop_hook,
        ),
    }
    stale_state_indexes -= set(hashes)
    if not dry_run:
        write_json(CODEX_HOOKS_FILE, data)
        update_codex_trust_state(
            CODEX_CONFIG_FILE, hashes, remove_indexes=stale_state_indexes
        )
    return {
        "host": "codex",
        "hooks_file": str(CODEX_HOOKS_FILE),
        "config_file": str(CODEX_CONFIG_FILE),
        "commands": {"user_prompt_submit": user_command, "stop": stop_command},
        "trusted_hashes": hashes,
        "dry_run": dry_run,
    }


def install_claude_code_hooks(
    command_prefix: str | None = None, *, dry_run: bool = False
) -> dict[str, Any]:
    prefix = command_prefix or default_hook_command_prefix()
    data = read_json(CLAUDE_SETTINGS_FILE)
    if not data:
        data = {"hooks": {}}
    user_command = f"{prefix} --host claude-code --event UserPromptSubmit --hook"
    stop_command = f"{prefix} --host claude-code --event Stop --hook"
    _replace_event_llm_wiki_hooks(
        data,
        "UserPromptSubmit",
        [_hook("command", user_command, USER_PROMPT_HOOK_TIMEOUT_MS)],
    )
    _replace_event_llm_wiki_hooks(
        data,
        "Stop",
        [_hook("command", stop_command, STOP_HOOK_TIMEOUT_MS)],
    )
    if not dry_run:
        write_json(CLAUDE_SETTINGS_FILE, data)
    return {
        "host": "claude-code",
        "settings_file": str(CLAUDE_SETTINGS_FILE),
        "commands": {"user_prompt_submit": user_command, "stop": stop_command},
        "dry_run": dry_run,
    }


def _state_key(event_and_index: str) -> str:
    return f"/Users/trafficsign/.config/codex/hooks.json:{event_and_index}"


def _section_header(key: str) -> str:
    return f'[hooks.state."{key}"]'


def _render_state_section(key: str, trusted_hash: str) -> list[str]:
    return [
        f"{_section_header(key)}\n",
        "enabled = true\n",
        f'trusted_hash = "{trusted_hash}"\n',
        "\n",
    ]


def update_codex_trust_state(
    config_file: Path,
    hashes: dict[str, str],
    *,
    remove_indexes: set[str] | None = None,
) -> None:
    try:
        lines = config_file.read_text(encoding="utf-8").splitlines(keepends=True)
    except OSError:
        lines = ["[hooks.state]\n", "\n"]

    desired = {
        _state_key(index): trusted_hash for index, trusted_hash in hashes.items()
    }
    remove = {_state_key(index) for index in (remove_indexes or set())}
    seen: set[str] = set()
    out: list[str] = []
    i = 0
    section_re = re.compile(r'^\[hooks\.state\."(.+)"\]\s*$')
    while i < len(lines):
        match = section_re.match(lines[i].strip())
        if not match:
            out.append(lines[i])
            i += 1
            continue
        key = match.group(1)
        j = i + 1
        while j < len(lines) and not lines[j].startswith("["):
            j += 1
        if key in remove:
            i = j
            continue
        if key in desired:
            out.extend(_render_state_section(key, desired[key]))
            seen.add(key)
            i = j
            continue
        out.extend(lines[i:j])
        i = j

    if not any(line.strip() == "[hooks.state]" for line in out):
        out.append("\n[hooks.state]\n\n")
    if out and out[-1].strip():
        out.append("\n")
    for key, trusted_hash in desired.items():
        if key not in seen:
            out.extend(_render_state_section(key, trusted_hash))
    config_file.parent.mkdir(parents=True, exist_ok=True)
    config_file.write_text("".join(out), encoding="utf-8")


def install_hooks(
    host: str, command_prefix: str | None = None, *, dry_run: bool = False
) -> dict[str, Any]:
    if host == "codex":
        return install_codex_hooks(command_prefix, dry_run=dry_run)
    if host == "claude-code":
        return install_claude_code_hooks(command_prefix, dry_run=dry_run)
    if host == "all":
        return {
            "host": "all",
            "results": [
                install_codex_hooks(command_prefix, dry_run=dry_run),
                install_claude_code_hooks(command_prefix, dry_run=dry_run),
            ],
        }
    raise ValueError(f"unsupported host: {host}")


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
    warnings = [
        {
            "host": host,
            "event": entry.get("event"),
            "command": entry.get("command"),
            "compatibility": entry.get("compatibility"),
            "warning": entry.get("warning"),
            "replacement": entry.get("replacement"),
            "removal_after": entry.get("removal_after"),
        }
        for host, entries in (("codex", codex), ("claude-code", claude))
        for entry in entries
        if entry.get("deprecated")
    ]
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
        "warnings": warnings,
    }


def _has_user_prompt_hook(entries: list[dict[str, Any]]) -> bool:
    for entry in entries:
        if entry.get("event") != "UserPromptSubmit":
            continue
        command = entry.get("command", "")
        if "llm-wiki-hook" in command and "--event UserPromptSubmit" in command:
            return True
        if "llm-wiki" in command and "recall" in command:
            return True
    return False


def _has_stop_hook(entries: list[dict[str, Any]]) -> bool:
    for entry in entries:
        if entry.get("event") != "Stop":
            continue
        command = entry.get("command", "")
        if "llm-wiki-hook" in command and "--event Stop" in command:
            return True
        if "llm-wiki" in command and ("save" in command or "audit" in command):
            return True
    return False


def doctor() -> dict[str, Any]:
    status = build_status()
    hooks = inspect_hooks()
    checks: list[dict[str, Any]] = []

    def check(name: str, ok: bool, detail: str) -> None:
        checks.append({"name": name, "ok": ok, "detail": detail})

    check("wiki.root", Path(status["wiki"]["root"]).exists(), status["wiki"]["root"])
    check(
        "wiki.raw",
        status["wiki"]["raw_files"] > 0,
        f"{status['wiki']['raw_files']} raw files",
    )
    check("wiki.pages", status["wiki"]["pages"] > 0, f"{status['wiki']['pages']} pages")
    check("config", status["config"]["exists"], status["config"]["path"])
    check(
        "codex.recall_hook",
        _has_user_prompt_hook(hooks["codex"]["entries"]),
        "Codex UserPromptSubmit recall hook",
    )
    check(
        "codex.stop_hook",
        _has_stop_hook(hooks["codex"]["entries"]),
        "Codex Stop save/audit hook",
    )
    check(
        "claude.recall_hook",
        _has_user_prompt_hook(hooks["claude_code"]["entries"]),
        "Claude Code UserPromptSubmit recall hook",
    )
    check(
        "claude.stop_hook",
        _has_stop_hook(hooks["claude_code"]["entries"]),
        "Claude Code Stop save/audit hook",
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


def build_parser() -> argparse.ArgumentParser:
    """Build the complete stable CLI command tree without executing a handler."""

    parser = argparse.ArgumentParser(description="Operate and inspect LLM Wiki.")
    sub = parser.add_subparsers(dest="command", required=True)
    status_parser = sub.add_parser(
        "status", help="Show content, recall, and runtime status."
    )
    status_parser.add_argument("--json", action="store_true")
    runtime_identity_parser = sub.add_parser(
        "runtime-identity",
        help="Show the installed package revision without reading Wiki content.",
    )
    runtime_identity_parser.add_argument("--json", action="store_true")
    doctor_parser = sub.add_parser("doctor", help="Run operational checks.")
    doctor_parser.add_argument("--json", action="store_true")
    hooks_parser = sub.add_parser("hooks", help="Inspect host hook configuration.")
    hooks_sub = hooks_parser.add_subparsers(dest="hooks_command", required=True)
    hooks_inspect = hooks_sub.add_parser("inspect", help="List configured host hooks.")
    hooks_inspect.add_argument("--json", action="store_true")
    hooks_install = hooks_sub.add_parser(
        "install", help="Install direct host hook entries."
    )
    hooks_install.add_argument(
        "--host", choices=("codex", "claude-code", "all"), required=True
    )
    hooks_install.add_argument(
        "--command-prefix", help="Override the llm-wiki-hook command prefix."
    )
    hooks_install.add_argument("--dry-run", action="store_true")
    hooks_install.add_argument("--json", action="store_true")
    health_parser = sub.add_parser("health", help="Show knowledge health KPIs.")
    health_parser.add_argument("--json", action="store_true")
    snapshot_parser = sub.add_parser(
        "wiki-snapshot", help="Commit ~/.wiki into its own git history."
    )
    snapshot_parser.add_argument("reason", nargs="?", default="manual")
    snapshot_parser.add_argument("--allow-empty", action="store_true")
    snapshot_parser.add_argument("--json", action="store_true")
    entities_parser = sub.add_parser(
        "entities", help="Maintain entity registry and frontmatter."
    )
    entities_sub = entities_parser.add_subparsers(
        dest="entities_command", required=True
    )
    entities_init = entities_sub.add_parser("init")
    entities_init.add_argument("--json", action="store_true")
    entities_backfill = entities_sub.add_parser("backfill")
    entities_backfill.add_argument("--limit", type=int, default=0)
    entities_backfill.add_argument("--max-frontier-calls", type=int, default=0)
    entities_backfill.add_argument("--dry-run", action="store_true")
    entities_backfill.add_argument("--include-reference", action="store_true")
    entities_backfill.add_argument("--json", action="store_true")
    raw_replay_parser = sub.add_parser(
        "raw-replay", help="Plan or run retroactive raw re-ingestion."
    )
    raw_replay_parser.add_argument("--since", default="")
    raw_replay_parser.add_argument("--limit", type=int, default=0)
    raw_replay_parser.add_argument("--run", action="store_true")
    raw_replay_parser.add_argument("--json", action="store_true")
    convergence_drain_parser = sub.add_parser(
        "convergence-drain",
        help="Drain only a durable snapshot of existing convergence keys.",
    )
    convergence_drain_sub = convergence_drain_parser.add_subparsers(
        dest="convergence_drain_command",
        required=True,
    )
    convergence_drain_plan = convergence_drain_sub.add_parser("plan")
    convergence_drain_plan.add_argument("--json", action="store_true")
    convergence_drain_start = convergence_drain_sub.add_parser("start")
    convergence_drain_start.add_argument(
        "--max-elapsed-seconds", type=float, default=1_800.0
    )
    convergence_drain_start.add_argument("--dry-run", action="store_true")
    convergence_drain_start.add_argument("--json", action="store_true")
    convergence_drain_resume = convergence_drain_sub.add_parser("resume")
    convergence_drain_resume.add_argument("--run-id", required=True)
    convergence_drain_resume.add_argument("--dry-run", action="store_true")
    convergence_drain_resume.add_argument("--json", action="store_true")
    convergence_drain_status = convergence_drain_sub.add_parser("status")
    convergence_drain_status.add_argument("--run-id", required=True)
    convergence_drain_status.add_argument("--json", action="store_true")
    memory_eval_parser = sub.add_parser(
        "memory-integrity", help="Evaluate raw-to-memory capture integrity."
    )
    memory_eval_parser.add_argument("--since", default="")
    memory_eval_parser.add_argument("--limit", type=int, default=100)
    memory_eval_parser.add_argument("--no-write", action="store_true")
    memory_eval_parser.add_argument("--json", action="store_true")
    cofire_parser = sub.add_parser("cofire", help="Build recall co-fire graph.")
    cofire_parser.add_argument("--limit", type=int, default=5000)
    cofire_parser.add_argument("--min-count", type=int, default=2)
    cofire_parser.add_argument("--no-write", action="store_true")
    cofire_parser.add_argument("--json", action="store_true")
    prefetch_parser = sub.add_parser(
        "prefetch", help="Build speculative recall prefetch cache."
    )
    prefetch_parser.add_argument("--limit", type=int, default=5000)
    prefetch_parser.add_argument("--no-write", action="store_true")
    prefetch_parser.add_argument("--json", action="store_true")
    sleep_parser = sub.add_parser("sleep", help="Run sleep-cycle consolidation.")
    sleep_parser.add_argument("--raw-limit", type=int, default=100)
    sleep_parser.add_argument("--eval-limit", type=int, default=100)
    sleep_parser.add_argument("--duplicate-limit", type=int, default=200)
    sleep_parser.add_argument("--dry-run", action="store_true")
    sleep_parser.add_argument("--json", action="store_true")
    claims_parser = sub.add_parser("claims", help="Build/search derived claim index.")
    claims_sub = claims_parser.add_subparsers(dest="claims_command", required=True)
    claims_rebuild = claims_sub.add_parser("rebuild")
    claims_rebuild.add_argument("--limit", type=int, default=0)
    claims_rebuild.add_argument("--json", action="store_true")
    claims_sanitize = claims_sub.add_parser("sanitize")
    claims_sanitize.add_argument("--no-write", action="store_true")
    claims_sanitize.add_argument("--json", action="store_true")
    claims_search = claims_sub.add_parser("search")
    claims_search.add_argument("query")
    claims_search.add_argument("--limit", type=int, default=10)
    claims_search.add_argument("--json", action="store_true")
    golden_parser = sub.add_parser(
        "golden-expand", help="Expand search golden set from recall_questions."
    )
    golden_parser.add_argument("--limit", type=int, default=0)
    golden_parser.add_argument("--include-reference", action="store_true")
    golden_parser.add_argument("--no-write", action="store_true")
    golden_parser.add_argument("--json", action="store_true")
    retention_parser = sub.add_parser(
        "retention", help="Build retention/time-prior scores."
    )
    retention_parser.add_argument("--limit", type=int, default=5000)
    retention_parser.add_argument("--no-write", action="store_true")
    retention_parser.add_argument("--json", action="store_true")
    reflect_parser = sub.add_parser(
        "reflect", help="Generate a memory reflection page."
    )
    reflect_parser.add_argument("--no-write", action="store_true")
    reflect_parser.add_argument("--json", action="store_true")
    hubs_parser = sub.add_parser("hubs", help="Generate auto-maintained hub pages.")
    hubs_parser.add_argument("--min-pages", type=int, default=3)
    hubs_parser.add_argument("--max-hubs", type=int, default=20)
    hubs_parser.add_argument("--no-write", action="store_true")
    hubs_parser.add_argument("--json", action="store_true")
    distill_parser = sub.add_parser(
        "distill", help="Export wiki QA pairs for distillation."
    )
    autonomy_parser = sub.add_parser(
        "autonomy", help="Run/install autonomous operation loops."
    )
    autonomy_sub = autonomy_parser.add_subparsers(
        dest="autonomy_command", required=True
    )
    autonomy_status = autonomy_sub.add_parser("status")
    autonomy_status.add_argument("--json", action="store_true")
    autonomy_watchdog = autonomy_sub.add_parser("watchdog")
    autonomy_watchdog.add_argument("--notify", action="store_true")
    autonomy_watchdog.add_argument("--json", action="store_true")
    autonomy_digest = autonomy_sub.add_parser("digest")
    autonomy_digest.add_argument("--json", action="store_true")
    autonomy_install = autonomy_sub.add_parser("install-launchd")
    autonomy_install.add_argument("--dry-run", action="store_true")
    autonomy_install.add_argument("--load", action="store_true")
    autonomy_install.add_argument("--json", action="store_true")
    distill_parser.add_argument("--limit", type=int, default=0)
    distill_parser.add_argument("--include-reference", action="store_true")
    distill_parser.add_argument("--no-write", action="store_true")
    distill_parser.add_argument("--json", action="store_true")
    oracle_parser = sub.add_parser(
        "oracle", help="Return cited wiki oracle evidence bundle."
    )
    oracle_parser.add_argument("query")
    oracle_parser.add_argument("--top-n", type=int, default=8)
    oracle_parser.add_argument("--claim-limit", type=int, default=12)
    oracle_parser.add_argument("--no-index-build", action="store_true")
    oracle_parser.add_argument("--json", action="store_true")
    recall_eval_parser = sub.add_parser(
        "recall-eval", help="Replay-evaluate recall decisions."
    )
    recall_eval_parser.add_argument("--config")
    recall_eval_parser.add_argument("--log-file", default=str(RECALL_LOG_FILE))
    recall_eval_parser.add_argument(
        "--feedback-file", default=str(RECALL_FEEDBACK_FILE)
    )
    recall_eval_parser.add_argument("--save-baseline", action="store_true")
    recall_eval_parser.add_argument("--config-override", action="append", default=[])
    recall_eval_parser.add_argument("--json", action="store_true")
    recall_improve_parser = sub.add_parser(
        "recall-improve", help="Run self-improving recall policy loop."
    )
    recall_improve_sub = recall_improve_parser.add_subparsers(
        dest="recall_improve_command", required=True
    )
    recall_improve_run = recall_improve_sub.add_parser(
        "run", help="Propose, replay-evaluate, and adopt a policy patch."
    )
    recall_improve_run.add_argument("--config")
    recall_improve_run.add_argument("--log-file", default=str(RECALL_LOG_FILE))
    recall_improve_run.add_argument(
        "--feedback-file", default=str(RECALL_FEEDBACK_FILE)
    )
    recall_improve_run.add_argument(
        "--models", help="Comma-separated Ollama proposer models."
    )
    recall_improve_run.add_argument(
        "--no-apply", dest="apply", action="store_false", default=True
    )
    recall_improve_run.add_argument(
        "--no-heuristic", dest="include_heuristic", action="store_false", default=True
    )
    recall_improve_run.add_argument("--min-improvement", type=float, default=0.05)
    recall_improve_run.add_argument("--max-examples", type=int, default=120)
    recall_improve_run.add_argument(
        "--frontier", choices=["always", "auto", "off"], default="auto"
    )
    recall_improve_run.add_argument("--frontier-timeout", type=int)
    recall_improve_run.add_argument("--json", action="store_true")
    recall_improve_due = recall_improve_sub.add_parser(
        "run-due", help="Run only when schedule/feedback gates are due."
    )
    recall_improve_due.add_argument("--config")
    recall_improve_due.add_argument("--log-file", default=str(RECALL_LOG_FILE))
    recall_improve_due.add_argument(
        "--feedback-file", default=str(RECALL_FEEDBACK_FILE)
    )
    recall_improve_due.add_argument(
        "--models", help="Comma-separated Ollama proposer models."
    )
    recall_improve_due.add_argument(
        "--no-apply", dest="apply", action="store_false", default=True
    )
    recall_improve_due.add_argument(
        "--no-heuristic", dest="include_heuristic", action="store_false", default=True
    )
    recall_improve_due.add_argument("--min-improvement", type=float, default=0.05)
    recall_improve_due.add_argument("--max-examples", type=int, default=80)
    recall_improve_due.add_argument("--min-interval-hours", type=float, default=24.0)
    recall_improve_due.add_argument("--min-new-feedback", type=int, default=5)
    recall_improve_due.add_argument("--min-total-feedback", type=int, default=3)
    recall_improve_due.add_argument(
        "--frontier", choices=["always", "auto", "off"], default="auto"
    )
    recall_improve_due.add_argument("--frontier-timeout", type=int)
    recall_improve_due.add_argument("--dry-run", action="store_true")
    recall_improve_due.add_argument("--json", action="store_true")
    recall_improve_status = recall_improve_sub.add_parser(
        "status", help="Show active recall improvement policy."
    )
    recall_improve_status.add_argument("--json", action="store_true")
    recall_improve_rollback = recall_improve_sub.add_parser(
        "rollback", help="Rollback accepted recall policy."
    )
    recall_improve_rollback.add_argument("--json", action="store_true")

    return parser


def dispatch(args: argparse.Namespace) -> int:
    """Dispatch one already-parsed command while preserving its exit contract."""

    if args.command == "status":
        data = build_status()
        if args.json:
            print(json.dumps(data, ensure_ascii=False, indent=2, default=str))
        else:
            print_plain_status(data)
        return 0
    if args.command == "runtime-identity":
        data = runtime_identity()
        if args.json:
            print(json.dumps(data, ensure_ascii=False, indent=2, default=str))
        else:
            print(f"commit_id\t{data.get('commit_id')}")
            print(f"expected_commit\t{data.get('expected_commit')}")
            print(f"archive_path\t{data.get('archive_path')}")
        return 0 if data.get("commit_id") else 1
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
            for host, section in (
                ("codex", data["codex"]),
                ("claude-code", data["claude_code"]),
            ):
                print(f"== {host} ==")
                for entry in section["entries"]:
                    print(
                        f"{entry['event']}:{entry['group']}:{entry['index']}\t"
                        f"{entry['compatibility']}\t{entry['command']}"
                    )
            for warning in data["warnings"]:
                print(
                    f"warning\t{warning['host']}\t{warning['warning']}\t"
                    f"replacement={warning['replacement']}\t"
                    f"removal_after={warning.get('removal_after') or 'unscheduled'}"
                )
        return 0
    if args.command == "hooks" and args.hooks_command == "install":
        data = install_hooks(args.host, args.command_prefix, dry_run=args.dry_run)
        if args.json:
            print(json.dumps(data, ensure_ascii=False, indent=2, default=str))
        else:
            results = data.get("results", [data])
            action = "would install" if args.dry_run else "installed"
            for result in results:
                print(f"{action}\t{result['host']}")
                for event_name, command in result["commands"].items():
                    print(f"{event_name}\t{command}")
        return 0
    if args.command == "health":
        from llm_wiki_mcp.health import health_snapshot

        data = health_snapshot()
        if args.json:
            print(json.dumps(data, ensure_ascii=False, indent=2, default=str))
        else:
            coverage = data["coverage"]
            capture = data["capture"]
            memory_integrity = data["memory_integrity"]
            cofire = data["cofire"]
            derived = data.get("derived", {})
            queues = data["queues"]
            print(f"summary_coverage\t{coverage['summary_coverage']:.3f}")
            print(
                f"recall_question_coverage\t{coverage['recall_question_coverage']:.3f}"
            )
            print(f"claim_coverage\t{capture['claim_coverage']}")
            print(f"memory_integrity_capture\t{memory_integrity.get('capture_rate')}")
            print(f"cofire_edges\t{cofire.get('edges', 0)}")
            print(f"claim_index_claims\t{derived.get('claims', 0)}")
            print(f"distill_rows\t{derived.get('distill_rows', 0)}")
            print(f"retention_pages\t{derived.get('retention_pages', 0)}")
            print(f"sensitivity_high\t{coverage.get('sensitivity', {}).get('high', 0)}")
            print(f"duplicate_candidates\t{queues['duplicate_candidates']}")
            print(f"lint_repair\t{queues['lint_repair']}")
            print(f"search_golden\t{queues['search_golden']}")
        return 0
    if args.command == "wiki-snapshot":
        from llm_wiki_mcp.wiki_snapshot import snapshot_wiki

        data = snapshot_wiki(args.reason, allow_empty=args.allow_empty)
        if args.json:
            print(json.dumps(data, ensure_ascii=False, indent=2, default=str))
        else:
            print("\t".join(f"{key}={value}" for key, value in data.items()))
        return 0 if data.get("status") in {"clean", "committed"} else 1
    if args.command == "entities":
        from llm_wiki_mcp import entities

        if args.entities_command == "init":
            path = entities.write_default_registry()
            data = {"status": "ok", "path": str(path)}
        else:
            data = entities.backfill_entities(
                limit=max(0, args.limit),
                dry_run=args.dry_run,
                include_reference=args.include_reference,
                max_frontier_calls=max(0, args.max_frontier_calls),
            )
        if args.json:
            print(json.dumps(data, ensure_ascii=False, indent=2, default=str))
        else:
            print(
                "\t".join(
                    f"{key}={value}" for key, value in data.items() if key != "pages"
                )
            )
        return 0
    if args.command == "raw-replay":
        from llm_wiki_mcp import raw_replay

        data = (
            raw_replay.run_replay(since=args.since, limit=max(1, args.limit or 1))
            if args.run
            else raw_replay.build_queue(since=args.since, limit=max(0, args.limit))
        )
        if args.json:
            print(json.dumps(data, ensure_ascii=False, indent=2, default=str))
        else:
            print(
                "\t".join(
                    f"{key}={value}" for key, value in data.items() if key != "runs"
                )
            )
        return 0
    if args.command == "convergence-drain":
        from llm_wiki_mcp import convergence_drain

        if args.convergence_drain_command == "plan":
            data = convergence_drain.plan()
        elif args.convergence_drain_command == "start":
            data = convergence_drain.start(
                max_elapsed_seconds=max(0.0, args.max_elapsed_seconds),
                dry_run=args.dry_run,
            )
        elif args.convergence_drain_command == "resume":
            data = convergence_drain.resume(
                run_id=args.run_id,
                dry_run=args.dry_run,
            )
        else:
            data = convergence_drain.status(run_id=args.run_id)
        if args.json:
            print(json.dumps(data, ensure_ascii=False, indent=2, default=str))
        else:
            for key in (
                "status",
                "run_id",
                "target_keys",
                "target_active",
                "target_terminal",
                "next_retry_at",
                "manifest_path",
            ):
                if key in data:
                    print(f"{key}\t{data[key]}")
        return 1 if data.get("status") in {"failed", "failed_frontier_activity"} else 0
    if args.command == "memory-integrity":
        from llm_wiki_mcp.memory_integrity import run_eval

        data = run_eval(
            since=args.since, limit=max(0, args.limit), write=not args.no_write
        )
        if args.json:
            print(json.dumps(data, ensure_ascii=False, indent=2, default=str))
        else:
            print(f"capture_rate\t{data['capture_rate']}")
            print(f"passed\t{data['passed']}")
            print(f"missed\t{data['missed']}")
        return 0
    if args.command == "cofire":
        from llm_wiki_mcp.cofire import build_cofire_graph

        data = build_cofire_graph(
            limit=max(1, args.limit),
            min_count=max(1, args.min_count),
            write=not args.no_write,
        )
        public = {key: value for key, value in data.items() if key != "graph"}
        if args.json:
            print(json.dumps(public, ensure_ascii=False, indent=2, default=str))
        else:
            print(f"episodes\t{public['episodes']}")
            print(f"nodes\t{public['nodes']}")
            print(f"edges\t{public['edges']}")
        return 0
    if args.command == "prefetch":
        from llm_wiki_mcp.prefetch import build_prefetch_cache

        data = build_prefetch_cache(limit=max(1, args.limit), write=not args.no_write)
        public = {
            key: value
            for key, value in data.items()
            if key not in {"buckets", "tokens"}
        }
        public["bucket_count"] = len(data.get("buckets", {}))
        public["token_count"] = len(data.get("tokens", {}))
        if args.json:
            print(json.dumps(public, ensure_ascii=False, indent=2, default=str))
        else:
            print(f"episodes\t{public['episodes']}")
            print(f"buckets\t{public['bucket_count']}")
            print(f"tokens\t{public['token_count']}")
        return 0
    if args.command == "sleep":
        from llm_wiki_mcp.sleep_cycle import render_summary, run_sleep_cycle

        data = run_sleep_cycle(
            raw_limit=max(0, args.raw_limit),
            eval_limit=max(0, args.eval_limit),
            duplicate_limit=max(0, args.duplicate_limit),
            dry_run=args.dry_run,
        )
        if args.json:
            print(json.dumps(data, ensure_ascii=False, indent=2, default=str))
        else:
            print(render_summary(data))
        return 0
    if args.command == "claims":
        from llm_wiki_mcp.claims import (
            rebuild_claim_index,
            sanitize_claim_ledger,
            search_claims,
        )

        if args.claims_command == "rebuild":
            data = rebuild_claim_index(limit=max(0, args.limit))
        elif args.claims_command == "sanitize":
            data = sanitize_claim_ledger(write=not args.no_write)
        else:
            data = {"claims": search_claims(args.query, limit=max(1, args.limit))}
        if args.json:
            print(json.dumps(data, ensure_ascii=False, indent=2, default=str))
        elif args.claims_command == "rebuild":
            print(f"claims\t{data['claims']}")
            print(f"path\t{data['path']}")
        elif args.claims_command == "sanitize":
            print(f"kept\t{data['kept']}")
            print(f"dropped\t{data['dropped']}")
        else:
            for row in data["claims"]:
                print(f"{row.get('score')}\t{row.get('claim_id')}\t{row.get('value')}")
        return 0
    if args.command == "golden-expand":
        from llm_wiki_mcp.golden_expand import expand_golden_from_recall_questions

        data = expand_golden_from_recall_questions(
            limit=max(0, args.limit),
            include_reference=args.include_reference,
            write=not args.no_write,
        )
        if args.json:
            print(json.dumps(data, ensure_ascii=False, indent=2, default=str))
        else:
            print(f"added\t{data['added']}")
            print(f"path\t{data['path']}")
        return 0
    if args.command == "retention":
        from llm_wiki_mcp.retention import build_retention_scores

        data = build_retention_scores(limit=max(1, args.limit), write=not args.no_write)
        public = {key: value for key, value in data.items() if key != "pages"}
        if args.json:
            print(json.dumps(public, ensure_ascii=False, indent=2, default=str))
        else:
            print(f"pages\t{public['counts']['pages']}")
            print(f"archive_candidates\t{public['counts']['archive_candidates']}")
        return 0
    if args.command == "reflect":
        from llm_wiki_mcp.reflection import write_reflection_page

        data = write_reflection_page(write=not args.no_write)
        if args.json:
            print(json.dumps(data, ensure_ascii=False, indent=2, default=str))
        else:
            print(f"path\t{data['path']}")
        return 0
    if args.command == "hubs":
        from llm_wiki_mcp.hubs import build_hub_pages

        data = build_hub_pages(
            min_pages=max(1, args.min_pages),
            max_hubs=max(1, args.max_hubs),
            write=not args.no_write,
        )
        if args.json:
            print(json.dumps(data, ensure_ascii=False, indent=2, default=str))
        else:
            print(f"hubs\t{data['hubs']}")
        return 0
    if args.command == "autonomy":
        from llm_wiki_mcp import autonomy

        if args.autonomy_command == "status":
            data = autonomy.status()
        elif args.autonomy_command == "watchdog":
            data = autonomy.watchdog_snapshot(notify=args.notify)
        elif args.autonomy_command == "digest":
            data = autonomy.build_digest(autonomy._read_json(autonomy.LATEST_FILE))
        else:
            data = autonomy.install_launchd(dry_run=args.dry_run, load=args.load)
        if args.json:
            print(json.dumps(data, ensure_ascii=False, indent=2, default=str))
        else:
            for key, value in data.items():
                if key in {"latest"}:
                    continue
                print(f"{key}\t{value}")
        return 0 if data.get("status") in {"ok", "alert"} else 1
    if args.command == "distill":
        from llm_wiki_mcp.distill import export_distill_dataset

        data = export_distill_dataset(
            limit=max(0, args.limit),
            include_reference=args.include_reference,
            write=not args.no_write,
        )
        if args.json:
            print(json.dumps(data, ensure_ascii=False, indent=2, default=str))
        else:
            print(f"rows\t{data['rows']}")
            print(f"path\t{data['path']}")
        return 0
    if args.command == "oracle":
        from llm_wiki_mcp.oracle import oracle_bundle

        data = oracle_bundle(
            args.query,
            top_n=max(1, args.top_n),
            claim_limit=max(1, args.claim_limit),
            ensure_claim_index=not args.no_index_build,
        )
        if args.json:
            print(json.dumps(data, ensure_ascii=False, indent=2, default=str))
        else:
            print(f"search_mode\t{data['search_mode']}")
            for page in data["pages"]:
                print(f"page\t{page['page_id']}\t{page['title']}")
            for claim in data["claims"]:
                print(f"claim\t{claim['claim_id']}\t{claim['value']}")
        return 0
    if args.command == "recall-eval":
        from llm_wiki_mcp.recall_eval import run_eval

        data = run_eval(
            config_file=Path(args.config).expanduser() if args.config else None,
            log_file=Path(args.log_file).expanduser(),
            feedback_file=Path(args.feedback_file).expanduser(),
            replay=True,
            save=args.save_baseline,
            overrides=args.config_override,
        )
        if args.json:
            print(json.dumps(data, ensure_ascii=False, indent=2, default=str))
        else:
            metrics = data["metrics"]
            print(f"examples\t{metrics['examples']}")
            print(f"recall@3\t{metrics['recall_at_3']:.3f}")
            print(f"waste_injection_rate\t{metrics['waste_injection_rate']:.3f}")
            print(f"latency_p95_ms\t{metrics['latency_ms']['p95']:.1f}")
        return 0
    if args.command == "recall-improve":
        from llm_wiki_mcp import recall_improvement

        if args.recall_improve_command == "run":
            data = recall_improvement.run_improvement(
                config_file=Path(args.config).expanduser() if args.config else None,
                log_file=Path(args.log_file).expanduser(),
                feedback_file=Path(args.feedback_file).expanduser(),
                models=args.models,
                apply=args.apply,
                include_heuristic=args.include_heuristic,
                min_improvement=max(0.0, args.min_improvement),
                max_examples=max(1, args.max_examples),
                frontier_mode=args.frontier,
                frontier_timeout=args.frontier_timeout,
            )
            if args.json:
                print(json.dumps(data, ensure_ascii=False, indent=2, default=str))
            else:
                print(f"run\t{data['run_id']}")
                print(f"status\t{data['status']}")
                print(f"applied\t{data['applied']}")
                print(f"reason\t{data['reason']}")
            return 0
        if args.recall_improve_command == "run-due":
            data = recall_improvement.run_due(
                config_file=Path(args.config).expanduser() if args.config else None,
                log_file=Path(args.log_file).expanduser(),
                feedback_file=Path(args.feedback_file).expanduser(),
                models=args.models,
                apply=args.apply,
                include_heuristic=args.include_heuristic,
                min_improvement=max(0.0, args.min_improvement),
                max_examples=max(1, args.max_examples),
                min_interval_hours=max(0.0, args.min_interval_hours),
                min_new_feedback=max(0, args.min_new_feedback),
                min_total_feedback=max(0, args.min_total_feedback),
                frontier_mode=args.frontier,
                frontier_timeout=args.frontier_timeout,
                dry_run=args.dry_run,
            )
            if args.json:
                print(json.dumps(data, ensure_ascii=False, indent=2, default=str))
            else:
                print(f"status\t{data.get('status')}")
                if data.get("result"):
                    print(
                        f"run\t{data['result'].get('run_id')}\t{data['result'].get('status')}"
                    )
            return 0
        if args.recall_improve_command == "status":
            data = recall_improvement.improvement_snapshot()
            if args.json:
                print(json.dumps(data, ensure_ascii=False, indent=2, default=str))
            else:
                active = data.get("active") or {}
                latest = data.get("latest") or {}
                print(f"status\t{data.get('status')}")
                print(f"active\t{active.get('run_id') or '--'}")
                print(
                    f"latest\t{latest.get('run_id') or '--'}\t{latest.get('status') or '--'}"
                )
            return 0
        if args.recall_improve_command == "rollback":
            data = recall_improvement.rollback_policy()
            if args.json:
                print(json.dumps(data, ensure_ascii=False, indent=2, default=str))
            else:
                print(
                    f"rollback\t{data['reason']}\t"
                    f"{data.get('from_run_id') or '--'} -> {data.get('to_run_id') or '--'}"
                )
            return 0
    return 0


def main(argv: list[str] | None = None) -> int:
    return dispatch(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
