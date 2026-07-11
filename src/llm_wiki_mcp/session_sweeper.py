"""Recover session tails that Stop hooks missed or failed to process."""

from __future__ import annotations

import argparse
import json
import time
from argparse import Namespace
from datetime import datetime, timezone
from datetime import timedelta
from pathlib import Path
from typing import Any

from llm_wiki_mcp.link_fix import atomic_write
from llm_wiki_mcp.wiki import WIKI_ROOT

STATUS_FILE = WIKI_ROOT / "runtime" / "session-sweeper-latest.json"
STATE_FILE = WIKI_ROOT / "runtime" / "session-sweeper-state.json"


def _is_user_codex_session(path: Path) -> bool:
    """Exclude autonomous exec/MCP/subagent transcripts from memory capture."""
    try:
        with path.open(encoding="utf-8") as handle:
            first = json.loads(handle.readline())
    except (OSError, json.JSONDecodeError):
        return False
    if first.get("type") != "session_meta" or not isinstance(first.get("payload"), dict):
        return False
    meta = first["payload"]
    source = meta.get("source")
    thread_source = meta.get("thread_source")
    originator = str(meta.get("originator") or "")
    if thread_source == "subagent" or isinstance(source, dict):
        return False
    if source == "mcp":
        return False
    if originator == "codex_exec" and thread_source != "user":
        return False
    return True


def _is_user_claude_session(path: Path) -> bool:
    return path.name.startswith("agent-") is False and "subagents" not in path.parts


def _pending_codex(path: Path) -> bool:
    from llm_wiki_mcp import codex_save

    state = codex_save.load_state(codex_save.DEFAULT_STATE_FILE)
    after = codex_save.saved_line_for(state, path)
    return bool(codex_save.extract_transcript_slice(path, after_line=after).records)


def _pending_claude(path: Path) -> bool:
    from llm_wiki_mcp import claude_code_save

    state = claude_code_save.load_state(claude_code_save.DEFAULT_STATE_FILE)
    after = claude_code_save.saved_line_for(state, path)
    return bool(claude_code_save.extract_transcript_slice(path, after_line=after).records)


def discover_pending(*, idle_seconds: int = 300) -> list[tuple[str, Path]]:
    from llm_wiki_mcp import claude_code_save, codex_save

    cutoff = time.time() - max(0, idle_seconds)
    candidates: list[tuple[str, Path]] = []
    roots = (
        ("codex", codex_save.default_sessions_root(), _pending_codex, _is_user_codex_session),
        ("claude-code", claude_code_save.claude_code_projects_root(), _pending_claude, _is_user_claude_session),
    )
    for host, root, pending, is_user_session in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.jsonl"):
            try:
                if path.stat().st_mtime > cutoff or not is_user_session(path) or not pending(path):
                    continue
            except OSError:
                continue
            candidates.append((host, path))
    candidates.sort(key=lambda pair: pair[1].stat().st_mtime)
    return candidates


def _run_one(host: str, path: Path) -> dict[str, Any]:
    if host == "codex":
        from llm_wiki_mcp import codex_save as saver

        args = Namespace(
            session_id=None, cwd=None, session_file=str(path), sessions_root=None,
            state_file=str(saver.DEFAULT_STATE_FILE), model=saver.DEFAULT_MEMORY_MODEL,
            reasoning_effort=saver.DEFAULT_REASONING_EFFORT,
            max_chars=saver.DEFAULT_MAX_CHARS, timeout=saver.DEFAULT_TIMEOUT_SECONDS,
            dry_run=False, save=True, extract_only=False, ignore_state=False,
            hook=False, trigger_ingest=True,
        )
    else:
        from llm_wiki_mcp import claude_code_save as saver

        args = Namespace(
            session_id=None, session_file=str(path), state_file=str(saver.DEFAULT_STATE_FILE),
            model=saver.DEFAULT_MEMORY_MODEL, reasoning_effort=saver.DEFAULT_REASONING_EFFORT,
            max_chars=saver.DEFAULT_MAX_CHARS, timeout=saver.DEFAULT_TIMEOUT_SECONDS,
            dry_run=False, save=True, extract_only=False, ignore_state=False,
            hook=False, trigger_ingest=True,
        )
    return saver.run(args)


def run_sweeper(*, limit: int = 4, idle_seconds: int = 300, write: bool = True) -> dict[str, Any]:
    pending = discover_pending(idle_seconds=idle_seconds)
    try:
        state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        state = {"version": 1, "failures": {}}
    failures = state.get("failures")
    if not isinstance(failures, dict):
        failures = {}
        state["failures"] = failures
    results: list[dict[str, Any]] = []
    now = datetime.now(timezone.utc)
    for host, path in pending:
        if len(results) >= max(0, limit):
            break
        failure_state = failures.get(str(path))
        if isinstance(failure_state, dict):
            try:
                if datetime.fromisoformat(str(failure_state.get("next_retry_at"))) > now:
                    continue
            except (TypeError, ValueError):
                pass
        try:
            result = _run_one(host, path)
            results.append({"host": host, "session_file": str(path), **result})
            failures.pop(str(path), None)
        except Exception as exc:
            previous = failures.get(str(path))
            attempts = int(previous.get("attempts") or 0) + 1 if isinstance(previous, dict) else 1
            delay = min(6 * 3600, 60 * (2 ** min(attempts - 1, 8)))
            failures[str(path)] = {
                "attempts": attempts,
                "last_error": f"{exc.__class__.__name__}: {exc}",
                "updated_at": now.isoformat(timespec="seconds"),
                "next_retry_at": (now + timedelta(seconds=delay)).isoformat(timespec="seconds"),
            }
            results.append({
                "host": host,
                "session_file": str(path),
                "status": "error",
                "error": f"{exc.__class__.__name__}: {exc}",
            })
    payload = {
        "status": "ok" if not any(r.get("status") == "error" for r in results) else "attention",
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pending": len(pending),
        "processed": len(results),
        "results": results,
    }
    if write:
        STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
        atomic_write(STATE_FILE, json.dumps(state, ensure_ascii=False, indent=2) + "\n")
        atomic_write(STATUS_FILE, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--idle-seconds", type=int, default=300)
    args = parser.parse_args(argv)
    print(json.dumps(run_sweeper(limit=max(0, args.limit), idle_seconds=max(0, args.idle_seconds)), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
