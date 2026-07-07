"""Autonomous operation layer for LLM Wiki.

This module replaces human review queues with reversible machine decisions:
safe items are applied, uncertain items are deferred for the next cycle, and
health regressions quarantine the batch instead of waiting for a person.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import plistlib
import re
import shlex
import shutil
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from llm_wiki_mcp.frontmatter import patch as patch_frontmatter
from llm_wiki_mcp.wiki import WIKI_ROOT, find_page


AUTONOMY_DIR = WIKI_ROOT / "autonomy"
DECISIONS_FILE = AUTONOMY_DIR / "decisions.jsonl"
LATEST_FILE = AUTONOMY_DIR / "latest.json"
WATCHDOG_FILE = AUTONOMY_DIR / "watchdog-latest.json"
WATCHDOG_HISTORY = AUTONOMY_DIR / "watchdog-history.jsonl"
DIGEST_FILE = AUTONOMY_DIR / "digest-latest.md"
QUARANTINE_FILE = AUTONOMY_DIR / "quarantine.json"
PROJECT_ROOT = Path(__file__).resolve().parents[2]

SLEEP_LABEL = "com.trafficsign.llm-wiki-sleep"
WATCHDOG_LABEL = "com.trafficsign.llm-wiki-watchdog"
LAUNCH_AGENT_DIR = Path.home() / "Library" / "LaunchAgents"
WRAPPER_DIR = WIKI_ROOT / "bin"


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _latest_jsonl(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as f:
            lines = [line for line in f if line.strip()]
    except OSError:
        return {}
    if not lines:
        return {}
    try:
        row = json.loads(lines[-1])
    except json.JSONDecodeError:
        return {}
    return row if isinstance(row, dict) else {}


def _parse_dt(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _norm(text: object) -> str:
    if not isinstance(text, str):
        return ""
    text = text.casefold()
    text = re.sub(r"[\W_]+", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def _git(args: list[str], *, cwd: Path = WIKI_ROOT) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def _page_meta(page_id: str) -> dict[str, Any]:
    from llm_wiki_mcp.index_store import get_store

    store = get_store()
    store.refresh()
    return store.meta(page_id) or {}


def _page_quality(page_id: str, meta: dict[str, Any] | None = None) -> float:
    from llm_wiki_mcp.index_store import get_store

    meta = meta or _page_meta(page_id)
    if not meta:
        return -1.0
    score = 0.0
    if str(meta.get("summary") or "").strip():
        score += 3.0
    questions = meta.get("recall_questions")
    if isinstance(questions, list):
        score += min(3.0, len(questions) * 0.75)
    path_value = meta.get("path")
    if isinstance(path_value, str):
        try:
            size = Path(path_value).stat().st_size
            score += min(3.0, size / 3000.0)
        except OSError:
            pass
    try:
        store = get_store()
        score += min(2.0, len(store.backlinks(page_id)) * 0.3)
        score += min(1.0, len(store.outlinks(page_id)) * 0.1)
    except Exception:
        pass
    updated = _parse_dt(str(meta.get("updated") or ""))
    if updated and datetime.now() - updated <= timedelta(days=90):
        score += 0.5
    return score


def _patch_page_status(page_id: str, updates: dict[str, Any]) -> dict[str, Any]:
    path = find_page(page_id)
    if path is None:
        return {"status": "skipped", "reason": "page_not_found", "page_id": page_id}
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        return {"status": "skipped", "reason": f"read_error: {exc}", "page_id": page_id}
    new_text = patch_frontmatter(text, updates)
    if new_text == text:
        return {"status": "unchanged", "page_id": page_id, "path": str(path)}
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(new_text, encoding="utf-8")
    tmp.replace(path)
    return {"status": "applied", "page_id": page_id, "path": str(path), "updates": updates}


def decide_duplicate(record: dict[str, Any]) -> dict[str, Any]:
    """Return a reversible autonomous decision for one duplicate candidate."""
    left = str(record.get("left") or "")
    right = str(record.get("right") or "")
    score = float(record.get("score") or 0.0)
    left_title = str(record.get("left_title") or "")
    right_title = str(record.get("right_title") or "")
    decision = {
        "type": "duplicate_decision",
        "ts": _now(),
        "left": left,
        "right": right,
        "score": score,
        "method": record.get("method"),
        "action": "defer",
        "apply": False,
        "reason": "insufficient_confidence",
    }
    if not left or not right or left == right:
        decision["reason"] = "invalid_pair"
        return decision
    if _norm(left_title) != _norm(right_title):
        decision["reason"] = "title_mismatch"
        return decision
    if score < 0.995:
        decision["reason"] = "score_below_auto_supersede_threshold"
        return decision
    left_meta = _page_meta(left)
    right_meta = _page_meta(right)
    if not left_meta or not right_meta:
        decision["reason"] = "missing_page"
        return decision
    if left_meta.get("sensitivity") == "high" or right_meta.get("sensitivity") == "high":
        decision["reason"] = "high_sensitivity_page"
        return decision
    left_q = _page_quality(left, left_meta)
    right_q = _page_quality(right, right_meta)
    if abs(left_q - right_q) < 1.0:
        decision["reason"] = "quality_tie"
        decision["quality"] = {"left": round(left_q, 3), "right": round(right_q, 3)}
        return decision
    winner, loser = (left, right) if left_q > right_q else (right, left)
    decision.update(
        {
            "action": "supersede",
            "apply": True,
            "winner": winner,
            "loser": loser,
            "reason": "exact_title_high_score_quality_gap",
            "quality": {"left": round(left_q, 3), "right": round(right_q, 3)},
        }
    )
    return decision


def resolve_duplicate_candidates(
    records: list[dict[str, Any]],
    *,
    apply: bool = True,
    write: bool = True,
) -> dict[str, Any]:
    decisions: list[dict[str, Any]] = []
    applied = 0
    deferred = 0
    for record in records:
        decision = decide_duplicate(record)
        if decision.get("apply") and apply:
            result = _patch_page_status(
                str(decision["loser"]),
                {
                    "status": "deprecated",
                    "superseded_by": str(decision["winner"]),
                    "autonomy_decision": "duplicate_supersede",
                    "autonomy_decision_at": decision["ts"],
                },
            )
            decision["result"] = result
            if result.get("status") in {"applied", "unchanged"}:
                applied += 1
            else:
                decision["apply"] = False
                decision["action"] = "defer"
                decision["reason"] = f"apply_failed:{result.get('reason', result.get('status'))}"
                deferred += 1
        else:
            deferred += 1
        decisions.append(decision)
        if write:
            _append_jsonl(DECISIONS_FILE, decision)
    return {
        "status": "ok",
        "candidates": len(records),
        "applied": applied,
        "deferred": deferred,
        "decisions": decisions[:20],
    }


def apply_retention_archives(
    retention_payload: dict[str, Any],
    *,
    apply: bool = True,
    write: bool = True,
    limit: int = 25,
) -> dict[str, Any]:
    candidates = retention_payload.get("archive_candidates")
    if not isinstance(candidates, list):
        candidates = []
    pages = retention_payload.get("pages")
    pages = pages if isinstance(pages, dict) else {}
    decisions: list[dict[str, Any]] = []
    applied = 0
    for page_id in [str(item) for item in candidates[:limit] if isinstance(item, str)]:
        row = pages.get(page_id) if isinstance(pages.get(page_id), dict) else {}
        decision = {
            "type": "archive_decision",
            "ts": _now(),
            "page_id": page_id,
            "action": "archive",
            "apply": apply,
            "score": row.get("score"),
            "reason": "retention_archive_candidate",
        }
        if apply:
            result = _patch_page_status(
                page_id,
                {
                    "status": "archived",
                    "autonomy_decision": "retention_archive",
                    "autonomy_decision_at": decision["ts"],
                    "archive_reason": "low_retention_reversible_soft_archive",
                },
            )
            decision["result"] = result
            if result.get("status") in {"applied", "unchanged"}:
                applied += 1
            else:
                decision["apply"] = False
                decision["action"] = "defer"
                decision["reason"] = f"archive_failed:{result.get('reason', result.get('status'))}"
        decisions.append(decision)
        if write:
            _append_jsonl(DECISIONS_FILE, decision)
    return {
        "status": "ok",
        "candidates": len(candidates),
        "considered": len(decisions),
        "applied": applied,
        "decisions": decisions[:20],
    }


def _capture_rate(health: dict[str, Any]) -> float | None:
    memory = health.get("memory_integrity")
    if not isinstance(memory, dict):
        return None
    value = memory.get("capture_rate")
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _queue_value(health: dict[str, Any], key: str) -> int:
    queues = health.get("queues")
    if not isinstance(queues, dict):
        return 0
    try:
        return int(queues.get(key) or 0)
    except (TypeError, ValueError):
        return 0


def watchdog_snapshot(
    *,
    before_health: dict[str, Any] | None = None,
    write: bool = True,
    notify: bool = False,
    max_sleep_age_hours: float = 30.0,
    min_capture_rate: float = 0.80,
) -> dict[str, Any]:
    from llm_wiki_mcp.health import health_snapshot
    from llm_wiki_mcp.sleep_cycle import HISTORY_FILE

    health = health_snapshot()
    latest_sleep = _latest_jsonl(HISTORY_FILE)
    alerts: list[dict[str, Any]] = []
    capture = _capture_rate(health)
    if capture is not None and capture < min_capture_rate:
        alerts.append({"type": "capture_rate_low", "value": capture, "threshold": min_capture_rate})
    sleep_started = _parse_dt(latest_sleep.get("started_at")) if latest_sleep else None
    if not latest_sleep:
        alerts.append({"type": "sleep_never_ran"})
    elif latest_sleep.get("status") != "ok":
        alerts.append({"type": "sleep_status_not_ok", "status": latest_sleep.get("status")})
    elif sleep_started and datetime.now() - sleep_started > timedelta(hours=max_sleep_age_hours):
        alerts.append({"type": "sleep_stale", "started_at": latest_sleep.get("started_at")})
    if _queue_value(health, "duplicate_candidates") > 500:
        alerts.append({"type": "duplicate_backlog_high", "value": _queue_value(health, "duplicate_candidates")})
    if _queue_value(health, "lint_repair") > 2000:
        alerts.append({"type": "lint_backlog_high", "value": _queue_value(health, "lint_repair")})

    if before_health:
        before_capture = _capture_rate(before_health)
        if before_capture is not None and capture is not None and capture + 0.15 < before_capture:
            alerts.append({"type": "capture_rate_regression", "before": before_capture, "after": capture})

    payload = {
        "status": "alert" if alerts else "ok",
        "ts": _now(),
        "alerts": alerts,
        "health": health,
        "latest_sleep": latest_sleep,
    }
    if write:
        _write_json(WATCHDOG_FILE, payload)
        _append_jsonl(WATCHDOG_HISTORY, payload)
    if notify and alerts:
        _send_notification("LLM Wiki watchdog", f"{len(alerts)} autonomy alert(s)")
    return payload


def _send_notification(title: str, body: str) -> dict[str, Any]:
    script = f"display notification {json.dumps(body)} with title {json.dumps(title)}"
    try:
        proc = subprocess.run(["osascript", "-e", script], text=True, capture_output=True, timeout=5)
    except Exception as exc:
        return {"sent": False, "error": str(exc)}
    return {"sent": proc.returncode == 0, "stderr": proc.stderr.strip()}


def regression_guard(
    *,
    before_health: dict[str, Any],
    after_watchdog: dict[str, Any],
    wiki_snapshot: dict[str, Any],
    auto_revert: bool,
    write: bool = True,
) -> dict[str, Any]:
    alerts = [
        item for item in after_watchdog.get("alerts", [])
        if isinstance(item, dict) and item.get("type") in {"capture_rate_regression"}
    ]
    payload = {
        "status": "ok" if not alerts else "regression",
        "ts": _now(),
        "alerts": alerts,
        "auto_revert": auto_revert,
        "reverted": False,
    }
    head = str(wiki_snapshot.get("head") or "")
    if alerts and auto_revert and head:
        reset = _git(["reset", "--hard", head])
        payload["revert"] = {
            "command": "git reset --hard <snapshot_head>",
            "head": head,
            "returncode": reset.returncode,
            "stdout": reset.stdout.strip(),
            "stderr": reset.stderr.strip(),
        }
        payload["reverted"] = reset.returncode == 0
        if payload["reverted"]:
            quarantine = _read_json(QUARANTINE_FILE)
            quarantined = quarantine.get("actions")
            if not isinstance(quarantined, list):
                quarantined = []
            quarantined.append({"ts": payload["ts"], "reason": "capture_rate_regression", "head": head})
            _write_json(QUARANTINE_FILE, {"actions": quarantined[-100:]})
    if write:
        _append_jsonl(DECISIONS_FILE, {"type": "regression_guard", **payload})
    return payload


def build_digest(payload: dict[str, Any], *, write: bool = True) -> dict[str, Any]:
    watchdog = payload.get("watchdog") if isinstance(payload.get("watchdog"), dict) else {}
    duplicate = payload.get("duplicates") if isinstance(payload.get("duplicates"), dict) else {}
    retention = payload.get("archives") if isinstance(payload.get("archives"), dict) else {}
    guard = payload.get("regression_guard") if isinstance(payload.get("regression_guard"), dict) else {}
    lines = [
        "# LLM Wiki Autonomy Digest",
        "",
        f"- Generated: {_now()}",
        f"- Status: {watchdog.get('status', payload.get('status', 'unknown'))}",
        f"- Alerts: {len(watchdog.get('alerts', []) or [])}",
        f"- Duplicate decisions: {duplicate.get('applied', 0)} applied, {duplicate.get('deferred', 0)} deferred",
        f"- Retention archives: {retention.get('applied', 0)} applied",
        f"- Regression guard: {guard.get('status', 'unknown')}",
    ]
    if watchdog.get("alerts"):
        lines.extend(["", "## Alerts"])
        for alert in watchdog["alerts"]:
            lines.append(f"- `{alert.get('type', 'unknown')}` {json.dumps(alert, ensure_ascii=False)}")
    text = "\n".join(lines).rstrip() + "\n"
    if write:
        DIGEST_FILE.parent.mkdir(parents=True, exist_ok=True)
        DIGEST_FILE.write_text(text, encoding="utf-8")
    return {"status": "ok", "path": str(DIGEST_FILE), "text": text}


def run_autonomy_cycle(
    *,
    duplicates: list[dict[str, Any]],
    retention: dict[str, Any],
    before_health: dict[str, Any],
    wiki_snapshot: dict[str, Any],
    dry_run: bool = False,
    auto_revert: bool = True,
) -> dict[str, Any]:
    duplicate_result = resolve_duplicate_candidates(duplicates, apply=not dry_run, write=not dry_run)
    archive_result = apply_retention_archives(retention, apply=not dry_run, write=not dry_run)
    watchdog = watchdog_snapshot(before_health=before_health, write=not dry_run)
    guard = regression_guard(
        before_health=before_health,
        after_watchdog=watchdog,
        wiki_snapshot=wiki_snapshot,
        auto_revert=(not dry_run and auto_revert),
        write=not dry_run,
    )
    payload = {
        "status": "ok" if watchdog.get("status") == "ok" and guard.get("status") == "ok" else "attention",
        "ts": _now(),
        "dry_run": dry_run,
        "duplicates": duplicate_result,
        "archives": archive_result,
        "watchdog": watchdog,
        "regression_guard": guard,
    }
    digest = build_digest(payload, write=not dry_run)
    payload["digest"] = {k: v for k, v in digest.items() if k != "text"}
    if not dry_run:
        _write_json(LATEST_FILE, payload)
    return payload


def _uv_path() -> str:
    return shutil.which("uv") or "/opt/homebrew/bin/uv"


def _plist(label: str, args: list[str], *, stdout: Path, stderr: Path, start_interval: int | None = None, calendar: dict[str, int] | None = None) -> dict[str, Any]:
    data: dict[str, Any] = {
        "Label": label,
        "ProgramArguments": args,
        "WorkingDirectory": str(PROJECT_ROOT),
        "StandardOutPath": str(stdout),
        "StandardErrorPath": str(stderr),
        "RunAtLoad": False,
        "EnvironmentVariables": {
            "PATH": "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin",
            "PYTHONUNBUFFERED": "1",
        },
    }
    if start_interval is not None:
        data["StartInterval"] = start_interval
    if calendar is not None:
        data["StartCalendarInterval"] = calendar
    return data


def _write_plist(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    plistlib.dump(data, buf, sort_keys=False)
    path.write_bytes(buf.getvalue())


def _write_wrapper(path: Path, command: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    script = "#!/bin/sh\nexec " + shlex.join(command) + "\n"
    path.write_text(script, encoding="utf-8")
    path.chmod(0o755)


def install_launchd(*, dry_run: bool = False, load: bool = False) -> dict[str, Any]:
    logs = WIKI_ROOT / "logs"
    uv = _uv_path()
    sleep_path = LAUNCH_AGENT_DIR / f"{SLEEP_LABEL}.plist"
    watchdog_path = LAUNCH_AGENT_DIR / f"{WATCHDOG_LABEL}.plist"
    sleep_wrapper = WRAPPER_DIR / "llm-wiki-sleep"
    watchdog_wrapper = WRAPPER_DIR / "llm-wiki-watchdog"
    sleep_command = [
        uv,
        "run",
        "--project",
        str(PROJECT_ROOT),
        "llm-wiki",
        "sleep",
        "--raw-limit",
        "200",
        "--eval-limit",
        "150",
        "--duplicate-limit",
        "300",
        "--json",
    ]
    watchdog_command = [
        uv,
        "run",
        "--project",
        str(PROJECT_ROOT),
        "llm-wiki",
        "autonomy",
        "watchdog",
        "--notify",
        "--json",
    ]
    sleep_plist = _plist(
        SLEEP_LABEL,
        [str(sleep_wrapper)],
        stdout=logs / "sleep-cycle.launchd.out.log",
        stderr=logs / "sleep-cycle.launchd.err.log",
        calendar={"Hour": 3, "Minute": 40},
    )
    watchdog_plist = _plist(
        WATCHDOG_LABEL,
        [str(watchdog_wrapper)],
        stdout=logs / "watchdog.launchd.out.log",
        stderr=logs / "watchdog.launchd.err.log",
        start_interval=900,
    )
    payload: dict[str, Any] = {
        "status": "ok",
        "dry_run": dry_run,
        "load": load,
        "plists": [
            {"label": SLEEP_LABEL, "path": str(sleep_path), "program": sleep_plist["ProgramArguments"]},
            {"label": WATCHDOG_LABEL, "path": str(watchdog_path), "program": watchdog_plist["ProgramArguments"]},
        ],
        "wrappers": [
            {"path": str(sleep_wrapper), "command": sleep_command},
            {"path": str(watchdog_wrapper), "command": watchdog_command},
        ],
    }
    if not dry_run:
        logs.mkdir(parents=True, exist_ok=True)
        _write_wrapper(sleep_wrapper, sleep_command)
        _write_wrapper(watchdog_wrapper, watchdog_command)
        _write_plist(sleep_path, sleep_plist)
        _write_plist(watchdog_path, watchdog_plist)
    if load and not dry_run:
        uid = os.getuid()
        loads = []
        for path in (sleep_path, watchdog_path):
            subprocess.run(["launchctl", "bootout", f"gui/{uid}", str(path)], text=True, capture_output=True)
            proc = subprocess.run(["launchctl", "bootstrap", f"gui/{uid}", str(path)], text=True, capture_output=True)
            loads.append({"path": str(path), "returncode": proc.returncode, "stderr": proc.stderr.strip()})
        payload["launchctl"] = loads
        if any(item["returncode"] != 0 for item in loads):
            payload["status"] = "error"
    return payload


def status() -> dict[str, Any]:
    return {
        "status": "ok",
        "latest": _read_json(LATEST_FILE),
        "watchdog": _read_json(WATCHDOG_FILE),
        "digest_path": str(DIGEST_FILE),
        "launchd": {
            "sleep": str(LAUNCH_AGENT_DIR / f"{SLEEP_LABEL}.plist"),
            "watchdog": str(LAUNCH_AGENT_DIR / f"{WATCHDOG_LABEL}.plist"),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run or install LLM Wiki autonomous operation.")
    sub = parser.add_subparsers(dest="command", required=True)
    status_parser = sub.add_parser("status")
    status_parser.add_argument("--json", action="store_true")
    watchdog_parser = sub.add_parser("watchdog")
    watchdog_parser.add_argument("--notify", action="store_true")
    watchdog_parser.add_argument("--json", action="store_true")
    digest_parser = sub.add_parser("digest")
    digest_parser.add_argument("--json", action="store_true")
    install_parser = sub.add_parser("install-launchd")
    install_parser.add_argument("--dry-run", action="store_true")
    install_parser.add_argument("--load", action="store_true")
    install_parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "status":
        payload = status()
    elif args.command == "watchdog":
        payload = watchdog_snapshot(notify=args.notify)
    elif args.command == "digest":
        payload = build_digest(_read_json(LATEST_FILE))
    else:
        payload = install_launchd(dry_run=args.dry_run, load=args.load)
    if getattr(args, "json", False):
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    else:
        print("\t".join(f"{key}={value}" for key, value in payload.items() if key != "latest"))
    return 0 if payload.get("status") in {"ok", "alert"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
