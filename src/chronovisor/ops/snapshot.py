"""Git snapshot support for the wiki data directory."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from chronovisor.core.store import CHRONOVISOR_ROOT


def _git(args: list[str], *, cwd: Path = CHRONOVISOR_ROOT, check: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=check,
    )


def is_git_repo(path: Path = CHRONOVISOR_ROOT) -> bool:
    try:
        result = _git(["rev-parse", "--is-inside-work-tree"], cwd=path)
    except OSError:
        return False
    return result.returncode == 0 and result.stdout.strip() == "true"


def ensure_git_repo(path: Path = CHRONOVISOR_ROOT) -> dict[str, Any]:
    if is_git_repo(path):
        return {"initialized": False, "path": str(path)}
    result = _git(["init"], cwd=path)
    if result.returncode != 0:
        return {"initialized": False, "path": str(path), "error": result.stderr.strip()}
    # Local-only identity; harmless if already configured globally.
    _git(["config", "user.name", "Chronovisor"], cwd=path)
    _git(["config", "user.email", "chronovisor@localhost"], cwd=path)
    return {"initialized": True, "path": str(path)}


def snapshot_chronovisor(
    reason: str,
    *,
    path: Path = CHRONOVISOR_ROOT,
    allow_empty: bool = False,
) -> dict[str, Any]:
    """Commit current wiki state if there are changes."""
    init = ensure_git_repo(path)
    if init.get("error"):
        return {"status": "error", "stage": "init", **init}
    _git(["add", "-A"], cwd=path)
    status = _git(["status", "--porcelain"], cwd=path)
    if status.returncode != 0:
        return {"status": "error", "stage": "status", "error": status.stderr.strip()}
    if not status.stdout.strip() and not allow_empty:
        head = _git(["rev-parse", "--short", "HEAD"], cwd=path)
        return {
            "status": "clean",
            "path": str(path),
            "head": head.stdout.strip() if head.returncode == 0 else "",
        }
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    message = f"chronovisor snapshot: {reason} ({stamp})"
    commit = _git(["commit", "-m", message], cwd=path)
    if commit.returncode != 0:
        return {"status": "error", "stage": "commit", "error": commit.stderr.strip()}
    head = _git(["rev-parse", "--short", "HEAD"], cwd=path)
    return {
        "status": "committed",
        "path": str(path),
        "head": head.stdout.strip() if head.returncode == 0 else "",
        "message": message,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Snapshot ~/.chronovisor into its own git history."
    )
    parser.add_argument("reason", nargs="?", default="manual")
    parser.add_argument("--allow-empty", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    payload = snapshot_chronovisor(args.reason, allow_empty=args.allow_empty)
    if args.json:
        print(json.dumps(payload, ensure_ascii=False))
    else:
        print("\t".join(f"{key}={value}" for key, value in payload.items()))
    return 0 if payload.get("status") in {"clean", "committed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
