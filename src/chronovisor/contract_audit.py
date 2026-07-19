"""Reject non-canonical product contracts in maintained source files."""

from __future__ import annotations

import fnmatch
import json
from pathlib import Path
from typing import Any


def _allowed(path: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(path, pattern) for pattern in patterns)


def audit(repo_root: Path) -> dict[str, Any]:
    manifest_path = repo_root / "chronovisor-contract-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    policy = manifest["audit"]
    violations: list[dict[str, Any]] = []
    scanned = 0
    ignored_roots = {".git", ".venv", ".pytest_cache", ".mypy_cache", "logs"}
    for path in sorted(repo_root.rglob("*")):
        if not path.is_file() or any(part in ignored_roots for part in path.parts):
            continue
        relative = path.relative_to(repo_root).as_posix()
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue
        scanned += 1
        if not _allowed(relative, policy["allowlisted_paths"]):
            for token in policy["forbidden"]:
                if token in text:
                    violations.append({"path": relative, "token": token})
    return {
        "schema": "chronovisor.contract-audit.v2",
        "status": "ok" if not violations else "violation",
        "scanned_files": scanned,
        "violations": violations,
    }


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    result = audit(repo_root)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
