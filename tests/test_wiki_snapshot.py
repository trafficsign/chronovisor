from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from llm_wiki_mcp.wiki_snapshot import snapshot_wiki


@pytest.mark.skipif(shutil.which("git") is None, reason="git is required")
def test_snapshot_wiki_commits_changes_in_target_path(tmp_path: Path) -> None:
    (tmp_path / "pages").mkdir()
    (tmp_path / "pages" / "alpha.md").write_text("# Alpha\n", encoding="utf-8")

    payload = snapshot_wiki("test", path=tmp_path)

    assert payload["status"] == "committed"
    head = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=tmp_path,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert head.returncode == 0
    assert head.stdout.strip() == payload["head"]
