from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from chronovisor.ops.snapshot import snapshot_chronovisor


@pytest.mark.skipif(shutil.which("git") is None, reason="git is required")
def test_snapshot_chronovisor_commits_changes_in_target_path(tmp_path: Path) -> None:
    (tmp_path / "pages").mkdir()
    (tmp_path / "pages" / "alpha.md").write_text("# Alpha\n", encoding="utf-8")

    payload = snapshot_chronovisor("test", path=tmp_path)

    assert payload["status"] == "committed"
    head = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert head.returncode == 0
    assert head.stdout.strip() == payload["head"]
