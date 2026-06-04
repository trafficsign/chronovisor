from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from llm_wiki_mcp import frontier_review


def test_run_codex_supplies_codex_home_from_config_dir(
    tmp_path: Path, monkeypatch
) -> None:
    seen: dict[str, object] = {}
    home = tmp_path / "home"
    config_home = home / ".config" / "codex"
    config_home.mkdir(parents=True)

    def fake_run(cmd, **kwargs):
        output_path = Path(cmd[cmd.index("-o") + 1])
        output_path.write_text(
            json.dumps(
                {
                    "decision": "approved",
                    "summary": "ok",
                    "tests_run": [],
                    "commit": None,
                    "committed": False,
                    "pushed": False,
                    "risk": None,
                    "notes": None,
                }
            )
        )
        seen["env"] = kwargs["env"]
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.delenv("CODEX_HOME", raising=False)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(frontier_review.shutil, "which", lambda _name: "/bin/codex")
    monkeypatch.setattr(frontier_review.subprocess, "run", fake_run)

    result = frontier_review._run_codex(
        "prompt", repo_root=tmp_path, timeout=1, execute_patch=False
    )

    assert result.decision == "approved"
    assert seen["env"]["CODEX_HOME"] == str(config_home)
