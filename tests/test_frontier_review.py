from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from llm_wiki_mcp import frontier_review


CODEX_EXEC_HELP = """
Options:
  -C, --cd <DIR>
      --skip-git-repo-check
      --ephemeral
      --ignore-rules
      --output-schema <FILE>
  -o, --output-last-message <FILE>
"""


def _preflight_response(cmd: list[str]) -> SimpleNamespace | None:
    if cmd[-1:] == ["--version"]:
        return SimpleNamespace(returncode=0, stdout="codex 1.2.3", stderr="")
    if cmd[-2:] == ["exec", "--help"]:
        return SimpleNamespace(returncode=0, stdout=CODEX_EXEC_HELP, stderr="")
    return None


def test_run_codex_supplies_codex_home_from_config_dir(
    tmp_path: Path, monkeypatch
) -> None:
    seen: dict[str, object] = {}
    home = tmp_path / "home"
    config_home = home / ".config" / "codex"
    config_home.mkdir(parents=True)
    (config_home / "auth.json").write_text("{}")

    def fake_run(cmd, **kwargs):
        preflight = _preflight_response(cmd)
        if preflight:
            return preflight
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


def test_frontier_schema_requires_all_declared_properties() -> None:
    schema = frontier_review.FRONTIER_DECISION_SCHEMA

    assert set(schema["required"]) == set(schema["properties"])


def test_run_codex_missing_auth_stops_before_subprocess(
    tmp_path: Path, monkeypatch
) -> None:
    home = tmp_path / "home"
    (home / ".config" / "codex").mkdir(parents=True)

    def fake_run(cmd, **_kwargs):
        preflight = _preflight_response(cmd)
        if preflight:
            return preflight
        raise AssertionError("codex exec should not run without auth")

    monkeypatch.delenv("CODEX_HOME", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(frontier_review.shutil, "which", lambda _name: "/bin/codex")
    monkeypatch.setattr(frontier_review.subprocess, "run", fake_run)

    result = frontier_review._run_codex(
        "prompt", repo_root=tmp_path, timeout=1, execute_patch=False
    )

    assert result.human_required is True
    assert result.rescue_status == "human_required"
    assert result.frontier_failure["failure_class"] == "auth_required"


def test_run_codex_schema_failure_records_rescue_attempt(
    tmp_path: Path, monkeypatch
) -> None:
    home = tmp_path / "home"
    config_home = home / ".config" / "codex"
    config_home.mkdir(parents=True)
    (config_home / "auth.json").write_text("{}")
    calls: list[list[str]] = []

    def fake_run(cmd, **_kwargs):
        calls.append(cmd)
        preflight = _preflight_response(cmd)
        if preflight:
            return preflight
        if "-o" in cmd:
            return SimpleNamespace(
                returncode=1,
                stdout="",
                stderr="invalid_json_schema: Missing 'commit'",
            )
        return SimpleNamespace(returncode=0, stdout="diagnosis", stderr="")

    monkeypatch.delenv("CODEX_HOME", raising=False)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(frontier_review.shutil, "which", lambda _name: "/bin/codex")
    monkeypatch.setattr(frontier_review.subprocess, "run", fake_run)

    result = frontier_review._run_codex(
        "prompt", repo_root=tmp_path, timeout=1, execute_patch=False
    )

    codex_exec_calls = [cmd for cmd in calls if cmd[:2] == ["/bin/codex", "exec"]]
    assert len(codex_exec_calls) == 3
    assert result.rescue_status == "pending_frontier_review"
    assert result.frontier_failure["failure_class"] == "schema_invalid"
    assert result.rescue_attempt["attempted"] is True
    assert result.rescue_attempt["ok"] is True


def test_redacts_secrets_and_allows_only_official_urls() -> None:
    text = "Authorization: Bearer sk-this-secret-should-disappear"

    redacted = frontier_review.redact_sensitive_text(text)

    assert "sk-this-secret" not in redacted
    assert (
        frontier_review.redact_sensitive_text("sandbox_permissions=[\"disk-full-read-access\"]")
        == "sandbox_permissions=[\"disk-full-read-access\"]"
    )
    assert frontier_review.is_allowed_official_url("https://platform.openai.com/docs")
    assert frontier_review.is_allowed_official_url("https://docs.anthropic.com/en/docs")
    assert frontier_review.is_allowed_official_url("https://github.com/openai/codex")
    assert not frontier_review.is_allowed_official_url("https://openai.example.com/docs")
    assert not frontier_review.is_allowed_official_url("https://github.com/random/repo")


def test_preflight_detects_incompatible_codex_exec_options(
    tmp_path: Path, monkeypatch
) -> None:
    home = tmp_path / "home"
    config_home = home / ".config" / "codex"
    config_home.mkdir(parents=True)
    (config_home / "auth.json").write_text("{}")

    def fake_run(cmd, **_kwargs):
        if cmd[-1:] == ["--version"]:
            return SimpleNamespace(returncode=0, stdout="codex 1.2.3", stderr="")
        if cmd[-2:] == ["exec", "--help"]:
            return SimpleNamespace(returncode=0, stdout="--cd\n--ephemeral\n", stderr="")
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.delenv("CODEX_HOME", raising=False)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(frontier_review.shutil, "which", lambda _name: "/bin/codex")
    monkeypatch.setattr(frontier_review.subprocess, "run", fake_run)

    result = frontier_review.run_frontier_preflight()

    assert result["ok"] is False
    assert result["failure"]["failure_class"] == "cli_option_invalid"
    assert result["failure"]["rescue_status"] == "pending_frontier_review"


def test_frontier_rescue_falls_back_to_claude_code(
    tmp_path: Path, monkeypatch
) -> None:
    calls: list[list[str]] = []

    def fake_which(name: str) -> str | None:
        if name == "codex":
            return "/bin/codex"
        if name == "claude":
            return "/bin/claude"
        return None

    def fake_run(cmd, **_kwargs):
        calls.append(cmd)
        if cmd[0] == "/bin/codex":
            return SimpleNamespace(returncode=1, stdout="", stderr="codex rescue failed")
        return SimpleNamespace(returncode=0, stdout="claude diagnosis", stderr="")

    monkeypatch.setattr(frontier_review.shutil, "which", fake_which)
    monkeypatch.setattr(frontier_review.subprocess, "run", fake_run)

    result = frontier_review._run_frontier_rescue(
        "invalid_json_schema",
        "prompt",
        repo_root=tmp_path,
        timeout=1,
    )

    assert result["attempted"] is True
    assert result["ok"] is True
    assert [attempt["reviewer"] for attempt in result["attempts"]] == [
        "codex",
        "claude-code",
    ]
    assert calls[0][0] == "/bin/codex"
    assert calls[1][0] == "/bin/claude"
