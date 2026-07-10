from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from llm_wiki_mcp import frontier_review


CODEX_EXEC_HELP = """
Options:
  -C, --cd <DIR>
      --skip-git-repo-check
      --ephemeral
      --ignore-rules
  -s, --sandbox <SANDBOX_MODE>
      --output-schema <FILE>
  -o, --output-last-message <FILE>
"""


def test_frontier_timeout_is_capped_by_sleep_cycle_deadline(monkeypatch) -> None:
    monkeypatch.setenv("LLM_WIKI_CYCLE_DEADLINE_MONOTONIC", "112.9")
    monkeypatch.setattr(frontier_review.time, "monotonic", lambda: 100.0)

    assert frontier_review._bounded_timeout(3600) == 12


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
        output_option = "-o" if "-o" in cmd else "--output-last-message"
        output_path = Path(cmd[cmd.index(output_option) + 1])
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
        seen["cmd"] = cmd
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
    assert "--sandbox" in seen["cmd"]
    assert seen["cmd"][seen["cmd"].index("--sandbox") + 1] == "read-only"


def test_frontier_schema_requires_all_declared_properties() -> None:
    schema = frontier_review.FRONTIER_DECISION_SCHEMA

    assert set(schema["required"]) == set(schema["properties"])


def test_schema_strictness_autofix_normalizes_required_fields() -> None:
    schema = {
        "type": "object",
        "properties": {
            "decision": {"type": "string"},
            "summary": {"type": "string"},
        },
        "required": ["decision"],
    }

    strict_schema, repair = frontier_review._strict_schema_with_repair(schema)

    assert strict_schema["required"] == ["decision", "summary"]
    assert strict_schema["additionalProperties"] is False
    assert repair["type"] == "schema_strictness_autofix"


def test_classifies_frontier_failures() -> None:
    quota = frontier_review.classify_frontier_failure("insufficient_quota billing")
    transient = frontier_review.classify_frontier_failure("request timed out")
    option = frontier_review.classify_frontier_failure("unknown option --foo")

    assert quota.failure_class == "quota_or_billing_required"
    assert quota.human_required is True
    assert transient.failure_class == "network_transient"
    assert transient.rescue_status == "frontier_retry"
    assert option.failure_class == "cli_option_invalid"
    assert option.rescue_status == "pending_frontier_review"


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
        if "-o" in cmd or "--output-last-message" in cmd:
            return SimpleNamespace(
                returncode=1,
                stdout="",
                stderr="invalid_json_schema: Missing 'commit'",
            )
        return SimpleNamespace(returncode=0, stdout="diagnosis", stderr="")

    monkeypatch.delenv("CODEX_HOME", raising=False)
    monkeypatch.setenv("LLM_WIKI_FRONTIER_DOC_LOOKUP", "0")
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


def test_preflight_reports_adaptive_codex_exec_options(
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

    assert result["ok"] is True
    assert result["codex"]["adaptive_required"] is True
    assert "--output-schema" in result["codex"]["missing_exec_options"]


def test_run_codex_adapts_to_missing_cli_options(
    tmp_path: Path, monkeypatch
) -> None:
    home = tmp_path / "home"
    config_home = home / ".config" / "codex"
    config_home.mkdir(parents=True)
    (config_home / "auth.json").write_text("{}")
    seen: dict[str, object] = {}

    def fake_run(cmd, **kwargs):
        if cmd[-1:] == ["--version"]:
            return SimpleNamespace(returncode=0, stdout="codex 1.2.3", stderr="")
        if cmd[-2:] == ["exec", "--help"]:
            return SimpleNamespace(returncode=0, stdout="--ephemeral\n", stderr="")
        seen["cmd"] = cmd
        seen["cwd"] = kwargs.get("cwd")
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
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
            ),
            stderr="",
        )

    monkeypatch.delenv("CODEX_HOME", raising=False)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(frontier_review.shutil, "which", lambda _name: "/bin/codex")
    monkeypatch.setattr(frontier_review.subprocess, "run", fake_run)

    result = frontier_review._run_codex(
        "prompt", repo_root=tmp_path, timeout=1, execute_patch=False
    )

    assert result.decision == "approved"
    assert seen["cwd"] == str(tmp_path)
    assert "--output-schema" not in seen["cmd"]
    assert "-o" not in seen["cmd"]
    assert result.access_repair["applied"] is True
    assert {
        repair["option"]
        for repair in result.access_repair["repairs"]
        if repair["type"] == "cli_option_adapted"
    } >= {"--cd", "--output-schema", "--output-last-message"}


def test_collect_official_frontier_docs_uses_allowlist(monkeypatch) -> None:
    class FakeResponse:
        status_code = 200
        url = "https://platform.openai.com/docs"
        text = "<html><body><h1>Codex exec</h1><p>Use official commands.</p></body></html>"

        def raise_for_status(self) -> None:
            return None

    class FakeHttpx:
        @staticmethod
        def get(url, **_kwargs):
            assert frontier_review.is_allowed_official_url(url)
            return FakeResponse()

    monkeypatch.setenv(
        "LLM_WIKI_FRONTIER_DOC_URLS",
        "https://platform.openai.com/docs,https://example.com/bad",
    )
    monkeypatch.setitem(sys.modules, "httpx", FakeHttpx)

    result = frontier_review.collect_official_frontier_docs("codex exec option")

    assert result["attempted"] is True
    assert result["documents"]
    assert all(frontier_review.is_allowed_official_url(url) for url in result["allowlist"])


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
    monkeypatch.setenv("LLM_WIKI_FRONTIER_DOC_LOOKUP", "0")

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


@pytest.mark.parametrize("use_custom_command", [True, False])
@pytest.mark.parametrize(
    ("failure_kind", "expected_class", "expected_human_required"),
    [
        ("timeout", "network_transient", False),
        ("missing_executable", "frontier_tool_unavailable", True),
    ],
)
def test_structured_review_normalizes_subprocess_exceptions(
    tmp_path: Path,
    monkeypatch,
    use_custom_command: bool,
    failure_kind: str,
    expected_class: str,
    expected_human_required: bool,
) -> None:
    command_env = "LLM_WIKI_TEST_STRUCTURED_REVIEW_CMD"
    if use_custom_command:
        monkeypatch.setenv(command_env, "/missing/frontier-review")
    else:
        monkeypatch.delenv(command_env, raising=False)
        monkeypatch.setattr(frontier_review.shutil, "which", lambda _name: "/bin/codex")
        monkeypatch.setattr(
            frontier_review,
            "run_frontier_preflight",
            lambda: {
                "ok": True,
                "codex": {"exec_help": {"output": CODEX_EXEC_HELP}},
                "repairs": [],
            },
        )

    def fail_run(cmd, **kwargs):
        if failure_kind == "timeout":
            raise subprocess.TimeoutExpired(
                cmd=cmd,
                timeout=kwargs["timeout"],
                output="partial output",
                stderr="401 Unauthorized from stale child output",
            )
        raise FileNotFoundError(2, "No such file or directory", str(cmd[0]))

    monkeypatch.setattr(frontier_review.subprocess, "run", fail_run)
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["decision"],
        "properties": {"decision": {"type": "string"}},
    }

    result = frontier_review.run_structured_review(
        "review this",
        schema,
        repo_root=tmp_path,
        timeout=1,
        command_env=command_env,
    )

    assert result["decision"] == "needs_retry"
    assert result["frontier_failure"]["failure_class"] == expected_class
    assert result["frontier_failure"]["human_required"] is expected_human_required
    assert result["human_required"] is expected_human_required
    if failure_kind == "timeout":
        assert result["frontier_failure"]["rescue_status"] == "frontier_retry"
        assert result["frontier_failure"]["notify_user"] is False


def test_structured_review_rejects_incomplete_approved_json(
    tmp_path: Path, monkeypatch
) -> None:
    command_env = "LLM_WIKI_TEST_STRUCTURED_REVIEW_CMD"
    monkeypatch.setenv(command_env, "/bin/reviewer")
    monkeypatch.setattr(
        frontier_review.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout='{"decision":"approved"}',
            stderr="",
        ),
    )
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["decision", "confidence", "summary"],
        "properties": {
            "decision": {
                "type": "string",
                "enum": ["approved", "rejected", "needs_retry"],
            },
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "summary": {"type": "string"},
        },
    }

    result = frontier_review.run_structured_review(
        "review this",
        schema,
        repo_root=tmp_path,
        command_env=command_env,
    )

    assert result["decision"] == "needs_retry"
    assert result["confidence"] == 0.0
    assert result["frontier_failure"]["failure_class"] == "schema_invalid"
    assert result["human_required"] is False
