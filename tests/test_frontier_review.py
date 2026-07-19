from __future__ import annotations

import json
import sys
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from chronovisor import (
    decision_router,
    frontier_review,
    ollama,
    semantic_hold,
)
from chronovisor.decision_router import DecisionRouterResult
from chronovisor.runtime_config import DecisionRouterConfig
from tests.semantic_hold_support import semantic_authority, semantic_review


CODEX_EXEC_HELP = """
Options:
  -C, --cd <DIR>
      --skip-git-repo-check
      --ephemeral
      --ignore-rules
      --disable <FEATURE>
  -s, --sandbox <SANDBOX_MODE>
      --output-schema <FILE>
  -o, --output-last-message <FILE>
"""


class _StartedPermit:
    def __init__(self) -> None:
        self.status = "reserved"

    def start(self, *, pid: int) -> None:
        assert pid > 0
        self.status = "started"


def _local_router_config() -> DecisionRouterConfig:
    return DecisionRouterConfig(
        primary_model="ornith:test",
        challenger_model="gpt-oss:test",
        tie_break_model="gemma:test",
        num_ctx=16_384,
        num_predict=256,
        read_timeout_ms=5000,
        max_input_chars=20_000,
        max_output_chars=1_000,
        max_feedback_chars=2_000,
        adaptive_residency=False,
    )


def _quarantined_router_class(
    *,
    valid: tuple[bool, bool, bool],
    signatures: tuple[str | None, str | None, str | None],
    reason: str,
    failure_class: str = "local_consensus_failed",
):
    router_audit = semantic_authority(artifact_sha256="d" * 64)["router"]
    assert isinstance(router_audit, dict)
    votes = tuple(
        SimpleNamespace(valid=is_valid, signature_sha256=signature)
        for is_valid, signature in zip(valid, signatures, strict=True)
    )

    class QuarantinedLocalRouter:
        def __init__(self, **_kwargs) -> None:
            self.policy = SimpleNamespace(
                source="adopted_artifact",
                audit_record=lambda: dict(router_audit),
            )

        def decide(self, _prompt, _schema):
            return SimpleNamespace(
                ok=False,
                decision=None,
                failure_class=failure_class,
                quarantine_reason=reason,
                votes=votes,
                audit_record=lambda: {
                    "status": "quarantined",
                    "ok": False,
                    "failure_class": failure_class,
                    "quarantine_reason": reason,
                    "votes": [
                        {
                            "valid": vote.valid,
                            "signature_sha256": vote.signature_sha256,
                        }
                        for vote in votes
                    ],
                },
            )

    return QuarantinedLocalRouter


def _cache_test_router_class(
    authority_box: dict[str, dict[str, object]],
    calls: list[str],
    *,
    outcome: str = "semantic",
):
    class CacheTestRouter:
        def __init__(self, **_kwargs) -> None:
            authority = authority_box["value"]
            router = authority["router"]
            assert isinstance(router, dict)
            self.policy = SimpleNamespace(
                source="adopted_artifact",
                audit_record=lambda: dict(router),
            )

        def decide(self, prompt, _schema, *, system=None):
            calls.append(f"{prompt}|{system}")
            authority = authority_box["value"]
            review = semantic_review(authority)
            consensus = review["local_consensus"]
            assert isinstance(consensus, dict)
            vote_rows = consensus["votes"]
            assert isinstance(vote_rows, list)
            votes = tuple(
                SimpleNamespace(
                    valid=bool(vote["valid"]),
                    signature_sha256=vote["signature_sha256"],
                )
                for vote in vote_rows
            )
            if outcome == "success":
                decision = {
                    "decision": "approved",
                    "summary": "approved locally",
                    "tests_run": [],
                    "commit": None,
                    "committed": False,
                    "pushed": False,
                    "risk": None,
                    "notes": None,
                }
                return SimpleNamespace(
                    ok=True,
                    decision=decision,
                    failure_class=None,
                    quarantine_reason=None,
                    votes=votes[:2],
                    audit_record=lambda: {
                        "status": "agreed",
                        "ok": True,
                        "votes": vote_rows[:2],
                    },
                )
            if outcome == "operational":
                return SimpleNamespace(
                    ok=False,
                    decision=None,
                    failure_class="local_consensus_failed",
                    quarantine_reason="fewer_than_two_valid_local_votes",
                    votes=votes[:2],
                    audit_record=lambda: {
                        **consensus,
                        "quarantine_reason": "fewer_than_two_valid_local_votes",
                        "votes": vote_rows[:2],
                    },
                )
            return SimpleNamespace(
                ok=False,
                decision=None,
                failure_class="local_consensus_failed",
                quarantine_reason="local_models_did_not_reach_two_vote_quorum",
                votes=votes,
                audit_record=lambda: consensus,
            )

    return CacheTestRouter


@pytest.fixture(autouse=True)
def isolate_frontier_activity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Other modules exercise context admission under patched contract inputs.
    # Keep this file independent from that process-global derived cache.
    decision_router._minimum_model_backed_context_tokens.cache_clear()
    monkeypatch.setattr(
        frontier_review,
        "FRONTIER_ACTIVITY_DIR",
        tmp_path / "frontier-activity",
    )
    monkeypatch.setattr(
        frontier_review.runtime_status,
        "safe_append_event",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        frontier_review,
        "STRUCTURED_REVIEW_HOLD_CACHE_ROOT",
        tmp_path / "structured-review-holds",
    )
    monkeypatch.setattr(
        frontier_review,
        "_structured_authority_observation",
        lambda _authority: "0" * 64,
    )
    monkeypatch.setattr(
        frontier_review,
        "_current_structured_authority",
        lambda lane: (
            semantic_authority(lane, artifact_sha256="d" * 64),
            None,
        ),
    )


def test_frontier_timeout_is_capped_by_sleep_cycle_deadline(monkeypatch) -> None:
    monkeypatch.setenv("CHRONOVISOR_CYCLE_DEADLINE_MONOTONIC", "112.9")
    monkeypatch.setattr(frontier_review.time, "monotonic", lambda: 100.0)

    assert frontier_review._bounded_timeout(3600) == 12


def test_frontier_invocation_disables_hooks_and_uses_selected_model(
    tmp_path: Path,
) -> None:
    invocation = frontier_review._build_codex_exec_invocation(
        "/bin/codex",
        repo_root=tmp_path,
        schema_path=tmp_path / "schema.json",
        output_path=tmp_path / "output.json",
        execute_patch=False,
        preflight={"codex": {"exec_help": {"output": CODEX_EXEC_HELP}}},
        model="gpt-5.6-luna",
        reasoning_effort="low",
    )

    cmd = invocation["cmd"]
    assert cmd[cmd.index("--disable") : cmd.index("--disable") + 2] == [
        "--disable",
        "hooks",
    ]
    assert cmd[cmd.index("--model") : cmd.index("--model") + 2] == [
        "--model",
        "gpt-5.6-luna",
    ]
    assert 'model_reasoning_effort="low"' in cmd


def test_frontier_env_marks_internal_children_and_disables_stop_work(
    monkeypatch,
) -> None:
    monkeypatch.delenv("CHRONOVISOR_INTERNAL_FRONTIER", raising=False)
    env = frontier_review._frontier_env()

    assert env["CHRONOVISOR_INTERNAL_FRONTIER"] == "1"
    assert env["CODEX_CHRONOVISOR_RECORD_ENABLED"] == "0"
    assert env["CHRONOVISOR_CONTENT_CORRECTION_ENABLED"] == "0"


def _preflight_response(cmd: list[str]) -> SimpleNamespace | None:
    if cmd[-1:] == ["--version"]:
        return SimpleNamespace(returncode=0, stdout="codex 1.2.3", stderr="")
    if cmd[-2:] == ["exec", "--help"]:
        return SimpleNamespace(returncode=0, stdout=CODEX_EXEC_HELP, stderr="")
    return None


def _use_fake_guarded_spawn(monkeypatch, fake_run) -> None:
    def fake_spawn(cmd, *, permit, **kwargs):
        permit.start(pid=12345)
        return fake_run(cmd, **kwargs)

    monkeypatch.setattr(frontier_review, "_spawn_guarded_process", fake_spawn)


def test_run_codex_uses_isolated_codex_home_with_shared_auth(
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
        isolated_home = Path(kwargs["env"]["CODEX_HOME"])
        seen["isolated_home_differs"] = isolated_home != config_home
        seen["auth_resolves_to_source"] = (isolated_home / "auth.json").resolve() == (
            config_home / "auth.json"
        ).resolve()
        seen["config"] = (isolated_home / "config.toml").read_text()
        seen["has_hooks"] = (isolated_home / "hooks.json").exists()
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.delenv("CODEX_HOME", raising=False)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(frontier_review.shutil, "which", lambda _name: "/bin/codex")
    monkeypatch.setattr(frontier_review.subprocess, "run", fake_run)
    _use_fake_guarded_spawn(monkeypatch, fake_run)

    result = frontier_review._run_codex(
        "prompt",
        repo_root=tmp_path,
        timeout=1,
        execute_patch=False,
        permit=_StartedPermit(),
    )

    assert result.decision == "approved"
    assert seen["isolated_home_differs"] is True
    assert seen["auth_resolves_to_source"] is True
    assert "mcp_servers" not in str(seen["config"])
    assert seen["has_hooks"] is False
    assert "--sandbox" in seen["cmd"]
    assert seen["cmd"][seen["cmd"].index("--sandbox") + 1] == "read-only"


def test_frontier_activity_file_exists_only_while_review_runs(tmp_path: Path) -> None:
    activity_dir = frontier_review.FRONTIER_ACTIVITY_DIR

    with frontier_review._frontier_activity(
        kind="semantic_judge",
        reviewer="codex",
        model="gpt-test",
        prompt="review this",
        repo_root=tmp_path,
    ):
        files = list(activity_dir.glob("*.json"))
        assert len(files) == 1
        record = json.loads(files[0].read_text())
        assert record["active"] is True
        assert record["model"] == "gpt-test"
        assert "review this" not in files[0].read_text()

    assert list(activity_dir.glob("*.json")) == []


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
    secret_store = frontier_review.classify_frontier_failure(
        "credential store access denied"
    )
    transient = frontier_review.classify_frontier_failure("request timed out")
    option = frontier_review.classify_frontier_failure("unknown option --foo")

    assert quota.failure_class == "quota_or_billing_required"
    assert quota.human_required is True
    assert secret_store.failure_class == "secret_store_permission_required"
    assert secret_store.human_required is True
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
    _use_fake_guarded_spawn(monkeypatch, fake_run)

    result = frontier_review._run_codex(
        "prompt",
        repo_root=tmp_path,
        timeout=1,
        execute_patch=False,
        permit=_StartedPermit(),
    )

    assert result.human_required is True
    assert result.rescue_status == "human_required"
    assert result.frontier_failure["failure_class"] == "auth_required"


def test_run_codex_schema_failure_does_not_spawn_an_unguarded_rescue_session(
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
    monkeypatch.setenv("CHRONOVISOR_FRONTIER_DOC_LOOKUP", "0")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(frontier_review.shutil, "which", lambda _name: "/bin/codex")
    monkeypatch.setattr(frontier_review.subprocess, "run", fake_run)
    _use_fake_guarded_spawn(monkeypatch, fake_run)

    permit = _StartedPermit()
    result = frontier_review._run_codex(
        "prompt",
        repo_root=tmp_path,
        timeout=1,
        execute_patch=False,
        permit=permit,
    )

    code_repair_calls = [
        cmd
        for cmd in calls
        if cmd[:2] == ["/bin/codex", "exec"] and cmd[-2:] != ["exec", "--help"]
    ]
    assert len(code_repair_calls) == 1
    assert permit.status == "started"
    assert result.rescue_status == "pending_frontier_review"
    assert result.frontier_failure["failure_class"] == "schema_invalid"
    assert result.rescue_attempt is None


def test_redacts_secrets_and_allows_only_official_urls() -> None:
    text = "Authorization: Bearer sk-this-secret-should-disappear"

    redacted = frontier_review.redact_sensitive_text(text)

    assert "sk-this-secret" not in redacted
    assert (
        frontier_review.redact_sensitive_text(
            'sandbox_permissions=["disk-full-read-access"]'
        )
        == 'sandbox_permissions=["disk-full-read-access"]'
    )
    assert frontier_review.is_allowed_official_url("https://platform.openai.com/docs")
    assert frontier_review.is_allowed_official_url("https://docs.anthropic.com/en/docs")
    assert frontier_review.is_allowed_official_url("https://github.com/openai/codex")
    assert not frontier_review.is_allowed_official_url(
        "https://openai.example.com/docs"
    )
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
            return SimpleNamespace(
                returncode=0, stdout="--cd\n--ephemeral\n", stderr=""
            )
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.delenv("CODEX_HOME", raising=False)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(frontier_review.shutil, "which", lambda _name: "/bin/codex")
    monkeypatch.setattr(frontier_review.subprocess, "run", fake_run)
    _use_fake_guarded_spawn(monkeypatch, fake_run)

    result = frontier_review.run_frontier_preflight()

    assert result["ok"] is True
    assert result["codex"]["adaptive_required"] is True
    assert "--output-schema" in result["codex"]["missing_exec_options"]


def test_run_codex_adapts_to_missing_cli_options(tmp_path: Path, monkeypatch) -> None:
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
    _use_fake_guarded_spawn(monkeypatch, fake_run)

    result = frontier_review._run_codex(
        "prompt",
        repo_root=tmp_path,
        timeout=1,
        execute_patch=False,
        permit=_StartedPermit(),
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
        text = (
            "<html><body><h1>Codex exec</h1><p>Use official commands.</p></body></html>"
        )

        def raise_for_status(self) -> None:
            return None

    class FakeHttpx:
        @staticmethod
        def get(url, **_kwargs):
            assert frontier_review.is_allowed_official_url(url)
            return FakeResponse()

    monkeypatch.setenv(
        "CHRONOVISOR_FRONTIER_DOC_URLS",
        "https://platform.openai.com/docs,https://example.com/bad",
    )
    monkeypatch.setitem(sys.modules, "httpx", FakeHttpx)

    result = frontier_review.collect_official_frontier_docs("codex exec option")

    assert result["attempted"] is True
    assert result["documents"]
    assert all(
        frontier_review.is_allowed_official_url(url) for url in result["allowlist"]
    )


def test_frontier_without_validated_evidence_starts_no_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*_args, **_kwargs):
        raise AssertionError("unvalidated frontier request must start no process")

    monkeypatch.setattr(frontier_review, "_run_codex", forbidden)
    monkeypatch.setattr(frontier_review.subprocess, "run", forbidden)
    monkeypatch.setenv("CHRONOVISOR_FRONTIER_CMD", "/bin/forbidden")

    result = frontier_review.run_frontier_review(
        {"failure_class": "model_json_invalid"},
        None,
        repo_root=tmp_path,
    )

    assert result.decision == "needs_retry"
    assert (
        result.frontier_failure["failure_class"] == "frontier_guard_evidence_required"
    )
    assert not hasattr(frontier_review, "_run_codex_rescue")
    assert not hasattr(frontier_review, "_run_claude_code_rescue")
    assert not hasattr(frontier_review, "_run_frontier_rescue")


def test_disabled_repair_lane_starts_no_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*_args, **_kwargs):
        raise AssertionError("disabled repair lane must not inspect or start Codex")

    monkeypatch.setenv("CHRONOVISOR_DECISION_POLICY_SYSTEM_CODE_REPAIR", "off")
    monkeypatch.setattr(frontier_review, "_capture_repair_baseline", forbidden)
    monkeypatch.setattr(frontier_review, "_run_codex", forbidden)

    result = frontier_review.run_frontier_review(
        {"failure_class": "system_health_snapshot_exception"},
        None,
        repo_root=tmp_path,
        evidence=object(),
    )

    assert result.execution_started is False
    assert result.frontier_failure["failure_class"] == "frontier_repair_policy_disabled"


def test_missing_reproduction_command_is_rejected_before_baseline_or_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chronovisor.frontier_guard import (
        RepairIncidentEvidence,
        repair_fingerprint,
    )

    incident = RepairIncidentEvidence(
        component="watchdog.health_snapshot",
        fingerprint=repair_fingerprint("watchdog.health_snapshot", "missing-command"),
        failure_class="system_health_snapshot_exception",
        occurrence_count=3,
        distinct_inputs=("input-a", "input-b"),
        local_repair_attempts=2,
        local_repair_evidence=("a" * 64, "b" * 64),
        reproduction_command=("pytest", "tests/test_ingest.py"),
        notes={
            "producer": "trusted_watchdog",
            "incident_key": "missing-command",
        },
    )
    object.__setattr__(incident, "reproduction_command", ())

    def forbidden(*_args, **_kwargs):
        raise AssertionError("invalid reproduction evidence must spend no token")

    monkeypatch.setattr(frontier_review, "_capture_repair_baseline", forbidden)
    monkeypatch.setattr(frontier_review, "_run_codex", forbidden)

    result = frontier_review.run_frontier_review(
        {"failure_class": "system_health_snapshot_exception"},
        None,
        repo_root=tmp_path,
        evidence=incident,
    )

    assert result.execution_started is False
    assert result.frontier_failure["failure_class"] == (
        "frontier_guard_evidence_invalid"
    )


def test_validated_repair_incident_gets_exactly_one_guarded_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chronovisor.frontier_guard import (
        FrontierGuard,
        RepairIncidentEvidence,
        repair_fingerprint,
    )

    evidence = RepairIncidentEvidence(
        component="watchdog.health_snapshot",
        fingerprint=repair_fingerprint("watchdog.health_snapshot", "health-error"),
        failure_class="system_health_snapshot_exception",
        occurrence_count=3,
        distinct_inputs=("input-a", "input-b"),
        local_repair_attempts=2,
        local_repair_evidence=("a" * 64, "b" * 64),
        reproduction_command=("pytest", "tests/test_ingest.py", "-k", "invalid_json"),
        notes={"producer": "trusted_watchdog", "incident_key": "frontier-review-test"},
    )
    guard = FrontierGuard(tmp_path / "frontier-guard")
    calls: list[str] = []

    def fake_run_codex(prompt: str, *, permit, **_kwargs):
        calls.append(prompt)
        permit.start(pid=12345)
        return frontier_review.FrontierResult(
            decision="approved",
            summary="fixed",
            tests_run=["pytest tests/test_ingest.py -k invalid_json"],
            committed=True,
            pushed=False,
            commit="c" * 40,
        )

    monkeypatch.setattr(frontier_review, "_run_codex", fake_run_codex)
    monkeypatch.setattr(
        frontier_review,
        "_capture_repair_baseline",
        lambda _repo: {
            "ok": True,
            "head": "b" * 40,
            "origin_main": "b" * 40,
            "clean": True,
        },
    )
    monkeypatch.setattr(
        frontier_review,
        "_verify_repair_result",
        lambda result, **_kwargs: frontier_review.replace(
            result,
            verified=True,
            verification={"ok": True},
        ),
    )
    monkeypatch.setattr(
        frontier_review,
        "_isolated_repair_checkout",
        lambda _repo, _baseline: nullcontext(tmp_path),
    )
    monkeypatch.delenv("CHRONOVISOR_FRONTIER_CMD", raising=False)

    first = frontier_review.run_frontier_review(
        {"failure_class": "model_json_invalid"},
        None,
        repo_root=tmp_path,
        evidence=evidence,
        guard=guard,
    )
    second = frontier_review.run_frontier_review(
        {"failure_class": "model_json_invalid"},
        None,
        repo_root=tmp_path,
        evidence=evidence,
        guard=guard,
    )

    assert first.decision == "approved"
    assert first.verified is True
    assert first.execution_started is True
    assert second.frontier_failure["failure_class"] == "frontier_guard_denied"
    assert second.execution_started is False
    assert len(calls) == 1


def test_frontier_repair_success_requires_independent_postconditions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    head = "c" * 40
    baseline = {"ok": True, "head": "b" * 40, "origin_main": "b" * 40}
    evidence = SimpleNamespace(reproduction_command=("uv", "run", "health-check"))
    result = frontier_review.FrontierResult(
        decision="approved",
        summary="fixed",
        tests_run=["uv run pytest -q"],
        committed=True,
        pushed=False,
        commit=head,
    )

    candidate_root = tmp_path / "candidate"

    def fake_git(repo, args, **_kwargs):
        if args[:2] == ["rev-parse", "HEAD"]:
            return SimpleNamespace(returncode=0, stdout=head + "\n", stderr="")
        if args == ["remote"] and repo == candidate_root:
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    calls: list[list[str]] = []

    def fake_verify(command, **_kwargs):
        calls.append(command)
        if "runtime-identity" in command:
            return {
                "ok": True,
                "returncode": 0,
                "stdout": json.dumps(
                    {"commit_id": head, "archive_path": "/tmp/archive"}
                ),
            }
        return {"ok": True, "returncode": 0, "stdout": ""}

    monkeypatch.setattr(frontier_review, "_git_probe", fake_git)
    monkeypatch.setattr(frontier_review, "_verification_command", fake_verify)
    monkeypatch.setattr(
        frontier_review,
        "_remote_main_sha",
        lambda _repo: baseline["head"],
    )
    monkeypatch.setattr(
        frontier_review,
        "_publish_verified_candidate",
        lambda **_kwargs: {"ok": True},
    )
    monkeypatch.setattr(
        frontier_review,
        "_restart_persistent_runtime_services",
        lambda _archive: {"ok": True, "services": []},
    )

    verified = frontier_review._verify_repair_result(
        result,
        repo_root=tmp_path,
        candidate_root=candidate_root,
        evidence=evidence,
        baseline=baseline,
    )

    assert verified.decision == "approved"
    assert verified.verified is True
    assert verified.verification["ok"] is True
    assert calls[0] == ["uv", "run", "health-check"]
    assert calls[1] == ["uv", "run", "pytest", "-q"]
    assert "runtime-identity" in calls[2]


def test_frontier_self_report_is_quarantined_when_commit_did_not_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline_head = "b" * 40
    result = frontier_review.FrontierResult(
        decision="approved",
        summary="claimed fixed",
        tests_run=["uv run pytest -q"],
        committed=True,
        pushed=False,
        commit=baseline_head,
    )

    candidate_root = tmp_path / "candidate"

    def fake_git(repo, args, **_kwargs):
        if args[:2] == ["rev-parse", "HEAD"]:
            return SimpleNamespace(returncode=0, stdout=baseline_head + "\n", stderr="")
        if args == ["remote"] and repo == candidate_root:
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(frontier_review, "_git_probe", fake_git)
    monkeypatch.setattr(
        frontier_review,
        "_remote_main_sha",
        lambda _repo: baseline_head,
    )
    monkeypatch.setattr(
        frontier_review,
        "_verification_command",
        lambda command, **_kwargs: {
            "ok": True,
            "returncode": 0,
            "stdout": (
                json.dumps({"commit_id": baseline_head, "archive_path": "/tmp/archive"})
                if "runtime-identity" in command
                else ""
            ),
        },
    )

    verified = frontier_review._verify_repair_result(
        result,
        repo_root=tmp_path,
        candidate_root=candidate_root,
        evidence=SimpleNamespace(reproduction_command=("uv", "run", "health-check")),
        baseline={"ok": True, "head": baseline_head, "origin_main": baseline_head},
    )

    assert verified.decision == "quarantined"
    assert verified.verified is False
    assert "no_new_commit" in verified.verification["failures"]


def test_isolated_repair_checkout_has_no_remote(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    for args in (
        ["git", "init", "-b", "main"],
        ["git", "config", "user.name", "Test"],
        ["git", "config", "user.email", "test@example.com"],
    ):
        completed = frontier_review.subprocess.run(
            args,
            cwd=repo,
            text=True,
            capture_output=True,
            check=False,
        )
        assert completed.returncode == 0
    (repo / "README.md").write_text("baseline\n", encoding="utf-8")
    frontier_review.subprocess.run(["git", "add", "README.md"], cwd=repo, check=True)
    frontier_review.subprocess.run(
        ["git", "commit", "-m", "baseline"], cwd=repo, check=True
    )
    head = frontier_review.subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()

    with frontier_review._isolated_repair_checkout(repo, {"head": head}) as candidate:
        remotes = frontier_review.subprocess.run(
            ["git", "remote"],
            cwd=candidate,
            text=True,
            capture_output=True,
            check=True,
        )
        candidate_head = frontier_review.subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=candidate,
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()

        assert remotes.stdout.strip() == ""
        assert candidate_head == head


def test_failed_parent_suite_never_publishes_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline_head = "b" * 40
    candidate_head = "c" * 40
    candidate_root = tmp_path / "candidate"
    result = frontier_review.FrontierResult(
        decision="approved",
        summary="fixed",
        tests_run=["uv run pytest -q"],
        committed=True,
        pushed=False,
        commit=candidate_head,
    )

    def fake_git(repo, args, **_kwargs):
        if args[:2] == ["rev-parse", "HEAD"]:
            return SimpleNamespace(
                returncode=0, stdout=candidate_head + "\n", stderr=""
            )
        if args == ["remote"] and repo == candidate_root:
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    verification_calls = 0

    def fake_verify(_command, **_kwargs):
        nonlocal verification_calls
        verification_calls += 1
        return {
            "ok": verification_calls == 1,
            "returncode": 0 if verification_calls == 1 else 1,
            "stdout": "",
        }

    monkeypatch.setattr(frontier_review, "_git_probe", fake_git)
    monkeypatch.setattr(
        frontier_review, "_remote_main_sha", lambda _repo: baseline_head
    )
    monkeypatch.setattr(frontier_review, "_verification_command", fake_verify)
    monkeypatch.setattr(
        frontier_review,
        "_publish_verified_candidate",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("a failing candidate must never be published")
        ),
    )

    verified = frontier_review._verify_repair_result(
        result,
        repo_root=tmp_path,
        candidate_root=candidate_root,
        evidence=SimpleNamespace(reproduction_command=("uv", "run", "health-check")),
        baseline={"ok": True, "head": baseline_head, "origin_main": baseline_head},
    )

    assert verified.decision == "quarantined"
    assert verified.pushed is False
    assert "full_test_suite_failed" in verified.verification["failures"]


def test_publish_stops_before_push_on_untracked_path_collision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline_head = "b" * 40
    candidate_head = "c" * 40

    def fake_git(_repo, args, **_kwargs):
        if args[:3] == ["status", "--porcelain=v1", "--untracked-files=no"]:
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if args[:2] == ["diff", "--name-only"]:
            return SimpleNamespace(returncode=0, stdout="new/module.py\n", stderr="")
        if args[:2] == ["ls-files", "--others"]:
            return SimpleNamespace(returncode=0, stdout="new/module.py\n", stderr="")
        if args and args[0] == "push":
            raise AssertionError("collision must be detected before push")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(frontier_review, "_git_probe", fake_git)
    monkeypatch.setattr(
        frontier_review, "_remote_main_sha", lambda _repo: baseline_head
    )

    published = frontier_review._publish_verified_candidate(
        repo_root=tmp_path,
        candidate_root=tmp_path / "candidate",
        candidate_head=candidate_head,
        baseline_head=baseline_head,
    )

    assert published["ok"] is False
    assert published["failure"] == "untracked_path_collision"


def test_frontier_prompt_forbids_model_push() -> None:
    prompt = frontier_review.build_frontier_prompt(
        {"failure_class": "system_health_snapshot_exception"},
        None,
        execute_patch=True,
    )

    assert "Never push" in prompt
    assert "pushed=false" in prompt


def test_runtime_restart_verifies_new_pid_and_archive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pids = {
        "com.trafficsign.chronovisor-dashboard": [101, 202],
        "com.trafficsign.chronovisor-ingest-drain": [None],
    }

    def fake_pid(label: str) -> int | None:
        values = pids[label]
        return values.pop(0) if len(values) > 1 else values[0]

    monkeypatch.setattr(frontier_review, "_launchd_pid", fake_pid)
    monkeypatch.setattr(
        frontier_review,
        "_pid_tree_uses_archive",
        lambda pid, path: pid == 202 and path == "/archive",
    )
    monkeypatch.setattr(
        frontier_review.shutil,
        "which",
        lambda name: "/bin/launchctl" if name == "launchctl" else None,
    )
    monkeypatch.setattr(
        frontier_review.subprocess,
        "run",
        lambda command, **_kwargs: SimpleNamespace(returncode=0, stdout="", stderr=""),
    )

    result = frontier_review._restart_persistent_runtime_services("/archive")

    assert result["ok"] is True
    assert result["services"][0]["old_pid"] == 101
    assert result["services"][0]["new_pid"] == 202
    assert result["services"][0]["archive_loaded"] is True
    assert result["services"][1]["status"] == "not_running"


def test_pid_tree_matches_runtime_identity_lib_path_to_exact_uv_archive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_base = tmp_path / "archive-v0"
    expected = archive_base / "expected-id"
    sibling = archive_base / "expected-id-other"
    identity_path = expected / "lib" / "python3.13"
    ps_output = "\n".join(
        (
            "100 1 /usr/bin/launch-wrapper dashboard",
            (
                f"101 100 {expected}/bin/python "
                f"{expected}/bin/chronovisor-dashboard --port 8765"
            ),
            f"102 100 {expected}/bin/python -c pass",
            (
                f"201 200 {sibling}/bin/python "
                f"{sibling}/bin/chronovisor-ingest-drain --watch"
            ),
        )
    )
    monkeypatch.setattr(
        frontier_review.subprocess,
        "run",
        lambda command, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout=ps_output,
            stderr="",
        ),
    )

    assert frontier_review._pid_tree_uses_archive(100, str(identity_path)) is True
    assert frontier_review._pid_tree_uses_archive(102, str(identity_path)) is False
    assert frontier_review._pid_tree_uses_archive(200, str(identity_path)) is False


def test_pid_tree_rejects_non_uv_archive_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        frontier_review.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("ps must not run for invalid identity"),
    )

    assert frontier_review._pid_tree_uses_archive(100, "/tmp/not-an-archive") is False


def test_routine_structured_review_uses_local_transport_and_never_subprocesses(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_chat(_messages, *, model: str, **_kwargs) -> str:
        calls.append(model)
        return json.dumps(
            {
                "decision": "approved",
                "summary": f"local vote from {model}",
                "tests_run": [],
                "commit": None,
                "committed": False,
                "pushed": False,
                "risk": None,
                "notes": None,
            }
        )

    def forbidden_subprocess(*_args, **_kwargs):
        raise AssertionError(
            "routine structured review must never start subprocess/Codex"
        )

    monkeypatch.setattr(
        decision_router,
        "load_decision_router_config",
        _local_router_config,
    )
    monkeypatch.setattr(ollama, "chat", fake_chat)
    monkeypatch.setattr(
        decision_router,
        "resolve_router_policy",
        lambda config, **_kwargs: decision_router.RouterPolicyResolution(
            config=config,
            source="adopted_artifact",
            artifact_sha256="d" * 64,
        ),
    )
    monkeypatch.setenv("CHRONOVISOR_DECISION_POLICY_RECALL_AUTO_APPLY", "enabled")
    # Block the actual Frontier entrypoint without sabotaging the local
    # residency broker's harmless host-memory probes (which also use the
    # process/shutil modules).
    monkeypatch.setattr(frontier_review, "_run_codex", forbidden_subprocess)
    monkeypatch.setenv("CHRONOVISOR_TEST_STRUCTURED_REVIEW_CMD", "/bin/forbidden")
    schema = frontier_review.FRONTIER_DECISION_SCHEMA

    result = frontier_review.run_structured_review(
        "review this",
        schema,
        repo_root=tmp_path,
        audit_root=tmp_path / "local-consensus-audit",
        timeout=1,
        execute_patch=True,
        command_env="CHRONOVISOR_TEST_STRUCTURED_REVIEW_CMD",
        decision_lane="recall_auto_apply",
    )

    assert result["decision"] == "approved"
    assert result["reviewer"] == "local_consensus"
    assert "frontier_failure" not in result
    assert result["local_consensus"]["status"] == "agreed"
    assert calls == ["ornith:test", "gpt-oss:test"]
    audit_rows = [
        json.loads(line)
        for line in (tmp_path / "local-consensus-audit" / "audit.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [row["role"] for row in audit_rows] == [
        "recall_auto_apply:primary",
        "recall_auto_apply:challenger",
        "recall_auto_apply",
    ]


def test_structured_review_defers_mutating_majority_with_conservative_vote(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    router = _quarantined_router_class(
        valid=(True, True, True),
        signatures=("a" * 64, "a" * 64, "b" * 64),
        reason="mutating_local_majority_vetoed_by_conservative_vote",
    )
    monkeypatch.setattr(decision_router, "DecisionRouter", router)
    monkeypatch.setenv("CHRONOVISOR_DECISION_POLICY_RECALL_AUTO_APPLY", "enabled")

    result = frontier_review.run_structured_review(
        "review this",
        frontier_review.FRONTIER_DECISION_SCHEMA,
        repo_root=tmp_path,
        audit_root=tmp_path / "local-consensus-audit",
        decision_lane="recall_auto_apply",
    )

    assert result["decision"] == "needs_retry"
    assert result["reviewer"] == "local_consensus"
    assert result["frontier_failure"]["failure_class"] == ("local_semantic_no_quorum")
    assert result["frontier_failure"]["rescue_status"] == "local_quarantined"
    assert (
        result["local_consensus"]["quarantine_reason"]
        == "mutating_local_majority_vetoed_by_conservative_vote"
    )
    assert result["human_required"] is False
    assert result["decision_policy"]["router_policy"]["artifact_sha256"] == ("d" * 64)


@pytest.mark.parametrize(
    ("valid", "signatures", "failure_class"),
    [
        (
            (True, True, True),
            ("a" * 64, "a" * 64, "malformed"),
            "local_consensus_failed",
        ),
        (
            (True, True, True),
            ("a" * 64, "a" * 64, "b" * 64),
            "local_resource_quarantined",
        ),
        (
            (True, True, False),
            ("a" * 64, "a" * 64, "b" * 64),
            "local_consensus_failed",
        ),
    ],
    ids=("malformed-signature", "wrong-failure-class", "invalid-vote"),
)
def test_structured_review_keeps_invalid_veto_evidence_operational(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    valid: tuple[bool, bool, bool],
    signatures: tuple[str | None, str | None, str | None],
    failure_class: str,
) -> None:
    router = _quarantined_router_class(
        valid=valid,
        signatures=signatures,
        reason="mutating_local_majority_vetoed_by_conservative_vote",
        failure_class=failure_class,
    )
    monkeypatch.setattr(decision_router, "DecisionRouter", router)
    monkeypatch.setenv("CHRONOVISOR_DECISION_POLICY_RECALL_AUTO_APPLY", "enabled")

    result = frontier_review.run_structured_review(
        "review this",
        frontier_review.FRONTIER_DECISION_SCHEMA,
        repo_root=tmp_path,
        decision_lane="recall_auto_apply",
    )

    assert result["decision"] == "needs_retry"
    assert result["frontier_failure"]["failure_class"] == "local_consensus_failed"
    assert result["frontier_failure"]["rescue_status"] == "local_quarantined"
    assert result["human_required"] is False


def test_structured_review_types_three_valid_distinct_semantic_no_quorum(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    router = _quarantined_router_class(
        valid=(True, True, True),
        signatures=("a" * 64, "b" * 64, "c" * 64),
        reason="local_models_did_not_reach_two_vote_quorum",
    )
    monkeypatch.setattr(decision_router, "DecisionRouter", router)
    monkeypatch.setenv("CHRONOVISOR_DECISION_POLICY_RECALL_AUTO_APPLY", "enabled")

    result = frontier_review.run_structured_review(
        "review this",
        frontier_review.FRONTIER_DECISION_SCHEMA,
        repo_root=tmp_path,
        decision_lane="recall_auto_apply",
    )

    assert result["decision"] == "needs_retry"
    assert result["frontier_failure"]["failure_class"] == ("local_semantic_no_quorum")
    assert result["frontier_failure"]["rescue_status"] == "local_quarantined"
    assert result["local_consensus"]["quarantine_reason"] == (
        "local_models_did_not_reach_two_vote_quorum"
    )
    assert result["decision_policy"]["router_policy"]["artifact_sha256"] == ("d" * 64)
    assert result["human_required"] is False


def test_structured_review_reuses_exact_epoch_semantic_hold_without_model_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority_box = {"value": semantic_authority()}
    calls: list[str] = []
    monkeypatch.setattr(
        decision_router,
        "DecisionRouter",
        _cache_test_router_class(authority_box, calls),
    )
    monkeypatch.setattr(
        frontier_review,
        "_current_structured_authority",
        lambda lane: (
            authority_box["value"],
            None,
        ),
    )
    monkeypatch.setenv("CHRONOVISOR_DECISION_POLICY_RECALL_AUTO_APPLY", "enabled")
    prompt = "SENSITIVE_PROMPT_MUST_NOT_BE_PERSISTED"
    system = "SENSITIVE_SYSTEM_MUST_NOT_BE_PERSISTED"

    first = frontier_review.run_structured_review(
        prompt,
        frontier_review.FRONTIER_DECISION_SCHEMA,
        repo_root=tmp_path,
        decision_lane="recall_auto_apply",
        system=system,
    )
    second = frontier_review.run_structured_review(
        prompt,
        frontier_review.FRONTIER_DECISION_SCHEMA,
        repo_root=tmp_path,
        decision_lane="recall_auto_apply",
        system=system,
    )

    assert second == first
    assert first["frontier_failure"]["failure_class"] == ("local_semantic_no_quorum")
    assert calls == [f"{prompt}|{system}"]
    cache_files = list(
        (tmp_path / "structured-review-holds" / "entries").glob("*.json")
    )
    assert len(cache_files) == 1
    raw = cache_files[0].read_text(encoding="utf-8")
    assert prompt not in raw
    assert system not in raw


def test_structured_review_reuses_cached_hold_after_observation_generation_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority_box = {"value": semantic_authority()}
    observation_box = {"value": "0" * 64}
    calls: list[str] = []
    monkeypatch.setattr(
        decision_router,
        "DecisionRouter",
        _cache_test_router_class(authority_box, calls),
    )
    monkeypatch.setattr(
        frontier_review,
        "_current_structured_authority",
        lambda _lane: (authority_box["value"], None),
    )
    monkeypatch.setattr(
        frontier_review,
        "_structured_authority_observation",
        lambda _authority: observation_box["value"],
    )
    monkeypatch.setenv("CHRONOVISOR_DECISION_POLICY_RECALL_AUTO_APPLY", "enabled")

    first = frontier_review.run_structured_review(
        "same request",
        frontier_review.FRONTIER_DECISION_SCHEMA,
        repo_root=tmp_path,
        decision_lane="recall_auto_apply",
    )
    observation_box["value"] = "1" * 64
    restored = frontier_review.run_structured_review(
        "same request",
        frontier_review.FRONTIER_DECISION_SCHEMA,
        repo_root=tmp_path,
        decision_lane="recall_auto_apply",
    )

    assert restored == first
    assert calls == ["same request|None"]
    entries = tmp_path / "structured-review-holds" / "entries"
    assert len(list(entries.glob("*.json"))) == 1


def test_structured_review_restores_cached_a_after_authority_a_b_a(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority_a = semantic_authority()
    authority_b = semantic_authority(artifact_sha256="9" * 64)
    authority_box = {"value": authority_a}
    observation_box = {"value": "0" * 64}
    calls: list[str] = []
    monkeypatch.setattr(
        decision_router,
        "DecisionRouter",
        _cache_test_router_class(authority_box, calls),
    )
    monkeypatch.setattr(
        frontier_review,
        "_current_structured_authority",
        lambda _lane: (authority_box["value"], None),
    )
    monkeypatch.setattr(
        frontier_review,
        "_structured_authority_observation",
        lambda _authority: observation_box["value"],
    )
    monkeypatch.setenv("CHRONOVISOR_DECISION_POLICY_RECALL_AUTO_APPLY", "enabled")

    result_a = frontier_review.run_structured_review(
        "same request",
        frontier_review.FRONTIER_DECISION_SCHEMA,
        repo_root=tmp_path,
        decision_lane="recall_auto_apply",
    )
    authority_box["value"] = authority_b
    observation_box["value"] = "1" * 64
    result_b = frontier_review.run_structured_review(
        "same request",
        frontier_review.FRONTIER_DECISION_SCHEMA,
        repo_root=tmp_path,
        decision_lane="recall_auto_apply",
    )
    authority_box["value"] = authority_a
    observation_box["value"] = "2" * 64
    restored_a = frontier_review.run_structured_review(
        "same request",
        frontier_review.FRONTIER_DECISION_SCHEMA,
        repo_root=tmp_path,
        decision_lane="recall_auto_apply",
    )

    assert restored_a == result_a
    assert result_b != result_a
    assert calls == ["same request|None", "same request|None"]
    entries = tmp_path / "structured-review-holds" / "entries"
    assert len(list(entries.glob("*.json"))) == 2


def test_structured_review_discards_cached_result_if_observation_drifts_during_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority_box = {"value": semantic_authority()}
    calls: list[str] = []
    monkeypatch.setattr(
        decision_router,
        "DecisionRouter",
        _cache_test_router_class(authority_box, calls),
    )
    monkeypatch.setattr(
        frontier_review,
        "_current_structured_authority",
        lambda _lane: (authority_box["value"], None),
    )
    monkeypatch.setenv("CHRONOVISOR_DECISION_POLICY_RECALL_AUTO_APPLY", "enabled")

    frontier_review.run_structured_review(
        "same request",
        frontier_review.FRONTIER_DECISION_SCHEMA,
        repo_root=tmp_path,
        decision_lane="recall_auto_apply",
    )
    observations = iter(["1" * 64, "1" * 64, "2" * 64])
    monkeypatch.setattr(
        frontier_review,
        "_structured_authority_observation",
        lambda _authority: next(observations),
    )

    result = frontier_review.run_structured_review(
        "same request",
        frontier_review.FRONTIER_DECISION_SCHEMA,
        repo_root=tmp_path,
        decision_lane="recall_auto_apply",
    )

    assert result["frontier_failure"]["failure_class"] == ("decision_authority_changed")
    assert result["frontier_failure"]["rescue_status"] == "local_retry"
    assert calls == ["same request|None"]


def test_structured_review_discards_current_result_if_authority_drifts_during_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority_box = {"value": semantic_authority()}
    observation_box = {"value": "0" * 64}
    calls: list[str] = []
    monkeypatch.setattr(
        decision_router,
        "DecisionRouter",
        _cache_test_router_class(authority_box, calls),
    )
    monkeypatch.setattr(
        frontier_review,
        "_current_structured_authority",
        lambda _lane: (authority_box["value"], None),
    )
    monkeypatch.setattr(
        frontier_review,
        "_structured_authority_observation",
        lambda _authority: observation_box["value"],
    )
    original_store = semantic_hold.StructuredReviewHoldLease.store

    def drifting_store(self, result):
        hold = original_store(self, result)
        observation_box["value"] = "1" * 64
        return hold

    monkeypatch.setattr(
        semantic_hold.StructuredReviewHoldLease,
        "store",
        drifting_store,
    )
    monkeypatch.setenv("CHRONOVISOR_DECISION_POLICY_RECALL_AUTO_APPLY", "enabled")

    drifted = frontier_review.run_structured_review(
        "same request",
        frontier_review.FRONTIER_DECISION_SCHEMA,
        repo_root=tmp_path,
        decision_lane="recall_auto_apply",
    )

    assert drifted["frontier_failure"]["failure_class"] == (
        "decision_authority_changed"
    )
    assert drifted["frontier_failure"]["rescue_status"] == "local_retry"
    entries = tmp_path / "structured-review-holds" / "entries"
    assert len(list(entries.glob("*.json"))) == 1
    assert calls == ["same request|None"]

    restored = frontier_review.run_structured_review(
        "same request",
        frontier_review.FRONTIER_DECISION_SCHEMA,
        repo_root=tmp_path,
        decision_lane="recall_auto_apply",
    )

    assert restored["frontier_failure"]["failure_class"] == ("local_semantic_no_quorum")
    assert calls == ["same request|None"]


def test_structured_review_does_not_use_cache_when_initial_observation_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority_box = {"value": semantic_authority()}
    calls: list[str] = []
    monkeypatch.setattr(
        decision_router,
        "DecisionRouter",
        _cache_test_router_class(authority_box, calls),
    )
    monkeypatch.setattr(
        frontier_review,
        "_current_structured_authority",
        lambda _lane: (authority_box["value"], None),
    )
    monkeypatch.setenv("CHRONOVISOR_DECISION_POLICY_RECALL_AUTO_APPLY", "enabled")

    frontier_review.run_structured_review(
        "same request",
        frontier_review.FRONTIER_DECISION_SCHEMA,
        repo_root=tmp_path,
        decision_lane="recall_auto_apply",
    )
    monkeypatch.setattr(
        frontier_review,
        "_structured_authority_observation",
        lambda _authority: None,
    )
    result = frontier_review.run_structured_review(
        "same request",
        frontier_review.FRONTIER_DECISION_SCHEMA,
        repo_root=tmp_path,
        decision_lane="recall_auto_apply",
    )

    assert result["frontier_failure"]["failure_class"] == (
        "decision_authority_unavailable"
    )
    assert calls == ["same request|None"]


def test_structured_review_does_not_return_cache_when_post_observation_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority_box = {"value": semantic_authority()}
    calls: list[str] = []
    monkeypatch.setattr(
        decision_router,
        "DecisionRouter",
        _cache_test_router_class(authority_box, calls),
    )
    monkeypatch.setattr(
        frontier_review,
        "_current_structured_authority",
        lambda _lane: (authority_box["value"], None),
    )
    monkeypatch.setenv("CHRONOVISOR_DECISION_POLICY_RECALL_AUTO_APPLY", "enabled")

    frontier_review.run_structured_review(
        "same request",
        frontier_review.FRONTIER_DECISION_SCHEMA,
        repo_root=tmp_path,
        decision_lane="recall_auto_apply",
    )
    observations = iter(["1" * 64, "1" * 64, None])
    monkeypatch.setattr(
        frontier_review,
        "_structured_authority_observation",
        lambda _authority: next(observations),
    )
    result = frontier_review.run_structured_review(
        "same request",
        frontier_review.FRONTIER_DECISION_SCHEMA,
        repo_root=tmp_path,
        decision_lane="recall_auto_apply",
    )

    assert result["frontier_failure"]["failure_class"] == (
        "decision_authority_unavailable"
    )
    assert calls == ["same request|None"]


def test_structured_review_cache_lock_entry_failure_falls_back_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority_box = {"value": semantic_authority()}
    calls: list[str] = []
    monkeypatch.setattr(
        decision_router,
        "DecisionRouter",
        _cache_test_router_class(authority_box, calls),
    )
    monkeypatch.setattr(
        frontier_review,
        "_current_structured_authority",
        lambda lane: (authority_box["value"], None),
    )

    @contextmanager
    def broken_lock(_self, **_kwargs):
        raise OSError("lock unavailable")
        yield

    monkeypatch.setattr(
        semantic_hold.StructuredReviewSemanticHoldCache,
        "locked",
        broken_lock,
    )
    monkeypatch.setenv("CHRONOVISOR_DECISION_POLICY_RECALL_AUTO_APPLY", "enabled")

    result = frontier_review.run_structured_review(
        "same request",
        frontier_review.FRONTIER_DECISION_SCHEMA,
        repo_root=tmp_path,
        decision_lane="recall_auto_apply",
    )

    assert result["frontier_failure"]["failure_class"] == ("local_semantic_no_quorum")
    assert calls == ["same request|None"]


def test_structured_review_cache_misses_prompt_system_and_authority_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority_box = {"value": semantic_authority()}
    calls: list[str] = []
    monkeypatch.setattr(
        decision_router,
        "DecisionRouter",
        _cache_test_router_class(authority_box, calls),
    )
    monkeypatch.setattr(
        frontier_review,
        "_current_structured_authority",
        lambda lane: (authority_box["value"], None),
    )
    monkeypatch.setenv("CHRONOVISOR_DECISION_POLICY_RECALL_AUTO_APPLY", "enabled")

    def run(prompt: str, system: str | None) -> dict[str, object]:
        return frontier_review.run_structured_review(
            prompt,
            frontier_review.FRONTIER_DECISION_SCHEMA,
            repo_root=tmp_path,
            decision_lane="recall_auto_apply",
            system=system,
        )

    run("prompt-a", "system-a")
    run("prompt-a", "system-a")
    run("prompt-b", "system-a")
    run("prompt-b", "system-b")
    authority_box["value"] = semantic_authority(artifact_sha256="9" * 64)
    run("prompt-b", "system-b")
    run("prompt-b", "system-b")

    assert calls == [
        "prompt-a|system-a",
        "prompt-b|system-a",
        "prompt-b|system-b",
        "prompt-b|system-b",
    ]


def test_structured_review_does_not_store_after_authority_aba_observation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority_box = {"value": semantic_authority()}
    calls: list[str] = []
    observations = iter(["0" * 64, "0" * 64, "1" * 64, *("1" * 64 for _ in range(7))])
    monkeypatch.setattr(
        decision_router,
        "DecisionRouter",
        _cache_test_router_class(authority_box, calls),
    )
    monkeypatch.setattr(
        frontier_review,
        "_current_structured_authority",
        lambda lane: (authority_box["value"], None),
    )
    monkeypatch.setattr(
        frontier_review,
        "_structured_authority_observation",
        lambda _authority: next(observations),
    )
    monkeypatch.setenv("CHRONOVISOR_DECISION_POLICY_RECALL_AUTO_APPLY", "enabled")

    results = []
    for _index in range(3):
        results.append(
            frontier_review.run_structured_review(
                "same request",
                frontier_review.FRONTIER_DECISION_SCHEMA,
                repo_root=tmp_path,
                decision_lane="recall_auto_apply",
            )
        )

    assert results[0]["frontier_failure"]["failure_class"] == (
        "decision_authority_changed"
    )
    assert results[0]["frontier_failure"]["rescue_status"] == "local_retry"
    assert results[1]["frontier_failure"]["failure_class"] == (
        "local_semantic_no_quorum"
    )
    assert results[2] == results[1]
    # The first no-quorum result was discarded because the mutable authority
    # generation changed during the model call. The second stored under the
    # new stable generation and the third was a zero-call cache hit.
    assert calls == ["same request|None", "same request|None"]


@pytest.mark.parametrize("outcome", ["semantic", "success", "operational"])
def test_structured_review_discards_any_model_result_after_authority_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    outcome: str,
) -> None:
    authority_box = {"value": semantic_authority()}
    calls: list[str] = []
    observations = iter(["0" * 64, "0" * 64, "1" * 64])
    monkeypatch.setattr(
        decision_router,
        "DecisionRouter",
        _cache_test_router_class(authority_box, calls, outcome=outcome),
    )
    monkeypatch.setattr(
        frontier_review,
        "_current_structured_authority",
        lambda lane: (authority_box["value"], None),
    )
    monkeypatch.setattr(
        frontier_review,
        "_structured_authority_observation",
        lambda _authority: next(observations),
    )
    monkeypatch.setenv("CHRONOVISOR_DECISION_POLICY_RECALL_AUTO_APPLY", "enabled")

    result = frontier_review.run_structured_review(
        "same request",
        frontier_review.FRONTIER_DECISION_SCHEMA,
        repo_root=tmp_path,
        decision_lane="recall_auto_apply",
    )

    assert result["decision"] == "needs_retry"
    assert result["frontier_failure"]["failure_class"] == ("decision_authority_changed")
    assert result["frontier_failure"]["rescue_status"] == "local_retry"
    assert calls == ["same request|None"]
    entries = tmp_path / "structured-review-holds" / "entries"
    assert not entries.exists() or not list(entries.glob("*.json"))


def test_structured_review_authority_guard_survives_cache_lock_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority_box = {"value": semantic_authority()}
    calls: list[str] = []
    observations = iter(["0" * 64, "0" * 64, "1" * 64])
    monkeypatch.setattr(
        decision_router,
        "DecisionRouter",
        _cache_test_router_class(authority_box, calls),
    )
    monkeypatch.setattr(
        frontier_review,
        "_current_structured_authority",
        lambda lane: (authority_box["value"], None),
    )
    monkeypatch.setattr(
        frontier_review,
        "_structured_authority_observation",
        lambda _authority: next(observations),
    )
    monkeypatch.setattr(
        frontier_review.semantic_hold.StructuredReviewSemanticHoldCache,
        "locked",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("lock failed")),
    )
    monkeypatch.setenv("CHRONOVISOR_DECISION_POLICY_RECALL_AUTO_APPLY", "enabled")

    result = frontier_review.run_structured_review(
        "same request",
        frontier_review.FRONTIER_DECISION_SCHEMA,
        repo_root=tmp_path,
        decision_lane="recall_auto_apply",
    )

    assert result["frontier_failure"]["failure_class"] == ("decision_authority_changed")
    assert calls == ["same request|None"]


@pytest.mark.parametrize("outcome", ["success", "operational"])
def test_structured_review_never_caches_non_semantic_outcomes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    outcome: str,
) -> None:
    authority_box = {"value": semantic_authority()}
    calls: list[str] = []
    monkeypatch.setattr(
        decision_router,
        "DecisionRouter",
        _cache_test_router_class(authority_box, calls, outcome=outcome),
    )
    monkeypatch.setattr(
        frontier_review,
        "_current_structured_authority",
        lambda lane: (authority_box["value"], None),
    )
    monkeypatch.setenv("CHRONOVISOR_DECISION_POLICY_RECALL_AUTO_APPLY", "enabled")

    for _index in range(2):
        frontier_review.run_structured_review(
            "same request",
            frontier_review.FRONTIER_DECISION_SCHEMA,
            repo_root=tmp_path,
            decision_lane="recall_auto_apply",
        )

    assert calls == ["same request|None", "same request|None"]
    entries = tmp_path / "structured-review-holds" / "entries"
    assert not entries.exists() or not list(entries.glob("*.json"))


@pytest.mark.parametrize(
    ("valid", "signatures"),
    [
        ((True, True, False), ("a" * 64, "b" * 64, None)),
        ((True, True, True), ("a" * 64, "a" * 64, "b" * 64)),
    ],
    ids=("invalid-tie-break", "duplicate-signature"),
)
def test_structured_review_keeps_non_three_way_no_quorum_operational(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    valid: tuple[bool, bool, bool],
    signatures: tuple[str | None, str | None, str | None],
) -> None:
    router = _quarantined_router_class(
        valid=valid,
        signatures=signatures,
        reason="local_models_did_not_reach_two_vote_quorum",
    )
    monkeypatch.setattr(decision_router, "DecisionRouter", router)
    monkeypatch.setenv("CHRONOVISOR_DECISION_POLICY_RECALL_AUTO_APPLY", "enabled")

    result = frontier_review.run_structured_review(
        "review this",
        frontier_review.FRONTIER_DECISION_SCHEMA,
        repo_root=tmp_path,
        decision_lane="recall_auto_apply",
    )

    assert result["decision"] == "needs_retry"
    assert result["frontier_failure"]["failure_class"] == "local_consensus_failed"
    assert result["frontier_failure"]["rescue_status"] == "local_quarantined"
    assert result["human_required"] is False


def test_structured_review_local_model_failures_quarantine_without_tie_or_frontier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def failed_chat(_messages, *, model: str, **_kwargs) -> str:
        calls.append(model)
        raise RuntimeError(f"{model} unavailable")

    monkeypatch.setattr(
        decision_router,
        "load_decision_router_config",
        _local_router_config,
    )
    monkeypatch.setattr(ollama, "chat", failed_chat)
    monkeypatch.setattr(
        frontier_review,
        "_run_codex",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("local failures must not fall back to frontier")
        ),
    )
    monkeypatch.setattr(
        decision_router,
        "resolve_router_policy",
        lambda config, **_kwargs: decision_router.RouterPolicyResolution(
            config=config,
            source="adopted_artifact",
        ),
    )
    monkeypatch.setenv("CHRONOVISOR_DECISION_POLICY_RECALL_AUTO_APPLY", "shadow")
    schema = frontier_review.FRONTIER_DECISION_SCHEMA

    result = frontier_review.run_structured_review(
        "review this",
        schema,
        repo_root=tmp_path,
        audit_root=tmp_path / "local-consensus-audit",
        decision_lane="recall_auto_apply",
    )

    assert result["decision"] == "needs_retry"
    assert result["frontier_failure"]["failure_class"] == "local_decision_shadow_only"
    assert result["frontier_failure"]["rescue_status"] == "local_quarantined"
    assert result["human_required"] is False
    assert calls == ["ornith:test", "gpt-oss:test"]
    assert (tmp_path / "local-consensus-audit" / "audit.jsonl").exists()


def test_structured_review_rejects_incomplete_approved_json(
    tmp_path: Path, monkeypatch
) -> None:
    class IncompleteLocalRouter:
        def __init__(self, **_kwargs) -> None:
            router_audit = semantic_authority(artifact_sha256="d" * 64)["router"]
            assert isinstance(router_audit, dict)
            self.policy = SimpleNamespace(
                source="adopted_artifact",
                audit_record=lambda: dict(router_audit),
            )

        def decide(self, _prompt, _schema):
            return DecisionRouterResult(
                status="agreed",
                value={"decision": "approved"},
                agreement_sha256="a" * 64,
            )

    monkeypatch.setattr(decision_router, "DecisionRouter", IncompleteLocalRouter)
    monkeypatch.setenv("CHRONOVISOR_DECISION_POLICY_RECALL_AUTO_APPLY", "enabled")
    monkeypatch.setattr(
        frontier_review.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("defensive schema rejection must stay local")
        ),
    )
    schema = frontier_review.FRONTIER_DECISION_SCHEMA

    result = frontier_review.run_structured_review(
        "review this",
        schema,
        repo_root=tmp_path,
        audit_root=tmp_path / "local-consensus-audit",
        decision_lane="recall_auto_apply",
    )

    assert result["decision"] == "needs_retry"
    assert result["frontier_failure"]["failure_class"] == "schema_invalid"
    assert result["human_required"] is False


def test_local_structured_result_preserves_optional_schema_properties() -> None:
    from chronovisor.ingest import INGEST_FRONTIER_DECISION_SCHEMA

    schema = INGEST_FRONTIER_DECISION_SCHEMA

    result = frontier_review._validated_structured_result(
        {
            "decision": "apply_available",
            "summary": "required fields are valid",
            "failed_operations_disposition": "none",
            "tests_run": [],
            "risk": None,
            "notes": None,
        },
        schema,
        reviewer="local_consensus",
    )

    assert result == {
        "decision": "apply_available",
        "summary": "required fields are valid",
        "failed_operations_disposition": "none",
        "tests_run": [],
        "risk": None,
        "notes": None,
        "reviewer": "local_consensus",
    }
    assert "frontier_failure" not in result
    assert "repair_option_id" not in result
    assert "invalid_tags" not in result
    assert "replacement_operations" not in result


def test_ingest_structured_failure_envelope_uses_retry_decision() -> None:
    from chronovisor.ingest import INGEST_FRONTIER_DECISION_SCHEMA

    result = frontier_review._validated_structured_result(
        None,
        INGEST_FRONTIER_DECISION_SCHEMA,
        reviewer="local_consensus",
    )

    assert result["decision"] == "retry"
    assert result["failed_operations_disposition"] == "none"
    assert result["frontier_failure"]["failure_class"] == "schema_invalid"


def test_structured_review_forwards_optional_system_to_local_router(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    class SystemCapturingLocalRouter:
        def __init__(self, **_kwargs) -> None:
            router_audit = semantic_authority(artifact_sha256="d" * 64)["router"]
            assert isinstance(router_audit, dict)
            self.policy = SimpleNamespace(
                source="adopted_artifact",
                audit_record=lambda: dict(router_audit),
            )

        def decide(self, prompt, schema, *, system=None):
            captured.update(prompt=prompt, schema=schema, system=system)
            return DecisionRouterResult(
                status="agreed",
                value={
                    "decision": "approved",
                    "summary": "local vote",
                    "tests_run": [],
                    "commit": None,
                    "committed": False,
                    "pushed": False,
                    "risk": None,
                    "notes": None,
                },
                agreement_sha256="a" * 64,
            )

    monkeypatch.setattr(
        decision_router,
        "DecisionRouter",
        SystemCapturingLocalRouter,
    )
    monkeypatch.setenv("CHRONOVISOR_DECISION_POLICY_RECALL_AUTO_APPLY", "enabled")
    marker = "CHRONOVISOR_READ_BACK_EVIDENCE_POLICY=1"

    result = frontier_review.run_structured_review(
        "review this",
        frontier_review.FRONTIER_DECISION_SCHEMA,
        repo_root=tmp_path,
        decision_lane="recall_auto_apply",
        system=marker,
    )

    assert result["decision"] == "approved"
    assert captured["system"] == marker
