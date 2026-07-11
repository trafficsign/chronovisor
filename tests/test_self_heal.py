from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture()
def isolated_wiki(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    wiki_root = tmp_path / "wiki"
    pages = wiki_root / "pages"
    raw = wiki_root / "raw"
    system = wiki_root / "system"
    runtime = wiki_root / "runtime"
    for d in (pages, raw, system, runtime):
        d.mkdir(parents=True, exist_ok=True)

    from llm_wiki_mcp import wiki, runtime_status

    monkeypatch.setattr(wiki, "WIKI_ROOT", wiki_root)
    monkeypatch.setattr(wiki, "PAGES_DIR", pages)
    monkeypatch.setattr(wiki, "RAW_DIR", raw)
    monkeypatch.setattr(wiki, "SYSTEM_DIR", system)
    monkeypatch.setattr(runtime_status, "RUNTIME_DIR", runtime)
    monkeypatch.setattr(runtime_status, "STATUS_FILE", runtime / "status.json")
    monkeypatch.setattr(runtime_status, "EVENTS_FILE", runtime / "events.jsonl")
    monkeypatch.setattr(runtime_status, "METRICS_FILE", runtime / "metrics.jsonl")
    return wiki_root


def _seed_page(wiki_root: Path, rel: str) -> None:
    path = wiki_root / "pages" / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("---\ntitle: T\nupdated: 2026-01-01\n---\nbody\n")


def _write_packet(wiki_root: Path) -> Path:
    packet = {
        "failure_id": "f1",
        "raw_file": "broken.md",
        "failure_class": "apply.update_target_not_found",
        "fingerprint": "apply.update_target_not_found:model-made-up-target",
        "attempts": 3,
        "error": "update target not found for page_id 'model-made-up-target'",
        "requested_page_id": "model-made-up-target",
        "similar_existing_pages": ["ai/canonical-target"],
        "status": "pending_local_repair",
    }
    path = wiki_root / "runtime" / "failures" / "packets" / "f1.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(packet))
    return path


def test_local_repair_adds_alias_restores_raw_and_retries(
    isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from llm_wiki_mcp import self_heal
    from llm_wiki_mcp.alias_store import load_aliases

    _seed_page(isolated_wiki, "ai/canonical-target.md")
    packet_path = _write_packet(isolated_wiki)
    quarantined = isolated_wiki / "runtime" / "failures" / "quarantined-raw" / "broken.md"
    quarantined.parent.mkdir(parents=True, exist_ok=True)
    quarantined.write_text("raw body")

    monkeypatch.setattr(
        self_heal,
        "_retry_ingest",
        lambda *, dry_run: {"triggered": True, "files_processed": ["broken.md"]},
    )
    monkeypatch.setattr(
        self_heal,
        "_run_frontier",
        lambda *_args, **_kwargs: {
            "decision": "approved",
            "summary": "exact alias repair is supported",
            "human_required": False,
        },
    )
    result = self_heal.handle_packet(
        packet_path,
        use_qwen=False,
        enable_frontier=True,
        dry_run=False,
    )

    assert result["status"] == "frontier_approved"
    assert load_aliases()["model-made-up-target"] == "ai/canonical-target"
    assert not quarantined.exists()
    assert (isolated_wiki / "raw" / "broken.md").exists()
    updated_packet = json.loads(packet_path.read_text())
    assert updated_packet["status"] == "frontier_approved"


def test_drill_returns_local_repair_decision(isolated_wiki: Path) -> None:
    from llm_wiki_mcp.self_heal import run_drill

    result = run_drill(use_qwen=False)

    assert result["decision"]["status"] == "resolved"
    assert result["decision"]["action"] == "resolve_update_target"


def test_auto_apply_error_packet_escalates_to_frontier_deterministically(isolated_wiki: Path) -> None:
    from llm_wiki_mcp.local_repair import propose_repair

    packet = {
        "failure_class": "recall.auto_apply_error",
        "fingerprint": "recall.auto_apply_error:page_tag:invalid_page_tag",
        "attempts": 425,
        "auto_apply_error": {"error_kind": "page_tag:invalid_page_tag"},
    }

    decision = propose_repair(packet, use_qwen=False)

    assert decision.status == "escalate"
    assert decision.action == "escalate_to_frontier"
    assert decision.confidence >= 0.85


def test_missing_update_without_candidate_retries_create_safe_raw(
    isolated_wiki: Path,
) -> None:
    from llm_wiki_mcp.local_repair import propose_repair

    packet = {
        "failure_class": "apply.update_target_not_found",
        "fingerprint": (
            "apply.update_target_not_found:"
            "claude-code-vs-claude-code-structural-analysis"
        ),
        "attempts": 3,
        "error": (
            "update target not found for page_id "
            "'claude-code-vs-claude-code-structural-analysis'"
        ),
        "requested_page_id": "claude-code-vs-claude-code-structural-analysis",
        "similar_existing_pages": [],
    }

    def stale_qwen(*_args, **_kwargs) -> str:
        raise AssertionError("deterministic safe repair should run before Qwen")

    decision = propose_repair(packet, generator=stale_qwen, use_qwen=True)

    assert decision.status == "resolved"
    assert decision.action == "retry_raw"
    assert (
        decision.requested_page_id
        == "claude-code-vs-claude-code-structural-analysis"
    )
    assert decision.confidence >= 0.85


def test_missing_update_with_unsafe_page_id_still_escalates(
    isolated_wiki: Path,
) -> None:
    from llm_wiki_mcp.local_repair import propose_repair

    packet = {
        "failure_class": "apply.update_target_not_found",
        "requested_page_id": "Claude Code Structural Analysis",
        "similar_existing_pages": [],
    }

    decision = propose_repair(packet, use_qwen=False)

    assert decision.status == "escalate"
    assert decision.action == "escalate_to_frontier"


def test_missing_update_retry_raw_restores_raw_and_retries(
    isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from llm_wiki_mcp import self_heal

    packet = {
        "failure_id": "f-no-candidate",
        "raw_file": "new-topic.md",
        "failure_class": "apply.update_target_not_found",
        "fingerprint": (
            "apply.update_target_not_found:"
            "claude-code-vs-claude-code-structural-analysis"
        ),
        "attempts": 3,
        "error": (
            "update target not found for page_id "
            "'claude-code-vs-claude-code-structural-analysis'"
        ),
        "requested_page_id": "claude-code-vs-claude-code-structural-analysis",
        "similar_existing_pages": [],
        "status": "pending_local_repair",
    }
    packet_path = (
        isolated_wiki / "runtime" / "failures" / "packets" / "f-no-candidate.json"
    )
    packet_path.parent.mkdir(parents=True, exist_ok=True)
    packet_path.write_text(json.dumps(packet), encoding="utf-8")
    quarantined = (
        isolated_wiki
        / "runtime"
        / "failures"
        / "quarantined-raw"
        / "new-topic.md"
    )
    quarantined.parent.mkdir(parents=True, exist_ok=True)
    quarantined.write_text("raw body", encoding="utf-8")

    monkeypatch.setattr(
        self_heal,
        "_retry_ingest",
        lambda *, dry_run: {"triggered": True, "files_processed": ["new-topic.md"]},
    )

    result = self_heal.handle_packet(
        packet_path,
        use_qwen=False,
        enable_frontier=False,
        dry_run=False,
    )

    assert result["status"] == "local_repair_applied"
    assert result["action"]["action"] == "retry_raw"
    assert not quarantined.exists()
    assert (isolated_wiki / "raw" / "new-topic.md").exists()
    updated_packet = json.loads(packet_path.read_text())
    assert updated_packet["status"] == "local_repair_applied"


def test_sandbox_drill_runs_pending_raw_to_self_heal(monkeypatch: pytest.MonkeyPatch) -> None:
    from llm_wiki_mcp.self_heal import run_sandbox_drill

    monkeypatch.setenv("LLM_WIKI_SELF_HEAL_AUTORUN", "0")

    result = run_sandbox_drill(use_qwen=False)

    assert result["status"] == "ok"
    assert result["packet_paths"]
    assert result["heal_result"]["status"] == "frontier_approved"
    assert result["pending_after"] == []
    assert (
        result["aliases"]["opus-4-7-evaluation-and-industry-geopolitics"]
        == "ai/opus-4.7-evaluation-and-industry-geopolitics"
    )


def test_frontier_human_required_sends_notification(
    isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from llm_wiki_mcp import self_heal
    from llm_wiki_mcp.local_repair import LocalRepairDecision

    packet = {
        "failure_id": "auth1",
        "raw_file": "auth-broken.md",
        "failure_class": "triage.parse_failed",
        "fingerprint": "triage.parse_failed",
        "attempts": 3,
        "error": "triage parse failed",
        "status": "pending_local_repair",
    }
    packet_path = isolated_wiki / "runtime" / "failures" / "packets" / "auth1.json"
    packet_path.parent.mkdir(parents=True, exist_ok=True)
    packet_path.write_text(json.dumps(packet), encoding="utf-8")
    sent: list[tuple[str, str]] = []

    monkeypatch.setattr(
        self_heal,
        "propose_repair",
        lambda *_args, **_kwargs: LocalRepairDecision(
            status="escalate",
            action="escalate_to_frontier",
            confidence=0.9,
            reason="needs frontier",
            source="deterministic",
        ),
    )
    monkeypatch.setattr(
        self_heal,
        "_run_frontier",
        lambda *_args, **_kwargs: {
            "decision": "needs_retry",
            "summary": "codex auth missing",
            "tests_run": [],
            "commit": None,
            "committed": False,
            "pushed": False,
            "risk": None,
            "notes": None,
            "raw_output": "401 Unauthorized",
            "frontier_failure": {
                "failure_class": "auth_required",
                "rescue_status": "human_required",
                "summary": "frontier API authentication is missing or invalid",
                "human_required": True,
                "notify_user": True,
            },
            "rescue_status": "human_required",
            "human_required": True,
            "notify_user": True,
            "rescue_attempt": None,
        },
    )
    monkeypatch.setattr(
        self_heal,
        "_send_mac_notification",
        lambda title, body: sent.append((title, body)) or {"sent": True},
    )

    result = self_heal.handle_packet(packet_path, use_qwen=False)

    assert result["status"] == "human_required"
    assert sent == [
        (
            self_heal.MAC_NOTIFICATION_TITLE,
            "Codex の認証が切れている可能性があります。ログイン確認が必要です。",
        )
    ]
    updated_packet = json.loads(packet_path.read_text())
    assert updated_packet["status"] == "human_required"
    assert updated_packet["human_notification"]["delivery"]["sent"] is True


def test_frontier_pending_review_writes_artifact(
    isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from llm_wiki_mcp import self_heal
    from llm_wiki_mcp.local_repair import LocalRepairDecision

    packet = {
        "failure_id": "schema1",
        "raw_file": "schema-broken.md",
        "failure_class": "triage.parse_failed",
        "fingerprint": "triage.parse_failed",
        "attempts": 3,
        "error": "triage parse failed",
        "status": "pending_local_repair",
    }
    packet_path = isolated_wiki / "runtime" / "failures" / "packets" / "schema1.json"
    packet_path.parent.mkdir(parents=True, exist_ok=True)
    packet_path.write_text(json.dumps(packet), encoding="utf-8")

    monkeypatch.setattr(
        self_heal,
        "propose_repair",
        lambda *_args, **_kwargs: LocalRepairDecision(
            status="escalate",
            action="escalate_to_frontier",
            confidence=0.9,
            reason="needs frontier",
            source="deterministic",
        ),
    )
    monkeypatch.setattr(
        self_heal,
        "_run_frontier",
        lambda *_args, **_kwargs: {
            "decision": "needs_retry",
            "summary": "codex schema needs review",
            "tests_run": [],
            "commit": None,
            "committed": False,
            "pushed": False,
            "risk": None,
            "notes": None,
            "frontier_failure": {
                "failure_class": "schema_invalid",
                "rescue_status": "pending_frontier_review",
                "summary": "schema repaired but needs review",
                "human_required": False,
                "notify_user": False,
            },
            "rescue_status": "pending_frontier_review",
            "human_required": False,
            "notify_user": False,
            "access_repair": {
                "applied": True,
                "repairs": [{"type": "schema_strictness_autofix"}],
            },
        },
    )

    result = self_heal.handle_packet(packet_path, use_qwen=False)

    assert result["status"] == "pending_frontier_review"
    pending_path = Path(result["pending_frontier_review_path"])
    assert pending_path.exists()
    pending = json.loads(pending_path.read_text())
    assert pending["status"] == "pending_frontier_review"
    assert pending["access_repair"]["applied"] is True
    updated_packet = json.loads(packet_path.read_text())
    assert updated_packet["pending_frontier_review_path"] == str(pending_path)


def test_human_required_notification_cooldown(
    isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from datetime import datetime, timedelta

    from llm_wiki_mcp import self_heal

    packet = {
        "failure_id": "auth1",
        "raw_file": "auth-broken.md",
        "fingerprint": "triage.parse_failed",
    }
    frontier_result = {
        "human_required": True,
        "notify_user": True,
        "frontier_failure": {"failure_class": "auth_required"},
        "rescue_status": "human_required",
    }
    sent: list[tuple[str, str]] = []
    monkeypatch.setattr(
        self_heal,
        "_send_mac_notification",
        lambda title, body: sent.append((title, body)) or {"sent": True},
    )
    now = datetime(2026, 6, 4, 22, 0, 0)

    first = self_heal.maybe_notify_human_required(packet, frontier_result, now=now)
    second = self_heal.maybe_notify_human_required(
        packet,
        frontier_result,
        now=now + timedelta(seconds=10),
    )

    assert first["delivery"]["sent"] is True
    assert second["reason"] == "cooldown"
    assert len(sent) == 1


def test_model_human_flag_cannot_widen_external_authority_boundary(
    isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from llm_wiki_mcp import self_heal

    packet = {"failure_id": "model1", "raw_file": "model-broken.md"}
    model_failure = {
        "decision": "needs_retry",
        "human_required": True,
        "notify_user": True,
        "frontier_failure": {
            "failure_class": "frontier_tool_unavailable",
            "human_required": True,
        },
    }
    sent: list[tuple[str, str]] = []
    monkeypatch.setattr(
        self_heal,
        "_send_mac_notification",
        lambda title, body: sent.append((title, body)) or {"sent": True},
    )

    assert self_heal._frontier_final_status(model_failure) == "frontier_retry"
    assert self_heal.maybe_notify_human_required(packet, model_failure) == {
        "sent": False,
        "reason": "not human required",
    }
    assert sent == []

    external_failure = {
        **model_failure,
        "human_required": False,
        "notify_user": False,
        "frontier_failure": {
            "failure_class": "oauth_required",
            "human_required": False,
        },
    }
    assert self_heal._frontier_final_status(external_failure) == "human_required"


def test_legacy_tool_unavailable_human_packet_reopens_autonomously(
    isolated_wiki: Path,
) -> None:
    from llm_wiki_mcp import self_heal

    packet_path = isolated_wiki / "runtime" / "failures" / "packets" / "legacy-tool.json"
    packet_path.parent.mkdir(parents=True, exist_ok=True)
    packet_path.write_text(
        json.dumps(
            {
                "failure_id": "legacy-tool",
                "status": "human_required",
                "frontier_attempts": 1,
                "frontier_result": {
                    "decision": "needs_retry",
                    "human_required": True,
                    "frontier_failure": {
                        "failure_class": "frontier_tool_unavailable",
                        "human_required": True,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    before = packet_path.read_bytes()

    assert packet_path in self_heal.pending_packets()
    result = self_heal.handle_packet(packet_path, dry_run=True)

    assert result["would_reclassify_human_boundary"] is True
    assert result["projected_status"] == "frontier_retry"
    assert packet_path.read_bytes() == before


def test_frontier_quarantine_reopens_after_cooldown(
    isolated_wiki: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from llm_wiki_mcp import self_heal

    packet_path = isolated_wiki / "runtime" / "failures" / "packets" / "quarantine.json"
    packet_path.parent.mkdir(parents=True, exist_ok=True)
    packet_path.write_text(
        json.dumps(
            {
                "failure_id": "quarantine",
                "status": "frontier_quarantined",
                "frontier_attempts": 3,
                "self_heal_attempts": 3,
                "quarantined_at": "2000-01-01T00:00:00",
                "local_decision": {
                    "status": "escalate",
                    "action": "escalate_to_frontier",
                    "confidence": 0.9,
                    "reason": "retry after model recovery",
                    "source": "deterministic",
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("LLM_WIKI_CONVERGENCE_QUARANTINE_RETRY_SECONDS", "1")
    monkeypatch.setattr(
        self_heal,
        "_run_frontier",
        lambda *_args, **_kwargs: {
            "decision": "rejected",
            "summary": "review completed after recovery",
            "human_required": False,
        },
    )

    assert packet_path in self_heal.pending_packets()
    result = self_heal.handle_packet(packet_path, use_qwen=False)

    updated = json.loads(packet_path.read_text(encoding="utf-8"))
    assert result["status"] == "frontier_rejected"
    assert updated["frontier_attempts"] == 1
    assert updated["quarantine_reopen_count"] == 1
    assert updated["terminal_resume_kind"] == "quarantine_cooldown"


def test_external_human_boundary_is_rechecked_after_fix_window(
    isolated_wiki: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from llm_wiki_mcp import self_heal

    packet_path = isolated_wiki / "runtime" / "failures" / "packets" / "oauth.json"
    packet_path.parent.mkdir(parents=True, exist_ok=True)
    packet_path.write_text(
        json.dumps(
            {
                "failure_id": "oauth",
                "status": "human_required",
                "human_required_at": "2000-01-01T00:00:00",
                "frontier_result": {
                    "decision": "needs_retry",
                    "frontier_failure": {"failure_class": "oauth_required"},
                },
                "local_decision": {
                    "status": "escalate",
                    "action": "escalate_to_frontier",
                    "confidence": 0.9,
                    "reason": "retry after login",
                    "source": "deterministic",
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("LLM_WIKI_HUMAN_REQUIRED_RECHECK_SECONDS", "1")
    monkeypatch.setattr(
        self_heal,
        "_run_frontier",
        lambda *_args, **_kwargs: {
            "decision": "approved",
            "summary": "authentication is now available",
            "human_required": False,
        },
    )

    assert packet_path in self_heal.pending_packets()
    result = self_heal.handle_packet(packet_path, use_qwen=False)

    updated = json.loads(packet_path.read_text(encoding="utf-8"))
    assert result["status"] == "frontier_approved"
    assert updated["human_recheck_count"] == 1
    assert updated["terminal_resume_kind"] == "external_authority_recheck"


def test_handle_packet_dry_run_is_byte_for_byte_read_only(isolated_wiki: Path) -> None:
    from llm_wiki_mcp import self_heal

    packet_path = _write_packet(isolated_wiki)
    before = packet_path.read_bytes()
    before = packet_path.read_bytes()

    result = self_heal.handle_packet(packet_path, use_qwen=False, dry_run=True)

    assert result["status"] == "dry_run"
    assert packet_path.read_bytes() == before
    failures = isolated_wiki / "runtime" / "failures"
    assert not (failures / "local-repair").exists()
    assert not (failures / "frontier-queue").exists()
    assert not (failures / "locks").exists()


def _force_frontier(monkeypatch: pytest.MonkeyPatch, self_heal) -> None:
    from llm_wiki_mcp.local_repair import LocalRepairDecision

    monkeypatch.setattr(
        self_heal,
        "propose_repair",
        lambda *_args, **_kwargs: LocalRepairDecision(
            status="escalate",
            action="escalate_to_frontier",
            confidence=0.9,
            reason="needs frontier",
            source="deterministic",
        ),
    )


class _RecordingBudget:
    def __init__(self, **allowed: bool) -> None:
        self.allowed = allowed
        self.calls: list[str] = []

    def consume(self, kind: str):
        self.calls.append(kind)
        permitted = self.allowed.get(kind, True)
        return permitted, "ok" if permitted else f"{kind}_budget_exhausted"

    def can_consume(self, kind: str):
        permitted = self.allowed.get(kind, True)
        return permitted, "ok" if permitted else f"{kind}_budget_exhausted"


def test_run_pending_local_budget_defer_is_no_progress_and_skips_proposal(
    isolated_wiki: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from llm_wiki_mcp import self_heal

    packet_path = _write_packet(isolated_wiki)
    before = packet_path.read_bytes()
    budget = _RecordingBudget(local=False)
    monkeypatch.setattr(
        self_heal,
        "propose_repair",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("local proposal must not run without budget")
        ),
    )

    result = self_heal.run_pending(
        max_packets=1,
        use_qwen=False,
        frontier_budget=budget,
    )

    assert result["results"][0]["status"] == "budget_deferred"
    assert result["results"][0]["budget_kind"] == "local"
    assert budget.calls == ["local"]
    assert packet_path.read_bytes() == before
    assert not (isolated_wiki / "runtime" / "failures" / "local-repair").exists()


def test_local_mutation_budget_defer_preserves_packet_progress(
    isolated_wiki: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from llm_wiki_mcp import self_heal

    _seed_page(isolated_wiki, "ai/canonical-target.md")
    packet_path = _write_packet(isolated_wiki)
    before = packet_path.read_bytes()
    budget = _RecordingBudget(local=True, mutation=False)
    monkeypatch.setattr(
        self_heal,
        "apply_local_decision",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("local mutation must not run without budget")
        ),
    )

    result = self_heal.handle_packet(
        packet_path,
        use_qwen=False,
        frontier_budget=budget,
    )

    assert result["status"] == "budget_deferred"
    assert result["budget_kind"] == "mutation"
    assert result["local_decision"]["status"] == "resolved"
    assert budget.calls == ["local"]
    assert packet_path.read_bytes() == before
    failures = isolated_wiki / "runtime" / "failures"
    assert not (failures / "local-repair").exists()
    assert not (failures / "applied-actions").exists()


def test_successful_local_repair_charges_local_and_mutation(
    isolated_wiki: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from llm_wiki_mcp import self_heal
    from llm_wiki_mcp.local_repair import LocalRepairDecision

    packet_path = _write_packet(isolated_wiki)
    budget = _RecordingBudget(local=True, mutation=True)
    monkeypatch.setattr(
        self_heal,
        "propose_repair",
        lambda *_args, **_kwargs: LocalRepairDecision(
            status="resolved",
            action="quarantine_raw",
            confidence=0.99,
            reason="keep quarantined",
            source="deterministic",
        ),
    )
    monkeypatch.setattr(
        self_heal,
        "_run_frontier",
        lambda *_args, **_kwargs: {
            "decision": "approved",
            "summary": "quarantine is semantically justified",
            "human_required": False,
        },
    )

    result = self_heal.handle_packet(
        packet_path,
        use_qwen=False,
        frontier_budget=budget,
    )

    assert result["status"] == "frontier_approved"
    assert result["action"]["action"] == "quarantine_raw"
    assert budget.calls == ["local", "frontier", "mutation"]


def test_frontier_only_executable_retry_charges_frontier_and_mutation_budget(
    isolated_wiki: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from llm_wiki_mcp import self_heal

    packet_path = _write_packet(isolated_wiki)
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    packet.update(
        {
            "status": "pending_frontier",
            "local_repair_attempts": 1,
            "frontier_attempts": 0,
            "local_decision": {
                "status": "escalate",
                "action": "escalate_to_frontier",
                "confidence": 0.9,
                "reason": "needs frontier",
                "source": "deterministic",
            },
        }
    )
    packet_path.write_text(json.dumps(packet), encoding="utf-8")
    budget = _RecordingBudget(frontier=True, mutation=True)
    monkeypatch.setattr(
        self_heal,
        "propose_repair",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("frontier-only retry must reuse the local decision")
        ),
    )
    monkeypatch.setattr(
        self_heal,
        "_run_frontier",
        lambda *_args, **_kwargs: {
            "decision": "rejected",
            "summary": "not safe",
            "human_required": False,
        },
    )

    result = self_heal.handle_packet(
        packet_path,
        use_qwen=False,
        frontier_budget=budget,
    )
    updated = json.loads(packet_path.read_text(encoding="utf-8"))

    assert result["status"] == "frontier_rejected"
    assert budget.calls == ["frontier", "mutation"]
    assert updated["local_repair_attempts"] == 1
    assert updated["frontier_attempts"] == 1


def test_frontier_budget_defer_does_not_consume_attempt(
    isolated_wiki: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from llm_wiki_mcp import self_heal

    packet_path = _write_packet(isolated_wiki)
    before = packet_path.read_bytes()
    _force_frontier(monkeypatch, self_heal)
    monkeypatch.setattr(
        self_heal,
        "_run_frontier",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("frontier must not run without budget")
        ),
    )

    class DeniedBudget:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def consume(self, kind: str):
            self.calls.append(kind)
            if kind == "local":
                return True, "ok"
            return False, "frontier_budget_exhausted"

        def can_consume(self, kind: str):
            if kind == "local":
                return True, "ok"
            if kind == "mutation":
                return True, "ok"
            return False, "frontier_budget_exhausted"

    budget = DeniedBudget()
    result = self_heal.handle_packet(
        packet_path,
        use_qwen=False,
        frontier_budget=budget,
        backoff_base_seconds=0,
    )
    packet = json.loads(packet_path.read_text(encoding="utf-8"))

    assert result["status"] == "budget_deferred"
    assert result["budget_kind"] == "frontier"
    assert packet["status"] == "pending_local_repair"
    assert int(packet.get("frontier_attempts") or 0) == 0
    assert int(packet.get("self_heal_attempts") or 0) == 0
    assert int(packet.get("local_repair_attempts") or 0) == 0
    assert budget.calls == ["local"]
    assert packet_path.read_bytes() == before


def test_frontier_exception_becomes_retry_and_releases_running_lease(
    isolated_wiki: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from llm_wiki_mcp import self_heal

    packet_path = _write_packet(isolated_wiki)
    _force_frontier(monkeypatch, self_heal)
    monkeypatch.setattr(
        self_heal,
        "_run_frontier",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(TimeoutError("timed out")),
    )

    result = self_heal.handle_packet(
        packet_path,
        use_qwen=False,
        backoff_base_seconds=0,
    )
    packet = json.loads(packet_path.read_text(encoding="utf-8"))

    assert result["status"] == "frontier_retry"
    assert result["frontier_error"]["exception_type"] == "TimeoutError"
    assert packet["status"] == "frontier_retry"
    assert packet["frontier_attempts"] == 1
    assert packet["self_heal_attempts"] == 1
    assert packet["lease_owner"] is None
    assert packet["lease_expires_at"] is None
    assert packet["next_attempt_at"] is not None


def test_transient_read_back_packet_is_retired_without_frontier(
    isolated_wiki: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from llm_wiki_mcp import self_heal

    packet_path = isolated_wiki / "runtime" / "failures" / "packets" / "read-back.json"
    packet_path.parent.mkdir(parents=True, exist_ok=True)
    packet_path.write_text(
        json.dumps(
            {
                "failure_id": "read-back",
                "raw_file": "read-back-target",
                "failure_class": "read_back.repeated_miss",
                "fingerprint": "read_back.repeated_miss:read-back-key",
                "attempts": 2,
                "error": (
                    "ingest read-back repair exhausted its bounded attempts: "
                    "model temporarily unavailable"
                ),
                "status": "frontier_running",
                "frontier_attempts": 1,
                "self_heal_attempts": 1,
                "lease_expires_at": "2000-01-01T00:00:00",
                "local_decision": {
                    "status": "escalate",
                    "action": "escalate_to_frontier",
                    "confidence": 1.0,
                    "reason": "bounded operational repair attempts were exhausted",
                    "source": "deterministic",
                },
                "raw_preview": json.dumps(
                    {
                        "failure": {
                            "page_id": "target",
                            "reason": "search-error",
                            "error": "model temporarily unavailable",
                        }
                    }
                ),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        self_heal,
        "propose_repair",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("transient read-back packets must not run local repair")
        ),
    )
    monkeypatch.setattr(
        self_heal,
        "_run_frontier",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("transient read-back packets must not call frontier")
        ),
    )

    result = self_heal.handle_packet(packet_path, use_qwen=False)

    updated = json.loads(packet_path.read_text(encoding="utf-8"))
    assert result["status"] == "frontier_rejected"
    assert result["reason"] == "transient_read_back_operational_failure"
    assert updated["status"] == "frontier_rejected"
    assert updated["frontier_status"] == "not_required"
    assert updated["frontier_attempts"] == 1
    assert updated["lease_owner"] is None
    assert updated["lease_expires_at"] is None
    assert Path(result["rejected_action_path"]).exists()


def test_exhausted_read_back_query_hint_packet_is_retired_without_frontier(
    isolated_wiki: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from llm_wiki_mcp import self_heal

    packet_path = isolated_wiki / "runtime" / "failures" / "packets" / "read-back.json"
    packet_path.parent.mkdir(parents=True, exist_ok=True)
    exhausted = "read-back miss persisted after exact query hint was applied"
    packet_path.write_text(
        json.dumps(
            {
                "failure_id": "read-back",
                "raw_file": "read-back-target",
                "failure_class": "read_back.repeated_miss",
                "fingerprint": "read_back.repeated_miss:read-back-key",
                "attempts": 3,
                "error": f"ingest read-back repair exhausted its bounded attempts: {exhausted}",
                "status": "frontier_running",
                "frontier_attempts": 1,
                "self_heal_attempts": 1,
                "lease_expires_at": "2000-01-01T00:00:00",
                "local_decision": {
                    "status": "escalate",
                    "action": "escalate_to_frontier",
                    "confidence": 1.0,
                    "reason": "bounded operational repair attempts were exhausted",
                    "source": "deterministic",
                },
                "raw_preview": json.dumps(
                    {
                        "failure": {
                            "page_id": "target",
                            "reason": "not-in-top-results",
                            "query": "How does Setouchi interact with Azure OpenAI?",
                        },
                        "ledger_entry": {
                            "last_error": exhausted,
                            "frontier_review": {"decision": "approved"},
                        },
                    }
                ),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        self_heal,
        "propose_repair",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("exhausted read-back hints must not run local repair")
        ),
    )
    monkeypatch.setattr(
        self_heal,
        "_run_frontier",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("exhausted read-back hints must not call frontier")
        ),
    )

    result = self_heal.handle_packet(packet_path, use_qwen=False)

    updated = json.loads(packet_path.read_text(encoding="utf-8"))
    assert result["status"] == "frontier_rejected"
    assert result["reason"] == "exhausted_read_back_query_hint"
    assert updated["status"] == "frontier_rejected"
    assert updated["frontier_status"] == "not_required"
    assert updated["frontier_attempts"] == 1
    assert updated["lease_owner"] is None
    assert updated["lease_expires_at"] is None
    assert Path(result["rejected_action_path"]).exists()


def test_unverifiable_read_back_query_hint_packet_is_retired_without_frontier(
    isolated_wiki: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from llm_wiki_mcp import self_heal

    packet_path = isolated_wiki / "runtime" / "failures" / "packets" / "read-back.json"
    packet_path.parent.mkdir(parents=True, exist_ok=True)
    error = (
        "The available workspace evidence does not include the target page "
        "`codex-session-memory-save-protocol` or matching content for the "
        "proposed `Self + AI` query"
    )
    packet_path.write_text(
        json.dumps(
            {
                "failure_id": "read-back",
                "raw_file": "read-back-codex-session-memory-save-protocol",
                "failure_class": "read_back.repeated_miss",
                "fingerprint": "read_back.repeated_miss:read-back-key",
                "attempts": 3,
                "error": f"ingest read-back repair exhausted its bounded attempts: {error}",
                "status": "frontier_running",
                "frontier_attempts": 1,
                "self_heal_attempts": 1,
                "lease_expires_at": "2000-01-01T00:00:00",
                "local_decision": {
                    "status": "escalate",
                    "action": "escalate_to_frontier",
                    "confidence": 1.0,
                    "reason": "bounded operational repair attempts were exhausted",
                    "source": "deterministic",
                },
                "raw_preview": json.dumps(
                    {
                        "failure": {
                            "page_id": "codex-session-memory-save-protocol",
                            "reason": "not-in-top-results",
                            "query": "What is the Self + AI integration model?",
                        },
                        "ledger_entry": {
                            "last_error": error,
                        },
                    }
                ),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        self_heal,
        "propose_repair",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("unverifiable read-back hints must not run local repair")
        ),
    )
    monkeypatch.setattr(
        self_heal,
        "_run_frontier",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("unverifiable read-back hints must not call frontier")
        ),
    )

    result = self_heal.handle_packet(packet_path, use_qwen=False)

    updated = json.loads(packet_path.read_text(encoding="utf-8"))
    assert result["status"] == "frontier_rejected"
    assert result["reason"] == "unverifiable_read_back_query_hint"
    assert updated["status"] == "frontier_rejected"
    assert updated["frontier_status"] == "not_required"
    assert updated["frontier_attempts"] == 1
    assert updated["lease_owner"] is None
    assert updated["lease_expires_at"] is None
    assert Path(result["rejected_action_path"]).exists()


def test_transient_read_back_dry_run_is_byte_for_byte_read_only(
    isolated_wiki: Path,
) -> None:
    from llm_wiki_mcp import self_heal

    packet_path = isolated_wiki / "runtime" / "failures" / "packets" / "read-back.json"
    packet_path.parent.mkdir(parents=True, exist_ok=True)
    packet_path.write_text(
        json.dumps(
            {
                "failure_id": "read-back",
                "raw_file": "read-back-target",
                "failure_class": "read_back.repeated_miss",
                "status": "pending_frontier",
                "error": "model temporarily unavailable",
                "raw_preview": json.dumps(
                    {
                        "ledger_entry": {
                            "failure": {
                                "page_id": "target",
                                "reason": "search-error",
                                "error": "model temporarily unavailable",
                            }
                        }
                    }
                ),
            }
        ),
        encoding="utf-8",
    )
    before = packet_path.read_bytes()

    result = self_heal.handle_packet(packet_path, dry_run=True)

    assert result["status"] == "dry_run"
    assert result["projected_status"] == "frontier_rejected"
    assert packet_path.read_bytes() == before
    assert not (isolated_wiki / "runtime" / "failures" / "rejected-actions").exists()


def test_pending_packets_recovers_only_expired_running_leases(
    isolated_wiki: Path,
) -> None:
    from datetime import datetime, timedelta

    from llm_wiki_mcp import self_heal

    now = datetime(2026, 7, 10, 12, 0, 0)
    packet_dir = isolated_wiki / "runtime" / "failures" / "packets"
    packet_dir.mkdir(parents=True, exist_ok=True)
    expired = packet_dir / "expired.json"
    active = packet_dir / "active.json"
    expired.write_text(
        json.dumps(
            {
                "status": "frontier_running",
                "lease_expires_at": (now - timedelta(seconds=1)).isoformat(),
            }
        ),
        encoding="utf-8",
    )
    active.write_text(
        json.dumps(
            {
                "status": "local_repairing",
                "lease_expires_at": (now + timedelta(hours=1)).isoformat(),
            }
        ),
        encoding="utf-8",
    )

    assert self_heal.pending_packets(now=now) == [expired]


def test_packet_lock_enforces_single_flight_without_packet_mutation(
    isolated_wiki: Path,
) -> None:
    from llm_wiki_mcp import self_heal

    packet_path = _write_packet(isolated_wiki)
    before = packet_path.read_bytes()

    with self_heal._packet_lock(packet_path) as acquired:
        assert acquired is True
        result = self_heal.handle_packet(packet_path, use_qwen=False)

    assert result["status"] == "busy"
    assert result["reason"] == "packet_already_running"
    assert packet_path.read_bytes() == before


def test_completed_packet_is_cached_instead_of_reprocessed(
    isolated_wiki: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from llm_wiki_mcp import self_heal

    _seed_page(isolated_wiki, "ai/canonical-target.md")
    packet_path = _write_packet(isolated_wiki)
    quarantined = isolated_wiki / "runtime" / "failures" / "quarantined-raw" / "broken.md"
    quarantined.parent.mkdir(parents=True, exist_ok=True)
    quarantined.write_text("raw body", encoding="utf-8")
    monkeypatch.setattr(
        self_heal,
        "_retry_ingest",
        lambda *, dry_run: {"triggered": True, "files_processed": ["broken.md"]},
    )
    monkeypatch.setattr(
        self_heal,
        "_run_frontier",
        lambda *_args, **_kwargs: {
            "decision": "approved",
            "summary": "exact alias repair is supported",
            "human_required": False,
        },
    )

    first = self_heal.handle_packet(packet_path, use_qwen=False)
    after_first = packet_path.read_bytes()
    monkeypatch.setattr(
        self_heal,
        "propose_repair",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("completed packet must not be proposed again")
        ),
    )
    second = self_heal.handle_packet(packet_path, use_qwen=False)

    assert first["status"] == "frontier_approved"
    assert second["status"] == "frontier_approved"
    assert second["cached"] is True
    assert packet_path.read_bytes() == after_first
