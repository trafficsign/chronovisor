from __future__ import annotations

import hashlib
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

    from llm_wiki_mcp import (
        claims,
        index_store,
        ingest,
        ollama,
        orchestrator,
        page_mutation,
        runtime_status,
        search,
        state_register,
        wiki,
    )

    monkeypatch.setattr(wiki, "WIKI_ROOT", wiki_root)
    monkeypatch.setattr(wiki, "PAGES_DIR", pages)
    monkeypatch.setattr(wiki, "RAW_DIR", raw)
    monkeypatch.setattr(wiki, "SYSTEM_DIR", system)
    monkeypatch.setattr(wiki, "INDEX_FILE", wiki_root / "index.md")
    monkeypatch.setattr(wiki, "LOG_FILE", wiki_root / "log.md")
    monkeypatch.setattr(ingest, "PAGES_DIR", pages)
    monkeypatch.setattr(ingest, "INDEX_FILE", wiki_root / "index.md")
    monkeypatch.setattr(ingest, "LOG_FILE", wiki_root / "log.md")
    monkeypatch.setattr(orchestrator, "RAW_DIR", raw)
    monkeypatch.setattr(orchestrator, "WIKI_ROOT", wiki_root)
    monkeypatch.setattr(orchestrator, "LOG_FILE", wiki_root / "log.md")
    monkeypatch.setattr(
        orchestrator,
        "STATE_FILE",
        wiki_root / ".orchestrator_state.json",
    )
    monkeypatch.setattr(
        page_mutation,
        "WIKI_MUTATION_LOCK",
        runtime / "wiki-mutation.lock",
    )
    monkeypatch.setattr(page_mutation, "WIKI_ROOT", wiki_root)
    monkeypatch.setattr(page_mutation, "PAGES_DIR", pages)
    monkeypatch.setattr(page_mutation, "SYSTEM_DIR", system)
    monkeypatch.setattr(runtime_status, "RUNTIME_DIR", runtime)
    monkeypatch.setattr(runtime_status, "STATUS_FILE", runtime / "status.json")
    monkeypatch.setattr(runtime_status, "EVENTS_FILE", runtime / "events.jsonl")
    monkeypatch.setattr(runtime_status, "METRICS_FILE", runtime / "metrics.jsonl")
    monkeypatch.setattr(state_register, "STATE_PAGE", system / "current-state.md")
    monkeypatch.setattr(
        state_register,
        "refresh_state_register",
        lambda *args, **kwargs: {
            "status": "unchanged",
            "path": str(system / "current-state.md"),
            "pages": list(args[0]) if args else [],
            "write": kwargs.get("write", True),
            "mutation": None,
        },
    )

    claims_dir = wiki_root / "claims"
    monkeypatch.setattr(claims, "WIKI_ROOT", wiki_root)
    monkeypatch.setattr(claims, "CLAIMS_DIR", claims_dir)
    monkeypatch.setattr(claims, "CLAIMS_FILE", claims_dir / "claims.jsonl")
    monkeypatch.setattr(
        claims,
        "CLAIM_INDEX_FILE",
        claims_dir / "claims-index.jsonl",
    )
    monkeypatch.setattr(
        claims,
        "CLAIM_CONFLICT_FILE",
        claims_dir / "claim-conflicts.jsonl",
    )
    monkeypatch.setattr(
        claims,
        "CLAIM_REVIEW_FILE",
        claims_dir / "claim-conflict-reviews.jsonl",
    )

    index_dir = wiki_root / ".index"
    index_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(index_store, "WIKI_ROOT", wiki_root)
    monkeypatch.setattr(index_store, "PAGES_DIR", pages)
    monkeypatch.setattr(index_store, "SYSTEM_DIR", system)
    monkeypatch.setattr(index_store, "INDEX_DIR", index_dir)
    monkeypatch.setattr(index_store, "PAGES_INDEX_FILE", index_dir / "pages.json")
    monkeypatch.setattr(
        index_store,
        "BACKLINKS_INDEX_FILE",
        index_dir / "backlinks.json",
    )
    monkeypatch.setattr(index_store, "_store", None)
    monkeypatch.setattr(ollama, "is_available", lambda: False)
    monkeypatch.setattr(search, "update_embeddings", lambda page_ids=None: 0)
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


def _mark_system_code_repair(
    packet_path: Path,
    *,
    local_repair_attempts: int = 2,
) -> None:
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    packet["job_id"] = "trusted-watchdog"
    packet["failure_class"] = "system_health_snapshot_exception"
    packet["incident_kind"] = "system_code_repair"
    packet["local_repair_attempts"] = local_repair_attempts
    repair_evidence = [f"{index + 1:064x}" for index in range(local_repair_attempts)]
    packet["local_repair_evidence"] = repair_evidence
    packet["repair_evidence"] = {
        "role": "code_repair",
        "incident_kind": "system_code_repair",
        "component": "watchdog.health_snapshot",
        "fingerprint": packet["fingerprint"],
        "failure_class": "system_health_snapshot_exception",
        "occurrence_count": 3,
        "distinct_inputs": ["input-a", "input-b"],
        "local_repair_attempts": local_repair_attempts,
        "local_repair_evidence": repair_evidence,
        "reproduction_command": [
            "uv",
            "run",
            "pytest",
            "-q",
            "tests/test_self_heal.py",
        ],
        "notes": {"producer": "trusted_watchdog", "incident_key": packet_path.stem},
    }
    packet_path.write_text(json.dumps(packet), encoding="utf-8")


def test_deterministic_single_alias_repair_applies_locally_without_frontier(
    isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from llm_wiki_mcp import self_heal
    from llm_wiki_mcp.alias_store import load_aliases

    _seed_page(isolated_wiki, "ai/canonical-target.md")
    packet_path = _write_packet(isolated_wiki)
    quarantined = (
        isolated_wiki / "runtime" / "failures" / "quarantined-raw" / "broken.md"
    )
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
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("routine semantic repair must not call frontier")
        ),
    )
    result = self_heal.handle_packet(
        packet_path,
        use_qwen=False,
        enable_frontier=True,
        dry_run=False,
    )

    assert result["status"] == "local_repair_applied"
    assert result["action"]["action"] == "resolve_update_target"
    assert load_aliases()["model-made-up-target"] == "ai/canonical-target"
    assert not quarantined.exists()
    assert (isolated_wiki / "raw" / "broken.md").exists()
    updated_packet = json.loads(packet_path.read_text())
    assert updated_packet["status"] == "local_repair_applied"
    assert not (isolated_wiki / "runtime" / "failures" / "frontier-queue").exists()


def test_drill_returns_local_repair_decision(isolated_wiki: Path) -> None:
    from llm_wiki_mcp.self_heal import run_drill

    result = run_drill(use_qwen=False)

    assert result["decision"]["status"] == "resolved"
    assert result["decision"]["action"] == "resolve_update_target"


def test_auto_apply_error_packet_preserves_local_test_case_deterministically(
    isolated_wiki: Path,
) -> None:
    from llm_wiki_mcp.local_repair import propose_repair

    packet = {
        "failure_class": "recall.auto_apply_error",
        "fingerprint": "recall.auto_apply_error:page_tag:invalid_page_tag",
        "attempts": 425,
        "auto_apply_error": {"error_kind": "page_tag:invalid_page_tag"},
    }

    decision = propose_repair(packet, use_qwen=False)

    assert decision.status == "escalate"
    assert decision.action == "propose_test_case"
    assert decision.confidence >= 0.85


def test_local_consensus_repair_carries_authority_seal(
    isolated_wiki: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from llm_wiki_mcp import decision_policy, decision_router, local_repair
    from llm_wiki_mcp.decision_router import canonical_agreement_signature
    from llm_wiki_mcp.local_repair import LOCAL_REPAIR_SCHEMA

    models = ["primary-model", "challenger-model", "tie-model"]
    router_audit = {
        "source": "adopted_artifact",
        "artifact_sha256": "d" * 64,
        "error": None,
        "models": models,
    }
    authority = {
        "source": "adopted_local_consensus",
        "authority_version": 1,
        "lane": "local_repair",
        "lane_contract_sha256": "a" * 64,
        "lane_contract_manifest_sha256": "b" * 64,
        "lane_contract_case_manifest_sha256": "c" * 64,
        "policy": {
            "kind": "consensus",
            "schema_name": "local_repair",
            "mode": "enabled",
            "error": None,
        },
        "router": router_audit,
    }
    policy = decision_policy.DECISION_POLICIES["local_repair"]
    action = {
        "status": "resolved",
        "action": "retry_raw",
        "confidence": 0.91,
        "reason": "retry the validated raw packet",
    }
    signature = canonical_agreement_signature(action, schema=LOCAL_REPAIR_SCHEMA)
    agreement = hashlib.sha256(signature.encode("utf-8")).hexdigest()

    class FakePolicy:
        source = "adopted_artifact"

        @staticmethod
        def audit_record():
            return router_audit

    class FakeResult:
        ok = True
        decision = action

        @staticmethod
        def audit_record():
            return {
                "status": "agreed",
                "ok": True,
                "agreement_sha256": agreement,
                "failure_class": None,
                "quarantine_reason": None,
                "num_ctx": 16_384,
                "residency": None,
                "votes": [
                    {
                        "role": role,
                        "model": model,
                        "valid": index < 2,
                        "signature_sha256": agreement if index < 2 else None,
                        "invalid_reason": None,
                    }
                    for index, (role, model) in enumerate(
                        zip(
                            ("primary", "challenger", "tie_break"),
                            models,
                            strict=True,
                        )
                    )
                ],
            }

    class FakeRouter:
        policy = FakePolicy()

        def __init__(self, **_kwargs) -> None:
            pass

        def decide(self, *_args, **_kwargs):
            return FakeResult()

    monkeypatch.setattr(
        decision_policy,
        "resolve_decision_policy",
        lambda _lane: (policy, "enabled", None),
    )
    monkeypatch.setattr(decision_router, "DecisionRouter", FakeRouter)
    monkeypatch.setattr(
        local_repair,
        "current_semantic_authority",
        lambda _lane: (authority, None),
    )

    decision = local_repair.propose_repair(
        {"failure_class": "semantic.ambiguous_repair"},
        use_qwen=True,
    )

    assert decision.source == "local_consensus"
    assert decision.authority == authority
    assert decision.decision_policy["mode"] == "enabled"
    assert decision.local_consensus["agreement_sha256"] == agreement


def test_local_consensus_repair_fails_closed_when_authority_changes_before_effect(
    isolated_wiki: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from llm_wiki_mcp import self_heal
    from llm_wiki_mcp.local_repair import LocalRepairDecision

    packet_path = _write_packet(isolated_wiki)
    decision = LocalRepairDecision(
        status="resolved",
        action="retry_raw",
        confidence=0.91,
        reason="retry",
        source="local_consensus",
        authority={
            "source": "injected_reviewer_boundary",
            "authority_version": 1,
            "lane": "local_repair",
        },
        decision_policy={},
        local_consensus={},
    )
    monkeypatch.setattr(
        self_heal,
        "propose_repair",
        lambda *_args, **_kwargs: decision,
    )
    monkeypatch.setattr(
        self_heal,
        "current_semantic_authority",
        lambda _lane: (None, "decision_lane_not_enabled:local_repair:shadow"),
    )
    monkeypatch.setattr(
        self_heal,
        "apply_local_decision",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("stale local decision must not be applied")
        ),
    )

    result = self_heal.handle_packet(packet_path, use_qwen=True)

    assert result["status"] == "local_repair_failed"
    assert "decision_lane_not_enabled" in result["local_error"]


@pytest.mark.parametrize(
    "failure_class",
    [
        "semantic.claim_conflict",
        "content.correction_requested",
        "triage.json_parse_failed",
    ],
)
def test_routine_packets_stay_local_even_when_frontier_switch_is_enabled(
    isolated_wiki: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_class: str,
) -> None:
    from llm_wiki_mcp import self_heal
    from llm_wiki_mcp.local_repair import LocalRepairDecision

    packet_path = _write_packet(isolated_wiki)
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    packet["failure_class"] = failure_class
    packet_path.write_text(json.dumps(packet), encoding="utf-8")
    monkeypatch.setattr(
        self_heal,
        "propose_repair",
        lambda *_args, **_kwargs: LocalRepairDecision(
            status="escalate",
            action="escalate_to_frontier",
            confidence=0.9,
            reason="local repair could not resolve the packet",
            source="deterministic",
        ),
    )
    monkeypatch.setattr(
        self_heal,
        "_run_frontier",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("routine packets must never call frontier")
        ),
    )

    result = self_heal.handle_packet(
        packet_path,
        use_qwen=False,
        enable_frontier=True,
        max_attempts=1,
    )

    assert result["status"] == "local_quarantined"
    assert result["reason"] == "frontier_repair_not_eligible"
    assert not (isolated_wiki / "runtime" / "failures" / "frontier-queue").exists()


def test_valid_system_code_evidence_is_forwarded_to_frontier(
    isolated_wiki: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from llm_wiki_mcp import self_heal
    from llm_wiki_mcp.frontier_guard import RepairIncidentEvidence

    packet_path = _write_packet(isolated_wiki)
    _mark_system_code_repair(packet_path)
    _force_frontier(monkeypatch, self_heal)
    captured: list[RepairIncidentEvidence] = []

    def fake_frontier(*_args, evidence: RepairIncidentEvidence, **_kwargs):
        captured.append(evidence)
        return {
            "decision": "rejected",
            "summary": "repair is not safe",
            "human_required": False,
        }

    monkeypatch.setattr(self_heal, "_run_frontier", fake_frontier)

    result = self_heal.handle_packet(
        packet_path,
        use_qwen=False,
        enable_frontier=True,
    )

    assert result["status"] == "frontier_rejected"
    assert len(captured) == 1
    assert captured[0].incident_kind == "system_code_repair"
    assert captured[0].local_repair_attempts == 2


def test_frontier_caller_passes_same_evidence_capability(
    isolated_wiki: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from llm_wiki_mcp import frontier_review, self_heal

    packet_path = _write_packet(isolated_wiki)
    _mark_system_code_repair(packet_path)
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    evidence = self_heal._repair_incident_evidence(packet)
    captured: dict[str, object] = {}

    class Result:
        def to_dict(self) -> dict[str, object]:
            return {"decision": "rejected", "summary": "test result"}

    def fake_run_frontier_review(*_args, **kwargs):
        captured.update(kwargs)
        return Result()

    monkeypatch.setattr(
        frontier_review,
        "run_frontier_review",
        fake_run_frontier_review,
    )

    result = self_heal._run_frontier(
        packet_path,
        packet,
        None,
        evidence=evidence,
        execute_patch=False,
    )

    assert result["decision"] == "rejected"
    assert captured["evidence"] is evidence
    assert captured["execute_patch"] is False


def test_incomplete_system_code_evidence_is_quarantined_locally(
    isolated_wiki: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from llm_wiki_mcp import self_heal

    packet_path = _write_packet(isolated_wiki)
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    packet.update(
        {
            "incident_kind": "system_code_repair",
            "repair_evidence": {"fingerprint": packet["fingerprint"]},
        }
    )
    packet_path.write_text(json.dumps(packet), encoding="utf-8")
    monkeypatch.setattr(
        self_heal,
        "_run_frontier",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("invalid evidence must not call frontier")
        ),
    )

    result = self_heal.handle_packet(
        packet_path,
        use_qwen=False,
        enable_frontier=True,
        max_attempts=1,
    )

    assert result["status"] == "local_quarantined"
    assert result["reason"] == "frontier_repair_not_eligible"
    assert "missing required fields" in result["frontier_eligibility_error"]


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
        decision.requested_page_id == "claude-code-vs-claude-code-structural-analysis"
    )
    assert decision.confidence >= 0.85


def test_frontier_budget_nonconvergence_retries_raw_deterministically(
    isolated_wiki: Path,
) -> None:
    from llm_wiki_mcp.local_repair import propose_repair

    packet = {
        "failure_class": "ingest.frontier_nonconvergent",
        "fingerprint": "ingest.frontier_nonconvergent",
        "attempts": 1,
        "error": (
            "frontier ingest review did not converge after 2 frontier calls: "
            "frontier call budget exhausted; route the raw to local self-heal "
            "instead of calling frontier again"
        ),
        "requested_page_id": None,
        "similar_existing_pages": [],
    }

    def stale_qwen(*_args, **_kwargs) -> str:
        raise AssertionError("frontier budget exhaustion must not ask Qwen")

    decision = propose_repair(packet, generator=stale_qwen, use_qwen=True)

    assert decision.status == "resolved"
    assert decision.action == "retry_raw"
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
    assert decision.action == "propose_test_case"


def test_missing_update_target_requires_exact_packet_and_request_evidence(
    isolated_wiki: Path,
) -> None:
    from llm_wiki_mcp.local_repair import propose_repair

    missing_request = {
        "failure_class": "apply.update_target_not_found",
        "requested_page_id": None,
        "similar_existing_pages": ["existing-page"],
    }
    multiple_candidates = {
        "failure_class": "apply.update_target_not_found",
        "requested_page_id": "missing-page",
        "similar_existing_pages": ["existing-page", "other-page"],
    }

    missing_request_decision = propose_repair(missing_request, use_qwen=False)
    multiple_candidate_decision = propose_repair(
        multiple_candidates,
        generator=lambda *_args, **_kwargs: json.dumps(
            {
                "status": "resolved",
                "action": "resolve_update_target",
                "confidence": 0.99,
                "requested_page_id": "missing-page",
                "target_page_id": "existing-page",
                "reason": "picked one candidate",
            }
        ),
        use_qwen=True,
    )

    assert missing_request_decision.action == "propose_test_case"
    assert multiple_candidate_decision.action == "propose_test_case"


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
        isolated_wiki / "runtime" / "failures" / "quarantined-raw" / "new-topic.md"
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


def test_frontier_nonconvergence_restores_raw_without_frontier(
    isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from llm_wiki_mcp import self_heal

    packet = {
        "failure_id": "frontier-loop",
        "raw_file": "frontier-loop.md",
        "failure_class": "ingest.frontier_nonconvergent",
        "fingerprint": "ingest.frontier_nonconvergent",
        "attempts": 1,
        "error": (
            "frontier ingest review did not converge after 2 frontier calls: "
            "frontier call budget exhausted; route the raw to local self-heal "
            "instead of calling frontier again"
        ),
        "requested_page_id": None,
        "similar_existing_pages": [],
        "status": "frontier_running",
        "frontier_attempts": 1,
        "self_heal_attempts": 1,
        "lease_expires_at": "2000-01-01T00:00:00",
        "local_decision": {
            "status": "escalate",
            "action": "escalate_to_frontier",
            "confidence": 0.7,
            "reason": "stale model decision",
            "source": "qwen",
        },
    }
    packet_path = (
        isolated_wiki / "runtime" / "failures" / "packets" / "frontier-loop.json"
    )
    packet_path.parent.mkdir(parents=True, exist_ok=True)
    packet_path.write_text(json.dumps(packet), encoding="utf-8")
    quarantined = (
        isolated_wiki / "runtime" / "failures" / "quarantined-raw" / "frontier-loop.md"
    )
    quarantined.parent.mkdir(parents=True, exist_ok=True)
    quarantined.write_text("raw body", encoding="utf-8")

    monkeypatch.setattr(
        self_heal,
        "_retry_ingest",
        lambda *, dry_run: {"triggered": True, "files_processed": ["frontier-loop.md"]},
    )
    monkeypatch.setattr(
        self_heal,
        "_run_frontier",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("frontier nonconvergence must re-enter local repair")
        ),
    )

    result = self_heal.handle_packet(
        packet_path,
        use_qwen=True,
        enable_frontier=True,
        dry_run=False,
    )

    updated_packet = json.loads(packet_path.read_text())
    assert result["status"] == "local_repair_applied"
    assert result["action"]["action"] == "retry_raw"
    assert not quarantined.exists()
    assert (isolated_wiki / "raw" / "frontier-loop.md").exists()
    assert updated_packet["status"] == "local_repair_applied"
    assert updated_packet["local_decision"]["source"] == "deterministic"


def test_sandbox_drill_keeps_semantic_repair_out_of_frontier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from llm_wiki_mcp.self_heal import run_sandbox_drill

    monkeypatch.setenv("LLM_WIKI_SELF_HEAL_AUTORUN", "0")

    result = run_sandbox_drill(use_qwen=False)

    assert result["status"] == "ok"
    assert result["packet_paths"]
    assert result["heal_result"]["status"] == "local_repair_applied"
    assert result["heal_result"]["action"]["action"] == "resolve_update_target"
    assert result["pending_after"] == []
    assert result["aliases"]["opus-4-7-evaluation-and-industry-geopolitics"] == (
        "ai/opus-4.7-evaluation-and-industry-geopolitics"
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
    _mark_system_code_repair(packet_path)
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

    result = self_heal.handle_packet(
        packet_path,
        use_qwen=False,
        enable_frontier=True,
    )

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
    _mark_system_code_repair(packet_path)

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

    result = self_heal.handle_packet(
        packet_path,
        use_qwen=False,
        enable_frontier=True,
    )

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

    packet_path = (
        isolated_wiki / "runtime" / "failures" / "packets" / "legacy-tool.json"
    )
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


def test_frontier_quarantine_is_terminal_after_execution_started(
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
                "failure_class": "adapter_contract_failure",
                "fingerprint": "adapter-contract:quarantine",
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
    _mark_system_code_repair(packet_path)
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

    before = packet_path.read_bytes()
    assert packet_path not in self_heal.pending_packets()
    assert packet_path.read_bytes() == before


def test_external_human_boundary_never_calls_frontier(
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
                "failure_class": "oauth_required",
                "fingerprint": "oauth-required",
                "status": "pending_local_repair",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        self_heal,
        "_run_frontier",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("external authority failures must not call frontier")
        ),
    )
    monkeypatch.setattr(
        self_heal,
        "_send_mac_notification",
        lambda *_args, **_kwargs: {"sent": True},
    )

    result = self_heal.handle_packet(
        packet_path,
        use_qwen=False,
        enable_frontier=True,
    )

    updated = json.loads(packet_path.read_text(encoding="utf-8"))
    assert result["status"] == "human_required"
    assert result["reason"] == "external_authority_boundary"
    assert updated["status"] == "human_required"
    assert updated["frontier_status"] == "not_attempted"


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


def test_deterministic_alias_repair_respects_mutation_budget(
    isolated_wiki: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from llm_wiki_mcp import self_heal

    _seed_page(isolated_wiki, "ai/canonical-target.md")
    packet_path = _write_packet(isolated_wiki)
    budget = _RecordingBudget(local=True, mutation=False)
    monkeypatch.setattr(
        self_heal,
        "apply_local_decision",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("mutation must not run without budget")
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
    assert budget.calls == ["local", "mutation"]
    failures = isolated_wiki / "runtime" / "failures"
    assert not (failures / "local-repair").exists()
    assert not (failures / "applied-actions").exists()


def test_deterministic_quarantine_completes_locally_without_frontier(
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
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("routine quarantine proposal must not call frontier")
        ),
    )

    result = self_heal.handle_packet(
        packet_path,
        use_qwen=False,
        frontier_budget=budget,
    )

    assert result["status"] == "local_repair_applied"
    assert result["action"] == {
        "action": "quarantine_raw",
        "kept_quarantined": True,
    }
    assert budget.calls == ["local", "mutation"]


def test_frontier_only_executable_retry_charges_only_local_mutation_budget(
    isolated_wiki: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from llm_wiki_mcp import self_heal

    packet_path = _write_packet(isolated_wiki)
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    packet.update(
        {
            "status": "pending_frontier",
            "local_repair_attempts": 2,
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
    _mark_system_code_repair(packet_path)
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
        enable_frontier=True,
        frontier_budget=budget,
    )
    updated = json.loads(packet_path.read_text(encoding="utf-8"))

    assert result["status"] == "frontier_rejected"
    assert budget.calls == ["mutation"]
    assert updated["local_repair_attempts"] == 2
    assert updated["frontier_attempts"] == 1


def test_mutation_budget_defer_does_not_consume_frontier_attempt(
    isolated_wiki: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from llm_wiki_mcp import self_heal

    packet_path = _write_packet(isolated_wiki)
    _mark_system_code_repair(packet_path)
    before = packet_path.read_bytes()
    _force_frontier(monkeypatch, self_heal)
    monkeypatch.setattr(
        self_heal,
        "_run_frontier",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("frontier must not run without mutation budget")
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
                return False, "mutation_budget_exhausted"
            return True, "ok"

    budget = DeniedBudget()
    result = self_heal.handle_packet(
        packet_path,
        use_qwen=False,
        enable_frontier=True,
        frontier_budget=budget,
        backoff_base_seconds=0,
    )
    packet = json.loads(packet_path.read_text(encoding="utf-8"))

    assert result["status"] == "budget_deferred"
    assert result["budget_kind"] == "mutation"
    assert packet["status"] == "pending_local_repair"
    assert int(packet.get("frontier_attempts") or 0) == 0
    assert int(packet.get("self_heal_attempts") or 0) == 0
    assert int(packet.get("local_repair_attempts") or 0) == 2
    assert budget.calls == ["local"]
    assert packet_path.read_bytes() == before


def test_guard_denial_does_not_consume_frontier_attempt(
    isolated_wiki: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from llm_wiki_mcp import self_heal

    packet_path = _write_packet(isolated_wiki)
    _mark_system_code_repair(packet_path)
    _force_frontier(monkeypatch, self_heal)
    monkeypatch.setattr(
        self_heal,
        "_run_frontier",
        lambda *_args, **_kwargs: {
            "decision": "needs_retry",
            "summary": "daily guard budget is unavailable",
            "rescue_status": "repair_deferred",
            "frontier_failure": {"failure_class": "frontier_guard_denied"},
            "execution_started": False,
        },
    )

    result = self_heal.handle_packet(
        packet_path,
        use_qwen=False,
        enable_frontier=True,
        backoff_base_seconds=0,
    )
    packet = json.loads(packet_path.read_text(encoding="utf-8"))

    assert result["status"] == "repair_deferred"
    assert packet["status"] == "repair_deferred"
    assert packet["frontier_attempts"] == 0
    assert packet["self_heal_attempts"] == 0


def test_guard_deferred_system_repair_resumes_code_patch_without_local_reproposal(
    isolated_wiki: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from llm_wiki_mcp import self_heal

    packet_path = _write_packet(isolated_wiki)
    _mark_system_code_repair(packet_path)
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    packet.update(
        {
            "status": "pending_frontier",
            "frontier_attempts": 0,
            "local_decision": {
                "status": "unresolved",
                "action": "none",
                "source": "trusted_system_incident_supervisor",
                "notes": "local repairs exhausted",
            },
        }
    )
    packet_path.write_text(json.dumps(packet), encoding="utf-8")
    persisted_decision = dict(packet["local_decision"])
    persisted_evidence = dict(packet["repair_evidence"])
    calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        self_heal,
        "propose_repair",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("deferred system repair must not become routine review")
        ),
    )

    def fake_frontier(
        _packet_path,
        _packet,
        local_decision,
        *,
        evidence,
        execute_patch,
    ):
        calls.append(
            {
                "local_decision": dict(local_decision),
                "evidence": evidence.to_dict(),
                "execute_patch": execute_patch,
            }
        )
        if len(calls) == 1:
            return {
                "decision": "needs_retry",
                "summary": "guard cooldown is active",
                "rescue_status": "repair_deferred",
                "rescue_attempt": {
                    "guard_reason": "frontier_cooldown_active",
                },
                "frontier_failure": {"failure_class": "frontier_guard_denied"},
                "execution_started": False,
            }
        return {
            "decision": "approved",
            "summary": "isolated code repair verified",
            "human_required": False,
            "verified": True,
            "execution_started": True,
        }

    monkeypatch.setattr(self_heal, "_run_frontier", fake_frontier)

    deferred = self_heal.handle_packet(
        packet_path,
        use_qwen=False,
        enable_frontier=True,
        backoff_base_seconds=0,
    )
    after_defer = json.loads(packet_path.read_text(encoding="utf-8"))

    assert deferred["status"] == "repair_deferred"
    assert after_defer["local_decision"] == persisted_decision
    assert after_defer["repair_evidence"] == persisted_evidence
    assert after_defer["frontier_attempts"] == 0

    recovered = self_heal.handle_packet(
        packet_path,
        use_qwen=False,
        enable_frontier=True,
        backoff_base_seconds=0,
    )
    after_recovery = json.loads(packet_path.read_text(encoding="utf-8"))

    assert recovered["status"] == "frontier_approved"
    assert len(calls) == 2
    assert calls[1]["local_decision"] == persisted_decision
    assert calls[1]["evidence"] == calls[0]["evidence"]
    assert calls[1]["execute_patch"] is True
    assert after_recovery["local_decision"] == persisted_decision
    assert after_recovery["repair_evidence"] == persisted_evidence
    assert after_recovery["frontier_attempts"] == 1


def test_frontier_exception_after_start_is_terminal_and_releases_running_lease(
    isolated_wiki: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from llm_wiki_mcp import self_heal

    packet_path = _write_packet(isolated_wiki)
    _mark_system_code_repair(packet_path)
    _force_frontier(monkeypatch, self_heal)
    monkeypatch.setattr(
        self_heal,
        "_run_frontier",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(TimeoutError("timed out")),
    )

    result = self_heal.handle_packet(
        packet_path,
        use_qwen=False,
        enable_frontier=True,
        backoff_base_seconds=0,
    )
    packet = json.loads(packet_path.read_text(encoding="utf-8"))

    assert result["status"] == "frontier_quarantined"
    assert result["frontier_error"]["exception_type"] == "TimeoutError"
    assert packet["status"] == "frontier_quarantined"
    assert packet["frontier_attempts"] == 1
    assert packet["self_heal_attempts"] == 1
    assert packet["lease_owner"] is None
    assert packet["lease_expires_at"] is None
    assert packet.get("next_attempt_at") is None


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


def test_empty_query_read_back_packet_is_retired_without_frontier(
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
                "raw_file": "read-back-empty",
                "failure_class": "read_back.repeated_miss",
                "fingerprint": "read_back.repeated_miss:read-back-key",
                "attempts": 2,
                "error": (
                    "ingest read-back repair exhausted its bounded attempts: "
                    "empty-query"
                ),
                "status": "frontier_running",
                "frontier_attempts": 2,
                "self_heal_attempts": 2,
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
                        "failure": {"page_id": "empty", "reason": "empty-query"},
                        "ledger_entry": {
                            "attempts": 2,
                            "failure": {
                                "page_id": "empty",
                                "reason": "empty-query",
                            },
                            "last_error": "empty-query",
                            "status": "quarantined",
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
            AssertionError("empty-query read-back packets must not run local repair")
        ),
    )
    monkeypatch.setattr(
        self_heal,
        "_run_frontier",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("empty-query read-back packets must not call frontier")
        ),
    )

    result = self_heal.handle_packet(packet_path, use_qwen=False)

    updated = json.loads(packet_path.read_text(encoding="utf-8"))
    assert result["status"] == "frontier_rejected"
    assert result["reason"] == "empty_query_read_back_failure"
    assert updated["status"] == "frontier_rejected"
    assert updated["frontier_status"] == "not_required"
    assert updated["frontier_attempts"] == 2
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
    monkeypatch.setattr(
        self_heal,
        "_run_frontier",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("routine packet must not call frontier")
        ),
    )

    first = self_heal.handle_packet(
        packet_path,
        use_qwen=False,
        max_attempts=1,
    )
    after_first = packet_path.read_bytes()
    monkeypatch.setattr(
        self_heal,
        "propose_repair",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("completed packet must not be proposed again")
        ),
    )
    second = self_heal.handle_packet(packet_path, use_qwen=False)

    assert first["status"] == "local_repair_applied"
    assert second["status"] == "local_repair_applied"
    assert second["cached"] is True
    assert packet_path.read_bytes() == after_first


def test_start_background_uses_durable_queue_without_popen(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from llm_wiki_mcp import background_jobs, self_heal

    packet = tmp_path / "packet.json"
    packet.write_text("{}", encoding="utf-8")
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setenv("LLM_WIKI_SELF_HEAL_AUTORUN", "1")
    monkeypatch.setattr(
        self_heal.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("routine self-heal must not detach"),
        raising=False,
    )
    seen: dict[str, object] = {}

    def fake_enqueue(**kwargs):
        seen.update(kwargs)
        return {"job_id": "job-1", "status": "queued", "enqueued": True}

    monkeypatch.setattr(background_jobs, "enqueue_job", fake_enqueue)

    result = self_heal.start_background(packet)

    assert result == {"job_id": "job-1", "status": "queued", "enqueued": True}
    assert seen["name"] == "self-heal"
    assert seen["module"] == "llm_wiki_mcp.self_heal"
    assert seen["args"] == ["--packet", str(packet.resolve())]


def test_background_exit_code_preserves_retry_and_terminal_states() -> None:
    from llm_wiki_mcp import background_jobs, self_heal

    assert self_heal._background_exit_code({"status": "local_repair_applied"}) == 0
    assert self_heal._background_exit_code({"status": "pending_local_repair"}) == (
        background_jobs.RETRYABLE_EXIT_CODE
    )
    assert self_heal._background_exit_code({"status": "human_required"}) == (
        background_jobs.QUARANTINE_EXIT_CODE
    )
    assert (
        self_heal._background_exit_code(
            {
                "status": "ok",
                "results": [
                    {"status": "local_repair_applied"},
                    {"status": "frontier_retry"},
                ],
            }
        )
        == background_jobs.RETRYABLE_EXIT_CODE
    )
