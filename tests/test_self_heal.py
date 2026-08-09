from __future__ import annotations

import hashlib
import json
import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture()
def ingest_ollama_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> list[str]:
    """Stub the import-by-value availability consumer used by ingest."""

    from chronovisor.ingest import orchestrator

    calls: list[str] = []

    def unavailable() -> bool:
        calls.append("is_available")
        return False

    monkeypatch.setattr(orchestrator, "is_available", unavailable)
    return calls


@pytest.fixture()
def isolated_wiki(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ingest_ollama_unavailable: list[str],
) -> Path:
    del ingest_ollama_unavailable
    chronovisor_root = tmp_path / "wiki"
    pages = chronovisor_root / "pages"
    raw = chronovisor_root / "raw"
    system = chronovisor_root / "system"
    runtime = chronovisor_root / "runtime"
    for d in (pages, raw, system, runtime):
        d.mkdir(parents=True, exist_ok=True)

    from chronovisor.core import ollama, page_mutation, store
    from chronovisor.ingest import ingest, orchestrator
    from chronovisor.ops import background_jobs, runtime_status, state_register
    from chronovisor.search import claims, index_store, search

    monkeypatch.setattr(store, "CHRONOVISOR_ROOT", chronovisor_root)
    monkeypatch.setattr(store, "PAGES_DIR", pages)
    monkeypatch.setattr(store, "RAW_DIR", raw)
    monkeypatch.setattr(store, "SYSTEM_DIR", system)
    monkeypatch.setattr(store, "INDEX_FILE", chronovisor_root / "index.md")
    monkeypatch.setattr(store, "LOG_FILE", chronovisor_root / "log.md")
    monkeypatch.setattr(ingest, "PAGES_DIR", pages)
    monkeypatch.setattr(ingest, "INDEX_FILE", chronovisor_root / "index.md")
    monkeypatch.setattr(ingest, "LOG_FILE", chronovisor_root / "log.md")
    monkeypatch.setattr(orchestrator, "RAW_DIR", raw)
    monkeypatch.setattr(orchestrator, "CHRONOVISOR_ROOT", chronovisor_root)
    monkeypatch.setattr(orchestrator, "LOG_FILE", chronovisor_root / "log.md")
    monkeypatch.setattr(
        orchestrator,
        "STATE_FILE",
        chronovisor_root / ".orchestrator_state.json",
    )
    monkeypatch.setattr(
        page_mutation,
        "CHRONOVISOR_MUTATION_LOCK",
        runtime / "wiki-mutation.lock",
    )
    monkeypatch.setattr(
        page_mutation,
        "DECISION_AUTHORITY_LOCK",
        runtime / "decision-authority.lock",
    )
    monkeypatch.setattr(page_mutation, "CHRONOVISOR_ROOT", chronovisor_root)
    monkeypatch.setattr(page_mutation, "PAGES_DIR", pages)
    monkeypatch.setattr(page_mutation, "SYSTEM_DIR", system)
    monkeypatch.setattr(runtime_status, "RUNTIME_DIR", runtime)
    monkeypatch.setattr(runtime_status, "STATUS_FILE", runtime / "status.json")
    monkeypatch.setattr(runtime_status, "EVENTS_FILE", runtime / "events.jsonl")
    monkeypatch.setattr(runtime_status, "METRICS_FILE", runtime / "metrics.jsonl")
    monkeypatch.setattr(background_jobs, "JOB_DIR", runtime / "background-jobs")
    monkeypatch.setattr(
        background_jobs,
        "STATE_FILE",
        runtime / "background-jobs" / "state.json",
    )
    monkeypatch.setattr(
        background_jobs,
        "LOCK_FILE",
        runtime / "background-jobs" / "state.lock",
    )
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

    claims_dir = chronovisor_root / "claims"
    monkeypatch.setattr(claims, "CHRONOVISOR_ROOT", chronovisor_root)
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

    index_dir = chronovisor_root / ".index"
    index_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(index_store, "CHRONOVISOR_ROOT", chronovisor_root)
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
    return chronovisor_root


def test_ingest_ollama_fixture_patches_import_by_value_consumer(
    monkeypatch: pytest.MonkeyPatch,
    ingest_ollama_unavailable: list[str],
) -> None:
    from chronovisor.core import ollama
    from chronovisor.ingest import orchestrator

    monkeypatch.setattr(ollama, "is_available", lambda: True)

    assert orchestrator.is_available() is False
    assert ingest_ollama_unavailable == ["is_available"]


def _seed_page(chronovisor_root: Path, rel: str) -> None:
    path = chronovisor_root / "pages" / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("---\ntitle: T\nupdated: 2026-01-01\n---\nbody\n")


def _write_packet(chronovisor_root: Path) -> Path:
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
    path = chronovisor_root / "runtime" / "failures" / "packets" / "f1.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(packet))
    return path


def _strict_no_quorum_audit(models: list[str]) -> dict:
    reason = "local_models_did_not_reach_two_vote_quorum"

    def vote(role: str, model: str, digit: str) -> dict:
        return {
            "role": role,
            "model": model,
            "requested_num_ctx": 32768,
            "valid": True,
            "signature_sha256": digit * 64,
            "invalid_reason": None,
            "runtime_observation": {
                "status": "observed",
                "model_size_bytes": 1024,
                "num_ctx": 32768,
            },
            "session": {
                "ok": True,
                "model": model,
                "failure_class": None,
                "first_pass_valid": True,
                "repair_turns": 0,
                "attempts": [
                    {
                        "index": 1,
                        "valid": True,
                        "output_sha256": digit * 64,
                        "output_chars": 16,
                        "normalized": False,
                        "error_fingerprint": None,
                        "issues": [],
                    }
                ],
            },
        }

    return {
        "status": "quarantined",
        "ok": False,
        "quorum_safety_policy_version": 1,
        "agreement_sha256": None,
        "failure_class": "local_consensus_failed",
        "quarantine_reason": reason,
        "num_ctx": 32768,
        "residency": {},
        "votes": [
            vote("primary", models[0], "a"),
            vote("challenger", models[1], "b"),
            vote("tie_break", models[2], "c"),
        ],
    }


def _install_local_no_quorum_router(
    monkeypatch: pytest.MonkeyPatch,
    *,
    cache_root: Path,
    artifact_digit: str = "d",
) -> tuple[dict, list[str]]:
    from chronovisor.decision import (
        decision_policy,
        decision_router,
        local_repair,
        routine_review,
        semantic_hold,
    )

    models = ["primary-model", "challenger-model", "tie-model"]
    router_audit = {
        "source": "adopted_artifact",
        "artifact_sha256": artifact_digit * 64,
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
    audit = _strict_no_quorum_audit(models)
    calls: list[str] = []

    class FakePolicy:
        source = "adopted_artifact"

        @staticmethod
        def audit_record():
            return router_audit

    class FakeResult:
        ok = False
        decision = None
        failure_class = "local_consensus_failed"
        quarantine_reason = "local_models_did_not_reach_two_vote_quorum"
        votes = tuple(
            SimpleNamespace(
                valid=True,
                signature_sha256=digit * 64,
            )
            for digit in ("a", "b", "c")
        )

        @staticmethod
        def audit_record():
            return audit

    class FakeRouter:
        policy = FakePolicy()

        def __init__(self, **_kwargs) -> None:
            pass

        def decide(self, *_args, **_kwargs):
            calls.append(artifact_digit)
            return FakeResult()

    monkeypatch.setattr(
        decision_policy,
        "resolve_decision_policy",
        lambda _lane: (
            decision_policy.DECISION_POLICIES["local_repair"],
            "enabled",
            None,
        ),
    )
    monkeypatch.setattr(decision_router, "DecisionRouter", FakeRouter)
    monkeypatch.setattr(
        local_repair,
        "current_semantic_authority",
        lambda _lane: (authority, None),
    )
    monkeypatch.setattr(
        routine_review,
        "STRUCTURED_REVIEW_HOLD_CACHE_ROOT",
        cache_root,
    )
    monkeypatch.setattr(
        routine_review,
        "_current_structured_authority",
        lambda lane: local_repair.current_semantic_authority(lane),
    )
    monkeypatch.setattr(
        routine_review,
        "_structured_authority_observation",
        lambda current: semantic_hold.canonical_sha256(
            {"authority": current, "fixture_generation": "stable"}
        ),
    )
    return authority, calls


def _write_operational_packet(chronovisor_root: Path) -> Path:
    path = _write_packet(chronovisor_root)
    packet = json.loads(path.read_text(encoding="utf-8"))
    packet.update(
        {
            "failure_class": "ingest.generation_context_window_exceeded",
            "fingerprint": "ingest.generation_context_window_exceeded",
            "error": (
                "ingest generation context_window_exceeded: required context "
                "268078 exceeds configured max_num_ctx 262144"
            ),
        }
    )
    path.write_text(json.dumps(packet), encoding="utf-8")
    return path


def _link_operational_system_incident(
    chronovisor_root: Path,
    source_packet_path: Path,
    *,
    status: str = "pending_frontier",
    frontier_attempts: int = 0,
    execution_started: bool = False,
) -> Path:
    source = json.loads(source_packet_path.read_text(encoding="utf-8"))
    incident_fingerprint = "f" * 64
    incident_path = (
        chronovisor_root
        / "runtime"
        / "failures"
        / "packets"
        / f"system-operational-{incident_fingerprint[:32]}.json"
    )
    incident = {
        "failure_id": incident_path.stem,
        "job_id": "trusted-operational-supervisor",
        "failure_class": "system_operational_failure",
        "fingerprint": incident_fingerprint,
        "incident_kind": "system_code_repair",
        "source_failure_class": source["failure_class"],
        "source_fingerprint": source["fingerprint"],
        "source_packet_paths": [str(source_packet_path.resolve())],
        "frontier_attempts": frontier_attempts,
        "status": status,
    }
    if execution_started:
        incident["frontier_result"] = {"execution_started": True}
    incident_path.write_text(json.dumps(incident), encoding="utf-8")
    source.update(
        {
            "system_incident_packet_path": str(incident_path.resolve()),
            "system_incident_fingerprint": incident_fingerprint,
            "system_incident_status": "packet_created",
        }
    )
    source_packet_path.write_text(json.dumps(source), encoding="utf-8")
    return incident_path


def _verified_git_state(commit: str) -> dict[str, str]:
    return {
        "git_commit_sha": commit,
        "checkout_head_sha": commit,
        "origin_main_sha": commit,
        "runtime_commit_sha": commit,
    }


def _expected_raw_manifest(*paths: Path) -> dict[str, str]:
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(paths, key=lambda item: item.name)
    }


def _generated_projection_child(chronovisor_root: Path) -> Path:
    from chronovisor.raw.raw_semantic_projection import (
        project_parent_raw,
        verify_projection_bundle,
    )
    from chronovisor.raw.save_transaction import (
        attach_save_transaction_marker,
        make_save_transaction,
    )

    parent = chronovisor_root / "raw" / "projection-parent.md"
    transaction = make_save_transaction(
        host="codex",
        session_file=chronovisor_root / "codex-session.jsonl",
        session_id="self-heal-projection",
        after_line=0,
        until_line=1,
    )
    content = "\n".join(
        [
            "# Codex Session Transcript Delta",
            "",
            "- Source: Codex",
            "- Capture mode: deterministic-lossless",
            "",
            "## Transcript Delta",
            "",
            "```json",
            json.dumps(
                [{"line": 1, "role": "user", "text": "repair evidence"}],
                ensure_ascii=False,
                indent=2,
            ),
            "```",
            "",
        ]
    )
    parent.write_text(
        attach_save_transaction_marker(transaction, content),
        encoding="utf-8",
    )
    projected = project_parent_raw(
        parent,
        output_dir=(chronovisor_root / "runtime" / "raw-projections" / "artifacts"),
        max_child_bytes=16_000,
    )
    assert projected.manifest_path is not None
    assert verify_projection_bundle(projected.manifest_path)["children"]
    assert len(projected.child_paths) == 1
    return projected.child_paths[0]


def _bind_operational_release_to_raw(
    chronovisor_root: Path,
    release_case: tuple[Path, dict[str, str]],
    raw_path: Path,
) -> tuple[Path, dict[str, str]]:
    packet_path, kwargs = release_case
    state_path = chronovisor_root / "runtime" / "failures" / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    entry = state["failures"].pop("broken.md")
    raw = raw_path.read_bytes()
    entry["raw_sha256"] = hashlib.sha256(raw).hexdigest()
    entry["raw_bytes"] = len(raw)
    state["failures"][raw_path.name] = entry
    state_path.write_text(json.dumps(state), encoding="utf-8")
    kwargs["expected_raw_sha256"] = {raw_path.name: hashlib.sha256(raw).hexdigest()}
    return packet_path, kwargs


@pytest.fixture()
def operational_release_case(
    isolated_wiki: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, dict[str, str]]:
    from chronovisor.ops import self_heal

    packet_path = _write_operational_packet(isolated_wiki)
    raw_path = isolated_wiki / "raw" / "broken.md"
    raw_path.write_text("immutable source", encoding="utf-8")
    state_path = isolated_wiki / "runtime" / "failures" / "state.json"
    state_path.write_text(
        json.dumps(
            {
                "failures": {
                    raw_path.name: {
                        "failure_class": "ingest.generation_context_window_exceeded",
                        "fingerprint": "ingest.generation_context_window_exceeded",
                        "self_heal_queued": True,
                        "packet_path": str(packet_path),
                        "raw_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
                        "raw_bytes": len(raw_path.read_bytes()),
                    }
                },
                "operational_failures": {
                    "ingest.generation_context_window_exceeded": {
                        "failure_class": "ingest.generation_context_window_exceeded",
                        "fingerprint": "ingest.generation_context_window_exceeded",
                        "self_heal_queued": True,
                        "packet_path": str(packet_path),
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        self_heal,
        "_verified_local_repair_git_state",
        _verified_git_state,
    )
    return packet_path, {
        "expected_status": "pending_local_repair",
        "expected_failure_class": "ingest.generation_context_window_exceeded",
        "expected_fingerprint": "ingest.generation_context_window_exceeded",
        "expected_raw_sha256": _expected_raw_manifest(raw_path),
        "repair_commit": "a" * 40,
        "reason": "bounded output repair",
    }


@pytest.mark.parametrize("bundle_state", ("valid", "tampered", "symlink"))
def test_verified_local_repair_reads_only_verified_projection_child_evidence(
    isolated_wiki: Path,
    operational_release_case: tuple[Path, dict[str, str]],
    bundle_state: str,
) -> None:
    from chronovisor.ops import self_heal

    child = _generated_projection_child(isolated_wiki)
    original = child.read_bytes()
    if bundle_state == "tampered":
        child.write_bytes(original + b" ")
    elif bundle_state == "symlink":
        outside = isolated_wiki / "runtime" / "projection-child-copy.md"
        outside.write_bytes(original)
        child.unlink()
        child.symlink_to(outside)
    packet_path, kwargs = _bind_operational_release_to_raw(
        isolated_wiki,
        operational_release_case,
        child,
    )
    before = packet_path.read_bytes()

    result = self_heal.release_operational_failure_after_local_repair(
        packet_path,
        **kwargs,
        dry_run=True,
    )

    assert packet_path.read_bytes() == before
    if bundle_state == "valid":
        assert result["accepted"] is True
        assert result["verified_local_repair"]["affected_raws"] == [
            {
                "filename": child.name,
                "sha256": hashlib.sha256(original).hexdigest(),
                "bytes": len(original),
                "binding_source": "failure_state",
            }
        ]
    else:
        assert result["accepted"] is False
        assert result["reason"] == "affected_raw_evidence_unavailable"


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


def test_operational_source_rejects_direct_raw_action(
    isolated_wiki: Path,
) -> None:
    from chronovisor.decision.local_repair import LocalRepairDecision
    from chronovisor.ops import self_heal

    packet = {
        "failure_class": "ingest.generation_transport_error",
        "raw_file": "generation-transport.md",
    }
    decision = LocalRepairDecision(
        status="resolved",
        action="retry_raw",
        confidence=1.0,
        reason="retry the raw",
        source="deterministic",
    )

    with pytest.raises(ValueError, match="guarded system-incident lane"):
        self_heal.apply_local_decision(packet, decision)


def test_operational_source_routes_resolved_raw_action_to_system_incident(
    isolated_wiki: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chronovisor.decision.local_repair import LocalRepairDecision
    from chronovisor.ops import self_heal

    packet_path = _write_packet(isolated_wiki)
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    packet.update(
        {
            "failure_class": "ingest.generation_transport_error",
            "fingerprint": "ingest.generation_transport_error:connection-reset",
            "raw_file": "generation-transport.md",
        }
    )
    packet_path.write_text(json.dumps(packet), encoding="utf-8")
    raw_path = isolated_wiki / "raw" / "generation-transport.md"
    raw_path.write_text("grounded source", encoding="utf-8")
    decision = LocalRepairDecision(
        status="resolved",
        action="retry_raw",
        confidence=1.0,
        reason="retry the raw",
        source="deterministic",
    )
    monkeypatch.setattr(self_heal, "propose_repair", lambda *_args, **_kwargs: decision)
    monkeypatch.setattr(
        self_heal,
        "apply_local_decision",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("operational raw action must not be applied directly")
        ),
    )
    promoted: list[tuple[Path, str]] = []

    def promote(path: Path, current: dict) -> dict:
        promoted.append((path, current["failure_class"]))
        return {
            "status": "pending",
            "packet_path": str(path.with_name("system-operational.json")),
        }

    monkeypatch.setattr(self_heal, "_promote_operational_source_packet", promote)

    result = self_heal.handle_packet(
        packet_path,
        use_qwen=False,
        enable_frontier=False,
        max_attempts=1,
        backoff_base_seconds=0,
    )

    updated = json.loads(packet_path.read_text(encoding="utf-8"))
    assert result["status"] == "local_quarantined"
    assert result["system_incident"]["status"] == "pending"
    assert updated["status"] == "local_quarantined"
    assert raw_path.exists()
    assert promoted == [(packet_path, "ingest.generation_transport_error")]


def test_deterministic_single_alias_repair_applies_locally_without_frontier(
    isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chronovisor.core.alias_store import load_aliases
    from chronovisor.ops import self_heal

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
    from chronovisor.ops.self_heal import run_drill

    result = run_drill(use_qwen=False)

    assert result["decision"]["status"] == "resolved"
    assert result["decision"]["action"] == "resolve_update_target"


def test_auto_apply_error_packet_preserves_local_test_case_deterministically(
    isolated_wiki: Path,
) -> None:
    from chronovisor.decision.local_repair import propose_repair

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
    from chronovisor.decision import (
        decision_policy,
        decision_router,
        local_repair,
        routine_review,
        semantic_hold,
    )
    from chronovisor.decision.decision_router import canonical_agreement_signature
    from chronovisor.decision.local_repair import LOCAL_REPAIR_SCHEMA

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
    monkeypatch.setattr(
        routine_review,
        "STRUCTURED_REVIEW_HOLD_CACHE_ROOT",
        isolated_wiki / "runtime" / "structured-review-holds",
    )
    monkeypatch.setattr(
        routine_review,
        "_current_structured_authority",
        lambda lane: local_repair.current_semantic_authority(lane),
    )
    monkeypatch.setattr(
        routine_review,
        "_structured_authority_observation",
        lambda current: semantic_hold.canonical_sha256(
            {"authority": current, "fixture_generation": "stable"}
        ),
    )

    decision = local_repair.propose_repair(
        {"failure_class": "semantic.ambiguous_repair"},
        use_qwen=True,
    )

    assert decision.source == "local_consensus"
    assert decision.authority == authority
    assert decision.decision_policy["mode"] == "enabled"
    assert decision.local_consensus["agreement_sha256"] == agreement


def test_local_repair_no_quorum_builds_strict_semantic_hold(
    isolated_wiki: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chronovisor.decision import local_repair
    from chronovisor.decision.semantic_hold import persisted_semantic_no_quorum_hold

    authority, calls = _install_local_no_quorum_router(
        monkeypatch,
        cache_root=isolated_wiki / "runtime" / "structured-review-holds",
    )
    packet = {
        "failure_id": "semantic-split",
        "failure_class": "semantic.ambiguous_repair",
        "fingerprint": "semantic.ambiguous_repair:one",
        "error": "two safe repairs remain plausible",
    }

    decision = local_repair.propose_repair(packet, use_qwen=True)

    assert calls == ["d"]
    assert decision.source == "semantic_hold"
    assert decision.status == "rejected"
    assert decision.action == "quarantine_raw"
    assert decision.authority == authority
    assert (
        persisted_semantic_no_quorum_hold(
            decision.to_dict(),
            "local_repair",
            epoch=local_repair.semantic_hold_epoch(packet),
            authority=authority,
        )
        == decision.semantic_hold
    )


def test_local_repair_semantic_hold_never_opens_live_authority_lock(
    isolated_wiki: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chronovisor.decision import local_repair

    _authority, calls = _install_local_no_quorum_router(
        monkeypatch,
        cache_root=isolated_wiki / "runtime" / "structured-review-holds",
    )
    live_lock = (Path.home() / ".chronovisor" / "runtime" / "decision-authority.lock").resolve(
        strict=False
    )
    isolated_lock = (isolated_wiki / "runtime" / "decision-authority.lock").resolve(
        strict=False
    )
    opened: list[Path] = []
    original_open = Path.open

    def guarded_open(path: Path, *args, **kwargs):
        resolved = path.resolve(strict=False)
        if resolved == live_lock:
            raise AssertionError("tests must never open the live authority lock")
        opened.append(resolved)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    decision = local_repair.propose_repair(
        {
            "failure_id": "isolated-authority-lock",
            "failure_class": "semantic.ambiguous_repair",
            "error": "two safe repairs remain plausible",
        },
        use_qwen=True,
    )

    assert decision.source == "semantic_hold"
    assert calls == ["d"]
    assert isolated_lock in opened
    assert live_lock not in opened


def test_local_repair_recovers_model_return_crash_from_common_cache_after_aba(
    isolated_wiki: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chronovisor.decision import local_repair, routine_review

    authority_a, calls = _install_local_no_quorum_router(
        monkeypatch,
        cache_root=isolated_wiki / "runtime" / "structured-review-holds",
    )
    authority_b = json.loads(json.dumps(authority_a))
    authority_b["lane_contract_sha256"] = "f" * 64
    authority_box = {"value": authority_a}
    observation_box = {"value": "0" * 64}
    monkeypatch.setattr(
        local_repair,
        "current_semantic_authority",
        lambda _lane: (authority_box["value"], None),
    )
    monkeypatch.setattr(
        routine_review,
        "_structured_authority_observation",
        lambda _authority: observation_box["value"],
    )
    packet = {
        "failure_id": "semantic-crash-window",
        "failure_class": "semantic.ambiguous_repair",
        "fingerprint": "semantic.ambiguous_repair:crash-window",
        "error": "two safe repairs remain plausible",
    }

    first_a = local_repair.propose_repair(packet, use_qwen=True)
    # Simulate a process crash after the common boundary persisted its result,
    # but before self-heal could write this returned lane decision to a packet.
    assert first_a.source == "semantic_hold"
    assert calls == ["d"]

    authority_box["value"] = authority_b
    observation_box["value"] = "1" * 64
    first_b = local_repair.propose_repair(packet, use_qwen=True)
    assert first_b.source == "semantic_hold"
    assert calls == ["d", "d"]

    authority_box["value"] = authority_a
    observation_box["value"] = "2" * 64
    recovered_a = local_repair.propose_repair(packet, use_qwen=True)

    assert recovered_a.source == "semantic_hold"
    assert recovered_a.authority == authority_a
    assert recovered_a.semantic_hold == first_a.semantic_hold
    assert calls == ["d", "d"]


def test_local_repair_semantic_epoch_matches_prompt_evidence_projection(
    isolated_wiki: Path,
) -> None:
    from chronovisor.decision import local_repair

    packet = {
        "failure_id": "semantic-split",
        "failure_class": "semantic.ambiguous_repair",
        "error": "two safe repairs remain plausible",
        "raw_preview": "grounded source evidence",
        "auto_apply_error": {"error_kind": "invalid_page_tag"},
        "status": "pending_local_repair",
    }
    initial_epoch = local_repair.semantic_hold_epoch(packet)
    initial_prompt = local_repair.build_prompt(packet)

    bookkeeping_changed = {
        **packet,
        "status": "local_quarantined",
        "updated_at": "2026-07-15T01:00:00",
        "local_decision": {"source": "semantic_hold"},
        "semantic_hold": {"prompt": "must never enter a repair prompt"},
        "semantic_hold_history": [{"raw_response": "also excluded"}],
    }
    assert local_repair.semantic_hold_epoch(bookkeeping_changed) == initial_epoch
    assert local_repair.build_prompt(bookkeeping_changed) == initial_prompt
    assert "semantic_hold" not in initial_prompt

    producer_evidence_changed = {
        **packet,
        "auto_apply_error": {"error_kind": "different_failure"},
    }
    assert local_repair.semantic_hold_epoch(producer_evidence_changed) != initial_epoch

    future_evidence_added = {**packet, "future_producer_evidence": {"value": 1}}
    assert local_repair.semantic_hold_epoch(future_evidence_added) != initial_epoch
    assert "future_producer_evidence" in local_repair.build_prompt(
        future_evidence_added
    )


def test_self_heal_no_quorum_is_terminal_until_exact_evidence_changes(
    isolated_wiki: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chronovisor.ops import self_heal

    authority, calls = _install_local_no_quorum_router(
        monkeypatch,
        cache_root=isolated_wiki / "runtime" / "structured-review-holds",
    )
    monkeypatch.setattr(
        self_heal,
        "current_semantic_authority",
        lambda _lane: (authority, None),
    )
    packet = {
        "failure_id": "semantic-split",
        "raw_file": "split.md",
        "failure_class": "semantic.ambiguous_repair",
        "fingerprint": "semantic.ambiguous_repair:one",
        "error": "two safe repairs remain plausible",
        "status": "pending_local_repair",
    }
    packet_path = (
        isolated_wiki / "runtime" / "failures" / "packets" / "semantic-split.json"
    )
    packet_path.parent.mkdir(parents=True, exist_ok=True)
    packet_path.write_text(json.dumps(packet), encoding="utf-8")

    first = self_heal.handle_packet(packet_path, enable_frontier=True)
    first_bytes = packet_path.read_bytes()
    cached = self_heal.handle_packet(packet_path, enable_frontier=True)
    dry_run = self_heal.handle_packet(
        packet_path,
        enable_frontier=True,
        dry_run=True,
    )

    assert first["status"] == "local_quarantined"
    assert first["semantic_deferred"] is True
    assert cached["cached"] is True
    assert dry_run["projected_status"] == "local_quarantined"
    assert packet_path.read_bytes() == first_bytes
    assert calls == ["d"]

    changed = json.loads(packet_path.read_text(encoding="utf-8"))
    changed["error"] = "the exact failure evidence changed"
    packet_path.write_text(json.dumps(changed), encoding="utf-8")
    assert packet_path in self_heal.pending_packets()

    reevaluated = self_heal.handle_packet(packet_path, enable_frontier=True)
    assert reevaluated["status"] == "local_quarantined"
    assert calls == ["d", "d"]
    latest = json.loads(packet_path.read_text(encoding="utf-8"))
    assert (
        latest["semantic_hold"]["epoch"]
        != first["local_decision"]["semantic_hold"]["epoch"]
    )


def test_existing_semantic_hold_dry_run_preserves_entire_isolated_tree(
    isolated_wiki: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chronovisor.ops import self_heal

    authority, calls = _install_local_no_quorum_router(
        monkeypatch,
        cache_root=isolated_wiki / "runtime" / "structured-review-holds",
    )
    monkeypatch.setattr(
        self_heal,
        "current_semantic_authority",
        lambda _lane: (authority, None),
    )
    packet_path = (
        isolated_wiki / "runtime" / "failures" / "packets" / "held-dry-run.json"
    )
    packet_path.parent.mkdir(parents=True, exist_ok=True)
    packet_path.write_text(
        json.dumps(
            {
                "failure_id": "held-dry-run",
                "failure_class": "semantic.ambiguous_repair",
                "fingerprint": "semantic.ambiguous_repair:held-dry-run",
                "error": "two safe repairs remain plausible",
                "status": "pending_local_repair",
            }
        ),
        encoding="utf-8",
    )

    first = self_heal.handle_packet(packet_path, enable_frontier=True)
    assert first["status"] == "local_quarantined"
    assert calls == ["d"]
    (isolated_wiki / "runtime" / "decision-authority.lock").unlink()

    def snapshot() -> tuple[list[str], dict[str, bytes]]:
        paths = sorted(
            path.relative_to(isolated_wiki).as_posix()
            for path in isolated_wiki.rglob("*")
        )
        files = {
            path.relative_to(isolated_wiki).as_posix(): path.read_bytes()
            for path in isolated_wiki.rglob("*")
            if path.is_file()
        }
        return paths, files

    before = snapshot()
    projected = self_heal.handle_packet(
        packet_path,
        enable_frontier=True,
        dry_run=True,
    )
    after = snapshot()

    assert projected["status"] == "dry_run"
    assert projected["cached"] is True
    assert projected["projected_status"] == "local_quarantined"
    assert after == before
    assert calls == ["d"]

    before_aggregate = snapshot()
    aggregate = self_heal.run_pending(dry_run=True)
    after_aggregate = snapshot()

    assert aggregate["status"] == "ok"
    assert aggregate["packets_seen"] == 0
    assert after_aggregate == before_aggregate
    assert calls == ["d"]


def test_self_heal_no_quorum_reuses_historical_hold_after_authority_aba(
    isolated_wiki: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chronovisor.decision import local_repair
    from chronovisor.ops import self_heal

    authority_a, calls = _install_local_no_quorum_router(
        monkeypatch,
        cache_root=isolated_wiki / "runtime" / "structured-review-holds",
    )
    authority_b = json.loads(json.dumps(authority_a))
    authority_b["lane_contract_sha256"] = "f" * 64
    current_authority = [authority_a]
    monkeypatch.setattr(
        local_repair,
        "current_semantic_authority",
        lambda _lane: (current_authority[0], None),
    )
    monkeypatch.setattr(
        self_heal,
        "current_semantic_authority",
        lambda _lane: (current_authority[0], None),
    )
    packet = {
        "failure_id": "semantic-aba",
        "raw_file": "split.md",
        "failure_class": "semantic.ambiguous_repair",
        "fingerprint": "semantic.ambiguous_repair:aba",
        "error": "two safe repairs remain plausible",
        "status": "pending_local_repair",
    }
    packet_path = (
        isolated_wiki / "runtime" / "failures" / "packets" / "semantic-aba.json"
    )
    packet_path.parent.mkdir(parents=True, exist_ok=True)
    packet_path.write_text(json.dumps(packet), encoding="utf-8")

    first_a = self_heal.handle_packet(packet_path, enable_frontier=True)
    assert first_a["status"] == "local_quarantined"
    assert calls == ["d"]

    current_authority[0] = authority_b
    first_b = self_heal.handle_packet(packet_path, enable_frontier=True)
    assert first_b["status"] == "local_quarantined"
    assert calls == ["d", "d"]

    current_authority[0] = authority_a
    restored_a = self_heal.handle_packet(packet_path, enable_frontier=True)
    assert restored_a["status"] == "local_quarantined"
    assert restored_a["cached"] is True
    assert calls == ["d", "d"]
    persisted = json.loads(packet_path.read_text(encoding="utf-8"))
    assert len(persisted["semantic_hold_history"]) == 2
    assert persisted["semantic_hold"]["authority"] == authority_a != authority_b


def test_local_consensus_repair_fails_closed_when_authority_changes_before_effect(
    isolated_wiki: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chronovisor.decision.local_repair import LocalRepairDecision
    from chronovisor.ops import self_heal

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
    from chronovisor.decision.local_repair import LocalRepairDecision
    from chronovisor.ops import self_heal

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
    from chronovisor.decision.frontier_guard import RepairIncidentEvidence
    from chronovisor.ops import self_heal

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
    from chronovisor.decision import frontier_review
    from chronovisor.ops import self_heal

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
    from chronovisor.ops import self_heal

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
    from chronovisor.decision.local_repair import propose_repair

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
    from chronovisor.decision.local_repair import propose_repair

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


def test_local_consensus_budget_nonconvergence_retries_raw_deterministically(
    isolated_wiki: Path,
) -> None:
    from chronovisor.decision.local_repair import propose_repair

    packet = {
        "failure_class": "ingest.local_consensus_nonconvergent",
        "fingerprint": "ingest.local_consensus_nonconvergent",
        "attempts": 1,
        "error": (
            "local consensus ingest review did not converge after "
            "2 local review calls: structured review budget exhausted (2/2)"
        ),
        "requested_page_id": None,
        "similar_existing_pages": [],
    }

    def stale_qwen(*_args, **_kwargs) -> str:
        raise AssertionError("local review budget exhaustion must not ask Qwen")

    decision = propose_repair(packet, generator=stale_qwen, use_qwen=True)

    assert decision.status == "resolved"
    assert decision.action == "retry_raw"
    assert decision.confidence >= 0.85


def test_missing_update_with_unsafe_page_id_still_escalates(
    isolated_wiki: Path,
) -> None:
    from chronovisor.decision.local_repair import propose_repair

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
    from chronovisor.decision.local_repair import propose_repair

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
    from chronovisor.ops import self_heal

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
    from chronovisor.ops import self_heal

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


def test_local_consensus_nonconvergence_restores_raw_without_frontier(
    isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chronovisor.ops import self_heal

    packet = {
        "failure_id": "local-consensus-loop",
        "raw_file": "local-consensus-loop.md",
        "failure_class": "ingest.local_consensus_nonconvergent",
        "fingerprint": "ingest.local_consensus_nonconvergent",
        "attempts": 1,
        "error": (
            "local consensus ingest review did not converge after "
            "2 local review calls: structured review budget exhausted (2/2)"
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
        isolated_wiki / "runtime" / "failures" / "packets" / "local-consensus-loop.json"
    )
    packet_path.parent.mkdir(parents=True, exist_ok=True)
    packet_path.write_text(json.dumps(packet), encoding="utf-8")
    quarantined = (
        isolated_wiki
        / "runtime"
        / "failures"
        / "quarantined-raw"
        / "local-consensus-loop.md"
    )
    quarantined.parent.mkdir(parents=True, exist_ok=True)
    quarantined.write_text("raw body", encoding="utf-8")

    monkeypatch.setattr(
        self_heal,
        "_retry_ingest",
        lambda *, dry_run: {
            "triggered": True,
            "files_processed": ["local-consensus-loop.md"],
        },
    )
    monkeypatch.setattr(
        self_heal,
        "_run_frontier",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("local consensus nonconvergence must stay local")
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
    assert (isolated_wiki / "raw" / "local-consensus-loop.md").exists()
    assert updated_packet["status"] == "local_repair_applied"
    assert updated_packet["local_decision"]["source"] == "deterministic"


def test_sandbox_drill_keeps_semantic_repair_out_of_frontier(
    monkeypatch: pytest.MonkeyPatch,
    ingest_ollama_unavailable: list[str],
) -> None:
    from chronovisor.ops.self_heal import run_sandbox_drill

    monkeypatch.setenv("CHRONOVISOR_SELF_HEAL_AUTORUN", "0")

    result = run_sandbox_drill(use_qwen=False)

    assert result["status"] == "ok"
    assert result["packet_paths"]
    assert result["heal_result"]["status"] == "local_repair_applied"
    assert result["heal_result"]["action"]["action"] == "resolve_update_target"
    assert result["pending_after"] == []
    assert result["aliases"]["opus-4-7-evaluation-and-industry-geopolitics"] == (
        "ai/opus-4.7-evaluation-and-industry-geopolitics"
    )
    assert ingest_ollama_unavailable


def test_frontier_human_required_sends_notification(
    isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chronovisor.decision.local_repair import LocalRepairDecision
    from chronovisor.ops import self_heal

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
    from chronovisor.decision.local_repair import LocalRepairDecision
    from chronovisor.ops import self_heal

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

    from chronovisor.ops import self_heal

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
    from chronovisor.ops import self_heal

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
    from chronovisor.ops import self_heal

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
    from chronovisor.ops import self_heal

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
    monkeypatch.setenv("CHRONOVISOR_CONVERGENCE_QUARANTINE_RETRY_SECONDS", "1")
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
    from chronovisor.ops import self_heal

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
    from chronovisor.ops import self_heal

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


def test_fresh_semantic_dry_run_never_starts_local_model_or_writes_audit(
    isolated_wiki: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chronovisor.decision import decision_router
    from chronovisor.ops import self_heal

    packet_path = (
        isolated_wiki / "runtime" / "failures" / "packets" / "ambiguous-dry-run.json"
    )
    packet_path.parent.mkdir(parents=True, exist_ok=True)
    packet_path.write_text(
        json.dumps(
            {
                "failure_id": "ambiguous-dry-run",
                "failure_class": "semantic.ambiguous_repair",
                "fingerprint": "semantic.ambiguous_repair:dry-run",
                "error": "two safe repairs remain plausible",
                "status": "pending_local_repair",
            }
        ),
        encoding="utf-8",
    )
    before = {
        path.relative_to(isolated_wiki).as_posix(): path.read_bytes()
        for path in isolated_wiki.rglob("*")
        if path.is_file()
    }
    monkeypatch.setattr(
        decision_router,
        "DecisionRouter",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("dry-run must not construct a local model router")
        ),
    )

    result = self_heal.handle_packet(packet_path, dry_run=True)

    after = {
        path.relative_to(isolated_wiki).as_posix(): path.read_bytes()
        for path in isolated_wiki.rglob("*")
        if path.is_file()
    }
    assert result["status"] == "dry_run"
    assert result["projected_status"] == "local_review_required"
    assert result["model_review_skipped"] is True
    assert result["local_decision"]["source"] == "deterministic"
    assert after == before


def _force_frontier(monkeypatch: pytest.MonkeyPatch, self_heal) -> None:
    from chronovisor.decision.local_repair import LocalRepairDecision

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
    from chronovisor.ops import self_heal

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
    from chronovisor.ops import self_heal

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
    from chronovisor.decision.local_repair import LocalRepairDecision
    from chronovisor.ops import self_heal

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
    from chronovisor.ops import self_heal

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
    from chronovisor.ops import self_heal

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
    from chronovisor.ops import self_heal

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
    from chronovisor.ops import self_heal

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
    from chronovisor.ops import self_heal

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
    from chronovisor.ops import self_heal

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
    from chronovisor.ops import self_heal

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
    from chronovisor.ops import self_heal

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
    from chronovisor.ops import self_heal

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
    from chronovisor.ops import self_heal

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

    from chronovisor.ops import self_heal

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
    from chronovisor.ops import self_heal

    packet_path = _write_packet(isolated_wiki)
    before = packet_path.read_bytes()

    with self_heal._packet_lock(packet_path) as acquired:
        assert acquired is True
        result = self_heal.handle_packet(packet_path, use_qwen=False)

    assert result["status"] == "busy"
    assert result["reason"] == "packet_already_running"
    assert packet_path.read_bytes() == before


def test_operational_failure_state_binds_each_raw_exact_bytes(
    isolated_wiki: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chronovisor.raw import failure_supervisor

    primary = isolated_wiki / "raw" / "primary.md"
    related = isolated_wiki / "raw" / "related.md"
    primary.write_bytes(b"primary\r\nsource\n")
    related.write_bytes(b"related\x00source\n")
    monkeypatch.setattr(failure_supervisor, "_launch_self_heal", lambda _path: None)

    result = failure_supervisor.record_raw_failure(
        raw_path=primary,
        related_raw_paths=(related,),
        raw_text=primary.read_text(encoding="utf-8"),
        error=(
            "ingest generation context_window_exceeded: required context "
            "268078 exceeds configured max_num_ctx 262144"
        ),
    )

    state = json.loads(
        (isolated_wiki / "runtime" / "failures" / "state.json").read_text(
            encoding="utf-8"
        )
    )
    assert result.packet_path
    for path in (primary, related):
        entry = state["failures"][path.name]
        assert entry["raw_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert entry["raw_bytes"] == len(path.read_bytes())


@pytest.mark.parametrize(
    "stored_binding",
    [
        {"raw_sha256": "valid_digest_is_injected_below"},
        {"raw_bytes": 0},
        {"raw_sha256": "malformed", "raw_bytes": 0},
    ],
    ids=("sha-only", "bytes-only", "malformed-complete"),
)
def test_repeated_operational_failure_preserves_invalid_binding_and_release_refuses(
    isolated_wiki: Path,
    monkeypatch: pytest.MonkeyPatch,
    stored_binding: dict[str, object],
) -> None:
    from chronovisor.ops import self_heal
    from chronovisor.raw import failure_supervisor

    raw_path = isolated_wiki / "raw" / "partial-binding.md"
    raw_path.write_bytes(b"immutable\r\nsource\n")
    error = (
        "ingest generation context_window_exceeded: required context "
        "268078 exceeds configured max_num_ctx 262144"
    )
    monkeypatch.setattr(failure_supervisor, "_launch_self_heal", lambda _path: None)
    first = failure_supervisor.record_raw_failure(
        raw_path=raw_path,
        raw_text=raw_path.read_text(encoding="utf-8"),
        error=error,
    )
    assert first.packet_path is not None

    binding = dict(stored_binding)
    if binding.get("raw_sha256") == "valid_digest_is_injected_below":
        binding["raw_sha256"] = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    if "raw_bytes" in binding:
        binding["raw_bytes"] = len(raw_path.read_bytes())
    state_path = isolated_wiki / "runtime" / "failures" / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    entry = state["failures"][raw_path.name]
    entry.pop("raw_sha256", None)
    entry.pop("raw_bytes", None)
    entry.update(binding)
    state_path.write_text(json.dumps(state), encoding="utf-8")

    repeated = failure_supervisor.record_raw_failure(
        raw_path=raw_path,
        raw_text=raw_path.read_text(encoding="utf-8"),
        error=error,
    )

    assert repeated.packet_path == first.packet_path
    refreshed = json.loads(state_path.read_text(encoding="utf-8"))["failures"][
        raw_path.name
    ]
    for field in ("raw_sha256", "raw_bytes"):
        assert refreshed.get(field) == binding.get(field)
        assert (field in refreshed) is (field in binding)

    monkeypatch.setattr(
        self_heal,
        "_verified_local_repair_git_state",
        lambda _commit: pytest.fail("invalid state binding must fail before git ACK"),
    )
    release = self_heal.release_operational_failure_after_local_repair(
        Path(first.packet_path),
        expected_status="pending_local_repair",
        expected_failure_class=first.failure_class,
        expected_fingerprint=first.fingerprint,
        expected_raw_sha256=_expected_raw_manifest(raw_path),
        repair_commit="a" * 40,
        reason="must remain fail closed",
    )

    assert release["accepted"] is False
    assert release["reason"] == "failure_state_raw_binding_invalid"


def test_verified_local_repair_releases_exact_operational_packet(
    isolated_wiki: Path,
    operational_release_case: tuple[Path, dict[str, str]],
) -> None:
    from chronovisor.ops import self_heal
    from chronovisor.raw import failure_supervisor

    packet_path, kwargs = operational_release_case
    raw_path = isolated_wiki / "raw" / "broken.md"
    second_raw = isolated_wiki / "raw" / "second.md"
    second_raw.write_text("second immutable source", encoding="utf-8")
    state_path = isolated_wiki / "runtime" / "failures" / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["failures"][second_raw.name] = dict(state["failures"][raw_path.name])
    state["failures"][second_raw.name]["raw_sha256"] = hashlib.sha256(
        second_raw.read_bytes()
    ).hexdigest()
    state["failures"][second_raw.name]["raw_bytes"] = len(second_raw.read_bytes())
    state_path.write_text(json.dumps(state), encoding="utf-8")
    kwargs["expected_raw_sha256"] = _expected_raw_manifest(raw_path, second_raw)
    assert failure_supervisor.operational_deferred_raw_files(
        [raw_path, second_raw]
    ) == {
        raw_path.name: "pending_local_repair",
        second_raw.name: "pending_local_repair",
    }
    result = self_heal.release_operational_failure_after_local_repair(
        packet_path,
        **kwargs,
        verification_command="pytest -q tests/test_ingest.py",
        verification_result="4 passed",
    )

    updated = json.loads(packet_path.read_text(encoding="utf-8"))
    assert result["accepted"] is True
    assert result["status"] == "local_repair_applied"
    assert result["cached"] is False
    assert updated["status"] == "local_repair_applied"
    assert updated["self_heal_queued"] is False
    assert updated["verified_local_repair"]["git_commit_sha"] == "a" * 40
    assert updated["verified_local_repair"]["failure_class"] == (
        "ingest.generation_context_window_exceeded"
    )
    assert updated["verified_local_repair"]["verification_command"] == (
        "pytest -q tests/test_ingest.py"
    )
    assert updated["verified_local_repair"]["verification_result"] == "4 passed"
    assert updated["verified_local_repair"]["affected_raw_scope"] == (
        "fingerprint_group"
    )
    assert updated["verified_local_repair"]["affected_raws"] == [
        {
            "filename": raw_path.name,
            "sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
            "bytes": len(raw_path.read_bytes()),
            "binding_source": "failure_state",
        },
        {
            "filename": second_raw.name,
            "sha256": hashlib.sha256(second_raw.read_bytes()).hexdigest(),
            "bytes": len(second_raw.read_bytes()),
            "binding_source": "failure_state",
        },
    ]
    assert (
        failure_supervisor.operational_deferred_raw_files([raw_path, second_raw]) == {}
    )


def test_verified_local_repair_releases_operational_local_quarantine(
    isolated_wiki: Path,
    operational_release_case: tuple[Path, dict[str, str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chronovisor.decision import local_repair
    from chronovisor.ops import self_heal
    from chronovisor.raw import failure_supervisor

    packet_path, kwargs = operational_release_case
    raw_path = isolated_wiki / "raw" / "broken.md"
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    _install_local_no_quorum_router(
        monkeypatch,
        cache_root=isolated_wiki / "runtime" / "structured-review-holds",
    )
    hold_decision = local_repair.propose_repair(
        {
            "failure_id": "strict-hold",
            "failure_class": "semantic.ambiguous_repair",
            "fingerprint": "semantic.ambiguous_repair:strict-hold",
            "error": "two safe repairs remain plausible",
        },
        use_qwen=True,
    )
    semantic_hold = hold_decision.semantic_hold
    assert semantic_hold is not None
    packet.update(
        {
            "status": "local_quarantined",
            "semantic_hold": semantic_hold,
            "terminal_reason": "semantic_no_quorum",
            "quarantined_at": "2026-07-15T10:28:08",
        }
    )
    packet_path.write_text(json.dumps(packet), encoding="utf-8")
    kwargs["expected_status"] = "local_quarantined"

    result = self_heal.release_operational_failure_after_local_repair(
        packet_path, **kwargs
    )

    updated = json.loads(packet_path.read_text(encoding="utf-8"))
    assert result["accepted"] is True
    assert result["status"] == "local_repair_applied"
    assert updated["status"] == "local_repair_applied"
    assert updated["semantic_hold"] is None
    assert updated["invalidated_semantic_hold"] == semantic_hold
    assert updated["semantic_hold_history"] == [semantic_hold]
    assert updated["terminal_reason"] is None
    assert updated["quarantined_at"] is None
    assert failure_supervisor.operational_deferred_raw_files([raw_path]) == {}
    assert packet_path not in self_heal.pending_packets()


def test_verified_local_repair_supersedes_unstarted_linked_incident_and_job(
    isolated_wiki: Path,
    operational_release_case: tuple[Path, dict[str, str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chronovisor.ops import background_jobs, self_heal, system_incident_supervisor

    packet_path, kwargs = operational_release_case
    source = json.loads(packet_path.read_text(encoding="utf-8"))
    source["status"] = "local_quarantined"
    packet_path.write_text(json.dumps(source), encoding="utf-8")
    kwargs["expected_status"] = "local_quarantined"
    incident_path = _link_operational_system_incident(isolated_wiki, packet_path)
    monkeypatch.setattr(
        system_incident_supervisor,
        "validate_operational_incident_packet",
        lambda path: {"status": "valid", "packet_path": str(path)},
    )
    args = ["--packet", str(incident_path), "--enable-frontier-repair"]
    job = background_jobs.enqueue_job(
        name="system-code-repair",
        module="chronovisor.ops.self_heal",
        args=args,
        env={},
        stdin_text="",
    )
    assert background_jobs._claim(job["job_id"])["status"] == "running"

    result = self_heal.release_operational_failure_after_local_repair(
        packet_path,
        **kwargs,
    )

    updated_source = json.loads(packet_path.read_text(encoding="utf-8"))
    updated_incident = json.loads(incident_path.read_text(encoding="utf-8"))
    job_state = json.loads(background_jobs.STATE_FILE.read_text(encoding="utf-8"))[
        "jobs"
    ][job["job_id"]]
    assert result["accepted"] is True
    assert updated_source["status"] == "local_repair_applied"
    assert updated_incident["status"] == (
        self_heal.VERIFIED_LOCAL_REPAIR_CANCELLATION_STATUS
    )
    assert updated_incident["superseded_by_packet"] == str(packet_path.resolve())
    assert result["linked_system_incident"]["background_job_cancellation"][
        "cancelled_job_ids"
    ] == [job["job_id"]]
    assert job_state["status"] == "cancelled"
    assert incident_path not in self_heal.pending_packets()
    stale_enqueue = self_heal.enqueue_system_code_repair(incident_path)
    assert stale_enqueue["status"] == (
        self_heal.VERIFIED_LOCAL_REPAIR_CANCELLATION_STATUS
    )
    assert stale_enqueue["enqueued"] is False
    assert stale_enqueue["cancelled"] is True
    refreshed_jobs = json.loads(
        background_jobs.STATE_FILE.read_text(encoding="utf-8")
    )["jobs"]
    assert not any(
        row.get("status") in background_jobs.ACTIVE_STATUSES
        for row in refreshed_jobs.values()
    )

    monkeypatch.setattr(
        self_heal,
        "_run_frontier",
        lambda *_args, **_kwargs: pytest.fail(
            "a snapshot worker must observe durable incident supersession"
        ),
    )
    dry_snapshot_worker = self_heal.handle_packet(
        incident_path,
        use_qwen=False,
        enable_frontier=True,
        dry_run=True,
    )
    snapshot_worker = self_heal.handle_packet(
        incident_path,
        use_qwen=False,
        enable_frontier=True,
    )
    finished = background_jobs._finish(job["job_id"], exit_code=0, output="cached")
    assert snapshot_worker["status"] == (
        self_heal.VERIFIED_LOCAL_REPAIR_CANCELLATION_STATUS
    )
    assert dry_snapshot_worker["projected_status"] == (
        self_heal.VERIFIED_LOCAL_REPAIR_CANCELLATION_STATUS
    )
    assert finished["status"] == "cancelled"


@pytest.mark.parametrize("path_value", ("missing", None))
def test_verified_local_repair_rejects_partial_link_metadata(
    isolated_wiki: Path,
    operational_release_case: tuple[Path, dict[str, str]],
    path_value: str | None,
) -> None:
    from chronovisor.ops import self_heal

    packet_path, kwargs = operational_release_case
    incident_path = _link_operational_system_incident(isolated_wiki, packet_path)
    source = json.loads(packet_path.read_text(encoding="utf-8"))
    if path_value == "missing":
        source.pop("system_incident_packet_path")
    else:
        source["system_incident_packet_path"] = None
    packet_path.write_text(json.dumps(source), encoding="utf-8")
    source_before = packet_path.read_bytes()

    result = self_heal.release_operational_failure_after_local_repair(
        packet_path,
        **kwargs,
    )

    assert result["accepted"] is False
    assert result["reason"] == "linked_system_incident_path_invalid"
    assert packet_path.read_bytes() == source_before
    assert json.loads(incident_path.read_text(encoding="utf-8"))["status"] == (
        "pending_frontier"
    )


def test_verified_local_repair_linked_incident_dry_run_is_read_only(
    isolated_wiki: Path,
    operational_release_case: tuple[Path, dict[str, str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chronovisor.ops import self_heal, system_incident_supervisor

    packet_path, kwargs = operational_release_case
    incident_path = _link_operational_system_incident(isolated_wiki, packet_path)
    monkeypatch.setattr(
        system_incident_supervisor,
        "validate_operational_incident_packet",
        lambda path: {"status": "valid", "packet_path": str(path)},
    )
    source_before = packet_path.read_bytes()
    incident_before = incident_path.read_bytes()

    result = self_heal.release_operational_failure_after_local_repair(
        packet_path,
        **kwargs,
        dry_run=True,
    )

    assert result["accepted"] is True
    assert result["linked_system_incident"]["would_supersede"] is True
    assert result["linked_system_incident"]["would_cancel_background_job"] is True
    assert packet_path.read_bytes() == source_before
    assert incident_path.read_bytes() == incident_before
    assert not (
        isolated_wiki / "runtime" / "failures" / "packet-cancellations"
    ).exists()
    assert not (isolated_wiki / "runtime" / "background-jobs").exists()


def test_verified_local_repair_resumes_after_linked_incident_cancel_commit(
    isolated_wiki: Path,
    operational_release_case: tuple[Path, dict[str, str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chronovisor.ops import self_heal, system_incident_supervisor

    packet_path, kwargs = operational_release_case
    incident_path = _link_operational_system_incident(isolated_wiki, packet_path)
    monkeypatch.setattr(
        system_incident_supervisor,
        "validate_operational_incident_packet",
        lambda path: {"status": "valid", "packet_path": str(path)},
    )
    source_before = packet_path.read_bytes()
    source = json.loads(source_before)

    prepared = self_heal._prepare_linked_incident_for_verified_release(
        packet_path.resolve(),
        source,
        dry_run=False,
    )

    assert prepared["accepted"] is True
    assert packet_path.read_bytes() == source_before
    assert json.loads(incident_path.read_text(encoding="utf-8"))["status"] == (
        self_heal.VERIFIED_LOCAL_REPAIR_CANCELLATION_STATUS
    )
    result = self_heal.release_operational_failure_after_local_repair(
        packet_path,
        **kwargs,
    )
    assert result["accepted"] is True
    assert result["linked_system_incident"]["state"] == "superseded"
    assert json.loads(packet_path.read_text(encoding="utf-8"))["status"] == (
        "local_repair_applied"
    )


def test_verified_local_repair_retries_after_background_cancel_failure(
    isolated_wiki: Path,
    operational_release_case: tuple[Path, dict[str, str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chronovisor.ops import background_jobs, self_heal, system_incident_supervisor

    packet_path, kwargs = operational_release_case
    incident_path = _link_operational_system_incident(isolated_wiki, packet_path)
    monkeypatch.setattr(
        system_incident_supervisor,
        "validate_operational_incident_packet",
        lambda path: {"status": "valid", "packet_path": str(path)},
    )
    source_before = packet_path.read_bytes()
    original_cancel = background_jobs.cancel_matching_jobs
    monkeypatch.setattr(
        background_jobs,
        "cancel_matching_jobs",
        lambda **_kwargs: (_ for _ in ()).throw(OSError("ledger unavailable")),
    )

    failed = self_heal.release_operational_failure_after_local_repair(
        packet_path,
        **kwargs,
    )

    assert failed["accepted"] is False
    assert failed["reason"] == "linked_system_incident_job_cancel_failed"
    assert packet_path.read_bytes() == source_before
    assert json.loads(incident_path.read_text(encoding="utf-8"))["status"] == (
        self_heal.VERIFIED_LOCAL_REPAIR_CANCELLATION_STATUS
    )
    monkeypatch.setattr(background_jobs, "cancel_matching_jobs", original_cancel)
    recovered = self_heal.release_operational_failure_after_local_repair(
        packet_path,
        **kwargs,
    )
    assert recovered["accepted"] is True
    assert json.loads(packet_path.read_text(encoding="utf-8"))["status"] == (
        "local_repair_applied"
    )


def test_verified_local_repair_with_real_incident_binding_is_idempotent(
    isolated_wiki: Path,
    operational_release_case: tuple[Path, dict[str, str]],
) -> None:
    from chronovisor.ops import self_heal

    packet_path, kwargs = operational_release_case
    source = json.loads(packet_path.read_text(encoding="utf-8"))
    source.update(
        {
            "status": "local_quarantined",
            "local_repair_attempts": 2,
            "operational_local_repair_evidence": ["a" * 64, "b" * 64],
        }
    )
    kwargs["expected_status"] = "local_quarantined"
    supervisor_root = isolated_wiki / "runtime" / "system-incidents"
    artifact = supervisor_root / "reproduction-artifacts" / "failure.json"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_bytes(b'{"status":"failed"}\n')
    receipt = supervisor_root / "reproduction-receipts" / "failure.json"
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "producer": "trusted_system_incident_supervisor",
                "outcome": "reproducibly_failed",
                "source_packet_path": str(packet_path.resolve()),
                "source_failure_class": source["failure_class"],
                "source_fingerprint": source["fingerprint"],
                "artifact": str(artifact.resolve()),
                "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                "failing_test": "tests/test_ingest.py::test_schema_contract",
            }
        ),
        encoding="utf-8",
    )
    source["deterministic_reproduction_receipt"] = str(receipt.resolve())
    packet_path.write_text(json.dumps(source), encoding="utf-8")

    promoted = self_heal._promote_operational_source_packet(packet_path, source)

    assert promoted is not None
    assert promoted["status"] == "packet_created"
    incident_path = Path(str(promoted["packet_path"]))
    tree_before_dry_run = {
        str(path.relative_to(isolated_wiki)): path.read_bytes()
        for path in isolated_wiki.rglob("*")
        if path.is_file()
    }
    dry_run = self_heal.release_operational_failure_after_local_repair(
        packet_path,
        **kwargs,
        dry_run=True,
    )
    tree_after_dry_run = {
        str(path.relative_to(isolated_wiki)): path.read_bytes()
        for path in isolated_wiki.rglob("*")
        if path.is_file()
    }
    assert dry_run["accepted"] is True
    assert dry_run["linked_system_incident"]["would_supersede"] is True
    assert tree_after_dry_run == tree_before_dry_run

    applied = self_heal.release_operational_failure_after_local_repair(
        packet_path,
        **kwargs,
    )
    source_after = packet_path.read_bytes()
    incident_after = incident_path.read_bytes()
    background_state_after = (
        isolated_wiki / "runtime" / "background-jobs" / "state.json"
    ).read_bytes()
    cached = self_heal.release_operational_failure_after_local_repair(
        packet_path,
        **kwargs,
    )

    assert applied["accepted"] is True
    assert cached["accepted"] is True
    assert cached["cached"] is True
    assert packet_path.read_bytes() == source_after
    assert incident_path.read_bytes() == incident_after
    assert (
        isolated_wiki / "runtime" / "background-jobs" / "state.json"
    ).read_bytes() == background_state_after
    stale_enqueue = self_heal.enqueue_system_code_repair(incident_path)
    assert stale_enqueue["enqueued"] is False
    assert stale_enqueue["cancelled"] is True


@pytest.mark.parametrize(
    ("status", "frontier_attempts", "execution_started", "reason"),
    [
        ("frontier_running", 0, False, "linked_system_incident_already_started"),
        ("pending_frontier", 1, False, "linked_system_incident_already_started"),
        ("pending_frontier", 0, True, "linked_system_incident_already_started"),
        ("frontier_rejected", 0, False, "linked_system_incident_terminal"),
        ("human_required", 0, False, "linked_system_incident_terminal"),
    ],
)
def test_verified_local_repair_refuses_started_or_terminal_linked_incident(
    isolated_wiki: Path,
    operational_release_case: tuple[Path, dict[str, str]],
    monkeypatch: pytest.MonkeyPatch,
    status: str,
    frontier_attempts: int,
    execution_started: bool,
    reason: str,
) -> None:
    from chronovisor.ops import self_heal, system_incident_supervisor

    packet_path, kwargs = operational_release_case
    incident_path = _link_operational_system_incident(
        isolated_wiki,
        packet_path,
        status=status,
        frontier_attempts=frontier_attempts,
        execution_started=execution_started,
    )
    monkeypatch.setattr(
        system_incident_supervisor,
        "validate_operational_incident_packet",
        lambda path: {"status": "valid", "packet_path": str(path)},
    )
    source_before = packet_path.read_bytes()
    incident_before = incident_path.read_bytes()

    result = self_heal.release_operational_failure_after_local_repair(
        packet_path,
        **kwargs,
    )

    assert result["accepted"] is False
    assert result["reason"] == reason
    assert packet_path.read_bytes() == source_before
    assert incident_path.read_bytes() == incident_before


def test_verified_local_repair_refuses_busy_linked_incident(
    isolated_wiki: Path,
    operational_release_case: tuple[Path, dict[str, str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chronovisor.ops import self_heal, system_incident_supervisor

    packet_path, kwargs = operational_release_case
    incident_path = _link_operational_system_incident(isolated_wiki, packet_path)
    monkeypatch.setattr(
        system_incident_supervisor,
        "validate_operational_incident_packet",
        lambda path: {"status": "valid", "packet_path": str(path)},
    )
    source_before = packet_path.read_bytes()
    incident_before = incident_path.read_bytes()

    with self_heal._packet_lock(incident_path) as acquired:
        assert acquired is True
        result = self_heal.release_operational_failure_after_local_repair(
            packet_path,
            **kwargs,
        )

    assert result["accepted"] is False
    assert result["reason"] == "linked_system_incident_busy"
    assert packet_path.read_bytes() == source_before
    assert incident_path.read_bytes() == incident_before


def test_verified_local_repair_accepts_remaining_partial_group(
    isolated_wiki: Path,
    operational_release_case: tuple[Path, dict[str, str]],
) -> None:
    from chronovisor.ops import self_heal

    packet_path, kwargs = operational_release_case
    remaining = isolated_wiki / "raw" / "remaining.md"
    remaining.write_text("remaining immutable source", encoding="utf-8")
    state_path = isolated_wiki / "runtime" / "failures" / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    original = state["failures"].pop("broken.md")
    original.pop("raw_sha256", None)
    original.pop("raw_bytes", None)
    state["failures"][remaining.name] = original
    state_path.write_text(json.dumps(state), encoding="utf-8")
    kwargs["expected_raw_sha256"] = _expected_raw_manifest(remaining)

    result = self_heal.release_operational_failure_after_local_repair(
        packet_path, **kwargs
    )

    assert result["accepted"] is True
    assert result["verified_local_repair"]["packet_raw_file"] == "broken.md"
    assert result["verified_local_repair"]["affected_raws"] == [
        {
            "filename": remaining.name,
            "sha256": hashlib.sha256(remaining.read_bytes()).hexdigest(),
            "bytes": len(remaining.read_bytes()),
            "binding_source": "expected_manifest_legacy",
        }
    ]
    assert result["verified_local_repair"]["legacy_state_raws"] == [remaining.name]


def test_verified_local_repair_refuses_expected_status_mismatch(
    operational_release_case: tuple[Path, dict[str, str]],
) -> None:
    from chronovisor.ops import self_heal

    packet_path, kwargs = operational_release_case
    kwargs["expected_status"] = "frontier_retry"

    result = self_heal.release_operational_failure_after_local_repair(
        packet_path, **kwargs
    )

    assert result["accepted"] is False
    assert result["reason"] == "packet_status_mismatch"
    assert result["observed_status"] == "pending_local_repair"


def test_verified_local_repair_refuses_incomplete_group_manifest(
    isolated_wiki: Path,
    operational_release_case: tuple[Path, dict[str, str]],
) -> None:
    from chronovisor.ops import self_heal

    packet_path, kwargs = operational_release_case
    second = isolated_wiki / "raw" / "second.md"
    second.write_text("second source", encoding="utf-8")
    state_path = isolated_wiki / "runtime" / "failures" / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    second_entry = dict(state["failures"]["broken.md"])
    second_entry["raw_sha256"] = hashlib.sha256(second.read_bytes()).hexdigest()
    second_entry["raw_bytes"] = len(second.read_bytes())
    state["failures"][second.name] = second_entry
    state_path.write_text(json.dumps(state), encoding="utf-8")

    result = self_heal.release_operational_failure_after_local_repair(
        packet_path, **kwargs
    )

    assert result["accepted"] is False
    assert result["reason"] == "failure_state_group_manifest_mismatch"
    assert result["observed_raw_files"] == ["broken.md", "second.md"]


def test_verified_local_repair_refuses_raw_changed_since_failure(
    isolated_wiki: Path,
    operational_release_case: tuple[Path, dict[str, str]],
) -> None:
    from chronovisor.ops import self_heal

    packet_path, kwargs = operational_release_case
    raw_path = isolated_wiki / "raw" / "broken.md"
    raw_path.write_text("mutated source", encoding="utf-8")

    stale_expected = self_heal.release_operational_failure_after_local_repair(
        packet_path, **kwargs
    )
    assert stale_expected["reason"] == "expected_raw_sha256_mismatch"

    kwargs["expected_raw_sha256"] = _expected_raw_manifest(raw_path)
    state_mismatch = self_heal.release_operational_failure_after_local_repair(
        packet_path, **kwargs
    )
    assert state_mismatch["reason"] == "failure_state_raw_binding_mismatch"


def test_verified_local_repair_refuses_cancelled_packet(
    isolated_wiki: Path,
    operational_release_case: tuple[Path, dict[str, str]],
) -> None:
    from chronovisor.ops import self_heal

    packet_path, kwargs = operational_release_case
    before = packet_path.read_bytes()
    cancellation = self_heal.request_packet_cancellation(
        packet_path,
        reason="semantic defer superseded this repair",
        superseded_by_packet=(
            isolated_wiki / "runtime" / "failures" / "packets" / "semantic.json"
        ),
    )
    assert cancellation["accepted"] is True

    result = self_heal.release_operational_failure_after_local_repair(
        packet_path, **kwargs
    )

    assert result["accepted"] is False
    assert result["reason"] == "packet_cancellation_requested"
    assert packet_path.read_bytes() == before


def test_verified_local_repair_late_cancellation_wins_before_packet_commit(
    isolated_wiki: Path,
    operational_release_case: tuple[Path, dict[str, str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chronovisor.ops import self_heal

    packet_path, kwargs = operational_release_case

    def cancel_during_verification(commit: str) -> dict[str, str]:
        cancellation = self_heal.request_packet_cancellation(
            packet_path,
            reason="semantic defer superseded repair during verification",
            superseded_by_packet=(
                isolated_wiki
                / "runtime"
                / "failures"
                / "packets"
                / "semantic-late.json"
            ),
        )
        assert cancellation["accepted"] is True
        return _verified_git_state(commit)

    monkeypatch.setattr(
        self_heal,
        "_verified_local_repair_git_state",
        cancel_during_verification,
    )

    result = self_heal.release_operational_failure_after_local_repair(
        packet_path, **kwargs
    )

    assert result["accepted"] is False
    assert result["reason"] == "packet_cancellation_requested"
    updated = json.loads(packet_path.read_text(encoding="utf-8"))
    assert updated["status"] == self_heal.PACKET_CANCELLATION_STATUS
    assert "verified_local_repair" not in updated


@pytest.mark.parametrize(
    ("expected_class", "expected_fingerprint", "reason"),
    [
        (
            "ingest.generation_output_truncated",
            "ingest.generation_context_window_exceeded",
            "failure_class_mismatch",
        ),
        (
            "ingest.generation_context_window_exceeded",
            "ingest.generation_context_window_exceeded:other",
            "fingerprint_mismatch",
        ),
    ],
)
def test_verified_local_repair_refuses_stale_packet_identity(
    isolated_wiki: Path,
    monkeypatch: pytest.MonkeyPatch,
    expected_class: str,
    expected_fingerprint: str,
    reason: str,
) -> None:
    from chronovisor.ops import self_heal

    packet_path = _write_operational_packet(isolated_wiki)
    before = packet_path.read_bytes()
    monkeypatch.setattr(
        self_heal,
        "_verified_local_repair_git_state",
        lambda _expected: pytest.fail("identity mismatch must fail before git ACK"),
    )

    result = self_heal.release_operational_failure_after_local_repair(
        packet_path,
        expected_status="pending_local_repair",
        expected_failure_class=expected_class,
        expected_fingerprint=expected_fingerprint,
        expected_raw_sha256={"broken.md": "c" * 64},
        repair_commit="a" * 40,
        reason="claimed repair",
    )

    assert result["accepted"] is False
    assert result["reason"] == reason
    assert packet_path.read_bytes() == before


def test_verified_local_repair_never_releases_semantic_defer(
    isolated_wiki: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chronovisor.ops import self_heal

    packet_path = _write_operational_packet(isolated_wiki)
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    packet.update(
        {
            "failure_class": "ingest.semantic_no_quorum",
            "fingerprint": f"ingest.semantic_no_quorum:{'b' * 64}",
            "status": "local_quarantined",
            "terminal_deferred": True,
            "defer_reason": "semantic_no_quorum",
        }
    )
    packet_path.write_text(json.dumps(packet), encoding="utf-8")
    before = packet_path.read_bytes()
    monkeypatch.setattr(
        self_heal,
        "_verified_local_repair_git_state",
        lambda _expected: pytest.fail("semantic packet must fail before git ACK"),
    )

    result = self_heal.release_operational_failure_after_local_repair(
        packet_path,
        expected_status="local_quarantined",
        expected_failure_class="ingest.semantic_no_quorum",
        expected_fingerprint=f"ingest.semantic_no_quorum:{'b' * 64}",
        expected_raw_sha256={"broken.md": "c" * 64},
        repair_commit="a" * 40,
        reason="must not apply",
    )

    assert result["accepted"] is False
    assert result["reason"] == "semantic_defer_not_releasable"
    assert packet_path.read_bytes() == before


def test_verified_local_repair_is_idempotent_and_packet_locked(
    operational_release_case: tuple[Path, dict[str, str]],
) -> None:
    from chronovisor.ops import self_heal

    packet_path, kwargs = operational_release_case

    with self_heal._packet_lock(packet_path) as acquired:
        assert acquired is True
        busy = self_heal.release_operational_failure_after_local_repair(
            packet_path, **kwargs
        )
    assert busy["accepted"] is False
    assert busy["reason"] == "packet_already_running"

    first = self_heal.release_operational_failure_after_local_repair(
        packet_path, **kwargs
    )
    after_first = packet_path.read_bytes()
    second = self_heal.release_operational_failure_after_local_repair(
        packet_path, **kwargs
    )

    assert first["accepted"] is True
    assert first["cached"] is False
    assert second["accepted"] is True
    assert second["cached"] is True
    assert packet_path.read_bytes() == after_first


def test_verified_local_repair_cached_retry_survives_partial_and_full_state_cleanup(
    isolated_wiki: Path,
    operational_release_case: tuple[Path, dict[str, str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chronovisor.ops import self_heal
    from chronovisor.raw import failure_supervisor

    packet_path, kwargs = operational_release_case
    first_raw = isolated_wiki / "raw" / "broken.md"
    second_raw = isolated_wiki / "raw" / "second.md"
    second_raw.write_text("second immutable source", encoding="utf-8")
    state_path = isolated_wiki / "runtime" / "failures" / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    second_entry = dict(state["failures"][first_raw.name])
    second_entry["raw_sha256"] = hashlib.sha256(second_raw.read_bytes()).hexdigest()
    second_entry["raw_bytes"] = len(second_raw.read_bytes())
    state["failures"][second_raw.name] = second_entry
    state_path.write_text(json.dumps(state), encoding="utf-8")
    kwargs["expected_raw_sha256"] = _expected_raw_manifest(first_raw, second_raw)

    first = self_heal.release_operational_failure_after_local_repair(
        packet_path, **kwargs
    )
    assert first["accepted"] is True
    assert first["cached"] is False
    packet_after_first = packet_path.read_bytes()
    monkeypatch.setattr(
        self_heal,
        "_verified_local_repair_git_state",
        lambda _commit: pytest.fail("cached retry must not revalidate current git"),
    )

    failure_supervisor.reset_raw_failure(first_raw.name)
    state_after_partial_cleanup = state_path.read_bytes()
    partial = self_heal.release_operational_failure_after_local_repair(
        packet_path, **kwargs
    )

    assert partial["accepted"] is True
    assert partial["cached"] is True
    assert packet_path.read_bytes() == packet_after_first
    assert state_path.read_bytes() == state_after_partial_cleanup

    failure_supervisor.reset_raw_failure(second_raw.name)
    state_after_full_cleanup = state_path.read_bytes()
    complete = self_heal.release_operational_failure_after_local_repair(
        packet_path, **kwargs
    )

    assert complete["accepted"] is True
    assert complete["cached"] is True
    assert packet_path.read_bytes() == packet_after_first
    assert state_path.read_bytes() == state_after_full_cleanup


def test_verified_local_repair_cached_retry_refuses_different_receipt_evidence(
    isolated_wiki: Path,
    operational_release_case: tuple[Path, dict[str, str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chronovisor.ops import self_heal
    from chronovisor.raw import failure_supervisor

    packet_path, fixture_kwargs = operational_release_case
    raw_path = isolated_wiki / "raw" / "broken.md"
    baseline: dict[str, object] = {
        **fixture_kwargs,
        "verification_command": "pytest -q focused",
        "verification_result": "passed",
    }
    first = self_heal.release_operational_failure_after_local_repair(
        packet_path, **baseline
    )
    assert first["accepted"] is True
    failure_supervisor.reset_raw_failure(raw_path.name)
    packet_after_first = packet_path.read_bytes()
    state_path = isolated_wiki / "runtime" / "failures" / "state.json"
    state_after_cleanup = state_path.read_bytes()
    monkeypatch.setattr(
        self_heal,
        "_verified_local_repair_git_state",
        lambda _commit: pytest.fail("completed retry must not inspect current git"),
    )

    changed_inputs = [
        {"expected_status": "frontier_retry"},
        {"expected_raw_sha256": {raw_path.name: "b" * 64}},
        {"repair_commit": "b" * 40},
        {"reason": "different repair"},
        {"verification_command": "pytest -q different"},
        {"verification_result": "failed"},
    ]
    for changed in changed_inputs:
        retry = dict(baseline)
        retry.update(changed)
        result = self_heal.release_operational_failure_after_local_repair(
            packet_path, **retry
        )
        assert result["accepted"] is False
        assert result["reason"] == "completed_packet_repair_evidence_mismatch"
        assert packet_path.read_bytes() == packet_after_first
        assert state_path.read_bytes() == state_after_cleanup

    class_mismatch = dict(baseline)
    class_mismatch["expected_failure_class"] = "ingest.generation_output_truncated"
    assert (
        self_heal.release_operational_failure_after_local_repair(
            packet_path, **class_mismatch
        )["reason"]
        == "failure_class_mismatch"
    )
    fingerprint_mismatch = dict(baseline)
    fingerprint_mismatch["expected_fingerprint"] = "different:fingerprint"
    assert (
        self_heal.release_operational_failure_after_local_repair(
            packet_path, **fingerprint_mismatch
        )["reason"]
        == "fingerprint_mismatch"
    )


def test_verified_local_repair_freezes_group_until_packet_commit(
    isolated_wiki: Path,
    operational_release_case: tuple[Path, dict[str, str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chronovisor.ops import self_heal
    from chronovisor.raw import failure_supervisor

    packet_path, kwargs = operational_release_case
    late_raw = isolated_wiki / "raw" / "late.md"
    late_raw.write_text("late immutable source", encoding="utf-8")
    started = threading.Event()
    finished = threading.Event()
    attached: dict[str, object] = {}
    worker: list[threading.Thread] = []
    original_update = self_heal._update_packet
    monkeypatch.setattr(failure_supervisor, "_launch_self_heal", lambda _path: None)

    def attach_late_raw() -> None:
        started.set()
        attached["result"] = failure_supervisor.record_raw_failure(
            raw_path=late_raw,
            error=(
                "ingest generation context_window_exceeded: required context "
                "268078 exceeds configured max_num_ctx 262144"
            ),
            raw_text=late_raw.read_text(encoding="utf-8"),
        )
        finished.set()

    def update_packet(path: Path, packet: dict, **updates: object) -> None:
        thread = threading.Thread(target=attach_late_raw)
        worker.append(thread)
        thread.start()
        assert started.wait(timeout=1)
        assert not finished.wait(timeout=0.05)
        original_update(path, packet, **updates)

    monkeypatch.setattr(self_heal, "_update_packet", update_packet)

    result = self_heal.release_operational_failure_after_local_repair(
        packet_path, **kwargs
    )
    worker[0].join(timeout=2)

    assert finished.is_set()
    assert [
        row["filename"] for row in result["verified_local_repair"]["affected_raws"]
    ] == ["broken.md"]
    assert attached["result"].packet_path != str(packet_path)


def test_operational_release_lock_order_does_not_block_state_writer(
    isolated_wiki: Path,
    operational_release_case: tuple[Path, dict[str, str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chronovisor.ops import self_heal
    from chronovisor.raw import failure_supervisor

    packet_path, kwargs = operational_release_case
    late_raw = isolated_wiki / "raw" / "late.md"
    late_raw.write_text("late immutable source", encoding="utf-8")
    held = threading.Event()
    release_holder = threading.Event()
    writer_done = threading.Event()
    writer_result: dict[str, object] = {}
    monkeypatch.setattr(failure_supervisor, "_launch_self_heal", lambda _path: None)

    def hold_packet() -> None:
        with self_heal._packet_lock(packet_path) as acquired:
            assert acquired is True
            held.set()
            assert release_holder.wait(timeout=2)

    def write_state() -> None:
        writer_result["value"] = failure_supervisor.record_raw_failure(
            raw_path=late_raw,
            error=(
                "ingest generation context_window_exceeded: required context "
                "268078 exceeds configured max_num_ctx 262144"
            ),
            raw_text=late_raw.read_text(encoding="utf-8"),
        )
        writer_done.set()

    holder = threading.Thread(target=hold_packet)
    holder.start()
    assert held.wait(timeout=1)
    writer = threading.Thread(target=write_state)
    writer.start()
    try:
        assert writer_done.wait(timeout=1)
        busy = self_heal.release_operational_failure_after_local_repair(
            packet_path, **kwargs
        )
    finally:
        release_holder.set()
        holder.join(timeout=2)
        writer.join(timeout=2)

    assert busy["reason"] == "packet_already_running"
    assert writer_result["value"].packet_path == str(packet_path)
    kwargs["expected_raw_sha256"] = _expected_raw_manifest(
        isolated_wiki / "raw" / "broken.md",
        late_raw,
    )
    released = self_heal.release_operational_failure_after_local_repair(
        packet_path, **kwargs
    )
    assert [
        row["filename"] for row in released["verified_local_repair"]["affected_raws"]
    ] == ["broken.md", "late.md"]


def test_verified_local_repair_dry_run_is_byte_for_byte_read_only(
    isolated_wiki: Path,
    operational_release_case: tuple[Path, dict[str, str]],
) -> None:
    from chronovisor.ops import self_heal

    packet_path, kwargs = operational_release_case
    before = packet_path.read_bytes()

    result = self_heal.release_operational_failure_after_local_repair(
        packet_path,
        **kwargs,
        dry_run=True,
    )

    assert result["accepted"] is True
    assert result["status"] == "dry_run"
    assert result["projected_status"] == "local_repair_applied"
    assert packet_path.read_bytes() == before
    assert not (isolated_wiki / "runtime" / "failures" / "locks").exists()


@pytest.mark.parametrize("python_directory", ("python3.13", "python3.14"))
def test_verified_local_repair_git_state_binds_clean_pushed_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    python_directory: str,
) -> None:
    from chronovisor.core import runtime_config
    from chronovisor.ops import self_heal

    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=repo,
            text=True,
            capture_output=True,
            check=True,
        )
        return result.stdout.strip()

    git("init", "-q")
    git("config", "user.name", "Test")
    git("config", "user.email", "test@example.com")
    tracked = repo / "repair.py"
    tracked.write_text("fixed = True\n", encoding="utf-8")
    git("add", "repair.py")
    git("commit", "-qm", "repair")
    commit = git("rev-parse", "HEAD")
    git("update-ref", "refs/remotes/origin/main", commit)
    monkeypatch.setattr(self_heal, "_repo_root", lambda: repo)
    archive_root = tmp_path / "cache" / "archive-v0" / "runtime-id"
    archive_path = archive_root / "lib" / python_directory
    module_path = archive_path / "site-packages" / "chronovisor" / "runtime_config.py"
    module_path.parent.mkdir(parents=True)
    module_path.write_text("# installed runtime\n", encoding="utf-8")

    def identity(**overrides: object) -> dict[str, object]:
        value: dict[str, object] = {
            "commit_id": commit,
            "expected_commit": commit,
            "drift": False,
            "archive_path": str(archive_path),
            "module_path": str(module_path),
            "runtime_source": ("git+ssh://git@github.com/trafficsign/chronovisor"),
            "direct_url": {
                "url": "ssh://git@github.com/trafficsign/chronovisor",
                "vcs_info": {"vcs": "git", "commit_id": commit},
            },
        }
        value.update(overrides)
        return value

    monkeypatch.setattr(
        runtime_config,
        "runtime_identity",
        identity,
    )

    evidence = self_heal._verified_local_repair_git_state(commit)

    assert evidence["git_commit_sha"] == commit
    assert evidence["checkout_head_sha"] == commit
    assert evidence["origin_main_sha"] == commit
    assert evidence["runtime_commit_sha"] == commit
    assert evidence["runtime_archive_root"] == str(archive_root)
    assert evidence["runtime_drift"] is False

    monkeypatch.setattr(
        runtime_config, "runtime_identity", lambda: identity(commit_id=None)
    )
    with pytest.raises(ValueError, match="repair_runtime_commit_unavailable"):
        self_heal._verified_local_repair_git_state(commit)
    monkeypatch.setattr(
        runtime_config,
        "runtime_identity",
        lambda: identity(commit_id="b" * 40),
    )
    with pytest.raises(ValueError, match="repair_commit_not_executing_runtime"):
        self_heal._verified_local_repair_git_state(commit)

    monkeypatch.setattr(
        runtime_config,
        "runtime_identity",
        lambda: identity(expected_commit="b" * 40),
    )
    with pytest.raises(ValueError, match="repair_runtime_expected_commit_mismatch"):
        self_heal._verified_local_repair_git_state(commit)
    monkeypatch.setattr(
        runtime_config,
        "runtime_identity",
        lambda: identity(drift=True),
    )
    with pytest.raises(ValueError, match="repair_runtime_drift_not_false"):
        self_heal._verified_local_repair_git_state(commit)
    monkeypatch.setattr(
        runtime_config,
        "runtime_identity",
        lambda: identity(archive_path=str(repo), module_path=str(tracked)),
    )
    with pytest.raises(ValueError, match="repair_runtime_not_uv_archive"):
        self_heal._verified_local_repair_git_state(commit)
    monkeypatch.setattr(
        runtime_config,
        "runtime_identity",
        lambda: identity(
            direct_url={
                "url": "ssh://git@github.com/attacker/chronovisor",
                "vcs_info": {"vcs": "git", "commit_id": commit},
            }
        ),
    )
    with pytest.raises(ValueError, match="repair_runtime_source_not_exact_github_vcs"):
        self_heal._verified_local_repair_git_state(commit)
    monkeypatch.setattr(
        runtime_config,
        "runtime_identity",
        identity,
    )

    tracked.write_text("fixed = False\n", encoding="utf-8")
    with pytest.raises(
        ValueError, match="repair_checkout_has_uncommitted_tracked_changes"
    ):
        self_heal._verified_local_repair_git_state(commit)


def test_release_operational_repair_cli_routes_all_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from chronovisor.ops import self_heal

    packet_path = tmp_path / "packet.json"
    packet_path.write_text("{}", encoding="utf-8")
    seen: dict[str, object] = {}

    def release(path: Path, **kwargs: object) -> dict[str, object]:
        seen["path"] = path
        seen.update(kwargs)
        return {"status": "dry_run", "accepted": True}

    monkeypatch.setattr(
        self_heal,
        "release_operational_failure_after_local_repair",
        release,
    )

    exit_code = self_heal.main(
        [
            "--release-operational-repair",
            str(packet_path),
            "--expected-status",
            "pending_local_repair",
            "--expected-failure-class",
            "ingest.generation_context_window_exceeded",
            "--expected-fingerprint",
            "ingest.generation_context_window_exceeded",
            "--expected-raw-sha256",
            f"broken.md={'b' * 64}",
            "--expected-raw-sha256",
            f"second.md={'c' * 64}",
            "--repair-commit",
            "a" * 40,
            "--repair-reason",
            "adaptive output budget",
            "--verification-command",
            "pytest focused",
            "--verification-result",
            "passed",
            "--dry-run",
        ]
    )

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out)["accepted"] is True
    assert seen == {
        "path": packet_path,
        "expected_status": "pending_local_repair",
        "expected_failure_class": "ingest.generation_context_window_exceeded",
        "expected_fingerprint": "ingest.generation_context_window_exceeded",
        "expected_raw_sha256": [
            f"broken.md={'b' * 64}",
            f"second.md={'c' * 64}",
        ],
        "repair_commit": "a" * 40,
        "reason": "adaptive output budget",
        "verification_command": "pytest focused",
        "verification_result": "passed",
        "dry_run": True,
    }


def test_release_operational_repair_cli_releases_legacy_state_with_manifest(
    isolated_wiki: Path,
    operational_release_case: tuple[Path, dict[str, str]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    from chronovisor.ops import self_heal

    packet_path, kwargs = operational_release_case
    raw_path = isolated_wiki / "raw" / "broken.md"
    state_path = isolated_wiki / "runtime" / "failures" / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state_entry = state["failures"][raw_path.name]
    state_entry.pop("raw_sha256")
    state_entry.pop("raw_bytes")
    state_path.write_text(json.dumps(state), encoding="utf-8")
    digest = hashlib.sha256(raw_path.read_bytes()).hexdigest()

    exit_code = self_heal.main(
        [
            "--release-operational-repair",
            str(packet_path),
            "--expected-status",
            str(kwargs["expected_status"]),
            "--expected-failure-class",
            str(kwargs["expected_failure_class"]),
            "--expected-fingerprint",
            str(kwargs["expected_fingerprint"]),
            "--expected-raw-sha256",
            f"{raw_path.name}={digest}",
            "--repair-commit",
            str(kwargs["repair_commit"]),
            "--repair-reason",
            str(kwargs["reason"]),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["accepted"] is True
    assert payload["verified_local_repair"]["legacy_state_raws"] == [raw_path.name]
    assert payload["verified_local_repair"]["affected_raws"] == [
        {
            "filename": raw_path.name,
            "sha256": digest,
            "bytes": len(raw_path.read_bytes()),
            "binding_source": "expected_manifest_legacy",
        }
    ]
    assert json.loads(packet_path.read_text(encoding="utf-8"))["status"] == (
        "local_repair_applied"
    )


@pytest.mark.parametrize(
    "conflict",
    [
        ["--packet", "/tmp/other.json"],
        ["--sandbox-drill"],
        ["--drill"],
        ["--auto-apply-errors"],
    ],
)
def test_release_operational_repair_cli_refuses_other_actions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    conflict: list[str],
) -> None:
    from chronovisor.ops import self_heal

    packet_path = tmp_path / "packet.json"
    packet_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        self_heal,
        "release_operational_failure_after_local_repair",
        lambda *_args, **_kwargs: pytest.fail("conflicting action must not release"),
    )

    exit_code = self_heal.main(
        ["--release-operational-repair", str(packet_path), *conflict]
    )

    assert exit_code == 2
    assert json.loads(capsys.readouterr().out)["reason"] == (
        "release_operational_repair_action_conflict"
    )


@pytest.mark.parametrize(
    "evidence_flag",
    [
        "--expected-status",
        "--expected-failure-class",
        "--expected-fingerprint",
        "--expected-raw-sha256",
        "--repair-commit",
        "--repair-reason",
        "--verification-command",
        "--verification-result",
    ],
)
def test_release_evidence_flags_require_release_action(
    capsys: pytest.CaptureFixture[str],
    evidence_flag: str,
) -> None:
    from chronovisor.ops import self_heal

    exit_code = self_heal.main([evidence_flag, "unexpected"])

    assert exit_code == 2
    assert json.loads(capsys.readouterr().out)["reason"] == (
        "release_operational_repair_evidence_without_action"
    )


def test_completed_packet_is_cached_instead_of_reprocessed(
    isolated_wiki: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chronovisor.ops import self_heal

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
    from chronovisor.ops import background_jobs, self_heal

    packet = tmp_path / "packet.json"
    packet.write_text("{}", encoding="utf-8")
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setenv("CHRONOVISOR_SELF_HEAL_AUTORUN", "1")
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
    assert seen["module"] == "chronovisor.ops.self_heal"
    assert seen["args"] == ["--packet", str(packet.resolve())]


def test_background_exit_code_preserves_retry_and_terminal_states() -> None:
    from chronovisor.ops import background_jobs, self_heal

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
