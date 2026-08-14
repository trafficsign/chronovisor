from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from chronovisor.core.runtime_config import SearchEmbeddingConfig
from chronovisor.ops import autonomy, health
from tests.semantic_hold_support import semantic_authority


def test_recall_distillation_kpi_uses_public_snapshot_without_raw_content(
    tmp_path: Path, monkeypatch
) -> None:
    chronovisor_root = tmp_path / "wiki"
    snapshot_module = SimpleNamespace(
        snapshot=lambda _root: {
            "schema": 1,
            "status": "running",
            "state": "backfill",
            "rollout": 5.0,
            "state_sha256": "a" * 64,
            "active_policy_id": "active-policy",
            "candidate_policy_id": "candidate-policy",
            "lkg_policy_id": "lkg-policy",
            "feature_revision": "recall-distill-text-v2",
            "teacher_only": 12,
            "verified_truth": 3,
            "probe_not_truth": 4,
            "paired_denominator": 7,
            "hold_reason": "verified_truth_below_floor",
        }
    )
    monkeypatch.setattr(health, "CHRONOVISOR_ROOT", chronovisor_root)
    monkeypatch.setitem(
        sys.modules,
        "chronovisor.recall.recall_distillation_store",
        snapshot_module,
    )

    payload = health.recall_distillation_kpi()

    assert payload["worker_status"] == "backfill"
    assert payload["rollout_percent"] == 5.0
    assert payload["active_policy_id"] == "active-policy"
    assert payload["candidate_policy_id"] == "candidate-policy"
    assert payload["lkg_policy_id"] == "lkg-policy"
    assert payload["teacher_only"] == 12
    assert payload["verified_truth"] == 3
    assert payload["probe_not_truth"] == 4
    assert payload["paired_denominator"] == 7
    assert payload["hold_reason"] == "verified_truth_below_floor"
    assert payload["feature_revision"] == "recall-distill-text-v2"
    assert payload["alert"] is False


def test_recall_distillation_kpi_surfaces_tamper_without_alerting_core_recall(
    tmp_path: Path, monkeypatch
) -> None:
    snapshot_module = SimpleNamespace(
        snapshot=lambda _root: {"status": "tampered", "state": "halted", "rollout": 0}
    )
    monkeypatch.setattr(health, "CHRONOVISOR_ROOT", tmp_path / "wiki")
    monkeypatch.setitem(
        sys.modules,
        "chronovisor.recall.recall_distillation_store",
        snapshot_module,
    )

    payload = health.recall_distillation_kpi()

    assert payload["status"] == "tampered"
    assert payload["worker_status"] == "halted"
    assert payload["alert"] is True


def test_recall_distillation_kpi_reads_real_sealed_state_and_pointers(
    tmp_path: Path, monkeypatch
) -> None:
    from chronovisor.recall import recall_distillation_store as store

    chronovisor_root = tmp_path / "wiki"
    state_path = store.distillation_dir(chronovisor_root) / store.STATE_FILE
    store.write_sealed_state(
        state_path,
        {
            "status": "ready",
            "worker_status": "idle",
            "rollout_percent": 5,
            "last_success_at": "2026-08-14T20:00:00Z",
            "hold_reason": "verified_truth_below_floor",
        },
    )
    store.append_chain(
        store.distillation_dir(chronovisor_root) / "label-ledger.jsonl",
        {"authority": "teacher-only", "assignment": {"probe": True}},
    )
    store.append_chain(
        store.distillation_dir(chronovisor_root) / "label-ledger.jsonl",
        {"authority": "verified", "assignment": {"probe": False}},
    )
    store.append_chain(
        store.distillation_dir(chronovisor_root) / "shadow-observation-receipts.jsonl",
        {"kind": "shadow-policy-observation"},
    )
    policy_ids = []
    for kind, marker in (("active", "a"), ("candidate", "b"), ("lkg", "c")):
        policy_id, _, _ = store.write_immutable(
            store.distillation_dir(chronovisor_root) / "policies",
            {"kind": "test-policy", "marker": marker},
            schema="chronovisor.recall-distill-policy.v2",
        )
        policy_ids.append(policy_id)
        store.write_pointer(chronovisor_root, kind, policy_id)
    monkeypatch.setattr(health, "CHRONOVISOR_ROOT", chronovisor_root)

    payload = health.recall_distillation_kpi()

    assert payload["status"] == "idle"
    assert payload["worker_status"] == "idle"
    assert payload["rollout_status"] == "ready"
    assert payload["rollout_percent"] == 5.0
    assert payload["active_policy_id"] == policy_ids[0][:12]
    assert payload["candidate_policy_id"] == policy_ids[1][:12]
    assert payload["lkg_policy_id"] == policy_ids[2][:12]
    assert payload["teacher_only"] == 1
    assert payload["verified_truth"] == 1
    # A raw probe label and unbound observation receipt are not authority:
    # only baseline locked-test probe pairs and stage-bound paired receipts count.
    assert payload["probe_not_truth"] == 0
    assert payload["paired_denominator"] == 0
    assert payload["hold_reason"] == "verified_truth_below_floor"
    assert payload["feature_revision"] == "recall-distill-text-v2"
    assert payload["alert"] is False


def test_recall_distillation_kpi_marks_tampered_real_state_visible(
    tmp_path: Path, monkeypatch
) -> None:
    from chronovisor.recall import recall_distillation_store as store

    chronovisor_root = tmp_path / "wiki"
    state_path = store.distillation_dir(chronovisor_root) / store.STATE_FILE
    store.write_sealed_state(state_path, {"status": "capture_only"})
    tampered = json.loads(state_path.read_text(encoding="utf-8"))
    tampered["status"] = "active"
    state_path.write_text(json.dumps(tampered), encoding="utf-8")
    monkeypatch.setattr(health, "CHRONOVISOR_ROOT", chronovisor_root)

    payload = health.recall_distillation_kpi()

    assert payload["status"] == "tampered"
    assert payload["worker_status"] == "tampered"
    assert payload["rollout_status"] == "tampered"
    assert payload["alert"] is True


def test_semantic_index_kpi_is_inactive_when_rollout_is_off(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "chronovisor.core.runtime_config.load_search_embedding_config",
        lambda: SearchEmbeddingConfig(rollout_mode="off"),
    )

    assert health.semantic_index_kpi()["status"] == "inactive"


def test_semantic_index_kpi_requires_matching_live_service(
    tmp_path: Path, monkeypatch
) -> None:
    chronovisor_root = tmp_path / "wiki"
    runtime = chronovisor_root / "runtime"
    runtime.mkdir(parents=True)
    socket_path = runtime / "semantic.sock"
    (runtime / "semantic-service-status.json").write_text(
        json.dumps(
            {
                "ready": True,
                "pid": os.getpid(),
                    "generation_id": "generation-a",
                    "observed_at_epoch": time.time(),
                    "routes": {
                        "search.semantic.foreground": {
                            "role": "search.semantic.foreground",
                            "provider": "remote",
                            "model": "embed-model",
                            "location": "remote",
                        }
                    },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(health, "CHRONOVISOR_ROOT", chronovisor_root)
    monkeypatch.setattr(
        Path,
        "is_socket",
        lambda path: path == socket_path,
    )
    monkeypatch.setattr(
        "chronovisor.core.runtime_config.load_search_embedding_config",
        lambda: SearchEmbeddingConfig(
            enabled=True,
            rollout_mode="on",
            socket=str(socket_path),
        ),
    )
    monkeypatch.setattr(
        "chronovisor.core.semantic_index.semantic_index_status",
        lambda **_kwargs: {
            "status": "ok",
            "coverage": 1.0,
            "generation_id": "generation-a",
        },
    )
    monkeypatch.setattr(
        "chronovisor.core.semantic_jobs.job_status",
        lambda: {"status": "ok", "counts": {}},
    )
    payload = health.semantic_index_kpi()

    assert payload["status"] == "ok"
    assert payload["service_process_alive"] is True
    assert payload["generation_matches"] is True


def test_lint_queue_kpi_counts_only_unresolved_issue_keys(tmp_path: Path) -> None:
    queue = tmp_path / "lint-repair-queue.jsonl"
    convergence = tmp_path / "state.json"
    rows = [
        {"issue_key": "handled", "page": "a"},
        {"issue_key": "active", "page": "b"},
        {"issue_key": "new", "page": "c"},
        {"page": "missing-key"},
    ]
    queue.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    convergence.write_text(
        json.dumps(
            {
                "items": {
                    "handled": {
                        "lane": "lint_repair",
                        "status": "rejected",
                        "metadata": {"issue_key": "handled"},
                    },
                    "active": {
                        "lane": "lint_repair",
                        "status": "pending_local",
                        "metadata": {"issue_key": "active"},
                    },
                    "other-lane": {
                        "lane": "orphan_link",
                        "status": "pending_local",
                        "metadata": {"issue_key": "new"},
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    payload = health._lint_queue_kpi(queue, convergence)

    assert payload == {
        "total": 4,
        "unique": 4,
        "actionable": 3,
        "active": 1,
        "untracked": 2,
        "handled": 1,
    }


def test_capture_kpi_counts_raw_claim_coverage(tmp_path: Path, monkeypatch) -> None:
    chronovisor_root = tmp_path / "wiki"
    raw_dir = chronovisor_root / "raw"
    claims_dir = chronovisor_root / "claims"
    raw_dir.mkdir(parents=True)
    claims_dir.mkdir(parents=True)
    (raw_dir / "20260706-codex-a.md").write_text("a", encoding="utf-8")
    (raw_dir / "20260706-claude-b.md").write_text("b", encoding="utf-8")
    (claims_dir / "claims.jsonl").write_text(
        json.dumps({"source_raw": "20260706-codex-a.md"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(health, "CHRONOVISOR_ROOT", chronovisor_root)
    monkeypatch.setattr(health, "RAW_DIR", raw_dir)

    payload = health.capture_kpi()

    assert payload["raw_files"] == 2
    assert payload["claimed_raw_files"] == 1
    assert payload["claim_coverage"] == 0.5
    assert payload["raw_by_host"] == {"codex": 1, "claude": 1}


def test_latest_memory_integrity_reads_summary(tmp_path: Path, monkeypatch) -> None:
    chronovisor_root = tmp_path / "wiki"
    eval_dir = chronovisor_root / "eval"
    eval_dir.mkdir(parents=True)
    (eval_dir / "memory-integrity-latest.json").write_text(
        json.dumps(
            {"status": "ok", "total": 2, "passed": 1, "missed": 1, "capture_rate": 0.5}
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(health, "CHRONOVISOR_ROOT", chronovisor_root)

    payload = health.latest_memory_integrity()

    assert payload["status"] == "ok"
    assert payload["capture_rate"] == 0.5


@pytest.mark.parametrize(
    "runtime_status",
    ["waiting_for_ingest_runtime", "waiting_for_ollama"],
)
def test_ingest_liveness_kpi_alerts_when_runtime_blocks_pending_raws(
    tmp_path: Path, monkeypatch, runtime_status: str
) -> None:
    chronovisor_root = tmp_path / "wiki"
    state_path = chronovisor_root / "runtime" / "ingest-liveness.json"
    state_path.parent.mkdir(parents=True)
    state_path.write_text(
        json.dumps(
            {
                "status": runtime_status,
                "pending_raws": 7,
                "observed_at": "2026-07-17T12:00:00",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(health, "CHRONOVISOR_ROOT", chronovisor_root)

    payload = health.ingest_liveness_kpi()

    assert payload["status"] == "alert"
    assert payload["runtime_status"] == runtime_status
    assert payload["pending_raws"] == 7


def test_ingest_liveness_kpi_alerts_on_invalid_authority_without_pending_raws(
    tmp_path: Path, monkeypatch
) -> None:
    chronovisor_root = tmp_path / "wiki"
    state_path = chronovisor_root / "runtime" / "ingest-liveness.json"
    state_path.parent.mkdir(parents=True)
    state_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "status": "blocked_by_decision_authority",
                "pending_raws": 0,
                "alert": True,
                "error": "adoption artifact policy version mismatch",
                "observed_at": "2026-07-28T20:42:42",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(health, "CHRONOVISOR_ROOT", chronovisor_root)

    payload = health.ingest_liveness_kpi()

    assert payload["status"] == "alert"
    assert payload["runtime_status"] == "blocked_by_decision_authority"
    assert payload["pending_raws"] == 0
    assert payload["alert"] is True


def test_cofire_kpi_reads_graph_summary(tmp_path: Path, monkeypatch) -> None:
    chronovisor_root = tmp_path / "wiki"
    recall_dir = chronovisor_root / "recall"
    recall_dir.mkdir(parents=True)
    (recall_dir / "cofire.json").write_text(
        json.dumps({"status": "ok", "nodes": 3, "edges": 4}),
        encoding="utf-8",
    )
    monkeypatch.setattr(health, "CHRONOVISOR_ROOT", chronovisor_root)

    payload = health.cofire_kpi()

    assert payload["nodes"] == 3
    assert payload["edges"] == 4


def test_derived_memory_kpi_counts_generated_artifacts(
    tmp_path: Path, monkeypatch
) -> None:
    chronovisor_root = tmp_path / "wiki"
    (chronovisor_root / "claims").mkdir(parents=True)
    (chronovisor_root / "recall").mkdir(parents=True)
    (chronovisor_root / "distill").mkdir(parents=True)
    (chronovisor_root / "pages" / "hubs").mkdir(parents=True)
    (chronovisor_root / "claims" / "claims-index.jsonl").write_text(
        "{}\n{}\n", encoding="utf-8"
    )
    (chronovisor_root / "recall" / "search-golden.jsonl").write_text(
        "{}\n", encoding="utf-8"
    )
    (chronovisor_root / "recall" / "retention.json").write_text(
        json.dumps({"counts": {"pages": 5, "deprecation_candidates": 1}}),
        encoding="utf-8",
    )
    (chronovisor_root / "distill" / "wiki-qa.jsonl").write_text(
        "{}\n{}\n{}\n", encoding="utf-8"
    )
    (chronovisor_root / "pages" / "hubs" / "ai-hub.md").write_text(
        "---\ntitle: AI hub\nstatus: stable\ntype: knowledge\n---\nhub",
        encoding="utf-8",
    )
    (chronovisor_root / "pages" / "hubs" / "draft.md").write_text(
        "---\ntitle: Draft\nstatus: draft\ntype: knowledge\n---\ndraft",
        encoding="utf-8",
    )
    (chronovisor_root / "pages" / "hubs" / "invalid.md").write_text(
        "---\ntitle: Invalid\nstatus: stable\n---\ninvalid",
        encoding="utf-8",
    )
    monkeypatch.setattr(health, "CHRONOVISOR_ROOT", chronovisor_root)

    payload = health.derived_memory_kpi()

    assert payload["claims"] == 2
    assert payload["golden"] == 1
    assert payload["retention_pages"] == 5
    assert payload["distill_rows"] == 3
    assert payload["hubs"] == 1


def test_convergence_kpi_splits_semantic_defer_from_operational_quarantine(
    tmp_path: Path,
    monkeypatch,
) -> None:
    chronovisor_root = tmp_path / "wiki"
    runtime = chronovisor_root / "runtime" / "convergence"
    runtime.mkdir(parents=True)
    (runtime / "state.json").write_text(
        json.dumps(
            {
                "items": {
                    "semantic": {
                        "status": "quarantined",
                        "lane": "content_correction",
                        "input_hash": "a" * 64,
                        "last_failure_class": "local_semantic_no_quorum",
                        "quarantine_reason": (
                            "semantic_no_quorum:content_correction_review"
                        ),
                        "result": {
                            "terminal_reason": "semantic_no_quorum",
                            "semantic_hold": {
                                "kind": "content_correction_semantic_no_quorum",
                                "decision_lane": "content_correction_review",
                                "input_hash": "a" * 64,
                                "proposal_sha256": "b" * 64,
                                "page_evidence_hashes": {"page": "c" * 64},
                                "authority": {"source": "adopted_local_consensus"},
                            },
                        },
                    },
                    "legacy-no-quorum-retry-exhausted": {
                        "status": "quarantined",
                        "lane": "autonomy_duplicate_resolution",
                        "last_failure_class": "local_semantic_no_quorum",
                        "quarantine_reason": "retry_exhausted:frontier",
                    },
                    "raw-semantic-class": {
                        "status": "quarantined",
                        "lane": "content_correction",
                        "last_failure_class": "ingest.semantic_no_quorum",
                        "quarantine_reason": "semantic_no_quorum:ingest",
                        "result": {
                            "terminal_reason": "semantic_no_quorum",
                            "semantic_hold": {"kind": "ingest_semantic_no_quorum"},
                        },
                    },
                    "operational": {
                        "status": "quarantined",
                        "lane": "content_correction",
                        "last_failure_class": "review_artifact_invalid",
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(health, "CHRONOVISOR_ROOT", chronovisor_root)

    payload = health.convergence_kpi()

    assert payload["items"] == 4
    assert payload["semantic_deferred"] == 1
    assert payload["quarantined"] == 3
    assert payload["by_status"] == {"quarantined": 3, "semantic_deferred": 1}
    assert payload["by_lane"]["autonomy_duplicate_resolution"] == {"quarantined": 1}
    assert payload["by_lane"]["content_correction"] == {
        "quarantined": 2,
        "semantic_deferred": 1,
    }


def test_health_accepts_only_self_hashed_legacy_semantic_migration() -> None:
    lane = autonomy.DUPLICATE_FRONTIER_LANE
    authority = semantic_authority(lane)
    original = {
        "quarantine_reason": "retry_exhausted:frontier",
        "frontier_attempts": 3,
        "last_error": "three-way split",
    }
    marker = autonomy._legacy_semantic_hold(
        lane=lane,
        epoch={"input_hash": "a" * 64},
        authority=authority,
        item=original,
    )
    item = {
        "status": "quarantined",
        "lane": lane,
        "last_failure_class": "local_semantic_no_quorum",
        "quarantine_reason": f"semantic_no_quorum_legacy:{lane}",
        "result": {
            "terminal_reason": "semantic_no_quorum",
            "legacy_semantic_hold": marker,
        },
    }

    assert health._is_terminal_semantic_defer(item) is True
    marker["epoch"]["input_hash"] = "b" * 64
    assert health._is_terminal_semantic_defer(item) is False
