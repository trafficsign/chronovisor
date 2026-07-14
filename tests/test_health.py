from __future__ import annotations

import json
from pathlib import Path

from llm_wiki_mcp import autonomy, health


def test_capture_kpi_counts_raw_claim_coverage(tmp_path: Path, monkeypatch) -> None:
    wiki_root = tmp_path / "wiki"
    raw_dir = wiki_root / "raw"
    claims_dir = wiki_root / "claims"
    raw_dir.mkdir(parents=True)
    claims_dir.mkdir(parents=True)
    (raw_dir / "20260706-codex-a.md").write_text("a", encoding="utf-8")
    (raw_dir / "20260706-claude-b.md").write_text("b", encoding="utf-8")
    (claims_dir / "claims.jsonl").write_text(
        json.dumps({"source_raw": "20260706-codex-a.md"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(health, "WIKI_ROOT", wiki_root)
    monkeypatch.setattr(health, "RAW_DIR", raw_dir)

    payload = health.capture_kpi()

    assert payload["raw_files"] == 2
    assert payload["claimed_raw_files"] == 1
    assert payload["claim_coverage"] == 0.5
    assert payload["raw_by_host"] == {"codex": 1, "claude": 1}


def test_latest_memory_integrity_reads_summary(tmp_path: Path, monkeypatch) -> None:
    wiki_root = tmp_path / "wiki"
    eval_dir = wiki_root / "eval"
    eval_dir.mkdir(parents=True)
    (eval_dir / "memory-integrity-latest.json").write_text(
        json.dumps(
            {"status": "ok", "total": 2, "passed": 1, "missed": 1, "capture_rate": 0.5}
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(health, "WIKI_ROOT", wiki_root)

    payload = health.latest_memory_integrity()

    assert payload["status"] == "ok"
    assert payload["capture_rate"] == 0.5


def test_cofire_kpi_reads_graph_summary(tmp_path: Path, monkeypatch) -> None:
    wiki_root = tmp_path / "wiki"
    recall_dir = wiki_root / "recall"
    recall_dir.mkdir(parents=True)
    (recall_dir / "cofire.json").write_text(
        json.dumps({"status": "ok", "nodes": 3, "edges": 4}),
        encoding="utf-8",
    )
    monkeypatch.setattr(health, "WIKI_ROOT", wiki_root)

    payload = health.cofire_kpi()

    assert payload["nodes"] == 3
    assert payload["edges"] == 4


def test_derived_memory_kpi_counts_generated_artifacts(
    tmp_path: Path, monkeypatch
) -> None:
    wiki_root = tmp_path / "wiki"
    (wiki_root / "claims").mkdir(parents=True)
    (wiki_root / "recall").mkdir(parents=True)
    (wiki_root / "distill").mkdir(parents=True)
    (wiki_root / "pages" / "hubs").mkdir(parents=True)
    (wiki_root / "claims" / "claims-index.jsonl").write_text(
        "{}\n{}\n", encoding="utf-8"
    )
    (wiki_root / "recall" / "search-golden.jsonl").write_text("{}\n", encoding="utf-8")
    (wiki_root / "recall" / "retention.json").write_text(
        json.dumps({"counts": {"pages": 5, "archive_candidates": 1}}),
        encoding="utf-8",
    )
    (wiki_root / "distill" / "wiki-qa.jsonl").write_text(
        "{}\n{}\n{}\n", encoding="utf-8"
    )
    (wiki_root / "pages" / "hubs" / "ai-hub.md").write_text("hub", encoding="utf-8")
    monkeypatch.setattr(health, "WIKI_ROOT", wiki_root)

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
    wiki_root = tmp_path / "wiki"
    runtime = wiki_root / "runtime" / "convergence"
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
    monkeypatch.setattr(health, "WIKI_ROOT", wiki_root)

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
    authority = {
        "source": "injected_reviewer_boundary",
        "authority_version": 1,
        "lane": lane,
    }
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
