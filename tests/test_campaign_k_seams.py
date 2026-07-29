from __future__ import annotations

import ast
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

from chronovisor.classification import classification_engine
from chronovisor.ingest import ingest, ingest_review_apply, orchestrator
from chronovisor.librarian import collection_authority
from chronovisor.ops import self_heal
from chronovisor.recall import content_correction


def test_ingest_job_result_keeps_legacy_alias_without_page_bodies() -> None:
    frontier = {
        "status": "apply_available",
        "proposal_sha256": "a" * 64,
        "source_key": "source",
        "review": {"decision": "apply_available"},
        "audit": {"mode": "mandatory"},
        "recovered_artifact": True,
        "reused_review": False,
        "raw_content": "must not escape",
    }
    failed = [{"filename": "broken.md"}]
    read_back = {"failed": ["page"], "checked": 1}

    result = ingest._build_ingest_job_result(frontier, failed, read_back)

    assert result["frontier"] == result["local_consensus"]
    assert result["partial"] is True
    assert result["failed_ops"] == failed
    assert result["read_back"] == read_back
    assert "raw_content" not in json.dumps(result)


def test_orchestrator_result_builders_preserve_public_envelopes(tmp_path: Path) -> None:
    projection = SimpleNamespace(
        kind="children",
        manifest_path=tmp_path / "manifest.json",
        projection_paths=[tmp_path / "projection.jsonl"],
        child_paths=[tmp_path / "child.jsonl"],
        noop_receipt_path=None,
        parent_sha256="a" * 64,
        projection_sha256="b" * 64,
        record_count=3,
        selected_record_count=2,
        child_count=1,
        role_counts={"user": 2},
    )
    projection_result = orchestrator._projection_result_summary(projection)
    continuation = {"kind": "ingest_review_shard_continuation"}
    raw_result = orchestrator._raw_unit_result(
        filename="raw.jsonl",
        source_files=["raw.jsonl"],
        job_id="job",
        succeeded=False,
        deferred=False,
        continued=True,
        continuation=continuation,
        fragment_record_sha256="c" * 64,
        projection=projection_result,
        supervision=None,
        completion_ack={"resumed": True},
    )

    result = orchestrator._ingest_batch_result(
        reason="forced",
        job_ids=["job"],
        filenames=["raw.jsonl"],
        succeeded_filenames=[],
        deferred_filenames=[],
        continued_filenames=["raw.jsonl"],
        failed_filenames=0,
        fragment_quarantined=[{"files": ["bad-a", "bad-b"]}],
        fragment_deferred=[],
        resumed_fragment_quarantines=[],
        per_raw=[raw_result],
        processor="ollama",
        elapsed=1.236,
    )

    assert raw_result["continuation"] == continuation
    assert raw_result["projection"]["child_count"] == 1
    assert result["files_quarantined"] == ["bad-a", "bad-b"]
    assert result["elapsed_seconds"] == 1.24


def test_frontier_attempt_outcome_is_deterministic() -> None:
    now = datetime(2026, 7, 29, 12, 0, tzinfo=UTC)

    retry = self_heal._frontier_attempt_outcome(
        {"decision": "needs_retry"},
        attempt=2,
        max_attempts=3,
        backoff_base_seconds=10,
        now=now,
    )
    exhausted = self_heal._frontier_attempt_outcome(
        {"decision": "needs_retry"},
        attempt=3,
        max_attempts=3,
        backoff_base_seconds=10,
        now=now,
    )

    assert retry == ("frontier_retry", "2026-07-29T12:00:20+00:00")
    assert exhausted == ("frontier_quarantined", None)


def test_content_correction_audit_row_is_a_pure_projection() -> None:
    mutation = SimpleNamespace(correction_id="correction-1")
    row = content_correction._content_correction_audit_row(
        key="item-1",
        event={"source_decision_id": "decision-1"},
        proposal={"decision": "wrong_retrieval", "proposals": [{"page_id": "p"}]},
        review={"decision": "approved"},
        apply_result={"status": "applied"},
        verification={"status": "ok"},
        mutations=[mutation],
        page_ids=["p"],
        timestamp="2026-07-29T12:00:00+00:00",
    )

    assert row["kind"] == "content_correction"
    assert row["correction_id"] == "correction-1"
    assert row["classification"] == "wrong_retrieval"
    assert row["pages"] == ["p"]


def test_review_artifact_projection_requires_exact_applied_postimages() -> None:
    artifact = {
        "review": {"decision": "apply_available"},
        "authority": {"lane": "ingest_reconciliation"},
    }

    projected = ingest_review_apply.inspect_ingest_review_artifact(
        artifact,
        has_planned_operations=True,
        planned_postimages_fully_applied=True,
    )

    assert projected.review == artifact["review"]
    assert projected.authority == artifact["authority"]
    assert projected.exact_postimages_already_applied is True


def test_collection_review_projection_preserves_nonmutation_contract() -> None:
    primary = collection_authority._apply_collection_review_result(
        {"status": "queued"},
        {"decision": "review_recommended", "suggested_collection_slug": "ai"},
        role="primary",
        model="model",
        model_digest="digest",
        prompt_sha256="prompt",
        reviewed_at="2026-07-29T12:00:00+00:00",
    )
    challenger = collection_authority._apply_collection_review_result(
        primary,
        {"decision": "review_recommended", "suggested_collection_slug": "ai"},
        role="challenger",
        model="challenger",
        model_digest="challenger-digest",
        prompt_sha256="prompt",
        reviewed_at="2026-07-29T12:01:00+00:00",
    )

    assert primary["status"] == "review_recommended"
    assert challenger["challenge_status"] == "consensus_recommended"
    assert "assignment" not in challenger
    assert collection_authority._collection_worker_contract_matches(
        {
            "schema": "worker",
            "model": "model",
            "model_digest": "digest",
            "prompt_sha256": "prompt",
            "model_calls": 1,
            "page_mutations": 0,
            "assignment_mutations": 0,
            "result": {},
        },
        schema="worker",
        model="model",
        model_digest="digest",
        prompt_sha256="prompt",
    )


def test_classification_batch_identity_and_cache_projection_are_stable() -> None:
    evidence = {"summary": "grounded"}
    row = {
        "uid": "page-1",
        "source_sha256": "a" * 64,
        "candidates": [{"notation": "004"}],
        "evidence_card": evidence,
    }
    batch_input = classification_engine._classification_batch_input(
        [row],
        package_checksum="b" * 64,
        adjudication_mode="proposal-audit",
        stage_cache_epoch="default",
    )
    expected_digest = hashlib.sha256(
        json.dumps(
            evidence,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()

    assert batch_input["pages"][0]["evidence_card_sha256"] == f"sha256:{expected_digest}"
    assert classification_engine._cached_classification_decisions(
        {"status": "applied", "result": {"decisions": [{"uid": "page-1"}]}}
    ) == [{"uid": "page-1"}]
    assert (
        classification_engine._cached_classification_decisions(
            {"status": "local_pending", "result": {"decisions": []}}
        )
        is None
    )


def test_selected_orchestrators_stay_below_campaign_k_size_caps() -> None:
    package = Path(__file__).resolve().parents[1] / "src" / "chronovisor"
    targets = {
        package / "ingest" / "ingest.py": {"run_ingest": 525},
        package / "ingest" / "orchestrator.py": {"run_pending_ingest": 750},
        package / "ops" / "self_heal.py": {"_handle_packet_unlocked": 780},
        package / "recall" / "content_correction.py": {
            "_process_frontier_item": 860
        },
        package / "ingest" / "ingest_review_apply.py": {
            "review_and_apply_ingest_operations": 700
        },
        package / "librarian" / "collection_authority.py": {
            "review_collection_queue": 220
        },
        package / "classification" / "classification_engine.py": {
            "run_consensus_batches": 150
        },
    }

    for path, limits in targets.items():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        functions = {
            node.name: node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        for name, maximum in limits.items():
            function = functions[name]
            assert function.end_lineno is not None
            assert function.end_lineno - function.lineno + 1 <= maximum
