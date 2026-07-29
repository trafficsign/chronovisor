from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from chronovisor.core.durable_state import write_sealed_json
from chronovisor.librarian.librarian import run_legacy_udc_shadow, run_shadow
from chronovisor.librarian.librarian_status import (
    _derive_code,
    _library_evidence_status,
    _soak_status,
)
from chronovisor.ops import dashboard


def test_dashboard_static_contract_exposes_librarian_progress() -> None:
    html = (dashboard.STATIC_DIR / "index.html").read_text(encoding="utf-8")
    js = (dashboard.STATIC_DIR / "app.js").read_text(encoding="utf-8")
    for identifier in (
        "librarian-state",
        "librarian-swept-generation",
        "librarian-uid",
        "librarian-classification",
        "librarian-links",
        "librarian-migration",
        "librarian-sweep",
        "librarian-queue",
        "librarian-receipts",
        "librarian-authority",
        "librarian-quality",
        "librarian-collection-count",
        "librarian-assignment",
        "librarian-crosswalk",
        "librarian-top-share",
        "librarian-review-queue",
        "librarian-split-proposals",
        "librarian-rollout",
        "librarian-soak",
        "librarian-recovery",
        "librarian-evidence-status",
        "librarian-evidence-fixture",
        "librarian-evidence-external",
        "librarian-evidence-resource",
        "librarian-evidence-authority",
        "librarian-evidence-update",
    ):
        assert f'id="{identifier}"' in html
    assert "function renderLibrarian" in js
    assert "renderLibrarian(snapshot.librarian || {})" in js


def test_collection_first_status_exposes_registry_quality(
    tmp_path: Path,
) -> None:
    pages = (
        ("ai", "model"),
        ("career", "interview"),
        ("workplace", "meeting"),
    )
    for index, (folder, name) in enumerate(pages, 1):
        uid = (
            f"019f0000-000{index}-7000-8000-"
            f"{index:012d}"
        )
        path = tmp_path / "pages" / folder / f"{name}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "---\n"
            f"title: {name}\n"
            f"uid: {uid}\n"
            "updated: 2026-07-27\n"
            "---\n\n"
            f"# {name}\n",
            encoding="utf-8",
        )
    archived = tmp_path / "pages" / "ai" / "archived.md"
    archived.write_text(
        "---\n"
        "title: Archived\n"
        "uid: 019f0000-0004-7000-8000-000000000004\n"
        "updated: 2026-07-27\n"
        "status: archived\n"
        "---\n\n"
        "# Archived\n",
        encoding="utf-8",
    )

    result = run_shadow(root=tmp_path, full_sweep=True)

    from chronovisor.librarian.librarian_status import build_librarian_status

    status = build_librarian_status(tmp_path)
    assert result["observed"] == 3
    assert status["authority"]["mode"] == "collection-first"
    assert status["progress"]["collection_assignment"]["numerator"] == 3
    assert status["progress"]["collection_assignment"]["denominator"] == 3
    assert status["progress"]["full_sweep"]["current"] is True
    assert (
        status["collection_authority"]["metrics"]["assignment_coverage"]
        == 1.0
    )
    assert status["quality"]["legacy_page_udc_gate"] == (
        "superseded_by_collection_authority_v1"
    )


def test_collection_review_queue_is_visible_but_not_catch_up_work(
    tmp_path: Path,
) -> None:
    page = tmp_path / "pages" / "ai" / "model.md"
    page.parent.mkdir(parents=True)
    page.write_text(
        "---\n"
        "title: Model\n"
        "uid: 019f0000-0100-7000-8000-000000000100\n"
        "updated: 2026-07-27\n"
        "---\n\n"
        "# Model\n",
        encoding="utf-8",
    )
    run_shadow(root=tmp_path, full_sweep=True)
    write_sealed_json(
        tmp_path
        / "runtime"
        / "librarian"
        / "collection-review-queue.json",
        {
            "candidate_count": 5,
            "open": 5,
            "completed": 1,
            "reviewer_calls": 3,
            "frontier_calls": 0,
            "items": {
                "dismissed": {
                    "status": "dismissed",
                    "model_review": {"decision": "no_issue"},
                },
                "consensus": {
                    "status": "review_recommended",
                    "challenge_status": "consensus_recommended",
                    "model_review": {"decision": "review_recommended"},
                    "challenger_review": {
                        "decision": "review_recommended"
                    },
                },
                "pending": {"status": "queued"},
            },
        },
    )

    from chronovisor.librarian.librarian_status import build_librarian_status

    status = build_librarian_status(tmp_path)

    assert status["queue"]["actionable"] == 0
    assert status["debts"]["collection_review_queue"] == 5
    assert status["collection_authority"]["queue"]["open"] == 5
    assert status["collection_authority"]["queue"]["completed"] == 1
    assert status["collection_authority"]["queue"]["primary_reviews"] == 2
    assert status["collection_authority"]["queue"]["challenger_reviews"] == 1
    assert (
        status["collection_authority"]["queue"]["consensus_recommended"] == 1
    )
    assert status["collection_authority"]["queue"]["status_counts"] == {
        "dismissed": 1,
        "queued": 1,
        "review_recommended": 1,
    }
    assert status["progress"]["classification_terminal"] == {
        "denominator": 1,
        "numerator": 1,
        "scope_generation": status["scope_generation"],
    }
    assert (
        _derive_code(
            {
                "enabled": True,
                "authority": {"active": True},
                "blocked_reasons": [],
                "initial_organization_complete_at": (
                    "2026-07-27T00:00:00+00:00"
                ),
                "progress": {"full_sweep": {"current": True}},
            },
            status["queue"],
        )
        == "STEADY_CLEAN"
    )


def test_collection_review_required_is_terminal_with_visible_hold(
    tmp_path: Path,
) -> None:
    page = tmp_path / "pages" / "misc" / "note.md"
    page.parent.mkdir(parents=True)
    page.write_text(
        "---\n"
        "title: Note\n"
        "uid: 019f0000-0101-7000-8000-000000000101\n"
        "updated: 2026-07-27\n"
        "---\n\n"
        "# Note\n",
        encoding="utf-8",
    )
    run_shadow(root=tmp_path, full_sweep=True)

    from chronovisor.librarian.librarian_status import build_librarian_status

    status = build_librarian_status(tmp_path)

    assert status["queue"]["actionable"] == 0
    assert status["queue"]["held"] == 1
    assert status["progress"]["classification_terminal"]["numerator"] == 1
    assert status["progress"]["classification_terminal"]["denominator"] == 1
    assert (
        _derive_code(
            {
                "enabled": True,
                "authority": {"active": True},
                "blocked_reasons": [],
                "initial_organization_complete_at": (
                    "2026-07-27T00:00:00+00:00"
                ),
                "progress": {"full_sweep": {"current": True}},
            },
            status["queue"],
        )
        == "STEADY_WITH_HOLDS"
    )


def test_fast_status_payload_can_be_built_from_shadow_state(tmp_path: Path) -> None:
    page = tmp_path / "pages" / "alpha.md"
    page.parent.mkdir(parents=True)
    page.write_text(
        "---\ntitle: Alpha\nupdated: 2026-07-25\ntags: [d/ai]\n---\n\n# Alpha\n",
        encoding="utf-8",
    )
    run_legacy_udc_shadow(root=tmp_path, full_sweep=True)
    write_sealed_json(
        tmp_path / "runtime" / "librarian" / "rollout.json",
        {
            "schema": "chronovisor.librarian-rollout.v1",
            "status": "running",
            "stage": "phase0_fixture_adjudication",
            "updated_at": "2026-07-25T10:12:33+00:00",
            "detail": {},
        },
    )

    from chronovisor.librarian.librarian_status import build_librarian_status

    status = build_librarian_status(tmp_path)
    assert status["state"] == "NOT_READY"
    assert status["reason_codes"]
    assert "threshold_version" in status
    assert status["rollout"] == {
        "status": "running",
        "stage": "phase0_fixture_adjudication",
        "updated_at": "2026-07-25T10:12:33+00:00",
    }
    assert status["progress"]["uid"] == {
        "numerator": 1,
        "denominator": 1,
        "scope_generation": status["scope_generation"],
    }


def test_library_evidence_dashboard_reports_in_progress_runtime_stage(
    tmp_path: Path,
) -> None:
    write_sealed_json(
        tmp_path / "classification" / "library-evidence" / "state.json",
        {
            "schema": "chronovisor.classification-library-pilot-state.v1",
            "status": "running",
            "stage": "e0_adjudicate",
            "fixture_cursor": 15,
            "fixture_accepted": 12,
        },
    )

    status = _library_evidence_status(tmp_path)

    assert status["status"] == "running"
    assert status["stage"] == "e0_adjudicate"
    assert status["fixture"]["adjudication_cursor"] == 15
    assert status["fixture"]["adjudication_accepted"] == 12


def test_library_evidence_dashboard_prefers_annif_runtime(tmp_path: Path) -> None:
    write_sealed_json(
        tmp_path / "classification" / "library-evidence" / "state.json",
        {
            "schema": "chronovisor.classification-library-pilot-state.v1",
            "status": "rejected",
            "stage": "e0_early_sample_rejected",
            "fixture_cursor": 50,
            "fixture_accepted": 50,
        },
        backup=False,
    )
    write_sealed_json(
        tmp_path / "classification" / "annif-pilot" / "state.json",
        {
            "schema": "chronovisor.classification-annif-pilot-state.v1",
            "status": "running",
            "stage": "download-czech-bibliography",
        },
        backup=False,
    )
    write_sealed_json(
        tmp_path
        / "classification"
        / "annif-pilot"
        / "early-council-review.json",
        {
            "decision": "reject-council",
            "council_hit_count": 3,
            "source_completed_rows": 50,
            "cases": [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}],
        },
        backup=False,
    )

    status = _library_evidence_status(tmp_path)

    assert status["method"] == "annif"
    assert status["status"] == "running"
    assert status["stage"] == "download-czech-bibliography"
    assert status["annif"]["council_hit_count"] == 3
    assert status["annif"]["council_case_count"] == 10


def test_library_evidence_dashboard_prefers_latest_profile_gate(
    tmp_path: Path,
) -> None:
    write_sealed_json(
        tmp_path / "classification" / "annif-pilot" / "state.json",
        {
            "schema": "chronovisor.classification-annif-pilot-state.v1",
            "status": "rejected",
            "stage": "early-gate-complete",
        },
        backup=False,
    )
    profile_root = tmp_path / "classification" / "profile-retrieval-pilot"
    write_sealed_json(
        profile_root / "state.json",
        {
            "schema": "chronovisor.classification-profile-pilot-state.v1",
            "status": "rejected",
            "stage": "fixed-ten-retrieval-complete",
            "decision": "reject-profile-retrieval",
        },
        backup=False,
    )
    write_sealed_json(
        profile_root / "manifest.json",
        {
            "schema": "chronovisor.classification-profile-index.v1",
            "profile_count": 1850,
            "embedding_model": "bge-m3",
            "dimensions": 1024,
            "working_set_bytes": 10_013_043,
            "external_library_records_used": 0,
            "local_page_label_associations_used": 0,
            "llm_calls": 0,
        },
        backup=False,
    )
    write_sealed_json(
        profile_root / "evaluation.json",
        {
            "schema": "chronovisor.classification-profile-evaluation.v1",
            "decision": "reject-profile-retrieval",
            "case_count": 10,
            "baseline_hit_count": 4,
            "baseline_recall_at_12": 0.4,
            "baseline_mrr": 0.2667,
            "profile_hit_count": 4,
            "profile_recall_at_12": 0.4,
            "profile_mrr": 0.2183,
            "minimum_profile_hits": 8,
            "larger_evaluation_authorized": False,
            "llm_calls": 0,
        },
        backup=False,
    )

    status = _library_evidence_status(tmp_path)

    assert status["method"] == "udc-profile-dense"
    assert status["status"] == "rejected"
    assert status["stage"] == "fixed-ten-retrieval-complete"
    assert status["profile_retrieval"] == {
        "status": "rejected",
        "stage": "fixed-ten-retrieval-complete",
        "decision": "reject-profile-retrieval",
        "case_count": 10,
        "baseline_hit_count": 4,
        "baseline_recall_at_12": 0.4,
        "baseline_mrr": 0.2667,
        "profile_hit_count": 4,
        "profile_recall_at_12": 0.4,
        "profile_mrr": 0.2183,
        "minimum_profile_hits": 8,
        "larger_evaluation_authorized": False,
        "profile_count": 1850,
        "embedding_model": "bge-m3",
        "dimensions": 1024,
        "working_set_bytes": 10_013_043,
        "external_library_records_used": 0,
        "local_page_label_associations_used": 0,
        "llm_calls": 0,
    }


def test_library_evidence_dashboard_prefers_query2doc_gate(
    tmp_path: Path,
) -> None:
    query_root = tmp_path / "classification" / "query2doc-pilot"
    write_sealed_json(
        query_root / "state.json",
        {
            "schema": "chronovisor.classification-query2doc-pilot-state.v1",
            "status": "qualified",
            "stage": "fixed-ten-query2doc-complete",
            "decision": "qualify-query2doc-retrieval",
        },
        backup=False,
    )
    write_sealed_json(
        query_root / "manifest.json",
        {
            "schema": "chronovisor.classification-query2doc-manifest.v1",
            "query_count": 10,
        },
        backup=False,
    )
    write_sealed_json(
        query_root / "evaluation.json",
        {
            "schema": "chronovisor.classification-query2doc-evaluation.v1",
            "decision": "qualify-query2doc-retrieval",
            "case_count": 10,
            "model": "ornith:test",
            "model_digest": "sha256:model",
            "prompt_sha256": "sha256:prompt",
            "model_calls": 10,
            "model_attempts": 10,
            "metrics": {
                "raw_lexical": {"hit_count": 4, "recall_at_12": 0.4},
                "raw_dense": {"hit_count": 4, "recall_at_12": 0.4},
                "query2doc_lexical": {"hit_count": 7, "recall_at_12": 0.7},
                "query2doc_dense": {"hit_count": 9, "recall_at_12": 0.9},
                "fused": {"hit_count": 9, "recall_at_12": 0.9},
            },
            "minimum_fused_hits": 8,
            "unseen_evaluation_authorized": True,
            "larger_corpus_evaluation_authorized": False,
            "classification_judge_calls": 0,
            "page_mutations": 0,
        },
        backup=False,
    )

    status = _library_evidence_status(tmp_path)

    assert status["method"] == "query2doc-rrf"
    assert status["status"] == "qualified"
    assert status["stage"] == "fixed-ten-query2doc-complete"
    assert status["query2doc"]["fused"]["hit_count"] == 9
    assert status["query2doc"]["query2doc_dense"]["hit_count"] == 9
    assert status["query2doc"]["unseen_evaluation_authorized"] is True
    assert status["query2doc"]["larger_corpus_evaluation_authorized"] is False
    assert status["query2doc"]["classification_judge_calls"] == 0
    assert status["query2doc"]["query_count"] == 10


def test_library_evidence_dashboard_prefers_unseen_query2doc_gate(
    tmp_path: Path,
) -> None:
    fixed_root = tmp_path / "classification" / "query2doc-pilot"
    write_sealed_json(
        fixed_root / "state.json",
        {
            "schema": "chronovisor.classification-query2doc-pilot-state.v1",
            "status": "qualified",
            "stage": "fixed-ten-query2doc-complete",
        },
        backup=False,
    )
    unseen_root = tmp_path / "classification" / "query2doc-unseen"
    write_sealed_json(
        unseen_root / "state.json",
        {
            "schema": "chronovisor.classification-query2doc-unseen-state.v1",
            "status": "rejected",
            "stage": "unseen-query2doc-complete",
            "decision": "reject-unseen-query2doc-retrieval",
        },
        backup=False,
    )
    write_sealed_json(
        unseen_root / "manifest.json",
        {
            "schema": "chronovisor.classification-query2doc-unseen-manifest.v1",
            "query_count": 30,
        },
        backup=False,
    )
    write_sealed_json(
        unseen_root / "evaluation.json",
        {
            "schema": "chronovisor.classification-query2doc-unseen-evaluation.v1",
            "decision": "reject-unseen-query2doc-retrieval",
            "case_count": 30,
            "model": "ornith:test",
            "model_calls": 30,
            "model_attempts": 31,
            "metrics": {
                "raw_lexical": {"hit_count": 16, "recall_at_12": 16 / 30},
                "raw_dense": {"hit_count": 8, "recall_at_12": 8 / 30},
                "query2doc_lexical": {
                    "hit_count": 18,
                    "recall_at_12": 0.6,
                },
                "query2doc_dense": {
                    "hit_count": 13,
                    "recall_at_12": 13 / 30,
                },
                "fused": {"hit_count": 19, "recall_at_12": 19 / 30},
            },
            "minimum_fused_hits": 24,
            "best_raw_hit_count": 16,
            "decision_trial_authorized": False,
            "larger_corpus_evaluation_authorized": False,
            "classification_judge_calls": 0,
            "page_mutations": 0,
        },
        backup=False,
    )

    status = _library_evidence_status(tmp_path)

    assert status["method"] == "query2doc-unseen-rrf"
    assert status["status"] == "rejected"
    assert status["stage"] == "unseen-query2doc-complete"
    assert status["query2doc_unseen"]["fused"]["hit_count"] == 19
    assert status["query2doc_unseen"]["best_raw_hit_count"] == 16
    assert status["query2doc_unseen"]["decision_trial_authorized"] is False
    assert status["query2doc_unseen"]["query_count"] == 30


def test_status_overlays_latest_locked_calibration_quality(tmp_path: Path) -> None:
    page = tmp_path / "pages" / "alpha.md"
    page.parent.mkdir(parents=True)
    page.write_text(
        "---\ntitle: Alpha\nupdated: 2026-07-25\ntags: [d/ai]\n---\n\n# Alpha\n",
        encoding="utf-8",
    )
    run_legacy_udc_shadow(root=tmp_path, full_sweep=True)
    write_sealed_json(
        tmp_path / "classification" / "calibration.json",
        {
            "schema": "chronovisor.classification-calibration.v1",
            "status": "rejected",
            "holdout_metrics": {
                "exact_match_rate": 0.78,
                "forced_misclassification_rate": 0.12,
            },
            "gates": {"forced_misclassification": False},
        },
    )

    from chronovisor.librarian.librarian_status import build_librarian_status

    status = build_librarian_status(tmp_path)

    assert status["quality"]["locked_holdout"] == "rejected"
    assert status["quality"]["holdout_metrics"] == {
        "exact_match_rate": 0.78,
        "forced_misclassification_rate": 0.12,
    }
    assert status["quality"]["forced_misclassification_gate"] is False


def test_dashboard_reports_migration_observation_elapsed_time(
    tmp_path: Path,
) -> None:
    started = datetime(2026, 7, 25, 10, tzinfo=UTC)
    path = tmp_path / "runtime" / "librarian" / "soak.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "schema": "chronovisor.librarian-soak.v2",
                "status": "running",
                "observation_mode": "concurrent_migration",
                "starts_at": started.isoformat(),
            }
        ),
        encoding="utf-8",
    )

    status = _soak_status(tmp_path, started + timedelta(hours=3))

    assert status["status"] == "running"
    assert status["remaining_seconds"] == 0
    assert status["elapsed_seconds"] == 3 * 3600


def test_valid_transaction_preimage_is_retained_insurance_not_quarantine(
    tmp_path: Path,
) -> None:
    from chronovisor.librarian.librarian_status import _transaction_preimages

    manifest = (
        tmp_path
        / "runtime"
        / "librarian"
        / "transaction-preimages"
        / "merge-example"
        / "manifest.json"
    )
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "created_at": "2026-07-27T00:00:00+00:00",
                "expires_at": "2026-08-03T00:00:00+00:00",
                "files": [{"path": "pages/a.md"}],
                "input_uids": ["uid-a"],
                "canonical_uid": "uid-a",
            }
        ),
        encoding="utf-8",
    )

    preimages = _transaction_preimages(tmp_path)

    assert preimages["count"] == 1
    assert preimages["invalid"] == 0
    assert preimages["recent"][0]["status"] == "retained_insurance"


@pytest.mark.parametrize(
    ("overrides", "queue", "expected"),
    [
        ({"blocked_reasons": ["gate"]}, {}, "BLOCKED"),
        ({"authority": {"active": False}}, {}, "NOT_READY"),
        ({"initial_organization_complete_at": None}, {}, "MIGRATING"),
        ({}, {"oldest_age_seconds": 8 * 86_400}, "FALLING_BEHIND"),
        ({}, {"actionable": 1}, "CATCHING_UP"),
        ({}, {"held": 1}, "STEADY_WITH_HOLDS"),
        ({}, {}, "STEADY_CLEAN"),
    ],
)
def test_librarian_status_matrix_is_deterministic(
    overrides: dict,
    queue: dict,
    expected: str,
) -> None:
    state = {
        "enabled": True,
        "authority": {"active": True},
        "blocked_reasons": [],
        "initial_organization_complete_at": "2026-07-25T00:00:00+00:00",
        "progress": {"full_sweep": {"current": True}},
        **overrides,
    }

    assert _derive_code(state, queue) == expected
