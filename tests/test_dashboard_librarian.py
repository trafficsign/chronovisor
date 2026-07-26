from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from chronovisor import dashboard
from chronovisor.durable_state import write_sealed_json
from chronovisor.librarian import run_shadow
from chronovisor.librarian_status import (
    _derive_code,
    _library_evidence_status,
    _soak_status,
)


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


def test_fast_status_payload_can_be_built_from_shadow_state(tmp_path: Path) -> None:
    page = tmp_path / "pages" / "alpha.md"
    page.parent.mkdir(parents=True)
    page.write_text(
        "---\ntitle: Alpha\nupdated: 2026-07-25\ntags: [d/ai]\n---\n\n# Alpha\n",
        encoding="utf-8",
    )
    run_shadow(root=tmp_path, full_sweep=True)
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

    from chronovisor.librarian_status import build_librarian_status

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


def test_status_overlays_latest_locked_calibration_quality(tmp_path: Path) -> None:
    page = tmp_path / "pages" / "alpha.md"
    page.parent.mkdir(parents=True)
    page.write_text(
        "---\ntitle: Alpha\nupdated: 2026-07-25\ntags: [d/ai]\n---\n\n# Alpha\n",
        encoding="utf-8",
    )
    run_shadow(root=tmp_path, full_sweep=True)
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

    from chronovisor.librarian_status import build_librarian_status

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
