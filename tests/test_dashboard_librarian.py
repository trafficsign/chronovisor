from __future__ import annotations

from pathlib import Path

import pytest

from chronovisor import dashboard
from chronovisor.librarian import run_shadow
from chronovisor.librarian_status import _derive_code


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
        "librarian-soak",
        "librarian-recovery",
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

    from chronovisor.librarian_status import build_librarian_status

    status = build_librarian_status(tmp_path)
    assert status["state"] == "NOT_READY"
    assert status["reason_codes"]
    assert "threshold_version" in status
    assert status["progress"]["uid"] == {
        "numerator": 1,
        "denominator": 1,
        "scope_generation": status["scope_generation"],
    }


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
