from __future__ import annotations

from pathlib import Path

from chronovisor import dashboard
from chronovisor.librarian import run_shadow


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
    assert status["progress"]["uid"] == {
        "numerator": 1,
        "denominator": 1,
        "scope_generation": status["scope_generation"],
    }
