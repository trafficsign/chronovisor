"""Characterization tests for the modular ingest compatibility facade."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


INGEST_MODULES = (
    "chronovisor.ingest_schemas",
    "chronovisor.ingest_transport",
    "chronovisor.ingest_review_plan",
    "chronovisor.ingest_review",
    "chronovisor.ingest_review_authority",
    "chronovisor.ingest_review_store",
    "chronovisor.ingest_review_recovery",
    "chronovisor.ingest_review_execution",
    "chronovisor.ingest_triage",
    "chronovisor.ingest_generation",
    "chronovisor.ingest_prepare",
    "chronovisor.ingest_apply",
    "chronovisor.ingest_readback",
    "chronovisor.ingest_recovery_runtime",
    "chronovisor.ingest_review_apply",
)


@pytest.mark.parametrize("reverse", [False, True])
def test_ingest_modules_import_in_both_orders(reverse: bool) -> None:
    modules = tuple(reversed(INGEST_MODULES)) if reverse else INGEST_MODULES
    script = "\n".join(f"import {name}" for name in modules)
    script += "\nimport chronovisor.ingest\n"
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_artifact_paths_resolve_patched_pages_dir_at_call_time(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chronovisor import ingest

    pages = tmp_path / "isolated-wiki" / "pages"
    monkeypatch.setattr(ingest, "PAGES_DIR", pages)

    proposal, review = ingest._ingest_artifact_paths("a" * 64)

    expected_root = tmp_path / "isolated-wiki" / "runtime" / "ingest-frontier"
    assert proposal.parent == expected_root
    assert review.parent == expected_root
    assert str(proposal).startswith(str(tmp_path))
