"""Characterization tests for the modular ingest compatibility facade."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

INGEST_MODULES = (
    "chronovisor.ingest.ingest_schemas",
    "chronovisor.ingest.ingest_transport",
    "chronovisor.ingest.ingest_review_plan",
    "chronovisor.ingest.ingest_review",
    "chronovisor.ingest.ingest_review_authority",
    "chronovisor.ingest.ingest_review_store",
    "chronovisor.ingest.ingest_review_recovery",
    "chronovisor.ingest.ingest_review_execution",
    "chronovisor.ingest.ingest_triage",
    "chronovisor.ingest.ingest_generation",
    "chronovisor.ingest.ingest_prepare",
    "chronovisor.ingest.ingest_apply",
    "chronovisor.ingest.ingest_readback",
    "chronovisor.ingest.ingest_recovery_runtime",
    "chronovisor.ingest.ingest_review_apply",
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
    from chronovisor.ingest import ingest

    pages = tmp_path / "isolated-wiki" / "pages"
    monkeypatch.setattr(ingest, "PAGES_DIR", pages)

    proposal, review = ingest._ingest_artifact_paths("a" * 64)

    expected_root = tmp_path / "isolated-wiki" / "runtime" / "ingest-frontier"
    assert proposal.parent == expected_root
    assert review.parent == expected_root
    assert str(proposal).startswith(str(tmp_path))


def test_read_back_log_paths_resolve_patched_pages_dir_at_call_time(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chronovisor.ingest import ingest

    first_pages = tmp_path / "first" / "pages"
    monkeypatch.setattr(ingest, "PAGES_DIR", first_pages)
    assert ingest._read_back_failure_log() == (
        tmp_path / "first" / "runtime" / "ingest-read-back-failures.jsonl"
    )
    assert ingest._read_back_run_log() == (
        tmp_path / "first" / "runtime" / "ingest-read-back-runs.jsonl"
    )

    second_pages = tmp_path / "second" / "pages"
    monkeypatch.setattr(ingest, "PAGES_DIR", second_pages)
    assert ingest._read_back_failure_log().parent == tmp_path / "second" / "runtime"
    assert ingest._read_back_run_log().parent == tmp_path / "second" / "runtime"
