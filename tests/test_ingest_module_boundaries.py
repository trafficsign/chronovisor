"""Characterization tests for the modular ingest compatibility facade."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


INGEST_MODULES = (
    "llm_wiki_mcp.ingest_schemas",
    "llm_wiki_mcp.ingest_transport",
    "llm_wiki_mcp.ingest_review_plan",
    "llm_wiki_mcp.ingest_review",
    "llm_wiki_mcp.ingest_review_authority",
    "llm_wiki_mcp.ingest_review_store",
    "llm_wiki_mcp.ingest_review_recovery",
    "llm_wiki_mcp.ingest_review_execution",
    "llm_wiki_mcp.ingest_triage",
    "llm_wiki_mcp.ingest_generation",
    "llm_wiki_mcp.ingest_prepare",
    "llm_wiki_mcp.ingest_apply",
    "llm_wiki_mcp.ingest_readback",
    "llm_wiki_mcp.ingest_recovery_runtime",
    "llm_wiki_mcp.ingest_review_apply",
)


@pytest.mark.parametrize("reverse", [False, True])
def test_ingest_modules_import_in_both_orders(reverse: bool) -> None:
    modules = tuple(reversed(INGEST_MODULES)) if reverse else INGEST_MODULES
    script = "\n".join(f"import {name}" for name in modules)
    script += "\nimport llm_wiki_mcp.ingest\n"
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
    from llm_wiki_mcp import ingest

    pages = tmp_path / "isolated-wiki" / "pages"
    monkeypatch.setattr(ingest, "PAGES_DIR", pages)

    proposal, review = ingest._ingest_artifact_paths("a" * 64)

    expected_root = tmp_path / "isolated-wiki" / "runtime" / "ingest-frontier"
    assert proposal.parent == expected_root
    assert review.parent == expected_root
    assert str(proposal).startswith(str(tmp_path))
