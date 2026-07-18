from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_wiki_mcp.research_eval import DEFAULT_FIXTURE, run_eval


def test_locked_research_holdout_passes_adoption_gate() -> None:
    result = run_eval(DEFAULT_FIXTURE)
    assert result["status"] == "pass"
    assert result["agentic_rescue_rate"] > result["baseline_rescue_rate"]
    assert result["source_backed_claim_precision"] == 1.0
    assert result["unknown_retention"] == 1.0
    assert result["waste_rate"] == 0.0


def test_holdout_rejects_non_locked_rows(tmp_path: Path) -> None:
    fixture = tmp_path / "bad.jsonl"
    fixture.write_text(json.dumps({"case_id": "x", "split": "dev"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="locked-test"):
        run_eval(fixture)
