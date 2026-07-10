from __future__ import annotations

from datetime import date
from pathlib import Path

from llm_wiki_mcp import reflection


def test_build_reflection_markdown_includes_health_signals() -> None:
    doc = reflection.build_reflection_markdown(
        {
            "coverage": {"knowledge_pages": 10, "summary_coverage": 0.5},
            "memory_integrity": {"capture_rate": 0.75},
            "queues": {"duplicate_candidates": 2, "search_golden": 3},
        },
        today=date(2026, 7, 6),
    )

    assert "Memory Reflection 2026-07-06" in doc
    assert "Summary coverage: 0.500" in doc


def test_write_reflection_uses_verified_wiki_mutation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(reflection, "health_snapshot", lambda: {})

    result = reflection.write_reflection_page(output_dir=tmp_path)

    assert result["status"] == "ok"
    assert result["mutation"]["status"] == "applied"
    assert Path(result["path"]).exists()
