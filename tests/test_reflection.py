from __future__ import annotations

from datetime import date
from pathlib import Path

from chronovisor.ops import reflection


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
    assert "status: stable" in doc
    assert "description: Sleep-cycle reflection" in doc
    assert "summary:" not in doc
    assert "[[retention]]" not in doc


def test_write_reflection_uses_verified_wiki_mutation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(reflection, "health_snapshot", lambda: {})

    result = reflection.write_reflection_page(output_dir=tmp_path)

    assert result["status"] == "ok"
    assert result["mutation"]["status"] == "applied"
    assert Path(result["path"]).exists()


def test_write_reflection_updates_existing_page_id_without_relocation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    pages_dir = tmp_path / "pages"
    existing_dir = pages_dir / "chronovisor"
    preferred_dir = pages_dir / "insights"
    existing_dir.mkdir(parents=True)
    page_id = f"memory-reflection-{date.today().isoformat()}"
    existing = existing_dir / f"{page_id}.md"
    existing.write_text(
        "---\ntitle: Old Reflection\nstatus: stable\ntype: knowledge\n---\nold reflection\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(reflection, "health_snapshot", lambda: {})
    monkeypatch.setattr(reflection, "INSIGHTS_DIR", preferred_dir)
    monkeypatch.setattr(
        reflection, "find_page", lambda value: existing if value == page_id else None
    )

    result = reflection.write_reflection_page()

    assert result["status"] == "ok"
    assert result["path"] == str(existing)
    assert "Memory Reflection" in existing.read_text(encoding="utf-8")
    assert not (preferred_dir / f"{page_id}.md").exists()
