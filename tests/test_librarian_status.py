from __future__ import annotations

from pathlib import Path

from chronovisor.ingest.page_registry import PageRegistry
from chronovisor.recall.librarian_status import _observed_scope


def _page(path: Path, body: str = "body") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"---\ntitle: {path.stem}\nstatus: stable\ntype: knowledge\n---\n{body}\n",
        encoding="utf-8",
    )


def test_observed_scope_ignores_reserved_documents(tmp_path: Path) -> None:
    concept = tmp_path / "pages" / "concept-index.md"
    system_schema = tmp_path / "system" / "schema.md"
    _page(concept)
    _page(system_schema)
    for relative in (
        "pages/index.md",
        "pages/log.md",
        "pages/schema.md",
        "pages/nested/index.md",
        "pages/nested/log.md",
        "pages/nested/schema.md",
        "system/index.md",
        "system/log.md",
    ):
        _page(tmp_path / relative)
    registry = PageRegistry(tmp_path)
    registry.ensure_manifest()
    state = registry.load()
    for index, relative in enumerate(
        ("pages/index.md", "pages/nested/log.md", "system/index.md")
    ):
        state["pages"][f"legacy-reserved-{index}"] = {
            "path": relative,
            "status": "stable",
        }

    before = _observed_scope(tmp_path, state)
    _page(tmp_path / "pages" / "nested" / "index.md", "changed")
    after = _observed_scope(tmp_path, state)

    assert before["actual_total"] == 2
    assert before["collection_actual_total"] == 1
    assert before["actionable"] == 0
    assert before["missing"] == []
    assert after["scope_generation"] == before["scope_generation"]
    assert after["collection_scope_generation"] == before["collection_scope_generation"]
    assert after["actionable"] == 0
