from __future__ import annotations

import json
from pathlib import Path

from chronovisor.recall import recall_runtime
from chronovisor.recall.recall_runtime import (
    RecallPolicy,
    RecallRequest,
    render_output,
    run_recall,
)
from chronovisor.ingest import page_mutation
from chronovisor.ops import state_register


def test_state_register_context_is_injected_for_codex(monkeypatch) -> None:
    monkeypatch.setattr(
        recall_runtime, "should_inject_state", lambda host: host == "codex"
    )
    monkeypatch.setattr(
        recall_runtime,
        "format_state_context",
        lambda *, host, cwd: (
            "[WORKING_MEMORY]\ncurrent project state\n[/WORKING_MEMORY]"
        ),
    )
    request = RecallRequest(
        host="codex", event="UserPromptSubmit", prompt="うん", cwd="/tmp"
    )
    result = run_recall(
        request,
        RecallPolicy(judge_mode="off", log_decisions=False),
        perform_search=False,
    )

    assert result.decision == "none"
    assert "current project state" in result.context
    assert "state register injected" in result.reasons

    output = json.loads(render_output(result, "codex"))
    assert output["hookSpecificOutput"]["additionalContext"] == result.context


def test_format_state_context_marks_stale_state(tmp_path: Path) -> None:
    path = tmp_path / "current-state.md"
    path.write_text(
        "---\ntitle: Current State\nupdated: 2026-04-17\n---\n# Current State\n\nold body",
        encoding="utf-8",
    )

    context = state_register.format_state_context(host="codex", cwd="/tmp", path=path)

    assert "updated=2026-04-17" in context
    assert "stale=true" in context
    assert "old body" in context


def test_format_state_context_includes_only_allowlisted_core_memory(
    tmp_path: Path,
) -> None:
    for page_id, body in (
        ("current-state", "current project"),
        ("user-profile", "prefers local models"),
        ("lessons-learned", "verify before completion"),
        ("arbitrary-page", "must not be injected"),
    ):
        (tmp_path / f"{page_id}.md").write_text(
            f"---\ntitle: {page_id}\nupdated: 2026-07-17\n---\n# {page_id}\n\n{body}",
            encoding="utf-8",
        )

    context = state_register.format_state_context(
        host="codex",
        path=tmp_path / "current-state.md",
        max_chars=1200,
    )

    assert "current project" in context
    assert "prefers local models" in context
    assert "verify before completion" in context
    assert "must not be injected" not in context
    assert len(context) <= 1200


def test_refresh_state_register_writes_recent_pages(
    tmp_path: Path, monkeypatch
) -> None:
    path = tmp_path / "current-state.md"

    class FakeStore:
        def refresh(self) -> None:
            pass

        def meta(self, page_id: str):
            return {
                "page_id": page_id,
                "title": "Recent Page",
                "summary": "Summary",
                "updated": "2026-07-06",
                "page_type": "knowledge",
            }

        def all_pages_meta(self, include_system: bool = False):
            return []

    monkeypatch.setattr("chronovisor.search.index_store.get_store", lambda: FakeStore())

    payload = state_register.refresh_state_register(["recent-page"], path=path)

    assert payload["pages"] == ["recent-page"]
    assert payload["mutation"]["status"] == "applied"
    assert "[[recent-page]]" in path.read_text(encoding="utf-8")


def test_refresh_preserves_approved_current_state_correction(
    tmp_path: Path,
    monkeypatch,
) -> None:
    pages = tmp_path / "pages"
    system = tmp_path / "system"
    pages.mkdir()
    system.mkdir()
    path = system / "current-state.md"
    path.write_text(
        "---\ntitle: Current State\nupdated: 2026-07-10\ntype: state\n"
        "summary: Working memory.\n---\n# Current State\n\n"
        "- [[machine]] — Machine (2026-07-10) — Installed RAM is 16GB.\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(page_mutation, "PAGES_DIR", pages)
    monkeypatch.setattr(page_mutation, "SYSTEM_DIR", system)
    monkeypatch.setattr(
        page_mutation,
        "CHRONOVISOR_MUTATION_LOCK",
        tmp_path / "runtime" / "wiki-mutation.lock",
    )
    monkeypatch.setattr(page_mutation, "find_page", lambda _page_id: None)
    prepared = page_mutation.prepare_page_mutation(
        "current-state",
        [
            {
                "old_text": "Installed RAM is 16GB.",
                "new_text": "Installed RAM is 32GB.",
            }
        ],
        correction_id="corr-current-state-ram",
    )
    assert page_mutation.apply_prepared_mutations([prepared])["status"] == "applied"

    class FakeStore:
        def refresh(self) -> None:
            pass

        def meta(self, page_id: str):
            return {
                "page_id": page_id,
                "title": "Machine",
                # Simulate the stale generated source that caused the original
                # current-state claim. The durable constraint must rebase it.
                "summary": "Installed RAM is 16GB.",
                "updated": "2026-07-10",
                "page_type": "knowledge",
            }

        def all_pages_meta(self, include_system: bool = False):
            return []

    monkeypatch.setattr("chronovisor.search.index_store.get_store", lambda: FakeStore())
    payload = state_register.refresh_state_register(["machine"], path=path)
    written = path.read_text(encoding="utf-8")

    assert payload["status"] == "ok"
    assert payload["mutation"]["status"] == "applied"
    assert "Installed RAM is 16GB." not in written
    assert "Installed RAM is 32GB." in written
    assert "applied_corrections: [corr-current-state-ram]" in written
    assert (
        payload["mutation"]["correction_constraints"]["current-state"][0][
            "correction_id"
        ]
        == "corr-current-state-ram"
    )


def test_refresh_state_register_skips_placeholder_pages(
    tmp_path: Path, monkeypatch
) -> None:
    path = tmp_path / "current-state.md"
    real = tmp_path / "real.md"
    real.write_text(
        "---\ntitle: Real\nupdated: 2026-07-06\nsummary: Useful\n---\nBody\n",
        encoding="utf-8",
    )

    class FakeStore:
        def refresh(self) -> None:
            pass

        def meta(self, page_id: str):
            if page_id == "baz":
                return {
                    "page_id": "baz",
                    "title": "Baz",
                    "summary": "body",
                    "updated": "2026-04-28",
                    "page_type": "knowledge",
                }
            return {
                "page_id": page_id,
                "title": "Real",
                "summary": "Useful",
                "updated": "2026-07-06",
                "page_type": "knowledge",
                "path": str(real),
            }

        def all_pages_meta(self, include_system: bool = False):
            return [{"page_id": "baz"}, {"page_id": "real"}]

    monkeypatch.setattr("chronovisor.search.index_store.get_store", lambda: FakeStore())

    payload = state_register.refresh_state_register(path=path)

    assert payload["pages"] == ["real"]
    assert "[[baz]]" not in path.read_text(encoding="utf-8")


def test_refresh_state_register_skips_deprecated_pages(
    tmp_path: Path, monkeypatch
) -> None:
    path = tmp_path / "current-state.md"
    deprecated = tmp_path / "old.md"
    active = tmp_path / "active.md"
    deprecated.write_text(
        "---\ntitle: Old\nupdated: 2026-07-06\n---\nOld\n", encoding="utf-8"
    )
    active.write_text(
        "---\ntitle: Active\nupdated: 2026-07-06\n---\nActive\n", encoding="utf-8"
    )

    class FakeStore:
        def refresh(self) -> None:
            pass

        def meta(self, page_id: str):
            if page_id == "old":
                return {
                    "page_id": "old",
                    "title": "Old",
                    "summary": "Deprecated duplicate",
                    "updated": "2026-07-06",
                    "status": "deprecated",
                    "page_type": "knowledge",
                    "path": str(deprecated),
                }
            return {
                "page_id": "active",
                "title": "Active",
                "summary": "Current",
                "updated": "2026-07-06",
                "status": "active",
                "page_type": "knowledge",
                "path": str(active),
            }

        def all_pages_meta(self, include_system: bool = False):
            return [{"page_id": "old"}, {"page_id": "active"}]

    monkeypatch.setattr("chronovisor.search.index_store.get_store", lambda: FakeStore())

    payload = state_register.refresh_state_register(path=path)

    assert payload["pages"] == ["active"]
    assert "[[old]]" not in path.read_text(encoding="utf-8")
