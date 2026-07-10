"""Tests for lint engine — focus on plan-4 tag rules.

Existing lint behavior (broken links, orphans, stale, duplicates) is
already exercised by integration paths in test_ingest.py. This file
adds direct coverage for the tag taxonomy rules introduced in plan-4.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import pytest


@pytest.fixture()
def isolated_wiki(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Throw-away wiki tree wired through every module that holds path
    constants (mirror of test_ingest.py's fixture). Kept local here so
    test_lint can run independently of test_ingest's fixture."""
    wiki_root = tmp_path / "wiki"
    pages = wiki_root / "pages"
    raw = wiki_root / "raw"
    system = wiki_root / "system"
    index_dir = wiki_root / ".index"
    for d in (pages, raw, system, index_dir):
        d.mkdir(parents=True, exist_ok=True)

    from llm_wiki_mcp import wiki, ingest, index_store, lint, tags as tags_mod

    monkeypatch.setattr(wiki, "WIKI_ROOT", wiki_root)
    monkeypatch.setattr(wiki, "PAGES_DIR", pages)
    monkeypatch.setattr(wiki, "RAW_DIR", raw)
    monkeypatch.setattr(wiki, "SYSTEM_DIR", system)
    monkeypatch.setattr(wiki, "INDEX_FILE", wiki_root / "index.md")
    monkeypatch.setattr(wiki, "LOG_FILE", wiki_root / "log.md")
    monkeypatch.setattr(ingest, "PAGES_DIR", pages)
    monkeypatch.setattr(ingest, "INDEX_FILE", wiki_root / "index.md")
    monkeypatch.setattr(ingest, "LOG_FILE", wiki_root / "log.md")
    monkeypatch.setattr(index_store, "WIKI_ROOT", wiki_root)
    monkeypatch.setattr(index_store, "PAGES_DIR", pages)
    monkeypatch.setattr(index_store, "SYSTEM_DIR", system)
    monkeypatch.setattr(index_store, "INDEX_DIR", index_dir)
    monkeypatch.setattr(index_store, "PAGES_INDEX_FILE", index_dir / "pages.json")
    monkeypatch.setattr(
        index_store, "BACKLINKS_INDEX_FILE", index_dir / "backlinks.json"
    )
    monkeypatch.setattr(index_store, "_store", None)
    monkeypatch.setattr(lint, "SYSTEM_DIR", system)
    monkeypatch.setattr(tags_mod, "SYSTEM_DIR", system)
    # Reset the lint check cache between tests; without this a
    # per-corpus-version cache hit could replay a previous test's issues.
    import llm_wiki_mcp.lint as lint_mod
    monkeypatch.setattr(lint_mod, "_CHECK_CACHE_VERSION", None)
    monkeypatch.setattr(lint_mod, "_CHECK_CACHE_RESULT", None)
    return wiki_root


def _seed(wiki_root: Path, rel: str, body: str) -> Path:
    path = wiki_root / "pages" / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    return path


def _by_type(issues: list[dict], type_: str, page_id: str | None = None) -> list[dict]:
    return [
        i for i in issues
        if i["type"] == type_ and (page_id is None or i["page"] == page_id)
    ]


def _frontier_decision(decision: str, summary: str = "reviewed exact proposal") -> dict:
    return {
        "decision": decision,
        "summary": summary,
        "tests_run": ["checked exact page hashes and diff"],
        "commit": None,
        "committed": False,
        "pushed": False,
        "risk": None,
        "notes": None,
    }


# ---------------------------------------------------------------------------
# tag_missing — high severity
# ---------------------------------------------------------------------------


class TestTagMissing:
    def test_no_tags_field_flagged(self, isolated_wiki: Path) -> None:
        from llm_wiki_mcp.lint import check

        _seed(
            isolated_wiki,
            "p.md",
            "---\ntitle: P\nupdated: 2026-05-08\n---\nbody\n",
        )
        issues = check()
        flagged = _by_type(issues, "tag_missing", "p")
        assert len(flagged) == 1
        assert flagged[0]["severity"] == "high"
        assert flagged[0]["auto_fixable"] is False

    def test_empty_tags_list_also_flagged(self, isolated_wiki: Path) -> None:
        from llm_wiki_mcp.lint import check

        _seed(
            isolated_wiki,
            "q.md",
            "---\ntitle: Q\nupdated: 2026-05-08\ntags: []\n---\nbody\n",
        )
        issues = check()
        assert _by_type(issues, "tag_missing", "q")

    def test_reference_pages_are_not_linted(self, isolated_wiki: Path) -> None:
        from llm_wiki_mcp.lint import check

        _seed(
            isolated_wiki,
            "car-spec/123.md",
            "---\ntitle: 123\nupdated: 2020-01-01\ntype: reference\n---\n[[missing]]\n",
        )
        issues = check()
        assert [issue for issue in issues if issue["page"] == "123"] == []

    def test_complete_tag_set_not_flagged(self, isolated_wiki: Path) -> None:
        from llm_wiki_mcp.lint import check

        _seed(
            isolated_wiki,
            "r.md",
            "---\ntitle: R\nupdated: 2026-05-08\n"
            "tags: [d/ai-industry, t/analysis, s/2026]\n"
            "---\nbody\n",
        )
        issues = check()
        assert _by_type(issues, "tag_missing", "r") == []
        assert _by_type(issues, "tag_invalid", "r") == []
        assert _by_type(issues, "tag_count_violation", "r") == []


# ---------------------------------------------------------------------------
# tag_invalid — medium severity, auto-fixable
# ---------------------------------------------------------------------------


class TestTagInvalid:
    def test_invalid_tag_flagged(self, isolated_wiki: Path) -> None:
        from llm_wiki_mcp.lint import check

        _seed(
            isolated_wiki,
            "p.md",
            "---\ntitle: P\nupdated: 2026-05-08\n"
            "tags: [d/ai-industry, no-prefix, t/analysis, s/2026]\n"
            "---\nbody\n",
        )
        issues = check()
        flagged = _by_type(issues, "tag_invalid", "p")
        assert len(flagged) == 1
        assert flagged[0]["auto_fixable"] is True
        assert "no-prefix" in flagged[0]["detail"]

    def test_apply_safe_fixes_drops_invalid(self, isolated_wiki: Path) -> None:
        from llm_wiki_mcp.lint import apply_safe_fixes, check

        path = _seed(
            isolated_wiki,
            "p.md",
            "---\ntitle: P\nupdated: 2026-05-08\n"
            "tags: [d/ai-industry, no-prefix, t/analysis, s/2026]\n"
            "---\nbody\n",
        )
        issues = check()
        actions = apply_safe_fixes(
            issues,
            reviewer=lambda _prompt, _schema: _frontier_decision("approved"),
        )
        assert any("no-prefix" in a for a in actions)
        text = path.read_text()
        assert "no-prefix" not in text
        # Valid ones survived.
        assert "d/ai-industry" in text
        assert "t/analysis" in text
        assert "s/2026" in text

    def test_apply_safe_fixes_preserves_correction_that_lands_before_cas(
        self,
        isolated_wiki: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from llm_wiki_mcp import lint as lint_mod

        path = _seed(
            isolated_wiki,
            "p.md",
            "---\ntitle: P\ntags: [d/ai-industry, invalid, t/analysis, s/2026]\n---\nold fact\n",
        )
        corrected = (
            "---\ntitle: P\ntags: [d/ai-industry, invalid, t/analysis, s/2026]\n"
            "---\nuser-corrected fact\n"
        )

        @contextmanager
        def correction_wins():
            path.write_text(corrected, encoding="utf-8")
            yield

        monkeypatch.setattr(lint_mod, "wiki_mutation_lock", correction_wins)
        actions = lint_mod.apply_safe_fixes(
            [{"type": "tag_invalid", "page": "p", "auto_fixable": True}],
            reviewer=lambda _prompt, _schema: _frontier_decision("approved"),
        )

        assert actions == []
        assert path.read_text(encoding="utf-8") == corrected

    def test_local_proposal_cannot_mutate_without_frontier_approval(
        self,
        isolated_wiki: Path,
    ) -> None:
        from llm_wiki_mcp.lint import apply_safe_fixes, check

        path = _seed(
            isolated_wiki,
            "p.md",
            "---\ntitle: P\ntags: [d/ai-industry, invalid, t/analysis, s/2026]\n---\nbody\n",
        )
        original = path.read_text(encoding="utf-8")
        actions = apply_safe_fixes(
            check(),
            reviewer=lambda _prompt, _schema: _frontier_decision("needs_retry"),
        )

        assert actions and actions[0].startswith("[frontier-retry]")
        assert path.read_text(encoding="utf-8") == original

    def test_frontier_rejection_is_durable_and_does_not_mutate(
        self,
        isolated_wiki: Path,
    ) -> None:
        from llm_wiki_mcp.lint import apply_safe_fixes, check

        path = _seed(
            isolated_wiki,
            "p.md",
            "---\ntitle: P\ntags: [d/ai-industry, invalid, t/analysis, s/2026]\n---\nbody\n",
        )
        original = path.read_text(encoding="utf-8")
        calls = 0

        def reject(_prompt, _schema):
            nonlocal calls
            calls += 1
            return _frontier_decision("rejected")

        issues = check()
        first = apply_safe_fixes(issues, reviewer=reject)
        second = apply_safe_fixes(
            issues,
            reviewer=lambda _prompt, _schema: (_ for _ in ()).throw(
                AssertionError("durable rejection must be reused")
            ),
        )

        assert calls == 1
        assert first[0].startswith("[frontier-rejected]")
        assert second[0].startswith("[frontier-rejected]")
        assert path.read_text(encoding="utf-8") == original
        artifact_root = isolated_wiki / "runtime" / "lint-safe-fixes"
        assert len(list((artifact_root / "proposals").glob("*.json"))) == 1
        assert len(list((artifact_root / "frontier-verdicts").glob("*.json"))) == 1

    def test_durable_frontier_approval_is_reused_after_pre_apply_crash(
        self,
        isolated_wiki: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from llm_wiki_mcp import lint as lint_mod

        path = _seed(
            isolated_wiki,
            "p.md",
            "---\ntitle: P\ntags: [d/ai-industry, invalid, t/analysis, s/2026]\n---\nbody\n",
        )
        issues = lint_mod.check()
        real_apply = lint_mod._atomic_write_if_unchanged
        monkeypatch.setattr(lint_mod, "_atomic_write_if_unchanged", lambda *_args: False)
        first = lint_mod.apply_safe_fixes(
            issues,
            reviewer=lambda _prompt, _schema: _frontier_decision("approved"),
        )
        assert first == []
        assert "invalid" in path.read_text(encoding="utf-8")

        monkeypatch.setattr(lint_mod, "_atomic_write_if_unchanged", real_apply)
        second = lint_mod.apply_safe_fixes(
            issues,
            reviewer=lambda _prompt, _schema: (_ for _ in ()).throw(
                AssertionError("durable approval must be reused")
            ),
        )

        assert second and "dropped 1 invalid tag" in second[0]
        assert "invalid" not in path.read_text(encoding="utf-8")

    def test_dry_run_is_read_only_and_does_not_call_frontier(
        self,
        isolated_wiki: Path,
    ) -> None:
        from llm_wiki_mcp.lint import apply_safe_fixes, check

        path = _seed(
            isolated_wiki,
            "p.md",
            "---\ntitle: P\ntags: [d/ai-industry, invalid, t/analysis, s/2026]\n---\nbody\n",
        )
        original = path.read_text(encoding="utf-8")
        actions = apply_safe_fixes(
            check(),
            dry_run=True,
            reviewer=lambda _prompt, _schema: (_ for _ in ()).throw(
                AssertionError("dry-run must not call frontier")
            ),
        )

        assert actions and actions[0].startswith("[dry-run]")
        assert path.read_text(encoding="utf-8") == original
        assert not (isolated_wiki / "runtime" / "lint-safe-fixes").exists()


class TestBrokenLinkFrontierGate:
    def test_retarget_requires_frontier_and_binds_exact_preimage(
        self,
        isolated_wiki: Path,
    ) -> None:
        from llm_wiki_mcp.lint import apply_safe_fixes, check

        source = _seed(
            isolated_wiki,
            "source.md",
            "---\ntitle: Source\ntags: [d/ai, t/analysis, s/2026]\n---\n[[known-pag|Known]]\n",
        )
        _seed(
            isolated_wiki,
            "known-page.md",
            "---\ntitle: Known\ntags: [d/ai, t/analysis, s/2026]\n---\nbody\n",
        )
        prompts: list[str] = []

        def approve(prompt, _schema):
            prompts.append(prompt)
            return _frontier_decision("approved")

        actions = apply_safe_fixes(check(), reviewer=approve)

        assert actions == ["[source] [[known-pag]] → [[known-page]] (1x)"]
        assert "[[known-page|Known]]" in source.read_text(encoding="utf-8")
        assert len(prompts) == 1
        assert '"expected_sha256"' in prompts[0]
        assert '"operation": "broken_link_retarget"' in prompts[0]


# ---------------------------------------------------------------------------
# tag_count_violation — medium severity, NOT auto-fixable
# ---------------------------------------------------------------------------


class TestTagCountViolation:
    def test_too_few_d_tags_flagged(self, isolated_wiki: Path) -> None:
        from llm_wiki_mcp.lint import check

        _seed(
            isolated_wiki,
            "p.md",
            "---\ntitle: P\nupdated: 2026-05-08\n"
            "tags: [t/analysis, s/2026]\n"  # missing d/
            "---\nbody\n",
        )
        issues = check()
        flagged = _by_type(issues, "tag_count_violation", "p")
        assert len(flagged) == 1
        assert flagged[0]["auto_fixable"] is False
        assert "d/" in flagged[0]["detail"]

    def test_too_many_d_tags_flagged(self, isolated_wiki: Path) -> None:
        from llm_wiki_mcp.lint import check

        _seed(
            isolated_wiki,
            "p.md",
            "---\ntitle: P\nupdated: 2026-05-08\n"
            "tags: [d/a, d/b, d/c, d/d, t/analysis, s/2026]\n"
            "---\nbody\n",
        )
        issues = check()
        flagged = _by_type(issues, "tag_count_violation", "p")
        assert len(flagged) == 1

    def test_too_many_t_tags_flagged(self, isolated_wiki: Path) -> None:
        from llm_wiki_mcp.lint import check

        _seed(
            isolated_wiki,
            "p.md",
            "---\ntitle: P\nupdated: 2026-05-08\n"
            "tags: [d/ai-industry, t/analysis, t/howto, s/2026]\n"
            "---\nbody\n",
        )
        issues = check()
        assert _by_type(issues, "tag_count_violation", "p")
