"""Tests for lint engine — focus on plan-4 tag rules.

Existing lint behavior (broken links, orphans, stale, duplicates) is
already exercised by integration paths in test_ingest.py. This file
adds direct coverage for the tag taxonomy rules introduced in plan-4.
"""

from __future__ import annotations

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
        actions = apply_safe_fixes(issues)
        assert any("no-prefix" in a for a in actions)
        text = path.read_text()
        assert "no-prefix" not in text
        # Valid ones survived.
        assert "d/ai-industry" in text
        assert "t/analysis" in text
        assert "s/2026" in text


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
