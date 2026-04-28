"""Regression tests for the ingest pipeline.

Each test pins a bug we shipped at least once. Do not delete one without
replacing it with something that catches the same class of mistake.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_wiki_mcp.ingest import (
    IngestApplyError,
    _apply_operations,
    _extract_json_array,
    _extract_page_body,
    _has_frontmatter,
    _reconcile_links,
    _strip_all_frontmatter,
)


# ---------------------------------------------------------------------------
# _extract_json_array
# ---------------------------------------------------------------------------


class TestExtractJsonArray:
    def test_plain_array(self) -> None:
        assert _extract_json_array("[]") == []
        assert _extract_json_array('[{"a":1}]') == [{"a": 1}]

    def test_preamble_stripped(self) -> None:
        assert _extract_json_array('---\n[{"type":"create"}]') == [
            {"type": "create"}
        ]
        assert _extract_json_array('Here is the plan:\n[{"x":1}]') == [{"x": 1}]

    def test_postamble_with_brackets(self) -> None:
        # rfind-based extractor would grab the [done] bracket. Lexer must not.
        out = _extract_json_array('[{"x":1}]\nNote: [done]')
        assert out == [{"x": 1}]

    def test_preamble_bracket_then_array(self) -> None:
        out = _extract_json_array(
            'Note [not json]\n[{"type":"create","filename":"ok.md"}]'
        )
        assert out == [{"type": "create", "filename": "ok.md"}]

    def test_literal_close_bracket_in_string(self) -> None:
        out = _extract_json_array('[{"summary":"see [doc]"}]')
        assert out == [{"summary": "see [doc]"}]

    def test_markdown_fence(self) -> None:
        assert _extract_json_array('```json\n[{"a":1}]\n```') == [{"a": 1}]

    def test_object_not_array(self) -> None:
        assert _extract_json_array('{"a":1}') is None

    def test_empty_or_garbage(self) -> None:
        assert _extract_json_array("") is None
        assert _extract_json_array("I cannot help with that.") is None
        assert _extract_json_array("[not json]") is None

    def test_picks_longest_valid_array(self) -> None:
        # Two arrays in output; the second is longer and is what we want.
        out = _extract_json_array('[{"a":1}]\nthen the real plan:\n[{"a":1},{"b":2}]')
        assert out == [{"a": 1}, {"b": 2}]


# ---------------------------------------------------------------------------
# Frontmatter helpers
# ---------------------------------------------------------------------------


class TestFrontmatter:
    def test_has_frontmatter_positive(self) -> None:
        assert _has_frontmatter("---\ntitle: X\nupdated: 2026-04-28\n---\nbody")

    def test_has_frontmatter_negative_no_block(self) -> None:
        assert not _has_frontmatter("body without frontmatter")

    def test_has_frontmatter_negative_no_title(self) -> None:
        assert not _has_frontmatter("---\nupdated: 2026-04-28\n---\nbody")

    def test_strip_removes_all_blocks(self) -> None:
        text = (
            "---\ntitle: A\n---\n"
            "body1\n"
            "---\ntitle: B\n---\n"
            "body2\n"
        )
        assert "title:" not in _strip_all_frontmatter(text)


# ---------------------------------------------------------------------------
# _extract_page_body — op-type contracts
# ---------------------------------------------------------------------------


class TestExtractPageBody:
    def test_create_strict_with_frontmatter(self) -> None:
        out = _extract_page_body(
            "=== NEW PAGE: foo.md ===\n"
            "---\ntitle: Foo\nupdated: 2026-04-28\n---\n"
            "\nbody.\n"
            "=== END PAGE ===",
            op_type="create",
        )
        assert out is not None and out.startswith("---\ntitle: Foo")

    def test_create_lenient_with_frontmatter(self) -> None:
        # gemma-style: drops "NEW PAGE:" prefix
        out = _extract_page_body(
            "=== career/foo.md ===\n"
            "---\ntitle: Foo\nupdated: 2026-04-28\n---\n"
            "\nbody.\n"
            "=== END PAGE ===",
            op_type="create",
        )
        assert out is not None and "title: Foo" in out

    def test_create_rejects_no_frontmatter(self) -> None:
        # Wrapper present but body has no frontmatter → must reject so we
        # never persist refusals or malformed pages.
        out = _extract_page_body(
            "=== career/foo.md ===\nNo frontmatter here\n=== END PAGE ===",
            op_type="create",
        )
        assert out is None

    def test_create_rejects_empty(self) -> None:
        assert _extract_page_body("", op_type="create") is None
        assert (
            _extract_page_body(
                "=== NEW PAGE: foo.md ===\n\n=== END PAGE ===", op_type="create"
            )
            is None
        )

    def test_update_strips_stray_frontmatter(self) -> None:
        # The original Critical bug: model emits full-page output for an
        # update, which used to be appended verbatim → multi-frontmatter.
        out = _extract_page_body(
            "=== UPDATE PAGE: foo.md ===\n"
            "---\ntitle: Foo\nupdated: 2026-04-28\n---\n"
            "\n## new section\n\nnotes.\n"
            "=== END PAGE ===",
            op_type="update",
        )
        assert out is not None
        assert "title:" not in out
        assert "## new section" in out

    def test_update_rejects_frontmatter_only(self) -> None:
        out = _extract_page_body(
            "=== UPDATE PAGE: foo.md ===\n"
            "---\ntitle: Foo\nupdated: 2026-04-28\n---\n"
            "=== END PAGE ===",
            op_type="update",
        )
        assert out is None


# ---------------------------------------------------------------------------
# _reconcile_links — code-fence safety + resolve/rewrite/unwrap
# ---------------------------------------------------------------------------


class TestReconcileLinks:
    def setup_method(self) -> None:
        self.allowed = {"foo", "bar", "switchbot-hub-3-purchase-logic"}

    def test_resolve_intact(self) -> None:
        out, s = _reconcile_links("See [[foo]] and [[bar#section|alias]].", self.allowed)
        assert out == "See [[foo]] and [[bar#section|alias]]."
        assert s["resolved"] == 2

    def test_folder_prefix_rewrite(self) -> None:
        out, s = _reconcile_links("Check [[personal/foo]] now.", self.allowed)
        assert out == "Check [[foo]] now."
        assert s["rewritten"] == 1

    def test_md_suffix_rewrite_with_anchor_and_alias(self) -> None:
        out, _s = _reconcile_links("[[bar.md#x|Bar]]", self.allowed)
        assert out == "[[bar#x|Bar]]"

    def test_unwrap_no_alias(self) -> None:
        out, s = _reconcile_links("Work at [[三菱重工]].", self.allowed)
        assert out == "Work at 三菱重工."
        assert s["unwrapped"] == 1

    def test_unwrap_with_alias(self) -> None:
        out, _s = _reconcile_links("[[ghost|display]]", self.allowed)
        assert out == "display"

    def test_fenced_code_is_untouched(self) -> None:
        # Critical: subscript / list indexing must not be eaten.
        text = "before\n```python\nx = data[[1]]\n```\nafter [[foo]] tail"
        out, s = _reconcile_links(text, self.allowed)
        assert "x = data[[1]]" in out
        assert "[[foo]]" in out  # outside fence still resolves
        assert s["resolved"] == 1
        assert s["unwrapped"] == 0

    def test_inline_code_is_untouched(self) -> None:
        text = "the regex `[[ghost]]` is a sample, but [[ghost]] is unresolved"
        out, _s = _reconcile_links(text, self.allowed)
        assert "`[[ghost]]`" in out  # inline code intact
        assert out.count("[[ghost]]") == 1  # only the inline one survived
        assert "is unresolved" in out and "ghost is unresolved" in out

    def test_frontmatter_is_untouched(self) -> None:
        # Frontmatter is data, not prose. We must not rewrite link-shaped values.
        text = "---\ntitle: [[ghost]]\n---\nbody [[foo]]"
        out, _s = _reconcile_links(text, self.allowed)
        assert "title: [[ghost]]" in out
        assert "[[foo]]" in out


# ---------------------------------------------------------------------------
# _apply_operations — fail-closed contracts
# ---------------------------------------------------------------------------


@pytest.fixture()
def isolated_wiki(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the whole package at a throw-away wiki under tmp_path.

    We monkeypatch every module-level constant the ingest pipeline reads,
    plus the IndexStore singleton, so each test gets a clean slate.
    """
    import importlib

    wiki_root = tmp_path / "wiki"
    pages = wiki_root / "pages"
    raw = wiki_root / "raw"
    system = wiki_root / "system"
    index_dir = wiki_root / ".index"
    for d in (pages, raw, system, index_dir):
        d.mkdir(parents=True, exist_ok=True)

    from llm_wiki_mcp import wiki, ingest, index_store

    monkeypatch.setattr(wiki, "WIKI_ROOT", wiki_root)
    monkeypatch.setattr(wiki, "PAGES_DIR", pages)
    monkeypatch.setattr(wiki, "RAW_DIR", raw)
    monkeypatch.setattr(wiki, "SYSTEM_DIR", system)
    monkeypatch.setattr(wiki, "INDEX_FILE", wiki_root / "index.md")
    monkeypatch.setattr(wiki, "LOG_FILE", wiki_root / "log.md")
    monkeypatch.setattr(ingest, "PAGES_DIR", pages)
    monkeypatch.setattr(ingest, "INDEX_FILE", wiki_root / "index.md")
    monkeypatch.setattr(ingest, "LOG_FILE", wiki_root / "log.md")

    # Reset the IndexStore singleton so it picks up the new paths.
    monkeypatch.setattr(index_store, "PAGES_INDEX_FILE", index_dir / "pages.json")
    monkeypatch.setattr(
        index_store, "BACKLINKS_INDEX_FILE", index_dir / "backlinks.json"
    )
    monkeypatch.setattr(index_store, "_store", None)

    return wiki_root


def _seed_page(wiki_root: Path, rel: str, body: str) -> Path:
    path = wiki_root / "pages" / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    return path


class TestApplyOperations:
    def test_create_writes_atomically(self, isolated_wiki: Path) -> None:
        ops = [
            {
                "type": "create",
                "filename": "misc/new-page.md",
                "content": "---\ntitle: T\nupdated: 2026-04-28\n---\nhello",
            }
        ]
        created, updated = _apply_operations(ops)
        assert created == ["new-page"]
        assert updated == []
        body = (isolated_wiki / "pages" / "misc" / "new-page.md").read_text()
        assert "title: T" in body and body.endswith("\n")

    def test_create_rejects_stem_collision(self, isolated_wiki: Path) -> None:
        _seed_page(
            isolated_wiki, "a/foo.md", "---\ntitle: existing\nupdated: 2026-01-01\n---\nold"
        )
        ops = [
            {
                "type": "create",
                "filename": "b/foo.md",
                "content": "---\ntitle: dup\nupdated: 2026-04-28\n---\nnew",
            }
        ]
        with pytest.raises(IngestApplyError, match="overwrite existing page_id"):
            _apply_operations(ops)

    def test_update_missing_target_fails(self, isolated_wiki: Path) -> None:
        ops = [
            {
                "type": "update",
                "filename": "ghost.md",
                "content": "addendum",
            }
        ]
        with pytest.raises(IngestApplyError, match="update target not found"):
            _apply_operations(ops)

    def test_update_appends_without_frontmatter_injection(
        self, isolated_wiki: Path
    ) -> None:
        # Even if the body slips a frontmatter block past _extract_page_body,
        # _apply_operations must not corrupt the existing page. Here we feed
        # a clean body (the contract): test that append works and there is
        # still exactly one frontmatter block.
        path = _seed_page(
            isolated_wiki,
            "career/x.md",
            "---\ntitle: X\nupdated: 2026-01-01\n---\noriginal\n",
        )
        ops = [
            {
                "type": "update",
                "filename": "x.md",
                "content": "## addendum\nnew lines",
            }
        ]
        created, updated = _apply_operations(ops)
        assert updated == ["x"]
        text = path.read_text()
        # Exactly one frontmatter delimiter pair.
        assert text.count("\n---\n") == 1
        assert "## addendum" in text
        assert "original" in text

    def test_index_store_failure_raises(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Force index_store.refresh to fail and confirm we abort instead of
        # silently destroying every link with an empty allowed_ids set.
        from llm_wiki_mcp import index_store

        def boom(*_a, **_kw):
            raise RuntimeError("simulated index failure")

        monkeypatch.setattr(index_store.IndexStore, "refresh", boom)
        ops = [
            {
                "type": "create",
                "filename": "a/x.md",
                "content": "---\ntitle: X\nupdated: 2026-04-28\n---\nbody",
            }
        ]
        with pytest.raises(IngestApplyError, match="index_store unavailable"):
            _apply_operations(ops)
