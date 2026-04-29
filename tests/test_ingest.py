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

    def test_truncated_outer_with_inner_keywords_returns_none(self) -> None:
        # Historical bug: when triage runs out of tokens mid-array, the
        # inner ``"keywords": [...]`` list is the longest *parseable* array
        # and used to be returned, silently routing truncated triage as
        # either "nothing wiki-worthy" (empty `[]`) or "schema invalid"
        # (string list). Both are wrong — the LLM had more to say.
        truncated_string_inner = (
            '[\n'
            '  {"type": "create", "filename": "a.md", "title": "A", '
            '"keywords": ["x", "y"], "summary": "..."},\n'
            '  {"type": "create", "filename": "b.md", "title": "B", '
            '"keywords": ['
        )
        assert _extract_json_array(truncated_string_inner) is None

        truncated_empty_inner = (
            '[\n'
            '  {"type": "create", "filename": "a.md", '
            '"keywords": []},\n'
            '  {"type": "create", "filename": "b.md", '
        )
        assert _extract_json_array(truncated_empty_inner) is None

    def test_truncated_outer_with_inner_dict_array_recovers(self) -> None:
        # Edge case: the outer array is broken but a complete dict-array
        # appears later. This *does* fit the contract, so accept it.
        text = 'broken [stuff\nlater: [{"type": "create", "filename": "a.md"}]'
        assert _extract_json_array(text) == [
            {"type": "create", "filename": "a.md"}
        ]


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

    def test_create_truncated_no_close_recovers(self) -> None:
        # The most common Ollama failure mode: wrapper opened, full
        # frontmatter + content emitted, but model ran out of tokens
        # before "=== END PAGE ===". Without the truncation fallback all
        # three earlier patterns fail and the body is silently discarded.
        out = _extract_page_body(
            "=== NEW PAGE: personal/smoking-habit-analysis.md ===\n"
            "---\ntitle: Smoking Habit Analysis\nupdated: 2026-04-29\n---\n"
            "\n# 概要\n\nニコチンの半減期は短い",
            op_type="create",
        )
        assert out is not None
        assert out.startswith("---\ntitle: Smoking Habit")
        assert "ニコチンの半減期" in out

    def test_create_truncated_partial_close_fence_stripped(self) -> None:
        # Output ends with a partially-emitted close fence ("=== EN").
        # We must strip it rather than letting it leak into the body.
        out = _extract_page_body(
            "=== NEW PAGE: foo.md ===\n"
            "---\ntitle: Foo\nupdated: 2026-04-28\n---\n"
            "\nbody text.\n=== EN",
            op_type="create",
        )
        assert out is not None
        assert "=== EN" not in out
        assert out.rstrip().endswith("body text.")

    def test_create_new_keyword_dropped_with_truncation(self) -> None:
        # gemma sometimes drops the "NEW" keyword and uses the filename
        # as the wrapper label. Combined with truncation, all three
        # earlier patterns fail. Truncation fallback peels generic
        # "=== ... ===" wrappers so this still recovers.
        out = _extract_page_body(
            "=== user-profile-background PAGE: user-profile-background ===\n"
            "---\ntitle: User Profile\nupdated: 2026-04-29\n---\n"
            "\nbody",
            op_type="create",
        )
        assert out is not None
        assert "title: User Profile" in out

    def test_update_truncated_no_close_recovers(self) -> None:
        # Same truncation pattern for updates: no frontmatter expected,
        # so the contract check passes any non-empty body.
        out = _extract_page_body(
            "=== UPDATE PAGE: foo.md ===\n"
            "\n## new section\n\nnotes...",
            op_type="update",
        )
        assert out is not None
        assert "## new section" in out
        assert "title:" not in out

    def test_create_truncated_broken_frontmatter_still_rejected(self) -> None:
        # Truncation BEFORE the closing "---" of the frontmatter cannot
        # be recovered: we'd persist a page with no proper frontmatter
        # block. The op_type contract check must still reject.
        out = _extract_page_body(
            "=== NEW PAGE: foo.md ===\n"
            "---\ntitle: Foo\nupdated: 2026",
            op_type="create",
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

    Every module that holds a copy of a wiki path constant gets patched.
    Without this an IndexStore.refresh() during the test would scan the
    real ``~/.wiki`` corpus.
    """
    wiki_root = tmp_path / "wiki"
    pages = wiki_root / "pages"
    raw = wiki_root / "raw"
    system = wiki_root / "system"
    index_dir = wiki_root / ".index"
    for d in (pages, raw, system, index_dir):
        d.mkdir(parents=True, exist_ok=True)

    from llm_wiki_mcp import wiki, ingest, index_store, orchestrator

    monkeypatch.setattr(wiki, "WIKI_ROOT", wiki_root)
    monkeypatch.setattr(wiki, "PAGES_DIR", pages)
    monkeypatch.setattr(wiki, "RAW_DIR", raw)
    monkeypatch.setattr(wiki, "SYSTEM_DIR", system)
    monkeypatch.setattr(wiki, "INDEX_FILE", wiki_root / "index.md")
    monkeypatch.setattr(wiki, "LOG_FILE", wiki_root / "log.md")
    monkeypatch.setattr(ingest, "PAGES_DIR", pages)
    monkeypatch.setattr(ingest, "INDEX_FILE", wiki_root / "index.md")
    monkeypatch.setattr(ingest, "LOG_FILE", wiki_root / "log.md")
    monkeypatch.setattr(orchestrator, "RAW_DIR", raw)
    monkeypatch.setattr(orchestrator, "WIKI_ROOT", wiki_root)
    monkeypatch.setattr(orchestrator, "STATE_FILE", wiki_root / ".orchestrator_state.json")

    # IndexStore reads its paths from module globals AND from wiki.PAGES_DIR
    # internally; patch both layers.
    monkeypatch.setattr(index_store, "WIKI_ROOT", wiki_root)
    monkeypatch.setattr(index_store, "PAGES_DIR", pages)
    monkeypatch.setattr(index_store, "SYSTEM_DIR", system)
    monkeypatch.setattr(index_store, "INDEX_DIR", index_dir)
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


# ---------------------------------------------------------------------------
# Path-traversal sanitization (R2-High)
# ---------------------------------------------------------------------------


class TestSafeResolvePagePath:
    def test_relative_ok(self, isolated_wiki: Path) -> None:
        from llm_wiki_mcp.ingest import _safe_resolve_page_path

        out = _safe_resolve_page_path("ai/foo.md")
        pages = isolated_wiki / "pages"
        assert out == (pages / "ai" / "foo.md").resolve()

    def test_relative_without_md_suffix(self, isolated_wiki: Path) -> None:
        from llm_wiki_mcp.ingest import _safe_resolve_page_path

        out = _safe_resolve_page_path("ai/foo")
        assert out.name == "foo.md"

    def test_absolute_rejected(self, isolated_wiki: Path) -> None:
        from llm_wiki_mcp.ingest import _safe_resolve_page_path

        with pytest.raises(IngestApplyError, match="absolute filename"):
            _safe_resolve_page_path("/etc/passwd")

    def test_parent_traversal_rejected(self, isolated_wiki: Path) -> None:
        from llm_wiki_mcp.ingest import _safe_resolve_page_path

        with pytest.raises(IngestApplyError, match="parent-traversal"):
            _safe_resolve_page_path("../../etc/passwd.md")
        with pytest.raises(IngestApplyError, match="parent-traversal"):
            _safe_resolve_page_path("ai/../../../etc/passwd.md")

    def test_empty_or_dot_md_rejected(self, isolated_wiki: Path) -> None:
        from llm_wiki_mcp.ingest import _safe_resolve_page_path

        with pytest.raises(IngestApplyError):
            _safe_resolve_page_path("")
        with pytest.raises(IngestApplyError):
            _safe_resolve_page_path("   ")

    def test_apply_rejects_traversal_before_writing(
        self, isolated_wiki: Path
    ) -> None:
        # Even a single traversal op poisons the whole batch — nothing writes.
        good = {
            "type": "create",
            "filename": "ok/safe.md",
            "content": "---\ntitle: T\nupdated: 2026-04-28\n---\nbody",
        }
        evil = {
            "type": "create",
            "filename": "../../tmp/escape.md",
            "content": "---\ntitle: E\nupdated: 2026-04-28\n---\nx",
        }
        with pytest.raises(IngestApplyError, match="parent-traversal"):
            _apply_operations([good, evil])
        # Confirm neither file was created.
        pages = isolated_wiki / "pages"
        assert not (pages / "ok" / "safe.md").exists()


# ---------------------------------------------------------------------------
# Update body — partial frontmatter rejection (R2-Medium)
# ---------------------------------------------------------------------------


class TestUpdatePartialFrontmatter:
    def test_unclosed_frontmatter_in_update_rejected(self) -> None:
        # Opening `---` with no closing — _strip_all_frontmatter can't
        # remove it, so the contract demands we reject rather than append.
        out = _extract_page_body(
            "=== UPDATE PAGE: foo.md ===\n"
            "---\ntitle: X\nupdated: 2026-04-28\n"
            "extra body but no closing fence\n"
            "=== END PAGE ===",
            op_type="update",
        )
        assert out is None

    def test_closed_frontmatter_then_body_in_update(self) -> None:
        # Closed FM followed by real body → strip the FM, keep the body.
        out = _extract_page_body(
            "=== UPDATE PAGE: foo.md ===\n"
            "---\ntitle: X\nupdated: 2026-04-28\n---\n"
            "## section\nnotes\n"
            "=== END PAGE ===",
            op_type="update",
        )
        assert out is not None
        assert "title:" not in out
        assert "## section" in out


# ---------------------------------------------------------------------------
# Unclosed fenced code preservation (R2-Medium)
# ---------------------------------------------------------------------------


class TestUnclosedFence:
    def test_unclosed_fence_protects_trailing_subscript(self) -> None:
        # Truncated LLM output: the fence opens but never closes. Everything
        # after the opener must be treated as code so we don't eat
        # `data[[1]]` -> `data1`.
        text = (
            "intro [[foo]] mid\n"
            "```python\n"
            "x = data[[1]]\n"
            "y = also[[2]]\n"
            # NOTE: no closing fence
        )
        out, stats = _reconcile_links(text, {"foo"})
        assert "x = data[[1]]" in out
        assert "y = also[[2]]" in out
        assert "[[foo]]" in out
        assert stats["resolved"] == 1
        assert stats["unwrapped"] == 0


# ---------------------------------------------------------------------------
# run_ingest integration: partial generate failure (R2-Critical)
# ---------------------------------------------------------------------------


class TestRunIngestPartialFailure:
    def test_partial_generate_does_not_mutate_disk(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Critical bug: in the prior version, a 2-of-3-ops generate failure
        wrote the 2 successful pages to disk, the 3rd raw stayed pending,
        and the next tick re-tried with the same page_ids → stem collision
        or duplicate append. Verify the apply step is now skipped entirely
        when ANY op failed, leaving the wiki untouched."""

        from llm_wiki_mcp import ingest, jobs

        # Stub triage → returns 3 ops.
        plan = [
            {"type": "create", "filename": f"misc/p{i}.md", "title": f"P{i}"}
            for i in range(3)
        ]
        monkeypatch.setattr(ingest, "_triage", lambda _content: plan)

        # Stub generate: succeed for op 0 and 1, fail for op 2.
        def fake_generate(op: dict, _raw: str) -> dict | None:
            if op["filename"].endswith("p2.md"):
                return None
            return {
                "type": "create",
                "filename": op["filename"],
                "content": (
                    "---\ntitle: X\nupdated: 2026-04-28\n---\nbody"
                ),
            }

        monkeypatch.setattr(ingest, "_generate_one", fake_generate)
        monkeypatch.setattr(ingest, "is_available", lambda: True)

        on_complete_called = []
        on_finally_calls = []

        job = jobs.job_store.create(processor="ollama")
        ingest.run_ingest(
            "raw content",
            job.job_id,
            on_complete=lambda: on_complete_called.append(True),
            on_finally=lambda failed, triage_failed: on_finally_calls.append(
                {"failed": failed, "triage_failed": triage_failed}
            ),
        )

        # No on_complete → raws stay pending.
        assert on_complete_called == []
        # on_finally fired exactly once with failed=True, triage_failed=False.
        assert on_finally_calls == [{"failed": True, "triage_failed": False}]
        # Job marked failed; no pages claimed as written.
        finished = jobs.job_store.get(job.job_id)
        assert finished.status == jobs.JobStatus.FAILED
        assert finished.pages_created == []
        assert finished.pages_updated == []
        # Disk: NEITHER p0 nor p1 should exist (apply was skipped).
        pages = isolated_wiki / "pages"
        assert not (pages / "misc" / "p0.md").exists()
        assert not (pages / "misc" / "p1.md").exists()
        assert not (pages / "misc" / "p2.md").exists()


# ---------------------------------------------------------------------------
# Orchestrator (R2-High)
# ---------------------------------------------------------------------------


class TestOrchestrator:
    def test_reset_stale_lock_clears_pending_sentinel(
        self, isolated_wiki: Path
    ) -> None:
        from llm_wiki_mcp import orchestrator

        # Simulate a server crash mid-`run_pending_ingest`: the sentinel was
        # written but the real job_id never replaced it.
        state = orchestrator._load_state()
        state["current_job_id"] = "__pending__"
        orchestrator._save_state(state)

        orchestrator.reset_stale_lock()
        assert orchestrator._load_state()["current_job_id"] is None

    def test_reset_stale_lock_clears_unknown_job(
        self, isolated_wiki: Path
    ) -> None:
        from llm_wiki_mcp import orchestrator

        state = orchestrator._load_state()
        state["current_job_id"] = "no-such-job-12345"
        orchestrator._save_state(state)

        orchestrator.reset_stale_lock()
        # job_store is in-memory → after restart, the id is unknown → cleared.
        assert orchestrator._load_state()["current_job_id"] is None

    def test_reset_stale_lock_keeps_known_job(
        self, isolated_wiki: Path
    ) -> None:
        from llm_wiki_mcp import orchestrator
        from llm_wiki_mcp import jobs

        job = jobs.job_store.create(processor="ollama")
        try:
            state = orchestrator._load_state()
            state["current_job_id"] = job.job_id
            orchestrator._save_state(state)

            orchestrator.reset_stale_lock()
            assert orchestrator._load_state()["current_job_id"] == job.job_id
        finally:
            # Cleanup so other tests aren't polluted.
            jobs.job_store._jobs.pop(job.job_id, None)

    def test_run_pending_ingest_refuses_double_trigger(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two run_pending_ingest calls back-to-back must not both spawn a
        worker. The second sees the in-flight marker and refuses."""
        from llm_wiki_mcp import orchestrator

        # Make 5 fake raws so should_ingest() fires.
        for i in range(5):
            (isolated_wiki / "raw" / f"r{i}.md").write_text("body")

        # Stub start_ingest so it doesn't actually try to run anything.
        captured = {"calls": 0}

        def fake_start(content: str, on_complete=None, on_finally=None) -> str:
            captured["calls"] += 1
            return "stub-job-id-1"

        from llm_wiki_mcp import ingest as ingest_mod

        monkeypatch.setattr(ingest_mod, "start_ingest", fake_start)
        monkeypatch.setattr(orchestrator, "is_available", lambda: True)

        first = orchestrator.run_pending_ingest()
        second = orchestrator.run_pending_ingest()

        assert first["triggered"] is True
        assert first["job_id"] == "stub-job-id-1"
        assert second["triggered"] is False
        assert "already in flight" in second["reason"]
        assert captured["calls"] == 1

    def test_release_lock_only_counts_triage_failures(
        self, isolated_wiki: Path
    ) -> None:
        from llm_wiki_mcp import orchestrator

        # Non-triage failure (e.g. apply error, generate parse) must NOT
        # advance the dead-letter counter.
        orchestrator._release_lock(failed=True, triage_failed=False)
        assert orchestrator._load_state()["triage_failure_count"] == 0

        # Triage failure does advance.
        orchestrator._release_lock(failed=True, triage_failed=True)
        orchestrator._release_lock(failed=True, triage_failed=True)
        assert orchestrator._load_state()["triage_failure_count"] == 2

        # Success resets.
        orchestrator._release_lock(failed=False)
        assert orchestrator._load_state()["triage_failure_count"] == 0


# ---------------------------------------------------------------------------
# Triage plan schema validation (R3-High)
# ---------------------------------------------------------------------------


class TestTriagePlanSchema:
    def test_valid_plan_passes_through(self) -> None:
        from llm_wiki_mcp.ingest import _validate_triage_plan

        plan = [
            {"type": "create", "filename": "ai/foo.md", "title": "Foo"},
            {"type": "update", "filename": "bar.md"},
        ]
        assert _validate_triage_plan(plan) == plan

    def test_empty_plan_passes(self) -> None:
        from llm_wiki_mcp.ingest import _validate_triage_plan

        assert _validate_triage_plan([]) == []

    def test_string_entry_rejected(self) -> None:
        from llm_wiki_mcp.ingest import _validate_triage_plan

        assert _validate_triage_plan(["not a dict"]) is None

    def test_unknown_type_rejected(self) -> None:
        from llm_wiki_mcp.ingest import _validate_triage_plan

        assert (
            _validate_triage_plan(
                [{"type": "delete", "filename": "x.md"}]
            )
            is None
        )

    def test_missing_filename_rejected(self) -> None:
        from llm_wiki_mcp.ingest import _validate_triage_plan

        assert _validate_triage_plan([{"type": "create"}]) is None
        assert _validate_triage_plan([{"type": "create", "filename": ""}]) is None
        assert (
            _validate_triage_plan([{"type": "create", "filename": "   "}]) is None
        )

    def test_non_string_filename_rejected(self) -> None:
        from llm_wiki_mcp.ingest import _validate_triage_plan

        assert (
            _validate_triage_plan([{"type": "create", "filename": 123}]) is None
        )


# ---------------------------------------------------------------------------
# Apply prepare phase: all-or-nothing collision (R3-Critical)
# ---------------------------------------------------------------------------


class TestApplyPreparePhase:
    def test_collision_rejects_whole_batch_before_writing(
        self, isolated_wiki: Path
    ) -> None:
        """The previous flow wrote op[0] before the prepare-phase check
        on op[1] failed. After the fix, every op is validated before any
        disk write so a single collision aborts the entire batch."""
        # Seed an existing page so op[1] collides.
        existing = isolated_wiki / "pages" / "a" / "blocking.md"
        existing.parent.mkdir(parents=True, exist_ok=True)
        existing.write_text(
            "---\ntitle: existing\nupdated: 2026-01-01\n---\noriginal\n"
        )

        ops = [
            {
                "type": "create",
                "filename": "fresh/safe-page.md",
                "content": "---\ntitle: Fresh\nupdated: 2026-04-28\n---\nbody",
            },
            {
                "type": "create",
                "filename": "other/blocking.md",  # different folder, same stem
                "content": "---\ntitle: Dup\nupdated: 2026-04-28\n---\nx",
            },
        ]
        with pytest.raises(IngestApplyError, match="overwrite existing"):
            _apply_operations(ops)

        # The first (would-have-succeeded) op must NOT be on disk.
        assert not (isolated_wiki / "pages" / "fresh" / "safe-page.md").exists()
        # The blocking page is untouched.
        assert (
            existing.read_text()
            == "---\ntitle: existing\nupdated: 2026-01-01\n---\noriginal\n"
        )

    def test_duplicate_page_id_within_batch_rejected(
        self, isolated_wiki: Path
    ) -> None:
        ops = [
            {
                "type": "create",
                "filename": "a/dup.md",
                "content": "---\ntitle: A\nupdated: 2026-04-28\n---\nbody1",
            },
            {
                "type": "create",
                "filename": "b/dup.md",
                "content": "---\ntitle: B\nupdated: 2026-04-28\n---\nbody2",
            },
        ]
        with pytest.raises(IngestApplyError, match="duplicate page_id"):
            _apply_operations(ops)
        # Neither file written.
        assert not (isolated_wiki / "pages" / "a" / "dup.md").exists()
        assert not (isolated_wiki / "pages" / "b" / "dup.md").exists()

    def test_rollback_on_write_failure_restores_previous_state(
        self,
        isolated_wiki: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Force atomic_write to fail on the second op and verify the first
        op's effect is rolled back."""
        from llm_wiki_mcp import ingest as ingest_mod

        # Seed a page so op[0] is a real update we can roll back.
        target = isolated_wiki / "pages" / "x" / "page.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            "---\ntitle: X\nupdated: 2026-01-01\n---\noriginal body\n"
        )
        original = target.read_text()

        # First call: real write (the update succeeds). Second call: explode
        # (the create's write fails). Subsequent calls succeed so the
        # rollback path can actually run.
        from llm_wiki_mcp import link_fix
        real_write = link_fix.atomic_write
        call_count = {"n": 0}

        def flaky_write(path, content):
            call_count["n"] += 1
            if call_count["n"] == 2:
                raise OSError("simulated disk full")
            real_write(path, content)

        monkeypatch.setattr(link_fix, "atomic_write", flaky_write)

        ops = [
            {
                "type": "update",
                "filename": "page.md",
                "content": "addendum block 1",
            },
            {
                "type": "create",
                "filename": "y/new.md",
                "content": "---\ntitle: Y\nupdated: 2026-04-28\n---\nbody",
            },
        ]
        with pytest.raises(IngestApplyError, match="apply write failed"):
            _apply_operations(ops)

        # Update was rolled back to the original text.
        assert target.read_text() == original
        # Create never wrote (write failed before append).
        assert not (isolated_wiki / "pages" / "y" / "new.md").exists()

    def test_casefold_collision_detected(self, isolated_wiki: Path) -> None:
        """``Foo.md`` and ``foo.md`` resolve to the same inode on
        case-insensitive macOS filesystems. We catch that even when
        ``find_page`` (strict-case) wouldn't."""
        existing = isolated_wiki / "pages" / "a" / "Foo.md"
        existing.parent.mkdir(parents=True, exist_ok=True)
        existing.write_text(
            "---\ntitle: cased\nupdated: 2026-01-01\n---\nbody\n"
        )
        ops = [
            {
                "type": "create",
                "filename": "b/foo.md",  # lowercase variant
                "content": "---\ntitle: dup\nupdated: 2026-04-28\n---\nx",
            }
        ]
        with pytest.raises(IngestApplyError, match="overwrite existing"):
            _apply_operations(ops)


# ---------------------------------------------------------------------------
# wiki_ingest now persists raw + uses orchestrator (R3-Medium)
# ---------------------------------------------------------------------------


class TestWikiIngestRouting:
    def test_wiki_ingest_writes_raw_and_consults_orchestrator(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from llm_wiki_mcp import server, orchestrator
        # The MCP tool object wraps the function; call .fn.
        tool_fn = server.wiki_ingest.fn if hasattr(server.wiki_ingest, "fn") else server.wiki_ingest

        # Patch RAW_DIR in server (it grabbed the path at import time).
        monkeypatch.setattr(server, "RAW_DIR", isolated_wiki / "raw")

        # Stub the orchestrator path so we don't actually start ingest.
        captured = {"called": False, "force": None}

        def fake_run(force: bool = False) -> dict:
            captured["called"] = True
            captured["force"] = force
            return {"triggered": False, "reason": "test stub"}

        monkeypatch.setattr(orchestrator, "run_pending_ingest", fake_run)

        result = tool_fn("hello world content")
        # Must have written exactly one new raw file with the supplied content.
        raws = list((isolated_wiki / "raw").glob("*.md"))
        assert len(raws) == 1
        assert raws[0].read_text() == "hello world content"
        # Must have consulted the orchestrator with force=True (default).
        assert captured["called"] is True
        assert captured["force"] is True
        assert "test stub" in result


# ---------------------------------------------------------------------------
# R4-Critical: log failures must not break rollback inclusion
# ---------------------------------------------------------------------------


class TestLogFailuresDontBreakRollback:
    def test_log_failure_does_not_drop_entry_from_rollback_set(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If _append_log raised AFTER atomic_write succeeded but BEFORE the
        previous code did `written.append(entry)`, the page would be
        modified on disk yet absent from the rollback list — silently
        partial state. Reordering plus _safe_log fixes this."""
        from llm_wiki_mcp import ingest as ingest_mod

        # Seed an existing page so op[0] becomes a real update we can roll back.
        target = isolated_wiki / "pages" / "x" / "page.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            "---\ntitle: X\nupdated: 2026-01-01\n---\noriginal\n"
        )
        original = target.read_text()

        # Make _append_log raise on every call: that used to mask the
        # rollback set; now _safe_log swallows it.
        def boom(*_a, **_kw):
            raise RuntimeError("simulated log disk failure")

        monkeypatch.setattr(ingest_mod, "_append_log", boom)

        # Make the second write fail to trigger rollback.
        from llm_wiki_mcp import link_fix
        real_write = link_fix.atomic_write
        n = {"calls": 0}

        def flaky(path, content):
            n["calls"] += 1
            if n["calls"] == 2:
                raise OSError("disk full")
            real_write(path, content)

        monkeypatch.setattr(link_fix, "atomic_write", flaky)

        ops = [
            {"type": "update", "filename": "page.md", "content": "addendum"},
            {
                "type": "create",
                "filename": "y/new.md",
                "content": "---\ntitle: Y\nupdated: 2026-04-28\n---\nbody",
            },
        ]
        with pytest.raises(IngestApplyError, match="apply write failed"):
            _apply_operations(ops)

        # Update was rolled back even though _append_log raised on every call.
        assert target.read_text() == original

    def test_rollback_skips_when_other_writer_modified_file(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """CAS check: if another process modified the file between our
        write and our rollback (e.g. wiki_apply ran), we must NOT clobber
        their change with our pre-batch snapshot. Skip and log."""
        from llm_wiki_mcp import link_fix

        target = isolated_wiki / "pages" / "x" / "page.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            "---\ntitle: X\nupdated: 2026-01-01\n---\noriginal\n"
        )

        real_write = link_fix.atomic_write
        n = {"calls": 0}
        # After the first successful write, a "rogue" writer overwrites the
        # file with their own content. Then op 2 fails and rollback runs.
        def rogue_then_fail(path, content):
            n["calls"] += 1
            if n["calls"] == 1:
                real_write(path, content)
                # Simulate a concurrent writer changing the file.
                path.write_text("---\ntitle: rogue\nupdated: 2026-04-28\n---\nrogue body\n")
                return
            if n["calls"] == 2:
                raise OSError("disk full")
            real_write(path, content)

        monkeypatch.setattr(link_fix, "atomic_write", rogue_then_fail)

        ops = [
            {"type": "update", "filename": "page.md", "content": "addendum"},
            {
                "type": "create",
                "filename": "y/new.md",
                "content": "---\ntitle: Y\nupdated: 2026-04-28\n---\nbody",
            },
        ]
        with pytest.raises(IngestApplyError, match="apply write failed"):
            _apply_operations(ops)

        # Rogue writer's content should remain (CAS check skipped rollback).
        assert "rogue body" in target.read_text()


# ---------------------------------------------------------------------------
# R4-High: raw filename collision avoidance
# ---------------------------------------------------------------------------


class TestRawFilenameCollision:
    def test_allocate_raw_path_returns_unique_paths_under_contention(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from llm_wiki_mcp import server

        # Patch RAW_DIR in server so allocate writes into the isolated wiki.
        monkeypatch.setattr(server, "RAW_DIR", isolated_wiki / "raw")

        paths = {server._allocate_raw_path() for _ in range(50)}
        assert len(paths) == 50  # all unique
        for p in paths:
            assert p.exists()
            assert p.parent == isolated_wiki / "raw"


# ---------------------------------------------------------------------------
# R4-High: wiki_ingest force=True bypasses threshold
# ---------------------------------------------------------------------------


class TestWikiIngestForce:
    def test_force_triggers_below_threshold(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from llm_wiki_mcp import orchestrator, ingest as ingest_mod

        # 1 pending raw — far below INGEST_THRESHOLD (5). Without force, the
        # orchestrator should refuse; with force=True, it should trigger.
        (isolated_wiki / "raw" / "single.md").write_text("body")

        monkeypatch.setattr(orchestrator, "is_available", lambda: True)
        monkeypatch.setattr(
            ingest_mod, "start_ingest", lambda *a, **kw: "stub-job"
        )

        deferred = orchestrator.run_pending_ingest(force=False)
        assert deferred["triggered"] is False

        forced = orchestrator.run_pending_ingest(force=True)
        assert forced["triggered"] is True
        assert "force=True" in forced["reason"]


# ---------------------------------------------------------------------------
# R4-Medium: filename schema hardening
# ---------------------------------------------------------------------------


class TestFilenameSchemaStrict:
    def test_control_char_rejected(self) -> None:
        from llm_wiki_mcp.ingest import _validate_triage_plan

        for c in ("\x00", "\n", "\t", "\x07", "\x7f"):
            assert (
                _validate_triage_plan(
                    [{"type": "create", "filename": f"foo{c}bar.md"}]
                )
                is None
            ), c

    def test_long_filename_rejected(self) -> None:
        from llm_wiki_mcp.ingest import _validate_triage_plan

        long_name = "a" * 250 + ".md"
        assert (
            _validate_triage_plan([{"type": "create", "filename": long_name}])
            is None
        )

    def test_non_kebab_case_rejected(self) -> None:
        from llm_wiki_mcp.ingest import _validate_triage_plan

        for bad in (
            "Foo.md",         # uppercase
            "snake_case.md",  # underscore
            "a/b/c.md",       # nested folders
            "a..md",          # consecutive dots-ish
            "-leading.md",    # leading dash
            "trailing-.md",   # trailing dash before suffix
        ):
            assert (
                _validate_triage_plan([{"type": "create", "filename": bad}])
                is None
            ), bad

    def test_kebab_with_optional_folder_accepted(self) -> None:
        from llm_wiki_mcp.ingest import _validate_triage_plan

        for good in ("ai/foo.md", "foo.md", "foo", "ai/career-note"):
            out = _validate_triage_plan([{"type": "create", "filename": good}])
            assert out is not None, good


# ---------------------------------------------------------------------------
# R4-Medium: Unicode NFC/NFD collision detection
# ---------------------------------------------------------------------------


class TestUnicodeCollision:
    def test_nfc_vs_nfd_treated_as_same_page(self, isolated_wiki: Path) -> None:
        """café (NFC, 4 chars) vs café (NFD, 5 chars: e + combining acute)
        live as the same logical page on macOS APFS. Both should map to one
        normalized key for collision detection.

        Note: validation in _validate_triage_plan rejects non-ASCII so this
        is exercised at the apply layer. We bypass triage and call apply
        directly with a non-ASCII filename to confirm the collision logic
        itself catches it. (A real plan would never reach this case because
        _validate_triage_plan filters non-ASCII first — that's defense in
        depth.)
        """
        import unicodedata
        from llm_wiki_mcp.ingest import _normalize_for_collision

        nfc = unicodedata.normalize("NFC", "café")
        nfd = unicodedata.normalize("NFD", "café")
        assert nfc != nfd  # bytes differ
        assert _normalize_for_collision(nfc) == _normalize_for_collision(nfd)
        # Casefold also handled.
        assert _normalize_for_collision("CAFÉ") == _normalize_for_collision("café")


# ---------------------------------------------------------------------------
# R4-Medium: rebuild_index failure is non-fatal
# ---------------------------------------------------------------------------


class TestRebuildIndexNonFatal:
    def test_rebuild_index_error_does_not_block_completion(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """index.md is a derived artifact. If rebuild fails after pages
        have been written, we must still report COMPLETED and call
        on_complete — otherwise raws stay pending and retry will collide
        on every page we already created."""
        from llm_wiki_mcp import ingest as ingest_mod, jobs

        monkeypatch.setattr(ingest_mod, "is_available", lambda: True)
        monkeypatch.setattr(
            ingest_mod,
            "_triage",
            lambda _content: [
                {"type": "create", "filename": "ai/foo.md", "title": "Foo"}
            ],
        )
        monkeypatch.setattr(
            ingest_mod,
            "_generate_one",
            lambda _op, _raw: {
                "type": "create",
                "filename": "ai/foo.md",
                "content": (
                    "---\ntitle: Foo\nupdated: 2026-04-28\n---\nbody"
                ),
            },
        )

        def boom() -> None:
            raise RuntimeError("simulated rebuild failure")

        monkeypatch.setattr(ingest_mod, "_rebuild_index", boom)

        on_complete_calls = []
        job = jobs.job_store.create(processor="ollama")
        ingest_mod.run_ingest(
            "raw",
            job.job_id,
            on_complete=lambda: on_complete_calls.append(True),
        )

        finished = jobs.job_store.get(job.job_id)
        assert finished.status == jobs.JobStatus.COMPLETED, finished.error
        assert finished.pages_created == ["foo"]
        assert on_complete_calls == [True]
        # Page is on disk despite rebuild failure.
        assert (isolated_wiki / "pages" / "ai" / "foo.md").exists()

    def test_real_rebuild_index_io_failure_still_completes(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Stronger version: don't mock _rebuild_index — let it actually run,
        but make INDEX_FILE be a directory so the real write_text raises
        IsADirectoryError. We still expect COMPLETED + on_complete."""
        from llm_wiki_mcp import ingest as ingest_mod, jobs

        # Replace INDEX_FILE with a directory (cannot write_text).
        idx_path = isolated_wiki / "index.md"
        idx_path.mkdir(parents=True, exist_ok=False)
        monkeypatch.setattr(ingest_mod, "INDEX_FILE", idx_path)

        monkeypatch.setattr(ingest_mod, "is_available", lambda: True)
        monkeypatch.setattr(
            ingest_mod,
            "_triage",
            lambda _content: [
                {"type": "create", "filename": "ai/bar.md", "title": "Bar"}
            ],
        )
        monkeypatch.setattr(
            ingest_mod,
            "_generate_one",
            lambda _op, _raw: {
                "type": "create",
                "filename": "ai/bar.md",
                "content": (
                    "---\ntitle: Bar\nupdated: 2026-04-28\n---\nbody"
                ),
            },
        )

        on_complete_calls = []
        job = jobs.job_store.create(processor="ollama")
        ingest_mod.run_ingest(
            "raw",
            job.job_id,
            on_complete=lambda: on_complete_calls.append(True),
        )

        finished = jobs.job_store.get(job.job_id)
        assert finished.status == jobs.JobStatus.COMPLETED, finished.error
        assert on_complete_calls == [True]


# ---------------------------------------------------------------------------
# R5: parallel raw allocation
# ---------------------------------------------------------------------------


class TestRawAllocationParallel:
    def test_concurrent_threads_get_unique_paths(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Real concurrency test: 50 threads racing into _allocate_raw_path
        must each receive a distinct path. The sequential test in R4 only
        proved the single-thread case."""
        from concurrent.futures import ThreadPoolExecutor
        import threading
        from llm_wiki_mcp import server

        monkeypatch.setattr(server, "RAW_DIR", isolated_wiki / "raw")

        N = 50
        barrier = threading.Barrier(N)

        def worker() -> Path:
            barrier.wait()  # all threads release at once → maximize collision
            return server._allocate_raw_path()

        with ThreadPoolExecutor(max_workers=N) as ex:
            paths = list(ex.map(lambda _i: worker(), range(N)))

        assert len(set(paths)) == N
        for p in paths:
            assert p.exists()

    def test_session_id_with_traversal_chars_sanitized(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A malicious or malformed session_id ('../escape') must NOT let
        the raw file land outside RAW_DIR."""
        from llm_wiki_mcp import server

        monkeypatch.setattr(server, "RAW_DIR", isolated_wiki / "raw")

        path = server._allocate_raw_path(prefix="../../etc/passwd")
        assert path.parent == (isolated_wiki / "raw")
        # Sanitizer collapses path-traversal chars to dashes.
        assert "../" not in path.name
        assert "/" not in path.name


# ---------------------------------------------------------------------------
# R5-Critical: post-apply _append_log failures don't override COMPLETED
# ---------------------------------------------------------------------------


class TestPostApplyLogSafety:
    def test_safe_log_actually_calls_append_log(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Regression test for the recursion bug Codex caught: a stray sed
        pass once rewrote the body of ``_safe_log`` to call itself, making
        every "safe" log silently no-op (and burn a recursion limit). All
        the atomicity tests passed only because nothing was ever logged.
        Verify that calling _safe_log with a working _append_log does in
        fact write through."""
        from llm_wiki_mcp import ingest as ingest_mod

        captured: list[str] = []
        monkeypatch.setattr(
            ingest_mod, "_append_log", lambda msg: captured.append(msg)
        )

        ingest_mod._safe_log("hello via safe_log")
        assert captured == ["hello via safe_log"]

    def test_log_failure_after_apply_still_completes_and_calls_on_complete(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The R5-Critical regression: after _apply_operations + COMPLETED
        was set, a raising _append_log used to fall through to the outer
        except, override status with FAILED, and skip on_complete. Pages
        persist but raws stay pending → next tick collides on every page.

        With _safe_log wrapping every post-apply log call, this can't
        happen. Verify by patching _append_log to raise on every call."""
        from llm_wiki_mcp import ingest as ingest_mod, jobs

        monkeypatch.setattr(ingest_mod, "is_available", lambda: True)
        monkeypatch.setattr(
            ingest_mod,
            "_triage",
            lambda _content: [
                {"type": "create", "filename": "ai/baz.md", "title": "Baz"}
            ],
        )
        monkeypatch.setattr(
            ingest_mod,
            "_generate_one",
            lambda _op, _raw: {
                "type": "create",
                "filename": "ai/baz.md",
                "content": (
                    "---\ntitle: Baz\nupdated: 2026-04-28\n---\nbody"
                ),
            },
        )

        def boom(*_a, **_kw):
            raise RuntimeError("simulated log disk failure")

        monkeypatch.setattr(ingest_mod, "_append_log", boom)

        on_complete_calls = []
        job = jobs.job_store.create(processor="ollama")
        ingest_mod.run_ingest(
            "raw",
            job.job_id,
            on_complete=lambda: on_complete_calls.append(True),
        )

        finished = jobs.job_store.get(job.job_id)
        assert finished.status == jobs.JobStatus.COMPLETED, finished.error
        assert finished.pages_created == ["baz"]
        assert on_complete_calls == [True]


# ---------------------------------------------------------------------------
# R5-Medium: filename schema split — legacy update should still work
# ---------------------------------------------------------------------------


class TestFilenameSchemaUpdateLeniency:
    def test_legacy_filename_accepted_for_update(self) -> None:
        from llm_wiki_mcp.ingest import _validate_triage_plan

        # These would never pass the create regex, but for update we let
        # them through so legacy corpus pages remain updatable. The actual
        # existence check happens in _apply_operations.
        for legacy in (
            "Foo.md",
            "snake_case_page.md",
            "MixedCase/foo.md",
            "ai/UPPERCASE.md",
        ):
            out = _validate_triage_plan([{"type": "update", "filename": legacy}])
            assert out is not None, legacy

    def test_create_still_strict(self) -> None:
        from llm_wiki_mcp.ingest import _validate_triage_plan

        # Same names rejected for create.
        for bad in ("Foo.md", "snake_case.md", "MixedCase/foo.md"):
            out = _validate_triage_plan([{"type": "create", "filename": bad}])
            assert out is None, bad

    def test_control_char_still_rejected_for_update(self) -> None:
        from llm_wiki_mcp.ingest import _validate_triage_plan

        assert (
            _validate_triage_plan(
                [{"type": "update", "filename": "foo\x00.md"}]
            )
            is None
        )

    def test_traversal_rejected_for_both_op_types(self) -> None:
        from llm_wiki_mcp.ingest import _validate_triage_plan

        for op_type in ("create", "update"):
            assert (
                _validate_triage_plan(
                    [{"type": op_type, "filename": "../../etc/passwd.md"}]
                )
                is None
            ), op_type


# ---------------------------------------------------------------------------
# R6: session bootstrap contract
# ---------------------------------------------------------------------------


class TestWikiInit:
    def test_returns_system_pages_with_status(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from llm_wiki_mcp import ollama, server

        for page_id in ("user-profile", "current-state", "lessons-learned"):
            (isolated_wiki / "system" / f"{page_id}.md").write_text(
                f"---\ntitle: {page_id}\nupdated: 2026-04-28\n---\n"
                f"body for [[{page_id}]]\n"
            )
        (isolated_wiki / "raw" / "pending.md").write_text("raw")

        monkeypatch.setattr(server, "WIKI_ROOT", isolated_wiki)
        monkeypatch.setattr(server, "RAW_DIR", isolated_wiki / "raw")
        monkeypatch.setattr(server, "SYSTEM_DIR", isolated_wiki / "system")
        monkeypatch.setattr(ollama, "is_available", lambda: False)

        payload = json.loads(server.wiki_init())

        assert payload["status"]["page_count"] == 0
        assert payload["status"]["raw_total"] == 1
        assert payload["status"]["raw_pending"] == 1
        assert payload["status"]["ollama_status"] == "stopped"
        assert set(payload["system_pages"]) == {
            "user-profile",
            "current-state",
            "lessons-learned",
        }
        assert "body for [[user-profile]]" in payload["system_pages"]["user-profile"]["content"]


# ---------------------------------------------------------------------------
# R6: protected regions in link extraction and auto-fix
# ---------------------------------------------------------------------------


class TestLinkFixProtectedRegions:
    def test_unclosed_fence_links_are_ignored_by_extractor(self) -> None:
        from llm_wiki_mcp.link_fix import extract_targets

        text = "before [[real]]\n```python\nx = data[[1]]\ny = [[not-a-link]]\n"
        assert extract_targets(text, strip=True) == ["real"]

    def test_lint_replace_leaves_code_frontmatter_and_inline_code(self) -> None:
        from llm_wiki_mcp.lint import _replace_link_in_content

        content = (
            "---\ntitle: [[ghost]]\n---\n"
            "body [[ghost#sec|Ghost Page]] and [[other]]\n"
            "`[[ghost]]`\n"
            "```python\nx = data[[ghost]]\n```\n"
        )
        new_content, count = _replace_link_in_content(
            content, "ghost", "real-page"
        )

        assert count == 1
        assert "title: [[ghost]]" in new_content
        assert "`[[ghost]]`" in new_content
        assert "x = data[[ghost]]" in new_content
        assert "[[real-page#sec|Ghost Page]]" in new_content
        assert "[[other]]" in new_content

    def test_lint_plaintext_fallback_uses_alias(self) -> None:
        from llm_wiki_mcp.lint import _replace_link_in_content

        new_content, count = _replace_link_in_content(
            "See [[ghost|visible name]] and [[ghost#old]].",
            "ghost",
            None,
        )

        assert count == 2
        assert new_content == "See visible name and ghost."
