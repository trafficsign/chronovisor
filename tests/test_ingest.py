"""Regression tests for the ingest pipeline.

Each test pins a bug we shipped at least once. Do not delete one without
replacing it with something that catches the same class of mistake.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import date, datetime
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

    def test_safe_python_literal_fallback(self) -> None:
        assert _extract_json_array(
            "[{'type': 'create', 'filename': 'memory/fact.md', "
            "'keywords': ['fact'], 'enabled': True}]"
        ) == [
            {
                "type": "create",
                "filename": "memory/fact.md",
                "keywords": ["fact"],
                "enabled": True,
            }
        ]

    def test_python_only_literal_types_are_rejected(self) -> None:
        assert _extract_json_array("[{'keywords': ('not', 'json')}]") is None

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

    def test_update_bare_markdown_recovers(self) -> None:
        # Qwen often follows the body contract but omits the wrapper
        # entirely. For updates that is acceptable Markdown to append.
        out = _extract_page_body(
            "## 6.5 Stop Hook 最適化\n\n本文だけ返ってきたケース。",
            op_type="update",
        )
        assert out == "## 6.5 Stop Hook 最適化\n\n本文だけ返ってきたケース。"

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


def test_generate_one_supplies_current_date_and_forbids_date_inference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from llm_wiki_mcp import ingest

    captured: dict[str, str] = {}
    today = date.today().isoformat()

    monkeypatch.setattr(
        ingest,
        "_build_focused_context",
        lambda _op, _raw: "Focused context",
    )

    def fake_generate(prompt: str, *, system: str, progress_callback=None) -> str:
        captured["prompt"] = prompt
        captured["system"] = system
        return (
            "=== NEW PAGE: memory/date-contract.md ===\n"
            "---\n"
            "title: Date contract\n"
            f"updated: {today}\n"
            "tags: [d/tools-config, t/reference, s/2026]\n"
            "---\n\n"
            "本文\n"
            "=== END PAGE ==="
        )

    monkeypatch.setattr(ingest, "_generate_with_progress", fake_generate)

    result = ingest._generate_one(
        {
            "type": "create",
            "filename": "memory/date-contract.md",
            "title": "Date contract",
            "summary": "date handling",
        },
        "No date appears in this raw evidence.",
    )

    assert result is not None
    assert f"Current date: {today}" in captured["prompt"]
    assert "Do not add or infer any other date" in captured["prompt"]
    assert "exact current date" in captured["system"]


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

    from llm_wiki_mcp import (
        wiki,
        ingest,
        index_store,
        ollama,
        orchestrator,
        page_mutation,
        runtime_status,
        search,
    )

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
    monkeypatch.setattr(
        page_mutation,
        "WIKI_MUTATION_LOCK",
        wiki_root / "runtime" / "wiki-mutation.lock",
    )
    monkeypatch.setattr(page_mutation, "PAGES_DIR", pages)
    monkeypatch.setattr(runtime_status, "RUNTIME_DIR", wiki_root / "runtime")
    monkeypatch.setattr(runtime_status, "STATUS_FILE", wiki_root / "runtime" / "status.json")
    monkeypatch.setattr(runtime_status, "EVENTS_FILE", wiki_root / "runtime" / "events.jsonl")
    monkeypatch.setattr(runtime_status, "METRICS_FILE", wiki_root / "runtime" / "metrics.jsonl")

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
    monkeypatch.setattr(ollama, "is_available", lambda: False)
    monkeypatch.setattr(search, "update_embeddings", lambda page_ids=None: 0)
    def isolated_frontier_review(proposal, *, reviewer=None):
        if reviewer is not None:
            return ingest._normalize_ingest_frontier_review(
                reviewer(proposal),
                proposal=proposal,
            )
        has_prepared = bool(proposal.get("prepared_operations"))
        has_failed = bool(proposal.get("failed_operation_specs"))
        return {
            "decision": "apply_available" if has_prepared else "confirmed_noop",
            "summary": "isolated ingest fixture disposition",
            "failed_operations_disposition": (
                "confirmed_unnecessary" if has_failed else "none"
            ),
            "tests_run": [],
            "risk": None,
            "notes": None,
        }

    monkeypatch.setattr(ingest, "_run_ingest_frontier_review", isolated_frontier_review)

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
        assert f"updated: {date.today().isoformat()}" in body

    def test_create_collision_converts_to_update(self, isolated_wiki: Path) -> None:
        path = _seed_page(
            isolated_wiki, "a/foo.md", "---\ntitle: existing\nupdated: 2026-01-01\n---\nold"
        )
        ops = [
            {
                "type": "create",
                "filename": "b/foo.md",
                "content": "---\ntitle: dup\nupdated: 2026-04-28\n---\nnew",
            }
        ]
        created, updated = _apply_operations(ops)

        assert created == []
        assert updated == ["foo"]
        text = path.read_text()
        assert "old" in text
        assert "new" in text
        assert "title: dup" not in text
        assert not (isolated_wiki / "pages" / "b" / "foo.md").exists()

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

    def test_update_resolves_single_loose_page_id(
        self, isolated_wiki: Path
    ) -> None:
        path = _seed_page(
            isolated_wiki,
            "ai/opus-4.7-evaluation-and-industry-geopolitics.md",
            "---\ntitle: Opus\nupdated: 2026-01-01\n---\nold",
        )
        ops = [
            {
                "type": "update",
                "filename": "opus-4-7-evaluation-and-industry-geopolitics.md",
                "content": "new",
            }
        ]

        created, updated = _apply_operations(ops)

        assert created == []
        assert updated == ["opus-4.7-evaluation-and-industry-geopolitics"]
        text = path.read_text()
        assert "old" in text
        assert "new" in text

    def test_update_ambiguous_loose_page_id_fails(
        self, isolated_wiki: Path
    ) -> None:
        _seed_page(
            isolated_wiki,
            "a/foo.bar.md",
            "---\ntitle: A\nupdated: 2026-01-01\n---\nold",
        )
        _seed_page(
            isolated_wiki,
            "b/foo-bar.md",
            "---\ntitle: B\nupdated: 2026-01-01\n---\nold",
        )
        ops = [
            {
                "type": "update",
                "filename": "foo_bar.md",
                "content": "new",
            }
        ]

        with pytest.raises(IngestApplyError, match="ambiguous loose page_id"):
            _apply_operations(ops)

    def test_update_resolves_page_id_alias(self, isolated_wiki: Path) -> None:
        from llm_wiki_mcp.alias_store import add_alias

        path = _seed_page(
            isolated_wiki,
            "ai/canonical-target.md",
            "---\ntitle: Canonical\nupdated: 2026-01-01\n---\nold",
        )
        add_alias("model-made-up-target", "ai/canonical-target")
        ops = [
            {
                "type": "update",
                "filename": "model-made-up-target.md",
                "content": "new",
            }
        ]

        created, updated = _apply_operations(ops)

        assert created == []
        assert updated == ["canonical-target"]
        assert "new" in path.read_text()

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

    def test_stale_ingest_cannot_reintroduce_applied_content_correction(
        self, isolated_wiki: Path
    ) -> None:
        from llm_wiki_mcp import page_mutation

        path = _seed_page(
            isolated_wiki,
            "hardware/display.md",
            "---\ntitle: Display\nupdated: 2026-07-10\n---\n"
            "The setup has two G32P 32-inch 6K displays.\n",
        )
        prepared = page_mutation.prepare_page_mutation(
            "display",
            [
                {
                    "old_text": "The setup has two G32P 32-inch 6K displays.",
                    "new_text": "The setup has one G32P 32-inch 6K display.",
                }
            ],
            correction_id="corr-display-count",
        )
        assert page_mutation.apply_prepared_mutations([prepared])["status"] == "applied"

        created, updated = _apply_operations(
            [
                {
                    "type": "update",
                    "filename": "display.md",
                    "content": (
                        "Stale replay says: The setup has two G32P 32-inch 6K displays. "
                        "The desk is 180 cm wide."
                    ),
                }
            ]
        )

        assert created == []
        assert updated == ["display"]
        written = path.read_text(encoding="utf-8")
        assert "two G32P 32-inch 6K displays" not in written
        assert written.count("one G32P 32-inch 6K display") >= 2
        assert "The desk is 180 cm wide." in written
        assert "applied_corrections: [corr-display-count]" in written

    def test_stale_ingest_cannot_resurrect_correction_under_new_slug(
        self, isolated_wiki: Path
    ) -> None:
        from llm_wiki_mcp import page_mutation

        _seed_page(
            isolated_wiki,
            "hardware/display.md",
            "---\ntitle: Display\nupdated: 2026-07-10\n---\n"
            "The setup has two G32P 32-inch 6K displays.\n",
        )
        prepared = page_mutation.prepare_page_mutation(
            "display",
            [
                {
                    "old_text": "The setup has two G32P 32-inch 6K displays.",
                    "new_text": "The setup has one G32P 32-inch 6K display.",
                }
            ],
            correction_id="corr-display-count-global",
        )
        assert page_mutation.apply_prepared_mutations([prepared])["status"] == "applied"

        created, updated = _apply_operations(
            [
                {
                    "type": "create",
                    "filename": "alternate-display-memory.md",
                    "content": (
                        "---\ntitle: Alternate display memory\nupdated: 2026-07-11\n---\n"
                        "The setup has two G32P 32-inch 6K displays.\n"
                    ),
                }
            ]
        )

        assert updated == []
        assert created == ["alternate-display-memory"]
        alternate = page_mutation.PAGES_DIR / "alternate-display-memory.md"
        written = alternate.read_text(encoding="utf-8")
        assert "two G32P 32-inch 6K displays" not in written
        assert "one G32P 32-inch 6K display" in written
        assert "corr-display-count-global" in written

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


class TestIngestFrontierGate:
    @staticmethod
    def _create_op() -> dict:
        return {
            "type": "create",
            "filename": "memory/frontier-only.md",
            "content": (
                "---\ntitle: Frontier only\nupdated: 2026-07-11\n---\n"
                "The exact proposed fact.\n"
            ),
        }

    def test_local_prepare_alone_cannot_mutate_page(
        self, isolated_wiki: Path
    ) -> None:
        from llm_wiki_mcp import ingest

        planned, _totals = ingest._prepare_operations([self._create_op()])

        assert len(planned) == 1
        assert "The exact proposed fact." in planned[0].new_body
        assert not (isolated_wiki / "pages" / "memory" / "frontier-only.md").exists()

    def test_low_risk_proposal_bypasses_frontier_but_keeps_durable_verdict(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from llm_wiki_mcp import ingest

        raw = next(
            candidate
            for index in range(1000)
            if (
                candidate := f"ordinary observation {index}"
            )
            and int(ingest._ingest_source_key(candidate, None)[:12], 16)
            / float(16**12)
            >= 0.10
        )
        monkeypatch.setattr(
            ingest,
            "_run_ingest_frontier_review",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("low-risk proposal should not call frontier")
            ),
        )

        result = ingest._review_and_apply_ingest_operations(
            [self._create_op()],
            raw_content=raw,
        )

        assert result["status"] == "apply_available"
        assert result["audit"]["mode"] == "local"
        assert result["review"]["reviewer"] == "local_policy"
        assert (isolated_wiki / "pages" / "memory" / "frontier-only.md").exists()

    def test_explicit_correction_signal_requires_frontier(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from llm_wiki_mcp import ingest

        captured: list[dict] = []
        monkeypatch.setattr(
            ingest,
            "_run_ingest_frontier_review",
            lambda proposal, **_kwargs: (
                captured.append(proposal)
                or {
                    "decision": "apply_available",
                    "summary": "correction is grounded",
                    "failed_operations_disposition": "none",
                }
            ),
        )

        result = ingest._review_and_apply_ingest_operations(
            [self._create_op()],
            raw_content="その記憶は違う。訂正して。",
        )

        assert result["status"] == "apply_available"
        assert result["audit"]["mode"] == "mandatory"
        assert len(captured) == 1

    def test_frontier_confirmed_noop_is_durable_and_non_mutating(
        self, isolated_wiki: Path
    ) -> None:
        from llm_wiki_mcp import ingest

        result = ingest._review_and_apply_ingest_operations(
            [self._create_op()],
            raw_content="raw evidence",
            reviewer=lambda _proposal: {
                "decision": "confirmed_noop",
                "summary": "claim is not grounded",
                "failed_operations_disposition": "none",
            },
        )

        assert result["status"] == "confirmed_noop"
        assert result["created"] == []
        assert not (isolated_wiki / "pages" / "memory" / "frontier-only.md").exists()
        proposal_path, review_path = ingest._ingest_artifact_paths(result["source_key"])
        assert proposal_path.exists()
        assert review_path.exists()
        review = json.loads(review_path.read_text(encoding="utf-8"))
        assert review["proposal_sha256"] == result["proposal_sha256"]
        assert review["review"]["decision"] == "confirmed_noop"

    def test_frontier_retry_keeps_proposal_but_writes_no_verdict_or_page(
        self, isolated_wiki: Path
    ) -> None:
        from llm_wiki_mcp import ingest

        result = ingest._review_and_apply_ingest_operations(
            [self._create_op()],
            raw_content="retry evidence",
            reviewer=lambda _proposal: {
                "decision": "needs_retry",
                "summary": "frontier transport unavailable",
            },
        )

        assert result["status"] == "needs_retry"
        proposal_path, review_path = ingest._ingest_artifact_paths(result["source_key"])
        assert proposal_path.exists()
        assert not review_path.exists()
        assert not (isolated_wiki / "pages" / "memory" / "frontier-only.md").exists()

    def test_frontier_approval_reviews_exact_raw_preimage_and_postimage(
        self, isolated_wiki: Path
    ) -> None:
        from llm_wiki_mcp import ingest

        original = "---\ntitle: Existing\nupdated: 2026-01-01\n---\nold fact\n"
        path = _seed_page(isolated_wiki, "memory/existing.md", original)
        captured: list[dict] = []

        result = ingest._review_and_apply_ingest_operations(
            [
                {
                    "type": "update",
                    "filename": "memory/existing.md",
                    "content": "## New evidence\nnew fact",
                }
            ],
            raw_content="verbatim raw evidence",
            source_raw="raw/session.md",
            reviewer=lambda proposal: (
                captured.append(proposal)
                or {
                    "decision": "apply_available",
                    "summary": "exact evidence supports it",
                    "failed_operations_disposition": "none",
                }
            ),
        )

        assert result["status"] == "apply_available"
        assert result["updated"] == ["existing"]
        assert "new fact" in path.read_text(encoding="utf-8")
        proposal = captured[0]
        assert proposal["raw_content"] == "verbatim raw evidence"
        exact = proposal["prepared_operations"][0]
        assert exact["previous_text"] == original
        assert exact["previous_sha256"] == hashlib.sha256(original.encode()).hexdigest()
        assert exact["proposed_sha256"] == hashlib.sha256(
            exact["proposed_text"].encode()
        ).hexdigest()

    def test_page_race_after_reviewed_prepare_fails_closed(
        self, isolated_wiki: Path
    ) -> None:
        from llm_wiki_mcp import ingest

        path = _seed_page(
            isolated_wiki,
            "memory/race.md",
            "---\ntitle: Race\nupdated: 2026-01-01\n---\nold\n",
        )
        planned, totals = ingest._prepare_operations(
            [
                {
                    "type": "update",
                    "filename": "memory/race.md",
                    "content": "reviewed proposal",
                }
            ]
        )
        path.write_text(
            "---\ntitle: Race\nupdated: 2026-01-01\n---\nconcurrent correction\n",
            encoding="utf-8",
        )

        with pytest.raises(IngestApplyError, match="page changed before ingest apply"):
            ingest._apply_prepared_operations(planned, link_totals=totals)
        assert "concurrent correction" in path.read_text(encoding="utf-8")
        assert "reviewed proposal" not in path.read_text(encoding="utf-8")

    def test_approved_artifact_recovers_without_second_frontier_call(
        self,
        isolated_wiki: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from llm_wiki_mcp import ingest

        real_apply = ingest._apply_prepared_operations
        monkeypatch.setattr(
            ingest,
            "_apply_prepared_operations",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                IngestApplyError("simulated power loss before page replace")
            ),
        )
        with pytest.raises(IngestApplyError, match="simulated power loss"):
            ingest._review_and_apply_ingest_operations(
                [self._create_op()],
                raw_content="recoverable raw",
                reviewer=lambda _proposal: {
                    "decision": "apply_available",
                    "summary": "approved before crash",
                    "failed_operations_disposition": "none",
                },
            )

        monkeypatch.setattr(ingest, "_apply_prepared_operations", real_apply)
        result = ingest._review_and_apply_ingest_operations(
            [self._create_op()],
            raw_content="recoverable raw",
            reviewer=lambda _proposal: (_ for _ in ()).throw(
                AssertionError("durable verdict should be reused")
            ),
        )

        assert result["status"] == "apply_available"
        assert result["recovered_artifact"] is True
        assert result["reused_review"] is True
        assert (isolated_wiki / "pages" / "memory" / "frontier-only.md").exists()

    def test_dry_run_is_completely_read_only(self, isolated_wiki: Path) -> None:
        from llm_wiki_mcp import ingest

        before = {
            path.relative_to(isolated_wiki).as_posix(): path.read_bytes()
            for path in isolated_wiki.rglob("*")
            if path.is_file()
        }
        result = ingest._review_and_apply_ingest_operations(
            [self._create_op()],
            raw_content="dry-run raw",
            dry_run=True,
        )
        after = {
            path.relative_to(isolated_wiki).as_posix(): path.read_bytes()
            for path in isolated_wiki.rglob("*")
            if path.is_file()
        }

        assert result["status"] == "dry_run"
        assert result["artifact_written"] is False
        assert after == before
        assert not (isolated_wiki / "pages" / "memory" / "frontier-only.md").exists()


# ---------------------------------------------------------------------------
# Phase 4: raw_keywords frontmatter patch in apply prepare phase
# ---------------------------------------------------------------------------


class TestApplyRawKeywordsPatch:
    """``_apply_operations`` patches ``raw_keywords`` onto the page
    frontmatter in the prepare phase only — never in the write phase. The
    write phase keeps the existing single-atomic-write contract so partial
    failure rolls back to the pre-batch state.
    """

    def test_create_writes_raw_keywords_to_frontmatter(
        self, isolated_wiki: Path
    ) -> None:
        ops = [
            {
                "type": "create",
                "filename": "misc/p.md",
                "content": "---\ntitle: P\nupdated: 2026-04-28\n---\nbody",
                "raw_keywords": ["alpha", "beta"],
            }
        ]
        _apply_operations(ops)
        text = (isolated_wiki / "pages" / "misc" / "p.md").read_text()
        assert "raw_keywords: [alpha, beta]" in text
        assert "title: P" in text

    def test_create_without_raw_keywords_leaves_field_absent(
        self, isolated_wiki: Path
    ) -> None:
        """When the op carries no raw_keywords (e.g. raw frontmatter had
        none), the resulting page must not gain a stray empty field."""
        ops = [
            {
                "type": "create",
                "filename": "misc/q.md",
                "content": "---\ntitle: Q\nupdated: 2026-04-28\n---\nbody",
            }
        ]
        _apply_operations(ops)
        text = (isolated_wiki / "pages" / "misc" / "q.md").read_text()
        assert "raw_keywords" not in text

    def test_create_empty_list_skips_patch(self, isolated_wiki: Path) -> None:
        """Empty list = no information — don't bloat the frontmatter."""
        ops = [
            {
                "type": "create",
                "filename": "misc/r.md",
                "content": "---\ntitle: R\nupdated: 2026-04-28\n---\nbody",
                "raw_keywords": [],
            }
        ]
        _apply_operations(ops)
        text = (isolated_wiki / "pages" / "misc" / "r.md").read_text()
        assert "raw_keywords" not in text

    def test_update_unions_with_existing_preserving_order(
        self, isolated_wiki: Path
    ) -> None:
        _seed_page(
            isolated_wiki,
            "career/x.md",
            "---\ntitle: X\nupdated: 2026-01-01\nraw_keywords: [a, b]\n---\noriginal\n",
        )
        ops = [
            {
                "type": "update",
                "filename": "x.md",
                "content": "## addendum",
                "raw_keywords": ["b", "c", "d"],
            }
        ]
        _apply_operations(ops)
        text = (isolated_wiki / "pages" / "career" / "x.md").read_text()
        # Order-preserving dedupe: a, b come from existing; c, d are appended.
        assert "raw_keywords: [a, b, c, d]" in text
        # ``updated:`` was bumped to today as part of the existing contract.
        assert f"updated: {date.today().isoformat()}" in text
        # Body append still works.
        assert "## addendum" in text and "original" in text

    def test_update_recovers_from_broken_existing_value(
        self, isolated_wiki: Path
    ) -> None:
        """If a page's existing raw_keywords field is malformed (wrong
        type, manual edit, legacy), apply must self-heal by treating the
        existing value as empty rather than aborting the whole op."""
        _seed_page(
            isolated_wiki,
            "misc/y.md",
            # Scalar instead of list — broken shape.
            "---\ntitle: Y\nupdated: 2026-01-01\nraw_keywords: oops\n---\nbody\n",
        )
        ops = [
            {
                "type": "update",
                "filename": "y.md",
                "content": "## more",
                "raw_keywords": ["clean1", "clean2"],
            }
        ]
        _apply_operations(ops)
        text = (isolated_wiki / "pages" / "misc" / "y.md").read_text()
        assert "raw_keywords: [clean1, clean2]" in text

    def test_update_without_raw_keywords_leaves_existing_intact(
        self, isolated_wiki: Path
    ) -> None:
        """An update op with no raw_keywords must not erase or rewrite
        the existing field."""
        _seed_page(
            isolated_wiki,
            "misc/z.md",
            "---\ntitle: Z\nupdated: 2026-01-01\nraw_keywords: [keep, me]\n---\nbody\n",
        )
        ops = [
            {
                "type": "update",
                "filename": "z.md",
                "content": "## tail",
            }
        ]
        _apply_operations(ops)
        text = (isolated_wiki / "pages" / "misc" / "z.md").read_text()
        assert "raw_keywords: [keep, me]" in text

    def test_write_phase_rollback_restores_pre_batch_text(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Phase 4 must keep the prepare/write split intact — the
        rollback path restores the on-disk text as it was BEFORE the
        batch ran, not as it was after the in-memory raw_keywords patch.
        """
        original = (
            "---\ntitle: A\nupdated: 2026-01-01\nraw_keywords: [old]\n---\nbody\n"
        )
        path_a = _seed_page(isolated_wiki, "misc/a.md", original)

        # Make atomic_write fail on the SECOND op so the first op's write
        # is committed and then must be rolled back.
        from llm_wiki_mcp import ingest as ingest_mod

        real_atomic = ingest_mod.atomic_write if hasattr(ingest_mod, "atomic_write") else None
        from llm_wiki_mcp import link_fix

        original_atomic = link_fix.atomic_write
        call_count = {"n": 0}

        def flaky_atomic(p, content):
            call_count["n"] += 1
            if call_count["n"] == 2:
                raise OSError("simulated disk full on op 2")
            return original_atomic(p, content)

        monkeypatch.setattr(link_fix, "atomic_write", flaky_atomic)

        ops = [
            {
                "type": "update",
                "filename": "a.md",
                "content": "## first",
                "raw_keywords": ["new1"],
            },
            {
                "type": "create",
                "filename": "misc/never.md",
                "content": "---\ntitle: N\nupdated: 2026-04-28\n---\nbody",
                "raw_keywords": ["nope"],
            },
        ]
        with pytest.raises(IngestApplyError):
            _apply_operations(ops)

        # The first op was rolled back to ORIGINAL — not to the
        # raw_keywords-patched intermediate.
        assert path_a.read_text() == original


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
    def test_partial_generate_applies_successful_ops(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Contract: a partial generate failure (2 of 3 ops succeed, even
        after the per-op retry) writes the 2 successful pages, marks raws
        processed (so the next tick won't re-triage and collide on stem),
        and records the failed op in ``job.result`` for autonomous follow-up.

        Replaces the prior 'discard everything on any failure' contract.
        Discarding both halved the data the wiki captured AND looped on
        raws that kept failing; partial apply + raws-processed avoids both.
        """

        from llm_wiki_mcp import ingest, jobs

        # Stub triage → returns 3 ops.
        plan = [
            {"type": "create", "filename": f"misc/p{i}.md", "title": f"P{i}"}
            for i in range(3)
        ]
        monkeypatch.setattr(ingest, "_triage", lambda _content: plan)

        # Stub generate: always succeed for p0/p1, always fail for p2 (so
        # the retry also fails — exercises the dead-letter path).
        def fake_generate(op: dict, _raw: str, **_kw) -> dict | None:
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

        # on_complete fires → raws marked processed (no infinite retry).
        assert on_complete_called == [True]
        # on_finally fires with failed=False (we did write pages successfully).
        assert on_finally_calls == [{"failed": False, "triage_failed": False}]
        # Job COMPLETED with partial flag + failed_ops in result.
        finished = jobs.job_store.get(job.job_id)
        assert finished.status == jobs.JobStatus.COMPLETED
        assert sorted(finished.pages_created) == ["p0", "p1"]
        assert finished.pages_updated == []
        assert finished.result is not None
        assert finished.result.get("partial") is True
        failed_ops = finished.result.get("failed_ops", [])
        assert len(failed_ops) == 1
        assert failed_ops[0]["filename"].endswith("p2.md")
        # Disk: p0 and p1 were written; p2 was not.
        pages = isolated_wiki / "pages"
        assert (pages / "misc" / "p0.md").exists()
        assert (pages / "misc" / "p1.md").exists()
        assert not (pages / "misc" / "p2.md").exists()

    def test_partial_generate_retries_once_before_dead_lettering(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Per-op retry: a transient generate failure on the first attempt
        is retried once. If the retry succeeds, the op is applied; the job
        completes cleanly with no partial flag and no failed_ops."""

        from llm_wiki_mcp import ingest, jobs

        plan = [
            {"type": "create", "filename": "misc/p0.md", "title": "P0"},
            {"type": "create", "filename": "misc/p1.md", "title": "P1"},
        ]
        monkeypatch.setattr(ingest, "_triage", lambda _content: plan)

        attempts: dict[str, int] = {}

        def flaky_generate(op: dict, _raw: str, **_kw) -> dict | None:
            fname = op["filename"]
            attempts[fname] = attempts.get(fname, 0) + 1
            # p1 fails on first attempt, succeeds on second.
            if fname.endswith("p1.md") and attempts[fname] == 1:
                return None
            return {
                "type": "create",
                "filename": fname,
                "content": "---\ntitle: X\nupdated: 2026-04-28\n---\nbody",
            }

        monkeypatch.setattr(ingest, "_generate_one", flaky_generate)
        monkeypatch.setattr(ingest, "is_available", lambda: True)

        on_complete_called = []
        job = jobs.job_store.create(processor="ollama")
        ingest.run_ingest(
            "raw content",
            job.job_id,
            on_complete=lambda: on_complete_called.append(True),
        )

        # p1 was attempted twice; p0 only once.
        assert attempts.get("misc/p1.md") == 2
        assert attempts.get("misc/p0.md") == 1
        # Full success — no partial flag.
        finished = jobs.job_store.get(job.job_id)
        assert finished.status == jobs.JobStatus.COMPLETED
        assert sorted(finished.pages_created) == ["p0", "p1"]
        assert finished.result is None or not finished.result.get("partial")
        assert on_complete_called == [True]

    def test_all_ops_fail_marks_raw_processed(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If every generate op fails (even after retry), the wiki must
        not be mutated, but on_complete still fires so one bad model
        output does not block the pending drain forever."""

        from llm_wiki_mcp import ingest, jobs

        plan = [
            {"type": "create", "filename": f"misc/p{i}.md", "title": f"P{i}"}
            for i in range(2)
        ]
        monkeypatch.setattr(ingest, "_triage", lambda _content: plan)
        monkeypatch.setattr(
            ingest, "_generate_one", lambda _op, _raw, **_kw: None
        )
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

        # Nothing succeeded, but raws are marked processed to avoid an
        # infinite retry loop on deterministic malformed output.
        assert on_complete_called == [True]
        assert on_finally_calls == [{"failed": False, "triage_failed": False}]
        finished = jobs.job_store.get(job.job_id)
        assert finished.status == jobs.JobStatus.COMPLETED
        assert finished.pages_created == []
        assert finished.pages_updated == []
        assert finished.result is not None
        assert finished.result.get("partial") is True
        assert len(finished.result.get("failed_ops", [])) == 2
        # Disk untouched.
        pages = isolated_wiki / "pages"
        assert not (pages / "misc" / "p0.md").exists()
        assert not (pages / "misc" / "p1.md").exists()

    def test_missing_update_target_is_generated_as_create(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A model can misclassify a brand-new topic as an update.

        If the requested target is absent and the filename is safe for a new
        page, normalize the op before generation instead of letting apply
        quarantine the raw with update_target_not_found.
        """

        from llm_wiki_mcp import ingest, jobs

        plan = [
            {
                "type": "update",
                "filename": "career-transition-strategy-2026.md",
                "summary": "Capture career transition strategy discussion",
                "keywords": ["career", "strategy"],
            }
        ]
        seen_ops: list[dict] = []
        monkeypatch.setattr(ingest, "_triage", lambda _content: plan)

        def fake_generate(op: dict, _raw: str, *, raw_keywords=None, **_kw) -> dict:
            seen_ops.append(op)
            generated = {
                "type": op["type"],
                "filename": op["filename"],
                "content": (
                    "---\n"
                    "title: Career Transition Strategy 2026\n"
                    "updated: 2026-06-23\n"
                    "tags: [d/personal-strategy, t/analysis, s/2026]\n"
                    "---\n"
                    "body"
                ),
            }
            if raw_keywords is not None:
                generated["raw_keywords"] = raw_keywords
            return generated

        monkeypatch.setattr(ingest, "_generate_one", fake_generate)
        monkeypatch.setattr(ingest, "is_available", lambda: True)

        on_complete_called = []
        job = jobs.job_store.create(processor="ollama")
        ingest.run_ingest(
            "raw content",
            job.job_id,
            on_complete=lambda: on_complete_called.append(True),
        )

        assert seen_ops[0]["type"] == "create"
        assert seen_ops[0]["title"] == "Career Transition Strategy 2026"
        finished = jobs.job_store.get(job.job_id)
        assert finished.status == jobs.JobStatus.COMPLETED
        assert finished.pages_created == ["career-transition-strategy-2026"]
        assert finished.pages_updated == []
        assert on_complete_called == [True]
        assert (
            isolated_wiki
            / "pages"
            / "career-transition-strategy-2026.md"
        ).exists()

    def test_failure_packet_missing_update_target_becomes_create(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from llm_wiki_mcp import ingest, jobs

        plan = [
            {
                "type": "update",
                "filename": "claude-code-vs-claude-code-structural-analysis.md",
                "summary": "Capture Claude Code and Cursor adoption analysis",
            }
        ]
        seen_ops: list[dict] = []
        monkeypatch.setattr(ingest, "_triage", lambda _content: plan)

        def fake_generate(op: dict, _raw: str, *, raw_keywords=None, **_kw) -> dict:
            seen_ops.append(op)
            generated = {
                "type": op["type"],
                "filename": op["filename"],
                "content": (
                    "---\n"
                    "title: Claude Code Vs Claude Code Structural Analysis\n"
                    "updated: 2026-06-28\n"
                    "tags: [d/ai-tools, t/analysis, s/2026]\n"
                    "---\n"
                    "body"
                ),
            }
            if raw_keywords is not None:
                generated["raw_keywords"] = raw_keywords
            return generated

        monkeypatch.setattr(ingest, "_generate_one", fake_generate)
        monkeypatch.setattr(ingest, "is_available", lambda: True)

        job = jobs.job_store.create(processor="ollama")
        ingest.run_ingest(
            "raw content",
            job.job_id,
            metadata={"raw_keywords": ["Claude Code", "Cursor", "Mac Studio"]},
        )

        assert seen_ops[0]["type"] == "create"
        assert seen_ops[0]["keywords"] == [
            "claude",
            "code",
            "vs",
            "claude",
            "code",
            "structural",
            "analysis",
        ]
        finished = jobs.job_store.get(job.job_id)
        assert finished.status == jobs.JobStatus.COMPLETED
        assert finished.pages_created == [
            "claude-code-vs-claude-code-structural-analysis"
        ]
        assert finished.pages_updated == []
        page = (
            isolated_wiki
            / "pages"
            / "claude-code-vs-claude-code-structural-analysis.md"
        )
        assert page.exists()
        assert (
            "raw_keywords: [Claude Code, Cursor, Mac Studio]" in page.read_text()
        )


class TestRunIngestFrontierDisposition:
    def test_frontier_content_replacement_is_re_reviewed_before_apply(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from llm_wiki_mcp import ingest, jobs

        plan = [{
            "type": "create",
            "filename": "memory/observed-tools.md",
            "title": "Observed tools",
            "summary": "visible and responsive tools",
        }]
        triage_calls: list[str] = []

        def fake_triage(content: str, **_kwargs):
            triage_calls.append(content)
            return plan

        monkeypatch.setattr(ingest, "_triage", fake_triage)
        monkeypatch.setattr(
            ingest,
            "_generate_one",
            lambda op, *_args, **_kwargs: {
                "type": "create",
                "filename": op["filename"],
                "content": (
                    "---\ntitle: Observed tools\nupdated: 2026-07-11\n---\n"
                    "Tool A was visible but failed and should not be used.\n"
                ),
            },
        )
        monkeypatch.setattr(ingest, "is_available", lambda: True)
        reviews: list[dict] = []
        replacement = (
            "---\ntitle: Observed tools\nupdated: 2026-07-11\n---\n"
            "Tool A was visible. Tool B was confirmed responsive.\n"
        )

        def reviewer(proposal: dict) -> dict:
            reviews.append(proposal)
            proposed = proposal["prepared_operations"][0]["proposed_text"]
            if "should not be used" in proposed:
                return {
                    "decision": "retry",
                    "summary": "Replace the unsupported recommendation.",
                    "failed_operations_disposition": "none",
                    "replacement_operations": [{
                        "filename": "memory/observed-tools.md",
                        "content": replacement,
                    }],
                }
            return {
                "decision": "apply_available",
                "summary": "The replacement is narrowly grounded.",
                "failed_operations_disposition": "none",
                "replacement_operations": [],
            }

        job = jobs.job_store.create(processor="ollama")
        ingest.run_ingest(
            "Tool A visible; Tool B responsive.",
            job.job_id,
            frontier_reviewer=reviewer,
        )

        assert jobs.job_store.get(job.job_id).status == jobs.JobStatus.COMPLETED
        assert len(reviews) == 2
        assert len(triage_calls) == 1
        page = isolated_wiki / "pages" / "memory" / "observed-tools.md"
        written = page.read_text(encoding="utf-8")
        assert "Tool B was confirmed responsive." in written
        assert "should not be used" not in written

    def test_frontier_invalid_tag_uses_minimal_repair_before_regeneration(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from llm_wiki_mcp import ingest, jobs

        plan = [{
            "type": "create",
            "filename": "memory/workflow-note.md",
            "title": "Workflow note",
            "summary": "grounded workflow",
        }]
        triage_calls: list[str] = []

        def fake_triage(content: str, **_kwargs):
            triage_calls.append(content)
            return plan

        monkeypatch.setattr(ingest, "_triage", fake_triage)
        monkeypatch.setattr(
            ingest,
            "_generate_one",
            lambda op, *_args, **_kwargs: {
                "type": "create",
                "filename": op["filename"],
                "content": (
                    "---\ntitle: Workflow note\nupdated: 2026-07-11\n"
                    "tags: [d/ai-tools, t/hardware, s/2026]\n---\n"
                    "Grounded workflow fact.\n"
                ),
            },
        )
        monkeypatch.setattr(ingest, "is_available", lambda: True)
        reviews: list[dict] = []

        def reviewer(proposal: dict) -> dict:
            reviews.append(proposal)
            proposed = proposal["prepared_operations"][0]["proposed_text"]
            if "t/hardware" in proposed:
                return {
                    "decision": "retry",
                    "summary": "The incorrect tag `t/hardware` must be removed.",
                    "failed_operations_disposition": "none",
                    "invalid_tags": ["t/hardware"],
                }
            return {
                "decision": "apply_available",
                "summary": "The metadata-only repair is grounded.",
                "failed_operations_disposition": "none",
                "invalid_tags": [],
            }

        job = jobs.job_store.create(processor="ollama")
        ingest.run_ingest(
            "raw workflow fact",
            job.job_id,
            frontier_reviewer=reviewer,
        )

        assert jobs.job_store.get(job.job_id).status == jobs.JobStatus.COMPLETED
        assert len(reviews) == 2
        assert len(triage_calls) == 1
        page = isolated_wiki / "pages" / "memory" / "workflow-note.md"
        written = page.read_text(encoding="utf-8")
        assert "t/hardware" not in written
        assert "d/ai-tools" in written

    def test_frontier_content_replacement_preserves_unrejected_taxonomy(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from llm_wiki_mcp import ingest, jobs

        plan = [{
            "type": "create",
            "filename": "ai/socialization-scenario.md",
            "title": "Socialization scenario",
        }]
        monkeypatch.setattr(ingest, "_triage", lambda *_args, **_kwargs: plan)
        monkeypatch.setattr(
            ingest,
            "_generate_one",
            lambda op, *_args, **_kwargs: {
                "type": "create",
                "filename": op["filename"],
                "content": (
                    "---\ntitle: Socialization scenario\nupdated: 2026-07-11\n"
                    "tags: [d/scenario, t/ai-socialization]\n---\n"
                    "Grounded fact plus unsupported claim.\n"
                ),
            },
        )
        monkeypatch.setattr(ingest, "is_available", lambda: True)
        reviews: list[dict] = []

        def reviewer(proposal: dict) -> dict:
            reviews.append(proposal)
            proposed = proposal["prepared_operations"][0]["proposed_text"]
            if "unsupported claim" in proposed:
                return {
                    "decision": "retry",
                    "summary": "Remove only the unsupported claim.",
                    "failed_operations_disposition": "none",
                    "replacement_operations": [{
                        "filename": "ai/socialization-scenario.md",
                        "content": (
                            "---\ntitle: Socialization scenario\nupdated: 2026-07-11\n"
                            "tags: [d/event, t/ai-socialization]\n---\n"
                            "Grounded fact.\n"
                        ),
                    }],
                }
            assert "d/scenario" in proposed
            assert "d/event" not in proposed
            return {
                "decision": "apply_available",
                "summary": "The bounded repair is grounded.",
                "failed_operations_disposition": "none",
            }

        job = jobs.job_store.create(processor="ollama")
        ingest.run_ingest(
            "Grounded socialization scenario.",
            job.job_id,
            frontier_reviewer=reviewer,
        )

        assert jobs.job_store.get(job.job_id).status == jobs.JobStatus.COMPLETED
        assert len(reviews) == 2
        written = (
            isolated_wiki / "pages" / "ai" / "socialization-scenario.md"
        ).read_text(encoding="utf-8")
        assert "d/scenario" in written
        assert "d/event" not in written

    def test_frontier_rejection_regenerates_with_feedback_in_same_job(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from llm_wiki_mcp import ingest, jobs

        plan = [
            {
                "type": "create",
                "filename": "memory/converged.md",
                "title": "Converged",
                "summary": "one explicit fact",
            }
        ]
        triage_feedback: list[str | None] = []
        generate_feedback: list[str | None] = []

        def fake_triage(_content: str, *, frontier_feedback=None):
            triage_feedback.append(frontier_feedback)
            return plan

        def fake_generate(
            op: dict,
            _raw: str,
            *,
            raw_keywords=None,
            progress_callback=None,
            frontier_feedback=None,
        ):
            generate_feedback.append(frontier_feedback)
            fact = "grounded fact" if frontier_feedback else "unsupported inference"
            return {
                "type": op["type"],
                "filename": op["filename"],
                "content": (
                    "---\ntitle: Converged\nupdated: 2026-07-11\n---\n" + fact
                ),
            }

        reviews: list[dict] = []

        def reviewer(proposal: dict) -> dict:
            reviews.append(proposal)
            if len(reviews) == 1:
                return {
                    "decision": "retry",
                    "summary": "Remove the unsupported inference; keep only the grounded fact.",
                    "failed_operations_disposition": "none",
                }
            return {
                "decision": "apply_available",
                "summary": "The regenerated proposal is grounded.",
                "failed_operations_disposition": "none",
            }

        monkeypatch.setattr(ingest, "_triage", fake_triage)
        monkeypatch.setattr(ingest, "_generate_one", fake_generate)
        monkeypatch.setattr(ingest, "is_available", lambda: True)
        completed: list[bool] = []
        job = jobs.job_store.create(processor="ollama")

        ingest.run_ingest(
            "raw grounded fact",
            job.job_id,
            on_complete=lambda: completed.append(True),
            frontier_reviewer=reviewer,
        )

        finished = jobs.job_store.get(job.job_id)
        assert finished.status == jobs.JobStatus.COMPLETED
        assert completed == [True]
        assert len(reviews) == 2
        assert triage_feedback == [
            None,
            "Remove the unsupported inference; keep only the grounded fact.",
        ]
        assert generate_feedback == triage_feedback
        page = isolated_wiki / "pages" / "memory" / "converged.md"
        assert "grounded fact" in page.read_text(encoding="utf-8")
        assert "unsupported inference" not in page.read_text(encoding="utf-8")

    def test_local_noop_requires_frontier_confirmation(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from llm_wiki_mcp import ingest, jobs

        monkeypatch.setattr(ingest, "_triage", lambda _content: [])
        monkeypatch.setattr(ingest, "is_available", lambda: True)
        captured: list[dict] = []
        completed: list[bool] = []
        job = jobs.job_store.create(processor="ollama")
        ingest.run_ingest(
            "raw must not disappear",
            job.job_id,
            on_complete=lambda: completed.append(True),
            frontier_reviewer=lambda proposal: (
                captured.append(proposal)
                or {
                    "decision": "retry",
                    "summary": "local no-op is not yet proven",
                    "failed_operations_disposition": "none",
                }
            ),
        )

        finished = jobs.job_store.get(job.job_id)
        assert finished.status == jobs.JobStatus.FAILED
        assert completed == []
        assert captured[0]["raw_content"] == "raw must not disappear"
        assert captured[0]["triage_plan"] == []
        assert captured[0]["local_disposition"] == "triage_no_operations"
        proposal_path, _review_path = ingest._ingest_artifact_paths(
            captured[0]["source_key"]
        )
        assert proposal_path.exists()

    def test_all_generation_failures_require_frontier_disposition(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from llm_wiki_mcp import ingest, jobs

        plan = [
            {
                "type": "create",
                "filename": "memory/missing.md",
                "title": "Missing",
                "summary": "must be retained",
            }
        ]
        monkeypatch.setattr(ingest, "_triage", lambda _content: plan)
        monkeypatch.setattr(ingest, "_generate_one", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(ingest, "is_available", lambda: True)
        captured: list[dict] = []
        completed: list[bool] = []
        job = jobs.job_store.create(processor="ollama")
        ingest.run_ingest(
            "all-failed raw",
            job.job_id,
            on_complete=lambda: completed.append(True),
            frontier_reviewer=lambda proposal: (
                captured.append(proposal)
                or {
                    "decision": "retry",
                    "summary": "regenerate the missing operation",
                    "failed_operations_disposition": "retry_required",
                }
            ),
        )

        finished = jobs.job_store.get(job.job_id)
        assert finished.status == jobs.JobStatus.FAILED
        assert completed == []
        assert captured[0]["local_disposition"] == "all_generation_failed"
        assert captured[0]["triage_plan"] == plan
        [failure] = captured[0]["failed_operation_specs"]
        assert failure["filename"] == "memory/missing.md"
        assert failure["attempts"] == 2
        assert "parse failed" in failure["error"]
        assert captured[0]["prepared_operations"] == []

    def test_partial_apply_without_explicit_failed_disposition_retries(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from llm_wiki_mcp import ingest, jobs

        plan = [
            {"type": "create", "filename": "memory/ready.md", "title": "Ready"},
            {"type": "create", "filename": "memory/failed.md", "title": "Failed"},
        ]
        monkeypatch.setattr(ingest, "_triage", lambda _content: plan)

        def generate(op, _raw, **_kwargs):
            if op["filename"].endswith("failed.md"):
                return None
            return {
                "type": "create",
                "filename": op["filename"],
                "content": "---\ntitle: Ready\nupdated: 2026-07-11\n---\nready\n",
            }

        monkeypatch.setattr(ingest, "_generate_one", generate)
        monkeypatch.setattr(ingest, "is_available", lambda: True)
        completed: list[bool] = []
        job = jobs.job_store.create(processor="ollama")
        ingest.run_ingest(
            "partial raw",
            job.job_id,
            on_complete=lambda: completed.append(True),
            frontier_reviewer=lambda _proposal: {
                "decision": "apply_available",
                "summary": "ready page looks valid but failed op was not dispositioned",
                # Deliberately omit failed_operations_disposition.
            },
        )

        finished = jobs.job_store.get(job.job_id)
        assert finished.status == jobs.JobStatus.FAILED
        assert completed == []
        assert not (isolated_wiki / "pages" / "memory" / "ready.md").exists()
        assert not (isolated_wiki / "pages" / "memory" / "failed.md").exists()

    def test_complete_proposal_ignores_redundant_failed_disposition(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from llm_wiki_mcp import ingest, jobs

        plan = [
            {"type": "create", "filename": "memory/ready.md", "title": "Ready"},
        ]
        monkeypatch.setattr(ingest, "_triage", lambda _content: plan)
        monkeypatch.setattr(
            ingest,
            "_generate_one",
            lambda op, _raw, **_kwargs: {
                "type": "create",
                "filename": op["filename"],
                "content": "---\ntitle: Ready\nupdated: 2026-07-11\n---\nready\n",
            },
        )
        monkeypatch.setattr(ingest, "is_available", lambda: True)

        job = jobs.job_store.create(processor="ollama")
        ingest.run_ingest(
            "complete raw",
            job.job_id,
            frontier_reviewer=lambda _proposal: {
                "decision": "apply_available",
                "summary": "prepared operation is grounded",
                "failed_operations_disposition": "confirmed_unnecessary",
            },
        )

        finished = jobs.job_store.get(job.job_id)
        assert finished.status == jobs.JobStatus.COMPLETED
        assert finished.pages_created == ["ready"]
        assert (isolated_wiki / "pages" / "memory" / "ready.md").exists()

    def test_retryable_partial_proposal_does_not_pin_later_complete_generation(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from llm_wiki_mcp import ingest, jobs

        plan = [
            {"type": "create", "filename": "memory/one.md", "title": "One"},
            {"type": "create", "filename": "memory/two.md", "title": "Two"},
        ]
        monkeypatch.setattr(ingest, "_triage", lambda _content: plan)
        monkeypatch.setattr(ingest, "is_available", lambda: True)
        phase = {"complete": False}

        def generate(op, _raw, **_kwargs):
            if not phase["complete"] and op["filename"].endswith("two.md"):
                return None
            title = op["title"]
            return {
                "type": "create",
                "filename": op["filename"],
                "content": (
                    f"---\ntitle: {title}\nupdated: 2026-07-11\n---\n{title}\n"
                ),
            }

        monkeypatch.setattr(ingest, "_generate_one", generate)
        first = jobs.job_store.create(processor="ollama")
        ingest.run_ingest(
            "same replayable raw",
            first.job_id,
            frontier_reviewer=lambda _proposal: {
                "decision": "retry",
                "summary": "regenerate the missing page",
                "failed_operations_disposition": "retry_required",
            },
        )
        assert jobs.job_store.get(first.job_id).status == jobs.JobStatus.FAILED

        phase["complete"] = True
        second = jobs.job_store.create(processor="ollama")
        ingest.run_ingest(
            "same replayable raw",
            second.job_id,
            frontier_reviewer=lambda _proposal: {
                "decision": "apply_available",
                "summary": "complete regenerated proposal is grounded",
                "failed_operations_disposition": "none",
            },
        )

        finished = jobs.job_store.get(second.job_id)
        assert finished.status == jobs.JobStatus.COMPLETED
        assert sorted(finished.pages_created) == ["one", "two"]
        assert (isolated_wiki / "pages" / "memory" / "one.md").exists()
        assert (isolated_wiki / "pages" / "memory" / "two.md").exists()


# ---------------------------------------------------------------------------
# Phase 3b: raw_keywords metadata propagation through run_ingest
# ---------------------------------------------------------------------------


class TestRawKeywordsMetadataPropagation:
    """run_ingest must lift raw_keywords from metadata and ride it on
    every operation generated from this raw — without leaking into the
    triage prompt or being fabricated when absent.
    """

    def test_metadata_raw_keywords_lands_on_every_operation(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from llm_wiki_mcp import ingest, jobs

        plan = [
            {"type": "create", "filename": f"misc/p{i}.md", "title": f"P{i}"}
            for i in range(3)
        ]
        monkeypatch.setattr(ingest, "_triage", lambda _content: plan)

        captured_raw_keywords: list[list[str] | None] = []

        def stub_generate(op, _raw, *, raw_keywords=None):
            captured_raw_keywords.append(raw_keywords)
            return {
                "type": "create",
                "filename": op["filename"],
                "content": "---\ntitle: X\nupdated: 2026-04-28\n---\nbody",
            }

        monkeypatch.setattr(ingest, "_generate_one", stub_generate)
        monkeypatch.setattr(ingest, "is_available", lambda: True)

        job = jobs.job_store.create(processor="ollama")
        ingest.run_ingest(
            "raw content",
            job.job_id,
            metadata={"raw_keywords": ["alpha", "beta"]},
        )

        # Every _generate_one call saw the same raw_keywords payload.
        assert captured_raw_keywords == [["alpha", "beta"]] * 3

    def test_no_metadata_means_no_raw_keywords_field(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Distinguishes "no propagation requested" from an empty list:
        when metadata is None the operation dict has NO ``raw_keywords``
        key at all (so the apply layer can skip patching).
        """
        from llm_wiki_mcp import ingest

        op = {"type": "create", "filename": "misc/p0.md", "title": "P"}
        monkeypatch.setattr(
            ingest,
            "generate",
            lambda *_a, **_kw: (
                "---\ntitle: P\nupdated: 2026-04-28\n---\nbody"
            ),
        )

        result = ingest._generate_one(op, "raw content", raw_keywords=None)
        assert result is not None
        assert "raw_keywords" not in result

    def test_explicit_empty_list_survives(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An empty list is intent — the apply layer should see ``[]`` and
        decide for itself, not have it elided into "no propagation".
        """
        from llm_wiki_mcp import ingest

        op = {"type": "create", "filename": "misc/p0.md", "title": "P"}
        monkeypatch.setattr(
            ingest,
            "generate",
            lambda *_a, **_kw: (
                "---\ntitle: P\nupdated: 2026-04-28\n---\nbody"
            ),
        )

        result = ingest._generate_one(op, "raw content", raw_keywords=[])
        assert result is not None
        assert result.get("raw_keywords") == []

    def test_invalid_metadata_normalized_to_no_propagation(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Non-list / list with non-str items in metadata is treated as
        "no propagation". Important defensive behavior so a malformed raw
        frontmatter can't produce mojibake page metadata.
        """
        from llm_wiki_mcp import ingest, jobs

        plan = [{"type": "create", "filename": "misc/p0.md", "title": "P"}]
        monkeypatch.setattr(ingest, "_triage", lambda _content: plan)

        captured: list[list[str] | None] = []

        def stub_generate(op, _raw, *, raw_keywords=None):
            captured.append(raw_keywords)
            return {
                "type": "create",
                "filename": op["filename"],
                "content": "---\ntitle: X\nupdated: 2026-04-28\n---\nbody",
            }

        monkeypatch.setattr(ingest, "_generate_one", stub_generate)
        monkeypatch.setattr(ingest, "is_available", lambda: True)

        for bad in ("not-a-list", 42, None, ["ok", 123], {"k": "v"}):
            captured.clear()
            job = jobs.job_store.create(processor="ollama")
            ingest.run_ingest(
                "raw content",
                job.job_id,
                metadata={"raw_keywords": bad},
            )
            assert captured == [None], f"bad={bad!r}"


# ---------------------------------------------------------------------------
# Orchestrator (R2-High)
# ---------------------------------------------------------------------------


class TestOrchestrator:
    def test_retracted_raw_is_not_pending_and_body_is_preserved(
        self, isolated_wiki: Path
    ) -> None:
        from llm_wiki_mcp import orchestrator

        raw_dir = isolated_wiki / "raw"
        retracted = raw_dir / "retracted.md"
        retracted_text = (
            "---\n"
            "raw_keywords: [kuycon, p24u]\n"
            "raw_status: RETRACTED\n"
            "retraction_reason: entity_fusion\n"
            "---\n"
            "Original raw body must remain byte-for-byte unchanged.\n"
        )
        retracted.write_text(retracted_text, encoding="utf-8")
        (raw_dir / "active.md").write_text(
            "---\nraw_status: active\n---\nactive body\n",
            encoding="utf-8",
        )
        (raw_dir / "legacy.md").write_text("legacy body\n", encoding="utf-8")
        (raw_dir / "unknown.md").write_text(
            "---\nraw_status: corrected\n---\nunknown status remains compatible\n",
            encoding="utf-8",
        )

        pending = {path.name for path in orchestrator.get_pending_raw_files()}

        assert pending == {"active.md", "legacy.md", "unknown.md"}
        assert retracted.read_text(encoding="utf-8") == retracted_text
        assert retracted.name not in orchestrator._load_state()["processed_raw_files"]

    def test_force_ingest_declines_when_only_raw_is_retracted(
        self, isolated_wiki: Path
    ) -> None:
        from llm_wiki_mcp import orchestrator

        (isolated_wiki / "raw" / "retracted.md").write_text(
            "---\nraw_status: retracted\n---\nbody\n",
            encoding="utf-8",
        )

        result = orchestrator.run_pending_ingest(force=True)

        assert result == {"triggered": False, "reason": "no pending raws"}

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

    def test_reset_stale_lock_keeps_live_cross_process_lock(
        self, isolated_wiki: Path
    ) -> None:
        from llm_wiki_mcp import orchestrator

        state = orchestrator._load_state()
        state["current_job_id"] = "job-in-another-process"
        state["current_job_pid"] = os.getpid()
        state["current_job_started_at"] = datetime.now().isoformat()
        orchestrator._save_state(state)

        orchestrator.reset_stale_lock()
        assert orchestrator._load_state()["current_job_id"] == "job-in-another-process"

    def test_reset_stale_lock_clears_dead_cross_process_lock(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from llm_wiki_mcp import orchestrator

        state = orchestrator._load_state()
        state["current_job_id"] = "job-from-dead-process"
        state["current_job_pid"] = 12345
        state["current_job_started_at"] = datetime.now().isoformat()
        orchestrator._save_state(state)
        monkeypatch.setattr(orchestrator, "_pid_is_alive", lambda _pid: False)

        orchestrator.reset_stale_lock()
        loaded = orchestrator._load_state()
        assert loaded["current_job_id"] is None
        assert loaded["current_job_pid"] is None
        assert loaded["current_job_started_at"] is None

    def test_run_pending_ingest_serial_then_idempotent(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Per-raw synchronous serial design: the first call ingests every
        pending raw individually; the second call sees them all marked
        processed and declines with a no-pending / threshold-not-met
        reason. Concurrency safety against true parallel callers is
        enforced by ``_INGEST_LOCK`` (in-process) and tested separately.
        """
        from llm_wiki_mcp import orchestrator

        # Make 5 fake raws so should_ingest() fires.
        for i in range(5):
            (isolated_wiki / "raw" / f"r{i}.md").write_text("body")

        # Stub run_ingest to simulate full-success: invoke on_complete so
        # the orchestrator marks each raw processed individually.
        captured = {"calls": 0, "metadata_keys": []}

        def fake_run_ingest(
            content,
            job_id,
            on_complete=None,
            on_finally=None,
            *,
            metadata=None,
        ):
            captured["calls"] += 1
            captured["metadata_keys"].append(
                sorted((metadata or {}).keys())
            )
            if on_complete:
                on_complete()
            if on_finally:
                on_finally(failed=False, triage_failed=False)

        from llm_wiki_mcp import ingest as ingest_mod

        monkeypatch.setattr(ingest_mod, "run_ingest", fake_run_ingest)
        monkeypatch.setattr(orchestrator, "is_available", lambda: True)

        first = orchestrator.run_pending_ingest()
        second = orchestrator.run_pending_ingest()

        assert first["triggered"] is True
        assert len(first["job_ids"]) == 5
        assert sorted(first["files_processed"]) == [f"r{i}.md" for i in range(5)]
        # Every raw fed metadata that includes the raw_keywords side channel.
        assert all("raw_keywords" in keys for keys in captured["metadata_keys"])
        assert captured["calls"] == 5

        # Second call: every raw is now marked processed → below threshold.
        assert second["triggered"] is False
        assert "threshold" in second["reason"].lower() or "pending" in second["reason"].lower()

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
# Phase 6: per-raw orchestrator invariants (attribution / mark / fallback)
# ---------------------------------------------------------------------------


class TestPerRawOrchestrator:
    """Verifies the Phase 2-5 contract end-to-end at the orchestrator
    level: each raw's keywords reach run_ingest as its own metadata,
    success and failure are tracked per-file, and legacy raws written
    before the field rename are still readable."""

    def test_attribution_two_raws_distinct_keywords(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Each raw's frontmatter keywords must reach run_ingest as that
        raw's own metadata — no blanket copy across the batch."""
        from llm_wiki_mcp import orchestrator, ingest as ingest_mod

        (isolated_wiki / "raw" / "a.md").write_text(
            "---\nraw_keywords: [alpha-1, alpha-2]\n---\nA body\n"
        )
        (isolated_wiki / "raw" / "b.md").write_text(
            "---\nraw_keywords: [beta-1]\n---\nB body\n"
        )
        # The orchestrator's threshold is 5; force=True bypasses it.
        for i in range(3):
            (isolated_wiki / "raw" / f"f{i}.md").write_text("body")

        seen: list[tuple[str, list[str] | None]] = []

        def fake_run_ingest(
            content,
            job_id,
            on_complete=None,
            on_finally=None,
            *,
            metadata=None,
        ):
            kw = (metadata or {}).get("raw_keywords")
            # Identify which raw this call belongs to by the leading body
            # line (we wrote distinct bodies above).
            tag = (
                "a"
                if "A body" in content
                else "b"
                if "B body" in content
                else "f"
            )
            seen.append((tag, kw))
            if on_complete:
                on_complete()
            if on_finally:
                on_finally(failed=False, triage_failed=False)

        monkeypatch.setattr(ingest_mod, "run_ingest", fake_run_ingest)
        monkeypatch.setattr(orchestrator, "is_available", lambda: True)

        result = orchestrator.run_pending_ingest(force=True)
        assert result["triggered"] is True
        # Every distinct raw_keywords payload was delivered to its OWN raw.
        a_calls = [kw for tag, kw in seen if tag == "a"]
        b_calls = [kw for tag, kw in seen if tag == "b"]
        f_calls = [kw for tag, kw in seen if tag == "f"]
        assert a_calls == [["alpha-1", "alpha-2"]]
        assert b_calls == [["beta-1"]]
        # The keyword-less raws got an empty list (not the neighbors' list).
        assert all(kw == [] for kw in f_calls)

    def test_legacy_keywords_field_falls_back(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Pre-rename raws written with ``keywords:`` (not ``raw_keywords:``)
        must still propagate via the legacy fallback path."""
        from llm_wiki_mcp import orchestrator, ingest as ingest_mod

        (isolated_wiki / "raw" / "legacy.md").write_text(
            "---\nkeywords: [legacy-only]\n---\nlegacy body\n"
        )

        seen: list[list[str] | None] = []

        def fake_run_ingest(
            content, job_id, on_complete=None, on_finally=None, *, metadata=None
        ):
            seen.append((metadata or {}).get("raw_keywords"))
            if on_complete:
                on_complete()
            if on_finally:
                on_finally(failed=False, triage_failed=False)

        monkeypatch.setattr(ingest_mod, "run_ingest", fake_run_ingest)
        monkeypatch.setattr(orchestrator, "is_available", lambda: True)

        orchestrator.run_pending_ingest(force=True)
        assert seen == [["legacy-only"]]

    def test_raw_keywords_preferred_over_legacy_when_both_present(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If a raw has both fields (transitional state), the new
        ``raw_keywords`` wins — fallback is a strict else branch."""
        from llm_wiki_mcp import orchestrator, ingest as ingest_mod

        (isolated_wiki / "raw" / "both.md").write_text(
            "---\nraw_keywords: [new-name]\nkeywords: [old-name]\n---\nbody\n"
        )

        seen: list[list[str] | None] = []

        def fake_run_ingest(
            content, job_id, on_complete=None, on_finally=None, *, metadata=None
        ):
            seen.append((metadata or {}).get("raw_keywords"))
            if on_complete:
                on_complete()
            if on_finally:
                on_finally(failed=False, triage_failed=False)

        monkeypatch.setattr(ingest_mod, "run_ingest", fake_run_ingest)
        monkeypatch.setattr(orchestrator, "is_available", lambda: True)

        orchestrator.run_pending_ingest(force=True)
        assert seen == [["new-name"]]

    def test_per_raw_mark_partial_failure(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A failed raw must remain pending while its successful peers
        get marked processed individually."""
        from llm_wiki_mcp import orchestrator, ingest as ingest_mod

        (isolated_wiki / "raw" / "ok.md").write_text("ok body")
        (isolated_wiki / "raw" / "broken.md").write_text("broken body")

        def fake_run_ingest(
            content, job_id, on_complete=None, on_finally=None, *, metadata=None
        ):
            # ``ok`` succeeds (calls on_complete), ``broken`` fails
            # (skips on_complete) — mirrors run_ingest's contract for
            # full success vs failure.
            if "ok body" in content:
                if on_complete:
                    on_complete()
                if on_finally:
                    on_finally(failed=False, triage_failed=False)
            else:
                if on_finally:
                    on_finally(failed=True, triage_failed=False)

        monkeypatch.setattr(ingest_mod, "run_ingest", fake_run_ingest)
        monkeypatch.setattr(orchestrator, "is_available", lambda: True)

        result = orchestrator.run_pending_ingest(force=True)

        assert result["files_processed"] == ["ok.md"]
        # ``broken.md`` is still pending for the next tick.
        pending = {p.name for p in orchestrator.get_pending_raw_files()}
        assert "broken.md" in pending
        assert "ok.md" not in pending

    def test_repeated_apply_failure_quarantines_raw(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Repeated deterministic apply failures leave the queue and become
        self-healing packets instead of being retried forever."""
        from llm_wiki_mcp import orchestrator, ingest as ingest_mod
        from llm_wiki_mcp.jobs import JobStatus, job_store

        raw_path = isolated_wiki / "raw" / "broken.md"
        raw_path.write_text("broken body")
        _seed_page(
            isolated_wiki,
            "ai/opus-4.7-evaluation-and-industry-geopolitics.md",
            "---\ntitle: Opus\nupdated: 2026-01-01\n---\nold",
        )

        def fake_run_ingest(
            content, job_id, on_complete=None, on_finally=None, *, metadata=None
        ):
            job_store.update(
                job_id,
                status=JobStatus.FAILED,
                error=(
                    "update target not found for page_id "
                    "'opus-4-7-evaluation-and-industry-geopolitics'"
                ),
            )
            if on_finally:
                on_finally(failed=True, triage_failed=False)

        monkeypatch.setattr(ingest_mod, "run_ingest", fake_run_ingest)
        monkeypatch.setattr(orchestrator, "is_available", lambda: True)

        first = orchestrator.run_pending_ingest(force=True)
        second = orchestrator.run_pending_ingest(force=True)
        third = orchestrator.run_pending_ingest(force=True)

        assert first["per_raw"][0]["supervision"]["attempts"] == 1
        assert second["per_raw"][0]["supervision"]["attempts"] == 2
        supervision = third["per_raw"][0]["supervision"]
        assert supervision["attempts"] == 3
        assert supervision["quarantined"] is True
        assert not raw_path.exists()
        packet_path = Path(supervision["packet_path"])
        assert packet_path.exists()
        packet = json.loads(packet_path.read_text())
        assert packet["failure_class"] == "apply.update_target_not_found"
        assert packet["similar_existing_pages"] == [
            "ai/opus-4.7-evaluation-and-industry-geopolitics"
        ]
        assert orchestrator.get_pending_raw_files() == []

    def test_frontier_nonconvergence_immediately_queues_self_heal(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from llm_wiki_mcp import failure_supervisor, self_heal

        raw_path = isolated_wiki / "raw" / "frontier-loop.md"
        raw_path.write_text("grounded source", encoding="utf-8")
        started: list[Path] = []
        monkeypatch.setattr(self_heal, "start_background", started.append)

        result = failure_supervisor.record_raw_failure(
            raw_path=raw_path,
            error=(
                "frontier ingest review did not converge after 3 attempts: "
                "taxonomy oscillated from scenario to event"
            ),
            raw_text="grounded source",
        )

        assert result.failure_class == "ingest.frontier_nonconvergent"
        assert result.attempts == 1
        assert result.quarantined is True
        assert result.packet_path is not None
        assert started == [Path(result.packet_path)]
        packet = json.loads(Path(result.packet_path).read_text(encoding="utf-8"))
        assert packet["fingerprint"] == "ingest.frontier_nonconvergent"
        assert not raw_path.exists()

    def test_ollama_unavailable_stays_pending_without_self_heal_packet(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A local model outage is infrastructure state, not a bad raw.

        It must not accumulate per-raw attempts or quarantine otherwise valid
        source material into the self-heal queue.
        """
        from llm_wiki_mcp import orchestrator, ingest as ingest_mod

        raw_path = isolated_wiki / "raw" / "model-memory.md"
        raw_path.write_text("body")

        monkeypatch.setattr(orchestrator, "is_available", lambda: False)
        monkeypatch.setattr(ingest_mod, "is_available", lambda: False)

        result = {}
        for _ in range(3):
            result = orchestrator.run_pending_ingest(force=True)

        supervision = result["per_raw"][0]["supervision"]
        assert supervision["failure_class"] == "ingest.ollama_unavailable"
        assert supervision["attempts"] == 0
        assert supervision["tracked"] is False
        assert supervision["transient"] is True
        assert supervision["quarantined"] is False
        assert raw_path.exists()
        assert orchestrator.get_pending_raw_files() == [raw_path]
        packets_dir = isolated_wiki / "runtime" / "failures" / "packets"
        assert not packets_dir.exists() or list(packets_dir.iterdir()) == []

    def test_legacy_sonnet_fallback_error_is_not_tracked_as_raw_failure(
        self, isolated_wiki: Path
    ) -> None:
        from llm_wiki_mcp import failure_supervisor

        raw_path = isolated_wiki / "raw" / "legacy.md"
        raw_path.write_text("body")

        result = failure_supervisor.record_raw_failure(
            raw_path=raw_path,
            error="Sonnet fallback not yet implemented",
            raw_text="body",
        )

        assert result.failure_class == "ingest.ollama_unavailable"
        assert result.attempts == 0
        assert result.tracked is False
        assert result.transient is True
        assert raw_path.exists()
        assert not (
            isolated_wiki / "runtime" / "failures" / "state.json"
        ).exists()

    def test_serial_execution_no_concurrent_threads(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The Phase 2 design replaces ``start_ingest`` (which spawned
        a worker thread) with a synchronous ``run_ingest`` call. We
        verify by counting the active thread delta around the batch:
        no per-raw worker thread should be spawned.
        """
        import threading
        from llm_wiki_mcp import orchestrator, ingest as ingest_mod

        for i in range(3):
            (isolated_wiki / "raw" / f"r{i}.md").write_text("body")

        thread_counts: list[int] = []

        def fake_run_ingest(
            content, job_id, on_complete=None, on_finally=None, *, metadata=None
        ):
            # Sample active thread count INSIDE each "ingest" invocation.
            thread_counts.append(threading.active_count())
            if on_complete:
                on_complete()
            if on_finally:
                on_finally(failed=False, triage_failed=False)

        monkeypatch.setattr(ingest_mod, "run_ingest", fake_run_ingest)
        monkeypatch.setattr(orchestrator, "is_available", lambda: True)

        baseline = threading.active_count()
        orchestrator.run_pending_ingest(force=True)

        # Every per-raw call ran on the same thread as the batch driver
        # (no Thread() per raw). Allow +/- 1 for unrelated background
        # threads, but assert no growth across calls.
        assert all(c <= baseline + 1 for c in thread_counts), thread_counts


# ---------------------------------------------------------------------------
# Phase 6: signature backward compatibility + field-naming independence
# ---------------------------------------------------------------------------


class TestPhase6Compatibility:
    def test_run_ingest_positional_signature_still_works(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The Phase 3 keyword-only ``metadata`` parameter must not
        break callers that still pass the original 4 positional args."""
        from llm_wiki_mcp import ingest, jobs

        plan = [{"type": "create", "filename": "misc/p.md", "title": "P"}]
        monkeypatch.setattr(ingest, "_triage", lambda _content: plan)
        monkeypatch.setattr(
            ingest,
            "_generate_one",
            lambda _op, _raw, **_kw: {
                "type": "create",
                "filename": "misc/p.md",
                "content": "---\ntitle: P\nupdated: 2026-04-28\n---\nbody",
            },
        )
        monkeypatch.setattr(ingest, "is_available", lambda: True)

        on_complete_called = []
        on_finally_called = []
        job = jobs.job_store.create(processor="ollama")

        # Original 4-positional call shape — no metadata kwarg.
        ingest.run_ingest(
            "raw",
            job.job_id,
            lambda: on_complete_called.append(True),
            lambda failed, triage_failed: on_finally_called.append(
                (failed, triage_failed)
            ),
        )

        finished = jobs.job_store.get(job.job_id)
        assert finished.status == jobs.JobStatus.COMPLETED
        assert finished.pages_created == ["p"]
        assert on_complete_called == [True]
        assert on_finally_called == [(False, False)]

    def test_triage_keywords_field_independent_of_raw_keywords(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The triage plan dict can carry its own ``keywords`` field for
        related-page search (``ingest._build_focused_context``) without
        being conflated with the raw frontmatter's ``raw_keywords``.
        Verify by feeding a triage plan with ``keywords`` AND a metadata
        ``raw_keywords`` — they must reach the operation as two distinct
        signals."""
        from llm_wiki_mcp import ingest, jobs

        plan = [
            {
                "type": "create",
                "filename": "misc/p.md",
                "title": "P",
                "keywords": ["search-only-1", "search-only-2"],
            }
        ]
        monkeypatch.setattr(ingest, "_triage", lambda _content: plan)

        captured_op: dict = {}
        captured_kw: list[list[str] | None] = []

        def stub_generate(op, _raw, *, raw_keywords=None):
            # The op coming into generate still has its triage-side
            # ``keywords`` field — that's the search-related signal.
            captured_op.update(op)
            captured_kw.append(raw_keywords)
            return {
                "type": "create",
                "filename": op["filename"],
                "content": "---\ntitle: P\nupdated: 2026-04-28\n---\nbody",
            }

        monkeypatch.setattr(ingest, "_generate_one", stub_generate)
        monkeypatch.setattr(ingest, "is_available", lambda: True)

        job = jobs.job_store.create(processor="ollama")
        ingest.run_ingest(
            "raw",
            job.job_id,
            metadata={"raw_keywords": ["fm-only"]},
        )

        # Triage's ``keywords`` survived on the op (used downstream by
        # _build_focused_context), distinct from raw_keywords.
        assert captured_op.get("keywords") == ["search-only-1", "search-only-2"]
        # raw_keywords reached generate as the dedicated metadata channel,
        # NOT mixed into the triage keywords list.
        assert captured_kw == [["fm-only"]]


class TestSearchBeforeCreate:
    def test_create_with_existing_title_is_rewritten_to_update(
        self, isolated_wiki: Path
    ) -> None:
        from llm_wiki_mcp import ingest

        _seed_page(
            isolated_wiki,
            "work/workplace-stress-and-mentoring.md",
            "---\n"
            "title: Workplace Stress and Mentoring\n"
            "updated: 2026-07-01\n"
            "tags: [d/career, t/analysis, s/2026]\n"
            "---\n"
            "Existing body.\n",
        )
        plan = [
            {
                "type": "create",
                "filename": "work/workplace-stress-mentoring.md",
                "title": "Workplace Stress and Mentoring",
                "summary": "Mentoring and workplace stress notes.",
            }
        ]

        rewritten = ingest._dedupe_create_ops_with_existing(plan, "raw")

        assert rewritten[0]["type"] == "update"
        assert rewritten[0]["filename"] == "work/workplace-stress-and-mentoring.md"
        assert rewritten[0]["existing_page_id"] == "workplace-stress-and-mentoring"


class TestReadBackVerification:
    def test_changed_page_passes_when_search_returns_page(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from llm_wiki_mcp import index_store, ingest, search
        from llm_wiki_mcp.search_types import ScoredPage

        class Store:
            def refresh(self) -> None:
                pass

            def meta(self, page_id: str) -> dict:
                return {
                    "page_id": page_id,
                    "title": "P",
                    "updated": "2026-07-06",
                    "path": str(isolated_wiki / "pages" / "p.md"),
                    "summary": "Durable test summary",
                    "recall_questions": ["How do we recall P?"],
                }

        monkeypatch.setattr(index_store, "get_store", lambda: Store())
        monkeypatch.setattr(
            search,
            "search",
            lambda _query, top_n=10, semantic=True: (
                [ScoredPage("p", "P", "", "2026-07-06", 1.0)],
                "hybrid",
            ),
        )

        result = ingest._verify_changed_pages_read_back(["p"])

        assert result == {"checked": 1, "passed": 1, "failed": []}


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

    def test_live_triage_missing_update_is_retyped_to_create(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from llm_wiki_mcp import ingest

        plan = [
            {
                "type": "update",
                "filename": "career-transition-strategy-2026.md",
                "summary": "Career transition strategy memory",
            }
        ]
        monkeypatch.setattr(
            ingest,
            "_generate_with_progress",
            lambda *_args, **_kwargs: json.dumps(plan),
        )

        out = ingest._triage("raw content")

        assert out == [
            {
                "type": "create",
                "filename": "career-transition-strategy-2026.md",
                "summary": "Career transition strategy memory",
                "title": "Career Transition Strategy 2026",
                "keywords": ["career", "transition", "strategy", "2026"],
            }
        ]

    def test_live_triage_existing_update_stays_update(
        self, isolated_wiki: Path
    ) -> None:
        from llm_wiki_mcp.ingest import _validate_triage_plan

        _seed_page(
            isolated_wiki,
            "career-transition-strategy-2026.md",
            "---\ntitle: Career\nupdated: 2026-01-01\n---\nold",
        )
        plan = [
            {
                "type": "update",
                "filename": "career-transition-strategy-2026.md",
                "summary": "Append new memory",
            }
        ]

        assert _validate_triage_plan(plan, coerce_missing_updates=True) == plan

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
    def test_collision_converts_without_blocking_batch(
        self, isolated_wiki: Path
    ) -> None:
        """A create for an existing page_id is treated as an update, so it
        does not strand unrelated successful operations in the same batch."""
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
        created, updated = _apply_operations(ops)

        assert created == ["safe-page"]
        assert updated == ["blocking"]
        assert (isolated_wiki / "pages" / "fresh" / "safe-page.md").exists()
        text = existing.read_text()
        assert "original" in text
        assert "\nx\n" in text
        assert "title: Dup" not in text

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

    def test_casefold_collision_converts_to_update(self, isolated_wiki: Path) -> None:
        """``Foo.md`` and ``foo.md`` resolve to the same inode on
        case-insensitive macOS filesystems. We route the create into the
        existing page rather than writing a second logical duplicate."""
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
        created, updated = _apply_operations(ops)

        assert created == []
        assert updated == ["Foo"]
        text = existing.read_text()
        assert "body" in text
        assert "\nx\n" in text
        assert not (isolated_wiki / "pages" / "b" / "foo.md").exists()


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


class TestWikiSaveRawRouting:
    def test_idempotency_key_reuses_first_complete_raw(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from llm_wiki_mcp import orchestrator, server

        tool_fn = server.wiki_save_raw.fn if hasattr(server.wiki_save_raw, "fn") else server.wiki_save_raw
        monkeypatch.setattr(server, "RAW_DIR", isolated_wiki / "raw")
        monkeypatch.setattr(orchestrator, "should_ingest", lambda: (False, "below threshold"))
        key = "codex-0123456789abcdef01234567-from0-to5"

        first = json.loads(
            tool_fn("first complete payload", trigger_ingest=False, idempotency_key=key)
        )
        second = json.loads(
            tool_fn("first complete payload", trigger_ingest=False, idempotency_key=key)
        )

        assert first["saved"] == f"save-{key}.md"
        assert second["saved"] == first["saved"]
        assert first["deduplicated"] is False
        assert second["deduplicated"] is True
        assert (isolated_wiki / "raw" / first["saved"]).read_text() == "first complete payload"
        assert len(list((isolated_wiki / "raw").glob("*.md"))) == 1

    def test_idempotency_key_rejects_different_unverified_payload(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from llm_wiki_mcp import orchestrator, server

        tool_fn = server.wiki_save_raw.fn if hasattr(server.wiki_save_raw, "fn") else server.wiki_save_raw
        monkeypatch.setattr(server, "RAW_DIR", isolated_wiki / "raw")
        monkeypatch.setattr(orchestrator, "should_ingest", lambda: (False, "below threshold"))
        key = "codex-0123456789abcdef01234567-from0-to5"

        tool_fn("first complete payload", trigger_ingest=False, idempotency_key=key)
        with pytest.raises(RuntimeError, match="idempotency key collision"):
            tool_fn("corrupt retry payload", trigger_ingest=False, idempotency_key=key)

        target = isolated_wiki / "raw" / f"save-{key}.md"
        assert target.read_text() == "first complete payload"

    def test_idempotency_key_accepts_different_self_verified_retry_receipt(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from llm_wiki_mcp import orchestrator, server
        from llm_wiki_mcp.save_transaction import (
            attach_save_transaction_marker,
            make_save_transaction,
        )

        tool_fn = server.wiki_save_raw.fn if hasattr(server.wiki_save_raw, "fn") else server.wiki_save_raw
        monkeypatch.setattr(server, "RAW_DIR", isolated_wiki / "raw")
        monkeypatch.setattr(orchestrator, "should_ingest", lambda: (False, "below threshold"))
        transaction = make_save_transaction(
            host="codex",
            session_file=isolated_wiki / "session.jsonl",
            session_id="session-1",
            after_line=0,
            until_line=5,
        )
        first_content = attach_save_transaction_marker(transaction, "first writer output")
        retry_content = attach_save_transaction_marker(transaction, "different retry output")

        first = json.loads(
            tool_fn(
                first_content,
                trigger_ingest=False,
                idempotency_key=transaction.idempotency_key,
            )
        )
        retry = json.loads(
            tool_fn(
                retry_content,
                trigger_ingest=False,
                idempotency_key=transaction.idempotency_key,
            )
        )

        assert retry["saved"] == first["saved"]
        assert retry["deduplicated"] is True
        assert (isolated_wiki / "raw" / first["saved"]).read_text() == first_content

    def test_idempotency_key_rejects_corrupt_existing_receipt(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from llm_wiki_mcp import orchestrator, server
        from llm_wiki_mcp.save_transaction import (
            attach_save_transaction_marker,
            make_save_transaction,
        )

        tool_fn = server.wiki_save_raw.fn if hasattr(server.wiki_save_raw, "fn") else server.wiki_save_raw
        monkeypatch.setattr(server, "RAW_DIR", isolated_wiki / "raw")
        monkeypatch.setattr(orchestrator, "should_ingest", lambda: (False, "below threshold"))
        transaction = make_save_transaction(
            host="codex",
            session_file=isolated_wiki / "session.jsonl",
            session_id="session-1",
            after_line=0,
            until_line=5,
        )
        content = attach_save_transaction_marker(transaction, "complete payload")
        result = json.loads(
            tool_fn(
                content,
                trigger_ingest=False,
                idempotency_key=transaction.idempotency_key,
            )
        )
        target = isolated_wiki / "raw" / result["saved"]
        target.write_text(content.replace("complete payload", "corrupt payload"))

        with pytest.raises(RuntimeError, match="different or corrupt"):
            tool_fn(
                content,
                trigger_ingest=False,
                idempotency_key=transaction.idempotency_key,
            )

    def test_trigger_ingest_false_defers_threshold_ingest(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from llm_wiki_mcp import server, orchestrator

        tool_fn = server.wiki_save_raw.fn if hasattr(server.wiki_save_raw, "fn") else server.wiki_save_raw
        monkeypatch.setattr(server, "RAW_DIR", isolated_wiki / "raw")
        monkeypatch.setattr(orchestrator, "should_ingest", lambda: (True, "threshold met"))

        captured = {"called": False}

        def fake_run_pending_ingest() -> dict:
            captured["called"] = True
            return {"triggered": True}

        monkeypatch.setattr(orchestrator, "run_pending_ingest", fake_run_pending_ingest)

        result = json.loads(tool_fn("hello world content", trigger_ingest=False))

        assert result["ingest_pending"] is True
        assert result["ingest_deferred"] is True
        assert "ingest_triggered" not in result
        assert captured["called"] is False

    def test_save_raw_filename_includes_readable_keyword_slug(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from llm_wiki_mcp import server, orchestrator

        tool_fn = server.wiki_save_raw.fn if hasattr(server.wiki_save_raw, "fn") else server.wiki_save_raw
        monkeypatch.setattr(server, "RAW_DIR", isolated_wiki / "raw")
        monkeypatch.setattr(orchestrator, "should_ingest", lambda: (False, "below threshold"))

        result = json.loads(
            tool_fn(
                "body",
                session_id="codex-019e80cb-8df9-7aa1-ba75-cc58c6f759ae",
                keywords=["llm-wiki", "dashboard", "self-heal"],
                trigger_ingest=False,
            )
        )

        assert result["saved"].startswith("20")
        assert "-codex-llm-wiki-dashboard-self-heal-" in result["saved"]
        assert result["saved"].endswith(".md")
        assert result["raw_slug"] == "llm-wiki-dashboard-self-heal"

    def test_save_raw_filename_falls_back_to_content_slug(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from llm_wiki_mcp import server, orchestrator

        tool_fn = server.wiki_save_raw.fn if hasattr(server.wiki_save_raw, "fn") else server.wiki_save_raw
        monkeypatch.setattr(server, "RAW_DIR", isolated_wiki / "raw")
        monkeypatch.setattr(orchestrator, "should_ingest", lambda: (False, "below threshold"))

        result = json.loads(
            tool_fn(
                "# Codex Session Memory Save\n\n## Memory\n\nDashboard filename cleanup",
                session_id="codex-019e80cb-8df9-7aa1-ba75-cc58c6f759ae",
                keywords=[],
                trigger_ingest=False,
            )
        )

        assert "-codex-dashboard-filename-cleanup-" in result["saved"]


# ---------------------------------------------------------------------------
# R4-Critical: log failures must not break rollback inclusion
# ---------------------------------------------------------------------------


class TestLogFailuresDontBreakRollback:
    def test_shared_lock_entry_rechecks_update_preimage(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A correction committed after ingest prepare must not be overwritten."""
        from contextlib import contextmanager
        from llm_wiki_mcp import page_mutation

        target = isolated_wiki / "pages" / "x" / "page.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            "---\ntitle: X\nupdated: 2026-01-01\n---\noriginal\n"
        )
        correction = (
            "---\ntitle: X\nupdated: 2026-07-11\n---\nfrontier correction\n"
        )

        @contextmanager
        def correction_wins_before_ingest_commit():
            target.write_text(correction)
            yield

        monkeypatch.setattr(
            page_mutation,
            "wiki_mutation_lock",
            correction_wins_before_ingest_commit,
        )

        with pytest.raises(IngestApplyError, match="page changed before ingest apply"):
            _apply_operations(
                [{"type": "update", "filename": "page.md", "content": "stale addendum"}]
            )

        assert target.read_text() == correction

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

    def test_allocate_raw_path_creates_raw_dir_if_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from llm_wiki_mcp import server

        raw_dir = tmp_path / "wiki" / "raw"
        monkeypatch.setattr(server, "RAW_DIR", raw_dir)

        path = server._allocate_raw_path(prefix="codex", topic_slug="readable-name")

        assert path.exists()
        assert path.parent == raw_dir
        assert "-codex-readable-name-" in path.name


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

        def _noop_run_ingest(
            content, job_id, on_complete=None, on_finally=None, *, metadata=None
        ):
            if on_complete:
                on_complete()
            if on_finally:
                on_finally(failed=False, triage_failed=False)

        monkeypatch.setattr(ingest_mod, "run_ingest", _noop_run_ingest)

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
            lambda _op, _raw, **_kw: {
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
            lambda _op, _raw, **_kw: {
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
    def test_save_publish_is_invisible_to_ingest_until_content_is_complete(
        self, isolated_wiki: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Ingest must never observe the zero-byte reservation window."""
        from concurrent.futures import ThreadPoolExecutor
        import threading
        from llm_wiki_mcp import orchestrator, server

        monkeypatch.setattr(server, "RAW_DIR", isolated_wiki / "raw")
        tool_fn = server.wiki_save_raw.fn if hasattr(server.wiki_save_raw, "fn") else server.wiki_save_raw
        real_link = server._link_raw_no_replace
        ready_to_publish = threading.Event()
        allow_publish = threading.Event()

        def paused_link(staging: Path, target: Path) -> None:
            # _publish_raw reaches this point only after write + fsync.
            ready_to_publish.set()
            assert allow_publish.wait(5)
            real_link(staging, target)

        monkeypatch.setattr(server, "_link_raw_no_replace", paused_link)
        content = "complete raw payload\n" * 100

        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(
                tool_fn,
                content,
                "codex-test-session",
                ["atomic-publish"],
                False,
            )
            assert ready_to_publish.wait(5)
            try:
                assert orchestrator.get_pending_raw_files() == []
                assert list((isolated_wiki / "raw").glob("*.md")) == []
                staging = list((isolated_wiki / "raw").glob(".*.tmp"))
                assert len(staging) == 1
                staged_content = staging[0].read_text()
                assert staged_content.endswith(content)
                assert "raw_keywords: [atomic-publish]" in staged_content
            finally:
                allow_publish.set()
            result = json.loads(future.result(timeout=5))

        published = isolated_wiki / "raw" / result["saved"]
        assert published.read_text() == staged_content
        assert orchestrator.get_pending_raw_files() == [published]
        assert list((isolated_wiki / "raw").glob(".*.tmp")) == []

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
            lambda _op, _raw, **_kw: {
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
