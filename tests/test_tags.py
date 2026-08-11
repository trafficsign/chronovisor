"""Tests for tag taxonomy + generation rules."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from chronovisor.core import page_mutation
from chronovisor.ingest import tag_lifecycle as tags_mod
from chronovisor.ingest.tag_lifecycle import (
    AXIS_LIMITS,
    SEED_TAGS,
    VALID_PREFIXES,
    dedupe_with_existing,
    parse_tags,
    record_new_tag,
    validate_axis_counts,
    validate_tag,
)


@pytest.fixture()
def isolated_changelog(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    system = tmp_path / "system"
    system.mkdir()
    (tmp_path / "pages").mkdir()
    for name in ("index.md", "log.md", "schema.md"):
        (tmp_path / name).write_text(f"# {name}\n", encoding="utf-8")
    monkeypatch.setattr(tags_mod, "SYSTEM_DIR", system)
    monkeypatch.setattr(page_mutation, "CHRONOVISOR_ROOT", tmp_path)
    monkeypatch.setattr(page_mutation, "PAGES_DIR", tmp_path / "pages")
    monkeypatch.setattr(page_mutation, "SYSTEM_DIR", system)
    return system


# ---------------------------------------------------------------------------
# validate_tag — form rules 1-6
# ---------------------------------------------------------------------------


class TestValidateTag:
    @pytest.mark.parametrize(
        "tag",
        [
            "d/ai-industry",
            "t/analysis",
            "s/2026",
            "s/evergreen",
            "d/personal-strategy",  # 2 words
        ],
    )
    def test_valid_tags(self, tag: str) -> None:
        ok, reason = validate_tag(tag)
        assert ok, reason

    @pytest.mark.parametrize(
        "tag,reason_substr",
        [
            ("ai-industry", "missing required prefix"),
            ("d/", "empty body"),
            ("d/AI-Industry", "kebab-case"),
            ("d/ai_industry", "kebab-case"),  # underscore not allowed
            ("d/2026", "starting with a letter"),  # digit-leading on d/
            ("t/2026", "starting with a letter"),  # digit-leading on t/
            ("d/three-word-tag", "too many words"),
            ("d/", "empty body"),
            ("", "empty"),
            ("  d/foo  ", "whitespace"),
        ],
    )
    def test_invalid_tags(self, tag: str, reason_substr: str) -> None:
        ok, reason = validate_tag(tag)
        assert not ok
        assert reason_substr.lower() in reason.lower(), (
            f"expected {reason_substr!r} in {reason!r}"
        )

    def test_non_string(self) -> None:
        for bad in (None, 42, ["d/foo"], {"k": "v"}):
            ok, _ = validate_tag(bad)  # type: ignore[arg-type]
            assert not ok


# ---------------------------------------------------------------------------
# parse_tags / validate_axis_counts
# ---------------------------------------------------------------------------


class TestParseAxisCounts:
    def test_parse_groups_by_prefix(self) -> None:
        parsed = parse_tags(
            [
                "d/ai-industry",
                "d/finance",
                "t/analysis",
                "s/2026",
                "no-prefix",
            ]
        )
        assert parsed["d/"] == ["d/ai-industry", "d/finance"]
        assert parsed["t/"] == ["t/analysis"]
        assert parsed["s/"] == ["s/2026"]
        assert parsed[""] == ["no-prefix"]

    def test_axis_count_ok_at_min(self) -> None:
        parsed = parse_tags(["d/ai-industry", "t/analysis", "s/2026"])
        assert validate_axis_counts(parsed) == []

    def test_axis_count_ok_at_max_for_d(self) -> None:
        parsed = parse_tags(
            [
                "d/ai-industry",
                "d/finance",
                "d/japan",  # 3 of 3 max
                "t/analysis",
                "s/2026",
            ]
        )
        assert validate_axis_counts(parsed) == []

    def test_too_few_d_tags(self) -> None:
        parsed = parse_tags(["t/analysis", "s/2026"])
        msgs = validate_axis_counts(parsed)
        assert any("d/" in m and "at least 1" in m for m in msgs)

    def test_too_many_d_tags(self) -> None:
        parsed = parse_tags(
            [
                "d/ai-industry",
                "d/finance",
                "d/japan",
                "d/health",  # 4 > max 3
                "t/analysis",
                "s/2026",
            ]
        )
        msgs = validate_axis_counts(parsed)
        assert any("d/" in m and "at most 3" in m for m in msgs)

    def test_too_many_t_tags(self) -> None:
        parsed = parse_tags(
            ["d/ai-industry", "t/analysis", "t/howto", "s/2026"]
        )
        msgs = validate_axis_counts(parsed)
        assert any("t/" in m and "at most 1" in m for m in msgs)

    def test_missing_s_tag(self) -> None:
        parsed = parse_tags(["d/ai-industry", "t/analysis"])
        msgs = validate_axis_counts(parsed)
        assert any("s/" in m and "at least 1" in m for m in msgs)

    def test_unknown_prefix_surfaced(self) -> None:
        parsed = parse_tags(["d/ai-industry", "t/analysis", "s/2026", "x/foo"])
        msgs = validate_axis_counts(parsed)
        assert any("unknown prefix" in m for m in msgs)


# ---------------------------------------------------------------------------
# dedupe_with_existing — embedding-driven, soft-fail
# ---------------------------------------------------------------------------


class TestDedupeWithExisting:
    def test_returns_existing_when_above_threshold(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from chronovisor.core import embedding as emb_mod

        # Force ``most_similar`` to claim an existing tag is super close.
        def fake_most_similar(query, candidates, threshold):
            return ("d/ai-industry", 0.95)

        monkeypatch.setattr(emb_mod, "most_similar", fake_most_similar)

        result = dedupe_with_existing("d/ai-industries", ["d/ai-industry", "d/finance"])
        assert result == "d/ai-industry"

    def test_returns_new_when_below_threshold(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from chronovisor.core import embedding as emb_mod

        monkeypatch.setattr(emb_mod, "most_similar", lambda *_a, **_k: None)
        result = dedupe_with_existing("d/blockchain", ["d/finance"])
        assert result == "d/blockchain"

    def test_only_compares_same_axis(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A new ``d/`` tag must not collapse into a ``t/`` tag even if
        the bodies are textually close."""
        from chronovisor.core import embedding as emb_mod

        captured: dict = {}

        def fake_most_similar(query, candidates, threshold):
            captured["candidates"] = list(candidates)
            return None

        monkeypatch.setattr(emb_mod, "most_similar", fake_most_similar)
        dedupe_with_existing(
            "d/analysis-stuff",
            ["d/ai-industry", "t/analysis", "s/2026"],
        )
        assert captured["candidates"] == ["d/ai-industry"]

    def test_invalid_tag_passes_through(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If the candidate is malformed, dedup is a no-op so the form
        validator can flag it later."""
        from chronovisor.core import embedding as emb_mod

        def boom(*_a, **_k):
            raise AssertionError("must not call")

        monkeypatch.setattr(emb_mod, "most_similar", boom)
        assert dedupe_with_existing("not-a-real-tag", ["d/foo"]) == "not-a-real-tag"

    def test_embedding_failure_falls_back_to_new(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from chronovisor.core import embedding as emb_mod

        def boom(*_a, **_k):
            raise RuntimeError("ollama down")

        monkeypatch.setattr(emb_mod, "most_similar", boom)
        assert dedupe_with_existing("d/new", ["d/old"]) == "d/new"

    def test_no_same_axis_candidates(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from chronovisor.core import embedding as emb_mod

        def boom(*_a, **_k):
            raise AssertionError("must not call when no candidates")

        monkeypatch.setattr(emb_mod, "most_similar", boom)
        assert dedupe_with_existing("d/new", ["t/howto", "s/2026"]) == "d/new"


# ---------------------------------------------------------------------------
# record_new_tag — append-only changelog
# ---------------------------------------------------------------------------


class TestRecordNewTag:
    def test_creates_file_with_header(self, isolated_changelog: Path) -> None:
        record_new_tag("d/blockchain", reason="ingest:first-occurrence")
        text = (isolated_changelog / "tag-changelog.md").read_text()
        assert "# Tag Changelog" in text
        assert "d/blockchain" in text
        assert "ingest:first-occurrence" in text

    def test_appends_to_existing_file(self, isolated_changelog: Path) -> None:
        record_new_tag("d/foo")
        record_new_tag("d/bar")
        text = (isolated_changelog / "tag-changelog.md").read_text()
        assert "d/foo" in text and "d/bar" in text

    def test_same_day_dedupe(self, isolated_changelog: Path) -> None:
        record_new_tag("d/dup")
        record_new_tag("d/dup")
        text = (isolated_changelog / "tag-changelog.md").read_text()
        assert text.count("d/dup") == 1

    def test_concurrent_append_is_lossless_and_deduplicated(
        self,
        isolated_changelog: Path,
    ) -> None:
        values = [f"d/tag-{index % 5}" for index in range(40)]
        with ThreadPoolExecutor(max_workers=8) as executor:
            list(executor.map(record_new_tag, values))

        text = (isolated_changelog / "tag-changelog.md").read_text(encoding="utf-8")
        for index in range(5):
            assert text.count(f"d/tag-{index}") == 1

    def test_io_failure_swallowed(
        self, isolated_changelog: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Replace SYSTEM_DIR with a path under a file (cannot mkdir).
        bad = isolated_changelog / "blocker.md"
        bad.write_text("not a dir")
        monkeypatch.setattr(tags_mod, "SYSTEM_DIR", bad / "subdir")
        # Must not raise.
        record_new_tag("d/foo")


# ---------------------------------------------------------------------------
# Sanity: every seed tag in SEED_TAGS validates cleanly
# ---------------------------------------------------------------------------


class TestSeedTags:
    def test_every_seed_validates(self) -> None:
        for prefix, lst in SEED_TAGS.items():
            for tag in lst:
                ok, reason = validate_tag(tag)
                assert ok, f"seed {tag!r} invalid: {reason}"
                assert tag.startswith(prefix)

    def test_axis_limits_match_taxonomy(self) -> None:
        # Each axis defined in SEED_TAGS must have a limit, and vice versa.
        assert set(AXIS_LIMITS.keys()) == set(SEED_TAGS.keys())
        assert set(VALID_PREFIXES) == set(AXIS_LIMITS.keys())
