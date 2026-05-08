"""Tests for orphan_link module (plan-2).

The LLM and semantic search are both mocked so these tests run fast and
deterministically. The real-LLM dry-run is exercised by the script
end-to-end at runtime, not in CI.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_wiki_mcp import orphan_link as ol_mod
from llm_wiki_mcp.orphan_link import (
    OrphanReport,
    Suggestion,
    format_report,
    gather_candidates,
    parse_llm_response,
    run_dry_run,
    score_candidate,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeStore:
    """In-memory IndexStore stand-in. Just enough for orphan_link to work."""

    def __init__(self) -> None:
        self.pages: dict[str, dict] = {}
        # backlinks[target] = [source, ...]
        self.backlinks_map: dict[str, list[str]] = {}

    def add_page(
        self,
        page_id: str,
        *,
        title: str | None = None,
        is_system: bool = False,
        body: str = "",
    ) -> None:
        self.pages[page_id] = {
            "page_id": page_id,
            "title": title or page_id,
            "is_system": is_system,
            "updated": "2026-05-08",
            "body": body,
        }

    def link(self, source: str, target: str) -> None:
        self.backlinks_map.setdefault(target, []).append(source)

    def meta(self, page_id: str):
        p = self.pages.get(page_id)
        if not p:
            return None
        return {
            "page_id": p["page_id"],
            "title": p["title"],
            "is_system": p["is_system"],
            "updated": p["updated"],
        }

    def backlinks(self, page_id: str) -> list[str]:
        return list(self.backlinks_map.get(page_id, []))

    def orphans(self, include_system: bool = False) -> list[str]:
        out = []
        for pid, p in self.pages.items():
            if not include_system and p["is_system"]:
                continue
            if not self.backlinks_map.get(pid):
                out.append(pid)
        return sorted(out)

    def all_page_ids(self, include_system: bool = True) -> set[str]:
        if include_system:
            return set(self.pages.keys())
        return {pid for pid, p in self.pages.items() if not p["is_system"]}

    def refresh(self) -> None:
        pass


@pytest.fixture()
def isolated_pages(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Stub ``find_page`` so ``_page_head`` reads from tmp_path instead
    of the real corpus."""
    pages_dir = tmp_path / "pages"
    pages_dir.mkdir()

    def fake_find_page(page_id: str):
        cand = pages_dir / f"{page_id}.md"
        return cand if cand.exists() else None

    monkeypatch.setattr(ol_mod, "find_page", fake_find_page)
    return pages_dir


def _seed_page(pages_dir: Path, page_id: str, body: str = "body content") -> None:
    (pages_dir / f"{page_id}.md").write_text(
        f"---\ntitle: {page_id}\nupdated: 2026-05-08\n---\n{body}\n"
    )


class _ScoredPage:
    """Minimal duck-type for what ``gather_candidates`` reads from a search hit."""

    def __init__(self, page_id: str, score: float = 0.5):
        self.page_id = page_id
        self.score = score


# ---------------------------------------------------------------------------
# parse_llm_response — the contract is strict
# ---------------------------------------------------------------------------


class TestParseLLMResponse:
    def test_valid_json_round_trip(self) -> None:
        raw = json.dumps(
            {
                "confidence": 0.82,
                "reason": "両方とも MCP のキーワード機能を扱う",
                "suggested_anchor": "MCP のキーワード",
                "suggested_section": "関連",
            }
        )
        parsed = parse_llm_response(raw)
        assert parsed is not None
        assert parsed["confidence"] == pytest.approx(0.82)

    def test_strips_code_fences(self) -> None:
        body = json.dumps(
            {
                "confidence": 0.5,
                "reason": "x",
                "suggested_anchor": "",
                "suggested_section": "関連",
            }
        )
        raw = f"```json\n{body}\n```"
        assert parse_llm_response(raw) is not None

    def test_extra_field_rejected(self) -> None:
        raw = json.dumps(
            {
                "confidence": 0.6,
                "reason": "ok",
                "suggested_anchor": "",
                "suggested_section": "関連",
                "page_id": "should-not-be-here",  # fabrication attempt
            }
        )
        assert parse_llm_response(raw) is None

    def test_missing_field_rejected(self) -> None:
        raw = json.dumps({"confidence": 0.6, "reason": "x"})
        assert parse_llm_response(raw) is None

    @pytest.mark.parametrize("bad_conf", [-0.1, 1.5, "0.5", None])
    def test_bad_confidence_rejected(self, bad_conf) -> None:
        raw = json.dumps(
            {
                "confidence": bad_conf,
                "reason": "x",
                "suggested_anchor": "",
                "suggested_section": "",
            }
        )
        assert parse_llm_response(raw) is None

    def test_non_json_rejected(self) -> None:
        assert parse_llm_response("hello world") is None
        assert parse_llm_response("") is None


# ---------------------------------------------------------------------------
# gather_candidates — filtering + ordering
# ---------------------------------------------------------------------------


class TestGatherCandidates:
    def test_skips_system_orphan_self(self, isolated_pages: Path) -> None:
        store = _FakeStore()
        store.add_page("orphan", title="Orphan")
        store.add_page("p1", title="P1")
        store.add_page("p2", title="P2")
        store.add_page("sys1", title="Sys", is_system=True)
        _seed_page(isolated_pages, "orphan")

        # Pretend semantic search returned everything.
        def fake_search(query, top_n):
            return [_ScoredPage(pid, 0.5) for pid in ["orphan", "sys1", "p1", "p2"]]

        result = gather_candidates(
            "orphan", store, semantic_search_fn=fake_search, max_candidates=10
        )
        assert "orphan" not in result
        assert "sys1" not in result
        assert set(result) == {"p1", "p2"}

    def test_well_connected_first(self, isolated_pages: Path) -> None:
        store = _FakeStore()
        store.add_page("orphan", title="Orphan")
        store.add_page("hub", title="Hub")  # 3 backlinks
        store.add_page("mid", title="Mid")  # 1 backlink
        store.add_page("lone", title="Lone")  # 0 backlinks
        store.link("a", "hub")
        store.link("b", "hub")
        store.link("c", "hub")
        store.link("a", "mid")
        _seed_page(isolated_pages, "orphan")

        def fake_search(query, top_n):
            # Return in shuffled order to ensure the function does the sort.
            return [
                _ScoredPage("lone", 0.9),
                _ScoredPage("mid", 0.5),
                _ScoredPage("hub", 0.1),
            ]

        result = gather_candidates(
            "orphan", store, semantic_search_fn=fake_search, max_candidates=3
        )
        assert result == ["hub", "mid", "lone"]

    def test_max_candidates_truncates(self, isolated_pages: Path) -> None:
        store = _FakeStore()
        store.add_page("orphan")
        for i in range(8):
            store.add_page(f"p{i}")
        _seed_page(isolated_pages, "orphan")

        def fake_search(query, top_n):
            return [_ScoredPage(f"p{i}", 0.5) for i in range(8)]

        result = gather_candidates(
            "orphan", store, semantic_search_fn=fake_search, max_candidates=3
        )
        assert len(result) == 3


# ---------------------------------------------------------------------------
# score_candidate
# ---------------------------------------------------------------------------


class TestScoreCandidate:
    def test_happy_path(self, isolated_pages: Path) -> None:
        store = _FakeStore()
        store.add_page("src", title="Source")
        store.add_page("orph", title="Orphan")
        _seed_page(isolated_pages, "src")
        _seed_page(isolated_pages, "orph")

        def fake_generate(prompt: str, system: str | None = None) -> str:
            return json.dumps(
                {
                    "confidence": 0.7,
                    "reason": "両方 LLM 関連",
                    "suggested_anchor": "LLM",
                    "suggested_section": "関連",
                }
            )

        out = score_candidate("src", "orph", store, fake_generate)
        assert out is not None
        assert out["confidence"] == pytest.approx(0.7)

    def test_llm_exception_returns_none(self, isolated_pages: Path) -> None:
        store = _FakeStore()
        store.add_page("src")
        store.add_page("orph")
        _seed_page(isolated_pages, "src")
        _seed_page(isolated_pages, "orph")

        def fake_generate(*_a, **_kw):
            raise RuntimeError("ollama down")

        assert score_candidate("src", "orph", store, fake_generate) is None

    def test_malformed_json_returns_none(self, isolated_pages: Path) -> None:
        store = _FakeStore()
        store.add_page("src")
        store.add_page("orph")
        _seed_page(isolated_pages, "src")
        _seed_page(isolated_pages, "orph")
        assert (
            score_candidate("src", "orph", store, lambda *_a, **_kw: "not json")
            is None
        )


# ---------------------------------------------------------------------------
# run_dry_run end-to-end
# ---------------------------------------------------------------------------


class TestRunDryRun:
    def test_writes_report_with_suggestions(
        self, tmp_path: Path, isolated_pages: Path
    ) -> None:
        store = _FakeStore()
        store.add_page("orph", title="Orphan A")
        store.add_page("p1", title="Source One")
        store.add_page("p2", title="Source Two")
        # p1 and p2 mutually link so neither is itself an orphan.
        store.link("p1", "p2")
        store.link("p2", "p1")
        _seed_page(isolated_pages, "orph")
        _seed_page(isolated_pages, "p1")
        _seed_page(isolated_pages, "p2")

        def fake_search(query, top_n):
            return [_ScoredPage("p1", 0.5), _ScoredPage("p2", 0.5)]

        # Per-pair canned responses keyed off whichever source is in the prompt.
        def fake_generate(prompt: str, system: str | None = None) -> str:
            confidence = 0.85 if "Source One" in prompt else 0.4  # p2 below threshold
            return json.dumps(
                {
                    "confidence": confidence,
                    "reason": "x",
                    "suggested_anchor": "",
                    "suggested_section": "関連",
                }
            )

        output = tmp_path / "report.md"
        stats = run_dry_run(
            output,
            store=store,
            generate_fn=fake_generate,
            semantic_search_fn=fake_search,
            confidence_threshold=0.5,
        )

        assert stats["orphans_total"] == 1
        assert stats["with_suggestion"] == 1
        assert stats["total_suggestions"] == 1  # p2 dropped by threshold

        text = output.read_text()
        assert "Orphan A" in text
        assert "p1" in text
        # p2 was scored under threshold and dropped — must NOT be in the
        # suggestions section. It can still appear elsewhere conceptually
        # but not as a checked-list suggestion line.
        suggestion_lines = [ln for ln in text.splitlines() if "- [ ]" in ln]
        assert all("p1" in ln for ln in suggestion_lines)

    def test_no_pages_changed(self, tmp_path: Path, isolated_pages: Path) -> None:
        """Pages on disk must be byte-identical before and after."""
        store = _FakeStore()
        store.add_page("orph", title="Orphan")
        store.add_page("p1", title="P1")
        store.add_page("hub", title="Hub")
        # Make p1 well-connected so it isn't itself an orphan.
        store.link("hub", "p1")
        store.link("p1", "hub")
        _seed_page(isolated_pages, "orph", body="ORIGINAL_ORPHAN_BODY")
        _seed_page(isolated_pages, "p1", body="ORIGINAL_P1_BODY")
        _seed_page(isolated_pages, "hub", body="HUB")

        before = {p.name: p.read_bytes() for p in isolated_pages.iterdir()}

        run_dry_run(
            tmp_path / "report.md",
            store=store,
            generate_fn=lambda *_a, **_kw: json.dumps(
                {
                    "confidence": 0.9,
                    "reason": "x",
                    "suggested_anchor": "",
                    "suggested_section": "関連",
                }
            ),
            semantic_search_fn=lambda q, n: [_ScoredPage("p1", 0.5)],
        )

        after = {p.name: p.read_bytes() for p in isolated_pages.iterdir()}
        assert before == after

    def test_orphan_limit(self, tmp_path: Path, isolated_pages: Path) -> None:
        store = _FakeStore()
        for i in range(5):
            store.add_page(f"orph{i}")
            _seed_page(isolated_pages, f"orph{i}")

        called = {"n": 0}

        def fake_generate(*_a, **_kw):
            called["n"] += 1
            return json.dumps(
                {
                    "confidence": 0.0,
                    "reason": "",
                    "suggested_anchor": "",
                    "suggested_section": "",
                }
            )

        stats = run_dry_run(
            tmp_path / "report.md",
            store=store,
            generate_fn=fake_generate,
            semantic_search_fn=lambda q, n: [],  # no candidates → no LLM calls
            orphan_limit=2,
        )
        assert stats["orphans_total"] == 2


# ---------------------------------------------------------------------------
# format_report — quick sanity
# ---------------------------------------------------------------------------


class TestFormatReport:
    def test_zero_suggestions_marker(self) -> None:
        rep = OrphanReport(
            orphan_page_id="lonely", orphan_title="Lonely", candidates_considered=0
        )
        text = format_report([rep])
        assert "lonely" in text
        assert "no suggestion above threshold" in text

    def test_suggestion_renders_as_checklist(self) -> None:
        rep = OrphanReport(
            orphan_page_id="orph",
            orphan_title="Orph",
            candidates_considered=1,
            suggestions=[
                Suggestion(
                    source_page_id="src",
                    confidence=0.91,
                    reason="共通テーマ",
                    suggested_anchor="LLM",
                    suggested_section="関連",
                )
            ],
        )
        text = format_report([rep])
        assert "- [ ]" in text
        assert "src" in text
        assert "0.91" in text
