"""Tests for orphan_link module (plan-2).

The LLM and semantic search are both mocked so these tests run fast and
deterministically. The real-LLM dry-run is exercised by the script
end-to-end at runtime, not in CI.
"""

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from chronovisor.decision.decision_authority import AUTHORITY_VERSION
from chronovisor.ops import orphan_link as ol_mod
from chronovisor.ops.orphan_link import (
    OrphanReport,
    Suggestion,
    apply_suggestion,
    format_report,
    gather_candidates,
    parse_llm_response,
    run_autonomous,
    run_dry_run,
    score_candidate,
)
from tests.semantic_hold_support import semantic_authority, semantic_review

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


def test_default_session_uses_fixed_orphan_runtime_role(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class Session:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

        def run(self, *_args: object, **_kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(
                ok=True,
                value={
                    "confidence": 0.9,
                    "reason": "related",
                    "suggested_anchor": "anchor",
                    "suggested_section": "Related",
                },
            )

    monkeypatch.setattr(ol_mod, "_build_prompt", lambda *_args: "prompt")
    monkeypatch.setattr(ol_mod, "LocalStructuredSession", Session)
    monkeypatch.setattr(
        ol_mod,
        "load_decision_router_config",
        lambda: SimpleNamespace(
            num_ctx=65_536,
            num_predict=8_192,
            primary_keep_alive="20m",
            read_timeout_ms=660_000,
            max_input_chars=64_000,
            max_output_chars=8_000,
            max_feedback_chars=2_000,
        ),
    )

    assert score_candidate("source", "orphan", object()) == {
        "confidence": 0.9,
        "reason": "related",
        "suggested_anchor": "anchor",
        "suggested_section": "Related",
    }
    assert captured["model"] is None
    assert captured["runtime_role"] == "lint.orphan_link"
    assert captured["source_data_class"] == "page"
    assert captured["source_sensitivity"] == "high"
    assert captured["num_predict"] == 1_024


def test_injected_generator_uses_no_runtime_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class Session:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

        def run(self, *_args: object, **_kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(
                ok=True,
                value={
                    "confidence": 0.9,
                    "reason": "related",
                    "suggested_anchor": "anchor",
                    "suggested_section": "Related",
                },
            )

    monkeypatch.setattr(ol_mod, "_build_prompt", lambda *_args: "prompt")
    monkeypatch.setattr(ol_mod, "LocalStructuredSession", Session)
    monkeypatch.setattr(
        ol_mod,
        "load_decision_router_config",
        lambda: pytest.fail("injected generator resolved runtime config"),
    )

    score_candidate(
        "source",
        "orphan",
        object(),
        generate_fn=lambda *_args, **_kwargs: "{}",
    )

    assert captured["model"] == "injected:orphan-link"
    assert "runtime_role" not in captured


def test_apply_suggestion_inserts_frontier_approved_link(isolated_pages: Path) -> None:
    _seed_page(isolated_pages, "source", "The durable anchor belongs here.")
    _seed_page(isolated_pages, "target", "Target body")
    suggestion = Suggestion(
        source_page_id="source",
        confidence=0.9,
        reason="related",
        suggested_anchor="durable anchor",
        suggested_section="Related",
    )

    result = apply_suggestion("target", suggestion)

    assert result["status"] == "applied"
    assert "[durable anchor](<target.md>)" in (
        isolated_pages / "source.md"
    ).read_text()


def test_apply_suggestion_renders_nested_relative_link_and_detects_fragment(
    isolated_pages: Path,
) -> None:
    _seed_page(isolated_pages, "nested/source", "The durable anchor belongs here.")
    _seed_page(isolated_pages, "topics/target", "## Details\n\nTarget body")
    suggestion = Suggestion(
        source_page_id="source",
        confidence=0.9,
        reason="related",
        suggested_anchor="durable anchor",
        suggested_section="Related",
    )

    applied = apply_suggestion("target", suggestion)

    source = isolated_pages / "nested" / "source.md"
    assert applied["status"] == "applied"
    assert "[durable anchor](<../topics/target.md>)" in source.read_text(
        encoding="utf-8"
    )

    _seed_page(
        isolated_pages,
        "nested/linked",
        "Existing [Target](<../topics/target.md#details>).",
    )
    linked = apply_suggestion(
        "target",
        Suggestion(
            source_page_id="linked",
            confidence=0.9,
            reason="related",
            suggested_anchor="Target",
            suggested_section="Related",
        ),
    )

    assert linked["status"] == "already_applied"
    assert (isolated_pages / "nested" / "linked.md").read_text(
        encoding="utf-8"
    ).count("target.md#details") == 1


def test_apply_suggestion_fails_closed_on_page_system_id_collision(
    isolated_pages: Path,
) -> None:
    _seed_page(isolated_pages, "source", "The durable anchor belongs here.")
    _seed_page(isolated_pages, "target", "Target body")
    system_dir = ol_mod.SYSTEM_DIR
    system_dir.mkdir()
    (system_dir / "target.md").write_text(
        "---\ntitle: System target\nstatus: stable\n---\nSystem body\n",
        encoding="utf-8",
    )
    before = (isolated_pages / "source.md").read_bytes()

    result = apply_suggestion(
        "target",
        Suggestion(
            source_page_id="source",
            confidence=0.9,
            reason="related",
            suggested_anchor="durable anchor",
            suggested_section="Related",
        ),
    )

    assert result == {"status": "error", "reason": "source_or_target_missing"}
    assert (isolated_pages / "source.md").read_bytes() == before


def test_apply_suggestion_preserves_correction_that_lands_before_locked_cas(
    isolated_pages: Path,
    monkeypatch,
) -> None:
    _seed_page(isolated_pages, "source", "The durable anchor belongs here.")
    _seed_page(isolated_pages, "target", "Target body")
    source = isolated_pages / "source.md"
    corrected = source.read_text(encoding="utf-8") + "user correction\n"

    @contextmanager
    def correction_wins():
        source.write_text(corrected, encoding="utf-8")
        yield

    monkeypatch.setattr(ol_mod, "chronovisor_mutation_lock", correction_wins)
    suggestion = Suggestion(
        source_page_id="source",
        confidence=0.9,
        reason="related",
        suggested_anchor="durable anchor",
        suggested_section="Related",
    )

    result = apply_suggestion("target", suggestion)

    assert result == {"status": "retry", "reason": "source_changed_before_apply"}
    assert source.read_text(encoding="utf-8") == corrected


def test_apply_suggestion_rolls_back_only_its_owned_write_under_lock(
    isolated_pages: Path,
    monkeypatch,
) -> None:
    _seed_page(isolated_pages, "source", "The durable anchor belongs here.")
    _seed_page(isolated_pages, "target", "Target body")
    source = isolated_pages / "source.md"
    target = isolated_pages / "target.md"
    original = source.read_text(encoding="utf-8")
    real_atomic_write = ol_mod.atomic_write
    writes = 0

    def change_target_after_source_write(path: Path, content: str) -> None:
        nonlocal writes
        real_atomic_write(path, content)
        writes += 1
        if writes == 1:
            target.write_text(
                target.read_text(encoding="utf-8") + "foreign target edit\n"
            )

    monkeypatch.setattr(ol_mod, "atomic_write", change_target_after_source_write)
    suggestion = Suggestion(
        source_page_id="source",
        confidence=0.9,
        reason="related",
        suggested_anchor="durable anchor",
        suggested_section="Related",
    )

    result = apply_suggestion("target", suggestion)

    assert result == {"status": "error", "reason": "post_write_verification_failed"}
    assert source.read_text(encoding="utf-8") == original


def test_autonomous_orphan_lane_applies_once(
    tmp_path: Path, isolated_pages: Path
) -> None:
    from chronovisor.ingest.convergence import ConvergenceStore, RetryPolicy

    _seed_page(isolated_pages, "source", "durable anchor")
    _seed_page(isolated_pages, "target", "target topic")
    store = _FakeStore()
    store.add_page("source")
    store.add_page("target")
    store.link("other", "source")
    state = ConvergenceStore(
        tmp_path / "state.json",
        policy=RetryPolicy(local_base_delay_seconds=0, frontier_base_delay_seconds=0),
    )

    def semantic(_query, _top_n):
        return [_ScoredPage("source", 0.9)]

    generated = json.dumps(
        {
            "confidence": 0.92,
            "reason": "related",
            "suggested_anchor": "durable anchor",
            "suggested_section": "Related",
        }
    )

    def reviewer(_candidate):
        return {"decision": "approved", "confidence": 0.95, "summary": "ok"}

    first = run_autonomous(
        orphan_limit=1,
        store=store,
        generate_fn=lambda *_args, **_kwargs: generated,
        semantic_search_fn=semantic,
        reviewer=reviewer,
        convergence_store=state,
    )
    second = run_autonomous(
        orphan_limit=1,
        store=store,
        generate_fn=lambda *_args, **_kwargs: generated,
        semantic_search_fn=semantic,
        reviewer=reviewer,
        convergence_store=state,
    )

    assert first["results"][0]["status"] == "applied"
    assert second["results"][0]["status"] == "applied"
    assert (isolated_pages / "source.md").read_text().count("(<target.md>)") == 1


def test_orphan_no_quorum_is_cached_until_authority_epoch_changes(
    tmp_path: Path,
    isolated_pages: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chronovisor.ingest.convergence import ConvergenceStore, RetryPolicy

    _seed_page(isolated_pages, "source", "durable anchor")
    _seed_page(isolated_pages, "target", "target topic")
    store = _FakeStore()
    store.add_page("source")
    store.add_page("target")
    store.link("other", "source")
    state = ConvergenceStore(
        tmp_path / "state.json",
        policy=RetryPolicy(local_base_delay_seconds=0, frontier_base_delay_seconds=0),
    )
    authority_a = semantic_authority(
        ol_mod.DECISION_LANE,
        schema_name="orphan_link",
    )
    authority_b = semantic_authority(
        ol_mod.DECISION_LANE,
        schema_name="orphan_link",
        artifact_sha256="9" * 64,
    )
    current = [authority_a]
    monkeypatch.setattr(
        ol_mod,
        "current_semantic_authority",
        lambda *_args, **_kwargs: (current[0], None),
    )
    generated = json.dumps(
        {
            "confidence": 0.92,
            "reason": "related",
            "suggested_anchor": "durable anchor",
            "suggested_section": "Related",
        }
    )
    calls = 0

    def reviewer(_candidate: dict) -> dict:
        nonlocal calls
        calls += 1
        return {
            **semantic_review(current[0], lane=ol_mod.DECISION_LANE),
            "confidence": 0.0,
        }

    kwargs = {
        "orphan_limit": 1,
        "store": store,
        "generate_fn": lambda *_args, **_kwargs: generated,
        "semantic_search_fn": lambda _query, _top_n: [_ScoredPage("source", 0.9)],
        "reviewer": reviewer,
        "convergence_store": state,
    }
    first = run_autonomous(**kwargs)
    same_epoch = run_autonomous(**kwargs)
    current[0] = authority_b
    changed_authority = run_autonomous(**kwargs)

    assert first["results"][0]["status"] == "quarantined"
    assert same_epoch["results"][0]["status"] == "quarantined"
    assert same_epoch["results"][0]["cached"] is True
    assert changed_authority["results"][0]["status"] == "quarantined"
    assert calls == 2
    held_authorities = {
        item["result"]["semantic_hold"]["authority"]["router"]["routes"][0][
            "ollama"
        ]["digest"]
        for item in state.list_items(lane=ol_mod.DECISION_LANE)
        if isinstance(item.get("result"), dict)
        and isinstance(item["result"].get("semantic_hold"), dict)
    }
    assert held_authorities == {"a" * 64, "9" * 64}
    assert "(<target.md>)" not in (isolated_pages / "source.md").read_text()


@pytest.fixture()
def isolated_pages(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point canonical orphan-link resolution at an isolated Wiki."""
    pages_dir = tmp_path / "pages"
    pages_dir.mkdir()

    from chronovisor.core import page_mutation

    for name in ("index.md", "log.md", "schema.md"):
        (tmp_path / name).write_text("legacy\n", encoding="utf-8")
    monkeypatch.setattr(
        page_mutation,
        "DECISION_AUTHORITY_LOCK",
        tmp_path / "runtime" / "decision-authority.lock",
    )
    monkeypatch.setattr(page_mutation, "CHRONOVISOR_ROOT", tmp_path)
    monkeypatch.setattr(page_mutation, "PAGES_DIR", pages_dir)
    monkeypatch.setattr(page_mutation, "SYSTEM_DIR", tmp_path / "system")
    monkeypatch.setattr(ol_mod, "PAGES_DIR", pages_dir)
    monkeypatch.setattr(ol_mod, "SYSTEM_DIR", tmp_path / "system")
    return pages_dir


def _seed_page(pages_dir: Path, page_id: str, body: str = "body content") -> None:
    path = pages_dir / f"{page_id}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"---\ntitle: {page_id}\nupdated: 2026-05-08\n"
        f"status: stable\ntype: knowledge\n---\n{body}\n"
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
            score_candidate("src", "orph", store, lambda *_a, **_kw: "not json") is None
        )

    def test_schema_error_is_repaired_in_same_structured_session(
        self,
        isolated_pages: Path,
    ) -> None:
        store = _FakeStore()
        store.add_page("src")
        store.add_page("orph")
        _seed_page(isolated_pages, "src")
        _seed_page(isolated_pages, "orph")
        replies = iter(
            [
                json.dumps({"confidence": 0.8, "reason": "missing fields"}),
                json.dumps(
                    {
                        "confidence": 0.8,
                        "reason": "関連あり",
                        "suggested_anchor": "anchor",
                        "suggested_section": "関連",
                    }
                ),
            ]
        )
        prompts: list[str] = []

        def fake_generate(prompt: str, **_kwargs) -> str:
            prompts.append(prompt)
            return next(replies)

        result = score_candidate("src", "orph", store, fake_generate)

        assert result is not None
        assert result["confidence"] == pytest.approx(0.8)
        assert len(prompts) == 2
        assert "Validator errors" in prompts[1]


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


def test_apply_suggestion_never_nests_inside_existing_markdown_link(
    isolated_pages: Path,
) -> None:
    _seed_page(
        isolated_pages,
        "source",
        "Existing [durable anchor](<other.md>) and [Beta](<beta.md>).",
    )
    _seed_page(isolated_pages, "other", "Other body")
    _seed_page(isolated_pages, "beta", "Beta body")
    _seed_page(isolated_pages, "target", "Target body")
    suggestion = Suggestion(
        source_page_id="source",
        confidence=0.9,
        reason="related",
        suggested_anchor="durable anchor",
        suggested_section="Related\n\n## injected",
    )

    result = apply_suggestion("target", suggestion)
    text = (isolated_pages / "source.md").read_text(encoding="utf-8")

    assert result["status"] == "applied"
    assert "[durable anchor](<other.md>)" in text
    assert "[Beta](<beta.md>)" in text
    assert "## Related\n\n- [durable anchor](<target.md>)" in text
    assert "injected" not in text


def test_apply_suggestion_skips_anchor_inside_markdown_link_target(
    isolated_pages: Path,
) -> None:
    _seed_page(isolated_pages, "source", "Existing [Beta](<beta.md>).")
    _seed_page(isolated_pages, "beta", "Beta body")
    _seed_page(isolated_pages, "target", "Target body")

    result = apply_suggestion(
        "target",
        Suggestion(
            source_page_id="source",
            confidence=0.9,
            reason="related",
            suggested_anchor="beta.md",
            suggested_section="Related",
        ),
    )
    text = (isolated_pages / "source.md").read_text(encoding="utf-8")

    assert result["status"] == "applied"
    assert "Existing [Beta](<beta.md>)." in text
    assert "## Related\n\n- [beta.md](<target.md>)" in text


def _autonomous_state(tmp_path: Path, *, max_local_attempts: int = 2):
    from chronovisor.ingest.convergence import ConvergenceStore, RetryPolicy

    return ConvergenceStore(
        tmp_path / "state.json",
        policy=RetryPolicy(
            max_local_attempts=max_local_attempts,
            local_base_delay_seconds=0,
            frontier_base_delay_seconds=0,
        ),
    )


def _autonomous_fixture(isolated_pages: Path) -> tuple[_FakeStore, object]:
    store = _FakeStore()
    store.add_page("source")
    store.add_page("target")
    store.link("other", "source")
    _seed_page(isolated_pages, "source", "durable anchor")
    _seed_page(isolated_pages, "target", "target topic")

    def semantic(_query, _top_n):
        return [_ScoredPage("source", 0.9)]

    return store, semantic


def _high_local_suggestion() -> str:
    return json.dumps(
        {
            "confidence": 0.92,
            "reason": "related",
            "suggested_anchor": "durable anchor",
            "suggested_section": "Related",
        }
    )


def test_autonomous_drain_prioritizes_oldest_pending_orphan(
    tmp_path: Path,
    isolated_pages: Path,
) -> None:
    store = _FakeStore()
    store.add_page("target-one")
    store.add_page("target-two")
    store.add_page("source")
    store.add_page("other")
    store.link("other", "source")
    store.link("source", "other")
    for page_id in ("target-one", "target-two", "source", "other"):
        _seed_page(isolated_pages, page_id, f"{page_id} body")
    state = _autonomous_state(tmp_path)
    authority, error = ol_mod.current_semantic_authority(
        ol_mod.DECISION_LANE,
        injected_reviewer=True,
    )
    assert error is None
    state.merge_item(
        lane=ol_mod.DECISION_LANE,
        source_id="orphan:target-two",
        input_data={
            "orphan": "target-two",
            "orphan_hash": ol_mod._content_hash("target-two"),
            "decision_authority": authority,
            "candidates": [
                {
                    "source": "source",
                    "source_hash": ol_mod._content_hash("source"),
                }
            ],
        },
        resolver_version=ol_mod.RESOLVER_VERSION,
        metadata={
            "orphan": "target-two",
            "source": "source",
            "candidate_discovery_error": None,
        },
    )

    result = run_autonomous(
        orphan_limit=1,
        store=store,
        generate_fn=lambda *_args, **_kwargs: _high_local_suggestion(),
        semantic_search_fn=lambda _query, _top_n: [_ScoredPage("source", 0.9)],
        reviewer=lambda _candidate: {
            "decision": "rejected",
            "confidence": 0.95,
            "summary": "keep separate",
        },
        convergence_store=state,
    )

    assert result["results"][0]["orphan"] == "target-two"
    assert state.list_items(lane=ol_mod.DECISION_LANE)[0]["status"] == "rejected"


def test_frontier_approval_is_not_overridden_by_confidence_metadata(
    tmp_path: Path,
    isolated_pages: Path,
) -> None:
    store, semantic = _autonomous_fixture(isolated_pages)
    state = _autonomous_state(tmp_path)

    result = run_autonomous(
        orphan_limit=1,
        store=store,
        generate_fn=lambda *_args, **_kwargs: _high_local_suggestion(),
        semantic_search_fn=semantic,
        reviewer=lambda _candidate: {
            "decision": "approved",
            "confidence": 0.2,
            "summary": "weak relation",
        },
        convergence_store=state,
    )

    assert result["results"][0]["status"] == "applied"
    assert "(<target.md>)" in (
        isolated_pages / "source.md"
    ).read_text(encoding="utf-8")
    durable = state.list_items()[0]["result"]
    assert durable["schema_version"] == 2
    assert durable["authority"] == {
        "source": "injected_reviewer_boundary",
        "authority_version": AUTHORITY_VERSION,
        "lane": "orphan_link",
    }


def test_orphan_effect_fails_closed_when_authority_changes_before_apply(
    tmp_path: Path,
    isolated_pages: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, semantic = _autonomous_fixture(isolated_pages)
    state = _autonomous_state(tmp_path)
    authority = {
        "source": "injected_reviewer_boundary",
        "authority_version": AUTHORITY_VERSION,
        "lane": "orphan_link",
    }
    calls = 0

    def authority_then_disable(_lane: str, *, injected_reviewer: bool = False):
        nonlocal calls
        assert injected_reviewer is True
        calls += 1
        if calls >= 3:
            return None, "decision_lane_not_enabled:orphan_link:shadow"
        return authority, None

    monkeypatch.setattr(ol_mod, "current_semantic_authority", authority_then_disable)
    result = run_autonomous(
        orphan_limit=1,
        store=store,
        generate_fn=lambda *_args, **_kwargs: _high_local_suggestion(),
        semantic_search_fn=semantic,
        reviewer=lambda _candidate: {
            "decision": "approved",
            "confidence": 0.95,
            "summary": "good",
        },
        convergence_store=state,
    )

    assert result["results"][0]["status"] == "frontier_retry"
    assert "(<target.md>)" not in (
        isolated_pages / "source.md"
    ).read_text(encoding="utf-8")


def test_orphan_exact_postimage_recovers_before_absent_source_retirement(
    tmp_path: Path,
    isolated_pages: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, semantic = _autonomous_fixture(isolated_pages)
    state = _autonomous_state(tmp_path)
    original_complete = state.complete

    def crash_before_complete(key: str, status: str, **kwargs):
        result = kwargs.get("result") or {}
        if status == "applied" and result.get("semantic_effect") is True:
            raise KeyboardInterrupt()
        return original_complete(key, status, **kwargs)

    monkeypatch.setattr(state, "complete", crash_before_complete)
    with pytest.raises(KeyboardInterrupt):
        run_autonomous(
            orphan_limit=1,
            store=store,
            generate_fn=lambda *_args, **_kwargs: _high_local_suggestion(),
            semantic_search_fn=semantic,
            reviewer=lambda _candidate: {
                "decision": "approved",
                "confidence": 0.95,
                "summary": "good",
            },
            convergence_store=state,
        )

    pending = state.list_items()[0]
    metadata = pending["metadata"]
    artifact_path = Path(metadata["effect_artifact_path"])
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    source = (isolated_pages / "source.md").read_bytes()
    assert artifact_path.exists()
    assert artifact["schema_version"] == 2
    assert artifact["convergence_key"] == pending["key"]
    assert artifact["source_postimage_sha256"] == hashlib.sha256(source).hexdigest()
    assert (isolated_pages / "source.md").read_text().count("(<target.md>)") == 1

    # A refreshed index no longer reports the target as orphaned. Recovery
    # must therefore happen before retire_absent_sources can reject the item.
    store.link("source", "target")
    monkeypatch.setattr(state, "complete", original_complete)
    recovered = run_autonomous(
        orphan_limit=1,
        store=store,
        generate_fn=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("recovery must not rerun local semantics")
        ),
        semantic_search_fn=semantic,
        reviewer=lambda _candidate: (_ for _ in ()).throw(
            AssertionError("recovery must not rerun semantic review")
        ),
        convergence_store=state,
    )

    assert recovered["results"] == [
        {
            "key": pending["key"],
            "orphan": "target",
            "status": "applied",
            "recovery_only": True,
            "semantic_effect": False,
        }
    ]
    durable = state.get(pending["key"])
    assert durable["result"]["recovery_only"] is True
    assert durable["result"]["semantic_effect"] is False
    assert (isolated_pages / "source.md").read_text().count("(<target.md>)") == 1


def test_orphan_recovery_never_reapplies_when_postimage_is_not_exact(
    tmp_path: Path,
    isolated_pages: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, semantic = _autonomous_fixture(isolated_pages)
    state = _autonomous_state(tmp_path)
    original_complete = state.complete

    def crash_before_complete(key: str, status: str, **kwargs):
        result = kwargs.get("result") or {}
        if status == "applied" and result.get("semantic_effect") is True:
            raise KeyboardInterrupt()
        return original_complete(key, status, **kwargs)

    monkeypatch.setattr(state, "complete", crash_before_complete)
    with pytest.raises(KeyboardInterrupt):
        run_autonomous(
            orphan_limit=1,
            store=store,
            generate_fn=lambda *_args, **_kwargs: _high_local_suggestion(),
            semantic_search_fn=semantic,
            reviewer=lambda _candidate: {
                "decision": "approved",
                "confidence": 0.95,
                "summary": "good",
            },
            convergence_store=state,
        )
    pending = state.list_items()[0]
    source_path = isolated_pages / "source.md"
    source_path.write_text(
        source_path.read_text(encoding="utf-8") + "foreign edit\n",
        encoding="utf-8",
    )
    store.link("source", "target")
    monkeypatch.setattr(state, "complete", original_complete)

    result = run_autonomous(
        orphan_limit=1,
        store=store,
        generate_fn=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("non-exact recovery must not rerun local semantics")
        ),
        semantic_search_fn=semantic,
        reviewer=lambda _candidate: (_ for _ in ()).throw(
            AssertionError("non-exact recovery must not rerun semantic review")
        ),
        convergence_store=state,
    )

    assert pending["key"] in result["retired"]
    assert state.get(pending["key"])["status"] == "rejected"
    assert source_path.read_text(encoding="utf-8").count("(<target.md>)") == 1
    assert source_path.read_text(encoding="utf-8").endswith("foreign edit\n")


def test_terminal_orphan_verdict_is_not_reused_when_lane_becomes_unavailable(
    tmp_path: Path,
    isolated_pages: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, semantic = _autonomous_fixture(isolated_pages)
    state = _autonomous_state(tmp_path)
    kwargs = {
        "orphan_limit": 1,
        "store": store,
        "generate_fn": lambda *_args, **_kwargs: _high_local_suggestion(),
        "semantic_search_fn": semantic,
        "reviewer": lambda _candidate: {
            "decision": "approved",
            "confidence": 0.95,
            "summary": "good",
        },
        "convergence_store": state,
    }
    first = run_autonomous(**kwargs)
    monkeypatch.setattr(
        ol_mod,
        "current_semantic_authority",
        lambda *_args, **_kwargs: (
            None,
            "decision_lane_not_enabled:orphan_link:shadow",
        ),
    )

    second = run_autonomous(**kwargs)

    assert first["results"][0]["status"] == "applied"
    assert second["results"][0]["status"] == "decision_authority_unavailable"
    assert second["results"][0].get("cached") is not True


def test_malformed_frontier_approval_retries_without_mutation(
    tmp_path: Path,
    isolated_pages: Path,
) -> None:
    store, semantic = _autonomous_fixture(isolated_pages)
    state = _autonomous_state(tmp_path)

    result = run_autonomous(
        orphan_limit=1,
        store=store,
        generate_fn=lambda *_args, **_kwargs: _high_local_suggestion(),
        semantic_search_fn=semantic,
        reviewer=lambda _candidate: {
            "decision": "approved",
            "summary": "missing confidence",
        },
        convergence_store=state,
    )

    assert result["results"][0]["status"] == "frontier_retry"
    assert state.list_items()[0]["status"] == "frontier_retry"
    assert "(<target.md>)" not in (
        isolated_pages / "source.md"
    ).read_text(encoding="utf-8")


def test_local_model_error_retries_but_valid_low_score_rejects(
    tmp_path: Path,
    isolated_pages: Path,
) -> None:
    store, semantic = _autonomous_fixture(isolated_pages)
    retry_state = _autonomous_state(tmp_path / "retry")

    retry = run_autonomous(
        orphan_limit=1,
        store=store,
        generate_fn=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("offline")
        ),
        semantic_search_fn=semantic,
        reviewer=lambda _candidate: {
            "decision": "approved",
            "confidence": 0.95,
            "summary": "retry is warranted",
        },
        convergence_store=retry_state,
    )

    low_state = _autonomous_state(tmp_path / "low")
    low = run_autonomous(
        orphan_limit=1,
        store=store,
        generate_fn=lambda *_args, **_kwargs: json.dumps(
            {
                "confidence": 0.1,
                "reason": "weak",
                "suggested_anchor": "",
                "suggested_section": "Related",
            }
        ),
        semantic_search_fn=semantic,
        reviewer=lambda _candidate: {
            "decision": "approved",
            "confidence": 0.95,
            "summary": "no safe link",
        },
        convergence_store=low_state,
    )

    assert retry["results"][0]["status"] == "frontier_retry"
    assert (
        retry_state.list_items()[0]["last_failure_class"]
        == "local_model_or_schema_error"
    )
    assert low["results"][0]["status"] == "rejected"
    assert low_state.list_items()[0]["result"]["proposal"]["reason"].startswith(
        "all_local_candidates"
    )


def test_no_candidate_does_not_starve_later_real_work(
    tmp_path: Path,
    isolated_pages: Path,
) -> None:
    store = _FakeStore()
    for page_id in ("a-empty", "b-target", "source"):
        store.add_page(page_id)
        _seed_page(isolated_pages, page_id, "durable anchor")
    store.link("other", "source")

    def semantic(query: str, _top_n: int):
        return [] if "a-empty" in query else [_ScoredPage("source", 0.9)]

    state = _autonomous_state(tmp_path)
    result = run_autonomous(
        orphan_limit=1,
        store=store,
        generate_fn=lambda *_args, **_kwargs: _high_local_suggestion(),
        semantic_search_fn=semantic,
        reviewer=lambda _candidate: {
            "decision": "approved",
            "confidence": 0.95,
            "summary": "good",
        },
        convergence_store=state,
    )

    assert [entry["status"] for entry in result["results"]] == ["rejected", "applied"]
    assert result["work_items"] == 1
    assert "(<b-target.md>)" in (
        isolated_pages / "source.md"
    ).read_text(encoding="utf-8")


def test_frontier_retry_keeps_durable_local_suggestion(
    tmp_path: Path,
    isolated_pages: Path,
) -> None:
    from chronovisor.ingest.convergence import CycleBudget

    store, semantic = _autonomous_fixture(isolated_pages)
    state = _autonomous_state(tmp_path)
    first_budget = CycleBudget(
        max_local_calls=1,
        max_frontier_calls=0,
        max_mutations=1,
        max_elapsed_seconds=60,
    )

    first = run_autonomous(
        orphan_limit=1,
        store=store,
        generate_fn=lambda *_args, **_kwargs: _high_local_suggestion(),
        semantic_search_fn=semantic,
        reviewer=lambda _candidate: (_ for _ in ()).throw(
            AssertionError("frontier should be deferred")
        ),
        convergence_store=state,
        budget=first_budget,
    )
    second = run_autonomous(
        orphan_limit=1,
        store=store,
        generate_fn=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("local suggestion should be reused")
        ),
        semantic_search_fn=semantic,
        reviewer=lambda _candidate: {
            "decision": "approved",
            "confidence": 0.95,
            "summary": "good",
        },
        convergence_store=state,
    )

    assert first["results"][0]["status"] == "frontier_budget_exhausted"
    assert second["results"][0]["status"] == "applied"
    assert "(<target.md>)" in (
        isolated_pages / "source.md"
    ).read_text(encoding="utf-8")


def test_elapsed_budget_stops_before_candidate_discovery(
    tmp_path: Path,
    isolated_pages: Path,
) -> None:
    from chronovisor.ingest.convergence import CycleBudget

    store = _FakeStore()
    store.add_page("orphan", body="Target")
    state = _autonomous_state(tmp_path)
    ticks = [0.0, 61.0]

    def clock() -> float:
        return ticks.pop(0) if len(ticks) > 1 else ticks[0]

    budget = CycleBudget(max_elapsed_seconds=60, clock=clock)

    result = run_autonomous(
        orphan_limit=1,
        store=store,
        semantic_search_fn=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("candidate discovery must not start after the deadline")
        ),
        convergence_store=state,
        budget=budget,
    )

    assert result["stop_reason"] == "elapsed_budget_exhausted"
    assert result["orphans_seen"] == 0
    assert result["work_items"] == 0


def test_exhausted_local_lane_stops_before_active_candidate_discovery(
    tmp_path: Path,
) -> None:
    from chronovisor.ingest.convergence import CycleBudget

    store = _FakeStore()
    store.add_page("orphan", body="Target")
    state = _autonomous_state(tmp_path)
    state.merge_item(
        lane="orphan_link",
        source_id="orphan:orphan",
        input_data={"orphan": "orphan"},
        resolver_version=ol_mod.RESOLVER_VERSION,
    )
    budget = CycleBudget(
        max_local_calls=0,
        max_frontier_calls=1,
        max_mutations=1,
        max_elapsed_seconds=60,
    )

    result = run_autonomous(
        orphan_limit=1,
        store=store,
        semantic_search_fn=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("candidate discovery must not start without local budget")
        ),
        convergence_store=state,
        budget=budget,
    )

    assert result["stop_reason"] == "local_lane_budget_exhausted"
    assert result["orphans_seen"] == 0


def test_no_candidate_terminal_decision_is_durable(
    tmp_path: Path,
    isolated_pages: Path,
) -> None:
    store = _FakeStore()
    store.add_page("target")
    _seed_page(isolated_pages, "target", "target topic")
    state = _autonomous_state(tmp_path)

    first = run_autonomous(
        orphan_limit=1,
        store=store,
        semantic_search_fn=lambda _query, _top_n: [],
        generate_fn=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("no candidate means no local model call")
        ),
        reviewer=lambda _candidate: {
            "decision": "approved",
            "confidence": 0.95,
            "summary": "no semantic candidate",
        },
        convergence_store=state,
    )
    second = run_autonomous(
        orphan_limit=1,
        store=store,
        semantic_search_fn=lambda _query, _top_n: [],
        generate_fn=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("terminal decision must be reused")
        ),
        reviewer=lambda _candidate: (_ for _ in ()).throw(
            AssertionError("terminal frontier decision must be reused")
        ),
        convergence_store=state,
    )

    assert first["results"][0]["status"] == "rejected"
    assert first["work_items"] == 0
    assert second["results"][0]["status"] == "rejected"
    assert second["results"][0]["cached"] is True
    assert len(state.list_items()) == 1


def test_production_candidate_discovery_uses_strict_semantic_health_boundary(
    isolated_pages: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chronovisor.core import search

    store = _FakeStore()
    store.add_page("target")
    _seed_page(isolated_pages, "target", "target topic")
    calls: list[tuple[str, int, bool]] = []

    def semantic(query: str, top_n: int, *, strict: bool = False):
        calls.append((query, top_n, strict))
        return []

    monkeypatch.setattr(search, "semantic_search", semantic)

    assert gather_candidates("target", store) == []
    assert calls and calls[0][2] is True


def test_candidate_discovery_error_retries_then_quarantines(
    tmp_path: Path,
    isolated_pages: Path,
) -> None:
    store = _FakeStore()
    store.add_page("target")
    _seed_page(isolated_pages, "target", "target topic")
    state = _autonomous_state(tmp_path, max_local_attempts=2)

    def unavailable(_query: str, _top_n: int):
        raise RuntimeError("embedding service unavailable")

    first = run_autonomous(
        orphan_limit=1,
        store=store,
        semantic_search_fn=unavailable,
        reviewer=lambda _candidate: {
            "decision": "approved",
            "confidence": 0.95,
            "summary": "candidate discovery should retry",
        },
        convergence_store=state,
    )
    second = run_autonomous(
        orphan_limit=1,
        store=store,
        semantic_search_fn=unavailable,
        reviewer=lambda _candidate: {
            "decision": "approved",
            "confidence": 0.95,
            "summary": "candidate discovery should retry",
        },
        convergence_store=state,
    )
    third = run_autonomous(
        orphan_limit=1,
        store=store,
        semantic_search_fn=unavailable,
        reviewer=lambda _candidate: {
            "decision": "approved",
            "confidence": 0.95,
            "summary": "candidate discovery should retry",
        },
        convergence_store=state,
    )

    assert first["results"][0]["status"] == "frontier_retry"
    assert second["results"][0]["status"] == "frontier_retry"
    assert third["results"][0]["status"] == "quarantined"
    assert first["work_items"] == second["work_items"] == third["work_items"] == 1
    assert state.list_items()[0]["last_failure_class"] == "candidate_discovery_error"
