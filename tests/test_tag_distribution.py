"""Tests for tag_distribution module (plan-3).

Sampling determinism, LLM JSON contract, aggregation arithmetic, and
report formatting. Real LLM calls are never made — every test injects
``generate_fn`` mocks.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from chronovisor.librarian import tag_distribution as td
from chronovisor.librarian.tag_distribution import (
    PageAnalysis,
    SamplingPlan,
    aggregate,
    analyze_page,
    format_report,
    minority_sample,
    parse_llm_response,
    proportional_sample,
    run_dry_run,
)

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeStore:
    def __init__(self, pages: dict[str, str]) -> None:
        # pages = {page_id: folder}
        self._pages: dict[str, dict] = {
            pid: {
                "page_id": pid,
                "title": pid,
                "is_system": False,
                "updated": "2026-05-08",
                "path": f"/fake/pages/{folder}/{pid}.md" if folder else f"/fake/pages/{pid}.md",
            }
            for pid, folder in pages.items()
        }

    def all_page_ids(self, include_system: bool = False) -> set[str]:
        return set(self._pages.keys())

    def meta(self, page_id: str):
        return self._pages.get(page_id)

    def refresh(self) -> None:
        pass


@pytest.fixture()
def isolated_pages(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    pages_dir = tmp_path / "pages"
    pages_dir.mkdir()

    # Patch store.PAGES_DIR so the folder-extraction helper resolves paths.
    from chronovisor.core import store as wiki_mod
    monkeypatch.setattr(wiki_mod, "PAGES_DIR", pages_dir)

    class PromptStore:
        def refresh(self) -> None:
            return None

        def meta(self, page_id: str):
            matches = list(pages_dir.rglob(f"{page_id}.md"))
            if len(matches) != 1:
                return None
            path = matches[0]
            return {
                "status": "stable",
                "path": str(path),
                "relative_path": path.relative_to(pages_dir).as_posix(),
                "is_system": False,
            }

    monkeypatch.setattr(td, "get_store", PromptStore)
    monkeypatch.setattr(td, "PAGES_DIR", pages_dir)
    monkeypatch.setattr(td, "SYSTEM_DIR", tmp_path / "system")
    return pages_dir


def _seed(pages_dir: Path, page_id: str, *, folder: str = "", body: str = "body") -> Path:
    target = pages_dir / folder / f"{page_id}.md" if folder else pages_dir / f"{page_id}.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        f"---\ntitle: {page_id}\nupdated: 2026-05-08\n"
        f"status: stable\ntype: knowledge\n---\n{body}\n"
    )
    return target


class _FakeStoreWithPaths:
    """Real-disk-backed FakeStore that mirrors paths actually written by ``_seed``."""

    def __init__(self, pages_dir: Path) -> None:
        self._pages_dir = pages_dir
        self._pages: dict[str, dict] = {}

    def add(self, page_id: str, folder: str = "") -> None:
        path = (
            self._pages_dir / folder / f"{page_id}.md"
            if folder
            else self._pages_dir / f"{page_id}.md"
        )
        self._pages[page_id] = {
            "page_id": page_id,
            "title": page_id,
            "is_system": False,
            "updated": "2026-05-08",
            "path": str(path),
            "status": "stable",
            "relative_path": path.relative_to(self._pages_dir).as_posix(),
        }

    def all_page_ids(self, include_system: bool = False) -> set[str]:
        return set(self._pages.keys())

    def meta(self, page_id: str):
        return self._pages.get(page_id)

    def refresh(self) -> None:
        pass


def test_default_session_uses_fixed_librarian_runtime_role(
    isolated_pages: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seed(isolated_pages, "page")
    captured: dict[str, object] = {}

    class Session:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

        def run(self, *_args: object, **_kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(
                ok=True,
                value={
                    "main_topic": "topic",
                    "assigned_tags": [],
                    "tag_evidence": {},
                    "rejected_assigned_tags": [],
                    "suggested_missing_categories": [],
                    "confidence": 0.5,
                },
            )

    monkeypatch.setattr(td, "LocalStructuredSession", Session)
    monkeypatch.setattr(
        td,
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

    assert analyze_page("page", []).main_topic == "topic"
    assert captured["model"] is None
    assert captured["runtime_role"] == "librarian.review"
    assert captured["source_data_class"] == "page"
    assert captured["source_sensitivity"] == "high"
    assert captured["num_predict"] == 1_536


def test_injected_generator_uses_no_runtime_config(
    isolated_pages: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seed(isolated_pages, "page")
    captured: dict[str, object] = {}

    class Session:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

        def run(self, *_args: object, **_kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(
                ok=True,
                value={
                    "main_topic": "topic",
                    "assigned_tags": [],
                    "tag_evidence": {},
                    "rejected_assigned_tags": [],
                    "suggested_missing_categories": [],
                    "confidence": 0.5,
                },
            )

    monkeypatch.setattr(td, "LocalStructuredSession", Session)
    monkeypatch.setattr(
        td,
        "load_decision_router_config",
        lambda: pytest.fail("injected generator resolved runtime config"),
    )

    analyze_page("page", [], generate_fn=lambda *_args, **_kwargs: "{}")

    assert captured["model"] == "injected:tag-distribution"
    assert "runtime_role" not in captured


# ---------------------------------------------------------------------------
# parse_llm_response
# ---------------------------------------------------------------------------


class TestParseLLMResponse:
    def _master(self) -> set[str]:
        return {"d/ai-industry", "t/analysis", "s/2026"}

    def _valid_payload(self, **overrides) -> dict:
        base = {
            "main_topic": "topic",
            "assigned_tags": [],
            "tag_evidence": {},
            "rejected_assigned_tags": [],
            "suggested_missing_categories": [],
            "confidence": 0.5,
        }
        base.update(overrides)
        return base

    def test_valid_response(self) -> None:
        raw = json.dumps(
            self._valid_payload(
                main_topic="AI industry analysis note",
                assigned_tags=["d/ai-industry", "t/analysis", "s/2026"],
                tag_evidence={
                    "d/ai-industry": "AI industry brief",
                    "t/analysis": "deep dive",
                    "s/2026": "2026 outlook",
                },
                confidence=0.9,
            )
        )
        out = parse_llm_response(raw, self._master())
        assert out is not None
        assert out["assigned_tags"] == ["d/ai-industry", "t/analysis", "s/2026"]
        assert out["main_topic"] == "AI industry analysis note"
        assert out["tag_evidence"]["d/ai-industry"] == "AI industry brief"

    def test_leaked_tag_moved_to_rejected(self) -> None:
        raw = json.dumps(
            self._valid_payload(
                assigned_tags=["d/ai-industry", "d/automotive"],
                tag_evidence={
                    "d/ai-industry": "industry note",
                    "d/automotive": "car spec",
                },
                confidence=0.6,
            )
        )
        out = parse_llm_response(raw, self._master())
        assert out is not None
        assert "d/automotive" not in out["assigned_tags"]
        assert "d/automotive" in out["rejected_assigned_tags"]

    def test_missing_evidence_drops_tag(self) -> None:
        # Tag in assigned_tags but no key in tag_evidence → reclassified.
        raw = json.dumps(
            self._valid_payload(
                assigned_tags=["d/ai-industry", "t/analysis"],
                tag_evidence={"d/ai-industry": "industry brief"},
            )
        )
        out = parse_llm_response(raw, self._master())
        assert out is not None
        assert "t/analysis" not in out["assigned_tags"]
        assert "t/analysis" in out["rejected_assigned_tags"]

    def test_evidence_too_long_rejects_response(self) -> None:
        raw = json.dumps(
            self._valid_payload(
                assigned_tags=["d/ai-industry"],
                tag_evidence={
                    "d/ai-industry": "this evidence is much too long for the limit"
                },
            )
        )
        assert parse_llm_response(raw, self._master()) is None

    def test_main_topic_too_long_rejected(self) -> None:
        raw = json.dumps(self._valid_payload(main_topic="x" * 51))
        assert parse_llm_response(raw, self._master()) is None

    def test_main_topic_with_newline_rejected(self) -> None:
        raw = json.dumps(self._valid_payload(main_topic="line one\nline two"))
        assert parse_llm_response(raw, self._master()) is None

    def test_smc_with_extra_field_rejected(self) -> None:
        raw = json.dumps(
            self._valid_payload(
                suggested_missing_categories=[
                    {
                        "label": "x",
                        "justification": "y",
                        "fallback_axis": "d/",
                        "stowaway": "z",
                    }
                ]
            )
        )
        assert parse_llm_response(raw, self._master()) is None

    def test_extra_top_level_field_rejected(self) -> None:
        payload = self._valid_payload()
        payload["page_id"] = "should-not-leak"
        raw = json.dumps(payload)
        assert parse_llm_response(raw, self._master()) is None

    def test_bad_confidence_rejected(self) -> None:
        for bad in (-0.1, 1.5, "0.5", None):
            raw = json.dumps(self._valid_payload(confidence=bad))
            assert parse_llm_response(raw, self._master()) is None


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


class TestProportionalSample:
    def test_seed_is_reproducible(self, isolated_pages: Path) -> None:
        # 100 pages across 3 folders.
        store = _FakeStoreWithPaths(isolated_pages)
        for i in range(60):
            store.add(f"car{i}", folder="car-spec")
        for i in range(30):
            store.add(f"ai{i}", folder="ai")
        for i in range(10):
            store.add(f"misc{i}")

        page_ids = sorted(store.all_page_ids())
        a = proportional_sample(page_ids, store, n=20, seed=42)
        b = proportional_sample(page_ids, store, n=20, seed=42)
        assert a.page_ids == b.page_ids

    def test_proportional_distribution(self, isolated_pages: Path) -> None:
        store = _FakeStoreWithPaths(isolated_pages)
        for i in range(60):
            store.add(f"car{i}", folder="car-spec")
        for i in range(30):
            store.add(f"ai{i}", folder="ai")
        for i in range(10):
            store.add(f"misc{i}")

        plan = proportional_sample(
            sorted(store.all_page_ids()), store, n=10, seed=42
        )
        # car-spec ~ 60% → 6, ai ~30% → 3, root ~10% → 1
        folders = [
            "car-spec" if p.startswith("car") else "ai" if p.startswith("ai") else ""
            for p in plan.page_ids
        ]
        assert folders.count("car-spec") == 6
        assert folders.count("ai") == 3
        assert folders.count("") == 1


class TestMinoritySample:
    def test_excludes_dominant_folder(self, isolated_pages: Path) -> None:
        store = _FakeStoreWithPaths(isolated_pages)
        for i in range(60):
            store.add(f"car{i}", folder="car-spec")
        for i in range(20):
            store.add(f"ai{i}", folder="ai")
        for i in range(20):
            store.add(f"misc{i}", folder="misc")

        plan = minority_sample(
            sorted(store.all_page_ids()),
            store,
            n=10,
            seed=42,
            dominant_threshold=0.5,
        )
        # car-spec is 60/100 = 60% ≥ 50% threshold → excluded.
        assert all(not pid.startswith("car") for pid in plan.page_ids)


# ---------------------------------------------------------------------------
# analyze_page
# ---------------------------------------------------------------------------


class TestAnalyzePage:
    def test_happy_path(self, isolated_pages: Path) -> None:
        _seed(isolated_pages, "p1", body="content for p1")
        master = ["d/ai-industry", "t/analysis", "s/2026"]

        def fake_generate(prompt, system=None):
            return json.dumps(
                {
                    "main_topic": "happy path",
                    "assigned_tags": ["d/ai-industry", "t/analysis", "s/2026"],
                    "tag_evidence": {
                        "d/ai-industry": "industry brief",
                        "t/analysis": "deep dive",
                        "s/2026": "2026 outlook",
                    },
                    "rejected_assigned_tags": [],
                    "suggested_missing_categories": [],
                    "confidence": 0.85,
                }
            )

        analysis = analyze_page("p1", master, fake_generate)
        assert analysis.page_id == "p1"
        assert analysis.assigned_tags == ["d/ai-industry", "t/analysis", "s/2026"]
        assert analysis.tag_evidence["d/ai-industry"] == "industry brief"
        assert analysis.main_topic == "happy path"
        assert analysis.confidence == pytest.approx(0.85)

    def test_llm_exception_records_audit(self, isolated_pages: Path) -> None:
        _seed(isolated_pages, "p1")
        master = ["d/x"]

        def fake_generate(*_a, **_kw):
            raise RuntimeError("ollama down")

        analysis = analyze_page("p1", master, fake_generate)
        # Page IDs survive even on failure so the report can show them.
        assert analysis.page_id == "p1"
        assert analysis.assigned_tags == []
        assert "ollama error" in analysis.raw_response

    def test_malformed_response_keeps_raw(self, isolated_pages: Path) -> None:
        _seed(isolated_pages, "p1")
        master = ["d/x"]
        analysis = analyze_page(
            "p1", master, lambda *_a, **_kw: "not json"
        )
        assert analysis.assigned_tags == []
        assert analysis.raw_response == "not json"

    def test_schema_error_is_repaired_in_same_structured_session(
        self,
        isolated_pages: Path,
    ) -> None:
        _seed(isolated_pages, "p1", body="MCP config")
        replies = iter(
            [
                json.dumps({"main_topic": "missing fields"}),
                json.dumps(
                    {
                        "main_topic": "MCP設定",
                        "assigned_tags": ["d/tools-config"],
                        "tag_evidence": {"d/tools-config": "MCP config"},
                        "rejected_assigned_tags": [],
                        "suggested_missing_categories": [],
                        "confidence": 0.9,
                    }
                ),
            ]
        )
        prompts: list[str] = []

        def fake_generate(prompt: str, **_kwargs) -> str:
            prompts.append(prompt)
            return next(replies)

        analysis = analyze_page("p1", ["d/tools-config"], fake_generate)

        assert analysis.assigned_tags == ["d/tools-config"]
        assert len(prompts) == 2
        assert "Validator errors" in prompts[1]


# ---------------------------------------------------------------------------
# aggregate
# ---------------------------------------------------------------------------


class TestAggregate:
    def test_weighted_vs_unweighted(self) -> None:
        plan_a = SamplingPlan(
            page_ids=["a1", "a2"],
            weights={"a1": 0.7, "a2": 0.3},
            folder_distribution={},
            seed=1,
        )
        plan_b = SamplingPlan(
            page_ids=["b1"], weights={"b1": 1.0}, folder_distribution={}, seed=2
        )
        results_a = [
            PageAnalysis(
                page_id="a1",
                assigned_tags=["d/ai-industry", "t/analysis", "s/2026"],
                confidence=0.9,
                raw_response="ok",
            ),
            PageAnalysis(
                page_id="a2",
                assigned_tags=["d/ai-industry"],
                confidence=0.5,
                raw_response="ok",
            ),
        ]
        results_b = [
            PageAnalysis(
                page_id="b1",
                assigned_tags=["d/finance"],
                confidence=0.6,
                raw_response="ok",
            )
        ]
        stats = aggregate(plan_a, plan_b, results_a, results_b)
        # a1 (0.7) + a2 (0.3) → d/ai-industry weight = 1.0
        assert stats["weighted_master_frequency"]["d/ai-industry"] == pytest.approx(1.0)
        assert stats["unweighted_master_frequency_b"]["d/finance"] == 1

    def test_hallucination_count(self) -> None:
        plan_a = SamplingPlan(
            page_ids=["a1"], weights={"a1": 1.0}, folder_distribution={}, seed=1
        )
        plan_b = SamplingPlan(page_ids=[], weights={}, folder_distribution={}, seed=2)
        analysis = PageAnalysis(
            page_id="a1",
            assigned_tags=["d/ai-industry"],
            rejected_assigned_tags=["d/oops"],
            confidence=0.7,
            raw_response="ok",
        )
        stats = aggregate(plan_a, plan_b, [analysis], [])
        assert stats["hallucination_pages"] == 1
        assert stats["hallucination_rate"] == pytest.approx(1.0)

    def test_taxonomy_gap_aggregated(self) -> None:
        plan_a = SamplingPlan(
            page_ids=["a1", "a2"],
            weights={"a1": 1.0, "a2": 1.0},
            folder_distribution={},
            seed=1,
        )
        plan_b = SamplingPlan(page_ids=[], weights={}, folder_distribution={}, seed=2)
        results = [
            PageAnalysis(
                page_id="a1",
                suggested_missing_categories=[
                    {"label": "automotive-engineering", "justification": "x", "fallback_axis": "d/"}
                ],
                confidence=0.5,
                raw_response="ok",
            ),
            PageAnalysis(
                page_id="a2",
                suggested_missing_categories=[
                    {"label": "automotive-engineering", "justification": "y", "fallback_axis": "d/"},
                    {"label": "ml-ops", "justification": "z", "fallback_axis": "d/"},
                ],
                confidence=0.6,
                raw_response="ok",
            ),
        ]
        stats = aggregate(plan_a, plan_b, results, [])
        assert dict(stats["taxonomy_gap_top30"])["automotive-engineering"] == 2
        assert dict(stats["taxonomy_gap_top30"])["ml-ops"] == 1


# ---------------------------------------------------------------------------
# run_dry_run end-to-end (no real Ollama)
# ---------------------------------------------------------------------------


class TestRunDryRun:
    def test_writes_report_and_raw_log(self, tmp_path: Path, isolated_pages: Path) -> None:
        store = _FakeStoreWithPaths(isolated_pages)
        for i in range(20):
            store.add(f"a{i}", folder="ai")
            _seed(isolated_pages, f"a{i}", folder="ai", body=f"body {i}")
        for i in range(20):
            store.add(f"m{i}", folder="misc")
            _seed(isolated_pages, f"m{i}", folder="misc", body=f"body {i}")

        def fake_generate(prompt, system=None):
            # Master tags only; everyone gets the same canonical set.
            return json.dumps(
                {
                    "main_topic": "test page summary",
                    "assigned_tags": ["d/ai-industry", "t/analysis", "s/2026"],
                    "tag_evidence": {
                        "d/ai-industry": "industry brief",
                        "t/analysis": "deep dive",
                        "s/2026": "2026 outlook",
                    },
                    "rejected_assigned_tags": [],
                    "suggested_missing_categories": [],
                    "confidence": 0.8,
                }
            )

        output = tmp_path / "report.md"
        raw_log = tmp_path / "raw.jsonl"
        stats = run_dry_run(
            output,
            raw_log,
            sample_a_n=5,
            sample_b_n=5,
            seed=42,
            store=store,
            generate_fn=fake_generate,
        )

        assert stats["sample_a_n"] == 5
        assert stats["sample_b_n"] == 5
        assert stats["population_total"] == 40
        assert output.exists()
        assert raw_log.exists()
        # Each sample contributes one line per page.
        lines = raw_log.read_text().strip().splitlines()
        assert len(lines) == 10
        # Every line is valid JSON with the expected shape.
        for ln in lines:
            rec = json.loads(ln)
            assert {"run_id", "sample", "page_id", "assigned_tags"}.issubset(rec.keys())

    def test_pages_unchanged(self, tmp_path: Path, isolated_pages: Path) -> None:
        store = _FakeStoreWithPaths(isolated_pages)
        for i in range(5):
            store.add(f"p{i}")
            _seed(isolated_pages, f"p{i}", body=f"ORIGINAL_{i}")

        before = {p.name: p.read_bytes() for p in isolated_pages.rglob("*.md")}

        run_dry_run(
            tmp_path / "report.md",
            tmp_path / "raw.jsonl",
            sample_a_n=3,
            sample_b_n=2,
            seed=42,
            store=store,
            generate_fn=lambda *_a, **_kw: json.dumps(
                {
                    "main_topic": "empty",
                    "assigned_tags": [],
                    "tag_evidence": {},
                    "rejected_assigned_tags": [],
                    "suggested_missing_categories": [],
                    "confidence": 0.0,
                }
            ),
        )

        after = {p.name: p.read_bytes() for p in isolated_pages.rglob("*.md")}
        assert before == after


# ---------------------------------------------------------------------------
# format_report — quick render sanity
# ---------------------------------------------------------------------------


class TestFormatReport:
    def test_renders_sections(self) -> None:
        plan_a = SamplingPlan(
            page_ids=["a1"], weights={"a1": 1.0}, folder_distribution={"ai": 1}, seed=1
        )
        plan_b = SamplingPlan(
            page_ids=["b1"], weights={"b1": 1.0}, folder_distribution={"misc": 1}, seed=2
        )
        results_a = [
            PageAnalysis(page_id="a1", assigned_tags=["d/ai-industry"], confidence=0.7, raw_response="ok")
        ]
        results_b = [
            PageAnalysis(page_id="b1", assigned_tags=["d/finance"], confidence=0.4, raw_response="ok")
        ]
        stats = aggregate(plan_a, plan_b, results_a, results_b)
        text = format_report(
            plan_a, plan_b, results_a, results_b, stats,
            population_total=2, elapsed=1.5,
        )
        assert "Tag Distribution Report" in text
        assert "sample A" in text
        assert "sample B" in text
        assert "d/ai-industry" in text
        assert "d/finance" in text
        assert "Hallucination" in text
