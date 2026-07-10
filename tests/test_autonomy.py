from __future__ import annotations

from contextlib import contextmanager
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from llm_wiki_mcp import autonomy
from llm_wiki_mcp.convergence import ConvergenceStore, CycleBudget, RetryPolicy
from llm_wiki_mcp.frontmatter import parse as parse_frontmatter


NOW = datetime(2026, 7, 10, 12, 0, tzinfo=timezone.utc)


def _write_page(path: Path, title: str, body: str) -> None:
    path.write_text(
        f"---\ntitle: {title}\nstatus: active\nupdated: 2026-07-10\n---\n{body}\n",
        encoding="utf-8",
    )


def _convergence_store(
    tmp_path: Path,
    *,
    max_local_attempts: int = 2,
    max_frontier_attempts: int = 3,
    local_delay: int = 300,
    frontier_delay: int = 60,
) -> ConvergenceStore:
    runtime = tmp_path / "convergence"
    return ConvergenceStore(
        runtime / "state.json",
        events_file=runtime / "events.jsonl",
        lock_file=runtime / "state.lock",
        policy=RetryPolicy(
            max_local_attempts=max_local_attempts,
            max_frontier_attempts=max_frontier_attempts,
            local_base_delay_seconds=local_delay,
            frontier_base_delay_seconds=frontier_delay,
            lease_seconds=30,
        ),
    )


def _frontier_budget(*, calls: int = 3, mutations: int = 3) -> CycleBudget:
    return CycleBudget(
        max_local_calls=10,
        max_frontier_calls=calls,
        max_mutations=mutations,
        max_elapsed_seconds=300,
    )


def _deferred_record(left: str = "a", right: str = "b") -> dict:
    return {
        "left": left,
        "right": right,
        "left_title": f"Title {left}",
        "right_title": f"Different {right}",
        "score": 0.999,
        "method": "embedding",
    }


def _exact_record(left: str = "a", right: str = "b") -> dict:
    return {
        "left": left,
        "right": right,
        "left_title": "Same",
        "right_title": "Same",
        "score": 1.0,
        "method": "title",
    }


def test_duplicate_decision_defers_uncertain_pair(monkeypatch) -> None:
    monkeypatch.setattr(autonomy, "_page_meta", lambda page_id: {"page_id": page_id})
    decision = autonomy.decide_duplicate(
        {
            "left": "a",
            "right": "b",
            "left_title": "Alpha",
            "right_title": "Beta",
            "score": 0.999,
            "method": "embedding",
        }
    )

    assert decision["action"] == "defer"
    assert decision["reason"] == "title_mismatch"


def test_duplicate_resolution_routes_exact_high_confidence_pair_without_mutation(monkeypatch, tmp_path: Path) -> None:
    pages = {
        "rich": {
            "page_id": "rich",
            "title": "Same",
            "summary": "Useful",
            "recall_questions": ["q1", "q2"],
            "path": str(tmp_path / "rich.md"),
        },
        "thin": {
            "page_id": "thin",
            "title": "Same",
            "summary": "",
            "recall_questions": [],
            "path": str(tmp_path / "thin.md"),
        },
    }
    (tmp_path / "rich.md").write_text("---\ntitle: Same\nupdated: 2026-07-06\n---\nRich\n", encoding="utf-8")
    (tmp_path / "thin.md").write_text("---\ntitle: Same\nupdated: 2026-07-06\n---\nThin\n", encoding="utf-8")
    monkeypatch.setattr(autonomy, "_page_meta", lambda page_id: pages.get(page_id, {}))
    monkeypatch.setattr(autonomy, "find_page", lambda page_id: tmp_path / f"{page_id}.md")
    monkeypatch.setattr("llm_wiki_mcp.index_store.get_store", lambda: None)
    monkeypatch.setattr(autonomy, "_page_quality", lambda page_id, meta=None: 5.0 if page_id == "rich" else 1.0)
    writes: list[dict] = []
    monkeypatch.setattr(autonomy, "_append_jsonl", lambda path, row: writes.append(row))

    payload = autonomy.resolve_duplicate_candidates(
        [
            {
                "left": "rich",
                "right": "thin",
                "left_title": "Same",
                "right_title": "Same",
                "score": 1.0,
                "method": "title",
            }
        ],
        apply=True,
        write=True,
    )

    assert payload["applied"] == 0
    assert payload["deferred"] == 1
    text = (tmp_path / "thin.md").read_text(encoding="utf-8")
    assert "status: deprecated" not in text
    assert "superseded_by: rich" not in text
    assert writes[0]["action"] == "defer"
    assert writes[0]["reason"] == "frontier_approval_required"
    assert writes[0]["proposal"]["winner"] == "rich"


def test_deterministic_duplicate_proposal_never_calls_lifecycle_writer(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _write_page(tmp_path / "a.md", "Same", "Winner")
    _write_page(tmp_path / "b.md", "Same", "Loser")
    monkeypatch.setattr(autonomy, "find_page", lambda page_id: tmp_path / f"{page_id}.md")
    monkeypatch.setattr(autonomy, "_page_meta", lambda page_id: {"page_id": page_id})
    monkeypatch.setattr(
        autonomy,
        "_page_quality",
        lambda page_id, meta=None: 5.0 if page_id == "a" else 1.0,
    )
    before = (tmp_path / "b.md").read_bytes()
    monkeypatch.setattr(
        autonomy,
        "_soft_supersede_page",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("lifecycle writer called")),
    )

    result = autonomy.resolve_duplicate_candidates([_exact_record()], write=False)

    assert result["applied"] == 0
    assert result["deferred"] == 1
    assert result["decisions"][0]["result"]["reason"] == "deterministic_heuristic_is_proposal_only"
    meta, body = parse_frontmatter((tmp_path / "b.md").read_text(encoding="utf-8"))
    assert meta["status"] == "active"
    assert "superseded_by" not in meta
    assert body == "Loser\n"
    assert (tmp_path / "b.md").read_bytes() == before


def test_soft_supersede_preserves_correction_that_lands_before_locked_cas(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _write_page(tmp_path / "a.md", "Same", "Winner")
    _write_page(tmp_path / "b.md", "Same", "Loser")
    monkeypatch.setattr(autonomy, "find_page", lambda page_id: tmp_path / f"{page_id}.md")
    loser_snapshot = autonomy._duplicate_page_snapshot("b")
    winner_snapshot = autonomy._duplicate_page_snapshot("a")
    loser = tmp_path / "b.md"
    corrected = loser.read_text(encoding="utf-8") + "user correction\n"

    @contextmanager
    def correction_wins():
        loser.write_text(corrected, encoding="utf-8")
        yield

    monkeypatch.setattr(autonomy, "wiki_mutation_lock", correction_wins)

    result = autonomy._soft_supersede_page(
        loser="b",
        winner="a",
        expected_loser_hash=loser_snapshot["content_hash"],
        expected_winner_hash=winner_snapshot["content_hash"],
        decision_at=NOW.isoformat(),
    )

    assert result == {"status": "retry", "reason": "content_changed_before_apply"}
    assert loser.read_text(encoding="utf-8") == corrected


def test_lifecycle_writers_defer_pages_with_pending_content_correction(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _write_page(tmp_path / "winner.md", "Same", "Winner")
    _write_page(tmp_path / "loser.md", "Same", "Loser")
    _write_page(tmp_path / "old.md", "Old", "Archive candidate")
    monkeypatch.setattr(autonomy, "find_page", lambda page_id: tmp_path / f"{page_id}.md")
    store = _convergence_store(tmp_path)
    store.merge_item(
        lane="content_correction",
        source_id="codex:session:turn",
        input_data={"correction": "wrong count"},
        metadata={"candidate_pages": ["loser", "old"]},
    )
    loser_snapshot = autonomy._duplicate_page_snapshot("loser")
    winner_snapshot = autonomy._duplicate_page_snapshot("winner")

    supersede = autonomy._soft_supersede_page(
        loser="loser",
        winner="winner",
        expected_loser_hash=loser_snapshot["content_hash"],
        expected_winner_hash=winner_snapshot["content_hash"],
        decision_at=NOW.isoformat(),
        correction_store=store,
    )
    archive = autonomy.apply_retention_archives(
        {
            "archive_candidates": ["old"],
            "pages": {"old": {"score": 0.1}},
        },
        write=False,
        correction_store=store,
        reviewer=lambda _candidate: {
            "decision": "archive",
            "confidence": 0.99,
            "summary": "Obsolete duplicate",
        },
        now=NOW,
    )

    assert supersede["status"] == "retry"
    assert supersede["reason"] == "pending_content_correction"
    loser_meta, _ = parse_frontmatter((tmp_path / "loser.md").read_text(encoding="utf-8"))
    old_meta, _ = parse_frontmatter((tmp_path / "old.md").read_text(encoding="utf-8"))
    assert loser_meta["status"] == "active"
    assert old_meta["status"] == "active"
    assert archive["applied"] == 0
    assert archive["decisions"][0]["reason"] == "pending_content_correction"


def test_lifecycle_guard_keeps_auto_resumable_quarantine_protected(
    tmp_path: Path,
) -> None:
    page = tmp_path / "old.md"
    _write_page(page, "Old", "Claim awaiting correction")
    store = _convergence_store(tmp_path)
    merged = store.merge_item(
        lane="content_correction",
        source_id="codex:session:quarantined-turn",
        input_data={"correction": "the claim is wrong"},
        metadata={"candidate_pages": ["old"]},
    )
    key = merged["item"]["key"]
    store.quarantine(key, reason="frontier temporarily unavailable")

    with autonomy._lifecycle_mutation_guard(
        ["old"], page_path=page, correction_store=store
    ) as quarantined_guard:
        assert quarantined_guard["allowed"] is False
        assert quarantined_guard["reason"] == "pending_content_correction"

    store.resume_quarantined(key, stage="frontier")
    store.complete(key, "applied", result={"correction": "verified"})

    with autonomy._lifecycle_mutation_guard(
        ["old"], page_path=page, correction_store=store
    ) as resolved_guard:
        assert resolved_guard["allowed"] is True


def test_lifecycle_guard_avoids_inverse_lock_order(
    monkeypatch,
    tmp_path: Path,
) -> None:
    page = tmp_path / "old.md"
    _write_page(page, "Old", "Body")
    store = _convergence_store(tmp_path)
    order: list[str] = []
    real_state_lock = store._exclusive_lock

    @contextmanager
    def tracked_state_lock():
        order.append("state-enter")
        with real_state_lock():
            yield
        order.append("state-exit")

    @contextmanager
    def tracked_wiki_lock():
        assert order == ["state-enter"]
        order.append("wiki-enter")
        yield
        order.append("wiki-exit")

    monkeypatch.setattr(store, "_exclusive_lock", tracked_state_lock)
    monkeypatch.setattr(autonomy, "wiki_mutation_lock", tracked_wiki_lock)

    with autonomy._lifecycle_mutation_guard(
        ["old"], page_path=page, correction_store=store
    ) as guard:
        assert guard["allowed"] is True

    assert order == ["state-enter", "wiki-enter", "wiki-exit", "state-exit"]


def test_autonomy_temp_names_are_unique_for_same_page_and_preimage(tmp_path: Path) -> None:
    page = tmp_path / "old.md"
    _write_page(page, "Old", "Body")
    first = autonomy._write_unique_temp(page, b"one", token="same")
    second = autonomy._write_unique_temp(page, b"two", token="same")
    try:
        assert first != second
        assert first.read_bytes() == b"one"
        assert second.read_bytes() == b"two"
    finally:
        first.unlink(missing_ok=True)
        second.unlink(missing_ok=True)


def test_deterministic_duplicate_proposal_does_not_spend_mutation_budget(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _write_page(tmp_path / "a.md", "Same", "Winner")
    _write_page(tmp_path / "b.md", "Same", "Loser")
    before = (tmp_path / "b.md").read_bytes()
    monkeypatch.setattr(autonomy, "find_page", lambda page_id: tmp_path / f"{page_id}.md")
    monkeypatch.setattr(autonomy, "_page_meta", lambda page_id: {"page_id": page_id})
    monkeypatch.setattr(
        autonomy,
        "_page_quality",
        lambda page_id, meta=None: 5.0 if page_id == "a" else 1.0,
    )
    budget = _frontier_budget(mutations=0)

    result = autonomy.resolve_duplicate_candidates(
        [_exact_record()],
        write=False,
        budget=budget,
    )

    assert result["applied"] == 0
    assert result["deferred"] == 1
    assert result["decisions"][0]["reason"] == "frontier_approval_required"
    assert budget.snapshot()["used"]["mutation"] == 0
    assert (tmp_path / "b.md").read_bytes() == before


def test_soft_supersede_rolls_back_only_its_owned_write_on_postverify_failure(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _write_page(tmp_path / "a.md", "Alpha", "Winner")
    _write_page(tmp_path / "b.md", "Beta", "Loser")
    original_loser = (tmp_path / "b.md").read_bytes()
    monkeypatch.setattr(autonomy, "find_page", lambda page_id: tmp_path / f"{page_id}.md")
    loser_snapshot = autonomy._duplicate_page_snapshot("b")
    winner_snapshot = autonomy._duplicate_page_snapshot("a")
    real_replace = autonomy.os.replace
    replace_calls = 0

    def replace_then_change_winner(src, dst):
        nonlocal replace_calls
        real_replace(src, dst)
        replace_calls += 1
        if replace_calls == 1:
            winner = tmp_path / "a.md"
            winner.write_text(
                winner.read_text(encoding="utf-8") + "concurrent winner update\n",
                encoding="utf-8",
            )

    monkeypatch.setattr(autonomy.os, "replace", replace_then_change_winner)

    result = autonomy._soft_supersede_page(
        loser="b",
        winner="a",
        expected_loser_hash=loser_snapshot["content_hash"],
        expected_winner_hash=winner_snapshot["content_hash"],
        decision_at=NOW.isoformat(),
    )

    assert result == {
        "status": "retry",
        "reason": "post_write_verification_failed",
        "rolled_back": True,
    }
    assert (tmp_path / "b.md").read_bytes() == original_loser
    assert (tmp_path / "a.md").read_text(encoding="utf-8").endswith(
        "concurrent winner update\n"
    )


def test_soft_supersede_does_not_rollback_over_a_foreign_postwrite_change(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _write_page(tmp_path / "a.md", "Alpha", "Winner")
    _write_page(tmp_path / "b.md", "Beta", "Loser")
    monkeypatch.setattr(autonomy, "find_page", lambda page_id: tmp_path / f"{page_id}.md")
    loser_snapshot = autonomy._duplicate_page_snapshot("b")
    winner_snapshot = autonomy._duplicate_page_snapshot("a")
    real_replace = autonomy.os.replace
    replace_calls = 0

    def replace_then_change_both(src, dst):
        nonlocal replace_calls
        real_replace(src, dst)
        replace_calls += 1
        if replace_calls == 1:
            loser = tmp_path / "b.md"
            winner = tmp_path / "a.md"
            loser.write_text(
                loser.read_text(encoding="utf-8") + "foreign loser update\n",
                encoding="utf-8",
            )
            winner.write_text(
                winner.read_text(encoding="utf-8") + "foreign winner update\n",
                encoding="utf-8",
            )

    monkeypatch.setattr(autonomy.os, "replace", replace_then_change_both)

    result = autonomy._soft_supersede_page(
        loser="b",
        winner="a",
        expected_loser_hash=loser_snapshot["content_hash"],
        expected_winner_hash=winner_snapshot["content_hash"],
        decision_at=NOW.isoformat(),
    )

    assert result == {
        "status": "retry",
        "reason": "post_write_verification_failed",
        "rolled_back": False,
    }
    assert (tmp_path / "b.md").read_text(encoding="utf-8").endswith(
        "foreign loser update\n"
    )


def test_frontier_consumer_persists_approval_before_deferring_empty_mutation_budget(
    monkeypatch,
    tmp_path: Path,
) -> None:
    pages = tmp_path / "pages"
    pages.mkdir()
    _write_page(pages / "a.md", "Same", "Winner")
    _write_page(pages / "b.md", "Same", "Loser")
    before = (pages / "b.md").read_bytes()
    monkeypatch.setattr(autonomy, "find_page", lambda page_id: pages / f"{page_id}.md")
    monkeypatch.setattr(autonomy, "_page_meta", lambda page_id: {"page_id": page_id})
    monkeypatch.setattr(
        autonomy,
        "_page_quality",
        lambda page_id, meta=None: 5.0 if page_id == "a" else 1.0,
    )
    store = _convergence_store(tmp_path)
    budget = _frontier_budget(calls=1, mutations=0)

    result = autonomy.resolve_deferred_duplicates_with_frontier(
        [_exact_record()],
        convergence_store=store,
        budget=budget,
        reviewer=lambda _candidate: {
            "decision": "supersede_right",
            "confidence": 0.99,
            "summary": "A subsumes B",
        },
        now=NOW,
        write=False,
    )

    assert result["status_counts"] == {"frontier_retry": 1}
    item = store.list_items(lane=autonomy.DUPLICATE_FRONTIER_LANE)[0]
    assert item["status"] == "frontier_retry"
    assert item["local_attempts"] == 0
    assert item["frontier_attempts"] == 1
    assert budget.snapshot()["used"] == {
        "local": 0,
        "frontier": 1,
        "mutation": 0,
        "raw_bytes": 0,
    }
    approval_files = list((store.state_file.parent / "approvals").rglob("*.json"))
    assert len(approval_files) == 1
    assert (pages / "b.md").read_bytes() == before


def test_duplicate_reuses_durable_frontier_approval_for_mutation_retry(
    monkeypatch,
    tmp_path: Path,
) -> None:
    pages = tmp_path / "pages"
    pages.mkdir()
    _write_page(pages / "a.md", "Same", "Winner")
    _write_page(pages / "b.md", "Same", "Loser")
    monkeypatch.setattr(autonomy, "find_page", lambda page_id: pages / f"{page_id}.md")
    monkeypatch.setattr(autonomy, "_page_meta", lambda page_id: {"page_id": page_id})
    monkeypatch.setattr(
        autonomy,
        "_page_quality",
        lambda page_id, meta=None: 5.0 if page_id == "a" else 1.0,
    )
    store = _convergence_store(tmp_path)
    calls: list[dict] = []

    def reviewer(candidate: dict) -> dict:
        calls.append(candidate)
        return {
            "decision": "supersede_right",
            "confidence": 0.99,
            "summary": "A subsumes B",
        }

    first = autonomy.resolve_deferred_duplicates_with_frontier(
        [_exact_record()],
        convergence_store=store,
        budget=_frontier_budget(calls=1, mutations=0),
        reviewer=reviewer,
        now=NOW,
        write=False,
    )
    second = autonomy.resolve_deferred_duplicates_with_frontier(
        [_exact_record()],
        convergence_store=store,
        budget=_frontier_budget(calls=1, mutations=1),
        reviewer=lambda _candidate: (_ for _ in ()).throw(
            AssertionError("durable approval was not reused")
        ),
        now=NOW + timedelta(seconds=61),
        write=False,
    )

    assert first["status_counts"] == {"frontier_retry": 1}
    assert second["status_counts"] == {"applied": 1}
    assert second["frontier_calls"] == 0
    assert len(calls) == 1
    meta, body = parse_frontmatter((pages / "b.md").read_text(encoding="utf-8"))
    assert meta["status"] == "deprecated"
    assert meta["superseded_by"] == "a"
    assert meta["frontier_approval_key"].startswith("duplicate_frontier:")
    assert body == "Loser\n"


def test_duplicate_recovers_convergence_after_crash_following_approved_write(
    monkeypatch,
    tmp_path: Path,
) -> None:
    pages = tmp_path / "pages"
    pages.mkdir()
    _write_page(pages / "a.md", "Same", "Winner")
    _write_page(pages / "b.md", "Same", "Loser")
    monkeypatch.setattr(autonomy, "find_page", lambda page_id: pages / f"{page_id}.md")
    store = _convergence_store(tmp_path)
    real_complete = store.complete
    crashed = False

    def crash_after_write(key, status, **kwargs):
        nonlocal crashed
        if status == "applied" and not crashed:
            crashed = True
            raise RuntimeError("simulated process crash")
        return real_complete(key, status, **kwargs)

    monkeypatch.setattr(store, "complete", crash_after_write)
    with pytest.raises(RuntimeError, match="simulated process crash"):
        autonomy.resolve_deferred_duplicates_with_frontier(
            [_deferred_record()],
            convergence_store=store,
            budget=_frontier_budget(calls=1, mutations=1),
            reviewer=lambda _candidate: {
                "decision": "supersede_right",
                "confidence": 0.99,
                "summary": "A subsumes B",
            },
            now=NOW,
            write=False,
        )

    meta, _body = parse_frontmatter((pages / "b.md").read_text(encoding="utf-8"))
    assert meta["status"] == "deprecated"
    monkeypatch.setattr(store, "complete", real_complete)
    recovered = autonomy.resolve_deferred_duplicates_with_frontier(
        [_deferred_record()],
        convergence_store=store,
        budget=_frontier_budget(calls=0, mutations=0),
        reviewer=lambda _candidate: (_ for _ in ()).throw(
            AssertionError("reviewer called during receipt recovery")
        ),
        now=NOW + timedelta(seconds=31),
        write=False,
    )

    assert recovered["status_counts"] == {"already_applied": 1}
    assert recovered["results"][0]["convergence_status"] == "applied"
    item = store.list_items(lane=autonomy.DUPLICATE_FRONTIER_LANE)[0]
    assert item["status"] == "applied"


def test_frontier_consumer_routes_exact_title_heuristic_directly_to_frontier(
    monkeypatch,
    tmp_path: Path,
) -> None:
    pages = tmp_path / "pages"
    pages.mkdir()
    _write_page(pages / "a.md", "Same", "Winner")
    _write_page(pages / "b.md", "Same", "Loser")
    monkeypatch.setattr(autonomy, "find_page", lambda page_id: pages / f"{page_id}.md")
    monkeypatch.setattr(autonomy, "_page_meta", lambda page_id: {"page_id": page_id})
    monkeypatch.setattr(
        autonomy,
        "_page_quality",
        lambda page_id, meta=None: 5.0 if page_id == "a" else 1.0,
    )
    monkeypatch.setattr(autonomy, "DECISIONS_FILE", tmp_path / "decisions.jsonl")
    store = _convergence_store(tmp_path, max_local_attempts=1)

    frontier = autonomy.resolve_deferred_duplicates_with_frontier(
        [_exact_record()],
        convergence_store=store,
        budget=_frontier_budget(calls=1, mutations=1),
        reviewer=lambda _candidate: {
            "decision": "keep_both",
            "confidence": 0.95,
            "summary": "Distinct evidence",
        },
        now=NOW,
    )

    assert frontier["status_counts"] == {"rejected": 1}
    item = store.list_items(lane=autonomy.DUPLICATE_FRONTIER_LANE)[0]
    assert item["status"] == "rejected"
    assert item["local_attempts"] == 0
    assert item["frontier_attempts"] == 1


def test_frontier_consumer_deterministic_dry_run_is_byte_for_byte_read_only(
    monkeypatch,
    tmp_path: Path,
) -> None:
    pages = tmp_path / "pages"
    pages.mkdir()
    _write_page(pages / "a.md", "Same", "Winner")
    _write_page(pages / "b.md", "Same", "Loser")
    before = {path.name: path.read_bytes() for path in pages.iterdir()}
    monkeypatch.setattr(autonomy, "find_page", lambda page_id: pages / f"{page_id}.md")
    monkeypatch.setattr(autonomy, "_page_meta", lambda page_id: {"page_id": page_id})
    monkeypatch.setattr(
        autonomy,
        "_page_quality",
        lambda page_id, meta=None: 5.0 if page_id == "a" else 1.0,
    )
    store = _convergence_store(tmp_path)
    budget = _frontier_budget(calls=1, mutations=1)

    result = autonomy.resolve_deferred_duplicates_with_frontier(
        [_exact_record()],
        convergence_store=store,
        budget=budget,
        reviewer=lambda _candidate: (_ for _ in ()).throw(AssertionError("reviewer called")),
        now=NOW,
        dry_run=True,
    )

    assert result["status_counts"] == {"would_review": 1}
    assert budget.snapshot()["used"] == {
        "local": 0,
        "frontier": 0,
        "mutation": 0,
        "raw_bytes": 0,
    }
    assert {path.name: path.read_bytes() for path in pages.iterdir()} == before
    assert not store.state_file.exists()
    assert not store.events_file.exists()
    assert not store.lock_file.exists()


def test_frontier_duplicate_soft_supersede_preserves_bodies_and_is_exact_once(
    monkeypatch,
    tmp_path: Path,
) -> None:
    pages = tmp_path / "pages"
    pages.mkdir()
    _write_page(pages / "a.md", "Alpha", "Body A stays exactly here.")
    _write_page(pages / "b.md", "Beta", "Body B stays exactly here.")
    original_a = (pages / "a.md").read_bytes()
    monkeypatch.setattr(autonomy, "find_page", lambda page_id: pages / f"{page_id}.md")
    monkeypatch.setattr(autonomy, "DECISIONS_FILE", tmp_path / "decisions.jsonl")
    store = _convergence_store(tmp_path)
    calls: list[dict] = []

    def reviewer(candidate: dict) -> dict:
        calls.append(candidate)
        assert candidate["left"] == "a"
        assert candidate["right"] == "b"
        return {"decision": "supersede_right", "confidence": 0.97, "summary": "A subsumes B"}

    first = autonomy.resolve_deferred_duplicates_with_frontier(
        [_deferred_record()],
        convergence_store=store,
        budget=_frontier_budget(calls=1, mutations=1),
        reviewer=reviewer,
        now=NOW,
    )
    second = autonomy.resolve_deferred_duplicates_with_frontier(
        [_deferred_record()],
        convergence_store=store,
        budget=_frontier_budget(calls=1, mutations=1),
        reviewer=reviewer,
        now=NOW + timedelta(minutes=1),
    )

    assert first["applied"] == 1
    assert second["status_counts"] == {"already_applied": 1}
    assert len(calls) == 1
    assert (pages / "a.md").read_bytes() == original_a
    b_meta, b_body = parse_frontmatter((pages / "b.md").read_text(encoding="utf-8"))
    assert b_meta["status"] == "deprecated"
    assert b_meta["superseded_by"] == "a"
    assert b_body == "Body B stays exactly here.\n"
    assert sorted(path.name for path in pages.iterdir()) == ["a.md", "b.md"]
    assert len((tmp_path / "decisions.jsonl").read_text(encoding="utf-8").splitlines()) == 1


def test_frontier_duplicate_key_is_pair_order_stable_and_content_sensitive(
    monkeypatch,
    tmp_path: Path,
) -> None:
    pages = tmp_path / "pages"
    pages.mkdir()
    _write_page(pages / "a.md", "Alpha", "A")
    _write_page(pages / "b.md", "Beta", "B")
    monkeypatch.setattr(autonomy, "find_page", lambda page_id: pages / f"{page_id}.md")
    monkeypatch.setattr(autonomy, "DECISIONS_FILE", tmp_path / "decisions.jsonl")
    store = _convergence_store(tmp_path)
    calls: list[dict] = []

    def reviewer(candidate: dict) -> dict:
        calls.append(candidate)
        return {"decision": "keep_both", "confidence": 0.91, "summary": "Distinct scope"}

    first = autonomy.resolve_deferred_duplicates_with_frontier(
        [_deferred_record("a", "b")],
        convergence_store=store,
        budget=_frontier_budget(calls=1),
        reviewer=reviewer,
        now=NOW,
    )
    reversed_record = _deferred_record("b", "a")
    second = autonomy.resolve_deferred_duplicates_with_frontier(
        [reversed_record],
        convergence_store=store,
        budget=_frontier_budget(calls=1),
        reviewer=reviewer,
        now=NOW + timedelta(minutes=1),
    )
    (pages / "b.md").write_text(
        (pages / "b.md").read_text(encoding="utf-8") + "new evidence\n",
        encoding="utf-8",
    )
    changed = autonomy.resolve_deferred_duplicates_with_frontier(
        [_deferred_record("a", "b")],
        convergence_store=store,
        budget=_frontier_budget(calls=1),
        reviewer=reviewer,
        now=NOW + timedelta(minutes=2),
    )

    assert first["kept_both"] == 1
    assert second["frontier_calls"] == 0
    assert second["status_counts"] == {"rejected": 1}
    assert changed["frontier_calls"] == 1
    assert len(calls) == 2
    assert len(store.list_items(lane=autonomy.DUPLICATE_FRONTIER_LANE)) == 2


def test_frontier_duplicate_content_cas_refuses_concurrent_page_change(
    monkeypatch,
    tmp_path: Path,
) -> None:
    pages = tmp_path / "pages"
    pages.mkdir()
    _write_page(pages / "a.md", "Alpha", "A")
    _write_page(pages / "b.md", "Beta", "B")
    monkeypatch.setattr(autonomy, "find_page", lambda page_id: pages / f"{page_id}.md")
    monkeypatch.setattr(autonomy, "DECISIONS_FILE", tmp_path / "decisions.jsonl")
    store = _convergence_store(tmp_path)

    def reviewer(_candidate: dict) -> dict:
        (pages / "b.md").write_text(
            (pages / "b.md").read_text(encoding="utf-8") + "concurrent update\n",
            encoding="utf-8",
        )
        return {"decision": "supersede_right", "confidence": 0.99, "summary": "A subsumes B"}

    result = autonomy.resolve_deferred_duplicates_with_frontier(
        [_deferred_record()],
        convergence_store=store,
        budget=_frontier_budget(calls=1, mutations=1),
        reviewer=reviewer,
        now=NOW,
    )

    assert result["status_counts"] == {"frontier_retry": 1}
    meta, body = parse_frontmatter((pages / "b.md").read_text(encoding="utf-8"))
    assert meta["status"] == "active"
    assert "superseded_by" not in meta
    assert body.endswith("concurrent update\n")


def test_frontier_duplicate_retries_low_confidence_decision_without_mutation(
    monkeypatch,
    tmp_path: Path,
) -> None:
    pages = tmp_path / "pages"
    pages.mkdir()
    _write_page(pages / "a.md", "Alpha", "A")
    _write_page(pages / "b.md", "Beta", "B")
    before = {path.name: path.read_bytes() for path in pages.iterdir()}
    monkeypatch.setattr(autonomy, "find_page", lambda page_id: pages / f"{page_id}.md")
    monkeypatch.setattr(autonomy, "DECISIONS_FILE", tmp_path / "decisions.jsonl")
    store = _convergence_store(tmp_path)
    budget = _frontier_budget(calls=1, mutations=1)

    result = autonomy.resolve_deferred_duplicates_with_frontier(
        [_deferred_record()],
        convergence_store=store,
        budget=budget,
        reviewer=lambda _candidate: {
            "decision": "supersede_right",
            "confidence": 0.79,
            "summary": "Probably duplicate",
        },
        now=NOW,
    )

    assert result["status_counts"] == {"frontier_retry": 1}
    assert result["kept_both"] == 0
    assert result["results"][0]["decision"] == "needs_retry"
    assert budget.snapshot()["used"]["mutation"] == 0
    assert {path.name: path.read_bytes() for path in pages.iterdir()} == before


def test_frontier_duplicate_accepts_structured_runner_reviewer_annotation() -> None:
    normalized = autonomy._normalize_duplicate_frontier_review(
        {
            "decision": "keep_both",
            "confidence": 0.95,
            "summary": "Distinct pages",
            "reviewer": "frontier",
        }
    )

    assert normalized["schema_valid"] is True
    assert normalized["decision"] == "keep_both"


def test_frontier_duplicate_invalid_supersede_schema_retries_without_mutation(
    monkeypatch,
    tmp_path: Path,
) -> None:
    pages = tmp_path / "pages"
    pages.mkdir()
    _write_page(pages / "a.md", "Alpha", "A")
    _write_page(pages / "b.md", "Beta", "B")
    before = {path.name: path.read_bytes() for path in pages.iterdir()}
    monkeypatch.setattr(autonomy, "find_page", lambda page_id: pages / f"{page_id}.md")
    monkeypatch.setattr(autonomy, "DECISIONS_FILE", tmp_path / "decisions.jsonl")
    store = _convergence_store(tmp_path)
    budget = _frontier_budget(calls=1, mutations=1)

    result = autonomy.resolve_deferred_duplicates_with_frontier(
        [_deferred_record()],
        convergence_store=store,
        budget=budget,
        reviewer=lambda _candidate: {
            "decision": "supersede_right",
            "confidence": 0.99,
            "summary": "Duplicate",
            "unexpected": "must not be accepted",
        },
        now=NOW,
    )

    assert result["status_counts"] == {"frontier_retry": 1}
    assert result["results"][0]["decision"] == "needs_retry"
    assert budget.snapshot()["used"]["mutation"] == 0
    assert {path.name: path.read_bytes() for path in pages.iterdir()} == before


def test_frontier_duplicate_retries_with_backoff_then_quarantines(
    monkeypatch,
    tmp_path: Path,
) -> None:
    pages = tmp_path / "pages"
    pages.mkdir()
    _write_page(pages / "a.md", "Alpha", "A")
    _write_page(pages / "b.md", "Beta", "B")
    monkeypatch.setattr(autonomy, "find_page", lambda page_id: pages / f"{page_id}.md")
    monkeypatch.setattr(autonomy, "DECISIONS_FILE", tmp_path / "decisions.jsonl")
    store = _convergence_store(tmp_path, max_frontier_attempts=2, frontier_delay=60)
    calls = 0

    def reviewer(_candidate: dict) -> dict:
        nonlocal calls
        calls += 1
        return {"decision": "needs_retry", "confidence": 0.0, "summary": "temporary ambiguity"}

    first = autonomy.resolve_deferred_duplicates_with_frontier(
        [_deferred_record()],
        convergence_store=store,
        budget=_frontier_budget(calls=1),
        reviewer=reviewer,
        now=NOW,
    )
    early = autonomy.resolve_deferred_duplicates_with_frontier(
        [_deferred_record()],
        convergence_store=store,
        budget=_frontier_budget(calls=1),
        reviewer=reviewer,
        now=NOW + timedelta(seconds=30),
    )
    exhausted = autonomy.resolve_deferred_duplicates_with_frontier(
        [_deferred_record()],
        convergence_store=store,
        budget=_frontier_budget(calls=1),
        reviewer=reviewer,
        now=NOW + timedelta(seconds=61),
    )

    assert first["status_counts"] == {"frontier_retry": 1}
    assert early["results"][0]["reason"] == "backoff"
    assert exhausted["status_counts"] == {"quarantined": 1}
    assert calls == 2


def test_frontier_duplicate_uses_narrow_human_boundary_and_cycle_budget(
    monkeypatch,
    tmp_path: Path,
) -> None:
    pages = tmp_path / "pages"
    pages.mkdir()
    for page_id in ("a", "b", "c", "d", "e", "f"):
        _write_page(pages / f"{page_id}.md", page_id.upper(), page_id)
    monkeypatch.setattr(autonomy, "find_page", lambda page_id: pages / f"{page_id}.md")
    monkeypatch.setattr(autonomy, "DECISIONS_FILE", tmp_path / "decisions.jsonl")
    store = _convergence_store(tmp_path)
    calls: list[str] = []

    def reviewer(candidate: dict) -> dict:
        calls.append(candidate["left"])
        if candidate["left"] == "a":
            return {
                "decision": "needs_retry",
                "confidence": 0.0,
                "summary": "login required",
                "frontier_failure": {"failure_class": "auth_required", "human_required": True},
            }
        return {
            "decision": "needs_retry",
            "confidence": 0.0,
            "summary": "model asks for a person",
            "human_required": True,
            "frontier_failure": {"failure_class": "model_uncertain", "human_required": True},
        }

    result = autonomy.resolve_deferred_duplicates_with_frontier(
        [_deferred_record("a", "b"), _deferred_record("c", "d"), _deferred_record("e", "f")],
        convergence_store=store,
        budget=_frontier_budget(calls=2),
        reviewer=reviewer,
        now=NOW,
    )

    assert calls == ["a", "c"]
    assert result["status_counts"] == {
        "frontier_retry": 1,
        "human_required": 1,
        "pending_frontier": 1,
    }
    budget_deferred = next(row for row in result["results"] if row.get("pair") == ["e", "f"])
    assert budget_deferred["reason"] == "frontier_budget_exhausted"


def test_frontier_duplicate_dry_run_creates_nothing_and_spends_no_budget(
    monkeypatch,
    tmp_path: Path,
) -> None:
    pages = tmp_path / "pages"
    pages.mkdir()
    _write_page(pages / "a.md", "Alpha", "A")
    _write_page(pages / "b.md", "Beta", "B")
    before = {path.name: path.read_bytes() for path in pages.iterdir()}
    monkeypatch.setattr(autonomy, "find_page", lambda page_id: pages / f"{page_id}.md")
    decisions = tmp_path / "decisions.jsonl"
    monkeypatch.setattr(autonomy, "DECISIONS_FILE", decisions)
    store = _convergence_store(tmp_path)
    budget = _frontier_budget(calls=1, mutations=1)

    result = autonomy.resolve_deferred_duplicates_with_frontier(
        [_deferred_record()],
        convergence_store=store,
        budget=budget,
        reviewer=lambda _candidate: (_ for _ in ()).throw(AssertionError("reviewer called")),
        now=NOW,
        dry_run=True,
    )

    assert result["status_counts"] == {"would_review": 1}
    assert budget.snapshot()["used"] == {"local": 0, "frontier": 0, "mutation": 0, "raw_bytes": 0}
    assert {path.name: path.read_bytes() for path in pages.iterdir()} == before
    assert not store.state_file.exists()
    assert not store.events_file.exists()
    assert not store.lock_file.exists()
    assert not decisions.exists()


def test_retention_archive_obeys_mutation_budget(monkeypatch, tmp_path: Path) -> None:
    page = tmp_path / "old.md"
    _write_page(page, "Old", "Body")
    monkeypatch.setattr(autonomy, "find_page", lambda _page_id: page)
    payload = {"archive_candidates": ["old"], "pages": {"old": {"score": 0.1}}}
    budget = _frontier_budget(mutations=0)

    result = autonomy.apply_retention_archives(
        payload,
        write=False,
        budget=budget,
        convergence_store=_convergence_store(tmp_path),
        reviewer=lambda _candidate: {
            "decision": "archive",
            "confidence": 0.99,
            "summary": "Obsolete",
        },
        now=NOW,
    )

    assert result["applied"] == 0
    assert result["decisions"][0]["status"] == "frontier_retry"
    meta, _body = parse_frontmatter(page.read_text(encoding="utf-8"))
    assert meta["status"] == "active"


def test_retention_skips_already_archived_without_starving_next_candidate(
    monkeypatch, tmp_path: Path
) -> None:
    archived = tmp_path / "archived.md"
    active = tmp_path / "active.md"
    _write_page(archived, "Archived", "Old")
    _write_page(active, "Active", "Old")
    archived.write_text(
        archived.read_text(encoding="utf-8").replace("status: active", "status: archived"),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        autonomy,
        "find_page",
        lambda page_id: {"archived": archived, "active": active}.get(page_id),
    )
    payload = {
        "archive_candidates": ["archived", "active"],
        "pages": {"archived": {"score": 0.1}, "active": {"score": 0.1}},
    }
    budget = _frontier_budget(mutations=1)

    result = autonomy.apply_retention_archives(
        payload,
        write=False,
        limit=1,
        budget=budget,
        convergence_store=_convergence_store(tmp_path),
        reviewer=lambda _candidate: {
            "decision": "archive",
            "confidence": 0.99,
            "summary": "Obsolete",
        },
        now=NOW,
    )

    assert result["applied"] == 1
    assert [row["action"] for row in result["decisions"]] == [
        "already_archived",
        "archive",
    ]
    active_meta, _body = parse_frontmatter(active.read_text(encoding="utf-8"))
    assert active_meta["status"] == "archived"
    assert budget.snapshot()["used"]["mutation"] == 1


def test_retention_frontier_rejection_keeps_page_active_and_is_cached(
    monkeypatch,
    tmp_path: Path,
) -> None:
    page = tmp_path / "useful.md"
    _write_page(page, "Useful", "Distinct source of truth")
    monkeypatch.setattr(autonomy, "find_page", lambda _page_id: page)
    store = _convergence_store(tmp_path)
    payload = {
        "archive_candidates": ["useful"],
        "pages": {"useful": {"score": 0.05}},
    }
    calls = 0

    def reviewer(_candidate: dict) -> dict:
        nonlocal calls
        calls += 1
        return {
            "decision": "keep_active",
            "confidence": 0.98,
            "summary": "The page contains a distinct current fact",
        }

    first = autonomy.apply_retention_archives(
        payload,
        write=False,
        convergence_store=store,
        reviewer=reviewer,
        now=NOW,
    )
    second = autonomy.apply_retention_archives(
        payload,
        write=False,
        convergence_store=store,
        reviewer=lambda _candidate: (_ for _ in ()).throw(
            AssertionError("terminal retention decision was not cached")
        ),
        now=NOW + timedelta(seconds=1),
    )

    assert first["status_counts"] == {"rejected": 1}
    assert second["status_counts"] == {"rejected": 1}
    assert calls == 1
    meta, body = parse_frontmatter(page.read_text(encoding="utf-8"))
    assert meta["status"] == "active"
    assert body == "Distinct source of truth\n"


def test_retention_reuses_durable_approval_after_mutation_budget_retry(
    monkeypatch,
    tmp_path: Path,
) -> None:
    page = tmp_path / "old.md"
    _write_page(page, "Old", "Redundant historical cache")
    monkeypatch.setattr(autonomy, "find_page", lambda _page_id: page)
    store = _convergence_store(tmp_path)
    payload = {"archive_candidates": ["old"], "pages": {"old": {"score": 0.01}}}
    calls = 0

    def reviewer(_candidate: dict) -> dict:
        nonlocal calls
        calls += 1
        return {"decision": "archive", "confidence": 0.99, "summary": "Redundant"}

    first = autonomy.apply_retention_archives(
        payload,
        write=False,
        budget=_frontier_budget(calls=1, mutations=0),
        convergence_store=store,
        reviewer=reviewer,
        now=NOW,
    )
    second = autonomy.apply_retention_archives(
        payload,
        write=False,
        budget=_frontier_budget(calls=1, mutations=1),
        convergence_store=store,
        reviewer=lambda _candidate: (_ for _ in ()).throw(
            AssertionError("durable retention approval was not reused")
        ),
        now=NOW + timedelta(seconds=61),
    )

    assert first["status_counts"] == {"frontier_retry": 1}
    assert second["status_counts"] == {"applied": 1}
    assert second["frontier_calls"] == 0
    assert calls == 1
    meta, body = parse_frontmatter(page.read_text(encoding="utf-8"))
    assert meta["status"] == "archived"
    assert meta["autonomy_decision"] == "retention_frontier_archive"
    assert meta["frontier_approval_key"].startswith("retention_frontier:")
    assert body == "Redundant historical cache\n"


def test_retention_recovers_convergence_after_crash_following_archive(
    monkeypatch,
    tmp_path: Path,
) -> None:
    page = tmp_path / "old.md"
    _write_page(page, "Old", "Redundant")
    monkeypatch.setattr(autonomy, "find_page", lambda _page_id: page)
    store = _convergence_store(tmp_path)
    real_complete = store.complete
    crashed = False

    def crash_after_write(key, status, **kwargs):
        nonlocal crashed
        if status == "applied" and not crashed:
            crashed = True
            raise RuntimeError("simulated retention crash")
        return real_complete(key, status, **kwargs)

    monkeypatch.setattr(store, "complete", crash_after_write)
    payload = {"archive_candidates": ["old"], "pages": {"old": {"score": 0.01}}}
    with pytest.raises(RuntimeError, match="simulated retention crash"):
        autonomy.apply_retention_archives(
            payload,
            write=False,
            convergence_store=store,
            reviewer=lambda _candidate: {
                "decision": "archive",
                "confidence": 0.99,
                "summary": "Redundant",
            },
            now=NOW,
        )

    meta, _body = parse_frontmatter(page.read_text(encoding="utf-8"))
    assert meta["status"] == "archived"
    monkeypatch.setattr(store, "complete", real_complete)
    recovered = autonomy.apply_retention_archives(
        payload,
        write=False,
        convergence_store=store,
        reviewer=lambda _candidate: (_ for _ in ()).throw(
            AssertionError("reviewer called during retention receipt recovery")
        ),
        now=NOW + timedelta(seconds=1),
    )

    assert recovered["decisions"][0]["action"] == "already_archived"
    assert recovered["decisions"][0]["convergence_status"] == "applied"
    item = store.list_items(lane=autonomy.RETENTION_FRONTIER_LANE)[0]
    assert item["status"] == "applied"


def test_retention_dry_run_is_byte_for_byte_and_state_read_only(
    monkeypatch,
    tmp_path: Path,
) -> None:
    page = tmp_path / "old.md"
    _write_page(page, "Old", "Body")
    before = page.read_bytes()
    monkeypatch.setattr(autonomy, "find_page", lambda _page_id: page)
    store = _convergence_store(tmp_path)
    budget = _frontier_budget(calls=1, mutations=1)

    result = autonomy.apply_retention_archives(
        {"archive_candidates": ["old"], "pages": {"old": {"score": 0.01}}},
        apply=False,
        write=False,
        budget=budget,
        convergence_store=store,
        reviewer=lambda _candidate: (_ for _ in ()).throw(
            AssertionError("reviewer called during dry run")
        ),
        now=NOW,
    )

    assert result["status_counts"] == {"would_review": 1}
    assert page.read_bytes() == before
    assert not store.state_file.exists()
    assert not store.events_file.exists()
    assert not store.lock_file.exists()
    assert budget.snapshot()["used"] == {
        "local": 0,
        "frontier": 0,
        "mutation": 0,
        "raw_bytes": 0,
    }


def test_page_status_patch_rejects_stale_snapshot(monkeypatch, tmp_path: Path) -> None:
    page = tmp_path / "old.md"
    _write_page(page, "Old", "Body")
    monkeypatch.setattr(autonomy, "find_page", lambda _page_id: page)

    result = autonomy._patch_page_status(
        "old", {"status": "archived"}, expected_hash="stale"
    )

    assert result["status"] == "retry"
    assert result["reason"] == "page_changed_before_apply"
    meta, _body = parse_frontmatter(page.read_text(encoding="utf-8"))
    assert meta["status"] == "active"


def test_watchdog_alerts_when_sleep_never_ran(monkeypatch) -> None:
    monkeypatch.setattr(
        "llm_wiki_mcp.health.health_snapshot",
        lambda: {
            "memory_integrity": {"capture_rate": 0.95},
            "queues": {"duplicate_candidates": 0, "lint_repair": 0},
        },
    )
    monkeypatch.setattr(autonomy, "_latest_jsonl", lambda path: {})
    writes: list[tuple[Path, dict]] = []
    history: list[dict] = []
    monkeypatch.setattr(autonomy, "_write_json", lambda path, payload: writes.append((path, payload)))
    monkeypatch.setattr(autonomy, "_write_watchdog_history", lambda payload: history.append(payload))

    payload = autonomy.watchdog_snapshot(write=True)

    assert payload["status"] == "alert"
    assert payload["alerts"][0]["type"] == "sleep_never_ran"
    assert writes[0][0] == autonomy.WATCHDOG_FILE
    assert history == [payload]


def test_watchdog_history_is_compact_and_bounded_to_1000_lines(monkeypatch, tmp_path: Path) -> None:
    history_file = tmp_path / "watchdog-history.jsonl"
    old_rows = [
        {
            "ts": f"2026-07-01T00:{index % 60:02d}:00",
            "status": "ok",
            "alerts": [],
            "health": {
                "memory_integrity": {"capture_rate": 0.9},
                "queues": {"duplicate_candidates": index, "lint_repair": index + 1},
                "huge": "x" * 100,
            },
            "latest_sleep": {"status": "ok", "started_at": "old", "full": "y" * 100},
        }
        for index in range(1005)
    ]
    history_file.write_text(
        "".join(json.dumps(row) + "\n" for row in old_rows),
        encoding="utf-8",
    )
    monkeypatch.setattr(autonomy, "WATCHDOG_HISTORY", history_file)
    monkeypatch.setattr(autonomy, "WATCHDOG_FILE", tmp_path / "watchdog-latest.json")
    monkeypatch.setattr(
        "llm_wiki_mcp.health.health_snapshot",
        lambda: {
            "memory_integrity": {"capture_rate": 0.96},
            "queues": {"duplicate_candidates": 7, "lint_repair": 9},
            "large_unneeded_section": {"blob": "z" * 5000},
        },
    )
    monkeypatch.setattr(
        autonomy,
        "_latest_jsonl",
        lambda path: {"status": "ok", "started_at": "2026-07-10T03:40:00", "payload": "q" * 5000},
    )

    payload = autonomy.watchdog_snapshot(write=True)

    rows = [json.loads(line) for line in history_file.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1000
    assert all("health" not in row and "alerts" not in row for row in rows)
    assert rows[-1]["capture_rate"] == 0.96
    assert rows[-1]["queues"] == {"duplicate_candidates": 7, "lint_repair": 9}
    assert rows[-1]["latest_sleep"] == {
        "status": "ok",
        "started_at": "2026-07-10T03:40:00",
        "finished_at": None,
        "run_id": None,
    }
    assert payload["latest_sleep"] == rows[-1]["latest_sleep"]
    assert "payload" not in payload["latest_sleep"]
    persisted = json.loads((tmp_path / "watchdog-latest.json").read_text(encoding="utf-8"))
    assert persisted["latest_sleep"] == rows[-1]["latest_sleep"]


def test_regression_guard_quarantines_cycle_without_global_git_reset(monkeypatch, tmp_path: Path) -> None:
    quarantine_file = tmp_path / "quarantine.json"
    decisions_file = tmp_path / "decisions.jsonl"
    monkeypatch.setattr(autonomy, "QUARANTINE_FILE", quarantine_file)
    monkeypatch.setattr(autonomy, "DECISIONS_FILE", decisions_file)
    monkeypatch.setattr(
        autonomy,
        "_git",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("global git reset attempted")),
    )

    payload = autonomy.regression_guard(
        before_health={"memory_integrity": {"capture_rate": 0.95}},
        after_watchdog={"alerts": [{"type": "capture_rate_regression", "before": 0.95, "after": 0.7}]},
        wiki_snapshot={"head": "abc123"},
        auto_revert=True,
        write=True,
    )

    assert payload["status"] == "regression"
    assert payload["reverted"] is False
    assert payload["global_reset_disabled"] is True
    assert payload["rollback_scope"] == "mutation_cas_only"
    assert "revert" not in payload
    quarantine = json.loads(quarantine_file.read_text(encoding="utf-8"))
    assert quarantine["actions"] == [payload["quarantine"]]
    assert quarantine["actions"][0]["scope"] == "cycle"

    before = quarantine_file.read_bytes()
    autonomy.regression_guard(
        before_health={},
        after_watchdog={"alerts": [{"type": "capture_rate_regression"}]},
        wiki_snapshot={"head": "def456"},
        auto_revert=True,
        write=False,
    )
    assert quarantine_file.read_bytes() == before


def test_install_launchd_dry_run_builds_sleep_and_watchdog_plists(monkeypatch) -> None:
    monkeypatch.setattr(autonomy, "_uv_path", lambda: "/opt/homebrew/bin/uv")

    payload = autonomy.install_launchd(dry_run=True, load=False)

    assert payload["status"] == "ok"
    assert payload["dry_run"] is True
    labels = {item["label"] for item in payload["plists"]}
    assert autonomy.SLEEP_LABEL in labels
    assert autonomy.WATCHDOG_LABEL in labels
    programs = {item["label"]: item["program"] for item in payload["plists"]}
    assert Path(programs[autonomy.SLEEP_LABEL][0]).name == "llm-wiki-sleep"
    assert Path(programs[autonomy.WATCHDOG_LABEL][0]).name == "llm-wiki-watchdog"
    watchdog_plist = next(
        item for item in payload["plists"] if item["label"] == autonomy.WATCHDOG_LABEL
    )
    assert watchdog_plist["stdout"] == os.devnull
    assert payload["wrappers"][0]["command"][0] == "/opt/homebrew/bin/uv"
    assert "--json" not in payload["wrappers"][0]["command"]
