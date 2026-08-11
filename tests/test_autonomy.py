from __future__ import annotations

import hashlib
import json
import os
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from chronovisor.core import canonical_document, index_store
from chronovisor.core.frontmatter import parse as parse_frontmatter
from chronovisor.decision.decision_router import canonical_agreement_signature
from chronovisor.decision.decision_schema_manifest import production_decision_schemas
from chronovisor.ingest.convergence import ConvergenceStore, CycleBudget, RetryPolicy
from chronovisor.ops import autonomy
from tests.semantic_hold_support import semantic_authority, semantic_review
from tests.test_decision_authority import _vote_audit

NOW = datetime(2026, 7, 10, 12, 0, tzinfo=UTC)


@pytest.fixture(autouse=True)
def isolate_decision_authority_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chronovisor.core import page_mutation, retention

    for name in ("index.md", "log.md", "schema.md"):
        (tmp_path / name).write_text("legacy\n", encoding="utf-8")

    monkeypatch.setattr(
        page_mutation,
        "DECISION_AUTHORITY_LOCK",
        tmp_path / "runtime" / "decision-authority.lock",
    )
    monkeypatch.setattr(retention, "CHRONOVISOR_ROOT", tmp_path)
    monkeypatch.setattr(autonomy, "CHRONOVISOR_ROOT", tmp_path)
    monkeypatch.setattr(autonomy, "PAGES_DIR", tmp_path)
    monkeypatch.setattr(autonomy, "SYSTEM_DIR", tmp_path / "system")
    monkeypatch.setattr(autonomy, "find_page", None, raising=False)
    monkeypatch.setattr(page_mutation, "CHRONOVISOR_ROOT", tmp_path)


def _write_page(path: Path, title: str, body: str) -> None:
    path.write_text(
        f"---\ntitle: {title}\nstatus: stable\ntype: knowledge\n"
        f"updated: 2026-07-10\n---\n{body}\n",
        encoding="utf-8",
    )


def _restore_page_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    page_id: str = "old",
) -> tuple[Path, index_store.IndexStore]:
    pages = tmp_path / "pages"
    system = tmp_path / "system"
    pages.mkdir(exist_ok=True)
    system.mkdir(exist_ok=True)
    page = pages / f"{page_id}.md"
    page.write_bytes(
        b"---\n"
        b"title: Old\n"
        b"status: deprecated\n"
        b"type: knowledge\n"
        b"updated: 2026-07-01\n"
        b"superseded_by: winner\n"
        b"autonomy_decision: duplicate_frontier_supersede\n"
        b"autonomy_decision_at: 2026-07-01T12:00:00\n"
        b"frontier_approval_key: approval\n"
        b"---\n"
        b"Body with exact trailing spaces  \r\n"
    )
    store = index_store.IndexStore(tmp_path)
    store.refresh()
    monkeypatch.setattr(autonomy, "PAGES_DIR", pages)
    monkeypatch.setattr(autonomy, "SYSTEM_DIR", system)
    monkeypatch.setattr(
        autonomy,
        "DECISIONS_FILE",
        tmp_path / "autonomy" / "decisions.jsonl",
    )
    monkeypatch.setattr(autonomy, "_now", lambda: "2026-08-11T12:00:00")
    monkeypatch.setattr(index_store, "get_store", lambda: store)
    return page, store


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


def _authority(lane: str, *, artifact_sha256: str = "a" * 64) -> dict:
    schema_name = (
        "duplicate_resolution"
        if lane == autonomy.DUPLICATE_FRONTIER_LANE
        else "retention"
    )
    return semantic_authority(
        lane,
        artifact_sha256=artifact_sha256,
        schema_name=schema_name,
    )


def _decision_policy(authority: dict) -> dict:
    return {
        **authority["policy"],
        "router_policy": authority["router"],
    }


def _local_consensus_proof(review: dict, authority: dict) -> dict:
    schema = production_decision_schemas()[authority["policy"]["schema_name"]]
    signature = canonical_agreement_signature(review, schema=schema)
    agreement = hashlib.sha256(signature.encode("utf-8")).hexdigest()
    routes = authority["router"]["routes"]
    return {
        "status": "agreed",
        "ok": True,
        "conservative_veto_fired": False,
        "conservative_veto_bypassed_by_lane_policy": False,
        "dissent_effect_class": None,
        "quorum_safety_policy_version": authority["quorum_safety_policy_version"],
        "agreement_sha256": agreement,
        "failure_class": None,
        "quarantine_reason": None,
        "num_ctx": 16_384,
        "residency": {},
        "votes": [
            _vote_audit("primary", routes[0], agreement),
            _vote_audit("challenger", routes[1], agreement),
        ],
    }


def _authority_bound_review(
    decision: str,
    confidence: float,
    summary: str,
    authority: dict,
) -> dict:
    review = {
        "decision": decision,
        "confidence": confidence,
        "summary": summary,
        "decision_policy": _decision_policy(authority),
    }
    review["local_consensus"] = _local_consensus_proof(review, authority)
    return review


def _semantic_no_quorum_review(lane: str, authority: dict) -> dict:
    review = semantic_review(authority, lane=lane)
    return {
        "decision": "needs_retry",
        "confidence": 0.0,
        "summary": review["summary"],
        "reviewer": review["reviewer"],
        "frontier_failure": review["frontier_failure"],
        "human_required": False,
        "decision_policy": review["decision_policy"],
        "local_consensus": review["local_consensus"],
    }


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


def test_duplicate_resolution_routes_exact_high_confidence_pair_without_mutation(
    monkeypatch, tmp_path: Path
) -> None:
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
    _write_page(tmp_path / "rich.md", "Same", "Rich")
    _write_page(tmp_path / "thin.md", "Same", "Thin")
    monkeypatch.setattr(autonomy, "_page_meta", lambda page_id: pages.get(page_id, {}))
    monkeypatch.setattr(
        autonomy, "find_page", lambda page_id: tmp_path / f"{page_id}.md"
    )
    monkeypatch.setattr("chronovisor.core.index_store.get_store", lambda: None)
    monkeypatch.setattr(
        autonomy,
        "_page_quality",
        lambda page_id, meta=None: 5.0 if page_id == "rich" else 1.0,
    )
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
    monkeypatch.setattr(
        autonomy, "find_page", lambda page_id: tmp_path / f"{page_id}.md"
    )
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
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("lifecycle writer called")
        ),
    )

    result = autonomy.resolve_duplicate_candidates([_exact_record()], write=False)

    assert result["applied"] == 0
    assert result["deferred"] == 1
    assert (
        result["decisions"][0]["result"]["reason"]
        == "deterministic_heuristic_is_proposal_only"
    )
    meta, body = parse_frontmatter((tmp_path / "b.md").read_text(encoding="utf-8"))
    assert meta["status"] == "stable"
    assert "superseded_by" not in meta
    assert body == "Loser\n"
    assert (tmp_path / "b.md").read_bytes() == before


def test_soft_supersede_preserves_correction_that_lands_before_locked_cas(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _write_page(tmp_path / "a.md", "Same", "Winner")
    _write_page(tmp_path / "b.md", "Same", "Loser")
    monkeypatch.setattr(
        autonomy, "find_page", lambda page_id: tmp_path / f"{page_id}.md"
    )
    loser_snapshot = autonomy._duplicate_page_snapshot("b")
    winner_snapshot = autonomy._duplicate_page_snapshot("a")
    loser = tmp_path / "b.md"
    corrected = loser.read_text(encoding="utf-8") + "user correction\n"

    @contextmanager
    def correction_wins():
        loser.write_text(corrected, encoding="utf-8")
        yield

    monkeypatch.setattr(autonomy, "chronovisor_mutation_lock", correction_wins)

    result = autonomy._soft_supersede_page(
        loser="b",
        winner="a",
        expected_loser_hash=loser_snapshot["content_hash"],
        expected_winner_hash=winner_snapshot["content_hash"],
        decision_at=NOW.isoformat(),
    )

    assert result == {"status": "retry", "reason": "content_changed_before_apply"}
    assert loser.read_text(encoding="utf-8") == corrected


def test_duplicate_frontier_never_supersedes_draft_loser(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    winner = tmp_path / "winner.md"
    loser = tmp_path / "loser.md"
    _write_page(winner, "Same", "Winner")
    _write_page(loser, "Same", "Loser")
    loser.write_text(
        loser.read_text(encoding="utf-8").replace("status: stable", "status: draft"),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        autonomy, "find_page", lambda page_id: {"winner": winner, "loser": loser}.get(page_id)
    )
    loser_hash = hashlib.sha256(loser.read_bytes()).hexdigest()
    winner_hash = hashlib.sha256(winner.read_bytes()).hexdigest()

    effect, error = autonomy._prepare_duplicate_effect_receipt(
        decision="supersede_left",
        left="loser",
        right="winner",
        page_hashes={"loser": loser_hash, "winner": winner_hash},
        decision_at=NOW.isoformat(),
        approval_key="approval",
    )
    result = autonomy._soft_supersede_page(
        loser="loser",
        winner="winner",
        expected_loser_hash=loser_hash,
        expected_winner_hash=winner_hash,
        decision_at=NOW.isoformat(),
    )

    assert effect is None
    assert error == "loser_is_not_stable"
    assert result == {"status": "retry", "reason": "loser_is_not_stable"}
    assert "status: draft" in loser.read_text(encoding="utf-8")


def test_lifecycle_writers_defer_pages_with_pending_content_correction(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _write_page(tmp_path / "winner.md", "Same", "Winner")
    _write_page(tmp_path / "loser.md", "Same", "Loser")
    _write_page(tmp_path / "old.md", "Old", "Archive candidate")
    monkeypatch.setattr(
        autonomy, "find_page", lambda page_id: tmp_path / f"{page_id}.md"
    )
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
    deprecate = autonomy.apply_retention_archives(
        {
            "deprecation_candidates": ["old"],
            "pages": {"old": {"score": 0.1}},
        },
        write=False,
        correction_store=store,
        reviewer=lambda _candidate: {
            "decision": "deprecate",
            "confidence": 0.99,
            "summary": "Obsolete duplicate",
        },
        now=NOW,
    )

    assert supersede["status"] == "retry"
    assert supersede["reason"] == "pending_content_correction"
    loser_meta, _ = parse_frontmatter(
        (tmp_path / "loser.md").read_text(encoding="utf-8")
    )
    old_meta, _ = parse_frontmatter((tmp_path / "old.md").read_text(encoding="utf-8"))
    assert loser_meta["status"] == "stable"
    assert old_meta["status"] == "stable"
    assert deprecate["applied"] == 0
    assert deprecate["decisions"][0]["reason"] == "pending_content_correction"


def test_restore_deprecated_page_preserves_body_refreshes_index_and_audits(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    page, store = _restore_page_fixture(tmp_path, monkeypatch)
    original = page.read_bytes()
    original_document = canonical_document.parse_document(original)
    expected_sha256 = hashlib.sha256(original).hexdigest()

    result = autonomy.restore_deprecated_page(
        "old",
        expected_sha256,
        "approved mistaken deprecation",
    )

    restored = canonical_document.parse_document(page.read_bytes())
    assert result["status"] == "applied"
    assert result["sha256"] == hashlib.sha256(page.read_bytes()).hexdigest()
    assert restored.body == original_document.body
    assert restored.metadata["status"] == "stable"
    assert restored.metadata["updated"] == "2026-08-11"
    assert not {
        "superseded_by",
        "autonomy_decision",
        "autonomy_decision_at",
        "frontier_approval_key",
    }.intersection(restored.metadata)
    assert store.meta("old")["status"] == "stable"
    rows = [
        json.loads(line)
        for line in autonomy.DECISIONS_FILE.read_text(encoding="utf-8").splitlines()
    ]
    assert [row["phase"] for row in rows] == ["intent", "applied"]
    assert rows[0]["expected_sha256"] == expected_sha256
    assert rows[0]["post_sha256"] == result["sha256"]
    assert rows[0]["reason"] == "approved mistaken deprecation"


def test_restore_deprecated_page_rejects_stale_system_reserved_and_stable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    page, _store = _restore_page_fixture(tmp_path, monkeypatch)
    before = page.read_bytes()

    stale = autonomy.restore_deprecated_page(
        "old",
        "0" * 64,
        "stale request",
    )
    assert stale["reason"] == "expected_sha256_mismatch"
    assert page.read_bytes() == before

    system_page = tmp_path / "system" / "system-only.md"
    system_page.write_bytes(before)
    system_result = autonomy.restore_deprecated_page(
        "system-only",
        hashlib.sha256(before).hexdigest(),
        "must stay system",
    )
    assert system_result["reason"] == "page_not_found_or_not_pages"
    assert system_page.read_bytes() == before

    reserved = tmp_path / "pages" / "index.md"
    reserved.write_bytes(before)
    reserved_result = autonomy.restore_deprecated_page(
        "index",
        hashlib.sha256(before).hexdigest(),
        "must stay reserved",
    )
    assert reserved_result["reason"] == "page_not_found_or_not_pages"
    assert reserved.read_bytes() == before

    stable = tmp_path / "pages" / "stable.md"
    stable.write_bytes(before.replace(b"status: deprecated", b"status: stable"))
    stable_before = stable.read_bytes()
    stable_result = autonomy.restore_deprecated_page(
        "stable",
        hashlib.sha256(stable_before).hexdigest(),
        "already stable",
    )
    assert stable_result["reason"] == "page_is_not_deprecated"
    assert stable.read_bytes() == stable_before
    assert not autonomy.DECISIONS_FILE.exists()


def test_restore_deprecated_page_rejects_pending_content_correction(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    page, _store = _restore_page_fixture(tmp_path, monkeypatch)
    before = page.read_bytes()
    correction_store = autonomy._content_correction_store_for_page(page)
    correction_store.merge_item(
        lane=autonomy.CONTENT_CORRECTION_LANE,
        source_id="correction",
        input_data={"claim": "pending"},
        metadata={"candidate_pages": ["old"]},
    )

    result = autonomy.restore_deprecated_page(
        "old",
        hashlib.sha256(before).hexdigest(),
        "restore after correction",
    )

    assert result["reason"] == "pending_content_correction"
    assert page.read_bytes() == before
    assert not autonomy.DECISIONS_FILE.exists()


def test_restore_deprecated_page_recovers_intent_only_postimage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    page, store = _restore_page_fixture(tmp_path, monkeypatch)
    expected_sha256 = hashlib.sha256(page.read_bytes()).hexdigest()
    real_append = autonomy.append_jsonl_durable
    calls = 0

    def crash_before_terminal(path, rows, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated terminal audit crash")
        real_append(path, rows, **kwargs)

    monkeypatch.setattr(autonomy, "append_jsonl_durable", crash_before_terminal)
    with pytest.raises(RuntimeError, match="simulated terminal audit crash"):
        autonomy.restore_deprecated_page(
            "old",
            expected_sha256,
            "recover exact postimage",
        )

    assert canonical_document.parse_document(page.read_bytes()).metadata["status"] == (
        "stable"
    )
    monkeypatch.setattr(autonomy, "append_jsonl_durable", real_append)
    recovered = autonomy.restore_deprecated_page(
        "old",
        expected_sha256,
        "recover exact postimage",
    )
    repeated = autonomy.restore_deprecated_page(
        "old",
        expected_sha256,
        "recover exact postimage",
    )

    assert recovered["status"] == "recovered"
    assert repeated["status"] == "recovered"
    assert store.meta("old")["status"] == "stable"
    rows = [
        json.loads(line)
        for line in autonomy.DECISIONS_FILE.read_text(encoding="utf-8").splitlines()
    ]
    assert [row["phase"] for row in rows] == ["intent", "recovered"]


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
    monkeypatch.setattr(autonomy, "chronovisor_mutation_lock", tracked_wiki_lock)

    with autonomy._lifecycle_mutation_guard(
        ["old"], page_path=page, correction_store=store
    ) as guard:
        assert guard["allowed"] is True

    assert order == ["state-enter", "wiki-enter", "wiki-exit", "state-exit"]


def test_autonomy_temp_names_are_unique_for_same_page_and_preimage(
    tmp_path: Path,
) -> None:
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
    monkeypatch.setattr(
        autonomy, "find_page", lambda page_id: tmp_path / f"{page_id}.md"
    )
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
    monkeypatch.setattr(
        autonomy, "find_page", lambda page_id: tmp_path / f"{page_id}.md"
    )
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
    assert (
        (tmp_path / "a.md")
        .read_text(encoding="utf-8")
        .endswith("concurrent winner update\n")
    )


def test_soft_supersede_does_not_rollback_over_a_foreign_postwrite_change(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _write_page(tmp_path / "a.md", "Alpha", "Winner")
    _write_page(tmp_path / "b.md", "Beta", "Loser")
    monkeypatch.setattr(
        autonomy, "find_page", lambda page_id: tmp_path / f"{page_id}.md"
    )
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
    assert (
        (tmp_path / "b.md")
        .read_text(encoding="utf-8")
        .endswith("foreign loser update\n")
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
    assert meta["frontier_approval_key"].startswith("autonomy_duplicate_resolution:")
    assert body == "Loser\n"


def test_duplicate_complete_inventory_retires_absent_pending_pair(tmp_path: Path) -> None:
    store = _convergence_store(tmp_path)
    merged = store.merge_item(
        lane=autonomy.DUPLICATE_FRONTIER_LANE,
        source_id="old-left<->old-right",
        input_data={"pair": ["old-left", "old-right"]},
        resolver_version=autonomy.DUPLICATE_FRONTIER_RESOLVER_VERSION,
    )
    key = merged["item"]["key"]

    partial = autonomy.resolve_deferred_duplicates_with_frontier(
        [],
        convergence_store=store,
        inventory_complete=False,
        now=NOW,
    )
    assert partial["retired_absent"] == []
    assert store.get(key)["status"] == "pending_local"

    complete = autonomy.resolve_deferred_duplicates_with_frontier(
        [],
        convergence_store=store,
        inventory_complete=True,
        now=NOW,
    )
    assert complete["retired_absent"] == [key]
    assert complete["retired"] == [key]
    assert store.get(key)["status"] == "rejected"


def test_duplicate_stale_approval_cannot_cross_adoption_epoch(
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
    first = autonomy.resolve_deferred_duplicates_with_frontier(
        [_exact_record()],
        convergence_store=store,
        budget=_frontier_budget(calls=1, mutations=0),
        reviewer=lambda _candidate: {
            "decision": "supersede_right",
            "confidence": 0.99,
            "summary": "A subsumes B",
        },
        now=NOW,
        write=False,
    )
    assert first["status_counts"] == {"frontier_retry": 1}

    approval_path = next((store.state_file.parent / "approvals").rglob("*.json"))
    approval = json.loads(approval_path.read_text(encoding="utf-8"))
    old_authority = _authority(autonomy.DUPLICATE_FRONTIER_LANE)
    approval["authority"] = old_authority
    approval["review"]["decision_policy"] = _decision_policy(old_authority)
    approval_path.write_text(json.dumps(approval), encoding="utf-8")
    current_authority = _authority(
        autonomy.DUPLICATE_FRONTIER_LANE,
        artifact_sha256="e" * 64,
    )
    monkeypatch.setattr(
        autonomy,
        "_current_autonomy_authority",
        lambda *_args, **_kwargs: (current_authority, None),
    )

    second = autonomy.resolve_deferred_duplicates_with_frontier(
        [_exact_record()],
        convergence_store=store,
        budget=_frontier_budget(calls=1, mutations=1),
        reviewer=lambda _candidate: (_ for _ in ()).throw(
            AssertionError("stale approval must not be resampled")
        ),
        now=NOW + timedelta(seconds=61),
        write=False,
    )

    assert second["status_counts"] == {"frontier_retry": 1}
    assert "authority changed" in second["results"][0]["reason"]
    assert second["frontier_calls"] == 0
    assert (pages / "b.md").read_bytes() == before


def test_duplicate_effect_revalidates_authority_inside_lock(
    monkeypatch,
    tmp_path: Path,
) -> None:
    pages = tmp_path / "pages"
    pages.mkdir()
    _write_page(pages / "a.md", "Alpha", "Winner")
    _write_page(pages / "b.md", "Beta", "Loser")
    before = (pages / "b.md").read_bytes()
    monkeypatch.setattr(autonomy, "find_page", lambda page_id: pages / f"{page_id}.md")
    store = _convergence_store(tmp_path)
    initial_authority = _authority(autonomy.DUPLICATE_FRONTIER_LANE)
    changed_authority = _authority(
        autonomy.DUPLICATE_FRONTIER_LANE,
        artifact_sha256="e" * 64,
    )
    effect_lock = False
    lock_entries = 0

    @contextmanager
    def authority_lock():
        nonlocal effect_lock, lock_entries
        lock_entries += 1
        effect_lock = lock_entries >= 2
        try:
            yield
        finally:
            effect_lock = False

    monkeypatch.setattr(autonomy, "decision_authority_lock", authority_lock)
    monkeypatch.setattr(
        autonomy,
        "_current_autonomy_authority",
        lambda *_args, **_kwargs: (
            changed_authority if effect_lock else initial_authority,
            None,
        ),
    )
    result = autonomy.resolve_deferred_duplicates_with_frontier(
        [_deferred_record()],
        convergence_store=store,
        budget=_frontier_budget(calls=1, mutations=1),
        reviewer=lambda _candidate: _authority_bound_review(
            "supersede_right",
            0.99,
            "A subsumes B",
            initial_authority,
        ),
        now=NOW,
        write=False,
    )

    assert lock_entries == 2
    assert result["status_counts"] == {"frontier_retry": 1}
    assert "authority changed" in result["results"][0]["apply"]["reason"]
    assert (pages / "b.md").read_bytes() == before


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
    assert recovered["results"][0]["recovery_only"] is True
    assert recovered["results"][0]["semantic_effect"] is False
    item = store.list_items(lane=autonomy.DUPLICATE_FRONTIER_LANE)[0]
    assert item["status"] == "applied"


def test_duplicate_does_not_recover_from_frontmatter_only_postimage_match(
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

    def crash_after_write(key, status, **kwargs):
        if status == "applied":
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

    approval_path = next((store.state_file.parent / "approvals").rglob("*.json"))
    approval = json.loads(approval_path.read_text(encoding="utf-8"))
    exact = (pages / "b.md").read_bytes()
    assert approval["effect_receipt"]["postimages"]["b"] == {
        "sha256": hashlib.sha256(exact).hexdigest(),
        "size_bytes": len(exact),
    }
    # Preserve the approved lifecycle frontmatter while concurrently changing
    # the body. A field-only recovery check would incorrectly terminalize it.
    (pages / "b.md").write_bytes(exact + b"concurrent body change\n")

    monkeypatch.setattr(store, "complete", real_complete)
    result = autonomy.resolve_deferred_duplicates_with_frontier(
        [_deferred_record()],
        convergence_store=store,
        budget=_frontier_budget(calls=0, mutations=0),
        reviewer=lambda _candidate: (_ for _ in ()).throw(
            AssertionError("reviewer called during non-exact recovery")
        ),
        now=NOW + timedelta(seconds=31),
        write=False,
    )

    assert result["status_counts"] == {"already_applied": 1}
    assert "convergence_status" not in result["results"][0]
    item = store.list_items(lane=autonomy.DUPLICATE_FRONTIER_LANE)[0]
    assert item["status"] != "applied"


def test_postimage_recovery_requires_every_mutated_page(
    monkeypatch, tmp_path: Path
) -> None:
    first = tmp_path / "first.md"
    second = tmp_path / "second.md"
    first.write_bytes(b"first exact\n")
    second.write_bytes(b"second changed\n")
    monkeypatch.setattr(
        autonomy,
        "find_page",
        lambda page_id: {"first": first, "second": second}.get(page_id),
    )
    receipt = {
        "postimages": {
            "first": autonomy._bytes_receipt(first.read_bytes()),
            "second": autonomy._bytes_receipt(b"second reviewed\n"),
        }
    }

    assert autonomy._current_pages_match_postimages(receipt) is False


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
        reviewer=lambda _candidate: (_ for _ in ()).throw(
            AssertionError("reviewer called")
        ),
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
        return {
            "decision": "supersede_right",
            "confidence": 0.97,
            "summary": "A subsumes B",
        }

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
    assert (
        len((tmp_path / "decisions.jsonl").read_text(encoding="utf-8").splitlines())
        == 1
    )


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
        return {
            "decision": "keep_both",
            "confidence": 0.91,
            "summary": "Distinct scope",
        }

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


def test_typed_yaml_snapshot_review_boundaries_build_both_prompts(
    tmp_path: Path,
) -> None:
    from chronovisor.decision.decision_lane_prompts import (
        build_autonomy_duplicate_review_prompt,
        build_autonomy_retention_review_prompt,
    )

    def write_typed(page_id: str, title: str, body: str) -> None:
        (tmp_path / f"{page_id}.md").write_text(
            "---\n"
            f"title: {title}\n"
            "status: stable\n"
            "type: knowledge\n"
            "updated: 2026-08-11\n"
            "features: !!set\n"
            "  ? gamma\n"
            "  ? alpha\n"
            "  ? beta\n"
            "---\n"
            f"{body}\n",
            encoding="utf-8",
        )

    write_typed("a", "Alpha", "Alpha evidence")
    write_typed("b", "Beta", "Beta evidence")
    write_typed("retained", "Retained", "Distinct current fact")
    prompt_hashes: dict[str, str] = {}

    def duplicate_reviewer(candidate: dict) -> dict:
        assert candidate["left_meta"]["kind"] == "canonical_yaml"
        assert candidate["right_meta"]["kind"] == "canonical_yaml"
        prompt = build_autonomy_duplicate_review_prompt(candidate)
        prompt_hashes["duplicate"] = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        return {
            "decision": "keep_both",
            "confidence": 0.99,
            "summary": "Distinct typed pages",
        }

    duplicate = autonomy.resolve_deferred_duplicates_with_frontier(
        [_deferred_record("a", "b")],
        convergence_store=_convergence_store(tmp_path / "duplicate-store"),
        budget=_frontier_budget(calls=1),
        reviewer=duplicate_reviewer,
        now=NOW,
    )

    def retention_reviewer(candidate: dict) -> dict:
        assert candidate["meta"]["kind"] == "canonical_yaml"
        prompt = build_autonomy_retention_review_prompt(candidate)
        prompt_hashes["retention"] = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        return {
            "decision": "keep_stable",
            "confidence": 0.99,
            "summary": "Distinct current fact",
        }

    retention = autonomy.apply_retention_archives(
        {
            "deprecation_candidates": ["retained"],
            "pages": {"retained": {"score": 0.05, "distinct_event": True}},
        },
        write=False,
        convergence_store=_convergence_store(tmp_path / "retention-store"),
        reviewer=retention_reviewer,
        now=NOW,
    )

    assert duplicate["status_counts"] == {"rejected": 1}
    assert retention["status_counts"] == {"rejected": 1}
    assert set(prompt_hashes) == {"duplicate", "retention"}
    assert all(len(value) == 64 for value in prompt_hashes.values())


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
        return {
            "decision": "supersede_right",
            "confidence": 0.99,
            "summary": "A subsumes B",
        }

    result = autonomy.resolve_deferred_duplicates_with_frontier(
        [_deferred_record()],
        convergence_store=store,
        budget=_frontier_budget(calls=1, mutations=1),
        reviewer=reviewer,
        now=NOW,
    )

    assert result["status_counts"] == {"frontier_retry": 1}
    meta, body = parse_frontmatter((pages / "b.md").read_text(encoding="utf-8"))
    assert meta["status"] == "stable"
    assert "superseded_by" not in meta
    assert body.endswith("concurrent update\n")


def test_frontier_duplicate_decision_is_not_overridden_by_confidence_metadata(
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

    assert result["status_counts"] == {"applied": 1}
    assert result["kept_both"] == 0
    assert result["results"][0]["decision"] == "supersede_right"
    assert budget.snapshot()["used"]["mutation"] == 1
    assert {path.name: path.read_bytes() for path in pages.iterdir()} != before


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


def test_autonomy_normalizers_preserve_production_authority_audits() -> None:
    duplicate_authority = _authority(autonomy.DUPLICATE_FRONTIER_LANE)
    retention_authority = _authority(autonomy.RETENTION_FRONTIER_LANE)
    duplicate = autonomy._normalize_duplicate_frontier_review(
        _authority_bound_review(
            "keep_both",
            0.95,
            "Distinct pages",
            duplicate_authority,
        )
    )
    retention = autonomy._normalize_retention_frontier_review(
        _authority_bound_review(
            "keep_stable",
            0.95,
            "Still current",
            retention_authority,
        )
    )

    assert duplicate["decision_policy"] == _decision_policy(duplicate_authority)
    assert duplicate["local_consensus"] == _local_consensus_proof(
        duplicate, duplicate_authority
    )
    assert retention["decision_policy"] == _decision_policy(retention_authority)
    assert retention["local_consensus"] == _local_consensus_proof(
        retention, retention_authority
    )


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
        return {
            "decision": "needs_retry",
            "confidence": 0.0,
            "summary": "temporary ambiguity",
        }

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


def test_duplicate_no_quorum_holds_exact_epoch_until_authority_changes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pages = tmp_path / "pages"
    pages.mkdir()
    _write_page(pages / "a.md", "Alpha", "A")
    _write_page(pages / "b.md", "Beta", "B")
    monkeypatch.setattr(autonomy, "find_page", lambda page_id: pages / f"{page_id}.md")
    store = _convergence_store(tmp_path)
    authority_a = semantic_authority(
        autonomy.DUPLICATE_FRONTIER_LANE,
        schema_name="duplicate_resolution",
    )
    authority_b = semantic_authority(
        autonomy.DUPLICATE_FRONTIER_LANE,
        schema_name="duplicate_resolution",
        artifact_sha256="9" * 64,
    )
    current = [authority_a]
    monkeypatch.setattr(
        autonomy,
        "_current_autonomy_authority",
        lambda *_args, **_kwargs: (current[0], None),
    )
    calls = 0

    def reviewer(_candidate: dict) -> dict:
        nonlocal calls
        calls += 1
        return _semantic_no_quorum_review(
            autonomy.DUPLICATE_FRONTIER_LANE,
            current[0],
        )

    first = autonomy.resolve_deferred_duplicates_with_frontier(
        [_deferred_record()],
        convergence_store=store,
        budget=_frontier_budget(calls=1),
        reviewer=reviewer,
        now=NOW,
    )
    same_epoch = autonomy.resolve_deferred_duplicates_with_frontier(
        [_deferred_record()],
        convergence_store=store,
        budget=_frontier_budget(calls=1),
        reviewer=reviewer,
        now=NOW + timedelta(seconds=1),
    )
    current[0] = authority_b
    changed_authority = autonomy.resolve_deferred_duplicates_with_frontier(
        [_deferred_record()],
        convergence_store=store,
        budget=_frontier_budget(calls=1),
        reviewer=reviewer,
        now=NOW + timedelta(seconds=2),
    )
    current[0] = authority_a
    restored_authority = autonomy.resolve_deferred_duplicates_with_frontier(
        [_deferred_record()],
        convergence_store=store,
        budget=_frontier_budget(calls=1),
        reviewer=lambda _candidate: (_ for _ in ()).throw(
            AssertionError("A-B-A must restore the A hold without resampling")
        ),
        now=NOW + timedelta(seconds=3),
    )

    assert first["status_counts"] == {"quarantined": 1}
    assert same_epoch["results"][0]["semantic_deferred"] is True
    assert same_epoch["frontier_calls"] == 0
    assert changed_authority["status_counts"] == {"quarantined": 1}
    assert changed_authority["frontier_calls"] == 1
    assert restored_authority["status_counts"] == {"quarantined": 1}
    assert restored_authority["results"][0]["restored_semantic_hold"] is True
    assert calls == 2
    item = store.list_items(lane=autonomy.DUPLICATE_FRONTIER_LANE)[0]
    assert item["frontier_attempts"] == 1
    assert item["result"]["semantic_hold"]["authority"] == authority_a
    assert [hold["authority"] for hold in item["result"]["semantic_hold_history"]] == [
        authority_b
    ]


def test_legacy_duplicate_no_quorum_migrates_without_resampling(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pages = tmp_path / "pages"
    pages.mkdir()
    _write_page(pages / "a.md", "Alpha", "A")
    _write_page(pages / "b.md", "Beta", "B")
    monkeypatch.setattr(autonomy, "find_page", lambda page_id: pages / f"{page_id}.md")
    store = _convergence_store(tmp_path)
    record = _deferred_record()
    candidate = autonomy._canonical_duplicate_record(record)
    assert candidate is not None
    input_data = {
        "pair": ["a", "b"],
        "content_hashes": {
            page_id: autonomy._duplicate_page_snapshot(page_id)["content_hash"]
            for page_id in ("a", "b")
        },
    }
    item = store.merge_item(
        lane=autonomy.DUPLICATE_FRONTIER_LANE,
        source_id="a<->b",
        input_data=input_data,
        resolver_version=autonomy.DUPLICATE_FRONTIER_RESOLVER_VERSION,
        metadata={"candidate": candidate},
        now=NOW,
    )["item"]
    state = store.load()
    state["items"][item["key"]].update(
        {
            "status": "quarantined",
            "frontier_attempts": 3,
            "quarantine_reason": "retry_exhausted:frontier",
            "last_failure_class": "local_semantic_no_quorum",
            "last_error": "mutating_local_majority_vetoed_by_conservative_vote",
        }
    )
    store.state_file.write_text(json.dumps(state), encoding="utf-8")
    calls = 0

    def reviewer(_candidate: dict) -> dict:
        nonlocal calls
        calls += 1
        return {
            "decision": "needs_retry",
            "confidence": 0.0,
            "summary": "operational retry after evidence change",
        }

    state_before = store.state_file.read_bytes()
    events_before = store.events_file.read_bytes()
    preview = autonomy.resolve_deferred_duplicates_with_frontier(
        [record],
        convergence_store=store,
        budget=_frontier_budget(calls=1),
        reviewer=reviewer,
        now=NOW + timedelta(milliseconds=500),
        dry_run=True,
    )
    assert preview["results"][0]["semantic_deferred"] is True
    assert store.state_file.read_bytes() == state_before
    assert store.events_file.read_bytes() == events_before
    migrated = autonomy.resolve_deferred_duplicates_with_frontier(
        [record],
        convergence_store=store,
        budget=_frontier_budget(calls=1),
        reviewer=reviewer,
        now=NOW + timedelta(seconds=1),
    )
    repeated = autonomy.resolve_deferred_duplicates_with_frontier(
        [record],
        convergence_store=store,
        budget=_frontier_budget(calls=1),
        reviewer=reviewer,
        now=NOW + timedelta(seconds=2),
    )

    assert calls == 0
    assert store.state_file.read_bytes() != state_before
    assert migrated["results"][0]["semantic_deferred"] is True
    assert repeated["results"][0]["semantic_deferred"] is True
    migrated_item = store.get(item["key"])
    assert migrated_item["quarantine_reason"] == (
        f"semantic_no_quorum_legacy:{autonomy.DUPLICATE_FRONTIER_LANE}"
    )
    assert migrated_item["result"]["legacy_semantic_hold"]["kind"] == (
        autonomy.LEGACY_SEMANTIC_HOLD_KIND
    )

    _write_page(pages / "b.md", "Beta", "B changed")
    changed = autonomy.resolve_deferred_duplicates_with_frontier(
        [record],
        convergence_store=store,
        budget=_frontier_budget(calls=1),
        reviewer=reviewer,
        now=NOW + timedelta(seconds=3),
    )
    assert calls == 1
    assert changed["status_counts"] == {"frontier_retry": 1}


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
                "frontier_failure": {
                    "failure_class": "auth_required",
                    "human_required": True,
                },
            }
        return {
            "decision": "needs_retry",
            "confidence": 0.0,
            "summary": "model asks for a person",
            "human_required": True,
            "frontier_failure": {
                "failure_class": "model_uncertain",
                "human_required": True,
            },
        }

    result = autonomy.resolve_deferred_duplicates_with_frontier(
        [
            _deferred_record("a", "b"),
            _deferred_record("c", "d"),
            _deferred_record("e", "f"),
        ],
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
    budget_deferred = next(
        row for row in result["results"] if row.get("pair") == ["e", "f"]
    )
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
        reviewer=lambda _candidate: (_ for _ in ()).throw(
            AssertionError("reviewer called")
        ),
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
    assert not decisions.exists()


def test_retention_archive_obeys_mutation_budget(monkeypatch, tmp_path: Path) -> None:
    page = tmp_path / "old.md"
    _write_page(page, "Old", "Body")
    monkeypatch.setattr(autonomy, "find_page", lambda _page_id: page)
    payload = {"deprecation_candidates": ["old"], "pages": {"old": {"score": 0.1}}}
    budget = _frontier_budget(mutations=0)

    result = autonomy.apply_retention_archives(
        payload,
        write=False,
        budget=budget,
        convergence_store=_convergence_store(tmp_path),
        reviewer=lambda _candidate: {
            "decision": "deprecate",
            "confidence": 0.99,
            "summary": "Obsolete",
        },
        now=NOW,
    )

    assert result["applied"] == 0
    assert result["decisions"][0]["status"] == "frontier_retry"
    meta, _body = parse_frontmatter(page.read_text(encoding="utf-8"))
    assert meta["status"] == "stable"


def test_retention_frontier_never_deprecates_draft_page(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    page = tmp_path / "old.md"
    _write_page(page, "Old", "Body")
    page.write_text(
        page.read_text(encoding="utf-8").replace("status: stable", "status: draft"),
        encoding="utf-8",
    )
    monkeypatch.setattr(autonomy, "find_page", lambda _page_id: page)
    page_hash = hashlib.sha256(page.read_bytes()).hexdigest()

    effect, error = autonomy._prepare_retention_effect_receipt(
        decision="deprecate",
        page_id="old",
        page_hash=page_hash,
        decision_at=NOW.isoformat(),
        approval_key="approval",
    )
    patched = autonomy._patch_page_status(
        "old", {"status": "deprecated"}, expected_hash=page_hash
    )
    result = autonomy.apply_retention_archives(
        {"deprecation_candidates": ["old"], "pages": {"old": {"score": 0.1}}},
        write=False,
        convergence_store=_convergence_store(tmp_path),
        reviewer=lambda _candidate: pytest.fail("draft page must not be reviewed"),
        now=NOW,
    )

    assert effect is None
    assert error == "page_is_not_stable"
    assert patched == {
        "status": "retry",
        "reason": "page_is_not_stable",
        "page_id": "old",
    }
    assert result["applied"] == 0
    assert result["decisions"][0]["reason"] == "deprecation_page_is_not_stable"
    assert "status: draft" in page.read_text(encoding="utf-8")


def test_retention_skips_already_deprecated_without_starving_next_candidate(
    monkeypatch, tmp_path: Path
) -> None:
    deprecated = tmp_path / "deprecated.md"
    stable = tmp_path / "stable.md"
    _write_page(deprecated, "Archived", "Old")
    _write_page(stable, "Active", "Old")
    deprecated.write_text(
        deprecated.read_text(encoding="utf-8").replace(
            "status: stable", "status: deprecated"
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        autonomy,
        "find_page",
        lambda page_id: {"deprecated": deprecated, "stable": stable}.get(page_id),
    )
    payload = {
        "deprecation_candidates": ["deprecated", "stable"],
        "pages": {"deprecated": {"score": 0.1}, "stable": {"score": 0.1}},
    }
    budget = _frontier_budget(mutations=1)

    result = autonomy.apply_retention_archives(
        payload,
        write=False,
        limit=1,
        budget=budget,
        convergence_store=_convergence_store(tmp_path),
        reviewer=lambda _candidate: {
            "decision": "deprecate",
            "confidence": 0.99,
            "summary": "Obsolete",
        },
        now=NOW,
    )

    assert result["applied"] == 1
    assert [row["action"] for row in result["decisions"]] == [
        "already_deprecated",
        "deprecate",
    ]
    active_meta, _body = parse_frontmatter(stable.read_text(encoding="utf-8"))
    assert active_meta["status"] == "deprecated"
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
        "deprecation_candidates": ["useful"],
        "pages": {"useful": {"score": 0.05}},
    }
    calls = 0

    def reviewer(_candidate: dict) -> dict:
        nonlocal calls
        calls += 1
        return {
            "decision": "keep_stable",
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
    assert meta["status"] == "stable"
    assert body == "Distinct source of truth\n"


def test_retention_no_quorum_is_terminal_and_reused_without_model_call(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    page = tmp_path / "old.md"
    _write_page(page, "Old", "Still semantically ambiguous")
    monkeypatch.setattr(autonomy, "find_page", lambda _page_id: page)
    store = _convergence_store(tmp_path)
    payload = {"deprecation_candidates": ["old"], "pages": {"old": {"score": 0.01}}}
    authority = semantic_authority(
        autonomy.RETENTION_FRONTIER_LANE,
        schema_name="retention",
    )
    monkeypatch.setattr(
        autonomy,
        "_current_autonomy_authority",
        lambda *_args, **_kwargs: (authority, None),
    )
    calls = 0

    def reviewer(_candidate: dict) -> dict:
        nonlocal calls
        calls += 1
        return _semantic_no_quorum_review(
            autonomy.RETENTION_FRONTIER_LANE,
            authority,
        )

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
        reviewer=reviewer,
        now=NOW + timedelta(seconds=1),
    )

    assert first["status_counts"] == {"quarantined": 1}
    assert second["decisions"][0]["semantic_deferred"] is True
    assert second["frontier_calls"] == 0
    assert calls == 1
    assert (
        store.list_items(lane=autonomy.RETENTION_FRONTIER_LANE)[0]["frontier_attempts"]
        == 1
    )


def test_retention_reuses_durable_approval_after_mutation_budget_retry(
    monkeypatch,
    tmp_path: Path,
) -> None:
    page = tmp_path / "old.md"
    _write_page(page, "Old", "Redundant historical cache")
    monkeypatch.setattr(autonomy, "find_page", lambda _page_id: page)
    store = _convergence_store(tmp_path)
    payload = {"deprecation_candidates": ["old"], "pages": {"old": {"score": 0.01}}}
    calls = 0

    def reviewer(_candidate: dict) -> dict:
        nonlocal calls
        calls += 1
        return {"decision": "deprecate", "confidence": 0.01, "summary": "Redundant"}

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
    assert meta["status"] == "deprecated"
    assert meta["autonomy_decision"] == "retention_frontier_deprecate"
    assert meta["frontier_approval_key"].startswith("autonomy_retention:")
    assert body == "Redundant historical cache\n"


def test_retention_effect_revalidates_authority_inside_lock(
    monkeypatch,
    tmp_path: Path,
) -> None:
    page = tmp_path / "old.md"
    _write_page(page, "Old", "Redundant historical cache")
    before = page.read_bytes()
    monkeypatch.setattr(autonomy, "find_page", lambda _page_id: page)
    store = _convergence_store(tmp_path)
    initial_authority = _authority(autonomy.RETENTION_FRONTIER_LANE)
    changed_authority = _authority(
        autonomy.RETENTION_FRONTIER_LANE,
        artifact_sha256="e" * 64,
    )
    effect_lock = False
    lock_entries = 0

    @contextmanager
    def authority_lock():
        nonlocal effect_lock, lock_entries
        lock_entries += 1
        effect_lock = lock_entries >= 2
        try:
            yield
        finally:
            effect_lock = False

    monkeypatch.setattr(autonomy, "decision_authority_lock", authority_lock)
    monkeypatch.setattr(
        autonomy,
        "_current_autonomy_authority",
        lambda *_args, **_kwargs: (
            changed_authority if effect_lock else initial_authority,
            None,
        ),
    )
    result = autonomy.apply_retention_archives(
        {"deprecation_candidates": ["old"], "pages": {"old": {"score": 0.01}}},
        write=False,
        budget=_frontier_budget(calls=1, mutations=1),
        convergence_store=store,
        reviewer=lambda _candidate: _authority_bound_review(
            "deprecate",
            0.99,
            "Redundant",
            initial_authority,
        ),
        now=NOW,
    )

    assert lock_entries == 2
    assert result["status_counts"] == {"frontier_retry": 1}
    assert "authority changed" in result["decisions"][0]["result"]["reason"]
    assert page.read_bytes() == before


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
    payload = {"deprecation_candidates": ["old"], "pages": {"old": {"score": 0.01}}}
    with pytest.raises(RuntimeError, match="simulated retention crash"):
        autonomy.apply_retention_archives(
            payload,
            write=False,
            convergence_store=store,
            reviewer=lambda _candidate: {
                "decision": "deprecate",
                "confidence": 0.99,
                "summary": "Redundant",
            },
            now=NOW,
        )

    meta, _body = parse_frontmatter(page.read_text(encoding="utf-8"))
    assert meta["status"] == "deprecated"
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

    assert recovered["decisions"][0]["action"] == "already_deprecated"
    assert recovered["decisions"][0]["convergence_status"] == "applied"
    assert recovered["decisions"][0]["recovery_only"] is True
    assert recovered["decisions"][0]["semantic_effect"] is False
    item = store.list_items(lane=autonomy.RETENTION_FRONTIER_LANE)[0]
    assert item["status"] == "applied"


def test_retention_does_not_recover_from_frontmatter_only_postimage_match(
    monkeypatch,
    tmp_path: Path,
) -> None:
    page = tmp_path / "old.md"
    _write_page(page, "Old", "Redundant")
    monkeypatch.setattr(autonomy, "find_page", lambda _page_id: page)
    store = _convergence_store(tmp_path)
    real_complete = store.complete

    def crash_after_write(key, status, **kwargs):
        if status == "applied":
            raise RuntimeError("simulated retention crash")
        return real_complete(key, status, **kwargs)

    monkeypatch.setattr(store, "complete", crash_after_write)
    payload = {"deprecation_candidates": ["old"], "pages": {"old": {"score": 0.01}}}
    with pytest.raises(RuntimeError, match="simulated retention crash"):
        autonomy.apply_retention_archives(
            payload,
            write=False,
            convergence_store=store,
            reviewer=lambda _candidate: {
                "decision": "deprecate",
                "confidence": 0.99,
                "summary": "Redundant",
            },
            now=NOW,
        )

    approval_path = next((store.state_file.parent / "approvals").rglob("*.json"))
    approval = json.loads(approval_path.read_text(encoding="utf-8"))
    exact = page.read_bytes()
    assert approval["effect_receipt"]["postimages"]["old"] == {
        "sha256": hashlib.sha256(exact).hexdigest(),
        "size_bytes": len(exact),
    }
    page.write_bytes(exact + b"concurrent body change\n")

    monkeypatch.setattr(store, "complete", real_complete)
    result = autonomy.apply_retention_archives(
        payload,
        write=False,
        convergence_store=store,
        reviewer=lambda _candidate: (_ for _ in ()).throw(
            AssertionError("reviewer called during non-exact retention recovery")
        ),
        now=NOW + timedelta(seconds=1),
    )

    assert result["decisions"][0]["action"] == "already_deprecated"
    assert "convergence_status" not in result["decisions"][0]
    item = store.list_items(lane=autonomy.RETENTION_FRONTIER_LANE)[0]
    assert item["status"] != "applied"


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
        {"deprecation_candidates": ["old"], "pages": {"old": {"score": 0.01}}},
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
        "old", {"status": "deprecated"}, expected_hash="stale"
    )

    assert result["status"] == "retry"
    assert result["reason"] == "page_changed_before_apply"
    meta, _body = parse_frontmatter(page.read_text(encoding="utf-8"))
    assert meta["status"] == "stable"


def test_watchdog_alerts_when_sleep_never_ran(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(autonomy, "WATCHDOG_FILE", tmp_path / "watchdog.json")
    monkeypatch.setattr(
        "chronovisor.ops.health.health_snapshot",
        lambda: {
            "memory_integrity": {"capture_rate": 0.95},
            "queues": {"duplicate_candidates": 0, "lint_repair": 0},
        },
    )
    monkeypatch.setattr(autonomy, "_latest_jsonl", lambda path: {})
    monkeypatch.setattr(
        "chronovisor.core.runtime_config.load_decision_router_config",
        lambda: SimpleNamespace(adoption_artifact=""),
    )
    writes: list[tuple[Path, dict]] = []
    history: list[dict] = []
    monkeypatch.setattr(
        autonomy, "_write_json", lambda path, payload: writes.append((path, payload))
    )
    monkeypatch.setattr(
        autonomy, "_write_watchdog_history", lambda payload: history.append(payload)
    )

    payload = autonomy.watchdog_snapshot(write=True)

    assert payload["status"] == "alert"
    assert payload["alerts"][0]["type"] == "sleep_never_ran"
    assert writes[0][0] == autonomy.WATCHDOG_FILE
    assert history == [payload]


def test_watchdog_does_not_alert_on_convergence_semantic_defer(monkeypatch) -> None:
    now = datetime.now().isoformat(timespec="seconds")
    monkeypatch.setattr(
        "chronovisor.ops.health.health_snapshot",
        lambda: {
            "memory_integrity": {"capture_rate": 0.95},
            "queues": {"duplicate_candidates": 0, "lint_repair": 0},
            "convergence": {
                "semantic_deferred": 1,
                "quarantined": 0,
                "expired_running": 0,
                "oldest_actionable_age_hours": 0,
            },
            "capture_pipeline": {
                "background_jobs": {"by_status": {"completed": 1}},
                "session_sweeper": {"status": "ok"},
            },
            "runtime": {"commit_id": "abc123", "drift": False},
        },
    )
    monkeypatch.setattr(
        autonomy,
        "_latest_jsonl",
        lambda _path: {"status": "ok", "started_at": now},
    )

    payload = autonomy.watchdog_snapshot(write=False)

    assert payload["status"] == "ok"
    assert not any(
        alert.get("type") == "convergence_quarantined" for alert in payload["alerts"]
    )


def test_watchdog_does_not_alert_on_managed_lint_catch_up(monkeypatch) -> None:
    now = datetime.now().isoformat(timespec="seconds")
    monkeypatch.setattr(
        "chronovisor.ops.health.health_snapshot",
        lambda: {
            "memory_integrity": {"capture_rate": 0.95},
            "queues": {
                "duplicate_candidates": 0,
                "lint_repair": 334,
                "lint_repair_active": 334,
                "lint_repair_untracked": 0,
            },
            "convergence": {
                "quarantined": 0,
                "expired_running": 0,
                "oldest_actionable_age_hours": 6,
            },
            "capture_pipeline": {
                "background_jobs": {"by_status": {"completed": 1}},
                "session_sweeper": {"status": "ok"},
            },
            "runtime": {"commit_id": "abc123", "drift": False},
        },
    )
    monkeypatch.setattr(
        autonomy,
        "_latest_jsonl",
        lambda _path: {"status": "ok", "started_at": now},
    )

    payload = autonomy.watchdog_snapshot(write=False)

    assert payload["status"] == "ok"
    assert payload["alerts"] == []


def test_watchdog_alerts_on_unmanaged_lint_backlog(monkeypatch) -> None:
    now = datetime.now().isoformat(timespec="seconds")
    monkeypatch.setattr(
        "chronovisor.ops.health.health_snapshot",
        lambda: {
            "memory_integrity": {"capture_rate": 0.95},
            "queues": {
                "duplicate_candidates": 0,
                "lint_repair": 700,
                "lint_repair_active": 100,
                "lint_repair_untracked": 600,
            },
            "convergence": {
                "quarantined": 0,
                "expired_running": 0,
                "oldest_actionable_age_hours": 0,
            },
            "capture_pipeline": {
                "background_jobs": {"by_status": {"completed": 1}},
                "session_sweeper": {"status": "ok"},
            },
            "runtime": {"commit_id": "abc123", "drift": False},
        },
    )
    monkeypatch.setattr(
        autonomy,
        "_latest_jsonl",
        lambda _path: {"status": "ok", "started_at": now},
    )

    payload = autonomy.watchdog_snapshot(write=False)

    assert payload["status"] == "alert"
    assert payload["alerts"] == [
        {
            "type": "lint_backlog_high",
            "value": 600,
            "total": 700,
            "managed": 100,
        }
    ]


def test_watchdog_alerts_on_operational_convergence_quarantine(monkeypatch) -> None:
    now = datetime.now().isoformat(timespec="seconds")
    monkeypatch.setattr(
        "chronovisor.ops.health.health_snapshot",
        lambda: {
            "memory_integrity": {"capture_rate": 0.95},
            "queues": {"duplicate_candidates": 0, "lint_repair": 0},
            "convergence": {
                "semantic_deferred": 1,
                "quarantined": 2,
                "expired_running": 0,
                "oldest_actionable_age_hours": 0,
            },
            "capture_pipeline": {
                "background_jobs": {"by_status": {"completed": 1}},
                "session_sweeper": {"status": "ok"},
            },
            "runtime": {"commit_id": "abc123", "drift": False},
        },
    )
    monkeypatch.setattr(
        autonomy,
        "_latest_jsonl",
        lambda _path: {"status": "ok", "started_at": now},
    )

    payload = autonomy.watchdog_snapshot(write=False)

    assert payload["status"] == "alert"
    assert payload["alerts"] == [{"type": "convergence_quarantined", "value": 2}]


def test_watchdog_still_alerts_on_operational_background_quarantine(
    monkeypatch,
) -> None:
    now = datetime.now().isoformat(timespec="seconds")
    monkeypatch.setattr(
        "chronovisor.ops.health.health_snapshot",
        lambda: {
            "memory_integrity": {"capture_rate": 0.95},
            "queues": {"duplicate_candidates": 0, "lint_repair": 0},
            "convergence": {
                "semantic_deferred": 1,
                "quarantined": 0,
                "expired_running": 0,
                "oldest_actionable_age_hours": 0,
            },
            "capture_pipeline": {
                "background_jobs": {"by_status": {"quarantined": 2}},
                "session_sweeper": {"status": "ok"},
            },
            "runtime": {"commit_id": "abc123", "drift": False},
        },
    )
    monkeypatch.setattr(
        autonomy,
        "_latest_jsonl",
        lambda _path: {"status": "ok", "started_at": now},
    )

    payload = autonomy.watchdog_snapshot(write=False)

    assert payload["status"] == "alert"
    assert payload["alerts"] == [{"type": "background_jobs_quarantined", "value": 2}]


def test_watchdog_ignores_retained_old_background_quarantine(monkeypatch) -> None:
    now = datetime.now().isoformat(timespec="seconds")
    monkeypatch.setattr(
        "chronovisor.ops.health.health_snapshot",
        lambda: {
            "memory_integrity": {"capture_rate": 0.95},
            "queues": {"duplicate_candidates": 0, "lint_repair": 0},
            "convergence": {
                "quarantined": 0,
                "expired_running": 0,
                "oldest_actionable_age_hours": 0,
            },
            "capture_pipeline": {
                "background_jobs": {
                    "by_status": {"quarantined": 67},
                    "quarantined_24h": 0,
                    "latest_quarantined_at": "2026-07-11T04:10:26+00:00",
                },
                "session_sweeper": {"status": "ok"},
            },
            "runtime": {"commit_id": "abc123", "drift": False},
        },
    )
    monkeypatch.setattr(
        autonomy,
        "_latest_jsonl",
        lambda _path: {"status": "ok", "started_at": now},
    )

    payload = autonomy.watchdog_snapshot(write=False)

    assert payload["status"] == "ok"
    assert payload["alerts"] == []


def test_watchdog_notification_suppresses_small_drift_until_reminder() -> None:
    previous = {
        "last_sent_at": NOW.isoformat(),
        "last_notified_alerts": [{"type": "lint_backlog_high", "value": 700}],
    }

    unchanged = autonomy._watchdog_notification_plan(
        [{"type": "lint_backlog_high", "value": 701}],
        previous,
        now=NOW + timedelta(hours=1),
    )
    increased = autonomy._watchdog_notification_plan(
        [{"type": "lint_backlog_high", "value": 1100}],
        previous,
        now=NOW + timedelta(hours=1),
    )
    reminder = autonomy._watchdog_notification_plan(
        [{"type": "lint_backlog_high", "value": 701}],
        previous,
        now=NOW + timedelta(hours=6),
    )

    assert unchanged == {"send": False, "reason": "unchanged"}
    assert increased == {"send": True, "reason": "severity_increased"}
    assert reminder == {"send": True, "reason": "reminder"}


def test_watchdog_notification_sends_once_and_reports_recovery(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []
    test_now = datetime.now()
    monkeypatch.setattr(autonomy, "_now", lambda: test_now.isoformat())
    monkeypatch.setattr(
        autonomy,
        "_send_notification",
        lambda title, body: calls.append((title, body)) or {"sent": True},
    )
    alerts = [{"type": "lint_backlog_high", "value": 700}]

    first = autonomy._watchdog_notification_state(alerts, {}, enabled=True)
    second = autonomy._watchdog_notification_state(alerts, first, enabled=True)
    recovered = autonomy._watchdog_notification_state([], first, enabled=True)
    healthy_again = autonomy._watchdog_notification_state([], recovered, enabled=True)

    assert first["status"] == "sent"
    assert first["reason"] == "new_alerts"
    assert second["status"] == "suppressed"
    assert second["reason"] == "unchanged"
    assert recovered["status"] == "sent"
    assert recovered["reason"] == "recovered"
    assert healthy_again["status"] == "suppressed"
    assert len(calls) == 2
    assert calls[1][1] == "Autonomy recovered"


def test_watchdog_read_only_never_sends_notification(monkeypatch) -> None:
    monkeypatch.setattr(
        "chronovisor.ops.health.health_snapshot",
        lambda: {
            "memory_integrity": {"capture_rate": 0.95},
            "queues": {"duplicate_candidates": 0, "lint_repair": 0},
        },
    )
    monkeypatch.setattr(autonomy, "_latest_jsonl", lambda _path: {})
    monkeypatch.setattr(
        autonomy,
        "_send_notification",
        lambda *_args: (_ for _ in ()).throw(AssertionError("notification sent")),
    )

    payload = autonomy.watchdog_snapshot(write=False, notify=True)

    assert payload["status"] == "alert"
    assert "notification" not in payload


def test_watchdog_history_is_compact_and_bounded_to_1000_lines(
    monkeypatch, tmp_path: Path
) -> None:
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
        "chronovisor.ops.health.health_snapshot",
        lambda: {
            "memory_integrity": {"capture_rate": 0.96},
            "queues": {"duplicate_candidates": 7, "lint_repair": 9},
            "large_unneeded_section": {"blob": "z" * 5000},
        },
    )
    monkeypatch.setattr(
        autonomy,
        "_latest_jsonl",
        lambda path: {
            "status": "ok",
            "started_at": "2026-07-10T03:40:00",
            "payload": "q" * 5000,
        },
    )

    payload = autonomy.watchdog_snapshot(write=True)

    rows = [
        json.loads(line)
        for line in history_file.read_text(encoding="utf-8").splitlines()
    ]
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
    persisted = json.loads(
        (tmp_path / "watchdog-latest.json").read_text(encoding="utf-8")
    )
    assert persisted["latest_sleep"] == rows[-1]["latest_sleep"]


def test_regression_guard_quarantines_cycle_without_global_git_reset(
    monkeypatch, tmp_path: Path
) -> None:
    quarantine_file = tmp_path / "quarantine.json"
    decisions_file = tmp_path / "decisions.jsonl"
    monkeypatch.setattr(autonomy, "QUARANTINE_FILE", quarantine_file)
    monkeypatch.setattr(autonomy, "DECISIONS_FILE", decisions_file)
    monkeypatch.setattr(
        autonomy,
        "_git",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("global git reset attempted")
        ),
    )

    payload = autonomy.regression_guard(
        before_health={"memory_integrity": {"capture_rate": 0.95}},
        after_watchdog={
            "alerts": [
                {"type": "capture_rate_regression", "before": 0.95, "after": 0.7}
            ]
        },
        snapshot={"head": "abc123"},
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
        snapshot={"head": "def456"},
        auto_revert=True,
        write=False,
    )
    assert quarantine_file.read_bytes() == before


def test_install_launchd_dry_run_builds_sleep_and_watchdog_plists(monkeypatch) -> None:
    monkeypatch.setattr(autonomy, "_uvx_path", lambda: "/opt/homebrew/bin/uvx")

    payload = autonomy.install_launchd(dry_run=True, load=False)

    assert payload["status"] == "ok"
    assert payload["dry_run"] is True
    labels = {item["label"] for item in payload["plists"]}
    assert autonomy.SLEEP_LABEL in labels
    assert autonomy.CONVERGE_LABEL in labels
    assert autonomy.WATCHDOG_LABEL in labels
    assert autonomy.DEADMAN_LABEL in labels
    assert autonomy.SOAK_LABEL in labels
    programs = {item["label"]: item["program"] for item in payload["plists"]}
    assert Path(programs[autonomy.SLEEP_LABEL][0]).name == "chronovisor-sleep"
    assert Path(programs[autonomy.WATCHDOG_LABEL][0]).name == "chronovisor-watchdog"
    sleep_plist = next(
        item for item in payload["plists"] if item["label"] == autonomy.SLEEP_LABEL
    )
    assert sleep_plist["run_at_load"] is True
    assert (
        Path(programs[autonomy.DEADMAN_LABEL][0]).name
        == "chronovisor-deadman-observer"
    )
    assert "/usr/bin/python3" not in programs[autonomy.DEADMAN_LABEL]
    watchdog_plist = next(
        item for item in payload["plists"] if item["label"] == autonomy.WATCHDOG_LABEL
    )
    assert watchdog_plist["stdout"] == os.devnull
    command = payload["wrappers"][0]["command"]
    assert command[0] == "/opt/homebrew/bin/uvx"
    assert "--refresh-package" in command
    assert "git+ssh://git@github.com/trafficsign/chronovisor" in command
    assert "--json" not in payload["wrappers"][0]["command"]
    converge = next(
        item for item in payload["plists"] if item["label"] == autonomy.CONVERGE_LABEL
    )
    assert converge["program"][0].endswith("chronovisor-converge")
    converge_wrapper = next(
        item
        for item in payload["wrappers"]
        if Path(item["path"]).name == "chronovisor-converge"
    )
    assert "--no-sleep" in converge_wrapper["command"]
    assert "--with-sleep" not in converge_wrapper["command"]
    soak_wrapper = next(
        item
        for item in payload["wrappers"]
        if Path(item["path"]).name == "chronovisor-soak"
    )
    assert "chronovisor-burn-monitor" in soak_wrapper["command"]
    assert "--expected-commit" in soak_wrapper["command"]
    assert "--output" not in soak_wrapper["command"]


def test_uninstall_launchd_dry_run_lists_all_generated_artifacts() -> None:
    payload = autonomy.uninstall_launchd(dry_run=True, unload=True)

    assert payload["status"] == "ok"
    assert payload["dry_run"] is True
    assert len(payload["plists"]) == 5
    assert any(path.endswith("chronovisor-soak.plist") for path in payload["plists"])
    assert any(path.endswith("chronovisor-soak") for path in payload["wrappers"])


def test_install_then_uninstall_launchd_round_trip_in_fixture(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / ".chronovisor"
    launch_agents = tmp_path / "LaunchAgents"
    wrappers = root / "bin"
    monkeypatch.setattr(autonomy, "CHRONOVISOR_ROOT", root)
    monkeypatch.setattr(autonomy, "LAUNCH_AGENT_DIR", launch_agents)
    monkeypatch.setattr(autonomy, "WRAPPER_DIR", wrappers)
    monkeypatch.setattr(autonomy, "_uvx_path", lambda: "/opt/homebrew/bin/uvx")
    monkeypatch.setattr(
        autonomy,
        "runtime_identity",
        lambda: {"expected_commit": "a" * 40},
    )

    installed = autonomy.install_launchd(dry_run=False, load=False)

    assert installed["status"] == "ok"
    assert len(list(launch_agents.glob("com.trafficsign.chronovisor-*.plist"))) == 5
    assert (wrappers / "chronovisor-soak").exists()
    assert "a" * 40 in (wrappers / "chronovisor-soak").read_text(encoding="utf-8")
    observer = wrappers / "chronovisor-deadman-observer"
    observer_source = Path(autonomy.__file__).parents[1] / "deadman_observer.py"
    assert observer.read_bytes() == observer_source.read_bytes()
    assert observer.stat().st_mode & 0o111

    removed = autonomy.uninstall_launchd(dry_run=False, unload=False)

    assert removed["status"] == "ok"
    assert not list(launch_agents.glob("*.plist"))
    assert not any(Path(path).exists() for path in removed["wrappers"])
