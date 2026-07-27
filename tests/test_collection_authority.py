from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from chronovisor import collection_anomaly_worker, collection_authority
from chronovisor.collection_authority import (
    CollectionAuthorityError,
    CollectionRegistry,
    adjudicate_collection_review_queue,
    build_review_candidates,
    collection_quality_snapshot,
    evaluate_unseen40,
    load_contract,
    load_crosswalk,
)
from chronovisor.durable_state import read_sealed_json, write_sealed_json
from chronovisor.page_identity import new_page_uid
from chronovisor.page_registry import PageRegistry


def _page(path: Path, uid: str, *, links: tuple[str, ...] = ()) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "---\n"
        f"title: {path.stem}\n"
        f"uid: {uid}\n"
        "updated: 2026-07-27\n"
        "---\n\n"
        f"# {path.stem}\n\n"
        + "\n".join(f"[[{value}]]" for value in links)
        + "\n",
        encoding="utf-8",
    )


def _uids(count: int, *, start: int = 1) -> list[str]:
    return [
        new_page_uid(
            timestamp_ms=1_725_000_000_000 + index,
            random_bits=index,
        )
        for index in range(start, start + count)
    ]


def test_collection_contract_and_crosswalk_are_frozen_and_fully_audited() -> None:
    contract = load_contract()
    crosswalk = load_crosswalk()

    assert contract["decision"] == "existing_collection_is_primary_authority"
    assert contract["anomaly_reviewer"]["assignment_mutation_capability"] is False
    assert crosswalk["epoch"] == "collection-crosswalk-v2"
    assert len(crosswalk["entries"]) == 66
    assert len(crosswalk["by_slug"]) == 66
    assert crosswalk["by_slug"]["misc"]["review_required"] is True
    assert {
        mapping["relation"]
        for mapping in crosswalk["by_slug"]["chronovisor"]["mappings"]
    } == {"exact", "broad"}


def test_collection_sync_is_stable_and_direct_pages_fail_closed(
    tmp_path: Path,
) -> None:
    ai_uid, loose_uid = _uids(2)
    _page(tmp_path / "pages" / "ai" / "model.md", ai_uid)
    _page(tmp_path / "pages" / "loose.md", loose_uid)
    registry = CollectionRegistry(
        tmp_path,
        uid_factory=iter(_uids(10, start=100)).__next__,
    )

    first = registry.sync_from_pages()
    second = registry.sync_from_pages()
    state = registry.load()
    collections = state["collections"]
    ai_collection = collections[state["slug_index"]["ai"]]
    unresolved = collections[state["slug_index"]["_unclassified"]]

    assert first["assignment_count"] == 2
    assert second["created_collections"] == []
    assert second["generation"] == first["generation"]
    assert ai_collection["uid"] == state["assignments"][ai_uid]["collection_uid"]
    assert unresolved["is_unclassified"] is True
    assert state["assignments"][loose_uid]["status"] == "unclassified"
    page_state = PageRegistry(tmp_path).load()
    assert page_state["pages"][ai_uid]["collection_uid"] == ai_collection["uid"]
    assert (
        page_state["pages"][loose_uid]["collection_status"]
        == "review_required"
    )
    receipts = list(
        (tmp_path / "runtime" / "librarian" / "collection-receipts").glob(
            "*.json"
        )
    )
    assert len(receipts) == 2
    assert all(read_sealed_json(path)["page_mutations"] == 0 for path in receipts)


def _review_worker_result(
    *,
    decision: str = "no_issue",
    model: str = "gemma4:26b",
    digest: str = "digest",
    suggested: str | None = None,
) -> SimpleNamespace:
    if suggested is None:
        suggested = "ai" if decision == "review_recommended" else ""
    return SimpleNamespace(
        status="completed",
        error=None,
        value={
            "schema": collection_anomaly_worker.WORKER_SCHEMA,
            "model": model,
            "model_digest": digest,
            "prompt_sha256": collection_anomaly_worker.PROMPT_SHA256,
            "model_calls": 1,
            "page_mutations": 0,
            "assignment_mutations": 0,
            "result": {
                "schema": collection_anomaly_worker.REVIEW_SCHEMA,
                "decision": decision,
                "suggested_collection_slug": suggested,
                "rationale": "The original collection remains defensible.",
                "evidence": "The page content matches its original order.",
            },
        },
    )


def test_collection_no_issue_review_is_checkpointed_and_dismissed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _page(
        tmp_path / "pages" / "misc" / "note.md",
        _uids(1, start=50)[0],
    )
    CollectionRegistry(tmp_path).sync_from_pages()
    collection_authority.refresh_review_queue(tmp_path)
    monkeypatch.setattr(
        "chronovisor.ollama.model_digests",
        lambda _models: {"gemma4:26b": "digest"},
    )
    monkeypatch.setattr(
        collection_authority,
        "research_lane",
        lambda *_args, **_kwargs: nullcontext(object()),
    )
    monkeypatch.setattr(
        collection_authority,
        "run_cancellable_command",
        lambda *_args, **_kwargs: _review_worker_result(),
    )

    result = collection_authority.review_collection_queue(
        tmp_path,
        limit=1,
        model="gemma4:26b",
    )
    queue = read_sealed_json(
        tmp_path
        / "runtime"
        / "librarian"
        / "collection-review-queue.json"
    )
    item = next(iter(queue["items"].values()))

    assert result["reviewer_calls"] == 1
    assert item["status"] == "dismissed"
    assert item["resolution"] == "model_no_issue_preserve_original_order"
    assert queue["open"] == 0
    assert queue["completed"] == 1
    assert queue["reviewer_calls"] == 1
    assert queue["assignment_mutations"] == 0
    assert queue["page_mutations"] == 0


def test_existing_no_issue_review_is_reconciled_without_another_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _page(
        tmp_path / "pages" / "misc" / "note.md",
        _uids(1, start=55)[0],
    )
    CollectionRegistry(tmp_path).sync_from_pages()
    queue = collection_authority.refresh_review_queue(tmp_path)
    item = next(iter(queue["items"].values()))
    item["model_review"] = {
        "schema": collection_anomaly_worker.REVIEW_SCHEMA,
        "decision": "no_issue",
        "suggested_collection_slug": "",
        "rationale": "The original collection remains defensible.",
        "evidence": "The page content matches its original order.",
        "model": "gemma4:26b",
        "model_digest": "digest",
        "prompt_sha256": collection_anomaly_worker.PROMPT_SHA256,
        "reviewed_at": "2026-07-27T00:00:00+00:00",
    }
    queue_path = (
        tmp_path
        / "runtime"
        / "librarian"
        / "collection-review-queue.json"
    )
    write_sealed_json(queue_path, queue, backup=True)
    monkeypatch.setattr(
        "chronovisor.ollama.model_digests",
        lambda _models: {"gemma4:26b": "digest"},
    )

    result = collection_authority.review_collection_queue(
        tmp_path,
        limit=0,
        model="gemma4:26b",
    )
    persisted = read_sealed_json(queue_path)
    persisted_item = next(iter(persisted["items"].values()))

    assert result["reviewer_calls"] == 0
    assert result["reconciled"] == [item["candidate_id"]]
    assert persisted_item["status"] == "dismissed"
    assert persisted["open"] == 0
    assert persisted["completed"] == 1


def test_collection_challenger_rejects_move_and_preserves_original_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _page(
        tmp_path / "pages" / "misc" / "note.md",
        _uids(1, start=58)[0],
    )
    CollectionRegistry(tmp_path).sync_from_pages()
    queue = collection_authority.refresh_review_queue(tmp_path)
    item = next(iter(queue["items"].values()))
    item["status"] = "review_recommended"
    item["model_review"] = {
        "schema": collection_anomaly_worker.REVIEW_SCHEMA,
        "decision": "review_recommended",
        "suggested_collection_slug": "ai",
        "rationale": "The page may fit AI.",
        "evidence": "The page mentions AI.",
        "model": "gemma4:26b",
        "model_digest": "primary-digest",
        "prompt_sha256": collection_anomaly_worker.PROMPT_SHA256,
        "reviewed_at": "2026-07-27T00:00:00+00:00",
    }
    queue_path = (
        tmp_path
        / "runtime"
        / "librarian"
        / "collection-review-queue.json"
    )
    write_sealed_json(queue_path, queue, backup=True)
    monkeypatch.setattr(
        "chronovisor.ollama.model_digests",
        lambda _models: {"gpt-oss:20b": "challenger-digest"},
    )
    monkeypatch.setattr(
        collection_authority,
        "research_lane",
        lambda *_args, **_kwargs: nullcontext(object()),
    )
    monkeypatch.setattr(
        collection_authority,
        "run_cancellable_command",
        lambda *_args, **_kwargs: _review_worker_result(
            model="gpt-oss:20b",
            digest="challenger-digest",
        ),
    )

    result = collection_authority.review_collection_queue(
        tmp_path,
        limit=1,
        model="gpt-oss:20b",
        role="challenger",
    )
    persisted = read_sealed_json(queue_path)
    persisted_item = next(iter(persisted["items"].values()))

    assert result["role"] == "challenger"
    assert result["reviewer_calls"] == 1
    assert persisted_item["status"] == "dismissed"
    assert persisted_item["challenge_status"] == "rejected_recommendation"
    assert persisted_item["resolution"] == (
        "challenger_no_issue_preserve_original_order"
    )
    assert persisted["open"] == 0
    assert persisted["assignment_mutations"] == 0


def test_collection_challenger_records_consensus_without_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _page(
        tmp_path / "pages" / "misc" / "note.md",
        _uids(1, start=59)[0],
    )
    CollectionRegistry(tmp_path).sync_from_pages()
    queue = collection_authority.refresh_review_queue(tmp_path)
    item = next(iter(queue["items"].values()))
    item["status"] = "review_recommended"
    item["model_review"] = {
        "schema": collection_anomaly_worker.REVIEW_SCHEMA,
        "decision": "review_recommended",
        "suggested_collection_slug": "ai",
        "rationale": "The page may fit AI.",
        "evidence": "The page mentions AI.",
        "model": "gemma4:26b",
        "model_digest": "primary-digest",
        "prompt_sha256": collection_anomaly_worker.PROMPT_SHA256,
        "reviewed_at": "2026-07-27T00:00:00+00:00",
    }
    queue_path = (
        tmp_path
        / "runtime"
        / "librarian"
        / "collection-review-queue.json"
    )
    write_sealed_json(queue_path, queue, backup=True)
    monkeypatch.setattr(
        "chronovisor.ollama.model_digests",
        lambda _models: {"gpt-oss:20b": "challenger-digest"},
    )
    monkeypatch.setattr(
        collection_authority,
        "research_lane",
        lambda *_args, **_kwargs: nullcontext(object()),
    )
    monkeypatch.setattr(
        collection_authority,
        "run_cancellable_command",
        lambda *_args, **_kwargs: _review_worker_result(
            decision="review_recommended",
            model="gpt-oss:20b",
            digest="challenger-digest",
            suggested="ai",
        ),
    )

    collection_authority.review_collection_queue(
        tmp_path,
        limit=1,
        model="gpt-oss:20b",
        role="challenger",
    )
    persisted = read_sealed_json(queue_path)
    persisted_item = next(iter(persisted["items"].values()))

    assert persisted_item["status"] == "review_recommended"
    assert persisted_item["challenge_status"] == "consensus_recommended"
    assert (
        persisted_item["challenger_review"]["suggested_collection_slug"]
        == "ai"
    )
    assert persisted["open"] == 1
    assert persisted["assignment_mutations"] == 0


def test_collection_review_checkpoint_survives_later_worker_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_uid, second_uid = _uids(2, start=60)
    _page(tmp_path / "pages" / "misc" / "first.md", first_uid)
    _page(tmp_path / "pages" / "misc" / "second.md", second_uid)
    CollectionRegistry(tmp_path).sync_from_pages()
    collection_authority.refresh_review_queue(tmp_path)
    monkeypatch.setattr(
        "chronovisor.ollama.model_digests",
        lambda _models: {"gemma4:26b": "digest"},
    )
    monkeypatch.setattr(
        collection_authority,
        "research_lane",
        lambda *_args, **_kwargs: nullcontext(object()),
    )
    calls = 0

    def run_worker(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return _review_worker_result()
        raise RuntimeError("simulated worker crash")

    monkeypatch.setattr(
        collection_authority,
        "run_cancellable_command",
        run_worker,
    )

    with pytest.raises(RuntimeError, match="simulated worker crash"):
        collection_authority.review_collection_queue(
            tmp_path,
            limit=2,
            model="gemma4:26b",
        )

    queue = read_sealed_json(
        tmp_path
        / "runtime"
        / "librarian"
        / "collection-review-queue.json"
    )
    items = list(queue["items"].values())
    assert sum("model_review" in item for item in items) == 1
    assert sum(item["status"] == "dismissed" for item in items) == 1
    assert queue["reviewer_calls"] == 1
    assert queue["completed"] == 1


def test_collection_review_stops_batch_when_model_lane_is_deferred(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_uid, second_uid = _uids(2, start=70)
    _page(tmp_path / "pages" / "misc" / "first.md", first_uid)
    _page(tmp_path / "pages" / "misc" / "second.md", second_uid)
    CollectionRegistry(tmp_path).sync_from_pages()
    collection_authority.refresh_review_queue(tmp_path)
    monkeypatch.setattr(
        "chronovisor.ollama.model_digests",
        lambda _models: {"gemma4:26b": "digest"},
    )
    monkeypatch.setattr(
        collection_authority,
        "research_lane",
        lambda *_args, **_kwargs: nullcontext(object()),
    )
    calls = 0

    def defer_worker(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return SimpleNamespace(
            status="deferred",
            value=None,
            error="sync_pending",
        )

    monkeypatch.setattr(
        collection_authority,
        "run_cancellable_command",
        defer_worker,
    )

    result = collection_authority.review_collection_queue(
        tmp_path,
        limit=2,
        model="gemma4:26b",
    )

    assert calls == 1
    assert result["status"] == "partial"
    assert result["reviewer_calls"] == 0
    assert len(result["deferred"]) == 1
    assert result["deferred"][0]["reason"] == "sync_pending"


def test_collection_lifecycle_is_cas_receipted_and_non_destructive(
    tmp_path: Path,
) -> None:
    first_uid, second_uid = _uids(2)
    first_path = tmp_path / "pages" / "ai" / "first.md"
    second_path = tmp_path / "pages" / "career" / "second.md"
    _page(first_path, first_uid)
    _page(second_path, second_uid)
    registry = CollectionRegistry(
        tmp_path,
        uid_factory=iter(_uids(20, start=200)).__next__,
    )
    synced = registry.sync_from_pages()
    state = registry.load()
    ai_collection = state["slug_index"]["ai"]
    career_collection = state["slug_index"]["career"]

    renamed = registry.apply_lifecycle(
        "rename",
        expected_generation=synced["generation"],
        collection_uid=ai_collection,
        new_label="AI systems",
    )
    moved = registry.apply_lifecycle(
        "move",
        expected_generation=renamed["generation_after"],
        target_collection_uid=ai_collection,
        page_uids=[second_uid],
    )
    split = registry.apply_lifecycle(
        "split",
        expected_generation=moved["generation_after"],
        collection_uid=ai_collection,
        page_uids=[second_uid],
        new_slug="ai-career",
        new_label="AI career",
    )
    merged = registry.apply_lifecycle(
        "merge",
        expected_generation=split["generation_after"],
        collection_uid=split["created_collection_uid"],
        target_collection_uid=career_collection,
    )

    assert first_path.is_file()
    assert second_path.is_file()
    assert renamed["page_mutations"] == 0
    assert moved["affected_page_uids"] == [second_uid]
    assert merged["affected_page_uids"] == [second_uid]
    final = registry.load()
    assert (
        final["assignments"][second_uid]["collection_uid"]
        == career_collection
    )
    with pytest.raises(CollectionAuthorityError, match="generation changed"):
        registry.apply_lifecycle(
            "rename",
            expected_generation=synced["generation"],
            collection_uid=ai_collection,
            new_label="stale",
        )


def test_collection_batch_move_is_one_cas_and_preserves_page_bytes(
    tmp_path: Path,
) -> None:
    first_uid, second_uid = _uids(2, start=250)
    first_path = tmp_path / "pages" / "misc" / "first.md"
    second_path = tmp_path / "pages" / "misc" / "second.md"
    _page(first_path, first_uid)
    _page(second_path, second_uid)
    registry = CollectionRegistry(
        tmp_path,
        uid_factory=iter(_uids(20, start=280)).__next__,
    )
    synced = registry.sync_from_pages()
    state = registry.load()
    ai_row = registry._new_collection(
        slug="ai",
        label="Artificial intelligence",
        source_path=None,
        created_by="test",
    )
    career_row = registry._new_collection(
        slug="career",
        label="Career",
        source_path=None,
        created_by="test",
    )
    state["collections"][ai_row["uid"]] = ai_row
    state["collections"][career_row["uid"]] = career_row
    state["slug_index"]["ai"] = ai_row["uid"]
    state["slug_index"]["career"] = career_row["uid"]
    state["generation"] = synced["generation"] + 1
    write_sealed_json(registry.path, state, backup=True)
    before = {
        first_uid: first_path.read_bytes(),
        second_uid: second_path.read_bytes(),
    }

    receipt = registry.apply_batch_moves(
        {
            first_uid: ai_row["uid"],
            second_uid: career_row["uid"],
        },
        expected_generation=state["generation"],
    )

    final = registry.load()
    assert receipt["operation"] == "batch_move"
    assert receipt["generation_after"] == state["generation"] + 1
    assert final["assignments"][first_uid]["collection_uid"] == ai_row["uid"]
    assert (
        final["assignments"][second_uid]["collection_uid"]
        == career_row["uid"]
    )
    assert first_path.read_bytes() == before[first_uid]
    assert second_path.read_bytes() == before[second_uid]
    assert receipt["page_mutations"] == 0


def test_host_adjudication_moves_review_required_and_preserves_affinity(
    tmp_path: Path,
) -> None:
    misc_uid, ai_uid, career_a, career_b, career_c = _uids(5, start=320)
    misc_path = tmp_path / "pages" / "misc" / "orphan.md"
    ai_path = tmp_path / "pages" / "ai" / "misplaced.md"
    _page(misc_path, misc_uid)
    _page(
        ai_path,
        ai_uid,
        links=("career-a", "career-b", "career-c"),
    )
    _page(tmp_path / "pages" / "career" / "career-a.md", career_a)
    _page(tmp_path / "pages" / "career" / "career-b.md", career_b)
    _page(tmp_path / "pages" / "career" / "career-c.md", career_c)
    registry = CollectionRegistry(
        tmp_path,
        uid_factory=iter(_uids(30, start=380)).__next__,
    )
    registry.sync_from_pages()
    index = {
        "entries": {
            "orphan": {"outlinks": []},
            "misplaced": {
                "outlinks": ["career-a", "career-b", "career-c"]
            },
            "career-a": {"outlinks": []},
            "career-b": {"outlinks": []},
            "career-c": {"outlinks": []},
        }
    }
    index_path = tmp_path / ".index" / "pages.json"
    index_path.parent.mkdir(parents=True)
    index_path.write_text(json.dumps(index), encoding="utf-8")
    queue = collection_authority.refresh_review_queue(tmp_path)
    misc_candidate = next(
        row
        for row in queue["items"].values()
        if row["reason"] == "collection_requires_review"
    )
    before = {
        misc_uid: misc_path.read_bytes(),
        ai_uid: ai_path.read_bytes(),
    }
    result = adjudicate_collection_review_queue(
        tmp_path,
        {
            "schema": collection_authority.COLLECTION_DECISION_SCHEMA,
            "status": "approved",
            "approved_at": "2026-07-27T00:00:00+00:00",
            "decision_authority": "test_host",
            "expected_registry_generation": registry.load()["generation"],
            "preserve_remaining_reasons": [
                "cross_collection_link_affinity"
            ],
            "decisions": [
                {
                    "candidate_id": misc_candidate["candidate_id"],
                    "action": "move",
                    "target_collection_slug": "ai",
                    "rationale": "The page is an AI note.",
                }
            ],
        },
    )

    final = registry.load()
    persisted = read_sealed_json(
        tmp_path
        / "runtime"
        / "librarian"
        / "collection-review-queue.json"
    )
    ai_collection = final["slug_index"]["ai"]
    assert final["assignments"][misc_uid]["collection_uid"] == ai_collection
    assert result["moves"] == 1
    assert result["preserves"] == 1
    assert result["queue"]["open"] == 0
    assert {
        row["status"] for row in persisted["items"].values()
    } == {"move_approved", "dismissed"}
    assert persisted["host_assignment_mutations"] == 1
    assert persisted["page_mutations"] == 0
    assert misc_path.read_bytes() == before[misc_uid]
    assert ai_path.read_bytes() == before[ai_uid]
    page_state = PageRegistry(tmp_path).load()
    assert page_state["pages"][misc_uid]["collection_uid"] == ai_collection
    assert page_state["pages"][misc_uid]["collection_status"] == "assigned"


def test_review_candidates_detect_misc_and_cross_collection_affinity(
    tmp_path: Path,
) -> None:
    misc_uid, ai_uid, career_a, career_b, career_c = _uids(5)
    _page(tmp_path / "pages" / "misc" / "orphan.md", misc_uid)
    _page(
        tmp_path / "pages" / "ai" / "misplaced.md",
        ai_uid,
        links=("career-a", "career-b", "career-c"),
    )
    _page(tmp_path / "pages" / "career" / "career-a.md", career_a)
    _page(tmp_path / "pages" / "career" / "career-b.md", career_b)
    _page(tmp_path / "pages" / "career" / "career-c.md", career_c)
    registry = CollectionRegistry(
        tmp_path,
        uid_factory=iter(_uids(20, start=300)).__next__,
    )
    state = registry.sync_from_pages()["registry"]
    index = {
        "entries": {
            "orphan": {"outlinks": []},
            "misplaced": {
                "outlinks": ["career-a", "career-b", "career-c"]
            },
            "career-a": {"outlinks": []},
            "career-b": {"outlinks": []},
            "career-c": {"outlinks": []},
        }
    }
    index_path = tmp_path / ".index" / "pages.json"
    index_path.parent.mkdir(parents=True)
    index_path.write_text(json.dumps(index), encoding="utf-8")

    candidates = build_review_candidates(tmp_path, state=state)
    by_uid = {}
    for row in candidates:
        by_uid.setdefault(row["page_uid"], []).append(row)

    assert {row["reason"] for row in by_uid[misc_uid]} == {
        "collection_requires_review"
    }
    affinity = next(
        row
        for row in by_uid[ai_uid]
        if row["reason"] == "cross_collection_link_affinity"
    )
    assert affinity["proposed_collection_slug"] == "career"
    assert affinity["assignment_mutation"] is False


def test_quality_gate_warns_and_proposes_without_auto_split(
    tmp_path: Path,
) -> None:
    identifiers = _uids(8)
    for index, uid in enumerate(identifiers[:6]):
        _page(tmp_path / "pages" / "ai" / f"ai-{index}.md", uid)
    for index, uid in enumerate(identifiers[6:]):
        _page(tmp_path / "pages" / "career" / f"career-{index}.md", uid)
    registry = CollectionRegistry(
        tmp_path,
        uid_factory=iter(_uids(20, start=400)).__next__,
    )
    state = registry.sync_from_pages()["registry"]
    page_index = {
        "entries": {
            f"ai-{index}": {
                "outlinks": [f"ai-{(index + 1) % 6}"]
            }
            for index in range(6)
        }
    }
    index_path = tmp_path / ".index" / "pages.json"
    index_path.parent.mkdir(parents=True)
    index_path.write_text(json.dumps(page_index), encoding="utf-8")

    quality = collection_quality_snapshot(
        tmp_path,
        state=state,
        queue={
            "candidate_count": 0,
            "open": 0,
        },
    )

    assert quality["metrics"]["assignment_coverage"] == 1.0
    assert quality["metrics"]["top_collection_share"] == 0.75
    assert "top_collection_share" in quality["hard_failures"]
    assert quality["split_proposals"][0]["auto_split"] is False
    assert (
        quality["split_proposals"][0]["algorithm"]
        == "deterministic_label_propagation_v1"
    )


def test_unseen_evaluation_honors_locked_assignment_or_review_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identifiers = _uids(40, start=500)
    for index, uid in enumerate(identifiers):
        slug = "misc" if index == 14 else "ai"
        _page(tmp_path / "pages" / slug / f"page-{index}.md", uid)
    registry = CollectionRegistry(
        tmp_path,
        uid_factory=iter(_uids(20, start=700)).__next__,
    )
    registry.sync_from_pages()
    selection_path = tmp_path / "selection.json"
    sealed = write_sealed_json(
        selection_path,
        {
            "schema": "chronovisor.cvo-ab-unseen-selection.v1",
            "case_count": 40,
            "cases": [{"uid": uid} for uid in identifiers],
        },
        backup=False,
    )
    prereg_path = tmp_path / "prereg.json"
    gold_path = tmp_path / "gold.json"
    prereg_path.write_text(
        json.dumps(
            {
                "schema": "chronovisor.collection-authority-unseen-prereg.v1",
                "epoch": "test",
                "status": "locked-before-evaluation",
                "selection_seal_sha256": sealed["seal_sha256"],
                "case_count": 40,
                "evaluation_contract": {
                    "assignment_or_review_rate_min": 1.0,
                    "major_error_max": 0,
                    "crosswalk_invalid_max": 0,
                    "page_mutations_max": 0,
                    "model_calls": 0,
                },
            }
        ),
        encoding="utf-8",
    )
    gold_path.write_text(
        json.dumps(
            {
                "schema": "chronovisor.collection-authority-unseen-gold.v1",
                "epoch": "test",
                "status": "sealed-before-evaluation",
                "selection_seal_sha256": sealed["seal_sha256"],
                "cases": [
                    {
                        "uid": uid,
                        "disposition": "review" if index == 14 else "assigned",
                        "acceptable_collection_slugs": (
                            ["chronovisor"] if index == 14 else ["ai"]
                        ),
                    }
                    for index, uid in enumerate(identifiers)
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        collection_authority,
        "default_preregistration_path",
        lambda: prereg_path,
    )
    monkeypatch.setattr(
        collection_authority,
        "default_gold_path",
        lambda: gold_path,
    )

    result = evaluate_unseen40(tmp_path, selection_path=selection_path)

    assert result["decision"] == "adopt"
    assert result["assigned_correct"] == 39
    assert result["review_correct"] == 1
    assert result["major_error_count"] == 0
    assert result["model_calls"] == 0


def test_anomaly_worker_is_review_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        collection_anomaly_worker.ollama,
        "model_digests",
        lambda _models: {"gemma4:test": "sha256:model"},
    )
    monkeypatch.setattr(
        collection_anomaly_worker.ollama,
        "chat",
        lambda *_args, **_kwargs: json.dumps(
            {
                "decision": "review_recommended",
                "suggested_collection_slug": "chronovisor",
                "rationale": "The page describes Chronovisor validation.",
                "evidence": "Title and body both describe a soak test.",
            }
        ),
    )

    result = collection_anomaly_worker.run(
        {
            "schema": collection_anomaly_worker.WORKER_SCHEMA,
            "model": "gemma4:test",
            "model_digest": "sha256:model",
            "candidate": {
                "current_collection_slug": "misc",
                "reason": "collection_requires_review",
            },
            "document": {
                "title": "Chronovisor soak",
                "summary": "",
                "evidence_excerpt": "Validation.",
            },
            "collections": [
                {"slug": "misc", "label": "Misc"},
                {"slug": "chronovisor", "label": "Chronovisor"},
            ],
        }
    )

    assert result["result"]["decision"] == "review_recommended"
    assert result["model_calls"] == 1
    assert result["page_mutations"] == 0
    assert result["assignment_mutations"] == 0


def test_anomaly_worker_gpt_oss_reserves_bounded_reasoning_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}
    monkeypatch.setattr(
        collection_anomaly_worker.ollama,
        "model_digests",
        lambda _models: {"gpt-oss:20b": "sha256:model"},
    )

    def _chat(*_args: object, **kwargs: object) -> str:
        observed.update(kwargs)
        return json.dumps(
            {
                "decision": "no_issue",
                "suggested_collection_slug": "",
                "rationale": "The current collection is defensible.",
                "evidence": "The title and excerpt match the collection.",
            }
        )

    monkeypatch.setattr(collection_anomaly_worker.ollama, "chat", _chat)

    collection_anomaly_worker.run(
        {
            "schema": collection_anomaly_worker.WORKER_SCHEMA,
            "model": "gpt-oss:20b",
            "model_digest": "sha256:model",
            "candidate": {
                "current_collection_slug": "chronovisor",
                "reason": "collection_requires_review",
            },
            "document": {
                "title": "Chronovisor",
                "summary": "",
                "evidence_excerpt": "Validation.",
            },
            "collections": [{"slug": "chronovisor", "label": "Chronovisor"}],
        }
    )

    assert observed["num_predict"] == 1_800
    assert observed["think"] == "low"
