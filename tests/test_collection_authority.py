from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from chronovisor.core.durable_state import read_sealed_json, write_sealed_json
from chronovisor.core.page_identity import new_page_uid
from chronovisor.ingest.page_registry import PageRegistry
from chronovisor.recall import collection_anomaly_worker, collection_authority
from chronovisor.recall.collection_authority import (
    CollectionAuthorityError,
    CollectionRegistry,
    adjudicate_collection_review_queue,
    autonomously_finalize_collection_queue,
    build_review_candidates,
    collection_quality_snapshot,
    ensure_autonomous_crosswalk,
    evaluate_unseen40,
    load_contract,
    load_crosswalk,
)


def _page(
    path: Path,
    uid: str,
    *,
    links: tuple[str, ...] = (),
    sensitivity: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "---\n"
        f"title: {path.stem}\n"
        "status: stable\n"
        "type: knowledge\n"
        f"uid: {uid}\n"
        "updated: 2026-07-27\n"
        + (f"sensitivity: {sensitivity}\n" if sensitivity else "")
        + "---\n\n"
        + f"# {path.stem}\n\n"
        + "\n".join(f"[{value}](<{value}.md>)" for value in links)
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
    assert contract["consensus_adjudicator"]["runtime_roles"] == [
        "librarian.review",
        "librarian.review.challenger",
    ]
    assert crosswalk["epoch"] == "collection-crosswalk-v3"
    assert len(crosswalk["entries"]) == 67
    assert len(crosswalk["by_slug"]) == 67
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
    assert page_state["pages"][loose_uid]["collection_status"] == "review_required"
    receipts = list(
        (tmp_path / "runtime" / "librarian" / "collection-receipts").glob("*.json")
    )
    assert len(receipts) == 2
    assert all(read_sealed_json(path)["page_mutations"] == 0 for path in receipts)


def test_new_collection_crosswalk_is_local_consensus_and_runtime_sealed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from chronovisor.core import ollama

    page_uid = _uids(1, start=115)[0]
    _page(tmp_path / "pages" / "new-topic" / "note.md", page_uid)
    registry = CollectionRegistry(
        tmp_path,
        uid_factory=iter(_uids(10, start=120)).__next__,
    )
    state = registry.sync_from_pages()["registry"]
    routes = (
        ollama.RuntimeGenerationRoute(
            role="classification.anchor.primary",
            provider="local",
            model="gemma4:26b",
            location="local",
            structured_output=True,
        ),
        ollama.RuntimeGenerationRoute(
            role="classification.anchor.challenger",
            provider="local",
            model="gpt-oss:20b",
            location="local",
            structured_output=True,
        ),
    )
    monkeypatch.setattr(ollama, "runtime_generation_routes", lambda _roles: routes)
    monkeypatch.setattr(
        ollama,
        "model_digests",
        lambda models: {
            str(model): f"digest-{index}" for index, model in enumerate(models)
        },
    )
    monkeypatch.setattr(
        collection_authority,
        "_propose_collection_crosswalk",
        lambda *args, **kwargs: (
            "cvo:anchor:0001",
            [{"model": "gemma4:26b"}, {"model": "gpt-oss:20b"}],
            4,
        ),
    )

    result = ensure_autonomous_crosswalk(
        tmp_path,
        state=state,
        use_models=True,
    )
    crosswalk = load_crosswalk(root=tmp_path)

    assert result["changed"] is True
    assert result["model_calls"] == 4
    assert crosswalk["by_slug"]["new-topic"]["review_required"] is False
    assert crosswalk["by_slug"]["new-topic"]["mappings"] == [
        {"anchor_id": "cvo:anchor:0001", "relation": "exact"}
    ]
    assert read_sealed_json(Path(result["path"]))["frontier_calls"] == 0


def test_remote_collection_crosswalk_never_touches_local_ollama_controls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chronovisor.core import ollama
    from chronovisor.recall.classification_anchor_worker import (
        PROMPT_SHA256,
        WORKER_SCHEMA,
    )

    page_uid = _uids(1, start=215)[0]
    _page(
        tmp_path / "pages" / "remote-topic" / "note.md",
        page_uid,
        sensitivity="normal",
    )
    registry = CollectionRegistry(
        tmp_path,
        uid_factory=iter(_uids(10, start=220)).__next__,
    )
    state = registry.sync_from_pages()["registry"]
    routes = (
        ollama.RuntimeGenerationRoute(
            role="classification.anchor.primary",
            provider="remote-a",
            model="model-a",
            location="remote",
            structured_output=True,
        ),
        ollama.RuntimeGenerationRoute(
            role="classification.anchor.challenger",
            provider="remote-b",
            model="model-b",
            location="remote",
            structured_output=True,
        ),
    )
    by_role = {route.role: route for route in routes}

    def call_worker(*, payload, purpose, timeout_seconds=720.0):
        del purpose, timeout_seconds
        assert payload["source_sensitivity"] == "normal"
        route = by_role[str(payload["runtime_role"])]
        operation = str(payload["operation"])
        result = (
            {
                "central_subject": "Remote topic",
                "secondary_subjects": [],
                "rationale": "The page has one principal subject.",
            }
            if operation == "extract"
            else {
                "primary_anchor_id": "cvo:anchor:0001",
                "secondary_anchor_ids": [],
                "rationale": "The anchor contains the subject.",
            }
        )
        return {
            "schema": WORKER_SCHEMA,
            "operation": operation,
            "model": route.model,
            "model_digest": None,
            "route_identity": {
                "role": route.role,
                "provider": route.provider,
                "model": route.model,
                "location": route.location,
            },
            "prompt_sha256": PROMPT_SHA256,
            "model_calls": 1,
            "result": result,
        }

    def forbidden(*_args, **_kwargs):
        pytest.fail("remote crosswalk touched Ollama metadata")

    monkeypatch.setattr(ollama, "runtime_generation_routes", lambda _roles: routes)
    monkeypatch.setattr(ollama, "model_digests", forbidden)
    monkeypatch.setattr(
        collection_authority,
        "_call_crosswalk_anchor_worker",
        call_worker,
    )

    result = ensure_autonomous_crosswalk(tmp_path, state=state, use_models=True)
    runtime = read_sealed_json(Path(result["path"]))
    entry = next(row for row in runtime["entries"] if row["slug"] == "remote-topic")

    assert result["model_calls"] == 4
    assert entry["review_required"] is False
    assert entry["authority"] == "model_consensus"
    assert entry["route_identities"] == [
        {
            "role": route.role,
            "provider": route.provider,
            "model": route.model,
            "location": route.location,
        }
        for route in routes
    ]
    assert "model_digests" not in entry


def test_collection_capsule_defaults_unknown_sensitivity_high(tmp_path: Path) -> None:
    page_uid = _uids(1, start=315)[0]
    _page(tmp_path / "pages" / "sensitive-topic" / "note.md", page_uid)
    registry = CollectionRegistry(
        tmp_path,
        uid_factory=iter(_uids(10, start=320)).__next__,
    )
    state = registry.sync_from_pages()["registry"]
    collection_uid = next(
        uid
        for uid, row in state["collections"].items()
        if row["slug"] == "sensitive-topic"
    )

    capsule = collection_authority._collection_crosswalk_capsule(
        tmp_path,
        state=state,
        collection_uid=collection_uid,
    )

    assert capsule["source_sensitivity"] == "high"


def test_collection_crosswalk_rejects_duplicate_route_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chronovisor.core import ollama

    page_uid = _uids(1, start=415)[0]
    _page(
        tmp_path / "pages" / "duplicate-route" / "note.md",
        page_uid,
        sensitivity="normal",
    )
    registry = CollectionRegistry(
        tmp_path,
        uid_factory=iter(_uids(10, start=420)).__next__,
    )
    state = registry.sync_from_pages()["registry"]
    routes = tuple(
        ollama.RuntimeGenerationRoute(
            role=role,
            provider="remote",
            model="same-model",
            location="remote",
            structured_output=True,
        )
        for role in (
            "classification.anchor.primary",
            "classification.anchor.challenger",
        )
    )

    monkeypatch.setattr(ollama, "runtime_generation_routes", lambda _roles: routes)
    monkeypatch.setattr(
        collection_authority,
        "_propose_collection_crosswalk",
        lambda *_args, **_kwargs: pytest.fail("duplicate route reached inference"),
    )

    result = ensure_autonomous_crosswalk(tmp_path, state=state, use_models=True)
    runtime = read_sealed_json(Path(result["path"]))
    entry = next(
        row for row in runtime["entries"] if row["slug"] == "duplicate-route"
    )

    assert result["model_calls"] == 0
    assert entry["review_required"] is True
    assert entry["route_identities"] == []


def _review_worker_result(
    stdin_text: str,
    *,
    route,
    decision: str = "no_issue",
    digest: str | None = "digest-primary",
    suggested: str | None = None,
) -> SimpleNamespace:
    payload = json.loads(stdin_text)
    if suggested is None:
        suggested = "ai" if decision == "review_recommended" else ""
    return SimpleNamespace(
        status="completed",
        error=None,
        value={
            "schema": collection_anomaly_worker.WORKER_SCHEMA,
            "model": route.model,
            "route_identity": collection_anomaly_worker.route_identity(route),
            "model_digest": digest,
            "prompt_sha256": collection_anomaly_worker.PROMPT_SHA256,
            "review_input_sha256": payload["review_input_sha256"],
            "source_data_class": payload["source_data_class"],
            "source_sensitivity": payload["source_sensitivity"],
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


def _review_routes(monkeypatch: pytest.MonkeyPatch):
    from chronovisor.core import ollama

    routes = (
        ollama.RuntimeGenerationRoute(
            "librarian.review", "ollama", "gemma4:26b", "local", True
        ),
        ollama.RuntimeGenerationRoute(
            "librarian.review.challenger",
            "ollama",
            "gpt-oss:20b",
            "local",
            True,
        ),
    )
    monkeypatch.setattr(ollama, "runtime_generation_routes", lambda _roles: routes)
    monkeypatch.setattr(
        ollama,
        "model_digests",
        lambda _models: {
            "gemma4:26b": "digest-primary",
            "gpt-oss:20b": "digest-challenger",
        },
    )
    return routes


def _bound_review(
    root: Path,
    row: dict,
    route,
    digest: str | None,
    *,
    decision: str,
    suggested: str = "",
) -> dict:
    state = CollectionRegistry(root).load()
    collections = sorted(
        [
            {"slug": value["slug"], "label": value["label"]}
            for value in state["collections"].values()
            if value["status"] == "active"
        ],
        key=lambda value: (value["slug"], value["label"]),
    )
    current = collection_authority._current_review_input(
        root,
        row,
        PageRegistry(root).load(),
        collections,
    )
    assert current is not None
    _, _, source_class, sensitivity, input_sha256 = current
    return {
        "schema": collection_anomaly_worker.REVIEW_SCHEMA,
        "decision": decision,
        "suggested_collection_slug": suggested,
        "rationale": "The original collection remains defensible.",
        "evidence": "The page content matches its original order.",
        **collection_authority._review_binding(
            collection_anomaly_worker.route_identity(route),
            digest,
            collection_anomaly_worker.PROMPT_SHA256,
            input_sha256,
            source_class,
            sensitivity,
        ),
        "reviewed_at": "2026-07-27T00:00:00+00:00",
    }


@pytest.mark.parametrize(
    ("sensitivity", "expected"),
    [("normal", "normal"), (None, "high"), ("unknown", "high")],
)
def test_collection_review_source_requires_exact_registry_and_frontmatter_sensitivity(
    tmp_path: Path,
    sensitivity: str | None,
    expected: str,
) -> None:
    page_uid = _uids(1, start=48)[0]
    _page(tmp_path / "system" / "current-state.md", page_uid, sensitivity=sensitivity)
    page = PageRegistry(tmp_path).ensure_manifest()["registry"]["pages"][page_uid]

    document = collection_authority._review_document(tmp_path, page)

    assert document is not None
    assert document[1:] == ("system", expected)


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
    routes = _review_routes(monkeypatch)
    monkeypatch.setattr(
        collection_authority,
        "research_lane",
        lambda *_args, **_kwargs: nullcontext(object()),
    )
    monkeypatch.setattr(
        collection_authority,
        "run_cancellable_command",
        lambda _command, stdin, *_args, **_kwargs: _review_worker_result(
            stdin, route=routes[0]
        ),
    )

    result = collection_authority.review_collection_queue(
        tmp_path,
        limit=1,
        model="gemma4:26b",
    )
    queue = read_sealed_json(
        tmp_path / "runtime" / "librarian" / "collection-review-queue.json"
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
    routes = _review_routes(monkeypatch)
    item["model_review"] = _bound_review(
        tmp_path,
        item,
        routes[0],
        "digest-primary",
        decision="no_issue",
    )
    queue_path = tmp_path / "runtime" / "librarian" / "collection-review-queue.json"
    write_sealed_json(queue_path, queue, backup=True)
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
    routes = _review_routes(monkeypatch)
    item["status"] = "review_recommended"
    item["model_review"] = _bound_review(
        tmp_path,
        item,
        routes[0],
        "digest-primary",
        decision="review_recommended",
        suggested="ai",
    )
    queue_path = tmp_path / "runtime" / "librarian" / "collection-review-queue.json"
    write_sealed_json(queue_path, queue, backup=True)
    monkeypatch.setattr(
        collection_authority,
        "research_lane",
        lambda *_args, **_kwargs: nullcontext(object()),
    )
    monkeypatch.setattr(
        collection_authority,
        "run_cancellable_command",
        lambda _command, stdin, *_args, **_kwargs: _review_worker_result(
            stdin,
            route=routes[1],
            digest="digest-challenger",
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
    routes = _review_routes(monkeypatch)
    item["status"] = "review_recommended"
    item["model_review"] = _bound_review(
        tmp_path,
        item,
        routes[0],
        "digest-primary",
        decision="review_recommended",
        suggested="ai",
    )
    queue_path = tmp_path / "runtime" / "librarian" / "collection-review-queue.json"
    write_sealed_json(queue_path, queue, backup=True)
    monkeypatch.setattr(
        collection_authority,
        "research_lane",
        lambda *_args, **_kwargs: nullcontext(object()),
    )
    monkeypatch.setattr(
        collection_authority,
        "run_cancellable_command",
        lambda _command, stdin, *_args, **_kwargs: _review_worker_result(
            stdin,
            route=routes[1],
            decision="review_recommended",
            digest="digest-challenger",
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
    assert persisted_item["challenger_review"]["suggested_collection_slug"] == "ai"
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
    routes = _review_routes(monkeypatch)
    monkeypatch.setattr(
        collection_authority,
        "research_lane",
        lambda *_args, **_kwargs: nullcontext(object()),
    )
    calls = 0

    def run_worker(_command, stdin, *_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return _review_worker_result(stdin, route=routes[0])
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
        tmp_path / "runtime" / "librarian" / "collection-review-queue.json"
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
    _review_routes(monkeypatch)
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


def test_collection_review_rejects_duplicate_routes_before_model_controls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chronovisor.core import ollama

    routes = tuple(
        ollama.RuntimeGenerationRoute(role, "remote", "same", "remote", True)
        for role in ("librarian.review", "librarian.review.challenger")
    )
    monkeypatch.setattr(ollama, "runtime_generation_routes", lambda _roles: routes)
    monkeypatch.setattr(
        ollama,
        "model_digests",
        lambda _models: pytest.fail("duplicate routes queried local digests"),
    )
    monkeypatch.setattr(
        collection_authority,
        "run_cancellable_command",
        lambda *_args, **_kwargs: pytest.fail("duplicate routes spawned a worker"),
    )

    with pytest.raises(CollectionAuthorityError, match="not independent"):
        collection_authority.review_collection_queue(tmp_path)


def test_review_model_is_only_a_resolved_model_assertion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    routes = _review_routes(monkeypatch)
    digest_calls = 0

    def digests(_models):
        nonlocal digest_calls
        digest_calls += 1
        return {}

    monkeypatch.setattr(collection_anomaly_worker.ollama, "model_digests", digests)
    monkeypatch.setattr(
        collection_authority,
        "run_cancellable_command",
        lambda *_args, **_kwargs: pytest.fail("model assertion spawned a worker"),
    )

    with pytest.raises(CollectionAuthorityError, match="assertion does not match"):
        collection_authority.review_collection_queue(
            tmp_path,
            model="not-" + routes[0].model,
        )

    assert digest_calls == 0


def test_remote_collection_review_captures_normal_page_source_without_controls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chronovisor.core import ollama

    _page(
        tmp_path / "pages" / "misc" / "note.md",
        _uids(1, start=90)[0],
        sensitivity="normal",
    )
    CollectionRegistry(tmp_path).sync_from_pages()
    collection_authority.refresh_review_queue(tmp_path)
    routes = (
        ollama.RuntimeGenerationRoute(
            "librarian.review", "remote-a", "model-a", "remote", True
        ),
        ollama.RuntimeGenerationRoute(
            "librarian.review.challenger", "remote-b", "model-b", "remote", True
        ),
    )
    monkeypatch.setattr(ollama, "runtime_generation_routes", lambda _roles: routes)
    monkeypatch.setattr(
        ollama,
        "model_digests",
        lambda _models: pytest.fail("remote route queried Ollama digests"),
    )
    lanes = []
    monkeypatch.setattr(
        collection_authority,
        "research_lane",
        lambda *_args, **kwargs: lanes.append(kwargs) or nullcontext(object()),
    )
    captured = []

    def worker(_command, stdin, *_args, **_kwargs):
        captured.append(json.loads(stdin))
        return _review_worker_result(stdin, route=routes[0], digest=None)

    monkeypatch.setattr(collection_authority, "run_cancellable_command", worker)

    result = collection_authority.review_collection_queue(tmp_path, limit=1)

    assert result["route_identity"]["location"] == "remote"
    assert result["model_digest"] is None
    assert captured[0]["source_data_class"] == "page"
    assert captured[0]["source_sensitivity"] == "normal"
    assert lanes[0]["needs_model"] is False


@pytest.mark.parametrize("drift", ["prompt", "route", "digest", "input"])
def test_collection_review_identity_drift_is_stale_without_inference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    drift: str,
) -> None:
    page_uid = _uids(1, start=95)[0]
    path = tmp_path / "pages" / "misc" / "note.md"
    _page(path, page_uid, sensitivity="normal")
    CollectionRegistry(tmp_path).sync_from_pages()
    queue = collection_authority.refresh_review_queue(tmp_path)
    routes = _review_routes(monkeypatch)
    row = next(iter(queue["items"].values()))
    row["model_review"] = _bound_review(
        tmp_path,
        row,
        routes[0],
        "digest-primary",
        decision="no_issue",
    )
    if drift == "prompt":
        row["model_review"]["prompt_sha256"] = "changed"
    elif drift == "route":
        row["model_review"]["route_identity"]["provider"] = "changed"
    elif drift == "digest":
        row["model_review"]["model_digest"] = "changed"
    else:
        path.write_text(path.read_text() + "changed\n", encoding="utf-8")
    queue_path = tmp_path / "runtime" / "librarian" / "collection-review-queue.json"
    write_sealed_json(queue_path, queue, backup=True)
    monkeypatch.setattr(
        collection_authority,
        "run_cancellable_command",
        lambda *_args, **_kwargs: pytest.fail("stale cache invoked inference"),
    )

    result = collection_authority.review_collection_queue(tmp_path, limit=0)
    stale = read_sealed_json(queue_path)["items"][row["candidate_id"]]

    assert result["reviewer_calls"] == 0
    assert stale["status"] == "queued"
    assert stale["stale_review_reason"] == "primary_evidence_not_current"


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
    assert final["assignments"][second_uid]["collection_uid"] == career_collection
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
    assert final["assignments"][second_uid]["collection_uid"] == career_row["uid"]
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
            "misplaced": {"outlinks": ["career-a", "career-b", "career-c"]},
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
            "preserve_remaining_reasons": ["cross_collection_link_affinity"],
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
        tmp_path / "runtime" / "librarian" / "collection-review-queue.json"
    )
    ai_collection = final["slug_index"]["ai"]
    assert final["assignments"][misc_uid]["collection_uid"] == ai_collection
    assert result["moves"] == 1
    assert result["preserves"] == 1
    assert result["queue"]["open"] == 0
    assert {row["status"] for row in persisted["items"].values()} == {
        "move_approved",
        "dismissed",
    }
    assert persisted["host_assignment_mutations"] == 1
    assert persisted["page_mutations"] == 0
    assert misc_path.read_bytes() == before[misc_uid]
    assert ai_path.read_bytes() == before[ai_uid]
    page_state = PageRegistry(tmp_path).load()
    assert page_state["pages"][misc_uid]["collection_uid"] == ai_collection
    assert page_state["pages"][misc_uid]["collection_status"] == "assigned"


def test_incremental_adjudication_does_not_reopen_terminal_required_items(
    tmp_path: Path,
) -> None:
    first_uid, second_uid, ai_uid = _uids(3, start=410)
    _page(tmp_path / "pages" / "misc" / "first.md", first_uid)
    _page(tmp_path / "pages" / "ai" / "reference.md", ai_uid)
    registry = CollectionRegistry(
        tmp_path,
        uid_factory=iter(_uids(20, start=430)).__next__,
    )
    registry.sync_from_pages()
    first_queue = collection_authority.refresh_review_queue(tmp_path)
    first_candidate = next(
        row
        for row in first_queue["items"].values()
        if row["page_uid"] == first_uid
        and row["reason"] == "collection_requires_review"
    )
    first_result = adjudicate_collection_review_queue(
        tmp_path,
        {
            "schema": collection_authority.COLLECTION_DECISION_SCHEMA,
            "status": "approved",
            "approved_at": "2026-07-27T00:00:00+00:00",
            "decision_authority": "test_host",
            "expected_registry_generation": registry.load()["generation"],
            "preserve_remaining_reasons": [],
            "decisions": [
                {
                    "candidate_id": first_candidate["candidate_id"],
                    "action": "move",
                    "target_collection_slug": "ai",
                    "rationale": "The first page is an AI note.",
                }
            ],
        },
    )
    assert first_result["moves"] == 1

    _page(tmp_path / "pages" / "misc" / "second.md", second_uid)
    registry.sync_from_pages()
    second_queue = collection_authority.refresh_review_queue(tmp_path)
    second_candidate = next(
        row
        for row in second_queue["items"].values()
        if row["page_uid"] == second_uid
        and row["reason"] == "collection_requires_review"
    )
    second_result = adjudicate_collection_review_queue(
        tmp_path,
        {
            "schema": collection_authority.COLLECTION_DECISION_SCHEMA,
            "status": "approved",
            "approved_at": "2026-07-27T00:05:00+00:00",
            "decision_authority": "test_host",
            "expected_registry_generation": registry.load()["generation"],
            "preserve_remaining_reasons": [],
            "decisions": [
                {
                    "candidate_id": second_candidate["candidate_id"],
                    "action": "move",
                    "target_collection_slug": "ai",
                    "rationale": "The second page is also an AI note.",
                }
            ],
        },
    )

    assert second_result["moves"] == 1
    assert second_result["queue"]["open"] == 0
    persisted = read_sealed_json(
        tmp_path / "runtime" / "librarian" / "collection-review-queue.json"
    )
    assert persisted["items"][first_candidate["candidate_id"]]["status"] == (
        "move_approved"
    )
    assert persisted["items"][second_candidate["candidate_id"]]["status"] == (
        "move_approved"
    )


def test_local_consensus_moves_and_disagreement_preserves_without_host(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    move_uid, preserve_uid, ai_uid = _uids(3, start=470)
    _page(tmp_path / "pages" / "misc" / "move.md", move_uid)
    _page(tmp_path / "pages" / "misc" / "preserve.md", preserve_uid)
    _page(tmp_path / "pages" / "ai" / "reference.md", ai_uid)
    registry = CollectionRegistry(
        tmp_path,
        uid_factory=iter(_uids(20, start=490)).__next__,
    )
    registry.sync_from_pages()
    queue = collection_authority.refresh_review_queue(tmp_path)
    routes = _review_routes(monkeypatch)
    for candidate_id, raw_row in queue["items"].items():
        row = dict(raw_row)
        if row["page_uid"] == move_uid:
            primary_slug = challenger_slug = "ai"
        elif row["page_uid"] == preserve_uid:
            primary_slug, challenger_slug = "ai", "career"
        else:
            continue
        row["status"] = "review_recommended"
        row["model_review"] = _bound_review(
            tmp_path,
            row,
            routes[0],
            "digest-primary",
            decision="review_recommended",
            suggested=primary_slug,
        )
        row["challenger_review"] = _bound_review(
            tmp_path,
            row,
            routes[1],
            "digest-challenger",
            decision="review_recommended",
            suggested=challenger_slug,
        )
        queue["items"][candidate_id] = row
    write_sealed_json(
        tmp_path / "runtime" / "librarian" / "collection-review-queue.json",
        queue,
        backup=True,
    )

    result = autonomously_finalize_collection_queue(tmp_path)

    final = registry.load()
    assert result["moves"] == 1
    assert result["terminal_preserves"] == 1
    assert result["queue_open"] == 0
    assert final["assignments"][move_uid]["collection_uid"] == final["slug_index"]["ai"]
    assert (
        final["assignments"][preserve_uid]["collection_uid"]
        == final["slug_index"]["misc"]
    )
    persisted = read_sealed_json(
        tmp_path / "runtime" / "librarian" / "collection-review-queue.json"
    )
    preserve_row = next(
        row for row in persisted["items"].values() if row["page_uid"] == preserve_uid
    )
    assert preserve_row["status"] == "dismissed"
    assert preserve_row["resolution"] == (
        "autonomous_fail_closed_preserve_original_order"
    )


@pytest.mark.parametrize(
    "legacy_schema",
    ["chronovisor.collection-review-queue.v1", None],
)
def test_legacy_open_review_stays_stale_after_schema_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    legacy_schema: str | None,
) -> None:
    page_uid, ai_uid = _uids(2, start=540)
    _page(tmp_path / "pages" / "misc" / "note.md", page_uid)
    _page(tmp_path / "pages" / "ai" / "reference.md", ai_uid)
    registry = CollectionRegistry(tmp_path)
    registry.sync_from_pages()
    queue = collection_authority.refresh_review_queue(tmp_path)
    routes = _review_routes(monkeypatch)
    row = next(
        value for value in queue["items"].values() if value["page_uid"] == page_uid
    )
    row["status"] = "review_recommended"
    row["model_review"] = _bound_review(
        tmp_path,
        row,
        routes[0],
        "digest-primary",
        decision="review_recommended",
        suggested="ai",
    )
    row["challenger_review"] = _bound_review(
        tmp_path,
        row,
        routes[1],
        "digest-challenger",
        decision="review_recommended",
        suggested="ai",
    )
    if legacy_schema is not None:
        for review in (row["model_review"], row["challenger_review"]):
            review.pop("route_identity")
            review.pop("review_input_sha256")
    if legacy_schema is None:
        queue.pop("schema")
    else:
        queue["schema"] = legacy_schema
    queue_path = tmp_path / "runtime" / "librarian" / "collection-review-queue.json"
    write_sealed_json(queue_path, queue, backup=True)

    review = collection_authority.review_collection_queue(tmp_path, limit=0)
    finalized = autonomously_finalize_collection_queue(tmp_path)
    persisted = read_sealed_json(queue_path)
    stale = persisted["items"][row["candidate_id"]]

    assert review["reviewer_calls"] == 0
    assert finalized["moves"] == 0
    assert stale["status"] == "queued"
    assert stale["stale_review_reason"] == "primary_evidence_not_current"
    assert (
        registry.load()["assignments"][page_uid]["collection_uid"]
        == registry.load()["slug_index"]["misc"]
    )


def test_legacy_terminal_review_history_is_preserved(tmp_path: Path) -> None:
    page_uid = _uids(1, start=560)[0]
    _page(tmp_path / "pages" / "misc" / "note.md", page_uid)
    CollectionRegistry(tmp_path).sync_from_pages()
    queue = collection_authority.refresh_review_queue(tmp_path)
    row = next(iter(queue["items"].values()))
    row["status"] = "dismissed"
    row["model_review"] = {"schema": "chronovisor.collection-anomaly-review.v1"}
    queue["schema"] = "chronovisor.collection-review-queue.v1"
    queue_path = tmp_path / "runtime" / "librarian" / "collection-review-queue.json"
    write_sealed_json(queue_path, queue, backup=True)

    refreshed = collection_authority.refresh_review_queue(tmp_path)

    persisted = refreshed["items"][row["candidate_id"]]
    assert refreshed["schema"] == collection_authority.COLLECTION_QUEUE_SCHEMA
    assert persisted["model_review"] == row["model_review"]
    assert "stale_review_reason" not in persisted


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
            "misplaced": {"outlinks": ["career-a", "career-b", "career-c"]},
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

    assert {row["reason"] for row in by_uid[misc_uid]} == {"collection_requires_review"}
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
            f"ai-{index}": {"outlinks": [f"ai-{(index + 1) % 6}"]} for index in range(6)
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


def _anomaly_worker_payload(
    *,
    review_role: str = "primary",
    source_data_class: str = "page",
    source_sensitivity: str = "normal",
) -> dict:
    candidate = {
        "current_collection_slug": "misc",
        "reason": "collection_requires_review",
    }
    document = {
        "title": "Chronovisor soak",
        "summary": "",
        "evidence_excerpt": "Validation.",
    }
    collections = [
        {"slug": "misc", "label": "Misc"},
        {"slug": "chronovisor", "label": "Chronovisor"},
    ]
    return {
        "schema": collection_anomaly_worker.WORKER_SCHEMA,
        "review_role": review_role,
        "source_data_class": source_data_class,
        "source_sensitivity": source_sensitivity,
        "read_timeout_ms": 660_000,
        "candidate": candidate,
        "document": document,
        "collections": collections,
        "review_input_sha256": collection_anomaly_worker.review_input_sha256(
            candidate,
            document,
            collections,
            source_data_class=source_data_class,
            source_sensitivity=source_sensitivity,
        ),
    }


def test_anomaly_worker_is_review_only_and_binds_local_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chronovisor.core import ollama

    route = ollama.RuntimeGenerationRoute(
        "librarian.review", "ollama", "gemma4:test", "local", True
    )
    digest_calls = 0

    def digests(_models):
        nonlocal digest_calls
        digest_calls += 1
        return {"gemma4:test": "sha256:model"}

    monkeypatch.setattr(ollama, "runtime_generation_routes", lambda _roles: (route,))
    monkeypatch.setattr(ollama, "model_digests", digests)
    monkeypatch.setattr(
        ollama,
        "runtime_structured_chat",
        lambda *_args, **_kwargs: ollama.ChatResponse(
            content=json.dumps(
                {
                    "decision": "review_recommended",
                    "suggested_collection_slug": "chronovisor",
                    "rationale": "The page describes Chronovisor validation.",
                    "evidence": "Title and body both describe a soak test.",
                }
            )
        ),
    )

    result = collection_anomaly_worker.run(_anomaly_worker_payload())

    assert result["result"]["decision"] == "review_recommended"
    assert result["route_identity"] == collection_anomaly_worker.route_identity(route)
    assert result["model_digest"] == "sha256:model"
    assert digest_calls == 1
    assert result["model_calls"] == 1
    assert result["page_mutations"] == 0
    assert result["assignment_mutations"] == 0


def test_anomaly_worker_gpt_oss_reserves_bounded_reasoning_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chronovisor.core import ollama

    observed: dict[str, object] = {}
    route = ollama.RuntimeGenerationRoute(
        "librarian.review", "ollama", "gpt-oss:20b", "local", True
    )
    monkeypatch.setattr(ollama, "runtime_generation_routes", lambda _roles: (route,))
    monkeypatch.setattr(
        ollama,
        "model_digests",
        lambda _models: {"gpt-oss:20b": "sha256:model"},
    )

    def structured_chat(*args: object, **kwargs: object):
        observed["messages"] = args[0]
        observed.update(kwargs)
        return ollama.ChatResponse(
            content=json.dumps(
                {
                    "decision": "no_issue",
                    "suggested_collection_slug": "",
                    "rationale": "The current collection is defensible.",
                    "evidence": "The title and excerpt match the collection.",
                }
            )
        )

    monkeypatch.setattr(ollama, "runtime_structured_chat", structured_chat)

    collection_anomaly_worker.run(_anomaly_worker_payload())

    assert observed["num_predict"] == 1_800
    assert observed["think"] == "low"
    assert observed["keep_alive"] == "30s"
    prompt = json.loads(observed["messages"][1]["content"])
    assert "decision, suggested_collection_slug, rationale, and evidence" in " ".join(
        prompt["policy"]["rules"]
    )


def _remote_anomaly_runtime(monkeypatch: pytest.MonkeyPatch, backend_calls: list):
    from chronovisor.core import llm_config
    from chronovisor.core.llm_runtime import (
        BackendCapabilities,
        GenerationResult,
        GenerationRoute,
        LLMRuntime,
        RouteLocation,
    )

    class Backend:
        provider = "remote-test"
        location = RouteLocation.REMOTE

        def generate(self, request, *, model):
            backend_calls.append((request, model))
            return GenerationResult(
                content=json.dumps(
                    {
                        "decision": "no_issue",
                        "suggested_collection_slug": "",
                        "rationale": "The current collection is defensible.",
                        "evidence": "The page content matches the collection.",
                    }
                ),
                provider=self.provider,
                model=model,
            )

    runtime = LLMRuntime(
        generation={
            "librarian.review": GenerationRoute(
                Backend(),
                "remote-model",
                BackendCapabilities(
                    generation=True,
                    embedding=False,
                    structured_output=True,
                ),
            )
        }
    )
    monkeypatch.setattr(llm_config, "load_default_llm_runtime", lambda: runtime)
    return runtime


def _forbid_ollama_controls(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*_args, **_kwargs):
        raise AssertionError("remote review touched an Ollama control")

    for name in (
        "chat",
        "model_digests",
        "model_resource_lease",
        "plan_model_residency",
        "resident_model_rows",
        "unload_named_model",
        "unload_model",
    ):
        monkeypatch.setattr(collection_anomaly_worker.ollama, name, forbidden)


def test_remote_anomaly_worker_succeeds_without_ollama_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend_calls: list = []
    _remote_anomaly_runtime(monkeypatch, backend_calls)
    _forbid_ollama_controls(monkeypatch)

    result = collection_anomaly_worker.run(_anomaly_worker_payload())

    assert len(backend_calls) == 1
    assert result["route_identity"] == {
        "role": "librarian.review",
        "provider": "remote-test",
        "model": "remote-model",
        "location": "remote",
    }
    assert result["model_digest"] is None


def test_local_non_ollama_anomaly_worker_has_no_guessed_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chronovisor.core import ollama

    route = ollama.RuntimeGenerationRoute(
        "librarian.review", "local-engine", "local-model", "local", True
    )
    monkeypatch.setattr(ollama, "runtime_generation_routes", lambda _roles: (route,))
    monkeypatch.setattr(
        ollama,
        "model_digests",
        lambda _models: pytest.fail("non-Ollama route queried an Ollama digest"),
    )
    monkeypatch.setattr(
        ollama,
        "runtime_structured_chat",
        lambda *_args, **_kwargs: ollama.ChatResponse(
            content=json.dumps(
                {
                    "decision": "no_issue",
                    "suggested_collection_slug": "",
                    "rationale": "The current collection is defensible.",
                    "evidence": "The page matches the collection.",
                }
            )
        ),
    )

    result = collection_anomaly_worker.run(_anomaly_worker_payload())

    assert result["model_digest"] is None


def test_remote_high_system_anomaly_worker_is_denied_before_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chronovisor.core import ollama

    backend_calls: list = []
    _remote_anomaly_runtime(monkeypatch, backend_calls)
    _forbid_ollama_controls(monkeypatch)

    with pytest.raises(ollama.RuntimeBridgeError) as failure:
        collection_anomaly_worker.run(
            _anomaly_worker_payload(
                source_data_class="system",
                source_sensitivity="high",
            )
        )

    assert failure.value.category == "egress_denied"
    assert backend_calls == []


@pytest.mark.parametrize("override", ["model", "provider", "runtime_role"])
def test_anomaly_worker_rejects_route_overrides_before_resolve(
    monkeypatch: pytest.MonkeyPatch,
    override: str,
) -> None:
    resolves = 0

    def resolve(_roles):
        nonlocal resolves
        resolves += 1
        return ()

    monkeypatch.setattr(
        collection_anomaly_worker.ollama,
        "runtime_generation_routes",
        resolve,
    )
    payload = _anomaly_worker_payload()
    payload[override] = "forbidden"

    with pytest.raises(CollectionAuthorityError, match="overrides are forbidden"):
        collection_anomaly_worker.run(payload)

    assert resolves == 0


def test_anomaly_worker_missing_local_digest_never_calls_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chronovisor.core import ollama

    route = ollama.RuntimeGenerationRoute(
        "librarian.review", "ollama", "missing", "local", True
    )
    backends = 0
    monkeypatch.setattr(ollama, "runtime_generation_routes", lambda _roles: (route,))
    monkeypatch.setattr(ollama, "model_digests", lambda _models: {})

    def backend(*_args, **_kwargs):
        nonlocal backends
        backends += 1
        return "{}"

    monkeypatch.setattr(ollama, "runtime_structured_chat", backend)

    with pytest.raises(CollectionAuthorityError, match="digest is unavailable"):
        collection_anomaly_worker.run(_anomaly_worker_payload())

    assert backends == 0


def test_anomaly_worker_requires_structured_route_before_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chronovisor.core import ollama

    route = ollama.RuntimeGenerationRoute(
        "librarian.review", "ollama", "model", "local", False
    )
    monkeypatch.setattr(ollama, "runtime_generation_routes", lambda _roles: (route,))
    monkeypatch.setattr(
        ollama,
        "model_digests",
        lambda _models: pytest.fail("unstructured route queried model controls"),
    )
    monkeypatch.setattr(
        ollama,
        "runtime_structured_chat",
        lambda *_args, **_kwargs: pytest.fail("unstructured route called backend"),
    )

    with pytest.raises(CollectionAuthorityError, match="requires structured output"):
        collection_anomaly_worker.run(_anomaly_worker_payload())
