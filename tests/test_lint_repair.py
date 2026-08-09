from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from chronovisor.core.frontmatter import parse as parse_frontmatter
from chronovisor.decision.decision_router import canonical_agreement_signature
from chronovisor.decision.decision_schema_manifest import production_decision_schemas
from chronovisor.ingest.convergence import ConvergenceStore, CycleBudget, RetryPolicy
from chronovisor.ops import lint_repair
from tests.semantic_hold_support import semantic_authority, semantic_review

NOW = datetime(2026, 7, 10, 12, 0, tzinfo=UTC)
VALID_TAGS = ["d/tools-config", "t/howto", "s/evergreen"]


@pytest.fixture(autouse=True)
def isolate_decision_authority_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chronovisor.core import page_mutation

    monkeypatch.setattr(
        page_mutation,
        "DECISION_AUTHORITY_LOCK",
        tmp_path / "runtime" / "decision-authority.lock",
    )


def _semantic_authority(digest: str) -> dict:
    return {
        "source": "adopted_local_consensus",
        "authority_version": 1,
        "lane": lint_repair.TAG_REPAIR_DECISION_LANE,
        "lane_contract_sha256": "1" * 64,
        "lane_contract_manifest_sha256": "2" * 64,
        "lane_contract_case_manifest_sha256": "3" * 64,
        "policy": {
            "kind": "consensus",
            "schema_name": "lint_tag_repair",
            "mode": "enabled",
            "error": None,
        },
        "router": {
            "source": "adopted_artifact",
            "artifact_sha256": digest,
            "error": None,
            "models": ["primary", "challenger", "tie"],
        },
    }


def _local_consensus_proof(review: dict, authority: dict) -> dict:
    schema = production_decision_schemas()[authority["policy"]["schema_name"]]
    signature = canonical_agreement_signature(review, schema=schema)
    agreement = hashlib.sha256(signature.encode("utf-8")).hexdigest()
    models = authority["router"]["models"]
    return {
        "status": "agreed",
        "ok": True,
        "agreement_sha256": agreement,
        "failure_class": None,
        "quarantine_reason": None,
        "votes": [
            {
                "role": "primary",
                "model": models[0],
                "valid": True,
                "signature_sha256": agreement,
                "invalid_reason": None,
            },
            {
                "role": "challenger",
                "model": models[1],
                "valid": True,
                "signature_sha256": agreement,
                "invalid_reason": None,
            },
        ],
    }


def _authority_bound_tag_decision(
    decision: str,
    authority: dict,
    *,
    tags: list[str] | None = None,
) -> dict:
    value = {
        "decision": decision,
        "tags": list(tags if tags is not None else VALID_TAGS),
        "reason": "authority-bound exact tag review",
        "decision_policy": {
            "lane": authority["lane"],
            **authority["policy"],
            "router_policy": authority["router"],
        },
    }
    if decision == "rejected":
        value["tags"] = []
    value["local_consensus"] = _local_consensus_proof(value, authority)
    return value


def _semantic_no_quorum_tag_decision(authority: dict) -> dict:
    review = semantic_review(
        authority,
        lane=lint_repair.TAG_REPAIR_DECISION_LANE,
    )
    return {
        "decision": "needs_retry",
        "tags": [],
        "reason": review["summary"],
        "reviewer": review["reviewer"],
        "frontier_failure": review["frontier_failure"],
        "human_required": False,
        "decision_policy": review["decision_policy"],
        "local_consensus": review["local_consensus"],
    }


def test_default_local_reviewer_repairs_schema_error_in_same_session(
    tmp_path: Path,
) -> None:
    requests = []
    responses = iter(
        [
            json.dumps({"decision": "approved", "tags": "bad", "reason": "x"}),
            json.dumps(
                {"decision": "approved", "tags": VALID_TAGS, "reason": "matches page"}
            ),
        ]
    )

    def transport(request):
        requests.append(request)
        return next(responses)

    result = lint_repair._default_local_reviewer(
        "repair these tags",
        lint_repair.TAG_REPAIR_SCHEMA,
        transport=transport,
        audit_root=tmp_path / "audit",
    )

    assert result["tags"] == VALID_TAGS
    assert len(requests) == 2
    assert requests[1].messages[-2]["role"] == "assistant"
    assert "Validator errors" in requests[1].messages[-1]["content"]


def test_default_local_reviewer_rejects_oversized_input_before_transport(
    tmp_path: Path,
) -> None:
    calls = 0

    def transport(_request):
        nonlocal calls
        calls += 1
        raise AssertionError("transport must not start")

    with pytest.raises(ValueError, match="input_too_large|context_window_exceeded"):
        lint_repair._default_local_reviewer(
            "x" * 80_000,
            lint_repair.TAG_REPAIR_SCHEMA,
            transport=transport,
            audit_root=tmp_path / "audit",
        )

    assert calls == 0


def _store(tmp_path: Path, *, policy: RetryPolicy | None = None) -> ConvergenceStore:
    return ConvergenceStore(
        tmp_path / "runtime" / "convergence" / "state.json",
        policy=policy,
    )


def _budget() -> CycleBudget:
    return CycleBudget(
        max_local_calls=20,
        max_frontier_calls=20,
        max_mutations=20,
        max_elapsed_seconds=60,
    )


def _page(
    path: Path,
    *,
    tags: list[str] | None = None,
    body: str = "# Page\n\nUseful content.\n",
) -> str:
    tag_line = "" if tags is None else f"tags: [{', '.join(tags)}]\n"
    text = f"---\ntitle: Test Page\nupdated: 2026-01-01\n{tag_line}---\n\n{body}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return text


def _row(
    page: str,
    *,
    lane: str = "heavy_model_batch",
    issue_type: str = "tag_missing",
) -> dict[str, object]:
    return {
        "type": "lint_repair_candidate",
        "issue_key": f"key-{page}-{issue_type}",
        "lane": lane,
        "issue_type": issue_type,
        "severity": "high",
        "page": page,
        "detail": f"test {issue_type}",
        "auto_fixable": False,
    }


def _queue(path: Path, rows: list[dict[str, object]]) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    path.write_text(payload, encoding="utf-8")
    return payload.encode()


def _find_only(page_id: str, expected_id: str, path: Path) -> Path | None:
    return path if page_id == expected_id else None


def _never(*_args, **_kwargs):
    raise AssertionError("reviewer must not be called")


def test_normalize_tag_decision_is_fail_closed() -> None:
    approved = lint_repair.normalize_tag_decision(
        {"decision": "approved", "tags": VALID_TAGS, "reason": "matches page"}
    )
    invalid_axes = lint_repair.normalize_tag_decision(
        {"decision": "approved", "tags": ["d/tools-config"], "reason": "incomplete"}
    )
    malformed_rejection = lint_repair.normalize_tag_decision(
        {"decision": "rejected", "tags": [], "reason": "no", "unexpected": True}
    )
    duplicate_tags = lint_repair.normalize_tag_decision(
        {
            "decision": "approved",
            "tags": ["d/tools-config", "d/tools-config", "t/howto", "s/evergreen"],
            "reason": "duplicate domain tag",
        }
    )

    assert approved["decision"] == "approved"
    assert approved["valid"] is True
    assert invalid_axes["decision"] == "needs_retry"
    assert invalid_axes["valid"] is False
    assert any("t/ has 0" in error for error in invalid_axes["validation_errors"])
    assert malformed_rejection["decision"] == "needs_retry"
    assert malformed_rejection["valid"] is False
    assert duplicate_tags["decision"] == "needs_retry"
    assert "duplicate tags" in duplicate_tags["validation_errors"]


def test_normalize_tag_decision_preserves_local_consensus_authority_audit() -> None:
    authority = _semantic_authority("a" * 64)
    normalized = lint_repair.normalize_tag_decision(
        _authority_bound_tag_decision("approved", authority)
    )

    assert normalized["valid"] is True
    assert normalized["decision_policy"]["router_policy"] == authority["router"]
    assert normalized["local_consensus"]["status"] == "agreed"


def test_local_approved_tags_require_frontier_approval_before_apply(
    tmp_path: Path,
    monkeypatch,
) -> None:
    page_id = "test-page"
    page_path = tmp_path / "pages" / f"{page_id}.md"
    _page(page_path)
    queue_path = tmp_path / "review" / "lint-repair-queue.jsonl"
    _queue(queue_path, [_row(page_id)])
    monkeypatch.setattr(
        lint_repair.chronovisor_store,
        "find_page",
        lambda candidate: _find_only(candidate, page_id, page_path),
    )
    store = _store(tmp_path)
    budget = _budget()
    frontier_prompts: list[str] = []

    def frontier_review(prompt, _schema):
        frontier_prompts.append(prompt)
        return {
            "decision": "approved",
            "tags": VALID_TAGS,
            "reason": "frontier independently verified the proposal",
        }

    result = lint_repair.run_lint_repair(
        queue_file=queue_path,
        store=store,
        budget=budget,
        local_reviewer=lambda _prompt, _schema: {
            "decision": "approved",
            "tags": VALID_TAGS,
            "reason": "page is a configuration how-to",
        },
        frontier_reviewer=frontier_review,
        now=NOW,
    )

    meta, body = parse_frontmatter(page_path.read_text(encoding="utf-8"))
    item = store.list_items()[0]
    assert result["applied"] == 1
    assert result["escalated"] == 1
    assert result["rejected"] == 0
    assert result["quarantined"] == 0
    assert result["budget"]["used"] == {
        "local": 1,
        "frontier": 1,
        "mutation": 1,
        "raw_bytes": 0,
    }
    assert meta["tags"] == VALID_TAGS
    assert body == "\n# Page\n\nUseful content.\n"
    assert item["status"] == "applied"
    assert item["result"]["review_stage"] == "frontier"
    assert len(frontier_prompts) == 1
    assert "Tag review contract version: 2." in frontier_prompts[0]
    assert "<LOCAL_TAG_PROPOSAL_UNTRUSTED_JSON>" in frontier_prompts[0]
    assert "page is a configuration how-to" in frontier_prompts[0]


def test_tag_apply_preserves_correction_that_lands_before_locked_cas(
    tmp_path: Path,
    monkeypatch,
) -> None:
    page_path = tmp_path / "pages" / "race.md"
    original = _page(page_path)
    corrected = original.replace("Useful content.", "User-corrected content.")

    @contextmanager
    def correction_wins():
        page_path.write_text(corrected, encoding="utf-8")
        yield

    monkeypatch.setattr(lint_repair, "chronovisor_mutation_lock", correction_wins)

    result = lint_repair.apply_tags_cas(
        page_path,
        expected_text=original,
        tags=VALID_TAGS,
    )

    assert result["status"] == "cas_conflict"
    assert page_path.read_text(encoding="utf-8") == corrected


def test_local_approval_cannot_mutate_when_frontier_rejects(
    tmp_path: Path,
    monkeypatch,
) -> None:
    page_id = "local-is-not-final"
    page_path = tmp_path / "pages" / f"{page_id}.md"
    original = _page(page_path)
    queue_path = tmp_path / "review" / "lint-repair-queue.jsonl"
    _queue(queue_path, [_row(page_id)])
    monkeypatch.setattr(lint_repair.chronovisor_store, "find_page", lambda _page_id: page_path)
    store = _store(tmp_path)

    result = lint_repair.run_lint_repair(
        queue_file=queue_path,
        store=store,
        budget=_budget(),
        local_reviewer=lambda _prompt, _schema: {
            "decision": "approved",
            "tags": VALID_TAGS,
            "reason": "local proposal",
        },
        frontier_reviewer=lambda _prompt, _schema: {
            "decision": "rejected",
            "tags": [],
            "reason": "page evidence does not support the proposed taxonomy",
        },
        now=NOW,
    )

    item = store.list_items()[0]
    assert result["rejected"] == 1
    assert result["applied"] == 0
    assert result["budget"]["used"] == {
        "local": 1,
        "frontier": 1,
        "mutation": 0,
        "raw_bytes": 0,
    }
    assert page_path.read_text(encoding="utf-8") == original
    assert item["status"] == "rejected"
    assert item["result"]["decision"]["reason"].startswith("page evidence")
    assert lint_repair._review_artifact_path(store, str(item["key"])).exists()


def test_durable_frontier_verdict_is_reused_after_pre_apply_budget_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    page_id = "durable-frontier"
    page_path = tmp_path / "pages" / f"{page_id}.md"
    original = _page(page_path)
    queue_path = tmp_path / "review" / "lint-repair-queue.jsonl"
    _queue(queue_path, [_row(page_id)])
    monkeypatch.setattr(lint_repair.chronovisor_store, "find_page", lambda _page_id: page_path)
    store = _store(tmp_path)
    frontier_calls = 0

    def frontier_review(_prompt, _schema):
        nonlocal frontier_calls
        frontier_calls += 1
        return {
            "decision": "approved",
            "tags": VALID_TAGS,
            "reason": "authoritative frontier verdict",
        }

    first = lint_repair.run_lint_repair(
        queue_file=queue_path,
        store=store,
        budget=CycleBudget(
            max_local_calls=1,
            max_frontier_calls=1,
            max_mutations=0,
            max_elapsed_seconds=60,
        ),
        local_reviewer=lambda _prompt, _schema: {
            "decision": "approved",
            "tags": VALID_TAGS,
            "reason": "local proposal",
        },
        frontier_reviewer=frontier_review,
        now=NOW,
    )

    item = store.list_items()[0]
    artifact = lint_repair._review_artifact_path(store, str(item["key"]))
    assert first["results"][0]["status"] == "budget_exhausted"
    assert item["status"] == "frontier_retry"
    assert artifact.exists()
    artifact_payload = json.loads(artifact.read_text(encoding="utf-8"))
    assert artifact_payload["schema_version"] == 2
    assert artifact_payload["authority"] == {
        "source": "injected_reviewer_boundary",
        "authority_version": 1,
        "lane": lint_repair.TAG_REPAIR_DECISION_LANE,
    }
    assert page_path.read_text(encoding="utf-8") == original
    assert frontier_calls == 1

    second = lint_repair.run_lint_repair(
        queue_file=queue_path,
        store=store,
        budget=CycleBudget(
            max_local_calls=0,
            max_frontier_calls=0,
            max_mutations=1,
            max_elapsed_seconds=60,
        ),
        local_reviewer=_never,
        frontier_reviewer=_never,
        now=NOW + timedelta(seconds=901),
    )

    meta, _body = parse_frontmatter(page_path.read_text(encoding="utf-8"))
    assert second["applied"] == 1
    assert second["budget"]["used"]["frontier"] == 0
    assert second["budget"]["used"]["mutation"] == 1
    assert meta["tags"] == VALID_TAGS
    assert frontier_calls == 1


def test_stale_durable_verdict_is_not_reused_across_authority_epoch(
    tmp_path: Path,
    monkeypatch,
) -> None:
    page_id = "stale-authority"
    page_path = tmp_path / "pages" / f"{page_id}.md"
    original = _page(page_path)
    queue_path = tmp_path / "review" / "lint-repair-queue.jsonl"
    _queue(queue_path, [_row(page_id)])
    monkeypatch.setattr(lint_repair.chronovisor_store, "find_page", lambda _page_id: page_path)
    store = _store(tmp_path)
    authority_a = _semantic_authority("a" * 64)
    authority_b = _semantic_authority("b" * 64)
    current = authority_a
    reviewer_calls = 0

    def authority(_lane, *, injected_reviewer=False):
        del injected_reviewer
        return current, None

    def reviewer(_prompt, _schema):
        nonlocal reviewer_calls
        reviewer_calls += 1
        return _authority_bound_tag_decision("approved", current)

    monkeypatch.setattr(lint_repair, "current_semantic_authority", authority)
    first = lint_repair.run_lint_repair(
        queue_file=queue_path,
        store=store,
        budget=CycleBudget(
            max_local_calls=1,
            max_frontier_calls=1,
            max_mutations=0,
            max_elapsed_seconds=60,
        ),
        local_reviewer=lambda _prompt, _schema: {
            "decision": "approved",
            "tags": VALID_TAGS,
            "reason": "local proposal",
        },
        frontier_reviewer=reviewer,
        now=NOW,
    )
    assert first["results"][0]["status"] == "budget_exhausted"
    assert reviewer_calls == 1
    current = authority_b

    second = lint_repair.run_lint_repair(
        queue_file=queue_path,
        store=store,
        budget=CycleBudget(
            max_local_calls=0,
            max_frontier_calls=1,
            max_mutations=0,
            max_elapsed_seconds=60,
        ),
        local_reviewer=_never,
        frontier_reviewer=reviewer,
        now=NOW + timedelta(seconds=901),
    )

    artifact = lint_repair._review_artifact_path(store, store.list_items()[0]["key"])
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    assert second["results"][0]["status"] == "budget_exhausted"
    assert reviewer_calls == 2
    assert payload["authority"] == authority_b
    assert page_path.read_text(encoding="utf-8") == original


@pytest.mark.parametrize("decision", ["approved", "rejected"])
def test_authority_change_before_effect_blocks_page_and_terminal_transition(
    tmp_path: Path,
    monkeypatch,
    decision: str,
) -> None:
    page_id = f"authority-race-{decision}"
    page_path = tmp_path / "pages" / f"{page_id}.md"
    original = _page(page_path)
    queue_path = tmp_path / "review" / "lint-repair-queue.jsonl"
    _queue(queue_path, [_row(page_id)])
    monkeypatch.setattr(lint_repair.chronovisor_store, "find_page", lambda _page_id: page_path)
    store = _store(tmp_path)
    authority_a = _semantic_authority("a" * 64)
    authority_b = _semantic_authority("b" * 64)
    authority_calls = 0

    def changing_authority(_lane, *, injected_reviewer=False):
        nonlocal authority_calls
        del injected_reviewer
        authority_calls += 1
        return (authority_a if authority_calls <= 2 else authority_b), None

    monkeypatch.setattr(
        lint_repair,
        "current_semantic_authority",
        changing_authority,
    )
    result = lint_repair.run_lint_repair(
        queue_file=queue_path,
        store=store,
        budget=_budget(),
        local_reviewer=lambda _prompt, _schema: {
            "decision": "approved",
            "tags": VALID_TAGS,
            "reason": "local proposal",
        },
        frontier_reviewer=lambda _prompt, _schema: _authority_bound_tag_decision(
            decision,
            authority_a,
        ),
        now=NOW,
    )

    item = store.list_items()[0]
    assert result["results"][0]["status"] == "frontier_retry"
    assert item["status"] == "frontier_retry"
    assert item["last_failure_class"] == "decision_authority_changed"
    assert page_path.read_text(encoding="utf-8") == original


def test_production_verdict_without_policy_audit_cannot_be_persisted_or_applied(
    tmp_path: Path,
    monkeypatch,
) -> None:
    page_id = "missing-policy-audit"
    page_path = tmp_path / "pages" / f"{page_id}.md"
    original = _page(page_path)
    queue_path = tmp_path / "review" / "lint-repair-queue.jsonl"
    _queue(queue_path, [_row(page_id)])
    monkeypatch.setattr(lint_repair.chronovisor_store, "find_page", lambda _page_id: page_path)
    authority = _semantic_authority("a" * 64)
    monkeypatch.setattr(
        lint_repair,
        "current_semantic_authority",
        lambda _lane, *, injected_reviewer=False: (authority, None),
    )
    store = _store(tmp_path)

    result = lint_repair.run_lint_repair(
        queue_file=queue_path,
        store=store,
        budget=_budget(),
        local_reviewer=lambda _prompt, _schema: {
            "decision": "approved",
            "tags": VALID_TAGS,
            "reason": "local proposal",
        },
        frontier_reviewer=lambda _prompt, _schema: {
            "decision": "approved",
            "tags": VALID_TAGS,
            "reason": "missing required production policy audit",
        },
        now=NOW,
    )

    item = store.list_items()[0]
    assert result["results"][0]["status"] == "frontier_error"
    assert item["last_failure_class"] == "decision_authority_changed"
    assert not lint_repair._review_artifact_path(store, item["key"]).exists()
    assert page_path.read_text(encoding="utf-8") == original


def test_frontier_artifact_write_failure_is_fail_closed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    page_id = "artifact-write-failure"
    page_path = tmp_path / "pages" / f"{page_id}.md"
    original = _page(page_path)
    queue_path = tmp_path / "review" / "lint-repair-queue.jsonl"
    _queue(queue_path, [_row(page_id)])
    monkeypatch.setattr(lint_repair.chronovisor_store, "find_page", lambda _page_id: page_path)

    def fail_artifact(*_args, **_kwargs):
        raise OSError("disk unavailable")

    monkeypatch.setattr(lint_repair, "_write_frontier_review_artifact", fail_artifact)
    store = _store(tmp_path)
    result = lint_repair.run_lint_repair(
        queue_file=queue_path,
        store=store,
        budget=_budget(),
        local_reviewer=lambda _prompt, _schema: {
            "decision": "approved",
            "tags": VALID_TAGS,
            "reason": "local proposal",
        },
        frontier_reviewer=lambda _prompt, _schema: {
            "decision": "approved",
            "tags": VALID_TAGS,
            "reason": "frontier approval",
        },
        now=NOW,
    )

    item = store.list_items()[0]
    assert result["results"][0]["status"] == "frontier_error"
    assert result["applied"] == 0
    assert result["budget"]["used"]["mutation"] == 0
    assert item["status"] == "frontier_retry"
    assert item["last_failure_class"] == "review_artifact_write_error"
    assert page_path.read_text(encoding="utf-8") == original


def test_invalid_local_proposal_cannot_be_replaced_by_review_panel(
    tmp_path: Path,
    monkeypatch,
) -> None:
    page_id = "frontier-page"
    page_path = tmp_path / "pages" / f"{page_id}.md"
    _page(page_path)
    queue_path = tmp_path / "review" / "lint-repair-queue.jsonl"
    _queue(queue_path, [_row(page_id, issue_type="tag_count_violation")])
    monkeypatch.setattr(lint_repair.chronovisor_store, "find_page", lambda _page_id: page_path)
    store = _store(tmp_path)
    budget = _budget()

    result = lint_repair.run_lint_repair(
        queue_file=queue_path,
        store=store,
        budget=budget,
        local_reviewer=lambda _prompt, _schema: {
            "decision": "approved",
            "tags": ["d/tools-config"],
            "reason": "malformed local proposal",
        },
        frontier_reviewer=lambda _prompt, _schema: {
            "decision": "approved",
            "tags": VALID_TAGS,
            "reason": "frontier supplied all required axes",
        },
        now=NOW,
    )

    meta, _body = parse_frontmatter(page_path.read_text(encoding="utf-8"))
    item = store.list_items()[0]
    assert result["applied"] == 0
    assert result["escalated"] == 1
    assert result["budget"]["used"] == {
        "local": 1,
        "frontier": 1,
        "mutation": 0,
        "raw_bytes": 0,
    }
    assert "tags" not in meta
    assert item["status"] in {"frontier_retry", "quarantined"}
    assert item["local_attempts"] == 1
    assert item["frontier_attempts"] == 1
    assert result["results"][0]["status"] == "frontier_retry"


def test_frontier_rejection_is_terminal_and_does_not_mutate_page(
    tmp_path: Path,
    monkeypatch,
) -> None:
    page_id = "rejected-page"
    page_path = tmp_path / "pages" / f"{page_id}.md"
    original = _page(page_path)
    queue_path = tmp_path / "review" / "lint-repair-queue.jsonl"
    _queue(queue_path, [_row(page_id)])
    monkeypatch.setattr(lint_repair.chronovisor_store, "find_page", lambda _page_id: page_path)
    store = _store(tmp_path)

    result = lint_repair.run_lint_repair(
        queue_file=queue_path,
        store=store,
        budget=_budget(),
        local_reviewer=lambda _prompt, _schema: {
            "decision": "uncertain",
            "tags": [],
            "reason": "excerpt is ambiguous",
        },
        frontier_reviewer=lambda _prompt, _schema: {
            "decision": "rejected",
            "tags": [],
            "reason": "no defensible semantic tags",
        },
        now=NOW,
    )

    item = store.list_items()[0]
    assert result["rejected"] == 1
    assert result["applied"] == 0
    assert result["budget"]["used"]["mutation"] == 0
    assert page_path.read_text(encoding="utf-8") == original
    assert item["status"] == "rejected"
    assert item["result"]["action"] == "tag_repair_rejected"


def test_tag_no_quorum_reuses_hold_until_exact_authority_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    page_id = "semantic-split"
    page_path = tmp_path / "pages" / f"{page_id}.md"
    original = _page(page_path)
    queue_path = tmp_path / "review" / "lint-repair-queue.jsonl"
    _queue(queue_path, [_row(page_id)])
    monkeypatch.setattr(lint_repair.chronovisor_store, "find_page", lambda _page_id: page_path)
    store = _store(tmp_path)
    authority_a = semantic_authority(
        lint_repair.TAG_REPAIR_DECISION_LANE,
        schema_name="lint_tag_repair",
    )
    authority_b = semantic_authority(
        lint_repair.TAG_REPAIR_DECISION_LANE,
        schema_name="lint_tag_repair",
        artifact_sha256="9" * 64,
    )
    current = [authority_a]
    monkeypatch.setattr(
        lint_repair,
        "current_semantic_authority",
        lambda *_args, **_kwargs: (current[0], None),
    )
    calls = 0

    def frontier(_prompt: str, _schema: dict) -> dict:
        nonlocal calls
        calls += 1
        return _semantic_no_quorum_tag_decision(current[0])

    kwargs = {
        "queue_file": queue_path,
        "store": store,
        "local_reviewer": lambda _prompt, _schema: {
            "decision": "approved",
            "tags": VALID_TAGS,
            "reason": "exact local proposal",
        },
        "frontier_reviewer": frontier,
    }
    first = lint_repair.run_lint_repair(**kwargs, now=NOW)
    same_epoch = lint_repair.run_lint_repair(
        **kwargs,
        now=NOW + timedelta(seconds=1),
    )
    current[0] = authority_b
    changed_authority = lint_repair.run_lint_repair(
        **kwargs,
        now=NOW + timedelta(seconds=2),
    )
    current[0] = authority_a
    restored_authority = lint_repair.run_lint_repair(
        **{
            **kwargs,
            "frontier_reviewer": lambda _prompt, _schema: (_ for _ in ()).throw(
                AssertionError("A-B-A must restore the A hold without resampling")
            ),
        },
        now=NOW + timedelta(seconds=3),
    )

    assert first["quarantined"] == 1
    assert first["deferred"] == 0
    assert same_epoch["terminal_skipped"] == 1
    assert changed_authority["quarantined"] == 1
    assert restored_authority["quarantined"] == 1
    assert restored_authority["results"][0]["restored_semantic_hold"] is True
    assert calls == 2
    item = store.list_items()[0]
    assert item["frontier_attempts"] == 0
    assert item["result"]["semantic_hold"]["authority"] == authority_a
    assert [hold["authority"] for hold in item["result"]["semantic_hold_history"]] == [
        authority_b
    ]
    assert page_path.read_text(encoding="utf-8") == original


def test_frontier_auth_failure_is_the_only_kind_that_requires_a_human(
    tmp_path: Path,
    monkeypatch,
) -> None:
    page_id = "auth-page"
    page_path = tmp_path / "pages" / f"{page_id}.md"
    original = _page(page_path)
    queue_path = tmp_path / "review" / "lint-repair-queue.jsonl"
    _queue(queue_path, [_row(page_id)])
    monkeypatch.setattr(lint_repair.chronovisor_store, "find_page", lambda _page_id: page_path)
    store = _store(tmp_path)

    result = lint_repair.run_lint_repair(
        queue_file=queue_path,
        store=store,
        budget=_budget(),
        local_reviewer=lambda _prompt, _schema: {
            "decision": "uncertain",
            "tags": [],
            "reason": "local model cannot decide",
        },
        frontier_reviewer=lambda _prompt, _schema: {
            "decision": "needs_retry",
            "summary": "sign-in required",
            "frontier_failure": {"failure_class": "auth_required"},
            "human_required": True,
        },
        now=NOW,
    )

    item = store.list_items()[0]
    assert item["status"] == "human_required"
    assert item["human_required"] is True
    assert item["last_failure_class"] == "auth_required"
    assert result["human_required"] == 1
    assert result["results"][0]["status"] == "human_required"
    assert result["quarantined"] == 0
    assert page_path.read_text(encoding="utf-8") == original


def test_stale_is_observed_and_duplicate_orphan_are_terminally_routed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    queue_path = tmp_path / "review" / "lint-repair-queue.jsonl"
    _queue(
        queue_path,
        [
            _row("stale-page", lane="monitor", issue_type="stale"),
            _row("duplicate-page", lane="review", issue_type="duplicate"),
            _row("orphan-page", lane="review", issue_type="orphan"),
        ],
    )
    monkeypatch.setattr(lint_repair.chronovisor_store, "find_page", lambda _page_id: None)
    store = _store(tmp_path)

    result = lint_repair.run_lint_repair(
        queue_file=queue_path,
        store=store,
        budget=_budget(),
        local_reviewer=_never,
        frontier_reviewer=_never,
        now=NOW,
    )

    items = {item["source_id"].split(":", 1)[0]: item for item in store.list_items()}
    assert result["processed"] == 3
    assert result["observed"] == 1
    assert result["routed"] == 2
    assert result["quarantined"] == 0
    assert result["budget"]["used"] == {
        "local": 0,
        "frontier": 0,
        "mutation": 0,
        "raw_bytes": 0,
    }
    assert items["stale"]["status"] == "applied"
    assert items["stale"]["result"]["action"] == "observed"
    assert items["duplicate"]["status"] == "applied"
    assert items["duplicate"]["result"]["target_lane"] == "duplicate_review"
    assert items["orphan"]["status"] == "applied"
    assert items["orphan"]["result"]["target_lane"] == "orphan_link"


def test_dry_run_is_byte_for_byte_read_only_and_calls_no_models(
    tmp_path: Path,
    monkeypatch,
) -> None:
    page_id = "dry-run-page"
    page_path = tmp_path / "pages" / f"{page_id}.md"
    page_before = _page(page_path).encode()
    queue_path = tmp_path / "review" / "lint-repair-queue.jsonl"
    queue_before = _queue(queue_path, [_row(page_id)])
    monkeypatch.setattr(lint_repair.chronovisor_store, "find_page", lambda _page_id: page_path)
    store = _store(tmp_path)
    budget = _budget()

    result = lint_repair.run_lint_repair(
        queue_file=queue_path,
        store=store,
        budget=budget,
        dry_run=True,
        local_reviewer=_never,
        frontier_reviewer=_never,
        now=NOW,
    )

    assert result["status"] == "dry_run"
    assert result["results"][0]["status"] == "would_review_tags"
    assert result["budget"]["used"] == {
        "local": 0,
        "frontier": 0,
        "mutation": 0,
        "raw_bytes": 0,
    }
    assert page_path.read_bytes() == page_before
    assert queue_path.read_bytes() == queue_before
    assert not store.state_file.exists()
    assert not store.events_file.exists()
    assert not store.lock_file.exists()


def test_max_items_bounds_queue_work(tmp_path: Path, monkeypatch) -> None:
    queue_path = tmp_path / "review" / "lint-repair-queue.jsonl"
    _queue(
        queue_path,
        [
            _row("first", lane="monitor", issue_type="stale"),
            _row("second", lane="monitor", issue_type="stale"),
        ],
    )
    monkeypatch.setattr(lint_repair.chronovisor_store, "find_page", lambda _page_id: None)
    store = _store(tmp_path)

    result = lint_repair.run_lint_repair(
        queue_file=queue_path,
        store=store,
        budget=_budget(),
        max_items=1,
        local_reviewer=_never,
        frontier_reviewer=_never,
        now=NOW,
    )

    assert result["bounded"] == 1
    assert result["remaining_unseen"] == 1
    assert result["processed"] == 1
    assert len(store.list_items()) == 1

    second = lint_repair.run_lint_repair(
        queue_file=queue_path,
        store=store,
        budget=_budget(),
        max_items=1,
        local_reviewer=_never,
        frontier_reviewer=_never,
        now=NOW,
    )

    assert second["bounded"] == 1
    assert second["rows_scanned"] == 2
    assert second["remaining_unseen"] == 0
    assert second["terminal_skipped"] == 1
    assert second["processed"] == 1
    assert second["observed"] == 1
    assert len(store.list_items()) == 2


def test_deterministic_rows_run_before_model_backed_rows(
    tmp_path: Path, monkeypatch
) -> None:
    page_path = tmp_path / "pages" / "heavy.md"
    _page(page_path)
    queue_path = tmp_path / "review" / "lint-repair-queue.jsonl"
    _queue(
        queue_path,
        [
            _row("heavy"),
            _row("stale", lane="monitor", issue_type="stale"),
            _row("orphan", lane="review", issue_type="orphan"),
        ],
    )
    monkeypatch.setattr(
        lint_repair.chronovisor_store,
        "find_page",
        lambda page_id: page_path if page_id == "heavy" else None,
    )
    store = _store(tmp_path)

    result = lint_repair.run_lint_repair(
        queue_file=queue_path,
        store=store,
        budget=_budget(),
        max_items=2,
        local_reviewer=_never,
        frontier_reviewer=_never,
        now=NOW,
    )

    assert result["processed"] == 2
    assert result["observed"] == 1
    assert result["routed"] == 1
    assert [row["status"] for row in result["results"]] == ["observed", "routed"]


def test_cas_conflict_quarantines_without_overwriting_concurrent_change(
    tmp_path: Path,
    monkeypatch,
) -> None:
    page_id = "cas-page"
    page_path = tmp_path / "pages" / f"{page_id}.md"
    _page(page_path)
    queue_path = tmp_path / "review" / "lint-repair-queue.jsonl"
    _queue(queue_path, [_row(page_id)])
    monkeypatch.setattr(lint_repair.chronovisor_store, "find_page", lambda _page_id: page_path)
    store = _store(tmp_path)
    concurrent = (
        "---\ntitle: Concurrent\nupdated: 2026-07-10\n---\n\n# Changed elsewhere\n"
    )

    def local_review(_prompt, _schema):
        return {
            "decision": "approved",
            "tags": VALID_TAGS,
            "reason": "valid proposal for the old preimage",
        }

    def frontier_review(_prompt, _schema):
        page_path.write_text(concurrent, encoding="utf-8")
        return {
            "decision": "approved",
            "tags": VALID_TAGS,
            "reason": "frontier approved the old preimage",
        }

    result = lint_repair.run_lint_repair(
        queue_file=queue_path,
        store=store,
        budget=_budget(),
        local_reviewer=local_review,
        frontier_reviewer=frontier_review,
        now=NOW,
    )

    item = store.list_items()[0]
    assert result["quarantined"] == 1
    assert result["applied"] == 0
    assert item["status"] == "quarantined"
    assert item["quarantine_reason"] == "tag_repair_cas_conflict"
    assert page_path.read_text(encoding="utf-8") == concurrent


def test_backoff_row_does_not_starve_later_actionable_row(
    tmp_path: Path,
    monkeypatch,
) -> None:
    page_path = tmp_path / "pages" / "retrying.md"
    _page(page_path)
    queue_path = tmp_path / "review" / "lint-repair-queue.jsonl"
    _queue(
        queue_path,
        [
            _row("retrying"),
            _row("later", lane="unsupported", issue_type="unknown"),
        ],
    )
    monkeypatch.setattr(
        lint_repair.chronovisor_store,
        "find_page",
        lambda page_id: page_path if page_id == "retrying" else None,
    )
    store = _store(
        tmp_path,
        policy=RetryPolicy(
            max_local_attempts=2,
            local_base_delay_seconds=3600,
        ),
    )

    first = lint_repair.run_lint_repair(
        queue_file=queue_path,
        store=store,
        budget=_budget(),
        max_items=1,
        local_reviewer=_never,
        frontier_reviewer=_never,
        now=NOW,
    )
    second = lint_repair.run_lint_repair(
        queue_file=queue_path,
        store=store,
        budget=_budget(),
        max_items=1,
        local_reviewer=_never,
        frontier_reviewer=_never,
        now=NOW,
    )

    assert first["results"][0]["status"] == "local_error"
    assert second["bounded"] == 1
    assert second["rows_scanned"] == 2
    assert second["processed"] == 1
    assert second["deferred"] == 1
    assert second["quarantined"] == 1
    assert [result["status"] for result in second["results"]] == [
        "deferred",
        "quarantined",
    ]


def test_existing_valid_tags_finish_without_calling_a_model(
    tmp_path: Path, monkeypatch
) -> None:
    page_id = "already-valid"
    page_path = tmp_path / "pages" / f"{page_id}.md"
    original = _page(page_path, tags=VALID_TAGS)
    queue_path = tmp_path / "review" / "lint-repair-queue.jsonl"
    _queue(queue_path, [_row(page_id)])
    monkeypatch.setattr(lint_repair.chronovisor_store, "find_page", lambda _page_id: page_path)
    store = _store(tmp_path)

    result = lint_repair.run_lint_repair(
        queue_file=queue_path,
        store=store,
        budget=_budget(),
        local_reviewer=_never,
        frontier_reviewer=_never,
        now=NOW,
    )

    item = store.list_items()[0]
    assert result["applied"] == 1
    assert result["budget"]["used"]["local"] == 0
    assert result["budget"]["used"]["mutation"] == 0
    assert item["status"] == "applied"
    assert item["result"]["action"] == "already_resolved_observed"
    assert item["result"]["semantic_effect"] is False
    assert item["result"]["recovery_only"] is False
    assert page_path.read_text(encoding="utf-8") == original


def test_exact_already_applied_recovery_only_finalizes_bookkeeping(
    tmp_path: Path,
    monkeypatch,
) -> None:
    page_id = "exact-applied-recovery"
    page_path = tmp_path / "pages" / f"{page_id}.md"
    _page(page_path)
    queue_path = tmp_path / "review" / "lint-repair-queue.jsonl"
    _queue(queue_path, [_row(page_id)])
    monkeypatch.setattr(lint_repair.chronovisor_store, "find_page", lambda _page_id: page_path)
    store = _store(tmp_path)
    real_complete = store.complete

    def fail_after_page_write(key, status, **kwargs):
        result = kwargs.get("result")
        if (
            status == "applied"
            and isinstance(result, dict)
            and result.get("action") == "tag_repair"
        ):
            raise RuntimeError("simulated crash after exact page write")
        return real_complete(key, status, **kwargs)

    monkeypatch.setattr(store, "complete", fail_after_page_write)
    with pytest.raises(RuntimeError, match="simulated crash"):
        lint_repair.run_lint_repair(
            queue_file=queue_path,
            store=store,
            budget=_budget(),
            local_reviewer=lambda _prompt, _schema: {
                "decision": "approved",
                "tags": VALID_TAGS,
                "reason": "local proposal",
            },
            frontier_reviewer=lambda _prompt, _schema: {
                "decision": "approved",
                "tags": VALID_TAGS,
                "reason": "reviewed exact proposal",
            },
            now=NOW,
        )

    monkeypatch.setattr(store, "complete", real_complete)
    second = lint_repair.run_lint_repair(
        queue_file=queue_path,
        store=store,
        budget=_budget(),
        local_reviewer=_never,
        frontier_reviewer=_never,
        now=NOW + timedelta(seconds=1),
    )

    recovered = second["results"][0]
    assert recovered["status"] == "exact_already_applied_recovery"
    assert recovered["state"]["result"]["action"] == ("exact_already_applied_recovery")
    assert recovered["state"]["result"]["recovery_only"] is True
    assert recovered["state"]["result"]["semantic_effect"] is False
    meta, _body = parse_frontmatter(page_path.read_text(encoding="utf-8"))
    assert meta["tags"] == VALID_TAGS


def test_missing_page_is_rejected_once(tmp_path: Path, monkeypatch) -> None:
    queue_path = tmp_path / "review" / "lint-repair-queue.jsonl"
    _queue(queue_path, [_row("missing-page")])
    monkeypatch.setattr(lint_repair.chronovisor_store, "find_page", lambda _page_id: None)
    store = _store(tmp_path)

    result = lint_repair.run_lint_repair(
        queue_file=queue_path,
        store=store,
        budget=_budget(),
        local_reviewer=_never,
        frontier_reviewer=_never,
        now=NOW,
    )

    assert result["processed"] == 1
    assert result["rejected"] == 1
    assert store.list_items()[0]["status"] == "rejected"
