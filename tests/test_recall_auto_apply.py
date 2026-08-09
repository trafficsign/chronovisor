from __future__ import annotations

import hashlib
import json
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from chronovisor.core import page_mutation, store
from chronovisor.core.frontmatter import parse as parse_frontmatter
from chronovisor.core.link_fix import atomic_write
from chronovisor.decision.decision_router import canonical_agreement_signature
from chronovisor.decision.decision_schema_manifest import production_decision_schemas
from chronovisor.ingest import recall_hints
from chronovisor.ingest.convergence import CycleBudget
from chronovisor.recall import recall_auto_apply
from chronovisor.recall.recall_runtime import RecallPolicy, collect_context


@pytest.fixture(autouse=True)
def _frontier_approves_existing_auto_apply_tests(monkeypatch) -> None:
    monkeypatch.setattr(
        recall_auto_apply.decision_authority,
        "current_semantic_authority",
        lambda lane, **_kwargs: (
            {
                "source": "injected_reviewer_boundary",
                "authority_version": 1,
                "lane": lane,
            },
            None,
        ),
    )
    monkeypatch.setattr(
        recall_auto_apply,
        "review_auto_apply_with_frontier",
        lambda _proposal, **_kwargs: {
            "decision": "approved",
            "summary": "test frontier approval",
        },
    )


def _production_authority(epoch: str) -> dict[str, object]:
    digest = (epoch[:1] or "a") * 64
    return {
        "source": "adopted_local_consensus",
        "authority_version": 1,
        "lane": "recall_auto_apply",
        "lane_contract_sha256": "1" * 64,
        "lane_contract_manifest_sha256": "2" * 64,
        "lane_contract_case_manifest_sha256": "3" * 64,
        "policy": {
            "kind": "semantic",
            "schema_name": "generic_decision",
            "mode": "enabled",
            "error": None,
        },
        "router": {
            "source": "adopted_artifact",
            "artifact_sha256": digest,
            "error": None,
            "models": ["primary", "challenger", "tie-break"],
        },
    }


def _review(decision: str, authority: dict[str, object]) -> dict[str, object]:
    policy = dict(authority["policy"])
    policy["router_policy"] = authority["router"]
    review: dict[str, object] = {
        "decision": decision,
        "summary": "authority-specific verdict",
        "tests_run": [],
        "commit": None,
        "committed": False,
        "pushed": False,
        "risk": None,
        "notes": None,
        "decision_policy": policy,
    }
    schema_name = str(policy["schema_name"])
    signature = canonical_agreement_signature(
        review,
        schema=production_decision_schemas()[schema_name],
    )
    agreement = hashlib.sha256(signature.encode("utf-8")).hexdigest()
    router = authority["router"]
    assert isinstance(router, dict)
    models = router["models"]
    assert isinstance(models, list)
    review["local_consensus"] = {
        "status": "agreed",
        "ok": True,
        "agreement_sha256": agreement,
        "failure_class": None,
        "quarantine_reason": None,
        "votes": [
            {
                "valid": True,
                "signature_sha256": agreement,
                "invalid_reason": None,
                "model": models[0],
                "role": "primary",
            },
            {
                "valid": True,
                "signature_sha256": agreement,
                "invalid_reason": None,
                "model": models[1],
                "role": "challenger",
            },
        ],
    }
    return review


def _page(root: Path, page_id: str, body: str = "Recall hook body") -> Path:
    path = root / "pages" / "ai" / f"{page_id}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"---\ntitle: {page_id}\nupdated: 2026-06-02\ntags: [d/tools-config, t/analysis, s/2026]\n---\n\n{body}\n",
        encoding="utf-8",
    )
    return path


def _candidate(
    action_type: str, *, page_id: str, payload: dict[str, object] | None = None
) -> dict[str, object]:
    return {
        "kind": "missed_candidate",
        "source": "auditor",
        "host": "codex",
        "prompt": "昨日の recall hook の続き",
        "expected_pages": [page_id],
        "ref": "decision-1",
        "reason_code": "gate_missed",
        "missing_signal": "recall hook",
        "normalize_key": f"gate_missed:recall-hook:{page_id}",
        "action_type": action_type,
        "action_payload": payload or {},
        "lane": "auto",
        "auto_apply_eligible": True,
    }


def test_bounded_page_evidence_seals_snapshot_status_and_full_hash(
    tmp_path, monkeypatch
) -> None:
    pages_root = tmp_path / "wiki"
    path = _page(pages_root, "target-page", body="0123456789")
    monkeypatch.setattr(store, "CHRONOVISOR_ROOT", pages_root)
    monkeypatch.setattr(store, "PAGES_DIR", pages_root / "pages")

    evidence = recall_auto_apply._bounded_page_evidence("target-page", max_chars=8)

    assert evidence["snapshot_status"] == "verified"
    assert evidence["exists"] is True
    assert evidence["content_truncated"] is True
    assert evidence["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_query_hint_auto_apply_feeds_runtime_context(tmp_path, monkeypatch) -> None:
    pages_root = tmp_path / "wiki"
    _page(pages_root, "claude-code-recall-hook-implementation")
    hints_file = tmp_path / "query-hints.json"
    monkeypatch.setattr(store, "CHRONOVISOR_ROOT", pages_root)
    monkeypatch.setattr(store, "PAGES_DIR", pages_root / "pages")
    monkeypatch.setattr(recall_hints, "QUERY_HINTS_FILE", hints_file)

    result = recall_auto_apply.apply_feedback_records(
        [_candidate("query_hint", page_id="claude-code-recall-hook-implementation")],
        policy=recall_auto_apply.AutoApplyPolicy(min_count=1),
        log_file=tmp_path / "auto-apply.jsonl",
    )

    assert result["status"] == "ok"
    assert result["actions"][0]["status"] == "applied"
    assert hints_file.exists()

    context = collect_context(
        ["昨日の recall hook の続き"],
        "read",
        RecallPolicy(max_pages=1, semantic=False),
    )

    assert [item.page_id for item in context] == [
        "claude-code-recall-hook-implementation"
    ]


def test_query_hint_ignores_generic_context_tokens() -> None:
    hint = {
        "page_id": "wrong-page",
        "query": "old broad prompt",
        "query_key": "old broad prompt",
        "tokens": ["assistant", "codex", "context", "project", "wiki"],
    }

    assert not recall_hints.hint_matches_query(hint, "codex context window 切り分け")


def test_auto_apply_min_count_groups_by_normalize_key(tmp_path, monkeypatch) -> None:
    pages_root = tmp_path / "wiki"
    _page(pages_root, "claude-code-recall-hook-implementation")
    monkeypatch.setattr(store, "CHRONOVISOR_ROOT", pages_root)
    monkeypatch.setattr(store, "PAGES_DIR", pages_root / "pages")
    monkeypatch.setattr(recall_hints, "QUERY_HINTS_FILE", tmp_path / "query-hints.json")

    one = [_candidate("query_hint", page_id="claude-code-recall-hook-implementation")]
    assert (
        recall_auto_apply.apply_feedback_records(
            one,
            policy=recall_auto_apply.AutoApplyPolicy(min_count=2),
            log_file=tmp_path / "auto-apply.jsonl",
        )["actions"]
        == []
    )

    two = one + [
        _candidate("query_hint", page_id="claude-code-recall-hook-implementation")
    ]
    assert (
        recall_auto_apply.apply_feedback_records(
            two,
            policy=recall_auto_apply.AutoApplyPolicy(min_count=2),
            log_file=tmp_path / "auto-apply.jsonl",
        )["actions"][0]["status"]
        == "applied"
    )


def test_page_tag_auto_apply_patches_frontmatter(tmp_path, monkeypatch) -> None:
    pages_root = tmp_path / "wiki"
    page = _page(pages_root, "chronovisor-recall-audit-architecture")
    monkeypatch.setattr(store, "CHRONOVISOR_ROOT", pages_root)
    monkeypatch.setattr(store, "PAGES_DIR", pages_root / "pages")

    record = _candidate(
        "page_tag",
        page_id="chronovisor-recall-audit-architecture",
        payload={"tag": "d/theory"},
    )
    result = recall_auto_apply.apply_feedback_records(
        [record],
        policy=recall_auto_apply.AutoApplyPolicy(min_count=1),
        log_file=tmp_path / "auto-apply.jsonl",
    )
    meta, _body = parse_frontmatter(page.read_text(encoding="utf-8"))

    assert result["actions"][0]["status"] == "applied"
    assert "d/theory" in meta["tags"]


@pytest.mark.parametrize(
    ("action_type", "payload"),
    [
        ("query_hint", {"query": "exact missing context"}),
        ("alias", {"alias": "new-recall-alias"}),
        ("page_tag", {"tag": "d/theory"}),
    ],
)
def test_frontier_rejection_blocks_every_auto_mutation_and_is_durable(
    action_type,
    payload,
    tmp_path,
    monkeypatch,
) -> None:
    from chronovisor.core import alias_store

    pages_root = tmp_path / "wiki"
    page = _page(pages_root, "target-page")
    before = page.read_bytes()
    hints_file = tmp_path / "query-hints.json"
    monkeypatch.setattr(store, "CHRONOVISOR_ROOT", pages_root)
    monkeypatch.setattr(store, "PAGES_DIR", pages_root / "pages")
    monkeypatch.setattr(recall_hints, "QUERY_HINTS_FILE", hints_file)
    seen: list[dict[str, object]] = []

    def reject(proposal, **_kwargs):
        seen.append(proposal)
        return {"decision": "rejected", "summary": "mapping is unsupported"}

    result = recall_auto_apply.apply_feedback_records(
        [_candidate(action_type, page_id="target-page", payload=payload)],
        policy=recall_auto_apply.AutoApplyPolicy(min_count=1),
        log_file=tmp_path / "auto-apply.jsonl",
        review_dir=tmp_path / "reviews",
        frontier_reviewer=reject,
    )

    action = result["actions"][0]
    assert action["status"] == "frontier_rejected"
    assert action["convergence_status"] == "rejected"
    assert seen[0]["effective_action"] == action_type
    artifact = Path(action["frontier_artifact"])
    assert artifact.exists()
    assert json.loads(artifact.read_text())["review"]["decision"] == "rejected"
    assert page.read_bytes() == before
    assert not hints_file.exists()
    assert "new-recall-alias" not in alias_store.load_aliases()


def test_approved_frontier_artifact_survives_mutation_budget_deferral(
    tmp_path,
    monkeypatch,
) -> None:
    pages_root = tmp_path / "wiki"
    _page(pages_root, "target-page")
    hints_file = tmp_path / "query-hints.json"
    log_file = tmp_path / "auto-apply.jsonl"
    review_dir = tmp_path / "reviews"
    monkeypatch.setattr(store, "CHRONOVISOR_ROOT", pages_root)
    monkeypatch.setattr(store, "PAGES_DIR", pages_root / "pages")
    monkeypatch.setattr(recall_hints, "QUERY_HINTS_FILE", hints_file)
    calls = 0

    def approve(_proposal, **_kwargs):
        nonlocal calls
        calls += 1
        return {"decision": "approved", "summary": "exact mapping is supported"}

    record = _candidate(
        "query_hint",
        page_id="target-page",
        payload={"query": "exact missing context"},
    )
    deferred = recall_auto_apply.apply_feedback_records(
        [record],
        policy=recall_auto_apply.AutoApplyPolicy(min_count=1),
        log_file=log_file,
        review_dir=review_dir,
        frontier_reviewer=approve,
        budget=CycleBudget(max_frontier_calls=1, max_mutations=0),
    )

    artifact = Path(deferred["actions"][0]["frontier_artifact"])
    assert deferred["actions"][0]["status"] == "budget_deferred"
    assert artifact.exists()
    assert not hints_file.exists()

    def must_not_review_again(*_args, **_kwargs):
        raise AssertionError("durable frontier verdict should be reused")

    applied = recall_auto_apply.apply_feedback_records(
        [record],
        policy=recall_auto_apply.AutoApplyPolicy(min_count=1),
        log_file=log_file,
        review_dir=review_dir,
        frontier_reviewer=must_not_review_again,
        budget=CycleBudget(max_frontier_calls=0, max_mutations=1),
    )

    assert calls == 1
    assert applied["actions"][0]["status"] == "applied"
    assert applied["actions"][0]["frontier_artifact_reused"] is True
    assert hints_file.exists()


def test_saved_approval_is_not_reused_after_authority_changes(
    tmp_path,
    monkeypatch,
) -> None:
    pages_root = tmp_path / "wiki"
    _page(pages_root, "target-page")
    hints_file = tmp_path / "query-hints.json"
    log_file = tmp_path / "auto-apply.jsonl"
    review_dir = tmp_path / "reviews"
    monkeypatch.setattr(store, "CHRONOVISOR_ROOT", pages_root)
    monkeypatch.setattr(store, "PAGES_DIR", pages_root / "pages")
    monkeypatch.setattr(recall_hints, "QUERY_HINTS_FILE", hints_file)
    authority = _production_authority("a")
    monkeypatch.setattr(
        recall_auto_apply,
        "_current_review_authority",
        lambda **_kwargs: (dict(authority), None),
    )
    calls = 0

    def review(_proposal):
        nonlocal calls
        calls += 1
        return _review("approved" if calls == 1 else "rejected", authority)

    record = _candidate(
        "query_hint",
        page_id="target-page",
        payload={"query": "exact missing context"},
    )
    deferred = recall_auto_apply.apply_feedback_records(
        [record],
        policy=recall_auto_apply.AutoApplyPolicy(min_count=1),
        log_file=log_file,
        review_dir=review_dir,
        frontier_reviewer=review,
        budget=CycleBudget(max_frontier_calls=1, max_mutations=0),
    )
    artifact = Path(deferred["actions"][0]["frontier_artifact"])
    assert (
        json.loads(artifact.read_text())["authority"]["router"]["artifact_sha256"]
        == "a" * 64
    )

    authority = _production_authority("b")
    rejected = recall_auto_apply.apply_feedback_records(
        [record],
        policy=recall_auto_apply.AutoApplyPolicy(min_count=1),
        log_file=log_file,
        review_dir=review_dir,
        frontier_reviewer=review,
        budget=CycleBudget(max_frontier_calls=1, max_mutations=1),
    )

    assert calls == 2
    assert rejected["actions"][0]["status"] == "frontier_rejected"
    assert rejected["actions"][0]["frontier_artifact_reused"] is False
    assert (
        json.loads(artifact.read_text())["authority"]["router"]["artifact_sha256"]
        == "b" * 64
    )
    assert not hints_file.exists()


def test_production_verdict_without_router_audit_is_not_persisted(
    tmp_path,
    monkeypatch,
) -> None:
    pages_root = tmp_path / "wiki"
    _page(pages_root, "target-page")
    hints_file = tmp_path / "query-hints.json"
    log_file = tmp_path / "auto-apply.jsonl"
    review_dir = tmp_path / "reviews"
    authority = _production_authority("a")
    monkeypatch.setattr(store, "CHRONOVISOR_ROOT", pages_root)
    monkeypatch.setattr(store, "PAGES_DIR", pages_root / "pages")
    monkeypatch.setattr(recall_hints, "QUERY_HINTS_FILE", hints_file)
    monkeypatch.setattr(
        recall_auto_apply,
        "_current_review_authority",
        lambda **_kwargs: (dict(authority), None),
    )

    result = recall_auto_apply.apply_feedback_records(
        [
            _candidate(
                "query_hint",
                page_id="target-page",
                payload={"query": "exact missing context"},
            )
        ],
        policy=recall_auto_apply.AutoApplyPolicy(min_count=1),
        log_file=log_file,
        review_dir=review_dir,
        frontier_reviewer=lambda _proposal: {
            "decision": "approved",
            "summary": "model-authored approval without trusted audit",
        },
    )

    action = result["actions"][0]
    assert action["status"] == "frontier_retry"
    assert "authority audit is missing" in action["result"]["reason"]
    assert not list(review_dir.glob("*.json"))
    assert not hints_file.exists()


def test_authority_change_after_review_blocks_artifact_persistence(
    tmp_path,
    monkeypatch,
) -> None:
    pages_root = tmp_path / "wiki"
    _page(pages_root, "target-page")
    hints_file = tmp_path / "query-hints.json"
    review_dir = tmp_path / "reviews"
    authority_a = _production_authority("a")
    authorities = iter(
        [
            (authority_a, None),
            (_production_authority("b"), None),
            (_production_authority("b"), None),
        ]
    )
    monkeypatch.setattr(store, "CHRONOVISOR_ROOT", pages_root)
    monkeypatch.setattr(store, "PAGES_DIR", pages_root / "pages")
    monkeypatch.setattr(recall_hints, "QUERY_HINTS_FILE", hints_file)
    monkeypatch.setattr(
        recall_auto_apply,
        "_current_review_authority",
        lambda **_kwargs: next(authorities),
    )

    result = recall_auto_apply.apply_feedback_records(
        [
            _candidate(
                "query_hint",
                page_id="target-page",
                payload={"query": "exact missing context"},
            )
        ],
        policy=recall_auto_apply.AutoApplyPolicy(min_count=1),
        log_file=tmp_path / "auto-apply.jsonl",
        review_dir=review_dir,
        frontier_reviewer=lambda _proposal: _review("approved", authority_a),
    )

    assert result["actions"][0]["status"] == "retry"
    assert result["actions"][0]["authority_transition_blocked"] is True
    assert not list(review_dir.glob("*.json"))
    assert not hints_file.exists()


def test_authority_is_rechecked_immediately_before_mutation(
    tmp_path,
    monkeypatch,
) -> None:
    pages_root = tmp_path / "wiki"
    _page(pages_root, "target-page")
    hints_file = tmp_path / "query-hints.json"
    monkeypatch.setattr(store, "CHRONOVISOR_ROOT", pages_root)
    monkeypatch.setattr(store, "PAGES_DIR", pages_root / "pages")
    monkeypatch.setattr(recall_hints, "QUERY_HINTS_FILE", hints_file)
    authorities = iter(
        [
            (_production_authority("a"), None),
            (_production_authority("a"), None),
            (_production_authority("b"), None),
            (_production_authority("b"), None),
        ]
    )
    monkeypatch.setattr(
        recall_auto_apply,
        "_current_review_authority",
        lambda **_kwargs: next(authorities),
    )

    result = recall_auto_apply.apply_feedback_records(
        [
            _candidate(
                "query_hint",
                page_id="target-page",
                payload={"query": "exact missing context"},
            )
        ],
        policy=recall_auto_apply.AutoApplyPolicy(min_count=1),
        log_file=tmp_path / "auto-apply.jsonl",
        review_dir=tmp_path / "reviews",
        frontier_reviewer=lambda _proposal: _review(
            "approved", _production_authority("a")
        ),
    )

    assert result["actions"][0]["status"] == "retry"
    assert "authority changed" in result["actions"][0]["result"]["reason"]
    assert not hints_file.exists()


def test_authority_epoch_is_held_through_approved_mutation(
    tmp_path,
    monkeypatch,
) -> None:
    pages_root = tmp_path / "wiki"
    _page(pages_root, "target-page")
    hints_file = tmp_path / "query-hints.json"
    monkeypatch.setattr(store, "CHRONOVISOR_ROOT", pages_root)
    monkeypatch.setattr(store, "PAGES_DIR", pages_root / "pages")
    monkeypatch.setattr(recall_hints, "QUERY_HINTS_FILE", hints_file)
    epoch_held = False

    @contextmanager
    def authority_epoch():
        nonlocal epoch_held
        assert not epoch_held
        epoch_held = True
        try:
            yield
        finally:
            epoch_held = False

    real_apply = recall_auto_apply.apply_query_hint

    def apply_while_epoch_held(record, *, dry_run):
        if not dry_run:
            assert epoch_held
        return real_apply(record, dry_run=dry_run)

    monkeypatch.setattr(recall_auto_apply, "decision_authority_lock", authority_epoch)
    monkeypatch.setattr(recall_auto_apply, "apply_query_hint", apply_while_epoch_held)

    result = recall_auto_apply.apply_feedback_records(
        [
            _candidate(
                "query_hint",
                page_id="target-page",
                payload={"query": "exact missing context"},
            )
        ],
        policy=recall_auto_apply.AutoApplyPolicy(min_count=1),
        log_file=tmp_path / "auto-apply.jsonl",
        review_dir=tmp_path / "reviews",
        frontier_reviewer=lambda _proposal: {
            "decision": "approved",
            "summary": "stable authority",
        },
    )

    assert result["actions"][0]["status"] == "applied"
    assert hints_file.exists()


def test_existing_effect_uses_recovery_only_convergence_path(
    tmp_path,
    monkeypatch,
) -> None:
    pages_root = tmp_path / "wiki"
    _page(pages_root, "target-page")
    hints_file = tmp_path / "query-hints.json"
    monkeypatch.setattr(store, "CHRONOVISOR_ROOT", pages_root)
    monkeypatch.setattr(store, "PAGES_DIR", pages_root / "pages")
    monkeypatch.setattr(recall_hints, "QUERY_HINTS_FILE", hints_file)
    recall_hints.add_query_hint(
        page_id="target-page",
        query="exact missing context",
        path=hints_file,
    )

    result = recall_auto_apply.apply_feedback_records(
        [
            _candidate(
                "query_hint",
                page_id="target-page",
                payload={
                    "page_id": "target-page",
                    "query": "exact missing context",
                },
            )
        ],
        policy=recall_auto_apply.AutoApplyPolicy(min_count=1),
        log_file=tmp_path / "auto-apply.jsonl",
    )

    action = result["actions"][0]
    assert action["status"] == "already_applied"
    assert action["recovery_only"] is True
    assert len(action["recovery_proposal_sha256"]) == 64


def test_rejected_transition_is_not_committed_after_authority_race(
    tmp_path,
    monkeypatch,
) -> None:
    pages_root = tmp_path / "wiki"
    _page(pages_root, "target-page")
    authority_a = _production_authority("a")
    authorities = iter(
        [
            (authority_a, None),
            (authority_a, None),
            (_production_authority("b"), None),
        ]
    )
    log_file = tmp_path / "auto-apply.jsonl"
    monkeypatch.setattr(store, "CHRONOVISOR_ROOT", pages_root)
    monkeypatch.setattr(store, "PAGES_DIR", pages_root / "pages")
    monkeypatch.setattr(
        recall_auto_apply,
        "_current_review_authority",
        lambda **_kwargs: next(authorities),
    )

    result = recall_auto_apply.apply_feedback_records(
        [_candidate("query_hint", page_id="target-page")],
        policy=recall_auto_apply.AutoApplyPolicy(min_count=1),
        log_file=log_file,
        review_dir=tmp_path / "reviews",
        frontier_reviewer=lambda _proposal: _review("rejected", authority_a),
    )

    action = result["actions"][0]
    assert action["status"] == "retry"
    assert action["authority_transition_blocked"] is True
    assert not log_file.exists()


def test_review_migration_recovers_effect_when_authority_changes_before_log(
    tmp_path,
    monkeypatch,
) -> None:
    pages_root = tmp_path / "wiki"
    _page(pages_root, "target-page")
    hints_file = tmp_path / "query-hints.json"
    log_file = tmp_path / "auto-apply.jsonl"
    authority_a = _production_authority("a")
    authority_b = _production_authority("b")
    calls = 0

    def authority(**_kwargs):
        nonlocal calls
        calls += 1
        return (authority_a if calls <= 4 else authority_b), None

    monkeypatch.setattr(store, "CHRONOVISOR_ROOT", pages_root)
    monkeypatch.setattr(store, "PAGES_DIR", pages_root / "pages")
    monkeypatch.setattr(recall_hints, "QUERY_HINTS_FILE", hints_file)
    monkeypatch.setattr(recall_auto_apply, "_current_review_authority", authority)
    record = {
        "kind": "missed_candidate",
        "source": "auditor",
        "lane": "review",
        "action_type": "few_shot",
        "normalize_key": "few-shot:authority-race",
        "missing_signal": "specific recall",
        "expected_pages": ["target-page"],
        "ref": "d-authority",
    }

    def reviewer(_proposal):
        return _review("approved", authority_a)

    raced = recall_auto_apply.apply_review_feedback_records(
        [record],
        log_file=log_file,
        review_dir=tmp_path / "reviews",
        frontier_reviewer=reviewer,
    )
    recovered = recall_auto_apply.apply_review_feedback_records(
        [record],
        log_file=log_file,
        review_dir=tmp_path / "reviews",
        frontier_reviewer=reviewer,
    )

    assert raced["actions"][0]["status"] == "retry"
    assert raced["actions"][0]["authority_transition_blocked"] is True
    assert hints_file.exists()
    assert recovered["actions"][0]["status"] == "already_applied"
    assert recovered["actions"][0]["recovery_only"] is True
    assert recall_auto_apply.read_applied_keys(log_file)


def test_page_tag_does_not_overwrite_concurrent_content_correction(
    tmp_path, monkeypatch
) -> None:
    pages_root = tmp_path / "wiki"
    page = _page(
        pages_root,
        "chronovisor-recall-audit-architecture",
        body="The recalled display is a Kuycon G32P.",
    )
    monkeypatch.setattr(store, "CHRONOVISOR_ROOT", pages_root)
    monkeypatch.setattr(store, "PAGES_DIR", pages_root / "pages")
    monkeypatch.setattr(
        page_mutation,
        "CHRONOVISOR_MUTATION_LOCK",
        pages_root / "runtime" / "wiki-mutation.lock",
    )

    correction_locked = threading.Event()
    auto_read_preimage = threading.Event()
    apply_correction = threading.Event()
    correction_errors: list[BaseException] = []
    result: dict[str, object] = {}

    def correction_writer() -> None:
        try:
            with page_mutation.chronovisor_mutation_lock():
                correction_locked.set()
                if not apply_correction.wait(timeout=5):
                    raise TimeoutError("auto-apply did not reach its preimage read")
                corrected = page.read_text(encoding="utf-8").replace(
                    "The recalled display is a Kuycon G32P.",
                    "The recalled displays are two Kuycon P24U units.",
                )
                atomic_write(page, corrected)
        except BaseException as exc:  # pragma: no cover - asserted below
            correction_errors.append(exc)

    original_parse = recall_auto_apply.parse_frontmatter

    def parse_after_read(text: str):
        auto_read_preimage.set()
        return original_parse(text)

    monkeypatch.setattr(recall_auto_apply, "parse_frontmatter", parse_after_read)
    correction_thread = threading.Thread(target=correction_writer, daemon=True)
    correction_thread.start()
    assert correction_locked.wait(timeout=5)

    record = _candidate(
        "page_tag",
        page_id="chronovisor-recall-audit-architecture",
        payload={"tag": "d/theory"},
    )

    def apply_tag() -> None:
        result.update(recall_auto_apply.apply_page_tag(record, dry_run=False))

    auto_thread = threading.Thread(target=apply_tag, daemon=True)
    auto_thread.start()
    assert auto_read_preimage.wait(timeout=5)
    apply_correction.set()
    correction_thread.join(timeout=5)
    auto_thread.join(timeout=5)

    assert not correction_thread.is_alive()
    assert not auto_thread.is_alive()
    assert correction_errors == []
    assert result["status"] == "retry"
    assert result["reason"] == "page changed before page_tag apply"
    final_text = page.read_text(encoding="utf-8")
    final_meta, final_body = parse_frontmatter(final_text)
    assert "two Kuycon P24U units" in final_body
    assert "Kuycon G32P" not in final_body
    assert "d/theory" not in final_meta["tags"]


def test_invalid_page_tag_falls_back_to_query_hint(tmp_path, monkeypatch) -> None:
    pages_root = tmp_path / "wiki"
    _page(pages_root, "chronovisor-recall-audit-architecture")
    hints_file = tmp_path / "query-hints.json"
    monkeypatch.setattr(store, "CHRONOVISOR_ROOT", pages_root)
    monkeypatch.setattr(store, "PAGES_DIR", pages_root / "pages")
    monkeypatch.setattr(recall_hints, "QUERY_HINTS_FILE", hints_file)

    record = _candidate(
        "page_tag",
        page_id="chronovisor-recall-audit-architecture",
        payload={"tag": "Assistant wrote a prose reason instead of a taxonomy tag."},
    )
    result = recall_auto_apply.apply_feedback_records(
        [record],
        policy=recall_auto_apply.AutoApplyPolicy(min_count=1),
        log_file=tmp_path / "auto-apply.jsonl",
    )

    action = result["actions"][0]
    assert action["status"] == "fallback_applied"
    assert action["result"]["fallback_to"] == "query_hint"
    assert hints_file.exists()


def test_page_tag_without_target_is_skipped_not_error(tmp_path, monkeypatch) -> None:
    pages_root = tmp_path / "wiki"
    monkeypatch.setattr(store, "CHRONOVISOR_ROOT", pages_root)
    monkeypatch.setattr(store, "PAGES_DIR", pages_root / "pages")
    monkeypatch.setattr(recall_hints, "QUERY_HINTS_FILE", tmp_path / "query-hints.json")

    record = _candidate(
        "page_tag",
        page_id="",
        payload={"tag": "Assistant wrote a prose reason instead of a taxonomy tag."},
    )
    result = recall_auto_apply.apply_feedback_records(
        [record],
        policy=recall_auto_apply.AutoApplyPolicy(min_count=1),
        log_file=tmp_path / "auto-apply.jsonl",
    )

    action = result["actions"][0]
    assert action["status"] == "skipped"
    assert action["result"]["fallback_to"] == "query_hint"


def test_query_hint_without_target_is_skipped_not_error(tmp_path, monkeypatch) -> None:
    pages_root = tmp_path / "wiki"
    hints_file = tmp_path / "query-hints.json"
    log_file = tmp_path / "auto-apply.jsonl"
    monkeypatch.setattr(store, "CHRONOVISOR_ROOT", pages_root)
    monkeypatch.setattr(store, "PAGES_DIR", pages_root / "pages")
    monkeypatch.setattr(recall_hints, "QUERY_HINTS_FILE", hints_file)

    record = _candidate(
        "query_hint", page_id="", payload={"query": "specific missing context"}
    )
    result = recall_auto_apply.apply_feedback_records(
        [record],
        policy=recall_auto_apply.AutoApplyPolicy(min_count=1),
        log_file=log_file,
    )

    action = result["actions"][0]
    assert action["status"] == "skipped"
    assert action["result"]["reason"] == "query_hint missing page_id"
    assert "auto_apply_self_heal" not in result
    assert not hints_file.exists()
    assert (
        recall_auto_apply.apply_feedback_records(
            [record],
            policy=recall_auto_apply.AutoApplyPolicy(min_count=1),
            log_file=log_file,
        )["actions"]
        == []
    )


def test_alias_auto_apply_uses_existing_alias_store(tmp_path, monkeypatch) -> None:
    from chronovisor.core import alias_store

    pages_root = tmp_path / "wiki"
    _page(pages_root, "canonical-recall-page")
    monkeypatch.setattr(store, "CHRONOVISOR_ROOT", pages_root)
    monkeypatch.setattr(store, "PAGES_DIR", pages_root / "pages")

    record = _candidate(
        "alias",
        page_id="canonical-recall-page",
        payload={"alias": "made-up-recall-page"},
    )
    result = recall_auto_apply.apply_feedback_records(
        [record],
        policy=recall_auto_apply.AutoApplyPolicy(min_count=1),
        log_file=tmp_path / "auto-apply.jsonl",
    )

    assert result["actions"][0]["status"] == "applied"
    assert (
        alias_store.load_aliases()["made-up-recall-page"] == "ai/canonical-recall-page"
    )


def test_invalid_alias_falls_back_to_query_hint(tmp_path, monkeypatch) -> None:
    pages_root = tmp_path / "wiki"
    _page(pages_root, "canonical-recall-page")
    hints_file = tmp_path / "query-hints.json"
    monkeypatch.setattr(store, "CHRONOVISOR_ROOT", pages_root)
    monkeypatch.setattr(store, "PAGES_DIR", pages_root / "pages")
    monkeypatch.setattr(recall_hints, "QUERY_HINTS_FILE", hints_file)

    record = _candidate(
        "alias",
        page_id="canonical-recall-page",
        payload={"alias": "自然言語の説明は alias page_id ではない"},
    )
    result = recall_auto_apply.apply_feedback_records(
        [record],
        policy=recall_auto_apply.AutoApplyPolicy(min_count=1),
        log_file=tmp_path / "auto-apply.jsonl",
    )

    assert result["actions"][0]["status"] == "fallback_applied"
    assert hints_file.exists()


def test_invalid_alias_target_falls_back_to_expected_page_hint(
    tmp_path, monkeypatch
) -> None:
    pages_root = tmp_path / "wiki"
    _page(pages_root, "canonical-recall-page")
    hints_file = tmp_path / "query-hints.json"
    monkeypatch.setattr(store, "CHRONOVISOR_ROOT", pages_root)
    monkeypatch.setattr(store, "PAGES_DIR", pages_root / "pages")
    monkeypatch.setattr(recall_hints, "QUERY_HINTS_FILE", hints_file)

    record = _candidate(
        "alias",
        page_id="canonical-recall-page",
        payload={"alias": "missing-target", "target_page": "does-not-exist"},
    )
    result = recall_auto_apply.apply_feedback_records(
        [record],
        policy=recall_auto_apply.AutoApplyPolicy(min_count=1),
        log_file=tmp_path / "auto-apply.jsonl",
    )

    assert result["actions"][0]["status"] == "fallback_applied"
    assert hints_file.exists()


def test_error_apply_keys_are_retriable(tmp_path) -> None:
    log_file = tmp_path / "auto-apply.jsonl"
    log_file.write_text(
        "\n".join(
            [
                json.dumps({"apply_key": "failed", "status": "error"}),
                json.dumps({"apply_key": "ok", "status": "applied"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    assert recall_auto_apply.read_applied_keys(log_file) == {"ok"}


def test_full_apply_history_does_not_resurrect_old_terminal_keys(tmp_path) -> None:
    log_file = tmp_path / "auto-apply.jsonl"
    rows = [{"apply_key": "old", "status": "applied", "convergence_status": "applied"}]
    rows.extend(
        {
            "apply_key": f"retry-{index}",
            "status": "error",
            "convergence_status": "retry_wait",
            "attempt": 1,
        }
        for index in range(5_100)
    )
    log_file.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    assert "old" in recall_auto_apply.read_applied_keys(log_file)
    assert (
        recall_auto_apply.read_apply_states(log_file)["old"]["convergence_status"]
        == "applied"
    )


def test_pull_log_candidate_is_consumed_by_validated_auto_lane(
    tmp_path, monkeypatch
) -> None:
    record = _candidate("query_hint", page_id="target")
    record["source"] = "pull-log"
    record["session_id"] = "session-1"
    record["pull_event"] = {"session_id": "session-1"}
    monkeypatch.setattr(
        recall_auto_apply,
        "apply_record",
        lambda _record, dry_run=False: {"status": "applied"},
    )

    result = recall_auto_apply.apply_feedback_records(
        [record],
        policy=recall_auto_apply.AutoApplyPolicy(min_count=1),
        log_file=tmp_path / "auto-apply.jsonl",
    )

    assert result["actions"][0]["status"] == "applied"


def test_apply_feedback_budget_defers_without_burning_attempt(
    tmp_path, monkeypatch
) -> None:
    pages_root = tmp_path / "wiki"
    _page(pages_root, "target")
    hints_file = tmp_path / "query-hints.json"
    feedback_file = tmp_path / "feedback.jsonl"
    log_file = tmp_path / "auto-apply.jsonl"
    feedback_file.write_text(
        json.dumps(_candidate("query_hint", page_id="target")) + "\n"
    )
    monkeypatch.setattr(store, "CHRONOVISOR_ROOT", pages_root)
    monkeypatch.setattr(store, "PAGES_DIR", pages_root / "pages")
    monkeypatch.setattr(recall_hints, "QUERY_HINTS_FILE", hints_file)
    monkeypatch.setattr(recall_auto_apply, "AUTO_APPLY_LOG_FILE", log_file)
    deferred = recall_auto_apply.apply_feedback_file(
        feedback_file=feedback_file,
        config_file=tmp_path / "missing.toml",
        budget=CycleBudget(max_mutations=0),
    )

    assert deferred["status"] == "budget_deferred"
    assert deferred["actions"][0]["attempt"] == 0
    assert not hints_file.exists()
    assert not log_file.exists()

    applied = recall_auto_apply.apply_feedback_file(
        feedback_file=feedback_file,
        config_file=tmp_path / "missing.toml",
        budget=CycleBudget(max_mutations=1),
    )
    assert applied["actions"][0]["attempt"] == 1
    assert hints_file.exists()


def test_existing_query_hint_is_terminal_without_incrementing_evidence_count(
    tmp_path, monkeypatch
) -> None:
    pages_root = tmp_path / "wiki"
    _page(pages_root, "target")
    hints_file = tmp_path / "query-hints.json"
    monkeypatch.setattr(store, "CHRONOVISOR_ROOT", pages_root)
    monkeypatch.setattr(store, "PAGES_DIR", pages_root / "pages")
    monkeypatch.setattr(recall_hints, "QUERY_HINTS_FILE", hints_file)
    recall_hints.add_query_hint(page_id="target", query="exact query", path=hints_file)
    record = _candidate(
        "query_hint",
        page_id="target",
        payload={"page_id": "target", "query": "exact query"},
    )

    result = recall_auto_apply.apply_feedback_records(
        [record],
        policy=recall_auto_apply.AutoApplyPolicy(min_count=1),
        log_file=tmp_path / "auto-apply.jsonl",
    )

    assert result["actions"][0]["status"] == "already_applied"
    assert recall_hints.load_query_hints(hints_file)[0]["count"] == 1


def test_skipped_auto_apply_is_bounded_not_permanently_consumed(
    tmp_path, monkeypatch
) -> None:
    log_file = tmp_path / "auto-apply.jsonl"
    record = _candidate(
        "query_hint", page_id="missing", payload={"query": "q", "page_id": "missing"}
    )
    monkeypatch.setattr(
        recall_auto_apply,
        "apply_record",
        lambda _record, dry_run=False: {"status": "skipped"},
    )
    policy = recall_auto_apply.AutoApplyPolicy(min_count=1)

    first = recall_auto_apply.apply_feedback_records(
        [record],
        policy=policy,
        log_file=log_file,
        max_attempts=2,
        backoff_base_seconds=0,
    )
    second = recall_auto_apply.apply_feedback_records(
        [record],
        policy=policy,
        log_file=log_file,
        max_attempts=2,
        backoff_base_seconds=0,
    )

    assert first["actions"][0]["convergence_status"] == "retry_wait"
    assert second["actions"][0]["convergence_status"] == "quarantined"
    assert recall_auto_apply.read_applied_keys(log_file) == set()


def test_quarantined_auto_apply_resumes_with_fresh_attempt_budget(
    tmp_path,
    monkeypatch,
) -> None:
    log_file = tmp_path / "auto-apply.jsonl"
    record = _candidate(
        "query_hint",
        page_id="missing",
        payload={"query": "q", "page_id": "missing"},
    )
    policy = recall_auto_apply.AutoApplyPolicy(min_count=1)
    started = datetime(2026, 7, 11, 0, 0, 0)
    monkeypatch.setattr(
        recall_auto_apply,
        "apply_record",
        lambda _record, dry_run=False: {"status": "skipped"},
    )

    quarantined = recall_auto_apply.apply_feedback_records(
        [record],
        policy=policy,
        log_file=log_file,
        max_attempts=1,
        quarantine_cooldown_seconds=6 * 60 * 60,
        now=started,
    )
    too_early = recall_auto_apply.apply_feedback_records(
        [record],
        policy=policy,
        log_file=log_file,
        max_attempts=1,
        quarantine_cooldown_seconds=6 * 60 * 60,
        now=started + timedelta(hours=6) - timedelta(seconds=1),
    )
    monkeypatch.setattr(
        recall_auto_apply,
        "apply_record",
        lambda _record, dry_run=False: {"status": "applied"},
    )
    resumed = recall_auto_apply.apply_feedback_records(
        [record],
        policy=policy,
        log_file=log_file,
        max_attempts=1,
        quarantine_cooldown_seconds=6 * 60 * 60,
        now=started + timedelta(hours=6, seconds=1),
    )

    assert quarantined["actions"][0]["convergence_status"] == "quarantined"
    assert too_early["actions"] == []
    assert resumed["actions"][0]["convergence_status"] == "applied"
    assert resumed["actions"][0]["attempt"] == 1
    assert resumed["actions"][0]["resumed_from_quarantine"] is True
    assert resumed["actions"][0]["quarantine_resume_count"] == 1


def test_human_required_auto_apply_state_never_auto_resumes(tmp_path) -> None:
    log_file = tmp_path / "auto-apply.jsonl"
    record = _candidate("query_hint", page_id="target", payload={"query": "q"})
    key = recall_auto_apply.apply_key_for(record)
    recall_auto_apply.record_apply_log(
        {
            "ts": "2026-07-01T00:00:00",
            "apply_key": key,
            "convergence_status": "human_required",
            "attempt": 3,
        },
        log_file,
    )

    result = recall_auto_apply.apply_feedback_records(
        [record],
        policy=recall_auto_apply.AutoApplyPolicy(min_count=1),
        log_file=log_file,
        now=datetime(2026, 7, 11, 0, 0, 0),
    )

    assert result["actions"] == []


def test_threshold_review_action_routes_to_recall_lab(tmp_path) -> None:
    log_file = tmp_path / "auto-apply.jsonl"
    record = {
        "kind": "missed_candidate",
        "source": "auditor",
        "lane": "review",
        "action_type": "threshold",
        "normalize_key": "threshold:systemic",
        "missing_signal": "systemic false positives",
        "ref": "d1",
    }

    result = recall_auto_apply.apply_review_feedback_records(
        [record], log_file=log_file
    )

    assert result["actions"][0]["status"] == "routed_to_recall_lab"
    assert result["actions"][0]["convergence_status"] == "applied"


def test_quarantined_review_action_resumes_after_cooldown(
    tmp_path, monkeypatch
) -> None:
    log_file = tmp_path / "auto-apply.jsonl"
    record = {
        "kind": "missed_candidate",
        "source": "auditor",
        "lane": "review",
        "action_type": "few_shot",
        "normalize_key": "few-shot:retry",
        "missing_signal": "specific recall",
        "expected_pages": ["target"],
        "ref": "d2",
    }
    started = datetime(2026, 7, 11, 0, 0, 0)
    monkeypatch.setattr(
        recall_auto_apply,
        "apply_query_hint",
        lambda _record, dry_run=False: {"status": "skipped"},
    )
    first = recall_auto_apply.apply_review_feedback_records(
        [record],
        log_file=log_file,
        max_attempts=1,
        now=started,
    )
    monkeypatch.setattr(
        recall_auto_apply,
        "apply_query_hint",
        lambda _record, dry_run=False: {"status": "applied"},
    )
    resumed = recall_auto_apply.apply_review_feedback_records(
        [record],
        log_file=log_file,
        max_attempts=1,
        now=started + timedelta(hours=6, seconds=1),
    )

    assert first["actions"][0]["convergence_status"] == "quarantined"
    assert resumed["actions"][0]["convergence_status"] == "applied"
    assert resumed["actions"][0]["attempt"] == 1
    assert resumed["actions"][0]["resumed_from_quarantine"] is True


def test_query_hint_accepts_system_pages(tmp_path, monkeypatch) -> None:
    pages_root = tmp_path / "wiki"
    system_dir = pages_root / "system"
    system_dir.mkdir(parents=True)
    (system_dir / "lessons-learned.md").write_text(
        "---\ntitle: Lessons Learned\n---\n\nbody\n",
        encoding="utf-8",
    )
    hints_file = tmp_path / "query-hints.json"
    monkeypatch.setattr(store, "CHRONOVISOR_ROOT", pages_root)
    monkeypatch.setattr(store, "PAGES_DIR", pages_root / "pages")
    monkeypatch.setattr(store, "SYSTEM_DIR", system_dir)
    monkeypatch.setattr(recall_hints, "QUERY_HINTS_FILE", hints_file)

    hint = recall_hints.add_query_hint(
        page_id="lessons-learned", query="反省ルール", path=hints_file
    )

    assert hint["page_id"] == "lessons-learned"
    assert hints_file.exists()


def test_auditor_recording_invokes_auto_apply(tmp_path, monkeypatch, capsys) -> None:
    from chronovisor.recall import recall_auditor, recall_runtime

    pages_root = tmp_path / "wiki"
    _page(pages_root, "claude-code-recall-hook-implementation")
    feedback_file = tmp_path / "feedback.jsonl"
    log_file = tmp_path / "recall-log.jsonl"
    hints_file = tmp_path / "query-hints.json"
    prompt = "昨日の recall hook の続き"
    decision_id = "decision-1"
    log_file.write_text(
        json.dumps(
            {
                "decision_id": decision_id,
                "host": "codex",
                "session_id": "s1",
                "prompt_hash": recall_runtime.stable_prompt_hash(prompt),
                "prompt_preview": prompt,
                "decision": "none",
                "confidence": 0.2,
                "queries": [],
                "pages": [],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(store, "CHRONOVISOR_ROOT", pages_root)
    monkeypatch.setattr(store, "PAGES_DIR", pages_root / "pages")
    monkeypatch.setattr(recall_runtime, "RECALL_FEEDBACK_FILE", feedback_file)
    monkeypatch.setattr(recall_runtime, "RECALL_LOG_FILE", log_file)
    monkeypatch.setattr(recall_auditor, "RECALL_LOG_FILE", log_file)
    monkeypatch.setattr(recall_hints, "QUERY_HINTS_FILE", hints_file)
    monkeypatch.setattr(
        recall_auto_apply, "AUTO_APPLY_LOG_FILE", tmp_path / "auto-apply.jsonl"
    )
    monkeypatch.setattr(
        recall_auditor, "collect_top_pages", lambda _prompt, _policy: ([], "bm25")
    )
    auditor_json = json.dumps(
        {
            "missed": True,
            "confidence": 0.9,
            "reason_code": "gate_missed",
            "auditor_reason": "missed context",
            "expected_pages": ["claude-code-recall-hook-implementation"],
            "missing_signal": "recall hook",
            "action_type": "query_hint",
            "action_payload": {"query": prompt},
        },
        ensure_ascii=False,
    )

    assert (
        recall_auditor.main(
            [
                "--host",
                "codex",
                "--session-id",
                "s1",
                "--prompt",
                prompt,
                "--assistant-response",
                "続きです。",
                "--decision-id",
                decision_id,
                "--state-file",
                str(tmp_path / "state.json"),
                "--auditor-json",
                auditor_json,
            ]
        )
        == 0
    )
    output = json.loads(capsys.readouterr().out)

    assert output["status"] == "recorded"
    assert output["auto_apply"]["actions"][0]["status"] == "applied"
    assert hints_file.exists()
