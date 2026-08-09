"""Tests for lint engine — focus on plan-4 tag rules.

Existing lint behavior (broken links, orphans, stale, duplicates) is
already exercised by integration paths in test_ingest.py. This file
adds direct coverage for the tag taxonomy rules introduced in plan-4.
"""

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from pathlib import Path

import pytest

from chronovisor.decision.decision_router import canonical_agreement_signature
from chronovisor.decision.decision_schema_manifest import (
    production_decision_schemas,
    schema_sha256,
)


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


@pytest.fixture()
def isolated_wiki(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Throw-away wiki tree wired through every module that holds path
    constants (mirror of test_ingest.py's fixture). Kept local here so
    test_lint can run independently of test_ingest's fixture."""
    chronovisor_root = tmp_path / "wiki"
    pages = chronovisor_root / "pages"
    raw = chronovisor_root / "raw"
    system = chronovisor_root / "system"
    index_dir = chronovisor_root / ".index"
    for d in (pages, raw, system, index_dir):
        d.mkdir(parents=True, exist_ok=True)

    from chronovisor.core import store
    from chronovisor.ingest import ingest, lint
    from chronovisor.ingest import tag_lifecycle as tags_mod
    from chronovisor.search import index_store

    monkeypatch.setattr(store, "CHRONOVISOR_ROOT", chronovisor_root)
    monkeypatch.setattr(store, "PAGES_DIR", pages)
    monkeypatch.setattr(store, "RAW_DIR", raw)
    monkeypatch.setattr(store, "SYSTEM_DIR", system)
    monkeypatch.setattr(store, "INDEX_FILE", chronovisor_root / "index.md")
    monkeypatch.setattr(store, "LOG_FILE", chronovisor_root / "log.md")
    monkeypatch.setattr(ingest, "PAGES_DIR", pages)
    monkeypatch.setattr(ingest, "INDEX_FILE", chronovisor_root / "index.md")
    monkeypatch.setattr(ingest, "LOG_FILE", chronovisor_root / "log.md")
    monkeypatch.setattr(index_store, "CHRONOVISOR_ROOT", chronovisor_root)
    monkeypatch.setattr(index_store, "PAGES_DIR", pages)
    monkeypatch.setattr(index_store, "SYSTEM_DIR", system)
    monkeypatch.setattr(index_store, "INDEX_DIR", index_dir)
    monkeypatch.setattr(index_store, "PAGES_INDEX_FILE", index_dir / "pages.json")
    monkeypatch.setattr(
        index_store, "BACKLINKS_INDEX_FILE", index_dir / "backlinks.json"
    )
    monkeypatch.setattr(index_store, "_store", None)
    monkeypatch.setattr(lint, "SYSTEM_DIR", system)
    monkeypatch.setattr(tags_mod, "SYSTEM_DIR", system)
    # Reset the lint check cache between tests; without this a
    # per-corpus-version cache hit could replay a previous test's issues.
    import chronovisor.ingest.lint as lint_mod

    monkeypatch.setattr(lint_mod, "_CHECK_CACHE_VERSION", None)
    monkeypatch.setattr(lint_mod, "_CHECK_CACHE_RESULT", None)
    return chronovisor_root


def _seed(chronovisor_root: Path, rel: str, body: str) -> Path:
    path = chronovisor_root / "pages" / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    return path


def _by_type(issues: list[dict], type_: str, page_id: str | None = None) -> list[dict]:
    return [
        i
        for i in issues
        if i["type"] == type_ and (page_id is None or i["page"] == page_id)
    ]


def _frontier_decision(decision: str, summary: str = "reviewed exact proposal") -> dict:
    return {
        "decision": decision,
        "summary": summary,
        "tests_run": ["checked exact page hashes and diff"],
        "commit": None,
        "committed": False,
        "pushed": False,
        "risk": None,
        "notes": None,
    }


def _semantic_authority(lane: str, artifact_digest: str) -> dict:
    return {
        "source": "adopted_local_consensus",
        "authority_version": 1,
        "lane": lane,
        "lane_contract_sha256": "1" * 64,
        "lane_contract_manifest_sha256": "2" * 64,
        "lane_contract_case_manifest_sha256": "3" * 64,
        "policy": {
            "kind": "consensus",
            "schema_name": "lint_safe_semantic_mutation",
            "mode": "enabled",
            "error": None,
        },
        "router": {
            "source": "adopted_artifact",
            "artifact_sha256": artifact_digest,
            "error": None,
            "models": ["primary", "challenger", "tie"],
        },
    }


def _authority_bound_decision(decision: str, authority: dict) -> dict:
    value = _frontier_decision(decision)
    value["decision_policy"] = {
        "lane": authority["lane"],
        **authority["policy"],
        "router_policy": authority["router"],
    }
    schema = production_decision_schemas()[authority["policy"]["schema_name"]]
    signature = canonical_agreement_signature(value, schema=schema)
    agreement = hashlib.sha256(signature.encode("utf-8")).hexdigest()
    models = authority["router"]["models"]
    value["local_consensus"] = {
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
    return value


def _authority_bound_semantic_no_quorum(authority: dict) -> dict:
    reason = "local_models_did_not_reach_two_vote_quorum"
    schema = production_decision_schemas()[authority["policy"]["schema_name"]]

    def vote(role: str, model: str, digit: str) -> dict:
        return {
            "role": role,
            "model": model,
            "requested_num_ctx": 32768,
            "valid": True,
            "signature_sha256": digit * 64,
            "invalid_reason": None,
            "runtime_observation": {
                "status": "observed",
                "model_size_bytes": 1024,
                "num_ctx": 32768,
            },
            "session": {
                "ok": True,
                "model": model,
                "failure_class": None,
                "first_pass_valid": True,
                "repair_turns": 0,
                "attempts": [
                    {
                        "index": 1,
                        "valid": True,
                        "output_sha256": digit * 64,
                        "output_chars": 16,
                        "normalized": False,
                        "error_fingerprint": None,
                        "issues": [],
                    }
                ],
            },
        }

    value = _frontier_decision("needs_retry", reason)
    value.update(
        {
            "reviewer": "local_consensus",
            "human_required": False,
            "frontier_failure": {
                "failure_class": "local_semantic_no_quorum",
                "rescue_status": "local_quarantined",
                "summary": reason,
                "human_required": False,
                "notify_user": False,
            },
            "decision_policy": {
                "lane": authority["lane"],
                **authority["policy"],
                "expected_schema_sha256": schema_sha256(schema),
                "actual_schema_sha256": schema_sha256(schema),
                "router_policy": authority["router"],
            },
            "local_consensus": {
                "status": "quarantined",
                "ok": False,
                "quorum_safety_policy_version": 1,
                "agreement_sha256": None,
                "failure_class": "local_consensus_failed",
                "quarantine_reason": reason,
                "num_ctx": 32768,
                "residency": {},
                "votes": [
                    vote("primary", authority["router"]["models"][0], "a"),
                    vote("challenger", authority["router"]["models"][1], "b"),
                    vote("tie_break", authority["router"]["models"][2], "c"),
                ],
            },
        }
    )
    return value


@pytest.mark.parametrize(
    "lane",
    [
        "entity_backfill",
        "lint_safe_semantic_mutation",
        "metadata_backfill",
        "page_normalize",
    ],
)
def test_shared_semantic_review_preserves_local_consensus_artifact(
    lane: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chronovisor.ingest import lint as lint_mod

    authority = _semantic_authority(lane, "a" * 64)
    monkeypatch.setattr(
        lint_mod,
        "current_semantic_authority",
        lambda _lane, *, injected_reviewer=False: (authority, None),
    )
    proposal = lint_mod.build_semantic_mutation_proposal(
        page_id="p",
        operation="shared_lane_test",
        expected_text="before",
        updated_text="after",
        details={"lane": lane},
    )
    artifact_dir = tmp_path / lane

    reviewed = lint_mod.review_semantic_mutation(
        proposal,
        expected_text="before",
        updated_text="after",
        reviewer=lambda *_args: _authority_bound_decision("approved", authority),
        artifact_dir=artifact_dir,
        decision_lane=lane,
    )
    reused = lint_mod.review_semantic_mutation(
        proposal,
        expected_text="before",
        updated_text="after",
        reviewer=lambda *_args: (_ for _ in ()).throw(
            AssertionError("authority-bound verdict must be reusable")
        ),
        artifact_dir=artifact_dir,
        decision_lane=lane,
    )

    assert reviewed["local_consensus"]["status"] == "agreed"
    assert reused["local_consensus"] == reviewed["local_consensus"]
    assert reused["reused"] is True
    artifact = next((artifact_dir / "frontier-verdicts").glob("*.json"))
    stored = json.loads(artifact.read_text(encoding="utf-8"))
    assert stored["verdict"]["local_consensus"] == reviewed["local_consensus"]


@pytest.mark.parametrize(
    "lane",
    [
        "entity_backfill",
        "lint_safe_semantic_mutation",
        "metadata_backfill",
        "page_normalize",
    ],
)
def test_shared_semantic_no_quorum_is_exact_epoch_hold_with_aba_reuse(
    lane: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chronovisor.ingest import lint as lint_mod

    authority_a = _semantic_authority(lane, "a" * 64)
    authority_b = _semantic_authority(lane, "b" * 64)
    active = [authority_a]
    monkeypatch.setattr(
        lint_mod,
        "current_semantic_authority",
        lambda _lane, *, injected_reviewer=False: (active[0], None),
    )
    proposal = lint_mod.build_semantic_mutation_proposal(
        page_id="p",
        operation="shared_lane_test",
        expected_text="before",
        updated_text="after",
        details={"lane": lane},
    )
    calls = 0

    def split(_prompt, _schema):
        nonlocal calls
        calls += 1
        return _authority_bound_semantic_no_quorum(active[0])

    def run() -> dict:
        return lint_mod.review_semantic_mutation(
            proposal,
            expected_text="before",
            updated_text="after",
            reviewer=split,
            artifact_dir=tmp_path / lane,
            decision_lane=lane,
        )

    first = run()
    same_epoch = run()
    active[0] = authority_b
    changed_authority = run()
    active[0] = authority_a
    rolled_back = run()

    assert calls == 2
    assert first["semantic_hold"]["authority"] == authority_a
    assert same_epoch["reused"] is True
    assert changed_authority["semantic_hold"]["authority"] == authority_b
    assert rolled_back["reused"] is True
    assert rolled_back["semantic_hold"] == first["semantic_hold"]
    assert len(list((tmp_path / lane / "semantic-holds").glob("*.json"))) == 2


def test_shared_semantic_no_quorum_hold_changes_with_exact_proposal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chronovisor.ingest import lint as lint_mod

    lane = "lint_safe_semantic_mutation"
    authority = _semantic_authority(lane, "a" * 64)
    monkeypatch.setattr(
        lint_mod,
        "current_semantic_authority",
        lambda _lane, *, injected_reviewer=False: (authority, None),
    )
    calls = 0

    def split(_prompt, _schema):
        nonlocal calls
        calls += 1
        return _authority_bound_semantic_no_quorum(authority)

    for updated in ("after-one", "after-two"):
        proposal = lint_mod.build_semantic_mutation_proposal(
            page_id="p",
            operation="shared_lane_test",
            expected_text="before",
            updated_text=updated,
            details={},
        )
        lint_mod.review_semantic_mutation(
            proposal,
            expected_text="before",
            updated_text=updated,
            reviewer=split,
            artifact_dir=tmp_path,
            decision_lane=lane,
        )
    assert calls == 2


def test_shared_operational_review_failure_is_not_a_semantic_hold(
    tmp_path: Path,
) -> None:
    from chronovisor.ingest import lint as lint_mod

    proposal = lint_mod.build_semantic_mutation_proposal(
        page_id="p",
        operation="shared_lane_test",
        expected_text="before",
        updated_text="after",
        details={},
    )
    calls = 0

    def unavailable(_prompt, _schema):
        nonlocal calls
        calls += 1
        return {
            **_frontier_decision("needs_retry"),
            "frontier_failure": {
                "failure_class": "local_resource_quarantined",
            },
        }

    for _ in range(2):
        lint_mod.review_semantic_mutation(
            proposal,
            expected_text="before",
            updated_text="after",
            reviewer=unavailable,
            artifact_dir=tmp_path,
            decision_lane="lint_safe_semantic_mutation",
            injected_reviewer=True,
        )
    assert calls == 2
    assert not (tmp_path / "semantic-holds").exists()


# ---------------------------------------------------------------------------
# tag_missing — high severity
# ---------------------------------------------------------------------------


class TestTagMissing:
    def test_no_tags_field_flagged(self, isolated_wiki: Path) -> None:
        from chronovisor.ingest.lint import check

        _seed(
            isolated_wiki,
            "p.md",
            "---\ntitle: P\nupdated: 2026-05-08\n---\nbody\n",
        )
        issues = check()
        flagged = _by_type(issues, "tag_missing", "p")
        assert len(flagged) == 1
        assert flagged[0]["severity"] == "high"
        assert flagged[0]["auto_fixable"] is False

    def test_empty_tags_list_also_flagged(self, isolated_wiki: Path) -> None:
        from chronovisor.ingest.lint import check

        _seed(
            isolated_wiki,
            "q.md",
            "---\ntitle: Q\nupdated: 2026-05-08\ntags: []\n---\nbody\n",
        )
        issues = check()
        assert _by_type(issues, "tag_missing", "q")

    def test_reference_pages_are_not_linted(self, isolated_wiki: Path) -> None:
        from chronovisor.ingest.lint import check

        _seed(
            isolated_wiki,
            "car-spec/123.md",
            "---\ntitle: 123\nupdated: 2020-01-01\ntype: reference\n---\n[[missing]]\n",
        )
        issues = check()
        assert [issue for issue in issues if issue["page"] == "123"] == []

    def test_complete_tag_set_not_flagged(self, isolated_wiki: Path) -> None:
        from chronovisor.ingest.lint import check

        _seed(
            isolated_wiki,
            "r.md",
            "---\ntitle: R\nupdated: 2026-05-08\n"
            "tags: [d/ai-industry, t/analysis, s/2026]\n"
            "---\nbody\n",
        )
        issues = check()
        assert _by_type(issues, "tag_missing", "r") == []
        assert _by_type(issues, "tag_invalid", "r") == []
        assert _by_type(issues, "tag_count_violation", "r") == []


# ---------------------------------------------------------------------------
# tag_invalid — medium severity, auto-fixable
# ---------------------------------------------------------------------------


class TestTagInvalid:
    def test_invalid_tag_flagged(self, isolated_wiki: Path) -> None:
        from chronovisor.ingest.lint import check

        _seed(
            isolated_wiki,
            "p.md",
            "---\ntitle: P\nupdated: 2026-05-08\n"
            "tags: [d/ai-industry, no-prefix, t/analysis, s/2026]\n"
            "---\nbody\n",
        )
        issues = check()
        flagged = _by_type(issues, "tag_invalid", "p")
        assert len(flagged) == 1
        assert flagged[0]["auto_fixable"] is True
        assert "no-prefix" in flagged[0]["detail"]

    def test_apply_safe_fixes_drops_invalid(self, isolated_wiki: Path) -> None:
        from chronovisor.ingest.lint import apply_safe_fixes, check

        path = _seed(
            isolated_wiki,
            "p.md",
            "---\ntitle: P\nupdated: 2026-05-08\n"
            "tags: [d/ai-industry, no-prefix, t/analysis, s/2026]\n"
            "---\nbody\n",
        )
        issues = check()
        actions = apply_safe_fixes(
            issues,
            reviewer=lambda _prompt, _schema: _frontier_decision("approved"),
        )
        assert any("no-prefix" in a for a in actions)
        text = path.read_text()
        assert "no-prefix" not in text
        # Valid ones survived.
        assert "d/ai-industry" in text
        assert "t/analysis" in text
        assert "s/2026" in text

    def test_apply_safe_fixes_preserves_correction_that_lands_before_cas(
        self,
        isolated_wiki: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from chronovisor.ingest import lint as lint_mod

        path = _seed(
            isolated_wiki,
            "p.md",
            "---\ntitle: P\ntags: [d/ai-industry, invalid, t/analysis, s/2026]\n---\nold fact\n",
        )
        corrected = (
            "---\ntitle: P\ntags: [d/ai-industry, invalid, t/analysis, s/2026]\n"
            "---\nuser-corrected fact\n"
        )

        @contextmanager
        def correction_wins():
            path.write_text(corrected, encoding="utf-8")
            yield

        monkeypatch.setattr(lint_mod, "chronovisor_mutation_lock", correction_wins)
        actions = lint_mod.apply_safe_fixes(
            [{"type": "tag_invalid", "page": "p", "auto_fixable": True}],
            reviewer=lambda _prompt, _schema: _frontier_decision("approved"),
        )

        assert actions == []
        assert path.read_text(encoding="utf-8") == corrected

    def test_local_proposal_cannot_mutate_without_frontier_approval(
        self,
        isolated_wiki: Path,
    ) -> None:
        from chronovisor.ingest.lint import apply_safe_fixes, check

        path = _seed(
            isolated_wiki,
            "p.md",
            "---\ntitle: P\ntags: [d/ai-industry, invalid, t/analysis, s/2026]\n---\nbody\n",
        )
        original = path.read_text(encoding="utf-8")
        actions = apply_safe_fixes(
            check(),
            reviewer=lambda _prompt, _schema: _frontier_decision("needs_retry"),
        )

        assert actions and actions[0].startswith("[frontier-retry]")
        assert path.read_text(encoding="utf-8") == original

    def test_frontier_rejection_is_durable_and_does_not_mutate(
        self,
        isolated_wiki: Path,
    ) -> None:
        from chronovisor.ingest.lint import apply_safe_fixes, check

        path = _seed(
            isolated_wiki,
            "p.md",
            "---\ntitle: P\ntags: [d/ai-industry, invalid, t/analysis, s/2026]\n---\nbody\n",
        )
        original = path.read_text(encoding="utf-8")
        calls = 0

        def reject(_prompt, _schema):
            nonlocal calls
            calls += 1
            return _frontier_decision("rejected")

        issues = check()
        first = apply_safe_fixes(issues, reviewer=reject)
        second = apply_safe_fixes(
            issues,
            reviewer=lambda _prompt, _schema: (_ for _ in ()).throw(
                AssertionError("durable rejection must be reused")
            ),
        )

        assert calls == 1
        assert first[0].startswith("[frontier-rejected]")
        assert second[0].startswith("[frontier-rejected]")
        assert path.read_text(encoding="utf-8") == original
        artifact_root = isolated_wiki / "runtime" / "lint-safe-fixes"
        assert len(list((artifact_root / "proposals").glob("*.json"))) == 1
        artifacts = list((artifact_root / "frontier-verdicts").glob("*.json"))
        assert len(artifacts) == 1
        envelope = json.loads(artifacts[0].read_text(encoding="utf-8"))
        assert envelope["schema_version"] == 3
        assert envelope["authority"] == {
            "source": "injected_reviewer_boundary",
            "authority_version": 1,
            "lane": "lint_safe_semantic_mutation",
        }

    def test_authority_change_invalidates_durable_verdict_and_rereviews(
        self,
        isolated_wiki: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from chronovisor.ingest import lint as lint_mod

        lane = "lint_safe_semantic_mutation"
        first_authority = _semantic_authority(lane, "a" * 64)
        second_authority = _semantic_authority(lane, "b" * 64)
        active = [first_authority]
        monkeypatch.setattr(
            lint_mod,
            "current_semantic_authority",
            lambda _lane, *, injected_reviewer=False: (active[0], None),
        )
        proposal = lint_mod.build_semantic_mutation_proposal(
            page_id="p",
            operation="drop_invalid_tags",
            expected_text="before",
            updated_text="after",
            details={"dropped_tags": ["invalid"]},
        )
        artifact_dir = isolated_wiki / "runtime" / "authority-test"
        calls: list[str] = []

        def review(_prompt, _schema):
            calls.append(active[0]["router"]["artifact_sha256"])
            return _authority_bound_decision("rejected", active[0])

        first = lint_mod.review_semantic_mutation(
            proposal,
            expected_text="before",
            updated_text="after",
            reviewer=review,
            artifact_dir=artifact_dir,
            decision_lane=lane,
        )
        active[0] = second_authority
        second = lint_mod.review_semantic_mutation(
            proposal,
            expected_text="before",
            updated_text="after",
            reviewer=review,
            artifact_dir=artifact_dir,
            decision_lane=lane,
        )
        third = lint_mod.review_semantic_mutation(
            proposal,
            expected_text="before",
            updated_text="after",
            reviewer=lambda *_args: (_ for _ in ()).throw(
                AssertionError("current authority verdict must be reused")
            ),
            artifact_dir=artifact_dir,
            decision_lane=lane,
        )

        assert first["authority"] == first_authority
        assert second["authority"] == second_authority
        assert third["authority"] == second_authority
        assert third["reused"] is True
        assert calls == ["a" * 64, "b" * 64]
        artifact = next((artifact_dir / "frontier-verdicts").glob("*.json"))
        assert (
            json.loads(artifact.read_text(encoding="utf-8"))["authority"]
            == second_authority
        )

    def test_authority_change_during_review_prevents_artifact_and_effect(
        self,
        isolated_wiki: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from chronovisor.ingest import lint as lint_mod

        lane = "lint_safe_semantic_mutation"
        first_authority = _semantic_authority(lane, "a" * 64)
        second_authority = _semantic_authority(lane, "b" * 64)
        authorities = iter([first_authority, second_authority])
        monkeypatch.setattr(
            lint_mod,
            "current_semantic_authority",
            lambda _lane, *, injected_reviewer=False: (next(authorities), None),
        )
        proposal = lint_mod.build_semantic_mutation_proposal(
            page_id="p",
            operation="drop_invalid_tags",
            expected_text="before",
            updated_text="after",
            details={"dropped_tags": ["invalid"]},
        )
        artifact_dir = isolated_wiki / "runtime" / "authority-race"

        review = lint_mod.review_semantic_mutation(
            proposal,
            expected_text="before",
            updated_text="after",
            reviewer=lambda *_args: _authority_bound_decision(
                "approved", first_authority
            ),
            artifact_dir=artifact_dir,
            decision_lane=lane,
        )

        assert review["decision"] == "needs_retry"
        assert review["valid"] is False
        assert not list((artifact_dir / "frontier-verdicts").glob("*.json"))

    def test_authority_change_before_page_effect_fails_closed(
        self,
        isolated_wiki: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from chronovisor.ingest import lint as lint_mod

        path = _seed(
            isolated_wiki,
            "p.md",
            "---\ntitle: P\ntags: [d/ai, invalid, t/analysis, s/2026]\n---\nbody\n",
        )
        original = path.read_text(encoding="utf-8")
        lane = "lint_safe_semantic_mutation"
        first_authority = _semantic_authority(lane, "a" * 64)
        second_authority = _semantic_authority(lane, "b" * 64)
        authorities = iter([first_authority, first_authority, second_authority])
        monkeypatch.setattr(
            lint_mod,
            "current_semantic_authority",
            lambda _lane, *, injected_reviewer=False: (next(authorities), None),
        )

        actions = lint_mod.apply_safe_fixes(
            lint_mod.check(),
            reviewer=lambda *_args: _authority_bound_decision(
                "approved", first_authority
            ),
        )

        assert actions and actions[0].startswith("[frontier-retry]")
        assert path.read_text(encoding="utf-8") == original

    def test_disabled_authority_never_calls_reviewer_or_mutates(
        self,
        isolated_wiki: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from chronovisor.ingest import lint as lint_mod

        path = _seed(
            isolated_wiki,
            "p.md",
            "---\ntitle: P\ntags: [d/ai, invalid, t/analysis, s/2026]\n---\nbody\n",
        )
        original = path.read_text(encoding="utf-8")
        monkeypatch.setattr(
            lint_mod,
            "current_semantic_authority",
            lambda *_args, **_kwargs: (None, "decision_lane_not_enabled"),
        )

        actions = lint_mod.apply_safe_fixes(
            lint_mod.check(),
            reviewer=lambda *_args: (_ for _ in ()).throw(
                AssertionError("disabled authority must not reach reviewer")
            ),
        )

        assert actions and actions[0].startswith("[frontier-retry]")
        assert path.read_text(encoding="utf-8") == original

    def test_durable_frontier_approval_is_reused_after_pre_apply_crash(
        self,
        isolated_wiki: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from chronovisor.ingest import lint as lint_mod

        path = _seed(
            isolated_wiki,
            "p.md",
            "---\ntitle: P\ntags: [d/ai-industry, invalid, t/analysis, s/2026]\n---\nbody\n",
        )
        issues = lint_mod.check()
        real_apply = lint_mod._atomic_write_if_unchanged
        monkeypatch.setattr(
            lint_mod, "_atomic_write_if_unchanged", lambda *_args: False
        )
        first = lint_mod.apply_safe_fixes(
            issues,
            reviewer=lambda _prompt, _schema: _frontier_decision("approved"),
        )
        assert first == []
        assert "invalid" in path.read_text(encoding="utf-8")

        monkeypatch.setattr(lint_mod, "_atomic_write_if_unchanged", real_apply)
        second = lint_mod.apply_safe_fixes(
            issues,
            reviewer=lambda _prompt, _schema: (_ for _ in ()).throw(
                AssertionError("durable approval must be reused")
            ),
        )

        assert second and "dropped 1 invalid tag" in second[0]
        assert "invalid" not in path.read_text(encoding="utf-8")

    def test_dry_run_is_read_only_and_does_not_call_frontier(
        self,
        isolated_wiki: Path,
    ) -> None:
        from chronovisor.ingest.lint import apply_safe_fixes, check

        path = _seed(
            isolated_wiki,
            "p.md",
            "---\ntitle: P\ntags: [d/ai-industry, invalid, t/analysis, s/2026]\n---\nbody\n",
        )
        original = path.read_text(encoding="utf-8")
        actions = apply_safe_fixes(
            check(),
            dry_run=True,
            reviewer=lambda _prompt, _schema: (_ for _ in ()).throw(
                AssertionError("dry-run must not call frontier")
            ),
        )

        assert actions and actions[0].startswith("[dry-run]")
        assert path.read_text(encoding="utf-8") == original
        assert not (isolated_wiki / "runtime" / "lint-safe-fixes").exists()


class TestBrokenLinkFrontierGate:
    def test_retarget_requires_frontier_and_binds_exact_preimage(
        self,
        isolated_wiki: Path,
    ) -> None:
        from chronovisor.ingest.lint import apply_safe_fixes, check

        source = _seed(
            isolated_wiki,
            "source.md",
            "---\ntitle: Source\ntags: [d/ai, t/analysis, s/2026]\n---\n[[known-pag|Known]]\n",
        )
        _seed(
            isolated_wiki,
            "known-page.md",
            "---\ntitle: Known\ntags: [d/ai, t/analysis, s/2026]\n---\nbody\n",
        )
        prompts: list[str] = []

        def approve(prompt, _schema):
            prompts.append(prompt)
            return _frontier_decision("approved")

        actions = apply_safe_fixes(check(), reviewer=approve)

        assert actions == ["[source] [[known-pag]] → [[known-page]] (1x)"]
        assert "[[known-page|Known]]" in source.read_text(encoding="utf-8")
        assert len(prompts) == 1
        assert '"expected_sha256"' in prompts[0]
        assert '"operation": "broken_link_retarget"' in prompts[0]

    def test_plaintext_requires_lookup_receipt_and_rechecks_index_before_write(
        self,
        isolated_wiki: Path,
    ) -> None:
        from chronovisor.ingest import lint as lint_mod

        source = _seed(
            isolated_wiki,
            "source.md",
            "---\ntitle: Source\ntags: [d/ai, t/analysis, s/2026]\n---\n[[missing-target]]\n",
        )
        original = source.read_text(encoding="utf-8")

        def approve_after_target_appears(_prompt, _schema):
            _seed(
                isolated_wiki,
                "missing-target.md",
                "---\ntitle: Arrived\ntags: [d/ai, t/analysis, s/2026]\n---\nbody\n",
            )
            return _frontier_decision("approved")

        actions = lint_mod.apply_safe_fixes(
            lint_mod.check(),
            reviewer=approve_after_target_appears,
        )

        assert actions == []
        assert source.read_text(encoding="utf-8") == original
        proposal_path = next(
            (isolated_wiki / "runtime" / "lint-safe-fixes" / "proposals").glob("*.json")
        )
        proposal = json.loads(proposal_path.read_text(encoding="utf-8"))["proposal"]
        receipt = proposal["details"]["target_lookup_receipt"]
        assert receipt["target_absent"] is True
        assert receipt["fuzzy_candidate"] is None
        assert receipt["no_acceptable_fuzzy_candidate"] is True
        assert len(receipt["receipt_sha256"]) == 64

    def test_retarget_rechecks_lookup_receipt_before_write(
        self,
        isolated_wiki: Path,
    ) -> None:
        from chronovisor.ingest import lint as lint_mod

        source = _seed(
            isolated_wiki,
            "source.md",
            "---\ntitle: Source\ntags: [d/ai, t/analysis, s/2026]\n---\n"
            "[[known-pag]]\n",
        )
        original = source.read_text(encoding="utf-8")
        _seed(
            isolated_wiki,
            "known-page.md",
            "---\ntitle: Fuzzy\ntags: [d/ai, t/analysis, s/2026]\n---\nbody\n",
        )

        def approve_after_exact_target_appears(_prompt, _schema):
            _seed(
                isolated_wiki,
                "known-pag.md",
                "---\ntitle: Exact\ntags: [d/ai, t/analysis, s/2026]\n---\nbody\n",
            )
            return _frontier_decision("approved")

        actions = lint_mod.apply_safe_fixes(
            lint_mod.check(),
            reviewer=approve_after_exact_target_appears,
        )

        assert actions == []
        assert source.read_text(encoding="utf-8") == original


class TestSafeFixReviewPackets:
    @pytest.mark.parametrize("decision", ["quarantined", "needs_retry"])
    def test_durable_hold_reuses_exact_proposal_evidence_and_authority(
        self,
        decision: str,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from chronovisor.ingest import lint as lint_mod

        proposal = lint_mod.build_semantic_mutation_proposal(
            page_id="p",
            operation="shared_lane_test",
            expected_text="before\n",
            updated_text="after\n",
            details={"reason": "test"},
        )
        calls = 0

        def review(_prompt, _schema):
            nonlocal calls
            calls += 1
            return _frontier_decision(decision)

        first = lint_mod.review_semantic_mutation(
            proposal,
            expected_text="before\n",
            updated_text="after\n",
            reviewer=review,
            artifact_dir=tmp_path,
            decision_lane="lint_safe_semantic_mutation",
            injected_reviewer=True,
        )
        monkeypatch.setattr(
            lint_mod,
            "_build_safe_fix_prompt",
            lambda *_args, **_kwargs: "prompt implementation changed",
        )
        second = lint_mod.review_semantic_mutation(
            proposal,
            expected_text="before\n",
            updated_text="after\n",
            reviewer=review,
            artifact_dir=tmp_path,
            decision_lane="lint_safe_semantic_mutation",
            injected_reviewer=True,
        )

        assert calls == 1
        assert first["decision"] == decision
        assert second["decision"] == decision
        assert second["reused"] is True
        artifact = json.loads(
            next((tmp_path / "frontier-verdicts").glob("*.json")).read_text(
                encoding="utf-8"
            )
        )
        assert artifact["schema_version"] == 3
        assert artifact["evidence_sha256"] == first["evidence_sha256"]
        assert artifact["authority_sha256"]
        assert artifact["hold_sha256"]

    def test_new_review_packet_invalidates_needs_retry_hold(
        self, tmp_path: Path
    ) -> None:
        from chronovisor.ingest import lint as lint_mod

        calls = 0

        def retry(_prompt, _schema):
            nonlocal calls
            calls += 1
            return _frontier_decision("needs_retry")

        first_proposal = lint_mod.build_semantic_mutation_proposal(
            page_id="p",
            operation="shared_lane_test",
            expected_text="before",
            updated_text="after-one",
            details={},
        )
        second_proposal = lint_mod.build_semantic_mutation_proposal(
            page_id="p",
            operation="shared_lane_test",
            expected_text="before",
            updated_text="after-two",
            details={},
        )
        lint_mod.review_semantic_mutation(
            first_proposal,
            expected_text="before",
            updated_text="after-one",
            reviewer=retry,
            artifact_dir=tmp_path,
            decision_lane="lint_safe_semantic_mutation",
            injected_reviewer=True,
        )
        lint_mod.review_semantic_mutation(
            second_proposal,
            expected_text="before",
            updated_text="after-two",
            reviewer=retry,
            artifact_dir=tmp_path,
            decision_lane="lint_safe_semantic_mutation",
            injected_reviewer=True,
        )

        assert calls == 2
        assert len(list((tmp_path / "frontier-verdicts").glob("*.json"))) == 2

    def test_large_page_uses_complete_changed_spans_without_truncation(
        self,
        tmp_path: Path,
    ) -> None:
        from chronovisor.ingest import lint as lint_mod

        before_lines = [f"unchanged line {index:05d}\n" for index in range(8_000)]
        after_lines = list(before_lines)
        after_lines[4_000] = "changed exact value\n"
        before = "".join(before_lines)
        after = "".join(after_lines)
        proposal = lint_mod.build_semantic_mutation_proposal(
            page_id="large",
            operation="resolve_nested_frontmatter_conflict",
            expected_text=before,
            updated_text=after,
            details={"conflicts": {"title": ["old", "new"]}},
        )
        prompts: list[str] = []

        result = lint_mod.review_semantic_mutation(
            proposal,
            expected_text=before,
            updated_text=after,
            reviewer=lambda prompt, _schema: (
                prompts.append(prompt) or _frontier_decision("approved")
            ),
            artifact_dir=tmp_path,
            decision_lane="page_normalize",
            injected_reviewer=True,
        )

        packet = proposal["review_packet"]
        assert result["decision"] == "approved"
        assert packet["mode"] == "changed_spans"
        assert packet["coverage"]["all_changed_spans_rendered"] is True
        assert packet["coverage"]["changed_span_count"] == len(packet["changed_spans"])
        assert proposal["details"]["review_receipt"]["truncated"] is False
        assert "changed exact value" in prompts[0]
        assert "bounded review payload" not in prompts[0]

    def test_unreviewable_repacket_is_durable_without_model_loop(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from chronovisor.decision import lint_mutation_contract as lint_contract
        from chronovisor.ingest import lint as lint_mod

        monkeypatch.setattr(
            lint_contract,
            "SAFE_FIX_REVIEW_PACKET_MAX_CHARS",
            400,
        )
        before = "source " * 2_000
        after = "generated " * 2_000
        proposal = lint_mod.build_semantic_mutation_proposal(
            page_id="large-metadata",
            operation="backfill_recall_metadata",
            expected_text=before,
            updated_text=after,
            details={"generated_frontmatter": {"summary": "generated"}},
        )

        first = lint_mod.review_semantic_mutation(
            proposal,
            expected_text=before,
            updated_text=after,
            reviewer=lambda *_args: (_ for _ in ()).throw(
                AssertionError("insufficient packet must not call reviewer")
            ),
            artifact_dir=tmp_path,
            decision_lane="metadata_backfill",
            injected_reviewer=True,
        )
        second = lint_mod.review_semantic_mutation(
            proposal,
            expected_text=before,
            updated_text=after,
            reviewer=lambda *_args: (_ for _ in ()).throw(
                AssertionError("durable insufficient hold must be reused")
            ),
            artifact_dir=tmp_path,
            decision_lane="metadata_backfill",
            injected_reviewer=True,
        )

        assert proposal["review_packet"]["mode"] == "insufficient"
        assert first["decision"] == "needs_retry"
        assert first["valid"] is True
        assert second["reused"] is True
        assert len(list((tmp_path / "frontier-verdicts").glob("*.json"))) == 1

    def test_review_recomputes_packet_from_exact_postimage_before_model(
        self,
        tmp_path: Path,
    ) -> None:
        from chronovisor.decision import lint_mutation_contract as lint_contract
        from chronovisor.ingest import lint as lint_mod

        proposal = lint_mod.build_semantic_mutation_proposal(
            page_id="p",
            operation="shared_lane_test",
            expected_text="before",
            updated_text="after",
            details={},
        )
        tampered = json.loads(json.dumps(proposal))
        tampered["review_packet"]["postimage"] = "attacker-selected postimage"
        tampered["details"]["review_receipt"] = lint_contract.review_receipt_from_packet(
            tampered["review_packet"]
        )

        result = lint_mod.review_semantic_mutation(
            tampered,
            expected_text="before",
            updated_text="after",
            reviewer=lambda *_args: (_ for _ in ()).throw(
                AssertionError("rehashed untrusted packet must not reach model")
            ),
            artifact_dir=tmp_path,
            decision_lane="lint_safe_semantic_mutation",
            injected_reviewer=True,
        )

        assert result["decision"] == "needs_retry"
        assert result["valid"] is True
        assert "recomputation" in result["summary"]


class TestPageNormalizeIdentityReceipt:
    def test_normalize_pages_repairs_yaml_without_changing_body(
        self,
        isolated_wiki: Path,
    ) -> None:
        from chronovisor.core.frontmatter import parse
        from chronovisor.ops import page_normalize

        page = _seed(
            isolated_wiki,
            "yaml.md",
            "---\ntitle: Agents-A1: Model Analysis\n"
            "recall_questions: [What changed?, Why now?]\n---\n"
            "# Body\n\nKeep this exact.\n",
        )
        original_body = parse(page.read_text(encoding="utf-8"))[1]

        preview = page_normalize.normalize_pages(
            root=isolated_wiki / "pages",
            write=False,
        )
        written = page_normalize.normalize_pages(
            root=isolated_wiki / "pages",
            write=True,
        )

        normalized = page.read_text(encoding="utf-8")
        assert preview["changed"] == 1
        assert written["changed"] == 1
        assert 'title: "Agents-A1: Model Analysis"' in normalized
        assert 'recall_questions: ["What changed?", "Why now?"]' in normalized
        assert parse(normalized)[1] == original_body

    def test_permalink_conflict_builds_durable_identity_quarantine(
        self,
        isolated_wiki: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from chronovisor.decision.decision_lane_prompts import (
            validate_identity_preflight_receipt,
        )
        from chronovisor.ops import page_normalize

        page = _seed(
            isolated_wiki,
            "account.md",
            "---\ntitle: Account\n"
            "permalink: wiki/pages/system/account\n---\n"
            "---\ntitle: Account\n"
            "permalink: wiki/pages/people/different-account\n---\nBody.\n",
        )
        original = page.read_text(encoding="utf-8")
        monkeypatch.setattr(page_normalize, "CHRONOVISOR_ROOT", isolated_wiki)
        prompts: list[str] = []

        first = page_normalize.normalize_pages(
            root=isolated_wiki / "pages",
            write=True,
            reviewer=lambda prompt, _schema: (
                prompts.append(prompt) or _frontier_decision("quarantined")
            ),
        )
        second = page_normalize.normalize_pages(
            root=isolated_wiki / "pages",
            write=True,
            reviewer=lambda *_args: (_ for _ in ()).throw(
                AssertionError("durable identity quarantine must be reused")
            ),
        )

        assert first["resolved_conflicts"] == []
        assert second["resolved_conflicts"] == []
        assert page.read_text(encoding="utf-8") == original
        assert len(prompts) == 1
        assert '"identity_preflight"' in prompts[0]
        artifact = json.loads(
            next(
                (isolated_wiki / "runtime" / "page-normalize" / "proposals").glob(
                    "*.json"
                )
            ).read_text(encoding="utf-8")
        )
        receipt = artifact["proposal"]["details"]["identity_preflight"]
        assert validate_identity_preflight_receipt(receipt) is True
        assert {binding["identity"] for binding in receipt["bindings"]} == {
            "wiki/pages/system/account",
            "wiki/pages/people/different-account",
        }

    def test_cached_holds_do_not_consume_page_normalize_call_budget(
        self,
        isolated_wiki: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from chronovisor.ops import page_normalize

        for name in ("a", "b", "c", "d"):
            _seed(
                isolated_wiki,
                f"{name}.md",
                f"---\ntitle: Outer {name}\n---\n"
                f"---\ntitle: Inner {name}\n---\nBody {name}.\n",
            )
        monkeypatch.setattr(page_normalize, "CHRONOVISOR_ROOT", isolated_wiki)
        calls: list[str] = []

        def first_reviewer(prompt, _schema):
            calls.append(prompt)
            return _frontier_decision("quarantined")

        first = page_normalize.normalize_pages(
            root=isolated_wiki / "pages",
            write=True,
            max_frontier_calls=3,
            reviewer=first_reviewer,
        )

        def second_reviewer(prompt, _schema):
            calls.append(prompt)
            return _frontier_decision("approved")

        second = page_normalize.normalize_pages(
            root=isolated_wiki / "pages",
            write=True,
            max_frontier_calls=1,
            reviewer=second_reviewer,
        )

        assert first["frontier_calls"] == 3
        assert second["frontier_calls"] == 1
        assert len(calls) == 4
        assert second["resolved_conflicts"] == [str(isolated_wiki / "pages" / "d.md")]


# ---------------------------------------------------------------------------
# tag_count_violation — medium severity, NOT auto-fixable
# ---------------------------------------------------------------------------


class TestTagCountViolation:
    def test_too_few_d_tags_flagged(self, isolated_wiki: Path) -> None:
        from chronovisor.ingest.lint import check

        _seed(
            isolated_wiki,
            "p.md",
            "---\ntitle: P\nupdated: 2026-05-08\n"
            "tags: [t/analysis, s/2026]\n"  # missing d/
            "---\nbody\n",
        )
        issues = check()
        flagged = _by_type(issues, "tag_count_violation", "p")
        assert len(flagged) == 1
        assert flagged[0]["auto_fixable"] is False
        assert "d/" in flagged[0]["detail"]

    def test_too_many_d_tags_flagged(self, isolated_wiki: Path) -> None:
        from chronovisor.ingest.lint import check

        _seed(
            isolated_wiki,
            "p.md",
            "---\ntitle: P\nupdated: 2026-05-08\n"
            "tags: [d/a, d/b, d/c, d/d, t/analysis, s/2026]\n"
            "---\nbody\n",
        )
        issues = check()
        flagged = _by_type(issues, "tag_count_violation", "p")
        assert len(flagged) == 1

    def test_too_many_t_tags_flagged(self, isolated_wiki: Path) -> None:
        from chronovisor.ingest.lint import check

        _seed(
            isolated_wiki,
            "p.md",
            "---\ntitle: P\nupdated: 2026-05-08\n"
            "tags: [d/ai-industry, t/analysis, t/howto, s/2026]\n"
            "---\nbody\n",
        )
        issues = check()
        assert _by_type(issues, "tag_count_violation", "p")
