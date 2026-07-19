from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from chronovisor import decision_policy, decision_router, frontier_review
from chronovisor.decision_policy import (
    DECISION_POLICIES,
    decision_policy_snapshot,
    resolve_decision_policy,
)
from chronovisor.decision_router import DecisionRouterResult
from chronovisor.decision_schema_manifest import production_decision_schemas
from tests.semantic_hold_support import semantic_authority


SCHEMA = frontier_review.FRONTIER_DECISION_SCHEMA


class FakeRouter:
    source = "bootstrap_current_policy"
    calls = 0
    router_audit: dict[str, object] = {}

    def __init__(self, **_kwargs) -> None:
        policy_audit = {**type(self).router_audit, "source": self.source}
        self.policy = SimpleNamespace(
            source=self.source,
            audit_record=lambda: dict(policy_audit),
        )

    def decide(self, _prompt, _schema):
        type(self).calls += 1
        return DecisionRouterResult(
            status="agreed",
            value={
                "decision": "approved",
                "summary": "accepted",
                "tests_run": [],
                "commit": None,
                "committed": False,
                "pushed": False,
                "risk": None,
                "notes": None,
            },
            agreement_sha256="a" * 64,
        )


@pytest.fixture(autouse=True)
def isolate_structured_review_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> SimpleNamespace:
    """Keep adopted authority and semantic-cache locks deterministic and local."""

    artifact_sha256 = "d" * 64
    authority = semantic_authority(artifact_sha256=artifact_sha256)
    router_audit = authority["router"]
    assert isinstance(router_audit, dict)
    FakeRouter.source = "bootstrap_current_policy"
    FakeRouter.calls = 0
    FakeRouter.router_audit = dict(router_audit)
    cache_root = tmp_path / "structured-review-holds"
    cache_roots: list[Path] = []
    real_cache = frontier_review.semantic_hold.StructuredReviewSemanticHoldCache

    def isolated_cache(*, root: Path | None = None):
        assert root is not None
        resolved_root = Path(root).resolve()
        assert resolved_root == cache_root.resolve()
        cache_roots.append(resolved_root)
        return real_cache(root=root)

    monkeypatch.setattr(
        frontier_review,
        "STRUCTURED_REVIEW_HOLD_CACHE_ROOT",
        cache_root,
    )
    monkeypatch.setattr(
        frontier_review.semantic_hold,
        "StructuredReviewSemanticHoldCache",
        isolated_cache,
    )
    monkeypatch.setattr(
        frontier_review,
        "_current_structured_authority",
        lambda lane: (
            semantic_authority(lane, artifact_sha256=artifact_sha256),
            None,
        ),
    )
    monkeypatch.setattr(
        frontier_review,
        "_structured_authority_observation",
        lambda _authority: "0" * 64,
    )
    return SimpleNamespace(cache_root=cache_root, cache_roots=cache_roots)


def test_unknown_lane_fails_closed_without_starting_models(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeRouter.calls = 0
    monkeypatch.setattr(decision_router, "DecisionRouter", FakeRouter)

    result = frontier_review.run_structured_review(
        "review",
        SCHEMA,
        repo_root=tmp_path,
        decision_lane="not-registered",
    )

    assert result["decision"] == "needs_retry"
    assert (
        result["frontier_failure"]["failure_class"] == "local_decision_policy_blocked"
    )
    assert FakeRouter.calls == 0


def test_shadow_lane_collects_vote_but_cannot_authorize_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeRouter.calls = 0
    FakeRouter.source = "bootstrap_current_policy"
    monkeypatch.setattr(decision_router, "DecisionRouter", FakeRouter)
    monkeypatch.setattr(decision_policy, "load_toml_file", lambda *_args: {})

    result = frontier_review.run_structured_review(
        "review",
        SCHEMA,
        repo_root=tmp_path,
        decision_lane="recall_auto_apply",
    )

    assert FakeRouter.calls == 1
    assert result["decision"] == "needs_retry"
    assert result["decision_policy"]["mode"] == "shadow"
    assert result["frontier_failure"]["failure_class"] == "local_decision_shadow_only"


def test_enabled_lane_requires_adopted_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeRouter.calls = 0
    FakeRouter.source = "bootstrap_current_policy"
    monkeypatch.setattr(decision_router, "DecisionRouter", FakeRouter)
    monkeypatch.setenv("CHRONOVISOR_DECISION_POLICY_RECALL_AUTO_APPLY", "enabled")

    result = frontier_review.run_structured_review(
        "review",
        SCHEMA,
        repo_root=tmp_path,
        decision_lane="recall_auto_apply",
    )

    assert result["decision"] == "needs_retry"
    assert (
        result["frontier_failure"]["failure_class"]
        == "local_decision_artifact_required"
    )


def test_enabled_lane_can_return_only_adopted_consensus(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolate_structured_review_runtime: SimpleNamespace,
) -> None:
    FakeRouter.calls = 0
    FakeRouter.source = "adopted_artifact"
    monkeypatch.setattr(decision_router, "DecisionRouter", FakeRouter)
    monkeypatch.setenv("CHRONOVISOR_DECISION_POLICY_RECALL_AUTO_APPLY", "enabled")

    result = frontier_review.run_structured_review(
        "review",
        SCHEMA,
        repo_root=tmp_path,
        decision_lane="recall_auto_apply",
    )

    assert result["decision"] == "approved"
    assert result["decision_policy"]["router_policy"]["source"] == "adopted_artifact"
    cache_root = isolate_structured_review_runtime.cache_root
    assert frontier_review.STRUCTURED_REVIEW_HOLD_CACHE_ROOT == cache_root
    assert isolate_structured_review_runtime.cache_roots == [cache_root.resolve()]
    assert list((cache_root / "locks").glob("*.lock"))


def test_every_registered_lane_names_a_production_schema() -> None:
    schemas = production_decision_schemas()
    assert DECISION_POLICIES
    structured = {
        policy.schema_name
        for policy in DECISION_POLICIES.values()
        if policy.kind in {"consensus", "local_batch"}
    }
    assert structured <= set(schemas)


def test_every_production_structured_review_call_names_a_lane() -> None:
    src_root = Path(__file__).resolve().parents[1] / "src" / "chronovisor"
    missing: list[str] = []
    for path in sorted(src_root.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            function = node.func
            name = (
                function.attr
                if isinstance(function, ast.Attribute)
                else function.id
                if isinstance(function, ast.Name)
                else ""
            )
            if name != "run_structured_review":
                continue
            keywords = {keyword.arg for keyword in node.keywords if keyword.arg}
            if "decision_lane" not in keywords:
                missing.append(f"{path.name}:{node.lineno}")
    assert missing == []


def test_policy_snapshot_separates_structured_shadow_from_deterministic_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(decision_policy, "load_toml_file", lambda *_args: {})
    for lane in DECISION_POLICIES:
        monkeypatch.delenv(
            "CHRONOVISOR_DECISION_POLICY_" + lane.upper(),
            raising=False,
        )
    snapshot = decision_policy_snapshot()
    structured = sum(
        policy.kind in {"consensus", "local_batch"}
        for policy in DECISION_POLICIES.values()
    )
    deterministic = len(DECISION_POLICIES) - structured
    assert snapshot["counts"]["shadow"] == structured
    assert snapshot["counts"]["enabled"] == deterministic
    assert resolve_decision_policy(None)[2] == "decision_lane_required"


def test_schema_mismatch_stops_before_any_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeRouter.calls = 0
    monkeypatch.setattr(decision_router, "DecisionRouter", FakeRouter)
    monkeypatch.setenv("CHRONOVISOR_DECISION_POLICY_RECALL_AUTO_APPLY", "enabled")

    result = frontier_review.run_structured_review(
        "review",
        {
            "type": "object",
            "additionalProperties": False,
            "required": ["unexpected"],
            "properties": {"unexpected": {"type": "string"}},
        },
        repo_root=tmp_path,
        decision_lane="recall_auto_apply",
    )

    assert (
        result["frontier_failure"]["failure_class"] == "local_decision_schema_mismatch"
    )
    assert FakeRouter.calls == 0


def test_non_structured_lane_cannot_enter_model_router(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeRouter.calls = 0
    monkeypatch.setattr(decision_router, "DecisionRouter", FakeRouter)

    result = frontier_review.run_structured_review(
        "review",
        SCHEMA,
        repo_root=tmp_path,
        decision_lane="raw_capture",
    )

    assert result["decision"] == "needs_retry"
    assert (
        result["frontier_failure"]["failure_class"]
        == "local_decision_policy_kind_invalid"
    )
    assert FakeRouter.calls == 0
