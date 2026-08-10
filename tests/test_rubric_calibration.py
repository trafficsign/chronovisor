from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from chronovisor.core import ollama
from chronovisor.core.durable_state import read_sealed_json, write_sealed_json
from chronovisor.recall import rubric_calibration
from chronovisor.recall.rubric_calibration import (
    DEFAULT_RUBRIC,
    STRATA,
    build_locked_gold_cycle,
    build_rubric_artifact,
    evaluate_judges,
    load_active_rubric,
    promote_candidate,
    run_calibration_cycle,
    select_diverse_cases,
    write_candidate,
)
from chronovisor.search.search_eval import SearchExample, write_sealed_manifest


def _route(
    role: str,
    model: str,
    *,
    provider: str = "ollama",
    location: str = "local",
    structured_output: bool = True,
) -> ollama.RuntimeGenerationRoute:
    return ollama.RuntimeGenerationRoute(
        role=role,
        provider=provider,
        model=model,
        location=location,
        structured_output=structured_output,
    )


def _router(result: object | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        config_error=None,
        routes={
            role: _route(f"classification.{role}", f"{role}:test")
            for role in ("primary", "challenger", "tie_break")
        },
        decide=lambda *_args, **_kwargs: result,
    )


def _runtime(
    result: object | None = None,
) -> tuple[ollama.RuntimeGenerationRoute, SimpleNamespace, dict[str, object], str]:
    variant = _route("recall.rubric.variant", "variant:test")
    router = _router(result)
    manifest = {
        "schema_version": 1,
        "routes": [
            {
                "role": route.role,
                "provider": route.provider,
                "model": route.model,
                "location": route.location,
                "model_digest": None,
            }
            for route in [variant, *router.routes.values()]
        ],
    }
    return (
        variant,
        router,
        manifest,
        rubric_calibration.canonical_json_sha256_stringifying(manifest),
    )


def test_diverse_selection_excludes_query_and_session_duplicates() -> None:
    rows = [
        {
            "stratum": stratum,
            "query_sha256": f"q-{index}",
            "session_hash": f"s-{index}",
        }
        for index, stratum in enumerate(
            ["relevant", "multi_hop", "hub_false_positive", "topic_switch"]
        )
    ]
    rows.append({"stratum": "stale_info", "query_sha256": "q-0", "session_hash": "new"})

    selected = select_diverse_cases(rows, limit=10)

    assert len(selected) == 4
    assert len({row["query_sha256"] for row in selected}) == len(selected)
    assert len({row["session_hash"] for row in selected}) == len(selected)


def test_judge_metrics_include_calibration_correlation_and_ensemble_gain() -> None:
    rows = [
        {
            "gold": True,
            "primary": True,
            "challenger": False,
            "tie_break": True,
            "ensemble": True,
            "primary_confidence": 0.9,
            "challenger_confidence": 0.6,
            "tie_break_confidence": 0.8,
            "ensemble_confidence": 0.9,
        },
        {
            "gold": False,
            "primary": False,
            "challenger": False,
            "tie_break": True,
            "ensemble": False,
            "primary_confidence": 0.9,
            "challenger_confidence": 0.8,
            "tie_break_confidence": 0.6,
            "ensemble_confidence": 0.9,
        },
    ]

    metrics = evaluate_judges(rows)

    assert metrics["models"]["ensemble"]["accuracy"] == 1.0
    assert "primary:challenger" in metrics["pairwise_error_correlation"]
    assert metrics["unanimous_wrong_rate"] == 0.0
    assert metrics["ensemble_gain"] == 0.0


def test_active_rubric_is_sealed_and_promotion_fails_closed(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.json"
    active = tmp_path / "active.json"
    lkg = tmp_path / "lkg.json"
    artifact = build_rubric_artifact(rubric_text="Only useful grounded evidence.")
    write_candidate(artifact, candidate)

    held = promote_candidate(
        candidate_file=candidate,
        active_file=active,
        last_known_good_file=lkg,
        metrics={
            "ensemble_gain": 0.1,
            "models": {"ensemble": {"precision": 1.0, "ece": 0.0, "abstention": 0.0}},
            "strata_counts": {stratum: 1 for stratum in STRATA},
            "session_count": 5,
            "split_counts": {
                split: {"positive": 1, "negative": 1}
                for split in ("train", "dev", "locked-test")
            },
        },
        gold_count=10,
    )
    assert held["status"] == "held"
    assert not active.exists()

    insufficient_sessions = promote_candidate(
        candidate_file=candidate,
        active_file=active,
        last_known_good_file=lkg,
        metrics={
            "ensemble_gain": 0.1,
            "models": {"ensemble": {"precision": 1.0, "ece": 0.0, "abstention": 0.0}},
            "strata_counts": {stratum: 1 for stratum in STRATA},
            "session_count": 4,
            "split_counts": {
                split: {"positive": 1, "negative": 1}
                for split in ("train", "dev", "locked-test")
            },
        },
        gold_count=30,
    )
    assert insufficient_sessions["status"] == "held"
    assert insufficient_sessions["gates"]["session_diversity"] is False
    assert not active.exists()

    no_ensemble_gain = promote_candidate(
        candidate_file=candidate,
        active_file=active,
        last_known_good_file=lkg,
        metrics={
            "ensemble_gain": 0.0,
            "models": {"ensemble": {"precision": 1.0, "ece": 0.0, "abstention": 0.0}},
            "strata_counts": {stratum: 1 for stratum in STRATA},
            "session_count": 5,
            "split_counts": {
                split: {"positive": 1, "negative": 1}
                for split in ("train", "dev", "locked-test")
            },
        },
        gold_count=30,
    )
    assert no_ensemble_gain["status"] == "held"
    assert no_ensemble_gain["gates"]["ensemble_value"] is False
    assert not active.exists()

    adopted = promote_candidate(
        candidate_file=candidate,
        active_file=active,
        last_known_good_file=lkg,
        metrics={
            "ensemble_gain": 0.1,
            "models": {"ensemble": {"precision": 1.0, "ece": 0.0, "abstention": 0.0}},
            "strata_counts": {stratum: 1 for stratum in STRATA},
            "session_count": 5,
            "split_counts": {
                split: {"positive": 1, "negative": 1}
                for split in ("train", "dev", "locked-test")
            },
        },
        gold_count=30,
    )
    assert adopted["status"] == "adopted"
    assert load_active_rubric(active)["rubric_text"] == "Only useful grounded evidence."

    active.write_text("{}", encoding="utf-8")
    assert load_active_rubric(active)["rubric_text"] == DEFAULT_RUBRIC


def test_calibration_cycle_requires_background_local_consensus(
    tmp_path: Path, monkeypatch
) -> None:
    rows_file = tmp_path / "locked-gold.jsonl"
    consensus_result = SimpleNamespace(
        ok=True,
        value={
            "decision": "approved",
            "holdout_non_regression": True,
            "calibration_improved": True,
            "coverage_preserved": True,
            "rollback_safe": True,
        },
        agreement_sha256="a" * 64,
        failure_class=None,
        votes=(),
    )
    runtime = _runtime(consensus_result)
    prompts: list[str] = []
    sources: list[object] = []

    def decide(prompt: str, *_args: object, **kwargs: object) -> object:
        prompts.append(prompt)
        sources.append(kwargs["source"])
        return consensus_result

    runtime[1].decide = decide
    route_manifest = runtime[2]
    route_manifest_sha256 = runtime[3]
    rows = []
    for index in range(30):
        gold = index % 2 == 0
        rows.append(
            {
                "case_id": f"case-{index}",
                "reviewed": True,
                "stratum": STRATA[index % len(STRATA)],
                "split": ("train", "dev", "locked-test")[index % 3],
                "query_sha256": f"query-{index}",
                "session_hash": f"{index:064x}",
                "gold": gold,
                "current": gold,
                "generated": gold,
                "diverse_few_shot": gold,
                "calibrated": gold,
                "primary": (not gold) if index % 5 == 0 else gold,
                "challenger": (not gold) if index % 7 == 0 else gold,
                "tie_break": (not gold) if index % 11 == 0 else gold,
                "ensemble": gold,
                "primary_confidence": 0.95,
                "challenger_confidence": 0.95,
                "tie_break_confidence": 0.95,
                "ensemble_confidence": 0.95,
                "route_manifest": route_manifest,
                "route_manifest_sha256": route_manifest_sha256,
            }
        )
    rows_file.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    monkeypatch.setattr(
        rubric_calibration,
        "_cycle_runtime",
        lambda _lane: runtime,
    )

    result = run_calibration_cycle(
        rows_file=rows_file,
        candidate_file=tmp_path / "candidate.json",
        active_file=tmp_path / "active.json",
        last_known_good_file=tmp_path / "lkg.json",
        status_file=tmp_path / "status.json",
        outcomes_file=tmp_path / "outcomes.jsonl",
    )

    assert result["status"] == "adopted"
    assert result["gates"]["local_consensus"] is True
    assert result["consensus"]["passed"] is True
    assert result["route_manifest_sha256"] == route_manifest_sha256
    assert result["consensus"]["route_manifest_sha256"] == route_manifest_sha256
    assert result["external_model_calls"] == 0
    assert str(tmp_path) not in prompts[0]
    assert ollama.source_data_classification_values(sources[0]) == (
        "derived_snippet",
        "high",
    )
    receipt = json.loads(
        (tmp_path / "consensus-receipts.jsonl").read_text(encoding="utf-8")
    )
    assert receipt["route_manifest_sha256"] == route_manifest_sha256


def test_locked_gold_builder_is_incremental_local_and_privacy_safe(
    tmp_path: Path, monkeypatch
) -> None:
    (tmp_path / "pages").mkdir()
    (tmp_path / "pages" / "answer.md").write_text(
        "Grounded answer body.", encoding="utf-8"
    )
    golden = tmp_path / "recall" / "search-golden.jsonl"
    golden.parent.mkdir()
    golden.write_text(
        json.dumps(
            {
                "query": "What is the grounded answer?",
                "expected_pages": ["answer"],
                "negative_pages": [],
                "stale_pages": [],
                "reviewed": True,
                "ref": "manual-1",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "recall" / "feedback.jsonl").write_text(
        json.dumps(
            {
                "ref": "manual-1",
                "snapshot": {"session_id": "private-session-id"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "runtime" / "search-eval" / "manual-94-manifest.json"
    manifest.parent.mkdir(parents=True)
    write_sealed_manifest(
        [
            SearchExample(
                query="What is the grounded answer?",
                expected_pages=("answer",),
                reviewed=True,
                ref="manual-1",
            )
        ],
        manifest,
    )
    monkeypatch.setattr(
        rubric_calibration,
        "_judge_variant",
        lambda *_args, **_kwargs: (True, 0.95, 0),
    )
    monkeypatch.setattr(
        rubric_calibration,
        "_judge_consensus",
        lambda *_args, **_kwargs: {
            "primary": True,
            "primary_confidence": 0.95,
            "challenger": True,
            "challenger_confidence": 0.95,
            "tie_break": "abstain",
            "tie_break_confidence": 0.0,
            "ensemble": True,
            "ensemble_confidence": 0.95,
            "consensus_receipt_sha256": "a" * 64,
            "external_model_calls": 0,
            "retry": False,
        },
    )
    monkeypatch.setattr(
        rubric_calibration,
        "_cycle_runtime",
        lambda _lane: _runtime(),
    )
    output = tmp_path / "runtime" / "recall-rubric" / "locked-gold.jsonl"
    state = output.parent / "state.json"

    for _ in range(5):
        result = build_locked_gold_cycle(
            root=tmp_path,
            golden_file=golden,
            output_file=output,
            state_file=state,
            max_steps_per_day=10,
        )

    row = json.loads(output.read_text(encoding="utf-8"))
    assert result["cases"] == 1
    assert row["gold"] is True
    assert row["ensemble"] is True
    assert len(row["session_hash"]) == 64
    assert row["session_hash"] != "private-session-id"
    assert "query" not in row
    assert "page_id" not in row
    assert "Grounded answer body" not in output.read_text(encoding="utf-8")

    tampered = json.loads(manifest.read_text(encoding="utf-8"))
    tampered["entries"][0]["source"] = "tampered"
    manifest.write_text(json.dumps(tampered), encoding="utf-8")
    assert rubric_calibration._gold_cases(tmp_path, golden) == []


def test_variant_route_binds_exact_model_location_source_and_session_broker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "pages").mkdir()
    (tmp_path / "pages" / "answer.md").write_text("page", encoding="utf-8")
    sessions: list[dict[str, object]] = []

    class Session:
        def __init__(self, **kwargs: object) -> None:
            sessions.append(kwargs)

        def run(self, *_args: object, **_kwargs: object) -> object:
            return SimpleNamespace(
                ok=True,
                value={
                    "decision": "approved",
                    "topically_relevant": True,
                    "marginally_useful": True,
                    "read_worthy": True,
                    "stale_or_harmful": False,
                    "confidence": 0.9,
                },
                failure_class=None,
            )

    monkeypatch.setattr(rubric_calibration, "LocalStructuredSession", Session)
    prediction, confidence, external_calls = rubric_calibration._judge_variant(
        {"query": "RAW-CANARY", "page_id": "answer", "stratum": "relevant"},
        rubric_name="current",
        route=_route(
            "recall.rubric.variant",
            "remote-exact",
            provider="openai",
            location="remote",
        ),
        root=tmp_path,
    )

    assert prediction is True
    assert confidence == 0.9
    assert external_calls == 1
    assert sessions == [
        {
            "model": "remote-exact",
            "runtime_role": "recall.rubric.variant",
            "runtime_location": "remote",
            "source_data_class": "raw",
            "source_sensitivity": "high",
            "role": "recall_rubric:current",
            "audit_root": tmp_path
            / "runtime"
            / "recall-rubric"
            / "structured-audit",
            "num_ctx": 8_192,
            "num_predict": 160,
            "keep_alive": "20m",
            "read_timeout_ms": 180_000,
            "max_input_chars": 12_000,
            "max_output_chars": 2_000,
            "max_responses": 2,
            "resource_lease_timeout_ms": 25,
        }
    ]
    assert "resource_managed" not in sessions[0]


def test_variant_remote_system_and_egress_denial_do_not_count_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "system").mkdir()
    (tmp_path / "system" / "private.md").write_text("SYSTEM-CANARY", encoding="utf-8")
    calls = 0

    class Session:
        def __init__(self, **_kwargs: object) -> None:
            nonlocal calls
            calls += 1

        def run(self, *_args: object, **_kwargs: object) -> object:
            return SimpleNamespace(ok=False, value=None, failure_class="egress_denied")

    monkeypatch.setattr(rubric_calibration, "LocalStructuredSession", Session)
    remote = _route(
        "recall.rubric.variant",
        "remote",
        provider="openai",
        location="remote",
    )

    assert rubric_calibration._judge_variant(
        {"query": "RAW-CANARY", "page_id": "private"},
        rubric_name="current",
        route=remote,
        root=tmp_path,
    ) == (None, 0.0, 0)
    assert calls == 0

    (tmp_path / "pages").mkdir()
    (tmp_path / "pages" / "answer.md").write_text("page", encoding="utf-8")
    assert rubric_calibration._judge_variant(
        {"query": "RAW-CANARY", "page_id": "answer"},
        rubric_name="current",
        route=remote,
        root=tmp_path,
    ) == (None, 0.0, 0)
    assert calls == 1


def test_variant_local_system_uses_system_high(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "system").mkdir()
    (tmp_path / "system" / "private.md").write_text("system", encoding="utf-8")
    sessions: list[dict[str, object]] = []

    class Session:
        def __init__(self, **kwargs: object) -> None:
            sessions.append(kwargs)

        def run(self, *_args: object, **_kwargs: object) -> object:
            return SimpleNamespace(ok=False, value=None, failure_class="backend_error")

    monkeypatch.setattr(rubric_calibration, "LocalStructuredSession", Session)
    rubric_calibration._judge_variant(
        {"query": "RAW", "page_id": "private"},
        rubric_name="current",
        route=_route("recall.rubric.variant", "local"),
        root=tmp_path,
    )

    assert sessions[0]["source_data_class"] == "system"
    assert sessions[0]["source_sensitivity"] == "high"


def test_consensus_source_is_explicit_and_remote_system_fails_closed(
    tmp_path: Path,
) -> None:
    (tmp_path / "pages").mkdir()
    (tmp_path / "system").mkdir()
    (tmp_path / "pages" / "answer.md").write_text("page", encoding="utf-8")
    (tmp_path / "system" / "private.md").write_text("system", encoding="utf-8")
    sources: list[object] = []
    result = SimpleNamespace(
        ok=True,
        value={"decision": "abstained", "confidence": 0.4},
        agreement_sha256="a" * 64,
        votes=(),
    )
    router = _router(result)
    router.decide = lambda *_args, **kwargs: sources.append(kwargs["source"]) or result

    page = rubric_calibration._judge_consensus(
        {"query": "RAW-CANARY", "page_id": "answer"},
        root=tmp_path,
        router=router,
    )
    assert page["retry"] is False
    assert ollama.source_data_classification_values(sources[-1]) == ("raw", "high")

    router.routes["tie_break"] = _route(
        "classification.tie_break",
        "remote",
        provider="openai",
        location="remote",
    )
    system = rubric_calibration._judge_consensus(
        {"query": "RAW-CANARY", "page_id": "private"},
        root=tmp_path,
        router=router,
    )
    assert system["retry"] is True
    assert system["external_model_calls"] == 0
    assert len(sources) == 1

    router.routes["tie_break"] = _route("classification.tie_break", "tie")
    rubric_calibration._judge_consensus(
        {"query": "RAW-CANARY", "page_id": "private"},
        root=tmp_path,
        router=router,
    )
    assert ollama.source_data_classification_values(sources[-1]) == (
        "system",
        "high",
    )


def test_route_manifest_hashes_ordered_routes_and_optional_local_digests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    variant = _route("recall.rubric.variant", "variant")
    router = _router()
    router.routes["challenger"] = _route(
        "classification.challenger",
        "challenger",
        provider="nemotron",
    )
    router.routes["tie_break"] = _route(
        "classification.tie_break",
        "tie",
        provider="openai",
        location="remote",
    )
    requested: list[tuple[str, ...]] = []

    def digests(models: tuple[str, ...]) -> dict[str, str]:
        requested.append(models)
        return {"variant": "variant-digest", "primary:test": ""}

    monkeypatch.setattr(ollama, "model_digests", digests)
    manifest, missing_hash = rubric_calibration._route_manifest(variant, router)

    assert requested == [("variant", "primary:test")]
    assert [row["role"] for row in manifest["routes"]] == [
        "recall.rubric.variant",
        "classification.primary",
        "classification.challenger",
        "classification.tie_break",
    ]
    assert [row["model_digest"] for row in manifest["routes"]] == [
        "variant-digest",
        None,
        None,
        None,
    ]

    monkeypatch.setattr(
        ollama,
        "model_digests",
        lambda _models: {"variant": "variant-digest", "primary:test": "old"},
    )
    _, prior_hash = rubric_calibration._route_manifest(variant, router)
    assert missing_hash != prior_hash


@pytest.mark.parametrize("mode", ["empty", "dry", "budget"])
def test_locked_builder_zero_work_does_not_resolve_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    cases = [] if mode == "empty" else [{"case_id": "case"}]
    monkeypatch.setattr(rubric_calibration, "_gold_cases", lambda *_args: cases)
    monkeypatch.setattr(
        rubric_calibration,
        "_cycle_runtime",
        lambda _lane: pytest.fail("zero-work path resolved runtime"),
    )
    state = tmp_path / "state.json"
    if mode == "budget":
        write_sealed_json(
            state,
            {
                "date": rubric_calibration.datetime.now(
                    rubric_calibration.UTC
                ).date().isoformat(),
                "steps_today": 1,
            },
        )

    result = build_locked_gold_cycle(
        root=tmp_path,
        golden_file=tmp_path / "gold.jsonl",
        output_file=tmp_path / "locked.jsonl",
        state_file=state,
        max_steps_per_day=1,
        dry_run=mode == "dry",
    )

    assert result["external_model_calls"] == 0


def test_manifest_drift_clears_pending_predictions_without_raw_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = {
        "case_id": "case-1",
        "query": "RAW-QUERY-CANARY",
        "query_sha256": "q",
        "page_id": "PAGE-ID-CANARY",
        "page_id_sha256": "p",
        "gold": True,
        "stratum": "relevant",
        "split": "train",
        "session_hash": "1" * 64,
        "review_receipt_id": "receipt",
    }
    monkeypatch.setattr(rubric_calibration, "_gold_cases", lambda *_args: [case])
    runtime = _runtime()
    monkeypatch.setattr(rubric_calibration, "_cycle_runtime", lambda _lane: runtime)
    judged: list[str] = []

    def judge(*_args: object, rubric_name: str, **_kwargs: object) -> tuple[None, float, int]:
        judged.append(rubric_name)
        return None, 0.0, 0

    monkeypatch.setattr(rubric_calibration, "_judge_variant", judge)
    state = tmp_path / "state.json"
    write_sealed_json(
        state,
        {
            "date": rubric_calibration.datetime.now(
                rubric_calibration.UTC
            ).date().isoformat(),
            "pending": {
                "case_id": "case-1",
                "route_manifest_sha256": "0" * 64,
                "predictions": {"generated": True},
            },
        },
    )

    build_locked_gold_cycle(
        root=tmp_path,
        golden_file=tmp_path / "gold.jsonl",
        output_file=tmp_path / "locked.jsonl",
        state_file=state,
        max_steps_per_day=10,
    )

    payload = read_sealed_json(state)
    assert judged == ["current"]
    assert payload["pending"]["predictions"] == {}
    durable = state.read_text(encoding="utf-8")
    assert "RAW-QUERY-CANARY" not in durable
    assert "PAGE-ID-CANARY" not in durable


def test_current_locked_rows_require_exact_sealed_manifest(tmp_path: Path) -> None:
    _, _, manifest, manifest_sha256 = _runtime()
    required = {
        "reviewed": True,
        "split": "train",
        "session_hash": "1" * 64,
        **{name: True for name in (*rubric_calibration.RUBRIC_VARIANTS, "primary", "challenger", "tie_break", "ensemble")},
    }
    other_manifest = {**manifest, "schema_version": 2}
    rows = [
        required,
        {
            **required,
            "route_manifest": other_manifest,
            "route_manifest_sha256": rubric_calibration.canonical_json_sha256_stringifying(
                other_manifest
            ),
        },
        {
            **required,
            "route_manifest": manifest,
            "route_manifest_sha256": manifest_sha256,
        },
    ]
    path = tmp_path / "locked.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    assert len(
        rubric_calibration._locked_gold_rows(
            path, manifest_sha256=manifest_sha256
        )
    ) == 1


def test_artifact_identity_uses_current_manifest_hash() -> None:
    _, _, manifest, manifest_sha256 = _runtime()
    artifact = build_rubric_artifact(
        rubric_text="bounded",
        model_sha256="legacy-model-id",
        route_manifest=manifest,
        route_manifest_sha256=manifest_sha256,
    )
    changed_manifest = {**manifest, "schema_version": 2}
    changed_hash = rubric_calibration.canonical_json_sha256_stringifying(
        changed_manifest
    )
    changed = build_rubric_artifact(
        rubric_text="bounded",
        model_sha256="legacy-model-id",
        route_manifest=changed_manifest,
        route_manifest_sha256=changed_hash,
    )

    assert artifact["model_sha256"] == manifest_sha256
    assert artifact["rubric_id"] != changed["rubric_id"]


def test_route_failure_is_safe_and_does_not_touch_router(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    monkeypatch.setattr(
        ollama,
        "runtime_generation_routes",
        lambda _roles: (_route("wrong.role", "RAW-ENDPOINT-CANARY"),),
    )

    def router(*_args: object) -> object:
        nonlocal calls
        calls += 1
        return _router()

    monkeypatch.setattr(rubric_calibration, "router_for_producer", router)

    with pytest.raises(ollama.RuntimeBridgeError) as error:
        rubric_calibration._cycle_runtime("recall_usefulness_judgment")
    assert error.value.category == "route_configuration_invalid"
    assert "RAW-ENDPOINT-CANARY" not in str(error.value)
    assert calls == 0


def test_page_excerpt_rejects_namespace_symlink(tmp_path: Path) -> None:
    (tmp_path / "pages").mkdir()
    (tmp_path / "system").mkdir()
    outside = tmp_path / "outside.md"
    outside.write_text("RAW-SYMLINK-CANARY", encoding="utf-8")
    (tmp_path / "system" / "private.md").symlink_to(outside)

    assert rubric_calibration._page_excerpt_source(tmp_path, "private") == (
        "",
        "page",
    )


def test_empty_calibration_does_not_resolve_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        rubric_calibration,
        "_cycle_runtime",
        lambda _lane: pytest.fail("empty calibration resolved runtime"),
    )

    result = run_calibration_cycle(
        rows_file=tmp_path / "missing.jsonl",
        status_file=tmp_path / "status.json",
    )

    assert result["status"] == "collecting"
    assert result["external_model_calls"] == 0
