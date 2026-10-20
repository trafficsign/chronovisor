from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from chronovisor.core import llm_config, model_execution
from chronovisor.core.llm_runtime import (
    EgressDeniedError,
    GenerationRequest,
    GenerationResult,
    GenerationRoute,
    LLMRuntime,
    MessageGenerationRequest,
    RouteLocation,
    SourceDataClass,
    SourceDataClassification,
    SourceSensitivity,
)
from chronovisor.core.omlx_adapter import OMLXAdapter
from chronovisor.decision import frontier_review, native_luna_authority
from chronovisor.decision.local_structured import LocalStructuredSession
from chronovisor.ops.model_execution_status import observed_drain
from chronovisor.recall import recall_distillation as distill
from scripts import recall_luna_supervisor as supervisor


def test_cloud_revocation_blocks_native_overrides_and_direct_spawn(tmp_path, monkeypatch):
    monkeypatch.setenv("CHRONOVISOR_ROOT", str(tmp_path))
    monkeypatch.setenv(native_luna_authority.NATIVE_LUNA_AUTHORITY_ENV, "native_luna")
    (tmp_path / "config.toml").write_text('[search_label_review]\nauthority="native_luna"\n')
    monkeypatch.setattr(supervisor.subprocess, "Popen", lambda *_a, **_k: pytest.fail("cloud spawn"))
    assert not model_execution.automatic_cloud_allowed(tmp_path)
    assert native_luna_authority.native_luna_authority_for_lane(native_luna_authority.NATIVE_LUNA_LANE, root=tmp_path) == "local"
    with pytest.raises(RuntimeError, match="automatic_cloud_models_disabled"):
        supervisor.start_round(SimpleNamespace(root=tmp_path), {}, 0)
    with pytest.raises(native_luna_authority.NativeLunaIdentityError, match="automatic_cloud_models_disabled"):
        native_luna_authority._execute_native_luna_process(
            codex=tmp_path / "absent", prompt="synthetic", repo_root=tmp_path,
            schema={}, timeout=1, root=tmp_path, run_id="synthetic", capacity=None,
        )
    result = frontier_review.run_frontier_review({}, None, repo_root=tmp_path)
    assert result.frontier_failure["failure_class"] == "automatic_cloud_models_disabled"
    assert not result.execution_started and not result.human_required
    with pytest.raises(llm_config.LLMConfigError):
        llm_config.compose_remote_generation_backend(None, None)


def test_cached_runtime_rechecks_owner_policy_before_each_cloud_request(tmp_path, monkeypatch):
    monkeypatch.setenv("CHRONOVISOR_ROOT", str(tmp_path))
    path = tmp_path / "config.toml"
    path.write_text("[model_execution]\nallow_automatic_cloud_models=true\n")
    calls = []

    class Backend:
        provider = "synthetic"
        location = RouteLocation.REMOTE

        def generate(self, request, *, model):
            calls.append(request)
            return GenerationResult("ok", self.provider, model)

    source = SourceDataClassification(SourceDataClass.RAW, SourceSensitivity.HIGH)
    runtime = LLMRuntime(
        generation={"teacher": GenerationRoute(Backend(), "synthetic")},
        remote_egress_opt_ins={("teacher", SourceDataClass.RAW)},
        cloud_execution_allowed=model_execution.automatic_cloud_allowed,
    )
    request = GenerationRequest("synthetic", source=source)
    assert runtime.generate("teacher", request).content == "ok"
    path.write_text("[model_execution]\nallow_automatic_cloud_models=false\n")
    with pytest.raises(EgressDeniedError):
        runtime.generate("teacher", request)
    assert len(calls) == 1


@pytest.mark.parametrize("level", ["off", "low", "medium", "xhigh"])
@pytest.mark.parametrize("model", [None, "qwen3.8-flash-next-chat"])
def test_supported_reasoning_reaches_template_without_budget_reduction(tmp_path, monkeypatch, level, model):
    monkeypatch.setenv("CHRONOVISOR_ROOT", str(tmp_path))
    (tmp_path / "config.toml").write_text(f'''[model_execution.local_reasoning]
model="qwen3.8-flash-next-chat"
roles=["recall.distill.teacher.a", "classification.authority"]
level="low"
num_predict=16384
num_ctx=65536
[model_execution.local_reasoning.level_by_role]
"recall.distill.teacher.a"="{level}"
[llm.roles."recall.distill.teacher.a"]
model="qwen3.8-flash-next-chat"
''')
    session = LocalStructuredSession(model=model, runtime_role="recall.distill.teacher.a", audit_root=tmp_path)
    template = session._reasoning_request_template(
        schema=None, effective_num_ctx=65536, requested_num_ctx=65536,
        required_num_ctx=1000, activity_update=None,
    )
    expected = False if level == "off" else level
    assert template.think == expected
    assert template.ollama_think == expected
    assert session.local_reasoning["level"] == level
    assert model_execution.local_reasoning_settings("classification.authority", "qwen3.8-flash-next-chat", root=tmp_path)["level"] == "low"
    assert template.num_predict == session._output_reservation() == 16384
    payload = OMLXAdapter()._request_payload(MessageGenerationRequest(
        messages=({"role": "user", "content": "synthetic"},),
        source=SourceDataClassification(SourceDataClass.SYSTEM, SourceSensitivity.HIGH),
        think=template.think, max_output_tokens=template.num_predict,
        format=None, num_ctx=65536, keep_alive="0", timeout_ms=1000, max_output_chars=8000,
    ), model="qwen3.8-flash-next-chat")
    if level == "off":
        assert payload["enable_thinking"] is False
        assert payload["chat_template_kwargs"]["enable_thinking"] is False
        assert "reasoning_effort" not in payload
    else:
        assert payload["reasoning_effort"] == payload["chat_template_kwargs"]["reasoning_effort"] == level
    assert payload["max_tokens"] == 16384


@pytest.mark.parametrize("override", ['level_by_role="off"', '[model_execution.local_reasoning.level_by_role]\n"recall.distill.teacher.a"="invalid"'])
def test_invalid_role_reasoning_fails_closed(tmp_path, override):
    (tmp_path / "config.toml").write_text('''[model_execution.local_reasoning]
model="synthetic"
roles=["recall.distill.teacher.a"]
level="low"
num_predict=16384
num_ctx=65536
''' + override)
    with pytest.raises(ValueError, match="local reasoning"):
        model_execution.local_reasoning_settings("recall.distill.teacher.a", "synthetic", root=tmp_path)


def test_local_profile_preserves_prior_labels_and_requires_old_independence_gate(tmp_path):
    snapshots = {"synthetic-rally": {"candidates": [
        {"candidate_id": "previous", "text_sha256": "a" * 64},
        {"candidate_id": "pending", "text_sha256": "b" * 64},
    ]}}
    tasks, work = distill._prepare_local_teacher_work(
        snapshots=snapshots, rally_by_id={"synthetic-rally": {"rally_id": "synthetic-rally"}},
        split_assignments={}, split_plan_id="", profile=distill.LOCAL_SINGLE_PROFILE,
        prior_pairs={("synthetic-rally", "previous")}, reasoning={"level": "xhigh"},
    )
    assert len(tasks) == len(work) == 1
    assert next(iter(tasks.values()))["candidate"]["candidate_id"] == "pending"
    assert next(iter(tasks.values()))["assignment"]["routes"] == [distill.TEACHER_ROLES[0]]
    assert distill._local_teacher_workset_path(tmp_path, distill.LOCAL_SINGLE_PROFILE) != distill._local_teacher_workset_path(tmp_path, distill.LOCAL_TRIAD_PROFILE)
    config = distill.DistillationConfig(teacher_profile=distill.LOCAL_SINGLE_PROFILE)
    gate = distill._offline_training_gate([], config, root=tmp_path)
    assert not gate["passed"]
    assert "teacher_models_not_distinct" in gate["reasons"]


def test_local_single_selector_preserves_the_disabled_legacy_supervisor_contract(tmp_path):
    from chronovisor.ops import health
    from chronovisor.ops.model_execution_status import model_execution_status

    path = tmp_path / "config.toml"
    path.write_text('''[recall.distillation]
enabled=true
capture_enabled=true
teacher_profile="local-triad-v1"
local_single_teacher=true
native_luna_enabled=false
''')
    config = distill.load_distillation_config(path)
    assert config.enabled and not config.native_luna_enabled
    assert config.teacher_profile == distill.LOCAL_SINGLE_PROFILE
    assert model_execution_status(tmp_path)["teacher"]["profile"] == distill.LOCAL_SINGLE_PROFILE
    assert health._distillation_config(tmp_path)["teacher_profile"] == distill.LOCAL_SINGLE_PROFILE
    gate = distill._offline_training_gate([], config, root=tmp_path)
    assert not gate["passed"] and "teacher_models_not_distinct" in gate["reasons"]


@pytest.mark.parametrize("table", [
    {"local_single_teacher": "true"},
    {"local_single_teacher": True, "native_luna_enabled": True},
    {"local_single_teacher": True, "teacher_profile": "native-luna-single-v1"},
])
def test_local_single_selector_rejects_inconsistent_profiles(tmp_path, table):
    from chronovisor.ops import health
    from chronovisor.ops.model_execution_status import model_execution_status

    path = tmp_path / "config.toml"
    path.write_text("[recall.distillation]\n" + "\n".join(
        f"{key}={json.dumps(value)}" for key, value in table.items()
    ))
    with pytest.raises(distill.DistillationError):
        distill.load_distillation_config(path)
    assert model_execution_status(tmp_path)["status"] == "config_unavailable"
    assert health._distillation_config(tmp_path)["config_error"] == "invalid_local_teacher_selector"


def test_local_single_selector_survives_config_migration_without_rewriting_it(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text("")
    distill.migrate_distillation_config(path, apply=True)
    raw = path.read_text().replace("local_single_teacher = false", "local_single_teacher = true")
    path.write_text(raw)
    assert distill.migrate_distillation_config(path, apply=True)["status"] == "noop"
    assert path.read_text() == raw
    assert distill.load_distillation_config(path).teacher_profile == distill.LOCAL_SINGLE_PROFILE


def test_drain_rate_excludes_luna_history_and_slow_progress_is_visible():
    result = observed_drain([
        {"status": "done", "at": "2026-10-18T10:00:00Z"},
        {"status": "done", "at": "2026-10-18T12:30:00Z"},
        {"status": "pending", "at": "2026-10-18T12:30:00Z"},
    ], status_field="status", terminal_statuses={"done"}, timestamp_fields=("at",),
        pending=100, switched_at="2026-10-18T12:00:00Z", now=datetime(2026, 10, 18, 13, tzinfo=UTC))
    assert result["terminal_items"] == result["items_per_hour"] == 1
    assert result["eta_hours_at_observed_rate"] == 100
    assert result["status"] == "draining"


def test_local_profile_index_keeps_native_and_local_labels_separate(tmp_path):
    from chronovisor.recall import recall_distillation_store as store

    path = store.distillation_dir(tmp_path) / "label-ledger.jsonl"
    store.append_chain_batch(path, [
        {"kind": "teacher-label", "rally_id": "synthetic-old", "candidate_id": "old",
         "teacher_profile": distill.NATIVE_LUNA_SINGLE_PROFILE},
        {"kind": "teacher-label", "rally_id": "synthetic-local", "candidate_id": "new",
         "teacher_profile": distill.LOCAL_SINGLE_PROFILE, "profile": distill.LOCAL_SINGLE_PROFILE},
    ])
    before = path.read_bytes()
    assert distill._local_single_prior_pairs(path) == {("synthetic-old", "old")}
    assert [row["candidate_id"] for row in distill._local_single_label_rows(path)] == ["new"]
    assert path.read_bytes() == before


def test_local_profile_cohort_is_stable_before_and_after_first_labels(monkeypatch):
    from chronovisor.core import ollama

    roles = (distill.TEACHER_ROLES[0], "recall.distill.answer_generator", "recall.distill.utility_judge")
    routes = tuple(SimpleNamespace(role=role, model="synthetic-local", location="local") for role in roles)
    digests = {role: "a" * 64 if index < 2 else "b" * 64 for index, role in enumerate(roles)}
    monkeypatch.setattr(ollama, "runtime_generation_routes", lambda _roles: routes)
    monkeypatch.setattr(ollama, "runtime_generation_route_fingerprints", lambda _routes: digests)
    monkeypatch.setattr(model_execution, "local_reasoning_settings", lambda *_args, **_kwargs: {"level": "xhigh", "num_predict": 16384, "num_ctx": 65536})
    row = {"source": "teacher-label", "profile": distill.LOCAL_SINGLE_PROFILE,
           "route": roles[0], "model_digest": digests[roles[0]], "rally_id": "synthetic-new",
           "split_plan_id": "synthetic-fixed", "split": "test"}
    _, before = distill._active_training_cohort([], teacher_profile=distill.LOCAL_SINGLE_PROFILE)
    selected, after = distill._active_training_cohort([row], teacher_profile=distill.LOCAL_SINGLE_PROFILE)
    assert before == after
    assert selected == [row]


def test_local_daily_collects_new_embargo_evidence_without_changing_its_split(tmp_path):
    arguments = {
        "snapshots": {"synthetic-new": {"candidates": [{"candidate_id": "synthetic-candidate", "text_sha256": "a" * 64}]}},
        "rally_by_id": {"synthetic-new": {"rally_id": "synthetic-new"}},
        "split_assignments": {"synthetic-new": "embargo"}, "split_plan_id": "synthetic-fixed-plan",
    }
    assert distill._prepare_local_teacher_work(**arguments, profile=distill.LOCAL_TRIAD_PROFILE) == ({}, [])
    tasks, work = distill._prepare_local_teacher_work(**arguments, profile=distill.LOCAL_SINGLE_PROFILE)
    assert len(tasks) == len(work) == 1
    assert work[0]["temporal_split"]["split"] == "embargo"
    assert work[0]["temporal_split"]["split_plan_id"] == "synthetic-fixed-plan"
    evidence = {"authority": "teacher-only", "verdict": "relevant", "dimension": "relevance",
                "rally_id": "synthetic-new", "candidate_id": "synthetic-candidate", "split": "embargo",
                "features": {key: 0.0 for key in distill.FAST_FEATURE_KEYS}}
    assert distill.train_tiny_policy([evidence]) == distill.train_tiny_policy([])
    gate = distill._offline_training_gate([], distill.DistillationConfig(teacher_profile=distill.LOCAL_SINGLE_PROFILE), root=tmp_path)
    assert not gate["passed"] and "chronological_split_incomplete" in gate["reasons"]


def test_drain_includes_normal_stale_source_terminal_receipts():
    result = observed_drain([
        {"status": "source_stale", "source_stale_metadata": {"checked_at": "2026-10-18T12:30:00Z"}},
    ], status_field="status", terminal_statuses={"source_stale"},
        timestamp_fields=("source_stale_metadata.checked_at",), pending=50,
        switched_at="2026-10-18T12:00:00Z", now=datetime(2026, 10, 18, 13, tzinfo=UTC))
    assert result["terminal_items"] == result["items_per_hour"] == 1


def test_drain_rate_uses_local_clock_for_legacy_naive_queue_timestamps(monkeypatch):
    with monkeypatch.context() as patch:
        patch.setenv("TZ", "Asia/Tokyo")
        time.tzset()
        try:
            result = observed_drain([
                {"status": "done", "at": "2026-10-19T00:30:00"},
            ], status_field="status", terminal_statuses={"done"}, timestamp_fields=("at",),
                pending=50, switched_at="2026-10-18T15:00:00Z", now=datetime(2026, 10, 18, 16, tzinfo=UTC))
        finally:
            patch.undo()
            time.tzset()
    assert result["terminal_items"] == result["items_per_hour"] == 1
    assert result["eta_hours_at_observed_rate"] == 50


def test_daily_cycle_uses_normal_worker_and_seals_only_local_receipt(tmp_path, monkeypatch):
    from chronovisor.core import ollama
    from chronovisor.recall import recall_distillation_store as store
    from chronovisor.recall import recall_local_daily as daily

    monkeypatch.setattr(daily.distill, "load_distillation_config", lambda _path: distill.DistillationConfig(
        enabled=True, teacher_profile=distill.LOCAL_SINGLE_PROFILE,
    ))
    route = ollama.RuntimeGenerationRoute(
        role=distill.TEACHER_ROLES[0], provider="omlx", model="synthetic-local",
        location="local", structured_output=True, revision="synthetic-revision",
    )
    monkeypatch.setattr(daily.ollama, "runtime_generation_routes", lambda _roles: (route,))
    monkeypatch.setattr(daily, "local_reasoning_settings", lambda *_args, **_kwargs: {
        "level": "xhigh", "num_predict": 16384, "num_ctx": 65536,
    })
    calls = []

    def worker(**kwargs):
        calls.append(kwargs)
        return {"status": "capture_only", "labels_written": 4, "p5_allowed": False, "local_workset": {"backlog": 100}}

    monkeypatch.setattr(daily.distill, "run_distillation_chunk", worker)
    result = daily.run_cycle(tmp_path)
    assert calls == [{"root": tmp_path, "config_path": tmp_path / "config.toml", "cold_start": True, "max_elapsed_seconds": 900}]
    assert result["labels_written"] == 4 and not result["promotion_allowed"]
    assert result["drain"]["pending"] == 100
    assert result["drain"]["basis"] == "last_full_local_cycle_including_shared_runtime_wait"
    directory = store.distillation_dir(tmp_path) / "local-daily"
    status = store.read_sealed(directory / "status.json")
    receipt = store.read_sealed(directory / "cycles" / f'{status["receipt_id"]}.json', schema=daily.RECEIPT_SCHEMA)
    assert receipt["teacher"]["reasoning"]["level"] == "xhigh"
    assert receipt["teacher"]["location"] == "local"
    assert receipt["stage_wall_seconds"]["cycle_total"] >= 0
    assert daily.distill.CYCLE_STAGE_SECONDS.get() is None
    lock = store.acquire_nonblocking_lock(directory / "daily-worker.lock")
    assert lock is not None
    try:
        assert daily.run_cycle(tmp_path)["reason"] == "daily_worker_busy"
    finally:
        store.release_lock(lock)
    assert len(calls) == 1
    marker = tmp_path / "runtime/recall-luna-bulk/supervisor-maintenance.json"
    marker.parent.mkdir(parents=True)
    marker.write_text('{}')
    assert daily.run_cycle(tmp_path) == {"status": "deferred", "reason": "maintenance"}
    assert len(calls) == 1


def test_cycle_stage_preserves_exception_and_accumulates_only_in_context(monkeypatch):
    timings = {}
    clock = iter((1.0, 3.5, 4.0, 5.0))
    monkeypatch.setattr(distill.time, "monotonic", lambda: next(clock))
    token = distill.CYCLE_STAGE_SECONDS.set(timings)
    try:
        with pytest.raises(RuntimeError, match="interrupted"):
            with distill._cycle_stage("setup"):
                raise RuntimeError("interrupted")
        with distill._cycle_stage("setup"):
            pass
    finally:
        distill.CYCLE_STAGE_SECONDS.reset(token)
    assert timings == {"setup": 3.5}
