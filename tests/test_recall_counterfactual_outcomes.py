from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from chronovisor.core import ollama, research_scheduler
from chronovisor.ops import health
from chronovisor.recall import recall_distillation as distill
from chronovisor.recall import recall_distillation_store as store
from chronovisor.recall import recall_distillation_workset as workset


def _route(role: str, model: str) -> dict[str, str]:
    return {
        "role": role,
        "provider": "ollama",
        "model": model,
        "location": "local",
    }


def _valid_response(*, verdict: str = "helpful", order_agreement: bool = True) -> dict[str, Any]:
    return {
        "verdict": verdict,
        "reason": "",
        "a0_sha256": "0" * 64,
        "a1_sha256": "1" * 64,
        "blind_orders": ["a0_first", "a1_first"],
        "order_agreement": order_agreement,
        "generator_route_identity": _route(
            "recall.distill.answer_generator", "answer-model"
        ),
        "generator_model_digest": "a" * 64,
        "judge_route_identity": _route("recall.distill.utility_judge", "judge-model"),
        "judge_model_digest": "b" * 64,
    }


def _payload() -> dict[str, Any]:
    return {
        "rally_id": "rally",
        "candidate_id": "candidate",
        "query": "query",
        "context": [],
        "a0_evidence": [],
        "a1_evidence": ["candidate"],
        "mode": "add",
    }


def _work_item() -> dict[str, Any]:
    return {
        "work_id": "local-counterfactual-test",
        "kind": "local-counterfactual",
        "payload_ref": "candidate-snapshot:rally:candidate",
        "payload_digest": "c" * 64,
        "temporal_split": {"split": "test"},
        "provenance": {"profile": "local-triad-v1"},
        "priority": 0,
    }


def _claim(root: Path, *, attempt: int = 1) -> tuple[workset.DistillationWorkset, Any]:
    queue = workset.DistillationWorkset(
        store.distillation_dir(root) / "local-workset.sqlite3"
    )
    label_path = store.distillation_dir(root) / store.LABEL_LEDGER_FILE
    watermark = distill._local_workset_watermark(
        root=root,
        raw_dir=root / "raw",
        label_path=label_path,
        snapshots={},
        progress_kind="local-counterfactual-v1",
    )
    queue.advance(
        [_work_item()],
        watermark,
        progress=distill._local_workset_progress(watermark),
    )
    for _ in range(attempt - 1):
        prior = queue.claim("local-counterfactual", 1, "test-owner", 300)[0]
        queue.commit(
            [prior],
            [{"status": "retry", "error_class": "test_retry", "retry_after_seconds": 0}],
        )
    return queue, queue.claim("local-counterfactual", 1, "test-owner", 300)[0]


def _compare(
    root: Path,
    queue: workset.DistillationWorkset,
    claim: Any,
    counterfactual: Any,
) -> distill._CounterfactualBlockResult:
    return distill._compare_and_commit_counterfactual(
        workset=queue,
        claim=claim,
        counterfactual=counterfactual,
        payload=_payload(),
        label_path=store.distillation_dir(root) / store.LABEL_LEDGER_FILE,
        root=root,
        raw_dir=root / "raw",
        snapshots={},
        exposure={},
        exact_features={},
        rally_id="rally",
        candidate={"candidate_id": "candidate"},
        mode="add",
        config=distill.DistillationConfig(enabled=True),
    )


def _configure_health(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.toml").write_text(
        "[recall.distillation]\n"
        "enabled = true\n"
        "capture_enabled = true\n"
        "teacher_profile = \"local-triad-v1\"\n",
        encoding="utf-8",
    )


def _write_health_state(root: Path, fields: dict[str, Any]) -> None:
    _configure_health(root)
    store.write_sealed_state(
        store.distillation_dir(root) / store.STATE_FILE,
        {
            "kind": "worker-state",
            "status": "ready",
            "worker_status": "idle",
            "rollout_percent": 0,
            "teacher_profile": "local-triad-v1",
            "last_label_at": datetime.now(UTC).isoformat(),
            "hold_reason": "",
            **fields,
        },
    )


def _repeated_failure_fields(failure_class: str) -> dict[str, Any]:
    first = distill._counterfactual_failure_fields({}, failure_class, 0)
    return distill._counterfactual_failure_fields(first, failure_class, 0)


def _native_uncertain_worker(monkeypatch: pytest.MonkeyPatch) -> distill._WorkerCounterfactual:
    routes = {
        "recall.distill.answer_generator": _route(
            "recall.distill.answer_generator", "answer-model"
        ),
        "recall.distill.utility_judge": _route(
            "recall.distill.utility_judge", "judge-model"
        ),
    }
    digests = {
        "recall.distill.answer_generator": "a" * 64,
        "recall.distill.utility_judge": "b" * 64,
    }
    answers = iter(("without", "with"))

    def fake_worker_call(
        operation: str, role: str, _payload: object, **_kwargs: object
    ) -> dict[str, Any]:
        if operation == "answer":
            return {
                "answer": next(answers),
                "_model_digest": digests[role],
                "_route_identity": routes[role],
            }
        return {
            "blind_choice": "a",
            "rationale": "",
            "_model_digest": digests[role],
            "_route_identity": routes[role],
        }

    monkeypatch.setattr(distill, "_worker_call", fake_worker_call)
    return distill._WorkerCounterfactual(12_000, routes, digests)


def test_repeated_data_reject_quarantines_without_label_and_health_is_not_human_wait(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "data-reject"
    queue, claim = _claim(root, attempt=3)
    native = _native_uncertain_worker(monkeypatch)
    response = native.compare(_payload())
    assert response["verdict"] == "uncertain"
    assert response["order_agreement"] is False

    result = _compare(root, queue, claim, SimpleNamespace(local=True, compare=lambda _payload: response))

    assert result.failure_class == "counterfactual_data_reject"
    assert queue.status()["quarantined"] == 1
    assert queue.status()["completed"] == 0
    assert not (store.distillation_dir(root) / store.LABEL_LEDGER_FILE).exists()

    _write_health_state(root, _repeated_failure_fields("counterfactual_data_reject"))
    diagnostic = health.recall_distillation_health(root)

    assert diagnostic["distillation_mainline_failure"]["worker_failure_count"] == 2
    assert (
        diagnostic["distillation_mainline_failure"]["worker_error_kind"]
        == "counterfactual_data_reject"
    )
    assert "distillation_run_repeated_failure" not in diagnostic["reason_codes"]
    assert all(item["category"] != "human_wait" for item in diagnostic["alerts"])


def test_valid_counterfactual_metadata_still_writes_label_and_completes_claim(
    tmp_path: Path,
) -> None:
    root = tmp_path / "valid"
    queue, claim = _claim(root)
    response = _valid_response(order_agreement=True)
    counterfactual = SimpleNamespace(local=True, compare=lambda _payload: response)

    result = _compare(root, queue, claim, counterfactual)

    assert result.failure_class == ""
    assert result.written == 1
    assert queue.status()["completed"] == 1
    rows = store.read_chain(store.distillation_dir(root) / store.LABEL_LEDGER_FILE)
    assert len(rows) == 1
    assert rows[0]["order_agreement"] is True
    assert rows[0]["verdict"] == "helpful"


def test_order_disagreement_with_valid_metadata_is_data_reject(
    tmp_path: Path,
) -> None:
    root = tmp_path / "order-disagreement"
    queue, claim = _claim(root, attempt=3)
    response = _valid_response(verdict="helpful", order_agreement=False)
    counterfactual = SimpleNamespace(local=True, compare=lambda _payload: response)

    result = _compare(root, queue, claim, counterfactual)

    assert result.failure_class == "counterfactual_data_reject"
    assert queue.status()["quarantined"] == 1
    assert not (store.distillation_dir(root) / store.LABEL_LEDGER_FILE).exists()


@pytest.mark.parametrize(
    "failure_class",
    [
        "counterfactual_invalid",
        "counterfactual_timeout",
        "counterfactual_transport",
        "counterfactual_binding",
    ],
)
def test_counterfactual_system_failure_kinds_remain_human_wait(
    tmp_path: Path, failure_class: str
) -> None:
    root = tmp_path / failure_class
    fields = _repeated_failure_fields(failure_class)
    _write_health_state(root, fields)

    diagnostic = health.recall_distillation_health(root)

    assert "distillation_run_repeated_failure" in diagnostic["reason_codes"]
    assert any(item["category"] == "human_wait" for item in diagnostic["alerts"])


def test_counterfactual_timeout_oserror_path_remains_system_alert(
    tmp_path: Path,
) -> None:
    root = tmp_path / "timeout"
    queue, claim = _claim(root, attempt=3)

    class TimeoutCounterfactual:
        local = True

        def compare(self, _payload: object) -> dict[str, Any]:
            raise TimeoutError("synthetic timeout")

    result = _compare(root, queue, claim, TimeoutCounterfactual())

    assert result.failure_class in {"counterfactual_transport", "counterfactual_timeout"}
    assert queue.status()["quarantined"] == 1
    _write_health_state(root, _repeated_failure_fields(result.failure_class))
    diagnostic = health.recall_distillation_health(root)
    assert "distillation_run_repeated_failure" in diagnostic["reason_codes"]


def test_mainline_crash_without_counterfactual_kind_remains_human_wait(
    tmp_path: Path,
) -> None:
    root = tmp_path / "crash"
    _write_health_state(
        root,
        {
            "error_code": "distillation_run_failed",
            "worker_error_stage": "mainline",
            "worker_error_type": "RuntimeError",
            "worker_error_kind": "",
            "worker_error_sha256": "d" * 64,
            "worker_failure_count": 2,
            "last_failure_at": datetime.now(UTC).isoformat(),
        },
    )

    diagnostic = health.recall_distillation_health(root)

    assert "distillation_run_repeated_failure" in diagnostic["reason_codes"]
    assert diagnostic["distillation_mainline_failure"]["worker_error_type"] == "RuntimeError"


def test_missing_exposure_replaces_stale_data_kind_with_binding_system_error(
    tmp_path: Path,
) -> None:
    root = tmp_path / "binding"
    rally = {
        "rally_id": "rally",
        "session_cluster_id": "session",
        "as_of": "2026-10-10T00:00:00Z",
        "actual_answer_refs": [{"semantic_sha256": "a" * 64}],
    }
    snapshot = {
        "snapshot_sha256": "b" * 64,
        "candidates": [{"candidate_id": "candidate", "text_sha256": "c" * 64}],
    }
    items, keys = distill._prepare_counterfactual_work(
        root=root, snapshots={"rally": snapshot}, rally_by_id={"rally": rally}
    )
    queue = workset.DistillationWorkset(
        store.distillation_dir(root) / "local-workset.sqlite3"
    )
    queue.advance(items, {"source": "test"})
    claim = queue.claim("local-counterfactual", 1, "test-owner", 300)[0]
    target = next(pair for pair, work_id in keys.items() if work_id == claim.work_id)

    result = distill._run_counterfactual_claim(
        workset=queue,
        claim=claim,
        target=target,
        root=root,
        raw_dir=root / "raw",
        config=distill.DistillationConfig(enabled=True),
        counterfactual=SimpleNamespace(local=True),
        snapshots={"rally": snapshot},
        rally_by_id={"rally": rally},
        texts={},
        label_path=store.distillation_dir(root) / store.LABEL_LEDGER_FILE,
    )

    assert result.failure_class == "counterfactual_binding"
    previous = _repeated_failure_fields("counterfactual_data_reject")
    first_binding = distill._counterfactual_failure_fields(
        previous, result.failure_class, result.written
    )
    fields = distill._counterfactual_failure_fields(
        first_binding, result.failure_class, result.written
    )
    assert fields["worker_error_kind"] == "counterfactual_system_error"
    assert fields["worker_error_type"] == "CounterfactualBindingError"
    _write_health_state(root, fields)
    diagnostic = health.recall_distillation_health(root)
    assert "distillation_run_repeated_failure" in diagnostic["reason_codes"]


def _patch_worker_envelope(
    monkeypatch: pytest.MonkeyPatch,
    *,
    failure_class: str,
    returned_route: dict[str, str],
    status: str = "completed",
) -> tuple[str, dict[str, str]]:
    role = "recall.distill.utility_judge"
    route = SimpleNamespace(
        role=role,
        provider="ollama",
        model="judge-model",
        location="local",
        structured_output=True,
    )
    expected_route = _route(role, "judge-model")
    monkeypatch.setattr(ollama, "runtime_generation_routes", lambda _roles: (route,))

    @contextmanager
    def lane(*_args: object, **_kwargs: object):
        yield object()

    monkeypatch.setattr(research_scheduler, "research_lane", lane)

    def command(
        _argv: object, encoded: str, _lease: object, **_kwargs: object
    ) -> SimpleNamespace:
        request = json.loads(encoded)
        value = {
            "schema": "chronovisor.recall-distillation-worker.v1",
            "ok": False,
            "operation": request["operation"],
            "role": request["role"],
            "request_id": request["request_id"],
            "route_identity": returned_route,
            "model_digest": "d" * 64,
            "result": {},
            "failure_class": failure_class,
        }
        return SimpleNamespace(status=status, value=value)

    monkeypatch.setattr(research_scheduler, "run_cancellable_command", command)
    return role, expected_route


def test_worker_envelope_data_classification_requires_exact_route_and_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_route = _route("recall.distill.utility_judge", "judge-model")
    role, expected_route = _patch_worker_envelope(
        monkeypatch,
        failure_class="feedback_too_large",
        returned_route=expected_route,
    )
    with pytest.raises(distill._CounterfactualDataRejected):
        distill._worker_call(
            "utility",
            role,
            {"candidate_id": "candidate"},
            max_input_bytes=12_000,
            expected_route=expected_route,
            expected_digest="d" * 64,
        )

    wrong_route = _route(role, "wrong-model")
    _patch_worker_envelope(
        monkeypatch,
        failure_class="feedback_too_large",
        returned_route=wrong_route,
    )
    with pytest.raises(distill.DistillationError) as invalid:
        distill._worker_call(
            "utility",
            role,
            {"candidate_id": "candidate"},
            max_input_bytes=12_000,
            expected_route=expected_route,
            expected_digest="d" * 64,
        )
    assert not isinstance(invalid.value, distill._CounterfactualDataRejected)
    _write_health_state(
        tmp_path / "wrong-route", _repeated_failure_fields("counterfactual_invalid")
    )
    diagnostic = health.recall_distillation_health(tmp_path / "wrong-route")
    assert "distillation_run_repeated_failure" in diagnostic["reason_codes"]


def test_worker_envelope_timeout_and_crash_are_system_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    route = _route("recall.distill.utility_judge", "judge-model")
    role, expected_route = _patch_worker_envelope(
        monkeypatch,
        failure_class="transport_timeout",
        returned_route=route,
    )
    with pytest.raises(distill.DistillationDeferred) as timeout:
        distill._worker_call(
            "utility",
            role,
            {"candidate_id": "candidate"},
            max_input_bytes=12_000,
            expected_route=expected_route,
            expected_digest="d" * 64,
        )
    assert timeout.value.failure_class == "transport_timeout"

    _patch_worker_envelope(
        monkeypatch,
        failure_class="backend_error",
        returned_route=route,
        status="failed",
    )
    with pytest.raises(distill.DistillationError):
        distill._worker_call(
            "utility",
            role,
            {"candidate_id": "candidate"},
            max_input_bytes=12_000,
            expected_route=expected_route,
            expected_digest="d" * 64,
        )
