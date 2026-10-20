"""Avoid a second answer only when the unchanged utility gate must reject it."""

import copy
import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from chronovisor.core import canonical_json, ollama, research_scheduler
from chronovisor.recall import recall_distillation as distill


def _worker(monkeypatch: pytest.MonkeyPatch, answers: list[str], limit: int = 12_000):
    roles = ("recall.distill.answer_generator", "recall.distill.utility_judge")
    routes = {
        role: {"role": role, "provider": "ollama", "model": f"synthetic-{index}",
               "location": "local"}
        for index, role in enumerate(roles)
    }
    digests = {roles[0]: "a" * 64, roles[1]: "b" * 64}
    calls: list[dict[str, Any]] = []
    leases: list[str] = []
    remaining = iter(answers)
    monkeypatch.setattr(ollama, "runtime_generation_routes", lambda names: tuple(
        SimpleNamespace(**routes[name], structured_output=True) for name in names
    ))

    @contextmanager
    def lane(*_args: Any, **_kwargs: Any):
        leases.append("enter")
        try:
            yield SimpleNamespace(admission=SimpleNamespace(admitted=True))
        finally:
            leases.append("exit")

    def command(_argv: Any, encoded: str, _lease: Any, **_kwargs: Any):
        request = json.loads(encoded)
        calls.append(request)
        result = (
            {"answer": next(remaining)} if request["operation"] == "answer"
            else {"blind_choice": "tie", "rationale": "synthetic comparison"}
        )
        return SimpleNamespace(status="completed", value={
            **{key: request[key] for key in ("schema", "operation", "role", "request_id")},
            "ok": True, "route_identity": routes[request["role"]],
            "model_digest": digests[request["role"]], "result": result,
        })

    monkeypatch.setattr(research_scheduler, "research_lane", lane)
    monkeypatch.setattr(research_scheduler, "run_cancellable_command", command)
    return distill._WorkerCounterfactual(limit, routes, digests, 300_000), calls, leases


def _payload(query: str) -> dict[str, Any]:
    return {
        "rally_id": "synthetic-rally", "candidate_id": "synthetic-candidate",
        "query": query, "context": [], "a0_evidence": [],
        "a1_evidence": ["synthetic evidence"],
    }


@pytest.mark.parametrize("limit,answer", [
    (12_000, "あ" * 600), (24_000, "z" * 1800), (4096, "\n" * 900),
], ids=["utf8", "absolute_cap", "escaped_small_config"])
def test_first_answer_that_already_exceeds_utility_limit_skips_second_generation(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
    limit: int, answer: str,
) -> None:
    worker, calls, leases = _worker(monkeypatch, [answer, "second answer"], limit)
    payload = _payload("q" * (min(limit, 12_000) - 1500))
    before = copy.deepcopy(payload)
    with pytest.raises(distill.DistillationError, match="input exceeds fixed byte limit"):
        worker.compare(payload)
    assert [call["operation"] for call in calls] == ["answer"]
    assert leases == ["enter", "exit"]
    assert payload == before
    assert caplog.messages == [
        "counterfactual_input_early_reject skipped_answer_generations=1",
    ]
    assert answer not in caplog.text and payload["query"] not in caplog.text


@pytest.mark.parametrize("attempt", [1, 3])
def test_early_input_rejection_quarantines_only_the_item(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, attempt: int,
) -> None:
    worker, calls, leases = _worker(monkeypatch, ["あ" * 600, "second answer"])
    commits: list[Any] = []
    queue = SimpleNamespace(commit=lambda *args: commits.append(args))
    result = distill._compare_and_commit_counterfactual(
        workset=queue, claim=SimpleNamespace(attempt=attempt), counterfactual=worker,
        payload=_payload("q" * 10_500), label_path=tmp_path / "labels",
        root=tmp_path, raw_dir=tmp_path, snapshots={}, exposure={}, exact_features={},
        rally_id="synthetic-rally", candidate={}, mode="add",
        config=distill.DistillationConfig(),
    )
    assert result.failure_class == "counterfactual_input_limit" and not result.deferred
    assert commits[0][1] == [{
        "status": "quarantined",
        "error_class": "counterfactual_input_limit",
        "retry_after_seconds": 0,
    }]
    assert len(calls) == 1 and leases == ["enter", "exit"]
    assert not (tmp_path / "labels").exists()


@pytest.mark.parametrize("overflow", [0, 1])
def test_combined_answer_boundary_keeps_both_generations_and_original_inputs(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, overflow: int,
) -> None:
    answers = ["a" * 100, "b" * 101]
    payload = _payload("")
    base = {key: payload[key] for key in ("rally_id", "candidate_id", "query", "context")}
    utility = {**base, "answer_a": answers[0], "answer_b": answers[1], "blind_order": "a_first"}
    remaining = 12_000 - len(canonical_json.canonical_json_bytes_strict(utility))
    payload["query"] = "q" * (remaining + overflow)
    before = copy.deepcopy(payload)
    worker, calls, leases = _worker(monkeypatch, answers)
    if overflow:
        with pytest.raises(distill.DistillationError, match="input exceeds fixed byte limit"):
            worker.compare(payload)
        assert [call["operation"] for call in calls] == ["answer", "answer"]
    else:
        result = worker.compare(payload)
        assert result["verdict"] == "neutral" and result["order_agreement"] is True
        assert [call["operation"] for call in calls] == ["answer", "answer", "utility", "utility"]
        for call in calls[2:]:
            assert len(canonical_json.canonical_json_bytes_strict(call["input"])) == 12_000
        assert set(calls[2]["input"]) == set(utility)
        assert calls[2]["input"]["answer_a"] == calls[3]["input"]["answer_b"]
        assert calls[2]["input"]["answer_b"] == calls[3]["input"]["answer_a"]
    for index, call in enumerate(calls[:2]):
        assert call["input"] == {
            **{key: before[key] for key in ("rally_id", "candidate_id", "query", "context")},
            "evidence": before[f"a{index}_evidence"],
        }
        assert call["deadline_ms"] == 300_000
    assert leases == ["enter", "exit"] and payload == before
    assert not caplog.messages


def test_oversize_first_input_still_rejects_before_any_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker, calls, leases = _worker(monkeypatch, [])
    with pytest.raises(distill.DistillationError, match="input exceeds fixed byte limit"):
        worker.compare(_payload("q" * 12_000))
    assert not calls and not leases
