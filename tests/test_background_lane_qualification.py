from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from chronovisor.core import ollama
from chronovisor.core.runtime_config import DecisionRouterConfig
from chronovisor.decision import background_lane_qualification as qualification
from chronovisor.decision.decision_lane_contract_cases import (
    background_decision_lane_contract_case_specs,
)
from chronovisor.decision.local_structured import ChatRequest
from chronovisor.lab.local_model_eval import evaluate_replays

LANE = "recall_answer_adjudication"


def _config() -> DecisionRouterConfig:
    return DecisionRouterConfig(
        primary_model="primary:test",
        challenger_model="challenger:test",
        tie_break_model="tie:test",
        primary_keep_alive="1m",
        challenger_keep_alive="1m",
        tie_break_keep_alive="1m",
        num_ctx=131_072,
        num_predict=512,
        read_timeout_ms=5_000,
        max_input_chars=100_000,
        max_output_chars=10_000,
        max_feedback_chars=10_000,
        quorum=2,
    )


def _metadata(models: Sequence[str]) -> dict[str, object]:
    return {
        "engine": {"name": "ollama", "version": "test"},
        "models": {
            model: {
                "name": model,
                "digest": f"digest-{index}",
                "size": 100 + index,
                "details": {
                    "format": "gguf",
                    "parameter_size": "test",
                    "quantization_level": "Q5_K_M",
                },
            }
            for index, model in enumerate(models)
        },
    }


def _authority(config: DecisionRouterConfig) -> dict[str, object]:
    return {
        "policy": {"schema_name": LANE},
        "router": {
            "source": "adopted_artifact",
            "artifact_sha256": "a" * 64,
            "error": None,
            "models": [
                config.primary_model,
                config.challenger_model,
                config.tie_break_model,
            ],
        },
    }


class _ExactContractTransport:
    def __init__(self) -> None:
        self.calls = 0
        self.cases = [
            case
            for case in background_decision_lane_contract_case_specs()
            if case.lane == LANE
        ]

    def __call__(self, request: ChatRequest) -> ollama.ChatResponse:
        self.calls += 1
        matches = [case for case in self.cases if case.prompt in request.messages[-1]["content"]]
        assert len(matches) == 1
        return ollama.ChatResponse(
            content=json.dumps(matches[0].expected),
            prompt_eval_count=100,
            eval_count=50,
        )


def test_background_qualification_bootstraps_once_and_validator_never_infers(
    tmp_path: Path, monkeypatch
) -> None:
    from chronovisor.decision import decision_authority

    config = _config()
    authority = _authority(config)
    transport = _ExactContractTransport()
    monkeypatch.setattr(
        decision_authority,
        "base_semantic_authority",
        lambda lane: (authority, None) if lane == LANE else (None, "wrong_lane"),
    )
    monkeypatch.setattr(
        qualification, "_resolved_adopted_config", lambda _authority: config
    )

    def evaluator(input_path: Path, output_path: Path, **kwargs: object) -> dict:
        options = dict(kwargs)
        options.pop("live_resource_control", None)
        return evaluate_replays(
            input_path,
            output_path,
            transport=transport,
            live_resource_control=False,
            **options,
        )

    result = qualification.qualify_background_lane(
        LANE,
        chronovisor_root=tmp_path,
        metadata_provider=_metadata,
        evaluator=evaluator,
    )
    calls_after_qualification = transport.calls
    check = qualification.validate_current_background_lane_qualification(
        LANE,
        base_authority=authority,
        chronovisor_root=tmp_path,
        metadata_provider=_metadata,
        evaluator=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("read-only validation must not run an evaluator")
        ),
    )

    assert result["status"] == "passed", result
    assert calls_after_qualification == 10
    assert check["passed"] is True
    assert transport.calls == calls_after_qualification

    evaluation_path = next(
        (tmp_path / "runtime" / "decision-qualification").glob("*.evaluation.json")
    )
    artifact = json.loads(evaluation_path.read_text(encoding="utf-8"))
    assert isinstance(artifact, Mapping)
    artifact["status"] = "complete-forged"
    evaluation_path.write_text(json.dumps(artifact), encoding="utf-8")
    held = qualification.validate_current_background_lane_qualification(
        LANE,
        base_authority=authority,
        chronovisor_root=tmp_path,
        metadata_provider=_metadata,
        evaluator=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("tamper validation must not infer")
        ),
    )
    assert held == {
        "passed": False,
        "reason": "decision_lane_qualification_missing",
    }
