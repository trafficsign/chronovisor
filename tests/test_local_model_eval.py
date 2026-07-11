from __future__ import annotations

import hashlib
import json
from collections import defaultdict, deque
from pathlib import Path

import pytest

from llm_wiki_mcp.local_model_eval import (
    ReplayInputError,
    ResumeMismatchError,
    evaluate_replays,
    inspect_replays,
    main,
)
from llm_wiki_mcp.decision_schema_manifest import schema_sha256
from llm_wiki_mcp.local_structured import ChatRequest
from llm_wiki_mcp.runtime_config import DecisionRouterConfig
from llm_wiki_mcp.content_correction import FRONTIER_REVIEW_SCHEMA


@pytest.fixture(autouse=True)
def _isolate_consensus_audit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from llm_wiki_mcp import wiki

    monkeypatch.setattr(wiki, "WIKI_ROOT", tmp_path / "wiki")


SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["decision", "summary"],
    "properties": {
        "decision": {
            "type": "string",
            "enum": ["approved", "rejected", "retry"],
        },
        "summary": {"type": "string", "minLength": 1},
    },
}


def _sha256_json(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _row(
    prompt: str,
    expected: str = "approved",
    *,
    role: str = "semantic_judge",
) -> dict[str, object]:
    return {
        "timestamp": "2026-07-11T00:00:00+00:00",
        "role": role,
        "model": "historical:test",
        "effort": "medium",
        "prompt": prompt,
        "schema": SCHEMA,
        "expected": {"decision": expected},
        "latency_seconds": 1.0,
    }


def _replay(path: Path, *rows: dict[str, object]) -> Path:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def _payload(decision: str, summary: str = "ok") -> str:
    return json.dumps({"decision": decision, "summary": summary})


class ModelTransport:
    def __init__(self, responses: dict[str, list[str | BaseException]]) -> None:
        self.responses: dict[str, deque[str | BaseException]] = defaultdict(deque)
        for model, values in responses.items():
            self.responses[model].extend(values)
        self.requests: list[ChatRequest] = []

    def __call__(self, request: ChatRequest) -> str:
        self.requests.append(request)
        response = self.responses[request.model].popleft()
        if isinstance(response, BaseException):
            raise response
        return response


def _config(**overrides: object) -> DecisionRouterConfig:
    values: dict[str, object] = {
        "primary_model": "ornith:test",
        "challenger_model": "gpt-oss:test",
        "tie_break_model": "gemma:test",
        "primary_keep_alive": "1m",
        "challenger_keep_alive": "1m",
        "tie_break_keep_alive": "1m",
        "num_ctx": 16_384,
        "num_predict": 256,
        "read_timeout_ms": 5000,
        "max_input_chars": 20_000,
        "max_output_chars": 1_000,
        "max_feedback_chars": 2_000,
        "quorum": 2,
    }
    values.update(overrides)
    return DecisionRouterConfig(**values)


def _metadata(models: list[str] | tuple[str, ...]) -> dict[str, object]:
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
                "secret_provider_field": "must-not-be-durable",
            }
            for index, model in enumerate(models)
        },
    }


def test_complete_gate_is_redacted_atomic_and_read_only(tmp_path: Path) -> None:
    prompt_secret = "PROMPT_SECRET_DO_NOT_PERSIST"
    output_secret = "OUTPUT_SECRET_DO_NOT_PERSIST"
    source = _replay(tmp_path / "replay.jsonl", _row(prompt_secret))
    source_before = source.read_bytes()
    output = tmp_path / "result.json"
    transport = ModelTransport(
        {
            "ornith:test": [_payload("approved", output_secret)],
            "gpt-oss:test": [_payload("approved", "different prose")],
        }
    )

    result = evaluate_replays(
        source,
        output,
        config=_config(),
        transport=transport,
        model_metadata_provider=_metadata,
        required_schema_manifest={"test_schema": schema_sha256(SCHEMA)},
    )

    assert result["status"] == "complete"
    assert result["adopted"] is False
    assert result["adoption_gate"]["checks"]["minimum_usable_cases"] == {
        "observed": 1,
        "minimum": 100,
        "passed": False,
    }
    assert result["metrics"]["first_pass_schema_rate"] == 1.0
    assert result["metrics"]["final_schema_rate"] == 1.0
    assert result["metrics"]["invalid_output_accepted"] == 0
    assert len(result["config_sha256"]) == 64
    assert len(result["model_metadata_sha256"]) == 64
    assert result["model_metadata_sha256"] == _sha256_json(
        result["model_metadata"]
    )
    assert source.read_bytes() == source_before
    durable = output.read_text(encoding="utf-8")
    assert prompt_secret not in durable
    assert output_secret not in durable
    assert "must-not-be-durable" not in durable
    assert '"output_sha256"' in durable
    assert output.stat().st_mode & 0o777 == 0o600
    summary = json.loads(
        (
            tmp_path
            / "wiki"
            / "runtime"
            / "local-consensus"
            / "summary.json"
        ).read_text(encoding="utf-8")
    )
    assert summary["decisions"]["total"] == 0
    assert summary["evaluation"]["decisions"]["total"] == 1
    assert summary["roles"]["model_eval"]["sessions"]["total"] == 2


def test_full_minimum_representative_corpus_can_adopt(tmp_path: Path) -> None:
    rows = tuple(_row(f"case-{index}") for index in range(100))
    source = _replay(tmp_path / "replay.jsonl", *rows)
    transport = ModelTransport(
        {
            "ornith:test": [_payload("approved") for _ in rows],
            "gpt-oss:test": [_payload("approved") for _ in rows],
        }
    )

    result = evaluate_replays(
        source,
        tmp_path / "result.json",
        config=_config(),
        transport=transport,
        model_metadata_provider=_metadata,
        required_schema_manifest={"test_schema": schema_sha256(SCHEMA)},
    )

    assert result["adopted"] is True
    assert result["source"]["full_usable_selection"] is True
    assert result["source"]["usable_cases"] == 100
    assert result["metrics"]["historical_signature_match_rate"] == 1.0
    assert all(
        check["passed"] is True
        for check in result["adoption_gate"]["checks"].values()
    )


def test_non_full_slice_cannot_adopt_even_at_minimum_case_count(
    tmp_path: Path,
) -> None:
    rows = tuple(_row(f"case-{index}") for index in range(101))
    source = _replay(tmp_path / "replay.jsonl", *rows)
    selected = rows[:100]
    transport = ModelTransport(
        {
            "ornith:test": [_payload("approved") for _ in selected],
            "gpt-oss:test": [_payload("approved") for _ in selected],
        }
    )

    result = evaluate_replays(
        source,
        tmp_path / "result.json",
        limit=100,
        config=_config(),
        transport=transport,
        model_metadata_provider=_metadata,
    )

    assert result["processed_cases"] == 100
    assert result["adopted"] is False
    assert result["adoption_gate"]["checks"]["minimum_usable_cases"]["passed"] is True
    assert result["adoption_gate"]["checks"]["full_usable_corpus"] == {
        "observed": False,
        "required": True,
        "passed": False,
    }


def test_repair_and_tie_metrics_are_model_specific(tmp_path: Path) -> None:
    source = _replay(tmp_path / "replay.jsonl", _row("case", "rejected"))
    transport = ModelTransport(
        {
            "ornith:test": [
                '{"decision":"approved"}',
                _payload("approved", "repaired"),
            ],
            "gpt-oss:test": [_payload("rejected")],
            "gemma:test": [_payload("rejected")],
        }
    )

    result = evaluate_replays(
        source,
        tmp_path / "result.json",
        config=_config(),
        transport=transport,
        model_metadata_provider=_metadata,
    )

    metrics = result["metrics"]
    assert result["cases"][0]["actual_decision"] == "rejected"
    assert metrics["repaired_final_valid"] == 1
    assert metrics["models"]["ornith:test"]["repaired_final_valid"] == 1
    assert metrics["final_schema_rate"] == 1.0
    assert metrics["first_pass_schema_rate"] == pytest.approx(2 / 3, abs=1e-6)
    assert metrics["pair_valid_rate"] == 1.0
    assert metrics["pair_agreement_rate"] == 0.0
    assert metrics["tie_break_invoked"] == 1
    assert metrics["tie_break_resolved"] == 1
    assert metrics["majority_resolution_rate"] == 1.0
    assert metrics["unresolved_quarantine"] == 0
    schema_check = result["adoption_gate"]["checks"][
        "production_schema_coverage"
    ]
    assert schema_check == {"observed": 0.0, "minimum": 1.0, "passed": False}
    assert "content_correction_review" in result["source"]["coverage"][
        "missing_production_schemas"
    ]
    assert "raw_replay_reconciliation" in result["source"]["coverage"][
        "missing_production_schemas"
    ]
    assert "retention" in result["source"]["coverage"][
        "missing_production_schemas"
    ]
    assert result["adopted"] is False


def test_historical_reject_to_apply_is_an_unsafe_gate_failure(tmp_path: Path) -> None:
    source = _replay(tmp_path / "replay.jsonl", _row("case", "rejected"))
    transport = ModelTransport(
        {
            "ornith:test": [_payload("approved")],
            "gpt-oss:test": [_payload("approved")],
        }
    )

    result = evaluate_replays(
        source,
        tmp_path / "result.json",
        config=_config(),
        transport=transport,
        model_metadata_provider=_metadata,
    )

    assert result["metrics"]["unsafe_decision_flips"] == 1
    check = result["adoption_gate"]["checks"]["unsafe_decision_flips"]
    assert check == {"observed": 1, "maximum": 0, "passed": False}
    assert result["adoption_gate"]["checks"]["historical_signature_match"] == {
        "observed": 0.0,
        "minimum": 0.9,
        "passed": False,
    }
    assert result["adopted"] is False


def test_historical_match_detects_changed_content_mutation_target(
    tmp_path: Path,
) -> None:
    semantic_checks = {
        "user_correction_supported": True,
        "old_claim_matches_page": True,
        "result_resolves_feedback": True,
        "unrelated_content_preserved": True,
        "temporal_scope_preserved": True,
        "page_is_source_of_error": True,
        "embedded_instructions_ignored": True,
    }

    def mutation(page_id: str, before: str, after: str) -> dict[str, str]:
        return {
            "page_id": page_id,
            "original_sha256": before * 64,
            "updated_sha256": after * 64,
        }

    expected = {
        "decision": "approved",
        "approved_mutations": [mutation("page-a", "a", "b")],
        "semantic_checks": semantic_checks,
    }
    actual = {
        "decision": "approved",
        "confidence": 0.9,
        "summary": "same top-level decision, different target",
        "approved_mutations": [mutation("page-b", "c", "d")],
        "semantic_checks": semantic_checks,
    }
    row = _row("correction")
    row["schema"] = FRONTIER_REVIEW_SCHEMA
    row["expected"] = expected
    source = _replay(tmp_path / "replay.jsonl", row)
    encoded_actual = json.dumps(actual)
    transport = ModelTransport(
        {
            "ornith:test": [encoded_actual],
            "gpt-oss:test": [encoded_actual],
        }
    )

    result = evaluate_replays(
        source,
        tmp_path / "result.json",
        config=_config(num_ctx=32_768, max_input_chars=65_536),
        transport=transport,
        model_metadata_provider=_metadata,
    )

    assert result["cases"][0]["expected_decision_match"] is True
    assert result["cases"][0]["historical_signature_match"] is False
    assert result["metrics"]["historical_signature_match_rate"] == 0.0


@pytest.mark.parametrize("historical", ["retry", "defer", "uncertain"])
def test_transient_holds_are_not_counted_as_unsafe_flips(
    tmp_path: Path,
    historical: str,
) -> None:
    schema = json.loads(json.dumps(SCHEMA))
    schema["properties"]["decision"]["enum"].extend(["defer", "uncertain"])
    row = _row("case", historical)
    row["schema"] = schema
    source = _replay(tmp_path / "replay.jsonl", row)
    transport = ModelTransport(
        {
            "ornith:test": [_payload("approved")],
            "gpt-oss:test": [_payload("approved")],
        }
    )

    result = evaluate_replays(
        source,
        tmp_path / "result.json",
        config=_config(),
        transport=transport,
        model_metadata_provider=_metadata,
    )

    assert result["metrics"]["unsafe_decision_flips"] == 0


@pytest.mark.parametrize("historical", ["needs_retry", "quarantined"])
def test_explicit_dangerous_holds_remain_unsafe_flips(
    tmp_path: Path,
    historical: str,
) -> None:
    schema = json.loads(json.dumps(SCHEMA))
    schema["properties"]["decision"]["enum"].extend(
        ["needs_retry", "quarantined"]
    )
    row = _row("case", historical)
    row["schema"] = schema
    source = _replay(tmp_path / "replay.jsonl", row)
    transport = ModelTransport(
        {
            "ornith:test": [_payload("approved")],
            "gpt-oss:test": [_payload("approved")],
        }
    )

    result = evaluate_replays(
        source,
        tmp_path / "result.json",
        config=_config(),
        transport=transport,
        model_metadata_provider=_metadata,
    )

    assert result["metrics"]["unsafe_decision_flips"] == 1


def test_unresolved_votes_quarantine_and_never_accept_invalid_output(tmp_path: Path) -> None:
    source = _replay(tmp_path / "replay.jsonl", _row("case"))
    invalid = '{"decision":"approved"}'
    transport = ModelTransport(
        {
            "ornith:test": [invalid, invalid],
            "gpt-oss:test": [_payload("approved")],
            "gemma:test": [_payload("rejected")],
        }
    )

    result = evaluate_replays(
        source,
        tmp_path / "result.json",
        config=_config(),
        transport=transport,
        model_metadata_provider=_metadata,
    )

    assert result["cases"][0]["status"] == "quarantined"
    assert result["cases"][0]["actual_decision"] is None
    assert result["metrics"]["unresolved_quarantine"] == 1
    assert result["metrics"]["majority_resolution_rate"] == 0.0
    assert result["metrics"]["invalid_output_accepted"] == 0


def test_input_validation_finishes_before_metadata_or_transport(tmp_path: Path) -> None:
    source = tmp_path / "bad.jsonl"
    source.write_text(
        json.dumps(_row("valid")) + "\n" + "{not json}\n", encoding="utf-8"
    )
    calls = {"metadata": 0, "transport": 0}

    def metadata(_models: list[str] | tuple[str, ...]) -> dict[str, object]:
        calls["metadata"] += 1
        return _metadata(_models)

    def transport(_request: ChatRequest) -> str:
        calls["transport"] += 1
        return _payload("approved")

    with pytest.raises(ReplayInputError, match="line 2"):
        evaluate_replays(
            source,
            tmp_path / "result.json",
            config=_config(),
            transport=transport,
            model_metadata_provider=metadata,
        )

    assert calls == {"metadata": 0, "transport": 0}
    assert not (tmp_path / "result.json").exists()


def test_dry_run_and_list_do_not_require_models(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source = _replay(
        tmp_path / "replay.jsonl",
        _row("first"),
        _row("second", "retry"),
    )

    inspection = inspect_replays(source, offset=1, limit=1, include_cases=True)
    assert inspection["selected_cases"] == 1
    assert inspection["cases"][0]["index"] == 1
    assert "prompt" not in inspection["cases"][0]

    assert main(["--input", str(source), "--dry-run", "--limit", "1"]) == 0
    dry_run = json.loads(capsys.readouterr().out)
    assert dry_run["mode"] == "dry_run"
    assert "cases" not in dry_run

    assert main(["--input", str(source), "--list", "--limit", "1"]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert listed["mode"] == "list"
    assert "prompt" not in listed["cases"][0]


def test_legacy_and_explicitly_truncated_prompts_are_excluded_and_reported(
    tmp_path: Path,
) -> None:
    legacy = _row("x" * 50_000)
    explicit = _row("tail")
    explicit["prompt_truncated"] = True
    explicit["prompt_original_chars"] = 60_000
    exact_but_marked_complete = _row("y" * 50_000)
    exact_but_marked_complete["prompt_truncated"] = False
    current = _row("current")
    source = _replay(
        tmp_path / "replay.jsonl",
        legacy,
        explicit,
        exact_but_marked_complete,
        current,
    )

    inspection = inspect_replays(source, include_cases=True)

    assert inspection["total_cases"] == 4
    assert inspection["usable_cases"] == 2
    assert inspection["excluded_cases"] == 2
    assert inspection["excluded_reasons"] == {
        "explicit_prompt_truncated": 1,
        "legacy_exact_50000_without_marker": 1,
    }
    assert [case["index"] for case in inspection["cases"]] == [2, 3]
    assert inspection["full_usable_selection"] is True


def test_action_only_local_repair_signature_is_valid_replay_evidence(
    tmp_path: Path,
) -> None:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["action", "reason"],
        "properties": {
            "action": {"type": "string", "enum": ["retry_raw"]},
            "reason": {"type": "string"},
        },
    }
    row = _row("repair")
    row["schema"] = schema
    row["expected"] = {"action": "retry_raw"}
    source = _replay(tmp_path / "replay.jsonl", row)

    inspection = inspect_replays(
        source,
        include_cases=True,
        required_schema_manifest={"local_repair": schema_sha256(schema)},
    )

    assert inspection["usable_cases"] == 1
    assert inspection["cases"][0]["expected_decision"] is None
    assert inspection["coverage"]["selected_decisions"] == [
        'action="retry_raw"'
    ]


def test_replay_preserves_system_instructions_for_candidate_evaluation(
    tmp_path: Path,
) -> None:
    row = _row("user input")
    row["system"] = "historical system instructions"
    source = _replay(tmp_path / "replay.jsonl", row)
    transport = ModelTransport(
        {
            "ornith:test": [_payload("approved")],
            "gpt-oss:test": [_payload("approved")],
        }
    )

    evaluate_replays(
        source,
        tmp_path / "result.json",
        config=_config(),
        transport=transport,
        model_metadata_provider=_metadata,
        required_schema_manifest={"test_schema": schema_sha256(SCHEMA)},
    )

    assert len(transport.requests) == 2
    for request in transport.requests:
        assert request.messages[0]["role"] == "system"
        assert request.messages[0]["content"].startswith(
            "historical system instructions\n\n"
        )
        assert request.messages[1] == {"role": "user", "content": "user input"}


def test_interrupted_run_resumes_without_replaying_completed_cases(tmp_path: Path) -> None:
    source = _replay(
        tmp_path / "replay.jsonl",
        _row("first"),
        _row("second"),
    )
    output = tmp_path / "result.json"
    first_transport = ModelTransport(
        {
            "ornith:test": [_payload("approved"), KeyboardInterrupt()],
            "gpt-oss:test": [_payload("approved")],
        }
    )

    with pytest.raises(KeyboardInterrupt):
        evaluate_replays(
            source,
            output,
            config=_config(),
            transport=first_transport,
            model_metadata_provider=_metadata,
        )

    partial = json.loads(output.read_text(encoding="utf-8"))
    assert partial["status"] == "in_progress"
    assert partial["processed_cases"] == 1
    second_transport = ModelTransport(
        {
            "ornith:test": [_payload("approved")],
            "gpt-oss:test": [_payload("approved")],
        }
    )

    resumed = evaluate_replays(
        source,
        output,
        resume=True,
        config=_config(),
        transport=second_transport,
        model_metadata_provider=_metadata,
    )

    assert resumed["status"] == "complete"
    assert resumed["processed_cases"] == 2
    assert len(second_transport.requests) == 2
    assert [case["index"] for case in resumed["cases"]] == [0, 1]


def test_resume_identity_mismatch_stops_before_inference(tmp_path: Path) -> None:
    source = _replay(tmp_path / "replay.jsonl", _row("case"))
    output = tmp_path / "result.json"
    initial = ModelTransport(
        {
            "ornith:test": [_payload("approved")],
            "gpt-oss:test": [_payload("approved")],
        }
    )
    evaluate_replays(
        source,
        output,
        config=_config(),
        transport=initial,
        model_metadata_provider=_metadata,
    )
    unexpected = ModelTransport({})

    with pytest.raises(ResumeMismatchError, match="identity"):
        evaluate_replays(
            source,
            output,
            resume=True,
            config=_config(num_predict=512),
            transport=unexpected,
            model_metadata_provider=_metadata,
        )

    assert unexpected.requests == []


def test_replay_input_cannot_be_used_as_output(tmp_path: Path) -> None:
    source = _replay(tmp_path / "replay.jsonl", _row("case"))

    with pytest.raises(ValueError, match="read-only replay input"):
        evaluate_replays(
            source,
            source,
            config=_config(),
            transport=ModelTransport({}),
            model_metadata_provider=_metadata,
        )
