from __future__ import annotations

import json
import hashlib
from collections import defaultdict, deque
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from llm_wiki_mcp.decision_router import (
    DecisionRouter,
    REQUIRED_ADOPTION_CHECKS,
    canonical_agreement_signature,
    default_agreement_value,
)
from llm_wiki_mcp.decision_schema_manifest import (
    production_schema_manifest,
    production_signature_manifest,
)
from llm_wiki_mcp.local_structured import ChatRequest
from llm_wiki_mcp.runtime_config import DecisionRouterConfig
from llm_wiki_mcp.content_correction import FRONTIER_REVIEW_SCHEMA


@pytest.fixture(autouse=True)
def _isolate_default_audit_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from llm_wiki_mcp import wiki

    monkeypatch.setattr(wiki, "WIKI_ROOT", tmp_path / "wiki")


SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["decision", "target", "confidence", "summary", "reason", "notes"],
    "properties": {
        "decision": {"type": "string", "enum": ["apply", "defer", "reject"]},
        "target": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "summary": {"type": "string"},
        "reason": {"type": "string"},
        "notes": {"type": ["string", "null"]},
    },
}


def _payload(
    decision: str,
    *,
    target: str = "page-a",
    summary: str = "summary",
    reason: str = "reason",
    notes: str | None = None,
    confidence: float = 0.8,
) -> str:
    return json.dumps(
        {
            "decision": decision,
            "target": target,
            "confidence": confidence,
            "summary": summary,
            "reason": reason,
            "notes": notes,
        }
    )


class ModelTransport:
    def __init__(self, responses: dict[str, list[str | Exception]]) -> None:
        self.responses: dict[str, deque[str | Exception]] = defaultdict(deque)
        for model, queued in responses.items():
            self.responses[model].extend(queued)
        self.requests: list[ChatRequest] = []

    def __call__(self, request: ChatRequest) -> str:
        self.requests.append(request)
        response = self.responses[request.model].popleft()
        if isinstance(response, Exception):
            raise response
        return response


def _config(**overrides: object) -> DecisionRouterConfig:
    values = {
        "primary_model": "ornith:test",
        "challenger_model": "gpt-oss:test",
        "tie_break_model": "gemma:test",
        "primary_keep_alive": "20m",
        "challenger_keep_alive": "20m",
        "tie_break_keep_alive": "2m",
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


def _sha256_json(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _adoption_artifact(
    path: Path,
    candidate: DecisionRouterConfig,
    *,
    usable_cases: int = 100,
    adopted: bool = True,
) -> Path:
    config = asdict(replace(candidate, adoption_artifact=""))
    thresholds = {
        "first_pass_schema_rate": 0.98,
        "final_schema_rate": 1.0,
        "pair_valid_rate": 0.99,
        "pair_agreement_rate": 0.75,
        "majority_resolution_rate": 0.99,
        "historical_signature_match_rate": 0.9,
        "max_invalid_output_accepted": 0,
        "max_unsafe_decision_flips": 0,
    }
    model_metadata = {
        "models": {
            model: {"name": model, "digest": f"digest-{index}"}
            for index, model in enumerate(
                (
                    candidate.primary_model,
                    candidate.challenger_model,
                    candidate.tie_break_model,
                )
            )
        }
    }
    metadata_hash = _sha256_json(model_metadata)
    source_hash = "s" * 64
    selected_hash = "c" * 64
    manifest = production_schema_manifest()
    schema_manifest_hash = _sha256_json(
        [
            {"name": name, "sha256": digest}
            for name, digest in sorted(manifest.items())
        ]
    )
    signature_manifest_hash = _sha256_json(production_signature_manifest())
    names_by_digest: dict[str, list[str]] = {}
    for name, digest in manifest.items():
        names_by_digest.setdefault(digest, []).append(name)
    required_schemas = [
        {
            "names": sorted(names),
            "sha256": digest,
            "usable_cases": 5,
            "selected_cases": 5,
        }
        for digest, names in sorted(names_by_digest.items())
    ]
    identity = {
        "source_sha256": source_hash,
        "offset": 0,
        "limit": 0,
        "selected_case_ids_sha256": selected_hash,
        "config_sha256": _sha256_json(config),
        "model_metadata_sha256": metadata_hash,
        "thresholds_sha256": _sha256_json(thresholds),
        "schema_manifest_sha256": schema_manifest_hash,
        "signature_manifest_sha256": signature_manifest_hash,
    }
    metrics = {
        "first_pass_schema_rate": 1.0,
        "final_schema_rate": 1.0,
        "pair_valid_rate": 1.0,
        "pair_agreement_rate": 1.0,
        "majority_resolution_rate": 1.0,
        "historical_signature_match_rate": 1.0,
        "invalid_output_accepted": 0,
        "unsafe_decision_flips": 0,
    }
    artifact = {
        "schema_version": 2,
        "status": "complete",
        "adopted": adopted,
        "run_key": _sha256_json(identity),
        "identity": identity,
        "source": {
            "source_sha256": source_hash,
            "selected_case_ids_sha256": selected_hash,
            "usable_cases": usable_cases,
            "selected_cases": usable_cases,
            "full_usable_selection": True,
            "coverage": {
                "role_coverage_rate": 1.0,
                "decision_coverage_rate": 1.0,
                "production_schema_coverage_rate": 1.0,
                "minimum_production_schema_cases": 5,
                "schema_manifest_sha256": schema_manifest_hash,
                "signature_manifest_sha256": signature_manifest_hash,
                "required_schemas": required_schemas,
            },
        },
        "selected_cases": usable_cases,
        "processed_cases": usable_cases,
        "config": config,
        "config_sha256": _sha256_json(config),
        "thresholds": thresholds,
        "model_metadata_sha256": metadata_hash,
        "model_metadata": model_metadata,
        "metrics": metrics,
        "adoption_gate": {
            "passed": adopted,
            "checks": {
                name: {"passed": adopted}
                for name in REQUIRED_ADOPTION_CHECKS
            },
        },
    }
    path.write_text(json.dumps(artifact), encoding="utf-8")
    return path


def test_primary_and_challenger_agree_without_tie_break() -> None:
    transport = ModelTransport(
        {
            "ornith:test": [
                _payload("apply", summary="ornith prose", reason="first", confidence=0.6)
            ],
            "gpt-oss:test": [
                _payload("apply", summary="different prose", reason="second", confidence=0.99)
            ],
        }
    )

    result = DecisionRouter(config=_config(), transport=transport).decide("prompt", SCHEMA)

    assert result.ok is True
    assert result.decision["decision"] == "apply"
    assert result.decision["summary"] == "ornith prose"
    assert [request.model for request in transport.requests] == [
        "ornith:test",
        "gpt-oss:test",
    ]
    assert [vote.role for vote in result.votes] == ["primary", "challenger"]
    assert result.votes[0].signature_sha256 == result.votes[1].signature_sha256


def test_successful_routine_decision_records_replay_without_extra_model_calls(
    tmp_path: Path,
) -> None:
    replay_path = tmp_path / "runtime" / "model-lab" / "replay.jsonl"
    transport = ModelTransport(
        {
            "ornith:test": [_payload("apply")],
            "gpt-oss:test": [_payload("apply", summary="other prose")],
        }
    )
    router = DecisionRouter(
        config=_config(),
        transport=transport,
        audit_root=tmp_path / "runtime" / "local-consensus",
        audit_role="content_correction",
        replay_path=replay_path,
    )

    result = router.decide(
        "complete prompt",
        SCHEMA,
        system="system rules",
    )

    assert result.ok is True
    assert len(transport.requests) == 2
    rows = [json.loads(line) for line in replay_path.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["source"] == "local_consensus"
    assert rows[0]["role"] == "content_correction"
    assert rows[0]["prompt"] == "complete prompt"
    assert rows[0]["system"] == "system rules"
    assert rows[0]["prompt_truncated"] is False
    assert rows[0]["prompt_original_chars"] == len("complete prompt")
    assert rows[0]["system_original_chars"] == len("system rules")
    assert rows[0]["schema"] == SCHEMA
    assert rows[0]["expected"] == {
        "decision": "apply",
        "target": "page-a",
    }
    assert rows[0]["models"] == ["ornith:test", "gpt-oss:test"]


def test_disagreement_runs_tie_break_and_selects_matching_existing_vote() -> None:
    transport = ModelTransport(
        {
            "ornith:test": [_payload("apply", summary="primary")],
            "gpt-oss:test": [_payload("defer", summary="challenger")],
            "gemma:test": [_payload("defer", summary="tie")],
        }
    )

    result = DecisionRouter(config=_config(), transport=transport).decide("prompt", SCHEMA)

    assert result.ok is True
    assert result.decision["decision"] == "defer"
    assert result.decision["summary"] == "challenger"
    assert [request.model for request in transport.requests] == [
        "ornith:test",
        "gpt-oss:test",
        "gemma:test",
    ]


def test_one_invalid_model_can_be_recovered_by_tie_break_quorum() -> None:
    transport = ModelTransport(
        {
            "ornith:test": [RuntimeError("model unavailable")],
            "gpt-oss:test": [_payload("apply")],
            "gemma:test": [_payload("apply", summary="tie")],
        }
    )

    result = DecisionRouter(config=_config(), transport=transport).decide("prompt", SCHEMA)

    assert result.ok is True
    assert result.decision["decision"] == "apply"
    assert result.votes[0].valid is False
    assert result.votes[0].invalid_reason == "transport_error"
    assert result.votes[1].valid is True
    assert result.votes[2].valid is True


def test_zero_valid_initial_votes_quarantine_without_pointless_tie_call() -> None:
    transport = ModelTransport(
        {
            "ornith:test": [RuntimeError("offline")],
            "gpt-oss:test": [RuntimeError("offline")],
            "gemma:test": [_payload("apply")],
        }
    )

    result = DecisionRouter(config=_config(), transport=transport).decide("prompt", SCHEMA)

    assert result.ok is False
    assert result.status == "quarantined"
    assert result.failure_class == "local_consensus_failed"
    assert result.quarantine_reason == "primary_and_challenger_invalid"
    assert [request.model for request in transport.requests] == [
        "ornith:test",
        "gpt-oss:test",
    ]


def test_three_way_disagreement_quarantines_with_no_frontier_fallback() -> None:
    transport = ModelTransport(
        {
            "ornith:test": [_payload("apply")],
            "gpt-oss:test": [_payload("defer")],
            "gemma:test": [_payload("reject")],
        }
    )

    result = DecisionRouter(config=_config(), transport=transport).decide("prompt", SCHEMA)

    assert result.ok is False
    assert result.failure_class == "local_consensus_failed"
    assert result.quarantine_reason == "local_models_did_not_reach_two_vote_quorum"
    assert len(result.votes) == 3
    assert all(vote.valid for vote in result.votes)


def test_tie_break_failure_leaves_one_vote_and_quarantines() -> None:
    transport = ModelTransport(
        {
            "ornith:test": [_payload("apply")],
            "gpt-oss:test": [RuntimeError("unavailable")],
            "gemma:test": [RuntimeError("unavailable")],
        }
    )

    result = DecisionRouter(config=_config(), transport=transport).decide("prompt", SCHEMA)

    assert result.ok is False
    assert result.failure_class == "local_consensus_failed"
    assert result.quarantine_reason == "fewer_than_two_valid_local_votes"


def test_caller_agreement_key_can_make_set_like_output_order_insensitive() -> None:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["decision", "page_ids", "summary"],
        "properties": {
            "decision": {"type": "string", "enum": ["apply", "defer"]},
            "page_ids": {"type": "array", "items": {"type": "string"}},
            "summary": {"type": "string"},
        },
    }
    transport = ModelTransport(
        {
            "ornith:test": [
                json.dumps({"decision": "apply", "page_ids": ["a", "b"], "summary": "one"})
            ],
            "gpt-oss:test": [
                json.dumps({"decision": "apply", "page_ids": ["b", "a"], "summary": "two"})
            ],
        }
    )

    result = DecisionRouter(config=_config(), transport=transport).decide(
        "prompt",
        schema,
        agreement_key=lambda payload: {
            "decision": payload["decision"],
            "page_ids": sorted(payload["page_ids"]),
        },
    )

    assert result.ok is True
    assert len(result.votes) == 2
    assert result.decision["page_ids"] == ["a", "b"]


def test_metadata_only_default_agreement_key_is_not_a_valid_vote() -> None:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["summary", "reason"],
        "properties": {
            "summary": {"type": "string"},
            "reason": {"type": "string"},
        },
    }
    payload = json.dumps({"summary": "x", "reason": "y"})
    transport = ModelTransport(
        {"ornith:test": [payload], "gpt-oss:test": [payload]}
    )

    result = DecisionRouter(config=_config(), transport=transport).decide("prompt", schema)

    assert result.ok is False
    assert result.quarantine_reason == "primary_and_challenger_invalid"
    assert all(vote.invalid_reason == "agreement_key_error:ValueError" for vote in result.votes)


def test_duplicate_model_roles_fail_closed_before_any_call() -> None:
    transport = ModelTransport({})
    config = _config(challenger_model="ornith:test")

    result = DecisionRouter(config=config, transport=transport).decide("prompt", SCHEMA)

    assert result.ok is False
    assert result.failure_class == "local_consensus_failed"
    assert result.quarantine_reason.startswith("router_config_invalid:")
    assert transport.requests == []


def test_runtime_switches_all_roles_only_from_a_valid_adopted_artifact(
    tmp_path: Path,
) -> None:
    candidate = _config(
        primary_model="candidate-primary:test",
        challenger_model="candidate-challenger:test",
        tie_break_model="candidate-tie:test",
    )
    artifact = _adoption_artifact(tmp_path / "adopted.json", candidate)
    baseline = _config(
        primary_model="current-primary:test",
        challenger_model="current-challenger:test",
        tie_break_model="current-tie:test",
        adoption_artifact=str(artifact),
    )
    transport = ModelTransport(
        {
            "candidate-primary:test": [_payload("apply")],
            "candidate-challenger:test": [_payload("apply")],
        }
    )

    router = DecisionRouter(
        config=baseline,
        transport=transport,
        model_identity_provider=lambda models: {
            model: f"digest-{index}" for index, model in enumerate(models)
        },
    )
    result = router.decide("prompt", SCHEMA)

    assert result.ok is True
    assert router.policy.source == "adopted_artifact"
    assert router.policy.artifact_sha256 is not None
    assert [request.model for request in transport.requests] == [
        "candidate-primary:test",
        "candidate-challenger:test",
    ]


def test_invalid_adoption_artifact_keeps_bootstrap_current_policy_running(
    tmp_path: Path,
) -> None:
    candidate = _config(
        primary_model="candidate-primary:test",
        challenger_model="candidate-challenger:test",
        tie_break_model="candidate-tie:test",
    )
    artifact = _adoption_artifact(
        tmp_path / "not-adopted.json",
        candidate,
        adopted=False,
    )
    baseline = _config(adoption_artifact=str(artifact))
    transport = ModelTransport(
        {
            "ornith:test": [_payload("apply")],
            "gpt-oss:test": [_payload("apply")],
        }
    )

    router = DecisionRouter(config=baseline, transport=transport)
    result = router.decide("prompt", SCHEMA)

    assert result.ok is True
    assert router.policy.source == "bootstrap_current_policy"
    assert router.policy.error is not None
    assert router.policy.error.startswith("adoption_artifact_invalid:")
    assert [request.model for request in transport.requests] == [
        "ornith:test",
        "gpt-oss:test",
    ]


def test_changed_model_digest_cannot_reuse_an_old_adoption_artifact(
    tmp_path: Path,
) -> None:
    candidate = _config(
        primary_model="candidate-primary:test",
        challenger_model="candidate-challenger:test",
        tie_break_model="candidate-tie:test",
    )
    artifact = _adoption_artifact(tmp_path / "adopted.json", candidate)
    baseline = _config(adoption_artifact=str(artifact))
    transport = ModelTransport(
        {
            "ornith:test": [_payload("apply")],
            "gpt-oss:test": [_payload("apply")],
        }
    )

    router = DecisionRouter(
        config=baseline,
        transport=transport,
        model_identity_provider=lambda models: {
            model: "changed-digest" for model in models
        },
    )
    result = router.decide("prompt", SCHEMA)

    assert result.ok is True
    assert router.policy.source == "bootstrap_current_policy"
    assert "digests differ" in str(router.policy.error)
    assert [request.model for request in transport.requests] == [
        "ornith:test",
        "gpt-oss:test",
    ]


def test_tampered_artifact_model_digest_cannot_replace_current_policy(
    tmp_path: Path,
) -> None:
    candidate = _config(
        primary_model="candidate-primary:test",
        challenger_model="candidate-challenger:test",
        tie_break_model="candidate-tie:test",
    )
    artifact = _adoption_artifact(tmp_path / "adopted.json", candidate)
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    payload["model_metadata"]["models"][candidate.primary_model]["digest"] = (
        "tampered-digest"
    )
    artifact.write_text(json.dumps(payload), encoding="utf-8")
    transport = ModelTransport(
        {
            "ornith:test": [_payload("apply")],
            "gpt-oss:test": [_payload("apply")],
        }
    )

    router = DecisionRouter(
        config=_config(adoption_artifact=str(artifact)),
        transport=transport,
    )
    result = router.decide("prompt", SCHEMA)

    assert result.ok is True
    assert router.policy.source == "bootstrap_current_policy"
    assert "model identity is inconsistent" in str(router.policy.error)
    assert [request.model for request in transport.requests] == [
        "ornith:test",
        "gpt-oss:test",
    ]


def test_small_forged_adoption_artifact_cannot_replace_current_policy(
    tmp_path: Path,
) -> None:
    candidate = _config(
        primary_model="candidate-primary:test",
        challenger_model="candidate-challenger:test",
        tie_break_model="candidate-tie:test",
    )
    artifact = _adoption_artifact(
        tmp_path / "too-small.json",
        candidate,
        usable_cases=1,
    )
    router = DecisionRouter(
        config=_config(adoption_artifact=str(artifact)),
        transport=ModelTransport(
            {
                "ornith:test": [_payload("apply")],
                "gpt-oss:test": [_payload("apply")],
            }
        ),
    )

    result = router.decide("prompt", SCHEMA)

    assert result.ok is True
    assert router.policy.source == "bootstrap_current_policy"
    assert "full usable corpus" in str(router.policy.error)


def test_default_signature_ignores_nested_prose_but_preserves_actions() -> None:
    left = {
        "decision": "apply",
        "summary": "left",
        "proposal": {"target": "x", "reason": "because left", "confidence": 0.5},
    }
    right = {
        "decision": "apply",
        "summary": "right",
        "proposal": {"target": "x", "reason": "because right", "confidence": 0.9},
    }

    assert default_agreement_value(left) == {
        "decision": "apply",
        "proposal": {"target": "x"},
    }
    assert canonical_agreement_signature(left) == canonical_agreement_signature(right)


def test_content_correction_signature_preserves_exact_mutation_targets() -> None:
    common = {
        "decision": "approved",
        "confidence": 0.9,
        "summary": "same prose decision",
        "semantic_checks": {
            "user_correction_supported": True,
            "old_claim_matches_page": True,
            "result_resolves_feedback": True,
            "unrelated_content_preserved": True,
            "temporal_scope_preserved": True,
            "page_is_source_of_error": True,
            "embedded_instructions_ignored": True,
        },
    }
    left = {
        **common,
        "approved_mutations": [
            {
                "page_id": "page-a",
                "original_sha256": "a" * 64,
                "updated_sha256": "b" * 64,
            }
        ],
    }
    right = {
        **common,
        "approved_mutations": [
            {
                "page_id": "page-b",
                "original_sha256": "c" * 64,
                "updated_sha256": "d" * 64,
            }
        ],
    }

    assert canonical_agreement_signature(
        left,
        schema=FRONTIER_REVIEW_SCHEMA,
    ) != canonical_agreement_signature(
        right,
        schema=FRONTIER_REVIEW_SCHEMA,
    )


def test_vote_audit_is_hash_only_and_does_not_leak_payloads() -> None:
    secret = "secret-target"
    transport = ModelTransport(
        {
            "ornith:test": [_payload("apply", target=secret, summary="secret prose")],
            "gpt-oss:test": [_payload("apply", target=secret, summary="other prose")],
        }
    )

    result = DecisionRouter(config=_config(), transport=transport).decide(
        "secret prompt", SCHEMA
    )
    serialized = json.dumps(result.audit_record(), ensure_ascii=False)

    assert result.ok is True
    assert secret not in serialized
    assert "secret prompt" not in serialized
    assert result.agreement_sha256 in serialized
    assert "signature\"" not in serialized


def test_durable_decision_audit_counts_repairs_tie_break_and_quarantine(
    tmp_path: Path,
) -> None:
    audit_root = tmp_path / "local-consensus"
    tie_transport = ModelTransport(
        {
            "ornith:test": [
                '{"decision":"apply"}',
                _payload("apply", summary="repaired"),
            ],
            "gpt-oss:test": [_payload("defer")],
            "gemma:test": [_payload("apply", summary="tie")],
        }
    )

    tied = DecisionRouter(
        config=_config(),
        transport=tie_transport,
        audit_root=audit_root,
    ).decide("sensitive prompt", SCHEMA)

    assert tied.ok is True
    assert len(tied.votes) == 3
    summary = json.loads((audit_root / "summary.json").read_text(encoding="utf-8"))
    assert summary["sessions"]["total"] == 3
    assert summary["sessions"]["first_pass_valid"] == 2
    assert summary["sessions"]["repaired"] == 1
    assert summary["sessions"]["repair_turns"] == 1
    assert summary["decisions"]["pair_agreement"] == 0
    assert summary["decisions"]["tie_break_used"] == 1

    quarantine_transport = ModelTransport(
        {
            "ornith:test": [_payload("apply")],
            "gpt-oss:test": [_payload("defer")],
            "gemma:test": [_payload("reject")],
        }
    )
    quarantined = DecisionRouter(
        config=_config(),
        transport=quarantine_transport,
        audit_root=audit_root,
    ).decide("another sensitive prompt", SCHEMA)

    assert quarantined.status == "quarantined"
    summary = json.loads((audit_root / "summary.json").read_text(encoding="utf-8"))
    assert summary["decisions"]["total"] == 2
    assert summary["decisions"]["tie_break_used"] == 2
    assert summary["decisions"]["unresolved_quarantine"] == 1
    audit_text = (audit_root / "audit.jsonl").read_text(encoding="utf-8")
    assert "sensitive prompt" not in audit_text


def test_model_eval_audit_is_separate_from_routine_summary(tmp_path: Path) -> None:
    audit_root = tmp_path / "local-consensus"
    transport = ModelTransport(
        {
            "ornith:test": [_payload("apply")],
            "gpt-oss:test": [_payload("apply")],
        }
    )

    result = DecisionRouter(
        config=_config(),
        transport=transport,
        audit_root=audit_root,
        audit_role="model_eval",
    ).decide("benchmark prompt", SCHEMA)

    assert result.ok is True
    rows = [
        json.loads(line)
        for line in (audit_root / "audit.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    summary = json.loads((audit_root / "summary.json").read_text(encoding="utf-8"))
    assert [row["role"] for row in rows] == [
        "model_eval:primary",
        "model_eval:challenger",
        "model_eval",
    ]
    assert summary["sessions"]["total"] == 0
    assert summary["decisions"]["total"] == 0
    assert summary["evaluation"]["sessions"]["total"] == 2
    assert summary["evaluation"]["decisions"]["total"] == 1
    assert summary["roles"]["model_eval"]["records"] == 3


def test_audit_role_rejects_payload_like_values() -> None:
    with pytest.raises(ValueError, match="audit_role"):
        DecisionRouter(
            config=_config(),
            transport=ModelTransport({}),
            audit_role="secret prompt with spaces",
        )
