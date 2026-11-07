"""Built-in runtime-routed answer runner/scorer for Recall benchmarks."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from chronovisor.core import ollama

# isort: split
from chronovisor.core import reranker_client
from chronovisor.core.decision_mlx import (
    ANSWER_AXES,
    ANSWER_AXIS_QUESTIONS,
    JUDGE_ANSWER_AXES,
    JUDGE_POLICY_SHA256,
    decision_route,
)
from chronovisor.core.durable_state import canonical_sha256
from chronovisor.core.model_execution import local_reasoning_settings
from chronovisor.core.runtime_config import load_decision_router_config

# isort: split
from chronovisor.core.runtime_config import load_reranker_config

RUNNER_SYSTEM = (
    "Answer the user's question using only the supplied Recall context. "
    "If the context is insufficient, say so. Return the registered JSON object."
)
IDENTITY_SCHEMA = "chronovisor.recall-answer-adapter-identity.v2"
RUNNER_RUNTIME_ROLE = "recall.answer.runner"
SCORER_RUNTIME_ROLE = "recall.answer.scorer"
SCORER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["correctness", "grounding", "citation"],
    "properties": {
        key: {"type": "number", "minimum": 0, "maximum": 1} for key in ANSWER_AXES
    },
}
SCORER_SYSTEM = (
    "Deterministic StartLux decision readout for answer correctness, grounding, "
    "and citation against frozen reference evidence."
)
SCORER_TIMEOUT_MS = 10_000
RUNNER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["answer"],
    "properties": {"answer": {"type": "string", "minLength": 1}},
}
RUNNER_POLICY_SHA256 = canonical_sha256(
    {"version": 1, "system": RUNNER_SYSTEM, "schema": RUNNER_SCHEMA}
)


def _scorer_policy_sha256(decision_revision: str) -> str:
    return canonical_sha256(
        {
            "version": 2,
            "kind": "startlux-answer-axes",
            "system": SCORER_SYSTEM,
            "schema": SCORER_SCHEMA,
            "questions": ANSWER_AXIS_QUESTIONS,
            "judge_policy_sha256": JUDGE_POLICY_SHA256,
            "decision_revision": decision_revision,
        }
    )


SAMPLER_SHA256 = canonical_sha256(
    {"temperature": 0, "seed": "pair_seed", "think": False}
)
DECISION_SAMPLER_SHA256 = canonical_sha256(
    {"kind": "deterministic-decision-readout", "temperature": 0, "seed": 0}
)


def _sampler_identity(role: str) -> str:
    settings = local_reasoning_settings(role, "")
    if settings is None:
        return SAMPLER_SHA256
    config = load_decision_router_config()
    return canonical_sha256(
        {
            "temperature": 0,
            "seed": "pair_seed",
            "think": settings["level"],
            "runtime_budgets": {
                "num_ctx": max(config.num_ctx, settings["num_ctx"]),
                "num_predict": max(config.num_predict, settings["num_predict"]),
            },
        }
    )


def builtin_answer_adapter_identities(
    *, rubric_sha256: str, evidence_manifest_sha256: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    routes = ollama.runtime_generation_routes((RUNNER_RUNTIME_ROLE,))
    if (
        tuple(route.role for route in routes) != (RUNNER_RUNTIME_ROLE,)
        or not routes[0].structured_output
    ):
        raise RuntimeError("independent answer runner/scorer models unavailable")
    runner = routes[0]
    local_models = (
        [runner.model]
        if runner.provider == "ollama" and runner.location == "local"
        else []
    )
    digests = ollama.model_digests(local_models) if local_models else {}
    runner_digest = (
        str(digests.get(runner.model) or "")
        if runner.provider == "ollama" and runner.location == "local"
        else None
    )
    if runner_digest == "":
        raise RuntimeError("independent answer runner/scorer models unavailable")
    try:
        scorer = decision_route(SCORER_RUNTIME_ROLE)
    except Exception as exc:
        raise RuntimeError(
            "independent answer runner/scorer models unavailable"
        ) from exc
    if (
        runner.provider,
        runner.model,
        runner.location,
    ) == (scorer.provider, scorer.model, scorer.location):
        raise RuntimeError("independent answer runner/scorer models unavailable")
    runner_route = {
        "role": runner.role,
        "provider": runner.provider,
        "model": runner.model,
        "location": runner.location,
        "model_digest": runner_digest,
    }
    scorer_route = {
        "role": scorer.role,
        "provider": scorer.provider,
        "model": scorer.model,
        "location": scorer.location,
        # The adapter identity schema permits a digest only for local ollama.
        # The model revision is pinned inside policy_sha256 instead.
        "model_digest": None,
    }
    runner_identity = {
        "identity_schema": IDENTITY_SCHEMA,
        "runner_id": "builtin-runtime-answer-runner-v2",
        "route_identity": runner_route,
        "model": runner_route["model"],
        "model_digest": runner_route["model_digest"],
        "system_sha256": canonical_sha256(RUNNER_SYSTEM),
        "sampler_sha256": _sampler_identity(RUNNER_RUNTIME_ROLE),
        "policy_sha256": RUNNER_POLICY_SHA256,
    }
    scorer_identity = {
        "identity_schema": IDENTITY_SCHEMA,
        "scorer_id": "builtin-startlux-evidence-scorer-v1",
        "version": "1",
        "route_identity": scorer_route,
        "model": scorer_route["model"],
        "model_digest": None,
        "system_sha256": canonical_sha256(SCORER_SYSTEM),
        "sampler_sha256": DECISION_SAMPLER_SHA256,
        "policy_sha256": _scorer_policy_sha256(scorer.revision),
        "rubric_sha256": rubric_sha256,
        "evidence_manifest_sha256": evidence_manifest_sha256,
        "calibration_protocol_sha256": canonical_sha256(
            {"version": 2, "kind": "machine-consensus-controls"}
        ),
    }
    return runner_identity, scorer_identity


def _chat_json(
    *,
    runtime_role: str,
    messages: list[dict[str, str]],
    schema: Mapping[str, Any],
    seed: int,
) -> dict[str, Any]:
    config = load_decision_router_config()
    settings = local_reasoning_settings(runtime_role, "")
    response = ollama.runtime_structured_chat(
        messages,
        runtime_role=runtime_role,
        source_data_class="raw",
        source_sensitivity="high",
        format=dict(schema),
        num_ctx=max(config.num_ctx, settings["num_ctx"])
        if settings
        else config.num_ctx,
        num_predict=max(config.num_predict, settings["num_predict"])
        if settings
        else config.num_predict,
        keep_alive="0",
        read_timeout_ms=config.read_timeout_ms,
        max_output_chars=config.max_output_chars,
        temperature=0,
        seed=seed,
        think=settings["level"] if settings else False,
    )
    value = json.loads(response.content)
    if not isinstance(value, dict):
        raise ValueError("answer adapter returned a non-object")
    return value


def builtin_ollama_answer_runner(
    prompt: str, context: str, generation: Mapping[str, Any]
) -> Mapping[str, Any]:
    identity, _scorer = builtin_answer_adapter_identities(
        rubric_sha256="0" * 64, evidence_manifest_sha256="0" * 64
    )
    seed = int(generation.get("seed") or 0)
    value = _chat_json(
        runtime_role=RUNNER_RUNTIME_ROLE,
        messages=[
            {"role": "system", "content": RUNNER_SYSTEM},
            {
                "role": "user",
                "content": f"QUESTION:\n{prompt}\n\nRECALL CONTEXT:\n{context}",
            },
        ],
        schema=RUNNER_SCHEMA,
        seed=seed,
    )
    answer = value.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("answer runner returned an empty answer")
    return {
        "answer": answer.strip(),
        "identity": identity,
        "reset_receipt": {
            "seed": seed,
            "base_state_sha256": generation.get("base_state_sha256"),
            "reset_protocol_sha256": identity["policy_sha256"],
        },
    }


def builtin_startlux_answer_scorer(
    prompt: str,
    answer: str,
    gold: Mapping[str, Any],
    scoring: Mapping[str, Any],
) -> Mapping[str, Any]:
    evidence = gold.get("evidence")
    evidence_map = evidence if isinstance(evidence, Mapping) else {}
    source_packet = evidence_map.get("source_packet")
    source_map = source_packet if isinstance(source_packet, Mapping) else {}
    chunks = source_map.get("evidence_chunks")
    if not isinstance(chunks, list):
        raise ValueError("scorer gold evidence is missing")
    reference = "\n\n".join(
        f"[PAGE {chunk.get('page_id')}]\n{chunk.get('excerpt')}"
        for chunk in chunks
        if isinstance(chunk, Mapping)
    )
    _runner, identity = builtin_answer_adapter_identities(
        rubric_sha256=str(gold.get("rubric_sha256") or "0" * 64),
        evidence_manifest_sha256=str(
            scoring.get("evidence_manifest_sha256") or "0" * 64
        ),
    )
    seed = int(scoring.get("seed") or 0)
    response = reranker_client.decision_judge(
        JUDGE_ANSWER_AXES,
        {"query": prompt, "answer": answer, "reference": reference},
        role=SCORER_RUNTIME_ROLE,
        config=load_reranker_config(),
        timeout_ms=SCORER_TIMEOUT_MS,
    )
    raw_scores = response.get("scores")
    if not isinstance(raw_scores, Mapping) or set(raw_scores) != set(ANSWER_AXES):
        raise ValueError("decision scorer returned invalid dimensions")
    dimensions = {axis: float(raw_scores[axis]) for axis in ANSWER_AXES}
    return {
        "identity": identity,
        "evidence_sha256": gold.get("evidence_sha256"),
        "reset_receipt": {
            "seed": seed,
            "base_state_sha256": scoring.get("base_state_sha256"),
            "reset_protocol_sha256": identity["policy_sha256"],
        },
        "dimensions": dimensions,
    }


# Keep the old import name working until every caller is migrated.
builtin_ollama_answer_scorer = builtin_startlux_answer_scorer


__all__ = [
    "IDENTITY_SCHEMA",
    "RUNNER_RUNTIME_ROLE",
    "SCORER_RUNTIME_ROLE",
    "builtin_answer_adapter_identities",
    "builtin_ollama_answer_runner",
    "builtin_ollama_answer_scorer",
    "builtin_startlux_answer_scorer",
]
