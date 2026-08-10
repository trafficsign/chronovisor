"""Built-in runtime-routed answer runner/scorer for Recall benchmarks."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from chronovisor.core import ollama
from chronovisor.core.durable_state import canonical_sha256
from chronovisor.core.runtime_config import load_decision_router_config

RUNNER_SYSTEM = (
    "Answer the user's question using only the supplied Recall context. "
    "If the context is insufficient, say so. Return the registered JSON object."
)
IDENTITY_SCHEMA = "chronovisor.recall-answer-adapter-identity.v2"
RUNNER_RUNTIME_ROLE = "recall.answer.runner"
SCORER_RUNTIME_ROLE = "recall.answer.scorer"
_RUNTIME_ROLES = (RUNNER_RUNTIME_ROLE, SCORER_RUNTIME_ROLE)
SCORER_SYSTEM = (
    "Score whether the candidate answer is supported by the frozen independent "
    "reference evidence. Do not reward lexical similarity alone. Return the "
    "registered JSON object with correctness, grounding, and citation in [0,1]."
)
RUNNER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["answer"],
    "properties": {"answer": {"type": "string", "minLength": 1}},
}
SCORER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["correctness", "grounding", "citation"],
    "properties": {
        key: {"type": "number", "minimum": 0, "maximum": 1}
        for key in ("correctness", "grounding", "citation")
    },
}
RUNNER_POLICY_SHA256 = canonical_sha256(
    {"version": 1, "system": RUNNER_SYSTEM, "schema": RUNNER_SCHEMA}
)
SCORER_POLICY_SHA256 = canonical_sha256(
    {"version": 1, "system": SCORER_SYSTEM, "schema": SCORER_SCHEMA}
)
SAMPLER_SHA256 = canonical_sha256(
    {"temperature": 0, "seed": "pair_seed", "think": False}
)


def builtin_answer_adapter_identities(
    *, rubric_sha256: str, evidence_manifest_sha256: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    routes = ollama.runtime_generation_routes(_RUNTIME_ROLES)
    if (
        tuple(route.role for route in routes) != _RUNTIME_ROLES
        or not all(route.structured_output for route in routes)
        or len({(route.provider, route.model, route.location) for route in routes}) != 2
    ):
        raise RuntimeError("independent answer runner/scorer models unavailable")
    local_models = list(
        dict.fromkeys(
            route.model
            for route in routes
            if route.provider == "ollama" and route.location == "local"
        )
    )
    digests = ollama.model_digests(local_models) if local_models else {}

    def route_identity(route: ollama.RuntimeGenerationRoute) -> dict[str, Any]:
        digest = (
            str(digests.get(route.model) or "")
            if route.provider == "ollama" and route.location == "local"
            else None
        )
        if digest == "":
            raise RuntimeError("independent answer runner/scorer models unavailable")
        return {
            "role": route.role,
            "provider": route.provider,
            "model": route.model,
            "location": route.location,
            "model_digest": digest,
        }

    runner_route = route_identity(routes[0])
    scorer_route = route_identity(routes[1])
    if runner_route["model_digest"] is not None and (
        runner_route["model_digest"] == scorer_route["model_digest"]
    ):
        raise RuntimeError("independent answer runner/scorer models unavailable")
    runner = {
        "identity_schema": IDENTITY_SCHEMA,
        "runner_id": "builtin-runtime-answer-runner-v2",
        "route_identity": runner_route,
        "model": runner_route["model"],
        "model_digest": runner_route["model_digest"],
        "system_sha256": canonical_sha256(RUNNER_SYSTEM),
        "sampler_sha256": SAMPLER_SHA256,
        "policy_sha256": RUNNER_POLICY_SHA256,
    }
    scorer = {
        "identity_schema": IDENTITY_SCHEMA,
        "scorer_id": "builtin-runtime-evidence-scorer-v2",
        "version": "2",
        "route_identity": scorer_route,
        "model": scorer_route["model"],
        "model_digest": scorer_route["model_digest"],
        "system_sha256": canonical_sha256(SCORER_SYSTEM),
        "sampler_sha256": SAMPLER_SHA256,
        "policy_sha256": SCORER_POLICY_SHA256,
        "rubric_sha256": rubric_sha256,
        "evidence_manifest_sha256": evidence_manifest_sha256,
        "calibration_protocol_sha256": canonical_sha256(
            {"version": 2, "kind": "machine-consensus-controls"}
        ),
    }
    return runner, scorer


def _chat_json(
    *,
    runtime_role: str,
    messages: list[dict[str, str]],
    schema: Mapping[str, Any],
    seed: int,
) -> dict[str, Any]:
    config = load_decision_router_config()
    response = ollama.runtime_structured_chat(
        messages,
        runtime_role=runtime_role,
        source_data_class="raw",
        source_sensitivity="high",
        format=dict(schema),
        num_ctx=config.num_ctx,
        num_predict=config.num_predict,
        keep_alive="0",
        read_timeout_ms=config.read_timeout_ms,
        max_output_chars=config.max_output_chars,
        temperature=0,
        seed=seed,
        think=False,
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


def builtin_ollama_answer_scorer(
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
    dimensions = _chat_json(
        runtime_role=SCORER_RUNTIME_ROLE,
        messages=[
            {"role": "system", "content": SCORER_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"QUESTION:\n{prompt}\n\nCANDIDATE ANSWER:\n{answer}"
                    f"\n\nFROZEN REFERENCE EVIDENCE:\n{reference}"
                ),
            },
        ],
        schema=SCORER_SCHEMA,
        seed=seed,
    )
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


__all__ = [
    "IDENTITY_SCHEMA",
    "RUNNER_RUNTIME_ROLE",
    "SCORER_RUNTIME_ROLE",
    "builtin_answer_adapter_identities",
    "builtin_ollama_answer_runner",
    "builtin_ollama_answer_scorer",
]
