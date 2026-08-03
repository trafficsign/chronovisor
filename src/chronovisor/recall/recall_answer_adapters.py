"""Built-in Ollama answer runner/scorer for autonomous Recall benchmarks."""

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
    config = load_decision_router_config()
    runner_model = config.primary_model
    scorer_model = config.tie_break_model
    digests = ollama.model_digests([runner_model, scorer_model])
    runner_digest = str(digests.get(runner_model) or "")
    scorer_digest = str(digests.get(scorer_model) or "")
    if (
        not runner_digest
        or not scorer_digest
        or runner_model == scorer_model
        or runner_digest == scorer_digest
    ):
        raise RuntimeError("independent answer runner/scorer models unavailable")
    runner = {
        "runner_id": "builtin-ollama-answer-runner-v1",
        "model": runner_model,
        "model_digest": runner_digest,
        "system_sha256": canonical_sha256(RUNNER_SYSTEM),
        "sampler_sha256": SAMPLER_SHA256,
        "policy_sha256": RUNNER_POLICY_SHA256,
    }
    scorer = {
        "scorer_id": "builtin-ollama-evidence-scorer-v1",
        "version": "1",
        "model": scorer_model,
        "model_digest": scorer_digest,
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
    *, model: str, messages: list[dict[str, str]], schema: Mapping[str, Any], seed: int
) -> dict[str, Any]:
    config = load_decision_router_config()
    with ollama.model_resource_lease(exclusive=True):
        raw = ollama.chat(
            messages,
            model=model,
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
    value = json.loads(raw)
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
        model=str(identity["model"]),
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
        model=str(identity["model"]),
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
    "builtin_answer_adapter_identities",
    "builtin_ollama_answer_runner",
    "builtin_ollama_answer_scorer",
]
