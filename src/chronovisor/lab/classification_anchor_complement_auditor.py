"""Audit one complementary retrieval anchor without core-anchor presumption."""

from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from typing import Any

from chronovisor.core import ollama
from chronovisor.classification.classification import ClassificationError
from chronovisor.classification.classification_anchor import UNRESOLVED_ANCHOR_ID

WORKER_SCHEMA = "chronovisor.classification-anchor-complement-auditor.v1"
AUDIT_SCHEMA = "chronovisor.classification-anchor-complement-audit.v1"
NONE = "NONE"
POLICY = {
    "contract_version": 1,
    "purpose": "find one independently useful co-primary retrieval route",
    "presumption": "neither one nor two anchors is preferred",
    "admission_axes": [
        "different_principal_axis",
        "independent_retrieval_route",
        "explicit_document_evidence",
        "not_incidental_context",
    ],
    "rules": [
        "Admit at most one complement and only when all four axes are true.",
        "The fixed core is one valid route, not proof that every other route is subsumed.",
        "Document purpose can be independently principal: interview, resume, evaluation, market analysis, or personal planning.",
        "A title-level subject, repeated section-level subject, or explicit document purpose is strong evidence.",
        "An employer name, tool, material, implementation detail, example, or motivation alone is incidental.",
        "Uncertainty between labels is not evidence for two anchors.",
    ],
    "positive_examples": [
        "A benchmark evaluating mathematical structure in LLMs needs both mathematics and AI retrieval routes.",
        "A defense engineering profile written to evaluate a job transition needs defense and career routes.",
        "A document analyzing both market evolution and OSS firmware needs business and software routes.",
        "Defense-industry interview guidance needs career and defense routes when both are explicitly analyzed.",
    ],
    "negative_examples": [
        "An automotive component standard does not gain manufacturing merely because welding or material is mentioned.",
        "A software pipeline does not gain AI merely because an AI model is one dependency.",
        "A career roadmap does not gain every named employer industry when their technology is not analyzed.",
    ],
    "forbidden_inputs": [
        "acceptable_anchor_sets",
        "defensible_anchor_ids",
        "gold_basis",
        "target_anchor_ids",
        "expected_primary_anchor_ids",
        "udc_scope",
    ],
}
PROMPT_SHA256 = "sha256:" + hashlib.sha256(
    json.dumps(POLICY, ensure_ascii=False, sort_keys=True).encode()
).hexdigest()


def _audit_schema(anchor_ids: Sequence[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "second_anchor_id",
            "different_principal_axis",
            "independent_retrieval_route",
            "explicit_document_evidence",
            "not_incidental_context",
            "rationale",
        ],
        "properties": {
            "second_anchor_id": {
                "type": "string",
                "enum": [NONE, *anchor_ids],
            },
            "different_principal_axis": {"type": "boolean"},
            "independent_retrieval_route": {"type": "boolean"},
            "explicit_document_evidence": {"type": "boolean"},
            "not_incidental_context": {"type": "boolean"},
            "rationale": {"type": "string", "minLength": 1, "maxLength": 500},
        },
    }


def validate_audit(
    value: Mapping[str, Any],
    *,
    core_anchor_id: str,
    anchor_ids: Sequence[str],
) -> dict[str, Any]:
    second = str(value.get("second_anchor_id") or NONE)
    axes = {
        "different_principal_axis": bool(
            value.get("different_principal_axis")
        ),
        "independent_retrieval_route": bool(
            value.get("independent_retrieval_route")
        ),
        "explicit_document_evidence": bool(
            value.get("explicit_document_evidence")
        ),
        "not_incidental_context": bool(value.get("not_incidental_context")),
    }
    rationale = str(value.get("rationale") or "").strip()[:500]
    invalid_reason = ""
    if (
        second not in anchor_ids
        or second in {core_anchor_id, UNRESOLVED_ANCHOR_ID}
    ):
        second = NONE
        if str(value.get("second_anchor_id") or NONE) != NONE:
            invalid_reason = "invalid_second_anchor"
    admitted = second != NONE and all(axes.values()) and bool(rationale)
    if not admitted:
        second = NONE
    if not rationale:
        invalid_reason = invalid_reason or "missing_rationale"
    return {
        "schema": AUDIT_SCHEMA,
        "second_anchor_id": second,
        **axes,
        "admitted": admitted,
        "rationale": rationale,
        "invalid_reason": invalid_reason,
    }


def run(payload: Mapping[str, Any]) -> dict[str, Any]:
    if payload.get("schema") != WORKER_SCHEMA:
        raise ClassificationError("unsupported complement auditor schema")
    model = str(payload.get("model") or "")
    expected_digest = str(payload.get("model_digest") or "")
    page = payload.get("page")
    subject = payload.get("subject")
    core = payload.get("core_anchor")
    alternatives = payload.get("alternative_anchors")
    if (
        not model
        or not expected_digest
        or not isinstance(page, Mapping)
        or set(page) - {"title", "summary", "evidence_excerpt"}
        or not isinstance(subject, Mapping)
        or not isinstance(core, Mapping)
        or not isinstance(alternatives, list)
    ):
        raise ClassificationError("complement auditor input is incomplete")
    core_id = str(core.get("id") or "")
    cards = [
        {
            "id": str(row.get("id") or ""),
            "label_ja": str(row.get("label_ja") or ""),
            "label_en": str(row.get("label_en") or ""),
            "definition_ja": str(row.get("definition_ja") or ""),
            "definition_en": str(row.get("definition_en") or ""),
            "includes": list(row.get("includes") or []),
            "excludes": list(row.get("excludes") or []),
        }
        for row in alternatives
        if isinstance(row, Mapping)
    ]
    anchor_ids = [card["id"] for card in cards]
    if (
        not core_id
        or core_id == UNRESOLVED_ANCHOR_ID
        or not 29 <= len(cards) <= 59
        or core_id in anchor_ids
        or len(anchor_ids) != len(set(anchor_ids))
    ):
        raise ClassificationError("complement auditor cards are invalid")
    observed_digest = ollama.model_digests([model]).get(model, "")
    if observed_digest != expected_digest:
        raise ClassificationError("complement auditor model digest changed")
    response = ollama.chat(
        [
            {
                "role": "system",
                "content": (
                    "You audit retrieval coverage. The fixed core is valid but "
                    "does not automatically subsume document purpose or another "
                    "section-level subject. Add one complement only when it "
                    "creates an independent principal retrieval route."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "policy": POLICY,
                        "document_evidence": dict(page),
                        "extracted_subject": dict(subject),
                        "fixed_core_anchor": dict(core),
                        "alternative_anchor_cards": cards,
                        "instruction": (
                            "Test what a reader would lose if only the core "
                            "route existed. Return NONE for related context."
                        ),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            },
        ],
        model=model,
        format=_audit_schema(anchor_ids),
        num_ctx=16_384,
        num_predict=900,
        keep_alive=str(payload.get("keep_alive") or "20m"),
        read_timeout_ms=int(payload.get("read_timeout_ms") or 660_000),
        max_output_chars=8_000,
        temperature=0,
        seed=0,
        think=False,
    )
    try:
        raw = json.loads(str(response))
    except json.JSONDecodeError as exc:
        raise ClassificationError(
            "complement auditor returned malformed JSON"
        ) from exc
    if not isinstance(raw, Mapping):
        raise ClassificationError("complement auditor returned a non-object")
    result = validate_audit(
        raw,
        core_anchor_id=core_id,
        anchor_ids=anchor_ids,
    )
    return {
        "schema": WORKER_SCHEMA,
        "model": model,
        "model_digest": observed_digest,
        "prompt_sha256": PROMPT_SHA256,
        "model_calls": 1,
        "result": result,
    }


def main() -> int:
    try:
        payload = json.loads(sys.stdin.read())
        if not isinstance(payload, Mapping):
            raise ClassificationError(
                "complement audit payload must be an object"
            )
        print(json.dumps(run(payload), ensure_ascii=False, sort_keys=True))
        return 0
    except (
        ClassificationError,
        OSError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
