"""Independent local-model audit for admitting a second co-primary CVO anchor."""

from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from typing import Any

from chronovisor.core import ollama
from chronovisor.classification.classification import ClassificationError
from chronovisor.classification.classification_anchor import UNRESOLVED_ANCHOR_ID

WORKER_SCHEMA = "chronovisor.classification-anchor-second-auditor.v1"
AUDIT_SCHEMA = "chronovisor.classification-anchor-second-audit.v1"
NONE = "NONE"
POLICY = {
    "contract_version": 1,
    "purpose": "audit whether a fixed core anchor needs one co-primary peer",
    "presumption": "one core anchor is sufficient",
    "admission_axes": [
        "independent_principal_subject",
        "not_subsumed_by_core",
        "not_incidental_context",
        "explicit_document_evidence",
    ],
    "rules": [
        "Return NONE unless all four admission axes are true.",
        "A tool, employer, implementation detail, example, motivation, or setting is incidental.",
        "A specialized method inside the core domain is subsumed, not co-primary.",
        "A second anchor needs its own repeated claims, heading, or explicit retrieval intent.",
        "Uncertainty between two labels is not evidence for two subjects.",
    ],
    "counterexamples": [
        "An interview for an aerospace employer remains career unless aerospace engineering is independently analyzed.",
        "An automotive component standard remains automotive when welding or material is only the implementation method.",
        "A sports record remains sports when personal context is incidental.",
        "Software using an AI model remains software unless model capability itself is independently analyzed.",
    ],
    "forbidden_inputs": [
        "target_anchor_ids",
        "defensible_anchor_ids",
        "gold_rationale",
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
            "independent_principal_subject",
            "not_subsumed_by_core",
            "not_incidental_context",
            "explicit_document_evidence",
            "rationale",
        ],
        "properties": {
            "second_anchor_id": {
                "type": "string",
                "enum": [NONE, *anchor_ids],
            },
            "independent_principal_subject": {"type": "boolean"},
            "not_subsumed_by_core": {"type": "boolean"},
            "not_incidental_context": {"type": "boolean"},
            "explicit_document_evidence": {"type": "boolean"},
            "rationale": {"type": "string", "minLength": 1, "maxLength": 420},
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
        "independent_principal_subject": bool(
            value.get("independent_principal_subject")
        ),
        "not_subsumed_by_core": bool(value.get("not_subsumed_by_core")),
        "not_incidental_context": bool(value.get("not_incidental_context")),
        "explicit_document_evidence": bool(
            value.get("explicit_document_evidence")
        ),
    }
    rationale = str(value.get("rationale") or "").strip()[:420]
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
        raise ClassificationError("unsupported second-anchor auditor schema")
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
        raise ClassificationError("second-anchor auditor input is incomplete")
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
        raise ClassificationError("second-anchor auditor cards are invalid")
    observed_digest = ollama.model_digests([model]).get(model, "")
    if observed_digest != expected_digest:
        raise ClassificationError("second-anchor auditor model digest changed")
    response = ollama.chat(
        [
            {
                "role": "system",
                "content": (
                    "You are a conservative independent auditor. The core anchor "
                    "is already fixed. Presume it is sufficient. Admit a second "
                    "anchor only when every boolean admission condition is true."
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
                            "Audit necessity, not plausibility. Return NONE when "
                            "the alternative is merely related or contextual."
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
        num_predict=800,
        keep_alive=str(payload.get("keep_alive") or "20m"),
        read_timeout_ms=int(payload.get("read_timeout_ms") or 660_000),
        max_output_chars=7_000,
        temperature=0,
        seed=0,
        think=False,
    )
    try:
        raw = json.loads(str(response))
    except json.JSONDecodeError as exc:
        raise ClassificationError(
            "second-anchor auditor returned malformed JSON"
        ) from exc
    if not isinstance(raw, Mapping):
        raise ClassificationError("second-anchor auditor returned a non-object")
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
            raise ClassificationError("second-anchor audit payload must be an object")
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
