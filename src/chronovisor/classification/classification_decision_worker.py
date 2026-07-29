"""Isolated local-model worker for candidate-bounded UDC decisions."""

from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from typing import Any

from chronovisor.classification.classification import ClassificationError
from chronovisor.core import ollama

WORKER_SCHEMA = "chronovisor.classification-decision-worker.v1"
DECISION_SCHEMA = "chronovisor.classification-candidate-decision.v1"
HOLD = "HOLD"
SUPPORT_VALUES = ("yes", "no", "uncertain")
EVIDENCE_VALUES = ("direct", "inferred", "none", "contradicted")
DECISION_POLICY = {
    "role": "candidate-bounded-library-classifier",
    "task": (
        "Evaluate every official UDC candidate independently against the document "
        "before choosing. A vivid shared word, named product, metaphor, or example "
        "is not enough. Mark support=yes only when the candidate describes the "
        "document's principal subject or a legitimate primary shelf. Use uncertain "
        "when evidence is real but insufficient for that specificity. After all "
        "independent assessments, assign one supported candidate only when its "
        "specificity is justified. Otherwise return HOLD. HOLD is required when no "
        "candidate fits, evidence supports only an absent broader class, or the "
        "remaining candidates cross incompatible principal classes. Never invent a "
        "notation and never use a candidate merely because it is the least bad."
    ),
    "input_fields": ["uid", "title", "summary", "excerpt", "official_candidates"],
    "forbidden_input_fields": [
        "expected_primary_notations",
        "gold_primary_notation",
        "gold_rationale",
        "case_number",
        "tags",
        "raw_keywords",
    ],
    "output_fields": [
        "assessments",
        "principal_class",
        "disposition",
        "selected_notation",
        "specificity_safe",
        "rationale",
    ],
}
DECISION_PROMPT_SHA256 = "sha256:" + hashlib.sha256(
    json.dumps(
        DECISION_POLICY,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()


def _format_schema(candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    choices = [str(row.get("notation") or "") for row in candidates]
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "assessments",
            "principal_class",
            "disposition",
            "selected_notation",
            "specificity_safe",
            "rationale",
        ],
        "properties": {
            "assessments": {
                "type": "array",
                "minItems": len(choices),
                "maxItems": len(choices),
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["notation", "support", "evidence", "reason"],
                    "properties": {
                        "notation": {"type": "string", "enum": choices},
                        "support": {
                            "type": "string",
                            "enum": list(SUPPORT_VALUES),
                        },
                        "evidence": {
                            "type": "string",
                            "enum": list(EVIDENCE_VALUES),
                        },
                        "reason": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": 160,
                        },
                    },
                },
            },
            "principal_class": {
                "type": "string",
                "enum": ["0", "1", "2", "3", "5", "6", "7", "8", "9"],
            },
            "disposition": {"type": "string", "enum": ["assign", "hold"]},
            "selected_notation": {
                "type": "string",
                "enum": [*choices, HOLD],
            },
            "specificity_safe": {"type": "boolean"},
            "rationale": {
                "type": "string",
                "minLength": 1,
                "maxLength": 280,
            },
        },
    }


def _fail_closed_decision(
    candidates: Sequence[Mapping[str, Any]],
    *,
    principal_class: str,
    rationale: str,
    invalid_reason: str,
    assessments: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    by_notation = {
        str(row.get("notation") or ""): row
        for row in assessments or []
        if isinstance(row, Mapping)
    }
    normalized = []
    for candidate in candidates:
        notation = str(candidate.get("notation") or "")
        raw = by_notation.get(notation, {})
        support = str(raw.get("support") or "uncertain")
        evidence = str(raw.get("evidence") or "none")
        normalized.append(
            {
                "notation": notation,
                "support": support if support in SUPPORT_VALUES else "uncertain",
                "evidence": evidence if evidence in EVIDENCE_VALUES else "none",
                "reason": str(raw.get("reason") or "No valid independent assessment.")[
                    :160
                ],
            }
        )
    return {
        "schema": DECISION_SCHEMA,
        "assessments": normalized,
        "principal_class": principal_class,
        "disposition": "hold",
        "selected_notation": HOLD,
        "specificity_safe": False,
        "rationale": rationale[:280],
        "invalid_reason": invalid_reason,
    }


def validate_decision(
    value: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Normalize a structured model result and fail closed on inconsistency."""

    choices = [str(row.get("notation") or "") for row in candidates]
    if not choices or any(not choice for choice in choices) or len(set(choices)) != len(
        choices
    ):
        raise ClassificationError("decision worker candidates are invalid")
    principal_class = str(value.get("principal_class") or "")
    if principal_class not in {"0", "1", "2", "3", "5", "6", "7", "8", "9"}:
        principal_class = "0"
    rationale = str(value.get("rationale") or "").strip()
    raw_assessments = value.get("assessments")
    if not isinstance(raw_assessments, list):
        return _fail_closed_decision(
            candidates,
            principal_class=principal_class,
            rationale=rationale or "Assessment list was invalid.",
            invalid_reason="invalid_assessment_list",
        )
    notation_counts: dict[str, int] = {}
    for row in raw_assessments:
        if isinstance(row, Mapping):
            notation = str(row.get("notation") or "")
            notation_counts[notation] = notation_counts.get(notation, 0) + 1
    if set(notation_counts) != set(choices) or any(
        count != 1 for count in notation_counts.values()
    ):
        return _fail_closed_decision(
            candidates,
            principal_class=principal_class,
            rationale=rationale or "Candidate coverage was incomplete.",
            invalid_reason="candidate_assessment_coverage_mismatch",
            assessments=raw_assessments,
        )
    by_notation = {
        str(row.get("notation") or ""): row
        for row in raw_assessments
        if isinstance(row, Mapping)
    }
    assessments = []
    for notation in choices:
        raw = by_notation[notation]
        support = str(raw.get("support") or "")
        evidence = str(raw.get("evidence") or "")
        reason = str(raw.get("reason") or "").strip()
        if (
            support not in SUPPORT_VALUES
            or evidence not in EVIDENCE_VALUES
            or not reason
        ):
            return _fail_closed_decision(
                candidates,
                principal_class=principal_class,
                rationale=rationale or "Candidate assessment was malformed.",
                invalid_reason="malformed_candidate_assessment",
                assessments=raw_assessments,
            )
        assessments.append(
            {
                "notation": notation,
                "support": support,
                "evidence": evidence,
                "reason": reason[:160],
            }
        )
    disposition = str(value.get("disposition") or "")
    selected = str(value.get("selected_notation") or "")
    specificity_safe = bool(value.get("specificity_safe"))
    if disposition == "hold":
        selected = HOLD
        specificity_safe = False
    elif disposition != "assign" or selected not in choices:
        return _fail_closed_decision(
            candidates,
            principal_class=principal_class,
            rationale=rationale or "Decision disposition was inconsistent.",
            invalid_reason="invalid_final_disposition",
            assessments=assessments,
        )
    if disposition == "assign":
        selected_assessment = next(
            row for row in assessments if row["notation"] == selected
        )
        if (
            selected_assessment["support"] != "yes"
            or selected_assessment["evidence"] not in {"direct", "inferred"}
            or not specificity_safe
        ):
            return _fail_closed_decision(
                candidates,
                principal_class=principal_class,
                rationale=rationale or "Selected candidate lacked positive evidence.",
                invalid_reason="selected_candidate_not_supported",
                assessments=assessments,
            )
        if selected[:1] != principal_class:
            return _fail_closed_decision(
                candidates,
                principal_class=principal_class,
                rationale=rationale or "Principal class vetoed the selected candidate.",
                invalid_reason="principal_class_veto",
                assessments=assessments,
            )
    return {
        "schema": DECISION_SCHEMA,
        "assessments": assessments,
        "principal_class": principal_class,
        "disposition": disposition,
        "selected_notation": selected,
        "specificity_safe": specificity_safe,
        "rationale": rationale[:280],
        "invalid_reason": "",
    }


def run(payload: Mapping[str, Any]) -> dict[str, Any]:
    if payload.get("schema") != WORKER_SCHEMA:
        raise ClassificationError("unsupported decision worker schema")
    model = str(payload.get("model") or "")
    expected_digest = str(payload.get("model_digest") or "")
    page = payload.get("page")
    candidates = payload.get("candidates")
    if (
        not model
        or not expected_digest
        or not isinstance(page, Mapping)
        or not isinstance(candidates, list)
        or not 1 <= len(candidates) <= 12
    ):
        raise ClassificationError("decision worker input is incomplete")
    if set(page) - {"uid", "title", "summary", "excerpt"}:
        raise ClassificationError("decision worker page contains forbidden fields")
    uid = str(page.get("uid") or "")
    title = str(page.get("title") or "")
    excerpt = str(page.get("excerpt") or "")
    if not uid or not title or not excerpt:
        raise ClassificationError("decision worker requires uid, title and excerpt")
    candidate_cards = [
        {
            "notation": str(row.get("notation") or ""),
            "label_en": str(row.get("label_en") or ""),
            "label_ja": str(row.get("label_ja") or ""),
        }
        for row in candidates
        if isinstance(row, Mapping)
    ]
    if len(candidate_cards) != len(candidates):
        raise ClassificationError("decision worker candidate card is invalid")
    observed_digest = ollama.model_digests([model]).get(model, "")
    if not observed_digest or observed_digest != expected_digest:
        raise ClassificationError("decision worker model digest changed")
    model_input = {
        "policy": DECISION_POLICY,
        "document": dict(page),
        "official_candidates": candidate_cards,
    }
    response = ollama.chat(
        [
            {
                "role": "system",
                "content": (
                    "You are a conservative professional library classifier. "
                    "Return only schema-valid JSON. Judge each candidate independently "
                    "before making one bounded final decision. False lexical friends "
                    "must be rejected. HOLD is a valid and preferred result when the "
                    "official candidates do not support the document's primary shelf."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    model_input,
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            },
        ],
        model=model,
        format=_format_schema(candidate_cards),
        num_ctx=16_384,
        num_predict=1_900,
        keep_alive=str(payload.get("keep_alive") or "20m"),
        read_timeout_ms=int(payload.get("read_timeout_ms") or 660_000),
        max_output_chars=18_000,
        temperature=0,
        seed=0,
        think=False,
    )
    try:
        raw_decision = json.loads(str(response))
    except json.JSONDecodeError as exc:
        raise ClassificationError("decision worker returned malformed JSON") from exc
    if not isinstance(raw_decision, Mapping):
        raise ClassificationError("decision worker returned a non-object")
    decision = validate_decision(raw_decision, candidate_cards)
    return {
        "schema": WORKER_SCHEMA,
        "uid": uid,
        "model": model,
        "model_digest": observed_digest,
        "prompt_sha256": DECISION_PROMPT_SHA256,
        "model_calls": 1,
        "decision": decision,
    }


def main() -> int:
    try:
        payload = json.loads(sys.stdin.read())
        if not isinstance(payload, Mapping):
            raise ClassificationError("decision worker payload must be an object")
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
