"""Isolated local-model worker for hierarchical UDC navigation."""

from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from typing import Any

from chronovisor import ollama
from chronovisor.classification import ClassificationError
from chronovisor.classification_hierarchy import ROOT_NOTATIONS

WORKER_SCHEMA = "chronovisor.classification-hierarchy-worker.v2"
EXTRACTION_SCHEMA = "chronovisor.classification-hierarchy-subject.v1"
STEP_SCHEMA = "chronovisor.classification-hierarchy-step.v1"
AUDIT_SCHEMA = "chronovisor.classification-hierarchy-audit.v1"
HOLD = "HOLD"
STEP_STOP = "__STOP_AT_PARENT__"
POLICY = {
    "contract_version": 2,
    "purpose": "top-down UDC navigation without semantic candidate retrieval",
    "principles": [
        "Classify the principal subject, not incidental words or metaphors.",
        "Treat each option as an inclusive library shelf.",
        "Use raw evidence to correct an imperfect extracted subject.",
        "At a child step, stop when the current parent is valid but no child is.",
        "Choose two branches only for genuine cross-disciplinary ambiguity.",
        "Audit may shorten explored paths but may never invent a sibling path.",
    ],
    "forbidden_inputs": [
        "expected_primary_notations",
        "gold_rationale",
        "retrieval_scores",
        "legacy_consensus",
    ],
}
PROMPT_SHA256 = "sha256:" + hashlib.sha256(
    json.dumps(POLICY, ensure_ascii=False, sort_keys=True).encode()
).hexdigest()


def _subject_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["central_subject", "secondary_subjects", "rationale"],
        "properties": {
            "central_subject": {"type": "string", "minLength": 1, "maxLength": 180},
            "secondary_subjects": {
                "type": "array",
                "maxItems": 3,
                "items": {"type": "string", "minLength": 1, "maxLength": 120},
            },
            "rationale": {"type": "string", "minLength": 1, "maxLength": 320},
        },
    }


def _step_schema(choices: Sequence[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["selected_notations", "corrected_central_subject", "rationale"],
        "properties": {
            "selected_notations": {
                "type": "array",
                "minItems": 1,
                "maxItems": 2,
                "uniqueItems": True,
                "items": {"type": "string", "enum": [*choices, STEP_STOP]},
            },
            "corrected_central_subject": {
                "type": "string",
                "maxLength": 180,
            },
            "rationale": {"type": "string", "minLength": 1, "maxLength": 320},
        },
    }


def _audit_schema(choices: Sequence[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["selected_notation", "rationale"],
        "properties": {
            "selected_notation": {
                "type": "string",
                "enum": [*choices, HOLD],
            },
            "rationale": {"type": "string", "minLength": 1, "maxLength": 320},
        },
    }


def validate_subject(value: Mapping[str, Any]) -> dict[str, Any]:
    central = str(value.get("central_subject") or "").strip()
    rationale = str(value.get("rationale") or "").strip()
    secondary_raw = value.get("secondary_subjects")
    secondary = [
        str(item).strip()[:120]
        for item in secondary_raw
        if str(item).strip()
    ] if isinstance(secondary_raw, list) else []
    if not central or not rationale:
        raise ClassificationError("hierarchy subject extraction is incomplete")
    return {
        "schema": EXTRACTION_SCHEMA,
        "central_subject": central[:180],
        "secondary_subjects": secondary[:3],
        "rationale": rationale[:320],
    }


def validate_step(
    value: Mapping[str, Any],
    choices: Sequence[str],
) -> dict[str, Any]:
    raw_selected = value.get("selected_notations")
    selected = [
        str(item)
        for item in raw_selected
        if str(item) in {*choices, STEP_STOP}
    ] if isinstance(raw_selected, list) else []
    selected = list(dict.fromkeys(selected))[:2]
    corrected = str(value.get("corrected_central_subject") or "").strip()[:180]
    rationale = str(value.get("rationale") or "").strip()[:320]
    invalid_reason = ""
    if STEP_STOP in selected:
        action = "stop"
        selected = []
    elif selected:
        action = "descend"
    else:
        invalid_reason = "invalid_navigation_action"
        action = "stop"
        selected = []
    if not rationale:
        invalid_reason = invalid_reason or "missing_rationale"
        action = "stop"
        selected = []
    return {
        "schema": STEP_SCHEMA,
        "action": action,
        "selected_notations": selected,
        "corrected_central_subject": corrected,
        "rationale": rationale,
        "invalid_reason": invalid_reason,
    }


def validate_audit(
    value: Mapping[str, Any],
    allowed_notations: Sequence[str],
) -> dict[str, Any]:
    selected = str(value.get("selected_notation") or "")
    rationale = str(value.get("rationale") or "").strip()[:320]
    invalid_reason = ""
    if selected in ROOT_NOTATIONS:
        selected = HOLD
        invalid_reason = "root_equivalent_hold"
    elif selected != HOLD and selected not in allowed_notations:
        selected = HOLD
        invalid_reason = "audit_left_explored_paths"
    if not rationale:
        selected = HOLD
        invalid_reason = invalid_reason or "missing_rationale"
    return {
        "schema": AUDIT_SCHEMA,
        "selected_notation": selected,
        "rationale": rationale,
        "invalid_reason": invalid_reason,
    }


def _chat(
    *,
    model: str,
    model_digest: str,
    expected_digest: str,
    payload: Mapping[str, Any],
    schema: Mapping[str, Any],
    keep_alive: str,
    read_timeout_ms: int,
) -> Mapping[str, Any]:
    if model_digest != expected_digest:
        raise ClassificationError("hierarchy worker model digest changed")
    response = ollama.chat(
        [
            {
                "role": "system",
                "content": (
                    "You are a professional multilingual library classifier. "
                    "Return only schema-valid JSON. Ignore incidental literal "
                    "terms, product names and metaphors. Use the document's "
                    "principal subject and the inclusive meaning of each shelf."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=False, sort_keys=True),
            },
        ],
        model=model,
        format=dict(schema),
        num_ctx=16_384,
        num_predict=700,
        keep_alive=keep_alive,
        read_timeout_ms=read_timeout_ms,
        max_output_chars=6_000,
        temperature=0,
        seed=0,
        think=False,
    )
    try:
        value = json.loads(str(response))
    except json.JSONDecodeError as exc:
        raise ClassificationError("hierarchy worker returned malformed JSON") from exc
    if not isinstance(value, Mapping):
        raise ClassificationError("hierarchy worker returned a non-object")
    return value


def run(payload: Mapping[str, Any]) -> dict[str, Any]:
    if payload.get("schema") != WORKER_SCHEMA:
        raise ClassificationError("unsupported hierarchy worker schema")
    operation = str(payload.get("operation") or "")
    model = str(payload.get("model") or "")
    expected_digest = str(payload.get("model_digest") or "")
    page = payload.get("page")
    if (
        operation not in {"extract", "navigate", "audit"}
        or not model
        or not expected_digest
        or not isinstance(page, Mapping)
        or set(page) - {"title", "summary", "evidence_excerpt"}
    ):
        raise ClassificationError("hierarchy worker input is incomplete")
    observed_digest = ollama.model_digests([model]).get(model, "")
    common = {
        "policy": POLICY,
        "operation": operation,
        "document_evidence": dict(page),
    }
    if operation == "extract":
        raw = _chat(
            model=model,
            model_digest=observed_digest,
            expected_digest=expected_digest,
            payload={
                **common,
                "instruction": (
                    "State the document's central library subject independently "
                    "of any classification label. Separate genuinely secondary "
                    "subjects from examples and implementation details."
                ),
            },
            schema=_subject_schema(),
            keep_alive=str(payload.get("keep_alive") or "20m"),
            read_timeout_ms=int(payload.get("read_timeout_ms") or 660_000),
        )
        result = validate_subject(raw)
    elif operation == "navigate":
        options = payload.get("options")
        if not isinstance(options, list) or not 1 <= len(options) <= 40:
            raise ClassificationError("hierarchy navigation options are invalid")
        cards = [
            {
                "notation": str(card.get("notation") or ""),
                "label_en": str(card.get("label_en") or ""),
                "label_ja": str(card.get("label_ja") or ""),
                "has_children": bool(card.get("has_children")),
                "is_range": bool(card.get("is_range")),
            }
            for card in options
            if isinstance(card, Mapping)
        ]
        choices = [card["notation"] for card in cards]
        if len(cards) != len(options) or len(choices) != len(set(choices)):
            raise ClassificationError("hierarchy navigation cards are invalid")
        raw = _chat(
            model=model,
            model_digest=observed_digest,
            expected_digest=expected_digest,
            payload={
                **common,
                "extracted_subject": payload.get("subject"),
                "current_path": payload.get("current_path") or [],
                "options": cards,
                "prior_attempts": payload.get("prior_attempts") or [],
                "instruction": (
                    "Return one offered notation, or at most two for genuine "
                    "cross-disciplinary ambiguity. Return __STOP_AT_PARENT__ only "
                    "when a non-root current parent is valid but no child is. At "
                    "root you must select an offered main class unless the document "
                    "has no classifiable subject. Correct the extracted subject if "
                    "the raw evidence disproves it."
                ),
            },
            schema=_step_schema(choices),
            keep_alive=str(payload.get("keep_alive") or "20m"),
            read_timeout_ms=int(payload.get("read_timeout_ms") or 660_000),
        )
        result = validate_step(raw, choices)
    else:
        paths = payload.get("explored_paths")
        allowed = [
            str(value)
            for value in payload.get("allowed_notations") or []
            if str(value)
        ]
        if not isinstance(paths, list) or not allowed:
            raise ClassificationError("hierarchy audit paths are invalid")
        raw = _chat(
            model=model,
            model_digest=observed_digest,
            expected_digest=expected_digest,
            payload={
                **common,
                "extracted_subject": payload.get("subject"),
                "explored_paths": paths,
                "allowed_notations": allowed,
                "root_attempts": payload.get("root_attempts") or [],
                "instruction": (
                    "Select the deepest justified notation on an explored path. "
                    "You may shorten toward an ancestor or return HOLD. Never jump "
                    "to an unexplored sibling. A root-class-only result is HOLD."
                ),
            },
            schema=_audit_schema(allowed),
            keep_alive=str(payload.get("keep_alive") or "20m"),
            read_timeout_ms=int(payload.get("read_timeout_ms") or 660_000),
        )
        result = validate_audit(raw, allowed)
    return {
        "schema": WORKER_SCHEMA,
        "operation": operation,
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
            raise ClassificationError("hierarchy worker payload must be an object")
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
