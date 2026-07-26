"""Isolated local-model worker for flat CVO anchor selection."""

from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from typing import Any

from chronovisor import ollama
from chronovisor.classification import ClassificationError
from chronovisor.classification_anchor import UNRESOLVED_ANCHOR_ID

WORKER_SCHEMA = "chronovisor.classification-anchor-worker.v1"
SUBJECT_SCHEMA = "chronovisor.classification-anchor-subject.v1"
SELECTION_SCHEMA = "chronovisor.classification-anchor-selection.v1"
POLICY = {
    "contract_version": 1,
    "purpose": "select a Chronovisor operational anchor without direct UDC mapping",
    "rules": [
        "Classify the principal subject, not a literal product name or metaphor.",
        "Use modern bilingual anchor definitions, inclusions and exclusions.",
        "Choose exactly one primary anchor.",
        "Choose at most one secondary anchor only for a genuine second subject.",
        "Use unresolved only when no existing anchor contains the subject.",
    ],
    "forbidden_inputs": [
        "expected_primary_anchor_ids",
        "expected_primary_notations",
        "gold_rationale",
        "udc_scope",
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


def _selection_schema(anchor_ids: Sequence[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["primary_anchor_id", "secondary_anchor_ids", "rationale"],
        "properties": {
            "primary_anchor_id": {"type": "string", "enum": list(anchor_ids)},
            "secondary_anchor_ids": {
                "type": "array",
                "maxItems": 1,
                "uniqueItems": True,
                "items": {"type": "string", "enum": list(anchor_ids)},
            },
            "rationale": {"type": "string", "minLength": 1, "maxLength": 320},
        },
    }


def validate_subject(value: Mapping[str, Any]) -> dict[str, Any]:
    central = str(value.get("central_subject") or "").strip()
    rationale = str(value.get("rationale") or "").strip()
    raw_secondary = value.get("secondary_subjects")
    secondary = [
        str(item).strip()[:120]
        for item in raw_secondary
        if str(item).strip()
    ] if isinstance(raw_secondary, list) else []
    if not central or not rationale:
        raise ClassificationError("CVO subject extraction is incomplete")
    return {
        "schema": SUBJECT_SCHEMA,
        "central_subject": central[:180],
        "secondary_subjects": secondary[:3],
        "rationale": rationale[:320],
    }


def validate_selection(
    value: Mapping[str, Any],
    anchor_ids: Sequence[str],
) -> dict[str, Any]:
    primary = str(value.get("primary_anchor_id") or "")
    rationale = str(value.get("rationale") or "").strip()[:320]
    raw_secondary = value.get("secondary_anchor_ids")
    secondary = [
        str(item)
        for item in raw_secondary
        if str(item) in anchor_ids and str(item) != primary
    ] if isinstance(raw_secondary, list) else []
    secondary = list(dict.fromkeys(secondary))[:1]
    invalid_reason = ""
    if primary not in anchor_ids:
        primary = UNRESOLVED_ANCHOR_ID
        secondary = []
        invalid_reason = "primary_outside_anchor_set"
    elif primary == UNRESOLVED_ANCHOR_ID:
        secondary = []
    if not rationale:
        primary = UNRESOLVED_ANCHOR_ID
        secondary = []
        invalid_reason = invalid_reason or "missing_rationale"
    return {
        "schema": SELECTION_SCHEMA,
        "primary_anchor_id": primary,
        "secondary_anchor_ids": secondary,
        "rationale": rationale,
        "invalid_reason": invalid_reason,
    }


def _chat(
    *,
    model: str,
    expected_digest: str,
    content: Mapping[str, Any],
    schema: Mapping[str, Any],
    keep_alive: str,
    read_timeout_ms: int,
) -> tuple[Mapping[str, Any], str]:
    observed_digest = ollama.model_digests([model]).get(model, "")
    if observed_digest != expected_digest:
        raise ClassificationError("CVO anchor worker model digest changed")
    response = ollama.chat(
        [
            {
                "role": "system",
                "content": (
                    "You are a multilingual classifier for a personal knowledge "
                    "system. Return only schema-valid JSON. Prefer the document's "
                    "principal operational domain over incidental names, examples "
                    "or implementation details."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(content, ensure_ascii=False, sort_keys=True),
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
        raise ClassificationError("CVO anchor worker returned malformed JSON") from exc
    if not isinstance(value, Mapping):
        raise ClassificationError("CVO anchor worker returned a non-object")
    return value, observed_digest


def run(payload: Mapping[str, Any]) -> dict[str, Any]:
    if payload.get("schema") != WORKER_SCHEMA:
        raise ClassificationError("unsupported CVO anchor worker schema")
    operation = str(payload.get("operation") or "")
    model = str(payload.get("model") or "")
    model_digest = str(payload.get("model_digest") or "")
    page = payload.get("page")
    if (
        operation not in {"extract", "classify"}
        or not model
        or not model_digest
        or not isinstance(page, Mapping)
        or set(page) - {"title", "summary", "evidence_excerpt"}
    ):
        raise ClassificationError("CVO anchor worker input is incomplete")
    common = {"policy": POLICY, "document_evidence": dict(page)}
    if operation == "extract":
        raw, observed_digest = _chat(
            model=model,
            expected_digest=model_digest,
            content={
                **common,
                "instruction": (
                    "State the principal library subject independently of anchor "
                    "labels. Separate genuine secondary subjects from examples."
                ),
            },
            schema=_subject_schema(),
            keep_alive=str(payload.get("keep_alive") or "20m"),
            read_timeout_ms=int(payload.get("read_timeout_ms") or 660_000),
        )
        result = validate_subject(raw)
    else:
        subject = payload.get("subject")
        anchors = payload.get("anchors")
        if not isinstance(subject, Mapping) or not isinstance(anchors, list):
            raise ClassificationError("CVO anchor classification input is incomplete")
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
            for row in anchors
            if isinstance(row, Mapping)
        ]
        anchor_ids = [card["id"] for card in cards]
        if (
            not 30 <= len(cards) <= 60
            or len(anchor_ids) != len(set(anchor_ids))
            or UNRESOLVED_ANCHOR_ID not in anchor_ids
        ):
            raise ClassificationError("CVO anchor cards are invalid")
        raw, observed_digest = _chat(
            model=model,
            expected_digest=model_digest,
            content={
                **common,
                "extracted_subject": dict(subject),
                "anchor_cards": cards,
                "instruction": (
                    "Select the one operational anchor that best contains the "
                    "principal subject. Add one secondary anchor only when the "
                    "document has a genuine second subject. Do not use UDC."
                ),
            },
            schema=_selection_schema(anchor_ids),
            keep_alive=str(payload.get("keep_alive") or "20m"),
            read_timeout_ms=int(payload.get("read_timeout_ms") or 660_000),
        )
        result = validate_selection(raw, anchor_ids)
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
            raise ClassificationError("CVO anchor payload must be an object")
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
