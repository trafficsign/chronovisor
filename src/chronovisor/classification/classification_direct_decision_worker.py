"""Isolated direct-choice worker for qualified UDC candidate sets."""

from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from typing import Any

from chronovisor.core import ollama
from chronovisor.classification.classification import ClassificationError

WORKER_SCHEMA = "chronovisor.classification-direct-decision-worker.v2"
DECISION_SCHEMA = "chronovisor.classification-direct-decision.v2"
HOLD = "HOLD"
DIRECT_POLICY = {
    "role": "professional-library-classifier",
    "task": (
        "Place the document on the most specific appropriate shelf among the official "
        "candidates. UDC captions are inclusive shelf names: a document about one "
        "specific vehicle part belongs under Vehicle structure; a document about one "
        "software refactoring belongs under Software engineering. Do not reject a "
        "correct class merely because the document is more specific than its caption. "
        "First state the document's central subject without copying a candidate label. "
        "Then choose the most specific candidate that contains that subject. Reject "
        "incidental, metaphorical, homonymous, or example-only matches. Return HOLD "
        "only when none of the candidates contains the principal subject, or when two "
        "incompatible principal classes remain genuinely unresolved."
    ),
    "input_fields": [
        "uid",
        "title",
        "summary",
        "excerpt",
        "candidate_blind_subject_headings",
        "official_candidates",
    ],
    "forbidden_input_fields": [
        "expected_primary_notations",
        "gold_rationale",
        "gold_ambiguity",
        "case_number",
        "tags",
        "raw_keywords",
        "retrieval_scores",
    ],
}
DIRECT_PROMPT_SHA256 = "sha256:" + hashlib.sha256(
    json.dumps(
        DIRECT_POLICY,
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
            "central_subject",
            "principal_class",
            "disposition",
            "selected_notation",
            "rationale",
        ],
        "properties": {
            "central_subject": {
                "type": "string",
                "minLength": 1,
                "maxLength": 180,
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
            "rationale": {
                "type": "string",
                "minLength": 1,
                "maxLength": 320,
            },
        },
    }


def validate_direct_decision(
    value: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    choices = [str(row.get("notation") or "") for row in candidates]
    if not choices or len(choices) != len(set(choices)):
        raise ClassificationError("direct decision candidates are invalid")
    central_subject = str(value.get("central_subject") or "").strip()
    principal_class = str(value.get("principal_class") or "")
    disposition = str(value.get("disposition") or "")
    selected = str(value.get("selected_notation") or "")
    rationale = str(value.get("rationale") or "").strip()
    invalid_reason = ""
    if principal_class not in {"0", "1", "2", "3", "5", "6", "7", "8", "9"}:
        invalid_reason = "invalid_principal_class"
    elif disposition == "hold":
        selected = HOLD
    elif disposition != "assign" or selected not in choices:
        invalid_reason = "notation_outside_candidates"
    elif selected[:1] != principal_class:
        invalid_reason = "principal_class_veto"
    if not central_subject or not rationale:
        invalid_reason = invalid_reason or "missing_explanation"
    if invalid_reason:
        disposition = "hold"
        selected = HOLD
    return {
        "schema": DECISION_SCHEMA,
        "central_subject": central_subject[:180],
        "principal_class": principal_class,
        "disposition": disposition,
        "selected_notation": selected,
        "rationale": rationale[:320],
        "invalid_reason": invalid_reason,
    }


def run(payload: Mapping[str, Any]) -> dict[str, Any]:
    if payload.get("schema") != WORKER_SCHEMA:
        raise ClassificationError("unsupported direct decision worker schema")
    model = str(payload.get("model") or "")
    expected_digest = str(payload.get("model_digest") or "")
    page = payload.get("page")
    subject_headings = payload.get("subject_headings")
    candidates = payload.get("candidates")
    if (
        not model
        or not expected_digest
        or not isinstance(page, Mapping)
        or not isinstance(subject_headings, list)
        or not isinstance(candidates, list)
        or not 1 <= len(candidates) <= 12
    ):
        raise ClassificationError("direct decision worker input is incomplete")
    if set(page) - {"uid", "title", "summary", "excerpt"}:
        raise ClassificationError("direct decision page contains forbidden fields")
    uid = str(page.get("uid") or "")
    if not uid or not str(page.get("title") or "") or not str(page.get("excerpt") or ""):
        raise ClassificationError("direct decision requires uid, title and excerpt")
    headings = [str(value).strip() for value in subject_headings if str(value).strip()]
    if not headings:
        raise ClassificationError("direct decision requires subject headings")
    cards = [
        {
            "notation": str(row.get("notation") or ""),
            "label_en": str(row.get("label_en") or ""),
            "label_ja": str(row.get("label_ja") or ""),
        }
        for row in candidates
        if isinstance(row, Mapping)
    ]
    if len(cards) != len(candidates):
        raise ClassificationError("direct decision candidate card is invalid")
    observed_digest = ollama.model_digests([model]).get(model, "")
    if observed_digest != expected_digest:
        raise ClassificationError("direct decision model digest changed")
    response = ollama.chat(
        [
            {
                "role": "system",
                "content": (
                    "You are a professional library classifier. Return only "
                    "schema-valid JSON. Classification shelves contain specific "
                    "documents; do not demand that a document discuss an entire "
                    "caption in general. Choose the best available official shelf. "
                    "Use HOLD only for a genuine candidate-set failure."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "policy": DIRECT_POLICY,
                        "document": dict(page),
                        "candidate_blind_subject_headings": headings,
                        "official_candidates": cards,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            },
        ],
        model=model,
        format=_format_schema(cards),
        num_ctx=16_384,
        num_predict=700,
        keep_alive=str(payload.get("keep_alive") or "20m"),
        read_timeout_ms=int(payload.get("read_timeout_ms") or 660_000),
        max_output_chars=6_000,
        temperature=0,
        seed=0,
        think=False,
    )
    try:
        raw = json.loads(str(response))
    except json.JSONDecodeError as exc:
        raise ClassificationError("direct decision returned malformed JSON") from exc
    if not isinstance(raw, Mapping):
        raise ClassificationError("direct decision returned a non-object")
    decision = validate_direct_decision(raw, cards)
    return {
        "schema": WORKER_SCHEMA,
        "uid": uid,
        "model": model,
        "model_digest": observed_digest,
        "prompt_sha256": DIRECT_PROMPT_SHA256,
        "model_calls": 1,
        "decision": decision,
    }


def main() -> int:
    try:
        payload = json.loads(sys.stdin.read())
        if not isinstance(payload, Mapping):
            raise ClassificationError("direct decision payload must be an object")
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
