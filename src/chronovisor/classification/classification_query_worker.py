"""Isolated local-model worker for candidate-blind subject query generation."""

from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Mapping
from typing import Any

from chronovisor.core import ollama
from chronovisor.recall.classification import (
    ClassificationError,
    resolve_structured_route,
    route_identity,
)

WORKER_SCHEMA = "chronovisor.classification-query2doc-worker.v1"
QUERY_SCHEMA = "chronovisor.classification-subject-query.v1"
RUNTIME_ROLE = "classification.query"
QUERY_POLICY = {
    "role": "professional-library-subject-indexer",
    "task": (
        "Identify what the document is principally about at a conceptual subject "
        "level. Ignore incidental named entities, metaphors, examples, interface "
        "words, and literal words whose domain differs from the document's actual "
        "subject. Produce two or three concise library subject headings in both "
        "Japanese and English. Do not output any classification code, notation, "
        "candidate label, or taxonomy identifier."
    ),
    "input_fields": ["uid", "title", "summary", "excerpt"],
    "forbidden_input_fields": [
        "candidates",
        "expected_primary_notations",
        "gold_primary_notation",
        "case_number",
        "tags",
        "raw_keywords",
    ],
    "output_fields": [
        "subject_headings_ja",
        "subject_headings_en",
        "literal_terms_to_ignore",
        "evidence_basis",
    ],
}
QUERY_PROMPT_SHA256 = "sha256:" + hashlib.sha256(
    json.dumps(
        QUERY_POLICY,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()


def _format_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "subject_headings_ja",
            "subject_headings_en",
            "literal_terms_to_ignore",
            "evidence_basis",
        ],
        "properties": {
            "subject_headings_ja": {
                "type": "array",
                "minItems": 2,
                "maxItems": 3,
                "items": {"type": "string", "minLength": 1, "maxLength": 80},
            },
            "subject_headings_en": {
                "type": "array",
                "minItems": 2,
                "maxItems": 3,
                "items": {"type": "string", "minLength": 1, "maxLength": 80},
            },
            "literal_terms_to_ignore": {
                "type": "array",
                "maxItems": 8,
                "items": {"type": "string", "minLength": 1, "maxLength": 50},
            },
            "evidence_basis": {
                "type": "string",
                "minLength": 1,
                "maxLength": 200,
            },
        },
    }


def _string_list(
    value: object,
    *,
    minimum: int,
    maximum: int,
    field: str,
) -> list[str]:
    if not isinstance(value, list):
        raise ClassificationError(f"query2doc {field} must be a list")
    output = []
    for item in value:
        text = str(item).strip()
        if text and text not in output:
            output.append(text)
    if not minimum <= len(output) <= maximum:
        raise ClassificationError(
            f"query2doc {field} must contain {minimum}..{maximum} unique values"
        )
    return output


def run(payload: Mapping[str, Any]) -> dict[str, Any]:
    if payload.get("schema") != WORKER_SCHEMA:
        raise ClassificationError("unsupported query2doc worker schema")
    page = payload.get("page")
    if not isinstance(page, Mapping):
        raise ClassificationError("query2doc worker input is incomplete")
    if set(page) - set(QUERY_POLICY["input_fields"]):
        raise ClassificationError("query2doc page contains forbidden fields")
    uid = str(page.get("uid") or "")
    title = str(page.get("title") or "")
    excerpt = str(page.get("excerpt") or "")
    if not uid or not title or not excerpt:
        raise ClassificationError("query2doc page requires uid, title and excerpt")
    route, observed_digest, sensitivity = resolve_structured_route(
        payload, role=RUNTIME_ROLE
    )
    response = ollama.runtime_structured_chat(
        [
            {
                "role": "system",
                "content": (
                    "You are a professional library subject indexer. Return only "
                    "schema-valid JSON. Determine the conceptual subject rather "
                    "than copying vivid surface words."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "policy": QUERY_POLICY,
                        "page": dict(page),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            },
        ],
        runtime_role=route.role,
        source_data_class="page",
        source_sensitivity=sensitivity,
        format=_format_schema(),
        num_ctx=16_384,
        num_predict=512,
        keep_alive=str(payload.get("keep_alive") or "20m"),
        read_timeout_ms=int(payload.get("read_timeout_ms") or 660_000),
        max_output_chars=4_000,
        temperature=0,
        seed=0,
        think=False,
    )
    try:
        raw_query = json.loads(response.content)
    except json.JSONDecodeError as exc:
        raise ClassificationError("query2doc model returned malformed JSON") from exc
    if not isinstance(raw_query, Mapping):
        raise ClassificationError("query2doc model returned a non-object")
    evidence_basis = str(raw_query.get("evidence_basis") or "").strip()
    if not evidence_basis or len(evidence_basis) > 200:
        raise ClassificationError("query2doc evidence_basis is invalid")
    query = {
        "schema": QUERY_SCHEMA,
        "subject_headings_ja": _string_list(
            raw_query.get("subject_headings_ja"),
            minimum=2,
            maximum=3,
            field="subject_headings_ja",
        ),
        "subject_headings_en": _string_list(
            raw_query.get("subject_headings_en"),
            minimum=2,
            maximum=3,
            field="subject_headings_en",
        ),
        "literal_terms_to_ignore": _string_list(
            raw_query.get("literal_terms_to_ignore"),
            minimum=0,
            maximum=8,
            field="literal_terms_to_ignore",
        ),
        "evidence_basis": evidence_basis,
    }
    return {
        "schema": WORKER_SCHEMA,
        "uid": uid,
        "model": route.model,
        "model_digest": observed_digest,
        "route_identity": route_identity(route),
        "prompt_sha256": QUERY_PROMPT_SHA256,
        "model_calls": 1,
        "query": query,
    }


def main() -> None:
    payload = json.load(sys.stdin)
    print(json.dumps(run(payload), ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
