"""Isolated local-model worker for role-separated subject query generation."""

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

WORKER_SCHEMA = "chronovisor.classification-query2doc-worker.v2"
QUERY_SCHEMA = "chronovisor.classification-subject-query.v2"
RUNTIME_ROLE = "classification.query_v2"
HEADING_ROLES = ("principal_shelf", "problem_or_activity", "context")
QUERY_POLICY = {
    "role": "professional-library-subject-indexer",
    "task": (
        "Create two complementary views. First create two or three broad bilingual "
        "library subject headings for the document as a whole. Then create exactly "
        "three independent bilingual role headings. The first role heading "
        "must name the principal library shelf: what the document as a whole is "
        "about, not a vivid title word, implementation detail, metaphor, product "
        "name, or example. The second must name the problem, activity, or decision "
        "being discussed. The third may name a genuine technical or social context. "
        "Do not concatenate the headings. Distinguish human memory, reflective "
        "journals, and knowledge management from computer storage; distinguish "
        "model quantization and inference from memory hardware; distinguish "
        "employment, interviewing, remote work, purchasing, and payment from tools "
        "or industries merely mentioned. A named entity that is the actual subject "
        "must remain in a heading. Put a term in surface_terms_to_ignore only when "
        "its literal domain is incidental or misleading. Do not repeat such a term "
        "in a heading. Do not output classification codes, notations, candidate "
        "labels, or taxonomy identifiers."
    ),
    "heading_roles": list(HEADING_ROLES),
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
        "broad_headings_ja",
        "broad_headings_en",
        "headings",
        "surface_terms_to_ignore",
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
            "broad_headings_ja",
            "broad_headings_en",
            "headings",
            "surface_terms_to_ignore",
            "evidence_basis",
        ],
        "properties": {
            "broad_headings_ja": {
                "type": "array",
                "minItems": 2,
                "maxItems": 3,
                "items": {"type": "string", "minLength": 1, "maxLength": 80},
            },
            "broad_headings_en": {
                "type": "array",
                "minItems": 2,
                "maxItems": 3,
                "items": {"type": "string", "minLength": 1, "maxLength": 80},
            },
            "headings": {
                "type": "array",
                "minItems": 3,
                "maxItems": 3,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["role", "ja", "en"],
                    "properties": {
                        "role": {"type": "string", "enum": list(HEADING_ROLES)},
                        "ja": {"type": "string", "minLength": 1, "maxLength": 80},
                        "en": {"type": "string", "minLength": 1, "maxLength": 80},
                    },
                },
            },
            "surface_terms_to_ignore": {
                "type": "array",
                "maxItems": 8,
                "items": {"type": "string", "minLength": 1, "maxLength": 50},
            },
            "evidence_basis": {
                "type": "string",
                "minLength": 1,
                "maxLength": 240,
            },
        },
    }


def _string_list(
    value: object,
    *,
    minimum: int = 0,
    maximum: int,
    field: str,
) -> list[str]:
    if not isinstance(value, list):
        raise ClassificationError(f"query2doc v2 {field} must be a list")
    output = []
    for item in value:
        text = str(item).strip()
        if text and text not in output:
            output.append(text)
    if not minimum <= len(output) <= maximum:
        raise ClassificationError(
            f"query2doc v2 {field} must contain "
            f"{minimum}..{maximum} unique values"
        )
    return output


def _headings(value: object) -> list[dict[str, str]]:
    if not isinstance(value, list) or len(value) != len(HEADING_ROLES):
        raise ClassificationError("query2doc v2 requires exactly three headings")
    by_role: dict[str, dict[str, str]] = {}
    for raw in value:
        if not isinstance(raw, Mapping):
            raise ClassificationError("query2doc v2 heading must be an object")
        role = str(raw.get("role") or "").strip()
        ja = str(raw.get("ja") or "").strip()
        en = str(raw.get("en") or "").strip()
        if role not in HEADING_ROLES or role in by_role or not ja or not en:
            raise ClassificationError("query2doc v2 heading contract mismatch")
        by_role[role] = {"role": role, "ja": ja, "en": en}
    if set(by_role) != set(HEADING_ROLES):
        raise ClassificationError("query2doc v2 heading roles are incomplete")
    return [by_role[role] for role in HEADING_ROLES]


def run(payload: Mapping[str, Any]) -> dict[str, Any]:
    if payload.get("schema") != WORKER_SCHEMA:
        raise ClassificationError("unsupported query2doc v2 worker schema")
    page = payload.get("page")
    if not isinstance(page, Mapping):
        raise ClassificationError("query2doc v2 worker input is incomplete")
    if set(page) - set(QUERY_POLICY["input_fields"]):
        raise ClassificationError("query2doc v2 page contains forbidden fields")
    uid = str(page.get("uid") or "")
    title = str(page.get("title") or "")
    excerpt = str(page.get("excerpt") or "")
    if not uid or not title or not excerpt:
        raise ClassificationError("query2doc v2 requires uid, title and excerpt")
    route, observed_digest, sensitivity = resolve_structured_route(
        payload, role=RUNTIME_ROLE
    )
    response = ollama.runtime_structured_chat(
        [
            {
                "role": "system",
                "content": (
                    "You are a professional library subject indexer. Return only "
                    "schema-valid JSON. Produce both broad whole-document headings "
                    "and three independent role headings. Separate the principal "
                    "shelf, the discussed problem or activity, and the genuine "
                    "context. Resist vivid surface words and title poisoning."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {"policy": QUERY_POLICY, "page": dict(page)},
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
        num_predict=640,
        keep_alive=str(payload.get("keep_alive") or "20m"),
        read_timeout_ms=int(payload.get("read_timeout_ms") or 660_000),
        max_output_chars=5_000,
        temperature=0,
        seed=0,
        think=False,
    )
    try:
        raw_query = json.loads(response.content)
    except json.JSONDecodeError as exc:
        raise ClassificationError(
            "query2doc v2 model returned malformed JSON"
        ) from exc
    if not isinstance(raw_query, Mapping):
        raise ClassificationError("query2doc v2 model returned a non-object")
    evidence_basis = str(raw_query.get("evidence_basis") or "").strip()
    if not evidence_basis or len(evidence_basis) > 240:
        raise ClassificationError("query2doc v2 evidence_basis is invalid")
    broad_ja = _string_list(
        raw_query.get("broad_headings_ja"),
        minimum=2,
        maximum=3,
        field="broad_headings_ja",
    )
    broad_en = _string_list(
        raw_query.get("broad_headings_en"),
        minimum=2,
        maximum=3,
        field="broad_headings_en",
    )
    headings = _headings(raw_query.get("headings"))
    ignored = _string_list(
        raw_query.get("surface_terms_to_ignore"),
        maximum=8,
        field="surface_terms_to_ignore",
    )
    folded_headings = " ".join(
        [*broad_ja, *broad_en]
        + [f"{heading['ja']} {heading['en']}" for heading in headings]
    ).casefold()
    ignored = [term for term in ignored if term.casefold() not in folded_headings]
    query = {
        "schema": QUERY_SCHEMA,
        "broad_headings_ja": broad_ja,
        "broad_headings_en": broad_en,
        "headings": headings,
        "surface_terms_to_ignore": ignored,
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


def main() -> int:
    try:
        payload = json.loads(sys.stdin.read())
        if not isinstance(payload, Mapping):
            raise ClassificationError("query2doc v2 worker payload must be an object")
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
