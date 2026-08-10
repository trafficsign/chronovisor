"""Isolated fixed-role reviewer for collection-placement anomalies."""

from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from typing import Any

from chronovisor.core import ollama
from chronovisor.recall.collection_authority import CollectionAuthorityError

WORKER_SCHEMA = "chronovisor.collection-anomaly-worker.v2"
REVIEW_SCHEMA = "chronovisor.collection-anomaly-review.v2"
RUNTIME_ROLES = {
    "primary": "librarian.review",
    "challenger": "librarian.review.challenger",
}
_OVERRIDES = frozenset(
    {"model", "model_digest", "provider", "route_identity", "runtime_role"}
)
_FIELDS = frozenset(
    {
        "schema",
        "review_role",
        "source_data_class",
        "source_sensitivity",
        "read_timeout_ms",
        "candidate",
        "document",
        "collections",
        "review_input_sha256",
    }
)
POLICY = {
    "role": "review-only collection placement anomaly detector",
    "rules": [
        "Judge whether the current collection is semantically defensible.",
        "A recommendation never changes collection or page content.",
        "Prefer no_issue when the current collection is reasonably defensible.",
        "Use review_recommended only when the proposed collection is materially better.",
        "Use insufficient_evidence rather than guessing.",
    ],
    "mutation_capability": False,
}
PROMPT_SHA256 = (
    "sha256:"
    + hashlib.sha256(
        json.dumps(POLICY, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()
)


def review_input_sha256(
    candidate: Mapping[str, Any],
    document: Mapping[str, Any],
    collections: Sequence[Mapping[str, Any]],
    *,
    source_data_class: str,
    source_sensitivity: str,
) -> str:
    payload = {
        "candidate": dict(candidate),
        "document": dict(document),
        "collections": [dict(row) for row in collections],
        "source_data_class": source_data_class,
        "source_sensitivity": source_sensitivity,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def route_identity(route: ollama.RuntimeGenerationRoute) -> dict[str, str]:
    return {
        "role": route.role,
        "provider": route.provider,
        "model": route.model,
        "location": route.location,
    }


def route_model_digest(route: ollama.RuntimeGenerationRoute) -> str | None:
    if route.provider != "ollama" or route.location != "local":
        return None
    digest = ollama.model_digests([route.model]).get(route.model, "")
    if not digest:
        raise CollectionAuthorityError("anomaly reviewer model digest is unavailable")
    return digest


def _generation_profile(model: str) -> tuple[int, bool | str]:
    """Return a bounded profile that leaves room for model-family reasoning."""

    if model.partition(":")[0] == "gpt-oss":
        # gpt-oss needs a small explicit reasoning budget to reliably reach
        # the schema-constrained answer within this bounded review call.
        return 1_800, "low"
    return 700, False


def _schema(slugs: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "decision",
            "suggested_collection_slug",
            "rationale",
            "evidence",
        ],
        "properties": {
            "decision": {
                "type": "string",
                "enum": [
                    "no_issue",
                    "review_recommended",
                    "insufficient_evidence",
                ],
            },
            "suggested_collection_slug": {
                "type": "string",
                "enum": ["", *slugs],
            },
            "rationale": {"type": "string", "minLength": 1, "maxLength": 500},
            "evidence": {"type": "string", "minLength": 1, "maxLength": 300},
        },
    }


def run(payload: Mapping[str, Any]) -> dict[str, Any]:
    if _OVERRIDES.intersection(payload):
        raise CollectionAuthorityError("anomaly worker route overrides are forbidden")
    if set(payload) != _FIELDS:
        raise CollectionAuthorityError("anomaly worker payload fields are invalid")
    if payload.get("schema") != WORKER_SCHEMA:
        raise CollectionAuthorityError("unsupported anomaly worker schema")
    review_role = payload.get("review_role")
    source_data_class = payload.get("source_data_class")
    source_sensitivity = payload.get("source_sensitivity")
    read_timeout_ms = payload.get("read_timeout_ms")
    candidate = payload.get("candidate")
    document = payload.get("document")
    collections = payload.get("collections")
    if (
        review_role not in RUNTIME_ROLES
        or source_data_class not in {"page", "system"}
        or source_sensitivity not in {"normal", "high"}
        or not isinstance(read_timeout_ms, int)
        or isinstance(read_timeout_ms, bool)
        or read_timeout_ms < 1
        or not isinstance(candidate, Mapping)
        or not isinstance(document, Mapping)
        or not isinstance(collections, list)
        or not all(isinstance(row, Mapping) for row in collections)
    ):
        raise CollectionAuthorityError("anomaly worker input is incomplete")
    expected_input_sha256 = review_input_sha256(
        candidate,
        document,
        collections,
        source_data_class=str(source_data_class),
        source_sensitivity=str(source_sensitivity),
    )
    if payload.get("review_input_sha256") != expected_input_sha256:
        raise CollectionAuthorityError("anomaly worker input digest changed")
    runtime_role = RUNTIME_ROLES[str(review_role)]
    route = ollama.runtime_generation_routes((runtime_role,))[0]
    if not route.structured_output:
        raise CollectionAuthorityError("anomaly reviewer requires structured output")
    model_digest = route_model_digest(route)
    slugs = sorted(
        {
            str(row.get("slug") or "")
            for row in collections
            if str(row.get("slug") or "")
        }
    )
    num_predict, think = _generation_profile(route.model)
    response = ollama.runtime_structured_chat(
        [
            {
                "role": "system",
                "content": (
                    "You review suspicious collection placement in a personal "
                    "knowledge archive. Return schema-valid JSON only. You are "
                    "not an authority and cannot move or modify anything."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "policy": POLICY,
                        "candidate": dict(candidate),
                        "document": dict(document),
                        "available_collections": collections,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            },
        ],
        runtime_role=runtime_role,
        source_data_class=str(source_data_class),
        source_sensitivity=str(source_sensitivity),
        format=_schema(slugs),
        num_ctx=8_192,
        num_predict=num_predict,
        keep_alive="0",
        read_timeout_ms=read_timeout_ms,
        max_output_chars=6_000,
        temperature=0,
        seed=0,
        think=think,
    )
    try:
        raw = json.loads(response.content)
    except json.JSONDecodeError as exc:
        raise CollectionAuthorityError(
            "anomaly reviewer returned malformed JSON"
        ) from exc
    if not isinstance(raw, Mapping):
        raise CollectionAuthorityError("anomaly reviewer returned non-object")
    decision = str(raw.get("decision") or "")
    suggested = str(raw.get("suggested_collection_slug") or "")
    rationale = str(raw.get("rationale") or "").strip()[:500]
    evidence = str(raw.get("evidence") or "").strip()[:300]
    if (
        decision not in {"no_issue", "review_recommended", "insufficient_evidence"}
        or suggested not in {"", *slugs}
        or not rationale
        or not evidence
    ):
        raise CollectionAuthorityError("anomaly reviewer output is invalid")
    if decision != "review_recommended":
        suggested = ""
    return {
        "schema": WORKER_SCHEMA,
        "model": route.model,
        "route_identity": route_identity(route),
        "model_digest": model_digest,
        "prompt_sha256": PROMPT_SHA256,
        "review_input_sha256": expected_input_sha256,
        "source_data_class": source_data_class,
        "source_sensitivity": source_sensitivity,
        "model_calls": 1,
        "page_mutations": 0,
        "assignment_mutations": 0,
        "result": {
            "schema": REVIEW_SCHEMA,
            "decision": decision,
            "suggested_collection_slug": suggested,
            "rationale": rationale,
            "evidence": evidence,
        },
    }


def main() -> int:
    try:
        payload = json.loads(sys.stdin.read())
        if not isinstance(payload, Mapping):
            raise CollectionAuthorityError("anomaly worker input must be object")
        print(json.dumps(run(payload), ensure_ascii=False, sort_keys=True))
        return 0
    except (
        CollectionAuthorityError,
        OSError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
