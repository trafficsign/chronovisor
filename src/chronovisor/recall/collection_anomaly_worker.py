"""Isolated local-model reviewer for collection-placement anomalies."""

from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Mapping
from typing import Any

from chronovisor.core import ollama
from chronovisor.recall.collection_authority import CollectionAuthorityError

WORKER_SCHEMA = "chronovisor.collection-anomaly-worker.v1"
REVIEW_SCHEMA = "chronovisor.collection-anomaly-review.v1"
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
PROMPT_SHA256 = "sha256:" + hashlib.sha256(
    json.dumps(POLICY, ensure_ascii=False, sort_keys=True).encode()
).hexdigest()


def _generation_profile(model: str) -> tuple[int, bool | str]:
    """Return a bounded profile that leaves room for model-family reasoning."""

    if model.partition(":")[0] == "gpt-oss":
        # gpt-oss may consume the entire output budget in its hidden reasoning
        # channel when thinking is disabled. A small explicit reasoning budget
        # produces the schema-constrained answer reliably while remaining
        # bounded and local-only.
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
    if payload.get("schema") != WORKER_SCHEMA:
        raise CollectionAuthorityError("unsupported anomaly worker schema")
    model = str(payload.get("model") or "")
    expected_digest = str(payload.get("model_digest") or "")
    candidate = payload.get("candidate")
    document = payload.get("document")
    collections = payload.get("collections")
    if (
        not model
        or not expected_digest
        or not isinstance(candidate, Mapping)
        or not isinstance(document, Mapping)
        or not isinstance(collections, list)
    ):
        raise CollectionAuthorityError("anomaly worker input is incomplete")
    slugs = sorted(
        {
            str(row.get("slug") or "")
            for row in collections
            if isinstance(row, Mapping) and str(row.get("slug") or "")
        }
    )
    observed_digest = ollama.model_digests([model]).get(model, "")
    if observed_digest != expected_digest:
        raise CollectionAuthorityError("anomaly reviewer model digest changed")
    num_predict, think = _generation_profile(model)
    response = ollama.chat(
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
        model=model,
        format=_schema(slugs),
        num_ctx=8_192,
        num_predict=num_predict,
        keep_alive="0",
        read_timeout_ms=int(payload.get("read_timeout_ms") or 660_000),
        max_output_chars=6_000,
        temperature=0,
        seed=0,
        think=think,
    )
    try:
        raw = json.loads(str(response))
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
        decision
        not in {"no_issue", "review_recommended", "insufficient_evidence"}
        or suggested not in {"", *slugs}
        or not rationale
        or not evidence
    ):
        raise CollectionAuthorityError("anomaly reviewer output is invalid")
    if decision != "review_recommended":
        suggested = ""
    return {
        "schema": WORKER_SCHEMA,
        "model": model,
        "model_digest": observed_digest,
        "prompt_sha256": PROMPT_SHA256,
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
