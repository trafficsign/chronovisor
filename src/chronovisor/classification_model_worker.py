"""Isolated local-model worker for Librarian classification batches."""

from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from chronovisor import ollama
from chronovisor.classification import load_udc_package
from chronovisor.classification_engine import CONSENSUS_SCHEMA
from chronovisor.runtime_config import load_decision_router_config


def _schema(count: int) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["decisions"],
        "properties": {
            "decisions": {
                "type": "array",
                "minItems": count,
                "maxItems": count,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "uid",
                        "primary_notation",
                        "secondary_notations",
                        "confidence",
                        "rationale",
                    ],
                    "properties": {
                        "uid": {"type": "string"},
                        "primary_notation": {"type": "string"},
                        "secondary_notations": {
                            "type": "array",
                            "maxItems": 3,
                            "items": {"type": "string"},
                        },
                        "confidence": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                        },
                        "rationale": {"type": "string", "maxLength": 160},
                    },
                },
            }
        },
    }


def _page_prompt(page: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "uid": str(page["uid"]),
        "title": str(page.get("title") or ""),
        "summary": str(page.get("summary") or ""),
        "tags": [str(value) for value in page.get("tags") or []],
        "raw_keywords": [
            str(value) for value in (page.get("raw_keywords") or [])[:30]
        ],
        "excerpt": str(page.get("excerpt") or "")[:1_200],
        "allowed_candidates": [
            {
                "notation": str(candidate["notation"]),
                "label_en": str(candidate.get("label_en") or ""),
                "label_ja": str(candidate.get("label_ja") or ""),
                "broader_notation": str(candidate.get("broader_notation") or ""),
            }
            for candidate in page.get("candidates") or []
        ],
    }


def _call(
    *,
    model: str,
    keep_alive: str,
    pages: Sequence[Mapping[str, Any]],
    role: str,
    prior: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], int]:
    config = load_decision_router_config()
    prompt = {
        "task": (
            "Classify every page into the single best UDC Summary concept. "
            "The primary and all secondary notations MUST come from that page's "
            "allowed_candidates. Prefer the most specific concept supported by "
            "the text. Do not omit or reorder pages."
        ),
        "role": role,
        "pages": [_page_prompt(page) for page in pages],
    }
    if prior is not None:
        prompt["proposal_to_audit"] = list(prior)
        prompt["audit_instruction"] = (
            "Independently verify every proposal. Return your corrected complete "
            "decision list; do not merely approve or comment."
        )
    response = ollama.chat(
        [
            {
                "role": "system",
                "content": (
                    "You are a conservative professional librarian. Return only "
                    "schema-valid JSON. Never invent a UDC notation."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(prompt, ensure_ascii=False, sort_keys=True),
            },
        ],
        model=model,
        format=_schema(len(pages)),
        num_ctx=min(config.num_ctx, 65_536),
        num_predict=max(1_536, min(4_096, len(pages) * 110)),
        keep_alive=keep_alive,
        read_timeout_ms=config.read_timeout_ms,
        max_output_chars=max(16_000, len(pages) * 1_000),
        temperature=0,
        seed=0,
        think=False,
    )
    try:
        payload = json.loads(str(response))
    except json.JSONDecodeError:
        if len(pages) > 1:
            midpoint = len(pages) // 2
            prior_left = prior[:midpoint] if prior is not None else None
            prior_right = prior[midpoint:] if prior is not None else None
            left, left_calls = _call(
                model=model,
                keep_alive=keep_alive,
                pages=pages[:midpoint],
                role=role,
                prior=prior_left,
            )
            right, right_calls = _call(
                model=model,
                keep_alive=keep_alive,
                pages=pages[midpoint:],
                role=role,
                prior=prior_right,
            )
            return left + right, 1 + left_calls + right_calls
        page = pages[0]
        return (
            [
                {
                    "uid": str(page["uid"]),
                    "primary_notation": str(page["candidates"][0]["notation"]),
                    "secondary_notations": [],
                    "confidence": 0.0,
                    "rationale": f"{model} returned truncated JSON.",
                    "_invalid_reason": "model_json_truncated",
                }
            ],
            1,
        )
    decisions = payload.get("decisions")
    if not isinstance(decisions, list):
        raise TypeError(f"{model} returned no decision list")
    by_uid = {str(row.get("uid") or ""): dict(row) for row in decisions}
    ordered: list[dict[str, Any]] = []
    for page in pages:
        uid = str(page["uid"])
        decision = by_uid.get(uid)
        if decision is None:
            decision = {
                "uid": uid,
                "primary_notation": str(page["candidates"][0]["notation"]),
                "secondary_notations": [],
                "confidence": 0.0,
                "rationale": f"{model} omitted this page.",
                "_invalid_reason": "model_omitted_page",
            }
        allowed = {
            str(candidate["notation"]) for candidate in page.get("candidates") or []
        }
        primary = str(decision.get("primary_notation") or "")
        secondary = [str(value) for value in decision.get("secondary_notations") or []]
        if primary not in allowed or any(value not in allowed for value in secondary):
            decision["_invalid_reason"] = "notation_outside_host_candidates"
            decision["confidence"] = 0.0
        ordered.append(decision)
    return ordered, 1


def _decision_digest(
    uid: str,
    primary: Mapping[str, Any],
    challenger: Mapping[str, Any],
    tie: Mapping[str, Any] | None,
) -> str:
    payload = json.dumps(
        {
            "uid": uid,
            "primary": primary,
            "challenger": challenger,
            "tie": tie,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def run(payload: Mapping[str, Any]) -> dict[str, Any]:
    if payload.get("schema") != CONSENSUS_SCHEMA:
        raise ValueError("unsupported classification consensus schema")
    root = Path(str(payload["root"]))
    package = load_udc_package(root)
    if not package.complete:
        raise ValueError("local consensus requires a complete UDC package")
    pages = payload.get("pages")
    if not isinstance(pages, list) or not pages:
        raise ValueError("classification worker requires pages")
    config = load_decision_router_config()
    primary, primary_calls = _call(
        model=config.primary_model,
        keep_alive=config.primary_keep_alive,
        pages=pages,
        role="primary-proposer",
    )
    challenger, challenger_calls = _call(
        model=config.challenger_model,
        keep_alive=config.challenger_keep_alive,
        pages=pages,
        role="independent-challenger",
        prior=primary,
    )
    disagreements = [
        index
        for index, (left, right) in enumerate(zip(primary, challenger, strict=True))
        if left.get("_invalid_reason")
        or right.get("_invalid_reason")
        or str(left["primary_notation"]) != str(right["primary_notation"])
    ]
    tie_by_uid: dict[str, dict[str, Any]] = {}
    model_calls = primary_calls + challenger_calls
    if disagreements:
        tie_pages = [pages[index] for index in disagreements]
        paired_prior = [
            {
                "uid": str(pages[index]["uid"]),
                "primary_proposal": primary[index],
                "challenger_proposal": challenger[index],
            }
            for index in disagreements
        ]
        tie, tie_calls = _call(
            model=config.tie_break_model,
            keep_alive=config.tie_break_keep_alive,
            pages=tie_pages,
            role="tie-break-adjudicator",
            prior=paired_prior,
        )
        tie_by_uid = {str(row["uid"]): row for row in tie}
        model_calls += tie_calls
    decisions = []
    for page_index, (page, left, right) in enumerate(
        zip(pages, primary, challenger, strict=True)
    ):
        uid = str(page["uid"])
        tie = tie_by_uid.get(uid)
        left_valid = not left.get("_invalid_reason")
        right_valid = not right.get("_invalid_reason")
        tie_valid = tie is not None and not tie.get("_invalid_reason")
        if (
            left_valid
            and right_valid
            and left["primary_notation"] == right["primary_notation"]
        ):
            winner = left
            votes = 2
        elif tie_valid and (
            (left_valid and tie["primary_notation"] == left["primary_notation"])
            or (
                right_valid
                and tie["primary_notation"] == right["primary_notation"]
            )
        ):
            winner = tie
            votes = 2
        else:
            winner = (
                tie
                if tie_valid
                else left
                if left_valid
                else right
                if right_valid
                else {
                    "primary_notation": str(
                        pages[page_index]["candidates"][0]["notation"]
                    ),
                    "secondary_notations": [],
                    "confidence": 0.0,
                    "rationale": "No schema-valid two-model quorum.",
                }
            )
            votes = 1
        confidence = min(
            float(winner.get("confidence") or 0.0),
            0.99 if votes >= 2 else 0.49,
        )
        decisions.append(
            {
                "uid": uid,
                "primary_notation": str(winner["primary_notation"]),
                "secondary_notations": [
                    str(value)
                    for value in winner.get("secondary_notations") or []
                ][:3],
                "confidence": confidence,
                "rationale": str(winner.get("rationale") or "")[:400],
                "status": "proposed" if votes >= 2 else "held",
                "quorum": votes,
                "primary_model": config.primary_model,
                "challenger_model": config.challenger_model,
                "tie_break_model": config.tie_break_model if tie is not None else None,
                "consensus_sha256": _decision_digest(uid, left, right, tie),
            }
        )
    return {
        "schema": CONSENSUS_SCHEMA,
        "package_checksum": package.checksum,
        "model_calls": model_calls,
        "decisions": decisions,
    }


def main() -> None:
    payload = json.loads(sys.stdin.read())
    result = run(payload)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
