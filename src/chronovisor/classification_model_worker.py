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
from chronovisor.durable_state import (
    DurableStateError,
    verify_sealed_object,
    write_sealed_json,
)
from chronovisor.runtime_config import load_decision_router_config

STAGE_CACHE_SCHEMA = "chronovisor.classification-stage-cache.v2"
# A page's classification must not depend on unrelated pages sharing its batch.
STAGE_CHUNK_SIZE = 1


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
        if role == "tie-break-adjudicator":
            prompt["audit_instruction"] = (
                "Resolve each disagreement by choosing primary_notation strictly "
                "from primary_proposal.primary_notation or "
                "challenger_proposal.primary_notation. Return the complete "
                "decision list; a third primary notation is forbidden."
            )
        else:
            prompt["audit_instruction"] = (
                "Independently verify every proposal. Return your corrected "
                "complete decision list; do not merely approve or comment."
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
        if primary not in allowed:
            decision["_invalid_reason"] = "notation_outside_host_candidates"
            decision["confidence"] = 0.0
        else:
            rejected_secondary = [
                value for value in secondary if value not in allowed
            ]
            decision["secondary_notations"] = [
                value
                for value in secondary
                if value in allowed and value != primary
            ]
            if rejected_secondary:
                decision["_rejected_secondary_notations"] = rejected_secondary
        ordered.append(decision)
    return ordered, 1


def _tie_candidate_pages(
    pages: Sequence[Mapping[str, Any]],
    primary: Sequence[Mapping[str, Any]],
    challenger: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Limit tie-break primary choices to the two independently proposed classes."""

    output: list[dict[str, Any]] = []
    for page, left, right in zip(pages, primary, challenger, strict=True):
        choices = {
            str(left.get("primary_notation") or ""),
            str(right.get("primary_notation") or ""),
        }
        candidates = [
            dict(candidate)
            for candidate in page.get("candidates") or []
            if str(candidate.get("notation") or "") in choices
        ]
        output.append(
            {
                **dict(page),
                "candidates": candidates
                or [dict(candidate) for candidate in page.get("candidates") or []],
            }
        )
    return output


def _stage_cache_path(
    root: Path,
    pages: Sequence[Mapping[str, Any]],
    *,
    primary_model: str,
    challenger_model: str,
    tie_break_model: str,
) -> tuple[str, Path]:
    identity = {
        "schema": STAGE_CACHE_SCHEMA,
        "consensus_schema": CONSENSUS_SCHEMA,
        "models": {
            "primary": primary_model,
            "challenger": challenger_model,
            "tie_break": tie_break_model,
        },
        "pages": [
            {
                "uid": str(page["uid"]),
                "source_sha256": str(page["source_sha256"]),
                "candidates": [
                    str(candidate["notation"])
                    for candidate in page.get("candidates") or []
                ],
            }
            for page in pages
        ],
    }
    digest = hashlib.sha256(
        json.dumps(
            identity,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    path = (
        root
        / "runtime"
        / "librarian"
        / "classification-stage-cache"
        / f"{digest}.json"
    )
    return digest, path


def _load_stage_cache(path: Path, cache_key: str) -> dict[str, Any]:
    try:
        payload = verify_sealed_object(
            json.loads(path.read_text(encoding="utf-8"))
        )
    except (OSError, ValueError, TypeError, DurableStateError):
        return {
            "schema": STAGE_CACHE_SCHEMA,
            "cache_key": cache_key,
            "stages": {},
        }
    if (
        payload.get("schema") != STAGE_CACHE_SCHEMA
        or payload.get("cache_key") != cache_key
        or not isinstance(payload.get("stages"), dict)
    ):
        return {
            "schema": STAGE_CACHE_SCHEMA,
            "cache_key": cache_key,
            "stages": {},
        }
    return payload


def _valid_cached_stage(
    rows: object,
    pages: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]] | None:
    if not isinstance(rows, list) or len(rows) != len(pages):
        return None
    by_uid = {
        str(row.get("uid") or ""): dict(row)
        for row in rows
        if isinstance(row, Mapping)
    }
    output: list[dict[str, Any]] = []
    for page in pages:
        uid = str(page["uid"])
        row = by_uid.get(uid)
        if row is None:
            return None
        allowed = {
            str(candidate["notation"]) for candidate in page.get("candidates") or []
        }
        if str(row.get("primary_notation") or "") not in allowed:
            return None
        if any(
            str(value) not in allowed
            for value in row.get("secondary_notations") or []
        ):
            return None
        output.append(row)
    return output


def _cached_stage_call(
    *,
    cache: dict[str, Any],
    cache_path: Path,
    stage: str,
    model: str,
    keep_alive: str,
    pages: Sequence[Mapping[str, Any]],
    role: str,
    prior: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], int]:
    stages = cache.setdefault("stages", {})
    raw_cached = stages.get(stage)
    cached = _valid_cached_stage(raw_cached, pages)
    if cached is not None:
        return cached, 0
    decisions: list[dict[str, Any]] = []
    if isinstance(raw_cached, list) and len(raw_cached) < len(pages):
        valid_prefix = _valid_cached_stage(
            raw_cached,
            pages[: len(raw_cached)],
        )
        if valid_prefix is not None:
            decisions = valid_prefix
    calls = 0
    for offset in range(len(decisions), len(pages), STAGE_CHUNK_SIZE):
        chunk = pages[offset : offset + STAGE_CHUNK_SIZE]
        prior_chunk = (
            prior[offset : offset + STAGE_CHUNK_SIZE]
            if prior is not None
            else None
        )
        chunk_decisions, chunk_calls = _call(
            model=model,
            keep_alive=keep_alive,
            pages=chunk,
            role=role,
            prior=prior_chunk,
        )
        decisions.extend(chunk_decisions)
        calls += chunk_calls
        stages[stage] = decisions
        write_sealed_json(cache_path, cache, backup=False)
    return decisions, calls


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
    cache_key, cache_path = _stage_cache_path(
        root,
        pages,
        primary_model=config.primary_model,
        challenger_model=config.challenger_model,
        tie_break_model=config.tie_break_model,
    )
    cache = _load_stage_cache(cache_path, cache_key)
    primary, primary_calls = _cached_stage_call(
        cache=cache,
        cache_path=cache_path,
        stage="primary",
        model=config.primary_model,
        keep_alive=config.primary_keep_alive,
        pages=pages,
        role="primary-proposer",
    )
    challenger, challenger_calls = _cached_stage_call(
        cache=cache,
        cache_path=cache_path,
        stage="challenger",
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
        disputed_pages = [pages[index] for index in disagreements]
        disputed_primary = [primary[index] for index in disagreements]
        disputed_challenger = [challenger[index] for index in disagreements]
        tie_pages = _tie_candidate_pages(
            disputed_pages,
            disputed_primary,
            disputed_challenger,
        )
        paired_prior = [
            {
                "uid": str(pages[index]["uid"]),
                "primary_proposal": primary[index],
                "challenger_proposal": challenger[index],
            }
            for index in disagreements
        ]
        tie, tie_calls = _cached_stage_call(
            cache=cache,
            cache_path=cache_path,
            stage="tie_break",
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
