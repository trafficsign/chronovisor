"""Isolated provider-neutral worker for Librarian classification batches."""

from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from chronovisor.core import ollama
from chronovisor.core.durable_state import (
    DurableStateError,
    verify_sealed_object,
    write_sealed_json,
)
from chronovisor.core.runtime_config import load_decision_router_config
from chronovisor.recall.classification import (
    AUTHORITY_KINDS,
    ClassificationError,
    load_udc_package,
    resolve_consensus_runtime_routes,
    resolve_single_runtime_route,
)
from chronovisor.recall.classification_engine import AUTHORITY_SCHEMA, CONSENSUS_SCHEMA

STAGE_CACHE_SCHEMA = "chronovisor.classification-stage-cache.v4"
# A page's classification must not depend on unrelated pages sharing its batch.
STAGE_CHUNK_SIZE = 1


def _schema(count: int, *, dual_blind: bool = False) -> dict[str, Any]:
    required = [
        "uid",
        "primary_notation",
        "secondary_notations",
        "confidence",
        "rationale",
    ]
    if dual_blind:
        required.append("expected_status")
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
                    "required": required,
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
                        "expected_status": {
                            "type": "string",
                            "enum": ["proposed", "held"],
                        },
                    },
                },
            }
        },
    }


def _page_prompt(page: Mapping[str, Any]) -> dict[str, Any]:
    payload = {
        "uid": str(page["uid"]),
        "title": str(page.get("title") or ""),
        "summary": str(page.get("summary") or ""),
        "tags": [str(value) for value in page.get("tags") or []],
        "raw_keywords": [str(value) for value in (page.get("raw_keywords") or [])[:30]],
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
    evidence_card = page.get("evidence_card")
    if isinstance(evidence_card, Mapping):
        payload["evidence_card"] = dict(evidence_card)
    return payload


def _call(
    *,
    route: Mapping[str, Any],
    keep_alive: str,
    pages: Sequence[Mapping[str, Any]],
    role: str,
    source_sensitivity: str = "high",
    prior: Sequence[Mapping[str, Any]] | None = None,
    dual_blind: bool = False,
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
    if dual_blind:
        prompt["adjudication_contract"] = (
            "This is an independent gold-fixture adjudication. Do not infer "
            "another reviewer's answer. Set expected_status=proposed only when "
            "the page text supports one allowed UDC concept. Set "
            "expected_status=held when the text is genuinely ambiguous or "
            "insufficient; still return the nearest allowed parent as "
            "primary_notation for auditability."
        )
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
    response = ollama.runtime_structured_chat(
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
        runtime_role=str(route["role"]),
        source_data_class="page",
        source_sensitivity=source_sensitivity,
        format=_schema(len(pages), dual_blind=dual_blind),
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
        payload = json.loads(response.content)
    except json.JSONDecodeError:
        if len(pages) > 1:
            midpoint = len(pages) // 2
            prior_left = prior[:midpoint] if prior is not None else None
            prior_right = prior[midpoint:] if prior is not None else None
            left, left_calls = _call(
                route=route,
                keep_alive=keep_alive,
                pages=pages[:midpoint],
                role=role,
                source_sensitivity=source_sensitivity,
                prior=prior_left,
                dual_blind=dual_blind,
            )
            right, right_calls = _call(
                route=route,
                keep_alive=keep_alive,
                pages=pages[midpoint:],
                role=role,
                source_sensitivity=source_sensitivity,
                prior=prior_right,
                dual_blind=dual_blind,
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
                    "rationale": f"{route['model']} returned truncated JSON.",
                    **({"expected_status": "held"} if dual_blind else {}),
                    "_invalid_reason": "model_json_truncated",
                }
            ],
            1,
        )
    decisions = payload.get("decisions")
    if not isinstance(decisions, list):
        raise TypeError(f"{route['model']} returned no decision list")
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
                "rationale": f"{route['model']} omitted this page.",
                **({"expected_status": "held"} if dual_blind else {}),
                "_invalid_reason": "model_omitted_page",
            }
        allowed = {
            str(candidate["notation"]) for candidate in page.get("candidates") or []
        }
        primary = str(decision.get("primary_notation") or "")
        secondary = [str(value) for value in decision.get("secondary_notations") or []]
        if dual_blind and decision.get("expected_status") not in {
            "proposed",
            "held",
        }:
            decision["_invalid_reason"] = "invalid_expected_status"
            decision["confidence"] = 0.0
        if primary not in allowed:
            decision["_invalid_reason"] = "notation_outside_host_candidates"
            decision["confidence"] = 0.0
        else:
            rejected_secondary = [value for value in secondary if value not in allowed]
            decision["secondary_notations"] = [
                value for value in secondary if value in allowed and value != primary
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
    runtime_routes: Sequence[Mapping[str, Any]],
    adjudication_mode: str,
    stage_cache_epoch: str,
    authority_kind: str = "quorum_v1",
) -> tuple[str, Path]:
    schema = AUTHORITY_SCHEMA if authority_kind == "single_model_v1" else CONSENSUS_SCHEMA
    identity = {
        "schema": STAGE_CACHE_SCHEMA,
        "authority_kind": authority_kind,
        "stage_schema": schema,
        "adjudication_mode": adjudication_mode,
        "stage_cache_epoch": stage_cache_epoch,
        "runtime_routes": [dict(route) for route in runtime_routes],
        "pages": [
            {
                "uid": str(page["uid"]),
                "source_sha256": str(page["source_sha256"]),
                "candidates": [
                    str(candidate["notation"])
                    for candidate in page.get("candidates") or []
                ],
                "evidence_card_sha256": hashlib.sha256(
                    json.dumps(
                        page.get("evidence_card") or {},
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest(),
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
        root / "runtime" / "librarian" / "classification-stage-cache" / f"{digest}.json"
    )
    return digest, path


def _load_stage_cache(path: Path, cache_key: str) -> dict[str, Any]:
    try:
        payload = verify_sealed_object(json.loads(path.read_text(encoding="utf-8")))
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
        str(row.get("uid") or ""): dict(row) for row in rows if isinstance(row, Mapping)
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
        primary = str(row.get("primary_notation") or "")
        safely_marked_invalid = (
            row.get("_invalid_reason") == "notation_outside_host_candidates"
            and float(row.get("confidence") or 0.0) == 0.0
        )
        if primary not in allowed and not safely_marked_invalid:
            return None
        if any(
            str(value) not in allowed for value in row.get("secondary_notations") or []
        ):
            return None
        output.append(row)
    return output


def _cached_stage_call(
    *,
    cache: dict[str, Any],
    cache_path: Path,
    stage: str,
    route: Mapping[str, Any],
    keep_alive: str,
    pages: Sequence[Mapping[str, Any]],
    role: str,
    source_sensitivity: str = "high",
    prior: Sequence[Mapping[str, Any]] | None = None,
    dual_blind: bool = False,
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
            prior[offset : offset + STAGE_CHUNK_SIZE] if prior is not None else None
        )
        chunk_decisions, chunk_calls = _call(
            route=route,
            keep_alive=keep_alive,
            pages=chunk,
            role=role,
            source_sensitivity=source_sensitivity,
            prior=prior_chunk,
            dual_blind=dual_blind,
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


def _authority_digest(uid: str, authority: Mapping[str, Any]) -> str:
    payload = json.dumps(
        {"uid": uid, "authority": dict(authority)},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _run_single_authority(
    *,
    root: Path,
    package: Any,
    pages: Sequence[Mapping[str, Any]],
    authority_route: Mapping[str, Any],
    config: Any,
    adjudication_mode: str,
    stage_cache_epoch: str,
    source_sensitivity: str,
) -> dict[str, Any]:
    runtime_routes = (dict(authority_route),)
    cache_key, cache_path = _stage_cache_path(
        root,
        pages,
        runtime_routes=runtime_routes,
        adjudication_mode=adjudication_mode,
        stage_cache_epoch=stage_cache_epoch,
        authority_kind="single_model_v1",
    )
    cache = _load_stage_cache(cache_path, cache_key)
    keep_alive = getattr(config, "authority_keep_alive", None) or getattr(
        config, "primary_keep_alive", "20m"
    )
    authority_rows, model_calls = _cached_stage_call(
        cache=cache,
        cache_path=cache_path,
        stage="authority",
        route=authority_route,
        keep_alive=keep_alive,
        pages=pages,
        role="single-authority",
        source_sensitivity=source_sensitivity,
        dual_blind=False,
    )
    decisions: list[dict[str, Any]] = []
    for page, row in zip(pages, authority_rows, strict=True):
        uid = str(page["uid"])
        notation = str(row.get("primary_notation") or "")
        secondary = [
            str(value) for value in row.get("secondary_notations") or []
        ][:3]
        invalid_reason = str(row.get("_invalid_reason") or "")
        valid = not invalid_reason
        try:
            confidence = float(row.get("confidence") or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
            valid = False
            invalid_reason = invalid_reason or "invalid_confidence"
        confidence = max(0.0, min(1.0, confidence)) if valid else 0.0
        authority_digest = _authority_digest(uid, row)
        decisions.append(
            {
                "uid": uid,
                "primary_notation": notation,
                "secondary_notations": secondary,
                "confidence": confidence,
                "rationale": str(row.get("rationale") or "")[:400],
                "status": "proposed" if valid else "held",
                "authority_kind": "single_model_v1",
                "authority_model": str(authority_route["model"]),
                "authority_digest": authority_digest,
                "validation_count": 1,
                **({"_invalid_reason": invalid_reason} if invalid_reason else {}),
            }
        )
    authority = {
        "kind": "single_model_v1",
        "route": dict(authority_route),
        "model": str(authority_route["model"]),
        "revision": authority_route.get("revision"),
        "result_sha256": _authority_digest(
            "batch",
            {"decisions": decisions, "route": authority_route},
        ),
        "validation_count": 1,
        "attempts": [
            {
                "stage": "authority",
                "model": str(authority_route["model"]),
                "calls": model_calls,
            }
        ],
    }
    for decision in decisions:
        decision["authority"] = dict(authority)
    return {
        "schema": AUTHORITY_SCHEMA,
        "authority_kind": "single_model_v1",
        "authority": authority,
        "package_checksum": package.checksum,
        "model_calls": model_calls,
        "runtime_routes": [dict(authority_route)],
        "decisions": decisions,
    }


def run(payload: Mapping[str, Any]) -> dict[str, Any]:
    schema = payload.get("schema")
    if schema not in {CONSENSUS_SCHEMA, AUTHORITY_SCHEMA}:
        raise ValueError("unsupported classification authority schema")
    authority_kind = str(
        payload.get("authority_kind")
        or ("single_model_v1" if schema == AUTHORITY_SCHEMA else "quorum_v1")
    )
    if authority_kind not in AUTHORITY_KINDS:
        raise ClassificationError("classification authority kind is invalid")
    if schema == AUTHORITY_SCHEMA and authority_kind != "single_model_v1":
        raise ClassificationError("authority schema requires single_model_v1")
    if schema == CONSENSUS_SCHEMA and authority_kind != "quorum_v1":
        raise ClassificationError("consensus schema requires quorum_v1")
    root = Path(str(payload["root"]))
    package = load_udc_package(root)
    if not package.complete:
        raise ValueError("local consensus requires a complete UDC package")
    pages = payload.get("pages")
    if not isinstance(pages, list) or not pages:
        raise ValueError("classification worker requires pages")
    forbidden_overrides = {
        "runtime_role",
        "model",
        "primary_model",
        "challenger_model",
        "tie_break_model",
        "authority_model",
    }
    if forbidden_overrides & payload.keys():
        raise ClassificationError("classification runtime route override is forbidden")
    adjudication_mode = payload.get("adjudication_mode", "proposal-audit")
    if not isinstance(adjudication_mode, str) or adjudication_mode not in {
        "proposal-audit",
        "dual-blind",
    }:
        raise ClassificationError("unsupported classification adjudication mode")
    stage_cache_epoch = payload.get("stage_cache_epoch", "default")
    if (
        not isinstance(stage_cache_epoch, str)
        or not stage_cache_epoch.strip()
        or len(stage_cache_epoch) > 80
    ):
        raise ClassificationError("classification stage cache epoch is invalid")
    source_sensitivity = payload.get("source_sensitivity", "high")
    if not isinstance(source_sensitivity, str) or source_sensitivity not in {
        "normal",
        "high",
    }:
        raise ClassificationError("classification source sensitivity is invalid")
    if authority_kind == "single_model_v1":
        runtime_route_payload = payload.get("runtime_route")
        if runtime_route_payload is None:
            runtime_route_payload = payload.get("runtime_routes")
        authority_route = resolve_single_runtime_route(runtime_route_payload)
        config = load_decision_router_config()
        return _run_single_authority(
            root=root,
            package=package,
            pages=pages,
            authority_route=authority_route,
            config=config,
            adjudication_mode=adjudication_mode,
            stage_cache_epoch=stage_cache_epoch,
            source_sensitivity=source_sensitivity,
        )
    runtime_routes = resolve_consensus_runtime_routes(payload.get("runtime_routes"))
    primary_route, challenger_route, tie_break_route = runtime_routes
    config = load_decision_router_config()
    dual_blind = adjudication_mode == "dual-blind"
    cache_key, cache_path = _stage_cache_path(
        root,
        pages,
        runtime_routes=runtime_routes,
        adjudication_mode=adjudication_mode,
        stage_cache_epoch=stage_cache_epoch,
        authority_kind=authority_kind,
    )
    cache = _load_stage_cache(cache_path, cache_key)
    primary, primary_calls = _cached_stage_call(
        cache=cache,
        cache_path=cache_path,
        stage="primary",
        route=primary_route,
        keep_alive=config.primary_keep_alive,
        pages=pages,
        role="primary-proposer",
        source_sensitivity=source_sensitivity,
        dual_blind=dual_blind,
    )
    challenger, challenger_calls = _cached_stage_call(
        cache=cache,
        cache_path=cache_path,
        stage="challenger",
        route=challenger_route,
        keep_alive=config.challenger_keep_alive,
        pages=pages,
        role="independent-challenger",
        source_sensitivity=source_sensitivity,
        prior=None if dual_blind else primary,
        dual_blind=dual_blind,
    )
    disagreements = [
        index
        for index, (left, right) in enumerate(zip(primary, challenger, strict=True))
        if left.get("_invalid_reason")
        or right.get("_invalid_reason")
        or (
            dual_blind
            and str(left.get("expected_status") or "")
            != str(right.get("expected_status") or "")
        )
        or (
            (not dual_blind or str(left.get("expected_status") or "") == "proposed")
            and str(left["primary_notation"]) != str(right["primary_notation"])
        )
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
            route=tie_break_route,
            keep_alive=config.tie_break_keep_alive,
            pages=tie_pages,
            role="tie-break-adjudicator",
            source_sensitivity=source_sensitivity,
            prior=paired_prior,
            dual_blind=dual_blind,
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
        left_status = str(left.get("expected_status") or "proposed")
        right_status = str(right.get("expected_status") or "proposed")
        tie_status = str(
            (tie.get("expected_status") if tie is not None else None) or "proposed"
        )
        if dual_blind and (
            left_valid and right_valid and left_status == right_status == "held"
        ):
            winner = left
            votes = 2
            expected_status = "held"
        elif (
            left_valid
            and right_valid
            and left["primary_notation"] == right["primary_notation"]
            and (not dual_blind or left_status == right_status == "proposed")
        ):
            winner = left
            votes = 2
            expected_status = "proposed"
        elif tie_valid and (
            (
                dual_blind
                and tie_status == "held"
                and (
                    (left_valid and left_status == "held")
                    or (right_valid and right_status == "held")
                )
            )
            or (
                tie_status == "proposed"
                and left_valid
                and tie["primary_notation"] == left["primary_notation"]
                and (not dual_blind or left_status == "proposed")
            )
            or (
                tie_status == "proposed"
                and right_valid
                and tie["primary_notation"] == right["primary_notation"]
                and (not dual_blind or right_status == "proposed")
            )
        ):
            winner = tie
            votes = 2
            expected_status = tie_status if dual_blind else "proposed"
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
            expected_status = "held" if dual_blind else "proposed"
        confidence = min(
            float(winner.get("confidence") or 0.0),
            0.99 if votes >= 2 else 0.49,
        )
        decisions.append(
            {
                "uid": uid,
                "primary_notation": str(winner["primary_notation"]),
                "secondary_notations": [
                    str(value) for value in winner.get("secondary_notations") or []
                ][:3],
                "confidence": confidence,
                "rationale": str(winner.get("rationale") or "")[:400],
                "status": (
                    expected_status
                    if dual_blind and votes >= 2
                    else "proposed"
                    if votes >= 2
                    else "held"
                ),
                "expected_status": (expected_status if votes >= 2 else "unresolved"),
                "adjudication_mode": adjudication_mode,
                "quorum": votes,
                "primary_model": primary_route["model"],
                "challenger_model": challenger_route["model"],
                "tie_break_model": tie_break_route["model"] if tie is not None else None,
                "consensus_sha256": _decision_digest(uid, left, right, tie),
            }
        )
    return {
        "schema": CONSENSUS_SCHEMA,
        "authority_kind": "quorum_v1",
        "package_checksum": package.checksum,
        "model_calls": model_calls,
        "runtime_routes": list(runtime_routes),
        "decisions": decisions,
    }


def main() -> None:
    payload = json.loads(sys.stdin.read())
    result = run(payload)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
