"""Replay and adoption gate for the local structured-decision ensemble.

The evaluator consumes historical decisions as read-only evidence.  It never
imports or calls the frontier repair plane, and its durable artifact contains
only hashes, decision labels, validation diagnostics, and aggregate metrics --
never prompts or literal model responses.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from llm_wiki_mcp import ollama
from llm_wiki_mcp.decision_router import DecisionRouter, DecisionRouterResult
from llm_wiki_mcp.decision_schema_manifest import (
    decision_signature_value,
    default_decision_value,
    production_schema_manifest,
    production_signature_manifest,
)
from llm_wiki_mcp.local_structured import (
    ChatRequest,
    ChatTransport,
    validate_json,
    validate_schema_definition,
)
from llm_wiki_mcp.model_lab import REPLAY_FILE
from llm_wiki_mcp.runtime_config import (
    DecisionRouterConfig,
    load_decision_router_config,
)

ARTIFACT_SCHEMA_VERSION = 2
LEGACY_REPLAY_PROMPT_LIMIT = 50_000
# This is deliberately not configurable from the CLI.  A tiny hand-picked
# slice is useful for smoke testing, but it must never become an adoption
# artifact merely because every answer in that slice happened to agree.
MIN_ADOPTION_USABLE_CASES = 100
MIN_CASES_PER_PRODUCTION_SCHEMA = 5
UNSAFE_HOLD_DECISIONS = frozenset(
    {
        "blocked",
        "invalid",
        "needs_retry",
        "quarantined",
        "reject",
        "rejected",
    }
)
APPLY_DECISIONS = frozenset(
    {
        "accept",
        "accepted",
        "apply",
        "apply_available",
        "approved",
        "confirmed_noop",
        "supersede_left",
        "supersede_right",
    }
)


class ReplayInputError(ValueError):
    """Raised before inference when the selected replay corpus is invalid."""


class ResumeMismatchError(ValueError):
    """Raised when an artifact cannot safely resume the requested run."""


@dataclass(frozen=True)
class AdoptionThresholds:
    """Minimum conditions from the local-consensus rollout plan."""

    first_pass_schema_rate: float = 0.98
    final_schema_rate: float = 1.0
    pair_valid_rate: float = 0.99
    pair_agreement_rate: float = 0.75
    majority_resolution_rate: float = 0.99
    historical_signature_match_rate: float = 0.90
    max_invalid_output_accepted: int = 0
    max_unsafe_decision_flips: int = 0


@dataclass(frozen=True)
class ReplayCase:
    index: int
    case_id: str
    role: str
    prompt: str = field(repr=False)
    system: str | None = field(repr=False)
    schema: dict[str, Any] = field(repr=False)
    expected: dict[str, Any] = field(repr=False)
    schema_sha256: str
    expected_signature_sha256: str

    @property
    def expected_decision(self) -> str | None:
        value = self.expected.get("decision")
        return value if isinstance(value, str) else None

    @property
    def expected_coverage_label(self) -> str | None:
        for key in ("decision", "action", "classification", "approved"):
            if key not in self.expected:
                continue
            value = self.expected[key]
            if isinstance(value, (str, bool, int, float)):
                return f"{key}={_canonical_json(value)}"
        return None

    def listing(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "case_id": self.case_id,
            "role": self.role,
            "schema_sha256": self.schema_sha256,
            "expected_signature_sha256": self.expected_signature_sha256,
            "expected_decision": self.expected_decision,
        }


@dataclass(frozen=True)
class ReplayCorpus:
    path: Path
    source_sha256: str
    total_cases: int
    usable_cases: int
    excluded_cases: int
    excluded_reasons: tuple[tuple[str, int], ...]
    offset: int
    limit: int
    cases: tuple[ReplayCase, ...]
    usable_roles: tuple[str, ...]
    usable_decisions: tuple[str, ...]
    required_schema_manifest: tuple[tuple[str, str], ...]
    usable_schema_counts: tuple[tuple[str, int], ...]
    signature_manifest_sha256: str

    @property
    def full_usable_selection(self) -> bool:
        return self.offset == 0 and len(self.cases) == self.usable_cases

    def inspection(self, *, include_cases: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "status": "valid",
            "source_path": str(self.path),
            "source_sha256": self.source_sha256,
            "total_cases": self.total_cases,
            "usable_cases": self.usable_cases,
            "excluded_cases": self.excluded_cases,
            "excluded_reasons": dict(self.excluded_reasons),
            "offset": self.offset,
            "limit": self.limit,
            "selected_cases": len(self.cases),
            "full_usable_selection": self.full_usable_selection,
            "coverage": self.coverage(),
            "selected_case_ids_sha256": _sha256_json(
                [case.case_id for case in self.cases]
            ),
        }
        if include_cases:
            payload["cases"] = [case.listing() for case in self.cases]
        return payload

    def coverage(self) -> dict[str, Any]:
        selected_roles = tuple(sorted({case.role for case in self.cases}))
        selected_decisions = tuple(
            sorted(
                {
                    decision
                    for case in self.cases
                    if (decision := case.expected_coverage_label) is not None
                }
            )
        )
        selected_schema_counts = Counter(case.schema_sha256 for case in self.cases)
        usable_schema_counts = dict(self.usable_schema_counts)
        schema_names_by_digest: dict[str, list[str]] = {}
        for name, digest in self.required_schema_manifest:
            schema_names_by_digest.setdefault(digest, []).append(name)
        required_schemas = [
            {
                "names": sorted(names),
                "sha256": digest,
                "usable_cases": usable_schema_counts.get(digest, 0),
                "selected_cases": selected_schema_counts.get(digest, 0),
            }
            for digest, names in sorted(schema_names_by_digest.items())
        ]
        schema_manifest = [
            {"name": name, "sha256": digest}
            for name, digest in self.required_schema_manifest
        ]
        selected_required_counts = [
            int(row["selected_cases"]) for row in required_schemas
        ]
        covered_schemas = sum(count > 0 for count in selected_required_counts)
        schema_denominator = len(required_schemas)
        return {
            "usable_roles": list(self.usable_roles),
            "selected_roles": list(selected_roles),
            "role_coverage_rate": _set_coverage_rate(
                selected_roles, self.usable_roles
            ),
            "usable_decisions": list(self.usable_decisions),
            "selected_decisions": list(selected_decisions),
            "decision_coverage_rate": _set_coverage_rate(
                selected_decisions, self.usable_decisions
            ),
            "required_schemas": required_schemas,
            "schema_manifest_sha256": _sha256_json(schema_manifest),
            "signature_manifest_sha256": self.signature_manifest_sha256,
            "production_schema_coverage_rate": _rate(
                covered_schemas, schema_denominator
            ),
            "minimum_production_schema_cases": (
                min(selected_required_counts) if selected_required_counts else 0
            ),
            "missing_production_schemas": [
                name
                for row in required_schemas
                if int(row["selected_cases"]) == 0
                for name in row["names"]
            ],
            "underrepresented_production_schemas": [
                name
                for row in required_schemas
                if int(row["selected_cases"])
                < MIN_CASES_PER_PRODUCTION_SCHEMA
                for name in row["names"]
            ],
        }


ModelMetadataProvider = Callable[[Sequence[str]], Mapping[str, Any]]


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _set_coverage_rate(selected: Sequence[str], available: Sequence[str]) -> float:
    expected = set(available)
    if not expected:
        return 0.0
    return round(len(set(selected) & expected) / len(expected), 6)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _decision_signature(
    value: Any,
    schema: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    selected = (
        decision_signature_value(schema, value)
        if schema is not None
        else default_decision_value(value)
    )
    return dict(selected) if isinstance(selected, Mapping) else {}


def replay_agreement_value(value: Any) -> dict[str, Any]:
    """Use the same bounded historical decision signature for every vote."""

    signature = _decision_signature(value)
    if not signature:
        raise ValueError("replay output has no decision signature")
    return signature


def _case_id(
    index: int,
    prompt: str,
    system: str | None,
    schema: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> str:
    payload: dict[str, Any] = {
        "index": index,
        "prompt": prompt,
        "schema": schema,
        "expected": expected,
    }
    if system is not None:
        payload["system"] = system
    return _sha256_json(payload)


def _validate_expected(
    expected: Mapping[str, Any], schema: Mapping[str, Any], *, line_number: int
) -> None:
    properties = schema.get("properties")
    if not isinstance(properties, Mapping):
        raise ReplayInputError(
            f"line {line_number}: replay schema must expose object properties"
        )
    if not expected:
        raise ReplayInputError(f"line {line_number}: expected signature is empty")
    for key, value in expected.items():
        child = properties.get(key)
        if not isinstance(child, Mapping):
            raise ReplayInputError(
                f"line {line_number}: expected field {key!r} is absent from schema"
            )
        issues = validate_json(value, child)
        if issues:
            issue = issues[0]
            raise ReplayInputError(
                f"line {line_number}: expected field {key!r} violates "
                f"{issue.keyword} at {issue.pointer or '/'}"
            )
    if not _decision_signature(expected, schema):
        raise ReplayInputError(
            f"line {line_number}: expected must contain a decision signature"
        )


def load_replay_corpus(
    path: Path | str = REPLAY_FILE,
    *,
    offset: int = 0,
    limit: int = 0,
    required_schema_manifest: Mapping[str, str] | None = None,
) -> ReplayCorpus:
    """Load and validate replay JSONL without invoking any model."""

    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise ReplayInputError("offset must be an integer >= 0")
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
        raise ReplayInputError("limit must be an integer >= 0")
    uses_production_manifest = required_schema_manifest is None
    manifest = dict(
        production_schema_manifest()
        if uses_production_manifest
        else required_schema_manifest
    )
    if not manifest or any(
        not isinstance(name, str)
        or not name
        or not isinstance(digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        for name, digest in manifest.items()
    ):
        raise ReplayInputError(
            "required_schema_manifest must contain named SHA-256 digests"
        )
    signature_manifest = (
        production_signature_manifest()
        if uses_production_manifest
        else {
            name: {
                "policy_version": 1,
                "schema_sha256": digest,
                "fields": [],
            }
            for name, digest in manifest.items()
        }
    )
    source = Path(path).expanduser()
    try:
        raw = source.read_bytes()
    except OSError as exc:
        raise ReplayInputError(f"cannot read replay input: {exc}") from exc
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ReplayInputError(f"replay input is not UTF-8: {exc}") from exc

    rows: list[ReplayCase] = []
    excluded_reasons: dict[str, int] = {}
    total_cases = 0
    for index, line in enumerate(text.splitlines()):
        line_number = index + 1
        if not line.strip():
            raise ReplayInputError(f"line {line_number}: blank JSONL record")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ReplayInputError(
                f"line {line_number}: invalid JSON at column {exc.colno}"
            ) from exc
        if not isinstance(row, dict):
            raise ReplayInputError(f"line {line_number}: record must be an object")
        total_cases += 1
        prompt = row.get("prompt")
        system = row.get("system")
        schema = row.get("schema")
        expected = row.get("expected")
        role = row.get("role")
        if not isinstance(prompt, str) or not prompt:
            raise ReplayInputError(
                f"line {line_number}: prompt must be a non-empty string"
            )
        if system is not None and not isinstance(system, str):
            raise ReplayInputError(
                f"line {line_number}: system must be a string or null"
            )
        prompt_truncated = row.get("prompt_truncated")
        if prompt_truncated is not None and not isinstance(prompt_truncated, bool):
            raise ReplayInputError(
                f"line {line_number}: prompt_truncated must be a boolean"
            )
        exclusion_reason: str | None = None
        if prompt_truncated is True:
            exclusion_reason = "explicit_prompt_truncated"
        elif (
            prompt_truncated is None
            and len(prompt) == LEGACY_REPLAY_PROMPT_LIMIT
        ):
            # Older record_replay_case() silently retained the last 50,000
            # characters.  An exact-length unmarked prompt from that format
            # cannot prove that its leading instructions are intact.
            exclusion_reason = "legacy_exact_50000_without_marker"
        if exclusion_reason is not None:
            excluded_reasons[exclusion_reason] = (
                excluded_reasons.get(exclusion_reason, 0) + 1
            )
            continue
        if not isinstance(schema, dict):
            raise ReplayInputError(f"line {line_number}: schema must be an object")
        if not isinstance(expected, dict):
            raise ReplayInputError(f"line {line_number}: expected must be an object")
        if not isinstance(role, str) or not role:
            raise ReplayInputError(f"line {line_number}: role must be non-empty")
        try:
            validate_schema_definition(schema)
        except Exception as exc:
            raise ReplayInputError(f"line {line_number}: invalid schema: {exc}") from exc
        _validate_expected(expected, schema, line_number=line_number)
        schema_copy = json.loads(_canonical_json(schema))
        expected_copy = json.loads(_canonical_json(expected))
        signature = _decision_signature(expected_copy, schema_copy)
        rows.append(
            ReplayCase(
                index=index,
                case_id=_case_id(
                    index,
                    prompt,
                    system,
                    schema_copy,
                    expected_copy,
                ),
                role=role,
                prompt=prompt,
                system=system,
                schema=schema_copy,
                expected=expected_copy,
                schema_sha256=_sha256_json(schema_copy),
                expected_signature_sha256=_sha256_json(signature),
            )
        )

    usable_roles = tuple(sorted({case.role for case in rows}))
    usable_decisions = tuple(
        sorted(
            {
                decision
                for case in rows
                if (decision := case.expected_coverage_label) is not None
            }
        )
    )
    usable_schema_counts = Counter(case.schema_sha256 for case in rows)
    end = None if limit == 0 else offset + limit
    selected = tuple(rows[offset:end])
    if not selected:
        raise ReplayInputError(
            "selection is empty "
            f"(total={total_cases}, usable={len(rows)}, "
            f"excluded={total_cases - len(rows)}, offset={offset}, limit={limit})"
        )
    return ReplayCorpus(
        path=source,
        source_sha256=hashlib.sha256(raw).hexdigest(),
        total_cases=total_cases,
        usable_cases=len(rows),
        excluded_cases=total_cases - len(rows),
        excluded_reasons=tuple(sorted(excluded_reasons.items())),
        offset=offset,
        limit=limit,
        cases=selected,
        usable_roles=usable_roles,
        usable_decisions=usable_decisions,
        required_schema_manifest=tuple(sorted(manifest.items())),
        usable_schema_counts=tuple(sorted(usable_schema_counts.items())),
        signature_manifest_sha256=_sha256_json(signature_manifest),
    )


def inspect_replays(
    path: Path | str = REPLAY_FILE,
    *,
    offset: int = 0,
    limit: int = 0,
    include_cases: bool = False,
    required_schema_manifest: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Inspect corpus integrity and selection without model/metadata calls."""

    return load_replay_corpus(
        path,
        offset=offset,
        limit=limit,
        required_schema_manifest=required_schema_manifest,
    ).inspection(
        include_cases=include_cases
    )


def fetch_local_model_metadata(models: Sequence[str]) -> Mapping[str, Any]:
    """Read Ollama engine/tag metadata without loading or running a model."""

    timeout = httpx.Timeout(connect=3.0, read=10.0, write=3.0, pool=3.0)
    with httpx.Client(base_url=ollama.OLLAMA_URL, timeout=timeout) as client:
        version_response = client.get("/api/version")
        version_response.raise_for_status()
        tags_response = client.get("/api/tags")
        tags_response.raise_for_status()
    version_body = version_response.json()
    tags_body = tags_response.json()
    available = tags_body.get("models") if isinstance(tags_body, dict) else []
    available = available if isinstance(available, list) else []
    records: dict[str, Any] = {}
    for requested in models:
        match = next(
            (
                row
                for row in available
                if isinstance(row, dict)
                and requested in {str(row.get("name") or ""), str(row.get("model") or "")}
            ),
            None,
        )
        records[requested] = match or {"name": requested, "status": "missing"}
    return {
        "engine": {
            "name": "ollama",
            "version": version_body.get("version")
            if isinstance(version_body, dict)
            else None,
        },
        "models": records,
    }


def _safe_model_metadata(payload: Mapping[str, Any], models: Sequence[str]) -> dict[str, Any]:
    engine = payload.get("engine")
    safe_engine = {
        "name": str(engine.get("name") or "") if isinstance(engine, Mapping) else "",
        "version": str(engine.get("version") or "") if isinstance(engine, Mapping) else "",
    }
    source_models = payload.get("models")
    source_models = source_models if isinstance(source_models, Mapping) else {}
    safe_models: dict[str, Any] = {}
    detail_keys = (
        "families",
        "family",
        "format",
        "parameter_size",
        "parent_model",
        "quantization_level",
    )
    for model in models:
        source = source_models.get(model)
        source = source if isinstance(source, Mapping) else {}
        details = source.get("details")
        details = details if isinstance(details, Mapping) else {}
        safe_models[model] = {
            "name": str(source.get("name") or source.get("model") or model),
            "digest": str(source.get("digest") or ""),
            "size": source.get("size") if isinstance(source.get("size"), int) else None,
            "modified_at": str(source.get("modified_at") or ""),
            "status": str(
                source.get("status") or ("available" if source else "missing")
            ),
            "details": {key: details.get(key) for key in detail_keys if key in details},
            "metadata_sha256": _sha256_json(source),
        }
    return {"engine": safe_engine, "models": safe_models}


def _live_transport(request: ChatRequest) -> str:
    return ollama.chat(
        [dict(message) for message in request.messages],
        model=request.model,
        format=request.schema,
        num_ctx=request.num_ctx,
        num_predict=request.num_predict,
        keep_alive=request.keep_alive,
        read_timeout_ms=request.read_timeout_ms,
        max_output_chars=request.max_output_chars,
    )


class _TimingTransport:
    def __init__(self, transport: ChatTransport) -> None:
        self.transport = transport
        self.events: list[dict[str, Any]] = []

    def __call__(self, request: ChatRequest) -> str:
        started = time.perf_counter()
        ok = False
        try:
            response = self.transport(request)
            ok = True
            return response
        finally:
            self.events.append(
                {
                    "model": request.model,
                    "latency_ms": round((time.perf_counter() - started) * 1000, 3),
                    "ok": ok,
                }
            )

    def mark(self) -> int:
        return len(self.events)

    def since(self, mark: int) -> list[dict[str, Any]]:
        return self.events[mark:]


def _unsafe_flip(expected: str | None, actual: str | None) -> bool:
    return bool(expected in UNSAFE_HOLD_DECISIONS and actual in APPLY_DECISIONS)


def _case_record(
    case: ReplayCase,
    result: DecisionRouterResult,
    *,
    latency_ms: float,
    transport_events: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    timing_by_model: dict[str, dict[str, Any]] = {}
    for event in transport_events:
        model = str(event.get("model") or "")
        aggregate = timing_by_model.setdefault(
            model, {"transport_calls": 0, "transport_failures": 0, "latency_ms": 0.0}
        )
        aggregate["transport_calls"] += 1
        aggregate["transport_failures"] += 0 if event.get("ok") is True else 1
        aggregate["latency_ms"] += float(event.get("latency_ms") or 0.0)

    vote_records: list[dict[str, Any]] = []
    invalid_output_accepted = 0
    for vote in result.votes:
        attempts = vote.result.attempts
        final_attempt_valid = bool(attempts and attempts[-1].valid)
        if vote.result.ok and not final_attempt_valid:
            invalid_output_accepted += 1
        timing = timing_by_model.get(
            vote.model,
            {"transport_calls": 0, "transport_failures": 0, "latency_ms": 0.0},
        )
        vote_records.append(
            {
                "role": vote.role,
                "model": vote.model,
                "vote_valid": vote.valid,
                "first_pass_schema_valid": vote.result.first_pass_valid,
                "final_schema_valid": vote.result.ok,
                "repaired_final_valid": bool(
                    vote.result.ok and vote.result.repair_turns > 0
                ),
                "repair_turns": vote.result.repair_turns,
                "transport_calls": timing["transport_calls"],
                "transport_failures": timing["transport_failures"],
                "latency_ms": round(float(timing["latency_ms"]), 3),
                "audit": vote.audit_record(),
            }
        )

    pair = result.votes[:2]
    pair_valid = len(pair) == 2 and all(vote.valid for vote in pair)
    pair_agreed = bool(
        pair_valid
        and pair[0].signature_sha256
        and pair[0].signature_sha256 == pair[1].signature_sha256
    )
    tie_invoked = len(result.votes) == 3
    actual_signature = (
        _decision_signature(result.value, case.schema) if result.ok else {}
    )
    actual_decision = actual_signature.get("decision")
    actual_decision = actual_decision if isinstance(actual_decision, str) else None
    expected_decision = case.expected_decision
    comparable = actual_decision is not None and expected_decision is not None
    return {
        "index": case.index,
        "case_id": case.case_id,
        "role": case.role,
        "schema_sha256": case.schema_sha256,
        "expected_signature_sha256": case.expected_signature_sha256,
        "actual_signature_sha256": _sha256_json(actual_signature)
        if actual_signature
        else None,
        "expected_decision": expected_decision,
        "actual_decision": actual_decision,
        "expected_decision_comparable": comparable,
        "expected_decision_match": comparable and actual_decision == expected_decision,
        "historical_signature_match": bool(
            result.ok
            and _decision_signature(case.expected, case.schema) == actual_signature
        ),
        "unsafe_decision_flip": _unsafe_flip(expected_decision, actual_decision),
        "status": result.status,
        "failure_class": result.failure_class,
        "quarantine_reason": result.quarantine_reason,
        "pair_valid": pair_valid,
        "pair_agreed": pair_agreed,
        "tie_break_invoked": tie_invoked,
        "tie_break_resolved": bool(tie_invoked and result.ok),
        "invalid_output_accepted": invalid_output_accepted,
        "latency_ms": round(latency_ms, 3),
        "votes": vote_records,
    }


def _rate(numerator: int, denominator: int, *, empty: float = 0.0) -> float:
    if denominator <= 0:
        return empty
    return round(numerator / denominator, 6)


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    fraction = position - low
    return round(ordered[low] + (ordered[high] - ordered[low]) * fraction, 3)


def _metrics(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    model_rows: dict[str, list[Mapping[str, Any]]] = {}
    vote_rows: list[Mapping[str, Any]] = []
    for case in cases:
        for vote in case.get("votes", []):
            if not isinstance(vote, Mapping):
                continue
            vote_rows.append(vote)
            model_rows.setdefault(str(vote.get("model") or ""), []).append(vote)

    def model_metric(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        sessions = len(rows)
        first = sum(row.get("first_pass_schema_valid") is True for row in rows)
        final = sum(row.get("final_schema_valid") is True for row in rows)
        repaired = sum(row.get("repaired_final_valid") is True for row in rows)
        calls = sum(int(row.get("transport_calls") or 0) for row in rows)
        return {
            "sessions": sessions,
            "first_pass_schema_valid": first,
            "first_pass_schema_rate": _rate(first, sessions),
            "final_schema_valid": final,
            "final_schema_rate": _rate(final, sessions),
            "repaired_final_valid": repaired,
            "invalid_final": sessions - final,
            "repair_turns": sum(int(row.get("repair_turns") or 0) for row in rows),
            "transport_calls": calls,
            "transport_failures": sum(
                int(row.get("transport_failures") or 0) for row in rows
            ),
            "latency_ms_total": round(
                sum(float(row.get("latency_ms") or 0.0) for row in rows), 3
            ),
        }

    processed = len(cases)
    pair_valid = sum(case.get("pair_valid") is True for case in cases)
    pair_agreed = sum(case.get("pair_agreed") is True for case in cases)
    ties = sum(case.get("tie_break_invoked") is True for case in cases)
    ties_resolved = sum(case.get("tie_break_resolved") is True for case in cases)
    comparable = sum(
        case.get("expected_decision_comparable") is True for case in cases
    )
    matched = sum(case.get("expected_decision_match") is True for case in cases)
    historical_signature_matches = sum(
        case.get("historical_signature_match") is True for case in cases
    )
    latencies = [float(case.get("latency_ms") or 0.0) for case in cases]
    overall = model_metric(vote_rows)
    return {
        "processed_cases": processed,
        "model_sessions": overall["sessions"],
        "first_pass_schema_valid": overall["first_pass_schema_valid"],
        "first_pass_schema_rate": overall["first_pass_schema_rate"],
        "final_schema_valid": overall["final_schema_valid"],
        "final_schema_rate": overall["final_schema_rate"],
        "repaired_final_valid": overall["repaired_final_valid"],
        "invalid_final": overall["invalid_final"],
        "invalid_output_accepted": sum(
            int(case.get("invalid_output_accepted") or 0) for case in cases
        ),
        "pair_valid_cases": pair_valid,
        "pair_valid_rate": _rate(pair_valid, processed),
        "pair_agreement_cases": pair_agreed,
        "pair_agreement_rate": _rate(pair_agreed, pair_valid),
        "pair_agreement_rate_of_all": _rate(pair_agreed, processed),
        "tie_break_invoked": ties,
        "tie_break_resolved": ties_resolved,
        "majority_resolution_rate": _rate(ties_resolved, ties, empty=1.0),
        "unresolved_quarantine": sum(
            case.get("status") == "quarantined" for case in cases
        ),
        "expected_decision_comparable": comparable,
        "expected_decision_matches": matched,
        "expected_decision_match_rate": _rate(matched, comparable),
        "historical_signature_matches": historical_signature_matches,
        "historical_signature_match_rate": _rate(
            historical_signature_matches, processed
        ),
        "unsafe_decision_flips": sum(
            case.get("unsafe_decision_flip") is True for case in cases
        ),
        "latency_ms": {
            "mean": round(sum(latencies) / len(latencies), 3) if latencies else None,
            "p50": _percentile(latencies, 0.50),
            "p95": _percentile(latencies, 0.95),
            "max": round(max(latencies), 3) if latencies else None,
        },
        "models": {
            model: model_metric(rows) for model, rows in sorted(model_rows.items())
        },
    }


def _gate(
    metrics: Mapping[str, Any],
    thresholds: AdoptionThresholds,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    coverage = source.get("coverage")
    coverage = coverage if isinstance(coverage, Mapping) else {}
    checks = {
        "full_usable_corpus": {
            "observed": source.get("full_usable_selection") is True,
            "required": True,
            "passed": source.get("full_usable_selection") is True,
        },
        "minimum_usable_cases": {
            "observed": source.get("usable_cases"),
            "minimum": MIN_ADOPTION_USABLE_CASES,
        },
        "role_coverage": {
            "observed": coverage.get("role_coverage_rate"),
            "minimum": 1.0,
        },
        "historical_decision_coverage": {
            "observed": coverage.get("decision_coverage_rate"),
            "minimum": 1.0,
        },
        "production_schema_coverage": {
            "observed": coverage.get("production_schema_coverage_rate"),
            "minimum": 1.0,
        },
        "minimum_cases_per_production_schema": {
            "observed": coverage.get("minimum_production_schema_cases"),
            "minimum": MIN_CASES_PER_PRODUCTION_SCHEMA,
        },
        "first_pass_schema_success": {
            "observed": metrics.get("first_pass_schema_rate"),
            "minimum": thresholds.first_pass_schema_rate,
        },
        "final_schema_success": {
            "observed": metrics.get("final_schema_rate"),
            "minimum": thresholds.final_schema_rate,
        },
        "pair_valid_vote": {
            "observed": metrics.get("pair_valid_rate"),
            "minimum": thresholds.pair_valid_rate,
        },
        "pair_agreement": {
            "observed": metrics.get("pair_agreement_rate"),
            "minimum": thresholds.pair_agreement_rate,
        },
        "three_model_majority_resolution": {
            "observed": metrics.get("majority_resolution_rate"),
            "minimum": thresholds.majority_resolution_rate,
        },
        "historical_signature_match": {
            "observed": metrics.get("historical_signature_match_rate"),
            "minimum": thresholds.historical_signature_match_rate,
        },
        "invalid_output_accepted": {
            "observed": metrics.get("invalid_output_accepted"),
            "maximum": thresholds.max_invalid_output_accepted,
        },
        "unsafe_decision_flips": {
            "observed": metrics.get("unsafe_decision_flips"),
            "maximum": thresholds.max_unsafe_decision_flips,
        },
    }
    for check in checks.values():
        if "passed" in check:
            continue
        observed = check["observed"]
        if "minimum" in check:
            check["passed"] = bool(
                isinstance(observed, (int, float)) and observed >= check["minimum"]
            )
        else:
            check["passed"] = bool(
                isinstance(observed, (int, float)) and observed <= check["maximum"]
            )
    return {
        "passed": all(check["passed"] for check in checks.values()),
        "checks": checks,
    }


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            os.unlink(temporary)
        except OSError:
            pass


def _read_artifact(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ResumeMismatchError(f"cannot read resume artifact: {exc}") from exc
    if not isinstance(payload, dict):
        raise ResumeMismatchError("resume artifact must be a JSON object")
    return payload


def _refresh_artifact(
    artifact: dict[str, Any],
    *,
    status: str,
    thresholds: AdoptionThresholds,
) -> None:
    artifact["status"] = status
    artifact["updated_at"] = _now()
    artifact["processed_cases"] = len(artifact["cases"])
    metrics = _metrics(artifact["cases"])
    artifact["metrics"] = metrics
    artifact["adoption_gate"] = _gate(metrics, thresholds, artifact["source"])
    artifact["adopted"] = bool(
        status == "complete"
        and artifact["processed_cases"] == artifact["selected_cases"]
        and artifact["adoption_gate"]["passed"]
    )


def evaluate_replays(
    input_path: Path | str,
    output_path: Path | str,
    *,
    offset: int = 0,
    limit: int = 0,
    resume: bool = False,
    config: DecisionRouterConfig | None = None,
    transport: ChatTransport | None = None,
    model_metadata_provider: ModelMetadataProvider | None = None,
    thresholds: AdoptionThresholds | None = None,
    required_schema_manifest: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Evaluate a replay slice and atomically checkpoint a redacted artifact."""

    corpus = load_replay_corpus(
        input_path,
        offset=offset,
        limit=limit,
        required_schema_manifest=required_schema_manifest,
    )
    output = Path(output_path).expanduser()
    if corpus.path.resolve() == output.resolve():
        raise ValueError("output_path must not overwrite the read-only replay input")
    config = config or load_decision_router_config()
    thresholds = thresholds or AdoptionThresholds()
    config_payload = asdict(config)
    models = (
        config.primary_model,
        config.challenger_model,
        config.tie_break_model,
    )
    provider = model_metadata_provider or fetch_local_model_metadata
    metadata_payload = provider(models)
    if not isinstance(metadata_payload, Mapping):
        raise ValueError("model metadata provider must return a mapping")
    safe_metadata = _safe_model_metadata(metadata_payload, models)
    # Bind the identity to the exact redacted metadata stored in the artifact,
    # so runtime can recompute it instead of trusting an unattached digest.
    metadata_sha256 = _sha256_json(safe_metadata)
    identity = {
        "source_sha256": corpus.source_sha256,
        "offset": corpus.offset,
        "limit": corpus.limit,
        "selected_case_ids_sha256": _sha256_json(
            [case.case_id for case in corpus.cases]
        ),
        "config_sha256": _sha256_json(config_payload),
        "model_metadata_sha256": metadata_sha256,
        "thresholds_sha256": _sha256_json(asdict(thresholds)),
        "schema_manifest_sha256": corpus.coverage()[
            "schema_manifest_sha256"
        ],
        "signature_manifest_sha256": corpus.coverage()[
            "signature_manifest_sha256"
        ],
    }
    run_key = _sha256_json(identity)

    if resume:
        artifact = _read_artifact(output)
        if artifact.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
            raise ResumeMismatchError("artifact schema version does not match")
        if artifact.get("run_key") != run_key:
            raise ResumeMismatchError("artifact identity does not match this replay run")
        existing_cases = artifact.get("cases")
        if not isinstance(existing_cases, list):
            raise ResumeMismatchError("artifact cases must be an array")
        valid_ids = {case.case_id for case in corpus.cases}
        observed_ids = [
            row.get("case_id") for row in existing_cases if isinstance(row, Mapping)
        ]
        if len(observed_ids) != len(existing_cases) or len(set(observed_ids)) != len(
            observed_ids
        ):
            raise ResumeMismatchError("artifact contains invalid or duplicate case ids")
        if not set(observed_ids).issubset(valid_ids):
            raise ResumeMismatchError("artifact contains cases outside this selection")
        if artifact.get("status") == "complete" and len(existing_cases) == len(
            corpus.cases
        ):
            return artifact
    else:
        if output.exists():
            raise FileExistsError(
                f"output artifact already exists; use --resume or a new path: {output}"
            )
        artifact = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "status": "in_progress",
            "started_at": _now(),
            "updated_at": _now(),
            "run_key": run_key,
            "identity": identity,
            "source": corpus.inspection(include_cases=False),
            "selected_cases": len(corpus.cases),
            "processed_cases": 0,
            "config": config_payload,
            "config_sha256": identity["config_sha256"],
            "model_metadata": safe_metadata,
            "model_metadata_sha256": metadata_sha256,
            "thresholds": asdict(thresholds),
            "cases": [],
            "metrics": _metrics([]),
            "adoption_gate": _gate(
                _metrics([]), thresholds, corpus.inspection(include_cases=False)
            ),
            "adopted": False,
        }
        _atomic_json(output, artifact)

    timing_transport = _TimingTransport(transport or _live_transport)
    router = DecisionRouter(
        config=config,
        transport=timing_transport,
        audit_role="model_eval",
        resolve_adoption=False,
        record_replay=False,
    )
    completed_ids = {
        str(row.get("case_id"))
        for row in artifact["cases"]
        if isinstance(row, Mapping)
    }
    for case in corpus.cases:
        if case.case_id in completed_ids:
            continue
        mark = timing_transport.mark()
        started = time.perf_counter()
        result = router.decide(
            case.prompt,
            case.schema,
            system=case.system,
        )
        record = _case_record(
            case,
            result,
            latency_ms=(time.perf_counter() - started) * 1000,
            transport_events=timing_transport.since(mark),
        )
        artifact["cases"].append(record)
        _refresh_artifact(artifact, status="in_progress", thresholds=thresholds)
        _atomic_json(output, artifact)

    artifact["cases"].sort(key=lambda row: int(row["index"]))
    _refresh_artifact(artifact, status="complete", thresholds=thresholds)
    _atomic_json(output, artifact)
    return artifact


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay-gate the local LLM Wiki decision ensemble without frontier calls."
    )
    parser.add_argument("--input", type=Path, default=REPLAY_FILE)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument(
        "--limit", type=int, default=0, help="0 evaluates through end of corpus"
    )
    parser.add_argument("--resume", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--list", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.dry_run or args.list:
            payload = inspect_replays(
                args.input,
                offset=args.offset,
                limit=args.limit,
                include_cases=args.list,
            )
            payload["mode"] = "list" if args.list else "dry_run"
            print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
            return 0
        if args.output is None:
            parser.error("--output is required unless --dry-run or --list is used")
        config = load_decision_router_config(args.config)
        artifact = evaluate_replays(
            args.input,
            args.output,
            offset=args.offset,
            limit=args.limit,
            resume=args.resume,
            config=config,
        )
    except (ReplayInputError, ResumeMismatchError, FileExistsError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")
    summary = {
        "status": artifact["status"],
        "adopted": artifact["adopted"],
        "output": str(args.output),
        "processed_cases": artifact["processed_cases"],
        "selected_cases": artifact["selected_cases"],
        "source": {
            key: artifact["source"].get(key)
            for key in (
                "total_cases",
                "usable_cases",
                "excluded_cases",
                "excluded_reasons",
                "full_usable_selection",
                "coverage",
            )
        },
        "metrics": artifact["metrics"],
        "adoption_gate": artifact["adoption_gate"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if artifact["adopted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AdoptionThresholds",
    "ReplayCase",
    "ReplayCorpus",
    "ReplayInputError",
    "ResumeMismatchError",
    "evaluate_replays",
    "fetch_local_model_metadata",
    "inspect_replays",
    "load_replay_corpus",
    "main",
    "replay_agreement_value",
]
